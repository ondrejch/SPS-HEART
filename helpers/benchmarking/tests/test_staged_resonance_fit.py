"""Tests for the staged resonance-decay fit driver (card TASK-2026-08-31, P3).

Hermetic by default: the end-to-end smoke uses a synthetic NA49 table in
``tmp_path`` and the committed golden fixture caches, and the stage0
reference path uses the same synthetic table (no external data reads).
The driver's seed protocol is checked against the committed golden
fixture summaries, which carry the pre-registered component seeds.

The stage-0 golden reference cache policy (archive self-containment fix,
outer c89e333 / nested 8345a4d) is pinned on a ``tmp_path`` skeleton of
``dev/mixing2006`` that carries only the committed golden JSON summaries:
with complete caches the reference entry point is a pure refit (no
regeneration, no fixture writes); with absent caches it rebuilds them
through the seeded capture path, validates against the committed
baselines, and installs only in-band regenerations, raising a
RuntimeError naming the drift (out-of-band track count, kinematic-failure
mismatch, component-set mismatch) when validation fails. None of these
tests read the spshadrons mirror or the real
``dev/mixing2006/tests/fixtures/components/`` caches.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, cast

import numpy as np
import pytest

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]
MIXING2006 = REPO_ROOT / "dev" / "mixing2006"
DRIVER = REPO_ROOT / "helpers" / "benchmarking" / "staged_resonance_fit.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import helpers.benchmarking.staged_resonance_fit as sfr  # noqa: E402
from helpers.benchmarking.benchmark_sps_species import (  # noqa: E402
    _component_seed,
)
from helpers.benchmarking.check_helpers_layout_sync import (  # noqa: E402
    main as layout_sync_main,
)


def _fixture_summary(species_tag: str) -> dict:
    path = MIXING2006 / "tests" / "fixtures" / f"golden_stage0_summary_{species_tag}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _synthetic_na49_table(tmp_path: Path, name: str = "synthetic_piminus.txt") -> Path:
    """A small four-column NA49-style table inside the paper window."""
    lines = [
        "# synthetic invariant cross section table (xf pt value error)",
        "# for driver smoke tests only",
    ]
    rng = np.random.default_rng(12345)
    xf_vals = np.linspace(0.02, 0.38, 20)
    pt_vals = np.linspace(0.08, 1.45, 12)
    for xf in xf_vals:
        for pt in pt_vals:
            value = 8.0 + 25.0 * np.exp(-pt) * (1.0 - xf)
            sigma = 0.15 * value
            lines.append(f"{xf:.4f} {pt:.4f} {value:.6f} {sigma:.6f}")
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_seed_protocol_matches_golden_fixtures() -> None:
    """Pre-registered per-component base seeds must match the fixtures."""
    for tag in ("pi_minus", "pi_plus"):
        summary = _fixture_summary(tag)
        base_seed = int(summary["base_seed"])
        stride = int(summary["band_seed_stride"])
        assert base_seed == sfr.DEFAULT_BASE_SEED
        assert stride == sfr.DEFAULT_BAND_SEED_STRIDE
        assert int(summary["band_seeds"]) == sfr.DEFAULT_BAND_SEEDS
        for label, entry in sorted(summary["components"].items()):
            spec_key = str(entry["spec_key"])
            expected = _component_seed(base_seed, spec_key)
            got = int(entry["component_seed"])
            assert got == expected, f"{tag}/{label}: {got} != {expected}"
            # Replicate k=1 seed per the pre-registered additive rule.
            assert expected + stride == _component_seed(base_seed, spec_key) + stride


def test_openness_guard_flags_below_floor() -> None:
    assert sfr.OPENNESS_GUARD_FLOOR == 0.05
    audit = {
        "channel_requested": {
            "9911201|2212,9900113": 1.0,
            "9911201|2212,1114": 0.5,  # non-mesonic (K daughter) channel
            "9900113|2212,-2212": 0.8,
        },
        "openness_fraction": {
            "9911201|2212,9900113": 0.03,
            "9900113|2212,-2212": 0.9,
        },
        "parent_openness_fraction": {"9900113|2212,-2212": 0.04},
        "channel_realized": {},
    }
    guard = sfr._evaluate_openness_guard(audit)
    assert guard["status"] == "evaluated"
    assert set(cast(Any, guard["enabled_channels"])) == {
        "9911201|2212,9900113",
        "9900113|2212,-2212",
    }
    assert guard["tail_only_flags"] == [
        {"channel": "9911201|2212,9900113", "openness_fraction": 0.03}
    ]
    assert guard["violations"] == [
        {"channel": "9900113|2212,-2212", "parent_openness_fraction": 0.04}
    ]
    # Stage A (mesonics off): no mesonic channel requested -> not applicable.
    stage_a_audit = {
        "channel_requested": {"9911201|2212,1114": 1.0},
        "openness_fraction": {"9911201|2212,1114": 0.99},
        "parent_openness_fraction": {},
        "channel_realized": {},
    }
    guard2 = sfr._evaluate_openness_guard(stage_a_audit)
    assert guard2["status"] == "no_enabled_mesonic_channels"
    assert guard2["tail_only_flags"] == []
    assert sfr._evaluate_openness_guard(None)["status"] == "audit_unavailable"


def test_band_check_gate_logic() -> None:
    passing = sfr._band_check_payload(
        base_wrrmse=0.10,
        band_wrrmseq=[0.11, 0.09, 0.10],
        stage0_wrrmse=0.12,
        band_seeds=3,
    )
    assert passing["evaluated"] is True
    assert passing["passes"] is True
    assert passing["bound"] == pytest.approx(0.12 + 2.0 * passing["band_sigma_ddof1"])
    failing = sfr._band_check_payload(
        base_wrrmse=0.30,
        band_wrrmseq=[0.10, 0.11, 0.12],
        stage0_wrrmse=0.12,
        band_seeds=3,
    )
    assert failing["evaluated"] is True
    assert failing["passes"] is False
    deferred = sfr._band_check_payload(
        base_wrrmse=0.10,
        band_wrrmseq=[0.11],
        stage0_wrrmse=0.12,
        band_seeds=1,
    )
    assert deferred["evaluated"] is False
    assert deferred["passes"] is None
    no_ref = sfr._band_check_payload(
        base_wrrmse=0.10,
        band_wrrmseq=[0.11, 0.12],
        stage0_wrrmse=None,
        band_seeds=2,
    )
    assert no_ref["evaluated"] is False
    assert no_ref["passes"] is None


def test_arm_spec_parsing_and_stage_b_merge() -> None:
    name, overrides = sfr._parse_arm_spec("cand:foo=0.25,bar=1.5", "default")
    assert name == "cand"
    assert overrides == {"foo": 0.25, "bar": 1.5}
    name2, overrides2 = sfr._parse_arm_spec("plain", "default")
    assert name2 == "plain"
    assert overrides2 == {}
    with pytest.raises(ValueError):
        sfr._parse_arm_spec("bad:no_equals_here", "default")

    boxes, pair_key = sfr._import_stage_b_api(MIXING2006)
    assert pair_key == "mestar_eff_ppbar_strength"
    assert set(boxes) == {1, 2, 3}
    merged, level, box = sfr._apply_stage_b_candidate(
        {"nstar_eff_l3_br_nstarpi": 0.4}, "B-850", boxes
    )
    assert level == 2
    window = cast(Any, box["window_overrides"])
    assert merged["nstar_eff_tower_max_mass"] == window["nstar_eff_tower_max_mass"]
    assert "mestar_eff_tower_max_mass" in merged
    ablated = sfr._apply_b0_ablation(merged)
    assert ablated["mestar_eff_ppbar_strength"] == 0.0
    with pytest.raises(ValueError):
        sfr._apply_stage_b_candidate({}, "B-999", boxes)


def test_artifact_writer_and_sha256_sidecar(tmp_path: Path) -> None:
    payload = {"n": 3, "values": [1.0, np.float32(2.5), np.int64(7)], "inf": float("inf")}
    path = tmp_path / "artifact.json"
    sfr._write_json(path, payload)
    digest = sfr._write_sha256_sidecar(path)
    text = path.read_text(encoding="utf-8")
    loaded = json.loads(text)  # strict parse: no Infinity tokens allowed
    assert loaded["n"] == 3
    assert loaded["values"] == [1.0, 2.5, 7]
    assert loaded["inf"] is None
    sidecar = path.with_suffix(".json.sha256").read_text(encoding="utf-8")
    assert sidecar == f"{digest}  {path.name}\n"
    import hashlib

    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_builder_returns_and_persists_audit_tables(tmp_path: Path) -> None:
    """REV-CUM-01: the builder exports the per-channel audit tables and
    persists them in the cache archive (meta_channel_audit_json)."""
    from helpers.benchmarking.benchmark_sps_species import (
        _build_resonance_component_cache_one,
        ResonanceSpec,
    )

    gen_file = (
        MIXING2006
        / "na49_mixing"
        / "generators"
        / "michal"
        / "deltas-to-pion"
        / "delta++.gen"
    )
    spec = ResonanceSpec(label="delta++", gen_file=gen_file, use_soft_decay_factory=True)
    cache_path = tmp_path / "delta++.npz"
    task = {
        "spec_key": "test|delta++::x",
        "label": "delta++",
        "gen_file": str(gen_file),
        "decay_files": [],
        "use_kaon_decay_factory": False,
        "use_soft_decay_factory": True,
        "cache_file": str(cache_path),
        "component_seed": 1234567,
        "fragmentation_mode": "both",
        "soft_decay_overrides": {},
        "weak_feed_policy": "inclusive_weak",
        "nucleus": None,
        "target_wound_range": None,
        "beam_pdg": 2212,
        "target_pdg": 2212,
        "max_undecayed_fraction": 0.01,
        "decay_lib_fingerprint": "",
    }
    result = _build_resonance_component_cache_one(
        mixing2006_dir=str(MIXING2006),
        task=task,
        n_events_per_res=20,
        max_decay_passes=8,
        fragmentation_mode="both",
        chunk_workers=1,
    )
    assert result["ok"] is True
    assert "channel_audit_json" in result
    audit = json.loads(str(result["channel_audit_json"]))
    assert set(audit) == {
        "channel_requested",
        "channel_realized",
        "openness_fraction",
        "parent_openness_fraction",
    }
    with np.load(cache_path, allow_pickle=False) as z:
        assert "meta_channel_audit_json" in z
        persisted = json.loads(str(z["meta_channel_audit_json"]))
    assert persisted == audit


def test_stageA_smoke_end_to_end(tmp_path: Path) -> None:
    """Hermetic end-to-end: synthetic table + small generation + finalize."""
    table = _synthetic_na49_table(tmp_path)
    out_dir = tmp_path / "run"
    rc = sfr.main(
        [
            "--mode", "stageA",
            "--species", "pi-",
            "--disable-tower",
            "--events-per-component", "30",
            "--replicates", "0",
            "--band-seeds", "0",
            "--res-workers", "1",
            "--na49-data", str(table),
            "--out-dir", str(out_dir),
            "--finalize",
        ]
    )
    assert rc == sfr.EXIT_OK
    artifact_path = out_dir / sfr.STAGE_A_ARTIFACT_FILENAME
    assert artifact_path.exists()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["schema_version"] == 1
    assert artifact["resonance_decay_lib_version"] == "2026-09-03.3"
    sp_payload = artifact["species"]["pi-"]
    base = sp_payload["base_replicate"]
    assert base["n_points"] > 0
    assert len(base["design_columns"]) > 0
    assert all(np.isfinite(list(base["weights"].values())))
    assert base["fit_info"]["n_components"] == len(base["design_columns"])
    assert base["openness"]["guard_floor"] == sfr.OPENNESS_GUARD_FLOOR
    # band_seeds=0 -> band check deferred, not failed
    assert artifact["band_check"]["per_species"]["pi-"]["passes"] is None
    # stage0 reference computed from the committed fixtures + synthetic table
    s0 = artifact["stage0_reference"]["pi-"]
    assert s0 is not None
    assert s0["fixture_provenance"]["salt_in_force"] == "2026-08-26.1"
    assert s0["n_points"] == base["n_points"]
    # seed protocol recorded
    assert artifact["seed_protocol"]["base_seed"] == sfr.DEFAULT_BASE_SEED
    # sha256 sidecar present and consistent
    sidecar = (out_dir / (sfr.STAGE_A_ARTIFACT_FILENAME + ".sha256")).read_text()
    digest = sidecar.split()[0]
    assert digest == sfr._sha256_file(artifact_path)


# --- Stage-0 golden reference cache policy (archive self-containment) ----
#
# These tests run on a ``tmp_path`` skeleton of ``dev/mixing2006`` that
# carries the committed golden JSON summaries and no component caches
# (the source-only archive scenario).  The skeleton reuses the real
# package-relative layout, so the seeded capture path
# (``_resonance_specs_for_particle`` + ``_resonance_spec_key`` +
# ``_component_seed``) reproduces the pre-registered per-component seeds
# without touching the real fixture tree.


def _stage0_fixture_skeleton(tmp_path: Path, tag: str, summary: Optional[dict] = None) -> Path:
    """A source-only checkout skeleton of ``dev/mixing2006`` for stage-0.

    Carries the committed golden summary (or an explicit tampered copy)
    and no component caches. Returns the skeleton root that
    ``_run_stage0_reference`` takes as ``mixing2006_dir``.
    """
    root = tmp_path / "mixing2006"
    fixture_dir = root / "tests" / "fixtures"
    fixture_dir.mkdir(parents=True)
    name = f"golden_stage0_summary_{tag}.json"
    committed = MIXING2006 / "tests" / "fixtures" / name
    if summary is None:
        (fixture_dir / name).write_text(committed.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        (fixture_dir / name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return root


def _stub_component_cache(path: Path, n_target: int, target_pdg: int, kin_failures: int) -> None:
    """A minimal component cache: ``n_target`` species tracks plus proton
    tracks, and the meta scalar the regeneration validation reads.

    Each cache file gets its own kinematic cloud (stable per-label seed,
    unsorted (xF, pT) pairs) so the stage-0 design columns stay linearly
    independent. Sorting both coordinates independently made every
    component the same monotonic curve and exhausted SciPy nnls.
    """
    label = path.stem.encode("utf-8")
    seed = 20260911
    for byte in label:
        seed = (seed * 131 + int(byte)) % 2_147_483_647
    rng = np.random.default_rng(seed)
    n_other = 40
    pdg = np.concatenate(
        [
            np.full(n_target, target_pdg, dtype=np.int32),
            np.full(n_other, 2212, dtype=np.int32),
        ]
    )
    n = int(pdg.size)
    xf = rng.uniform(0.02, 0.35, size=n)
    pt = rng.uniform(0.10, 1.20, size=n)
    np.savez(
        path,
        pdg=pdg,
        xf=xf,
        pt=pt,
        meta_n_decay_kinematic_failures=np.int64(kin_failures),
    )


def _run_stage0_reference_entry(
    tmp_path: Path, root: Path, species_key: str, tag: str
) -> Optional[dict]:
    """The stage-0 reference entry point on a synthetic table (hermetic)."""
    table = _synthetic_na49_table(tmp_path, f"synthetic_{tag}.txt")
    return sfr._run_stage0_reference(
        mixing2006_dir=root,
        species_key=species_key,
        table_path=table,
        base_seed=sfr.DEFAULT_BASE_SEED,
        train_frac=sfr.DEFAULT_TRAIN_FRAC,
        beam_momentum_gev_c=sfr.DEFAULT_BEAM_MOMENTUM_GEV_C,
        xf_min=sfr.PAPER_XF_MIN,
        xf_max=sfr.PAPER_XF_MAX,
        pt_min=sfr.PAPER_PT_MIN,
        pt_max=sfr.PAPER_PT_MAX,
    )


def test_stage0_reference_fast_path_does_not_regenerate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Complete caches: the entry point is a pure refit. No regeneration,
    no fixture writes, and the committed summary stays byte-identical."""
    tag, species_key = "pi_minus", "pi-"
    summary = _fixture_summary(tag)
    root = _stage0_fixture_skeleton(tmp_path, tag)
    comp_dir = root / "tests" / "fixtures" / "components" / tag
    comp_dir.mkdir(parents=True)
    target_pdg = int(cast(Any, sfr.resolve_species(species_key)).pdg)
    for label in summary["component_labels"]:
        entry = summary["components"][label]
        _stub_component_cache(
            comp_dir / f"{label}.npz",
            int(entry["n_target_species"]),
            target_pdg,
            int(entry["n_decay_kinematic_failures"]),
        )
    before = {p.name: p.read_bytes() for p in comp_dir.iterdir()}

    def _boom(**kwargs: Any) -> None:
        raise AssertionError("fast path must not regenerate the fixture caches")

    monkeypatch.setattr(sfr, "_regenerate_golden_component_caches", _boom)

    result = _run_stage0_reference_entry(tmp_path, root, species_key, tag)
    assert result is not None
    assert result["fixture_provenance"]["salt_in_force"] == "2026-08-26.1"
    # Components without target species tracks carry no design column
    # (the design-matrix builder drops them), so the column set is the
    # labels with a positive committed baseline count.
    expected_columns = [
        label
        for label in summary["component_labels"]
        if int(summary["components"][label]["n_target_species"]) > 0
    ]
    assert list(result["design_columns"]) == expected_columns
    for label in summary["component_labels"]:
        entry = summary["components"][label]
        assert result["components"][label]["n_target_species"] == int(
            entry["n_target_species"]
        )
        assert result["components"][label]["component_seed"] == int(
            entry["component_seed"]
        )
    # No files were written or rewritten by the pure refit.
    after = {p.name: p.read_bytes() for p in comp_dir.iterdir()}
    assert after == before
    golden_name = f"golden_stage0_summary_{tag}.json"
    assert (root / "tests" / "fixtures" / golden_name).read_bytes() == (
        MIXING2006 / "tests" / "fixtures" / golden_name
    ).read_bytes()


def test_stage0_reference_regenerates_absent_caches_in_band(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absent caches: the entry point rebuilds them through the seeded
    capture path, and validation accepts an in-band regeneration."""
    tag, species_key = "pi_plus", "pi+"
    summary = _fixture_summary(tag)
    root = _stage0_fixture_skeleton(tmp_path, tag)
    comp_dir = root / "tests" / "fixtures" / "components" / tag
    assert not comp_dir.exists()
    target_pdg = int(cast(Any, sfr.resolve_species(species_key)).pdg)
    # In-band, non-exact counts: half the pre-registered band offset
    # (exact match where the band is degenerate at zero).
    stub_counts = {
        label: int(summary["components"][label]["n_target_species"])
        + int(
            float(summary["components"][label]["mc_band"]["n_sigma_multiplier"])
            * float(summary["components"][label]["mc_band"]["n_target_sigma_used"])
            // 2
        )
        for label in summary["component_labels"]
    }
    calls: list[dict] = []

    def _stub_build(
        *,
        mixing2006_dir: str,
        task: dict,
        n_events_per_res: int,
        max_decay_passes: int,
        fragmentation_mode: str,
        chunk_workers: int,
    ) -> dict:
        label = str(task["label"])
        entry = summary["components"][label]
        calls.append(
            {
                "label": label,
                "component_seed": int(task["component_seed"]),
                "n_events_per_res": int(n_events_per_res),
            }
        )
        _stub_component_cache(
            Path(task["cache_file"]),
            stub_counts[label],
            target_pdg,
            int(entry["n_decay_kinematic_failures"]),
        )
        return {"ok": True}

    monkeypatch.setattr(sfr, "_build_resonance_component_cache_one", _stub_build)

    result = _run_stage0_reference_entry(tmp_path, root, species_key, tag)
    # One seeded build per label, in the committed label order, at the
    # pre-registered component seeds and statistics.
    assert [c["label"] for c in calls] == list(summary["component_labels"])
    for call in calls:
        entry = summary["components"][call["label"]]
        assert call["component_seed"] == int(entry["component_seed"])
        assert call["n_events_per_res"] == int(summary["events_per_component"])
    # Accepted as in-band and installed in place, one cache per label.
    for label, n in stub_counts.items():
        cache = comp_dir / f"{label}.npz"
        assert cache.is_file()
        with np.load(cache, allow_pickle=False) as z:
            assert int(
                np.count_nonzero(np.asarray(z["pdg"], dtype=np.int32) == target_pdg)
            ) == n
    assert result is not None
    assert list(result["design_columns"]) == [
        label for label in summary["component_labels"] if stub_counts[label] > 0
    ]
    for label in summary["component_labels"]:
        assert result["components"][label]["n_target_species"] == stub_counts[label]


def test_stage0_reference_out_of_band_regeneration_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A regeneration outside the pre-registered band raises a
    RuntimeError naming the drift, and no cache is installed."""
    tag, species_key = "pi_plus", "pi+"
    summary = _fixture_summary(tag)
    root = _stage0_fixture_skeleton(tmp_path, tag)
    comp_dir = root / "tests" / "fixtures" / "components" / tag
    target_pdg = int(cast(Any, sfr.resolve_species(species_key)).pdg)
    drift = 1000

    def _stub_build(
        *,
        mixing2006_dir: str,
        task: dict,
        n_events_per_res: int,
        max_decay_passes: int,
        fragmentation_mode: str,
        chunk_workers: int,
    ) -> dict:
        label = str(task["label"])
        entry = summary["components"][label]
        _stub_component_cache(
            Path(task["cache_file"]),
            int(entry["n_target_species"]) + drift,
            target_pdg,
            int(entry["n_decay_kinematic_failures"]),
        )
        return {"ok": True}

    monkeypatch.setattr(sfr, "_build_resonance_component_cache_one", _stub_build)

    first = summary["component_labels"][0]
    first_entry = summary["components"][first]
    with pytest.raises(RuntimeError) as excinfo:
        _run_stage0_reference_entry(tmp_path, root, species_key, tag)
    message = str(excinfo.value)
    assert f"{tag}/{first}" in message
    assert "n_target_species baseline" in message
    assert str(int(first_entry["n_target_species"])) in message
    assert "pre-registered band" in message
    assert "salt in force" in message
    # The rejected cache is not installed: the components dir was never
    # created (the first component fails validation before any install).
    assert not comp_dir.exists()


def test_stage0_reference_kinematic_failure_mismatch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An in-band track count with a mismatched kinematic-failure count
    still fails validation loudly, and nothing is installed."""
    tag, species_key = "pi_plus", "pi+"
    summary = _fixture_summary(tag)
    root = _stage0_fixture_skeleton(tmp_path, tag)
    comp_dir = root / "tests" / "fixtures" / "components" / tag
    target_pdg = int(cast(Any, sfr.resolve_species(species_key)).pdg)

    def _stub_build(
        *,
        mixing2006_dir: str,
        task: dict,
        n_events_per_res: int,
        max_decay_passes: int,
        fragmentation_mode: str,
        chunk_workers: int,
    ) -> dict:
        label = str(task["label"])
        entry = summary["components"][label]
        _stub_component_cache(
            Path(task["cache_file"]),
            int(entry["n_target_species"]),
            target_pdg,
            int(entry["n_decay_kinematic_failures"]) + 1,
        )
        return {"ok": True}

    monkeypatch.setattr(sfr, "_build_resonance_component_cache_one", _stub_build)

    first = summary["component_labels"][0]
    baseline_kin = int(summary["components"][first]["n_decay_kinematic_failures"])
    with pytest.raises(RuntimeError) as excinfo:
        _run_stage0_reference_entry(tmp_path, root, species_key, tag)
    message = str(excinfo.value)
    assert f"{tag}/{first}" in message
    assert (
        f"n_decay_kinematic_failures baseline {baseline_kin}, "
        f"regenerated {baseline_kin + 1}"
    ) in message
    assert not comp_dir.exists()


def test_stage0_reference_label_mismatch_is_rejected_before_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A summary recording a different component set than the seeded
    capture path would build is refused before any component build."""
    tag, species_key = "pi_plus", "pi+"
    summary = _fixture_summary(tag)
    tampered = copy.deepcopy(summary)
    dropped = str(tampered["component_labels"][-1])
    del tampered["component_labels"][-1]
    root = _stage0_fixture_skeleton(tmp_path, tag, summary=tampered)
    builds: list[str] = []

    def _stub_build(
        *,
        mixing2006_dir: str,
        task: dict,
        n_events_per_res: int,
        max_decay_passes: int,
        fragmentation_mode: str,
        chunk_workers: int,
    ) -> dict:
        builds.append(str(task["label"]))
        return {"ok": True}

    monkeypatch.setattr(sfr, "_build_resonance_component_cache_one", _stub_build)

    with pytest.raises(RuntimeError) as excinfo:
        _run_stage0_reference_entry(tmp_path, root, species_key, tag)
    message = str(excinfo.value)
    assert "different component set" in message
    assert dropped in message
    assert builds == []  # the drift is rejected before any component build
    assert not (root / "tests" / "fixtures" / "components").exists()


def test_golden_component_caches_complete_predicate(tmp_path: Path) -> None:
    """The fast-path predicate: every committed label needs its npz."""
    summary = _fixture_summary("pi_minus")
    labels = [str(label) for label in summary["component_labels"]]
    comp_dir = tmp_path / "components"
    comp_dir.mkdir()
    # An absent directory is incomplete for a non-empty label set, but a
    # summary with no labels is trivially complete.
    assert not sfr._golden_component_caches_complete(tmp_path / "missing", summary)
    assert sfr._golden_component_caches_complete(
        tmp_path / "missing", {"component_labels": []}
    )
    for label in labels:
        (comp_dir / f"{label}.npz").write_bytes(b"stub")
    assert sfr._golden_component_caches_complete(comp_dir, summary)
    (comp_dir / f"{labels[0]}.npz").unlink()
    assert not sfr._golden_component_caches_complete(comp_dir, summary)


def _stage_b_boxes_normalized() -> dict:
    boxes, _pair_key = sfr._import_stage_b_api(MIXING2006)
    return {
        int(k): {str(kk): (list(vv) if isinstance(vv, (list, tuple)) else vv) for kk, vv in v.items()}
        for k, v in boxes.items()
    }


def test_stage_b_grid_tables_pre_registered() -> None:
    """Grid tables are pre-registered: counts, ranges, coarse subset rule."""
    boxes = _stage_b_boxes_normalized()
    expected_counts = {
        "B-450": 45,
        "B-850": 288,
        "B-1250": 288,
        "B-850W": 480,
        "B-1250W": 480,
        "B-850W2": 576,
        "B-1250W2": 576,
        "B-850G": 16,
        "B-850N": 12,
        "B-850R": 18,
        "B-850C": 12,
        "B-850S": 9,
        "B-850WR": 4,
        "B-850P": 5,
        "B-850L": 6,
    }
    expected_limits = {
        "B-450": 60,
        "B-850": 300,
        "B-1250": 300,
        "B-850W": 600,
        "B-1250W": 600,
        "B-850W2": 600,
        "B-1250W2": 600,
        "B-850G": 60,
        "B-850N": 60,
        "B-850R": 60,
        "B-850C": 60,
        "B-850S": 60,
        "B-850WR": 60,
        "B-850P": 60,
        "B-850L": 60,
    }
    assert dict(sfr.STAGE_B_GRID_LIMITS) == expected_limits
    for candidate in ("B-450", "B-850", "B-1250"):
        table = sfr.STAGE_B_GRID_TABLES[candidate]
        level = int(table["level"])
        box = cast(Any, boxes[level])
        assert box["label"] == candidate
        # Point counts within the pre-registered limits.
        full = sfr._stage_b_grid_points(candidate, "full")
        assert len(full) == expected_counts[candidate]
        assert len(full) <= sfr.STAGE_B_GRID_LIMITS[candidate]
        # Deterministic, unique point ids.
        ids = [p["point_id"] for p in full]
        assert len(set(ids)) == len(ids)
        # Coarse grid is a per-axis subset of the full grid.
        coarse = sfr._stage_b_grid_points(candidate, "coarse")
        free_full = {tuple(sorted(p["free"].items())) for p in full}
        free_coarse = {tuple(sorted(p["free"].items())) for p in coarse}
        assert free_coarse <= free_full
        assert len(coarse) < len(full)
        # Coarse rule: first / middle (len//2) / last per axis.
        axes = sfr._stage_b_grid_axes(table)
        for axis, values in axes.items():
            expected_subset = set(sfr._stage_b_coarse_subset(values))
            got = {p["free"][axis] for p in coarse}
            assert got == expected_subset, (candidate, axis)
        # Mass axis concentrates at the frozen box lower edge.
        mass_axis = axes[f"nstar_eff_l{level}_mass"]
        nstar_range = [float(v) for v in cast(Any, box["nstar_mass_range"])]
        assert min(mass_axis) == pytest.approx(nstar_range[0], abs=1e-9)
        assert max(mass_axis) == pytest.approx(nstar_range[1], abs=1e-9)
        for value in mass_axis:
            assert nstar_range[0] - 1e-9 <= value <= nstar_range[1] + 1e-9
        # M* axes exist exactly on the pair-capable candidates, inside the box.
        if bool(box["pair_capable"]):
            assert f"mestar_eff_l{level}_mass" in axes
            mestar_range = [float(v) for v in cast(Any, box["mestar_mass_range"])]
            assert min(axes[f"mestar_eff_l{level}_mass"]) == pytest.approx(
                mestar_range[0], abs=1e-9
            )
            assert max(axes[f"mestar_eff_l{level}_mass"]) == pytest.approx(
                mestar_range[1], abs=1e-9
            )
        else:
            assert f"mestar_eff_l{level}_mass" not in axes
            assert f"mestar_eff_l{level}_width" not in axes
    with pytest.raises(ValueError):
        sfr._stage_b_grid_points("B-450", "sideways")
    with pytest.raises(ValueError):
        sfr._stage_b_grid_points("B-999", "full")


def test_stage_b_point_overrides_and_validation() -> None:
    """Per-point override sets: pins, D4 split, pair strength, box cross-check."""
    boxes = _stage_b_boxes_normalized()
    from na49_mixing import parallel as _parallel

    # The pre-registered tables pass the frozen-box cross-check.
    for candidate, level in (("B-450", 1), ("B-850", 2), ("B-1250", 3)):
        sfr._validate_stage_b_grid_table(candidate, level, boxes, MIXING2006)
    for candidate in ("B-450", "B-850", "B-1250"):
        table = sfr.STAGE_B_GRID_TABLES[candidate]
        level = int(table["level"])
        base_k = float(_parallel.STAGE_B_BASE_NSTARPI[level])
        for point in sfr._stage_b_grid_points(candidate, "coarse"):
            ovr = sfr._stage_b_point_overrides(candidate, point, MIXING2006)
            # Box windows.
            assert ovr["nstar_eff_tower_max_mass"] == 5.0
            assert ovr["mestar_eff_tower_max_mass"] == 2.5
            free = dict(point["free"])
            # Candidate level free coordinates.
            assert ovr[f"nstar_eff_l{level}_mass"] == free[f"nstar_eff_l{level}_mass"]
            assert ovr[f"nstar_eff_l{level}_width"] == free[f"nstar_eff_l{level}_width"]
            # D4 split: (s, base - s), non-negative on the pre-registered axes.
            br_n = ovr[f"nstar_eff_l{level}_br_nstarmeson"]
            br_p = ovr[f"nstar_eff_l{level}_br_nstarpi"]
            assert br_n == pytest.approx(free["strength"])
            assert br_p == pytest.approx(base_k - free["strength"])
            assert br_n >= 0.0 and br_p >= 0.0
            # Non-candidate levels pinned to the pre-registered defaults.
            for j in (1, 2, 3):
                if j != level:
                    assert ovr[f"nstar_eff_l{j}_mass"] == sfr.STAGE_B_NSTAR_MASS_DEFAULTS[j]
                    assert ovr[f"nstar_eff_l{j}_width"] == sfr.STAGE_B_NSTAR_WIDTH_DEFAULTS[j]
                    assert ovr[f"mestar_eff_l{j}_mass"] == sfr.STAGE_B_MSTAR_MASS_DEFAULTS[j]
                    assert ovr[f"mestar_eff_l{j}_width"] == sfr.STAGE_B_MSTAR_WIDTH_DEFAULTS[j]
            # Pair family: present exactly on the pair-capable candidates,
            # at the pre-registered constant.
            pair_capable = level != 1
            if pair_capable:
                assert ovr["mestar_eff_ppbar_strength"] == sfr.STAGE_B_PAIR_STRENGTH
                assert ovr[f"mestar_eff_l{level}_mass"] == free[f"mestar_eff_l{level}_mass"]
                assert ovr[f"mestar_eff_l{level}_width"] == free[f"mestar_eff_l{level}_width"]
            else:
                # Non-pair candidate: no pair strength, and the candidate-level
                # M* coordinates stay pinned to the pre-registered defaults.
                assert "mestar_eff_ppbar_strength" not in ovr
                assert ovr[f"mestar_eff_l{level}_mass"] == sfr.STAGE_B_MSTAR_MASS_DEFAULTS[level]
                assert ovr[f"mestar_eff_l{level}_width"] == sfr.STAGE_B_MSTAR_WIDTH_DEFAULTS[level]
    # Validation rejects a table outside the frozen boxes.
    broken = {
        "level": 1,
        "nstar_mass": (1.50, 2.20, 2.40, 3.20, 5.00),  # below the box lower edge
        "nstar_width": (0.125, 0.25, 0.50),
        "strength": (0.0, 0.075, 0.15),
    }
    saved = copy.deepcopy(sfr.STAGE_B_GRID_TABLES["B-450"])
    try:
        sfr.STAGE_B_GRID_TABLES["B-450"] = broken
        with pytest.raises(ValueError):
            sfr._validate_stage_b_grid_table("B-450", 1, boxes, MIXING2006)
    finally:
        sfr.STAGE_B_GRID_TABLES["B-450"] = saved


def test_stage_b_wide_strength_round2() -> None:
    """Round-2 W tables: extended strength budget, pi branch clamped at zero.

    Pre-registered in notes/resonance_fit_stage_progress.md: the W grids
    extend the strength axis past the frozen D4 budget with
    br_nstarpi = max(base - s, 0), so every round-1 strength point
    reproduces exactly and the cascade branch grows past the exhausted
    pi branch.
    """
    boxes = _stage_b_boxes_normalized()
    from na49_mixing import parallel as _parallel

    # The W tables pass the frozen-box cross-check and the point ceilings.
    for candidate, level in (("B-850W", 2), ("B-1250W", 3)):
        sfr._validate_stage_b_grid_table(candidate, level, boxes, MIXING2006)
        table = sfr.STAGE_B_GRID_TABLES[candidate]
        n_points = 1
        for key in (
            "nstar_mass",
            "nstar_width",
            "mestar_mass",
            "mestar_width",
            "strength",
        ):
            if key in table:
                n_points *= len(table[key])
        assert n_points <= sfr.STAGE_B_GRID_LIMITS[candidate]

    def _point(candidate: str, strength: float) -> Dict[str, float]:
        table = sfr.STAGE_B_GRID_TABLES[candidate]
        level = int(table["level"])
        free: Dict[str, float] = {
            f"nstar_eff_l{level}_mass": float(table["nstar_mass"][-1]),
            f"nstar_eff_l{level}_width": float(table["nstar_width"][0]),
            "strength": float(strength),
        }
        if "mestar_mass" in table:
            free[f"mestar_eff_l{level}_mass"] = float(table["mestar_mass"][0])
            free[f"mestar_eff_l{level}_width"] = float(table["mestar_width"][0])
        return sfr._stage_b_point_overrides(candidate, {"free": free}, MIXING2006)

    # B-850W: round-1 points reproduce (s <= base 0.30); above it the pi
    # branch stays clamped at zero while the cascade branch grows.
    assert float(_parallel.STAGE_B_BASE_NSTARPI[2]) == pytest.approx(0.30)
    ovr = _point("B-850W", 0.30)
    assert ovr["nstar_eff_l2_br_nstarmeson"] == pytest.approx(0.30)
    assert ovr["nstar_eff_l2_br_nstarpi"] == pytest.approx(0.0)
    ovr = _point("B-850W", 0.45)
    assert ovr["nstar_eff_l2_br_nstarmeson"] == pytest.approx(0.45)
    assert ovr["nstar_eff_l2_br_nstarpi"] == pytest.approx(0.0)
    ovr = _point("B-850W", 0.60)
    assert ovr["nstar_eff_l2_br_nstarmeson"] == pytest.approx(0.60)
    assert ovr["nstar_eff_l2_br_nstarpi"] == pytest.approx(0.0)

    # B-1250W: the round-1 optimum (s = 0.45, pi exhausted) sits inside
    # the W axis; the extension reaches 0.90 with pi clamped at zero.
    assert float(_parallel.STAGE_B_BASE_NSTARPI[3]) == pytest.approx(0.45)
    ovr = _point("B-1250W", 0.45)
    assert ovr["nstar_eff_l3_br_nstarmeson"] == pytest.approx(0.45)
    assert ovr["nstar_eff_l3_br_nstarpi"] == pytest.approx(0.0)
    ovr = _point("B-1250W", 0.90)
    assert ovr["nstar_eff_l3_br_nstarmeson"] == pytest.approx(0.90)
    assert ovr["nstar_eff_l3_br_nstarpi"] == pytest.approx(0.0)

    # Pair family unchanged on the W candidates.
    assert ovr["mestar_eff_ppbar_strength"] == sfr.STAGE_B_PAIR_STRENGTH


def test_stage_b_w2_round3_bound_scan() -> None:
    """Round-3 W2 tables: the strength scan reaches the physical bound.

    Pre-registered in notes/resonance_fit_stage_progress.md: budget 1.0
    with every other axis frozen, the whole axis in the
    pi-branch-exhausted regime, both round-2 selection points inside the
    axes. At s = 1.0 the level is single-channel by construction
    (br_nstarmeson = 1, br_nstarpi = 0).
    """
    boxes = _stage_b_boxes_normalized()

    for candidate, level in (("B-850W2", 2), ("B-1250W2", 3)):
        sfr._validate_stage_b_grid_table(candidate, level, boxes, MIXING2006)
        table = sfr.STAGE_B_GRID_TABLES[candidate]
        assert float(table["strength_budget"]) == pytest.approx(1.00)
        n_points = 1
        for key in (
            "nstar_mass",
            "nstar_width",
            "mestar_mass",
            "mestar_width",
            "strength",
        ):
            if key in table:
                n_points *= len(table[key])
        assert n_points == 576
        assert n_points <= sfr.STAGE_B_GRID_LIMITS[candidate]
        # The round-2 selection strength sits inside the round-3 axis.
        assert float(table["strength"][0]) <= {"B-850W2": 0.60, "B-1250W2": 0.90}[candidate]
        assert table["strength"][-1] == pytest.approx(1.00)

    def _point(candidate: str, strength: float) -> Dict[str, float]:
        table = sfr.STAGE_B_GRID_TABLES[candidate]
        level = int(table["level"])
        free: Dict[str, float] = {
            f"nstar_eff_l{level}_mass": float(table["nstar_mass"][0]),
            f"nstar_eff_l{level}_width": float(table["nstar_width"][0]),
            "strength": float(strength),
        }
        if "mestar_mass" in table:
            free[f"mestar_eff_l{level}_mass"] = float(table["mestar_mass"][2])
            free[f"mestar_eff_l{level}_width"] = float(table["mestar_width"][0])
        return sfr._stage_b_point_overrides(candidate, {"free": free}, MIXING2006)

    # At the physical bound the level is a pure cascade state.
    for candidate, level in (("B-850W2", 2), ("B-1250W2", 3)):
        ovr = _point(candidate, 1.00)
        assert ovr[f"nstar_eff_l{level}_br_nstarmeson"] == pytest.approx(1.00)
        assert ovr[f"nstar_eff_l{level}_br_nstarpi"] == pytest.approx(0.0)
    # Round-2 selection points reproduce exactly (pi branch exhausted).
    ovr = _point("B-850W2", 0.60)
    assert ovr["nstar_eff_l2_br_nstarmeson"] == pytest.approx(0.60)
    assert ovr["nstar_eff_l2_br_nstarpi"] == pytest.approx(0.0)
    ovr = _point("B-1250W2", 0.90)
    assert ovr["nstar_eff_l3_br_nstarmeson"] == pytest.approx(0.90)
    assert ovr["nstar_eff_l3_br_nstarpi"] == pytest.approx(0.0)
    # Pair family unchanged.
    assert ovr["mestar_eff_ppbar_strength"] == sfr.STAGE_B_PAIR_STRENGTH


def test_stage_b_round456_tables_overrides_and_validation() -> None:
    """Rounds 4-6 tables: validation, overrides, and the new pair keys.

    Round-4 B-850G/B-850N/B-850R, round-5 B-850C, and round-6 B-850S/
    B-850WR carry the pair_exponent, pair_law_flag, pair_m0, and
    pair_rolloff axes through the per-point override sets, and B-850R
    selects the wide-M* box variant with its own mestar window.
    """
    boxes = _stage_b_boxes_normalized()
    from na49_mixing import parallel as _parallel

    # All six tables pass the frozen-box cross-check (B-850R with the
    # wide-M* variant, which the validation receives via the driver's
    # box selection; validate it against the wide boxes explicitly).
    for candidate, level in (
        ("B-850G", 2),
        ("B-850N", 2),
        ("B-850C", 2),
        ("B-850S", 2),
        ("B-850WR", 2),
    ):
        sfr._validate_stage_b_grid_table(candidate, level, boxes, MIXING2006)
    wide_boxes_raw = _parallel.stage_b_candidate_boxes_wide_mestar()
    wide_boxes = {
        int(k): {str(kk): vv for kk, vv in dict(v).items()}
        for k, v in wide_boxes_raw.items()
    }
    sfr._validate_stage_b_grid_table("B-850R", 2, wide_boxes, MIXING2006)

    def _point(candidate: str, free_overrides: Dict[str, float]) -> Dict[str, float]:
        table = sfr.STAGE_B_GRID_TABLES[candidate]
        level = int(table["level"])
        free: Dict[str, float] = {
            f"nstar_eff_l{level}_mass": float(table["nstar_mass"][0]),
            f"nstar_eff_l{level}_width": float(table["nstar_width"][0]),
            "strength": float(table["strength"][0]),
        }
        if "mestar_mass" in table:
            free[f"mestar_eff_l{level}_mass"] = float(table["mestar_mass"][0])
            free[f"mestar_eff_l{level}_width"] = float(table["mestar_width"][0])
        free.update(free_overrides)
        return sfr._stage_b_point_overrides(candidate, {"free": free}, MIXING2006)

    # E2 pair exponent reaches the override set.
    ovr = _point("B-850N", {"pair_exponent": 0.5})
    assert ovr["mestar_eff_ppbar_exponent"] == pytest.approx(0.5)
    # Round-5 combination carries the exponent axis too.
    ovr = _point("B-850C", {"pair_exponent": 0.25})
    assert ovr["mestar_eff_ppbar_exponent"] == pytest.approx(0.25)
    # Round-6 step-roll-off law selection and its shape parameters.
    ovr = _point(
        "B-850S",
        {"pair_exponent": 0.25, "pair_law_flag": 1.0, "pair_m0": 3.0, "pair_rolloff": 0.4},
    )
    assert ovr["mestar_eff_ppbar_law_flag"] == pytest.approx(1.0)
    assert ovr["mestar_eff_ppbar_rollon_m0"] == pytest.approx(3.0)
    assert ovr["mestar_eff_ppbar_rolloff_w"] == pytest.approx(0.4)
    # The wide-M* mestar window reaches the B-850R overrides; the other
    # tables keep the 2.5 window.
    ovr = _point("B-850R", {})
    assert ovr["mestar_eff_tower_max_mass"] == pytest.approx(2.75)
    ovr = _point("B-850G", {})
    assert ovr["mestar_eff_tower_max_mass"] == pytest.approx(2.5)
    # Round-4/5/6 strength budgets.
    for candidate, budget in (
        ("B-850G", 1.00),
        ("B-850N", 1.00),
        ("B-850R", 1.00),
        ("B-850C", 1.00),
        ("B-850S", 1.00),
        ("B-850WR", 1.00),
    ):
        assert float(sfr.STAGE_B_GRID_TABLES[candidate]["strength_budget"]) == pytest.approx(budget)


def test_stage_b_round7_tables_overrides_and_validation() -> None:
    """Round-7 tables: validation, pi0-share plumbing, composition pins.

    B-850P scans the three-body p pbar pi0 admixture through the
    mestar_eff_ppbar_pi0_share override (variant pinned ON; share 0
    reproduces the two-body family). B-850L composes the level-1
    cascade under the retained l2 configuration through the general
    pin_overrides key.
    """
    boxes = _stage_b_boxes_normalized()

    for candidate, level in (("B-850P", 2), ("B-850L", 1)):
        sfr._validate_stage_b_grid_table(candidate, level, boxes, MIXING2006)

    def _point(candidate: str, free_overrides: Dict[str, float]) -> Dict[str, float]:
        table = sfr.STAGE_B_GRID_TABLES[candidate]
        level = int(table["level"])
        free: Dict[str, float] = {
            f"nstar_eff_l{level}_mass": float(table["nstar_mass"][0]),
            f"nstar_eff_l{level}_width": float(table["nstar_width"][0]),
            "strength": float(table["strength"][0]),
        }
        if "mestar_mass" in table:
            free[f"mestar_eff_l{level}_mass"] = float(table["mestar_mass"][0])
            free[f"mestar_eff_l{level}_width"] = float(table["mestar_width"][0])
        for axis in ("pair_exponent", "pair_law_flag", "pair_m0", "pair_rolloff", "pair_pi0_share"):
            if axis in table:
                free.setdefault(axis, float(table[axis][0]))
        free.update(free_overrides)
        return sfr._stage_b_point_overrides(candidate, {"free": free}, MIXING2006)

    # E-A: the pi0 share reaches the override set with the variant ON;
    # share 0 reproduces the pure two-body family's weights.
    ovr = _point("B-850P", {"pair_pi0_share": 0.0})
    assert ovr["mestar_eff_ppbar_pi0_variant"] == pytest.approx(1.0)
    assert ovr["mestar_eff_ppbar_pi0_share"] == pytest.approx(0.0)
    ovr = _point("B-850P", {"pair_pi0_share": 1.0})
    assert ovr["mestar_eff_ppbar_pi0_share"] == pytest.approx(1.0)

    # E-B: the composition pins land in the override set; the candidate
    # level-1 split uses the frozen l1 budget.
    ovr = _point("B-850L", {})
    assert ovr["nstar_eff_l2_mass"] == pytest.approx(4.50)
    assert ovr["nstar_eff_l2_width"] == pytest.approx(0.06)
    assert ovr["mestar_eff_l2_mass"] == pytest.approx(2.25)
    assert ovr["nstar_eff_l2_br_nstarmeson"] == pytest.approx(1.00)
    assert ovr["nstar_eff_l2_br_nstarpi"] == pytest.approx(0.00)
    assert ovr["mestar_eff_ppbar_strength"] == pytest.approx(1.00)
    assert ovr["mestar_eff_ppbar_exponent"] == pytest.approx(0.25)
    assert ovr["nstar_eff_l1_br_nstarmeson"] == pytest.approx(0.0)
    assert ovr["nstar_eff_l1_br_nstarpi"] == pytest.approx(0.15)


def test_stage_b_relevance_partition() -> None:
    """Tower-primary components carry the full set; base components the empty set."""
    from helpers.benchmarking.benchmark_sps_species import _materialize_regge_tower_gen_files

    out_dir = Path("/tmp/opencode/stageb_test_partition")
    out_dir.mkdir(parents=True, exist_ok=True)
    tower_files = _materialize_regge_tower_gen_files(
        mixing2006_dir=MIXING2006,
        tower_gen_root=out_dir / "tower_gen",
        min_mass=1.9,
        max_mass=5.0,
        level_count=3,
        shape="linear",
        shape_param=0.25,
    )
    full = {"nstar_eff_l2_mass": 4.5}
    expected_tower = {"pi-": 3, "pi+": 3, "p": 3, "pbar": 3}
    for species in ("pi-", "pi+", "p", "pbar"):
        sp = sfr.resolve_species(species)
        specs = sfr._specs_for_species(MIXING2006, sp.pdg, tower_files=tower_files)
        plan = sfr._stage_b_component_plan(specs, full, "_p000")
        n_tower = 0
        for spec in specs:
            label = str(spec.label)
            entry = cast(Any, plan[label])
            if entry["ns_suffix"] == "_p000":
                assert entry["overrides"] == full
                n_tower += 1
            else:
                assert entry["overrides"] == {}
                assert entry["ns_suffix"] == ""
        assert n_tower == expected_tower[species]


def test_stage_b_selection_spec_roundtrip_and_tie_break() -> None:
    """The selection arm spec round-trips through _parse_arm_spec exactly."""
    boxes = _stage_b_boxes_normalized()
    point = sfr._stage_b_grid_points("B-850", "full")[10]
    ovr = sfr._stage_b_point_overrides("B-850", point, MIXING2006)
    spec = sfr._stage_b_selection_spec_string("B-850-sel", ovr)
    name, parsed = sfr._parse_arm_spec(spec, "default")
    assert name == "B-850-sel"
    assert set(parsed) == set(ovr)
    for key in ovr:
        assert parsed[key] == ovr[key], f"{key}: {parsed[key]} != {ovr[key]}"
    # Tie-break order: (objective, displacement, point id).
    points = [
        {"point_id": "B850-p001", "joint_train_wrrmse": 0.5, "displacement": 1.0},
        {"point_id": "B850-p000", "joint_train_wrrmse": 0.5, "displacement": 1.0},
        {"point_id": "B850-p002", "joint_train_wrrmse": 0.5, "displacement": 0.5},
        {"point_id": "B850-p003", "joint_train_wrrmse": 0.4, "displacement": 9.0},
    ]
    best = min(
        points,
        key=lambda p: (float(p["joint_train_wrrmse"]), float(p["displacement"]), str(p["point_id"])),
    )
    assert best["point_id"] == "B850-p003"  # lower objective wins outright
    tied = points[:3]
    best_tied = min(
        tied,
        key=lambda p: (float(p["joint_train_wrrmse"]), float(p["displacement"]), str(p["point_id"])),
    )
    assert best_tied["point_id"] == "B850-p002"  # displacement beats point id


def test_stageB_smoke_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hermetic stageB smoke: a two-point monkeypatched B-450 grid.

    Checks the cross-point cache reuse (base components built once, tower
    components rebuilt per point), finite joint objectives, the grid
    surface / selection payloads, and the ab-handoff spec round-trip.
    """
    table = _synthetic_na49_table(tmp_path)
    out_dir = tmp_path / "run"
    # Two-point B-450 grid for the smoke (module-level monkeypatch; the
    # pre-registered tables themselves are covered by the unit tests above).
    monkeypatch.setitem(
        sfr.STAGE_B_GRID_TABLES,
        "B-450",
        {
            "level": 1,
            "nstar_mass": (2.07, 2.20),
            "nstar_width": (0.25,),
            "strength": (0.075,),
        },
    )
    rc = sfr.main(
        [
            "--mode", "stageB",
            "--stage-b-candidate", "B-450",
            "--stage-b-grid", "full",
            "--species", "pi-",
            "--events-per-component", "20",
            "--band-seeds", "0",
            "--res-workers", "1",
            "--na49-data", str(table),
            "--out-dir", str(out_dir),
        ]
    )
    assert rc == sfr.EXIT_OK
    grid_path = out_dir / "stage_b_grid_B450.json"
    selection_path = out_dir / "stage_b_selection_B450.json"
    assert grid_path.exists()
    assert selection_path.exists()
    assert grid_path.with_suffix(".json.sha256").exists()
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    assert grid["candidate"] == "B-450"
    assert grid["candidate_level"] == 1
    assert grid["grid_table"]["point_count"] == 2
    assert grid["pair_strength"] is None  # B-450 is not pair-capable
    assert len(grid["points"]) == 2
    point0 = cast(Any, grid["points"][0])
    point1 = cast(Any, grid["points"][1])
    # Point 0 builds everything (15 base + 3 tower); point 1 reuses the 15
    # base caches and rebuilds the 3 tower caches at the new overrides.
    assert point0["builds"] == {"built": 18, "reused": 0}
    assert point1["builds"] == {"built": 3, "reused": 15}
    for point in (point0, point1):
        assert math.isfinite(point["joint_train_wrrmse"])
        assert set(point["species_train_wrrmse"]) == {"pi-"}
        assert point["openness"]["guard_floor"] == sfr.OPENNESS_GUARD_FLOOR
        # The results file exists per (point, species).
        for sp_payload in point["species"].values():
            assert Path(str(sp_payload["results_file"])).exists()
    # B-450 point overrides: no pair family, D4 split sums to base.
    ovr0 = point0["overrides"]
    assert "mestar_eff_ppbar_strength" not in ovr0
    assert ovr0["nstar_eff_l1_mass"] == 2.07
    assert ovr0["nstar_eff_l2_mass"] == 2.1  # pinned non-candidate level
    assert ovr0["nstar_eff_l1_br_nstarmeson"] + ovr0["nstar_eff_l1_br_nstarpi"] == pytest.approx(0.15)
    # Selection payload and ab-handoff round-trip.
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert selection["candidate"] == "B-450"
    assert selection["selected_point_id"] in {point0["point_id"], point1["point_id"]}
    name, parsed = sfr._parse_arm_spec(str(selection["selection_spec"]), "default")
    assert name == "B-450-sel"
    selected = point0 if selection["selected_point_id"] == point0["point_id"] else point1
    assert set(parsed) == set(selected["overrides"])
    for key in selected["overrides"]:
        assert parsed[key] == selected["overrides"][key]
    # The grid selection block agrees with the selection artifact.
    assert grid["selection"]["selected_point_id"] == selection["selected_point_id"]
    assert grid["totals"] == {"built": 21, "reused": 15}


def test_ab_mode_per_arm_tower_window(tmp_path: Path) -> None:
    """ab mode materializes each arm's N* tower from its own window.

    Arm A keeps the CLI window (the frozen Stage-A baseline); arm B widens
    via the ``nstar_eff_tower_max_mass`` key of its selection spec, so the
    realized candidate line shape matches the Stage-B grid point it came
    from. A plain spec without the key falls back to the CLI window.
    """
    table = _synthetic_na49_table(tmp_path)
    out_dir = tmp_path / "ab"
    spec_b = (
        "B-850-sel:nstar_eff_tower_max_mass=5.0,"
        "nstar_eff_l2_mass=4.6,nstar_eff_l2_width=0.15"
    )
    rc = sfr.main(
        [
            "--mode", "ab",
            "--config-a", "stageA",
            "--config-b", spec_b,
            "--species", "pi-", "pi+",
            "--events-per-component", "1000",
            "--band-seeds", "0",
            "--replicates", "0",
            "--res-workers", "2",
            "--na49-data", str(table),
            "--out-dir", str(out_dir),
        ]
    )
    assert rc == sfr.EXIT_OK
    assert (out_dir / sfr.AB_COMPARISON_FILENAME).exists()

    # (a) Per-arm tower windows materialize under the per-arm config_ns.
    tower_a = out_dir / "tower_gen" / "stageA" / "mmin1900_mmax2300"
    tower_b = out_dir / "tower_gen" / "B-850-sel" / "mmin1900_mmax5000"
    l1_a = tower_a / "nstar_eff1900_zero.gen"
    l1_b = tower_b / "nstar_eff1900_zero.gen"
    l2_a = tower_a / "nstar_eff2100_zero.gen"
    l2_b = tower_b / "nstar_eff2100_zero.gen"
    for path in (l1_a, l1_b, l2_a, l2_b):
        assert path.exists(), path
    # The l1 card is window-independent: byte-identical across arms.
    assert l1_a.read_bytes() == l1_b.read_bytes()

    # (b) The l2 cards equal fresh materializations via the driver's own
    #     materializer at the CLI (2.3) and grid-point (5.0) windows, and
    #     differ across arms (window-dependent nominal mass/width plus the
    #     damping bounds in the card).
    ref_a = cast(
        Dict[str, Path],
        sfr._materialize_tower_files(
            mixing2006_dir=MIXING2006,
            out_dir=tmp_path,
            config_ns="ref_a",
            tower_min_mass=1.9,
            tower_max_mass=2.3,
            tower_level_count=3,
            tower_shape="linear",
            tower_shape_param=0.25,
            disable_tower=False,
        ),
    )
    ref_b = cast(
        Dict[str, Path],
        sfr._materialize_tower_files(
            mixing2006_dir=MIXING2006,
            out_dir=tmp_path,
            config_ns="ref_b",
            tower_min_mass=1.9,
            tower_max_mass=5.0,
            tower_level_count=3,
            tower_shape="linear",
            tower_shape_param=0.25,
            disable_tower=False,
        ),
    )
    assert l2_a.read_bytes() == ref_a["nstar_eff2100_zero"].read_bytes()
    assert l2_b.read_bytes() == ref_b["nstar_eff2100_zero"].read_bytes()
    assert l2_a.read_bytes() != l2_b.read_bytes()
    # [.. mass_type schwel rmas brei pbrei dumplow dumphigh]
    assert l2_a.read_text().split()[-5:] == [
        "2.100000", "0.300000", "1", "2.700000", "3.400000",
    ]
    assert l2_b.read_text().split()[-5:] == [
        "3.450000", "0.637500", "1", "4.725000", "5.362500",
    ]

    # (c) The realized candidate mass law: arm B's l2 tower component cache
    #     carries the effective pole with the 5.0-window damping bounds;
    #     arm A's is the nominal 2.3-window card (no overrides).
    res_a = json.loads(
        (out_dir / "ab_results_stageA_pi_minus_rep0.json").read_text(encoding="utf-8")
    )
    res_b = json.loads(
        (out_dir / "ab_results_B-850-sel_pi_minus_rep0.json").read_text(encoding="utf-8")
    )
    cache_a = Path(str(cast(Any, res_a["components"]["nstar_eff2100_zero"])["cache_file"]))
    cache_b = Path(str(cast(Any, res_b["components"]["nstar_eff2100_zero"])["cache_file"]))
    with np.load(cache_a, allow_pickle=False) as z:
        law_a = cast(Any, json.loads(str(z["meta_mass_law_json"])))
    with np.load(cache_b, allow_pickle=False) as z:
        law_b = cast(Any, json.loads(str(z["meta_mass_law_json"])))
    assert law_a["type"] == "breit_wigner"
    assert [float(v) for v in law_a["params"]] == pytest.approx(
        [1.22, 2.1, 0.3, 1.0, 2.7, 3.4]
    )
    assert law_b["type"] == "breit_wigner"
    assert [float(v) for v in law_b["params"]] == pytest.approx(
        [1.22, 4.6, 0.15, 1.0, 4.725, 5.3625]
    )

    # Fallback: a spec without the window key keeps the CLI window.
    out_fallback = tmp_path / "ab_fallback"
    rc2 = sfr.main(
        [
            "--mode", "ab",
            "--config-a", "stageA",
            "--config-b", "plainB:nstar_eff_l2_mass=2.5",
            "--species", "pi-",
            "--events-per-component", "20",
            "--band-seeds", "0",
            "--replicates", "0",
            "--res-workers", "2",
            "--na49-data", str(table),
            "--out-dir", str(out_fallback),
        ]
    )
    assert rc2 == sfr.EXIT_OK
    plain_b = out_fallback / "tower_gen" / "plainB"
    assert (plain_b / "mmin1900_mmax2300").is_dir()
    assert not (plain_b / "mmin1900_mmax5000").exists()

    # (d) Regression: the --stage-b-candidate flow must not leak the box
    #     window into arm A. The shared args.tower_max_mass stays at the CLI
    #     value, so arm A keeps the frozen-baseline window while arm B
    #     widens via the box window merged by _apply_stage_b_candidate.
    out_cand = tmp_path / "ab_cand"
    rc3 = sfr.main(
        [
            "--mode", "ab",
            "--config-a", "stageA",
            "--config-b", "B-850-sel",
            "--stage-b-candidate", "B-850",
            "--species", "pi-",
            "--events-per-component", "20",
            "--band-seeds", "0",
            "--replicates", "0",
            "--res-workers", "2",
            "--na49-data", str(table),
            "--out-dir", str(out_cand),
        ]
    )
    assert rc3 == sfr.EXIT_OK
    assert (
        out_cand / "tower_gen" / "stageA" / "mmin1900_mmax2300" / "nstar_eff1900_zero.gen"
    ).exists()
    assert not (out_cand / "tower_gen" / "stageA" / "mmin1900_mmax5000").exists()
    cand_b = out_cand / "tower_gen" / "B-850-sel"
    assert (cand_b / "mmin1900_mmax5000" / "nstar_eff2100_zero.gen").exists()
    assert not (cand_b / "mmin1900_mmax2300").exists()


def test_layout_mirror_stays_in_sync() -> None:
    """The monorepo and vendored driver copies stay logically in sync."""
    rc = layout_sync_main(["--quiet"])
    assert rc == 0, "helper trees drifted (run check_helpers_layout_sync.py)"
