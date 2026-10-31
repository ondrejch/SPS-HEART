#!/usr/bin/env python3
"""Offline mini-pipeline smoke and zero-column drop tests.

Covers the committed behavior of the offline mini-pipeline (nested SHAs
da1644d, 15ff879, b2bbb5b) and the all-zero design-column drop in
``benchmark_multispecies_constrained.py`` (outer SHA 8dccd2a, mirrored in the
nested tree):

1. the committed smoke path runs end to end from pytest: ``bash run_smoke.sh
   OUT_DIR`` exits 0 and prints the two final ``PASS mini-pipeline smoke``
   lines (run validation and run directory). The script guarantees a
   hermetic run: no network, no ``spshadrons.web.cern.ch/`` tables, no
   external generators, no UrQMD tarball;
2. the zero-column drop is tested directly and fast, without the smoke run.
   The drop decision lives inside the driver's ``main()`` (committed code has
   no lower-level seam), so the tests drive ``main()`` in-process on a tiny
   synthetic component cache with stubbed data loading, spec table, cache
   index, and SciPy NNLS -- the stubbing style of
   ``test_production_fit_validation.py`` (real SciPy never runs);
3. the committed fixture is self-consistent without executing the run: the
   manifest-recorded cache file list matches the 43 ``*.npz`` files on disk,
   the absolute-yield prior carries the documented round-number constants,
   and the two smoke validator modules compile in-process;
4. a clean extraction of the vendored tree (no ``.git``, no provenance
   environment variables, only the release archive's own ``SHA256SUMS.txt``
   inventory at the archive root) runs the same smoke end to end and records
   the file inventory as the run's source identity, accepted by the
   vendored copy of the provenance validator.

All tests are hermetic and carry no ``external_data`` marker.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import py_compile
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import helpers.benchmarking.benchmark_multispecies_constrained as bmc
from helpers.benchmarking.particle_definitions import resolve_species
from helpers.benchmarking.sps_core import FULL_RANK_STAMP, PDG_MASS_GEV

MINI_PIPELINE_DIR = REPO_ROOT / "dev" / "mixing2006" / "sps_resonances_paper" / "mini_pipeline"
SMOKE_SCRIPT = REPO_ROOT / "dev" / "mixing2006" / "sps_resonances_paper" / "run_smoke.sh"
FIXTURE_DIR = MINI_PIPELINE_DIR / "fixture"

SQRT_S_GEV = float(bmc.bss._sqrt_s_from_fixed_target_beam_momentum(158.0))
SMOKE_TIMEOUT_S = 600.0

# Synthetic 2x2 (xF, pT) data grid: four data points, one per bin.
_X_EDGES = np.array([0.0, 0.2, 0.4])
_P_EDGES = np.array([0.05, 0.25, 0.45])
_IX_DATA = np.array([0, 0, 1, 1], dtype=np.int64)
_IP_DATA = np.array([0, 1, 0, 1], dtype=np.int64)
_XF_CENTERS = np.array([0.1, 0.1, 0.3, 0.3])
_PT_CENTERS = np.array([0.15, 0.35, 0.15, 0.35])

# Synthetic component caches. Label semantics mirror the production labels:
# a phi_1020 component decays to K+ K- only, so its stacked design columns
# are identically zero for a pions-only fit; delta and rho0 produce charged
# pions; the kstar component produces pi+ only (zero for pi- alone, which
# the policy must keep).
_PI_MINUS = np.int32(-211)
_PI_PLUS = np.int32(211)

_DELTA_TRACKS = {
    "pdg": np.array([_PI_MINUS, _PI_MINUS], dtype=np.int32),
    "xf": np.array([0.1, 0.3]),
    "pt": np.array([0.15, 0.35]),
}
_PHI_TRACKS = {
    "pdg": np.array([321, -321], dtype=np.int32),
    "xf": np.array([0.1, 0.3]),
    "pt": np.array([0.15, 0.35]),
}
_RHO_TRACKS = {
    "pdg": np.array([_PI_PLUS, _PI_MINUS], dtype=np.int32),
    "xf": np.array([0.1, 0.3]),
    "pt": np.array([0.35, 0.15]),
}
_KSTAR_TRACKS = {
    "pdg": np.array([_PI_PLUS], dtype=np.int32),
    "xf": np.array([0.1]),
    "pt": np.array([0.15]),
}

_SPECIES_KEYS = ("pi-", "pi+")


def _smoke_skip_reason() -> str | None:
    if shutil.which("bash") is None:
        return "bash is unavailable"
    if not SMOKE_SCRIPT.is_file():
        return f"mini-pipeline smoke script not present: {SMOKE_SCRIPT}"
    return None


# ---------------------------------------------------------------------------
# Area 1: hermetic end-to-end smoke run of the committed script.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_smoke_skip_reason() is not None, reason=_smoke_skip_reason() or "")
def test_run_smoke_script_end_to_end(tmp_path):
    """run_smoke.sh completes offline and prints its two final PASS lines."""
    out_dir = tmp_path / "smoke_run"
    proc = subprocess.run(
        ["bash", str(SMOKE_SCRIPT), str(out_dir)],
        capture_output=True,
        text=True,
        timeout=SMOKE_TIMEOUT_S,
        cwd=str(SMOKE_SCRIPT.parent),
    )
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-4000:]

    pass_lines = [
        line for line in proc.stdout.splitlines()
        if line.startswith("PASS mini-pipeline smoke")
    ]
    assert len(pass_lines) == 2, proc.stdout
    assert any(line.startswith("PASS mini-pipeline smoke validation:") for line in pass_lines)
    assert any(line.startswith("PASS mini-pipeline smoke run directory:") for line in pass_lines)

    # The run directory carries the production-fit artifacts, and the
    # pions-only smoke fit drops exactly the kaon-only phi component.
    summary = json.loads(
        (out_dir / "multispecies_constrained_summary.json").read_text(encoding="utf-8")
    )
    assert summary["resonance"]["dropped_zero_column_components"] == ["phi_1020"]
    assert summary["fit_solver"]["identifiability"] == FULL_RANK_STAMP


# ---------------------------------------------------------------------------
# Area 2: zero-column drop unit tests (in-process main() on synthetic caches).
# ---------------------------------------------------------------------------


def _synthetic_species(key: str, table_path: Path) -> bmc.SpeciesData:
    """One species on the synthetic grid with placeholder y and sigma."""
    spec = resolve_species(key)
    zeros = np.zeros(4, dtype=float)
    return bmc.SpeciesData(
        key=key,
        name=spec.name,
        tag=spec.tag,
        pdg=spec.pdg,
        mass_gev=float(PDG_MASS_GEV.get(spec.pdg, 0.1396)),
        na49_path=table_path,
        xf=_XF_CENTERS.copy(),
        pt=_PT_CENTERS.copy(),
        y=zeros.copy(),
        sigma=np.full(4, 0.1),
        xedges=_X_EDGES.copy(),
        pedges=_P_EDGES.copy(),
        ix_data=_IX_DATA.copy(),
        ip_data=_IP_DATA.copy(),
        train_idx=np.arange(4, dtype=np.int64),
        test_idx=np.array([], dtype=np.int64),
        matrix=np.zeros((4, 0), dtype=float),
        pred_res=zeros.copy(),
        pred_res_sigma=zeros.copy(),
    )


def _run_driver(
    monkeypatch,
    tmp_path: Path,
    tracks_by_label: dict[str, dict[str, np.ndarray]],
    planted_weights: dict[str, float],
) -> dict:
    """Drive the driver's ``main()`` on a synthetic component cache.

    Stubs the species-table loading, the spec table, the component-cache
    index, and SciPy's ``nnls`` (real SciPy must not run). The data vectors
    are planted as ``y = A @ w`` from the real column builder, so the stubbed
    solve recovers ``planted_weights`` exactly and any weight disagreement
    means the assembled design changed.
    """
    cache_dir = tmp_path / "resonance_component_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_index: dict[str, Path] = {}
    for label, tracks in tracks_by_label.items():
        path = cache_dir / f"component__{label}.npz"
        np.savez(path, **tracks)
        cache_index[label] = path

    species_data = {}
    for key in _SPECIES_KEYS:
        table_path = tmp_path / f"na49_table_{key}.txt"
        table_path.write_text("synthetic stand-in for the NA49 table\n", encoding="utf-8")
        species_data[key] = _synthetic_species(key, table_path)

    labels_sorted = sorted(tracks_by_label)
    full_weights = np.array(
        [planted_weights.get(label, 0.0) for label in labels_sorted], dtype=float
    )
    for sd in species_data.values():
        cols = np.column_stack(
            [
                bmc._species_column_from_tracks(
                    pdg_arr=tracks["pdg"],
                    xf_arr=tracks["xf"],
                    pt_arr=tracks["pt"],
                    sd=sd,
                    convention="invariant",
                    sqrt_s_gev=SQRT_S_GEV,
                )
                for tracks in (tracks_by_label[label] for label in labels_sorted)
            ]
        )
        sd.y = cols @ full_weights
        sd.sigma = 0.1 * (1.0 + np.abs(sd.y))

    specs = {label: types.SimpleNamespace(label=label) for label in labels_sorted}
    w_planted = np.array(
        [planted_weights[label] for label in sorted(planted_weights)], dtype=float
    )

    def _fake_load_species_points(*, species_key: str, **_kwargs) -> bmc.SpeciesData:
        return species_data[species_key]

    def _fake_nnls(a, b):
        return w_planted.copy(), 0.0

    argv = [
        "benchmark_multispecies_constrained.py",
        "--run-description", "zero_column_unit",
        "--out-dir", str(tmp_path / "run"),
        "--species", ",".join(_SPECIES_KEYS),
        "--data-system", "pp",
        "--pa-a", "0",
        "--beam-momentum", "158.0",
        "--convention", "invariant",
        "--seed", "20260830",
        "--res-events-per-res", "40",
        "--res-workers", "1",
        "--res-fragmentation-mode", "both",
        "--res-disable-regge-tower",
        "--res-cache-dir", str(cache_dir),
        "--fit-objective", "nnls_weighted",
        "--skip-external",
        "--repro-lock", str(tmp_path / "absent_reproducibility.lock.json"),
    ]

    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(bmc, "_load_species_points", _fake_load_species_points)
    monkeypatch.setattr(bmc.bss, "_all_resonance_specs", lambda *a, **k: dict(specs))
    monkeypatch.setattr(
        bmc.bss, "_build_resonance_component_cache", lambda *a, **k: dict(cache_index)
    )
    monkeypatch.setattr(bmc, "nnls", _fake_nnls)
    bmc.main()

    return json.loads(
        (tmp_path / "run" / "multispecies_constrained_summary.json").read_text(
            encoding="utf-8"
        )
    )


def _details_weights(tmp_path: Path) -> tuple[list[str], np.ndarray]:
    details_path = tmp_path / "run" / "multispecies_constrained_details.npz"
    with np.load(details_path, allow_pickle=False) as z:
        return list(z["component_labels"]), np.asarray(z["component_weights"], dtype=float)


def test_zero_column_component_is_dropped_and_recorded(monkeypatch, tmp_path, capsys):
    """A kaon-only component is dropped, recorded, and the fit stays full-rank."""
    tracks = {"delta_1232": _DELTA_TRACKS, "phi_1020": _PHI_TRACKS, "rho0_770": _RHO_TRACKS}
    planted = {"delta_1232": 1.4, "rho0_770": 0.8}

    summary = _run_driver(monkeypatch, tmp_path, tracks, planted)
    resonance = summary["resonance"]

    # Decision payload and exclusion from the assembled design.
    assert resonance["dropped_zero_column_components"] == ["phi_1020"]
    assert resonance["component_labels"] == ["delta_1232", "rho0_770"]
    assert resonance["n_components"] == 2
    labels_npz, weights_npz = _details_weights(tmp_path)
    assert labels_npz == ["delta_1232", "rho0_770"]
    np.testing.assert_allclose(weights_npz, [1.4, 0.8], rtol=0, atol=0)

    # component-note message on stdout.
    out = capsys.readouterr().out
    assert "component-note: dropped 1 component(s)" in out
    assert "['phi_1020']" in out

    # The rank/identifiability policy evaluates the reduced design.
    solver = summary["fit_solver"]
    assert solver["success"] is True
    assert solver["design_rank"] == solver["design_n_components"] == 2
    assert solver["identifiability"] == FULL_RANK_STAMP
    assert resonance["weights_by_label_sum"] == {"delta_1232": 1.4, "rho0_770": 0.8}


def test_no_zero_column_component_drops_nothing(monkeypatch, tmp_path, capsys):
    """An all-components-non-zero cache drops nothing and emits no note."""
    tracks = {"delta_1232": _DELTA_TRACKS, "rho0_770": _RHO_TRACKS}
    planted = {"delta_1232": 1.4, "rho0_770": 0.8}

    summary = _run_driver(monkeypatch, tmp_path, tracks, planted)
    resonance = summary["resonance"]

    assert resonance["dropped_zero_column_components"] == []
    assert resonance["component_labels"] == ["delta_1232", "rho0_770"]
    assert summary["fit_solver"]["design_rank"] == 2
    assert summary["fit_solver"]["identifiability"] == FULL_RANK_STAMP
    assert "component-note" not in capsys.readouterr().out


def test_zero_column_for_one_species_only_is_kept(monkeypatch, tmp_path):
    """A column that is zero for pi- but not for pi+ stays in the design."""
    tracks = {
        "delta_1232": _DELTA_TRACKS,
        "kstar_plus_892": _KSTAR_TRACKS,
        "rho0_770": _RHO_TRACKS,
    }
    planted = {"delta_1232": 1.4, "kstar_plus_892": 0.6, "rho0_770": 0.8}

    summary = _run_driver(monkeypatch, tmp_path, tracks, planted)
    resonance = summary["resonance"]

    assert resonance["dropped_zero_column_components"] == []
    assert resonance["component_labels"] == ["delta_1232", "kstar_plus_892", "rho0_770"]
    solver = summary["fit_solver"]
    assert solver["design_rank"] == solver["design_n_components"] == 3
    assert solver["identifiability"] == FULL_RANK_STAMP


def test_only_zero_column_components_is_fatal(monkeypatch, tmp_path):
    """A cache whose every component is zero-column leaves no design to fit."""
    tracks = {"phi_1020": _PHI_TRACKS}
    with pytest.raises(RuntimeError, match="No resonance components available"):
        _run_driver(monkeypatch, tmp_path, tracks, {})


# ---------------------------------------------------------------------------
# Area 3: committed fixture self-consistency, without executing the run.
# ---------------------------------------------------------------------------


def _load_expected_smoke_module():
    path = MINI_PIPELINE_DIR / "expected_smoke.py"
    spec = importlib.util.spec_from_file_location("_mini_pipeline_expected_smoke", path)
    assert spec is not None and spec.loader is not None, f"cannot import {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(
    not FIXTURE_DIR.is_dir(), reason=f"mini-pipeline fixture not present: {FIXTURE_DIR}"
)
def test_fixture_manifest_records_the_committed_cache_files():
    """Manifest, cache manifest, and disk agree on the 43 committed arrays."""
    manifest = json.loads(
        (FIXTURE_DIR / "fixture_manifest.json").read_text(encoding="utf-8")
    )
    cache_dir = FIXTURE_DIR / "resonance_component_cache"
    cache_manifest = json.loads(
        (cache_dir / "cache_manifest.json").read_text(encoding="utf-8")
    )

    disk_files = sorted(path.name for path in cache_dir.glob("*.npz"))
    recorded_files = sorted(
        Path(entry["cache_file"]).name for entry in cache_manifest["components"]
    )

    assert manifest["n_cache_files"] == 43
    assert len(disk_files) == 43
    assert recorded_files == disk_files

    # Documented fixture-generation constants.
    assert manifest["component_cache_seed"] == manifest["smoke_seed"] + 19
    assert manifest["data_system"] == "pPb"
    assert manifest["events_per_component"] == 40
    assert manifest["regge_tower_disabled"] is True

    # The component the pions-only smoke fit drops is part of the cache.
    labels = {entry["label"] for entry in cache_manifest["components"]}
    assert "phi_1020" in labels


@pytest.mark.skipif(
    not FIXTURE_DIR.is_dir(), reason=f"mini-pipeline fixture not present: {FIXTURE_DIR}"
)
def test_absyield_prior_targets_are_the_documented_constants():
    """The fixture prior carries the documented round-number constants."""
    prior = json.loads(
        (FIXTURE_DIR / "absyield_prior.json").read_text(encoding="utf-8")
    )
    assert prior["targets"] == {"pi-": 60.0, "pi+": 65.0}
    assert prior["sigmas"] == {"pi-": 6.0, "pi+": 6.5}
    assert prior["window"] == {"xf_min": 0.0, "xf_max": 0.4}
    assert _load_expected_smoke_module().EXPECTED_PRIOR_TARGETS == prior["targets"]


@pytest.mark.skipif(
    not MINI_PIPELINE_DIR.is_dir(), reason=f"mini_pipeline directory not present: {MINI_PIPELINE_DIR}"
)
def test_smoke_modules_compile_in_process(tmp_path):
    """validate_smoke.py and expected_smoke.py are valid Python sources."""
    for name in ("validate_smoke.py", "expected_smoke.py"):
        py_compile.compile(
            str(MINI_PIPELINE_DIR / name),
            cfile=str(tmp_path / f"{name}c"),
            doraise=True,
        )


# ---------------------------------------------------------------------------
# Area 4: clean-extraction smoke run -- the source identity must come from
# the release archive's file inventory when no git metadata is available.
# ---------------------------------------------------------------------------

VENDORED_SOURCE_TREES = ("helpers", "na49_mixing", "sps_resonances_paper")
INVENTORY_FILENAME = "SHA256SUMS.txt"
INVENTORY_TOOL_TIMEOUT_S = 120.0


def _load_module_from_path(alias: str, path: Path):
    spec = importlib.util.spec_from_file_location(alias, path)
    assert spec is not None and spec.loader is not None, f"cannot import {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_clean_extraction(archive_root: Path) -> Path:
    """Copy the vendored layout into <archive_root>/dev/mixing2006.

    Regular files only (no symlinks, no caches): the copy stands in for a
    release-archive extraction, which carries no ``.git`` and whose only
    provenance is the file inventory at the archive root.
    """
    src_root = REPO_ROOT / "dev" / "mixing2006"
    vendored = archive_root / "dev" / "mixing2006"
    vendored.mkdir(parents=True)
    for sub in VENDORED_SOURCE_TREES:
        shutil.copytree(
            src_root / sub,
            vendored / sub,
            symlinks=False,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
        )
    checker = REPO_ROOT / "helpers" / "benchmarking" / "make_archive_checksums.py"
    proc = subprocess.run(
        [sys.executable, str(checker), "--staging", str(archive_root)],
        capture_output=True,
        text=True,
        timeout=INVENTORY_TOOL_TIMEOUT_S,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-2000:]
    inventory_path = archive_root / INVENTORY_FILENAME
    assert inventory_path.is_file()
    return vendored


@pytest.mark.skipif(_smoke_skip_reason() is not None, reason=_smoke_skip_reason() or "")
def test_clean_extraction_smoke_records_inventory_source_identity(tmp_path):
    """Without git metadata or provenance env vars, the smoke run records
    the archive inventory's sha256 as its source identity, and the vendored
    validator accepts the record (review 8, Area 4 regression)."""
    archive_root = tmp_path / "na49-res-source"
    vendored = _build_clean_extraction(archive_root)
    inventory_digest = hashlib.sha256(
        (archive_root / INVENTORY_FILENAME).read_bytes()
    ).hexdigest()

    out_dir = tmp_path / "clean_extraction_smoke_out"
    env = dict(os.environ)
    env.pop("NA49_RUN_SOURCE_ARCHIVE", None)
    env["RUNS_ROOT"] = str(tmp_path / "runs")
    # Point the pinned-venv lookup at nothing so the script falls back to
    # the same interpreter the test runs under.
    env["VENV_DIR"] = str(tmp_path / "venv-absent")
    smoke_script = vendored / "sps_resonances_paper" / "run_smoke.sh"
    proc = subprocess.run(
        ["bash", str(smoke_script), str(out_dir)],
        capture_output=True,
        text=True,
        timeout=SMOKE_TIMEOUT_S,
        cwd=str(vendored / "sps_resonances_paper"),
        env=env,
    )
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-4000:]
    assert "PASS mini-pipeline smoke run directory" in proc.stdout

    record = json.loads(
        (out_dir / "run_provenance.json").read_text(encoding="utf-8")
    )
    source = record["source"]
    assert source["outer_head"] is None
    assert source["nested_head"] is None
    assert source["source_archive"] is None
    assert source["source_inventory_sha256"] == inventory_digest
    assert source["recorded_from"] == "source_inventory"
    assert Path(source["source_inventory_path"]).resolve() == (
        archive_root / INVENTORY_FILENAME
    ).resolve()

    # The validator shipped inside the extraction must accept the record the
    # extraction itself produced.
    vendored_provenance = _load_module_from_path(
        "_vendored_run_provenance",
        vendored / "helpers" / "benchmarking" / "run_provenance.py",
    )
    assert vendored_provenance.validate_run_provenance(out_dir / "run_provenance.json") == []
