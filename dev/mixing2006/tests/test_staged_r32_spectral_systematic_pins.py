"""Round 3 card 9: run R32 spectral-systematic pins (phase 1 register).

Task ``TASK-20260918-r3-run-r32-spectral-systematic-pp`` (plan section 9
run R32, line 644).  This module pins the phase-1 implementation state
and, when the sealed run dir is present (phase 2), the sealed run
evidence:

- ``test_dual_tree_driver_bit_identity``: the nested and outer driver
  copies are byte-identical at the pinned digest, the post-extension
  builder digest, the fixed card-2 engine digest, the committed catalog
  and continuum-config digests, and the frozen R31 / R3-SE-audit driver
  digests all match the on-disk files (the card's substrate list).
- ``test_freeze_manifest_repin``: the legacy_v1 manifest inventory pins
  the post-extension builder, the driver in both helper trees, and the
  two new test files at their on-disk digests, and carries the
  freeze_exceptions entry naming this card (the re-pin of the additive
  builder extension and the new pinned-root files).
- ``test_pre_extension_builder_reference``: the sealed R23 replica
  records pin the pre-extension builder digest (the byte-unchanged
  reference the provenance records).
- ``test_prior_sealed_dirs_byte_unchanged``: the six sealed run dirs of
  the card's substrate (R23, R30, R31, card-2, card-8a, card-8c)
  re-hash cleanly against their ``SHA256SUMS.txt`` inventories.
- The sealed-run pins (skip when ``dev/00runs/
  r32_spectral_systematic_pp_2026-09-18/`` is absent, i.e. before
  phase 2 lands): the run-dir inventory, the seal integrity, the
  pre-registration invariants, the provenance hashes, the gate clause
  verdicts, and the clean-environment record.

Sibling of ``tests/test_staged_r31_matching_scan_gate_pins.py`` (the
same pin style: sealed-dir discovery, skip when the gitignored archive
is absent, digest pins plus structural pins).  Read-only over the
sealed artifacts: nothing is recomputed, the sealed run is never
re-sealed, and the clean-environment record is never re-executed.

Hermetic: reads the committed sources and the sealed run dirs; no
computation runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

NESTED_ROOT = Path(__file__).resolve().parent.parent
MONO_ROOT = NESTED_ROOT.parent.parent
RUN_NAME = "r32_spectral_systematic_pp_2026-09-18"
RUN_DIR = MONO_ROOT / "dev" / "00runs" / RUN_NAME

DRIVER_REL = Path("helpers") / "benchmarking" / (
    "staged_r32_spectral_systematic.py"
)
NESTED_DRIVER = NESTED_ROOT / DRIVER_REL
OUTER_DRIVER = MONO_ROOT / DRIVER_REL
BUILDER_PATH = NESTED_ROOT / "na49_mixing" / "resonance_pp_builder.py"
ENGINE_PATH = NESTED_ROOT / "na49_mixing" / "threshold_metric.py"
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / (
    "resonance_catalog.json"
)
FAMILIES_PATH = NESTED_ROOT / "config" / "v2_singlet" / (
    "continuum_resonances.json"
)
MANIFEST_PATH = NESTED_ROOT / "config" / "v2_singlet" / (
    "legacy_v1_manifest.json"
)
R31_DRIVER_PATH = NESTED_ROOT / "helpers" / "benchmarking" / (
    "staged_r31_matching_scan.py"
)
SE_AUDIT_DRIVER_PATH = NESTED_ROOT / "helpers" / "benchmarking" / (
    "staged_r3_se_audit.py"
)

#: The phase-1 digests (the committed state of the card's changed
#: surfaces and frozen substrate).  The driver digest is the post-fix
#: state of the card's phase-1 review r1 (REV-b9df812-01: the smoke
#: chunk worker now draws its kernel substreams at the pre-registered
#: smoke block 5000 + 100*v + r, the stamped record field).
DRIVER_SHA256 = (
    "99454facce856c93505e6c4b0d4e47b85eeb0ef54f72b1631e3d29fcc511cb90"
)
BUILDER_POST_EXTENSION_SHA256 = (
    "4ab612609e1ee0bc1e037dcbb2815b6484306ad32402b7289e6636f8e829cdb1"
)
BUILDER_R23_PRE_EXTENSION_SHA256 = (
    "bde0fa9a228f3e9b713cc646bddfc64e2b66b7968a5db3e0c8050d169e81e036"
)
ENGINE_SHA256 = (
    "ee240f44645d2e592acfae462f7d1e9c3b3dd8cd78c96bd40b7b73509f7fcc63"
)
CATALOG_SHA256 = (
    "6e4de7372a5bbf99affd6fa3ccc295c4cef09404dab3b17acd8816cda762bdcd"
)
FAMILIES_SHA256 = (
    "242f906cf040eadbe3796dfcfb61d0d51535277399858c901a162e6653d635e1"
)
R31_DRIVER_SHA256 = (
    "b47b2a858c3880ecf69e143035e8c209b6e6e5e0708ca2b9058c6c384340afe6"
)
SE_AUDIT_DRIVER_SHA256 = (
    "2c72fa7e7b5a86a54076f6cc686bb33d165d2b280092df7ca6fd75d3b5635081"
)

#: The re-pinned legacy_v1 inventory entries of this card (the
#: post-extension builder, the dual-tree driver, the two new test
#: files, and the card-3 pin register whose builder digest pin follows
#: the sanctioned extension), and the freeze-exception card name.
REPINNED_FILES = (
    "dev/mixing2006/na49_mixing/resonance_pp_builder.py",
    "dev/mixing2006/helpers/benchmarking/staged_r32_spectral_systematic.py",
    "helpers/benchmarking/staged_r32_spectral_systematic.py",
    "dev/mixing2006/tests/test_staged_r32_spectral_systematic.py",
    "dev/mixing2006/tests/test_staged_r32_spectral_systematic_pins.py",
    "dev/mixing2006/tests/test_feeddown_lineage_convention.py",
)
FREEZE_EXCEPTION_CARD = (
    "TASK-20260918-r3-run-r32-spectral-systematic-pp"
)

#: The card's sealed substrate (read-only; their seals verify).
PRIOR_SEALED_DIRS = (
    "r23_resonance_only_pp_ablation_2026-09-13",
    "r30_spectral_normalization_2026-09-17",
    "r31_catalog_continuum_matching_scan_2026-09-17",
    "r3_threshold_metric_reregistration_2026-09-17",
    "r3_threshold_metric_engine_fix_2026-09-18",
    "r3_se_audit_2026-09-18",
)

#: The production spread-bounds contract (the gate verdicts pin these).
BIN_SPREAD_BOUND = 2.0
FEEDDOWN_SPREAD_BOUND = 3.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------
# Phase-1 pins (always runnable)
# ----------------------------------------------------------------------
def test_dual_tree_driver_bit_identity():
    """The nested and outer driver copies are byte-identical at the
    pinned digest, and the frozen substrate digests match the on-disk
    files."""
    assert NESTED_DRIVER.is_file() and OUTER_DRIVER.is_file()
    nested = _sha256(NESTED_DRIVER)
    outer = _sha256(OUTER_DRIVER)
    assert nested == outer == DRIVER_SHA256, (
        "the dual-tree driver digest drifted (the pin and the tree "
        f"disagree: nested {nested}, outer {outer})"
    )
    assert _sha256(BUILDER_PATH) == BUILDER_POST_EXTENSION_SHA256
    assert _sha256(ENGINE_PATH) == ENGINE_SHA256
    assert _sha256(CATALOG_PATH) == CATALOG_SHA256
    assert _sha256(FAMILIES_PATH) == FAMILIES_SHA256
    assert _sha256(R31_DRIVER_PATH) == R31_DRIVER_SHA256
    assert _sha256(SE_AUDIT_DRIVER_PATH) == SE_AUDIT_DRIVER_SHA256


def test_freeze_manifest_repin():
    """The legacy_v1 manifest inventory pins the re-pinned files at
    their on-disk digests and carries the freeze_exceptions entry
    naming this card."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    inventory = manifest["source_tree_sha256"]
    on_disk = {
        "dev/mixing2006/na49_mixing/resonance_pp_builder.py":
            BUILDER_POST_EXTENSION_SHA256,
        "dev/mixing2006/helpers/benchmarking/"
        "staged_r32_spectral_systematic.py": DRIVER_SHA256,
        "helpers/benchmarking/staged_r32_spectral_systematic.py":
            DRIVER_SHA256,
    }
    for rel, digest in on_disk.items():
        assert inventory.get(rel) == digest, (
            f"the manifest does not pin {rel} at {digest} (the re-pin "
            "is missing or drifted)"
        )
    for rel in (
        "dev/mixing2006/tests/test_staged_r32_spectral_systematic.py",
        "dev/mixing2006/tests/test_staged_r32_spectral_systematic_pins.py",
    ):
        path = MONO_ROOT / rel
        assert inventory.get(rel) == _sha256(path), (
            f"the manifest does not pin {rel} at its on-disk digest"
        )
    # The inventory is sorted by relative path (the strict order pin).
    assert list(inventory) == sorted(inventory)
    exceptions = manifest.get("freeze_exceptions", [])
    entries = [
        e for e in exceptions if e.get("card") == FREEZE_EXCEPTION_CARD
    ]
    assert len(entries) == 1, (
        "the freeze_exceptions carry exactly one entry naming this card"
    )
    named = set(entries[0].get("files", []))
    assert named == set(REPINNED_FILES), (
        "the freeze exception names exactly the re-pinned files"
    )


def test_pre_extension_builder_reference():
    """The sealed R23 replica records pin the pre-extension builder
    digest (the byte-unchanged reference of the additive extension)."""
    r23_dir = MONO_ROOT / "dev" / "00runs" / PRIOR_SEALED_DIRS[0]
    if not r23_dir.is_dir():
        pytest.skip(f"the sealed R23 run dir is absent: {r23_dir}")
    for seed in range(20300913, 20300921):
        record = json.loads(
            (r23_dir / f"replica_{seed}.json").read_text(encoding="utf-8")
        )
        assert record["builder_module_sha256"] == (
            BUILDER_R23_PRE_EXTENSION_SHA256
        )


@pytest.mark.parametrize("run_name", PRIOR_SEALED_DIRS)
def test_prior_sealed_dirs_byte_unchanged(run_name):
    """The six sealed run dirs of the card's substrate re-hash cleanly
    against their SHA256SUMS.txt inventories (read-only; skip when the
    artifact store is absent)."""
    run_dir = MONO_ROOT / "dev" / "00runs" / run_name
    sums_path = run_dir / "SHA256SUMS.txt"
    if not sums_path.is_file():
        pytest.skip(f"the sealed run dir is absent: {run_dir}")
    mismatches = []
    entries = 0
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        payload = MONO_ROOT / name
        if not payload.is_file():
            payload = run_dir / name
        entries += 1
        if not payload.is_file() or _sha256(payload) != digest:
            mismatches.append(name)
    assert entries > 0
    assert mismatches == [], (
        f"{run_dir}: {len(mismatches)} of {entries} sums entries do not "
        f"re-hash correctly (first: {mismatches[0]!r})"
    )


# ----------------------------------------------------------------------
# Sealed run pins (skip before phase 2 lands)
# ----------------------------------------------------------------------
def _require_run_dir() -> None:
    if not RUN_DIR.is_dir():
        pytest.skip(
            f"the sealed R32 run dir is absent on this checkout (phase "
            f"2 has not landed): {RUN_DIR}"
        )


def _load(name: str):
    return json.loads((RUN_DIR / name).read_text(encoding="utf-8"))


def test_sealed_run_dir_inventory():
    """The sealed run dir holds exactly the pre-registered deliverables."""
    _require_run_dir()
    expected = {
        "pre_registration.json",
        "provenance.json",
        "bit_identity_v0.json",
        "spread_report.json",
        "gate3_r32.json",
        "clean_environment_commands.json",
        "seal_manifest.json",
        "SHA256SUMS.txt",
    }
    expected |= {
        f"variant_{v}_replica_{r}.json"
        for v in range(3)
        for r in range(8)
    }
    actual = {p.name for p in RUN_DIR.iterdir() if p.is_file()}
    assert actual == expected, (
        "the sealed run dir inventory differs: "
        f"missing {sorted(expected - actual)}, "
        f"unexpected {sorted(actual - expected)}"
    )
    # No smoke artifacts in the sealed dir (the R31/8c refusal
    # convention).
    assert not [n for n in actual if n.startswith("smoke_")]


def test_seal_integrity():
    """The seal pair: root-relative sums over the payloads, and the
    seal manifest recording every other file."""
    _require_run_dir()
    sums_path = RUN_DIR / "SHA256SUMS.txt"
    lines = [
        line for line in
        sums_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    entries = {}
    for line in lines:
        digest, name = line.split("  ", 1)
        entries[name] = digest
    # The sums file and the seal manifest are excluded from their own
    # inventory; every other payload file is covered (the sums keys are
    # the monorepo-root-relative paths; the coverage check strips the
    # run-dir prefix).  ``clean_environment_commands.json`` is the
    # documented post-seal record (the R31 convention): written after
    # the seal and excluded from it.
    payloads = {
        p.name for p in RUN_DIR.iterdir() if p.is_file()
    } - {"SHA256SUMS.txt", "seal_manifest.json"}
    sealed_payloads = payloads - {"clean_environment_commands.json"}
    prefix = f"dev/00runs/{RUN_NAME}/"
    assert {
        name[len(prefix):] for name in entries
    } == sealed_payloads
    # The entries carry monorepo-root-relative paths and verify.
    for name, digest in entries.items():
        assert name.startswith(f"dev/00runs/{RUN_NAME}/")
        payload = MONO_ROOT / name
        assert payload.is_file() and _sha256(payload) == digest
    # The sums inventory is sorted by path.
    assert list(entries) == sorted(entries)
    seal = _load("seal_manifest.json")
    sealed = {e["path"] for e in seal["files"]}
    assert sealed == sealed_payloads | {"SHA256SUMS.txt"}
    assert seal["sha256_sums_sha256"] == _sha256(sums_path)
    assert seal["file_count"] == len(sealed)


def test_pre_registration_invariants():
    """The sealed pre-registration carries the card's contract."""
    _require_run_dir()
    pre = _load("pre_registration.json")
    assert pre["run"] == "R32"
    assert pre["card"] == (
        "TASK-20260918-r3-run-r32-spectral-systematic-pp"
    )
    assert pre["written_before_generation"] is True
    assert pre["no_fitting"] is True
    assert pre["optimizer_invoked"] is False
    assert [
        (int(v["variant"]), float(v["ramp_start_gev"]),
         float(v["ramp_width_gev"]))
        for v in pre["variants"]
    ] == [
        (0, 2.680194110727528, 0.5),
        (1, 2.480194110727528, 0.5),
        (2, 2.880194110727528, 0.5),
    ]
    assert pre["statistics"]["replicas_per_variant"] == 8
    assert pre["statistics"]["events_per_replica"] == 100_000
    assert pre["statistics"]["total_events"] == 2_400_000
    assert pre["seed_map"]["master_seed"] == 20260919
    assert pre["seed_map"]["raw_replica_seeds"]["0"] == list(
        range(20301519, 20301527)
    )
    assert pre["seed_map"]["raw_replica_seeds"]["1"] == list(
        range(20301619, 20301627)
    )
    assert pre["seed_map"]["raw_replica_seeds"]["2"] == list(
        range(20301719, 20301727)
    )
    assert pre["bit_identity_check"]["seed"] == 20300913
    assert pre["bit_identity_check"]["n_events"] == 10_000
    confirmation = pre["robust_statistics_cross_reference"]
    assert confirmation["raw_replica_seeds"]["0"] == list(
        range(20301819, 20301835)
    )
    assert confirmation["raw_replica_seeds"]["1"] == list(
        range(20301919, 20301935)
    )
    assert confirmation["raw_replica_seeds"]["2"] == list(
        range(20302019, 20302035)
    )
    spread = pre["cross_variant_spread"]
    assert spread["bin_spread_bound"] == BIN_SPREAD_BOUND
    assert spread["feeddown_spread_bound"] == FEEDDOWN_SPREAD_BOUND


def test_provenance_hashes():
    """The sealed provenance carries the driver (both trees), the
    engine, the pre/post-extension builder, the catalog and continuum
    config, and the verified seal states."""
    _require_run_dir()
    provenance = _load("provenance.json")
    driver_block = provenance["driver_sha256"]
    assert driver_block["nested"] == driver_block["outer"] == DRIVER_SHA256
    assert provenance["engine"]["sha256"] == ENGINE_SHA256
    builder = provenance["builder"]
    assert builder["pre_extension_r23_sha256"] == (
        BUILDER_R23_PRE_EXTENSION_SHA256
    )
    assert builder["post_extension_sha256"] == (
        BUILDER_POST_EXTENSION_SHA256
    )
    assert provenance["catalog"]["sha256"] == CATALOG_SHA256
    assert provenance["continuum_config"]["sha256"] == FAMILIES_SHA256
    assert provenance["frozen_drivers"]["r31_driver_sha256"] == (
        R31_DRIVER_SHA256
    )
    sealed_dirs = {
        entry["path"]: entry
        for entry in provenance["sealed_run_dirs"]
    }
    assert len(sealed_dirs) == len(PRIOR_SEALED_DIRS)
    for rel in PRIOR_SEALED_DIRS:
        assert sealed_dirs[f"dev/00runs/{rel}"]["verified"] is True


def test_bit_identity_evidence():
    """The sealed bit-identity report: the three source layers
    bit-identical over 10,000 events at the R23 replica-0 seed; the
    finals layers recorded."""
    _require_run_dir()
    report = _load("bit_identity_v0.json")
    assert report["pass"] is True
    assert report["variant"] == 0
    assert report["seed"] == 20300913
    assert report["n_events"] == 10_000
    for name in ("legacy_vs_builder", "builder_vs_v0"):
        for key in ("topology", "exchange", "spectral_state"):
            slot = report["asserted_layers"][name][key]
            assert slot["n_matched"] == slot["n_compared"] == 10_000
            assert slot["mismatches"] == []
    census = report["finals_census"]
    assert census["below_boundary_finals_compared"] == (
        census["below_boundary_finals_matched"]
    )
    assert census["legacy_finals_compared"] == census[
        "legacy_finals_matched"
    ]
    assert census["v0_continuum_routed"] > 0
    assert census["v0_catalog_routed"] > 0


def test_gate_clause_verdicts():
    """The sealed gate evidence: the clause shapes and verdicts, the
    named-residual register, and the cross-references."""
    _require_run_dir()
    gate = _load("gate3_r32.json")
    assert gate["gate"] == "gate 3 (plan section 9, line 650)"
    assert gate["run"] == "R32"
    assert gate["no_fitting"] is True
    assert gate["optimizer_invoked"] is False
    assert gate["sealed"] is True
    assert gate["events_checked"] == 2_400_000
    clauses = gate["clauses"]
    smoothness = clauses["smoothness"]
    assert smoothness["p_plus_p_consistency"] in ("pass", "fail")
    assert smoothness["r31_cross_reference"]["artifact"] == (
        "dev/00runs/r31_catalog_continuum_matching_scan_2026-09-17/"
        "gate3_r31.json"
    )
    assert "0/422" in smoothness["r31_cross_reference"][
        "card_8c_authoritative"
    ]
    dominance = clauses["no_boundary_dominance"]
    assert dominance["variants"] == 3
    assert dominance["events_checked"] == 2_400_000
    assert dominance["bit_identity_v0_vs_r23"] == {
        "layers": ["topology", "exchange", "spectral_state"],
        "events": 10_000,
        "pass": True,
    }
    assert dominance["bin_spread_bound"] == BIN_SPREAD_BOUND
    assert dominance["feeddown_spread_bound"] == FEEDDOWN_SPREAD_BOUND
    assert set(dominance["routing_census"]) == {"v0", "v1", "v2"}
    # The register entries follow the gate3_r31.json shape.
    for entry in gate["named_residual_register"]:
        assert {
            "index", "kind", "species", "variant_pair",
            "observed_spread", "sigma", "bound", "classification",
            "diagnosis",
        } <= set(entry)
        assert entry["classification"] in (
            "finite_mc_scatter", "systematic_candidate",
            "se_estimator_artifact",
        )
    # The confirmation cross-reference: absent unless triggered.
    if gate["confirmation_cross_reference"] is not None:
        assert "triggered" in gate["confirmation_cross_reference"]


def test_clean_environment_record():
    """The post-seal record: the seal re-verified and the clean
    interpreter reproduced the spread report byte for byte."""
    _require_run_dir()
    record = _load("clean_environment_commands.json")
    assert record["seal_verification"]["mismatches"] == []
    assert record["probe_returncode"] == 0
    assert record["hashes_match"] is True
    report_sha = record["sealed_report_sha256"]["spread_report.json"]
    assert report_sha == record["recomputed_report_sha256"][
        "spread_report.json"
    ]
    assert report_sha == _sha256(RUN_DIR / "spread_report.json")
    assert record["probe_tool"]["sha256"] == DRIVER_SHA256
