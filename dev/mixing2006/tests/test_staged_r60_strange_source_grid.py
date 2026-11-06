"""Register of the R60 strange source grid
(TASK-20260928-r6-run-r60-strange-source-grid): identity constants
against the seed ledger, dual-tree byte identity, a reduced serial smoke
run, determinism of a node, and the sealed archive when present."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

NESTED_ROOT = Path(__file__).resolve().parents[1]
MONO_ROOT = NESTED_ROOT.parents[1] if NESTED_ROOT.parent.name == "dev" else NESTED_ROOT
DRIVER_REL = Path("helpers") / "benchmarking" / "staged_r60_strange_source_grid.py"
RUN_DIR = MONO_ROOT / "dev" / "00runs" / "r60_strange_source_grid_2026-09-28"


@pytest.fixture(scope="module")
def driver():
    spec = importlib.util.spec_from_file_location("staged_r60", NESTED_ROOT / DRIVER_REL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dual_tree_byte_identity():
    outer = MONO_ROOT / DRIVER_REL
    if not outer.is_file() or outer.resolve() == (NESTED_ROOT / DRIVER_REL).resolve():
        pytest.skip("monorepo mirror not present")
    assert outer.read_bytes() == (NESTED_ROOT / DRIVER_REL).read_bytes()


def test_identity_constants(driver):
    ledger = json.loads((NESTED_ROOT / "config" / "v2_singlet" / "seed_ledger.json").read_text())
    entry = next(b for b in ledger["blocks"] if b["run"] == driver.LEDGER_RUN)
    assert (entry["raw_first"], entry["raw_last"], entry["partition_base"]) == (driver.RAW_FIRST, driver.RAW_LAST, driver.PARTITION_BASE)
    assert driver.N_JOBS == 120 and len(driver.MASS_NODES) == 30 and driver.MASS_NODES[0] == 1.05 and driver.MASS_NODES[-1] == 3.95
    assert driver.DECAYS_PER_NODE == 40_000
    assert driver.job_family(0) == "mesonic_neutral" and driver.job_family(119) == "baryonic_strange"
    assert driver.job_seed(119) == driver.RAW_LAST
    assert set(driver.FLAVOR_POINTS) >= {"identity", "gamma_s_0.5", "gamma_s_1.5", "r_v_0.5", "r_bbar_2.0"}


def test_reduced_smoke_run(driver, tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "SMOKE_DECAYS", 60)
    monkeypatch.setattr(driver, "SMOKE_NODES", (1.25, 2.55))
    monkeypatch.setattr(driver.RC, "run_replicas", lambda worker, jobs, max_workers=None: [worker(*job) for job in jobs])
    driver.step_smoke(tmp_path / "r60_smoke")
    report = json.loads((tmp_path / "r60_smoke" / "smoke_report.json").read_text())
    assert report["summary"]["nodes"] == 4
    assert set(report["grid"]) == set(driver.SMOKE_FAMILIES)
    node = json.loads((tmp_path / "r60_smoke" / "smoke_node_mesonic_neutral_2.55.json").read_text())
    assert node["conservation"]["violations"] == 0 and node["n_decays"] == 60
    assert abs(sum(node["flavor_response"]["identity"]["mean_multiplicity"][k] for k in ("k_plus",)) - node["mean_multiplicity"]["k_plus"]) < 1e-9


def test_node_is_deterministic(driver):
    a = driver.generate_node("mesonic_neutral", 2.05, driver.SMOKE_SEED, 25, driver.SMOKE_PARTITION)
    b = driver.generate_node("mesonic_neutral", 2.05, driver.SMOKE_SEED, 25, driver.SMOKE_PARTITION)
    assert a["event_rolling_sha256"] == b["event_rolling_sha256"]
    assert a["mean_multiplicity"] == b["mean_multiplicity"]


@pytest.mark.skipif(not (RUN_DIR / "seal_manifest.json").is_file(), reason="the sealed R60 run dir is not present")
def test_sealed_archive_on_disk(driver):
    report = json.loads((RUN_DIR / "report.json").read_text())
    assert report["summary"]["nodes"] == 120 and report["summary"]["decays_per_node"] == 40_000
    gate = json.loads((RUN_DIR / driver.GATE_RECORD_NAME).read_text())
    assert gate["summary"]["nodes"] == 120
    mismatches = [name for line in (RUN_DIR / "SHA256SUMS.txt").read_text().splitlines() if line.strip()
                  for digest, name in [line.split("  ", 1)] if hashlib.sha256((RUN_DIR / name).read_bytes()).hexdigest() != digest]
    assert mismatches == []
