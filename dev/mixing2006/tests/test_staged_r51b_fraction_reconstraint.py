"""Register of the R51b topology fraction re-constraint
(TASK-20260928-r5-gate5-followup-r51b-fraction-reconstraint).

Hermetic part: the driver's identity constants against the seed ledger
and the card, the dual-tree byte identity, and a reduced smoke run of the
driver into a temporary directory (4 classes x 120 events) that exercises
generation, the compact records, and the fit machinery.  Archive part
(skipped when the sealed run dir is absent): the pre-registration,
the seal inventory, and the fit report's verdict fields.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests._sealed_archive import sealed_mismatches


#: The sealed driver's per-event mass-shell check divides the deviation by
#: M^2, and the kernel's radiative channels (2026-10-02,
#: TASK-20261002-kernel-feedover-threshold-radiative) put massless photons
#: in the final state, so the driver's hermetic generation cannot pass on
#: the current kernel.  The sealed archive tests stay active; the current
#: pipeline is the R71 register's.
_SEALED_KERNEL_SKIP = pytest.mark.skip(
    reason="the sealed driver predates the radiative channels (massless photons) of the current kernel")

NESTED_ROOT = Path(__file__).resolve().parents[1]
MONO_ROOT = NESTED_ROOT.parents[1] if NESTED_ROOT.parent.name == "dev" else NESTED_ROOT
DRIVER_REL = Path("helpers") / "benchmarking" / "staged_r51b_fraction_reconstraint.py"
NESTED_DRIVER = NESTED_ROOT / DRIVER_REL
OUTER_DRIVER = MONO_ROOT / DRIVER_REL
RUN_DIR = MONO_ROOT / "dev" / "00runs" / "r51b_topology_fraction_reconstraint_2026-09-28"


def _load_driver():
    spec = importlib.util.spec_from_file_location("staged_r51b", NESTED_DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    return _load_driver()


def test_dual_tree_byte_identity():
    if not OUTER_DRIVER.is_file() or OUTER_DRIVER.resolve() == NESTED_DRIVER.resolve():
        pytest.skip("monorepo mirror not present")
    assert NESTED_DRIVER.read_bytes() == OUTER_DRIVER.read_bytes()
    for name in ("run_contract.py", "run_observables.py"):
        assert (NESTED_ROOT / "helpers" / "benchmarking" / name).read_bytes() == (
            MONO_ROOT / "helpers" / "benchmarking" / name).read_bytes()


def test_identity_constants_match_the_ledger(driver):
    ledger = json.loads((NESTED_ROOT / "config" / "v2_singlet" / "seed_ledger.json").read_text())
    entry = next(b for b in ledger["blocks"] if b["run"] == driver.LEDGER_RUN)
    assert (entry["raw_first"], entry["raw_last"]) == (driver.RAW_FIRST, driver.RAW_LAST)
    assert entry["partition_base"] == driver.PARTITION_BASE
    assert entry["master_seed"] == driver.MASTER_SEED == 20260928
    assert driver.RAW_LAST - driver.RAW_FIRST + 1 == driver.N_SLOTS * driver.MAX_GENERATIONS == 128
    assert driver.CLASS_ORDER == ("sd_projectile", "sd_target", "dd", "nd")
    assert (driver.REPLICAS, driver.EVENTS_PER_SLOT) == (8, 100_000)
    assert (driver.ESS_FLOOR, driver.ESS_INTERIOR, driver.NORM_PRIOR) == (0.1, 0.3, 0.02)
    assert driver.slot_seed(0, 0) == 20303428 and driver.slot_seed(3, 31) == 20303555
    smoke = next(b for b in ledger["blocks"] if b["run"] == "round6_smoke")
    assert smoke["raw_first"] <= driver.SMOKE_SEED <= smoke["raw_last"]
    assert smoke["partition_base"] == driver.SMOKE_PARTITION


def test_anchors_read_from_the_provenance_file(driver):
    a = driver.anchors()
    ext = json.loads((NESTED_ROOT / "config" / "v2_singlet" / "external_cross_sections.json").read_text())
    f = ext["fractions_of_inelastic"]
    assert a["f_sd_total"]["value"] == f["f_sd_total"]["value"]
    assert a["f_dd"]["sigma"] == f["f_dd"]["uncertainty"]
    assert abs(f["f_sd_total"]["value"] + f["f_dd"]["value"] + f["f_nd"]["value"] - 1.0) < 1e-9


@_SEALED_KERNEL_SKIP
def test_reduced_smoke_run(driver, tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "SMOKE_EVENTS_PER_CLASS", 120)
    # the driver is loaded by path here, so its worker cannot be pickled
    # into a process pool; the smoke runs its four slots serially
    monkeypatch.setattr(driver.RC, "run_replicas",
                        lambda worker, jobs, max_workers=None: [worker(*job) for job in jobs])
    driver.step_smoke(tmp_path / "r51b_smoke")
    report = json.loads((tmp_path / "r51b_smoke" / "smoke_fit_report.json").read_text())
    fitted = report["fitted"]
    assert abs(fitted["f_sd_total"] + fitted["f_dd"] + fitted["f_nd"] - 1.0) < 1e-9
    assert report["objective_at_optimum"]["ess_min"] > 0.0
    assert report["observable_set"]["total"] == 284
    for cls in driver.CLASS_ORDER:
        meta = json.loads((tmp_path / "r51b_smoke" / f"smoke_slot_{cls}_0_g0.json").read_text())
        assert meta["n_events"] == 120 and meta["conservation"]["violations"] == 0
        with np.load(tmp_path / "r51b_smoke" / f"smoke_slot_{cls}_0_g0.npz") as data:
            arrays = {k: data[k] for k in data.files}
        assert driver.arrays_digest(arrays) == meta["arrays_sha256"]
        assert arrays["n_ch"].size == 120


@_SEALED_KERNEL_SKIP
def test_generation_is_deterministic(driver):
    a = driver.generate_slot("nd", driver.SMOKE_SEED, 30, driver.SMOKE_PARTITION, driver.NOMINAL_EPSILON, driver.NOMINAL_B_SD)
    b = driver.generate_slot("nd", driver.SMOKE_SEED, 30, driver.SMOKE_PARTITION, driver.NOMINAL_EPSILON, driver.NOMINAL_B_SD)
    assert a["event_rolling_sha256"] == b["event_rolling_sha256"]
    assert driver.arrays_digest(a["arrays"]) == driver.arrays_digest(b["arrays"])


@pytest.mark.skipif(not (RUN_DIR / "seal_manifest.json").is_file(), reason="the sealed R51b run dir is not present")
def test_sealed_archive_on_disk(driver):
    pre = json.loads((RUN_DIR / "pre_registration.json").read_text())
    assert pre["run"] == driver.RUN_ID and pre["card"] == driver.CARD
    assert pre["seeds"]["raw_first"] == driver.RAW_FIRST
    report = json.loads((RUN_DIR / "report.json").read_text())
    assert set(report["fitted"]) >= {"f_sd_total", "f_dd", "f_nd", "epsilon", "b_sd"}
    assert "constrained" in report["verdict"]
    gate = json.loads((RUN_DIR / driver.GATE_RECORD_NAME).read_text())
    assert gate["verdict"] in ("CONSTRAINED", "NOT_CONSTRAINED")
    assert sealed_mismatches(RUN_DIR) == []
    assert (RUN_DIR / "seal_manifest.json").is_file()
