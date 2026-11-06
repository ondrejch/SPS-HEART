"""Register of the R51f joint proton and pion topology fit with the cascade
continuation floor on a grid (TASK-20260930-r5-gate5-followup-r51f-joint-proton-pion-fit).

Hermetic part: the driver's identity constants against the seed ledger and
the card, the dual-tree byte identity, the fixed-intact remnant
parametrization, the in-memory floor, and a reduced smoke run of the
driver into a temporary directory (4 classes x 120 events) that exercises
generation at a floor, the joint objective, and the held-out readouts.
Archive part (skipped when the sealed run dir is absent): the
pre-registration, the floor profile, the seal inventory, and the verdict
and acceptance fields.
"""

from __future__ import annotations

import importlib.util
import json
import math
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
DRIVER_REL = Path("helpers") / "benchmarking" / "staged_r51f_joint_fit.py"
NESTED_DRIVER = NESTED_ROOT / DRIVER_REL
OUTER_DRIVER = MONO_ROOT / DRIVER_REL
RUN_DIR = MONO_ROOT / "dev" / "00runs" / "r51f_joint_fit_2026-09-30"


def _load_driver():
    spec = importlib.util.spec_from_file_location("staged_r51f", NESTED_DRIVER)
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
    assert (entry["raw_first"], entry["raw_last"]) == (driver.RAW_FIRST, driver.RAW_LAST) == (20313000, 20313383)
    assert entry["partition_base"] == driver.PARTITION_BASE == 3_300_000_000
    assert entry["master_seed"] == driver.MASTER_SEED == 20260930
    assert driver.FLOOR_GRID == (2.0, 1.7, 1.4)
    assert driver.RAW_LAST - driver.RAW_FIRST + 1 == driver.N_FLOORS * driver.MAX_GENERATIONS * driver.N_SLOTS == 384
    assert driver.slot_seed(0, 0, 0) == 20313000 and driver.slot_seed(2, 3, 31) == 20313383
    assert len({driver.slot_seed(f, g, s) for f in range(3) for g in range(4) for s in range(32)}) == 384
    assert driver.CLASS_ORDER == ("sd_projectile", "sd_target", "dd", "nd")
    assert (driver.REPLICAS, driver.EVENTS_PER_SLOT) == (8, 100_000)
    assert (driver.ESS_FLOOR, driver.ESS_INTERIOR, driver.NORM_PRIOR) == (0.1, 0.3, 0.02)
    assert (driver.DISCREPANCY, driver.DISCREPANCY_VARIANT) == (0.05, 0.10)
    assert driver.SHAPE_NAMES == ("epsilon", "b_sd", "w_excited", "epsilon_nd")
    assert driver.ND_GENERATION_REMNANT_WEIGHTS == {"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.5,
                                                    "high_mass_baryonic_continuum": 0.5}
    assert (driver.GEN0_EPSILON, driver.GEN0_B_SD, driver.GEN0_EPSILON_ND) == (
        0.32945646983770666, 3.1648568299322037, -0.09945422217560174)
    smoke = next(b for b in ledger["blocks"] if b["run"] == "round7_smoke")
    assert smoke["raw_first"] <= driver.SMOKE_SEED <= smoke["raw_last"]
    assert driver.SMOKE_PARTITION >= smoke["partition_base"]
    assert "helpers/benchmarking/staged_r51e_topology_fit.py" in driver.FROZEN_RELS
    assert "dev/mixing2006/config/v2_singlet/continuum_resonances.json" in driver.FROZEN_RELS
    assert driver.WEIGHT_CACHE_ENTRIES == 64 and driver.SE_CACHE_ENTRIES == 512


def test_generation_zero_matches_the_sealed_r51e_optimum(driver):
    r51e = MONO_ROOT / "dev" / "00runs" / driver.R51E_RUN_ID / "report.json"
    if not r51e.is_file():
        pytest.skip("the sealed R51e archive is not present")
    fitted = json.loads(r51e.read_text())["fitted"]
    assert fitted["epsilon"] == driver.GEN0_EPSILON and fitted["b_sd"] == driver.GEN0_B_SD
    assert fitted["epsilon_nd"] == driver.GEN0_EPSILON_ND
    assert fitted["w_excited"] == driver.R51E_W_EXCITED and fitted["w_low_mass"] == driver.R51E_W_LOW


def test_fixed_intact_parametrization_round_trip(driver):
    a = driver.anchors()
    for free in (False, True):
        phys = {"f_sd_total": 0.2, "f_dd": 0.05, "epsilon": 0.1, "b_sd": 4.0, "w_excited": 0.63, "epsilon_nd": -0.4}
        back = driver.physical_from_x(driver.x_from_physical(phys, free), free, a)
        assert back["w_intact"] == 0.0
        assert math.isclose(back["w_excited"], 0.63, rel_tol=1e-12)
        assert math.isclose(back["w_low_mass"] + back["w_excited"], 1.0, rel_tol=1e-12)
        assert math.isclose(back["b_sd"], 4.0, rel_tol=1e-12)
        if free:
            assert math.isclose(back["f_sd_total"], 0.2, rel_tol=1e-9) and math.isclose(back["f_dd"], 0.05, rel_tol=1e-9)
        else:
            assert back["f_sd_total"] == a["f_sd_total"]["value"]
    assert len(driver.bounds_for(False)) == 4 and len(driver.bounds_for(True)) == 6


def test_floor_is_set_in_memory(driver):
    _, fams = driver._load_physics(1.7)
    assert all(f.continuation_floor == 1.7 for f in fams)
    committed = driver.families_from_json(str(NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json"))
    assert all(f.continuation_floor is None for f in committed)


@_SEALED_KERNEL_SKIP
def test_reduced_smoke_run(driver, tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "SMOKE_EVENTS_PER_CLASS", 120)
    monkeypatch.setattr(driver, "SMOKE_MESONIC_DECAYS", 20)
    # the driver is loaded by path, so its worker cannot be pickled into a
    # process pool; the smoke runs its four slots serially
    monkeypatch.setattr(driver.RC, "run_replicas",
                        lambda worker, jobs, max_workers=None: [worker(*job) for job in jobs])
    driver.step_smoke(tmp_path / "r51f_smoke")
    report = json.loads((tmp_path / "r51f_smoke" / "smoke_fit_report.json").read_text())
    segments = report["observable_set"]["segments"]
    assert report["observable_set"]["total"] == 826
    assert {k: v[1] - v[0] for k, v in segments.items()} == {"net_proton": 122, "proton_only": 162, "pi_minus": 282, "pi_plus": 260}
    assert set(report["objective_at_optimum"]["chi2_by_segment"]) == set(segments)
    fitted = report["fitted"]
    assert fitted["w_intact"] == 0.0 and abs(fitted["w_low_mass"] + fitted["w_excited"] - 1.0) < 1e-12
    assert report["continuation_floor"] == driver.FLOOR_GRID[driver.SMOKE_FLOOR_INDEX]
    rows = report["readout_mesonic_count"]["rows"]
    assert [r["mass"] for r in rows] == list(driver.MESONIC_MASSES)
    assert all(r["model"] > 0 and math.isclose(r["mark_i"], driver.MARK_I["a"] + driver.MARK_I["b"] * math.log(r["mass"] ** 2)) for r in rows)
    assert math.isfinite(report["readout_pt_profile"]["profile_ratio"])
    assert set(report["acceptance_block_b_stage2"]["criteria"]) == {"pion_total_yields", "mesonic_count", "pt_profile"}
    for cls in driver.CLASS_ORDER:
        name = "smoke_" + driver.SLOT_JSON_FMT.format(cls=cls, k=0, f=driver.SMOKE_FLOOR_INDEX, g=0)
        meta = json.loads((tmp_path / "r51f_smoke" / name).read_text())
        assert meta["continuation_floor"] == driver.FLOOR_GRID[driver.SMOKE_FLOOR_INDEX]
        assert meta["n_events"] == 120 and meta["conservation"]["violations"] == 0
        with np.load(tmp_path / "r51f_smoke" / name.replace(".json", ".npz")) as data:
            arrays = {k: data[k] for k in data.files}
        assert driver.arrays_digest(arrays) == meta["arrays_sha256"]
        if cls == "nd":
            assert set(np.unique(arrays["rem_kind"]).tolist()) <= {1, 2}
        else:
            assert np.all(arrays["rem_kind"] == -1)


@_SEALED_KERNEL_SKIP
def test_generation_is_deterministic_and_floor_dependent(driver):
    a = driver.generate_slot("nd", driver.SMOKE_SEED, 25, driver.SMOKE_PARTITION, driver.GEN0_EPSILON, driver.GEN0_B_SD, -0.5, 1.7)
    b = driver.generate_slot("nd", driver.SMOKE_SEED, 25, driver.SMOKE_PARTITION, driver.GEN0_EPSILON, driver.GEN0_B_SD, -0.5, 1.7)
    c = driver.generate_slot("nd", driver.SMOKE_SEED, 25, driver.SMOKE_PARTITION, driver.GEN0_EPSILON, driver.GEN0_B_SD, -0.5, 2.0)
    assert a["event_rolling_sha256"] == b["event_rolling_sha256"]
    assert driver.arrays_digest(a["arrays"]) == driver.arrays_digest(b["arrays"])
    assert a["event_rolling_sha256"] != c["event_rolling_sha256"]
    # the mass-law arrays depend on the exchange draws only, not on the floor
    assert np.array_equal(a["arrays"]["shape_k1"], c["arrays"]["shape_k1"])


@pytest.mark.skipif(not (RUN_DIR / "seal_manifest.json").is_file(), reason="the sealed R51f run dir is not present")
def test_sealed_archive_on_disk(driver):
    pre = json.loads((RUN_DIR / "pre_registration.json").read_text())
    assert pre["run"] == driver.RUN_ID and pre["card"] == driver.CARD
    assert pre["seeds"]["raw_first"] == driver.RAW_FIRST and pre["floor_grid"]["values_gev"] == list(driver.FLOOR_GRID)
    report = json.loads((RUN_DIR / "report.json").read_text())
    assert [r["continuation_floor"] for r in report["floor_profile"]["rows"]] == list(driver.FLOOR_GRID)
    assert report["fitted"]["w_intact"] == 0.0
    assert "acceptance_block_b_stage2" in report and "discrepancy_variant" in report
    gate = json.loads((RUN_DIR / driver.GATE_RECORD_NAME).read_text())
    assert gate["verdict"] in ("CONSTRAINED", "NOT_CONSTRAINED")
    assert sealed_mismatches(RUN_DIR) == []


def test_bounded_objective_caches(driver):
    text = NESTED_DRIVER.read_text()
    src = text[text.index("class Objective"):]
    assert "while len(self.cache) > WEIGHT_CACHE_ENTRIES" in src
    assert "while len(self.se_cache) > SE_CACHE_ENTRIES" in src
