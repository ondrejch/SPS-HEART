"""Register of the R51i fine floor scan of the joint proton and pion
topology fit (TASK-20261001-r5-gate5-followup-r51i-fine-floor-scan).

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
DRIVER_REL = Path("helpers") / "benchmarking" / "staged_r51i_joint_fit.py"
NESTED_DRIVER = NESTED_ROOT / DRIVER_REL
OUTER_DRIVER = MONO_ROOT / DRIVER_REL
RUN_DIR = MONO_ROOT / "dev" / "00runs" / "r51i_joint_fit_2026-10-01"


def _load_driver():
    spec = importlib.util.spec_from_file_location("staged_r51i", NESTED_DRIVER)
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
    assert (entry["raw_first"], entry["raw_last"]) == (driver.RAW_FIRST, driver.RAW_LAST) == (20314200, 20314711)
    assert entry["partition_base"] == driver.PARTITION_BASE == 3_600_000_000
    assert entry["master_seed"] == driver.MASTER_SEED == 20261001
    assert driver.FLOOR_GRID == (1.6, 1.7, 1.8, 1.9)
    assert driver.RAW_LAST - driver.RAW_FIRST + 1 == driver.MAX_GENERATIONS * driver.N_SLOTS == 512
    assert driver.slot_seed(0, 0, 0) == 20314200 and driver.slot_seed(3, 3, 127) == 20314711
    # common random numbers: the floor does not enter the seed
    assert all(driver.slot_seed(f, g, s) == driver.slot_seed(0, g, s) for f in range(4) for g in range(4) for s in range(128))
    assert len({driver.slot_seed(0, g, s) for g in range(4) for s in range(128)}) == 512
    assert driver.CLASS_ORDER == ("sd_projectile", "sd_target", "dd", "nd")
    assert (driver.REPLICAS, driver.EVENTS_PER_SLOT) == (32, 100_000)
    assert (driver.BLOCK_WORKERS_PER_START, driver.BLOCK_WORKERS_PER_VARIANT_START, driver.MASTER_BLOCK_WORKERS) == (6, 12, 32)
    assert (driver.ESS_FLOOR, driver.ESS_INTERIOR, driver.NORM_PRIOR) == (0.1, 0.3, 0.02)
    assert (driver.DISCREPANCY, driver.DISCREPANCY_VARIANT) == (0.05, 0.10)
    assert driver.SHAPE_NAMES == ("epsilon", "b_sd", "w_excited", "epsilon_nd")
    assert driver.ND_GENERATION_REMNANT_WEIGHTS == {"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.5,
                                                    "high_mass_baryonic_continuum": 0.5}
    assert (driver.GEN0_EPSILON, driver.GEN0_B_SD, driver.GEN0_EPSILON_ND) == (
        0.49815418511182347, 2.4114630077319634, 0.3294084567271025)
    assert driver.EPS_BOUNDS == (-0.5, 1.5)
    smoke = next(b for b in ledger["blocks"] if b["run"] == "round7_smoke")
    assert smoke["raw_first"] <= driver.SMOKE_SEED <= smoke["raw_last"]
    assert driver.SMOKE_PARTITION >= smoke["partition_base"]
    assert "helpers/benchmarking/staged_r51h_joint_fit.py" in driver.FROZEN_RELS
    assert "dev/mixing2006/spshadrons.web.cern.ch/data/xsect_ptintegrated_neutron_pp.txt" in driver.FROZEN_RELS
    assert "dev/mixing2006/config/v2_singlet/continuum_resonances.json" in driver.FROZEN_RELS
    assert driver.WEIGHT_CACHE_ENTRIES == 64 and driver.SE_CACHE_ENTRIES == 512


def test_generation_zero_matches_the_sealed_r51h_optimum(driver):
    r51h = MONO_ROOT / "dev" / "00runs" / driver.R51H_RUN_ID / "report.json"
    if not r51h.is_file():
        pytest.skip("the sealed R51h archive is not present")
    fitted = json.loads(r51h.read_text())["fitted"]
    assert fitted["epsilon"] == driver.GEN0_EPSILON and fitted["b_sd"] == driver.GEN0_B_SD
    assert fitted["epsilon_nd"] == driver.GEN0_EPSILON_ND
    assert fitted["w_excited"] == driver.R51H_W_EXCITED and fitted["w_low_mass"] == driver.R51H_W_LOW


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
    driver.step_smoke(tmp_path / "r51i_smoke")
    report = json.loads((tmp_path / "r51i_smoke" / "smoke_fit_report.json").read_text())
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
    assert report["readout_pi0"]["pi0_per_event"] > 0 and 0 < report["readout_pi0"]["neutral_share"] < 1
    assert [r["xf"] for r in report["readout_neutron_dndxf"]["rows"]] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9]
    for cls in driver.CLASS_ORDER:
        name = "smoke_" + driver.SLOT_JSON_FMT.format(cls=cls, k=0, f=driver.SMOKE_FLOOR_INDEX, g=0)
        meta = json.loads((tmp_path / "r51i_smoke" / name).read_text())
        assert meta["continuation_floor"] == driver.FLOOR_GRID[driver.SMOKE_FLOOR_INDEX]
        assert meta["n_events"] == 120 and meta["conservation"]["violations"] == 0
        with np.load(tmp_path / "r51i_smoke" / name.replace(".json", ".npz")) as data:
            arrays = {k: data[k] for k in data.files}
        assert driver.arrays_digest(arrays) == meta["arrays_sha256"]
        assert arrays["n_pi0"].size == 120 and arrays["neu_event"].size == arrays["neu_xf"].size
        if cls == "nd":
            assert set(np.unique(arrays["rem_kind"]).tolist()) <= {1, 2}
        else:
            assert np.all(arrays["rem_kind"] == -1)
    # the forked block pool reproduces the in-process objective
    check = report["pool_check"]
    assert math.isfinite(check["in_process"]) and check["relative_difference"] < 1e-12
    _check_pooled_against_in_process(driver, tmp_path / "r51i_smoke")


class _InProcessPool:
    """A block pool without processes: the same replies as the forked
    workers, from one in-process objective over all blocks."""

    def __init__(self, driver, objective):
        self.driver, self.objective = driver, objective
        self.n_blocks, self.point = objective.t.n_blocks, objective.t.point

    def unit(self):
        return self.driver.block_reply(self.objective, ("unit",))[1]

    def evaluate(self, eps, b_sd, eps_nd, remnant):
        _, obs, ess = self.driver.block_reply(self.objective, ("eval", eps, b_sd, eps_nd, tuple(remnant)))
        return obs, ess


def _check_pooled_against_in_process(driver, run_dir):
    """Three replica blocks (the smoke block and two with perturbed track
    weights): the pooled combination equals Objective.evaluate, primary and
    free-fraction parametrizations, value and per-bin details."""

    contract = driver._contract(run_dir)
    tables = driver.load_fit_tables()
    templates = driver.Templates(contract, driver.SMOKE_FLOOR_INDEX, 0, tables, smoke=True)
    rng = np.random.default_rng(7)
    for cls in driver.CLASS_ORDER:
        base = templates.by_class[cls][0]
        for _ in range(2):
            block = dict(base)
            block["binned"] = {name: (ev, pos, w * rng.uniform(0.5, 1.5, w.size), nb) for name, (ev, pos, w, nb) in base["binned"].items()}
            templates.by_class[cls].append(block)
    templates.n_blocks = 3
    a = driver.anchors()
    for free in (False, True):
        serial = driver.Objective(templates, a, free=free)
        pooled = driver.PooledObjective(tables, a, _InProcessPool(driver, driver.Objective(templates, a)), free=free)
        for shift in (0.0, 0.05, -0.08):
            phys = {"f_sd_total": 0.19, "f_dd": 0.06, "epsilon": driver.GEN0_EPSILON + shift, "b_sd": driver.GEN0_B_SD * (1 + shift),
                    "w_excited": 0.9 - shift, "epsilon_nd": driver.GEN0_EPSILON_ND - shift}
            x = driver.x_from_physical(phys, free)
            v_s, d_s = serial.evaluate(x, details=True)
            v_p, d_p = pooled.evaluate(x, details=True)
            assert math.isclose(v_p, v_s, rel_tol=1e-10)
            assert np.allclose(d_p["m"], d_s["m"], rtol=1e-10, atol=0.0)
            assert np.allclose(d_p["se"], d_s["se"], rtol=1e-10, atol=0.0)
            assert math.isclose(d_p["eta_hat"], d_s["eta_hat"], rel_tol=1e-9, abs_tol=1e-12)
            assert d_p["ess"] == d_s["ess"] and d_p["blocks"] == d_s["blocks"] == [0, 1, 2]
            assert np.any(d_s["se"] > driver.SE_FLOOR)


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


@pytest.mark.skipif(not (RUN_DIR / "seal_manifest.json").is_file(), reason="the sealed R51i run dir is not present")
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
