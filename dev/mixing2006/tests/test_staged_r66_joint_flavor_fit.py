"""Register of the R66 joint flavor fit on the constrained R51i topology
point (TASK-20261002-r6-joint-flavor-fit-r66).

Hermetic part: the driver's identity constants against the seed ledger and
the card, the dual-tree byte identity, the log flavor parametrization, the
generator against the R51i generator at the identity flavor point (the
ledger consumes no random number), and a reduced smoke run of the driver
into a temporary directory (4 classes x 120 events) that exercises the
ledgered generation, the six-species objective, the key-table reweighting
against flavor_fit_core, the forked block pool, and the readouts.
Archive part (skipped when the sealed run dir is absent): the
pre-registration, the seal inventory, and the verdict and acceptance
fields.
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
DRIVER_REL = Path("helpers") / "benchmarking" / "staged_r66_joint_flavor_fit.py"
NESTED_DRIVER = NESTED_ROOT / DRIVER_REL
OUTER_DRIVER = MONO_ROOT / DRIVER_REL
RUN_DIR = MONO_ROOT / "dev" / "00runs" / "r66_joint_flavor_fit_2026-10-02"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    return _load("staged_r66", NESTED_DRIVER)


def test_dual_tree_byte_identity():
    if not OUTER_DRIVER.is_file() or OUTER_DRIVER.resolve() == NESTED_DRIVER.resolve():
        pytest.skip("monorepo mirror not present")
    assert NESTED_DRIVER.read_bytes() == OUTER_DRIVER.read_bytes()
    for name in ("run_contract.py", "run_observables.py", "flavor_fit_core.py", "orchestrate_gateway_joint_fit.py"):
        assert (NESTED_ROOT / "helpers" / "benchmarking" / name).read_bytes() == (
            MONO_ROOT / "helpers" / "benchmarking" / name).read_bytes()


def test_identity_constants_match_the_ledger(driver):
    ledger = json.loads((NESTED_ROOT / "config" / "v2_singlet" / "seed_ledger.json").read_text())
    entry = next(b for b in ledger["blocks"] if b["run"] == driver.LEDGER_RUN)
    assert (entry["raw_first"], entry["raw_last"]) == (driver.RAW_FIRST, driver.RAW_LAST) == (20314800, 20315311)
    assert entry["partition_base"] == driver.PARTITION_BASE == 3_700_000_000
    assert entry["master_seed"] == driver.MASTER_SEED == 20261002
    assert driver.FLOOR_GRID == (1.7,) and driver.N_FLOORS == 1
    assert driver.RAW_LAST - driver.RAW_FIRST + 1 == driver.MAX_GENERATIONS * driver.N_SLOTS == 512
    assert driver.slot_seed(0, 0, 0) == 20314800 and driver.slot_seed(0, 3, 127) == 20315311
    assert driver.CLASS_ORDER == ("sd_projectile", "sd_target", "dd", "nd")
    assert (driver.REPLICAS, driver.EVENTS_PER_SLOT) == (32, 100_000)
    assert (driver.BLOCK_WORKERS_PER_START, driver.BLOCK_WORKERS_PER_VARIANT_START, driver.MASTER_BLOCK_WORKERS) == (6, 12, 32)
    assert (driver.ESS_FLOOR, driver.ESS_INTERIOR) == (0.1, 0.3)
    assert (driver.DISCREPANCY, driver.DISCREPANCY_VARIANT) == (0.05, 0.10)
    assert (driver.NORM_FIXED, driver.NORM_VARIANTS) == (0.0, (0.02, -0.02))
    assert driver.FLAVOR_ORDER == ("gamma_s_had", "r_v", "r_bbar", "assoc_norm")
    assert driver.FLAVOR_FREE == ("gamma_s_had", "assoc_norm", "r_bbar") and driver.R_V_FIXED == 1.0
    assert driver.IDENTITY_FLAVOR == (1.0, 1.0, 1.0, 1.0)
    assert [round(math.exp(b), 12) for b in driver.FLAVOR_LOG_BOUNDS] == [0.05, 20.0]
    assert driver.FIT_SPECIES == ("proton", "antiproton", "pi_minus", "pi_plus", "k_plus", "k_minus")
    assert (driver.KAON_YIELD_BAND, driver.PBAR_YIELD_BAND) == ((0.85, 1.15), (0.75, 1.25))
    assert driver.ND_GENERATION_REMNANT_WEIGHTS == {"intact_proton": 0.0, "low_mass_baryonic_resonance": driver.FIXED_W_LOW,
                                                    "high_mass_baryonic_continuum": driver.FIXED_W_EXCITED}
    smoke = next(b for b in ledger["blocks"] if b["run"] == "round7_smoke")
    assert smoke["raw_first"] <= driver.SMOKE_SEED <= smoke["raw_last"]
    assert driver.SMOKE_PARTITION >= smoke["partition_base"]
    for rel in ("helpers/benchmarking/staged_r51i_joint_fit.py", "helpers/benchmarking/flavor_fit_core.py",
                "dev/mixing2006/na49_mixing/flavor_weights.py", "dev/mixing2006/spshadrons.web.cern.ch/data/xsect_kaplus_pp.txt",
                "dev/mixing2006/spshadrons.web.cern.ch/data/xsect_kaminus_pp.txt",
                "dev/mixing2006/spshadrons.web.cern.ch/data/xsect_antiproton_pp.txt"):
        assert rel in driver.FROZEN_RELS
    assert driver.WEIGHT_CACHE_ENTRIES == 64 and driver.SE_CACHE_ENTRIES == 512


def test_fixed_topology_matches_the_sealed_r51i_optimum(driver):
    r51i = MONO_ROOT / "dev" / "00runs" / driver.R51I_RUN_ID / "report.json"
    if not r51i.is_file():
        pytest.skip("the sealed R51i archive is not present")
    report = json.loads(r51i.read_text())
    assert report["verdict"]["constrained"] and report["continuation_floor"] == driver.FLOOR_GRID[0]
    assert all(report["fitted"][k] == v for k, v in driver.FIXED_TOPOLOGY.items())


def test_log_flavor_parametrization_round_trip(driver):
    a = driver.anchors()
    for free in (False, True):
        phys = {"f_sd_total": 0.2, "f_dd": 0.05, "gamma_s_had": 0.63, "assoc_norm": 1.7, "r_bbar": 4.2}
        x = driver.x_from_physical(phys, free)
        back = driver.physical_from_x(x, free, a)
        for name in driver.FLAVOR_FREE:
            assert math.isclose(back[name], phys[name], rel_tol=1e-12)
        assert back["r_v"] == 1.0 and back["w_intact"] == 0.0 and back["epsilon"] == driver.FIXED_EPSILON
        if free:
            assert math.isclose(back["f_sd_total"], 0.2, rel_tol=1e-9) and math.isclose(back["f_dd"], 0.05, rel_tol=1e-9)
        else:
            assert back["f_sd_total"] == a["f_sd_total"]["value"]
    assert len(driver.bounds_for(False)) == 3 and len(driver.bounds_for(True)) == 5
    assert not driver._in_bounds(np.asarray([0.0, 0.0, math.log(25.0)]), False)
    assert driver.flavor_of(driver.physical_from_x(np.zeros(3), False, a)) == (1.0, 1.0, 1.0, 1.0)


@_SEALED_KERNEL_SKIP
def test_generator_matches_r51i_at_the_identity_point(driver):
    """At the identity flavor point the ledgered generator draws the R51i
    events (single diffraction: the same remnant split), and a non-identity
    point changes them deterministically."""

    r51i = _load("staged_r51i_for_r66", NESTED_ROOT / "helpers" / "benchmarking" / "staged_r51i_joint_fit.py")
    args = ("sd_projectile", driver.SMOKE_SEED, 25, driver.SMOKE_PARTITION)
    mine = driver.generate_slot(*args, driver.IDENTITY_FLAVOR, 1.7)
    theirs = r51i.generate_slot(*args, driver.FIXED_EPSILON, driver.FIXED_B_SD, driver.FIXED_EPSILON_ND, 1.7)
    common = sorted(set(theirs["arrays"]) & set(mine["arrays"]))
    assert len(common) == len(theirs["arrays"])
    for key in common:
        assert np.array_equal(mine["arrays"][key], theirs["arrays"][key], equal_nan=True), key
    assert mine["informative_vertices"] > 0 and mine["arrays"]["led_gw"].size >= 2 * mine["informative_vertices"]
    again = driver.generate_slot(*args, (0.6, 1.0, 3.0, 0.8), 1.7)
    once_more = driver.generate_slot(*args, (0.6, 1.0, 3.0, 0.8), 1.7)
    assert again["event_rolling_sha256"] == once_more["event_rolling_sha256"]
    assert again["event_rolling_sha256"] != mine["event_rolling_sha256"]
    assert again["flavor_generation"] == {"gamma_s_had": 0.6, "r_v": 1.0, "r_bbar": 3.0, "assoc_norm": 0.8}


@_SEALED_KERNEL_SKIP
def test_reduced_smoke_run(driver, tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "SMOKE_EVENTS_PER_CLASS", 120)
    monkeypatch.setattr(driver, "SMOKE_MESONIC_DECAYS", 20)
    # the driver is loaded by path, so its worker cannot be pickled into a
    # process pool; the smoke runs its four slots serially
    monkeypatch.setattr(driver.RC, "run_replicas",
                        lambda worker, jobs, max_workers=None: [worker(*job) for job in jobs])
    driver.step_smoke(tmp_path / "r66_smoke")
    report = json.loads((tmp_path / "r66_smoke" / "smoke_fit_report.json").read_text())
    segments = report["observable_set"]["segments"]
    assert report["observable_set"]["total"] == 1255
    assert {k: v[1] - v[0] for k, v in segments.items()} == {"proton": 284, "antiproton": 122, "pi_minus": 282, "pi_plus": 260,
                                                            "k_plus": 153, "k_minus": 154}
    assert set(report["objective_at_optimum"]["chi2_by_segment"]) == set(segments)
    assert report["objective_at_optimum"]["normalization"] == 0.0
    assert report["generation_point"] == {"gamma_s_had": 1.0, "r_v": 1.0, "r_bbar": 1.0, "assoc_norm": 1.0}
    fitted = report["fitted"]
    assert fitted["r_v"] == 1.0 and all(0.05 <= fitted[n] <= 20.0 for n in driver.FLAVOR_FREE)
    assert {k: fitted[k] for k in driver.FIXED_TOPOLOGY} == driver.FIXED_TOPOLOGY
    ident = report["identity_point"]
    assert set(ident["chi2_by_segment"]) == set(segments) and set(driver.FLAVOR_TARGETS) <= set(ident["total_yield_ratios"])
    assert set(report["acceptance_flavor_round"]["criteria"]) == {"kaon_total_yields", "antiproton_total_yield", "differential_clause"}
    assert set(report["readout_class_yields"]) == set(driver.CLASS_ORDER)
    assert report["readout_mesonic_count"]["flavor"] == {k: fitted[k] for k in driver.FLAVOR_ORDER}
    assert math.isfinite(report["readout_pt_profile"]["profile_ratio"])
    assert report["pool_check"]["relative_difference"] < 1e-12 and math.isfinite(report["pool_check"]["in_process"])
    assert report["ledger_check"]["max_abs_log_weight_difference"] < 1e-9
    for cls in driver.CLASS_ORDER:
        name = "smoke_" + driver.SLOT_JSON_FMT.format(cls=cls, k=0, f=driver.SMOKE_FLOOR_INDEX, g=0)
        meta = json.loads((tmp_path / "r66_smoke" / name).read_text())
        assert meta["n_events"] == 120 and meta["conservation"]["violations"] == 0
        assert meta["informative_vertices"] > 0
        with np.load(tmp_path / "r66_smoke" / name.replace(".json", ".npz")) as data:
            arrays = {k: data[k] for k in data.files}
        assert driver.arrays_digest(arrays) == meta["arrays_sha256"]
        assert arrays["led_event"].size == meta["informative_vertices"]
        if cls == "nd":
            assert set(np.unique(arrays["rem_kind"]).tolist()) <= {1, 2}
    _check_pooled_against_in_process(driver, tmp_path / "r66_smoke")


class _InProcessPool:
    """A block pool without processes: the same replies as the forked
    workers, from one in-process objective over all blocks."""

    def __init__(self, driver, objective):
        self.driver, self.objective = driver, objective
        self.n_blocks, self.point = objective.t.n_blocks, objective.t.point

    def unit(self):
        return self.driver.block_reply(self.objective, ("unit",))[1]

    def evaluate(self, flavor_point):
        _, obs, ess = self.driver.block_reply(self.objective, ("eval", tuple(flavor_point)))
        return obs, ess


def _check_pooled_against_in_process(driver, run_dir):
    """Three replica blocks (the smoke block and two with perturbed track
    weights): the pooled combination equals Objective.evaluate in both
    parametrizations and at a nonzero normalization, value and details."""

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
    for free, norm in ((False, 0.0), (True, 0.0), (False, 0.02)):
        serial = driver.Objective(templates, a, free=free, norm=norm)
        pooled = driver.PooledObjective(tables, a, _InProcessPool(driver, driver.Objective(templates, a)), free=free, norm=norm)
        for shift in (0.0, 0.3, -0.4):
            phys = {"f_sd_total": 0.19, "f_dd": 0.06, "gamma_s_had": math.exp(-0.3 + shift), "assoc_norm": math.exp(shift),
                    "r_bbar": math.exp(1.0 - shift)}
            x = driver.x_from_physical(phys, free)
            v_s, d_s = serial.evaluate(x, details=True)
            v_p, d_p = pooled.evaluate(x, details=True)
            assert math.isclose(v_p, v_s, rel_tol=1e-10)
            assert np.allclose(d_p["m"], d_s["m"], rtol=1e-10, atol=0.0)
            assert np.allclose(d_p["se"], d_s["se"], rtol=1e-10, atol=0.0)
            assert d_p["ess"] == d_s["ess"] and d_p["blocks"] == d_s["blocks"] == [0, 1, 2]
            assert np.any(d_s["se"] > driver.SE_FLOOR)


@pytest.mark.skipif(not (RUN_DIR / "seal_manifest.json").is_file(), reason="the sealed R66 run dir is not present")
def test_sealed_archive_on_disk(driver):
    pre = json.loads((RUN_DIR / "pre_registration.json").read_text())
    assert pre["run"] == driver.RUN_ID and pre["card"] == driver.CARD
    assert pre["seeds"]["raw_first"] == driver.RAW_FIRST and pre["flavor"]["free"] == list(driver.FLAVOR_FREE)
    report = json.loads((RUN_DIR / "report.json").read_text())
    assert report["fitted"]["r_v"] == 1.0 and report["objective_at_optimum"]["normalization"] == 0.0
    assert "acceptance_flavor_round" in report and set(report["variants"]) == {label for label, _, _ in driver.VARIANTS}
    gate = json.loads((RUN_DIR / driver.GATE_RECORD_NAME).read_text())
    assert gate["verdict"] in ("CONSTRAINED", "NOT_CONSTRAINED")
    assert sealed_mismatches(RUN_DIR) == []


def test_bounded_objective_caches(driver):
    text = NESTED_DRIVER.read_text()
    src = text[text.index("class Objective"):]
    assert "while len(self.cache) > WEIGHT_CACHE_ENTRIES" in src
    assert "while len(self.se_cache) > SE_CACHE_ENTRIES" in src
