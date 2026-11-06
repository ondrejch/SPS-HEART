"""Register of the R70 joint flavor and central-component fit on the
convolved quasi-two-body widths (TASK-20261002-r6-flavor-central-convolved-r70).
The R69 register's structure, with generation 0 at the sealed R69 optimum
and the loaded project modules in the frozen substrate.

Hermetic part: the driver's identity constants against the seed ledger and
the card, the dual-tree byte identity, the parametrization and the class
split by f_central, the shared seeds of the nd and nd0 replicas, the test
statistic on constructed profiles, and a reduced smoke run of the driver
into a temporary directory (5 classes x 120 events) that exercises the
two-component variant, the free and null fits, the forked block pool, and
the readouts.  Archive part (skipped when the sealed run dir is absent):
the pre-registration, the seal inventory, and the test fields.
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
DRIVER_REL = Path("helpers") / "benchmarking" / "staged_r70_flavor_central_convolved.py"
NESTED_DRIVER = NESTED_ROOT / DRIVER_REL
OUTER_DRIVER = MONO_ROOT / DRIVER_REL
RUN_DIR = MONO_ROOT / "dev" / "00runs" / "r70_flavor_central_convolved_2026-10-02"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    return _load("staged_r70", NESTED_DRIVER)


def test_dual_tree_byte_identity():
    if not OUTER_DRIVER.is_file() or OUTER_DRIVER.resolve() == NESTED_DRIVER.resolve():
        pytest.skip("monorepo mirror not present")
    assert NESTED_DRIVER.read_bytes() == OUTER_DRIVER.read_bytes()
    for name in ("run_contract.py", "run_observables.py", "orchestrate_gateway_joint_fit.py"):
        assert (NESTED_ROOT / "helpers" / "benchmarking" / name).read_bytes() == (
            MONO_ROOT / "helpers" / "benchmarking" / name).read_bytes()


def test_identity_constants_match_the_ledger(driver):
    ledger = json.loads((NESTED_ROOT / "config" / "v2_singlet" / "seed_ledger.json").read_text())
    entry = next(b for b in ledger["blocks"] if b["run"] == driver.LEDGER_RUN)
    assert (entry["raw_first"], entry["raw_last"]) == (driver.RAW_FIRST, driver.RAW_LAST) == (20317200, 20317711)
    assert entry["partition_base"] == driver.PARTITION_BASE == 4_100_000_000
    assert entry["master_seed"] == driver.MASTER_SEED == 20261002
    assert driver.CLASS_ORDER == ("sd_projectile", "sd_target", "dd", "nd", "nd0") and driver.ND_CLASSES == ("nd", "nd0")
    assert (driver.REPLICAS, driver.N_SLOTS, driver.SEEDED_SLOTS, driver.EVENTS_PER_SLOT) == (32, 160, 128, 100_000)
    assert driver.RAW_LAST - driver.RAW_FIRST + 1 == driver.MAX_GENERATIONS * driver.SEEDED_SLOTS == 512
    assert driver.FLOOR_GRID == (1.7,) and driver.FC_BOUNDS == (0.0, 1.0)
    assert driver.FC_SCAN == (0.1, 0.25, 0.5, 0.75, 1.0) and driver.TEST_DELTA_CHI2 == 3.84 and driver.N_FREE_PARAMETERS == 6
    assert (driver.NORM_FIXED, driver.NORM_VARIANTS, driver.DISCREPANCY, driver.DISCREPANCY_VARIANT) == (0.0, (0.02, -0.02), 0.05, 0.10)
    assert driver.FIT_SPECIES == ("proton", "antiproton", "pi_minus", "pi_plus", "k_plus", "k_minus")
    assert driver.GEN0_SHAPE == (2.9999999920746907, 2.572376312705535, 0.2790534289674131)
    assert (driver.GEN0_W_LOW, driver.GEN0_W_EXCITED, driver.GEN0_F_CENTRAL) == (0.07121542186415353, 0.9287845781358465, 0.3158012461067854)
    assert (driver.FIXED_EPSILON, driver.FIXED_B_SD) == driver.GEN0_SHAPE[:2]
    assert driver.CONTINUATION_CEILING == 17.3 and driver.N_FREE_PARAMETERS == 6
    assert driver.SHAPE_NAMES == ("epsilon_nd", "w_excited", "gamma_s_had", "assoc_norm", "r_bbar")
    assert driver.FLAVOR_FREE == ("gamma_s_had", "assoc_norm", "r_bbar") and driver.R_V_FIXED == 1.0
    assert driver.GEN_FLAVOR == (0.4856389143548471, 1.0, 1.1170593534175117, 0.4802597606677879)
    smoke = next(b for b in ledger["blocks"] if b["run"] == "round7_smoke")
    assert smoke["raw_first"] <= driver.SMOKE_SEED <= smoke["raw_last"] and driver.SMOKE_PARTITION >= smoke["partition_base"]
    for rel in ("helpers/benchmarking/staged_r69_flavor_central_uncapped.py", "helpers/benchmarking/staged_r68_central_component_uncapped.py",
                "dev/mixing2006/na49_mixing/topology_non_diffractive.py", "dev/mixing2006/na49_mixing/cascade_decay.py",
                "dev/mixing2006/na49_mixing/resonance_widths.py", "dev/mixing2006/na49_mixing/resonance_decay.py"):
        assert rel in driver.FROZEN_RELS


def test_the_frozen_substrate_covers_the_loaded_modules(driver):
    """External review q1 H1: every loaded project module is frozen,
    beside the FROZEN_RELS literal."""
    if MONO_ROOT == NESTED_ROOT:
        pytest.skip("monorepo layout not present")
    rels = driver.frozen_rels()
    assert rels[:len(driver.FROZEN_RELS)] == driver.FROZEN_RELS and len(set(rels)) == len(rels)
    for rel in ("dev/mixing2006/na49_mixing/rng_substreams.py", "dev/mixing2006/na49_mixing/cascade_alignment.py",
                "dev/mixing2006/na49_mixing/isospin.py", "dev/mixing2006/na49_mixing/pdg_database.py"):
        assert rel in rels


def test_the_ceiling_is_lifted_in_memory(driver):
    _, fams = driver._load_physics(1.7)
    assert all(f.continuation_ceiling == 17.3 and f.continuation_floor == 1.7 for f in fams)
    committed = driver.families_from_json(str(NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json"))
    assert all(f.continuation_ceiling is None for f in committed)


def test_generation_zero_matches_the_sealed_r69_optimum(driver):
    r69 = MONO_ROOT / "dev" / "00runs" / driver.R69_RUN_ID / "branch_report_f0.json"
    if not r69.is_file():
        pytest.skip("the sealed R69 archive is not present")
    branch = json.loads(r69.read_text())
    assert branch["digest"] == driver.R69_BRANCH_DIGEST
    fitted = branch["fitted"]
    assert (fitted["epsilon"], fitted["b_sd"], fitted["epsilon_nd"]) == driver.GEN0_SHAPE
    assert (fitted["w_low_mass"], fitted["w_excited"], fitted["f_central"]) == (driver.GEN0_W_LOW, driver.GEN0_W_EXCITED, driver.GEN0_F_CENTRAL)
    assert tuple(fitted[n] for n in driver.FLAVOR_ORDER) == driver.GEN_FLAVOR


def test_nd0_shares_the_nd_seeds(driver):
    nd = driver.CLASS_ORDER.index("nd") * driver.REPLICAS
    nd0 = driver.CLASS_ORDER.index("nd0") * driver.REPLICAS
    for g in range(driver.MAX_GENERATIONS):
        for k in range(driver.REPLICAS):
            assert driver.slot_seed(0, g, nd0 + k) == driver.slot_seed(0, g, nd + k)
            assert driver.slot_partition(nd0 + k) == driver.slot_partition(nd + k)
    seeds = {driver.slot_seed(0, g, s) for g in range(driver.MAX_GENERATIONS) for s in range(driver.N_SLOTS)}
    assert len(seeds) == 512 and min(seeds) == driver.RAW_FIRST and max(seeds) == driver.RAW_LAST
    assert driver._params_at("nd0", driver.GEN0_SHAPE).nd_central is False
    assert driver._params_at("nd", driver.GEN0_SHAPE).nd_central is True


def test_parametrization_and_the_class_split(driver):
    a = driver.anchors()
    phys = {"w_excited": 0.8, "epsilon_nd": 0.1, "f_central": 0.37, "gamma_s_had": 0.7, "assoc_norm": 0.3, "r_bbar": 2.2}
    for fc_fixed in (None, 0.0, 0.5):
        x = driver.x_from_physical(phys, fc_fixed)
        back = driver.physical_from_x(x, fc_fixed, a)
        assert math.isclose(back["w_excited"], 0.8, rel_tol=1e-12) and back["epsilon"] == driver.FIXED_EPSILON
        assert all(math.isclose(back[n], phys[n], rel_tol=1e-12) for n in driver.FLAVOR_FREE) and back["r_v"] == 1.0
        assert back["f_central"] == (0.37 if fc_fixed is None else fc_fixed) or math.isclose(back["f_central"], 0.37)
        frac = driver.class_fractions(back)
        assert math.isclose(sum(frac.values()), 1.0, rel_tol=1e-12)
        assert math.isclose(frac["nd"] + frac["nd0"], back["f_nd"], rel_tol=1e-12)
    assert len(driver.bounds_for(None)) == 6 and len(driver.bounds_for(0.0)) == 5


def test_the_test_statistic(driver):
    """Constructed profiles: a flat profile is compatible with zero; a
    steep one is not; the scaled interval crosses the threshold; a scan
    point below the free fit becomes the minimum and is flagged."""

    n_obs = 1255
    chi2 = 1249.0 * 4.0                     # S^2 = 4, threshold 15.36
    flat = driver.central_test(chi2, chi2 + 5.0, n_obs, 0.3, [{"f_central": 1.0, "objective": chi2 + 40.0}])
    assert flat["compatible_with_zero"] and math.isclose(flat["threshold"], 3.84 * 4.0)
    assert not flat["compatible_with_zero_raw"]
    assert flat["scaled_interval"]["upper"] is not None and 0.3 < flat["scaled_interval"]["upper"] < 1.0
    steep = driver.central_test(chi2, chi2 + 200.0, n_obs, 0.6, [{"f_central": 0.25, "objective": chi2 + 50.0}])
    assert not steep["compatible_with_zero"] and steep["scaled_interval"]["lower"] is not None
    missed = driver.central_test(chi2, chi2 + 30.0, n_obs, 0.6, [{"f_central": 0.25, "objective": chi2 - 10.0}])
    assert not missed["free_fit_is_minimum"] and missed["f_central_hat"] == 0.25 and missed["chi2_min"] == chi2 - 10.0


@_SEALED_KERNEL_SKIP
def test_reduced_smoke_run(driver, tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "SMOKE_EVENTS_PER_CLASS", 120)
    monkeypatch.setattr(driver, "SMOKE_MESONIC_DECAYS", 20)
    # the driver is loaded by path, so its worker cannot be pickled into a
    # process pool; the smoke runs its slots serially
    monkeypatch.setattr(driver.RC, "run_replicas",
                        lambda worker, jobs, max_workers=None: [worker(*job) for job in jobs])
    driver.step_smoke(tmp_path / "r70_smoke")
    report = json.loads((tmp_path / "r70_smoke" / "smoke_fit_report.json").read_text())
    assert report["observable_set"]["total"] == 1255
    assert report["common_random_numbers"] == {k: True for k in ("rem_kind", "shape_m1", "shape_m2", "shape_t1", "shape_t2")}
    assert report["pool_check"]["relative_difference"] < 1e-12 and math.isfinite(report["pool_check"]["in_process"])
    fitted = report["fitted"]
    assert 0.0 <= fitted["f_central"] <= 1.0 and fitted["w_intact"] == 0.0
    assert set(fitted["class_fractions"]) == set(driver.CLASS_ORDER)
    test = report["central_component_test"]
    for key in ("delta_chi2", "threshold", "compatible_with_zero", "scaled_interval", "profile", "chi2_min"):
        assert key in test
    assert report["null_fit"]["fitted"]["f_central"] == 0.0
    assert set(report["readout_class_yields"]) == set(driver.CLASS_ORDER)
    assert report["test_summary"]["compatible_in_primary"] == test["compatible_with_zero"]
    assert report["generation_point"]["gamma_s_had"] == driver.GEN_FLAVOR[0]
    # the key-table flavor reweighting reproduces flavor_fit_core.flavor_log_weights
    import flavor_fit_core as FC
    contract = driver._contract(tmp_path / "r70_smoke")
    templates = driver.Templates(contract, driver.SMOKE_FLOOR_INDEX, 0, driver.load_fit_tables(), smoke=True)
    target = (0.5, 1.0, 2.5, 0.4)
    for cls in driver.CLASS_ORDER:
        with np.load(tmp_path / "r70_smoke" / ("smoke_" + driver.SLOT_NPZ_FMT.format(cls=cls, k=0, f=driver.SMOKE_FLOOR_INDEX, g=0))) as data:
            arrays = {k: data[k] for k in data.files}
        block = templates.by_class[cls][0]
        mine = templates.flavor_log_weights(block, target)
        theirs = FC.flavor_log_weights(arrays, block["n"], driver.flavor_params(driver.GEN_FLAVOR), driver.flavor_params(target))
        assert np.max(np.abs(mine - theirs)) < 1e-9
    for cls in driver.CLASS_ORDER:
        name = "smoke_" + driver.SLOT_JSON_FMT.format(cls=cls, k=0, f=driver.SMOKE_FLOOR_INDEX, g=0)
        meta = json.loads((tmp_path / "r70_smoke" / name).read_text())
        assert meta["n_events"] == 120 and meta["conservation"]["violations"] == 0
        assert meta["nd_central"] == (cls != "nd0")
        with np.load(tmp_path / "r70_smoke" / name.replace(".json", ".npz")) as data:
            arrays = {k: data[k] for k in data.files}
        assert driver.arrays_digest(arrays) == meta["arrays_sha256"]
        if cls == "nd0":
            assert np.all(arrays["shape_mc2"] == 0.0)
        if cls == "nd":
            assert np.all(arrays["shape_mc2"] > 0.0)


@pytest.mark.skipif(not (RUN_DIR / "seal_manifest.json").is_file(), reason="the sealed R70 run dir is not present")
def test_sealed_archive_on_disk(driver):
    pre = json.loads((RUN_DIR / "pre_registration.json").read_text())
    assert pre["run"] == driver.RUN_ID and pre["card"] == driver.CARD and pre["seeds"]["raw_first"] == driver.RAW_FIRST
    report = json.loads((RUN_DIR / "report.json").read_text())
    assert "central_component_test" in report and "test_summary" in report
    assert set(report["variants"]) == {label for label, _, _ in driver.VARIANTS}
    gate = json.loads((RUN_DIR / driver.GATE_RECORD_NAME).read_text())
    assert gate["verdict"] in ("CONSTRAINED", "NOT_CONSTRAINED") and "compatible_with_zero" in gate["test"]
    assert sealed_mismatches(RUN_DIR) == []
