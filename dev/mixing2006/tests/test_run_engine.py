"""Register of the run engine (TASK-20261004-refactor-run-engine-kernel-config,
P2): the engine configured by ``config/runs/r74.json`` reproduces the sealed
R74 driver bit for bit (constants, starts, bounds, the optimizer transforms,
the kernel, and a reduced run through generation, the free, null and scan
fits, the Hessian, the block fits and the variants), its pre-registration
carries the seed block, the kernel configuration and the spec digest, and
the dual trees hold the same engine."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

NESTED_ROOT = Path(__file__).resolve().parents[1]
MONO_ROOT = NESTED_ROOT.parents[1] if NESTED_ROOT.parent.name == "dev" else NESTED_ROOT
ENGINE_REL = Path("helpers") / "benchmarking" / "run_engine.py"
R74_REL = Path("helpers") / "benchmarking" / "staged_r74_central_component_barrier_step.py"
SPEC = NESTED_ROOT / "config" / "runs" / "r74.json"
REPLAY_SPEC = NESTED_ROOT / "config" / "runs" / "r74_engine_replay.json"
IDENTITY_KEYS = {"run", "run_dir", "schema", "card", "driver", "driver_sha256", "digest", "branch_digest"}
CONSTANTS = ("RUN_ID", "CARD", "SCHEMA", "LEDGER_RUN", "GATE_RECORD_NAME", "REPLICAS", "EVENTS_PER_SLOT", "N_SLOTS", "SEEDED_SLOTS",
             "MAX_GENERATIONS", "FLOOR_GRID", "MASTER_SEED", "RAW_FIRST", "RAW_LAST", "PARTITION_BASE", "SMOKE_SEED", "SMOKE_PARTITION",
             "CONTINUATION_CEILING", "EPS_BOUNDS", "EPS_ND_BOUNDS", "B_SD_BOUNDS", "LOGIT_BOUNDS", "FC_BOUNDS", "FLAVOR_LOG_BOUNDS",
             "HESSIAN_STEPS", "SHAPE_NAMES", "FLAVOR_FREE", "R_V_FIXED", "FC_SCAN", "TEST_DELTA_CHI2", "N_FREE_PARAMETERS", "N_STARTS",
             "N_STARTS_NULL", "N_STARTS_VARIANT", "GEN_FLAVOR", "GEN0_EPSILON", "GEN0_B_SD", "GEN0_EPSILON_ND", "GEN0_W_EXCITED",
             "GEN0_W_LOW", "GEN0_F_CENTRAL", "GEN0_SHAPE", "ND_GENERATION_REMNANT_WEIGHTS", "VARIANTS")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def engine():
    e = _load("run_engine_under_test", NESTED_ROOT / ENGINE_REL)
    e.configure(SPEC)
    return e


@pytest.fixture(scope="module")
def r74():
    return _load("staged_r74_for_engine", NESTED_ROOT / R74_REL)


def _physics(report: dict) -> dict:
    return {k: v for k, v in report.items() if k not in IDENTITY_KEYS}


def _without_bound_piles(x):
    """The engine names replica-block minima piled at a bound (2026-10-27, R122), an informational residual that the
    R74 driver predates; the comparison drops it."""
    if isinstance(x, dict):
        return {k: ([r for r in v if r.get("kind") != "replica_block_minima_on_bound"] if k == "named_residual_register"
                    else _without_bound_piles(v)) for k, v in x.items()}
    if isinstance(x, list):
        return [_without_bound_piles(v) for v in x]
    return x


def test_dual_tree_byte_identity():
    outer = MONO_ROOT / ENGINE_REL
    if not outer.is_file() or outer.resolve() == (NESTED_ROOT / ENGINE_REL).resolve():
        pytest.skip("monorepo mirror not present")
    assert outer.read_bytes() == (NESTED_ROOT / ENGINE_REL).read_bytes()


def test_the_spec_reproduces_the_r74_constants(engine, r74):
    assert [n for n in CONSTANTS if getattr(engine, n) != getattr(r74, n)] == []
    assert engine.param_names(None) == r74.param_names(None) and engine.param_names(0.0) == r74.param_names(0.0)


def test_starts_bounds_and_transforms_are_bit_identical(engine, r74):
    a = r74.anchors()
    for fc in (None, 0.0):
        assert engine.bounds_for(fc) == r74.bounds_for(fc)
        mine, theirs = engine.dispersed_starts(fc), r74.dispersed_starts(fc)
        assert len(mine) == len(theirs) and all(np.array_equal(x, y) for x, y in zip(mine, theirs))
    rng = np.random.default_rng(20261004)
    for _ in range(300):
        x = np.array([rng.uniform(lo, hi) for lo, hi in r74.bounds_for(None)])
        assert engine.physical_from_x(x, None, a) == r74.physical_from_x(x, None, a)
        assert engine.physical_from_x(x[:-1], 0.0, a) == r74.physical_from_x(x[:-1], 0.0, a)
        phys = r74.physical_from_x(x, None, a)
        assert np.array_equal(engine.x_from_physical(phys, None), r74.x_from_physical(phys, None))


def test_the_kernel_is_the_r74_kernel(engine, r74):
    assert engine._load_physics(1.7)[1] == r74._load_physics(1.7)[1]


def test_the_replay_spec_differs_only_in_identity():
    spec, replay = json.loads(SPEC.read_text()), json.loads(REPLAY_SPEC.read_text())
    assert {k: v for k, v in spec.items() if k not in ("identity", "description")} == \
        {k: v for k, v in replay.items() if k not in ("identity", "description")}
    assert replay["identity"]["ledger_run"] == spec["identity"]["ledger_run"]
    assert replay["identity"]["run_id"] != spec["identity"]["run_id"]


def _reduced(module, run_dir, monkeypatch, n_events=100):
    """A two-replica run dir at the generation point (the R74 register's
    reduced run)."""
    monkeypatch.setattr(module, "REPLICAS", 2)
    monkeypatch.setattr(module, "N_SLOTS", len(module.CLASS_ORDER) * 2)
    monkeypatch.setattr(module, "SEEDED_SLOTS", (len(module.CLASS_ORDER) - 1) * 2)
    monkeypatch.setattr(module, "MESONIC_DECAYS", 30)
    monkeypatch.setattr(module, "N_STARTS", 2)
    monkeypatch.setattr(module, "N_STARTS_NULL", 2)
    monkeypatch.setattr(module, "N_STARTS_VARIANT", 2)
    monkeypatch.setattr(module, "FC_SCAN", (0.5,))
    monkeypatch.setattr(module.RC, "run_replicas", lambda worker, jobs, max_workers=None: [worker(*job) for job in jobs])
    contract = module._contract(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for slot in range(module.N_SLOTS):
        res = module.generate_slot(module.slot_class(slot), module.SMOKE_SEED + module.seed_slot(slot), n_events,
                                   module.SMOKE_PARTITION + 10_000 * module.seed_slot(slot), module.GEN0_SHAPE, module.FLOOR_GRID[0],
                                   module.GEN_FLAVOR)
        module.write_slot(contract, 0, 0, slot, res, "test")
    return contract


def test_a_reduced_run_is_bit_identical(engine, r74, tmp_path, monkeypatch):
    """Generation, the free, null and scan fits, the Hessian, the block fits,
    the readouts and the variants: the engine's reports equal the R74
    driver's, the identity block aside."""
    import fit_fanout as FO
    import run_contract as RC
    # the R74 driver forms the observables at the recorded sqrt(s) = 17.270
    # GeV; the engine uses the exact CM value since 2026-10-07 (review
    # 2026-10-06, Muse L-1), so the comparison runs it at the driver's value
    monkeypatch.setattr(engine, "SQRT_S_CM", engine.SQRT_S)
    reports = {}
    for label, module in (("r74", r74), ("engine", engine)):
        run_dir = tmp_path / label
        _reduced(module, run_dir, monkeypatch)
        monkeypatch.setattr(module, "_FANOUT", FO.FanOut(run_dir, module.FANOUT_REGISTRY, mode="local"))
        tables, a = module.load_fit_tables(), module.anchors()
        branch = module.evaluate_generation(module._contract(run_dir), 0, 0, tables, a)
        variants = module._variants(module._contract(run_dir), branch, tables)
        slots = sorted((p.name, json.loads(p.read_text())["arrays_sha256"]) for p in run_dir.glob("slot_*.json"))
        reports[label] = (RC.canonical_digest(_physics(_without_bound_piles(branch))),
                          RC.canonical_digest(_without_bound_piles(variants)), slots)
    assert reports["engine"] == reports["r74"]


def test_the_pre_registration(engine, tmp_path):
    r73 = MONO_ROOT / "dev" / "00runs" / engine.SPEC["generation_0"]["run"] / "seal_manifest.json"
    if MONO_ROOT == NESTED_ROOT or not r73.is_file():
        pytest.skip("the monorepo layout or the sealed R73 archive is not present")
    engine.step_pre_registration(tmp_path / "pre")
    pre = json.loads((tmp_path / "pre" / "pre_registration.json").read_text())
    assert (pre["seeds"]["raw_first"], pre["seeds"]["partition_base"]) == (engine.RAW_FIRST, engine.PARTITION_BASE)
    assert pre["kernel_config"] == {"continuation_floor": 1.7, "continuation_ceiling": 17.3,
                                    "continuation_mass_law": "density", "continuation_barrier_l": 1, "isospin_completion": [],
                                    "delta_lines": False, "isovector_meson_pairs": False,
                                    "continuation_sibling_weight": None, "three_body_phase_space": False,
                                "continuation_daughters": None, "continuation_t_h_alpha": None, "leading_split_fraction": None, "leading_split_kappa": None,
                                "leading_split_slope": None, "leading_split_epsilon": None, "continuation_q_min": None, "sibling_line_width": None, "meson_split_fraction": None, "meson_split_slope": None, "meson_split_slope_strange": None,
                                "string_step_mass": None, "string_step_slope": None, "string_step_epsilon": None, "leading_split_strange": None, "leading_split_regge": None, "leading_split_axis": None, "xi_tower": None,
                                "isovector_neutral_mesons": False, "isoscalar_dipion": False,
                                "floor_pools": [], "conjugate_towers": False, "baryon_resonance_pairs": False,
                                "non_diffractive_isospin_mixture": False, "light_scalar_mesons": False,
                                "nucleon_kstar_pairs": False,
                                "pdg_hyperons": False, "eta_prime": False, "hyperon_meson_partners": False,
                                "delta_isoscalar_pairs": False, "charged_scalar_pairs": False,
                                "xi_baryons": False, "pdg_meson_modes": False, "nstar_range_rule": False,
                                "three_body_endings": None, "dalitz_matrix_elements": False, "axial_isoscalars": False, "excited_kaons": False,
                                "pdg_nucleon_resonances": False, "associated_kstar_pairs": False,
                                "baryon_table_corrections": False, "baryon_pdg_widths": False,
                                "pdg_nucleon_resonance_siblings": False,
                                "resonance_pair_line_shapes": False, "direct_baryon_pairs_in_gap": False,
                                "baryon_pair_s_wave": False, "baryon_forward_hemisphere": False, "baryon_pair_completion": None, "baryon_pair_completion_members": "octet_decuplet", "baryon_pair_share": None, "omega_baryon": False, "closed_channel_reference": False, "baryon_line_kappa": None, "baryon_line_kappa_scope": "all", "baryon_line_kappa_final": None, "baryon_line_kappa_kaon": None, "baryon_line_kappa_strange": None, "baryon_line_kappa_axis": "beam"}
    assert pre["run_spec"]["path"] == engine.SPEC_REL
    engine_rel = Path(engine.__file__).resolve().relative_to(MONO_ROOT.resolve()).as_posix()
    assert engine.SPEC_REL in pre["frozen_substrate"] and engine_rel in pre["frozen_substrate"]
    assert pre["generation_0_point"]["provenance"]["fit"] == "null" and len(pre["free_parameters"]) == 8


def test_the_cli_needs_a_spec():
    e = _load("run_engine_cli", NESTED_ROOT / ENGINE_REL)
    with pytest.raises(SystemExit):
        e.main(["--evaluate"])


def test_the_regeneration_point_uses_trust_region_fits(engine):
    """R76 (2026-10-05): every null-fit start of generation 2 ended in the
    trust-region penalty at an arbitrary point (epsilon 3.1, w_excited 0.03),
    and the midpoint rule averaged it into the final generation point.  The
    rule now uses only fits inside the trust region."""
    def details(eps, gs, ok):
        phys = {"epsilon": eps, "b_sd": 3.0, "epsilon_nd": 0.2, "gamma_s_had": gs, "r_v": 1.0, "r_bbar": 1.5, "assoc_norm": 1.8}
        return {"physical": phys, "in_trust_region": ok}
    free, null, garbage = details(0.18, 0.24, True), details(0.30, 0.25, True), details(3.09, 2.07, False)
    point, basis = engine.regeneration_point_of(free, null)
    assert basis == "midpoint" and point["epsilon"] == 0.5 * (0.18 + 0.30)
    assert point["gamma_s_had"] == pytest.approx((0.24 * 0.25) ** 0.5)
    point, basis = engine.regeneration_point_of(free, garbage)
    assert basis == "free only" and point["epsilon"] == 0.18 and point["gamma_s_had"] == 0.24
    point, basis = engine.regeneration_point_of(garbage, null)
    assert basis == "null only" and point["epsilon"] == 0.30
    point, basis = engine.regeneration_point_of(garbage, details(2.5, 1.0, False))
    assert basis == "free, neither in the trust region" and point["epsilon"] == 3.09
