"""Hermetic tests for the R51 run driver (the topology fraction fit).

Task ``TASK-20260925-r5-run-r51-topology-fraction-fit`` (plan section 11
run R51, lines 909-918): the pre-registered replica seed map (8 seeds,
20300945-20300952) and the event-index partition (1_100_000_000 + i,
i < 100,000); the fit-set invariants (exactly five free parameters, no
strange-sector parameters, the blend ramp start fixed at B, the elastic
class excluded); the fraction-sum constraint on a pinned fixture; the
likelihood determinism on a pinned covariance; the replica-stability
algebra on a pinned fixture; the smoke-mode refusal conventions; the
analytic normalization identities of the importance-weight machinery;
the layout sync; and the freeze re-pin register.  No Monte Carlo
generation at run scale in the deterministic legs (the sealed-dir pins
read the sealed run dirs when the artifact store is present).
"""

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from tests._freeze_inventory import inventory_count as _inventory_count


def _load_driver():
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "helpers" / "benchmarking"
        / "staged_r51_topology_fraction_fit.py",
    ]
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location(
                "staged_r51_topology_fraction_fit", path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(
        "the R51 driver was not found next to the test file")


r51 = _load_driver()


# ---------------------------------------------------------------------------
# The run identity constants (the card's pre-registered identity)
# ---------------------------------------------------------------------------
def test_run_identity_constants():
    """The run identity: R51, the card, the execution-date run name (the
    card moves the date with the actual execution; the pre-registered
    2026-09-25 identity re-pins to the execution date), the T2
    statistics, the seed block, and the smoke block."""
    assert r51.RUN_ID == "R51"
    assert r51.CARD == "TASK-20260925-r5-run-r51-topology-fraction-fit"
    assert r51.RUN_NAME.startswith("r51_topology_fraction_fit_")
    assert r51.DEFAULT_RUN_DIR == f"dev/00runs/{r51.RUN_NAME}"
    assert r51.MASTER_SEED == 20260925
    assert r51.N_REPLICAS == 8
    assert r51.EVENTS_PER_REPLICA == 100_000
    assert r51.N_EVENTS_TOTAL == 800_000
    assert r51.REPLICA_SEEDS == tuple(
        20300945 + k for k in range(8))
    assert r51.EVENT_INDEX_BASE == 1_100_000_000
    assert r51.SMOKE_SEED == 20260925 + 40_000 + 12_000 == 20312925
    assert r51.SMOKE_EVENTS == 2_000
    assert r51.SMOKE_EVENT_INDEX_BASE == 1_600_000_000
    assert r51.PROBE_REPLICA == 0
    assert r51.CLASS_ORDER == (
        "sd_projectile", "sd_target", "dd", "nd")
    assert r51.FIT_PARAMETERS == (
        "sd_projectile_fraction", "sd_target_fraction", "dd_fraction",
        "epsilon", "b_sd")
    assert r51.MULTIPLICITY_BINS == tuple(range(2, 15))


def test_seed_map_and_partition():
    """The seed map (index item 9): the 8 raw replica seeds in order,
    the block disjoint from every previously used raw block and from
    every reserved Round 5 block, the event-index partition disjoint
    from every recorded partition, and the smoke block disjoint from
    the production block by seed and by partition."""
    seeds = [r51.replica_seed(k) for k in range(8)]
    assert seeds == list(range(20300945, 20300953))
    assert seeds == list(r51.REPLICA_SEEDS)
    report = r51.seed_disjointness_report()
    assert report["disjoint"] is True
    assert report["shared_raw_seeds"] == {}
    r51_lo, r51_hi = report["r51_event_index_range"]
    assert r51_lo == 1_100_000_000 and r51_hi == 1_100_100_000
    for name, (lo, hi) in r51.RESERVED_ROUND5_PARTITIONS.items():
        assert r51_hi <= lo or hi <= r51_lo, name
    for name, block in r51.PREVIOUS_USED_BLOCKS.items():
        p_range = block.get("event_index_range")
        if p_range is not None:
            assert r51_hi <= p_range[0] or p_range[1] <= r51_lo, name
    for name, (lo, hi) in r51.RESERVED_ROUND5_BLOCKS.items():
        assert not (set(range(lo, hi + 1)) & set(r51.REPLICA_SEEDS)), name
    assert r51.SMOKE_SEED not in set(r51.REPLICA_SEEDS)
    assert r51.SMOKE_EVENT_INDEX_BASE >= r51_hi


def test_fit_set_invariants():
    """The fit-set invariants (acceptance (iii); index item 3; plan
    section 24): exactly five free parameters with the gate template's
    names; the sum-rule closure over the inelastic sample (the elastic
    class excluded per plan line 824); no strange-sector parameter; the
    blend ramp start fixed at B and never in the fit set; the initial
    values the sealed R50 nominal point; the runner's nominal parameter
    surface intact."""
    status = r51.assert_fit_set()
    assert status["pass"] is True
    assert status["count"] == 5
    assert tuple(status["parameters"]) == (
        "sd_projectile_fraction", "sd_target_fraction", "dd_fraction",
        "epsilon", "b_sd")
    assert status["closure"] == (
        "f_nd = 1 - f_sd_p - f_sd_t - f_dd (the inelastic fractions sum "
        "to one over the inelastic sample)")
    assert status["initial_values"] == {
        "sd_projectile_fraction": 0.25, "sd_target_fraction": 0.25,
        "dd_fraction": 0.25, "epsilon": 0.0, "b_sd": 8.504140762083612,
    }
    assert status["initial_f_nd"] == 0.25
    excluded = status["excluded"]
    assert "plan line 918" in excluded["strange_sector"]
    assert "plan line 824" in excluded["elastic"]
    assert "obligation 4" in excluded["blend_ramp_start"]
    assert r51.BLEND_RAMP_START_GEV == 2.680194110727528
    assert all("strange" not in name.lower()
               for name in r51.FIT_PARAMETERS)
    # no exchange, topology, flavor, branching, or catalog parameter is
    # fit: the fit set is exactly the five names above (plan section 24)
    assert not (set(r51.FIT_PARAMETERS) & {
        "blend_ramp_start", "kappa_B", "kappa_M", "b_dec",
        "baryonic_nominal_width", "W_flavor", "p_n2"})
    # the runner's nominal surface (the generation point) is intact
    params = r51.NOMINAL_PARAMS
    assert params.epsilon == 0.0
    assert params.b_sd == 8.504140762083612
    assert params.b_sd.hex() == float(8.504140762083612).hex()
    assert dict(params.class_fractions) == {
        "sd_projectile": 0.25, "sd_target": 0.25, "dd": 0.25, "nd": 0.25}
    assert params.p_n2 == 0.0
    assert set(r51.INELASTIC_SET) == {
        "sd_projectile", "sd_target", "dd", "nd"}
    assert r51.TOPOLOGY_CLASSES["elastic"]["inelastic"] is False
    assert r51.TOPOLOGY_CLASSES["nd_multi"]["retained"] is False
    m8 = r51.m8_frozen_set_status()
    assert m8["verified"] is True
    assert m8["hash"] == (
        "358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1")
    assert m8["effective_transfer_slope_gev2"] == params.b_sd
    assert m8["blend_ramp_start_gev"] == 2.680194110727528


def test_optimizer_settings_pre_registered():
    """The optimizer settings are pre-registered constants (the
    deterministic algorithm, the tolerances, the evaluation budget) and
    the initial point equals the fit set's initial values in order."""
    settings = r51.OPTIMIZER_SETTINGS
    assert "Nelder-Mead" in settings["algorithm"]
    assert settings["xatol"] == 1e-3
    assert settings["fatol"] == 1e-6
    assert settings["maxiter"] == 600
    assert settings["maxfev"] == 1200
    assert settings["initial_point"] == [
        r51.INITIAL_VALUES[name] for name in r51.FIT_PARAMETERS]
    for name in r51.FIT_PARAMETERS:
        lo, hi = r51.PARAM_BOUNDS[name]
        assert lo < hi
    assert r51.HESSIAN_STEPS["b_sd"] == 0.05
    assert r51.STABILITY_SPREAD_BOUND == 0.20
    assert r51.NORM_NUISANCE_PRIOR == 0.35
    assert r51.CARRY_NUISANCE_PRIOR == 0.25


# ---------------------------------------------------------------------------
# The pinned synthetic-block fixture (the fit machinery's hermetic data)
# ---------------------------------------------------------------------------
def _synthetic_block(replica, n_events, seed=7):
    """A pinned synthetic block with the driver's Block structure: the
    per-class weight inputs, the gap flags, the charged-multiplicity
    bins, and the proton/antiproton lattice entries built from pinned
    values (no generation, no physics)."""
    rng = np.random.default_rng(seed)
    labels = np.asarray([r51.CLASS_ORDER[i % 4] for i in range(n_events)])
    block = r51.Block()
    block.replica = replica
    block.seed = 0
    block.n_events = n_events
    block.cls = np.asarray([r51.CLASS_INDEX[l] for l in labels])
    block.idx_by_class = {
        c: np.flatnonzero(block.cls == r51.CLASS_INDEX[c])
        for c in r51.CLASS_ORDER}
    m2 = np.exp(rng.uniform(np.log(0.88), np.log(200.0), n_events))
    arrays = {}
    for cls in r51.CLASS_ORDER:
        idx = block.idx_by_class[cls]
        n = idx.size
        if cls in ("sd_projectile", "sd_target"):
            arrays[cls] = {
                "m2": m2[idx],
                "t": -np.abs(rng.normal(0.2, 0.1, n)),
                "m_rem": np.full(n, r51.PROTON_MASS),
            }
        elif cls == "dd":
            arrays[cls] = {
                "m12": m2[idx], "m22": m2[idx] * 0.5,
                "t1": -np.abs(rng.normal(0.2, 0.1, n)),
                "t2": -np.abs(rng.normal(0.2, 0.1, n)),
            }
        else:
            arrays[cls] = {
                "t1": -np.abs(rng.normal(0.2, 0.1, n)),
                "t2": -np.abs(rng.normal(0.2, 0.1, n)),
                "t_c": -np.abs(rng.normal(0.4, 0.1, n)),
                "mc2": np.exp(rng.uniform(np.log(0.02), np.log(50.0), n)),
                "m1": np.full(n, r51.PROTON_MASS),
                "m2": np.full(n, 1.3),
            }
    block.arrays_by_class = arrays
    block.gap = rng.random(n_events) < 0.4
    block.nch_idx = rng.integers(0, len(r51.MULTIPLICITY_BINS), n_events)
    n_prot = n_events // 4
    block.prot_evt = np.asarray(rng.integers(0, n_events, n_prot))
    block.prot_flat = np.asarray(
        rng.integers(0, 144, n_prot), dtype=np.int64)
    block.prot_w = np.ones(n_prot, dtype=float)
    block.pbar_evt = np.zeros(0, dtype=np.int64)
    block.pbar_flat = np.zeros(0, dtype=np.int64)
    block.pbar_w = np.zeros(0, dtype=float)
    block.proton_out_of_window = 0
    block.antiproton_out_of_window = 0
    block.n_protons = n_prot
    block.n_antiprotons = 0
    return block


def _synthetic_geometry():
    """A pinned synthetic geometry matching the fixture's lattice dims
    (12 x 12 dense lattices, pick maps over 100 rows, unit areas)."""
    n_x, n_p = 12, 12
    x_edges = np.linspace(0.0, 0.4, n_x + 1)
    p_edges = np.linspace(0.0, 1.5, n_p + 1)
    rows = [(ip, ix) for ip in range(n_p) for ix in range(n_x)]
    pick = np.asarray([ip * n_x + ix for ip, ix in rows], dtype=np.int64)
    areas = np.full(pick.size, (0.4 / n_x) * (1.5 / n_p))
    geometry = {
        "edges": {"proton": (x_edges, p_edges),
                  "antiproton": (x_edges, p_edges)},
        "n_x": {"proton": n_x, "antiproton": n_x},
        "dense_size": {"proton": n_x * n_p, "antiproton": n_x * n_p},
        "pick": {"proton": pick, "antiproton": pick},
        "areas": {"proton": areas, "antiproton": areas},
        "unit_structure": {
            "leading_proton_spectrum": np.ones(pick.size, dtype=bool),
            "b_transport_spectrum": np.ones(pick.size, dtype=bool),
            "gap_fraction_abs_y_lt_0.5": np.asarray([True]),
            "charged_multiplicity_distribution":
                np.ones(len(r51.MULTIPLICITY_BINS), dtype=bool),
        },
    }
    return geometry


def _synthetic_stats(n_rows=144):
    """The pinned synthetic statistic set (the data side and the
    covariance pinned by construction); the row count matches the
    synthetic geometry's published-row pick map."""
    d_proton = np.linspace(0.05, 0.3, n_rows)
    v_proton = np.full(n_rows, 1e-4)
    d_b = np.linspace(0.04, 0.25, n_rows)
    v_b = np.full(n_rows, 1e-4)
    return [
        r51.Statistic(
            key="leading_proton_spectrum", kind="spectrum",
            species="proton", d=d_proton, v_data=v_proton, cells=None,
            carry_mask=None, note="synthetic"),
        r51.Statistic(
            key="b_transport_spectrum", kind="spectrum", species=None,
            d=d_b, v_data=v_b,
            cells=[{"bin_a": i, "bin_b": i, "xF_mid": 0.0, "pT_mid": 0.0}
                   for i in range(n_rows)],
            carry_mask=np.zeros(n_rows), note="synthetic"),
        r51.Statistic(
            key="gap_fraction_abs_y_lt_0.5", kind="fraction", species=None,
            d=np.asarray([0.4]), v_data=np.asarray([1e-6]), cells=None,
            carry_mask=None, note="synthetic"),
        r51.Statistic(
            key="charged_multiplicity_distribution", kind="distribution",
            species=None,
            d=np.full(len(r51.MULTIPLICITY_BINS), 1.0
                      / len(r51.MULTIPLICITY_BINS)),
            v_data=np.full(len(r51.MULTIPLICITY_BINS), 1e-6),
            cells=None, carry_mask=None, note="synthetic"),
    ]


# ---------------------------------------------------------------------------
# The fraction-sum constraint on a pinned fixture
# ---------------------------------------------------------------------------
def test_fraction_sum_constraint_pinned_fixture():
    """The fraction-sum constraint on the pinned fixture: the closure
    f_nd = 1 - sum is exact; the weights at a valid trial evaluate; a
    trial violating the sum rule (f_nd < 0) or a negative fraction
    returns an infinite objective and never a negative weight; the
    nominal identity (every weight exactly 1 at the initial point)
    holds on the fixture."""
    blocks = [_synthetic_block(0, 40, seed=11), _synthetic_block(1, 40,
                                                                 seed=12)]
    geometry = _synthetic_geometry()
    stats = _synthetic_stats()
    objective = r51.FitObjective(stats, blocks, geometry, r51.SQRT_S)
    objective.build_bin_structure()
    # the nominal identity: the log shift is exactly 0 and the ESS is
    # exactly 1 on every block (every weight exactly 1)
    at_initial = objective.evaluate(*r51.OPTIMIZER_SETTINGS["initial_point"])
    assert math.isfinite(at_initial["value"])
    assert at_initial["log_shift"] == 0.0
    assert all(v == 1.0 for v in at_initial["ess"]["per_block"])
    # the closure identity is exact at an interior trial
    x = (0.3, 0.2, 0.2, 0.1, 9.0)
    result = objective.evaluate(*x)
    assert result["fractions"]["nd"] == 1.0 - x[0] - x[1] - x[2]
    assert abs(result["fractions"]["nd"] - 0.3) < 1e-15
    assert result["fractions"]["nd"] > 0.0
    # a trial violating the sum rule returns an infinite objective
    result = objective.evaluate(0.5, 0.4, 0.2, 0.0, 9.0)  # sum = 1.1
    assert result["value"] == float("inf")
    assert result["fractions"]["nd"] < 0.0
    # a negative fraction returns an infinite objective
    result = objective.evaluate(-0.1, 0.3, 0.3, 0.0, 9.0)
    assert result["value"] == float("inf")
    # the optimizer's domain guard agrees
    assert r51._domain_ok(0.3, 0.2, 0.2) is True
    assert r51._domain_ok(0.5, 0.4, 0.2) is False
    assert r51._domain_ok(-0.1, 0.3, 0.3) is False


# ---------------------------------------------------------------------------
# The likelihood determinism on a pinned covariance
# ---------------------------------------------------------------------------
def test_likelihood_determinism_pinned_covariance():
    """The likelihood determinism on the pinned fixture: the profiled
    -2lnL reproduces bit for bit across repeated evaluations and across
    call orders, and the single-bin reduced form matches the
    generalized-least-squares profile computed independently in the
    test."""
    blocks = [_synthetic_block(0, 60, seed=21), _synthetic_block(1, 60,
                                                                 seed=22)]
    geometry = _synthetic_geometry()
    stats = _synthetic_stats()
    objective = r51.FitObjective(stats, blocks, geometry, r51.SQRT_S)
    objective.build_bin_structure()
    x = (0.22, 0.28, 0.21, 0.05, 9.4)
    a = objective.evaluate(*x)
    b = objective.evaluate(*x)
    assert a["value"] == b["value"]
    assert a["eta_hat"] == b["eta_hat"]
    c = objective.evaluate(x[4], x[3], x[2], x[1], x[0])  # a different trial
    assert c["value"] != a["value"] or x == x[::-1]
    # the single-bin GLS profile, computed independently: one bin with
    # d, sigma_data, sigma_mc, and one multiplicative normalization
    # nuisance (m -> m (1 + eta)) with the prior width w_prior.  With
    # w^2 = 1 / (sigma_data^2 + sigma_mc^2) the residual row is
    # (d - m) w and the design entry is m w, so the GLS optimum is
    # eta_hat = m w^2 (d - m) / (m^2 w^2 + 1/w_prior^2) and the profiled
    # value is the quadratic at the optimum, the residual term plus the
    # prior term: (d - m(1+eta))^2 w^2 + eta^2 / w_prior^2.
    d_value, sigma_data = 2.0, 0.5
    model, sigma_mc = 3.0, 0.5
    prior = r51.NORM_NUISANCE_PRIOR
    w_sq = 1.0 / (sigma_data ** 2 + sigma_mc ** 2)
    eta = (model * w_sq * (d_value - model)) / (
        model * model * w_sq + 1.0 / prior ** 2)
    independent = ((d_value - model * (1.0 + eta)) ** 2) * w_sq \
        + eta ** 2 / prior ** 2
    # the objective's nuisance profile solves the same equation: feed a
    # one-statistic one-bin objective and compare
    single_stat = [r51.Statistic(
        key="leading_proton_spectrum", kind="spectrum", species="proton",
        d=np.asarray([d_value]), v_data=np.asarray([sigma_data ** 2]),
        cells=None, carry_mask=None, note="synthetic")]
    single_geometry = {
        "edges": {}, "n_x": {}, "dense_size": {"proton": 1,
                                               "antiproton": 1},
        "pick": {"proton": np.zeros(1, dtype=np.int64),
                 "antiproton": np.zeros(1, dtype=np.int64)},
        "areas": {"proton": np.ones(1), "antiproton": np.ones(1)},
        "unit_structure": {
            "leading_proton_spectrum": np.ones(1, dtype=bool)},
        "_run_dir": None,
    }
    one_block = _synthetic_block(0, 10, seed=31)
    one_block.prot_flat = np.zeros(1, dtype=np.int64)
    one_block.prot_evt = np.zeros(1, dtype=np.int64)
    single_objective = r51.FitObjective(
        single_stat, [one_block], single_geometry, r51.SQRT_S)
    single_objective.build_bin_structure()
    # the model value is forced to `model` by construction: the fixture's
    # proton lands in cell 0 and the conversion divides by the unit area
    # and n_eff, so m = n_eff * w_0 / (pi sqrt_s n_eff) = ...  the test
    # instead verifies the PROFILE algebra directly through the private
    # method on a synthetic vector pair
    single_objective.kept["leading_proton_spectrum"] = np.ones(1, dtype=bool)
    value, eta_hat = single_objective._profiled_minus2lnl(
        [np.asarray([model])], [np.asarray([sigma_mc])])
    assert abs(eta_hat[0] - eta) < 1e-12
    assert abs(value - independent) < 1e-9
    # determinism of the machinery path: the same evaluation twice on
    # the full pinned fixture reproduces every pooled vector bit for bit
    for i in range(len(stats)):
        assert np.array_equal(a["pooled_vectors"][i],
                              b["pooled_vectors"][i])
        assert np.array_equal(a["se"][i], b["se"][i])


# ---------------------------------------------------------------------------
# The replica-stability algebra on a pinned fixture
# ---------------------------------------------------------------------------
def test_replica_stability_algebra_pinned_fixture():
    """The replica-stability algebra on a pinned fixture: the spread,
    the 20%-of-pooled-error bound, and the pass flag; an infinite pooled
    error (a non-positive curvature) fails the bound and is named."""
    fits = [{"replica": r, "minus2lnl": 0.0,
             "sd_projectile_fraction": 0.25 + 0.001 * r,
             "sd_target_fraction": 0.25,
             "dd_fraction": 0.25,
             "epsilon": 0.0,
             "b_sd": 8.5, "converged": True, "nfev": 10}
            for r in range(8)]
    pooled_errors = {
        "sd_projectile_fraction": 0.01, "sd_target_fraction": 0.01,
        "dd_fraction": 0.01, "epsilon": 0.01, "b_sd": 0.01}
    best_x = [0.25, 0.25, 0.25, 0.0, 8.5]
    table = r51._stability_table(best_x, fits, pooled_errors)
    values = [fit["sd_projectile_fraction"] for fit in fits]
    assert table["sd_projectile_fraction"]["spread"] == \
        max(values) - min(values)
    assert abs(table["sd_projectile_fraction"]["spread"] - 0.007) < 1e-15
    assert table["sd_projectile_fraction"]["bound"] == 0.002
    assert table["sd_projectile_fraction"]["pass"] is False
    assert table["sd_target_fraction"]["spread"] == 0.0
    assert table["sd_target_fraction"]["pass"] is True
    # an infinite pooled error fails the bound
    pooled_errors_inf = dict(pooled_errors)
    pooled_errors_inf["epsilon"] = float("inf")
    table2 = r51._stability_table(best_x, fits, pooled_errors_inf)
    assert table2["epsilon"]["pass"] is False
    assert math.isinf(table2["epsilon"]["bound"])
    # the boundary-parameter detector
    boundary = r51._boundary_parameters([0.0, 0.25, 0.25, 0.0, 8.5])
    assert [entry["parameter"] for entry in boundary] == [
        "sd_projectile_fraction"]
    boundary2 = r51._boundary_parameters([0.3, 0.3, 0.2, 0.5, 20.0])
    assert [entry["parameter"] for entry in boundary2] == ["epsilon", "b_sd"]
    # the quoted flag
    assert r51.quoted_flag(True, True, True, False) is True
    assert r51.quoted_flag(False, True, True, False) is False
    assert r51.quoted_flag(True, False, True, False) is False
    assert r51.quoted_flag(True, True, False, False) is False
    assert r51.quoted_flag(True, True, True, True) is False


# ---------------------------------------------------------------------------
# The analytic normalization identities of the weight machinery
# ---------------------------------------------------------------------------
def test_weight_normalization_identities():
    """The importance-weight machinery's analytic identities: the
    vectorized t-range mirror is bit-identical to the mass-law module's
    scalar t_range; the double-diffraction joint mass and the central
    mass integral match adaptive quadrature; the nominal identity (the
    log mass-law ratio exactly zero at the initial point) holds on
    synthetic class payloads."""
    from na49_mixing.topology_mass_laws import t_range

    rng = np.random.default_rng(7)
    m2s = np.exp(rng.uniform(np.log(0.88), np.log(260.0), 200))
    m_rems = np.asarray(rng.choice(
        [r51.PROTON_MASS, 1.3, 1.6, 2.2], 200), dtype=float)
    got = r51._t_range_min_vec(m2s, m_rems)
    for m2, mr, g in zip(m2s, m_rems, got):
        assert t_range(float(m2), float(mr), r51.SQRT_S)[0] == float(g)
    # the DD joint mass against the reduced one-dimensional quadrature
    import scipy.integrate as si

    mp = r51.PROTON_MASS
    s = r51.SQRT_S
    for eps in (-0.4, 0.0, 0.5):
        if abs(eps) < 1e-13:
            def integrand(u):
                return math.log((s - u) / mp) / u
        else:
            def integrand(u):
                j = ((s - u) ** (-2.0 * eps) - mp ** (-2.0 * eps)) / (
                    -2.0 * eps)
                return u ** (-1.0 - 2.0 * eps) * j
        value, _err = si.quad(integrand, mp, s - mp, limit=500,
                              epsabs=1e-12, epsrel=1e-12)
        reference = math.log(4.0 * value)
        assert abs(r51._log_dd_joint_mass(eps) - reference) < 1e-9
    # the central mass integral against quadrature
    a = r51.PI0_MASS ** 2
    c = (r51.SQRT_S - 2.0 * mp) ** 2
    for eps in (-0.3, 0.0, 0.3):
        for beta in (0.0, -2.0):
            if beta == 0.0:
                def integrand(x, e=eps):
                    return x ** (-1.0 - e)
            else:
                def integrand(x, e=eps, b=beta):
                    return x ** (-1.0 - e) * math.exp(b * x)
            value, _err = si.quad(integrand, a, c, limit=500,
                                  epsabs=1e-13, epsrel=1e-13)
            got = float(r51._central_mass_log_integral(
                np.asarray([eps]), np.asarray([beta]), a, c)[0])
            assert abs(math.exp(got) - value) <= 1e-8 * value
    # the nominal identity on synthetic class payloads
    fake_sd = {"m2": np.asarray([1.0, 30.0]), "t": np.asarray([-0.1, -0.5]),
               "m_rem": np.asarray([r51.PROTON_MASS, 1.3])}
    assert np.all(r51.class_log_law_weights(
        "sd_projectile", fake_sd, 0.0, r51.NOMINAL_B_SD) == 0.0)
    fake_dd = {"m12": np.asarray([2.0]), "m22": np.asarray([3.0]),
               "t1": np.asarray([-0.2]), "t2": np.asarray([-0.3])}
    assert np.all(r51.class_log_law_weights(
        "dd", fake_dd, 0.0, r51.NOMINAL_B_SD) == 0.0)
    fake_nd = {"t1": np.asarray([-0.2]), "t2": np.asarray([-0.3]),
               "t_c": np.asarray([-0.5]), "mc2": np.asarray([4.0]),
               "m1": np.asarray([r51.PROTON_MASS]),
               "m2": np.asarray([1.3])}
    assert np.all(r51.class_log_law_weights(
        "nd", fake_nd, 0.0, r51.NOMINAL_B_SD) == 0.0)


# ---------------------------------------------------------------------------
# The initial values against the sealed R50 (read-only; skipped on a
# fresh checkout without the artifact store)
# ---------------------------------------------------------------------------
def test_initial_values_against_sealed_r50():
    """The initial values are the sealed R50 nominal point exactly (the
    card's before-any-generation assertion; the R50 seal verifies)."""
    r50_dir = r51.MONO_ROOT / r51.R50_RUN_DIR
    if not (r50_dir / r51.R50_PREREGISTRATION_NAME).is_file():
        pytest.skip("the sealed R50 archive is absent on this checkout")
    status = r51.assert_initial_values_against_r50()
    assert status["match"] is True
    assert status["r50_nominal"]["fractions"] == {
        "sd_projectile": 0.25, "sd_target": 0.25, "dd": 0.25, "nd": 0.25}
    assert status["r50_nominal"]["epsilon"] == 0.0
    assert status["r50_nominal"]["b_sd_gev2"] == 8.504140762083612
    seal = r51.verify_sealed_dir(r51.R50_RUN_DIR)
    assert seal["intact"] is True


# ---------------------------------------------------------------------------
# The smoke-mode refusal conventions (the R31/8c convention)
# ---------------------------------------------------------------------------
def test_smoke_mode_refuses_default_run_dir():
    """--smoke refuses the default production run dir (the smoke
    artifacts are never sealed into the production run dir)."""
    default_dir = r51.MONO_ROOT / r51.DEFAULT_RUN_DIR
    with pytest.raises(SystemExit) as exc:
        r51.run_smoke(default_dir)
    assert exc.value.code == 2


def test_smoke_refusal_convention(tmp_path):
    """--gate / --seal / --clean-validate refuse smoke artifacts (the
    R31/8c convention); clean-validate additionally refuses the sealed
    dir (the card-19 seal-ordering discipline)."""
    run_dir = tmp_path / "smoke_run"
    run_dir.mkdir()
    (run_dir / r51.PREREGISTRATION_NAME).write_text(
        json.dumps({"mode": "smoke"}), encoding="utf-8")
    with pytest.raises(SystemExit):
        r51._refuse_smoke_artifacts(run_dir)
    run_dir2 = tmp_path / "prod_named"
    run_dir2.mkdir()
    (run_dir2 / r51.PREREGISTRATION_NAME).write_text(
        json.dumps({"mode": "production"}), encoding="utf-8")
    (run_dir2 / "smoke_replica.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        r51._refuse_smoke_artifacts(run_dir2)
    run_dir3 = tmp_path / "no_pre"
    run_dir3.mkdir()
    with pytest.raises(SystemExit):
        r51._refuse_smoke_artifacts(run_dir3)
    # the sealed dir refuses clean-validate
    run_dir4 = tmp_path / "sealed"
    run_dir4.mkdir()
    (run_dir4 / r51.PREREGISTRATION_NAME).write_text(
        json.dumps({"mode": "production"}), encoding="utf-8")
    (run_dir4 / r51.SEAL_NAME).write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        r51.run_clean_validate(run_dir4)


# ---------------------------------------------------------------------------
# The freeze re-pin register
# ---------------------------------------------------------------------------
MANIFEST_REL = "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json"
MY_TEST_REL = (
    "dev/mixing2006/tests/test_staged_r51_topology_fraction_fit.py")
SOLVER_CARD = "TASK-20260925-r5-run-r51-topology-fraction-fit"

#: The chain count-pin files re-pinned by this card (the 39 chain files
#: carrying the inventory count pin).
CHAIN_COUNT_PIN_FILES = (
    "dev/mixing2006/tests/test_cascade_ablation.py",
    "dev/mixing2006/tests/test_cascade_alignment_baryonic.py",
    "dev/mixing2006/tests/test_cascade_alignment_mesonic.py",
    "dev/mixing2006/tests/test_cascade_axes.py",
    "dev/mixing2006/tests/test_cascade_form_factor.py",
    "dev/mixing2006/tests/test_cascade_termination.py",
    "dev/mixing2006/tests/test_exchange_ledger_contract_pins.py",
    "dev/mixing2006/tests/test_r4_audit_blend_path_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_ablation_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_r4_cascade_alignment_baryonic_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_r4_cascade_alignment_mesonic_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_axes_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_decay_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_form_factor_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_termination_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r40_alignment_scan_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_staged_r41_nonstrange_pilot_fit_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r42_recoil_decay_separation.py",
    "dev/mixing2006/tests/"
    "test_staged_r42_recoil_decay_separation_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r43_cascade_ablation.py",
    "dev/mixing2006/tests/"
    "test_staged_r43_cascade_ablation_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r44_frozen_nonstrange_validation.py",
    "dev/mixing2006/tests/test_staged_r45_simplified_alignment_freeze.py",
    "dev/mixing2006/tests/"
    "test_staged_r45_simplified_alignment_freeze_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_staged_r46_simplified_recoil_decay_profile.py",
    "dev/mixing2006/tests/"
    "test_staged_r46_simplified_recoil_decay_profile_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r50_pure_topology_templates.py",
    "dev/mixing2006/tests/"
    "test_staged_r50_pure_topology_templates_contract_pins.py",
    "dev/mixing2006/tests/test_topology_double_diffraction_contract_pins.py",
    "dev/mixing2006/tests/test_topology_elastic_contract_pins.py",
    "dev/mixing2006/tests/test_topology_mass_laws_contract_pins.py",
    "dev/mixing2006/tests/test_topology_multi_neutral.py",
    "dev/mixing2006/tests/test_topology_multi_neutral_contract_pins.py",
    "dev/mixing2006/tests/test_topology_non_diffractive.py",
    "dev/mixing2006/tests/"
    "test_topology_non_diffractive_contract_pins.py",
    "dev/mixing2006/tests/test_topology_remnants_contract_pins.py",
    "dev/mixing2006/tests/test_topology_runner.py",
    "dev/mixing2006/tests/test_topology_runner_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_topology_single_diffraction_contract_pins.py",
)

#: This card's freeze-exception file list (path-sorted): the manifest,
#: the dual-tree driver, this test file, and the re-pinned chain count
#: pin files.
CARD_EXCEPTION_FILES = tuple(sorted((
    MANIFEST_REL,
    r51.DRIVER_REL_NESTED,
    r51.DRIVER_REL_OUTER,
    MY_TEST_REL,
    *CHAIN_COUNT_PIN_FILES,
)))

# re-pinned 532 -> 533 by Round 5 card 10 (the card's pre-registered
# literal; the operative growth followed the interim re-pins)
# re-pinned 533 -> 536 by Round 5 card 11 (TASK-20260925-r5-run-r51-
# topology-fraction-fit; +3 entries: the dual-tree driver
# helpers/benchmarking/staged_r51_topology_fraction_fit.py, its nested
# copy, and the test file tests/test_staged_r51_topology_fraction_fit.py)
# re-pinned 536 -> 537 by Round 5 card 11 test-author (TASK-20260925-r5-run-r51-topology-fraction-fit-test-author; +1 entry: the card-11 test-author register file tests/test_staged_r51_topology_fraction_fit_contract_pins.py).
# re-pinned 537 -> 540 by Round 5 card 12 (TASK-20260925-r5-run-r52-central-excitation-necessity; +3 entries: the dual-tree driver dev/mixing2006/helpers/benchmarking/staged_r52_central_excitation_necessity.py, its outer mirror helpers/benchmarking/staged_r52_central_excitation_necessity.py, and the test file dev/mixing2006/tests/test_staged_r52_central_excitation_necessity.py; the card's pre-registered 523 -> 526 literal is noted, the operative growth followed)
# re-pinned 540 -> 541 by Round 5 card 12 test-author (TASK-20260925-r5-run-r52-central-excitation-necessity-test-author; +1 entry: the card-12 test-author register file tests/test_staged_r52_central_excitation_necessity_contract_pins.py).
# re-pinned 541 -> 544 by Round 5 card 13 (TASK-20260925-r5-run-r53-multiple-neutral-excitation; +3 entries: the dual-tree driver dev/mixing2006/helpers/benchmarking/staged_r53_multiple_neutral_excitation.py, its outer mirror helpers/benchmarking/staged_r53_multiple_neutral_excitation.py, and the test file tests/test_staged_r53_multiple_neutral_excitation.py; the card's pre-registered 526 -> 529 literal is noted, the operative growth followed).
# re-pinned 544 -> 547 by Round 5 card 14 (TASK-20260925-r5-run-r54-local-ledger-audit; +3 entries: the dual-tree driver dev/mixing2006/helpers/benchmarking/staged_r54_local_ledger_audit.py, its outer mirror helpers/benchmarking/staged_r54_local_ledger_audit.py, and the test file tests/test_staged_r54_local_ledger_audit.py; the card's pre-registered 529 -> 532 literal is noted, the operative growth followed).
# re-pinned 547 -> 548 by Round 5 card 13 test-author (TASK-20260925-r5-run-r53-multiple-neutral-excitation-test-author; +1 entry: the card-13 test-author register file tests/test_staged_r53_multiple_neutral_excitation_contract_pins.py).
# re-pinned 548 -> 551 by Round 5 card 15 (TASK-20260925-r5-run-r55-frozen-topology-validation; +3 entries: the dual-tree driver dev/mixing2006/helpers/benchmarking/staged_r55_frozen_topology_validation.py, its outer mirror helpers/benchmarking/staged_r55_frozen_topology_validation.py, and the test file tests/test_staged_r55_frozen_topology_validation.py; the card's pre-registered 532 -> 535 literal is noted, the operative growth followed).
INVENTORY_COUNT = _inventory_count()
M8_HASH = (
    "358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1")
KERNEL_BOUNDARY = 2.680194110727528


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_of(rel):
    return (r51.MONO_ROOT / rel).read_text(encoding="utf-8")




def test_layout_sync():
    """The layout sync (the subprocess check from the monorepo root;
    ``logic_drift = 0``)."""
    checker = r51.MONO_ROOT / "helpers" / "benchmarking" / \
        "check_helpers_layout_sync.py"
    probe = subprocess.run(
        [sys.executable, str(checker)],
        capture_output=True, text=True, cwd=str(r51.MONO_ROOT))
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert "logic_drift=0" in probe.stdout
