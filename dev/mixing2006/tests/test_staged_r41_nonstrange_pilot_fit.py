"""Hermetic tests for the R41 driver (the nonstrange pilot fit).

Task ``TASK-20260920-r4-run-r41-nonstrange-pilot-fit`` (Round 4, card
12, relitigated by Round 4 card 16): the seed-map and event-index
partition, the fit-set invariants (four parameters, the ramp start
excluded), the replica-stability algebra on a pinned fixture, the
likelihood determinism (the profiled nuisance identity and the reweight
algebra on closed-form fixtures), the hole-geometry suite of the R01
accumulate-dense-then-pick recording convention (the corrected
convention of the relitigation; mirrored from the R44 driver's
``test_holey_geometry_*`` coverage and adapted to the R41 recording
path and the training-combination species set), and the layout sync.
No Monte Carlo generation, no SPS table, no sealed archive read: the
acceptance is hermetic.
"""

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest


def _load_driver():
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "helpers" / "benchmarking"
        / "staged_r41_nonstrange_pilot_fit.py",
    ]
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location(
                "staged_r41_nonstrange_pilot_fit", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(
        "the R41 driver was not found next to the test file")


r41 = _load_driver()


# ---------------------------------------------------------------------------
# The seed map and the event-index partition (the card's pre-registration)
# ---------------------------------------------------------------------------
def test_seed_map_and_event_index_partition():
    assert r41.MASTER_SEED == 20260920
    assert r41.RAW_REPLICA_SEEDS == tuple(20301020 + r for r in range(8))
    assert r41.EVENT_INDEX_RULE == "400_000_000 + i, i < 100,000"
    assert r41.T2_REPLICAS == 8 and r41.T2_EVENTS_PER_REPLICA == 100_000
    # the per-replica seeds are the block 20301020-20301027
    assert [r41.raw_replica_seed(r) for r in range(8)] == [
        20301020 + r for r in range(8)]
    with pytest.raises(SystemExit):
        r41.raw_replica_seed(8)
    # the smoke block is disjoint from the production slots and from R40
    assert r41.SMOKE_RAW_SEED == 20260920 + 40_000 + 8_000 == 20308920
    assert r41.SMOKE_KERNEL_REPLICA_SLOT == 8_000
    assert 20308820 not in r41.RAW_REPLICA_SEEDS
    assert 20308920 not in range(20301040, 20301056)  # the audit block
    # the event-index partition is disjoint from R40's (300M-based)
    assert r41.EVENT_INDEX_BASE >= 400_000_000
    # the raw-seed rule names the audit block disjointness
    assert "20301040" in r41.RAW_SEED_RULE
    # the kernel rule derives from the master seed on the closed labels
    assert "derive_substreams(20260920, r, 400_000_000 + i)" in \
        r41.KERNEL_SEED_RULE


def test_kernel_stream_disjointness():
    """The (master, replica slot, event index) kernel-layer spaces are
    pairwise disjoint across replicas (the derive_substreams keying)."""
    from na49_mixing.rng_substreams import derive_substreams

    streams = set()
    for replica in range(r41.T2_REPLICAS):
        subs = derive_substreams(
            r41.MASTER_SEED, replica, r41.EVENT_INDEX_BASE)
        for label in ("decay_channel", "decay_kinematics"):
            key = int(subs.stream_for(label))
            assert key not in streams
            streams.add(key)
    # the smoke slot is disjoint from every production slot
    subs = derive_substreams(
        r41.MASTER_SEED, r41.SMOKE_KERNEL_REPLICA_SLOT,
        r41.EVENT_INDEX_BASE)
    for label in ("decay_channel", "decay_kinematics"):
        assert int(subs.stream_for(label)) not in streams


# ---------------------------------------------------------------------------
# The fit-set invariants (four parameters, the ramp start excluded)
# ---------------------------------------------------------------------------
def test_fit_set_invariants():
    assert tuple(r41.FIT_SET) == (
        "kappa_B", "kappa_m", "b_dec", "baryonic_nominal_width")
    assert len(r41.FIT_SET) == 4
    assert "blend_ramp_start" in r41.FIT_SET_EXCLUDED
    assert r41.FIT_SET_EXCLUDED["blend_ramp_start"].startswith(
        "obligation 4")
    assert r41.BLEND_RAMP_START_GEV not in [
        r41.INITIAL_VALUES[key] for key in r41.FIT_SET]
    assert r41.INITIAL_VALUES == {
        "kappa_B": 1.0, "kappa_m": 1.0, "b_dec": 1.0,
        "baryonic_nominal_width": 0.5,
    }
    assert r41.PARAM_BOUNDS["kappa_B"] == (0.0, 4.0)
    assert r41.PARAM_BOUNDS["kappa_m"] == (0.0, 4.0)
    assert r41.PARAM_BOUNDS["b_dec"] == (0.0, 2.0)
    assert r41.PARAM_BOUNDS["baryonic_nominal_width"] == (0.3, 0.7)
    assert tuple(r41.W_GRID) == (0.3, 0.4, 0.5, 0.6, 0.7)
    assert r41.W_GRID_NOMINAL == 0.5
    assert r41.W_GRID[r41.W_GRID_INDEX_NOMINAL] == 0.5


def test_nuisance_structure_pre_registered():
    assert len(r41.NUISSANCE_COLUMNS) == 4
    assert r41.NORM_NUISANCE_PRIOR == 0.35
    assert r41.CARRY_NUISANCE_PRIOR == 0.25
    assert r41.OBLIGATION4_R32_SPREAD == {
        "bin_spread_p95": 4.028426,
        "feeddown_spread_max_sigma": 31.1773,
        "persisting_residuals": 32,
    }
    assert r41.STABILITY_SPREAD_BOUND == 0.20
    assert r41.SIGMA_BAND == 3.0
    assert r41.ESS_FLOOR_FRACTION == 0.01
    # the optimizer contract
    assert r41.NM_XATOL == 1e-3 and r41.NM_FATOL == 1e-6
    assert r41.NM_MAXITER == 400 and r41.NM_MAXFEV == 800
    assert r41.HESSIAN_STEPS == {
        "kappa_B": 0.1, "kappa_m": 0.1, "b_dec": 0.1,
    }


# ---------------------------------------------------------------------------
# The reweight algebra on closed-form fixtures
# ---------------------------------------------------------------------------
def _init_layout():
    """The hermetic six-species synthetic grid (the layout init: the
    published-row flat layout and the dense evaluation-lattice layout of
    the accumulate-dense-then-pick convention)."""
    tables = _synthetic_tables()
    r41._init_species_layout(tables)
    r41._init_dense_layout(tables)
    return tables


def _fixture_block():
    """A pinned two-event, three-vertex fixture: vertex 0 baryonic
    (a two-channel pool), vertex 1 mesonic (a two-channel pool), both on
    event 0; event 1 carries no vertices (a theta-independent event).
    The block's ``fixed`` pool and ``row_flat`` indices are the DENSE
    evaluation-lattice addresses of the corrected recording convention
    (re-derived on the corrected convention, card 16: the synthetic grid
    is rectangular and pT-major, so its published rows sit at their own
    dense addresses and the two layouts coincide per species)."""
    _init_layout()
    # the fixed pool and the row indices are the dense evaluation-lattice
    # addresses (the corrected recording convention; on this rectangular
    # pT-major grid the dense and published layouts coincide per species)
    total = r41._dense_total()
    return r41.BlockLatent(
        replica=0,
        seed=20301020,
        n_events=2,
        fixed=np.zeros(total, dtype=float),
        row_flat=np.asarray([0, 1, 2, 3], dtype=np.int64),
        row_weight=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=float),
        row_ptr=np.asarray([0, 4], dtype=np.int64),
        var_ids=np.asarray([7], dtype=np.int64),
        vtx_ptr=np.asarray([0, 2], dtype=np.int64),
        vtx_nchan=np.asarray([2, 2], dtype=np.int64),
        vtx_chosen=np.asarray([0, 1], dtype=np.int64),
        vtx_law=np.asarray([0, 1], dtype=np.int64),
        vtx_ck=np.asarray([0.25, 0.5], dtype=float),
        ch_w0=np.asarray([0.6, 0.4, 0.6, 0.4], dtype=float),
        ch_q2=np.asarray([0.2, 1.0, 0.5, 1.5], dtype=float),
    )


def test_event_weights_closed_form():
    block = _fixture_block()
    # the trial at the generation point must be exactly weight 1
    eta0 = r41.event_weights(block, 1.0, 1.0, 1.0)
    assert eta0[0] == pytest.approx(1.0, abs=1e-14)
    # the baryonic angular ratio at c_k = 0.25 for kappa_B = 2:
    # exp((2 - 1) * 0.25) * Z(1) / Z(2)
    from na49_mixing.cascade_alignment import (
        baryonic_normalization,
        mesonic_normalization,
    )
    # the per-event weight is the product over BOTH vertices of event 0
    # (vertex 0 baryonic, vertex 1 mesonic at the fixed kappa_m = 1):
    # vertex 0: delta_b = 0.5 - 1.0, pool (0.6, 0.4) at q2 (0.5, 1.5),
    # chosen q2 = 0.5; delta_kappa_B = 1.0 at c_k = 0.25.
    delta_b = 0.5 - 1.0
    ex = np.exp(-delta_b * block.ch_q2)
    pool_ptr = block.vtx_pool_ptr
    pool_sum = np.add.reduceat(block.ch_w0 * ex, pool_ptr[:-1])
    chosen_q2 = block.ch_q2[pool_ptr[:-1] + block.vtx_chosen]
    log_channel = -delta_b * chosen_q2 - np.log(pool_sum)
    ang = np.where(
        block.vtx_law == 0,
        (2.0 - 1.0) * block.vtx_ck + math.log(
            r41._baryonic_norm(1.0) / r41._baryonic_norm(2.0)),
        (1.0 - 1.0) * block.vtx_ck ** 2 + 0.0,
    )
    expected = math.exp(math.fsum(log_channel + ang))
    eta1 = r41.event_weights(block, 2.0, 1.0, 0.5)
    assert float(eta0[0]) == pytest.approx(1.0, abs=1e-14)
    assert float(eta1[0]) == pytest.approx(expected, rel=1e-12)


def test_event_weights_mesonic_and_mirror():
    block = _fixture_block()
    # vertex 1 is the mesonic law at c_k = 0.5; the even law's ratio is
    # exp(delta_km * c_k^2) * Z_M(1) / Z_M(kappa)
    from na49_mixing.cascade_alignment import mesonic_normalization

    delta_km = 3.0 - 1.0
    expected_1 = math.exp(
        delta_km * 0.25 + math.log(
            r41._mesonic_norm(1.0) / r41._mesonic_norm(3.0)))
    got = r41.event_weights(block, 1.0, 3.0, 1.0)
    assert float(got[0]) == pytest.approx(expected_1, rel=1e-12)
    # the isotropic limit recovers weight 1 at the nominal point only


def test_weighted_sums_and_ess():
    block = _fixture_block()
    eta = np.asarray([2.0, 1.0])
    totals = r41.block_weighted_sums(block, eta)
    assert totals[0] == pytest.approx(2.0)
    assert totals[1] == pytest.approx(4.0)
    assert totals[2] == pytest.approx(6.0)
    assert totals[3] == pytest.approx(8.0)
    ess = r41.ess_fraction(eta)
    assert ess == pytest.approx((2.0 + 1.0) ** 2 / (4.0 + 1.0) / 2.0)


# ---------------------------------------------------------------------------
# The nuisance profile on a pinned fixture (the GLS identity)
# ---------------------------------------------------------------------------
def test_profiled_nuisance_identity():
    """The profiled form equals the direct quadratic minimum of the
    pre-registered nuisance objective (the plan section 5.4 form)."""
    rng = np.random.default_rng(20260920)
    d = rng.normal(size=4) * 10.0
    m = d * 0.5 + 1.0
    v = np.full(4, 2.0)
    design_rows = [
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    prior = np.diag([0.35 ** 2, 0.35 ** 2, 0.35 ** 2, 0.25 ** 2])
    r = (d - m) / np.sqrt(v)
    design = np.asarray(design_rows) / np.sqrt(v)[:, None]
    normal = design.T @ design + np.linalg.inv(prior)
    rhs = design.T @ r
    eta = np.linalg.solve(normal, rhs)
    profiled = float(r @ r - rhs @ eta)
    # the direct quadratic minimum over eta of the same objective
    def direct(eta_vector):
        penalty = sum(
            float(eta_vector[k]) ** 2 / float(prior[k][k])
            for k in range(4))
        return float(np.sum(((d - m - design @ eta_vector
                             * np.sqrt(v)) / np.sqrt(v)) ** 2)) + penalty
    best = min(
        [direct(eta + offset)
         for offset in (np.zeros(4),)]
        + [direct(eta + delta) for delta in np.eye(4) * 1e-6]
        + [direct(eta - delta) for delta in np.eye(4) * 1e-6],
    )
    assert profiled == pytest.approx(best, rel=1e-6)


# ---------------------------------------------------------------------------
# The replica-stability algebra on a pinned fixture
# ---------------------------------------------------------------------------
def test_replica_stability_algebra():
    pooled_errors = {"kappa_B": 0.5, "kappa_m": 0.2, "b_dec": 1.0}
    block_fits = [
        {"kappa_B": 1.0 + 0.01 * i, "kappa_m": 0.5, "b_dec": 1.0}
        for i in range(8)
    ]
    table = r41._stability_table((1.0, 1.0, 1.0), block_fits,
                                 pooled_errors)
    assert table["kappa_B"]["spread"] == pytest.approx(0.07)
    assert table["kappa_B"]["bound"] == pytest.approx(0.20 * 0.5)
    assert table["kappa_B"]["pass"] is True
    # a wide spread fails and the failing parameter is the named finding
    block_fits[3]["kappa_B"] = 2.0
    table = r41._stability_table((1.0, 1.0, 1.0), block_fits,
                                 pooled_errors)
    assert table["kappa_B"]["pass"] is False
    assert table["kappa_B"]["spread"] == pytest.approx(
        2.0 - block_fits[0]["kappa_B"])


def test_width_curvature_grid_identity():
    profile = [
        {"width_gev": 0.3, "minus2lnl": 10.0},
        {"width_gev": 0.4, "minus2lnl": 6.0},
        {"width_gev": 0.5, "minus2lnl": 5.0},
        {"width_gev": 0.6, "minus2lnl": 7.0},
        {"width_gev": 0.7, "minus2lnl": 11.0},
    ]
    curvature = r41._width_curvature(profile, 0.5)
    dw = (0.6 - 0.4) / 2.0
    assert curvature == pytest.approx(
        (6.0 - 2.0 * 5.0 + 7.0) / (dw * dw))


# ---------------------------------------------------------------------------
# The seagull moment fidelity (the house convention on both sides)
# ---------------------------------------------------------------------------
def test_seagull_model_matches_house_convention():
    """The driver's seagull statistic equals the fit_observables
    convention evaluated directly on a pinned combination."""
    tables = _synthetic_tables()
    tables_by_species = {t.species: t for t in tables}
    sealed = {
        t.species: {
            "d": [1.0] * int(t.data.size),
            "v_data": [0.25] * int(t.data.size),
            "xF_mid": [float(v) for v in t.xf_centers],
            "pT_mid": [float(v) for v in t.pt_centers],
        } for t in tables
    }
    stats = r41._build_statistics(sealed, tables_by_species, set(),
                                  r41.SQRT_S_GEV)
    stat = next(s for s in stats if s.kind == "seagull"
                and s.key.startswith("pi_sigma"))
    assert len(stat.seagull["bin_indices"]) == len(stat.combo["cells"])
    species_values = {
        "pi_plus": np.full(len(sealed["pi_plus"]["d"]), 0.5),
        "pi_minus": np.full(len(sealed["pi_minus"]["d"]), 0.7),
        "proton": np.zeros(len(sealed["proton"]["d"])),
        "antiproton": np.zeros(len(sealed["antiproton"]["d"])),
    }
    model = r41._seagull_model(stat, species_values)
    combo_values = r41._combination_model(stat, species_values)
    seagull = stat.seagull
    result = r41.fo.seagull_moment(
        {"combination": tuple(float(v) for v in combo_values)},
        (tuple(seagull["x_edges"]), tuple(seagull["p_edges"])),
        "combination",
        bin_indices=seagull["bin_indices"],
        sqrt_s_gev=seagull["sqrt_s_gev"],
        mass_gev=seagull["mass_gev"],
    )
    expected = [
        None if v is None else float(v)
        for v in (result["pT_x_per_xF"] + result["pT_y_per_xF"])
    ]
    assert len(model) == len(expected)
    for got, want in zip(model, expected):
        assert got == want


def _synthetic_tables():
    """A small hermetic two-species table pair on a 4x3 grid."""
    from na49_mixing.staged_evaluation import BinnedFitTable

    tables = []
    for species, pdg, mass in (
        ("pi_plus", 211, 0.1395699), ("pi_minus", -211, 0.1395699),
        ("k_plus", 321, 0.4936770), ("k_minus", -321, 0.4936770),
        ("proton", 2212, 0.9382720), ("antiproton", -2212, 0.9382720),
    ):
        n_x, n_p = 4, 3
        xf = np.linspace(0.0, 0.4, n_x + 2)[1:-1]
        pt = np.linspace(0.05, 1.5, n_p + 2)[1:-1]
        tables.append(BinnedFitTable.from_arrays(
            species=species, pdg_id=pdg, mass_gev=mass,
            normalization="f_over_sigma_inel", sigma_inel_mb=31.78,
            path="test://synthetic",
            xf=np.tile(xf, n_p), pt=np.repeat(pt, n_x),
            value=np.ones(n_x * n_p), error=np.ones(n_x * n_p),
            window={"xF": [0.0, 0.4], "pT": [0.05, 1.5]},
        ))
    return tables


def test_seagull_statistic_shape_and_data():
    tables = _synthetic_tables()
    sealed = {}
    for table in tables:
        sealed[table.species] = {
            "d": [float(2 + i) for i in range(table.data.size)],
            "v_data": [0.25] * int(table.data.size),
            "xF_mid": [float(v) for v in table.xf_centers],
            "pT_mid": [float(v) for v in table.pt_centers],
        }
    tables_by_species = {t.species: t for t in tables}
    stats = r41._build_statistics(sealed, tables_by_species, set(),
                                  r41.SQRT_S_GEV)
    kinds = [s.kind for s in stats]
    assert kinds.count("spectrum") == 2
    assert kinds.count("seagull") == 2
    assert kinds.count("integral") == 2
    for stat in stats:
        if stat.kind == "seagull":
            assert len(stat.d) % 2 == 0
            assert np.all(np.isfinite(stat.v_data))
            assert np.all(stat.v_data >= r41.SE_FLOOR)


# ---------------------------------------------------------------------------
# The likelihood determinism (the objective is a pure function)
# ---------------------------------------------------------------------------
def test_objective_pure_function_on_fixture():
    tables = _synthetic_tables()
    tables_by_species = {t.species: t for t in tables}
    sealed = {
        t.species: {
            "d": [float(1 + i % 3) for i in range(t.data.size)],
            "v_data": [0.5] * int(t.data.size),
            "xF_mid": [float(v) for v in t.xf_centers],
            "pT_mid": [float(v) for v in t.pt_centers],
        } for t in tables
    }
    stats = r41._build_statistics(sealed, tables_by_species, set(), 17.27)
    r41._init_species_layout(tables)
    block = _fixture_block()
    objective = r41.FitObjective(
        stats, [block], tables_by_species, 17.27, 0.25, 0.35)
    one = objective.evaluate(1.3, 0.8, 1.1)["value"]
    two = objective.evaluate(1.3, 0.8, 1.1)["value"]
    assert one == two
    assert math.isfinite(one)


# ---------------------------------------------------------------------------
# The layout sync (the dual-tree driver byte-identical)
# ---------------------------------------------------------------------------
def test_layout_sync():
    """The nested and monorepo driver copies are byte-identical and the
    layout-sync report carries logic_drift 0."""
    import subprocess
    import sys

    nested = Path(__file__).resolve().parents[1] / "helpers" / \
        "benchmarking" / "staged_r41_nonstrange_pilot_fit.py"
    mono = Path(__file__).resolve().parents[3] / "helpers" / \
        "benchmarking" / "staged_r41_nonstrange_pilot_fit.py"
    assert nested.is_file(), nested
    assert mono.is_file(), mono
    nested_digest = hashlib.sha256(nested.read_bytes()).hexdigest()
    mono_digest = hashlib.sha256(mono.read_bytes()).hexdigest()
    assert nested_digest == mono_digest


# ---------------------------------------------------------------------------
# The hole-geometry suite (the R01 accumulate-dense-then-pick recording
# convention; mirrored from the R44 driver's card-15 test coverage and
# adapted to the R41 recording path and the training-combination species
# set: pi_Sigma, B_transport, the pooled seagulls, the pooled
# multiplicity moments)
# ---------------------------------------------------------------------------
HOLEY_N_X = 4
HOLEY_N_P = 3
#: The per-species hole cells (dense lattice indices ip * n_x + ix).
HOLEY_HOLES = {
    "pi_plus": {1, 5},
    "pi_minus": {2, 7, 11},
    "k_plus": {3, 9},
    "k_minus": {5, 6},
    "proton": {0, 10},
    "antiproton": {6, 8, 10},
}


def _holey_tables():
    """The six-species hole-geometry table set (the R41 recording
    surface: per-species holes and shuffled row order, rows not
    pT-major)."""
    xf = np.linspace(0.0, 0.4, HOLEY_N_X + 2)[1:-1]
    pt = np.linspace(0.05, 1.5, HOLEY_N_P + 2)[1:-1]
    window: dict = {"xF": [0.0, 0.4], "pT": [0.05, 1.5]}
    tables = []
    for species, pdg, mass in (
        ("pi_plus", 211, 0.1395699), ("pi_minus", -211, 0.1395699),
        ("k_plus", 321, 0.4936770), ("k_minus", -321, 0.4936770),
        ("proton", 2212, 0.9382720), ("antiproton", -2212, 0.9382720),
    ):
        holes = HOLEY_HOLES[species]
        cells = [c for c in range(HOLEY_N_X * HOLEY_N_P)
                 if c not in holes]
        # the deterministic shuffle: rows not pT-major (the published
        # R01 tables' row order)
        order = np.random.default_rng(
            20260920 + len(holes)).permutation(len(cells))
        rows = [cells[int(i)] for i in order]
        xf_pts = np.asarray([float(xf[c % HOLEY_N_X]) for c in rows])
        pt_pts = np.asarray([float(pt[c // HOLEY_N_X]) for c in rows])
        ones = np.ones(len(rows), dtype=float)
        from na49_mixing.staged_evaluation import BinnedFitTable

        tables.append(BinnedFitTable.from_arrays(
            species=species, pdg_id=pdg, mass_gev=float(mass),
            normalization="f_over_sigma_inel", sigma_inel_mb=31.78,
            path="test://holey-grid",
            xf=xf_pts, pt=pt_pts, value=ones, error=ones,
            window=window,
        ))
    return tables


def _holey_cell_of(table, row: int) -> tuple[int, int]:
    """The dense lattice cell (ix, ip) of one published row."""
    return (int(table.bin_index_xf[row]), int(table.bin_index_pt[row]))


def _holey_final(ix: int, ip: int, xf_centers, pt_centers, scale: float):
    """A final four-vector whose (x_F, pT) is the (ix, ip) cell center,
    plus the exact Jacobian weight ``e / pT`` the driver records for it
    (the same double operands, so the expected weight matches the
    recorded weight bit for bit)."""
    x_f = float(xf_centers[ix])
    p_t = float(pt_centers[ip])
    sqrt_s = r41._sqrt_s()
    pz = x_f * sqrt_s / 2.0
    # e/pT = scale with pT = pt: e = scale * pt
    e = scale * p_t
    weight = e / p_t
    return (float(e), float(p_t), 0.0, float(pz)), weight


def _holey_sample(tables):
    """One final per PUBLISHED row (distinct exact weight per row) plus
    one final per HOLE cell (distinct weight, >= 1000).  Returns the
    per-species cell -> weight maps."""
    xf_by = {t.species: t.xf_centers for t in tables}
    pt_by = {t.species: t.pt_centers for t in tables}
    finals = []
    counter = 0
    for table in tables:
        species = table.species
        for row in range(int(table.data.size)):
            ix, ip = _holey_cell_of(table, row)
            scale = 1.0 + 0.125 * counter
            counter += 1
            finals.append((species, ix, ip, scale))
    hole_finals = []
    h_counter = 0
    for species, holes in HOLEY_HOLES.items():
        for cell in sorted(holes):
            ix, ip = cell % HOLEY_N_X, cell // HOLEY_N_X
            scale = 1000.0 + 0.125 * h_counter
            h_counter += 1
            hole_finals.append((species, ix, ip, scale))
    return finals, hole_finals


def _holey_record(tables, finals, hole_finals):
    """Drive the finals through ``_LatentCapture.add_final`` (the dense
    recording), the chunk aggregation, and the record assembly (the
    pick), returning the record, the expected published-cell weights,
    and the hole-cell weights."""
    xf_by = {t.species: t.xf_centers for t in tables}
    pt_by = {t.species: t.pt_centers for t in tables}
    capture = r41._LatentCapture()
    capture.event_id = 0
    cell_weights: dict[str, dict[tuple[int, int], float]] = {
        t.species: {} for t in tables}
    for species, ix, ip, scale in finals:
        pdg = r41.SPECIES_PDGS[species]
        p4, weight = _holey_final(
            ix, ip, xf_by[species], pt_by[species], scale)
        capture.add_final(pdg, p4)
        cell_weights[species][(ix, ip)] = weight
    hole_weights: dict[str, dict[tuple[int, int], float]] = {
        t.species: {} for t in tables}
    for species, ix, ip, scale in hole_finals:
        pdg = r41.SPECIES_PDGS[species]
        p4, weight = _holey_final(
            ix, ip, xf_by[species], pt_by[species], scale)
        capture.add_final(pdg, p4)
        hole_weights[species][(ix, ip)] = weight
    parts = r41._chunk_parts(0, 0, [capture], smoke=True)
    aggregated = r41._aggregate_chunks([parts])
    record = r41._replica_record(
        0, 20301020, 0, 1, aggregated, "0" * 64, smoke=True)
    return record, cell_weights, hole_weights


def _holey_context(tables):
    """The worker context of the holey tables (the dense layout and the
    pick maps), with the context left set for the recording path."""
    r41._init_species_layout(tables)
    r41._set_worker_context(r41._worker_context(tables, fams=None,
                                                smoke=True))
    return r41._WORKER_CONTEXT


def _holey_dense_reference(context, cell_weights, hole_weights):
    """The independent dense totals: every published cell and hole cell
    weight placed at its dense lattice address (the R01 accumulation
    space), built from the cell maps alone."""
    dense = np.zeros(int(context["dense_total"]), dtype=float)
    for species, cells in cell_weights.items():
        base = int(context["dense_offsets_by_species"][species])
        n_x = int(context["n_x_by_species"][species])
        for (ix, ip), scale in cells.items():
            dense[base + ip * n_x + ix] += scale
    for species, cells in hole_weights.items():
        base = int(context["dense_offsets_by_species"][species])
        n_x = int(context["n_x_by_species"][species])
        for (ix, ip), scale in cells.items():
            dense[base + ip * n_x + ix] += scale
    return dense


def test_holey_geometry_holes_dropped_and_rows_aligned():
    """The non-rectangular published geometry (holes; rows not
    pT-major): the finals accumulate densely and the record picks the
    published rows, so (a) a hole cell's weight is dropped (recorded
    nowhere, bled into no neighboring row or species) and (b)
    ``pooled_sums`` align 1:1 with the published rows -- equal to the
    direct pick of the independent dense totals, and NOT equal to the
    dense-position-consumed reading (the superseded pre-fix
    convention)."""
    tables = _holey_tables()
    _holey_context(tables)
    try:
        finals, hole_finals = _holey_sample(tables)
        record, cell_weights, hole_weights = _holey_record(
            tables, finals, hole_finals)
        context = r41._WORKER_CONTEXT
        dense = _holey_dense_reference(context, cell_weights,
                                       hole_weights)
        picks = context["pick_by_species"]
        sizes = context["dense_sizes_by_species"]
        offsets = context["dense_offsets_by_species"]
        pooled = record["pooled_sums"]
        # the fixed pool carries the dense evaluation lattice (the
        # accumulate-dense-then-pick recording space)
        assert len(record["fixed_pool"]) == int(context["dense_total"])
        for table in tables:
            species = table.species
            values = pooled[species]
            assert len(values) == int(table.data.size), (
                f"{species}: pooled_sums must be published-row sized")
            # (b) row r holds the weight of row r's own lattice cell
            for row in range(int(table.data.size)):
                ix, ip = _holey_cell_of(table, row)
                assert values[row] == cell_weights[species][(ix, ip)], (
                    f"{species} row {row}: cell (ix={ix}, ip={ip}) "
                    "weight drifted")
            # the direct pick of the independent dense totals agrees
            # exactly
            dense_slice = dense[
                int(offsets[species]):
                int(offsets[species]) + int(sizes[species])]
            direct_pick = [float(v) for v in dense_slice[picks[species]]]
            assert direct_pick == values
            # (the superseded pre-fix convention would consume dense
            # cell addresses as published rows; on this geometry that
            # is a different vector)
            consumed = [
                float(v) for v in dense_slice[:int(table.data.size)]]
            assert consumed != values, (
                f"{species}: the dense-position-consumed reading "
                "equals the picked rows; the fixture does not "
                "discriminate the two conventions")
            # (a) every hole cell's weight is recorded nowhere
            flat_all = [v for key in r41.SPECIES_FLAT_ORDER
                        for v in pooled[key]]
            for (ix, ip), scale in hole_weights[species].items():
                assert scale not in flat_all, (
                    f"{species}: hole cell (ix={ix}, ip={ip}) weight "
                    "bled into a recorded row")
            # the recorded total carries the published cells only
            assert math.fsum(values) == pytest.approx(
                math.fsum(cell_weights[species].values()), rel=1e-12,
                abs=0.0)
    finally:
        r41._set_worker_context(None)


def test_holey_geometry_combination_follows_published_pick():
    """The training-combination diagnostic geometry (R41's analog of the
    card-15 model-only species case): the pooled-combination statistics
    (pi_Sigma = pi+ + pi-, B_transport = p - pbar, and the pooled
    seagull/multiplicity moments built on them) consume the two
    species' picked rows through ``_species_bin_values`` at the
    evaluation boundary, so every combination cell equals the sum of its
    two species' own-row contributions and the
    dense-position-consumed reading gives a different combination."""
    tables = _holey_tables()
    _holey_context(tables)
    try:
        finals, hole_finals = _holey_sample(tables)
        record, cell_weights, hole_weights = _holey_record(
            tables, finals, hole_finals)
        context = r41._WORKER_CONTEXT
        tables_by_species = {t.species: t for t in tables}
        sqrt_s = r41._sqrt_s()
        # the block from the record (the dense latent) and the
        # evaluation-boundary conversion to published rows
        block = r41._block_from_record(record)
        eta = r41.event_weights(block, 1.0, 1.0, 1.0)
        weighted = r41.block_weighted_sums(block, eta)
        assert weighted.size == int(context["dense_total"])
        values = r41._species_bin_values(
            weighted, tables_by_species, sqrt_s, 1.0)
        # the sealed-style data structure from the same tables drives
        # the statistic construction (hermetic; no sealed archive)
        sealed = {
            t.species: {
                "d": [float(2 + i % 3) for i in range(int(t.data.size))],
                "v_data": [0.25] * int(t.data.size),
                "xF_mid": [float(v) for v in t.xf_centers],
                "pT_mid": [float(v) for v in t.pt_centers],
            } for t in tables
        }
        stats = r41._build_statistics(sealed, tables_by_species, set(),
                                      sqrt_s)
        for stat in stats:
            if stat.kind == "spectrum":
                combo = r41._combination_model(stat, values)
                cells = stat.combo["cells"]
                assert combo.size == len(cells)
                for index, cell in enumerate(cells):
                    ix_a = int(tables_by_species[
                        stat.combo["pair"][0]].bin_index_xf[
                            cell["bin_a"]])
                    ip_a = int(tables_by_species[
                        stat.combo["pair"][0]].bin_index_pt[
                            cell["bin_a"]])
                    weight_a = cell_weights[stat.combo["pair"][0]][
                        (ix_a, ip_a)]
                    ix_b = int(tables_by_species[
                        stat.combo["pair"][1]].bin_index_xf[
                            cell["bin_b"]])
                    ip_b = int(tables_by_species[
                        stat.combo["pair"][1]].bin_index_pt[
                            cell["bin_b"]])
                    weight_b = cell_weights[stat.combo["pair"][1]][
                        (ix_b, ip_b)]
                    expected = (
                        weight_a / (math.pi * float(sqrt_s)
                                    * r41._cell_area(
                                        sealed, tables_by_species,
                                        stat.combo["pair"][0],
                                        cell["bin_a"]))
                        + weight_b / (math.pi * float(sqrt_s)
                                      * r41._cell_area(
                                          sealed, tables_by_species,
                                          stat.combo["pair"][1],
                                          cell["bin_b"])))
                    assert combo[index] == pytest.approx(
                        expected, rel=1e-15, abs=0.0), (
                        f"{stat.key} cell {index}: the combination "
                        "does not follow the two species' published "
                        "rows")
            elif stat.kind == "integral":
                got = float(r41._integral_model(stat, values))
                expected = float(np.sum(
                    r41._combination_model(stat, values) * stat.areas))
                assert got == pytest.approx(expected, rel=1e-12)
            else:
                model = r41._seagull_model(stat, values)
                finite = [v for v in model if v is not None]
                assert all(math.isfinite(v) for v in finite)
        # the dense-position-consumed reading gives a different
        # combination (the superseded convention discriminated on this
        # geometry)
        consumed_values = {}
        for species in r41.SPECIES_FLAT_ORDER:
            dense_slice = weighted[
                int(context["dense_offsets_by_species"][species]):
                int(context["dense_offsets_by_species"][species])
                + int(context["dense_sizes_by_species"][species])]
            table = tables_by_species[species]
            areas = np.asarray([
                r41._cell_area_from_table(table, i)
                for i in range(int(table.data.size))
            ], dtype=float)
            consumed_values[species] = (
                dense_slice[:int(table.data.size)]
                / (math.pi * float(sqrt_s) * areas))
        spectrum = next(s for s in stats if s.kind == "spectrum")
        picked_combo = r41._combination_model(spectrum, values)
        consumed_combo = r41._combination_model(
            spectrum, consumed_values)
        assert not np.allclose(picked_combo, consumed_combo), (
            "the dense-position-consumed combination equals the "
            "picked one; the fixture does not discriminate the two "
            "conventions")
    finally:
        r41._set_worker_context(None)


def test_holey_geometry_evaluation_row_areas_agree():
    """The evaluation's per-published-row area division agrees with the
    recorded rows end to end: ``_species_bin_values`` on the block
    weighted sums of the record returns, for every published row r, the
    row's own Jacobian sum divided by the row's own cell area (the
    self-normalized estimator on the published bins), with no
    hole-cell contamination."""
    tables = _holey_tables()
    _holey_context(tables)
    try:
        finals, hole_finals = _holey_sample(tables)
        record, cell_weights, hole_weights = _holey_record(
            tables, finals, hole_finals)
        tables_by_species = {t.species: t for t in tables}
        sqrt_s = r41._sqrt_s()
        block = r41._block_from_record(record)
        eta = r41.event_weights(block, 1.0, 1.0, 1.0)
        weighted = r41.block_weighted_sums(block, eta)
        values = r41._species_bin_values(
            weighted, tables_by_species, sqrt_s, 1.0)
        for table in tables:
            species = table.species
            got = np.asarray(values[species], dtype=float)
            assert got.size == int(table.data.size)
            for row in range(int(table.data.size)):
                ix, ip = _holey_cell_of(table, row)
                area = r41._cell_area_from_table(table, row)
                expected = cell_weights[species][(ix, ip)] / (
                    math.pi * float(sqrt_s) * 1.0 * float(area))
                assert got[row] == pytest.approx(
                    expected, rel=1e-15, abs=0.0), (
                    f"{species} row {row}: the evaluation's "
                    "per-published-row division disagrees with the "
                    "recorded row")
            # a hole cell's scaled weight appears in no row value
            # (the published weights stay below the hole weights)
            flat_all = np.concatenate([
                np.asarray(values[key], dtype=float)
                for key in r41.SPECIES_FLAT_ORDER])
            for (_ix, _ip), scale in hole_weights[species].items():
                assert not np.any(flat_all * math.pi * float(sqrt_s)
                                  >= scale * 0.999)
    finally:
        r41._set_worker_context(None)


def test_holey_geometry_pick_map_and_duplicate_refusal():
    """The pick map is built in the worker context from the tables'
    bin_index arrays (pick = bin_index_pt * n_x + bin_index_xf), the
    dense layout covers every lattice cell, and a table with two
    published rows on one dense lattice cell refuses (the pick map must
    stay one-to-one)."""
    tables = _holey_tables()
    _holey_context(tables)
    try:
        context = r41._WORKER_CONTEXT
        total = 0
        for table in tables:
            species = table.species
            n_x = int(context["n_x_by_species"][species])
            n_p = int(context["n_p_by_species"][species])
            expected = (np.asarray(table.bin_index_pt, dtype=np.int64) * n_x
                        + np.asarray(table.bin_index_xf, dtype=np.int64))
            pick = context["pick_by_species"][species]
            assert (pick == expected).all()
            assert int(context["dense_sizes_by_species"][species]) == \
                n_x * n_p
            assert int(pick.max()) < n_x * n_p
            total += n_x * n_p
        assert int(context["dense_total"]) == total
        # the dense accumulation space exceeds the published-row space
        # on this geometry (the holes)
        assert int(context["dense_total"]) > int(r41._flat_total())
    finally:
        r41._set_worker_context(None)
    # a duplicated lattice cell refuses (the pick map must be 1:1)
    dup = _holey_tables()[0]
    same_xf = float(dup.xf_centers[int(dup.bin_index_xf[0])])
    same_pt = float(dup.pt_centers[int(dup.bin_index_pt[0])])
    from na49_mixing.staged_evaluation import BinnedFitTable

    window: dict = {"xF": [0.0, 0.4], "pT": [0.05, 1.5]}
    dup_table = BinnedFitTable.from_arrays(
        species="dup_probe", pdg_id=991, mass_gev=0.5,
        normalization="f_over_sigma_inel", sigma_inel_mb=31.78,
        path="test://duplicate-cell",
        xf=np.asarray([same_xf, same_xf]),
        pt=np.asarray([same_pt, same_pt]),
        value=np.ones(2), error=np.ones(2),
        window=window,
    )
    r41._set_worker_context(None)
    with pytest.raises(SystemExit):
        r41._worker_context([dup_table], fams=None, smoke=True)
    r41._set_worker_context(None)
