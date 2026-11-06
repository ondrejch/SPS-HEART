"""Hermetic tests for the R42 driver (the recoil/decay separation).

Task ``TASK-20260920-r4-run-r42-recoil-decay-separation`` (Round 4, card
13; plan section 10 run R42, line 786), reworked by the card-17
relitigation (TASK-20260920-r4-run-r42-relitigation): the seed-map and
event-index partition, the profile-grid invariants, the reweighting
algebra against a hand-computed exponential-transfer case, the
correlation algebra on a pinned 2-D likelihood fixture, the profile
determinism, the R01 accumulate-dense-then-pick hole-geometry suite
(the card-17 recording-convention coverage, mirrored from the R44
card-15 pattern), and the layout sync.  No Monte Carlo generation, no
SPS table, no sealed archive read: the acceptance is hermetic.
"""

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
from tests._freeze_inventory import inventory_count as _inventory_count


def _load_driver():
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "helpers" / "benchmarking"
        / "staged_r42_recoil_decay_separation.py",
    ]
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location(
                "staged_r42_recoil_decay_separation", path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(
        "the R42 driver was not found next to the test file")


r42 = _load_driver()


# ---------------------------------------------------------------------------
# The seed map and the event-index partition (the card's pre-registration)
# ---------------------------------------------------------------------------
def test_seed_map_and_event_index_partition():
    assert r42.MASTER_SEED == 20260920
    assert r42.RAW_REPLICA_SEEDS == tuple(20301020 + r for r in range(8))
    assert r42.EVENT_INDEX_RULE == "500_000_000 + i, i < 100,000"
    assert r42.EVENT_INDEX_BASE == 500_000_000
    assert r42.T2_REPLICAS == 8 and r42.T2_EVENTS_PER_REPLICA == 100_000
    # the per-replica seeds are the block 20301020-20301027 (R41's block,
    # shared by pre-registration; the kernel spaces are disjoint by the
    # event-index partition)
    assert [r42.raw_replica_seed(r) for r in range(8)] == [
        20301020 + r for r in range(8)]
    with pytest.raises(SystemExit):
        r42.raw_replica_seed(8)
    # the smoke block (+9000) is disjoint from the production slots, from
    # the R41 +8000 smoke block, and from the audit block
    assert r42.SMOKE_SEED_STRIDE == 9_000
    assert r42.SMOKE_RAW_SEED == 20260920 + 40_000 + 9_000 == 20309920
    assert r42.SMOKE_KERNEL_REPLICA_SLOT == 9_000
    assert 20309920 not in r42.RAW_REPLICA_SEEDS
    assert 20309920 not in range(20301040, 20301056)  # the audit block
    assert 20309920 != 20308920  # the R41 smoke block
    # the event-index partition is disjoint from R41's (400M-based) and
    # from R40's (300M-based)
    assert r42.EVENT_INDEX_BASE > 400_000_000
    # the rules name the disjointness and the R41 sharing
    assert "R41" in r42.RAW_SEED_RULE
    assert "derive_substreams(20260920, r, 500_000_000 + i)" in \
        r42.KERNEL_SEED_RULE
    assert "20301040 + k" in r42.RAW_SEED_RULE or \
        "20301040 + k" in r42.KERNEL_SEED_RULE


def test_kernel_stream_disjointness():
    """The (master, replica slot, event index) kernel-layer spaces are
    pairwise disjoint across the R42 production slots and the smoke slot,
    and disjoint from the R41 (400M) and R40 (300M) spaces."""
    from na49_mixing.rng_substreams import derive_substreams

    streams = set()
    for replica in range(r42.T2_REPLICAS):
        subs = derive_substreams(
            r42.MASTER_SEED, replica, r42.EVENT_INDEX_BASE)
        for label in ("decay_channel", "decay_kinematics"):
            key = int(subs.stream_for(label))
            assert key not in streams
            streams.add(key)
    subs = derive_substreams(
        r42.MASTER_SEED, r42.SMOKE_KERNEL_REPLICA_SLOT,
        r42.EVENT_INDEX_BASE)
    for label in ("decay_channel", "decay_kinematics"):
        assert int(subs.stream_for(label)) not in streams
    for replica in range(r42.T2_REPLICAS):
        r41_subs = derive_substreams(r42.MASTER_SEED, replica, 400_000_000)
        r40_subs = derive_substreams(r42.MASTER_SEED, replica, 300_000_000)
        for label in ("decay_channel", "decay_kinematics"):
            assert int(r41_subs.stream_for(label)) not in streams
            assert int(r40_subs.stream_for(label)) not in streams


# ---------------------------------------------------------------------------
# The profile-grid invariants (index item 10, the pre-registered choice)
# ---------------------------------------------------------------------------
def test_profile_grid_invariants():
    assert r42.B_EFF_MULTIPLIERS == (
        0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5)
    assert len(r42.B_EFF_MULTIPLIERS) == 9
    assert r42.B_DEC_GRID == (0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6)
    assert len(r42.B_DEC_GRID) == 9
    assert r42.B_DEC_NOMINAL_GEV2 == 1.0
    assert r42.B_DEC_NOMINAL_GEV2 not in r42.B_DEC_GRID
    assert r42.B_DEC_MULT_STEP == 0.25
    assert r42.B_DEC_STEP_GEV2 == 0.3
    assert r42.B_EFF_FROZEN_GEV2 == 8.504140762083612
    # the applied M8 config agrees with the pinned frozen exchange slope
    config = r42._m8_config()
    assert float(config.regge.effective_transfer_slope) == \
        r42.B_EFF_FROZEN_GEV2
    # the pinned validity grid points are off-nominal and on the grid
    assert r42.VALIDITY_GRID_POINTS == ((1.5, 1.7), (2.0, 0.5))
    for mult, b_dec in r42.VALIDITY_GRID_POINTS:
        assert mult in r42.B_EFF_MULTIPLIERS
        assert b_dec in r42.B_DEC_GRID
        assert (mult, b_dec) != (1.0, r42.B_DEC_NOMINAL_GEV2)
    assert r42.VALIDITY_BOUND == 1e-6
    assert r42.VALIDITY_EVENTS == 2_000
    # the nominal alignment point of the generation and the whole profile
    assert r42.NOMINAL_KAPPA_B == 1.0 and r42.NOMINAL_KAPPA_M == 1.0
    assert r42.NOMINAL_WIDTH_GEV == 0.5
    # the pre-registered distinguishability bound
    assert r42.CORRELATION_BOUND == 0.9


# ---------------------------------------------------------------------------
# The reweighting algebra against a hand-computed exponential-transfer case
# ---------------------------------------------------------------------------
def test_transfer_log_weight_closed_form():
    t = np.asarray([-1.0])
    t_lo = np.asarray([-2.0])
    t_hi = np.asarray([0.0])
    b_gen, b_trial = 2.0, 4.0
    log_w = r42.transfer_log_weight(t, t_lo, t_hi, b_gen, b_trial)
    z_gen = (math.exp(b_gen * float(t_hi[0]))
             - math.exp(b_gen * float(t_lo[0]))) / b_gen
    z_trial = (math.exp(b_trial * float(t_hi[0]))
               - math.exp(b_trial * float(t_lo[0]))) / b_trial
    expected = math.exp((b_trial - b_gen) * float(t[0])) * z_gen / z_trial
    assert float(np.exp(log_w[0])) == pytest.approx(expected, rel=1e-12)
    # the generation point carries exactly weight 1
    identity = r42.transfer_log_weight(t, t_lo, t_hi, b_gen, b_gen)
    assert float(np.exp(identity[0])) == 1.0
    # the trial law is exactly normalized: the weight expectation under
    # the generation law is 1 (dense trapezoid on the shared support)
    b_gen, b_trial = 8.504140762083612, 21.0
    lo, hi = -50.0, -0.1
    grid = np.linspace(lo, hi, 400001)
    z_gen = (math.exp(b_gen * hi) - math.exp(b_gen * lo)) / b_gen
    density_gen = np.exp(b_gen * grid) / z_gen
    weights = np.exp(r42.transfer_log_weight(
        grid, np.full_like(grid, lo), np.full_like(grid, hi),
        b_gen, b_trial))
    assert float(np.trapz(density_gen * weights, grid)) == pytest.approx(
        1.0, rel=1e-6)


def test_transfer_implied_t_identity():
    """The implied trial realization equals the direct generation's
    realized t for the same uniform (the wide underflowing-support branch
    and the narrow branch)."""
    rng = np.random.default_rng(20260920)
    for b_gen, b_trial, t_lo, t_hi in (
        (8.504140762083612, 21.0, -50.0, -0.1),
        (8.504140762083612, 12.0, -5.0, -0.1),
    ):
        for _ in range(300):
            u = rng.random()
            a_gen = math.exp(b_gen * (t_lo - t_hi))
            t_gen = (math.log(a_gen + u * (1.0 - a_gen))
                     + b_gen * t_hi) / b_gen
            a_trial = math.exp(b_trial * (t_lo - t_hi))
            t_direct = (math.log(a_trial + u * (1.0 - a_trial))
                        + b_trial * t_hi) / b_trial
            t_implied = r42.transfer_implied_t(
                t_gen, b_gen, b_trial, t_lo, t_hi)
            deviation = abs(t_implied - t_direct) / max(
                1.0, abs(t_direct))
            assert deviation < 1e-12


def test_degenerate_support_weight():
    """The degenerate zero-width support carries transfer weight 1 (the
    measure-zero branch of the sampler)."""
    t = np.asarray([1.0])
    lo = np.asarray([1.0])
    hi = np.asarray([1.0 + 1e-13])
    log_w = r42.transfer_log_weight(t, lo, hi, 2.0, 5.0)
    assert float(log_w[0]) == 0.0


def _init_layout():
    """The hermetic six-species synthetic grid (the layout init): the
    published-row layout and the dense evaluation-lattice layout (the
    card-17 accumulate-dense-then-pick convention).  On this
    rectangular grid every published row is its own dense cell, so the
    dense cell addresses equal the published row indices."""
    tables = _synthetic_tables()
    r42._init_species_layout(tables)
    r42._set_worker_context(r42._worker_context(tables, fams=None,
                                                smoke=True))
    return tables


def _fixture_block():
    """A pinned two-event fixture: event 0 carries one continuum vertex
    (a two-channel pool) and a realized t; event 1 carries no vertices
    (a catalog-only event)."""
    _init_layout()
    return r42.BlockLatent(
        replica=0,
        seed=20301020,
        n_events=2,
        route_census=(4, 2, 2),
        row_flat=np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64),
        row_weight=np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float),
        row_ptr=np.asarray([0, 4, 6], dtype=np.int64),
        event_ids=np.asarray(
            [r42.EVENT_INDEX_BASE, r42.EVENT_INDEX_BASE + 1],
            dtype=np.int64),
        t_scatter=np.asarray([-1.0, -0.4], dtype=float),
        t_minus=np.asarray([-3.0, -1.0], dtype=float),
        t_plus=np.asarray([0.0, 0.0], dtype=float),
        vtx_ptr=np.asarray([0, 1, 1], dtype=np.int64),
        vtx_nchan=np.asarray([2], dtype=np.int64),
        vtx_chosen=np.asarray([0], dtype=np.int64),
        ch_w0=np.asarray([0.6, 0.4], dtype=float),
        ch_q2=np.asarray([0.2, 1.0], dtype=float),
    )


def test_event_weights_combined_closed_form():
    """The combined per-event weight: the closed-form transfer density
    ratio times the card-8 channel-choice ratio, hand-computed."""
    _init_layout()
    block = _fixture_block()
    # the trial at the generation point must be exactly weight 1
    eta0 = r42.event_weights(
        block, r42.B_EFF_FROZEN_GEV2, r42.B_DEC_NOMINAL_GEV2)
    assert float(eta0[0]) == pytest.approx(1.0, abs=1e-14)
    assert float(eta0[1]) == pytest.approx(1.0, abs=1e-14)
    # event 0: the transfer factor at B' = 1.5 x frozen on the event's
    # own support (t = -1, t_hi = 0, dt = 3), times the channel ratio at
    # b' = 1.7 GeV^-2 (pool (0.6, 0.4) at q2 (0.2, 1.0), chosen 0)
    b_gen = r42.B_EFF_FROZEN_GEV2
    b_trial = 1.5 * b_gen
    z_gen = (math.exp(b_gen * 0.0) - math.exp(b_gen * -3.0)) / b_gen
    z_trial = (math.exp(b_trial * 0.0)
               - math.exp(b_trial * -3.0)) / b_trial
    w_transfer = math.exp((b_trial - b_gen) * (-1.0)) * z_gen / z_trial
    delta_b = 1.7 - 1.0
    pool = (math.exp(-delta_b * 0.2) * 0.6
            + math.exp(-delta_b * 1.0) * 0.4)
    w_channel = math.exp(-delta_b * 0.2) / pool
    eta1 = r42.event_weights(block, b_trial, 1.7)
    # the honest hand-computed assertion: the combined weight is the
    # transfer density ratio times the channel-choice ratio
    expected_event0 = w_channel * w_transfer
    assert float(eta1[0]) == pytest.approx(expected_event0, rel=1e-10)
    # event 1 (no vertices): the transfer factor only (dt = 1, t = -0.4)
    z_gen1 = (math.exp(b_gen * 0.0) - math.exp(b_gen * -1.0)) / b_gen
    z_trial1 = (math.exp(b_trial * 0.0)
                - math.exp(b_trial * -1.0)) / b_trial
    wb1 = math.exp((b_trial - b_gen) * (-0.4)) * z_gen1 / z_trial1
    assert float(eta1[1]) == pytest.approx(wb1, rel=1e-10)


def test_weighted_sums_and_ess():
    _init_layout()
    block = _fixture_block()
    eta = np.asarray([2.0, 1.0])
    totals = r42.block_weighted_sums(block, eta)
    assert totals[0] == pytest.approx(2.0)
    assert totals[1] == pytest.approx(4.0)
    assert totals[2] == pytest.approx(6.0)
    assert totals[3] == pytest.approx(8.0)
    assert totals[4] == pytest.approx(5.0)
    assert totals[5] == pytest.approx(6.0)
    ess = r42.ess_fraction(eta)
    assert ess == pytest.approx((2.0 + 1.0) ** 2 / (4.0 + 1.0) / 2.0)


# ---------------------------------------------------------------------------
# The correlation algebra on a pinned 2-D likelihood fixture
# ---------------------------------------------------------------------------
def test_correlation_algebra_pinned_fixture():
    """rho = -H_01 / sqrt(H_00 H_11), the inverse-Hessian parameter
    correlation of the curvature matrix of -2lnL, cross-checked against
    the explicit inverse."""
    h = [[100.0, 30.0], [30.0, 40.0]]
    rho = r42.correlation_from_hessian(h)
    assert float(rho) == pytest.approx(-30.0 / math.sqrt(100.0 * 40.0))
    inverse = np.linalg.inv(np.asarray(h, dtype=float))
    rho_inverse = float(inverse[0][1]) / math.sqrt(
        float(inverse[0][0]) * float(inverse[1][1]))
    assert float(rho) == pytest.approx(rho_inverse, rel=1e-12)
    eigenvalues = r42.hessian_eigenvalues(h)
    assert eigenvalues[0] > 0.0 and eigenvalues[1] > 0.0
    assert r42.positive_definite_2x2(h) is True
    # a strong positive cross-curvature reaches the pre-registered bound
    strong = [[100.0, 95.0], [95.0, 92.0]]
    assert abs(r42.correlation_from_hessian(strong)) > 0.9
    # a non-positive diagonal curvature quotes None (the curvature
    # finding is named instead)
    flat = [[-5.0, 1.0], [1.0, 2.0]]
    assert r42.correlation_from_hessian(flat) is None
    assert not r42.positive_definite_2x2(flat)


def test_grid_hessian_identity():
    """The grid second differences reproduce the curvature of a quadratic
    surface exactly, with the honest one-sided boundary construct at an
    edge."""
    db = r42.B_DEC_MULT_STEP
    dk = r42.B_DEC_STEP_GEV2

    def value(x, k):
        return 2.0 * x * x - 3.0 * x * k + 5.0 * k * k

    surface = [
        [value(i * db, j * dk) for j in range(9)]
        for i in range(9)
    ]
    for i0, j0 in ((4, 4), (0, 3), (8, 0), (2, 8)):
        h, flags = r42.grid_hessian(surface, i0, j0)
        assert h[0][0] == pytest.approx(4.0, rel=1e-10)
        assert h[1][1] == pytest.approx(10.0, rel=1e-10)
        assert h[0][1] == pytest.approx(-3.0, rel=1e-10)
        assert flags["interior_bb"] == (0 < i0 < 8)
        assert flags["interior_kk"] == (0 < j0 < 8)
    assert r42.positive_definite_2x2(r42.grid_hessian(surface, 4, 4)[0])


def test_grid_hessian_boundary_corner_regression():
    """The one-sided boundary construct at a corner minimum evaluates the
    nearest interior-side triplets (the REV-e95cf2e-01 regression).

    The quadratic fixture above cannot catch an axis-index swap: a
    quadratic's second differences are location-independent, so swapped
    triplets return the same values.  The fixture here is an exponential
    surface whose curvature varies along each axis and whose
    unconstrained minimum lies off-grid, so the grid argmin is the corner
    (0, 8) with i0 != j0 (the sealed R42 topology).  The one-sided
    constructs at that corner are the axis-0 triplet over rows (0, 1, 2)
    of column 8 and the axis-1 triplet over columns (6, 7, 8) of row 0;
    the swapped construction evaluates the opposite-edge triplets (rows
    (6, 7, 8) and columns (0, 1, 2)) and fails these assertions."""
    db = r42.B_DEC_MULT_STEP
    dk = r42.B_DEC_STEP_GEV2

    def value(i, j):
        # increasing in the row index (the argmin row is 0) and
        # decreasing in the column index (the argmin column is 8); the
        # exponential curvature varies along each axis, so the second
        # difference at rows (0, 1, 2) differs from rows (6, 7, 8) and
        # columns (6, 7, 8) differs from columns (0, 1, 2)
        x = i * db
        y = j * dk
        return 3.0 * math.exp(1.5 * x) + 2.0 * math.exp(-0.75 * y)

    surface = [[value(i, j) for j in range(9)] for i in range(9)]
    i0, j0 = r42.grid_argmin(surface)
    assert (i0, j0) == (0, 8)
    h, flags = r42.grid_hessian(surface, i0, j0)
    # the hand-computed one-sided constructs at the corner (0, 8): the
    # axis-0 triplet over the nearest interior-side rows (0, 1, 2) of
    # column 8 and the axis-1 triplet over the nearest interior-side
    # columns (6, 7, 8) of row 0
    expected_h00 = (value(2, 8) - 2.0 * value(1, 8) + value(0, 8)) / db**2
    expected_h11 = (value(0, 8) - 2.0 * value(0, 7) + value(0, 6)) / dk**2
    assert h[0][0] == pytest.approx(expected_h00, rel=1e-12)
    assert h[0][0] == pytest.approx(9.936824994061055, rel=1e-12)
    assert h[1][1] == pytest.approx(expected_h11, rel=1e-12)
    assert h[1][1] == pytest.approx(0.23386763344828884, rel=1e-12)
    # the mixed term of the separable surface vanishes (the corner's
    # single interior diagonal feeds the same construct)
    assert h[0][1] == pytest.approx(0.0, abs=1e-12)
    # the corner flags name the boundary construct
    assert flags == {
        "i0": 0, "j0": 8,
        "interior_bb": False, "interior_kk": False, "all_interior": False,
    }


def test_grid_argmin_deterministic():
    surface = [[5.0, 1.0], [3.0, 2.0]]
    assert r42.grid_argmin(surface) == (0, 1)
    ties = [[2.0, 2.0], [2.0, 2.0]]
    assert r42.grid_argmin(ties) == (0, 0)


def _synthetic_tables():
    """A small hermetic six-species table set on a 4x3 grid."""
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


def _fixture_stats():
    """The hermetic statistic fixture: the driver's statistic set on a
    synthetic placeholder data vector with no carry set."""
    _init_layout()
    tables = _synthetic_tables()
    tables_by_species = {t.species: t for t in tables}
    sealed = {}
    for table in tables:
        sealed[table.species] = {
            "d": [float(1 + i % 3) for i in range(table.data.size)],
            "v_data": [0.5] * int(table.data.size),
            "xF_mid": [float(v) for v in table.xf_centers],
            "pT_mid": [float(v) for v in table.pt_centers],
        }
    stats = r42._build_statistics(
        sealed, tables_by_species, set(), r42.SQRT_S_GEV)
    return stats, tables_by_species


def test_statistic_set_shape():
    stats, _tables = _fixture_stats()
    kinds = [s.kind for s in stats]
    assert kinds.count("spectrum") == 1  # the B_transport spectrum only
    assert kinds.count("seagull") == 2   # the pooled pion and proton
    assert stats[0].key == "b_transport_spectrum"
    # the leading-baryon transfer pair is the plan-431 difference
    assert stats[0].combo["pair"] == ("proton", "antiproton", -1.0)
    seagull_keys = [s.key for s in stats if s.kind == "seagull"]
    assert seagull_keys == ["pi_sigma_seagull", "b_transport_seagull"]
    for stat in stats:
        if stat.kind == "seagull":
            assert len(stat.d) % 2 == 0
            assert np.all(np.isfinite(stat.v_data))
            assert np.all(stat.v_data >= r42.SE_FLOOR)


def test_objective_pure_function_on_fixture():
    stats, tables_by_species = _fixture_stats()
    block = _fixture_block()
    objective = r42.ProfileObjective(
        stats, [block], tables_by_species, r42.SQRT_S_GEV, 0.25, 0.35)
    one = objective.evaluate(1.3 * r42.B_EFF_FROZEN_GEV2, 1.1)["value"]
    two = objective.evaluate(1.3 * r42.B_EFF_FROZEN_GEV2, 1.1)["value"]
    assert one == two
    assert math.isfinite(one)


# ---------------------------------------------------------------------------
# The seagull moment fidelity (the house convention on both sides)
# ---------------------------------------------------------------------------
def test_seagull_model_matches_house_convention():
    _init_layout()
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
    stats = r42._build_statistics(
        sealed, tables_by_species, set(), r42.SQRT_S_GEV)
    stat = next(s for s in stats if s.kind == "seagull"
                and s.key.startswith("pi_sigma"))
    assert len(stat.seagull["bin_indices"]) == len(stat.combo["cells"])
    assert set(stat.seagull["regions"]) <= {"central", "forward"}
    species_values = {
        "pi_plus": np.full(len(sealed["pi_plus"]["d"]), 0.5),
        "pi_minus": np.full(len(sealed["pi_minus"]["d"]), 0.7),
        "proton": np.zeros(len(sealed["proton"]["d"])),
        "antiproton": np.zeros(len(sealed["antiproton"]["d"])),
    }
    model = r42._seagull_model(stat, species_values)
    combo_values = r42._combination_model(stat, species_values)
    seagull = stat.seagull
    result = r42.fo.seagull_moment(
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


# ---------------------------------------------------------------------------
# The CLI surface (the seven modes)
# ---------------------------------------------------------------------------
def test_cli_surface():
    parser = r42._build_parser()
    modes = {
        "--pre-registration", "--smoke", "--generate", "--profile",
        "--gate", "--seal", "--clean-validate",
    }
    auxiliary = {"--replica", "--run-dir", "--workers"}
    seen = set()
    for action in parser._actions:
        seen.update(action.option_strings)
    assert seen == modes | auxiliary | {"-h", "--help"}, (
        f"the CLI surface drifted: {sorted(seen)}")
    with pytest.raises(SystemExit) as exc:
        r42.main([])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        r42.main(["--profile", "--gate"])
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# The layout sync (the dual-tree driver byte-identical)
# ---------------------------------------------------------------------------
def test_layout_sync():
    """The nested and monorepo driver copies are byte-identical and sit
    at the driver's self-computed identity digest."""
    nested = Path(__file__).resolve().parents[1] / "helpers" / \
        "benchmarking" / "staged_r42_recoil_decay_separation.py"
    mono = Path(__file__).resolve().parents[3] / "helpers" / \
        "benchmarking" / "staged_r42_recoil_decay_separation.py"
    assert nested.is_file(), nested
    assert mono.is_file(), mono
    nested_digest = hashlib.sha256(nested.read_bytes()).hexdigest()
    mono_digest = hashlib.sha256(mono.read_bytes()).hexdigest()
    assert nested_digest == mono_digest
    assert nested_digest == r42.DRIVER_SHA256


# ---------------------------------------------------------------------------
# The hole-geometry suite (the R01 accumulate-dense-then-pick recording
# convention; mirrored from the R44 driver's card-15 test coverage and
# the card-16 R41 adaptation, adapted to the R42 recording path and
# species surface: the six paper-window species, the pooled-combination
# spectra and seagulls on the aligned cells)
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
    """The six-species hole-geometry table set (the R42 recording
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
    sqrt_s = r42._sqrt_s()
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
    capture = r42._LatentCapture()
    capture.event_id = 0
    cell_weights: dict[str, dict[tuple[int, int], float]] = {
        t.species: {} for t in tables}
    for species, ix, ip, scale in finals:
        pdg = r42.SPECIES_PDGS[species]
        p4, weight = _holey_final(
            ix, ip, xf_by[species], pt_by[species], scale)
        capture.add_final(pdg, p4)
        cell_weights[species][(ix, ip)] = weight
    hole_weights: dict[str, dict[tuple[int, int], float]] = {
        t.species: {} for t in tables}
    for species, ix, ip, scale in hole_finals:
        pdg = r42.SPECIES_PDGS[species]
        p4, weight = _holey_final(
            ix, ip, xf_by[species], pt_by[species], scale)
        capture.add_final(pdg, p4)
        hole_weights[species][(ix, ip)] = weight
    parts = r42._chunk_parts(0, 0, [capture], smoke=True)
    aggregated = r42._aggregate_chunks([parts])
    record = r42._replica_record(
        0, 20301020, 0, 1, aggregated, "0" * 64, smoke=True)
    return record, cell_weights, hole_weights


def _holey_context(tables):
    """The worker context of the holey tables (the dense layout and the
    pick maps), with the context left set for the recording path."""
    r42._init_species_layout(tables)
    r42._set_worker_context(r42._worker_context(tables, fams=None,
                                                smoke=True))
    return r42._WORKER_CONTEXT


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
        context = r42._WORKER_CONTEXT
        dense = _holey_dense_reference(context, cell_weights,
                                       hole_weights)
        picks = context["pick_by_species"]
        sizes = context["dense_sizes_by_species"]
        offsets = context["dense_offsets_by_species"]
        pooled = record["pooled_sums"]
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
            flat_all = [v for key in r42.SPECIES_FLAT_ORDER
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
        r42._set_worker_context(None)


def test_holey_geometry_combination_follows_published_pick():
    """The pooled-combination diagnostic geometry (R42's analog of the
    card-15 model-only species case): the pooled-combination statistics
    (pi_Sigma = pi+ + pi-, B_transport = p - pbar, and the pooled
    seagull moments built on them) consume the two species' picked rows
    through ``_species_bin_values`` at the evaluation boundary, so every
    combination cell equals the sum of its two species' own-row
    contributions and the dense-position-consumed reading gives a
    different combination."""
    tables = _holey_tables()
    _holey_context(tables)
    try:
        finals, hole_finals = _holey_sample(tables)
        record, cell_weights, hole_weights = _holey_record(
            tables, finals, hole_finals)
        context = r42._WORKER_CONTEXT
        tables_by_species = {t.species: t for t in tables}
        sqrt_s = r42._sqrt_s()
        # the block from the record (the dense latent) and the
        # evaluation-boundary conversion to published rows
        block = r42._block_from_record(record)
        eta = r42.event_weights(
            block, r42.B_EFF_FROZEN_GEV2, r42.B_DEC_NOMINAL_GEV2)
        weighted = r42.block_weighted_sums(block, eta)
        assert weighted.size == int(context["dense_total"])
        values = r42._species_bin_values(
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
        stats = r42._build_statistics(sealed, tables_by_species, set(),
                                      sqrt_s)
        for stat in stats:
            if stat.kind == "spectrum":
                combo = r42._combination_model_statistic(
                    stat.combo,
                    values[stat.combo["pair"][0]],
                    values[stat.combo["pair"][1]])
                cells = stat.combo["cells"]
                assert combo.size == len(cells)
                for index, cell in enumerate(cells):
                    pair_a = stat.combo["pair"][0]
                    pair_b = stat.combo["pair"][1]
                    ix_a = int(tables_by_species[pair_a].bin_index_xf[
                        cell["bin_a"]])
                    ip_a = int(tables_by_species[pair_a].bin_index_pt[
                        cell["bin_a"]])
                    weight_a = cell_weights[pair_a][(ix_a, ip_a)]
                    ix_b = int(tables_by_species[pair_b].bin_index_xf[
                        cell["bin_b"]])
                    ip_b = int(tables_by_species[pair_b].bin_index_pt[
                        cell["bin_b"]])
                    weight_b = cell_weights[pair_b][(ix_b, ip_b)]
                    expected = (
                        weight_a / (math.pi * float(sqrt_s)
                                    * r42._cell_area(
                                        sealed, tables_by_species,
                                        pair_a, cell["bin_a"]))
                        + stat.combo["pair"][2] * weight_b / (
                            math.pi * float(sqrt_s)
                            * r42._cell_area(
                                sealed, tables_by_species,
                                pair_b, cell["bin_b"])))
                    assert combo[index] == pytest.approx(
                        expected, rel=1e-15, abs=0.0), (
                        f"{stat.key} cell {index}: the combination "
                        "does not follow the two species' published "
                        "rows")
            else:
                model = r42._seagull_model(stat, values)
                finite = [v for v in model if v is not None]
                assert all(math.isfinite(v) for v in finite)
        # the dense-position-consumed reading gives a different
        # combination (the superseded convention discriminated on this
        # geometry)
        consumed_values = {}
        for species in r42.SPECIES_FLAT_ORDER:
            dense_slice = weighted[
                int(context["dense_offsets_by_species"][species]):
                int(context["dense_offsets_by_species"][species])
                + int(context["dense_sizes_by_species"][species])]
            table = tables_by_species[species]
            areas = np.asarray([
                r42._cell_area_from_table(table, i)
                for i in range(int(table.data.size))
            ], dtype=float)
            consumed_values[species] = (
                dense_slice[:int(table.data.size)]
                / (math.pi * float(sqrt_s) * areas))
        spectrum = next(s for s in stats if s.kind == "spectrum")
        picked_combo = r42._combination_model_statistic(
            spectrum.combo,
            values[spectrum.combo["pair"][0]],
            values[spectrum.combo["pair"][1]])
        consumed_combo = r42._combination_model_statistic(
            spectrum.combo,
            consumed_values[spectrum.combo["pair"][0]],
            consumed_values[spectrum.combo["pair"][1]])
        assert not np.allclose(picked_combo, consumed_combo), (
            "the dense-position-consumed combination equals the "
            "picked one; the fixture does not discriminate the two "
            "conventions")
    finally:
        r42._set_worker_context(None)


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
        sqrt_s = r42._sqrt_s()
        block = r42._block_from_record(record)
        eta = r42.event_weights(
            block, r42.B_EFF_FROZEN_GEV2, r42.B_DEC_NOMINAL_GEV2)
        weighted = r42.block_weighted_sums(block, eta)
        values = r42._species_bin_values(
            weighted, tables_by_species, sqrt_s, 1.0)
        for table in tables:
            species = table.species
            got = np.asarray(values[species], dtype=float)
            assert got.size == int(table.data.size)
            for row in range(int(table.data.size)):
                ix, ip = _holey_cell_of(table, row)
                area = r42._cell_area_from_table(table, row)
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
                for key in r42.SPECIES_FLAT_ORDER])
            for (_ix, _ip), scale in hole_weights[species].items():
                assert not np.any(flat_all * math.pi * float(sqrt_s)
                                  >= scale * 0.999)
    finally:
        r42._set_worker_context(None)


def test_holey_geometry_pick_map_and_duplicate_refusal():
    """The pick map is built in the worker context from the tables'
    bin_index arrays (pick = bin_index_pt * n_x + bin_index_xf), the
    dense layout covers every lattice cell, and a table with two
    published rows on one dense lattice cell refuses (the pick map must
    stay one-to-one)."""
    tables = _holey_tables()
    _holey_context(tables)
    try:
        context = r42._WORKER_CONTEXT
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
        assert int(context["dense_total"]) > int(r42._flat_total())
    finally:
        r42._set_worker_context(None)
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
    r42._set_worker_context(None)
    with pytest.raises(SystemExit):
        r42._worker_context([dup_table], fams=None, smoke=True)
    r42._set_worker_context(None)


# ---------------------------------------------------------------------------
# The freeze re-pin register (the card-13 landing)
# ---------------------------------------------------------------------------
MANIFEST_REL = "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json"
DRIVER_REL_NESTED = (
    "dev/mixing2006/helpers/benchmarking/"
    "staged_r42_recoil_decay_separation.py"
)
DRIVER_REL_OUTER = (
    "helpers/benchmarking/staged_r42_recoil_decay_separation.py"
)
MY_TEST_REL = (
    "dev/mixing2006/tests/"
    "test_staged_r42_recoil_decay_separation.py"
)
SOLVER_CARD = "TASK-20260920-r4-run-r42-recoil-decay-separation"

#: The sixteen register-chain files whose inventory-count pins this card
#: re-pins (480 -> 483, +3 entries): the twelve register-chain files
#: (cards 1, 4, 5, 6, 7, 8, 9, 10) plus the card-10 solver test's own
#: count constant (test_cascade_ablation.py), the card-11 test-author
#: register's own count constant
#: (test_staged_r40_alignment_scan_contract_pins.py), and the card-12
#: test-author register's own count constant
#: (test_staged_r41_nonstrange_pilot_fit_contract_pins.py).
CHAIN_COUNT_PIN_FILES = (
    "dev/mixing2006/tests/test_r4_audit_blend_path_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_decay_contract_pins.py",
    "dev/mixing2006/tests/test_cascade_axes.py",
    "dev/mixing2006/tests/test_r4_cascade_axes_contract_pins.py",
    "dev/mixing2006/tests/test_cascade_alignment_baryonic.py",
    "dev/mixing2006/tests/"
    "test_r4_cascade_alignment_baryonic_contract_pins.py",
    "dev/mixing2006/tests/test_cascade_alignment_mesonic.py",
    "dev/mixing2006/tests/"
    "test_r4_cascade_alignment_mesonic_contract_pins.py",
    "dev/mixing2006/tests/test_cascade_form_factor.py",
    "dev/mixing2006/tests/test_r4_cascade_form_factor_contract_pins.py",
    "dev/mixing2006/tests/test_cascade_termination.py",
    "dev/mixing2006/tests/test_r4_cascade_termination_contract_pins.py",
    "dev/mixing2006/tests/test_cascade_ablation.py",
    "dev/mixing2006/tests/test_r4_cascade_ablation_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_staged_r40_alignment_scan_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_staged_r41_nonstrange_pilot_fit_contract_pins.py",
)
# re-pinned 483 -> 484 by card 13 test-author (TASK-20260920-r4-run-
# r42-recoil-decay-separation-test-author; +1 entry: the card-13
# test-author register file).
# re-pinned 484 -> 487 by card 14 (TASK-20260920-r4-run-r43-
# cascade-ablation; +3 entries: the dual-tree R43 driver in
# both helper trees and the hermetic R43 test file).
# re-pinned 487 -> 488 by card 14 test-author (TASK-20260920-r4-run-
# r43-cascade-ablation-test-author; +1 entry: the card-14
# test-author register file).
# re-pinned 488 -> 491 by card 15 (TASK-20260920-r4-run-r44-
# frozen-nonstrange-validation; +3 entries: the dual-tree R44
# driver in both helper trees and the hermetic R44 test file).
# re-pinned 491 -> 495 by card 20 (TASK-20260920-r4-gate4-followup-simplified-alignment; +4 entries: the dual-tree R45 driver in both helper trees and the two new R45 test files).
# re-pinned 495 -> 499 by card 21 (TASK-20260920-r4-gate4-followup-recoil-decay-reprofile; +4 entries: the dual-tree R46 driver in both helper trees and the two new R46 test files).
# re-pinned 499 -> 501 by Round 5 card 1 (TASK-20260925-r5-elastic-reference; +2 entries: the elastic reference module na49_mixing/topology_elastic.py and the new elastic test file).
# re-pinned 501 -> 502 by Round 5 card 1 test-author (TASK-20260925-r5-elastic-reference-test-author; +1 entry: the card-1 test-author register file).
# re-pinned 502 -> 504 by Round 5 card 2 (TASK-20260925-r5-exchange-quantum-number-ledger; +2 entries: the exchange quantum-number ledger module na49_mixing/exchange_ledger.py and the new ledger test file tests/test_exchange_ledger.py).
# re-pinned 504 -> 505 by Round 5 card 2 test-author (TASK-20260925-r5-exchange-quantum-number-ledger-test-author; +1 entry: the card-2 test-author register file).
# re-pinned 505 -> 508 by the test-protocol tiering change (2026-09-25; +3 entries: helpers/benchmarking/select_tests_for_changes.py, its vendored copy, and helpers/benchmarking/tests/test_select_tests_for_changes.py).
# re-pinned 508 -> 510 by Round 5 card 3 (TASK-20260925-r5-intact-low-mass-remnants; +2 entries: the remnant module na49_mixing/topology_remnants.py and the new remnant test file tests/test_topology_remnants.py).
# re-pinned 510 -> 511 by Round 5 card 3 test-author (TASK-20260925-r5-intact-low-mass-remnants-test-author; +1 entry: the card-3 test-author register file tests/test_topology_remnants_contract_pins.py).
# re-pinned 511 -> 513 by Round 5 card 4 (TASK-20260925-r5-topology-mass-laws; +2 entries: the mass-laws module na49_mixing/topology_mass_laws.py and the new mass-laws test file tests/test_topology_mass_laws.py).
# re-pinned 513 -> 514 by Round 5 card 4 test-author (TASK-20260925-r5-topology-mass-laws-test-author; +1 entry: the card-4 test-author register file tests/test_topology_mass_laws_contract_pins.py).
# re-pinned 514 -> 516 by Round 5 card 5 (TASK-20260925-r5-single-diffraction; +2 entries: the single-diffraction module na49_mixing/topology_single_diffraction.py and the new single-diffraction test file tests/test_topology_single_diffraction.py).
# re-pinned 516 -> 517 by Round 5 card 5 test-author (TASK-20260925-r5-single-diffraction-test-author; +1 entry: the card-5 test-author register file tests/test_topology_single_diffraction_contract_pins.py).
# re-pinned 519 -> 520 by Round 5 card 6 test-author (TASK-20260925-r5-double-diffraction-test-author; +1 entry: the card-6 test-author register file tests/test_topology_double_diffraction_contract_pins.py).
# re-pinned 520 -> 522 by Round 5 card 7 (TASK-20260925-r5-non-diffrative-central-excitation; +2 entries: the module na49_mixing/topology_non_diffractive.py and the test file tests/test_topology_non_diffractive.py).
# re-pinned 522 -> 523 by Round 5 card 7 test-author (TASK-20260925-r5-non-diffrative-central-excitation-test-author; +1 entry: the card-7 test-author register file tests/test_topology_non_diffractive_contract_pins.py).
# re-pinned 523 -> 525 by Round 5 card 8 (TASK-20260925-r5-multiple-neutral-excitations; +2 entries: the module na49_mixing/topology_multi_neutral.py and the test file tests/test_topology_multi_neutral.py).
# re-pinned 525 -> 526 by Round 5 card 8 test-author (TASK-20260925-r5-multiple-neutral-excitations-test-author; +1 entry: the card-8 test-author register file tests/test_topology_multi_neutral_contract_pins.py).
# re-pinned 526 -> 528 by Round 5 card 9 (TASK-20260925-r5-topology-template-runner; +2 entries: the runner module na49_mixing/topology_runner.py and the test file tests/test_topology_runner.py).
# re-pinned 528 -> 529 by Round 5 card 9 test-author (TASK-20260925-r5-topology-template-runner-test-author; +1 entry: the card-9 test-author register file tests/test_topology_runner_contract_pins.py).
# re-pinned 529 -> 532 by Round 5 card 10 (TASK-20260925-r5-run-r50-pure-topology-templates; +3 entries: the dual-tree driver helpers/benchmarking/staged_r50_pure_topology_templates.py, its nested copy, and the test file tests/test_staged_r50_pure_topology_templates.py).
# re-pinned 532 -> 533 by Round 5 card 10 test-author (TASK-20260925-r5-run-r50-pure-topology-templates-test-author; +1 entry: the card-10 test-author register file tests/test_staged_r50_pure_topology_templates_contract_pins.py).
# re-pinned 533 -> 536 by Round 5 card 11 (TASK-20260925-r5-run-r51-topology-fraction-fit; +3 entries: the dual-tree driver helpers/benchmarking/staged_r51_topology_fraction_fit.py, its nested copy, and the test file tests/test_staged_r51_topology_fraction_fit.py).
# re-pinned 536 -> 537 by Round 5 card 11 test-author (TASK-20260925-r5-run-r51-topology-fraction-fit-test-author; +1 entry: the card-11 test-author register file tests/test_staged_r51_topology_fraction_fit_contract_pins.py).
# re-pinned 537 -> 540 by Round 5 card 12 (TASK-20260925-r5-run-r52-central-excitation-necessity; +3 entries: the dual-tree driver dev/mixing2006/helpers/benchmarking/staged_r52_central_excitation_necessity.py, its outer mirror helpers/benchmarking/staged_r52_central_excitation_necessity.py, and the test file dev/mixing2006/tests/test_staged_r52_central_excitation_necessity.py; the card's pre-registered 523 -> 526 literal is noted, the operative growth followed)
# re-pinned 540 -> 541 by Round 5 card 12 test-author (TASK-20260925-r5-run-r52-central-excitation-necessity-test-author; +1 entry: the card-12 test-author register file tests/test_staged_r52_central_excitation_necessity_contract_pins.py).
# re-pinned 541 -> 544 by Round 5 card 13 (TASK-20260925-r5-run-r53-multiple-neutral-excitation; +3 entries: the dual-tree driver dev/mixing2006/helpers/benchmarking/staged_r53_multiple_neutral_excitation.py, its outer mirror helpers/benchmarking/staged_r53_multiple_neutral_excitation.py, and the test file tests/test_staged_r53_multiple_neutral_excitation.py; the card's pre-registered 526 -> 529 literal is noted, the operative growth followed).
# re-pinned 544 -> 547 by Round 5 card 14 (TASK-20260925-r5-run-r54-local-ledger-audit; +3 entries: the dual-tree driver dev/mixing2006/helpers/benchmarking/staged_r54_local_ledger_audit.py, its outer mirror helpers/benchmarking/staged_r54_local_ledger_audit.py, and the test file tests/test_staged_r54_local_ledger_audit.py; the card's pre-registered 529 -> 532 literal is noted, the operative growth followed).
# re-pinned 547 -> 548 by Round 5 card 13 test-author (TASK-20260925-r5-run-r53-multiple-neutral-excitation-test-author; +1 entry: the card-13 test-author register file tests/test_staged_r53_multiple_neutral_excitation_contract_pins.py).
# re-pinned 548 -> 551 by Round 5 card 15 (TASK-20260925-r5-run-r55-frozen-topology-validation; +3 entries: the dual-tree driver dev/mixing2006/helpers/benchmarking/staged_r55_frozen_topology_validation.py, its outer mirror helpers/benchmarking/staged_r55_frozen_topology_validation.py, and the test file tests/test_staged_r55_frozen_topology_validation.py; the card's pre-registered 532 -> 535 literal is noted, the operative growth followed).
INVENTORY_COUNT = _inventory_count()
def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


