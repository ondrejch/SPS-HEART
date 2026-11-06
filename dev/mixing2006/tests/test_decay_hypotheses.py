"""M7 committed behavior: the three-hypothesis non-strange continuum selector.

Task card M7 (2026-09-11): the non-strange continuum decay kernel
(``na49_mixing/continuum_channels.py``) samples the non-strange source
mass under three selectable hypotheses (config key
``continuum.decay_hypothesis``):

H1  exact sampling of the Lorentz-invariant phase-space measure,
    cluster masses drawn from the PHSP grid ladders;
H2  only the explicit sequential resonances above threshold from the
    committed (p|n) x (pi+|pi0|pi-) and (pi+ x pi-) pairing rules, with
    resonance mass line shapes built from the PDG mass and width;
H3  rest-frame sequential sampling, byte-identical to the pre-M7 code.

Acceptance criteria (card M7):
1. Per hypothesis, generated events pass ``event.validate()``
   (4-momentum and integer QN residuals at 1e-10).
2. H1 is validated against the analytic 3-pion phase-space distribution
   (pre-registered chi-square check).
3. H2 emits only resonance intermediates above threshold, with a line
   shape from the PDG width.
4. H3 is byte-identical to the pre-M7 default stream.
5. The config selector (``ContinuumParameters.decay_hypothesis``)
   routes the committed value into the kernel.

Committed behavior is pinned at nested HEAD (fix commit c085269, which
supersedes the selector commit 3e0e131). Statistical quantities below
are pre-registered (seed, event count, tolerance); the 3-sigma
chi-square bound uses the one-sided tail 1.35e-3 convention established
by the M2 charge-balance test.

Two product bugs found while covering 3e0e131 are fixed at c085269;
their former strict-xfail tests are now plain pins of the fix:

1. The vectorized H1 pstar ladders (``_phsp_pstar_pair``,
   ``_phsp_pstar_parent``, ``_phsp_pstar_grid``) computed the Kallen
   function with the wrong constant term (``b + c`` instead of
   ``b**2 + c**2`` for ``b = m1**2``, ``c = m2**2``), sampling a
   threshold-distorted cluster-mass density (three-body chi-square
   ~1186 against the pre-registered bound 42.871, df 19). The ladders
   now share the scalar ``kallen`` formula and the
   ``two_body_momentum_safe`` threshold mask.
2. The ``_H2_DELTA_BY_PAIR`` table swapped the antineutron Delta-bar
   labels (``(-2112, -211) -> -1114`` and ``(-2112, 211) -> -2214``
   where charge conjugation requires ``-2214`` and ``-1114``).
   Kinemically invisible (line shapes key on ``abs_id`` and the
   daughters carry the quantum numbers); the table now matches charge
   conjugation of the neutron rows.
"""

from __future__ import annotations

import math

import numpy as np

#: numpy 2 renamed trapz to trapezoid (the code reviews of 2026-11-04)
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

import pytest
from scipy.stats import chi2 as chi2_dist

from na49_mixing.continuum_channels import (
    ContinuumParameters,
    SourceDecayer,
    _CascadeComponent,
    _decay_phsp,
    _h2_line_shape,
    _h2_resonance_mass,
    _h2_resonant_pairs,
    _phsp_pstar_grid,
    _phsp_pstar_pair,
    _phsp_pstar_parent,
)
from na49_mixing.event_builder_pp import PPEventBuilder, PPExcitationConfig
from na49_mixing.excitation_kinematics import (
    FourVector,
    draw_from_grid_density,
    two_body_momentum_safe,
    two_body_split_rest_frame,
    two_body_split_safe,
)
from na49_mixing.pdg_database import get_mass, get_width
from na49_mixing.quantum_numbers import QuantumNumbers
from na49_mixing.source_event import SourceRecord

# --- Pinned PDG values (pdg_database) -----------------------------------
M_PI0 = get_mass(111)  # 0.1349766
M_PI = get_mass(211)  # 0.1395699
M_P = get_mass(2212)  # 0.93827231
M_N = get_mass(2112)  # 0.93956541
M_DELTA = get_mass(2224)  # 1.232
M_RHO = get_mass(113)  # 0.77526

# --- Pre-registered statistics (card M7) --------------------------------
H1_SEED = 20260911
H1_SOURCE_MASS = 3.0
H1_N = 20000
H1_NBINS = 20
# 3-sigma one-sided chi-square upper-tail quantile, df = NBINS - 1 = 19.
H1_CHI2_BOUND = float(chi2_dist.ppf(1.0 - 1.35e-3, H1_NBINS - 1))  # 42.871
H3_GOLDEN_SEED = 20260913
H2_WINDOW_SEED = 20260914
H2_GENERATED_SEED = 20260916
BUILDER_SEED = 20260909
DIRECT_SEED = 20260901


def _make_source(mass: float) -> SourceRecord:
    """P_LIKE (B=1, Q=1, S=0) non-strange source at rest."""
    return SourceRecord(
        source_id=0,
        role="excitation",
        p4=FourVector(mass, 0.0, 0.0, 0.0),
        quantum_numbers=QuantumNumbers(1, 1, 0),
    )


def _invariant_mass(a: FourVector, b: FourVector) -> float:
    p = a.spatial() + b.spatial()
    e = a.e + b.e
    return math.sqrt(max(e * e - float(p @ p), 0.0))


def _res_component(d1_id: int, d2_id: int, res_id: int) -> _CascadeComponent:
    d1, d2 = get_mass(d1_id), get_mass(d2_id)
    return _CascadeComponent(
        kind="res",
        index=0,
        min_mass=d1 + d2,
        d1_mass=d1,
        d2_mass=d2,
        d1_id=d1_id,
        d2_id=d2_id,
        res_id=res_id,
        partner=1,
    )


# --- Kinematic primitive pins -------------------------------------------


def test_two_body_momentum_safe_matches_strict_above_threshold():
    m1, m2 = 0.3, 0.5
    for parent in (1.7, 2.0, 3.0):
        from na49_mixing.excitation_kinematics import two_body_momentum

        assert two_body_momentum_safe(parent, m1, m2) == pytest.approx(
            two_body_momentum(parent, m1, m2), rel=1e-15
        )


def test_two_body_momentum_safe_is_zero_at_and_below_threshold():
    assert two_body_momentum_safe(2.0 * M_PI, M_PI, M_PI) == 0.0
    assert two_body_momentum_safe(2.0 * M_PI - 1e-9, M_PI, M_PI) == 0.0
    assert two_body_momentum_safe(0.0, M_PI, M_PI) == 0.0
    assert two_body_momentum_safe(-1.0, M_PI, M_PI) == 0.0


def test_two_body_split_safe_clamps_one_ulp_overshoot_to_threshold_point():
    # A 1-ulp parent-mass overshoot makes the strict split fail; the safe
    # variant clamps to the exact threshold point (both daughters at rest
    # in the parent frame).
    m_a, m_b = 0.1, 0.9000000000000001  # m_a + m_b > 1.0 by 1 ulp
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="two-body threshold failed"):
        two_body_split_rest_frame(1.0, m_a, m_b, rng)
    a, b = two_body_split_safe(1.0, m_a, m_b, np.random.default_rng(1))
    assert a.e == pytest.approx(m_a, abs=1e-15)
    assert b.e == pytest.approx(m_b, abs=1e-15)
    assert float(np.linalg.norm(a.spatial())) <= 1e-14
    assert float(np.linalg.norm(b.spatial())) <= 1e-14


def test_two_body_split_safe_rejects_a_physical_threshold_violation():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="two-body split below threshold"):
        two_body_split_safe(1.0, 0.8, 0.4, rng)


def test_two_body_split_safe_conserves_mass_away_from_threshold():
    rng = np.random.default_rng(2)
    a, b = two_body_split_safe(3.0, M_PI, M_PI0, rng)
    assert a.e + b.e == pytest.approx(3.0, rel=1e-12)
    assert float(np.linalg.norm(a.spatial() + b.spatial())) <= 1e-12
    assert a.mass() == pytest.approx(M_PI, rel=1e-12)
    assert b.mass() == pytest.approx(M_PI0, rel=1e-12)


def test_draw_from_grid_density_samples_the_density_on_the_grid():
    xs = np.linspace(0.0, 1.0, 101)
    density = 2.0 * xs  # pdf on [0, 1]: mean 2/3, support [0, 1]
    rng = np.random.default_rng(6)
    draws = np.array([draw_from_grid_density(rng, xs, density) for _ in range(20000)])
    assert draws.min() >= xs[0]
    assert draws.max() <= xs[-1]
    expected = 2.0 / 3.0
    assert draws.mean() == pytest.approx(expected, abs=4.0 * draws.std() / math.sqrt(20000))


@pytest.mark.parametrize(
    "grid, density, message",
    [
        (np.linspace(0.1, 0.9, 5), np.zeros(5), "carries no mass"),
        (np.linspace(0.1, 0.9, 5), -np.ones(5), "carries no mass"),
        (np.linspace(0.1, 0.9, 1), np.ones(1), "at least two points"),
        (np.linspace(0.1, 0.9, 5), np.ones(4), "equal length"),
    ],
)
def test_draw_from_grid_density_rejects_degenerate_grids(grid, density, message):
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError, match=message):
        draw_from_grid_density(rng, grid, density)


# --- Hypothesis selector (acceptance criterion 5) ------------------------


def test_selector_default_is_h3():
    assert ContinuumParameters().decay_hypothesis == "H3"
    ContinuumParameters().validate()


@pytest.mark.parametrize("raw", ("H1", "H2", "H3", "h1", " h3 ", "H1\n"))
def test_selector_accepts_committed_hypotheses(raw):
    ContinuumParameters(decay_hypothesis=raw).validate()


@pytest.mark.parametrize("raw", ("H4", "phase_space", "", "h0", " H4 "))
def test_selector_rejects_uncommitted_hypotheses(raw):
    with pytest.raises(ValueError, match="decay_hypothesis must be one of H1, H2, or H3"):
        ContinuumParameters(decay_hypothesis=raw).validate()


def test_selector_dispatch_is_case_and_whitespace_insensitive():
    # decay() normalizes via strip().upper(); the dispatch must not depend
    # on the raw spelling of the committed value.
    source = _make_source(3.0)

    def stream(raw: str):
        decayer = SourceDecayer(
            ContinuumParameters(decay_hypothesis=raw), np.random.default_rng(21)
        )
        return [(p.pdg_id, p.p4.e, p.p4.px, p.p4.py, p.p4.pz) for p in decayer.decay(source)]

    assert stream("H1") == stream("h1")
    assert stream("H2") == stream(" h2 ")


# --- H3 default-stream golden (acceptance criterion 4) -------------------


def test_h3_default_stream_is_byte_identical_golden():
    # Pre-M7 default stream (seed 20260913, source mass 3.0, P_LIKE at
    # rest, default ContinuumParameters): family continuum_nonstrange,
    # daughters (p, pi0, pi0, pi0). H3 dispatch is byte-identical across
    # all committed spellings and differs from the H1/H2 dispatches.
    golden = [
        (2212, 1.3557902361605307, -0.12249849501264608, 0.39082611071591755, -0.8888539287519619),
        (111, 0.5530118014278377, 0.390039901871909, 0.2783169402525377, 0.240856649663486),
        (111, 0.8954827582843717, -0.33519404321275803, -0.6227743375410988, 0.5324169096976127),
        (111, 0.1957152041272598, 0.06765263635349518, -0.04636871342735646, 0.11558036939086322),
    ]
    source = _make_source(3.0)
    streams = {}
    for raw in ("H3", "h3", " H3 "):
        decayer = SourceDecayer(
            ContinuumParameters(decay_hypothesis=raw),
            np.random.default_rng(H3_GOLDEN_SEED),
        )
        streams[raw] = [
            (p.pdg_id, p.p4.e, p.p4.px, p.p4.py, p.p4.pz) for p in decayer.decay(source)
        ]
    assert streams["H3"] == streams["h3"] == streams[" H3 "]
    got = streams["H3"]
    assert len(got) == len(golden)
    for row, want in zip(got, golden):
        assert row[0] == want[0]
        for g, w in zip(row[1:], want[1:]):
            assert g == w  # byte-identical (strict float equality)
    for hyp in ("H1", "H2"):
        decayer = SourceDecayer(
            ContinuumParameters(decay_hypothesis=hyp),
            np.random.default_rng(H3_GOLDEN_SEED),
        )
        other = [
            (p.pdg_id, p.p4.e, p.p4.px, p.p4.py, p.p4.pz) for p in decayer.decay(source)
        ]
        assert other != streams["H3"]


# --- Conservation and integer QN (acceptance criterion 1) ----------------


@pytest.mark.parametrize(
    "hypothesis, n_decays", (("H1", 10), ("H2", 200), ("H3", 200))
)
def test_direct_decays_conserve_mass_momentum_and_qn(hypothesis, n_decays):
    source = _make_source(3.0)
    decayer = SourceDecayer(
        ContinuumParameters(decay_hypothesis=hypothesis),
        np.random.default_rng(DIRECT_SEED),
    )
    for _ in range(n_decays):
        daughters = decayer.decay(source)
        e_tot = sum(d.p4.e for d in daughters)
        p_tot = sum((d.p4.spatial() for d in daughters), np.zeros(3))
        assert e_tot == pytest.approx(source.mass, rel=1e-12)
        assert float(np.linalg.norm(p_tot)) <= 1e-9
        qn = sum((d.quantum_numbers for d in daughters), QuantumNumbers())
        assert qn == source.quantum_numbers


@pytest.mark.parametrize(
    "hypothesis, n_events", (("H1", 8), ("H2", 50), ("H3", 50))
)
def test_builder_events_validate_per_hypothesis(hypothesis, n_events):
    # Full builder (regge production, bundled default parameters, the
    # hypothesis injected through the committed continuum selector).
    config = PPExcitationConfig(
        production_mode="regge",
        continuum=ContinuumParameters(decay_hypothesis=hypothesis),
    )
    builder = PPEventBuilder(config, np.random.default_rng(BUILDER_SEED))
    for event_id in range(n_events):
        event = builder.generate(event_id)
        event.validate()
        assert len(event.sources) == 2


# --- H1 phase space (acceptance criterion 2) -----------------------------


def _pstar(parent: float, m1: float, m2: float) -> float:
    """Scalar two-body momentum (Kallen, threshold-safe)."""
    a, b, c = parent * parent, m1 * m1, m2 * m2
    lam = a * a + b * b + c * c - 2.0 * (a * b + a * c + b * c)
    return 0.5 * math.sqrt(max(lam, 0.0)) / parent


def test_h1_three_body_matches_analytic_uniform_phase_space():
    # Pins the H1 Kallen constant-term fix (c085269): H1 must reproduce
    # the analytic uniform-phase-space density. Pre-registered: seed
    # 20260911, 20000 decays, 20 bins over the
    # pair-invariant-mass support [2 m_pi, M - m_pi]. The analytic
    # uniform-phase-space density for the (daughter 1, daughter 2) pair
    # mass x is pstar(M, m, x) * pstar(x, m, m).
    m = M_PI
    mass = H1_SOURCE_MASS
    lo, hi = 2.0 * m, mass - m
    edges = np.linspace(lo, hi, H1_NBINS + 1)
    rng = np.random.default_rng(H1_SEED)
    pair_masses = [
        _invariant_mass(ds[1], ds[2])
        for ds in (_decay_phsp(mass, [m, m, m], rng) for _ in range(H1_N))
    ]
    obs, _ = np.histogram(pair_masses, bins=edges)

    grid = np.linspace(lo, hi, 4001)
    rho = np.array([_pstar(mass, m, x) for x in grid]) * np.array(
        [_pstar(x, m, m) for x in grid]
    )
    expected_total = float(_trapezoid(rho, grid))
    expected = np.array(
        [
            H1_N
            * float(_trapezoid(rho[(grid >= edges[i]) & (grid < edges[i + 1])],
                             grid[(grid >= edges[i]) & (grid < edges[i + 1])]))
            / expected_total
            for i in range(H1_NBINS)
        ]
    )
    chi2_value = float(np.sum((obs - expected) ** 2 / expected))
    assert chi2_value <= H1_CHI2_BOUND


def test_h1_pstar_helpers_match_scalar_two_body_momentum():
    # Pins the H1 Kallen constant-term fix (c085269): the vectorized
    # ladders must match the scalar two_body_momentum.
    m = M_PI
    parent = 1.0
    grid = np.linspace(2.0 * m, parent - m, 65)
    got_pair = _phsp_pstar_pair(grid, m, m)
    want_pair = np.array([two_body_momentum_safe(x, m, m) for x in grid])
    assert np.allclose(got_pair, want_pair, rtol=1e-12, atol=1e-14)
    got_parent = _phsp_pstar_parent(parent, m, grid)
    want_parent = np.array([two_body_momentum_safe(parent, m, x) for x in grid])
    assert np.allclose(got_parent, want_parent, rtol=1e-12, atol=1e-14)
    g2 = np.linspace(2.0 * m, parent - m, 9)
    got_grid = _phsp_pstar_grid(grid, m, g2)
    want_grid = np.array(
        [[two_body_momentum_safe(x, m, y) for y in g2] for x in grid]
    )
    assert np.allclose(got_grid, want_grid, rtol=1e-12, atol=1e-14)


def test_h1_pstar_helpers_vanish_at_threshold():
    # Pins the H1 Kallen constant-term fix (c085269): zero exactly at
    # the two-body threshold, per the two_body_momentum_safe contract.
    m = M_PI
    assert _phsp_pstar_pair(np.array([2.0 * m]), m, m)[0] == 0.0
    assert _phsp_pstar_parent(3.0 * m, m, np.array([2.0 * m]))[0] == 0.0
    assert _phsp_pstar_grid(np.array([3.0 * m]), m, np.array([2.0 * m]))[0, 0] == 0.0


def test_h1_three_body_at_exact_threshold_raises_degenerate_ladder():
    m = M_PI
    rng = np.random.default_rng(31)
    with pytest.raises(ValueError, match="degenerate phase-space ladder"):
        _decay_phsp(3.0 * m, [m, m, m], rng)


def test_h1_two_body_at_exact_threshold_emits_threshold_point():
    m = M_PI
    rng = np.random.default_rng(32)
    daughters = _decay_phsp(2.0 * m, [m, m], rng)
    assert daughters[0].e == pytest.approx(m, abs=1e-12)
    assert daughters[1].e == pytest.approx(m, abs=1e-12)
    assert float(np.linalg.norm(daughters[0].spatial())) == 0.0
    assert float(np.linalg.norm(daughters[1].spatial())) == 0.0


def test_h1_two_body_below_threshold_raises():
    m = M_PI
    rng = np.random.default_rng(33)
    with pytest.raises(ValueError, match="exceeds source mass"):
        _decay_phsp(2.0 * m - 1e-3, [m, m], rng)


@pytest.mark.parametrize("margin", (1e-6, 1e-4, 0.01))
def test_h1_three_body_near_threshold_still_samples(margin):
    m = M_PI
    mass = 3.0 * m + margin
    daughters = _decay_phsp(mass, [m, m, m], np.random.default_rng(34))
    assert len(daughters) == 3
    assert sum(d.e for d in daughters) == pytest.approx(mass, rel=1e-9)


# --- H2 resonance pairing and line shapes (acceptance criterion 3) ------


@pytest.mark.parametrize(
    "ids, res_id",
    [
        ((2212, 211), 2224),
        ((2212, 111), 2214),
        ((2212, -211), 2114),
        ((2112, 211), 2214),
        ((2112, 111), 2114),
        ((2112, -211), 1114),
        ((-2212, -211), -2224),
        ((-2212, 111), -2214),
        ((-2212, 211), -2114),
        # Antineutron rows are the charge conjugates of the neutron rows
        # (M7 finding fixed at c085269; see
        # test_h2_delta_table_is_charge_conjugate_consistent).
        ((-2112, -211), -2214),
        ((-2112, 111), -2114),
        ((-2112, 211), -1114),
    ],
)
def test_h2_delta_pairing_table(ids, res_id):
    assert _h2_resonant_pairs(list(ids)) == [(0, 1, res_id)]


def test_h2_delta_table_is_charge_conjugate_consistent():
    # Pins the antineutron Delta-bar label fix (c085269). The pairing
    # table is documented as "by charge-conjugate pair": the
    # antinucleon entry must be the charge conjugate of the nucleon
    # entry.
    for nucleon in (2212, 2112):
        for pion in (211, 111, -211):
            # pi0 (id 111) is its own antiparticle; negate charged pions.
            pion_conj = 111 if pion == 111 else -pion
            _, _, res = _h2_resonant_pairs([nucleon, pion])[0]
            _, _, res_conj = _h2_resonant_pairs([-nucleon, pion_conj])[0]
            assert res_conj == -res, (
                f"({nucleon}, {pion}) -> {res} but charge conjugate "
                f"({-nucleon}, {pion_conj}) -> {res_conj} != {-res}"
            )


@pytest.mark.parametrize(
    "ids, res_id",
    [
        ((211, -211), 113),
        ((211, 111), 213),
        ((-211, 211), 113),
        ((-211, 111), -213),
    ],
)
def test_h2_rho_pairing_table(ids, res_id):
    assert _h2_resonant_pairs(list(ids)) == [(0, 1, res_id)]


def test_h2_pairing_is_order_dependent_and_pi0_never_opens():
    # The opener must precede its partner in the permuted id list.
    assert _h2_resonant_pairs([211, 2212]) == []
    # pi0 is never an opener and never a nucleon partner.
    for ids in ((111, 211), (111, -211), (111, 111), (211, 211)):
        assert _h2_resonant_pairs(list(ids)) == []


def test_h2_pairing_composite_sources():
    assert _h2_resonant_pairs([111, 211, -211]) == [(1, 2, 113)]
    assert _h2_resonant_pairs([211, -211, 111]) == [(0, 1, 113)]
    assert _h2_resonant_pairs([2212, 211, -211, 111]) == [(0, 1, 2224), (2, 3, -213)]
    assert _h2_resonant_pairs([211, 211, -211, -211]) == [(0, 2, 113), (1, 3, 113)]
    assert _h2_resonant_pairs([211, 111, 111]) == [(0, 1, 213)]


@pytest.mark.parametrize(
    "d1, d2, res, damp",
    [
        (2212, 211, 2224, (1.6, 2.0)),
        (2212, 111, 2214, (1.6, 2.0)),
        (2212, -211, 2114, (1.6, 2.0)),
        (2112, 211, 2214, (1.6, 2.0)),
        (2112, 111, 2114, (1.6, 2.0)),
        (2112, -211, 1114, (1.6, 2.0)),
        (211, -211, 113, (1.2, 1.5)),
        (211, 111, 213, (1.2, 1.5)),
        (-211, 211, 113, (1.2, 1.5)),
        (-211, 111, -213, (1.2, 1.5)),
    ],
)
def test_h2_line_shapes_built_from_pdg_mass_and_width(d1, d2, res, damp):
    component = _res_component(d1, d2, res)
    bw = _h2_line_shape(component)
    # Parameter order: (schwel, rmas, brei, pbrei, dumplow, dumphigh).
    assert bw.get_parameter(0) == get_mass(d1) + get_mass(d2)
    assert bw.get_parameter(1) == get_mass(abs(res))
    assert bw.get_parameter(2) == get_width(abs(res))
    assert bw.get_parameter(3) == 1.0
    assert (bw.get_parameter(4), bw.get_parameter(5)) == damp


def test_h2_resonance_mass_stays_in_window_and_empty_window_returns_threshold():
    component = _res_component(2212, 211, 2224)
    lo = component.min_mass  # m_p + m_pi+
    hi = 3.0 - M_PI0
    rng = np.random.default_rng(H2_WINDOW_SEED)
    draws = [_h2_resonance_mass(component, 3.0, M_PI0, rng) for _ in range(500)]
    assert min(draws) >= lo - 1e-9
    # The draw is truncated at the damping edge (2.0 for the Delta family),
    # which sits below the kinematic window edge here.
    assert max(draws) <= min(hi, 2.0) + 1e-9
    assert float(np.std(draws)) > 0.02  # a line shape, not a threshold delta
    # Empty window (cluster mass at the daughters' threshold) emits at
    # threshold with no line.
    assert _h2_resonance_mass(component, lo + M_PI0, M_PI0, rng) == lo


def test_h2_generated_events_open_only_resonances_above_threshold():
    # P_LIKE source at 3.0 GeV: the base (p, pi0) channel opens Delta(+)
    # on most permutations; extra pions open rho lines. Every emitted
    # resonance pair must sit above its daughter-mass threshold.
    source = _make_source(3.0)
    decayer = SourceDecayer(
        ContinuumParameters(decay_hypothesis="H2"),
        np.random.default_rng(H2_GENERATED_SEED),
    )
    n_delta = n_rho = 0
    margins = []
    for _ in range(300):
        daughters = decayer.decay(source)
        ids = [d.pdg_id for d in daughters]
        for a, b, res_id in _h2_resonant_pairs(ids):
            m_inv = _invariant_mass(daughters[a].p4, daughters[b].p4)
            threshold = get_mass(ids[a]) + get_mass(ids[b])
            margins.append(m_inv - threshold)
            if abs(res_id) in (2224, 2214, 2114, 1114):
                n_delta += 1
            else:
                n_rho += 1
    assert n_delta >= 100
    assert n_rho >= 50
    assert min(margins) > -1e-9
