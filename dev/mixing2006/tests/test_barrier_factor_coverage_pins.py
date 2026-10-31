"""Coverage pins for the committed Blatt-Weisskopf barrier behavior
(Round 2, card 4; the test-author extension of
``tests/test_barrier_factors.py``).

Task ``TASK-20260913-r2-barrier-factors`` (plan section 22 task 12;
plan section 8.2 step 2).  Retargeted by the card-4 re-open of the
card-8 check round (the standard Blatt-Weisskopf polynomials support
every ``L >= 0``; the committed catalog carries F-wave ``L = 3`` and
G-wave ``L = 4`` channels): the committed module behavior pinned here
is the extended barrier at the re-open's working tree (the card's own
deterministic tests live in ``tests/test_barrier_factors.py`` and its
statistical test there runs the default barrier):

- ``test_f1_f2_at_several_q``: F_1 and F_2 against the documented
  polynomials at several q (zero, small, large) for
  r0 = BARRIER_R0 = 1.0 GeV^-1 to 1e-12, with the physical direction
  (F_1 suppresses, the re-normalized F_2 enhances, both monotonic for
  q > 0).
- ``test_l_rejection_outside_supported_partial_waves``: any negative L
  is rejected with a ValueError naming the L (the re-opened surface:
  every non-negative integer L is supported by the standard
  polynomials, while L 0, 1, 2 keep the documented forms on the same
  instance).
- ``test_hard_zero_returned_before_barrier_evaluation``: the hard zero
  below the threshold (and the closed-at-pole zero) is returned before
  the barrier is evaluated (a sentinel barrier that raises on any
  call completes the call unharmed); the barrier is finite at the
  threshold (F_L(0) = 1); the open, above-threshold path evaluates the
  barrier.
- ``test_pole_ratio_is_one_bit_exact``: at the pole mass the barrier
  ratio is 1 exactly, so the default-barrier width at the pole is the
  nominal width bit-exactly for every supported L (the BR = 1
  fixture); the two-channel BR-summing-to-1 fixture holds at the
  card's 1e-9 pole tolerance.
- ``test_identity_barrier_retained_as_identity``: IDENTITY_BARRIER is
  still exported at module level and remains F_L == 1 for every
  supported L and q >= 0 (the card-3 tests run under it); it is a
  distinct singleton from the Blatt-Weisskopf default.
- ``test_default_barrier_is_blatt_weisskopf_not_identity``: a fresh
  call (no explicit barrier argument) runs the Blatt-Weisskopf
  default on channel_width, total_width, spectral_function, and
  draw_mass: bit-identical to an explicit BLATT_WEISSKOPF_BARRIER
  call, matching the hand-computed Blatt-Weisskopf law at a non-pole
  mass, and distinguishable from the identity barrier (the squared
  barrier ratio differs from 1 off the pole).
- ``test_default_spectral_unit_integral_and_hand_law``: the default
  line shape integrates to 1 over the support (single-channel and
  channel-summed two-channel fixture) and is 0.0 outside; the density
  at three non-pole masses equals the hand-computed Blatt-Weisskopf
  shape divided by its own hand norm; a repeated evaluation is
  bit-identical (the pure compute cache).

Retargeted by the card-8 re-open of card 3 (round-2 fix round,
2026-09-14): the spectral normalization constant is the trapezoid
integral of the unnormalized shape on the pre-registered 10,001-node
reference grid (``_NORMALIZATION_GRID_POINTS``, the card-8
``SPECTRAL_QUADRATURE_NODES``), not the original 1024-point grid; the
unit-integral and hand-norm pins of
``test_default_spectral_unit_integral_and_hand_law`` run on that grid
at unchanged tolerances.  The 1024-point convention was
under-resolved for the narrow-width records (the card-8 check read
0.3334 for the eta and 1.0380 for the eta-prime on the pre-registered
grid).

Hermetic: hand-constructed records (Delta(1232)-like fixtures built on
the vendored-table masses of the card-2 catalog); no event generation,
no external tables, no catalog file reads.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from na49_mixing.hadron_catalog import (  # noqa: E402
    DecayChannel,
    HadronRecord,
)
from na49_mixing.resonance_widths import (  # noqa: E402
    BARRIER_R0,
    BLATT_WEISSKOPF_BARRIER,
    IDENTITY_BARRIER,
    _NORMALIZATION_GRID_POINTS,
    BlattWeisskopfBarrier,
    channel_width,
    draw_mass,
    spectral_function,
    total_width,
)

# ----------------------------------------------------------------------
# Hand-constructed Delta(1232)-like fixtures (the same hand-constructed
# record as the card's tests: card-2 catalog values, the vendored-table
# daughter masses the catalog thresholds are built from).  Provenance
# is None by design: hand-constructed fixtures sit before the catalog
# import, where the card-1 schema permits it.
# ----------------------------------------------------------------------
MP = 0.93827231            # vendored table proton mass, GeV/c^2
MPI = 0.1395699            # vendored table pi+ mass, GeV/c^2
MN = 0.93956541            # vendored table neutron mass, GeV/c^2
POLE = 1.232               # Delta(1232) pole mass, GeV/c^2
WIDTH = 0.117              # Delta(1232) nominal width, GeV

DAUGHTER_MASSES = {2212: MP, 211: MPI}
MASSES_2CH = {**DAUGHTER_MASSES, 2112: MN}

THRESHOLD_P_PION = math.fsum((MP, MPI))    # 1.07784221 (Delta++ -> p pi+)
M_UP = 2.0                                 # the documented dumphigh cut


def _channel(L: int = 1) -> DecayChannel:
    """One hand-constructed two-body Delta++ -> p pi+ channel."""
    return DecayChannel(
        parent_pdg=2224,
        daughters=(2212, 211),
        branching_fraction=1.0,
        orbital_angular_momentum=L,
        is_three_body=False,
        provenance=None,
    )


def _record(channels, nominal_width: float = WIDTH) -> HadronRecord:
    """A hand-constructed Delta(1232)-like catalog record."""
    lo = max(
        math.fsum(float(DAUGHTER_MASSES[int(d)]) for d in ch.daughters)
        for ch in channels
    )
    return HadronRecord(
        pdg_id=2224,
        name="Delta++(fixture)",
        B=1,
        Q=2,
        S=0,
        isospin=1.5,
        isospin3=3,
        spin=1.5,
        parity="+",
        pole_mass=POLE,
        nominal_width=nominal_width,
        mass_range=(lo, M_UP),
        decay_channels=tuple(channels),
        stable_for_decay=False,
        provenance=None,
    )


# ----------------------------------------------------------------------
# Helpers (independent arithmetic paths and the sentinel barrier)
# ----------------------------------------------------------------------
def _trapezoid(xs: np.ndarray, ys: np.ndarray) -> float:
    """Trapezoid integral (the module's documented convention)."""
    return float((0.5 * (ys[1:] + ys[:-1]) * (xs[1:] - xs[:-1])).sum())


def _hand_q_pair(M: float, mb: float, mc: float) -> float:
    """Decay momentum of the pair (mb, mc) from the factored Kaellen
    form (an independent computation path from the module's
    ``kallen``)."""
    lam = (M * M - (mb + mc) ** 2) * (M * M - (mb - mc) ** 2)
    return math.sqrt(lam) / (2.0 * M)


def _hand_f(L: int, q: float) -> float:
    """The Blatt-Weisskopf polynomials written out for L in (0, 1, 2)
    with r0 = BARRIER_R0 (the q side runs on the factored Kaellen
    form; the polynomials are the documented convention)."""
    z = (q * BARRIER_R0) ** 2
    if L == 0:
        return 1.0
    if L == 1:
        return 1.0 / math.sqrt(1.0 + z)
    return math.sqrt(9.0 / (z * z + 3.0 * z + 9.0))


def _hand_total_width(
    M: float, record: HadronRecord, daughter_masses
) -> float:
    """The plan section 8.2 law with the Blatt-Weisskopf barrier,
    hand-computed per channel from the factored Kaellen form and the
    written-out polynomials (each channel on its own leading pair)."""
    M0 = float(record.pole_mass)
    parts = []
    for channel in record.decay_channels:
        mb = float(daughter_masses[int(channel.daughters[0])])
        mc = float(daughter_masses[int(channel.daughters[1])])
        L = int(channel.orbital_angular_momentum)
        q0 = _hand_q_pair(M0, mb, mc)
        qm = _hand_q_pair(M, mb, mc)
        gamma_c0 = (
            float(channel.branching_fraction)
            * float(record.nominal_width)
        )
        parts.append(
            gamma_c0
            * (qm / q0) ** (2 * L + 1)
            * (M0 / M)
            * (_hand_f(L, qm) / _hand_f(L, q0)) ** 2
        )
    return math.fsum(parts)


def _two_channel_record() -> HadronRecord:
    """The hand-constructed two-channel Delta+-like record with BR
    0.6667 (p pi+) + 0.3333 (n pi+) (the card-3 fixture)."""
    return HadronRecord(
        pdg_id=2214,
        name="Delta+(fixture)",
        B=1,
        Q=2,
        S=0,
        isospin=1.5,
        isospin3=3,
        spin=1.5,
        parity="+",
        pole_mass=POLE,
        nominal_width=WIDTH,
        mass_range=(math.fsum((MN, MPI)), M_UP),
        decay_channels=(
            DecayChannel(
                parent_pdg=2214,
                daughters=(2212, 211),
                branching_fraction=0.6667,
                orbital_angular_momentum=1,
                is_three_body=False,
                provenance=None,
            ),
            DecayChannel(
                parent_pdg=2214,
                daughters=(2112, 211),
                branching_fraction=0.3333,
                orbital_angular_momentum=1,
                is_three_body=False,
                provenance=None,
            ),
        ),
        stable_for_decay=False,
        provenance=None,
    )


class _SentinelBarrier:
    """A barrier that raises on evaluation, so a width-law path that
    returns before the barrier is evaluated completes unharmed."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, L: int, q: float) -> float:
        self.calls += 1
        raise AssertionError("the barrier must not be evaluated")


# ----------------------------------------------------------------------
# Coverage pins (the committed module behavior, nested 215630e)
# ----------------------------------------------------------------------
def test_f1_f2_at_several_q():
    """F_1 and F_2 against the documented polynomials at several q
    (zero, small, large) for r0 = BARRIER_R0 = 1.0 GeV^-1, to 1e-12,
    and the physical direction: both the L = 1 and the L = 2 factors
    suppress for q > 0, monotonically (the pre-registered L = 2 form
    that grew with q was corrected on 2026-09-28, FAB-08)."""
    barrier = BLATT_WEISSKOPF_BARRIER
    assert BARRIER_R0 == 1.0
    for q in (0.0, 0.05, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0):
        z = (q * BARRIER_R0) ** 2
        f1 = 1.0 / math.sqrt(1.0 + z)
        f2 = math.sqrt(9.0 / (z * z + 3.0 * z + 9.0))
        assert abs(barrier(1, q) - f1) <= 1e-12
        assert abs(barrier(2, q) - f2) <= 1e-12
    assert barrier(1, 0.0) == 1.0 and barrier(2, 0.0) == 1.0
    assert barrier(1, 0.1) > barrier(1, 1.0) > barrier(1, 10.0)
    assert barrier(1, 10.0) < 1.0
    assert barrier(2, 0.1) > barrier(2, 1.0) > barrier(2, 10.0)
    assert barrier(2, 0.1) < 1.0


def test_l_rejection_outside_supported_partial_waves():
    """Any negative L is rejected with a ValueError naming the L (a
    negative orbital angular momentum is not a partial wave), while
    every non-negative integer L is accepted on the same instance
    (the standard von Hippel-Quigg polynomials for L >= 3; the
    documented forms for L 0, 1, 2)."""
    barrier = BLATT_WEISSKOPF_BARRIER
    for L in (-2, -1):
        with pytest.raises(ValueError, match=f"L = {L}"):
            barrier(L, 0.5)
    custom = BlattWeisskopfBarrier(r0=1.0)
    for L in (0, 1, 2, 3, 4, 5, 6):
        assert custom(L, 0.5) > 0.0
    with pytest.raises(ValueError, match="L = -1"):
        custom(-1, 0.5)


def test_hard_zero_returned_before_barrier_evaluation():
    """The hard zero below the channel threshold (and the
    closed-at-pole zero) is returned before the barrier is evaluated:
    a sentinel barrier that raises on any call completes the call
    unharmed.  The barrier is finite at the threshold itself
    (F_L(0) = 1), so the threshold continuity pinned by
    test_zero_below_threshold_unchanged in tests/test_barrier_factors.py
    rests on a finite ratio, not a singularity.  The open,
    above-threshold path evaluates the barrier (the sentinel raises):
    the sentinel is unused only on the early-return paths."""
    sentinel = _SentinelBarrier()
    record = _record([_channel(L=1)])
    channel = record.decay_channels[0]
    for M in (
        THRESHOLD_P_PION,
        THRESHOLD_P_PION - 1e-12,
        THRESHOLD_P_PION - 1e-6,
    ):
        assert (
            channel_width(M, channel, record, DAUGHTER_MASSES, sentinel)
            == 0.0
        )
    assert (
        total_width(
            THRESHOLD_P_PION - 1e-6, record, DAUGHTER_MASSES, sentinel
        )
        == 0.0
    )
    assert sentinel.calls == 0
    for L in (0, 1, 2, 3, 4):
        assert BLATT_WEISSKOPF_BARRIER(L, 0.0) == 1.0

    # The closed-at-pole zero (q(M0) = 0) also returns before the
    # barrier is evaluated.
    closed = HadronRecord(
        pdg_id=9995,
        name="closed-pole(fixture)",
        B=1,
        Q=2,
        S=0,
        isospin=1.5,
        isospin3=3,
        spin=1.5,
        parity="+",
        pole_mass=1.05,
        nominal_width=WIDTH,
        mass_range=(THRESHOLD_P_PION, M_UP),
        decay_channels=(channel,),
        stable_for_decay=False,
        provenance=None,
    )
    for M in (THRESHOLD_P_PION, THRESHOLD_P_PION + 1e-6, 1.2, 1.5):
        assert (
            channel_width(M, channel, closed, DAUGHTER_MASSES, sentinel)
            == 0.0
        )
    assert sentinel.calls == 0

    # Positive control: the open, above-threshold path evaluates the
    # barrier.
    with pytest.raises(AssertionError):
        channel_width(1.2, channel, record, DAUGHTER_MASSES, sentinel)
    assert sentinel.calls == 1


def test_pole_ratio_is_one_bit_exact():
    """At the pole mass the barrier ratio (F_L(q0) / F_L(q0))**2 and
    the phase-space factor are 1.0 exactly, so the default-barrier
    width at the pole is the nominal width bit-exactly (not merely to
    1e-9) for every supported L in (0, 1, 2, 3, 4) on the BR = 1
    fixture; the two-channel fixture with BR summing to 1 holds at the
    card's 1e-9 pole tolerance."""
    for L in (0, 1, 2, 3, 4):
        record = _record([_channel(L=L)])
        channel = record.decay_channels[0]
        assert channel_width(POLE, channel, record, DAUGHTER_MASSES) == WIDTH
        assert total_width(POLE, record, DAUGHTER_MASSES) == WIDTH
    assert abs(
        total_width(POLE, _two_channel_record(), MASSES_2CH) - WIDTH
    ) <= 1e-9


def test_identity_barrier_retained_as_identity():
    """IDENTITY_BARRIER is still exported at module level and remains
    the identity F_L == 1 for every supported L and q >= 0 (the
    card-3 width-law tests in tests/test_resonance_widths.py run under
    it); it is a distinct singleton from the Blatt-Weisskopf default,
    which is itself a BlattWeisskopfBarrier with the documented r0."""
    import na49_mixing.resonance_widths as resonance_widths

    assert resonance_widths.IDENTITY_BARRIER is IDENTITY_BARRIER
    assert (
        resonance_widths.BLATT_WEISSKOPF_BARRIER is BLATT_WEISSKOPF_BARRIER
    )
    assert IDENTITY_BARRIER is not BLATT_WEISSKOPF_BARRIER
    for L in (0, 1, 2, 3, 4):
        for q in (0.0, 0.05, 0.227, 1.0, 5.0):
            assert IDENTITY_BARRIER(L, q) == 1.0
    assert isinstance(BLATT_WEISSKOPF_BARRIER, BlattWeisskopfBarrier)
    fresh = BlattWeisskopfBarrier(r0=BARRIER_R0)
    for L in (0, 1, 2, 3, 4):
        for q in (0.0, 0.227, 1.0):
            assert BLATT_WEISSKOPF_BARRIER(L, q) == fresh(L, q)


def test_default_barrier_is_blatt_weisskopf_not_identity():
    """A fresh call (no explicit barrier argument) runs the documented
    Blatt-Weisskopf default: channel_width, total_width,
    spectral_function, and draw_mass all agree bit-identically with an
    explicit BLATT_WEISSKOPF_BARRIER call, and at the non-pole mass
    M = 1.6 GeV they differ from the explicit IDENTITY_BARRIER call
    (the barrier ratio (F_1(q(M)) / F_1(q(M0)))**2 differs from 1 off
    the pole).  The default-barrier width matches the hand-computed
    Blatt-Weisskopf law, and the two barrier values differ by exactly
    the squared barrier ratio (the phase-space factor is
    barrier-independent)."""
    record = _record([_channel(L=1)])
    channel = record.decay_channels[0]
    M = 1.6
    q0_hand = _hand_q_pair(POLE, MP, MPI)
    qm_hand = _hand_q_pair(M, MP, MPI)
    assert q0_hand > 0.0 and qm_hand > 0.0

    f_ratio_sq = (_hand_f(1, qm_hand) / _hand_f(1, q0_hand)) ** 2
    assert abs(f_ratio_sq - 1.0) > 0.05
    hand_bw = WIDTH * (qm_hand / q0_hand) ** 3 * (POLE / M) * f_ratio_sq
    hand_id = WIDTH * (qm_hand / q0_hand) ** 3 * (POLE / M)

    w_default = channel_width(M, channel, record, DAUGHTER_MASSES)
    w_bw = channel_width(
        M, channel, record, DAUGHTER_MASSES, BLATT_WEISSKOPF_BARRIER
    )
    w_id = channel_width(
        M, channel, record, DAUGHTER_MASSES, IDENTITY_BARRIER
    )
    assert w_default == w_bw
    assert abs(w_default - hand_bw) <= 1e-9 * hand_bw
    assert abs(w_id - hand_id) <= 1e-9 * hand_id
    assert w_default != w_id
    # The two barrier values differ by exactly the squared barrier
    # ratio (the phase-space factor is barrier-independent).
    assert abs(w_default / w_id - f_ratio_sq) <= 1e-9 * f_ratio_sq

    t_default = total_width(M, record, DAUGHTER_MASSES)
    t_bw = total_width(
        M, record, DAUGHTER_MASSES, BLATT_WEISSKOPF_BARRIER
    )
    t_id = total_width(M, record, DAUGHTER_MASSES, IDENTITY_BARRIER)
    assert t_default == t_bw
    assert t_default != t_id
    # Single channel: the channel-summed total is the channel width.
    assert t_default == w_default

    s_default = spectral_function(M, record, DAUGHTER_MASSES)
    s_bw = spectral_function(
        M, record, DAUGHTER_MASSES, BLATT_WEISSKOPF_BARRIER
    )
    s_id = spectral_function(M, record, DAUGHTER_MASSES, IDENTITY_BARRIER)
    assert s_default > 0.0
    assert s_default == s_bw
    assert s_default != s_id

    window = (1.4, 1.8)
    d_default = draw_mass(
        window, record, DAUGHTER_MASSES, np.random.default_rng(20260913)
    )
    d_bw = draw_mass(
        window, record, DAUGHTER_MASSES,
        np.random.default_rng(20260913), BLATT_WEISSKOPF_BARRIER,
    )
    d_id = draw_mass(
        window, record, DAUGHTER_MASSES,
        np.random.default_rng(20260913), IDENTITY_BARRIER,
    )
    assert d_default is not None and d_bw is not None and d_id is not None
    assert d_default == d_bw
    assert d_default != d_id


def test_default_spectral_unit_integral_and_hand_law():
    """The default (Blatt-Weisskopf) line shape: unit integral over
    the support under the documented 10,001-node normalization
    quadrature (``_NORMALIZATION_GRID_POINTS``, the pre-registered
    reference grid of the round-2 check suite; retargeted from the
    1024-point convention by the card-8 re-open fix round, tolerances
    unchanged) and exactly 0.0 outside (single-channel and the
    channel-summed two-channel fixture, whose separate cache entries
    keep the two barriers and the two records distinct); the density
    at three non-pole masses equals the hand-computed Blatt-Weisskopf
    shape divided by its own hand norm on the same grid (the
    independent factored-Kaellen arithmetic path); a repeated
    evaluation is bit-identical to the first (the pure compute cache
    stores one grid result)."""
    record = _record([_channel(L=1)])
    lo, hi = record.mass_range
    xs = np.linspace(lo, hi, _NORMALIZATION_GRID_POINTS)
    ys = np.array(
        [spectral_function(float(x), record, DAUGHTER_MASSES) for x in xs]
    )
    assert abs(_trapezoid(xs, ys) - 1.0) <= 1e-9
    for M in (lo - 1e-9, 0.5, hi + 1e-9):
        assert spectral_function(M, record, DAUGHTER_MASSES) == 0.0

    def _hand_shape(M: float) -> float:
        gamma = _hand_total_width(M, record, DAUGHTER_MASSES)
        return (
            M * M * gamma
            / ((M * M - POLE * POLE) ** 2 + POLE * POLE * gamma * gamma)
        )

    hand_ys = np.array([_hand_shape(float(x)) for x in xs])
    hand_norm = _trapezoid(xs, hand_ys)
    assert hand_norm > 0.0
    for M in (1.2, 1.6, 1.9):
        assert (
            abs(
                spectral_function(M, record, DAUGHTER_MASSES)
                - _hand_shape(M) / hand_norm
            )
            <= 1e-12
        )

    # The channel-summed two-channel fixture under the default
    # barrier: unit integral, and a bit-identical repeated
    # evaluation (the cached grid result).
    record2 = _two_channel_record()
    lo2, hi2 = record2.mass_range
    xs2 = np.linspace(lo2, hi2, _NORMALIZATION_GRID_POINTS)
    ys2 = np.array(
        [spectral_function(float(x), record2, MASSES_2CH) for x in xs2]
    )
    assert abs(_trapezoid(xs2, ys2) - 1.0) <= 1e-9
    for M in (1.2, POLE, 1.8):
        first = spectral_function(M, record2, MASSES_2CH)
        assert first > 0.0
        assert first == spectral_function(M, record2, MASSES_2CH)
