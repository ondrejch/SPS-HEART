"""Blatt-Weisskopf barrier-factor tests (Round 2, card 4).

Task ``TASK-20260913-r2-barrier-factors`` (plan section 22 task 12;
plan section 8.2 step 2), extended by the card-4 re-open of the
card-8 check round: the standard Blatt-Weisskopf polynomials now
support every ``L >= 0`` (the committed catalog carries F-wave
``L = 3`` channels on N*(1680) and rho3(1690) and a G-wave ``L = 4``
channel on f4(2050)).  The deterministic card tests:

- ``test_f0_is_one``: ``F_0(q) == 1.0`` for all checked ``q >= 0``.
- ``test_f1_values``: ``F_1(1.0) == 1/sqrt(2)`` to ``1e-12`` (with
  ``r0 = 1.0``); ``F_1(0.0) == 1.0``.
- ``test_f2_values``: ``F_2(0.0) == 1.0`` (re-normalization);
  ``F_2(1.0) == sqrt(9/5)`` to ``1e-12``.
- ``test_l3_l4_values``: the standard von Hippel-Quigg members --
  ``F_3(q)^2 = 225/(z^3 + 6z^2 + 45z + 225)`` and ``F_4(q)^2 =
  11025/(z^4 + 10z^3 + 135z^2 + 1575z + 11025)`` with ``z = (q r0)^2``
  -- to ``1e-12`` at the checked momenta, ``F_3(0) = F_4(0) = 1``,
  with the ``r0`` scaling and the written-out polynomials as the
  independent arithmetic path.
- ``test_negative_l_rejected``: a negative L raises ``ValueError``
  naming the L (a negative orbital angular momentum is not a partial
  wave); every non-negative integer L is supported.
- ``test_pole_normalization_unchanged``: with the default barrier,
  ``total_width`` at the pole mass equals ``nominal_width`` to ``1e-9``
  (the ratio term is 1 at the pole) for every L in (0, 1, 2, 3, 4) on
  the same hand-constructed record as card 3.
- ``test_zero_below_threshold_unchanged``: the hard-zero-below-
  threshold behavior of card 3 is preserved for every supported L,
  and the width stays continuous at the threshold (the barrier is
  finite there, ``F_L(0) = 1``, so the ``(q/q0)**(2L+1)`` right limit
  still vanishes).
- ``test_l3_width_law_hand_ratio``: the plan section 8.2 width law on
  a hand-constructed ``L = 3`` channel under the default barrier
  matches the hand-computed ratio (phase-space factor times the
  written-out ``F_3`` squared ratio) to ``1e-9`` relative -- the
  width law is evaluable for ``L > 2`` (the card-8 blocker).

Statistical card test:

- ``test_barrier_threshold_suppression`` (100,000 draws per mass point
  on an L = 1 hand-constructed channel): the sampled width ratio
  ``Gamma(M_low)/Gamma(M_high)`` at the masses where the decay
  momentum is ``q = 0.1`` and ``q = 1.0`` GeV agrees with the
  hand-computed Blatt-Weisskopf ratio to 3-sigma.  Each width is
  estimated as the sample mean of ``channel_width`` over uniform mass
  draws in a ``1e-5`` GeV window around the target mass (the MC
  standard error of the mean is the sigma of the check; the window
  curvature bias of the mean is below ``1e-2`` of the 3-sigma band,
  computed in the module run record).  The draws exercise the card-4
  default barrier (no explicit ``barrier`` argument), so the test also
  pins the default swap: under the identity barrier the same ratio
  would be smaller by the factor ``F_1(0.1)^2 / F_1(1.0)^2 = 2/1.01``,
  orders of magnitude outside the 3-sigma band.

The hand-computed side of the statistical test uses the factored
Kaellen form and the explicit Blatt-Weisskopf polynomials, an
independent arithmetic path from the module's ``kallen`` and
``BlattWeisskopfBarrier``.  The masses ``M_low`` and ``M_high`` are the
exact roots of ``q(M) = q_target`` above the channel threshold, from
the quadratic ``(M^2 - a)(M^2 - b) = 4 M^2 q^2`` in ``u = M^2``.

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
    BlattWeisskopfBarrier,
    channel_width,
    decay_momentum,
    total_width,
)

# ----------------------------------------------------------------------
# Hand-constructed Delta(1232)-like fixtures (the same hand-constructed
# record as card 3: card-2 catalog values, the vendored-table daughter
# masses the catalog thresholds are built from).  Provenance is None by
# design: hand-constructed fixtures sit before the catalog import,
# where the card-1 schema permits it.
# ----------------------------------------------------------------------
MP = 0.93827231            # vendored table proton mass, GeV/c^2
MPI = 0.1395699            # vendored table pi+ mass, GeV/c^2
POLE = 1.232               # Delta(1232) pole mass, GeV/c^2
WIDTH = 0.117              # Delta(1232) nominal width, GeV

DAUGHTER_MASSES = {2212: MP, 211: MPI}

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


def _record(
    channels,
    nominal_width: float = WIDTH,
) -> HadronRecord:
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
# Deterministic card tests
# ----------------------------------------------------------------------
def test_f0_is_one():
    """F_0(q) == 1.0 for all checked q >= 0 (S-wave carries no
    centrifugal barrier), for the default singleton and a custom-r0
    instance alike."""
    barrier = BLATT_WEISSKOPF_BARRIER
    custom = BlattWeisskopfBarrier(r0=0.7)
    for q in (0.0, 1e-9, 1e-3, 0.1, 0.227, 0.5, 1.0, 3.0, 10.0, 100.0):
        assert barrier(0, q) == 1.0
        assert custom(0, q) == 1.0


def test_f1_values():
    """F_1(1.0) == 1/sqrt(2) to 1e-12 with r0 = 1.0 (the documented
    constant) and F_1(0.0) == 1.0 (the factor is 1 at zero momentum)."""
    assert BARRIER_R0 == 1.0
    barrier = BLATT_WEISSKOPF_BARRIER
    assert barrier(1, 0.0) == 1.0
    assert abs(barrier(1, 1.0) - 1.0 / math.sqrt(2.0)) <= 1e-12
    # The r0 scaling: F_1(q) = 1/sqrt(1 + (q r0)^2).
    custom = BlattWeisskopfBarrier(r0=2.0)
    assert abs(custom(1, 1.0) - 1.0 / math.sqrt(5.0)) <= 1e-12


def test_f2_values():
    """F_2(0.0) == 1.0 (the re-normalized form, F_2(0) = 1) and
    F_2(1.0) == sqrt(9/13) to 1e-12: Z_2(z) = z^2 + 3 z + 9 at z = 1, the
    standard D-wave Blatt-Weisskopf member; the factor falls with q."""
    barrier = BLATT_WEISSKOPF_BARRIER
    assert barrier(2, 0.0) == 1.0
    assert abs(barrier(2, 1.0) - math.sqrt(9.0 / 13.0)) <= 1e-12
    assert barrier(2, 0.1) > barrier(2, 1.0) > barrier(2, 10.0)


def test_l3_l4_values():
    """The standard von Hippel-Quigg members for L = 3 and L = 4 in
    the same re-normalized squared-ratio form: F_3(0) = F_4(0) = 1;
    F_3(1.0) = 15/sqrt(277) and F_4(1.0) = 105/sqrt(12746) to 1e-12
    (z = 1: Z_3 = 1 + 6 + 45 + 225 = 277, Z_4 = 1 + 10 + 135 + 1575
    + 11025 = 12746); the written-out polynomials at several q as the
    independent arithmetic path; the r0 scaling through z = (q r0)^2;
    and the suppression direction (both factors decrease with q,
    bounded by 1)."""
    barrier = BLATT_WEISSKOPF_BARRIER
    assert barrier(3, 0.0) == 1.0
    assert barrier(4, 0.0) == 1.0
    assert abs(barrier(3, 1.0) - 15.0 / math.sqrt(277.0)) <= 1e-12
    assert abs(barrier(4, 1.0) - 105.0 / math.sqrt(12746.0)) <= 1e-12

    # The written-out polynomials (independent arithmetic path):
    # F_3(q)^2 = 225/(z^3 + 6z^2 + 45z + 225) and
    # F_4(q)^2 = 11025/(z^4 + 10z^3 + 135z^2 + 1575z + 11025).
    for q in (1e-9, 0.05, 0.227, 0.5, 1.0, 2.0, 5.0, 10.0):
        z = (q * BARRIER_R0) ** 2
        f3_sq = 225.0 / (z**3 + 6.0 * z * z + 45.0 * z + 225.0)
        f4_sq = 11025.0 / (
            z**4 + 10.0 * z**3 + 135.0 * z * z + 1575.0 * z + 11025.0
        )
        assert abs(barrier(3, q) ** 2 - f3_sq) <= 1e-12
        assert abs(barrier(4, q) ** 2 - f4_sq) <= 1e-12

    # The r0 scaling enters through z = (q r0)^2.
    custom = BlattWeisskopfBarrier(r0=2.0)
    z = (1.0 * 2.0) ** 2
    assert abs(
        custom(3, 1.0)
        - math.sqrt(225.0 / (z**3 + 6.0 * z * z + 45.0 * z + 225.0))
    ) <= 1e-12
    assert abs(
        custom(4, 1.0)
        - math.sqrt(
            11025.0
            / (z**4 + 10.0 * z**3 + 135.0 * z * z + 1575.0 * z + 11025.0)
        )
    ) <= 1e-12

    # Finiteness, positivity, and the monotone suppression direction.
    for q in (0.0, 1e-9, 1e-3, 0.1, 0.227, 0.5, 1.0, 3.0, 10.0, 100.0):
        for L in (3, 4):
            value = barrier(L, q)
            assert math.isfinite(value) and value > 0.0, (L, q, value)
    assert barrier(3, 0.1) > barrier(3, 1.0) > barrier(3, 10.0)
    assert barrier(4, 0.1) > barrier(4, 1.0) > barrier(4, 10.0)
    assert barrier(3, 0.1) <= 1.0 and barrier(4, 0.1) <= 1.0


def test_negative_l_rejected():
    """A negative L raises ValueError naming the L (a negative orbital
    angular momentum is not a partial wave); every non-negative
    integer L is supported (the standard polynomials are defined for
    all L >= 0)."""
    barrier = BLATT_WEISSKOPF_BARRIER
    for L in (-2, -1):
        with pytest.raises(ValueError, match=f"L = {L}"):
            barrier(L, 0.5)
        with pytest.raises(ValueError, match=f"L = {L}"):
            BlattWeisskopfBarrier(r0=1.0)(L, 0.0)
    # Non-negative integers are all supported: F_L(0) = 1 and the
    # factor is finite and positive for every L >= 0.
    for L in (0, 1, 2, 3, 4, 5, 6):
        assert barrier(L, 0.0) == 1.0
        assert barrier(L, 0.5) > 0.0


def test_pole_normalization_unchanged():
    """With the default barrier, total_width at the pole mass equals
    nominal_width to 1e-9 on the same hand-constructed record as card
    3 for every L in (0, 1, 2, 3, 4): the barrier ratio term is 1 at
    the pole (the same momentum enters numerator and denominator).
    The card-3 identity barrier, retained as an explicit argument,
    gives the same pole width."""
    for L in (0, 1, 2, 3, 4):
        record = _record([_channel(L=L)])
        assert abs(
            total_width(POLE, record, DAUGHTER_MASSES) - WIDTH
        ) <= 1e-9
        assert abs(
            total_width(POLE, record, DAUGHTER_MASSES, IDENTITY_BARRIER)
            - WIDTH
        ) <= 1e-9
        # The default is pinned at the function level: the module
        # functions called without a barrier argument run the
        # Blatt-Weisskopf singleton.
        assert abs(
            channel_width(
                POLE, record.decay_channels[0], record, DAUGHTER_MASSES
            )
            - WIDTH
        ) <= 1e-9


def test_zero_below_threshold_unchanged():
    """The card-3 threshold behavior is preserved under the new
    default barrier: channel_width is exactly 0.0 at, at minus
    1e-12, and below the channel threshold for every L in
    (0, 1, 2, 3, 4); the channel-summed width inherits the hard zero;
    and the width is continuous at the threshold (a positive right
    limit below the card-3 1e-12 GeV bound at threshold + 1e-9 on the
    1e-9 GeV width-scale fixture, since the barrier is finite there
    with F_L(0) = 1)."""
    for L in (0, 1, 2, 3, 4):
        record = _record([_channel(L=L)])
        channel = record.decay_channels[0]
        for M in (
            THRESHOLD_P_PION,
            THRESHOLD_P_PION - 1e-12,
            THRESHOLD_P_PION - 1e-6,
        ):
            assert channel_width(M, channel, record, DAUGHTER_MASSES) == 0.0
        assert total_width(
            THRESHOLD_P_PION - 1e-12, record, DAUGHTER_MASSES
        ) == 0.0
        # Positive just above the threshold (the hard zero is a limit
        # from below, not a plateau).
        assert (
            channel_width(
                THRESHOLD_P_PION + 1e-6, channel, record, DAUGHTER_MASSES
            )
            > 0.0
        )

    # Continuity at the threshold: the card-3 continuity fixture at
    # the 1e-9 GeV width scale, now under the default barrier.  The
    # right limit stays below the card's absolute 1e-12 GeV bound
    # (the finite F_L ratio multiplies the vanishing (q/q0)**(2L+1)
    # factor by at most F_1(q0)^-2 ~ 1.05 on this fixture for every
    # L in (0, 1, 2, 3, 4)).
    tiny_width = 1e-9
    for L in (0, 1, 2, 3, 4):
        record = _record([_channel(L=L)], nominal_width=tiny_width)
        channel = record.decay_channels[0]
        value = channel_width(
            THRESHOLD_P_PION + 1e-9, channel, record, DAUGHTER_MASSES
        )
        assert 0.0 < value < 1e-12, (L, value)


# ----------------------------------------------------------------------
# Statistical card test
# ----------------------------------------------------------------------
def _hand_mass_for_q(q_target: float) -> float:
    """The parent mass above the p pi+ threshold whose two-body decay
    momentum is exactly *q_target*, from the factored Kaellen form: the
    quadratic (M^2 - a)(M^2 - b) = 4 M^2 q^2 in u = M^2 has the roots
    u = [(a + b + 4 q^2) +/- sqrt((a + b + 4 q^2)^2 - 4 a b)] / 2, and
    the physical (above-threshold) root is the + one."""
    a = (MP + MPI) ** 2
    b = (MP - MPI) ** 2
    s = a + b + 4.0 * q_target * q_target
    u = 0.5 * (s + math.sqrt(s * s - 4.0 * a * b))
    return math.sqrt(u)


def _hand_f1(q: float) -> float:
    """The L = 1 Blatt-Weisskopf factor written out (independent
    arithmetic path from the module's ``BlattWeisskopfBarrier``)."""
    return 1.0 / math.sqrt(1.0 + (q * BARRIER_R0) ** 2)


def test_barrier_threshold_suppression():
    """100,000 draws per mass point on an L = 1 hand-constructed
    channel: the sampled width ratio Gamma(M_low)/Gamma(M_high), at
    the masses where the decay momentum is q = 0.1 and q = 1.0 GeV,
    agrees with the hand-computed Blatt-Weisskopf ratio to 3-sigma
    (the 1/sqrt(N) convention of the card-3 statistical test).  Seed
    20260913 (deterministic).  The draws run the card-4 default
    barrier; under the identity barrier the hand ratio would differ by
    the factor 2/1.01, far outside the band."""
    record = _record([_channel(L=1)])
    channel = record.decay_channels[0]

    q_low, q_high = 0.1, 1.0
    m_low = _hand_mass_for_q(q_low)
    m_high = _hand_mass_for_q(q_high)
    # The hand masses reproduce the target momenta in the module's
    # independent Kaellen path.
    assert abs(decay_momentum(m_low, MP, MPI) - q_low) <= 1e-9
    assert abs(decay_momentum(m_high, MP, MPI) - q_high) <= 1e-9

    # Hand-computed Blatt-Weisskopf ratio of the plan section 8.2 law
    # at the two masses (BR = 1, L = 1):
    #   Gamma(M_low)/Gamma(M_high)
    #     = (q_low/q_high)^3 * (M_high/M_low)
    #       * (F_1(q_low)/F_1(q_high))^2.
    expected_ratio = (
        (q_low / q_high) ** 3
        * (m_high / m_low)
        * (_hand_f1(q_low) / _hand_f1(q_high)) ** 2
    )

    # Sampled widths: 100,000 uniform mass draws per mass point in a
    # 1e-5 GeV window, each width evaluated with the default
    # (Blatt-Weisskopf) barrier; the sample mean estimates the width
    # at the window center and the standard error of the mean is the
    # sigma of the check.
    n_draws = 100_000
    half_window = 1e-5
    rng = np.random.default_rng(20260913)
    means = {}
    standard_errors = {}
    for label, m_c in (("low", m_low), ("high", m_high)):
        masses = rng.uniform(
            m_c - half_window, m_c + half_window, n_draws
        )
        widths = np.array(
            [
                channel_width(float(m), channel, record, DAUGHTER_MASSES)
                for m in masses
            ]
        )
        assert (widths > 0.0).all()
        means[label] = float(widths.mean())
        standard_errors[label] = float(
            widths.std(ddof=1) / math.sqrt(n_draws)
        )

    sampled_ratio = means["low"] / means["high"]
    sigma_ratio = sampled_ratio * math.sqrt(
        (standard_errors["low"] / means["low"]) ** 2
        + (standard_errors["high"] / means["high"]) ** 2
    )
    assert sigma_ratio > 0.0
    assert abs(sampled_ratio - expected_ratio) <= 3.0 * sigma_ratio, (
        sampled_ratio,
        expected_ratio,
        sigma_ratio,
    )


# ----------------------------------------------------------------------
# Card-4 re-open (the card-8 check round): the width law under the
# default barrier for L = 3 (the catalog's F-wave channels)
# ----------------------------------------------------------------------
def test_l3_width_law_hand_ratio():
    """The plan section 8.2 law on a hand-constructed L = 3 channel
    under the default (Blatt-Weisskopf) barrier: the width ratio
    Gamma(M_low)/Gamma(M_high), at the masses where the decay momentum
    is q = 0.1 and q = 1.0 GeV, equals the hand-computed law
    (q_low/q_high)^7 * (M_high/M_low) * (F_3(q_low)/F_3(q_high))^2 to
    1e-9 relative, with F_3 from the written-out polynomial (an
    independent arithmetic path).  The width law is evaluable for
    L = 3 under the default barrier (the card-8 blocker: rho and the
    decay kernel are no longer blocked on the F-wave and G-wave
    records)."""
    record = _record([_channel(L=3)])
    channel = record.decay_channels[0]

    q_low, q_high = 0.1, 1.0
    m_low = _hand_mass_for_q(q_low)
    m_high = _hand_mass_for_q(q_high)
    assert abs(decay_momentum(m_low, MP, MPI) - q_low) <= 1e-9
    assert abs(decay_momentum(m_high, MP, MPI) - q_high) <= 1e-9

    def _hand_f3(q: float) -> float:
        z = (q * BARRIER_R0) ** 2
        return math.sqrt(225.0 / (z**3 + 6.0 * z * z + 45.0 * z + 225.0))

    expected_ratio = (
        (q_low / q_high) ** 7
        * (m_high / m_low)
        * (_hand_f3(q_low) / _hand_f3(q_high)) ** 2
    )

    w_low = channel_width(m_low, channel, record, DAUGHTER_MASSES)
    w_high = channel_width(m_high, channel, record, DAUGHTER_MASSES)
    assert w_low > 0.0 and math.isfinite(w_low)
    assert w_high > 0.0 and math.isfinite(w_high)
    assert abs(w_low / w_high - expected_ratio) <= 1e-9 * expected_ratio, (
        w_low / w_high,
        expected_ratio,
    )

    # The same channel under the identity barrier: the ratio reduces
    # to the pure phase-space law (the barrier cancels), and the two
    # barrier choices differ by exactly the squared F_3 ratio.
    id_low = channel_width(
        m_low, channel, record, DAUGHTER_MASSES, IDENTITY_BARRIER
    )
    id_high = channel_width(
        m_high, channel, record, DAUGHTER_MASSES, IDENTITY_BARRIER
    )
    phase_ratio = (q_low / q_high) ** 7 * (m_high / m_low)
    assert abs(id_low / id_high - phase_ratio) <= 1e-9 * phase_ratio
    # The default-barrier width over the identity-barrier width equals
    # the squared written-out F_3 ratio to the pole momentum at each
    # mass (the phase-space factor is barrier-independent).
    q0 = decay_momentum(POLE, MP, MPI)
    f3_q0 = _hand_f3(q0)
    ratio_low = w_low / id_low
    ratio_high = w_high / id_high
    assert abs(ratio_low - (_hand_f3(q_low) / f3_q0) ** 2) <= 1e-9 * ratio_low
    assert (
        abs(ratio_high - (_hand_f3(q_high) / f3_q0) ** 2)
        <= 1e-9 * ratio_high
    )
    assert ratio_low != ratio_high
