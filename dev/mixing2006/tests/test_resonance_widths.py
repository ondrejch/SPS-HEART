"""Mass-dependent width and spectral-function tests (Round 2, card 3).

Task ``TASK-20260913-r2-mass-dependent-widths`` (plan section 22
task 11; plan section 8.2 step 2).  The deterministic card tests:

- ``test_zero_below_threshold``: ``channel_width`` is exactly ``0.0``
  for ``M`` at, at minus ``1e-12``, and below the channel threshold.
- ``test_continuity_at_threshold``: the right limit of
  ``channel_width`` at ``threshold + 1e-9`` is below ``1e-12`` GeV for
  L = 0, 1, 2 (the ``(q/q0)^(2L+1)`` factor vanishes).
- ``test_width_at_pole_equals_nominal``: at ``M == pole_mass`` with
  the identity barrier, ``total_width == nominal_width`` to ``1e-9``
  (the law normalizes at the pole by construction).
- ``test_l1_phase_space_scaling``: for a hand-constructed L = 1
  channel, ``channel_width(M2)/channel_width(M1)`` equals the
  hand-computed ``(q2/q1)^3 * (M1/M2)`` ratio to ``1e-9``.
- ``test_spectral_support``: ``spectral_function`` is ``0.0`` outside
  ``mass_range`` and positive inside.
- ``test_draw_in_window``: ``draw_mass`` with a window that carries no
  line-shape mass returns ``None``; a window inside the support
  returns a mass inside the window, deterministically for a fixed
  seed.

Statistical card test:

- ``test_spectral_moment_check`` (100,000 draws on a hand-constructed
  Delta(1232)-like record): the sample mean and RMS of the drawn
  masses agree with the hand-integrated moments of the truncated
  density to 3-sigma (the ``1/sqrt(N)`` convention).

Coverage pins added over the card's deterministic list (the
committed module behavior, nested 544dece):

- ``test_hard_zero_at_threshold_each_l``: the hard zero holds at, at
  minus ``1e-12``, and below the threshold for every L in (0, 1, 2);
  the right-limit continuity at the documented 1e-9 GeV width scale
  is pinned by ``test_continuity_at_threshold`` above.
- ``test_spectral_normalization_unit_integral``: the renormalized
  line shape integrates to 1.0 over ``[threshold, m_up]`` under the
  documented 10,001-node normalization quadrature (the pre-registered
  reference grid; the normalization is exactly that integral of the
  unnormalized shape, so the renormalized density reproduces it to
  float rounding; single-channel and two-channel channel-summed-width
  fixtures) and is exactly 0.0 outside the support.
- ``test_closed_at_pole_channel_carries_zero_width``: a channel
  closed at the pole mass (``q(M0) = 0``) carries zero width by
  convention at every M, and contributes nothing to the
  channel-summed total width (poles in the band where the committed
  module is correct).
- ``test_closed_at_pole_below_mass_difference_zero_width``: the same
  contract for a pole below ``|mb - mc|``, where the Kaellen
  function is positive in that forbidden band (the pre-fix
  ``decay_momentum`` leaked a spurious positive momentum there,
  REV-544dece-01).  Hard pin of the REV-544dece-01 fix (nested
  a84f8fb): ``decay_momentum`` is exactly ``0.0`` for the whole
  forbidden region at and below the threshold ``mb + mc``, and the
  closed-at-pole channel carries zero width at every M.
- ``test_multichannel_sum_at_nonpole_mass``: for a two-channel
  record, ``Gamma(M) = sum_c Gamma_c(M)`` at a non-pole mass, each
  channel evaluated with its own leading-pair momentum.
- ``test_three_body_provisional_leading_pair_convention``: three-body
  channels use the leading-pair two-body phase-space factor with the
  threshold the sum of all daughter masses (the provisional card-3
  convention, pending card 7).

Scope decision pinned here (the continuity fixture): the card's
absolute ``1e-12`` GeV bound at ``threshold + 1e-9`` is meaningful
only at a width scale matched to the vanishing phase-space factor.
The decay momentum rises like ``sqrt(M - threshold)`` above the
threshold, so at ``1e-9`` the ratio ``q/q0`` is of order ``1e-5`` and
the L = 0 width of a physical ~0.1 GeV resonance is of order ``1e-5``
GeV.  The continuity fixture therefore carries a hand-constructed
``1e-9`` GeV nominal width, the scale at which the card's bound
isolates the vanishing of the ``(q/q0)^(2L+1)`` factor; every other
test uses the Delta(1232)-scale fixture.  The threshold continuity
itself (right limit 0 for every L >= 0) is a property of the factor,
independent of the width scale.

Card-4 adaptation (``TASK-20260913-r2-barrier-factors``): the module
default barrier is the documented Blatt-Weisskopf factor from card 4
on, and these card-3 tests run with the identity barrier explicitly
(the card-4 constraint), so every pinned value below is the card-3
identity-barrier law the docstrings describe.  The Blatt-Weisskopf
default is pinned in ``tests/test_barrier_factors.py``.

Card-4 re-open (the card-8 check round): the standard Blatt-Weisskopf
polynomials support every ``L >= 0`` (the committed catalog carries
F-wave ``L = 3`` channels on N*(1680) and rho3(1690) and a G-wave
``L = 4`` channel on f4(2050)), so the card-3 width law and the
spectral function are evaluable under the default barrier for
``L > 2`` records; ``test_l_above_two_default_barrier_surface`` pins
that surface on hand-constructed F-wave and G-wave fixtures.

Card-8 re-open of card 3 (round-2 fix round, 2026-09-14): the
normalization constant of the spectral function is the trapezoid
integral of the unnormalized shape on the pre-registered 10,001-node
reference grid (``SPECTRAL_QUADRATURE_NODES`` of
``na49_mixing.resonance_decay``), not the original 1024-point grid.
The 1024-point convention was under-resolved for the narrow-width
records: on the pre-registered check grid the renormalized density
integrated to 0.3334 for the eta (hadronic width 1.31 keV; the grid
spacing is about 100 times the line width) and 1.0380 for the
eta-prime(958), with 11 further records marginally beyond the 1e-6
tolerance.  The unit-integral pins below run on the reference grid;
``test_narrow_width_grid_convergence`` pins the narrow-width behavior
on eta-like and eta-prime-like fixtures, and
``test_normalization_grid_convention_pins`` pins the grid identity and
the unchanged 1024-point draw convention.

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

from na49_mixing.hadron_catalog import (  # noqa: E402
    DecayChannel,
    HadronRecord,
)
from na49_mixing.resonance_decay import (  # noqa: E402
    SPECTRAL_QUADRATURE_NODES,
)
from na49_mixing.resonance_widths import (  # noqa: E402
    IDENTITY_BARRIER,
    _DRAW_GRID_POINTS,
    _NORMALIZATION_GRID_POINTS,
    channel_width,
    decay_momentum,
    draw_mass,
    kallen,
    spectral_function,
    total_width,
)

# ----------------------------------------------------------------------
# Hand-constructed Delta(1232)-like fixtures (card-2 catalog values;
# the vendored-table daughter masses the catalog thresholds are built
# from).  Provenance is None by design: hand-constructed fixtures sit
# before the catalog import, where the card-1 schema permits it.
# ----------------------------------------------------------------------
MP = 0.93827231            # vendored table proton mass, GeV/c^2
MPI = 0.1395699            # vendored table pi+ mass, GeV/c^2
MN = 0.93956541            # vendored table neutron mass, GeV/c^2
POLE = 1.232               # Delta(1232) pole mass, GeV/c^2
WIDTH = 0.117              # Delta(1232) nominal width, GeV

DAUGHTER_MASSES = {2212: MP, 211: MPI, 2112: MN}

THRESHOLD_P_PION = math.fsum((MP, MPI))    # 1.07784221 (Delta++ -> p pi+)
THRESHOLD_N_PION = math.fsum((MN, MPI))    # 1.07913531 (Delta0 -> n pi+)
M_UP = 2.0                                 # the documented dumphigh cut

# Narrow-width fixtures for the card-8 re-open acceptance (the catalog
# eta and eta-prime geometry; vendored-table pion masses, PDG widths).
MPI0 = 0.1349766                           # vendored table pi0 mass
ETA_POLE = 0.547862                        # eta(548) pole mass
ETA_WIDTH = 1.31e-06                       # eta(548) nominal width, GeV
ETAPRIME_POLE = 0.95778                    # eta-prime(958) pole mass
ETAPRIME_WIDTH = 1.88e-04                  # eta-prime(958) nominal width


def _channel(
    daughters: tuple[int, ...] = (2212, 211),
    branching_fraction: float = 1.0,
    L: int = 1,
) -> DecayChannel:
    """One hand-constructed two-body Delta channel."""
    return DecayChannel(
        parent_pdg=2224,
        daughters=tuple(daughters),
        branching_fraction=branching_fraction,
        orbital_angular_momentum=L,
        is_three_body=False,
        provenance=None,
    )


def _record(
    channels,
    nominal_width: float = WIDTH,
    pdg_id: int = 2224,
    name: str = "Delta++(fixture)",
) -> HadronRecord:
    """A hand-constructed Delta(1232)-like catalog record.

    ``mass_range[0]`` is the highest channel threshold (the card-1
    schema rule, hand-computed here); ``mass_range[1]`` is the
    documented Delta(1232) dumphigh cut 2.0.
    """
    lo = max(
        math.fsum(float(DAUGHTER_MASSES[int(d)]) for d in ch.daughters)
        for ch in channels
    )
    return HadronRecord(
        pdg_id=pdg_id,
        name=name,
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


def _hand_q(M: float) -> float:
    """Decay momentum from the factored Kaellen form (an independent
    computation path from the module's ``kallen``)."""
    lam = (M * M - (MP + MPI) ** 2) * (M * M - (MP - MPI) ** 2)
    return math.sqrt(lam) / (2.0 * M)


def _trapezoid(xs: np.ndarray, ys: np.ndarray) -> float:
    """Trapezoid integral (the module's documented convention)."""
    return float((0.5 * (ys[1:] + ys[:-1]) * (xs[1:] - xs[:-1])).sum())


# ----------------------------------------------------------------------
# Deterministic card tests
# ----------------------------------------------------------------------
def test_zero_below_threshold():
    """channel_width is exactly 0.0 at, at minus 1e-12, and below the
    channel threshold (hard zero, plan section 8.2)."""
    record = _record([_channel()])
    channel = record.decay_channels[0]
    threshold = THRESHOLD_P_PION
    for M in (threshold, threshold - 1e-12, threshold - 1e-6):
        assert channel_width(
            M, channel, record, DAUGHTER_MASSES, IDENTITY_BARRIER
        ) == 0.0
    # The channel-summed width inherits the hard zero below every
    # channel threshold.
    assert total_width(
        threshold - 1e-12, record, DAUGHTER_MASSES, IDENTITY_BARRIER
    ) == 0.0


def test_continuity_at_threshold():
    """The right limit of channel_width at threshold + 1e-9 is below
    1e-12 GeV for L = 0, 1, 2: the (q/q0)^(2L+1) factor vanishes at
    the threshold (q(threshold) = 0).  The fixture carries the 1e-9
    GeV width scale documented in the module docstring; the bound is
    the card's absolute GeV bound."""
    tiny_width = 1e-9
    for L in (0, 1, 2):
        record = _record([_channel(L=L)], nominal_width=tiny_width)
        channel = record.decay_channels[0]
        value = channel_width(
            THRESHOLD_P_PION + 1e-9,
            channel,
            record,
            DAUGHTER_MASSES,
            IDENTITY_BARRIER,
        )
        assert 0.0 < value < 1e-12, (L, value)


def test_width_at_pole_equals_nominal():
    """At M == pole_mass with the identity barrier, total_width equals
    nominal_width to 1e-9 (the law normalizes at the pole by
    construction): the single-channel Delta++ fixture with BR = 1 and
    the two-channel Delta+-like fixture with BR 0.6667 + 0.3333 = 1."""
    single = _record([_channel()])
    assert abs(
        total_width(POLE, single, DAUGHTER_MASSES, IDENTITY_BARRIER) - WIDTH
    ) <= 1e-9

    two_channel = _record(
        [
            _channel(branching_fraction=0.6667),
            _channel(daughters=(2112, 211), branching_fraction=0.3333),
        ],
        pdg_id=2214,
        name="Delta+(fixture)",
    )
    assert abs(
        total_width(
            POLE, two_channel, DAUGHTER_MASSES, IDENTITY_BARRIER
        ) - WIDTH
    ) <= 1e-9

    # The identity barrier pinned directly: F_L(q) == 1 for every L
    # and q, so the ratio term of the width law is exactly 1.
    assert IDENTITY_BARRIER(0, 0.05) == 1.0
    assert IDENTITY_BARRIER(1, 0.227) == 1.0
    assert IDENTITY_BARRIER(2, 0.9) == 1.0


def test_l1_phase_space_scaling():
    """For a hand-constructed L = 1 channel, the width ratio at two
    above-threshold masses equals the hand-computed
    (q2/q1)^3 * (M1/M2) to 1e-9 (phase-space factor times the M0/M
    Jacobian; the identity barrier cancels)."""
    record = _record([_channel(L=1)])
    channel = record.decay_channels[0]
    M1, M2 = 1.15, 1.6
    q1, q2 = _hand_q(M1), _hand_q(M2)
    expected = (q2 / q1) ** 3 * (M1 / M2)
    actual = (
        channel_width(M2, channel, record, DAUGHTER_MASSES, IDENTITY_BARRIER)
        / channel_width(M1, channel, record, DAUGHTER_MASSES, IDENTITY_BARRIER)
    )
    assert abs(actual - expected) <= 1e-9
    # The module's decay momentum agrees with the factored-form hand
    # computation (same quantity, independent arithmetic paths; the
    # two Kaellen forms differ only at the last-ULP rounding level).
    assert abs(decay_momentum(M1, MP, MPI) - _hand_q(M1)) <= 1e-15
    assert abs(decay_momentum(POLE, MP, MPI) - _hand_q(POLE)) <= 1e-15
    # kallen cross-check against the factored form at the same masses.
    lam_hand = (M1 * M1 - (MP + MPI) ** 2) * (M1 * M1 - (MP - MPI) ** 2)
    assert abs(kallen(M1 * M1, MP * MP, MPI * MPI) - lam_hand) <= 1e-12


def test_spectral_support():
    """spectral_function is 0.0 outside mass_range and positive
    inside; the density vanishes at the lower support endpoint, where
    every channel width is the hard zero at the threshold."""
    record = _record([_channel()])
    lo, hi = record.mass_range
    for M in (lo - 1e-9, 0.5, hi + 1e-9, hi + 0.3):
        assert spectral_function(
            M, record, DAUGHTER_MASSES, IDENTITY_BARRIER
        ) == 0.0
    assert spectral_function(
        lo, record, DAUGHTER_MASSES, IDENTITY_BARRIER
    ) == 0.0
    for M in (lo + 1e-6, 1.1, POLE, 1.5, 1.9, hi):
        assert spectral_function(
            M, record, DAUGHTER_MASSES, IDENTITY_BARRIER
        ) > 0.0


def test_draw_in_window():
    """draw_mass returns None for a window that carries no line-shape
    mass; a window inside the support returns a mass inside the
    window, deterministically for a fixed seed; exactly two uniform
    draws are consumed per non-None draw and none for None."""
    record = _record([_channel()])
    empty_below = (1.0, THRESHOLD_P_PION - 1e-3)
    empty_above = (M_UP + 0.05, M_UP + 0.4)
    inverted = (1.4, 1.1)
    for window in (empty_below, empty_above, inverted):
        assert draw_mass(
            window, record, DAUGHTER_MASSES,
            np.random.default_rng(1), IDENTITY_BARRIER,
        ) is None

    window = (1.1, 1.4)
    first = draw_mass(
        window, record, DAUGHTER_MASSES,
        np.random.default_rng(20260913), IDENTITY_BARRIER,
    )
    again = draw_mass(
        window, record, DAUGHTER_MASSES,
        np.random.default_rng(20260913), IDENTITY_BARRIER,
    )
    assert first is not None and again is not None
    assert window[0] <= first <= window[1]
    assert first == again

    # Two uniform draws consumed per non-None draw (the
    # draw_truncated_bw convention): the generator state after one
    # draw equals the state after two manual random() calls.
    spent = np.random.default_rng(11)
    draw_mass(window, record, DAUGHTER_MASSES, spent, IDENTITY_BARRIER)
    two = np.random.default_rng(11)
    two.random()
    two.random()
    assert spent.bit_generator.state == two.bit_generator.state

    # A None window consumes no draws.
    unspent = np.random.default_rng(11)
    draw_mass(empty_below, record, DAUGHTER_MASSES, unspent, IDENTITY_BARRIER)
    zero = np.random.default_rng(11)
    assert unspent.bit_generator.state == zero.bit_generator.state


# ----------------------------------------------------------------------
# Statistical card test
# ----------------------------------------------------------------------
def test_spectral_moment_check():
    """100,000 draws on a hand-constructed Delta(1232)-like record:
    the sample mean and RMS of the drawn masses agree with the
    hand-integrated moments of the truncated density to 3-sigma (the
    1/sqrt(N) convention).  Seed 20260913 (deterministic)."""
    record = _record([_channel(L=1)])
    lo, hi = record.mass_range
    n_draws = 100_000
    rng = np.random.default_rng(20260913)
    draws = np.empty(n_draws)
    for i in range(n_draws):
        draws[i] = draw_mass(
            (lo, hi), record, DAUGHTER_MASSES, rng, IDENTITY_BARRIER
        )
    assert (draws >= lo).all() and (draws <= hi).all()

    sample_mean = float(draws.mean())
    sample_rms = math.sqrt(float(np.mean(draws ** 2)))

    # Hand-integrated moments of the truncated density on the same
    # 1024-point grid (the density is already normalized; the
    # renormalization below removes the float-level residue).
    xs = np.linspace(lo, hi, 1024)
    ys = np.array(
        [
            spectral_function(
                float(x), record, DAUGHTER_MASSES, IDENTITY_BARRIER
            )
            for x in xs
        ]
    )
    norm = _trapezoid(xs, ys)
    mean_th = _trapezoid(xs, xs * ys) / norm
    rms_th = math.sqrt(_trapezoid(xs, xs * xs * ys) / norm)
    sigma_th = math.sqrt(max(rms_th * rms_th - mean_th * mean_th, 0.0))

    tolerance = 3.0 * sigma_th / math.sqrt(n_draws)
    assert abs(sample_mean - mean_th) <= tolerance, (
        sample_mean, mean_th, tolerance)
    assert abs(sample_rms - rms_th) <= tolerance, (
        sample_rms, rms_th, tolerance)


# ----------------------------------------------------------------------
# Coverage pins over the card's deterministic list (the committed
# module behavior, nested 544dece)
# ----------------------------------------------------------------------
def _hand_q_pair(M: float, mb: float, mc: float) -> float:
    """Decay momentum of the pair (mb, mc) from the factored Kaellen
    form (an independent computation path from the module's
    ``kallen``), for arbitrary daughter masses."""
    lam = (M * M - (mb + mc) ** 2) * (M * M - (mb - mc) ** 2)
    return math.sqrt(lam) / (2.0 * M)


def test_hard_zero_at_threshold_each_l():
    """The hard zero of channel_width holds for every L in (0, 1, 2)
    at, at minus 1e-12, and below the channel threshold (the
    (q/q0)^(2L+1) factor defines the width only above the threshold),
    and the width is strictly positive just above it.  The right-limit
    continuity at the documented 1e-9 GeV width scale is pinned by
    test_continuity_at_threshold."""
    for L in (0, 1, 2):
        record = _record([_channel(L=L)])
        channel = record.decay_channels[0]
        for M in (
            THRESHOLD_P_PION,
            THRESHOLD_P_PION - 1e-12,
            THRESHOLD_P_PION - 1e-6,
        ):
            assert channel_width(
                M, channel, record, DAUGHTER_MASSES, IDENTITY_BARRIER
            ) == 0.0
        assert (
            channel_width(
                THRESHOLD_P_PION + 1e-6,
                channel,
                record,
                DAUGHTER_MASSES,
                IDENTITY_BARRIER,
            )
            > 0.0
        )


def test_spectral_normalization_unit_integral():
    """The renormalized line shape integrates to 1.0 over
    [threshold, m_up] under the documented 10,001-node normalization
    quadrature (the pre-registered reference grid; the normalization
    is exactly that integral of the unnormalized shape, so the
    renormalized density reproduces it to float rounding), for the
    single-channel and the two-channel (channel-summed width)
    fixtures; it is exactly 0.0 outside the support."""
    records = (
        _record([_channel()]),
        _record(
            [
                _channel(branching_fraction=0.6667),
                _channel(daughters=(2112, 211), branching_fraction=0.3333),
            ],
            pdg_id=2214,
            name="Delta+(fixture)",
        ),
    )
    for record in records:
        lo, hi = record.mass_range
        xs = np.linspace(lo, hi, _NORMALIZATION_GRID_POINTS)
        ys = np.array(
            [
                spectral_function(
                    float(x), record, DAUGHTER_MASSES, IDENTITY_BARRIER
                )
                for x in xs
            ]
        )
        assert abs(_trapezoid(xs, ys) - 1.0) <= 1e-9
        # Exactly 0.0 outside the support, at both edges and away.
        for M in (lo - 1e-9, 0.5, hi + 1e-9, hi + 0.3):
            assert spectral_function(
                M, record, DAUGHTER_MASSES, IDENTITY_BARRIER
            ) == 0.0


def test_closed_at_pole_channel_carries_zero_width():
    """A channel closed at the pole mass (q(M0) = 0) carries zero
    width by convention at every M, and contributes nothing to the
    channel-summed total width: the law is normalized at a pole the
    closed channel never reaches.  The closed-at-pole poles here sit
    in (|mb - mc|, mb + mc), where the module's decay momentum is
    correctly 0; the below-|mb - mc| side of the same convention is
    pinned separately by
    test_closed_at_pole_below_mass_difference_zero_width (hard pin of
    the REV-544dece-01 fix)."""
    # Single closed-at-pole channel: the pole sits below the p pi+
    # threshold, so q(M0) = 0; the channel still has M > threshold
    # points where an open-at-pole channel would carry width.
    record_a = HadronRecord(
        pdg_id=9991,
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
        decay_channels=(_channel(),),
        stable_for_decay=False,
        provenance=None,
    )
    channel_a = record_a.decay_channels[0]
    assert decay_momentum(1.05, MP, MPI) == 0.0
    for M in (THRESHOLD_P_PION, THRESHOLD_P_PION + 1e-6, 1.2, 1.5):
        assert channel_width(
            M, channel_a, record_a, DAUGHTER_MASSES, IDENTITY_BARRIER
        ) == 0.0
    assert total_width(1.2, record_a, DAUGHTER_MASSES, IDENTITY_BARRIER) == 0.0

    # Two channels, one open at the pole and one closed at the pole:
    # at M == M0 the open channel carries BR * nominal_width exactly
    # (the law normalizes at the pole) and the closed channel
    # contributes zero to the sum.  The pole 0.85 sits in
    # (|MP - MPI|, MP + MPI) = (0.79870241, 1.07784221), where the
    # module's closed-at-pole handling is correct.
    open_ch = DecayChannel(
        parent_pdg=9992,
        daughters=(211, 211),
        branching_fraction=0.5,
        orbital_angular_momentum=1,
        is_three_body=False,
        provenance=None,
    )
    closed_ch = DecayChannel(
        parent_pdg=9992,
        daughters=(2212, 211),
        branching_fraction=0.5,
        orbital_angular_momentum=1,
        is_three_body=False,
        provenance=None,
    )
    record_b = HadronRecord(
        pdg_id=9992,
        name="mixed-pole(fixture)",
        B=0,
        Q=1,
        S=0,
        isospin=1.0,
        isospin3=1,
        spin=1.5,
        parity="+",
        pole_mass=0.85,
        nominal_width=WIDTH,
        mass_range=(THRESHOLD_P_PION, M_UP),
        decay_channels=(open_ch, closed_ch),
        stable_for_decay=False,
        provenance=None,
    )
    assert abs(
        channel_width(
            0.85, open_ch, record_b, DAUGHTER_MASSES, IDENTITY_BARRIER
        ) - 0.5 * WIDTH
    ) <= 1e-9
    assert channel_width(
        0.85, closed_ch, record_b, DAUGHTER_MASSES, IDENTITY_BARRIER
    ) == 0.0
    assert abs(
        total_width(
            0.85, record_b, DAUGHTER_MASSES, IDENTITY_BARRIER
        ) - 0.5 * WIDTH
    ) <= 1e-9
    # The closed-at-pole convention holds at every M, including well
    # above the channel's own threshold.
    assert channel_width(
        1.2, closed_ch, record_b, DAUGHTER_MASSES, IDENTITY_BARRIER
    ) == 0.0


def test_closed_at_pole_below_mass_difference_zero_width():
    """The closed-at-pole zero-width contract for a pole mass below
    |mb - mc|: the 0.78 GeV pole sits below |MP - MPI| = 0.79870241,
    so the p pi+ channel is closed at the pole and the physical decay
    momentum there is 0.  ``decay_momentum`` is documented to be
    exactly ``0.0`` for the whole forbidden region at and below the
    threshold ``mb + mc`` (the Kaellen function is positive also for
    0 < M < |mb - mc|, so the threshold guard, not the lambda <= 0
    guard, is the physical condition), and the ``channel_width``
    docstring documents that a channel closed at the pole carries
    zero width by convention at every M.  Hard pin of the
    REV-544dece-01 fix (nested a84f8fb)."""
    pole = 0.78
    record = HadronRecord(
        pdg_id=9994,
        name="closed-pole-low(fixture)",
        B=1,
        Q=2,
        S=0,
        isospin=1.5,
        isospin3=3,
        spin=1.5,
        parity="+",
        pole_mass=pole,
        nominal_width=WIDTH,
        mass_range=(THRESHOLD_P_PION, M_UP),
        decay_channels=(_channel(),),
        stable_for_decay=False,
        provenance=None,
    )
    channel = record.decay_channels[0]
    # The forbidden band 0 < M < |mb - mc|: exactly 0.0 at the 0.78
    # GeV pole, at the in-band point M = 0.5 flagged by the pre-fix
    # sign chart, and exactly at the mass-difference root |mb - mc|.
    assert decay_momentum(pole, MP, MPI) == 0.0
    assert decay_momentum(0.5, MP, MPI) == 0.0
    assert decay_momentum(abs(MP - MPI), MP, MPI) == 0.0
    # The threshold boundary of the physical two-body region (the p
    # pi+ channel is open at its own pole): exactly 0.0 at and
    # below mb + mc, positive just above it.
    threshold = THRESHOLD_P_PION
    assert decay_momentum(threshold, MP, MPI) == 0.0
    assert decay_momentum(threshold + 1e-9, MP, MPI) > 0.0
    # Documented convention: the closed-at-pole channel carries zero
    # width at every M, including M well above the threshold (no
    # spurious width leak from the forbidden-band q0).
    for M in (threshold, threshold + 1e-6, 1.2, 1.5, M_UP):
        assert channel_width(
            M, channel, record, DAUGHTER_MASSES, IDENTITY_BARRIER
        ) == 0.0
    assert total_width(1.2, record, DAUGHTER_MASSES, IDENTITY_BARRIER) == 0.0


def test_multichannel_sum_at_nonpole_mass():
    """Gamma(M) = sum_c Gamma_c(M) at a non-pole mass for a record
    with two open channels: each part matches the plan section 8.2
    law hand-computed with that channel's own leading-pair momentum
    (the p pi+ and n pi+ pair momenta differ), and the total is the
    exact channel sum."""
    record = _record(
        [
            _channel(branching_fraction=0.6667),
            _channel(daughters=(2112, 211), branching_fraction=0.3333),
        ],
        pdg_id=2214,
        name="Delta+(fixture)",
    )
    ch1, ch2 = record.decay_channels
    M = 1.3
    w1 = channel_width(M, ch1, record, DAUGHTER_MASSES, IDENTITY_BARRIER)
    w2 = channel_width(M, ch2, record, DAUGHTER_MASSES, IDENTITY_BARRIER)
    assert w1 > 0.0 and w2 > 0.0

    # Hand-computed law per channel (identity barrier cancels), each
    # with its own pair momentum.
    exp1 = 0.6667 * WIDTH * (
        _hand_q_pair(M, MP, MPI) / _hand_q_pair(POLE, MP, MPI)
    ) ** 3 * (POLE / M)
    exp2 = 0.3333 * WIDTH * (
        _hand_q_pair(M, MN, MPI) / _hand_q_pair(POLE, MN, MPI)
    ) ** 3 * (POLE / M)
    assert abs(w1 - exp1) <= 1e-9 * exp1
    assert abs(w2 - exp2) <= 1e-9 * exp2

    total = total_width(M, record, DAUGHTER_MASSES, IDENTITY_BARRIER)
    assert total == math.fsum([w1, w2])
    assert abs(total - math.fsum([exp1, exp2])) <= 1e-9 * (exp1 + exp2)


def test_three_body_provisional_leading_pair_convention():
    """Provisional card-3 convention for three-body channels, pinned
    as-is and flagged for card 7 (card 7 retargets this test; its
    refinement must not change two-body widths): the two-body
    phase-space factor is applied to the leading pair
    (daughters[0], daughters[1]) while the threshold remains the sum
    of all daughter masses, matching HadronCatalog.  The third
    daughter enters the width law only through that threshold."""
    threshold3 = math.fsum((MPI, MPI, MPI))   # 0.4187097 (3 pi)
    pole3 = 0.770
    width3 = 0.150
    channel = DecayChannel(
        parent_pdg=9993,
        daughters=(211, 211, 211),
        branching_fraction=1.0,
        orbital_angular_momentum=1,
        is_three_body=True,
        provenance=None,
    )

    def _three_body(pole: float) -> HadronRecord:
        return HadronRecord(
            pdg_id=9993,
            name="3pi(fixture)",
            B=0,
            Q=0,
            S=0,
            isospin=1.0,
            isospin3=0,
            spin=1.0,
            parity="-",
            pole_mass=pole,
            nominal_width=width3,
            mass_range=(threshold3, M_UP),
            decay_channels=(channel,),
            stable_for_decay=False,
            provenance=None,
        )

    record = _three_body(pole3)
    masses = {211: MPI}

    # The threshold is the sum of all daughters: hard zero at and
    # below it.
    for M in (threshold3, threshold3 - 1e-12, threshold3 - 1e-6):
        assert channel_width(M, channel, record, masses, IDENTITY_BARRIER) == 0.0

    # Above the threshold the width is the leading-pair law at the
    # two hand-computed points away from the pole (BR = 1, L = 1,
    # identity barrier): Gamma_c(M) = W * (q_pair(M)/q_pair(M0))**3
    # * (M0/M), with the pair momentum of (MPI, MPI) only.
    q0 = _hand_q_pair(pole3, MPI, MPI)
    M1, M2 = 0.60, 0.90
    q1 = _hand_q_pair(M1, MPI, MPI)
    q2 = _hand_q_pair(M2, MPI, MPI)
    exp1 = width3 * (q1 / q0) ** 3 * (pole3 / M1)
    exp2 = width3 * (q2 / q0) ** 3 * (pole3 / M2)
    w1 = channel_width(M1, channel, record, masses, IDENTITY_BARRIER)
    w2 = channel_width(M2, channel, record, masses, IDENTITY_BARRIER)
    assert abs(w1 - exp1) <= 1e-9 * exp1
    assert abs(w2 - exp2) <= 1e-9 * exp2
    # The phase-space ratio at the two points: (q2/q1)**3 * (M1/M2).
    assert abs(w2 / w1 - (q2 / q1) ** 3 * (M1 / M2)) <= 1e-9
    # The module's pair momentum agrees with the hand path.
    assert abs(decay_momentum(M1, MPI, MPI) - q1) <= 1e-15

    # Closed at the pole through the leading pair: a pole below the
    # pair threshold 2*MPI gives q_pair(M0) = 0, hence zero width at
    # every M, even well above the three-body threshold.
    record_closed = _three_body(0.20)
    assert decay_momentum(0.20, MPI, MPI) == 0.0
    assert channel_width(
        0.60, channel, record_closed, masses, IDENTITY_BARRIER
    ) == 0.0

    # The spectral function inherits the convention: zero outside
    # [threshold3, m_up], positive at the pole.
    assert spectral_function(
        threshold3 - 1e-9, record, masses, IDENTITY_BARRIER
    ) == 0.0
    assert spectral_function(
        M_UP + 1e-9, record, masses, IDENTITY_BARRIER
    ) == 0.0
    assert spectral_function(
        pole3, record, masses, IDENTITY_BARRIER
    ) > 0.0


# ----------------------------------------------------------------------
# Card-4 re-open (the card-8 check round): the card-3 surface under
# the default (Blatt-Weisskopf) barrier for L > 2 records
# ----------------------------------------------------------------------
def test_l_above_two_default_barrier_surface():
    """The card-3 width law and spectral function under the default
    (Blatt-Weisskopf) barrier for F-wave (L = 3) and G-wave (L = 4)
    hand-constructed channels -- the card-8 blocker resolved: the
    width is evaluable (finite, strictly positive above the channel
    threshold, the hard zero at and below it), the pole normalization
    Gamma(M0) = nominal_width holds (the barrier ratio is 1 at the
    pole), and rho is evaluable: 0.0 outside the support, positive
    inside, and the unit integral over the support under the
    documented 10,001-node normalization quadrature."""
    for L in (3, 4):
        record = _record([_channel(L=L)])
        channel = record.decay_channels[0]
        lo, hi = record.mass_range

        # Width law: hard zero at/below the threshold, finite and
        # strictly positive above it, pole normalization exact.
        for M in (lo, lo - 1e-12, lo - 1e-6):
            assert channel_width(M, channel, record, DAUGHTER_MASSES) == 0.0
        assert total_width(lo - 1e-12, record, DAUGHTER_MASSES) == 0.0
        assert (
            channel_width(lo + 1e-6, channel, record, DAUGHTER_MASSES) > 0.0
        )
        assert abs(
            total_width(POLE, record, DAUGHTER_MASSES) - WIDTH
        ) <= 1e-9
        for M in (lo + 1e-6, 1.15, POLE, 1.5, 1.9):
            width = channel_width(M, channel, record, DAUGHTER_MASSES)
            assert math.isfinite(width) and width > 0.0

        # Spectral function: 0.0 outside the support, positive inside,
        # unit integral under the documented convention.
        for M in (lo - 1e-9, 0.5, hi + 1e-9, hi + 0.3):
            assert spectral_function(M, record, DAUGHTER_MASSES) == 0.0
        for M in (lo + 1e-6, POLE, 1.5, 1.9, hi):
            assert spectral_function(M, record, DAUGHTER_MASSES) > 0.0
        xs = np.linspace(lo, hi, _NORMALIZATION_GRID_POINTS)
        ys = np.array(
            [
                spectral_function(float(x), record, DAUGHTER_MASSES)
                for x in xs
            ]
        )
        assert abs(_trapezoid(xs, ys) - 1.0) <= 1e-9


# ----------------------------------------------------------------------
# Card-8 re-open of card 3 (round-2 fix round): the narrow-width
# normalization on the pre-registered reference quadrature
# ----------------------------------------------------------------------
def _eta_like(pole: float, width: float, pdg_id: int, name: str):
    """A hand-constructed narrow three-body record with the committed
    catalog geometry: one (pi+, pi-, pi0) channel at L = 0, the
    threshold the daughter sum, ``m_up = pole + 5 * width`` (the
    card-2 catalog upper-cut rule).  Returns ``(record, masses)``."""
    channel = DecayChannel(
        parent_pdg=int(pdg_id),
        daughters=(211, -211, 111),
        branching_fraction=1.0,
        orbital_angular_momentum=0,
        is_three_body=True,
        provenance=None,
    )
    masses = {211: MPI, -211: MPI, 111: MPI0}
    threshold = math.fsum(masses[d] for d in (211, -211, 111))
    record = HadronRecord(
        pdg_id=int(pdg_id),
        name=name,
        B=0,
        Q=0,
        S=0,
        isospin=0.0,
        isospin3=0,
        spin=0.0,
        parity="-",
        pole_mass=float(pole),
        nominal_width=float(width),
        mass_range=(threshold, float(pole) + 5.0 * float(width)),
        decay_channels=(channel,),
        stable_for_decay=False,
        provenance=None,
    )
    return record, masses


def test_narrow_width_grid_convergence():
    """The card-8 re-open acceptance at the card-3 surface: for the
    hand-constructed eta-like (1.31 keV) and eta-prime-like (188 keV)
    narrow-width fixtures, the 10,001-node trapezoid integral of rho
    over the support is 1 within 1e-9 (the pre-registered reference
    quadrature), rho is strictly positive at the pole, and rho is
    exactly 0.0 outside the support.  Under the original 1024-point
    normalization the same fixtures read 0.3334 and 1.0380 on the
    reference grid (the committed-catalog violations this fix round
    resolves).  The hand-norm cross-check pins the constant's
    definition: the density equals the hand shape divided by its own
    10,001-node hand norm (a 1024-point constant shifts the ratio by
    3.0 for the eta fixture)."""
    fixtures = (
        _eta_like(ETA_POLE, ETA_WIDTH, 221, "eta(fixture)"),
        _eta_like(ETAPRIME_POLE, ETAPRIME_WIDTH, 377, "eta-prime(fixture)"),
    )
    for record, masses in fixtures:
        lo, hi = record.mass_range
        xs = np.linspace(lo, hi, _NORMALIZATION_GRID_POINTS)
        ys = np.array(
            [spectral_function(float(x), record, masses) for x in xs]
        )
        integral = _trapezoid(xs, ys)
        assert abs(integral - 1.0) <= 1e-9, (record.name, integral)
        assert spectral_function(
            float(record.pole_mass), record, masses
        ) > 0.0
        for M in (lo - 1e-12, hi + 1e-12):
            assert spectral_function(M, record, masses) == 0.0

    # The normalization constant is the 10,001-node trapezoid of the
    # unnormalized shape (hand arithmetic path, public width surface).
    record, masses = fixtures[0]
    lo, hi = record.mass_range
    xs = np.linspace(lo, hi, _NORMALIZATION_GRID_POINTS)

    def _hand_shape(M: float) -> float:
        gamma = total_width(M, record, masses)
        M0 = float(record.pole_mass)
        return (
            M * M * gamma
            / ((M * M - M0 * M0) ** 2 + M0 * M0 * gamma * gamma)
        )

    hand_norm = _trapezoid(xs, np.array(
        [_hand_shape(float(x)) for x in xs]
    ))
    assert hand_norm > 0.0
    M0 = float(record.pole_mass)
    for M in (M0, M0 - 2.0 * ETA_WIDTH, M0 + 3.0 * ETA_WIDTH):
        expected = _hand_shape(M) / hand_norm
        actual = spectral_function(M, record, masses)
        assert abs(actual - expected) <= 1e-12 * abs(expected), (M, actual)


def test_normalization_grid_convention_pins():
    """The normalization quadrature is the pre-registered 10,001-node
    reference grid of the round-2 check suite (the card-8
    ``SPECTRAL_QUADRATURE_NODES``; the module constant is private and
    pinned here), and the draw-table grid stays the documented
    1024-point ``draw_truncated_bw`` convention (the draw table
    re-normalizes over the window internally, so the fix round leaves
    the draw convention unchanged)."""
    assert _NORMALIZATION_GRID_POINTS == 10_001
    assert _NORMALIZATION_GRID_POINTS == SPECTRAL_QUADRATURE_NODES
    assert _DRAW_GRID_POINTS == 1024
