"""The projectile and target single-diffraction classes (Round 5, card 5).

``TASK-20260925-r5-single-diffraction``; plan section 22 task 29 (line
1754); plan section 11.1 items 2-3 (lines 825-826).  The card implements
the two single-diffraction event classes of the initial topology basis:

> 2. **Projectile single diffraction:** low-mass or intact target line
>    plus excited projectile-side baryonic system, or the beam-reversed
>    counterpart according to convention.
>
> 3. **Target single diffraction:** mirrored class.

The pre-registered instantiation (the card): one line intact (the
:class:`SDSide` selection names which side is excited), the other line a
single baryonic excitation with mass ``M`` from card 4's marginal law on
the exact limits ``[m_p^2, (sqrt(s) - m_remnant)^2]]``
(:func:`na49_mixing.topology_mass_laws.sd_limits`) and momentum transfer
``t`` from ``exp(B_SD t)`` on the exact two-body range
``[t_min(M, m_remnant), 0]``
(:func:`na49_mixing.topology_mass_laws.t_range`, closed at ``t = 0``).
The non-excited line carries a remnant of card 3 with the nominal R50
split (intact proton 0.5, low-mass baryonic resonance 0.5, high-mass 0.0;
plan line 825 names exactly the intact and low-mass options).  Central
production is absent: the class has exactly one exchange and two outgoing
lines.

Exchange and ledger conventions (plan section 11.4)
---------------------------------------------------

A single vacuum exchange carries the declared ``(B, Q, S, 2I3)`` transfer
``(0, 0, 0, 0)`` (plan line 877); both outgoing lines inherit their
incoming line quantum numbers ``(1, 1, 0, 1)`` after the zero transfer
(:func:`na49_mixing.exchange_ledger.apply_transfer_qn`, the pinned card-2
arithmetic).  The event record carries the two incoming proton lines, the
two outgoing line states (the remnant line and the excitation line, in
the fixed projectile-side-first order), and the single exchange record;
card 2's :func:`na49_mixing.exchange_ledger.audit_event` consumes these
records read-only with zero violations (the card's conservation
acceptance).

Exact kinematics
----------------

:func:`sd_kinematics` solves the outgoing four-momenta from ``(M, t,
m_remnant, sqrt_s)`` as the exact two-body scattering ``p + p -> X(M) +
R(m_remnant)`` in the center-of-mass frame of the pinned beam
configuration (beam :data:`BEAM_MOMENTUM = 158.0 GeV/c`, proton mass
:data:`PROTON_MASS = 0.93827231 GeV), boosted into the lab frame with the
exact fixed-target CM velocity.  The scattering angle follows from the
invariant transfer on the excitation line,
``t = m_p^2 + M^2 - 2 (E_p* E_M* - p* q* cos(theta*))``, and the
production plane is fixed to the x-z plane (the azimuthal orientation
carries no draw).  Energy-momentum closure holds by construction within
``1e-9`` relative.

One floating-point fact bounds the solvable domain and is recorded here
(the card's closed-at-0 convention, operative resolution): the exact
two-body forward edge ``t_fwd(M, m_remnant)`` lies below ``0`` for every
excited mass ``M > m_p`` (the excitation is heavier than the beam proton,
so even the forward configuration carries a negative ``t``), while the
pre-registered draw range is closed at ``t = 0`` exactly.  For a drawn
``t`` above the exact forward edge (a sliver of width up to about
23.6 GeV^2 at the heaviest excitation masses of the drawn domain, the
1.72 GeV low-mass remnant at the ``M^2`` window's upper edge) the solved
angle saturates at the forward configuration ``cos(theta*) = 1``; the
event's recorded ``t`` stays the drawn value (the exchange-kinematics
variable), and the solved four-momenta are the physical forward
configuration.  At the intact limit ``M = m_p`` (with the intact-proton
remnant) the forward edge is exactly ``0``, so the ``t = 0`` limit
returns the remnant four-vector equal to its beam four-vector (within
``1e-12``; the :func:`sd_kinematics` docstring contract).

Frame convention (operative resolution, documented): the excitation
decays through the frozen simplified cascade with the Round 4 modules
consumed read-only, and the cascade's per-vertex conservation checks
(:data:`na49_mixing.resonance_decay.DECAY_CLOSURE_TOL`, absolute
``1e-9`` GeV) run in the excitation's center-of-mass frame -- the frame
in which the frozen ``draw_decay`` machinery keeps its documented closure
precision at the pinned beam boost (``gamma ~ 30``; boosting the
near-threshold vertices through the lab-frame parent velocity
reconstructs the invariant through ``1 - beta^2`` and amplifies the
component rounding beyond the absolute tolerance).  The collected final
states and the remnant line are boosted into the lab frame once, at
assembly, with the exact forward boost; the event's four-momentum
closure against the incoming sum is ``~1e-14`` relative.

Draw budget (the card's pre-registered counts)
----------------------------------------------

Exactly one uniform on the closed ``spectral_state`` substream (the
card-3 remnant selection), exactly three uniforms on the closed
``exchange_kinematics`` substream (two for the ``M^2`` marginal draw,
one for the ``t`` draw, in that order, from card 4), and the frozen
Round 4 budget of one ``decay_channel`` plus two ``decay_kinematics``
uniforms per decay vertex (the cascade vertices through
:func:`na49_mixing.cascade_decay.draw_decay`; a catalog handoff
continues the same two substreams into the byte-unchanged Round 2
kernel's own documented budget, the card-9 handoff convention).  The
``topology_selection`` class draw belongs to the runner (card 9);
:func:`generate_sd` receives its side and draws no class uniform.

The gap size
------------

The gap size is the R50 stored quantity (plan line 903): the rapidity
interval between the remnant and the nearest final-state hadron of the
opposite side (:func:`gap_size`; the rapidity interval is ``|y_remnant -
y_hadron|`` minimized over the excitation-side final-state hadrons).  In
the ``t -> 0`` limit an intact-proton remnant sits at its beam
four-vector, so the remnant rapidity is its beam rapidity there (the
intact target line's beam rapidity is ``0`` in the lab frame; the
projectile line's is the beam rapidity of the 158 GeV/c proton).  The
gap feeds the R51 gap-fraction observable (index item 3: no hadronic
final-state particle in the central window ``|y| < 0.5``); the class
record carries every final-state hadron, so the runner computes the
fraction read-only.

Hermetic: standard library and numpy only; cards 1-4's modules, the
Round 4 cascade modules, and the committed catalog and continuum
configuration are consumed read-only; no event generation driver, no
external tables, no Monte Carlo run (the acceptance is hermetic: pinned
traces, exact kinematics, draw budgets; the runs R50-R55 exercise the
class statistically).
"""

from __future__ import annotations

from .flavor_weights import vertex_ledger_record

import math
from dataclasses import dataclass

import numpy as np

from .cascade_decay import (
    CATALOG_FAMILY_PREFIX,
    ISOTROPIC_ALIGNMENT_LABEL,
    KIND_CONTINUUM,
    MAX_CASCADE_DEPTH,
    _cascade_depth_limit,
    _family_by_qn,
    CascadeState,
    continuing_pool,
    draw_decay_with_pool,
    enumerate_channels,
)
from .cascade_termination import (
    CascadeDepthGuardError,
    handoff_decay,
    has_valid_direct_channel,
    is_catalog_handoff,
)
from .exchange_ledger import LineState
from .hadron_catalog import HadronCatalog
from .isovector_exchange import (ISOVECTOR_EXCHANGE_NAME, PROTON_LINE, ExchangeConfiguration, LineCharge, line_mass_threshold,
                                 line_root_family, select_line_remnant)
from .one_pion_exchange import PION_MASS, draw_ope, ope_component_of, ope_neutral_fraction, ope_parts
from .two_sided_hadronization import split_kinematics, split_masses
from .pdg_database import get_particle
from .quantum_numbers import quantum_numbers
from .nd_duality import duality_pick, isoscalar_exchange_records
from .resonance_decay import ResonanceDecayer
from .rng_substreams import SUBSTREAM_LABELS
from .topology_elastic import BEAM_MOMENTUM, PROTON_MASS, SQRT_S
from .topology_mass_laws import (
    MassLawParams,
    draw_marginal,
    draw_marginal_edge,
    draw_t,
    draw_t_window,
    draw_t_window_break,
    draw_marginal_pomeron,
    draw_t_pomeron,
    pomeron_gap_log,
    pp_t_window,
    sd_limits,
    t_range,
    vertex_slope_factor,
)
from .topology_remnants import (
    KIND_INTACT_PROTON,
    PROTON_PDG_ID,
    RemnantSelection,
    select_remnant,
)
from .spectral_density import ContinuumFamily

__all__ = [
    "SDSIDES",
    "SD_CLASS_LABEL_PROJECTILE",
    "SD_CLASS_LABEL_TARGET",
    "NOMINAL_SD_REMNANT_WEIGHTS",
    "NOMINAL_PARAMS",
    "BEAM_MOMENTUM",
    "PROTON_MASS",
    "SQRT_S",
    "FinalHadron",
    "SDEvent",
    "gap_size",
    "generate_sd",
    "mirror_sd",
    "sd_kinematics",
]


#: The closed side set of the single-diffraction class (the card's
#: closed ``SDSide`` set): the side naming the EXCITED baryonic line.
#: ``"projectile"`` is the excited projectile-side system with the
#: remnant on the target line (plan line 825); ``"target"`` is the
#: mirrored class (plan line 826).
SDSIDES: frozenset[str] = frozenset({"projectile", "target"})

#: The class labels (the inelastic-sample class labels of the R50
#: templates; the elastic class label of card 1 stays separate).
SD_CLASS_LABEL_PROJECTILE: str = "sd_projectile"
SD_CLASS_LABEL_TARGET: str = "sd_target"

#: The nominal R50 remnant-kind split of the non-excited line (index
#: item 2; fixed, not fit): intact proton 0.5, low-mass baryonic
#: resonance 0.5, high-mass 0.0 (plan line 825 names exactly the intact
#: and low-mass options; the high-mass weight is the pre-registered
#: zero of the nominal split, the kind stays legal through the card-3
#: interface).
NOMINAL_SD_REMNANT_WEIGHTS: dict[str, float] = {
    "intact_proton": 0.5,
    "low_mass_baryonic_resonance": 0.5,
    "high_mass_baryonic_continuum": 0.0,
}

#: The nominal effective mass-law parameters (the R50 nominal values,
#: index item 2: epsilon = 0, B_SD the M8-frozen effective transfer
#: slope; fixed here, fit in R51 by card 11).
NOMINAL_PARAMS: MassLawParams = MassLawParams(0.0, 8.504140762083612)

#: The proton quantum numbers in the doubled-I3 integer convention
#: ``(B, Q, S, 2I3)`` (the committed catalog record pdg_id 2212; the
#: incoming lines and both outgoing lines inherit it through the vacuum
#: exchange).
PROTON_QN: tuple[int, int, int, int] = (1, 1, 0, 1)

#: The vacuum exchange name (the single color-singlet exchange of the
#: class; the transfer is ``(0, 0, 0, 0)``, plan line 877).
VACUUM_EXCHANGE_NAME: str = "vacuum"
ZERO_TRANSFER: tuple[int, int, int, int] = (0, 0, 0, 0)

#: The frozen simplified-cascade continuum family of the excited
#: baryonic system (the activated baryonic ``(1, 1, 0)`` family of
#: ``config/v2_singlet/continuum_resonances.json``; the cascade line's
#: ``family_id``).
EXCITATION_FAMILY_ID: str = "baryonic_np"


@dataclass(frozen=True)
class FinalHadron:
    """One final-state hadron of the event: its catalog identity (``None``
    for a below-threshold excitation that stays a single baryonic system
    with no open cascade channel), its lab-frame four-momentum ``(E, px,
    py, pz)`` in GeV, its mass [GeV], its doubled-I3 integer quantum
    numbers ``(B, Q, S, 2I3)``, and its line of origin (``"remnant"`` or
    ``"excitation"``)."""

    pdg_id: int | None
    four_momentum: tuple[float, float, float, float]
    mass: float
    qn: tuple[int, int, int, int]
    line: str


@dataclass(frozen=True)
class SDEvent:
    """The frozen single-diffraction event record (the card's
    ledger-ready record).

    ``class_label`` is ``"sd_projectile"`` or ``"sd_target"`` and
    ``side`` the closed :data:`SDSIDES` value naming the EXCITED side.
    ``remnant`` is the card-3 selection of the non-excited line.
    ``excitation_mass`` is the excitation mass ``M`` [GeV] and ``t`` the
    drawn momentum transfer [GeV^2].  ``lines`` is the card-2 ledger
    record pair of the two OUTGOING lines in the fixed order (projectile
    side, target side); the incoming lines are the pinned beam
    configuration (the module's :func:`_incoming_lines`).  ``exchange``
    is the single vacuum exchange record with the zero transfer.
    ``final_state`` carries every final-state hadron (the remnant line
    and the excitation's decay products).  ``gap_size`` is the R50 gap
    quantity (the rapidity interval of :func:`gap_size`).
    ``exchange_configuration`` (Round 8, 2026-10-04) is the drawn
    :class:`~na49_mixing.isovector_exchange.ExchangeConfiguration` of a
    run with a positive isovector fraction (``None`` otherwise); under the
    isovector exchange ``exchange`` is the projectile line's record and
    the target line takes the opposite transfer.
    ``excitation_m_min`` (2026-10-06) is the lower edge [GeV] of the
    excitation mass law under ``line_thresholds`` (the root family's lowest
    direct threshold; ``None`` for the shared edge
    :data:`~na49_mixing.topology_mass_laws.EXCITATION_M_MIN`).
    ``slope_factor`` (2026-10-08) is the factorized vertex slope factor
    ``B / b`` of the transfer under ``slope_mass_exponent``
    (:func:`~na49_mixing.topology_mass_laws.vertex_slope_factor`; ``None``
    for the single slope), with the intact proton's term f under
    ``intact_vertex_factor`` (2026-10-15).
    ``pomeron`` (2026-11-01) is ``True`` for an event drawn from the
    Pomeron's shrinkage law under ``pomeron_slope`` (the coherence-region
    vacuum events), ``False`` for the other events of such a run and ``None``
    without the option.
    """

    class_label: str
    side: str
    remnant: RemnantSelection
    excitation_mass: float
    t: float
    lines: tuple[LineState, LineState]
    exchange: tuple[str, tuple[int, int, int, int]]
    final_state: tuple[FinalHadron, ...]
    gap_size: float
    exchange_configuration: ExchangeConfiguration | None = None
    excitation_m_min: float | None = None
    slope_factor: float | None = None
    tail: bool | None = None
    ope: bool | None = None
    pomeron: bool | None = None


# ----------------------------------------------------------------------
# The pinned beam configuration and the ledger lines
# ----------------------------------------------------------------------
def _beam_configuration() -> tuple[float, float, float, float, float, float]:
    """The exact fixed-target two-body quantities of the pinned M8 p+p
    configuration: ``(e_beam, e_total, sqrt_s_exact, p_star, beta,
    gamma)``.

    ``sqrt_s_exact`` is the exact invariant mass of the incoming pair
    (the elastic module's recorded :data:`SQRT_S = 17.270 GeV` is the
    rounded quote; the phase-space draws of card 4 run on the recorded
    value, the exact solve and the boost on the exact invariant mass so
    the energy-momentum closure holds against the incoming lab sum).
    """

    m = PROTON_MASS
    p_beam = BEAM_MOMENTUM
    e_beam = math.sqrt(m * m + p_beam * p_beam)
    e_total = e_beam + m
    s = m * m + m * m + 2.0 * m * e_beam
    sqrt_s_exact = math.sqrt(s)
    p_star = math.sqrt(max(0.0, s - 4.0 * m * m)) / 2.0
    beta = p_beam / e_total
    gamma = e_total / sqrt_s_exact
    return e_beam, e_total, sqrt_s_exact, p_star, beta, gamma


def _incoming_lines() -> tuple[LineState, LineState]:
    """The pinned incoming ledger lines (projectile side first): the beam
    proton and the target proton, with the doubled-I3 proton quantum
    numbers and their lab four-momenta."""

    e_beam, _e_total, _sq, _ps, _beta, _gamma = _beam_configuration()
    return (
        LineState(B=1, Q=1, S=0, I3x2=1,
                  four_momentum=(e_beam, 0.0, 0.0, BEAM_MOMENTUM)),
        LineState(B=1, Q=1, S=0, I3x2=1,
                  four_momentum=(PROTON_MASS, 0.0, 0.0, 0.0)),
    )


def _lab_boost_factory():
    """The exact lab boost of the pinned configuration: a callable
    mapping a center-of-mass four-momentum to the lab frame with the
    exact fixed-target CM velocity (the forward boost form
    ``E' = gamma (E + beta pz)``, no ``1 - beta^2`` reconstruction; the
    form keeps the assembly closure at the float-rounding level)."""

    _e_beam, _e_total, _sq, _ps, beta, gamma = _beam_configuration()

    def boost(vector):
        e, px, py, pz = (float(x) for x in vector)
        return (gamma * (e + beta * pz), px, py, gamma * (pz + beta * e))

    return boost


def _cm_mirror_factory():
    """The lab-frame image of the center-of-mass parity mirror: a callable
    mapping a lab four-momentum to the lab four-momentum of its CM mirror
    (``p_z* -> -p_z*``), i.e. the inverse boost, the flip and the forward
    boost of :func:`_lab_boost_factory` in one step.  The target-side
    single-diffraction event is the image of the projectile-side
    construction under this map (2026-10-02, the feed-down and feed-over
    review: the target class had been a copy of the projectile class)."""

    _e_beam, _e_total, _sq, _ps, beta, gamma = _beam_configuration()

    def mirror(vector):
        e, px, py, pz = (float(x) for x in vector)
        e_cm = gamma * (e - beta * pz)
        pz_cm = gamma * (pz - beta * e)
        return (gamma * (e_cm - beta * pz_cm), px, py, gamma * (beta * e_cm - pz_cm))

    return mirror


# ----------------------------------------------------------------------
# Exact kinematics (the card's sd_kinematics)
# ----------------------------------------------------------------------
def sd_kinematics(
    m: float,
    t: float,
    m_remnant: float,
    sqrt_s: float,
    *,
    exact_range: bool = False,
) -> tuple[tuple[float, float, float, float],
           tuple[float, float, float, float]]:
    """Exact outgoing four-momenta ``(excitation, remnant)`` in the lab
    frame from ``(M, t, m_remnant, sqrt_s)``.

    *exact_range* (2026-10-07; review 2026-10-06, Grok L2) solves with the
    true incoming protons, ``E_p* = sqrt(s)/2``, as the exact-window branch
    of :func:`generate_sd` does, and raises when the transfer lies outside
    the reachable window (``|cos(theta*)| > 1 + 1e-9``) instead of
    clamping.  The default keeps the committed ``(m_p, m_remnant)``
    incoming system bit for bit.

    The exact two-body scattering ``p + p -> X(M) + R(m_remnant)`` of the
    pinned beam configuration: the center-of-mass energies and momenta
    from the invariant mass *sqrt_s* (the caller passes the recorded
    :data:`SQRT_S` for the draw-domain bookkeeping; the solve uses the
    exact invariant mass of the pinned configuration so the lab closure
    holds), the scattering angle from the invariant transfer on the
    excitation line,

        ``t = m_p^2 + M^2 - 2 (E_p* E_M* - p* q* cos(theta*))``,

    and the lab boost with the exact fixed-target CM velocity.  The
    production plane is the x-z plane.  Energy-momentum closure holds
    within ``1e-9`` relative by construction.

    The ``t = 0`` limit at the intact configuration (``M = m_p`` with the
    intact-proton remnant) is the elastic-forward configuration: the
    excitation four-vector equals the beam proton four-vector and the
    remnant four-vector equals its beam four-vector (the target at rest)
    within ``1e-12``.  For ``M > m_p`` the closed-at-0 draw range extends
    past the exact forward edge ``t_fwd < 0`` (see the module docstring);
    the angle saturates at the forward configuration there and the
    recorded *t* stays the drawn value.
    """

    mass = float(m)
    transfer = float(t)
    m_rem = float(m_remnant)
    root_s = float(sqrt_s)
    if not (math.isfinite(mass) and mass > 0.0):
        raise ValueError(
            "sd_kinematics: the excitation mass must be a positive finite "
            f"GeV value, got {m!r}")
    if not (math.isfinite(m_rem) and m_rem > 0.0):
        raise ValueError(
            "sd_kinematics: the remnant mass must be a positive finite "
            f"GeV value, got {m_remnant!r}")
    if not (math.isfinite(transfer) and transfer <= 0.0):
        raise ValueError(
            "sd_kinematics: the momentum transfer must be a non-positive "
            f"finite GeV^2 value, got {t!r}")
    if not math.isfinite(root_s) or root_s <= mass + m_rem:
        raise ValueError(
            "sd_kinematics: sqrt_s must be a finite GeV value above the "
            f"two-body threshold {mass + m_rem!r}, got {sqrt_s!r}")

    e_beam, _e_total, sq_exact, p_star_cm, beta, gamma = _beam_configuration()
    s = sq_exact * sq_exact
    mp2 = PROTON_MASS * PROTON_MASS
    m2 = mass * mass
    mr2 = m_rem * m_rem

    # The two-body CM energies and momenta (incoming (m_p, m_remnant),
    # outgoing (M, m_remnant)).
    e_p_cm = 0.5 * sq_exact if exact_range else (s + mp2 - mr2) / (2.0 * sq_exact)
    e_m_cm = (s + m2 - mr2) / (2.0 * sq_exact)
    e_rem_cm = (s + mr2 - m2) / (2.0 * sq_exact)
    p_star = (math.sqrt(max(0.0, 0.25 * s - mp2)) if exact_range else
              math.sqrt(max(0.0, (s - (PROTON_MASS + m_rem) ** 2)
                            * (s - (PROTON_MASS - m_rem) ** 2))) / (2.0 * sq_exact))
    q_star = math.sqrt(max(0.0, (s - (mass + m_rem) ** 2)
                           * (s - (mass - m_rem) ** 2))) / (2.0 * sq_exact)
    if p_star <= 0.0 or q_star <= 0.0:
        raise ValueError(
            "sd_kinematics: the two-body system carries no phase space at "
            f"(M={mass!r}, m_remnant={m_rem!r}, sqrt_s={root_s!r}) on the "
            "pinned beam configuration")

    # The scattering angle from the invariant transfer on the excitation
    # line, t = m_p^2 + M^2 - 2 (E_p* E_M* - p* q* cos(theta*)) solved
    # for cos(theta*); the closed-at-0 forward convention saturates at
    # the forward configuration for the drawn sliver above the exact
    # forward edge.
    cos_theta = (e_p_cm * e_m_cm - (mp2 + m2) / 2.0
                 + transfer / 2.0) / (p_star * q_star)
    if exact_range and abs(cos_theta) > 1.0 + 1e-9:
        raise ValueError(f"sd_kinematics: the transfer {t!r} lies outside the reachable window "
                         f"(cos(theta*) = {cos_theta!r})")
    cos_theta = max(-1.0, min(1.0, cos_theta))
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))

    excitation_cm = (e_m_cm, q_star * sin_theta, 0.0, q_star * cos_theta)
    remnant_cm = (e_rem_cm, -q_star * sin_theta, 0.0, -q_star * cos_theta)
    return _lab_boost_factory()(excitation_cm), _lab_boost_factory()(remnant_cm)


# ----------------------------------------------------------------------
# The gap size (the R50 stored quantity)
# ----------------------------------------------------------------------
def _rapidity(four_momentum: tuple[float, float, float, float]) -> float:
    """The lab rapidity ``y = ln((E + pz)/(E - pz)) / 2`` of a
    four-momentum (``ValueError`` on a null or lightlike-in-z argument:
    checked, never repaired)."""

    e, _px, _py, pz = (float(x) for x in four_momentum)
    plus = e + pz
    minus = e - pz
    if not (plus > 0.0 and minus > 0.0):
        raise ValueError(
            "gap_size: the four-momentum carries a null or non-timelike "
            f"longitudinal component (E + pz = {plus!r}, E - pz = "
            f"{minus!r}); the rapidity is undefined"
        )
    return 0.5 * math.log(plus / minus)


def gap_size(event: SDEvent) -> float:
    """The R50 gap quantity of *event*: the rapidity interval between the
    remnant and the nearest final-state hadron of the opposite side.

    The interval is ``|y_remnant - y_hadron|`` minimized over the
    final-state hadrons carried on the excitation line (the
    ``"excitation"`` origin of :class:`FinalHadron`).  In the ``t -> 0``
    limit an intact-proton remnant sits at its beam four-vector, so its
    rapidity is its beam rapidity there (the intact target line's beam
    rapidity is ``0`` in the lab frame).  The function recomputes the
    interval from the event's records (read-only, zero random numbers)
    and raises ``ValueError`` on an event whose recorded gap disagrees
    beyond ``1e-12`` (checked, never repaired).
    """

    if not isinstance(event, SDEvent):
        raise ValueError(
            f"gap_size: expected an SDEvent, got {type(event)!r}")
    remnant_hadrons = [h for h in event.final_state if h.line == "remnant"]
    excitation_hadrons = [
        h for h in event.final_state if h.line == "excitation"]
    if not remnant_hadrons or not excitation_hadrons:
        raise ValueError(
            "gap_size: the event must carry the remnant system (one intact "
            "proton or a low-mass record's decay products) and at least one "
            f"excitation final state, got {len(remnant_hadrons)} remnant and "
            f"{len(excitation_hadrons)} excitation hadrons")
    y_remnant = _remnant_rapidity(event.final_state)
    gap = min(abs(_rapidity(h.four_momentum) - y_remnant)
              for h in excitation_hadrons)
    if abs(gap - float(event.gap_size)) > 1e-12:
        raise ValueError(
            "gap_size: the event's recorded gap "
            f"{float(event.gap_size)!r} disagrees with the recomputed "
            f"interval {gap!r} beyond 1e-12 (checked, never repaired)"
        )
    return gap


# ----------------------------------------------------------------------
# The excitation decay through the frozen simplified cascade
# ----------------------------------------------------------------------
def _final_from_catalog(catalog: HadronCatalog, pdg_id: int,
                        four_momentum, line: str) -> FinalHadron:
    """The final-state hadron record of a catalog daughter (its pole
    mass, its doubled-I3 quantum numbers ``(B, Q, S, 2I3)`` from the
    committed record's fields, and the caller's line of origin)."""

    p4 = (float(four_momentum[0]), float(four_momentum[1]),
          float(four_momentum[2]), float(four_momentum[3]))
    try:
        record = catalog.get(int(pdg_id))
    except KeyError:
        # a stable state outside the records (the photon of the
        # radiative channels, 2026-10-02): the vendored PDG table's mass
        # and the integer (B, Q, S), with 2 I3 = 2Q - B - S
        if int(pdg_id) not in catalog.stable_states():
            raise
        entry = get_particle(int(pdg_id))
        qn = quantum_numbers(int(pdg_id))
        return FinalHadron(
            pdg_id=int(pdg_id), four_momentum=p4, mass=float(entry.mass),
            qn=(int(qn.B), int(qn.Q), int(qn.S), 2 * int(qn.Q) - int(qn.B) - int(qn.S)),
            line=line)
    return FinalHadron(
        pdg_id=int(pdg_id),
        four_momentum=p4,
        mass=float(record.pole_mass),
        qn=(int(record.B), int(record.Q), int(record.S),
            int(record.isospin3)),
        line=line,
    )


def _kernel_finals(catalog: HadronCatalog, vertices, line: str):
    """The stable final-state hadrons of a completed Round 2 kernel
    vertex list: the daughters that never entered the kernel's decay
    queue (``is_resonance`` false), which are final by the kernel's own
    termination."""

    for vertex in vertices:
        for daughter in vertex.daughters:
            if not daughter.is_resonance:
                yield _final_from_catalog(catalog, daughter.pdg_id,
                                          daughter.four_momentum, line)


def _assemble_daughter(catalog: HadronCatalog, pdg_id: int, four_momentum, line: str,
                       rng_channel, rng_kinematics) -> list[FinalHadron]:
    """One established daughter into the final state: a stable record is
    final; an unstable record decays through the Round 2 kernel on the
    passed substreams (the continuation vertices' sibling rule, applied
    at every assembly point since the 2026-09-30 final-state fix; before
    it, the direct-channel daughters and the single-diffraction remnant
    entered the final state undecayed)."""

    pdg = int(pdg_id)
    if pdg in catalog.stable_states():
        return [_final_from_catalog(catalog, pdg, four_momentum, line)]
    kernel = ResonanceDecayer(catalog, 0, 0, 0)
    kernel._rng_channel = rng_channel
    kernel._rng_kinematics = rng_kinematics
    return list(_kernel_finals(
        catalog, kernel.decay_resonance(pdg, tuple(float(x) for x in four_momentum)), line))


def _remnant_rapidity(final_state) -> float:
    """The rapidity of the remnant system: the summed four-momentum of the
    remnant-line hadrons (one intact proton, or a low-mass record's decay
    products)."""

    rem = [h.four_momentum for h in final_state if h.line == "remnant"]
    total = tuple(math.fsum(float(p[j]) for p in rem) for j in range(4))
    return _rapidity(total)


def _record_decay(momentum_cm, mass: float, records, catalog: HadronCatalog, rng_channel, rng_kinematics,
                  line: str) -> list[FinalHadron]:
    """A low-mass line decaying as one of *records* (the diffractive
    duality, 2026-10-15): the record picked by
    :func:`~na49_mixing.nd_duality.duality_pick` on one uniform of
    *rng_channel*, then decayed by the kernel at the line's four-momentum."""
    pdg = duality_pick(float(mass), records, float(rng_channel.random()))
    kernel = ResonanceDecayer(catalog, 0, 0, 0)
    kernel._rng_channel = rng_channel
    kernel._rng_kinematics = rng_kinematics
    return list(_kernel_finals(catalog, kernel.decay_resonance(int(pdg), tuple(float(v) for v in momentum_cm)), line))


def _decay_excitation(
    excitation_cm: tuple[float, float, float, float],
    mass: float,
    catalog: HadronCatalog,
    fams: list[ContinuumFamily],
    rng_channel,
    rng_kinematics,
    flavor=None,
    ledger=None,
    line: str = "excitation",
    line_qn: tuple = (1, 1, 0),
    line_isospin: float | None = None,
) -> list[FinalHadron]:
    """One excitation decay through the frozen simplified cascade (the
    Round 4 modules consumed read-only): the card-9 termination rules
    with the caller's event assembly of the vertex siblings, in the
    center-of-mass frame of the excitation (the frame convention of the
    module docstring).

    At every line state, in the card-9 order: the hard depth guard; the
    catalog handoff (the byte-unchanged Round 2 kernel completes the
    remainder on the passed substreams and the kernel's stable daughters
    are final); the continuum continuation (one ``draw_decay`` step over
    the continuum pool; the established sibling is assembled -- a stable
    record is final, an unstable record decays through the kernel's own
    budget on the same substreams); the direct-channel termination (one
    ``draw_decay`` step over the direct pool; both daughters are
    assembled).  A state with no open channel at all is the
    below-threshold excitation (plan line 842: no minimum N-pi
    excitation is imposed): it stays a single baryonic final state at its
    mass with no vertex and zero decay draws.

    Every assembled final-state hadron carries the *line* label (``"excitation"``
    by default; the non-diffractive remnants pass their own line)
    of origin and its center-of-mass four-momentum (the caller boosts to
    the lab frame once, at assembly).

    The root carries the line's ``(B, Q, S)`` *line_qn* and starts in the
    continuum family of those quantum numbers (the general line model,
    2026-10-04; the vacuum exchange keeps every line at ``(1, 1, 0)``,
    the ``baryonic_np`` family, so the cascade is unchanged).  A line of
    the isovector exchange passes its *line_isospin*: the root is then the
    family of the line's quantum numbers and isospin
    (:func:`~na49_mixing.isovector_exchange.line_root_family`).
    """

    finals: list[FinalHadron] = []
    root_cm = (float(excitation_cm[0]), float(excitation_cm[1]),
               float(excitation_cm[2]), float(excitation_cm[3]))
    root_qn = tuple(int(v) for v in line_qn)
    if line_isospin is None:
        root_family = _family_by_qn(fams, root_qn)
    else:
        root_family = line_root_family(fams, LineCharge(root_qn[1], line_isospin))
    if root_family is None:
        raise ValueError(f"_decay_excitation: no continuum family carries the line quantum numbers {root_qn}")
    current = CascadeState(
        momentum=root_cm,
        mass=float(mass),
        qn=root_qn,
        quantum={},
        family_id=str(root_family.family_id),
        axis=(0.0, 0.0, 1.0),
        alignment_label=ISOTROPIC_ALIGNMENT_LABEL,
        parent=None,
        exchange_ancestry=(SD_CLASS_LABEL_PROJECTILE,),
        depth=0,
    )
    while True:
        # (1) the hard maximum depth guard (plan section 10.6, line 757)
        if int(current.depth) > _cascade_depth_limit(fams):
            raise CascadeDepthGuardError(
                f"single-diffraction excitation cascade: the line state of "
                f"family {current.family_id!r} at mass "
                f"{float(current.mass)!r} GeV carries depth "
                f"{int(current.depth)}, exceeding MAX_CASCADE_DEPTH = "
                f"{_cascade_depth_limit(fams)}")
        # (2) the catalog handoff: the Round 2 kernel completes the
        # remainder (its own documented draw budget on the same
        # substreams; card 9's handoff convention).
        if is_catalog_handoff(current, catalog):
            vertices = handoff_decay(current, catalog, rng_channel,
                                     rng_kinematics)
            finals.extend(_kernel_finals(catalog, vertices, line))
            return finals
        # (3) the enumeration at the realized mass (card 4, read-only)
        channels = enumerate_channels(current, catalog, fams, flavor=flavor)
        continuum = continuing_pool(current, channels, catalog, fams)
        if continuum:
            _channel, daughters, drawn = draw_decay_with_pool(
                current, continuum, catalog, fams,
                rng_channel, rng_kinematics)
            if ledger is not None:
                ledger.append(vertex_ledger_record(
                    current.qn, drawn, _channel, catalog, fams, flavor))
            if _channel.kind != KIND_CONTINUUM:
                # a direct baryon pair of the continuation gap
                # (KernelConfig.direct_baryon_pairs_in_gap, 2026-10-21):
                # both catalog daughters assembled, the branch terminates
                for daughter in daughters:
                    pdg = int(daughter.family_id[len(CATALOG_FAMILY_PREFIX):])
                    finals.extend(_assemble_daughter(
                        catalog, pdg, daughter.momentum, line,
                        rng_channel, rng_kinematics))
                return finals
            successor = None
            for daughter in daughters:
                if daughter.family_id.startswith(CATALOG_FAMILY_PREFIX):
                    pdg = int(daughter.family_id[len(CATALOG_FAMILY_PREFIX):])
                    if pdg in catalog.stable_states():
                        finals.append(_final_from_catalog(
                            catalog, pdg, daughter.momentum, line))
                    else:
                        # an unstable established sibling decays through
                        # the kernel's own budget (the caller's event
                        # assembly of card 9's dropped siblings)
                        kernel = ResonanceDecayer(catalog, 0, 0, 0)
                        kernel._rng_channel = rng_channel
                        kernel._rng_kinematics = rng_kinematics
                        sibling_cm = (float(daughter.momentum[0]),
                                      float(daughter.momentum[1]),
                                      float(daughter.momentum[2]),
                                      float(daughter.momentum[3]))
                        finals.extend(_kernel_finals(
                            catalog,
                            kernel.decay_resonance(pdg, sibling_cm),
                            line))
                else:
                    successor = daughter
            if successor is None:
                raise ValueError(
                    "single-diffraction excitation cascade: the continuum "
                    "vertex produced no continuing daughter (card 4's "
                    "enumeration produces exactly one; checked, never "
                    "repaired)")
            current = successor
            continue
        # (4) the direct-channel termination: both daughters assembled.
        if has_valid_direct_channel(current, channels):
            direct = [c for c in channels if c.kind != KIND_CONTINUUM]
            _channel, daughters, drawn = draw_decay_with_pool(
                current, direct, catalog, fams,
                rng_channel, rng_kinematics)
            if ledger is not None:
                ledger.append(vertex_ledger_record(
                    current.qn, drawn, _channel, catalog, fams, flavor))
            for daughter in daughters:
                pdg = int(daughter.family_id[len(CATALOG_FAMILY_PREFIX):])
                finals.extend(_assemble_daughter(
                    catalog, pdg, daughter.momentum, line,
                    rng_channel, rng_kinematics))
            return finals
        # (5) the below-threshold excitation: no open channel at all, so
        # the excitation stays a single baryonic final state at its mass
        # (no minimum N-pi excitation is imposed, plan line 842; zero
        # decay draws).
        finals.append(FinalHadron(
            pdg_id=None,
            four_momentum=(float(current.momentum[0]),
                           float(current.momentum[1]),
                           float(current.momentum[2]),
                           float(current.momentum[3])),
            mass=float(current.mass),
            # the line state's own (B, Q, S, 2 I3): only the root can be
            # below threshold here, but a continuation daughter carries its
            # family's quantum numbers (2026-10-01)
            qn=(int(current.qn[0]), int(current.qn[1]), int(current.qn[2]),
                2 * int(current.qn[1]) - int(current.qn[0]) - int(current.qn[2])),
            line=line,
        ))
        return finals


# ----------------------------------------------------------------------
# The event generator
# ----------------------------------------------------------------------
def generate_sd(
    side: str,
    params: MassLawParams,
    weights,
    catalog: HadronCatalog,
    fams: list[ContinuumFamily],
    rng_spectral_state,
    rng_exchange_kinematics,
    rng_decay_channel,
    rng_decay_kinematics,
    flavor=None,
    ledger=None,
    exchange: ExchangeConfiguration | None = None,
    exact_range: bool = False,
    line_thresholds: bool = False,
    slope_mass_exponent: float | None = None,
    low_mass_spin_isospin: bool = False,
    intact_vertex_factor: float | None = None,
    diffractive_duality: float | None = None,
    duality_line_shapes: bool = False,
    xi_max: float | None = None,
    tail: "tuple[float, float, float, float] | None" = None,
    tail_component: "tuple[float, ...] | None" = None,
    t_break: "tuple[float, float] | None" = None,
    ope_component: "tuple[float, ...] | None" = None,
    two_sided: "tuple | None" = None,
    pomeron_slope: float | None = None,
    pomeron_linear_t: float | None = None,
) -> SDEvent:
    """One single-diffraction event of the side *side*.

    The card's pre-registered draw budget on the closed substream labels
    (:data:`na49_mixing.rng_substreams.SUBSTREAM_LABELS`): exactly one
    uniform on ``spectral_state`` (the card-3 remnant selection over the
    caller's *weights*; the nominal split intact 0.5 / low-mass 0.5 /
    high-mass 0.0), exactly three uniforms on ``exchange_kinematics``
    (the ``M^2`` marginal draw of card 4 on the exact limits
    ``[m_p^2, (sqrt_s - m_remnant)^2]`` -- two uniforms -- then the ``t``
    draw of ``exp(B_SD t)`` on ``[t_min(M, m_remnant), 0]`` -- one
    uniform), and the frozen Round 4 budget of one channel plus two
    kinematics uniforms per decay vertex on ``decay_channel`` /
    ``decay_kinematics`` (the frozen simplified cascade; a catalog
    handoff continues into the byte-unchanged Round 2 kernel's own
    documented budget on the same substreams).  The
    ``topology_selection`` class draw belongs to the runner (card 9):
    the side is an argument, no class uniform is consumed.

    The kinematics solve and the lab assembly follow :func:`sd_kinematics`
    and the module docstring's frame convention.  The target side is the
    center-of-mass parity mirror of the projectile-side construction on
    the same draws (:func:`_cm_mirror_factory`; corrected 2026-10-02: the
    target class had been a copy of the projectile class).  The event's ``lines``
    are the two outgoing ledger line states (projectile side first); the
    ``exchange`` record is the vacuum exchange with the zero transfer;
    the ``final_state`` carries the remnant hadron and the excitation's
    decay products in the lab frame; the ``gap_size`` is the R50
    interval of :func:`gap_size`.

    *exchange* (Round 8, 2026-10-04) is the event's drawn exchange
    configuration (the runner draws it at a positive isovector fraction).
    ``None`` or the vacuum configuration gives the event above.  Under the
    isovector exchange the remnant line and the excited line take their
    configured charges and isospins: the remnant through
    :func:`~na49_mixing.isovector_exchange.select_line_remnant` (the same
    one uniform), the excitation's cascade from the line's root family;
    the draw budget is unchanged.

    *exact_range* (2026-10-05) draws t on the reachable window of the
    realized pair (:func:`~na49_mixing.topology_mass_laws.pp_t_window`, the
    true incoming protons) instead of ``[t_min, 0]``; the solve then never
    clamps the production angle.  Same draw budget.

    *line_thresholds* (2026-10-06; code-physics review M-6) starts the
    excitation mass law at the excited line's own lowest direct threshold
    (:func:`~na49_mixing.isovector_exchange.line_mass_threshold`: p pi0 for
    Q = 1, n pi0 for Q = 0, p pi+ for Q = 2) instead of the shared p pi0
    edge, so no root lies below its open channels.  The edge is recorded as
    ``excitation_m_min``.  Same draw budget.

    *diffractive_duality* (2026-10-15) is the mass below which the vacuum
    excitation decays as an I = 1/2 low-mass record whose range holds its
    mass (:func:`~na49_mixing.nd_duality.isoscalar_exchange_records`), picked
    with weight (2J + 1) times its Breit-Wigner on one ``decay_channel``
    uniform: the vacuum exchange carries I = 0 and excites the proton to
    I = 1/2 states.  ``None`` gives the continuum cascade.

    *intact_vertex_factor* (2026-10-15) gives the intact proton's vertex
    the term f of the factorized vertex slopes under
    *slope_mass_exponent*: the transfer of an event whose non-excited line
    is the intact proton follows ``B = b [f + (m_p/M_X)^kappa] / 2``.  The
    intact line's vertex is the elastic proton vertex, whose slope in p+p
    elastic scattering (about 11 GeV^-2 at sqrt(s) = 17 GeV, half of it
    per vertex) lies above the excited-vertex law at the proton mass.  A
    low-mass resonance remnant keeps its committed term, and so does the
    ground state of an isovector line when it is not the proton (the
    neutron or a Delta(1232), a charge-exchange vertex; review 2026-10-16).
    The recorded ``slope_factor`` carries f, so the slope reweighting is
    unchanged.
    Same draw budget; ``None`` gives the committed event.

    *ope_component* (2026-10-26) ``(share, xi_lo, xi_hi, alpha', r2)`` is
    the one-pion exchange (the form factor ``"proton"`` in place of ``r2``
    and a sixth entry ``"regge"`` for ``sigma_pi+p(M^2)``, see
    :func:`~na49_mixing.one_pion_exchange.ope_component_of`): the component uniform that picks the tail (one
    uniform on ``exchange_kinematics``, the vacuum events only) gives the
    band ``[s_tail, s_tail + share)`` to ``p + p -> n + X++``.  The
    non-excited line turns into a neutron (the remnant of the pi+ exchange
    configuration, the same one ``spectral_state`` uniform) and the excited
    system is the pi+ p line at Q = 2, I = 3/2; ``(M^2, t)`` follow the pion
    flux of :mod:`na49_mixing.one_pion_exchange` jointly on the reachable
    window on ``[xi_lo, xi_hi]`` (the same 3 uniforms), with no vertex
    factors.  The event records ``ope = True``.  Every other event is the
    one drawn without the component, bit for bit.  With the law's pion
    ``"isospin"`` (a seventh entry, 2026-10-27) the share is the whole pion
    exchange: the band's upper part, the neutral fraction of
    :func:`~na49_mixing.one_pion_exchange.ope_neutral_fraction` on the
    window at the ground-state remnants (0.347 for Eq. 39), is
    ``p + p -> p + X+``, the non-excited line a proton and the excited
    pi0 p line at Q = 1, I = 3/2 in the first two thirds of that part and
    I = 1/2 in the last (the Clebsch-Gordan shares of pi0 p), drawn on the
    neutral law; the same uniform, no new draw.  Those events record
    ``ope = True`` as well.

    *two_sided* (2026-10-26) ``(m_cut, eps_line, b, scope)`` hadronizes the
    excitations above ``m_cut`` of the scope (its letters: ``o`` the one-pion
    exchange, ``t`` the tail, ``c`` the coherence region) as two lines
    (:mod:`na49_mixing.two_sided_hadronization`): a baryonic line along the
    X flight direction and the exchanged object's line opposite, (0, +1, 0)
    for the one-pion exchange's pi+ and (0, 0, 0) otherwise (the pi0 of the
    neutral exchange among them), each then
    decayed through the cascade.  Four more uniforms on ``decay_kinematics``
    for a split event; every other event is unchanged.

    *pomeron_slope* (2026-11-01) alpha'_P [GeV^-2] draws the coherence-region
    vacuum events (not the tail, the one-pion exchange or an isovector
    exchange) from the triple-Pomeron law at fixed transfer,
    ``xi^(-1 - eps - 2 alpha' t)`` times the vertex law: ``(M^2, t)`` from
    ``(M^2)^(-1-eps) h_{B+c}(t) / Z_B`` with ``c = 2 alpha' ln(s m_p^2 / (M^2
    m_rem^2))`` (:func:`~na49_mixing.topology_mass_laws.draw_marginal_pomeron`,
    :func:`~na49_mixing.topology_mass_laws.draw_t_pomeron`; the section "The
    Pomeron's shrinkage" of that module).  Same draw budget (2 + 1 uniforms);
    needs *exact_range*.  The event records ``pomeron``.  ``None`` gives the
    committed event.  *pomeron_linear_t* t_lin holds the slope alpha' down to
    ``t = -t_lin`` and the exponent constant below it (the committed slope
    below the break, which must equal t_lin for the intact proton).

    ``ValueError`` on a side outside :data:`SDSIDES`, a weights mapping
    outside the card-3 kinds, or an unknown catalog/family surface
    (checked, never repaired).
    """

    if side not in SDSIDES:
        raise ValueError(
            f"generate_sd: the side {side!r} is outside the closed "
            f"single-diffraction side set {sorted(SDSIDES)}")
    if not isinstance(params, MassLawParams):
        raise ValueError(
            f"generate_sd: expected MassLawParams, got {type(params)!r}")
    if not isinstance(catalog, HadronCatalog):
        raise ValueError(
            f"generate_sd: expected a HadronCatalog, got {type(catalog)!r}")
    for label in ("spectral_state", "exchange_kinematics",
                  "decay_channel", "decay_kinematics"):
        if label not in SUBSTREAM_LABELS:
            raise ValueError(
                f"generate_sd: the substream label {label!r} is not in "
                f"the closed label set {SUBSTREAM_LABELS}")

    # (0) the class component (2026-10-26): one uniform on exchange_kinematics, before the mass draw, on the vacuum
    # events of a class with a component (the isovector events keep the coherence-region law): the leading-proton tail
    # below its share s (R112), the one-pion exchange in the next band of its share s_pi, the coherence region above.
    # The streams of (0) and (1) are independent, so the order leaves every committed draw unchanged.
    isovector = exchange is not None and not exchange.is_vacuum
    is_tail = None
    tail_edge = None
    is_ope = None
    ope_vals = None
    ope_law = None
    tail_vals = None
    if tail_component is not None:
        # (share, xi_lo, xi_hi, a, b) or, with the smooth upper edge (2026-10-25,
        # at the R116 optimum), (share, xi_lo, xi_hi, a, b, xi_c, w)
        tail_vals = tuple(float(v) for v in tail_component)
        if len(tail_vals) not in (5, 7):
            raise ValueError(f"generate_sd: tail_component is (share, xi_lo, xi_hi, a, b[, xi_c, w]), got {tail_component!r}")
    if ope_component is not None:
        # (share, xi_lo, xi_hi, alpha', r2 | "proton"[, pion_proton]): the one-pion-exchange law of
        # na49_mixing.one_pion_exchange
        if not exact_range:
            raise ValueError(f"generate_sd: ope_component needs the exact window, got {ope_component!r}")
        *ope_vals, ope_law = ope_component_of(ope_component)
    if tail_vals is not None or ope_vals is not None:
        share_t = tail_vals[0] if tail_vals is not None else 0.0
        share_o = ope_vals[0] if ope_vals is not None else 0.0
        if not 0.0 <= share_t + share_o < 1.0:
            raise ValueError(f"generate_sd: the component shares must sum below 1, got {share_t!r} + {share_o!r}")
        u_component = None if isovector else float(rng_exchange_kinematics.random())
        if tail_vals is not None:
            is_tail = u_component is not None and u_component < share_t
        if ope_vals is not None:
            is_ope = u_component is not None and share_t <= u_component < share_t + share_o
    if is_tail:
        share, xi_lo_c, xi_hi_c, a_c, b_c = tail_vals[:5]
        tail = (xi_lo_c, xi_hi_c, -1.0 - a_c, b_c)
        if len(tail_vals) == 7:
            if not (xi_lo_c < tail_vals[5] < xi_hi_c and tail_vals[6] > 0.0):
                raise ValueError(f"generate_sd: the tail's edge needs xi_lo < xi_c < xi_hi and w > 0, got {tail_component!r}")
            tail_edge = (tail_vals[5], tail_vals[6])
    neutral_isospin = None
    if is_ope:
        # the one-pion exchange (2026-10-26): the non-excited line turns into a neutron by emitting a pi+, which forms
        # the excited system with the other proton (Q = 2, I = 3/2, the only isospin of pi+ p)
        if ope_law.pion == "isospin":
            # the neutral partner (2026-10-27): the band's upper part f0 is the pi0 exchange, its pi0 p line at I = 3/2
            # in the first two thirds of that part and I = 1/2 in the last, on the same uniform
            f0 = _ope_neutral_fraction(ope_vals, ope_law)
            edge = share_t + share_o * (1.0 - f0)
            if u_component >= edge:
                neutral_isospin = 1.5 if u_component < edge + share_o * f0 * 2.0 / 3.0 else 0.5
            ope_law = ope_parts(ope_law)[0 if neutral_isospin is None else 1]
        exchange = _ope_configuration(side, neutral_isospin)
        isovector = True

    # (1) the remnant of the non-excited line (1 uniform, spectral_state)
    if isovector:
        excited_line, remnant_line = ((exchange.projectile, exchange.target) if side == "projectile"
                                      else (exchange.target, exchange.projectile))
        remnant = select_line_remnant(weights, catalog, rng_spectral_state, _never_mass_sampler, remnant_line)
    else:
        remnant = select_remnant(
            KIND_INTACT_PROTON, weights, catalog, rng_spectral_state,
            _never_mass_sampler, spin_isospin=low_mass_spin_isospin)

    # (2) the excitation mass and transfer (3 uniforms, exchange_
    # kinematics: 2 for M^2, 1 for t, card 4's order)
    m2_min, m2_max = sd_limits(remnant.mass, SQRT_S)
    if xi_max is not None and tail is None and not is_ope:
        # the coherence region (2026-10-23): the excited mass below M^2/s = xi_max
        m2_max = min(m2_max, float(xi_max) * SQRT_S * SQRT_S)
    m_edge = None
    if line_thresholds:
        m_edge = line_mass_threshold(fams, catalog, excited_line if isovector else None)
        m2_min = m_edge * m_edge
    mass_params = params
    # the Pomeron's shrinkage (2026-11-01): the coherence-region vacuum events
    pomeron = pomeron_slope is not None and not isovector and tail is None and not is_ope
    if pomeron and not exact_range:
        raise ValueError("generate_sd: the Pomeron's shrinkage needs the exact transfer window")
    pomeron_proton = remnant.pdg_id == PROTON_PDG_ID
    pomeron_f_term = (0.5 * (float(intact_vertex_factor) - 1.0)
                      if (slope_mass_exponent is not None and intact_vertex_factor is not None and pomeron_proton) else 0.0)
    pomeron_t_break = t_break if (t_break is not None and pomeron_proton) else None
    if tail is not None:
        # the leading-proton tail beyond the coherence region (2026-10-23, a probe
        # form of the reggeized pion exchange): the excited mass on
        # [xi_lo s, xi_hi s] with the marginal (M^2)^(-1-eps_tail) and the
        # transfer from exp(b_tail t) on the exact window, no vertex factors
        xi_lo, xi_hi, eps_tail, b_tail = (float(v) for v in tail)
        if not (exact_range and 0.0 < xi_lo < xi_hi < 1.0 and b_tail > 0.0):
            raise ValueError(f"generate_sd: the tail needs the exact window and 0 < xi_lo < xi_hi < 1, b > 0, got {tail!r}")
        m2_min = max(m2_min, xi_lo * SQRT_S * SQRT_S)
        m2_max = min(m2_max, xi_hi * SQRT_S * SQRT_S)
        mass_params = MassLawParams(epsilon=eps_tail, b_sd=b_tail)
    if is_ope:
        # the one-pion exchange's (M^2, t) jointly on the reachable window: the same 3 uniforms
        m2_min = max(m2_min, ope_vals[1] * SQRT_S * SQRT_S)
        m2_max = min(m2_max, ope_vals[2] * SQRT_S * SQRT_S)
        m2_excitation, transfer = draw_ope(m2_min, m2_max, remnant.mass * remnant.mass, SQRT_S * SQRT_S,
                                           ope_law, rng_exchange_kinematics)
    elif pomeron:
        # the shrinkage law's marginal (M^2)^(-1-eps) Z_{B+c} / Z_B on the same 2 uniforms
        m2_excitation = draw_marginal_pomeron(m2_min, m2_max, params.epsilon, params.b_sd, remnant.mass,
                                              slope_mass_exponent, pomeron_f_term, float(pomeron_slope),
                                              pomeron_t_break, rng_exchange_kinematics, t_lin=pomeron_linear_t)
    elif tail_edge is not None:
        # the tail's smooth upper edge in place of the wall at xi_hi (2026-10-25)
        m2_excitation = draw_marginal_edge(m2_min, m2_max, mass_params, SQRT_S * SQRT_S, tail_edge[0], tail_edge[1],
                                           rng_exchange_kinematics)
    else:
        m2_excitation = draw_marginal(m2_min, m2_max, mass_params, 0.0,
                                      rng_exchange_kinematics)
    slope_factor = None
    if is_ope:
        # the flux carries its own transfer law (no vertex factors); the record keeps a neutral slope factor
        slope_factor = 1.0 if slope_mass_exponent is not None else None
    elif exact_range and tail is not None:
        t_lo, t_hi = pp_t_window(m2_excitation, remnant.mass * remnant.mass)
        if t_break is not None and remnant.pdg_id == PROTON_PDG_ID:
            # the intact proton's two-slope transfer (2026-10-24), the tail's law above the break
            transfer = draw_t_window_break(t_lo, t_hi, mass_params, t_break, rng_exchange_kinematics)
        else:
            transfer = draw_t_window(t_lo, t_hi, mass_params, rng_exchange_kinematics)
        if slope_mass_exponent is not None:
            # the tail's slope as the factor B / b_sd of the record (2026-10-24: every event of a class carries the
            # same shape arrays; R112 stopped on the tail events' missing factor)
            slope_factor = float(mass_params.b_sd) / float(params.b_sd)
    elif exact_range:
        t_lo, t_hi = pp_t_window(m2_excitation, remnant.mass * remnant.mass)
        t_params = params
        if slope_mass_exponent is not None:
            # the factorized vertex slopes (2026-10-08): the transfer's slope
            # follows the two outgoing lines' masses
            slope_factor = vertex_slope_factor(math.sqrt(m2_excitation), remnant.mass, slope_mass_exponent)
            if intact_vertex_factor is not None and remnant.pdg_id == PROTON_PDG_ID:
                # the elastic proton vertex of a proton line (2026-10-15): its
                # term f in place of (m_p/m_p)^kappa = 1.  The proton carries it
                # whatever the remnant kind (the low-mass set includes the
                # ground state); the neutron or Delta ground state of an
                # isovector line is a charge-exchange vertex and keeps the
                # factorized law.  This is the remnant the reweighting of
                # log_density_ratio selects by its mass (reviews 2026-10-16:
                # M1, and M4 of the R99 review)
                slope_factor += 0.5 * (float(intact_vertex_factor) - 1.0)
            t_params = MassLawParams(epsilon=params.epsilon, b_sd=params.b_sd * slope_factor)
        if pomeron:
            # the shrinkage law's vertex law: every slope raised by c = 2 alpha' L (2026-11-01), the same 1 uniform
            c_shrink = 2.0 * float(pomeron_slope) * float(pomeron_gap_log(m2_excitation, remnant.mass * remnant.mass))
            transfer = draw_t_pomeron(t_lo, t_hi, float(t_params.b_sd), c_shrink, pomeron_t_break, rng_exchange_kinematics,
                                      t_lin=pomeron_linear_t)
        elif t_break is not None and remnant.pdg_id == PROTON_PDG_ID:
            # the intact proton's two-slope transfer (2026-10-24): exp(B t) above
            # -t_b and the slope rho B below it, one uniform as draw_t_window; the
            # reweighting selects the same events by the remnant mass
            if slope_mass_exponent is None:
                raise ValueError("generate_sd: the transfer break needs the factorized vertex slopes")
            transfer = draw_t_window_break(t_lo, t_hi, t_params, t_break, rng_exchange_kinematics)
        else:
            transfer = draw_t_window(t_lo, t_hi, t_params, rng_exchange_kinematics)
    else:
        if t_break is not None:
            raise ValueError("generate_sd: the transfer break needs the exact transfer window")
        if slope_mass_exponent is not None:
            raise ValueError("generate_sd: the factorized vertex slopes need the exact transfer window")
        t_min = t_range(m2_excitation, remnant.mass, SQRT_S)[0]
        transfer = draw_t(t_min, params, rng_exchange_kinematics)
    mass = math.sqrt(m2_excitation)

    # (3) the exact two-body solve (no draws); the cascade runs in the
    # excitation's center-of-mass frame (the module docstring's frame
    # convention) and the assembly boosts once into the lab frame.
    _e_beam, _e_total, sq_exact, _p_star, _beta, _gamma = _beam_configuration()
    s = sq_exact * sq_exact
    mp2 = PROTON_MASS * PROTON_MASS
    m2 = mass * mass
    mr2 = remnant.mass * remnant.mass
    # the incoming proton: its exact CM energy and momentum under the exact
    # transfer window (the committed solve uses the (m_p, m_remnant) values)
    e_p_cm = 0.5 * sq_exact if exact_range else (s + mp2 - mr2) / (2.0 * sq_exact)
    e_m_cm = (s + m2 - mr2) / (2.0 * sq_exact)
    e_rem_cm = (s + mr2 - m2) / (2.0 * sq_exact)
    p_star = (math.sqrt(max(0.0, 0.25 * s - mp2)) if exact_range else
              math.sqrt(max(0.0, (s - (PROTON_MASS + remnant.mass) ** 2)
                            * (s - (PROTON_MASS - remnant.mass) ** 2))) / (2.0 * sq_exact))
    q_star = math.sqrt(max(0.0, (s - (mass + remnant.mass) ** 2)
                           * (s - (mass - remnant.mass) ** 2))) / (2.0 * sq_exact)
    cos_theta = (e_p_cm * e_m_cm - (mp2 + m2) / 2.0
                 + transfer / 2.0) / (p_star * q_star)
    if exact_range and abs(cos_theta) > 1.0 + 1e-9:
        raise ValueError(f"generate_sd: the exact transfer window gives cos(theta*) = {cos_theta!r}")
    cos_theta = max(-1.0, min(1.0, cos_theta))
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    excitation_cm = (e_m_cm, q_star * sin_theta, 0.0, q_star * cos_theta)
    remnant_cm = (e_rem_cm, -q_star * sin_theta, 0.0, -q_star * cos_theta)

    # (4) the excitation decay through the frozen simplified cascade
    split_qn = None
    if two_sided is not None:
        # the two-sided hadronization of a heavy system (2026-10-26): (m_cut, eps_line, b, scope), the scope's
        # letters naming the one-pion exchange (o), the tail (t) and the coherence region (c)
        m_cut, eps_line, b_split, scope = two_sided
        if mass > float(m_cut):
            if is_ope and "o" in scope:
                split_qn = (0, 1, 0) if neutral_isospin is None else (0, 0, 0)
            elif not isovector and is_tail and "t" in scope:
                split_qn = (0, 0, 0)
            elif not isovector and not is_tail and "c" in scope:
                split_qn = (0, 0, 0)
    split_finals = None
    if split_qn is not None:
        split_finals = _decay_two_sided(
            excitation_cm, mass, catalog, fams, rng_decay_channel, rng_decay_kinematics, flavor, ledger, split_qn,
            float(eps_line), float(b_split), diffractive_duality, duality_line_shapes)
    if split_finals is not None:
        excitation_finals_cm = split_finals
    elif isovector:
        excitation_finals_cm = _decay_excitation(
            excitation_cm, mass, catalog, fams,
            rng_decay_channel, rng_decay_kinematics, flavor=flavor, ledger=ledger,
            line_qn=excited_line.qn, line_isospin=excited_line.isospin)
    else:
        recs = (None if diffractive_duality is None or not mass < float(diffractive_duality)
                else isoscalar_exchange_records(catalog, duality_line_shapes))
        if recs is not None and any(r.lo < mass < r.hi for r in recs):
            # the low-mass excitation as an I = 1/2 record (2026-10-15): one uniform on
            # decay_channel picks the record, then the record's own draws
            excitation_finals_cm = _record_decay(excitation_cm, mass, recs, catalog, rng_decay_channel,
                                                 rng_decay_kinematics, "excitation")
        else:
            excitation_finals_cm = _decay_excitation(
                excitation_cm, mass, catalog, fams,
                rng_decay_channel, rng_decay_kinematics, flavor=flavor, ledger=ledger)

    # (5) the lab assembly (one exact forward boost per four-momentum)
    boost = _lab_boost_factory()
    remnant_lab = boost(remnant_cm)
    excitation_finals = [
        FinalHadron(
            pdg_id=h.pdg_id,
            four_momentum=boost(h.four_momentum),
            mass=h.mass,
            qn=h.qn,
            line=h.line,
        )
        for h in excitation_finals_cm
    ]
    if remnant.pdg_id is not None and int(remnant.pdg_id) not in catalog.stable_states():
        # a low-mass resonance remnant decays through the Round 2 kernel
        # after the excitation, on the same substreams
        remnant_finals = tuple(_assemble_daughter(
            catalog, int(remnant.pdg_id), remnant_lab, "remnant",
            rng_decay_channel, rng_decay_kinematics))
    else:
        remnant_finals = (FinalHadron(
            pdg_id=remnant.pdg_id,
            four_momentum=remnant_lab,
            mass=remnant.mass,
            qn=remnant.qn,
            line="remnant",
        ),)
    final_state = remnant_finals + tuple(excitation_finals)

    # (5b) the target side: the CM parity mirror of the projectile-side
    # construction (plan line 826, "mirrored class"), so the excited
    # target system sits in the backward hemisphere and reaches the
    # forward one only by feed-over.  Every final-state four-momentum
    # maps through the lab image of the mirror, the line states through
    # the CM flip; the draws are those of the projectile construction.
    if side == "target":
        mirror = _cm_mirror_factory()
        excitation_finals = [
            FinalHadron(pdg_id=h.pdg_id, four_momentum=mirror(h.four_momentum),
                        mass=h.mass, qn=h.qn, line=h.line)
            for h in excitation_finals
        ]
        final_state = tuple(
            FinalHadron(pdg_id=h.pdg_id, four_momentum=mirror(h.four_momentum),
                        mass=h.mass, qn=h.qn, line=h.line)
            for h in remnant_finals) + tuple(excitation_finals)
        excitation_cm = _mirror_four_momentum(excitation_cm)
        remnant_cm = _mirror_four_momentum(remnant_cm)

    # (6) the ledger lines: the outgoing pair (projectile side first).
    lines = _outgoing_lines(side, excitation_cm, remnant_cm, boost,
                            exchange if isovector else None)

    # (7) the gap size
    y_remnant = _remnant_rapidity(final_state)
    gap = min(abs(_rapidity(h.four_momentum) - y_remnant)
              for h in excitation_finals)

    class_label = (SD_CLASS_LABEL_PROJECTILE if side == "projectile"
                   else SD_CLASS_LABEL_TARGET)
    return SDEvent(
        class_label=class_label,
        side=side,
        remnant=remnant,
        excitation_mass=mass,
        t=float(transfer),
        lines=lines,
        exchange=(exchange.records()[0] if isovector else (VACUUM_EXCHANGE_NAME, ZERO_TRANSFER)),
        final_state=final_state,
        gap_size=float(gap),
        exchange_configuration=exchange,
        excitation_m_min=m_edge,
        slope_factor=slope_factor,
        tail=is_tail,
        ope=is_ope,
        pomeron=(bool(pomeron) if pomeron_slope is not None else None),
    )


def _decay_two_sided(excitation_cm, mass, catalog, fams, rng_channel, rng_kinematics, flavor, ledger, mesonic_qn,
                     eps_line, b, duality_mass, duality_line_shapes):
    """The two-sided hadronization of a heavy excitation (2026-10-26;
    :mod:`na49_mixing.two_sided_hadronization`): a baryonic line (1, 1, 0)
    along the X flight direction and the exchanged object's line
    *mesonic_qn* opposite, each from its root family's lowest direct
    threshold, then each decayed through the cascade (the baryonic line
    below the diffractive duality mass as an I = 1/2 record, as the vacuum
    excitation).  ``None`` when the mass lies below the two thresholds (the
    caller then decays the single line)."""
    from .cascade_decay import _family_by_qn, _lowest_direct_threshold

    fam1, fam2 = _family_by_qn(fams, (1, 1, 0)), _family_by_qn(fams, tuple(mesonic_qn))
    if fam1 is None or fam2 is None:
        raise ValueError(f"generate_sd: the two-sided hadronization needs the continuum families (1, 1, 0) and {mesonic_qn}")
    lo1, lo2 = float(_lowest_direct_threshold(fam1, catalog)), float(_lowest_direct_threshold(fam2, catalog))
    if not float(mass) > lo1 + lo2:
        return None
    m1, m2 = split_masses(float(mass), lo1, lo2, eps_line, rng_kinematics)
    p1, p2 = split_kinematics(excitation_cm, float(mass), m1, m2, b, rng_kinematics)
    finals = []
    recs = (None if duality_mass is None or not m1 < float(duality_mass)
            else isoscalar_exchange_records(catalog, duality_line_shapes))
    if recs is not None and any(r.lo < m1 < r.hi for r in recs):
        finals.extend(_record_decay(p1, m1, recs, catalog, rng_channel, rng_kinematics, "excitation"))
    else:
        finals.extend(_decay_excitation(p1, m1, catalog, fams, rng_channel, rng_kinematics, flavor=flavor, ledger=ledger,
                                        line_qn=(1, 1, 0)))
    finals.extend(_decay_excitation(p2, m2, catalog, fams, rng_channel, rng_kinematics, flavor=flavor, ledger=ledger,
                                    line_qn=tuple(mesonic_qn)))
    return finals


def _ope_neutral_fraction(ope_vals, law):
    """The neutral exchange's part of the whole pion exchange (2026-10-27) on the
    component's window at the ground-state remnants: the masses from the p pi
    threshold (or xi_lo s) to xi_hi s, the neutron and the proton remnant."""
    s = SQRT_S * SQRT_S
    m2_lo = max(float(ope_vals[1]) * s, (PROTON_MASS + PION_MASS) ** 2)
    return ope_neutral_fraction(m2_lo, float(ope_vals[2]) * s, _NEUTRON_MASS * _NEUTRON_MASS, PROTON_MASS * PROTON_MASS,
                                s, law)


#: The neutron mass [GeV] of the charged exchange's ground-state remnant (PDG 2024).
_NEUTRON_MASS: float = 0.93956542


def _ope_configuration(side, neutral_isospin=None):
    """The pi+ exchange of the one-pion-exchange component (2026-10-26): the excited
    side's line at Q = 2, I = 3/2 and the other line a neutron (Q = 0, I = 1/2).
    With *neutral_isospin* the pi0 exchange (2026-10-27): the excited side's pi0 p
    line at Q = 1 and that isospin, the other line a proton."""
    if neutral_isospin is not None:
        excited = LineCharge(1, neutral_isospin)
        return ExchangeConfiguration(ISOVECTOR_EXCHANGE_NAME, 0, *((excited, PROTON_LINE) if side == "projectile"
                                                                   else (PROTON_LINE, excited)))
    if side == "projectile":
        return ExchangeConfiguration(ISOVECTOR_EXCHANGE_NAME, 1, LineCharge(2, 1.5), LineCharge(0, 0.5))
    return ExchangeConfiguration(ISOVECTOR_EXCHANGE_NAME, -1, LineCharge(0, 0.5), LineCharge(2, 1.5))


def _outgoing_lines(side, excitation_cm, remnant_cm, boost, exchange=None):
    """The two outgoing ledger line states (projectile side first): the
    excited side's line carries the excitation four-momentum, the other
    side's line the remnant four-momentum; both inherit the proton
    quantum numbers ``(1, 1, 0, 1)`` after the vacuum exchange's zero
    transfer, or the configured lines of an isovector *exchange*."""

    if side == "projectile":
        projectile_cm, target_cm = excitation_cm, remnant_cm
    else:
        projectile_cm, target_cm = remnant_cm, excitation_cm
    if exchange is not None:
        return exchange.line_states(boost(projectile_cm), boost(target_cm))
    return (
        LineState(B=1, Q=1, S=0, I3x2=1,
                  four_momentum=boost(projectile_cm)),
        LineState(B=1, Q=1, S=0, I3x2=1,
                  four_momentum=boost(target_cm)),
    )


def _never_mass_sampler(m_low: float, m_high: float) -> float:
    """The remnant selection's mass sampler must never be called on this
    card: the nominal split draws only the intact and low-mass kinds,
    which take their masses without the caller's law (the high-mass
    weight is the pre-registered 0.0)."""

    raise AssertionError(
        "generate_sd: the nominal single-diffraction split never draws "
        f"the high-mass continuum kind, got ({m_low!r}, {m_high!r})")


# ----------------------------------------------------------------------
# The beam-reverse mirror (plan line 826, "mirrored class")
# ----------------------------------------------------------------------
def _mirror_four_momentum(
    four_momentum: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """The longitudinal mirror of a four-momentum: ``(E, px, py, -pz)``."""

    e, px, py, pz = (float(x) for x in four_momentum)
    return (e, px, py, -pz)


def mirror_sd(event: SDEvent) -> SDEvent:
    """The beam-reversed counterpart of an event (and vice versa): every
    lab four-momentum mirrored in longitudinal momentum, identical quantum
    numbers, masses, and momentum transfer.  This is the reflection of the
    whole fixed-target configuration (the beam then runs along -z); the
    generated target class is instead the center-of-mass mirror with the
    beam kept along +z (:func:`generate_sd`, :func:`_cm_mirror_factory`).

    The side and class labels swap; the remnant selection, the
    excitation mass, and the transfer are unchanged; the gap size is the
    same rapidity interval (the mirror maps every rapidity to its
    negation).  The mirrored event's final state sums to the mirrored
    incoming four-momentum ``(E_total, 0, 0, -p_beam)``: the same
    conservation and ledger checks apply read-only.
    """

    if not isinstance(event, SDEvent):
        raise ValueError(
            f"mirror_sd: expected an SDEvent, got {type(event)!r}")
    if event.side not in SDSIDES:
        raise ValueError(
            f"mirror_sd: the event side {event.side!r} is outside the "
            f"closed side set {sorted(SDSIDES)}")
    mirrored_side = ("target" if event.side == "projectile"
                     else "projectile")
    mirrored_label = (SD_CLASS_LABEL_TARGET
                      if event.side == "projectile"
                      else SD_CLASS_LABEL_PROJECTILE)
    mirrored_lines = []
    for line in event.lines:
        if line.four_momentum is None:
            raise ValueError(
                "mirror_sd: the event's ledger line carries no "
                "four-momentum (the single-diffraction records always "
                "carry one; checked, never repaired)")
        mirrored_lines.append(LineState(
            B=line.B, Q=line.Q, S=line.S, I3x2=line.I3x2,
            four_momentum=_mirror_four_momentum(line.four_momentum)))
    lines = (mirrored_lines[0], mirrored_lines[1])
    final_state = tuple(
        FinalHadron(
            pdg_id=h.pdg_id,
            four_momentum=_mirror_four_momentum(h.four_momentum),
            mass=h.mass,
            qn=h.qn,
            line=h.line,
        )
        for h in event.final_state
    )
    return SDEvent(
        class_label=mirrored_label,
        side=mirrored_side,
        remnant=event.remnant,
        excitation_mass=float(event.excitation_mass),
        t=float(event.t),
        lines=lines,
        exchange=event.exchange,
        final_state=final_state,
        gap_size=float(event.gap_size),
        exchange_configuration=event.exchange_configuration,
        # the fields the reweighting and the observables read (the code review glm53f F7: the mirror dropped them)
        excitation_m_min=event.excitation_m_min,
        slope_factor=event.slope_factor,
        tail=event.tail,
        ope=event.ope,
        pomeron=event.pomeron,
    )
