"""The double-diffraction class (Round 5, card 6).

``TASK-20260925-r5-double-diffraction``; plan section 22 task 30 (line
1755); plan section 11.1 item 4 (line 827).  The card implements the
double-diffraction event class of the initial topology basis:

> 4. **Double diffraction:** both baryonic lines excited with a
>    central rapidity gap structure.

The pre-registered instantiation (the card): BOTH baryonic lines are
excited; there is no intact remnant and no separate remnant kind draw
(plan line 827 names the excited lines only).  Each line carries a
baryonic system ``(B, Q, S) = (1, 1, 0)`` with mass ``M_i`` from card
4's separate joint law for double excitation: ``M_1^2`` and ``M_2^2``
from the marginals on the exact limits ``[m_p^2, (sqrt(s) - m_p)^2]``
(:func:`na49_mixing.topology_mass_laws.sd_limits`) and ``t_1``, ``t_2``
from ``exp(B_SD t)`` on the exact two-body ranges
(:func:`na49_mixing.topology_mass_laws.t_range`, each computed with the
other line at the intact proton mass), accepted on the exact closure
``M_1 + M_2 <= sqrt_s`` by rejection with the pinned card-4 rejection
cap (:data:`na49_mixing.topology_mass_laws.REJECTION_CAP`;
:class:`na49_mixing.topology_mass_laws.PhaseSpaceRejectionExceeded` is
the only non-conservation exit).  The outgoing four-momenta are solved
from ``(M_1, t_1, M_2, t_2, sqrt_s)`` by exact kinematics: the two
systems share the total four-momentum (plan section 11.1 item 4; the
central system is quantum-number-free per plan section 11.4, so the
class has no central source at all -- the record carries no central
line and rule (e) of the ledger audit never fires).

Exchange and ledger conventions (plan section 11.4)
---------------------------------------------------

Each of the two color-singlet exchanges is a vacuum exchange carrying
the declared ``(B, Q, S, 2I3)`` transfer ``(0, 0, 0, 0)`` (plan line
877); both outgoing lines inherit their incoming line quantum numbers
``(1, 1, 0, 1)`` after the zero transfers
(:func:`na49_mixing.exchange_ledger.apply_transfer_qn`, the pinned
card-2 arithmetic).  The event record carries the two incoming proton
lines, the two outgoing line states (projectile side first, target
side second), and the two exchange records; card 2's
:func:`na49_mixing.exchange_ledger.audit_event` consumes these records
read-only with zero violations: rule (a) closed-set membership, rule
(b) the vacuum zero transfers, rule (c) the inheritance after the
declared transfers, rule (d) the local decay closure of both
excitations (each decay vertex's daughter ``(B, Q, S, 2I3)`` sum
equals the parent exactly), and rule (e) vacuously (the class produces
no central source: the central production of plan section 11.3 is the
separate non-diffractive class of plan line 828, and the plan section
11.1 item 4 instantiation carries no separate remnant kind draw).

Exact kinematics
----------------

:func:`dd_kinematics` solves the outgoing four-momenta of the two
excited systems from ``(M_1, t_1, M_2, t_2, sqrt_s)`` as the exact
two-body scattering ``p + p -> X_1(M_1) + X_2(M_2)`` in the
center-of-mass frame of the pinned beam configuration (beam
:data:`~na49_mixing.topology_elastic.BEAM_MOMENTUM = 158.0 GeV/c`,
proton mass :data:`~na49_mixing.topology_elastic.PROTON_MASS =
0.93827231 GeV`, boosted into the lab frame with the exact
fixed-target CM velocity).  The two systems share the total
four-momentum: in the center-of-mass frame of the incoming pair the
outgoing momenta are back-to-back with magnitude ``q*`` from the exact
two-body kinematics of ``(M_1, M_2)`` at the invariant mass
``sqrt_s``.

One exact fact of the 2-to-2 system bounds the realized transfers and
is recorded here: the two systems share the total four-momentum, so
momentum conservation makes the two exchanges carry the same transfer
four-vector, ``p_1 - p_3 = p_4 - p_2``, and the exact coupling of the
realized transfers is ``t_1 = t_2`` (the back-to-back configuration;
the two invariants coincide identically).  Card 4's joint law draws
``t_1`` and ``t_2`` independently (the factorized effective law; the
exact ``t_1 = t_2`` coupling of the 2-to-2 system is not claimed by
the draw, the identifiable-effective-combinations rule, plan line
894), so the drawn pair over-determines the single production angle:
the two drawn values generally disagree with each other and with the
exact equality the realized configuration must satisfy.  The operative
solve convention resolves the over-determination on the
projectile-side exchange: the angle follows from the drawn ``t_1``
through the documented invariant relation on the first baryonic line,
``t = m_p^2 + M_1^2 - 2 (E_p* E_1* - p* q* cos(theta*))`` -- the same
invariant the committed single-diffraction solve uses on its excitation
line -- and the realized target-side transfer equals the realized
beam-side transfer (the exact ``t_1 = t_2`` coupling with the angle
the drawn ``t_1`` fixes).  The recorded ``t_1`` and ``t_2`` stay the
drawn values (the joint law's exchange-kinematics record); the solved
configuration carries the exact energy-momentum closure.

The drawn domains and the exact two-body reach: the drawn ``t_i``
range is card 4's ``[t_min(M_i, m_p), 0]`` with the other line at the
intact proton mass, while the exact ``t_i`` reach of the realized
``(M_1, M_2)`` system is ``[m_p^2 + M_i^2 - 2 (E_p* E_i* + p* q*),
m_p^2 + M_i^2 - 2 (E_p* E_1* - p* q*)]``; the drawn ``t_i`` can land
above the exact forward edge (a forward sliver, the committed
single-diffraction closed-at-0 fact) and below the exact backward edge
(the heavier recoil partner of the realized system shifts the backward
edge), so ``cos(theta*)`` is clamped into ``[-1, 1]`` and the angle
saturates at the forward or backward configuration on either sliver.
The event's recorded transfers stay the drawn values (the
exchange-kinematics variables); the solved four-momenta are the exact
physical configuration at the clamped angle.

Frame convention (the single-diffraction card's convention): both
excitations decay through the frozen simplified cascade with the Round
4 modules consumed read-only, and the cascade's per-vertex conservation
checks run in the overall center-of-mass frame of the pinned beam
configuration -- the frame in which the frozen ``draw_decay`` machinery
keeps its documented closure precision at the pinned beam boost
(``gamma ~ 30``; boosting the near-threshold vertices through the
lab-frame parent velocity reconstructs the invariant through
``1 - beta^2`` and amplifies the component rounding beyond the absolute
tolerance).  The collected final states of both systems are boosted
into the lab frame once, at assembly, with the exact forward boost; the
event's four-momentum closure against the incoming sum is at the
float-rounding level.

The central rapidity gap
------------------------

The central gap is the R50 stored quantity (plan line 903): the
rapidity interval between the nearest final-state hadron of the
projectile-side system and the nearest final-state hadron of the
target-side system (:func:`central_gap`; the interval is the
``|y_projectile - y_target|`` pair distance minimized over the two
sides' final-state hadrons).  It feeds the R51 gap-fraction observable
(index item 3: no hadronic final-state particle in the central window
``|y| < 0.5``); the class record carries every final-state hadron, so
the runner computes the fraction read-only.  The gap is positive on the
pinned trace (the two systems' decay products do not close the central
gap at nominal parameters within ``1e-12``; the acceptance (iii)
pre-registration).

Draw budget (the card's pre-registered counts)
----------------------------------------------

Exactly six uniforms per accepted event on the closed
``exchange_kinematics`` substream (card 4's joint law: two marginals
and two transfers, the cell and within-cell convention of the M^2
draws, plus the rejection attempts on the same stream -- four uniforms
per rejected attempt, the two ``M^2`` draws), the frozen Round 4 budget
of one ``decay_channel`` plus two ``decay_kinematics`` uniforms per
decay vertex on ``decay_channel`` / ``decay_kinematics`` (both
excitation decays through :func:`na49_mixing.cascade_decay.draw_decay`;
a catalog handoff continues the same two substreams into the
byte-unchanged Round 2 kernel's own documented budget, the card-9
handoff convention), zero ``spectral_state`` uniforms (both lines are
excited; the record carries no remnant and no remnant kind draw), and
no ``topology_selection`` uniform (the class draw belongs to the
runner, card 9: :func:`generate_dd` receives no class stream and draws
no class uniform).

Hermetic: standard library and numpy only; cards 1-4's modules, the
committed single-diffraction module (card 5, consumed read-only for
the pinned beam configuration and the final-state record helpers), and
the Round 4 cascade modules are consumed read-only; no event generation
driver, no external tables, no Monte Carlo run (the acceptance is
hermetic: pinned traces, exact kinematics, draw budgets; the runs
R50-R55 exercise the class statistically).
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
from .isovector_exchange import ExchangeConfiguration, LineCharge, line_mass_threshold, line_root_family
from .nd_duality import isoscalar_exchange_records
from .resonance_decay import ResonanceDecayer
from .rng_substreams import SUBSTREAM_LABELS
from .spectral_density import ContinuumFamily
from .topology_elastic import BEAM_MOMENTUM, PROTON_MASS, SQRT_S
from .topology_mass_laws import (
    MassLawParams,
    draw_joint_dd,
    draw_joint_dd_exact,
    sd_limits,
    vertex_slope_factor,
)
from .topology_single_diffraction import (
    FinalHadron,
    _assemble_daughter,
    _beam_configuration,
    _final_from_catalog,
    _kernel_finals,
    _lab_boost_factory,
    _rapidity,
    _record_decay,
)

__all__ = [
    "DD_CLASS_LABEL",
    "DD_LINE_LABELS",
    "PROJECTILE_LINE_LABEL",
    "TARGET_LINE_LABEL",
    "NOMINAL_PARAMS",
    "BEAM_MOMENTUM",
    "PROTON_MASS",
    "SQRT_S",
    "FinalHadron",
    "DDEvent",
    "central_gap",
    "dd_kinematics",
    "generate_dd",
]


#: The class label of the double-diffraction class (the inelastic-sample
#: class label of the R50 templates; plan section 11.1 item 4, line 827).
DD_CLASS_LABEL: str = "dd"

#: The closed line-label set of the final-state hadrons: the two excited
#: baryonic systems of plan line 827 (the projectile side first, the
#: target side second; the beam direction carries the projectile side).
PROJECTILE_LINE_LABEL: str = "projectile"
TARGET_LINE_LABEL: str = "target"
DD_LINE_LABELS: frozenset[str] = frozenset(
    {PROJECTILE_LINE_LABEL, TARGET_LINE_LABEL})

#: The nominal effective mass-law parameters (the R50 nominal values,
#: index item 2: epsilon = 0, B_SD the M8-frozen effective transfer
#: slope; fixed here, fit in R51 by card 11).
NOMINAL_PARAMS: MassLawParams = MassLawParams(0.0, 8.504140762083612)

#: The proton quantum numbers in the doubled-I3 integer convention
#: ``(B, Q, S, 2I3)`` (the committed catalog record pdg_id 2212; both
#: outgoing lines inherit it through the vacuum exchanges).
PROTON_QN: tuple[int, int, int, int] = (1, 1, 0, 1)

#: The vacuum exchange name (each of the class's two color-singlet
#: exchanges; the transfer is ``(0, 0, 0, 0)``, plan line 877).
VACUUM_EXCHANGE_NAME: str = "vacuum"
ZERO_TRANSFER: tuple[int, int, int, int] = (0, 0, 0, 0)

#: The frozen simplified-cascade continuum family of both excited
#: baryonic systems (the activated baryonic ``(1, 1, 0)`` family of
#: ``config/v2_singlet/continuum_resonances.json``; the cascade line's
#: ``family_id``).
EXCITATION_FAMILY_ID: str = "baryonic_np"

#: The class label carried on the cascade lines' exchange ancestry (the
#: exchange-layer lineage of the Round 4 cascade states).
ANCESTRY_LABEL: str = "dd"


@dataclass(frozen=True)
class DDEvent:
    """The frozen double-diffraction event record (the card's
    ledger-ready record).

    ``class_label`` is ``"dd"``.  ``m1`` is the projectile-side
    excitation mass [GeV], ``m2`` the target-side excitation mass [GeV],
    ``t1`` and ``t2`` the drawn momentum transfers of the two exchanges
    [GeV^2].  ``lines`` is the card-2 ledger record pair of the two
    outgoing lines (projectile side first, target side), both carrying
    ``(1, 1, 0, 1)`` after the vacuum transfers (plan line 827: both
    lines excited).  ``exchanges`` is the pair of the class's two
    vacuum exchange records in the card-5 pair-tuple convention
    ``(name, transfer)`` per exchange.  ``final_state`` carries every
    final-state hadron (both systems' decay products).  ``gap_size`` is
    the central rapidity gap of :func:`central_gap`.

    The record carries no remnant selection and no remnant kind draw
    (plan line 827 names the excited lines only): both lines are excited
    baryonic systems, there is no intact-proton remnant, and there is no
    separate remnant-kind draw (the white-box ``test_no_intact_remnant``
    contract).

    ``exchange_configuration`` (Round 8, 2026-10-04) is the drawn
    :class:`~na49_mixing.isovector_exchange.ExchangeConfiguration` of a
    run with a positive isovector fraction (``None`` otherwise); under the
    isovector exchange ``lines`` and ``exchanges`` carry the configured
    line charges and the opposite transfers.
    ``line_m_min`` (2026-10-06) is the pair of lower mass edges [GeV]
    (projectile side first) under ``line_thresholds``, each the line root
    family's lowest direct threshold (``None`` for the shared edge).
    ``slope_factor`` (2026-10-08) is the factorized vertex slope factor
    ``B / b`` of the transfer under ``slope_mass_exponent``
    (:func:`~na49_mixing.topology_mass_laws.vertex_slope_factor`; ``None``
    for the single slope).
    """

    class_label: str
    m1: float
    m2: float
    t1: float
    t2: float
    lines: tuple[LineState, LineState]
    exchanges: tuple[tuple[str, tuple[int, int, int, int]],
                     tuple[str, tuple[int, int, int, int]]]
    final_state: tuple[FinalHadron, ...]
    gap_size: float
    exchange_configuration: ExchangeConfiguration | None = None
    line_m_min: tuple[float, float] | None = None
    slope_factor: float | None = None
    pomeron: bool | None = None


# ----------------------------------------------------------------------
# The pinned beam configuration and the ledger lines
# ----------------------------------------------------------------------
def _incoming_lines() -> tuple[LineState, LineState]:
    """The pinned incoming ledger lines (projectile side first): the
    beam proton and the target proton, with the doubled-I3 proton
    quantum numbers and their lab four-momenta (card 5's pinned
    incoming lines, consumed read-only)."""

    e_beam, _e_total, _sq, _ps, _beta, _gamma = _beam_configuration()
    return (
        LineState(B=1, Q=1, S=0, I3x2=1,
                  four_momentum=(e_beam, 0.0, 0.0, BEAM_MOMENTUM)),
        LineState(B=1, Q=1, S=0, I3x2=1,
                  four_momentum=(PROTON_MASS, 0.0, 0.0, 0.0)),
    )


# ----------------------------------------------------------------------
# Exact kinematics (the card's dd_kinematics)
# ----------------------------------------------------------------------
def _cm_pair(m1: float, t1: float, m2: float, t2: float, sqrt_s: float):
    """The exact center-of-mass four-momenta of the two excited systems
    from ``(M_1, t_1, M_2, t_2, sqrt_s)``: the back-to-back pair of the
    exact two-body configuration ``p + p -> X_1(M_1) + X_2(M_2)``,
    projectile side first.

    The exact two-body quantities of the pinned beam configuration (the
    caller passes the recorded :data:`~na49_mixing.topology_elastic.
    SQRT_S` for the draw-domain bookkeeping; the solve uses the exact
    invariant mass of the pinned configuration so the lab closure
    holds), the scattering angle from the drawn projectile-side
    transfer through the invariant relation
    ``t_1 = m_p^2 + M_1^2 - 2 (E_p* E_1* - p* q* cos(theta*))``
    (the documented convention; see the module docstring for the
    over-determination resolution and the drawn ``t_2``'s operative
    role), and the at-rest resolution when the drawn pair exhausts the
    two-body phase space within the joint-closure tolerance.
    """

    mass1 = float(m1)
    mass2 = float(m2)
    transfer1 = float(t1)
    transfer2 = float(t2)
    root_s = float(sqrt_s)
    if not (math.isfinite(mass1) and mass1 > 0.0):
        raise ValueError(
            "dd_kinematics: the projectile-side excitation mass must be "
            f"a positive finite GeV value, got {m1!r}")
    if not (math.isfinite(mass2) and mass2 > 0.0):
        raise ValueError(
            "dd_kinematics: the target-side excitation mass must be a "
            f"positive finite GeV value, got {m2!r}")
    for name, value in (("t1", transfer1), ("t2", transfer2)):
        if not (math.isfinite(value) and value <= 0.0):
            raise ValueError(
                f"dd_kinematics: the momentum transfer {name} must be a "
                f"non-positive finite GeV^2 value, got {value!r}")
    if not math.isfinite(root_s):
        raise ValueError(
            f"dd_kinematics: sqrt_s must be finite, got {sqrt_s!r}")
    # The drawn-domain check: the joint closure accepts
    # M_1 + M_2 <= sqrt_s + 1e-12 (card 4's tolerance); a pair above the
    # threshold beyond the tolerance is outside the drawn domain
    # (checked, never repaired).
    if root_s + 1e-12 < mass1 + mass2:
        raise ValueError(
            "dd_kinematics: the two-body system is closed: "
            f"M_1 + M_2 = {mass1 + mass2!r} GeV exceeds "
            f"sqrt_s = {root_s!r} GeV beyond the joint-closure tolerance "
            "1e-12")

    _e_beam, _e_total, sq_exact, _p_star, beta, gamma = _beam_configuration()
    s = sq_exact * sq_exact
    mp2 = PROTON_MASS * PROTON_MASS
    m2_1 = mass1 * mass1
    m2_2 = mass2 * mass2

    # The two-body CM energies and momenta of the realized (M_1, M_2)
    # system; at the drawn-closure sliver (M_1 + M_2 marginally above
    # sqrt_s within the tolerance) the momenta clamp to the at-rest
    # configuration (q* = 0; checked, never repaired).
    e_p_cm = s / (2.0 * sq_exact)
    p_star = math.sqrt(max(0.0, s / 4.0 - mp2))
    e_1_cm = (s + m2_1 - m2_2) / (2.0 * sq_exact)
    e_2_cm = (s + m2_2 - m2_1) / (2.0 * sq_exact)
    q_star = math.sqrt(max(0.0, (s - (mass1 + mass2) ** 2)
                           * (s - (mass1 - mass2) ** 2))) / (2.0 * sq_exact)
    if q_star <= 0.0:
        return ((e_1_cm, 0.0, 0.0, 0.0), (e_2_cm, 0.0, 0.0, 0.0))

    # The scattering angle from the drawn projectile-side transfer on
    # the excitation line, t_1 = m_p^2 + M_1^2
    # - 2 (E_p* E_1* - p* q* cos(theta*)); the drawn domains extend
    # past the exact forward and backward edges (the factorized joint
    # law's ranges are computed with the other line at the intact
    # proton mass), so the angle saturates at the clamped configuration
    # and the recorded transfers stay the drawn values.
    cos_theta = (transfer1 - mp2 - m2_1 + 2.0 * e_p_cm * e_1_cm) / (
        2.0 * p_star * q_star)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))

    system1_cm = (e_1_cm, q_star * sin_theta, 0.0, q_star * cos_theta)
    system2_cm = (e_2_cm, -q_star * sin_theta, 0.0, -q_star * cos_theta)
    return system1_cm, system2_cm


def dd_kinematics(
    m1: float,
    t1: float,
    m2: float,
    t2: float,
    sqrt_s: float,
) -> tuple[tuple[float, float, float, float],
           tuple[float, float, float, float]]:
    """Exact outgoing four-momenta ``(system_1, system_2)`` in the lab
    frame from ``(M_1, t_1, M_2, t_2, sqrt_s)``, projectile side first.

    The exact two-body scattering ``p + p -> X_1(M_1) + X_2(M_2)`` of
    the pinned beam configuration (the two systems share the total
    four-momentum): the center-of-mass energies and momenta from the
    invariant mass *sqrt_s* (the caller passes the recorded
    :data:`~na49_mixing.topology_elastic.SQRT_S` for the draw-domain
    bookkeeping; the solve uses the exact invariant mass of the pinned
    configuration so the lab closure holds), the scattering angle from
    the drawn projectile-side transfer through the invariant relation

        ``t_1 = m_p^2 + M_1^2 - 2 (E_p* E_1* - p* q* cos(theta*))``,

    and the lab boost with the exact fixed-target CM velocity.  The
    production plane is the x-z plane.  Energy-momentum closure holds
    within ``1e-9`` relative by construction (the CM pair sums to the
    total CM four-momentum and the boost maps it to the incoming lab
    sum).

    The drawn pair ``(t_1, t_2)`` is the joint law's exchange record
    (card 4's factorized effective law; the exact ``t_1 = t_2`` coupling
    of the 2-to-2 system is not claimed by the draw, plan line 894).
    The operative solve realizes the projectile-side transfer through
    the documented invariant; the realized target-side transfer equals
    the realized beam-side transfer (the exact two-body coupling of
    the back-to-back configuration on the shared total four-momentum),
    and the drawn ``t_2`` stays the recorded exchange-kinematics
    variable.  The ``t_2`` argument is
    consumed by the interface and the validation (the record's drawn
    state); the solved configuration is fixed by ``(M_1, t_1, M_2,
    sqrt_s)``.  The angle saturates at the clamped configuration on the
    drawn slivers beyond the exact reachable ``t_1`` range (the forward
    and backward saturation documented in the module docstring), and a
    pair at the drawn-closure edge resolves at the exact at-rest
    configuration.  ``ValueError`` on non-positive masses, positive or
    non-finite transfers, or a two-body system closed beyond the
    joint-closure tolerance (checked, never repaired).
    """

    _transfer2 = float(t2)  # the drawn target-side transfer: validated
    if not (math.isfinite(_transfer2) and _transfer2 <= 0.0):
        raise ValueError(
            "dd_kinematics: the momentum transfer t2 must be a "
            f"non-positive finite GeV^2 value, got {t2!r}")
    system1_cm, system2_cm = _cm_pair(m1, t1, m2, t2, sqrt_s)
    boost = _lab_boost_factory()
    return boost(system1_cm), boost(system2_cm)


# ----------------------------------------------------------------------
# The central gap (the R50 stored quantity, plan line 903)
# ----------------------------------------------------------------------
def central_gap(event: DDEvent) -> float:
    """The R50 central-gap quantity of *event*: the rapidity interval
    between the nearest final-state hadron of the projectile-side system
    and the nearest final-state hadron of the target-side system.

    The interval is ``|y_projectile - y_target|`` minimized over the
    cross-side pairs of the event's final-state hadrons (the
    ``"projectile"`` and ``"target"`` line labels).  The function
    recomputes the interval from the event's records (read-only, zero
    random numbers) and raises ``ValueError`` on an event whose recorded
    gap disagrees beyond ``1e-12`` (checked, never repaired).
    """

    if not isinstance(event, DDEvent):
        raise ValueError(
            f"central_gap: expected a DDEvent, got {type(event)!r}")
    projectile_ys = [
        _rapidity(h.four_momentum) for h in event.final_state
        if h.line == PROJECTILE_LINE_LABEL]
    target_ys = [
        _rapidity(h.four_momentum) for h in event.final_state
        if h.line == TARGET_LINE_LABEL]
    if not projectile_ys or not target_ys:
        raise ValueError(
            "central_gap: the event must carry at least one projectile-"
            f"side and one target-side final-state hadron, got "
            f"{len(projectile_ys)} projectile and {len(target_ys)} "
            "target hadrons")
    gap = min(abs(y_p - y_t)
              for y_p in projectile_ys for y_t in target_ys)
    if abs(gap - float(event.gap_size)) > 1e-12:
        raise ValueError(
            "central_gap: the event's recorded gap "
            f"{float(event.gap_size)!r} disagrees with the recomputed "
            f"interval {gap!r} beyond 1e-12 (checked, never repaired)"
        )
    return gap


# ----------------------------------------------------------------------
# The excitation decays through the frozen simplified cascade
# ----------------------------------------------------------------------
def _final_from_catalog_dd(catalog: HadronCatalog, pdg_id: int,
                           four_momentum, line: str) -> FinalHadron:
    """The final-state hadron record of a catalog daughter (the card-5
    helper at the double-diffraction line label)."""

    return _final_from_catalog(catalog, pdg_id, four_momentum, line)


def _kernel_finals_dd(catalog: HadronCatalog, vertices, line: str):
    """The stable final-state hadrons of a completed Round 2 kernel
    vertex list (the card-5 assembly at the double-diffraction line
    label)."""

    return _kernel_finals(catalog, vertices, line)


def _decay_system(
    system_cm: tuple[float, float, float, float],
    mass: float,
    catalog: HadronCatalog,
    fams: list[ContinuumFamily],
    rng_channel,
    rng_kinematics,
    line: str,
    flavor=None,
    ledger=None,
    line_qn: tuple = (1, 1, 0),
    line_isospin: float | None = None,
) -> list[FinalHadron]:
    """One excited system's decay through the frozen simplified cascade
    (the Round 4 modules consumed read-only with isotropic decay angles):
    the card-9 termination rules with the caller's event assembly of
    the vertex siblings, in the overall center-of-mass frame of the
    pinned beam configuration (the frame convention of the module
    docstring).

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
    excitation is imposed): it stays a single baryonic final state at
    its mass with no vertex and zero decay draws.

    Every assembled final-state hadron carries the caller's line of
    origin (``"projectile"`` or ``"target"``) and its overall-center-of-
    mass four-momentum (the caller boosts to the lab frame once, at
    assembly).
    """

    if line not in DD_LINE_LABELS:
        raise ValueError(
            f"_decay_system: the line label {line!r} is outside the "
            f"closed double-diffraction line-label set "
            f"{sorted(DD_LINE_LABELS)}")
    finals: list[FinalHadron] = []
    root_cm = (float(system_cm[0]), float(system_cm[1]),
               float(system_cm[2]), float(system_cm[3]))
    # the root carries the line's (B, Q, S) and starts in the continuum
    # family of those quantum numbers (the general line model, 2026-10-04;
    # the vacuum exchange keeps every line at (1, 1, 0), baryonic_np)
    root_qn = tuple(int(v) for v in line_qn)
    if line_isospin is None:
        root_family = _family_by_qn(fams, root_qn)
    else:
        # a line of the isovector exchange: the family of its quantum
        # numbers and isospin (Round 8, 2026-10-04)
        root_family = line_root_family(fams, LineCharge(root_qn[1], line_isospin))
    if root_family is None:
        raise ValueError(f"_decay_system: no continuum family carries the line quantum numbers {root_qn}")
    current = CascadeState(
        momentum=root_cm,
        mass=float(mass),
        qn=root_qn,
        quantum={},
        family_id=str(root_family.family_id),
        axis=(0.0, 0.0, 1.0),
        alignment_label=ISOTROPIC_ALIGNMENT_LABEL,
        parent=None,
        exchange_ancestry=(ANCESTRY_LABEL,),
        depth=0,
    )
    while True:
        # (1) the hard maximum depth guard (plan section 10.6, line 757)
        if int(current.depth) > _cascade_depth_limit(fams):
            raise CascadeDepthGuardError(
                f"double-diffraction excitation cascade: the line state "
                f"of family {current.family_id!r} at mass "
                f"{float(current.mass)!r} GeV carries depth "
                f"{int(current.depth)}, exceeding MAX_CASCADE_DEPTH = "
                f"{_cascade_depth_limit(fams)}")
        # (2) the catalog handoff: the Round 2 kernel completes the
        # remainder (its own documented draw budget on the same
        # substreams; card 9's handoff convention).
        if is_catalog_handoff(current, catalog):
            vertices = handoff_decay(current, catalog, rng_channel,
                                     rng_kinematics)
            finals.extend(_kernel_finals_dd(catalog, vertices, line))
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
                    pdg = int(daughter.family_id[
                        len(CATALOG_FAMILY_PREFIX):])
                    finals.extend(_assemble_daughter(
                        catalog, pdg, daughter.momentum, line,
                        rng_channel, rng_kinematics))
                return finals
            successor = None
            for daughter in daughters:
                if daughter.family_id.startswith(CATALOG_FAMILY_PREFIX):
                    pdg = int(daughter.family_id[
                        len(CATALOG_FAMILY_PREFIX):])
                    if pdg in catalog.stable_states():
                        finals.append(_final_from_catalog_dd(
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
                        finals.extend(_kernel_finals_dd(
                            catalog,
                            kernel.decay_resonance(pdg, sibling_cm),
                            line))
                else:
                    successor = daughter
            if successor is None:
                raise ValueError(
                    "double-diffraction excitation cascade: the continuum "
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
                # a stable record is final; an unstable one decays through
                # the Round 2 kernel (the 2026-09-30 final-state fix)
                finals.extend(_assemble_daughter(
                    catalog, pdg, daughter.momentum, line,
                    rng_channel, rng_kinematics))
            return finals
        # (5) the below-threshold excitation: no open channel at all, so
        # the system stays a single baryonic final state at its mass
        # (no minimum N-pi excitation is imposed, plan line 842; zero
        # decay draws).
        # the line state itself (its four-momentum, mass and quantum
        # numbers; the root unless a continuation step came first)
        finals.append(FinalHadron(
            pdg_id=None,
            four_momentum=(float(current.momentum[0]), float(current.momentum[1]),
                           float(current.momentum[2]), float(current.momentum[3])),
            mass=float(current.mass),
            qn=(int(current.qn[0]), int(current.qn[1]), int(current.qn[2]),
                2 * int(current.qn[1]) - int(current.qn[0]) - int(current.qn[2])),
            line=line,
        ))
        return finals


# ----------------------------------------------------------------------
# The event generator
# ----------------------------------------------------------------------
def generate_dd(
    params: MassLawParams,
    catalog: HadronCatalog,
    fams: list[ContinuumFamily],
    rng_exchange_kinematics,
    rng_decay_channel,
    rng_decay_kinematics,
    flavor=None,
    ledger=None,
    exchange: ExchangeConfiguration | None = None,
    exact_range: bool = False,
    line_thresholds: bool = False,
    slope_mass_exponent: float | None = None,
    diffractive_duality: float | None = None,
    duality_line_shapes: bool = False,
    xi_max: float | None = None,
    pomeron_slope: float | None = None,
    pomeron_linear_t: float | None = None,
) -> DDEvent:
    """One double-diffraction event.

    The card's pre-registered draw budget on the closed substream labels
    (:data:`na49_mixing.rng_substreams.SUBSTREAM_LABELS`): exactly six
    uniforms on ``exchange_kinematics`` per accepted event (card 4's
    joint law through :func:`na49_mixing.topology_mass_laws.draw_joint_dd`
    on the exact limits ``[m_p^2, (sqrt(s) - m_p)^2]`` for both
    marginals, the exact two-body ``t`` ranges, and the exact closure
    ``M_1 + M_2 <= sqrt_s`` by rejection on the pinned card-4 cap; a
    rejected attempt consumes four uniforms on the same stream), and
    the frozen Round 4 budget of one ``decay_channel`` plus two
    ``decay_kinematics`` uniforms per decay vertex on
    ``decay_channel`` / ``decay_kinematics`` (both excitations through
    the frozen simplified cascade; a catalog handoff continues into the
    byte-unchanged Round 2 kernel's own documented budget on the same
    substreams).  Zero ``spectral_state`` uniforms (both lines are
    excited; no remnant kind draw) and zero ``topology_selection``
    uniforms (the class draw belongs to the runner, card 9).

    The kinematics solve and the lab assembly follow :func:`dd_kinematics`
    and the module docstring's frame convention: both cascades run in
    the overall center-of-mass frame (projectile side first, then the
    target side, on the shared decay substreams in that order) and the
    assembly boosts once into the lab frame.  The event's ``lines`` are
    the two outgoing ledger line states (projectile side first), both
    inheriting the proton quantum numbers ``(1, 1, 0, 1)`` after the
    vacuum exchanges' zero transfers; the ``exchanges`` records are the
    two vacuum exchanges with the zero transfer; the ``final_state``
    carries both systems' decay products in the lab frame; the
    ``gap_size`` is the central rapidity gap of :func:`central_gap`.

    ``ValueError`` on a parameter surface outside the card-4 mass-law
    parameters or an unknown catalog/family surface (checked, never
    repaired); :class:`na49_mixing.topology_mass_laws.
    PhaseSpaceRejectionExceeded` after the pinned card-4 rejection cap
    (the only exit on a pathological configuration).

    *exchange* (Round 8, 2026-10-04) is the event's drawn exchange
    configuration (the runner draws it at a positive isovector fraction).
    ``None`` or the vacuum configuration gives the event above.  Under the
    isovector exchange each excitation's cascade starts in its line's root
    family; the draw budget is unchanged.

    *exact_range* (2026-10-05) draws the one transfer of the 2 -> 2 final
    state on the reachable window of the accepted pair
    (:func:`~na49_mixing.topology_mass_laws.draw_joint_dd_exact`; ``t1 =
    t2``, 5 uniforms per accepted draw instead of 6).

    *pomeron_slope* (2026-11-01) alpha'_P [GeV^-2] draws the vacuum events
    from the Pomeron's shrinkage law (``draw_joint_dd_exact``'s
    *pomeron_slope*: the pair kept with the probability ``Z_{B+c} / Z_B`` on
    one more uniform, the transfer at the slope ``B + c``, ``c = 2 alpha'
    ln(s m_p^2 / (M_1^2 M_2^2))`` clipped at zero); the event records
    ``pomeron``.  Needs *exact_range*; ``None`` gives the committed event.
    *pomeron_linear_t* t_lin raises the slope down to ``t = -t_lin`` only.

    *line_thresholds* (2026-10-06; code-physics review M-6) starts each
    line's marginal at its own lowest direct threshold
    (:func:`~na49_mixing.isovector_exchange.line_mass_threshold`), recorded
    as ``line_m_min``; the joint law and the draw budget are otherwise
    unchanged.

    *diffractive_duality* (2026-10-15) is the mass below which a vacuum
    system decays as an I = 1/2 low-mass record whose range holds its mass
    (the single-diffraction excitation's rule), on one ``decay_channel``
    uniform; ``None`` gives the continuum cascade.
    """

    if not isinstance(params, MassLawParams):
        raise ValueError(
            f"generate_dd: expected MassLawParams, got {type(params)!r}")
    if not isinstance(catalog, HadronCatalog):
        raise ValueError(
            f"generate_dd: expected a HadronCatalog, got {type(catalog)!r}")
    for label in ("exchange_kinematics", "decay_channel",
                  "decay_kinematics"):
        if label not in SUBSTREAM_LABELS:
            raise ValueError(
                f"generate_dd: the substream label {label!r} is not in "
                f"the closed label set {SUBSTREAM_LABELS}")

    # (1) the card-4 joint law (6 uniforms per accepted draw on
    # exchange_kinematics, plus 4 uniforms per rejected attempt on the
    # same stream; the rejection cap is the only pathological exit)
    m2_min, m2_max = sd_limits(PROTON_MASS, SQRT_S)
    if xi_max is not None:
        # the coherence region (2026-10-23): each excited mass below M^2/s =
        # xi_max; at xi_max <= 0.25 the closure M_1 + M_2 <= sqrt(s) holds
        m2_max = min(m2_max, float(xi_max) * SQRT_S * SQRT_S)
    isovector = exchange is not None and not exchange.is_vacuum
    m2_min_1 = m2_min_2 = m2_min
    edges = None
    if line_thresholds:
        edges = (line_mass_threshold(fams, catalog, exchange.projectile if isovector else None),
                 line_mass_threshold(fams, catalog, exchange.target if isovector else None))
        m2_min_1, m2_min_2 = edges[0] * edges[0], edges[1] * edges[1]
    slope_factor = None
    pomeron = pomeron_slope is not None and not isovector
    if pomeron_slope is not None and not exact_range:
        raise ValueError("generate_dd: the Pomeron's shrinkage needs the exact transfer window")
    if exact_range:
        m2_1, m2_2, transfer1, transfer2 = draw_joint_dd_exact(
            m2_min_1, m2_max, m2_min_2, m2_max, SQRT_S, params,
            rng_exchange_kinematics, slope_mass_exponent=slope_mass_exponent,
            pomeron_slope=(float(pomeron_slope) if pomeron else None), pomeron_linear_t=pomeron_linear_t)
        if slope_mass_exponent is not None:
            slope_factor = vertex_slope_factor(math.sqrt(m2_1), math.sqrt(m2_2), slope_mass_exponent)
    else:
        if slope_mass_exponent is not None:
            raise ValueError("generate_dd: the factorized vertex slopes need the exact transfer window")
        m2_1, m2_2, transfer1, transfer2 = draw_joint_dd(
            m2_min_1, m2_max, m2_min_2, m2_max, SQRT_S, params,
            rng_exchange_kinematics)
    mass1 = math.sqrt(m2_1)
    mass2 = math.sqrt(m2_2)

    # (2) the exact two-system solve (no draws): the CM four-momenta for
    # the cascades, the lab four-momenta for the record.
    system1_cm, system2_cm = _cm_pair(mass1, float(transfer1), mass2,
                                      float(transfer2), SQRT_S)
    boost = _lab_boost_factory()
    system1_lab = boost(system1_cm)
    system2_lab = boost(system2_cm)

    # (3) both excitations decay through the frozen simplified cascade
    # (projectile side first, then target side, on the shared decay
    # substreams in that order)
    if isovector:
        projectile_finals_cm = _decay_system(
            system1_cm, mass1, catalog, fams,
            rng_decay_channel, rng_decay_kinematics, PROJECTILE_LINE_LABEL, flavor=flavor, ledger=ledger,
            line_qn=exchange.projectile.qn, line_isospin=exchange.projectile.isospin)
        target_finals_cm = _decay_system(
            system2_cm, mass2, catalog, fams,
            rng_decay_channel, rng_decay_kinematics, TARGET_LINE_LABEL, flavor=flavor, ledger=ledger,
            line_qn=exchange.target.qn, line_isospin=exchange.target.isospin)
    else:
        recs = None if diffractive_duality is None else isoscalar_exchange_records(catalog, duality_line_shapes)

        def _system(system_cm, mass, label):
            # the low-mass system as an I = 1/2 record (the diffractive duality, 2026-10-15)
            if recs is not None and mass < float(diffractive_duality) and any(r.lo < mass < r.hi for r in recs):
                return _record_decay(system_cm, mass, recs, catalog, rng_decay_channel, rng_decay_kinematics, label)
            return _decay_system(system_cm, mass, catalog, fams, rng_decay_channel, rng_decay_kinematics, label,
                                 flavor=flavor, ledger=ledger)
        projectile_finals_cm = _system(system1_cm, mass1, PROJECTILE_LINE_LABEL)
        target_finals_cm = _system(system2_cm, mass2, TARGET_LINE_LABEL)

    # (4) the lab assembly (one exact forward boost per four-momentum)
    projectile_finals = [
        FinalHadron(
            pdg_id=h.pdg_id,
            four_momentum=boost(h.four_momentum),
            mass=h.mass,
            qn=h.qn,
            line=h.line,
        )
        for h in projectile_finals_cm
    ]
    target_finals = [
        FinalHadron(
            pdg_id=h.pdg_id,
            four_momentum=boost(h.four_momentum),
            mass=h.mass,
            qn=h.qn,
            line=h.line,
        )
        for h in target_finals_cm
    ]
    final_state = tuple(projectile_finals) + tuple(target_finals)

    # (5) the ledger lines: the two outgoing excited systems (projectile
    # side first), both inheriting the proton quantum numbers after the
    # vacuum exchanges' zero transfers.
    lines = (
        LineState(B=1, Q=1, S=0, I3x2=1, four_momentum=system1_lab),
        LineState(B=1, Q=1, S=0, I3x2=1, four_momentum=system2_lab),
    ) if not isovector else exchange.line_states(system1_lab, system2_lab)

    # (6) the central gap: the rapidity interval between the nearest
    # final-state hadrons of the two sides
    projectile_ys = [_rapidity(h.four_momentum) for h in projectile_finals]
    target_ys = [_rapidity(h.four_momentum) for h in target_finals]
    gap = min(abs(y_p - y_t)
              for y_p in projectile_ys for y_t in target_ys)

    return DDEvent(
        class_label=DD_CLASS_LABEL,
        m1=float(mass1),
        m2=float(mass2),
        t1=float(transfer1),
        t2=float(transfer2),
        lines=lines,
        exchanges=((VACUUM_EXCHANGE_NAME, ZERO_TRANSFER),
                   (VACUUM_EXCHANGE_NAME, ZERO_TRANSFER)) if not isovector else exchange.records(),
        final_state=final_state,
        gap_size=float(gap),
        exchange_configuration=exchange,
        line_m_min=edges,
        slope_factor=slope_factor,
        pomeron=(bool(pomeron) if pomeron_slope is not None else None),
    )
