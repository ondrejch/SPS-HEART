"""The one-neutral-excitation non-diffractive class (Round 5, card 7).

``TASK-20260925-r5-non-diffrative-central-excitation``; plan section 22
task 31 (line 1756); plan section 11.1 item 5 (line 828) with plan
section 11.3 (lines 844-870).  The card implements the non-diffractive
vacuum-excitation event class of the initial topology basis:

> 5. **Non-diffractive vacuum excitation:** both baryonic lines may be
>    excited and one neutral color-singlet mesonic excitation may be
>    produced centrally.

The pre-registered instantiation (the card): each baryonic line is a
card 3 remnant at the nominal R50 split (intact proton 0.5 / low-mass
baryonic resonance 0.5 / high-mass continuum 0.0 per line, one
``spectral_state`` uniform per line), and the central source carries
``(B, Q, S, 2I3) = (0, 0, 0, 0)`` exactly (plan line 849) with mass
``M_c`` from card 4's central-production law
(:func:`na49_mixing.topology_mass_laws.draw_central`) on the exact
limits ``[m_pi0^2, (sqrt(s) - M_1 - M_2)^2]`` and the total transfer
``t_c = t_1 + t_2`` of the two exchanges.  The central system decays
through the frozen simplified cascade (the Round 4 modules consumed
read-only with isotropic decay angles; the R45 alignment result is not
applied, review H-6; Round 6 owns the
flavor weights): the channel enumeration of the Round 4
``cascade_decay`` module over the committed catalog's ``(0, 0, 0)``
records and the activated neutral-mesonic continuum family realizes the
plan section 11.3 channels (mesonic resonances and pions; the K K-bar
direct channels; the cascade recursion through multiple lower-mass
neutral resonances), and the associated strangeness of plan section
11.3 appears in the baryonic excitation decays (the ``(1, 1, 0)``
catalog channels such as K Lambda), never centrally.

Exchange and ledger conventions (plan section 11.4)
---------------------------------------------------

Each of the two color-singlet exchanges is a vacuum exchange carrying
the declared ``(B, Q, S, 2I3)`` transfer ``(0, 0, 0, 0)`` (plan line
877); both outgoing baryonic lines inherit their incoming line quantum
numbers ``(1, 1, 0, 1)`` after the zero transfers
(:func:`na49_mixing.exchange_ledger.apply_transfer_qn`, the pinned
card-2 arithmetic), and the central source carries
:data:`CENTRAL_QN = (0, 0, 0, 0)` exactly.  The event record carries
the two incoming proton lines (the pinned beam configuration), the two
outgoing baryonic line states (projectile side first, target side), the
central-source line state, and the two exchange records in card 1's
:class:`~na49_mixing.topology_elastic.ExchangeRecord` form; card 2's
:func:`na49_mixing.exchange_ledger.audit_event` consumes these records
read-only with zero violations, rule (e) holding the central source at
``B = Q = S = 0`` (plan line 849).  :func:`check_central_neutrality` is
the mechanism-side check of plan section 11.4 line 881: the central
source acquires no charge, strangeness, or baryon number without a
physical color-singlet carrier, and every central decay vertex closes
locally to ``(0, 0, 0)`` (card 2 rule (e) plus the local closure of the
central source's decay vertex on the record).

Exact kinematics (the card's three-system solve)
------------------------------------------------

:func:`nd_kinematics` solves the three outgoing four-momenta from
``(M_1, t_1, M_2, t_2, M_c, sqrt_s)`` in the overall center-of-mass
frame of the pinned beam configuration (beam
:data:`~na49_mixing.topology_elastic.BEAM_MOMENTUM = 158.0 GeV/c`,
proton mass :data:`~na49_mixing.topology_elastic.PROTON_MASS =
0.93827231 GeV`), boosted into the lab frame with the exact
fixed-target CM velocity.  The operative solve convention (documented
here, the double-diffraction card's over-determination pattern at the
three-system extension): the central system is produced at rest in the
overall CM (the central production idealization: the source sits at
``y* = 0`` at the production vertex and the final-state spread comes
from the frozen cascade), the two baryonic systems recoil back-to-back
against it with the exact two-body kinematics of the ``(M_1, M_2)``
pair at the pair invariant ``sqrt_s - M_c``, and the production angle
follows from the drawn projectile-side transfer through the documented
invariant relation on the first baryonic line,

    ``t_1 = m_p^2 + M_1^2 - 2 (E_p* E_1* - p* q* cos(theta*))``,

the same invariant the committed single- and double-diffraction solves
use on their first baryonic line.  The realized target-side transfer
follows the exact back-to-back coupling of the solved configuration;
the recorded ``t_1`` and ``t_2`` stay the drawn values (the joint law's
exchange-kinematics record; the identifiable-effective-combinations
rule, plan line 894), and the drawn ``t_2`` acts on the exchange
kinematics through the total transfer ``t_c = t_1 + t_2`` of the
central-production law.  The drawn ``t`` domains are card 4's
factorized effective ranges
(:func:`na49_mixing.topology_mass_laws.t_range`, each computed with the
other line at the intact proton mass, the pre-registered card-4
joint-law convention); the drawn domains extend past the exact
reachable range of the realized three-system configuration on
slivers at either edge, where ``cos(theta*)`` clamps into ``[-1, 1]``
and the angle saturates at the clamped configuration (the committed
single- and double-diffraction sliver pattern).  At the drawn-closure
edge (``M_1 + M_2 + M_c`` marginally above ``sqrt_s`` within the card-4
joint-closure tolerance) the momenta resolve at the exact at-rest
configuration.  The event's four-momentum closure against the incoming
lab sum holds by construction within ``1e-9`` relative.

Frame convention (the double-diffraction card's convention): all three
systems decay through the frozen simplified cascade with the Round 4
modules consumed read-only, and the cascade's per-vertex conservation
checks run in the overall center-of-mass frame of the pinned beam
configuration -- the frame in which the frozen ``draw_decay`` machinery
keeps its documented closure precision (boosting near-threshold
vertices through the lab-frame parent velocity reconstructs the
invariant through ``1 - beta^2`` and amplifies the component rounding
beyond the absolute tolerance).  The collected final states of all
three systems are boosted into the lab frame once, at assembly, with
the exact forward boost.

The three-system decay routing
------------------------------

The card's three cascade decays run in the fixed order projectile line,
target line, central source, on the shared decay substreams:

- each baryonic system is a card 3 remnant routed through the frozen
  simplified cascade's established-record routing (the card-9 handoff's
  delegation target, the card-5/card-6 established-sibling assembly
  convention): a stable committed record (the intact proton and the
  proton's own catalog record, which the low-mass kind's record draw can
  select) stays the line's final state with zero decay draws, and an
  unstable record decays through the byte-unchanged Round 2 kernel (the
  kernel's own documented draw budget on the same substreams).  Two of
  the ten low-mass catalog records (Delta(1600)+ and Delta(1700)+)
  carry pole masses below their own recorded support windows, so the
  card-9 handoff predicate refuses them at their pole masses and the
  delegation runs through the kernel directly (identical consumption
  and bit-identical decay vertices);
- the central source is a neutral-mesonic continuum state (the
  activated ``(0, 0, 0)`` family ``mesonic_neutral`` of
  ``config/v2_singlet/continuum_resonances.json``) that follows the
  card-9 termination rules of the frozen simplified cascade: the hard
  depth guard, the catalog handoff (structurally present; the
  predicate is false on every line state of the continuum root), the
  continuum continuation (one ``draw_decay`` step over the continuum
  pool; the established sibling is assembled -- a stable record is
  final, an unstable record decays through the kernel's own budget),
  the direct-channel termination (one ``draw_decay`` step over the
  direct pool; both daughters are assembled), and the below-threshold
  branch (a source with no open channel at all stays a single
  quantum-number-free final state at its mass with no vertex and zero
  decay draws; the plan section 11.3 channel list carries no
  minimum-mass requirement).

Draw budget (the card's pre-registered counts)
----------------------------------------------

Exactly two uniforms per event on the closed ``spectral_state``
substream (one card-3 remnant selection per baryonic line, the nominal
split), exactly five uniforms on the closed ``exchange_kinematics``
substream (the ``t_1`` draw, the ``t_2`` draw, then the ``M_c^2``
central draw of card 4 -- two uniforms -- in that order; the closure is
exact by construction, so no rejection attempt exists), and the frozen
Round 4 budget of one ``decay_channel`` plus two ``decay_kinematics``
uniforms per decay vertex on ``decay_channel`` / ``decay_kinematics``
for all three systems (a catalog handoff continues the same two
substreams into the byte-unchanged Round 2 kernel's own documented
budget, the card-9 handoff convention).  The ``topology_selection``
class draw belongs to the runner (card 9): :func:`generate_nd`
receives no class stream and draws no class uniform.

Plan section 11.3 channel-set notes (the committed surfaces)
------------------------------------------------------------

The live central channel enumeration runs on the committed surfaces
read-only.  The K K-bar direct channels of plan line 855 are the
family's ``(321, -321)`` and ``(310, 130)`` classes, open above the
two-kaon threshold, and the same pair classes are recorded channels of
the committed ``(0, 0, 0)`` records f2(1270), rho3(1690)0, f4(2050),
and phi(1020) (the phi record carries the plan's phi route to K K-bar
as its recorded ``K+ K-`` and ``K0S K0L`` channels).  The K* route of
plan line 855 is recorded on the declared strange mesonic families'
allowed-decay classes (the ``K*(892)`` classes ``(313, 111)``,
``(-313, 111)``, ``(323, -211)``, ``(-323, 211)`` of the
``mesonic_antistrange`` and ``mesonic_strange`` records); no committed
``(0, 0, 0)`` record carries a ``K*0 K-bar*0`` channel, so the route
enters the live enumeration when the substrate declares it.  The
baryon-antibaryon clause of plan line 856 binds its exact two-body
thresholds wherever such channels exist; the committed ``(0, 0, 0)``
enumeration carries none, and every two-body channel the enumeration
opens is open above its daughter pole-mass threshold (the card-4
enumeration's own open condition, verified on the pinned traces).

Hermetic: standard library and numpy only; cards 1-4's modules, the
committed single-diffraction module (card 5, consumed read-only for
the pinned beam configuration and the final-state record helpers), and
the Round 4 cascade modules are consumed read-only; no event generation
driver, no external tables, no Monte Carlo run (the acceptance is
hermetic: pinned traces, exact quantum-number and kinematics checks,
draw budgets; the runs R50-R55 exercise the class statistically).
"""

from __future__ import annotations

from .flavor_weights import vertex_ledger_record

import dataclasses
import math
from dataclasses import dataclass

from .cascade_decay import (
    CATALOG_FAMILY_PREFIX,
    ISOTROPIC_ALIGNMENT_LABEL,
    KIND_CONTINUUM,
    MAX_CASCADE_DEPTH,
    _cascade_depth_limit,
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
from .resonance_decay import ResonanceDecayer
from .rng_substreams import SUBSTREAM_LABELS
from .spectral_density import ContinuumFamily
from .topology_elastic import (
    BEAM_MOMENTUM,
    PROTON_MASS,
    SQRT_S,
    ExchangeRecord,
)
from .topology_mass_laws import (
    draw_marginal,
    PI0_MASS,
    MassLawParams,
    draw_central,
    draw_t,
    draw_t_window,
    joint_closure,
    pp_t_window,
    t_range,
    vertex_slope_factor,
)
from .topology_remnants import (
    low_mass_records,
    KIND_INTACT_PROTON,
    RemnantSelection,
    select_remnant,
)
from .isovector_exchange import ExchangeConfiguration, line_mass_threshold, select_line_remnant
from .nd_duality import duality_pick, duality_records, duality_redraw_mass, duality_redraw_mass_shapes
from .topology_single_diffraction import (
    _assemble_daughter,
    _decay_excitation,
    FinalHadron,
    _beam_configuration,
    _final_from_catalog,
    _kernel_finals,
    _lab_boost_factory,
    _rapidity,
)

__all__ = [
    "ND_CLASS_LABEL",
    "ND_LINE_LABELS",
    "CENTRAL_LINE_LABEL",
    "PROJECTILE_LINE_LABEL",
    "TARGET_LINE_LABEL",
    "CENTRAL_QN",
    "PI0_MASS",
    "NOMINAL_PARAMS",
    "NOMINAL_ND_REMNANT_WEIGHTS",
    "BEAM_MOMENTUM",
    "PROTON_MASS",
    "SQRT_S",
    "CentralNeutralityViolation",
    "NDEvent",
    "check_central_neutrality",
    "remnant_excitation_window",
    "generate_nd",
    "nd_kinematics",
    "nd_limits",
]


#: The class label of the non-diffractive class (the inelastic-sample
#: class label of the R50 templates; plan section 11.1 item 5, line 828).
ND_CLASS_LABEL: str = "nd"

#: The closed line-label set of the final-state hadrons: the two baryonic
#: remnant lines (the projectile side first, the target side second; the
#: beam direction carries the projectile side) and the central source.
PROJECTILE_LINE_LABEL: str = "projectile"
TARGET_LINE_LABEL: str = "target"
CENTRAL_LINE_LABEL: str = "central"
ND_LINE_LABELS: frozenset[str] = frozenset(
    {PROJECTILE_LINE_LABEL, TARGET_LINE_LABEL, CENTRAL_LINE_LABEL})

#: The central source's quantum numbers in the doubled-I3 integer
#: convention ``(B, Q, S, 2I3)`` (plan line 849: for vacuum exchange a
#: central excitation must have B = Q = S = 0; the third isospin
#: component is zero with the charge).
CENTRAL_QN: tuple[int, int, int, int] = (0, 0, 0, 0)

#: The pion mass, the lower-mass reference of the central-production law
#: (the lightest neutral state of plan section 11.3).  The card's
#: interface carried the 0.1349768 quote, resolved to card 4's committed
#: catalog value :data:`~na49_mixing.topology_mass_laws.PI0_MASS`
#: (0.1349766 GeV, the committed catalog record pdg_id 111; the card-4
#: resolution precedent), consumed read-only here.
PI0_MASS: float = PI0_MASS

#: The nominal effective mass-law parameters (the R50 nominal values,
#: index item 2: epsilon = 0, B_SD the M8-frozen effective transfer
#: slope; fixed here, fit in R51 by card 11).
NOMINAL_PARAMS: MassLawParams = MassLawParams(0.0, 8.504140762083612)

#: The nominal R50 remnant-kind split of both baryonic lines (index
#: item 2; fixed, not fit: intact proton 0.5, low-mass baryonic
#: resonance 0.5, high-mass 0.0 per line; plan line 828 "may be
#: excited" names exactly the intact and low-mass options and the
#: high-mass weight is the pre-registered zero of the nominal split, the
#: kind stays legal through the card-3 interface).
NOMINAL_ND_REMNANT_WEIGHTS: dict[str, float] = {
    "intact_proton": 0.5,
    "low_mass_baryonic_resonance": 0.5,
    "high_mass_baryonic_continuum": 0.0,
}

#: The vacuum exchange name (each of the class's two color-singlet
#: exchanges; the transfer is ``(0, 0, 0, 0)``, plan line 877).
VACUUM_EXCHANGE_NAME: str = "vacuum"
ZERO_TRANSFER: tuple[int, int, int, int] = (0, 0, 0, 0)

#: The frozen simplified-cascade continuum family of the central source
#: (the activated neutral mesonic ``(0, 0, 0)`` family of
#: ``config/v2_singlet/continuum_resonances.json``; the cascade line's
#: ``family_id``).
CENTRAL_FAMILY_ID: str = "mesonic_neutral"

#: The class label carried on the cascade lines' exchange ancestry (the
#: exchange-layer lineage of the Round 4 cascade states).
ANCESTRY_LABEL: str = "nd"

#: The exact-closure float tolerance of the card-4 joint closure (the
#: pinned 1e-12, consumed through
#: :func:`na49_mixing.topology_mass_laws.joint_closure`).
_CLOSURE_TOLERANCE: float = 1e-12


class CentralNeutralityViolation(ValueError):
    """A central-neutrality violation of the non-diffractive ledger
    (checked, never repaired): the central source carries a nonzero
    ``(B, Q, S, 2I3)`` component against :data:`CENTRAL_QN`, or a
    central decay vertex on the record fails to close locally to
    ``(0, 0, 0)`` (plan line 849; card 2 rule (e))."""


@dataclass(frozen=True)
class NDEvent:
    """The frozen non-diffractive event record (the card's ledger-ready
    record).

    ``class_label`` is ``"nd"``.  ``remnants`` is the card-3 selection
    pair of the two baryonic lines (projectile side first, target side).
    ``central_mass`` is the central source mass ``M_c`` [GeV] and
    ``t_c`` the total transfer ``t_1 + t_2`` [GeV^2]; ``t1`` and ``t2``
    are the two drawn per-exchange transfers the card's own phase-space
    checks consume (the card's record sketch names ``t_c``; the two
    per-exchange values are its summands, carried for the pinned-trace
    identities).  ``lines`` is the card-2 ledger record pair of the two
    outgoing baryonic lines (projectile side first, target side), both
    carrying ``(1, 1, 0, 1)`` after the vacuum exchanges' zero
    transfers.  ``central_source`` is the card-2 ledger record of the
    central source, carrying :data:`CENTRAL_QN` exactly with its lab
    four-momentum.  ``exchanges`` is the pair of the class's two vacuum
    exchange records in card 1's
    :class:`~na49_mixing.topology_elastic.ExchangeRecord` form.
    ``final_state`` carries every final-state hadron of the three
    systems (the projectile line, the target line, the central line, in
    that order).  ``remnant_windows`` (2026-09-29) is the pair of
    excitation windows ``(lo, hi)`` [GeV] the two remnant draws used
    (projectile side first): the exact reweighting of the excited kinds
    in the exponent needs them, and the un-excited kinds ignore them.
    ``central_component`` (2026-10-02) is ``False`` for an event drawn
    without the central source (the two-component variant of
    :func:`generate_nd`): then ``central_mass`` is 0.0, ``central_source``
    is ``None``, and no final-state hadron carries the central line.
    ``exchange_configuration`` (Round 8, 2026-10-04) is the drawn
    :class:`~na49_mixing.isovector_exchange.ExchangeConfiguration` of a
    run with a positive isovector fraction (``None`` otherwise); under the
    isovector exchange ``remnants``, ``lines`` and ``exchanges`` carry the
    configured line charges and the opposite transfers.
    ``slope_factor`` (2026-10-08) is the factorized vertex slope factor
    ``B / b`` of the transfer under ``slope_mass_exponent``
    (:func:`~na49_mixing.topology_mass_laws.vertex_slope_factor`; ``None``
    for the single slope).
    ``duality_edges`` (2026-10-12) is the pair of shape edges ``Ms`` [GeV]
    of the two lines under ``duality`` (:mod:`na49_mixing.nd_duality`):
    the upper edge of the continuum share whose mass the records' line
    shapes redrew, ``nan`` for a line that kept its continuum mass;
    ``None`` without the option.
    """

    class_label: str
    remnants: tuple[RemnantSelection, RemnantSelection]
    central_mass: float
    t_c: float
    t1: float
    t2: float
    lines: tuple[LineState, LineState]
    central_source: LineState | None
    exchanges: tuple["ExchangeRecord", "ExchangeRecord"]
    final_state: tuple[FinalHadron, ...]
    remnant_windows: tuple[tuple[float, float], tuple[float, float]] = (
        (math.nan, math.nan), (math.nan, math.nan))
    central_component: bool = True
    exchange_configuration: ExchangeConfiguration | None = None
    slope_factor: float | None = None
    duality_edges: tuple[float, float] | None = None


# ----------------------------------------------------------------------
# The exact central limits (the card's nd_limits)
# ----------------------------------------------------------------------
def nd_limits(m1: float, m2: float, sqrt_s: float) -> tuple[float, float]:
    """Exact central limits ``(m2_min, m2_max)`` in GeV^2 of the
    central-production law: ``(m_pi0^2, (sqrt_s - M_1 - M_2)^2)``.

    ``m_pi0`` is the committed catalog pion mass
    (:data:`~na49_mixing.topology_mass_laws.PI0_MASS`, the lightest
    neutral state of plan section 11.3) and the upper edge places the
    drawn closure ``M_1 + M_2 + M_c <= sqrt_s`` exactly
    (:func:`na49_mixing.topology_mass_laws.joint_closure`'s pinned
    ``1e-12`` tolerance covers the float dust of the drawn window).
    ``ValueError`` on a non-positive or non-finite remnant mass, a
    non-finite ``sqrt_s``, or a system closed below the pion threshold
    (``sqrt_s <= M_1 + M_2 + m_pi0``: an empty central window, checked,
    never repaired).
    """

    mass1 = float(m1)
    mass2 = float(m2)
    root_s = float(sqrt_s)
    for name, value in (("m1", mass1), ("m2", mass2)):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(
                f"nd_limits: the remnant mass {name} must be a positive "
                f"finite GeV value, got {m1 if name == 'm1' else m2!r}")
    if not math.isfinite(root_s):
        raise ValueError(
            f"nd_limits: sqrt_s must be finite, got {sqrt_s!r}")
    m2_min = PI0_MASS * PI0_MASS
    residual = root_s - mass1 - mass2
    m2_max = residual ** 2
    if not residual > PI0_MASS or not m2_max > m2_min:
        raise ValueError(
            "nd_limits: the central window is empty: sqrt_s - M_1 - M_2 = "
            f"{root_s - mass1 - mass2!r} GeV leaves no room above the "
            f"neutral-pion threshold {PI0_MASS!r} GeV (checked, never "
            "repaired)")
    return (m2_min, m2_max)


# ----------------------------------------------------------------------
# The exact three-system solve (the card's nd_kinematics)
# ----------------------------------------------------------------------
def _cm_triplet(
    m1: float,
    t1: float,
    m2: float,
    m_c: float,
    sqrt_s: float,
):
    """The exact center-of-mass four-momenta ``(system_1, system_2,
    central)`` of the three-system configuration from ``(M_1, t_1, M_2,
    M_c, sqrt_s)``, projectile side first (the operative solve
    convention of the module docstring).

    The central system at rest in the overall CM; the two baryonic
    systems back-to-back against it with the exact two-body kinematics
    of the ``(M_1, M_2)`` pair at the pair invariant
    ``sqrt_s - M_c``; the production angle from the drawn
    projectile-side transfer through the documented invariant relation
    on the first baryonic line, clamped into ``[-1, 1]`` on the drawn
    slivers; the at-rest resolution when the drawn closure leaves no
    two-body momentum.
    """

    mass1 = float(m1)
    mass2 = float(m2)
    transfer1 = float(t1)
    central_mass = float(m_c)
    root_s = float(sqrt_s)
    if not (math.isfinite(mass1) and mass1 > 0.0):
        raise ValueError(
            "nd_kinematics: the projectile-side remnant mass must be a "
            f"positive finite GeV value, got {m1!r}")
    if not (math.isfinite(mass2) and mass2 > 0.0):
        raise ValueError(
            "nd_kinematics: the target-side remnant mass must be a "
            f"positive finite GeV value, got {m2!r}")
    if not (math.isfinite(central_mass) and central_mass > 0.0):
        raise ValueError(
            "nd_kinematics: the central source mass must be a positive "
            f"finite GeV value, got {m_c!r}")
    if not (math.isfinite(transfer1) and transfer1 <= 0.0):
        raise ValueError(
            "nd_kinematics: the momentum transfer t1 must be a "
            f"non-positive finite GeV^2 value, got {t1!r}")
    if not math.isfinite(root_s):
        raise ValueError(
            f"nd_kinematics: sqrt_s must be finite, got {sqrt_s!r}")
    if not joint_closure(mass1, mass2, root_s, central_mass):
        raise ValueError(
            "nd_kinematics: the three-system closure is violated: "
            f"M_1 + M_2 + M_c = {mass1 + mass2 + central_mass!r} GeV "
            f"exceeds sqrt_s = {root_s!r} GeV beyond the joint-closure "
            "tolerance (checked, never repaired)")

    _e_beam, _e_total, sq_exact, _p_star_beam, _beta, _gamma = (
        _beam_configuration())
    s = sq_exact * sq_exact
    mp2 = PROTON_MASS * PROTON_MASS
    m2_1 = mass1 * mass1
    m2_2 = mass2 * mass2

    # The central system at rest in the overall CM; the pair invariant
    # of the two baryonic systems and their exact two-body quantities.
    e_c_cm = central_mass
    e_pair = sq_exact - central_mass
    if e_pair <= 0.0:
        raise ValueError(
            "nd_kinematics: the pair invariant sqrt_s - M_c is "
            f"non-positive ({e_pair!r} GeV) at the drawn closure (checked, "
            "never repaired)")
    q_pair = math.sqrt(max(0.0, (e_pair * e_pair - (mass1 + mass2) ** 2)
                           * (e_pair * e_pair - (mass1 - mass2) ** 2))) / (
        2.0 * e_pair)
    e_1_cm = (e_pair * e_pair + m2_1 - m2_2) / (2.0 * e_pair)
    e_2_cm = (e_pair * e_pair + m2_2 - m2_1) / (2.0 * e_pair)
    if q_pair <= 0.0:
        # the drawn-closure sliver: the at-rest configuration (the
        # committed double-diffraction resolution; checked, never
        # repaired)
        return ((e_1_cm, 0.0, 0.0, 0.0),
                (e_2_cm, 0.0, 0.0, 0.0),
                (e_c_cm, 0.0, 0.0, 0.0))

    # The production angle from the drawn projectile-side transfer
    # through the invariant relation on the first baryonic line,
    # t_1 = m_p^2 + M_1^2 - 2 (E_p* E_1* - p* q* cos(theta*)); the
    # drawn domains extend past the exact reachable range on slivers at
    # either edge, where the angle saturates at the clamped
    # configuration and the recorded transfers stay the drawn values.
    e_p_cm = sq_exact / 2.0
    p_star = math.sqrt(max(0.0, s / 4.0 - mp2))
    cos_theta = (transfer1 - mp2 - m2_1 + 2.0 * e_p_cm * e_1_cm) / (
        2.0 * p_star * q_pair)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))

    system1_cm = (e_1_cm, q_pair * sin_theta, 0.0, q_pair * cos_theta)
    system2_cm = (e_2_cm, -q_pair * sin_theta, 0.0, -q_pair * cos_theta)
    central_cm = (e_c_cm, 0.0, 0.0, 0.0)
    return system1_cm, system2_cm, central_cm


def _cm_triplet_recoil(m1: float, t1: float, m2: float, t2: float, m_c: float, phi2: float):
    """The recoiling three-system solve in the overall CM (2026-10-07; card
    TASK-20261005-kernel-nd-dd-transfer-range, Stage B): ``(system_1,
    system_2, central)``, projectile side first (+z).

    In a non-diffractive event the two leading systems sit at the ends of a
    multiperipheral chain, each end link with its own momentum transfer;
    each line takes its own transverse momentum and the central part
    absorbs the recoil.  Line 1 carries ``k_1 = sqrt(-t_1)`` in the x-z
    plane, line 2 ``k_2 = sqrt(-t_2)`` at the azimuth *phi2*; the central
    system sits at rapidity 0 with the transverse recoil ``-(k_1 + k_2)``;
    the lines' longitudinal momenta follow from the light-cone quadratic
    ``a + b = W``, ``m_T1^2/a + m_T2^2/b = W`` at ``W = sqrt(s) - m_T,c``
    (the pair at zero longitudinal momentum), line 1 on the forward root.
    A configuration beyond the transverse closure ``W >= m_T1 + m_T2``
    scales both k by one common factor onto it (bisection, 60 steps), the
    analogue of the 2 -> 2 angle saturation.  The drawn variables keep their
    densities; the realized invariant transfers are recorded, not drawn.
    """
    sq = _beam_configuration()[2]
    mass1, mass2, central_mass = float(m1), float(m2), float(m_c)
    k1 = math.sqrt(max(0.0, -float(t1)))
    k2 = math.sqrt(max(0.0, -float(t2)))
    c2, s2 = math.cos(float(phi2)), math.sin(float(phi2))

    def config(lam: float):
        a1x, a1y = lam * k1, 0.0
        a2x, a2y = lam * k2 * c2, lam * k2 * s2
        kcx, kcy = -(a1x + a2x), -(a1y + a2y)
        mtc = math.sqrt(central_mass * central_mass + kcx * kcx + kcy * kcy)
        mt1 = math.sqrt(mass1 * mass1 + a1x * a1x + a1y * a1y)
        mt2 = math.sqrt(mass2 * mass2 + a2x * a2x + a2y * a2y)
        return (a1x, a1y, a2x, a2y, kcx, kcy, mtc, mt1, mt2, sq - mtc)

    c = config(1.0)
    if c[9] < c[7] + c[8]:
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            cm = config(mid)
            if cm[9] >= cm[7] + cm[8]:
                lo = mid
            else:
                hi = mid
        c = config(lo)
    a1x, a1y, a2x, a2y, kcx, kcy, mtc, mt1, mt2, w = c
    disc = max(0.0, (w * w - (mt1 + mt2) ** 2) * (w * w - (mt1 - mt2) ** 2))
    ap = (w * w + mt1 * mt1 - mt2 * mt2 + math.sqrt(disc)) / (2.0 * w)
    bp = w - ap
    system1 = (0.5 * (ap + mt1 * mt1 / ap), a1x, a1y, 0.5 * (ap - mt1 * mt1 / ap))
    system2 = (0.5 * (bp + mt2 * mt2 / bp), a2x, a2y, 0.5 * (bp - mt2 * mt2 / bp))
    central = (mtc, kcx, kcy, 0.0)
    return system1, system2, central


def _cm_pair(m1: float, t1: float, m2: float, sqrt_s: float):
    """The exact center-of-mass four-momenta ``(system_1, system_2)`` of
    the two-component variant (no central source, 2026-10-02): the two
    baryonic systems back-to-back with the exact two-body kinematics of
    the ``(M_1, M_2)`` pair at the full invariant ``sqrt_s``, and the
    production angle from the drawn projectile-side transfer through the
    invariant relation of :func:`_cm_triplet` (clamped into ``[-1, 1]``
    on the drawn slivers)."""

    mass1 = float(m1)
    mass2 = float(m2)
    transfer1 = float(t1)
    for name, value in (("projectile-side", mass1), ("target-side", mass2)):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(
                f"nd_kinematics: the {name} remnant mass must be a positive "
                f"finite GeV value, got {value!r}")
    if not (math.isfinite(transfer1) and transfer1 <= 0.0):
        raise ValueError(
            "nd_kinematics: the momentum transfer t1 must be a "
            f"non-positive finite GeV^2 value, got {t1!r}")
    if not joint_closure(mass1, mass2, float(sqrt_s)):
        raise ValueError(
            f"nd_kinematics: the two-system closure is violated: M_1 + M_2 = "
            f"{mass1 + mass2!r} GeV exceeds sqrt_s = {float(sqrt_s)!r} GeV "
            "(checked, never repaired)")
    _e_beam, _e_total, sq_exact, _p_star_beam, _beta, _gamma = (
        _beam_configuration())
    s = sq_exact * sq_exact
    mp2 = PROTON_MASS * PROTON_MASS
    m2_1 = mass1 * mass1
    m2_2 = mass2 * mass2
    e_pair = sq_exact
    q_pair = math.sqrt(max(0.0, (e_pair * e_pair - (mass1 + mass2) ** 2)
                           * (e_pair * e_pair - (mass1 - mass2) ** 2))) / (
        2.0 * e_pair)
    e_1_cm = (e_pair * e_pair + m2_1 - m2_2) / (2.0 * e_pair)
    e_2_cm = (e_pair * e_pair + m2_2 - m2_1) / (2.0 * e_pair)
    if q_pair <= 0.0:
        return ((e_1_cm, 0.0, 0.0, 0.0), (e_2_cm, 0.0, 0.0, 0.0))
    e_p_cm = sq_exact / 2.0
    p_star = math.sqrt(max(0.0, s / 4.0 - mp2))
    cos_theta = (transfer1 - mp2 - m2_1 + 2.0 * e_p_cm * e_1_cm) / (
        2.0 * p_star * q_pair)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    return ((e_1_cm, q_pair * sin_theta, 0.0, q_pair * cos_theta),
            (e_2_cm, -q_pair * sin_theta, 0.0, -q_pair * cos_theta))


def nd_kinematics(
    m1: float,
    t1: float,
    m2: float,
    t2: float,
    m_c: float,
    sqrt_s: float,
) -> tuple[tuple[float, float, float, float],
           tuple[float, float, float, float],
           tuple[float, float, float, float]]:
    """Exact outgoing four-momenta ``(system_1, system_2, central)`` in
    the lab frame from ``(M_1, t_1, M_2, t_2, M_c, sqrt_s)``, projectile
    side first, target side second, central source third.

    The exact three-system solve of the pinned beam configuration (the
    module docstring's operative convention): the central system at
    rest in the overall CM, the two baryonic systems back-to-back
    against it with the exact two-body kinematics of the ``(M_1, M_2)``
    pair at the pair invariant ``sqrt_s - M_c``, the production angle
    from the drawn projectile-side transfer through the invariant
    relation

        ``t_1 = m_p^2 + M_1^2 - 2 (E_p* E_1* - p* q* cos(theta*))``,

    and the lab boost with the exact fixed-target CM velocity.  The
    production plane is the x-z plane.  Energy-momentum closure holds
    within ``1e-9`` relative by construction (the CM triplet sums to
    the total CM four-momentum and the boost maps it to the incoming
    lab sum).

    The drawn pair ``(t_1, t_2)`` is the exchange-kinematics record
    (card 4's factorized effective law); the operative solve realizes
    the projectile-side transfer through the documented invariant, the
    realized target-side transfer follows the exact back-to-back
    coupling of the solved configuration, and the drawn ``t_2`` stays
    the recorded exchange-kinematics variable acting through the total
    transfer ``t_c = t_1 + t_2`` of the central-production law.  The
    ``t_2`` argument is consumed by the interface and the validation
    (the record's drawn state); the solved configuration is fixed by
    ``(M_1, t_1, M_2, M_c, sqrt_s)``.  The angle saturates at the
    clamped configuration on the drawn slivers beyond the exact
    reachable ``t_1`` range, and a pair at the drawn-closure edge
    resolves at the exact at-rest configuration.  ``ValueError`` on
    non-positive masses, a positive or non-finite ``t_1``, or a
    three-system closure beyond the joint-closure tolerance (checked,
    never repaired).
    """

    transfer2 = float(t2)  # the drawn target-side transfer: validated
    if not (math.isfinite(transfer2) and transfer2 <= 0.0):
        raise ValueError(
            "nd_kinematics: the momentum transfer t2 must be a "
            f"non-positive finite GeV^2 value, got {t2!r}")
    system1_cm, system2_cm, central_cm = _cm_triplet(
        m1, t1, m2, m_c, sqrt_s)
    boost = _lab_boost_factory()
    return boost(system1_cm), boost(system2_cm), boost(central_cm)


# ----------------------------------------------------------------------
# The ledger lines
# ----------------------------------------------------------------------
def _incoming_lines() -> tuple[LineState, LineState]:
    """The pinned incoming ledger lines (projectile side first): the
    beam proton and the target proton, with the doubled-I3 proton
    quantum numbers and their lab four-momenta (the card-5 pinned beam
    configuration, consumed read-only)."""

    e_beam, _e_total, _sq, _ps, _beta, _gamma = _beam_configuration()
    return (
        LineState(B=1, Q=1, S=0, I3x2=1,
                  four_momentum=(e_beam, 0.0, 0.0, BEAM_MOMENTUM)),
        LineState(B=1, Q=1, S=0, I3x2=1,
                  four_momentum=(PROTON_MASS, 0.0, 0.0, 0.0)),
    )


# ----------------------------------------------------------------------
# The isospin mixture of the non-diffractive lines (2026-10-10)
# ----------------------------------------------------------------------
#: The non-diffractive views of the family lists, keyed on the list's
#: identity; the entry holds the list, so its id stays unique while cached.
_MIXTURE_VIEWS: dict[int, tuple[list, list]] = {}


def non_diffractive_families(fams: list[ContinuumFamily] | None) -> list[ContinuumFamily] | None:
    """The families as the non-diffractive cascade sees them: a family that
    carries ``non_diffractive_isospin_mixture`` (the run option
    ``KernelConfig.non_diffractive_isospin_mixture``, on the B != 0
    families) takes ``isospin_mixture``, so its vertices count every charge
    combination equally.  The non-vacuum exchanges of a non-diffractive
    event leave each baryonic line an equal mixture of I = 1/2 and 3/2,
    while the vacuum exchange of the diffractive classes keeps their lines
    at I = 1/2.  Without the option the list comes back as passed."""
    if fams is None:
        return None
    hit = _MIXTURE_VIEWS.get(id(fams))
    if hit is not None and hit[0] is fams:
        return hit[1]
    if not any(getattr(f, "non_diffractive_isospin_mixture", False) for f in fams):
        view = fams
    else:
        view = [dataclasses.replace(f, isospin_mixture=True) if f.non_diffractive_isospin_mixture else f
                for f in fams]
    if len(_MIXTURE_VIEWS) >= 8:
        _MIXTURE_VIEWS.clear()
    _MIXTURE_VIEWS[id(fams)] = (fams, view)
    _MIXTURE_VIEWS[id(view)] = (view, view)
    return view


# ----------------------------------------------------------------------
# The baryonic system's decay (the card-9 catalog handoff)
# ----------------------------------------------------------------------
def _decay_remnant(
    selection: RemnantSelection,
    system_cm: tuple[float, float, float, float],
    catalog: HadronCatalog,
    rng_channel,
    rng_kinematics,
    line: str,
    fams: list[ContinuumFamily] | None = None,
    flavor=None,
    ledger=None,
    line_isospin: float | None = None,
    duality=None,
) -> list[FinalHadron]:
    """One baryonic system's decay through the frozen simplified
    cascade's established-record routing (the card-5/card-6 established-
    sibling assembly convention, the card-9 handoff's delegation target):
    a stable committed record (the intact proton and the proton's own
    catalog record) stays the line's final state with zero decay draws;
    an unstable record decays through the byte-unchanged Round 2 kernel
    (the kernel's own documented draw budget on the passed substreams,
    bound at the cascade's closed substream generators).

    Every assembled final-state hadron carries the caller's line of
    origin (``"projectile"`` or ``"target"``) and its overall-center-
    of-mass four-momentum (the caller boosts to the lab frame once, at
    assembly).  An excited remnant of the isovector exchange passes its
    line's *line_isospin* (the cascade root of the line, Round 8).

    *duality* (2026-10-12) is the pair ``(Md, records)`` of
    :mod:`na49_mixing.nd_duality`: an excited vacuum (1, 1, 0) remnant
    below Md whose mass a record's range holds decays as that record (one
    uniform on *rng_channel* for the pick, then the record's own draws).
    """

    if line not in (PROJECTILE_LINE_LABEL, TARGET_LINE_LABEL):
        raise ValueError(
            f"_decay_remnant: the line label {line!r} is outside the "
            "closed baryonic line-label set "
            f"{sorted((PROJECTILE_LINE_LABEL, TARGET_LINE_LABEL))}")
    if selection.pdg_id is None:
        # the excited baryonic remnant (the high-mass continuum kind of
        # plan section 11.2, released by the Round 5 follow-up card
        # TASK-20260928-r5-gate5-followup-nd-remnant-excitation): the
        # baryonic line cascades through the same frozen simplified
        # cascade as a single-diffraction excitation, on the same
        # substreams, and its final state carries this line's label
        if fams is None:
            raise ValueError(
                "_decay_remnant: an excited baryonic remnant needs the "
                "continuum families (fams is None; checked, never repaired)")
        mass = float(selection.mass)
        if (duality is not None and line_isospin is None and mass < float(duality[0])
                and tuple(int(v) for v in selection.qn[:3]) == (1, 1, 0)
                and any(r.lo < mass < r.hi for r in duality[1])):
            # the low-mass line as one of the N* and Delta records (local duality)
            pdg = duality_pick(mass, duality[1], float(rng_channel.random()))
            kernel = ResonanceDecayer(catalog, 0, 0, 0)
            kernel._rng_channel = rng_channel
            kernel._rng_kinematics = rng_kinematics
            return list(_kernel_finals(catalog, kernel.decay_resonance(int(pdg), tuple(
                float(v) for v in system_cm)), line))
        return _decay_excitation(
            (float(system_cm[0]), float(system_cm[1]), float(system_cm[2]),
             float(system_cm[3])),
            float(selection.mass), catalog, fams, rng_channel,
            rng_kinematics, flavor, ledger, line=line,
            line_qn=tuple(int(v) for v in selection.qn[:3]),
            line_isospin=line_isospin)
    system_p4 = (float(system_cm[0]), float(system_cm[1]),
                 float(system_cm[2]), float(system_cm[3]))
    pdg = int(selection.pdg_id)
    if pdg in catalog.stable_states():
        # a stable record: the state itself is the line's final state
        # (the stable handoff state's empty vertex list; zero decay
        # draws)
        return [FinalHadron(
            pdg_id=pdg,
            four_momentum=system_p4,
            mass=float(selection.mass),
            qn=(int(selection.qn[0]), int(selection.qn[1]),
                int(selection.qn[2]), int(selection.qn[3])),
            line=line,
        )]
    # an unstable established record: the Round 2 kernel decays the
    # remainder (the card-9 handoff's delegation, the card-5/card-6
    # established-sibling assembly; the kernel's own documented budget
    # on the same substreams)
    kernel = ResonanceDecayer(catalog, 0, 0, 0)
    kernel._rng_channel = rng_channel
    kernel._rng_kinematics = rng_kinematics
    return list(_kernel_finals(
        catalog, kernel.decay_resonance(pdg, system_p4), line))


# ----------------------------------------------------------------------
# The central source's decay (the frozen simplified cascade)
# ----------------------------------------------------------------------
def _decay_central(
    central_cm: tuple[float, float, float, float],
    mass: float,
    catalog: HadronCatalog,
    fams: list[ContinuumFamily],
    rng_channel,
    rng_kinematics,
    flavor=None,
    ledger=None,
) -> list[FinalHadron]:
    """The central source's decay through the frozen simplified cascade
    (the Round 4 modules consumed read-only with isotropic decay angles):
    the card-9 termination rules with the caller's event assembly of
    the vertex siblings, in the overall center-of-mass frame of the
    pinned beam configuration (the frame convention of the module
    docstring).

    At every line state, in the card-9 order: the hard depth guard; the
    continuum continuation (one ``draw_decay`` step over the continuum
    pool; the established sibling is assembled -- a stable record is
    final, an unstable record decays through the kernel's own budget on
    the same substreams); the direct-channel termination (one
    ``draw_decay`` step over the direct pool; both daughters are
    assembled).  A state with no open channel at all is the
    below-threshold central source (the plan section 11.3 channel list
    carries no minimum-mass requirement): it stays a single
    quantum-number-free final state at its mass with no vertex and zero
    decay draws.

    Every assembled final-state hadron carries the ``"central"`` line
    of origin and its overall-center-of-mass four-momentum (the caller
    boosts to the lab frame once, at assembly).
    """

    finals: list[FinalHadron] = []
    root_cm = (float(central_cm[0]), float(central_cm[1]),
               float(central_cm[2]), float(central_cm[3]))
    current = CascadeState(
        momentum=root_cm,
        mass=float(mass),
        qn=(0, 0, 0),
        quantum={},
        family_id=CENTRAL_FAMILY_ID,
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
                f"non-diffractive central cascade: the line state of "
                f"family {current.family_id!r} at mass "
                f"{float(current.mass)!r} GeV carries depth "
                f"{int(current.depth)}, exceeding MAX_CASCADE_DEPTH = "
                f"{_cascade_depth_limit(fams)}")
        # (2) the catalog handoff (plan section 10.6, line 754; the
        # card-9 rule order, consumed read-only): the central source is
        # a continuum state, so the predicate is false on every line
        # state by construction and the branch is structurally present
        # only
        if is_catalog_handoff(current, catalog):
            vertices = handoff_decay(current, catalog, rng_channel,
                                     rng_kinematics)
            finals.extend(_kernel_finals(catalog, vertices,
                                         CENTRAL_LINE_LABEL))
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
                        catalog, pdg, daughter.momentum, CENTRAL_LINE_LABEL,
                        rng_channel, rng_kinematics))
                return finals
            successor = None
            for daughter in daughters:
                if daughter.family_id.startswith(CATALOG_FAMILY_PREFIX):
                    pdg = int(daughter.family_id[len(CATALOG_FAMILY_PREFIX):])
                    if pdg in catalog.stable_states():
                        finals.append(_final_from_catalog(
                            catalog, pdg, daughter.momentum,
                            CENTRAL_LINE_LABEL))
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
                            CENTRAL_LINE_LABEL))
                else:
                    successor = daughter
            if successor is None:
                raise ValueError(
                    "non-diffractive central cascade: the continuum "
                    "vertex produced no continuing daughter (card 4's "
                    "enumeration produces exactly one; checked, never "
                    "repaired)")
            current = successor
            continue
        # (3) the direct-channel termination: both daughters assembled.
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
                    catalog, pdg, daughter.momentum, CENTRAL_LINE_LABEL,
                    rng_channel, rng_kinematics))
            return finals
        # (4) the below-threshold central source: no open channel at
        # all, so the source stays a single quantum-number-free final
        # state at its mass (the plan section 11.3 channel list carries
        # no minimum-mass requirement; zero decay draws).
        # the line state itself (its four-momentum, mass and quantum
        # numbers; the root unless a continuation step came first)
        finals.append(FinalHadron(
            pdg_id=None,
            four_momentum=(float(current.momentum[0]), float(current.momentum[1]),
                           float(current.momentum[2]), float(current.momentum[3])),
            mass=float(current.mass),
            qn=(int(current.qn[0]), int(current.qn[1]), int(current.qn[2]),
                2 * int(current.qn[1]) - int(current.qn[0]) - int(current.qn[2])),
            line=CENTRAL_LINE_LABEL,
        ))
        return finals


# ----------------------------------------------------------------------
# The event generator
# ----------------------------------------------------------------------
def generate_nd(
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
    central: bool = True,
    exchange: ExchangeConfiguration | None = None,
    exact_range: bool = False,
    line_thresholds: bool = False,
    central_recoil: bool = False,
    central_epsilon: float | None = None,
    slope_mass_exponent: float | None = None,
    low_mass_spin_isospin: bool = False,
    duality: tuple[float, float | None] | None = None,
    duality_line_shapes: bool = False,
    duality_redraw_shapes: bool = False,
) -> NDEvent:
    """One non-diffractive event.

    *central* (2026-10-02) selects the central source.  The default
    ``True`` is the card's event, bit for bit.  ``False`` draws the
    two-component variant: the same two remnant selections, transfers and
    remnant decays on the same substreams (the central draw is the last
    one on ``exchange_kinematics`` and the central decay the last one on
    the decay substreams), no central mass draw, the two baryonic systems
    sharing the full ``sqrt_s`` (:func:`_cm_pair`), and no central
    source.  The variant measures the strength of the central component
    against the pure two-component picture.

    The card's pre-registered draw budget on the closed substream labels
    (:data:`na49_mixing.rng_substreams.SUBSTREAM_LABELS`): exactly two
    uniforms on ``spectral_state`` (one card-3 remnant selection per
    baryonic line, projectile side first, over the caller's *weights*;
    the nominal split intact 0.5 / low-mass 0.5 / high-mass 0.0 per
    line), exactly five uniforms on ``exchange_kinematics`` (the ``t_1``
    draw of ``exp(B_SD t)`` on card 4's factorized exact range
    :func:`na49_mixing.topology_mass_laws.t_range` computed with the
    other line at the intact proton mass -- one uniform -- then the
    ``t_2`` draw on its range -- one uniform -- then the ``M_c^2``
    central draw of card 4's
    :func:`na49_mixing.topology_mass_laws.draw_central` on the exact
    limits :func:`nd_limits` at the total transfer ``t_c = t_1 + t_2``
    -- two uniforms -- in that order; the closure is exact by
    construction, so no rejection attempt exists), and the frozen Round
    4 budget of one ``decay_channel`` plus two ``decay_kinematics``
    uniforms per decay vertex on ``decay_channel`` /
    ``decay_kinematics`` for all three systems in the fixed order
    projectile line, target line, central source (a catalog handoff
    continues the same two substreams into the byte-unchanged Round 2
    kernel's own documented budget, the card-9 handoff convention; a
    stable handoff state and a below-threshold central source consume
    zero decay draws).  The ``topology_selection`` class draw belongs to
    the runner (card 9): no class uniform is consumed.

    The kinematics solve and the lab assembly follow :func:`nd_kinematics`
    and the module docstring's frame convention: the three cascades run
    in the overall center-of-mass frame (projectile line, target line,
    central source, on the shared decay substreams in that order) and
    the assembly boosts once into the lab frame.  The event's ``lines``
    are the two outgoing baryonic ledger line states (projectile side
    first), both inheriting the proton quantum numbers ``(1, 1, 0, 1)``
    after the vacuum exchanges' zero transfers; the ``central_source``
    carries :data:`CENTRAL_QN` exactly; the ``exchanges`` records are
    the two vacuum exchanges with the zero transfer in card 1's
    :class:`~na49_mixing.topology_elastic.ExchangeRecord` form; the
    ``final_state`` carries the three systems' decay products in the lab
    frame.

    *exchange* (Round 8, 2026-10-04) is the event's drawn exchange
    configuration (the runner draws it at a positive isovector fraction).
    ``None`` or the vacuum configuration gives the event above.  Under the
    isovector exchange each remnant takes its line's charge and isospin
    through :func:`~na49_mixing.isovector_exchange.select_line_remnant`
    (the same one uniform per line), and an excited remnant cascades from
    the line's root family; the draw budget is unchanged.

    *exact_range* (2026-10-05) draws the one transfer of the two-component
    variant (a 2 -> 2 final state, ``t1 = t2``) on the reachable window of
    the realized pair (1 uniform on ``exchange_kinematics`` instead of 2).
    The class with the central source keeps its construction (the central
    system at rest, one production angle) and refuses the option until a
    recoiling central construction exists.

    *line_thresholds* (2026-10-06; code-physics review M-6) starts each
    excited remnant's mass law at its line's own lowest direct threshold
    (:func:`~na49_mixing.isovector_exchange.line_mass_threshold`) instead
    of the shared p pi0 edge; the recorded ``remnant_windows`` carry the
    edge, so the reweighting follows.  Same draw budget.

    *duality* (2026-10-12) is the pair ``(Md, Ms)`` of
    :mod:`na49_mixing.nd_duality` (``Ms`` ``None`` for no redraw): the
    vacuum lines below Md decay as the low-mass N* and Delta records, and
    an excited vacuum line drawn below Ms takes its mass from the records'
    line shapes by the measure-preserving map of its continuum draw (no
    draw on ``spectral_state``); ``duality_edges`` records the shape edge
    of each redrawn line for the reweighting.  A mapped line takes one
    more uniform on ``decay_channel``.

    ``ValueError`` on a parameter surface outside the card-4 mass-law
    parameters, a weights mapping outside the card-3 kinds, or an
    unknown catalog/family surface (checked, never repaired).
    """

    if not isinstance(params, MassLawParams):
        raise ValueError(
            f"generate_nd: expected MassLawParams, got {type(params)!r}")
    if not isinstance(catalog, HadronCatalog):
        raise ValueError(
            f"generate_nd: expected a HadronCatalog, got {type(catalog)!r}")
    for label in ("spectral_state", "exchange_kinematics",
                  "decay_channel", "decay_kinematics"):
        if label not in SUBSTREAM_LABELS:
            raise ValueError(
                f"generate_nd: the substream label {label!r} is not in "
                f"the closed label set {SUBSTREAM_LABELS}")
    # the isospin mixture of the non-diffractive lines (2026-10-10)
    fams = non_diffractive_families(fams)

    # (1) the two card-3 remnant selections (2 uniforms, spectral_state;
    # projectile side first, then the target side; the requested-kind
    # validation anchor is the intact-proton kind, the actual kind is
    # drawn over the caller's weights)
    m_other_p = _heaviest_intact_or_low_mass(catalog) + PI0_MASS
    isovector = exchange is not None and not exchange.is_vacuum
    lo_p = lo_t = None
    if line_thresholds:
        lo_p = line_mass_threshold(fams, catalog, exchange.projectile if isovector else None)
        lo_t = line_mass_threshold(fams, catalog, exchange.target if isovector else None)
    dual = None
    shape_mass = None
    edges = [math.nan, math.nan]
    if duality is not None:
        records = duality_records(catalog, duality_line_shapes)
        dual = (float(duality[0]), records)
        shape_mass = None if duality[1] is None else float(duality[1])

    def _sampler(m_other, m_lo, slot):
        inner = make_remnant_mass_sampler(params, rng_spectral_state, m_other, m_lo=m_lo)
        if dual is None or shape_mass is None or isovector:
            return inner

        def sampler(m_low, m_high):
            # the line shapes below Ms (2026-10-12): the measure-preserving
            # map of the continuum draw, no further uniform
            mass = inner(m_low, m_high)
            lo_window, hi = remnant_excitation_window(m_other, m_lo=m_lo)
            lo = max(float(m_low), lo_window)
            edge = min(shape_mass, hi)
            redrawn = (duality_redraw_mass_shapes if duality_redraw_shapes else duality_redraw_mass)(
                mass, lo, edge, params.epsilon, records)
            if redrawn is None:
                return mass
            edges[slot] = edge
            return redrawn
        return sampler
    if isovector:
        remnant_p = select_line_remnant(
            weights, catalog, rng_spectral_state,
            make_remnant_mass_sampler(params, rng_spectral_state, m_other_p, m_lo=lo_p), exchange.projectile)
        remnant_t = select_line_remnant(
            weights, catalog, rng_spectral_state,
            make_remnant_mass_sampler(params, rng_spectral_state, remnant_p.mass, m_lo=lo_t), exchange.target)
    else:
        remnant_p = select_remnant(
            KIND_INTACT_PROTON, weights, catalog, rng_spectral_state,
            _sampler(m_other_p, lo_p, 0),
            spin_isospin=low_mass_spin_isospin)
        remnant_t = select_remnant(
            KIND_INTACT_PROTON, weights, catalog, rng_spectral_state,
            _sampler(remnant_p.mass, lo_t, 1),
            spin_isospin=low_mass_spin_isospin)
    # the excitation windows of the two remnant draws, recorded for the
    # exact exponent reweighting of the excited kinds (2026-09-29)
    remnant_windows = (remnant_excitation_window(m_other_p, m_lo=lo_p),
                       remnant_excitation_window(remnant_p.mass, m_lo=lo_t))

    # (2) the exchange kinematics (5 uniforms: t_1, t_2, M_c^2 in card
    # 4's order; the closure is exact by construction; 3 uniforms, t_1
    # and t_2, without the central source)
    if central_recoil and not central:
        raise ValueError("generate_nd: central_recoil needs the central source")
    if central_epsilon is not None and not central:
        raise ValueError("generate_nd: the power central law needs the central source")
    if exact_range and central:
        raise ValueError(
            "generate_nd: the exact transfer window is defined for the two-component variant; the class with "
            "the central source keeps the central system at rest and needs a recoiling construction")
    if central:
        m2_min, m2_max = nd_limits(remnant_p.mass, remnant_t.mass, SQRT_S)
    slope_factor = None
    if slope_mass_exponent is not None and not exact_range:
        raise ValueError("generate_nd: the factorized vertex slopes need the exact transfer window")
    if exact_range:
        t_lo, t_hi = pp_t_window(remnant_p.mass * remnant_p.mass, remnant_t.mass * remnant_t.mass)
        t_params = params
        if slope_mass_exponent is not None:
            # the factorized vertex slopes (2026-10-08)
            slope_factor = vertex_slope_factor(remnant_p.mass, remnant_t.mass, slope_mass_exponent)
            t_params = MassLawParams(epsilon=params.epsilon, b_sd=params.b_sd * slope_factor)
        transfer1 = draw_t_window(t_lo, t_hi, t_params, rng_exchange_kinematics)
        transfer2 = transfer1
    else:
        t1_min = t_range(remnant_p.mass * remnant_p.mass, PROTON_MASS,
                         SQRT_S)[0]
        transfer1 = draw_t(t1_min, params, rng_exchange_kinematics)
        t2_min = t_range(remnant_t.mass * remnant_t.mass, PROTON_MASS,
                         SQRT_S)[0]
        transfer2 = draw_t(t2_min, params, rng_exchange_kinematics)
    transfer_c = transfer1 + transfer2
    if central:
        if central_epsilon is not None:
            # the power central law (2026-10-07): (M_c^2)^(-1-eps_c) on the
            # window, its own exponent, no transfer coupling; the same two
            # uniforms as the committed draw
            m2_central = draw_central(m2_min, m2_max, MassLawParams(epsilon=float(central_epsilon), b_sd=params.b_sd),
                                      0.0, rng_exchange_kinematics)
        else:
            m2_central = draw_central(m2_min, m2_max, params, transfer_c,
                                      rng_exchange_kinematics)
        central_mass = math.sqrt(m2_central)

        # (3) the exact three-system solve: the CM four-momenta for the
        # cascades; under central_recoil (2026-10-07, Stage B) the recoiling
        # solve, with the azimuth of line 2 as one more uniform on
        # exchange_kinematics (after the central mass draw)
        if central_recoil:
            phi2 = 2.0 * math.pi * float(rng_exchange_kinematics.random())
            system1_cm, system2_cm, central_cm = _cm_triplet_recoil(
                remnant_p.mass, float(transfer1), remnant_t.mass, float(transfer2), central_mass, phi2)
        else:
            system1_cm, system2_cm, central_cm = _cm_triplet(
                remnant_p.mass, float(transfer1), remnant_t.mass, central_mass,
                SQRT_S)
    else:
        central_mass = 0.0
        system1_cm, system2_cm = _cm_pair(
            remnant_p.mass, float(transfer1), remnant_t.mass, SQRT_S)

    # (4) the three systems decay through the frozen simplified cascade
    # (projectile line, target line, central source, on the shared
    # decay substreams in that order)
    projectile_finals_cm = _decay_remnant(
        remnant_p, system1_cm, catalog, rng_decay_channel,
        rng_decay_kinematics, PROJECTILE_LINE_LABEL,
        fams, flavor, ledger,
        line_isospin=exchange.projectile.isospin if isovector else None,
        duality=None if isovector else dual)
    target_finals_cm = _decay_remnant(
        remnant_t, system2_cm, catalog, rng_decay_channel,
        rng_decay_kinematics, TARGET_LINE_LABEL,
        fams, flavor, ledger,
        line_isospin=exchange.target.isospin if isovector else None,
        duality=None if isovector else dual)
    central_finals_cm = (_decay_central(
        central_cm, central_mass, catalog, fams, rng_decay_channel,
        rng_decay_kinematics, flavor=flavor, ledger=ledger)
        if central else ())

    # (5) the lab assembly (one exact forward boost per four-momentum)
    boost = _lab_boost_factory()
    final_state = tuple(
        FinalHadron(
            pdg_id=h.pdg_id,
            four_momentum=boost(h.four_momentum),
            mass=h.mass,
            qn=h.qn,
            line=h.line,
        )
        for h in tuple(projectile_finals_cm) + tuple(target_finals_cm)
        + tuple(central_finals_cm)
    )

    # (6) the ledger lines: the two outgoing baryonic systems (projectile
    # side first), both inheriting the proton quantum numbers after the
    # vacuum exchanges' zero transfers, and the central source with
    # CENTRAL_QN exactly.
    lines = (
        LineState(B=1, Q=1, S=0, I3x2=1, four_momentum=boost(system1_cm)),
        LineState(B=1, Q=1, S=0, I3x2=1, four_momentum=boost(system2_cm)),
    ) if not isovector else exchange.line_states(boost(system1_cm), boost(system2_cm))
    central_source = (LineState(B=0, Q=0, S=0, I3x2=0,
                                four_momentum=boost(central_cm))
                      if central else None)

    return NDEvent(
        class_label=ND_CLASS_LABEL,
        remnants=(remnant_p, remnant_t),
        central_mass=float(central_mass),
        t_c=float(transfer_c),
        t1=float(transfer1),
        t2=float(transfer2),
        lines=lines,
        central_source=central_source,
        exchanges=(ExchangeRecord(name=VACUUM_EXCHANGE_NAME,
                                  transfer=ZERO_TRANSFER),
                   ExchangeRecord(name=VACUUM_EXCHANGE_NAME,
                                  transfer=ZERO_TRANSFER)) if not isovector else tuple(
            ExchangeRecord(name=name, transfer=transfer) for name, transfer in exchange.records()),
        final_state=final_state,
        remnant_windows=remnant_windows,
        central_component=bool(central),
        exchange_configuration=exchange,
        slope_factor=slope_factor,
        duality_edges=None if duality is None else (float(edges[0]), float(edges[1])),
    )


def _never_mass_sampler(m_low: float, m_high: float) -> float:
    """The remnant selection's mass sampler of the frozen Round 5 split:
    the nominal split draws only the intact and low-mass kinds, which
    take their masses without the caller's law (the high-mass weight is
    the pre-registered 0.0), so a call is a contract violation."""

    raise AssertionError(
        "generate_nd: the nominal non-diffractive split never draws "
        f"the high-mass continuum kind, got ({m_low!r}, {m_high!r})")


def _heaviest_intact_or_low_mass(catalog: HadronCatalog) -> float:
    """The largest mass the other line can take without excitation (the
    proton or the heaviest low-mass record), the reservation of the
    projectile-side excitation window."""
    return max([PROTON_MASS] + [float(r.pole_mass) for r in low_mass_records(catalog)])


def remnant_excitation_window(m_other: float, sqrt_s: float = SQRT_S, *,
                              m_lo: float | None = None) -> tuple[float, float]:
    """The excited-remnant mass window ``(lo, hi)`` [GeV] of
    :func:`make_remnant_mass_sampler` for the other line at *m_other*:
    ``lo = m_p + m_pi0`` (above the unphysical sub-threshold band) and
    ``hi = sqrt_s - m_other - m_pi0`` (the other remnant and a central
    window of at least one neutral pion).  Recorded on the event so the
    exponent reweighting can normalize the excited kind's law exactly.
    *m_lo* (2026-10-06) replaces the lower edge by the line's own lowest
    direct threshold (``generate_nd(..., line_thresholds=True)``)."""
    lo = PROTON_MASS + PI0_MASS if m_lo is None else float(m_lo)
    return (lo, float(sqrt_s) - float(m_other) - PI0_MASS)


def make_remnant_mass_sampler(params: MassLawParams, rng, m_other: float,
                              sqrt_s: float = SQRT_S, *, m_lo: float | None = None):
    """The excited-remnant mass law of the Round 5 follow-up card
    TASK-20260928-r5-gate5-followup-nd-remnant-excitation: the
    single-diffraction marginal law ``(M^2)^(-1-epsilon)`` at zero
    transfer (:func:`~na49_mixing.topology_mass_laws.draw_marginal`, the
    same law and exponent as the diffractive excitations) on the window
    ``[(m_p + m_pi0)^2, (sqrt_s - m_other - m_pi0)^2]``: above the
    unphysical sub-threshold band and leaving the other remnant at
    *m_other* and a central window of at least one neutral pion.  The
    projectile side passes the heaviest non-excited target kind plus a
    pion as *m_other* (so every target kind fits and the target's
    excitation window is never closed); the target side passes the drawn
    projectile mass.  The
    draws come from *rng* (the ``spectral_state`` substream that also
    carries the kind draw; the law consumes its documented two uniforms
    per excited remnant).  A window closed by a heavy other remnant
    returns the lower edge without a draw.  Of the ``(lo, hi)`` arguments
    of the :func:`~na49_mixing.topology_remnants.select_remnant`
    protocol only the floor applies; the upper argument is the spectral
    family's support edge, which the excitation law does not observe
    (the single-diffraction excitation reaches ``sqrt_s - m_remnant``)."""

    def sampler(m_low: float, m_high: float) -> float:
        # the protocol's upper argument is the spectral family's support
        # edge (CONTINUUM_M_MAX); the excitation law is not capped by it,
        # exactly as the single-diffraction excitation is not
        lo_window, hi = remnant_excitation_window(m_other, sqrt_s, m_lo=m_lo)
        lo = max(float(m_low), lo_window)
        if not hi > lo:
            return lo
        m2 = draw_marginal(lo * lo, hi * hi, params, 0.0, rng)
        return math.sqrt(float(m2))

    return sampler


# ----------------------------------------------------------------------
# The central-neutrality check (plan line 849; card 2 rule (e))
# ----------------------------------------------------------------------
def check_central_neutrality(event: NDEvent) -> None:
    """Exact integer check of the central source's neutrality (plan
    line 849; card 2 rule (e)): the central source carries
    :data:`CENTRAL_QN = (0, 0, 0, 0)` exactly and the central source's
    decay vertex on the record closes locally to ``(0, 0, 0)`` (the
    ``(B, Q, S, 2I3)`` sum of the central line's final-state hadrons
    equals the source's quantum numbers exactly).

    Raises :class:`CentralNeutralityViolation` naming the offending
    component or vertex on the first violation; a passing event returns
    ``None``.  The comparison is exact integer arithmetic on the
    doubled-I3 ``(B, Q, S, 2I3)`` tuples; the check consumes zero
    random numbers.
    """

    if not isinstance(event, NDEvent):
        raise ValueError(
            f"check_central_neutrality: expected an NDEvent, got "
            f"{type(event)!r}")
    if event.central_source is None:
        # the two-component variant: no central source, and no hadron on
        # the central line
        stray = [h for h in event.final_state if h.line == CENTRAL_LINE_LABEL]
        if stray:
            raise CentralNeutralityViolation(
                f"an event without a central source carries {len(stray)} "
                "final-state hadron(s) on the central line")
        return None
    source_qn = event.central_source.qn
    for label, value, wanted in zip(("B", "Q", "S", "2I3"), source_qn,
                                    CENTRAL_QN):
        if value != wanted:
            raise CentralNeutralityViolation(
                f"central source carries {label} = {value} against the "
                f"central quantum numbers {CENTRAL_QN} (plan line 849: "
                "for vacuum exchange a central excitation must have "
                "B = Q = S = 0; card 2 rule (e))")
    daughters = [h for h in event.final_state
                 if h.line == CENTRAL_LINE_LABEL]
    daughter_sum = (
        sum(int(h.qn[0]) for h in daughters),
        sum(int(h.qn[1]) for h in daughters),
        sum(int(h.qn[2]) for h in daughters),
        sum(int(h.qn[3]) for h in daughters),
    )
    if daughter_sum != CENTRAL_QN:
        raise CentralNeutralityViolation(
            "the central decay vertex is open: the central line's "
            f"{len(daughters)} final-state hadrons sum to (B, Q, S, 2I3) "
            f"{daughter_sum} against the central source "
            f"{tuple(source_qn)} (plan line 849; card 2 rule (e): every "
            "central decay vertex closes locally to (0, 0, 0))")
    return None


# ----------------------------------------------------------------------
# The rapidity helper (the R50 gap quantities, the card-5 helper)
# ----------------------------------------------------------------------
def _event_rapidities(event: NDEvent) -> dict[str, list[float]]:
    """The lab rapidities of the event's final-state hadrons, grouped by
    line of origin (the runner's R50 gap quantities consume these
    read-only; the card pre-registers no gap field on the record)."""

    if not isinstance(event, NDEvent):
        raise ValueError(
            f"_event_rapidities: expected an NDEvent, got {type(event)!r}")
    grouped: dict[str, list[float]] = {}
    for label in sorted(ND_LINE_LABELS):
        grouped[label] = []
    for hadron in event.final_state:
        grouped[hadron.line].append(_rapidity(hadron.four_momentum))
    return grouped
