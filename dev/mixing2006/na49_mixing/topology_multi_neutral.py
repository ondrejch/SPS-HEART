"""The multiple-neutral-excitation extension of the non-diffractive
class (Round 5, card 8).

``TASK-20260925-r5-multiple-neutral-excitations``; plan section 22 task
32 (line 1757); plan section 11.1 item 6 (line 829).  The card
implements the multiple-neutral-excitation class of the initial
topology basis as an **unretained** extension of card 7's neutral
central source:

> 6. **Multiple neutral excitations:** the excitation count remains
>    bounded.

The pre-registered instantiation (the card): the excitation count ``n``
is drawn on the ``topology_selection`` substream as a two-point
distribution ``{1, 2}``; ``P(n = 2)`` is bounded to ``[0, 0.5]``
(:data:`COUNT_BND`) with the unretained default ``0.0``
(:data:`DEFAULT_P2`; the registry flag ``retained: False``), and the
class enters a run only if the R53 multiplicity-tail trigger fires
(index scope item 5: the R52 zero-versus-one comparison decides
retention of the neutral sector, and the R53 trigger, if it fires,
turns the flag on with a bounded ``P(n >= 2)`` fit at the R53 step, not
before).  At ``n = 1`` the event is card 7's ND record bit-identical,
field for field, with the count draw as the only additional uniform
(drawn on ``topology_selection`` before card 7's draw order).  At
``n = 2`` the two neutral excitations ``(B, Q, S) = (0, 0, 0)`` each
decay through the frozen simplified cascade; the baryonic lines and
the remnant selections are card 7's.  The bounded count (``n`` in
``{1, 2}``) is the "remains bounded" requirement of plan line 829; no
count above 2 is represented in Round 5.

Draw budget (the card's pre-registered counts)
----------------------------------------------

The count draw is exactly one uniform on the closed
``topology_selection`` substream, drawn before card 7's draw order
(the count draw is the only additional uniform at ``n = 1``).  At
``n = 1`` the rest of the event is exactly card 7's
:func:`~na49_mixing.topology_non_diffractive.generate_nd` draw order:
exactly two uniforms on the closed ``spectral_state`` substream (one
card-3 remnant selection per baryonic line, the nominal split), exactly
four uniforms on the closed ``exchange_kinematics`` substream (the
``t_1`` draw, the ``t_2`` draw, then the ``M_c1^2`` central draw of
card 4 -- two uniforms -- in that order), and the frozen Round 4 budget
of one ``decay_channel`` plus two ``decay_kinematics`` uniforms per
decay vertex for the three systems.  At ``n = 2`` the second central
source is drawn on ``exchange_kinematics`` after card 7's order (the
``t_2c`` draw -- one uniform -- then the ``M_2c^2`` draw of card 4's
central law on the residual limits -- two uniforms per attempt -- with
the exact four-system closure ``M_1 + M_2 + M_c1 + M_c2 <= sqrt(s)``
by rejection: an attempt whose drawn configuration violates the
closure beyond the pinned joint-closure tolerance is retried, and
:class:`~na49_mixing.topology_mass_laws.PhaseSpaceRejectionExceeded`
is raised after
:data:`~na49_mixing.topology_mass_laws.REJECTION_CAP` consecutive
rejected attempts), and the frozen per-vertex decay budget runs for
the four systems in the fixed order projectile line, target line,
central source 1, central source 2.  The ``n = 2`` exchange budget is
therefore ``4 + 1 + 2 * attempts`` uniforms per event (the card's
decomposition: the ``t_2c`` draw plus the two-uniform ``M_2c^2`` draw
per attempt).

Exact kinematics at ``n = 2`` (the composite-central convention)
----------------------------------------------------------------

The operative four-system solve extends card 7's documented
three-system convention: the two neutral central sources are produced
as the central pair at rest in the overall center-of-mass frame (each
source carries zero momentum in the overall CM, so the pair's total
four-momentum is ``(M_c1 + M_c2, 0, 0, 0)`` and the pair carries zero
relative momentum -- the central-production idealization of card 7 at
the two-source extension), the two baryonic systems recoil back-to-back
against the central pair with the exact two-body kinematics of the
``(M_1, M_2)`` pair at the pair invariant ``sqrt_s - M_c1 - M_c2``,
and the production angle follows from the drawn projectile-side
transfer through the same documented invariant relation card 7 uses on
the first baryonic line,

    ``t_1 = m_p^2 + M_1^2 - 2 (E_p* E_1* - p* q* cos(theta*))``,

(the sliver saturation of the angle at the clamped configuration is
card 7's committed pattern).  The drawn transfers ``(t_1, t_2, t_2c)``
are the exchange-kinematics record (card 4's factorized effective law;
the identifiable-effective-combinations rule, plan line 894): the
recorded values stay the drawn values, the operative solve realizes
the projectile-side transfer through the documented invariant, the
realized target-side transfer follows the exact back-to-back coupling,
and the drawn ``t_2`` acts through the total transfer ``t_c = t_1 +
t_2`` of the central-production law (the record's ``t_c`` field; the
drawn ``t_2c`` is the second source's exchange-kinematics variable on
the residual law).  The event's four-momentum closure against the
incoming lab sum holds by construction within ``1e-9`` relative.

Frame convention (card 7's convention): all systems decay through the
frozen simplified cascade with the Round 4 modules consumed read-only,
the cascades run in the overall center-of-mass frame of the pinned
beam configuration, and the collected final states are boosted into
the lab frame once, at assembly, with the exact forward boost.

The second source's residual-limits law
---------------------------------------

The second central source's mass is drawn from card 4's
central-production law
(:func:`~na49_mixing.topology_mass_laws.draw_central`) on the residual
phase space: ``M_2c^2`` in ``[m_pi0^2, (sqrt(s) - M_1 - M_2 - M_c1)^2]``
(:func:`nd_multi_limits`), at the drawn total transfer ``t_c`` of the
two exchanges.  The exact four-system closure
``M_1 + M_2 + M_c1 + M_c2 <= sqrt(s)`` is enforced by rejection on the
pinned cap :data:`~na49_mixing.topology_mass_laws.REJECTION_CAP`: the
drawn window's upper edge places the closure exactly, so an accepted
draw satisfies the closure by construction and the rejection covers
the float dust of the drawn window at the upper edge (the card-4
joint-closure tolerance; an attempt whose drawn ``M_2c`` leaves the
four-system closure beyond the tolerance is retried).  The transfer
``t_2c`` of the second source's exchange is drawn from ``exp(B_SD t)``
on the exact two-body range
:func:`~na49_mixing.topology_mass_laws.t_range` computed with the
other baryonic line at the intact proton mass (the factorized
effective law; the exchange slopes act on the exchange kinematics
only; the daughter p_T dynamics comes from the isotropic cascade, which
carries no b_dec form factor on the topology path, review H-6; plan
section 24, line 1866).

Ledger conventions (card 2's records, plan section 11.4)
--------------------------------------------------------

Both neutral central sources carry ``(B, Q, S, 2I3) = (0, 0, 0, 0)``
exactly (plan line 849; card 2 rule (e)) and
:func:`check_multi_central_neutrality` applies card 7's central-
neutrality semantics to every central source.  The baryonic lines
inherit the incoming proton quantum numbers ``(1, 1, 0, 1)`` after the
two vacuum exchanges' zero transfers; the event record carries the two
exchange records in card 1's
:class:`~na49_mixing.topology_elastic.ExchangeRecord` form; card 2's
:func:`~na49_mixing.exchange_ledger.audit_event` consumes the records
read-only with zero violations (the per-vertex local closure of each
central decay is enforced by the cascade machinery's own conservation
check during generation, and the record-level check verifies the
central line's joint closure over the sources).

Registry and retention
----------------------

:data:`REGISTRY_ENTRY` is the class's registry entry: the class label
``"nd_multi"``, the module, the unretained flag (``retained: False``),
the bounded-count flag, the ``P(n = 2)`` bound (:data:`COUNT_BND`),
and the unretained default (:data:`DEFAULT_P2`).  The class is not
part of the default R50 run set (card 9's inelastic exclusion; index
scope item 2: "every class" is the five classes elastic, SD_p, SD_t,
DD, and ND); it enters a run only if the R53 trigger fires.

Hermetic: standard library and numpy only; card 7's module, cards
1-4's modules, and the Round 4 cascade modules are consumed read-only;
no event generation driver, no external tables, no Monte Carlo run
(the acceptance is hermetic: pinned traces, exact quantum-number and
kinematics checks, bit-identity, draw budgets; the runs R50-R55
exercise the class statistically).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .exchange_ledger import LineState
from .hadron_catalog import HadronCatalog
from .rng_substreams import SUBSTREAM_LABELS
from .spectral_density import ContinuumFamily
from .topology_elastic import (
    PROTON_MASS,
    SQRT_S,
    ExchangeRecord,
)
from .topology_mass_laws import (
    PI0_MASS,
    MassLawParams,
    PhaseSpaceRejectionExceeded,
    REJECTION_CAP,
    draw_central,
    draw_t,
    joint_closure,
    t_range,
)
from .topology_non_diffractive import (
    CENTRAL_LINE_LABEL,
    CENTRAL_QN,
    NDEvent,
    PROJECTILE_LINE_LABEL,
    TARGET_LINE_LABEL,
    VACUUM_EXCHANGE_NAME,
    ZERO_TRANSFER,
    _cm_triplet,
    _decay_central,
    _decay_remnant,
    _lab_boost_factory,
    _heaviest_intact_or_low_mass,
    make_remnant_mass_sampler,
    CentralNeutralityViolation,
    generate_nd,
    nd_limits,
    non_diffractive_families,
)
from .topology_remnants import (
    KIND_INTACT_PROTON,
    RemnantSelection,
    select_remnant,
)
from .topology_single_diffraction import FinalHadron

__all__ = [
    "ND_MULTI_CLASS_LABEL",
    "COUNT_BND",
    "DEFAULT_P2",
    "P_N2_TOLERANCE",
    "NDMultiEvent",
    "REGISTRY_ENTRY",
    "check_multi_central_neutrality",
    "generate_nd_multi",
    "nd_multi_kinematics",
    "nd_multi_limits",
    "validate_p_n2",
]


#: The class label of the multiple-neutral-excitation class (the card's
#: frozen record label; plan section 11.1 item 6, line 829).
ND_MULTI_CLASS_LABEL: str = "nd_multi"

#: The upper bound on ``P(n = 2)`` (the card's ``COUNT_BND``; plan line
#: 829 "bounded"; the R53 step fits inside this bound, never above).
COUNT_BND: float = 0.5

#: The unretained default ``P(n = 2)`` (the card's ``DEFAULT_P2``; the
#: class draws no ``n = 2`` events at the default).
DEFAULT_P2: float = 0.0

#: The float tolerance of the bound check (the card's pinned
#: ``0.5 + 1e-12`` refusal edge).
P_N2_TOLERANCE: float = 1e-12

#: The registry entry of the class (the card's pre-registered form; the
#: class is unretained and enters a run only if the R53 trigger fires).
REGISTRY_ENTRY: dict = {
    "class": "nd_multi",
    "module": "na49_mixing.topology_multi_neutral",
    "retained": False,
    "count_bounded": True,
    "p_n2_bound": COUNT_BND,
    "default_p_n2": DEFAULT_P2,
}


@dataclass(frozen=True)
class NDMultiEvent:
    """The frozen multiple-neutral-excitation event record (the card's
    ledger-ready record).

    ``class_label`` is ``"nd_multi"``.  ``count`` is the drawn
    excitation count, ``1`` or ``2`` exactly (bounded, plan line 829).
    ``p_n2`` is the draw probability used.  ``remnants`` is the card-7
    remnant selection pair of the two baryonic lines (projectile side
    first, target side).  ``central_masses`` carries one entry at
    ``n = 1`` and two at ``n = 2`` (GeV).  ``t_c`` is the total
    transfer ``t_1 + t_2`` [GeV^2]; ``t1`` and ``t2`` are the two drawn
    per-exchange transfers (the card-7 record's additive summand
    fields) and ``t_2c`` is the second source's drawn transfer (``0.0``
    at ``n = 1``, where no second source exists).  ``lines`` is the
    card-2 ledger record pair of the two outgoing baryonic lines
    (projectile side first, target side).  ``central_sources`` carries
    the card-2 ledger record of each central source (one at ``n = 1``,
    two at ``n = 2``), each carrying :data:`CENTRAL_QN` exactly with
    its lab four-momentum.  ``exchanges`` is the pair of the class's
    two vacuum exchange records in card 1's
    :class:`~na49_mixing.topology_elastic.ExchangeRecord` form.
    ``final_state`` carries every final-state hadron of the systems
    (the projectile line, the target line, the central source or
    sources, in that order).
    """

    class_label: str
    count: int
    p_n2: float
    remnants: tuple[RemnantSelection, RemnantSelection]
    central_masses: tuple[float, ...]
    t_c: float
    t1: float
    t2: float
    t_2c: float
    lines: tuple[LineState, LineState]
    central_sources: tuple[LineState, ...]
    exchanges: tuple["ExchangeRecord", "ExchangeRecord"]
    final_state: tuple[FinalHadron, ...]


# ----------------------------------------------------------------------
# The bound check (the card's constructor discipline)
# ----------------------------------------------------------------------
def validate_p_n2(p_n2: float) -> float:
    """The ``P(n = 2)`` validation (the card's bound check, enforced in
    the class's event constructor :func:`generate_nd_multi`): the value
    must be finite and inside the bounded interval
    ``[0, COUNT_BND + P_N2_TOLERANCE]``.

    ``ValueError`` on a non-finite value or a ``P(n = 2)`` above
    ``0.5 + 1e-12`` (plan line 829 "bounded"; checked, never repaired).
    The negative-side float dust (a value in ``[-P_N2_TOLERANCE, 0)``)
    is clamped to ``0.0`` (the unretained default's own value); any
    value below the tolerance is refused.
    """

    value = float(p_n2)
    if not math.isfinite(value):
        raise ValueError(
            "validate_p_n2: the draw probability P(n = 2) must be "
            f"finite, got {p_n2!r}")
    if value < -P_N2_TOLERANCE:
        raise ValueError(
            "validate_p_n2: the draw probability P(n = 2) must be "
            f"non-negative, got {p_n2!r}")
    if value > COUNT_BND + P_N2_TOLERANCE:
        raise ValueError(
            "validate_p_n2: the draw probability P(n = 2) exceeds the "
            f"bound COUNT_BND = {COUNT_BND!r}: got {p_n2!r} above the "
            f"refusal edge {COUNT_BND + P_N2_TOLERANCE!r} (plan line "
            "829: the excitation count remains bounded; checked, never "
            "repaired)")
    return 0.0 if value < 0.0 else value


# ----------------------------------------------------------------------
# The exact residual limits (the card's nd_multi_limits)
# ----------------------------------------------------------------------
def nd_multi_limits(m1: float, m2: float, m_c1: float,
                    sqrt_s: float) -> tuple[float, float]:
    """Exact second-source limits ``(m2_min, m2_max)`` in GeV^2 of the
    residual phase space: ``(m_pi0^2, (sqrt_s - M_1 - M_2 - M_c1)^2)``.

    ``m_pi0`` is the committed catalog pion mass
    (:data:`~na49_mixing.topology_mass_laws.PI0_MASS`, the card-4
    resolution precedent) and the upper edge places the drawn
    four-system closure ``M_1 + M_2 + M_c1 + M_c2 <= sqrt_s`` exactly
    (:func:`~na49_mixing.topology_mass_laws.joint_closure`'s pinned
    ``1e-12`` tolerance covers the float dust of the drawn window; the
    rejection on the pinned cap is the exact-closure enforcement).
    ``ValueError`` on a non-positive or non-finite mass, a non-finite
    ``sqrt_s``, or a system closed below the pion threshold
    (``sqrt_s <= M_1 + M_2 + M_c1 + m_pi0``: an empty residual window,
    checked, never repaired).
    """

    mass1 = float(m1)
    mass2 = float(m2)
    mass_c1 = float(m_c1)
    root_s = float(sqrt_s)
    for name, value in (("m1", m1), ("m2", m2), ("m_c1", m_c1)):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(
                "nd_multi_limits: every mass must be a positive finite "
                f"GeV value, got {name} = {value!r}")
    if not math.isfinite(root_s):
        raise ValueError(
            f"nd_multi_limits: sqrt_s must be finite, got {sqrt_s!r}")
    m2_min = PI0_MASS * PI0_MASS
    m2_max = (root_s - mass1 - mass2 - mass_c1) ** 2
    if not (root_s - mass1 - mass2 - mass_c1) > PI0_MASS or not m2_max > m2_min:
        raise ValueError(
            "nd_multi_limits: the residual window is empty: sqrt_s - "
            f"M_1 - M_2 - M_c1 = {root_s - mass1 - mass2 - mass_c1!r} "
            "GeV leaves no room above the neutral-pion threshold "
            f"{PI0_MASS!r} GeV (checked, never repaired)")
    return (m2_min, m2_max)


# ----------------------------------------------------------------------
# The exact four-system solve (the card's nd_multi_kinematics)
# ----------------------------------------------------------------------
def nd_multi_kinematics(
    m1: float,
    t1: float,
    m2: float,
    t2: float,
    m_c1: float,
    m_c2: float,
    sqrt_s: float,
) -> tuple[tuple[float, float, float, float],
           tuple[float, float, float, float],
           tuple[float, float, float, float],
           tuple[float, float, float, float]]:
    """Exact outgoing four-momenta ``(system_1, system_2, central_1,
    central_2)`` in the lab frame from ``(M_1, t_1, M_2, t_2, M_c1,
    M_c2, sqrt_s)``, projectile side first, target side second, central
    source 1 third, central source 2 fourth.

    The exact four-system solve of the pinned beam configuration (the
    module docstring's composite-central convention, card 7's
    three-system solve at the two-source extension): the central pair
    at rest in the overall CM (each source carries zero momentum
    there), the two baryonic systems back-to-back against the central
    pair with the exact two-body kinematics of the ``(M_1, M_2)`` pair
    at the pair invariant ``sqrt_s - M_c1 - M_c2``, the production
    angle from the drawn projectile-side transfer through the
    invariant relation on the first baryonic line, and the lab boost
    with the exact fixed-target CM velocity.  The production plane is
    the x-z plane.  Energy-momentum closure holds within ``1e-9``
    relative by construction (the CM quadruplet sums to the total CM
    four-momentum and the boost maps it to the incoming lab sum).

    The drawn pair ``(t_1, t_2)`` and the second source's drawn
    ``t_2c`` are the exchange-kinematics record (card 4's factorized
    effective law); the operative solve realizes the projectile-side
    transfer through the documented invariant, the realized
    target-side transfer follows the exact back-to-back coupling, and
    the recorded values stay the drawn ones (the ``t_2`` argument and
    the caller's ``t_2c`` are consumed by the interface and the
    validation; the solved configuration is fixed by ``(M_1, t_1, M_2,
    M_c1, M_c2, sqrt_s)``).  The angle saturates at the clamped
    configuration on the drawn slivers beyond the exact reachable
    ``t_1`` range, and a configuration at the drawn-closure edge
    resolves at the exact at-rest configuration (card 7's committed
    resolutions).  ``ValueError`` on non-positive masses, a positive
    or non-finite drawn transfer, or a four-system closure beyond the
    joint-closure tolerance (checked, never repaired).
    """

    transfer1 = float(t1)
    transfer2 = float(t2)
    if not (math.isfinite(transfer1) and transfer1 <= 0.0):
        raise ValueError(
            "nd_multi_kinematics: the momentum transfer t1 must be a "
            f"non-positive finite GeV^2 value, got {t1!r}")
    if not (math.isfinite(transfer2) and transfer2 <= 0.0):
        raise ValueError(
            "nd_multi_kinematics: the momentum transfer t2 must be a "
            f"non-positive finite GeV^2 value, got {t2!r}")
    mass1 = float(m1)
    mass2 = float(m2)
    mass_c1 = float(m_c1)
    mass_c2 = float(m_c2)
    root_s = float(sqrt_s)
    for name, value in (("M_1", m1), ("M_2", m2), ("M_c1", m_c1),
                        ("M_c2", m_c2)):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(
                "nd_multi_kinematics: every system mass must be a "
                f"positive finite GeV value, got {name} = {value!r}")
    if not math.isfinite(root_s):
        raise ValueError(
            f"nd_multi_kinematics: sqrt_s must be finite, got {sqrt_s!r}")
    if not joint_closure(mass1, mass2, root_s, mass_c1 + mass_c2):
        raise ValueError(
            "nd_multi_kinematics: the four-system closure is violated: "
            f"M_1 + M_2 + M_c1 + M_c2 = "
            f"{mass1 + mass2 + mass_c1 + mass_c2!r} GeV exceeds "
            f"sqrt_s = {root_s!r} GeV beyond the joint-closure "
            "tolerance (checked, never repaired)")

    # The composite-central solve: card 7's documented three-system
    # solve with the central pair's composite mass M_c1 + M_c2; the
    # composite resolves into the two sources at rest in the overall
    # CM (zero relative momentum in the pair).
    system1_cm, system2_cm, composite_cm = _cm_triplet(
        mass1, transfer1, mass2, mass_c1 + mass_c2, root_s)
    e_pair = composite_cm[0]
    central1_cm = (mass_c1, 0.0, 0.0, 0.0)
    central2_cm = (mass_c2, 0.0, 0.0, 0.0)
    # the composite's CM energy is the source-mass sum exactly (the
    # pair at rest); the float dust of the composition is covered by
    # the joint-closure tolerance checked above
    if abs(e_pair - (mass_c1 + mass_c2)) > 1e-9:
        raise ValueError(
            "nd_multi_kinematics: the composite-central solve is "
            f"inconsistent: the pair energy {e_pair!r} GeV disagrees "
            f"with the source-mass sum {mass_c1 + mass_c2!r} GeV "
            "beyond 1e-9 (checked, never repaired)")
    boost = _lab_boost_factory()
    return (boost(system1_cm), boost(system2_cm),
            boost(central1_cm), boost(central2_cm))


# ----------------------------------------------------------------------
# The event generator
# ----------------------------------------------------------------------
def generate_nd_multi(
    params: MassLawParams,
    weights,
    p_n2: float,
    catalog: HadronCatalog,
    fams: list[ContinuumFamily],
    rng_topology_selection,
    rng_spectral_state,
    rng_exchange_kinematics,
    rng_decay_channel,
    rng_decay_kinematics,
    flavor=None,
    ledger=None,
) -> NDMultiEvent:
    """One multiple-neutral-excitation event.

    The card's pre-registered draw budget on the closed substream
    labels (:data:`na49_mixing.rng_substreams.SUBSTREAM_LABELS`): the
    count draw first -- exactly one uniform on ``topology_selection``
    resolved against the bounded *p_n2* (a uniform below ``p_n2``
    selects ``n = 2``, otherwise ``n = 1``; the count draw is the only
    additional uniform at ``n = 1``, drawn before card 7's draw
    order).  At ``n = 1`` the rest of the event is exactly card 7's
    :func:`~na49_mixing.topology_non_diffractive.generate_nd` draw
    order (2 ``spectral_state`` + 4 ``exchange_kinematics`` + the
    frozen per-vertex decay budget for the three systems), and the
    event's shared fields are bit-identical to card 7's ND record on
    the same seed trace.  At ``n = 2`` the second central source is
    drawn on ``exchange_kinematics`` after card 7's order (the
    ``t_2c`` draw, then the ``M_2c^2`` draw on the residual limits
    with the exact four-system closure by rejection on the pinned
    cap), and the frozen Round 4 per-vertex decay budget runs for the
    four systems in the fixed order projectile line, target line,
    central source 1, central source 2 (each source through the frozen
    simplified cascade; the below-threshold branch consumes zero
    draws).  See the module docstring for the full budget.

    ``ValueError`` on a parameter surface outside the card-4 mass-law
    parameters, a weights mapping outside the card-3 kinds, a ``p_n2``
    outside the bounded interval (see :func:`validate_p_n2`), or an
    unknown catalog/family surface (checked, never repaired).
    """

    if not isinstance(params, MassLawParams):
        raise ValueError(
            "generate_nd_multi: expected MassLawParams, got "
            f"{type(params)!r}")
    if not isinstance(catalog, HadronCatalog):
        raise ValueError(
            "generate_nd_multi: expected a HadronCatalog, got "
            f"{type(catalog)!r}")
    p2 = validate_p_n2(p_n2)
    for label in ("topology_selection", "spectral_state",
                  "exchange_kinematics", "decay_channel",
                  "decay_kinematics"):
        if label not in SUBSTREAM_LABELS:
            raise ValueError(
                "generate_nd_multi: the substream label "
                f"{label!r} is not in the closed label set "
                f"{SUBSTREAM_LABELS}")
    # the isospin mixture of the non-diffractive lines (2026-10-10)
    fams = non_diffractive_families(fams)

    # (0) the count draw (1 uniform, topology_selection; the two-point
    # distribution {1, 2}, resolved against the bounded p_n2)
    count = 2 if rng_topology_selection.random() < p2 else 1

    if count == 1:
        # the n = 1 event: exactly card 7's draw order on the four
        # class streams (bit-identical to generate_nd on the same
        # trace); the card-7 ND record's shared fields are carried
        # into the NDMultiEvent surface.
        nd: NDEvent = generate_nd(params, weights, catalog, fams,
                                  rng_spectral_state,
                                  rng_exchange_kinematics,
                                  rng_decay_channel,
                                  rng_decay_kinematics, flavor=flavor, ledger=ledger)
        return NDMultiEvent(
            class_label=ND_MULTI_CLASS_LABEL,
            count=1,
            p_n2=float(p2),
            remnants=nd.remnants,
            central_masses=(nd.central_mass,),
            t_c=nd.t_c,
            t1=nd.t1,
            t2=nd.t2,
            t_2c=0.0,
            lines=nd.lines,
            central_sources=(nd.central_source,),
            exchanges=nd.exchanges,
            final_state=nd.final_state,
        )

    # (1) the n = 2 event: card 7's order through the first central
    # draw, then the residual-phase-space extension on
    # exchange_kinematics.
    remnant_p = select_remnant(
        KIND_INTACT_PROTON, weights, catalog, rng_spectral_state,
        make_remnant_mass_sampler(params, rng_spectral_state,
                                  _heaviest_intact_or_low_mass(catalog) + 2.0 * PI0_MASS))
    remnant_t = select_remnant(
        KIND_INTACT_PROTON, weights, catalog, rng_spectral_state,
        make_remnant_mass_sampler(params, rng_spectral_state,
                                  remnant_p.mass + PI0_MASS))

    m2_min_1, m2_max_1 = nd_limits(remnant_p.mass, remnant_t.mass,
                                    SQRT_S)
    t1_min = t_range(remnant_p.mass * remnant_p.mass, PROTON_MASS,
                     SQRT_S)[0]
    transfer1 = draw_t(t1_min, params, rng_exchange_kinematics)
    t2_min = t_range(remnant_t.mass * remnant_t.mass, PROTON_MASS,
                     SQRT_S)[0]
    transfer2 = draw_t(t2_min, params, rng_exchange_kinematics)
    transfer_c = transfer1 + transfer2
    m2_central_1 = draw_central(m2_min_1, m2_max_1, params, transfer_c,
                                rng_exchange_kinematics)
    central_mass_1 = math.sqrt(m2_central_1)

    # (1b) the second central source on the residual phase space: the
    # t_2c draw (1 uniform) then the M_2c^2 draw (2 uniforms per
    # attempt) with the exact four-system closure by rejection on the
    # pinned cap.
    t2c_min = t_range(remnant_p.mass * remnant_p.mass, PROTON_MASS,
                      SQRT_S)[0]
    transfer_2c = draw_t(t2c_min, params, rng_exchange_kinematics)
    m2_min_2, m2_max_2 = nd_multi_limits(
        remnant_p.mass, remnant_t.mass, central_mass_1, SQRT_S)
    attempts = 0
    while True:
        attempts += 1
        if attempts > REJECTION_CAP:
            raise PhaseSpaceRejectionExceeded(
                "generate_nd_multi: the residual four-system closure "
                f"rejection exceeded the cap of {REJECTION_CAP} "
                "attempts at the parameter point (epsilon="
                f"{params.epsilon!r}, b_sd={params.b_sd!r} GeV^-2) "
                "with limits (m2_min_2="
                f"{m2_min_2!r}, m2_max_2={m2_max_2!r}, sqrt_s="
                f"{SQRT_S!r}); the last attempt drew "
                f"M_c1={central_mass_1!r} GeV")
        m2_central_2 = draw_central(m2_min_2, m2_max_2, params,
                                    transfer_c, rng_exchange_kinematics)
        central_mass_2 = math.sqrt(m2_central_2)
        if joint_closure(remnant_p.mass, remnant_t.mass, SQRT_S,
                         central_mass_1 + central_mass_2):
            break

    # (2) the exact four-system solve (no draws): the CM momenta for
    # the cascades (the frame convention; the lab boost is the caller's
    # assembly step, applied once at (4)/(5))
    system1_cm, system2_cm, composite_cm = _cm_triplet(
        remnant_p.mass, float(transfer1), remnant_t.mass,
        central_mass_1 + central_mass_2, SQRT_S)
    central1_cm = (central_mass_1, 0.0, 0.0, 0.0)
    central2_cm = (central_mass_2, 0.0, 0.0, 0.0)

    # (3) the four systems decay through the frozen simplified cascade
    # (projectile line, target line, central source 1, central source
    # 2, on the shared decay substreams in that order)
    projectile_finals_cm = _decay_remnant(
        remnant_p, system1_cm, catalog, rng_decay_channel,
        rng_decay_kinematics, PROJECTILE_LINE_LABEL,
        fams, flavor, ledger)
    target_finals_cm = _decay_remnant(
        remnant_t, system2_cm, catalog, rng_decay_channel,
        rng_decay_kinematics, TARGET_LINE_LABEL,
        fams, flavor, ledger)
    central1_finals_cm = _decay_central(
        central1_cm, central_mass_1, catalog, fams,
        rng_decay_channel, rng_decay_kinematics, flavor=flavor, ledger=ledger)
    central2_finals_cm = _decay_central(
        central2_cm, central_mass_2, catalog, fams,
        rng_decay_channel, rng_decay_kinematics, flavor=flavor, ledger=ledger)

    # (4) the lab assembly (one exact forward boost per four-momentum)
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
        + tuple(central1_finals_cm) + tuple(central2_finals_cm)
    )

    # (5) the ledger lines: card 7's records for the baryonic lines and
    # the two central sources at CENTRAL_QN exactly.
    lines = (
        LineState(B=1, Q=1, S=0, I3x2=1, four_momentum=boost(system1_cm)),
        LineState(B=1, Q=1, S=0, I3x2=1, four_momentum=boost(system2_cm)),
    )
    central_sources = (
        LineState(B=0, Q=0, S=0, I3x2=0,
                  four_momentum=boost(central1_cm)),
        LineState(B=0, Q=0, S=0, I3x2=0,
                  four_momentum=boost(central2_cm)),
    )

    return NDMultiEvent(
        class_label=ND_MULTI_CLASS_LABEL,
        count=2,
        p_n2=float(p2),
        remnants=(remnant_p, remnant_t),
        central_masses=(float(central_mass_1), float(central_mass_2)),
        t_c=float(transfer_c),
        t1=float(transfer1),
        t2=float(transfer2),
        t_2c=float(transfer_2c),
        lines=lines,
        central_sources=central_sources,
        exchanges=(ExchangeRecord(name=VACUUM_EXCHANGE_NAME,
                                  transfer=ZERO_TRANSFER),
                   ExchangeRecord(name=VACUUM_EXCHANGE_NAME,
                                  transfer=ZERO_TRANSFER)),
        final_state=final_state,
    )


# ----------------------------------------------------------------------
# The central-neutrality check (plan line 849; card 2 rule (e), at the
# n >= 2 extension: every central source is checked)
# ----------------------------------------------------------------------
def check_multi_central_neutrality(event: NDMultiEvent) -> None:
    """Exact integer check of every central source's neutrality (plan
    line 849; card 2 rule (e), card 7's
    :func:`~na49_mixing.topology_non_diffractive.check_central_neutrality`
    semantics applied to every central source of the multi record):
    each central source in ``event.central_sources`` carries
    :data:`CENTRAL_QN = (0, 0, 0, 0)` exactly, and the central line's
    final-state hadrons sum to ``count * CENTRAL_QN`` elementwise (the
    joint local closure of the sources' decay vertices on the record;
    the per-vertex closure of each source's cascade is enforced by the
    cascade machinery's own conservation check during generation).

    Raises :class:`~na49_mixing.topology_non_diffractive.CentralNeutralityViolation`
    naming the offending source, component, or vertex on the first
    violation; a passing event returns ``None``.  The comparison is
    exact integer arithmetic on the doubled-I3 ``(B, Q, S, 2I3)``
    tuples; the check consumes zero random numbers.
    """

    if not isinstance(event, NDMultiEvent):
        raise ValueError(
            "check_multi_central_neutrality: expected an NDMultiEvent, "
            f"got {type(event)!r}")
    for index, source in enumerate(event.central_sources):
        for label, value, wanted in zip(("B", "Q", "S", "2I3"),
                                        source.qn, CENTRAL_QN):
            if value != wanted:
                raise CentralNeutralityViolation(
                    f"central source {index} carries {label} = {value} "
                    f"against the central quantum numbers {CENTRAL_QN} "
                    "(plan line 849: for vacuum exchange a central "
                    "excitation must have B = Q = S = 0; card 2 rule "
                    "(e))")
    daughters = [h for h in event.final_state
                 if h.line == CENTRAL_LINE_LABEL]
    daughter_sum = (
        sum(int(h.qn[0]) for h in daughters),
        sum(int(h.qn[1]) for h in daughters),
        sum(int(h.qn[2]) for h in daughters),
        sum(int(h.qn[3]) for h in daughters),
    )
    wanted = tuple(event.count * component for component in CENTRAL_QN)
    if daughter_sum != wanted:
        raise CentralNeutralityViolation(
            f"the central decay vertices are open: the central line's "
            f"{len(daughters)} final-state hadrons sum to (B, Q, S, "
            f"2I3) {daughter_sum} against the {event.count} central "
            f"sources' joint closure {wanted} (plan line 849; card 2 "
            "rule (e): every central decay vertex closes locally to "
            "(0, 0, 0))")
    return None
