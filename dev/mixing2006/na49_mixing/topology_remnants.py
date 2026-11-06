"""The leading remnant options of the outgoing lines (Round 5, card 3).

``TASK-20260925-r5-intact-low-mass-remnants``; plan section 22 task 33
(line 1758); plan section 11.2 (lines 833-842).  The outgoing
projectile or target line may be:

- an intact proton (:data:`KIND_INTACT_PROTON`, zero pions; this is the
  concrete content of "Do not impose a minimum N-pi excitation on
  every event", plan line 842);
- a neutron only through explicit charge exchange
  (:data:`KIND_NEUTRON_CHARGE_EXCHANGE`; the kind raises
  :class:`ChargeExchangeNotImplemented` naming plan task 47, Round 8,
  consistent with card 2's closed allowed-exchange set ``{VACUUM}`` and
  plan line 838);
- a low-mass baryonic resonance (:data:`KIND_LOW_MASS_BARYONIC_
  RESONANCE`; a committed-catalog ``(1, 1, 0)`` record with
  ``mass_range[0]`` below the frozen kernel boundary, pole mass from
  the catalog);
- a high-mass baryonic continuum excitation (:data:`KIND_HIGH_MASS_
  BARYONIC_CONTINUUM`; a mass from the caller-supplied law over the
  activated baryonic ``(1, 1, 0)`` family of
  ``config/v2_singlet/continuum_resonances.json``).

No minimum N-pi excitation is imposed anywhere in the module (plan
line 842): the intact proton is a legal remnant, and no selection path
counts pions or rejects an event for a small excitation.

The low-mass/high-mass split
----------------------------

"Low mass" and "high mass" split at the frozen kernel boundary
:data:`KERNEL_BOUNDARY = 2.680194110727528 GeV` (the M8-frozen
``continuum.match_mass``, the blend ramp start fixed at B, Round 4
obligation 4; a named constant, never fit).  A catalog ``(1, 1, 0)``
record with ``mass_range[0] < KERNEL_BOUNDARY`` is a low-mass baryonic
resonance (:func:`low_mass_records`); the complement is
:func:`high_mass_records`.  On the committed catalog (pinned at its
digest in the tests) the split is exact: ten low-mass records, an
empty high-mass side (every established ``(1, 1, 0)`` record threshold
lies below B; the minimum distance to the boundary is far beyond
1e-12), and no record sits on the boundary.

The high-mass kind draws its mass from the caller-supplied law over
the activated baryonic ``(1, 1, 0)`` family; the family's recorded
upper support end is :data:`CONTINUUM_M_MAX = 5.0 GeV` (the card-6
surface ``SUPPORT_M_MAX``; the equality is pinned in the tests against
the committed continuum configuration).  The law itself is card 4's
(the Round 5 supplier of the topology-specific mass laws, plan section
11.5): this module attaches it at the frozen interface
``mass_sampler(m_p, m_max)`` and consumes zero draws of its own
substream for it.

Selection semantics
-------------------

:func:`select_remnant` performs one remnant selection and consumes
exactly one uniform on the closed ``spectral_state`` substream
(:data:`na49_mixing.rng_substreams.SUBSTREAM_LABELS`) per selection,
zero elsewhere.  The uniform is the kind draw: it is resolved over the
normalized *weights* (the pre-registered nominal kind weights of the
R50 templates; index item 2: intact 0.5, low-mass 0.5, high-mass 0.0
for single diffraction, the other classes passing their own splits
through the same interface) in the deterministic sorted-kind order.
The *kind* argument is the caller's requested kind: it is validated
against :data:`REMNANT_KINDS` (``ValueError`` naming an unknown kind)
and must name one of the weights' kinds; the neutron request raises
:class:`ChargeExchangeNotImplemented` before any draw (zero uniforms).
A drawn neutron kind (the neutron named with positive weight) raises
:class:`ChargeExchangeNotImplemented` after the kind draw.  The
selection's mass, catalog identity, and quantum numbers follow the
drawn kind: ``m_p`` for the intact proton, the selected record's pole
mass for the low-mass kind, and ``mass_sampler(m_p, m_max)`` for the
high-mass kind.  A caller that fixes its line's remnant passes the
degenerate split ``{kind: 1.0}``; the low-mass record is then drawn
over the low-mass records by the same uniform (the residual fraction
of the kind draw inside the kind's cumulative interval, index
``int(residual * n)`` over the catalog-ordered records), so one
uniform serves the kind draw and the record draw together and the
budget stays at exactly one uniform per selection.

Every remnant carries the doubled-I3 integer quantum numbers
``(B, Q, S, 2I3) = (1, 1, 0, 1)`` (:data:`REMNANT_QN`, card 2's
convention; the Gell-Mann-Nishijima relation makes ``2I3 = 1``
exact for every ``(B, Q, S) = (1, 1, 0)`` record, and the tests pin
the catalog tie).  The proton mass is the catalog value
:data:`PROTON_MASS = 0.93827231 GeV`.

Hermetic: standard library and numpy only; the committed catalog and
continuum configuration are consumed read-only by the tests; no event
generation driver, no external tables, no Monte Carlo run (the
acceptance is hermetic: pinned traces, exact split identities, draw
budgets; the runs R50-R55 exercise the remnants statistically).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from .hadron_catalog import HadronCatalog, HadronRecord

__all__ = [
    "AVAILABLE_REMNANT_KINDS",
    "CHARGE_EXCHANGE_TASK_REF",
    "CONTINUUM_M_MAX",
    "KIND_HIGH_MASS_BARYONIC_CONTINUUM",
    "KIND_INTACT_PROTON",
    "KIND_LOW_MASS_BARYONIC_RESONANCE",
    "KIND_NEUTRON_CHARGE_EXCHANGE",
    "KERNEL_BOUNDARY",
    "REMNANT_KIND_INDEX",
    "PROTON_MASS",
    "PROTON_PDG_ID",
    "REMNANT_KINDS",
    "REMNANT_QN",
    "ChargeExchangeNotImplemented",
    "RemnantSelection",
    "high_mass_records",
    "low_mass_records",
    "select_remnant",
]


#: The closed set of outgoing remnant kinds (plan section 11.2, lines
#: 833-842).  Closed at this round: a new kind files its own card.
KIND_INTACT_PROTON: str = "intact_proton"
KIND_NEUTRON_CHARGE_EXCHANGE: str = "neutron_charge_exchange"
KIND_LOW_MASS_BARYONIC_RESONANCE: str = "low_mass_baryonic_resonance"
KIND_HIGH_MASS_BARYONIC_CONTINUUM: str = "high_mass_baryonic_continuum"

#: The integer index of each remnant kind in the recorded shape arrays
#: (2026-09-29): the run drivers store the drawn kinds per line so that
#: the exponent reweighting can apply the excited kind's own mass law
#: (na49_mixing.topology_mass_laws.log_density_ratio, the ``nd`` branch).
REMNANT_KIND_INDEX: dict = {
    KIND_INTACT_PROTON: 0,
    KIND_LOW_MASS_BARYONIC_RESONANCE: 1,
    KIND_HIGH_MASS_BARYONIC_CONTINUUM: 2,
}


#: The closed remnant-kind set (the four plan section 11.2 options).
REMNANT_KINDS: frozenset[str] = frozenset({
    KIND_INTACT_PROTON,
    KIND_NEUTRON_CHARGE_EXCHANGE,
    KIND_LOW_MASS_BARYONIC_RESONANCE,
    KIND_HIGH_MASS_BARYONIC_CONTINUUM,
})

#: The Round 5 availability set: the three non-neutron kinds (the
#: neutron kind raises :class:`ChargeExchangeNotImplemented`; charge
#: exchange is plan task 47, Round 8, outside card 2's closed
#: allowed-exchange set ``{VACUUM}``, plan line 838).
AVAILABLE_REMNANT_KINDS: frozenset[str] = frozenset({
    KIND_INTACT_PROTON,
    KIND_LOW_MASS_BARYONIC_RESONANCE,
    KIND_HIGH_MASS_BARYONIC_CONTINUUM,
})

#: The frozen kernel boundary B [GeV] (the M8-frozen
#: ``continuum.match_mass``, the blend ramp start fixed at B; Round 4
#: obligation 4).  The low-mass/high-mass split of the committed-catalog
#: ``(1, 1, 0)`` records runs on this value: ``mass_range[0] < B`` is a
#: low-mass baryonic resonance, ``mass_range[0] >= B`` the high-mass
#: side.  A named constant, never fit.
KERNEL_BOUNDARY: float = 2.680194110727528

#: The proton mass, the catalog value (the committed catalog record
#: pdg_id 2212, pole_mass) [GeV].
PROTON_MASS: float = 0.93827231

#: The proton's catalog identity (the committed catalog record pdg_id
#: 2212; the intact-proton remnant's ``pdg_id``).
PROTON_PDG_ID: int = 2212

#: The recorded upper support end of the activated baryonic ``(1, 1,
#: 0)`` continuum family (``baryonic_np``, ``m_max`` in
#: ``config/v2_singlet/continuum_resonances.json``; equal to the card-6
#: surface ``SUPPORT_M_MAX``; the equality is pinned in the tests)
#: [GeV].  The high-mass kind's mass law receives this value as its
#: second argument.
CONTINUUM_M_MAX: float = 5.0

#: The remnant quantum numbers in the doubled-I3 integer convention
#: ``(B, Q, S, 2I3)`` (card 2's convention; every ``(B, Q, S) = (1, 1,
#: 0)`` record carries ``2I3 = 1`` by the Gell-Mann-Nishijima
#: relation).
REMNANT_QN: tuple[int, int, int, int] = (1, 1, 0, 1)

#: The plan reference the neutron refusal names (charge exchange is
#: plan task 47, Round 8).
CHARGE_EXCHANGE_TASK_REF: str = "plan task 47"

#: The validation tolerance of the weights' unit sum (the card's
#: pre-registered bound).
WEIGHTS_SUM_TOL: float = 1e-12


class ChargeExchangeNotImplemented(RuntimeError):
    """Raised when the neutron remnant kind is requested: charge
    exchange is plan task 47 (Round 8) and is outside the Round 5
    closed allowed-exchange set ``{VACUUM}`` (plan line 838)."""


@dataclass(frozen=True)
class RemnantSelection:
    """One selected leading remnant: the drawn kind, its catalog
    identity, its mass [GeV], and its doubled-I3 integer quantum
    numbers ``(B, Q, S, 2I3)``.

    ``pdg_id`` is ``None`` for the continuum draw (the high-mass
    baryonic continuum excitation carries no catalog record); it is
    :data:`PROTON_PDG_ID` for the intact proton and the selected
    record's ``pdg_id`` for the low-mass kind.  ``record_id`` is the
    selected catalog record's name for the low-mass kind and ``None``
    otherwise.
    """

    kind: str
    pdg_id: int | None               # None for the continuum draw
    record_id: str | None            # catalog record name (low-mass)
    mass: float                      # GeV
    qn: tuple[int, int, int, int]    # (B, Q, S, 2I3) = (1, 1, 0, 1)


# ----------------------------------------------------------------------
# The low-mass/high-mass split of the committed catalog
# ----------------------------------------------------------------------
def _baryonic_110_records(catalog: HadronCatalog) -> list[HadronRecord]:
    """The committed-catalog ``(B, Q, S) = (1, 1, 0)`` records in
    catalog order."""

    if not isinstance(catalog, HadronCatalog):
        raise ValueError(
            "remnant catalog split: expected a HadronCatalog, got "
            f"{type(catalog)!r}")
    return [record for record in catalog.records()
            if (record.B, record.Q, record.S) == (1, 1, 0)]


def low_mass_records(catalog: HadronCatalog) -> list[HadronRecord]:
    """The committed-catalog ``(1, 1, 0)`` records with
    ``mass_range[0] < KERNEL_BOUNDARY``, in catalog order: the low-mass
    baryonic resonances of plan line 839 (the pole mass of the selected
    record is the low-mass remnant's mass)."""

    return [record for record in _baryonic_110_records(catalog)
            if record.mass_range[0] < KERNEL_BOUNDARY]


def low_mass_spin_isospin_records(catalog: HadronCatalog) -> list[tuple[HadronRecord, float]]:
    """The low-mass records of a vacuum line under the isoscalar exchange
    (2026-10-08): the I = 1/2 records of :func:`low_mass_records` (the
    proton included), each with the spin multiplicity 2J + 1 as its weight.
    An isoscalar exchange keeps the proton line at I = 1/2, so the
    I = 3/2 records stay on the isovector lines
    (:func:`~na49_mixing.isovector_exchange.line_low_mass_records`)."""

    return [(record, 2.0 * float(record.spin) + 1.0) for record in low_mass_records(catalog)
            if float(record.isospin) == 0.5]


def high_mass_records(catalog: HadronCatalog) -> list[HadronRecord]:
    """The committed-catalog ``(1, 1, 0)`` records with
    ``mass_range[0] >= KERNEL_BOUNDARY``, in catalog order: the exact
    complement of :func:`low_mass_records` (empty on the committed
    catalog; every established ``(1, 1, 0)`` record threshold lies
    below the frozen boundary)."""

    return [record for record in _baryonic_110_records(catalog)
            if record.mass_range[0] >= KERNEL_BOUNDARY]


# ----------------------------------------------------------------------
# The kind draw and the selection
# ----------------------------------------------------------------------
def _draw_kind(weights: Mapping[str, float], u: float
               ) -> tuple[str, float, float]:
    """The cumulative-resolution of one uniform *u* over the weights in
    the deterministic sorted-kind order.

    Returns ``(kind, lower, upper)``: the drawn kind and its cumulative
    interval ``[lower, upper)`` in ``[0, 1]`` (the record draw of the
    low-mass kind reuses the residual fraction of *u* inside the
    interval).  The weights are pre-validated by :func:`select_remnant`.
    The last kind with positive cumulative reach catches the ``u -> 1``
    float dust: a zero-weight kind is never selected.
    """

    cumulative = 0.0
    kinds = sorted(weights)
    for index, kind in enumerate(kinds):
        lower = cumulative
        cumulative += weights[kind]
        if u < cumulative:
            return kind, lower, cumulative
        if index == len(kinds) - 1:
            # the u -> 1 float dust falls to the last kind with a
            # nonempty interval: walk back past zero-weight kinds that
            # sort last (2026-11-04, the code review qwen38fn L20: the
            # dust returned the last kind in sort order even at weight 0)
            last = index
            while last > 0 and not weights[kinds[last]] > 0.0:
                last -= 1
            if last == index:
                return kind, lower, max(cumulative, math.nextafter(1.0, 0.0))
            top = math.fsum(weights[k] for k in kinds[:last + 1])
            return kinds[last], top - weights[kinds[last]], max(top, math.nextafter(1.0, 0.0))
    raise ValueError(
        "select_remnant: the weights mapping carries no kinds to draw "
        "over (the sum validation would already have rejected an empty "
        "mapping)")


def _validate_weights(weights: Mapping[str, float],
                      requested: str | None) -> dict[str, float]:
    """The pre-registered weights validation (no draw consumed):

    - every key is a kind of :data:`REMNANT_KINDS` (``ValueError``
      naming the foreign kind);
    - the weights sum to 1 within :data:`WEIGHTS_SUM_TOL`
      (``ValueError``);
    - every weight is finite and non-negative (``ValueError`` naming
      the kind; a zero weight is legal and never selected, the R50
      nominal high-mass weight is exactly 0.0);
    - the requested kind, when given, names one of the weights' kinds
      (``ValueError`` naming it).
    """

    for kind in weights:
        if kind not in REMNANT_KINDS:
            raise ValueError(
                "select_remnant: the weights naming the kind "
                f"{kind!r} is outside the closed remnant-kind set "
                f"{sorted(REMNANT_KINDS)}")
    total = math.fsum(float(w) for w in weights.values())
    if abs(total - 1.0) > WEIGHTS_SUM_TOL:
        raise ValueError(
            "select_remnant: the weights must sum to 1 within "
            f"{WEIGHTS_SUM_TOL!r}, got {total!r} over "
            f"{sorted(weights)}")
    for kind, weight in weights.items():
        if not math.isfinite(float(weight)) or float(weight) < 0.0:
            raise ValueError(
                "select_remnant: the weight of the kind "
                f"{kind!r} must be a finite non-negative number, got "
                f"{weight!r}")
    if requested is not None and requested not in weights:
        raise ValueError(
            "select_remnant: the requested kind "
            f"{requested!r} does not name one of the weights' kinds "
            f"{sorted(weights)}")
    return {kind: float(weight) for kind, weight in weights.items()}


def _weighted_pick(pool: list[tuple[HadronRecord, float]], residual: float) -> HadronRecord:
    """The record of the cumulative weights reached by *residual* in [0, 1)
    (the last record catches the u -> 1 edge)."""

    if not pool:
        raise ValueError("select_remnant: the I = 1/2 low-mass set is empty")
    target = float(residual) * math.fsum(w for _r, w in pool)
    cumulative = 0.0
    for record, weight in pool:
        cumulative += weight
        if target < cumulative:
            return record
    return pool[-1][0]


def select_remnant(
    kind: str,
    weights: Mapping[str, float],
    catalog: HadronCatalog,
    rng_kind: np.random.Generator,
    mass_sampler: Callable[[float, float], float],
    spin_isospin: bool = False,
) -> RemnantSelection:
    """One leading-remnant selection of plan section 11.2.

    *spin_isospin* (2026-10-08) draws the low-mass record over
    :func:`low_mass_spin_isospin_records` (the I = 1/2 records with weight
    2J + 1) by the same residual uniform; the draw budget is unchanged.

    Exactly one uniform is consumed from *rng_kind* (the closed
    ``spectral_state`` substream) per selection, zero elsewhere.  The
    uniform is the kind draw: it is resolved over the normalized
    *weights* in the deterministic sorted-kind order (the pre-registered
    nominal kind weights of the R50 templates; the degenerate split
    ``{kind: 1.0}`` fixes the line's remnant kind).  The selection's
    mass, catalog identity, and quantum numbers follow the drawn kind:

    - :data:`KIND_INTACT_PROTON`: the mass is :data:`PROTON_MASS`
      exactly (zero pions; plan line 842: no minimum N-pi excitation is
      imposed), the identity is the catalog proton record
      :data:`PROTON_PDG_ID`;
    - :data:`KIND_LOW_MASS_BARYONIC_RESONANCE`: the record is drawn
      over :func:`low_mass_records` by the same uniform (the residual
      fraction of the kind draw inside the kind's cumulative interval,
      ``index = int(residual * n)`` over the catalog-ordered records,
      clamped into range at the ``u -> 1`` edge) and the mass is the
      record's pole mass;
    - :data:`KIND_HIGH_MASS_BARYONIC_CONTINUUM`: the mass is
      ``mass_sampler(PROTON_MASS, CONTINUUM_M_MAX)`` — the
      caller-supplied law of card 4 over the activated baryonic ``(1,
      1, 0)`` continuum family; the callable consumes zero draws of
      this module's substream and the module performs no further draws;
      ``pdg_id`` and ``record_id`` are ``None``.

    The *kind* argument is the caller's requested kind.  The neutron
    request raises :class:`ChargeExchangeNotImplemented` naming plan
    task 47 before any draw (zero uniforms consumed); a drawn neutron
    kind raises the same error after the kind draw.  A kind outside
    :data:`REMNANT_KINDS` raises ``ValueError`` naming it.  The weights
    are validated before any draw: a key outside
    :data:`REMNANT_KINDS` raises ``ValueError`` naming the foreign
    kind, a sum off 1 by more than :data:`WEIGHTS_SUM_TOL` raises
    ``ValueError``, and the requested kind must name one of the
    weights' kinds.

    Every returned remnant carries
    ``(B, Q, S, 2I3) = (1, 1, 0, 1)`` (:data:`REMNANT_QN`).
    """

    normalized = _validate_weights(weights, None)

    if kind not in REMNANT_KINDS:
        raise ValueError(
            "select_remnant: the requested kind "
            f"{kind!r} is outside the closed remnant-kind set "
            f"{sorted(REMNANT_KINDS)}")
    if kind == KIND_NEUTRON_CHARGE_EXCHANGE:
        # zero draws: the refusal precedes the kind draw (plan line
        # 838; charge exchange is CHARGE_EXCHANGE_TASK_REF, Round 8,
        # outside card 2's closed allowed-exchange set {VACUUM})
        raise ChargeExchangeNotImplemented(
            "select_remnant: the neutron remnant kind "
            f"{KIND_NEUTRON_CHARGE_EXCHANGE!r} is not available at "
            "this round: the outgoing line may carry a neutron only "
            "through explicit charge exchange, and charge exchange is "
            f"{CHARGE_EXCHANGE_TASK_REF} (Round 8), outside the Round "
            "5 closed allowed-exchange set {vacuum} "
            "(na49_mixing.exchange_ledger.ALLOWED_EXCHANGES)")
    normalized = _validate_weights(normalized, kind)

    u = rng_kind.random()

    drawn, lower, upper = _draw_kind(normalized, u)
    if drawn == KIND_NEUTRON_CHARGE_EXCHANGE:
        raise ChargeExchangeNotImplemented(
            "select_remnant: the kind draw selected the neutron "
            f"remnant kind {KIND_NEUTRON_CHARGE_EXCHANGE!r}, which is "
            "not available at this round: the outgoing line may carry "
            "a neutron only through explicit charge exchange, and "
            f"charge exchange is {CHARGE_EXCHANGE_TASK_REF} (Round 8), "
            "outside the Round 5 closed allowed-exchange set {vacuum} "
            "(na49_mixing.exchange_ledger.ALLOWED_EXCHANGES)")

    if drawn == KIND_INTACT_PROTON:
        return RemnantSelection(
            kind=drawn,
            pdg_id=PROTON_PDG_ID,
            record_id=None,
            mass=PROTON_MASS,
            qn=REMNANT_QN,
        )

    if drawn == KIND_LOW_MASS_BARYONIC_RESONANCE:
        records = low_mass_records(catalog)
        if not records:
            raise ValueError(
                "select_remnant: the low-mass remnant kind "
                f"{KIND_LOW_MASS_BARYONIC_RESONANCE!r} has no "
                "committed-catalog (1, 1, 0) record below the frozen "
                f"kernel boundary {KERNEL_BOUNDARY!r} GeV")
        width = upper - lower
        residual = (u - lower) / width if width > 0.0 else 0.0
        if spin_isospin:
            record = _weighted_pick(low_mass_spin_isospin_records(catalog), residual)
        else:
            index = min(int(residual * len(records)), len(records) - 1)
            record = records[index]
        return RemnantSelection(
            kind=drawn,
            pdg_id=int(record.pdg_id),
            record_id=str(record.name),
            mass=float(record.pole_mass),
            qn=(int(record.B), int(record.Q), int(record.S),
                int(record.isospin3)),
        )

    if drawn == KIND_HIGH_MASS_BARYONIC_CONTINUUM:
        mass = float(mass_sampler(PROTON_MASS, CONTINUUM_M_MAX))
        return RemnantSelection(
            kind=drawn,
            pdg_id=None,
            record_id=None,
            mass=mass,
            qn=REMNANT_QN,
        )

    raise AssertionError(
        "select_remnant: the drawn kind "
        f"{drawn!r} has no selection branch (the closed remnant-kind "
        "set and the branch table are out of step; checked, never "
        "repaired)")
