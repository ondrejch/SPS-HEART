"""Sequential color-singlet resonance decay.

Round 4, card 4 (``TASK-20260920-r4-sequential-color-singlet-resonance-
decay``; plan section 22 task 21, plan sections 10.2-10.3, lines
676-714).  The module carries the aligned resonance cascade's core
mechanism: the cascade state record of plan section 10.1, the allowed
quasi-two-body channel enumeration of plan section 10.2, and the
transparent hadronic channel weight of plan section 10.3 with the fixed
hook interface for the alignment weight (cards 6-7) and the decay form
factor (card 8).  Cards 5-10 extend this surface; this card lands alone
with the hooks fixed at unity.

Cascade state (plan section 10.1, lines 664-674)
------------------------------------------------

:class:`CascadeState` is the frozen record of one active state.  Every
plan section 10.1 field is carried: the four-momentum and invariant
mass; the integer ``(B, Q, S)``; the ``quantum`` mapping with ``I``,
``I3``, ``J``, and ``P`` where defined (the committed catalog's fields;
``I3`` keeps the catalog's integer two-units convention; a continuum
state declares none of them beyond ``(B, Q, S)`` and carries an empty
mapping); the spectral-family identifier (a catalog record identity
``"catalog:<pdg_id>"`` or a continuum family id of
``config/v2_singlet/continuum_resonances.json``); the production axis in
the rest frame; the spin-density or alignment label; the parent link;
the exchange ancestry; and the cascade depth counter.  At this card's
landing the axis and the alignment label are caller-supplied on the root
state and are copied to the daughters verbatim: card 5 fills the axis
with the inheritance rules and cards 6-7 own the label semantics, both
without touching this module.

Decay construction (plan section 10.2, lines 676-692)
-----------------------------------------------------

:func:`enumerate_channels` enumerates the allowed quasi-two-body
channels ``H_a -> H_b + h_c`` and ``H_a -> H_b + H_c`` of one parent at
its realized mass.  At least one daughter is lower in mass, ensuring
termination, and every daughter is an established hadron, an
established resonance, or a lower-mass continuum resonance.

Two parent kinds are enumerated:

- **Catalog parent** (``family_id = "catalog:<pdg_id>"``): the open
  quasi-two-body set is the Round 2 kernel's own open-object
  construction at the realized mass, consumed byte-unchanged through
  :meth:`na49_mixing.resonance_decay.ResonanceDecayer._open_objects`
  (the isospin coefficient and the barrier factor reuse the Round 2
  kernel tables, byte-unchanged).  The weights are the card-5
  Clebsch-Gordan charge shares of the multiplet class distributed over
  the class's card-3/4 width law at the realized mass; the enumeration
  keeps the kernel's object order so the channel draw reproduces the
  Round 2 selection.  The Round 2 three-body objects are NOT re-enumerated
  here: the plan section 10.2 line 692 three-body clause ("a direct
  three-body hadronic decay may be included for channels known to
  require it") is satisfied by reusing the Round 2 three-body set
  through the kernel itself, which owns the three-body vertex and its
  draw budget; the catalog handoff of card 9 routes catalog states to
  the kernel for exactly that remainder.  A catalog parent with no open
  quasi-two-body object raises the kernel's
  :class:`na49_mixing.resonance_decay.NoOpenChannelError` (the Round 2
  threshold discipline, unchanged).

- **Continuum parent** (``family_id`` names a continuum family): the
  direct channels are the family's ``allowed_decay_classes`` concrete
  daughter pairs, open above their daughter pole-mass threshold; the
  cascade channels replace, in place, each class daughter whose
  ``(B, Q, S)`` matches a continuum family of the configuration with a
  lower-mass continuum resonance of that family ("continuum daughters
  are the lower-mass continuum resonances of the family's allowed
  hadronic decay classes").  Every declared family of the configuration
  is eligible as a decay daughter, the declared-inactive charged and
  strange mesonic families included: plan section 9.1 brings them into
  the final state through "an allowed parent decay", which this
  enumeration is; the card-5 ``active`` flag governs source sampling
  only.  Duplicate physical channels (one class pair generating the same
  replacement through several classes, or through both daughters) are
  enumerated once; no channel is enumerated twice.

The continuum daughter's realized mass is deterministic and consumes no
random number (the per-vertex draw budget below is frozen at three
uniforms).  The pre-registered card-4 minimal rule realizes the daughter
at the family's on-shell budget edge lowered by the family's
``nominal_width``, floored at the family's continuation floor::

    edge = min(ceiling, M_parent - m_other)
    m_daughter = max(floor, edge - family.nominal_width)

where ``ceiling`` is the family's ``continuation_ceiling`` when one is
set (2026-10-02) and the support edge ``SUPPORT_M_MAX`` otherwise (the
card-4 rule bit for bit), and ``floor`` is the family's ``continuation_floor`` when the config
sets one (Round 5 follow-up Block B stage 2) and the family minimum mass
``m_min`` otherwise, which is the card-4 rule bit for bit.  The floor
acts on continuation daughters only: the source density, its ramp and
``n_norm`` keep ``m_min``.

so the shed hadron is soft (its rest-frame momentum is the two-body
decay momentum at that split), the cascade depth stays far below the
guard across the accepted source-mass range, and the family width is
the parameter that controls the cascade multiplicity of the continuum
states (the round 4 index item 5 pre-registration that makes
``nominal_width`` the R41 continuum multiplicity candidate).  The
channel is open exactly when the budget window is non-empty
(``edge >= floor``).

The rule releases the fixed kinetic energy ``nominal_width`` (0.5 GeV in
the committed families) at every step, so the shed hadron's decay
momentum grows with its mass: about 0.6 GeV/c for a pion and 1.0 GeV/c
for a proton from a 6 GeV state.  A family may instead carry the
optional ``continuation_mass_law = "density"`` (2026-10-03; the R71
residual diagnosis, ``_knowledge/evidence/fixed-kernel-residual-
mechanisms-2026-10-03.md``).  Its continuation daughter's mass is then
drawn on ``[floor, edge]`` from the family's declared tower shape times
the two-body decay momentum,

    p(m) proportional to m**(-b_power) * exp(m / t_h) * q*(M; m, m_other),

the transparent weight of plan section 10.3 applied to a continuum
daughter, which FAB-10 already applies to the broad catalog daughters.
The optional ``continuation_barrier_l = L`` (2026-10-03, after R73)
completes that weight with the orbital barrier: the factor q* becomes
q*^(2L+1) F_L(q*)^2 with the Blatt-Weisskopf F_L of the catalog vertices
(r0 = 1 GeV^-1).  The absent key or L = 0 is the plain q* law.
The draw is the inverse CDF of the trapezoid integral in
``s = sqrt(edge - m)`` on a fixed 201-point grid over the last 16 t_h
below the edge (the tower factor is under 1e-7 further down) and
consumes one uniform on the ``decay_kinematics``
substream, before the two angle uniforms.  The channel weights keep
the card-4 mass (``edge - nominal_width``), so the law changes the
realized masses only.  Without the key the cascade is the card-4 one
bit for bit.

Channel weight (plan section 10.3, lines 694-714)
-------------------------------------------------

The transparent hadronic weight of plan section 10.3,

    w_{a->bc} proportional to C_iso (2J_b+1)(2J_c+1) Phi_2 |F_L(q)|^2
    W_flavor W_align,

is constructed per vertex inside :func:`enumerate_channels` and
normalized over the enumerated channels (the weights sum to 1):

- at **catalog vertices** the weight is the Round 2 kernel's open
  weight: the card-5 Clebsch-Gordan isospin coefficient and the card-4
  Blatt-Weisskopf barrier factor enter through the byte-unchanged width
  law, which already carries the exact two-body phase space;
- at **continuum vertices** the isospin selection is the family's
  declared allowed-decay-class list (``C_iso = 1``; a continuum family
  declares no parent isospin of its own), the barrier factor is
  ``|F_L|^2 = 1`` (a continuum state is a density of states, not a line
  shape: the Round 2 barrier is the established-resonance width law's
  factor), the spin factors ``(2J+1)`` come from the established
  daughters' catalog records (a continuum daughter declares no ``J``
  and contributes the factor 1), and ``Phi_2`` is the exact two-body
  phase space through the Kaellell decay momentum
  ``na49_mixing.resonance_widths.decay_momentum`` at the realized
  daughter masses.

:func:`channel_weight` applies the ``W_flavor`` and ``W_align`` hooks to
a channel's base weight.  Both hooks are fixed at 1.0 at this card's
landing: ``W_flavor = 1.0`` for Round 4 (the flavor weights land in
Round 6, plan section 22 task 37) and ``W_align = 1.0`` until cards 6-7
attach.  The hooks are multiplicative and consume zero random numbers;
the caller re-normalizes the hooked weights over the vertex, which
reproduces the hooked raw-weight shares exactly.  Card 8 wraps this
function with the decay form factor (its own named factor, distinct
from the Round 2 barrier).

Draw budget (frozen at this card's landing)
-------------------------------------------

:func:`draw_decay` is one decay step: **exactly one uniform** on the
``decay_channel`` substream picks the channel over the cumulative
weights (the Round 2 ``_sample_object`` walk), and **exactly two
uniforms** on the ``decay_kinematics`` substream generate the Round 2
two-body kinematics (``cos(theta) = 2u - 1`` isotropic at this card's
landing, ``phi = 2 pi u``; the daughters' realized masses are
deterministic, so no mass draw is consumed).  The budget is frozen at
1 channel + 2 kinematics per vertex through cards 6-8: the alignment
kernels replace the cos(theta) draw inside the existing kinematics
budget and the form factor consumes none; neither adds a uniform and
neither changes the per-vertex draw order.

The leading-baryon split (``KernelConfig.leading_split_fraction``,
2026-10-08) keeps the budget: at a baryonic continuum root the one
channel uniform ``u`` picks the branch and the channel (``u < F``: the
split channels at ``u / F``; otherwise every channel at
``(u - F) / (1 - F)``), the density uniform draws the mesonic
remainder's mass and the cos(theta) uniform the baryon's direction.

Round 2 bit-identity baseline
-----------------------------

At the hook defaults a catalog-parent decay through
:func:`enumerate_channels` + :func:`draw_decay` reproduces the Round 2
``ResonanceDecayer`` behavior at the same seed: the enumerated channel
list is the kernel's open set (same objects, same order, same relative
weights), the channel draw is the kernel's cumulative walk, and the
kinematics is the kernel's two-body convention with the same boost and
Kaellell machinery imported byte-unchanged.  The pinned catalog-only
single decay of the card test is bit-identical in the daughter
four-momenta.

Depth guard (plan section 10.6, line 757)
-----------------------------------------

:data:`MAX_CASCADE_DEPTH = 12` is defined here as the module constant
placeholder of the card; the depth counter is the
:class:`CascadeState` ``depth`` field, incremented by
:func:`draw_decay` on every daughter.  The guard enforcement (raise
``CascadeDepthGuardError`` at depth 12) lands in card 9
(``na49_mixing/cascade_termination.py``), which consumes this constant
and the depth counter read-only.

Conservation is checked, never repaired: every
:func:`draw_decay` vertex conserves ``(B, Q, S)`` exactly and the
daughter four-momenta sum to the parent four-momentum within
:data:`na49_mixing.resonance_decay.DECAY_CLOSURE_TOL`; a violation
raises :class:`na49_mixing.resonance_decay.KinematicsError` and stops
the run (no momentum repair).

Hermetic: standard library, numpy, the committed catalog and family
configuration, and the vendored ``na49_mixing`` modules only; no event
generation, no external tables, no Monte Carlo run (the acceptance is
hermetic: pinned traces and draw counts; the runs R40-R44 exercise the
surface statistically).
"""

from __future__ import annotations

import bisect
import dataclasses
import math
import weakref
from dataclasses import dataclass

from .hadron_catalog import HadronCatalog
from .pdg_database import get_particle
from .quantum_numbers import quantum_numbers
from .resonance_decay import (
    DECAY_CLOSURE_TOL,
    KinematicsError,
    ResonanceDecayer,
    _WEAK_NEUTRAL_KAON_FLAVOR,
    _boost_lab,
    _isospin3_of,
    _isospin_of,
    _multiplet_key,
    clebsch_gordan_squared,
)
from .resonance_widths import BLATT_WEISSKOPF_BARRIER, _draw_entry, decay_momentum
from .spectral_density import SUPPORT_M_MAX
from .two_sided_hadronization import draw_line_mass
from .flavor_weights import apply_flavor_weights

__all__ = [
    "MAX_CASCADE_DEPTH",
    "CATALOG_FAMILY_PREFIX",
    "CONJUGATE_FAMILY_PREFIX",
    "ISOTROPIC_ALIGNMENT_LABEL",
    "KIND_ESTABLISHED_HADRON",
    "KIND_ESTABLISHED_RESONANCE",
    "KIND_CONTINUUM",
    "CascadeState",
    "CascadeChannel",
    "enumerate_channels",
    "channel_weight",
    "draw_decay",
    "draw_decay_with_pool",
    "continuing_pool",
]


#: The hard maximum cascade depth (plan section 10.6, line 757), the
#: module constant placeholder of this card.  The depth counter is the
#: :class:`CascadeState` ``depth`` field; the enforcement (raise
#: ``CascadeDepthGuardError`` at depth 12) lands in card 9
#: (``na49_mixing/cascade_termination.py``).
MAX_CASCADE_DEPTH: int = 12

#: The family-id prefix of a state standing for a committed catalog
#: record: ``"catalog:<pdg_id>"``.  A continuum state carries the
#: family's own ``family_id``.
CATALOG_FAMILY_PREFIX: str = "catalog:"

#: The family-id prefix of a charge-conjugate baryonic family
#: (``KernelConfig.conjugate_towers``, 2026-10-09;
#: ``na49_mixing.conjugate_towers``): ``"anti_" + family_id``.
CONJUGATE_FAMILY_PREFIX: str = "anti_"

#: The spin-density / alignment label at this card's landing: the
#: spin-averaged (isotropic) kernel.  Cards 6-7 own the aligned labels.
ISOTROPIC_ALIGNMENT_LABEL: str = "isotropic"

#: The channel kinds (the daughter classification of plan section
#: 10.2, line 690).  ``established_hadron``: every daughter is a stable
#: established hadron (the vertex is fully terminal).  A channel with
#: an established-resonance daughter continues through the Round 2
#: kernel (the card-9 handoff); a channel with a continuum daughter
#: continues the cascade.
KIND_ESTABLISHED_HADRON: str = "established_hadron"
KIND_ESTABLISHED_RESONANCE: str = "established_resonance"
KIND_CONTINUUM: str = "continuum"
#: A three-body ending of stable hadrons (2026-10-20; na49_mixing.three_body_endings): a direct channel of the
#: floor's pool, drawn with flat three-body phase space.
KIND_THREE_BODY: str = "three_body"


@dataclass(frozen=True)
class CascadeState:
    """One active state of the sequential cascade (plan section 10.1).

    ``momentum`` is the four-momentum ``(E, px, py, pz)`` in GeV and
    ``mass`` its invariant mass in GeV/c^2 (checked consistent at use,
    never repaired).  ``qn`` is the integer ``(B, Q, S)``.  ``quantum``
    carries ``I``, ``I3`` (the catalog's integer two-units), ``J``, and
    ``P`` where defined (the committed catalog's fields; empty for a
    continuum state).  ``family_id`` is the spectral-family identifier:
    ``"catalog:<pdg_id>"`` for a catalog record, the family id of
    ``config/v2_singlet/continuum_resonances.json`` for a continuum
    state.  ``axis`` is the production axis in the rest frame (unit;
    caller-supplied on the root state, copied to the daughters at this
    card's landing, filled by card 5's inheritance rules).  ``parent``
    is the parent state (``None`` on a root) and ``exchange_ancestry``
    the exchange-layer lineage tuple inherited from the parent.
    ``depth`` is the cascade depth counter (0 on a root, +1 per
    :func:`draw_decay` vertex).
    """

    momentum: tuple[float, float, float, float]
    mass: float
    qn: tuple[int, int, int]
    quantum: dict
    family_id: str
    axis: tuple[float, float, float]
    alignment_label: str
    parent: "CascadeState | None"
    exchange_ancestry: tuple
    depth: int


@dataclass(frozen=True)
class CascadeChannel:
    """One allowed quasi-two-body channel of plan section 10.2.

    ``parent_id`` is the parent state's ``family_id``.  ``daughters``
    is the daughter identity pair ``(H_b, h_c)`` (or ``(H_b, H_c)``):
    ``"catalog:<pdg_id>"`` for an established daughter, the continuum
    family id for a continuum daughter, in the channel's daughter order
    (``daughters[0]`` carries the ``+direction`` of the two-body split,
    the Round 2 convention).  ``kind`` classifies the daughters:
    ``'established_hadron'`` (every daughter a stable established
    hadron), ``'established_resonance'`` (an unstable catalog record
    daughter; the card-9 handoff territory), or ``'continuum'`` (a
    lower-mass continuum resonance daughter; the cascade continues).
    ``weight`` is the normalized transparent weight of plan section
    10.3 at the hook defaults: the per-vertex shares sum to 1.
    """

    parent_id: str
    daughters: tuple[str, str]
    kind: str
    weight: float


# ----------------------------------------------------------------------
# Resolution helpers (pure; the same mass sources as the Round 2 kernel)
# ----------------------------------------------------------------------
def _daughter_pole_mass(catalog: HadronCatalog, pdg_id: int) -> float:
    """Pole mass of an established daughter: the catalog record's
    ``pole_mass`` when the daughter is a record, else the vendored PDG
    table mass (the Round 2 kernel's daughter-mass sources,
    byte-unchanged)."""

    try:
        return float(catalog.get(int(pdg_id)).pole_mass)
    except KeyError:
        entry = get_particle(int(pdg_id))
        if entry is None:
            raise KeyError(
                f"daughter pdg_id {int(pdg_id)} is neither a catalog "
                "record nor in the vendored PDG table"
            ) from None
        return float(entry.mass)


def _qn_of(catalog: HadronCatalog, pdg_id: int) -> tuple[int, int, int]:
    """The ``(B, Q, S)`` of an established daughter (the Round 2
    kernel's ``_bqs_of`` sources)."""

    try:
        record = catalog.get(int(pdg_id))
    except KeyError:
        try:
            qn = quantum_numbers(int(pdg_id))
        except (KeyError, ValueError) as exc:
            raise KinematicsError(
                f"daughter pdg {int(pdg_id)}: quantum numbers "
                f"unresolvable ({exc})"
            ) from exc
        return (int(qn.B), int(qn.Q), int(qn.S))
    return (int(record.B), int(record.Q), int(record.S))


def _quantum_of(catalog: HadronCatalog, pdg_id: int) -> dict:
    """The ``quantum`` mapping of an established daughter (the catalog
    record's ``I``, ``I3`` two-units, ``J``, ``P``); empty for a
    daughter outside the catalog."""

    try:
        record = catalog.get(int(pdg_id))
    except KeyError:
        return {}
    return {
        "I": float(record.isospin),
        "I3": int(record.isospin3),
        "J": float(record.spin),
        "P": record.parity,
    }


def _spin_factor(catalog: HadronCatalog, pdg_id: int) -> float:
    """The plan section 10.3 spin factor ``(2J + 1)`` of an established
    daughter (the catalog record's spin; 1.0 for a daughter outside the
    catalog, which declares no spin)."""

    try:
        return 2.0 * float(catalog.get(int(pdg_id)).spin) + 1.0
    except KeyError:
        return 1.0


def _established_kind(
    d0: int, d1: int, catalog: HadronCatalog
) -> str:
    """The channel kind of an all-established daughter pair: every
    daughter stable (in the catalog's termination set) is
    ``established_hadron``; any unstable record daughter makes the
    channel ``established_resonance`` (the card-9 handoff territory)."""

    stable = catalog.stable_states()
    if int(d0) in stable and int(d1) in stable:
        return KIND_ESTABLISHED_HADRON
    return KIND_ESTABLISHED_RESONANCE


def _family_by_id(fams, family_id: str):
    """The continuum family record with *family_id*; ``ValueError``
    naming the id when absent."""

    for family in fams:
        if str(family.family_id) == family_id:
            return family
    raise ValueError(
        f"no continuum family record carries family_id {family_id!r} "
        "(the continuum family configuration is the resolution source)"
    )


#: The lowest direct thresholds per family object and catalog (2026-10-20,
#: the probe profile: recomputed over every class on each of 420,000 calls,
#: a third of the warm generation time); the entry holds the family, so its
#: id is not reused while the entry lives, and catalog tokens are never reused.
_THRESHOLD_MEMO: dict[tuple, tuple] = {}


def _lowest_direct_threshold(family, catalog: HadronCatalog) -> float:
    """The lowest daughter pole-mass sum over the family's allowed decay
    classes: below it a state of the family has no open channel."""

    from .resonance_widths import _catalog_token
    key = (id(family), _catalog_token(catalog))
    entry = _THRESHOLD_MEMO.get(key)
    if entry is not None and entry[0] is family:
        return entry[1]
    value = min(_daughter_pole_mass(catalog, d0) + _daughter_pole_mass(catalog, d1)
                for d0, d1 in (_parse_class(c) for c in family.allowed_decay_classes))
    _THRESHOLD_MEMO[key] = (family, value)
    return value


def _continuation_floor(family, catalog: HadronCatalog | None = None) -> float:
    """The lowest realized mass of a continuation daughter of
    *family*: its ``continuation_floor`` when the config sets one, else
    ``m_min`` (the card-4 rule, bit for bit), and never below the
    family's lowest direct-channel threshold, so a continuation daughter
    always has an open channel (2026-10-01: a 1.1 GeV floor realized
    strange baryonic states below the Lambda mass; m_min lies above every
    threshold, so the default is unchanged)."""

    floor = getattr(family, "continuation_floor", None)
    base = float(family.m_min) if floor is None else float(floor)
    if floor is None or catalog is None:
        return base
    return max(base, _lowest_direct_threshold(family, catalog))


#: The grid of the density-weighted continuation mass draw (2026-10-03):
#: 201 nodes on [max(floor, edge - 16 t_h), edge]; below that window the
#: tower factor exp((m - edge) / t_h) is under e**-16 (about 1e-7).
_DENSITY_GRID_POINTS = 201
_DENSITY_WINDOW_T_H = 16.0

#: The hard depth guard of a cascade whose continuum families carry a
#: lifted continuation ceiling (2026-10-02): a state up to sqrt(s) then
#: steps down one hadron and one nominal width at a time, about 23 steps
#: from 16 GeV to a 1.7 GeV floor, beyond MAX_CASCADE_DEPTH = 12 (set when
#: every continuation daughter stayed below the 5 GeV support edge).
_LIFTED_CASCADE_DEPTH = 48


def _cascade_depth_limit(fams) -> int:
    """The hard depth guard for a cascade over *fams*:
    ``MAX_CASCADE_DEPTH`` (the card-9 guard, bit for bit), or
    ``_LIFTED_CASCADE_DEPTH`` when any family carries a continuation
    ceiling or draws its continuation masses from the density law (a
    smaller release per step; from 16 GeV the density law reaches depth
    13 on average and 16 at most in 60 cascades, 2026-10-03)."""

    if any(getattr(f, "continuation_ceiling", None) is not None
           or getattr(f, "continuation_mass_law", None) == "density" for f in fams):
        return _LIFTED_CASCADE_DEPTH
    return MAX_CASCADE_DEPTH


def _continuation_ceiling(family) -> float:
    """The highest realized mass of a continuation daughter of *family*:
    its ``continuation_ceiling`` when one is set (2026-10-02), else the
    support edge ``SUPPORT_M_MAX`` (the card-4 rule, bit for bit)."""

    ceiling = getattr(family, "continuation_ceiling", None)
    return SUPPORT_M_MAX if ceiling is None else float(ceiling)


def _i3x2_of_qn(qn) -> int:
    """Twice the I3 of a (B, Q, S) state (Gell-Mann-Nishijima)."""
    b, q, s = (int(v) for v in qn)
    return 2 * q - b - s


def _flavor_i3x2(pdg_id: int, catalog: HadronCatalog) -> int:
    """Twice the I3 of an established hadron, with the weak-eigenstate
    neutral kaons at their flavor states (K0S -> K0, K0L -> K0bar; the
    catalog records them at 2 I3 = 0, a mixture without definite I3), as
    the Round 2 kernel's multiplet keys do."""
    pid = int(pdg_id)
    return _isospin3_of(_WEAK_NEUTRAL_KAON_FLAVOR.get(pid, pid), catalog)


def _isospin_factor_direct(parent_i: float, parent_i3x2: int, d0: int, d1: int,
                           catalog: HadronCatalog) -> float:
    """The isospin share |<I_0 I3_0; I_1 I3_1 | I I3>|^2 of a concrete
    established pair, both orderings summed for two different members of
    one multiplet (the Round 2 kernel's convention: pi+ pi- : pi0 pi0 =
    2 : 1 from I = 0)."""
    i0, m0 = _isospin_of(d0, catalog), _flavor_i3x2(d0, catalog)
    i1, m1 = _isospin_of(d1, catalog), _flavor_i3x2(d1, catalog)
    w = clebsch_gordan_squared(parent_i, parent_i3x2, i0, m0, i1, m1)
    if d0 != d1 and _multiplet_key(d0, catalog) == _multiplet_key(d1, catalog):
        w += clebsch_gordan_squared(parent_i, parent_i3x2, i1, m1, i0, m0)
    return w


def _isospin_factor_continuum(parent_i: float, parent_i3x2: int, daughter_family,
                              established: int, catalog: HadronCatalog) -> float:
    """The isospin share of a continuation channel: the continuum
    daughter at its family's isospin, the established sibling at its
    record isospin."""
    if daughter_family.isospin is None:
        raise ValueError(
            f"continuum daughter family {daughter_family.family_id!r} carries "
            "no isospin while its parent does (one isospin rule per config; "
            "checked, never repaired)")
    return clebsch_gordan_squared(
        parent_i, parent_i3x2, float(daughter_family.isospin), _i3x2_of_qn(daughter_family.qn),
        _isospin_of(established, catalog), _flavor_i3x2(established, catalog))


def _daughter_families(parent_family, fams, qn: tuple[int, int, int]) -> list:
    """The continuum families a continuation step may enter for a daughter
    of the ``(B, Q, S)`` *qn*: the first family in config order under the
    committed rule; every family of that *qn* under the parent family's
    ``continuation_daughters = "isospin"`` (2026-10-06), each then weighted by
    its own isospin share in :func:`enumerate_channels`."""
    if getattr(parent_family, "continuation_daughters", None) == "isospin":
        wanted = tuple(int(v) for v in qn)
        return [f for f in fams if tuple(f.qn) == wanted]
    first = _family_by_qn(fams, qn)
    return [] if first is None else [first]


def _family_by_qn(fams, qn: tuple[int, int, int]):
    """The first continuum family record with the ``(B, Q, S)`` *qn*
    (config order; every declared family is eligible as a decay
    daughter, plan section 9.1's allowed-parent-decay route), or
    ``None``."""

    wanted = tuple(int(v) for v in qn)
    for family in fams:
        if tuple(family.qn) == wanted:
            return family
    return None


_PARSED_CLASSES: dict[str, tuple[int, int]] = {}


def _parse_class(class_id: str) -> tuple[int, int]:
    """The concrete daughter pair of one allowed-decay-class identifier
    ``"(d0,d1)"`` of the family configuration (memoized per string,
    2026-10-20: 16.5 million parses in the probe profile)."""

    hit = _PARSED_CLASSES.get(class_id) if isinstance(class_id, str) else None
    if hit is not None:
        return hit
    pair = _parse_class_text(class_id)
    if isinstance(class_id, str):
        _PARSED_CLASSES[class_id] = pair
    return pair


def _parse_class_text(class_id) -> tuple[int, int]:
    text = str(class_id).strip()
    if not (text.startswith("(") and text.endswith(")")):
        raise ValueError(
            f"malformed allowed-decay-class identifier {class_id!r}: "
            "expected '(d0,d1)'"
        )
    parts = text[1:-1].split(",")
    if len(parts) != 2:
        raise ValueError(
            f"malformed allowed-decay-class identifier {class_id!r}: "
            "expected two daughter ids"
        )
    return (int(parts[0]), int(parts[1]))


#: Pure compute cache of the Round 2 kernel shims per catalog object.
#: A shim is a ``ResonanceDecayer`` whose construction consumes no
#: random numbers (the substream derivation is pure) and whose open-set
#: construction ``_open_objects`` is pure; the shim's mass and class
#: caches are pure compute caches.  The cache is keyed weakly on the
#: catalog, so the shim lives as long as the catalog it serves.
_KERNEL_SHIMS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _kernel_shim(catalog: HadronCatalog) -> ResonanceDecayer:
    """The Round 2 kernel shim for the catalog-parent open-set
    construction (byte-unchanged kernel tables, consumed read-only)."""

    shim = _KERNEL_SHIMS.get(catalog)
    if shim is None:
        shim = ResonanceDecayer(catalog, 0, 0, 0)
        _KERNEL_SHIMS[catalog] = shim
    return shim


# ----------------------------------------------------------------------
# Channel enumeration (plan section 10.2)
# ----------------------------------------------------------------------
def enumerate_channels(
    state: CascadeState, catalog: HadronCatalog, fams, flavor=None
) -> list[CascadeChannel]:
    """The allowed quasi-two-body channels of plan section 10.2 for
    *state* at its realized mass.

    *flavor* (Round 6, task 37) is an optional
    :class:`~na49_mixing.flavor_weights.FlavorParams`: for a continuum
    parent the enumerated weights are multiplied by the plan section
    12.2 flavor factor and re-normalized over the vertex
    (:func:`~na49_mixing.flavor_weights.apply_flavor_weights`); a
    catalog parent keeps its measured branching fractions.  ``None`` or
    the identity point leaves the channels unchanged, object for object.

    Returns the open channels in enumeration order with the normalized
    transparent weights of plan section 10.3 at the hook defaults (the
    per-vertex shares sum to 1); no channel is enumerated twice.  An
    empty return means no quasi-two-body channel is open at the
    realized mass (the caller decides termination; a catalog parent
    with no open object raises the kernel's ``NoOpenChannelError`` per
    the Round 2 threshold discipline).

    The topology cascades (``topology_single_diffraction._decay_excitation``
    and its double- and non-diffractive counterparts) keep only the
    continuation channels whenever one is open: a continuum state continues
    while any continuation channel is open and splits into a direct
    two-body pair only at the floor.  The direct-channel shares never
    decide between continuing and terminating, and only the relative
    weights within the continuation set enter the draw (code-physics
    review M-2, 2026-10-06).  Under ``continuation_q_min`` a continuation
    channel opens only when its sibling reaches that decay momentum at the
    daughter's floor mass (2026-10-10).  The sibling weight of a continuation channel
    is the card-4 q* at the fixed mass ``max(floor, edge -
    nominal_width)`` unless the daughter family sets
    ``continuation_sibling_weight = "density"``
    (:func:`_density_integrated_phase_space`).

    See the module docstring for the two parent kinds, the
    continuum-daughter rule, and the deterministic daughter mass rule.
    """

    family_id = str(state.family_id)
    if family_id.startswith(CATALOG_FAMILY_PREFIX):
        return _enumerate_catalog_channels(state, catalog, family_id)
    channels = _enumerate_continuum_channels(state, catalog, fams, family_id)
    if flavor is None:
        return channels
    return apply_flavor_weights(channels, state.qn, catalog, fams, flavor)


def _enumerate_catalog_channels(
    state: CascadeState, catalog: HadronCatalog, family_id: str
) -> list[CascadeChannel]:
    """The catalog-parent open set: the Round 2 kernel's open objects
    at the realized mass, filtered to the quasi-two-body set (the
    Round 2 three-body set stays the kernel's handoff territory), in
    the kernel's object order, with the kernel's weights normalized
    over the filtered set."""

    pdg = int(family_id[len(CATALOG_FAMILY_PREFIX):])
    record = catalog.get(pdg)  # KeyError propagates: unknown record
    state_qn = (int(state.qn[0]), int(state.qn[1]), int(state.qn[2]))
    if (int(record.B), int(record.Q), int(record.S)) != state_qn:
        raise ValueError(
            f"cascade state quantum numbers {state_qn} disagree "
            f"with the catalog record {record.name} (pdg {pdg}) carrying "
            f"{(int(record.B), int(record.Q), int(record.S))}; "
            "conservation is checked, never repaired"
        )
    objects, _open_total = _kernel_shim(catalog)._open_objects(
        record, float(state.mass)
    )
    # The cascade places established daughters at their pole masses
    # (the card-4 minimal rule), so a channel that the convolved width
    # of a broad daughter opens below its pole-mass threshold (FAB-10,
    # 2026-10-02) is realized only through the kernel's joint
    # daughter-mass draw and is left out here.
    two_body = [
        (channel, combination, weight)
        for channel, combination, weight in objects
        if len(combination) == 2 and not channel.is_radiative
        and float(state.mass) > _daughter_pole_mass(catalog, int(combination[0]))
        + _daughter_pole_mass(catalog, int(combination[1]))
    ]
    total = math.fsum(weight for _, _, weight in two_body)
    if not (math.isfinite(total) and total > 0.0):
        # every open object was three-body: the quasi-two-body set is
        # empty and the three-body modes stay the kernel's territory
        return []
    channels: list[CascadeChannel] = []
    for channel, combination, weight in two_body:
        d0, d1 = int(combination[0]), int(combination[1])
        channels.append(CascadeChannel(
            parent_id=family_id,
            daughters=(f"{CATALOG_FAMILY_PREFIX}{d0}",
                       f"{CATALOG_FAMILY_PREFIX}{d1}"),
            kind=_established_kind(d0, d1, catalog),
            weight=float(weight) / total,
        ))
    return channels


def _enumerate_continuum_channels(
    state: CascadeState, catalog: HadronCatalog, fams, family_id: str
) -> list[CascadeChannel]:
    """The continuum-parent channels: the family's allowed-decay-class
    direct pairs and the lower-mass continuum replacements, deduplicated,
    with the plan section 10.3 transparent weights normalized over the
    vertex."""

    family = _family_by_id(fams, family_id)
    state_qn = (int(state.qn[0]), int(state.qn[1]), int(state.qn[2]))
    if tuple(family.qn) != state_qn:
        raise ValueError(
            f"cascade state quantum numbers {state_qn} disagree "
            f"with the continuum family {family.family_id} carrying "
            f"{tuple(family.qn)}; conservation is checked, never repaired"
        )
    parent_mass = float(state.mass)
    candidates: list[tuple[tuple[str, str], str, float]] = []
    seen: set[tuple[str, str]] = set()
    # the isospin shares (the config's isospin rule; None keeps the unit
    # couplings of the Round 4 cascade bit for bit).  A vertex of an isospin
    # mixture (2026-10-10; ContinuumFamily.isospin_mixture) counts every
    # charge combination equally
    mixture = bool(getattr(family, "isospin_mixture", False))
    parent_i = None if family.isospin is None or mixture else float(family.isospin)
    parent_i3x2 = _i3x2_of_qn(state_qn)
    # the running-width pair floor (2026-10-21; KernelConfig.
    # resonance_pair_line_shapes, piece P1a of the card
    # TASK-20261021-r106-pbar-and-forward-piplus-mechanisms): the family's
    # direct baryon-antibaryon pairs open from the daughters' threshold-mass
    # sum with the unstable daughter on its line shape
    pair_line_shapes = bool(getattr(family, "resonance_pair_line_shapes", False))

    # the density-integrated sibling phase space per (daughter family,
    # sibling mass) of this vertex (2026-10-06; pure compute)
    integrated: dict[tuple[str, float], float] = {}

    def _offer(
        daughters: tuple[str, str], kind: str, raw: float
    ) -> None:
        key = (min(daughters[0], daughters[1]), max(daughters[0], daughters[1]))
        if key in seen:
            return  # no channel is enumerated twice
        if not (math.isfinite(raw) and raw > 0.0):
            return  # a degenerate two-body split carries no phase space
        seen.add(key)
        candidates.append((daughters, kind, float(raw)))

    def _offer_direct(d0: int, d1: int, m0: float, m1: float, open_masses=None) -> None:
        if open_masses is None:
            if parent_mass < m0 + m1:
                return
            q0, q1 = m0, m1
        else:
            # the running-width pair floor (resonance_pair_line_shapes):
            # open from the threshold-mass sum, q* at the effective masses
            q0, q1 = open_masses
            if not parent_mass > q0 + q1:
                return
        q_star = decay_momentum(parent_mass, q0, q1)
        iso = 1.0 if parent_i is None else _isospin_factor_direct(parent_i, parent_i3x2, d0, d1, catalog)
        _offer(
            (f"{CATALOG_FAMILY_PREFIX}{d0}",
             f"{CATALOG_FAMILY_PREFIX}{d1}"),
            _established_kind(d0, d1, catalog),
            _spin_factor(catalog, d0) * _spin_factor(catalog, d1) * q_star * iso,
        )

    def _offer_three(class_id: str) -> None:
        from .three_body_endings import parse_triple, three_body_isospin_share, three_body_weight
        triple = parse_triple(class_id)
        key = tuple(sorted(f"{CATALOG_FAMILY_PREFIX}{d}" for d in triple))
        if key in seen:
            return
        masses = [_daughter_pole_mass(catalog, d) for d in triple]
        if not parent_mass > math.fsum(masses):
            return
        iso = 1.0 if parent_i is None else three_body_isospin_share(parent_i, parent_i3x2, triple, catalog)
        raw = three_body_weight(float(family.three_body_strength), parent_mass, triple, masses,
                                [_spin_factor(catalog, d) for d in triple], iso)
        if not (math.isfinite(raw) and raw > 0.0):
            return
        seen.add(key)
        candidates.append((tuple(f"{CATALOG_FAMILY_PREFIX}{d}" for d in triple), KIND_THREE_BODY, float(raw)))

    def _offer_class(class_id: str) -> None:
        d0, d1 = _parse_class(class_id)
        m0 = _daughter_pole_mass(catalog, d0)
        m1 = _daughter_pole_mass(catalog, d1)
        opened = None
        if pair_line_shapes and _is_baryon_antibaryon_pair(catalog, d0, d1):
            # the running-width pair floor (resonance_pair_line_shapes):
            # the direct pair opens from the threshold-mass sum, q* at the
            # effective masses below the pole sum (None keeps the committed
            # pole-mass rule)
            opened = _pair_open_masses(catalog, d0, d1, m0, m1, parent_mass)
        _offer_direct(d0, d1, m0, m1, opened)
        for d_lead, d_other, m_lead, m_other, daughter_family in (
            (dl, do, ml, mo, dfam)
            for dl, do, ml, mo in ((d0, d1, m0, m1), (d1, d0, m1, m0))
            for dfam in _daughter_families(family, fams, _qn_of(catalog, dl))
        ):
            edge = min(_continuation_ceiling(daughter_family), parent_mass - m_other)
            floor = _continuation_floor(daughter_family, catalog)
            if not edge >= floor:
                continue
            # the minimum step momentum (2026-10-10; KernelConfig.
            # continuation_q_min): the channel opens only when the sibling
            # reaches q_min at the daughter's floor mass
            q_min = getattr(daughter_family, "continuation_q_min", None)
            if q_min is not None and decay_momentum(parent_mass, floor, m_other) < float(q_min):
                continue
            m_lead_realized = max(
                floor,
                edge - float(daughter_family.nominal_width),
            )
            established = d_other if d_lead == d0 else d0
            slot = 0 if d_lead == d0 else 1
            daughter_id = str(daughter_family.family_id)
            established_id = f"{CATALOG_FAMILY_PREFIX}{established}"
            daughters = (
                daughter_id if slot == 0 else established_id,
                established_id if slot == 0 else daughter_id,
            )
            if getattr(daughter_family, "continuation_sibling_weight", None) == "density":
                key = (daughter_id, float(m_other))
                if key not in integrated:
                    integrated[key] = _density_integrated_phase_space(
                        daughter_family, parent_mass, m_other, floor, edge)
                q_star = integrated[key]
            else:
                q_star = decay_momentum(parent_mass, m_lead_realized, m_other)
            iso = 1.0 if parent_i is None else _isospin_factor_continuum(
                parent_i, parent_i3x2, daughter_family, established, catalog)
            _offer(
                daughters,
                KIND_CONTINUUM,
                _spin_factor(catalog, d_other) * q_star * iso,
            )

    for class_id in family.allowed_decay_classes:
        _offer_class(class_id)

    def _finish(cands, seen_set):
        # the floor pool (2026-10-09; na49_mixing.floor_pools): at a vertex
        # with no open continuation channel the family's floor classes join
        # the direct pairs with the same weight law; a vertex that continues
        # keeps the allowed decay classes alone.  The charge-conjugate
        # baryonic towers (2026-10-09; na49_mixing.conjugate_towers): at a
        # mesonic vertex the two conjugate baryon-pair continuations take
        # half the weight each, so the vertex's baryon-pair share stays the
        # committed one
        nonlocal candidates, seen
        saved = (candidates, seen)
        candidates, seen = cands, seen_set
        try:
            floor_classes = getattr(family, "floor_decay_classes", ())
            if floor_classes and not any(kind == KIND_CONTINUUM for _, kind, _ in candidates):
                for class_id in floor_classes:
                    d0, d1 = _parse_class(class_id)
                    _offer_direct(d0, d1, _daughter_pole_mass(catalog, d0), _daughter_pole_mass(catalog, d1))
            # the three-body endings (2026-10-20; na49_mixing.three_body_endings): at a vertex with no open
            # continuation channel the family's three-body classes join the direct pool
            three = getattr(family, "three_body_classes", ())
            if three and not any(kind == KIND_CONTINUUM for _, kind, _ in candidates):
                for class_id in three:
                    _offer_three(class_id)
            out = candidates
        finally:
            candidates, seen = saved
        if state_qn[0] == 0 and any(str(f.family_id).startswith(CONJUGATE_FAMILY_PREFIX) for f in fams):
            out = _halve_conjugate_pairs(out, catalog, fams,
                                         every=completion_here and getattr(family, "baryon_pair_completion", None) is not None)
        return out

    # the line-emitted pair share (2026-11-04; KernelConfig.baryon_pair_share = "line"): a family that offers its
    # completion only on a cluster a baryonic line emits directly keeps the committed vertex elsewhere
    completion_here = not getattr(family, "completion_line_only", False) or _emitted_by_baryonic_line(state)

    # the baryon-resonance pairs (2026-10-10; na49_mixing.resonance_pairs):
    # the family's resonance-pair classes join its allowed decay classes,
    # and the baryon-pair share of the continuation channels and of the
    # direct pairs each stays the share without them (the pair strength
    # r_BBbar keeps its meaning)
    pair_classes = tuple(getattr(family, "resonance_pair_classes", ()) or ())
    base = None
    if pair_classes:
        base = _finish(list(candidates), set(seen))
        for class_id in pair_classes:
            _offer_class(class_id)
    # the baryon-pair completion (2026-11-04; na49_mixing.baryon_pair_completion): the completion's octet classes
    # join with the weight law after the resonance pairs, its decuplet classes after them, both outside the
    # resonance pairs' share hold; the decuplet classes are then held at the octet classes' share
    free_octet, free_decuplet = set(), set()
    for classes, free in ((getattr(family, "completion_pair_classes", ()), free_octet),
                          (getattr(family, "completion_resonance_classes", ()), free_decuplet)):
        for class_id in (tuple(classes or ()) if completion_here else ()):
            n_before = len(candidates)
            _offer_class(class_id)
            free.update(daughters for daughters, _kind, _raw in candidates[n_before:])
    candidates = _finish(candidates, seen)
    if base is not None:
        candidates = _hold_pair_shares(candidates, base, catalog, fams, free=free_octet | free_decuplet)
    if free_decuplet:
        candidates = _hold_completion_share(candidates, free_octet, free_decuplet)
    # the universal baryon-pair share (2026-11-04; KernelConfig.baryon_pair_share): at a completed mesonic family
    # other than the reference, the completion's pairs that create no s-sbar take the reference family's committed
    # pair share at this mass, and the others keep their weight-law ratio to them
    reference = getattr(family, "pair_share_reference", None)
    if reference is not None and (free_octet or free_decuplet):
        targets = _reference_pair_shares(reference, state, catalog, fams)
        candidates = _hold_universal_share(candidates, free_octet | free_decuplet, targets, state_qn, catalog, fams)

    total = math.fsum(raw for _, _, raw in candidates)
    if not (math.isfinite(total) and total > 0.0):
        return []
    return [
        CascadeChannel(
            parent_id=family_id,
            daughters=daughters,
            kind=kind,
            weight=raw / total,
        )
        for daughters, kind, raw in candidates
    ]


def _hold_pair_shares(candidates, base, catalog: HadronCatalog, fams, free=frozenset()):
    """The raw candidates with the baryon-pair share (channels whose
    daughters carry |B| = 2 in all) of the continuation channels and of the
    direct pairs each set to its share among the *base* candidates (the
    vertex without the resonance-pair classes).  The topology cascades draw
    among the continuation channels while one is open and among the direct
    pairs only at the floor, so each set keeps its own share
    (``na49_mixing.resonance_pairs``).  The channels of *free* (the
    daughters of the baryon-pair completion's channels,
    ``na49_mixing.baryon_pair_completion``) keep their weights and stay out
    of the shares."""

    qn_b = {str(f.family_id): abs(int(f.qn[0])) for f in fams}

    def is_pair(daughters) -> bool:
        total = 0
        for d in daughters:
            if d.startswith(CATALOG_FAMILY_PREFIX):
                total += abs(int(catalog.get(int(d[len(CATALOG_FAMILY_PREFIX):])).B))
            else:
                total += qn_b[d]
        return total >= 2

    def share(cands, continuing: bool) -> float:
        chosen = [(daughters, raw) for daughters, kind, raw in cands
                  if (kind == KIND_CONTINUUM) == continuing and daughters not in free]
        total = math.fsum(raw for _, raw in chosen)
        if not total > 0.0:
            return 0.0
        return math.fsum(raw for daughters, raw in chosen if is_pair(daughters)) / total

    factor = {}
    for continuing in (True, False):
        s0, s1 = share(base, continuing), share(candidates, continuing)
        factor[continuing] = (s0 / s1, (1.0 - s0) / (1.0 - s1)) if 0.0 < s1 < 1.0 else (1.0, 1.0)
    return [(daughters, kind, raw if daughters in free else
             raw * factor[kind == KIND_CONTINUUM][0 if is_pair(daughters) else 1])
            for daughters, kind, raw in candidates]


def _hold_completion_share(candidates, octet, decuplet):
    """The raw candidates with the share of the baryon-pair completion's
    channels (the daughters in *octet* or *decuplet*) among the continuation
    channels and among the direct pairs each set to the share of the *octet*
    channels without the *decuplet* ones (``na49_mixing.
    baryon_pair_completion``): the decuplet pairs redistribute the octet
    pairs' share, as the committed resonance pairs do the p pbar and n nbar
    share (:func:`_hold_pair_shares`)."""

    sector = octet | decuplet
    factor = {}
    for continuing in (True, False):
        chosen = [(daughters, raw) for daughters, kind, raw in candidates if (kind == KIND_CONTINUUM) == continuing]
        rest = math.fsum(raw for daughters, raw in chosen if daughters not in sector)
        oct_sum = math.fsum(raw for daughters, raw in chosen if daughters in octet)
        sec_sum = math.fsum(raw for daughters, raw in chosen if daughters in sector)
        s0 = oct_sum / (rest + oct_sum) if rest + oct_sum > 0.0 else 0.0
        s1 = sec_sum / (rest + sec_sum) if rest + sec_sum > 0.0 else 0.0
        factor[continuing] = (s0 / s1, (1.0 - s0) / (1.0 - s1)) if 0.0 < s1 < 1.0 and s0 > 0.0 else (1.0, 1.0)
    return [(daughters, kind, raw * factor[kind == KIND_CONTINUUM][0 if daughters in sector else 1])
            for daughters, kind, raw in candidates]


_REFERENCE_SHARES: dict = {}


def _pair_total_b(daughters, catalog: HadronCatalog, fams) -> int:
    """The summed |B| of a channel's daughters (2 for a baryon-antibaryon pair)."""
    total = 0
    for d in daughters:
        if d.startswith(CATALOG_FAMILY_PREFIX):
            total += abs(int(catalog.get(int(d[len(CATALOG_FAMILY_PREFIX):])).B))
        else:
            total += abs(int(_family_by_id(fams, d).qn[0]))
    return total


def _created_strangeness(daughters, parent_qn, catalog: HadronCatalog, fams) -> int:
    """The s-sbar pairs a channel creates, (sum |S_daughters| - |S_parent|) / 2."""
    total = 0
    for d in daughters:
        if d.startswith(CATALOG_FAMILY_PREFIX):
            total += abs(int(catalog.get(int(d[len(CATALOG_FAMILY_PREFIX):])).S))
        else:
            total += abs(int(_family_by_id(fams, d).qn[2]))
    return (total - abs(int(parent_qn[2]))) // 2


def _reference_pair_shares(reference: str, state, catalog: HadronCatalog, fams) -> dict:
    """The committed baryon-pair share of the *reference* family (its
    completion classes left out) at the state's mass, among the continuation
    channels and among the direct pairs (``{True: ..., False: ...}``; the
    universal pair share, ``na49_mixing.baryon_pair_completion``)."""
    # keyed on the family list's id with the list kept and checked by identity (the code review glm53f F9: a
    # recycled address never serves a dead kernel's shares)
    key = (id(fams), reference, float(state.mass))
    hit = _REFERENCE_SHARES.get(key)
    if hit is not None and hit[0] is fams:
        return hit[1]
    ref = _family_by_id(fams, reference)
    # the committed reference: its completion and its own reference cleared, so a cyclic configuration cannot recurse
    committed = dataclasses.replace(ref, completion_pair_classes=(), completion_resonance_classes=(),
                                    baryon_pair_completion=None, pair_share_reference=None)
    fams_ref = [committed if f is ref else f for f in fams]
    ref_state = dataclasses.replace(state, qn=tuple(int(v) for v in ref.qn), family_id=reference)
    channels = _enumerate_continuum_channels(ref_state, catalog, fams_ref, reference)
    out = {}
    for continuing in (True, False):
        chosen = [ch for ch in channels if (ch.kind == KIND_CONTINUUM) == continuing]
        total = math.fsum(ch.weight for ch in chosen)
        pairs = math.fsum(ch.weight for ch in chosen if _pair_total_b(ch.daughters, catalog, fams_ref) >= 2)
        out[continuing] = pairs / total if total > 0.0 else 0.0
    if len(_REFERENCE_SHARES) > 4096:
        _REFERENCE_SHARES.clear()
    _REFERENCE_SHARES[key] = (fams, out)
    return out


def _emitted_by_baryonic_line(state) -> bool:
    """Whether *state* is a cluster a baryonic line emits directly: its
    parent is a baryonic state (|B| = 1).  A root, or a state continued from
    a mesonic one, is not (``KernelConfig.baryon_pair_share`` = "line")."""

    parent = getattr(state, "parent", None)
    return parent is not None and abs(int(parent.qn[0])) == 1


def _hold_universal_share(candidates, sector, targets, parent_qn, catalog: HadronCatalog, fams):
    """The raw candidates with the completion's channels (the daughters in
    *sector*) scaled by one factor per channel kind, so that those creating
    no s-sbar take the target share of the kind (*targets*, the reference
    family's committed pair share), and the other channels scaled to keep
    the kind's total.  A kind whose sector holds no such channel, or whose
    target cannot be met, is left as it is."""

    out = list(candidates)
    for continuing in (True, False):
        idx = [i for i, (_d, kind, _r) in enumerate(out) if (kind == KIND_CONTINUUM) == continuing]
        total = math.fsum(out[i][2] for i in idx)
        anchor = math.fsum(out[i][2] for i in idx if out[i][0] in sector
                           and _created_strangeness(out[i][0], parent_qn, catalog, fams) == 0)
        sec = math.fsum(out[i][2] for i in idx if out[i][0] in sector)
        rest = total - sec
        target = float(targets[continuing])
        if not (total > 0.0 and anchor > 0.0 and rest > 0.0):
            continue
        f = target * total / anchor
        g = (total - f * sec) / rest
        if not g > 0.0:
            continue
        for i in idx:
            d, kind, raw = out[i]
            out[i] = (d, kind, raw * (f if d in sector else g))
    return [c for c in out if c[2] > 0.0]


def _is_baryon_antibaryon_pair(catalog: HadronCatalog, d0: int, d1: int) -> bool:
    """Whether the established pair (d0, d1) is a baryon-antibaryon pair
    (|B| = 1 each, opposite signs): the pair class of the running-width
    floor (``resonance_pair_line_shapes``)."""

    b0, b1 = (_qn_of(catalog, d)[0] for d in (d0, d1))
    return abs(b0) == 1 and abs(b1) == 1 and b0 * b1 < 0


def _pair_threshold_mass(catalog: HadronCatalog, pdg_id: int) -> float:
    """The threshold mass of one baryon-pair daughter
    (``resonance_pair_line_shapes``): an unstable record's
    ``mass_range[0]`` (its lowest channel threshold, m_N + m_pi for the
    N* and Delta* records), a stable record's pole mass."""

    try:
        record = catalog.get(int(pdg_id))
    except KeyError:
        return _daughter_pole_mass(catalog, pdg_id)
    if bool(record.stable_for_decay):
        return float(record.pole_mass)
    return float(record.mass_range[0])


def _pair_open_masses(catalog: HadronCatalog, d0: int, d1: int, m0: float, m1: float,
                      parent_mass: float) -> tuple[float, float] | None:
    """The effective opening masses of a direct baryon-antibaryon pair
    (``resonance_pair_line_shapes``), or ``None`` for the committed rule:
    the pair stays closed below the threshold-mass sum
    ``m_th(d0) + m_th(d1)`` and keeps the poles on and above the pole-mass
    sum, and between the two the channel weight evaluates q* at the
    effective masses ``min(m_pole, (m_th + (M - m_other)) / 2)`` -- the
    fixed point of the two daughters' coupled rule, each daughter at the
    midpoint of its threshold and the mass left by the other's effective
    mass (iterated from the thresholds; a stable daughter's threshold is
    its pole, so it stays on its pole)."""

    th0 = _pair_threshold_mass(catalog, d0)
    th1 = _pair_threshold_mass(catalog, d1)
    if not parent_mass > th0 + th1:
        return None
    if parent_mass >= m0 + m1:
        return None
    a, b = th0, th1
    for _ in range(64):
        next_a = min(m0, 0.5 * (th0 + (parent_mass - b)))
        next_b = min(m1, 0.5 * (th1 + (parent_mass - a)))
        if next_a == a and next_b == b:
            break
        a, b = next_a, next_b
    return (a, b)


def continuing_pool(state: CascadeState, channels: list[CascadeChannel],
                    catalog: HadronCatalog, fams) -> list[CascadeChannel]:
    """The continuing draw pool of a continuum vertex (the topology
    cascades' continuation step): the continuation channels and, under
    ``direct_baryon_pairs_in_gap`` (2026-10-21; piece P1b of the card
    TASK-20261021-r106-pbar-and-forward-piplus-mechanisms), the open
    direct baryon-pair channels of a neutral mesonic vertex whose open
    continuation channels include no baryon pair.  The pair channels
    enter scaled by the ratio of the continuation weight to the non-pair
    direct weight, so their share of the pool equals their share of the
    direct pool -- the bare (p pbar, n nbar) share s0 the
    ``_hold_pair_shares`` mechanism holds.  A pool with no continuation
    channel comes back empty (the direct-channel termination of the
    caller)."""

    pool = [c for c in channels if c.kind == KIND_CONTINUUM]
    if not pool:
        return pool
    family_id = str(state.family_id)
    if family_id.startswith(CATALOG_FAMILY_PREFIX):
        return pool
    if not bool(getattr(_family_by_id(fams, family_id), "direct_baryon_pairs_in_gap", False)):
        return pool
    if (int(state.qn[0]), int(state.qn[1]), int(state.qn[2])) != (0, 0, 0):
        return pool
    qn_b = {str(f.family_id): abs(int(f.qn[0])) for f in fams}

    def is_pair(daughters) -> bool:
        total = 0
        for d in daughters:
            if d.startswith(CATALOG_FAMILY_PREFIX):
                total += abs(int(catalog.get(int(d[len(CATALOG_FAMILY_PREFIX):])).B))
            else:
                total += qn_b[d]
        return total >= 2

    if any(is_pair(c.daughters) for c in pool):
        return pool  # a baryon-pair continuation is open: the committed pool
    pairs = [c for c in channels if c.kind != KIND_CONTINUUM and is_pair(c.daughters)]
    if not pairs:
        return pool
    non_pair_direct = math.fsum(c.weight for c in channels
                                if c.kind != KIND_CONTINUUM and not is_pair(c.daughters))
    if not non_pair_direct > 0.0:
        return pool
    continuation_total = math.fsum(c.weight for c in pool)
    factor = continuation_total / non_pair_direct
    return pool + [dataclasses.replace(c, weight=c.weight * factor) for c in pairs]


#: Self-conjugate neutral mesons that join the catalog after the conjugate towers.
_SELF_CONJUGATE_LATE = frozenset({331})


def _charge_conjugate_pdg(pdg_id: int, catalog: HadronCatalog) -> int:
    """The charge conjugate of a catalog pdg id (``na49_mixing.conjugate_towers``):
    a record with nonzero (B, Q, S), or one whose negated id the catalog
    also holds, maps to the negated id; a neutral non-strange meson and the
    photon (22) are their own conjugates."""
    pdg = int(pdg_id)
    if abs(pdg) == 22:
        return pdg
    try:
        record = catalog.get(pdg)
    except KeyError:
        # the eta'(958) joins the catalog after the conjugate towers (KernelConfig.eta_prime): a mode of a baryon
        # record that ends in it (KernelConfig.pdg_nucleon_resonances, 2026-10-21) keeps it as its own conjugate
        return pdg if pdg in _SELF_CONJUGATE_LATE else -pdg
    if (int(record.B), int(record.Q), int(record.S)) != (0, 0, 0):
        return -pdg
    try:
        catalog.get(-pdg)
    except KeyError:
        return pdg
    return -pdg


def _halve_conjugate_pairs(candidates, catalog: HadronCatalog, fams, every: bool = False):
    """The raw candidates of a mesonic vertex with each baryon-pair
    continuation (an established baryon or antibaryon beside a continuum
    daughter of nonzero baryon number) at half its weight when its charge
    conjugate (the conjugate sibling beside the conjugate family) is also
    open.  *every* (a family of the baryon-pair completion, 2026-11-04;
    ``na49_mixing.baryon_pair_completion``) halves every baryon-pair
    continuation, so a pair class carries one unit over its two
    continuations also where the charge conjugate lives at the conjugate
    vertex; at the neutral family the two rules agree."""

    qn = {str(f.family_id): int(f.qn[0]) for f in fams}

    def key(daughters, kind):
        if kind != KIND_CONTINUUM:
            return None
        est = [d for d in daughters if d.startswith(CATALOG_FAMILY_PREFIX)]
        fam = [d for d in daughters if not d.startswith(CATALOG_FAMILY_PREFIX)]
        if len(est) != 1 or len(fam) != 1 or qn[fam[0]] == 0:
            return None
        return int(est[0][len(CATALOG_FAMILY_PREFIX):]), fam[0]

    keys = {key(daughters, kind) for daughters, kind, _ in candidates} - {None}
    out = []
    for daughters, kind, raw in candidates:
        k = key(daughters, kind)
        if k is not None:
            partner = (k[1][len(CONJUGATE_FAMILY_PREFIX):] if k[1].startswith(CONJUGATE_FAMILY_PREFIX)
                       else CONJUGATE_FAMILY_PREFIX + k[1])
            if every or (_charge_conjugate_pdg(k[0], catalog), partner) in keys:
                raw = 0.5 * raw
        out.append((daughters, kind, raw))
    return out


# ----------------------------------------------------------------------
# Channel weight (plan section 10.3) with the fixed hook interface
# ----------------------------------------------------------------------
def channel_weight(
    channel: CascadeChannel,
    w_flavor: float = 1.0,
    w_align: float = 1.0,
) -> float:
    """The plan section 10.3 transparent weight of *channel* with the
    ``W_flavor`` and ``W_align`` hooks applied.

    The base weight ``channel.weight`` is the per-vertex-normalized
    transparent share that :func:`enumerate_channels` constructed:
    ``C_iso * (2J_b+1) * (2J_c+1) * Phi_2 * |F_L(q)|^2`` up to the
    common normalization constant, with the Round 2 tables at catalog
    vertices (the card-5 Clebsch-Gordan isospin shares and the card-4
    Blatt-Weisskopf barrier inside the byte-unchanged width law) and
    the exact two-body phase space at continuum vertices.  This
    function applies the hooks multiplicatively,
    ``weight * W_flavor * W_align``; the caller re-normalizes the
    hooked weights over the vertex, which reproduces the hooked
    raw-weight shares exactly.

    The hooks are fixed at 1.0 at this card's landing: ``W_flavor =
    1.0`` for Round 4 (the flavor weights land in Round 6, plan
    section 22 task 37) and ``W_align = 1.0`` until cards 6-7 attach
    (the interface is frozen).  The function is pure: it consumes zero
    random numbers, so the hooks add zero uniforms to the per-vertex
    draw budget.  A non-finite or negative hook raises ``ValueError``.
    """

    for name, value in (("w_flavor", w_flavor), ("w_align", w_align)):
        value_f = float(value)
        if not (math.isfinite(value_f) and value_f >= 0.0):
            raise ValueError(
                f"channel_weight hook {name} = {value!r} must be finite "
                "and non-negative"
            )
    return channel.weight * float(w_flavor) * float(w_align)


# ----------------------------------------------------------------------
# One decay step (the frozen draw budget: 1 channel + 2 kinematics)
# ----------------------------------------------------------------------
def draw_decay(
    state: CascadeState,
    channels: list[CascadeChannel],
    catalog: HadronCatalog,
    fams,
    rng_channel,
    rng_kinematics,
) -> tuple[CascadeChannel, tuple[CascadeState, CascadeState]]:
    """One decay step: the weighted channel draw (1 uniform on the
    ``decay_channel`` substream) and the Round 2 two-body kinematics
    (2 uniforms on the ``decay_kinematics`` substream).

    Returns ``(channel, (daughter_b, daughter_c))``: the drawn channel
    and the two daughter states with conserved quantum numbers and
    four-momenta (the daughters' momenta sum to the parent's within
    ``DECAY_CLOSURE_TOL``).  The daughter states carry ``depth =
    parent.depth + 1``, the parent link, the inherited exchange
    ancestry, and the parent's axis and alignment label (the card-5
    machinery replaces the axis with the inheritance rules; cards 6-7
    own the label semantics).

    The channel is drawn over the passed list by the Round 2 cumulative
    walk (``target = u * total``, first channel with the cumulative
    weight at or above the target, the last channel as the ``u -> 1``
    clamp), so a hooked or filtered channel list re-normalizes
    automatically and the draw order is unchanged.  The daughters are
    realized on shell at deterministic masses: an established daughter
    at its catalog pole mass, a continuum daughter at the pre-registered
    card-4 mass rule (the module docstring), so no mass draw is
    consumed and the per-vertex budget is exactly 1 + 2 uniforms on the
    closed labels.  A continuum daughter of a family with
    ``continuation_mass_law = "density"`` draws its mass from one more
    uniform on the ``decay_kinematics`` substream, before the two angle
    uniforms (1 + 3 at that vertex).

    A drawn three-body channel raises ``ValueError``: the plan section
    10.2 line 692 three-body clause is the Round 2 kernel's three-body
    vertex, reached through the card-9 catalog handoff, outside this
    two-body step.  Conservation is checked, never repaired: a ``(B,
    Q, S)`` mismatch or a four-momentum closure beyond
    ``DECAY_CLOSURE_TOL`` raises ``KinematicsError``.

    The leading-baryon split draws over the split channels on its branch
    (``u < F``); :func:`draw_decay_with_pool` also returns the channel
    list the draw ran over, the pool of the flavor ledger.  At a root of
    a family with ``meson_split_fraction`` one more uniform on the
    ``decay_channel`` substream, before the channel uniform, picks the
    leading-meson branch, which draws over the meson's split channels.
    """
    selected, daughters, _pool = draw_decay_with_pool(
        state, channels, catalog, fams, rng_channel, rng_kinematics)
    return selected, daughters


#: The split recorder (2026-11-03, the fourteen-parameter fit): ``None`` (the default) or a list that
#: :func:`draw_decay_with_pool` extends with one record per root at which a leading split is eligible. A record holds
#: ``me`` (the meson split eligible), ``be`` (the baryon split eligible, ``None`` on the meson branch, where it is not
#: evaluated), ``branch`` (0 the ordinary draw, 1 the baryon split, 2 the meson split), ``f_b`` (the baryon split's share
#: at this root) and, on a split branch, ``c``, ``kappa`` and ``q``: the directed cosine the law draws, its kappa and the
#: decay momentum. The recorder draws no random number, so the events are those of the recorder off; the fit reweights
#: the split shares and the two direction laws from these records (run_engine, ``split_log_ratio``).
SPLIT_RECORDER: list | None = None


def draw_decay_with_pool(
    state: CascadeState,
    channels: list[CascadeChannel],
    catalog: HadronCatalog,
    fams,
    rng_channel,
    rng_kinematics,
) -> tuple[CascadeChannel, tuple[CascadeState, CascadeState], list[CascadeChannel]]:
    """:func:`draw_decay` with the channel list the draw ran over: the
    split channels on the leading-baryon split's branch, the passed list
    otherwise.  The flavor ledger records that list
    (:func:`~na49_mixing.flavor_weights.vertex_ledger_record`), so the
    reweight of a split-branch vertex normalizes over the split channels
    (review grok-4.7 M1, 2026-10-10)."""

    if not channels:
        raise ValueError(
            "draw_decay: the channel list is empty (no open channel at "
            f"the realized mass {float(state.mass)!r} GeV of family "
            f"{state.family_id!r})"
        )
    total = math.fsum(channel.weight for channel in channels)
    if not (math.isfinite(total) and total > 0.0):
        raise ValueError(
            "draw_decay: the channel weights do not sum to a finite "
            f"positive total ({total!r}); no channel is drawable"
        )

    # -- the leading-meson split (2026-10-11; meson_split_fraction): one
    # uniform before the channel uniform picks the branch, which draws over
    # the meson's split channels with the meson along the incoming baryon
    meson = _meson_split(state, channels, catalog, fams)
    meson_branch = meson is not None and float(rng_channel.random()) < float(meson[0].meson_split_fraction)

    # -- the channel draw: 1 uniform, the Round 2 cumulative walk -------
    u_channel = float(rng_channel.random())
    pool, u_pool, split = channels, u_channel, None
    lead = meson if meson_branch else _leading_split(state, channels, catalog, fams)
    if lead is not None:
        family, subset = lead
        fraction = float(family.leading_split_fraction)
        if u_channel < fraction:
            pool, u_pool, split = subset, u_channel / fraction, family
            total = math.fsum(channel.weight for channel in subset)
        else:
            u_pool = (u_channel - fraction) / (1.0 - fraction)
    split_record = None
    if SPLIT_RECORDER is not None and (meson is not None or lead is not None):
        split_record = {"me": meson is not None, "be": None if meson_branch else lead is not None,
                        "branch": 0 if split is None else (2 if meson_branch else 1),
                        "f_b": None if (meson_branch or lead is None) else fraction}
        SPLIT_RECORDER.append(split_record)
    if split is None and not str(state.family_id).startswith(CATALOG_FAMILY_PREFIX):
        # the string-like step (2026-10-27; string_step_mass): a heavy
        # vertex emits one established hadron at a time, so it draws over
        # its continuation channels; the direct pairs return where the
        # continuation has closed (no open continuation channel)
        step_family = _family_by_id(fams, str(state.family_id))
        step_mass = getattr(step_family, "string_step_mass", None)
        if step_mass is not None and float(state.mass) > float(step_mass):
            strung = [channel for channel in pool if channel.kind == KIND_CONTINUUM]
            if strung:
                pool = strung
                total = math.fsum(channel.weight for channel in pool)
    target = u_pool * total
    selected: CascadeChannel | None = None
    cumulative = 0.0
    for channel in pool:
        cumulative += channel.weight
        if cumulative >= target:
            selected = channel
            break
    if selected is None:
        selected = pool[-1]
    if selected.parent_id != str(state.family_id):
        raise ValueError(
            f"draw_decay: the drawn channel's parent_id "
            f"{selected.parent_id!r} does not match the state's "
            f"family_id {state.family_id!r}"
        )
    if selected.kind == KIND_THREE_BODY:
        return selected, _three_body_daughters(state, selected, catalog, fams, rng_kinematics), list(pool)
    if len(selected.daughters) != 2:
        raise ValueError(
            "draw_decay: the drawn channel "
            f"{tuple(selected.daughters)!r} is not a quasi-two-body "
            "channel; the plan section 10.2 line 692 three-body clause "
            "is the Round 2 kernel's three-body vertex, reached through "
            "the card-9 catalog handoff"
        )

    # -- the daughter identities ----------------------------------------
    identities = [
        _daughter_identity(catalog, fams, daughter_id)
        for daughter_id in selected.daughters
    ]
    # a density-law continuum daughter draws its mass first (1 uniform on
    # the decay_kinematics substream); the card-4 rule draws none
    line_width = None
    parent_family = None
    if not str(state.family_id).startswith(CATALOG_FAMILY_PREFIX):
        parent_family = _family_by_id(fams, str(state.family_id))
        line_width = getattr(parent_family, "sibling_line_width", None)
    masses = _daughter_masses(state, identities, catalog, rng_kinematics, split=split, line_width=line_width,
                              pair_line_shapes=bool(getattr(parent_family, "resonance_pair_line_shapes", False)))

    # -- the Round 2 two-body kinematics: 2 uniforms ---------------------
    u_cos = float(rng_kinematics.random())
    u_phi = float(rng_kinematics.random())
    cos_theta = 2.0 * u_cos - 1.0
    aligned_helicity = False
    if split is not None:
        if meson_branch and getattr(split, "meson_split_slope_strange", None) is not None:
            # the kaon's own direction law on the meson branch (2026-11-06; KernelConfig.meson_split_slope_strange):
            # a strange meson beside the baryonic remainder takes kappa = 2 b_K q*^2, on the same uniform
            cont = _continuing_baryon_slot(identities)
            if cont is not None and int(identities[1 - cont][2].S) != 0:
                split = dataclasses.replace(split, leading_split_kappa=None,
                                            leading_split_slope=float(split.meson_split_slope_strange))
                if split_record is not None:
                    split_record["ms"] = True
        cos_theta = _leading_split_cos(u_cos, state, identities, masses, split)
        if split_record is not None:
            c_law, kappa_law, q_law = _leading_split_law(u_cos, state, masses, split)
            split_record.update(c=c_law, kappa=kappa_law, q=q_law)
        if meson_branch:
            # the split law directs the B = 1 daughter; the meson takes that direction
            cos_theta = -cos_theta
    elif (parent_family is not None and getattr(parent_family, "string_step_mass", None) is not None
            and float(state.mass) > float(parent_family.string_step_mass)
            and _string_forward_slot(identities) is not None):
        # the string-like step of a heavy vertex (2026-10-27): the line's
        # carrier along the line's hemisphere, the sibling toward the center,
        # with kappa = 2 b q*^2 on the same cos(theta) uniform
        cos_theta = _string_step_cos(u_cos, state, identities, masses, parent_family)
    elif (parent_family is not None and getattr(parent_family, "baryon_line_kappa", None) is not None
            and int(state.qn[0]) != 0
            and _continuing_baryon_slot(identities, getattr(parent_family, "baryon_line_kappa_scope", "all")) is not None):
        # the aligned continuing baryonic step (2026-11-05; KernelConfig.baryon_line_kappa): at a baryonic vertex that
        # continues into a baryonic continuum daughter beside an established hadron, the continuing daughter's angle
        # to the line's beam axis follows exp(kappa cos theta) on the same cos(theta) uniform, so the line keeps its
        # baryon leading and the emitted hadron recoils toward the center; the continuation mass law is unchanged
        slot = _continuing_baryon_slot(identities, getattr(parent_family, "baryon_line_kappa_scope", "all"))
        side = 1.0 if float(state.momentum[3]) >= 0.0 else -1.0
        cos_theta = side * _aligned_cos(u_cos, float(parent_family.baryon_line_kappa))
        if slot == 1:
            cos_theta = -cos_theta
        aligned_helicity = getattr(parent_family, "baryon_line_kappa_axis", "beam") == "helicity"
    elif (parent_family is not None and getattr(parent_family, "baryon_line_kappa_strange", None) is not None
            and int(state.qn[0]) != 0 and _strange_transfer_slot(identities) is not None):
        # the strangeness-transfer step reversed (2026-11-05; KernelConfig.baryon_line_kappa_strange): at a continuing
        # baryonic step whose established hadron carries strangeness, the hadron goes along the line's beam axis by
        # exp(kappa cos theta) and the hyperon cluster recoils, on the same uniform
        slot = _strange_transfer_slot(identities)
        side = 1.0 if float(state.momentum[3]) >= 0.0 else -1.0
        cos_theta = -side * _aligned_cos(u_cos, float(parent_family.baryon_line_kappa_strange))
        if slot == 1:
            cos_theta = -cos_theta
    elif (parent_family is not None and getattr(parent_family, "baryon_line_kappa_kaon", None) is not None
            and int(state.qn[0]) != 0 and _yk_meson_slot(identities) is not None):
        # the kaon-led direct pair (2026-11-06; KernelConfig.baryon_line_kappa_kaon): the strange meson of an
        # established hyperon-meson pair along the line's beam axis by exp(kappa cos theta), the hyperon recoiling
        slot = _yk_meson_slot(identities)
        side = 1.0 if float(state.momentum[3]) >= 0.0 else -1.0
        cos_theta = side * _aligned_cos(u_cos, float(parent_family.baryon_line_kappa_kaon))
        if slot == 1:
            cos_theta = -cos_theta
    elif (parent_family is not None and getattr(parent_family, "baryon_line_kappa_final", None) is not None
            and int(state.qn[0]) != 0 and _forward_baryon_slot(identities) is not None):
        # the aligned final baryonic step (2026-11-05; KernelConfig.baryon_line_kappa_final): the established baryon
        # of a line's last step, beside a mesonic continuum daughter, along the line's beam axis by exp(kappa cos
        # theta) on the same uniform (the graded form of the hard hemisphere rule below)
        slot = _forward_baryon_slot(identities)
        side = 1.0 if float(state.momentum[3]) >= 0.0 else -1.0
        cos_theta = side * _aligned_cos(u_cos, float(parent_family.baryon_line_kappa_final))
        if slot == 1:
            cos_theta = -cos_theta
    elif parent_family is not None and bool(getattr(parent_family, "baryon_forward_hemisphere", False)) \
            and int(state.qn[0]) != 0:
        # the forward-hemisphere baryon rule (2026-10-21; KernelConfig.
        # baryon_forward_hemisphere, piece P4): at a non-split decay of a
        # baryonic state into one established baryon and one mesonic
        # continuum daughter, the baryon keeps the parent's forward
        # hemisphere and the mesonic cluster recoils backward, on the same
        # cos(theta) uniform; an antibaryonic line keeps its antibaryon
        # forward as the charge conjugate of the rule (the review of
        # 2026-10-22 removed a sign that sent it backward)
        baryon_slot = _forward_baryon_slot(identities)
        if baryon_slot is not None:
            side = 1.0 if float(state.momentum[3]) >= 0.0 else -1.0
            cos_theta = side * abs(2.0 * u_cos - 1.0)
            if baryon_slot == 1:
                # daughter 1 takes the mirrored direction: its cos(theta)
                # is the negated one
                cos_theta = -cos_theta
    phi = 2.0 * math.pi * u_phi
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    direction = (
        sin_theta * math.cos(phi),
        sin_theta * math.sin(phi),
        cos_theta,
    )
    if split is not None and getattr(parent_family, "leading_split_axis", None) == "gottfried_jackson":
        # the Gottfried-Jackson axis of the split (2026-11-04, the code review grok47; KernelConfig.
        # leading_split_axis): the angle drawn about the beam axis of the root's side is carried onto the incoming
        # baryon's direction in the root's rest frame, the azimuth kept
        direction = _onto_gottfried_jackson(direction, state)
    if aligned_helicity:
        # the aligned continuing step about the parent's own direction of motion (2026-11-05;
        # KernelConfig.baryon_line_kappa_axis = "helicity"): the angle drawn about the line's beam axis is carried
        # onto the parent's momentum direction, the rest-frame helicity axis, the azimuth kept
        direction = _onto_helicity(direction, state)

    parent_mass = float(state.mass)
    mass0, mass1 = masses
    q_star = decay_momentum(parent_mass, mass0, mass1)
    energy0 = (parent_mass ** 2 + mass0 ** 2 - mass1 ** 2) / (2.0 * parent_mass)
    energy1 = (parent_mass ** 2 - mass0 ** 2 + mass1 ** 2) / (2.0 * parent_mass)
    rest0 = (
        energy0,
        q_star * direction[0],
        q_star * direction[1],
        q_star * direction[2],
    )
    rest1 = (
        energy1,
        -q_star * direction[0],
        -q_star * direction[1],
        -q_star * direction[2],
    )
    energy, px, py, pz = (float(x) for x in state.momentum)
    beta = (px / energy, py / energy, pz / energy)
    daughter_p4_0 = _boost_lab(rest0, beta)
    daughter_p4_1 = _boost_lab(rest1, beta)

    # -- conservation, checked never repaired ----------------------------
    qn0 = _daughter_qn(state, identities[0], catalog)
    qn1 = _daughter_qn(state, identities[1], catalog)
    total_qn = (qn0[0] + qn1[0], qn0[1] + qn1[1], qn0[2] + qn1[2])
    state_qn = (int(state.qn[0]), int(state.qn[1]), int(state.qn[2]))
    if total_qn != state_qn:
        raise KinematicsError(
            f"cascade vertex {state.family_id!r} at mass {parent_mass!r} "
            f"-> {tuple(selected.daughters)}: daughters sum to (B, Q, S) "
            f"= {total_qn} but the parent carries {state_qn}"
        )
    closure = (
        abs(daughter_p4_0[0] + daughter_p4_1[0] - energy),
        abs(daughter_p4_0[1] + daughter_p4_1[1] - px),
        abs(daughter_p4_0[2] + daughter_p4_1[2] - py),
        abs(daughter_p4_0[3] + daughter_p4_1[3] - pz),
    )
    worst = max(closure)
    if not worst <= DECAY_CLOSURE_TOL:
        raise KinematicsError(
            f"cascade vertex {state.family_id!r} at mass {parent_mass!r} "
            f"-> {tuple(selected.daughters)}: the daughter four-momenta "
            f"miss the parent four-momentum by {worst!r} GeV (beyond "
            f"{DECAY_CLOSURE_TOL}); no momentum repair"
        )
    for daughter_p4, mass in ((daughter_p4_0, mass0), (daughter_p4_1, mass1)):
        e, dpx, dpy, dpz = daughter_p4
        m2 = e * e - dpx * dpx - dpy * dpy - dpz * dpz
        deviation = abs(m2 - mass * mass)
        if not deviation <= DECAY_CLOSURE_TOL:
            raise KinematicsError(
                f"cascade vertex {state.family_id!r} at mass "
                f"{parent_mass!r} -> {tuple(selected.daughters)}: a "
                f"daughter is off shell (p^2 - m^2 = {deviation!r} "
                f"GeV^2, beyond {DECAY_CLOSURE_TOL}); no momentum repair"
            )

    # -- the daughter states ---------------------------------------------
    state0 = _daughter_state(
        state, identities[0], daughter_p4_0, mass0, catalog
    )
    state1 = _daughter_state(
        state, identities[1], daughter_p4_1, mass1, catalog
    )
    return selected, (state0, state1), list(pool)


def _three_body_daughters(state: CascadeState, channel: CascadeChannel, catalog: HadronCatalog, fams,
                          rng_kinematics) -> tuple[CascadeState, CascadeState, CascadeState]:
    """The daughters of a three-body ending (2026-10-20): the three stable
    hadrons at their pole masses from flat three-body phase space in the
    parent's rest frame (:func:`~na49_mixing.resonance_decay.three_body_phase_space`
    with the ``phase_space`` law, 4 uniforms) turned by a uniform azimuth
    about the sampler's axis (1 uniform, so the point is isotropic) and
    boosted to the parent's frame; (B, Q, S) and the four-momentum closure
    checked, never repaired."""
    from .resonance_decay import three_body_phase_space
    identities = [_daughter_identity(catalog, fams, d) for d in channel.daughters]
    if any(i[0] != "catalog" for i in identities):
        raise ValueError(f"draw_decay: the three-body ending {tuple(channel.daughters)!r} carries a continuum daughter")
    masses = [float(i[2].pole_mass) for i in identities]
    parent_mass = float(state.mass)
    rest = three_body_phase_space(parent_mass, masses[0], masses[1], masses[2], rng_kinematics, law="phase_space")
    phi0 = 2.0 * math.pi * float(rng_kinematics.random())
    c, s = math.cos(phi0), math.sin(phi0)
    energy, px, py, pz = (float(x) for x in state.momentum)
    beta = (px / energy, py / energy, pz / energy)
    p4s = []
    for p in rest:
        e, x, y, z = (float(v) for v in p)
        p4s.append(_boost_lab((e, c * x - s * y, s * x + c * y, z), beta))
    state_qn = (int(state.qn[0]), int(state.qn[1]), int(state.qn[2]))
    qns = [_daughter_qn(state, i, catalog) for i in identities]
    total_qn = tuple(sum(q[j] for q in qns) for j in range(3))
    if total_qn != state_qn:
        raise KinematicsError(
            f"cascade vertex {state.family_id!r} at mass {parent_mass!r} -> {tuple(channel.daughters)}: daughters sum "
            f"to (B, Q, S) = {total_qn} but the parent carries {state_qn}")
    worst = max(abs(math.fsum(p[j] for p in p4s) - (energy, px, py, pz)[j]) for j in range(4))
    if not worst <= DECAY_CLOSURE_TOL:
        raise KinematicsError(
            f"cascade vertex {state.family_id!r} at mass {parent_mass!r} -> {tuple(channel.daughters)}: the daughter "
            f"four-momenta miss the parent four-momentum by {worst!r} GeV (beyond {DECAY_CLOSURE_TOL}); no repair")
    return tuple(_daughter_state(state, identities[k], p4s[k], masses[k], catalog) for k in range(3))  # type: ignore[return-value]


def _daughter_identity(catalog: HadronCatalog, fams, daughter_id: str):
    """The daughter identity behind a channel daughter id:
    ``("catalog", pdg_id, record)`` for an established daughter,
    ``("family", None, family)`` for a continuum daughter."""

    if daughter_id.startswith(CATALOG_FAMILY_PREFIX):
        pdg = int(daughter_id[len(CATALOG_FAMILY_PREFIX):])
        record = catalog.get(pdg)  # KeyError propagates
        return ("catalog", pdg, record)
    return ("family", None, _family_by_id(fams, daughter_id))


def _baryon_number_of(identity) -> int:
    kind, _pdg, obj = identity
    return int(obj.B) if kind == "catalog" else int(obj.qn[0])


def _aligned_cos(u: float, kappa: float) -> float:
    """cos(theta) drawn from exp(kappa cos theta) on [-1, 1] by the inverse
    CDF of the uniform *u* (``KernelConfig.baryon_line_kappa``)."""

    lo = math.exp(-2.0 * kappa)
    return 1.0 + math.log(lo + float(u) * (1.0 - lo)) / kappa


def _continuing_baryon_slot(identities, scope: str = "all") -> "int | None":
    """The daughter slot of the baryonic continuum daughter at a vertex whose
    two daughters are one baryonic continuum state (|B| = 1) and one
    established hadron with B = 0, or ``None``; *scope* "nonstrange" also
    needs the established hadron's S = 0 (``KernelConfig.baryon_line_kappa``,
    ``baryon_line_kappa_scope``)."""

    kinds = [identity[0] for identity in identities]
    if sorted(kinds) != ["catalog", "family"]:
        return None
    cont = 0 if kinds[0] == "family" else 1
    if abs(_baryon_number_of(identities[cont])) != 1 or _baryon_number_of(identities[1 - cont]) != 0:
        return None
    if scope == "nonstrange" and int(identities[1 - cont][2].S) != 0:
        return None
    return cont


def _strange_transfer_slot(identities) -> "int | None":
    """The slot of the baryonic continuum daughter at a continuing baryonic
    step whose established hadron (B = 0) carries strangeness, or ``None``
    (``KernelConfig.baryon_line_kappa_strange``)."""

    cont = _continuing_baryon_slot(identities)
    if cont is None or int(identities[1 - cont][2].S) == 0:
        return None
    return cont


def _yk_meson_slot(identities) -> "int | None":
    """The slot of the strange meson (B = 0, S != 0) at a vertex whose two
    daughters are established hadrons, one hyperon (|B| = 1, S != 0) and one
    strange meson, or ``None`` (``KernelConfig.baryon_line_kappa_kaon``)."""

    if [identity[0] for identity in identities] != ["catalog", "catalog"]:
        return None
    b = [_baryon_number_of(identity) for identity in identities]
    s = [int(identity[2].S) for identity in identities]
    for k in (0, 1):
        if b[k] == 0 and s[k] != 0 and abs(b[1 - k]) == 1 and s[1 - k] != 0:
            return k
    return None


def _forward_baryon_slot(identities) -> "int | None":
    """The daughter slot of the established baryon at a non-split baryonic
    vertex whose two daughters are one established baryon (|B| = 1) and
    one mesonic continuum daughter (B = 0), or ``None``
    (``baryon_forward_hemisphere``)."""

    kinds = [identity[0] for identity in identities]
    if sorted(kinds) != ["catalog", "family"]:
        return None
    baryon = 0 if kinds[0] == "catalog" else 1
    if abs(_baryon_number_of(identities[baryon])) != 1 or _baryon_number_of(identities[1 - baryon]) != 0:
        return None
    return baryon


def _leading_split(state: CascadeState, channels, catalog: HadronCatalog, fams):
    """``(family, split channels)`` at a baryonic continuum root whose family
    carries ``leading_split_fraction``, or ``None`` (every other vertex, or
    no split channel open).  The split channels are the continuation
    channels of an established baryon and a mesonic continuum daughter."""

    if int(state.depth) != 0 or int(state.qn[0]) != 1 or str(state.family_id).startswith(CATALOG_FAMILY_PREFIX):
        return None
    family = _family_by_id(fams, str(state.family_id))
    if getattr(family, "leading_split_fraction", None) is None:
        return None
    # leading_split_strange False (2026-10-28): the hyperon channels leave the
    # split and take their share of its budget with them, so the split
    # fraction at the vertex is F times the non-strange channels' weight
    # share and the non-strange leading baryons are drawn as before; the
    # hyperon channels are drawn in the ordinary isotropic branch
    no_strange = getattr(family, "leading_split_strange", None) is False
    subset, strange_weight = [], 0.0
    for channel in channels:
        ids = [_daughter_identity(catalog, fams, d) for d in channel.daughters]
        kinds = sorted(i[0] for i in ids)
        if len(ids) != 2 or kinds != ["catalog", "family"]:
            continue
        established = ids[0] if ids[0][0] == "catalog" else ids[1]
        continuum = ids[1] if ids[0][0] == "catalog" else ids[0]
        if _baryon_number_of(established) != 1 or _baryon_number_of(continuum) != 0:
            continue
        if no_strange and int(established[2].S) != 0:
            strange_weight += float(channel.weight)
            continue
        subset.append(channel)
    total = math.fsum(channel.weight for channel in subset)
    if not subset or not total > 0.0:
        return None
    if no_strange and strange_weight > 0.0:
        fraction = float(family.leading_split_fraction) * total / (total + strange_weight)
        family = dataclasses.replace(family, leading_split_fraction=fraction)
    return family, subset


def _meson_split(state: CascadeState, channels, catalog: HadronCatalog, fams):
    """``(family, split channels)`` at a root of the leading-baryon split
    whose family carries ``meson_split_fraction``, or ``None``.  The split
    channels are the continuation channels of an established meson and a
    baryonic continuum daughter; the returned family carries the branch's
    split law (fraction 1, and the slope ``meson_split_slope`` when set)."""

    if int(state.depth) != 0 or int(state.qn[0]) != 1 or str(state.family_id).startswith(CATALOG_FAMILY_PREFIX):
        return None
    family = _family_by_id(fams, str(state.family_id))
    if getattr(family, "leading_split_fraction", None) is None or getattr(family, "meson_split_fraction", None) is None:
        return None
    subset = []
    for channel in channels:
        ids = [_daughter_identity(catalog, fams, d) for d in channel.daughters]
        if len(ids) != 2 or sorted(i[0] for i in ids) != ["catalog", "family"]:
            continue
        established = ids[0] if ids[0][0] == "catalog" else ids[1]
        continuum = ids[1] if ids[0][0] == "catalog" else ids[0]
        if _baryon_number_of(established) == 0 and _baryon_number_of(continuum) == 1:
            subset.append(channel)
    if not subset or not math.fsum(channel.weight for channel in subset) > 0.0:
        return None
    options = {"leading_split_fraction": 1.0}
    if getattr(family, "meson_split_slope", None) is not None:
        options.update(leading_split_kappa=None, leading_split_slope=float(family.meson_split_slope))
    return dataclasses.replace(family, **options), subset


#: The Regge intercepts of the leading-baryon split's remainder law (2026-10-28, ``leading_split_regge``): the
#: trajectory the leading baryon exchanges with the root sets the remainder's mass law (mu^2)^(1 - 2 alpha(0)), the
#: triple-Regge flux xi^(1 - 2 alpha(0)) with xi = mu^2 / M^2; the committed law (mu^2)^(-1) is the Pomeron's
#: (alpha = 1). The f/omega family at the Donnachie-Landshoff intercept 1 - 0.4525 (the sigma_tot fit the one-pion
#: exchange uses), the pion and kaon trajectories at alpha' = 0.93 GeV^-2 (the one-pion exchange's slope) through
#: their masses: alpha(0) = -alpha' m^2.
REGGE_ALPHA_PRIME = 0.93
REGGE_INTERCEPTS = {"reggeon": 1.0 - 0.4525, "pion": -REGGE_ALPHA_PRIME * 0.13957 ** 2, "kaon": -REGGE_ALPHA_PRIME * 0.493677 ** 2}


def regge_split_epsilon(root_qn, baryon) -> float:
    """The remainder's epsilon of a leading-baryon split under the Regge-flux
    law: 2 alpha(0) - 2 for the trajectory the leading *baryon* (a catalog
    record) exchanges with the root of quantum numbers *root_qn* (B, Q, S): a
    strangeness exchange is the kaon, a charge exchange the pion, a same-charge
    non-strange baryon the f/omega family."""
    d_s = int(baryon.S) - int(root_qn[2])
    d_q = int(baryon.Q) - int(root_qn[1])
    if d_s != 0:
        alpha = REGGE_INTERCEPTS["kaon"]
    elif d_q != 0:
        alpha = REGGE_INTERCEPTS["pion"]
    else:
        alpha = REGGE_INTERCEPTS["reggeon"]
    return 2.0 * alpha - 2.0


def _split_remainder_mass(family, parent_mass: float, sibling_mass: float, catalog: HadronCatalog | None,
                          u: float | None, split, established=None, root_qn=None) -> float:
    """The mesonic remainder of a leading-baryon split: mu^2 from
    (mu^2)^(-1-epsilon) on [floor^2, edge^2] at the uniform *u* (the
    inverse CDF; epsilon = 0 is flat in ln mu^2).  Under
    ``leading_split_regge`` epsilon comes from the trajectory the
    *established* baryon exchanges with the root (``regge_split_epsilon``)."""

    edge = min(_continuation_ceiling(family), float(parent_mass) - float(sibling_mass))
    floor = _continuation_floor(family, catalog)
    if not edge >= floor:
        raise ValueError(
            f"continuum daughter {family.family_id!r}: the on-shell mass budget window [{floor!r}, {edge!r}] "
            f"is empty at parent mass {float(parent_mass)!r} GeV (a stale channel list; checked, never repaired)")
    if u is None:
        raise ValueError(f"continuum daughter {family.family_id!r}: the leading-baryon split needs the density uniform")
    a, b = floor * floor, edge * edge
    if not b > a:
        return floor
    if getattr(split, "leading_split_regge", None) and established is not None and root_qn is not None:
        eps = regge_split_epsilon(root_qn, established)
    else:
        eps = float(split.leading_split_epsilon or 0.0)
    if abs(eps) < 1e-12:
        mu2 = a * (b / a) ** float(u)
    else:
        mu2 = (a ** (-eps) + float(u) * (b ** (-eps) - a ** (-eps))) ** (-1.0 / eps)
    return math.sqrt(min(max(mu2, a), b))


def _leading_split_law(u: float, state: CascadeState, masses, split) -> tuple[float, float, float]:
    """``(c, kappa, q)`` of a leading split at the uniform *u*: the directed
    cosine c from p(c) = kappa exp(kappa (c - 1)) / (1 - exp(-2 kappa)) on
    [-1, 1] by the inverse CDF, its kappa (the fixed ``leading_split_kappa``
    or 2 b q*^2 under ``leading_split_slope`` b), and the decay momentum q*."""

    q = decay_momentum(float(state.mass), float(masses[0]), float(masses[1]))
    if split.leading_split_slope is not None:
        kappa = 2.0 * float(split.leading_split_slope) * q * q
    else:
        kappa = float(split.leading_split_kappa or 0.0)
    if kappa == 0.0:
        c = 2.0 * float(u) - 1.0
    else:
        inner = float(u) + (1.0 - float(u)) * math.exp(-2.0 * kappa)
        c = max(-1.0, 1.0 + math.log(inner) / kappa) if inner > 0.0 else -1.0
    return c, kappa, q


def _onto_helicity(direction, state: CascadeState):
    """*direction* (a rest-frame unit vector whose polar angle is taken about
    the beam axis of the state's side, (0, 0, sign p_z)) rotated so that the
    same angle is taken about the state's own direction of motion, the
    rest-frame helicity axis (``KernelConfig.baryon_line_kappa_axis``); a
    state at rest leaves *direction*."""

    px, py, pz = (float(x) for x in state.momentum[1:])
    p = math.sqrt(px * px + py * py + pz * pz)
    if not p > 0.0:
        return direction
    side = 1.0 if pz >= 0.0 else -1.0
    ax = (0.0, 0.0, side)
    n = (px / p, py / p, pz / p)
    c = ax[2] * n[2]
    k = (ax[1] * n[2] - ax[2] * n[1], ax[2] * n[0] - ax[0] * n[2], ax[0] * n[1] - ax[1] * n[0])
    s_ = math.sqrt(k[0] * k[0] + k[1] * k[1] + k[2] * k[2])
    if not s_ > 1e-15:
        return direction
    k = tuple(x / s_ for x in k)
    v = direction
    kv = k[0] * v[0] + k[1] * v[1] + k[2] * v[2]
    kxv = (k[1] * v[2] - k[2] * v[1], k[2] * v[0] - k[0] * v[2], k[0] * v[1] - k[1] * v[0])
    return tuple(v[j] * c + kxv[j] * s_ + k[j] * kv * (1.0 - c) for j in range(3))


def _onto_gottfried_jackson(direction, state: CascadeState):
    """*direction* (a rest-frame unit vector whose polar angle is taken about
    the beam axis of the root's side, (0, 0, sign p_z)) rotated so that the
    same angle is taken about the Gottfried-Jackson axis: the incoming
    baryon of that side, a lightlike vector along the state's axis in the
    c.m. frame (the proton at 158 GeV/c has beta 0.994 there), boosted into
    the root's rest frame.  The rotation is the shortest one from the beam
    axis to that axis (Rodrigues); a root at rest leaves *direction*."""

    energy, px, py, pz = (float(x) for x in state.momentum)
    side = 1.0 if pz >= 0.0 else -1.0
    ax = tuple(side * float(a) for a in state.axis)
    mass = float(state.mass)
    b = (px / energy, py / energy, pz / energy)
    b2 = b[0] * b[0] + b[1] * b[1] + b[2] * b[2]
    if not b2 > 0.0:
        return direction
    gamma = energy / mass
    bk = b[0] * ax[0] + b[1] * ax[1] + b[2] * ax[2]
    f = (gamma - 1.0) * bk / b2 - gamma
    n = tuple(ax[j] + f * b[j] for j in range(3))
    norm = math.sqrt(sum(x * x for x in n))
    n = tuple(x / norm for x in n)
    # the rotation taking ax onto n
    c = ax[0] * n[0] + ax[1] * n[1] + ax[2] * n[2]
    k = (ax[1] * n[2] - ax[2] * n[1], ax[2] * n[0] - ax[0] * n[2], ax[0] * n[1] - ax[1] * n[0])
    s_ = math.sqrt(k[0] * k[0] + k[1] * k[1] + k[2] * k[2])
    if not s_ > 1e-15:
        return direction
    k = tuple(x / s_ for x in k)
    v = direction
    kv = k[0] * v[0] + k[1] * v[1] + k[2] * v[2]
    kxv = (k[1] * v[2] - k[2] * v[1], k[2] * v[0] - k[0] * v[2], k[0] * v[1] - k[1] * v[0])
    return tuple(v[j] * c + kxv[j] * s_ + k[j] * kv * (1.0 - c) for j in range(3))


def _leading_split_cos(u: float, state: CascadeState, identities, masses, split) -> float:
    """cos(theta) of daughter 0 at a leading-baryon split: the baryon's
    direction from exp(kappa cos theta) about the incoming baryon, the
    sign of the root's c.m. p_z (the cascade runs in the c.m. frame, the
    beam along z), by the inverse CDF at *u* (:func:`_leading_split_law`)."""

    c, _kappa, _q = _leading_split_law(u, state, masses, split)
    side = 1.0 if float(state.momentum[3]) >= 0.0 else -1.0
    return side * c if _baryon_number_of(identities[0]) == 1 else -side * c


def _string_forward_slot(identities):
    """The daughter that carries the line on at a string-like step
    (2026-10-27): the baryon-number carrier when exactly one daughter has
    one, else the continuum daughter; ``None`` for a channel without exactly
    one continuum daughter (a direct pair, a two-continuum channel)."""

    kinds = [identity[0] for identity in identities]
    if sorted(kinds) != ["catalog", "family"]:
        return None
    b = [_baryon_number_of(identity) for identity in identities]
    if b[0] != 0 and b[1] == 0:
        return 0
    if b[1] != 0 and b[0] == 0:
        return 1
    return kinds.index("family")


def _string_step_cos(u: float, state: CascadeState, identities, masses, family) -> float:
    """cos(theta) of daughter 0 at a string-like step (2026-10-27): the
    line's carrier along the line's hemisphere (the sign of the c.m. p_z,
    the beam along z) and the sibling toward the center, from
    exp(kappa cos theta) with kappa = 2 b q*^2 (``string_step_slope`` b,
    4 GeV^-2 when unset), so the sibling's p_T follows exp(-b p_T^2) at
    every vertex mass; by the inverse CDF at *u*."""

    slope = getattr(family, "string_step_slope", None)
    b = 4.0 if slope is None else float(slope)
    q = decay_momentum(float(state.mass), float(masses[0]), float(masses[1]))
    kappa = 2.0 * b * q * q
    if kappa == 0.0:
        c = 2.0 * float(u) - 1.0
    else:
        inner = float(u) + (1.0 - float(u)) * math.exp(-2.0 * kappa)
        c = max(-1.0, 1.0 + math.log(inner) / kappa) if inner > 0.0 else -1.0
    side = 1.0 if float(state.momentum[3]) >= 0.0 else -1.0
    return side * c if _string_forward_slot(identities) == 0 else -side * c


#: The node count of the density-integrated sibling phase space (uniform in
#: ``s = sqrt(edge - m)``, the substitution of the density step).
_SIBLING_GRID_POINTS: int = 65


def _with_continuation_scale(family, parent_mass: float):
    """The family with the density step's scale at the parent mass: under
    ``continuation_t_h_alpha = alpha`` (2026-10-07) ``t_h`` becomes
    ``max(t_h, alpha * M_parent)``; otherwise the family itself."""
    alpha = getattr(family, "continuation_t_h_alpha", None)
    if alpha is None:
        return family
    t_h = max(float(family.t_h), float(alpha) * float(parent_mass))
    return family if t_h == float(family.t_h) else dataclasses.replace(family, t_h=t_h)


def _density_integrated_phase_space(family, parent_mass: float, sibling_mass: float,
                                    floor: float, edge: float) -> float:
    """The sibling weight of a continuation channel under
    ``continuation_sibling_weight = "density"`` (2026-10-06; the code and
    physics review's M-2, ``TASK-20261005-cascade-vertex-rules`` item 2):
    ``int rho_fam(m) q*^(2L+1) F_L(q*)^2 dm`` over the window of the density
    step, ``[max(floor, edge - 16 t_H), edge]``, where ``rho_fam(m) =
    m**(-b) exp(m / t_H)`` is normalized on the family support ``[floor,
    ceiling]`` and ``q* = q(M; m, m_sibling)``.  It replaces the card-4
    weight ``q*`` at the fixed mass ``max(floor, edge - nominal_width)``, so
    the channel weight and the realized daughter mass follow one law.  A
    light sibling leaves a heavier daughter with exponentially more states
    (the factor ``exp((edge - ceiling) / t_H)``).  Trapezoid on
    :data:`_SIBLING_GRID_POINTS` nodes in ``s``; scalar arithmetic only."""
    family = _with_continuation_scale(family, parent_mass)
    floor, edge = float(floor), float(edge)
    if edge < floor:
        raise ValueError(f"continuum daughter {family.family_id!r}: the on-shell edge {edge!r} GeV lies below "
                         f"the continuation floor {floor!r} GeV")
    if not edge > floor:
        return 0.0
    t_h, b_power = float(family.t_h), float(family.b_power)
    ceiling = _continuation_ceiling(family)
    barrier_l = int(getattr(family, "continuation_barrier_l", None) or 0)
    low = max(floor, edge - _DENSITY_WINDOW_T_H * t_h)
    s_max = math.sqrt(edge - low)
    n = _SIBLING_GRID_POINTS
    total, previous = 0.0, None
    for j in range(n):
        s = s_max * (n - 1 - j) / (n - 1)
        m = min(max(edge - s * s, low), edge)
        q = decay_momentum(float(parent_mass), m, float(sibling_mass))
        w = math.exp((m - ceiling) / t_h) * q
        if barrier_l:
            f = BLATT_WEISSKOPF_BARRIER(barrier_l, q)
            q2 = q * q
            for _ in range(barrier_l):
                w *= q2
            w *= f * f
        if b_power != 0.0:
            w *= m ** (-b_power)
        g = 2.0 * s * w
        if previous is not None:
            total += 0.5 * (g + previous[0]) * (previous[1] - s)
        previous = (g, s)
    return total / _family_support_norm(family, floor, ceiling)


#: Pure compute cache of the family support normalizations.
_SUPPORT_NORM_CACHE: dict[tuple, float] = {}


def _family_support_norm(family, floor: float, ceiling: float) -> float:
    """``int_floor^ceiling m**(-b) exp((m - ceiling) / t_H) dm``: closed form
    at ``b = 0``, a 4,001-node trapezoid otherwise."""
    t_h, b_power = float(family.t_h), float(family.b_power)
    key = (t_h, b_power, float(floor), float(ceiling))
    value = _SUPPORT_NORM_CACHE.get(key)
    if value is None:
        if b_power == 0.0:
            value = t_h * (1.0 - math.exp((floor - ceiling) / t_h))
        else:
            n = 4001
            xs = [floor + (ceiling - floor) * k / (n - 1) for k in range(n)]
            ys = [x ** (-b_power) * math.exp((x - ceiling) / t_h) for x in xs]
            value = math.fsum(0.5 * (ys[k] + ys[k - 1]) * (xs[k] - xs[k - 1]) for k in range(1, n))
        _SUPPORT_NORM_CACHE[key] = value
    return value


def _density_daughter_mass(family, parent_mass: float, sibling_mass: float,
                           floor: float, edge: float, u: float,
                           barrier_override: "int | None" = None) -> float:
    """The density-weighted realized mass of a continuation daughter on
    ``[floor, edge]``: the inverse CDF, at the uniform *u*, of
    ``m**(-b_power) * exp((m - edge) / t_h) * q*(M; m, m_other)``.

    With the family's ``continuation_barrier_l = L >= 1`` the factor
    ``q*`` becomes ``q***(2L+1) * F_L(q*)**2`` (the Blatt-Weisskopf barrier
    of the catalog vertices; the powers as products); L = 0 or the absent
    key is the plain ``q*`` law, bit for bit.  A *barrier_override* (an
    int) replaces that L in this draw (``baryon_pair_s_wave``, 2026-10-21:
    the S-wave of a baryon-antibaryon creation step, L = 0 on the same
    uniform).

    The integral runs in ``s = sqrt(edge - m)`` (``dm = 2 s ds``): the
    decay momentum opens like ``sqrt(edge - m)`` at the two-body
    threshold, so the integrand is smooth in ``s`` and the nodes crowd at
    the edge.  The grid has :data:`_DENSITY_GRID_POINTS` nodes uniform in
    ``s`` over ``[max(floor, edge - _DENSITY_WINDOW_T_H * t_h), edge]``;
    the CDF is the trapezoid integral, linear in ``s`` between the nodes
    (scalar arithmetic only)."""
    family = _with_continuation_scale(family, parent_mass)

    floor, edge = float(floor), float(edge)
    if edge < floor:
        # the callers skip or refuse a window closed below the floor, so a
        # closed window here is a contract violation (code-physics review
        # L-4, 2026-10-06), never repaired to the floor
        raise ValueError(
            f"continuum daughter {family.family_id!r}: the on-shell edge {edge!r} GeV lies below the "
            f"continuation floor {floor!r} GeV at parent mass {parent_mass!r} GeV"
        )
    if not edge > floor:
        return floor
    n = _DENSITY_GRID_POINTS
    t_h = float(family.t_h)
    b_power = float(family.b_power)
    low = max(floor, edge - _DENSITY_WINDOW_T_H * t_h)
    s_max = math.sqrt(edge - low)
    parent_mass, sibling_mass = float(parent_mass), float(sibling_mass)
    # nodes in ascending mass: s runs from s_max (m = low) down to 0 (m = edge)
    s_nodes = [s_max * (n - 1 - j) / (n - 1) for j in range(n)]
    m_nodes = [min(max(edge - s * s, low), edge) for s in s_nodes]
    m_nodes[0], m_nodes[-1] = low, edge
    g = []
    barrier_l = int(getattr(family, "continuation_barrier_l", None) or 0)
    if barrier_override is not None:
        barrier_l = int(barrier_override)
    for s, m in zip(s_nodes, m_nodes):
        q = decay_momentum(parent_mass, m, sibling_mass)
        w = math.exp((m - edge) / t_h) * q
        if barrier_l:
            f = BLATT_WEISSKOPF_BARRIER(barrier_l, q)
            q2 = q * q
            for _ in range(barrier_l):
                w *= q2
            w *= f * f
        if b_power != 0.0:
            w *= (edge / m) ** b_power
        g.append(2.0 * s * w)
    cumulative = [0.0]
    for j in range(1, n):
        cumulative.append(cumulative[-1] + 0.5 * (g[j] + g[j - 1]) * (s_nodes[j - 1] - s_nodes[j]))
    total = cumulative[-1]
    if not (math.isfinite(total) and total > 0.0):
        raise ValueError(
            f"continuum daughter {family.family_id!r}: the density-weighted "
            f"mass law has no weight on [{floor!r}, {edge!r}] at parent mass "
            f"{parent_mass!r} GeV (checked, never repaired)"
        )
    target = float(u) * total
    k = 1
    while k < n - 1 and cumulative[k] < target:
        k += 1
    lo, hi = cumulative[k - 1], cumulative[k]
    frac = 0.0 if not hi > lo else (target - lo) / (hi - lo)
    s = s_nodes[k - 1] + frac * (s_nodes[k] - s_nodes[k - 1])
    return min(max(edge - s * s, low), edge)


def _continuum_daughter_mass(family, parent_mass: float, sibling_mass: float,
                             catalog: HadronCatalog | None = None,
                             u: float | None = None,
                             barrier_override: "int | None" = None) -> float:
    """The realized mass of a continuum daughter: the pre-registered
    card-4 minimal rule (deterministic, no random number), or, for a
    family with ``continuation_mass_law = "density"``, the
    density-weighted draw at the uniform *u* (2026-10-03).  A
    *barrier_override* (an int) replaces the family's orbital barrier in
    that draw (``baryon_pair_s_wave``, 2026-10-21)."""

    edge = min(_continuation_ceiling(family), float(parent_mass) - float(sibling_mass))
    floor = _continuation_floor(family, catalog)
    if not edge >= floor:
        raise ValueError(
            f"continuum daughter {family.family_id!r}: the on-shell mass "
            f"budget window [{floor!r}, {edge!r}] is empty "
            f"at parent mass {float(parent_mass)!r} GeV (a stale channel "
            "list; checked, never repaired)"
        )
    step_mass = getattr(family, "string_step_mass", None)
    if step_mass is not None and float(parent_mass) > float(step_mass):
        # the string-like step of a heavy vertex (2026-10-27): the daughter's
        # mass from (M'^2)^(-1-eps) on [floor^2, edge^2], flat in ln M'^2 at
        # the default eps = 0, on the density law's uniform
        if u is None:
            raise ValueError(
                f"continuum daughter {family.family_id!r}: the string-like "
                "step needs a uniform (the decay_kinematics substream)"
            )
        if not edge > floor:
            return floor
        eps = getattr(family, "string_step_epsilon", None)
        return draw_line_mass(floor, edge, 0.0 if eps is None else float(eps), float(u))
    if getattr(family, "continuation_mass_law", None) == "density":
        if u is None:
            raise ValueError(
                f"continuum daughter {family.family_id!r}: the density mass "
                "law needs a uniform (the decay_kinematics substream)"
            )
        return _density_daughter_mass(family, parent_mass, sibling_mass, floor, edge, u,
                                      barrier_override=barrier_override)
    return max(
        floor,
        edge - float(family.nominal_width),
    )


def _daughter_masses(state: CascadeState, identities, catalog: HadronCatalog | None = None,
                     rng_kinematics=None, split=None, line_width=None,
                     pair_line_shapes: bool = False) -> tuple[float, float]:
    """The realized daughter masses: the established daughters at their
    pole masses, a continuum daughter at its family's mass law (which
    needs the sibling's realized mass; a two-continuum channel has no
    card-4 mass rule and raises).  A density-law daughter draws one
    uniform from *rng_kinematics*; the card-4 rule draws none.  At a
    leading-baryon split (*split*, the root's family) the mesonic
    daughter takes the split's power law at the same uniform.  With
    *line_width* (the parent family's ``sibling_line_width``) an
    established daughter at least that wide draws its mass from its line
    shape (:func:`_sibling_line_mass`, one uniform before the density
    uniform) on [threshold, min(m_up, M - m_other)]: m_other is the
    continuum daughter's floor, or the other established daughter's
    threshold (broad) or pole, and the second broad daughter of a pair
    takes the first one's realized mass.  With *pair_line_shapes* (the
    parent family's ``resonance_pair_line_shapes``, 2026-10-21, piece P1a)
    a direct channel's unstable baryon-antibaryon daughter (|B| = 1) also
    draws its mass from its line shape on the same window, one uniform per
    drawn daughter."""

    if line_width is not None or pair_line_shapes:
        return _line_shape_daughter_masses(state, identities, catalog, rng_kinematics, split,
                                           None if line_width is None else float(line_width),
                                           bool(pair_line_shapes))

    def _u(family):
        if getattr(family, "continuation_mass_law", None) != "density":
            return None
        if rng_kinematics is None:
            raise ValueError("draw_decay: the density mass law needs the decay_kinematics substream")
        return float(rng_kinematics.random())

    def _barrier_override(family) -> "int | None":
        # the S-wave law of a baryon-pair creation step (2026-10-21;
        # KernelConfig.baryon_pair_s_wave, piece P3): at a continuation
        # step of a B = 0 state into a baryonic family the density draw
        # uses the plain q* law (L = 0, the lowest wave of a pair from a
        # negative-parity cluster; a convention, the continuum has no J^P)
        if not getattr(family, "baryon_pair_s_wave", False):
            return None
        if int(state.qn[0]) != 0 or int(family.qn[0]) == 0:
            return None
        return 0

    kind0, _, obj0 = identities[0]
    kind1, _, obj1 = identities[1]
    if kind0 == "catalog" and kind1 == "catalog":
        return (float(obj0.pole_mass), float(obj1.pole_mass))
    # the established daughter (the leading baryon, or the leading meson of the meson split) for the Regge law
    established = obj0 if kind0 == "catalog" else (obj1 if kind1 == "catalog" else None)
    regge_record = established if (established is not None and int(established.B) != 0) else None
    mass_of = _continuum_daughter_mass if split is None else (
        lambda family, parent, sibling, cat, u, barrier_override=None:
            _split_remainder_mass(family, parent, sibling, cat, u, split, established=regge_record,
                                  root_qn=tuple(int(q) for q in state.qn)))
    if kind0 == "catalog" and kind1 == "family":
        m0 = float(obj0.pole_mass)
        return (m0, mass_of(obj1, state.mass, m0, catalog, _u(obj1), _barrier_override(obj1)))
    if kind0 == "family" and kind1 == "catalog":
        m1 = float(obj1.pole_mass)
        return (mass_of(obj0, state.mass, m1, catalog, _u(obj0), _barrier_override(obj0)), m1)
    raise ValueError(
        "draw_decay: the channel "
        f"({identities[0][2].family_id!r}, {identities[1][2].family_id!r}) "
        "carries two continuum daughters; the card-4 enumeration "
        "produces at most one continuum daughter per channel (the "
        "quasi-two-body cascade line)"
    )



def _is_line_shape_daughter(identity, line_width) -> bool:
    kind, _pdg, obj = identity
    return (kind == "catalog" and line_width is not None and not bool(obj.stable_for_decay)
            and float(obj.nominal_width) >= line_width)


def _is_pair_line_daughter(identity) -> bool:
    """A direct channel's established daughter that draws its mass from
    its line shape under ``resonance_pair_line_shapes`` (piece P1a,
    2026-10-21): an unstable catalog record of |B| = 1 (a baryon or
    antibaryon resonance; the stable p, n and their antibaryons stay at
    their poles and the meson daughters keep the committed rule)."""

    kind, _pdg, obj = identity
    return kind == "catalog" and not bool(obj.stable_for_decay) and abs(int(obj.B)) == 1


def _line_shape_daughter_masses(state: CascadeState, identities, catalog, rng_kinematics, split,
                                line_width: float | None, pair_line_shapes: bool = False) -> tuple[float, float]:
    """:func:`_daughter_masses` with the line-shape masses of the broad
    established daughters (the parent family's ``sibling_line_width``)
    and, under *pair_line_shapes*, of the unstable baryon-antibaryon
    daughters of a direct channel (``resonance_pair_line_shapes``)."""

    broad = [_is_line_shape_daughter(identity, line_width) for identity in identities]
    if pair_line_shapes and identities[0][0] == "catalog" and identities[1][0] == "catalog":
        broad = [b or _is_pair_line_daughter(identity) for b, identity in zip(broad, identities)]
    if not any(broad):
        return _daughter_masses(state, identities, catalog, rng_kinematics, split=split)
    if rng_kinematics is None or catalog is None:
        raise ValueError("draw_decay: the line-shape daughter masses need the catalog and the decay_kinematics substream")
    parent_mass = float(state.mass)
    kinds = [identity[0] for identity in identities]

    def _drawn(identity, other):
        record = identity[2]
        hi = min(float(record.mass_range[1]), parent_mass - float(other))
        return _sibling_line_mass(record, hi, catalog, float(rng_kinematics.random()))

    if kinds == ["catalog", "catalog"]:
        m0, m1 = float(identities[0][2].pole_mass), float(identities[1][2].pole_mass)
        if broad[0]:
            m0 = _drawn(identities[0], float(identities[1][2].mass_range[0]) if broad[1] else m1)
        if broad[1]:
            m1 = _drawn(identities[1], m0)
        return (m0, m1)
    i_cat = kinds.index("catalog")
    if not broad[i_cat]:
        return _daughter_masses(state, identities, catalog, rng_kinematics, split=split)
    family = identities[1 - i_cat][2]
    mass = _drawn(identities[i_cat], _continuation_floor(family, catalog))
    record = identities[i_cat][2]
    shifted = list(identities)
    shifted[i_cat] = (identities[i_cat][0], identities[i_cat][1], dataclasses.replace(record, pole_mass=mass))
    return _daughter_masses(state, shifted, catalog, rng_kinematics, split=split)


#: The line-shape tables of the broad established daughters, per catalog
#: (weakly keyed) and pdg id: the inverse-CDF table of the record's whole
#: support (resonance_widths._draw_entry), truncated per draw.
_LINE_SHAPE_TABLES: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _sibling_line_mass(record, hi: float, catalog: HadronCatalog, u: float) -> float:
    """The mass of a broad established daughter from its line shape (the
    resonance decayer's spectral function with its daughter-mass map and
    the Blatt-Weisskopf barrier) on [threshold, hi], by the inverse of the
    piecewise-linear CDF of the record's whole support truncated at *hi*,
    at the uniform *u*.  An empty window raises (checked, never repaired)."""

    tables = _LINE_SHAPE_TABLES.setdefault(catalog, {})
    pdg = int(record.pdg_id)
    entry = tables.get(pdg)
    if entry is None:
        decayer = ResonanceDecayer(catalog, 0, 0, 0)
        entry = _draw_entry(record, decayer._daughter_mass_map(record), float(record.mass_range[0]),
                            float(record.mass_range[1]), BLATT_WEISSKOPF_BARRIER, catalog)
        if entry is None:
            raise ValueError(f"draw_decay: the line shape of {record.name} (pdg {pdg}) carries no mass")
        tables[pdg] = entry
    lo_c, hi_c, cdf, cell = entry
    top_mass = min(float(hi), hi_c)
    if not top_mass >= lo_c:
        raise ValueError(f"draw_decay: the on-shell window of {record.name} (pdg {pdg}) is empty "
                         f"([{lo_c!r}, {float(hi)!r}] GeV; checked, never repaired)")
    if hasattr(cell, "shape"):
        # a narrow line's nonuniform grid (resonance_widths._draw_entry, 2026-11-04): the same inverse on its nodes
        xs = cell
        k = min(max(bisect.bisect_right(list(xs), top_mass) - 1, 0), cdf.size - 1)
        x = k + (top_mass - xs[k]) / (xs[k + 1] - xs[k])
    else:
        x = (top_mass - lo_c) / cell
        k = min(int(x), cdf.size - 1)
    below = float(cdf[k - 1]) if k > 0 else 0.0
    top = below + (x - k) * (float(cdf[k]) - below)
    if not top > 0.0:
        return lo_c
    target = float(u) * top
    idx = min(bisect.bisect_left(cdf, target), cdf.size - 1)
    below = float(cdf[idx - 1]) if idx > 0 else 0.0
    width = float(cdf[idx]) - below
    position = (target - below) / width if width > 0.0 else 0.5
    if hasattr(cell, "shape"):
        return float(min(max(float(cell[idx]) + position * (cell[idx + 1] - cell[idx]), lo_c), top_mass))
    return float(min(max(lo_c + (idx + position) * cell, lo_c), top_mass))

def _daughter_qn(
    state: CascadeState, identity, catalog: HadronCatalog
) -> tuple[int, int, int]:
    """The ``(B, Q, S)`` of one daughter."""

    kind, pdg, obj = identity
    if kind == "catalog":
        return (int(obj.B), int(obj.Q), int(obj.S))
    return (int(obj.qn[0]), int(obj.qn[1]), int(obj.qn[2]))


def _daughter_state(
    state: CascadeState,
    identity,
    four_momentum: tuple[float, float, float, float],
    mass: float,
    catalog: HadronCatalog,
) -> CascadeState:
    """The daughter :class:`CascadeState`: the realized four-momentum
    and mass, the daughter's quantum numbers, the family identity, the
    parent link, the inherited exchange ancestry, the parent's axis and
    alignment label (the card-5 machinery replaces the axis), and the
    incremented depth counter."""

    kind, pdg, obj = identity
    momentum = (
        float(four_momentum[0]),
        float(four_momentum[1]),
        float(four_momentum[2]),
        float(four_momentum[3]),
    )
    axis = (
        float(state.axis[0]),
        float(state.axis[1]),
        float(state.axis[2]),
    )
    parent = state
    ancestry = tuple(state.exchange_ancestry)
    depth = int(state.depth) + 1
    if kind == "catalog":
        return CascadeState(
            momentum=momentum,
            mass=float(mass),
            qn=(int(obj.B), int(obj.Q), int(obj.S)),
            quantum=_quantum_of(catalog, pdg),
            family_id=f"{CATALOG_FAMILY_PREFIX}{int(pdg)}",
            axis=axis,
            alignment_label=str(state.alignment_label),
            parent=parent,
            exchange_ancestry=ancestry,
            depth=depth,
        )
    return CascadeState(
        momentum=momentum,
        mass=float(mass),
        qn=(int(obj.qn[0]), int(obj.qn[1]), int(obj.qn[2])),
        quantum={},
        family_id=str(obj.family_id),
        axis=axis,
        alignment_label=str(state.alignment_label),
        parent=parent,
        exchange_ancestry=ancestry,
        depth=depth,
    )
