"""Isospin charge-channel completion and the recursive two-body decay
kernel for the Round 2 resonance catalog.

Task ``TASK-20260913-r2-isospin-charge-completion`` (Round 2, card 5;
plan section 4.3 part 1, plan section 8.3 step 3, plan section 22
task 13) and task ``TASK-20260913-r2-recursive-two-body-decays``
(Round 2, card 6; plan section 22 task 14, plan section 8.4 step 4).
The module derives the per-charge weights of the two-body
color-singlet decay channels from Clebsch-Gordan amplitudes where the
card-2 import left a representative charge channel, audits the
completed catalog so that every charge-channel weight is either
tabulated (with provenance) or derived (tagged ``isospin-derived``),
and implements the card-6 recursive two-body decay kernel over the
completed catalog.  Card 7 extends the kernel with the selected
three-body vertex; the isospin and two-body surfaces defined here are
stable for it.

The six pre-registered conventions (card, Physics rule):

1. **Tabulated wins, derived completes.**  Per-charge tabulated
   branching fractions are used as-is (the audit verifies their class
   sum and that their pattern matches the isospin-allowed set).  A
   single representative channel carrying the full multiplet branching
   fraction is completed from the Clebsch-Gordan shares
   ``w = BR * |CG|^2 / sum_allowed |CG|^2`` over the I3-conserving,
   non-forbidden combinations.  The multiplet branching fraction is
   preserved exactly: the representative's recorded value *is* the
   multiplet branching fraction, so completing the class redistributes
   it over the allowed charges and re-tags the representative as
   derived (the scope decision is documented in the run record).
2. **Total-I selection.**  The parent record's ``isospin`` selects the
   component of the daughter product (N-pi: ``1/2 (x) 1 =
   1/2 (oplus) 3/2``).  If the parent isospin is not in the product,
   the channel is forbidden: zero weight, and a non-zero catalog
   branching fraction is an audit error.
3. **Bose rule.**  Two identical daughters with odd
   ``orbital_angular_momentum`` have zero weight; a non-zero catalog
   branching fraction for such a channel (``rho0 -> pi0 pi0`` at L = 1)
   is an audit error.
4. **Anti-particle mirror.**  Anti-parent weights are the
   charge-conjugated mirror of the parent's; the audit verifies each
   reflect-partner record (``isospin.reflect_pdg``) and each existing
   charge-conjugate partner with mirrored weights and flags
   ``mirror_mismatches`` beyond ``1e-9``.
5. **Charge completeness.**  Every isospin-allowed charge partner
   record exists (the census runs per isospin multiplet and names
   missing charge states through ``isospin.reflect_pdg``), and every
   isospin-allowed charge-combination channel exists.
6. **B, Q, S conservation.**  Every weight assignment conserves
   ``(B, Q, S)`` exactly at the channel level.

Catalog conventions this module relies on (card 2): the neutral-kaon
flavor pair (K0, K0bar) is recorded in channels as the weak-eigenstate
pair (K0S, K0L) = (310, 130) with zero net strangeness, and the stable
nucleon pair (p, n) and anti pair (pbar, nbar) form one isospin
multiplet each although their record names differ.  The three-body
channels are outside the card-5 charge completion (card 7); the audit
still checks their I3 conservation and (B, Q, S) bookkeeping.

Card 6 (the recursive two-body decay kernel, plan section 8.4).  The
seven-step loop runs per vertex over a queue of unstable daughters:
(1) the parent mass is realized (the caller fixes the root mass through
the four-momentum it passes; a queued resonance daughter had its mass
drawn at its parent's vertex, step 5 below), (2)-(3) the two-body
multiplet classes are enumerated and their open weights evaluated from
the card-3/4 width law at the realized mass (the default Blatt-Weisskopf
barrier), (4) one (class, charge combination) object is sampled with a
single uniform on the ``decay_channel`` substream over the renormalized
open weights, (5) exact on-shell two-body kinematics are generated with
two uniforms on the ``decay_kinematics`` substream (cos(theta) uniform
in [-1, 1], phi uniform in [0, 2 pi); the spin-averaged kernel, round 2
index item 9) plus two uniforms per resonance daughter for the card-3
``draw_mass`` inside the on-shell mass budget window
``[threshold, min(m_up, M - sum(other daughter masses))]``, (6) the
unstable daughters enter the decay queue, (7) the loop stops only when
no unstable particle remains.

Open-weight convention (card 6, pre-registered here).  The sampled
object is one concrete charge combination of one two-body multiplet
class, and its weight is the card-3/4 partial width of the class
distributed over the class's card-5 charge combinations::

    weight(pair) = cw(pair) * Gamma_class(M) / BR_class,
    Gamma_class(M) = sum_c Gamma_c(M)  (the class channels),
    cw = derive_charge_weights(...)   (the card-5 weights),

which reproduces the pre-registered ``BR_c * Gamma_c(M)`` exactly for a
single-combination class (``cw(pair) = BR_c``, so the weight is the
partial width ``Gamma_c(M)``), gives the card-5 Clebsch-Gordan shares
as the within-class combination shares, and makes the class shares the
partial-width ratios of the width law.  The plan section 8.4 wording
"branching and threshold factors" is the branching-weighted,
threshold-suppressed width; the card-3/4 partial width already carries
the branching factor once, and the card-6 statistical test pins the
within-class charge split to the card-5 weights (2/3 : 1/3 for
Delta+ -> p pi0 : n pi+), which fixes this reading.  A channel whose
concrete combination is absent from a non-empty card-5 class map is
isospin-forbidden and is excluded (plan section 8.3 through the
kernel).  Three-body channels are outside the card-6 enumeration (card
7 routes them through the phase-space vertex).

The pre-registered protections (plan section 8.4).  Cycles: a strict
daughter-mass invariant (every resonance daughter's realized mass is
strictly below its parent's realized mass, enforced by the mass-budget
window and asserted at queue time) plus a depth guard of
:data:`MAX_CHAIN_DEPTH` = 64 vertices per chain; either violation
raises :class:`DecayCycleError` naming the chain.  Inaccessible
channels: if the total open weight at the realized mass is not finite
and positive, :class:`NoOpenChannelError` names the parent and mass
(no clamp, no rejection-with-retry; the legacy threshold discipline).
Width normalization: the open-weight renormalization is asserted
finite and > 0 before sampling, and ``total_width(M) > 0`` for every
realized mass; a zero total width is a :class:`NoOpenChannelError`
naming the parent and mass.  Conservation is checked, never repaired:
every vertex conserves ``(B, Q, S)`` exactly and the daughter
four-momenta sum to the parent four-momentum to 1e-9 GeV; a violation
raises :class:`KinematicsError` and stops the run.

Card 7 (``TASK-20260913-r2-selected-three-body-decays``; plan section
22 task 15, the three-body clause of plan section 10.2 line 692)
extends the kernel with the selected three-body vertex.  A sampled
channel with ``is_three_body = True`` routes through
:func:`three_body_phase_space` instead of the two-body vertex: the
two-step invariant-mass phase-space method pre-registered on the card
(the ``(b, c)`` invariant mass flat in ``M_bc**2`` over its kinematic
range ``[mb + mc, M - md]``; the virtual pair decayed as a two-body
system of mass ``M_bc`` with ``d`` as its partner through the card-6
two-body kinematics; the parent at rest, so nothing is boosted beyond
the pair boost), with no matrix element in this round (the alignment
and matrix-element refinements are Round 4, plan sections 10.4/10.5).
The draw budget of a three-body vertex is ``4 + 2`` per resonance
daughter uniforms on the ``decay_kinematics`` substream (the four
phase-space draws replace the two of the two-body direction; no other
stream is touched), consumed as the daughter masses first (the card-3
``draw_mass`` window, channel order) and then the four pre-registered
phase-space draws.  The open weight of a three-body object is the
card-3/4 partial width at the realized mass: the tabulated BR enters
through ``Gamma_c,0 = BR * nominal_width`` and the mass dependence
through the card-3 documented leading-pair phase-space factor, so the
channel selection shares the two-body weight scale and the R20 unit
grid verifies it as the pre-registered ``BR_c * Gamma_c(M) / sum`` open
weight.  A three-body channel is open above its three-body threshold
(the sum of all three daughter masses, the card-1 catalog convention)
and hard-zero below; the concrete charge combination of the object is
the channel's daughter tuple (three-body channels are outside the
card-5 charge completion, whose audit still checks their ``(B, Q, S)``
bookkeeping).  The four-uniform budget fixes the global azimuth of the
event (the ``d`` direction is generated with its polar angle only, its
azimuth at 0): every rotation-invariant content of the phase-space
model, the flat ``M_bc**2`` density and the isotropic internal ``(b,
c)`` direction, is sampled exactly, and the dropped global rotation is
invisible to the invariant distributions the vertices feed.

Card 8 (``TASK-20260913-r2-branching-spectral-normalization-tests``;
plan section 22 task 16, plan sections 8.2 and 8.4) extends this module
with the pure self-check suite over a whole catalog: the per-parent
branching closure report (the card-1 residual budget tightened into a
documented closure with the pre-registered ``1e-9`` over-normalization
and ``0.20`` residual bounds), the per-record spectral-normalization
report (the fixed-node trapezoid integral of the card-3 ``rho`` over the
support plus the support-discipline probes), and the open-weight sweep
(the plan section 8.4 width-normalization protection on the R20 grid
convention).  The checks are read-only over the catalog surfaces of
cards 2-5: they never repair a record, never relax a tolerance, and
report every violation as a string naming the record (a failing check on
a production catalog re-opens the originating card, not this module).
The card-2 re-open of the card-8 check round (the eta/eta-prime
radiative budgets) adjusts the residual-budget clause: the catalog
records may document an additive ``radiative_budget`` field (the
measured radiative/leptonic fraction of the total width, out of the
initial hadronic catalog scope), the closure evaluates the clause per
record as ``0.20 + radiative_budget`` where the field is documented
and keeps the plain 0.20 bound where it is absent, and
:func:`radiative_budgets_from_json` extracts the field from the raw
catalog payload because the card-1 ``HadronRecord`` carries the plan
section 8.1 field set verbatim.

The card-6 ``decay_source`` entry point accepts a minimal
:data:`SourceRecord` forward reference (a frozen record with the fixed
four-momentum, ``(B, Q, S)``, mass, and the catalog record the source
stands for; card 9 replaces/extends it with the pre-registered source
mixture) and returns the vertex list plus the final stable particles
as :class:`na49_mixing.source_event.FinalParticle` records, each with
its feed-down lineage (``parent_id`` is the index of the producing
vertex; the full ancestry is recoverable from the vertex list).

Hermetic: standard library, numpy, and the vendored ``na49_mixing``
modules only; no event generation, no external tables.
"""

from __future__ import annotations

import bisect
import json
import math
from collections import deque
from dataclasses import dataclass
from math import factorial
from pathlib import Path

import numpy as np

from .excitation_kinematics import FourVector
from .hadron_catalog import (
    PHOTON_PDG,
    THREE_BODY_LAWS,
    DecayChannel,
    HadronCatalog,
    HadronRecord,
    ProvenanceTag,
)
from .isospin import reflect_pdg
from .pdg_database import get_particle
from .quantum_numbers import quantum_numbers
from .resonance_widths import (
    broad_daughter_record,
    channel_width,
    decay_momentum,
    draw_broad_daughter_mass,
    draw_mass,
    spectral_function,
    total_width,
)
from .rng_substreams import derive_substreams
from .source_event import FinalParticle

__all__ = [
    "ISOSPIN_DERIVED_FIELD",
    "ISOSPIN_DERIVED_TRANSCRIBER",
    "DERIVED_ACCESSED_ON",
    "WEIGHT_SUM_TOL",
    "DERIVED_MATCH_TOL",
    "MIRROR_MATCH_TOL",
    "TABULATED_CONSISTENCY_TOL",
    "NEUTRAL_KAON_PAIR",
    "MAX_CHAIN_DEPTH",
    "DECAY_CLOSURE_TOL",
    "DerivedSibling",
    "IsospinDerivation",
    "IsospinCompletionReport",
    "PerParentIsospinReport",
    "SourceRecord",
    "DecayDaughter",
    "DecayVertex",
    "ResonanceDecayer",
    "three_body_phase_space",
    "NoOpenChannelError",
    "DecayCycleError",
    "KinematicsError",
    "clebsch_gordan_squared",
    "multiplet_charge_states",
    "derive_charge_weights",
    "isospin_derivation_plan",
    "isospin_closure_report",
    "apply_isospin_derivation",
    "BRANCHING_SUM_TOL",
    "RESIDUAL_BUDGET_BOUND",
    "SPECTRAL_INTEGRAL_TOL",
    "SPECTRAL_QUADRATURE_NODES",
    "SUPPORT_PROBE_EPS",
    "OPEN_WEIGHT_SWEEP_POINTS",
    "ParentBranching",
    "BranchingClosureReport",
    "RecordSpectral",
    "SpectralNormalizationReport",
    "OpenWeightSweepReport",
    "branching_closure_report",
    "radiative_budgets_from_json",
    "spectral_normalization_report",
    "open_weight_sweep",
]

# The card-5 provenance tag of every derived weight (card, Physics rule 1).
ISOSPIN_DERIVED_FIELD = "isospin-derived"
ISOSPIN_DERIVED_TRANSCRIBER = "na49-res r2 card 5"
#: the ``accessed_on`` date stamped on the derived provenance tags.
DERIVED_ACCESSED_ON = "2026-09-13"

# Deterministic tolerances (card, Physics rule).
WEIGHT_SUM_TOL = 1e-6         # class sum vs the multiplet branching fraction
DERIVED_MATCH_TOL = 1e-9      # re-derivation of a derived weight
MIRROR_MATCH_TOL = 1e-9       # anti-particle mirror (rule 4)
TABULATED_CONSISTENCY_TOL = 5e-2   # tabulated share vs the CG share (warning)

# The card-2 representation of the neutral-kaon pair (K0, K0bar): the
# weak eigenstates (K0S, K0L), recorded as the channel-id pair (310, 130)
# with zero net strangeness (the card-2 convention, e.g. phi(1020) and
# f2(1270)).  The Clebsch-Gordan enumeration runs on the flavor states
# (311, -311) with their proper I3; the concrete channel is the weak pair.
NEUTRAL_KAON_PAIR = (310, 130)
_WEAK_NEUTRAL_KAON_FLAVOR = {310: 311, 130: -311}

# Stable states outside a hand-built record set: (isospin, isospin3)
# with the weak-eigenstate neutral kaons carrying the isospin3 of their
# flavor states (K0S -> K0, K0L -> K0bar).
_STABLE_ISOSPIN = {
    2212: (0.5, 1), 2112: (0.5, -1),
    -2212: (0.5, -1), -2112: (0.5, 1),
    211: (1.0, 2), -211: (1.0, -2), 111: (1.0, 0),
    321: (0.5, 1), -321: (0.5, -1),
    311: (0.5, -1), -311: (0.5, 1),
    310: (0.5, -1), 130: (0.5, 1),
}

# Charge-state name families: the record name with the leading "a"
# (anti-particle prefix) and the trailing charge symbol removed.  The
# stable nucleons split into proton/neutron (and pbar/nbar) names and
# are merged explicitly: the framework pairs p <-> n (isospin.py).
_NUCLEON_FAMILY = {
    "proton": "N", "neutron": "N", "pbar": "aN", "nbar": "aN",
}
_CHARGE_SUFFIXES = ("++", "--", "+", "-", "0")


# ----------------------------------------------------------------------
# Clebsch-Gordan squared (exact closed form)
# ----------------------------------------------------------------------
def clebsch_gordan_squared(
    i_parent: float, i3_parent: int,
    i_b: float, i3_b: int,
    i_c: float, i3_c: int,
) -> float:
    """``|<I_b I3_b; I_c I3_c | I_parent I3_parent>|^2``, the standard
    (Condon-Shortley) Clebsch-Gordan coefficient squared.

    The isospin arguments are the catalog's ``isospin`` values; the
    ``I3`` arguments are the catalog's integer two-units
    (``isospin3 = 2 * I3``).  The value is the exact closed form for
    the half-integer isospins of the catalog (exact factorial
    arithmetic, Condon-Shortley phase); it is ``0.0`` when
    ``I3_b + I3_c != I3_parent`` or the coupling is forbidden.

    Pre-registered checks: Delta+ (I3 = +1/2) -> p pi0 : n pi+ =
    2/3 : 1/3; Delta++ (I3 = +3/2) -> p pi+ = 1; rho0 -> pi+ pi- = 1
    for the unordered pair (the pi0 pi0 partner has coefficient 0)."""
    a = int(round(2 * float(i_b)))
    b = int(round(2 * float(i_c)))
    c = int(round(2 * float(i_parent)))
    aa, bb, cc = int(i3_b), int(i3_c), int(i3_parent)
    if aa + bb != cc:
        return 0.0
    if not (-a <= aa <= a) or not (-b <= bb <= b) or not (-c <= cc <= c):
        return 0.0
    if not (abs(a - b) <= c <= a + b) or (a + b + c) % 2:
        return 0.0

    def fac2(n: int) -> int:
        # factorial of the half-integer argument n/2 (n is even here)
        return factorial(n // 2)

    tri = (c + 1) * fac2(a + b - c) * fac2(a - b + c) * fac2(-a + b + c) \
        / fac2(a + b + c + 2)
    six = fac2(c + cc) * fac2(c - cc) * fac2(a - aa) * fac2(a + aa) \
        * fac2(b - bb) * fac2(b + bb)
    total = 0.0
    for k in range(0, a + b + c + 2):
        d1 = k
        d2 = (a + b - c) // 2 - k
        d3 = (a - aa) // 2 - k
        d4 = (b + bb) // 2 - k
        d5 = (c - b + aa) // 2 + k
        d6 = (c - a - bb) // 2 + k
        if min(d1, d2, d3, d4, d5, d6) < 0:
            continue
        total += (-1) ** k / (
            factorial(d1) * factorial(d2) * factorial(d3)
            * factorial(d4) * factorial(d5) * factorial(d6)
        )
    value = math.sqrt(float(tri * six)) * total
    return value * value


# ----------------------------------------------------------------------
# Multiplet bookkeeping
# ----------------------------------------------------------------------
def _charge_family(name: str) -> str:
    """The record-name charge family (the isospin-multiplet label):
    the name with the leading ``a`` anti-particle prefix and the
    trailing charge symbol removed."""
    fam = name
    if fam.startswith("a") and len(fam) > 1:
        fam = fam[1:]
    for suf in ("++", "--", "+", "-", "0"):
        if fam.endswith(suf):
            return fam[: -len(suf)]
    return fam


# The stable nucleons split into proton/neutron (and pbar/nbar) record
# names while forming one isospin multiplet each (isospin.py pairs
# p <-> n); the family merge below restores the multiplet label.
_NUCLEON_FAMILY = {
    "proton": "N", "neutron": "N", "pbar": "aN", "nbar": "aN",
}

# Stable states that may be absent from a hand-built record set:
# (B, S, family name, isospin, isospin3) with the weak-eigenstate
# neutral kaons carrying the flavor-state isospin3.
_STABLE_BOOK = {
    2212: (1, 0, "proton", 0.5, 1),
    2112: (1, 0, "neutron", 0.5, -1),
    -2212: (-1, 0, "pbar", 0.5, -1),
    -2112: (-1, 0, "nbar", 0.5, 1),
    211: (0, 0, "pi", 1.0, 2),
    -211: (0, 0, "pi", 1.0, -2),
    111: (0, 0, "pi", 1.0, 0),
    321: (0, 1, "K", 0.5, 1),
    -321: (0, -1, "K", 0.5, -1),
    311: (0, 1, "K0", 0.5, -1),
    -311: (0, -1, "aK0", 0.5, 1),
    310: (0, 1, "K0", 0.5, -1),    # K0S stands for the K0 flavor state
    130: (0, -1, "aK0", 0.5, 1),   # K0L stands for the K0bar flavor state
}


def _multiplet_key(pdg_id: int, catalog: HadronCatalog) -> tuple:
    """The isospin-multiplet key ``(isospin, B, S, family)`` of a
    catalog member, with the stable-nucleon family merge and the
    weak-eigenstate neutral kaons resolved to their flavor multiplets
    (K0S -> K0, K0L -> K0bar)."""
    pid = int(pdg_id)
    if pid == PHOTON_PDG:
        # the photon of a radiative channel: a singlet of its own, outside
        # every hadron multiplet (2026-10-02)
        return (0.0, 0, 0, "gamma")
    if pid in _WEAK_NEUTRAL_KAON_FLAVOR:
        pid = _WEAK_NEUTRAL_KAON_FLAVOR[pid]
    try:
        record = catalog.get(pid)
    except KeyError:
        B, S, fam, iso, _ = _STABLE_BOOK[pid]
    else:
        B, S = int(record.B), int(record.S)
        fam, iso = _charge_family(record.name), float(record.isospin)
    fam = _NUCLEON_FAMILY.get(fam, fam)
    return (iso, B, S, fam)


def _isospin_of(pdg_id: int, catalog: HadronCatalog) -> float:
    """Isospin of a catalog record or stable state (the photon of a
    radiative channel carries 0: it changes no I3)."""
    pid = int(pdg_id)
    if pid == PHOTON_PDG:
        return 0.0
    try:
        return float(catalog.get(pid).isospin)
    except KeyError:
        return float(_STABLE_BOOK[pid][3])


def _isospin3_of(pdg_id: int, catalog: HadronCatalog) -> int:
    """Effective isospin3 (two-units) of a catalog member.

    The weak-eigenstate neutral kaons carry the isospin3 of their
    flavor states so the K-Kbar charge enumeration is exact; every
    other member keeps its record value.  The photon of a radiative
    channel carries 0 (an electromagnetic decay conserves I3)."""
    pid = int(pdg_id)
    if pid == PHOTON_PDG:
        return 0
    try:
        return int(catalog.get(pid).isospin3)
    except KeyError:
        return int(_STABLE_BOOK[pid][4])


def _effective_flavor_of(pdg_id: int) -> int:
    """The flavor id behind a weak-eigenstate neutral kaon (310 -> 311,
    130 -> -311); every other id is its own flavor state."""
    return _WEAK_NEUTRAL_KAON_FLAVOR.get(int(pdg_id), int(pdg_id))


def _concrete_pair(b: int, c: int) -> tuple[int, int]:
    """The recorded channel ids of a flavor charge combination.

    The (K0, K0bar) flavor pair is recorded as the weak-eigenstate pair
    (310, 130) (the card-2 convention); every other pair maps to
    itself."""
    if {int(b), int(c)} == {311, -311}:
        return NEUTRAL_KAON_PAIR
    return (int(b), int(c))


#: The charge states per catalog token and pdg id (2026-10-20, the
#: generation profile: a loop over every record on each of 23,738 calls, a
#: quarter of the warm generation time); a catalog is immutable and its
#: token is never reused, so an entry is the value of a fresh call.
_CHARGE_STATES: dict[tuple[int, int], tuple[int, ...]] = {}


def multiplet_charge_states(
    pdg_id: int, catalog: HadronCatalog
) -> tuple[int, ...]:
    """The flavor charge states of the isospin multiplet of *pdg_id*.

    Catalog members (records and the stable book) sharing the multiplet
    key (isospin, B, S, name family), sorted; the weak-eigenstate
    neutral kaons resolve to their flavor multiplet and never appear as
    charge-state candidates themselves (they join through
    :func:`_concrete_pair`)."""
    from .resonance_widths import _catalog_token
    try:
        memo_key = (_catalog_token(catalog), int(pdg_id))
    except TypeError:  # a catalog stand-in without weak references
        return _multiplet_charge_states(pdg_id, catalog)
    hit = _CHARGE_STATES.get(memo_key)
    if hit is None:
        hit = _CHARGE_STATES[memo_key] = _multiplet_charge_states(pdg_id, catalog)
    return hit


def _multiplet_charge_states(pdg_id: int, catalog: HadronCatalog) -> tuple[int, ...]:
    key = _multiplet_key(pdg_id, catalog)
    members: set[int] = set()
    for record in catalog.records():
        other = int(record.pdg_id)
        if other in _WEAK_NEUTRAL_KAON_FLAVOR:
            continue
        if _multiplet_key(other, catalog) == key:
            members.add(other)
    for stable, book in _STABLE_BOOK.items():
        if stable in _WEAK_NEUTRAL_KAON_FLAVOR:
            continue
        B, S, fam, iso, _ = book
        try:
            record = catalog.get(stable)
        except KeyError:
            fam_key = _NUCLEON_FAMILY.get(fam, fam)
            member_key = (float(iso), int(B), int(S), fam_key)
        else:
            member_key = _multiplet_key(stable, catalog)
        if member_key == key:
            members.add(stable)
    return tuple(sorted(members))


def class_pair_of(channel: DecayChannel) -> tuple[int, int]:
    """The recorded daughter pair of a channel (the representative
    multiplet argument of its class)."""
    return (int(channel.daughters[0]), int(channel.daughters[1]))


def _class_flavor_pair(
    daughters: tuple[int, ...],
) -> tuple[int, int]:
    """The flavor id pair behind a channel's daughters (the
    weak-eigenstate neutral kaons resolved)."""
    return (_effective_flavor_of(int(daughters[0])),
            _effective_flavor_of(int(daughters[1])))


def _class_key(
    flavor_pair: tuple[int, int], catalog: HadronCatalog
) -> tuple:
    """The isospin class key of a flavor daughter pair: the sorted pair
    of daughter multiplet keys."""
    k1 = _multiplet_key(flavor_pair[0], catalog)
    k2 = _multiplet_key(flavor_pair[1], catalog)
    return tuple(sorted((k1, k2)))


def _class_channels(
    parent: HadronRecord, multiplet: tuple[int, int], catalog: HadronCatalog
) -> tuple[int, list[DecayChannel]]:
    """The parent's two-body channels of the daughter-multiplet class
    and their shared orbital angular momentum."""
    target = _class_key(_class_flavor_pair(multiplet), catalog)
    channels: list[DecayChannel] = []
    for channel in parent.decay_channels:
        if channel.is_three_body or channel.is_radiative or len(channel.daughters) != 2:
            continue
        if _class_key(_class_flavor_pair(channel.daughters), catalog) == target:
            channels.append(channel)
    if not channels:
        raise ValueError(
            f"parent {parent.name} (pdg {parent.pdg_id}) has no two-body "
            f"channel of the multiplet class {multiplet}"
        )
    angular = {int(ch.orbital_angular_momentum) for ch in channels}
    if len(angular) != 1:
        raise ValueError(
            f"parent {parent.name} (pdg {parent.pdg_id}): the multiplet "
            f"class {multiplet} mixes orbital angular momenta "
            f"{sorted(angular)}"
        )
    return next(iter(angular)), channels



def _channel_pair_key(
    daughters: tuple[int, ...],
    multiplet: tuple[int, int],
    catalog: HadronCatalog,
) -> tuple[int, int]:
    """The canonical channel key of a recorded daughter pair relative to
    the class multiplet: ``(daughter-of-multiplet[0], daughter-of-
    multiplet[1])``, sorted within one multiplet paired with itself,
    with the weak-eigenstate neutral-kaon pair in its recorded
    (310, 130) form."""
    k0 = _multiplet_key(multiplet[0], catalog)
    k1 = _multiplet_key(multiplet[1], catalog)
    d0 = _effective_flavor_of(int(daughters[0]))
    d1 = _effective_flavor_of(int(daughters[1]))
    if k0 == k1:
        lo, hi = min(d0, d1), max(d0, d1)
        pair = _concrete_pair(lo, hi)
        return (min(pair), max(pair))
    first = d0 if _multiplet_key(d0, catalog) == k0 else d1
    second = d1 if first == d0 else d0
    return _concrete_pair(first, second)


def _allowed_charge_combinations(    parent: HadronRecord,
    multiplet: tuple[int, int],
    catalog: HadronCatalog,
    angular: int,
) -> dict[tuple[int, int], float]:
    """``{concrete pair: |CG|^2}`` over the I3-conserving, non-forbidden
    charge combinations of the class (the raw, unnormalized shares).

    The concrete pairs follow the ``(multiplet[0], multiplet[1])``
    daughter order; when the two multiplets are the same, the pair is
    canonicalized as ``(b, c)`` with ``b <= c`` and carries the sum of
    both orderings' coefficients.  Excluded (absent from the result):
    I3 non-conserving combinations, combinations forbidden by the
    total-I selection, (B, Q, S)-violating concrete pairs, and identical
    daughters with odd orbital angular momentum (the Bose rule)."""
    i3_parent = int(parent.isospin3)
    states_b = multiplet_charge_states(multiplet[0], catalog)
    states_c = multiplet_charge_states(multiplet[1], catalog)
    same_multiplet = _multiplet_key(multiplet[0], catalog) \
        == _multiplet_key(multiplet[1], catalog)

    combos: dict[tuple[int, int], float] = {}
    if same_multiplet:
        # unordered pairs {b, c} with b <= c; both orderings contribute
        for i, b in enumerate(states_b):
            for c in states_c[i:]:
                i3_b = _isospin3_of(b, catalog)
                i3_c = _isospin3_of(c, catalog)
                if i3_b + i3_c != i3_parent:
                    continue
                coeff = clebsch_gordan_squared(
                    float(parent.isospin), i3_parent,
                    _isospin_of(b, catalog), i3_b,
                    _isospin_of(c, catalog), i3_c)
                if b != c:
                    coeff += clebsch_gordan_squared(
                        float(parent.isospin), i3_parent,
                        _isospin_of(c, catalog), i3_c,
                        _isospin_of(b, catalog), i3_b)
                if coeff <= 0.0:
                    continue
                lo, hi = (b, c) if b <= c else (c, b)
                pair = _concrete_pair(lo, hi)
                if not _conserves_bqs(parent, pair, catalog):
                    continue
                if pair[0] == pair[1] and angular % 2 == 1:
                    continue  # Bose rule
                key = (min(pair), max(pair))
                combos[key] = combos.get(key, 0.0) + coeff
    else:
        for b in states_b:
            i3_b = _isospin3_of(b, catalog)
            for c in states_c:
                i3_c = _isospin3_of(c, catalog)
                if i3_b + i3_c != i3_parent:
                    continue
                coeff = clebsch_gordan_squared(
                    float(parent.isospin), i3_parent,
                    _isospin_of(b, catalog), i3_b,
                    _isospin_of(c, catalog), i3_c)
                if coeff <= 0.0:
                    continue
                pair = _concrete_pair(b, c)
                if not _conserves_bqs(parent, pair, catalog):
                    continue
                combos[pair] = combos.get(pair, 0.0) + coeff
    return combos


def _daughter_bqs(pdg_id: int, catalog: HadronCatalog | None = None) -> tuple[int, int, int]:
    """The (B, Q, S) of a daughter: the catalog record's fields for a
    record, the vendored PDG table for the rest (KeyError or ValueError
    when neither knows it).  The table lists neither the f0(500) nor the
    Lambda(1405) nor the Xi; read through the table alone, their classes
    found no allowed charge combination and the isospin audit flagged
    them (review 2026-10-16: 50 audit errors on the R98 catalog, each a
    class with such a daughter; the decayer then kept the tabulated
    channels unfiltered, which for these records are the Clebsch-Gordan
    ones)."""
    if catalog is not None:
        try:
            record = catalog.get(int(pdg_id))
        except KeyError:
            record = None
        if record is not None:
            return (int(record.B), int(record.Q), int(record.S))
    return quantum_numbers(int(pdg_id)).as_tuple()


def _conserves_bqs(parent: HadronRecord, pair: tuple[int, int], catalog: HadronCatalog | None = None) -> bool:
    """(B, Q, S) conservation of a concrete charge combination."""
    try:
        total = [sum(v) for v in zip(*(_daughter_bqs(pid, catalog) for pid in pair))]
    except (KeyError, ValueError):
        return False
    return tuple(total) == (int(parent.B), int(parent.Q), int(parent.S))


def derive_charge_weights(
    parent: HadronRecord,
    multiplet: tuple[int, int],
    catalog: HadronCatalog,
) -> dict[tuple[int, int], float]:
    """Charge weights for the two-body multiplet channel
    ``parent -> (multiplet[0], multiplet[1])``.

    Returns one weight per allowed concrete charge combination, ordered
    ``(daughter-of-multiplet[0], daughter-of-multiplet[1])`` and sorted
    within one multiplet paired with itself, normalized to the parent's
    branching fraction of that multiplet class.  Fully tabulated
    classes return their tabulated per-charge values as-is (tabulated
    wins); a class with missing allowed combinations is completed from
    the Clebsch-Gordan shares of the class branching fraction (the
    representative's recorded value is the multiplet branching
    fraction; completing the class redistributes it over the allowed
    charges and preserves the multiplet BR exactly).  Forbidden
    combinations (I3 non-conserving, total-I forbidden, Bose-forbidden,
    (B, Q, S)-violating) carry zero weight and are absent from the
    result; a class with no allowed combination at all returns ``{}``
    and is an audit error.  A radiative class (a photon daughter) is
    outside the charge completion and returns ``{}`` (2026-10-02)."""
    if PHOTON_PDG in (int(multiplet[0]), int(multiplet[1])):
        return {}
    angular, channels = _class_channels(parent, multiplet, catalog)
    combos = _allowed_charge_combinations(parent, multiplet, catalog, angular)
    allowed = {pair: coeff for pair, coeff in combos.items() if coeff > 0.0}
    if not allowed:
        return {}
    present: dict[tuple[int, int], float] = {}
    for channel in channels:
        key = _channel_pair_key(channel.daughters, multiplet, catalog)
        present[key] = float(channel.branching_fraction)
    if set(present) >= set(allowed):
        # Tabulated wins: the per-charge entries are used as-is.
        return {pair: present[pair] for pair in sorted(allowed)}
    br_class = math.fsum(present.values())
    norm = math.fsum(allowed.values())
    return {pair: br_class * coeff / norm for pair, coeff in allowed.items()}


# ----------------------------------------------------------------------
# Derivation plan (the catalog edit engine)
# ----------------------------------------------------------------------
def _derived_provenance(parent: HadronRecord, value: float) -> ProvenanceTag:
    """The card-5 provenance tag of a derived weight, citing the
    parent record's provenance document."""
    if parent.provenance is not None:
        document = parent.provenance.document
    else:
        document = ("na49_mixing/resonance_decay.py isospin completion "
                    "(parent record without provenance)")
    return ProvenanceTag(
        document=document,
        section_or_table=(
            f"isospin-derived charge channel of the {parent.name} "
            f"(pdg {parent.pdg_id}) multiplet; Clebsch-Gordan weights "
            f"over I = {parent.isospin}"
        ),
        field=ISOSPIN_DERIVED_FIELD,
        accessed_on=DERIVED_ACCESSED_ON,
        transcribed_value=float(value),
        transcriber=ISOSPIN_DERIVED_TRANSCRIBER,
    )


@dataclass(frozen=True)
class DerivedSibling:
    """One derived charge-combination channel of the plan.

    ``previous_value`` is set only for the representative rewrite of a
    completed class (the recorded multiplet branching fraction that the
    derived per-charge weight replaces)."""

    parent_pdg: int
    daughters: tuple[int, ...]
    branching_fraction: float
    orbital_angular_momentum: int
    provenance: ProvenanceTag
    previous_value: float | None


@dataclass(frozen=True)
class IsospinDerivation:
    """The per-parent isospin completion plan entry (pure)."""

    pdg_id: int
    name: str
    multiplet: tuple[int, int]
    class_branching_fraction: float
    siblings: tuple[DerivedSibling, ...]
    reweighted: tuple[DerivedSibling, ...]
    mass_range_low: float | None


def _sibling_daughters(
    pair: tuple[int, int], multiplet: tuple[int, int], catalog: HadronCatalog
) -> tuple[int, ...]:
    """The recorded daughter order of a derived channel: the
    ``(multiplet[0], multiplet[1])`` order, sorted within one
    multiplet, and the weak-eigenstate pair (310, 130) verbatim."""
    concrete = _concrete_pair(*pair)
    if concrete == NEUTRAL_KAON_PAIR:
        return NEUTRAL_KAON_PAIR
    if _multiplet_key(multiplet[0], catalog) == _multiplet_key(multiplet[1], catalog):
        return (min(concrete), max(concrete))
    return concrete


def isospin_derivation_plan(
    catalog: HadronCatalog,
) -> tuple[IsospinDerivation, ...]:
    """The isospin charge-completion plan for the whole catalog.

    For every parent and every two-body multiplet class with missing
    allowed charge combinations, the plan lists the derived sibling
    channels and the representative rewrites that complete the class
    (the multiplet branching fraction is preserved exactly).  Fully
    tabulated classes produce no plan entry."""
    plan: list[IsospinDerivation] = []
    for record in catalog.records():
        seen: set[tuple] = set()
        for channel in record.decay_channels:
            if channel.is_three_body or channel.is_radiative or len(channel.daughters) != 2:
                continue
            class_key = _class_key(_class_flavor_pair(channel.daughters),
                                   catalog)
            if class_key in seen:
                continue
            seen.add(class_key)
            entry = _plan_for_class(record, class_pair_of(channel), catalog)
            if entry is not None:
                plan.append(entry)
    return tuple(plan)


def _plan_for_class(
    parent: HadronRecord,
    class_pair: tuple[int, int],
    catalog: HadronCatalog,
) -> IsospinDerivation | None:
    """The completion plan of one multiplet class, or None when the
    class is fully tabulated (or has no allowed combination)."""
    angular, channels = _class_channels(parent, class_pair, catalog)
    combos = _allowed_charge_combinations(parent, class_pair, catalog, angular)
    allowed = {pair: coeff for pair, coeff in combos.items() if coeff > 0.0}
    if not allowed:
        return None
    present: dict[tuple[int, int], DecayChannel] = {}
    for channel in channels:
        key = _channel_pair_key(channel.daughters, class_pair, catalog)
        present[key] = channel
    if set(present) >= set(allowed):
        return None  # tabulated per-charge class: nothing to derive
    br_class = math.fsum(float(ch.branching_fraction) for ch in channels)
    norm = math.fsum(allowed.values())
    weights = {pair: br_class * coeff / norm for pair, coeff in allowed.items()}

    siblings: list[DerivedSibling] = []
    reweighted: list[DerivedSibling] = []
    for pair in sorted(allowed):
        channel = present.get(pair)
        if channel is None:
            siblings.append(DerivedSibling(
                parent_pdg=int(parent.pdg_id),
                daughters=_sibling_daughters(pair, class_pair, catalog),
                branching_fraction=float(weights[pair]),
                orbital_angular_momentum=angular,
                provenance=_derived_provenance(parent, weights[pair]),
                previous_value=None,
            ))
        elif float(channel.branching_fraction) != float(weights[pair]):
            reweighted.append(DerivedSibling(
                parent_pdg=int(parent.pdg_id),
                daughters=_sibling_daughters(pair, class_pair, catalog),
                branching_fraction=float(weights[pair]),
                orbital_angular_momentum=angular,
                provenance=_derived_provenance(parent, weights[pair]),
                previous_value=float(channel.branching_fraction),
            ))
    if not siblings and not reweighted:
        return None

    return IsospinDerivation(
        pdg_id=int(parent.pdg_id),
        name=parent.name,
        multiplet=class_pair,
        class_branching_fraction=float(br_class),
        siblings=tuple(siblings),
        reweighted=tuple(reweighted),
        mass_range_low=_class_threshold_low(parent, channels, siblings, catalog),
    )


def _class_threshold_low(
    parent: HadronRecord,
    channels: list[DecayChannel],
    siblings: list[DerivedSibling],
    catalog: HadronCatalog,
) -> float | None:
    """The lowest allowed ``mass_range[0]`` after the completion: the
    maximum channel threshold over the class channels and the derived
    siblings; None when it does not exceed the recorded value."""
    try:
        current = float(parent.mass_range[0])
    except (IndexError, TypeError):
        return None
    highest = -1.0
    for channel in channels:
        highest = max(highest, catalog.threshold(parent.pdg_id, channel))
    for sibling in siblings:
        channel = DecayChannel(
            parent_pdg=int(sibling.parent_pdg),
            daughters=tuple(int(d) for d in sibling.daughters),
            branching_fraction=sibling.branching_fraction,
            orbital_angular_momentum=sibling.orbital_angular_momentum,
            is_three_body=False,
            provenance=sibling.provenance,
        )
        try:
            highest = max(highest, catalog.threshold(parent.pdg_id, channel))
        except (KeyError, ValueError):
            return None
    if highest > current + 1e-12:
        return float(highest)
    return None


def apply_isospin_derivation(catalog: HadronCatalog) -> HadronCatalog:
    """Return the completed catalog: the derivation plan applied.

    Every plan entry of a parent is applied (a parent may carry several
    incomplete multiplet classes, one entry each).  The derived sibling
    channels are appended to their parent's channel list, the
    representative channels of a completed class carry the derived
    per-charge weight, and ``mass_range[0]`` follows the highest channel
    threshold over all completed classes (the card-1 structural rule;
    never a physics value).  Fully tabulated classes are untouched."""
    plan = isospin_derivation_plan(catalog)
    by_parent: dict[int, list[IsospinDerivation]] = {}
    for entry in plan:
        by_parent.setdefault(entry.pdg_id, []).append(entry)
    records: list[HadronRecord] = []
    for record in catalog.records():
        entries = by_parent.get(int(record.pdg_id))
        if not entries:
            records.append(record)
            continue
        new_channels: list[DecayChannel] = []
        for channel in record.decay_channels:
            rewrite = None
            if not channel.is_three_body and len(channel.daughters) == 2:
                # the entry of the channel's own class holds the
                # rewrite (distinct classes have disjoint daughter ids)
                for entry in entries:
                    pair = _channel_pair_key(channel.daughters,
                                             entry.multiplet, catalog)
                    rewrite = next(
                        (r for r in entry.reweighted
                         if tuple(sorted(int(d) for d in r.daughters))
                         == tuple(sorted(pair))),
                        None)
                    if rewrite is not None:
                        break
            if rewrite is not None:
                new_channels.append(DecayChannel(
                    parent_pdg=int(channel.parent_pdg),
                    daughters=tuple(int(d) for d in channel.daughters),
                    branching_fraction=float(rewrite.branching_fraction),
                    orbital_angular_momentum=int(channel.orbital_angular_momentum),
                    is_three_body=False,
                    provenance=rewrite.provenance,
                ))
            else:
                new_channels.append(channel)
        for entry in entries:
            for sibling in entry.siblings:
                new_channels.append(DecayChannel(
                    parent_pdg=int(sibling.parent_pdg),
                    daughters=tuple(int(d) for d in sibling.daughters),
                    branching_fraction=float(sibling.branching_fraction),
                    orbital_angular_momentum=int(sibling.orbital_angular_momentum),
                    is_three_body=False,
                    provenance=sibling.provenance,
                ))
        lows = [entry.mass_range_low for entry in entries
                if entry.mass_range_low is not None]
        low = max(lows) if lows else None
        mass_range = (
            float(low) if low is not None else float(record.mass_range[0]),
            float(record.mass_range[1]),
        )
        records.append(HadronRecord(
            pdg_id=int(record.pdg_id),
            name=record.name,
            B=int(record.B), Q=int(record.Q), S=int(record.S),
            isospin=float(record.isospin),
            isospin3=int(record.isospin3),
            spin=float(record.spin),
            parity=record.parity,
            pole_mass=float(record.pole_mass),
            nominal_width=float(record.nominal_width),
            mass_range=mass_range,
            decay_channels=tuple(new_channels),
            stable_for_decay=bool(record.stable_for_decay),
            provenance=record.provenance,
        ))
    return HadronCatalog(records)


# ----------------------------------------------------------------------
# Closure audit
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class PerParentIsospinReport:
    """Per-parent isospin audit outcome."""

    pdg_id: int
    missing_charge_partners: tuple[int, ...]      # expected by isospin
    weight_sum_errors: tuple[str, ...]
    forbidden_channels: tuple[str, ...]
    mirror_mismatches: tuple[str, ...]


@dataclass(frozen=True)
class IsospinCompletionReport:
    """Catalog-level isospin completion audit; empty ``errors`` means
    the charge closure holds."""

    per_parent: tuple[PerParentIsospinReport, ...]
    errors: tuple[str, ...]                        # empty == closure holds
    warnings: tuple[str, ...]


def _channel_map(
    record: HadronRecord,
) -> dict[tuple[int, int], float]:
    """Two-body channels of a record keyed by the sorted concrete
    daughter pair (three-body channels excluded)."""
    out: dict[tuple[int, int], float] = {}
    for channel in record.decay_channels:
        if channel.is_three_body or channel.is_radiative or len(channel.daughters) != 2:
            continue
        daughters = (int(channel.daughters[0]), int(channel.daughters[1]))
        key = (min(daughters), max(daughters))
        out.setdefault(key, 0.0)
        out[key] += float(channel.branching_fraction)
    return out


def _mirror_daughter(pdg_id: int, catalog: HadronCatalog) -> int:
    """The I3-reflected daughter: the member of the daughter's own
    isospin multiplet with I3 -> -I3 (the ``isospin.reflect_pdg``
    mechanic resolved within the multiplet, so the kaon flavor states
    stay in their strangeness sector)."""
    pid = _effective_flavor_of(int(pdg_id))
    target_i3 = -_isospin3_of(pid, catalog)
    candidates = [
        m for m in multiplet_charge_states(pid, catalog)
        if _isospin3_of(m, catalog) == target_i3
    ]
    if len(candidates) != 1:
        raise KeyError(
            f"no I3 -> -I3 partner for pdg {pid} in its multiplet"
        )
    return candidates[0]


def _mirror_daughters(
    daughters: tuple[int, ...], catalog: HadronCatalog
) -> tuple[int, int]:
    """The I3-reflected daughter pair."""
    reflected = [_mirror_daughter(d, catalog) for d in daughters]
    pair = _concrete_pair(reflected[0], reflected[1])
    return (min(pair), max(pair))


def _mirror_of(record: HadronRecord, catalog: HadronCatalog) -> HadronRecord | None:
    """The reflect partner of *record* (``isospin.reflect_pdg``), or
    None when the record is self-mirrored, unmapped, or the partner is
    absent."""
    try:
        partner_id = reflect_pdg(int(record.pdg_id))
    except KeyError:
        return None
    if partner_id == int(record.pdg_id):
        return None
    try:
        return catalog.get(partner_id)
    except KeyError:
        return None


def _charge_conjugate_of(
    record: HadronRecord, catalog: HadronCatalog
) -> HadronRecord | None:
    """The charge-conjugate partner of *record*: the catalog record
    with (B, S) -> (-B, -S), isospin3 -> -isospin3, the same isospin,
    spin, and parity, and the same pole mass (the resonance identity
    resolves the excited-state quantum-key degeneracy).  The
    weak-eigenstate neutral kaons are self-conjugate; isoscalar records
    are their own partners and need no check."""
    pid = int(record.pdg_id)
    if pid in _WEAK_NEUTRAL_KAON_FLAVOR:
        return record
    if float(record.isospin) == 0.0:
        return None
    matches: list[HadronRecord] = []
    for other in catalog.records():
        if int(other.pdg_id) == pid:
            continue
        if (float(other.isospin) != float(record.isospin)
                or int(other.isospin3) != -int(record.isospin3)
                or int(other.B) != -int(record.B)
                or int(other.S) != -int(record.S)
                or float(other.spin) != float(record.spin)
                or other.parity != record.parity):
            continue
        if abs(float(other.pole_mass) - float(record.pole_mass)) > 1e-6:
            continue
        matches.append(other)
    if len(matches) == 1:
        return matches[0]
    return None



def _conjugate_daughter(pdg_id: int, catalog: HadronCatalog) -> int:
    """The charge-conjugate daughter: the catalog member with
    (B, S) -> (-B, -S), isospin3 -> -isospin3, the same isospin, spin,
    parity, and pole mass.  Isoscalar daughters and the weak-eigenstate
    neutral kaons are their own conjugates."""
    pid = int(pdg_id)
    if pid in _WEAK_NEUTRAL_KAON_FLAVOR:
        return pid
    try:
        record = catalog.get(pid)
    except KeyError:
        B, S = _STABLE_BOOK[pid][0], _STABLE_BOOK[pid][1]
        iso, i3 = float(_STABLE_BOOK[pid][3]), int(_STABLE_BOOK[pid][4])
        spin, parity, mass = {
            2212: (0.5, "+", 0.93827231), -2212: (0.5, "+", 0.93827231),
            2112: (0.5, "+", 0.93956541), -2112: (0.5, "+", 0.93956541),
            211: (0.0, "-", 0.1395699), -211: (0.0, "-", 0.1395699),
            111: (0.0, "-", 0.1349766),
            321: (0.0, "-", 0.493677), -321: (0.0, "-", 0.493677),
        }[pid]
    else:
        iso = float(record.isospin)
        i3 = int(record.isospin3)
        B, S = int(record.B), int(record.S)
        spin, parity = float(record.spin), record.parity
        mass = float(record.pole_mass)
    if B == 0 and S == 0 and i3 == 0:
        # a neutral, non-strange member is its own charge conjugate
        # (pi0, rho0, omega, eta, eta-prime, ...)
        return pid
    matches: list[int] = []
    for other in catalog.records():
        oid = int(other.pdg_id)
        if oid == pid:
            continue
        if (float(other.isospin) != float(iso)
                or int(other.isospin3) != -int(i3)
                or int(other.B) != -int(B)
                or int(other.S) != -int(S)
                or float(other.spin) != float(spin)
                or other.parity != parity):
            continue
        if mass is not None and abs(float(other.pole_mass) - float(mass)) > 1e-6:
            continue
        matches.append(oid)
    if len(matches) != 1:
        raise KeyError(
            f"no unique charge-conjugate daughter for pdg {pid}"
        )
    return matches[0]


def _conjugate_daughters(
    daughters: tuple[int, ...], catalog: HadronCatalog
) -> tuple[int, int]:
    """The charge-conjugated daughter pair (the anti-particle mirror of
    rule 4: each daughter mapped through the charge-conjugation
    correspondence)."""
    conjugate = [_conjugate_daughter(d, catalog) for d in daughters]
    pair = _concrete_pair(conjugate[0], conjugate[1])
    return (min(pair), max(pair))


def _mirror_compare(
    record: HadronRecord, partner: HadronRecord, catalog: HadronCatalog,
    label: str, mode: str = "mirror",
) -> tuple[str, ...]:
    """Compare the two-body channel maps of *record* and *partner*
    under the daughter mapping selected by *mode*: ``mirror`` maps each
    daughter to the I3 -> -I3 member of its own multiplet (the
    ``isospin.reflect_pdg`` mechanic), ``conjugate`` maps it through
    the charge conjugation (the rule-4 anti-parent correspondence).
    Mismatch strings name the channel and the weights."""
    mapper = _mirror_daughter if mode == "mirror" else _conjugate_daughter
    mine = _channel_map(record)
    theirs = _channel_map(partner)
    mirrored: dict[tuple[int, int], float] = {}
    unmappable = False
    for pair, weight in theirs.items():
        try:
            mapped = [mapper(d, catalog) for d in pair]
            concrete = _concrete_pair(mapped[0], mapped[1])
        except KeyError:
            unmappable = True
            continue
        key = (min(concrete), max(concrete))
        mirrored.setdefault(key, 0.0)
        mirrored[key] += weight
    if unmappable:
        reason = ("no I3 -> -I3 daughter mapping" if mode == "mirror"
                  else "no charge-conjugate daughter mapping")
        return (f"{label}: a channel of the mirror partner "
                f"{partner.name} (pdg {partner.pdg_id}) has {reason}",)
    issues: list[str] = []
    for pair in sorted(set(mine) | set(mirrored)):
        w_mine = mine.get(pair)
        w_theirs = mirrored.get(pair)
        if w_mine is None or w_theirs is None:
            issues.append(
                f"channel {pair} present on one side only "
                f"(weight {w_mine} vs mirrored {w_theirs})")
        elif abs(w_mine - w_theirs) > MIRROR_MATCH_TOL:
            issues.append(
                f"channel {pair} weight {w_mine} != mirrored {w_theirs} "
                f"(beyond {MIRROR_MATCH_TOL})")
    return tuple(issues)


def _is_self_mirror(record: HadronRecord, catalog: HadronCatalog) -> bool:
    """True when the record is its own isospin mirror (an I3 = 0 record,
    including the weak-eigenstate neutral kaons)."""
    if int(record.pdg_id) in _WEAK_NEUTRAL_KAON_FLAVOR:
        return True
    try:
        return reflect_pdg(int(record.pdg_id)) == int(record.pdg_id)
    except KeyError:
        return int(record.isospin3) == 0


def _self_mirror_issues(
    record: HadronRecord, catalog: HadronCatalog
) -> tuple[str, ...]:
    """A self-mirrored record must carry each two-body channel together
    with its I3-reflected partner at the same weight."""
    issues: list[str] = []
    mine = _channel_map(record)
    mirrored: dict[tuple[int, int], float] = {}
    for pair, weight in mine.items():
        try:
            key = _mirror_daughters(pair, catalog)
        except KeyError:
            continue
        mirrored.setdefault(key, 0.0)
        mirrored[key] += weight
    for pair in sorted(set(mine) | set(mirrored)):
        w_mine = mine.get(pair)
        w_theirs = mirrored.get(pair)
        if w_mine is None or w_theirs is None:
            issues.append(
                f"self-mirror: channel {pair} is not closed under the "
                f"daughter I3 reflection (weight {w_mine} vs {w_theirs})")
        elif abs(w_mine - w_theirs) > MIRROR_MATCH_TOL:
            issues.append(
                f"self-mirror: channel {pair} weight {w_mine} != mirrored "
                f"{w_theirs} (beyond {MIRROR_MATCH_TOL})")
    return tuple(issues)


def isospin_closure_report(catalog: HadronCatalog) -> IsospinCompletionReport:
    """The charge closure audit over the whole catalog (plan section
    8.3; the card's six conventions).  Errors are structural: missing
    charge partners, weight-sum discrepancies, forbidden channels with
    non-zero branching fraction, and (B, Q, S) violations.  Warnings
    carry the unrepresented channel budgets, the tabulated-share
    findings, the mirror findings on tabulated values, and the
    isospin-census gaps the ``isospin`` reflection map cannot name."""
    per_parent: list[PerParentIsospinReport] = []
    errors: list[str] = []
    warnings: list[str] = []

    # -- record-level charge census (rule 5) ---------------------------
    census_missing: dict[int, list[int]] = {}
    _multiplet_census(catalog, errors, warnings, census_missing)

    for record in catalog.records():
        pid = int(record.pdg_id)
        label = f"{record.name} (pdg {pid})"

        # -- per-class channel audit -----------------------------------
        weight_errors: list[str] = []
        forbidden: list[str] = []
        seen: set[tuple] = set()
        for channel in record.decay_channels:
            if channel.is_three_body or channel.is_radiative or len(channel.daughters) != 2:
                _three_body_checks(record, channel, catalog, errors)
                continue
            class_key = _class_key(_class_flavor_pair(channel.daughters),
                                   catalog)
            if class_key in seen:
                continue
            seen.add(class_key)
            _class_checks(record, class_pair_of(channel), catalog,
                          weight_errors, forbidden, errors, warnings)

        # -- mirror checks (rule 4) ------------------------------------
        mirror_issues: list[str] = []
        partner = _mirror_of(record, catalog)
        if partner is not None:
            mirror_issues.extend(_mirror_compare(
                record, partner, catalog,
                f"{label} vs its reflect partner {partner.name} "
                f"(pdg {partner.pdg_id})"))
        if not mirror_issues and _is_self_mirror(record, catalog):
            mirror_issues.extend(_self_mirror_issues(record, catalog))
        conjugate = _charge_conjugate_of(record, catalog)
        if conjugate is not None and conjugate.pdg_id != record.pdg_id \
                and int(record.pdg_id) < int(conjugate.pdg_id) \
                and not mirror_issues:
            mirror_issues.extend(_mirror_compare(
                record, conjugate, catalog,
                f"{label} vs its charge conjugate {conjugate.name} "
                f"(pdg {conjugate.pdg_id})", mode="conjugate"))

        per_parent.append(PerParentIsospinReport(
            pdg_id=pid,
            missing_charge_partners=tuple(census_missing.get(pid, ())),
            weight_sum_errors=tuple(weight_errors),
            forbidden_channels=tuple(forbidden),
            mirror_mismatches=tuple(mirror_issues),
        ))

        # -- unrepresented channel budget (plan section 8.2) -----------
        if record.decay_channels:
            total = math.fsum(
                float(ch.branching_fraction) for ch in record.decay_channels)
            if 1.0 - total > 1e-12:
                warnings.append(
                    f"record {label}: unrepresented channel budget: "
                    f"branching fractions sum to {total} (residual "
                    f"{1.0 - total})"
                )

    for report in per_parent:
        errors.extend(
            f"record pdg {report.pdg_id}: {issue}"
            for issue in (*report.weight_sum_errors, *report.forbidden_channels)
        )
        if report.missing_charge_partners:
            errors.append(
                f"record pdg {report.pdg_id}: missing isospin charge "
                f"partner record(s) {list(report.missing_charge_partners)}"
            )
        if report.mirror_mismatches:
            warnings.append(
                f"record pdg {report.pdg_id}: mirror mismatch: "
                + "; ".join(report.mirror_mismatches)
            )

    return IsospinCompletionReport(
        per_parent=tuple(per_parent),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


# ----------------------------------------------------------------------
# audit helpers
# ----------------------------------------------------------------------
def _multiplet_census(
    catalog: HadronCatalog,
    errors: list[str],
    warnings: list[str],
    per_parent_missing: dict[int, list[int]],
) -> None:
    """The record-level charge census (rule 5): per isospin multiplet,
    every expected I3 must be realized; a missing charge state is named
    through ``isospin.reflect_pdg`` when the map covers it, otherwise
    the gap is a warning (the anti-N* singletons are the pre-registered
    card-2 scope)."""
    groups: dict[tuple, list[HadronRecord]] = {}
    weak_ids = set()
    for record in catalog.records():
        pid = int(record.pdg_id)
        if pid in _WEAK_NEUTRAL_KAON_FLAVOR:
            weak_ids.add(pid)
            continue
        if float(record.isospin) == 0.0:
            continue
        groups.setdefault(_multiplet_key(pid, catalog), []).append(record)
    if weak_ids:
        # the weak-eigenstate pair is complete as a pair; the check
        # applies only to catalogs that use the representation
        for weak_id in sorted(_WEAK_NEUTRAL_KAON_FLAVOR):
            if weak_id not in weak_ids:
                errors.append(
                    "the weak-eigenstate neutral-kaon representation is "
                    f"incomplete: pdg {weak_id} is missing from the catalog"
                )
    for key in sorted(groups, key=repr):
        members = groups[key]
        iso = key[0]
        if iso <= 0:
            continue
        two_i = int(round(2 * iso))
        expected = set(range(-two_i, two_i + 1, 2))
        realized = {int(r.isospin3) for r in members}
        for missing_i3 in sorted(expected - realized):
            named = None
            for source in members:
                if int(source.isospin3) != -missing_i3:
                    continue
                try:
                    candidate = reflect_pdg(int(source.pdg_id))
                except KeyError:
                    continue
                try:
                    if reflect_pdg(candidate) == int(source.pdg_id):
                        named = candidate
                        break
                except KeyError:
                    continue
            if named is None:
                warnings.append(
                    "the isospin census cannot name the missing charge "
                    f"state I3 = {missing_i3 / 2} of the multiplet "
                    f"{key} (the reflection map does not cover the "
                    "partner; the card-2 record set is the pre-registered "
                    "scope)"
                )
            elif named not in {int(r.pdg_id) for r in catalog.records()}:
                for member in members:
                    per_parent_missing.setdefault(
                        int(member.pdg_id), []).append(named)
                errors.append(
                    f"multiplet {key}: the isospin-allowed charge partner "
                    f"pdg {named} (I3 = {missing_i3 / 2}) has no record"
                )


def _three_body_checks(record: HadronRecord, channel: DecayChannel,
                       catalog: HadronCatalog, errors: list[str]) -> None:
    """Three-body channels are outside the card-5 charge completion
    (card 7); the audit checks their I3 conservation and (B, Q, S)
    bookkeeping only."""
    label = (f"channel {record.name} (pdg {record.pdg_id}) -> "
             f"{tuple(channel.daughters)}")
    try:
        total = tuple(sum(v) for v in zip(*(_daughter_bqs(int(d), catalog) for d in channel.daughters)))
    except (KeyError, ValueError) as exc:
        errors.append(
            f"{label}: daughter quantum numbers unresolvable: {exc}")
        return
    parent_qn = (int(record.B), int(record.Q), int(record.S))
    if total != parent_qn:
        errors.append(
            f"{label}: daughters sum to (B, Q, S) = {total} "
            f"but the parent carries {parent_qn}"
        )
    i3_sum = sum(_isospin3_of(int(d), catalog) for d in channel.daughters)
    if i3_sum != int(record.isospin3):
        errors.append(
            f"{label}: daughter isospin3 values sum to {i3_sum} but the "
            f"parent carries isospin3 {int(record.isospin3)}"
        )


def _class_checks(
    record: HadronRecord,
    class_pair: tuple[int, int],
    catalog: HadronCatalog,
    weight_errors: list[str],
    forbidden: list[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    """One multiplet class: charge completeness, weight sums, forbidden
    channels, and (B, Q, S) conservation."""
    label = f"record {record.name} (pdg {record.pdg_id})"
    try:
        angular, class_channels = _class_channels(record, class_pair, catalog)
    except ValueError as exc:
        errors.append(str(exc))
        return
    combos = _allowed_charge_combinations(record, class_pair, catalog, angular)
    allowed = {pair: coeff for pair, coeff in combos.items() if coeff > 0.0}
    present: dict[tuple[int, int], DecayChannel] = {}
    for channel in class_channels:
        key = _channel_pair_key(channel.daughters, class_pair, catalog)
        present[key] = channel
    br_class = math.fsum(float(ch.branching_fraction) for ch in class_channels)

    if not allowed:
        if br_class > 0.0:
            message = (
                f"{label}: the multiplet class {class_pair} has no "
                "isospin-allowed charge combination (the parent isospin "
                f"{record.isospin} is not in the daughter product) but the "
                f"branching fractions sum to {br_class}"
            )
            forbidden.append(message)
        return
    for pair in sorted(set(allowed) - set(present)):
        errors.append(
            f"{label}: the isospin-allowed charge partner channel {pair} "
            f"of the class {class_pair} is missing (multiplet branching "
            f"fraction {br_class})"
        )

    derived_class = any(
        ch.provenance is not None
        and ch.provenance.field == ISOSPIN_DERIVED_FIELD
        for ch in class_channels
    )
    if derived_class:
        # the derived weights must reproduce the Clebsch-Gordan vector
        norm = math.fsum(allowed.values())
        for pair, channel in sorted(present.items()):
            share = br_class * allowed.get(pair, 0.0) / norm
            if abs(float(channel.branching_fraction) - share) > DERIVED_MATCH_TOL:
                message = (
                    f"{label}: derived channel {tuple(channel.daughters)} "
                    f"weight {float(channel.branching_fraction)} deviates "
                    f"from the Clebsch-Gordan share {share} of the class "
                    f"branching fraction {br_class}"
                )
                weight_errors.append(message)
    else:
        # tabulated per-charge class: the shares are used as-is; a share
        # far from the Clebsch-Gordan share (a uniform split the CG does
        # not produce) is reported as a warning, never repaired
        if br_class > 0.0:
            norm = math.fsum(allowed.values())
            for pair, channel in sorted(present.items()):
                cg_share = allowed.get(pair)
                if cg_share is None:
                    continue  # reported by the forbidden pass below
                share = float(channel.branching_fraction) / br_class
                if abs(share - cg_share / norm) > TABULATED_CONSISTENCY_TOL:
                    warnings.append(
                        f"{label}: the tabulated charge share of "
                        f"{tuple(channel.daughters)} ({share:.6f} of the "
                        f"class) deviates from the Clebsch-Gordan share "
                        f"{cg_share / norm:.6f} beyond "
                        f"{TABULATED_CONSISTENCY_TOL}"
                    )

    # the forbidden-channel clause for individual channels (rules 2, 3)
    for pair, channel in sorted(present.items()):
        if float(channel.branching_fraction) <= 0.0:
            continue
        if pair not in allowed:
            daughters = tuple(int(d) for d in channel.daughters)
            if daughters[0] == daughters[1] and angular % 2 == 1:
                reason = ("identical daughters with odd orbital angular "
                          f"momentum L = {angular} (the Bose rule)")
            else:
                reason = ("zero Clebsch-Gordan weight in the parent's "
                          f"isospin component (I = {record.isospin})")
            message = (
                f"{label}: the channel {daughters} is forbidden ({reason}) "
                f"but carries the branching fraction "
                f"{float(channel.branching_fraction)}"
            )
            forbidden.append(message)

    # (B, Q, S) conservation per channel (rule 6)
    for channel in class_channels:
        try:
            daughter_sum = tuple(sum(v) for v in zip(*(_daughter_bqs(int(d), catalog) for d in channel.daughters)))
        except (KeyError, ValueError) as exc:
            errors.append(
                f"channel {record.name} (pdg {record.pdg_id}) -> "
                f"{tuple(channel.daughters)}: quantum numbers "
                f"unresolvable: {exc}"
            )
            continue
        parent_qn = (int(record.B), int(record.Q), int(record.S))
        if daughter_sum != parent_qn:
            errors.append(
                f"channel {record.name} (pdg {record.pdg_id}) -> "
                f"{tuple(channel.daughters)}: daughters sum to (B, Q, S) = "
                f"{daughter_sum} but the parent carries "
                f"{parent_qn}"
            )


# ======================================================================
# Round 2, card 6: the recursive two-body decay kernel (plan section 8.4)
# ======================================================================
#: The pre-registered depth guard of plan section 8.4: at most 64
#: vertices per decay chain; the 65th raises ``DecayCycleError``.
MAX_CHAIN_DEPTH: int = 64

#: Four-momentum closure and on-shell tolerance, GeV (the card's
#: pre-registered conservation assertion).
DECAY_CLOSURE_TOL: float = 1e-9

#: Tolerance for the on-shell agreement between a source mass and the
#: invariant of its fixed four-momentum, GeV.
SOURCE_MASS_TOL: float = 1e-9


class NoOpenChannelError(RuntimeError):
    """All channels of the realized parent are below threshold, or the
    total open weight at the realized mass is not finite and positive
    (the inaccessibility and width-normalization protections of plan
    section 8.4).  Names the parent and mass.  Never repaired by
    clamping and never retried (the legacy threshold discipline: a
    channel below threshold is ineligible)."""


class DecayCycleError(RuntimeError):
    """The cycle protection of plan section 8.4 fired: the strict
    daughter-mass invariant or the depth guard of
    :data:`MAX_CHAIN_DEPTH` vertices per chain was violated.  Names the
    chain."""


class KinematicsError(RuntimeError):
    """A conservation, on-shell, or kinematic-consistency violation of
    the decay kernel.  Conservation holds by construction; a violation
    is a bug that stops the run (no momentum repair)."""


@dataclass(frozen=True)
class SourceRecord:
    """The minimal card-6 forward reference for the ``decay_source``
    entry point (a card-9 concern, not yet defined): the fixed
    four-momentum ``(E, px, py, pz)`` in GeV, the integer ``(B, Q, S)``
    of the isolated source, the fixed source mass in GeV/c^2, and the
    catalog record the source stands for (the card-9 mixture resolves
    the record and its realized mass before calling the kernel; the
    degenerate-window rule of the card-9 mixture model fixes the mass
    at the grid point).  ``source_id`` is the staged-pipeline identity
    stamped on the returned final particles.  Card 9 replaces/extends
    this record; the package-level ``SourceRecord`` export remains the
    ``na49_mixing.source_event`` one."""

    pdg_id: int
    four_momentum: tuple[float, float, float, float]
    B: int
    Q: int
    S: int
    mass: float
    source_id: int = 0


@dataclass(frozen=True)
class DecayDaughter:
    """One on-shell daughter of a decay vertex.

    ``four_momentum`` is ``(E, px, py, pz)`` in GeV in the lab frame of
    the parent vertex.  ``is_resonance`` marks a daughter that entered
    the decay queue; a stable daughter is final.  ``parent_pdg`` is the
    direct feed-down lineage (the full chain is recoverable from the
    vertex list)."""

    pdg_id: int
    four_momentum: tuple[float, float, float, float]
    is_resonance: bool
    parent_pdg: int


@dataclass(frozen=True)
class DecayVertex:
    """One decay vertex of the card-6 kernel (two-body) or its card-7
    three-body extension.

    A two-body vertex carries the sampled charge combination as the
    daughter pair; a three-body vertex carries the channel's daughter
    tuple (the concrete charge combination of the flagged channel) and
    three daughters.  ``parent_four_momentum`` is the lab-frame
    four-momentum the vertex decayed; the daughter four-momenta sum to
    it within :data:`DECAY_CLOSURE_TOL`."""

    parent_pdg: int
    parent_mass: float
    parent_four_momentum: tuple[float, float, float, float]
    channel: DecayChannel
    charge_combination: tuple[int, ...]
    daughters: tuple[DecayDaughter, ...]


def _boost_lab(
    four_momentum: tuple[float, float, float, float],
    beta: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Boost *four_momentum* by the three-velocity *beta*.

    The same Lorentz-boost convention as the legacy ``_decay_rest_frame``
    of ``na49_mixing.continuum_channels`` (a rest-frame vector boosted
    with the parent's lab-frame velocity becomes a lab-frame vector),
    re-implemented here, not imported.  A zero beta returns the vector
    unchanged (the legacy convention)."""

    e, px, py, pz = (float(x) for x in four_momentum)
    bx, by, bz = (float(x) for x in beta)
    b2 = bx * bx + by * by + bz * bz
    if not b2 > 0.0:
        return (e, px, py, pz)
    if not (math.isfinite(b2) and b2 < 1.0):
        raise KinematicsError(
            f"boost requires |beta| < 1, got beta^2 = {b2!r}"
        )
    gamma = 1.0 / math.sqrt(1.0 - b2)
    bp = bx * px + by * py + bz * pz
    factor = ((gamma - 1.0) * bp / b2) + gamma * e
    return (
        gamma * (e + bp),
        px + factor * bx,
        py + factor * by,
        pz + factor * bz,
    )


def _invariant_mass(four_momentum: tuple[float, float, float, float]) -> float:
    """The invariant mass ``sqrt(E^2 - |p|^2)`` of a four-momentum."""

    e, px, py, pz = (float(x) for x in four_momentum)
    for name, value in (("E", e), ("px", px), ("py", py), ("pz", pz)):
        if not math.isfinite(value):
            raise KinematicsError(
                f"four-momentum component {name} = {value!r} is not finite"
            )
    m2 = e * e - px * px - py * py - pz * pz
    if not m2 > 0.0:
        raise KinematicsError(
            f"four-momentum ({e}, {px}, {py}, {pz}) is not timelike "
            f"(E^2 - p^2 = {m2!r})"
        )
    return math.sqrt(m2)


class ResonanceDecayer:
    """The card-6 recursive two-body decay kernel (plan section 8.4).

    Binds the closed substream labels ``decay_channel`` and
    ``decay_kinematics`` (``na49_mixing.rng_substreams.SUBSTREAM_LABELS``;
    the set is closed and is NOT extended by this card) through
    :func:`derive_substreams(master_seed, replica_index, event_index)`;
    the run cards fix the triple per decay, so one decayer serves one
    decay.  The kernel consumes the card-5-completed catalog: the
    card-5 charge weights of the concrete charge combinations are the
    channel branching fractions the open weights multiply.
    """

    def __init__(
        self,
        catalog: HadronCatalog,
        master_seed: int,
        replica_index: int,
        event_index: int,
    ) -> None:
        subs = derive_substreams(master_seed, replica_index, event_index)
        self._catalog = catalog
        self._stable = catalog.stable_states()
        self._rng_channel = np.random.default_rng(
            subs.stream_for("decay_channel")
        )
        self._rng_kinematics = np.random.default_rng(
            subs.stream_for("decay_kinematics")
        )
        # Pure compute caches (no random draws): pole-mass maps per
        # record and card-5 class maps per (record, class).
        self._mass_maps: dict[int, dict[int, float]] = {}
        self._class_maps: dict[tuple[int, tuple[int, int]], dict] = {}
        # the three-body vertex law of the catalog object (a run attribute;
        # "flat_m2" for a catalog that predates it)
        self._three_body_law = str(getattr(catalog, "three_body_law", "flat_m2"))

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------
    def decay_resonance(
        self,
        parent_pdg: int,
        four_momentum: tuple[float, float, float, float],
    ) -> list[DecayVertex]:
        """Decay one resonance recursively until every remaining
        particle is in ``catalog.stable_states()`` (plan section 8.4
        step 7).

        The parent's realized mass is the invariant mass of
        *four_momentum* (the caller owns the step-1 mass realization:
        the run cards pass the fixed grid mass at rest, and the card-9
        mixture model passes the record mass drawn from the card-3
        spectral function).  Every resonance daughter down the chain
        has its mass realized through the card-3 ``draw_mass`` inside
        the on-shell mass budget window at its parent's vertex.  A
        stable parent has no decay and returns an empty list.  Returns
        the full vertex list, parent first, in queue order."""

        pdg = int(parent_pdg)
        e, px, py, pz = (float(x) for x in four_momentum)
        p4 = (e, px, py, pz)
        mass = _invariant_mass(p4)
        if pdg in self._stable:
            return []
        record = self._catalog.get(pdg)
        if record.stable_for_decay:
            return []
        return self._decay_chain(pdg, p4, mass, (pdg,), 0)

    def decay_source(
        self, source: SourceRecord
    ) -> tuple[list[DecayVertex], list[FinalParticle]]:
        """Decay an isolated fixed-quantum-number source (the R21/R22
        entry point).

        The card-9 pre-registered source mixture (which record stands
        for the source and its realized mass) is a card-9 concern; this
        entry point accepts the resolved :data:`SourceRecord`, checks
        the record's ``(B, Q, S)`` against the source quantum numbers
        and the fixed mass against the four-momentum invariant (both
        checked, never repaired), and decays as
        :meth:`decay_resonance`.  Returns the vertex list and the final
        stable particles as ``na49_mixing.source_event.FinalParticle``
        records, each with its feed-down lineage (``parent_id`` is the
        index of the producing vertex in the returned list; the full
        ancestry is recoverable from the vertex list)."""

        pdg = int(source.pdg_id)
        record = self._catalog.get(pdg)
        source_qn = (int(source.B), int(source.Q), int(source.S))
        record_qn = (int(record.B), int(record.Q), int(record.S))
        if source_qn != record_qn:
            raise KinematicsError(
                f"source quantum numbers {source_qn} disagree with the "
                f"catalog record {record.name} (pdg {pdg}) carrying "
                f"{record_qn}; conservation is checked, never repaired"
            )
        e, px, py, pz = (float(x) for x in source.four_momentum)
        p4 = (e, px, py, pz)
        invariant = _invariant_mass(p4)
        if abs(invariant - float(source.mass)) > SOURCE_MASS_TOL:
            raise KinematicsError(
                f"source mass {float(source.mass)!r} disagrees with the "
                f"invariant mass {invariant!r} of its four-momentum "
                f"(beyond {SOURCE_MASS_TOL} GeV)"
            )
        if pdg in self._stable or record.stable_for_decay:
            return [], []
        vertices = self._decay_chain(pdg, p4, invariant, (pdg,), 0)
        finals: list[FinalParticle] = []
        for index, vertex in enumerate(vertices):
            for daughter in vertex.daughters:
                if daughter.is_resonance:
                    continue
                finals.append(FinalParticle(
                    particle_id=len(finals),
                    pdg_id=int(daughter.pdg_id),
                    p4=FourVector(*daughter.four_momentum),
                    source_id=int(source.source_id),
                    parent_id=index,
                    channel_family="resonance_decay",
                ))
        return vertices, finals

    # ------------------------------------------------------------------
    # The seven-step loop
    # ------------------------------------------------------------------
    def _decay_chain(
        self,
        pdg: int,
        p4: tuple[float, float, float, float],
        mass: float,
        path: tuple[int, ...],
        depth: int,
    ) -> list[DecayVertex]:
        """The queue loop: vertices parent first, queue order (BFS)."""

        vertices: list[DecayVertex] = []
        queue: deque[tuple[int, tuple, float, int, tuple[int, ...]]] = deque()
        queue.append((pdg, p4, mass, depth, path))
        while queue:
            item_pdg, item_p4, item_mass, item_depth, item_path = \
                queue.popleft()
            if item_depth >= MAX_CHAIN_DEPTH:
                raise DecayCycleError(
                    "decay cycle: the depth guard of "
                    f"{MAX_CHAIN_DEPTH} vertices per chain fired along "
                    f"the chain {' -> '.join(str(p) for p in item_path)}"
                )
            try:
                record = self._catalog.get(item_pdg)
            except KeyError as exc:
                raise KinematicsError(
                    f"queued decay parent pdg {item_pdg} is neither a "
                    "catalog record nor a stable state"
                ) from exc
            vertex, queued = self._one_vertex(
                record, item_p4, item_mass, item_path
            )
            vertices.append(vertex)
            for daughter_pdg, daughter_p4, daughter_mass in queued:
                queue.append((
                    int(daughter_pdg),
                    daughter_p4,
                    float(daughter_mass),
                    item_depth + 1,
                    item_path + (int(daughter_pdg),),
                ))
        return vertices

    def _daughter_mass_map(self, record: HadronRecord) -> dict[int, float]:
        """Pole masses of every channel daughter of *record* (the
        card-3 width law's daughter-mass map; the card-1 catalog
        threshold convention)."""

        pdg = int(record.pdg_id)
        cached = self._mass_maps.get(pdg)
        if cached is not None:
            return cached
        masses: dict[int, float] = {}
        for channel in record.decay_channels:
            for daughter in channel.daughters:
                daughter = int(daughter)
                if daughter in masses:
                    continue
                try:
                    daughter_record = self._catalog.get(daughter)
                except KeyError:
                    entry = get_particle(daughter)
                    if entry is None:
                        raise KeyError(
                            f"daughter pdg_id {daughter} is neither a "
                            "catalog record nor in the vendored PDG table"
                        ) from None
                    masses[daughter] = float(entry.mass)
                else:
                    masses[daughter] = float(daughter_record.pole_mass)
        self._mass_maps[pdg] = masses
        return masses

    def _class_map(
        self, record: HadronRecord, channel: DecayChannel
    ) -> dict:
        """The card-5 charge weights of the channel's multiplet class
        (cached per record and class)."""

        multiplet = class_pair_of(channel)
        key = (int(record.pdg_id), multiplet)
        cached = self._class_maps.get(key)
        if cached is None:
            cached = derive_charge_weights(record, multiplet, self._catalog)
            self._class_maps[key] = cached
        return cached

    def _open_objects(
        self, record: HadronRecord, mass: float
    ) -> tuple[list[tuple[DecayChannel, tuple[int, ...], float]], float]:
        """Steps 2-4 of plan section 8.4 at the realized parent mass.

        The open objects are the concrete charge combinations of the
        parent's two-body multiplet classes with the pre-registered open
        weight ``BR_c * Gamma_c(M)`` (plan section 8.4 steps 3-4: the
        branching and threshold factors of the card-3/4 width law at
        the realized mass, the default Blatt-Weisskopf barrier).  The
        card-3/4 partial width ``Gamma_c(M) = BR_c * Gamma_0 * shape``
        already carries the branching factor, so the implemented object
        weight is the partial width of the class distributed over its
        card-5 charge combinations::

            weight(pair) = cw(pair) * Gamma_class(M) / BR_class,
            Gamma_class(M) = sum_c Gamma_c(M) over the class channels,

        which reproduces ``BR_c * Gamma_c(M)`` exactly for a
        single-combination class (``cw(pair) = BR_c``), gives the
        card-5 Clebsch-Gordan shares as the within-class combination
        shares, and makes the class shares the partial-width ratios.
        A channel whose combination is absent from a non-empty card-5
        class map is isospin-forbidden and is excluded (plan section
        8.3 through the kernel).  The class-map weights are emitted
        over the open combinations only: the enumeration keeps a
        combination only while its channel width at the realized mass
        is positive, so a class-map combination that is kinematically
        closed (zero width at the realized mass) is not in the open set
        and carries weight zero, while the remaining open combinations
        keep their relative card-5 Clebsch-Gordan shares (plan section
        8.4 samples one allowed open channel; a closed combination is
        ineligible, not sampled-then-failed).  A class with an empty
        card-5 map (hand-built catalogs outside the card-5 quantum-
        number coverage) falls back to its channels' own partial
        widths.

        Card 7: every three-body channel (``is_three_body = True``
        with three daughters) is one open object of its own, weighted
        by the card-3/4 partial width at the realized mass -- the
        tabulated BR enters through ``Gamma_c,0 = BR * nominal_width``
        and the mass dependence through the documented leading-pair
        phase-space factor -- with the channel's daughter tuple as the
        concrete charge combination (three-body channels are outside
        the card-5 charge completion).  The channel is open above its
        three-body threshold (the sum of all three daughter masses)
        and hard-zero below, so a closed three-body channel is
        ineligible exactly like a closed two-body combination.

        The total open weight is asserted finite and > 0 before
        sampling (the inaccessibility and width-normalization
        protections); the failures raise :class:`NoOpenChannelError`
        naming the parent and mass."""

        masses = self._daughter_mass_map(record)
        gamma_total = total_width(mass, record, masses, catalog=self._catalog)
        classes: dict[tuple, list[DecayChannel]] = {}
        order: list[tuple] = []
        three_body_objects: list[
            tuple[DecayChannel, tuple[int, ...], float]
        ] = []
        radiative_objects: list[
            tuple[DecayChannel, tuple[int, ...], float]
        ] = []
        for channel in record.decay_channels:
            if channel.is_radiative and not channel.is_three_body:
                # a two-body radiative channel (2026-10-02): its own open
                # object at its partial width, outside the card-5 charge
                # completion (an electromagnetic decay does not conserve
                # isospin), with the channel's daughters as the combination
                if len(channel.daughters) != 2 or not float(channel.branching_fraction) > 0.0:
                    continue
                gamma_rad = channel_width(mass, channel, record, masses, catalog=self._catalog)
                if gamma_rad > 0.0:
                    radiative_objects.append((
                        channel, tuple(int(d) for d in channel.daughters), float(gamma_rad)))
                continue
            if channel.is_three_body:
                # Card 7: one open object per three-body channel, at
                # the card-3/4 partial width of the realized mass (the
                # documented leading-pair convention; the full
                # three-daughter threshold hard zero below).
                if len(channel.daughters) != 3:
                    continue
                if not float(channel.branching_fraction) > 0.0:
                    continue
                gamma_3b = channel_width(mass, channel, record, masses, catalog=self._catalog)
                if not gamma_3b > 0.0:
                    continue  # below the three-body threshold: hard zero
                three_body_objects.append((
                    channel,
                    tuple(int(d) for d in channel.daughters),
                    float(gamma_3b),
                ))
                continue
            if len(channel.daughters) != 2:
                continue
            key = _class_key(
                _class_flavor_pair(channel.daughters), self._catalog
            )
            if key not in classes:
                classes[key] = []
                order.append(key)
            classes[key].append(channel)

        objects: list[tuple[DecayChannel, tuple[int, ...], float]] = []
        for key in order:
            channels = classes[key]
            class_map = self._class_map(record, channels[0])
            allowed: list[tuple[DecayChannel, tuple[int, ...]]] = []
            for channel in channels:
                if not float(channel.branching_fraction) > 0.0:
                    continue
                gamma_c = channel_width(mass, channel, record, masses, catalog=self._catalog)
                if not gamma_c > 0.0:
                    continue
                pair = tuple(int(d) for d in channel.daughters)
                if class_map:
                    pair_key = _channel_pair_key(
                        channel.daughters, class_pair_of(channel),
                        self._catalog,
                    )
                    if pair_key not in class_map:
                        continue  # isospin-forbidden: zero weight
                allowed.append((channel, pair))
            if not allowed:
                continue
            gamma_class = math.fsum(
                channel_width(mass, channel, record, masses, catalog=self._catalog)
                for channel, _ in allowed
            )
            if not gamma_class > 0.0:
                continue
            if class_map:
                br_class = math.fsum(
                    float(channel.branching_fraction)
                    for channel, _ in allowed
                )
                if not br_class > 0.0:
                    continue
                # The class-map weights are emitted over the OPEN
                # combinations only (plan section 8.4 samples one
                # allowed open channel): every entry of `allowed` passed
                # the positive-width filter above and the pair-key
                # membership check, so each open combination is bound to
                # its own DecayChannel and carries its card-5 share of
                # the class width.  A class-map combination that is not
                # in `allowed` is kinematically closed at the realized
                # mass (zero channel width): it is ineligible, carries
                # weight zero, and is not emitted (no fallback
                # DecayChannel, no sampled-then-failed closed channel).
                for channel, pair in allowed:
                    cw = class_map[_channel_pair_key(
                        channel.daughters, class_pair_of(channel),
                        self._catalog,
                    )]
                    if not cw > 0.0:
                        continue  # zero tabulated share
                    weight = cw * gamma_class / br_class
                    objects.append((channel, pair, weight))
            else:
                # no card-5 coverage (e.g. hand-built catalogs outside
                # the quantum-number resolution): the class width is
                # distributed over the channels by their branching
                # shares (a single-channel class keeps its full width)
                br_sum = math.fsum(
                    float(channel.branching_fraction)
                    for channel, _ in allowed
                )
                for channel, pair in allowed:
                    share = (
                        float(channel.branching_fraction) / br_sum
                        if br_sum > 0.0 else 1.0 / len(allowed)
                    )
                    objects.append((channel, pair, share * gamma_class))
        # Card 7: the three-body objects join the open set after the
        # two-body classes (any fixed order is deterministic; the
        # sampling is weight-proportional).
        objects.extend(three_body_objects)
        objects.extend(radiative_objects)
        open_total = math.fsum(weight for _, _, weight in objects)
        if not (gamma_total > 0.0) or not (
            math.isfinite(open_total) and open_total > 0.0
        ):
            raise NoOpenChannelError(
                f"parent {record.name} (pdg {record.pdg_id}) at realized "
                f"mass {mass!r} GeV: no open channel (total open "
                f"weight {open_total!r}, total_width {gamma_total!r}; the "
                "width normalization is not finite and positive; no "
                "clamp, no rejection-with-retry)"
            )
        return objects, open_total

    def _sample_object(
        self,
        objects: list[tuple[DecayChannel, tuple[int, ...], float]],
        open_total: float,
    ) -> tuple[DecayChannel, tuple[int, ...]]:
        """Step 4: one uniform on the ``decay_channel`` substream picks
        the (channel, charge combination) object over the renormalized
        open weights."""

        target = float(self._rng_channel.random()) * open_total
        cumulative = 0.0
        for channel, combination, weight in objects:
            cumulative += weight
            if cumulative >= target:
                return channel, combination
        return objects[-1][0], objects[-1][1]

    def _daughter_mass(
        self,
        daughter_pdg: int,
        other_masses: float,
        parent_mass: float,
        parent_path: tuple[int, ...],
        orbital_l: int | None = None,
    ) -> float:
        """The realized mass of one resonance daughter: the card-3
        ``draw_mass`` inside the on-shell mass budget window
        ``[threshold, min(m_up, M - sum(other daughter masses))]``.

        A broad daughter of a two-body vertex (*orbital_l* given, the
        daughter broad under ``BROAD_DAUGHTER_MIN_WIDTH``) takes the
        joint draw of the quasi-two-body decay (FAB-10, 2026-10-02):
        its line shape times ``q**(2L+1) F_L(q)**2`` on the same window,
        the integrand of the convolved channel width.  Both draws use
        two uniforms."""

        daughter_record = self._catalog.get(daughter_pdg)
        lo = float(daughter_record.mass_range[0])
        hi = min(float(daughter_record.mass_range[1]),
                 parent_mass - other_masses)
        window = (lo, hi)
        broad = (
            broad_daughter_record(self._catalog, daughter_pdg)
            if orbital_l is not None else None
        )
        if broad is not None:
            drawn = draw_broad_daughter_mass(
                parent_mass,
                other_masses,
                int(orbital_l),
                broad,
                self._catalog,
                self._rng_kinematics,
            )
        else:
            drawn = draw_mass(
                window,
                daughter_record,
                self._daughter_mass_map(daughter_record),
                self._rng_kinematics,
                catalog=self._catalog,
            )
        if drawn is None:
            raise KinematicsError(
                f"daughter {daughter_record.name} (pdg {daughter_pdg}) of "
                f"the chain {' -> '.join(str(p) for p in parent_path)} has "
                f"an empty on-shell mass budget at parent mass "
                f"{parent_mass!r}: window {window!r} carries no line-shape "
                "mass (no clamp, no re-draw)"
            )
        # The strict daughter-mass invariant of plan section 8.4 (the
        # cycle protection): every resonance daughter's realized mass
        # is strictly below the parent's realized mass.  The mass-budget
        # window enforces it; this assertion catches a violation as the
        # pre-registered bug stop.
        if not drawn < parent_mass:
            raise DecayCycleError(
                "decay cycle: the resonance daughter "
                f"{daughter_record.name} (pdg {daughter_pdg}) realized "
                f"mass {drawn!r} is not strictly below the parent "
                f"realized mass {parent_mass!r} along the chain "
                f"{' -> '.join(str(p) for p in parent_path)}"
            )
        return float(drawn)

    def _bqs_of(self, pdg_id: int) -> tuple[int, int, int]:
        """The ``(B, Q, S)`` of a daughter: the catalog record fields
        for records, the exact PDG quantum numbers for stable states."""

        try:
            record = self._catalog.get(pdg_id)
        except KeyError:
            try:
                qn = quantum_numbers(pdg_id)
            except (KeyError, ValueError) as exc:
                raise KinematicsError(
                    f"daughter pdg {pdg_id}: quantum numbers unresolvable "
                    f"({exc})"
                ) from exc
            return (int(qn.B), int(qn.Q), int(qn.S))
        return (int(record.B), int(record.Q), int(record.S))

    def _one_vertex(
        self,
        record: HadronRecord,
        parent_p4: tuple[float, float, float, float],
        parent_mass: float,
        path: tuple[int, ...],
    ) -> tuple[DecayVertex, list[tuple[int, tuple, float]]]:
        """Steps 4-6 for one queued parent: sample the open object,
        generate the exact on-shell two-body kinematics, queue the
        unstable daughters."""

        objects, open_total = self._open_objects(record, parent_mass)
        channel, combination = self._sample_object(objects, open_total)

        # Card 7: a sampled three-body channel routes through the
        # phase-space vertex (its own draw budget; the two-body
        # direction draws below are not consumed on this path).
        if channel.is_three_body:
            return self._three_body_vertex(
                record, parent_p4, parent_mass, channel, combination, path
            )

        # Step 5: the spin-averaged isotropic direction (2 uniforms on
        # the decay_kinematics substream, the pre-registered first draw
        # of the vertex), then the daughter masses (2 uniforms per
        # resonance daughter, in channel order).
        u_cos = float(self._rng_kinematics.random())
        u_phi = float(self._rng_kinematics.random())
        cos_theta = 2.0 * u_cos - 1.0
        phi = 2.0 * math.pi * u_phi
        sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
        direction = (
            sin_theta * math.cos(phi),
            sin_theta * math.sin(phi),
            cos_theta,
        )

        d0_pdg, d1_pdg = int(combination[0]), int(combination[1])
        d0_is_resonance = d0_pdg not in self._stable
        d1_is_resonance = d1_pdg not in self._stable
        masses = self._daughter_mass_map(record)

        # A broad daughter takes the joint quasi-two-body draw at the
        # channel's orbital angular momentum (FAB-10); a narrow
        # resonance daughter keeps the line-shape draw.
        orbital_l = int(channel.orbital_angular_momentum)
        if d0_is_resonance:
            # the other daughter carries its pole mass in d0's budget
            mass0 = self._daughter_mass(
                d0_pdg, masses[d1_pdg], parent_mass, path, orbital_l
            )
        else:
            mass0 = masses[d0_pdg]
        if d1_is_resonance:
            # d1's budget uses d0's realized mass (the tighter on-shell
            # budget; for a stable d0 the two agree)
            mass1 = self._daughter_mass(
                d1_pdg, mass0, parent_mass, path, orbital_l
            )
        else:
            mass1 = masses[d1_pdg]

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
        energy, px, py, pz = parent_p4
        beta = (px / energy, py / energy, pz / energy)
        daughter_p4_0 = _boost_lab(rest0, beta)
        daughter_p4_1 = _boost_lab(rest1, beta)

        daughters = (
            DecayDaughter(
                pdg_id=d0_pdg,
                four_momentum=daughter_p4_0,
                is_resonance=d0_is_resonance,
                parent_pdg=int(record.pdg_id),
            ),
            DecayDaughter(
                pdg_id=d1_pdg,
                four_momentum=daughter_p4_1,
                is_resonance=d1_is_resonance,
                parent_pdg=int(record.pdg_id),
            ),
        )

        # Conservation, checked never repaired: (B, Q, S) exactly and
        # the daughter four-momenta summing to the parent four-momentum
        # within DECAY_CLOSURE_TOL GeV; every daughter on shell.
        parent_qn = (int(record.B), int(record.Q), int(record.S))
        daughter_qn = (self._bqs_of(d0_pdg), self._bqs_of(d1_pdg))
        total_qn = tuple(
            a + b for a, b in zip(daughter_qn[0], daughter_qn[1])
        )
        if total_qn != parent_qn:
            raise KinematicsError(
                f"vertex {record.name} (pdg {record.pdg_id}) -> "
                f"{combination}: daughters sum to (B, Q, S) = {total_qn} "
                f"but the parent carries {parent_qn}"
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
                f"vertex {record.name} (pdg {record.pdg_id}) -> "
                f"{combination}: the daughter four-momenta miss the "
                f"parent four-momentum by {worst!r} GeV (beyond "
                f"{DECAY_CLOSURE_TOL}); no momentum repair"
            )
        for daughter, mass in ((daughters[0], mass0), (daughters[1], mass1)):
            e, dpx, dpy, dpz = daughter.four_momentum
            m2 = e * e - dpx * dpx - dpy * dpy - dpz * dpz
            deviation = abs(m2 - mass * mass)
            if not deviation <= DECAY_CLOSURE_TOL:
                raise KinematicsError(
                    f"vertex {record.name} (pdg {record.pdg_id}) -> "
                    f"{combination}: daughter pdg {daughter.pdg_id} is off "
                    f"shell (p^2 - m^2 = {deviation!r} GeV^2, beyond "
                    f"{DECAY_CLOSURE_TOL}); no momentum repair"
                )

        vertex = DecayVertex(
            parent_pdg=int(record.pdg_id),
            parent_mass=float(parent_mass),
            parent_four_momentum=parent_p4,
            channel=channel,
            charge_combination=combination,
            daughters=daughters,
        )
        queued = [
            (daughter.pdg_id, daughter.four_momentum, mass)
            for daughter, mass in ((daughters[0], mass0),
                                   (daughters[1], mass1))
            if daughter.is_resonance
        ]
        return vertex, queued

    def _three_body_vertex(
        self,
        record: HadronRecord,
        parent_p4: tuple[float, float, float, float],
        parent_mass: float,
        channel: DecayChannel,
        combination: tuple[int, ...],
        path: tuple[int, ...],
    ) -> tuple[DecayVertex, list[tuple[int, tuple, float]]]:
        """Steps 4-6 for a sampled three-body channel (card 7): realize
        the daughter masses, generate the exact on-shell phase-space
        point in the parent rest frame, boost it to the lab frame with
        the parent's velocity (the card-6 convention), and queue the
        unstable daughters.

        Draw budget on the ``decay_kinematics`` substream: two uniforms
        per resonance daughter (the card-3 ``draw_mass`` window, in
        channel order, the other daughters' current masses as the
        budget), then the four :func:`three_body_phase_space` draws
        (the pre-registered ``4 + 2`` per resonance daughter of the
        card; the four replace the two of the two-body direction, and
        no other stream is touched).  Conservation is checked, never
        repaired: the ``(B, Q, S)`` sum of the three daughters equals
        the parent's, every daughter is on shell at its realized (or
        pole) mass, and the three lab four-momenta sum to the parent
        four-momentum within :data:`DECAY_CLOSURE_TOL`; a violation
        raises :class:`KinematicsError`."""

        masses = self._daughter_mass_map(record)
        daughter_pdgs = tuple(int(d) for d in combination)

        # Daughter masses first (the phase-space kinematics consume
        # them), in channel order: a stable daughter keeps its pole
        # mass, a resonance daughter draws inside its on-shell mass
        # budget window with the other daughters' current masses
        # (realized where already drawn, pole otherwise) subtracted --
        # the card-6 budget convention extended to three slots.  Every
        # slot starts at its pole mass, so a not-yet-drawn resonance
        # daughter contributes its pole mass to the earlier budgets.
        daughter_masses: list[float] = [masses[pdg] for pdg in daughter_pdgs]
        is_resonance = [pdg not in self._stable for pdg in daughter_pdgs]
        for index, pdg in enumerate(daughter_pdgs):
            if not is_resonance[index]:
                continue
            others = math.fsum(
                daughter_masses[other]
                for other in range(3) if other != index
            )
            daughter_masses[index] = self._daughter_mass(
                pdg, others, parent_mass, path
            )

        # The exact on-shell phase-space point in the parent rest frame
        # (four uniforms on the decay_kinematics substream).
        element = None
        if self._three_body_law == "phase_space_dalitz":
            # the measured Dalitz distributions (2026-10-20)
            from .dalitz_matrix_elements import dalitz_element_for
            element = dalitz_element_for(int(record.pdg_id), daughter_pdgs)
        rest_points = three_body_phase_space(
            parent_mass,
            daughter_masses[0], daughter_masses[1], daughter_masses[2],
            self._rng_kinematics,
            law=self._three_body_law,
            matrix_element=element,
        )

        # Boost the three rest-frame momenta with the parent's lab
        # velocity (the card-6 convention; a zero beta returns the
        # vectors unchanged).
        energy, px, py, pz = parent_p4
        beta = (px / energy, py / energy, pz / energy)
        daughters = tuple(
            DecayDaughter(
                pdg_id=daughter_pdgs[index],
                four_momentum=_boost_lab(
                    (float(rest_points[index][0]),
                     float(rest_points[index][1]),
                     float(rest_points[index][2]),
                     float(rest_points[index][3])),
                    beta,
                ),
                is_resonance=is_resonance[index],
                parent_pdg=int(record.pdg_id),
            )
            for index in range(3)
        )

        # Conservation, checked never repaired: (B, Q, S) exactly, the
        # three daughter four-momenta summing to the parent
        # four-momentum within DECAY_CLOSURE_TOL GeV, every daughter on
        # shell at its realized (or pole) mass.
        parent_qn = (int(record.B), int(record.Q), int(record.S))
        total_qn = [0, 0, 0]
        for daughter in daughters:
            daughter_qn = self._bqs_of(daughter.pdg_id)
            for k in range(3):
                total_qn[k] += daughter_qn[k]
        if tuple(total_qn) != parent_qn:
            raise KinematicsError(
                f"three-body vertex {record.name} (pdg {record.pdg_id}) "
                f"-> {daughter_pdgs}: daughters sum to (B, Q, S) = "
                f"{tuple(total_qn)} but the parent carries {parent_qn}"
            )
        closure = (
            abs(sum(d.four_momentum[0] for d in daughters) - energy),
            abs(sum(d.four_momentum[1] for d in daughters) - px),
            abs(sum(d.four_momentum[2] for d in daughters) - py),
            abs(sum(d.four_momentum[3] for d in daughters) - pz),
        )
        worst = max(closure)
        if not worst <= DECAY_CLOSURE_TOL:
            raise KinematicsError(
                f"three-body vertex {record.name} (pdg {record.pdg_id}) "
                f"-> {daughter_pdgs}: the daughter four-momenta miss the "
                f"parent four-momentum by {worst!r} GeV (beyond "
                f"{DECAY_CLOSURE_TOL}); no momentum repair"
            )
        for index, daughter in enumerate(daughters):
            e, dpx, dpy, dpz = daughter.four_momentum
            m2 = e * e - dpx * dpx - dpy * dpy - dpz * dpz
            deviation = abs(m2 - daughter_masses[index] ** 2)
            if not deviation <= DECAY_CLOSURE_TOL:
                raise KinematicsError(
                    f"three-body vertex {record.name} (pdg "
                    f"{record.pdg_id}) -> {daughter_pdgs}: daughter pdg "
                    f"{daughter.pdg_id} is off shell (p^2 - m^2 = "
                    f"{deviation!r} GeV^2, beyond "
                    f"{DECAY_CLOSURE_TOL}); no momentum repair"
                )

        vertex = DecayVertex(
            parent_pdg=int(record.pdg_id),
            parent_mass=float(parent_mass),
            parent_four_momentum=parent_p4,
            channel=channel,
            charge_combination=tuple(daughter_pdgs),
            daughters=daughters,
        )
        queued = [
            (daughter.pdg_id, daughter.four_momentum, daughter_masses[index])
            for index, daughter in enumerate(daughters)
            if daughter.is_resonance
        ]
        return vertex, queued


# ======================================================================
# Round 2, card 7: the selected three-body vertex (plan section 22
# task 15; the three-body clause of plan section 10.2, line 692)
# ======================================================================
#: The node count of the phase-space inverse CDF of the three-body
#: intermediate mass (uniform in phi with M_bc = lo + (hi - lo) sin^2 phi).
_THREE_BODY_GRID_POINTS: int = 257

#: Pure compute cache of the phase-space CDF tables keyed on the exact
#: ``(M, mb, mc, md)`` floats (a zero-width parent such as the eta repeats
#: its key); cleared when it reaches :data:`_THREE_BODY_CACHE_MAX` entries.
_THREE_BODY_TABLE_CACHE: dict[tuple[float, float, float, float], tuple] = {}
_THREE_BODY_CACHE_MAX: int = 4096


def _phase_space_pair_mass(M: float, mb: float, mc: float, md: float, u: float) -> float:
    """The (b, c) invariant mass at the uniform *u* under flat three-body
    phase space: the inverse CDF of ``q*(M_bc; mb, mc) q(M; M_bc, md)`` in
    ``dM_bc`` (the Dalitz strip at fixed ``M_bc^2`` has the length ``4 q*
    q M / M_bc`` and ``dM_bc^2 = 2 M_bc dM_bc``).  The substitution
    ``M_bc = lo + (hi - lo) sin^2(phi)`` makes the integrand vanish like
    ``sin^2`` and ``cos^2`` at the two ends; the CDF is the trapezoid on
    :data:`_THREE_BODY_GRID_POINTS` nodes uniform in ``phi``, linear in
    ``phi`` between the nodes (scalar arithmetic only)."""
    lo, hi = mb + mc, M - md
    n = _THREE_BODY_GRID_POINTS
    key = (M, mb, mc, md)
    table = _THREE_BODY_TABLE_CACHE.get(key)
    if table is None:
        half_pi = 0.5 * math.pi
        phis = [half_pi * j / (n - 1) for j in range(n)]
        g = []
        for phi in phis:
            sin_p, cos_p = math.sin(phi), math.cos(phi)
            m = lo + (hi - lo) * sin_p * sin_p
            g.append(decay_momentum(m, mb, mc) * decay_momentum(M, m, md) * sin_p * cos_p)
        cumulative = [0.0]
        for j in range(1, n):
            cumulative.append(cumulative[-1] + 0.5 * (g[j] + g[j - 1]) * (phis[j] - phis[j - 1]))
        if len(_THREE_BODY_TABLE_CACHE) >= _THREE_BODY_CACHE_MAX:
            _THREE_BODY_TABLE_CACHE.clear()
        table = _THREE_BODY_TABLE_CACHE[key] = (phis, cumulative)
    phis, cumulative = table
    target = float(u) * cumulative[-1]
    # the first node at or above the target (at least 1, at most n - 1)
    k = min(max(bisect.bisect_left(cumulative, target), 1), n - 1)
    c_lo, c_hi = cumulative[k - 1], cumulative[k]
    frac = 0.0 if not c_hi > c_lo else (target - c_lo) / (c_hi - c_lo)
    phi = phis[k - 1] + frac * (phis[k] - phis[k - 1])
    sin_p = math.sin(phi)
    return min(max(lo + (hi - lo) * sin_p * sin_p, lo), hi)


def three_body_phase_space(
    M: float,
    mb: float, mc: float, md: float,
    rng: np.random.Generator,
    *,
    law: str = "flat_m2",
    matrix_element=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One three-body phase-space point in the parent rest frame:
    three four-momenta (each shape ``(4,)``, ``(E, px, py, pz)`` in GeV)
    summing to ``(M, 0, 0, 0)``, each on shell at ``(mb, mc, md)``.

    Two-step method (pre-registered on the card; the phase-space-flat
    model of plan section 10.2, no matrix element in this round):

    (i) sample the ``(b, c)`` invariant mass ``M_bc`` from the flat
    density over its kinematic range ``[mb + mc, M - md]`` -- uniform in
    ``M_bc**2``, one uniform;

    (ii) decay the virtual ``(b, c)`` pair as a two-body system of mass
    ``M_bc`` with ``d`` as its partner, through the card-6 two-body
    kinematics: the exact on-shell ``(b, c)`` kinematics in the pair
    rest frame behind two uniforms (``cos(theta_bc)`` uniform in
    ``[-1, 1]``, ``phi_bc`` uniform in ``[0, 2 pi)``), the pair
    orientation against ``d`` behind one uniform (the ``d`` direction's
    polar angle, ``cos(theta_d) = 2 u - 1``), and the pair rest-frame
    vectors boosted with the pair velocity (the card-6 boost
    convention; the parent is at rest, so nothing is boosted beyond the
    pair boost).

    Consumes exactly 4 uniforms, in the order ``(M_bc, cos(theta_bc),
    phi_bc, cos(theta_d))``.  The pre-registered four-uniform budget
    fixes the global azimuth of the event (the ``d`` direction's
    azimuth is 0): the rotation-invariant content of the phase-space
    model -- the flat ``M_bc**2`` density and the isotropic internal
    ``(b, c)`` direction -- is sampled exactly, and the dropped global
    rotation is invisible to the invariant distributions the vertex
    feeds.

    The daughters are returned in the caller's ``(b, c, d)`` order; for
    a catalog channel the leading pair ``(b, c)`` is
    ``(daughters[0], daughters[1])`` (the card-3 leading-pair
    convention) and ``d`` is ``daughters[2]``.

    A parent mass at or below the three-body threshold ``mb + mc + md``
    (empty ``M_bc`` range) raises ``ValueError``: the channel is closed
    there, the hard zero is the caller's concern (the card-3 width law
    and the open-object enumeration), and this function is never
    called on a closed channel.  A non-finite or non-positive parent
    mass, or a non-finite or negative daughter mass, raises
    ``ValueError`` (a massless daughter is allowed: the photon of a
    radiative channel).

    *law* (2026-10-06; the code and physics review's M-3) selects the
    density of step (i).  ``"flat_m2"`` is the draw above.  It is not flat
    phase space: the review measured chi2/dof = 33,735/108 against a flat
    Dalitz plot for omega -> 3 pi at the pole on 400,000 decays, and the
    pi0 mean momentum 9% low.  ``"phase_space"`` draws ``M_bc`` from the
    flat three-body phase-space density ``q*(M_bc; mb, mc) q(M; M_bc, md)``
    in ``dM_bc`` (:func:`_phase_space_pair_mass`, the inverse CDF of the
    same first uniform), so the Dalitz plot is flat.  Both laws consume the
    same four uniforms in the same order; the run selects the law through
    ``HadronCatalog.three_body_law``.

    *matrix_element* (2026-10-20, :mod:`na49_mixing.dalitz_matrix_elements`)
    is a measured Dalitz distribution ``|A|^2`` of the decay: the (b, c)
    mass follows its marginal, the helicity cosine of b in the (b, c) frame
    about the pair's flight direction its conditional, and the third uniform
    the azimuth about that direction, the same four uniforms in the same
    order.  None (the default) is the law's flat vertex bit for bit.  The
    ``phase_space_dalitz`` law is ``phase_space`` here; the decayer passes
    the element of the decay."""

    if law not in THREE_BODY_LAWS:
        raise ValueError(f"three_body_phase_space: law must be one of {THREE_BODY_LAWS!r}, got {law!r}")
    M = float(M)
    mb = float(mb)
    mc = float(mc)
    md = float(md)
    if not (math.isfinite(M) and M > 0.0):
        raise ValueError(
            f"three_body_phase_space: M = {M!r} must be a finite positive mass"
        )
    for name, value in (("mb", mb), ("mc", mc), ("md", md)):
        # a daughter may be massless (the photon of a radiative channel,
        # 2026-10-02); the two-body steps are exact at zero mass
        if not (math.isfinite(value) and value >= 0.0):
            raise ValueError(
                f"three_body_phase_space: {name} = {value!r} must be a "
                "finite non-negative mass"
            )
    lo = (mb + mc) ** 2
    hi = (M - md) ** 2
    if not hi > lo:
        raise ValueError(
            f"three_body_phase_space: the parent mass M = {M!r} does not "
            f"exceed the three-body threshold mb + mc + md = "
            f"{mb + mc + md!r}; the (b, c) invariant-mass range "
            f"[{math.sqrt(lo)!r}, {math.sqrt(max(hi, 0.0))!r}] is empty "
            "(the channel is closed below the three-body threshold)"
        )

    # (i) the (b, c) invariant mass: flat in M_bc^2 over the kinematic
    # range (the pre-registered phase-space-flat density); one uniform.
    u_mbc = float(rng.random())
    if matrix_element is not None:
        if law == "flat_m2":
            raise ValueError("three_body_phase_space: a matrix element needs the phase_space law")
        from .dalitz_matrix_elements import dalitz_pair_mass
        m_bc = dalitz_pair_mass(M, mb, mc, md, matrix_element, u_mbc)
    elif law in ("phase_space", "phase_space_dalitz"):
        m_bc = _phase_space_pair_mass(M, mb, mc, md, u_mbc)
    else:
        m_bc_sq = lo + u_mbc * (hi - lo)
        m_bc = math.sqrt(m_bc_sq)

    # (ii) the exact on-shell (b, c) kinematics in the pair rest frame
    # (the card-6 two-body conventions; two uniforms).
    u_cos = float(rng.random())
    u_phi = float(rng.random())
    cos_theta = 2.0 * u_cos - 1.0
    phi = 2.0 * math.pi * u_phi
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    direction = (
        sin_theta * math.cos(phi),
        sin_theta * math.sin(phi),
        cos_theta,
    )
    q_star = decay_momentum(m_bc, mb, mc)
    energy_b = (m_bc * m_bc + mb * mb - mc * mc) / (2.0 * m_bc)
    energy_c = (m_bc * m_bc - mb * mb + mc * mc) / (2.0 * m_bc)
    rest_b = (
        energy_b,
        q_star * direction[0],
        q_star * direction[1],
        q_star * direction[2],
    )
    rest_c = (
        energy_c,
        -q_star * direction[0],
        -q_star * direction[1],
        -q_star * direction[2],
    )

    # The pair orientation against d (one uniform): the d direction's
    # polar angle, its azimuth fixed at 0 by the pre-registered
    # four-uniform budget.
    u_d = float(rng.random())
    cos_d = 2.0 * u_d - 1.0
    sin_d = math.sqrt(max(0.0, 1.0 - cos_d * cos_d))
    d_direction = (sin_d, 0.0, cos_d)
    if matrix_element is not None:
        # the helicity frame about the pair's flight direction n = -d
        from .dalitz_matrix_elements import dalitz_helicity_cosine
        cos_star = dalitz_helicity_cosine(M, mb, mc, md, m_bc, matrix_element, u_cos)
        sin_star = math.sqrt(max(0.0, 1.0 - cos_star * cos_star))
        n_axis = (-d_direction[0], 0.0, -d_direction[2])
        e1 = (n_axis[2], 0.0, -n_axis[0])
        direction = (
            sin_star * math.cos(phi) * e1[0] + cos_star * n_axis[0],
            sin_star * math.sin(phi),
            sin_star * math.cos(phi) * e1[2] + cos_star * n_axis[2],
        )
        rest_b = (energy_b, q_star * direction[0], q_star * direction[1], q_star * direction[2])
        rest_c = (energy_c, -q_star * direction[0], -q_star * direction[1], -q_star * direction[2])
    q_pair = decay_momentum(M, m_bc, md)
    energy_pair = (M * M + m_bc * m_bc - md * md) / (2.0 * M)
    energy_d = (M * M - m_bc * m_bc + md * md) / (2.0 * M)
    # The pair rest frame moves opposite the d direction with the pair
    # momentum q_pair; boosting the pair rest-frame vectors with that
    # velocity yields the parent rest-frame momenta.  The boosted pair
    # sums exactly to (energy_pair, -q_pair * d_direction), which with
    # the d four-momentum closes the parent four-momentum.
    beta_pair = (
        -q_pair * d_direction[0] / energy_pair,
        -q_pair * d_direction[1] / energy_pair,
        -q_pair * d_direction[2] / energy_pair,
    )
    p_b = np.asarray(_boost_lab(rest_b, beta_pair), dtype=np.float64)
    p_c = np.asarray(_boost_lab(rest_c, beta_pair), dtype=np.float64)
    p_d = np.asarray(
        (energy_d,
         q_pair * d_direction[0],
         q_pair * d_direction[1],
         q_pair * d_direction[2]),
        dtype=np.float64,
    )
    return p_b, p_c, p_d


# ======================================================================
# Round 2, card 8: the branching-closure and spectral-normalization
# self-check suite (plan section 22 task 16; plan sections 8.2, 8.4)
# ======================================================================
#: Per-parent hadronic branching-sum tolerance above 1 (card 8, Physics
#: rule; the same bound as the card-1 schema rule
#: ``BRANCHING_SUM_TOL`` in ``hadron_catalog``).  A sum beyond ``1 +
#: BRANCHING_SUM_TOL`` is an over-normalized catalog: an error, never
#: repaired.
BRANCHING_SUM_TOL: float = 1e-9

#: Pre-registered residual budget (card 8, Physics rule).  The residual
#: ``1 - hadronic_br_sum`` is the non-hadronic budget (radiative and
#: leptonic modes are out of the initial catalog scope, plan section
#: 8.1's hadronic channel list); a residual above this bound is an
#: unexplained gap, i.e. a suspected missing major hadronic channel,
#: and is a ``violations`` entry.  The clause is evaluated per parent
#: as ``RESIDUAL_BUDGET_BOUND + radiative_budget`` where the catalog
#: record documents a ``radiative_budget`` (the card-2 re-open of the
#: card-8 check round: the documented radiative/leptonic fraction of
#: the total width, extracted from the raw catalog payload by
#: :func:`radiative_budgets_from_json`); a record without the field
#: keeps the plain bound.  The pre-registered 0.20 constant itself is
#: unchanged.
RESIDUAL_BUDGET_BOUND: float = 0.20

#: Pre-registered spectral-normalization tolerance (card 8, Physics
#: rule): the fixed-node trapezoid integral of the card-3 ``rho`` over
#: the support ``[threshold, m_up]`` equals 1 within this bound.
SPECTRAL_INTEGRAL_TOL: float = 1e-6

#: Pre-registered node count of the spectral-normalization grid (card
#: 8, Physics rule).
SPECTRAL_QUADRATURE_NODES: int = 10_001

#: Pre-registered support-discipline probe offset (card 8, Physics
#: rule): ``rho`` is exactly ``0.0`` at ``threshold - SUPPORT_PROBE_EPS``
#: and at ``m_up + SUPPORT_PROBE_EPS``.
SUPPORT_PROBE_EPS: float = 1e-12

#: Pre-registered open-weight sweep resolution (card 8, Physics rule;
#: the R20 grid convention of plan section 8: 100 fixed-mass points
#: spanning threshold to upper support).
OPEN_WEIGHT_SWEEP_POINTS: int = 100


def _check_daughter_masses(
    catalog: HadronCatalog, record: HadronRecord
) -> dict[int, float]:
    """Pole masses of every channel daughter of *record* (the card-3
    width law's daughter-mass map, the card-1 catalog threshold
    convention: the catalog record's ``pole_mass`` for records, the
    vendored PDG table mass for stable states)."""

    masses: dict[int, float] = {}
    for channel in record.decay_channels:
        for daughter in channel.daughters:
            daughter = int(daughter)
            if daughter in masses:
                continue
            try:
                masses[daughter] = float(catalog.get(daughter).pole_mass)
            except KeyError:
                entry = get_particle(daughter)
                if entry is None:
                    raise KeyError(
                        f"daughter pdg_id {daughter} of record "
                        f"{record.name} (pdg {record.pdg_id}) is neither a "
                        "catalog record nor in the vendored PDG table"
                    ) from None
                masses[daughter] = float(entry.mass)
    return masses


def _check_trapezoid(xs: np.ndarray, ys: np.ndarray) -> float:
    """Trapezoid integral of *ys* over the abscissae *xs*."""
    if xs.size < 2:
        return 0.0
    return float((0.5 * (ys[1:] + ys[:-1]) * (xs[1:] - xs[:-1])).sum())


@dataclass(frozen=True)
class ParentBranching:
    """One parent's branching closure (card 8, pure).

    ``hadronic_br_sum`` is the parent's two-body + three-body branching
    sum, ``residual_budget = 1 - hadronic_br_sum`` is the non-hadronic
    budget (radiative and leptonic modes out of the initial catalog
    scope), ``radiative_budget`` is the documented radiative/leptonic
    fraction of the total width (0.0 when the catalog record documents
    none; the card-2 re-open of the card-8 check round), and
    ``within_bound`` carries the pre-registered sanity bound: the sum
    at most ``1 + BRANCHING_SUM_TOL`` and the residual at most
    ``RESIDUAL_BUDGET_BOUND + radiative_budget``."""

    pdg_id: int
    hadronic_br_sum: float
    residual_budget: float          # 1 - hadronic_br_sum
    within_bound: bool
    radiative_budget: float = 0.0   # documented field; 0.0 == absent


@dataclass(frozen=True)
class BranchingClosureReport:
    """Per-parent branching closure over a whole catalog (card 8).

    ``violations`` empty means the closure holds: no parent is
    over-normalized and no residual exceeds the pre-registered budget
    adjusted by the parent's documented radiative budget where one is
    documented.  A violation string names the parent record."""

    per_parent: tuple[ParentBranching, ...]
    max_residual: float
    violations: tuple[str, ...]     # empty == closure holds


def branching_closure_report(
    catalog: HadronCatalog,
    radiative_budgets: dict[int, float] | None = None,
) -> BranchingClosureReport:
    """Per-parent branching closure over the full catalog (card 8).

    Per parent: the two-body + three-body branching sum, the residual
    budget ``1 - sum`` (the non-hadronic channel budget: radiative,
    leptonic, and other modes outside the initial hadronic catalog
    scope, plan section 8.1's hadronic channel list), and the
    pre-registered sanity bound: ``hadronic_br_sum <= 1 + 1e-9`` (a
    violation is an error: an over-normalized catalog) and
    ``residual_budget <= RESIDUAL_BUDGET_BOUND + radiative_budget``
    (a violation is a ``violations`` entry: a gap unexplained by the
    documented radiative/leptonic modes, i.e. a suspected missing
    major hadronic channel).

    *radiative_budgets* maps pdg_id to the documented
    radiative/leptonic budget of the catalog record (the additive
    ``radiative_budget`` record field of the card-2 re-open of the
    card-8 check round; extract it from the raw catalog payload with
    :func:`radiative_budgets_from_json`, because
    :meth:`HadronCatalog.from_json` loads the plan section 8.1 field
    set only).  A record with a documented budget is evaluated against
    the adjusted bound ``0.20 + radiative_budget`` (the residual is
    explained when it stays inside); a record without one keeps the
    plain 0.20 bound (the residual is unexplained, as before).  The
    pre-registered 0.20 constant is unchanged.  The check never
    repairs a record.

    Records without decay channels (the stable states) carry no
    branching budget and stay outside the closure."""

    budgets: dict[int, float] = {}
    if radiative_budgets is not None:
        budgets = {int(k): float(v) for k, v in radiative_budgets.items()}
    per_parent: list[ParentBranching] = []
    violations: list[str] = []
    max_residual = 0.0
    for record in catalog.records():
        if not record.decay_channels:
            continue
        label = f"{record.name} (pdg {record.pdg_id})"
        br_sum = math.fsum(
            float(channel.branching_fraction)
            for channel in record.decay_channels
        )
        residual = 1.0 - br_sum
        budget = budgets.get(int(record.pdg_id))
        budget_value = 0.0 if budget is None else float(budget)
        residual_bound = RESIDUAL_BUDGET_BOUND + budget_value
        over_normalized = not br_sum <= 1.0 + BRANCHING_SUM_TOL
        gap = not residual <= residual_bound
        if over_normalized:
            violations.append(
                f"record {label}: hadronic branching sum {br_sum!r} "
                f"exceeds 1 + {BRANCHING_SUM_TOL!r} (over-normalized "
                "catalog; checked, never repaired)"
            )
        if gap:
            if budget is None:
                violations.append(
                    f"record {label}: residual budget {residual!r} exceeds "
                    f"the pre-registered bound {RESIDUAL_BUDGET_BOUND!r} "
                    "(unexplained gap: a suspected missing major hadronic "
                    "channel; radiative and leptonic modes are out of the "
                    "initial catalog scope)"
                )
            else:
                violations.append(
                    f"record {label}: residual budget {residual!r} exceeds "
                    f"the adjusted bound {residual_bound!r} = "
                    f"{RESIDUAL_BUDGET_BOUND!r} + documented radiative "
                    f"budget {budget_value!r} (unexplained beyond the "
                    "documented radiative/leptonic modes: a suspected "
                    "missing major hadronic channel)"
                )
        max_residual = max(max_residual, residual)
        per_parent.append(ParentBranching(
            pdg_id=int(record.pdg_id),
            hadronic_br_sum=float(br_sum),
            residual_budget=float(residual),
            within_bound=not (over_normalized or gap),
            radiative_budget=float(budget_value),
        ))
    return BranchingClosureReport(
        per_parent=tuple(per_parent),
        max_residual=float(max_residual),
        violations=tuple(violations),
    )


def radiative_budgets_from_json(path) -> dict[int, float]:
    """pdg_id -> documented radiative_budget, from the raw catalog JSON.

    The additive ``radiative_budget`` record field (the card-2 re-open
    of the card-8 check round) documents the radiative/leptonic
    fraction of the record's total width: the modes outside the initial
    hadronic catalog scope (plan section 8.1's hadronic channel list).
    The card-1 :class:`HadronRecord` carries the plan section 8.1 field
    set verbatim and :meth:`HadronCatalog.from_json` does not load the
    additive field, so the closure check reads it from the raw payload
    at this one place.  Records without the field are absent from the
    mapping (the plain pre-registered bound applies).  A value outside
    [0, 1] raises ``ValueError``: documentation is checked, never
    repaired."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    budgets: dict[int, float] = {}
    for raw in payload["records"]:
        if "radiative_budget" not in raw:
            continue
        pid = int(raw["pdg_id"])
        value = float(raw["radiative_budget"])
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"record pdg {pid}: radiative_budget {value!r} is "
                "outside [0, 1] (the documented radiative/leptonic "
                "fraction of the total width); checked, never repaired"
            )
        budgets[pid] = value
    return budgets


@dataclass(frozen=True)
class RecordSpectral:
    """One record's spectral normalization (card 8, pure).

    ``integral`` is the fixed-node trapezoid integral of the card-3
    ``rho`` over the support ``[threshold, m_up]``; the two probe
    fields carry the support-discipline values just outside the
    support and must be exactly ``0.0``.  A record whose width law is
    not evaluable under the card-4 barrier (a negative orbital angular
    momentum; the card-4 barrier supports ``L >= 0``) carries ``nan``
    fields and ``within_tolerance = False``; the report's
    ``violations`` entry names the reason."""

    pdg_id: int
    integral: float
    max_below_threshold_value: float   # must be 0.0
    max_above_uppercut_value: float     # must be 0.0
    within_tolerance: bool


@dataclass(frozen=True)
class SpectralNormalizationReport:
    """Per-record spectral normalization over a whole catalog (card 8).

    ``violations`` empty means every record's line shape integrates to
    1 within :data:`SPECTRAL_INTEGRAL_TOL` on the pre-registered grid
    and the support discipline holds.  A violation string names the
    record."""

    per_record: tuple[RecordSpectral, ...]
    max_abs_error: float
    violations: tuple[str, ...]


def spectral_normalization_report(
    catalog: HadronCatalog,
    quadrature_nodes: int = 10_001,
) -> SpectralNormalizationReport:
    """Per-record check of the card-3 spectral function (card 8).

    For every record with decay channels: the trapezoidal integral of
    the card-3 ``rho(M)`` (the relativistic Breit-Wigner with the
    channel-summed mass-dependent width, the card-4 Blatt-Weisskopf
    barrier default) over ``[mass_range[0], mass_range[1]] =
    [threshold, m_up]`` on a fixed grid of *quadrature_nodes* nodes
    (pre-registered 10,001) equals 1 within
    :data:`SPECTRAL_INTEGRAL_TOL`, and ``rho`` is exactly ``0.0`` at
    ``threshold - SUPPORT_PROBE_EPS`` and at ``m_up +
    SUPPORT_PROBE_EPS`` (the support discipline: zero just below the
    threshold and just above the upper mass cut).  The check never
    repairs a record.

    A record whose channel widths are not evaluable under the card-4
    barrier (a negative orbital angular momentum; the card-4 barrier
    supports ``L >= 0``) carries ``nan``
    fields, ``within_tolerance = False``, and a violation entry naming
    the record.  Records without decay channels carry no line shape
    and stay outside the report."""

    per_record: list[RecordSpectral] = []
    violations: list[str] = []
    max_abs_error = 0.0
    for record in catalog.records():
        if not record.decay_channels:
            continue
        pid = int(record.pdg_id)
        label = f"{record.name} (pdg {pid})"
        masses = _check_daughter_masses(catalog, record)
        lo = float(record.mass_range[0])
        hi = float(record.mass_range[1])
        integral = float("nan")
        below = float("nan")
        above = float("nan")
        try:
            xs = np.linspace(lo, hi, int(quadrature_nodes))
            ys = np.array([
                spectral_function(float(x), record, masses, catalog=catalog)
                for x in xs
            ])
            integral = _check_trapezoid(xs, ys)
            below = spectral_function(lo - SUPPORT_PROBE_EPS, record, masses, catalog=catalog)
            above = spectral_function(hi + SUPPORT_PROBE_EPS, record, masses, catalog=catalog)
        except ValueError as exc:
            violations.append(
                f"record {label}: the card-3 spectral function is not "
                f"evaluable over the support under the card-4 barrier "
                f"({exc}); the normalization check cannot run for this "
                "record"
            )
            per_record.append(RecordSpectral(
                pdg_id=pid,
                integral=integral,
                max_below_threshold_value=below,
                max_above_uppercut_value=above,
                within_tolerance=False,
            ))
            continue
        error = abs(integral - 1.0)
        support_ok = (below == 0.0) and (above == 0.0)
        within = error <= SPECTRAL_INTEGRAL_TOL and support_ok
        if error > SPECTRAL_INTEGRAL_TOL:
            violations.append(
                f"record {label}: the {int(quadrature_nodes)}-node "
                f"trapezoid integral of rho over the support is "
                f"{integral!r}, off 1 by {error!r} (beyond the "
                f"pre-registered {SPECTRAL_INTEGRAL_TOL!r})"
            )
        if below != 0.0 or above != 0.0:
            violations.append(
                f"record {label}: the support discipline is violated "
                f"(rho at threshold - {SUPPORT_PROBE_EPS!r} is {below!r}, "
                f"at m_up + {SUPPORT_PROBE_EPS!r} is {above!r}; both must "
                "be exactly 0.0)"
            )
        max_abs_error = max(max_abs_error, error)
        per_record.append(RecordSpectral(
            pdg_id=pid,
            integral=float(integral),
            max_below_threshold_value=float(below),
            max_above_uppercut_value=float(above),
            within_tolerance=bool(within),
        ))
    return SpectralNormalizationReport(
        per_record=tuple(per_record),
        max_abs_error=float(max_abs_error),
        violations=tuple(violations),
    )


@dataclass(frozen=True)
class OpenWeightSweepReport:
    """The width-normalization sweep over a whole catalog (card 8).

    ``records_checked`` counts the records with decay channels the
    sweep covered, ``points_checked`` the grid mass points actually
    evaluated, and ``violations`` empty means the guard holds: finite,
    non-negative channel widths everywhere and a finite, strictly
    positive total open weight at every grid mass strictly above the
    record threshold (hard zero at and below the threshold is the
    expected behavior, not a violation).  A violation string names the
    record and the mass (and channel where applicable)."""

    records_checked: int
    points_checked: int
    violations: tuple[str, ...]     # empty == guard holds


def open_weight_sweep(
    catalog: HadronCatalog,
    points_per_record: int = 100,
) -> OpenWeightSweepReport:
    """The width-normalization guard of plan section 8.4 at the sweep
    level (card 8).

    For every record with decay channels, on a *points_per_record*
    grid over its ``mass_range`` (pre-registered 100 points, the R20
    grid convention of plan section 8): every channel width is finite
    and non-negative, and the total open weight is finite and strictly
    positive for every M strictly above the record threshold.  The
    hard zero at and below the threshold is the expected behavior and
    is never a violation.  The check never repairs a record.

    A record whose channel widths are not evaluable under the card-4
    barrier (a negative orbital angular momentum; the card-4 barrier
    supports ``L >= 0``, the catalog maximum is ``L = 4``) yields one
    violation naming the record and the offending channel, and its
    grid is not evaluated (the guard cannot be verified for it).
    Records without decay channels carry no width law and stay
    outside the sweep."""

    violations: list[str] = []
    records_checked = 0
    points_checked = 0
    for record in catalog.records():
        if not record.decay_channels:
            continue
        records_checked += 1
        pid = int(record.pdg_id)
        label = f"{record.name} (pdg {pid})"
        blocked = [
            channel for channel in record.decay_channels
            if int(channel.orbital_angular_momentum) < 0
        ]
        if blocked:
            for channel in blocked:
                violations.append(
                    f"record {label}: channel "
                    f"{tuple(int(d) for d in channel.daughters)} carries "
                    f"orbital angular momentum L = "
                    f"{int(channel.orbital_angular_momentum)}; the card-4 "
                    "Blatt-Weisskopf barrier supports L >= 0 (negative L "
                    "is not a partial wave) and "
                    "the width law is not evaluable (the open-weight "
                    "guard cannot be verified for this record)"
                )
            continue
        masses = _check_daughter_masses(catalog, record)
        lo = float(record.mass_range[0])
        hi = float(record.mass_range[1])
        for node in np.linspace(lo, hi, int(points_per_record)):
            mass = float(node)
            points_checked += 1
            widths = [
                channel_width(mass, channel, record, masses, catalog=catalog)
                for channel in record.decay_channels
            ]
            for channel, width in zip(record.decay_channels, widths):
                if not (math.isfinite(width) and width >= 0.0):
                    violations.append(
                        f"record {label}: channel "
                        f"{tuple(int(d) for d in channel.daughters)} has "
                        f"width {width!r} at mass {mass!r} GeV (not "
                        "finite and non-negative)"
                    )
            total = math.fsum(widths)
            if mass > lo and not (math.isfinite(total) and total > 0.0):
                violations.append(
                    f"record {label}: total open weight {total!r} at "
                    f"mass {mass!r} GeV is not finite and strictly "
                    "positive (the width-normalization guard of plan "
                    "section 8.4; hard zero at and below the threshold "
                    "is the expected behavior)"
                )
    return OpenWeightSweepReport(
        records_checked=int(records_checked),
        points_checked=int(points_checked),
        violations=tuple(violations),
    )