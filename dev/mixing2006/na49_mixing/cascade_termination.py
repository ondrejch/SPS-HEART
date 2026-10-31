"""Cascade termination rules and the hard maximum depth guard.

Round 4, card 9 (``TASK-20260920-r4-termination-depth-tests``; plan
section 22 task 26, line 1748; plan section 10.6, lines 750-757).  The
module carries the cascade termination rules of plan section 10.6: the
catalog handoff predicate, the direct-channel termination predicate, the
hard maximum depth guard enforced on card 4's depth counter, and the
cascade driver that applies card 4's ``draw_decay`` steps until
termination.  Card 4's ``cascade_decay.py`` is consumed read-only (the
depth counter, the channel enumeration, and the per-vertex draw budget
are as defined there); the Round 2 ``ResonanceDecayer`` is consumed
byte-unchanged as the handoff target.  Cards 11-14 exercise the
termination and the guard statistically; this card lands alone, hermetic.

Termination rules (plan section 10.6, lines 752-755)
----------------------------------------------------

The plan terminates a cascade when

- "the parent matches an established resonance and is sampled from the
  catalog", or
- "no continuum channel is open and a valid direct hadronic channel
  exists".

The pre-registered instantiation (the card's objective) fixes both rules
as predicates on a :class:`~na49_mixing.cascade_decay.CascadeState`:

- **The catalog handoff** (:func:`is_catalog_handoff`): the parent state
  stands for a committed catalog record (its ``family_id`` is
  ``"catalog:<pdg_id>"``, card 4's :data:`~na49_mixing.cascade_decay.
  CATALOG_FAMILY_PREFIX` form) whose ``(B, Q, S)`` matches the state's
  and whose mass lies within the record's support
  ``[m_min, m_up]`` (the record's ``mass_range`` pair, the plan section
  8.2 line-shape support).  The handoff is a routing decision: the
  state's remainder is the Round 2 kernel's, not a re-implementation of
  the Round 2 decay.  A continuum state (a ``family_id`` of
  ``config/v2_singlet/continuum_resonances.json``) is never a handoff:
  the cascade is the continuum state's decay mechanism, so the predicate
  is false there even when a catalog record carries the same ``(B, Q,
  S)``.  An unknown record id is no match (false), and the QN match and
  the support window are checked, never repaired.

- **The direct-channel termination**
  (:func:`has_valid_direct_channel`): the state's card-4 enumeration
  opened no continuum channel and at least one valid direct hadronic
  channel (an ``established_hadron`` or ``established_resonance``
  channel of card 4's enumeration over the established daughters).  The
  driver decays such a state through one ``draw_decay`` step over the
  direct pool and the cascade line terminates: the plan section 10.2
  line 692 three-body clause and every further decay of the direct
  channel's daughters are the Round 2 kernel's territory, reached
  through the handoff routing by the caller (the run cards).

The hard maximum depth guard (plan section 10.6, line 757)
----------------------------------------------------------

:data:`MAX_CASCADE_DEPTH = 12` is card 4's constant placeholder,
re-exported here unchanged.  The depth counter is the
:class:`~na49_mixing.cascade_decay.CascadeState` ``depth`` field
(incremented by card 4's ``draw_decay`` on every daughter, 0 on a
root).  The guard is a hard enforcement point:
:func:`run_cascade` checks the counter on every line state before any
termination rule and raises :class:`CascadeDepthGuardError` when the
state's depth exceeds the maximum, naming the depth.  The error stops
the run (checked, never repaired, no silent truncation); the plan
line 757 clause "tune physics so the guard is never reached in accepted
events" is verified on the pinned trace of the card test: across the
accepted source-mass range (2-8 GeV, the baryonic and mesonic families)
the guard reaches zero activations and the deepest line state stays far
below the limit (the card-4 daughter-mass rule steps the cascade down
by the shed hadron's mass plus the family ``nominal_width`` per vertex
and caps the budget at :data:`~na49_mixing.spectral_density.
SUPPORT_M_MAX`, so the accepted cascade is shallow).

The cascade driver
------------------

:func:`run_cascade` follows one cascade line from the root state
through card 4's ``draw_decay`` steps until termination.  At every line
state, in order:

1. the hard depth guard (raise at ``depth > max_depth``);
2. the catalog handoff (:func:`is_catalog_handoff` true): the state's
   remainder is delegated to the Round 2
   :class:`~na49_mixing.resonance_decay.ResonanceDecayer` through
   :func:`handoff_decay` and the line terminates;
3. the continuation: a continuum channel is open in the state's card-4
   enumeration; one ``draw_decay`` step over the continuum pool
   realizes the vertex and the continuum daughter continues the line
   (the card-4 enumeration produces at most one continuum daughter per
   channel, so the continuation is unique);
4. the direct-channel termination
   (:func:`has_valid_direct_channel` true): one ``draw_decay`` step
   over the direct pool (the non-continuum channels) realizes the
   terminal hadronic vertex and the line terminates;
5. no channel at all: the state carries no open quasi-two-body channel
   and no direct hadronic channel, so neither pre-registered
   termination rule applies; the driver raises ``ValueError`` (checked,
   never repaired: a state outside the accepted model).

The driver returns the line as a list of
:class:`~na49_mixing.cascade_decay.CascadeState`, root first, ending at
the terminal state.  The terminal state identifies the termination mode
exactly:

- ``catalog_handoff`` (:data:`TERMINATION_CATALOG_HANDOFF`): the
  terminal state satisfies :func:`is_catalog_handoff` (a catalog record
  within support; the kernel took over the remainder).  A catalog state
  never reaches the enumeration in the driver, because the handoff
  check precedes it.
- ``direct_channel`` (:data:`TERMINATION_DIRECT_CHANNEL`): the terminal
  state is a continuum state whose enumeration opened no continuum
  channel; the driver performed the direct hadronic decay.  A continuum
  state never satisfies the handoff predicate, so the two modes
  separate on the terminal state alone.

The decay products of the continuum and direct vertices (the siblings
of the continuing daughter, the direct channel's daughter pair) are
terminal established states; they are the caller's event assembly, not
line states, and carry no entry in the returned trace.  The kernel
handoff's products (``DecayVertex`` records) are the Round 2 kernel's
own output of :func:`handoff_decay`.

The handoff delegation
----------------------

:func:`handoff_decay` constructs the Round 2
:class:`~na49_mixing.resonance_decay.ResonanceDecayer` on the committed
catalog (byte-unchanged; the construction consumes no random numbers,
the card-4 kernel-shim pattern) and binds the cascade's closed
``decay_channel`` / ``decay_kinematics`` substream generators into the
kernel's draw sites, so the delegated decay consumes the cascade's own
substreams and the kernel's decay logic, tables, and conservation checks
run byte-unchanged.  On a handoff state that is the first consumer of
fresh substream generators (a pinned root handoff), the delegated decay
is bit-identical to the standalone ``ResonanceDecayer`` at the same
(master seed, replica, event) triple -- the card's bit-identity
acceptance, exercised by the card test.  Deeper handoffs continue the
same substreams (the caller fixes one triple per cascade line; the run
cards own the per-call seed bookkeeping).

Draw budget
-----------

The driver adds no random number of its own: every draw it consumes is
card 4's frozen per-vertex budget (1 channel + 2 kinematics uniforms)
on the two closed substream generators passed in, plus the Round 2
kernel's own draws inside the handoff delegations (the kernel's
documented budget at its vertices).  The guard, both predicates, and
the mode recording consume zero uniforms.

Hermetic: standard library, numpy, the committed catalog and family
configuration, and the vendored ``na49_mixing`` modules only; no event
generation, no external tables, no Monte Carlo run (the acceptance is
hermetic: pinned traces, predicates, and the pinned synthetic
depth-13 fixture; the runs R40-R44 exercise the surface statistically).
"""

from __future__ import annotations

from .flavor_weights import vertex_ledger_record

from .cascade_decay import (
    CATALOG_FAMILY_PREFIX,
    KIND_CONTINUUM,
    KIND_ESTABLISHED_HADRON,
    KIND_ESTABLISHED_RESONANCE,
    MAX_CASCADE_DEPTH,
    CascadeChannel,
    CascadeState,
    draw_decay_with_pool,
    enumerate_channels,
)
from .resonance_decay import DecayVertex, ResonanceDecayer

__all__ = [
    "MAX_CASCADE_DEPTH",
    "CascadeDepthGuardError",
    "TERMINATION_CATALOG_HANDOFF",
    "TERMINATION_DIRECT_CHANNEL",
    "is_catalog_handoff",
    "has_valid_direct_channel",
    "handoff_decay",
    "run_cascade",
]


#: The hard maximum cascade depth: card 4's constant placeholder
#: (:data:`~na49_mixing.cascade_decay.MAX_CASCADE_DEPTH`), re-exported
#: unchanged.  The enforcement is this card's: :func:`run_cascade`
#: raises :class:`CascadeDepthGuardError` when a line state's depth
#: exceeds the constant (plan section 10.6, line 757).
MAX_CASCADE_DEPTH: int = MAX_CASCADE_DEPTH

#: The termination modes recorded per cascade (the card's objective:
#: "the termination mode (catalog handoff / direct channel) is recorded
#: per cascade").  The terminal state of the driver's trace identifies
#: the mode exactly; see the module docstring.
TERMINATION_CATALOG_HANDOFF: str = "catalog_handoff"
TERMINATION_DIRECT_CHANNEL: str = "direct_channel"

_DIRECT_KINDS = (KIND_ESTABLISHED_HADRON, KIND_ESTABLISHED_RESONANCE)


class CascadeDepthGuardError(RuntimeError):
    """Raised when a cascade exceeds depth
    :data:`MAX_CASCADE_DEPTH` (plan section 10.6, line 757).

    The error names the depth of the offending line state and the
    configured maximum.  The guard is a hard enforcement point, not a
    repair: the run stops (no truncation, no clamp, no re-draw).
    """


# ----------------------------------------------------------------------
# The termination predicates (plan section 10.6, lines 752-755)
# ----------------------------------------------------------------------
def is_catalog_handoff(state: CascadeState, catalog) -> bool:
    """True iff *state* hands off to the Round 2 ``ResonanceDecayer``.

    The pre-registered handoff rule (plan section 10.6 line 754, "the
    parent matches an established resonance and is sampled from the
    catalog"): the state's ``family_id`` stands for a committed catalog
    record (card 4's ``"catalog:<pdg_id>"`` form), the record's
    ``(B, Q, S)`` matches the state's, and the state's mass lies within
    the record's support ``[m_min, m_up]`` (the record's ``mass_range``
    pair; the bounds are inclusive).  The predicate is pure: it
    consumes zero random numbers and mutates nothing.

    False when the state is a continuum state (a cascade line never
    routes a continuum state to the kernel: the cascade is the
    continuum state's decay mechanism), when the record id is unknown,
    on a ``(B, Q, S)`` mismatch between the state and its record, and
    on a mass outside the record support.  All conditions are checked,
    never repaired.
    """

    family_id = str(state.family_id)
    if not family_id.startswith(CATALOG_FAMILY_PREFIX):
        return False
    try:
        record = catalog.get(int(family_id[len(CATALOG_FAMILY_PREFIX):]))
    except KeyError:
        return False  # no committed record: no handoff
    state_qn = (int(state.qn[0]), int(state.qn[1]), int(state.qn[2]))
    if (int(record.B), int(record.Q), int(record.S)) != state_qn:
        return False  # the QN mismatch is checked, never repaired
    mass = float(state.mass)
    return bool(
        float(record.mass_range[0]) <= mass <= float(record.mass_range[1])
    )


def has_valid_direct_channel(
    state: CascadeState, channels: list[CascadeChannel]
) -> bool:
    """True iff the direct-channel termination rule fires on *state*.

    The pre-registered direct-termination rule (plan section 10.6 line
    755, "no continuum channel is open and a valid direct hadronic
    channel exists"): *channels* is card 4's enumeration for *state*
    (each channel's ``parent_id`` must equal the state's ``family_id``;
    a mismatch raises ``ValueError``, checked never repaired).  The
    predicate is true exactly when no channel of kind
    ``continuum`` is open and at least one channel of kind
    ``established_hadron`` or ``established_resonance`` is open.  The
    predicate is pure: it consumes zero random numbers and mutates
    nothing.  The driver checks the catalog handoff before this
    predicate, so a catalog state (whose enumeration carries only
    established kinds) is routed to the kernel and never terminated
    through this rule.
    """

    state_family = str(state.family_id)
    for channel in channels:
        if str(channel.parent_id) != state_family:
            raise ValueError(
                f"has_valid_direct_channel: the channel list carries "
                f"parent_id {channel.parent_id!r} but the state's "
                f"family_id is {state_family!r} (card 4's enumeration "
                "is the resolution source; checked, never repaired)"
            )
    continuum_open = any(
        channel.kind == KIND_CONTINUUM for channel in channels
    )
    if continuum_open:
        return False
    return any(channel.kind in _DIRECT_KINDS for channel in channels)


# ----------------------------------------------------------------------
# The handoff delegation (the Round 2 kernel, byte-unchanged)
# ----------------------------------------------------------------------
def handoff_decay(
    state: CascadeState,
    catalog,
    rng_channel,
    rng_kinematics,
) -> list[DecayVertex]:
    """The catalog handoff delegation: the Round 2 kernel decays the
    remainder of *state*.

    The delegation is a routing decision, not a re-implementation of
    the Round 2 decay: the state must satisfy
    :func:`is_catalog_handoff` (a ``ValueError`` naming the state
    otherwise, checked never repaired), and the byte-unchanged
    :class:`~na49_mixing.resonance_decay.ResonanceDecayer` performs the
    decay through its own :meth:`~na49_mixing.resonance_decay.
    ResonanceDecayer.decay_resonance` at the state's record id and
    four-momentum, returning the full vertex list (parent first, queue
    order; a stable handoff state returns an empty list: the state is
    final).

    The kernel instance is constructed on the committed catalog with a
    pure construction that consumes no random numbers (the card-4
    kernel-shim pattern), and the cascade's closed
    ``decay_channel`` / ``decay_kinematics`` substream generators are
    bound into the kernel's draw sites, so the delegated decay consumes
    the cascade's own substreams and the kernel's decay logic, tables,
    and conservation checks run byte-unchanged.  On a handoff state
    that is the first consumer of fresh substream generators, the
    delegated decay is bit-identical to the standalone
    ``ResonanceDecayer`` at the same (master seed, replica, event)
    triple (the card's bit-identity acceptance).
    """

    if not is_catalog_handoff(state, catalog):
        raise ValueError(
            f"handoff_decay: the state {state.family_id!r} at mass "
            f"{float(state.mass)!r} GeV with (B, Q, S) = "
            f"{tuple(int(v) for v in state.qn)} is not a catalog "
            "handoff state (the committed catalog record within its "
            "support is the routing condition; checked, never repaired)"
        )
    pdg = int(str(state.family_id)[len(CATALOG_FAMILY_PREFIX):])
    decayer = ResonanceDecayer(catalog, 0, 0, 0)  # pure: no random draws
    # The binding: the cascade's own closed substream generators become
    # the kernel's draw sites.  The kernel module is byte-unchanged; the
    # binding replaces only the constructor-derived generator objects,
    # so the delegated decay continues the cascade's substreams (and is
    # bit-identical to the standalone kernel when the generators are
    # fresh).
    decayer._rng_channel = rng_channel
    decayer._rng_kinematics = rng_kinematics
    energy, px, py, pz = (float(x) for x in state.momentum)
    momentum = (energy, px, py, pz)
    return decayer.decay_resonance(pdg, momentum)


# ----------------------------------------------------------------------
# The cascade driver (plan section 10.6)
# ----------------------------------------------------------------------
def run_cascade(
    state: CascadeState,
    catalog,
    fams,
    rng_channel,
    rng_kinematics,
    max_depth: int = MAX_CASCADE_DEPTH,
    flavor=None,
    ledger=None,
) -> list[CascadeState]:
    """The cascade driver: card 4's ``draw_decay`` steps until
    termination.

    *flavor* (Round 6, task 37) reaches every continuum-parent
    enumeration of the line; ``None`` reproduces the Round 4 cascade
    bit for bit.
    *ledger* (Round 6 runs R61 to R65) is an optional list that receives
    one :func:`~na49_mixing.flavor_weights.vertex_ledger_record` per drawn
    continuum vertex (the reweighting sufficient statistics); ``None``
    records nothing, and the ledger consumes no random number.

    Follows one cascade line from *state* through the termination rules
    of plan section 10.6.  At every line state, in order: the hard
    depth guard (a ``CascadeDepthGuardError`` naming the depth when
    ``state.depth > max_depth``); the catalog handoff
    (:func:`is_catalog_handoff` true delegates the remainder to the
    Round 2 kernel through :func:`handoff_decay` and terminates); the
    continuation through one ``draw_decay`` step over the open
    continuum pool (the continuum daughter continues the line); the
    direct-channel termination (:func:`has_valid_direct_channel` true
    decays through one ``draw_decay`` step over the direct pool and
    terminates); and the no-channel case (``ValueError``, a state
    outside the accepted model, checked never repaired).

    Returns the line as a list of :class:`~na49_mixing.cascade_decay.
    CascadeState`, root first, ending at the terminal state.  The
    terminal state identifies the termination mode exactly:
    :data:`TERMINATION_CATALOG_HANDOFF` when the terminal state
    satisfies :func:`is_catalog_handoff` (a catalog record within
    support; the kernel took over the remainder),
    :data:`TERMINATION_DIRECT_CHANNEL` when the terminal state is a
    continuum state whose enumeration opened no continuum channel (the
    direct hadronic decay was performed).  The decay products of the
    vertices are terminal established states outside the line; the
    kernel handoff's products are :func:`handoff_decay`'s
    ``DecayVertex`` list.

    Every vertex conserves ``(B, Q, S)`` exactly: card 4's
    ``draw_decay`` checks the continuum and direct vertices and raises
    :class:`~na49_mixing.resonance_decay.KinematicsError` on a
    violation (checked, never repaired), the handoff predicate checks
    the routing vertex against the committed record, and the delegated
    kernel's own conservation checks cover the handoff remainder.  The
    driver adds no random number of its own: the passed generators
    serve the line's ``draw_decay`` steps and the handoff delegations
    in order.
    """

    limit = int(max_depth)
    if limit < 0:
        raise ValueError(
            f"run_cascade: max_depth {max_depth!r} must be non-negative"
        )
    trace: list[CascadeState] = [state]
    current = state
    while True:
        # (1) the hard maximum depth guard (plan section 10.6, line
        # 757): card 4's depth counter, checked before any termination
        # rule, raised naming the depth, never repaired.
        depth = int(current.depth)
        if depth > limit:
            raise CascadeDepthGuardError(
                f"cascade depth guard: the line state of family "
                f"{current.family_id!r} at mass {float(current.mass)!r} "
                f"GeV carries depth {depth}, exceeding "
                f"MAX_CASCADE_DEPTH = {limit} (plan section 10.6, line "
                "757); the cascade is stopped at the guard, no repair, "
                "no truncation"
            )
        # (2) the catalog handoff (plan section 10.6, line 754): the
        # Round 2 kernel takes the remainder, byte-unchanged.
        if is_catalog_handoff(current, catalog):
            handoff_decay(current, catalog, rng_channel, rng_kinematics)
            return trace  # TERMINATION_CATALOG_HANDOFF at trace[-1]
        # (3) the enumeration at the realized mass (card 4)
        channels = enumerate_channels(current, catalog, fams, flavor=flavor)
        continuum = [
            channel for channel in channels
            if channel.kind == KIND_CONTINUUM
        ]
        if continuum:
            # (4) the continuation: the continuum daughter continues
            # the line (card 4's enumeration produces at most one
            # continuum daughter per channel, so the continuation is
            # unique).
            _channel, daughters, drawn = draw_decay_with_pool(
                current, continuum, catalog, fams,
                rng_channel, rng_kinematics,
            )
            if ledger is not None:
                ledger.append(vertex_ledger_record(
                    current.qn, drawn, _channel, catalog, fams, flavor))
            successor = next(
                daughter for daughter in daughters
                if not daughter.family_id.startswith(CATALOG_FAMILY_PREFIX)
            )
            trace.append(successor)
            current = successor
            continue
        # (5) the direct-channel termination (plan section 10.6, line
        # 755): one draw_decay step over the direct pool, then the line
        # terminates.
        if has_valid_direct_channel(current, channels):
            direct = [
                channel for channel in channels
                if channel.kind != KIND_CONTINUUM
            ]
            _channel, _daughters, drawn = draw_decay_with_pool(
                current, direct, catalog, fams,
                rng_channel, rng_kinematics,
            )
            if ledger is not None:
                ledger.append(vertex_ledger_record(
                    current.qn, drawn, _channel, catalog, fams, flavor))
            return trace  # TERMINATION_DIRECT_CHANNEL at trace[-1]
        # (6) no channel at all: neither pre-registered termination
        # rule applies and the line cannot continue.
        raise ValueError(
            f"run_cascade: the line state of family "
            f"{current.family_id!r} at mass {float(current.mass)!r} GeV "
            f"carries no open continuum channel and no valid direct "
            "hadronic channel, so neither termination rule of plan "
            "section 10.6 applies (a state outside the accepted model; "
            "checked, never repaired)"
        )
