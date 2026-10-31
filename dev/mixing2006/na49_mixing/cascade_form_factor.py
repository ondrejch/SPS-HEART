"""The hadronic decay form factor of the aligned resonance cascade.

Round 4, card 8 (``TASK-20260920-r4-hadronic-decay-form-factor``; plan
section 22 task 25, line 1747, plan section 10.5, lines 734-748).  The
module carries the decay-transverse vertex factor of the aligned
cascade, the W_dec attachment of card 4's channel weight, and nothing
else: the exchange recoil layer stays the builder's byte-unchanged
territory, and the module is the mechanism-side guarantee of the plan
section 10.5 separation.

The separation contract (plan section 10.5, lines 734-748)
----------------------------------------------------------

Plan section 10.5 separates two transverse structures that must never
share one parameter:

- **The exchange layer.**  Exchange transfer uses an exchange-level
  distribution ``dP/dt`` proportional to the exponential of the
  transfer variable (plan line 739); it lives in the builder
  (``na49_mixing.resonance_pp_builder``, byte-unchanged at sha256
  ``4ab61260...``) with the exchange slope carried by
  ``na49_mixing.exchange_model``, and this module neither reads it nor
  feeds it.  Plan section 24 (line 1866) forbids letting exchange
  slopes control daughter-level transverse dynamics; the card 13 run
  (R42) profiles the two effects against each other, and this
  module's clean boundary is what makes that profile interpretable.
- **The decay layer.**  Hadronic decay transverse structure arises
  from the two-body momentum, the angular alignment (cards 6-7's
  kernels, consumed read-only), and the hadronic vertex form factor
  implemented here,

      |F_dec(q_perp)|^2 = exp(-b_dec q_perp^2),        (plan line 745)

  with ``b_dec`` a named decay parameter of its own: the pre-registered
  default ``1.0 GeV^-2`` (the conventional hadronic vertex form-factor
  scale, the R40 nominal point of round 4 index item 3), fitted in R41
  (card 12) and profiled against the exchange slope in R42 (card 13),
  refit nowhere else in Round 4.  ``b_dec = 0`` gives the factor exactly
  ``1.0`` (the off limit; the isotropic ablation baseline the R43
  cards 10 and 14 depend on).  The exchange slope is the
  exchange-layer parameter of plan line 739 and lives in the builder's
  exchange-model surface, byte-unchanged; no single parameter controls
  both source recoil and daughter transverse momentum (plan line
  748), and this module defines the decay-side name of the pair.

The Blatt-Weisskopf barrier factor ``F_L`` of plan section 10.3 is
Round 2 territory (round 2 index item 4) and stays inside the Round 2
kernel's width law, byte-unchanged; ``F_dec`` is the section 10.5
decay-transverse factor and is distinct from it by construction.

The q_perp convention
---------------------

``q_perp`` is the daughter momentum in the parent rest frame,
transverse to the unit inherited production axis: with ``p`` the
daughter three-momentum in the parent rest frame (the two-body momentum
of the channel; the spatial part of the daughter four-momentum card
4's ``draw_decay`` kinematics realizes) and ``a`` the parent state's
production axis (card 5's axis machinery, consumed read-only; at this
card's landing the axis is the caller-supplied state axis of card 4),

    q_perp^2 = |p|^2 - (p . a)^2,                    (plan line 745's
                                                      squared form)

the magnitude of the momentum component perpendicular to the axis.
The axis must be unit (the card-5 convention; validated within
:data:`na49_mixing.cascade_axes.AXIS_NORM_TOL`, checked never
repaired).  A zero daughter momentum carries no transverse component
(``q_perp = 0``, the factor 1), and the roundoff clamp of card 4's
``sin_theta`` (``max(0.0, q_perp^2)``) absorbs the floating-point
negative that a near-parallel configuration can produce.

The form factor is a deterministic weight: it consumes zero random
numbers on any substream, so the card-4 draw budget (1 channel + 2
kinematics per vertex, frozen through cards 6-8) is unchanged.

The attachment on card 4's channel weight
-----------------------------------------

:func:`attach_to_channel_weight` wraps card 4's
:func:`na49_mixing.cascade_decay.channel_weight` with the W_dec factor
(a separate named factor, distinct from the ``W_flavor`` / ``W_align``
hooks of card 4 and from the Round 2 ``F_L`` barrier).  The wrapped
callable has the signature

    wrapped(channel, daughter_momentum, axis,
            w_flavor = 1.0, w_align = 1.0) -> float

and returns

    channel_weight(channel, w_flavor, w_align)
        * |F_dec(q_perp)|^2,

with ``q_perp`` from the two-body momentum of the channel and the
inherited axis of the parent state.  The hooks are passed through, so
the wrapped weight is the card-4 hooked weight scaled by the form
factor: the normalized shares over a vertex change exactly by the
form-factor factor per channel, ``w_i F_i / sum_j w_j F_j``.  The
attachment changes no draw order and no per-vertex draw count: the
factor is a weight modification, not a draw site, and it is applied in
the cascade decay weighting only, never in the exchange layer.

Hermetic: standard library, the vendored ``na49_mixing`` constants
(the card-5 axis tolerance and the kernel's ``KinematicsError``), and
nothing else; no event generation, no external tables, no Monte Carlo
run (the acceptance is hermetic: pinned fixtures and the white-box
separation; the runs R40-R44 exercise the factor statistically).
"""

from __future__ import annotations

import math
from collections.abc import Callable

from .cascade_axes import AXIS_NORM_TOL
from .resonance_decay import KinematicsError

__all__ = [
    "DECAY_FORM_FACTOR_B_DEFAULT",
    "transverse_momentum",
    "decay_form_factor_weight",
    "attach_to_channel_weight",
]


#: The pre-registered default form-factor slope ``b_dec`` in GeV^-2
#: (plan section 10.5; the conventional hadronic vertex form-factor
#: scale, the R40 nominal point of round 4 index item 3; fitted in R41
#: and profiled against the exchange slope in R42).  ``b_dec = 0`` is
#: the off limit: the factor is exactly ``1.0`` there, reproducing the
#: card 4/6/7 baseline bit for bit (the R43 ablation baseline).
DECAY_FORM_FACTOR_B_DEFAULT: float = 1.0


# ----------------------------------------------------------------------
# Validation helpers (checked, never repaired)
# ----------------------------------------------------------------------
def _validate_daughter_momentum(
    daughter_momentum, name: str
) -> tuple[float, float, float, float]:
    """The float-cast four-momentum *name*d *daughter_momentum*; four
    finite components, else :class:`KinematicsError`.

    Only the spatial part enters the ``q_perp`` convention, and a
    massless daughter is legitimate (its four-momentum is null), so no
    energy-sign or timelike requirement is imposed: the pre-registered
    formula is purely spatial.
    """

    values = tuple(float(x) for x in daughter_momentum)
    if len(values) != 4:
        raise KinematicsError(
            f"cascade form factor: {name} must carry four components, "
            f"got {len(values)}"
        )
    for component in values:
        if not math.isfinite(component):
            raise KinematicsError(
                f"cascade form factor: {name} component {component!r} "
                "is not finite"
            )
    return values


def _validate_axis_unit(
    axis, name: str
) -> tuple[float, float, float]:
    """The float-cast 3-vector *name*d *axis*; finite and unit within
    :data:`na49_mixing.cascade_axes.AXIS_NORM_TOL` (the card-5 axis
    convention, consumed read-only), else :class:`KinematicsError`.

    The ``q_perp`` formula is defined for the unit axis; a caller
    carrying a scaled vector would silently misproject the daughter
    momentum, so the unit norm is checked, never repaired.
    """

    values = tuple(float(x) for x in axis)
    if len(values) != 3:
        raise KinematicsError(
            f"cascade form factor: {name} must carry three components, "
            f"got {len(values)}"
        )
    for component in values:
        if not math.isfinite(component):
            raise KinematicsError(
                f"cascade form factor: {name} component {component!r} "
                "is not finite"
            )
    norm = math.sqrt(
        values[0] * values[0] + values[1] * values[1] + values[2] * values[2]
    )
    if not abs(norm - 1.0) <= AXIS_NORM_TOL:
        raise KinematicsError(
            f"cascade form factor: {name} is not unit (norm {norm!r} "
            f"beyond the card-5 tolerance {AXIS_NORM_TOL!r}); the "
            "q_perp convention is defined for the unit inherited "
            "production axis"
        )
    return values


def _validate_q_perp(q_perp) -> float:
    """The float-cast *q_perp*; finite and non-negative, else
    ``ValueError`` (the weight-layer parameter discipline of card 4's
    hooks and the card-6/7 kappa validation)."""

    value = float(q_perp)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"cascade form factor: q_perp {q_perp!r} must be finite "
            "and non-negative"
        )
    return value


def _validate_b_dec(b_dec) -> float:
    """The float-cast *b_dec*; finite and non-negative, else
    ``ValueError``.  A negative slope would make the factor grow with
    ``q_perp^2`` (an unphysical vertex form factor), and the R40 scan
    grid and the R41 fit keep ``b_dec >= 0``."""

    value = float(b_dec)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"cascade form factor: b_dec {b_dec!r} must be finite "
            "and non-negative (a negative slope is an unphysical "
            "vertex form factor)"
        )
    return value


# ----------------------------------------------------------------------
# The q_perp convention (plan section 10.5)
# ----------------------------------------------------------------------
def transverse_momentum(
    daughter_momentum: tuple[float, float, float, float],
    axis: tuple[float, float, float],
) -> float:
    """``q_perp``: the daughter momentum in the parent rest frame,
    transverse to the unit inherited production axis.

    *daughter_momentum* is the daughter four-momentum ``(E, px, py,
    pz)`` in the parent rest frame (the two-body momentum of the
    channel; card 4's ``draw_decay`` kinematics realizes exactly this
    configuration).  *axis* is the parent state's production axis in
    the parent rest frame (unit; card 5's axis machinery, consumed
    read-only; at this card's landing the caller-supplied state axis).
    The convention is

        q_perp^2 = |p|^2 - (p . a)^2,

    the magnitude of the momentum component perpendicular to the axis,
    returned as the positive square root.  A zero daughter momentum
    returns ``0.0`` (no transverse component at the two-body
    threshold), and the roundoff clamp of card 4's ``sin_theta``
    absorbs the floating-point negative a near-parallel configuration
    can produce.  Deterministic: no random number is consumed.
    """

    p = _validate_daughter_momentum(daughter_momentum, "daughter_momentum")
    a = _validate_axis_unit(axis, "axis")
    px, py, pz = p[1], p[2], p[3]
    p2 = px * px + py * py + pz * pz
    pa = px * a[0] + py * a[1] + pz * a[2]
    q2 = p2 - pa * pa
    return math.sqrt(max(0.0, q2))


# ----------------------------------------------------------------------
# The form factor (plan section 10.5, line 745)
# ----------------------------------------------------------------------
def decay_form_factor_weight(q_perp: float, b_dec: float) -> float:
    """``|F_dec(q_perp)|^2 = exp(-b_dec * q_perp^2)``; exactly ``1.0``
    at ``b_dec = 0``.

    The decay-transverse vertex factor of plan section 10.5 (line
    745) as a deterministic weight: it consumes zero random numbers,
    and at the off limit ``b_dec = 0`` it returns ``1.0`` exactly (the
    IEEE ``exp(-0.0) == 1.0`` identity, so the attached weights are
    the unattached ones bit for bit; the R43 ablation baseline).  A
    non-finite or negative ``q_perp`` or ``b_dec`` raises
    ``ValueError``.
    """

    q = _validate_q_perp(q_perp)
    b = _validate_b_dec(b_dec)
    return math.exp(-b * q * q)


# ----------------------------------------------------------------------
# The W_dec attachment on card 4's channel weight
# ----------------------------------------------------------------------
def attach_to_channel_weight(
    channel_weight: Callable, b_dec: float
) -> Callable:
    """The separate named factor on card 4's channel weight: the
    weighted channel is scaled by :func:`decay_form_factor_weight` of
    the daughter ``q_perp``; no change to the channel draw order or
    the per-vertex draw budget.

    *channel_weight* is card 4's
    :func:`na49_mixing.cascade_decay.channel_weight` (a callable with
    the signature ``(channel, w_flavor = 1.0, w_align = 1.0)``).  The
    returned callable has the signature

        wrapped(channel, daughter_momentum, axis,
                w_flavor = 1.0, w_align = 1.0) -> float

    and returns the card-4 hooked weight scaled by the form factor at
    the ``q_perp`` of the channel's two-body daughter momentum against
    the parent state's inherited axis: the W_dec factor is
    multiplicative on top of the passed-through hooks, so the
    normalized shares over a vertex change exactly by the form-factor
    factor per channel (``w_i F_i / sum_j w_j F_j``), and at ``b_dec
    = 0`` the wrapped weights are the unattached ones bit for bit.
    The W_dec factor is distinct from the ``W_flavor`` / ``W_align``
    hooks of card 4 and from the Round 2 Blatt-Weisskopf barrier
    ``F_L`` (which stays in the Round 2 kernel's width law,
    byte-unchanged).  The wrapper is a pure weight: it consumes zero
    random numbers, changes no draw order, and adds no uniform to the
    card-4 per-vertex budget (1 channel + 2 kinematics).

    A non-callable *channel_weight* or a non-finite / negative *b_dec*
    raises ``ValueError`` at wrap time (checked, never repaired).
    """

    if not callable(channel_weight):
        raise ValueError(
            "attach_to_channel_weight: channel_weight must be card 4's "
            f"channel-weight callable, got {channel_weight!r}"
        )
    validated_b = _validate_b_dec(b_dec)

    def decay_form_factor_channel_weight(
        channel,
        daughter_momentum: tuple[float, float, float, float],
        axis: tuple[float, float, float],
        w_flavor: float = 1.0,
        w_align: float = 1.0,
    ) -> float:
        """The card-4 hooked channel weight scaled by the W_dec factor
        at the channel's daughter ``q_perp`` against the inherited
        axis (the module docstring's attachment contract)."""

        base = channel_weight(channel, w_flavor, w_align)
        q_perp = transverse_momentum(daughter_momentum, axis)
        return base * decay_form_factor_weight(q_perp, validated_b)

    decay_form_factor_channel_weight.__name__ = (
        "decay_form_factor_channel_weight"
    )
    decay_form_factor_channel_weight.__qualname__ = (
        "attach_to_channel_weight.<locals>."
        "decay_form_factor_channel_weight"
    )
    decay_form_factor_channel_weight.__doc__ = (
        "The card-4 hooked channel weight scaled by the W_dec factor "
        "at the channel's daughter q_perp against the inherited axis "
        "(the card-8 attachment contract; pure, zero random numbers)."
    )
    return decay_form_factor_channel_weight
