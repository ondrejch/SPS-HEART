# Not used by the HEART p+p fits (marked 2026-11-04): no production generation, fit, probe tool or paper script
# runs this module (an execution trace of the R137 generation and an import scan of the engine, the probe tools
# and the paper scripts). Code reviews of the p+p model can skip it. It stays until the cleanup and refactor after
# the p+A run.
"""Inherited production axes of the aligned resonance cascade.

Round 4, card 5 (``TASK-20260920-r4-inherited-production-axes``; plan
section 22 task 22, plan section 10.1, lines 664-674).  The module
carries the axis-memory machinery of the aligned cascade: the initial
production axis of a source state, the rest-frame boost that transports
an axis between two cascade states, and the inheritance rule that fills
card 4's ``CascadeState.axis`` field on the drawn daughters.  Cards 6-7
act on the inherited axes (their alignment kernels evaluate the plan
section 10.4 angular laws against them); card 9 verifies the depth of
cascades whose axes are inherited here.  Card 4's ``cascade_decay.py``
surface is consumed read-only: the ``axis`` field it copies verbatim is
replaced by this module's inheritance rules at the call sites, without
touching that module.

Axis convention (the round 4 index item 4 pre-registration)
-----------------------------------------------------------

Every cascade state carries the production axis in its own rest frame
(plan section 10.1, line 672).  The convention pins:

- **Initial source axis.**  In p+p events the production axis is the
  beam direction transported into the source rest frame (the
  longitudinal axis of the M8-frozen two-source topology; for the
  collinear sources the transport returns the beam axis exactly).  In
  isolated-source runs (R40) it is the lab z-axis: the interface marks
  the isolated source with ``beam_direction = None``.
- **Inheritance.**  The baryon-number-carrying daughter (``B != 0``)
  inherits the parent axis transported to the daughter rest frame: it
  continues the incoming baryonic line of plan section 10.4 (line 718),
  so its angular law is evaluated against the transported production
  axis.  A symmetric mesonic daughter (``B == 0``) takes the parent
  rest-frame axis; the even cos^2 law of plan section 10.4 (line 724)
  is sign-symmetric, and the pairing of the branch with the kernel and
  the alignment label is cards 6-7's territory (card 4's isotropic
  label placeholder stays until they land).
- **Determinism.**  The axis is a deterministic Lorentz-transformed
  unit vector: the machinery consumes zero random numbers on any
  substream, so the card-4 draw budget (1 channel + 2 kinematics per
  vertex) is unchanged.

Frame conventions
-----------------

The four-momenta passed to this module are expressed in one common
working frame (the frame the cascade states carry; card 4's
``draw_decay`` returns daughters in the parent's working frame).  Each
axis is expressed in the rest frame of the state that carries it, and
the rest-frame axes of every state are the pure-boost images of the
working-frame axes: the cascade carries no Wigner rotation, so the
axis transport is path-independent and the production axis chains from
the source through every vertex without an arbitrary twist.

The transport between two rest frames therefore composes the two
states' own boosts through the working frame:

1. the axis (a null four-vector along the direction) is boosted from
   the parent rest frame into the working frame with the parent's
   velocity;
2. the boosted null vector is boosted into the daughter rest frame
   with the negated daughter velocity;
3. the spatial part of the result is renormalized.

Each step is the kernel's byte-unchanged ``_boost_lab`` (the same Lorentz
boost card 4 uses for the daughter momenta).  This is the aberration
law ``cos(theta') = (cos(theta) - beta) / (1 - beta cos(theta))`` with
the azimuth preserved, applied per boost.  The composition is its own
exact inverse under the argument swap, so the round trip through a
daughter rest frame and back recovers the axis (within the
:data:`AXIS_NORM_TOL` floating-point tolerance), and the transported
axis has unit norm by construction.

A state at rest in the working frame (or a daughter at rest inside the
parent rest frame) leaves the axis unchanged.  Every input is checked,
never repaired: a non-timelike or non-finite four-momentum, a zero
direction, or a daughter without a rest frame raises the kernel's
:class:`~na49_mixing.resonance_decay.KinematicsError`; a malformed
quantum-number tuple raises ``ValueError``.

Hermetic: standard library and the vendored ``na49_mixing`` modules
only; no event generation, no external tables, no Monte Carlo run (the
acceptance is hermetic: pinned fixtures and Lorentz identities; the
runs R40-R44 exercise the surface statistically).
"""

from __future__ import annotations

import math

from .resonance_decay import KinematicsError, _boost_lab

__all__ = [
    "AXIS_NORM_TOL",
    "LAB_Z_AXIS",
    "initial_source_axis",
    "boost_axis_to_rest_frame",
    "inherit_axis",
]


#: The floating-point tolerance of the axis machinery (the card's
#: pre-registered unit-norm and round-trip bound).
AXIS_NORM_TOL: float = 1e-12

#: The lab z-axis, the production axis of an isolated source (the round
#: 4 index item 4 pre-registration for the isolated-source runs R40).
LAB_Z_AXIS: tuple[float, float, float] = (0.0, 0.0, 1.0)


# ----------------------------------------------------------------------
# Validation helpers (checked, never repaired)
# ----------------------------------------------------------------------
def _validate_axis(axis, name: str) -> tuple[tuple[float, float, float], float]:
    """The float-cast 3-vector *name*d *axis* and its norm; finite with
    a positive norm, else :class:`KinematicsError`."""

    values = tuple(float(x) for x in axis)
    if len(values) != 3:
        raise KinematicsError(
            f"cascade axes: {name} must carry three components, got "
            f"{len(values)}"
        )
    for component in values:
        if not math.isfinite(component):
            raise KinematicsError(
                f"cascade axes: {name} component {component!r} is not "
                "finite"
            )
    norm = math.sqrt(
        values[0] * values[0] + values[1] * values[1] + values[2] * values[2]
    )
    if not norm > 0.0:
        raise KinematicsError(
            f"cascade axes: {name} is the zero vector; a production "
            "axis is a direction"
        )
    return values, norm


def _validate_four_momentum(p4, name: str) -> tuple[float, float, float, float]:
    """The float-cast four-momentum *name*d *p4*; finite, future-directed
    and timelike (it has a rest frame), else :class:`KinematicsError`."""

    values = tuple(float(x) for x in p4)
    if len(values) != 4:
        raise KinematicsError(
            f"cascade axes: {name} must carry four components, got "
            f"{len(values)}"
        )
    for component in values:
        if not math.isfinite(component):
            raise KinematicsError(
                f"cascade axes: {name} component {component!r} is not "
                "finite"
            )
    energy = values[0]
    if not energy > 0.0:
        raise KinematicsError(
            f"cascade axes: {name} energy {energy!r} is not positive; "
            "the four-momentum is not future-directed"
        )
    m2 = (
        energy * energy
        - values[1] * values[1]
        - values[2] * values[2]
        - values[3] * values[3]
    )
    if not m2 > 0.0:
        raise KinematicsError(
            f"cascade axes: {name} is not timelike (E^2 - p^2 = {m2!r} "
            "GeV^2); it has no rest frame"
        )
    return values


def _transport_direction(
    direction: tuple[float, float, float],
    norm: float,
    beta: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Transport the spatial *direction* (of norm *norm*) into the rest
    frame of an object moving with velocity *beta*.

    The null four-vector ``(|direction|, direction)`` is boosted with
    the negated *beta* through the kernel's byte-unchanged
    ``_boost_lab`` and the spatial part of the boosted null vector is
    renormalized (light aberration; the boosted null vector keeps
    ``|spatial| = time component``, so the renormalized direction is
    the unit transported axis).  A zero *beta* leaves the direction
    unchanged (the two frames coincide).  Unit norm by construction.
    """

    bx, by, bz = (float(x) for x in beta)
    boosted = _boost_lab(
        (norm, direction[0], direction[1], direction[2]), (-bx, -by, -bz)
    )
    tx, ty, tz = boosted[1], boosted[2], boosted[3]
    transported_norm = math.sqrt(tx * tx + ty * ty + tz * tz)
    if not transported_norm > 0.0:
        raise KinematicsError(
            "cascade axes: the transported direction degenerated (the "
            f"boost velocity {beta!r} is at or beyond the speed of "
            "light for this direction); no rest-frame axis exists"
        )
    return (tx / transported_norm, ty / transported_norm, tz / transported_norm)


# ----------------------------------------------------------------------
# The initial source axis (round 4 index item 4)
# ----------------------------------------------------------------------
def initial_source_axis(
    source_momentum: tuple[float, float, float, float],
    beam_direction: tuple[float, float, float] | None,
) -> tuple[float, float, float]:
    """Unit production axis in the source rest frame.

    p+p: the beam direction (the M8-frozen two-source longitudinal
    axis) transported into the source rest frame; isolated source: the
    lab z-axis (``beam_direction is None``, the R40 pre-registration).
    *source_momentum* is the source four-momentum in the working frame;
    *beam_direction* is the beam direction in the working frame (any
    nonzero scale; the direction is normalized).  Deterministic: no
    random number is consumed.
    """

    energy, px, py, pz = _validate_four_momentum(
        source_momentum, "source_momentum"
    )
    if beam_direction is None:
        return (float(LAB_Z_AXIS[0]), float(LAB_Z_AXIS[1]), float(LAB_Z_AXIS[2]))
    direction, norm = _validate_axis(beam_direction, "beam_direction")
    beta = (px / energy, py / energy, pz / energy)
    return _transport_direction(direction, norm, beta)


# ----------------------------------------------------------------------
# The rest-frame boost of an axis
# ----------------------------------------------------------------------
def boost_axis_to_rest_frame(
    axis: tuple[float, float, float],
    parent_momentum: tuple[float, float, float, float],
    daughter_momentum: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """Transport the parent rest-frame axis into the daughter rest
    frame; unit norm preserved.

    *axis* is the production axis in the parent rest frame.
    *parent_momentum* and *daughter_momentum* are the two four-momenta
    in the common working frame (card 4's ``draw_decay`` daughter
    momenta qualify).  The axis null vector is boosted into the working
    frame with the parent velocity and into the daughter rest frame
    with the negated daughter velocity (the two ``_boost_lab`` steps of
    the module docstring), and the spatial part is renormalized.  The
    composition is its own exact inverse under the argument swap, so
    the round trip recovers the axis within :data:`AXIS_NORM_TOL`.
    Deterministic: no random number is consumed.
    """

    direction, norm = _validate_axis(axis, "axis")
    parent = _validate_four_momentum(parent_momentum, "parent_momentum")
    daughter = _validate_four_momentum(daughter_momentum, "daughter_momentum")
    beta_parent = (parent[1] / parent[0], parent[2] / parent[0], parent[3] / parent[0])
    beta_daughter = (
        daughter[1] / daughter[0],
        daughter[2] / daughter[0],
        daughter[3] / daughter[0],
    )
    in_working = _boost_lab(
        (norm, direction[0], direction[1], direction[2]), beta_parent
    )
    in_daughter_rest = _boost_lab(
        in_working, (-beta_daughter[0], -beta_daughter[1], -beta_daughter[2])
    )
    tx, ty, tz = in_daughter_rest[1], in_daughter_rest[2], in_daughter_rest[3]
    transported_norm = math.sqrt(tx * tx + ty * ty + tz * tz)
    if not transported_norm > 0.0:
        raise KinematicsError(
            "cascade axes: the transported direction degenerated (the "
            "boost velocities are at or beyond the speed of light for "
            "this direction); no rest-frame axis exists"
        )
    return (tx / transported_norm, ty / transported_norm, tz / transported_norm)


# ----------------------------------------------------------------------
# The inheritance rule (the card 4 daughter-state axis)
# ----------------------------------------------------------------------
def inherit_axis(
    daughter_qn: tuple[int, int, int],
    parent_axis: tuple[float, float, float],
    parent_momentum,
    daughter_momentum,
) -> tuple[float, float, float]:
    """The card 4 ``draw_decay`` daughter state axis.

    The baryon-carrying daughter (``B != 0``) inherits the parent axis
    transported to the daughter rest frame (it continues the incoming
    baryonic line, plan section 10.4 line 718).  A symmetric mesonic
    daughter (``B == 0``) takes the parent rest-frame axis; the even-law
    kernel choice and the alignment label are cards 6-7's territory.
    *parent_momentum* and *daughter_momentum* are the two four-momenta
    in the common working frame.  Deterministic: no random number is
    consumed.
    """

    qn = tuple(int(value) for value in daughter_qn)
    if len(qn) != 3:
        raise ValueError(
            f"inherit_axis: daughter_qn {daughter_qn!r} must carry the "
            "three integer quantum numbers (B, Q, S)"
        )
    if qn[0] != 0:
        return boost_axis_to_rest_frame(
            parent_axis, parent_momentum, daughter_momentum
        )
    direction, _norm = _validate_axis(parent_axis, "parent_axis")
    return direction
