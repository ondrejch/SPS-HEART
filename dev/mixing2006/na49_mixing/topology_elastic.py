"""The elastic reference event class (Round 5, card 1).

``TASK-20260925-r5-elastic-reference``; plan section 22 task 28
(line 1753); plan section 11.1 item 1 (line 824).  The card implements
the intact-line, no-excitation event class: both incoming hadronic
lines stay intact protons, a single vacuum exchange carries zero
quantum-number transfer, and no excited state, central system, or
decay vertex exists in the event record.  The class is the
normalization reference that every other Round 5 class is defined
against (plan line 824: "used for normalization and code checks, not
included in an inelastic sample"; the class is excluded from every
inelastic sample of runs R51-R55 by its ``"elastic"`` class label).

Exchange and ledger conventions (plan section 11.4, lines 871-881)
------------------------------------------------------------------

The event record carries the per-line incoming/outgoing quantum
numbers and the single exchange record in the doubled-I3 integer
convention ``(B, Q, S, 2I3)``; the proton is ``(B, Q, S, 2I3) =
(1, 1, 0, 1)``.  The vacuum exchange transfers ``(0, 0, 0, 0)``
(plan line 877), so every outgoing line qn equals its incoming line
qn; :func:`check_zero_transfer` verifies this in exact integer
arithmetic and raises :class:`ElasticTransferViolation` naming the
offending line or exchange.  Card 2's ledger audit consumes exactly
these record fields, read-only and without further translation.

The t law (the card's pre-registered instantiation)
---------------------------------------------------

The momentum transfer t is drawn from the Regge-motivated density
``dP/dt proportional to exp(B_SD t)`` on the exact two-body elastic
range ``[t_min, 0]``, with the closed-form normalization
``(1/B_SD) * (1 - exp(B_SD * t_min))`` and the inverse-CDF draw of one
uniform.  The nominal slope is fixed at the M8-frozen
``effective_transfer_slope`` :data:`ELASTIC_TRANSFER_SLOPE =
8.504140762083612 GeV^-2` as a named module constant; no trajectory
parameters, intercepts, or residue parameters are exposed (plan line
894).  The pinned M8 p+p configuration is beam :data:`BEAM_MOMENTUM =
158.0 GeV/c` at the recorded :data:`SQRT_S = 17.270 GeV`, and the
proton mass is the catalog value :data:`PROTON_MASS = 0.93827231 GeV`.

Two floating-point facts of the pinned configuration are recorded here
because they bound the reachable draw domain:

- ``B_SD * t_min`` is about ``-2506.6``, so ``exp(B_SD * t_min)``
  underflows to ``0.0`` in IEEE-754 double precision.  The closed-form
  normalization therefore evaluates as ``1/B_SD`` and the inverse-CDF
  draw maps a float64 uniform ``u in (0, 1]`` to ``t = ln(u)/B_SD``,
  so the smallest positive uniform bounds the reachable ``|t|`` far
  inside the two-body limit (no draw can reach the domain edge).
- the recorded ``SQRT_S = 17.270 GeV`` is the rounded quote of the
  exact fixed-target value ``17.270205201372406 GeV`` (the
  ``Reaction(158, 2212, 2212)`` invariant mass).  The nominal range
  edge ``-(2 m_p p_beam / SQRT_S)^2`` therefore sits about
  ``0.0070 GeV^2`` beyond the exact two-body limit
  ``s_exact - 4 m_p^2`` of the beam-momentum configuration; the guard
  in :func:`generate_elastic` checks the scattering angle against the
  exact limit and raises rather than repairing.

Exact elastic kinematics
------------------------

:func:`generate_elastic` solves the outgoing four-momenta exactly from
the drawn t: in the center-of-mass frame of the two incoming protons
the outgoing momentum magnitude is the elastic value ``q`` and the
scattering angle follows from ``t = -2 q^2 (1 - cos(theta*))``, so
``cos(theta*) = 1 + t/(2 q^2)``; the pair is boosted into the lab frame
with the fixed-target CM velocity ``beta = p_beam/(E_beam + m_p)``
along the beam axis.  The production plane is fixed to the x-z plane
(the azimuthal orientation carries no draw: the single-uniform budget
draws t only, and the event record fixes the convention).  Four-momentum
conservation holds by construction (the CM pair sums to ``(E_cm, 0, 0,
0)`` and the boost maps it to the incoming sum), and both outgoing
protons sit on the ``m_p`` mass shell.  The t variable acts on the
exchange kinematics only: the elastic event has no daughters, so no
exchange slope controls any daughter-level transverse dynamics (plan
section 24, line 1866).

Draw budget
-----------

:func:`generate_elastic` consumes exactly one uniform on the closed
``exchange_kinematics`` substream label
(:data:`na49_mixing.rng_substreams.SUBSTREAM_LABELS`) per event, zero
elsewhere; no other random consumption exists in the module.

Hermetic: standard library and numpy only; no event generation
driver, no external tables, no Monte Carlo run (the acceptance is
hermetic: pinned traces, closed-form identities, draw budgets; the
runs R50-R55 exercise the class statistically).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "BEAM_MOMENTUM",
    "ELASTIC_TRANSFER_SLOPE",
    "PROTON_MASS",
    "SQRT_S",
    "ElasticEvent",
    "ElasticTransferViolation",
    "ExchangeRecord",
    "LineState",
    "check_zero_transfer",
    "draw_elastic_t",
    "elastic_t_range",
    "generate_elastic",
]


#: The nominal elastic exchange slope, the M8-frozen
#: ``effective_transfer_slope`` (the R50 nominal value of the round 5
#: index item 2) [GeV^-2].  A named constant: no trajectory, intercept,
#: or residue parameter is exposed (plan line 894).
ELASTIC_TRANSFER_SLOPE: float = 8.504140762083612

#: The proton mass, the catalog value (the committed catalog record
#: pdg_id 2212, pole_mass) [GeV].
PROTON_MASS: float = 0.93827231

#: The M8 p+p beam momentum [GeV/c].
BEAM_MOMENTUM: float = 158.0

#: The recorded M8 p+p collision energy [GeV] (the rounded quote of the
#: exact fixed-target invariant mass 17.270205201372406 GeV; see the
#: module docstring).
SQRT_S: float = 17.270

#: The proton quantum numbers in the doubled-I3 integer convention
#: ``(B, Q, S, 2I3)`` (the committed catalog record pdg_id 2212).
PROTON_QN: tuple[int, int, int, int] = (1, 1, 0, 1)

#: The zero quantum-number transfer of the vacuum exchange (plan line
#: 877), the doubled-I3 integer convention.
ZERO_TRANSFER: tuple[int, int, int, int] = (0, 0, 0, 0)

#: The class label; the elastic class is excluded from every inelastic
#: sample by this label (plan line 824).
ELASTIC_CLASS_LABEL: str = "elastic"

#: The vacuum exchange name (the single color-singlet exchange of the
#: elastic class).
VACUUM_EXCHANGE_NAME: str = "vacuum"


class ElasticTransferViolation(Exception):
    """A zero-transfer violation of the elastic ledger (checked, never
    repaired): an outgoing line qn differs from its incoming line qn,
    or the exchange carries a nonzero ``(B, Q, S, 2I3)`` transfer."""


@dataclass(frozen=True)
class LineState:
    """One hadronic line: its four-momentum and its doubled-I3 integer
    quantum numbers ``(B, Q, S, 2I3)``."""

    four_momentum: tuple[float, float, float, float]
    qn: tuple[int, int, int, int]


@dataclass(frozen=True)
class ExchangeRecord:
    """One exchange: its name and its declared ``(B, Q, S, 2I3)``
    transfer (the vacuum exchange transfers zero, plan line 877)."""

    name: str
    transfer: tuple[int, int, int, int]


@dataclass(frozen=True)
class ElasticEvent:
    """The frozen elastic event record (the card's ledger-ready
    record): the class label, the two incoming and two outgoing line
    states, the single exchange record, and the momentum transfer t
    [GeV^2].  No excited state, central system, or decay vertex exists
    in the record."""

    class_label: str
    incoming: tuple[LineState, LineState]
    outgoing: tuple[LineState, LineState]
    exchange: ExchangeRecord
    t: float


# ----------------------------------------------------------------------
# The t law
# ----------------------------------------------------------------------
def elastic_t_range(sqrt_s: float) -> tuple[float, float]:
    """Exact elastic t range ``[t_min, 0]`` in GeV^2 for the M8 p+p
    configuration (two-body kinematics, both lines at :data:`PROTON_MASS`,
    beam :data:`BEAM_MOMENTUM`).

    The two-body range edge is ``t_min = -(2 m_p p_beam / sqrt_s)^2``,
    the squared four-momentum transfer of a back-to-back elastic pair
    at the fixed-target invariant mass *sqrt_s*.  ``sqrt_s`` must be a
    finite number above the two-proton threshold ``2 m_p``; the right
    edge is exactly ``0.0``.
    """

    s = float(sqrt_s)
    if not math.isfinite(s) or s <= 2.0 * PROTON_MASS:
        raise ValueError(
            "elastic_t_range: sqrt_s must be a finite GeV value above "
            f"the two-proton threshold {2.0 * PROTON_MASS}, got {sqrt_s!r}"
        )
    t_min = -((2.0 * PROTON_MASS * BEAM_MOMENTUM / s) ** 2)
    return (t_min, 0.0)


def elastic_t_normalization(t_min: float) -> float:
    """Closed-form normalization of ``exp(B_SD t)`` on ``[t_min, 0]``:
    ``(1/B_SD) * (1 - exp(B_SD * t_min))``.

    Exposed for the normalization identity check (acceptance (iii));
    the draw itself uses the same closed form internally.
    """

    if not math.isfinite(t_min) or t_min >= 0.0:
        raise ValueError(
            "elastic_t_normalization: t_min must be a finite negative "
            f"GeV^2 value, got {t_min!r}"
        )
    return (1.0 - math.exp(ELASTIC_TRANSFER_SLOPE * t_min)) / ELASTIC_TRANSFER_SLOPE


def elastic_t_quantile(u: float, t_min: float) -> float:
    """The inverse CDF of ``exp(B_SD t)`` on ``[t_min, 0]`` at the
    quantile *u*:

    ``t(u) = ln(exp(B_SD t_min) + u (1 - exp(B_SD t_min))) / B_SD``.

    ``u`` must lie in ``[0, 1]``.  For the pinned configuration the
    float64 evaluation of ``exp(B_SD t_min)`` underflows to ``0.0``, so
    ``u = 0`` raises ``ValueError`` (an empty log domain) and the
    drawn domain is effectively bounded by the smallest positive
    uniform; see the module docstring.
    """

    if not 0.0 <= u <= 1.0:
        raise ValueError(
            f"elastic_t_quantile: u must lie in [0, 1], got {u!r}"
        )
    e_lo = math.exp(ELASTIC_TRANSFER_SLOPE * t_min)
    if e_lo == 0.0 and u == 0.0:
        raise ValueError(
            "elastic_t_quantile: u = 0 maps to the range edge t_min = "
            f"{t_min!r}, whose density weight exp(B_SD t_min) underflows "
            "in double precision; no float64 uniform reaches the edge"
        )
    return math.log(e_lo + u * (1.0 - e_lo)) / ELASTIC_TRANSFER_SLOPE


def draw_elastic_t(rng: np.random.Generator) -> float:
    """One t draw from ``exp(B_SD t)`` on the pinned two-body range
    ``[t_min, 0]`` via the inverse CDF, consuming exactly one uniform
    from *rng* (the ``exchange_kinematics`` substream).

    The closed-form normalization is
    ``(1/B_SD) * (1 - exp(B_SD * t_min))``.
    """

    t_min, _t_max = elastic_t_range(SQRT_S)
    u = rng.random()
    return elastic_t_quantile(u, t_min)


# ----------------------------------------------------------------------
# Exact elastic kinematics
# ----------------------------------------------------------------------
def _beam_configuration() -> tuple[float, float, float, float, float, float]:
    """The exact fixed-target two-body quantities of the pinned M8 p+p
    configuration: ``(e_beam, e_total, e_cm, two_q2, beta, gamma)``.

    ``e_beam`` is the beam-proton energy, ``e_total`` the lab total
    energy, ``e_cm`` the exact invariant mass ``sqrt(s)`` of the
    incoming pair, ``two_q2`` the elastic ``2 q^2`` of the CM momentum
    squared, and ``(beta, gamma)`` the CM boost along the beam axis.
    """

    m = PROTON_MASS
    p_beam = BEAM_MOMENTUM
    e_beam = math.sqrt(m * m + p_beam * p_beam)
    e_total = e_beam + m
    s = m * m + m * m + 2.0 * m * e_beam
    e_cm = math.sqrt(s)
    two_q2 = 0.5 * (s - 4.0 * m * m)
    beta = p_beam / e_total
    gamma = 1.0 / math.sqrt(1.0 - beta * beta)
    return e_beam, e_total, e_cm, two_q2, beta, gamma


def elastic_outgoing_from_t(t: float) -> tuple[tuple[float, float, float, float],
                                               tuple[float, float, float, float]]:
    """The exact outgoing four-momenta of the elastic pair at the
    momentum transfer *t* [GeV^2], in the lab frame.

    The scattering angle follows from ``t = -2 q^2 (1 - cos(theta*))
    `` in the center-of-mass frame of the two incoming protons; the
    pair is boosted into the lab frame with the fixed-target CM
    velocity.  The pair conserves the incoming four-momentum sum by
    construction and both protons sit on the ``m_p`` mass shell.  A *t*
    beyond the exact two-body limit ``-2 q^2`` (possible only at the
    nominal range edge; see the module docstring) raises ``ValueError``
    naming the t and the limit: checked, never repaired.
    """

    e_beam, _e_total, e_cm, two_q2, beta, gamma = _beam_configuration()
    cos_theta = 1.0 + t / two_q2
    if not -1.0 <= cos_theta <= 1.0:
        raise ValueError(
            "elastic_outgoing_from_t: t = {!r} GeV^2 lies outside the "
            "exact two-body elastic limit -(2 q^2) = {!r} GeV^2 of the "
            "pinned beam configuration".format(t, -two_q2)
        )
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    e_star = 0.5 * e_cm
    q = math.sqrt(0.5 * two_q2)
    # CM pair: the beam-side proton scattered by theta*, the target-side
    # proton back-to-back against it.
    pz_star = q * cos_theta
    p_perp = q * sin_theta
    lab_e = gamma * (e_star + beta * pz_star)
    lab_pz = gamma * (pz_star + beta * e_star)
    outgoing_beam = (lab_e, p_perp, 0.0, lab_pz)
    outgoing_target = (2.0 * gamma * e_star - lab_e,
                       -p_perp, 0.0,
                       2.0 * gamma * beta * e_star - lab_pz)
    return outgoing_beam, outgoing_target


def generate_elastic(rng: np.random.Generator) -> ElasticEvent:
    """One elastic event: both lines intact protons, the single vacuum
    exchange with zero transfer, and the exact outgoing four-momenta
    solved from the t draw.

    Consumes exactly one uniform (the t draw) on the
    ``exchange_kinematics`` substream, zero elsewhere.  The record
    carries the per-line incoming/outgoing quantum numbers in the
    doubled-I3 integer convention and the exchange record with transfer
    ``(0, 0, 0, 0)`` (the card 2 ledger fields).
    """

    t = draw_elastic_t(rng)
    e_beam, _e_total, _e_cm, _two_q2, _beta, _gamma = _beam_configuration()
    out_beam, out_target = elastic_outgoing_from_t(t)
    proton_qn = PROTON_QN
    return ElasticEvent(
        class_label=ELASTIC_CLASS_LABEL,
        incoming=(
            LineState(four_momentum=(e_beam, 0.0, 0.0, BEAM_MOMENTUM), qn=proton_qn),
            LineState(four_momentum=(PROTON_MASS, 0.0, 0.0, 0.0), qn=proton_qn),
        ),
        outgoing=(
            LineState(four_momentum=out_beam, qn=proton_qn),
            LineState(four_momentum=out_target, qn=proton_qn),
        ),
        exchange=ExchangeRecord(name=VACUUM_EXCHANGE_NAME, transfer=ZERO_TRANSFER),
        t=t,
    )


# ----------------------------------------------------------------------
# The zero-transfer check
# ----------------------------------------------------------------------
def check_zero_transfer(event: ElasticEvent) -> None:
    """Exact integer check of the elastic ledger: every outgoing line
    qn equals its incoming line qn and the exchange transfer is
    ``(0, 0, 0, 0)``.

    Raises :class:`ElasticTransferViolation` naming the offending line
    (or the exchange) on the first violation; a passing event returns
    ``None``.  The comparison is exact integer arithmetic on the
    doubled-I3 ``(B, Q, S, 2I3)`` tuples.
    """

    if not isinstance(event, ElasticEvent):
        raise ValueError(
            f"check_zero_transfer: expected an ElasticEvent, got {type(event)!r}"
        )
    for index, (in_line, out_line) in enumerate(zip(event.incoming, event.outgoing)):
        if out_line.qn != in_line.qn:
            raise ElasticTransferViolation(
                f"elastic event line {index}: outgoing qn {tuple(out_line.qn)} "
                f"differs from incoming qn {tuple(in_line.qn)} "
                f"(transfer {tuple(o - i for o, i in zip(out_line.qn, in_line.qn))})"
            )
    if event.exchange.transfer != ZERO_TRANSFER:
        raise ElasticTransferViolation(
            f"elastic event exchange {event.exchange.name!r} carries the "
            f"nonzero transfer {tuple(event.exchange.transfer)}"
        )
