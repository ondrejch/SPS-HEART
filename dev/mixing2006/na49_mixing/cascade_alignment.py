# Not used by the HEART p+p fits (marked 2026-11-04): no production generation, fit, probe tool or paper script
# runs this module (an execution trace of the R137 generation and an import scan of the engine, the probe tools
# and the paper scripts). Code reviews of the p+p model can skip it. It stays until the cleanup and refactor after
# the p+A run.
"""Baryonic and mesonic alignment kernels of the aligned resonance
cascade.

Round 4, card 6 (``TASK-20260920-r4-baryonic-alignment-kernel``; plan
section 22 task 23, plan section 10.4, lines 716-732) landed the
baryonic surface below; card 7 (``TASK-20260920-r4-mesonic-alignment-
kernel``; plan section 22 task 24) extended the module additively with
the mesonic even law (the section at the end of this file).  The
baryonic part carries the axis-dependent density ``dP/dcos(theta)
proportional to exp[kappa_B cos(theta)]`` of plan section 10.4 (line
721) for a baryonic daughter that continues the incoming baryonic
line, its analytic normalization, the closed-form inverse-CDF draw
with the isotropic branch, and the vertex-level interface card 4's
``draw_decay`` hook consumes at the ``W_align`` attachment point.
Card 5's axis machinery is consumed read-only (it supplies the
inherited axis the laws are evaluated against); card 10 packages both
kernels for the R43 ablation; cards 11-15 exercise the kernels
statistically.

The angular law (plan section 10.4, lines 718-722)
--------------------------------------------------

For a baryonic daughter that continues the incoming baryonic line, the
initial minimal form of plan section 10.4 (line 718) is

    dP/dcos(theta) proportional to exp[kappa_B cos(theta)],

evaluated in the parent rest frame of the vertex against the parent
state's inherited production axis (the plan section 10.1 axis field,
filled by card 5's inheritance rules): ``cos(theta)`` is the cosine of
the angle between the baryonic daughter's emission direction and the
inherited axis.  The law carries an odd part in ``cos(theta)``: at
``kappa_B > 0`` the baryonic daughter is biased forward along the
inherited axis, with the analytic mean ``cos(theta) = coth(kappa_B) -
1/kappa_B`` (plan section 10.4's longitudinal memory of the baryonic
line).  The axis-even counterpart (``exp[kappa_M cos(theta)^2]`` for
symmetric mesonic decays, plan section 10.4 line 724) is card 7's
territory; at this card's landing a daughter off the baryonic line
keeps the Round 2 isotropic draw unchanged.

Interpretation (plan section 10.4, lines 730-732)
-------------------------------------------------

``kappa`` is interpreted as effective spin alignment and memory of the
production axis, never as a fragmentation function (plan section 10.4
line 730); no fragmentation-style mass or energy dependence is
introduced.  The constant-kappa simplification holds for Round 4:
``kappa_B`` is a constant of order one (the pre-registered default
:data:`BARYONIC_KAPPA_DEFAULT = 1.0`, the R40 nominal point of round 4
index item 3), and the mass dependence ``kappa_B(M)`` of plan line 721
stays reduced to the constant ``kappa_B`` until the R40 fixed-source
scan and the R44 validation data require it (plan line 732; the
pre-registered simplification path of the round 4 index gate mapping).

Draw-budget attachment (card 4's frozen 1 + 2 budget)
-----------------------------------------------------

Card 4 freezes the per-vertex budget at exactly one draw on the
``decay_channel`` substream (the channel) and exactly two draws on the
``decay_kinematics`` substream (the Round 2 two-body kinematics: the
cos(theta) draw first, the azimuth draw second).  The baryonic
attachment replaces the isotropic cos(theta) draw inside that budget:
:data:`draw_baryonic_alignment` and :data:`draw_vertex_cos_theta` are
pure functions of the already-drawn kinematics quantile and of the
daughter identities, so the attachment consumes no additional draw and
the azimuth stays the unchanged Round 2 second draw about the inherited
axis.  No new substream and no reordering: the substream derivation
order is untouched, and the per-vertex budget stays 1 channel + 2
kinematics through cards 6-8 exactly as card 4 froze it.

At a vertex the shared direction is the leading daughter's (``+``
direction, the card-4 daughter order); the trailing daughter is emitted
back to back.  When the baryon-carrying daughter is the leading one,
its emission angle is the kernel draw at the quantile ``u``.  When the
baryon-carrying daughter is the trailing one and the kernel is active,
its emission angle is the kernel draw at the same quantile ``u`` and
the leading direction carries its back-to-back mirror (the mirror rule
of :data:`draw_vertex_cos_theta`), so the baryonic daughter's angle
follows the plan law in both daughter orders with the single replaced
draw.  At the isotropic branch the mirror is dropped: every daughter
order draws the plain isotropic value, so the baseline reproduction
below is vertex-uniform.

The isotropic limit and the ablation baseline
---------------------------------------------

For ``|kappa| < :data:`ISOTROPIC_BRANCH_TOL` = 1e-8`` the draw takes
the branch ``cos(theta) = 2u - 1``, bit-stable with card 4's isotropic
draw (the same float expression on the same quantile): the aligned
cascade at the pre-registered isotropic point ``kappa_B = 0`` (round 4
index item 3) reproduces the card-4 isotropic baseline vertex by
vertex, in every daughter order, which is the R43 ablation baseline
contract (card 14, acceptance (ii); the runner is card 10).  The
alignment label moves with the law: :func:`assign_alignment_label`
marks a baryon-carrying daughter :data:`BARYONIC_ALIGNMENT_LABEL`
while the kernel is active and falls back to card 4's
``ISOTROPIC_ALIGNMENT_LABEL`` placeholder at the isotropic branch, and
for every daughter off the baryonic line (card 7's territory), so the
label bookkeeping and the drawn law agree at every vertex.

Conservation note: the committed catalog carries baryon numbers in
{-1, 0, 1} and the declared continuum families are the baryonic (1, 1,
0), strange baryonic (1, 0, -1), and mesonic (0, 0, 0) records.  At a
baryonic vertex exactly one daughter carries baryon number.  The Round
6 baryon-pair classes of the mesonic families (plan section 12.4,
2026-09-28) add the baryon-antibaryon vertex with net baryon number
zero: no baryonic line passes through it, and
:func:`draw_vertex_cos_theta` dispatches it like the symmetric mesonic
vertex (:func:`assign_alignment_label` takes ``pair_vertex=True`` for
its daughters).  A two-baryon pair of one sign stays rejected as
checked-never-repaired (the plan law names one baryonic line).

Hermetic: the standard library, numpy, and the card-4 alignment-label
constant only; no event generation, no external tables, no Monte Carlo
run (the acceptance is hermetic: closed-form identities, CDF-table
identities, and pinned quantile grids; the runs R40-R44 exercise the
kernels statistically).
"""

from __future__ import annotations

import math

import numpy as np

from .cascade_decay import ISOTROPIC_ALIGNMENT_LABEL

__all__ = [
    "BARYONIC_KAPPA_DEFAULT",
    "ISOTROPIC_BRANCH_TOL",
    "BARYONIC_ALIGNMENT_LABEL",
    "baryonic_kernel_density",
    "baryonic_normalization",
    "draw_baryonic_alignment",
    "continues_baryon_line",
    "draw_vertex_cos_theta",
    "assign_alignment_label",
    # card 7 (TASK-20260920-r4-mesonic-alignment-kernel): the mesonic
    # even law of plan section 10.4, line 724, added below this list
    "MESONIC_KAPPA_DEFAULT",
    "MESONIC_CDF_NODES",
    "MESONIC_ALIGNMENT_LABEL",
    "mesonic_kernel_density",
    "mesonic_normalization",
    "mesonic_cdf_table",
    "draw_mesonic_alignment",
]


#: The pre-registered constant baryonic alignment strength (plan
#: section 10.4 line 732: start with constant kappa_B of order one;
#: round 4 index item 3: the R40 nominal point kappa_B = 1).  No mass
#: dependence in Round 4: the kappa_B(M) of plan line 721 stays the
#: constant kappa_B until the R40 scan and the R44 validation data
#: require it.
BARYONIC_KAPPA_DEFAULT: float = 1.0

#: The |kappa| below which the draw takes the bit-stable isotropic
#: branch ``cos(theta) = 2u - 1`` (the card's pre-registered branch
#: tolerance): the aligned cascade at kappa_B = 0 reproduces the
#: card-4 isotropic baseline bit for bit (the R43 ablation baseline,
#: card 14 acceptance (ii)).
ISOTROPIC_BRANCH_TOL: float = 1e-8

#: The spin-density / alignment label of a baryon-carrying daughter
#: under the active baryonic kernel (the plan section 10.1 line 673
#: label): the card-4 ``ISOTROPIC_ALIGNMENT_LABEL`` placeholder is
#: replaced by this label on the baryonic line while the kernel is
#: active, and :func:`assign_alignment_label` falls back to the
#: placeholder at the isotropic branch and off the baryonic line
#: (card 7's even-law territory).
BARYONIC_ALIGNMENT_LABEL: str = "baryonic_alignment"


# ----------------------------------------------------------------------
# Validation helpers (checked, never repaired)
# ----------------------------------------------------------------------
def _validate_u(u) -> float:
    """The float-cast kinematics quantile *u*; finite with
    ``0.0 <= u <= 1.0``, else ``ValueError``."""

    value = float(u)
    if not (math.isfinite(value) and 0.0 <= value <= 1.0):
        raise ValueError(
            f"baryonic alignment kernel: the kinematics quantile "
            f"u = {u!r} must be finite in [0, 1]"
        )
    return value


def _validate_kappa(kappa) -> float:
    """The float-cast alignment strength *kappa*; finite, else
    ``ValueError`` (of order one by the plan simplification; the R40
    scan grid caps at 4)."""

    value = float(kappa)
    if not math.isfinite(value):
        raise ValueError(
            f"baryonic alignment kernel: kappa = {kappa!r} must be "
            "finite (effective spin alignment, plan section 10.4 line "
            "730)"
        )
    return value


def _validate_qn(daughter_qn, function: str) -> tuple[int, int, int]:
    """The integer ``(B, Q, S)`` triple *daughter_qn*; three values,
    else ``ValueError`` (the card-5 ``inherit_axis`` convention)."""

    qn = tuple(int(value) for value in daughter_qn)
    if len(qn) != 3:
        raise ValueError(
            f"{function}: daughter_qn {daughter_qn!r} must carry the "
            "three integer quantum numbers (B, Q, S)"
        )
    return qn


# ----------------------------------------------------------------------
# The baryonic kernel (plan section 10.4, lines 718-722)
# ----------------------------------------------------------------------
def baryonic_kernel_density(c: float, kappa: float) -> float:
    """exp(kappa * c) on c in [-1, 1] (unnormalized).

    The baryonic angular law of plan section 10.4 (line 721) at the
    cosine *c* against the inherited axis: forward for ``kappa > 0``
    (the daughter continues the incoming baryonic line), isotropic at
    the branch tolerance, backward-symmetric for ``kappa < 0``.  The
    normalized density is this kernel over :func:`baryonic_normalization`.
    *c* finite in ``[-1, 1]`` and *kappa* finite, else ``ValueError``
    (checked, never repaired).
    """

    c_value = float(c)
    if not (math.isfinite(c_value) and -1.0 <= c_value <= 1.0):
        raise ValueError(
            f"baryonic alignment kernel: cos(theta) = {c!r} must be a "
            "finite cosine in [-1, 1]"
        )
    kappa_value = _validate_kappa(kappa)
    return math.exp(kappa_value * c_value)


def baryonic_normalization(kappa: float) -> float:
    """int_{-1}^{1} exp(kappa c) dc = 2 sinh(kappa) / kappa, with the
    kappa -> 0 limit 2.

    The exact normalization of :func:`baryonic_kernel_density` over the
    kinematic cosine range.  For ``|kappa| <
    :data:`ISOTROPIC_BRANCH_TOL`` the limit value 2.0 is returned (the
    isotropic kernel's normalization; continuous across the branch at
    the 1e-16 level).  *kappa* finite, else ``ValueError``.
    """

    kappa_value = _validate_kappa(kappa)
    if abs(kappa_value) < ISOTROPIC_BRANCH_TOL:
        return 2.0
    return 2.0 * math.sinh(kappa_value) / kappa_value


def draw_baryonic_alignment(u: float, kappa: float) -> float:
    """cos(theta) sample via the closed-form inverse CDF.

    The pre-registered closed form at the kinematics quantile *u*,

        c = (1/kappa) * ln(exp(-kappa) + u * (exp(kappa) - exp(-kappa))),

    whose CDF ``(exp(kappa c) - exp(-kappa)) / (exp(kappa) - exp(-kappa))``
    maps *c* back to *u* (the inverse-CDF fidelity the card pins at
    1e-9 on the pinned quantile grid).  For ``|kappa| <
    :data:`ISOTROPIC_BRANCH_TOL`` the isotropic branch ``c = 2u - 1``
    is returned, bit-stable with card 4's isotropic draw (the same
    float expression on the same quantile; the R43 ablation baseline).
    The function is pure: it consumes the already-drawn quantile and
    nothing else, so the card-4 per-vertex budget is unchanged.  *u*
    finite in ``[0, 1]`` and *kappa* finite, else ``ValueError``.
    """

    u_value = _validate_u(u)
    kappa_value = _validate_kappa(kappa)
    if abs(kappa_value) < ISOTROPIC_BRANCH_TOL:
        return 2.0 * u_value - 1.0
    return (1.0 / kappa_value) * math.log(
        math.exp(-kappa_value)
        + u_value * (math.exp(kappa_value) - math.exp(-kappa_value))
    )


# ----------------------------------------------------------------------
# The vertex-level interface card 4's draw_decay hook consumes
# ----------------------------------------------------------------------
def continues_baryon_line(daughter_qn) -> bool:
    """True when the daughter carries baryon number (``B != 0``).

    The baryon-carrying daughter continues the incoming baryonic line
    (plan section 10.4 line 718) and takes the baryonic kernel; a
    daughter with ``B == 0`` stays on the Round 2 isotropic draw at
    this card's landing (the axis-even mesonic law is card 7's
    territory).  *daughter_qn* must carry the three integer quantum
    numbers ``(B, Q, S)``, else ``ValueError``.
    """

    qn = _validate_qn(daughter_qn, "continues_baryon_line")
    return qn[0] != 0


def draw_vertex_cos_theta(
    u: float,
    daughter_qn_pair,
    kappa_baryonic: float = BARYONIC_KAPPA_DEFAULT,
    kappa_mesonic: float = 0.0,
) -> float:
    """The cos(theta) draw of one card-4 decay vertex at the
    ``W_align`` attachment point.

    *daughter_qn_pair* is the ``(qn_leading, qn_trailing)`` pair of the
    drawn channel's daughters in the card-4 daughter order (the leading
    daughter carries the ``+`` direction of the two-body split; the
    trailing daughter is emitted back to back).  Exactly one uniform is
    consumed for the cos(theta) draw — the kernel's, at the
    already-drawn quantile *u* — in place of the Round 2 isotropic
    draw; the azimuth draw and the per-vertex budget of card 4 are
    unchanged.

    - baryon-carrying leading daughter: the kernel draw
      ``draw_baryonic_alignment(u, kappa_baryonic)`` (its emission
      angle follows the plan law directly);
    - baryon-carrying trailing daughter while the kernel is active:
      the kernel draw at the same quantile *u* for the baryonic
      daughter's angle, mirrored back-to-back onto the leading
      direction (``-`` the kernel draw): the baryonic daughter
      continues the incoming baryonic line in both daughter orders;
    - two mesonic daughters (both ``B == 0``, the symmetric mesonic
      vertex) while the mesonic kernel is active
      (``|kappa_mesonic| >= :data:`ISOTROPIC_BRANCH_TOL``): the even
      mesonic law draw ``draw_mesonic_alignment(u, kappa_mesonic)``
      (card 7; the even law of plan section 10.4 line 724 is symmetric
      in the emission hemisphere sign, so no daughter-order mirror
      applies and both orders draw the same value);
    - no baryon-carrying daughter at the mesonic isotropic branch
      (``|kappa_mesonic| < :data:`ISOTROPIC_BRANCH_TOL``), or the
      baryonic isotropic branch (``|kappa_baryonic| <
      :data:`ISOTROPIC_BRANCH_TOL``, mirror dropped): the Round 2
      isotropic draw ``2u - 1``, bit-stable;
    - a baryon-antibaryon pair (both daughters carry baryon number,
      summing to zero: the Round 6 baryon-pair channels of a mesonic
      parent, plan section 12.4): no baryonic line passes through the
      vertex, so the pair follows the symmetric mesonic dispatch (the
      even law while ``|kappa_mesonic| >= ISOTROPIC_BRANCH_TOL``, else
      ``2u - 1``, in either daughter order);
    - two baryon-carrying daughters of one sign: ``ValueError`` (the
      plan law names one baryonic line; unreachable through the
      declared families by ``(B, Q, S)`` conservation; checked, never
      repaired).

    A mixed baryon-meson vertex keeps the baryonic dispatch exactly
    (the baryonic daughter's law governs the shared emission axis; the
    card-7 mesonic law applies only at the symmetric mesonic vertex).
    The vertex-level default ``kappa_mesonic = 0.0`` keeps the card-6
    interface inert at the defaults (the card-6 baseline draws
    unchanged); the R40-R44 drivers pass the pre-registered
    :data:`MESONIC_KAPPA_DEFAULT`.

    At the pre-registered isotropic point ``kappa_baryonic = 0`` every
    baryonic branch and every daughter order returns ``2u - 1``
    bit-stably, and with the vertex default ``kappa_mesonic = 0.0`` so
    does the symmetric mesonic vertex: the aligned cascade at the
    isotropic point reproduces the card-4 isotropic baseline vertex by
    vertex (the R43 ablation baseline, card 14 acceptance (ii)).
    *u* finite in ``[0, 1]``, *kappa_baryonic* finite, *kappa_mesonic*
    finite, and both ``daughter_qn`` triples well-formed, else
    ``ValueError``.
    """

    qn_leading = _validate_qn(
        daughter_qn_pair[0], "draw_vertex_cos_theta")
    qn_trailing = _validate_qn(
        daughter_qn_pair[1], "draw_vertex_cos_theta")
    u_value = _validate_u(u)
    kappa_value = _validate_kappa(kappa_baryonic)
    kappa_mes_value = _validate_kappa(kappa_mesonic)
    lead_baryon = qn_leading[0] != 0
    trail_baryon = qn_trailing[0] != 0
    if lead_baryon and trail_baryon:
        if qn_leading[0] + qn_trailing[0] != 0:
            raise ValueError(
                "draw_vertex_cos_theta: both daughters carry baryon "
                f"number of one sign ({tuple(qn_leading)}, "
                f"{tuple(qn_trailing)}); the plan section 10.4 law names "
                "one baryonic line per vertex"
            )
        # The baryon-antibaryon pair of a mesonic parent (plan section
        # 12.4, the Round 6 baryon-pair channels, 2026-09-28): net
        # baryon number zero, so no baryonic line passes through the
        # vertex; the pair is a symmetric vertex of a mesonic source and
        # follows the even mesonic law while that kernel is active, else
        # the isotropic branch.  The law is even, so no daughter-order
        # mirror applies.
        if abs(kappa_mes_value) >= ISOTROPIC_BRANCH_TOL:
            return draw_mesonic_alignment(u_value, kappa_mes_value)
        return 2.0 * u_value - 1.0
    if lead_baryon:
        return draw_baryonic_alignment(u_value, kappa_value)
    if trail_baryon and abs(kappa_value) >= ISOTROPIC_BRANCH_TOL:
        return -draw_baryonic_alignment(u_value, kappa_value)
    if (qn_leading[0] == 0 and qn_trailing[0] == 0
            and abs(kappa_mes_value) >= ISOTROPIC_BRANCH_TOL):
        return draw_mesonic_alignment(u_value, kappa_mes_value)
    return 2.0 * u_value - 1.0


def assign_alignment_label(
    daughter_qn,
    kappa_baryonic: float = BARYONIC_KAPPA_DEFAULT,
    kappa_mesonic: float = 0.0,
    *,
    pair_vertex: bool = False,
) -> str:
    """The plan section 10.1 line 673 alignment label of one daughter.

    The baryon-carrying daughter carries
    :data:`BARYONIC_ALIGNMENT_LABEL` while the baryonic kernel is
    active (``|kappa_baryonic| >= :data:`ISOTROPIC_BRANCH_TOL``): the
    card-4 ``ISOTROPIC_ALIGNMENT_LABEL`` placeholder is replaced by the
    kernel's label on the baryonic line.  The card-7 mesonic daughter
    (``B == 0``) carries :data:`MESONIC_ALIGNMENT_LABEL` while the
    mesonic kernel is active (``|kappa_mesonic| >=
    :data:`ISOTROPIC_BRANCH_TOL``): the even law of plan section 10.4
    line 724 replaces the placeholder on the symmetric mesonic vertex.

    Vertex scope (the call-site convention that keeps the label
    bookkeeping and the drawn law agreeing at every vertex): the
    governing kernel of a drawn vertex is the baryonic one at any
    vertex with a baryon-carrying daughter (the plan law names one
    baryonic line; the shared emission axis follows the baryonic
    dispatch regardless of ``kappa_mesonic``) and the mesonic one at
    the symmetric mesonic vertex (both daughters ``B == 0``).  The
    vertex-exact call passes the governing kernel's kappa and holds
    the other at the isotropic branch (0.0): the baryonic line then
    labels its daughter :data:`BARYONIC_ALIGNMENT_LABEL` with the
    mesonic daughter on the placeholder, and the symmetric mesonic
    vertex labels both daughters :data:`MESONIC_ALIGNMENT_LABEL`.  At
    either isotropic branch the card-4 placeholder stands (the R43
    ablation baseline).  *daughter_qn* must carry the three integer
    quantum numbers ``(B, Q, S)`` and both kappa values must be
    finite, else ``ValueError``.  Pure: no draw is consumed.

    ``pair_vertex`` (keyword only, default ``False``) marks a daughter
    of a baryon-antibaryon pair vertex (the Round 6 baryon-pair
    channels of a mesonic parent, plan section 12.4): the pair carries
    no baryonic line, the mesonic kernel governs the vertex exactly as
    in :func:`draw_vertex_cos_theta`, and the daughter is labeled
    :data:`MESONIC_ALIGNMENT_LABEL` while ``|kappa_mesonic| >=
    ISOTROPIC_BRANCH_TOL`` and the placeholder otherwise, whatever its
    own baryon number.
    """

    qn = _validate_qn(daughter_qn, "assign_alignment_label")
    kappa_value = _validate_kappa(kappa_baryonic)
    kappa_mes_value = _validate_kappa(kappa_mesonic)
    if pair_vertex:
        if abs(kappa_mes_value) >= ISOTROPIC_BRANCH_TOL:
            return MESONIC_ALIGNMENT_LABEL
        return ISOTROPIC_ALIGNMENT_LABEL
    if qn[0] != 0 and abs(kappa_value) >= ISOTROPIC_BRANCH_TOL:
        return BARYONIC_ALIGNMENT_LABEL
    if qn[0] == 0 and abs(kappa_mes_value) >= ISOTROPIC_BRANCH_TOL:
        return MESONIC_ALIGNMENT_LABEL
    return ISOTROPIC_ALIGNMENT_LABEL


# ----------------------------------------------------------------------
# The mesonic kernel (plan section 10.4, lines 724-728) - card 7
# ----------------------------------------------------------------------
# Card 7 (TASK-20260920-r4-mesonic-alignment-kernel; plan section 22
# task 24) extends this module additively with the even axis-dependent
# angular law of plan section 10.4 (line 724) for a symmetric mesonic
# decay,
#
#     dP/dcos(theta) proportional to exp[kappa_M cos(theta)^2],
#
# its analytic normalization, the pre-registered 10,001-node CDF-table
# sampler with linear interpolation (the even law has no closed-form
# inverse CDF), and the mesonic branch of the vertex-level interface
# above: :func:`draw_vertex_cos_theta` dispatches the baryonic line
# (byte-stable card-6 behavior) against the symmetric mesonic vertex,
# and :func:`assign_alignment_label` extends the label bookkeeping
# (the vertex-exact call passes the governing kernel of the drawn
# vertex and holds the other kappa at the isotropic branch; see the
# label function's docstring).
# ``kappa_M`` is interpreted exactly like ``kappa_B`` (effective spin
# alignment and memory of the production axis, never a fragmentation
# function; plan section 10.4 line 730): a constant of order one,
# :data:`MESONIC_KAPPA_DEFAULT` = 1.0, with no mass dependence in
# Round 4 (plan line 732).  ``kappa_M > 0`` enhances the decay along
# the inherited axis, ``kappa_M < 0`` toward the transverse plane; the
# law is even in cos(theta), so a symmetric mesonic vertex needs no
# back-to-back mirror and no daughter-order distinction.
#
# The pre-registered vertex-level default of ``kappa_mesonic`` is 0.0
# (the isotropic branch) so the card-6 interface stays inert at the
# defaults; the R40-R44 drivers pass :data:`MESONIC_KAPPA_DEFAULT`.
# At ``kappa_M = 0`` the draw is the bit-stable isotropic branch
# ``2u - 1``, the card-4 baseline the R43 ablation (card 14) stands on.

#: The pre-registered constant mesonic alignment strength (plan
#: section 10.4 line 732: start with constant kappa_M of order one;
#: round 4 index item 3: the R40 nominal point kappa_M = 1).  No mass
#: dependence in Round 4: the kappa_M(M) of plan line 724 stays the
#: constant kappa_M until the R40 scan and the R44 validation data
#: require it.
MESONIC_KAPPA_DEFAULT: float = 1.0

#: The node count of the pre-registered CDF table on [-1, 1] (the
#: uniform cosine grid ``c_i = -1 + 2 i / (MESONIC_CDF_NODES - 1)`` of
#: the linear-interpolation sampler): the even law has no closed-form
#: inverse CDF, and the table is re-evaluated per kappa_M value on
#: demand (the R40 scan grid kappa_M in {0, 1, 2, 4} plus the R41 fit
#: space) and cached per value.
MESONIC_CDF_NODES: int = 10001

#: The spin-density / alignment label of a mesonic daughter (``B == 0``)
#: under the active mesonic kernel: the counterpart of
#: :data:`BARYONIC_ALIGNMENT_LABEL`; the card-4
#: ``ISOTROPIC_ALIGNMENT_LABEL`` placeholder stands at the isotropic
#: branch.
MESONIC_ALIGNMENT_LABEL: str = "mesonic_alignment"

#: The fixed uniform cosine grid of the CDF table (the table's i-th
#: entry is the CDF at this grid's i-th node); read-only by convention.
_MESONIC_COS_GRID: np.ndarray = np.linspace(-1.0, 1.0, MESONIC_CDF_NODES)
_MESONIC_COS_GRID.setflags(write=False)

#: Pure compute cache of the constructed CDF tables per kappa value
#: (the same pattern as cascade_decay's kernel shims: the construction
#: is deterministic, so a repeated kappa value serves the identical
#: table; no draw is consumed by construction or lookup).
_MESONIC_CDF_TABLES: dict = {}


def _mesonic_erfi(x: float) -> float:
    """The imaginary error function ``erfi(x) = (2/sqrt(pi)) int_0^x
    exp(t^2) dt``.

    The standard library carries ``erf`` but no ``erfi``, so the value
    comes from the everywhere-convergent power series ``erfi(x) =
    (2/sqrt(pi)) sum_n x^(2n+1) / (n! (2n + 1))`` (every term positive,
    no cancellation), summed exactly with :func:`math.fsum` until the
    term drops below the running total times 1e-18.  The realistic
    domain is ``sqrt(kappa)`` for kappa of order one (the R40 scan grid
    caps at ``sqrt(4) = 2``, where the series reaches double precision
    in about 60 terms); a kappa so large that the series fails to
    converge within 2000 terms raises ``ValueError`` (checked, never
    repaired).
    """

    x_value = float(x)
    if x_value < 0.0:
        return -_mesonic_erfi(-x_value)
    if x_value == 0.0:
        return 0.0
    x2 = x_value * x_value
    terms = [x_value]
    running = x_value
    n = 1
    while True:
        term = terms[-1] * x2 * (2.0 * n - 1.0) / (n * (2.0 * n + 1.0))
        terms.append(term)
        running += term
        if term <= running * 1e-18:
            break
        if n >= 2000:
            raise ValueError(
                f"mesonic alignment kernel: erfi({x!r}) did not "
                "converge within the series cap (kappa far outside "
                "the pre-registered order-one range); checked, never "
                "repaired"
            )
        n += 1
    return 2.0 / math.sqrt(math.pi) * math.fsum(terms)


def _validate_mesonic_u(u) -> float:
    """The float-cast kinematics quantile *u*; finite with
    ``0.0 <= u <= 1.0``, else ``ValueError`` (the mesonic wording of
    ``_validate_u``)."""

    value = float(u)
    if not (math.isfinite(value) and 0.0 <= value <= 1.0):
        raise ValueError(
            f"mesonic alignment kernel: the kinematics quantile "
            f"u = {u!r} must be finite in [0, 1]"
        )
    return value


def _validate_mesonic_kappa(kappa) -> float:
    """The float-cast mesonic alignment strength *kappa*; finite, else
    ``ValueError`` (of order one by the plan simplification; the R40
    scan grid caps at 4)."""

    value = float(kappa)
    if not math.isfinite(value):
        raise ValueError(
            f"mesonic alignment kernel: kappa = {kappa!r} must be "
            "finite (effective spin alignment, plan section 10.4 line "
            "730)"
        )
    return value


def mesonic_kernel_density(c: float, kappa: float) -> float:
    """exp(kappa * c**2) on c in [-1, 1] (unnormalized).

    The even angular law of plan section 10.4 (line 724) at the cosine
    *c* against the inherited axis, for a symmetric mesonic decay: the
    density is even in c (no forward/backward asymmetry, no
    daughter-order mirror), enhanced along the axis for
    ``kappa > 0`` and toward the transverse plane for ``kappa < 0``.
    The normalized density is this kernel over
    :func:`mesonic_normalization`.  *c* finite in ``[-1, 1]`` and
    *kappa* finite, else ``ValueError`` (checked, never repaired).
    """

    c_value = float(c)
    if not (math.isfinite(c_value) and -1.0 <= c_value <= 1.0):
        raise ValueError(
            f"mesonic alignment kernel: cos(theta) = {c!r} must be a "
            "finite cosine in [-1, 1]"
        )
    kappa_value = _validate_mesonic_kappa(kappa)
    return math.exp(kappa_value * c_value * c_value)


def mesonic_normalization(kappa: float) -> float:
    """int_{-1}^{1} exp(kappa c^2) dc, with the kappa -> 0 limit 2.

    The exact normalization of :func:`mesonic_kernel_density` over the
    kinematic cosine range: ``sqrt(pi/kappa) * erfi(sqrt(kappa))`` for
    ``kappa > 0`` (the real-axis form; the card's sketch names it as
    the equivalent of its primary formula) and
    ``sqrt(pi/|kappa|) * erf(sqrt(|kappa|))`` for ``kappa < 0``, each
    exact, with the limit value 2.0 inside the branch tolerance
    (continuous; quadrature cross-checked by the card test).  *kappa*
    finite, else ``ValueError``.
    """

    kappa_value = _validate_mesonic_kappa(kappa)
    if abs(kappa_value) < ISOTROPIC_BRANCH_TOL:
        return 2.0
    if kappa_value > 0.0:
        return (math.sqrt(math.pi / kappa_value)
                * _mesonic_erfi(math.sqrt(kappa_value)))
    return (math.sqrt(math.pi / -kappa_value)
            * math.erf(math.sqrt(-kappa_value)))


def mesonic_cdf_table(kappa: float) -> np.ndarray:
    """The 10,001-node cumulative distribution on [-1, 1], normalized
    to unit mass.

    The i-th entry is the normalized integral of
    :func:`mesonic_kernel_density` from -1 to the i-th node of the
    fixed uniform cosine grid ``c_i = -1 + 2 i / (MESONIC_CDF_NODES -
    1)`` (``table[0] = 0.0``, ``table[-1] = 1.0``), evaluated per node
    in closed form through the even law's antiderivative ratio
    ``F(c) = 0.5 * (1 + erfi(sqrt(kappa) c) / erfi(sqrt(kappa)))`` for
    ``kappa > 0`` (the erf analogue for ``kappa < 0``, the uniform
    table inside the branch tolerance), so the node values carry no
    quadrature error and the sampler's fidelity is the
    linear-interpolation bound alone.  The even symmetry ``F(-c) = 1 -
    F(c)`` is checked at construction within 1e-12 (checked, never
    repaired), and the table of a kappa value already constructed is
    served from the cache.  *kappa* finite, else ``ValueError``.
    """

    kappa_value = _validate_mesonic_kappa(kappa)
    cached = _MESONIC_CDF_TABLES.get(kappa_value)
    if cached is not None:
        return cached
    if abs(kappa_value) < ISOTROPIC_BRANCH_TOL:
        table = 0.5 * (_MESONIC_COS_GRID + 1.0)
    else:
        s = math.sqrt(abs(kappa_value))
        if kappa_value > 0.0:
            denom = _mesonic_erfi(s)
            values = [_mesonic_erfi(s * float(c)) / denom
                      for c in _MESONIC_COS_GRID]
        else:
            denom = math.erf(s)
            values = [math.erf(s * float(c)) / denom
                      for c in _MESONIC_COS_GRID]
        table = 0.5 * (1.0 + np.array(values, dtype=float))
        table[0] = 0.0
        table[-1] = 1.0
    worst = float(np.max(np.abs(table + table[::-1] - 1.0)))
    if not worst <= 1e-12:
        raise ValueError(
            "mesonic alignment kernel: the constructed CDF table "
            f"violates the even symmetry F(-c) = 1 - F(c) by {worst!r} "
            "(beyond 1e-12); checked, never repaired"
        )
    table = np.ascontiguousarray(table, dtype=float)
    table.setflags(write=False)
    _MESONIC_CDF_TABLES[kappa_value] = table
    return table


def draw_mesonic_alignment(u: float, kappa: float) -> float:
    """cos(theta) sample by linear interpolation on the CDF table.

    The pre-registered sampler at the kinematics quantile *u*: the
    bracket of *u* is located on the cached :func:`mesonic_cdf_table`
    nodes and the cosine is the linear interpolation between the
    bracketing nodes (the even law has no closed-form inverse CDF).
    For ``|kappa| < :data:`ISOTROPIC_BRANCH_TOL`` the isotropic branch
    ``c = 2u - 1`` is returned, bit-stable with card 4's isotropic
    draw and card 6's baryonic isotropic branch (the same float
    expression on the same quantile; the R43 ablation baseline).  The
    function is pure: it consumes the already-drawn quantile and
    nothing else, so the card-4 per-vertex budget is unchanged.  *u*
    finite in ``[0, 1]`` and *kappa* finite, else ``ValueError``.
    """

    u_value = _validate_mesonic_u(u)
    kappa_value = _validate_mesonic_kappa(kappa)
    if abs(kappa_value) < ISOTROPIC_BRANCH_TOL:
        return 2.0 * u_value - 1.0
    table = mesonic_cdf_table(kappa_value)
    idx = int(np.searchsorted(table, u_value, side="left"))
    if idx <= 0:
        return -1.0
    if idx >= MESONIC_CDF_NODES:
        return 1.0
    f_lo = float(table[idx - 1])
    f_hi = float(table[idx])
    span = f_hi - f_lo
    if not span > 0.0:
        raise ValueError(
            "mesonic alignment kernel: the CDF table bracket "
            f"[{idx - 1}, {idx}] is degenerate (span {span!r}); "
            "checked, never repaired"
        )
    t = (u_value - f_lo) / span
    c_lo = float(_MESONIC_COS_GRID[idx - 1])
    c_hi = float(_MESONIC_COS_GRID[idx])
    return c_lo + t * (c_hi - c_lo)
