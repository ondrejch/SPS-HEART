"""The topology-specific mass laws (Round 5, card 4).

``TASK-20260925-r5-topology-mass-laws``; plan section 22 task 34
(line 1759); plan section 11.5 (lines 883-894).  The card implements
the event-class mass distributions of plan section 11.5:

> Use topology-specific mass distributions. Initial choices may include
> Regge-motivated forms such as
>
> dP_SD/dM^2dt proportional to (M^2)^(-1-epsilon) * exp[B_SD (M^2) t]
>
> with exact phase-space limits, and a separate joint law for double
> excitation and central production.
>
> At one beam energy, fit only identifiable effective combinations.

(The plan lines are quoted in this docstring because the rule they
state is the module's design contract.  The plan's closing sentence
names the shape-only parameters that enter only through one effective
shape; they stay off the public surface -- see "The effective
parameter surface" below.)

The pre-registered instantiation (the card)
-------------------------------------------

The module exposes exactly two effective parameters, the mass-law
exponent ``epsilon`` and the effective transfer slope ``b_sd``
(:class:`MassLawParams`; the R51 fit set carries one epsilon and one
B_SD, no strange-sector parameters, plan line 918).  The nominal point
is ``epsilon = 0.0`` and ``b_sd = 8.504140762083612 GeV^-2`` (the
M8-frozen ``regge.effective_transfer_slope``, the R50 nominal of the
round 5 index item 2; R51 fits exactly these two effective parameters,
index item 3).

The marginal single-diffraction law is sampled on the limits
``m2_min = (m_p + m_pi0)^2`` (the N pi threshold of the excited
``(B, Q, S) = (1, 1, 0)`` system, :data:`EXCITATION_M_MIN`) and
``m2_max = (sqrt(s) - m_remnant)^2`` (the other line at its remnant
mass) -- :func:`sd_limits`.  The lower edge was ``m_p`` until
2026-10-02 (plan line 842, no minimum N pi excitation); an excitation
below the threshold has no open channel and stayed an unidentified
final state, up to 57% of the excitations at the R68 to R70 exponent
(the feed-down and feed-over review).  The class fractions count
inelastic diffraction, which starts at the threshold.  The marginal density of the plan form at a fixed
transfer is ``(M^2)^(-1-epsilon) * exp[b_sd * t * M^2]``
(:func:`marginal_density`, unnormalized), and the draw normalizes it on
the window by the documented sub-tolerance grid
(:func:`marginal_grid_mass`, :func:`draw_marginal`).

The momentum transfer t is drawn on the closed ``exchange_kinematics``
substream label
(:data:`na49_mixing.rng_substreams.SUBSTREAM_LABELS`) from
``exp(B_SD t)`` on the exact two-body t range
``[t_min(M, m_remnant), 0]`` (:func:`t_range`, :func:`draw_t`), so the
t factor of the law acts on the exchange kinematics only: the t
dependence is realized through the exchange-kinematics t draw with the
identifiable constant effective slope, and the nominal M^2 marginal
draw evaluates the plan form at ``t = 0`` (the pure power law).  The
exact two-body range is closed at ``t = 0`` (the forward edge; at the
intact limit ``M = m_p`` the range is the elastic two-body limit
exactly).

The separate joint law covers double excitation and central production
(plan line 892).  Double excitation draws both ``M^2`` from the
marginals on the caller's windows and accepts on the exact closure
``M_1 + M_2 + M_central <= sqrt_s`` (:func:`joint_closure`, the 1e-12
float tolerance) by rejection with the pinned cap
:data:`REJECTION_CAP` (:class:`PhaseSpaceRejectionExceeded` names the
attempt count and the parameter point); the accepted draw then draws
``t_1`` and ``t_2`` from ``exp(B_SD t)`` on each exchange's exact
two-body range.  The joint draw computes each exchange's range with the
other line at the intact proton mass (the factorized effective law: the
two exchanges are independent draws of the same slope, and the exact
``t_1 + t_2`` coupling of the 2-to-2 system is not claimed -- the
identifiable-effective-combinations rule, plan line 894).  Central
production draws the central mass ``M_c^2`` from the same functional
form with the total transfer ``t_c = t_1 + t_2`` of the two exchanges
(:func:`draw_central`; the lower mass reference of the central law is
the catalog pion :data:`PI0_MASS`, the lightest neutral state).  One
slope ``b_sd`` serves all classes (a single identifiable effective
slope, plan section 13.1 block A).

The effective parameter surface (plan line 894; plan section 24, line
1866)
---------------------------------------------------------------------

Only ``epsilon`` and ``b_sd`` exist on the public surface
(:class:`MassLawParams` carries exactly these two fields).  The shape
parameters the plan line names -- the ones that would enter only
through one effective shape -- are absent, and no exchange slope enters
any daughter-level transverse dynamics (plan section 24, line 1866):
the t factor acts on the exchange kinematics only, and the daughter
transverse dynamics stays under the frozen Round 4 decay kernel (the
separation of exchange recoil from decay transverse dynamics is not
claimed; the R46 outcome stands).

The grid convention
-------------------

The M^2 draws normalize the density on the window with trapezoid cell
masses on a fixed geometric 4096-point grid (uniform in ln M^2) and
draw by inverse CDF with one
uniform for the cell and one uniform within the cell -- the documented
sub-tolerance grid convention of
:func:`na49_mixing.excitation_kinematics.draw_from_grid_density`, shared
with ``na49_mixing.breit_wigner.draw_truncated_bw`` (order-1e-3 grid
bias for smooth densities over the full single-diffraction range, far
below every statistical tolerance in use; the 5e-3 round-trip bound of
the acceptance suite is the grid-bias bound and is not a statistical
tolerance).  The t draw is the closed-form inverse CDF of
``exp(B_SD t)`` on ``[t_min, 0]`` -- no grid, exact to float rounding.

Draw budgets
------------

Exactly 2 uniforms per marginal or central ``M^2`` draw and 1 uniform
per t draw (the Round 2 mass-draw convention: the cell uniform plus the
within-cell uniform, mirroring the two-uniform histogram sampler pair),
and 6 uniforms per accepted joint double-excitation draw (two marginals
and two transfers) with the rejection cap on the closure.  Every draw
consumes its uniforms from the caller's generator only; the module
holds no state.

Hermetic: standard library and numpy only; no event generation driver,
no external tables, no Monte Carlo run (the acceptance is hermetic:
pinned traces, closed-form identities, draw budgets; the runs R50-R55
exercise the laws statistically).
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass

import numpy as np

from .excitation_kinematics import kallen
from .topology_elastic import BEAM_MOMENTUM as _BEAM_MOMENTUM

__all__ = [
    "PROTON_MASS",
    "PI0_MASS",
    "EXCITATION_M_MIN",
    "REJECTION_CAP",
    "PhaseSpaceRejectionExceeded",
    "MassLawParams",
    "sd_limits",
    "t_range",
    "marginal_density",
    "marginal_grid_mass",
    "draw_marginal",
    "draw_marginal_edge",
    "draw_t",
    "joint_closure",
    "draw_joint_dd",
    "draw_central",
]


#: The proton mass, the catalog value (the committed catalog record
#: pdg_id 2212, pole_mass) [GeV].
PROTON_MASS: float = 0.93827231

#: The pion mass, the catalog value (the committed catalog record
#: pdg_id 111, pole_mass 0.1349766 GeV; the card's interface carried
#: the 0.1349768 quote, which is resolved here to the committed
#: catalog and vendored PDG value) [GeV].  The central-production law's
#: lower-mass reference: the lightest neutral state of plan section
#: 11.3.
PI0_MASS: float = 0.1349766

#: The lower edge of the diffractive excitation mass laws [GeV]: the
#: N pi threshold ``m_p + m_pi0`` of the excited (1, 1, 0) system (the
#: catalog's Delta+ support edge; the same edge as the non-diffractive
#: excited-remnant window, ``topology_non_diffractive.
#: remnant_excitation_window``).  The single- and double-diffraction
#: draws and their reweighting normalizations share it (2026-10-02).
EXCITATION_M_MIN: float = PROTON_MASS + PI0_MASS

#: The pinned rejection cap of the joint closure rejection: the joint
#: double-excitation draw raises
#: :class:`PhaseSpaceRejectionExceeded` after this many consecutive
#: rejected attempts (the card's pinned value).
REJECTION_CAP: int = 1000

#: The exact-closure float tolerance of :func:`joint_closure` (the
#: card's pinned 1e-12).
_CLOSURE_TOLERANCE: float = 1e-12

#: The documented sub-tolerance normalization grid of the M^2 draws
#: (the card's pinned 4096-point grid; the draw convention shared with
#: ``na49_mixing.breit_wigner.draw_truncated_bw``).
_MARGINAL_GRID_POINTS: int = 4096

#: Slope magnitude below which the t draw falls back to the uniform
#: limit of ``exp(B_SD t)`` (the same threshold as
#: :func:`na49_mixing.excitation_kinematics.sample_cos_from_exp_t`).
_FLAT_SLOPE_EPS: float = 1e-14


class PhaseSpaceRejectionExceeded(RuntimeError):
    """Raised when the joint closure rejection exceeds
    :data:`REJECTION_CAP` consecutive attempts (the message names the
    attempt count and the parameter point: the effective parameters,
    the phase-space limits, and the last rejected mass pair)."""


@dataclass(frozen=True)
class MassLawParams:
    """The effective parameter surface of the mass laws (plan line
    894): the mass-law exponent and the effective transfer slope.

    ``epsilon``
        the mass-law exponent of the plan form ``(M^2)^(-1-epsilon)``
        (nominal 0.0, the flat-in-ln M^2 law).
    ``b_sd``
        the effective transfer slope [GeV^-2] (nominal
        8.504140762083612, the M8-frozen ``regge.effective_transfer_slope``;
        one slope serves all classes, plan section 13.1 block A).
    """

    epsilon: float
    b_sd: float


# ----------------------------------------------------------------------
# Exact phase-space limit helpers
# ----------------------------------------------------------------------
def sd_limits(m_remnant: float, sqrt_s: float) -> tuple[float, float]:
    """Exact marginal limits ``(m2_min, m2_max)`` in GeV^2.

    ``m2_min = EXCITATION_M_MIN^2`` (the N pi threshold; it was ``m_p^2``
    until 2026-10-02) and ``m2_max = (sqrt_s - m_remnant)^2`` (the other
    line at its remnant mass).
    ``sqrt_s`` must be a finite GeV value above the two-body threshold
    ``m_p + m_remnant``.
    """

    m_rem = float(m_remnant)
    root_s = float(sqrt_s)
    if not math.isfinite(m_rem) or m_rem <= 0.0:
        raise ValueError(
            f"sd_limits: m_remnant must be a positive finite GeV mass, "
            f"got {m_remnant!r}"
        )
    if not math.isfinite(root_s) or root_s <= m_rem + PROTON_MASS:
        raise ValueError(
            "sd_limits: sqrt_s must be a finite GeV value above the "
            f"two-body threshold {m_rem + PROTON_MASS}, got {sqrt_s!r}"
        )
    m2_min = EXCITATION_M_MIN * EXCITATION_M_MIN
    m2_max = (root_s - m_rem) * (root_s - m_rem)
    return (m2_min, m2_max)


def t_range(
    m2: float, m_remnant: float, sqrt_s: float
) -> tuple[float, float]:
    """Exact two-body t range ``[t_min, 0]`` in GeV^2 for the ``(M,
    m_remnant, sqrt_s)`` system, closed at ``t = 0``.

    The lower edge is the back-to-back two-body value
    ``t_min = m_p^2 + M^2 - 2 (E_p E_M + p* q)`` with the exact
    center-of-mass energies and momenta of the ``(m_p, m_remnant) ->
    (M, m_remnant)`` system at the invariant mass ``sqrt_s`` (the
    Kallen arithmetic shared with
    :func:`na49_mixing.excitation_kinematics.two_body_momentum`).  At
    the intact limit ``M = m_p`` the range is the elastic two-body
    limit ``4 m_p^2 - s`` exactly (within float rounding of the shared
    Kallen composition); the upper edge is exactly ``0.0`` for every
    argument (the card's closed-at-0 convention, exact at the intact
    limit).  The system must be open: ``M + m_remnant`` above
    ``sqrt_s`` beyond the closure tolerance raises ``ValueError``
    naming the values (checked, never repaired).
    """

    mass2 = float(m2)
    m_rem = float(m_remnant)
    root_s = float(sqrt_s)
    if not math.isfinite(mass2) or mass2 <= 0.0:
        raise ValueError(
            f"t_range: m2 must be a positive finite GeV^2 mass squared, "
            f"got {m2!r}"
        )
    if not math.isfinite(m_rem) or m_rem <= 0.0:
        raise ValueError(
            f"t_range: m_remnant must be a positive finite GeV mass, "
            f"got {m_remnant!r}"
        )
    if not math.isfinite(root_s):
        raise ValueError(
            f"t_range: sqrt_s must be finite, got {sqrt_s!r}"
        )
    s = root_s * root_s
    m2_rem = m_rem * m_rem
    m_p2 = PROTON_MASS * PROTON_MASS
    if mass2 + m2_rem > s + _CLOSURE_TOLERANCE:
        raise ValueError(
            "t_range: the two-body system is closed: "
            f"sqrt(m2) + m_remnant = {math.sqrt(mass2) + m_rem!r} GeV "
            f"exceeds sqrt_s = {root_s!r} GeV"
        )
    e_p = (s + m_p2 - m2_rem) / (2.0 * root_s)
    e_m = (s + mass2 - m2_rem) / (2.0 * root_s)
    p_star = math.sqrt(max(kallen(s, m_p2, m2_rem), 0.0)) / (2.0 * root_s)
    q = math.sqrt(max(kallen(s, mass2, m2_rem), 0.0)) / (2.0 * root_s)
    t_min = m_p2 + mass2 - 2.0 * (e_p * e_m + p_star * q)
    return (t_min, 0.0)


# ----------------------------------------------------------------------
# The marginal density and its grid normalization
# ----------------------------------------------------------------------
def marginal_density(m2: float, epsilon: float, b_sd: float, t: float) -> float:
    """The plan-form marginal density at the fixed transfer *t*
    (unnormalized):

    ``(M^2)^(-1-epsilon) * exp[b_sd * t * M^2]``.

    ``m2`` must be positive and finite (the physical domain starts at
    ``m_p^2``); the parameters must be finite.  At ``epsilon = 0``,
    ``t = 0`` the density is exactly ``1 / M^2`` (the flat-in-ln M^2
    law).  The nominal domain of *t* is the exchange range ``[t_min,
    0]``; the density is well defined for any finite *t* and the
    phase-space limits are the caller's contract.
    """

    x = float(m2)
    eps = float(epsilon)
    slope = float(b_sd)
    transfer = float(t)
    if not math.isfinite(x) or x <= 0.0:
        raise ValueError(
            f"marginal_density: m2 must be a positive finite GeV^2 mass "
            f"squared, got {m2!r}"
        )
    if not (math.isfinite(eps) and math.isfinite(slope)
            and math.isfinite(transfer)):
        raise ValueError(
            "marginal_density: epsilon, b_sd, and t must be finite, got "
            f"({epsilon!r}, {b_sd!r}, {t!r})"
        )
    return x ** (-1.0 - eps) * math.exp(slope * transfer * x)


def _marginal_log_cells(
    lo: float, hi: float, epsilon: float, b_sd: float, t: float
) -> tuple[np.ndarray, np.ndarray]:
    """The trapezoid cell masses of the marginal density on the
    geometric 4096-point grid (uniform in ``y = ln M^2``).

    With ``g(y) = f(e^y) e^y = (M^2)^(-epsilon) exp[b_sd t M^2]`` the
    integral ``int f dM^2 = int g dy`` is preserved, and ``g`` is smooth
    and slowly varying where the linear grid failed: on the central
    window ``[m_pi0^2, (sqrt_s - M_1 - M_2)^2]`` a linear 4096-point grid
    had a step of 0.058 GeV^2, three times the lower edge 0.018 GeV^2,
    and smeared the steep ``(M^2)^(-1-epsilon)`` law over its first
    cells (drawn P(M_c^2 < 0.05 GeV^2) 0.320 against the analytic 0.367
    at epsilon = 0; corrected 2026-09-28, FAB-03).  Returns the grid in
    ``y`` and the per-cell masses.
    """
    ys = np.linspace(math.log(lo), math.log(hi), _MARGINAL_GRID_POINTS)
    xs = np.exp(ys)
    g = np.power(xs, -float(epsilon)) * np.exp(float(b_sd) * float(t) * xs)
    cell = 0.5 * (g[1:] + g[:-1]) * (ys[1:] - ys[:-1])
    return ys, cell


def marginal_grid_mass(
    m2_min: float, m2_max: float, epsilon: float, b_sd: float, t: float
) -> float:
    """The 4096-point trapezoid mass of the marginal density on
    ``[m2_min, m2_max]`` (the grid normalization the M^2 draws use; the
    grid is geometric, uniform in ``ln M^2``, since 2026-09-28).

    The documented sub-tolerance convention: trapezoid cell masses on
    the fixed grid, shared with
    :func:`na49_mixing.excitation_kinematics.draw_from_grid_density` and
    ``na49_mixing.breit_wigner.draw_truncated_bw``.  The window must
    satisfy ``0 < m2_min < m2_max``; a window whose grid mass vanishes
    or is non-finite raises ``ValueError`` (a degenerate or ineligible
    configuration, in the house draw convention's wording).
    """

    lo = float(m2_min)
    hi = float(m2_max)
    if not (math.isfinite(lo) and math.isfinite(hi)) or not hi > lo > 0.0:
        raise ValueError(
            "marginal_grid_mass: the window must satisfy 0 < m2_min < "
            f"m2_max, got ({m2_min!r}, {m2_max!r})"
        )
    eps = float(epsilon)
    slope = float(b_sd)
    transfer = float(t)
    if not (math.isfinite(eps) and math.isfinite(slope)
            and math.isfinite(transfer)):
        raise ValueError(
            "marginal_grid_mass: epsilon, b_sd, and t must be finite, got "
            f"({epsilon!r}, {b_sd!r}, {t!r})"
        )
    _ys, cell = _marginal_log_cells(lo, hi, eps, slope, transfer)
    total = float(cell.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(
            "marginal_grid_mass: the marginal density carries no mass on "
            f"the grid over ({m2_min!r}, {m2_max!r}) at (epsilon={eps!r}, "
            f"b_sd={slope!r}, t={transfer!r})"
        )
    return total


# ----------------------------------------------------------------------
# The draws
# ----------------------------------------------------------------------
def draw_marginal(
    m2_min: float,
    m2_max: float,
    params: MassLawParams,
    t: float,
    rng: np.random.Generator,
) -> float:
    """One ``M^2`` draw from the marginal density on ``[m2_min,
    m2_max]`` at the fixed transfer *t*, via the geometric 4096-point
    grid normalization (uniform in ``ln M^2``); exactly 2 uniforms from
    *rng*.

    The inverse CDF of the trapezoid cell masses in ``ln M^2``: the first
    uniform selects the cell, the second is uniform within the cell (the
    Round 2 mass-draw convention, mirroring
    ``na49_mixing.breit_wigner.draw_truncated_bw``).  The nominal
    single-diffraction instantiation passes ``t = 0.0`` (the pure
    power-law marginal; the t factor of the law acts on the exchange
    kinematics only).  The draw is clipped to the window; a window
    whose grid mass vanishes raises ``ValueError``.
    """

    if not isinstance(params, MassLawParams):
        raise ValueError(
            f"draw_marginal: expected MassLawParams, got {type(params)!r}"
        )
    lo = float(m2_min)
    hi = float(m2_max)
    total = marginal_grid_mass(lo, hi, params.epsilon, params.b_sd, t)
    ys, cell = _marginal_log_cells(lo, hi, params.epsilon, params.b_sd, t)
    cdf = np.cumsum(cell) / total
    u_segment = rng.random()
    idx = int(np.searchsorted(cdf, u_segment, side="right"))
    if idx >= cdf.size:
        idx = cdf.size - 1
    u_within = rng.random()
    # uniform in ln M^2 inside the selected cell (the geometric grid's
    # cell is a constant ratio, so the position is a power of it)
    y_value = ys[idx] + u_within * (ys[idx + 1] - ys[idx])
    value = math.exp(y_value)
    return float(min(max(value, lo), hi))


def draw_marginal_edge(
    m2_min: float,
    m2_max: float,
    params: MassLawParams,
    s: float,
    xi_c: float,
    width: float,
    rng: np.random.Generator,
) -> float:
    """One ``M^2`` draw from the marginal density at ``t = 0`` times the
    smooth upper edge ``1 / (1 + exp((M^2/s - xi_c) / width))``, on the
    same geometric 4096-point grid as :func:`draw_marginal`; exactly 2
    uniforms from *rng*.

    The leading-proton tail of single diffraction (2026-10-25, at the R116
    optimum): its law ``dsigma/dxi ~ xi^a`` peaks at its upper end, and a
    wall there puts a step into the soft protons at ``x_F = 1 - xi_hi``
    (0.287 -> 0.457 of the data's 0.315 -> 0.367 at p_T 0.1 GeV/c between
    x_F 0.45 and 0.50).  The reggeized pion exchange fades as the proton
    stops leading; the edge gives it that fall-off, with the window's upper
    end well above ``xi_c``.
    """

    if not isinstance(params, MassLawParams):
        raise ValueError(
            f"draw_marginal_edge: expected MassLawParams, got {type(params)!r}"
        )
    lo = float(m2_min)
    hi = float(m2_max)
    ss, xc, w = float(s), float(xi_c), float(width)
    if not (ss > 0.0 and w > 0.0 and math.isfinite(xc)):
        raise ValueError(
            f"draw_marginal_edge: needs s > 0, width > 0 and a finite xi_c, got ({s!r}, {xi_c!r}, {width!r})"
        )
    ys, cell = _marginal_log_cells(lo, hi, params.epsilon, params.b_sd, 0.0)
    # the edge at the cell's midpoint in ln M^2 (the grid is fine: 4095 cells
    # across the window, against an edge width of order 0.05 s)
    x_mid = np.exp(0.5 * (ys[1:] + ys[:-1]))
    z = np.clip((x_mid / ss - xc) / w, -700.0, 700.0)
    cell = cell / (1.0 + np.exp(z))
    total = float(cell.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(
            f"draw_marginal_edge: the density carries no mass over ({m2_min!r}, {m2_max!r})"
        )
    cdf = np.cumsum(cell) / total
    u_segment = rng.random()
    idx = int(np.searchsorted(cdf, u_segment, side="right"))
    if idx >= cdf.size:
        idx = cdf.size - 1
    u_within = rng.random()
    y_value = ys[idx] + u_within * (ys[idx + 1] - ys[idx])
    value = math.exp(y_value)
    return float(min(max(value, lo), hi))


def draw_t(
    t_min: float, params: MassLawParams, rng: np.random.Generator
) -> float:
    """One t draw from ``exp(b_sd t)`` on ``[t_min, 0]`` via the
    inverse CDF; exactly 1 uniform from *rng*.

    The closed-form quantile is
    ``t(u) = ln(exp(b_sd t_min) + u (1 - exp(b_sd t_min))) / b_sd``.
    A slope magnitude at or below 1e-14 falls back to the uniform limit
    on the range (the same flat-slope threshold as
    :func:`na49_mixing.excitation_kinematics.sample_cos_from_exp_t`).
    ``t_min`` must be a finite non-positive GeV^2 value; the zero
    quantile of an underflowed range edge returns ``t_min`` exactly
    (the true quantile of the closed form).
    """

    lo = float(t_min)
    if not math.isfinite(lo) or lo > 0.0:
        raise ValueError(
            f"draw_t: t_min must be a finite non-positive GeV^2 value, "
            f"got {t_min!r}"
        )
    if not isinstance(params, MassLawParams):
        raise ValueError(
            f"draw_t: expected MassLawParams, got {type(params)!r}"
        )
    slope = float(params.b_sd)
    if not math.isfinite(slope):
        raise ValueError(
            f"draw_t: b_sd must be finite, got {params.b_sd!r}"
        )
    u = rng.random()
    if abs(slope) <= _FLAT_SLOPE_EPS:
        return lo + u * (0.0 - lo)
    e_lo = math.exp(slope * lo)
    if e_lo == 0.0 and u == 0.0:
        return lo
    value = math.log(e_lo + u * (1.0 - e_lo)) / slope
    return float(min(max(value, lo), 0.0))


# ----------------------------------------------------------------------
# The exact transfer window of a 2 -> 2 final state (2026-10-05)
# ----------------------------------------------------------------------
#
# The code and physics review of 2026-10-05 (finding H-1; card
# TASK-20261005-kernel-nd-dd-transfer-range): drawing t on [t_min, 0] and
# solving the production angle from it clamps cos(theta*) into [-1, 1] for
# the draws outside the reachable range, which leaves the excited line at
# p_T = 0 (42% of the non-diffractive, 48% of the double-diffractive and 12%
# of the single-diffractive events at the R75 optimum).  For p + p ->
# X_1(M_1) + X_2(M_2) the reachable range of t = (p_a - p_1)^2 = (p_b - p_2)^2
# is the window [t_-, t_+] of cos(theta*) = -1 and +1 at the pinned beam
# configuration; the generators draw on it under
# TopologyNominalParams.exact_transfer_range and the reweighting below uses
# the same window.

#: The exact invariant mass of the pinned beam configuration (the solves of
#: the class modules use it; the recorded SQRT_S = 17.270 is its rounding).
PINNED_SQRT_S_EXACT: float = math.sqrt(
    2.0 * PROTON_MASS * PROTON_MASS
    + 2.0 * PROTON_MASS * math.sqrt(PROTON_MASS * PROTON_MASS + _BEAM_MOMENTUM * _BEAM_MOMENTUM))


def pp_t_window(m2_1: float, m2_2: float, sqrt_s: float = PINNED_SQRT_S_EXACT) -> tuple[float, float]:
    """The reachable transfer window ``(t_-, t_+)`` [GeV^2] of
    ``p + p -> X_1 + X_2`` at *sqrt_s* for the masses squared *m2_1*,
    *m2_2*: ``t_+- = m_p^2 + M_1^2 - 2 (E_p E_1 -+ p* q)`` with the incoming
    proton energy ``sqrt_s / 2``, its momentum ``p*`` and the outgoing
    ``E_1``, ``q`` of the pair.  ``t_+`` is clipped at 0 (it is negative
    except at the elastic point).  ``ValueError`` on a closed pair."""
    a, b, root_s = float(m2_1), float(m2_2), float(sqrt_s)
    if not (a > 0.0 and b > 0.0 and math.isfinite(a) and math.isfinite(b)):
        raise ValueError(f"pp_t_window: the masses squared must be positive and finite, got {m2_1!r}, {m2_2!r}")
    s = root_s * root_s
    if math.sqrt(a) + math.sqrt(b) > root_s + _CLOSURE_TOLERANCE:
        raise ValueError(f"pp_t_window: the pair is closed at sqrt_s = {root_s!r} GeV")
    m_p2 = PROTON_MASS * PROTON_MASS
    e_p = 0.5 * root_s
    p_star = math.sqrt(max(0.25 * s - m_p2, 0.0))
    e_1 = (s + a - b) / (2.0 * root_s)
    q = math.sqrt(max(kallen(s, a, b), 0.0)) / (2.0 * root_s)
    base = m_p2 + a - 2.0 * e_p * e_1
    return (base - 2.0 * p_star * q, min(base + 2.0 * p_star * q, 0.0))


def pp_t_window_vec(m2_1, m2_2, sqrt_s: float = PINNED_SQRT_S_EXACT) -> tuple[np.ndarray, np.ndarray]:
    """:func:`pp_t_window` elementwise (the same arithmetic in the same order)."""
    a = np.asarray(m2_1, dtype=float)
    b = np.asarray(m2_2, dtype=float)
    root_s = float(sqrt_s)
    s = root_s * root_s
    m_p2 = PROTON_MASS * PROTON_MASS
    e_p = 0.5 * root_s
    p_star = math.sqrt(max(0.25 * s - m_p2, 0.0))
    e_1 = (s + a - b) / (2.0 * root_s)
    q = np.sqrt(np.maximum(kallen(s, a, b), 0.0)) / (2.0 * root_s)
    base = m_p2 + a - 2.0 * e_p * e_1
    return base - 2.0 * p_star * q, np.minimum(base + 2.0 * p_star * q, 0.0)


def draw_t_window(t_lo: float, t_hi: float, params: "MassLawParams", rng: np.random.Generator) -> float:
    """One t draw from ``exp(b_sd t)`` on ``[t_lo, t_hi]`` by the inverse
    CDF; exactly 1 uniform from *rng*.  The quantile is written from the
    upper edge, ``t(u) = t_hi + ln(e^(b (t_lo - t_hi)) + u (1 - e^(b (t_lo -
    t_hi)))) / b``, which stays exact when ``b t_lo`` underflows.  A slope at
    or below 1e-14 falls back to the uniform law on the window."""
    lo, hi = float(t_lo), float(t_hi)
    if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi <= 0.0):
        raise ValueError(f"draw_t_window: the window must satisfy t_lo <= t_hi <= 0, got ({t_lo!r}, {t_hi!r})")
    if not isinstance(params, MassLawParams):
        raise ValueError(f"draw_t_window: expected MassLawParams, got {type(params)!r}")
    slope = float(params.b_sd)
    u = rng.random()
    if abs(slope) <= _FLAT_SLOPE_EPS or hi == lo:
        return lo + u * (hi - lo)
    e = math.exp(slope * (lo - hi))
    value = hi + math.log(e + u * (1.0 - e)) / slope
    return float(min(max(value, lo), hi))


def _validated_t_break(t_break) -> "tuple[float, float]":
    """The two-slope transfer's ``(t_b, rho)``: a break at ``t = -t_b`` with
    ``t_b > 0`` and the slope ratio ``0 < rho <= 1`` below it."""
    if t_break is None or len(t_break) != 2 or any(isinstance(v, bool) for v in t_break):
        raise ValueError(f"the transfer break is (t_b, rho), got {t_break!r}")
    t_b, rho = float(t_break[0]), float(t_break[1])
    if not (math.isfinite(t_b) and t_b > 0.0 and math.isfinite(rho) and 0.0 < rho <= 1.0):
        raise ValueError(f"the transfer break needs t_b > 0 and 0 < rho <= 1, got {t_break!r}")
    return t_b, rho


def draw_t_window_break(t_lo: float, t_hi: float, params: "MassLawParams", t_break, rng: np.random.Generator) -> float:
    """One t draw from the two-slope transfer of the intact proton's vertex
    (2026-10-24): ``exp(b t)`` above ``-t_b`` and the continuous
    ``exp(-b t_b + rho b (t + t_b))`` below it, on ``[t_lo, t_hi]``, by the
    inverse CDF on exactly 1 uniform from *rng* with the orientation of
    :func:`draw_t_window` (``u = 0`` at ``t_lo``).  A window above the break
    is drawn by :func:`draw_t_window` itself, and ``rho = 1`` gives its law.
    The x_F 0.95 protons of p+p at 158 GeV/c change slope near
    ``|t| = 1 GeV^2`` (the dip region of proton-proton diffraction)."""
    lo, hi = float(t_lo), float(t_hi)
    if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi <= 0.0):
        raise ValueError(f"draw_t_window_break: the window must satisfy t_lo <= t_hi <= 0, got ({t_lo!r}, {t_hi!r})")
    t_b, rho = _validated_t_break(t_break)
    slope = float(params.b_sd)
    if not slope > _FLAT_SLOPE_EPS:
        raise ValueError(f"draw_t_window_break: the break needs a positive slope, got {slope!r}")
    xb = -t_b
    if lo >= xb or hi == lo:
        return draw_t_window(lo, hi, params, rng)
    u = rng.random()
    a, c = max(lo, xb), min(hi, xb)
    # the masses of the two pieces, the upper one from its own top edge
    n_a = math.exp(slope * hi) * -math.expm1(slope * (a - hi)) / slope if hi > a else 0.0
    rb = rho * slope
    n_b = math.exp(-slope * t_b + rb * (c - xb)) * -math.expm1(rb * (lo - c)) / rb
    r = (1.0 - u) * (n_a + n_b)
    if r <= n_a and n_a > 0.0:
        value = math.log(max(math.exp(slope * hi) - slope * r, 1e-300)) / slope
        return float(min(max(value, a), hi))
    r -= n_a
    value = xb + math.log(max(math.exp(rb * (c - xb)) - r * rb * math.exp(slope * t_b), 1e-300)) / rb
    return float(min(max(value, lo), c))


def vertex_slope_factor(m1: float, m2: float, kappa: float) -> float:
    """The factorized vertex slopes of one 2 -> 2 transfer (2026-10-08):
    under ``TopologyNominalParams.transfer_slope_mass_exponent`` kappa the
    transfer of a final state with outgoing line masses *m1* and *m2*
    [GeV] follows ``exp(B t)`` at ``B = b [(m_p/M_1)^kappa +
    (m_p/M_2)^kappa] / 2``.  Each vertex carries half of the slope b at the
    proton mass and less above it, the slope-mass correlation of
    diffraction dissociation.  Returns ``B / b``; a line at or below m_p
    counts as m_p, and kappa = 0 gives 1 exactly.  Scalar ``math.pow``, so
    the factor recorded on the event is the same on every CPU."""
    k = float(kappa)
    if not (math.isfinite(k) and k >= 0.0):
        raise ValueError(f"vertex_slope_factor: kappa must be finite and non-negative, got {kappa!r}")
    r1 = PROTON_MASS / max(float(m1), PROTON_MASS)
    r2 = PROTON_MASS / max(float(m2), PROTON_MASS)
    return 0.5 * (math.pow(r1, k) + math.pow(r2, k))


def _log_t_window_norm_slopes(b, t_lo, t_hi) -> np.ndarray:
    """:func:`log_t_window_norm_vec` with one slope per element (the
    factorized vertex slopes, 2026-10-08); a negative slope weights the
    lower edge, as in :func:`log_t_window_norm_vec` (review grok-4.7 L1,
    2026-10-10)."""
    slope = np.asarray(b, dtype=float)
    lo = np.asarray(t_lo, dtype=float)
    hi = np.asarray(t_hi, dtype=float)
    slope, lo, hi = np.broadcast_arrays(slope, lo, hi)
    out = np.zeros(slope.shape, dtype=float)
    live = (hi - lo) > 0.0
    flat = live & (np.abs(slope) <= _FLAT_SLOPE_EPS)
    out[flat] = np.log((hi - lo)[flat])
    steep = live & ~flat & (slope > 0.0)
    s, a, c = slope[steep], lo[steep], hi[steep]
    out[steep] = s * c + np.log(-np.expm1(np.minimum(s * (a - c), 0.0)) / s)
    falling = live & ~flat & (slope < 0.0)
    s, a, c = slope[falling], lo[falling], hi[falling]
    out[falling] = s * a + np.log(-np.expm1(s * (c - a)) / (-s))
    return out


def log_t_window_norm_vec(b: float, t_lo, t_hi) -> np.ndarray:
    """``log`` of ``int_{t_lo}^{t_hi} exp(b t) dt`` elementwise (the
    normalization of :func:`draw_t_window`); a degenerate window contributes 0."""
    lo = np.asarray(t_lo, dtype=float)
    hi = np.asarray(t_hi, dtype=float)
    width = hi - lo
    out = np.zeros(np.broadcast(lo, hi).shape, dtype=float)
    live = width > 0.0
    b_float = float(b)
    if abs(b_float) <= _FLAT_SLOPE_EPS:
        out[live] = np.log(width[live])
        return out
    if b_float < 0.0:
        # a negative slope weights the lower edge: Z = exp(b t_lo)
        # (1 - exp(-|b| (t_hi - t_lo))) / |b| (2026-10-07; review
        # 2026-10-06, Grok L1); the b > 0 branch below is unchanged
        out[live] = b_float * lo[live] + np.log(-np.expm1(b_float * width[live]) / (-b_float))
        return out
    arg = np.minimum(b_float * (lo - hi), 0.0)
    z = (1.0 - np.exp(arg)) / b_float
    out[live] = b_float * hi[live] + np.log(z[live])
    return out


def joint_closure(
    m1: float, m2: float, sqrt_s: float, m_central: float = 0.0
) -> bool:
    """The exact joint closure ``M_1 + M_2 + M_central <= sqrt_s``
    with the pinned 1e-12 float tolerance.

    The double-excitation draw accepts on ``M_1 + M_2 <= sqrt_s``
    (``m_central`` at its default 0.0); the central-production class
    passes the central mass through ``m_central``.  All arguments must
    be finite.
    """

    a = float(m1)
    b = float(m2)
    root_s = float(sqrt_s)
    m_c = float(m_central)
    if not (math.isfinite(a) and math.isfinite(b)
            and math.isfinite(root_s) and math.isfinite(m_c)):
        raise ValueError(
            "joint_closure: m1, m2, sqrt_s, and m_central must be finite, "
            f"got ({m1!r}, {m2!r}, {sqrt_s!r}, {m_central!r})"
        )
    return a + b + m_c <= root_s + _CLOSURE_TOLERANCE


def draw_joint_dd(
    m2_min_1: float,
    m2_max_1: float,
    m2_min_2: float,
    m2_max_2: float,
    sqrt_s: float,
    params: MassLawParams,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    """The separate joint law for double excitation: one accepted
    ``(M_1^2, M_2^2, t_1, t_2)`` draw.

    Each attempt draws ``M_1^2`` and ``M_2^2`` from the marginals on
    the caller's windows (2 uniforms each, at ``t = 0``), accepts on
    the exact closure ``M_1 + M_2 <= sqrt_s`` (:func:`joint_closure`),
    and then draws ``t_1`` and ``t_2`` from ``exp(b_sd t)`` on each
    exchange's exact two-body range :func:`t_range` computed with the
    other line at the intact proton mass (the factorized effective
    law; the exact ``t_1 + t_2`` coupling of the 2-to-2 system is not
    claimed).  Exactly 6 uniforms per accepted draw; a rejected
    attempt is retried, and
    :class:`PhaseSpaceRejectionExceeded` is raised after
    :data:`REJECTION_CAP` consecutive rejected attempts (the message
    names the attempt count and the parameter point).
    """

    if not isinstance(params, MassLawParams):
        raise ValueError(
            f"draw_joint_dd: expected MassLawParams, got {type(params)!r}"
        )
    root_s = float(sqrt_s)
    if not math.isfinite(root_s):
        raise ValueError(
            f"draw_joint_dd: sqrt_s must be finite, got {sqrt_s!r}"
        )
    last_m1 = 0.0
    last_m2 = 0.0
    for attempt in range(1, REJECTION_CAP + 1):
        m2_1 = draw_marginal(m2_min_1, m2_max_1, params, 0.0, rng)
        m2_2 = draw_marginal(m2_min_2, m2_max_2, params, 0.0, rng)
        m1 = math.sqrt(m2_1)
        m2 = math.sqrt(m2_2)
        last_m1 = m1
        last_m2 = m2
        if joint_closure(m1, m2, root_s):
            t_min_1 = t_range(m2_1, PROTON_MASS, root_s)[0]
            t_1 = draw_t(t_min_1, params, rng)
            t_min_2 = t_range(m2_2, PROTON_MASS, root_s)[0]
            t_2 = draw_t(t_min_2, params, rng)
            return (m2_1, m2_2, t_1, t_2)
    raise PhaseSpaceRejectionExceeded(
        f"joint double-excitation closure rejection exceeded the cap of "
        f"{REJECTION_CAP} attempts at the parameter point (epsilon="
        f"{params.epsilon!r}, b_sd={params.b_sd!r} GeV^-2) with limits "
        f"(m2_min_1={float(m2_min_1)!r}, m2_max_1={float(m2_max_1)!r}, "
        f"m2_min_2={float(m2_min_2)!r}, m2_max_2={float(m2_max_2)!r}, "
        f"sqrt_s={root_s!r}); the last attempt drew M_1={last_m1!r} GeV, "
        f"M_2={last_m2!r} GeV"
    )


def draw_joint_dd_exact(
    m2_min_1: float,
    m2_max_1: float,
    m2_min_2: float,
    m2_max_2: float,
    sqrt_s: float,
    params: MassLawParams,
    rng: np.random.Generator,
    slope_mass_exponent: float | None = None,
    pomeron_slope: float | None = None,
    pomeron_linear_t: float | None = None,
) -> tuple[float, float, float, float]:
    """The double-excitation draw with the exact transfer window
    (2026-10-05): the masses as in :func:`draw_joint_dd` (2 + 2 uniforms per
    attempt, the closure rejection), then the one transfer of the 2 -> 2
    final state on :func:`pp_t_window` of the accepted pair (1 uniform).
    Returns ``(M_1^2, M_2^2, t, t)``: ``t_1 = t_2`` by momentum conservation.
    Exactly 5 uniforms per accepted draw.  *slope_mass_exponent* (2026-10-08)
    draws the transfer at the pair's factorized vertex slope
    (:func:`vertex_slope_factor`).  *pomeron_slope* (2026-11-01) alpha'
    draws the Pomeron's shrinkage law (the section "The Pomeron's shrinkage"):
    a pair that passes the closure is kept with the probability
    ``R = Z_{B+c} / Z_B <= 1`` on one more uniform (a rejected pair starts a
    new attempt), and the transfer follows the slope ``B + c``
    (*pomeron_linear_t*: down to ``-t_lin``, the slope ``B`` below it)."""
    if not isinstance(params, MassLawParams):
        raise ValueError(f"draw_joint_dd_exact: expected MassLawParams, got {type(params)!r}")
    root_s = float(sqrt_s)
    for _attempt in range(1, REJECTION_CAP + 1):
        m2_1 = draw_marginal(m2_min_1, m2_max_1, params, 0.0, rng)
        m2_2 = draw_marginal(m2_min_2, m2_max_2, params, 0.0, rng)
        if joint_closure(math.sqrt(m2_1), math.sqrt(m2_2), root_s):
            t_lo, t_hi = pp_t_window(m2_1, m2_2)
            t_params = params
            if slope_mass_exponent is not None:
                factor = vertex_slope_factor(math.sqrt(m2_1), math.sqrt(m2_2), slope_mass_exponent)
                t_params = MassLawParams(epsilon=params.epsilon, b_sd=params.b_sd * factor)
            if pomeron_slope is not None:
                big_b = float(t_params.b_sd)
                c = 2.0 * float(pomeron_slope) * float(pomeron_gap_log(m2_1, m2_2))
                lo_a, hi_a = np.asarray([t_lo]), np.asarray([t_hi])
                s1, s2, tx = _pomeron_slopes(big_b, c, None, pomeron_linear_t)
                log_r = float(_log_pw_norm(s1, s2, lo_a, hi_a, tx)[0]
                              - _log_t_window_norm_slopes(np.asarray([big_b]), lo_a, hi_a)[0])
                if rng.random() >= math.exp(min(log_r, 0.0)):
                    continue
                t = draw_t_pomeron(t_lo, t_hi, big_b, c, None, rng, t_lin=pomeron_linear_t)
                return (m2_1, m2_2, t, t)
            t = draw_t_window(t_lo, t_hi, t_params, rng)
            return (m2_1, m2_2, t, t)
    raise PhaseSpaceRejectionExceeded(
        f"joint double-excitation closure rejection exceeded the cap of {REJECTION_CAP} attempts at "
        f"(epsilon={params.epsilon!r}, b_sd={params.b_sd!r} GeV^-2)")


def draw_central(
    m2_min: float,
    m2_max: float,
    params: MassLawParams,
    t_c: float,
    rng: np.random.Generator,
) -> float:
    """The central-production mass draw: the same functional form as
    the marginal law with the total transfer *t_c* (the sum of the two
    exchanges' transfers); exactly 2 uniforms.

    The central mass ``M_c^2`` is drawn from ``(M_c^2)^(-1-epsilon) *
    exp[b_sd * t_c * M_c^2]`` on the caller's window by the same
    4096-point grid inverse CDF as :func:`draw_marginal`.  The lower
    mass reference of the central law is the catalog pion
    :data:`PI0_MASS` (the lightest neutral state); the window is the
    caller's phase-space contract.
    """

    return draw_marginal(m2_min, m2_max, params, t_c, rng)


# ----------------------------------------------------------------------
# Exact per-event density ratios of the mass laws (the importance
# weights of the topology fits; 2026-09-28)
# ----------------------------------------------------------------------
#
# The R51 driver carried these ratios as private functions and added a
# spurious single-marginal normalization ratio to the double-diffraction
# branch (review record .collab/reviews/fable-speedup-2026-09-28.md,
# FAB-04: measured mean weight 2.03 at epsilon = 0.3 and 0.39 at
# epsilon = -0.3 over a nominal sample, against the exact 1.0).  The
# library form below is the corrected reference: for every class the
# mean of exp(log_density_ratio) over a sample generated at
# (eps0, b_sd0) is 1 (tests/test_importance.py).
#
# Array conventions per class (the stored-event arrays of the run
# drivers):
#   single diffraction:   m2 (M^2), t, m_rem (the other line's mass)
#   double diffraction:   m12, m22 (M_1^2, M_2^2), t1, t2
#   non-diffractive:      t1, t2, t_c (= t1 + t2), mc2 (M_c^2), m1, m2
#                         plus k1, k2 (remnant kind index: intact 0, low-mass
#                         1, excited 2) and lo1, hi1, lo2, hi2 (the excited
#                         kinds' mass windows [GeV]; 2026-09-29)
#                         (the two remnant masses)

_POWER_EPS_BRANCH: float = 1e-13
_GAMMA_S_BRANCH: float = 1e-8
_DD_GL_ORDER: int = 256
_DD_GL_NODES, _DD_GL_WEIGHTS = np.polynomial.legendre.leggauss(_DD_GL_ORDER)
_DD_JOINT_CACHE: dict[tuple[float, float], float] = {}


def log_power_integral(eps: float, a: float, c: float) -> float:
    """``log`` of ``A(eps; a, c) = int_a^c x^(-1-eps) dx`` (``eps = 0``
    is the logarithmic limit ``log(c/a)``); ``0 < a < c``."""
    a = float(a)
    c = float(c)
    if not (math.isfinite(a) and math.isfinite(c)) or not c > a > 0.0:
        raise ValueError(f"log_power_integral: the window ({a!r}, {c!r}) is invalid")
    eps = float(eps)
    if abs(eps) < _POWER_EPS_BRANCH:
        return math.log(math.log(c / a))
    num = math.expm1(-eps * math.log(c)) - math.expm1(-eps * math.log(a))
    value = num / (-eps)
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(
            f"log_power_integral: non-finite or non-positive at (eps={eps!r}, a={a!r}, c={c!r})")
    return math.log(value)


def log_power_integral_vec(eps: float, a, c) -> np.ndarray:
    """Elementwise :func:`log_power_integral` over window arrays ``a < c``
    (the excited-remnant windows of the exponent reweighting; 2026-09-29)."""
    a = np.asarray(a, dtype=float)
    c = np.asarray(c, dtype=float)
    a, c = np.broadcast_arrays(a, c)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(c)) and np.all(c > a) and np.all(a > 0.0)):
        raise ValueError("log_power_integral_vec: every window must satisfy 0 < a < c")
    eps = float(eps)
    if abs(eps) < _POWER_EPS_BRANCH:
        return np.log(np.log(c / a))
    num = np.expm1(-eps * np.log(c)) - np.expm1(-eps * np.log(a))
    value = num / (-eps)
    if not np.all(np.isfinite(value) & (value > 0.0)):
        raise ValueError(f"log_power_integral_vec: non-finite or non-positive at eps={eps!r}")
    return np.log(value)


def log_t_norm_vec(b: float, t_min) -> np.ndarray:
    """``log`` of ``Z_t(b, t_min) = int_{t_min}^0 exp(b t) dt``
    elementwise; a degenerate range ``t_min = 0`` contributes 0."""
    t_min = np.asarray(t_min, dtype=float)
    out = np.zeros(t_min.shape, dtype=float)
    live = t_min != 0.0
    b_float = float(b)
    if abs(b_float) <= _FLAT_SLOPE_EPS:
        out[live] = np.log(-t_min[live])
        return out
    if b_float < 0.0:
        # a negative slope weights the far edge t_min: Z = exp(b t_min)
        # (1 - exp(-|b| |t_min|)) / |b| (2026-10-07; review 2026-10-06,
        # Grok L1); the b > 0 branch below is unchanged bit for bit
        btm = b_float * t_min[live]
        out[live] = btm + np.log(-np.expm1(-btm) / (-b_float))
        return out
    btm = b_float * t_min
    z = np.where(btm < -700.0, 1.0 / b_float, (1.0 - np.exp(np.minimum(btm, 0.0))) / b_float)
    if live.any() and (not np.all(np.isfinite(z[live])) or not np.all(z[live] > 0.0)):
        raise ValueError(f"log_t_norm_vec: non-finite or non-positive normalization at b={b_float!r}")
    out[live] = np.log(z[live])
    return out


def t_range_min_vec(m2, m_rem, sqrt_s: float) -> np.ndarray:
    """The lower edge of :func:`t_range` elementwise (the same Kallen
    arithmetic in the same order)."""
    m2 = np.asarray(m2, dtype=float)
    m_rem = np.asarray(m_rem, dtype=float)
    root_s = float(sqrt_s)
    s = root_s * root_s
    m_p2 = PROTON_MASS * PROTON_MASS
    m2_rem = m_rem * m_rem
    e_p = (s + m_p2 - m2_rem) / (2.0 * root_s)
    e_m = (s + m2 - m2_rem) / (2.0 * root_s)
    p_star = np.sqrt(np.maximum(kallen(s, m_p2, m2_rem), 0.0)) / (2.0 * root_s)
    q = np.sqrt(np.maximum(kallen(s, m2, m2_rem), 0.0)) / (2.0 * root_s)
    return m_p2 + m2 - 2.0 * (e_p * e_m + p_star * q)


def log_dd_joint_mass(eps: float, sqrt_s: float, lo1: float = EXCITATION_M_MIN,
                      lo2: float = EXCITATION_M_MIN) -> float:
    """``log`` of the closure-restricted joint mass of the two
    unnormalized double-diffraction marginals,
    ``A_DD(eps) = int int_{M_1 + M_2 <= sqrt_s} x^(-1-eps) y^(-1-eps) dx dy``
    over ``x >= lo1^2, y >= lo2^2`` (the draw windows of :func:`sd_limits`,
    both edges ``EXCITATION_M_MIN`` by default; the per-line edges of
    ``line_thresholds`` otherwise, 2026-10-06; the closure bounds ``M_1``
    by ``sqrt_s - lo2``), by 256-point Gauss-Legendre on the exact
    one-dimensional reduction."""
    lo1, lo2 = float(lo1), float(lo2)
    key = (float(eps), float(sqrt_s)) if lo1 == lo2 == EXCITATION_M_MIN else (float(eps), float(sqrt_s), lo1, lo2)
    cached = _DD_JOINT_CACHE.get(key)
    if cached is not None:
        return cached
    eps = float(eps)
    lo = lo1
    hi = float(sqrt_s) - lo2
    mid = 0.5 * (hi + lo)
    half = 0.5 * (hi - lo)
    u = mid + half * _DD_GL_NODES
    if abs(eps) < _POWER_EPS_BRANCH:
        integrand = np.log((float(sqrt_s) - u) / lo2) / u
    else:
        j_values = (np.power(float(sqrt_s) - u, -2.0 * eps) - lo2 ** (-2.0 * eps)) / (-2.0 * eps)
        integrand = np.power(u, -1.0 - 2.0 * eps) * j_values
    mass = 4.0 * half * float(np.dot(_DD_GL_WEIGHTS, integrand))
    if not (math.isfinite(mass) and mass > 0.0):
        raise ValueError(f"log_dd_joint_mass: non-finite or non-positive joint mass at eps={eps!r}")
    _DD_JOINT_CACHE[key] = math.log(mass)
    return _DD_JOINT_CACHE[key]


def _dd_joint(eps: float, sqrt_s: float, lo1: float, lo2: float, xi_max: "float | None") -> float:
    """The double-diffraction joint mass: :func:`log_dd_joint_mass` on the
    kinematic windows, or under the coherence edge *xi_max* (2026-10-23) the
    product of the two marginal masses on ``[lo_i^2, xi_max s]``."""
    if xi_max is None:
        return log_dd_joint_mass(eps, sqrt_s, lo1, lo2)
    cap = float(xi_max) * float(sqrt_s) ** 2
    if not 2.0 * math.sqrt(cap) <= float(sqrt_s):
        raise ValueError(f"_dd_joint: the coherence edge {xi_max!r} lets the closure bind (needs xi_max <= 0.25)")
    return log_power_integral(eps, float(lo1) ** 2, cap) + log_power_integral(eps, float(lo2) ** 2, cap)


def _gamma_upper_vec(s, z):
    """Upper incomplete gamma ``Gamma(s, z)``, vectorized: the regularized
    product for ``s > 0``, ``E1`` near ``s = 0``, the ``s + 1`` recurrence
    for ``-1 < s < 0``, and the recurrence applied recursively for
    ``s <= -1`` (2026-10-02, external review g1 B1: one step left a
    non-positive order and NaN at ``epsilon >= 1``)."""
    import scipy.special as _sp

    s = np.asarray(s, dtype=float)
    z = np.asarray(z, dtype=float)
    s, z = np.broadcast_arrays(s, z)
    out = np.empty(s.shape, dtype=float)
    small = np.abs(s) < _GAMMA_S_BRANCH
    pos = (~small) & (s > 0.0)
    neg = (~small) & (s < 0.0) & (s > -1.0)
    deep = (~small) & (s <= -1.0)
    if small.any():
        out[small] = _sp.exp1(z[small])
    if pos.any():
        out[pos] = _sp.gamma(s[pos]) * _sp.gammaincc(s[pos], z[pos])
    if neg.any():
        s1 = s + 1.0
        out[neg] = (
            _sp.gamma(s1[neg]) * _sp.gammaincc(s1[neg], z[neg])
            - np.power(z[neg], s[neg]) * np.exp(-z[neg])
        ) / s[neg]
    if deep.any():
        sd, zd = s[deep], z[deep]
        out[deep] = (_gamma_upper_vec(sd + 1.0, zd) - np.power(zd, sd) * np.exp(-zd)) / sd
    return out


def log_central_mass_integral(eps, beta, a: float, c) -> np.ndarray:
    """``log`` of ``A_c(eps, beta; a, c) = int_a^c x^(-1-eps) exp(beta x) dx``
    elementwise (``beta <= 0``): the incomplete-gamma closed form
    ``(-beta)^eps [Gamma(-eps, -beta a) - Gamma(-eps, -beta c)]`` for
    ``beta < 0`` and the power integral for ``beta = 0``."""
    eps = np.asarray(eps, dtype=float)
    beta = np.asarray(beta, dtype=float)
    c = np.asarray(c, dtype=float)
    a = float(a)
    if not (math.isfinite(a) and a > 0.0):
        raise ValueError(f"log_central_mass_integral: the lower edge {a!r} is invalid")
    eps, beta, c = np.broadcast_arrays(eps, beta, c)
    out = np.empty(beta.shape, dtype=float)
    flat = np.abs(beta) < _FLAT_SLOPE_EPS
    if flat.any():
        out[flat] = [log_power_integral(float(e), a, float(cc)) for e, cc in zip(eps[flat], c[flat])]
    live = ~flat
    if live.any():
        s = -eps[live]
        beta_f = beta[live]
        diff = _gamma_upper_vec(s, -beta_f * a) - _gamma_upper_vec(s, -beta_f * c[live])
        with np.errstate(divide="ignore", invalid="ignore"):
            out[live] = eps[live] * np.log(-beta_f) + np.log(diff)
    if np.isnan(out).any():
        raise ValueError("log_central_mass_integral: NaN at "
                         f"eps = {np.unique(eps[np.isnan(out)]).tolist()[:5]} (checked, never repaired)")
    return out


#: The recorded kind index of an excited (high-mass continuum) remnant in
#: the non-diffractive shape arrays (na49_mixing.topology_remnants.
#: REMNANT_KIND_INDEX[KIND_HIGH_MASS_BARYONIC_CONTINUUM]; pinned by a test).
EXCITED_REMNANT_KIND_INDEX: int = 2


#: The distinct line-edge pairs of a class's event arrays and each event's
#: pair (2026-10-20, the fit profile: the grouping ran as a Python loop over
#: every event at every objective evaluation, a third of the fit's time).
#: Keyed on the source array objects (weakly, so a released template frees
#: its entry) and the scalar that enters the pair; each event takes its
#: pair's value, so every result is the float of the per-pair loop.
_PAIR_GROUPS: dict = {}
_PAIR_GROUPS_MAX = 256


def _pair_groups(sources: tuple, a: np.ndarray, b: np.ndarray, extra: tuple = ()) -> tuple[list, np.ndarray]:
    """The distinct ``(a, b)`` pairs and each element's pair index, cached
    on the identity of the *sources* arrays from which *a* and *b* derive."""
    key = tuple(id(x) for x in sources) + tuple(extra)
    hit = _PAIR_GROUPS.get(key)
    if hit is not None and all(r() is x for r, x in zip(hit[0], sources)):
        return hit[1], hit[2]
    pairs = list({(float(x), float(y)) for x, y in zip(a, b)})
    index = np.full(len(a), -1, dtype=np.intp)
    for k, (x, y) in enumerate(pairs):
        index[(a == x) & (b == y)] = k
    if np.any(index < 0):
        raise ValueError("log_density_ratio: an event's line edges match no pair (a non-finite edge)")
    if len(_PAIR_GROUPS) >= _PAIR_GROUPS_MAX:
        _PAIR_GROUPS.clear()
    _PAIR_GROUPS[key] = (tuple(weakref.ref(x) for x in sources), pairs, index)
    return pairs, index


def log_density_ratio(
    cls: str,
    arrays: dict,
    eps: float,
    b_sd: float,
    *,
    eps0: float,
    b_sd0: float,
    sqrt_s: float,
    exact_range: bool = False,
    central_law: str = "committed",
    eps_c: float | None = None,
    eps_c0: float | None = None,
    intact_factor: "tuple[float, float] | None" = None,
    xi_max: "float | None" = None,
    t_break: "tuple[float, float] | None" = None,
    pomeron: "tuple | None" = None,
) -> np.ndarray:
    """The per-event ``log rho_c(event; eps, b_sd) - log rho_c(event; eps0, b_sd0)``
    of one topology class, the exact importance weight of the class mass
    law between two parameter points (the mixture-fraction ratio is the
    caller's).  See the array conventions above.  The mean of
    ``exp`` of the result over a sample generated at ``(eps0, b_sd0)``
    is 1 for every class.  *exact_range* (2026-10-05) is the transfer law of
    events generated under ``TopologyNominalParams.exact_transfer_range``:
    one transfer per 2 -> 2 event on the window :func:`pp_t_window_vec` of
    its masses (the two-component non-diffractive variant only; an event
    with a central source is refused).  *central_law* (2026-10-07) is the
    central mass law of the class with the central source: ``"committed"``
    (``(M_c^2)^(-1-eps) exp(b_sd t_c M_c^2)`` with the non-diffractive
    exponent) or ``"power"`` (``(M_c^2)^(-1-eps_c)`` on the same window,
    its own exponent, reweighted from *eps_c0* to *eps_c*).  An event
    generated under ``TopologyNominalParams.transfer_slope_mass_exponent``
    (2026-10-08) carries its factorized vertex slope factor ``B / b`` as
    the array ``slope_factor`` (:func:`vertex_slope_factor`); its transfer
    term is then that of the slope ``b * slope_factor`` on the exact
    window (the law is defined with *exact_range* only).
    *intact_factor* ``(f, f0)`` (2026-10-15) reweights the intact proton's
    vertex term of a single-diffraction event (an event whose non-excited
    line ``m_rem`` is the proton) from the generation's f0 to f: its
    trial factor is the recorded one plus ``(f - f0) / 2``, and the other
    events keep theirs (``TopologyNominalParams.intact_vertex_factor``).
    *t_break* ``(t_b, rho)`` (2026-10-24) is the intact proton's two-slope
    transfer (``TopologyNominalParams.intact_t_break``, drawn by
    :func:`draw_t_window_break`): the intact-proton events of the
    single-diffraction classes take the ratio of the normalized two-slope
    densities, the break fixed and the slopes as above.
    *xi_max* (2026-10-23) is the coherence edge of the diffractive classes
    (``TopologyNominalParams.diffractive_xi_max``): every excited mass window
    ends at ``xi_max s``, and the double-diffraction joint mass is then the
    product of the two marginal masses (the closure never binds at
    ``xi_max <= 0.25``).
    *pomeron* ``(alpha', kappa, f)`` (2026-11-01) is the Pomeron's shrinkage
    of the diffractive classes (``TopologyNominalParams.diffractive_pomeron_slope``,
    the section "The Pomeron's shrinkage" below): the events whose array
    ``pomeron`` is 1 take the ratio of the shrinkage law's densities, kappa the
    vertex exponent and f the intact proton's vertex factor when it is fixed
    (``None`` when absent or reweighted by *intact_factor*); every other event
    keeps the committed ratio.  alpha' is the same at both points.
    """
    if pomeron is not None and cls in ("sd_projectile", "sd_target", "dd") and "pomeron" in arrays:
        shr = np.asarray(arrays["pomeron"], dtype=float) > 0.5
        base = log_density_ratio(cls, arrays, eps, b_sd, eps0=eps0, b_sd0=b_sd0, sqrt_s=sqrt_s, exact_range=exact_range,
                                 central_law=central_law, eps_c=eps_c, eps_c0=eps_c0, intact_factor=intact_factor,
                                 xi_max=xi_max, t_break=t_break)
        if not np.any(shr):
            return base
        if not exact_range:
            raise ValueError("log_density_ratio: the Pomeron's shrinkage is defined on the exact transfer window only")
        out = np.array(base, dtype=float, copy=True)
        out[shr] = _pomeron_log_ratio(cls, arrays, shr, float(eps), float(b_sd), float(eps0), float(b_sd0), tuple(pomeron),
                                      intact_factor, xi_max, t_break if cls != "dd" else None, float(sqrt_s))
        return out
    if central_law not in ("committed", "power"):
        raise ValueError(f"log_density_ratio: unknown central law {central_law!r}")
    if central_law == "power" and (eps_c is None or eps_c0 is None):
        raise ValueError("log_density_ratio: the power central law needs eps_c and eps_c0")
    eps = float(eps)
    b_sd = float(b_sd)
    eps0 = float(eps0)
    b0 = float(b_sd0)
    root_s = float(sqrt_s)
    delta_b = b_sd - b0
    m_p2 = PROTON_MASS * PROTON_MASS
    m2_low = EXCITATION_M_MIN * EXCITATION_M_MIN  # the sd_limits lower edge
    factor = arrays.get("slope_factor")
    if factor is not None and not exact_range:
        raise ValueError("log_density_ratio: the factorized vertex slopes are defined on the exact transfer window only")
    if t_break is not None and (cls not in ("sd_projectile", "sd_target") or factor is None or not exact_range):
        raise ValueError("log_density_ratio: the transfer break reweights the single-diffraction classes under the "
                         "factorized vertex slopes on the exact window only")
    if intact_factor is not None and (cls not in ("sd_projectile", "sd_target") or factor is None):
        raise ValueError("log_density_ratio: the intact vertex factor reweights the single-diffraction classes under the "
                         "factorized vertex slopes only")
    if cls in ("sd_projectile", "sd_target"):
        m2 = np.asarray(arrays["m2"], dtype=float)
        t = np.asarray(arrays["t"], dtype=float)
        m_rem = np.asarray(arrays["m_rem"], dtype=float)
        m2_max = (root_s - m_rem) ** 2
        if xi_max is not None:
            m2_max = np.minimum(m2_max, float(xi_max) * root_s * root_s)
        log_a_trial = np.empty(m2.shape)
        log_a_nom = np.empty(m2.shape)
        if "m2_low" in arrays:
            # the line's own lower edge (line_thresholds, 2026-10-06)
            low = np.asarray(arrays["m2_low"], dtype=float)
            pairs, index = _pair_groups((arrays["m2_low"], arrays["m_rem"]), low, m2_max, (root_s,))
            log_a_trial = np.asarray([log_power_integral(eps, edge, window) for edge, window in pairs], dtype=float)[index]
            log_a_nom = np.asarray([log_power_integral(eps0, edge, window) for edge, window in pairs], dtype=float)[index]
        else:
            for window in np.unique(m2_max):
                mask = m2_max == window
                log_a_trial[mask] = log_power_integral(eps, m2_low, float(window))
                log_a_nom[mask] = log_power_integral(eps0, m2_low, float(window))
        if exact_range:
            t_lo, t_hi = pp_t_window_vec(m2, m_rem * m_rem)
            if factor is not None:
                trial = None
                intact = m_rem <= PROTON_MASS * (1.0 + 1e-9)
                if intact_factor is not None:
                    f, f0 = (float(v) for v in intact_factor)
                    trial = np.asarray(factor, dtype=float) + 0.5 * (f - f0) * intact
                transfer = _vertex_transfer_term(factor, t, t_lo, t_hi, b_sd, b0, trial)
                if t_break is not None and np.any(intact):
                    # the intact proton's two-slope transfer (2026-10-24): its events
                    # take the ratio of the normalized two-slope densities
                    transfer = np.array(transfer, dtype=float, copy=True)
                    g = None if trial is None else np.asarray(trial, dtype=float)[intact]
                    transfer[intact] = _vertex_transfer_term_break(
                        np.asarray(factor, dtype=float)[intact], t[intact], np.asarray(t_lo)[intact],
                        np.asarray(t_hi)[intact], b_sd, b0, t_break, g)
                return (
                    -(eps - eps0) * np.log(m2)
                    + (log_a_nom - log_a_trial)
                    + transfer
                )
            t_term = log_t_window_norm_vec(b0, t_lo, t_hi) - log_t_window_norm_vec(b_sd, t_lo, t_hi)
        else:
            t_min = t_range_min_vec(m2, m_rem, root_s)
            t_term = log_t_norm_vec(b0, t_min) - log_t_norm_vec(b_sd, t_min)
        return (
            -(eps - eps0) * np.log(m2)
            + (log_a_nom - log_a_trial)
            + delta_b * t
            + t_term
        )
    if cls == "dd":
        m12 = np.asarray(arrays["m12"], dtype=float)
        m22 = np.asarray(arrays["m22"], dtype=float)
        t1 = np.asarray(arrays["t1"], dtype=float)
        t2 = np.asarray(arrays["t2"], dtype=float)
        if "m_lo1" in arrays:
            # the per-line lower edges (line_thresholds, 2026-10-06)
            lo1 = np.asarray(arrays["m_lo1"], dtype=float)
            lo2 = np.asarray(arrays["m_lo2"], dtype=float)
            pairs, index = _pair_groups((arrays["m_lo1"], arrays["m_lo2"]), lo1, lo2)
            joint = np.asarray([_dd_joint(eps0, root_s, a, b, xi_max) - _dd_joint(eps, root_s, a, b, xi_max)
                                for a, b in pairs], dtype=float)[index]
        else:
            joint = (_dd_joint(eps0, root_s, EXCITATION_M_MIN, EXCITATION_M_MIN, xi_max)
                     - _dd_joint(eps, root_s, EXCITATION_M_MIN, EXCITATION_M_MIN, xi_max))
        if exact_range:
            # one transfer of the 2 -> 2 pair on its window (t1 = t2)
            t_lo, t_hi = pp_t_window_vec(m12, m22)
            if factor is not None:
                return (
                    -(eps - eps0) * (np.log(m12) + np.log(m22))
                    + joint
                    + _vertex_transfer_term(factor, t1, t_lo, t_hi, b_sd, b0)
                )
            return (
                -(eps - eps0) * (np.log(m12) + np.log(m22))
                + joint
                + delta_b * t1
                + (log_t_window_norm_vec(b0, t_lo, t_hi) - log_t_window_norm_vec(b_sd, t_lo, t_hi))
            )
        t_min1 = t_range_min_vec(m12, PROTON_MASS, root_s)
        t_min2 = t_range_min_vec(m22, PROTON_MASS, root_s)
        # the accepted-pair density is x^(-1-eps) y^(-1-eps) 1[closure] / A_DD(eps):
        # the single-marginal normalizations of the two draws cancel in the
        # rejection, so only the joint mass enters (FAB-04)
        return (
            -(eps - eps0) * (np.log(m12) + np.log(m22))
            + joint
            + delta_b * (t1 + t2)
            + (log_t_norm_vec(b0, t_min1) - log_t_norm_vec(b_sd, t_min1))
            + (log_t_norm_vec(b0, t_min2) - log_t_norm_vec(b_sd, t_min2))
        )
    if cls == "nd":
        t1 = np.asarray(arrays["t1"], dtype=float)
        t2 = np.asarray(arrays["t2"], dtype=float)
        t_c = np.asarray(arrays["t_c"], dtype=float)
        mc2 = np.asarray(arrays["mc2"], dtype=float)
        m1 = np.asarray(arrays["m1"], dtype=float)
        m2 = np.asarray(arrays["m2"], dtype=float)
        m2_max = (root_s - m1 - m2) ** 2
        if exact_range:
            if np.any(mc2 > 0.0):
                raise ValueError("log_density_ratio: the exact transfer window covers the two-component variant only")
            t_lo, t_hi = pp_t_window_vec(m1 * m1, m2 * m2)
            if factor is not None:
                out = _vertex_transfer_term(factor, t1, t_lo, t_hi, b_sd, b0)
            else:
                out = delta_b * t1 + (log_t_window_norm_vec(b0, t_lo, t_hi) - log_t_window_norm_vec(b_sd, t_lo, t_hi))
            return _excited_remnant_terms(out, arrays, m1, m2, eps, eps0)
        t_min1 = t_range_min_vec(m1 * m1, PROTON_MASS, root_s)
        t_min2 = t_range_min_vec(m2 * m2, PROTON_MASS, root_s)
        a = PI0_MASS * PI0_MASS
        c = np.maximum(m2_max, a * (1.0 + 1e-12))
        if np.any(mc2 < 0.0):
            raise ValueError("log_density_ratio: a negative central mass squared")
        if central_law == "power":
            # the central system's own exponent, no transfer coupling (2026-10-07)
            has_c = mc2 > 0.0
            out = (
                delta_b * (t1 + t2)
                + (log_t_norm_vec(b0, t_min1) - log_t_norm_vec(b_sd, t_min1))
                + (log_t_norm_vec(b0, t_min2) - log_t_norm_vec(b_sd, t_min2))
            )
            if np.any(has_c):
                n_c = int(np.sum(has_c))
                zeros = np.zeros(n_c)
                log_ac_trial = log_central_mass_integral(np.full(n_c, float(eps_c)), zeros, a, c[has_c])
                log_ac_nom = log_central_mass_integral(np.full(n_c, float(eps_c0)), zeros, a, c[has_c])
                out[has_c] += -(float(eps_c) - float(eps_c0)) * np.log(mc2[has_c]) + (log_ac_nom - log_ac_trial)
            return _excited_remnant_terms(out, arrays, m1, m2, eps, eps0)
        if np.all(mc2 > 0.0):
            log_ac_trial = log_central_mass_integral(np.full(t_c.shape, eps), b_sd * t_c, a, c)
            log_ac_nom = log_central_mass_integral(np.full(t_c.shape, eps0), b0 * t_c, a, c)
            out = (
                delta_b * (t1 + t2)
                + (log_t_norm_vec(b0, t_min1) - log_t_norm_vec(b_sd, t_min1))
                + (log_t_norm_vec(b0, t_min2) - log_t_norm_vec(b_sd, t_min2))
                - (eps - eps0) * np.log(mc2)
                + delta_b * t_c * mc2
                + (log_ac_nom - log_ac_trial)
            )
        else:
            # events without a central source (mc2 = 0, the two-component
            # variant of 2026-10-02) carry no central mass law: only the
            # transfer and remnant terms apply to them
            has_c = mc2 > 0.0
            out = (
                delta_b * (t1 + t2)
                + (log_t_norm_vec(b0, t_min1) - log_t_norm_vec(b_sd, t_min1))
                + (log_t_norm_vec(b0, t_min2) - log_t_norm_vec(b_sd, t_min2))
            )
            if np.any(has_c):
                n_c = int(np.sum(has_c))
                log_ac_trial = log_central_mass_integral(np.full(n_c, eps), b_sd * t_c[has_c], a, c[has_c])
                log_ac_nom = log_central_mass_integral(np.full(n_c, eps0), b0 * t_c[has_c], a, c[has_c])
                out[has_c] += (
                    - (eps - eps0) * np.log(mc2[has_c])
                    + delta_b * t_c[has_c] * mc2[has_c]
                    + (log_ac_nom - log_ac_trial)
                )
        return _excited_remnant_terms(out, arrays, m1, m2, eps, eps0)
    raise ValueError(f"log_density_ratio: unknown topology class {cls!r}")


def _vertex_transfer_term(factor, t, t_lo, t_hi, b_sd, b0, trial=None) -> np.ndarray:
    """The transfer term of the factorized vertex slopes (2026-10-08): the
    log ratio of ``exp(b f t)`` normalized on the exact window between the
    slopes b_sd and b0, with f the event's recorded factor.  *trial*
    (2026-10-15) is the trial point's factor where it differs from the
    recorded one (the intact vertex term): the ratio is then that of the
    slopes ``b_sd * trial`` and ``b0 * factor``."""
    f = np.asarray(factor, dtype=float)
    t = np.asarray(t, dtype=float)
    out = (b_sd - b0) * f * t + (_log_t_window_norm_slopes(b0 * f, t_lo, t_hi)
                                 - _log_t_window_norm_slopes(b_sd * f, t_lo, t_hi))
    if trial is None:
        return out
    # the events whose factor moves; the others keep the committed term bit for bit
    g, f, t, lo, hi = np.broadcast_arrays(np.asarray(trial, dtype=float), f, t, np.asarray(t_lo, dtype=float),
                                          np.asarray(t_hi, dtype=float))
    moved = g != f
    out = np.array(out, dtype=float, copy=True)
    out[moved] = (b_sd * g[moved] - b0 * f[moved]) * t[moved] + (
        _log_t_window_norm_slopes(b0 * f[moved], lo[moved], hi[moved])
        - _log_t_window_norm_slopes(b_sd * g[moved], lo[moved], hi[moved]))
    return out


def _log_two_slope_norm(b, t_lo, t_hi, t_b: float, rho: float) -> np.ndarray:
    """``log`` of the two-slope transfer's normalization on ``[t_lo, t_hi]``
    elementwise, one slope per element: ``int exp(b t)`` above ``-t_b`` plus
    ``exp(-b t_b) int exp(rho b (t + t_b))`` below it (2026-10-24)."""
    b, lo, hi = np.broadcast_arrays(np.asarray(b, dtype=float), np.asarray(t_lo, dtype=float), np.asarray(t_hi, dtype=float))
    xb = -float(t_b)
    a = np.maximum(lo, xb)
    c = np.minimum(hi, xb)
    upper = np.where(hi > a, _log_t_window_norm_slopes(b, a, hi), -np.inf)
    lower = np.where(c > lo, -b * float(t_b) + _log_t_window_norm_slopes(float(rho) * b, lo - xb, c - xb), -np.inf)
    return np.logaddexp(upper, lower)


def _log_two_slope_density(t, b, t_b: float, rho: float) -> np.ndarray:
    """The two-slope transfer's unnormalized ``log`` density at *t*."""
    t, b = np.broadcast_arrays(np.asarray(t, dtype=float), np.asarray(b, dtype=float))
    return np.where(t >= -float(t_b), b * t, -b * float(t_b) + float(rho) * b * (t + float(t_b)))


def _vertex_transfer_term_break(factor, t, t_lo, t_hi, b_sd, b0, t_break, trial=None) -> np.ndarray:
    """:func:`_vertex_transfer_term` for the two-slope transfer of the intact
    proton's vertex (2026-10-24): the log ratio of the normalized two-slope
    densities at the slopes ``b_sd * trial`` (``trial`` the recorded factor
    where it does not move) and ``b0 * factor``, the break fixed."""
    t_b, rho = _validated_t_break(t_break)
    f = np.asarray(factor, dtype=float)
    g = f if trial is None else np.asarray(trial, dtype=float)
    b1, b2 = b_sd * g, b0 * f
    return ((_log_two_slope_density(t, b1, t_b, rho) - _log_two_slope_norm(b1, t_lo, t_hi, t_b, rho))
            - (_log_two_slope_density(t, b2, t_b, rho) - _log_two_slope_norm(b2, t_lo, t_hi, t_b, rho)))


def _excited_remnant_terms(out, arrays, m1, m2, eps, eps0):
    """The excited remnants' mass-law terms of the non-diffractive class."""
    # The excited remnants' own mass law (2026-09-29): a high-mass
    # remnant mass was drawn from (M^2)^(-1-eps) on its window
    # [lo^2, hi^2] at zero transfer, so the exponent reweighting
    # carries one power-law ratio per excited kind.  The kind and
    # window columns are required: without them the ratio treats the
    # remnant masses as fixed inputs and is wrong for eps != eps0.
    missing = [k for k in ("k1", "k2", "lo1", "hi1", "lo2", "hi2") if k not in arrays]
    if missing:
        raise ValueError(
            "log_density_ratio: the non-diffractive arrays lack the remnant "
            f"kind and window columns {missing} (recorded since 2026-09-29; "
            "the excited remnants' mass law is part of the exponent "
            "reweighting)")
    # A line whose mass the records' line shapes redrew below its shape edge
    # Ms (2026-10-12, na49_mixing.nd_duality; the columns "ms1" and "ms2",
    # nan for a continuum mass) carries the continuum's share below Ms: its
    # term is the ratio of the power-law integrals on [lo^2, Ms^2].
    for kind_key, mass, lo_key, hi_key, ms_key in (("k1", m1, "lo1", "hi1", "ms1"), ("k2", m2, "lo2", "hi2", "ms2")):
        kind = np.asarray(arrays[kind_key], dtype=float)
        lo_v = np.asarray(arrays[lo_key], dtype=float)
        hi_v = np.asarray(arrays[hi_key], dtype=float)
        excited = (kind == EXCITED_REMNANT_KIND_INDEX) & (hi_v > lo_v)
        if np.any(excited):
            a2 = lo_v[excited] ** 2
            c2 = hi_v[excited] ** 2
            law = -(eps - eps0) * np.log(mass[excited] ** 2)
            if ms_key in arrays:
                ms_v = np.asarray(arrays[ms_key], dtype=float)[excited]
                shaped = np.isfinite(ms_v)
                if np.any(shaped):
                    e2 = ms_v[shaped] ** 2
                    law[shaped] = (log_power_integral_vec(eps, a2[shaped], e2)
                                   - log_power_integral_vec(eps0, a2[shaped], e2))
            out[excited] += law + (log_power_integral_vec(eps0, a2, c2) - log_power_integral_vec(eps, a2, c2))
    return out


# ----------------------------------------------------------------------
# The Pomeron's shrinkage in the diffractive classes (2026-11-01)
# ----------------------------------------------------------------------
# The triple-Pomeron term of the triple-Regge expansion at fixed transfer is
# d^2 sigma / dxi dt = G(t) s^eps xi^(-1 - eps - 2 alpha'_P t): the pure power
# (M^2)^(-1-eps) is its t = 0 form, and at t < 0 the factor xi^(-2 alpha' t) =
# exp(2 alpha' t ln(1/xi)) steepens the transfer law at small xi (the shrinkage
# of the diffraction peak, slope B + 2 alpha' ln(1/xi)).  The committed law
# draws M^2 from (M^2)^(-1-eps) and t from the vertex law h_B(t) normalized
# per mass, the zero-transfer form.  Under the option
# ``diffractive_pomeron_slope`` alpha' the vacuum events of the coherence
# region carry that law times the Regge factor:
#
#     p(M^2, t)  ~  (M^2)^(-1-eps) h_{B + c}(t) / Z_B(M^2),   c = 2 alpha' L,
#
# with L = ln(s m_p^2 / (M_1^2 M_2^2)) the rapidity gap of the 2 -> 2 final
# state (ln(1/xi) for an intact proton, clipped at zero where the two
# double-diffraction systems leave no gap), h_{B+c} the vertex law with every
# slope raised by c (the intact proton's two slopes B and rho B become B + c
# and rho B + c) and Z_B the vertex law's committed normalization on the
# event's transfer window.  The mass marginal gains the factor
# R(M^2) = Z_{B+c} / Z_B <= 1, and at alpha' = 0 the law is the committed one.
# The option ``diffractive_pomeron_linear_t`` t_lin holds the slope alpha'
# down to t = -t_lin and the exponent constant below it (the factor
# xi^(-2 alpha' max(t, -t_lin))): the raised slope B + c above -t_lin, the
# committed slope below (rho B for the intact proton, whose break t_b must then
# equal t_lin; B for every other vertex law).

#: The Gauss-Legendre orders of the shrinkage law's mass normalization (single
#: diffraction in ln M^2, double diffraction in ln M_1^2 and ln M_2^2).
_POMERON_GL_ORDER: int = 256
_POMERON_GL_ORDER_DD: int = 96
_POMERON_GRID_CACHE: dict = {}
_POMERON_GRID_CACHE_MAX: int = 64


def pomeron_gap_log(m2_1, m2_2, sqrt_s: float = PINNED_SQRT_S_EXACT) -> np.ndarray:
    """``L = max(0, ln(s m_p^2 / (M_1^2 M_2^2)))`` elementwise: the rapidity
    gap of a diffractive 2 -> 2 final state with the outgoing masses squared
    *m2_1*, *m2_2* [GeV^2] (``ln(s / M^2)`` for single diffraction with an
    intact proton), zero where the two systems leave no gap."""
    a = np.asarray(m2_1, dtype=float)
    b = np.asarray(m2_2, dtype=float)
    s = float(sqrt_s) * float(sqrt_s)
    return np.maximum(np.log(s * PROTON_MASS * PROTON_MASS / (a * b)), 0.0)


def _log_pw_norm(s1, s2, t_lo, t_hi, t_b) -> np.ndarray:
    """``log`` of the integral on ``[t_lo, t_hi]`` of the vertex law with the
    slope *s1* above ``-t_b`` and the continuous slope *s2* below it
    (``exp(-s1 t_b + s2 (t + t_b))``); *t_b* ``None`` is the single slope *s1*.
    Elementwise; the slopes positive."""
    s1, s2, lo, hi = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (s1, s2, t_lo, t_hi)))
    if t_b is None:
        return _log_t_window_norm_slopes(s1, lo, hi)
    tb = float(t_b)
    xb = -tb
    a = np.maximum(lo, xb)
    c = np.minimum(hi, xb)
    upper = np.where(hi > a, _log_t_window_norm_slopes(s1, a, hi), -np.inf)
    lower = np.where(c > lo, -s1 * tb + _log_t_window_norm_slopes(s2, lo - xb, c - xb), -np.inf)
    return np.logaddexp(upper, lower)


def _log_pw_density(t, s1, s2, t_b) -> np.ndarray:
    """The vertex law's unnormalized ``log`` density at *t* (see :func:`_log_pw_norm`)."""
    t, s1, s2 = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (t, s1, s2)))
    if t_b is None:
        return s1 * t
    tb = float(t_b)
    return np.where(t >= -tb, s1 * t, -s1 * tb + s2 * (t + tb))


def _pomeron_slopes(big_b, c, t_break, t_lin):
    """``(s1, s2, t_x)`` of the shrinkage law's vertex law at the committed
    slope *big_b* and the raise *c*: above ``-t_x`` the slope *s1*, below it
    the continuous slope *s2* (``t_x`` ``None``: the single slope *s1*).  The
    committed law's break *t_break* ``(t_b, rho)`` and the linear range *t_lin*
    (``None``: the raise applies at every t) set the pieces; a break and a
    linear range must coincide."""
    if t_lin is None:
        if t_break is None:
            return big_b + c, big_b + c, None
        return big_b + c, float(t_break[1]) * big_b + c, float(t_break[0])
    if t_break is None:
        return big_b + c, big_b, float(t_lin)
    if float(t_break[0]) != float(t_lin):
        raise ValueError(f"the Pomeron's linear range {t_lin!r} must equal the transfer break {t_break!r}")
    return big_b + c, float(t_break[1]) * big_b, float(t_lin)


def _sd_slope_factors(m2, m_rem: float, kappa, f_term: float) -> np.ndarray:
    """The factorized vertex slope factor of single-diffraction masses squared
    *m2* beside the remnant *m_rem* (:func:`vertex_slope_factor` elementwise
    plus the intact proton's term *f_term*); *kappa* ``None`` is the single
    slope (factor 1)."""
    m2 = np.asarray(m2, dtype=float)
    if kappa is None:
        return np.ones(m2.shape)
    k = float(kappa)
    r1 = PROTON_MASS / np.maximum(np.sqrt(m2), PROTON_MASS)
    r2 = PROTON_MASS / max(float(m_rem), PROTON_MASS)
    return 0.5 * (np.power(r1, k) + math.pow(r2, k)) + float(f_term)


def sd_pomeron_log_factor(m2, m_rem: float, b: float, kappa, f_term: float, alpha: float,
                          t_break=None, sqrt_s: float = PINNED_SQRT_S_EXACT, t_lin=None) -> np.ndarray:
    """``log R(M^2) = log Z_{B+c} - log Z_B`` of the single-diffraction
    shrinkage law (the module section above) at the masses squared *m2* beside
    the remnant *m_rem*: the vertex slope ``B = b`` times the slope factor
    (:func:`_sd_slope_factors`), ``c = 2 alpha' ln(s m_p^2 / (M^2 m_rem^2))``,
    the intact proton's two slopes under *t_break* ``(t_b, rho)``, the
    linear range *t_lin* (:func:`_pomeron_slopes`)."""
    m2 = np.asarray(m2, dtype=float)
    mr2 = float(m_rem) * float(m_rem)
    big_b = float(b) * _sd_slope_factors(m2, m_rem, kappa, f_term)
    c = 2.0 * float(alpha) * pomeron_gap_log(m2, mr2, sqrt_s)
    rho, tb = (1.0, None) if t_break is None else (float(t_break[1]), float(t_break[0]))
    lo, hi = pp_t_window_vec(m2, np.full(m2.shape, mr2), sqrt_s)
    s1, s2, tx = _pomeron_slopes(big_b, c, t_break, t_lin)
    return _log_pw_norm(s1, s2, lo, hi, tx) - _log_pw_norm(big_b, rho * big_b, lo, hi, tb)


def draw_marginal_pomeron(m2_min: float, m2_max: float, epsilon: float, b: float, m_rem: float, kappa, f_term: float,
                          alpha: float, t_break, rng: np.random.Generator, t_lin=None) -> float:
    """One ``M^2`` draw of the single-diffraction shrinkage law's marginal
    ``(M^2)^(-1-eps) R(M^2)`` on ``[m2_min, m2_max]``: the geometric 4096-point
    grid of :func:`draw_marginal` (uniform in ``ln M^2``, trapezoid cells, the
    grid cached per parameter point), exactly 2 uniforms from *rng*."""
    key = (float(m2_min), float(m2_max), float(epsilon), float(b), float(m_rem),
           None if kappa is None else float(kappa), float(f_term), float(alpha),
           None if t_break is None else (float(t_break[0]), float(t_break[1])),
           None if t_lin is None else float(t_lin))
    hit = _POMERON_GRID_CACHE.get(key)
    if hit is None:
        lo, hi = key[0], key[1]
        if not (math.isfinite(lo) and math.isfinite(hi)) or not hi > lo > 0.0:
            raise ValueError(f"draw_marginal_pomeron: the window must satisfy 0 < m2_min < m2_max, got ({m2_min!r}, {m2_max!r})")
        ys = np.linspace(math.log(lo), math.log(hi), _MARGINAL_GRID_POINTS)
        g = np.exp(-float(epsilon) * ys + sd_pomeron_log_factor(np.exp(ys), m_rem, b, kappa, f_term, alpha, t_break,
                                                                t_lin=t_lin))
        cell = 0.5 * (g[1:] + g[:-1]) * (ys[1:] - ys[:-1])
        total = float(cell.sum())
        if not (math.isfinite(total) and total > 0.0):
            raise ValueError(f"draw_marginal_pomeron: the marginal carries no mass on ({m2_min!r}, {m2_max!r})")
        if len(_POMERON_GRID_CACHE) >= _POMERON_GRID_CACHE_MAX:
            _POMERON_GRID_CACHE.clear()
        hit = (ys, np.cumsum(cell) / total)
        _POMERON_GRID_CACHE[key] = hit
    ys, cdf = hit
    idx = int(np.searchsorted(cdf, rng.random(), side="right"))
    if idx >= cdf.size:
        idx = cdf.size - 1
    y_value = ys[idx] + rng.random() * (ys[idx + 1] - ys[idx])
    return float(min(max(math.exp(y_value), key[0]), key[1]))


def draw_t_pomeron(t_lo: float, t_hi: float, big_b: float, c: float, t_break, rng: np.random.Generator,
                   t_lin=None) -> float:
    """One t draw of the shrinkage law's vertex law on ``[t_lo, t_hi]``: the
    slope ``B + c``, under *t_break* ``(t_b, rho)`` the two slopes ``B + c``
    and ``rho B + c`` continuous at ``-t_b``, under the linear range *t_lin*
    the committed slope below ``-t_lin`` (:func:`_pomeron_slopes`); exactly 1
    uniform (the inverse CDF of :func:`draw_t_window` /
    :func:`draw_t_window_break`)."""
    if t_break is not None:
        _validated_t_break(t_break)
    s1, s2, tx = _pomeron_slopes(float(big_b), float(c), t_break, t_lin)
    params = MassLawParams(epsilon=0.0, b_sd=s1)
    if tx is None:
        return draw_t_window(t_lo, t_hi, params, rng)
    return draw_t_window_break(t_lo, t_hi, params, (tx, s2 / s1), rng)


def _gl_nodes(order: int, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(int(order))
    half = 0.5 * (b - a)
    return 0.5 * (b + a) + half * x, half * w


def log_sd_pomeron_mass_norm(epsilon: float, b: float, kappa, f_term: float, alpha: float, t_break,
                             m2_low: float, m2_max: float, m_rem: float, sqrt_s: float = PINNED_SQRT_S_EXACT,
                             t_lin=None) -> float:
    """``log`` of the single-diffraction shrinkage law's mass normalization
    ``int (M^2)^(-1-eps) R(M^2) dM^2`` on ``[m2_low, m2_max]`` (Gauss-Legendre
    in ``ln M^2``, order :data:`_POMERON_GL_ORDER`)."""
    ys, ws = _gl_nodes(_POMERON_GL_ORDER, math.log(float(m2_low)), math.log(float(m2_max)))
    log_g = -float(epsilon) * ys + sd_pomeron_log_factor(np.exp(ys), m_rem, b, kappa, f_term, alpha, t_break, sqrt_s,
                                                         t_lin=t_lin)
    peak = float(np.max(log_g))
    return peak + math.log(float(np.sum(ws * np.exp(log_g - peak))))


def dd_pomeron_log_factor(m2_1, m2_2, b: float, kappa, alpha: float, sqrt_s: float = PINNED_SQRT_S_EXACT,
                          t_lin=None) -> np.ndarray:
    """``log R = log Z_{B+c} - log Z_B`` of the double-diffraction shrinkage law
    at the masses squared *m2_1*, *m2_2* (one slope ``B = b`` times the pair's
    factorized vertex slope factor, ``c = 2 alpha' L``)."""
    a = np.asarray(m2_1, dtype=float)
    d = np.asarray(m2_2, dtype=float)
    a, d = np.broadcast_arrays(a, d)
    if kappa is None:
        factor = np.ones(a.shape)
    else:
        k = float(kappa)
        factor = 0.5 * (np.power(PROTON_MASS / np.maximum(np.sqrt(a), PROTON_MASS), k)
                        + np.power(PROTON_MASS / np.maximum(np.sqrt(d), PROTON_MASS), k))
    big_b = float(b) * factor
    c = 2.0 * float(alpha) * pomeron_gap_log(a, d, sqrt_s)
    lo, hi = pp_t_window_vec(a, d, sqrt_s)
    s1, s2, tx = _pomeron_slopes(big_b, c, None, t_lin)
    return _log_pw_norm(s1, s2, lo, hi, tx) - _log_t_window_norm_slopes(big_b, lo, hi)


def log_dd_pomeron_mass_norm(epsilon: float, b: float, kappa, alpha: float, m_lo1: float, m_lo2: float, m2_max: float,
                             sqrt_s: float = PINNED_SQRT_S_EXACT, t_lin=None) -> float:
    """``log`` of the double-diffraction shrinkage law's joint mass
    normalization ``int int (M_1^2 M_2^2)^(-1-eps) R dM_1^2 dM_2^2`` on
    ``[m_lo1^2, m2_max] x [m_lo2^2, m2_max]`` (the coherence edge
    ``m2_max <= 0.25 s`` keeps the closure; Gauss-Legendre in ``ln M_i^2``)."""
    y1, w1 = _gl_nodes(_POMERON_GL_ORDER_DD, 2.0 * math.log(float(m_lo1)), math.log(float(m2_max)))
    y2, w2 = _gl_nodes(_POMERON_GL_ORDER_DD, 2.0 * math.log(float(m_lo2)), math.log(float(m2_max)))
    g1, g2 = np.meshgrid(y1, y2, indexing="ij")
    log_g = -float(epsilon) * (g1 + g2) + dd_pomeron_log_factor(np.exp(g1), np.exp(g2), b, kappa, alpha, sqrt_s, t_lin)
    peak = float(np.max(log_g))
    return peak + math.log(float(np.sum(np.outer(w1, w2) * np.exp(log_g - peak))))


def _pomeron_log_ratio(cls: str, arrays: dict, sel: np.ndarray, eps: float, b_sd: float, eps0: float, b0: float,
                       pomeron: tuple, intact_factor, xi_max, t_break, root_s: float) -> np.ndarray:
    """The importance weight of the shrinkage law's events *sel* between two
    parameter points (the trial ``(eps, b_sd)`` and the generation ``(eps0,
    b0)``; *pomeron* ``(alpha', kappa, f[, t_lin])`` with ``f`` the intact
    vertex factor when it is fixed, ``None`` without it, and t_lin the linear
    range, ``None`` or absent for every t)."""
    alpha, kappa, f_fixed = float(pomeron[0]), pomeron[1], pomeron[2]
    t_lin = pomeron[3] if len(pomeron) > 3 else None
    if cls in ("sd_projectile", "sd_target"):
        m2 = np.asarray(arrays["m2"], dtype=float)[sel]
        t = np.asarray(arrays["t"], dtype=float)[sel]
        m_rem = np.asarray(arrays["m_rem"], dtype=float)[sel]
        factor = np.asarray(arrays["slope_factor"], dtype=float)[sel] if "slope_factor" in arrays else np.ones(m2.shape)
        intact = m_rem <= PROTON_MASS * (1.0 + 1e-9)
        if intact_factor is not None:
            f, f0 = (float(v) for v in intact_factor)
        else:
            f = f0 = None if f_fixed is None else float(f_fixed)
        trial = factor + (0.5 * (f - f0) * intact if (f is not None and f0 is not None) else 0.0)
        c = 2.0 * alpha * pomeron_gap_log(m2, m_rem * m_rem)
        t_lo, t_hi = pp_t_window_vec(m2, m_rem * m_rem)
        out = np.empty(m2.shape)
        m2_max = (root_s - m_rem) ** 2
        if xi_max is not None:
            m2_max = np.minimum(m2_max, float(xi_max) * root_s * root_s)
        low = (np.asarray(arrays["m2_low"], dtype=float)[sel] if "m2_low" in arrays
               else np.full(m2.shape, EXCITATION_M_MIN * EXCITATION_M_MIN))
        # the intact proton's two slopes under t_break, the single slope otherwise
        groups = ([(intact, (float(t_break[0]), float(t_break[1]))), (~intact, None)] if t_break is not None
                  else [(np.ones(m2.shape, dtype=bool), None)])
        for pick, tbr in groups:
            if not np.any(pick):
                continue
            tb, rho = (tbr[0], tbr[1]) if tbr is not None else (None, 1.0)
            b_t, b_g = b_sd * trial[pick], b0 * factor[pick]
            ct, tt, lo, hi = c[pick], t[pick], t_lo[pick], t_hi[pick]
            s1t, s2t, txt = _pomeron_slopes(b_t, ct, tbr, t_lin)
            s1g, s2g, txg = _pomeron_slopes(b_g, ct, tbr, t_lin)
            out[pick] = ((_log_pw_density(tt, s1t, s2t, txt) - _log_pw_norm(b_t, rho * b_t, lo, hi, tb))
                         - (_log_pw_density(tt, s1g, s2g, txg) - _log_pw_norm(b_g, rho * b_g, lo, hi, tb)))
        out += -(eps - eps0) * np.log(m2)
        # the mass normalization per (lower edge, remnant, upper edge) group
        keys = np.stack([low, m_rem, m2_max], axis=1)
        uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
        norm = np.empty(len(uniq))
        for k, (lo2, mr, hi2) in enumerate(uniq):
            is_p = mr <= PROTON_MASS * (1.0 + 1e-9)
            tb_g = tuple(t_break) if (t_break is not None and is_p) else None
            ft = (0.5 * (f - 1.0) if (is_p and f is not None) else 0.0) if kappa is not None else 0.0
            ft0 = (0.5 * (f0 - 1.0) if (is_p and f0 is not None) else 0.0) if kappa is not None else 0.0
            # the transfer windows and gaps at the exact sqrt(s), the mass windows at the caller's (as the draws)
            norm[k] = (log_sd_pomeron_mass_norm(eps, b_sd, kappa, ft, alpha, tb_g, lo2, hi2, mr, t_lin=t_lin)
                       - log_sd_pomeron_mass_norm(eps0, b0, kappa, ft0, alpha, tb_g, lo2, hi2, mr, t_lin=t_lin))
        return out - norm[np.asarray(inverse).reshape(-1)]
    if cls == "dd":
        if xi_max is None or float(xi_max) > 0.25:
            raise ValueError("log_density_ratio: the double-diffraction shrinkage law needs the coherence edge xi_max <= 0.25")
        m12 = np.asarray(arrays["m12"], dtype=float)[sel]
        m22 = np.asarray(arrays["m22"], dtype=float)[sel]
        t = np.asarray(arrays["t1"], dtype=float)[sel]
        factor = np.asarray(arrays["slope_factor"], dtype=float)[sel] if "slope_factor" in arrays else np.ones(m12.shape)
        c = 2.0 * alpha * pomeron_gap_log(m12, m22)
        t_lo, t_hi = pp_t_window_vec(m12, m22)
        b_t, b_g = b_sd * factor, b0 * factor
        s1t, s2t, txt = _pomeron_slopes(b_t, c, None, t_lin)
        s1g, s2g, txg = _pomeron_slopes(b_g, c, None, t_lin)
        out = ((_log_pw_density(t, s1t, s2t, txt) - _log_t_window_norm_slopes(b_t, t_lo, t_hi))
               - (_log_pw_density(t, s1g, s2g, txg) - _log_t_window_norm_slopes(b_g, t_lo, t_hi))
               - (eps - eps0) * (np.log(m12) + np.log(m22)))
        hi2 = min((root_s - PROTON_MASS) ** 2, float(xi_max) * root_s * root_s)
        lo1 = (np.asarray(arrays["m_lo1"], dtype=float)[sel] if "m_lo1" in arrays else np.full(m12.shape, EXCITATION_M_MIN))
        lo2 = (np.asarray(arrays["m_lo2"], dtype=float)[sel] if "m_lo2" in arrays else np.full(m12.shape, EXCITATION_M_MIN))
        keys = np.stack([lo1, lo2], axis=1)
        uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
        norm = np.asarray([log_dd_pomeron_mass_norm(eps, b_sd, kappa, alpha, a, d, hi2, t_lin=t_lin)
                           - log_dd_pomeron_mass_norm(eps0, b0, kappa, alpha, a, d, hi2, t_lin=t_lin) for a, d in uniq])
        return out - norm[np.asarray(inverse).reshape(-1)]
    raise ValueError(f"log_density_ratio: the shrinkage law covers the diffractive classes, got {cls!r}")
