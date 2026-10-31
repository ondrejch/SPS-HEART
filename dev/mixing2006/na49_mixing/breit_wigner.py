"""p-wave Breit-Wigner mass distribution.

Ported from ``TBW.h`` / ``TBW.cxx``.

The ``BreitWigner`` class pre-computes a 500-bin histogram of the BW
distribution and draws random masses from it via the inverse-CDF method
(avoiding ROOT's ``FillRandom`` / ``GetRandom``).
"""

from __future__ import annotations
import math
import numpy as np
from typing import Any, Dict, Optional, Tuple

from .histogram import Histogram1D


# Default BW parameters for the Δ(1232) resonance
_DEFAULT_PARAMS = (1.07, 1.232, 0.12, 1.0, 2.0, 2.5)

_PARAM_NAMES = ("schwel", "rmas", "brei", "pbrei", "dumplow", "dumphigh")


def _validate_params(
    schwel: float,
    rmas: float,
    brei: float,
    pbrei: float,
    dumplow: float,
    dumphigh: float,
) -> None:
    """Validate a full Breit-Wigner parameter set; raise ``ValueError`` if invalid.

    A zero nominal width ``brei`` is legal (it yields the delta-at-pole
    behavior described in the class docstring); only negative widths are
    rejected.
    """
    vals = (schwel, rmas, brei, pbrei, dumplow, dumphigh)
    for nm, v in zip(_PARAM_NAMES, vals):
        v = float(v)
        if not math.isfinite(v):
            raise ValueError(
                f"BreitWigner parameter {nm}={v!r} must be finite"
            )
    if not float(brei) >= 0.0:
        raise ValueError(
            f"BreitWigner width brei={brei!r} must be non-negative "
            "(zero width is allowed and samples as a delta at rmas)"
        )
    if not float(rmas) > float(schwel):
        raise ValueError(
            f"BreitWigner pole mass rmas={rmas!r} must be greater than "
            f"threshold schwel={schwel!r}"
        )
    if not float(dumphigh) > float(schwel):
        raise ValueError(
            f"BreitWigner upper cutoff dumphigh={dumphigh!r} must be greater "
            f"than threshold schwel={schwel!r}"
        )
    if not float(dumphigh) > float(dumplow):
        raise ValueError(
            f"BreitWigner damping window requires dumplow < dumphigh, got "
            f"dumplow={dumplow!r}, dumphigh={dumphigh!r}"
        )


def bw_new(x: float, params) -> float:
    """Evaluate the p-wave Breit-Wigner at invariant mass *x*.

    Parameters
    ----------
    x : float
        Invariant mass (GeV/c²).
    params : sequence of 6 floats
        ``[schwel, rmas, brei, pbrei, dumplow, dumphigh]``
        - schwel   — threshold (GeV/c²)
        - rmas     — nominal resonance mass (GeV/c²)
        - brei     — nominal width (GeV)
        - pbrei    — p-wave phase-space factor (usually 1)
        - dumplow  — start of linear damping region (GeV/c²)
        - dumphigh — upper cutoff of distribution (GeV/c²)
    """
    schwel, rmas, brei, pbrei, dumplow, dumphigh = params

    if x <= schwel:
        return 0.0

    gamma = (
        brei
        * ((x - schwel) / (rmas - schwel)) ** 3
        * 2.0 * (rmas - schwel) * (rmas - schwel)
    )
    gamma /= (x - schwel) ** 2 + (rmas - schwel) ** 2

    bwdist = (x * rmas * gamma * pbrei) / (
        (x * x - rmas * rmas) ** 2 + rmas * rmas * gamma * gamma
    )

    if x >= dumplow:
        bwdist *= (dumphigh - x) / (dumphigh - dumplow)
        if x >= dumphigh:
            bwdist = 0.0

    return max(bwdist, 0.0)


def bw_new_array(xs: np.ndarray, params) -> np.ndarray:
    """Vectorised version of :func:`bw_new` over a NumPy array *xs*."""
    schwel, rmas, brei, pbrei, dumplow, dumphigh = params
    x = np.asarray(xs, dtype=np.float64)
    result = np.zeros_like(x)

    mask = x > schwel
    xm = x[mask]
    g = (
        brei
        * ((xm - schwel) / (rmas - schwel)) ** 3
        * 2.0 * (rmas - schwel) ** 2
    )
    g /= (xm - schwel) ** 2 + (rmas - schwel) ** 2

    bw = (xm * rmas * g * pbrei) / (
        (xm * xm - rmas * rmas) ** 2 + rmas * rmas * g * g
    )

    # damping region
    damp_mask = xm >= dumplow
    bw[damp_mask] *= (dumphigh - xm[damp_mask]) / (dumphigh - dumplow)
    bw[xm >= dumphigh] = 0.0
    bw = np.maximum(bw, 0.0)
    result[mask] = bw
    return result


class BreitWigner:
    """p-wave Breit-Wigner distribution with histogram-based sampling.

    A Breit-Wigner whose total density is zero (for example a zero nominal
    width ``brei``, which tower-width overrides produce legitimately)
    represents a delta distribution: every draw on both the scalar
    (:meth:`get_random`) and the vectorised (:meth:`get_random_n`) path
    returns the pole mass ``rmas`` exactly. The generic ``Histogram1D``
    empty-distribution midpoint is deliberately not used, because the
    midpoint of the sampling support differs from the pole mass.

    Parameters
    ----------
    schwel : float
        Threshold mass (GeV/c²).
    rmas : float
        Nominal resonance mass (GeV/c²).
    brei : float
        Nominal width (GeV).
    pbrei : float
        p-wave phase-space factor.
    dumplow : float
        Start of the linear dumping region (GeV/c²).
    dumphigh : float
        Upper cutoff (GeV/c²).
    nbins : int
        Number of bins in the pre-computed histogram (default 500).
    n_sample : int
        Number of points used to fill the histogram (default 1 000 000).
    rng : numpy.random.Generator, optional
        Random-number generator to use.

    Raises
    ------
    ValueError
        If any parameter is non-finite, ``brei`` is negative, ``rmas``
        equals or falls below ``schwel``, ``dumphigh`` equals or falls
        below ``schwel``, or ``dumplow >= dumphigh``. A zero width
        ``brei == 0`` is explicitly valid (delta distribution at rmas).
    """

    def __init__(
        self,
        schwel: float = _DEFAULT_PARAMS[0],
        rmas: float = _DEFAULT_PARAMS[1],
        brei: float = _DEFAULT_PARAMS[2],
        pbrei: float = _DEFAULT_PARAMS[3],
        dumplow: float = _DEFAULT_PARAMS[4],
        dumphigh: float = _DEFAULT_PARAMS[5],
        nbins: int = 500,
        n_sample: int = 1_000_000,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        _validate_params(schwel, rmas, brei, pbrei, dumplow, dumphigh)
        self._params = [float(schwel), float(rmas), float(brei),
                        float(pbrei), float(dumplow), float(dumphigh)]
        self._nbins = nbins
        self._n_sample = n_sample
        # Pole mass of the delta draw when the distribution has zero total
        # density; ``None`` for an ordinary (sampleable) distribution. Set
        # by _build_hist().
        self._delta_mass: Optional[float] = None
        # Cached total line-shape integral over [schwel, dumphigh]; reset by
        # _build_hist() whenever the parameters change. Used by the
        # truncated, explicitly renormalized draw helpers below.
        self._full_integral: Optional[float] = None
        # Cached above-threshold fractions of bw_fraction_above(), keyed on
        # (lower, upper_cap); reset by _build_hist() whenever the parameters
        # change, same discipline as _full_integral above. Pure compute
        # cache for the parent-side openness audit: a stored value is the
        # bit-identical result of the one grid evaluation, and the
        # primitive consumes no random numbers either way.
        self._fraction_above_cache: Dict[Tuple[float, Optional[float]], float] = {}
        self._rng: np.random.Generator = rng if rng is not None else np.random.default_rng()
        self._hist: Histogram1D = self._build_hist()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _build_hist(self) -> Histogram1D:
        """Pre-fill a histogram via inverse-CDF sampling of the BW PDF."""
        schwel = self._params[0]
        dumphigh = self._params[5]
        # Evaluate BW on a fine grid and build a PDF
        xs = np.linspace(schwel, dumphigh, 10_000)
        ys = bw_new_array(xs, self._params)
        total = ys.sum()
        h = Histogram1D(self._nbins, schwel, dumphigh)
        if total > 0.0:
            # Directly set bin contents proportional to the BW value
            # (much faster than rejection sampling)
            nbins = self._nbins
            bin_width = (dumphigh - schwel) / nbins
            bin_xs = schwel + (np.arange(nbins) + 0.5) * bin_width
            bin_ys = bw_new_array(bin_xs, self._params)
            h._contents[1:nbins + 1] = np.maximum(bin_ys, 0.0)
            h._cdf = None
            self._delta_mass = None
        else:
            # Zero total density: act as a delta distribution at the pole
            # mass rmas. Recorded explicitly so both sampling paths return
            # rmas exactly instead of falling back to the histogram
            # midpoint policy.
            self._delta_mass = float(self._params[1])
        # Parameter changes rebuild the histogram; the cached total
        # integral must follow.
        self._full_integral = None
        # ... and the memoized above-threshold fractions with it.
        self._fraction_above_cache = {}
        return h

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def get_value(self, mass: float) -> float:
        """Evaluate the BW distribution at *mass*."""
        return bw_new(mass, self._params)

    def get_random(self) -> float:
        """Sample one random mass from the BW distribution.

        For zero total density this returns the pole mass exactly on every
        call (delta distribution).
        """
        if self._delta_mass is not None:
            return self._delta_mass
        return self._hist.get_random(self._rng)

    def get_random_n(self, n: int) -> np.ndarray:
        """Sample *n* random masses at once.

        For zero total density every element equals the pole mass,
        consistent with :meth:`get_random`.
        """
        if self._delta_mass is not None:
            return np.full(n, self._delta_mass, dtype=np.float64)
        return self._hist.get_random_n(n, self._rng)

    def set_parameters(
        self,
        schwel: float,
        rmas: float,
        brei: float,
        pbrei: float,
        dumplow: float,
        dumphigh: float,
    ) -> None:
        """Update all BW parameters and rebuild the histogram.

        The full parameter set is validated before anything is assigned, so
        a rejected update leaves the object unchanged. This is also the
        atomic entry point for callers that need to move several parameters
        at once (e.g. mass/width overrides that would pass through
        transiently inconsistent intermediate states one index at a time).
        """
        _validate_params(schwel, rmas, brei, pbrei, dumplow, dumphigh)
        self._params = [float(schwel), float(rmas), float(brei),
                        float(pbrei), float(dumplow), float(dumphigh)]
        self._hist = self._build_hist()

    def set_parameter(self, i: int, val: float) -> None:
        """Set parameter at index *i* (0-based) and rebuild.

        The candidate parameter set is validated as a whole before the
        assignment; a rejected update leaves the object unchanged.
        """
        candidate = list(self._params)
        candidate[i] = float(val)
        _validate_params(*candidate)
        self._params = candidate
        self._hist = self._build_hist()

    def get_parameter(self, i: int) -> float:
        """Return parameter at index *i* (0-based)."""
        return self._params[i]

    def full_support_integral(self) -> float:
        """Total line-shape integral over ``[schwel, dumphigh]`` (cached).

        Used as the normalization reference of the truncated,
        explicitly renormalized draw helpers below. The cache is reset by
        :meth:`_build_hist` whenever the parameters change, so it always
        reflects the committed parameter set.
        """
        value = self._full_integral
        if value is None:
            lo = self._params[0]
            hi = self._params[5]
            xs = np.linspace(lo, hi, 4096)
            ys = bw_new_array(xs, self._params)
            value = _trapezoid_mass(xs, ys)
            self._full_integral = value
        return value

    @property
    def params(self):
        """All 6 BW parameters as a list."""
        return list(self._params)

    def __repr__(self) -> str:
        return f"BreitWigner(params={self._params})"

    def __str__(self) -> str:
        return " ".join(str(p) for p in self._params)

    @classmethod
    def from_string(cls, s: str, **kwargs: Any) -> "BreitWigner":
        """Construct from a space-separated string of 6 parameters.

        Tokens are converted left to right, so a malformed number raises
        exactly as it does for the leading parameters. Short parameter
        strings rely on the constructor defaults for the missing trailing
        values; extra tokens beyond the six BW parameters are converted
        and ignored.
        """
        vals = [float(v) for v in s.split()]
        # Variadic-construction contract: the ``*vals[:6]`` splat carries a
        # statically unknown length (0 to 6 positionals). mypy checks the
        # worst-case tail against the trailing constructor parameters
        # (``nbins``, ``n_sample``, ``rng``) and reports arg-type. Runtime
        # semantics intentionally rest on the constructor defaults for
        # short lists, so the splat is kept verbatim and suppressed at
        # this single site instead of reshaping the public signature.
        return cls(*vals[:6], **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Truncated, explicitly renormalized line-shape draws
# ---------------------------------------------------------------------------
#
# The source-constrained decay machinery (card TASK-2026-08-31, decision D1)
# samples resonance daughter masses from the line shape restricted to the
# kinematically allowed interval [m_threshold, min(M_source, M_match)], with
# the truncation normalized BY CONSTRUCTION: the draw is the inverse CDF of
# the line shape integrated over the window, not a full draw followed by
# rejection. A window carrying no line-shape mass makes the state ineligible.

_TRUNCATED_DRAW_GRID = 1024


def _trapezoid_mass(xs: np.ndarray, ys: np.ndarray) -> float:
    """Trapezoid integral of *ys* over the abscissae *xs* (finite part)."""
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.size < 2:
        return 0.0
    return float((0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])).sum())


def bw_draw_window(bw: "BreitWigner", upper_cap: float) -> Optional[tuple]:
    """Legal draw window for a Breit-Wigner daughter mass, or ``None``.

    The window is ``[schwel, min(upper_cap, dumphigh)]`` for an ordinary
    (sampleable) line shape. A zero-total-density Breit-Wigner — the legal
    zero-width configuration, e.g. ``brei == 0`` produced by clamped
    tower-width overrides — is a delta at the pole: the window collapses
    to the single point ``(pole, pole)`` when the pole lies inside the
    cap, and the state is ineligible (``None``) when it does not.

    Returns ``None`` when the truncated line-shape integral vanishes.
    """
    upper = float(upper_cap)
    delta_mass = getattr(bw, "_delta_mass", None)
    if delta_mass is not None:
        if float(delta_mass) <= upper:
            return (float(delta_mass), float(delta_mass))
        return None
    schwel = float(bw.get_parameter(0))
    dumphigh = float(bw.get_parameter(5))
    hi = min(upper, dumphigh)
    if not hi > schwel:
        return None
    return (schwel, hi)


def bw_truncated_fraction(bw: "BreitWigner", lo: float, hi: float) -> float:
    """Fraction of the line-shape integral inside the window ``[lo, hi]``.

    Delta-aware: a zero-total-density Breit-Wigner returns 1.0 when its
    pole lies inside the window and 0.0 otherwise. For an ordinary line
    shape the fraction is the trapezoid integral over the clipped window
    divided by the cached total integral over the full support; this is
    the openness fraction of a channel that carries this daughter.

    Documented sub-tolerance grid biases (review REV-fc520a5-02, comment
    only): the numerator is evaluated on a fixed 1024-point grid while
    the cached denominator (``full_support_integral``) uses 4096 points,
    so the fraction carries an order-1e-3 grid-mismatch error. That is
    orders of magnitude below every guard floor in use (the stage-B
    openness guard floor is 0.05), so the mixed grid sizes are kept for
    stability of the cached denominator.
    """
    delta_mass = getattr(bw, "_delta_mass", None)
    if delta_mass is not None:
        return 1.0 if float(lo) <= float(delta_mass) <= float(hi) else 0.0
    schwel = float(bw.get_parameter(0))
    dumphigh = float(bw.get_parameter(5))
    lo_c = max(float(lo), schwel)
    hi_c = min(float(hi), dumphigh)
    if not hi_c > lo_c:
        return 0.0
    if float(lo) <= schwel and float(hi) >= dumphigh:
        # The window covers the full sampling support: the truncated
        # integral IS the total integral.
        return 1.0
    total = bw.full_support_integral()
    if not total > 0.0:
        return 0.0
    xs = np.linspace(lo_c, hi_c, 1024)
    ys = bw_new_array(xs, bw.params)
    return _trapezoid_mass(xs, ys) / total


def bw_fraction_above(
    bw: "BreitWigner",
    lower: float,
    upper_cap: Optional[float] = None,
) -> float:
    """Fraction of the line-shape integral above *lower* (GeV/c²).

    Parent-side openness primitive of the stage-B machinery: for a
    channel whose minimum daughter-mass sum is *lower*, this is the
    fraction of the parent line shape that admits the channel (card
    decision D1's peak-openness diagnostic; see
    ``na49_mixing.decays.channel_parent_openness``). The optional
    *upper_cap* clips the integral from above (the resolved-region cap).

    Delta-aware: a zero-total-density Breit-Wigner returns 1.0 when its
    pole lies at or above *lower* and 0.0 otherwise. Numerator and
    denominator are evaluated on the same fixed 1024-point grid, so the
    fraction is grid-consistent by construction (the daughter-slot helper
    :func:`bw_truncated_fraction` mixes 1024- and 4096-point grids; see
    its documented sub-tolerance bias note).

    The grid result is memoized per line-shape instance, keyed on
    ``(lower, upper_cap)``; the cache is reset by ``_build_hist()``
    whenever the parameters change (the same discipline as the cached
    ``full_support_integral``), so a parameter update invalidates the
    stored fractions automatically and repeat calls are O(1). The audit
    path calls this once per primary decay attempt at a fixed parameter
    set, which is where the memo pays (review REV-5b2f37d-01). The memo
    is a pure compute cache: a stored value is the bit-identical result
    of the one grid evaluation, and the primitive consumes no random
    numbers on either path, so no sampling stream can change.
    """
    delta_mass = getattr(bw, "_delta_mass", None)
    if delta_mass is not None:
        return 1.0 if float(delta_mass) >= float(lower) else 0.0
    schwel = float(bw.get_parameter(0))
    dumphigh = float(bw.get_parameter(5))
    lo = max(float(lower), schwel)
    hi = dumphigh if upper_cap is None else min(float(upper_cap), dumphigh)
    if not hi > lo:
        return 0.0
    if float(lower) <= schwel and (upper_cap is None or float(upper_cap) >= dumphigh):
        return 1.0
    key = (float(lower), None if upper_cap is None else float(upper_cap))
    cache = getattr(bw, "_fraction_above_cache", None)
    if cache is not None and key in cache:
        return cache[key]
    xs = np.linspace(lo, hi, 1024)
    ys = np.maximum(bw_new_array(xs, bw.params), 0.0)
    numerator = _trapezoid_mass(xs, ys)
    if not numerator > 0.0:
        if cache is not None:
            cache[key] = 0.0
        return 0.0
    xs_full = np.linspace(schwel, dumphigh, 1024)
    ys_full = np.maximum(bw_new_array(xs_full, bw.params), 0.0)
    total = _trapezoid_mass(xs_full, ys_full)
    if not total > 0.0:
        return 0.0
    fraction = numerator / total
    if cache is not None:
        cache[key] = fraction
    return fraction


def draw_truncated_bw(
    bw: "BreitWigner",
    rng: np.random.Generator,
    lo: float,
    hi: float,
) -> Optional[float]:
    """Draw one mass from the line shape restricted to ``[lo, hi]``.

    The truncated density is normalized by construction: the draw inverts
    the CDF of the line shape integrated over the window (trapezoid cell
    masses on a fixed 1024-point grid), so no full draw followed by
    rejection is involved. Two uniform random numbers are consumed,
    mirroring the histogram sampler's bin-selection plus within-bin pair,
    which keeps the per-iteration draw count stable across the two paths.

    Documented sub-tolerance grid bias (review REV-fc520a5-02, comment
    only): within a CDF cell the draw is uniform, which approximates the
    piecewise-linear density interpolant only to first order in the
    1024-point grid. The resulting bias is orders of magnitude below the
    statistical tolerances of every consumer (the density-following test
    in the acceptance suite passes with wide margin).

    A zero-total-density Breit-Wigner (delta at the pole) returns the pole
    exactly and consumes no random numbers when the pole lies inside the
    window; it returns ``None`` when it does not, because the state is
    then ineligible.

    Returns ``None`` when the window carries no line-shape mass.
    """
    delta_mass = getattr(bw, "_delta_mass", None)
    if delta_mass is not None:
        if float(lo) <= float(delta_mass) <= float(hi):
            return float(delta_mass)
        return None
    schwel = float(bw.get_parameter(0))
    dumphigh = float(bw.get_parameter(5))
    lo_c = max(float(lo), schwel)
    hi_c = min(float(hi), dumphigh)
    if not hi_c > lo_c:
        return None
    xs = np.linspace(lo_c, hi_c, _TRUNCATED_DRAW_GRID)
    ys = np.maximum(bw_new_array(xs, bw.params), 0.0)
    total = _trapezoid_mass(xs, ys)
    if not (math.isfinite(total) and total > 0.0):
        return None
    # Per-cell masses of the piecewise-linear density interpolant.
    cell = 0.5 * (ys[1:] + ys[:-1]) * (xs[1:] - xs[:-1])
    cdf = np.cumsum(cell) / total
    u_segment = rng.random()
    idx = int(np.searchsorted(cdf, u_segment, side="right"))
    if idx >= cdf.size:
        idx = cdf.size - 1
    u_within = rng.random()
    mass = lo_c + (idx + u_within) * (hi_c - lo_c) / cdf.size
    return float(min(max(mass, lo_c), hi_c))
