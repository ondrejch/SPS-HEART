# The legacy p+A path (marked 2026-11-04): the HEART p+p fits do not run this module; the p+A builder
# (event_builder_pA, parallel, glauber) and its imports do. Code reviews of the p+p model can skip it; review it
# with the p+A work. The cleanup and refactor after the p+A run decides its future.
"""Custom 1-D histogram with fast inverse-CDF sampling.

Ported from ``TH1Dour.h`` / ``TH1Dour.cxx`` (which extends ROOT's ``TH1D``).
ROOT's ``TH1D`` is replaced by a plain NumPy array + scipy-based sampling.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional, Protocol


def _nonfinite_rejection(
    method: str, kind: str, count: int, first: int, sample: float
) -> ValueError:
    """Build the histogram non-finite-policy ``ValueError``.

    The message names the calling method, whether a value or a weight
    offended, the first offending position, and the total count of
    non-finite elements.
    """
    noun = kind if count == 1 else kind + "s"
    return ValueError(
        f"{method}: {count} non-finite {noun} rejected "
        f"(first at position {first}, got {sample!r}); "
        "only finite inputs are accepted"
    )


def _nonfinite_scalar_rejection(method: str, kind: str, value: float) -> ValueError:
    """Build the scalar-input variant of the non-finite-policy ``ValueError``.

    Mirrors the elementwise :func:`_nonfinite_rejection` message shape:
    the calling method, the offending scalar quantity and its value, and
    the remediation. Raised before any mutation happens.
    """
    return ValueError(
        f"{method}: non-finite {kind} rejected (got {value!r}); "
        "only finite inputs are accepted"
    )


def _reject_nonfinite_storage(
    method: str, attempted: str, candidate: np.ndarray
) -> None:
    """Enforce the finite-storage invariant on a private candidate array.

    Called after the requested update was computed on a private array and
    before anything is committed to ``_contents``. A silent return means
    *candidate* is all finite and the caller may commit it; otherwise the
    storage-invariant ``ValueError`` is raised naming the calling method,
    the rejected attempt, the number of offending elements and the first
    one. A rejected call must leave the histogram bit-for-bit unchanged
    and keep its sampling caches exactly as they were.
    """
    finite = np.isfinite(candidate)
    if finite.all():
        return
    bad = np.flatnonzero(~finite)
    noun = "bin content" if bad.size == 1 else "bin contents"
    raise ValueError(
        f"{method}: {attempted} would leave {int(bad.size)} non-finite "
        f"{noun} (first at index {int(bad[0])}, got "
        f"{float(candidate[int(bad[0])])!r}); nothing was committed, the "
        "histogram contents and sampling caches are unchanged"
    )


def _require_finite_bin_geometry(
    nbins: int, xlow: float, xhigh: float, *, label: str
) -> float:
    """Require a finite positive span and a finite positive bin width.

    The caller has already required finite edges with ``xhigh > xlow``.
    Opposite-sign near-maximum edges can still overflow
    ``xhigh - xlow`` to infinity, after which bin indices no longer
    represent the declared interval. The returned value is the common
    bin width ``(xhigh - xlow) / nbins``.
    """
    span = xhigh - xlow
    if not math.isfinite(span) or span <= 0.0:
        raise ValueError(
            f"{label} span must be finite and positive, got "
            f"xhigh-xlow={span!r} from xlow={xlow!r}, xhigh={xhigh!r}"
        )
    width = span / nbins
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError(
            f"{label} bin_width must be finite and positive, got "
            f"{width!r} (nbins={nbins}, xlow={xlow!r}, xhigh={xhigh!r})"
        )
    return width


def _coerce_histogram_contents(
    contents: object, nbins: int, *, label: str
) -> np.ndarray:
    """Validate histogram storage and return an owned ``float64`` copy.

    Shared by :meth:`Histogram1D.from_dict` and the persistence
    validator in ``data_io``. *contents* must be a one-dimensional
    real-numeric array of exact length ``nbins + 2`` whose values are
    all finite. The returned array never aliases the caller's memory.
    """
    arr = np.asarray(contents)
    if arr.ndim != 1:
        raise ValueError(
            f"{label} contents must be one-dimensional, got an array with "
            f"ndim={arr.ndim} and shape {arr.shape}"
        )
    if arr.dtype.kind not in "fiu":
        raise ValueError(
            f"{label} contents must have a real numeric dtype "
            f"(kind in {{'f', 'i', 'u'}}), got {arr.dtype!r}"
        )
    if arr.shape[0] != nbins + 2:
        raise ValueError(
            f"{label} contents length {arr.shape[0]} does not match "
            f"nbins+2={nbins + 2}"
        )
    finite = np.isfinite(arr)
    if not finite.all():
        bad = np.flatnonzero(~finite)
        raise ValueError(
            f"{label} contents contain {bad.size} non-finite value(s) "
            f"(first at position {int(bad[0])}, got "
            f"{float(arr[int(bad[0])])!r}); stored bin contents must "
            "be finite"
        )
    return arr.astype(np.float64, copy=True)


class RandomSampler(Protocol):
    """Sampling interface shared by ``Histogram1D`` and ``TabulatedPDF``.

    Both implementations expose a scalar and a vectorised draw that
    optionally accept a shared random generator. Generator-level sampling
    slots (pT, longitudinal momentum, mass law) hold either class, so they
    are typed against this protocol rather than against one concrete
    implementation.
    """

    def get_random(self, rng: Optional[np.random.Generator] = None) -> float:
        ...

    def get_random_n(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        ...


class Histogram1D:
    """1-D histogram with equal-width bins.

    Empty-distribution policy
    -------------------------
    When every bin content is zero or negative the distribution carries no
    probability mass and cannot be sampled. Both :meth:`get_random` and
    :meth:`get_random_n` then return the range midpoint
    ``(xlow + xhigh) / 2`` for every draw, so the scalar and vectorised
    paths agree by construction. (The Breit-Wigner sampler implements its
    own empty-policy: a delta draw at the pole mass, which the generic
    midpoint rule would misplace.)

    Non-finite input policy
    -----------------------
    Values and weights passed to :meth:`fill`, :meth:`fill_array` and
    :meth:`find_bin` must be finite. A NaN, ``+inf`` or ``-inf`` value or
    weight raises :class:`ValueError` naming the method, whether a value
    or a weight offended, the first offending position, and the total
    count of non-finite elements. Rejection is all-or-nothing: when any
    element of a vectorised fill offends, every bin retains its previous
    content bit-for-bit and the cached CDF/probabilities stay untouched.
    Zero and negative weights are accepted. For finite inputs the routing
    is unchanged: values below ``xlow`` go to underflow (bin 0), values
    ``x >= xhigh`` go to overflow (bin ``nbins + 1``), and in-range
    values map to ``floor((x - xlow) / bin_width) + 1`` clipped to
    ``[1, nbins]``.

    Non-finite storage policy
    -------------------------
    The mutating operations :meth:`scale`, :meth:`add`, :meth:`fill` and
    :meth:`fill_array` commit their update only when the complete
    resulting storage stays finite. Each update is computed on a private
    array first; if any element would come out NaN or infinite, the call
    raises ``ValueError`` and commits nothing: ``_contents`` is left
    bit-for-bit unchanged and the cached CDF/probabilities keep their
    exact state (a ``None`` cache stays ``None``, a built array stays
    the same object). Covered cases include finite scalar factors whose
    application overflows to infinity (contents 1e300 scaled by 1e300),
    bins accumulated past the largest finite double by repeated finite
    weights, and ``0 * inf`` products when one operand carries tainted
    storage. :meth:`set_bin_content` intentionally bypasses this policy
    so non-conforming states remain constructible for testing; the
    persistence layer refuses them at save time.
    :meth:`from_dict` uses the same contents invariant as the file
    loaders: a one-dimensional real-numeric array of length
    ``nbins + 2`` with finite values, stored as a fresh ``float64``
    copy. :meth:`set_bin_content` remains the only intentional
    invariant bypass.

    Sampling
    --------
    Inverse-CDF construction divides positive interior contents by
    their maximum before summing, so several bins near the largest
    finite double still form a usable distribution. Non-finite stored
    contents (reachable only through :meth:`set_bin_content`) raise
    ``ValueError`` when a draw is requested.

    Parameters
    ----------
    nbins : int
        Number of bins (must be a positive integer).
    xlow : float
        Lower edge of the first bin.
    xhigh : float
        Upper edge of the last bin.
    name : str, optional
        Descriptive name (informational only).

    Raises
    ------
    ValueError
        If ``nbins`` is not a positive integer, if ``xlow`` or ``xhigh``
        is not finite, if ``xhigh <= xlow``, or if the derived span
        ``xhigh - xlow`` or the bin width is not finite and positive.
    """

    def __init__(
        self,
        nbins: int,
        xlow: float,
        xhigh: float,
        name: str = "",
    ) -> None:
        self.name = name
        nb = int(nbins)
        if nb != nbins or nb < 1:
            raise ValueError(
                f"Histogram1D requires a positive integer nbins, got {nbins!r}"
            )
        xl = float(xlow)
        xh = float(xhigh)
        if not (math.isfinite(xl) and math.isfinite(xh)):
            raise ValueError(
                "Histogram1D edges must be finite, got "
                f"xlow={xlow!r}, xhigh={xhigh!r}"
            )
        if xh <= xl:
            raise ValueError(
                f"Histogram1D requires xlow < xhigh, got xlow={xl!r}, xhigh={xh!r}"
            )
        width = _require_finite_bin_geometry(nb, xl, xh, label="Histogram1D")
        self._nbins = nb
        self._xlow = xl
        self._xhigh = xh
        self._bin_width = width
        # bin contents (index 0 … nbins-1 are the real bins;
        # indices -1 and nbins are the under/overflow — not used here)
        self._contents = np.zeros(self._nbins + 2, dtype=np.float64)
        # CDF cache — rebuilt lazily
        self._cdf: Optional[np.ndarray] = None
        self._probs: Optional[np.ndarray] = None
        self._bin_centers: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Bin access helpers
    # ------------------------------------------------------------------
    @property
    def nbins(self) -> int:
        return self._nbins

    @property
    def xlow(self) -> float:
        return self._xlow

    @property
    def xhigh(self) -> float:
        return self._xhigh

    @property
    def bin_width(self) -> float:
        return self._bin_width

    def get_nbins_x(self) -> int:
        return self._nbins

    def get_bin_low_edge(self, ibin: int) -> float:
        """Lower edge of bin *ibin* (1-indexed, ROOT convention)."""
        return self._xlow + (ibin - 1) * self._bin_width

    def get_bin_up_edge(self, ibin: int) -> float:
        return self._xlow + ibin * self._bin_width

    def get_bin_center(self, ibin: int) -> float:
        return self._xlow + (ibin - 0.5) * self._bin_width

    def get_bin_width_at(self, ibin: int) -> float:
        return self._bin_width

    def find_bin(self, x: float) -> int:
        """Find the bin number (1-indexed) for value *x*.

        Finite *x* below ``xlow`` maps to 0 (underflow) and finite *x*
        ``>= xhigh`` maps to ``nbins + 1`` (overflow); in-range values
        map to ``floor((x - xlow) / bin_width) + 1``. Non-finite *x*
        (NaN, ``+inf``, ``-inf``) raises ``ValueError`` under the
        non-finite input policy documented in the class docstring.
        """
        xf = float(x)
        if not math.isfinite(xf):
            raise _nonfinite_rejection("find_bin", "value", 1, 0, xf)
        if xf < self._xlow:
            return 0
        if xf >= self._xhigh:
            return self._nbins + 1
        return int((xf - self._xlow) / self._bin_width) + 1

    # ------------------------------------------------------------------
    # Content manipulation
    # ------------------------------------------------------------------
    def set_bin_content(self, ibin: int, value: float) -> None:
        """Set bin content (1-indexed bin)."""
        self._contents[ibin] = float(value)
        self._cdf = None  # invalidate cache
        self._probs = None

    def get_bin_content(self, ibin: int) -> float:
        return float(self._contents[ibin])

    def fill(self, x: float, weight: float = 1.0) -> None:
        """Fill bin corresponding to *x* with *weight*.

        Values below ``xlow`` go to underflow (bin 0); values
        ``x >= xhigh`` go to overflow (bin ``nbins + 1``).

        Non-finite input policy: a NaN, ``+inf`` or ``-inf`` value or
        weight raises ``ValueError`` and leaves the histogram untouched.
        Zero and negative weights are accepted. See the class docstring.

        Non-finite storage policy: a finite deposit that would leave any
        stored element non-finite — including a bin accumulated past the
        largest finite double by repeated finite weights — is rejected
        atomically; the update is computed on a private array and
        committed only when every element stays finite.
        """
        xf = float(x)
        wf = float(weight)
        if not math.isfinite(xf):
            raise _nonfinite_rejection("fill", "value", 1, 0, xf)
        if not math.isfinite(wf):
            raise _nonfinite_rejection("fill", "weight", 1, 0, wf)
        ibin = self.find_bin(xf)
        candidate = self._contents.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            candidate[ibin] += wf
            _reject_nonfinite_storage(
                "fill", f"a deposit of {wf!r} into bin {ibin}", candidate
            )
        self._contents = candidate
        self._cdf = None
        self._probs = None

    def fill_array(self, xs: np.ndarray, weights: Optional[np.ndarray] = None) -> None:
        """Vectorised fill using the same edges as :meth:`fill`.

        ``np.histogram`` would put ``x == xhigh`` in the last in-range bin
        and drop underflow; this path matches :meth:`find_bin` instead.

        Non-finite input policy: NaN, ``+inf`` or ``-inf`` in the values
        or in the weights raises ``ValueError`` naming the method, the
        offending quantity (value or weight), the first offending
        position within the flattened input, and the total count of
        non-finite elements. Rejection is all-or-nothing: a rejected call
        leaves every bin bit-for-bit unchanged and does not invalidate
        the cached CDF/probabilities. Zero and negative weights are
        accepted. Under the non-finite storage policy a single call whose
        bin sums would overflow to infinity — from individually finite
        weights — is likewise rejected atomically. See the class
        docstring.
        """
        vals = np.asarray(xs, dtype=np.float64).ravel()
        if vals.size == 0:
            return
        if weights is None:
            w = np.ones(vals.shape, dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64).ravel()
            if w.shape != vals.shape:
                raise ValueError(
                    "fill_array: weights must match xs, got "
                    f"xs.size={vals.size}, weights.size={w.size}"
                )
        # Non-finite policy: validate before constructing bin indices so a
        # rejected call mutates nothing and no NumPy RuntimeWarning (e.g.
        # "invalid value encountered in cast") can be emitted.
        vfin = np.isfinite(vals)
        if not vfin.all():
            bad = np.flatnonzero(~vfin)
            raise _nonfinite_rejection(
                "fill_array", "value", int(bad.size), int(bad[0]),
                float(vals[bad[0]]),
            )
        wfin = np.isfinite(w)
        if not wfin.all():
            bad = np.flatnonzero(~wfin)
            raise _nonfinite_rejection(
                "fill_array", "weight", int(bad.size), int(bad[0]),
                float(w[bad[0]]),
            )
        ibin = np.empty(vals.shape, dtype=np.int64)
        under = vals < self._xlow
        over = vals >= self._xhigh
        mid = ~under & ~over
        ibin[under] = 0
        ibin[over] = self._nbins + 1
        if np.any(mid):
            idx = np.floor((vals[mid] - self._xlow) / self._bin_width).astype(np.int64) + 1
            ibin[mid] = np.clip(idx, 1, self._nbins)
        # Non-finite storage policy: accumulate on a private array and
        # commit only an all-finite result (covers bins whose sums
        # overflow to infinity from individually finite weights).
        candidate = self._contents.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            np.add.at(candidate, ibin, w)
            _reject_nonfinite_storage(
                "fill_array", f"a deposit of {int(vals.size)} value(s)", candidate
            )
        self._contents = candidate
        self._cdf = None
        self._probs = None

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def _build_cdf(self) -> None:
        """Build the cumulative distribution function for sampling.

        Positive interior contents are scaled by their maximum before
        the total is formed, so several finite bins near the largest
        double still yield probabilities that sum to one. Non-finite
        stored contents are refused here as a defensive check; the
        public fill path cannot create them.
        """
        c = np.asarray(self._contents[1: self._nbins + 1], dtype=np.float64)
        if not np.isfinite(c).all():
            bad = np.flatnonzero(~np.isfinite(c))
            raise ValueError(
                "Histogram1D sampling refuses non-finite stored bin "
                f"contents ({int(bad.size)} non-finite, first at "
                f"interior index {int(bad[0])}, got "
                f"{float(c[int(bad[0])])!r})"
            )
        positive = np.maximum(c, 0.0)
        peak = float(np.max(positive)) if positive.size else 0.0
        if peak <= 0.0:
            self._cdf = np.zeros(self._nbins)
            self._probs = None
        else:
            scaled = positive / peak
            try:
                with np.errstate(over="raise", invalid="raise"):
                    total = float(scaled.sum())
            except FloatingPointError as exc:
                raise ValueError(
                    "Histogram1D sampling cannot form a finite "
                    "probability distribution from the stored contents"
                ) from exc
            if not math.isfinite(total) or total <= 0.0:
                raise ValueError(
                    "Histogram1D sampling cannot form a finite "
                    "probability distribution from the stored contents "
                    f"(normalized mass {total!r})"
                )
            probs = scaled / total
            self._probs = probs
            self._cdf = np.cumsum(probs)
        self._bin_centers = (
            self._xlow + (np.arange(self._nbins) + 0.5) * self._bin_width
        )

    def get_random(self, rng: Optional[np.random.Generator] = None) -> float:
        """Sample one value from the histogram distribution.

        Uses inverse CDF (piecewise-uniform within each bin). For an empty
        distribution the range midpoint is returned, matching
        :meth:`get_random_n` (see the class docstring).
        """
        if self._cdf is None:
            self._build_cdf()
        if rng is None:
            rng = np.random.default_rng()
        if self._probs is None:
            # Shared empty-distribution policy: return the range midpoint.
            return (self._xlow + self._xhigh) / 2.0
        cdf = self._cdf
        # _build_cdf() above assigns a concrete array; the assertion is the
        # static narrowing witness for the Optional attribute type.
        assert cdf is not None
        # find bin via searchsorted
        ibin = int(np.searchsorted(cdf, rng.random()))
        if ibin >= self._nbins:
            ibin = self._nbins - 1
        # uniform within the bin
        return self._xlow + ibin * self._bin_width + rng.random() * self._bin_width

    def get_random_n(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample *n* values at once (vectorised)."""
        if self._cdf is None:
            self._build_cdf()
        if rng is None:
            rng = np.random.default_rng()
        if self._probs is None:
            return np.full(n, (self._xlow + self._xhigh) / 2.0)
        ibins = rng.choice(self._nbins, size=n, p=self._probs)
        lo = self._xlow + ibins * self._bin_width
        return lo + rng.random(n) * self._bin_width

    # ------------------------------------------------------------------
    # Integration / scaling
    # ------------------------------------------------------------------
    def integral(self, ibin1: int = 1, ibin2: Optional[int] = None) -> float:
        if ibin2 is None:
            ibin2 = self._nbins
        return float(self._contents[ibin1: ibin2 + 1].sum())

    def scale(self, factor: float) -> None:
        """Scale every bin content by *factor*.

        The factor must be a finite scalar. A NaN, ``+inf`` or ``-inf``
        factor raises ``ValueError`` (naming the method, the factor and
        the remediation) before anything is mutated. Under the non-finite
        storage policy a finite factor whose application would leave any
        element non-finite — e.g. contents 1e300 scaled by 1e300 — is
        rejected atomically: the multiplied contents are computed on a
        private array and committed only when all elements stay finite,
        leaving ``_contents`` bit-for-bit unchanged and the caches
        exactly as they were.
        """
        factored = float(factor)
        if not math.isfinite(factored):
            raise _nonfinite_scalar_rejection("scale", "factor", factored)
        with np.errstate(over="ignore", invalid="ignore"):
            candidate = self._contents * factored
            _reject_nonfinite_storage(
                "scale", f"scaling by {factored!r}", candidate
            )
        self._contents = candidate
        self._cdf = None
        self._probs = None

    def add(self, other: "Histogram1D", scale: float = 1.0) -> None:
        """Add *other* into ``self``, bin by bin, weighted by *scale*.

        The *scale* factor must be a finite scalar; a NaN, ``+inf`` or
        ``-inf`` scale raises ``ValueError`` naming the method, the
        factor and the remediation. This validation runs before the axis
        compatibility check and before any mutation.

        Both histograms must describe the same axis: ``nbins``, ``xlow``
        and ``xhigh`` must be exactly equal, because these three values
        define the persisted bin geometry. A pair that differs in any of
        them raises ``ValueError`` naming both histograms, both axes,
        and each differing field; this check runs before any mutation, so
        a rejected call leaves ``self`` bit-for-bit unchanged and keeps
        its cached CDF/probabilities. Rebinning is deliberately out of
        scope here.

        Under the non-finite storage policy the weighted sum is computed
        on a private array and committed only when every element stays
        finite: a finite scale whose application overflows to infinity,
        or a ``0 * inf`` product when one operand carries tainted (e.g.
        infinite) storage while self holds zeros there, both reject with
        nothing committed.
        """
        scaled = float(scale)
        if not math.isfinite(scaled):
            raise _nonfinite_scalar_rejection("add", "scale", scaled)
        mismatches = []
        if self._nbins != other._nbins:
            mismatches.append(f"nbins ({self._nbins} vs {other._nbins})")
        if self._xlow != other._xlow:
            mismatches.append(f"xlow ({self._xlow!r} vs {other._xlow!r})")
        if self._xhigh != other._xhigh:
            mismatches.append(f"xhigh ({self._xhigh!r} vs {other._xhigh!r})")
        if mismatches:
            raise ValueError(
                "Histogram1D.add(): histograms describe different axes; "
                "exact equality of nbins, xlow and xhigh is required "
                "(equal bin counts alone do not qualify), got "
                f"{self!r} versus {other!r}; differing fields: "
                + "; ".join(mismatches)
                + "; nothing was added"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            candidate = self._contents + other._contents * scaled
            _reject_nonfinite_storage(
                "add",
                f"adding {other!r} weighted by {scaled!r}",
                candidate,
            )
        self._contents = candidate
        self._cdf = None
        self._probs = None

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------
    def __iadd__(self, other: "Histogram1D") -> "Histogram1D":
        self.add(other)
        return self

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def get_entries(self) -> float:
        return float(self._contents[1: self._nbins + 1].sum())

    def get_mean(self) -> float:
        c = self._contents[1: self._nbins + 1]
        centers = self._xlow + (np.arange(self._nbins) + 0.5) * self._bin_width
        s = c.sum()
        if s == 0.0:
            return 0.0
        return float((c * centers).sum() / s)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "nbins": self._nbins,
            "xlow": self._xlow,
            "xhigh": self._xhigh,
            "contents": self._contents.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Histogram1D":
        """Reconstruct a histogram from :meth:`to_dict` output.

        Geometry goes through the constructor. Contents pass the same
        rank, length, dtype-kind, and finiteness checks as persisted
        payloads, and are stored as a fresh ``float64`` copy.
        """
        h = cls(d["nbins"], d["xlow"], d["xhigh"], name=d.get("name", ""))
        if "contents" not in d:
            raise ValueError(
                "Histogram1D.from_dict() requires 'contents'"
            )
        h._contents = _coerce_histogram_contents(
            d["contents"], h._nbins, label="Histogram1D.from_dict()"
        )
        h._cdf = None
        h._probs = None
        h._bin_centers = None
        return h

    def __repr__(self) -> str:
        return (
            f"Histogram1D(name={self.name!r}, nbins={self._nbins}, "
            f"xlow={self._xlow}, xhigh={self._xhigh})"
        )


# ---------------------------------------------------------------------------
# Spline helper — ported from ``spliner.cxx``
# ---------------------------------------------------------------------------

def _get_poly_line_value(hist: Histogram1D, t: float, _depth: int = 0) -> float:
    """Polyline interpolation of *hist* at parameter *t* ∈ [0, 1]."""
    if _depth > 50:
        # Guard against extreme values; clamp to boundary
        t = max(0.0, min(1.0, t))
    if t < 0.0:
        return 2.0 * _get_poly_line_value(hist, 0.0) - _get_poly_line_value(hist, -t, _depth + 1)
    if t > 1.0:
        return 2.0 * _get_poly_line_value(hist, 1.0) - _get_poly_line_value(hist, 2.0 - t, _depth + 1)

    xmin = hist.xlow
    xmax = hist.xhigh
    x = xmin + t * (xmax - xmin)
    ibin = hist.find_bin(x)  # 1-indexed
    if ibin <= hist.nbins:
        x1 = hist.get_bin_low_edge(ibin)
        x2 = hist.get_bin_up_edge(ibin)
        v1 = hist.get_bin_content(ibin)
        v2 = hist.get_bin_content(ibin + 1)
        return v1 + (v2 - v1) * ((x - x1) / (x2 - x1))
    else:
        return hist.get_bin_content(hist.nbins)


def _spline_kernel(t: float) -> float:
    """B-spline kernel (cubic)."""
    if t < -2.0:
        return 0.0
    if t < -1.0:
        return (2.0 + t) ** 3 / 6.0
    if t < 0.0:
        return (4.0 - 6.0 * t * t - 3.0 * t ** 3) / 6.0
    if t < 1.0:
        return (4.0 - 6.0 * t * t + 3.0 * t ** 3) / 6.0
    if t < 2.0:
        return (2.0 - t) ** 3 / 6.0
    return 0.0


def spliner(hist: Histogram1D, nbins: int = 200) -> Histogram1D:
    """Return a fine-grained spline approximation of *hist* with *nbins* bins.

    Ported from ``Spliner()`` in ``spliner.cxx``.
    """
    n_int = 50  # integration points
    trange = hist.nbins / nbins
    rescale = 4.0 / trange
    out = Histogram1D(nbins, hist.xlow, hist.xhigh, name=hist.name + "_splined")

    for ibin in range(1, nbins + 2):
        tval = (ibin - 1 + 1e-5) / nbins
        value = 0.0
        for k in range(n_int):
            t = -trange / 2.0 + (k + 0.5) * trange / n_int
            value += rescale * _spline_kernel(rescale * t) * _get_poly_line_value(hist, tval + t)
        value = value * trange / n_int
        out.set_bin_content(ibin, value)

    return out
