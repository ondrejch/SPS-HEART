"""MC particle generator — port of ``TGenerPart.h`` / ``TGenerPart.cxx``.

Generates identified particles from y/y_max and p_T parametrisations.
Tabulated distributions are handled by :class:`TabulatedPDF`, which preserves
the supplied abscissae exactly (piecewise-linear or bin-constant densities,
nonuniform spacing supported).  The historical uniform-grid + spline path is
superseded; ``spliner`` remains available in ``histogram`` for legacy callers.
"""

from __future__ import annotations
import math
import sys
from typing import Optional, List
import numpy as np

from .histogram import Histogram1D, RandomSampler, spliner
from .breit_wigner import BreitWigner
from .particle import Particle
from .pdg_database import get_mass, get_charge
from .reaction import Reaction


class TabulatedPDF:
    """Probability density tabulated on supplied abscissae.

    Two input conventions are supported:

    - ``len(values) == len(coords)``: piecewise-linear density through the
      supplied points (trapezoid integration);
    - ``len(values) == len(coords) - 1``: piecewise-constant density per
      interval between consecutive coordinates.

    All supplied coordinates are preserved exactly; no re-binning onto a
    uniform grid takes place.  Sampling is by inverse CDF over the exact
    piecewise integral, so nonuniform spacing is honoured both in shape and
    in normalisation.

    Raises
    ------
    ValueError
        If ``coords`` or ``values`` is not one-dimensional, if fewer than
        two coordinates are supplied, if any coordinate is not finite
        (a NaN coordinate would slip past the monotonicity comparison),
        if consecutive differences overflow, if coordinates are not
        strictly increasing, if any density value is negative or
        non-finite, if the value count matches neither supported
        convention, if any derived segment mass is non-finite, or if the
        total integral is non-positive or non-finite.
    """

    def __init__(self, coords, values, name: str = "") -> None:
        x = np.asarray(coords, dtype=float)
        y = np.asarray(values, dtype=float)
        if x.ndim != 1:
            raise ValueError(
                "TabulatedPDF coordinates must be one-dimensional, "
                f"got ndim={x.ndim} with shape {x.shape}"
            )
        if y.ndim != 1:
            raise ValueError(
                "TabulatedPDF values must be one-dimensional, "
                f"got ndim={y.ndim} with shape {y.shape}"
            )
        if x.size < 2:
            raise ValueError("TabulatedPDF needs at least two coordinates")
        if np.any(~np.isfinite(x)):
            # NaN coordinates pass the strictly-increasing comparison below
            # (all comparisons with NaN are false), so reject them first.
            bad = x[~np.isfinite(x)][0]
            raise ValueError(
                f"TabulatedPDF coordinates must be finite, got {bad!r}"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            dx = np.diff(x)
        if np.any(~np.isfinite(dx)):
            raise ValueError(
                "TabulatedPDF coordinate spans must be finite; the "
                "supplied abscissae overflow when consecutive "
                "differences are formed"
            )
        if np.any(dx <= 0.0):
            raise ValueError("TabulatedPDF coordinates must be strictly increasing")
        if np.any(~np.isfinite(y)) or np.any(y < 0.0):
            raise ValueError("TabulatedPDF values must be finite and non-negative")
        if y.size == x.size:
            linear = True
        elif y.size == x.size - 1:
            linear = False
        else:
            raise ValueError(
                "values must match coords (linear) or coords-1 (bin-constant)"
            )
        peak = float(np.max(y)) if y.size else 0.0
        if not math.isfinite(peak) or peak <= 0.0:
            raise ValueError("TabulatedPDF has non-positive total integral")
        y_s = y / peak
        with np.errstate(over="ignore", invalid="ignore"):
            if linear:
                seg = 0.5 * (y_s[1:] + y_s[:-1]) * dx
            else:
                seg = y_s * dx
        if (not np.all(np.isfinite(seg))) or np.any(seg < 0.0):
            raise ValueError(
                "TabulatedPDF segment masses must be finite; the supplied "
                "densities overflow when multiplied by the coordinate spans"
            )
        seg_peak = float(np.max(seg))
        if not math.isfinite(seg_peak) or seg_peak <= 0.0:
            raise ValueError("TabulatedPDF has non-positive total integral")
        seg_n = seg / seg_peak
        total_n = float(seg_n.sum())
        if not math.isfinite(total_n) or total_n <= 0.0:
            raise ValueError(
                "TabulatedPDF total integral overflows; the tabulated "
                "density cannot be normalized to a finite probability "
                "distribution"
            )
        self.name = name
        self._linear = linear
        self._x = x
        self._y = y
        self._y_draw = y_s
        self._seg_raw = seg
        self._seg = seg_n / total_n
        self._cdf = np.cumsum(self._seg)
        if not np.all(np.isfinite(self._cdf)):
            raise ValueError(
                "TabulatedPDF total integral overflows; the tabulated "
                "density cannot be normalized to a finite probability "
                "distribution"
            )

    @property
    def domain(self) -> tuple[float, float]:
        return float(self._x[0]), float(self._x[-1])

    @property
    def coords(self) -> np.ndarray:
        return self._x

    @property
    def values(self) -> np.ndarray:
        return self._y

    def mean(self) -> float:
        """Numerical mean of the density (trapezoid/bin rule)."""
        if self._linear:
            trapz = getattr(np, "trapezoid", None) or np.trapz
            m = trapz(self._x * self._y, self._x)
            norm = trapz(self._y, self._x)
        else:
            mid = 0.5 * (self._x[1:] + self._x[:-1])
            m = float(np.sum(mid * self._seg))
            norm = 1.0
        return float(m / norm)

    # ------------------------------------------------------------------
    # Sampling (Histogram1D-compatible interface)
    # ------------------------------------------------------------------
    def _sample_array(self, u: np.ndarray) -> np.ndarray:
        cdf = self._cdf
        idx = np.searchsorted(cdf, u, side="right")
        idx = np.minimum(idx, cdf.size - 1)
        prev = np.where(idx > 0, cdf[np.maximum(idx - 1, 0)], 0.0)
        rem = np.clip((u - prev) / np.maximum(self._seg[idx], 1e-300), 0.0, 1.0)
        x0 = self._x[idx]
        x1 = self._x[idx + 1]
        L = x1 - x0
        if self._linear:
            y0 = self._y_draw[idx]
            y1 = self._y_draw[idx + 1]
            # Invert the within-segment CDF
            #   F(t) = y0*t + (y1-y0)*t^2/(2L)  =  rem * cap,
            # with cap the unnormalised segment mass.
            cap = self._seg_raw[idx]
            target = rem * cap
            dy = y1 - y0
            flat = np.abs(dy) <= 1e-12 * (np.abs(y0) + np.abs(y1) + 1e-300)
            # Quadratic branch: a t^2 + b t - target = 0 with a = dy/(2L).
            a = 0.5 * dy / L
            b = y0
            tiny = 1e-300
            disc = np.maximum(b * b + 4.0 * a * target, 0.0)
            t_quad = (-b + np.sqrt(disc)) / np.where(
                np.abs(a) > tiny, 2.0 * a, tiny
            )
            t_lin = target / np.maximum(b, tiny)
            t = np.where(flat, t_lin, t_quad)
            t = np.clip(t, 0.0, L)
        else:
            t = rem * L
        return x0 + t

    def get_random(self, rng=None) -> float:
        if rng is None:
            rng = np.random.default_rng()
        return float(self._sample_array(rng.random(1))[0])

    def get_random_n(self, n: int, rng=None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        n = int(n)
        if n <= 0:
            return np.empty(0)
        return self._sample_array(rng.random(n))


class Generator:
    """MC generator for one particle species.

    Parameters
    ----------
    pdg_id : int
        PDG ID of the particle to generate.
    reaction : Reaction
        Reaction object providing Lorentz boost parameters.
    rng : numpy.random.Generator, optional
        Shared RNG (a new generator is created if not supplied).
    """

    def __init__(
        self,
        pdg_id: int = 0,
        reaction: Optional[Reaction] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._id = int(pdg_id)
        self._id_mass: float = 0.0
        self._id_charge: float = 0.0
        self._react = reaction
        self._rng: np.random.Generator = rng if rng is not None else np.random.default_rng()

        self._pt_type: int = 0           # 0=unset, 1=gauss, 2=exp, 3=histogram
        self._pt_param: float = 0.0
        # pT / longitudinal / mass sampling slots hold either histogram- or
        # tabulation-backed samplers; both expose the shared interface.
        self._pt_hist: Optional[RandomSampler] = None

        self._pl_type: int = 1           # 1=y/ymax, 2=xF
        self._y_ymax: Optional[RandomSampler] = None

        self._mass_type: int = 0         # 0=PDG, 1=BW (short), 2=histogram, 3=BW (full)
        self._mass_hist: Optional[RandomSampler] = None
        self._mass_params: List[float] = [0.0] * 6
        self._bw: Optional[BreitWigner] = None
        self._refresh_id_cache()

    def _refresh_id_cache(self) -> None:
        self._id_mass = get_mass(self._id, default=0.0)
        self._id_charge = get_charge(self._id, default=0.0)

    # ------------------------------------------------------------------
    # Configuration methods
    # ------------------------------------------------------------------
    def set_pt(self, pt_type: int, pt_param: float = 0.0) -> None:
        """Configure pT sampling.

        pt_type == 1 → Gaussian (param = 1/2σ²)
        pt_type == 2 → exponential (param = slope)

        A non-positive or non-finite ``pt_param`` would build a degenerate
        sampling law (flat or exponentially growing in pT) and is rejected
        with :class:`ValueError` before any state is modified.
        """
        if pt_type in (1, 2):
            p = float(pt_param)
            if not (math.isfinite(p) and p > 0.0):
                raise ValueError(
                    f"Generator.set_pt(): pt_param must be finite and "
                    f"positive for pt_type={pt_type}, got pt_param={pt_param!r}"
                )
        self._pt_type = pt_type
        self._pt_param = float(pt_param)
        nbins = 500
        xs = np.linspace(0.0, 5.0, nbins)
        if pt_type == 1:
            ys = xs * np.exp(-pt_param * xs * xs)
        elif pt_type == 2:
            ys = xs * np.exp(-pt_param * xs)
        else:
            return
        h = Histogram1D(nbins, 0.0, 5.0)
        for i, y in enumerate(ys):
            h.set_bin_content(i + 1, float(y))
        self._pt_hist = h

    def set_pt_hist(self, hist: Histogram1D) -> None:
        self._pt_type = 3
        self._pt_hist = hist

    def set_y_ymax(self, pl_type: int, y_ymax) -> None:
        self._pl_type = 2 if pl_type == 2 else 1
        self._y_ymax = y_ymax

    def set_mass_bw(self, bw: BreitWigner) -> None:
        self._mass_type = 3
        self._bw = bw

    def set_mass_hist(self, hist: Histogram1D) -> None:
        self._mass_type = 2
        self._mass_hist = hist

    # ------------------------------------------------------------------
    # File-based configuration (mirrors TGenerPart::SetAll)
    # ------------------------------------------------------------------
    def set_all(self, filename: str) -> None:
        """Configure from a ``.gen`` text file."""
        with open(filename) as fh:
            tokens = fh.read().split()
        idx = 0

        def next_token() -> str:
            nonlocal idx
            v = tokens[idx]; idx += 1; return v

        self._id = int(next_token())
        self._refresh_id_cache()
        pt_type = int(next_token())
        if pt_type == 3:
            nbins = int(next_token())
            edges = []
            conts = []
            for _ in range(nbins):
                edges.append(float(next_token()))
                conts.append(float(next_token()))
            edges.append(float(next_token()))
            # Variable-width bin-constant density; all supplied edges are
            # preserved (no uniform re-binning).
            self._pt_hist = TabulatedPDF(edges, conts, name="pt_pdf")
            self._pt_type = 3
        else:
            pt_param = float(next_token())
            self.set_pt(pt_type, pt_param)

        pl_type = int(next_token())
        nbins = int(next_token())
        coords = []
        conts = []
        for _ in range(nbins + 1):
            coords.append(float(next_token()))
            conts.append(float(next_token()))
        # Piecewise-linear tabulated density on the SUPPLIED abscissae.
        # The historical implementation built a uniform histogram over
        # [coords[0], coords[-2]], discarding every interior coordinate and
        # truncating the domain; that shifted symmetric meson cards by
        # ~0.2 in xF. TabulatedPDF preserves the full table exactly.
        y_ymax = TabulatedPDF(coords, conts, name="pl_pdf")
        self._y_ymax = y_ymax
        self._pl_type = 2 if pl_type == 2 else 1

        mass_type = int(next_token())
        self._mass_type = mass_type
        if mass_type == 1:
            # Legacy short BW encoding from the original C++ format:
            # (p0, p1, p2, p4, p5) with p3 implicitly fixed to 1.0.
            p0 = float(next_token())
            p1 = float(next_token())
            p2 = float(next_token())
            p4 = float(next_token())
            p5 = float(next_token())
            self._bw = BreitWigner(p0, p1, p2, 1.0, p4, p5, rng=self._rng)
        elif mass_type == 2:
            nbins = int(next_token())
            edges = []
            conts = []
            for _ in range(nbins):
                edges.append(float(next_token()))
                conts.append(float(next_token()))
            edges.append(float(next_token()))
            self._mass_hist = TabulatedPDF(edges, conts, name="mass_pdf")
        elif mass_type == 3:
            # Full BW encoding stores all six parameters explicitly.
            p0 = float(next_token())
            p1 = float(next_token())
            p2 = float(next_token())
            p3 = float(next_token())
            p4 = float(next_token())
            p5 = float(next_token())
            self._bw = BreitWigner(p0, p1, p2, p3, p4, p5, rng=self._rng)

    # ------------------------------------------------------------------
    # Particle generation
    # ------------------------------------------------------------------
    def generate(self) -> Optional[Particle]:
        """Generate one particle and return it as a :class:`~na49_mixing.particle.Particle`.

        Returns ``None`` if the generator is not fully configured.
        """
        pt_hist = self._pt_hist
        y_ymax = self._y_ymax
        react = self._react
        rng = self._rng
        if pt_hist is None or y_ymax is None:
            print(
                "Generator.generate(): pt_hist or y_ymax not set",
                file=sys.stderr,
            )
            return None

        # --- mass ---
        if self._mass_type == 0:
            mass = self._id_mass
        elif self._mass_type in (1, 3):
            mass = self._bw.get_random()  # type: ignore[union-attr]
        else:
            mass = self._mass_hist.get_random(rng)  # type: ignore[union-attr]

        # --- phi, pT ---
        phi = rng.random() * 2.0 * math.pi
        pt = pt_hist.get_random(rng)
        px = pt * math.cos(phi)
        py = pt * math.sin(phi)

        # --- pz in CMS ---
        if self._pl_type == 1:
            if react is None:
                raise RuntimeError("Reaction not set on Generator")
            plmax = react.plmax()
            emax = math.sqrt(plmax * plmax + mass * mass)
            pt2 = pt * pt + mass * mass
            ymax = math.log(
                (emax + math.sqrt(max(emax * emax - pt2, 0.0))) / math.sqrt(pt2)
            )
            y = ymax * y_ymax.get_random(rng)
            pz_cms = math.sqrt(pt2) * math.sinh(y)
        elif self._pl_type == 2:
            if react is None:
                raise RuntimeError("Reaction not set on Generator")
            pz_cms = y_ymax.get_random(rng) * react.plmax()
        else:
            print("Generator: unknown pl_type", file=sys.stderr)
            return None

        # --- boost CMS → LAB ---
        en = math.sqrt(px * px + py * py + pz_cms * pz_cms + mass * mass)
        if react is not None:
            pz = react.gamma() * (pz_cms + react.betha() * en)
        else:
            pz = pz_cms

        part = Particle(px, py, pz, self._id_charge, mass, self._id, react)
        return part

    def generate_n(self, n: int) -> List[Particle]:
        """Generate *n* particles with vectorised sampling where possible.

        Physics matches :meth:`generate` (same RNG consumption order is not
        guaranteed relative to repeated ``generate()`` calls). Returns only
        successfully constructed particles; misconfiguration raises.
        """
        n = int(n)
        if n <= 0:
            return []
        pt_hist = self._pt_hist
        y_ymax = self._y_ymax
        react = self._react
        rng = self._rng
        if pt_hist is None or y_ymax is None:
            raise RuntimeError("Generator.generate_n(): pt_hist or y_ymax not set")
        if react is None and self._pl_type in (1, 2):
            raise RuntimeError("Reaction not set on Generator")

        # --- masses ---
        if self._mass_type == 0:
            masses = np.full(n, self._id_mass, dtype=float)
        elif self._mass_type in (1, 3):
            masses = np.asarray(self._bw.get_random_n(n), dtype=float)  # type: ignore[union-attr]
        else:
            masses = np.asarray(self._mass_hist.get_random_n(n, rng), dtype=float)  # type: ignore[union-attr]

        phi = rng.random(n) * 2.0 * math.pi
        pt = np.asarray(pt_hist.get_random_n(n, rng), dtype=float)
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)

        if self._pl_type == 1:
            plmax = float(react.plmax())  # type: ignore[union-attr]
            emax = np.sqrt(plmax * plmax + masses * masses)
            pt2 = pt * pt + masses * masses
            ymax = np.log(
                (emax + np.sqrt(np.maximum(emax * emax - pt2, 0.0))) / np.sqrt(pt2)
            )
            y = ymax * np.asarray(y_ymax.get_random_n(n, rng), dtype=float)
            pz_cms = np.sqrt(pt2) * np.sinh(y)
        elif self._pl_type == 2:
            pz_cms = np.asarray(y_ymax.get_random_n(n, rng), dtype=float) * float(react.plmax())  # type: ignore[union-attr]
        else:
            raise RuntimeError(f"Generator: unknown pl_type={self._pl_type}")

        en = np.sqrt(px * px + py * py + pz_cms * pz_cms + masses * masses)
        if react is not None:
            pz = react.gamma() * (pz_cms + react.betha() * en)
        else:
            pz = pz_cms

        out: List[Particle] = []
        charge = self._id_charge
        pdg = self._id
        for i in range(n):
            out.append(
                Particle(
                    float(px[i]),
                    float(py[i]),
                    float(pz[i]),
                    charge,
                    float(masses[i]),
                    pdg,
                    react,
                )
            )
        return out

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"Generator(id={self._id}, pt_type={self._pt_type}, pl_type={self._pl_type})"


def generator_from_file(
    filename: str,
    reaction: Reaction,
    rng: Optional[np.random.Generator] = None,
) -> Generator:
    """Convenience factory: construct a :class:`Generator` from a ``.gen`` file."""
    gen = Generator(reaction=reaction, rng=rng)
    gen.set_all(filename)
    return gen
