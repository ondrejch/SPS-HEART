"""The low-mass non-diffractive lines through the N* and Delta records
(2026-10-12): ``TopologyNominalParams.nd_duality_mass`` and
``nd_duality_shape_mass``.

The excited-line mass law (M^2)^(-1-epsilon) puts its largest density at
the N pi threshold, where the measured N pi spectrum is the Delta(1232) and
the N* resonances.  A vacuum non-diffractive line below the duality mass
``Md`` decays as one of the catalog's low-mass (1, 1, 0) records at the
line's own mass and momentum (local duality): the record r with weight
``(2J_r + 1) BW_r(M)``, ``BW_r`` the Breit-Wigner of its pole and width
inside its mass range, equal for both isospins (the equal mixture of the
non-diffractive lines).  The record decays through its PDG channels at
the line's mass, on the line's decay substreams; the pick takes one
uniform on ``decay_channel`` before the record's own draws.

Below the shape mass ``Ms <= Md`` the line's mass also follows the
records' line shapes: the share of the lines below Ms stays the
continuum's, and inside that share M^2 follows the mixture
``sum_r (2J_r + 1) BW_r(M^2)`` on ``[lo^2, Ms^2]``.  The redraw is a
measure-preserving map of the continuum draw (no random number): the
continuum mass M fixes ``v = F(M) / F(Ms)`` with ``F`` the CDF of the
generation law on the line's window, and ``v`` selects the record by its
Breit-Wigner integral over the window and then ``M'^2`` by that record's
inverse CDF.  The exponent reweighting of a redrawn line is the ratio of
the continuum's shares below Ms
(:func:`na49_mixing.topology_mass_laws.log_density_ratio`).

At the R92 optimum the mapping below 2.0 GeV gave -294 with the yields
fixed and -410 with one normalization per species, and the shape below
1.4 GeV -253 and -339 (the fixed-kernel note,
``_knowledge/evidence/fixed-kernel-residual-mechanisms-2026-10-03.md``,
the low-mass lines through the N* and Delta records).

``TopologyNominalParams.duality_line_shapes`` (2026-10-18) weights each
record at the line's mass by its own line shape, the normalized spectral
function the decayer samples for it (the running width Gamma(M) of
:func:`na49_mixing.resonance_widths.spectral_function`), in place of the
fixed-width Breit-Wigner: one line shape per record everywhere.  A fixed
width overstates the broad N(1440)'s tail near the N pi threshold, where
the Delta(1232) carries the measured N pi spectrum.  At the R100 optimum
the pick with the line shapes gave -197 +- 47 with the normalizations
fitted and -292 +- 49 with the yields fixed (paired, two seed blocks; the
fixed-width form normalized over the same support +296 +- 36).

``TopologyNominalParams.duality_redraw_shapes`` (2026-10-20) carries the
same rule to the redraw below the shape mass: the line mass follows the
mixture ``sum_r (2J_r + 1) rho_r(M)`` of the records' normalized line
shapes on ``[lo, Ms]`` (:func:`duality_redraw_mass_shapes`), the same
measure-preserving map of the continuum draw.  The running widths empty
the region just above the N pi threshold and gather the lines at the
Delta(1232) peak.  At the R102 optimum it moved chi2 by +3 +- 13 with the
normalizations fitted (two seed blocks), inside the noise, and it enters
on the rule of one line shape per record.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

from .hadron_catalog import HadronCatalog
from .topology_remnants import low_mass_records

__all__ = ["DualityRecord", "duality_records", "duality_redraw_mass", "duality_redraw_mass_shapes", "duality_pick"]


@dataclass(frozen=True)
class DualityRecord:
    """One low-mass record of the duality mapping: its pdg id, pole mass
    and width [GeV], spin weight 2J + 1, and mass range [GeV]; ``shape``
    the record's normalized line shape at a mass (the duality line shapes,
    2026-10-18) or None for the fixed-width Breit-Wigner."""

    pdg_id: int
    pole: float
    width: float
    weight: float
    lo: float
    hi: float
    shape: object = None


def _line_shape(mass: float, record, masses, catalog) -> float:
    from .resonance_widths import spectral_function
    return spectral_function(float(mass), record, masses, catalog=catalog)


def _shape_of(catalog: HadronCatalog, record):
    from .resonance_decay import _check_daughter_masses
    return functools.partial(_line_shape, record=record, masses=_check_daughter_masses(catalog, record), catalog=catalog)


def duality_records(catalog: HadronCatalog, line_shapes: bool = False) -> tuple[DualityRecord, ...]:
    """The catalog's low-mass (1, 1, 0) records with a positive width (the
    proton excluded), in catalog order; with *line_shapes* each carries its
    normalized line shape."""
    out = []
    for r in low_mass_records(catalog):
        width = float(r.nominal_width)
        if width > 0.0:
            out.append(DualityRecord(int(r.pdg_id), float(r.pole_mass), width, 2.0 * float(r.spin) + 1.0,
                                     float(r.mass_range[0]), float(r.mass_range[1]),
                                     _shape_of(catalog, r) if line_shapes else None))
    if not out:
        raise ValueError("nd_duality: the catalog carries no low-mass (1, 1, 0) record with a width")
    return tuple(out)


_I12_CACHE: dict = {}


def isoscalar_exchange_records(catalog: HadronCatalog, line_shapes: bool = False) -> tuple[DualityRecord, ...]:
    """The I = 1/2 records of :func:`duality_records`: the states a vacuum
    (I = 0) exchange excites the proton to, for the diffractive duality
    (``TopologyNominalParams.diffractive_duality_mass``, 2026-10-15)."""
    key = (id(catalog), bool(line_shapes))
    hit = _I12_CACHE.get(key)
    if hit is not None and hit[0] is catalog:
        return hit[1]
    recs = tuple(r for r in duality_records(catalog, line_shapes) if float(catalog.get(r.pdg_id).isospin) == 0.5)
    if not recs:
        raise ValueError("nd_duality: the catalog carries no low-mass I = 1/2 (1, 1, 0) record with a width")
    _I12_CACHE[key] = (catalog, recs)
    return recs


def _log_law_integral(eps: float, a2: float, c2: float) -> float:
    """``int_a2^c2 x^(-1-eps) dx`` (the logarithmic limit at eps = 0)."""
    if abs(eps) < 1e-12:
        return math.log(c2 / a2)
    return (math.expm1(-eps * math.log(c2)) - math.expm1(-eps * math.log(a2))) / (-eps)


def duality_redraw_mass(mass: float, lo: float, shape_mass: float, eps: float,
                        records: tuple[DualityRecord, ...]) -> float | None:
    """The line-shape mass of a continuum line of mass *mass* drawn from
    ``(M^2)^(-1-eps)`` on a window starting at *lo*, or ``None`` when the
    line keeps its mass (at or above *shape_mass*, or no record inside
    ``[lo, shape_mass]``).  Pure: the continuum mass's own CDF value is the
    uniform."""
    m, lo, ms = float(mass), float(lo), float(shape_mass)
    if not (lo < m < ms):
        return None
    parts = []
    for r in records:
        a, b = max(lo, r.lo), min(ms, r.hi)
        if not b > a:
            continue
        mg = r.pole * r.width
        ta = math.atan((a * a - r.pole * r.pole) / mg)
        tb = math.atan((b * b - r.pole * r.pole) / mg)
        w = r.weight * (tb - ta) / r.pole
        if w > 0.0:
            parts.append((w, r, ta, tb))
    if not parts:
        return None
    v = _log_law_integral(eps, lo * lo, m * m) / _log_law_integral(eps, lo * lo, ms * ms)
    v = min(max(v, 0.0), 1.0)
    total = math.fsum(p[0] for p in parts)
    target, acc = v * total, 0.0
    w, r, ta, tb = parts[-1]
    residual = 1.0
    for part in parts:
        if target < acc + part[0]:
            w, r, ta, tb = part
            residual = (target - acc) / part[0]
            break
        acc += part[0]
    s = r.pole * r.pole + r.pole * r.width * math.tan(ta + min(max(residual, 0.0), 1.0) * (tb - ta))
    return float(min(max(math.sqrt(s), max(lo, r.lo)), min(ms, r.hi)))


#: The cumulative line-shape tables of the redraw (pdg id, support) -> (grid, cdf), pure compute.
_SHAPE_CDF_CACHE: dict = {}
_SHAPE_CDF_POINTS = 3001


def _shape_cdf(r: DualityRecord):
    key = (int(r.pdg_id), float(r.lo), float(r.hi), id(r.shape.keywords.get("catalog")) if hasattr(r.shape, "keywords") else 0)
    hit = _SHAPE_CDF_CACHE.get(key)
    if hit is None:
        n = _SHAPE_CDF_POINTS
        step = (float(r.hi) - float(r.lo)) / (n - 1)
        grid = [float(r.lo) + k * step for k in range(n)]
        rho = [float(r.shape(x)) for x in grid]
        cdf = [0.0]
        for k in range(1, n):
            cdf.append(cdf[-1] + 0.5 * (rho[k] + rho[k - 1]) * (grid[k] - grid[k - 1]))
        hit = _SHAPE_CDF_CACHE[key] = (grid, cdf)
    return hit


def _interp(x: float, xs, ys) -> float:
    """Piecewise-linear interpolation of the increasing table (xs, ys) at x, clamped to its ends."""
    import bisect
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    k = bisect.bisect_right(xs, x)
    x0, x1, y0, y1 = xs[k - 1], xs[k], ys[k - 1], ys[k]
    return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _inverse(c: float, grid, cdf) -> float:
    """The mass at the cumulative value c of a record's table (the first cell whose cdf reaches c)."""
    import bisect
    k = bisect.bisect_left(cdf, c)
    if k <= 0:
        return grid[0]
    if k >= len(cdf):
        return grid[-1]
    c0, c1 = cdf[k - 1], cdf[k]
    return grid[k] if c1 == c0 else grid[k - 1] + (grid[k] - grid[k - 1]) * (c - c0) / (c1 - c0)


def duality_redraw_mass_shapes(mass: float, lo: float, shape_mass: float, eps: float,
                               records: tuple[DualityRecord, ...]) -> float | None:
    """:func:`duality_redraw_mass` with the records' own normalized line
    shapes (each record carries ``shape``): the record by its (2J + 1)
    line-shape integral over the window, then the mass by that record's
    cumulative line shape at the residual.  Pure: the continuum mass's own
    CDF value is the uniform; records without a shape raise."""
    m, lo, ms = float(mass), float(lo), float(shape_mass)
    if not (lo < m < ms):
        return None
    parts = []
    for r in records:
        if r.shape is None:
            raise ValueError("duality_redraw_mass_shapes: the records carry no line shapes (duality_line_shapes)")
        a, b = max(lo, r.lo), min(ms, r.hi)
        if not b > a:
            continue
        grid, cdf = _shape_cdf(r)
        ca, cb = _interp(a, grid, cdf), _interp(b, grid, cdf)
        w = r.weight * (cb - ca)
        if w > 0.0:
            parts.append((w, r, ca, cb, grid, cdf))
    if not parts:
        return None
    v = _log_law_integral(eps, lo * lo, m * m) / _log_law_integral(eps, lo * lo, ms * ms)
    v = min(max(v, 0.0), 1.0)
    total = math.fsum(p[0] for p in parts)
    target, acc = v * total, 0.0
    sel, residual = parts[-1], 1.0
    for part in parts:
        if target < acc + part[0]:
            sel, residual = part, (target - acc) / part[0]
            break
        acc += part[0]
    _w, r, ca, cb, grid, cdf = sel
    mm = _inverse(ca + min(max(residual, 0.0), 1.0) * (cb - ca), grid, cdf)
    return float(min(max(mm, max(lo, r.lo)), min(ms, r.hi)))


def duality_pick(mass: float, records: tuple[DualityRecord, ...], u: float) -> int | None:
    """The record a line of mass *mass* decays as: each record whose mass
    range holds the mass, with weight ``(2J + 1) Gamma / ((M^2 - m^2)^2 +
    m^2 Gamma^2)`` (``(2J + 1)`` times its line shape when the record
    carries one), at the uniform *u*; ``None`` when no record holds it."""
    m2 = float(mass) ** 2
    cand = [(r.weight * (r.shape(float(mass)) if r.shape is not None
                         else r.width / ((m2 - r.pole * r.pole) ** 2 + (r.pole * r.width) ** 2)), r.pdg_id)
            for r in records if r.lo < float(mass) < r.hi]
    if not cand:
        return None
    target, acc = float(u) * math.fsum(w for w, _ in cand), 0.0
    for w, pdg in cand:
        acc += w
        if target < acc:
            return pdg
    return cand[-1][1]
