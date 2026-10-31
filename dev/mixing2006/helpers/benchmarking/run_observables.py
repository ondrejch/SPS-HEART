"""Observables of the staged topology runs, shared by the Round 5
follow-up and Round 6 drivers (review record code fix C2).

Conventions (the sealed R01/R51/R55 estimator):

- the lab to overall-CM boost of the pinned beam configuration,
  ``beta = p_beam / (E_beam + m_p)``, ``gamma = (E_beam + m_p) / sqrt_s``
  (``topology_runner._cm_boost_parameters``).  The function defaults use
  :data:`SQRT_S_CM`, the exact invariant mass of the beam pair
  (17.2702 GeV), so ``gamma`` and ``beta`` describe one Lorentz boost and
  ``x_F`` divides by the CM energy (2026-10-07; review 2026-10-06, Muse
  L-1).  :data:`SQRT_S` = 17.270 GeV is the generators' recorded draw
  value (the mass-marginal ranges of card 4), which a reweighting must
  pass to ``log_density_ratio`` to mirror the draw;
- ``x_F = 2 p_L^CM / sqrt_s``, ``p_T`` invariant, the per-track Jacobian
  weight ``E^CM / p_T``;
- the per-bin prediction ``M = sum(E/p_T) / (pi sqrt_s N_events dx_F dp_T)``
  at the table's midpoint-edge grid, equal to the bin average of
  ``f / sigma_inel`` in 1/GeV^2; the data are the table ``f`` divided by
  ``sigma_inel = 31.78`` mb, tabulated at the bin centers.
  :func:`bin_center_factors` converts the bin average to the center value
  from the data's own local shape (the bin-center correction, 2026-10-07).

Species are the NA49 p+p tables of ``config/staged_excitation.json``
(``fit.binned.tables``): pi_minus, pi_plus, k_plus, k_minus, proton, antiproton.  The default window is
the paper window (x_F in [0, 0.4], p_T in [0.05, 1.5]); a driver may open
the x_F window for the leading-baryon region of the proton table.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import sys

import numpy as np

_HERE = Path(__file__).resolve()
for _base in (_HERE.parents[2], _HERE.parents[2] / "dev" / "mixing2006"):
    if (_base / "na49_mixing").is_dir() and str(_base) not in sys.path:
        sys.path.insert(0, str(_base))
        break
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from na49_mixing.staged_evaluation import BinnedFitTable  # noqa: E402
from na49_mixing.topology_remnants import REMNANT_KIND_INDEX
from na49_mixing.topology_elastic import BEAM_MOMENTUM, PROTON_MASS  # noqa: E402
from na49_mixing.topology_mass_laws import PINNED_SQRT_S_EXACT  # noqa: E402

import sps_core  # noqa: E402  (the helper tree's own module; both trees carry it)

__all__ = [
    "BEAM_MOMENTUM",
    "PROTON_MASS",
    "bin_areas",
    "bin_center_factors",
    "bin_center_factors_1d",
    "SQRT_S",
    "SQRT_S_CM",
    "SIGMA_INEL_MB",
    "SPECIES_ORDER",
    "SPECIES_PDG",
    "cm_boost",
    "to_cm",
    "load_tables",
    "track_arrays",
    "bin_track_arrays",
    "prediction_from_sums",
    "leading_proton_xf",
    "charged_multiplicity",
    "shape_arrays",
    "strange_counts",
    "read_json",
]

#: the generators' recorded draw value (the card-4 mass-marginal ranges);
#: a reweighting passes it to ``log_density_ratio`` to mirror the draw
SQRT_S = 17.270
#: the exact invariant mass of the pinned beam pair: the CM boost, x_F and
#: the estimator's pi sqrt(s) (2026-10-07)
SQRT_S_CM = PINNED_SQRT_S_EXACT
SIGMA_INEL_MB = 31.78
SPECIES_ORDER = ("pi_minus", "pi_plus", "k_plus", "k_minus", "proton", "antiproton")
SPECIES_PDG = {"pi_minus": -211, "pi_plus": 211, "k_plus": 321, "k_minus": -321, "proton": 2212, "antiproton": -2212}
SPECIES_MASS = {"pi_minus": 0.13957039, "pi_plus": 0.13957039, "k_plus": 0.493677, "k_minus": 0.493677,
                "proton": 0.93827208816, "antiproton": 0.93827208816}
STRANGE_GROUPS = {
    "k_plus": (321,), "k_minus": (-321,), "k_zero": (311, -311, 310, 130),
    "lambda": (3122,), "antilambda": (-3122,), "sigma": (3222, 3212, 3112),
    "antisigma": (-3222, -3212, -3112), "pbar": (-2212,), "antineutron": (-2112,),
}


def read_json(path: Path):
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cm_boost(sqrt_s: float = SQRT_S_CM) -> tuple[float, float]:
    e_beam = math.sqrt(PROTON_MASS * PROTON_MASS + BEAM_MOMENTUM * BEAM_MOMENTUM)
    e_total = e_beam + PROTON_MASS
    return BEAM_MOMENTUM / e_total, e_total / float(sqrt_s)


def to_cm(p4, beta: float, gamma: float) -> tuple[float, float, float, float]:
    e, px, py, pz = (float(x) for x in p4)
    return gamma * (e - beta * pz), px, py, gamma * (pz - beta * e)


def load_tables(nested_root: Path, *, species: Sequence[str] = SPECIES_ORDER,
                windows: dict | None = None, config_rel: str = "config/staged_excitation.json") -> dict:
    """``{species: BinnedFitTable}`` from the config's binned tables, each
    with its own window (``windows[species]``, default the paper window)."""
    config = read_json(Path(nested_root) / config_rel)
    entries = {e["species"]: e for e in config["fit"]["binned"]["tables"]}
    windows = windows or {}
    out = {}
    for name in species:
        entry = entries[name]
        out[name] = BinnedFitTable.from_path(
            Path(nested_root) / entry["path"], species=name, pdg_id=int(entry["pdg"]),
            mass_gev=float(entry["mass_gev"]), normalization=str(entry["normalization"]),
            sigma_inel_mb=SIGMA_INEL_MB, window=windows.get(name),
        )
    return out


def track_arrays(finals: Iterable, species_pdg: dict, beta: float, gamma: float,
                 sqrt_s: float = SQRT_S_CM) -> dict:
    """Compact per-track arrays ``{species: (x_F, p_T, w)}`` of one event's
    final state for the requested species (``w = E^CM / p_T``; tracks with
    ``p_T = 0`` are dropped, as in the sealed estimator)."""
    by_pdg = {}
    for name, pdg in species_pdg.items():
        by_pdg.setdefault(int(pdg), []).append(name)
    out = {name: ([], [], []) for name in species_pdg}
    for h in finals:
        pdg = getattr(h, "pdg_id", None)
        if pdg is None or int(pdg) not in by_pdg:
            continue
        e_cm, px, py, pz_cm = to_cm(h.four_momentum, beta, gamma)
        pt = math.hypot(px, py)
        if pt <= 0.0:
            continue
        xf = 2.0 * pz_cm / float(sqrt_s)
        for name in by_pdg[int(pdg)]:
            xs, ps, ws = out[name]
            xs.append(xf); ps.append(pt); ws.append(e_cm / pt)
    return out


def bin_track_arrays(table: BinnedFitTable, xf, pt, w) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin weight sums and counts at the table's bins (the
    ``_bin_track_sums`` convention: midpoint edges, ``grid_indices``,
    pT-major flat layout, in-bounds drop)."""
    xedges = sps_core.edges_from_centers(table.xf_centers)
    pedges = sps_core.edges_from_centers(table.pt_centers)
    n_x = xedges.size - 1
    n_p = pedges.size - 1
    pick = table.bin_index_pt * n_x + table.bin_index_xf
    xf = np.asarray(xf, dtype=float); pt = np.asarray(pt, dtype=float); w = np.asarray(w, dtype=float)
    if xf.size == 0:
        return np.zeros(pick.size), np.zeros(pick.size, dtype=np.int64)
    ix, ip = sps_core.grid_indices(xf, pt, xedges, pedges)
    ok = (ix >= 0) & (ix < n_x) & (ip >= 0) & (ip < n_p)
    flat = ip[ok] * n_x + ix[ok]
    sums = np.bincount(flat, weights=w[ok], minlength=n_p * n_x)[: n_p * n_x]
    counts = np.bincount(flat, minlength=n_p * n_x)[: n_p * n_x]
    return np.asarray(sums[pick], dtype=float), np.asarray(counts[pick], dtype=np.int64)


def bin_areas(table: BinnedFitTable) -> np.ndarray:
    xedges = sps_core.edges_from_centers(table.xf_centers)
    pedges = sps_core.edges_from_centers(table.pt_centers)
    n_x = xedges.size - 1
    area = (np.diff(pedges)[:, None] * np.diff(xedges)[None, :]).ravel()
    return area[table.bin_index_pt * n_x + table.bin_index_xf]


_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(16)


def _local_log_shape(xs: np.ndarray, gs: np.ndarray, x0: float, rel=None):
    """The local law of ``g = log f`` around *x0* as a callable of
    ``x - x0``, normalized to 0 at ``x0``: the weighted least-squares
    quadratic through the five nearest tabulated points of the row
    (one-sided at its ends; linear with two points, flat with one), with
    weights ``1 / (rel**2 + 0.02**2)`` from the points' relative errors
    *rel*, so a poorly measured point cannot set the curvature.  Three
    points give the exact interpolating quadratic.  At the end point of a
    row the bin reaches beyond the last measurement, and the law is the
    fitted tangent at *x0* (no extrapolated curvature)."""
    order = np.argsort(xs)
    xs, gs = xs[order], gs[order]
    rel = np.zeros(xs.size) if rel is None else np.asarray(rel, dtype=float)[order]
    i = int(np.argmin(np.abs(xs - x0)))
    lo = min(max(i - 2, 0), max(xs.size - 5, 0))
    sel = slice(lo, min(lo + 5, xs.size))
    u, g, r = xs[sel] - x0, gs[sel], rel[sel]
    if u.size == 1:
        return lambda v: 0.0 * v
    deg = 2 if u.size >= 3 else 1
    w = np.sqrt(1.0 / (r * r + 0.02 * 0.02))
    coef = np.polynomial.polynomial.polyfit(u, g, deg, w=w)
    slope, curv = float(coef[1]), (float(coef[2]) if deg == 2 else 0.0)
    if i == 0 or i == xs.size - 1:
        curv = 0.0
    return lambda v: slope * v + curv * v * v


def _mean_exp(shape, lo: float, hi: float, x0: float) -> float:
    """``(1 / (hi - lo)) int_lo^hi exp(shape(x - x0)) dx`` (16-point
    Gauss-Legendre)."""
    x = 0.5 * (hi - lo) * _GL_NODES + 0.5 * (hi + lo)
    return float(0.5 * np.sum(_GL_WEIGHTS * np.exp(shape(x - x0))))


def bin_center_factors(table: BinnedFitTable) -> np.ndarray:
    """Per-bin factors that convert the model's bin average of ``f`` to its
    value at the tabulated point (the bin-center correction, 2026-10-07;
    card TASK-20261005-bin-center-convention, review 2026-10-06 Muse M-5).

    The data are invariant cross sections at the tabulated ``(x_F, p_T)``
    points; :func:`prediction_from_sums` is the average of ``f`` over the
    point's midpoint-edge bin.  The factor is ``f(x0, p0) / <f>_bin`` for
    the data's own local shape: ``log f`` through the point and its nearest
    tabulated neighbors along ``x_F`` at fixed ``p_T`` and along ``p_T`` at
    fixed ``x_F`` (the full table, so the window's edge points keep their
    outer neighbors), quadratic with two neighbors, linear with one, and
    the separable product of the two one-dimensional averages.  The factors
    depend on the data only, so they are fixed per bin and leave the
    reweighting untouched.  Points without a positive value use 1.
    """
    xf_all, pt_all, val_all, err_all = sps_core.read_na49_points(table.path)
    xf_all, pt_all, val_all, err_all = (np.asarray(a, dtype=float) for a in (xf_all, pt_all, val_all, err_all))
    good = np.isfinite(val_all) & (val_all > 0.0)
    rel_all = np.where(np.isfinite(err_all) & (err_all > 0.0), err_all / np.where(good, val_all, 1.0), 0.0)
    xedges = sps_core.edges_from_centers(table.xf_centers)
    pedges = sps_core.edges_from_centers(table.pt_centers)
    out = np.ones(table.data.size)
    for k in range(table.data.size):
        ix, ip = int(table.bin_index_xf[k]), int(table.bin_index_pt[k])
        x0, p0 = float(table.xf_centers[ix]), float(table.pt_centers[ip])
        here = good & np.isclose(xf_all, x0, atol=1e-9) & np.isclose(pt_all, p0, atol=1e-9)
        if not here.any():
            continue
        row = good & np.isclose(pt_all, p0, atol=1e-9)
        col = good & np.isclose(xf_all, x0, atol=1e-9)
        shape_x = _local_log_shape(xf_all[row], np.log(val_all[row]), x0, rel_all[row])
        shape_p = _local_log_shape(pt_all[col], np.log(val_all[col]), p0, rel_all[col])
        mean = (_mean_exp(shape_x, float(xedges[ix]), float(xedges[ix + 1]), x0)
                * _mean_exp(shape_p, float(pedges[ip]), float(pedges[ip + 1]), p0))
        out[k] = 1.0 / mean
    return out


def bin_center_factors_1d(centers, values, edges, errors=None) -> np.ndarray:
    """:func:`bin_center_factors` for a one-dimensional distribution
    tabulated at *centers* with the bin *edges* (the neutron dN/dx_F)."""
    centers = np.asarray(centers, dtype=float); values = np.asarray(values, dtype=float)
    edges = np.asarray(edges, dtype=float)
    good = np.isfinite(values) & (values > 0.0)
    rel = (np.zeros(centers.size) if errors is None
           else np.asarray(errors, dtype=float) / np.where(good, values, 1.0))
    out = np.ones(centers.size)
    for k in range(centers.size):
        if not good[k]:
            continue
        shape = _local_log_shape(centers[good], np.log(values[good]), float(centers[k]), rel[good])
        out[k] = 1.0 / _mean_exp(shape, float(edges[k]), float(edges[k + 1]), float(centers[k]))
    return out


def prediction_from_sums(table: BinnedFitTable, sums, n_events: float, sqrt_s: float = SQRT_S_CM) -> np.ndarray:
    """``M = sums / (pi sqrt_s N dx_F dp_T)``, the ``f / sigma_inel`` estimate."""
    return np.asarray(sums, dtype=float) / (math.pi * float(sqrt_s) * float(n_events) * bin_areas(table))


def leading_proton_xf(finals: Iterable, beta: float, gamma: float, sqrt_s: float = SQRT_S_CM):
    """The largest proton ``x_F`` of the event (the projectile-side leading
    baryon), or ``None`` without a final proton."""
    best = None
    for h in finals:
        if getattr(h, "pdg_id", None) is None or int(h.pdg_id) != 2212:
            continue
        xf = 2.0 * to_cm(h.four_momentum, beta, gamma)[3] / float(sqrt_s)
        best = xf if best is None else max(best, xf)
    return best


def charged_multiplicity(finals: Iterable) -> int:
    return sum(1 for h in finals if int(h.qn[1]) != 0)


def shape_arrays(event) -> dict:
    """The mass-law arrays of one event for
    :func:`na49_mixing.topology_mass_laws.log_density_ratio`."""
    label = event.class_label
    if label in ("sd_projectile", "sd_target"):
        out = {"m2": float(event.excitation_mass) ** 2, "t": float(event.t), "m_rem": float(event.remnant.mass)}
        # the line's own lower mass edge under line_thresholds (2026-10-06)
        edge = getattr(event, "excitation_m_min", None)
        if edge is not None:
            out["m2_low"] = float(edge) ** 2
        # the leading-proton tail (2026-10-23, R112): 1 for a tail event, 0 for a
        # coherence-region one, absent without the tail component
        tail = getattr(event, "tail", None)
        if tail is not None:
            out["tail"] = 1.0 if tail else 0.0
        # the one-pion exchange (2026-10-26): 1 for a p + p -> n + X++ event, 0 otherwise, absent without it
        ope = getattr(event, "ope", None)
        if ope is not None:
            out["ope"] = 1.0 if ope else 0.0
        _slope_factor(event, out)
        return out
    if label == "dd":
        out = {"m12": float(event.m1) ** 2, "m22": float(event.m2) ** 2, "t1": float(event.t1), "t2": float(event.t2)}
        edges = getattr(event, "line_m_min", None)
        if edges is not None:
            out["m_lo1"], out["m_lo2"] = float(edges[0]), float(edges[1])
        _slope_factor(event, out)
        return out
    if label == "nd":
        # the remnant kinds and excitation windows (2026-09-29): the exponent
        # reweighting applies the excited kind's own mass law per line
        windows = getattr(event, "remnant_windows", ((math.nan, math.nan), (math.nan, math.nan)))
        out = {"t1": float(event.t1), "t2": float(event.t2), "t_c": float(event.t_c),
               "mc2": float(event.central_mass) ** 2,
               "m1": float(event.remnants[0].mass), "m2": float(event.remnants[1].mass),
               "k1": float(REMNANT_KIND_INDEX[event.remnants[0].kind]),
               "k2": float(REMNANT_KIND_INDEX[event.remnants[1].kind]),
               "lo1": float(windows[0][0]), "hi1": float(windows[0][1]),
               "lo2": float(windows[1][0]), "hi2": float(windows[1][1])}
        # the shape edges of the lines the records' line shapes redrew
        # (2026-10-12, na49_mixing.nd_duality; nan for a continuum mass)
        edges = getattr(event, "duality_edges", None)
        if edges is not None:
            out["ms1"], out["ms2"] = float(edges[0]), float(edges[1])
        _slope_factor(event, out)
        return out
    raise ValueError(f"shape_arrays: no mass law for class {label!r}")


def _slope_factor(event, out: dict) -> None:
    """The event's factorized vertex slope factor under
    transfer_slope_mass_exponent (2026-10-08), recorded for the transfer
    reweighting; absent for the single slope."""
    factor = getattr(event, "slope_factor", None)
    if factor is not None:
        out["slope_factor"] = float(factor)


def strange_counts(finals: Iterable) -> dict:
    counts = {name: 0 for name in STRANGE_GROUPS}
    for h in finals:
        pdg = getattr(h, "pdg_id", None)
        if pdg is None:
            continue
        for name, ids in STRANGE_GROUPS.items():
            if int(pdg) in ids:
                counts[name] += 1
    return counts
