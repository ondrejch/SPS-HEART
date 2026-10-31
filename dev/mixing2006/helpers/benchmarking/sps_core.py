"""Shared SPS benchmark primitives: I/O, paper window, metrics, simple fits.

This module is the extracted library surface for the large benchmark scripts.
Prefer importing from here in new code and tests; the long CLI drivers re-export
selected helpers for backward compatibility.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import nnls

# Paper / protocol defaults (see _knowledge/operations/benchmark-protocol.md)
BEAM_MOMENTUM_GEV = 158.0
PAPER_XF_MIN = 0.0
PAPER_XF_MAX = 0.4
PAPER_PT_MIN = 0.05
PAPER_PT_MAX = 1.5
MPROTON_GEV = 0.9382720813

FIT_WEIGHT_MODES: Tuple[str, ...] = ("sigma", "relative", "relative_xfpt_balanced")


def sqrt_s_from_fixed_target_beam_momentum(
    beam_p_lab: float,
    mproj: float = MPROTON_GEV,
    mtarg: float = MPROTON_GEV,
) -> float:
    e_beam = math.sqrt(beam_p_lab * beam_p_lab + mproj * mproj)
    s = mproj * mproj + mtarg * mtarg + 2.0 * mtarg * e_beam
    return math.sqrt(max(s, 0.0))


def disambiguate_duplicate_pt_labels(xf: np.ndarray, pt: np.ndarray) -> np.ndarray:
    """Recover distinct pT centers when a table reprints the same label twice.

    Thesis p+Pb tables print adjacent bins with one rounded pT (0.10 twice,
    0.20 twice). Unique-ifying those labels maps both measurements onto one
    histogram cell. Within each xF group a run of k copies of printed label
    L after the previous distinct printed label L_prev is expanded to
    L_prev + i*(L - L_prev)/k for i = 1..k, which reconstructs the 0.05-step
    fine grid used by the un-rounded thesis tables. Groups without repeated
    labels are left unchanged.
    """
    xf_arr = np.asarray(xf, dtype=float)
    printed_pt = np.asarray(pt, dtype=float)
    pt_out = printed_pt.copy()
    n = int(pt_out.size)
    if n == 0:
        return pt_out
    i = 0
    while i < n:
        j = i + 1
        while j < n and xf_arr[j] == xf_arr[i]:
            j += 1
        prev_printed: Optional[float] = None
        k = i
        while k < j:
            m = k + 1
            while m < j and printed_pt[m] == printed_pt[k]:
                m += 1
            run_len = m - k
            label = float(printed_pt[k])
            if run_len > 1:
                lo = (
                    label - 0.10 * run_len
                    if prev_printed is None
                    else float(prev_printed)
                )
                step = (label - lo) / run_len
                for r in range(run_len):
                    pt_out[k + r] = lo + (r + 1) * step
            prev_printed = label
            k = m
        i = j
    return pt_out


def read_na49_points(path: Path | str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read SPS HADRONS table: columns xF pT value error."""
    xs: List[float] = []
    pts: List[float] = []
    ys: List[float] = []
    es: List[float] = []
    with Path(path).open(encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) < 4:
                continue
            xf, pt, val, err = map(float, cols[:4])
            xs.append(xf)
            pts.append(pt)
            ys.append(val)
            es.append(err)
    xf_arr = np.asarray(xs, dtype=float)
    pt_arr = np.asarray(pts, dtype=float)
    return (
        xf_arr,
        disambiguate_duplicate_pt_labels(xf_arr, pt_arr),
        np.asarray(ys, dtype=float),
        np.asarray(es, dtype=float),
    )


def paper_window_mask(
    xf: np.ndarray,
    pt: np.ndarray,
    sigma: np.ndarray | None = None,
    *,
    xf_min: float = PAPER_XF_MIN,
    xf_max: float = PAPER_XF_MAX,
    pt_min: float = PAPER_PT_MIN,
    pt_max: float = PAPER_PT_MAX,
    require_positive_sigma: bool = True,
) -> np.ndarray:
    mask = (
        (np.asarray(xf, dtype=float) >= float(xf_min))
        & (np.asarray(xf, dtype=float) <= float(xf_max))
        & (np.asarray(pt, dtype=float) >= float(pt_min))
        & (np.asarray(pt, dtype=float) <= float(pt_max))
    )
    if require_positive_sigma and sigma is not None:
        mask = mask & (np.asarray(sigma, dtype=float) > 0.0)
    return mask


def filter_paper_window(
    xf: np.ndarray,
    pt: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    **window_kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = paper_window_mask(xf, pt, sigma, **window_kwargs)
    return xf[mask], pt[mask], y[mask], sigma[mask]


def edges_from_centers(vals: np.ndarray) -> np.ndarray:
    vals = np.unique(np.asarray(vals, dtype=float))
    vals.sort()
    if vals.size == 1:
        return np.array([vals[0] - 0.5, vals[0] + 0.5], dtype=float)
    mids = 0.5 * (vals[:-1] + vals[1:])
    first = vals[0] - 0.5 * (vals[1] - vals[0])
    last = vals[-1] + 0.5 * (vals[-1] - vals[-2])
    return np.concatenate([[first], mids, [last]])


def grid_indices(
    xf: np.ndarray,
    pt: np.ndarray,
    xedges: np.ndarray,
    pedges: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    ix = np.digitize(xf, xedges) - 1
    ip = np.digitize(pt, pedges) - 1
    return ix, ip


def counts_on_grid(
    sample_xf: np.ndarray,
    sample_pt: np.ndarray,
    xedges: np.ndarray,
    pedges: np.ndarray,
    ix_data: np.ndarray,
    ip_data: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    h, _, _ = np.histogram2d(sample_xf, sample_pt, bins=[xedges, pedges],
                             weights=weights)
    return h[ix_data, ip_data]


# Rest masses (GeV/c^2) of the benchmarked identified-hadron species.
PDG_MASS_GEV: Dict[int, float] = {
    211: 0.1395699,
    -211: 0.1395699,
    321: 0.493677,
    -321: 0.493677,
    2212: 0.93827231,
    -2212: 0.93827231,
    2112: 0.93956541,
    -2112: 0.93956541,
}


def invariant_track_weights(
    pdg: np.ndarray | int,
    xf: np.ndarray,
    pt: np.ndarray,
    sqrt_s_gev: float,
    mass_gev: float | None = None,
) -> np.ndarray:
    """Per-track Jacobian weight E/pT for the invariant cross section.

    With xF = 2 pL / sqrt(s) and azimuthal symmetry, the tabulated NA49
    quantity f = E d^3sigma/dp^3 relates to track densities as
        f_bin ~ (sum_n E_n/pT_n) / (dxF * dpT)
    up to a constant absorbed by fitted normalisations.  This helper returns
    E_n/pT_n with pL = xf * sqrt(s)/2 and the species rest mass.
    """
    xv = np.asarray(xf, dtype=float)
    pv = np.asarray(pt, dtype=float)
    if mass_gev is None:
        pid = int(pdg if np.isscalar(pdg) else np.asarray(pdg).ravel()[0])
        try:
            m = PDG_MASS_GEV[pid]
        except KeyError as exc:
            raise ValueError(f"unknown species pdg {pid} for invariant weights") from exc
    else:
        m = float(mass_gev)
    pl = xv * (0.5 * float(sqrt_s_gev))
    energy = np.sqrt(m * m + pl * pl + pv * pv)
    return energy / np.maximum(pv, 1e-12)


def grid_point_areas(
    xedges: np.ndarray,
    pedges: np.ndarray,
    ix_data: np.ndarray,
    ip_data: np.ndarray,
) -> np.ndarray:
    """Bin area dx*dp for every data point (for count->density division)."""
    dx = np.diff(np.asarray(xedges, dtype=float))
    dp = np.diff(np.asarray(pedges, dtype=float))
    return dx[np.asarray(ix_data, dtype=np.int64)] * dp[np.asarray(ip_data, dtype=np.int64)]


def fit_single_scale(
    train_pred: np.ndarray,
    train_y: np.ndarray,
    train_sigma: np.ndarray,
    row_scale: np.ndarray | None = None,
) -> float:
    if row_scale is None:
        w = 1.0 / np.maximum(train_sigma, 1e-12) ** 2
    else:
        rs = np.asarray(row_scale, dtype=float)
        w = np.maximum(rs, 0.0) ** 2
    num = np.sum(w * train_pred * train_y)
    den = np.sum(w * train_pred * train_pred)
    if den <= 0:
        return 0.0
    return float(num / den)


def rank_bin_indices(vals: np.ndarray, n_bins: int) -> np.ndarray:
    n = int(vals.shape[0])
    nb = max(int(n_bins), 1)
    if n == 0 or nb == 1:
        return np.zeros(n, dtype=np.int64)
    order = np.argsort(np.asarray(vals, dtype=float), kind="mergesort")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n, dtype=np.int64)
    bins = (ranks * nb) // max(n, 1)
    bins[bins >= nb] = nb - 1
    return bins


def compute_fit_row_scales(
    *,
    y: np.ndarray,
    sigma: np.ndarray,
    xf: np.ndarray,
    pt: np.ndarray,
    mode: str,
    balance_xf_bins: int,
    balance_pt_bins: int,
) -> Tuple[np.ndarray, Dict[str, float | int | str]]:
    """Return per-row objective scales for linear least-squares fits.

    Returned ``row_scale`` is applied as ``A * row_scale[:,None]`` and
    ``b * row_scale``.
    """
    mode_norm = str(mode).strip().lower()
    if mode_norm not in FIT_WEIGHT_MODES:
        raise ValueError(f"Unsupported fit weight mode '{mode}'. Valid: {FIT_WEIGHT_MODES}")

    yv = np.asarray(y, dtype=float)
    sv = np.asarray(sigma, dtype=float)
    xv = np.asarray(xf, dtype=float)
    pv = np.asarray(pt, dtype=float)
    sigma_safe = np.maximum(sv, 1e-12)

    if mode_norm == "sigma":
        rs = 1.0 / sigma_safe
        return rs, {"mode": mode_norm}

    rel_scale = np.maximum(yv, 1e-30) / sigma_safe
    if mode_norm == "relative":
        return rel_scale, {"mode": mode_norm}

    # relative_xfpt_balanced — match historical benchmark_sps_species formula
    n_xf = max(int(balance_xf_bins), 1)
    n_pt = max(int(balance_pt_bins), 1)
    xb = rank_bin_indices(xv, n_xf)
    pb = rank_bin_indices(pv, n_pt)
    gid = xb * n_pt + pb
    n_groups = n_xf * n_pt

    w_raw = rel_scale * rel_scale
    sum_by_group = np.zeros(n_groups, dtype=float)
    np.add.at(sum_by_group, gid, w_raw)

    nz = sum_by_group > 0.0
    if not np.any(nz):
        return rel_scale, {
            "mode": mode_norm,
            "xf_bins": n_xf,
            "pt_bins": n_pt,
            "nonempty_groups": 0,
        }

    target = float(np.mean(sum_by_group[nz]))
    factors = np.ones(n_groups, dtype=float)
    factors[nz] = target / np.maximum(sum_by_group[nz], 1e-30)
    w_bal = w_raw * factors[gid]
    rs = np.sqrt(np.maximum(w_bal, 0.0))

    sum_bal = np.zeros(n_groups, dtype=float)
    np.add.at(sum_bal, gid, w_bal)
    shares = sum_bal[nz] / np.maximum(np.sum(sum_bal[nz]), 1e-30)
    return rs, {
        "mode": mode_norm,
        "xf_bins": n_xf,
        "pt_bins": n_pt,
        "nonempty_groups": int(np.count_nonzero(nz)),
        "target_group_weight": target,
        "group_share_min": float(np.min(shares)),
        "group_share_max": float(np.max(shares)),
    }


def chi2(pred: np.ndarray, y: np.ndarray, sigma: np.ndarray, n_params: int) -> Tuple[float, float]:
    r = (pred - y) / np.maximum(sigma, 1e-12)
    chi2_val = float(np.sum(r * r))
    ndof = max(len(y) - n_params, 1)
    return chi2_val, chi2_val / ndof


def log_chi2(
    pred: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    n_params: int,
) -> Tuple[float, float, int]:
    mask = (pred > 0.0) & (y > 0.0) & (sigma > 0.0)
    if not np.any(mask):
        return 0.0, 0.0, 0
    pred_m = pred[mask]
    y_m = y[mask]
    sigma_m = sigma[mask]
    sigma_log = np.maximum(sigma_m / np.maximum(y_m, 1e-30), 1e-12)
    r_log = (np.log(pred_m) - np.log(y_m)) / sigma_log
    chi2_log = float(np.sum(r_log * r_log))
    ndof = max(int(mask.sum()) - int(n_params), 1)
    return chi2_log, chi2_log / ndof, int(mask.sum())


def smape_percent(pred: np.ndarray, y: np.ndarray) -> float:
    denom = np.maximum((np.abs(pred) + np.abs(y)) * 0.5, 1e-12)
    return float(100.0 * np.mean(np.abs(pred - y) / denom))


def relative_rmse_weighted(pred: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> float:
    """Weighted relative RMSE (WRRMSE) used as a primary ranking metric."""
    mask = (y > 0.0) & (sigma > 0.0)
    if not np.any(mask):
        return 0.0
    rel = (pred[mask] - y[mask]) / np.maximum(y[mask], 1e-30)
    rel_sigma = np.maximum(sigma[mask] / np.maximum(y[mask], 1e-30), 1e-12)
    w = 1.0 / (rel_sigma * rel_sigma)
    return float(np.sqrt(np.sum(w * rel * rel) / np.sum(w)))


def metrics_bundle(
    pred: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    n_params: int,
    prefix: str,
) -> Dict[str, float | int]:
    c2, c2_dof = chi2(pred, y, sigma, n_params=n_params)
    c2_log, c2_log_dof, n_log = log_chi2(pred, y, sigma, n_params=n_params)
    payload: Dict[str, float | int] = {
        f"{prefix}_chi2": c2,
        f"{prefix}_chi2_dof": c2_dof,
        f"{prefix}_smape_percent": smape_percent(pred, y),
        f"{prefix}_weighted_rel_rmse": relative_rmse_weighted(pred, y, sigma),
    }
    if n_log > 0:
        payload[f"{prefix}_log_chi2"] = c2_log
        payload[f"{prefix}_log_chi2_dof"] = c2_log_dof
        payload[f"{prefix}_log_points"] = n_log
        payload[f"{prefix}_log_pull_rms"] = float(np.sqrt(c2_log / max(n_log, 1)))
    return payload


# Numerical slack allowed on the non-negativity bound of a fitted weight
# vector. L-BFGS-B and SLSQP can return entries a few ulp below zero even with
# an explicit lower bound of 0.0, and both fit paths clip such tiny negatives
# to exactly 0.0. A raw solver output below -FIT_BOUND_CLIP_TOLERANCE is a real
# bound violation and is rejected instead of clipped.
FIT_BOUND_CLIP_TOLERANCE = 1e-9

# Identifiability stamps recorded in every fit-solver info payload. A fit whose
# design has full numerical column rank is stamped "full_rank_identifiable".
# A fit that accepted a rank-deficient design through the explicit
# allow_rank_deficient opt-out is stamped "rank_deficient_non_identifiable",
# so no JSON or Markdown artifact can present that fit as identifiable
# without reading the stamp.
FULL_RANK_STAMP = "full_rank_identifiable"
RANK_DEFICIENT_STAMP = "rank_deficient_non_identifiable"

# Active-weight convention shared with the benchmark's n_eff counter: a
# component counts as active/free at the solution when its weight exceeds
# this threshold.
ACTIVE_WEIGHT_THRESHOLD = 1e-12


def _first_nonfinite_note(a: np.ndarray, *, two_d: bool) -> str:
    """Describe the first non-finite entry of an array for error messages."""
    bad = np.flatnonzero(~np.isfinite(a))
    i = int(bad[0])
    value = a.flat[i]
    if two_d and a.ndim == 2:
        row, col = divmod(i, int(a.shape[1]))
        return (
            f"first offender at row {row}, column {col} "
            f"(flat index {i}) with value {float(value)!r}"
        )
    return f"first offender at index {i} with value {float(value)!r}"


def validate_fit_design(
    A: np.ndarray,
    n_components: int,
    *,
    name: str = "design matrix",
) -> np.ndarray:
    """Validate a 2-D design matrix against the component count.

    Returns the input as a float ndarray. Raises ValueError when the matrix is
    not 2-D, when its column count differs from n_components, or when any
    entry is non-finite. The message names the matrix, its shape, and the
    first offending entry.
    """
    a = np.asarray(A, dtype=float)
    n_comp = int(n_components)
    if a.ndim != 2:
        raise ValueError(
            f"fit {name} must be 2-D with shape (n_points, n_components={n_comp}); "
            f"got ndim={a.ndim}, shape={a.shape}"
        )
    if a.shape[1] != n_comp:
        raise ValueError(
            f"fit {name} has {a.shape[1]} columns but the fit declares "
            f"n_components={n_comp}; expected shape (n_points, {n_comp}), got {a.shape}"
        )
    if not bool(np.isfinite(a).all()):
        raise ValueError(
            f"fit {name} contains non-finite values; {_first_nonfinite_note(a, two_d=True)}"
        )
    return a


def validate_fit_target(
    y: np.ndarray,
    n_rows: int,
    *,
    name: str = "target vector",
) -> np.ndarray:
    """Validate a 1-D target vector against the design-matrix row count.

    Returns the input as a float ndarray. Raises ValueError when the vector is
    not 1-D, when its length differs from n_rows, or when any entry is
    non-finite. The message names the vector, its shape, and the first
    offending entry.
    """
    yv = np.asarray(y, dtype=float)
    n = int(n_rows)
    if yv.ndim != 1:
        raise ValueError(
            f"fit {name} must be 1-D with one entry per design-matrix row ({n}); "
            f"got ndim={yv.ndim}, shape={yv.shape}"
        )
    if yv.shape[0] != n:
        raise ValueError(
            f"fit {name} has length {yv.shape[0]} but the design matrix has {n} rows"
        )
    if not bool(np.isfinite(yv).all()):
        raise ValueError(
            f"fit {name} contains non-finite values; {_first_nonfinite_note(yv, two_d=False)}"
        )
    return yv


def validate_fit_sigma(
    sigma: np.ndarray,
    n_rows: int,
    *,
    name: str = "sigma vector",
) -> np.ndarray:
    """Validate a 1-D uncertainty vector against the design-matrix row count.

    Same length and finiteness contract as validate_fit_target. Flooring of
    non-positive uncertainties (if any) stays the caller's decision; this check
    only guarantees that NaN or Inf uncertainties can never enter a weighted
    fit silently.
    """
    return validate_fit_target(sigma, n_rows, name=name)


def validate_fit_prior_rows(
    A_prior: np.ndarray,
    b_prior: np.ndarray,
    n_components: int,
    *,
    name: str = "prior rows",
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate optional linear-prior rows against the component count.

    The prior design matrix must be 2-D with exactly n_components columns, the
    prior target vector must carry one entry per prior row, and all entries
    must be finite. Returns both inputs as float ndarrays; raises ValueError
    with the offending shape or entry named otherwise.
    """
    ap = np.asarray(A_prior, dtype=float)
    bp = np.asarray(b_prior, dtype=float)
    n_comp = int(n_components)
    if ap.ndim != 2 or ap.shape[1] != n_comp:
        raise ValueError(
            f"fit {name}: prior design matrix A_prior must be 2-D with shape "
            f"(n_rows, n_components={n_comp}); got ndim={ap.ndim}, shape={ap.shape}"
        )
    if bp.ndim != 1 or bp.shape[0] != ap.shape[0]:
        raise ValueError(
            f"fit {name}: prior target vector b_prior must be 1-D with one entry "
            f"per prior row ({ap.shape[0]}); got ndim={bp.ndim}, shape={bp.shape}"
        )
    if not bool(np.isfinite(ap).all()):
        raise ValueError(
            f"fit {name}: prior design matrix A_prior contains non-finite values; "
            f"{_first_nonfinite_note(ap, two_d=True)}"
        )
    if not bool(np.isfinite(bp).all()):
        raise ValueError(
            f"fit {name}: prior target vector b_prior contains non-finite values; "
            f"{_first_nonfinite_note(bp, two_d=False)}"
        )
    return ap, bp


def validate_fit_table9_fraction_arrays(
    rows: np.ndarray,
    targets: np.ndarray,
    n_components: int,
    *,
    name: str = "Table-9 fraction rows",
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate Table-9 yield-fraction prior rows against the component count.

    The fraction rows form a 2-D matrix with exactly n_components columns and
    the targets carry one measured fraction per row; all entries must be
    finite. Returns both inputs as float ndarrays; raises ValueError with the
    offending shape or entry named otherwise.
    """
    t9r = np.asarray(rows, dtype=float)
    t9f = np.asarray(targets, dtype=float)
    n_comp = int(n_components)
    if t9r.ndim != 2 or t9r.shape[1] != n_comp:
        raise ValueError(
            f"fit {name}: fraction row matrix must be 2-D with shape "
            f"(n_rows, n_components={n_comp}); got ndim={t9r.ndim}, shape={t9r.shape}"
        )
    if t9f.ndim != 1 or t9f.shape[0] != t9r.shape[0]:
        raise ValueError(
            f"fit {name}: fraction target vector must be 1-D with one entry per "
            f"fraction row ({t9r.shape[0]}); got ndim={t9f.ndim}, shape={t9f.shape}"
        )
    if not bool(np.isfinite(t9r).all()):
        raise ValueError(
            f"fit {name}: fraction row matrix contains non-finite values; "
            f"{_first_nonfinite_note(t9r, two_d=True)}"
        )
    if not bool(np.isfinite(t9f).all()):
        raise ValueError(
            f"fit {name}: fraction target vector contains non-finite values; "
            f"{_first_nonfinite_note(t9f, two_d=False)}"
        )
    return t9r, t9f


def validate_fit_result(
    w: np.ndarray,
    *,
    n_components: int,
    success: bool,
    status: Optional[int],
    message: str,
    objective: float,
    objective_name: str = "objective",
    context: str = "fit",
    tolerance: float = FIT_BOUND_CLIP_TOLERANCE,
) -> np.ndarray:
    """Validate a fitted weight vector and reject invalid solver outcomes.

    The result is accepted only when the optimizer reports success, the
    objective value is finite, every weight is finite, the result has exactly
    (n_components,) entries, and no weight lies below the lower bound 0.0 by
    more than tolerance. The bound is checked on the raw solver output, before
    the caller clips tiny negative entries to exactly zero, so a genuinely
    negative solution is rejected instead of being silently projected onto the
    bound. There is no upper bound on the weights.

    Raises RuntimeError naming the optimizer verdict, the offending objective
    or weight, the shape, or the bound violation; otherwise returns the input
    as a float ndarray.
    """
    wv = np.asarray(w, dtype=float)
    n_comp = int(n_components)
    if not bool(success):
        raise RuntimeError(
            f"{context} rejected: the optimizer did not converge "
            f"(success={bool(success)!r}, status={status!r}, message={str(message)!r})"
        )
    obj = float(objective)
    if not math.isfinite(obj):
        raise RuntimeError(
            f"{context} rejected: the {objective_name} at the returned solution "
            f"is non-finite ({obj!r})"
        )
    if wv.shape != (n_comp,):
        raise RuntimeError(
            f"{context} rejected: the returned weight vector has shape {wv.shape}; "
            f"expected ({n_comp},)"
        )
    if not bool(np.isfinite(wv).all()):
        bad = ", ".join(
            f"w[{k}]={float(wv[k])!r}"
            for k in range(n_comp)
            if not math.isfinite(float(wv[k]))
        )
        raise RuntimeError(
            f"{context} rejected: the returned weights are non-finite ({bad})"
        )
    k_min = int(np.argmin(wv))
    w_min = float(wv[k_min])
    if w_min < -float(tolerance):
        raise RuntimeError(
            f"{context} rejected: the returned weight w[{k_min}]={w_min!r} lies "
            f"below the lower bound 0.0 by more than the clip tolerance "
            f"{float(tolerance)!r}"
        )
    return wv


def condition_number_2norm(matrix: np.ndarray) -> float:
    """Return the 2-norm condition number of a matrix.

    Uses ``numpy.linalg.cond(matrix, p=2)``, the ratio of the largest to the
    smallest singular value, following the MultiFitter
    ``design_condition_number`` convention in ``na49_mixing/fits.py``. An
    empty matrix (zero rows or zero columns) has no singular values, so the
    result is ``inf`` there instead of a numpy ``LinAlgError``.
    """
    m = np.asarray(matrix, dtype=float)
    if m.size == 0:
        return float("inf")
    return float(np.linalg.cond(m, p=2))


def compute_fit_design_diagnostics(design: np.ndarray) -> Dict[str, object]:
    """Compute identifiability diagnostics for one fit design matrix.

    The conventions follow the MultiFitter rank contract in
    ``na49_mixing/fits.py``: the numerical rank comes from
    ``numpy.linalg.matrix_rank`` under its default SVD tolerance
    (``S.max() * max(shape) * eps``, the one place a numerical tolerance is
    allowed), and the condition number is the 2-norm
    ``numpy.linalg.cond(design, p=2)``, which may be ``inf`` for a singular
    design. The returned dict carries:

    - ``design_rank``: numerical column rank of the design;
    - ``design_n_components``: column count;
    - ``design_singular_values``: full singular-value list, descending;
    - ``design_condition_number``: 2-norm condition number (may be ``inf``).

    Degenerate-but-valid shapes (zero rows or zero columns) yield rank 0, an
    empty singular-value list, and condition number ``inf`` instead of
    raising, so callers can diagnose every shape the validators accept.
    """
    d = np.asarray(design, dtype=float)
    n_comp = int(d.shape[1]) if d.ndim == 2 else 0
    if d.size == 0:
        return {
            "design_rank": 0,
            "design_n_components": n_comp,
            "design_singular_values": [],
            "design_condition_number": float("inf"),
        }
    svals = np.linalg.svd(d, compute_uv=False)
    return {
        "design_rank": int(np.linalg.matrix_rank(d)),
        "design_n_components": n_comp,
        "design_singular_values": [float(s) for s in svals],
        "design_condition_number": float(np.linalg.cond(d, p=2)),
    }


def enforce_fit_design_rank(
    design: np.ndarray,
    *,
    n_components: int,
    allow_rank_deficient: bool,
    context: str,
    design_scope: str,
) -> Dict[str, object]:
    """Compute design diagnostics and refuse a rank-deficient design by default.

    A design whose numerical column rank falls below ``n_components`` leaves
    at least one combination of weights undetermined, so the individual
    weights are not identifiable. The refusal follows the MultiFitter wording
    in ``na49_mixing/fits.py``: it names the numerical rank, the component
    count, and the 2-norm condition number of the offending matrix. Passing
    ``allow_rank_deficient=True`` opts out for deliberate exploratory fits;
    callers must then stamp their result payloads with
    ``identifiability=RANK_DEFICIENT_STAMP``.

    ``design_scope`` names the matrix being diagnosed (for example
    "augmented_weighted_design" or "weighted_linear_component_matrix") and
    appears in the refusal message. Returns the diagnostics dict from
    :func:`compute_fit_design_diagnostics`.
    """
    diag = compute_fit_design_diagnostics(design)
    rank = int(diag["design_rank"])  # type: ignore[arg-type]
    n_comp = int(n_components)
    if rank < n_comp and not bool(allow_rank_deficient):
        cond = float(diag["design_condition_number"])  # type: ignore[arg-type]
        raise ValueError(
            f"{context} rejects the fit because the {design_scope} does not "
            f"have full column rank: numerical rank {rank} < n_components = "
            f"{n_comp} (2-norm condition number {cond!r}); this dependence "
            "leaves at least one combination of weights undetermined, so the "
            "individual weights are not identifiable; pass "
            "allow_rank_deficient=True only for a deliberate exploratory "
            f"fit, which stamps its result identifiability="
            f"{RANK_DEFICIENT_STAMP!r}"
        )
    return diag


def fit_nnls_weighted(
    A: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    *,
    A_prior: np.ndarray | None = None,
    b_prior: np.ndarray | None = None,
    allow_rank_deficient: bool = True,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Sigma-weighted non-negative least squares with optional prior rows.

    Inputs are validated before SciPy is called and the solver output is
    validated after it returns; malformed or non-finite inputs and non-finite
    or wrong-shape solutions raise ValueError or RuntimeError with the
    offending quantity named. The raw uncertainty vector is validated before
    the positive floor is applied, so a non-finite sigma cannot be masked.
    ``A_prior`` and ``b_prior`` must be supplied as a pair: exactly one of
    the two present raises ValueError before any solver call; both present
    and consistently empty keeps the documented no-op prior behavior.

    SciPy ``nnls`` is called with ``maxiter = max(30 n, 1000)``. The
    library default of ``3 n`` can stop on poorly conditioned designs
    before a solution is reached. A remaining iteration failure raises
    ``RuntimeError`` naming the cap.

    Identifiability diagnostics are computed from the design the solver
    consumes: the sigma-weighted data rows stacked on top of the active
    linear-prior rows (``design_scope`` "augmented_weighted_design"), or the
    weighted data rows alone when no prior rows are given
    (``design_scope`` "weighted_data_rows"). The info payload records
    ``design_scope``, ``design_rank``, ``design_n_components``,
    ``design_singular_values`` (full descending list),
    ``design_condition_number`` (2-norm, may be ``inf``),
    ``n_active_weights`` (weights > ACTIVE_WEIGHT_THRESHOLD, the same
    convention as the benchmark's n_eff counter), and ``identifiability``
    (FULL_RANK_STAMP or RANK_DEFICIENT_STAMP).

    A rank-deficient design raises ValueError before the solve unless
    ``allow_rank_deficient`` is set. The default ``allow_rank_deficient=True``
    preserves this helper's documented behavior for its existing callers,
    which fit small deliberately-shaped designs that may contain an
    identically zero component column; every such result carries the
    ``identifiability=RANK_DEFICIENT_STAMP`` stamp in its payload so no
    consumer can present the fit as identifiable without reading that field.
    Pass ``allow_rank_deficient=False`` for strict refusal semantics.
    """
    a = np.asarray(A, dtype=float)
    if a.ndim != 2:
        raise ValueError(
            "fit design matrix 'A' must be 2-D with shape (n_points, n_components); "
            f"got ndim={a.ndim}, shape={a.shape}"
        )
    n_comp = int(a.shape[1])
    yv = np.asarray(y, dtype=float)
    sv_raw = np.asarray(sigma, dtype=float)
    a = validate_fit_design(a, n_comp, name="design matrix 'A'")
    yv = validate_fit_target(yv, a.shape[0], name="target vector 'y'")
    sv_raw = validate_fit_sigma(sv_raw, a.shape[0], name="sigma vector 'sigma'")
    sv = np.maximum(sv_raw, 1e-12)
    a_w = a / sv[:, None]
    b_w = yv / sv
    if (A_prior is None) != (b_prior is None):
        raise ValueError(
            "fit A_prior/b_prior: linear priors must be supplied as a pair; "
            f"exactly one side is present (A_prior: {A_prior is not None}, "
            f"b_prior: {b_prior is not None})"
        )
    n_prior_rows = 0
    if A_prior is not None and b_prior is not None:
        ap, bp = validate_fit_prior_rows(A_prior, b_prior, n_comp, name="prior rows")
        n_prior_rows = int(ap.shape[0])
        if int(ap.size) > 0:
            A_fit = np.vstack([a_w, ap])
            b_fit = np.concatenate([b_w, bp])
            design_scope = "augmented_weighted_design"
        else:
            # Both arrays present and consistently empty: the documented
            # no-op prior behavior (validated, but no rows entered).
            A_fit = a_w
            b_fit = b_w
            design_scope = "weighted_data_rows"
    else:
        A_fit = a_w
        b_fit = b_w
        design_scope = "weighted_data_rows"
    design_diag = enforce_fit_design_rank(
        A_fit,
        n_components=n_comp,
        allow_rank_deficient=allow_rank_deficient,
        context="fit_nnls_weighted",
        design_scope=design_scope,
    )
    n_unknowns = int(A_fit.shape[1])
    # SciPy's default maxiter is 3*n, which the synthetic stage-0
    # regeneration matrix (and other poorly conditioned designs) can
    # exhaust. 30*n is still a hard cap; a remaining failure is reported
    # as a convergence error instead of an unlabeled RuntimeError.
    nnls_maxiter = max(30 * n_unknowns, 1000)
    try:
        w, rnorm = nnls(A_fit, b_fit, maxiter=nnls_maxiter)
    except RuntimeError as exc:
        raise RuntimeError(
            f"fit_nnls_weighted: scipy.optimize.nnls did not converge "
            f"(maxiter={nnls_maxiter}, n_components={n_unknowns}): {exc}"
        ) from exc
    w = validate_fit_result(
        np.asarray(w, dtype=float),
        n_components=n_comp,
        success=True,
        status=0,
        message="ok",
        objective=float(rnorm),
        objective_name="residual norm rnorm",
        context="fit_nnls_weighted",
    )
    identifiability = (
        RANK_DEFICIENT_STAMP
        if int(design_diag["design_rank"]) < n_comp  # type: ignore[arg-type]
        else FULL_RANK_STAMP
    )
    return w, {
        "method": "scipy.optimize.nnls",
        "nnls_maxiter": int(nnls_maxiter),
        "rnorm": float(rnorm),
        "n_components": int(w.shape[0]),
        "n_data": int(yv.shape[0]),
        "n_prior_rows": n_prior_rows,
        "design_scope": design_scope,
        **design_diag,
        "n_active_weights": int(np.count_nonzero(w > ACTIVE_WEIGHT_THRESHOLD)),
        "identifiability": identifiability,
    }


def load_species_paper_window(
    table_path: Path | str,
    *,
    xf_min: float = PAPER_XF_MIN,
    xf_max: float = PAPER_XF_MAX,
    pt_min: float = PAPER_PT_MIN,
    pt_max: float = PAPER_PT_MAX,
) -> Dict[str, np.ndarray]:
    """Load an SPS table and return paper-window arrays plus grid edges."""
    xf, pt, y, sigma = read_na49_points(table_path)
    xf, pt, y, sigma = filter_paper_window(
        xf,
        pt,
        y,
        sigma,
        xf_min=xf_min,
        xf_max=xf_max,
        pt_min=pt_min,
        pt_max=pt_max,
    )
    xedges = edges_from_centers(xf)
    pedges = edges_from_centers(pt)
    ix, ip = grid_indices(xf, pt, xedges, pedges)
    return {
        "xf": xf,
        "pt": pt,
        "y": y,
        "sigma": sigma,
        "xedges": xedges,
        "pedges": pedges,
        "ix_data": ix,
        "ip_data": ip,
    }


# --- multispecies_constrained_details.npz schema ------------------------------
#
# benchmark_multispecies_constrained.py writes one details NPZ per run next to
# multispecies_constrained_summary.json. String metadata is stored as
# Unicode-dtype arrays (never object dtype), so every consumer can load the
# file with allow_pickle=False; the loader below validates the payload before
# handing it on. Older result directories written with dtype=object metadata
# are intentionally unloadable and must be regenerated.

DETAILS_NPZ_SCHEMA_VERSION = 1

DETAILS_NPZ_ARRAY_STRING_KEYS: Tuple[str, ...] = (
    "component_keys",
    "component_labels",
    "species_keys",
)

DETAILS_NPZ_SCALAR_STRING_KEYS: Tuple[str, ...] = (
    "prior_report_json",
    "fit_weight_mode",
    "fit_objective",
    "meta_weak_feed_policy",
    "meta_histogram_convention",
    "fit_solver_json",
)

DETAILS_NPZ_REQUIRED_STRING_KEYS: Tuple[str, ...] = (
    DETAILS_NPZ_ARRAY_STRING_KEYS + DETAILS_NPZ_SCALAR_STRING_KEYS
)

DETAILS_NPZ_FLOAT_KEYS: Tuple[str, ...] = (
    "component_weights",
    "resonance_global_full_pred",
    "resonance_global_full_y",
    "resonance_global_full_sigma",
)

DETAILS_NPZ_SPECIES_FLOAT_SUFFIXES: Tuple[str, ...] = (
    "xf",
    "pt",
    "y",
    "sigma",
    "pred_res",
    "pred_res_sigma",
)

DETAILS_NPZ_EXTERNAL_PRED_SUFFIXES: Tuple[str, ...] = (
    "pred_pythia",
    "pred_epos",
    "pred_urqmd",
)


def _details_schema_error(details_path: Path | str, key: str, detail: str) -> ValueError:
    return ValueError(
        f"{details_path}: invalid multispecies_constrained_details.npz "
        f"schema for key '{key}': {detail}"
    )


def _summary_subdict(payload: object, key: str) -> Optional[Dict[str, object]]:
    val = payload.get(key) if isinstance(payload, dict) else None  # type: ignore[union-attr]
    return dict(val) if isinstance(val, dict) else None


def validate_multispecies_details_npz(
    zarr: Any,
    *,
    details_path: Path | str,
    summary_payload: Optional[Dict[str, object]] = None,
) -> None:
    """Validate one ``multispecies_constrained_details.npz`` payload.

    ``zarr`` is an open :class:`numpy.lib.npyio.NpzFile` (or any mapping of
    dataset name to array). Checks:

    - presence of every required dataset plus the ``schema_version`` marker;
    - Unicode string dtypes for metadata arrays (1-D for component/species
      keys, 0-d for scalar policy strings) and numeric float/int dtypes for
      value arrays;
    - internal length consistency between component keys, labels and weights
      and among the three global stacked arrays;
    - when ``summary_payload`` (the parsed companion
      ``multispecies_constrained_summary.json``) is given: ordered component
      lists and ordered species keys against the summary (the producer writes
      both artifacts from one mapping order), per-species array lengths against
      the recorded point counts, external prediction lengths where present,
      and the global stack length against the summed point counts.

    Failures raise ``ValueError`` naming the file path and the offending key.
    """
    files = set(getattr(zarr, "files", None) or zarr.keys())  # type: ignore[attr-defined]

    required = ("schema_version",) + DETAILS_NPZ_REQUIRED_STRING_KEYS + DETAILS_NPZ_FLOAT_KEYS
    for key in required:
        if key not in files:
            detail = "missing required dataset"
            if key == "schema_version":
                # A details NPZ without the marker is a pre-schema artifact
                # written by an older producer run; it must be regenerated,
                # not repaired in place.
                detail += "; artifact was written by an older producer, regenerate with the current producer"
            raise _details_schema_error(details_path, key, detail)

    sv = np.asarray(zarr["schema_version"])
    if sv.dtype.kind != "i" or sv.ndim != 0:
        raise _details_schema_error(
            details_path,
            "schema_version",
            f"expected a 0-d signed integer marker, got dtype={sv.dtype}, shape={sv.shape}",
        )
    if int(sv) != DETAILS_NPZ_SCHEMA_VERSION:
        raise _details_schema_error(
            details_path,
            "schema_version",
            f"expected {DETAILS_NPZ_SCHEMA_VERSION}, found {int(sv)}; regenerate with the current producer",
        )

    str_meta_arrays: Dict[str, np.ndarray] = {}
    for key in DETAILS_NPZ_ARRAY_STRING_KEYS:
        arr = np.asarray(zarr[key])
        if arr.dtype.kind != "U":
            raise _details_schema_error(
                details_path, key, f"expected Unicode string dtype, got dtype={arr.dtype}"
            )
        if arr.ndim != 1:
            raise _details_schema_error(details_path, key, f"expected a 1-D string array, got shape={arr.shape}")
        str_meta_arrays[key] = arr

    for key in DETAILS_NPZ_SCALAR_STRING_KEYS:
        arr = np.asarray(zarr[key])
        if arr.dtype.kind != "U":
            raise _details_schema_error(
                details_path, key, f"expected Unicode string dtype, got dtype={arr.dtype}"
            )
        if arr.ndim != 0:
            raise _details_schema_error(details_path, key, f"expected a 0-d string dataset, got shape={arr.shape}")

    float_arrays: Dict[str, np.ndarray] = {}
    for key in DETAILS_NPZ_FLOAT_KEYS:
        arr = np.asarray(zarr[key])
        if arr.dtype.kind != "f":
            raise _details_schema_error(details_path, key, f"expected floating-point dtype, got dtype={arr.dtype}")
        if arr.ndim != 1:
            raise _details_schema_error(details_path, key, f"expected a 1-D float array, got shape={arr.shape}")
        float_arrays[key] = arr

    n_keys = int(str_meta_arrays["component_keys"].shape[0])
    n_labels = int(str_meta_arrays["component_labels"].shape[0])
    n_weights = int(float_arrays["component_weights"].shape[0])
    if not (n_keys == n_labels == n_weights):
        raise _details_schema_error(
            details_path,
            "component_weights",
            f"component key/label/weight length mismatch ({n_keys}/{n_labels}/{n_weights})",
        )

    n_pred = int(float_arrays["resonance_global_full_pred"].shape[0])
    n_gy = int(float_arrays["resonance_global_full_y"].shape[0])
    n_gsig = int(float_arrays["resonance_global_full_sigma"].shape[0])
    if not (n_pred == n_gy == n_gsig):
        raise _details_schema_error(
            details_path,
            "resonance_global_full_pred",
            f"global full-stack length mismatch pred/y/sigma ({n_pred}/{n_gy}/{n_gsig})",
        )

    if "n_effective" in files:
        ne = np.asarray(zarr["n_effective"])
        if ne.dtype.kind != "i":
            raise _details_schema_error(
                details_path, "n_effective", f"expected integer dtype, got dtype={ne.dtype}"
            )

    summary_payload_dict: Optional[Dict[str, object]] = (
        dict(summary_payload) if isinstance(summary_payload, dict) else None
    )
    if summary_payload_dict is None:
        return

    res_meta = _summary_subdict(summary_payload_dict, "resonance")
    if res_meta is not None:
        json_component_keys = res_meta.get("component_keys")
        if isinstance(json_component_keys, list):
            want_keys = [str(x) for x in json_component_keys]
            got_keys = [str(x) for x in str_meta_arrays["component_keys"].tolist()]
            if want_keys != got_keys:
                raise _details_schema_error(
                    details_path,
                    "component_keys",
                    f"component order does not match summary JSON: expected {want_keys}, found {got_keys}",
                )
        json_component_labels = res_meta.get("component_labels")
        if isinstance(json_component_labels, list):
            want_labels = [str(x) for x in json_component_labels]
            got_labels = [str(x) for x in str_meta_arrays["component_labels"].tolist()]
            if want_labels != got_labels:
                raise _details_schema_error(
                    details_path,
                    "component_labels",
                    f"component labels do not match summary JSON: expected {want_labels}, found {got_labels}",
                )
        n_comp_json = res_meta.get("n_components")
        if isinstance(n_comp_json, int) and int(n_comp_json) != n_keys:
            raise _details_schema_error(
                details_path,
                "component_keys",
                f"summary JSON records n_components={int(n_comp_json)}, NPZ stores {n_keys} keys",
            )

    species_map = _summary_subdict(summary_payload_dict, "species")
    total_points = 0
    saw_numeric_points = False
    if species_map is not None:
        got_species = [str(x) for x in str_meta_arrays["species_keys"].tolist()]
        want_species = [str(k) for k in species_map.keys()]
        if got_species != want_species:
            raise _details_schema_error(
                details_path,
                "species_keys",
                f"species order does not match summary JSON: expected {want_species}, found {got_species}",
            )
        for skey in sorted(species_map.keys()):
            smeta = species_map.get(skey)
            if not isinstance(smeta, dict):
                continue
            tag = str(smeta.get("tag", "")).strip()
            n_points_raw = smeta.get("n_points")
            if not tag or not isinstance(n_points_raw, int) or isinstance(n_points_raw, bool):
                continue
            n_points = int(n_points_raw)
            total_points += n_points
            saw_numeric_points = True
            expected_shape = (n_points,)
            for suffix in DETAILS_NPZ_SPECIES_FLOAT_SUFFIXES:
                tkey = f"{tag}_{suffix}"
                if tkey not in files:
                    raise _details_schema_error(details_path, tkey, f"missing required dataset for species '{skey}'")
                tarr = np.asarray(zarr[tkey])
                if tarr.dtype.kind != "f":
                    raise _details_schema_error(
                        details_path, tkey, f"expected floating-point dtype, got dtype={tarr.dtype}"
                    )
                if tarr.shape != expected_shape:
                    raise _details_schema_error(
                        details_path,
                        tkey,
                        f"summary JSON records n_points={n_points} for '{skey}', NPZ stores shape={tarr.shape}",
                    )
            idx_key = f"{tag}_full_idx"
            if idx_key not in files:
                raise _details_schema_error(details_path, idx_key, f"missing required dataset for species '{skey}'")
            iarr = np.asarray(zarr[idx_key])
            if iarr.dtype.kind != "i":
                raise _details_schema_error(
                    details_path, idx_key, f"expected integer dtype, got dtype={iarr.dtype}"
                )
            if iarr.shape != expected_shape:
                raise _details_schema_error(
                    details_path,
                    idx_key,
                    f"summary JSON records n_points={n_points} for '{skey}', NPZ stores shape={iarr.shape}",
                )
            for suffix in DETAILS_NPZ_EXTERNAL_PRED_SUFFIXES:
                for ekey in (f"{tag}_{suffix}", f"{tag}_{suffix}_sigma"):
                    if ekey not in files:
                        continue
                    earr = np.asarray(zarr[ekey])
                    if earr.dtype.kind != "f":
                        raise _details_schema_error(
                            details_path, ekey, f"expected floating-point dtype, got dtype={earr.dtype}"
                        )
                    if earr.shape != expected_shape:
                        raise _details_schema_error(
                            details_path,
                            ekey,
                            f"summary JSON records n_points={n_points} for '{skey}', NPZ stores shape={earr.shape}",
                        )

    if saw_numeric_points and n_pred != total_points:
        raise _details_schema_error(
            details_path,
            "resonance_global_full_pred",
            f"global full-stack holds {n_pred} points but the summary JSON species entries sum to {total_points}",
        )


_DETAILS_PICKLE_ERROR_MARKERS: Tuple[str, ...] = ("allow_pickle", "object array", "pickled")


def _remediate_pickle_load_error(details_path: Path, exc: ValueError) -> None:
    """Re-raise a numpy ValueError as pickle remediation when it matches.

    Any other ValueError is left untouched so schema-validation errors keep
    their own precise wording. The rewritten message names the details file
    and its run directory and tells the operator to regenerate with the
    current producer; pickle loading is never attempted.
    """
    text = str(exc).lower()
    if not any(marker in text for marker in _DETAILS_PICKLE_ERROR_MARKERS):
        return
    raise ValueError(
        f"{details_path}: multispecies_constrained_details.npz contains pickled object-dtype "
        f"metadata and cannot be loaded with allow_pickle=False "
        f"(run directory '{details_path.parent}'); regenerate with the current producer, "
        f"which stores string metadata as Unicode arrays instead of object arrays"
    ) from exc


@contextmanager
def load_multispecies_details_npz(
    details_path: Path | str,
    *,
    summary_payload: Optional[Dict[str, object]] = None,
) -> Iterator[Any]:
    """Open and validate a details NPZ file pickle-free as a context manager.

    Loads ``multispecies_constrained_details.npz`` with ``allow_pickle=False``,
    runs :func:`validate_multispecies_details_npz`, yields the open
    :class:`numpy.lib.npyio.NpzFile`, and closes it afterwards whether
    validation succeeded or raised. A payload whose numpy error mentions
    pickled object arrays (either at open or at lazy member access, which
    happens during validation) is re-raised as ``ValueError`` naming the run
    directory with remediation guidance; pickle loading is never attempted.
    """
    path = Path(details_path)
    try:
        zarr = np.load(str(path), allow_pickle=False)
    except ValueError as exc:
        _remediate_pickle_load_error(path, exc)
        raise
    try:
        try:
            validate_multispecies_details_npz(zarr, details_path=path, summary_payload=summary_payload)
            yield zarr
        except ValueError as exc:
            _remediate_pickle_load_error(path, exc)
            raise
    finally:
        zarr.close()
