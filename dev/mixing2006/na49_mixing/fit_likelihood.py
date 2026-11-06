# Not used by the HEART p+p fits (marked 2026-11-04): no production generation, fit, probe tool or paper script
# runs this module (an execution trace of the R137 generation and an import scan of the engine, the probe tools
# and the paper scripts). Code reviews of the p+p model can skip it. It stays until the cleanup and refactor after
# the p+A run.
"""Binned likelihood with combined data and MC covariance and per-table
normalization nuisances (Round 1, task card
TASK-20260911-r1-covariance-nuisances; plan section 7 task 1 and section
5.4).

The module is a parallel statistics layer next to the M8 binned objective
in :mod:`na49_mixing.staged_evaluation`.  The M8 objective combines the
per-bin data sigma with the MC sigma of the replica-mean prediction into
a diagonal combined error and forms the per-bin chi-square.  This module
generalizes that objective in two directions and reproduces it exactly in
its limit:

- a full data covariance slot (``V = V_data + V_MC``), so an
  off-diagonal 2-D covariance can replace the per-bin statistical
  diagonal once a published covariance becomes available; and
- a correlated per-table normalization nuisance, because the SPS tables
  publish per-bin statistical errors but no published 2-D covariance, so
  the table's normalization uncertainty enters as a nuisance parameter
  with a documented width rather than as a per-bin error.

Likelihood (plan section 5.4)::

    -2 ln L = (d - m - shift)^T V^{-1} (d - m - shift)
              + sum_k (eta_k / sigma_eta_k)^2

Representation conventions (documented choices where the task card is
silent):

- Covariance flattening.  Every covariance tuple in this module
  (``BinnedObservable.cov``, ``mc_cov``, ``V``, and the
  :func:`combined_covariance` return) is the upper triangle of the
  symmetric n-by-n matrix in row-major order including the diagonal:
  ``(0,0), (0,1), ..., (0,n-1), (1,1), (1,2), ..., (n-1,n-1)``, with
  ``n (n + 1) / 2`` entries.  A diagonal matrix is that tuple with zero
  off-diagonal entries; :func:`diagonal_covariance` builds it from
  per-bin sigmas.

- Data covariance of an observable.  ``BinnedObservable.cov = None``
  means ``V_data = diag(stat_errors**2)``; a provided tuple is the full
  ``V_data`` in the flattened convention above.

- Combined covariance.  :func:`combined_covariance` returns the
  element-wise sum ``V_data + V_MC`` of two equally shaped flattened
  tuples; ``data_cov = None`` contributes no data term (it returns
  ``mc_cov``), so the observable's statistical diagonal enters through
  ``diagonal_covariance(d.stat_errors)`` at the call site.  The recipe is::

      V_data = d.cov if d.cov is not None else diagonal_covariance(d.stat_errors)
      V = combined_covariance(V_data, mc_cov)
      minus2lnL, components = evaluate_likelihood(d, m, V, nuisances)

  :func:`evaluate_likelihood` reconstructs ``V_data`` from the
  observable the same way and recovers ``V_MC = V - V_data``.

- Nominal value of the nuisance shift.  The per-bin nominal value is the
  per-bin data value ``d.values[i]``.  A normalization nuisance
  parameterizes a fractional shift of the table's normalization, so bin
  i in the nuisance's scope receives
  ``shift_i = values_i * sum(eta_k)`` and the residual is
  ``d_i - m_i - shift_i``.  ``eta = 0`` is the nominal likelihood; the
  sign convention is that positive ``eta`` moves the data values away
  from the model for positive residuals ``d - m``.

- Nuisance scope.  A nuisance applies to an observable when the
  observable's ``normalization_nuisance`` names it, or when the
  observable carries no explicit reference and the nuisance's
  ``table_scope`` is ``"all"`` or equals the observable's ``species``.
  The bundled config :file:`config/v2_singlet/likelihood_nuisances.json`
  declares one nuisance per SPS table with ``table_scope`` equal to the
  species key of the matching ``fit.binned.tables`` entry.

- Nuisance penalty.  ``chi2_nuisance`` sums ``(eta / sigma_eta)**2``
  over every nuisance passed to the call, independent of scope; the
  scope only controls the shift.  A driver that evaluates several
  observables against one shared nuisance must account the penalty once.

- Component split.  With ``w = V^{-1} (d - m - shift)`` the components
  are ``chi2_data = w^T V_data w`` and ``chi2_mc = w^T V_MC w`` with
  ``V_MC = V - V_data``, so ``chi2_data + chi2_mc`` equals the combined
  quadratic form ``(d - m - shift)^T V^{-1} (d - m - shift)`` exactly,
  through the identity ``V^{-1} = V^{-1} (V_data + V_MC) V^{-1}``.
  ``chi2_total`` is that combined quadratic form plus the nuisance
  penalty, i.e. the returned ``-2 ln L``.  In the limit
  ``sigma_MC -> 0`` the data component reduces to the pure data
  quadratic form ``(d - m - shift)^T V_data^{-1} (d - m - shift)``, and
  symmetrically for the MC component; an exactly zero component matrix
  contributes 0.

- M8 limit.  With ``nuisances = None`` and ``data_cov = None`` the
  returned value equals the M8 combined chi-square of
  :func:`na49_mixing.staged_evaluation._fit_regge_binned` (per bin
  ``sigma_combined**2 = sigma_data**2 + sigma_MC**2``, residual
  ``(m - d) / sigma_combined``, chi-square the sum of squares, summed
  over the tables) to floating-point rounding, far inside the card's
  1e-9 relative requirement.

The nuisance parameter ``eta`` is an input frozen at construction.  This
card fits nothing (the eta fit is a later-round card), and no existing
code path reads the bundled config in Round 1.

Sector-wise deviance decomposition (Round 1, task card
TASK-20260911-r1-sector-wise-fit-reports; plan section 5.4, plan
section 7 task 1 diagnostic half).  The decomposition is a purely
statistical partition of the binned deviance of one observable; it is a
diagnostic report (the names ``sector_decomposition`` /
``write_sector_report`` name regions of a pre-registered kinematic grid,
not strings, parton splitting, or constituent breakup), it fits no
parameter, and it changes no likelihood, covariance, or nuisance.

- Pre-registered grid.  ``SECTOR_REGIONS`` is the three-way split of the
  paper window's xF axis (``[0, 0.4]``: central, transition, forward)
  and of its pT axis (``[0.05, 1.5]``: low, mid, high).  The grid is a
  diagnostic convention, not a physics claim.

- Sector key encoding (documented choice).  The card's example key
  ``"pi-|f_xsect|xF_forward"`` names one region while its acceptance
  criterion counts ``n_sectors == 54 = 9 regions x 6 species``, so a
  sector is the cross-product pair (xF region, pT region) and the key
  encodes both: ``"{species}|{observable}|{xF_region}|{pT_region}"``,
  e.g. ``"pi-|f_xsect|xF_forward|pT_high"``.  The species and
  observable names are the observable's own ``species`` / ``observable``
  fields.  Sectors are enumerated xF-major, pT-minor, in the numeric
  region order ``("xF_central", "xF_transition", "xF_forward")`` and
  ``("pT_low", "pT_mid", "pT_high")``.

- Region membership (documented choice).  A bin belongs to the sector
  of the region pair whose xF interval contains the bin's xF midpoint
  and whose pT interval contains the bin's pT midpoint.  Midpoints are
  read from the observable's ``bin_edges``, which
  :func:`sector_decomposition` requires as the nested pair
  ``(xF_edges, pT_edges)`` in one of two forms: the grid form (the
  concatenated ``(n_x + 1)`` xF and ``(n_p + 1)`` pT edges of a dense
  ``n_x * n_p`` grid, bin ``i`` at ``(ix, ip) = (i mod n_x, i // n_x)``
  — the pT-major, xF-minor layout of ``_bin_tracks``) or the per-bin
  form (``2n`` edges per axis; each bin carries its own cell edges —
  the sparse-tabulation convention of the NA49 tables, whose 229 pi-
  points fill only 229 cells of an 18 x 18 center grid).  The grid
  form takes precedence when both fit.  Interval membership is
  half-open ``[lo, hi)`` on every region of an axis, with the last
  region of the axis closed ``[lo, hi]``; abutting midpoints therefore
  land in the higher region and every midpoint of the paper window is
  assigned exactly once (``unassigned_bins == 0`` on the bundled
  tables).

- Partition convention.  With ``w = V^{-1} (d - m - shift)`` the
  per-bin quadratic contribution is ``q_i = r_i w_i`` (so
  ``sum_i q_i = r^T V^{-1} r`` exactly, for any ``V``), and the
  per-bin component contributions are ``w_i (V_data w)_i`` and
  ``w_i (V_MC w)_i`` with the card-7 ``V_data`` / ``V_MC = V - V_data``
  split.  Each sector reports the sums of its bins' contributions, so
  ``chi2_data + chi2_mc == chi2_total - chi2_nuisance`` per sector and
  the sector sums reproduce the observable's ``evaluate_likelihood``
  total exactly, for any covariance, not only the diagonal one.

- Nuisance penalty booking.  ``chi2_nuisance`` per observable is the
  card-7 per-call penalty (every nuisance passed to the call,
  independent of scope).  The penalty is not bin-local; it is booked
  once, into the sector containing the observable's lowest-index
  assigned bin, so the sector partition stays exact.  An observable
  with no assigned bin emits no sector and leaves its penalty
  unbooked; the driver's global ``total_chi2`` remains the value of
  record.

- Pull convention.  ``residual_mean`` and ``worst_bin.pull`` use the
  card's unshifted difference: ``pull_i = (d_i - m_i) / sigma_i`` with
  the nominal model value ``m_i`` (no nuisance shift applied) and
  ``sigma_i = sqrt(V_ii)`` the combined per-bin error.  Squared pulls
  agree with the M8 objective's ``(m - d) / sigma_combined`` residual;
  the sign convention here follows the card's ``d - m`` formula.
  ``residual_mean`` is the unweighted mean of the sector's pulls.
  ``worst_bin`` is the bin with the largest ``|pull|``, ties broken to
  the lowest bin index.

- Sector payload.  Every (xF region, pT region) pair of an observable
  is emitted, including empty sectors (all-zero contributions,
  ``worst_bin: null``, ``residual_mean: 0.0``), so ``n_sectors`` is
  ``9 x (number of observables)`` by construction.  Each sector's
  ``fraction_of_total`` is its ``chi2_total`` divided by the
  observable's own total (``chi2_quad + penalty``, the
  ``evaluate_likelihood`` return of the same call), so the fractions of
  one observable sum to 1.  :func:`write_sector_report` writes the
  decomposition with the top-level block (``total_chi2``, ``dof``,
  ``n_sectors``, ``unassigned_bins``, and the per-species, per-
  observable, and per-region totals keyed by the six named regions of
  ``SECTOR_REGIONS`` as marginals over the cross-product sectors).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "BinnedObservable",
    "NormalizationNuisance",
    "SECTOR_REGIONS",
    "combined_covariance",
    "diagonal_covariance",
    "evaluate_likelihood",
    "flatten_covariance",
    "sector_decomposition",
    "unflatten_covariance",
    "write_sector_report",
]


@dataclass(frozen=True)
class BinnedObservable:
    """One windowed binned observable (one NA49 table's fit bins).

    ``values`` and ``stat_errors`` are the per-bin data values and
    statistical errors in the table's fit units; for the SPS tables that
    is the invariant cross section per inelastic event in 1/GeV^2.
    ``bin_edges`` holds the concatenated ``(xF edges, pT edges)`` of the
    bin grid as a provenance field; the likelihood math uses bin counts,
    not edges.  ``cov`` optionally carries a full off-diagonal data
    covariance in the flattened upper-triangle convention; ``None``
    means the diagonal of ``stat_errors**2``.
    ``normalization_nuisance`` optionally names the
    :class:`NormalizationNuisance` that shifts this table.
    """

    species: str  # "pi-", "pi+", "K+", "K-", "p", "pbar"
    observable: str  # "f_xsect" (invariant cross section per sigma_inel)
    bin_edges: tuple[float, ...] | tuple[tuple[float, ...], tuple[float, ...]]
    # (xF edges, pT edges): the flat concatenated provenance form of card 7,
    # or the nested pair (xF_edges, pT_edges) that sector_decomposition
    # reads as the bin grid; the likelihood math uses bin counts, not edges.
    values: tuple[float, ...]  # per-bin data values
    stat_errors: tuple[float, ...]  # per-bin statistical errors
    cov: tuple[float, ...] | None = None  # flattened upper triangle
    normalization_nuisance: str | None = None


@dataclass(frozen=True)
class NormalizationNuisance:
    """One correlated per-table normalization nuisance.

    ``eta`` is the nuisance parameter (0 = nominal); ``sigma_eta`` is its
    documented width, the normalization uncertainty of the table(s) named
    by ``table_scope``.  For the six-species SPS binned set the width is
    the 1 percent relative uncertainty of the sigma_inel convention: the
    fit normalizes by the literature inelastic p+p cross section
    31.78 mb, and the NA49-extracted value 31.46 mb agrees with it to
    within one percent (Table 1 of
    ``notes/more_docs/raw_text/pp-pi_layout.txt``; recorded in
    ``_knowledge/operations/mixing2006-conventions.md``).  ``eta`` is
    frozen at construction; this module fits nothing.
    """

    name: str  # e.g. "sps_piminus_norm"
    eta: float  # nuisance parameter (0 = nominal)
    sigma_eta: float  # width (the normalization uncertainty)
    table_scope: str  # which table(s) the nuisance correlates over


def _dimension_for_length(length: int) -> int:
    """Dimension ``n`` of an upper-triangle flattened symmetric matrix."""
    if length <= 0:
        raise ValueError(f"a covariance needs at least 1 entry, got {length}")
    n = int(round((math.sqrt(8.0 * length + 1.0) - 1.0) / 2.0))
    if n * (n + 1) // 2 != length:
        raise ValueError(
            "a flattened upper-triangle covariance needs n(n+1)/2 entries, "
            f"got {length}"
        )
    return n


def flatten_covariance(matrix) -> tuple[float, ...]:
    """Flatten a symmetric (n, n) covariance to its upper triangle.

    The order is row-major including the diagonal, matching the
    module-level flattening convention.
    """
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"covariance must be square, got shape {arr.shape}")
    return tuple(float(v) for v in arr[np.triu_indices(arr.shape[0])])


def unflatten_covariance(flat) -> np.ndarray:
    """Rebuild the symmetric (n, n) matrix from a flattened upper
    triangle (row-major, including the diagonal)."""
    seq = tuple(float(v) for v in flat)
    n = _dimension_for_length(len(seq))
    mat = np.zeros((n, n), dtype=float)
    iu = np.triu_indices(n)
    mat[iu] = seq
    return mat + mat.T - np.diag(np.diag(mat))


def diagonal_covariance(sigmas) -> tuple[float, ...]:
    """Flattened covariance with per-bin variances ``sigma**2`` on the
    diagonal and zero off-diagonal entries."""
    values = tuple(float(s) for s in sigmas)
    if not values:
        raise ValueError("a covariance needs at least 1 entry")
    flat: list[float] = []
    n = len(values)
    for i, sigma in enumerate(values):
        row = [0.0] * (n - i)
        row[0] = sigma * sigma
        flat.extend(row)
    return tuple(flat)


def combined_covariance(
    data_cov: tuple[float, ...] | None,
    mc_cov: tuple[float, ...],
) -> tuple[float, ...]:
    """Return ``V_data + V_MC`` as a flattened upper-triangle tuple.

    ``mc_cov`` is the MC covariance in the flattened convention.
    ``data_cov = None`` contributes no data term; the observable's
    per-bin statistical diagonal enters through
    :func:`diagonal_covariance` at the call site (see the module
    docstring for the call recipe).  Both provided tuples must have the
    same valid flattened length.
    """
    mc = tuple(float(v) for v in mc_cov)
    _dimension_for_length(len(mc))
    if data_cov is None:
        return mc
    dc = tuple(float(v) for v in data_cov)
    if len(dc) != len(mc):
        raise ValueError(
            "data and MC covariances must have equal shape, got "
            f"{len(dc)} and {len(mc)} entries"
        )
    return tuple(a + b for a, b in zip(dc, mc))


def _nuisance_applies(
    nuisance: NormalizationNuisance,
    observable: BinnedObservable,
) -> bool:
    """Scope rule; see the module docstring."""
    if observable.normalization_nuisance is not None:
        return observable.normalization_nuisance == nuisance.name
    scope = nuisance.table_scope.strip()
    return scope == "all" or scope == observable.species


def _require_positive_definite(matrix: np.ndarray, label: str) -> None:
    """Raise ``ValueError`` unless the matrix is positive definite."""
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{label} is not positive definite") from exc


def evaluate_likelihood(
    d: BinnedObservable,
    m: tuple[float, ...],
    V: tuple[float, ...],
    nuisances: list[NormalizationNuisance] | None = None,
) -> tuple[float, dict]:
    """Evaluate the binned likelihood; return ``(-2 ln L, components)``.

    ``m`` is the per-bin model prediction, ``V`` the combined covariance
    ``V_data + V_MC`` in the flattened upper-triangle convention.  The
    component dict has exactly the keys ``chi2_data``, ``chi2_mc``,
    ``chi2_nuisance``, and ``chi2_total``; see the module docstring for
    the component split and the nuisance conventions.  ``eta`` is not
    fitted here.
    """
    values = tuple(float(v) for v in d.values)
    n = len(values)
    if n == 0:
        raise ValueError("the observable needs at least one bin")
    stat_errors = tuple(float(s) for s in d.stat_errors)
    if len(stat_errors) != n:
        raise ValueError(
            f"stat_errors has {len(stat_errors)} entries for {n} bins"
        )
    model = tuple(float(v) for v in m)
    if len(model) != n:
        raise ValueError(f"the model vector has {len(model)} entries for {n} bins")
    v_flat = tuple(float(v) for v in V)
    _dimension_for_length(len(v_flat))

    nuisance_list = list(nuisances) if nuisances else []
    for item in nuisance_list:
        sigma_eta = float(item.sigma_eta)
        if not (math.isfinite(sigma_eta) and sigma_eta > 0.0):
            raise ValueError(
                f"nuisance {item.name!r} needs a positive sigma_eta, "
                f"got {item.sigma_eta!r}"
            )

    values_arr = np.asarray(values, dtype=float)
    shift = np.zeros(n, dtype=float)
    for item in nuisance_list:
        if _nuisance_applies(item, d):
            shift += float(item.eta) * values_arr
    residual = values_arr - np.asarray(model, dtype=float) - shift

    v_mat = unflatten_covariance(v_flat)
    try:
        np.linalg.cholesky(v_mat)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "the combined covariance V is not positive definite"
        ) from exc

    # Data covariance implied by the observable: the per-bin statistical
    # diagonal, or the full flattened covariance it carries.
    if d.cov is None:
        if not all(math.isfinite(s) and s > 0.0 for s in stat_errors):
            raise ValueError(
                "stat_errors must be positive when the observable carries "
                "no covariance"
            )
        v_data = np.diag(np.asarray(stat_errors, dtype=float) ** 2)
    else:
        data_flat = tuple(float(v) for v in d.cov)
        _dimension_for_length(len(data_flat))
        if len(data_flat) != len(v_flat):
            raise ValueError(
                "the observable's covariance and V must have equal shape, "
                f"got {len(data_flat)} and {len(v_flat)} entries"
            )
        v_data = unflatten_covariance(data_flat)
    v_mc = v_mat - v_data
    v_data_zero = not np.any(v_data)
    v_mc_zero = not np.any(v_mc)
    if not v_data_zero:
        _require_positive_definite(v_data, "the data covariance V_data")
    if not v_mc_zero:
        _require_positive_definite(
            v_mc, "the MC covariance V_MC = V - V_data"
        )

    w = np.linalg.solve(v_mat, residual)
    chi2_quad = float(residual @ w)
    chi2_nuisance = sum(
        (float(item.eta) / float(item.sigma_eta)) ** 2 for item in nuisance_list
    )
    chi2_data = 0.0 if v_data_zero else float(w @ (v_data @ w))
    chi2_mc = 0.0 if v_mc_zero else float(w @ (v_mc @ w))
    chi2_total = chi2_quad + chi2_nuisance

    components = {
        "chi2_data": chi2_data,
        "chi2_mc": chi2_mc,
        "chi2_nuisance": float(chi2_nuisance),
        "chi2_total": chi2_total,
    }
    return chi2_total, components


# ---------------------------------------------------------------------------
# Sector-wise deviance decomposition (Round 1, card 8, additive)
# ---------------------------------------------------------------------------

SECTOR_REGIONS: dict[str, tuple[float, float]] = {
    "xF_forward": (0.2, 0.4),
    "xF_central": (0.0, 0.10),
    "xF_transition": (0.10, 0.2),
    "pT_low": (0.05, 0.5),
    "pT_mid": (0.5, 1.0),
    "pT_high": (1.0, 1.5),
}

# The sector grid axes in numeric order (SECTOR_REGIONS above is the
# pre-registered dict verbatim; these tuples fix the enumeration order of
# the sector keys and the half-open/closed boundary rule per axis).
SECTOR_XF_REGION_ORDER = ("xF_central", "xF_transition", "xF_forward")
SECTOR_PT_REGION_ORDER = ("pT_low", "pT_mid", "pT_high")

_SECTOR_REPORT_SCHEMA = "na49_mixing.sector_report/v1"


def _sector_region_of(
    value: float,
    region_order: tuple[str, ...],
) -> str | None:
    """Region of ``value`` on one axis; ``None`` when no region contains it.

    Membership is half-open ``[lo, hi)`` on every region of the axis, with
    the last region closed ``[lo, hi]`` (see the module docstring).
    """
    last = len(region_order) - 1
    for position, name in enumerate(region_order):
        lo, hi = SECTOR_REGIONS[name]
        if lo <= value < hi:
            return name
        if position == last and value == hi:
            return name
    return None


def _sector_bin_midpoints(
    d: BinnedObservable,
) -> tuple[list[float], list[float]]:
    """Per-bin (xF midpoint, pT midpoint) from the observable's bin grid.

    ``bin_edges`` must be the nested pair ``(xF_edges, pT_edges)`` in one
    of two documented forms:

    - Grid form: ``(n_x + 1)`` xF edges and ``(n_p + 1)`` pT edges with
      ``n == n_x * n_p`` bins.  Bin ``i`` sits at ``(ix, ip) =
      (i mod n_x, i // n_x)``, the pT-major, xF-minor layout of
      ``_bin_tracks``; each midpoint is the average of its bin's two
      grid edges (for ``edges_from_centers`` grids the interior
      midpoints reproduce the tabulated centers exactly).
    - Per-bin form: ``2n`` edges per axis.  Bin ``i``'s midpoint is the
      average of its own ``(lo, hi)`` edge pair; the sparse-tabulation
      convention of the NA49 tables (229 points on an 18 x 18 center
      grid), where every bin carries the edges of its own cell in the
      fit binning.

    The grid form takes precedence when both forms fit.
    """
    values_n = len(d.values)
    # The declared field type of bin_edges is the flat provenance tuple of
    # card 7; the sector machinery reads the nested pair at runtime, so the
    # local alias deliberately widens the static type.
    edges_any: Any = d.bin_edges
    if not (
        isinstance(edges_any, tuple)
        and len(edges_any) == 2
        and all(isinstance(part, tuple) for part in edges_any)
    ):
        raise ValueError(
            "sector_decomposition reads d.bin_edges as the nested pair "
            "(xF_edges, pT_edges); the flat provenance form carries no "
            "grid layout"
        )
    x_edges = tuple(float(v) for v in edges_any[0])
    p_edges = tuple(float(v) for v in edges_any[1])
    n_x = len(x_edges) - 1
    n_p = len(p_edges) - 1
    if n_x >= 1 and n_p >= 1 and n_x * n_p == values_n:
        x_mids = [0.5 * (x_edges[ix] + x_edges[ix + 1]) for ix in range(n_x)]
        p_mids = [0.5 * (p_edges[ip] + p_edges[ip + 1]) for ip in range(n_p)]
        return (
            [x_mids[i % n_x] for i in range(values_n)],
            [p_mids[i // n_x] for i in range(values_n)],
        )
    if len(x_edges) == 2 * values_n and len(p_edges) == 2 * values_n:
        return (
            [
                0.5 * (x_edges[2 * i] + x_edges[2 * i + 1])
                for i in range(values_n)
            ],
            [
                0.5 * (p_edges[2 * i] + p_edges[2 * i + 1])
                for i in range(values_n)
            ],
        )
    raise ValueError(
        "d.bin_edges must be the nested pair (xF_edges, pT_edges) either "
        f"as the grid form ({values_n} bins = n_x * n_p with (n_x + 1) + "
        "(n_p + 1) edges) or as the per-bin form (2 edges per bin per "
        f"axis, i.e. {2 * values_n} entries per axis); got "
        f"{len(x_edges)} xF edges and {len(p_edges)} pT edges"
    )


def sector_decomposition(
    d: BinnedObservable,
    m: tuple[float, ...],
    V: tuple[float, ...],
    nuisances: list[NormalizationNuisance] | None = None,
) -> dict:
    """Partition one observable's binned deviance over the sector grid.

    ``d`` is the observable (``bin_edges`` as the nested pair
    ``(xF_edges, pT_edges)`` in the grid form or the per-bin form; see
    the module docstring), ``m`` the per-bin model prediction, ``V``
    the combined covariance in the flattened upper-triangle convention,
    and ``nuisances`` the card-7 normalization nuisances (penalty and
    shift conventions as in :func:`evaluate_likelihood`).  Returns
    ``{"sectors": {key: payload}, "unassigned_bins": int}`` with the key
    encoding and the payload fields documented in the module docstring;
    see also the docstring for the partition, penalty-booking, and pull
    conventions.  The decomposition is a diagnostic: it fits nothing and
    changes no likelihood.
    """
    values = tuple(float(v) for v in d.values)
    n = len(values)
    if n == 0:
        raise ValueError("the observable needs at least one bin")
    model = tuple(float(v) for v in m)
    if len(model) != n:
        raise ValueError(f"the model vector has {len(model)} entries for {n} bins")
    stat_errors = tuple(float(s) for s in d.stat_errors)
    if len(stat_errors) != n:
        raise ValueError(
            f"stat_errors has {len(stat_errors)} entries for {n} bins"
        )
    v_flat = tuple(float(v) for v in V)
    v_mat = unflatten_covariance(v_flat)
    if v_mat.shape[0] != n:
        raise ValueError(
            f"the covariance is {v_mat.shape[0]}-dimensional for {n} bins"
        )

    x_mid_per_bin, p_mid_per_bin = _sector_bin_midpoints(d)

    nuisance_list = list(nuisances) if nuisances else []
    for item in nuisance_list:
        sigma_eta = float(item.sigma_eta)
        if not (math.isfinite(sigma_eta) and sigma_eta > 0.0):
            raise ValueError(
                f"nuisance {item.name!r} needs a positive sigma_eta, "
                f"got {item.sigma_eta!r}"
            )
    values_arr = np.asarray(values, dtype=float)
    shift = np.zeros(n, dtype=float)
    for item in nuisance_list:
        if _nuisance_applies(item, d):
            shift += float(item.eta) * values_arr
    residual = values_arr - np.asarray(model, dtype=float) - shift

    try:
        np.linalg.cholesky(v_mat)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "the combined covariance V is not positive definite"
        ) from exc

    # Data / MC split of the covariance, exactly as in evaluate_likelihood.
    if d.cov is None:
        if not all(math.isfinite(s) and s > 0.0 for s in stat_errors):
            raise ValueError(
                "stat_errors must be positive when the observable carries "
                "no covariance"
            )
        v_data = np.diag(np.asarray(stat_errors, dtype=float) ** 2)
    else:
        data_flat = tuple(float(v) for v in d.cov)
        _dimension_for_length(len(data_flat))
        if len(data_flat) != len(v_flat):
            raise ValueError(
                "the observable's covariance and V must have equal shape, "
                f"got {len(data_flat)} and {len(v_flat)} entries"
            )
        v_data = unflatten_covariance(data_flat)
    v_mc = v_mat - v_data
    v_data_zero = not np.any(v_data)
    v_mc_zero = not np.any(v_mc)
    if not v_data_zero:
        _require_positive_definite(v_data, "the data covariance V_data")
    if not v_mc_zero:
        _require_positive_definite(
            v_mc, "the MC covariance V_MC = V - V_data"
        )

    w = np.linalg.solve(v_mat, residual)
    # Per-bin quadratic contributions (row-restricted; exact partition for
    # any V, see the module docstring): q_i = r_i w_i = w_i (V w)_i.
    quad_bins = residual * w
    v_data_w = np.zeros(n, dtype=float) if v_data_zero else v_data @ w
    v_mc_w = np.zeros(n, dtype=float) if v_mc_zero else v_mc @ w
    data_bins = w * v_data_w
    mc_bins = w * v_mc_w

    # Pulls against the nominal (unshifted) model and the combined per-bin
    # error sqrt(V_ii); the card's (d - m) / sigma convention.
    diag = np.diag(v_mat)
    if not np.all(np.isfinite(diag) & (diag > 0.0)):
        raise ValueError(
            "every covariance diagonal entry must be positive and finite "
            "to form per-bin pulls"
        )
    sigma_bins = np.sqrt(diag)
    pulls = (values_arr - np.asarray(model, dtype=float)) / sigma_bins

    chi2_penalty = sum(
        (float(item.eta) / float(item.sigma_eta)) ** 2 for item in nuisance_list
    )

    # Assign every bin to its (xF region, pT region) sector.
    sector_bins: dict[tuple[str, str], list[int]] = {
        (xf, pt): [] for xf in SECTOR_XF_REGION_ORDER for pt in SECTOR_PT_REGION_ORDER
    }
    unassigned = 0
    for i in range(n):
        xf_region = _sector_region_of(x_mid_per_bin[i], SECTOR_XF_REGION_ORDER)
        pt_region = _sector_region_of(p_mid_per_bin[i], SECTOR_PT_REGION_ORDER)
        if xf_region is None or pt_region is None:
            unassigned += 1
            continue
        sector_bins[(xf_region, pt_region)].append(i)

    quad_total = float(np.sum(quad_bins))
    observable_total = quad_total + chi2_penalty
    assigned_bins = sorted(
        b for members in sector_bins.values() for b in members
    )
    first_assigned = assigned_bins[0] if assigned_bins else None
    penalty_booked = False
    sectors: dict[str, dict] = {}
    for (xf_region, pt_region), bins in sector_bins.items():
        key = f"{d.species}|{d.observable}|{xf_region}|{pt_region}"
        if bins:
            chi2_data_s = float(np.sum(data_bins[bins]))
            chi2_mc_s = float(np.sum(mc_bins[bins]))
            chi2_quad_s = float(np.sum(quad_bins[bins]))
            residual_mean_s = float(np.mean(pulls[bins]))
            worst = max(bins, key=lambda i: abs(float(pulls[i])))
            worst_bin = {
                "bin_index": int(worst),
                "d": float(values[worst]),
                "m": float(model[worst]),
                "sigma": float(sigma_bins[worst]),
                "pull": float(pulls[worst]),
            }
            n_bins_s = len(bins)
        else:
            chi2_data_s = 0.0
            chi2_mc_s = 0.0
            chi2_quad_s = 0.0
            residual_mean_s = 0.0
            worst_bin = None
            n_bins_s = 0
        # The nuisance penalty is not bin-local; it is booked once, into
        # the sector of the observable's lowest-index assigned bin.
        chi2_nuisance_s = 0.0
        if bins and not penalty_booked and bins[0] == first_assigned:
            chi2_nuisance_s = chi2_penalty
            penalty_booked = True
        chi2_total_s = chi2_quad_s + chi2_nuisance_s
        fraction = (
            chi2_total_s / observable_total if observable_total > 0.0 else 0.0
        )
        sectors[key] = {
            "chi2_data": chi2_data_s,
            "chi2_mc": chi2_mc_s,
            "chi2_nuisance": chi2_nuisance_s,
            "chi2_total": chi2_total_s,
            "fraction_of_total": fraction,
            "residual_mean": residual_mean_s,
            "worst_bin": worst_bin,
            "n_bins": n_bins_s,
        }
    return {"sectors": sectors, "unassigned_bins": int(unassigned)}


def write_sector_report(
    path: str,
    decomposition: dict,
    total_chi2: float,
    dof: int,
) -> None:
    """Write ``sector_report.json``: the decomposition plus the top-level
    block (``total_chi2``, ``dof``, ``n_sectors``, ``unassigned_bins``,
    and the per-species / per-observable / per-region totals keyed by the
    six named regions of ``SECTOR_REGIONS`` as marginals).  ``dof`` is
    the driver's degree-of-freedom count (on the staged-evaluation smoke
    path: the windowed bin count, no fitted parameters)."""
    sectors = dict(decomposition.get("sectors", {}))
    unassigned = int(decomposition.get("unassigned_bins", 0))
    per_species: dict[str, float] = {}
    per_observable: dict[str, float] = {}
    per_region: dict[str, float] = {name: 0.0 for name in SECTOR_REGIONS}
    for key, payload in sectors.items():
        species, observable, xf_region, pt_region = key.split("|", 3)
        value = float(payload["chi2_total"])
        per_species[species] = per_species.get(species, 0.0) + value
        per_observable[observable] = per_observable.get(observable, 0.0) + value
        per_region[xf_region] = per_region.get(xf_region, 0.0) + value
        per_region[pt_region] = per_region.get(pt_region, 0.0) + value
    payload = {
        "schema": _SECTOR_REPORT_SCHEMA,
        "total_chi2": float(total_chi2),
        "dof": int(dof),
        "n_sectors": len(sectors),
        "unassigned_bins": unassigned,
        "per_species_totals": per_species,
        "per_observable_totals": per_observable,
        "per_region_totals": per_region,
        "sectors": sectors,
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
