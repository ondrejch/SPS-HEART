# Not used by the HEART p+p fits (marked 2026-11-04): no production generation, fit, probe tool or paper script
# runs this module (an execution trace of the R137 generation and an import scan of the engine, the probe tools
# and the paper scripts). Code reviews of the p+p model can skip it. It stays until the cleanup and refactor after
# the p+A run.
"""Physics-separated observable combinations of plan section 7.3 (Round 1,
task card TASK-20260911-r1-run-r11-legacy-observable-decomposition; plan
section 4.3 names this module).

The module evaluates the pre-registered physics-separated combinations

    pi_sigma      = f(pi-) + f(pi+)
    pi_delta      = f(pi-) - f(pi+)
    k_pair        = 2 * f(K-)
    k_associated  = f(K+) - f(K-)
    b_pair        = 2 * f(pbar)
    b_transport   = f(p) - f(pbar)

as element-wise operations on per-bin binned values, plus the plan
section 7.3 moment/integral observables: the forward / central / backward
xF integrals, the two mean-transverse-momentum conventions (the seagull
moments), and the ranked residual map.  Everything here is a mathematical
operation on already-binned values; no parameter is fit and no model is
changed (the R11 run is a pure evaluation of the sealed R01 p+p archive).

Documented choices where the card is silent:

- Bin layout.  ``bin_edges`` is accepted in the same two forms as
  :func:`na49_mixing.fit_likelihood.sector_decomposition` reads it: the
  nested pair ``(xF_edges, pT_edges)`` with the grid form (bin ``i`` at
  ``(ix, ip) = (i mod n_x, i // n_x)``, the pT-major, xF-minor layout of
  ``_bin_tracks``) taking precedence, or the per-bin form (``2n`` edges
  per axis; each bin carries its own cell edges, the sparse-tabulation
  convention of the NA49 tables).  A flat 1-D edge tuple (grid form
  ``n + 1`` edges or per-bin form ``2n`` edges) is also accepted for
  1-D tests; it carries no pT widths.  Because the NA49 tables are
  sparse (e.g. 229 pi- points on an 18 x 18 center grid), the per-bin
  ``(ix, ip)`` grid indices can be passed explicitly through the
  optional ``bin_indices`` argument; the dense-grid layout rule is used
  when it is omitted.

- Region split of the integral.  The forward / central / backward split
  is the card's own two-boundary split at the bin xF midpoints: forward
  ``xF > 0.2``, central ``0.0 <= xF <= 0.2``, backward ``xF < 0.0``.
  It is deliberately coarser than the three-way
  :data:`na49_mixing.fit_likelihood.SECTOR_REGIONS` xF split used for
  the ranked residual map (which has a transition region).

- Integral rule.  Each bin contributes ``value * (xF width) * (pT
  width)`` (the midpoint-rectangle sum over the bin midpoints; the
  card's "trapezoidal rule over the bin midpoints").  With a flat 1-D
  edge tuple the pT width is absent and each bin contributes
  ``value * (xF width)``.  For the p+p paper window ``xF in [0, 0.4]``
  no bin midpoint lies below 0, so the backward integral is 0; the
  backward region exists for the p+A extension.

- Seagull densities.  The two conventions are Fischer 2022's equations
  (48) and (49): the xF-convention density is ``(pT / E) * f`` and the
  y-convention density is ``pT * f``.  The mean transverse momentum per
  xF bin is ``sum(pT * density * dpT) / sum(density * dpT)`` over the
  pT cells of that xF bin (the xF width is constant within one xF bin
  and cancels in the ratio).  The particle energy is reconstructed from
  the binned kinematics as ``E = sqrt(m^2 + pL^2 + pT^2)`` with
  ``pL = xF * sqrt(s) / 2`` at the cell midpoints, so the function needs
  the species mass and ``sqrt(s)`` (keyword-only ``mass_gev`` /
  ``sqrt_s_gev``).  A bin whose density weight vanishes yields ``None``
  for that convention (recorded, never silently dropped).  The two
  conventions differ by up to 0.1 GeV/c for pions; the output keys
  carry the mandatory subscripts ``pT_x_per_xF`` and ``pT_y_per_xF``.

- Combination alignment.  :func:`evaluate_combination` operates
  element-wise and requires the species tuples to be aligned bin by
  bin; aligning two species onto their common ``(xF, pT)`` cells is the
  driver's responsibility (the R11 driver aligns on the intersection of
  the sealed tables' tabulated midpoints).  A ``ratio`` bin whose
  denominator is 0 yields ``None`` and is flagged through
  :func:`undefined_combination_bins`.

- Residual cells.  A cell of the ranked residual map is
  (sector x observable x region x bin); its key is the pipe-separated
  ``"{sector}|{observable}|{region}|{bin_index}"`` encoding used by the
  sector report, with ``region`` one of the xF regions of
  :data:`na49_mixing.fit_likelihood.SECTOR_REGIONS`.  The pull is the
  card-7 combined-uncertainty residual ``delta_over_sigma = (d - m) /
  sigma`` with ``sigma = sqrt(V)`` the combined data + MC + nuisance
  uncertainty; ``pull`` and ``delta_over_sigma`` carry the same value.
  A cell with infinite ``V`` (an empty MC bin, the R01 convention) has
  an infinite sigma and a pull of 0.0: it is recorded, ranked last, and
  never silently dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "ObservableCombination",
    "OBSERVABLE_COMBINATIONS",
    "COMBINATION_NAMES",
    "FORWARD_XF_BOUND",
    "RANKED_RESIDUAL_THRESHOLD",
    "evaluate_combination",
    "undefined_combination_bins",
    "forward_central_backward_integral",
    "seagull_moment",
    "ranked_residual_map",
    "residual_cell_key",
    "parse_residual_cell_key",
]


@dataclass(frozen=True)
class ObservableCombination:
    """One pre-registered physics-separated observable combination."""

    name: str  # e.g. "pi_sigma", "k_pair", "b_transport"
    species: tuple[str, ...]  # the species involved
    operator: str  # "sum", "difference", "ratio"
    definition: str  # the mathematical definition (a string, for the report)


OBSERVABLE_COMBINATIONS: tuple[ObservableCombination, ...] = (
    ObservableCombination(
        "pi_sigma", ("pi-", "pi+"), "sum", "f(pi-) + f(pi+)"
    ),
    ObservableCombination(
        "pi_delta", ("pi-", "pi+"), "difference", "f(pi-) - f(pi+)"
    ),
    ObservableCombination("k_pair", ("K-",), "sum", "2 * f(K-)"),
    ObservableCombination(
        "k_associated", ("K+", "K-"), "difference", "f(K+) - f(K-)"
    ),
    ObservableCombination("b_pair", ("pbar",), "sum", "2 * f(pbar)"),
    ObservableCombination(
        "b_transport", ("p", "pbar"), "difference", "f(p) - f(pbar)"
    ),
)

COMBINATION_NAMES: tuple[str, ...] = tuple(
    combo.name for combo in OBSERVABLE_COMBINATIONS
)

# The card's forward/central/backward split is a two-boundary split at
# xF = 0.2 with the backward boundary at xF = 0.0.
FORWARD_XF_BOUND = 0.2

# Flag threshold of the ranked residual map (the card-10 failure-map
# threshold).
RANKED_RESIDUAL_THRESHOLD = 3.0


def combination_by_name(name: str) -> ObservableCombination:
    """The pre-registered combination with ``name`` (ValueError otherwise)."""
    for combo in OBSERVABLE_COMBINATIONS:
        if combo.name == name:
            return combo
    raise ValueError(
        f"unknown combination {name!r}; the pre-registered set is "
        f"{COMBINATION_NAMES}"
    )


def evaluate_combination(
    combo: ObservableCombination,
    binned: dict[str, tuple[float, ...]],
) -> tuple[float | None, ...]:
    """Per-bin combination value, element-wise over the aligned species.

    ``"sum"`` adds the per-bin values of the combination's species (a
    single-species sum is multiplied by 2, the k_pair / b_pair form);
    ``"difference"`` subtracts the second species from the first (the
    order is the ``species`` tuple order); ``"ratio"`` divides the first
    by the second with a zero-division guard: a bin whose denominator is
    0 yields ``None`` and is flagged through
    :func:`undefined_combination_bins`.  All species tuples must have
    the same length (the caller aligns them on the common bins).
    """
    if combo.operator not in {"sum", "difference", "ratio"}:
        raise ValueError(
            f"combination {combo.name!r} carries unsupported operator "
            f"{combo.operator!r}"
        )
    if not combo.species:
        raise ValueError(f"combination {combo.name!r} names no species")
    series: list[tuple[float, ...]] = []
    for species in combo.species:
        if species not in binned:
            raise ValueError(
                f"combination {combo.name!r} needs binned values for "
                f"species {species!r}"
            )
        series.append(tuple(float(v) for v in binned[species]))
    length = len(series[0])
    for species, values in zip(combo.species, series):
        if len(values) != length:
            raise ValueError(
                f"combination {combo.name!r} needs aligned per-bin tuples; "
                f"species {species!r} has {len(values)} bins, the first "
                f"species has {length}"
            )
    if combo.operator == "sum":
        if len(series) == 1:
            return tuple(2.0 * v for v in series[0])
        totals = [0.0] * length
        for values in series:
            for i, v in enumerate(values):
                totals[i] += v
        return tuple(totals)
    first, second = series[0], series[1]
    if combo.operator == "difference":
        return tuple(a - b for a, b in zip(first, second))
    # ratio with the zero-division guard
    guarded: list[float | None] = []
    for a, b in zip(first, second):
        guarded.append(None if b == 0.0 else a / b)
    return tuple(guarded)


def undefined_combination_bins(
    values: tuple[float | None, ...],
) -> tuple[int, ...]:
    """Indices of the combination's undefined (None) bins."""
    return tuple(i for i, v in enumerate(values) if v is None)


def _per_bin_edges(
    bin_edges: tuple,
    n_values: int,
    bin_indices=None,
) -> list[tuple[float, float, float | None, float | None]]:
    """Per-bin ``(xF_lo, xF_hi, pT_lo, pT_hi)`` (pT None for a flat tuple).

    Accepts the nested pair ``(xF_edges, pT_edges)`` (grid form with
    precedence, or per-bin form) or a flat 1-D xF edge tuple (grid form
    ``n + 1`` or per-bin form ``2n``).  ``bin_indices`` optionally
    carries the explicit per-bin ``(ix, ip)`` grid indices for sparse
    tables; the dense-grid layout rule ``(i mod n_x, i // n_x)`` applies
    when it is omitted.
    """
    if (
        isinstance(bin_edges, tuple)
        and len(bin_edges) == 2
        and all(isinstance(part, (tuple, list)) for part in bin_edges)
    ):
        x_edges = tuple(float(v) for v in bin_edges[0])
        p_edges = tuple(float(v) for v in bin_edges[1])
        n_x = len(x_edges) - 1
        n_p = len(p_edges) - 1
        if bin_indices is not None and n_x >= 1 and n_p >= 1:
            # Explicit per-bin (ix, ip) grid indices: the sparse-table
            # convention (e.g. 229 pi- points filling cells of an 18 x 18
            # center grid), where n_x * n_p may exceed the bin count.
            pairs = _validated_indices(bin_indices, n_values, n_x, n_p)
            return [
                (x_edges[ix], x_edges[ix + 1], p_edges[ip], p_edges[ip + 1])
                for ix, ip in pairs
            ]
        if n_x >= 1 and n_p >= 1 and n_x * n_p == n_values:
            pairs = [(i % n_x, i // n_x) for i in range(n_values)]
            return [
                (x_edges[ix], x_edges[ix + 1], p_edges[ip], p_edges[ip + 1])
                for ix, ip in pairs
            ]
        if len(x_edges) == 2 * n_values and len(p_edges) == 2 * n_values:
            return [
                (
                    x_edges[2 * i],
                    x_edges[2 * i + 1],
                    p_edges[2 * i],
                    p_edges[2 * i + 1],
                )
                for i in range(n_values)
            ]
        raise ValueError(
            "bin_edges must be the nested pair (xF_edges, pT_edges) either "
            f"in grid form ((n_x + 1) * (n_p + 1) edges for {n_values} bins) "
            f"or per-bin form ({2 * n_values} edges per axis)"
        )
    edges = tuple(float(v) for v in bin_edges)
    if len(edges) == n_values + 1:
        return [
            (edges[i], edges[i + 1], None, None) for i in range(n_values)
        ]
    if len(edges) == 2 * n_values:
        return [
            (edges[2 * i], edges[2 * i + 1], None, None)
            for i in range(n_values)
        ]
    raise ValueError(
        "bin_edges must be a flat xF edge tuple in grid form (n + 1 edges) "
        f"or per-bin form (2n edges) for {n_values} bins"
    )


def _validated_indices(bin_indices, n_values: int, n_x: int, n_p: int):
    pairs = list(bin_indices)
    if len(pairs) != n_values:
        raise ValueError(
            f"bin_indices carries {len(pairs)} entries for {n_values} bins"
        )
    out = []
    for i, pair in enumerate(pairs):
        ix, ip = int(pair[0]), int(pair[1])
        if not (0 <= ix < n_x and 0 <= ip < n_p):
            raise ValueError(
                f"bin_indices[{i}] = {(ix, ip)} outside the "
                f"{n_x} x {n_p} grid"
            )
        out.append((ix, ip))
    return out


def _species_values(
    binned: dict[str, tuple[float, ...]], species: str
) -> tuple[float, ...]:
    if species not in binned:
        raise ValueError(f"binned values for species {species!r} are missing")
    values = tuple(float(v) for v in binned[species])
    if not values:
        raise ValueError(f"species {species!r} carries no bins")
    for i, v in enumerate(values):
        if v is None or not math.isfinite(v):
            raise ValueError(
                f"species {species!r} bin {i} carries a non-finite value"
            )
    return values


def forward_central_backward_integral(
    binned: dict[str, tuple[float, ...]],
    bin_edges: tuple,
    species: str,
    *,
    bin_indices=None,
) -> dict[str, float]:
    """Forward / central / backward xF integrals of one species' bins.

    Each bin contributes ``value * (xF width) * (pT width)`` (the
    midpoint-rectangle sum; a flat 1-D edge tuple has no pT widths).
    Region membership is the bin xF midpoint against the card's split:
    forward ``> 0.2``, central ``[0.0, 0.2]``, backward ``< 0.0``.  For
    the p+p paper window ``xF in [0, 0.4]`` the backward integral is 0
    (documented; the region exists for the p+A extension).
    """
    values = _species_values(binned, species)
    cells = _per_bin_edges(bin_edges, len(values), bin_indices)
    forward = central = backward = 0.0
    for value, (x_lo, x_hi, p_lo, p_hi) in zip(values, cells):
        p_width = 1.0 if p_lo is None or p_hi is None else p_hi - p_lo
        width = (x_hi - x_lo) * p_width
        mid = 0.5 * (x_lo + x_hi)
        if mid > FORWARD_XF_BOUND:
            forward += value * width
        elif mid >= 0.0:
            central += value * width
        else:
            backward += value * width
    return {"forward": forward, "central": central, "backward": backward}


def seagull_moment(
    binned: dict[str, tuple[float, ...]],
    bin_edges: tuple,
    species: str,
    *,
    bin_indices=None,
    sqrt_s_gev: float | None = None,
    mass_gev: float | None = None,
) -> dict[str, list[float | None]]:
    """The two mean-pT conventions per xF bin (the seagull moments).

    ``<pT>_x`` is the mean of pT over the density ``(pT / E) * f``
    (Fischer 2022 equation 48) and ``<pT>_y`` the mean over the density
    ``pT * f`` (equation 49), per xF bin.  The output keys carry the
    mandatory subscripts: ``{"pT_x_per_xF": [...], "pT_y_per_xF":
    [...]}``, one entry per distinct xF bin in ascending xF order.  A
    bin whose density weight vanishes yields ``None`` for that
    convention.  ``sqrt_s_gev`` and ``mass_gev`` are required: the
    ``(pT / E)`` density needs the particle energy, reconstructed from
    the binned kinematics as ``E = sqrt(m^2 + pL^2 + pT^2)`` with
    ``pL = xF * sqrt(s) / 2`` at the cell midpoints.
    """
    values = _species_values(binned, species)
    if sqrt_s_gev is None or mass_gev is None:
        raise ValueError(
            "seagull_moment needs sqrt_s_gev and mass_gev: the (pT / E) "
            "density reconstructs E from the binned kinematics"
        )
    sqrt_s = float(sqrt_s_gev)
    mass = float(mass_gev)
    if not (math.isfinite(sqrt_s) and sqrt_s > 0.0):
        raise ValueError(f"sqrt_s_gev must be positive and finite, got {sqrt_s!r}")
    if not (math.isfinite(mass) and mass >= 0.0):
        raise ValueError(f"mass_gev must be non-negative and finite, got {mass!r}")
    cells = _per_bin_edges(bin_edges, len(values), bin_indices)
    grouped: dict[float, list[tuple[float, float, float, float]]] = {}
    for value, cell in zip(values, cells):
        x_lo, x_hi, p_lo, p_hi = cell
        if p_lo is None or p_hi is None:
            raise ValueError(
                "seagull_moment needs the nested (xF_edges, pT_edges) bin "
                "edges: the densities are pT-weighted"
            )
        p_lo_f, p_hi_f = float(p_lo), float(p_hi)
        x_mid = 0.5 * (x_lo + x_hi)
        p_mid = 0.5 * (p_lo_f + p_hi_f)
        grouped.setdefault(x_mid, []).append(
            (value, p_mid, p_hi_f - p_lo_f, x_hi - x_lo)
        )
    pT_x: list[float | None] = []
    pT_y: list[float | None] = []
    for x_mid in sorted(grouped):
        p_l = x_mid * sqrt_s / 2.0
        num_x = den_x = num_y = den_y = 0.0
        for value, p_mid, dp, _dx in grouped[x_mid]:
            # The particle energy is per cell: E = sqrt(m^2 + pL^2 + pT^2)
            # at the cell's (xF, pT) midpoints.
            energy = math.sqrt(mass * mass + p_l * p_l + p_mid * p_mid)
            w_x = (p_mid / energy) * value
            w_y = p_mid * value
            num_x += p_mid * w_x * dp
            den_x += w_x * dp
            num_y += p_mid * w_y * dp
            den_y += w_y * dp
        pT_x.append(num_x / den_x if den_x != 0.0 else None)
        pT_y.append(num_y / den_y if den_y != 0.0 else None)
    return {"pT_x_per_xF": pT_x, "pT_y_per_xF": pT_y}


def residual_cell_key(
    sector: str, observable: str, region: str, bin_index: int
) -> str:
    """The pipe-separated cell key ``"{sector}|{observable}|{region}|{i}"``."""
    return f"{sector}|{observable}|{region}|{int(bin_index)}"


def parse_residual_cell_key(key: str) -> tuple[str, str, str, int]:
    """Parse a cell key into ``(sector, observable, region, bin_index)``."""
    parts = key.split("|")
    if len(parts) != 4:
        raise ValueError(
            f"cell key {key!r} must be "
            "'{sector}|{observable}|{region}|{bin_index}'"
        )
    sector, observable, region, bin_text = parts
    try:
        bin_index = int(bin_text)
    except ValueError:
        raise ValueError(
            f"cell key {key!r} carries a non-integer bin index {bin_text!r}"
        ) from None
    return sector, observable, region, bin_index


def ranked_residual_map(
    d: dict[str, tuple[float, ...]],
    m: dict[str, tuple[float, ...]],
    V: dict[str, tuple[float, ...]],
    top_n: int = 100,
    threshold: float = RANKED_RESIDUAL_THRESHOLD,
) -> dict:
    """Rank the residual cells by ``|pull|`` descending.

    ``d``, ``m``, and ``V`` are keyed by :func:`residual_cell_key` cell
    keys, each carrying a one-entry tuple (the cell's data value, model
    value, and combined variance).  ``sigma = sqrt(V)`` is the combined
    data + MC + nuisance uncertainty of card 7; ``pull =
    delta_over_sigma = (d - m) / sigma``.  A cell with infinite variance
    (an empty MC bin, the R01 convention) has an infinite sigma and a
    pull of 0.0: it is recorded, ranked last, and never silently
    dropped.  Ties break by sector, observable, region, then bin index
    (lexicographic).  The result carries every cell's count in
    ``n_total_cells`` and the count of cells with ``|pull| > threshold``
    in ``n_flagged``; ``rows`` holds the top ``top_n`` rows.
    """
    if not d:
        raise ValueError("ranked_residual_map needs at least one cell")
    for name, payload in ({"d": d, "m": m, "V": V}).items():
        missing = set(d) ^ set(payload)
        if missing:
            raise ValueError(
                f"{name} carries cells the other dicts do not: "
                f"{sorted(missing)[:5]}"
            )
    rows = []
    n_flagged = 0
    for key in sorted(d):
        sector, observable, region, bin_index = parse_residual_cell_key(key)
        d_vals, m_vals, v_vals = d[key], m[key], V[key]
        for name, vals in (("d", d_vals), ("m", m_vals), ("V", v_vals)):
            if len(vals) != 1:
                raise ValueError(
                    f"cell {key!r}: {name} must carry exactly one value, "
                    f"got {len(vals)}"
                )
        d_val, m_val, v_val = float(d_vals[0]), float(m_vals[0]), float(v_vals[0])
        for name, val in (("d", d_val), ("m", m_val)):
            if not math.isfinite(val):
                raise ValueError(f"cell {key!r}: {name} is not finite")
        if math.isnan(v_val) or v_val < 0.0:
            raise ValueError(f"cell {key!r}: V is not a variance ({v_val!r})")
        sigma = math.sqrt(v_val) if math.isfinite(v_val) else math.inf
        if math.isinf(sigma):
            pull = 0.0  # infinite combined error: recorded, never dropped
        elif sigma == 0.0:
            pull = 0.0 if d_val == m_val else math.inf
        else:
            pull = (d_val - m_val) / sigma
        if abs(pull) > threshold:
            n_flagged += 1
        rows.append(
            {
                "rank": 0,
                "sector": sector,
                "observable": observable,
                "region": region,
                "bin_index": bin_index,
                "d": d_val,
                "m": m_val,
                "sigma": sigma,
                "delta_over_sigma": pull,
                "pull": pull,
            }
        )
    rows.sort(
        key=lambda row: (
            -abs(row["pull"]),
            row["sector"],
            row["observable"],
            row["region"],
            row["bin_index"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return {
        "rows": rows[: max(int(top_n), 0)],
        "threshold": float(threshold),
        "n_flagged": n_flagged,
        "n_total_cells": len(rows),
    }
