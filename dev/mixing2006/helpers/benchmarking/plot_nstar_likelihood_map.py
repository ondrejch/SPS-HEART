#!/usr/bin/env python3
"""Plot likelihood maps over N* tower settings from ladder scan outputs."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Row:
    scenario: str
    tower_max_mass: float
    shape_tag: str
    level_count: int
    summary_json: Path
    full_log_chi2: float
    nll: float
    delta_nll: float


def _parse_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _shape_sort_key(shape: str) -> Tuple[int, float]:
    s = str(shape)
    if s == "linear":
        return (0, 0.0)
    if s == "regge_m2":
        return (1, 0.0)
    if s.startswith("hagedorn_t"):
        try:
            val = float(s.replace("hagedorn_t", "")) / 1000.0
        except ValueError:
            val = 0.0
        return (2, val)
    return (3, 0.0)


def _rows_from_csv(csv_path: Path) -> List[Row]:
    raw_rows = _parse_csv(csv_path)
    tmp: List[Tuple[str, float, str, int, Path, float]] = []
    for rr in raw_rows:
        summary_json = Path(rr["summary_json"]).resolve()
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
        g = payload["resonance"]["global_metrics"]
        full_log_chi2 = float(g["full_log_chi2"])
        tmp.append(
            (
                str(rr["scenario"]),
                float(rr["tower_max_mass"]),
                str(rr["shape_tag"]),
                int(rr["level_count"]),
                summary_json,
                full_log_chi2,
            )
        )

    if not tmp:
        raise RuntimeError(f"No rows found in {csv_path}")

    best_nll = 0.5 * min(x[5] for x in tmp)
    rows: List[Row] = []
    for scenario, umax, shape, nlev, sj, log_chi2 in tmp:
        nll = 0.5 * float(log_chi2)
        rows.append(
            Row(
                scenario=scenario,
                tower_max_mass=float(umax),
                shape_tag=shape,
                level_count=int(nlev),
                summary_json=sj,
                full_log_chi2=float(log_chi2),
                nll=float(nll),
                delta_nll=float(nll - best_nll),
            )
        )
    return rows


def _build_grid(
    rows: List[Row],
    *,
    shape: str,
    umax_vals: np.ndarray,
    level_vals: np.ndarray,
) -> np.ndarray:
    grid = np.full((len(level_vals), len(umax_vals)), np.nan, dtype=float)
    by_key: Dict[Tuple[float, int], Row] = {}
    for r in rows:
        if r.shape_tag != shape:
            continue
        by_key[(float(r.tower_max_mass), int(r.level_count))] = r
    for iy, nlev in enumerate(level_vals):
        for ix, umax in enumerate(umax_vals):
            rr = by_key.get((float(umax), int(nlev)))
            if rr is not None:
                grid[iy, ix] = float(rr.delta_nll)
    return grid


def _annotate_best(ax: plt.Axes, rows: List[Row], umax_vals: np.ndarray, level_vals: np.ndarray, shape: str) -> None:
    cand = [r for r in rows if r.shape_tag == shape]
    if not cand:
        return
    best = min(cand, key=lambda r: r.delta_nll)
    ix = int(np.where(np.isclose(umax_vals, best.tower_max_mass))[0][0])
    iy = int(np.where(level_vals == best.level_count)[0][0])
    ax.scatter([ix], [iy], marker="*", s=220, color="black", edgecolors="white", linewidths=0.8, zorder=5)


def _plot_single_map(rows: List[Row], out_png: Path, title: str) -> None:
    shape_vals = sorted({r.shape_tag for r in rows}, key=_shape_sort_key)
    umax_vals = np.asarray(sorted({float(r.tower_max_mass) for r in rows}), dtype=float)
    level_vals = np.asarray(sorted({int(r.level_count) for r in rows}), dtype=int)

    ncols = len(shape_vals)
    fig, axes = plt.subplots(1, ncols, figsize=(3.8 * ncols, 4.5), squeeze=False)
    axes = axes[0]

    vmin = 0.0
    vmax = max(float(r.delta_nll) for r in rows)
    if vmax <= 0.0:
        vmax = 1.0

    im = None
    for i, shape in enumerate(shape_vals):
        ax = axes[i]
        grid = _build_grid(rows, shape=shape, umax_vals=umax_vals, level_vals=level_vals)
        im = ax.imshow(grid, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(shape.replace("_", " "), fontsize=10)
        ax.set_xticks(np.arange(len(umax_vals)))
        ax.set_xticklabels([f"{x:.1f}" for x in umax_vals], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(level_vals)))
        ax.set_yticklabels([str(x) for x in level_vals], fontsize=8)
        ax.set_xlabel(r"$u_{\max}$ [GeV]", fontsize=9)
        if i == 0:
            ax.set_ylabel("tower levels", fontsize=9)
        _annotate_best(ax, rows, umax_vals, level_vals, shape)

        # Light annotation for readable values.
        for iy in range(grid.shape[0]):
            for ix in range(grid.shape[1]):
                val = grid[iy, ix]
                if np.isfinite(val):
                    txt = "0" if abs(val) < 5e-4 else f"{val:.2f}"
                    ax.text(ix, iy, txt, ha="center", va="center", fontsize=7, color="white")

    cbar = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)
    cbar.set_label(r"$\Delta$NLL (0.5 $\times$ $\Delta\chi^2_{\log}$), lower is better", fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.95))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _plot_combined_many(runs: List[Tuple[List[Row], str]], out_png: Path) -> None:
    if not runs:
        raise ValueError("No runs passed to _plot_combined_many().")
    shape_vals = sorted(
        {r.shape_tag for rows, _ in runs for r in rows},
        key=_shape_sort_key,
    )
    umax_vals = np.asarray(
        sorted({float(r.tower_max_mass) for rows, _ in runs for r in rows}),
        dtype=float,
    )
    level_vals = np.asarray(
        sorted({int(r.level_count) for rows, _ in runs for r in rows}),
        dtype=int,
    )

    nrows = len(runs)
    ncols = len(shape_vals)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.6 * nrows + 1.0), squeeze=False)

    vmax = max(float(r.delta_nll) for rows, _ in runs for r in rows)
    if vmax <= 0.0:
        vmax = 1.0
    vmin = 0.0

    im = None
    for row_idx, (rows, row_title) in enumerate(runs):
        for col_idx, shape in enumerate(shape_vals):
            ax = axes[row_idx][col_idx]
            grid = _build_grid(rows, shape=shape, umax_vals=umax_vals, level_vals=level_vals)
            im = ax.imshow(grid, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
            if row_idx == 0:
                ax.set_title(shape.replace("_", " "), fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"{row_title}\nlevels", fontsize=9)
            ax.set_xticks(np.arange(len(umax_vals)))
            ax.set_xticklabels([f"{x:.1f}" for x in umax_vals], rotation=45, ha="right", fontsize=8)
            ax.set_yticks(np.arange(len(level_vals)))
            ax.set_yticklabels([str(x) for x in level_vals], fontsize=8)
            if row_idx == nrows - 1:
                ax.set_xlabel(r"$u_{\max}$ [GeV]", fontsize=9)
            _annotate_best(ax, rows, umax_vals, level_vals, shape)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.92, pad=0.01)
    cbar.set_label(r"$\Delta$NLL (0.5 $\times$ $\Delta\chi^2_{\log}$), lower is better", fontsize=9)
    fig.suptitle("Tower-settings likelihood map by shape, upper bound, and level count", fontsize=12)
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.95))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-a", required=True, help="First ladder summary CSV.")
    parser.add_argument("--csv-b", default="", help="Optional second ladder summary CSV.")
    parser.add_argument("--csv-c", default="", help="Optional third ladder summary CSV.")
    parser.add_argument("--label-a", default="Run A")
    parser.add_argument("--label-b", default="Run B")
    parser.add_argument("--label-c", default="Run C")
    parser.add_argument("--out-a", default="notes/figures/nstar_likelihood_map_a.png")
    parser.add_argument("--out-b", default="notes/figures/nstar_likelihood_map_b.png")
    parser.add_argument("--out-c", default="notes/figures/nstar_likelihood_map_c.png")
    parser.add_argument("--out-combined", default="notes/figures/nstar_likelihood_map_combined.png")
    args = parser.parse_args()

    rows_a = _rows_from_csv(Path(args.csv_a).resolve())
    _plot_single_map(rows_a, Path(args.out_a).resolve(), f"{args.label_a}: likelihood map over tower settings")
    print(f"saved {Path(args.out_a).resolve()}")

    runs_for_combined: List[Tuple[List[Row], str]] = [(rows_a, str(args.label_a))]

    if str(args.csv_b).strip():
        rows_b = _rows_from_csv(Path(args.csv_b).resolve())
        _plot_single_map(rows_b, Path(args.out_b).resolve(), f"{args.label_b}: likelihood map over tower settings")
        print(f"saved {Path(args.out_b).resolve()}")
        runs_for_combined.append((rows_b, str(args.label_b)))

    if str(args.csv_c).strip():
        rows_c = _rows_from_csv(Path(args.csv_c).resolve())
        _plot_single_map(rows_c, Path(args.out_c).resolve(), f"{args.label_c}: likelihood map over tower settings")
        print(f"saved {Path(args.out_c).resolve()}")
        runs_for_combined.append((rows_c, str(args.label_c)))

    if len(runs_for_combined) >= 2:
        _plot_combined_many(runs_for_combined, Path(args.out_combined).resolve())
        print(f"saved {Path(args.out_combined).resolve()}")


if __name__ == "__main__":
    main()
