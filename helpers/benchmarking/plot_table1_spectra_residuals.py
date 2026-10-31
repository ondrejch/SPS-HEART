#!/usr/bin/env python3
"""Plot SPS spectra for Table-1 style model comparisons.

This script consumes benchmark summary/details artifacts and reconstructs
external-model predictions (EPOS/UrQMD) with the same scaling logic as
`benchmark_sps_species.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "dev" / "00runs"


def _edges_from_centers(vals: np.ndarray) -> np.ndarray:
    vals = np.unique(np.asarray(vals))
    vals.sort()
    if vals.size == 1:
        return np.array([vals[0] - 0.5, vals[0] + 0.5], dtype=float)
    mids = 0.5 * (vals[:-1] + vals[1:])
    first = vals[0] - 0.5 * (vals[1] - vals[0])
    last = vals[-1] + 0.5 * (vals[-1] - vals[-2])
    return np.concatenate([[first], mids, [last]])


def _fit_single_scale(train_pred: np.ndarray, train_y: np.ndarray, train_sigma: np.ndarray) -> float:
    w = 1.0 / np.maximum(train_sigma, 1e-12) ** 2
    num = np.sum(w * train_pred * train_y)
    den = np.sum(w * train_pred * train_pred)
    return 0.0 if den <= 0 else float(num / den)


def _load_external_grid_counts(
    csv_path: Path,
    xedges: np.ndarray,
    pedges: np.ndarray,
    ix_data: np.ndarray,
    ip_data: np.ndarray,
) -> np.ndarray:
    xs: List[float] = []
    ps: List[float] = []
    vs: List[float] = []
    with csv_path.open(encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            xs.append(float(row["xf"]))
            ps.append(float(row["pt"]))
            vs.append(float(row["value"]))

    xs = np.asarray(xs)
    ps = np.asarray(ps)
    vs = np.asarray(vs)
    hsum = np.zeros((len(xedges) - 1, len(pedges) - 1), dtype=float)
    ix = np.digitize(xs, xedges) - 1
    ip = np.digitize(ps, pedges) - 1
    for i, j, v in zip(ix, ip, vs):
        if 0 <= i < hsum.shape[0] and 0 <= j < hsum.shape[1]:
            hsum[i, j] += v
    return hsum[ix_data, ip_data]


def _nearest_values(targets: List[float], available: np.ndarray) -> np.ndarray:
    out = []
    for t in targets:
        idx = int(np.argmin(np.abs(available - t)))
        out.append(float(available[idx]))
    return np.array(sorted(set(out)), dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make spectra panels for benchmark models (Table-1 style)."
    )
    parser.add_argument("--run-description", default="local_default")
    parser.add_argument(
        "--summary-json",
        default="",
    )
    parser.add_argument(
        "--details-npz",
        default="",
    )
    parser.add_argument(
        "--out-png",
        default="",
    )
    parser.add_argument(
        "--xf-slices",
        default="0.00,0.05,0.10,0.20,0.30,0.40",
        help="Comma-separated xF slice centers (nearest available values used).",
    )
    parser.add_argument("--species-label", default=r"$\pi^-$")
    parser.add_argument(
        "--resonance-label",
        default="na49_mixing (best)",
        help="Legend label for the resonance-fit curve.",
    )
    args = parser.parse_args()

    run_dir = RUNS_ROOT / args.run_description
    summary_path = Path(args.summary_json) if args.summary_json else (run_dir / "na49_benchmark_piminus_gateway_pythia_summary.json")
    details_path = Path(args.details_npz) if args.details_npz else (run_dir / "na49_benchmark_piminus_gateway_pythia_details.npz")
    out_png = Path(args.out_png) if args.out_png else (run_dir / "figures" / "table1_piminus_spectra_residuals.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    with summary_path.open(encoding="utf-8") as fh:
        summary = json.load(fh)
    z = np.load(details_path)

    xf = z["xf"]
    pt = z["pt"]
    y = z["y"]
    sigma = z["sigma"]
    train_idx = z["train_idx"]
    resonance_pred = z["resonance_pred"]
    pythia_pred = z["pythia_pred"]

    xedges = _edges_from_centers(xf)
    pedges = _edges_from_centers(pt)
    ix_data = np.digitize(xf, xedges) - 1
    ip_data = np.digitize(pt, pedges) - 1

    resonance_label = str(args.resonance_label).strip() or "na49_mixing (best)"
    model_preds: Dict[str, np.ndarray] = {resonance_label: resonance_pred}
    model_sigma: Dict[str, np.ndarray] = {}

    # Resonance statistical uncertainty from finite component samples.
    M = z["resonance_matrix"]
    w = z["resonance_weights"]
    if M.size and w.size and M.shape[1] == w.shape[0]:
        var_res = (M * (w ** 2)).sum(axis=1)
        model_sigma[resonance_label] = np.sqrt(np.maximum(var_res, 0.0))

    if pythia_pred.size == y.size:
        model_preds["PYTHIA8"] = pythia_pred
        # Fallback visual envelope when per-bin raw MC counts are unavailable.
        model_sigma["PYTHIA8"] = 0.08 * np.maximum(np.abs(pythia_pred), 1e-12)

    cfg = summary.get("config", {})
    for model_name, key in (("EPOS", "epos_csv"), ("UrQMD", "urqmd_csv")):
        csv_name = cfg.get(key, "")
        if not csv_name:
            continue
        csv_path = Path(csv_name)
        if not csv_path.exists():
            continue
        raw = _load_external_grid_counts(csv_path, xedges, pedges, ix_data, ip_data)
        alpha = _fit_single_scale(raw[train_idx], y[train_idx], sigma[train_idx])
        model_preds[model_name] = alpha * raw
        model_sigma[model_name] = abs(alpha) * np.sqrt(np.maximum(raw, 0.0))

    available_xf = np.unique(np.round(xf, 8))
    xf_targets = [float(x.strip()) for x in args.xf_slices.split(",") if x.strip()]
    xf_slices = _nearest_values(xf_targets, available_xf)

    # Fixed 2x3 publication layout.
    n_rows = 2
    n_cols = 3
    n_cells = n_rows * n_cols
    xf_slices = xf_slices[:n_cells]
    n = len(xf_slices)

    fig = plt.figure(figsize=(13.2, 8.2), constrained_layout=False)
    outer = fig.add_gridspec(
        n_rows,
        n_cols,
        left=0.06,
        right=0.995,
        bottom=0.07,
        top=0.88,
        wspace=0.12,
        hspace=0.14,
    )

    colors = {
        resonance_label: "#1b9e77",
        "PYTHIA8": "#d95f02",
        "EPOS": "#7570b3",
        "UrQMD": "#e7298a",
    }

    plot_axes = []
    for i, xsel in enumerate(xf_slices):
        r = i // n_cols
        c_idx = i % n_cols
        ax = fig.add_subplot(outer[r, c_idx])
        plot_axes.append(ax)

        mask = np.isclose(xf, xsel, atol=1e-8)
        x = pt[mask]
        yv = y[mask]
        sv = sigma[mask]
        order = np.argsort(x)
        x = x[order]
        yv = yv[order]
        sv = sv[order]

        ax.errorbar(
            x,
            yv,
            yerr=sv,
            fmt="o",
            color="black",
            mfc="white",
            mec="black",
            ms=4.6,
            lw=1.2,
            elinewidth=1.2,
            capsize=3,
            capthick=1.0,
            label="NA49",
            zorder=5,
        )

        for model, pred in model_preds.items():
            pv = pred[mask][order]
            ps = model_sigma.get(model, None)
            c = colors.get(model, None)
            if ps is not None:
                psv = ps[mask][order]
                # Avoid visually dominant bands in very low-stat bins.
                psv = np.minimum(psv, 0.60 * np.maximum(np.abs(pv), 1e-12))
                low = np.maximum(pv - psv, 1e-8)
                high = np.maximum(pv + psv, 1e-8)
                ax.fill_between(x, low, high, color=c, alpha=0.18, lw=0.0)
                ax.plot(x, pv, "-o", ms=2.8, lw=1.3, color=c, label=model, alpha=0.95)
            else:
                ax.plot(x, pv, "-o", ms=2.8, lw=1.3, color=c, label=model)

        ax.set_yscale("log")
        if c_idx == 0:
            ax.set_ylabel("yield", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.text(
            0.98,
            0.92,
            f"$x_F={xsel:.2f}$",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="right",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "0.75", "alpha": 0.8},
        )
        if r == (n_rows - 1):
            ax.set_xlabel(r"$p_T$ [GeV/$c$]")
        ax.tick_params(axis="both", labelsize=8)

    # Blank cells if fewer than 6 slices were provided.
    for i in range(n, n_cells):
        r = i // n_cols
        c_idx = i % n_cols
        ax_blank = fig.add_subplot(outer[r, c_idx])
        ax_blank.axis("off")

    handles, labels = plot_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=len(labels),
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.06,
        0.972,
        f"NA49 {args.species_label}: xF-sliced spectra vs $p_T$, 158 GeV/$c$",
        fontsize=10,
    )

    fig.savefig(out_png, dpi=180)
    print(f"Saved figure: {out_png}")


if __name__ == "__main__":
    main()
