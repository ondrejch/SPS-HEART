#!/usr/bin/env python3
"""Plot shared-fit spectra panels for one species from constrained multispecies run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from helpers.benchmarking.sps_core import load_multispecies_details_npz


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "dev" / "00runs"


def _nearest_values(targets: List[float], available: np.ndarray) -> np.ndarray:
    out: List[float] = []
    for t in targets:
        idx = int(np.argmin(np.abs(available - t)))
        out.append(float(available[idx]))
    return np.array(sorted(set(out)), dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot xF-sliced spectra from shared multispecies fit.")
    parser.add_argument("--run-description", default="sharedfit_bestmix_allmodels_2026-03-30")
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--details-npz", default="")
    parser.add_argument("--species", default="pi-", help="Canonical species key: pi-, pi+, k+, k-, p, pbar")
    parser.add_argument("--species-label", default="", help="Plot title species label override")
    parser.add_argument(
        "--xf-slices",
        default="0.00,0.05,0.10,0.20,0.30,0.40",
        help="Comma-separated xF slice centers (nearest available values used).",
    )
    parser.add_argument("--out-png", default="")
    args = parser.parse_args()

    run_dir = RUNS_ROOT / args.run_description
    summary_path = Path(args.summary_json) if args.summary_json else (run_dir / "multispecies_constrained_summary.json")
    details_path = Path(args.details_npz) if args.details_npz else (run_dir / "multispecies_constrained_details.npz")
    out_png = Path(args.out_png) if args.out_png else (run_dir / "figures" / f"{args.species}_sharedfit_spectra.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    species = payload.get("species", {})
    if args.species not in species:
        raise SystemExit(f"species '{args.species}' not found in {summary_path}")

    tag = str(species[args.species]["tag"])
    sp_label = str(args.species_label).strip() or str(species[args.species]["name"])

    with load_multispecies_details_npz(details_path, summary_payload=payload) as z:
        xf = np.asarray(z[f"{tag}_xf"], dtype=float)
        pt = np.asarray(z[f"{tag}_pt"], dtype=float)
        y = np.asarray(z[f"{tag}_y"], dtype=float)
        sigma = np.asarray(z[f"{tag}_sigma"], dtype=float)
        pred_res = np.asarray(z[f"{tag}_pred_res"], dtype=float)

        model_preds: Dict[str, np.ndarray] = {
            "na49_mixing (shared)": pred_res,
        }
        if f"{tag}_pred_pythia" in z.files:
            model_preds["PYTHIA8"] = np.asarray(z[f"{tag}_pred_pythia"], dtype=float)
        if f"{tag}_pred_epos" in z.files:
            model_preds["EPOS"] = np.asarray(z[f"{tag}_pred_epos"], dtype=float)
        if f"{tag}_pred_urqmd" in z.files:
            model_preds["UrQMD"] = np.asarray(z[f"{tag}_pred_urqmd"], dtype=float)

    # Visual uncertainty envelopes: conservative fixed fractional bands.
    model_sigma: Dict[str, np.ndarray] = {
        k: 0.08 * np.maximum(np.abs(v), 1e-12) for k, v in model_preds.items()
    }

    available_xf = np.unique(np.round(xf, 8))
    xf_targets = [float(x.strip()) for x in str(args.xf_slices).split(",") if x.strip()]
    xf_slices = _nearest_values(xf_targets, available_xf)

    n_rows, n_cols = 2, 3
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
        "na49_mixing (shared)": "#1b9e77",
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
            psv = model_sigma[model][mask][order]
            c = colors.get(model)
            psv = np.minimum(psv, 0.60 * np.maximum(np.abs(pv), 1e-12))
            low = np.maximum(pv - psv, 1e-8)
            high = np.maximum(pv + psv, 1e-8)
            ax.fill_between(x, low, high, color=c, alpha=0.18, lw=0.0)
            ax.plot(x, pv, "-o", ms=2.8, lw=1.3, color=c, label=model, alpha=0.95)

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
        f"NA49 {sp_label}: xF-sliced spectra vs $p_T$, shared resonance fit",
        fontsize=10,
    )

    fig.savefig(out_png, dpi=180)
    print(f"Saved figure: {out_png}")


if __name__ == "__main__":
    main()

