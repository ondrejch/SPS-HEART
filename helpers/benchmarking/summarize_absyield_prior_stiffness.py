#!/usr/bin/env python3
"""Summarize absolute-yield prior stiffness scans and plot heatmaps."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "dev" / "00runs"


def _heatmap(
    ax: plt.Axes,
    rows: List[Dict[str, float | str]],
    metric: str,
    mode: str,
    title: str,
    cmap: str = "viridis",
) -> None:
    rr = [r for r in rows if str(r["mode"]) == mode]
    lams = sorted({float(r["lambda"]) for r in rr})
    sigs = sorted({float(r["sigma_rel"]) for r in rr})
    mat = np.full((len(sigs), len(lams)), np.nan, dtype=float)
    for r in rr:
        i = sigs.index(float(r["sigma_rel"]))
        j = lams.index(float(r["lambda"]))
        mat[i, j] = float(r[metric])
    im = ax.imshow(mat, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(lams)))
    ax.set_xticklabels([f"{x:.2g}" for x in lams])
    ax.set_yticks(np.arange(len(sigs)))
    ax.set_yticklabels([f"{x:.2g}" for x in sigs])
    ax.set_xlabel(r"$\lambda_{abs}$")
    ax.set_ylabel(r"$\sigma_{rel,min}$")
    ax.set_title(f"{mode}: {title}")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8, color="white")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-stamp", default="2026-03-31r2")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    pat = re.compile(
        rf"constrained_refit_(table9|pdg)_absyield_scan_l(\d+)_s(\d+)_{re.escape(str(args.run_stamp))}$"
    )
    rows: List[Dict[str, float | str]] = []
    for p in sorted(RUNS_ROOT.glob(f"constrained_refit_*_absyield_scan_*_{args.run_stamp}")):
        m = pat.fullmatch(p.name)
        if not m:
            continue
        summary = p / "multispecies_constrained_summary.json"
        if not summary.exists():
            continue
        obj = json.loads(summary.read_text(encoding="utf-8"))
        gm = obj.get("resonance", {}).get("global_metrics", {})
        ay = obj.get("resonance", {}).get("priors", {}).get("absolute_yield", {})
        rows.append(
            {
                "run_name": p.name,
                "mode": str(m.group(1)),
                "lambda": int(m.group(2)) / 1000.0,
                "sigma_rel": int(m.group(3)) / 1000.0,
                "full_log_pull_rms": float(gm.get("full_log_pull_rms", np.nan)),
                "full_weighted_rel_rmse": float(gm.get("full_weighted_rel_rmse", np.nan)),
                "full_smape_percent": float(gm.get("full_smape_percent", np.nan)),
                "abs_yield_chi2_per_row": float(ay.get("fit_chi2_per_row", np.nan)),
                "abs_yield_chi2": float(ay.get("fit_chi2", np.nan)),
            }
        )
    if not rows:
        raise RuntimeError(f"No scan runs found for stamp {args.run_stamp}")

    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else (RUNS_ROOT / f"absyield_prior_stiffness_{args.run_stamp}")
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "absyield_prior_stiffness_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Compromise score: equally weight z-scored log-pull and log10(abs-chi2/row)
    by_mode: Dict[str, List[Dict[str, float | str]]] = {"table9": [], "pdg": []}
    for r in rows:
        by_mode[str(r["mode"])].append(r)
    for mode in by_mode:
        rr = by_mode[mode]
        a = np.asarray([float(x["full_log_pull_rms"]) for x in rr], dtype=float)
        b = np.log10(np.maximum(np.asarray([float(x["abs_yield_chi2_per_row"]) for x in rr], dtype=float), 1e-12))
        az = (a - np.mean(a)) / max(np.std(a), 1e-12)
        bz = (b - np.mean(b)) / max(np.std(b), 1e-12)
        for i, item in enumerate(rr):
            item["compromise_score"] = float(0.5 * az[i] + 0.5 * bz[i])

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.8))
    _heatmap(axes[0, 0], rows, "full_log_pull_rms", "table9", "log-pull RMS", cmap="viridis")
    _heatmap(axes[0, 1], rows, "full_log_pull_rms", "pdg", "log-pull RMS", cmap="viridis")
    _heatmap(axes[1, 0], rows, "abs_yield_chi2_per_row", "table9", r"$\chi^2_{\rm abs}/N_{\rm rows}$", cmap="magma")
    _heatmap(axes[1, 1], rows, "abs_yield_chi2_per_row", "pdg", r"$\chi^2_{\rm abs}/N_{\rm rows}$", cmap="magma")
    fig.suptitle("Absolute-yield prior stiffness scan", fontsize=13)
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.97))
    fig_path = out_dir / "absyield_prior_stiffness_heatmaps.png"
    fig.savefig(fig_path, dpi=170)
    plt.close(fig)

    md = out_dir / "ABSYIELD_PRIOR_STIFFNESS_SUMMARY.md"
    lines: List[str] = []
    lines.append("# Absolute-Yield Prior Stiffness Summary")
    lines.append("")
    lines.append(f"- run_stamp: `{args.run_stamp}`")
    lines.append(f"- scenarios: `{len(rows)}`")
    lines.append(f"- metrics: `{csv_path}`")
    lines.append(f"- heatmap: `{fig_path}`")
    lines.append("")
    lines.append("| mode | best log-pull | lambda | sigma_rel | best abs chi2/row | lambda | sigma_rel | best compromise | lambda | sigma_rel |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mode in ("table9", "pdg"):
        rr = by_mode[mode]
        best_log = min(rr, key=lambda x: float(x["full_log_pull_rms"]))
        best_abs = min(rr, key=lambda x: float(x["abs_yield_chi2_per_row"]))
        best_cmp = min(rr, key=lambda x: float(x["compromise_score"]))
        lines.append(
            f"| {mode} | {float(best_log['full_log_pull_rms']):.6g} | {float(best_log['lambda']):.3g} | {float(best_log['sigma_rel']):.3g} | "
            f"{float(best_abs['abs_yield_chi2_per_row']):.6g} | {float(best_abs['lambda']):.3g} | {float(best_abs['sigma_rel']):.3g} | "
            f"{float(best_cmp['compromise_score']):+.4f} | {float(best_cmp['lambda']):.3g} | {float(best_cmp['sigma_rel']):.3g} |"
        )
    lines.append("")
    lines.append("| mode | lambda | sigma_rel | log-pull RMS | WRRMSE | abs chi2/row | compromise score | run |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for mode in ("table9", "pdg"):
        rr = sorted(by_mode[mode], key=lambda x: (float(x["lambda"]), float(x["sigma_rel"])))
        for r in rr:
            lines.append(
                f"| {mode} | {float(r['lambda']):.3g} | {float(r['sigma_rel']):.3g} | {float(r['full_log_pull_rms']):.6g} | "
                f"{float(r['full_weighted_rel_rmse']):.6g} | {float(r['abs_yield_chi2_per_row']):.6g} | "
                f"{float(r['compromise_score']):+.4f} | `{r['run_name']}` |"
            )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {fig_path}")
    print(f"wrote {md}")


if __name__ == "__main__":
    main()

