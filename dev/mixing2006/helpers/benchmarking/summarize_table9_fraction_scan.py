#!/usr/bin/env python3
"""Summarize gateway Table-9 fraction-prior scan runs and plot key metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "00runs"


def _parse_name(name: str, stamp: str) -> tuple[str, float] | None:
    suffix = f"_{stamp}"
    if not name.endswith(suffix):
        return None
    core = name[: -len(suffix)]
    if "_absyield_t9frac_" not in core:
        return None
    left, frac_tag = core.split("_absyield_t9frac_", 1)
    if not left.startswith("constrained_refit_"):
        return None
    mode = left.replace("constrained_refit_", "", 1)
    if not frac_tag.startswith("f"):
        return None
    raw = frac_tag[1:]
    try:
        frac_lambda = float(raw.replace("p", "."))
    except Exception:
        return None
    return mode, frac_lambda


def _collect(stamp: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for run_dir in sorted(RUNS_ROOT.glob(f"constrained_refit_*_absyield_t9frac_*_{stamp}")):
        parsed = _parse_name(run_dir.name, stamp)
        if parsed is None:
            continue
        mode, frac_lambda = parsed
        summary_json = run_dir / "multispecies_constrained_summary.json"
        if not summary_json.exists():
            continue
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
        res = payload.get("resonance", {})
        gm = res.get("global_metrics", {})
        pri = res.get("priors", {})
        t9 = pri.get("table9", {})
        ay = pri.get("absolute_yield", {})
        fit_rows = t9.get("fit_rows", []) if isinstance(t9, dict) else []
        active_rows = 0
        eta_fit_fraction = np.nan
        for rr in fit_rows:
            if not isinstance(rr, dict):
                continue
            if float(rr.get("fit_pi_minus_yield", 0.0)) > 0.0:
                active_rows += 1
            if str(rr.get("name", "")) == "eta(548)":
                eta_fit_fraction = float(rr.get("fit_fraction", np.nan))
        rows.append(
            {
                "run_name": run_dir.name,
                "mode": mode,
                "fraction_lambda": float(frac_lambda),
                "full_log_pull_rms": float(gm.get("full_log_pull_rms", np.nan)),
                "full_weighted_rel_rmse": float(gm.get("full_weighted_rel_rmse", np.nan)),
                "abs_yield_chi2_per_row": float(ay.get("fit_chi2_per_row", np.nan)),
                "table9_fraction_l2_per_row": float(t9.get("fit_fraction_l2_per_row", np.nan)),
                "table9_fit_total_pi_minus": float(t9.get("fit_total_pi_minus", np.nan)),
                "table9_active_rows": int(active_rows),
                "eta_fit_fraction": float(eta_fit_fraction),
                "table9_lambda": float(t9.get("lambda", np.nan)),
                "table9_fraction_lambda": float(t9.get("lambda_fraction", np.nan)),
                "table9_total_lambda": float(t9.get("lambda_total", np.nan)),
            }
        )
    return rows


def _plot(rows: List[Dict[str, object]], out_png: Path) -> None:
    if not rows:
        return
    modes = sorted(set(str(r["mode"]) for r in rows))
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    colors = {"table9": "#1f77b4", "pdg": "#d62728"}
    for mode in modes:
        rr = sorted([r for r in rows if str(r["mode"]) == mode], key=lambda x: float(x["fraction_lambda"]))
        if not rr:
            continue
        x = np.asarray([float(r["fraction_lambda"]) for r in rr], dtype=float)
        x_plot = np.maximum(x, 1e-12)
        c = colors.get(mode, None)
        ax1.plot(x_plot, [float(r["full_log_pull_rms"]) for r in rr], marker="o", label=mode, color=c)
        ax2.plot(x_plot, [float(r["full_weighted_rel_rmse"]) for r in rr], marker="o", label=mode, color=c)
        ax3.plot(x_plot, [float(r["table9_fraction_l2_per_row"]) for r in rr], marker="o", label=mode, color=c)
        ax4.plot(x_plot, [float(r["table9_active_rows"]) for r in rr], marker="o", label=mode, color=c)
    for ax in (ax1, ax2, ax3, ax4):
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(r"$\lambda_{\mathrm{T9,frac}}$")
    ax1.set_ylabel("log-pull RMS")
    ax1.set_title("Global Fit Quality")
    ax2.set_ylabel("WRRMSE")
    ax2.set_title("Global Relative Error")
    ax3.set_ylabel("Table-9 fraction L2 / row")
    ax3.set_title("Table-9 Fraction Mismatch")
    ax4.set_ylabel("Active Table-9 selectors")
    ax4.set_title("Nonzero Table-9 Occupancy")
    ax1.legend(loc="best")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-stamp", required=True)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    rows = _collect(str(args.run_stamp))
    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else (RUNS_ROOT / f"table9_fraction_scan_{args.run_stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "table9_fraction_scan_metrics.csv"
    fields = [
        "run_name",
        "mode",
        "fraction_lambda",
        "full_log_pull_rms",
        "full_weighted_rel_rmse",
        "abs_yield_chi2_per_row",
        "table9_fraction_l2_per_row",
        "table9_fit_total_pi_minus",
        "table9_active_rows",
        "eta_fit_fraction",
        "table9_lambda",
        "table9_fraction_lambda",
        "table9_total_lambda",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for r in sorted(rows, key=lambda x: (str(x["mode"]), float(x["fraction_lambda"]))):
            wr.writerow({k: r.get(k) for k in fields})

    plot_png = out_dir / "table9_fraction_scan_metrics.png"
    _plot(rows, plot_png)

    md_lines: List[str] = []
    md_lines.append("# Table-9 Fraction-Prior Scan Summary")
    md_lines.append("")
    md_lines.append(f"- stamp: `{args.run_stamp}`")
    md_lines.append(f"- scenarios found: `{len(rows)}`")
    md_lines.append(f"- csv: `{csv_path}`")
    md_lines.append(f"- figure: `{plot_png}`")
    md_lines.append("")
    md_lines.append("| mode | frac lambda | log-pull RMS | WRRMSE | abs chi2/row | T9 frac L2/row | T9 active rows | eta fit fraction |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda x: (str(x["mode"]), float(x["fraction_lambda"]))):
        md_lines.append(
            f"| {r['mode']} | {float(r['fraction_lambda']):.6g} | {float(r['full_log_pull_rms']):.6g} | "
            + f"{float(r['full_weighted_rel_rmse']):.6g} | {float(r['abs_yield_chi2_per_row']):.6g} | "
            + f"{float(r['table9_fraction_l2_per_row']):.6g} | {int(r['table9_active_rows'])} | {float(r['eta_fit_fraction']):.6g} |"
        )
    md_path = out_dir / "TABLE9_FRACTION_SCAN_SUMMARY.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {plot_png}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()

