#!/usr/bin/env python3
"""Plot multispecies seagull profiles from constrained benchmark outputs."""

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


MODEL_ORDER: List[str] = [
    "Data",
    "Resonance_shared",
    "PYTHIA8_external",
    "EPOS_external",
    "UrQMD_external",
]
MODEL_LABEL: Dict[str, str] = {
    "Data": "NA49 data",
    "Resonance_shared": "na49_mixing (shared)",
    "PYTHIA8_external": "PYTHIA8",
    "EPOS_external": "EPOS",
    "UrQMD_external": "UrQMD",
}
MODEL_COLOR: Dict[str, str] = {
    "Data": "#111111",
    "Resonance_shared": "#1b9e77",
    "PYTHIA8_external": "#d95f02",
    "EPOS_external": "#7570b3",
    "UrQMD_external": "#e7298a",
}


def _load_profiles(profiles_csv: Path, split: str) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    rows: Dict[str, Dict[str, List[float]]] = {}
    with profiles_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        has_split = "split" in (reader.fieldnames or [])
        has_err = "pt_mean_err" in (reader.fieldnames or [])
        for row in reader:
            if has_split and str(row.get("split", "")) != str(split):
                continue
            species = str(row.get("species", "")).strip()
            model = str(row.get("model", "")).strip()
            if (not species) or (not model):
                continue
            xf = float(row["xf"])
            pt_mean = float(row["pt_mean"])
            if has_err:
                try:
                    pt_err = float(row.get("pt_mean_err", "nan"))
                except ValueError:
                    pt_err = float("nan")
            else:
                pt_err = float("nan")
            if species not in rows:
                rows[species] = {}
            if model not in rows[species]:
                rows[species][model] = {"xf": [], "pt_mean": [], "pt_mean_err": []}
            rows[species][model]["xf"].append(xf)
            rows[species][model]["pt_mean"].append(pt_mean)
            rows[species][model]["pt_mean_err"].append(pt_err)

    out: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for species, by_model in rows.items():
        out[species] = {}
        for model, vals in by_model.items():
            x = np.asarray(vals["xf"], dtype=float)
            y = np.asarray(vals["pt_mean"], dtype=float)
            e = np.asarray(vals["pt_mean_err"], dtype=float)
            order = np.argsort(x)
            out[species][model] = {
                "xf": x[order],
                "pt_mean": y[order],
                "pt_mean_err": e[order],
            }
    return out


def _fit_linear_slope(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2:
        return None
    p = np.polyfit(x, y, 1)
    return float(p[0])


def _write_slope_summary_csv(
    *,
    profiles: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    min_xf: float,
    out_csv: Path,
) -> Dict[str, Dict[str, float | None]]:
    slope_map: Dict[str, Dict[str, float | None]] = {}
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["species", "model", "xf_min", "n_points", "slope"])
        for species in sorted(profiles.keys()):
            slope_map[species] = {}
            for model in MODEL_ORDER:
                arr = profiles.get(species, {}).get(model)
                if arr is None:
                    slope_map[species][model] = None
                    writer.writerow([species, model, min_xf, 0, ""])
                    continue
                x = np.asarray(arr.get("xf", np.zeros(0, dtype=float)), dtype=float)
                y = np.asarray(arr.get("pt_mean", np.zeros(0, dtype=float)), dtype=float)
                m = x >= float(min_xf)
                x = x[m]
                y = y[m]
                slope = _fit_linear_slope(x, y)
                slope_map[species][model] = slope
                writer.writerow([species, model, min_xf, int(x.size), slope if slope is not None else ""])
    return slope_map


def _species_sort_key(name: str) -> tuple[int, str]:
    order = {"pi-": 0, "pi+": 1, "K+": 2, "K-": 3, "p": 4, "pbar": 5}
    return (order.get(name, 99), name)


def _plot_profile_grid(
    *,
    profiles: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    min_xf: float,
    out_png: Path,
) -> None:
    species_names = sorted(profiles.keys(), key=_species_sort_key)
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12.8, 7.6), sharex=True, sharey=True)

    y_min = np.inf
    y_max = -np.inf
    for species in species_names:
        for model in MODEL_ORDER:
            arr = profiles.get(species, {}).get(model)
            if arr is None:
                continue
            x_all = np.asarray(arr.get("xf", np.zeros(0, dtype=float)), dtype=float)
            y_all = np.asarray(arr.get("pt_mean", np.zeros(0, dtype=float)), dtype=float)
            m = x_all >= float(min_xf)
            if not np.any(m):
                continue
            y_min = min(y_min, float(np.min(y_all[m])))
            y_max = max(y_max, float(np.max(y_all[m])))
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        y_min, y_max = 0.1, 1.0
    pad = 0.10 * (y_max - y_min if y_max > y_min else 1.0)
    y_lo = max(0.0, y_min - pad)
    y_hi = y_max + pad

    axes_flat = axes.ravel()
    for i, species in enumerate(species_names):
        ax = axes_flat[i]
        for model in MODEL_ORDER:
            arr = profiles.get(species, {}).get(model)
            if arr is None:
                continue
            x_all = np.asarray(arr.get("xf", np.zeros(0, dtype=float)), dtype=float)
            y_all = np.asarray(arr.get("pt_mean", np.zeros(0, dtype=float)), dtype=float)
            e_all = np.asarray(arr.get("pt_mean_err", np.zeros(0, dtype=float)), dtype=float)
            m = x_all >= float(min_xf)
            if not np.any(m):
                continue
            x = x_all[m]
            y = y_all[m]
            e = e_all[m]
            has_err = np.any(np.isfinite(e) & (e > 0.0))
            if model == "Data":
                ax.errorbar(
                    x,
                    y,
                    yerr=e if has_err else None,
                    fmt="o-",
                    color=MODEL_COLOR[model],
                    lw=1.2,
                    ms=3.6,
                    mfc="white",
                    mec=MODEL_COLOR[model],
                    capsize=2.0 if has_err else 0.0,
                    label=MODEL_LABEL[model],
                )
            else:
                ax.plot(
                    x,
                    y,
                    "o-",
                    color=MODEL_COLOR[model],
                    lw=1.4,
                    ms=3.2,
                    label=MODEL_LABEL[model],
                )
                if has_err:
                    ylo = y - e
                    yhi = y + e
                    ax.fill_between(x, ylo, yhi, color=MODEL_COLOR[model], alpha=0.14, linewidth=0.0)
        ax.set_title(species, fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(float(min_xf) - 0.01, 0.405)
        ax.set_ylim(y_lo, y_hi)
        if i % n_cols == 0:
            ax.set_ylabel(r"$\langle p_T\rangle$ [GeV/$c$]")
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel(r"$x_F$")
        ax.tick_params(labelsize=8)

    for j in range(len(species_names), n_rows * n_cols):
        axes_flat[j].axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(
        rf"Seagull profiles: $\langle p_T\rangle(x_F)$ in acceptance-safe region $x_F \geq {min_xf:.2f}$",
        y=0.995,
        fontsize=11,
    )
    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.93))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _plot_slope_bars(
    *,
    slope_map: Dict[str, Dict[str, float | None]],
    out_png: Path,
    min_xf: float,
) -> None:
    species_names = sorted(slope_map.keys(), key=_species_sort_key)
    x = np.arange(len(species_names), dtype=float)
    width = 0.16
    offsets = {
        "Data": -2.0 * width,
        "Resonance_shared": -1.0 * width,
        "PYTHIA8_external": 0.0,
        "EPOS_external": 1.0 * width,
        "UrQMD_external": 2.0 * width,
    }

    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    for model in MODEL_ORDER:
        vals = []
        for sp in species_names:
            v = slope_map.get(sp, {}).get(model)
            vals.append(np.nan if v is None else float(v))
        ax.bar(x + offsets[model], vals, width=width, color=MODEL_COLOR[model], alpha=0.9, label=MODEL_LABEL[model])

    ax.axhline(0.0, color="0.35", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(species_names, fontsize=9)
    ax.set_ylabel(r"linear slope $d\langle p_T\rangle/dx_F$")
    ax.set_title(rf"Seagull slope summary in $x_F \geq {min_xf:.2f}$")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot seagull profiles from constrained benchmark outputs.")
    parser.add_argument("--run-description", default="sharedfit_bestmix_allmodels_gaussmle_2026-03-30_seagull")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--split", default="full", choices=["full", "train", "test"])
    parser.add_argument("--min-xf", type=float, default=0.10)
    parser.add_argument("--profiles-csv", default="")
    parser.add_argument("--out-grid-png", default="")
    parser.add_argument("--out-slope-png", default="")
    parser.add_argument("--out-slope-csv", default="")
    parser.add_argument(
        "--with-slope-summary",
        action="store_true",
        help="Also write slope CSV and slope-bar figure. Disabled by default.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if str(args.run_dir).strip() else (RUNS_ROOT / args.run_description)
    profiles_csv = Path(args.profiles_csv) if str(args.profiles_csv).strip() else (
        run_dir / "multispecies_constrained_seagull_profiles.csv"
    )
    out_grid_png = Path(args.out_grid_png) if str(args.out_grid_png).strip() else (
        run_dir / "figures" / "seagull_xf_meanpt_all6_species_grid.png"
    )
    out_slope_png = Path(args.out_slope_png) if str(args.out_slope_png).strip() else (
        run_dir / "figures" / "seagull_xf_slope_all6_species.png"
    )
    out_slope_csv = Path(args.out_slope_csv) if str(args.out_slope_csv).strip() else (
        run_dir / "seagull_slope_summary_xfmin.csv"
    )

    profiles = _load_profiles(profiles_csv, split=str(args.split))
    if not profiles:
        raise SystemExit(f"No rows found for split='{args.split}' in {profiles_csv}")

    _plot_profile_grid(profiles=profiles, min_xf=float(args.min_xf), out_png=out_grid_png)
    if bool(args.with_slope_summary):
        slope_map = _write_slope_summary_csv(profiles=profiles, min_xf=float(args.min_xf), out_csv=out_slope_csv)
        _plot_slope_bars(slope_map=slope_map, out_png=out_slope_png, min_xf=float(args.min_xf))

    manifest = {
        "run_dir": str(run_dir),
        "split": str(args.split),
        "min_xf": float(args.min_xf),
        "profiles_csv": str(profiles_csv),
        "out_grid_png": str(out_grid_png),
        "with_slope_summary": bool(args.with_slope_summary),
    }
    if bool(args.with_slope_summary):
        manifest["out_slope_png"] = str(out_slope_png)
        manifest["out_slope_csv"] = str(out_slope_csv)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
