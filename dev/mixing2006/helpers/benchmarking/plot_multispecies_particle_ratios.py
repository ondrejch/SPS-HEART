#!/usr/bin/env python3
"""Plot integrated particle-ratio benchmark results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "00runs"


# Policy: external generators carry one fitted normalization per species,
# so their cross-species ratios are not generator yield predictions; they
# are fenced out of particle-ratio artifacts (see
# postprocess_multispecies_windows.MODEL_ORDER) and are not plotted here,
# even when an older ratio CSV still carries their columns.
MODEL_ORDER: List[str] = ["data", "resonance_shared"]
MODEL_LABEL: Dict[str, str] = {
    "data": "NA49 data",
    "resonance_shared": "na49_mixing (shared)",
}
MODEL_COLOR: Dict[str, str] = {
    "data": "#111111",
    "resonance_shared": "#1b9e77",
}


def _window_sort_key(name: str) -> Tuple[float, float, str]:
    if not name.startswith("xf_"):
        return (999.0, 999.0, name)
    parts = name.split("_")
    if len(parts) < 3:
        return (999.0, 999.0, name)
    try:
        return (float(parts[1]), float(parts[2]), name)
    except ValueError:
        return (999.0, 999.0, name)


def _load_ratio_rows(path: Path) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    windows: Dict[str, Dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        has_window = "window" in (reader.fieldnames or [])
        has_xf_min = "xf_min" in (reader.fieldnames or [])
        has_xf_max = "xf_max" in (reader.fieldnames or [])
        for row in reader:
            split = str(row.get("window", "full")).strip() if has_window else "full"
            ratio = str(row.get("ratio", "")).strip()
            if (not split) or (not ratio):
                continue
            if split not in out:
                out[split] = {}
            if split not in windows:
                windows[split] = {}
            if has_xf_min:
                raw = row.get("xf_min", "")
                if raw is not None and str(raw).strip() != "":
                    windows[split]["xf_min"] = float(raw)
            if has_xf_max:
                raw = row.get("xf_max", "")
                if raw is not None and str(raw).strip() != "":
                    windows[split]["xf_max"] = float(raw)
            vals: Dict[str, float] = {}
            for k in MODEL_ORDER:
                raw = row.get(k, "")
                if raw is None or str(raw).strip() == "":
                    continue
                vals[k] = float(raw)
            out[split][ratio] = vals
    return out, windows


def _ratio_sort_key(name: str) -> tuple[int, str]:
    order = {
        "pi+/pi-": 0,
        "k+/k-": 1,
        "pbar/p": 2,
        "k++k-/pi++pi-": 3,
    }
    return (order.get(name.lower(), 99), name)


def _plot_ratio_panel(
    *,
    ax: plt.Axes,
    ratio_map: Dict[str, Dict[str, float]],
    split_label: str,
) -> None:
    ratios = sorted(ratio_map.keys(), key=_ratio_sort_key)
    x = np.arange(len(ratios), dtype=float)
    width = 0.16
    # Bars centered on each tick for whatever model set remains.
    offsets = {
        model: (i - (len(MODEL_ORDER) - 1) / 2.0) * width
        for i, model in enumerate(MODEL_ORDER)
    }

    for model in MODEL_ORDER:
        vals = []
        for r in ratios:
            v = ratio_map.get(r, {}).get(model)
            vals.append(np.nan if v is None else float(v))
        if model == "data":
            ax.bar(
                x + offsets[model],
                vals,
                width=width,
                color=MODEL_COLOR[model],
                alpha=0.95,
                edgecolor="white",
                linewidth=0.7,
                label=MODEL_LABEL[model],
            )
        else:
            ax.bar(
                x + offsets[model],
                vals,
                width=width,
                color=MODEL_COLOR[model],
                alpha=0.90,
                label=MODEL_LABEL[model],
            )

    ax.set_xticks(x)
    ax.set_xticklabels(ratios, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("integrated ratio")
    ax.set_title(split_label)
    ax.grid(True, axis="y", alpha=0.25)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot integrated particle-ratio benchmark outputs.")
    parser.add_argument("--run-description", default="sharedfit_bestmix_allmodels_gaussmle_2026-03-30_seagull")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--ratio-csv", default="")
    parser.add_argument("--out-png", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if str(args.run_dir).strip() else (RUNS_ROOT / args.run_description)
    ratio_csv = Path(args.ratio_csv) if str(args.ratio_csv).strip() else (
        run_dir / "multispecies_constrained_particle_ratios.csv"
    )
    out_png = Path(args.out_png) if str(args.out_png).strip() else (
        run_dir / "figures" / "particle_ratio_comparison_full_test.png"
    )

    rows, windows = _load_ratio_rows(ratio_csv)
    if not rows:
        raise SystemExit(f"No rows found in {ratio_csv}")

    window_names = sorted(rows.keys(), key=_window_sort_key)
    n_panels = len(window_names)
    n_cols = min(3, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 4.6 * n_rows), sharey=False)
    axes_flat = np.asarray(axes).ravel()
    for i, win in enumerate(window_names):
        ratio_map = rows.get(win, {})
        meta = windows.get(win, {})
        if ("xf_min" in meta) and ("xf_max" in meta):
            title = rf"$x_F\in[{float(meta['xf_min']):.2f},{float(meta['xf_max']):.2f}]$"
        else:
            title = win
        _plot_ratio_panel(ax=axes_flat[i], ratio_map=ratio_map, split_label=title)
    for j in range(n_panels, axes_flat.size):
        axes_flat[j].axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(MODEL_ORDER),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Integrated particle-ratio benchmark by xF window", y=0.995, fontsize=11)
    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.95))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    manifest = {
        "run_dir": str(run_dir),
        "ratio_csv": str(ratio_csv),
        "out_png": str(out_png),
        "windows_available": window_names,
    }
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
