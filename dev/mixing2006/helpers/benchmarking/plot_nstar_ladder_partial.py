#!/usr/bin/env python3
"""Plot partial/complete N* ladder scan results from available scenario summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


SCENARIO_RE = re.compile(
    r"^umax(?P<umax_mev>\d+)_(?P<shape_tag>linear|regge_m2|hagedorn_t\d+)_n(?P<levels>\d+)$"
)


def _shape_label(shape_tag: str) -> str:
    if shape_tag == "linear":
        return "linear"
    if shape_tag == "regge_m2":
        return "regge m^2"
    if shape_tag.startswith("hagedorn_t"):
        suffix = shape_tag.split("hagedorn_t", 1)[1]
        try:
            t_val = float(suffix) / 1000.0
            return f"hagedorn T={t_val:.3f}"
        except ValueError:
            return shape_tag
    return shape_tag


def _load_rows(run_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted(run_root.glob("*/multispecies_constrained_summary.json")):
        scenario = path.parent.name
        match = SCENARIO_RE.match(scenario)
        if not match:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        global_metrics = payload["resonance"]["global_metrics"]
        wrrmse = global_metrics.get("full_weighted_rel_rmse")
        if wrrmse is None:
            wrrmse = global_metrics.get("test_weighted_rel_rmse")
        log_pull = global_metrics.get("full_log_pull_rms")
        if log_pull is None:
            log_pull = global_metrics.get("test_log_pull_rms")
        if wrrmse is None or log_pull is None:
            continue
        row: Dict[str, object] = {
            "scenario": scenario,
            "path": str(path),
            "tower_max_mass": int(match.group("umax_mev")) / 1000.0,
            "shape_tag": match.group("shape_tag"),
            "level_count": int(match.group("levels")),
            "global_weighted_rel_rmse": float(wrrmse),
            "global_log_pull_rms": float(log_pull),
            "n_active": int(payload["resonance"]["n_active"]),
        }
        rows.append(row)
    return rows


def _load_total_expected(run_root: Path) -> Optional[int]:
    manifests = sorted(run_root.glob("gateway_*manifest.json"))
    if not manifests:
        return None
    try:
        manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        scan = manifest.get("scan", {})
        umax = [x.strip() for x in str(scan.get("tower_max_values", "")).split(",") if x.strip()]
        levels = [x.strip() for x in str(scan.get("level_counts", "")).split(",") if x.strip()]
        shapes = [x.strip() for x in str(scan.get("shapes", "")).split(",") if x.strip()]
        if umax and levels and shapes:
            return len(umax) * len(levels) * len(shapes)
    except Exception:
        return None
    return None


def _write_csv(rows: List[Dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "tower_max_mass",
        "shape_tag",
        "level_count",
        "global_weighted_rel_rmse",
        "global_log_pull_rms",
        "n_active",
        "path",
    ]
    rows_sorted = sorted(rows, key=lambda r: float(r["global_weighted_rel_rmse"]))
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow(row)


def _write_md(
    rows: List[Dict[str, object]],
    out_md: Path,
    *,
    total_expected: Optional[int],
) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda r: float(r["global_weighted_rel_rmse"]))
    lines: List[str] = []
    lines.append("# N* Ladder Partial Results")
    lines.append("")
    if total_expected is None:
        lines.append(f"- completed scenarios: `{len(rows_sorted)}`")
    else:
        frac = 100.0 * float(len(rows_sorted)) / float(max(total_expected, 1))
        lines.append(
            f"- completed scenarios: `{len(rows_sorted)}/{int(total_expected)}` ({frac:.1f}%)"
        )
    lines.append("")
    lines.append("| rank | scenario | WRRMSE | log-pull RMS | n_active |")
    lines.append("|---:|---|---:|---:|---:|")
    for idx, row in enumerate(rows_sorted[:20], start=1):
        lines.append(
            "| "
            + f"{idx} | {row['scenario']} | {float(row['global_weighted_rel_rmse']):.6f} | "
            + f"{float(row['global_log_pull_rms']):.3f} | {int(row['n_active'])} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_heatmap(
    rows: List[Dict[str, object]],
    out_png: Path,
    *,
    total_expected: Optional[int],
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No completed scenario summaries found.")

    shape_tags = sorted({str(r["shape_tag"]) for r in rows}, key=lambda s: (s.startswith("hagedorn"), s))
    # Keep linear/regge first, then hagedorn in increasing T order.
    def _shape_sort_key(tag: str) -> Tuple[int, float, str]:
        if tag == "linear":
            return (0, 0.0, tag)
        if tag == "regge_m2":
            return (1, 0.0, tag)
        if tag.startswith("hagedorn_t"):
            try:
                return (2, float(tag.split("hagedorn_t", 1)[1]), tag)
            except ValueError:
                return (2, math.inf, tag)
        return (3, math.inf, tag)

    shape_tags = sorted(shape_tags, key=_shape_sort_key)
    level_counts = sorted({int(r["level_count"]) for r in rows})
    umax_values = sorted({float(r["tower_max_mass"]) for r in rows})

    min_metric = min(float(r["global_weighted_rel_rmse"]) for r in rows)
    max_metric = max(float(r["global_weighted_rel_rmse"]) for r in rows)

    ncols = len(shape_tags)
    fig, axes = plt.subplots(1, ncols, figsize=(3.6 * ncols + 1.8, 5.0), sharey=True)
    if ncols == 1:
        axes = [axes]

    im = None
    for ax, shape_tag in zip(axes, shape_tags):
        grid = np.full((len(umax_values), len(level_counts)), np.nan, dtype=float)
        for row in rows:
            if str(row["shape_tag"]) != shape_tag:
                continue
            i = umax_values.index(float(row["tower_max_mass"]))
            j = level_counts.index(int(row["level_count"]))
            val = float(row["global_weighted_rel_rmse"])
            if np.isnan(grid[i, j]) or val < grid[i, j]:
                grid[i, j] = val

        im = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="viridis_r",
            vmin=min_metric,
            vmax=max_metric,
        )
        ax.set_title(_shape_label(shape_tag))
        ax.set_xticks(np.arange(len(level_counts)))
        ax.set_xticklabels([str(x) for x in level_counts])
        ax.set_xlabel("tower levels")
        if ax is axes[0]:
            ax.set_yticks(np.arange(len(umax_values)))
            ax.set_yticklabels([f"{x:.1f}" for x in umax_values])
            ax.set_ylabel("tower max mass [GeV]")
        else:
            ax.set_yticks(np.arange(len(umax_values)))
            ax.set_yticklabels([])

        for i in range(len(umax_values)):
            for j in range(len(level_counts)):
                if np.isnan(grid[i, j]):
                    ax.text(j, i, " ", ha="center", va="center", color="white", fontsize=8)
                else:
                    ax.text(
                        j,
                        i,
                        f"{grid[i, j]:.3f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )

    assert im is not None
    cbar = fig.colorbar(im, ax=axes, shrink=0.92, pad=0.02)
    cbar.set_label("global test WRRMSE (lower better)")

    if total_expected is None:
        title = f"N* ladder scan (partial): {len(rows)} completed scenarios"
    else:
        frac = 100.0 * float(len(rows)) / float(max(total_expected, 1))
        title = (
            "N* ladder scan (partial): "
            + f"{len(rows)}/{int(total_expected)} completed ({frac:.1f}%)"
        )
    fig.suptitle(title)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        required=True,
        help="Run directory containing scenario subdirectories.",
    )
    parser.add_argument(
        "--out-prefix",
        default="nstar_ladder_partial",
        help="Output file prefix under run-root.",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    rows = _load_rows(run_root)
    if not rows:
        raise SystemExit("No scenario summaries found in run root.")

    total_expected = _load_total_expected(run_root)
    out_csv = run_root / f"{args.out_prefix}.csv"
    out_md = run_root / f"{args.out_prefix}.md"
    out_png = run_root / f"{args.out_prefix}.png"

    _write_csv(rows, out_csv)
    _write_md(rows, out_md, total_expected=total_expected)
    _plot_heatmap(rows, out_png, total_expected=total_expected)

    print(f"wrote {out_csv}")
    print(f"wrote {out_md}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
