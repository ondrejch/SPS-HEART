#!/usr/bin/env python3
"""Profile-likelihood summary over N* tower settings from ladder result CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _shape_sort_key(shape: str) -> Tuple[int, float]:
    s = str(shape)
    if s == "linear":
        return (0, 0.0)
    if s == "regge_m2":
        return (1, 0.0)
    if s.startswith("hagedorn_t"):
        try:
            return (2, float(s.replace("hagedorn_t", "")) / 1000.0)
        except Exception:
            return (2, 0.0)
    return (3, 0.0)


def _coerce_numeric(x: object) -> object:
    try:
        xf = float(x)
        if np.isfinite(xf):
            return xf
    except Exception:
        pass
    return str(x)


def _load_rows(csv_path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for rr in csv.DictReader(fh):
            sj = Path(str(rr.get("summary_json", ""))).resolve()
            if not sj.exists():
                continue
            payload = json.loads(sj.read_text(encoding="utf-8"))
            gm = payload.get("resonance", {}).get("global_metrics", {})
            logchi2 = float(gm.get("full_log_chi2", np.nan))
            if not np.isfinite(logchi2):
                continue
            rows.append(
                {
                    "scenario": str(rr.get("scenario", "")),
                    "tower_max_mass": float(rr.get("tower_max_mass", np.nan)),
                    "level_count": int(float(rr.get("level_count", np.nan))),
                    "shape_tag": str(rr.get("shape_tag", "")),
                    "full_log_chi2": logchi2,
                    "nll": 0.5 * logchi2,
                }
            )
    if not rows:
        raise RuntimeError(f"No usable rows in {csv_path}")
    return rows


def _profile_one(rows: Sequence[Dict[str, object]], param: str) -> List[Dict[str, object]]:
    nll_global_min = min(float(r["nll"]) for r in rows)
    values = sorted({r[param] for r in rows}, key=_shape_sort_key if param == "shape_tag" else None)  # type: ignore[arg-type]
    prof: List[Dict[str, object]] = []
    for v in values:
        nll_v = min(float(r["nll"]) for r in rows if r[param] == v)
        prof.append(
            {
                "param": param,
                "value": v,
                "nll_min_at_value": nll_v,
                "delta_nll": float(nll_v - nll_global_min),
            }
        )
    if not prof:
        return []
    rel = np.exp(-np.asarray([float(x["delta_nll"]) for x in prof], dtype=float))
    norm = float(np.sum(rel))
    if norm <= 0.0 or not np.isfinite(norm):
        weights = np.zeros_like(rel)
    else:
        weights = rel / norm
    csum = np.cumsum(weights)
    for i, item in enumerate(prof):
        item["profile_weight"] = float(weights[i])
        item["profile_cumulative_weight"] = float(csum[i])
        item["within_delta_nll_0p5"] = bool(float(item["delta_nll"]) <= 0.5)
        item["within_delta_nll_2p0"] = bool(float(item["delta_nll"]) <= 2.0)
    return prof


def _write_profile_csv(path: Path, run_label: str, profile_rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "run_label",
                "param",
                "value",
                "nll_min_at_value",
                "delta_nll",
                "profile_weight",
                "profile_cumulative_weight",
                "within_delta_nll_0p5",
                "within_delta_nll_2p0",
            ]
        )
        for rr in profile_rows:
            w.writerow(
                [
                    run_label,
                    rr.get("param"),
                    rr.get("value"),
                    rr.get("nll_min_at_value"),
                    rr.get("delta_nll"),
                    rr.get("profile_weight"),
                    rr.get("profile_cumulative_weight"),
                    int(bool(rr.get("within_delta_nll_0p5", False))),
                    int(bool(rr.get("within_delta_nll_2p0", False))),
                ]
            )


def _plot_run_profile(path: Path, run_label: str, by_param: Dict[str, List[Dict[str, object]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    order = ["tower_max_mass", "level_count", "shape_tag"]
    titles = {
        "tower_max_mass": r"$u_{max}$ [GeV]",
        "level_count": "tower levels",
        "shape_tag": "shape",
    }
    for ax, p in zip(axes, order):
        rows = by_param.get(p, [])
        if not rows:
            ax.set_title(f"{titles[p]} (empty)")
            continue
        x_raw = [r["value"] for r in rows]
        y = [float(r["delta_nll"]) for r in rows]
        if p == "shape_tag":
            x = np.arange(len(x_raw))
            ax.bar(x, y, color="#2c7fb8")
            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in x_raw], rotation=25, ha="right", fontsize=8)
        else:
            x = np.asarray([float(v) for v in x_raw], dtype=float)
            ax.plot(x, y, marker="o", lw=1.8, color="#1d3557")
            ax.set_xticks(x)
        ax.axhline(0.5, ls="--", lw=1.0, color="#e76f51", alpha=0.8)
        ax.axhline(2.0, ls="--", lw=1.0, color="#f4a261", alpha=0.8)
        ax.set_title(titles[p])
        ax.set_ylabel(r"$\Delta$NLL")
        ax.grid(alpha=0.25, lw=0.6)
    fig.suptitle(f"Profile likelihood by tower parameter: {run_label}", fontsize=12)
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.93))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", action="append", required=True, help="Ladder summary CSV (repeatable).")
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Run label (repeatable, aligns with --csv order). Default: CSV stem.",
    )
    parser.add_argument(
        "--out-dir",
        default="dev/00runs/profile_likelihood_2026-03-31",
        help="Output directory for profile tables/plots.",
    )
    args = parser.parse_args()

    csv_paths = [Path(x).resolve() for x in args.csv]
    labels = list(args.label)
    while len(labels) < len(csv_paths):
        labels.append(csv_paths[len(labels)].stem)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    md_lines: List[str] = ["# Profile-Likelihood Over Tower Settings", ""]
    for csv_path, label in zip(csv_paths, labels):
        rows = _load_rows(csv_path)
        by_param = {
            "tower_max_mass": _profile_one(rows, "tower_max_mass"),
            "level_count": _profile_one(rows, "level_count"),
            "shape_tag": _profile_one(rows, "shape_tag"),
        }
        flat_rows: List[Dict[str, object]] = []
        for rr in by_param.values():
            flat_rows.extend(rr)
        profile_csv = out_dir / f"profile_{label.replace(' ', '_')}.csv"
        _write_profile_csv(profile_csv, label, flat_rows)
        profile_png = out_dir / f"profile_{label.replace(' ', '_')}.png"
        _plot_run_profile(profile_png, label, by_param)

        best = min(rows, key=lambda r: float(r["nll"]))
        md_lines.append(f"## {label}")
        md_lines.append("")
        md_lines.append(f"- input: `{csv_path}`")
        md_lines.append(f"- best scenario: `{best['scenario']}`")
        md_lines.append(f"- profile csv: `{profile_csv}`")
        md_lines.append(f"- profile plot: `{profile_png}`")
        md_lines.append("")
        md_lines.append("| param | value | deltaNLL | profile weight |")
        md_lines.append("|---|---:|---:|---:|")
        for p in ("tower_max_mass", "level_count", "shape_tag"):
            prof = by_param.get(p, [])
            if not prof:
                continue
            best_p = min(prof, key=lambda x: float(x["delta_nll"]))
            md_lines.append(
                f"| {p} | {best_p['value']} | {float(best_p['delta_nll']):.6g} | "
                f"{float(best_p['profile_weight']):.6g} |"
            )
        md_lines.append("")

    md_path = out_dir / "PROFILE_LIKELIHOOD_SUMMARY.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
