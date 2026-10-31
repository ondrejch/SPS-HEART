#!/usr/bin/env python3
"""Summarize N* ladder grid results and select the best scenario."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


SCENARIO_RE = re.compile(
    r"^umax(?P<umax_mev>\d+)_(?P<shape_tag>linear|regge_m2|hagedorn_t\d+)_n(?P<levels>\d+)$"
)


def _metric_pick(metrics: Dict[str, object], *keys: str) -> float:
    for k in keys:
        if k in metrics:
            return float(metrics[k])
    raise KeyError(f"Missing metric keys: {keys}")


@dataclass
class ScenarioRow:
    scenario: str
    scenario_dir: Path
    summary_json: Path
    details_npz: Path
    tower_max_mass: float
    shape_tag: str
    level_count: int
    global_wrrmse: float
    global_log_pull_rms: float
    n_active: int
    by_species: Dict[str, Dict[str, float]]


def _load_expected_count(run_dir: Path) -> int | None:
    manifests = sorted(run_dir.glob("gateway_*manifest.json"))
    if not manifests:
        return None
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    scan = payload.get("scan", {})
    umax = [x.strip() for x in str(scan.get("tower_max_values", "")).split(",") if x.strip()]
    levels = [x.strip() for x in str(scan.get("level_counts", "")).split(",") if x.strip()]
    shapes = [x.strip() for x in str(scan.get("shapes", "")).split(",") if x.strip()]
    if not (umax and levels and shapes):
        return None
    return int(len(umax) * len(levels) * len(shapes))


def _collect_rows(run_dir: Path) -> List[ScenarioRow]:
    rows: List[ScenarioRow] = []
    for summary_json in sorted(run_dir.glob("*/multispecies_constrained_summary.json")):
        scenario_dir = summary_json.parent
        scenario = scenario_dir.name
        m = SCENARIO_RE.match(scenario)
        if not m:
            continue
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
        g = payload["resonance"]["global_metrics"]
        global_wrrmse = _metric_pick(g, "full_weighted_rel_rmse", "test_weighted_rel_rmse")
        global_log_pull_rms = _metric_pick(g, "full_log_pull_rms", "test_log_pull_rms")
        per_species = payload.get("per_species", {})
        by_species: Dict[str, Dict[str, float]] = {}
        for sp_key, by_model in per_species.items():
            out: Dict[str, float] = {}
            for model_key, vals in by_model.items():
                if not isinstance(vals, dict):
                    continue
                if "full_weighted_rel_rmse" in vals:
                    out[model_key] = float(vals["full_weighted_rel_rmse"])
                elif "test_weighted_rel_rmse" in vals:
                    out[model_key] = float(vals["test_weighted_rel_rmse"])
            if out:
                by_species[sp_key] = out

        details_npz = scenario_dir / "multispecies_constrained_details.npz"
        rows.append(
            ScenarioRow(
                scenario=scenario,
                scenario_dir=scenario_dir,
                summary_json=summary_json,
                details_npz=details_npz,
                tower_max_mass=float(int(m.group("umax_mev")) / 1000.0),
                shape_tag=str(m.group("shape_tag")),
                level_count=int(m.group("levels")),
                global_wrrmse=float(global_wrrmse),
                global_log_pull_rms=float(global_log_pull_rms),
                n_active=int(payload["resonance"]["n_active"]),
                by_species=by_species,
            )
        )
    rows.sort(key=lambda r: r.global_wrrmse)
    return rows


def _write_csv(rows: List[ScenarioRow], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "rank",
                "scenario",
                "tower_max_mass",
                "shape_tag",
                "level_count",
                "global_wrrmse",
                "global_log_pull_rms",
                "n_active",
                "summary_json",
                "details_npz",
            ]
        )
        for i, r in enumerate(rows, start=1):
            w.writerow(
                [
                    i,
                    r.scenario,
                    f"{r.tower_max_mass:.3f}",
                    r.shape_tag,
                    r.level_count,
                    f"{r.global_wrrmse:.12g}",
                    f"{r.global_log_pull_rms:.12g}",
                    r.n_active,
                    str(r.summary_json),
                    str(r.details_npz),
                ]
            )


def _write_md(rows: List[ScenarioRow], out_md: Path, expected: int | None) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# N* ladder result summary")
    lines.append("")
    if expected is None:
        lines.append(f"- scenarios found: `{len(rows)}`")
    else:
        frac = 100.0 * float(len(rows)) / float(max(expected, 1))
        lines.append(f"- scenarios found: `{len(rows)}/{expected}` ({frac:.1f}%)")
    lines.append("")
    lines.append("| rank | scenario | WRRMSE | log-pull RMS | n_active |")
    lines.append("|---:|---|---:|---:|---:|")
    for i, r in enumerate(rows[:20], start=1):
        lines.append(
            "| "
            + f"{i} | {r.scenario} | {r.global_wrrmse:.6f} | {r.global_log_pull_rms:.3f} | {r.n_active} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_best_json(rows: List[ScenarioRow], out_json: Path) -> None:
    best = rows[0]
    payload = {
        "scenario": best.scenario,
        "scenario_dir": str(best.scenario_dir),
        "summary_json": str(best.summary_json),
        "details_npz": str(best.details_npz),
        "tower_max_mass": best.tower_max_mass,
        "shape_tag": best.shape_tag,
        "level_count": best.level_count,
        "global_wrrmse": best.global_wrrmse,
        "global_log_pull_rms": best.global_log_pull_rms,
        "n_active": best.n_active,
        "per_species_model_wrrmse": best.by_species,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Grid run directory.")
    parser.add_argument("--out-prefix", default="nstar_ladder_summary")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    rows = _collect_rows(run_dir)
    if not rows:
        raise SystemExit(f"No scenario summaries found under {run_dir}")

    expected = _load_expected_count(run_dir)
    out_csv = run_dir / f"{args.out_prefix}.csv"
    out_md = run_dir / f"{args.out_prefix}.md"
    out_best = run_dir / f"{args.out_prefix}_best.json"
    _write_csv(rows, out_csv)
    _write_md(rows, out_md, expected)
    _write_best_json(rows, out_best)

    print(f"wrote {out_csv}")
    print(f"wrote {out_md}")
    print(f"wrote {out_best}")


if __name__ == "__main__":
    main()
