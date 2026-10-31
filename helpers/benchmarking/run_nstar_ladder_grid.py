#!/usr/bin/env python3
"""Run an N* ladder grid (upper bound x levels x bunching shapes)."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import helpers.benchmarking.benchmark_sps_species as bss
from helpers.benchmarking.study_nstar_bunching import (
    BESTMIX_SOFT_OVERRIDES,
    _parse_int_csv,
    _parse_shapes,
    _parse_species,
    _shape_tag,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "dev" / "00runs"


def _pick_metric(metrics: Dict[str, object], *candidates: str) -> float:
    for key in candidates:
        if key in metrics:
            return float(metrics[key])
    raise KeyError(f"none of metric keys present: {candidates}")


def _pick_species_metric(metrics: Dict[str, object], *candidates: str) -> float:
    for key in candidates:
        if key in metrics:
            return float(metrics[key])
    raise KeyError(f"none of species metric keys present: {candidates}")


def _parse_float_csv(raw: str) -> List[float]:
    vals: List[float] = []
    for part in str(raw).split(","):
        tok = part.strip()
        if not tok:
            continue
        vals.append(float(tok))
    if not vals:
        raise ValueError("No float values were provided.")
    out = sorted(set(vals))
    return out


def _scenario_name(upper_mass: float, shape: str, shape_param: float, level_count: int) -> str:
    return (
        f"umax{int(round(float(upper_mass) * 1000.0)):04d}_"
        f"{_shape_tag(shape, shape_param)}_n{int(level_count):02d}"
    )


def _soft_overrides_for_upper(*, upper_mass: float, min_mass: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, val in BESTMIX_SOFT_OVERRIDES.items():
        if key in {"nstar_eff_tower_max_mass", "nstar_eff_tower_min_mass"}:
            continue
        out[key] = float(val)
    out["nstar_eff_tower_min_mass"] = float(min_mass)
    out["nstar_eff_tower_max_mass"] = float(upper_mass)
    return out


def _run_one(
    *,
    out_root: Path,
    species_csv: str,
    upper_mass: float,
    min_mass: float,
    shape: str,
    shape_param: float,
    level_count: int,
    seed: int,
    res_events_per_res: int,
    res_workers: int,
    max_decay_passes: int,
    fit_objective: str,
    fit_weight_mode: str,
    fit_opt_maxiter: int,
    fit_opt_ftol: float,
    fit_balance_xf_bins: int,
    fit_balance_pt_bins: int,
    prior_table9_lambda: float,
    prior_table9_json: str,
    prior_strange_lambda: float,
    prior_hyperon_lambda: float,
    prior_kminus_over_kplus_target: float,
    hyperon_pdgs: str,
    skip_external: bool,
    external_run_dir: str,
    res_cache_dir: Path,
) -> Dict[str, object]:
    scenario = _scenario_name(upper_mass, shape, shape_param, level_count)
    scenario_dir = out_root / scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "helpers" / "benchmarking" / "benchmark_multispecies_constrained.py"),
        "--run-description",
        scenario,
        "--out-dir",
        str(scenario_dir),
        "--species",
        species_csv,
        "--seed",
        str(int(seed)),
        "--res-events-per-res",
        str(int(res_events_per_res)),
        "--res-max-decay-passes",
        str(int(max_decay_passes)),
        "--res-workers",
        str(int(res_workers)),
        "--res-fragmentation-mode",
        "both",
        "--res-regge-tower-min-mass",
        f"{float(min_mass):.6f}",
        "--res-regge-tower-max-mass",
        f"{float(upper_mass):.6f}",
        "--res-regge-tower-level-count",
        str(int(level_count)),
        "--res-regge-tower-shape",
        str(shape),
        "--res-regge-tower-shape-param",
        f"{float(shape_param):.12g}",
        "--fit-objective",
        str(fit_objective),
        "--fit-weight-mode",
        str(fit_weight_mode),
        "--fit-opt-maxiter",
        str(int(fit_opt_maxiter)),
        "--fit-opt-ftol",
        f"{float(fit_opt_ftol):.12g}",
        "--fit-balance-xf-bins",
        str(int(fit_balance_xf_bins)),
        "--fit-balance-pt-bins",
        str(int(fit_balance_pt_bins)),
        "--prior-table9-lambda",
        f"{float(prior_table9_lambda):.12g}",
        "--prior-table9-json",
        str(prior_table9_json),
        "--prior-strange-lambda",
        f"{float(prior_strange_lambda):.12g}",
        "--prior-hyperon-lambda",
        f"{float(prior_hyperon_lambda):.12g}",
        "--prior-kminus-over-kplus-target",
        f"{float(prior_kminus_over_kplus_target):.12g}",
        "--hyperon-pdgs",
        str(hyperon_pdgs),
        "--external-run-dir",
        str(external_run_dir),
        "--res-cache-dir",
        str(res_cache_dir),
    ]
    if bool(skip_external):
        cmd.append("--skip-external")

    for key, val in sorted(_soft_overrides_for_upper(upper_mass=upper_mass, min_mass=min_mass).items()):
        cmd.extend(["--res-soft-override", f"{key}={float(val):.12g}"])

    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    summary_path = scenario_dir / "multispecies_constrained_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    masses = bss._compute_regge_tower_mass_ladder(
        float(min_mass),
        float(upper_mass),
        level_count=int(level_count),
        shape=str(shape),
        shape_param=float(shape_param),
    )
    spacings = np.diff(masses) if masses.size > 1 else np.asarray([0.0], dtype=float)
    spacing_ratio_last_first = float(spacings[-1] / spacings[0]) if spacings[0] > 0 else float("nan")

    global_metrics = payload["resonance"]["global_metrics"]
    global_wrrmse = _pick_metric(
        global_metrics,
        "full_weighted_rel_rmse",
        "test_weighted_rel_rmse",
    )
    global_log_pull_rms = _pick_metric(
        global_metrics,
        "full_log_pull_rms",
        "test_log_pull_rms",
    )

    per_species_res: Dict[str, float] = {}
    per_species = payload.get("per_species", {})
    for sp in sorted(per_species.keys()):
        per_species_res[sp] = _pick_species_metric(
            per_species[sp]["Resonance_NNLS_shared"],
            "full_weighted_rel_rmse",
            "test_weighted_rel_rmse",
        )

    return {
        "scenario": scenario,
        "tower_max_mass": float(upper_mass),
        "tower_min_mass": float(min_mass),
        "shape": str(shape),
        "shape_param": float(shape_param),
        "level_count": int(level_count),
        "tower_masses": [float(x) for x in masses.tolist()],
        "tower_spacings": [float(x) for x in spacings.tolist()],
        "spacing_ratio_last_first": spacing_ratio_last_first,
        "global_weighted_rel_rmse": float(global_wrrmse),
        "global_log_pull_rms": float(global_log_pull_rms),
        "n_active_components": int(payload["resonance"]["n_active"]),
        "per_species_resonance_test_wrrmse": per_species_res,
        "summary_path": str(summary_path),
        "details_path": str(scenario_dir / "multispecies_constrained_details.npz"),
    }


def _write_outputs(
    *,
    out_root: Path,
    rows: List[Dict[str, object]],
    species_csv: str,
    tower_max_values: List[float],
    level_counts: List[int],
    shapes: List[Tuple[str, float]],
    shard_count: int,
    shard_index: int,
) -> None:
    json_path = out_root / f"nstar_ladder_grid_shard{int(shard_index):02d}_of_{int(shard_count):02d}.json"
    csv_path = out_root / f"nstar_ladder_grid_shard{int(shard_index):02d}_of_{int(shard_count):02d}.csv"
    md_path = out_root / f"NSTAR_LADDER_GRID_SHARD{int(shard_index):02d}_OF_{int(shard_count):02d}.md"

    payload = {
        "species": species_csv,
        "tower_max_values": [float(x) for x in tower_max_values],
        "level_counts": [int(x) for x in level_counts],
        "shapes": [{"shape": s, "shape_param": p} for s, p in shapes],
        "shard_count": int(shard_count),
        "shard_index": int(shard_index),
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    fieldnames = [
        "scenario",
        "tower_max_mass",
        "tower_min_mass",
        "shape",
        "shape_param",
        "level_count",
        "spacing_ratio_last_first",
        "global_weighted_rel_rmse",
        "global_log_pull_rms",
        "n_active_components",
        "summary_path",
        "details_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    lines: List[str] = []
    lines.append("# N* Ladder Grid Shard")
    lines.append("")
    lines.append(f"- shard: `{int(shard_index)}/{int(shard_count)}`")
    lines.append(f"- species: `{species_csv}`")
    lines.append("- tower max values [GeV]: `" + ", ".join(f"{x:.3f}" for x in tower_max_values) + "`")
    lines.append("- level counts: `" + ", ".join(str(x) for x in level_counts) + "`")
    lines.append(
        "- shapes: `"
        + ", ".join(f"{s}:{p:.3f}" if s == "hagedorn" else s for s, p in shapes)
        + "`"
    )
    lines.append("")
    lines.append("| scenario | umax [GeV] | shape | levels | global WRRMSE |")
    lines.append("|---|---:|---|---:|---:|")
    for row in rows:
        lines.append(
            "| "
            + f"{row['scenario']} | "
            + f"{float(row['tower_max_mass']):.3f} | "
            + f"{row['shape']} | "
            + f"{int(row['level_count'])} | "
            + f"{float(row['global_weighted_rel_rmse']):.6f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run N* ladder grid scenarios.")
    parser.add_argument("--run-description", default="nstar_ladder_grid")
    parser.add_argument("--out-root", default="")
    parser.add_argument("--species", default="pi-,pi+,k+,k-,p,pbar")
    parser.add_argument("--tower-min-mass", type=float, default=1.9)
    parser.add_argument("--tower-max-values", default="1.9,2.1,2.3,2.5,2.7")
    parser.add_argument("--level-counts", default="3,5,10,15")
    parser.add_argument("--shapes", default="linear,regge_m2,hagedorn:0.12,hagedorn:0.16,hagedorn:0.25")
    parser.add_argument("--seed", type=int, default=20260330)
    parser.add_argument("--res-events-per-res", type=int, default=1200000)
    parser.add_argument("--res-workers", type=int, default=64)
    parser.add_argument("--max-decay-passes", type=int, default=16)
    parser.add_argument("--fit-objective", default="nnls_weighted")
    parser.add_argument("--fit-weight-mode", default="relative_xfpt_balanced")
    parser.add_argument("--fit-opt-maxiter", type=int, default=3000)
    parser.add_argument("--fit-opt-ftol", type=float, default=1e-10)
    parser.add_argument("--fit-balance-xf-bins", type=int, default=4)
    parser.add_argument("--fit-balance-pt-bins", type=int, default=4)
    parser.add_argument("--prior-table9-lambda", type=float, default=0.0)
    parser.add_argument(
        "--prior-table9-json",
        default=str(REPO_ROOT / "helpers" / "benchmarking" / "config" / "fischer2022_table9_pi_minus.json"),
    )
    parser.add_argument("--prior-strange-lambda", type=float, default=0.0)
    parser.add_argument("--prior-hyperon-lambda", type=float, default=0.0)
    parser.add_argument("--prior-kminus-over-kplus-target", type=float, default=0.5726872246696035)
    parser.add_argument("--hyperon-pdgs", default="3122,3212,3222,3112,3312,3322,3334")
    parser.add_argument(
        "--external-run-dir",
        default=str(RUNS_ROOT / "gw_multispecies_allmodels_singlepass_pythia_2026-03-30"),
    )
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--res-cache-dir", default="")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--scenario-indices",
        default="",
        help="Optional comma-separated scenario indices to run. Overrides shard modulo selection.",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root) if str(args.out_root).strip() else (RUNS_ROOT / args.run_description)
    out_root.mkdir(parents=True, exist_ok=True)
    res_cache_dir = Path(args.res_cache_dir) if str(args.res_cache_dir).strip() else (out_root / "res_cache")
    res_cache_dir.mkdir(parents=True, exist_ok=True)

    species_csv = _parse_species(args.species)
    tower_max_values = _parse_float_csv(args.tower_max_values)
    level_counts = _parse_int_csv(args.level_counts)
    shapes = _parse_shapes(args.shapes)
    shard_count = max(int(args.shard_count), 1)
    shard_index = int(args.shard_index)
    if not (0 <= shard_index < shard_count):
        raise ValueError(f"Invalid shard index/count: index={shard_index}, count={shard_count}")

    scenarios: List[Tuple[float, str, float, int]] = []
    for umax in tower_max_values:
        for shape, shape_param in shapes:
            for nlevels in level_counts:
                scenarios.append((float(umax), str(shape), float(shape_param), int(nlevels)))

    selected_indices: List[int] = []
    if str(args.scenario_indices).strip():
        for tok in str(args.scenario_indices).split(","):
            txt = tok.strip()
            if not txt:
                continue
            idx = int(txt)
            if idx < 0 or idx >= len(scenarios):
                raise ValueError(f"scenario index out of range: {idx} not in [0,{len(scenarios)-1}]")
            selected_indices.append(idx)
        # de-duplicate while preserving order
        seen = set()
        selected_indices = [i for i in selected_indices if (i not in seen and not seen.add(i))]
    else:
        for i in range(len(scenarios)):
            if (i % shard_count) == shard_index:
                selected_indices.append(i)

    rows: List[Dict[str, object]] = []
    for i in selected_indices:
        (umax, shape, shape_param, nlevels) = scenarios[i]
        row = {
            "scenario_index": int(i),
        }
        row.update(
            _run_one(
                out_root=out_root,
                species_csv=species_csv,
                upper_mass=float(umax),
                min_mass=float(args.tower_min_mass),
                shape=str(shape),
                shape_param=float(shape_param),
                level_count=int(nlevels),
                seed=int(args.seed),
                res_events_per_res=int(args.res_events_per_res),
                res_workers=max(int(args.res_workers), 1),
                max_decay_passes=int(args.max_decay_passes),
                fit_objective=str(args.fit_objective),
                fit_weight_mode=str(args.fit_weight_mode),
                fit_opt_maxiter=int(args.fit_opt_maxiter),
                fit_opt_ftol=float(args.fit_opt_ftol),
                fit_balance_xf_bins=int(args.fit_balance_xf_bins),
                fit_balance_pt_bins=int(args.fit_balance_pt_bins),
                prior_table9_lambda=float(args.prior_table9_lambda),
                prior_table9_json=str(args.prior_table9_json),
                prior_strange_lambda=float(args.prior_strange_lambda),
                prior_hyperon_lambda=float(args.prior_hyperon_lambda),
                prior_kminus_over_kplus_target=float(args.prior_kminus_over_kplus_target),
                hyperon_pdgs=str(args.hyperon_pdgs),
                skip_external=bool(args.skip_external),
                external_run_dir=str(args.external_run_dir),
                res_cache_dir=res_cache_dir,
            )
        )
        rows.append(
            row
        )

    rows.sort(key=lambda r: float(r["global_weighted_rel_rmse"]))
    _write_outputs(
        out_root=out_root,
        rows=rows,
        species_csv=species_csv,
        tower_max_values=tower_max_values,
        level_counts=level_counts,
        shapes=shapes,
        shard_count=shard_count,
        shard_index=shard_index,
    )


if __name__ == "__main__":
    main()
