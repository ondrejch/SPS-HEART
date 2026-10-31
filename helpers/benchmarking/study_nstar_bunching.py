#!/usr/bin/env python3
"""Side study: effective N* tower bunching (level count + mass-ladder shape)."""

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


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "dev" / "00runs"


BESTMIX_SOFT_OVERRIDES: Dict[str, float] = {
    "mestar_eff_enable": 1.0,
    "nstar_eff_tower_min_mass": 1.9,
    "nstar_eff_tower_max_mass": 2.3,
    "mestar_eff_tower_min_mass": 0.45,
    "mestar_eff_tower_max_mass": 1.25,
    "nstar_eff_l1_br_npi": 0.01,
    "nstar_eff_l1_br_deltapi": 0.02,
    "nstar_eff_l1_br_nrho": 0.02,
    "nstar_eff_l1_br_neta": 0.01,
    "nstar_eff_l1_br_lambdak": 0.01,
    "nstar_eff_l1_br_nstarpi": 0.70,
    "nstar_eff_l1_br_nstarmeson": 0.23,
    "nstar_eff_l2_br_npi": 0.01,
    "nstar_eff_l2_br_deltapi": 0.02,
    "nstar_eff_l2_br_nrho": 0.02,
    "nstar_eff_l2_br_neta": 0.01,
    "nstar_eff_l2_br_lambdak": 0.01,
    "nstar_eff_l2_br_nstarpi": 0.78,
    "nstar_eff_l2_br_nstarmeson": 0.15,
    "nstar_eff_l3_br_npi": 0.01,
    "nstar_eff_l3_br_deltapi": 0.02,
    "nstar_eff_l3_br_nrho": 0.02,
    "nstar_eff_l3_br_neta": 0.01,
    "nstar_eff_l3_br_lambdak": 0.01,
    "nstar_eff_l3_br_nstarpi": 0.85,
    "nstar_eff_l3_br_nstarmeson": 0.08,
}


def _parse_species(raw: str) -> str:
    txt = ",".join([x.strip() for x in str(raw).split(",") if x.strip()])
    if not txt:
        raise ValueError("No species were provided.")
    return txt


def _parse_int_csv(raw: str) -> List[int]:
    vals: List[int] = []
    for part in str(raw).split(","):
        p = part.strip()
        if not p:
            continue
        vals.append(int(p))
    if not vals:
        raise ValueError("No integer values were provided.")
    out = sorted(set(max(v, 1) for v in vals))
    return out


def _parse_shapes(raw: str) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for part in str(raw).split(","):
        tok = part.strip()
        if not tok:
            continue
        if ":" in tok:
            shp, par = tok.split(":", 1)
            shp = shp.strip().lower()
            parv = float(par.strip())
        else:
            shp = tok.strip().lower()
            parv = 0.25
        if shp not in bss.REGGE_TOWER_SHAPES:
            raise ValueError(f"Unknown shape '{shp}'. Valid: {', '.join(bss.REGGE_TOWER_SHAPES)}")
        out.append((shp, parv))
    if not out:
        raise ValueError("No shapes provided.")
    uniq: List[Tuple[str, float]] = []
    seen: set[Tuple[str, float]] = set()
    for item in out:
        if item in seen:
            continue
        seen.add(item)
        uniq.append(item)
    return uniq


def _shape_tag(shape: str, param: float) -> str:
    if shape == "hagedorn":
        return f"{shape}_t{int(round(float(param) * 1000.0)):04d}"
    return shape


def _run_one(
    *,
    out_dir: Path,
    species_csv: str,
    shape: str,
    shape_param: float,
    level_count: int,
    min_mass: float,
    max_mass: float,
    seed: int,
    res_events_per_res: int,
    res_workers: int,
    max_decay_passes: int,
    fit_weight_mode: str,
    fit_balance_xf_bins: int,
    fit_balance_pt_bins: int,
    offshell_kappa: float,
    offshell_beta: float,
    offshell_window_scale: float,
    skip_external: bool,
    external_run_dir: str,
) -> Dict[str, object]:
    scenario = f"{_shape_tag(shape, shape_param)}_n{int(level_count):02d}"
    if float(offshell_kappa) > 0.0 or abs(float(offshell_window_scale) - 1.0) > 1e-12:
        scenario += (
            f"_offk{int(round(float(offshell_kappa) * 1000.0)):04d}"
            f"_offb{int(round(float(offshell_beta) * 1000.0)):04d}"
            f"_offw{int(round(float(offshell_window_scale) * 1000.0)):04d}"
        )
    scenario_dir = out_dir / scenario
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
        f"{float(max_mass):.6f}",
        "--res-regge-tower-level-count",
        str(int(level_count)),
        "--res-regge-tower-shape",
        str(shape),
        "--res-regge-tower-shape-param",
        f"{float(shape_param):.12g}",
        "--res-regge-tower-offshell-kappa",
        f"{float(offshell_kappa):.12g}",
        "--res-regge-tower-offshell-beta",
        f"{float(offshell_beta):.12g}",
        "--res-regge-tower-offshell-window-scale",
        f"{float(offshell_window_scale):.12g}",
        "--fit-weight-mode",
        str(fit_weight_mode),
        "--fit-balance-xf-bins",
        str(int(fit_balance_xf_bins)),
        "--fit-balance-pt-bins",
        str(int(fit_balance_pt_bins)),
        "--external-run-dir",
        str(external_run_dir),
        "--res-cache-dir",
        str(out_dir / "res_cache"),
    ]
    if bool(skip_external):
        cmd.append("--skip-external")

    for key, val in sorted(BESTMIX_SOFT_OVERRIDES.items()):
        cmd.extend(["--res-soft-override", f"{key}={float(val):.12g}"])

    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    summary_path = scenario_dir / "multispecies_constrained_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    masses = bss._compute_regge_tower_mass_ladder(
        float(min_mass),
        float(max_mass),
        level_count=int(level_count),
        shape=str(shape),
        shape_param=float(shape_param),
    )
    spacings = np.diff(masses) if masses.size > 1 else np.asarray([0.0], dtype=float)
    spacing_ratio_last_first = float(spacings[-1] / spacings[0]) if spacings[0] > 0 else float("nan")

    per_species_res: Dict[str, float] = {}
    per_species = payload.get("per_species", {})
    for sp in sorted(per_species.keys()):
        per_species_res[sp] = float(per_species[sp]["Resonance_NNLS_shared"]["full_weighted_rel_rmse"])

    return {
        "scenario": scenario,
        "shape": str(shape),
        "shape_param": float(shape_param),
        "level_count": int(level_count),
        "tower_min_mass": float(min_mass),
        "tower_max_mass": float(max_mass),
        "tower_masses": [float(x) for x in masses.tolist()],
        "tower_spacings": [float(x) for x in spacings.tolist()],
        "spacing_ratio_last_first": spacing_ratio_last_first,
        "global_full_weighted_rel_rmse": float(payload["resonance"]["global_metrics"]["full_weighted_rel_rmse"]),
        "global_full_log_pull_rms": float(payload["resonance"]["global_metrics"]["full_log_pull_rms"]),
        "n_active_components": int(payload["resonance"]["n_active"]),
        "per_species_resonance_full_wrrmse": per_species_res,
        "summary_path": str(summary_path),
        "details_path": str(scenario_dir / "multispecies_constrained_details.npz"),
    }


def _write_outputs(
    *,
    out_dir: Path,
    rows: List[Dict[str, object]],
    species_csv: str,
    shapes: List[Tuple[str, float]],
    level_counts: List[int],
    min_mass: float,
    max_mass: float,
) -> None:
    json_path = out_dir / "nstar_bunching_scan_results.json"
    csv_path = out_dir / "nstar_bunching_scan_results.csv"
    md_path = out_dir / "NSTAR_BUNCHING_SCAN.md"

    payload = {
        "species": species_csv,
        "shapes": [{"shape": s, "shape_param": p} for s, p in shapes],
        "level_counts": level_counts,
        "tower_min_mass": float(min_mass),
        "tower_max_mass": float(max_mass),
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    fieldnames = [
        "scenario",
        "shape",
        "shape_param",
        "level_count",
        "tower_min_mass",
        "tower_max_mass",
        "spacing_ratio_last_first",
        "global_full_weighted_rel_rmse",
        "global_full_log_pull_rms",
        "n_active_components",
        "summary_path",
        "details_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    best = min(rows, key=lambda r: float(r["global_full_weighted_rel_rmse"])) if rows else None

    lines: List[str] = []
    lines.append("# N* Tower Bunching Side Study")
    lines.append("")
    lines.append("Species:")
    lines.append(f"- `{species_csv}`")
    lines.append("Tower mass range [GeV]:")
    lines.append(f"- `{float(min_mass):.3f} .. {float(max_mass):.3f}`")
    lines.append("Shapes:")
    lines.append(
        "- `"
        + ", ".join([f"{s}:{p:.3f}" if s == "hagedorn" else s for s, p in shapes])
        + "`"
    )
    lines.append("Level counts:")
    lines.append(f"- `{', '.join(str(x) for x in level_counts)}`")
    lines.append("")
    lines.append("| scenario | shape | levels | spacing ratio last/first | global WRRMSE | active comps |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| "
            + f"{row['scenario']} | "
            + f"{row['shape']} | "
            + f"{int(row['level_count'])} | "
            + f"{float(row['spacing_ratio_last_first']):.4f} | "
            + f"{float(row['global_full_weighted_rel_rmse']):.6f} | "
            + f"{int(row['n_active_components'])} |"
        )
    lines.append("")
    if best is not None:
        lines.append("Best scenario (lowest global WRRMSE):")
        lines.append(
            "- "
            + f"`{best['scenario']}` "
            + f"(shape `{best['shape']}`, levels `{int(best['level_count'])}`, "
            + f"WRRMSE `{float(best['global_full_weighted_rel_rmse']):.6f}`)"
        )
        lines.append("")
        lines.append("Best scenario tower masses [GeV]:")
        lines.append("- `" + ", ".join(f"{float(x):.4f}" for x in list(best["tower_masses"])) + "`")
        lines.append("")

    lines.append("Artifacts:")
    lines.append(f"- JSON: `{json_path}`")
    lines.append(f"- CSV: `{csv_path}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan N* tower bunching hypotheses.")
    parser.add_argument("--run-description", default="nstar_bunching_sideproject_2026-03-30")
    parser.add_argument("--species", default="pi-,pi+,k+,k-,p,pbar")
    parser.add_argument(
        "--shapes",
        default="linear,regge_m2,hagedorn:0.20,hagedorn:0.25,hagedorn:0.30",
        help="CSV of shapes; use hagedorn:<T_H_GeV> for shape parameter.",
    )
    parser.add_argument("--level-counts", default="3,5,7,9")
    parser.add_argument("--tower-min-mass", type=float, default=1.9)
    parser.add_argument("--tower-max-mass", type=float, default=2.3)
    parser.add_argument("--seed", type=int, default=20260330)
    parser.add_argument("--res-events-per-res", type=int, default=15000)
    parser.add_argument("--res-workers", type=int, default=18)
    parser.add_argument("--max-decay-passes", type=int, default=16)
    parser.add_argument("--fit-weight-mode", default="relative_xfpt_balanced")
    parser.add_argument("--fit-balance-xf-bins", type=int, default=4)
    parser.add_argument("--fit-balance-pt-bins", type=int, default=4)
    parser.add_argument(
        "--offshell-kappa",
        type=float,
        default=0.0,
        help="Stage-1 off-shell broadening strength forwarded to tower materialization.",
    )
    parser.add_argument(
        "--offshell-beta",
        type=float,
        default=1.0,
        help="Exponent beta for off-shell width broadening law.",
    )
    parser.add_argument(
        "--offshell-window-scale",
        type=float,
        default=1.0,
        help="Mass-window scale factor for truncated off-shell sampling windows.",
    )
    parser.add_argument(
        "--external-run-dir",
        default=str(RUNS_ROOT / "gw_multispecies_allmodels_singlepass_pythia_2026-03-30"),
    )
    parser.add_argument("--skip-external", action="store_true")
    args = parser.parse_args()

    out_dir = RUNS_ROOT / args.run_description
    out_dir.mkdir(parents=True, exist_ok=True)

    species_csv = _parse_species(args.species)
    shapes = _parse_shapes(args.shapes)
    level_counts = _parse_int_csv(args.level_counts)

    rows: List[Dict[str, object]] = []
    for shape, shape_param in shapes:
        for nlevels in level_counts:
            rows.append(
                _run_one(
                    out_dir=out_dir,
                    species_csv=species_csv,
                    shape=str(shape),
                    shape_param=float(shape_param),
                    level_count=int(nlevels),
                    min_mass=float(args.tower_min_mass),
                    max_mass=float(args.tower_max_mass),
                    seed=int(args.seed),
                    res_events_per_res=int(args.res_events_per_res),
                    res_workers=max(int(args.res_workers), 1),
                    max_decay_passes=int(args.max_decay_passes),
                    fit_weight_mode=str(args.fit_weight_mode),
                    fit_balance_xf_bins=int(args.fit_balance_xf_bins),
                    fit_balance_pt_bins=int(args.fit_balance_pt_bins),
                    offshell_kappa=float(args.offshell_kappa),
                    offshell_beta=float(args.offshell_beta),
                    offshell_window_scale=float(args.offshell_window_scale),
                    skip_external=bool(args.skip_external),
                    external_run_dir=str(args.external_run_dir),
                )
            )

    rows.sort(
        key=lambda r: (
            str(r["shape"]),
            float(r["shape_param"]),
            int(r["level_count"]),
        )
    )

    _write_outputs(
        out_dir=out_dir,
        rows=rows,
        species_csv=species_csv,
        shapes=shapes,
        level_counts=level_counts,
        min_mass=float(args.tower_min_mass),
        max_mass=float(args.tower_max_mass),
    )


if __name__ == "__main__":
    main()
