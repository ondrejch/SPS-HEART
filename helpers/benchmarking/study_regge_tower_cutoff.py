#!/usr/bin/env python3
"""Side study: scan high-mass N* tower cutoff and cascade assumptions."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from helpers.benchmarking.particle_definitions import resolve_species

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "dev" / "00runs"
DEFAULT_EXTERNAL_RUN = RUNS_ROOT / "gw_multispecies_singlepass_hi_2026-03-29"


CASCADE_MODEL_OVERRIDES: Dict[str, Dict[str, float]] = {
    # Legacy effective-tower cascade defaults from build_soft_resonance_decays.
    "baseline": {},
    # Weak cascade: high-mass states decay mostly directly to hadronic channels.
    "direct": {
        "nstar_eff_l1_br_npi": 0.20,
        "nstar_eff_l1_br_deltapi": 0.40,
        "nstar_eff_l1_br_nrho": 0.25,
        "nstar_eff_l1_br_neta": 0.05,
        "nstar_eff_l1_br_lambdak": 0.05,
        "nstar_eff_l1_br_nstarpi": 0.05,
        "nstar_eff_l2_br_npi": 0.20,
        "nstar_eff_l2_br_deltapi": 0.35,
        "nstar_eff_l2_br_nrho": 0.30,
        "nstar_eff_l2_br_neta": 0.05,
        "nstar_eff_l2_br_lambdak": 0.05,
        "nstar_eff_l2_br_nstarpi": 0.05,
        "nstar_eff_l3_br_npi": 0.22,
        "nstar_eff_l3_br_deltapi": 0.33,
        "nstar_eff_l3_br_nrho": 0.30,
        "nstar_eff_l3_br_neta": 0.05,
        "nstar_eff_l3_br_lambdak": 0.05,
        "nstar_eff_l3_br_nstarpi": 0.05,
    },
    # Strong cascade: favor R -> R' + pi links in upper tower levels.
    "cascade_strong": {
        "nstar_eff_l1_br_npi": 0.10,
        "nstar_eff_l1_br_deltapi": 0.30,
        "nstar_eff_l1_br_nrho": 0.20,
        "nstar_eff_l1_br_neta": 0.05,
        "nstar_eff_l1_br_lambdak": 0.05,
        "nstar_eff_l1_br_nstarpi": 0.30,
        "nstar_eff_l2_br_npi": 0.06,
        "nstar_eff_l2_br_deltapi": 0.21,
        "nstar_eff_l2_br_nrho": 0.13,
        "nstar_eff_l2_br_neta": 0.05,
        "nstar_eff_l2_br_lambdak": 0.05,
        "nstar_eff_l2_br_nstarpi": 0.50,
        "nstar_eff_l3_br_npi": 0.04,
        "nstar_eff_l3_br_deltapi": 0.14,
        "nstar_eff_l3_br_nrho": 0.08,
        "nstar_eff_l3_br_neta": 0.04,
        "nstar_eff_l3_br_lambdak": 0.05,
        "nstar_eff_l3_br_nstarpi": 0.65,
    },
    # Stress-test very deep cascading in unresolved region.
    "cascade_extreme": {
        "nstar_eff_l1_br_npi": 0.08,
        "nstar_eff_l1_br_deltapi": 0.24,
        "nstar_eff_l1_br_nrho": 0.13,
        "nstar_eff_l1_br_neta": 0.03,
        "nstar_eff_l1_br_lambdak": 0.02,
        "nstar_eff_l1_br_nstarpi": 0.50,
        "nstar_eff_l2_br_npi": 0.04,
        "nstar_eff_l2_br_deltapi": 0.14,
        "nstar_eff_l2_br_nrho": 0.07,
        "nstar_eff_l2_br_neta": 0.03,
        "nstar_eff_l2_br_lambdak": 0.02,
        "nstar_eff_l2_br_nstarpi": 0.70,
        "nstar_eff_l3_br_npi": 0.03,
        "nstar_eff_l3_br_deltapi": 0.08,
        "nstar_eff_l3_br_nrho": 0.04,
        "nstar_eff_l3_br_neta": 0.02,
        "nstar_eff_l3_br_lambdak": 0.03,
        "nstar_eff_l3_br_nstarpi": 0.80,
    },
}


def _parse_cutoffs(raw: str) -> List[float]:
    vals: List[float] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        vals.append(float(text))
    if not vals:
        raise ValueError("No cutoff values were provided.")
    return sorted(set(vals))


def _parse_cascade_models(raw: str) -> List[str]:
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if not names:
        raise ValueError("No cascade models were provided.")
    uniq: List[str] = []
    for name in names:
        if name not in CASCADE_MODEL_OVERRIDES:
            valid = ", ".join(sorted(CASCADE_MODEL_OVERRIDES.keys()))
            raise ValueError(f"Unknown cascade model '{name}'. Valid: {valid}")
        if name not in uniq:
            uniq.append(name)
    return uniq


def _scenario_tag(cutoff: float, cascade_model: str) -> str:
    return f"{cascade_model}_towermax{int(round(float(cutoff) * 1000.0)):04d}"


def _default_external_csvs(spec_tag: str, external_run_dir: Path) -> Dict[str, Path]:
    return {
        "epos": external_run_dir / f"epos_lhc_pp158_{spec_tag}_gateway.csv",
        "urqmd": external_run_dir / f"urqmd_pp158_{spec_tag}_gateway.csv",
    }


def _run_benchmark_one(
    *,
    species_name: str,
    species_tag: str,
    out_dir: Path,
    cutoff: float,
    min_mass: float,
    cascade_model: str,
    soft_overrides: Dict[str, float],
    res_events_per_res: int,
    res_max_decay_passes: int,
    res_workers: int,
    external_csvs: Dict[str, Path],
    seed: int,
    force_rebuild_cache: bool,
) -> Dict[str, object]:
    bench_script = REPO_ROOT / "helpers" / "benchmarking" / "benchmark_sps_species.py"
    scenario = _scenario_tag(cutoff, cascade_model)
    out_prefix = f"na49_benchmark_{species_tag}_{scenario}"
    cache_dir = out_dir / "res_cache" / cascade_model

    cmd = [
        sys.executable,
        str(bench_script),
        "--species",
        species_name,
        "--out-dir",
        str(out_dir),
        "--out-prefix",
        out_prefix,
        "--seed",
        str(int(seed)),
        "--disable-pythia",
        "--res-events-per-res",
        str(int(res_events_per_res)),
        "--res-max-decay-passes",
        str(int(res_max_decay_passes)),
        "--res-workers",
        str(int(res_workers)),
        "--res-cache-dir",
        str(cache_dir),
        "--res-cache-scope",
        "species",
        "--res-regge-tower-min-mass",
        f"{float(min_mass):.6f}",
        "--res-regge-tower-max-mass",
        f"{float(cutoff):.6f}",
    ]
    for key in sorted(soft_overrides.keys()):
        cmd.extend(["--res-soft-override", f"{key}={float(soft_overrides[key]):.12g}"])
    if force_rebuild_cache:
        cmd.append("--res-cache-force-rebuild")

    epos_csv = external_csvs.get("epos")
    urqmd_csv = external_csvs.get("urqmd")
    if epos_csv is not None and epos_csv.exists():
        cmd.extend(["--epos-csv", str(epos_csv)])
    if urqmd_csv is not None and urqmd_csv.exists():
        cmd.extend(["--urqmd-csv", str(urqmd_csv)])

    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    summary_path = out_dir / f"{out_prefix}_summary.json"
    details_path = out_dir / f"{out_prefix}_details.npz"
    return {
        "scenario": scenario,
        "cascade_model": cascade_model,
        "summary_path": summary_path,
        "details_path": details_path,
    }


def _extract_metrics(
    *,
    summary_path: Path,
    details_path: Path,
    low_pt_max: float,
) -> Dict[str, float | int]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    res_row = None
    for item in payload.get("summary", []):
        if item.get("model") == "Resonance_NNLS":
            res_row = item
            break
    if res_row is None:
        raise RuntimeError(f"Resonance_NNLS row not found in {summary_path}")
    if "status" in res_row:
        raise RuntimeError(f"Resonance_NNLS failed in {summary_path}: {res_row['status']}")

    with np.load(details_path, allow_pickle=False) as z:
        pt = np.asarray(z["pt"], dtype=float)
        y = np.asarray(z["y"], dtype=float)
        sigma = np.asarray(z["sigma"], dtype=float)
        pred = np.asarray(z["resonance_pred"], dtype=float)
        test_idx = np.asarray(z["test_idx"], dtype=int)

    test_mask = np.zeros_like(pt, dtype=bool)
    test_mask[test_idx] = True
    low_mask = test_mask & (pt <= float(low_pt_max)) & (y > 0.0) & (sigma > 0.0)

    if np.any(low_mask):
        rel = (pred[low_mask] - y[low_mask]) / np.maximum(y[low_mask], 1e-30)
        pull = (pred[low_mask] - y[low_mask]) / np.maximum(sigma[low_mask], 1e-12)
        low_rel_bias = float(np.mean(rel))
        low_rel_abs_mean = float(np.mean(np.abs(rel)))
        low_pull_mean = float(np.mean(pull))
        low_pull_abs_mean = float(np.mean(np.abs(pull)))
        low_points = int(np.sum(low_mask))
    else:
        low_rel_bias = float("nan")
        low_rel_abs_mean = float("nan")
        low_pull_mean = float("nan")
        low_pull_abs_mean = float("nan")
        low_points = 0

    return {
        "n_params": int(res_row.get("n_params", 0)),
        "test_weighted_rel_rmse": float(res_row.get("test_weighted_rel_rmse", float("nan"))),
        "test_log_pull_rms": float(res_row.get("test_log_pull_rms", float("nan"))),
        "test_smape_percent": float(res_row.get("test_smape_percent", float("nan"))),
        "test_chi2_dof": float(res_row.get("test_chi2_dof", float("nan"))),
        "low_pt_points_test": int(low_points),
        "low_pt_rel_bias_test": low_rel_bias,
        "low_pt_rel_abs_mean_test": low_rel_abs_mean,
        "low_pt_pull_mean_test": low_pull_mean,
        "low_pt_pull_abs_mean_test": low_pull_abs_mean,
    }


def _write_outputs(
    *,
    out_dir: Path,
    rows: List[Dict[str, object]],
    species: List[str],
    cutoffs: List[float],
    cascade_models: List[str],
    low_pt_max: float,
) -> None:
    json_path = out_dir / "regge_tower_cutoff_scan_results.json"
    csv_path = out_dir / "regge_tower_cutoff_scan_results.csv"
    md_path = out_dir / "REGGE_TOWER_CUTOFF_SCAN.md"
    plot_path = out_dir / "regge_tower_cutoff_scan.png"

    json_path.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")

    fieldnames = [
        "species",
        "cascade_model",
        "tower_max_mass",
        "n_params",
        "test_weighted_rel_rmse",
        "test_log_pull_rms",
        "test_smape_percent",
        "test_chi2_dof",
        "low_pt_points_test",
        "low_pt_rel_bias_test",
        "low_pt_rel_abs_mean_test",
        "low_pt_pull_mean_test",
        "low_pt_pull_abs_mean_test",
        "summary_path",
        "details_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    lines: List[str] = []
    lines.append("# Regge Tower Cutoff Side Study")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- Species: {', '.join(species)}")
    lines.append(f"- Cascade models: {', '.join(cascade_models)}")
    lines.append(f"- Tower max mass scan [GeV]: {', '.join(f'{x:.2f}' for x in cutoffs)}")
    lines.append(f"- Low-pT region for bias check: pT <= {low_pt_max:.3f} GeV/c (test split only)")
    lines.append("")

    for sp in species:
        lines.append(f"## Species `{sp}`")
        lines.append("")
        lines.append("| cascade model | tower max [GeV] | WRRMSE | log pull RMS | low-pT rel bias | low-pT abs pull mean | n_params |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        subset = [r for r in rows if str(r["species"]) == sp]
        subset.sort(key=lambda x: (str(x["cascade_model"]), float(x["tower_max_mass"])))
        for r in subset:
            lines.append(
                "| "
                f"{str(r['cascade_model'])} | "
                f"{float(r['tower_max_mass']):.2f} | "
                f"{float(r['test_weighted_rel_rmse']):.4f} | "
                f"{float(r['test_log_pull_rms']):.3f} | "
                f"{float(r['low_pt_rel_bias_test']):+.4f} | "
                f"{float(r['low_pt_pull_abs_mean_test']):.3f} | "
                f"{int(r['n_params'])} |"
            )

        lines.append("")
        by_model = sorted({str(r["cascade_model"]) for r in subset})
        for model in by_model:
            model_rows = [r for r in subset if str(r["cascade_model"]) == model]
            if not model_rows:
                continue
            best = min(model_rows, key=lambda x: float(x["test_weighted_rel_rmse"]))
            lines.append(
                f"- Best WRRMSE for `{model}` at {float(best['tower_max_mass']):.2f} GeV "
                f"(WRRMSE={float(best['test_weighted_rel_rmse']):.4f}, "
                f"low-pT rel bias={float(best['low_pt_rel_bias_test']):+.4f})."
            )

        if subset:
            best_all = min(subset, key=lambda x: float(x["test_weighted_rel_rmse"]))
            lines.append(
                f"- Overall best at `{str(best_all['cascade_model'])}` and "
                f"{float(best_all['tower_max_mass']):.2f} GeV "
                f"(WRRMSE={float(best_all['test_weighted_rel_rmse']):.4f}, "
                f"low-pT rel bias={float(best_all['low_pt_rel_bias_test']):+.4f})."
            )
        lines.append("")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("plot: matplotlib unavailable, skipping plot generation", flush=True)
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for sp in species:
        for model in cascade_models:
            subset = [
                r
                for r in rows
                if str(r["species"]) == sp and str(r["cascade_model"]) == model
            ]
            if not subset:
                continue
            subset.sort(key=lambda x: float(x["tower_max_mass"]))
            xs = np.array([float(r["tower_max_mass"]) for r in subset], dtype=float)
            ys_wr = np.array([float(r["test_weighted_rel_rmse"]) for r in subset], dtype=float)
            ys_bias = np.array([float(r["low_pt_rel_bias_test"]) for r in subset], dtype=float)
            label = f"{sp}/{model}"
            axes[0].plot(xs, ys_wr, marker="o", label=label)
            axes[1].plot(xs, ys_bias, marker="o", label=label)
    axes[0].set_ylabel("Test WRRMSE")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    axes[1].set_ylabel("Low-pT Test Relative Bias")
    axes[1].set_xlabel("Effective Tower Max Mass [GeV]")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("NA49 Resonance Side Study: Tower Cutoff x Cascade Assumption")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan effective N* tower mass cutoff and compare resonance fit metrics."
    )
    parser.add_argument("--species", default="pi-", help="Comma-separated species names, e.g. pi-,pi+,p,pbar")
    parser.add_argument(
        "--cascade-models",
        default="baseline,direct,cascade_strong",
        help="Comma-separated cascade assumptions.",
    )
    parser.add_argument(
        "--tower-max-values",
        default="2.3,2.7,3.1,3.5,4.0,4.5,5.0",
        help="Comma-separated tower max masses [GeV].",
    )
    parser.add_argument("--tower-min-mass", type=float, default=1.9, help="Tower min mass [GeV].")
    parser.add_argument("--low-pt-max", type=float, default=0.20, help="Low-pT threshold [GeV/c] for bias summary.")
    parser.add_argument("--run-description", default="regge_tower_cutoff_scan_2026-03-29")
    parser.add_argument("--res-events-per-res", type=int, default=30000)
    parser.add_argument("--res-max-decay-passes", type=int, default=8)
    parser.add_argument("--res-workers", type=int, default=18)
    parser.add_argument("--seed", type=int, default=20260329)
    parser.add_argument(
        "--external-run-dir",
        default=str(DEFAULT_EXTERNAL_RUN),
        help="Directory with merged EPOS/UrQMD CSVs. If missing for a species, external baselines are skipped.",
    )
    parser.add_argument("--force-rebuild-cache", action="store_true")
    args = parser.parse_args()

    species_names = [s.strip() for s in args.species.split(",") if s.strip()]
    species_defs = [resolve_species(name) for name in species_names]
    cutoffs = _parse_cutoffs(args.tower_max_values)
    cascade_models = _parse_cascade_models(args.cascade_models)

    out_dir = RUNS_ROOT / args.run_description
    out_dir.mkdir(parents=True, exist_ok=True)
    external_run_dir = Path(args.external_run_dir)

    rows: List[Dict[str, object]] = []
    for sp in species_defs:
        external_csvs = _default_external_csvs(sp.tag, external_run_dir)
        for cascade_model in cascade_models:
            soft_overrides = dict(CASCADE_MODEL_OVERRIDES[cascade_model])
            for cutoff in cutoffs:
                run_info = _run_benchmark_one(
                    species_name=sp.name,
                    species_tag=sp.tag,
                    out_dir=out_dir,
                    cutoff=cutoff,
                    min_mass=float(args.tower_min_mass),
                    cascade_model=cascade_model,
                    soft_overrides=soft_overrides,
                    res_events_per_res=int(args.res_events_per_res),
                    res_max_decay_passes=int(args.res_max_decay_passes),
                    res_workers=max(int(args.res_workers), 1),
                    external_csvs=external_csvs,
                    seed=int(args.seed),
                    force_rebuild_cache=bool(args.force_rebuild_cache),
                )
                metrics = _extract_metrics(
                    summary_path=Path(str(run_info["summary_path"])),
                    details_path=Path(str(run_info["details_path"])),
                    low_pt_max=float(args.low_pt_max),
                )
                row: Dict[str, object] = {
                    "species": sp.name,
                    "species_tag": sp.tag,
                    "cascade_model": cascade_model,
                    "tower_max_mass": float(cutoff),
                    "summary_path": str(run_info["summary_path"]),
                    "details_path": str(run_info["details_path"]),
                }
                row.update(metrics)
                rows.append(row)
                print(
                    f"result: species={sp.name} cascade={cascade_model} cutoff={cutoff:.2f} "
                    f"WRRMSE={float(metrics['test_weighted_rel_rmse']):.4f} "
                    f"lowPtBias={float(metrics['low_pt_rel_bias_test']):+.4f}",
                    flush=True,
                )

    _write_outputs(
        out_dir=out_dir,
        rows=rows,
        species=[sp.name for sp in species_defs],
        cutoffs=cutoffs,
        cascade_models=cascade_models,
        low_pt_max=float(args.low_pt_max),
    )
    print(f"saved: {out_dir / 'regge_tower_cutoff_scan_results.json'}")
    print(f"saved: {out_dir / 'regge_tower_cutoff_scan_results.csv'}")
    print(f"saved: {out_dir / 'REGGE_TOWER_CUTOFF_SCAN.md'}")
    print(f"saved: {out_dir / 'regge_tower_cutoff_scan.png'}")


if __name__ == "__main__":
    main()
