#!/usr/bin/env python3
"""Side study: push effective N* tower toward maximal cascading."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "00runs"


CASCADE_PROFILES: Dict[str, Dict[str, float]] = {
    # Current "strong meson-tower" style used in recent side studies.
    "reference": {
        "mestar_eff_enable": 1.0,
        "nstar_eff_l1_br_nstarpi": 0.00,
        "nstar_eff_l2_br_nstarpi": 0.05,
        "nstar_eff_l3_br_nstarpi": 0.05,
        "nstar_eff_l1_br_nstarmeson": 0.15,
        "nstar_eff_l2_br_nstarmeson": 0.25,
        "nstar_eff_l3_br_nstarmeson": 0.40,
    },
    # Maximal cascading hypothesis: suppress direct channels, emphasize N*->N*+pi/M*.
    # Values are set to sum near 1.0 per level (normalization-safe).
    "maximal": {
        "mestar_eff_enable": 1.0,
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
    },
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
        raise ValueError("No pass values were provided.")
    uniq = sorted(set(vals))
    if any(v <= 0 for v in uniq):
        raise ValueError("All pass values must be > 0.")
    return uniq


def _parse_profiles(raw: str) -> List[str]:
    names = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not names:
        raise ValueError("No profiles were provided.")
    uniq: List[str] = []
    for name in names:
        if name not in CASCADE_PROFILES:
            valid = ", ".join(sorted(CASCADE_PROFILES.keys()))
            raise ValueError(f"Unknown profile '{name}'. Valid: {valid}")
        if name not in uniq:
            uniq.append(name)
    return uniq


def _run_one(
    *,
    out_dir: Path,
    species_csv: str,
    profile_name: str,
    pass_count: int,
    res_events_per_res: int,
    res_workers: int,
    seed: int,
    tower_min_mass: float,
    tower_max_mass: float,
    prior_strange_lambda: float,
    prior_hyperon_lambda: float,
    fit_weight_mode: str,
    fit_balance_xf_bins: int,
    fit_balance_pt_bins: int,
    skip_external: bool,
) -> Dict[str, object]:
    scenario = f"{profile_name}_passes{int(pass_count):02d}"
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
        str(int(pass_count)),
        "--res-workers",
        str(int(res_workers)),
        "--res-fragmentation-mode",
        "both",
        "--res-regge-tower-min-mass",
        f"{float(tower_min_mass):.6f}",
        "--res-regge-tower-max-mass",
        f"{float(tower_max_mass):.6f}",
        "--prior-strange-lambda",
        f"{float(prior_strange_lambda):.12g}",
        "--prior-hyperon-lambda",
        f"{float(prior_hyperon_lambda):.12g}",
        "--fit-weight-mode",
        str(fit_weight_mode),
        "--fit-balance-xf-bins",
        str(int(fit_balance_xf_bins)),
        "--fit-balance-pt-bins",
        str(int(fit_balance_pt_bins)),
    ]
    if bool(skip_external):
        cmd.append("--skip-external")

    # Keep caches local to this side project and profile.
    cmd.extend(["--res-cache-dir", str(out_dir / "res_cache" / profile_name)])

    for key, val in sorted(CASCADE_PROFILES[profile_name].items()):
        cmd.extend(["--res-soft-override", f"{key}={float(val):.12g}"])

    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    summary_path = scenario_dir / "multispecies_constrained_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    global_test_wrrmse = float(payload["resonance"]["global_metrics"]["test_weighted_rel_rmse"])
    per_species = {}
    for sp in payload["per_species"].keys():
        per_species[sp] = float(payload["per_species"][sp]["Resonance_NNLS_shared"]["test_weighted_rel_rmse"])

    return {
        "profile": profile_name,
        "max_decay_passes": int(pass_count),
        "global_test_weighted_rel_rmse": global_test_wrrmse,
        "per_species_test_wrrmse": per_species,
        "summary_path": str(summary_path),
        "details_path": str(scenario_dir / "multispecies_constrained_details.npz"),
    }


def _write_outputs(
    *,
    out_dir: Path,
    rows: List[Dict[str, object]],
    species_csv: str,
    profiles: List[str],
    pass_values: List[int],
) -> None:
    json_path = out_dir / "max_cascade_results.json"
    csv_path = out_dir / "max_cascade_results.csv"
    md_path = out_dir / "MAX_CASCADE_STUDY.md"

    json_path.write_text(
        json.dumps(
            {
                "species": species_csv,
                "profiles": profiles,
                "pass_values": pass_values,
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    fieldnames = [
        "profile",
        "max_decay_passes",
        "global_test_weighted_rel_rmse",
        "summary_path",
        "details_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    lines: List[str] = []
    lines.append("# Maximal Cascade Side Study")
    lines.append("")
    lines.append("Species:")
    lines.append(f"- `{species_csv}`")
    lines.append("Profiles:")
    lines.append(f"- `{', '.join(profiles)}`")
    lines.append("Pass values:")
    lines.append(f"- `{', '.join(str(x) for x in pass_values)}`")
    lines.append("")
    lines.append("| profile | max_decay_passes | global test WRRMSE |")
    lines.append("|---|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['profile']} | {int(row['max_decay_passes'])} | {float(row['global_test_weighted_rel_rmse']):.6f} |"
        )
    lines.append("")
    lines.append("Per-species test WRRMSE:")
    lines.append("")
    for row in rows:
        ps = row.get("per_species_test_wrrmse", {})
        if not isinstance(ps, dict):
            continue
        lines.append(f"- `{row['profile']}` / passes `{int(row['max_decay_passes'])}`:")
        for sp in sorted(ps.keys()):
            lines.append(f"  - `{sp}`: `{float(ps[sp]):.6f}`")
    lines.append("")
    lines.append("Artifacts:")
    lines.append(f"- JSON: `{json_path}`")
    lines.append(f"- CSV: `{csv_path}`")
    lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run maximal-cascade resonance side study.")
    parser.add_argument("--run-description", default="max_cascade_sideproject_2026-03-29")
    parser.add_argument("--species", default="pi-,pi+,k+,k-,p,pbar")
    parser.add_argument("--profiles", default="reference,maximal")
    parser.add_argument("--pass-values", default="8,12,16")
    parser.add_argument("--seed", type=int, default=20260329)
    parser.add_argument("--res-events-per-res", type=int, default=20000)
    parser.add_argument("--res-workers", type=int, default=18)
    parser.add_argument("--tower-min-mass", type=float, default=1.9)
    parser.add_argument("--tower-max-mass", type=float, default=5.0)
    parser.add_argument("--prior-strange-lambda", type=float, default=1.0)
    parser.add_argument("--prior-hyperon-lambda", type=float, default=1.0)
    parser.add_argument("--fit-weight-mode", default="sigma")
    parser.add_argument("--fit-balance-xf-bins", type=int, default=4)
    parser.add_argument("--fit-balance-pt-bins", type=int, default=4)
    parser.add_argument("--skip-external", action="store_true")
    args = parser.parse_args()

    out_dir = RUNS_ROOT / args.run_description
    out_dir.mkdir(parents=True, exist_ok=True)

    species_csv = _parse_species(args.species)
    profiles = _parse_profiles(args.profiles)
    pass_values = _parse_int_csv(args.pass_values)

    rows: List[Dict[str, object]] = []
    for prof in profiles:
        for mp in pass_values:
            row = _run_one(
                out_dir=out_dir,
                species_csv=species_csv,
                profile_name=prof,
                pass_count=int(mp),
                res_events_per_res=int(args.res_events_per_res),
                res_workers=max(int(args.res_workers), 1),
                seed=int(args.seed),
                tower_min_mass=float(args.tower_min_mass),
                tower_max_mass=float(args.tower_max_mass),
                prior_strange_lambda=float(args.prior_strange_lambda),
                prior_hyperon_lambda=float(args.prior_hyperon_lambda),
                fit_weight_mode=str(args.fit_weight_mode),
                fit_balance_xf_bins=int(args.fit_balance_xf_bins),
                fit_balance_pt_bins=int(args.fit_balance_pt_bins),
                skip_external=bool(args.skip_external),
            )
            rows.append(row)

    rows.sort(key=lambda r: (str(r["profile"]), int(r["max_decay_passes"])))
    _write_outputs(
        out_dir=out_dir,
        rows=rows,
        species_csv=species_csv,
        profiles=profiles,
        pass_values=pass_values,
    )


if __name__ == "__main__":
    main()
