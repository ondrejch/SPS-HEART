#!/usr/bin/env python3
"""Side study: scan baryon/meson effective-tower maximum masses."""

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
    "reference": {
        "mestar_eff_enable": 1.0,
        "nstar_eff_l1_br_nstarpi": 0.00,
        "nstar_eff_l2_br_nstarpi": 0.05,
        "nstar_eff_l3_br_nstarpi": 0.05,
        "nstar_eff_l1_br_nstarmeson": 0.15,
        "nstar_eff_l2_br_nstarmeson": 0.25,
        "nstar_eff_l3_br_nstarmeson": 0.40,
    },
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


def _parse_float_csv(raw: str) -> List[float]:
    vals: List[float] = []
    for part in str(raw).split(","):
        p = part.strip()
        if not p:
            continue
        vals.append(float(p))
    if not vals:
        raise ValueError("No values were provided.")
    return sorted(set(vals), reverse=True)


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


def _tag_mass(mass: float) -> str:
    return f"{int(round(float(mass) * 1000.0)):04d}"


def _run_one(
    *,
    out_dir: Path,
    species_csv: str,
    profile_name: str,
    baryon_min_mass: float,
    baryon_max_mass: float,
    meson_min_mass: float,
    meson_max_mass: float,
    max_decay_passes: int,
    res_events_per_res: int,
    res_workers: int,
    seed: int,
    prior_strange_lambda: float,
    prior_hyperon_lambda: float,
    fit_weight_mode: str,
    fit_balance_xf_bins: int,
    fit_balance_pt_bins: int,
    skip_external: bool,
) -> Dict[str, object]:
    scenario = (
        f"{profile_name}"
        f"_bmax{_tag_mass(baryon_max_mass)}"
        f"_mmax{_tag_mass(meson_max_mass)}"
        f"_passes{int(max_decay_passes):02d}"
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
        f"{float(baryon_min_mass):.6f}",
        "--res-regge-tower-max-mass",
        f"{float(baryon_max_mass):.6f}",
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

    cmd.extend(["--res-cache-dir", str(out_dir / "res_cache" / profile_name)])

    # Tower mass controls (applied in na49_mixing.parallel via soft overrides).
    cmd.extend(["--res-soft-override", f"nstar_eff_tower_min_mass={float(baryon_min_mass):.12g}"])
    cmd.extend(["--res-soft-override", f"nstar_eff_tower_max_mass={float(baryon_max_mass):.12g}"])
    cmd.extend(["--res-soft-override", f"mestar_eff_tower_min_mass={float(meson_min_mass):.12g}"])
    cmd.extend(["--res-soft-override", f"mestar_eff_tower_max_mass={float(meson_max_mass):.12g}"])

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
        "baryon_min_mass": float(baryon_min_mass),
        "baryon_max_mass": float(baryon_max_mass),
        "meson_min_mass": float(meson_min_mass),
        "meson_max_mass": float(meson_max_mass),
        "max_decay_passes": int(max_decay_passes),
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
    baryon_max_values: List[float],
    meson_max_values: List[float],
) -> None:
    json_path = out_dir / "tower_limit_scan_results.json"
    csv_path = out_dir / "tower_limit_scan_results.csv"
    md_path = out_dir / "TOWER_LIMIT_SCAN.md"

    json_path.write_text(
        json.dumps(
            {
                "species": species_csv,
                "profiles": profiles,
                "baryon_max_values": baryon_max_values,
                "meson_max_values": meson_max_values,
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    fieldnames = [
        "profile",
        "baryon_min_mass",
        "baryon_max_mass",
        "meson_min_mass",
        "meson_max_mass",
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

    best = min(rows, key=lambda r: float(r["global_test_weighted_rel_rmse"])) if rows else None

    lines: List[str] = []
    lines.append("# Tower Limit Side Study")
    lines.append("")
    lines.append("Species:")
    lines.append(f"- `{species_csv}`")
    lines.append("Profiles:")
    lines.append(f"- `{', '.join(profiles)}`")
    lines.append("Baryon max masses [GeV]:")
    lines.append(f"- `{', '.join(f'{v:.3f}' for v in baryon_max_values)}`")
    lines.append("Meson max masses [GeV]:")
    lines.append(f"- `{', '.join(f'{v:.3f}' for v in meson_max_values)}`")
    lines.append("")
    lines.append("| profile | baryon max [GeV] | meson max [GeV] | passes | global test WRRMSE |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| "
            + f"{row['profile']} | "
            + f"{float(row['baryon_max_mass']):.3f} | "
            + f"{float(row['meson_max_mass']):.3f} | "
            + f"{int(row['max_decay_passes'])} | "
            + f"{float(row['global_test_weighted_rel_rmse']):.6f} |"
        )
    lines.append("")

    if best is not None:
        lines.append("Best scenario (lowest global test WRRMSE):")
        lines.append(
            "- "
            + f"`{best['profile']}`, "
            + f"baryon max `{float(best['baryon_max_mass']):.3f}` GeV, "
            + f"meson max `{float(best['meson_max_mass']):.3f}` GeV, "
            + f"passes `{int(best['max_decay_passes'])}`, "
            + f"WRRMSE `{float(best['global_test_weighted_rel_rmse']):.6f}`"
        )
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
    parser = argparse.ArgumentParser(description="Scan baryon/meson tower maximum masses.")
    parser.add_argument("--run-description", default="tower_limit_scan_sideproject_2026-03-29")
    parser.add_argument("--species", default="pi-,pi+,k+,k-,p,pbar")
    parser.add_argument("--profiles", default="maximal")
    parser.add_argument("--baryon-min-mass", type=float, default=1.9)
    parser.add_argument("--baryon-max-values", default="5.0,4.5,4.0,3.5,3.0,2.7,2.3")
    parser.add_argument("--meson-min-mass", type=float, default=0.45)
    parser.add_argument("--meson-max-values", default="1.25,1.10,0.95,0.80,0.65,0.55,0.45")
    parser.add_argument("--max-decay-passes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260329)
    parser.add_argument("--res-events-per-res", type=int, default=12000)
    parser.add_argument("--res-workers", type=int, default=18)
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
    baryon_max_values = _parse_float_csv(args.baryon_max_values)
    meson_max_values = _parse_float_csv(args.meson_max_values)

    rows: List[Dict[str, object]] = []
    for profile_name in profiles:
        for bmax in baryon_max_values:
            if float(bmax) < float(args.baryon_min_mass):
                continue
            for mmax in meson_max_values:
                if float(mmax) < float(args.meson_min_mass):
                    continue
                rows.append(
                    _run_one(
                        out_dir=out_dir,
                        species_csv=species_csv,
                        profile_name=profile_name,
                        baryon_min_mass=float(args.baryon_min_mass),
                        baryon_max_mass=float(bmax),
                        meson_min_mass=float(args.meson_min_mass),
                        meson_max_mass=float(mmax),
                        max_decay_passes=int(args.max_decay_passes),
                        res_events_per_res=int(args.res_events_per_res),
                        res_workers=max(int(args.res_workers), 1),
                        seed=int(args.seed),
                        prior_strange_lambda=float(args.prior_strange_lambda),
                        prior_hyperon_lambda=float(args.prior_hyperon_lambda),
                        fit_weight_mode=str(args.fit_weight_mode),
                        fit_balance_xf_bins=int(args.fit_balance_xf_bins),
                        fit_balance_pt_bins=int(args.fit_balance_pt_bins),
                        skip_external=bool(args.skip_external),
                    )
                )

    rows.sort(
        key=lambda r: (
            str(r["profile"]),
            float(r["baryon_max_mass"]),
            float(r["meson_max_mass"]),
            int(r["max_decay_passes"]),
        )
    )

    _write_outputs(
        out_dir=out_dir,
        rows=rows,
        species_csv=species_csv,
        profiles=profiles,
        baryon_max_values=baryon_max_values,
        meson_max_values=meson_max_values,
    )


if __name__ == "__main__":
    main()
