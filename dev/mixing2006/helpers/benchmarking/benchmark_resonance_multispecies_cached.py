#!/usr/bin/env python3
"""Run shared-cache resonance benchmarks across multiple species.

This driver builds resonance component tracks once, stores them as
`(pdg, xF, pT)` cache files, and reuses them for per-species NNLS fits.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from helpers.benchmarking.particle_definitions import resolve_species

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "00runs"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_species_csv(species_csv: str) -> List[str]:
    keys = [s.strip() for s in species_csv.split(",") if s.strip()]
    if not keys:
        raise ValueError("No species provided.")
    for key in keys:
        resolve_species(key)
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Shared-cache multi-species resonance benchmark.")
    parser.add_argument("--run-description", default="resonance_multispecies_cached_default")
    parser.add_argument("--species", default="pi-,pi+,p,pbar", help="CSV species list")
    parser.add_argument("--out-dir", default="", help="Output directory (default: 00runs/<run_description>)")
    parser.add_argument("--cache-dir", default="", help="Cache directory (default: <out_dir>/resonance_component_cache)")
    parser.add_argument("--seed", type=int, default=20260329)
    parser.add_argument("--beam-momentum", type=float, default=158.0)
    parser.add_argument("--res-events-per-res", type=int, default=30000)
    parser.add_argument("--res-max-decay-passes", type=int, default=8)
    parser.add_argument("--res-workers", type=int, default=1, help="Per-job worker processes for resonance generation.")
    parser.add_argument(
        "--res-fragmentation-mode",
        choices=("projectile", "target", "both"),
        default="both",
        help="Event-level resonance source mode for each component.",
    )
    parser.add_argument(
        "--res-disable-regge-tower",
        action="store_true",
        help="Disable effective high-mass N* tower components.",
    )
    parser.add_argument("--res-cache-force-rebuild", action="store_true")
    parser.add_argument("--mixing2006-dir", default=str(REPO_ROOT))
    parser.add_argument("--xf-min", type=float, default=0.0)
    parser.add_argument("--xf-max", type=float, default=0.4)
    parser.add_argument("--pt-min", type=float, default=0.05)
    parser.add_argument("--pt-max", type=float, default=1.5)
    parser.add_argument("--train-frac", type=float, default=0.7)
    args = parser.parse_args()

    species_keys = _parse_species_csv(args.species)
    out_dir = Path(args.out_dir) if args.out_dir else (RUNS_ROOT / args.run_description)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (out_dir / "resonance_component_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    benchmark_script = REPO_ROOT / "helpers" / "benchmarking" / "benchmark_sps_species.py"

    jobs = []
    for idx, species_key in enumerate(species_keys):
        spec = resolve_species(species_key)
        out_prefix = f"na49_benchmark_{spec.tag}_sharedcache"

        # Warm cache once with global component scope, then use species scope.
        cache_scope = "all" if idx == 0 else "species"

        cmd = [
            sys.executable,
            str(benchmark_script),
            "--species",
            species_key,
            "--beam-momentum",
            str(args.beam_momentum),
            "--seed",
            str(args.seed),
            "--disable-pythia",
            "--res-events-per-res",
            str(args.res_events_per_res),
            "--res-max-decay-passes",
            str(args.res_max_decay_passes),
            "--res-workers",
            str(max(int(args.res_workers), 1)),
            "--res-fragmentation-mode",
            str(args.res_fragmentation_mode),
            "--mixing2006-dir",
            str(args.mixing2006_dir),
            "--xf-min",
            str(args.xf_min),
            "--xf-max",
            str(args.xf_max),
            "--pt-min",
            str(args.pt_min),
            "--pt-max",
            str(args.pt_max),
            "--train-frac",
            str(args.train_frac),
            "--out-dir",
            str(out_dir),
            "--out-prefix",
            out_prefix,
            "--res-cache-dir",
            str(cache_dir),
            "--res-cache-scope",
            cache_scope,
        ]
        if idx == 0 and args.res_cache_force_rebuild:
            cmd.append("--res-cache-force-rebuild")
        if args.res_disable_regge_tower:
            cmd.append("--res-disable-regge-tower")

        print(f"[{idx + 1}/{len(species_keys)}] run {species_key} (cache_scope={cache_scope})", flush=True)
        subprocess.run(cmd, check=True)

        jobs.append(
            {
                "species": species_key,
                "species_tag": spec.tag,
                "cache_scope": cache_scope,
                "summary_json": str(out_dir / f"{out_prefix}_summary.json"),
                "details_npz": str(out_dir / f"{out_prefix}_details.npz"),
                "command": cmd,
            }
        )

    manifest = {
        "created_at": _utc_now(),
        "run_description": args.run_description,
        "out_dir": str(out_dir),
        "cache_dir": str(cache_dir),
        "seed": args.seed,
        "beam_momentum": args.beam_momentum,
        "res_events_per_res": args.res_events_per_res,
        "res_max_decay_passes": args.res_max_decay_passes,
        "res_workers": max(int(args.res_workers), 1),
        "res_fragmentation_mode": str(args.res_fragmentation_mode),
        "species": species_keys,
        "jobs": jobs,
    }
    manifest_path = out_dir / "resonance_multispecies_cached_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
