#!/usr/bin/env python3
"""Run PYTHIA8 SoftQCD and export selected hadron samples as benchmark CSV.

CSV schema:
    xf,pt,value

where `value` is 1 for each sampled particle row.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

from helpers.benchmarking.particle_definitions import canonical_species_keys, resolve_species

MPROTON_GEV = 0.9382720813
REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "dev" / "00runs"


def _parse_species_keys(species_arg: str) -> list[str]:
    raw = species_arg.strip().lower()
    if raw in {"all", "*"}:
        return canonical_species_keys()
    keys = [item.strip() for item in species_arg.split(",") if item.strip()]
    if not keys:
        raise ValueError("No species selected.")
    seen: set[str] = set()
    ordered: list[str] = []
    for key in keys:
        spec = resolve_species(key)
        if spec.tag in seen:
            continue
        seen.add(spec.tag)
        ordered.append(key)
    return ordered


def _resolve_output_paths(
    *,
    species_tags: list[str],
    run_description: str,
    out_csv: str,
    out_csv_template: str,
) -> Dict[str, Path]:
    outputs: Dict[str, Path] = {}
    if out_csv_template:
        for tag in species_tags:
            outputs[tag] = Path(out_csv_template.format(species_tag=tag)).resolve()
        return outputs

    if out_csv:
        if "{species_tag}" in out_csv:
            for tag in species_tags:
                outputs[tag] = Path(out_csv.format(species_tag=tag)).resolve()
            return outputs
        if len(species_tags) != 1:
            raise ValueError(
                "--out-csv must include '{species_tag}' when exporting multiple species."
            )
        outputs[species_tags[0]] = Path(out_csv).resolve()
        return outputs

    for tag in species_tags:
        outputs[tag] = (RUNS_ROOT / run_description / f"pythia8_softqcd_pp158_{tag}.csv").resolve()
    return outputs


def _sqrt_s_from_collinear_momenta(
    p_proj: float,
    p_targ: float,
    m_proj: float = MPROTON_GEV,
    m_targ: float = MPROTON_GEV,
) -> float:
    e_proj = math.sqrt(p_proj * p_proj + m_proj * m_proj)
    e_targ = math.sqrt(p_targ * p_targ + m_targ * m_targ)
    s = (e_proj + e_targ) ** 2 - (p_proj + p_targ) ** 2
    return math.sqrt(max(s, 0.0))


def _write_species_csvs(
    samples_by_tag: Dict[str, List[Tuple[float, float]]],
    outputs: Dict[str, Path],
) -> Dict[str, int]:
    counts = {tag: 0 for tag in outputs}
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    for tag, path in outputs.items():
        pts = samples_by_tag.get(tag, [])
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["xf", "pt", "value"])
            for xf, pt in pts:
                writer.writerow([f"{xf:.10g}", f"{pt:.10g}", "1"])
                counts[tag] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PYTHIA8 SoftQCD and export selected hadron CSV.")
    parser.add_argument(
        "--species",
        default="pi-",
        help="Species key, CSV list, or 'all' (pi-,pi+,k+,k-,p,pbar).",
    )
    parser.add_argument("--events", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=20260329)
    parser.add_argument(
        "--beam-momentum",
        type=float,
        default=158.0,
        help="Projectile momentum [GeV/c] for fixed-target sqrt(s) normalization.",
    )
    parser.add_argument(
        "--target-momentum",
        type=float,
        default=0.0,
        help="Target momentum [GeV/c] (0 for fixed-target).",
    )
    parser.add_argument(
        "--sqrt-s",
        type=float,
        default=0.0,
        help="Override sqrt(s) [GeV]. If <=0, derive from beam/target momenta.",
    )
    parser.add_argument(
        "--out-csv",
        default="",
        help="Output CSV path for one species, or template containing {species_tag}.",
    )
    parser.add_argument(
        "--out-csv-template",
        default="",
        help="Template path containing {species_tag}; preferred for multispecies export.",
    )
    parser.add_argument("--run-description", default="local_default")
    parser.add_argument(
        "--tag",
        default="",
        help="Run tag for logs; defaults to pythia8_softqcd_pp158_<species|all>.",
    )
    args = parser.parse_args()

    # Import lazily so non-PYTHIA workflows can still import this module.
    import pythia8mc as p8

    species_keys = _parse_species_keys(args.species)
    species_specs = [resolve_species(key) for key in species_keys]
    species_tags = [spec.tag for spec in species_specs]
    selected = {spec.pdg: spec.tag for spec in species_specs}

    outputs = _resolve_output_paths(
        species_tags=species_tags,
        run_description=args.run_description,
        out_csv=args.out_csv,
        out_csv_template=args.out_csv_template,
    )

    tag_suffix = species_tags[0] if len(species_tags) == 1 else "all"
    run_tag = args.tag or f"pythia8_softqcd_pp158_{tag_suffix}"

    sqrt_s_eff = (
        float(args.sqrt_s)
        if float(args.sqrt_s) > 0.0
        else _sqrt_s_from_collinear_momenta(float(args.beam_momentum), float(args.target_momentum))
    )
    pmax = sqrt_s_eff / 2.0

    p = p8.Pythia()
    p.readString("Print:quiet = on")
    p.readString("Random:setSeed = on")
    p.readString(f"Random:seed = {int(args.seed)}")
    p.readString("Beams:idA = 2212")
    p.readString("Beams:idB = 2212")
    p.readString(f"Beams:eCM = {sqrt_s_eff:.10g}")
    p.readString("SoftQCD:all = on")
    if not p.init():
        raise RuntimeError("PYTHIA8 initialization failed.")

    samples: Dict[str, List[Tuple[float, float]]] = {tag: [] for tag in species_tags}
    accepted = 0
    requested = max(int(args.events), 0)
    for _ in range(requested):
        if not p.next():
            continue
        accepted += 1
        ev = p.event
        for i in range(ev.size()):
            par = ev[i]
            if not par.isFinal():
                continue
            tag = selected.get(int(par.id()))
            if tag is None:
                continue
            xf = float(par.pz()) / pmax
            pt = float(par.pT())
            if not (math.isfinite(xf) and math.isfinite(pt)):
                continue
            samples[tag].append((xf, pt))

    if accepted == 0:
        raise RuntimeError("PYTHIA8 produced zero accepted events.")

    counts = _write_species_csvs(samples, outputs)
    print(f"run tag: {run_tag}")
    print(f"species selection: {', '.join(species_tags)}")
    print(f"requested events: {requested}")
    print(f"accepted events: {accepted}")
    print(f"sqrt(s) [GeV]: {sqrt_s_eff:.10g}")
    for tag in species_tags:
        print(f"{tag}: {counts[tag]} rows -> {outputs[tag]}")


if __name__ == "__main__":
    main()

