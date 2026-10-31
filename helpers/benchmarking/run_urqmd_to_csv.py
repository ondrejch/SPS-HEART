#!/usr/bin/env python3
"""Run UrQMD pp at 158 GeV/c and export selected hadron samples as CSV.

CSV schema:
    xf,pt,value

where `value` is 1 for each sampled particle row.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Tuple

from helpers.benchmarking.particle_definitions import canonical_species_keys, resolve_species

MPROTON_GEV = 0.9382720813
REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "dev" / "00runs"


def _float_token(token: str) -> float:
    return float(token.replace("D", "E"))


def _iter_species_from_f13(
    f13_path: Path,
    pmax: float,
    selected: Dict[Tuple[int, int], str],
) -> Iterable[Tuple[str, float, float]]:
    lines = f13_path.read_text(errors="replace").splitlines()
    i = 0
    n = len(lines)
    while i < n:
        cols = lines[i].split()
        if len(cols) != 2:
            i += 1
            continue
        try:
            npart = int(cols[0])
            _ = int(cols[1])
        except ValueError:
            i += 1
            continue
        if npart < 0 or npart > 200000:
            i += 1
            continue

        event_has_collision = True
        if i + 1 < n:
            stat_cols = lines[i + 1].split()
            if stat_cols:
                try:
                    event_has_collision = int(stat_cols[0]) > 0
                except ValueError:
                    event_has_collision = True

        i += 2
        for _ in range(npart):
            if i >= n:
                return
            pcols = lines[i].split()
            i += 1
            if not event_has_collision or len(pcols) < 12:
                continue
            try:
                ityp = int(pcols[9])
                iso3 = int(pcols[10])
            except ValueError:
                continue
            species_tag = selected.get((ityp, iso3))
            if species_tag is None:
                continue
            px = _float_token(pcols[5])
            py = _float_token(pcols[6])
            pz = _float_token(pcols[7])
            pt = math.hypot(px, py)
            xf = pz / pmax
            yield species_tag, xf, pt


def _write_species_csvs(
    samples: Iterable[Tuple[str, float, float]],
    outputs: Dict[str, Path],
) -> Dict[str, int]:
    counts = {tag: 0 for tag in outputs}
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    files = {tag: path.open("w", newline="") for tag, path in outputs.items()}
    try:
        writers = {tag: csv.writer(fh) for tag, fh in files.items()}
        for w in writers.values():
            w.writerow(["xf", "pt", "value"])
        for species_tag, xf, pt in samples:
            writer = writers.get(species_tag)
            if writer is None:
                continue
            writer.writerow([f"{xf:.10g}", f"{pt:.10g}", "1"])
            counts[species_tag] += 1
    finally:
        for fh in files.values():
            fh.close()
    return counts


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
        outputs[tag] = (RUNS_ROOT / run_description / f"urqmd_pp158_{tag}.csv").resolve()
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


def _run_urqmd(
    urqmd_dir: Path,
    events: int,
    seed: int,
    beam_momentum: float,
    impact_min: float,
    impact_max: float,
    time_fm: float,
    tag: str,
) -> Tuple[Path, Path]:
    exe = urqmd_dir / "urqmd.x86_64"
    if not exe.exists():
        raise FileNotFoundError(f"UrQMD executable not found: {exe}")

    run_dir = urqmd_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    in_name = f"{tag}.in"
    f13_name = f"{tag}.f13"
    log_name = f"{tag}.log"

    input_path = run_dir / in_name
    f13_path = run_dir / f13_name
    log_path = run_dir / log_name

    lines = [
        "pro 1 1",
        "tar 1 1",
        f"nev {events}",
        f"IMP {impact_min} {impact_max}",
        f"plb {beam_momentum}",
        f"rsd {seed}",
        f"tim {time_fm} {time_fm}",
    ]
    lines.extend(["f14", "f15", "f16", "f19", "f20", "xxx", ""])
    input_path.write_text("\n".join(lines), encoding="utf-8")

    env = os.environ.copy()
    env["ftn09"] = str(Path("runs") / in_name)
    env["ftn13"] = str(Path("runs") / f13_name)
    env["ftn14"] = str(Path("runs") / f"{tag}.f14")
    env["ftn15"] = str(Path("runs") / f"{tag}.f15")
    env["ftn16"] = str(Path("runs") / f"{tag}.f16")
    env["ftn19"] = str(Path("runs") / f"{tag}.f19")
    env["ftn20"] = str(Path("runs") / f"{tag}.f20")

    with log_path.open("w", encoding="utf-8") as log_fh:
        subprocess.run(
            [str(exe)],
            cwd=str(urqmd_dir),
            env=env,
            check=True,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
    return f13_path, log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UrQMD pp@158 GeV/c and export selected hadron CSV.")
    parser.add_argument(
        "--species",
        default="pi-",
        help="Species key, CSV list, or 'all' (pi-,pi+,k+,k-,p,pbar).",
    )
    parser.add_argument(
        "--urqmd-dir",
        default=str(REPO_ROOT / "dev" / "external" / "urqmd-4.0"),
        help="Path containing urqmd.x86_64",
    )
    parser.add_argument("--events", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=20260329)
    parser.add_argument("--beam-momentum", type=float, default=158.0, help="GeV/c in lab frame")
    parser.add_argument("--impact-min", type=float, default=0.0)
    parser.add_argument("--impact-max", type=float, default=1.0)
    parser.add_argument("--time-fm", type=float, default=200.0)
    parser.add_argument(
        "--target-momentum",
        type=float,
        default=0.0,
        help="Target momentum [GeV/c] for xF normalization (0 for fixed target).",
    )
    parser.add_argument(
        "--sqrt-s",
        type=float,
        default=0.0,
        help="Override sqrt(s) [GeV] for xF normalization; if <=0, derive from beam/target momenta.",
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
        help="Run tag used for UrQMD run files; defaults to urqmd_pp158_<species|all>",
    )
    args = parser.parse_args()

    species_keys = _parse_species_keys(args.species)
    species_specs = [resolve_species(key) for key in species_keys]
    species_tags = [spec.tag for spec in species_specs]
    selected = {(spec.urqmd_ityp, spec.urqmd_iso3): spec.tag for spec in species_specs}
    urqmd_dir = Path(args.urqmd_dir).resolve()

    outputs = _resolve_output_paths(
        species_tags=species_tags,
        run_description=args.run_description,
        out_csv=args.out_csv,
        out_csv_template=args.out_csv_template,
    )

    tag_suffix = species_tags[0] if len(species_tags) == 1 else "all"
    tag = args.tag or f"urqmd_pp158_{tag_suffix}"

    sqrt_s_eff = (
        args.sqrt_s
        if args.sqrt_s > 0.0
        else _sqrt_s_from_collinear_momenta(args.beam_momentum, args.target_momentum)
    )
    pmax = sqrt_s_eff / 2.0
    f13_path, log_path = _run_urqmd(
        urqmd_dir=urqmd_dir,
        events=args.events,
        seed=args.seed,
        beam_momentum=args.beam_momentum,
        impact_min=args.impact_min,
        impact_max=args.impact_max,
        time_fm=args.time_fm,
        tag=tag,
    )
    counts = _write_species_csvs(
        _iter_species_from_f13(
            f13_path,
            pmax=pmax,
            selected=selected,
        ),
        outputs,
    )

    print(f"species selection: {', '.join(species_tags)}")
    print(f"UrQMD run log: {log_path}")
    print(f"UrQMD f13: {f13_path}")
    print(f"sqrt(s)_effective [GeV]: {sqrt_s_eff:.6f}")
    for spec in species_specs:
        out_csv = outputs[spec.tag]
        nrows = counts.get(spec.tag, 0)
        print(
            f"{spec.name} (pdg={spec.pdg}, urqmd=({spec.urqmd_ityp},{spec.urqmd_iso3})) "
            f"-> {out_csv} rows={nrows}"
        )


if __name__ == "__main__":
    main()
