#!/usr/bin/env python3
"""Run UrQMD pp at 158 GeV/c and export pi- samples as benchmark CSV.

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
from typing import Iterable, Tuple

MPROTON_GEV = 0.9382720813
REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_ROOT = REPO_ROOT / "00runs"


def _float_token(token: str) -> float:
    return float(token.replace("D", "E"))


def _iter_piminus_from_f13(f13_path: Path, pmax: float) -> Iterable[Tuple[float, float]]:
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
            _ = int(cols[1])  # output time in fm/c
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
        # Skip collision-stat line.
        i += 2
        for _ in range(npart):
            if i >= n:
                return
            pcols = lines[i].split()
            i += 1
            if not event_has_collision:
                continue
            if len(pcols) < 12:
                continue
            try:
                ityp = int(pcols[9])
                iso3 = int(pcols[10])
            except ValueError:
                continue
            # UrQMD pion triplet: ityp=101, iso3 in {-2,0,+2}
            if ityp != 101 or iso3 != -2:
                continue
            px = _float_token(pcols[5])
            py = _float_token(pcols[6])
            pz = _float_token(pcols[7])
            pt = math.hypot(px, py)
            xf = pz / pmax
            yield xf, pt


def _write_csv(samples: Iterable[Tuple[float, float]], out_csv: Path) -> int:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["xf", "pt", "value"])
        for xf, pt in samples:
            writer.writerow([f"{xf:.10g}", f"{pt:.10g}", "1"])
            count += 1
    return count


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
    lines.extend(
        [
            "f14",
            "f15",
            "f16",
            "f19",
            "f20",
            "xxx",
            "",
        ]
    )
    input_path.write_text("\n".join(lines))

    env = os.environ.copy()
    env["ftn09"] = str(Path("runs") / in_name)
    env["ftn13"] = str(Path("runs") / f13_name)
    env["ftn14"] = str(Path("runs") / f"{tag}.f14")
    env["ftn15"] = str(Path("runs") / f"{tag}.f15")
    env["ftn16"] = str(Path("runs") / f"{tag}.f16")
    env["ftn19"] = str(Path("runs") / f"{tag}.f19")
    env["ftn20"] = str(Path("runs") / f"{tag}.f20")

    with log_path.open("w") as log_fh:
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
    parser = argparse.ArgumentParser(description="Run UrQMD pp@158 GeV/c and export pi- CSV.")
    parser.add_argument(
        "--urqmd-dir",
        default=str(REPO_ROOT / "external" / "urqmd-4.0"),
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
    )
    parser.add_argument("--run-description", default="mixing2006_example_default")
    parser.add_argument(
        "--tag",
        default="urqmd_pp158",
        help="Short run tag used for run/input/output filenames",
    )
    args = parser.parse_args()

    urqmd_dir = Path(args.urqmd_dir).resolve()
    out_csv = Path(args.out_csv).resolve() if args.out_csv else (RUNS_ROOT / args.run_description / "urqmd_pp158_piminus.csv").resolve()

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
        tag=args.tag,
    )
    nrows = _write_csv(_iter_piminus_from_f13(f13_path, pmax=pmax), out_csv)

    print(f"UrQMD run log: {log_path}")
    print(f"UrQMD f13: {f13_path}")
    print(f"sqrt(s)_effective [GeV]: {sqrt_s_eff:.6f}")
    print(f"Output CSV: {out_csv}")
    print(f"pi- rows written: {nrows}")


if __name__ == "__main__":
    main()
