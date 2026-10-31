#!/usr/bin/env python3
"""Run CRMC EPOS-LHC and export final-state pi- samples as benchmark CSV.

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


def _iter_piminus_from_hepmc(hepmc_path: Path, pmax: float, beta_cms: float) -> Iterable[Tuple[float, float]]:
    gamma_cms = 1.0 / math.sqrt(max(1.0 - beta_cms * beta_cms, 1e-16))
    with hepmc_path.open(errors="replace") as fh:
        for raw in fh:
            if not raw.startswith("P "):
                continue
            cols = raw.split()
            # HepMC2 particle line:
            # P barcode pdgid px py pz E m status ...
            if len(cols) < 9:
                continue
            try:
                pdgid = int(cols[2])
                status = int(cols[8])
            except ValueError:
                continue
            if pdgid != -211 or status != 1:
                continue
            px = float(cols[3])
            py = float(cols[4])
            pz = float(cols[5])
            energy = float(cols[6])
            # Convert longitudinal momentum to the center-of-mass frame for fixed-target runs.
            pz_cms = gamma_cms * (pz - beta_cms * energy)
            pt = math.hypot(px, py)
            xf = pz_cms / pmax
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


def _beta_cms_from_collinear_momenta(
    p_proj: float,
    p_targ: float,
    m_proj: float = MPROTON_GEV,
    m_targ: float = MPROTON_GEV,
) -> float:
    e_proj = math.sqrt(p_proj * p_proj + m_proj * m_proj)
    e_targ = math.sqrt(p_targ * p_targ + m_targ * m_targ)
    denom = e_proj + e_targ
    if denom <= 0.0:
        return 0.0
    beta = (p_proj + p_targ) / denom
    if beta >= 1.0:
        return 1.0 - 1e-12
    if beta <= -1.0:
        return -1.0 + 1e-12
    return beta


def _render_crmc_param(base_text: str, tabs_dir: Path) -> str:
    tabs = str(tabs_dir.resolve())
    if not tabs.endswith("/"):
        tabs = tabs + "/"
    repl = {
        "fdpmjet path": f"{tabs}",
        "fqgsjet dat": f"{tabs}qgsjet.dat",
        "fqgsjet ncs": f"{tabs}qgsjet.ncs",
        "fqgsjetII03 dat": f"{tabs}qgsdat-II-03.lzma",
        "fqgsjetII03 ncs": f"{tabs}sectnu-II-03",
        "fqgsjetII dat": f"{tabs}qgsdat-II-04.lzma",
        "fqgsjetII ncs": f"{tabs}sectnu-II-04",
        "fname initl": f"{tabs}epos.initl",
        "fname iniev": f"{tabs}epos.iniev",
        "fname inirj": f"{tabs}epos.inirj",
        "fname inics": f"{tabs}epos.inics",
        "fname inihy": f"{tabs}epos.inihy",
    }
    out_lines = []
    for line in base_text.splitlines():
        replaced = False
        for key, value in repl.items():
            if line.strip().startswith(key):
                out_lines.append(f"{key}  {value}")
                replaced = True
                break
        if not replaced:
            out_lines.append(line)
    return "\n".join(out_lines) + "\n"


def _run_crmc_epos(
    crmc_install: Path,
    hepmc_lib: Path,
    events: int,
    seed: int,
    sqrt_s: float,
    use_sqrts: bool,
    projectile_momentum: float,
    target_momentum: float,
    projectile_id: int,
    target_id: int,
    model: int,
    tag: str,
) -> Tuple[Path, Path]:
    exe = crmc_install / "bin/crmc"
    if not exe.exists():
        raise FileNotFoundError(f"CRMC executable not found: {exe}")

    run_dir = crmc_install / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    hepmc_path = run_dir / f"{tag}.hepmc"
    log_path = run_dir / f"{tag}.log"
    param_path = run_dir / f"{tag}.crmc.param"

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        f"{crmc_install / 'lib'}:{hepmc_lib}:"
        f"{env.get('LD_LIBRARY_PATH', '')}"
    )

    base_param = crmc_install / "crmc.param"
    if not base_param.exists():
        raise FileNotFoundError(f"CRMC parameter file not found: {base_param}")
    param_path.write_text(_render_crmc_param(base_param.read_text(errors="replace"), crmc_install / "tabs"))

    cmd = [
        str(exe),
        "-o",
        "hepmc",
        f"-n{events}",
        f"-m{model}",
        f"-i{projectile_id}",
        f"-I{target_id}",
        "-c",
        str(param_path),
        f"-s{seed}",
        "-f",
        str(hepmc_path),
    ]
    if use_sqrts:
        cmd.insert(3, f"-S{sqrt_s}")
    else:
        cmd.insert(3, f"-p{projectile_momentum}")
        cmd.insert(4, f"-P{target_momentum}")

    with log_path.open("w") as log_fh:
        subprocess.run(
            cmd,
            cwd=str(crmc_install),
            env=env,
            check=True,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )

    return hepmc_path, log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CRMC EPOS-LHC and export pi- CSV.")
    parser.add_argument(
        "--crmc-install",
        default=str(REPO_ROOT / "external" / "crmc" / "install"),
        help="Path containing bin/crmc and lib/",
    )
    parser.add_argument(
        "--hepmc-lib",
        default=str(REPO_ROOT / "external" / "hepmc2-install" / "lib"),
        help="Directory containing libHepMC*.so",
    )
    parser.add_argument("--events", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=20260329)
    parser.add_argument(
        "--sqrt-s",
        type=float,
        default=17.27,
        help="Used when --use-sqrts is enabled (collider-style input).",
    )
    parser.add_argument(
        "--use-sqrts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use CRMC -S sqrt(s) mode. Default is fixed-target momentum mode.",
    )
    parser.add_argument("--projectile-momentum", type=float, default=158.0, help="Projectile pz [GeV/c]")
    parser.add_argument("--target-momentum", type=float, default=0.0, help="Target pz [GeV/c] (0 for fixed target)")
    parser.add_argument("--projectile-id", type=int, default=1, help="CRMC projectile id shortcut (1=proton)")
    parser.add_argument("--target-id", type=int, default=1, help="CRMC target id shortcut (1=proton)")
    parser.add_argument("--model", type=int, default=0, help="CRMC model id (0=EPOS-LHC)")
    parser.add_argument(
        "--out-csv",
        default="",
    )
    parser.add_argument("--run-description", default="mixing2006_example_default")
    parser.add_argument(
        "--tag",
        default="epos_lhc_pp_s17p27",
        help="Short run tag used for HepMC/log filenames",
    )
    args = parser.parse_args()

    crmc_install = Path(args.crmc_install).resolve()
    hepmc_lib = Path(args.hepmc_lib).resolve()
    out_csv = Path(args.out_csv).resolve() if args.out_csv else (RUNS_ROOT / args.run_description / "epos_lhc_pp_s17p27_piminus.csv").resolve()

    sqrt_s_eff = (
        args.sqrt_s
        if args.use_sqrts
        else _sqrt_s_from_collinear_momenta(args.projectile_momentum, args.target_momentum)
    )
    beta_cms = (
        0.0
        if args.use_sqrts
        else _beta_cms_from_collinear_momenta(args.projectile_momentum, args.target_momentum)
    )
    pmax = sqrt_s_eff / 2.0
    hepmc_path, log_path = _run_crmc_epos(
        crmc_install=crmc_install,
        hepmc_lib=hepmc_lib,
        events=args.events,
        seed=args.seed,
        sqrt_s=args.sqrt_s,
        use_sqrts=args.use_sqrts,
        projectile_momentum=args.projectile_momentum,
        target_momentum=args.target_momentum,
        projectile_id=args.projectile_id,
        target_id=args.target_id,
        model=args.model,
        tag=args.tag,
    )
    nrows = _write_csv(_iter_piminus_from_hepmc(hepmc_path, pmax=pmax, beta_cms=beta_cms), out_csv)

    print(f"CRMC run log: {log_path}")
    print(f"HepMC output: {hepmc_path}")
    print(f"sqrt(s)_effective [GeV]: {sqrt_s_eff:.6f}")
    print(f"beta_cms: {beta_cms:.8f}")
    print(f"Output CSV: {out_csv}")
    print(f"pi- rows written: {nrows}")


if __name__ == "__main__":
    main()
