#!/usr/bin/env python3
"""Benchmark NA49 pi- spectra against MC and resonance hypotheses.

Runs:
- PYTHIA8 (via `pythia8mc` Python bindings, if available)
- Resonance-mixture hypothesis using `na49_mixing` generators
- Optional external model files (EPOS/UrQMD) if provided as CSV

Data source:
- SPS HADRONS pi- table `xsect_piminus.txt`, passed to `--na49-data`.
  In full development checkouts the table lives under
  `spshadrons.web.cern.ch/data/xsect_piminus.txt`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import nnls

import na49_mixing
from na49_mixing.decays import build_soft_resonance_decays
from na49_mixing.event import Event
from na49_mixing.generator import Generator
from na49_mixing.reaction import Reaction

MPROTON_GEV = 0.9382720813
# Resolve paths against the installed na49_mixing package so the script works
# both in the standalone repository layout and in nested monorepo checkouts.
PACKAGE_ROOT = Path(na49_mixing.__file__).resolve().parents[1]
RUNS_ROOT = PACKAGE_ROOT / "00runs"


def _sqrt_s_from_fixed_target_beam_momentum(beam_p_lab: float, mproj: float = MPROTON_GEV, mtarg: float = MPROTON_GEV) -> float:
    e_beam = math.sqrt(beam_p_lab * beam_p_lab + mproj * mproj)
    s = mproj * mproj + mtarg * mtarg + 2.0 * mtarg * e_beam
    return math.sqrt(max(s, 0.0))


def _read_na49_points(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs: List[float] = []
    pts: List[float] = []
    ys: List[float] = []
    es: List[float] = []
    with path.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) < 4:
                continue
            xf, pt, val, err = map(float, cols[:4])
            xs.append(xf)
            pts.append(pt)
            ys.append(val)
            es.append(err)
    return np.asarray(xs), np.asarray(pts), np.asarray(ys), np.asarray(es)


def _edges_from_centers(vals: np.ndarray) -> np.ndarray:
    vals = np.unique(np.asarray(vals))
    vals.sort()
    if vals.size == 1:
        return np.array([vals[0] - 0.5, vals[0] + 0.5], dtype=float)
    mids = 0.5 * (vals[:-1] + vals[1:])
    first = vals[0] - 0.5 * (vals[1] - vals[0])
    last = vals[-1] + 0.5 * (vals[-1] - vals[-2])
    return np.concatenate([[first], mids, [last]])


def _grid_indices(xf: np.ndarray, pt: np.ndarray, xedges: np.ndarray, pedges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ix = np.digitize(xf, xedges) - 1
    ip = np.digitize(pt, pedges) - 1
    return ix, ip


def _counts_on_grid(
    sample_xf: np.ndarray,
    sample_pt: np.ndarray,
    xedges: np.ndarray,
    pedges: np.ndarray,
    ix_data: np.ndarray,
    ip_data: np.ndarray,
) -> np.ndarray:
    h, _, _ = np.histogram2d(sample_xf, sample_pt, bins=[xedges, pedges])
    return h[ix_data, ip_data]


def _fit_single_scale(train_pred: np.ndarray, train_y: np.ndarray, train_sigma: np.ndarray) -> float:
    w = 1.0 / np.maximum(train_sigma, 1e-12) ** 2
    num = np.sum(w * train_pred * train_y)
    den = np.sum(w * train_pred * train_pred)
    if den <= 0:
        return 0.0
    return float(num / den)


def _chi2(pred: np.ndarray, y: np.ndarray, sigma: np.ndarray, n_params: int) -> Tuple[float, float]:
    r = (pred - y) / np.maximum(sigma, 1e-12)
    chi2 = float(np.sum(r * r))
    ndof = max(len(y) - n_params, 1)
    return chi2, chi2 / ndof


def _run_pythia(
    n_events: int,
    seed: int,
    sqrt_s: float,
    xedges: np.ndarray,
    pedges: np.ndarray,
    ix_data: np.ndarray,
    ip_data: np.ndarray,
) -> np.ndarray:
    import pythia8mc as p8

    p = p8.Pythia()
    p.readString("Print:quiet = on")
    p.readString("Random:setSeed = on")
    p.readString(f"Random:seed = {seed}")
    p.readString("Beams:idA = 2212")
    p.readString("Beams:idB = 2212")
    # Use equivalent center-of-mass energy derived from fixed-target beam momentum.
    p.readString(f"Beams:eCM = {sqrt_s:.8f}")
    p.readString("SoftQCD:all = on")
    ok = p.init()
    if not ok:
        raise RuntimeError("PYTHIA failed to initialize.")

    xf_vals: List[float] = []
    pt_vals: List[float] = []
    pmax = sqrt_s / 2.0

    accepted = 0
    for _ in range(n_events):
        if not p.next():
            continue
        accepted += 1
        ev = p.event
        for i in range(ev.size()):
            par = ev[i]
            if not par.isFinal():
                continue
            if par.id() != -211:
                continue
            xf = par.pz() / pmax
            pt = par.pT()
            xf_vals.append(xf)
            pt_vals.append(pt)

    if accepted == 0:
        raise RuntimeError("PYTHIA generated zero accepted events.")

    return _counts_on_grid(
        np.asarray(xf_vals),
        np.asarray(pt_vals),
        xedges,
        pedges,
        ix_data,
        ip_data,
    )


def _run_resonance_components(
    base_dir: Path,
    n_events_per_res: int,
    seed: int,
    xedges: np.ndarray,
    pedges: np.ndarray,
    ix_data: np.ndarray,
    ip_data: np.ndarray,
) -> Tuple[List[str], np.ndarray]:
    reaction = Reaction(158.0)
    rng = np.random.default_rng(seed)

    # (label, gen_path)
    specs = [
        (
            "delta++",
            base_dir / "na49_mixing/generators/michal/deltas-to-pion/delta++.gen",
        ),
        (
            "delta+",
            base_dir / "na49_mixing/generators/michal/deltas-to-pion/delta+.gen",
        ),
        (
            "delta0",
            base_dir / "na49_mixing/generators/michal/deltas-to-pion/delta0.gen",
        ),
        (
            "delta-",
            base_dir / "na49_mixing/generators/michal/deltas-to-pion/delta-.gen",
        ),
        (
            "rho0",
            base_dir / "na49_mixing/generators/michal/mesons-to-pion/rho0.gen",
        ),
        (
            "omega",
            base_dir / "na49_mixing/generators/michal/mesons-to-pion/omega.gen",
        ),
    ]

    labels: List[str] = []
    mat_cols: List[np.ndarray] = []

    for label, gen_file in specs:
        if not gen_file.exists():
            continue
        gen = Generator(reaction=reaction, rng=rng)
        gen.set_all(str(gen_file))
        ev = Event(rng=rng)
        ev._decay_chain = build_soft_resonance_decays(rng=rng)

        xf_vals: List[float] = []
        pt_vals: List[float] = []
        for _ in range(n_events_per_res):
            part = gen.generate()
            if part is None:
                continue
            ev.add_particle(part)
            ev.decay()
            for p in ev:
                if p.get_id() == -211:
                    xf_vals.append(p.get_xf())
                    pt_vals.append(p.get_pt())
            ev.delete()

        if not xf_vals:
            continue

        counts = _counts_on_grid(
            np.asarray(xf_vals),
            np.asarray(pt_vals),
            xedges,
            pedges,
            ix_data,
            ip_data,
        )
        labels.append(label)
        mat_cols.append(counts)

    if not mat_cols:
        raise RuntimeError("No resonance components produced any pi- samples.")
    return labels, np.column_stack(mat_cols)


def _load_external_grid_counts(
    csv_path: Path,
    xedges: np.ndarray,
    pedges: np.ndarray,
    ix_data: np.ndarray,
    ip_data: np.ndarray,
) -> np.ndarray:
    # Expected columns: xf,pt,value
    xs: List[float] = []
    ps: List[float] = []
    vs: List[float] = []
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            xs.append(float(row["xf"]))
            ps.append(float(row["pt"]))
            vs.append(float(row["value"]))
    xs = np.asarray(xs)
    ps = np.asarray(ps)
    vs = np.asarray(vs)
    hsum = np.zeros((len(xedges) - 1, len(pedges) - 1), dtype=float)
    # deposit values in bins
    ix = np.digitize(xs, xedges) - 1
    ip = np.digitize(ps, pedges) - 1
    for i, j, v in zip(ix, ip, vs):
        if 0 <= i < hsum.shape[0] and 0 <= j < hsum.shape[1]:
            hsum[i, j] += v
    return hsum[ix_data, ip_data]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark NA49 pi- spectra vs model hypotheses.")
    parser.add_argument(
        "--na49-data",
        required=True,
        help="Required path to the NA49 pi- invariant cross section table; "
             "use the SPS HADRONS file 'xsect_piminus.txt', shipped as "
             "spshadrons.web.cern.ch/data/xsect_piminus.txt in full "
             "development checkouts",
    )
    parser.add_argument("--out-dir", default="", help="Output directory; defaults to 00runs/<run_description>")
    parser.add_argument("--run-description", default="mixing2006_example_default")
    parser.add_argument("--seed", type=int, default=20260329)
    parser.add_argument("--pythia-events", type=int, default=120000)
    parser.add_argument("--res-events-per-res", type=int, default=30000)
    parser.add_argument("--xf-min", type=float, default=0.0)
    parser.add_argument("--xf-max", type=float, default=0.4)
    parser.add_argument("--pt-min", type=float, default=0.05)
    parser.add_argument("--pt-max", type=float, default=1.5)
    parser.add_argument("--beam-momentum", type=float, default=158.0, help="Fixed-target proton beam momentum [GeV/c]")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--epos-csv", default="", help="Optional external EPOS CSV (xf,pt,value)")
    parser.add_argument("--urqmd-csv", default="", help="Optional external UrQMD CSV (xf,pt,value)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (RUNS_ROOT / args.run_description)
    out_dir.mkdir(parents=True, exist_ok=True)

    xf, pt, y, sigma = _read_na49_points(Path(args.na49_data))
    mask = (
        (xf >= args.xf_min)
        & (xf <= args.xf_max)
        & (pt >= args.pt_min)
        & (pt <= args.pt_max)
        & (sigma > 0.0)
    )
    xf = xf[mask]
    pt = pt[mask]
    y = y[mask]
    sigma = sigma[mask]

    xedges = _edges_from_centers(xf)
    pedges = _edges_from_centers(pt)
    ix_data, ip_data = _grid_indices(xf, pt, xedges, pedges)

    n = len(y)
    sqrt_s_eff = _sqrt_s_from_fixed_target_beam_momentum(args.beam_momentum)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_train = int(args.train_frac * n)
    train_idx = np.sort(perm[:n_train])
    test_idx = np.sort(perm[n_train:])

    summary: List[Dict[str, float | int | str]] = []

    # ------------------------------------------------------------------
    # PYTHIA
    # ------------------------------------------------------------------
    pythia_pred = None
    pythia_status = "not_run"
    try:
        pythia_counts = _run_pythia(
            args.pythia_events,
            args.seed + 7,
            sqrt_s_eff,
            xedges,
            pedges,
            ix_data,
            ip_data,
        )
        alpha = _fit_single_scale(
            pythia_counts[train_idx],
            y[train_idx],
            sigma[train_idx],
        )
        pythia_pred = alpha * pythia_counts
        chi2_tr, chi2dof_tr = _chi2(pythia_pred[train_idx], y[train_idx], sigma[train_idx], n_params=1)
        chi2_te, chi2dof_te = _chi2(pythia_pred[test_idx], y[test_idx], sigma[test_idx], n_params=1)
        summary.append(
            {
                "model": "PYTHIA8_SoftQCD_all",
                "n_params": 1,
                "train_chi2": chi2_tr,
                "train_chi2_dof": chi2dof_tr,
                "test_chi2": chi2_te,
                "test_chi2_dof": chi2dof_te,
            }
        )
        pythia_status = "ok"
    except Exception as exc:
        summary.append({"model": "PYTHIA8_SoftQCD_all", "status": f"failed: {exc}"})
        pythia_status = f"failed: {exc}"

    # ------------------------------------------------------------------
    # Resonance-mixture hypothesis
    # ------------------------------------------------------------------
    labels, M = _run_resonance_components(
        PACKAGE_ROOT,
        args.res_events_per_res,
        args.seed + 19,
        xedges,
        pedges,
        ix_data,
        ip_data,
    )
    A = M[train_idx] / sigma[train_idx, None]
    b = y[train_idx] / sigma[train_idx]
    w, _ = nnls(A, b)
    pred_res = M @ w
    n_eff = int(np.sum(w > 1e-12))
    chi2_tr, chi2dof_tr = _chi2(pred_res[train_idx], y[train_idx], sigma[train_idx], n_params=max(n_eff, 1))
    chi2_te, chi2dof_te = _chi2(pred_res[test_idx], y[test_idx], sigma[test_idx], n_params=max(n_eff, 1))
    summary.append(
        {
            "model": "Resonance_NNLS",
            "n_params": n_eff,
            "train_chi2": chi2_tr,
            "train_chi2_dof": chi2dof_tr,
            "test_chi2": chi2_te,
            "test_chi2_dof": chi2dof_te,
        }
    )

    # ------------------------------------------------------------------
    # Optional external models
    # ------------------------------------------------------------------
    for tag, csv_path in (("EPOS_external", args.epos_csv), ("UrQMD_external", args.urqmd_csv)):
        if not csv_path:
            summary.append({"model": tag, "status": "not_provided"})
            continue
        p = Path(csv_path)
        if not p.exists():
            summary.append({"model": tag, "status": f"missing_file:{csv_path}"})
            continue
        raw = _load_external_grid_counts(p, xedges, pedges, ix_data, ip_data)
        alpha = _fit_single_scale(raw[train_idx], y[train_idx], sigma[train_idx])
        pred = alpha * raw
        chi2_tr, chi2dof_tr = _chi2(pred[train_idx], y[train_idx], sigma[train_idx], n_params=1)
        chi2_te, chi2dof_te = _chi2(pred[test_idx], y[test_idx], sigma[test_idx], n_params=1)
        summary.append(
            {
                "model": tag,
                "n_params": 1,
                "train_chi2": chi2_tr,
                "train_chi2_dof": chi2dof_tr,
                "test_chi2": chi2_te,
                "test_chi2_dof": chi2dof_te,
            }
        )

    # Save outputs
    summary_path = out_dir / "na49_benchmark_summary.json"
    details_path = out_dir / "na49_benchmark_details.npz"
    with summary_path.open("w") as fh:
        json.dump(
            {
                "config": vars(args),
                "derived": {"sqrt_s_from_beam_momentum": sqrt_s_eff},
                "n_points": int(n),
                "train_points": int(len(train_idx)),
                "test_points": int(len(test_idx)),
                "resonance_components": labels,
                "resonance_weights": {lbl: float(val) for lbl, val in zip(labels, w)},
                "pythia_status": pythia_status,
                "summary": summary,
            },
            fh,
            indent=2,
        )

    np.savez_compressed(
        details_path,
        xf=xf,
        pt=pt,
        y=y,
        sigma=sigma,
        train_idx=train_idx,
        test_idx=test_idx,
        resonance_pred=pred_res,
        resonance_matrix=M,
        resonance_weights=w,
        pythia_pred=np.array([] if pythia_pred is None else pythia_pred),
    )

    print(f"Saved summary: {summary_path}")
    print(f"Saved details: {details_path}")
    print("Model summary:")
    for item in summary:
        print(item)


if __name__ == "__main__":
    main()
