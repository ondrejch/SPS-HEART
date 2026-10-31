#!/usr/bin/env python3
"""Investigate local seagull dips around xF ~ 0.12/0.125."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import helpers.benchmarking.benchmark_sps_species as bss
from helpers.benchmarking.sps_core import load_multispecies_details_npz


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "00runs"
MODEL_SUFFIX = {
    "Data": "y",
    "Resonance_shared": "pred_res",
    "PYTHIA8_external": "pred_pythia",
    "EPOS_external": "pred_epos",
    "UrQMD_external": "pred_urqmd",
}

_PDG_MASS_GEV: Dict[int, float] = {
    211: 0.13957,
    321: 0.493677,
    2212: 0.9382720813,
}


def _pdg_mass_gev(pdg_id: int) -> float:
    return float(_PDG_MASS_GEV.get(abs(int(pdg_id)), _PDG_MASS_GEV[211]))


def _nearest_xf_bins(xf_vals: np.ndarray, targets: List[float]) -> List[float]:
    x_unique = np.unique(np.round(np.asarray(xf_vals, dtype=float), 10))
    out: List[float] = []
    for t in targets:
        idx = int(np.argmin(np.abs(x_unique - float(t))))
        out.append(float(x_unique[idx]))
    return out


def _bin_widths_from_centers(pt_vals: np.ndarray) -> np.ndarray:
    centers = np.asarray(np.round(pt_vals, 10), dtype=float)
    unique = np.unique(centers)
    unique.sort()
    edges = bss._edges_from_centers(unique)
    idx = np.searchsorted(unique, centers)
    return np.diff(edges)[idx]


def _mean_pt_for_bin(
    pt_vals: np.ndarray,
    y_vals: np.ndarray,
    sigma_vals: np.ndarray | None,
    *,
    mass_gev: float,
    xf_value: float,
    sqrt_s: float,
) -> Tuple[float, float]:
    if pt_vals.size <= 0:
        return float("nan"), float("nan")
    dp = _bin_widths_from_centers(pt_vals)
    p_l = float(xf_value) * (0.5 * float(sqrt_s))
    energy = np.sqrt(np.maximum(float(mass_gev) ** 2 + float(p_l) ** 2 + np.asarray(pt_vals, dtype=float) ** 2, 1e-30))
    kernel = (np.asarray(pt_vals, dtype=float) / energy) * np.asarray(dp, dtype=float)
    w = np.asarray(y_vals, dtype=float) * kernel
    den = float(np.sum(w))
    if (not np.isfinite(den)) or den <= 0.0:
        return float("nan"), float("nan")
    num = float(np.sum(np.asarray(pt_vals, dtype=float) * w))
    mu = float(num / den)
    mu_err = float("nan")
    if sigma_vals is not None:
        s = np.maximum(np.asarray(sigma_vals, dtype=float), 0.0)
        dmu_dy = kernel * ((np.asarray(pt_vals, dtype=float) * den) - num) / max(den * den, 1e-30)
        var_mu = float(np.sum((dmu_dy * s) ** 2))
        if np.isfinite(var_mu) and var_mu >= 0.0:
            mu_err = float(np.sqrt(var_mu))
    return mu, mu_err


def _local_common_window(xf: np.ndarray, pt: np.ndarray, xbins: List[float]) -> Tuple[float, float]:
    pmins: List[float] = []
    pmaxs: List[float] = []
    for xv in xbins:
        m = np.isclose(np.asarray(xf, dtype=float), float(xv), atol=1e-10)
        pv = np.asarray(pt[m], dtype=float)
        if pv.size <= 0:
            return float("nan"), float("nan")
        pmins.append(float(np.min(pv)))
        pmaxs.append(float(np.max(pv)))
    return float(max(pmins)), float(min(pmaxs))


def _dip_depth(v1: float, v2: float, v3: float) -> float:
    if not (np.isfinite(v1) and np.isfinite(v2) and np.isfinite(v3)):
        return float("nan")
    return float(v2 - 0.5 * (v1 + v3))


def _dip_depth_err(e1: float, e2: float, e3: float) -> float:
    if not (np.isfinite(e1) and np.isfinite(e2) and np.isfinite(e3)):
        return float("nan")
    var = float(e2 * e2 + 0.25 * (e1 * e1 + e3 * e3))
    if (not np.isfinite(var)) or var < 0.0:
        return float("nan")
    return float(np.sqrt(var))


def _model_prediction(z: np.lib.npyio.NpzFile, tag: str, model: str) -> np.ndarray | None:
    suffix = MODEL_SUFFIX.get(model)
    if suffix is None:
        return None
    key = f"{tag}_{suffix}"
    if key not in z.files:
        return None
    return np.asarray(z[key], dtype=float)


def _model_prediction_sigma(
    z: np.lib.npyio.NpzFile,
    tag: str,
    model: str,
) -> np.ndarray | None:
    if model == "Data":
        key = f"{tag}_sigma"
    elif model == "Resonance_shared":
        key = f"{tag}_pred_res_sigma"
    elif model == "PYTHIA8_external":
        key = f"{tag}_pred_pythia_sigma"
    elif model == "EPOS_external":
        key = f"{tag}_pred_epos_sigma"
    elif model == "UrQMD_external":
        key = f"{tag}_pred_urqmd_sigma"
    else:
        return None
    if key not in z.files:
        return None
    return np.asarray(z[key], dtype=float)


def _format_float(x: float) -> str:
    if not np.isfinite(float(x)):
        return "nan"
    return f"{float(x):.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Investigate seagull dip around xF~0.12/0.125.")
    parser.add_argument("--run-description", default="gw_nstar_ladder_hi100x_u31_dynamic_2026-03-30/umax2500_hagedorn_t0250_n05")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--out-csv", default="")
    parser.add_argument("--out-md", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if str(args.run_dir).strip() else (RUNS_ROOT / args.run_description)
    summary_path = run_dir / "multispecies_constrained_summary.json"
    details_path = run_dir / "multispecies_constrained_details.npz"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary JSON: {summary_path}")
    if not details_path.exists():
        raise FileNotFoundError(f"Missing details NPZ: {details_path}")

    out_csv = Path(args.out_csv) if str(args.out_csv).strip() else (run_dir / "seagull_dip_xf0125_analysis.csv")
    out_md = Path(args.out_md) if str(args.out_md).strip() else (run_dir / "SEAGULL_DIP_XF0125_ANALYSIS.md")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    species_map = summary.get("species", {})
    if not isinstance(species_map, dict):
        raise RuntimeError("Invalid summary format: missing species map.")
    derived = summary.get("derived", {})
    if isinstance(derived, dict) and np.isfinite(float(derived.get("sqrt_s_from_beam_momentum", float("nan")))):
        sqrt_s = float(derived.get("sqrt_s_from_beam_momentum"))
    else:
        beam = float(summary.get("config", {}).get("beam_momentum", 158.0))
        sqrt_s = float(bss._sqrt_s_from_fixed_target_beam_momentum(beam))

    rows: List[Dict[str, object]] = []
    with load_multispecies_details_npz(details_path, summary_payload=summary) as z:
        for _, row in species_map.items():
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", ""))
            tag = str(row.get("tag", ""))
            if not name or not tag:
                continue
            pdg = int(row.get("pdg", 211))
            mass_gev = _pdg_mass_gev(pdg)
            xf_key = f"{tag}_xf"
            pt_key = f"{tag}_pt"
            if xf_key not in z.files or pt_key not in z.files:
                continue

            xf = np.asarray(z[xf_key], dtype=float)
            pt = np.asarray(z[pt_key], dtype=float)
            targets = [0.1, 0.125, 0.15] if name in {"K+", "K-", "p", "pbar"} else [0.1, 0.12, 0.15]
            xbins = _nearest_xf_bins(xf, targets)
            pt_lo, pt_hi = _local_common_window(xf, pt, xbins)

            for model in MODEL_SUFFIX.keys():
                pred = _model_prediction(z, tag, model)
                if pred is None:
                    continue
                pred_sigma = _model_prediction_sigma(z, tag, model)
                vals_raw: List[float] = []
                vals_raw_err: List[float] = []
                vals_common: List[float] = []
                vals_common_err: List[float] = []
                for xv in xbins:
                    m = np.isclose(xf, float(xv), atol=1e-10)
                    p_bin = pt[m]
                    y_bin = pred[m]
                    s_bin = None if pred_sigma is None else pred_sigma[m]
                    mu_raw, err_raw = _mean_pt_for_bin(
                        p_bin,
                        y_bin,
                        s_bin,
                        mass_gev=mass_gev,
                        xf_value=float(xv),
                        sqrt_s=float(sqrt_s),
                    )
                    vals_raw.append(mu_raw)
                    vals_raw_err.append(err_raw)

                    m_common = m & (pt >= float(pt_lo) - 1e-12) & (pt <= float(pt_hi) + 1e-12)
                    s_common = None if pred_sigma is None else pred_sigma[m_common]
                    mu_common, err_common = _mean_pt_for_bin(
                        pt[m_common],
                        pred[m_common],
                        s_common,
                        mass_gev=mass_gev,
                        xf_value=float(xv),
                        sqrt_s=float(sqrt_s),
                    )
                    vals_common.append(mu_common)
                    vals_common_err.append(err_common)

                rows.append(
                    {
                        "species": name,
                        "model": model,
                        "xf_left": float(xbins[0]),
                        "xf_center": float(xbins[1]),
                        "xf_right": float(xbins[2]),
                        "common_pt_min": float(pt_lo),
                        "common_pt_max": float(pt_hi),
                        "meanpt_raw_left": float(vals_raw[0]),
                        "meanpt_raw_center": float(vals_raw[1]),
                        "meanpt_raw_right": float(vals_raw[2]),
                        "meanpt_raw_left_err": float(vals_raw_err[0]),
                        "meanpt_raw_center_err": float(vals_raw_err[1]),
                        "meanpt_raw_right_err": float(vals_raw_err[2]),
                        "meanpt_common_left": float(vals_common[0]),
                        "meanpt_common_center": float(vals_common[1]),
                        "meanpt_common_right": float(vals_common[2]),
                        "meanpt_common_left_err": float(vals_common_err[0]),
                        "meanpt_common_center_err": float(vals_common_err[1]),
                        "meanpt_common_right_err": float(vals_common_err[2]),
                        "dip_raw": _dip_depth(vals_raw[0], vals_raw[1], vals_raw[2]),
                        "dip_raw_err": _dip_depth_err(vals_raw_err[0], vals_raw_err[1], vals_raw_err[2]),
                        "dip_common": _dip_depth(vals_common[0], vals_common[1], vals_common[2]),
                        "dip_common_err": _dip_depth_err(vals_common_err[0], vals_common_err[1], vals_common_err[2]),
                    }
                )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "species",
                "model",
                "xf_left",
                "xf_center",
                "xf_right",
                "common_pt_min",
                "common_pt_max",
                "meanpt_raw_left",
                "meanpt_raw_center",
                "meanpt_raw_right",
                "meanpt_raw_left_err",
                "meanpt_raw_center_err",
                "meanpt_raw_right_err",
                "meanpt_common_left",
                "meanpt_common_center",
                "meanpt_common_right",
                "meanpt_common_left_err",
                "meanpt_common_center_err",
                "meanpt_common_right_err",
                "dip_raw",
                "dip_raw_err",
                "dip_common",
                "dip_common_err",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    lines: List[str] = []
    lines.append("# Seagull Dip Study Near xF ~ 0.12/0.125")
    lines.append("")
    lines.append(f"- run_dir: `{run_dir}`")
    lines.append(f"- input summary: `{summary_path}`")
    lines.append(f"- input details: `{details_path}`")
    lines.append("")
    lines.append("Dip definition:")
    lines.append("- `dip = meanpt(center) - 0.5*(meanpt(left)+meanpt(right))`")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- negative dip: center bin is lower than local neighbor average.")
    lines.append("- if `dip_common` is much closer to zero than `dip_raw`, acceptance mismatch dominates.")
    lines.append("")

    species_order = {"pi-": 0, "pi+": 1, "K+": 2, "K-": 3, "p": 4, "pbar": 5}
    model_order = {k: i for i, k in enumerate(MODEL_SUFFIX.keys())}
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            species_order.get(str(r["species"]), 99),
            model_order.get(str(r["model"]), 99),
        ),
    )

    lines.append("| species | model | xF triplet | local common pT [GeV/c] | dip_raw | dip_common |")
    lines.append("|---|---|---|---|---:|---:|")
    for r in rows_sorted:
        triplet = f"{float(r['xf_left']):.3f},{float(r['xf_center']):.3f},{float(r['xf_right']):.3f}"
        pwin = f"{float(r['common_pt_min']):.3f}..{float(r['common_pt_max']):.3f}"
        lines.append(
            "| "
            + f"{r['species']} | {r['model']} | {triplet} | {pwin} | "
            + f"{_format_float(float(r['dip_raw']))} +/- {_format_float(float(r['dip_raw_err']))} | "
            + f"{_format_float(float(r['dip_common']))} +/- {_format_float(float(r['dip_common_err']))} |"
        )
    lines.append("")
    lines.append("Artifacts:")
    lines.append(f"- CSV: `{out_csv}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
