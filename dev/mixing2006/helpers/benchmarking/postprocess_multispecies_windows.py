#!/usr/bin/env python3
"""Postprocess integrated ratio windows and seagull pT-coverage diagnostics.

The details NPZ input must be a current-schema
``multispecies_constrained_details.npz`` artifact written by the present
producer. It is opened through the shared pickle-free loader
``load_multispecies_details_npz`` from helpers.benchmarking.sps_core, which
validates the payload against the companion summary JSON; run directories
written by older producers fail fast there with regeneration guidance
instead of producing partial outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import helpers.benchmarking.benchmark_sps_species as bss
from helpers.benchmarking.benchmark_multispecies_constrained import _pdg_mass_gev
from helpers.benchmarking.sps_core import load_multispecies_details_npz


# Policy: external generators carry one fitted normalization per species,
# so their cross-species ratios are not generator yield predictions.  They
# are fenced out of every particle-ratio artifact here, mirroring the
# omission policy of benchmark_multispecies_constrained.
# _build_particle_ratio_summary; the baselines remain available as
# species-by-species shape-only fits elsewhere.
MODEL_ORDER: List[str] = [
    "Data",
    "Resonance_shared",
]


def _window_label(xf_min: float, xf_max: float) -> str:
    lo = float(min(xf_min, xf_max))
    hi = float(max(xf_min, xf_max))
    return f"xf_{lo:.3f}_{hi:.3f}"


def _parse_windows(spec: str) -> List[Tuple[str, float, float]]:
    out: List[Tuple[str, float, float]] = []
    seen: set[str] = set()
    for chunk in str(spec).split(","):
        txt = chunk.strip()
        if not txt:
            continue
        if ":" not in txt:
            raise ValueError(f"Invalid xF window '{txt}'. Expected min:max")
        a, b = txt.split(":", 1)
        lo = float(a.strip())
        hi = float(b.strip())
        if hi < lo:
            lo, hi = hi, lo
        lab = _window_label(lo, hi)
        if lab in seen:
            continue
        seen.add(lab)
        out.append((lab, lo, hi))
    if not out:
        raise ValueError("No valid windows parsed.")
    return out


def _species_info(summary: Dict[str, object]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    species = summary.get("species", {})
    if not isinstance(species, dict):
        return out
    for key, row in species.items():
        if not isinstance(row, dict):
            continue
        out[str(key)] = {
            "name": str(row.get("name", key)),
            "tag": str(row.get("tag", key)),
            "pdg": str(row.get("pdg", "")),
        }
    return out


def _prediction(z: np.lib.npyio.NpzFile, tag: str, model: str) -> np.ndarray | None:
    if model == "Data":
        k = f"{tag}_y"
    elif model == "Resonance_shared":
        k = f"{tag}_pred_res"
    else:
        # External {tag}_pred_* arrays are deliberately not read here:
        # cross-species external ratios built from independently
        # per-species-fitted normalizations must never reach a ratio
        # artifact (see MODEL_ORDER policy note).
        return None
    if k not in z.files:
        return None
    return np.asarray(z[k], dtype=float)


def _window_areas(xf: np.ndarray, pt: np.ndarray, xf_min: float, xf_max: float) -> np.ndarray:
    xedges = bss._edges_from_centers(np.asarray(xf, dtype=float))
    pedges = bss._edges_from_centers(np.asarray(pt, dtype=float))
    ix, ip = bss._grid_indices(xf, pt, xedges, pedges)
    x_left = xedges[ix]
    x_right = xedges[ix + 1]
    p_width = np.diff(pedges)[ip]
    lo = float(min(xf_min, xf_max))
    hi = float(max(xf_min, xf_max))
    overlap = np.maximum(0.0, np.minimum(x_right, hi) - np.maximum(x_left, lo))
    return overlap * p_width


def _window_yield_kernel(
    *,
    xf: np.ndarray,
    pt: np.ndarray,
    mass_gev: float,
    sqrt_s_gev: float,
    xf_min: float,
    xf_max: float,
) -> np.ndarray:
    """Yield kernel w=(pT/E)*overlap(delta_xF)*delta_pT per point.

    Integrates the invariant cross section f like a yield, with
    E=sqrt(m^2+pL^2+pT^2) and pL=xF*sqrt(s)/2, matching the kernel used by
    benchmark_multispecies_constrained._invariant_integration_weights.
    """
    p_l = np.asarray(xf, dtype=float) * (0.5 * float(sqrt_s_gev))
    energy = np.sqrt(
        np.maximum(float(mass_gev) ** 2 + p_l**2 + np.asarray(pt, dtype=float) ** 2, 1e-30)
    )
    jacobian = np.asarray(pt, dtype=float) / energy
    return jacobian * _window_areas(xf, pt, xf_min, xf_max)


def _sqrt_s_from_summary(summary: Dict[str, object]) -> float:
    derived = summary.get("derived", {})
    if isinstance(derived, dict):
        val = derived.get("sqrt_s_from_beam_momentum")
        if isinstance(val, (int, float)) and float(val) > 0.0:
            return float(val)
    # Fall back to the NA49 beam-momentum setting used by the paper pipeline.
    return float(bss._sqrt_s_from_fixed_target_beam_momentum(158.0))


def _ratio_defs(species_map: Dict[str, Dict[str, str]]) -> List[Tuple[str, List[str], List[str]]]:
    by_name = {v["name"].strip().lower(): k for k, v in species_map.items()}
    out: List[Tuple[str, List[str], List[str]]] = []
    if "pi+" in by_name and "pi-" in by_name:
        out.append(("pi+/pi-", ["pi+"], ["pi-"]))
    if "k+" in by_name and "k-" in by_name:
        out.append(("k+/k-", ["k+"], ["k-"]))
    if "pbar" in by_name and "p" in by_name:
        out.append(("pbar/p", ["pbar"], ["p"]))
    if all(x in by_name for x in ("k+", "k-", "pi+", "pi-")):
        out.append(("(k++k-)/(pi++pi-)", ["k+", "k-"], ["pi+", "pi-"]))
    return out


def _name_to_key(species_map: Dict[str, Dict[str, str]], name: str) -> str | None:
    target = str(name).strip().lower()
    for k, row in species_map.items():
        if row["name"].strip().lower() == target:
            return k
    return None


def _write_ratios(
    *,
    z: np.lib.npyio.NpzFile,
    species_map: Dict[str, Dict[str, str]],
    windows: List[Tuple[str, float, float]],
    out_csv: Path,
    sqrt_s_gev: float,
) -> Dict[str, object]:
    ratio_defs = _ratio_defs(species_map)
    kernel_by_window: Dict[str, Dict[str, np.ndarray]] = {}
    for win_label, win_lo, win_hi in windows:
        kernel_by_window[win_label] = {}
        for key, row in species_map.items():
            tag = row["tag"]
            xf = np.asarray(z[f"{tag}_xf"], dtype=float)
            pt = np.asarray(z[f"{tag}_pt"], dtype=float)
            mass_gev = _pdg_mass_gev(int(row["pdg"])) if str(row.get("pdg", "")).strip() else 0.0
            kernel_by_window[win_label][key] = _window_yield_kernel(
                xf=xf,
                pt=pt,
                mass_gev=mass_gev,
                sqrt_s_gev=sqrt_s_gev,
                xf_min=win_lo,
                xf_max=win_hi,
            )

    values_by_window: Dict[str, Dict[str, Dict[str, float | None]]] = {}
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["window", "xf_min", "xf_max", "ratio", "data", "resonance_shared"]
        )
        for win_label, win_lo, win_hi in windows:
            values_by_window[win_label] = {}
            for ratio_name, nums, dens in ratio_defs:
                row_vals: Dict[str, float | None] = {}
                for model in MODEL_ORDER:
                    num = 0.0
                    den = 0.0
                    ok = True
                    for sp_name in nums:
                        k = _name_to_key(species_map, sp_name)
                        if k is None:
                            ok = False
                            break
                        tag = species_map[k]["tag"]
                        pred = _prediction(z, tag, model)
                        if pred is None:
                            ok = False
                            break
                        num += float(np.sum(pred * kernel_by_window[win_label][k]))
                    if ok:
                        for sp_name in dens:
                            k = _name_to_key(species_map, sp_name)
                            if k is None:
                                ok = False
                                break
                            tag = species_map[k]["tag"]
                            pred = _prediction(z, tag, model)
                            if pred is None:
                                ok = False
                                break
                            den += float(np.sum(pred * kernel_by_window[win_label][k]))
                    if (not ok) or den <= 0.0 or (not np.isfinite(den)):
                        row_vals[model] = None
                    else:
                        row_vals[model] = float(num / den)

                values_by_window[win_label][ratio_name] = row_vals
                writer.writerow(
                    [
                        win_label,
                        win_lo,
                        win_hi,
                        ratio_name,
                        row_vals.get("Data"),
                        row_vals.get("Resonance_shared"),
                    ]
                )

    return {
        "windows": {w: {"xf_min": lo, "xf_max": hi} for w, lo, hi in windows},
        "values_by_window": values_by_window,
        "integration": "sum(y * (pT/E) * overlap(delta_xF,[xf_min,xf_max]) * delta_pT) over benchmark points (yield kernel)",
    }


def _write_seagull_coverage(*, z: np.lib.npyio.NpzFile, species_map: Dict[str, Dict[str, str]], out_csv: Path) -> None:
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["species", "xf", "n_pt", "pt_min", "pt_max", "pt_span"])
        for key, row in sorted(species_map.items(), key=lambda kv: kv[1]["name"]):
            tag = row["tag"]
            species_name = row["name"]
            xf = np.asarray(z[f"{tag}_xf"], dtype=float)
            pt = np.asarray(z[f"{tag}_pt"], dtype=float)
            x_unique = np.unique(np.round(xf, 10))
            for xv in x_unique:
                m = np.isclose(xf, xv, atol=1e-10)
                p = np.unique(np.round(pt[m], 10))
                if p.size <= 0:
                    continue
                pmin = float(np.min(p))
                pmax = float(np.max(p))
                writer.writerow([species_name, float(xv), int(p.size), pmin, pmax, pmax - pmin])


def main() -> None:
    parser = argparse.ArgumentParser(description="Postprocess windowed ratios and seagull pT coverage.")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument(
        "--details-npz",
        required=True,
        help=(
            "Details NPZ of the constrained multispecies run "
            "(multispecies_constrained_details.npz). Must follow the current "
            "producer schema (schema_version marker); pre-schema artifacts "
            "fail fast and must be regenerated with the current producer."
        ),
    )
    parser.add_argument(
        "--xf-windows",
        default="0.0:0.4,0.0:0.1,0.1:0.2,0.2:0.3,0.3:0.4,0.0:0.2,0.2:0.4",
        help="Comma-separated xF windows (min:max).",
    )
    parser.add_argument("--out-ratio-csv", default="")
    parser.add_argument("--out-coverage-csv", default="")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    details_path = Path(args.details_npz)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    species_map = _species_info(summary)
    windows = _parse_windows(args.xf_windows)
    sqrt_s_gev = _sqrt_s_from_summary(summary)

    run_dir = summary_path.parent
    out_ratio_csv = (
        Path(args.out_ratio_csv)
        if str(args.out_ratio_csv).strip()
        else (run_dir / "multispecies_constrained_particle_ratios_xf_windows.csv")
    )
    out_coverage_csv = (
        Path(args.out_coverage_csv)
        if str(args.out_coverage_csv).strip()
        else (run_dir / "multispecies_constrained_seagull_pt_coverage.csv")
    )
    out_json = (
        Path(args.out_json)
        if str(args.out_json).strip()
        else (run_dir / "multispecies_constrained_postprocess_windows.json")
    )

    with load_multispecies_details_npz(details_path, summary_payload=summary) as z:
        ratio_payload = _write_ratios(
            z=z,
            species_map=species_map,
            windows=windows,
            out_csv=out_ratio_csv,
            sqrt_s_gev=sqrt_s_gev,
        )
        _write_seagull_coverage(z=z, species_map=species_map, out_csv=out_coverage_csv)

    payload = {
        "summary_json": str(summary_path),
        "details_npz": str(details_path),
        "out_ratio_csv": str(out_ratio_csv),
        "out_coverage_csv": str(out_coverage_csv),
        "ratio_payload": ratio_payload,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

