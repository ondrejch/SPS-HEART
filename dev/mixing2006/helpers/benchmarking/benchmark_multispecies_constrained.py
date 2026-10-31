#!/usr/bin/env python3
"""Constrained multi-species resonance benchmark with shared NNLS coefficients.

This script fits one global non-negative resonance-weight vector across multiple
species simultaneously (pi+/-, K+/-, p, pbar). It reuses the shared component
cache and then scores resonance and optional external models per species.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize, nnls

import helpers.benchmarking.benchmark_sps_species as bss
from helpers.benchmarking.particle_definitions import (
    available_data_systems,
    canonical_species_keys,
    na49_table_for,
    resolve_species,
)
from helpers.benchmarking.run_provenance import (
    RUN_PROVENANCE_FILENAME,
    check_inputs_against_manifest,
    collect_source_record,
    lock_file_records,
    manifest_crosscheck_problems,
    manifest_input_hashes,
    validate_run_provenance,
    write_run_provenance,
)
from helpers.benchmarking.sps_core import (
    ACTIVE_WEIGHT_THRESHOLD,
    DETAILS_NPZ_SCHEMA_VERSION,
    FULL_RANK_STAMP,
    RANK_DEFICIENT_STAMP,
    condition_number_2norm,
    enforce_fit_design_rank,
    validate_fit_design,
    validate_fit_prior_rows,
    validate_fit_result,
    validate_fit_sigma,
    validate_fit_table9_fraction_arrays,
    validate_fit_target,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "00runs"
DEFAULT_TABLE9_JSON = REPO_ROOT / "helpers" / "benchmarking" / "config" / "fischer2022_table9_pi_minus.json"
FIT_OBJECTIVES: Tuple[str, ...] = ("nnls_weighted", "gaussian_point_mle", "lognormal_point_mle")

# Environment variables recorded in each run's provenance when set. The
# allowlist is explicit and closed; the whole environment is never dumped
# into a run record.
PROVENANCE_ENV_ALLOWLIST: Tuple[str, ...] = (
    "NA49_RUN_SOURCE_ARCHIVE",
    "NA49_CLUSTER_GW_TASKS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)

# SPS HADRONS world-data anchor at sqrt(s) ~= 17.2 GeV (NA49 beam momentum setting):
# <n_K+> = 0.227, <n_K-> = 0.130
DEFAULT_KMINUS_OVER_KPLUS = 0.130 / 0.227

_PDG_MASS_GEV: Dict[int, float] = {
    211: 0.13957,       # charged pion
    321: 0.493677,      # charged kaon
    2212: 0.9382720813, # proton
}


def _pdg_mass_gev(pdg_id: int) -> float:
    return float(_PDG_MASS_GEV.get(abs(int(pdg_id)), _PDG_MASS_GEV[211]))


def validate_data_system_generation(
    data_system: str,
    pa_a: int,
    *,
    allow_mismatch: bool = False,
) -> None:
    """Refuse p+p generation scored on nuclear tables, and the inverse.

    ``--data-system`` only selects NA49 files; ``--pa-a`` selects the
    wounded-nucleon generator. A pPb table with no nucleus would silently
    fit elementary p+p components to nuclear spectra.
    """
    if allow_mismatch:
        return
    ds = str(data_system)
    a = int(pa_a)
    if ds in ("pC", "pPb") and a <= 0:
        raise ValueError(
            f"--data-system {ds} requires --pa-a > 0 so resonance "
            "components are wounded-nucleon p+A events, not p+p"
        )
    if ds == "pp" and a > 0:
        raise ValueError(
            "--pa-a generates p+A events; pair it with --data-system pC "
            "or pPb, or pass --allow-system-mismatch"
        )


def _as_float_or_none(val: float) -> float | None:
    if val is None:
        return None
    x = float(val)
    if not np.isfinite(x):
        return None
    return x


def _label_match_mask(labels: List[str], selector: str) -> np.ndarray:
    if selector == "delta0":
        return np.asarray([lbl == "delta0" for lbl in labels], dtype=bool)
    if selector == "delta-":
        return np.asarray([lbl == "delta-" for lbl in labels], dtype=bool)
    if selector == "n1440":
        return np.asarray([lbl.startswith("n1440") for lbl in labels], dtype=bool)
    if selector == "n1520":
        return np.asarray([lbl.startswith("n1520") for lbl in labels], dtype=bool)
    if selector == "n1680":
        return np.asarray([lbl.startswith("n1680") for lbl in labels], dtype=bool)
    if selector == "eta":
        return np.asarray([lbl == "eta" for lbl in labels], dtype=bool)
    if selector == "rho0":
        return np.asarray([lbl == "rho0" for lbl in labels], dtype=bool)
    if selector == "rho-":
        return np.asarray([lbl == "rho_minus" for lbl in labels], dtype=bool)
    if selector == "omega":
        return np.asarray([lbl == "omega" for lbl in labels], dtype=bool)
    if selector == "f2":
        return np.asarray([lbl in ("f2", "f2_1270") for lbl in labels], dtype=bool)
    if selector == "rho3^0":
        return np.asarray([lbl in ("rho3_zero", "rho3_zero_1690") for lbl in labels], dtype=bool)
    if selector == "rho3-":
        return np.asarray([lbl in ("rho3_minus", "rho3_minus_1690") for lbl in labels], dtype=bool)
    if selector == "f4":
        return np.asarray([lbl in ("f4", "f4_2050") for lbl in labels], dtype=bool)
    return np.zeros(len(labels), dtype=bool)


def _is_tower_component_label(label: str) -> bool:
    txt = str(label).strip().lower()
    return "nstar_eff" in txt


def _build_nstar_ladder_smooth_prior(*, component_labels: List[str]) -> Dict[str, object]:
    """Build adjacent-level smoothness rows for effective N* ladders.

    Each row enforces `w(level_i) - w(level_{i+1}) ~= 0` within a fixed ladder
    branch (baryon/antibaryon and charge channel), leaving absolute scale free.
    """

    pat = re.compile(r"^(anti_)?nstar_effL(\d+)_([A-Za-z0-9+-]+)$", re.IGNORECASE)
    groups: Dict[Tuple[str, str], List[Tuple[int, int, str]]] = {}
    for idx, lbl in enumerate(component_labels):
        m = pat.match(str(lbl).strip())
        if m is None:
            continue
        anti_prefix = "anti_" if m.group(1) else ""
        level = int(m.group(2))
        charge = str(m.group(3)).lower()
        gkey = (anti_prefix, charge)
        groups.setdefault(gkey, []).append((level, int(idx), str(lbl)))

    rows: List[np.ndarray] = []
    targets: List[float] = []
    meta: List[Dict[str, object]] = []
    n_comp = len(component_labels)
    for (anti_prefix, charge), items in groups.items():
        items_sorted = sorted(items, key=lambda t: t[0])
        if len(items_sorted) < 2:
            continue
        for (lvl_lo, idx_lo, lbl_lo), (lvl_hi, idx_hi, lbl_hi) in zip(items_sorted[:-1], items_sorted[1:]):
            row = np.zeros(n_comp, dtype=float)
            row[int(idx_lo)] = 1.0
            row[int(idx_hi)] = -1.0
            rows.append(row)
            targets.append(0.0)
            meta.append(
                {
                    "ladder": f"{anti_prefix}nstar_eff::{charge}",
                    "level_lo": int(lvl_lo),
                    "level_hi": int(lvl_hi),
                    "index_lo": int(idx_lo),
                    "index_hi": int(idx_hi),
                    "label_lo": str(lbl_lo),
                    "label_hi": str(lbl_hi),
                }
            )

    if not rows:
        return {"ok": False, "reason": "no_nstar_ladders"}
    return {
        "ok": True,
        "A": np.vstack(rows),
        "b": np.asarray(targets, dtype=float),
        "n_rows": int(len(rows)),
        "n_ladders": int(len(groups)),
        "row_meta": meta,
    }


def _infer_source_side_from_xf(xf: np.ndarray) -> np.ndarray:
    x = np.asarray(xf, dtype=float)
    return np.where(x >= 0.0, 1, -1).astype(np.int8)


def _feedover_keep_mask(
    xf: np.ndarray,
    source_side: np.ndarray,
    *,
    target_feedover_xf_max: float,
    projectile_feedover_xf_max: float,
) -> np.ndarray:
    x = np.asarray(xf, dtype=float)
    s = np.asarray(source_side, dtype=np.int8)
    keep = np.ones(x.shape[0], dtype=bool)
    tmax = float(target_feedover_xf_max)
    pmax = float(projectile_feedover_xf_max)
    if np.isfinite(tmax) and tmax >= 0.0:
        keep &= ~((s < 0) & (x > tmax))
    if np.isfinite(pmax) and pmax >= 0.0:
        keep &= ~((s > 0) & (x < -pmax))
    return keep


def _build_table9_composition_prior(
    *,
    component_labels: List[str],
    pi_minus_per_component: np.ndarray,
    table9_json: Path,
) -> Dict[str, object]:
    payload = json.loads(table9_json.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        return {"ok": False, "reason": "invalid_rows"}

    selector_map = {
        "Delta0(1232)": "delta0",
        "Delta-(1232)": "delta-",
        "N*(1440)": "n1440",
        "N*(1520)": "n1520",
        "N*(1680)": "n1680",
        "eta(548)": "eta",
        "rho0(770)": "rho0",
        "rho-(770)": "rho-",
        "omega(782)": "omega",
        "f2(1270)": "f2",
        "rho3^0(1690)": "rho3^0",
        "rho3-(1690)": "rho3-",
        "f4(2050)": "f4",
    }

    table_total = 0.0
    row_defs: List[Dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", ""))
        selector = selector_map.get(name)
        if selector is None:
            continue
        target = float(row.get("table_pi_minus_yield", 0.0))
        if target <= 0.0:
            continue
        m = _label_match_mask(component_labels, selector)
        if not np.any(m):
            continue
        vec = np.zeros(len(component_labels), dtype=float)
        vec[m] = pi_minus_per_component[m]
        if float(np.sum(np.abs(vec))) <= 0.0:
            continue
        table_total += target
        row_defs.append({"name": name, "target": target, "vec": vec, "selector": selector})

    if not row_defs or table_total <= 0.0:
        return {"ok": False, "reason": "no_matching_rows"}

    total_vec = np.zeros(len(component_labels), dtype=float)
    for item in row_defs:
        total_vec += np.asarray(item["vec"], dtype=float)
    if float(np.sum(np.abs(total_vec))) <= 0.0:
        return {"ok": False, "reason": "zero_total_vec"}

    prior_rows: List[np.ndarray] = []
    prior_targets: List[float] = []
    prior_meta: List[Dict[str, object]] = []
    for item in row_defs:
        frac = float(item["target"]) / float(table_total)
        vec = np.asarray(item["vec"], dtype=float) - frac * total_vec
        norm = float(np.linalg.norm(vec))
        if norm <= 0.0:
            continue
        prior_rows.append(vec)
        prior_targets.append(0.0)
        prior_meta.append(
            {
                "name": str(item["name"]),
                "selector": str(item["selector"]),
                "table_pi_minus_yield": float(item["target"]),
                "table_fraction": float(frac),
                "row_norm": norm,
            }
        )

    if not prior_rows:
        return {"ok": False, "reason": "empty_composition_rows"}

    return {
        "ok": True,
        "A": np.vstack(prior_rows),
        "b": np.asarray(prior_targets, dtype=float),
        "total_vec": np.asarray(total_vec, dtype=float),
        "raw_rows": np.vstack([np.asarray(item["vec"], dtype=float) for item in row_defs]),
        "target_fractions": np.asarray(
            [float(item["target"]) / float(table_total) for item in row_defs],
            dtype=float,
        ),
        "meta": prior_meta,
        "table_total": float(table_total),
    }


def _build_strange_hyperon_priors(
    *,
    kplus_per_component: np.ndarray,
    kminus_per_component: np.ndarray,
    hyperon_net_per_component: np.ndarray,
    kminus_over_kplus_target: float,
) -> Dict[str, object]:
    """Build soft rows for the strange/hyperon balance proxy prior.

    The rows encode a phenomenological balance proxy between selected kaon
    and hyperon yields. The proxy omits neutral kaons, multi-strange baryons
    outside the selected set, acceptance effects, and the analysis-window
    restriction; because the cascade decays hyperons weakly, final-state
    track counts also differ from the pre-weak hadron yields being balanced.
    """
    # Ratio proxy row: K- - r*K+ = 0
    r = float(kminus_over_kplus_target)
    row_ratio = np.asarray(kminus_per_component, dtype=float) - r * np.asarray(kplus_per_component, dtype=float)

    # Associated-production balance proxy (see caveat above):
    # net-hyperon - (K+ - K-) = 0
    row_balance = np.asarray(hyperon_net_per_component, dtype=float) - (
        np.asarray(kplus_per_component, dtype=float) - np.asarray(kminus_per_component, dtype=float)
    )

    return {
        "kaon_ratio_row": row_ratio,
        "kaon_ratio_target": 0.0,
        "hyperon_balance_row": row_balance,
        "hyperon_balance_target": 0.0,
    }


CONVENTION = "invariant"  # 'raw' reproduces superseded pre-audit (2026-08-23) campaigns
SQRT_S_GEV = None  # set in main() from --beam-momentum


def _counts_to_invariant_kernel(sd):
    """Kernel converting raw bin counts to the tabulated invariant cross section.

    f_ij = counts_ij * E/(pT*dx*dp), constants absorbed by fitted scales.
    Validated by the 2026-08-23 closure audit
    (notes/benchmarks/JACOBIAN_AUDIT_2026-08-23.md).
    """
    p_l = sd.xf * (0.5 * SQRT_S_GEV)
    energy = np.sqrt(sd.mass_gev ** 2 + p_l ** 2 + sd.pt ** 2)
    return energy / (sd.pt * _point_bin_areas(sd))


def _invariant_integration_weights(sd):
    """Weights W=(pT/E)*dx*dp so that pred @ W integrates f like a yield."""
    p_l = sd.xf * (0.5 * SQRT_S_GEV)
    energy = np.sqrt(sd.mass_gev ** 2 + p_l ** 2 + sd.pt ** 2)
    return (sd.pt / energy) * _point_bin_areas(sd)


def _species_column_from_tracks(
    *,
    pdg_arr: np.ndarray,
    xf_arr: np.ndarray,
    pt_arr: np.ndarray,
    sd,
    convention: str,
    sqrt_s_gev: float,
) -> np.ndarray:
    """Design column of one species from component tracks on its data grid.

    Under the invariant convention each matching track carries its E/pT
    Jacobian weight and counts are divided by bin area, giving a
    bin-averaged invariant-cross-section template consistent with
    benchmark_sps_species._append_component_columns_from_tracks.
    """
    m_sp = pdg_arr == int(sd.pdg)
    if not np.any(m_sp):
        return np.zeros(len(sd.y), dtype=float)
    if convention == "invariant":
        w = bss._invariant_track_weights(int(sd.pdg), xf_arr[m_sp], pt_arr[m_sp], sqrt_s_gev)
        return (
            bss._counts_on_grid(
                xf_arr[m_sp],
                pt_arr[m_sp],
                sd.xedges,
                sd.pedges,
                sd.ix_data,
                sd.ip_data,
                weights=w,
            )
            / bss._grid_point_areas(sd.xedges, sd.pedges, sd.ix_data, sd.ip_data)
        )
    return bss._counts_on_grid(
        xf_arr[m_sp],
        pt_arr[m_sp],
        sd.xedges,
        sd.pedges,
        sd.ix_data,
        sd.ip_data,
    )


@dataclass
class SpeciesData:
    key: str
    name: str
    tag: str
    pdg: int
    mass_gev: float
    na49_path: Path
    xf: np.ndarray
    pt: np.ndarray
    y: np.ndarray
    sigma: np.ndarray
    xedges: np.ndarray
    pedges: np.ndarray
    ix_data: np.ndarray
    ip_data: np.ndarray
    train_idx: np.ndarray
    test_idx: np.ndarray
    matrix: np.ndarray
    pred_res: np.ndarray
    pred_res_sigma: np.ndarray


def _point_bin_areas(sd: SpeciesData) -> np.ndarray:
    dx = np.diff(np.asarray(sd.xedges, dtype=float))
    dp = np.diff(np.asarray(sd.pedges, dtype=float))
    return dx[np.asarray(sd.ix_data, dtype=np.int64)] * dp[np.asarray(sd.ip_data, dtype=np.int64)]


def _point_bin_window_areas(sd: SpeciesData, xf_min: float, xf_max: float) -> np.ndarray:
    ix = np.asarray(sd.ix_data, dtype=np.int64)
    ip = np.asarray(sd.ip_data, dtype=np.int64)
    xedges = np.asarray(sd.xedges, dtype=float)
    pedges = np.asarray(sd.pedges, dtype=float)
    x_left = xedges[ix]
    x_right = xedges[ix + 1]
    p_width = np.diff(pedges)[ip]
    lo = float(min(xf_min, xf_max))
    hi = float(max(xf_min, xf_max))
    overlap = np.maximum(0.0, np.minimum(x_right, hi) - np.maximum(x_left, lo))
    return overlap * p_width


def _integrated_yield(values: np.ndarray, areas: np.ndarray, idx: np.ndarray) -> float:
    ii = np.asarray(idx, dtype=np.int64)
    if ii.size <= 0:
        return float("nan")
    vv = np.asarray(values, dtype=float)
    aa = np.asarray(areas, dtype=float)
    return float(np.sum(vv[ii] * aa[ii]))


def _build_particle_ratio_summary(
    *,
    species_data_by_key: Dict[str, SpeciesData],
    xf_windows: List[Tuple[str, float, float]],
) -> Dict[str, object]:
    key_by_name = {str(sd.name).strip().lower(): key for key, sd in species_data_by_key.items()}
    point_indices_by_key: Dict[str, np.ndarray] = {
        key: np.arange(len(sd.y), dtype=np.int64) for key, sd in species_data_by_key.items()
    }

    ratio_defs: List[Tuple[str, List[str], List[str]]] = []
    if "pi+" in key_by_name and "pi-" in key_by_name:
        ratio_defs.append(("pi_plus_over_pi_minus", ["pi+"], ["pi-"]))
    if "k+" in key_by_name and "k-" in key_by_name:
        ratio_defs.append(("k_plus_over_k_minus", ["k+"], ["k-"]))
    if "pbar" in key_by_name and "p" in key_by_name:
        ratio_defs.append(("pbar_over_p", ["pbar"], ["p"]))
    if all(name in key_by_name for name in ("k+", "k-", "pi+", "pi-")):
        ratio_defs.append(("k_sum_over_pi_sum", ["k+", "k-"], ["pi+", "pi-"]))

    def _prediction_for(model_key: str, species_key: str) -> np.ndarray | None:
        if model_key == "Data":
            return np.asarray(species_data_by_key[species_key].y, dtype=float)
        if model_key == "Resonance_shared":
            return np.asarray(species_data_by_key[species_key].pred_res, dtype=float)
        # Policy: external generators are omitted from particle-ratio windows.
        return None

    model_keys = ["Data", "Resonance_shared"]
    values_by_window: Dict[str, Dict[str, Dict[str, float | None]]] = {}
    window_defs: Dict[str, Dict[str, float]] = {}
    for win_label, win_lo, win_hi in xf_windows:
        # Yield-like integration of the invariant quantity requires the
        # Jacobian factor pT/E per point (see paper Sec. on conventions):
        #   N ~ sum_i (pT_i/E_i) * overlap_xF_i * dPT_i.
        areas_by_key = {
            key: _invariant_integration_weights(sd)
            / np.maximum(_point_bin_areas(sd), 1e-30)
            * _point_bin_window_areas(sd, xf_min=win_lo, xf_max=win_hi)
            for key, sd in species_data_by_key.items()
        }
        ratio_values: Dict[str, Dict[str, float | None]] = {}
        for ratio_name, num_names, den_names in ratio_defs:
            model_vals: Dict[str, float | None] = {}
            for model_key in model_keys:
                num_total = 0.0
                den_total = 0.0
                ok = True

                for sp_name in num_names:
                    sp_key = key_by_name.get(sp_name)
                    if sp_key is None:
                        ok = False
                        break
                    pred = _prediction_for(model_key, sp_key)
                    if pred is None:
                        ok = False
                        break
                    yld = _integrated_yield(pred, areas_by_key[sp_key], point_indices_by_key[sp_key])
                    if not np.isfinite(yld):
                        ok = False
                        break
                    num_total += float(yld)

                if ok:
                    for sp_name in den_names:
                        sp_key = key_by_name.get(sp_name)
                        if sp_key is None:
                            ok = False
                            break
                        pred = _prediction_for(model_key, sp_key)
                        if pred is None:
                            ok = False
                            break
                        yld = _integrated_yield(pred, areas_by_key[sp_key], point_indices_by_key[sp_key])
                        if not np.isfinite(yld):
                            ok = False
                            break
                        den_total += float(yld)

                if (not ok) or (not np.isfinite(den_total)) or den_total <= 0.0:
                    model_vals[model_key] = None
                else:
                    model_vals[model_key] = float(num_total / den_total)
            ratio_values[ratio_name] = model_vals

        window_defs[win_label] = {"xf_min": float(win_lo), "xf_max": float(win_hi)}
        values_by_window[win_label] = ratio_values

    default_window = xf_windows[0][0] if xf_windows else "full"
    default_values = values_by_window.get(default_window, {})
    return {
        "integration": (
            "sum(y * (pT/E) * overlap(delta_xF,[xf_min,xf_max]) * delta_pT) over benchmark points (yield kernel)"
        ),
        "definitions": {
            name: {"numerator": num, "denominator": den}
            for name, num, den in ratio_defs
        },
        "external_normalization": {
            "PYTHIA8_external": "per_species_shape",
            "EPOS_external": "per_species_shape",
            "UrQMD_external": "per_species_shape",
            "note": (
                "External generators are omitted conservatively from these "
                "particle-ratio windows. Each baseline is fitted with one "
                "normalization per species, so its cross-species ratios are "
                "not generator yield predictions; the baselines remain "
                "available as species-by-species shape-only fits."
            ),
        },
        "windows": window_defs,
        "default_window": default_window,
        "values_by_window": values_by_window,
        "values": default_values,
    }


def _seagull_profile_for_prediction(
    *,
    sd: SpeciesData,
    prediction: np.ndarray,
    prediction_sigma: np.ndarray | None,
    idx: np.ndarray,
    sqrt_s_from_beam_momentum: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ii = np.asarray(idx, dtype=np.int64)
    if ii.size <= 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    xf = np.asarray(sd.xf[ii], dtype=float)
    pt = np.asarray(sd.pt[ii], dtype=float)
    ip = np.asarray(sd.ip_data[ii], dtype=np.int64)
    dp_all = np.diff(np.asarray(sd.pedges, dtype=float))
    dp = dp_all[ip]
    yv = np.asarray(prediction[ii], dtype=float)
    sv = None if prediction_sigma is None else np.asarray(prediction_sigma[ii], dtype=float)

    # Group by xF centers present in the selected benchmark sample.
    xf_round = np.round(xf, 10)
    xf_unique = np.unique(xf_round)
    if xf_unique.size <= 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    # Guard against sparse/interleaved SPS xF slices with incompatible pT support
    # (e.g. low-pT-only and high-pT-only slices): these produce artificial seagull spikes.
    pt_global_min = float(np.min(pt))
    pt_global_max = float(np.max(pt))
    pt_low_anchor = pt_global_min + 0.05
    pt_high_anchor = min(pt_global_max - 0.30, 1.0)
    if not np.isfinite(pt_high_anchor):
        pt_high_anchor = 1.0
    pt_high_anchor = max(pt_high_anchor, pt_low_anchor)

    xf_valid: List[float] = []
    for xval in xf_unique:
        m_cov = xf_round == xval
        if int(np.count_nonzero(m_cov)) <= 0:
            continue
        pt_slice = pt[m_cov]
        if pt_slice.size <= 0:
            continue
        if float(np.min(pt_slice)) <= (pt_low_anchor + 1e-12) and float(np.max(pt_slice)) >= (pt_high_anchor - 1e-12):
            xf_valid.append(float(xval))
    xf_loop = np.asarray(xf_valid, dtype=float)

    x_out: List[float] = []
    pt_out: List[float] = []
    pt_err_out: List[float] = []
    for xval in xf_loop:
        m = xf_round == xval
        if int(np.count_nonzero(m)) <= 0:
            continue

        # Correct invariant-cross-section integration kernel for pT moments:
        #   dN/dxF  ~ \int (f/E) dpT^2 = \int (f/E) * 2 pT dpT
        # so, up to xF-independent constants that cancel in ratios:
        #   weight_i = y_i * (pT_i/E_i) * delta_pT_i.
        p_l = xf[m] * (0.5 * float(sqrt_s_from_beam_momentum))
        energy = np.sqrt(np.maximum(float(sd.mass_gev) ** 2 + p_l**2 + pt[m] ** 2, 1e-30))
        kernel = (pt[m] / energy) * dp[m]

        w = yv[m] * kernel
        den = float(np.sum(w))
        if (not np.isfinite(den)) or den <= 0.0:
            continue
        num = float(np.sum(pt[m] * w))
        if not np.isfinite(num):
            continue
        mean_pt = float(num / den)

        mean_pt_err = float("nan")
        if sv is not None:
            sigma_m = np.maximum(sv[m], 0.0)
            # Propagate uncertainty for mu = S1/S0 with S0=sum(y*k), S1=sum(pt*y*k), k=(pT/E)*dp.
            dmu_dy = kernel * ((pt[m] * den) - num) / max(den * den, 1e-30)
            var_mu = float(np.sum((dmu_dy * sigma_m) ** 2))
            if np.isfinite(var_mu) and var_mu >= 0.0:
                mean_pt_err = float(np.sqrt(var_mu))

        x_out.append(float(np.mean(xf[m])))
        pt_out.append(mean_pt)
        pt_err_out.append(mean_pt_err)

    if not x_out:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    x_arr = np.asarray(x_out, dtype=float)
    y_arr = np.asarray(pt_out, dtype=float)
    e_arr = np.asarray(pt_err_out, dtype=float)
    order = np.argsort(x_arr)
    return x_arr[order], y_arr[order], e_arr[order]


def _seagull_shape_metrics(xf: np.ndarray, pt_mean: np.ndarray) -> Dict[str, float | int | None]:
    x = np.asarray(xf, dtype=float)
    y = np.asarray(pt_mean, dtype=float)
    n = int(x.shape[0])
    out: Dict[str, float | int | None] = {
        "n_xf_bins": n,
        "linear_slope": None,
        "linear_intercept": None,
        "quadratic_coeff": None,
        "delta_first_last": None,
    }
    if n <= 0:
        return out
    if n >= 2:
        p1 = np.polyfit(x, y, 1)
        out["linear_slope"] = _as_float_or_none(float(p1[0]))
        out["linear_intercept"] = _as_float_or_none(float(p1[1]))
        out["delta_first_last"] = _as_float_or_none(float(y[-1] - y[0]))
    if n >= 3:
        p2 = np.polyfit(x, y, 2)
        out["quadratic_coeff"] = _as_float_or_none(float(p2[0]))
    return out


def _build_seagull_summary(
    *,
    species_data_by_key: Dict[str, SpeciesData],
    external_pred_by_model: Dict[str, Dict[str, np.ndarray]],
    external_sigma_by_model: Dict[str, Dict[str, np.ndarray]],
    sqrt_s_from_beam_momentum: float,
) -> Dict[str, object]:
    point_indices_by_key: Dict[str, np.ndarray] = {
        key: np.arange(len(sd.y), dtype=np.int64) for key, sd in species_data_by_key.items()
    }

    def _prediction_for(model_key: str, species_key: str) -> np.ndarray | None:
        if model_key == "Data":
            return np.asarray(species_data_by_key[species_key].y, dtype=float)
        if model_key == "Resonance_shared":
            return np.asarray(species_data_by_key[species_key].pred_res, dtype=float)
        if model_key == "PYTHIA8_external":
            return external_pred_by_model.get("PYTHIA8_external", {}).get(species_key)
        if model_key == "EPOS_external":
            return external_pred_by_model.get("EPOS_external", {}).get(species_key)
        if model_key == "UrQMD_external":
            return external_pred_by_model.get("UrQMD_external", {}).get(species_key)
        return None

    def _prediction_sigma_for(model_key: str, species_key: str) -> np.ndarray | None:
        if model_key == "Data":
            return np.asarray(species_data_by_key[species_key].sigma, dtype=float)
        if model_key == "Resonance_shared":
            return np.asarray(species_data_by_key[species_key].pred_res_sigma, dtype=float)
        if model_key == "PYTHIA8_external":
            return external_sigma_by_model.get("PYTHIA8_external", {}).get(species_key)
        if model_key == "EPOS_external":
            return external_sigma_by_model.get("EPOS_external", {}).get(species_key)
        if model_key == "UrQMD_external":
            return external_sigma_by_model.get("UrQMD_external", {}).get(species_key)
        return None

    model_keys = ["Data", "Resonance_shared", "PYTHIA8_external", "EPOS_external", "UrQMD_external"]
    by_species: Dict[str, object] = {}
    for skey, sd in species_data_by_key.items():
        models_payload: Dict[str, object] = {}
        for model_key in model_keys:
            pred = _prediction_for(model_key, skey)
            if pred is None:
                models_payload[model_key] = {"status": "missing_prediction"}
                continue
            pred_sigma = _prediction_sigma_for(model_key, skey)
            xf_prof, pt_prof, pt_err_prof = _seagull_profile_for_prediction(
                sd=sd,
                prediction=np.asarray(pred, dtype=float),
                prediction_sigma=None if pred_sigma is None else np.asarray(pred_sigma, dtype=float),
                idx=point_indices_by_key[skey],
                sqrt_s_from_beam_momentum=float(sqrt_s_from_beam_momentum),
            )
            if xf_prof.size <= 0:
                models_payload[model_key] = {"status": "empty_profile"}
                continue
            row = _seagull_shape_metrics(xf_prof, pt_prof)
            row.update(
                {
                    "xf": [float(v) for v in xf_prof.tolist()],
                    "pt_mean": [float(v) for v in pt_prof.tolist()],
                    "pt_mean_err": [float(v) for v in pt_err_prof.tolist()],
                }
            )
            models_payload[model_key] = row
        by_species[skey] = {
            "species_name": sd.name,
            "models": models_payload,
        }

    return {
        "definition": (
            "<pT>(xF)=sum[pT * y * (pT/E) * delta_pT] / sum[y * (pT/E) * delta_pT] at fixed xF, "
            "with E=sqrt(m^2+pL^2+pT^2), pL=xF*sqrt(s)/2"
        ),
        "notes": (
            "profiles are computed on the same selected benchmark points for data and models; "
            "point uncertainties are propagated with linear error propagation for weighted means; "
            "xF bins with sparse/incompatible pT support are excluded using per-species anchors "
            "(min pT <= pT_min+0.05 and max pT >= min(pT_max-0.30,1.0))."
        ),
        "values": by_species,
    }


def _parse_species_csv(species_csv: str) -> List[str]:
    keys = [s.strip() for s in species_csv.split(",") if s.strip()]
    if not keys:
        raise ValueError("No species provided.")
    uniq: List[str] = []
    for key in keys:
        # validate and canonicalize by display name only
        _ = resolve_species(key)
        if key not in uniq:
            uniq.append(key)
    return uniq


def _format_xf_window_label(xf_min: float, xf_max: float) -> str:
    lo = float(min(xf_min, xf_max))
    hi = float(max(xf_min, xf_max))
    return f"xf_{lo:.3f}_{hi:.3f}"


def _parse_xf_windows(spec: str) -> List[Tuple[str, float, float]]:
    out: List[Tuple[str, float, float]] = []
    seen: set[str] = set()
    for chunk in str(spec).split(","):
        txt = chunk.strip()
        if not txt:
            continue
        if ":" not in txt:
            raise ValueError(f"Invalid xF window '{txt}'. Expected format 'min:max'.")
        a, b = txt.split(":", 1)
        lo = float(a.strip())
        hi = float(b.strip())
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(f"Invalid non-finite xF window '{txt}'.")
        if hi < lo:
            lo, hi = hi, lo
        label = _format_xf_window_label(lo, hi)
        if label in seen:
            continue
        seen.add(label)
        out.append((label, lo, hi))
    if not out:
        raise ValueError("No valid xF windows provided.")
    return out


def _parse_single_xf_window(spec: str) -> Tuple[float, float]:
    windows = _parse_xf_windows(spec)
    if len(windows) != 1:
        raise ValueError(
            "Expected exactly one xF window in format 'min:max' for absolute-yield prior; "
            f"got {len(windows)} windows."
        )
    _, lo, hi = windows[0]
    return float(lo), float(hi)


def _species_alias_map(species_data_by_key: Dict[str, SpeciesData]) -> Dict[str, str]:
    alias: Dict[str, str] = {}
    for key, sd in species_data_by_key.items():
        k = str(key)
        alias[k.strip().lower()] = k
        alias[str(sd.name).strip().lower()] = k
        alias[str(sd.tag).strip().lower()] = k
        alias[str(int(sd.pdg))] = k
    return alias


def _resolve_species_token(token: str, alias_map: Dict[str, str]) -> str:
    t = str(token).strip().lower()
    if not t:
        raise ValueError("Empty species token.")
    if t in alias_map:
        return alias_map[t]
    try:
        spec = resolve_species(token)
        for cand in (spec.name, spec.tag, str(spec.pdg)):
            key = alias_map.get(str(cand).strip().lower())
            if key is not None:
                return key
    except Exception:
        pass
    valid = sorted(set(alias_map.keys()))
    raise ValueError(f"Unknown species token '{token}'. Known aliases include: {', '.join(valid)}")


def _resolve_abs_yield_species_csv(species_csv: str, species_data_by_key: Dict[str, SpeciesData]) -> List[str]:
    if not str(species_csv).strip():
        return list(species_data_by_key.keys())
    alias = _species_alias_map(species_data_by_key)
    out: List[str] = []
    seen: set[str] = set()
    for tok in str(species_csv).split(","):
        t = tok.strip()
        if not t:
            continue
        key = _resolve_species_token(t, alias)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    if not out:
        raise ValueError("No valid species selected for absolute-yield prior.")
    return out


def _load_abs_yield_targets_json(
    path: Path,
    *,
    species_data_by_key: Dict[str, SpeciesData],
) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Absolute-yield target JSON must be an object: {path}")

    window = payload.get("window", {})
    xf_min_json = None
    xf_max_json = None
    if isinstance(window, dict) and ("xf_min" in window) and ("xf_max" in window):
        xf_min_json = float(window["xf_min"])
        xf_max_json = float(window["xf_max"])
        if not np.isfinite(xf_min_json) or not np.isfinite(xf_max_json):
            raise ValueError(f"Non-finite window bounds in absolute-yield JSON: {path}")
        if xf_max_json < xf_min_json:
            xf_min_json, xf_max_json = xf_max_json, xf_min_json

    targets_raw = payload.get("targets", {})
    if not isinstance(targets_raw, dict) or not targets_raw:
        raise ValueError(f"Absolute-yield JSON must contain non-empty object key 'targets': {path}")
    sigmas_raw = payload.get("sigmas", {})
    if sigmas_raw is None:
        sigmas_raw = {}
    if not isinstance(sigmas_raw, dict):
        raise ValueError(f"Absolute-yield JSON key 'sigmas' must be an object when present: {path}")

    alias = _species_alias_map(species_data_by_key)
    targets_by_key: Dict[str, float] = {}
    sigmas_by_key: Dict[str, float] = {}
    for raw_key, raw_val in targets_raw.items():
        key = _resolve_species_token(str(raw_key), alias)
        val = float(raw_val)
        if not np.isfinite(val):
            raise ValueError(f"Non-finite target value for '{raw_key}' in {path}")
        targets_by_key[key] = val

    for raw_key, raw_val in sigmas_raw.items():
        key = _resolve_species_token(str(raw_key), alias)
        val = abs(float(raw_val))
        if not np.isfinite(val):
            raise ValueError(f"Non-finite sigma value for '{raw_key}' in {path}")
        sigmas_by_key[key] = val

    return {
        "targets_by_key": targets_by_key,
        "sigmas_by_key": sigmas_by_key,
        "xf_min": xf_min_json,
        "xf_max": xf_max_json,
    }


def _build_absolute_yield_prior(
    *,
    species_data_by_key: Dict[str, SpeciesData],
    selected_species_keys: List[str],
    xf_min: float,
    xf_max: float,
    target_overrides: Dict[str, float],
    sigma_overrides: Dict[str, float],
    min_sigma_rel: float,
) -> Dict[str, object]:
    prior_rows: List[np.ndarray] = []
    prior_targets: List[float] = []
    row_meta: List[Dict[str, object]] = []
    eval_rows: List[Dict[str, object]] = []

    min_sigma_rel_eff = max(float(min_sigma_rel), 0.0)
    for key in selected_species_keys:
        sd = species_data_by_key.get(key)
        if sd is None:
            continue
        idx = np.arange(len(sd.y), dtype=np.int64)
        areas = _point_bin_window_areas(sd, xf_min=xf_min, xf_max=xf_max)
        if CONVENTION == "invariant":
            # Direct partial-overlap weight: w_i = (pT_i/E_i)*overlap_i.
            areas = (
                _invariant_integration_weights(sd)
                / np.maximum(_point_bin_areas(sd), 1e-30)
                * _point_bin_window_areas(sd, xf_min=xf_min, xf_max=xf_max)
            )
        target_data = _integrated_yield(sd.y, areas, idx)
        sigma_data = float(np.sqrt(np.sum((np.asarray(sd.sigma, dtype=float) * np.asarray(areas, dtype=float)) ** 2)))
        target_val = float(target_overrides.get(key, target_data))
        sigma_val = float(sigma_overrides.get(key, sigma_data))
        sigma_floor = max(abs(target_val) * min_sigma_rel_eff, 1e-12)
        sigma_eff = max(abs(sigma_val), sigma_floor)

        vec = np.asarray(sd.matrix, dtype=float).T @ np.asarray(areas, dtype=float)
        vec_norm = float(np.linalg.norm(vec))
        if vec_norm <= 0.0:
            continue

        prior_rows.append(vec / sigma_eff)
        prior_targets.append(float(target_val / sigma_eff))
        row_meta.append(
            {
                "species_key": str(key),
                "species_name": str(sd.name),
                "target": float(target_val),
                "sigma_data": float(sigma_data),
                "sigma_used": float(sigma_eff),
                "target_from_override": bool(key in target_overrides),
                "sigma_from_override": bool(key in sigma_overrides),
                "row_norm": vec_norm / max(sigma_eff, 1e-30),
            }
        )
        eval_rows.append(
            {
                "species_key": str(key),
                "species_name": str(sd.name),
                "target": float(target_val),
                "sigma": float(sigma_eff),
                "vec": np.asarray(vec, dtype=float),
            }
        )

    if not prior_rows:
        return {"ok": False, "reason": "no_matching_species_rows"}

    return {
        "ok": True,
        "A": np.vstack(prior_rows),
        "b": np.asarray(prior_targets, dtype=float),
        "row_meta": row_meta,
        "eval_rows": eval_rows,
        "xf_min": float(xf_min),
        "xf_max": float(xf_max),
        "min_sigma_rel": float(min_sigma_rel_eff),
    }


def _full_sample_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    ii = np.arange(int(n), dtype=np.int64)
    # Keep both fields for internal compatibility, but they are identical by construction.
    return ii, ii


def _stable_seed(base_seed: int, tag: str) -> int:
    v = int(base_seed)
    for ch in tag:
        v = (v * 131 + ord(ch)) % (2**31 - 1)
    return v if v > 0 else 1


def _load_species_points(
    *,
    species_key: str,
    xf_min: float,
    xf_max: float,
    pt_min: float,
    pt_max: float,
    seed: int,
    data_system: str = "pp",
) -> SpeciesData:
    spec = resolve_species(species_key)
    na49_path = Path(na49_table_for(species_key, data_system))
    xf, pt, y, sigma = bss._read_na49_points(na49_path)
    mask = (
        (xf >= float(xf_min))
        & (xf <= float(xf_max))
        & (pt >= float(pt_min))
        & (pt <= float(pt_max))
        & (sigma > 0.0)
    )
    xf = xf[mask]
    pt = pt[mask]
    y = y[mask]
    sigma = sigma[mask]
    xedges = bss._edges_from_centers(xf)
    pedges = bss._edges_from_centers(pt)
    ix_data, ip_data = bss._grid_indices(xf, pt, xedges, pedges)
    _ = _stable_seed(seed, spec.tag)
    train_idx, test_idx = _full_sample_indices(len(y))

    return SpeciesData(
        key=species_key,
        name=spec.name,
        tag=spec.tag,
        pdg=spec.pdg,
        mass_gev=_pdg_mass_gev(spec.pdg),
        na49_path=na49_path,
        xf=xf,
        pt=pt,
        y=y,
        sigma=sigma,
        xedges=xedges,
        pedges=pedges,
        ix_data=ix_data,
        ip_data=ip_data,
        train_idx=train_idx,
        test_idx=test_idx,
        matrix=np.zeros((len(y), 0), dtype=float),
        pred_res=np.zeros_like(y),
        pred_res_sigma=np.zeros_like(y),
    )


def _fit_single_external_model(
    species_data: SpeciesData,
    csv_path: Path,
    *,
    row_scale_full: np.ndarray | None = None,
    return_pred: bool = False,
) -> Dict[str, float | int | str] | tuple[Dict[str, float | int | str], np.ndarray, np.ndarray, float]:
    mass = None
    if CONVENTION == "invariant":
        try:
            from helpers.benchmarking.sps_core import PDG_MASS_GEV as _PDG_MASS
            mass = _PDG_MASS.get(int(species_data.pdg))
        except Exception:
            mass = None
        if mass is None:
            mass = float(species_data.mass_gev)
    raw = bss._load_external_grid_counts(
        csv_path,
        species_data.xedges,
        species_data.pedges,
        species_data.ix_data,
        species_data.ip_data,
        convention=CONVENTION,
        sqrt_s_gev=SQRT_S_GEV,
        mass_gev=mass,
    )
    full_idx = np.arange(len(species_data.y), dtype=np.int64)
    alpha = bss._fit_single_scale(
        raw[full_idx],
        species_data.y[full_idx],
        species_data.sigma[full_idx],
        row_scale=row_scale_full,
    )
    pred = alpha * raw
    row: Dict[str, float | int | str] = {"n_params": 1, "scale": float(alpha)}
    row.update(
        bss._metrics_bundle(
            pred[full_idx],
            species_data.y[full_idx],
            species_data.sigma[full_idx],
            n_params=1,
            prefix="full",
        )
    )
    if return_pred:
        return row, pred, raw, float(alpha)
    return row


def _fit_global_nonnegative_weights(
    *,
    objective: str,
    A_train_raw: np.ndarray,
    y_train: np.ndarray,
    sigma_train: np.ndarray,
    A_train_weighted: np.ndarray,
    b_train_weighted: np.ndarray,
    A_prior: np.ndarray | None,
    b_prior: np.ndarray | None,
    table9_fraction_rows: np.ndarray | None,
    table9_fraction_targets: np.ndarray | None,
    table9_fraction_lambda: float,
    table9_fraction_eps: float,
    maxiter: int,
    ftol: float,
    allow_rank_deficient: bool = False,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Fit the shared non-negative weight vector for one chosen objective.

    Identifiability contract: before any solve, the design the fit consumes
    is diagnosed (numerical rank under the default SVD tolerance, singular
    values, 2-norm condition number) and a rank-deficient design is refused
    with a ValueError naming rank, component count, and condition number,
    following the MultiFitter wording in ``na49_mixing/fits.py``. The NNLS
    objective consumes the augmented weighted design (weighted data rows
    plus active linear-prior rows). The point-likelihood objectives consume
    the sigma-scaled raw data rows plus the active linear-prior rows
    (``design_scope`` "weighted_linear_component_matrix[_with_prior_rows]");
    their Table-9 fraction term enters the objective only through the
    normalized nonlinear fraction y_rows/sum(y_rows), so it contributes no
    linear rows and is excluded from that design and from the Jacobian.
    Passing ``allow_rank_deficient=True`` opts out of the refusal for
    deliberate exploratory fits; every result payload then carries
    ``identifiability=RANK_DEFICIENT_STAMP`` whenever the diagnosed design is
    rank deficient, and every payload always records the identifiability
    stamp and ``n_active_weights`` (weights > ACTIVE_WEIGHT_THRESHOLD, the
    same convention as the caller's n_eff counter).

    Input contract: the raw sigma vector is validated before the positive
    floor is applied, so a non-finite uncertainty (including -inf) fails the
    fit instead of being floored into a tiny positive value. Linear priors
    must be supplied as a pair: exactly one of ``A_prior``/``b_prior``
    present raises ValueError before any solver call, and both present and
    consistently empty keeps the documented no-op prior behavior. Table-9
    fraction inputs follow the same pair rule; a positive fraction lambda
    with both arrays missing raises ValueError, arrays supplied with a zero
    lambda are validated and reported as inactive through
    ``table9_fraction_active`` in the result payload, and an active
    fraction prior on the nnls_weighted objective is refused (point-
    likelihood objectives only).
    """
    objective_norm = str(objective).strip().lower()
    if objective_norm not in FIT_OBJECTIVES:
        raise ValueError(f"Unsupported fit objective '{objective}'. Valid: {FIT_OBJECTIVES}")

    n_comp = int(A_train_raw.shape[1]) if A_train_raw.ndim == 2 else 0
    if n_comp <= 0:
        raise RuntimeError("Cannot fit empty component matrix.")

    a0 = np.asarray(A_train_raw, dtype=float)
    y0 = np.asarray(y_train, dtype=float)
    s0 = np.asarray(sigma_train, dtype=float)
    a_w = np.asarray(A_train_weighted, dtype=float)
    b_w = np.asarray(b_train_weighted, dtype=float)
    ap = None if A_prior is None else np.asarray(A_prior, dtype=float)
    bp = None if b_prior is None else np.asarray(b_prior, dtype=float)
    t9r = None if table9_fraction_rows is None else np.asarray(table9_fraction_rows, dtype=float)
    t9f = None if table9_fraction_targets is None else np.asarray(table9_fraction_targets, dtype=float)
    lam_t9_frac = max(float(table9_fraction_lambda), 0.0)
    eps_t9_frac = max(float(table9_fraction_eps), 1e-12)

    # Validate fit inputs before any SciPy call so a malformed or non-finite
    # matrix, target, uncertainty, or prior array fails here with the
    # offending quantity named. The raw uncertainty vector is validated
    # before the documented positive floor is applied, so a non-finite value
    # (including -inf) cannot be masked into a tiny positive sigma.
    a0 = validate_fit_design(a0, n_comp, name="A_train_raw")
    y0 = validate_fit_target(y0, a0.shape[0], name="y_train")
    s0 = validate_fit_sigma(s0, a0.shape[0], name="sigma_train")
    s0 = np.maximum(s0, 1e-12)
    a_w = validate_fit_design(a_w, n_comp, name="A_train_weighted")
    b_w = validate_fit_target(b_w, a_w.shape[0], name="b_train_weighted")
    if (A_prior is None) != (b_prior is None):
        raise ValueError(
            "fit A_prior/b_prior: linear priors must be supplied as a pair; "
            f"exactly one side is present (A_prior: {A_prior is not None}, "
            f"b_prior: {b_prior is not None})"
        )
    if ap is not None and bp is not None:
        ap, bp = validate_fit_prior_rows(ap, bp, n_comp, name="A_prior/b_prior")
    if (t9r is None) != (t9f is None):
        raise ValueError(
            "fit table9_fraction_rows/table9_fraction_targets: Table-9 fraction "
            "priors must be supplied as a pair; exactly one side is present "
            f"(rows: {t9r is not None}, targets: {t9f is not None})"
        )
    if t9r is None and t9f is None and lam_t9_frac > 0.0:
        raise ValueError(
            "fit table9 fraction prior: a positive table9_fraction_lambda "
            "requires both table9_fraction_rows and table9_fraction_targets"
        )
    if t9r is not None and t9f is not None:
        t9r, t9f = validate_fit_table9_fraction_arrays(t9r, t9f, n_comp)
    t9_active = (
        lam_t9_frac > 0.0
        and t9r is not None
        and t9f is not None
        and int(t9r.size) > 0
    )

    if objective_norm == "nnls_weighted":
        if t9_active:
            raise ValueError(
                "Nonlinear Table-9 fraction prior is only supported for point-likelihood objectives "
                "(gaussian_point_mle / lognormal_point_mle)."
            )
        if ap is not None and bp is not None and ap.size > 0:
            A_fit = np.vstack([a_w, ap])
            b_fit = np.concatenate([b_w, bp])
            design_scope = "augmented_weighted_design"
        else:
            A_fit = a_w
            b_fit = b_w
            design_scope = "weighted_data_rows"
        # Identifiability of the augmented matrix the solver consumes,
        # checked before the solve: weighted data rows plus active
        # linear-prior rows. With no prior rows the augmented matrix is the
        # weighted data matrix itself.
        design_diag = enforce_fit_design_rank(
            A_fit,
            n_components=n_comp,
            allow_rank_deficient=allow_rank_deficient,
            context=f"{objective_norm} fit",
            design_scope=design_scope,
        )
        w, rnorm = nnls(A_fit, b_fit)
        w = validate_fit_result(
            np.asarray(w, dtype=float),
            n_components=n_comp,
            success=True,
            status=0,
            message="ok",
            objective=float(rnorm),
            objective_name="residual norm rnorm",
            context=f"{objective_norm} fit",
        )
        identifiability = (
            RANK_DEFICIENT_STAMP
            if int(design_diag["design_rank"]) < n_comp  # type: ignore[arg-type]
            else FULL_RANK_STAMP
        )
        info: Dict[str, object] = {
            "objective": objective_norm,
            "method": "scipy.optimize.nnls",
            "success": True,
            "status": 0,
            "message": "ok",
            "nfev": None,
            "nit": None,
            "design_scope": design_scope,
            **design_diag,
            "table9_fraction_active": t9_active,
            "n_active_weights": int(np.count_nonzero(w > ACTIVE_WEIGHT_THRESHOLD)),
            "identifiability": identifiability,
        }
        return w, info

    # The Table-9 fraction arrays were validated above on every path; the
    # fraction term itself is only consumed by the point-likelihood
    # objectives below (t9_active is false on the nnls_weighted path, which
    # refuses an active fraction prior there).

    # Identifiability of the linear part of the point-likelihood objective,
    # checked before any solve. The objective consumes the sigma-scaled raw
    # data rows (the same rows the warm start below uses) plus the active
    # linear-prior rows. The Table-9 fraction term enters only through the
    # normalized nonlinear fraction y_rows/sum(y_rows), so it contributes no
    # linear rows and is excluded from this design; design_scope records
    # that exclusion.
    a_lin = a0 / s0[:, None]
    if ap is not None and bp is not None and ap.size > 0:
        A_lin = np.vstack([a_lin, ap])
        design_scope = "weighted_linear_component_matrix_with_prior_rows"
    else:
        A_lin = a_lin
        design_scope = "weighted_linear_component_matrix"
    design_diag = enforce_fit_design_rank(
        A_lin,
        n_components=n_comp,
        allow_rank_deficient=allow_rank_deficient,
        context=f"{objective_norm} fit",
        design_scope=design_scope,
    )

    # Warm start from sigma-weighted NNLS on raw point matrix.
    A_init = a0 / s0[:, None]
    b_init = y0 / s0
    w0, _ = nnls(A_init, b_init)

    # Precompute masks/submatrices for log-normal likelihood.
    log_mask = (y0 > 0.0) & (s0 > 0.0)
    if objective_norm == "lognormal_point_mle":
        if not np.any(log_mask):
            raise RuntimeError("No positive points available for lognormal point-likelihood fit.")
        A_log = a0[log_mask]
        y_log = y0[log_mask]
        rel_sigma_log = np.maximum(s0[log_mask] / np.maximum(y_log, 1e-30), 1e-12)
    else:
        A_log = np.zeros((0, n_comp), dtype=float)
        y_log = np.zeros(0, dtype=float)
        rel_sigma_log = np.zeros(0, dtype=float)

    def _objective_and_grad(w_vec: np.ndarray) -> Tuple[float, np.ndarray]:
        wv = np.asarray(w_vec, dtype=float)
        pred = a0 @ wv
        grad = np.zeros_like(wv)

        if objective_norm == "gaussian_point_mle":
            resid = (pred - y0) / s0
            f_data = 0.5 * float(np.dot(resid, resid))
            grad += a0.T @ (resid / s0)
        elif objective_norm == "lognormal_point_mle":
            pred_log = (A_log @ wv) + 1e-18
            resid_log = (np.log(pred_log) - np.log(y_log)) / rel_sigma_log
            f_data = 0.5 * float(np.dot(resid_log, resid_log))
            grad += A_log.T @ (resid_log / (pred_log * rel_sigma_log))
        else:
            raise RuntimeError(f"Internal objective dispatch failure: {objective_norm}")

        f_prior = 0.0
        if ap is not None and bp is not None and ap.size > 0:
            r_prior = (ap @ wv) - bp
            f_prior = 0.5 * float(np.dot(r_prior, r_prior))
            grad += ap.T @ r_prior
        if lam_t9_frac > 0.0 and t9r is not None and t9f is not None and t9r.size > 0:
            y_rows = t9r @ wv
            total = float(np.sum(y_rows))
            denom = max(total + eps_t9_frac, eps_t9_frac)
            frac_fit = y_rows / denom
            resid_frac = frac_fit - t9f
            f_prior += 0.5 * lam_t9_frac * float(np.dot(resid_frac, resid_frac))
            col_sum = np.sum(t9r, axis=0)
            jtr = (t9r.T @ resid_frac) / denom - col_sum * (float(np.dot(resid_frac, y_rows)) / (denom * denom))
            grad += lam_t9_frac * jtr

        return f_data + f_prior, grad

    bounds = [(0.0, None)] * n_comp
    res = minimize(
        _objective_and_grad,
        x0=w0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "maxiter": max(int(maxiter), 1),
            "ftol": max(float(ftol), 0.0),
        },
    )
    # Solver provenance: the primary run is L-BFGS-B; the SLSQP fallback is
    # attempted only when the primary fails on the lognormal objective.
    # method_used names the optimizer whose result is actually returned, so
    # a fit accepted from the fallback is never reported as L-BFGS-B.
    primary_method = "L-BFGS-B"
    primary_success = bool(res.success)
    method_used = primary_method
    fallback_attempted = False
    fallback_success = False
    fallback_note = ""
    if (not bool(res.success)) and objective_norm == "lognormal_point_mle":
        fallback_attempted = True
        res_alt = minimize(
            _objective_and_grad,
            x0=np.maximum(np.asarray(res.x, dtype=float), 0.0),
            method="SLSQP",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": max(int(maxiter), 1),
                "ftol": max(float(ftol), 0.0),
            },
        )
        fallback_success = bool(res_alt.success)
        if fallback_success:
            # The alternate replaces the primary only when the alternate itself
            # reports success and its full result passes validation. A lower
            # objective on a failed run is still a failed fit.
            validate_fit_result(
                np.asarray(res_alt.x, dtype=float),
                n_components=n_comp,
                success=True,
                status=int(getattr(res_alt, "status", 0)),
                message=str(getattr(res_alt, "message", "")),
                objective=float(getattr(res_alt, "fun", float("nan"))),
                context=(
                    f"{objective_norm} SLSQP fallback fit after failed primary "
                    f"L-BFGS-B run (success={bool(res.success)}, "
                    f"status={int(getattr(res, 'status', 0))}, "
                    f"message={str(getattr(res, 'message', ''))!r})"
                ),
            )
            res = res_alt
            method_used = "SLSQP"
        else:
            fallback_note = (
                f"; the SLSQP fallback also reported failure "
                f"(success={bool(res_alt.success)}, "
                f"status={int(getattr(res_alt, 'status', 0))}, "
                f"message={str(getattr(res_alt, 'message', ''))!r})"
            )
    validate_fit_result(
        np.asarray(res.x, dtype=float),
        n_components=n_comp,
        success=bool(res.success),
        status=int(getattr(res, "status", 0)),
        message=str(getattr(res, "message", "")),
        objective=float(getattr(res, "fun", float("nan"))),
        context=f"{objective_norm} fit" + fallback_note,
    )
    w_fit = np.asarray(res.x, dtype=float)
    w_fit[w_fit < 0.0] = 0.0

    def _jacobian_at_solution(w_vec: np.ndarray) -> np.ndarray:
        """Jacobian of the objective's residual vector at the returned weights.

        Formed from the per-point residual derivatives implied by
        ``_objective_and_grad``. For ``gaussian_point_mle`` the per-point
        residual is r_i = (a0_i @ w - y_i) / s_i, so dr_i/dw = a0_i / s_i and
        the Jacobian is the sigma-scaled data matrix stacked on the active
        linear-prior rows (identical to A_lin). For ``lognormal_point_mle``
        the residual is r_i = (log(A_log_i @ w + 1e-18) - log(y_i)) /
        rel_sigma_i, so dr_i/dw = A_log_i / ((A_log_i @ w + 1e-18) *
        rel_sigma_i), evaluated at the returned solution, stacked on the
        active linear-prior rows. The Table-9 fraction term is excluded from
        the Jacobian because it enters the objective only through the
        normalized nonlinear fraction y_rows/sum(y_rows), whose rows carry
        weight-dependent scaling; the design_scope key records that
        exclusion.
        """
        blocks: List[np.ndarray] = []
        if objective_norm == "gaussian_point_mle":
            blocks.append(a0 / s0[:, None])
        else:
            pred_log_sol = (A_log @ w_vec) + 1e-18
            blocks.append(A_log / (pred_log_sol[:, None] * rel_sigma_log[:, None]))
        if ap is not None and bp is not None and ap.size > 0:
            blocks.append(ap)
        return np.vstack(blocks)

    jacobian_condition_number = condition_number_2norm(_jacobian_at_solution(w_fit))
    identifiability = (
        RANK_DEFICIENT_STAMP
        if int(design_diag["design_rank"]) < n_comp  # type: ignore[arg-type]
        else FULL_RANK_STAMP
    )
    info = {
        "objective": objective_norm,
        "method": f"scipy.optimize.minimize({method_used})",
        "method_used": method_used,
        "primary_method": primary_method,
        "primary_success": bool(primary_success),
        "fallback_attempted": fallback_attempted,
        "fallback_method": "SLSQP" if fallback_attempted else None,
        "fallback_success": fallback_success if fallback_attempted else None,
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "nfev": int(getattr(res, "nfev", 0)),
        "nit": int(getattr(res, "nit", 0)),
        "fun": float(getattr(res, "fun", float("nan"))),
        "design_scope": design_scope,
        **design_diag,
        "table9_fraction_active": t9_active,
        "jacobian_condition_number": jacobian_condition_number,
        "n_active_weights": int(np.count_nonzero(w_fit > ACTIVE_WEIGHT_THRESHOLD)),
        "identifiability": identifiability,
    }
    return w_fit, info


def _write_markdown_summary(
    *,
    md_path: Path,
    run_description: str,
    species_data_by_key: Dict[str, SpeciesData],
    per_species_summary: Dict[str, Dict[str, Dict[str, float | int | str]]],
    n_components: int,
    n_active: int,
    global_row: Dict[str, float | int],
    prior_report: Dict[str, object],
    fit_weighting: Dict[str, object],
    fit_solver: Dict[str, object],
    particle_ratios: Dict[str, object] | None = None,
    seagull_metrics: Dict[str, object] | None = None,
) -> None:
    lines: List[str] = []
    lines.append("# Multi-species constrained resonance fit")
    lines.append("")
    lines.append(f"- run: `{run_description}`")
    lines.append(f"- species: `{', '.join(species_data_by_key.keys())}`")
    lines.append(f"- resonance components: `{n_components}` (active: `{n_active}`)")
    lines.append(f"- fit objective: `{str(fit_solver.get('objective', 'nnls_weighted'))}`")
    lines.append(f"- fit solver: `{str(fit_solver.get('method', 'unknown'))}`")
    lines.append(f"- solver success: `{bool(fit_solver.get('success', False))}`")
    lines.append(
        f"- identifiability: `{str(fit_solver.get('identifiability', 'not_recorded'))}`"
    )
    if "design_rank" in fit_solver:
        design_rank_val = fit_solver.get("design_rank")
        design_n_comp_val = fit_solver.get("design_n_components")
        if isinstance(design_rank_val, int) and isinstance(design_n_comp_val, int):
            lines.append(
                f"- design rank: `{design_rank_val}` of `{design_n_comp_val}` "
                f"components ({str(fit_solver.get('design_scope', 'design matrix'))})"
            )
    lines.append(f"- fit weight mode: `{fit_weighting.get('mode', 'sigma')}`")
    if str(fit_weighting.get("mode", "")) == "relative_xfpt_balanced":
        lines.append(
            f"- balance bins: xF `{int(fit_weighting.get('balance_xf_bins', 0))}`, "
            f"pT `{int(fit_weighting.get('balance_pt_bins', 0))}`"
        )
    lines.append("")
    lines.append("## Global resonance metrics")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    for key in (
        "full_weighted_rel_rmse",
        "full_log_pull_rms",
        "full_smape_percent",
        "full_chi2_dof",
    ):
        if key in global_row:
            val = global_row[key]
            lines.append(f"| {key} | {float(val):.6g} |")
    lines.append("")
    lines.append("## Per-species WRRMSE")
    lines.append("")
    lines.append("| species | Resonance(shared) | PYTHIA8 | EPOS | UrQMD |")
    lines.append("|---|---:|---:|---:|---:|")
    for key, sd in species_data_by_key.items():
        by_model = per_species_summary.get(key, {})
        res = by_model.get("Resonance_NNLS_shared", {})
        pythia = by_model.get("PYTHIA8_external", {})
        epos = by_model.get("EPOS_external", {})
        urqmd = by_model.get("UrQMD_external", {})
        def _fmt(d: Dict[str, float | int | str]) -> str:
            v = d.get("full_weighted_rel_rmse")
            if isinstance(v, (int, float)):
                return f"{float(v):.6f}"
            st = d.get("status")
            if isinstance(st, str):
                return st
            return "-"
        lines.append(f"| {sd.name} | {_fmt(res)} | {_fmt(pythia)} | {_fmt(epos)} | {_fmt(urqmd)} |")
    lines.append("")

    if isinstance(particle_ratios, dict) and particle_ratios:
        defs = particle_ratios.get("definitions", {})
        values_default = particle_ratios.get("values", {})
        values_by_window = particle_ratios.get("values_by_window", {})
        windows = particle_ratios.get("windows", {})
        default_window = str(particle_ratios.get("default_window", ""))
        lines.append("## Particle Ratios (Integrated)")
        lines.append("")
        integ = particle_ratios.get("integration")
        if isinstance(integ, str) and integ.strip():
            lines.append(f"- integration rule: `{integ}`")
            lines.append("")
        if isinstance(values_by_window, dict) and values_by_window:
            for win_label, values in values_by_window.items():
                if not isinstance(values, dict) or not values:
                    continue
                win = windows.get(win_label, {}) if isinstance(windows, dict) else {}
                xf_min = win.get("xf_min")
                xf_max = win.get("xf_max")
                title = f"### xF window `{win_label}`"
                if isinstance(xf_min, (int, float)) and isinstance(xf_max, (int, float)):
                    title += f" (`{float(xf_min):.3f}` to `{float(xf_max):.3f}`)"
                if win_label == default_window:
                    title += " [default]"
                lines.append(title)
                lines.append("")
                lines.append("| ratio | Data | Resonance(shared) | PYTHIA8 | EPOS | UrQMD |")
                lines.append("|---|---:|---:|---:|---:|---:|")
                for ratio_name, vals in values.items():
                    if not isinstance(vals, dict):
                        continue
                    ratio_def = defs.get(ratio_name, {}) if isinstance(defs, dict) else {}
                    if isinstance(ratio_def, dict):
                        num = ratio_def.get("numerator", [])
                        den = ratio_def.get("denominator", [])
                        if isinstance(num, list) and isinstance(den, list) and num and den:
                            ratio_label = f"{'+'.join(num)}/{'+'.join(den)}"
                        else:
                            ratio_label = ratio_name
                    else:
                        ratio_label = ratio_name

                    def _fmt_ratio(v: object) -> str:
                        if isinstance(v, (int, float)):
                            return f"{float(v):.6f}"
                        return "-"

                    lines.append(
                        f"| {ratio_label} | "
                        f"{_fmt_ratio(vals.get('Data'))} | "
                        f"{_fmt_ratio(vals.get('Resonance_shared'))} | "
                        f"{_fmt_ratio(vals.get('PYTHIA8_external'))} | "
                        f"{_fmt_ratio(vals.get('EPOS_external'))} | "
                        f"{_fmt_ratio(vals.get('UrQMD_external'))} |"
                    )
                lines.append("")
        elif isinstance(values_default, dict) and values_default:
            lines.append("| ratio | Data | Resonance(shared) | PYTHIA8 | EPOS | UrQMD |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for ratio_name, vals in values_default.items():
                if not isinstance(vals, dict):
                    continue
                ratio_def = defs.get(ratio_name, {}) if isinstance(defs, dict) else {}
                if isinstance(ratio_def, dict):
                    num = ratio_def.get("numerator", [])
                    den = ratio_def.get("denominator", [])
                    if isinstance(num, list) and isinstance(den, list) and num and den:
                        ratio_label = f"{'+'.join(num)}/{'+'.join(den)}"
                    else:
                        ratio_label = ratio_name
                else:
                    ratio_label = ratio_name

                def _fmt_ratio(v: object) -> str:
                    if isinstance(v, (int, float)):
                        return f"{float(v):.6f}"
                    return "-"

                lines.append(
                    f"| {ratio_label} | "
                    f"{_fmt_ratio(vals.get('Data'))} | "
                    f"{_fmt_ratio(vals.get('Resonance_shared'))} | "
                    f"{_fmt_ratio(vals.get('PYTHIA8_external'))} | "
                    f"{_fmt_ratio(vals.get('EPOS_external'))} | "
                    f"{_fmt_ratio(vals.get('UrQMD_external'))} |"
                )
            lines.append("")

    if isinstance(seagull_metrics, dict) and seagull_metrics:
        lines.append("## Seagull Metric")
        lines.append("")
        seag_def = seagull_metrics.get("definition")
        if isinstance(seag_def, str) and seag_def.strip():
            lines.append(f"- definition: `{seag_def}`")
        seag_note = seagull_metrics.get("notes")
        if isinstance(seag_note, str) and seag_note.strip():
            lines.append(f"- note: `{seag_note}`")
        lines.append("")

        values = seagull_metrics.get("values", {})
        if isinstance(values, dict) and values:
            lines.append("### Linear slope d<pT>/dxF")
            lines.append("")
            lines.append("| species | Data | Resonance(shared) | PYTHIA8 | EPOS | UrQMD |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for skey, sd in species_data_by_key.items():
                row = values.get(skey, {}) if isinstance(values, dict) else {}
                models = row.get("models", {}) if isinstance(row, dict) else {}

                def _fmt_slope(model_key: str) -> str:
                    m = models.get(model_key, {}) if isinstance(models, dict) else {}
                    if isinstance(m, dict):
                        v = m.get("linear_slope")
                        if isinstance(v, (int, float)):
                            return f"{float(v):.6f}"
                    return "-"

                lines.append(
                    f"| {sd.name} | "
                    f"{_fmt_slope('Data')} | "
                    f"{_fmt_slope('Resonance_shared')} | "
                    f"{_fmt_slope('PYTHIA8_external')} | "
                    f"{_fmt_slope('EPOS_external')} | "
                    f"{_fmt_slope('UrQMD_external')} |"
                )
            lines.append("")

    lines.append("## Priors")
    lines.append("")
    t9 = prior_report.get("table9", {}) if isinstance(prior_report, dict) else {}
    st = prior_report.get("strange", {}) if isinstance(prior_report, dict) else {}
    ay = prior_report.get("absolute_yield", {}) if isinstance(prior_report, dict) else {}
    ns = prior_report.get("nstar_smooth", {}) if isinstance(prior_report, dict) else {}
    lines.append(f"- Table-9 prior enabled: `{bool(t9.get('enabled', False))}`")
    if bool(t9.get("enabled", False)):
        lines.append(f"- Table-9 lambda: `{float(t9.get('lambda', 0.0)):.6g}`")
        if float(t9.get("lambda_fraction", 0.0)) > 0.0:
            lines.append(
                f"- Table-9 fraction-prior lambda: `{float(t9.get('lambda_fraction', 0.0)):.6g}` "
                + f"(eps `{float(t9.get('fraction_eps', float('nan'))):.3g}`)"
            )
        if float(t9.get("lambda_total", 0.0)) > 0.0:
            lines.append(
                f"- Table-9 total-anchor lambda: `{float(t9.get('lambda_total', 0.0)):.6g}` "
                + f"(target `{float(t9.get('total_target', float('nan'))):.6g}`)"
            )
        if "fit_chi2_per_row" in t9:
            lines.append(
                f"- Table-9 fit chi2/row (yield-space, unscaled): "
                + f"`{float(t9.get('fit_chi2_per_row', float('nan'))):.6g}`"
            )
        if "fit_fraction_l2_per_row" in t9:
            lines.append(
                f"- Table-9 fraction L2/row: "
                + f"`{float(t9.get('fit_fraction_l2_per_row', float('nan'))):.6g}`"
            )
    lines.append(f"- Strange prior enabled: `{bool(st.get('enabled', False))}`")
    if bool(st.get("enabled", False)):
        lines.append(f"- Strange lambda (K-/K+): `{float(st.get('lambda_kaon_ratio', 0.0)):.6g}`")
        lines.append(f"- Hyperon-balance lambda: `{float(st.get('lambda_hyperon_balance', 0.0)):.6g}`")
        fit = st.get("fit", {})
        if isinstance(fit, dict) and "kminus_over_kplus" in fit:
            lines.append(
                f"- Fitted K-/K+: `{float(fit['kminus_over_kplus']):.6g}` "
                f"(target `{float(st.get('kminus_over_kplus_target', float('nan'))):.6g}`)"
            )
    lines.append(f"- Absolute-yield prior enabled: `{bool(ay.get('enabled', False))}`")
    if bool(ay.get("enabled", False)):
        lines.append(f"- Absolute-yield lambda: `{float(ay.get('lambda', 0.0)):.6g}`")
        lines.append(
            f"- Absolute-yield window: `[{float(ay.get('xf_min', float('nan'))):.3f}, "
            + f"{float(ay.get('xf_max', float('nan'))):.3f}]`"
        )
        src = str(ay.get("source", ""))
        if src:
            lines.append(f"- Absolute-yield source: `{src}`")
        fit_rows = ay.get("fit_rows", [])
        if isinstance(fit_rows, list) and fit_rows:
            lines.append("")
            lines.append("| abs-yield species | target | fit | pull |")
            lines.append("|---|---:|---:|---:|")
            for rr in fit_rows:
                if not isinstance(rr, dict):
                    continue
                lines.append(
                    f"| {rr.get('species_name', rr.get('species_key', ''))} | "
                    f"{float(rr.get('target', float('nan'))):.6g} | "
                    f"{float(rr.get('fit', float('nan'))):.6g} | "
                    f"{float(rr.get('pull', float('nan'))):.6g} |"
                )
    lines.append(f"- N* smoothness prior enabled: `{bool(ns.get('enabled', False))}`")
    if bool(ns.get("enabled", False)):
        lines.append(f"- N* smoothness lambda: `{float(ns.get('lambda', 0.0)):.6g}`")
        lines.append(f"- N* smoothness rows: `{int(ns.get('n_rows', 0))}`")
        if "n_ladders" in ns:
            lines.append(f"- N* ladders covered: `{int(ns.get('n_ladders', 0))}`")
        fit_ns = ns.get("fit", {})
        if isinstance(fit_ns, dict):
            if "adjacent_diff_rms" in fit_ns:
                lines.append(f"- Fitted adjacent diff RMS: `{float(fit_ns.get('adjacent_diff_rms', float('nan'))):.6g}`")
            if "adjacent_diff_max_abs" in fit_ns:
                lines.append(
                    f"- Fitted adjacent diff max |delta|: `{float(fit_ns.get('adjacent_diff_max_abs', float('nan'))):.6g}`"
                )
    lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_provenance_mode(
    run_path: Path,
    *,
    repro_lock_arg: str,
    mixing2006_dir: Path,
) -> int:
    """Audit one run directory from its provenance record; return the exit code.

    The audit validates the record structure, recomputes the sha256 of every
    recorded input, re-checks the recorded lock-file digests against the
    files on disk, and cross-checks the recorded input hashes against the
    reproducibility manifest when that manifest is reachable. Prints one line
    per problem and returns 1, or prints a summary line and returns 0.
    """
    provenance_path = run_path if run_path.is_file() else (run_path / RUN_PROVENANCE_FILENAME)
    problems: List[str] = validate_run_provenance(provenance_path)
    record: Dict[str, object] = {}
    try:
        loaded = json.loads(provenance_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            record = loaded
    except (OSError, ValueError):
        record = {}
    if not problems:
        recorded_inputs = [
            Path(str(entry.get("path")))
            for entry in record.get("inputs", [])
            if isinstance(entry, dict) and str(entry.get("path", "")).strip()
        ]
        problems = validate_run_provenance(provenance_path, expected_input_paths=recorded_inputs)
        lock_path = (
            Path(repro_lock_arg)
            if str(repro_lock_arg).strip()
            else (mixing2006_dir / "sps_resonances_paper" / "reproducibility.lock.json")
        )
        manifest_info = manifest_input_hashes(
            lock_path,
            web_root=REPO_ROOT / "spshadrons.web.cern.ch",
            config_roots=(REPO_ROOT, mixing2006_dir),
        )
        manifest_state = str(manifest_info.get("manifest", "absent"))
        if manifest_state == "absent":
            print(f"manifest note: no reproducibility manifest at {lock_path}; input cross-check skipped")
        elif manifest_state == "unreadable":
            problems.append(
                f"{lock_path}: reproducibility manifest is present but not readable JSON"
            )
        else:
            problems.extend(
                manifest_crosscheck_problems(record, manifest_info.get("entries", {}))
            )
    if problems:
        for problem in problems:
            print(f"provenance problem: {problem}")
        print(f"provenance INVALID: {provenance_path} ({len(problems)} problem(s))")
        return 1
    n_inputs = len(record.get("inputs", [])) if isinstance(record, dict) else 0
    n_locks = len(record.get("lock_files", [])) if isinstance(record, dict) else 0
    print(f"provenance OK: {provenance_path} (inputs={n_inputs}, lock_files={n_locks})")
    return 0


def _refuse_on_manifest_mismatch(manifest_check: Dict[str, object]) -> None:
    """Raise SystemExit when a consumed input contradicts the manifest.

    The refusal message names every offending file with the recorded and the
    freshly computed sha256, so the operator can restore or re-record the
    input. An absent manifest never reaches this branch.
    """
    if str(manifest_check.get("manifest", "")) != "mismatched":
        return
    mismatches = manifest_check.get("mismatches", [])
    details = "; ".join(
        f"{entry.get('path')} expected {entry.get('expected')}, found {entry.get('actual')}"
        for entry in mismatches
        if isinstance(entry, dict)
    )
    raise SystemExit(
        "refusing to consume an input whose sha256 does not match the "
        f"reproducibility manifest ({manifest_check.get('manifest_path', '')}): {details}"
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Constrained multi-species resonance benchmark.")
    parser.add_argument("--run-description", default="multispecies_constrained_default")
    parser.add_argument("--out-dir", default="", help="Output directory; defaults to dev/00runs/<run_description>")
    parser.add_argument(
        "--species",
        default="pi-,pi+,k+,k-,p,pbar",
        help=f"CSV species list; supported: {', '.join(canonical_species_keys())}",
    )
    parser.add_argument("--seed", type=int, default=20260329)
    parser.add_argument("--beam-momentum", type=float, default=158.0)
    parser.add_argument(
        "--convention",
        choices=["invariant", "raw"],
        default="invariant",
        help="Counts-to-observable convention: invariant (corrected, default) or raw (superseded pre-audit).",
    )
    parser.add_argument(
        "--weak-feed-policy",
        choices=["inclusive_weak", "primary_only"],
        default="primary_only",
        help=(
            "Weak-decay feed-down policy for resonance component generation: "
            "'primary_only' drops tracks with any weak decay in their cascade "
            "ancestry; 'inclusive_weak' keeps all generated tracks."
        ),
    )
    parser.add_argument(
        "--data-system",
        choices=list(available_data_systems()),
        default="pp",
        help="Which NA49 data tables to benchmark against (pp or pC/pPb where published).",
    )
    # --- p+A wounded-nucleon generation ---------------------------------
    parser.add_argument(
        "--allow-system-mismatch",
        action="store_true",
        help=(
            "Permit --data-system and --pa-a combinations that mix p+p "
            "generation with nuclear tables (diagnostic only)."
        ),
    )
    parser.add_argument("--pa-a", type=int, default=0,
                        help="Target mass number A; 0 keeps p+p generation.")
    parser.add_argument("--pa-z", type=int, default=1)
    parser.add_argument("--pa-name", default="",
                        help="Label recorded in manifests (e.g. Pb208).")
    parser.add_argument("--pa-profile", default="woods_saxon",
                        choices=["point", "woods_saxon", "harmonic_oscillator"])
    parser.add_argument("--pa-radius-fm", type=float, default=6.62)
    parser.add_argument("--pa-diffuseness-fm", type=float, default=0.546)
    parser.add_argument("--pa-quad-alpha", type=float, default=0.0)
    parser.add_argument("--sigma-nn-inel-mb", type=float, default=31.7)
    parser.add_argument("--target-wound-range", default="",
                        help="Centrality slice on wounded target nucleons, 'lo:hi' inclusive; empty = minimum bias.")
    parser.add_argument(
        "--train-frac",
        type=float,
        default=1.0,
        help="Deprecated and ignored; constrained benchmark uses all points for fitting/evaluation.",
    )
    parser.add_argument("--xf-min", type=float, default=0.0)
    parser.add_argument("--xf-max", type=float, default=0.4)
    parser.add_argument("--pt-min", type=float, default=0.05)
    parser.add_argument("--pt-max", type=float, default=1.5)
    parser.add_argument(
        "--ratio-xf-windows",
        default="0.0:0.4",
        help=(
            "Comma-separated xF windows for integrated ratio diagnostics, e.g. "
            "'0.0:0.4,0.0:0.1,0.1:0.2,0.2:0.3,0.3:0.4'. "
            "Boundary overlap with data bins is accounted for in integration."
        ),
    )
    parser.add_argument(
        "--fit-objective",
        choices=FIT_OBJECTIVES,
        default="nnls_weighted",
        help=(
            "Shared resonance objective: "
            "'nnls_weighted' (current weighted NNLS), "
            "'gaussian_point_mle' (point-level Gaussian likelihood), "
            "'lognormal_point_mle' (point-level lognormal likelihood on positive points)."
        ),
    )
    parser.add_argument(
        "--fit-weight-mode",
        choices=bss.FIT_WEIGHT_MODES,
        default="sigma",
        help=(
            "Objective row-weighting mode for constrained fit: "
            "'sigma' (1/sigma), 'relative' (y/sigma), "
            "'relative_xfpt_balanced' (relative with balanced xF/pT macro-bins)."
        ),
    )
    parser.add_argument(
        "--fit-opt-maxiter",
        type=int,
        default=3000,
        help="Maximum iterations for non-linear point-likelihood objectives.",
    )
    parser.add_argument(
        "--fit-opt-ftol",
        type=float,
        default=1e-10,
        help="Function tolerance for non-linear point-likelihood objectives.",
    )
    parser.add_argument(
        "--fit-balance-xf-bins",
        type=int,
        default=4,
        help="Number of rank bins in xF used for relative_xfpt_balanced mode.",
    )
    parser.add_argument(
        "--fit-balance-pt-bins",
        type=int,
        default=4,
        help="Number of rank bins in pT used for relative_xfpt_balanced mode.",
    )
    parser.add_argument(
        "--allow-rank-deficient",
        action="store_true",
        help=(
            "Accept a rank-deficient fit design instead of refusing it. Every "
            "summary and details artifact from such a run carries "
            f"identifiability={RANK_DEFICIENT_STAMP!r} in its fit_solver "
            "payload, so the fit can never be presented as identifiable."
        ),
    )
    parser.add_argument("--mixing2006-dir", default=str(REPO_ROOT))
    parser.add_argument("--res-events-per-res", type=int, default=30000)
    parser.add_argument("--res-max-decay-passes", type=int, default=8)
    parser.add_argument("--res-workers", type=int, default=1)
    parser.add_argument(
        "--res-fragmentation-mode",
        choices=bss.FRAGMENTATION_MODES,
        default="both",
        help="Event-level resonance source mode for component generation.",
    )
    parser.add_argument(
        "--res-split-tower-by-source",
        action="store_true",
        help=(
            "Split effective high-mass N* tower components into projectile/target "
            "subcomponents (requires --res-fragmentation-mode both for effect)."
        ),
    )
    parser.add_argument(
        "--res-target-feedover-xf-max",
        type=float,
        default=-1.0,
        help=(
            "If >=0, keep target-source tracks in projectile hemisphere only up to xF<=value. "
            "Negative disables."
        ),
    )
    parser.add_argument(
        "--res-projectile-feedover-xf-max",
        type=float,
        default=-1.0,
        help=(
            "If >=0, keep projectile-source tracks in target hemisphere only up to |xF|<=value. "
            "Negative disables."
        ),
    )
    parser.add_argument("--res-cache-dir", default="", help="Shared component cache path")
    parser.add_argument("--res-cache-force-rebuild", action="store_true")
    parser.add_argument("--res-disable-regge-tower", action="store_true")
    parser.add_argument("--res-regge-tower-min-mass", type=float, default=1.9)
    parser.add_argument("--res-regge-tower-max-mass", type=float, default=2.3)
    parser.add_argument(
        "--res-regge-tower-level-count",
        type=int,
        default=3,
        help="Number of effective N* tower levels to materialize in production templates.",
    )
    parser.add_argument(
        "--res-regge-tower-shape",
        choices=bss.REGGE_TOWER_SHAPES,
        default="linear",
        help=(
            "Mass-ladder shape for effective N* tower: "
            "linear in m, linear in m^2 (regge_m2), or Hagedorn-like exponential density (hagedorn)."
        ),
    )
    parser.add_argument(
        "--res-regge-tower-shape-param",
        type=float,
        default=0.25,
        help=(
            "Shape parameter used for --res-regge-tower-shape=hagedorn "
            "(interpreted as T_H in GeV for rho(m)~exp(m/T_H))."
        ),
    )
    parser.add_argument(
        "--res-regge-tower-offshell-kappa",
        type=float,
        default=0.0,
        help=(
            "Stage-1 off-shell broadening strength for effective N* ladder widths: "
            "Gamma_eff = Gamma_base * (1 + kappa * x^beta), x in [0,1] along the ladder."
        ),
    )
    parser.add_argument(
        "--res-regge-tower-offshell-beta",
        type=float,
        default=1.0,
        help="Exponent beta in the off-shell width broadening law for effective N* ladder levels.",
    )
    parser.add_argument(
        "--res-regge-tower-offshell-window-scale",
        type=float,
        default=1.0,
        help=(
            "Scale factor for truncated mass-window bounds in tower generator files. "
            "Values >1 broaden the sampled off-shell window."
        ),
    )
    parser.add_argument(
        "--res-soft-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override soft resonance branching values (repeatable).",
    )
    parser.add_argument(
        "--prior-table9-lambda",
        type=float,
        default=0.0,
        help="Weight for Table-9 composition prior (0 disables).",
    )
    parser.add_argument(
        "--prior-table9-json",
        default=str(DEFAULT_TABLE9_JSON),
        help="Table-9 resonance-yield JSON used for composition prior.",
    )
    parser.add_argument(
        "--prior-table9-total-lambda",
        type=float,
        default=0.0,
        help=(
            "Optional weight for Table-9 subset total pi- anchor. "
            "Use >0 to prevent trivial near-zero occupancy in the homogeneous composition prior."
        ),
    )
    parser.add_argument(
        "--prior-table9-total-target",
        type=float,
        default=0.0,
        help=(
            "Target total pi- yield for the Table-9 subset anchor. "
            "If <=0 and --prior-table9-total-lambda>0, defaults to the Table-9 total from the selected JSON."
        ),
    )
    parser.add_argument(
        "--prior-table9-fraction-lambda",
        type=float,
        default=0.0,
        help=(
            "Optional nonlinear Table-9 fraction prior weight. "
            "Applies residuals on normalized selector fractions and prevents homogeneous collapse."
        ),
    )
    parser.add_argument(
        "--prior-table9-fraction-eps",
        type=float,
        default=1e-6,
        help="Stability epsilon for nonlinear Table-9 fraction prior denominator.",
    )
    parser.add_argument(
        "--prior-strange-lambda",
        type=float,
        default=0.0,
        help="Weight for global K-/K+ ratio prior (0 disables).",
    )
    parser.add_argument(
        "--prior-hyperon-lambda",
        type=float,
        default=0.0,
        help="Weight for net-hyperon vs net-kaon balance prior (0 disables).",
    )
    parser.add_argument(
        "--prior-kminus-over-kplus-target",
        type=float,
        default=DEFAULT_KMINUS_OVER_KPLUS,
        help="Target global K-/K+ multiplicity ratio for strange prior.",
    )
    parser.add_argument(
        "--prior-abs-yield-lambda",
        type=float,
        default=0.0,
        help="Weight for absolute integrated-yield prior (0 disables).",
    )
    parser.add_argument(
        "--prior-abs-yield-json",
        default="",
        help=(
            "Optional JSON defining absolute-yield targets by species and optional uncertainties/window. "
            "If omitted, targets are derived from the selected NA49 points in --prior-abs-yield-window."
        ),
    )
    parser.add_argument(
        "--prior-abs-yield-window",
        default="0.0:0.4",
        help="Single xF window min:max used for absolute-yield prior integration.",
    )
    parser.add_argument(
        "--prior-abs-yield-species",
        default="",
        help=(
            "Comma-separated species list for absolute-yield prior "
            "(keys/names/tags accepted). Default: all loaded species."
        ),
    )
    parser.add_argument(
        "--prior-abs-yield-min-sigma-rel",
        type=float,
        default=0.02,
        help="Minimum relative uncertainty floor for absolute-yield prior rows.",
    )
    parser.add_argument(
        "--prior-nstar-smooth-lambda",
        type=float,
        default=0.0,
        help=(
            "Weight for effective N* ladder smoothness prior. "
            "Adds adjacent-level penalties w(Li)-w(Li+1)=0 for matched N*/anti-N* branches."
        ),
    )
    parser.add_argument(
        "--hyperon-pdgs",
        default="3122,3212,3222,3112,3312,3322,3334",
        help="Comma-separated absolute PDGs treated as hyperons in strange prior.",
    )
    parser.add_argument(
        "--external-run-dir",
        default=str(RUNS_ROOT / "gw_multispecies_singlepass_hi_2026-03-29"),
        help="Directory containing merged external CSVs (epos_lhc_pp158_<tag>_gateway.csv etc.).",
    )
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument(
        "--validate-provenance",
        default="",
        metavar="RUN_DIR",
        help=(
            "Audit mode: validate the run_provenance.json in RUN_DIR (or the "
            "file itself when given a file path), recompute every recorded "
            "input hash, and cross-check against the reproducibility "
            "manifest; exits 0 on a consistent run and 1 with the problem "
            "list otherwise. No fit runs in this mode."
        ),
    )
    parser.add_argument(
        "--repro-lock",
        default="",
        metavar="PATH",
        help=(
            "Path to reproducibility.lock.json; default is "
            "<mixing2006-dir>/sps_resonances_paper/reproducibility.lock.json. "
            "When the file is absent the run records 'manifest: absent' in "
            "its provenance and continues."
        ),
    )
    parser.add_argument(
        "--requirements-lock",
        default="",
        metavar="PATH",
        help=(
            "Pinned requirements file digested into the run provenance; "
            "default is <mixing2006-dir>/sps_resonances_paper/"
            "requirements-paper.lock.txt, and an absent file is recorded as "
            "absent."
        ),
    )
    parser.add_argument(
        "--source-archive",
        default="",
        metavar="DIGEST",
        help=(
            "64-hex sha256 digest of the source archive identifying the code "
            "a run came from, for runs from a release archive without git "
            "metadata; also read from the NA49_RUN_SOURCE_ARCHIVE environment "
            "variable. When git metadata is absent and no digest is supplied, "
            "the sha256 of the archive's SHA256SUMS.txt file inventory is "
            "recorded automatically instead."
        ),
    )
    return parser


def build_details_npz_payload(
    *,
    component_weights: Sequence[float],
    component_keys: Sequence[str],
    component_labels: Sequence[str],
    species_data_by_key: Dict[str, SpeciesData],
    pred_full_stack: np.ndarray,
    y_full_stack: np.ndarray,
    sigma_full_stack: np.ndarray,
    n_effective: int,
    prior_report: Dict[str, object],
    fit_solver: Dict[str, object],
    fit_weight_mode: str,
    fit_objective: str,
    weak_feed_policy: str,
    histogram_convention: str,
    external_pred_by_model: Dict[str, Dict[str, np.ndarray]],
    external_sigma_by_model: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """Assemble the ``multispecies_constrained_details.npz`` dataset mapping.

    String metadata is stored as Unicode-dtype arrays (never object dtype), and
    an integer ``schema_version`` marker is included, so every consumer can
    load the file pickle-free via
    :func:`helpers.benchmarking.sps_core.load_multispecies_details_npz`.
    """
    payload: Dict[str, np.ndarray] = {
        "schema_version": np.asarray(DETAILS_NPZ_SCHEMA_VERSION, dtype=np.int64),
        "component_weights": np.asarray(component_weights, dtype=float),
        "component_keys": np.asarray(list(component_keys), dtype=str),
        "component_labels": np.asarray(list(component_labels), dtype=str),
        "species_keys": np.asarray(list(species_data_by_key.keys()), dtype=str),
        "resonance_global_full_pred": pred_full_stack,
        "resonance_global_full_y": y_full_stack,
        "resonance_global_full_sigma": sigma_full_stack,
        "n_effective": np.asarray(int(n_effective), dtype=np.int64),
        "prior_report_json": np.asarray(json.dumps(prior_report, sort_keys=True), dtype=str),
        "fit_weight_mode": np.asarray(str(fit_weight_mode), dtype=str),
        "fit_objective": np.asarray(str(fit_objective), dtype=str),
        "meta_weak_feed_policy": np.asarray(str(weak_feed_policy), dtype=str),
        "meta_histogram_convention": np.asarray(str(histogram_convention), dtype=str),
        "fit_solver_json": np.asarray(json.dumps(fit_solver, sort_keys=True), dtype=str),
    }
    for key, sd in species_data_by_key.items():
        tag = sd.tag
        payload[f"{tag}_xf"] = sd.xf
        payload[f"{tag}_pt"] = sd.pt
        payload[f"{tag}_y"] = sd.y
        payload[f"{tag}_sigma"] = sd.sigma
        payload[f"{tag}_full_idx"] = np.arange(len(sd.y), dtype=np.int64)
        payload[f"{tag}_pred_res"] = sd.pred_res
        payload[f"{tag}_pred_res_sigma"] = sd.pred_res_sigma
        pred_py = external_pred_by_model["PYTHIA8_external"].get(key)
        if pred_py is not None:
            payload[f"{tag}_pred_pythia"] = pred_py
        sig_py = external_sigma_by_model["PYTHIA8_external"].get(key)
        if sig_py is not None:
            payload[f"{tag}_pred_pythia_sigma"] = sig_py
        pred_ep = external_pred_by_model["EPOS_external"].get(key)
        if pred_ep is not None:
            payload[f"{tag}_pred_epos"] = pred_ep
        sig_ep = external_sigma_by_model["EPOS_external"].get(key)
        if sig_ep is not None:
            payload[f"{tag}_pred_epos_sigma"] = sig_ep
        pred_uq = external_pred_by_model["UrQMD_external"].get(key)
        if pred_uq is not None:
            payload[f"{tag}_pred_urqmd"] = pred_uq
        sig_uq = external_sigma_by_model["UrQMD_external"].get(key)
        if sig_uq is not None:
            payload[f"{tag}_pred_urqmd_sigma"] = sig_uq
    return payload


def main() -> None:
    args = _build_arg_parser().parse_args()
    if str(args.validate_provenance).strip():
        raise SystemExit(
            _validate_provenance_mode(
                Path(str(args.validate_provenance)),
                repro_lock_arg=str(args.repro_lock),
                mixing2006_dir=Path(args.mixing2006_dir),
            )
        )
    global CONVENTION, SQRT_S_GEV
    CONVENTION = str(args.convention)
    SQRT_S_GEV = bss._sqrt_s_from_fixed_target_beam_momentum(float(args.beam_momentum))

    species_keys = _parse_species_csv(args.species)
    try:
        validate_data_system_generation(
            str(args.data_system),
            int(args.pa_a),
            allow_mismatch=bool(args.allow_system_mismatch),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if str(args.data_system) != "pp":
        unavailable = []
        for key in species_keys:
            try:
                na49_table_for(key, str(args.data_system))
            except ValueError:
                unavailable.append(key)
        if unavailable:
            raise SystemExit(
                f"--data-system {args.data_system}: no NA49 tables for "
                f"species {sorted(unavailable)}"
            )
    ratio_xf_windows = _parse_xf_windows(args.ratio_xf_windows)
    out_dir = Path(args.out_dir) if args.out_dir else (RUNS_ROOT / args.run_description)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.res_cache_dir) if args.res_cache_dir else (out_dir / "resonance_component_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    mixing2006_dir = Path(args.mixing2006_dir)
    repro_lock_path = (
        Path(args.repro_lock)
        if str(args.repro_lock).strip()
        else (mixing2006_dir / "sps_resonances_paper" / "reproducibility.lock.json")
    )
    requirements_lock_path = (
        Path(args.requirements_lock)
        if str(args.requirements_lock).strip()
        else (mixing2006_dir / "sps_resonances_paper" / "requirements-paper.lock.txt")
    )
    consumed_input_paths: List[Path] = []

    soft_decay_overrides = bss._parse_soft_decay_overrides(list(args.res_soft_override))

    # Load species data first so each species keeps its own NA49 binning.
    species_data_by_key: Dict[str, SpeciesData] = {}
    for key in species_keys:
        species_data_by_key[key] = _load_species_points(
            species_key=key,
            xf_min=args.xf_min,
            xf_max=args.xf_max,
            pt_min=args.pt_min,
            pt_max=args.pt_max,
            seed=args.seed,
            data_system=str(args.data_system),
        )
    for sd in species_data_by_key.values():
        consumed_input_paths.append(Path(sd.na49_path))

    # Record the manifest verdict on the NA49 tables before any further
    # work: a mismatched table must stop the run before the expensive
    # component cache is touched.
    manifest_info = manifest_input_hashes(
        repro_lock_path,
        web_root=REPO_ROOT / "spshadrons.web.cern.ch",
        config_roots=(REPO_ROOT, mixing2006_dir),
    )
    manifest_check = check_inputs_against_manifest(consumed_input_paths, manifest_info)
    _refuse_on_manifest_mismatch(manifest_check)

    abs_targets_by_key: Dict[str, float] = {}
    abs_sigmas_by_key: Dict[str, float] = {}
    abs_window_override: Tuple[float, float] | None = None
    abs_targets_json_path = ""
    abs_targets_source = "na49_data_window_integral"
    if str(args.prior_abs_yield_json).strip():
        p_abs = Path(str(args.prior_abs_yield_json)).expanduser()
        if not p_abs.is_absolute():
            p_abs = (REPO_ROOT / p_abs).resolve()
        if not p_abs.exists():
            raise FileNotFoundError(f"Missing --prior-abs-yield-json: {p_abs}")
        abs_payload = _load_abs_yield_targets_json(
            p_abs,
            species_data_by_key=species_data_by_key,
        )
        abs_targets_by_key = {
            str(k): float(v) for k, v in dict(abs_payload.get("targets_by_key", {})).items()
        }
        abs_sigmas_by_key = {
            str(k): float(v) for k, v in dict(abs_payload.get("sigmas_by_key", {})).items()
        }
        xf_min_json = abs_payload.get("xf_min")
        xf_max_json = abs_payload.get("xf_max")
        if isinstance(xf_min_json, (int, float)) and isinstance(xf_max_json, (int, float)):
            abs_window_override = (float(xf_min_json), float(xf_max_json))
        abs_targets_json_path = str(p_abs)
        abs_targets_source = "json_targets"
        consumed_input_paths.append(p_abs)

    regge_tower_gen_files: Optional[Dict[str, Path]] = None
    if not bool(args.res_disable_regge_tower):
        regge_tower_gen_files = bss._materialize_regge_tower_gen_files(
            mixing2006_dir=mixing2006_dir,
            tower_gen_root=out_dir / "tower_gen",
            min_mass=float(args.res_regge_tower_min_mass),
            max_mass=float(args.res_regge_tower_max_mass),
            level_count=int(args.res_regge_tower_level_count),
            shape=str(args.res_regge_tower_shape),
            shape_param=float(args.res_regge_tower_shape_param),
            offshell_kappa=float(args.res_regge_tower_offshell_kappa),
            offshell_beta=float(args.res_regge_tower_offshell_beta),
            offshell_window_scale=float(args.res_regge_tower_offshell_window_scale),
        )
        masses = bss._compute_regge_tower_mass_ladder(
            float(args.res_regge_tower_min_mass),
            float(args.res_regge_tower_max_mass),
            level_count=int(args.res_regge_tower_level_count),
            shape=str(args.res_regge_tower_shape),
            shape_param=float(args.res_regge_tower_shape_param),
        )
        print(
            "resonance-tower: "
            + f"levels={int(args.res_regge_tower_level_count)} "
            + f"shape={str(args.res_regge_tower_shape)} "
            + f"mass-range=[{float(np.min(masses)):.3f}, {float(np.max(masses)):.3f}] GeV "
            + f"offshell(kappa={float(args.res_regge_tower_offshell_kappa):.3f}, "
            + f"beta={float(args.res_regge_tower_offshell_beta):.3f}, "
            + f"window_scale={float(args.res_regge_tower_offshell_window_scale):.3f})",
            flush=True,
        )

    # Build global component cache exactly once.
    pa_nucleus_dict: Optional[Dict[str, object]] = None
    pa_wound_range: Optional[Tuple[int, int]] = None
    if int(args.pa_a) > 0:
        pa_nucleus_dict = {
            "name": str(args.pa_name) or f"{args.pa_profile}{args.pa_a}",
            "A": int(args.pa_a),
            "Z": int(args.pa_z),
            "profile": str(args.pa_profile),
            "radius_fm": float(args.pa_radius_fm),
            "diffuseness_fm": float(args.pa_diffuseness_fm),
            "quad_alpha": float(args.pa_quad_alpha),
            "sigma_nn_inel_mb": float(args.sigma_nn_inel_mb),
        }
        if str(args.target_wound_range).strip():
            lo_s, hi_s = str(args.target_wound_range).split(":")
            pa_wound_range = (int(lo_s), int(hi_s))
        print(
            "pa-wounded: "
            f"nucleus={pa_nucleus_dict['name']} A={args.pa_a} Z={args.pa_z} "
            f"profile={args.pa_profile} sigmaNN={args.sigma_nn_inel_mb} mb "
            f"wounds={pa_wound_range or 'min-bias'}",
            flush=True,
        )
    elif str(args.target_wound_range).strip():
        raise SystemExit(
            "--target-wound-range requires --pa-a (no nucleus in p+p mode)"
        )

    all_specs = bss._all_resonance_specs(
        mixing2006_dir,
        include_regge_tower=(not bool(args.res_disable_regge_tower)),
        regge_tower_gen_files=regge_tower_gen_files,
    )
    cache_index = bss._build_resonance_component_cache(
        mixing2006_dir=mixing2006_dir,
        cache_dir=cache_dir,
        specs_by_key=all_specs,
        n_events_per_res=int(args.res_events_per_res),
        seed=int(args.seed + 19),
        max_decay_passes=int(args.res_max_decay_passes),
        fragmentation_mode=str(args.res_fragmentation_mode),
        force_rebuild=bool(args.res_cache_force_rebuild),
        res_workers=max(int(args.res_workers), 1),
        soft_decay_overrides=soft_decay_overrides,
        weak_feed_policy=str(args.weak_feed_policy),
        nucleus=pa_nucleus_dict,
        target_wound_range=pa_wound_range,
        cache_namespace=bss._canonical_nucleus_namespace(
            pa_nucleus_dict, pa_wound_range, data_system=str(args.data_system)
        ),
    )
    consumed_input_paths.extend(cache_index.values())
    component_keys: List[str] = []
    component_labels: List[str] = []
    pi_minus_per_component_list: List[float] = []
    kplus_per_component_list: List[float] = []
    kminus_per_component_list: List[float] = []
    hyperon_net_per_component_list: List[float] = []

    hyperon_abs_pdgs: List[int] = []
    for txt in str(args.hyperon_pdgs).split(","):
        t = txt.strip()
        if not t:
            continue
        hyperon_abs_pdgs.append(abs(int(t)))
    hyperon_abs_set = set(hyperon_abs_pdgs)

    species_cols: Dict[str, List[np.ndarray]] = {k: [] for k in species_data_by_key.keys()}
    zero_column_component_labels: List[str] = []
    split_tower_by_source = bool(args.res_split_tower_by_source) and str(args.res_fragmentation_mode) == "both"

    # Fill columns component-by-component (single cache read per component), with optional
    # tower-side splitting and feed-over filtering.
    for key in sorted(cache_index.keys()):
        # Cache keys may carry a pA namespace prefix ("{ns}|{orig}");
        # recover the original specs_by_key key for the label.
        spec_lookup_key = key.split("|", 1)[1] if "|" in key else key
        base_label = str(all_specs[spec_lookup_key].label)
        cache_path = cache_index[key]
        with np.load(cache_path, mmap_mode="r", allow_pickle=False) as z:
            pdg = np.asarray(z["pdg"], dtype=np.int32)
            xf = np.asarray(z["xf"], dtype=float)
            pt = np.asarray(z["pt"], dtype=float)
            if "source_side" in z:
                source_side = np.asarray(z["source_side"], dtype=np.int8)
            else:
                source_side = _infer_source_side_from_xf(xf)

        keep_feed = _feedover_keep_mask(
            xf,
            source_side,
            target_feedover_xf_max=float(args.res_target_feedover_xf_max),
            projectile_feedover_xf_max=float(args.res_projectile_feedover_xf_max),
        )
        if not np.any(keep_feed):
            continue

        subdefs: List[tuple[str, np.ndarray]] = []
        if split_tower_by_source and _is_tower_component_label(base_label):
            subdefs = [
                ("proj", keep_feed & (source_side > 0)),
                ("targ", keep_feed & (source_side < 0)),
            ]
        else:
            subdefs = [("", keep_feed)]

        for suffix, mask_sub in subdefs:
            if not np.any(mask_sub):
                continue

            col_label = f"{base_label}_{suffix}" if suffix else base_label
            col_key = f"{key}::{suffix}" if suffix else key

            xf_sub = xf[mask_sub]
            pt_sub = pt[mask_sub]
            pdg_sub = pdg[mask_sub]

            # Assemble the per-species design columns first. A component that
            # produces no tracks of any loaded species would enter the design
            # as an identically zero column, which the identifiability policy
            # refuses. Such a component carries no information for the loaded
            # species, so it is dropped here and the decision is recorded in
            # the summary payload.
            species_counts_by_key = {
                skey: _species_column_from_tracks(
                    pdg_arr=pdg_sub,
                    xf_arr=xf_sub,
                    pt_arr=pt_sub,
                    sd=sd,
                    convention=CONVENTION,
                    sqrt_s_gev=SQRT_S_GEV,
                )
                for skey, sd in species_data_by_key.items()
            }
            if all(not np.any(counts) for counts in species_counts_by_key.values()):
                zero_column_component_labels.append(col_label)
                continue

            nev = float(args.res_events_per_res)
            if nev > 0:
                pi_minus_per_component_list.append(float(np.count_nonzero(pdg_sub == -211)) / nev)
                kplus_per_component_list.append(float(np.count_nonzero(pdg_sub == 321)) / nev)
                kminus_per_component_list.append(float(np.count_nonzero(pdg_sub == -321)) / nev)
                if hyperon_abs_set:
                    abs_pdg_sub = np.abs(pdg_sub)
                    m_h = np.isin(abs_pdg_sub, list(hyperon_abs_set))
                    if np.any(m_h):
                        net = float(np.sum(np.sign(pdg_sub[m_h])))
                        hyperon_net_per_component_list.append(net / nev)
                    else:
                        hyperon_net_per_component_list.append(0.0)
                else:
                    hyperon_net_per_component_list.append(0.0)
            else:
                pi_minus_per_component_list.append(0.0)
                kplus_per_component_list.append(0.0)
                kminus_per_component_list.append(0.0)
                hyperon_net_per_component_list.append(0.0)

            component_keys.append(col_key)
            component_labels.append(col_label)

            for skey, counts in species_counts_by_key.items():
                species_cols[skey].append(counts)

    if zero_column_component_labels:
        print(
            "component-note: dropped "
            f"{len(zero_column_component_labels)} component(s) that produce no "
            "tracks of any loaded species in the fit window (their design "
            f"columns would be identically zero): {zero_column_component_labels}",
            flush=True,
        )

    n_comp = len(component_labels)
    if n_comp == 0:
        raise RuntimeError("No resonance components available in cache for constrained fit.")

    pi_minus_per_component = np.asarray(pi_minus_per_component_list, dtype=float)
    kplus_per_component = np.asarray(kplus_per_component_list, dtype=float)
    kminus_per_component = np.asarray(kminus_per_component_list, dtype=float)
    hyperon_net_per_component = np.asarray(hyperon_net_per_component_list, dtype=float)

    for skey, sd in species_data_by_key.items():
        cols = species_cols[skey]
        if cols:
            sd.matrix = np.column_stack(cols)
        else:
            sd.matrix = np.zeros((len(sd.y), 0), dtype=float)

    # Build stacked full-sample matrices:
    # - raw point rows (for point-likelihood objectives),
    # - weighted rows (for legacy weighted NNLS objective).
    a_rows_raw: List[np.ndarray] = []
    y_rows_raw: List[np.ndarray] = []
    sigma_rows_raw: List[np.ndarray] = []
    a_rows_weighted: List[np.ndarray] = []
    b_rows_weighted: List[np.ndarray] = []
    fit_row_scale_full_by_key: Dict[str, np.ndarray] = {}
    fit_weight_full_meta_by_key: Dict[str, Dict[str, float | int | str]] = {}
    for sd in species_data_by_key.values():
        full_idx = np.arange(len(sd.y), dtype=np.int64)
        full_matrix = sd.matrix[full_idx]
        full_y = sd.y[full_idx]
        full_sigma = sd.sigma[full_idx]
        a_rows_raw.append(full_matrix)
        y_rows_raw.append(full_y)
        sigma_rows_raw.append(full_sigma)

        row_scale, row_meta = bss._compute_fit_row_scales(
            y=full_y,
            sigma=full_sigma,
            xf=sd.xf[full_idx],
            pt=sd.pt[full_idx],
            mode=str(args.fit_weight_mode),
            balance_xf_bins=int(args.fit_balance_xf_bins),
            balance_pt_bins=int(args.fit_balance_pt_bins),
        )
        fit_row_scale_full_by_key[sd.key] = row_scale
        fit_weight_full_meta_by_key[sd.key] = row_meta
        a_rows_weighted.append(full_matrix * row_scale[:, None])
        b_rows_weighted.append(full_y * row_scale)
    A_train_raw = np.vstack(a_rows_raw)
    y_train_raw = np.concatenate(y_rows_raw)
    sigma_train_raw = np.concatenate(sigma_rows_raw)
    A_train_weighted = np.vstack(a_rows_weighted)
    b_train_weighted = np.concatenate(b_rows_weighted)

    prior_rows: List[np.ndarray] = []
    prior_targets: List[float] = []
    prior_report: Dict[str, object] = {
        "table9": {"enabled": False},
        "strange": {"enabled": False},
        "absolute_yield": {"enabled": False},
        "nstar_smooth": {"enabled": False},
    }
    abs_yield_eval_rows: List[Dict[str, object]] = []
    table9_fraction_rows: np.ndarray | None = None
    table9_fraction_targets: np.ndarray | None = None

    lam_table9 = max(float(args.prior_table9_lambda), 0.0)
    lam_table9_total = max(float(args.prior_table9_total_lambda), 0.0)
    lam_table9_frac = max(float(args.prior_table9_fraction_lambda), 0.0)
    eps_table9_frac = max(float(args.prior_table9_fraction_eps), 1e-12)
    if lam_table9 > 0.0 or lam_table9_total > 0.0 or lam_table9_frac > 0.0:
        table9_json = Path(args.prior_table9_json)
        if table9_json.exists():
            consumed_input_paths.append(table9_json)
            t9 = _build_table9_composition_prior(
                component_labels=component_labels,
                pi_minus_per_component=pi_minus_per_component,
                table9_json=table9_json,
            )
            if bool(t9.get("ok", False)):
                A_t9 = np.asarray(t9["A"], dtype=float)
                b_t9 = np.asarray(t9["b"], dtype=float)
                n_comp_rows = 0
                if lam_table9 > 0.0:
                    sc = float(np.sqrt(lam_table9))
                    for i in range(A_t9.shape[0]):
                        prior_rows.append(sc * A_t9[i])
                        prior_targets.append(sc * float(b_t9[i]))
                    n_comp_rows = int(A_t9.shape[0])
                if lam_table9_frac > 0.0:
                    table9_fraction_rows = np.asarray(t9.get("raw_rows", np.zeros((0, n_comp), dtype=float)), dtype=float)
                    table9_fraction_targets = np.asarray(t9.get("target_fractions", np.zeros(0, dtype=float)), dtype=float)
                n_total_rows = 0
                total_target = float(args.prior_table9_total_target)
                if lam_table9_total > 0.0:
                    if total_target <= 0.0:
                        total_target = float(t9.get("table_total", 0.0))
                    total_vec = np.asarray(t9.get("total_vec", np.zeros(n_comp, dtype=float)), dtype=float)
                    if np.any(np.abs(total_vec) > 0.0):
                        sc_total = float(np.sqrt(lam_table9_total))
                        prior_rows.append(sc_total * total_vec)
                        prior_targets.append(sc_total * float(total_target))
                        n_total_rows = 1
                prior_report["table9"] = {
                    "enabled": True,
                    "lambda": lam_table9,
                    "lambda_total": lam_table9_total,
                    "lambda_fraction": lam_table9_frac,
                    "fraction_eps": eps_table9_frac,
                    "total_target": _as_float_or_none(total_target),
                    "n_rows": int(n_comp_rows + n_total_rows),
                    "n_rows_composition": int(n_comp_rows),
                    "n_rows_total_anchor": int(n_total_rows),
                    "table_json": str(table9_json),
                    "row_meta": t9.get("meta", []),
                    "table_total": _as_float_or_none(float(t9.get("table_total", float("nan")))),
                }
            else:
                prior_report["table9"] = {
                    "enabled": False,
                    "lambda": lam_table9,
                    "lambda_total": lam_table9_total,
                    "lambda_fraction": lam_table9_frac,
                    "fraction_eps": eps_table9_frac,
                    "table_json": str(table9_json),
                    "reason": str(t9.get("reason", "unknown")),
                }
        else:
            prior_report["table9"] = {
                "enabled": False,
                "lambda": lam_table9,
                "lambda_total": lam_table9_total,
                "lambda_fraction": lam_table9_frac,
                "fraction_eps": eps_table9_frac,
                "table_json": str(table9_json),
                "reason": "missing_json",
            }

    lam_strange = max(float(args.prior_strange_lambda), 0.0)
    lam_hyperon = max(float(args.prior_hyperon_lambda), 0.0)
    if lam_strange > 0.0 or lam_hyperon > 0.0:
        sp = _build_strange_hyperon_priors(
            kplus_per_component=kplus_per_component,
            kminus_per_component=kminus_per_component,
            hyperon_net_per_component=hyperon_net_per_component,
            kminus_over_kplus_target=float(args.prior_kminus_over_kplus_target),
        )
        if lam_strange > 0.0:
            prior_rows.append(float(np.sqrt(lam_strange)) * np.asarray(sp["kaon_ratio_row"], dtype=float))
            prior_targets.append(float(np.sqrt(lam_strange)) * float(sp["kaon_ratio_target"]))
        if lam_hyperon > 0.0:
            prior_rows.append(float(np.sqrt(lam_hyperon)) * np.asarray(sp["hyperon_balance_row"], dtype=float))
            prior_targets.append(float(np.sqrt(lam_hyperon)) * float(sp["hyperon_balance_target"]))
        prior_report["strange"] = {
            "enabled": True,
            "lambda_kaon_ratio": lam_strange,
            "lambda_hyperon_balance": lam_hyperon,
            "kminus_over_kplus_target": float(args.prior_kminus_over_kplus_target),
            "hyperon_abs_pdgs": sorted(list(hyperon_abs_set)),
        }

    lam_abs_yield = max(float(args.prior_abs_yield_lambda), 0.0)
    if lam_abs_yield > 0.0:
        xf_lo, xf_hi = _parse_single_xf_window(str(args.prior_abs_yield_window))
        if abs_window_override is not None:
            xf_lo, xf_hi = abs_window_override
        selected_abs_species = _resolve_abs_yield_species_csv(
            str(args.prior_abs_yield_species),
            species_data_by_key,
        )
        ay = _build_absolute_yield_prior(
            species_data_by_key=species_data_by_key,
            selected_species_keys=selected_abs_species,
            xf_min=float(xf_lo),
            xf_max=float(xf_hi),
            target_overrides=abs_targets_by_key,
            sigma_overrides=abs_sigmas_by_key,
            min_sigma_rel=float(args.prior_abs_yield_min_sigma_rel),
        )
        if bool(ay.get("ok", False)):
            A_ay = np.asarray(ay["A"], dtype=float)
            b_ay = np.asarray(ay["b"], dtype=float)
            sc = float(np.sqrt(lam_abs_yield))
            for i in range(A_ay.shape[0]):
                prior_rows.append(sc * A_ay[i])
                prior_targets.append(sc * float(b_ay[i]))
            abs_yield_eval_rows = list(ay.get("eval_rows", []))
            prior_report["absolute_yield"] = {
                "enabled": True,
                "lambda": float(lam_abs_yield),
                "xf_min": float(ay.get("xf_min", float("nan"))),
                "xf_max": float(ay.get("xf_max", float("nan"))),
                "source": str(abs_targets_source),
                "targets_json": str(abs_targets_json_path),
                "min_sigma_rel": float(ay.get("min_sigma_rel", float(args.prior_abs_yield_min_sigma_rel))),
                "n_rows": int(A_ay.shape[0]),
                "species_keys": list(selected_abs_species),
                "row_meta": list(ay.get("row_meta", [])),
            }
        else:
            prior_report["absolute_yield"] = {
                "enabled": False,
                "lambda": float(lam_abs_yield),
                "source": str(abs_targets_source),
                "targets_json": str(abs_targets_json_path),
                "reason": str(ay.get("reason", "unknown")),
            }

    lam_nstar_smooth = max(float(args.prior_nstar_smooth_lambda), 0.0)
    if lam_nstar_smooth > 0.0:
        ns = _build_nstar_ladder_smooth_prior(component_labels=component_labels)
        if bool(ns.get("ok", False)):
            A_ns = np.asarray(ns["A"], dtype=float)
            b_ns = np.asarray(ns["b"], dtype=float)
            sc = float(np.sqrt(lam_nstar_smooth))
            for i in range(A_ns.shape[0]):
                prior_rows.append(sc * A_ns[i])
                prior_targets.append(sc * float(b_ns[i]))
            prior_report["nstar_smooth"] = {
                "enabled": True,
                "lambda": float(lam_nstar_smooth),
                "n_rows": int(A_ns.shape[0]),
                "n_ladders": int(ns.get("n_ladders", 0)),
                "row_meta": list(ns.get("row_meta", [])),
            }
        else:
            prior_report["nstar_smooth"] = {
                "enabled": False,
                "lambda": float(lam_nstar_smooth),
                "reason": str(ns.get("reason", "unknown")),
            }

    A_prior = np.vstack(prior_rows) if prior_rows else None
    b_prior = np.asarray(prior_targets, dtype=float) if prior_rows else None

    if str(args.fit_objective) != "nnls_weighted" and str(args.fit_weight_mode) != "sigma":
        print(
            "fit-note: point-likelihood objective ignores --fit-weight-mode for resonance weights; "
            "it always uses raw point-level likelihood with reported NA49 uncertainties.",
            flush=True,
        )

    # Final manifest verdict over every input consumed so far (including the
    # Table-9 JSON), recorded verbatim in the run provenance.
    manifest_check = check_inputs_against_manifest(consumed_input_paths, manifest_info)
    _refuse_on_manifest_mismatch(manifest_check)

    w, fit_solver = _fit_global_nonnegative_weights(
        objective=str(args.fit_objective),
        A_train_raw=A_train_raw,
        y_train=y_train_raw,
        sigma_train=sigma_train_raw,
        A_train_weighted=A_train_weighted,
        b_train_weighted=b_train_weighted,
        A_prior=A_prior,
        b_prior=b_prior,
        table9_fraction_rows=table9_fraction_rows,
        table9_fraction_targets=table9_fraction_targets,
        table9_fraction_lambda=lam_table9_frac,
        table9_fraction_eps=eps_table9_frac,
        maxiter=int(args.fit_opt_maxiter),
        ftol=float(args.fit_opt_ftol),
        allow_rank_deficient=bool(args.allow_rank_deficient),
    )
    n_eff = int(np.sum(w > 1e-12))
    n_eff_for_metrics = max(n_eff, 1)

    # Predict and score per species.
    per_species_summary: Dict[str, Dict[str, Dict[str, float | int | str]]] = {}
    for key, sd in species_data_by_key.items():
        full_idx = np.arange(len(sd.y), dtype=np.int64)
        sd.pred_res = sd.matrix @ w
        # Poisson propagation from per-component cached counts:
        # Var(sum_k w_k * n_k) = sum_k w_k^2 * n_k
        sd.pred_res_sigma = np.sqrt(np.maximum(sd.matrix @ np.asarray(w, dtype=float) ** 2, 0.0))
        row_res: Dict[str, float | int | str] = {
            "model": f"Resonance_shared_{str(args.fit_objective)}",
            "n_params": n_eff,
        }
        row_res.update(
            bss._metrics_bundle(
                sd.pred_res[full_idx],
                sd.y[full_idx],
                sd.sigma[full_idx],
                n_params=n_eff_for_metrics,
                prefix="full",
            )
        )
        per_species_summary[key] = {"Resonance_NNLS_shared": row_res}

    # Global resonance metrics on full stacked vectors.
    pred_full_stack = np.concatenate([sd.pred_res for sd in species_data_by_key.values()])
    y_full_stack = np.concatenate([sd.y for sd in species_data_by_key.values()])
    sigma_full_stack = np.concatenate([sd.sigma for sd in species_data_by_key.values()])
    global_row: Dict[str, float | int] = {"n_params": n_eff}
    global_row.update(
        bss._metrics_bundle(
            pred_full_stack,
            y_full_stack,
            sigma_full_stack,
            n_params=n_eff_for_metrics,
            prefix="full",
        )
    )

    if bool(prior_report.get("table9", {}).get("enabled", False)):
        table9_meta = prior_report["table9"].get("row_meta", [])
        row_vals: List[float] = []
        for item in table9_meta:
            selector = str(item.get("selector", ""))
            m = _label_match_mask(component_labels, selector)
            vec = np.zeros(n_comp, dtype=float)
            vec[m] = pi_minus_per_component[m]
            row_vals.append(float(vec @ w))
        total_val = float(sum(row_vals))
        chi2_raw = 0.0
        chi2_frac = 0.0
        rows_out = []
        for item, val in zip(table9_meta, row_vals):
            frac_fit = (val / total_val) if total_val > 0 else float("nan")
            frac_tgt = float(item.get("table_fraction", float("nan")))
            resid_raw = val - (frac_tgt * total_val)
            resid_frac = frac_fit - frac_tgt if np.isfinite(frac_fit) and np.isfinite(frac_tgt) else float("nan")
            if np.isfinite(resid_raw):
                chi2_raw += float(resid_raw * resid_raw)
            if np.isfinite(resid_frac):
                chi2_frac += float(resid_frac * resid_frac)
            rows_out.append(
                {
                    "name": str(item.get("name", "")),
                    "selector": str(item.get("selector", "")),
                    "fit_pi_minus_yield": val,
                    "fit_fraction": _as_float_or_none(frac_fit),
                    "target_fraction": _as_float_or_none(frac_tgt),
                    "fit_minus_target_fraction": _as_float_or_none(resid_frac),
                    "fit_minus_target_yield": _as_float_or_none(resid_raw),
                }
            )
        prior_report["table9"]["fit_total_pi_minus"] = total_val
        prior_report["table9"]["fit_rows"] = rows_out
        prior_report["table9"]["fit_chi2"] = float(chi2_raw)
        prior_report["table9"]["fit_chi2_per_row"] = float(chi2_raw / max(len(rows_out), 1))
        prior_report["table9"]["fit_fraction_l2"] = float(chi2_frac)
        prior_report["table9"]["fit_fraction_l2_per_row"] = float(chi2_frac / max(len(rows_out), 1))

    if bool(prior_report.get("strange", {}).get("enabled", False)):
        kplus_tot = float(kplus_per_component @ w)
        kminus_tot = float(kminus_per_component @ w)
        hnet_tot = float(hyperon_net_per_component @ w)
        ratio_fit = (kminus_tot / kplus_tot) if kplus_tot > 0 else float("nan")
        prior_report["strange"]["fit"] = {
            "kplus_total": kplus_tot,
            "kminus_total": kminus_tot,
            "kminus_over_kplus": _as_float_or_none(ratio_fit),
            "kminus_over_kplus_minus_target": _as_float_or_none(
                ratio_fit - float(args.prior_kminus_over_kplus_target)
            ),
            "hyperon_net_total": hnet_tot,
            "net_kaon_excess": kplus_tot - kminus_tot,
            "hyperon_balance_residual": hnet_tot - (kplus_tot - kminus_tot),
        }

    if bool(prior_report.get("absolute_yield", {}).get("enabled", False)) and abs_yield_eval_rows:
        rows_out: List[Dict[str, object]] = []
        chi2_abs = 0.0
        for item in abs_yield_eval_rows:
            vec = np.asarray(item.get("vec", np.zeros(n_comp, dtype=float)), dtype=float)
            target = float(item.get("target", float("nan")))
            sigma = max(float(item.get("sigma", float("nan"))), 1e-12)
            fit_val = float(vec @ w)
            pull = (fit_val - target) / sigma
            chi2_abs += float(pull * pull)
            rows_out.append(
                {
                    "species_key": str(item.get("species_key", "")),
                    "species_name": str(item.get("species_name", "")),
                    "target": float(target),
                    "sigma": float(sigma),
                    "fit": float(fit_val),
                    "fit_minus_target": float(fit_val - target),
                    "pull": float(pull),
                }
            )
        prior_report["absolute_yield"]["fit_rows"] = rows_out
        prior_report["absolute_yield"]["fit_chi2"] = float(chi2_abs)
        prior_report["absolute_yield"]["fit_chi2_per_row"] = float(chi2_abs / max(len(rows_out), 1))

    if bool(prior_report.get("nstar_smooth", {}).get("enabled", False)):
        ns_meta = prior_report["nstar_smooth"].get("row_meta", [])
        diffs: List[float] = []
        rows_out: List[Dict[str, object]] = []
        if isinstance(ns_meta, list):
            for item in ns_meta:
                if not isinstance(item, dict):
                    continue
                i = int(item.get("index_lo", -1))
                j = int(item.get("index_hi", -1))
                if i < 0 or j < 0 or i >= n_comp or j >= n_comp:
                    continue
                d = float(w[i] - w[j])
                diffs.append(d)
                rows_out.append(
                    {
                        "ladder": str(item.get("ladder", "")),
                        "level_lo": int(item.get("level_lo", -1)),
                        "level_hi": int(item.get("level_hi", -1)),
                        "label_lo": str(item.get("label_lo", "")),
                        "label_hi": str(item.get("label_hi", "")),
                        "weight_lo": float(w[i]),
                        "weight_hi": float(w[j]),
                        "diff": d,
                    }
                )
        if diffs:
            arr = np.asarray(diffs, dtype=float)
            prior_report["nstar_smooth"]["fit"] = {
                "adjacent_diff_rms": float(np.sqrt(np.mean(arr * arr))),
                "adjacent_diff_max_abs": float(np.max(np.abs(arr))),
                "rows": rows_out,
            }

    # Optional external model scoring (per-species independent normalization).
    ext_dir = Path(args.external_run_dir)
    external_pred_by_model: Dict[str, Dict[str, np.ndarray]] = {
        "PYTHIA8_external": {},
        "EPOS_external": {},
        "UrQMD_external": {},
    }
    external_sigma_by_model: Dict[str, Dict[str, np.ndarray]] = {
        "PYTHIA8_external": {},
        "EPOS_external": {},
        "UrQMD_external": {},
    }
    if not args.skip_external:
        for key, sd in species_data_by_key.items():
            pythia_csv = ext_dir / f"pythia8_softqcd_pp158_{sd.tag}_gateway.csv"
            epos_csv = ext_dir / f"epos_lhc_pp158_{sd.tag}_gateway.csv"
            urqmd_csv = ext_dir / f"urqmd_pp158_{sd.tag}_gateway.csv"
            row_scale_external = (
                fit_row_scale_full_by_key.get(key) if str(args.fit_objective) == "nnls_weighted" else None
            )
            if pythia_csv.exists():
                consumed_input_paths.append(pythia_csv)
                try:
                    row_py, pred_py, raw_py, alpha_py = _fit_single_external_model(
                        sd,
                        pythia_csv,
                        row_scale_full=row_scale_external,
                        return_pred=True,
                    )
                    per_species_summary[key]["PYTHIA8_external"] = row_py
                    external_pred_by_model["PYTHIA8_external"][key] = pred_py
                    external_sigma_by_model["PYTHIA8_external"][key] = np.abs(float(alpha_py)) * np.sqrt(
                        np.maximum(np.asarray(raw_py, dtype=float), 0.0)
                    )
                except Exception as exc:
                    per_species_summary[key]["PYTHIA8_external"] = {"status": f"failed:{exc}"}
            else:
                per_species_summary[key]["PYTHIA8_external"] = {"status": f"missing_file:{pythia_csv}"}
            if epos_csv.exists():
                consumed_input_paths.append(epos_csv)
                try:
                    row_ep, pred_ep, raw_ep, alpha_ep = _fit_single_external_model(
                        sd,
                        epos_csv,
                        row_scale_full=row_scale_external,
                        return_pred=True,
                    )
                    per_species_summary[key]["EPOS_external"] = row_ep
                    external_pred_by_model["EPOS_external"][key] = pred_ep
                    external_sigma_by_model["EPOS_external"][key] = np.abs(float(alpha_ep)) * np.sqrt(
                        np.maximum(np.asarray(raw_ep, dtype=float), 0.0)
                    )
                except Exception as exc:
                    per_species_summary[key]["EPOS_external"] = {"status": f"failed:{exc}"}
            else:
                per_species_summary[key]["EPOS_external"] = {"status": f"missing_file:{epos_csv}"}
            if urqmd_csv.exists():
                consumed_input_paths.append(urqmd_csv)
                try:
                    row_uq, pred_uq, raw_uq, alpha_uq = _fit_single_external_model(
                        sd,
                        urqmd_csv,
                        row_scale_full=row_scale_external,
                        return_pred=True,
                    )
                    per_species_summary[key]["UrQMD_external"] = row_uq
                    external_pred_by_model["UrQMD_external"][key] = pred_uq
                    external_sigma_by_model["UrQMD_external"][key] = np.abs(float(alpha_uq)) * np.sqrt(
                        np.maximum(np.asarray(raw_uq, dtype=float), 0.0)
                    )
                except Exception as exc:
                    per_species_summary[key]["UrQMD_external"] = {"status": f"failed:{exc}"}
            else:
                per_species_summary[key]["UrQMD_external"] = {"status": f"missing_file:{urqmd_csv}"}

    particle_ratio_summary = _build_particle_ratio_summary(
        species_data_by_key=species_data_by_key,
        xf_windows=ratio_xf_windows,
    )
    seagull_summary = _build_seagull_summary(
        species_data_by_key=species_data_by_key,
        external_pred_by_model=external_pred_by_model,
        external_sigma_by_model=external_sigma_by_model,
        sqrt_s_from_beam_momentum=float(
            bss._sqrt_s_from_fixed_target_beam_momentum(float(args.beam_momentum))
        ),
    )

    weights_by_key = {key: float(val) for key, val in zip(component_keys, w)}
    weights_by_label_sum: Dict[str, float] = {}
    for lbl, val in zip(component_labels, w):
        weights_by_label_sum[lbl] = float(weights_by_label_sum.get(lbl, 0.0) + float(val))
    active_weights_by_key = {k: v for k, v in weights_by_key.items() if v > 1e-12}
    active_weights_by_key_sorted = dict(sorted(active_weights_by_key.items(), key=lambda kv: kv[1], reverse=True))
    active_weights_by_label_sorted = dict(
        sorted(
            ((k, v) for k, v in weights_by_label_sum.items() if v > 1e-12),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )

    summary_json = out_dir / "multispecies_constrained_summary.json"
    details_npz = out_dir / "multispecies_constrained_details.npz"
    summary_md = out_dir / "MULTISPECIES_CONSTRAINED_SUMMARY.md"
    summary_csv = out_dir / "multispecies_constrained_wrrmse.csv"
    ratio_csv = out_dir / "multispecies_constrained_particle_ratios.csv"
    seagull_slopes_csv = out_dir / "multispecies_constrained_seagull_slopes.csv"
    seagull_profiles_csv = out_dir / "multispecies_constrained_seagull_profiles.csv"

    payload = {
        "weak_feed_policy": str(args.weak_feed_policy),
        "histogram_convention": str(CONVENTION),
        "numpy_version": str(np.__version__),
        "numpy_bit_generator": type(np.random.default_rng().bit_generator).__name__,
        "config": vars(args),
        "species": {
            key: {
                "name": sd.name,
                "tag": sd.tag,
                "pdg": int(sd.pdg),
                "na49_table": str(sd.na49_path),
                "n_points": int(len(sd.y)),
                "full_points": int(len(sd.y)),
            }
            for key, sd in species_data_by_key.items()
        },
        "derived": {
            "sqrt_s_from_beam_momentum": float(
                bss._sqrt_s_from_fixed_target_beam_momentum(float(args.beam_momentum))
            )
        },
        "resonance": {
            "n_components": int(n_comp),
            "n_active": int(n_eff),
            "dropped_zero_column_components": zero_column_component_labels,
            "component_keys": component_keys,
            "component_labels": component_labels,
            "weights_by_key": weights_by_key,
            "weights_by_label_sum": weights_by_label_sum,
            "active_weights_by_key_sorted": active_weights_by_key_sorted,
            "active_weights_by_label_sorted": active_weights_by_label_sorted,
            "global_metrics": global_row,
            "priors": prior_report,
        },
        "fit_weighting": {
            "mode": str(args.fit_weight_mode),
            "balance_xf_bins": int(args.fit_balance_xf_bins),
            "balance_pt_bins": int(args.fit_balance_pt_bins),
            "per_species_full_meta": fit_weight_full_meta_by_key,
        },
        "fit_solver": fit_solver,
        "per_species": per_species_summary,
        "particle_ratios": particle_ratio_summary,
        "seagull_metrics": seagull_summary,
        "soft_decay_overrides": soft_decay_overrides,
    }
    summary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Every completed run records its provenance next to the summary, before
    # any other artifact could fail and leave the directory unauditable.
    provenance_record = write_run_provenance(
        out_dir,
        input_paths=consumed_input_paths,
        argv=sys.argv,
        env_var_names=PROVENANCE_ENV_ALLOWLIST,
        rng={
            "type": "numpy.random.Generator",
            "bit_generator": type(np.random.default_rng().bit_generator).__name__,
            "seeds": {
                "base_seed": int(args.seed),
                "component_cache_seed": int(args.seed) + 19,
            },
            "derivation": "component-cache seed = base seed + 19; per-species "
            "seeds derive from _stable_seed(base_seed, species tag)",
        },
        source_record=collect_source_record(
            outer_repo=REPO_ROOT,
            nested_repo=mixing2006_dir,
            source_archive=str(
                args.source_archive or os.environ.get("NA49_RUN_SOURCE_ARCHIVE", "")
            ),
        ),
        lock_records=lock_file_records(
            [
                ("reproducibility_manifest", repro_lock_path),
                ("requirements_lock", requirements_lock_path),
            ]
        ),
        input_verification=manifest_check,
        producer="helpers.benchmarking.benchmark_multispecies_constrained",
    )
    provenance_json = out_dir / RUN_PROVENANCE_FILENAME

    npz_payload = build_details_npz_payload(
        component_weights=w,
        component_keys=component_keys,
        component_labels=component_labels,
        species_data_by_key=species_data_by_key,
        pred_full_stack=pred_full_stack,
        y_full_stack=y_full_stack,
        sigma_full_stack=sigma_full_stack,
        n_effective=n_eff,
        prior_report=prior_report,
        fit_solver=fit_solver,
        fit_weight_mode=str(args.fit_weight_mode),
        fit_objective=str(args.fit_objective),
        weak_feed_policy=str(args.weak_feed_policy),
        histogram_convention=CONVENTION,
        external_pred_by_model=external_pred_by_model,
        external_sigma_by_model=external_sigma_by_model,
    )
    np.savez_compressed(details_npz, **npz_payload)

    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "species",
                "resonance_shared_full_wrrmse",
                "pythia_full_wrrmse",
                "epos_full_wrrmse",
                "urqmd_full_wrrmse",
            ]
        )
        for key, sd in species_data_by_key.items():
            by_model = per_species_summary.get(key, {})
            res = by_model.get("Resonance_NNLS_shared", {}).get("full_weighted_rel_rmse", "")
            pythia = by_model.get("PYTHIA8_external", {}).get("full_weighted_rel_rmse", "")
            epos = by_model.get("EPOS_external", {}).get("full_weighted_rel_rmse", "")
            urqmd = by_model.get("UrQMD_external", {}).get("full_weighted_rel_rmse", "")
            writer.writerow([sd.name, res, pythia, epos, urqmd])

    with ratio_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["window", "xf_min", "xf_max", "ratio", "data", "resonance_shared", "pythia8", "epos", "urqmd"]
        )
        defs = particle_ratio_summary.get("definitions", {}) if isinstance(particle_ratio_summary, dict) else {}
        windows = particle_ratio_summary.get("windows", {}) if isinstance(particle_ratio_summary, dict) else {}
        values_by_window = (
            particle_ratio_summary.get("values_by_window", {}) if isinstance(particle_ratio_summary, dict) else {}
        )
        if isinstance(values_by_window, dict):
            for window, values in values_by_window.items():
                if not isinstance(values, dict):
                    continue
                win = windows.get(window, {}) if isinstance(windows, dict) else {}
                xf_min = win.get("xf_min")
                xf_max = win.get("xf_max")
                for ratio_name, vals in values.items():
                    ratio_label = ratio_name
                    ratio_def = defs.get(ratio_name, {}) if isinstance(defs, dict) else {}
                    if isinstance(ratio_def, dict):
                        num = ratio_def.get("numerator", [])
                        den = ratio_def.get("denominator", [])
                        if isinstance(num, list) and isinstance(den, list) and num and den:
                            ratio_label = f"{'+'.join(num)}/{'+'.join(den)}"
                    if isinstance(vals, dict):
                        writer.writerow(
                            [
                                window,
                                xf_min,
                                xf_max,
                                ratio_label,
                                vals.get("Data"),
                                vals.get("Resonance_shared"),
                                vals.get("PYTHIA8_external"),
                                vals.get("EPOS_external"),
                                vals.get("UrQMD_external"),
                            ]
                        )

    with seagull_slopes_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "species",
                "model",
                "n_xf_bins",
                "linear_slope",
                "linear_intercept",
                "delta_first_last",
                "quadratic_coeff",
            ]
        )
        values = seagull_summary.get("values", {}) if isinstance(seagull_summary, dict) else {}
        if isinstance(values, dict):
            for skey, row in values.items():
                species_name = skey
                models = {}
                if isinstance(row, dict):
                    species_name = str(row.get("species_name", skey))
                    models = row.get("models", {})
                if not isinstance(models, dict):
                    continue
                for model_key, m in models.items():
                    if not isinstance(m, dict):
                        continue
                    writer.writerow(
                        [
                            species_name,
                            model_key,
                            m.get("n_xf_bins"),
                            m.get("linear_slope"),
                            m.get("linear_intercept"),
                            m.get("delta_first_last"),
                            m.get("quadratic_coeff"),
                        ]
                    )

    with seagull_profiles_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["species", "model", "xf", "pt_mean", "pt_mean_err"])
        values = seagull_summary.get("values", {}) if isinstance(seagull_summary, dict) else {}
        if isinstance(values, dict):
            for skey, row in values.items():
                species_name = skey
                models = {}
                if isinstance(row, dict):
                    species_name = str(row.get("species_name", skey))
                    models = row.get("models", {})
                if not isinstance(models, dict):
                    continue
                for model_key, m in models.items():
                    if not isinstance(m, dict):
                        continue
                    xf_vals = m.get("xf")
                    pt_vals = m.get("pt_mean")
                    pt_err_vals = m.get("pt_mean_err")
                    if not isinstance(xf_vals, list) or not isinstance(pt_vals, list):
                        continue
                    if not isinstance(pt_err_vals, list):
                        pt_err_vals = [float("nan")] * len(pt_vals)
                    n = min(len(xf_vals), len(pt_vals), len(pt_err_vals))
                    for i in range(n):
                        writer.writerow([species_name, model_key, xf_vals[i], pt_vals[i], pt_err_vals[i]])

    _write_markdown_summary(
        md_path=summary_md,
        run_description=args.run_description,
        species_data_by_key=species_data_by_key,
        per_species_summary=per_species_summary,
        n_components=n_comp,
        n_active=n_eff,
        global_row=global_row,
        prior_report=prior_report,
        fit_weighting=payload["fit_weighting"],
        fit_solver=payload["fit_solver"],
        particle_ratios=particle_ratio_summary,
        seagull_metrics=seagull_summary,
    )

    print(
        "Fit config:",
        {
            "objective": str(args.fit_objective),
            "weight_mode": str(args.fit_weight_mode),
            "balance_xf_bins": int(args.fit_balance_xf_bins),
            "balance_pt_bins": int(args.fit_balance_pt_bins),
            "solver": fit_solver,
        },
    )
    print(f"Saved summary: {summary_json}")
    print(f"Saved provenance: {provenance_json}")
    print(f"Saved details: {details_npz}")
    print(f"Saved markdown: {summary_md}")
    print(f"Saved csv: {summary_csv}")
    print(f"Saved ratio csv: {ratio_csv}")
    print(f"Saved seagull slopes csv: {seagull_slopes_csv}")
    print(f"Saved seagull profiles csv: {seagull_profiles_csv}")
    print("Global resonance metrics:")
    print(global_row)
    print("Prior summary:")
    print(prior_report)
    print("Per-species WRRMSE (resonance_shared / PYTHIA8 / EPOS / UrQMD):")
    for key, sd in species_data_by_key.items():
        by_model = per_species_summary.get(key, {})
        res = by_model.get("Resonance_NNLS_shared", {}).get("full_weighted_rel_rmse", "n/a")
        pythia = by_model.get("PYTHIA8_external", {}).get("full_weighted_rel_rmse", "n/a")
        epos = by_model.get("EPOS_external", {}).get("full_weighted_rel_rmse", "n/a")
        urqmd = by_model.get("UrQMD_external", {}).get("full_weighted_rel_rmse", "n/a")
        print(f"  {sd.name}: {res} / {pythia} / {epos} / {urqmd}")
    print("Integrated particle ratios:")
    ratio_values = particle_ratio_summary.get("values", {}) if isinstance(particle_ratio_summary, dict) else {}
    ratio_values_by_window = (
        particle_ratio_summary.get("values_by_window", {}) if isinstance(particle_ratio_summary, dict) else {}
    )
    ratio_windows = particle_ratio_summary.get("windows", {}) if isinstance(particle_ratio_summary, dict) else {}
    ratio_defs = particle_ratio_summary.get("definitions", {}) if isinstance(particle_ratio_summary, dict) else {}
    if isinstance(ratio_values_by_window, dict) and ratio_values_by_window:
        for win_label, win_vals in ratio_values_by_window.items():
            win = ratio_windows.get(win_label, {}) if isinstance(ratio_windows, dict) else {}
            if isinstance(win, dict):
                xf_min = win.get("xf_min")
                xf_max = win.get("xf_max")
                if isinstance(xf_min, (int, float)) and isinstance(xf_max, (int, float)):
                    print(f"  window {win_label} [{float(xf_min):.3f}, {float(xf_max):.3f}]")
                else:
                    print(f"  window {win_label}")
            else:
                print(f"  window {win_label}")
            if not isinstance(win_vals, dict):
                continue
            for ratio_name, vals in win_vals.items():
                label = ratio_name
                rdef = ratio_defs.get(ratio_name, {}) if isinstance(ratio_defs, dict) else {}
                if isinstance(rdef, dict):
                    num = rdef.get("numerator", [])
                    den = rdef.get("denominator", [])
                    if isinstance(num, list) and isinstance(den, list) and num and den:
                        label = f"{'+'.join(num)}/{'+'.join(den)}"
                if isinstance(vals, dict):
                    print(
                        f"    {label}: data={vals.get('Data', 'n/a')} "
                        f"res={vals.get('Resonance_shared', 'n/a')} "
                        f"py={vals.get('PYTHIA8_external', 'n/a')} "
                        f"ep={vals.get('EPOS_external', 'n/a')} "
                        f"uq={vals.get('UrQMD_external', 'n/a')}"
                    )
    elif isinstance(ratio_values, dict) and ratio_values:
        for ratio_name, vals in ratio_values.items():
            label = ratio_name
            rdef = ratio_defs.get(ratio_name, {}) if isinstance(ratio_defs, dict) else {}
            if isinstance(rdef, dict):
                num = rdef.get("numerator", [])
                den = rdef.get("denominator", [])
                if isinstance(num, list) and isinstance(den, list) and num and den:
                    label = f"{'+'.join(num)}/{'+'.join(den)}"
            if isinstance(vals, dict):
                print(
                    f"  {label}: data={vals.get('Data', 'n/a')} "
                    f"res={vals.get('Resonance_shared', 'n/a')} "
                    f"py={vals.get('PYTHIA8_external', 'n/a')} "
                    f"ep={vals.get('EPOS_external', 'n/a')} "
                    f"uq={vals.get('UrQMD_external', 'n/a')}"
                )
    print("Seagull slopes d<pT>/dxF:")
    seag_values = seagull_summary.get("values", {}) if isinstance(seagull_summary, dict) else {}
    if isinstance(seag_values, dict) and seag_values:
        for key, sd in species_data_by_key.items():
            row = seag_values.get(key, {}) if isinstance(seag_values, dict) else {}
            models = row.get("models", {}) if isinstance(row, dict) else {}

            def _sl(model_key: str) -> object:
                if not isinstance(models, dict):
                    return "n/a"
                m = models.get(model_key, {})
                if not isinstance(m, dict):
                    return "n/a"
                return m.get("linear_slope", "n/a")

            print(
                f"  {sd.name}: "
                + f"data={_sl('Data')} "
                + f"res={_sl('Resonance_shared')} "
                + f"py={_sl('PYTHIA8_external')} "
                + f"ep={_sl('EPOS_external')} "
                + f"uq={_sl('UrQMD_external')}"
            )


if __name__ == "__main__":
    main()
