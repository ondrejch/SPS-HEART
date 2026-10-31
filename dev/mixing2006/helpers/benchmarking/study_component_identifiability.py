#!/usr/bin/env python3
"""Study resonance-component identifiability in constrained multispecies fits."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import helpers.benchmarking.benchmark_multispecies_constrained as bmc
import helpers.benchmarking.benchmark_sps_species as bss
from helpers.benchmarking.sps_core import (
    RANK_DEFICIENT_STAMP,
    load_multispecies_details_npz,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "00runs"


@dataclass
class ComponentInfo:
    key: str
    label: str
    base_key: str
    side_suffix: str
    cache_file: Path


def _as_abs_path(path_like: str | Path) -> Path:
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def _load_resolved_args(run_dir: Path) -> Dict[str, object]:
    manifest = run_dir / "constrained_refit_manifest.json"
    if not manifest.exists():
        return {}
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    out = payload.get("resolved_benchmark_args", {})
    return dict(out) if isinstance(out, dict) else {}


def _cache_component_map(cache_dir: Path) -> Dict[str, Dict[str, object]]:
    manifest = cache_dir / "cache_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing cache manifest: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    comps = payload.get("components", [])
    out: Dict[str, Dict[str, object]] = {}
    for item in comps:
        if not isinstance(item, dict):
            continue
        spec_key = str(item.get("spec_key", "")).strip()
        if not spec_key:
            continue
        out[spec_key] = item
    if not out:
        raise RuntimeError(f"No component entries in {manifest}")
    return out


def _canonical_spec_key(spec_key: str) -> str:
    txt = str(spec_key)
    if "::" not in txt:
        return txt
    head, tail = txt.split("::", 1)
    if "/tower_gen/" in tail:
        tail = tail.split("/tower_gen/", 1)[1]
        return f"{head}::tower_gen/{tail}"
    return txt


def _resolve_component_infos(
    component_keys: List[str],
    component_labels: List[str],
    cache_map: Dict[str, Dict[str, object]],
    cache_dir: Path,
) -> List[ComponentInfo]:
    canon_map: Dict[str, str] = {}
    for k in cache_map.keys():
        ck = _canonical_spec_key(k)
        if ck not in canon_map:
            canon_map[ck] = k

    infos: List[ComponentInfo] = []
    for key, label in zip(component_keys, component_labels):
        side = ""
        base = key
        if key not in cache_map:
            parts = key.rsplit("::", 1)
            if len(parts) == 2 and parts[1] in ("proj", "targ") and parts[0] in cache_map:
                base = parts[0]
                side = parts[1]
            else:
                if len(parts) == 2 and parts[1] in ("proj", "targ"):
                    base = parts[0]
                    side = parts[1]
                cbase = _canonical_spec_key(base)
                if cbase in canon_map:
                    base = canon_map[cbase]
                else:
                    raise KeyError(f"Component key not found in cache manifest: {key}")
        else:
            base = key
        cache_file = Path(str(cache_map[base].get("cache_file", "")))
        if not cache_file.is_absolute():
            cache_file = (cache_dir / cache_file).resolve()
        infos.append(
            ComponentInfo(
                key=str(key),
                label=str(label),
                base_key=str(base),
                side_suffix=str(side),
                cache_file=cache_file,
            )
        )
    return infos


def _objective_value(
    *,
    A: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    w: np.ndarray,
    A_prior: np.ndarray | None,
    b_prior: np.ndarray | None,
) -> float:
    resid = (A @ w - y) / np.maximum(sigma, 1e-12)
    val = 0.5 * float(np.dot(resid, resid))
    if A_prior is not None and b_prior is not None and A_prior.size > 0:
        rp = (A_prior @ w) - b_prior
        val += 0.5 * float(np.dot(rp, rp))
    return val


def _connected_components_from_corr(abs_corr: np.ndarray, threshold: float) -> List[List[int]]:
    n = int(abs_corr.shape[0])
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if float(abs_corr[i, j]) >= float(threshold):
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)
    return [sorted(v) for v in groups.values()]


def _parse_hyperon_set(hyperon_pdgs: str) -> set[int]:
    out: set[int] = set()
    for tok in str(hyperon_pdgs).split(","):
        t = tok.strip()
        if not t:
            continue
        out.add(abs(int(t)))
    return out


def _build_abs_targets_from_json(
    args_map: Dict[str, object],
    species_data_by_key: Dict[str, bmc.SpeciesData],
) -> Tuple[Dict[str, float], Dict[str, float], Tuple[float, float] | None]:
    abs_targets_by_key: Dict[str, float] = {}
    abs_sigmas_by_key: Dict[str, float] = {}
    abs_window_override: Tuple[float, float] | None = None
    p_abs = str(args_map.get("prior_abs_yield_json", "")).strip()
    if not p_abs:
        return abs_targets_by_key, abs_sigmas_by_key, abs_window_override
    path = _as_abs_path(p_abs)
    payload = bmc._load_abs_yield_targets_json(path, species_data_by_key=species_data_by_key)
    abs_targets_by_key = {
        str(k): float(v) for k, v in dict(payload.get("targets_by_key", {})).items()
    }
    abs_sigmas_by_key = {
        str(k): float(v) for k, v in dict(payload.get("sigmas_by_key", {})).items()
    }
    xf_min_json = payload.get("xf_min")
    xf_max_json = payload.get("xf_max")
    if isinstance(xf_min_json, (int, float)) and isinstance(xf_max_json, (int, float)):
        abs_window_override = (float(xf_min_json), float(xf_max_json))
    return abs_targets_by_key, abs_sigmas_by_key, abs_window_override


def _stack_species_matrix(
    *,
    species_order: List[str],
    species_data_by_key: Dict[str, bmc.SpeciesData],
    component_infos: List[ComponentInfo],
    target_feedover_xf_max: float,
    projectile_feedover_xf_max: float,
    n_events_per_res: int,
    hyperon_abs_set: set[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    species_cols: Dict[str, List[np.ndarray]] = {k: [] for k in species_order}
    pi_minus_per_component: List[float] = []
    kplus_per_component: List[float] = []
    kminus_per_component: List[float] = []
    hyperon_net_per_component: List[float] = []

    for info in component_infos:
        if not info.cache_file.exists():
            raise FileNotFoundError(f"Missing cached component file: {info.cache_file}")
        with np.load(info.cache_file, mmap_mode="r", allow_pickle=False) as z:
            pdg = np.asarray(z["pdg"], dtype=np.int32)
            xf = np.asarray(z["xf"], dtype=float)
            pt = np.asarray(z["pt"], dtype=float)
            if "source_side" in z:
                source_side = np.asarray(z["source_side"], dtype=np.int8)
            else:
                source_side = bmc._infer_source_side_from_xf(xf)

        keep_feed = bmc._feedover_keep_mask(
            xf,
            source_side,
            target_feedover_xf_max=float(target_feedover_xf_max),
            projectile_feedover_xf_max=float(projectile_feedover_xf_max),
        )
        if info.side_suffix == "proj":
            keep = keep_feed & (source_side > 0)
        elif info.side_suffix == "targ":
            keep = keep_feed & (source_side < 0)
        else:
            keep = keep_feed

        if not np.any(keep):
            pdg_sub = np.zeros(0, dtype=np.int32)
            xf_sub = np.zeros(0, dtype=float)
            pt_sub = np.zeros(0, dtype=float)
        else:
            pdg_sub = pdg[keep]
            xf_sub = xf[keep]
            pt_sub = pt[keep]

        nev = max(int(n_events_per_res), 1)
        pi_minus_per_component.append(float(np.count_nonzero(pdg_sub == -211)) / float(nev))
        kplus_per_component.append(float(np.count_nonzero(pdg_sub == 321)) / float(nev))
        kminus_per_component.append(float(np.count_nonzero(pdg_sub == -321)) / float(nev))
        if hyperon_abs_set and pdg_sub.size > 0:
            abs_pdg = np.abs(pdg_sub)
            m_h = np.isin(abs_pdg, list(hyperon_abs_set))
            if np.any(m_h):
                net = float(np.sum(np.sign(pdg_sub[m_h])))
                hyperon_net_per_component.append(net / float(nev))
            else:
                hyperon_net_per_component.append(0.0)
        else:
            hyperon_net_per_component.append(0.0)

        for skey in species_order:
            sd = species_data_by_key[skey]
            m_sp = (pdg_sub == int(sd.pdg))
            if np.any(m_sp):
                counts = bss._counts_on_grid(
                    xf_sub[m_sp],
                    pt_sub[m_sp],
                    sd.xedges,
                    sd.pedges,
                    sd.ix_data,
                    sd.ip_data,
                )
            else:
                counts = np.zeros(len(sd.y), dtype=float)
            species_cols[skey].append(np.asarray(counts, dtype=float))

    A_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []
    sigma_rows: List[np.ndarray] = []
    for skey in species_order:
        sd = species_data_by_key[skey]
        cols = species_cols[skey]
        A = np.column_stack(cols) if cols else np.zeros((len(sd.y), 0), dtype=float)
        A_rows.append(A)
        y_rows.append(np.asarray(sd.y, dtype=float))
        sigma_rows.append(np.asarray(sd.sigma, dtype=float))
    A_stack = np.vstack(A_rows)
    y_stack = np.concatenate(y_rows)
    sigma_stack = np.concatenate(sigma_rows)
    return (
        A_stack,
        y_stack,
        sigma_stack,
        np.asarray(pi_minus_per_component, dtype=float),
        np.asarray(kplus_per_component, dtype=float),
        np.asarray(kminus_per_component, dtype=float),
        np.asarray(hyperon_net_per_component, dtype=float),
    )


def _build_linear_priors(
    *,
    args_map: Dict[str, object],
    species_data_by_key: Dict[str, bmc.SpeciesData],
    component_labels: List[str],
    pi_minus_per_component: np.ndarray,
    kplus_per_component: np.ndarray,
    kminus_per_component: np.ndarray,
    hyperon_net_per_component: np.ndarray,
    species_order: List[str],
) -> Tuple[np.ndarray | None, np.ndarray | None, Dict[str, object]]:
    prior_rows: List[np.ndarray] = []
    prior_targets: List[float] = []
    report: Dict[str, object] = {}

    # The driver sets these module globals inside its own main(); direct
    # consumers of the prior builders (this study) must initialize them
    # from the resolved run arguments before any invariant-convention
    # weight touches sqrt(s).
    if bmc.SQRT_S_GEV is None:
        bmc.CONVENTION = str(args_map.get("convention", bmc.CONVENTION))
        bmc.SQRT_S_GEV = bss._sqrt_s_from_fixed_target_beam_momentum(
            float(args_map.get("beam_momentum", 158.0))
        )

    lam_table9 = max(float(args_map.get("prior_table9_lambda", 0.0)), 0.0)
    lam_table9_total = max(float(args_map.get("prior_table9_total_lambda", 0.0)), 0.0)
    table9_json_raw = str(args_map.get("prior_table9_json", bmc.DEFAULT_TABLE9_JSON))
    table9_json = _as_abs_path(table9_json_raw)
    report["table9_enabled"] = False
    if (lam_table9 > 0.0 or lam_table9_total > 0.0) and table9_json.exists():
        t9 = bmc._build_table9_composition_prior(
            component_labels=component_labels,
            pi_minus_per_component=pi_minus_per_component,
            table9_json=table9_json,
        )
        if bool(t9.get("ok", False)):
            if lam_table9 > 0.0:
                A_t9 = np.asarray(t9["A"], dtype=float)
                b_t9 = np.asarray(t9["b"], dtype=float)
                sc = float(np.sqrt(lam_table9))
                for i in range(A_t9.shape[0]):
                    prior_rows.append(sc * A_t9[i])
                    prior_targets.append(sc * float(b_t9[i]))
            if lam_table9_total > 0.0:
                total_target = float(args_map.get("prior_table9_total_target", 0.0))
                if total_target <= 0.0:
                    total_target = float(t9.get("table_total", 0.0))
                total_vec = np.asarray(t9.get("total_vec", np.zeros(len(component_labels), dtype=float)), dtype=float)
                if np.any(np.abs(total_vec) > 0.0):
                    sc = float(np.sqrt(lam_table9_total))
                    prior_rows.append(sc * total_vec)
                    prior_targets.append(sc * total_target)
            report["table9_enabled"] = True
            report["table9_lambda"] = lam_table9
            report["table9_total_lambda"] = lam_table9_total
            report["table9_rows"] = int(np.asarray(t9["A"]).shape[0])
    lam_strange = max(float(args_map.get("prior_strange_lambda", 0.0)), 0.0)
    lam_hyperon = max(float(args_map.get("prior_hyperon_lambda", 0.0)), 0.0)
    if lam_strange > 0.0 or lam_hyperon > 0.0:
        sp = bmc._build_strange_hyperon_priors(
            kplus_per_component=kplus_per_component,
            kminus_per_component=kminus_per_component,
            hyperon_net_per_component=hyperon_net_per_component,
            kminus_over_kplus_target=float(args_map.get("prior_kminus_over_kplus_target", bmc.DEFAULT_KMINUS_OVER_KPLUS)),
        )
        if lam_strange > 0.0:
            prior_rows.append(float(np.sqrt(lam_strange)) * np.asarray(sp["kaon_ratio_row"], dtype=float))
            prior_targets.append(float(np.sqrt(lam_strange)) * float(sp["kaon_ratio_target"]))
        if lam_hyperon > 0.0:
            prior_rows.append(float(np.sqrt(lam_hyperon)) * np.asarray(sp["hyperon_balance_row"], dtype=float))
            prior_targets.append(float(np.sqrt(lam_hyperon)) * float(sp["hyperon_balance_target"]))
        report["strange_enabled"] = True
        report["strange_lambda"] = lam_strange
        report["hyperon_lambda"] = lam_hyperon
    else:
        report["strange_enabled"] = False

    lam_abs = max(float(args_map.get("prior_abs_yield_lambda", 0.0)), 0.0)
    if lam_abs > 0.0:
        xf_window = str(args_map.get("prior_abs_yield_window", "0.0:0.4"))
        xf_lo, xf_hi = bmc._parse_single_xf_window(xf_window)
        species_selected = bmc._resolve_abs_yield_species_csv(
            str(args_map.get("prior_abs_yield_species", "")),
            species_data_by_key,
        )
        abs_targets, abs_sigmas, abs_window_override = _build_abs_targets_from_json(
            args_map,
            species_data_by_key=species_data_by_key,
        )
        if abs_window_override is not None:
            xf_lo, xf_hi = abs_window_override
        ay = bmc._build_absolute_yield_prior(
            species_data_by_key=species_data_by_key,
            selected_species_keys=species_selected,
            xf_min=xf_lo,
            xf_max=xf_hi,
            target_overrides=abs_targets,
            sigma_overrides=abs_sigmas,
            min_sigma_rel=float(args_map.get("prior_abs_yield_min_sigma_rel", 0.02)),
        )
        if bool(ay.get("ok", False)):
            A_ay = np.asarray(ay["A"], dtype=float)
            b_ay = np.asarray(ay["b"], dtype=float)
            sc = float(np.sqrt(lam_abs))
            for i in range(A_ay.shape[0]):
                prior_rows.append(sc * A_ay[i])
                prior_targets.append(sc * float(b_ay[i]))
            report["abs_enabled"] = True
            report["abs_lambda"] = lam_abs
            report["abs_rows"] = int(A_ay.shape[0])
        else:
            report["abs_enabled"] = False
    else:
        report["abs_enabled"] = False

    lam_nstar_smooth = max(float(args_map.get("prior_nstar_smooth_lambda", 0.0)), 0.0)
    if lam_nstar_smooth > 0.0:
        ns = bmc._build_nstar_ladder_smooth_prior(component_labels=component_labels)
        if bool(ns.get("ok", False)):
            A_ns = np.asarray(ns["A"], dtype=float)
            b_ns = np.asarray(ns["b"], dtype=float)
            sc = float(np.sqrt(lam_nstar_smooth))
            for i in range(A_ns.shape[0]):
                prior_rows.append(sc * A_ns[i])
                prior_targets.append(sc * float(b_ns[i]))
            report["nstar_smooth_enabled"] = True
            report["nstar_smooth_lambda"] = lam_nstar_smooth
            report["nstar_smooth_rows"] = int(A_ns.shape[0])
        else:
            report["nstar_smooth_enabled"] = False
    else:
        report["nstar_smooth_enabled"] = False

    A_prior = np.vstack(prior_rows) if prior_rows else None
    b_prior = np.asarray(prior_targets, dtype=float) if prior_rows else None
    report["prior_total_rows"] = int(A_prior.shape[0]) if A_prior is not None else 0
    return A_prior, b_prior, report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        default=str(RUNS_ROOT / "constrained_refit_pdg_absyield_scan_l0300_s0050_2026-03-31r2"),
        help="Run directory containing multispecies_constrained_summary/details and constrained_refit_manifest.json.",
    )
    ap.add_argument("--cache-dir", default="", help="Override component cache dir. Defaults to run manifest.")
    ap.add_argument("--out-dir", default="", help="Output directory for identifiability artifacts.")
    ap.add_argument("--active-threshold", type=float, default=1e-12)
    ap.add_argument("--corr-threshold", type=float, default=0.95)
    ap.add_argument("--top-pairs", type=int, default=40)
    ap.add_argument("--run-reduced-basis", action="store_true")
    ap.add_argument("--reduce-corr-threshold", type=float, default=0.995)
    ap.add_argument(
        "--strict-rank",
        action="store_true",
        help=(
            "Refuse rank-deficient refits instead of accepting them. The "
            "default accepts them because this study deliberately fits "
            "reduced component sets where rank deficiency is the object of "
            f"study; every accepted refit carries identifiability="
            f"{RANK_DEFICIENT_STAMP!r} in its solver payload, which the "
            "CSV/JSON/Markdown outputs below record."
        ),
    )
    args = ap.parse_args()

    run_dir = _as_abs_path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run dir: {run_dir}")
    summary_json = run_dir / "multispecies_constrained_summary.json"
    details_npz = run_dir / "multispecies_constrained_details.npz"
    if not summary_json.exists() or not details_npz.exists():
        raise FileNotFoundError(f"Missing run summary/details in {run_dir}")

    out_dir = _as_abs_path(args.out_dir) if str(args.out_dir).strip() else (run_dir / "component_identifiability")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    run_args = _load_resolved_args(run_dir)
    if not run_args:
        raise RuntimeError(f"Could not load resolved benchmark args from {run_dir}/constrained_refit_manifest.json")

    with load_multispecies_details_npz(details_npz, summary_payload=summary) as zarr:
        species_order = [str(x) for x in zarr["species_keys"]]
        component_keys = [str(x) for x in zarr["component_keys"]]
        component_labels = [str(x) for x in zarr["component_labels"]]
        w = np.asarray(zarr["component_weights"], dtype=float)
        pred_saved = np.asarray(zarr["resonance_global_full_pred"], dtype=float)

    cache_dir = _as_abs_path(args.cache_dir) if str(args.cache_dir).strip() else _as_abs_path(
        str(run_args.get("res_cache_dir", run_dir / "resonance_component_cache"))
    )
    cache_map = _cache_component_map(cache_dir)
    comp_infos = _resolve_component_infos(component_keys, component_labels, cache_map, cache_dir)

    xf_min = float(run_args.get("xf_min", 0.0))
    xf_max = float(run_args.get("xf_max", 0.4))
    pt_min = float(run_args.get("pt_min", 0.05))
    pt_max = float(run_args.get("pt_max", 1.5))
    seed = int(run_args.get("seed", 20260331))
    species_data_by_key: Dict[str, bmc.SpeciesData] = {}
    for skey in species_order:
        species_data_by_key[skey] = bmc._load_species_points(
            species_key=skey,
            xf_min=xf_min,
            xf_max=xf_max,
            pt_min=pt_min,
            pt_max=pt_max,
            seed=seed,
        )

    target_feedover_xf_max = float(run_args.get("res_target_feedover_xf_max", -1.0))
    projectile_feedover_xf_max = float(run_args.get("res_projectile_feedover_xf_max", -1.0))
    n_events_per_res = int(run_args.get("res_events_per_res", 30000))
    hyperon_abs_set = _parse_hyperon_set(str(run_args.get("hyperon_pdgs", "3122,3212,3222,3112,3312,3322,3334")))

    (
        A,
        y,
        sigma,
        pi_minus_per_component,
        kplus_per_component,
        kminus_per_component,
        hyperon_net_per_component,
    ) = _stack_species_matrix(
        species_order=species_order,
        species_data_by_key=species_data_by_key,
        component_infos=comp_infos,
        target_feedover_xf_max=target_feedover_xf_max,
        projectile_feedover_xf_max=projectile_feedover_xf_max,
        n_events_per_res=n_events_per_res,
        hyperon_abs_set=hyperon_abs_set,
    )

    pred_reco = A @ w
    pred_max_abs_diff = float(np.max(np.abs(pred_reco - pred_saved)))
    pred_rms_diff = float(np.sqrt(np.mean((pred_reco - pred_saved) ** 2)))

    A_prior, b_prior, prior_report = _build_linear_priors(
        args_map=run_args,
        species_data_by_key=species_data_by_key,
        component_labels=component_labels,
        pi_minus_per_component=pi_minus_per_component,
        kplus_per_component=kplus_per_component,
        kminus_per_component=kminus_per_component,
        hyperon_net_per_component=hyperon_net_per_component,
        species_order=species_order,
    )

    sigma_safe = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    Aw = np.asarray(A, dtype=float) / sigma_safe[:, None]
    H_data = Aw.T @ Aw
    H_total = H_data.copy()
    if A_prior is not None and b_prior is not None and A_prior.size > 0:
        H_total += np.asarray(A_prior, dtype=float).T @ np.asarray(A_prior, dtype=float)

    pinv_data = np.linalg.pinv(H_data, rcond=1e-12)
    pinv_total = np.linalg.pinv(H_total, rcond=1e-12)
    std_data = np.sqrt(np.maximum(np.diag(pinv_data), 0.0))
    std_total = np.sqrt(np.maximum(np.diag(pinv_total), 0.0))
    z_total = np.divide(w, std_total, out=np.full_like(w, np.nan), where=std_total > 0.0)

    norms = np.linalg.norm(Aw, axis=0)
    norm_safe = np.where(norms > 0.0, norms, 1.0)
    N = Aw / norm_safe[None, :]
    C = N.T @ N
    abs_C = np.abs(C)
    np.fill_diagonal(abs_C, 0.0)
    nearest_idx = np.argmax(abs_C, axis=1)
    nearest_corr = abs_C[np.arange(abs_C.shape[0]), nearest_idx]

    C_reg = C + 1e-6 * np.eye(C.shape[0], dtype=float)
    C_inv = np.linalg.pinv(C_reg, rcond=1e-12)
    vif = np.maximum(np.diag(C_inv), 1.0)
    r2 = 1.0 - (1.0 / vif)
    uniqueness = np.clip(1.0 - r2, 0.0, 1.0)

    pred_energy = (w ** 2) * (norms ** 2)
    pred_energy_sum = float(np.sum(pred_energy))
    pred_energy_share = np.divide(
        pred_energy,
        pred_energy_sum if pred_energy_sum > 0.0 else 1.0,
        out=np.zeros_like(pred_energy),
    )

    svals = np.linalg.svd(Aw, compute_uv=False)
    smax = float(np.max(svals)) if svals.size else 0.0
    smin_pos = float(np.min(svals[svals > 0.0])) if np.any(svals > 0.0) else np.nan
    cond_num = float(smax / smin_pos) if (smax > 0.0 and np.isfinite(smin_pos) and smin_pos > 0.0) else np.inf
    rank_1e3 = int(np.sum(svals >= (1e-3 * smax))) if smax > 0 else 0
    rank_1e6 = int(np.sum(svals >= (1e-6 * smax))) if smax > 0 else 0
    rank_1e9 = int(np.sum(svals >= (1e-9 * smax))) if smax > 0 else 0
    p = svals / np.sum(svals) if np.sum(svals) > 0 else np.zeros_like(svals)
    eff_rank = float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0])))) if np.any(p > 0) else 0.0

    active_mask = w > float(args.active_threshold)
    active_idx = np.flatnonzero(active_mask)
    f_full = _objective_value(A=A, y=y, sigma=sigma_safe, w=w, A_prior=A_prior, b_prior=b_prior)
    base_metrics = bss._metrics_bundle(pred_reco, y, sigma_safe, n_params=int(np.count_nonzero(active_mask)), prefix="full")
    drop_rows: List[Dict[str, object]] = []
    fit_objective = str(run_args.get("fit_objective", "gaussian_point_mle"))
    fit_opt_maxiter = int(run_args.get("fit_opt_maxiter", 3000))
    fit_opt_ftol = float(run_args.get("fit_opt_ftol", 1e-10))
    for j in active_idx.tolist():
        keep = np.ones(len(w), dtype=bool)
        keep[j] = False
        A_r = A[:, keep]
        if A_prior is not None and b_prior is not None and A_prior.size > 0:
            A_prior_r = np.asarray(A_prior, dtype=float)[:, keep]
            b_prior_r = np.asarray(b_prior, dtype=float)
        else:
            A_prior_r = None
            b_prior_r = None
        w_r, solver_info = bmc._fit_global_nonnegative_weights(
            objective=fit_objective,
            A_train_raw=A_r,
            y_train=y,
            sigma_train=sigma_safe,
            A_train_weighted=A_r / sigma_safe[:, None],
            b_train_weighted=y / sigma_safe,
            A_prior=A_prior_r,
            b_prior=b_prior_r,
            table9_fraction_rows=None,
            table9_fraction_targets=None,
            table9_fraction_lambda=0.0,
            table9_fraction_eps=1e-6,
            maxiter=fit_opt_maxiter,
            ftol=fit_opt_ftol,
            allow_rank_deficient=not bool(args.strict_rank),
        )
        pred_r = A_r @ w_r
        f_r = _objective_value(A=A_r, y=y, sigma=sigma_safe, w=w_r, A_prior=A_prior_r, b_prior=b_prior_r)
        m_r = bss._metrics_bundle(pred_r, y, sigma_safe, n_params=int(np.count_nonzero(w_r > args.active_threshold)), prefix="full")
        drop_rows.append(
            {
                "idx": int(j),
                "label": component_labels[j],
                "key": component_keys[j],
                "weight": float(w[j]),
                "delta_objective": float(f_r - f_full),
                "delta_log_pull_rms": float(m_r["full_log_pull_rms"] - base_metrics["full_log_pull_rms"]),
                "delta_wrrmse": float(m_r["full_weighted_rel_rmse"] - base_metrics["full_weighted_rel_rmse"]),
                "solver_success": bool(solver_info.get("success", False)),
                "solver_identifiability": str(solver_info.get("identifiability", "not_recorded")),
                "solver_design_rank": int(solver_info.get("design_rank", -1)),
            }
        )

    reduced_report: Dict[str, object] = {}
    pruning_rows: List[Dict[str, object]] = []
    if bool(args.run_reduced_basis):
        components = _connected_components_from_corr(abs_C, threshold=float(args.reduce_corr_threshold))
        keep_idx: List[int] = []
        drop_idx: List[int] = []
        for comp in components:
            if len(comp) == 1:
                keep_idx.append(comp[0])
                continue
            # Keep the most relevant basis vector in each highly-correlated cluster:
            # prefer non-zero fitted weight, then larger weighted prediction-energy share.
            best = sorted(
                comp,
                key=lambda j: (
                    float(w[j]) > float(args.active_threshold),
                    float(w[j]),
                    float(pred_energy_share[j]),
                    float(norms[j]),
                ),
                reverse=True,
            )[0]
            keep_idx.append(int(best))
            for j in comp:
                if int(j) != int(best):
                    drop_idx.append(int(j))
                    pruning_rows.append(
                        {
                            "cluster_size": int(len(comp)),
                            "keep_idx": int(best),
                            "keep_label": component_labels[int(best)],
                            "drop_idx": int(j),
                            "drop_label": component_labels[int(j)],
                            "abs_corr_to_keep": float(abs_C[int(best), int(j)]),
                            "drop_weight": float(w[int(j)]),
                        }
                    )
        keep_idx = sorted(set(keep_idx))
        drop_idx = sorted(set(drop_idx))

        A_red = A[:, keep_idx]
        if A_prior is not None and b_prior is not None and A_prior.size > 0:
            A_prior_red = np.asarray(A_prior, dtype=float)[:, keep_idx]
            b_prior_red = np.asarray(b_prior, dtype=float)
        else:
            A_prior_red = None
            b_prior_red = None
        w_red, info_red = bmc._fit_global_nonnegative_weights(
            objective=fit_objective,
            A_train_raw=A_red,
            y_train=y,
            sigma_train=sigma_safe,
            A_train_weighted=A_red / sigma_safe[:, None],
            b_train_weighted=y / sigma_safe,
            A_prior=A_prior_red,
            b_prior=b_prior_red,
            table9_fraction_rows=None,
            table9_fraction_targets=None,
            table9_fraction_lambda=0.0,
            table9_fraction_eps=1e-6,
            maxiter=fit_opt_maxiter,
            ftol=fit_opt_ftol,
            allow_rank_deficient=not bool(args.strict_rank),
        )
        pred_red = A_red @ w_red
        f_red = _objective_value(
            A=A_red,
            y=y,
            sigma=sigma_safe,
            w=w_red,
            A_prior=A_prior_red,
            b_prior=b_prior_red,
        )
        m_red = bss._metrics_bundle(
            pred_red,
            y,
            sigma_safe,
            n_params=int(np.count_nonzero(w_red > args.active_threshold)),
            prefix="full",
        )
        w_red_full = np.zeros_like(w)
        for loc, src in enumerate(keep_idx):
            w_red_full[int(src)] = float(w_red[loc])
        reduced_report = {
            "enabled": True,
            "corr_threshold": float(args.reduce_corr_threshold),
            "n_components_full": int(len(w)),
            "n_components_keep": int(len(keep_idx)),
            "n_components_drop": int(len(drop_idx)),
            "n_active_full": int(np.count_nonzero(w > args.active_threshold)),
            "n_active_reduced": int(np.count_nonzero(w_red > args.active_threshold)),
            "objective_full": float(f_full),
            "objective_reduced": float(f_red),
            "delta_objective": float(f_red - f_full),
            "delta_log_pull_rms": float(m_red["full_log_pull_rms"] - base_metrics["full_log_pull_rms"]),
            "delta_wrrmse": float(m_red["full_weighted_rel_rmse"] - base_metrics["full_weighted_rel_rmse"]),
            "solver_success": bool(info_red.get("success", False)),
            "identifiability": str(info_red.get("identifiability", "not_recorded")),
            "design_scope": str(info_red.get("design_scope", "not_recorded")),
            "design_rank": int(info_red.get("design_rank", -1)),
            "design_condition_number": float(info_red.get("design_condition_number", float("nan"))),
            "keep_indices": [int(x) for x in keep_idx],
            "drop_indices": [int(x) for x in drop_idx],
        }
    else:
        reduced_report = {"enabled": False}

    comp_rows: List[Dict[str, object]] = []
    for j, info in enumerate(comp_infos):
        comp_rows.append(
            {
                "idx": int(j),
                "label": info.label,
                "key": info.key,
                "weight": float(w[j]),
                "active": int(active_mask[j]),
                "data_col_norm": float(norms[j]),
                "pred_energy_share": float(pred_energy_share[j]),
                "max_abs_corr": float(nearest_corr[j]),
                "nearest_label": component_labels[int(nearest_idx[j])],
                "nearest_key": component_keys[int(nearest_idx[j])],
                "uniqueness": float(uniqueness[j]),
                "vif": float(vif[j]),
                "std_data": float(std_data[j]),
                "std_total": float(std_total[j]),
                "z_total": float(z_total[j]) if np.isfinite(z_total[j]) else np.nan,
            }
        )

    pair_rows: List[Dict[str, object]] = []
    n = C.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            c = float(C[i, j])
            ac = abs(c)
            if ac < float(args.corr_threshold):
                continue
            pair_rows.append(
                {
                    "i": int(i),
                    "j": int(j),
                    "label_i": component_labels[i],
                    "label_j": component_labels[j],
                    "key_i": component_keys[i],
                    "key_j": component_keys[j],
                    "corr": float(c),
                    "abs_corr": float(ac),
                    "weight_i": float(w[i]),
                    "weight_j": float(w[j]),
                }
            )
    pair_rows.sort(key=lambda r: float(r["abs_corr"]), reverse=True)

    comp_csv = out_dir / "component_identifiability_metrics.csv"
    with comp_csv.open("w", newline="", encoding="utf-8") as fh:
        fields = [
            "idx",
            "label",
            "key",
            "weight",
            "active",
            "data_col_norm",
            "pred_energy_share",
            "max_abs_corr",
            "nearest_label",
            "nearest_key",
            "uniqueness",
            "vif",
            "std_data",
            "std_total",
            "z_total",
        ]
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for row in sorted(comp_rows, key=lambda r: float(r["weight"]), reverse=True):
            wr.writerow({k: row.get(k) for k in fields})

    pair_csv = out_dir / "component_correlation_pairs.csv"
    with pair_csv.open("w", newline="", encoding="utf-8") as fh:
        fields = [
            "i",
            "j",
            "label_i",
            "label_j",
            "key_i",
            "key_j",
            "corr",
            "abs_corr",
            "weight_i",
            "weight_j",
        ]
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for row in pair_rows[: int(args.top_pairs)]:
            wr.writerow({k: row.get(k) for k in fields})

    drop_csv = out_dir / "component_dropone_active.csv"
    with drop_csv.open("w", newline="", encoding="utf-8") as fh:
        fields = [
            "idx",
            "label",
            "key",
            "weight",
            "delta_objective",
            "delta_log_pull_rms",
            "delta_wrrmse",
            "solver_success",
            "solver_identifiability",
            "solver_design_rank",
        ]
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for row in sorted(drop_rows, key=lambda r: float(r["delta_objective"]), reverse=True):
            wr.writerow({k: row.get(k) for k in fields})

    prune_csv = out_dir / "component_reduced_basis_pruning.csv"
    with prune_csv.open("w", newline="", encoding="utf-8") as fh:
        fields = [
            "cluster_size",
            "keep_idx",
            "keep_label",
            "drop_idx",
            "drop_label",
            "abs_corr_to_keep",
            "drop_weight",
        ]
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for row in sorted(pruning_rows, key=lambda r: float(r["abs_corr_to_keep"]), reverse=True):
            wr.writerow({k: row.get(k) for k in fields})

    fig = plt.figure(figsize=(12.5, 5.2), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    x = np.arange(1, len(svals) + 1)
    ax1.plot(x, svals, marker="o", ms=3)
    ax1.set_yscale("log")
    ax1.set_xlabel("Singular-value index")
    ax1.set_ylabel("sigma_k(A/sigma)")
    ax1.set_title("Design-Matrix Spectrum")
    ax1.grid(True, alpha=0.3)

    w_plot = np.maximum(np.asarray(w, dtype=float), 1e-16)
    active_colors = np.where(active_mask, "#1b9e77", "#9e9e9e")
    ax2.scatter(uniqueness, w_plot, s=36, c=active_colors, alpha=0.9)
    ax2.set_yscale("log")
    ax2.set_xlabel("Uniqueness (1 - R2)")
    ax2.set_ylabel("Fitted weight")
    ax2.set_title("Weight vs Uniqueness")
    ax2.grid(True, alpha=0.3)
    fig_png = out_dir / "component_identifiability_overview.png"
    fig.savefig(fig_png, dpi=150)
    plt.close(fig)

    md_lines: List[str] = []
    md_lines.append("# Component Identifiability Study")
    md_lines.append("")
    if bool(reduced_report.get("enabled", False)):
        md_lines.append("## Reduced-Basis Refit")
        md_lines.append("")
        md_lines.append(
            f"- correlation threshold for clustering: `{float(reduced_report.get('corr_threshold', float('nan'))):.6g}`"
        )
        md_lines.append(
            f"- components kept/dropped: `{int(reduced_report.get('n_components_keep', 0))}` / "
            + f"`{int(reduced_report.get('n_components_drop', 0))}`"
        )
        md_lines.append(
            f"- active components full/reduced: `{int(reduced_report.get('n_active_full', 0))}` / "
            + f"`{int(reduced_report.get('n_active_reduced', 0))}`"
        )
        md_lines.append(f"- delta objective (reduced-full): `{float(reduced_report.get('delta_objective', float('nan'))):.6g}`")
        md_lines.append(f"- delta log-pull RMS: `{float(reduced_report.get('delta_log_pull_rms', float('nan'))):.6g}`")
        md_lines.append(f"- delta WRRMSE: `{float(reduced_report.get('delta_wrrmse', float('nan'))):.6g}`")
        md_lines.append(f"- solver success: `{bool(reduced_report.get('solver_success', False))}`")
        md_lines.append(f"- identifiability: `{str(reduced_report.get('identifiability', 'not_recorded'))}`")
        md_lines.append(
            f"- reduced design rank: `{int(reduced_report.get('design_rank', -1))}` of "
            f"`{int(reduced_report.get('n_components_keep', 0))}` kept components "
            f"({str(reduced_report.get('design_scope', 'not_recorded'))})"
        )
        md_lines.append("")

    md_lines.append("## Run Context")
    md_lines.append("")
    md_lines.append(f"- run_dir: `{run_dir}`")
    md_lines.append(f"- cache_dir: `{cache_dir}`")
    md_lines.append(f"- objective: `{fit_objective}`")
    md_lines.append(f"- species_order: `{','.join(species_order)}`")
    md_lines.append(f"- matrix shape: `{A.shape[0]} x {A.shape[1]}`")
    md_lines.append(f"- active components (weight>{args.active_threshold:.1e}): `{int(np.count_nonzero(active_mask))}`")
    md_lines.append(f"- prior rows (linear): `{int(prior_report.get('prior_total_rows', 0))}`")
    md_lines.append("")
    md_lines.append("## Matrix Diagnostics")
    md_lines.append("")
    md_lines.append(f"- max singular value: `{smax:.6g}`")
    md_lines.append(f"- min positive singular value: `{smin_pos:.6g}`")
    md_lines.append(f"- condition number (positive spectrum): `{cond_num:.6g}`")
    md_lines.append(f"- rank @1e-3 smax: `{rank_1e3}`")
    md_lines.append(f"- rank @1e-6 smax: `{rank_1e6}`")
    md_lines.append(f"- rank @1e-9 smax: `{rank_1e9}`")
    md_lines.append(f"- effective rank: `{eff_rank:.6g}`")
    md_lines.append("")
    md_lines.append("## Reconstruction Check")
    md_lines.append("")
    md_lines.append(f"- max |A*w - saved_pred|: `{pred_max_abs_diff:.6g}`")
    md_lines.append(f"- RMS(A*w - saved_pred): `{pred_rms_diff:.6g}`")
    md_lines.append("")
    md_lines.append("## Top Components By Weight")
    md_lines.append("")
    md_lines.append("| label | weight | pred_energy_share | max_abs_corr | nearest | uniqueness | z_total |")
    md_lines.append("|---|---:|---:|---:|---|---:|---:|")
    for row in sorted(comp_rows, key=lambda r: float(r["weight"]), reverse=True)[:12]:
        md_lines.append(
            f"| {row['label']} | {float(row['weight']):.6g} | {float(row['pred_energy_share']):.6g} | "
            + f"{float(row['max_abs_corr']):.6g} | {row['nearest_label']} | {float(row['uniqueness']):.6g} | "
            + f"{float(row['z_total']):.6g} |"
        )
    md_lines.append("")
    md_lines.append("## Most Correlated Component Pairs")
    md_lines.append("")
    md_lines.append("| i | j | label_i | label_j | abs_corr | w_i | w_j |")
    md_lines.append("|---:|---:|---|---|---:|---:|---:|")
    for row in pair_rows[: min(int(args.top_pairs), 15)]:
        md_lines.append(
            f"| {row['i']} | {row['j']} | {row['label_i']} | {row['label_j']} | "
            + f"{float(row['abs_corr']):.6g} | {float(row['weight_i']):.6g} | {float(row['weight_j']):.6g} |"
        )
    md_lines.append("")
    md_lines.append("## Active-Component Drop-One Refit")
    md_lines.append("")
    md_lines.append("| label | weight | delta_objective | delta_log_pull_rms | delta_wrrmse |")
    md_lines.append("|---|---:|---:|---:|---:|")
    for row in sorted(drop_rows, key=lambda r: float(r["delta_objective"]), reverse=True):
        md_lines.append(
            f"| {row['label']} | {float(row['weight']):.6g} | {float(row['delta_objective']):.6g} | "
            + f"{float(row['delta_log_pull_rms']):.6g} | {float(row['delta_wrrmse']):.6g} |"
        )

    md = out_dir / "COMPONENT_IDENTIFIABILITY_SUMMARY.md"
    md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    reduced_json = out_dir / "component_reduced_basis_report.json"
    reduced_json.write_text(json.dumps(reduced_report, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {comp_csv}")
    print(f"wrote {pair_csv}")
    print(f"wrote {drop_csv}")
    print(f"wrote {prune_csv}")
    print(f"wrote {fig_png}")
    print(f"wrote {md}")
    print(f"wrote {reduced_json}")


if __name__ == "__main__":
    main()
