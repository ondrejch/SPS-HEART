#!/usr/bin/env python3
"""Integrated-yield agreement study: 1-scale, 2-scale (proj/targ), full shared fit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import nnls

import helpers.benchmarking.benchmark_multispecies_constrained as bmc
import helpers.benchmarking.benchmark_sps_species as bss
from helpers.benchmarking.sps_core import load_multispecies_details_npz


REPO_ROOT = Path(__file__).resolve().parents[2]


def _map_cache_files(cache_dir: Path) -> Dict[str, Path]:
    manifest = cache_dir / "cache_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing cache manifest: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    out: Dict[str, Path] = {}
    for row in payload.get("components", []):
        if not isinstance(row, dict):
            continue
        key = str(row.get("spec_key", "")).strip()
        raw_file = str(row.get("cache_file", "")).strip()
        if not key or not raw_file:
            continue
        out[key] = cache_dir / Path(raw_file).name
    if not out:
        raise RuntimeError(f"No components in cache manifest: {manifest}")
    return out


def _split_component_key(key: str) -> Tuple[str, str]:
    if key.endswith("::proj"):
        return key[:-6], "proj"
    if key.endswith("::targ"):
        return key[:-6], "targ"
    return key, ""


def _resolve_cache_path_for_component(
    base_key: str,
    cache_by_key: Dict[str, Path],
) -> Path | None:
    direct = cache_by_key.get(base_key)
    if direct is not None:
        return direct
    if "::" not in base_key:
        return None
    label, gen = base_key.split("::", 1)
    gen_name = Path(gen).name
    cands: List[Tuple[str, Path]] = []
    for k, p in cache_by_key.items():
        if not k.startswith(f"{label}::"):
            continue
        try:
            gen_k = k.split("::", 1)[1]
        except Exception:
            continue
        if Path(gen_k).name == gen_name:
            cands.append((k, p))
    if len(cands) == 1:
        return cands[0][1]
    return None


def _load_species_data_from_run(summary: Dict[str, object]) -> Dict[str, bmc.SpeciesData]:
    cfg = dict(summary.get("config", {}))
    out: Dict[str, bmc.SpeciesData] = {}
    species = [x.strip() for x in str(cfg.get("species", "")).split(",") if x.strip()]
    if not species:
        species = list(dict(summary.get("species", {})).keys())
    for sk in species:
        out[sk] = bmc._load_species_points(
            species_key=sk,
            xf_min=float(cfg.get("xf_min", 0.0)),
            xf_max=float(cfg.get("xf_max", 0.4)),
            pt_min=float(cfg.get("pt_min", 0.05)),
            pt_max=float(cfg.get("pt_max", 1.5)),
            seed=int(cfg.get("seed", 1)),
        )
    return out


def _build_proj_targ_predictions(
    *,
    summary: Dict[str, object],
    cache_by_key: Dict[str, Path],
    species_data: Dict[str, bmc.SpeciesData],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    cfg = dict(summary.get("config", {}))
    weights_by_key = dict(summary.get("resonance", {}).get("weights_by_key", {}))
    pred_proj = {k: np.zeros(len(sd.y), dtype=float) for k, sd in species_data.items()}
    pred_targ = {k: np.zeros(len(sd.y), dtype=float) for k, sd in species_data.items()}

    for col_key, raw_w in weights_by_key.items():
        w = float(raw_w)
        if abs(w) <= 0.0:
            continue
        base_key, suffix = _split_component_key(str(col_key))
        cpath = _resolve_cache_path_for_component(base_key, cache_by_key)
        if cpath is None or (not cpath.exists()):
            raise FileNotFoundError(f"Missing cache for component '{base_key}' (from '{col_key}')")
        with np.load(cpath, mmap_mode="r", allow_pickle=False) as z:
            pdg = np.asarray(z["pdg"], dtype=np.int32)
            xf = np.asarray(z["xf"], dtype=float)
            pt = np.asarray(z["pt"], dtype=float)
            if "source_side" in z.files:
                src = np.asarray(z["source_side"], dtype=np.int8)
            else:
                src = bmc._infer_source_side_from_xf(xf)
        keep_feed = bmc._feedover_keep_mask(
            xf,
            src,
            target_feedover_xf_max=float(cfg.get("res_target_feedover_xf_max", -1.0)),
            projectile_feedover_xf_max=float(cfg.get("res_projectile_feedover_xf_max", -1.0)),
        )
        if not np.any(keep_feed):
            continue

        if suffix == "proj":
            masks = [("proj", keep_feed & (src > 0))]
        elif suffix == "targ":
            masks = [("targ", keep_feed & (src < 0))]
        else:
            masks = [("proj", keep_feed & (src > 0)), ("targ", keep_feed & (src < 0))]

        for side, mm in masks:
            if not np.any(mm):
                continue
            pdg_m = pdg[mm]
            xf_m = xf[mm]
            pt_m = pt[mm]
            for sk, sd in species_data.items():
                use = (pdg_m == int(sd.pdg))
                if not np.any(use):
                    continue
                counts = bss._counts_on_grid(
                    xf_m[use],
                    pt_m[use],
                    sd.xedges,
                    sd.pedges,
                    sd.ix_data,
                    sd.ip_data,
                )
                if side == "proj":
                    pred_proj[sk] += w * counts
                else:
                    pred_targ[sk] += w * counts
    return pred_proj, pred_targ


def _stack_species(
    species_data: Dict[str, bmc.SpeciesData],
    pred_by_species: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys: List[np.ndarray] = []
    ss: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    for sk in species_data.keys():
        sd = species_data[sk]
        ys.append(np.asarray(sd.y, dtype=float))
        ss.append(np.asarray(sd.sigma, dtype=float))
        ps.append(np.asarray(pred_by_species[sk], dtype=float))
    return np.concatenate(ps), np.concatenate(ys), np.concatenate(ss)


def _fit_one_scale(pred: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> Tuple[float, np.ndarray]:
    alpha = float(bss._fit_single_scale(pred, y, sigma, row_scale=None))
    if alpha < 0.0:
        alpha = 0.0
    return alpha, alpha * pred


def _fit_two_scales(
    pred_proj: np.ndarray,
    pred_targ: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    A = np.column_stack([np.asarray(pred_proj, dtype=float), np.asarray(pred_targ, dtype=float)])
    s = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    w, _ = nnls(A / s[:, None], np.asarray(y, dtype=float) / s)
    pred = A @ w
    return np.asarray(w, dtype=float), np.asarray(pred, dtype=float)


def _integrated_yields_for_model(
    *,
    species_data: Dict[str, bmc.SpeciesData],
    pred_by_species: Dict[str, np.ndarray],
    xf_min: float,
    xf_max: float,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for sk, sd in species_data.items():
        areas = bmc._point_bin_window_areas(sd, xf_min=xf_min, xf_max=xf_max)
        idx = np.arange(len(sd.y), dtype=np.int64)
        out[sk] = float(bmc._integrated_yield(np.asarray(pred_by_species[sk], dtype=float), areas, idx))
    return out


def _plot_rel_diff(
    out_png: Path,
    rows: List[Dict[str, object]],
    species_order: List[str],
) -> None:
    models = ["one_scale", "two_scale_proj_targ", "full_shared"]
    x = np.arange(len(species_order), dtype=float)
    w = 0.26
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    colors = {
        "one_scale": "#4e79a7",
        "two_scale_proj_targ": "#f28e2b",
        "full_shared": "#59a14f",
    }
    labels = {
        "one_scale": "1-scale",
        "two_scale_proj_targ": "2-scale proj/targ",
        "full_shared": "full shared",
    }
    for i, m in enumerate(models):
        vals = []
        for sp in species_order:
            rr = next(r for r in rows if (r["model"] == m and r["species_key"] == sp))
            vals.append(float(rr["rel_diff"]))
        ax.bar(x + (i - 1) * w, vals, width=w, label=labels[m], color=colors[m], alpha=0.9)
    ax.axhline(0.0, color="black", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(species_order)
    ax.set_ylabel(r"$(Y_{fit}-Y_{target})/Y_{target}$")
    ax.set_title("Integrated-yield agreement by species")
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.95))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--cache-dir", default="dev/00runs/res_component_cache_shared_2026-03-31")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = (REPO_ROOT / cache_dir).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "absyield_closure_study")
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_json = run_dir / "multispecies_constrained_summary.json"
    details_npz = run_dir / "multispecies_constrained_details.npz"
    summary = json.loads(summary_json.read_text(encoding="utf-8"))

    with load_multispecies_details_npz(details_npz, summary_payload=summary) as z:
        species_data = _load_species_data_from_run(summary)
        species_order = list(species_data.keys())
        species_meta = dict(summary.get("species", {}))
        for sk, sd in species_data.items():
            tag = str(species_meta.get(sk, {}).get("tag", sd.tag))
            yz = np.asarray(z[f"{tag}_y"], dtype=float)
            if len(yz) != len(sd.y) or float(np.max(np.abs(yz - sd.y))) > 1e-9:
                raise RuntimeError(f"Species data mismatch for {sk}; run and loader ordering differ.")

        # Full shared predictions from saved details.
        pred_full_by_species: Dict[str, np.ndarray] = {}
        for sk, sd in species_data.items():
            tag = str(species_meta.get(sk, {}).get("tag", sd.tag))
            pred_full_by_species[sk] = np.asarray(z[f"{tag}_pred_res"], dtype=float)

    pred_full_stack, y_stack, sigma_stack = _stack_species(species_data, pred_full_by_species)

    # One-scale normalization on top of full-shape prediction.
    alpha1, pred_one_stack = _fit_one_scale(pred_full_stack, y_stack, sigma_stack)
    pred_one_by_species: Dict[str, np.ndarray] = {}
    ptr = 0
    for sk, sd in species_data.items():
        n = len(sd.y)
        pred_one_by_species[sk] = pred_one_stack[ptr : ptr + n]
        ptr += n

    # Build source-resolved base predictions and fit two global source scales.
    cache_by_key = _map_cache_files(cache_dir)
    pred_proj_by_species, pred_targ_by_species = _build_proj_targ_predictions(
        summary=summary,
        cache_by_key=cache_by_key,
        species_data=species_data,
    )
    proj_stack, _, _ = _stack_species(species_data, pred_proj_by_species)
    targ_stack, _, _ = _stack_species(species_data, pred_targ_by_species)
    w2, pred_two_stack = _fit_two_scales(proj_stack, targ_stack, y_stack, sigma_stack)
    pred_two_by_species: Dict[str, np.ndarray] = {}
    ptr = 0
    for sk, sd in species_data.items():
        n = len(sd.y)
        pred_two_by_species[sk] = pred_two_stack[ptr : ptr + n]
        ptr += n

    n_full = int(summary.get("resonance", {}).get("global_metrics", {}).get("n_params", 1))
    metrics = {
        "one_scale": bss._metrics_bundle(pred_one_stack, y_stack, sigma_stack, n_params=1, prefix="full"),
        "two_scale_proj_targ": bss._metrics_bundle(pred_two_stack, y_stack, sigma_stack, n_params=2, prefix="full"),
        "full_shared": bss._metrics_bundle(pred_full_stack, y_stack, sigma_stack, n_params=n_full, prefix="full"),
    }

    ay = dict(summary.get("resonance", {}).get("priors", {}).get("absolute_yield", {}))
    if not bool(ay.get("enabled", False)):
        raise RuntimeError("Run does not include absolute-yield prior; no integrated-yield targets to compare.")
    xf_min = float(ay.get("xf_min", 0.0))
    xf_max = float(ay.get("xf_max", 0.4))
    fit_rows = list(ay.get("fit_rows", []))
    target_by_species = {str(r["species_key"]): float(r["target"]) for r in fit_rows}
    sigma_by_species = {str(r["species_key"]): float(r["sigma"]) for r in fit_rows}

    yint = {
        "one_scale": _integrated_yields_for_model(
            species_data=species_data,
            pred_by_species=pred_one_by_species,
            xf_min=xf_min,
            xf_max=xf_max,
        ),
        "two_scale_proj_targ": _integrated_yields_for_model(
            species_data=species_data,
            pred_by_species=pred_two_by_species,
            xf_min=xf_min,
            xf_max=xf_max,
        ),
        "full_shared": _integrated_yields_for_model(
            species_data=species_data,
            pred_by_species=pred_full_by_species,
            xf_min=xf_min,
            xf_max=xf_max,
        ),
    }

    # Metric table
    metric_rows: List[Dict[str, object]] = []
    for model in ("one_scale", "two_scale_proj_targ", "full_shared"):
        mm = metrics[model]
        chi2_abs = 0.0
        for sk in target_by_species:
            sig = max(sigma_by_species[sk], 1e-12)
            pull = (float(yint[model][sk]) - float(target_by_species[sk])) / sig
            chi2_abs += pull * pull
        metric_rows.append(
            {
                "model": model,
                "n_params_effective": 1 if model == "one_scale" else (2 if model == "two_scale_proj_targ" else n_full),
                "full_log_pull_rms": float(mm.get("full_log_pull_rms", np.nan)),
                "full_weighted_rel_rmse": float(mm.get("full_weighted_rel_rmse", np.nan)),
                "full_smape_percent": float(mm.get("full_smape_percent", np.nan)),
                "abs_yield_chi2": float(chi2_abs),
                "abs_yield_chi2_per_row": float(chi2_abs / max(len(target_by_species), 1)),
                "one_scale_alpha": float(alpha1) if model == "one_scale" else "",
                "two_scale_proj_alpha": float(w2[0]) if model == "two_scale_proj_targ" else "",
                "two_scale_targ_alpha": float(w2[1]) if model == "two_scale_proj_targ" else "",
            }
        )

    metric_csv = out_dir / "absyield_closure_metrics.csv"
    with metric_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(metric_rows[0].keys()))
        w.writeheader()
        w.writerows(metric_rows)

    # Per-species integrated yields.
    yrows: List[Dict[str, object]] = []
    for model in ("one_scale", "two_scale_proj_targ", "full_shared"):
        for sk in species_order:
            if sk not in target_by_species:
                continue
            target = float(target_by_species[sk])
            fit = float(yint[model][sk])
            sigma = float(sigma_by_species[sk])
            yrows.append(
                {
                    "model": model,
                    "species_key": sk,
                    "target": target,
                    "fit": fit,
                    "fit_minus_target": fit - target,
                    "rel_diff": (fit - target) / max(abs(target), 1e-30),
                    "pull": (fit - target) / max(sigma, 1e-12),
                    "sigma": sigma,
                }
            )
    y_csv = out_dir / "absyield_closure_integrated_yields.csv"
    with y_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(yrows[0].keys()))
        w.writeheader()
        w.writerows(yrows)

    fig_png = out_dir / "absyield_closure_rel_diff.png"
    _plot_rel_diff(fig_png, yrows, [sk for sk in species_order if sk in target_by_species])

    label = str(args.label).strip() or run_dir.name
    md = out_dir / "ABSYIELD_CLOSURE_SUMMARY.md"
    lines: List[str] = []
    lines.append(f"# Integrated-Yield Agreement Study: {label}")
    lines.append("")
    lines.append(f"- run_dir: `{run_dir}`")
    lines.append(f"- cache_dir: `{cache_dir}`")
    lines.append(f"- absolute-yield window: `xF in [{xf_min}, {xf_max}]`")
    lines.append(f"- outputs: `{metric_csv.name}`, `{y_csv.name}`, `{fig_png.name}`")
    lines.append(
        "- integrated-yield agreement here means agreement with the SPS integrated-yield targets "
        "in the selected `xF` window, distinct from differential-shape agreement."
    )
    lines.append("")
    lines.append("| model | log-pull RMS | WRRMSE | abs-yield chi2/row |")
    lines.append("|---|---:|---:|---:|")
    for rr in metric_rows:
        lines.append(
            f"| {rr['model']} | {float(rr['full_log_pull_rms']):.6g} | "
            f"{float(rr['full_weighted_rel_rmse']):.6g} | {float(rr['abs_yield_chi2_per_row']):.6g} |"
        )
    lines.append("")
    lines.append("| model | species | target | fit | rel diff | pull |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for model in ("one_scale", "two_scale_proj_targ", "full_shared"):
        for sk in [s for s in species_order if s in target_by_species]:
            rr = next(r for r in yrows if (r["model"] == model and r["species_key"] == sk))
            lines.append(
                f"| {model} | {sk} | {float(rr['target']):.6g} | {float(rr['fit']):.6g} | "
                f"{float(rr['rel_diff']):+.4f} | {float(rr['pull']):+.3f} |"
            )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {metric_csv}")
    print(f"wrote {y_csv}")
    print(f"wrote {fig_png}")
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
