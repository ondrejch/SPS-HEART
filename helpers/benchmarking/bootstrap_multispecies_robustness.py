#!/usr/bin/env python3
"""Bootstrap robustness bands for multispecies constrained-fit runs."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from helpers.benchmarking.sps_core import load_multispecies_details_npz


MODEL_SUFFIX = {
    "Resonance_NNLS_shared": "pred_res",
    "PYTHIA8_external": "pred_pythia",
    "EPOS_external": "pred_epos",
    "UrQMD_external": "pred_urqmd",
}

MODEL_SHORT = {
    "Resonance_NNLS_shared": "Resonance",
    "PYTHIA8_external": "PYTHIA8",
    "EPOS_external": "EPOS",
    "UrQMD_external": "UrQMD",
}


@dataclass
class SpeciesArrays:
    y: np.ndarray
    sigma: np.ndarray
    pred_by_model: Dict[str, np.ndarray]


@dataclass
class RunPayload:
    label: str
    run_dir: Path
    species_arrays: Dict[str, SpeciesArrays]
    n_params_by_model: Dict[str, int]


def _metrics(pred: np.ndarray, y: np.ndarray, sigma: np.ndarray, n_params: int) -> Dict[str, float]:
    s = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    yy = np.asarray(y, dtype=float)
    pp = np.asarray(pred, dtype=float)

    resid = (pp - yy) / s
    chi2 = float(np.sum(resid * resid))
    ndof = max(int(len(yy)) - int(n_params), 1)

    rel = (pp - yy) / np.maximum(yy, 1e-30)
    rel_sigma = np.maximum(s / np.maximum(yy, 1e-30), 1e-12)
    w = 1.0 / (rel_sigma * rel_sigma)
    wrrmse = float(np.sqrt(np.sum(w * rel * rel) / max(np.sum(w), 1e-30)))

    mask = (pp > 0.0) & (yy > 0.0) & (s > 0.0)
    if np.any(mask):
        pp_m = pp[mask]
        yy_m = yy[mask]
        s_m = s[mask]
        sigma_log = np.maximum(s_m / np.maximum(yy_m, 1e-30), 1e-12)
        rlog = (np.log(pp_m) - np.log(yy_m)) / sigma_log
        log_chi2 = float(np.sum(rlog * rlog))
        n_log = int(mask.sum())
        log_pull_rms = float(np.sqrt(log_chi2 / max(n_log, 1)))
    else:
        log_chi2 = 0.0
        n_log = 0
        log_pull_rms = 0.0

    return {
        "chi2": chi2,
        "chi2_dof": chi2 / ndof,
        "wrrmse": wrrmse,
        "log_chi2": log_chi2,
        "log_pull_rms": log_pull_rms,
        "n_points": float(len(yy)),
        "n_log": float(n_log),
    }


def _species_selector(species_csv: str, available: Sequence[str]) -> List[str]:
    raw = [x.strip() for x in str(species_csv).split(",") if x.strip()]
    if not raw:
        return list(available)
    avail_set = set(available)
    out = [x for x in raw if x in avail_set]
    if not out:
        raise ValueError(f"No requested species found in run payload: requested={raw}, available={list(available)}")
    return out


def _load_run(run_dir: Path, label: str, species_csv: str) -> RunPayload:
    summary = run_dir / "multispecies_constrained_summary.json"
    details = run_dir / "multispecies_constrained_details.npz"
    if not summary.exists():
        raise FileNotFoundError(f"Missing summary: {summary}")
    if not details.exists():
        raise FileNotFoundError(f"Missing details: {details}")

    payload = json.loads(summary.read_text(encoding="utf-8"))
    species_arrays: Dict[str, SpeciesArrays] = {}
    with load_multispecies_details_npz(details, summary_payload=payload) as arr:
        species_meta = payload.get("species", {})
        for sp_key, meta in species_meta.items():
            if not isinstance(meta, dict):
                continue
            tag = str(meta.get("tag", "")).strip()
            if not tag:
                continue
            y_key = f"{tag}_y"
            s_key = f"{tag}_sigma"
            if y_key not in arr.files or s_key not in arr.files:
                continue
            pred_by_model: Dict[str, np.ndarray] = {}
            for model_name, suffix in MODEL_SUFFIX.items():
                k = f"{tag}_{suffix}"
                if k in arr.files:
                    pred_by_model[model_name] = np.asarray(arr[k], dtype=float)
            if not pred_by_model:
                continue
            species_arrays[sp_key] = SpeciesArrays(
                y=np.asarray(arr[y_key], dtype=float),
                sigma=np.asarray(arr[s_key], dtype=float),
                pred_by_model=pred_by_model,
            )

    selected = _species_selector(species_csv, sorted(species_arrays.keys()))
    species_arrays = {k: species_arrays[k] for k in selected}
    if not species_arrays:
        raise RuntimeError(f"No selected species with arrays in {run_dir}")

    n_params_by_model: Dict[str, int] = {}
    n_params_by_model["Resonance_NNLS_shared"] = int(
        payload.get("resonance", {}).get("global_metrics", {}).get("n_params", 1)
    )
    for model_name in ("PYTHIA8_external", "EPOS_external", "UrQMD_external"):
        # Each external model uses one independent scale parameter per species.
        n_params_by_model[model_name] = int(len(species_arrays))

    return RunPayload(
        label=label,
        run_dir=run_dir,
        species_arrays=species_arrays,
        n_params_by_model=n_params_by_model,
    )


def _concat_by_model(species_arrays: Dict[str, SpeciesArrays]) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out: Dict[str, Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]] = {}
    for model_name in MODEL_SUFFIX:
        out[model_name] = ([], [], [])
    for sa in species_arrays.values():
        for model_name in MODEL_SUFFIX:
            pred = sa.pred_by_model.get(model_name)
            if pred is None:
                continue
            out[model_name][0].append(np.asarray(pred, dtype=float))
            out[model_name][1].append(np.asarray(sa.y, dtype=float))
            out[model_name][2].append(np.asarray(sa.sigma, dtype=float))
    flat: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for model_name, (preds, ys, sigmas) in out.items():
        if not preds:
            continue
        flat[model_name] = (
            np.concatenate(preds).astype(float, copy=False),
            np.concatenate(ys).astype(float, copy=False),
            np.concatenate(sigmas).astype(float, copy=False),
        )
    return flat


def _bootstrap_run(
    rp: RunPayload,
    n_boot: int,
    seed: int,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, np.ndarray]], Dict[str, float]]:
    rng = np.random.default_rng(int(seed))
    flat = _concat_by_model(rp.species_arrays)
    available_models = [m for m in MODEL_SUFFIX if m in flat]
    if not available_models:
        raise RuntimeError(f"No usable model arrays in {rp.run_dir}")

    baseline: Dict[str, Dict[str, float]] = {}
    for model_name in available_models:
        pred, y, sigma = flat[model_name]
        baseline[model_name] = _metrics(pred, y, sigma, n_params=rp.n_params_by_model.get(model_name, 1))

    boot_by_model: Dict[str, Dict[str, List[float]]] = {}
    for model_name in available_models:
        boot_by_model[model_name] = {
            "chi2": [],
            "chi2_dof": [],
            "wrrmse": [],
            "log_chi2": [],
            "log_pull_rms": [],
        }

    species_items = list(rp.species_arrays.items())
    for _ in range(max(int(n_boot), 1)):
        per_model_pred: Dict[str, List[np.ndarray]] = {m: [] for m in available_models}
        per_model_y: Dict[str, List[np.ndarray]] = {m: [] for m in available_models}
        per_model_sigma: Dict[str, List[np.ndarray]] = {m: [] for m in available_models}
        for _, sa in species_items:
            n = int(len(sa.y))
            idx = rng.integers(0, n, size=n, endpoint=False)
            yb = sa.y[idx]
            sb = sa.sigma[idx]
            for model_name in available_models:
                pb = sa.pred_by_model[model_name][idx]
                per_model_pred[model_name].append(pb)
                per_model_y[model_name].append(yb)
                per_model_sigma[model_name].append(sb)
        for model_name in available_models:
            pred = np.concatenate(per_model_pred[model_name])
            y = np.concatenate(per_model_y[model_name])
            sigma = np.concatenate(per_model_sigma[model_name])
            mm = _metrics(pred, y, sigma, n_params=rp.n_params_by_model.get(model_name, 1))
            for key in boot_by_model[model_name]:
                boot_by_model[model_name][key].append(float(mm[key]))

    boot_arr: Dict[str, Dict[str, np.ndarray]] = {}
    for model_name in available_models:
        boot_arr[model_name] = {k: np.asarray(v, dtype=float) for k, v in boot_by_model[model_name].items()}

    # Probability each model is best in log-pull metric.
    n_rep = len(next(iter(boot_arr.values()))["log_pull_rms"])
    best_count = {m: 0 for m in available_models}
    for i in range(n_rep):
        vals = {m: float(boot_arr[m]["log_pull_rms"][i]) for m in available_models}
        best_model = min(vals.items(), key=lambda kv: kv[1])[0]
        best_count[best_model] += 1
    p_best = {m: float(best_count[m] / max(n_rep, 1)) for m in available_models}

    return baseline, boot_arr, p_best


def _quantile_stats(values: np.ndarray) -> Dict[str, float]:
    vv = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(vv)),
        "std": float(np.std(vv, ddof=1)) if vv.size > 1 else 0.0,
        "q16": float(np.quantile(vv, 0.16)),
        "q50": float(np.quantile(vv, 0.50)),
        "q84": float(np.quantile(vv, 0.84)),
        "q025": float(np.quantile(vv, 0.025)),
        "q975": float(np.quantile(vv, 0.975)),
    }


def _plot_run(path: Path, label: str, stats_rows: Sequence[Dict[str, object]]) -> None:
    rows = list(stats_rows)
    models = [str(r["model_short"]) for r in rows if str(r["metric"]) == "log_pull_rms"]
    y_log = [float(r["median"]) for r in rows if str(r["metric"]) == "log_pull_rms"]
    lo_log = [float(r["median"]) - float(r["q16"]) for r in rows if str(r["metric"]) == "log_pull_rms"]
    hi_log = [float(r["q84"]) - float(r["median"]) for r in rows if str(r["metric"]) == "log_pull_rms"]
    y_wr = [float(r["median"]) for r in rows if str(r["metric"]) == "wrrmse"]
    lo_wr = [float(r["median"]) - float(r["q16"]) for r in rows if str(r["metric"]) == "wrrmse"]
    hi_wr = [float(r["q84"]) - float(r["median"]) for r in rows if str(r["metric"]) == "wrrmse"]

    x = np.arange(len(models), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))

    axes[0].errorbar(x, y_log, yerr=np.vstack([lo_log, hi_log]), fmt="o", color="#1d3557", capsize=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=20, ha="right")
    axes[0].set_ylabel("log-pull RMS")
    axes[0].set_title("Bootstrap 68% band")
    axes[0].grid(alpha=0.25, lw=0.6)

    axes[1].errorbar(x, y_wr, yerr=np.vstack([lo_wr, hi_wr]), fmt="o", color="#457b9d", capsize=3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=20, ha="right")
    axes[1].set_ylabel("weighted relative RMSE")
    axes[1].set_title("Bootstrap 68% band")
    axes[1].grid(alpha=0.25, lw=0.6)

    fig.suptitle(f"Model robustness bands: {label}", fontsize=12)
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True, help="Run directory containing summary+details.")
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Optional label (repeatable, aligned to --run-dir order). Default: run-dir name.",
    )
    parser.add_argument("--species", default="", help="Comma-separated species subset. Default: all.")
    parser.add_argument("--n-bootstrap", type=int, default=2500, help="Bootstrap replicates.")
    parser.add_argument("--seed", type=int, default=20260331, help="RNG seed.")
    parser.add_argument(
        "--out-dir",
        default="dev/00runs/bootstrap_robustness_2026-03-31",
        help="Output directory for csv/plots/summary.",
    )
    args = parser.parse_args()

    run_dirs = [Path(x).resolve() for x in args.run_dir]
    labels = list(args.label)
    while len(labels) < len(run_dirs):
        labels.append(run_dirs[len(labels)].name)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_md: List[str] = ["# Bootstrap Robustness Bands", ""]
    for idx, (run_dir, label) in enumerate(zip(run_dirs, labels)):
        rp = _load_run(run_dir=run_dir, label=label, species_csv=str(args.species))
        baseline, boot_arr, p_best = _bootstrap_run(
            rp,
            n_boot=int(args.n_bootstrap),
            seed=int(args.seed) + int(idx) * 101,
        )

        stats_rows: List[Dict[str, object]] = []
        for model_name in MODEL_SUFFIX:
            if model_name not in boot_arr:
                continue
            for metric in ("log_pull_rms", "wrrmse", "log_chi2", "chi2"):
                qq = _quantile_stats(boot_arr[model_name][metric])
                stats_rows.append(
                    {
                        "run_label": label,
                        "run_dir": str(run_dir),
                        "model": model_name,
                        "model_short": MODEL_SHORT.get(model_name, model_name),
                        "metric": metric,
                        "baseline": float(baseline[model_name][metric]),
                        "mean": qq["mean"],
                        "std": qq["std"],
                        "q16": qq["q16"],
                        "median": qq["q50"],
                        "q84": qq["q84"],
                        "q025": qq["q025"],
                        "q975": qq["q975"],
                        "p_best_log_pull": float(p_best.get(model_name, 0.0)),
                    }
                )

        csv_path = out_dir / f"bootstrap_model_bands_{label.replace(' ', '_')}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "run_label",
                    "run_dir",
                    "model",
                    "model_short",
                    "metric",
                    "baseline",
                    "mean",
                    "std",
                    "q16",
                    "median",
                    "q84",
                    "q025",
                    "q975",
                    "p_best_log_pull",
                ],
            )
            writer.writeheader()
            writer.writerows(stats_rows)

        png_path = out_dir / f"bootstrap_model_bands_{label.replace(' ', '_')}.png"
        _plot_run(png_path, label, stats_rows)

        summary_md.append(f"## {label}")
        summary_md.append("")
        summary_md.append(f"- run_dir: `{run_dir}`")
        summary_md.append(f"- species: `{', '.join(sorted(rp.species_arrays.keys()))}`")
        summary_md.append(f"- bootstrap replicates: `{int(args.n_bootstrap)}`")
        summary_md.append(f"- results csv: `{csv_path}`")
        summary_md.append(f"- plot: `{png_path}`")
        summary_md.append("")
        summary_md.append("| model | baseline log-pull RMS | 68% band | p(best log-pull) |")
        summary_md.append("|---|---:|---:|---:|")
        for model_name in MODEL_SUFFIX:
            if model_name not in boot_arr:
                continue
            q = _quantile_stats(boot_arr[model_name]["log_pull_rms"])
            summary_md.append(
                f"| {MODEL_SHORT.get(model_name, model_name)} | "
                f"{float(baseline[model_name]['log_pull_rms']):.6g} | "
                f"[{q['q16']:.6g}, {q['q84']:.6g}] | {float(p_best.get(model_name, 0.0)):.3f} |"
            )
        summary_md.append("")

    md_path = out_dir / "BOOTSTRAP_ROBUSTNESS_SUMMARY.md"
    md_path.write_text("\n".join(summary_md) + "\n", encoding="utf-8")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()

