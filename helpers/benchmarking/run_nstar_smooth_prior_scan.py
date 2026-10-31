#!/usr/bin/env python3
"""Run and summarize a lambda scan for the effective N* ladder smoothness prior."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "dev" / "00runs"
DEFAULT_BASE_MANIFEST = (
    RUNS_ROOT
    / "constrained_refit_pdg_absyield_scan_l0300_s0050_2026-03-31r2"
    / "constrained_refit_manifest.json"
)


def _parse_lambdas(spec: str) -> List[float]:
    out: List[float] = []
    for tok in str(spec).split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(float(t))
    if not out:
        raise ValueError("No lambdas specified.")
    return out


def _tag_float(x: float) -> str:
    s = f"{float(x):.6g}"
    return s.replace("+", "").replace("-", "m").replace(".", "p")


def _resolved_args_to_cli(resolved: Dict[str, object]) -> List[str]:
    cli: List[str] = []
    for key in sorted(resolved.keys()):
        if key in {"run_description", "out_dir"}:
            continue
        val = resolved[key]
        if val is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(val, bool):
            if val:
                cli.append(flag)
            continue
        if key == "res_soft_override":
            if isinstance(val, list):
                items = [str(x) for x in val]
            else:
                items = str(val).split(",")
            for item in items:
                txt = str(item).strip()
                if not txt:
                    continue
                cli.extend([flag, txt])
            continue
        if isinstance(val, list):
            for item in val:
                cli.extend([flag, str(item)])
            continue
        cli.extend([flag, str(val)])
    return cli


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize_run(run_dir: Path, lam: float) -> Dict[str, object]:
    summary = _load_json(run_dir / "multispecies_constrained_summary.json")
    fit_solver = summary.get("fit_solver", {}) if isinstance(summary, dict) else {}
    resonance = summary.get("resonance", {}) if isinstance(summary, dict) else {}
    gm = resonance.get("global_metrics", {}) if isinstance(resonance, dict) else {}
    priors = resonance.get("priors", {}) if isinstance(resonance, dict) else {}
    t9 = priors.get("table9", {}) if isinstance(priors, dict) else {}
    ay = priors.get("absolute_yield", {}) if isinstance(priors, dict) else {}
    ns = priors.get("nstar_smooth", {}) if isinstance(priors, dict) else {}
    ns_fit = ns.get("fit", {}) if isinstance(ns, dict) else {}
    wlbl = resonance.get("weights_by_label_sum", {}) if isinstance(resonance, dict) else {}
    active_by_key_raw = resonance.get("active_weights_by_key_sorted", []) if isinstance(resonance, dict) else []

    table9_rows = t9.get("fit_rows", []) if isinstance(t9, dict) else []
    table9_active = 0
    eta_frac = float("nan")
    if isinstance(table9_rows, list) and table9_rows:
        for row in table9_rows:
            if not isinstance(row, dict):
                continue
            frac = float(row.get("fit_fraction", 0.0) or 0.0)
            if frac > 1e-6:
                table9_active += 1
            if str(row.get("selector", "")).strip() == "eta":
                eta_frac = frac

    active_labels: List[str] = []
    if isinstance(active_by_key_raw, list):
        for item in active_by_key_raw[:12]:
            if isinstance(item, dict):
                lbl = str(item.get("label", "")).strip()
                if lbl:
                    active_labels.append(lbl)
    elif isinstance(active_by_key_raw, dict):
        items = sorted(
            ((str(k), float(v)) for k, v in active_by_key_raw.items()),
            key=lambda kv: kv[1],
            reverse=True,
        )
        for k, _ in items[:12]:
            lbl = k.split("::", 1)[0].strip()
            if lbl:
                active_labels.append(lbl)

    row: Dict[str, object] = {
        "lambda_nstar_smooth": float(lam),
        "run_dir": str(run_dir),
        "fit_success": bool(fit_solver.get("success", False)),
        "fit_fun": float(fit_solver.get("fun", float("nan"))),
        "full_log_pull_rms": float(gm.get("full_log_pull_rms", float("nan"))),
        "full_weighted_rel_rmse": float(gm.get("full_weighted_rel_rmse", float("nan"))),
        "full_reduced_chi2": float(gm.get("full_reduced_chi2", float("nan"))),
        "n_active": int(resonance.get("n_active", 0) or 0),
        "table9_fit_fraction_l2_per_row": float(t9.get("fit_fraction_l2_per_row", float("nan"))),
        "table9_active_rows": int(table9_active),
        "table9_eta_fit_fraction": float(eta_frac),
        "abs_yield_fit_chi2_per_row": float(ay.get("fit_chi2_per_row", float("nan"))),
        "nstar_adjacent_diff_rms": float(ns_fit.get("adjacent_diff_rms", float("nan"))),
        "nstar_adjacent_diff_max_abs": float(ns_fit.get("adjacent_diff_max_abs", float("nan"))),
        "eta_weight": float(wlbl.get("eta", float("nan"))) if isinstance(wlbl, dict) else float("nan"),
        "phi_1020_weight": float(wlbl.get("phi_1020", float("nan"))) if isinstance(wlbl, dict) else float("nan"),
        "active_components": ";".join(active_labels),
    }
    return row


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for row in rows:
            wr.writerow(row)


def _write_plot(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    rows_sorted = sorted(rows, key=lambda r: float(r["lambda_nstar_smooth"]))
    x = np.asarray([float(r["lambda_nstar_smooth"]) for r in rows_sorted], dtype=float)
    y_wr = np.asarray([float(r["full_weighted_rel_rmse"]) for r in rows_sorted], dtype=float)
    y_lp = np.asarray([float(r["full_log_pull_rms"]) for r in rows_sorted], dtype=float)
    y_ns = np.asarray([float(r["nstar_adjacent_diff_rms"]) for r in rows_sorted], dtype=float)
    y_eta = np.asarray([float(r["table9_eta_fit_fraction"]) for r in rows_sorted], dtype=float)

    fig = plt.figure(figsize=(10.5, 8.5), constrained_layout=True)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax1.plot(x, y_wr, marker="o", lw=1.6, label="WRRMSE")
    ax1.plot(x, y_lp, marker="s", lw=1.4, label="log-pull RMS")
    ax1.set_xscale("symlog", linthresh=1e-3)
    ax1.set_xlabel("N* smoothness lambda")
    ax1.set_ylabel("Global fit metric")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", frameon=False)
    ax1.set_title("Global Metrics vs N* Smoothness Prior")

    ax2.plot(x, y_ns, marker="^", lw=1.6, label="adjacent diff RMS")
    ax2.plot(x, y_eta, marker="d", lw=1.4, label="Table-9 eta fraction")
    ax2.set_xscale("symlog", linthresh=1e-3)
    ax2.set_xlabel("N* smoothness lambda")
    ax2.set_ylabel("Regularization / occupancy")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", frameon=False)

    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_md(path: Path, rows: List[Dict[str, object]], out_root: Path) -> None:
    lines: List[str] = []
    lines.append("# N* Smoothness-Prior Scan")
    lines.append("")
    lines.append(f"- generated_utc: `{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}`")
    lines.append(f"- out_root: `{out_root}`")
    lines.append("")
    lines.append("| lambda | fit_fun | log-pull RMS | WRRMSE | abs-yield chi2/row | T9 active rows | eta fraction | N* adj RMS |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(rows, key=lambda r: float(r["lambda_nstar_smooth"])):
        lines.append(
            f"| {float(row['lambda_nstar_smooth']):.6g} | "
            f"{float(row['fit_fun']):.6g} | "
            f"{float(row['full_log_pull_rms']):.6g} | "
            f"{float(row['full_weighted_rel_rmse']):.6g} | "
            f"{float(row['abs_yield_fit_chi2_per_row']):.6g} | "
            f"{int(row['table9_active_rows'])} | "
            f"{float(row['table9_eta_fit_fraction']):.6g} | "
            f"{float(row['nstar_adjacent_diff_rms']):.6g} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-manifest", default=str(DEFAULT_BASE_MANIFEST))
    ap.add_argument("--lambdas", default="0,0.1,0.3,1,3,10,30,100,300,1000")
    ap.add_argument("--out-root", default="")
    ap.add_argument("--python", default=sys.executable, help="Python executable for benchmark runs.")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    base_manifest = Path(args.base_manifest).expanduser()
    if not base_manifest.is_absolute():
        base_manifest = (REPO_ROOT / base_manifest).resolve()
    if not base_manifest.exists():
        raise FileNotFoundError(f"Missing base manifest: {base_manifest}")
    base = _load_json(base_manifest)
    resolved = base.get("resolved_benchmark_args", {})
    if not isinstance(resolved, dict) or not resolved:
        raise RuntimeError(f"Missing resolved_benchmark_args in {base_manifest}")

    base_name = str(resolved.get("run_description", "nstar_smooth_base")).strip()
    lambdas = _parse_lambdas(args.lambdas)

    stamp = datetime.now().strftime("%Y-%m-%d")
    out_root = (
        Path(args.out_root).expanduser()
        if str(args.out_root).strip()
        else (RUNS_ROOT / f"nstar_smooth_prior_scan_{stamp}")
    )
    if not out_root.is_absolute():
        out_root = (REPO_ROOT / out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    cli_base = _resolved_args_to_cli(resolved)
    bench_script = REPO_ROOT / "helpers" / "benchmarking" / "benchmark_multispecies_constrained.py"

    manifest_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    for lam in lambdas:
        tag = _tag_float(lam)
        run_name = f"{base_name}_nstarsm_l{tag}"
        run_dir = out_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(Path(args.python).expanduser()),
            str(bench_script),
            *cli_base,
            "--prior-nstar-smooth-lambda",
            str(float(lam)),
            "--run-description",
            run_name,
            "--out-dir",
            str(run_dir),
        ]
        log_path = run_dir / "run.log"
        status = "skipped"
        rc = 0
        if args.skip_existing and (run_dir / "multispecies_constrained_summary.json").exists():
            status = "existing"
        else:
            with log_path.open("w", encoding="utf-8") as log_fh:
                proc = subprocess.run(
                    cmd,
                    cwd=str(REPO_ROOT),
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            rc = int(proc.returncode)
            status = "ok" if rc == 0 else "failed"
        manifest_rows.append(
            {
                "lambda_nstar_smooth": float(lam),
                "run_name": run_name,
                "run_dir": str(run_dir),
                "status": status,
                "return_code": int(rc),
                "command_shell": " ".join(shlex.quote(c) for c in cmd),
            }
        )
        if status in {"ok", "existing"} and (run_dir / "multispecies_constrained_summary.json").exists():
            summary_rows.append(_summarize_run(run_dir, lam))

    manifest_json = out_root / "nstar_smooth_prior_scan_manifest.json"
    manifest_json.write_text(json.dumps({"rows": manifest_rows}, indent=2) + "\n", encoding="utf-8")

    summary_rows = sorted(summary_rows, key=lambda r: float(r["lambda_nstar_smooth"]))
    csv_path = out_root / "nstar_smooth_prior_scan_metrics.csv"
    _write_csv(csv_path, summary_rows)
    png_path = out_root / "nstar_smooth_prior_scan_metrics.png"
    _write_plot(png_path, summary_rows)
    md_path = out_root / "NSTAR_SMOOTH_PRIOR_SCAN_SUMMARY.md"
    _write_md(md_path, summary_rows, out_root)

    print(f"wrote {manifest_json}")
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
