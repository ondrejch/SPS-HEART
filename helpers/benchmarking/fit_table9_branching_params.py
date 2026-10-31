#!/usr/bin/env python3
"""Fit soft-cascade branching knobs to Fischer 2022 Table 9 pi- yields."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Set, Tuple

import numpy as np
from scipy.optimize import minimize


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "dev" / "00runs"
DEFAULT_TABLE9_JSON = REPO_ROOT / "helpers" / "benchmarking" / "config" / "fischer2022_table9_pi_minus.json"
DEFAULT_BOUNDS_JSON = REPO_ROOT / "helpers" / "benchmarking" / "config" / "table9_branching_fit_bounds_pdg2025.json"


@dataclass(frozen=True)
class ResonanceRow:
    name: str
    pdg_id: int
    sector: str
    total_yield: float
    table_branching_fraction: float
    table_pi_minus_yield: float


def _load_rows(table9_path: Path) -> Tuple[List[ResonanceRow], dict]:
    data = json.loads(table9_path.read_text(encoding="utf-8"))
    rows = [ResonanceRow(**row) for row in data["rows"]]
    return rows, data


def _load_fit_config(bounds_path: Path) -> dict:
    return json.loads(bounds_path.read_text(encoding="utf-8"))


def _load_decay_builder() -> Callable:
    mixing2006_dir = REPO_ROOT / "dev" / "mixing2006"
    if str(mixing2006_dir) not in sys.path:
        sys.path.insert(0, str(mixing2006_dir))
    from na49_mixing.decays import build_soft_resonance_decays  # pylint: disable=import-error

    return build_soft_resonance_decays


def _expected_pi_minus_per_parent(
    pdg_id: int,
    chain: dict,
    memo: Dict[int, float],
    visiting: Set[int],
) -> float:
    if pdg_id == -211:
        return 1.0
    if pdg_id in memo:
        return memo[pdg_id]
    if pdg_id in visiting:
        raise RuntimeError(f"Cycle detected in decay chain at PDG {pdg_id}.")

    pdecays = chain.get(pdg_id)
    if pdecays is None:
        memo[pdg_id] = 0.0
        return 0.0

    visiting.add(pdg_id)
    total = 0.0
    prev = 0.0
    for cum, mode in zip(pdecays._probabilities, pdecays._modes):
        prob = max(float(cum) - prev, 0.0)
        prev = float(cum)
        daughters_pi = 0.0
        for daughter_id in mode.daughters:
            daughters_pi += _expected_pi_minus_per_parent(daughter_id, chain, memo, visiting)
        total += prob * daughters_pi
    visiting.remove(pdg_id)
    memo[pdg_id] = total
    return total


def _evaluate_rows(
    rows: List[ResonanceRow],
    build_soft_resonance_decays: Callable,
    overrides: Dict[str, float],
) -> dict:
    chain = build_soft_resonance_decays(branching_overrides=overrides)
    memo: Dict[int, float] = {}
    out_rows: List[dict] = []
    rel_residuals: List[float] = []

    for row in rows:
        exp_pi = _expected_pi_minus_per_parent(row.pdg_id, chain, memo, set())
        model_yield = row.total_yield * exp_pi
        delta = model_yield - row.table_pi_minus_yield
        rel = delta / max(row.table_pi_minus_yield, 1e-12)
        rel_residuals.append(rel)
        out_rows.append(
            {
                "name": row.name,
                "pdg_id": row.pdg_id,
                "sector": row.sector,
                "total_yield": row.total_yield,
                "table_branching_fraction": row.table_branching_fraction,
                "table_pi_minus_yield": row.table_pi_minus_yield,
                "model_expected_pi_minus_per_parent": exp_pi,
                "model_pi_minus_yield": model_yield,
                "delta_model_minus_table": delta,
                "relative_delta": rel,
            }
        )

    obj = float(np.sum(np.square(np.asarray(rel_residuals, dtype=float))))
    total_model = float(sum(r["model_pi_minus_yield"] for r in out_rows))
    baryon_model = float(sum(r["model_pi_minus_yield"] for r in out_rows if r["sector"] == "baryon"))
    meson_model = float(sum(r["model_pi_minus_yield"] for r in out_rows if r["sector"] == "meson"))
    total_table = float(sum(r.table_pi_minus_yield for r in rows))

    return {
        "rows": out_rows,
        "totals": {
            "model_pi_minus_total": total_model,
            "model_pi_minus_from_baryons": baryon_model,
            "model_pi_minus_from_mesons": meson_model,
            "table_pi_minus_total": total_table,
            "model_minus_table_total": total_model - total_table,
            "model_over_table_total": (total_model / total_table) if total_table > 0 else math.nan,
        },
        "objective_relative_sse": obj,
    }


def _build_problem(fit_cfg: dict) -> Tuple[List[str], np.ndarray, List[Tuple[float, float]], List[dict]]:
    params = fit_cfg["parameters"]
    names = [p["name"] for p in params]
    x0 = np.asarray([float(p["initial"]) for p in params], dtype=float)
    bounds = [(float(p["min"]), float(p["max"])) for p in params]
    idx = {name: i for i, name in enumerate(names)}

    constraints: List[dict] = []
    for c in fit_cfg.get("constraints", []):
        if c.get("type") != "sum_leq":
            continue
        inds = [idx[name] for name in c["params"]]
        rhs = float(c["rhs"])

        def _fun(x, inds=inds, rhs=rhs):
            return rhs - float(np.sum(x[inds]))

        constraints.append({"type": "ineq", "fun": _fun, "name": c.get("name", "sum_leq")})
    return names, x0, bounds, constraints


def _constraints_ok(x: np.ndarray, constraints: List[dict], tol: float = 1e-7) -> bool:
    for c in constraints:
        if c["type"] != "ineq":
            continue
        if float(c["fun"](x)) < -tol:
            return False
    return True


def _fit(
    rows: List[ResonanceRow],
    fit_cfg: dict,
    build_soft_resonance_decays: Callable,
    starts: int,
    seed: int,
    maxiter: int,
) -> dict:
    names, x0, bounds, constraints = _build_problem(fit_cfg)
    rng = np.random.default_rng(seed)

    def _x_to_overrides(x: np.ndarray) -> Dict[str, float]:
        return {name: float(val) for name, val in zip(names, x)}

    def _obj(x: np.ndarray) -> float:
        eval_out = _evaluate_rows(rows, build_soft_resonance_decays, _x_to_overrides(x))
        return float(eval_out["objective_relative_sse"])

    start_points = [x0]
    for _ in range(max(0, starts - 1)):
        x = np.asarray([rng.uniform(lo, hi) for lo, hi in bounds], dtype=float)
        start_points.append(x)

    best = None
    runs = []
    for i, xstart in enumerate(start_points):
        res = minimize(
            _obj,
            xstart,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": int(maxiter), "ftol": 1e-12, "disp": False},
        )
        feasible = _constraints_ok(np.asarray(res.x, dtype=float), constraints)
        score = float(res.fun) if np.isfinite(res.fun) else math.inf
        runs.append(
            {
                "run_index": i,
                "success": bool(res.success),
                "status": int(res.status),
                "message": str(res.message),
                "score": score,
                "feasible": feasible,
            }
        )
        candidate_ok = feasible and np.isfinite(score)
        if not candidate_ok:
            continue
        if best is None or score < best["score"]:
            best = {"score": score, "x": np.asarray(res.x, dtype=float), "res": res}

    if best is None:
        raise RuntimeError("No feasible fit result found.")

    x_best = best["x"]
    overrides = _x_to_overrides(x_best)
    baseline_overrides = _x_to_overrides(x0)
    baseline_eval = _evaluate_rows(rows, build_soft_resonance_decays, baseline_overrides)
    tuned_eval = _evaluate_rows(rows, build_soft_resonance_decays, overrides)

    fit_params = []
    for name, init, val, (lo, hi) in zip(names, x0, x_best, bounds):
        fit_params.append(
            {
                "name": name,
                "initial": float(init),
                "tuned": float(val),
                "min": float(lo),
                "max": float(hi),
                "at_lower_bound": abs(float(val) - float(lo)) < 1e-8,
                "at_upper_bound": abs(float(val) - float(hi)) < 1e-8,
            }
        )

    return {
        "fit_parameters": fit_params,
        "initial_overrides": baseline_overrides,
        "tuned_overrides": overrides,
        "baseline": baseline_eval,
        "tuned": tuned_eval,
        "runs": runs,
        "best_score": float(best["score"]),
    }


def _write_outputs(
    out_json: Path,
    out_md: Path,
    out_overrides: Path,
    table9_path: Path,
    bounds_path: Path,
    table9_data: dict,
    fit_cfg: dict,
    fit_result: dict,
) -> None:
    payload = {
        "inputs": {
            "table9_json": str(table9_path),
            "bounds_json": str(bounds_path),
        },
        "table9_metadata": table9_data.get("metadata", {}),
        "fit_config_metadata": fit_cfg.get("metadata", {}),
        "fit_config_sources": fit_cfg.get("sources", {}),
        "fit": fit_result,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    out_overrides_payload = {
        "metadata": {
            "name": "fischer2022_table9_branching_fit_overrides",
            "source_table": str(table9_path),
            "source_bounds": str(bounds_path),
        },
        "branching_overrides": fit_result["tuned_overrides"],
    }
    out_overrides.parent.mkdir(parents=True, exist_ok=True)
    out_overrides.write_text(json.dumps(out_overrides_payload, indent=2), encoding="utf-8")

    base = fit_result["baseline"]
    tuned = fit_result["tuned"]
    lines: List[str] = []
    lines.append("# Table 9 Branching Fit (Constrained)")
    lines.append("")
    lines.append("Inputs:")
    lines.append(f"- Table 9 JSON: `{table9_path}`")
    lines.append(f"- Bounds JSON: `{bounds_path}`")
    lines.append("")
    lines.append("Objective:")
    lines.append(f"- Baseline relative SSE: `{base['objective_relative_sse']:.8f}`")
    lines.append(f"- Tuned relative SSE: `{tuned['objective_relative_sse']:.8f}`")
    lines.append(f"- Improvement factor: `{(base['objective_relative_sse'] / tuned['objective_relative_sse']):.4f}`")
    lines.append("")
    lines.append("Totals:")
    lines.append(
        f"- Baseline model pi- total: `{base['totals']['model_pi_minus_total']:.6f}` "
        f"(ratio `{base['totals']['model_over_table_total']:.6f}` vs table)"
    )
    lines.append(
        f"- Tuned model pi- total: `{tuned['totals']['model_pi_minus_total']:.6f}` "
        f"(ratio `{tuned['totals']['model_over_table_total']:.6f}` vs table)"
    )
    lines.append("")
    lines.append("Tuned parameter values:")
    lines.append("| Parameter | Initial | Tuned | Min | Max |")
    lines.append("|---|---:|---:|---:|---:|")
    for p in fit_result["fit_parameters"]:
        lines.append(
            f"| {p['name']} | {p['initial']:.6f} | {p['tuned']:.6f} | {p['min']:.6f} | {p['max']:.6f} |"
        )
    lines.append("")
    lines.append("Row-wise yields (tuned):")
    lines.append("| Resonance | Table pi- yield | Tuned model pi- yield | Delta |")
    lines.append("|---|---:|---:|---:|")
    for row in tuned["rows"]:
        lines.append(
            f"| {row['name']} | {row['table_pi_minus_yield']:.6f} | {row['model_pi_minus_yield']:.6f} | "
            f"{row['delta_model_minus_table']:.6f} |"
        )
    lines.append("")
    lines.append("Outputs:")
    lines.append(f"- JSON fit report: `{out_json}`")
    lines.append(f"- Tuned overrides JSON: `{out_overrides}`")
    lines.append(
        "- You can re-run the checker with tuned overrides via:"
    )
    lines.append(
        f"  `python helpers/benchmarking/check_fischer2022_table9_pi_minus.py --overrides-json {out_overrides}`"
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit branch knobs to Fischer 2022 Table 9 pi- yields.")
    parser.add_argument("--table9-json", default=str(DEFAULT_TABLE9_JSON))
    parser.add_argument("--bounds-json", default=str(DEFAULT_BOUNDS_JSON))
    parser.add_argument("--run-description", default="local_default")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    parser.add_argument("--out-overrides-json", default="")
    parser.add_argument("--starts", type=int, default=32, help="Number of multi-start attempts.")
    parser.add_argument("--seed", type=int, default=20260329)
    parser.add_argument("--maxiter", type=int, default=1000)
    args = parser.parse_args()

    table9_path = Path(args.table9_json)
    bounds_path = Path(args.bounds_json)
    run_dir = RUNS_ROOT / args.run_description
    out_json = Path(args.out_json) if args.out_json else (run_dir / "fischer2022_table9_branching_fit.json")
    out_md = Path(args.out_md) if args.out_md else (run_dir / "FISCHER2022_TABLE9_BRANCHING_FIT.md")
    out_overrides = (
        Path(args.out_overrides_json)
        if args.out_overrides_json
        else (run_dir / "fischer2022_table9_branching_fit_overrides.json")
    )

    if not table9_path.exists():
        raise FileNotFoundError(f"Missing table JSON: {table9_path}")
    if not bounds_path.exists():
        raise FileNotFoundError(f"Missing bounds JSON: {bounds_path}")

    rows, table9_data = _load_rows(table9_path)
    fit_cfg = _load_fit_config(bounds_path)
    build_soft_resonance_decays = _load_decay_builder()
    fit_result = _fit(
        rows=rows,
        fit_cfg=fit_cfg,
        build_soft_resonance_decays=build_soft_resonance_decays,
        starts=int(args.starts),
        seed=int(args.seed),
        maxiter=int(args.maxiter),
    )
    _write_outputs(
        out_json=out_json,
        out_md=out_md,
        out_overrides=out_overrides,
        table9_path=table9_path,
        bounds_path=bounds_path,
        table9_data=table9_data,
        fit_cfg=fit_cfg,
        fit_result=fit_result,
    )

    print(f"Saved fit JSON: {out_json}")
    print(f"Saved fit Markdown: {out_md}")
    print(f"Saved tuned overrides JSON: {out_overrides}")


if __name__ == "__main__":
    main()
