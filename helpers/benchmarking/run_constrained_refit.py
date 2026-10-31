#!/usr/bin/env python3
"""Dedicated constrained shared-weight refit runner."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "dev" / "00runs"
BENCHMARK_SCRIPT = REPO_ROOT / "helpers" / "benchmarking" / "benchmark_multispecies_constrained.py"
DEFAULT_CONFIG_JSON = (
    REPO_ROOT / "helpers" / "benchmarking" / "config" / "constrained_refit_table9_strange_defaults.json"
)
PATH_LIKE_KEYS = {
    "out_dir",
    "mixing2006_dir",
    "external_run_dir",
    "res_cache_dir",
    "prior_table9_json",
    "prior_abs_yield_json",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _normalize_key(key: str) -> str:
    return str(key).strip().replace("-", "_")


def _parse_set_arg(raw: str) -> Tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"Invalid --set override '{raw}'. Expected KEY=VALUE.")
    key, val = raw.split("=", 1)
    k = _normalize_key(key)
    txt = val.strip()
    try:
        parsed = json.loads(txt)
    except Exception:
        low = txt.lower()
        if low == "true":
            parsed = True
        elif low == "false":
            parsed = False
        elif low == "null":
            parsed = None
        else:
            parsed = txt
    return k, parsed


def _as_path_string(value: Any) -> str:
    p = Path(str(value)).expanduser()
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return str(p)


def _load_config(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config is not a JSON object: {path}")
    bench = payload.get("benchmark_args")
    if not isinstance(bench, dict):
        raise ValueError(f"Config missing object key 'benchmark_args': {path}")
    return payload


def _to_cli_args(benchmark_args: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for key, val in benchmark_args.items():
        opt = "--" + str(key).replace("_", "-")
        if isinstance(val, bool):
            if val:
                out.append(opt)
            continue
        if val is None:
            continue
        if isinstance(val, list):
            for item in val:
                out.extend([opt, str(item)])
            continue
        out.extend([opt, str(val)])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run dedicated Table-9 + strange/hyperon constrained shared-weight refit."
    )
    parser.add_argument("--config-json", default=str(DEFAULT_CONFIG_JSON))
    parser.add_argument("--run-description", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--res-events-per-res", type=int, default=None)
    parser.add_argument("--res-workers", type=int, default=None)
    parser.add_argument("--res-max-decay-passes", type=int, default=None)
    parser.add_argument("--external-run-dir", default="")
    parser.add_argument("--res-cache-dir", default="")
    parser.add_argument("--mixing2006-dir", default="")
    parser.add_argument("--fit-objective", default="")
    parser.add_argument("--fit-weight-mode", default="")
    parser.add_argument("--prior-table9-lambda", type=float, default=None)
    parser.add_argument("--prior-strange-lambda", type=float, default=None)
    parser.add_argument("--prior-hyperon-lambda", type=float, default=None)
    parser.add_argument("--prior-abs-yield-lambda", type=float, default=None)
    parser.add_argument("--prior-abs-yield-json", default="")
    parser.add_argument("--prior-abs-yield-window", default="")
    parser.add_argument("--prior-abs-yield-species", default="")
    parser.add_argument("--prior-abs-yield-min-sigma-rel", type=float, default=None)
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--res-cache-force-rebuild", action="store_true")
    parser.add_argument("--clear-soft-overrides", action="store_true")
    parser.add_argument(
        "--append-soft-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Append additional --res-soft-override entries.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Generic benchmark argument override; VALUE may be JSON literal.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config_json).expanduser()
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config JSON not found: {config_path}")

    payload = _load_config(config_path)
    benchmark_args = dict(payload["benchmark_args"])

    for key in PATH_LIKE_KEYS:
        if key in benchmark_args and benchmark_args[key]:
            benchmark_args[key] = _as_path_string(benchmark_args[key])

    if args.run_description:
        benchmark_args["run_description"] = str(args.run_description).strip()
    if args.out_dir:
        benchmark_args["out_dir"] = _as_path_string(args.out_dir)
    if args.seed is not None:
        benchmark_args["seed"] = int(args.seed)
    if args.res_events_per_res is not None:
        benchmark_args["res_events_per_res"] = int(args.res_events_per_res)
    if args.res_workers is not None:
        benchmark_args["res_workers"] = max(int(args.res_workers), 1)
    if args.res_max_decay_passes is not None:
        benchmark_args["res_max_decay_passes"] = max(int(args.res_max_decay_passes), 1)
    if args.external_run_dir:
        benchmark_args["external_run_dir"] = _as_path_string(args.external_run_dir)
    if args.res_cache_dir:
        benchmark_args["res_cache_dir"] = _as_path_string(args.res_cache_dir)
    if args.mixing2006_dir:
        benchmark_args["mixing2006_dir"] = _as_path_string(args.mixing2006_dir)
    if args.fit_objective:
        benchmark_args["fit_objective"] = str(args.fit_objective).strip()
    if args.fit_weight_mode:
        benchmark_args["fit_weight_mode"] = str(args.fit_weight_mode).strip()
    if args.prior_table9_lambda is not None:
        benchmark_args["prior_table9_lambda"] = float(args.prior_table9_lambda)
    if args.prior_strange_lambda is not None:
        benchmark_args["prior_strange_lambda"] = float(args.prior_strange_lambda)
    if args.prior_hyperon_lambda is not None:
        benchmark_args["prior_hyperon_lambda"] = float(args.prior_hyperon_lambda)
    if args.prior_abs_yield_lambda is not None:
        benchmark_args["prior_abs_yield_lambda"] = float(args.prior_abs_yield_lambda)
    if args.prior_abs_yield_json:
        benchmark_args["prior_abs_yield_json"] = _as_path_string(args.prior_abs_yield_json)
    if args.prior_abs_yield_window:
        benchmark_args["prior_abs_yield_window"] = str(args.prior_abs_yield_window).strip()
    if args.prior_abs_yield_species:
        benchmark_args["prior_abs_yield_species"] = str(args.prior_abs_yield_species).strip()
    if args.prior_abs_yield_min_sigma_rel is not None:
        benchmark_args["prior_abs_yield_min_sigma_rel"] = float(args.prior_abs_yield_min_sigma_rel)
    if args.skip_external:
        benchmark_args["skip_external"] = True
    if args.res_cache_force_rebuild:
        benchmark_args["res_cache_force_rebuild"] = True

    if args.clear_soft_overrides:
        benchmark_args["res_soft_override"] = []
    if args.append_soft_override:
        current = benchmark_args.get("res_soft_override", [])
        if not isinstance(current, list):
            current = [str(current)]
        current.extend([str(x) for x in args.append_soft_override if str(x).strip()])
        benchmark_args["res_soft_override"] = current

    for raw in args.set:
        key, val = _parse_set_arg(raw)
        if key in PATH_LIKE_KEYS and isinstance(val, str) and val.strip():
            benchmark_args[key] = _as_path_string(val)
        else:
            benchmark_args[key] = val

    run_description = str(benchmark_args.get("run_description", "")).strip()
    if not run_description:
        run_description = f"constrained_refit_{datetime.now().strftime('%Y-%m-%d')}"
        benchmark_args["run_description"] = run_description
    out_dir_value = benchmark_args.get("out_dir")
    out_dir = Path(str(out_dir_value)) if out_dir_value else (RUNS_ROOT / run_description)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    benchmark_args["out_dir"] = str(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    command = [sys.executable, str(BENCHMARK_SCRIPT), *_to_cli_args(benchmark_args)]
    manifest = {
        "created_utc": _utc_now(),
        "config_json": str(config_path),
        "benchmark_script": str(BENCHMARK_SCRIPT),
        "run_description": run_description,
        "out_dir": str(out_dir),
        "resolved_benchmark_args": benchmark_args,
        "command": command,
        "command_shell": shlex.join(command),
    }
    manifest_path = out_dir / "constrained_refit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"manifest: {manifest_path}", flush=True)
    print(f"command: {manifest['command_shell']}", flush=True)
    if args.dry_run:
        print("dry-run: benchmark command not executed.", flush=True)
        return

    subprocess.run(command, check=True, cwd=str(REPO_ROOT))
    print("done: dedicated constrained refit completed.", flush=True)


if __name__ == "__main__":
    main()
