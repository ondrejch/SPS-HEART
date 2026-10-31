#!/usr/bin/env python3
"""Wait for two 10x ladder campaigns, then summarize and compare them."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_done(path: Path) -> bool:
    if not path.exists():
        return False
    payload = _load_json(path)
    chunk_count = int(payload.get("chunk_count", 0))
    chunk_final = payload.get("chunk_final", {})
    return chunk_count > 0 and len(chunk_final) >= chunk_count


def _manifest_stats(path: Path) -> Tuple[int, int, int]:
    payload = _load_json(path)
    chunk_count = int(payload.get("chunk_count", 0))
    chunk_final = payload.get("chunk_final", {})
    completed = sum(1 for item in chunk_final.values() if item.get("status") == "COMPLETED")
    failed = sum(1 for item in chunk_final.values() if item.get("status") != "COMPLETED")
    return chunk_count, completed, failed


def _top_n(csv_path: Path, n: int = 3) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if i >= n:
                break
            out.append((str(row["rank"]), str(row["scenario"]), str(row["global_wrrmse"])))
    return out


def _best(best_json_path: Path) -> tuple[str, float]:
    payload = _load_json(best_json_path)
    return str(payload["scenario"]), float(payload["global_wrrmse"])


def _run_summary(run_dir: Path, out_prefix: str) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "helpers" / "benchmarking" / "summarize_nstar_ladder_results.py"),
        "--run-dir",
        str(run_dir),
        "--out-prefix",
        str(out_prefix),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--poll-seconds", type=float, default=120.0)
    parser.add_argument("--out-note", default="notes/RUN_10X_RESONANCES_EVAL.md")
    parser.add_argument("--prefix-a", default="nstar_ladder_summary_10x_unconstrained")
    parser.add_argument("--prefix-b", default="nstar_ladder_summary_10x_constrained")
    args = parser.parse_args()

    run_a = (REPO_ROOT / str(args.run_a)).resolve()
    run_b = (REPO_ROOT / str(args.run_b)).resolve()
    man_a = run_a / "gateway_nstar_ladder_dynamic_manifest.json"
    man_b = run_b / "gateway_nstar_ladder_dynamic_manifest.json"

    while True:
        done_a = _manifest_done(man_a)
        done_b = _manifest_done(man_b)
        if done_a and done_b:
            break
        print(
            f"[eval-wait] done_a={done_a} done_b={done_b}; "
            + f"sleep={float(args.poll_seconds):.0f}s",
            flush=True,
        )
        time.sleep(max(float(args.poll_seconds), 1.0))

    _run_summary(run_a, str(args.prefix_a))
    _run_summary(run_b, str(args.prefix_b))

    a_count, a_completed, a_failed = _manifest_stats(man_a)
    b_count, b_completed, b_failed = _manifest_stats(man_b)
    a_best_scenario, a_best_wrrmse = _best(run_a / f"{args.prefix_a}_best.json")
    b_best_scenario, b_best_wrrmse = _best(run_b / f"{args.prefix_b}_best.json")
    a_top3 = _top_n(run_a / f"{args.prefix_a}.csv", n=3)
    b_top3 = _top_n(run_b / f"{args.prefix_b}.csv", n=3)

    lines: list[str] = []
    lines.append("# 10x Resonance Runs: Separate Evaluation")
    lines.append("")
    lines.append("## Run A (Priors Off)")
    lines.append(f"- run dir: `{run_a}`")
    lines.append(f"- manifest completion: `{a_completed}/{a_count}` completed, `{a_failed}` failed")
    lines.append(f"- best scenario: `{a_best_scenario}`")
    lines.append(f"- best global WRRMSE: `{a_best_wrrmse:.12g}`")
    lines.append("- top 3:")
    for rank, scenario, wrrmse in a_top3:
        lines.append(f"  - `{rank}`. `{scenario}` -> `{wrrmse}`")
    lines.append("")
    lines.append("## Run B (Priors On)")
    lines.append(f"- run dir: `{run_b}`")
    lines.append(f"- manifest completion: `{b_completed}/{b_count}` completed, `{b_failed}` failed")
    lines.append(f"- best scenario: `{b_best_scenario}`")
    lines.append(f"- best global WRRMSE: `{b_best_wrrmse:.12g}`")
    lines.append("- top 3:")
    for rank, scenario, wrrmse in b_top3:
        lines.append(f"  - `{rank}`. `{scenario}` -> `{wrrmse}`")
    lines.append("")
    lines.append("## Side-by-Side")
    lines.append(f"- delta best WRRMSE (B - A): `{(b_best_wrrmse - a_best_wrrmse):.12g}`")

    out_note = (REPO_ROOT / str(args.out_note)).resolve()
    out_note.parent.mkdir(parents=True, exist_ok=True)
    out_note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_note}", flush=True)


if __name__ == "__main__":
    main()

