#!/usr/bin/env python3
"""Run balanced-fit constrained campaigns and side studies on the gateway."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from helpers.na49_gateway.cluster_client import ClusterGatewayClient, _default_agent_command
from helpers.na49_gateway.gateway_config import get_config_int, get_config_list, get_config_str, load_user_config

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "00runs"
TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELED", "SUBMIT_FAILED"}
SYNC_TARGETS: tuple[str, ...] = (
    "helpers/benchmarking",
    "na49_mixing",
    "pyproject.toml",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _remote_root_from_config(config: dict[str, Any]) -> str:
    rel = get_config_str(config, "remote_project_relpath")
    if not rel:
        raise ValueError("Config key 'remote_project_relpath' is required.")
    rel = rel.strip()
    if rel in {"~", "$HOME"}:
        return "~"
    if rel.startswith("~/"):
        return rel
    if rel.startswith("$HOME/"):
        return "~/" + rel[6:]
    return "~/" + rel


def _default_workers(config: dict[str, Any]) -> list[str]:
    workers = get_config_list(config, "workers") or []
    if not workers:
        raise ValueError("No workers configured in gateway config.")
    return workers


def _client_from_config(config: dict[str, Any]) -> ClusterGatewayClient:
    return ClusterGatewayClient(
        host=get_config_str(config, "host"),
        agent_command=_default_agent_command(config),
        ssh_options=tuple(get_config_list(config, "ssh_options") or []),
        jobs_root=get_config_str(config, "jobs_root"),
    )


def _sync_remote_code(host: str, remote_root: str, ssh_options: tuple[str, ...]) -> dict[str, Any]:
    if shutil.which("rsync") is None:
        raise RuntimeError("rsync is required for remote code sync but was not found on PATH.")

    ssh_cmd = " ".join(shlex.quote(tok) for tok in (["ssh", *ssh_options]))
    synced: list[str] = []
    for rel in SYNC_TARGETS:
        local_path = REPO_ROOT / rel
        if not local_path.exists():
            raise FileNotFoundError(f"Sync target missing locally: {local_path}")

        remote_path = f"{remote_root}/{rel}"
        remote_parent = remote_path.rsplit("/", 1)[0]
        subprocess.run(["ssh", *ssh_options, host, f"mkdir -p {remote_parent}"], check=True)

        if local_path.is_dir():
            cmd = [
                "rsync",
                "-az",
                "--delete",
                "-e",
                ssh_cmd,
                f"{local_path}/",
                f"{host}:{remote_path}/",
            ]
        else:
            cmd = [
                "rsync",
                "-az",
                "-e",
                ssh_cmd,
                str(local_path),
                f"{host}:{remote_path}",
            ]
        subprocess.run(cmd, check=True)
        synced.append(rel)
        print(f"sync: {rel} -> {host}:{remote_path}", flush=True)

    return {"synced_at": _utc_now(), "targets": synced}


def _wait_for_jobs(client: ClusterGatewayClient, jobs: list[dict[str, Any]], poll_seconds: float) -> dict[str, dict[str, Any]]:
    pending = {j["job_id"] for j in jobs}
    final: dict[str, dict[str, Any]] = {}

    while pending:
        listed = {item["job_id"]: item for item in client.list(limit=max(5000, len(jobs) + 200))}
        newly_done = 0
        for jid in list(pending):
            view = listed.get(jid)
            if view is None:
                try:
                    view = client.status(jid)
                except Exception:
                    continue
            state = str(view.get("state", ""))
            if state in TERMINAL_STATES:
                final[jid] = view
                pending.remove(jid)
                newly_done += 1

        completed = len(final)
        failed = sum(1 for v in final.values() if v.get("state") != "COMPLETED")
        print(
            f"wait: completed={completed}/{len(jobs)} pending={len(pending)} failed={failed} newly_done={newly_done}",
            flush=True,
        )
        if pending:
            time.sleep(max(poll_seconds, 2.0))

    return final


def _mk_main_cmd(
    *,
    python_bin: str,
    remote_root: str,
    run_name: str,
    events_per_res: int,
    res_workers: int,
    max_decay_passes: int,
    fit_weight_mode: str,
    fit_balance_xf_bins: int,
    fit_balance_pt_bins: int,
    prior_table9_lambda: float,
    prior_strange_lambda: float,
    prior_hyperon_lambda: float,
) -> str:
    out_dir = f"{remote_root}/00runs/{run_name}"
    parts = [
        python_bin,
        f"{remote_root}/helpers/benchmarking/benchmark_multispecies_constrained.py",
        "--run-description",
        run_name,
        "--out-dir",
        out_dir,
        "--species",
        "pi-,pi+,k+,k-,p,pbar",
        "--seed",
        "20260329",
        "--res-events-per-res",
        str(int(events_per_res)),
        "--res-max-decay-passes",
        str(int(max_decay_passes)),
        "--res-workers",
        str(int(res_workers)),
        "--res-fragmentation-mode",
        "both",
        "--fit-weight-mode",
        str(fit_weight_mode),
        "--fit-balance-xf-bins",
        str(int(fit_balance_xf_bins)),
        "--fit-balance-pt-bins",
        str(int(fit_balance_pt_bins)),
        "--prior-table9-lambda",
        f"{float(prior_table9_lambda):.12g}",
        "--prior-strange-lambda",
        f"{float(prior_strange_lambda):.12g}",
        "--prior-hyperon-lambda",
        f"{float(prior_hyperon_lambda):.12g}",
        "--skip-external",
    ]
    return " ".join(parts)


def _mk_max_cascade_cmd(
    *,
    python_bin: str,
    remote_root: str,
    run_name: str,
    events_per_res: int,
    res_workers: int,
    fit_weight_mode: str,
    fit_balance_xf_bins: int,
    fit_balance_pt_bins: int,
) -> str:
    parts = [
        python_bin,
        f"{remote_root}/helpers/benchmarking/study_max_cascade.py",
        "--run-description",
        run_name,
        "--pass-values",
        "8,16",
        "--res-events-per-res",
        str(int(events_per_res)),
        "--res-workers",
        str(int(res_workers)),
        "--fit-weight-mode",
        str(fit_weight_mode),
        "--fit-balance-xf-bins",
        str(int(fit_balance_xf_bins)),
        "--fit-balance-pt-bins",
        str(int(fit_balance_pt_bins)),
        "--skip-external",
    ]
    return " ".join(parts)


def _mk_tower_limits_cmd(
    *,
    python_bin: str,
    remote_root: str,
    run_name: str,
    events_per_res: int,
    res_workers: int,
    fit_weight_mode: str,
    fit_balance_xf_bins: int,
    fit_balance_pt_bins: int,
) -> str:
    parts = [
        python_bin,
        f"{remote_root}/helpers/benchmarking/study_tower_limits.py",
        "--run-description",
        run_name,
        "--profiles",
        "maximal",
        "--baryon-max-values",
        "5.0,4.0,3.0,2.3",
        "--meson-max-values",
        "1.25,0.95,0.65,0.45",
        "--res-events-per-res",
        str(int(events_per_res)),
        "--res-workers",
        str(int(res_workers)),
        "--fit-weight-mode",
        str(fit_weight_mode),
        "--fit-balance-xf-bins",
        str(int(fit_balance_xf_bins)),
        "--fit-balance-pt-bins",
        str(int(fit_balance_pt_bins)),
        "--skip-external",
    ]
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gateway orchestrator for balanced-fit reruns.")
    parser.add_argument("--run-description", default="gw_balanced_fit_rerun_2026-03-29")
    parser.add_argument("--workers", default="")
    parser.add_argument("--max-tasks-per-worker", type=int, default=0)
    parser.add_argument("--res-workers", type=int, default=64)
    parser.add_argument("--main-events-per-res", type=int, default=80000)
    parser.add_argument("--side-events-per-res", type=int, default=20000)
    parser.add_argument("--main-max-decay-passes", type=int, default=16)
    parser.add_argument("--fit-weight-mode", default="relative_xfpt_balanced")
    parser.add_argument("--fit-balance-xf-bins", type=int, default=4)
    parser.add_argument("--fit-balance-pt-bins", type=int, default=4)
    parser.add_argument(
        "--python-bin",
        default="",
        help=(
            "Remote python executable for submitted commands. "
            "Defaults to <remote_root>/.venv-gw/bin/python."
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--skip-remote-sync", action="store_true")
    args = parser.parse_args()

    config = load_user_config()
    workers = [x.strip() for x in args.workers.split(",") if x.strip()] if args.workers else _default_workers(config)
    max_tasks_per_worker = (
        int(args.max_tasks_per_worker)
        if int(args.max_tasks_per_worker) > 0
        else int(get_config_int(config, "max_tasks_per_worker", 64) or 64)
    )
    remote_root = _remote_root_from_config(config)
    client = _client_from_config(config)

    host = get_config_str(config, "host")
    ssh_options = tuple(get_config_list(config, "ssh_options") or [])
    if not args.skip_remote_sync:
        _sync_remote_code(host=host, remote_root=remote_root, ssh_options=ssh_options)

    python_bin = str(args.python_bin).strip() if str(args.python_bin).strip() else f"{remote_root}/.venv-gw/bin/python"
    run_root = RUNS_ROOT / args.run_description
    run_root.mkdir(parents=True, exist_ok=True)

    scenarios: list[dict[str, Any]] = []

    # Main constrained 4-step campaign.
    scenarios.append(
        {
            "name": f"{args.run_description}_main_step1_baseline",
            "cmd": _mk_main_cmd(
                python_bin=python_bin,
                remote_root=remote_root,
                run_name=f"{args.run_description}_main_step1_baseline",
                events_per_res=int(args.main_events_per_res),
                res_workers=int(args.res_workers),
                max_decay_passes=int(args.main_max_decay_passes),
                fit_weight_mode=str(args.fit_weight_mode),
                fit_balance_xf_bins=int(args.fit_balance_xf_bins),
                fit_balance_pt_bins=int(args.fit_balance_pt_bins),
                prior_table9_lambda=0.0,
                prior_strange_lambda=0.0,
                prior_hyperon_lambda=0.0,
            ),
            "tasks": int(args.res_workers),
        }
    )
    scenarios.append(
        {
            "name": f"{args.run_description}_main_step2_table9",
            "cmd": _mk_main_cmd(
                python_bin=python_bin,
                remote_root=remote_root,
                run_name=f"{args.run_description}_main_step2_table9",
                events_per_res=int(args.main_events_per_res),
                res_workers=int(args.res_workers),
                max_decay_passes=int(args.main_max_decay_passes),
                fit_weight_mode=str(args.fit_weight_mode),
                fit_balance_xf_bins=int(args.fit_balance_xf_bins),
                fit_balance_pt_bins=int(args.fit_balance_pt_bins),
                prior_table9_lambda=1.0,
                prior_strange_lambda=0.0,
                prior_hyperon_lambda=0.0,
            ),
            "tasks": int(args.res_workers),
        }
    )
    scenarios.append(
        {
            "name": f"{args.run_description}_main_step3_strange",
            "cmd": _mk_main_cmd(
                python_bin=python_bin,
                remote_root=remote_root,
                run_name=f"{args.run_description}_main_step3_strange",
                events_per_res=int(args.main_events_per_res),
                res_workers=int(args.res_workers),
                max_decay_passes=int(args.main_max_decay_passes),
                fit_weight_mode=str(args.fit_weight_mode),
                fit_balance_xf_bins=int(args.fit_balance_xf_bins),
                fit_balance_pt_bins=int(args.fit_balance_pt_bins),
                prior_table9_lambda=0.0,
                prior_strange_lambda=1.0,
                prior_hyperon_lambda=1.0,
            ),
            "tasks": int(args.res_workers),
        }
    )
    scenarios.append(
        {
            "name": f"{args.run_description}_main_step4_both",
            "cmd": _mk_main_cmd(
                python_bin=python_bin,
                remote_root=remote_root,
                run_name=f"{args.run_description}_main_step4_both",
                events_per_res=int(args.main_events_per_res),
                res_workers=int(args.res_workers),
                max_decay_passes=int(args.main_max_decay_passes),
                fit_weight_mode=str(args.fit_weight_mode),
                fit_balance_xf_bins=int(args.fit_balance_xf_bins),
                fit_balance_pt_bins=int(args.fit_balance_pt_bins),
                prior_table9_lambda=1.0,
                prior_strange_lambda=1.0,
                prior_hyperon_lambda=1.0,
            ),
            "tasks": int(args.res_workers),
        }
    )

    # Relevant constrained side studies.
    scenarios.append(
        {
            "name": f"{args.run_description}_side_max_cascade",
            "cmd": _mk_max_cascade_cmd(
                python_bin=python_bin,
                remote_root=remote_root,
                run_name=f"{args.run_description}_side_max_cascade",
                events_per_res=int(args.side_events_per_res),
                res_workers=int(args.res_workers),
                fit_weight_mode=str(args.fit_weight_mode),
                fit_balance_xf_bins=int(args.fit_balance_xf_bins),
                fit_balance_pt_bins=int(args.fit_balance_pt_bins),
            ),
            "tasks": int(args.res_workers),
        }
    )
    scenarios.append(
        {
            "name": f"{args.run_description}_side_tower_limits",
            "cmd": _mk_tower_limits_cmd(
                python_bin=python_bin,
                remote_root=remote_root,
                run_name=f"{args.run_description}_side_tower_limits",
                events_per_res=int(args.side_events_per_res),
                res_workers=int(args.res_workers),
                fit_weight_mode=str(args.fit_weight_mode),
                fit_balance_xf_bins=int(args.fit_balance_xf_bins),
                fit_balance_pt_bins=int(args.fit_balance_pt_bins),
            ),
            "tasks": int(args.res_workers),
        }
    )

    jobs: list[dict[str, Any]] = []
    for idx, sc in enumerate(scenarios, start=1):
        job = client.submit(
            sc["cmd"],
            workdir=remote_root,
            workers=workers,
            name=sc["name"],
            tasks=int(sc["tasks"]),
            max_tasks_per_worker=max_tasks_per_worker,
            wait_for_slot=True,
            poll_seconds=2.0,
        )
        rec = {
            "scenario": sc["name"],
            "cmd": sc["cmd"],
            "tasks": int(sc["tasks"]),
            "job_id": job["job_id"],
            "submitted_at": _utc_now(),
        }
        jobs.append(rec)
        print(f"[{idx}/{len(scenarios)}] submitted {sc['name']} -> {job['job_id']}", flush=True)

    final = _wait_for_jobs(client, jobs, poll_seconds=max(float(args.poll_seconds), 2.0))

    results: list[dict[str, Any]] = []
    for rec in jobs:
        jid = rec["job_id"]
        view = final.get(jid, {})
        item = dict(rec)
        item["state"] = view.get("state", "UNKNOWN")
        item["worker"] = view.get("worker")
        item["exit_code"] = view.get("exit_code")
        try:
            result_view = client.result(jid, tail_lines=80)
        except Exception as exc:
            result_view = {"error": str(exc)}
        item["result"] = result_view
        results.append(item)

    payload = {
        "run_description": args.run_description,
        "fit_weight_mode": args.fit_weight_mode,
        "fit_balance_xf_bins": int(args.fit_balance_xf_bins),
        "fit_balance_pt_bins": int(args.fit_balance_pt_bins),
        "workers": workers,
        "max_tasks_per_worker": int(max_tasks_per_worker),
        "res_workers": int(args.res_workers),
        "main_events_per_res": int(args.main_events_per_res),
        "side_events_per_res": int(args.side_events_per_res),
        "submitted_at": jobs[0]["submitted_at"] if jobs else _utc_now(),
        "finished_at": _utc_now(),
        "results": results,
    }

    out_json = run_root / "gateway_balanced_rerun_jobs.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    out_md = run_root / "GATEWAY_BALANCED_RERUN_SUMMARY.md"
    lines = [
        "# Gateway Balanced-Fit Rerun",
        "",
        f"- run: `{args.run_description}`",
        f"- fit mode: `{args.fit_weight_mode}`",
        f"- balance bins: xF `{int(args.fit_balance_xf_bins)}`, pT `{int(args.fit_balance_pt_bins)}`",
        f"- workers: `{', '.join(workers)}`",
        f"- max tasks/worker: `{int(max_tasks_per_worker)}`",
        f"- per-job tasks: `{int(args.res_workers)}`",
        "",
        "## Jobs",
        "",
        "| scenario | job_id | state | worker | exit_code |",
        "|---|---|---|---|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item['scenario']} | {item['job_id']} | {item.get('state','')} | "
            f"{item.get('worker','')} | {item.get('exit_code','')} |"
        )
    lines.append("")
    lines.append(f"- JSON: `{out_json}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    failed = [r for r in results if r.get("state") != "COMPLETED"]
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    if failed:
        print(f"FAILED JOBS: {len(failed)}")
        raise SystemExit(1)
    print("All jobs completed successfully.")


if __name__ == "__main__":
    main()
