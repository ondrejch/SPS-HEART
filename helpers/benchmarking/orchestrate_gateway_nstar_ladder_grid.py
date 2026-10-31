#!/usr/bin/env python3
"""Submit the N* ladder grid scan to the gateway as worker-pinned shards."""

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
RUNS_ROOT = REPO_ROOT / "dev" / "00runs"
TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELED", "SUBMIT_FAILED"}
SYNC_TARGETS: tuple[str, ...] = (
    "helpers/benchmarking",
    "dev/mixing2006/na49_mixing",
    "dev/mixing2006/pyproject.toml",
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


def _mk_shard_cmd(
    *,
    python_bin: str,
    remote_root: str,
    run_description: str,
    species: str,
    tower_min_mass: float,
    tower_max_values: str,
    level_counts: str,
    shapes: str,
    seed: int,
    res_events_per_res: int,
    res_workers: int,
    max_decay_passes: int,
    fit_objective: str,
    fit_weight_mode: str,
    fit_opt_maxiter: int,
    fit_opt_ftol: float,
    fit_balance_xf_bins: int,
    fit_balance_pt_bins: int,
    external_run_dir: str,
    shard_count: int,
    shard_index: int,
    res_cache_tag: str,
    skip_external: bool,
) -> str:
    out_root = f"{remote_root}/dev/00runs/{run_description}"
    parts = [
        python_bin,
        f"{remote_root}/helpers/benchmarking/run_nstar_ladder_grid.py",
        "--run-description",
        run_description,
        "--out-root",
        out_root,
        "--species",
        species,
        "--tower-min-mass",
        f"{float(tower_min_mass):.6f}",
        "--tower-max-values",
        str(tower_max_values),
        "--level-counts",
        str(level_counts),
        "--shapes",
        str(shapes),
        "--seed",
        str(int(seed)),
        "--res-events-per-res",
        str(int(res_events_per_res)),
        "--res-workers",
        str(int(res_workers)),
        "--max-decay-passes",
        str(int(max_decay_passes)),
        "--fit-objective",
        str(fit_objective),
        "--fit-weight-mode",
        str(fit_weight_mode),
        "--fit-opt-maxiter",
        str(int(fit_opt_maxiter)),
        "--fit-opt-ftol",
        f"{float(fit_opt_ftol):.12g}",
        "--fit-balance-xf-bins",
        str(int(fit_balance_xf_bins)),
        "--fit-balance-pt-bins",
        str(int(fit_balance_pt_bins)),
        "--external-run-dir",
        str(external_run_dir),
        "--shard-count",
        str(int(shard_count)),
        "--shard-index",
        str(int(shard_index)),
        "--res-cache-dir",
        f"{out_root}/{res_cache_tag}",
    ]
    if bool(skip_external):
        parts.append("--skip-external")
    return " ".join(parts)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit N* ladder grid shards on the gateway.")
    parser.add_argument("--run-description", default="gw_nstar_ladder_grid_hi100x_2026-03-30")
    parser.add_argument("--species", default="pi-,pi+,k+,k-,p,pbar")
    parser.add_argument("--tower-min-mass", type=float, default=1.9)
    parser.add_argument("--tower-max-values", default="1.9,2.1,2.3,2.5,2.7")
    parser.add_argument("--level-counts", default="3,5,10,15")
    parser.add_argument("--shapes", default="linear,regge_m2,hagedorn:0.12,hagedorn:0.16,hagedorn:0.25")
    parser.add_argument("--seed", type=int, default=20260330)
    parser.add_argument("--res-events-per-res", type=int, default=1200000)
    parser.add_argument("--res-workers", type=int, default=64)
    parser.add_argument("--max-decay-passes", type=int, default=16)
    parser.add_argument("--fit-objective", default="nnls_weighted")
    parser.add_argument("--fit-weight-mode", default="relative_xfpt_balanced")
    parser.add_argument("--fit-opt-maxiter", type=int, default=3000)
    parser.add_argument("--fit-opt-ftol", type=float, default=1e-10)
    parser.add_argument("--fit-balance-xf-bins", type=int, default=4)
    parser.add_argument("--fit-balance-pt-bins", type=int, default=4)
    parser.add_argument(
        "--external-run-dir",
        default="dev/00runs/gw_multispecies_allmodels_singlepass_pythia_2026-03-30",
    )
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--workers", default="")
    parser.add_argument("--tasks-per-job", type=int, default=64)
    parser.add_argument("--max-tasks-per-worker", type=int, default=64)
    parser.add_argument(
        "--python-bin",
        default="__AUTO__",
        help="Remote Python path. '__AUTO__' resolves to <remote_root>/.venv-gw/bin/python.",
    )
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--skip-remote-sync", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument(
        "--res-cache-scope",
        choices=("shared", "per_shard"),
        default="per_shard",
        help=(
            "Cache directory policy. 'per_shard' avoids cross-shard cache-file races and is the default."
        ),
    )
    args = parser.parse_args()

    config = load_user_config()
    host = get_config_str(config, "host")
    remote_root = _remote_root_from_config(config)
    python_bin = (
        f"{remote_root}/.venv-gw/bin/python"
        if str(args.python_bin).strip() == "__AUTO__"
        else str(args.python_bin)
    )
    workers = (
        [w.strip() for w in str(args.workers).split(",") if w.strip()]
        if str(args.workers).strip()
        else _default_workers(config)
    )
    max_tasks_per_worker = int(args.max_tasks_per_worker) if int(args.max_tasks_per_worker) > 0 else get_config_int(config, "max_tasks_per_worker", 64)

    run_dir = RUNS_ROOT / args.run_description
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "gateway_nstar_ladder_grid_manifest.json"

    sync_info = None
    if not bool(args.skip_remote_sync):
        sync_info = _sync_remote_code(host, remote_root, tuple(get_config_list(config, "ssh_options") or []))

    client = _client_from_config(config)
    shard_count = len(workers)
    submissions: list[dict[str, Any]] = []

    for shard_index, worker in enumerate(workers):
        res_cache_tag = (
            "res_cache"
            if str(args.res_cache_scope) == "shared"
            else f"res_cache_shard{int(shard_index):02d}"
        )
        cmd = _mk_shard_cmd(
            python_bin=str(python_bin),
            remote_root=remote_root,
            run_description=str(args.run_description),
            species=str(args.species),
            tower_min_mass=float(args.tower_min_mass),
            tower_max_values=str(args.tower_max_values),
            level_counts=str(args.level_counts),
            shapes=str(args.shapes),
            seed=int(args.seed),
            res_events_per_res=int(args.res_events_per_res),
            res_workers=int(args.res_workers),
            max_decay_passes=int(args.max_decay_passes),
            fit_objective=str(args.fit_objective),
            fit_weight_mode=str(args.fit_weight_mode),
            fit_opt_maxiter=int(args.fit_opt_maxiter),
            fit_opt_ftol=float(args.fit_opt_ftol),
            fit_balance_xf_bins=int(args.fit_balance_xf_bins),
            fit_balance_pt_bins=int(args.fit_balance_pt_bins),
            external_run_dir=str(args.external_run_dir),
            shard_count=shard_count,
            shard_index=shard_index,
            res_cache_tag=str(res_cache_tag),
            skip_external=bool(args.skip_external),
        )
        job = client.submit(
            command=cmd,
            workdir=remote_root,
            workers=workers,
            worker=worker,
            name=f"{args.run_description}_sh{shard_index:02d}",
            tasks=max(int(args.tasks_per_job), 1),
            max_tasks_per_worker=max(max_tasks_per_worker, 1),
            wait_for_slot=True,
            poll_seconds=float(args.poll_seconds),
        )
        submissions.append(
            {
                "shard_index": shard_index,
                "worker": worker,
                "command": cmd,
                "job": job,
                "submitted_at": _utc_now(),
            }
        )
        print(
            f"submitted shard={shard_index}/{shard_count} worker={worker} job_id={job.get('job_id')} state={job.get('state')}",
            flush=True,
        )

    manifest: dict[str, Any] = {
        "run_description": str(args.run_description),
        "submitted_at": _utc_now(),
        "remote_root": remote_root,
        "workers": workers,
        "tasks_per_job": int(args.tasks_per_job),
        "max_tasks_per_worker": int(max_tasks_per_worker),
        "python_bin": str(python_bin),
        "res_events_per_res": int(args.res_events_per_res),
        "res_workers": int(args.res_workers),
        "scan": {
            "tower_min_mass": float(args.tower_min_mass),
            "tower_max_values": str(args.tower_max_values),
            "level_counts": str(args.level_counts),
            "shapes": str(args.shapes),
        },
        "fit": {
            "fit_objective": str(args.fit_objective),
            "fit_weight_mode": str(args.fit_weight_mode),
            "fit_opt_maxiter": int(args.fit_opt_maxiter),
            "fit_opt_ftol": float(args.fit_opt_ftol),
            "fit_balance_xf_bins": int(args.fit_balance_xf_bins),
            "fit_balance_pt_bins": int(args.fit_balance_pt_bins),
        },
        "external_run_dir": str(args.external_run_dir),
        "skip_external": bool(args.skip_external),
        "res_cache_scope": str(args.res_cache_scope),
        "skip_remote_sync": bool(args.skip_remote_sync),
        "sync": sync_info,
        "jobs": submissions,
    }

    if bool(args.wait):
        final = _wait_for_jobs(client, [x["job"] for x in submissions], float(args.poll_seconds))
        manifest["wait_completed_at"] = _utc_now()
        manifest["final_status"] = final

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
