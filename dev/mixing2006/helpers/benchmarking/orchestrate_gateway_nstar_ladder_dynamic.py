#!/usr/bin/env python3
"""Dynamic gateway scheduler for N* ladder grid with per-worker cache reuse."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from helpers.na49_gateway.cluster_client import ClusterGatewayClient, _default_agent_command
from helpers.na49_gateway.gateway_config import (
    get_config_int,
    get_config_list,
    get_config_str,
    load_user_config,
)


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


def _remote_root_from_config(config: Dict[str, Any]) -> str:
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


def _default_workers(config: Dict[str, Any]) -> List[str]:
    workers = get_config_list(config, "workers") or []
    if not workers:
        raise ValueError("No workers configured in gateway config.")
    return workers


def _client_from_config(config: Dict[str, Any]) -> ClusterGatewayClient:
    return ClusterGatewayClient(
        host=get_config_str(config, "host"),
        agent_command=_default_agent_command(config),
        ssh_options=tuple(get_config_list(config, "ssh_options") or []),
        jobs_root=get_config_str(config, "jobs_root"),
    )


def _sync_remote_code(host: str, remote_root: str, ssh_options: Tuple[str, ...]) -> Dict[str, Any]:
    if shutil.which("rsync") is None:
        raise RuntimeError("rsync is required for remote code sync but was not found on PATH.")

    ssh_cmd = " ".join(shlex.quote(tok) for tok in (["ssh", *ssh_options]))
    synced: List[str] = []
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


def _parse_float_csv(raw: str) -> List[float]:
    vals: List[float] = []
    for tok in str(raw).split(","):
        t = tok.strip()
        if not t:
            continue
        vals.append(float(t))
    if not vals:
        raise ValueError("No float values provided.")
    return sorted(set(vals))


def _parse_int_csv(raw: str) -> List[int]:
    vals: List[int] = []
    for tok in str(raw).split(","):
        t = tok.strip()
        if not t:
            continue
        vals.append(int(t))
    if not vals:
        raise ValueError("No int values provided.")
    return sorted(set(vals))


def _parse_shapes(raw: str) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for tok in str(raw).split(","):
        t = tok.strip()
        if not t:
            continue
        if ":" in t:
            name, param = t.split(":", 1)
            out.append((name.strip(), float(param)))
        else:
            out.append((t, 0.25))
    if not out:
        raise ValueError("No shape values provided.")
    dedup: List[Tuple[str, float]] = []
    seen = set()
    for item in out:
        key = (item[0], float(item[1]))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(item)
    return dedup


def _scenario_count(tower_max_values: str, level_counts: str, shapes: str) -> int:
    return len(_parse_float_csv(tower_max_values)) * len(_parse_int_csv(level_counts)) * len(_parse_shapes(shapes))


def _mk_chunk_cmd(
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
    prior_table9_lambda: float,
    prior_table9_json: str,
    prior_strange_lambda: float,
    prior_hyperon_lambda: float,
    prior_kminus_over_kplus_target: float,
    hyperon_pdgs: str,
    external_run_dir: str,
    skip_external: bool,
    scenario_indices_csv: str,
    chunk_count: int,
    chunk_index: int,
    worker_cache_tag: str,
) -> str:
    out_root = f"{remote_root}/00runs/{run_description}"
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
        "--prior-table9-lambda",
        f"{float(prior_table9_lambda):.12g}",
        "--prior-table9-json",
        str(prior_table9_json),
        "--prior-strange-lambda",
        f"{float(prior_strange_lambda):.12g}",
        "--prior-hyperon-lambda",
        f"{float(prior_hyperon_lambda):.12g}",
        "--prior-kminus-over-kplus-target",
        f"{float(prior_kminus_over_kplus_target):.12g}",
        "--hyperon-pdgs",
        str(hyperon_pdgs),
        "--external-run-dir",
        str(external_run_dir),
        "--scenario-indices",
        str(scenario_indices_csv),
        "--shard-count",
        str(int(chunk_count)),
        "--shard-index",
        str(int(chunk_index)),
        "--res-cache-dir",
        f"{out_root}/{worker_cache_tag}",
    ]
    if bool(skip_external):
        parts.append("--skip-external")
    return " ".join(parts)


def _write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic scheduler for N* ladder grid on gateway workers.")
    parser.add_argument("--run-description", default="gw_nstar_ladder_hi100x_u31_dynamic_2026-03-30")
    parser.add_argument("--species", default="pi-,pi+,k+,k-,p,pbar")
    parser.add_argument("--tower-min-mass", type=float, default=1.9)
    parser.add_argument("--tower-max-values", default="1.9,2.1,2.3,2.5,2.7,2.9,3.1")
    parser.add_argument("--level-counts", default="3,5,10,15")
    parser.add_argument("--shapes", default="linear,regge_m2,hagedorn:0.12,hagedorn:0.16,hagedorn:0.25")
    parser.add_argument("--seed", type=int, default=20260330)
    parser.add_argument("--res-events-per-res", type=int, default=1200000)
    parser.add_argument("--res-workers", type=int, default=64)
    parser.add_argument("--max-decay-passes", type=int, default=16)
    parser.add_argument("--fit-objective", default="nnls_weighted")
    parser.add_argument("--fit-weight-mode", default="sigma")
    parser.add_argument("--fit-opt-maxiter", type=int, default=3000)
    parser.add_argument("--fit-opt-ftol", type=float, default=1e-10)
    parser.add_argument("--fit-balance-xf-bins", type=int, default=4)
    parser.add_argument("--fit-balance-pt-bins", type=int, default=4)
    parser.add_argument("--prior-table9-lambda", type=float, default=0.0)
    parser.add_argument(
        "--prior-table9-json",
        default="helpers/benchmarking/config/fischer2022_table9_pi_minus.json",
    )
    parser.add_argument("--prior-strange-lambda", type=float, default=0.0)
    parser.add_argument("--prior-hyperon-lambda", type=float, default=0.0)
    parser.add_argument("--prior-kminus-over-kplus-target", type=float, default=0.5726872246696035)
    parser.add_argument("--hyperon-pdgs", default="3122,3212,3222,3112,3312,3322,3334")
    parser.add_argument(
        "--external-run-dir",
        default="00runs/gw_multispecies_allmodels_singlepass_pythia_2026-03-30",
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
    parser.add_argument("--chunk-size", type=int, default=1, help="Number of scenarios per submitted job.")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--skip-remote-sync", action="store_true")
    parser.add_argument("--max-retries", type=int, default=0, help="Automatic retries per failed chunk.")
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
    max_tasks_per_worker = (
        int(args.max_tasks_per_worker)
        if int(args.max_tasks_per_worker) > 0
        else get_config_int(config, "max_tasks_per_worker", 64)
    )
    chunk_size = max(int(args.chunk_size), 1)

    run_dir = RUNS_ROOT / args.run_description
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "gateway_nstar_ladder_dynamic_manifest.json"

    sync_info = None
    if not bool(args.skip_remote_sync):
        sync_info = _sync_remote_code(host, remote_root, tuple(get_config_list(config, "ssh_options") or []))

    total = _scenario_count(args.tower_max_values, args.level_counts, args.shapes)
    all_indices = list(range(total))
    chunks: List[List[int]] = [all_indices[i : i + chunk_size] for i in range(0, total, chunk_size)]

    manifest: Dict[str, Any] = {
        "run_description": str(args.run_description),
        "submitted_at": _utc_now(),
        "remote_root": remote_root,
        "python_bin": str(python_bin),
        "workers": workers,
        "tasks_per_job": int(args.tasks_per_job),
        "max_tasks_per_worker": int(max_tasks_per_worker),
        "chunk_size": int(chunk_size),
        "chunk_count": int(len(chunks)),
        "scenario_count": int(total),
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
        "priors": {
            "prior_table9_lambda": float(args.prior_table9_lambda),
            "prior_table9_json": str(args.prior_table9_json),
            "prior_strange_lambda": float(args.prior_strange_lambda),
            "prior_hyperon_lambda": float(args.prior_hyperon_lambda),
            "prior_kminus_over_kplus_target": float(args.prior_kminus_over_kplus_target),
            "hyperon_pdgs": str(args.hyperon_pdgs),
        },
        "external_run_dir": str(args.external_run_dir),
        "skip_external": bool(args.skip_external),
        "skip_remote_sync": bool(args.skip_remote_sync),
        "sync": sync_info,
        "chunk_attempts": [],
        "chunk_final": {},
        "events": [],
    }
    _write_manifest(manifest_path, manifest)

    client = _client_from_config(config)
    active_by_worker: Dict[str, Dict[str, Any]] = {}
    pending_chunk_ids = list(range(len(chunks)))
    retry_left = {idx: int(max(args.max_retries, 0)) for idx in range(len(chunks))}

    def submit_chunk(worker: str, chunk_id: int) -> None:
        scenario_indices = chunks[chunk_id]
        scenario_csv = ",".join(str(x) for x in scenario_indices)
        cmd = _mk_chunk_cmd(
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
            prior_table9_lambda=float(args.prior_table9_lambda),
            prior_table9_json=str(args.prior_table9_json),
            prior_strange_lambda=float(args.prior_strange_lambda),
            prior_hyperon_lambda=float(args.prior_hyperon_lambda),
            prior_kminus_over_kplus_target=float(args.prior_kminus_over_kplus_target),
            hyperon_pdgs=str(args.hyperon_pdgs),
            external_run_dir=str(args.external_run_dir),
            skip_external=bool(args.skip_external),
            scenario_indices_csv=scenario_csv,
            chunk_count=len(chunks),
            chunk_index=chunk_id,
            worker_cache_tag=f"res_cache_worker_{worker}",
        )
        job = client.submit(
            command=cmd,
            workdir=remote_root,
            workers=workers,
            worker=worker,
            name=f"{args.run_description}_chunk{chunk_id:04d}",
            tasks=max(int(args.tasks_per_job), 1),
            max_tasks_per_worker=max(int(max_tasks_per_worker), 1),
            wait_for_slot=True,
            poll_seconds=float(args.poll_seconds),
        )
        rec = {
            "chunk_id": int(chunk_id),
            "worker": str(worker),
            "scenarios": scenario_indices,
            "command": cmd,
            "job": job,
            "submitted_at": _utc_now(),
        }
        manifest["chunk_attempts"].append(rec)
        manifest["events"].append(
            {
                "ts": _utc_now(),
                "event": "submitted",
                "worker": str(worker),
                "chunk_id": int(chunk_id),
                "job_id": str(job.get("job_id")),
            }
        )
        active_by_worker[worker] = {
            "chunk_id": int(chunk_id),
            "job_id": str(job.get("job_id")),
            "submitted_at": _utc_now(),
        }
        print(
            f"submit worker={worker} chunk={chunk_id}/{len(chunks)} "
            f"scenario_indices={scenario_csv} job_id={job.get('job_id')}",
            flush=True,
        )
        _write_manifest(manifest_path, manifest)

    # initial fill
    for worker in workers:
        if not pending_chunk_ids:
            break
        chunk_id = pending_chunk_ids.pop(0)
        submit_chunk(worker, chunk_id)

    while active_by_worker or pending_chunk_ids:
        time.sleep(max(float(args.poll_seconds), 1.0))
        for worker in list(active_by_worker.keys()):
            job_id = active_by_worker[worker]["job_id"]
            chunk_id = int(active_by_worker[worker]["chunk_id"])
            view = client.status(job_id)
            state = str(view.get("state", ""))
            if state not in TERMINAL_STATES:
                continue
            manifest["events"].append(
                {
                    "ts": _utc_now(),
                    "event": "finished",
                    "worker": str(worker),
                    "chunk_id": int(chunk_id),
                    "job_id": str(job_id),
                    "state": state,
                    "exit_code": view.get("exit_code"),
                }
            )
            if state == "COMPLETED":
                manifest["chunk_final"][str(chunk_id)] = {
                    "status": "COMPLETED",
                    "job_id": str(job_id),
                    "worker": str(worker),
                    "finished_at": _utc_now(),
                }
            else:
                if retry_left.get(chunk_id, 0) > 0:
                    retry_left[chunk_id] = int(retry_left[chunk_id] - 1)
                    pending_chunk_ids.append(chunk_id)
                    print(
                        f"chunk {chunk_id} failed on {worker} (state={state}); "
                        + f"requeue with retries_left={retry_left[chunk_id]}",
                        flush=True,
                    )
                    manifest["events"].append(
                        {
                            "ts": _utc_now(),
                            "event": "requeued",
                            "chunk_id": int(chunk_id),
                            "retries_left": int(retry_left[chunk_id]),
                        }
                    )
                else:
                    manifest["chunk_final"][str(chunk_id)] = {
                        "status": str(state),
                        "job_id": str(job_id),
                        "worker": str(worker),
                        "finished_at": _utc_now(),
                    }
            del active_by_worker[worker]
            _write_manifest(manifest_path, manifest)

        # fill idle workers
        for worker in workers:
            if worker in active_by_worker:
                continue
            if not pending_chunk_ids:
                continue
            chunk_id = pending_chunk_ids.pop(0)
            submit_chunk(worker, chunk_id)

        done = sum(1 for x in manifest["chunk_final"].values() if x.get("status") == "COMPLETED")
        failed = sum(1 for x in manifest["chunk_final"].values() if x.get("status") != "COMPLETED")
        print(
            f"progress completed={done}/{len(chunks)} failed={failed} "
            + f"active={len(active_by_worker)} pending={len(pending_chunk_ids)}",
            flush=True,
        )

    manifest["completed_at"] = _utc_now()
    final_completed = sum(1 for x in manifest["chunk_final"].values() if x.get("status") == "COMPLETED")
    final_failed = sum(1 for x in manifest["chunk_final"].values() if x.get("status") != "COMPLETED")
    manifest["summary"] = {
        "chunk_count": int(len(chunks)),
        "completed_chunks": int(final_completed),
        "failed_chunks": int(final_failed),
    }
    _write_manifest(manifest_path, manifest)
    print(
        f"done: chunks completed={final_completed}/{len(chunks)} failed={final_failed}; "
        + f"manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
