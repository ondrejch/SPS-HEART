#!/usr/bin/env python3
"""Gateway orchestrator for absolute-yield prior stiffness scans."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from helpers.na49_gateway.cluster_client import ClusterGatewayClient, _default_agent_command
from helpers.na49_gateway.gateway_config import get_config_list, get_config_str, load_user_config


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "00runs"
TERMINAL = {"COMPLETED", "FAILED", "CANCELED", "SUBMIT_FAILED"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_float_csv(raw: str) -> List[float]:
    out: List[float] = []
    for tok in str(raw).split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(float(t))
    if not out:
        raise ValueError("No float values provided.")
    return sorted(set(out))


def _remote_root(cfg: Dict[str, Any]) -> str:
    rel = get_config_str(cfg, "remote_project_relpath").strip()
    if not rel:
        raise ValueError("gateway config missing remote_project_relpath")
    if rel.startswith("~/"):
        return rel
    if rel.startswith("$HOME/"):
        return "~/" + rel[6:]
    return "~/" + rel


def _client(cfg: Dict[str, Any]) -> ClusterGatewayClient:
    return ClusterGatewayClient(
        host=get_config_str(cfg, "host"),
        agent_command=_default_agent_command(cfg),
        ssh_options=tuple(get_config_list(cfg, "ssh_options") or []),
        jobs_root=get_config_str(cfg, "jobs_root"),
    )


def _sync_remote(host: str, ssh_options: List[str], remote_root: str) -> None:
    sync_targets = [
        "helpers/benchmarking/benchmark_multispecies_constrained.py",
        "helpers/benchmarking/run_constrained_refit.py",
        "helpers/benchmarking/config/constrained_refit_table9_strange_absyield_defaults.json",
        "helpers/benchmarking/config/fischer2022_pdg_pi_minus_composition.json",
    ]
    ssh_cmd = " ".join(shlex.quote(x) for x in ["ssh", *ssh_options])
    for rel in sync_targets:
        src = REPO_ROOT / rel
        dst = f"{remote_root}/{rel}"
        subprocess.run(["ssh", *ssh_options, host, f"mkdir -p {shlex.quote(str(Path(dst).parent))}"], check=True)
        subprocess.run(
            ["rsync", "-az", "-e", ssh_cmd, str(src), f"{host}:{dst}"],
            check=True,
        )


def _scenario_name(mode: str, lam: float, sig: float, stamp: str) -> str:
    return f"constrained_refit_{mode}_absyield_scan_l{int(round(lam*1000)):04d}_s{int(round(sig*1000)):04d}_{stamp}"


def _cmd(
    *,
    remote_root: str,
    run_name: str,
    mode: str,
    lam: float,
    sig: float,
) -> str:
    py = f"{remote_root}/.venv-gw/bin/python"
    cfg = "helpers/benchmarking/config/constrained_refit_table9_strange_absyield_defaults.json"
    parts = [
        py,
        "helpers/benchmarking/run_constrained_refit.py",
        "--config-json",
        cfg,
        "--run-description",
        run_name,
        "--out-dir",
        f"00runs/{run_name}",
        "--res-cache-dir",
        "00runs/res_component_cache_shared_2026-03-31",
        "--set",
        f"prior_abs_yield_lambda={lam}",
        "--set",
        f"prior_abs_yield_min_sigma_rel={sig}",
    ]
    if mode == "pdg":
        parts.extend(
            [
                "--set",
                "prior_table9_json=helpers/benchmarking/config/fischer2022_pdg_pi_minus_composition.json",
            ]
        )
    # Keep a plain token-joined command string; per-token shell quoting here would
    # be interpreted literally by the gateway agent.
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", default="table9,pdg")
    parser.add_argument("--lambda-values", default="0.1,0.3,1.0")
    parser.add_argument("--sigma-values", default="0.02,0.05,0.1")
    parser.add_argument("--run-stamp", default="2026-03-31")
    parser.add_argument("--manifest-json", default="")
    parser.add_argument("--tasks-per-job", type=int, default=64)
    parser.add_argument("--max-tasks-per-worker", type=int, default=64)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--skip-remote-sync", action="store_true")
    parser.add_argument("--sync-back", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    cfg = load_user_config()
    workers = get_config_list(cfg, "workers") or []
    if not workers:
        raise ValueError("No workers configured.")
    max_tasks = int(args.max_tasks_per_worker)
    if max_tasks <= 0:
        raise ValueError("--max-tasks-per-worker must be > 0")
    host = get_config_str(cfg, "host")
    ssh_options = list(get_config_list(cfg, "ssh_options") or [])
    remote_root = _remote_root(cfg)
    client = _client(cfg)

    if not args.skip_remote_sync:
        _sync_remote(host, ssh_options, remote_root)

    modes = [x.strip() for x in str(args.modes).split(",") if x.strip()]
    lams = _parse_float_csv(args.lambda_values)
    sigs = _parse_float_csv(args.sigma_values)
    scenarios: List[Dict[str, Any]] = []
    for mode in modes:
        for lam in lams:
            for sig in sigs:
                run_name = _scenario_name(mode, lam, sig, args.run_stamp)
                scenarios.append(
                    {
                        "mode": mode,
                        "lambda": lam,
                        "sigma_rel": sig,
                        "run_name": run_name,
                        "remote_out_dir": f"{remote_root}/00runs/{run_name}",
                        "cmd": _cmd(
                            remote_root=remote_root,
                            run_name=run_name,
                            mode=mode,
                            lam=lam,
                            sig=sig,
                        ),
                    }
                )
    if bool(args.skip_existing):
        scenarios = [
            sc for sc in scenarios if not (RUNS_ROOT / sc["run_name"] / "multispecies_constrained_summary.json").exists()
        ]

    manifest_path = (
        Path(args.manifest_json).resolve()
        if str(args.manifest_json).strip()
        else (RUNS_ROOT / f"gw_absyield_prior_scan_{args.run_stamp}" / "gateway_absyield_prior_scan_manifest.json")
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    state: Dict[str, Any] = {
        "created_utc": _utc_now(),
        "workers": workers,
        "max_tasks_per_worker": max_tasks,
        "tasks_per_job": int(args.tasks_per_job),
        "scenarios": scenarios,
        "submitted": {},
        "final": {},
    }
    manifest_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    pending = list(scenarios)
    active: Dict[str, Dict[str, Any]] = {}
    while pending or active:
        # Refresh active states.
        for jid in list(active.keys()):
            view = client.status(jid)
            st = str(view.get("state", ""))
            if st in TERMINAL:
                state["final"][jid] = view
                active.pop(jid, None)

        # Submit as many as we can.
        cap = client.capacity(workers=workers, max_tasks_per_worker=max_tasks)
        free_by_worker = {str(r["worker"]): int(r.get("free_tasks", 0)) for r in cap}
        for worker in workers:
            if not pending:
                break
            if free_by_worker.get(worker, 0) < int(args.tasks_per_job):
                continue
            sc = pending.pop(0)
            try:
                job = client.submit(
                    sc["cmd"],
                    workdir=f"{remote_root}",
                    workers=workers,
                    worker=worker,
                    name=sc["run_name"],
                    tasks=int(args.tasks_per_job),
                    max_tasks_per_worker=max_tasks,
                    wait_for_slot=False,
                )
            except Exception:
                # Capacity can change between capacity() and submit(); retry in next poll cycle.
                pending.insert(0, sc)
                continue
            jid = str(job["job_id"])
            active[jid] = {"scenario": sc, "submitted_job": job}
            state["submitted"][jid] = {"scenario": sc, "job": job}
            free_by_worker[worker] = free_by_worker.get(worker, 0) - int(args.tasks_per_job)

        state["updated_utc"] = _utc_now()
        state["pending_count"] = len(pending)
        state["active_count"] = len(active)
        state["completed_count"] = len(state["final"])
        manifest_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

        if pending or active:
            time.sleep(max(float(args.poll_seconds), 2.0))

    # Optional sync-back.
    if bool(args.sync_back):
        for sc in scenarios:
            local_dir = RUNS_ROOT / sc["run_name"]
            local_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "rsync",
                    "-az",
                    f"{host}:{remote_root}/00runs/{sc['run_name']}/",
                    f"{local_dir}/",
                ],
                check=True,
            )

    state["finished_utc"] = _utc_now()
    manifest_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")
    print(f"scenarios: {len(scenarios)}")
    failed = sum(1 for v in state["final"].values() if str(v.get("state", "")) != "COMPLETED")
    print(f"failed: {failed}")


if __name__ == "__main__":
    main()
