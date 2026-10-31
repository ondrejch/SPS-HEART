#!/usr/bin/env python3
"""Submit, monitor, merge, and score the SPS multi-species gateway campaign."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from helpers.benchmarking.particle_definitions import canonical_species_keys, resolve_species
from helpers.na49_gateway.cluster_client import ClusterGatewayClient, _default_agent_command
from helpers.na49_gateway.gateway_config import get_config_int, get_config_list, get_config_str, load_user_config

TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELED", "SUBMIT_FAILED"}
REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "00runs"
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


def _remote_run_dir(remote_root: str, run_description: str) -> str:
    return f"{remote_root}/00runs/{run_description}"


def _urqmd_remote_output(remote_run_dir: str, species_tag: str, chunk: int) -> str:
    return f"{remote_run_dir}/urqmd_chunk_{chunk:02d}_{species_tag}.csv"


def _epos_remote_output(remote_run_dir: str, species_tag: str, chunk: int) -> str:
    return f"{remote_run_dir}/epos_chunk_{chunk:02d}_{species_tag}.csv"


def _pythia_remote_output(remote_run_dir: str, species_tag: str, chunk: int) -> str:
    return f"{remote_run_dir}/pythia_chunk_{chunk:02d}_{species_tag}.csv"


def _urqmd_remote_merged(remote_run_dir: str, species_tag: str) -> str:
    return f"{remote_run_dir}/urqmd_pp158_{species_tag}_gateway.csv"


def _epos_remote_merged(remote_run_dir: str, species_tag: str) -> str:
    return f"{remote_run_dir}/epos_lhc_pp158_{species_tag}_gateway.csv"


def _pythia_remote_merged(remote_run_dir: str, species_tag: str) -> str:
    return f"{remote_run_dir}/pythia8_softqcd_pp158_{species_tag}_gateway.csv"


def _urqmd_local_merged(local_run_dir: Path, species_tag: str) -> Path:
    return local_run_dir / f"urqmd_pp158_{species_tag}_gateway.csv"


def _epos_local_merged(local_run_dir: Path, species_tag: str) -> Path:
    return local_run_dir / f"epos_lhc_pp158_{species_tag}_gateway.csv"


def _pythia_local_merged(local_run_dir: Path, species_tag: str) -> Path:
    return local_run_dir / f"pythia8_softqcd_pp158_{species_tag}_gateway.csv"


def _submit_jobs(
    client: ClusterGatewayClient,
    *,
    remote_root: str,
    remote_run_dir: str,
    python_bin: str,
    workers: list[str],
    max_tasks_per_worker: int,
    chunks: int,
    events_per_chunk: int,
    beam_momentum: float,
    target_momentum: float,
    urqmd_seed_base: int,
    epos_seed_base: int,
    pythia_seed_base: int,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    submitted = 0

    for chunk in range(chunks):
        urqmd_seed = urqmd_seed_base + chunk
        urqmd_name = f"urqmd_chunk_{chunk:02d}"
        urqmd_tag = f"urqmd_chunk_{chunk:02d}"
        urqmd_out_template = f"{remote_run_dir}/urqmd_chunk_{chunk:02d}_{{species_tag}}.csv"
        urqmd_cmd = (
            f"{python_bin} {remote_root}/helpers/benchmarking/run_urqmd_to_csv.py "
            f"--species all --events {events_per_chunk} --seed {urqmd_seed} "
            f"--beam-momentum {beam_momentum} --target-momentum {target_momentum} "
            f"--urqmd-dir {remote_root}/external/urqmd-4.0 "
            f"--out-csv-template {urqmd_out_template} --tag {urqmd_tag}"
        )
        urqmd_job = client.submit(
            urqmd_cmd,
            workdir=remote_root,
            workers=workers,
            name=urqmd_name,
            tasks=1,
            max_tasks_per_worker=max_tasks_per_worker,
            wait_for_slot=True,
            poll_seconds=2.0,
        )
        jobs.append(
            {
                "job_id": urqmd_job["job_id"],
                "model": "urqmd",
                "chunk": chunk,
                "seed": urqmd_seed,
                "name": urqmd_name,
                "output_template": urqmd_out_template,
            }
        )
        submitted += 1
        print(f"[{submitted}] submitted {urqmd_name} -> {urqmd_job['job_id']}", flush=True)

        epos_seed = epos_seed_base + chunk
        epos_name = f"epos_chunk_{chunk:02d}"
        epos_tag = f"epos_chunk_{chunk:02d}"
        epos_out_template = f"{remote_run_dir}/epos_chunk_{chunk:02d}_{{species_tag}}.csv"
        epos_cmd = (
            f"{python_bin} {remote_root}/helpers/benchmarking/run_epos_crmc_to_csv.py "
            f"--species all --events {events_per_chunk} --seed {epos_seed} "
            f"--projectile-momentum {beam_momentum} --target-momentum {target_momentum} "
            f"--crmc-install {remote_root}/external/crmc/install "
            f"--hepmc-lib {remote_root}/external/hepmc2-install/lib "
            f"--out-csv-template {epos_out_template} --tag {epos_tag}"
        )
        epos_job = client.submit(
            epos_cmd,
            workdir=remote_root,
            workers=workers,
            name=epos_name,
            tasks=1,
            max_tasks_per_worker=max_tasks_per_worker,
            wait_for_slot=True,
            poll_seconds=2.0,
        )
        jobs.append(
            {
                "job_id": epos_job["job_id"],
                "model": "epos",
                "chunk": chunk,
                "seed": epos_seed,
                "name": epos_name,
                "output_template": epos_out_template,
            }
        )
        submitted += 1
        print(f"[{submitted}] submitted {epos_name} -> {epos_job['job_id']}", flush=True)

        pythia_seed = pythia_seed_base + chunk
        pythia_name = f"pythia_chunk_{chunk:02d}"
        pythia_tag = f"pythia_chunk_{chunk:02d}"
        pythia_out_template = f"{remote_run_dir}/pythia_chunk_{chunk:02d}_{{species_tag}}.csv"
        pythia_cmd = (
            f"{python_bin} {remote_root}/helpers/benchmarking/run_pythia8_to_csv.py "
            f"--species all --events {events_per_chunk} --seed {pythia_seed} "
            f"--beam-momentum {beam_momentum} --target-momentum {target_momentum} "
            f"--out-csv-template {pythia_out_template} --tag {pythia_tag}"
        )
        pythia_job = client.submit(
            pythia_cmd,
            workdir=remote_root,
            workers=workers,
            name=pythia_name,
            tasks=1,
            max_tasks_per_worker=max_tasks_per_worker,
            wait_for_slot=True,
            poll_seconds=2.0,
        )
        jobs.append(
            {
                "job_id": pythia_job["job_id"],
                "model": "pythia",
                "chunk": chunk,
                "seed": pythia_seed,
                "name": pythia_name,
                "output_template": pythia_out_template,
            }
        )
        submitted += 1
        print(f"[{submitted}] submitted {pythia_name} -> {pythia_job['job_id']}", flush=True)

    return jobs


def _wait_for_jobs(client: ClusterGatewayClient, jobs: list[dict[str, Any]], poll_seconds: float) -> dict[str, dict[str, Any]]:
    pending = {j["job_id"] for j in jobs}
    final: dict[str, dict[str, Any]] = {}

    while pending:
        listed = {
            item["job_id"]: item
            for item in client.list(limit=max(5000, len(jobs) + 200))
        }
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


def _ssh_python(host: str, script: str) -> str:
    proc = subprocess.run(["ssh", host, "python3 -"], input=script, text=True, capture_output=True, check=True)
    return proc.stdout


def _merge_remote_csv(host: str, pattern: str, output_csv: str) -> dict[str, Any]:
    script = f"""
import csv
import glob
import os

pattern = os.path.expanduser({pattern!r})
out = os.path.expanduser({output_csv!r})
parts = sorted(glob.glob(pattern))
if not parts:
    raise SystemExit(f"No files for pattern: {{pattern}}")
rows = 0
with open(out, "w", newline="") as f_out:
    writer = csv.writer(f_out)
    writer.writerow(["xf", "pt", "value"])
    for p in parts:
        with open(p, "r", newline="") as f_in:
            reader = csv.DictReader(f_in)
            for row in reader:
                writer.writerow([row["xf"], row["pt"], row["value"]])
                rows += 1
print(parts[0])
print(len(parts))
print(rows)
print(out)
"""
    out = _ssh_python(host, script).strip().splitlines()
    return {
        "first_part": out[0],
        "num_parts": int(out[1]),
        "rows": int(out[2]),
        "output_csv": out[3],
    }


def _scp_from_remote(host: str, remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["scp", f"{host}:{remote_path}", str(local_path)], check=True)


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


def _run_local_species_benchmark(
    species_key: str,
    local_run_dir: Path,
    pythia_csv: Path,
    epos_csv: Path,
    urqmd_csv: Path,
    beam_momentum: float,
    pythia_events: int,
    res_events_per_res: int,
    res_max_decay_passes: int,
    fit_weight_mode: str,
    fit_balance_xf_bins: int,
    fit_balance_pt_bins: int,
    disable_resonance_score: bool,
) -> tuple[Path, Path]:
    spec = resolve_species(species_key)
    out_prefix = f"na49_benchmark_{spec.tag}_gateway"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "helpers" / "benchmarking" / "benchmark_sps_species.py"),
        "--species",
        species_key,
        "--beam-momentum",
        str(beam_momentum),
        "--xf-min",
        "0.0",
        "--xf-max",
        "0.4",
        "--pt-min",
        "0.05",
        "--pt-max",
        "1.5",
        "--out-dir",
        str(local_run_dir),
        "--out-prefix",
        out_prefix,
        "--pythia-events",
        str(pythia_events),
        "--pythia-csv",
        str(pythia_csv),
        "--res-events-per-res",
        str(res_events_per_res),
        "--res-max-decay-passes",
        str(res_max_decay_passes),
        "--fit-weight-mode",
        str(fit_weight_mode),
        "--fit-balance-xf-bins",
        str(int(fit_balance_xf_bins)),
        "--fit-balance-pt-bins",
        str(int(fit_balance_pt_bins)),
        "--epos-csv",
        str(epos_csv),
        "--urqmd-csv",
        str(urqmd_csv),
    ]
    if disable_resonance_score:
        cmd.append("--disable-resonance")
    subprocess.run(cmd, check=True)
    return (
        local_run_dir / f"{out_prefix}_summary.json",
        local_run_dir / f"{out_prefix}_details.npz",
    )


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full multi-species gateway campaign and local scoring.")
    parser.add_argument("--run-description", default="gateway_multispecies_default")
    parser.add_argument("--species", default=",".join(canonical_species_keys()), help="CSV of species keys")
    parser.add_argument("--chunks", type=int, default=32, help="Chunks per model per species")
    parser.add_argument("--events-per-chunk", type=int, default=5000)
    parser.add_argument("--beam-momentum", type=float, default=158.0)
    parser.add_argument("--target-momentum", type=float, default=0.0)
    parser.add_argument("--urqmd-seed-base", type=int, default=410000)
    parser.add_argument("--epos-seed-base", type=int, default=510000)
    parser.add_argument("--pythia-seed-base", type=int, default=610000)
    parser.add_argument(
        "--python-bin",
        default="",
        help="Remote Python executable for gateway jobs. Defaults to <remote_root>/.venv-gw/bin/python.",
    )
    parser.add_argument("--pythia-events", type=int, default=120000)
    parser.add_argument("--res-events-per-res", type=int, default=30000)
    parser.add_argument("--res-max-decay-passes", type=int, default=8)
    parser.add_argument(
        "--fit-weight-mode",
        default="relative_xfpt_balanced",
        help="Forwarded to benchmark_sps_species.py for scoring normalization/fits.",
    )
    parser.add_argument("--fit-balance-xf-bins", type=int, default=4)
    parser.add_argument("--fit-balance-pt-bins", type=int, default=4)
    parser.add_argument(
        "--disable-resonance-score",
        action="store_true",
        help="Disable resonance model in per-species scoring pass (external-model comparison only).",
    )
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--workers", default="", help="Override worker CSV")
    parser.add_argument("--manifest", default="", help="Path to manifest JSON. Defaults to 00runs/<run_description>/multispecies_gateway_jobs.json")
    parser.add_argument(
        "--skip-remote-sync",
        action="store_true",
        help="Skip pre-submit rsync of local helpers/benchmarking and mixing2006 code to remote repo copy.",
    )
    parser.add_argument("--skip-submit", action="store_true")
    parser.add_argument("--skip-wait", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-score", action="store_true")
    args = parser.parse_args()

    species_keys = [s.strip() for s in args.species.split(",") if s.strip()]
    for key in species_keys:
        resolve_species(key)

    config = load_user_config()
    remote_root = _remote_root_from_config(config)
    remote_run_dir = _remote_run_dir(remote_root, args.run_description)
    python_bin = str(args.python_bin).strip() if str(args.python_bin).strip() else f"{remote_root}/.venv-gw/bin/python"
    local_run_dir = RUNS_ROOT / args.run_description
    host = get_config_str(config, "host")
    if not host:
        raise ValueError("Gateway config must define 'host'.")
    max_tasks_per_worker = get_config_int(config, "max_tasks_per_worker", 64) or 64
    workers = [w.strip() for w in args.workers.split(",") if w.strip()] if args.workers else _default_workers(config)
    ssh_options = tuple(get_config_list(config, "ssh_options") or [])

    manifest_path = Path(args.manifest) if args.manifest else (local_run_dir / "multispecies_gateway_jobs.json")
    client = _client_from_config(config)
    sync_info: dict[str, Any] = {}

    if args.skip_submit:
        manifest = _load_manifest(manifest_path)
        jobs = manifest["jobs"]
    else:
        if args.skip_remote_sync:
            print("sync: skipped by --skip-remote-sync", flush=True)
            sync_info = {"synced_at": "", "targets": [], "status": "skipped"}
        else:
            sync_info = _sync_remote_code(host=host, remote_root=remote_root, ssh_options=ssh_options)

        jobs = _submit_jobs(
            client,
            remote_root=remote_root,
            remote_run_dir=remote_run_dir,
            python_bin=python_bin,
            workers=workers,
            max_tasks_per_worker=max_tasks_per_worker,
            chunks=args.chunks,
            events_per_chunk=args.events_per_chunk,
            beam_momentum=args.beam_momentum,
            target_momentum=args.target_momentum,
            urqmd_seed_base=args.urqmd_seed_base,
            epos_seed_base=args.epos_seed_base,
            pythia_seed_base=args.pythia_seed_base,
        )
        manifest = {
            "created_at": _utc_now(),
            "host": host,
            "remote_root": remote_root,
            "remote_run_dir": remote_run_dir,
            "local_run_dir": str(local_run_dir),
            "run_description": args.run_description,
            "workers": workers,
            "max_tasks_per_worker": max_tasks_per_worker,
            "species": species_keys,
            "chunks": args.chunks,
            "events_per_chunk": args.events_per_chunk,
            "beam_momentum": args.beam_momentum,
            "target_momentum": args.target_momentum,
            "python_bin": python_bin,
            "urqmd_seed_base": args.urqmd_seed_base,
            "epos_seed_base": args.epos_seed_base,
            "pythia_seed_base": args.pythia_seed_base,
            "pythia_events": args.pythia_events,
            "res_events_per_res": args.res_events_per_res,
            "res_max_decay_passes": args.res_max_decay_passes,
            "fit_weight_mode": args.fit_weight_mode,
            "fit_balance_xf_bins": args.fit_balance_xf_bins,
            "fit_balance_pt_bins": args.fit_balance_pt_bins,
            "disable_resonance_score": bool(args.disable_resonance_score),
            "remote_code_sync": sync_info,
            "jobs": jobs,
        }
        _write_manifest(manifest_path, manifest)
        print(f"manifest written: {manifest_path}")

    if not args.skip_wait:
        final_states = _wait_for_jobs(client, jobs, poll_seconds=args.poll_seconds)
        manifest["completed_at"] = _utc_now()
        manifest["job_states"] = {jid: final_states[jid].get("state", "") for jid in final_states}
        manifest["failed_jobs"] = [jid for jid, view in final_states.items() if view.get("state") != "COMPLETED"]
        _write_manifest(manifest_path, manifest)
        print(f"wait completed; failures={len(manifest['failed_jobs'])}")

    merged: dict[str, dict[str, str]] = {}
    if not args.skip_merge:
        for species_key in species_keys:
            spec = resolve_species(species_key)
            urqmd_pattern = f"{remote_run_dir}/urqmd_chunk_*_{spec.tag}.csv"
            urqmd_remote = _urqmd_remote_merged(remote_run_dir, spec.tag)
            urqmd_info = _merge_remote_csv(host, urqmd_pattern, urqmd_remote)
            _scp_from_remote(host, urqmd_remote, _urqmd_local_merged(local_run_dir, spec.tag))

            epos_pattern = f"{remote_run_dir}/epos_chunk_*_{spec.tag}.csv"
            epos_remote = _epos_remote_merged(remote_run_dir, spec.tag)
            epos_info = _merge_remote_csv(host, epos_pattern, epos_remote)
            _scp_from_remote(host, epos_remote, _epos_local_merged(local_run_dir, spec.tag))

            pythia_pattern = f"{remote_run_dir}/pythia_chunk_*_{spec.tag}.csv"
            pythia_remote = _pythia_remote_merged(remote_run_dir, spec.tag)
            pythia_info = _merge_remote_csv(host, pythia_pattern, pythia_remote)
            _scp_from_remote(host, pythia_remote, _pythia_local_merged(local_run_dir, spec.tag))

            merged[species_key] = {
                "urqmd_remote": urqmd_remote,
                "urqmd_local": str(_urqmd_local_merged(local_run_dir, spec.tag)),
                "urqmd_rows": str(urqmd_info["rows"]),
                "epos_remote": epos_remote,
                "epos_local": str(_epos_local_merged(local_run_dir, spec.tag)),
                "epos_rows": str(epos_info["rows"]),
                "pythia_remote": pythia_remote,
                "pythia_local": str(_pythia_local_merged(local_run_dir, spec.tag)),
                "pythia_rows": str(pythia_info["rows"]),
            }
            print(
                f"merged {species_key}: urqmd_rows={urqmd_info['rows']} epos_rows={epos_info['rows']} "
                f"pythia_rows={pythia_info['rows']}",
                flush=True,
            )

        manifest["merged_outputs"] = merged
        _write_manifest(manifest_path, manifest)

    if not args.skip_score:
        summaries: dict[str, str] = {}
        for species_key in species_keys:
            spec = resolve_species(species_key)
            pythia_csv = _pythia_local_merged(local_run_dir, spec.tag)
            epos_csv = _epos_local_merged(local_run_dir, spec.tag)
            urqmd_csv = _urqmd_local_merged(local_run_dir, spec.tag)
            if not pythia_csv.exists() or not epos_csv.exists() or not urqmd_csv.exists():
                raise FileNotFoundError(
                    f"Missing merged CSVs for {species_key}: {pythia_csv}, {epos_csv}, {urqmd_csv}"
                )
            summary_path, _ = _run_local_species_benchmark(
                species_key,
                local_run_dir,
                pythia_csv,
                epos_csv,
                urqmd_csv,
                args.beam_momentum,
                args.pythia_events,
                args.res_events_per_res,
                args.res_max_decay_passes,
                args.fit_weight_mode,
                args.fit_balance_xf_bins,
                args.fit_balance_pt_bins,
                bool(args.disable_resonance_score),
            )
            summaries[species_key] = str(summary_path)
            print(f"scored {species_key}: {summary_path}", flush=True)

        manifest["benchmark_summaries"] = summaries
        manifest["scored_at"] = _utc_now()
        _write_manifest(manifest_path, manifest)

    print("campaign complete")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
