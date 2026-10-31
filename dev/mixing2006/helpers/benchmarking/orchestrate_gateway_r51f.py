"""Run the R51f joint fit on the cluster gateway (card
TASK-20260930-r5-gate5-followup-r51f-joint-proton-pion-fit).

Every driver step runs on the gateway, in the remote checkout, under
REMOTE_PYTHON: the pre-registration (1 task), generation (one job per slot,
1 task each, all open branches at once), evaluation (one job per floor
branch, EVAL_TASKS tasks, branches concurrently), the profile, gate, clean
validation and seal.  The run dir lives on the cluster, so every record's
run_dir identity and every digest is computed there; the sealed dir is
fetched back into the local artifact store at the end.

The orchestrator is resumable: it reads the remote run dir before every
phase and keeps a local ledger of submitted jobs, so a restart never
duplicates a job whose output exists or whose job is still alive.  A LOST
verdict (the shared-filesystem heartbeat stalls of R44/R45) triggers a
liveness probe of the job's wrapper process on its worker: a live wrapper
is waited for, and a job is resubmitted only when it FAILED, or when it is
LOST, its wrapper is gone, and its output is missing.  (The first R51f
evaluation round resubmitted a live LOST job after a fixed grace time; the
duplicate was killed before it wrote anything, 2026-09-30 13:57.)

Usage: python3 helpers/benchmarking/orchestrate_gateway_r51f.py [--sync-only]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
# the slot-waiting submit outlives the 60 s default ssh bound on a full
# cluster (the 2026-10-01 incident of the joint-fit orchestrator)
os.environ.setdefault("NA49_GW_TIMEOUT", "14400")

from helpers.na49_gateway.cluster_client import ClusterGatewayClient, _default_agent_command  # noqa: E402
from helpers.na49_gateway.gateway_config import get_config_list, get_config_str, load_user_config  # noqa: E402
from helpers.benchmarking.orchestrate_gateway_joint_fit import (  # noqa: E402
    frozen_rels_of, ledger_path, read_ledger, write_ledger)

RUN_ID = "r51f_joint_fit_2026-09-30"
DRIVER_REL = "helpers/benchmarking/staged_r51f_joint_fit.py"
REMOTE_PYTHON = "/usr/bin/python3.12"
N_FLOORS = 3
N_SLOTS = 32
MAX_GENERATIONS = 4
MAX_TASKS_PER_WORKER = 32
EVAL_TASKS = 24
PROFILE_TASKS = 8
POLL_SECONDS = 120
MAX_RESUBMITS = 2
LEDGER = ledger_path("r51f")

SYNC_DIRS = ("helpers/benchmarking", "dev/mixing2006/na49_mixing", "dev/mixing2006/helpers", "dev/mixing2006/config")
SYNC_FILES = (
    "dev/mixing2006/pyproject.toml",
    *(f"dev/mixing2006/spshadrons.web.cern.ch/data/{name}" for name in (
        "xsect_piminus.txt", "xsect_piplus.txt", "xsect_kaplus_pp.txt", "xsect_kaminus_pp.txt",
        "xsect_proton_pp.txt", "xsect_antiproton_pp.txt")),
    "dev/00runs/r51e_topology_fit_2026-09-29/report.json",
    "dev/00runs/r51e_topology_fit_2026-09-29/seal_manifest.json",
)
RSYNC_EXCLUDES = ("__pycache__", "*.pyc", ".pytest_cache")


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False).stdout.strip()


class Gateway:
    def __init__(self) -> None:
        config = load_user_config()
        self.host = get_config_str(config, "host")
        self.ssh_options = tuple(get_config_list(config, "ssh_options") or [])
        self.remote_root = "~/" + get_config_str(config, "remote_project_relpath")
        self.client = ClusterGatewayClient(host=self.host, agent_command=_default_agent_command(config),
                                           ssh_options=self.ssh_options, jobs_root=get_config_str(config, "jobs_root"),
                                           stale_heartbeat_seconds=3600, launch_grace_seconds=14400)
        self.run_dir = f"{self.remote_root}/dev/00runs/{RUN_ID}"
        # the remote agent has no config of its own: the worker pool comes from
        # the local per-user config with every submission
        self.workers = get_config_list(config, "workers") or []
        if not self.workers:
            raise RuntimeError("no workers in the per-user gateway config")

    # -- ssh helpers ---------------------------------------------------
    def ssh(self, command: str, *, input_text: str | None = None, check: bool = True) -> str:
        proc = subprocess.run(["ssh", *self.ssh_options, self.host, command], input=input_text, text=True,
                              capture_output=True, check=False, timeout=600)
        if check and proc.returncode != 0:
            raise RuntimeError(f"ssh {command!r} failed: {proc.stderr.strip()}")
        return proc.stdout

    def remote_python(self, script: str) -> str:
        return self.ssh("python3 -", input_text=script)

    def remote_files(self) -> set[str]:
        out = self.ssh(f"ls -1 {self.run_dir} 2>/dev/null || true", check=True)
        return {line.strip() for line in out.splitlines() if line.strip()}

    def job_alive(self, job_id: str, worker: str | None) -> bool | None:
        """True when the job's wrapper process runs on its worker, False when
        it is gone, None when the probe itself fails (never a resubmit)."""
        if not worker:
            return None
        probe = f"ssh -o BatchMode=yes {worker} 'ps -eo args | grep -c \"[/]{job_id}/run.sh\"'"
        proc = subprocess.run(["ssh", *self.ssh_options, self.host, probe], text=True, capture_output=True,
                              check=False, timeout=120)
        out = proc.stdout.strip().splitlines()
        if not out or not out[-1].isdigit():
            return None
        return int(out[-1]) > 0

    def remote_json(self, name: str) -> dict | None:
        out = self.ssh(f"cat {self.run_dir}/{name} 2>/dev/null || true")
        return json.loads(out) if out.strip() else None

    # -- sync ----------------------------------------------------------
    def sync(self) -> dict:
        ssh_cmd = " ".join(shlex.quote(t) for t in ["ssh", *self.ssh_options])
        excludes = [f"--exclude={e}" for e in RSYNC_EXCLUDES]
        for rel in SYNC_DIRS:
            self.ssh(f"mkdir -p {self.remote_root}/{rel}")
            subprocess.run(["rsync", "-az", "--delete", *excludes, "-e", ssh_cmd, f"{REPO_ROOT / rel}/",
                            f"{self.host}:{self.remote_root}/{rel}/"], check=True)
            log(f"sync: {rel}/")
        for rel in SYNC_FILES:
            parent = rel.rsplit("/", 1)[0]
            self.ssh(f"mkdir -p {self.remote_root}/{parent}")
            subprocess.run(["rsync", "-az", "-e", ssh_cmd, str((REPO_ROOT / rel).resolve()),
                            f"{self.host}:{self.remote_root}/{rel}"], check=True)
            log(f"sync: {rel}")
        # the digests the driver freezes, plus the driver and the data tables
        spec_rels = list(_frozen_rels()) + [DRIVER_REL] + [r for r in SYNC_FILES if "/data/" in r]
        local = {rel: sha256((REPO_ROOT / rel).resolve()) for rel in spec_rels}
        manifest = {
            "schema": "na49_mixing.gateway_sync/1", "run": RUN_ID, "synced_utc": now(),
            "source": {"outer": {"head": git(REPO_ROOT, "rev-parse", "HEAD"), "dirty_paths": git(REPO_ROOT, "status", "--porcelain").splitlines()},
                       "nested": {"head": git(REPO_ROOT / "dev" / "mixing2006", "rev-parse", "HEAD"),
                                  "dirty_paths": git(REPO_ROOT / "dev" / "mixing2006", "status", "--porcelain").splitlines()}},
            "targets": {"dirs": list(SYNC_DIRS), "files": list(SYNC_FILES)},
            "remote_python": REMOTE_PYTHON, "sha256": local,
        }
        self.ssh(f"cat > {self.remote_root}/SYNC_MANIFEST.json", input_text=json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        script = ("import hashlib, json, os\nroot = os.path.expanduser(%r)\nrels = %r\nout = {}\n"
                  "for r in rels:\n    p = os.path.join(root, r)\n"
                  "    out[r] = hashlib.sha256(open(p, 'rb').read()).hexdigest() if os.path.isfile(p) else None\n"
                  "print(json.dumps(out))\n") % (self.remote_root, spec_rels)
        remote = json.loads(self.remote_python(script))
        bad = {r: (local[r][:12], (remote.get(r) or "missing")[:12]) for r in spec_rels if remote.get(r) != local[r]}
        if bad:
            raise RuntimeError(f"remote digests differ after sync: {bad}")
        log(f"sync verified: {len(spec_rels)} digests match (outer {manifest['source']['outer']['head'][:7]}, "
            f"nested {manifest['source']['nested']['head'][:7]})")
        return manifest


def _frozen_rels() -> tuple:
    return frozen_rels_of(REPO_ROOT / DRIVER_REL)


class Jobs:
    """The local ledger of submitted jobs, keyed by the expected output."""

    def __init__(self, gw: Gateway) -> None:
        self.gw = gw
        self.state = read_ledger(LEDGER)

    def save(self) -> None:
        write_ledger(LEDGER, self.state)

    def ensure(self, key: str, args: str, tasks: int, name: str) -> None:
        """Submit the driver step unless a live or finished job exists for it."""
        rec = self.state.get(key)
        if rec and rec.get("state") not in ("FAILED", "SUBMIT_FAILED", "RESUBMIT"):
            return
        attempts = (rec or {}).get("attempts", 0)
        if attempts > MAX_RESUBMITS:
            raise RuntimeError(f"{key}: {attempts} attempts failed; see the job logs")
        cmd = f"{REMOTE_PYTHON} {DRIVER_REL} {args}"
        job = self.gw.client.submit(cmd, workdir=self.gw.remote_root, name=name, tasks=tasks, workers=self.gw.workers,
                                    max_tasks_per_worker=MAX_TASKS_PER_WORKER, wait_for_slot=True, poll_seconds=10.0)
        self.state[key] = {"job_id": job["job_id"], "cmd": cmd, "tasks": tasks, "submitted_utc": now(),
                           "submitted_epoch": time.time(), "state": job.get("state", "SUBMITTED"), "attempts": attempts + 1}
        self.save()
        log(f"submitted {name} ({tasks} tasks): {job['job_id']}")

    def wait_for(self, expected: dict[str, str], label: str) -> None:
        """Wait until every key's expected output file exists in the remote run dir."""
        while True:
            files = self.gw.remote_files()
            missing = {k: f for k, f in expected.items() if f not in files}
            if not missing:
                log(f"{label}: all {len(expected)} outputs present")
                return
            try:
                listing = {j["job_id"]: j for j in self.gw.client.list(limit=500)}
            except Exception as exc:  # noqa: BLE001 - an ssh hiccup is retried at the next poll
                log(f"{label}: job listing failed ({exc}); retrying")
                time.sleep(POLL_SECONDS)
                continue
            for key in list(missing):
                rec = self.state.get(key)
                if not rec or rec["job_id"] not in listing:
                    continue
                st = listing[rec["job_id"]].get("state")
                rec["state"] = st
                if st == "FAILED" or st == "SUBMIT_FAILED" or st == "CANCELED":
                    tail = self.gw.client.result(rec["job_id"], tail_lines=20)
                    err = tail.get("stderr_tail") or []
                    err = err if isinstance(err, list) else str(err).splitlines()
                    log(f"{key}: {st} (exit {tail.get('exit_code')}); stderr tail: {' | '.join(err[-5:])}")
                    rec["state"] = "RESUBMIT"
                elif st == "COMPLETED":
                    # a shared-filesystem listing may lag the write; after ten
                    # minutes a completed job without its output is a logic
                    # error, never a resubmission
                    seen = rec.setdefault("completed_seen", time.time())
                    if time.time() - seen > 600:
                        self.save()
                        raise RuntimeError(f"{key}: COMPLETED without its output {missing[key]}")
                elif st == "LOST":
                    alive = self.gw.job_alive(rec["job_id"], listing[rec["job_id"]].get("worker"))
                    if alive is False:
                        log(f"{key}: LOST, its wrapper is gone and its output is missing; resubmitting")
                        rec["state"] = "RESUBMIT"
                    elif alive:
                        rec["state"] = "LOST_ALIVE"
                        if time.time() - rec.get("alive_logged", 0) > 3600:
                            log(f"{key}: reported LOST, but its wrapper runs on {listing[rec['job_id']].get('worker')}; waiting")
                            rec["alive_logged"] = time.time()
            self.save()
            resub = [k for k in missing if (self.state.get(k) or {}).get("state") == "RESUBMIT"]
            for key in resub:
                rec = self.state[key]
                self.ensure(key, rec["cmd"].split(DRIVER_REL, 1)[1].strip(), rec["tasks"], key)
            log(f"{label}: {len(expected) - len(missing)}/{len(expected)} outputs present")
            time.sleep(POLL_SECONDS)


def slot_name(f: int, g: int, s: int) -> str:
    cls = ("sd_projectile", "sd_target", "dd", "nd")[s // 8]
    return f"slot_{cls}_{s % 8}_f{f}_g{g}.json"


def points(gw: Gateway, f: int) -> list[dict]:
    doc = gw.remote_json(f"generation_points_f{f}.json")
    return doc["points"] if doc else []


def main() -> int:
    parser = argparse.ArgumentParser(description="R51f on the cluster gateway")
    parser.add_argument("--sync-only", action="store_true")
    parser.add_argument("--no-sync", action="store_true", help="skip the code sync (resume after a local restart)")
    args = parser.parse_args()
    gw = Gateway()
    jobs = Jobs(gw)
    files = gw.remote_files()
    if not args.no_sync and "pre_registration.json" not in files:
        gw.sync()
    elif not args.no_sync:
        log("the remote pre-registration exists; the code is frozen for this run, so no re-sync")
    if args.sync_only:
        return 0
    if "pre_registration.json" not in files:
        jobs.ensure("prereg", "--pre-registration", 1, "r51f_prereg")
        jobs.wait_for({"prereg": "pre_registration.json"}, "pre-registration")
    for rnd in range(MAX_GENERATIONS):
        files = gw.remote_files()
        open_branches = [f for f in range(N_FLOORS) if f"branch_report_f{f}.json" not in files]
        if not open_branches:
            break
        # generation: the latest registered generation of every open branch
        expected = {}
        for f in open_branches:
            g = max(int(p["generation"]) for p in points(gw, f))
            for s in range(N_SLOTS):
                cls = ("sd_projectile", "sd_target", "dd", "nd")[s // 8]
                name = f"slot_{cls}_{s % 8}_f{f}_g{g}.json"
                key = f"gen_f{f}_g{g}_s{s}"
                expected[key] = name
                if name not in files:
                    jobs.ensure(key, f"--generate --floor-index {f} --generation {g} --slot {s} --workers 1", 1, key)
        jobs.wait_for(expected, f"generation round {rnd}")
        # evaluation: one job per open branch whose latest complete generation
        # has no fit report yet, concurrently (a generation registered by an
        # evaluation that finished during this round is generated next round)
        expected = {}
        files = gw.remote_files()
        for f in open_branches:
            if f"branch_report_f{f}.json" in files:
                continue
            complete = [int(p["generation"]) for p in points(gw, f) if all(slot_name(f, int(p["generation"]), s) in files for s in range(N_SLOTS))]
            if not complete:
                continue
            g = max(complete)
            key = f"eval_f{f}_g{g}"
            expected[key] = f"fit_report_f{f}_g{g}.json"
            if expected[key] not in files:
                jobs.ensure(key, f"--evaluate --floor-index {f}", EVAL_TASKS, key)
        jobs.wait_for(expected, f"evaluation round {rnd}")
        files = gw.remote_files()
        for f in open_branches:
            state = "final" if f"branch_report_f{f}.json" in files else "regeneration requested"
            log(f"branch {f}: {state} after round {rnd}")
    files = gw.remote_files()
    if any(f"branch_report_f{f}.json" not in files for f in range(N_FLOORS)):
        raise RuntimeError("a branch still requests regeneration after the last generation")
    for key, args_, tasks, out in (("profile", "--profile", PROFILE_TASKS, "report.json"),
                                   ("gate", "--gate", 1, "gate5_followup_r51f.json"),
                                   ("clean", "--clean-validate", EVAL_TASKS, "clean_environment_commands.json"),
                                   ("seal", "--seal", 1, "seal_manifest.json")):
        if out not in gw.remote_files():
            jobs.ensure(key, args_, tasks, f"r51f_{key}")
            jobs.wait_for({key: out}, key)
    # fetch the sealed run dir into the local artifact store
    local = REPO_ROOT / "dev" / "00runs" / RUN_ID
    local.mkdir(parents=True, exist_ok=True)
    ssh_cmd = " ".join(shlex.quote(t) for t in ["ssh", *gw.ssh_options])
    subprocess.run(["rsync", "-az", "-e", ssh_cmd, f"{gw.host}:{gw.run_dir}/", f"{local}/"], check=True)
    log(f"fetched the sealed run dir to {local}")
    print("ALL DONE", now(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
