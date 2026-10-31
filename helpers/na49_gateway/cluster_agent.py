#!/usr/bin/env python3
"""SSH-native job agent for dispatching NA49 workloads to worker nodes.

This module runs on a login node and launches commands on worker nodes over SSH.
Job state is tracked in a shared filesystem directory so callers can inspect
status, logs, and results without keeping a persistent daemon alive.

Job IDs supplied to ``status``, ``result``, and ``logs`` must match the
generated syntax (``JOB_ID_PATTERN``); every job directory is resolved and
verified to lie inside the configured jobs root before any file access, and
symlinked job directories are refused. The SSH job launch carries an explicit
timeout (``DEFAULT_LAUNCH_TIMEOUT_SECONDS``, overridable through the
``NA49_CLUSTER_GW_LAUNCH_TIMEOUT_SECONDS`` environment variable).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

try:
    from .gateway_config import (
        config_path_hint,
        get_config_int,
        get_config_list,
        get_config_str,
        load_user_config,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from gateway_config import (  # type: ignore[no-redef]
        config_path_hint,
        get_config_int,
        get_config_list,
        get_config_str,
        load_user_config,
    )

DEFAULT_MAX_TASKS_PER_WORKER = 64
DEFAULT_JOBS_ROOT = "~/.na49_cluster_gw/jobs"
TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELED", "SUBMIT_FAILED", "LOST"}

# Job-ID format. Minting and validation share these constants so the two can
# never drift: _mint_job_id() builds IDs as <UTC timestamp>-<8 hex characters>
# and asserts the result against JOB_ID_PATTERN, and resolve_job_dir() accepts
# only IDs matching the same pattern. Anything else (absolute paths, path
# separators, '.'/'..') is rejected before any filesystem access.
JOB_ID_TIME_FORMAT = "%Y%m%dT%H%M%SZ"
JOB_ID_PATTERN = r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$"
_JOB_ID_RE = re.compile(JOB_ID_PATTERN)

# SSH job-launch timeout. The launch handshake (nohup + PID echo) should
# return within this many seconds; a dead SSH transport fails the submit
# instead of hanging it. Override with NA49_CLUSTER_GW_LAUNCH_TIMEOUT_SECONDS.
DEFAULT_LAUNCH_TIMEOUT_SECONDS = 30.0
LAUNCH_TIMEOUT_SECONDS_ENV_VAR = "NA49_CLUSTER_GW_LAUNCH_TIMEOUT_SECONDS"

# Heartbeat / stale-job recovery knobs.
#
# The worker wrapper refreshes the ``last_seen`` marker file every
# NA49_CLUSTER_GW_HEARTBEAT_SECONDS (default: DEFAULT_HEARTBEAT_SECONDS). A job
# without an exit marker counts as RUNNING only while its heartbeat is fresh;
# a stale heartbeat, or a launch that never produced a heartbeat within
# DEFAULT_LAUNCH_GRACE_SECONDS, yields the terminal state LOST and stops
# reserving task capacity. All three values are configurable.
DEFAULT_HEARTBEAT_SECONDS = 20
HEARTBEAT_SECONDS_ENV_VAR = "NA49_CLUSTER_GW_HEARTBEAT_SECONDS"
DEFAULT_STALE_HEARTBEAT_SECONDS = 300
STALE_HEARTBEAT_SECONDS_CONFIG_KEY = "stale_heartbeat_seconds"
DEFAULT_LAUNCH_GRACE_SECONDS = 900
LAUNCH_GRACE_SECONDS_CONFIG_KEY = "launch_grace_seconds"
PRUNE_AGE_DAYS_CONFIG_KEY = "prune_age_days"
DEFAULT_PRUNE_AGE_DAYS = 14.0
_PRIVATE_DIR_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600

# A pid_probe callable takes (worker_hostname, pid) and returns True (alive),
# False (confirmed dead), or None (unknown, e.g. SSH transport failure). Only a
# confirmed-dead result may push a fresh-heartbeat job to LOST; unknown results
# never mark jobs LOST on their own.
PidProbe = Callable[[str, int], Optional[bool]]


def _local_pid_alive(pid: int) -> bool | None:
    """Best-effort liveness check for a PID in this machine's PID namespace.

    Returns True, False, or None when liveness cannot be determined. Meaningful
    only when the job's recorded hostname equals the local hostname; worker-node
    PIDs live in a different PID space than the login node.
    """

    if pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def _utc_now_iso() -> str:
    """Return the current UTC time in a compact ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from *path*."""

    return json.loads(path.read_text(encoding="utf-8"))


def _read_meta_safe(path: Path) -> dict[str, Any] | None:
    """Read *path* as a JSON object, returning None when it is missing or malformed.

    Listing and capacity paths must survive partially written or corrupted
    metadata files without crashing.
    """

    try:
        payload = _read_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _ensure_private_directory(
    path: Path, *, parents: bool = True, exist_ok: bool = True
) -> None:
    """Create *path* and force owner-only access (mode 0700)."""

    path.mkdir(parents=parents, exist_ok=exist_ok)
    os.chmod(path, _PRIVATE_DIR_MODE)


def _chmod_private_file(path: Path) -> None:
    """Force owner-only read/write on *path* (mode 0600)."""

    os.chmod(path, _PRIVATE_FILE_MODE)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON object to *path* atomically.

    The payload lands in a temporary file inside the target directory first and
    is moved into place with ``os.replace``, so concurrent readers observe
    either the previous or the new complete file and never a partial write.
    The resulting file is always mode 0600; the process umask is not applied
    as a world-readable default.
    """

    directory = path.parent
    handle_fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_name, _PRIVATE_FILE_MODE)
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _parse_ts_to_epoch(text: str | None) -> float | None:
    """Parse an ISO-8601 timestamp into a POSIX epoch float, or return None."""

    if not text:
        return None
    normalized = text.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _emit_json(payload: dict[str, Any], *, stream: Any = None) -> None:
    """Write a JSON payload to *stream* (default: ``sys.stdout`` at call time)."""

    if stream is None:
        stream = sys.stdout
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()


def _mint_job_id() -> str:
    """Return a fresh job ID in the generated ``<UTC timestamp>-<8 hex>`` format."""

    job_id = f"{datetime.now(timezone.utc):{JOB_ID_TIME_FORMAT}}-{uuid.uuid4().hex[:8]}"
    if not _JOB_ID_RE.match(job_id):  # pragma: no cover - guards format drift
        raise RuntimeError(f"Minted job ID {job_id!r} does not match {JOB_ID_PATTERN}")
    return job_id


def resolve_job_dir(jobs_root: Path, job_id: str) -> Path:
    """Return the job directory for *job_id*, confined to the jobs root.

    Only generated job IDs (pattern ``JOB_ID_PATTERN``) are accepted; absolute
    paths, path separators, and ``.``/``..`` components fail the pattern check
    before any filesystem access. As defense in depth, the resolved candidate
    directory must lie strictly beneath the resolved jobs root, and a symlinked
    job directory is refused.
    """

    if not isinstance(job_id, str) or not _JOB_ID_RE.match(job_id):
        raise ValueError(
            f"Invalid job_id {job_id!r}: expected the generated format "
            f"{JOB_ID_PATTERN} (for example 20260830T120000Z-0123abcd)."
        )
    resolved_root = jobs_root.resolve()
    candidate = jobs_root / job_id
    if candidate.is_symlink():
        raise ValueError(
            f"Invalid job_id {job_id!r}: job directory {candidate} is a symlink; "
            f"symlinked job directories are not accepted."
        )
    resolved_candidate = candidate.resolve(strict=False)
    if resolved_candidate != resolved_root and resolved_root not in resolved_candidate.parents:
        raise ValueError(
            f"Invalid job_id {job_id!r}: resolves to {resolved_candidate}, "
            f"outside the jobs root {resolved_root}."
        )
    return candidate


def _resolve_seconds(value: float | None, default: float) -> float:
    """Resolve an optional CLI/config threshold against its default."""

    return default if value is None else float(value)


def _optional_text(path: Path) -> str | None:
    """Read text from *path* if it exists."""

    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _optional_int(path: Path) -> int | None:
    """Read an integer value from *path* if it exists."""

    raw = _optional_text(path)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _recorded_pid(
    job_directory: Path,
    meta: dict[str, Any],
) -> tuple[str | None, str | None, int | None]:
    """Return the (hostname_file, submit_alias, pid) evidence for a job.

    ``submit_alias`` is ``meta["worker"]``, the hostname submits actually
    dial over SSH; SSH PID probes must target it. The wrapper-written
    ``hostname`` marker identifies the machine whose PID namespace the
    recorded PID belongs to and is only consulted for local liveness
    checks against the current host.

    PID preference order: ``worker_pid`` in meta (written by submits that
    observed the worker-side wrapper), ``launcher_pid`` in meta (the nohup
    PID captured over SSH), then the ``worker_pid`` marker file written by
    the wrapper.
    """

    hostname = _optional_text(job_directory / "hostname")
    raw_pid = meta.get("worker_pid")
    if raw_pid in (None, ""):
        raw_pid = meta.get("launcher_pid")
    if raw_pid in (None, ""):
        raw_pid = _optional_int(job_directory / "worker_pid")
    try:
        pid = int(raw_pid) if raw_pid not in (None, "") else None
    except (TypeError, ValueError):
        pid = None
    worker_alias = meta.get("worker")
    return (
        str(hostname) if hostname else None,
        str(worker_alias) if worker_alias else None,
        pid,
    )


def _heartbeat_age_seconds(job_directory: Path, *, now: float | None = None) -> float | None:
    """Return the age in seconds of the freshest worker heartbeat.

    The ``last_seen`` marker holds a POSIX epoch written by the wrapper's
    heartbeat loop. Returns None when no heartbeat was ever recorded.
    """

    if now is None:
        now = time.time()
    last_seen = _optional_int(job_directory / "last_seen")
    if last_seen is None:
        return None
    return max(0.0, now - float(last_seen))


def _running_or_lost(
    job_directory: Path,
    meta: dict[str, Any],
    *,
    now: float | None = None,
    pid_probe: PidProbe | None = None,
    stale_heartbeat_seconds: float = DEFAULT_STALE_HEARTBEAT_SECONDS,
    launch_grace_seconds: float = DEFAULT_LAUNCH_GRACE_SECONDS,
) -> tuple[str, str | None]:
    """Decide RUNNING versus LOST for a job without exit markers.

    A job stays RUNNING only while no positive evidence says otherwise. Two
    signals can produce the terminal LOST state: a PID probe confirming the
    recorded worker process is dead, and an unfresh heartbeat, either a stale
    ``last_seen`` value or no heartbeat at all beyond the launch grace window.
    Probe results of None (transport failure) are inconclusive and never mark a
    job LOST by themselves. A probe result of True (the worker process is
    alive) vetoes the heartbeat signals: a live process whose heartbeat
    writes are delayed by a shared-filesystem stall is RUNNING, not LOST.
    """

    if now is None:
        now = time.time()
    reasons: list[str] = []

    hostname_file, worker_alias, pid = _recorded_pid(job_directory, meta)
    alive: bool | None = None
    if pid is not None:
        if pid_probe is not None:
            alive = pid_probe(worker_alias or "", pid)
        elif hostname_file is not None and hostname_file == socket.gethostname():
            alive = _local_pid_alive(pid)
        else:
            alive = None
        if alive is False:
            reasons.append("dead_pid")

    # A worker process confirmed alive by the probe is running whatever
    # its heartbeat file says: a shared-filesystem stall delays every
    # heartbeat write at once (the 2026-09-24 R45 LOST wave), and a stale
    # or missing heartbeat is then no evidence of a lost job. Only an
    # unconfirmed (None) or dead (False) probe lets the heartbeat decide.
    heartbeat_age = _heartbeat_age_seconds(job_directory, now=now)
    if alive is True:
        pass
    elif heartbeat_age is not None:
        if heartbeat_age > stale_heartbeat_seconds:
            reasons.append("stale_heartbeat")
    else:
        reference_epoch = (
            _parse_ts_to_epoch(meta.get("launched_at"))
            if isinstance(meta.get("launched_at"), str)
            else None
        ) or (
            _parse_ts_to_epoch(meta.get("submitted_at"))
            if isinstance(meta.get("submitted_at"), str)
            else None
        )
        if reference_epoch is None:
            try:
                reference_epoch = job_directory.stat().st_mtime
            except OSError:
                reference_epoch = now
        if (now - reference_epoch) > launch_grace_seconds:
            reasons.append("no_heartbeat_past_launch_grace")

    if reasons:
        return "LOST", "+".join(reasons)
    return "RUNNING", None


def _compute_state(
    job_directory: Path,
    *,
    now: float | None = None,
    pid_probe: PidProbe | None = None,
    stale_heartbeat_seconds: float = DEFAULT_STALE_HEARTBEAT_SECONDS,
    launch_grace_seconds: float = DEFAULT_LAUNCH_GRACE_SECONDS,
) -> tuple[str, int | None]:
    """Infer job state and exit code from tracked files and liveness evidence.

    Marker files decide terminal outcomes: CANCELED, COMPLETED, FAILED, and
    SUBMIT_FAILED. An explicitly materialized LOST verdict in meta.json is
    sticky. Without an exit marker, the job is RUNNING only while the liveness
    ladder in :func:`_running_or_lost` finds no counter-evidence (PID alive
    per probe, heartbeat fresh); otherwise it is LOST.
    """

    if (job_directory / "canceled_at").exists():
        return "CANCELED", None

    exit_code = _optional_int(job_directory / "exit_code")
    if exit_code is not None:
        return ("COMPLETED" if exit_code == 0 else "FAILED"), exit_code

    meta = _read_meta_safe(job_directory / "meta.json") or {}
    if meta.get("state") == "SUBMIT_FAILED":
        return "SUBMIT_FAILED", None
    if meta.get("state") == "LOST":
        return "LOST", None

    state, _lost_reason = _running_or_lost(
        job_directory,
        meta,
        now=now,
        pid_probe=pid_probe,
        stale_heartbeat_seconds=stale_heartbeat_seconds,
        launch_grace_seconds=launch_grace_seconds,
    )
    return state, None


def _tail_lines(path: Path, line_count: int) -> list[str]:
    """Return the last *line_count* lines from *path*."""

    if line_count <= 0 or not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return list(deque(handle, maxlen=line_count))


def _parse_workers(worker_csv: str | None) -> list[str]:
    """Parse a comma-separated worker list."""

    if not worker_csv:
        raise ValueError(
            "No workers configured. Pass --workers/--worker or configure "
            f"'workers' in {config_path_hint()}."
        )
    workers = [item.strip() for item in worker_csv.split(",") if item.strip()]
    if not workers:
        raise ValueError("No workers configured.")
    return workers


def _job_tasks(meta: dict[str, Any]) -> int:
    """Return the number of reserved task slots for a job metadata record."""

    try:
        tasks = int(meta.get("tasks", 1))
    except (TypeError, ValueError):
        tasks = 1
    return max(tasks, 1)


def _active_tasks_by_worker(
    jobs_root: Path,
    *,
    stale_heartbeat_seconds: float | None = None,
    launch_grace_seconds: float | None = None,
) -> dict[str, int]:
    """Return active reserved task counts keyed by worker hostname."""

    stale = _resolve_seconds(stale_heartbeat_seconds, DEFAULT_STALE_HEARTBEAT_SECONDS)
    grace = _resolve_seconds(launch_grace_seconds, DEFAULT_LAUNCH_GRACE_SECONDS)
    counts: dict[str, int] = {}
    if not jobs_root.exists():
        return counts

    for child in jobs_root.iterdir():
        if not child.is_dir() or child.is_symlink():
            continue
        meta_path = child / "meta.json"
        if not meta_path.exists():
            continue
        meta = _read_meta_safe(meta_path)
        if meta is None:
            continue
        worker = meta.get("worker")
        if not worker:
            continue
        state, _ = _compute_state(
            child,
            stale_heartbeat_seconds=stale,
            launch_grace_seconds=grace,
        )
        if state not in TERMINAL_STATES:
            counts[worker] = counts.get(worker, 0) + _job_tasks(meta)
    return counts


def _active_jobs_by_worker(
    jobs_root: Path,
    *,
    stale_heartbeat_seconds: float | None = None,
    launch_grace_seconds: float | None = None,
) -> dict[str, int]:
    """Return active job counts keyed by worker hostname."""

    stale = _resolve_seconds(stale_heartbeat_seconds, DEFAULT_STALE_HEARTBEAT_SECONDS)
    grace = _resolve_seconds(launch_grace_seconds, DEFAULT_LAUNCH_GRACE_SECONDS)
    counts: dict[str, int] = {}
    if not jobs_root.exists():
        return counts

    for child in jobs_root.iterdir():
        if not child.is_dir() or child.is_symlink():
            continue
        meta_path = child / "meta.json"
        if not meta_path.exists():
            continue
        meta = _read_meta_safe(meta_path)
        if meta is None:
            continue
        worker = meta.get("worker")
        if not worker:
            continue
        state, _ = _compute_state(
            child,
            stale_heartbeat_seconds=stale,
            launch_grace_seconds=grace,
        )
        if state not in TERMINAL_STATES:
            counts[worker] = counts.get(worker, 0) + 1
    return counts


def _pick_worker(
    jobs_root: Path,
    workers: list[str],
    *,
    requested_tasks: int,
    max_tasks_per_worker: int,
    stale_heartbeat_seconds: float | None = None,
    launch_grace_seconds: float | None = None,
) -> str | None:
    """Pick the least-loaded worker that has enough free task capacity."""

    counts = _active_tasks_by_worker(
        jobs_root,
        stale_heartbeat_seconds=stale_heartbeat_seconds,
        launch_grace_seconds=launch_grace_seconds,
    )
    candidates = [
        worker
        for worker in workers
        if counts.get(worker, 0) + requested_tasks <= max_tasks_per_worker
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda worker: (counts.get(worker, 0), worker))


def _build_submit_command(args: argparse.Namespace) -> str:
    """Build a shell command string from CLI submit arguments."""

    if args.cmd and args.command:
        raise ValueError("Use either --cmd or positional command tokens, not both.")

    if args.cmd:
        command = args.cmd.strip()
        if not command:
            raise ValueError("--cmd cannot be empty.")
        return command

    tokens = args.command
    if tokens and tokens[0] == "--":
        tokens = tokens[1:]
    if not tokens:
        raise ValueError("No command provided. Use --cmd or append command tokens after '--'.")
    return " ".join(shlex.quote(token) for token in tokens)


def _build_run_script(
    job_directory: Path,
    workdir: Path,
    command: str,
    tasks: int,
    *,
    worker_bin_dir: str | None = None,
) -> str:
    """Create the bash wrapper executed on the worker node."""

    worker_bin_block = ""
    if worker_bin_dir:
        worker_bin_block = (
            f"WORKER_BIN_DIR={shlex.quote(worker_bin_dir)}\n"
            'export PATH="$WORKER_BIN_DIR:$PATH"\n'
        )

    return f"""#!/usr/bin/env bash
set -u

JOB_DIR={shlex.quote(str(job_directory))}
WORKDIR={shlex.quote(str(workdir))}
STDOUT_FILE={shlex.quote(str(job_directory / "stdout.log"))}
STDERR_FILE={shlex.quote(str(job_directory / "stderr.log"))}
EXIT_FILE={shlex.quote(str(job_directory / "exit_code"))}
START_FILE={shlex.quote(str(job_directory / "started_at"))}
END_FILE={shlex.quote(str(job_directory / "finished_at"))}
WORKER_PID_FILE={shlex.quote(str(job_directory / "worker_pid"))}
LAST_SEEN_FILE={shlex.quote(str(job_directory / "last_seen"))}
CMD={shlex.quote(command)}
TASKS={tasks}

umask 077
date -Is > "$START_FILE"
echo "${{HOSTNAME:-unknown}}" > "$JOB_DIR/hostname"
echo "$TASKS" > "$JOB_DIR/tasks_used"
echo "$$" > "$WORKER_PID_FILE"

# Heartbeat loop for stale-job detection. The wrapper refreshes LAST_SEEN every
# HEARTBEAT_SECONDS; the loop watches its parent so an orphaned writer cannot
# fake liveness after the wrapper is killed. Writes go through an atomic rename.
HEARTBEAT_SECONDS=${{NA49_CLUSTER_GW_HEARTBEAT_SECONDS:-{DEFAULT_HEARTBEAT_SECONDS}}}
date +%s > "$LAST_SEEN_FILE"
(
  while kill -0 "$PPID" 2>/dev/null; do
    sleep "$HEARTBEAT_SECONDS"
    date +%s > "$LAST_SEEN_FILE.tmp" && mv -f "$LAST_SEEN_FILE.tmp" "$LAST_SEEN_FILE"
  done
) &
HEARTBEAT_PID=$!
trap 'kill "$HEARTBEAT_PID" 2>/dev/null' EXIT

# Default to one threaded math/runtime libraries so reserved task slots are not
# silently oversubscribed by nested BLAS/OpenMP threading.
export NA49_CLUSTER_GW_TASKS="$TASKS"
{worker_bin_block}export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-1}}"
export OPENBLAS_NUM_THREADS="${{OPENBLAS_NUM_THREADS:-1}}"
export MKL_NUM_THREADS="${{MKL_NUM_THREADS:-1}}"
export NUMEXPR_NUM_THREADS="${{NUMEXPR_NUM_THREADS:-1}}"
export VECLIB_MAXIMUM_THREADS="${{VECLIB_MAXIMUM_THREADS:-1}}"
export BLIS_NUM_THREADS="${{BLIS_NUM_THREADS:-1}}"

cd "$WORKDIR" || {{
  echo "Failed to cd to $WORKDIR" > "$STDERR_FILE"
  echo 200 > "$EXIT_FILE"
  date -Is > "$END_FILE"
  exit 200
}}

bash -lc "$CMD" > "$STDOUT_FILE" 2> "$STDERR_FILE"
rc=$?
echo "$rc" > "$EXIT_FILE"
date -Is > "$END_FILE"
exit "$rc"
"""


def _launch_timeout_seconds() -> float:
    """Resolve the SSH job-launch timeout from the environment."""

    raw = os.environ.get(LAUNCH_TIMEOUT_SECONDS_ENV_VAR)
    try:
        return float(raw) if raw else DEFAULT_LAUNCH_TIMEOUT_SECONDS
    except (TypeError, ValueError):
        return DEFAULT_LAUNCH_TIMEOUT_SECONDS


def _launch_remote_job(
    *,
    ssh_bin: str,
    ssh_options: list[str],
    worker: str,
    run_script_path: Path,
    timeout_seconds: float | None = None,
) -> str:
    """Launch *run_script_path* on *worker* over SSH and return the background PID.

    The launch handshake is bounded by *timeout_seconds* (default:
    ``DEFAULT_LAUNCH_TIMEOUT_SECONDS``, overridable through
    ``NA49_CLUSTER_GW_LAUNCH_TIMEOUT_SECONDS``) so a dead SSH transport fails
    the submission instead of hanging it.
    """

    resolved_timeout = _resolve_seconds(timeout_seconds, _launch_timeout_seconds())
    remote_command = f"nohup bash {shlex.quote(str(run_script_path))} >/dev/null 2>&1 & echo $!"
    cmd = [ssh_bin, *ssh_options, worker, remote_command]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=resolved_timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"SSH launch on {worker} timed out after {resolved_timeout:g} s "
            f"(set {LAUNCH_TIMEOUT_SECONDS_ENV_VAR} to raise the limit)."
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"SSH launch failed on {worker}: rc={proc.returncode}; "
            f"stdout={proc.stdout.strip()!r}; stderr={proc.stderr.strip()!r}"
        )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"SSH launch on {worker} returned no PID.")
    return lines[-1]


def _pid_alive_over_ssh(
    *,
    ssh_bin: str,
    ssh_options: list[str],
    worker: str,
    pid: int,
    connect_timeout_seconds: int = 10,
) -> bool | None:
    """Check *pid* liveness on *worker* over SSH.

    Returns True when the remote process exists, False when the remote shell
    reports it gone, and None when the transport itself fails or times out.
    None is inconclusive by design: a transient SSH failure must never mark a
    job LOST on its own.
    """

    remote_command = f"kill -0 {int(pid)} >/dev/null 2>&1 && echo ALIVE || echo DEAD"
    cmd = [
        ssh_bin,
        *ssh_options,
        "-o",
        f"ConnectTimeout={connect_timeout_seconds}",
        "-o",
        "BatchMode=yes",
        worker,
        remote_command,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=connect_timeout_seconds + 20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = proc.stdout.strip()
    if "ALIVE" in output:
        return True
    if "DEAD" in output:
        return False
    return None


def _load_job_view(
    jobs_root: Path,
    job_id: str,
    *,
    stale_heartbeat_seconds: float | None = None,
    launch_grace_seconds: float | None = None,
) -> dict[str, Any]:
    """Load status and metadata for *job_id*."""

    job_directory = resolve_job_dir(jobs_root, job_id)
    meta_path = job_directory / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Unknown job_id: {job_id}")
    meta = _read_meta_safe(meta_path)
    if meta is None:
        raise ValueError(f"Malformed or unreadable meta.json for job {job_id}")

    state, exit_code = _compute_state(
        job_directory,
        stale_heartbeat_seconds=_resolve_seconds(
            stale_heartbeat_seconds, DEFAULT_STALE_HEARTBEAT_SECONDS
        ),
        launch_grace_seconds=_resolve_seconds(
            launch_grace_seconds, DEFAULT_LAUNCH_GRACE_SECONDS
        ),
    )
    stdout_path = job_directory / "stdout.log"
    stderr_path = job_directory / "stderr.log"

    view: dict[str, Any] = dict(meta)
    view["state"] = state
    view["exit_code"] = exit_code
    view["started_at"] = _optional_text(job_directory / "started_at")
    view["finished_at"] = _optional_text(job_directory / "finished_at")
    view["hostname"] = _optional_text(job_directory / "hostname")
    view["worker_pid_file"] = _optional_int(job_directory / "worker_pid")
    view["last_seen"] = _optional_int(job_directory / "last_seen")
    heartbeat_age = _heartbeat_age_seconds(job_directory)
    view["heartbeat_age_seconds"] = round(heartbeat_age, 1) if heartbeat_age is not None else None
    view["stdout_path"] = str(stdout_path)
    view["stderr_path"] = str(stderr_path)
    view["stdout_bytes"] = stdout_path.stat().st_size if stdout_path.exists() else 0
    view["stderr_bytes"] = stderr_path.stat().st_size if stderr_path.exists() else 0
    return view


def _stream_log_until_complete(
    job_directory: Path,
    log_path: Path,
    poll_seconds: float,
    *,
    stale_heartbeat_seconds: float | None = None,
    launch_grace_seconds: float | None = None,
) -> None:
    """Follow *log_path* until the job reaches a terminal state."""

    stale = _resolve_seconds(stale_heartbeat_seconds, DEFAULT_STALE_HEARTBEAT_SECONDS)
    grace = _resolve_seconds(launch_grace_seconds, DEFAULT_LAUNCH_GRACE_SECONDS)
    position = 0
    while True:
        if log_path.exists():
            with log_path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(position)
                chunk = handle.read()
                if chunk:
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                position = handle.tell()
        state, _ = _compute_state(
            job_directory,
            stale_heartbeat_seconds=stale,
            launch_grace_seconds=grace,
        )
        if state in TERMINAL_STATES:
            if log_path.exists():
                with log_path.open("r", encoding="utf-8", errors="replace") as handle:
                    handle.seek(position)
                    chunk = handle.read()
                    if chunk:
                        sys.stdout.write(chunk)
                        sys.stdout.flush()
            break
        time.sleep(max(poll_seconds, 0.05))


@contextmanager
def _submission_lock(jobs_root: Path) -> Iterator[None]:
    """Serialize submission-time capacity accounting on the shared jobs root."""

    _ensure_private_directory(jobs_root)
    lock_path = jobs_root / ".submit.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        _chmod_private_file(lock_path)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _reserve_job_slot(
    *,
    args: argparse.Namespace,
    jobs_root: Path,
    workers: list[str],
    command: str,
) -> tuple[str, Path, Path, dict[str, Any]]:
    """Reserve worker capacity and create the initial job record."""

    if args.tasks <= 0:
        raise ValueError("--tasks must be a positive integer.")
    if args.max_tasks_per_worker <= 0:
        raise ValueError("--max-tasks-per-worker must be a positive integer.")
    if args.tasks > args.max_tasks_per_worker:
        raise ValueError("--tasks cannot exceed --max-tasks-per-worker.")

    submit_started = time.monotonic()
    timeout_seconds = args.wait_timeout_seconds

    while True:
        with _submission_lock(jobs_root):
            if args.worker:
                if args.worker not in workers:
                    raise ValueError(f"Worker {args.worker!r} not in allowed workers: {workers}")
                active_by_worker = _active_tasks_by_worker(
                    jobs_root,
                    stale_heartbeat_seconds=args.stale_heartbeat_seconds,
                    launch_grace_seconds=args.launch_grace_seconds,
                )
                chosen_worker = None
                if active_by_worker.get(args.worker, 0) + args.tasks <= args.max_tasks_per_worker:
                    chosen_worker = args.worker
            else:
                chosen_worker = _pick_worker(
                    jobs_root,
                    workers,
                    requested_tasks=args.tasks,
                    max_tasks_per_worker=args.max_tasks_per_worker,
                    stale_heartbeat_seconds=args.stale_heartbeat_seconds,
                    launch_grace_seconds=args.launch_grace_seconds,
                )

            if chosen_worker is not None:
                job_id = _mint_job_id()
                job_directory = resolve_job_dir(jobs_root, job_id)
                _ensure_private_directory(job_directory, parents=False, exist_ok=False)
                run_script_path = job_directory / "run.sh"
                run_script_path.write_text(
                    _build_run_script(
                        job_directory,
                        Path(args.workdir).expanduser(),
                        command,
                        args.tasks,
                        worker_bin_dir=getattr(args, "worker_bin_dir", None),
                    ),
                    encoding="utf-8",
                )
                os.chmod(run_script_path, 0o700)

                # The metadata file is written while holding the lock so the task
                # reservation becomes visible before any competing submit call
                # makes its own scheduling decision.
                meta = {
                    "job_id": job_id,
                    "name": args.name,
                    "worker": chosen_worker,
                    "workers": workers,
                    "command": command,
                    "workdir": str(Path(args.workdir).expanduser()),
                    "jobs_root": str(jobs_root),
                    "submitted_at": _utc_now_iso(),
                    "submit_host": socket.gethostname(),
                    "state": "SUBMITTED",
                    "tasks": args.tasks,
                    "max_tasks_per_worker": args.max_tasks_per_worker,
                }
                _write_json(job_directory / "meta.json", meta)
                command_path = job_directory / "command.sh"
                command_path.write_text(command + "\n", encoding="utf-8")
                _chmod_private_file(command_path)
                return chosen_worker, job_directory, run_script_path, meta

        if not args.wait_for_slot:
            raise RuntimeError("No worker has enough free task capacity for this submission.")
        if timeout_seconds is not None and (time.monotonic() - submit_started) > timeout_seconds:
            raise TimeoutError("Timed out waiting for a worker task slot.")
        time.sleep(max(args.poll_seconds, 0.05))


def _worker_capacity_view(
    jobs_root: Path,
    workers: list[str],
    *,
    max_tasks_per_worker: int,
    stale_heartbeat_seconds: float | None = None,
    launch_grace_seconds: float | None = None,
) -> list[dict[str, Any]]:
    """Return a structured worker-capacity summary."""

    task_counts = _active_tasks_by_worker(
        jobs_root,
        stale_heartbeat_seconds=stale_heartbeat_seconds,
        launch_grace_seconds=launch_grace_seconds,
    )
    job_counts = _active_jobs_by_worker(
        jobs_root,
        stale_heartbeat_seconds=stale_heartbeat_seconds,
        launch_grace_seconds=launch_grace_seconds,
    )
    view: list[dict[str, Any]] = []
    for worker in workers:
        active_tasks = task_counts.get(worker, 0)
        view.append(
            {
                "worker": worker,
                "active_jobs": job_counts.get(worker, 0),
                "active_tasks": active_tasks,
                "free_tasks": max(0, max_tasks_per_worker - active_tasks),
                "max_tasks": max_tasks_per_worker,
            }
        )
    return view


def _materialize_lost(job_directory: Path, meta: dict[str, Any], reason: str) -> None:
    """Persist an explicit LOST verdict into *meta* atomically."""

    meta["state"] = "LOST"
    meta["lost_at"] = _utc_now_iso()
    meta["lost_reason"] = reason
    _write_json(job_directory / "meta.json", meta)


def _terminal_marker_epoch(job_directory: Path, meta: dict[str, Any]) -> float:
    """Return the best available time at which a terminal job finished.

    Preference order: marker-file timestamps (finished_at, canceled_at), the
    materialized lost_at field, then file modification times as a fallback.
    """

    candidates = [
        _parse_ts_to_epoch(_optional_text(job_directory / "finished_at")),
        _parse_ts_to_epoch(_optional_text(job_directory / "canceled_at")),
        _parse_ts_to_epoch(meta.get("lost_at")) if isinstance(meta.get("lost_at"), str) else None,
    ]
    timestamps = [value for value in candidates if value is not None]
    if timestamps:
        return max(timestamps)
    for name in ("exit_code",):
        path = job_directory / name
        if path.exists():
            timestamps.append(path.stat().st_mtime)
    if timestamps:
        return max(timestamps)
    return job_directory.stat().st_mtime


def _prune_terminal_jobs(
    jobs_root: Path,
    *,
    older_than_epoch: float,
    stale_heartbeat_seconds: float | None = None,
    launch_grace_seconds: float | None = None,
) -> list[str]:
    """Delete terminal job directories whose end time precedes the cutoff."""

    stale = _resolve_seconds(stale_heartbeat_seconds, DEFAULT_STALE_HEARTBEAT_SECONDS)
    grace = _resolve_seconds(launch_grace_seconds, DEFAULT_LAUNCH_GRACE_SECONDS)
    pruned: list[str] = []
    for child in sorted(jobs_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if child.is_symlink():
            # Never delete through a symlinked entry; skip it.
            continue
        meta = _read_meta_safe(child / "meta.json")
        if meta is None:
            continue
        try:
            state, _ = _compute_state(
                child,
                stale_heartbeat_seconds=stale,
                launch_grace_seconds=grace,
            )
            if state not in TERMINAL_STATES:
                continue
            if _terminal_marker_epoch(child, meta) > older_than_epoch:
                continue
            shutil.rmtree(child)
        except OSError:
            continue
        pruned.append(str(meta.get("job_id") or child.name))
    return pruned


def _command_reconcile(args: argparse.Namespace) -> int:
    """Handle the ``reconcile`` subcommand.

    Scans every job directory, applies the LOST liveness ladder (with optional
    SSH PID probes), materializes newly lost verdicts into meta.json so they
    stop reserving capacity, and optionally prunes terminal directories older
    than the retention age.
    """

    jobs_root = Path(args.jobs_root).expanduser().resolve()
    summary: dict[str, Any] = {
        "ok": True,
        "jobs_root": str(jobs_root),
        "probe": "ssh" if args.probe else "disabled",
        "scanned": 0,
        "active": 0,
        "terminal": 0,
        "already_lost": 0,
        "malformed": 0,
        "lost_materialized": [],
        "prune_requested": bool(args.prune),
        "pruned": [],
    }

    pid_probe: PidProbe | None = None
    if args.probe:
        ssh_options = list(args.ssh_option or [])

        def _ssh_probe(worker: str, worker_pid: int) -> bool | None:
            return _pid_alive_over_ssh(
                ssh_bin=args.ssh_bin,
                ssh_options=ssh_options,
                worker=worker,
                pid=worker_pid,
            )

        pid_probe = _ssh_probe

    stale_seconds = _resolve_seconds(args.stale_heartbeat_seconds, DEFAULT_STALE_HEARTBEAT_SECONDS)
    grace_seconds = _resolve_seconds(args.launch_grace_seconds, DEFAULT_LAUNCH_GRACE_SECONDS)

    job_directories: list[Path] = []
    if jobs_root.exists():
        job_directories = sorted(
            (
                child
                for child in jobs_root.iterdir()
                if child.is_dir() and not child.is_symlink() and not child.name.startswith(".")
            ),
            key=lambda path: path.name,
        )

    for job_directory in job_directories:
        meta = _read_meta_safe(job_directory / "meta.json")
        if meta is None:
            summary["malformed"] += 1
            continue
        summary["scanned"] += 1

        state, _exit_code = _compute_state(
            job_directory,
            pid_probe=pid_probe,
            stale_heartbeat_seconds=stale_seconds,
            launch_grace_seconds=grace_seconds,
        )
        if state == "RUNNING":
            summary["active"] += 1
            continue

        summary["terminal"] += 1
        if state != "LOST":
            continue
        if meta.get("state") == "LOST":
            summary["already_lost"] += 1
            continue
        _, reason = _running_or_lost(
            job_directory,
            meta,
            pid_probe=pid_probe,
            stale_heartbeat_seconds=stale_seconds,
            launch_grace_seconds=grace_seconds,
        )
        _materialize_lost(job_directory, meta, reason or "unspecified")
        summary["lost_materialized"].append(str(meta.get("job_id") or job_directory.name))

    if args.prune:
        cutoff_epoch = time.time() - float(args.prune_age_days) * 86400.0
        summary["pruned"] = _prune_terminal_jobs(
            jobs_root,
            older_than_epoch=cutoff_epoch,
            stale_heartbeat_seconds=stale_seconds,
            launch_grace_seconds=grace_seconds,
        )

    _emit_json(summary)
    return 0


def _command_submit(args: argparse.Namespace) -> int:
    """Handle the ``submit`` subcommand."""

    jobs_root = Path(args.jobs_root).expanduser().resolve()
    _ensure_private_directory(jobs_root)

    workers = _parse_workers(args.workers or args.worker)
    command = _build_submit_command(args)
    chosen_worker, job_directory, run_script_path, meta = _reserve_job_slot(
        args=args,
        jobs_root=jobs_root,
        workers=workers,
        command=command,
    )

    try:
        launcher_pid = _launch_remote_job(
            ssh_bin=args.ssh_bin,
            ssh_options=list(args.ssh_option or []),
            worker=chosen_worker,
            run_script_path=run_script_path,
        )
        meta["launcher_pid"] = launcher_pid
        meta["launched_at"] = _utc_now_iso()
        meta["state"] = "RUNNING"
        _write_json(job_directory / "meta.json", meta)
    except Exception as exc:  # noqa: BLE001
        meta["state"] = "SUBMIT_FAILED"
        meta["submit_error"] = str(exc)
        _write_json(job_directory / "meta.json", meta)
        _emit_json({"ok": False, "job_id": meta["job_id"], "error": str(exc)}, stream=sys.stderr)
        return 2

    _emit_json(
        {
            "ok": True,
            "job_id": meta["job_id"],
            "worker": chosen_worker,
            "launcher_pid": launcher_pid,
            "job_dir": str(job_directory),
            "tasks": meta["tasks"],
        }
    )
    return 0


def _command_status(args: argparse.Namespace) -> int:
    """Handle the ``status`` subcommand."""

    jobs_root = Path(args.jobs_root).expanduser().resolve()
    try:
        payload = _load_job_view(
            jobs_root,
            args.job_id,
            stale_heartbeat_seconds=args.stale_heartbeat_seconds,
            launch_grace_seconds=args.launch_grace_seconds,
        )
    except FileNotFoundError as exc:
        _emit_json({"ok": False, "error": str(exc)}, stream=sys.stderr)
        return 1
    _emit_json({"ok": True, "job": payload})
    return 0


def _command_result(args: argparse.Namespace) -> int:
    """Handle the ``result`` subcommand."""

    jobs_root = Path(args.jobs_root).expanduser().resolve()
    try:
        payload = _load_job_view(
            jobs_root,
            args.job_id,
            stale_heartbeat_seconds=args.stale_heartbeat_seconds,
            launch_grace_seconds=args.launch_grace_seconds,
        )
    except FileNotFoundError as exc:
        _emit_json({"ok": False, "error": str(exc)}, stream=sys.stderr)
        return 1

    job_directory = resolve_job_dir(jobs_root, args.job_id)
    payload["stdout_tail"] = _tail_lines(job_directory / "stdout.log", args.tail_lines)
    payload["stderr_tail"] = _tail_lines(job_directory / "stderr.log", args.tail_lines)
    _emit_json({"ok": True, "job": payload})
    return 0


def _command_list(args: argparse.Namespace) -> int:
    """Handle the ``list`` subcommand."""

    jobs_root = Path(args.jobs_root).expanduser().resolve()
    jobs: list[dict[str, Any]] = []
    if jobs_root.exists():
        for child in sorted(jobs_root.iterdir(), key=lambda path: path.name, reverse=True):
            if not child.is_dir() or child.is_symlink():
                continue
            meta_path = child / "meta.json"
            if not meta_path.exists():
                continue
            try:
                jobs.append(
                    _load_job_view(
                        jobs_root,
                        child.name,
                        stale_heartbeat_seconds=args.stale_heartbeat_seconds,
                        launch_grace_seconds=args.launch_grace_seconds,
                    )
                )
            except Exception:  # noqa: BLE001
                continue
            if len(jobs) >= args.limit:
                break
    _emit_json({"ok": True, "jobs": jobs})
    return 0


def _command_logs(args: argparse.Namespace) -> int:
    """Handle the ``logs`` subcommand."""

    jobs_root = Path(args.jobs_root).expanduser().resolve()
    try:
        job_directory = resolve_job_dir(jobs_root, args.job_id)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if not job_directory.exists():
        print(f"Unknown job_id: {args.job_id}", file=sys.stderr)
        return 1

    log_name = "stdout.log" if args.stream == "stdout" else "stderr.log"
    log_path = job_directory / log_name

    if not args.follow:
        for line in _tail_lines(log_path, args.lines):
            sys.stdout.write(line)
        sys.stdout.flush()
        return 0

    for line in _tail_lines(log_path, args.lines):
        sys.stdout.write(line)
    sys.stdout.flush()
    _stream_log_until_complete(
        job_directory,
        log_path,
        poll_seconds=args.poll_seconds,
        stale_heartbeat_seconds=args.stale_heartbeat_seconds,
        launch_grace_seconds=args.launch_grace_seconds,
    )
    return 0


def _command_capacity(args: argparse.Namespace) -> int:
    """Handle the ``capacity`` subcommand."""

    jobs_root = Path(args.jobs_root).expanduser().resolve()
    workers = _parse_workers(args.workers)
    _emit_json(
        {
            "ok": True,
            "workers": _worker_capacity_view(
                jobs_root,
                workers,
                max_tasks_per_worker=args.max_tasks_per_worker,
                stale_heartbeat_seconds=args.stale_heartbeat_seconds,
                launch_grace_seconds=args.launch_grace_seconds,
            ),
        }
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        prog="na49-cluster-agent",
        description=(
            "Run commands on worker nodes via SSH and track job "
            "state in a shared filesystem."
        ),
    )
    parser.add_argument(
        "--jobs-root",
        default=None,
        help=(
            "Directory for job metadata and logs. Defaults to 'jobs_root' from the "
            "user config, then ~/.na49_cluster_gw/jobs."
        ),
    )
    parser.add_argument(
        "--stale-heartbeat-seconds",
        type=float,
        default=None,
        help=(
            "Heartbeat age beyond which a running job counts as LOST. Applies "
            "to every subcommand. Defaults to configured "
            f"'{STALE_HEARTBEAT_SECONDS_CONFIG_KEY}', then {DEFAULT_STALE_HEARTBEAT_SECONDS}."
        ),
    )
    parser.add_argument(
        "--launch-grace-seconds",
        type=float,
        default=None,
        help=(
            "Grace period for jobs that never produced a heartbeat. Applies "
            "to every subcommand. Defaults to configured "
            f"'{LAUNCH_GRACE_SECONDS_CONFIG_KEY}', then {DEFAULT_LAUNCH_GRACE_SECONDS}."
        ),
    )

    sub = parser.add_subparsers(dest="subcommand", required=True)

    submit = sub.add_parser("submit", help="Submit a command to a worker node.")
    submit.add_argument("--workdir", default=".", help="Working directory for the command.")
    submit.add_argument(
        "--workers",
        default=None,
        help="CSV list of worker hostnames. Defaults to configured 'workers'.",
    )
    submit.add_argument("--worker", default=None, help="Force a specific worker.")
    submit.add_argument("--name", default="", help="Optional human-readable label.")
    submit.add_argument("--tasks", type=int, default=1, help="Reserved task slots for this job (default: 1).")
    submit.add_argument(
        "--max-tasks-per-worker",
        type=int,
        default=None,
        help=(
            "Maximum reserved task slots per worker. Defaults to configured "
            f"'max_tasks_per_worker', then {DEFAULT_MAX_TASKS_PER_WORKER}."
        ),
    )
    submit.add_argument(
        "--worker-bin-dir",
        default=None,
        help=(
            "Optional bin directory prepended to PATH on workers. "
            "Defaults to configured 'worker_bin_dir'."
        ),
    )
    submit.add_argument(
        "--wait-for-slot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait for worker capacity when no slot is immediately free (default: enabled).",
    )
    submit.add_argument(
        "--poll-seconds",
        type=float,
        default=5.0,
        help="Polling interval while waiting for a free worker slot.",
    )
    submit.add_argument(
        "--wait-timeout-seconds",
        type=float,
        default=None,
        help="Optional timeout while waiting for worker capacity.",
    )
    submit.add_argument("--cmd", default=None, help="Command string run with bash -lc.")
    submit.add_argument("--ssh-bin", default="ssh", help="SSH executable (default: ssh).")
    submit.add_argument(
        "--ssh-option",
        action="append",
        default=[],
        help="Additional option passed to SSH. Repeat for multiple values; defaults to configured 'ssh_options'.",
    )
    submit.add_argument("command", nargs=argparse.REMAINDER, help="Command tokens, optionally after '--'.")
    submit.set_defaults(handler=_command_submit)

    status = sub.add_parser("status", help="Get job status.")
    status.add_argument("job_id", help="Job ID from submit output.")
    status.set_defaults(handler=_command_status)

    result = sub.add_parser("result", help="Get status and output tail.")
    result.add_argument("job_id", help="Job ID from submit output.")
    result.add_argument("--tail-lines", type=int, default=40, help="Number of tail lines for stdout/stderr.")
    result.set_defaults(handler=_command_result)

    list_cmd = sub.add_parser("list", help="List recent jobs.")
    list_cmd.add_argument("--limit", type=int, default=20, help="Max jobs to return.")
    list_cmd.set_defaults(handler=_command_list)

    logs = sub.add_parser("logs", help="Print job logs.")
    logs.add_argument("job_id", help="Job ID from submit output.")
    logs.add_argument("--stream", choices=("stdout", "stderr"), default="stdout", help="Which log to print.")
    logs.add_argument("--lines", type=int, default=100, help="Initial tail lines.")
    logs.add_argument("--follow", action="store_true", help="Follow log stream until job completes.")
    logs.add_argument(
        "--poll-seconds",
        type=float,
        default=0.5,
        help="Polling interval while following logs.",
    )
    logs.set_defaults(handler=_command_logs)

    capacity = sub.add_parser("capacity", help="Show per-worker task capacity.")
    capacity.add_argument(
        "--workers",
        default=None,
        help="CSV list of worker hostnames. Defaults to configured 'workers'.",
    )
    capacity.add_argument(
        "--max-tasks-per-worker",
        type=int,
        default=None,
        help=(
            "Maximum reserved task slots per worker. Defaults to configured "
            f"'max_tasks_per_worker', then {DEFAULT_MAX_TASKS_PER_WORKER}."
        ),
    )
    capacity.set_defaults(handler=_command_capacity)

    reconcile_cmd = sub.add_parser(
        "reconcile",
        help=(
            "Apply stale-job liveness checks, materialize LOST verdicts, and "
            "optionally prune old terminal job directories."
        ),
    )
    reconcile_cmd.add_argument(
        "--probe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify recorded PIDs over SSH before trusting heartbeat staleness (default: enabled).",
    )
    reconcile_cmd.add_argument("--ssh-bin", default="ssh", help="SSH executable (default: ssh).")
    reconcile_cmd.add_argument(
        "--ssh-option",
        action="append",
        default=[],
        help="Additional option passed to SSH. Repeat for multiple values; defaults to configured 'ssh_options'.",
    )
    reconcile_cmd.add_argument(
        "--prune",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also delete terminal job directories older than --prune-age-days (default: disabled).",
    )
    reconcile_cmd.add_argument(
        "--prune-age-days",
        type=float,
        default=None,
        help=(
            "Retention age for terminal job directories when pruning. Defaults "
            f"to configured '{PRUNE_AGE_DAYS_CONFIG_KEY}', then {DEFAULT_PRUNE_AGE_DAYS}."
        ),
    )
    reconcile_cmd.set_defaults(handler=_command_reconcile)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the agent CLI."""

    config = load_user_config()
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.jobs_root = args.jobs_root or get_config_str(config, "jobs_root") or DEFAULT_JOBS_ROOT
    if hasattr(args, "workers") and not args.workers:
        configured_workers = get_config_list(config, "workers")
        if configured_workers:
            args.workers = ",".join(configured_workers)
    if hasattr(args, "max_tasks_per_worker") and args.max_tasks_per_worker is None:
        args.max_tasks_per_worker = get_config_int(
            config,
            "max_tasks_per_worker",
            DEFAULT_MAX_TASKS_PER_WORKER,
        )
    if hasattr(args, "worker_bin_dir") and not args.worker_bin_dir:
        args.worker_bin_dir = get_config_str(config, "worker_bin_dir")
    if hasattr(args, "ssh_option") and not args.ssh_option:
        args.ssh_option = get_config_list(config, "ssh_options") or []
    if hasattr(args, "stale_heartbeat_seconds") and args.stale_heartbeat_seconds is None:
        configured_stale = get_config_int(
            config,
            STALE_HEARTBEAT_SECONDS_CONFIG_KEY,
            DEFAULT_STALE_HEARTBEAT_SECONDS,
        )
        args.stale_heartbeat_seconds = float(
            DEFAULT_STALE_HEARTBEAT_SECONDS if configured_stale is None else configured_stale
        )
    if hasattr(args, "launch_grace_seconds") and args.launch_grace_seconds is None:
        configured_grace = get_config_int(
            config,
            LAUNCH_GRACE_SECONDS_CONFIG_KEY,
            DEFAULT_LAUNCH_GRACE_SECONDS,
        )
        args.launch_grace_seconds = float(
            DEFAULT_LAUNCH_GRACE_SECONDS if configured_grace is None else configured_grace
        )
    if hasattr(args, "prune_age_days") and args.prune_age_days is None:
        configured_prune_age = get_config_int(config, PRUNE_AGE_DAYS_CONFIG_KEY)
        args.prune_age_days = (
            float(configured_prune_age) if configured_prune_age is not None else DEFAULT_PRUNE_AGE_DAYS
        )
    try:
        return args.handler(args)
    except Exception as exc:  # noqa: BLE001
        _emit_json({"ok": False, "error": str(exc)}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
