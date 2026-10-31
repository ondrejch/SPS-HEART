#!/usr/bin/env python3
"""Client helpers for invoking the NA49 cluster gateway agent over SSH.

Every SSH subprocess call carries an explicit timeout so a dead transport
fails the call instead of hanging it. The timeout defaults to
``DEFAULT_CLIENT_TIMEOUT_SECONDS`` and is overridable through the
``NA49_GW_TIMEOUT`` environment variable; it bounds each individual SSH
invocation, including a ``logs --follow`` stream. Expiry raises
``ClusterGatewayError`` naming the timeout and the environment variable.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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


TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELED", "SUBMIT_FAILED", "LOST"}
DEFAULT_MAX_TASKS_PER_WORKER = 64
# Per-call SSH timeout. Each individual SSH invocation (JSON calls, text
# calls, and log streaming) is bounded by this many seconds so a dead
# transport surfaces as an error instead of hanging. Override with the
# NA49_GW_TIMEOUT environment variable (a non-negative number of seconds).
DEFAULT_CLIENT_TIMEOUT_SECONDS = 60.0
CLIENT_TIMEOUT_ENV_VAR = "NA49_GW_TIMEOUT"
_UNSET = object()


def _subprocess_timeout() -> float:
    """Return the SSH timeout from ``NA49_GW_TIMEOUT`` or its default."""

    raw = os.environ.get(CLIENT_TIMEOUT_ENV_VAR)
    try:
        value = float(raw) if raw else DEFAULT_CLIENT_TIMEOUT_SECONDS
    except (TypeError, ValueError):
        return DEFAULT_CLIENT_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_CLIENT_TIMEOUT_SECONDS


def _remote_home_shell_path(path: str | Path) -> str:
    """Map a local home-relative path to a remote ``$HOME`` shell expression.

    If *path* is outside the current user's home directory, the expanded path is
    returned unchanged.
    """

    raw = str(path)
    if raw.startswith("$HOME/"):
        return f"~/{raw[6:]}"
    if raw == "$HOME":
        return "~"
    if raw.startswith("~/") or raw == "~":
        return raw

    expanded = Path(raw).expanduser()
    home = Path.home().resolve()
    try:
        rel = expanded.resolve().relative_to(home)
    except (OSError, RuntimeError, ValueError):
        return str(expanded)

    rel_text = rel.as_posix()
    if rel_text in ("", "."):
        return "~"
    return f"~/{rel_text}"


def _normalize_remote_path_arg(path: str | None) -> str | None:
    """Normalize a remote path argument to use the target user's home directory."""

    if path is None:
        return None
    return _remote_home_shell_path(path)


def _default_agent_command(config: dict[str, Any] | None = None) -> str:
    """Return the default remote command used to invoke the agent."""

    if config:
        configured_command = get_config_str(config, "agent_command")
        if configured_command:
            return configured_command
        remote_project_relpath = get_config_str(config, "remote_project_relpath")
        if remote_project_relpath:
            repo_rel = remote_project_relpath.strip()
            if repo_rel == "$HOME" or repo_rel == "~":
                repo_rel = ""
            elif repo_rel.startswith("$HOME/"):
                repo_rel = repo_rel[6:]
            elif repo_rel.startswith("~/"):
                repo_rel = repo_rel[2:]
            repo_root = _remote_home_shell_path(f"~/{repo_rel}" if repo_rel else "~")
            return f"python3 {repo_root}/helpers/na49_gateway/cluster_agent.py"
    agent_path = Path(__file__).resolve().with_name("cluster_agent.py")
    return f"python3 {_remote_home_shell_path(agent_path)}"


class ClusterGatewayError(RuntimeError):
    """Raised when communication with the remote agent fails."""


@dataclass
class ClusterGatewayClient:
    """Thin Python API around ``ssh <host> cluster_agent.py ...``."""

    host: str | None = None
    agent_command: str | None = None
    ssh_bin: str = "ssh"
    ssh_options: tuple[str, ...] = ()
    jobs_root: str | None = None
    # Global agent options forwarded ahead of every subcommand so status,
    # list, capacity, wait, result, logs, submit, and reconcile agree on
    # one liveness verdict. Defaults resolve on the agent side.
    stale_heartbeat_seconds: float | None = None
    launch_grace_seconds: float | None = None

    def _ssh_base_command(self) -> list[str]:
        """Return the SSH command prefix for remote agent invocations."""

        if not self.host:
            raise ClusterGatewayError(
                "No SSH host configured. Pass --host or set 'host' in "
                f"{config_path_hint()}."
            )
        if not self.agent_command:
            raise ClusterGatewayError(
                "No remote agent command configured. Pass --agent-command or set "
                f"'agent_command'/'remote_project_relpath' in {config_path_hint()}."
            )
        return [self.ssh_bin, *self.ssh_options, self.host]

    def _compose_remote_command(self, args: list[str]) -> str:
        """Compose a shell-safe remote command line for the SSH transport."""

        if not self.agent_command:
            raise ClusterGatewayError("Remote agent command is not configured.")
        quoted_args = " ".join(shlex.quote(item) for item in args)
        if quoted_args:
            return f"{self.agent_command} {quoted_args}"
        return self.agent_command

    def _run_json(self, args: list[str]) -> dict[str, Any]:
        """Execute a remote agent command and parse its JSON response."""

        cmd = self._ssh_base_command() + [self._compose_remote_command(args)]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, check=False, timeout=_subprocess_timeout()
            )
        except subprocess.TimeoutExpired as exc:
            raise ClusterGatewayError(
                "Remote agent call timed out after "
                f"{_subprocess_timeout():g} s (set {CLIENT_TIMEOUT_ENV_VAR} to raise "
                f"the limit): cmd={cmd!r}"
            ) from exc
        if proc.returncode != 0:
            raise ClusterGatewayError(
                "Remote agent call failed: "
                f"rc={proc.returncode}, cmd={cmd!r}, stdout={proc.stdout.strip()!r}, stderr={proc.stderr.strip()!r}"
            )
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise ClusterGatewayError(
                f"Remote agent returned non-JSON output for cmd={cmd!r}: {proc.stdout!r}"
            ) from exc
        if not payload.get("ok", False):
            raise ClusterGatewayError(f"Remote agent reported failure: {payload!r}")
        return payload

    def _run_text(self, args: list[str], *, capture_output: bool = True) -> str | int:
        """Execute a remote agent command that returns plain text output."""

        cmd = self._ssh_base_command() + [self._compose_remote_command(args)]
        timeout = _subprocess_timeout()
        if capture_output:
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, check=False, timeout=timeout
                )
            except subprocess.TimeoutExpired as exc:
                raise ClusterGatewayError(
                    "Remote logs call timed out after "
                    f"{timeout:g} s (set {CLIENT_TIMEOUT_ENV_VAR} to raise the limit): "
                    f"cmd={cmd!r}"
                ) from exc
            if proc.returncode != 0:
                raise ClusterGatewayError(
                    "Remote logs call failed: "
                    f"rc={proc.returncode}, cmd={cmd!r}, stdout={proc.stdout.strip()!r}, stderr={proc.stderr.strip()!r}"
                )
            return proc.stdout
        try:
            proc = subprocess.run(cmd, check=False, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise ClusterGatewayError(
                "Remote logs stream timed out after "
                f"{timeout:g} s (set {CLIENT_TIMEOUT_ENV_VAR} to raise the limit): "
                f"cmd={cmd!r}"
            ) from exc
        return proc.returncode

    def _agent_args(
        self,
        *items: str,
        stale_heartbeat_seconds: Any = _UNSET,
        launch_grace_seconds: Any = _UNSET,
    ) -> list[str]:
        """Prefix agent arguments with the global options and jobs-root override.

        The threshold options are accepted by the agent before any
        subcommand and apply to all of them, so they are emitted first.
        Callers may override the instance thresholds for a single
        invocation; those overrides still appear before *items*.
        """

        if stale_heartbeat_seconds is _UNSET:
            stale_heartbeat_seconds = self.stale_heartbeat_seconds
        if launch_grace_seconds is _UNSET:
            launch_grace_seconds = self.launch_grace_seconds
        args: list[str] = []
        if self.jobs_root:
            normalized_jobs_root = _normalize_remote_path_arg(self.jobs_root)
            if normalized_jobs_root is not None:
                args.extend(["--jobs-root", normalized_jobs_root])
        if stale_heartbeat_seconds is not None:
            args.extend(["--stale-heartbeat-seconds", str(stale_heartbeat_seconds)])
        if launch_grace_seconds is not None:
            args.extend(["--launch-grace-seconds", str(launch_grace_seconds)])
        args.extend(items)
        return args

    def submit(
        self,
        command: str | Iterable[str],
        *,
        workdir: str = ".",
        workers: Iterable[str] | None = None,
        worker: str | None = None,
        name: str | None = None,
        tasks: int = 1,
        max_tasks_per_worker: int = 64,
        worker_bin_dir: str | None = None,
        wait_for_slot: bool = True,
        poll_seconds: float = 5.0,
        wait_timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Submit a command to the remote worker pool."""

        args = self._agent_args(
            "submit",
            "--workdir",
            _normalize_remote_path_arg(workdir) or workdir,
            "--tasks",
            str(tasks),
            "--max-tasks-per-worker",
            str(max_tasks_per_worker),
            "--poll-seconds",
            str(poll_seconds),
        )
        if worker_bin_dir:
            args.extend(["--worker-bin-dir", worker_bin_dir])
        if not wait_for_slot:
            args.append("--no-wait-for-slot")
        if wait_timeout_seconds is not None:
            args.extend(["--wait-timeout-seconds", str(wait_timeout_seconds)])
        if workers:
            args.extend(["--workers", ",".join(workers)])
        if worker:
            args.extend(["--worker", worker])
        if name:
            args.extend(["--name", name])

        if isinstance(command, str):
            args.extend(["--cmd", command])
        else:
            tokens = list(command)
            if not tokens:
                raise ValueError("command cannot be empty")
            args.append("--")
            args.extend(tokens)
        payload = self._run_json(args)
        return payload["job"] if "job" in payload else payload

    def status(self, job_id: str) -> dict[str, Any]:
        """Return status information for *job_id*."""

        payload = self._run_json(self._agent_args("status", job_id))
        return payload["job"]

    def result(self, job_id: str, *, tail_lines: int = 40) -> dict[str, Any]:
        """Return status and output tails for *job_id*."""

        payload = self._run_json(self._agent_args("result", job_id, "--tail-lines", str(tail_lines)))
        return payload["job"]

    def list(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent jobs known to the remote agent."""

        payload = self._run_json(self._agent_args("list", "--limit", str(limit)))
        return payload["jobs"]

    def capacity(self, *, workers: Iterable[str] | None = None, max_tasks_per_worker: int = 64) -> list[dict[str, Any]]:
        """Return the current per-worker task-capacity view."""

        args = self._agent_args("capacity", "--max-tasks-per-worker", str(max_tasks_per_worker))
        if workers:
            args.extend(["--workers", ",".join(workers)])
        payload = self._run_json(args)
        return payload["workers"]

    def reconcile(
        self,
        *,
        probe: bool = True,
        prune: bool = False,
        prune_age_days: float | None = None,
        stale_heartbeat_seconds: float | None = None,
        launch_grace_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Run stale-job recovery on the remote agent and return its summary.

        ``stale_heartbeat_seconds`` / ``launch_grace_seconds`` are global
        agent options. Instance values are emitted by :meth:`_agent_args`
        ahead of the subcommand; the explicit arguments here override
        them for this call only, still ahead of ``reconcile``.
        """

        extra: dict[str, Any] = {}
        if stale_heartbeat_seconds is not None:
            extra["stale_heartbeat_seconds"] = stale_heartbeat_seconds
        if launch_grace_seconds is not None:
            extra["launch_grace_seconds"] = launch_grace_seconds
        args = self._agent_args("reconcile", **extra)
        if not probe:
            args.append("--no-probe")
        if prune:
            args.append("--prune")
        if prune_age_days is not None:
            args.extend(["--prune-age-days", str(prune_age_days)])
        payload = self._run_json(args)
        return payload

    def logs(
        self,
        job_id: str,
        *,
        stream: str = "stdout",
        lines: int = 100,
        follow: bool = False,
        poll_seconds: float = 0.5,
    ) -> str | int:
        """Return or stream logs for *job_id*."""

        args = self._agent_args(
            "logs",
            job_id,
            "--stream",
            stream,
            "--lines",
            str(lines),
            "--poll-seconds",
            str(poll_seconds),
        )
        if follow:
            args.append("--follow")
            return self._run_text(args, capture_output=False)
        return str(self._run_text(args, capture_output=True))

    def wait(
        self,
        job_id: str,
        *,
        poll_seconds: float = 5.0,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Poll job status until *job_id* reaches a terminal state."""

        start = time.monotonic()
        while True:
            job = self.status(job_id)
            if job.get("state") in TERMINAL_STATES:
                return job
            if timeout_seconds is not None and (time.monotonic() - start) > timeout_seconds:
                raise TimeoutError(f"Timed out waiting for job {job_id}")
            time.sleep(max(poll_seconds, 0.05))


def _print_json(payload: dict[str, Any]) -> None:
    """Write a JSON payload to stdout."""

    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        prog="na49-cluster-client",
        description=(
            "Local SSH client for the remote NA49 cluster gateway agent. "
            f"Each SSH call is bounded by {DEFAULT_CLIENT_TIMEOUT_SECONDS:g} s; "
            f"override with the {CLIENT_TIMEOUT_ENV_VAR} environment variable."
        ),
    )
    parser.add_argument(
        "--host",
        default=None,
        help="SSH host alias where the agent runs. Defaults to configured 'host'.",
    )
    parser.add_argument(
        "--agent-command",
        default=None,
        help=(
            "Remote agent command. Defaults to configured 'agent_command' "
            "or a repo-relative path derived from 'remote_project_relpath'."
        ),
    )
    parser.add_argument("--ssh-bin", default="ssh", help="SSH executable.")
    parser.add_argument(
        "--ssh-option",
        action="append",
        default=[],
        help="Additional option passed to SSH. Repeat for multiple values; defaults to configured 'ssh_options'.",
    )
    parser.add_argument(
        "--jobs-root",
        default=None,
        help="Override remote jobs root. Defaults to configured 'jobs_root'.",
    )
    parser.add_argument(
        "--stale-heartbeat-seconds",
        type=float,
        default=None,
        help="Heartbeat age beyond which a running job counts as LOST (forwarded to every agent subcommand).",
    )
    parser.add_argument(
        "--launch-grace-seconds",
        type=float,
        default=None,
        help="Grace period for jobs that never produced a heartbeat (forwarded to every agent subcommand).",
    )

    sub = parser.add_subparsers(dest="subcommand", required=True)

    submit = sub.add_parser("submit", help="Submit a command to the remote worker pool.")
    submit.add_argument("--workdir", default=".", help="Working directory on the remote side.")
    submit.add_argument("--workers", default=None, help="CSV list of workers.")
    submit.add_argument("--worker", default=None, help="Force a specific worker.")
    submit.add_argument("--name", default=None, help="Optional job label.")
    submit.add_argument("--tasks", type=int, default=1, help="Reserved task slots for this job.")
    submit.add_argument(
        "--max-tasks-per-worker",
        type=int,
        default=None,
        help=(
            "Maximum task slots allowed per worker. Defaults to configured "
            f"'max_tasks_per_worker', then {DEFAULT_MAX_TASKS_PER_WORKER}."
        ),
    )
    submit.add_argument(
        "--wait-for-slot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait for worker capacity if needed (default: enabled).",
    )
    submit.add_argument("--poll-seconds", type=float, default=5.0, help="Polling interval while waiting for slots.")
    submit.add_argument(
        "--wait-timeout-seconds",
        type=float,
        default=None,
        help="Optional timeout while waiting for worker capacity.",
    )
    submit.add_argument(
        "--worker-bin-dir",
        default=None,
        help="Optional bin directory prepended to PATH on workers.",
    )
    submit.add_argument("--cmd", default=None, help="Command string to run remotely.")
    submit.add_argument("command", nargs=argparse.REMAINDER, help="Command tokens, optionally after '--'.")

    status = sub.add_parser("status", help="Get job status.")
    status.add_argument("job_id")

    result = sub.add_parser("result", help="Get job result and tail.")
    result.add_argument("job_id")
    result.add_argument("--tail-lines", type=int, default=40)

    list_cmd = sub.add_parser("list", help="List recent jobs.")
    list_cmd.add_argument("--limit", type=int, default=20)

    capacity = sub.add_parser("capacity", help="Show current worker task capacity.")
    capacity.add_argument("--workers", default=None, help="CSV list of workers.")
    capacity.add_argument(
        "--max-tasks-per-worker",
        type=int,
        default=None,
        help=(
            "Maximum task slots allowed per worker. Defaults to configured "
            f"'max_tasks_per_worker', then {DEFAULT_MAX_TASKS_PER_WORKER}."
        ),
    )

    logs = sub.add_parser("logs", help="Print logs for a job.")
    logs.add_argument("job_id")
    logs.add_argument("--stream", choices=("stdout", "stderr"), default="stdout")
    logs.add_argument("--lines", type=int, default=100)
    logs.add_argument("--follow", action="store_true")
    logs.add_argument("--poll-seconds", type=float, default=0.5)

    wait = sub.add_parser("wait", help="Wait until a job finishes.")
    wait.add_argument("job_id")
    wait.add_argument("--poll-seconds", type=float, default=5.0)
    wait.add_argument("--timeout-seconds", type=float, default=None)

    reconcile = sub.add_parser(
        "reconcile",
        help="Run stale-job recovery (LOST materialization, optional prune) on the remote agent.",
    )
    reconcile.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip SSH PID liveness probes; rely on heartbeat files only.",
    )
    reconcile.add_argument("--prune", action="store_true", help="Also delete old terminal job directories.")
    reconcile.add_argument("--prune-age-days", type=float, default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the client CLI."""

    config = load_user_config()
    parser = _build_parser()
    args = parser.parse_args(argv)
    client = ClusterGatewayClient(
        host=args.host or get_config_str(config, "host"),
        agent_command=args.agent_command or _default_agent_command(config),
        ssh_bin=args.ssh_bin,
        ssh_options=tuple(args.ssh_option or get_config_list(config, "ssh_options") or []),
        jobs_root=args.jobs_root or get_config_str(config, "jobs_root"),
        stale_heartbeat_seconds=args.stale_heartbeat_seconds,
        launch_grace_seconds=args.launch_grace_seconds,
    )
    configured_workers = get_config_list(config, "workers")
    configured_max_tasks = get_config_int(
        config,
        "max_tasks_per_worker",
        DEFAULT_MAX_TASKS_PER_WORKER,
    )
    configured_worker_bin_dir = get_config_str(config, "worker_bin_dir")

    try:
        if args.subcommand == "submit":
            workers = [w.strip() for w in args.workers.split(",") if w.strip()] if args.workers else configured_workers
            max_tasks_per_worker = (
                args.max_tasks_per_worker
                if args.max_tasks_per_worker is not None
                else configured_max_tasks
            )
            if args.cmd and args.command:
                raise ValueError("Use either --cmd or positional command tokens, not both.")
            if args.cmd:
                job = client.submit(
                    args.cmd,
                    workdir=args.workdir,
                    workers=workers,
                    worker=args.worker,
                    name=args.name,
                    tasks=args.tasks,
                    max_tasks_per_worker=max_tasks_per_worker,
                    worker_bin_dir=args.worker_bin_dir or configured_worker_bin_dir,
                    wait_for_slot=args.wait_for_slot,
                    poll_seconds=args.poll_seconds,
                    wait_timeout_seconds=args.wait_timeout_seconds,
                )
            else:
                tokens = args.command
                if tokens and tokens[0] == "--":
                    tokens = tokens[1:]
                if not tokens:
                    raise ValueError("No command provided.")
                job = client.submit(
                    tokens,
                    workdir=args.workdir,
                    workers=workers,
                    worker=args.worker,
                    name=args.name,
                    tasks=args.tasks,
                    max_tasks_per_worker=max_tasks_per_worker,
                    worker_bin_dir=args.worker_bin_dir or configured_worker_bin_dir,
                    wait_for_slot=args.wait_for_slot,
                    poll_seconds=args.poll_seconds,
                    wait_timeout_seconds=args.wait_timeout_seconds,
                )
            _print_json(job)
            return 0

        if args.subcommand == "status":
            _print_json(client.status(args.job_id))
            return 0

        if args.subcommand == "result":
            _print_json(client.result(args.job_id, tail_lines=args.tail_lines))
            return 0

        if args.subcommand == "list":
            _print_json({"jobs": client.list(limit=args.limit)})
            return 0

        if args.subcommand == "capacity":
            workers = [w.strip() for w in args.workers.split(",") if w.strip()] if args.workers else configured_workers
            max_tasks_per_worker = (
                args.max_tasks_per_worker
                if args.max_tasks_per_worker is not None
                else configured_max_tasks
            )
            _print_json({"workers": client.capacity(workers=workers, max_tasks_per_worker=max_tasks_per_worker)})
            return 0

        if args.subcommand == "wait":
            _print_json(
                client.wait(
                    args.job_id,
                    poll_seconds=args.poll_seconds,
                    timeout_seconds=args.timeout_seconds,
                )
            )
            return 0

        if args.subcommand == "reconcile":
            _print_json(
                client.reconcile(
                    probe=not args.no_probe,
                    prune=args.prune,
                    prune_age_days=args.prune_age_days,
                )
            )
            return 0

        if args.subcommand == "logs":
            result = client.logs(
                args.job_id,
                stream=args.stream,
                lines=args.lines,
                follow=args.follow,
                poll_seconds=args.poll_seconds,
            )
            if isinstance(result, str):
                sys.stdout.write(result)
            return 0 if result in (None, 0, "") or isinstance(result, str) else int(result)
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 2

    raise RuntimeError(f"Unhandled subcommand: {args.subcommand}")


if __name__ == "__main__":
    raise SystemExit(main())
