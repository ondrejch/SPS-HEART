"""Nits from .collab/na49_code_review2.md: reconcile argv and job permissions."""

from __future__ import annotations

import json
import os
import stat
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from helpers.na49_gateway import cluster_agent
from helpers.na49_gateway import cluster_client


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _capture_run_json(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    captured: dict[str, list[str]] = {}

    def fake_run(self: cluster_client.ClusterGatewayClient, args: list[str]) -> dict:
        captured["args"] = list(args)
        return {"ok": True}

    monkeypatch.setattr(cluster_client.ClusterGatewayClient, "_run_json", fake_run)
    return captured


def test_reconcile_per_call_thresholds_parse_on_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_run_json(monkeypatch)
    client = cluster_client.ClusterGatewayClient(
        host="unused",
        agent_command="python3 unused.py",
        jobs_root="/jr",
        stale_heartbeat_seconds=10,
        launch_grace_seconds=20,
    )
    client.reconcile(stale_heartbeat_seconds=30, launch_grace_seconds=40, probe=False)

    args = captured["args"]
    assert "reconcile" in args
    assert args.index("--stale-heartbeat-seconds") < args.index("reconcile")
    assert args.index("--launch-grace-seconds") < args.index("reconcile")
    assert args.count("--stale-heartbeat-seconds") == 1
    assert args.count("--launch-grace-seconds") == 1
    parsed = cluster_agent._build_parser().parse_args(args)
    assert parsed.subcommand == "reconcile"
    assert parsed.stale_heartbeat_seconds == 30.0
    assert parsed.launch_grace_seconds == 40.0
    assert parsed.probe is False


def test_client_cli_reconcile_global_thresholds_parse_on_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_run_json(monkeypatch)
    monkeypatch.setattr(cluster_client, "load_user_config", lambda: {})
    rc = cluster_client.main(
        [
            "--host",
            "unused",
            "--agent-command",
            "python3 unused.py",
            "--jobs-root",
            "/jr",
            "--stale-heartbeat-seconds",
            "10",
            "--launch-grace-seconds",
            "20",
            "reconcile",
            "--no-probe",
        ]
    )
    assert rc == 0
    args = captured["args"]
    assert args.index("--stale-heartbeat-seconds") < args.index("reconcile")
    assert args.index("--launch-grace-seconds") < args.index("reconcile")
    parsed = cluster_agent._build_parser().parse_args(args)
    assert parsed.subcommand == "reconcile"
    assert parsed.stale_heartbeat_seconds == 10.0
    assert parsed.launch_grace_seconds == 20.0
    assert parsed.probe is False


def test_write_json_mode_0600_under_umask_022(tmp_path: Path) -> None:
    path = tmp_path / "meta.json"
    path.write_text("{}\n", encoding="utf-8")
    os.chmod(path, 0o644)
    old = os.umask(0o022)
    try:
        cluster_agent._write_json(path, {"job_id": "j1"})
        assert _mode(path) == 0o600
    finally:
        os.umask(old)


def test_reserved_job_records_are_owner_only_under_umask_022(tmp_path: Path) -> None:
    jobs_root = tmp_path / "jobs"
    args = types.SimpleNamespace(
        tasks=1,
        max_tasks_per_worker=64,
        wait_timeout_seconds=None,
        worker="w1",
        wait_for_slot=False,
        poll_seconds=0.05,
        workdir=str(tmp_path),
        name="perm-test",
        stale_heartbeat_seconds=None,
        launch_grace_seconds=None,
        worker_bin_dir=None,
    )
    old = os.umask(0o022)
    try:
        chosen, job_directory, run_script, _meta = cluster_agent._reserve_job_slot(
            args=args,
            jobs_root=jobs_root,
            workers=["w1"],
            command="echo secret-token",
        )
    finally:
        os.umask(old)

    assert chosen == "w1"
    assert _mode(jobs_root) == 0o700
    assert _mode(job_directory) == 0o700
    assert _mode(job_directory / "meta.json") == 0o600
    assert _mode(job_directory / "command.sh") == 0o600
    assert _mode(jobs_root / ".submit.lock") == 0o600
    assert _mode(run_script) == 0o700
    record = json.loads((job_directory / "meta.json").read_text(encoding="utf-8"))
    assert record["command"] == "echo secret-token"
