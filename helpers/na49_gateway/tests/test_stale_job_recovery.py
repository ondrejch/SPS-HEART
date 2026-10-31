"""Stale-job recovery tests for the NA49 cluster gateway.

Every test runs against a fake jobs directory with marker files. No live SSH
connection or cluster submission is performed; the only subprocesses used are
local ``bash``/``sleep`` calls that emulate worker wrapper behavior.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from helpers.na49_gateway import cluster_agent
from helpers.na49_gateway import cluster_client


def _iso(epoch: float) -> str:
    """Format *epoch* the way the gateway records timestamps."""

    return (
        datetime.fromtimestamp(epoch, tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def make_job(
    root: Path,
    job_id: str = "20260826T000000Z-abcdef01",
    *,
    worker: str = "w1",
    meta: dict | None = None,
    last_seen_epoch: float | None = None,
    exit_code: int | None = None,
    canceled: bool = False,
    started: bool = False,
    finished_epoch: float | None = None,
) -> Path:
    """Create a fake job directory with marker files."""

    directory = root / job_id
    directory.mkdir(parents=True)
    record = {
        "job_id": job_id,
        "name": "",
        "worker": worker,
        "tasks": 2,
        "submitted_at": _iso(time.time()),
        "state": "RUNNING",
    }
    if meta:
        record.update(meta)
    (directory / "meta.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    if last_seen_epoch is not None:
        (directory / "last_seen").write_text(f"{int(last_seen_epoch)}\n")
    if exit_code is not None:
        (directory / "exit_code").write_text(f"{exit_code}\n")
    if canceled:
        (directory / "canceled_at").write_text(_iso(time.time()) + "\n")
    if started:
        (directory / "started_at").write_text(_iso(time.time() - 60) + "\n")
        (directory / "worker_pid").write_text("12345\n")
        (directory / "hostname").write_text(f"{worker}\n")
    if finished_epoch is not None:
        (directory / "finished_at").write_text(_iso(finished_epoch) + "\n")
    return directory


def read_meta(directory: Path) -> dict:
    return json.loads((directory / "meta.json").read_text())


# ---------------------------------------------------------------------------
# _compute_state liveness ladder
# ---------------------------------------------------------------------------


def test_fresh_heartbeat_without_probe_is_running(tmp_path: Path) -> None:
    directory = make_job(tmp_path, last_seen_epoch=time.time() - 5)
    assert cluster_agent._compute_state(directory) == ("RUNNING", None)


def test_stale_heartbeat_marks_lost_without_probe(tmp_path: Path) -> None:
    now = time.time()
    directory = make_job(
        tmp_path,
        last_seen_epoch=now - 10_000,
        meta={"launched_at": _iso(now - 10_000)},
    )
    state, reason = cluster_agent._running_or_lost(
        directory, read_meta(directory), now=now
    )
    assert state == "LOST"
    assert reason == "stale_heartbeat"
    assert cluster_agent._compute_state(directory)[0] == "LOST"


def test_missing_exit_code_with_dead_local_pid_is_lost(tmp_path: Path) -> None:
    """Required coverage: missing exit_code + dead PID resolves to LOST.

    Uses a genuinely dead local PID (a finished child process); no mocking of
    the liveness check itself.
    """

    proc = subprocess.Popen(["sleep", "0"])
    proc.wait()
    now = time.time()
    directory = make_job(
        tmp_path,
        job_id="dead-pid-job",
        meta={"state": "RUNNING", "submitted_at": _iso(now)},
    )
    (directory / "started_at").write_text(_iso(now) + "\n")
    (directory / "hostname").write_text(socket.gethostname() + "\n")
    (directory / "worker_pid").write_text(f"{proc.pid}\n")

    # Without a heartbeat and inside the launch grace window the job would be
    # RUNNING on file evidence alone; the confirmed-dead PID must win.
    assert cluster_agent._compute_state(directory)[0] == "LOST"
    _, reason = cluster_agent._running_or_lost(directory, read_meta(directory))
    assert reason == "dead_pid"


def test_alive_local_pid_with_fresh_heartbeat_is_running(tmp_path: Path) -> None:
    proc = subprocess.Popen(["sleep", "30"])
    try:
        directory = make_job(
            tmp_path,
            job_id="alive-job",
            last_seen_epoch=time.time() - 2,
            started=True,
        )
        (directory / "hostname").write_text(socket.gethostname() + "\n")
        (directory / "worker_pid").write_text(f"{proc.pid}\n")
        assert cluster_agent._compute_state(directory) == ("RUNNING", None)
    finally:
        proc.terminate()
        proc.wait()


def test_confirmed_dead_remote_pid_beats_fresh_heartbeat(tmp_path: Path) -> None:
    directory = make_job(
        tmp_path,
        last_seen_epoch=time.time() - 5,
        meta={"launcher_pid": 424242},
    )
    probes: list[tuple[str, int]] = []

    def fake_probe(worker: str, pid: int) -> bool | None:
        probes.append((worker, pid))
        return False

    assert cluster_agent._compute_state(directory, pid_probe=fake_probe)[0] == "LOST"
    assert probes == [("w1", 424242)]


def test_ssh_probe_targets_submit_alias_not_hostname_file(tmp_path: Path) -> None:
    """SSH PID probes dial meta['worker'], never the wrapper hostname file.

    The wrapper's ``hostname`` marker records the machine whose PID
    namespace the recorded PID belongs to; it can differ from the submit
    alias and must not be used as an SSH destination.
    """

    directory = make_job(
        tmp_path,
        last_seen_epoch=time.time() - 5,
        meta={"launcher_pid": 424242},
    )
    (directory / "hostname").write_text("not-the-worker\n")

    seen: list[tuple[str, int]] = []

    def fake_probe(worker: str, pid: int) -> bool | None:
        seen.append((worker, pid))
        return False

    state = cluster_agent._compute_state(directory, pid_probe=fake_probe)[0]
    assert state == "LOST"
    assert seen == [("w1", 424242)]


def test_local_pid_check_uses_hostname_file(tmp_path: Path) -> None:
    """The local namespace check keys on the hostname file alone.

    With no probe supplied, a job whose wrapper hostname matches this host
    gets an os.kill liveness check even when meta['worker'] names some
    other alias.
    """

    proc = subprocess.Popen(["sleep", "0"])
    proc.wait()
    now = time.time()
    directory = make_job(
        tmp_path,
        job_id="local-alias-job",
        meta={
            "state": "RUNNING",
            "submitted_at": _iso(now),
            "worker": "some-remote-alias",
        },
    )
    (directory / "started_at").write_text(_iso(now) + "\n")
    (directory / "hostname").write_text(socket.gethostname() + "\n")
    (directory / "worker_pid").write_text(f"{proc.pid}\n")

    assert cluster_agent._compute_state(directory)[0] == "LOST"
    _, reason = cluster_agent._running_or_lost(directory, read_meta(directory))
    assert reason == "dead_pid"


def test_live_pid_vetoes_stale_heartbeat(tmp_path: Path) -> None:
    """A worker process confirmed alive by the probe is RUNNING even when
    its heartbeat is ten times past the stale threshold: a shared-filesystem
    stall delays every heartbeat write at once (the 2026-09-24 R45 LOST
    wave, 122 resubmissions of live jobs; FAB-05)."""

    directory = make_job(
        tmp_path,
        last_seen_epoch=time.time() - 10 * cluster_agent.DEFAULT_STALE_HEARTBEAT_SECONDS,
        meta={"launcher_pid": 424242},
    )
    state, reason = cluster_agent._running_or_lost(
        directory, read_meta(directory), pid_probe=lambda w, p: True)
    assert (state, reason) == ("RUNNING", None)
    assert cluster_agent._compute_state(directory, pid_probe=lambda w, p: True)[0] == "RUNNING"


def test_live_pid_vetoes_missing_heartbeat_past_launch_grace(tmp_path: Path) -> None:
    """The same veto for a job that never wrote a heartbeat: with the
    process alive past the launch grace window the job stays RUNNING."""

    now = time.time()
    directory = make_job(
        tmp_path,
        meta={
            "launcher_pid": 424242,
            "submitted_at": _iso(now - 3 * cluster_agent.DEFAULT_LAUNCH_GRACE_SECONDS),
            "launched_at": _iso(now - 3 * cluster_agent.DEFAULT_LAUNCH_GRACE_SECONDS),
        },
    )
    state, reason = cluster_agent._running_or_lost(
        directory, read_meta(directory), pid_probe=lambda w, p: True)
    assert (state, reason) == ("RUNNING", None)
    # without the live probe the same job is LOST past the grace window
    state, reason = cluster_agent._running_or_lost(
        directory, read_meta(directory), pid_probe=lambda w, p: None)
    assert state == "LOST" and reason == "no_heartbeat_past_launch_grace"


def test_unknown_probe_result_keeps_fresh_job_running(tmp_path: Path) -> None:
    """An inconclusive probe (SSH hiccup) must never mark a job LOST alone."""

    directory = make_job(
        tmp_path,
        last_seen_epoch=time.time() - 5,
        meta={"launcher_pid": 999999},
    )
    assert cluster_agent._compute_state(directory, pid_probe=lambda w, p: None)[0] == "RUNNING"


def test_exit_marker_beats_stale_heartbeat_and_cancel_precedence(tmp_path: Path) -> None:
    failed = make_job(
        tmp_path,
        job_id="failed-stale",
        exit_code=3,
        last_seen_epoch=time.time() - 9000,
    )
    assert cluster_agent._compute_state(failed) == ("FAILED", 3)

    canceled = make_job(
        tmp_path,
        job_id="canceled-done",
        exit_code=0,
        canceled=True,
        last_seen_epoch=time.time() - 5,
    )
    assert cluster_agent._compute_state(canceled)[0] == "CANCELED"


def test_interrupted_submit_does_not_reserve_slots_forever(tmp_path: Path) -> None:
    """A submit killed after writing meta.json frees its slots after grace."""

    now = time.time()
    stale_submit = make_job(
        tmp_path,
        job_id="stale-submit",
        meta={"state": "SUBMITTED", "submitted_at": _iso(now - 7200), "tasks": 8},
    )
    young_submit = make_job(
        tmp_path,
        job_id="young-submit",
        meta={"state": "SUBMITTED", "submitted_at": _iso(now - 60), "tasks": 8},
    )

    assert cluster_agent._compute_state(stale_submit)[0] == "LOST"
    assert cluster_agent._compute_state(young_submit)[0] == "RUNNING"

    counts = cluster_agent._active_tasks_by_worker(tmp_path)
    assert counts == {"w1": 8}

    picked = cluster_agent._pick_worker(
        tmp_path, ["w1"], requested_tasks=56, max_tasks_per_worker=64
    )
    assert picked == "w1"


def test_launch_grace_override_keeps_recent_unlaunched_job_running(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Thresholds load once in main() and reach every read path.

    A submit killed 1000 s ago is LOST under the default 900 s grace but
    stays RUNNING (slots reserved) when --launch-grace-seconds=14400 is
    passed at the top level.
    """

    now = time.time()
    make_job(
        tmp_path,
        job_id="20260826T000000Z-00000001",
        meta={"state": "SUBMITTED", "submitted_at": _iso(now - 1000), "tasks": 4},
    )

    # Defaults: derived LOST frees the slots.
    assert cluster_agent.main(["--jobs-root", str(tmp_path), "capacity", "--workers", "w1"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["workers"][0]["active_tasks"] == 0

    argv = ["--jobs-root", str(tmp_path), "--launch-grace-seconds", "14400"]

    # Capacity honors the extended grace.
    assert cluster_agent.main(argv + ["capacity", "--workers", "w1"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["workers"][0]["active_tasks"] == 4

    # list and status honor it too.
    assert cluster_agent.main(argv + ["list"]) == 0
    jobs = json.loads(capsys.readouterr().out)["jobs"]
    assert [job["state"] for job in jobs] == ["RUNNING"]

    assert cluster_agent.main(argv + ["status", "20260826T000000Z-00000001"]) == 0
    job = json.loads(capsys.readouterr().out)["job"]
    assert job["state"] == "RUNNING"

    # Direct accounting/view helpers accept the same overrides.
    counts = cluster_agent._active_tasks_by_worker(
        tmp_path, launch_grace_seconds=14400
    )
    assert counts == {"w1": 4}
    view = cluster_agent._load_job_view(tmp_path, "20260826T000000Z-00000001", launch_grace_seconds=14400)
    assert view["state"] == "RUNNING"

    # Prune decisions follow the same thresholds: an active job under the
    # extended grace is never pruned, while default grace makes it terminal.
    pruned_active = cluster_agent._prune_terminal_jobs(
        tmp_path,
        older_than_epoch=now + 60.0,
        launch_grace_seconds=14400,
    )
    assert pruned_active == []
    assert (tmp_path / "20260826T000000Z-00000001").exists()


def test_prune_honors_default_thresholds_for_stale_records(tmp_path: Path) -> None:
    """With default thresholds the same record is terminal and prunable."""

    now = time.time()
    directory = make_job(
        tmp_path,
        job_id="stale-prunable",
        meta={"state": "SUBMITTED", "submitted_at": _iso(now - 1000), "tasks": 4},
    )
    pruned = cluster_agent._prune_terminal_jobs(tmp_path, older_than_epoch=now + 60.0)
    assert pruned == ["stale-prunable"]
    assert not directory.exists()


def test_capacity_excludes_all_terminal_states(tmp_path: Path) -> None:
    now = time.time()
    make_job(
        tmp_path,
        job_id="j-completed",
        exit_code=0,
        finished_epoch=now - 30,
        last_seen_epoch=now - 5,
    )
    make_job(
        tmp_path,
        job_id="j-failed",
        exit_code=2,
        finished_epoch=now - 30,
        last_seen_epoch=now - 5,
    )
    make_job(tmp_path, job_id="j-canceled", canceled=True)
    make_job(
        tmp_path,
        job_id="j-submit-failed",
        meta={"state": "SUBMIT_FAILED"},
        last_seen_epoch=now - 5,
    )
    make_job(
        tmp_path,
        job_id="j-lost",
        meta={"state": "LOST"},
        last_seen_epoch=now - 5000,
    )
    make_job(tmp_path, job_id="j-running", last_seen_epoch=now - 5)

    assert cluster_agent._active_jobs_by_worker(tmp_path) == {"w1": 1}
    assert cluster_agent._active_tasks_by_worker(tmp_path) == {"w1": 2}


def test_malformed_meta_json_does_not_crash_listing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_job(tmp_path, job_id="20260826T000000Z-00000002", last_seen_epoch=time.time() - 3)
    bad_directory = tmp_path / "20260826T000000Z-00000003"
    bad_directory.mkdir()
    (bad_directory / "meta.json").write_text("{not valid json!!")

    rc = cluster_agent.main(["--jobs-root", str(tmp_path), "list"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["ok"] is True
    assert [job["job_id"] for job in payload["jobs"]] == ["20260826T000000Z-00000002"]

    rc = cluster_agent.main(
        ["--jobs-root", str(tmp_path), "capacity", "--workers", "w1"]
    )
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["workers"][0]["active_tasks"] == 2

    assert cluster_agent._active_tasks_by_worker(tmp_path) == {"w1": 2}
    # Deriving state from a corrupt record must not raise either.
    state, _ = cluster_agent._compute_state(bad_directory)
    assert state in {"RUNNING", "LOST"}

    # A direct status query reports a clean error instead of crashing.
    rc = cluster_agent.main(["--jobs-root", str(tmp_path), "status", "20260826T000000Z-00000003"])
    captured = capsys.readouterr()
    error_payload = json.loads(captured.err)
    assert rc == 2
    assert error_payload["ok"] is False


# ---------------------------------------------------------------------------
# atomic meta writes
# ---------------------------------------------------------------------------


def test_write_json_atomic_no_temp_residue_and_cleanup_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "meta.json"
    cluster_agent._write_json(target, {"a": 1, "b": "x"})
    assert json.loads(target.read_text()) == {"a": 1, "b": "x"}
    assert list(tmp_path.iterdir()) == [target]

    with pytest.raises(TypeError):
        cluster_agent._write_json(target, {"bad": {1, 2}})
    # The previous complete file survives and no temp files are left behind.
    assert json.loads(target.read_text()) == {"a": 1, "b": "x"}
    assert list(tmp_path.iterdir()) == [target]


# ---------------------------------------------------------------------------
# reconcile command
# ---------------------------------------------------------------------------


def test_reconcile_materializes_lost_and_preserves_others(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    now = time.time()
    stale = make_job(
        tmp_path,
        job_id="j-stale",
        last_seen_epoch=now - 5000,
        meta={"launched_at": _iso(now - 5000)},
    )
    healthy = make_job(tmp_path, job_id="j-ok", last_seen_epoch=now - 5)
    done = make_job(
        tmp_path,
        job_id="j-done",
        exit_code=0,
        finished_epoch=now - 30,
        last_seen_epoch=now - 40,
    )

    argv = ["--jobs-root", str(tmp_path), "reconcile", "--no-probe"]
    assert cluster_agent.main(argv) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["ok"] is True
    assert summary["scanned"] == 3
    assert summary["lost_materialized"] == ["j-stale"]
    assert summary["already_lost"] == 0
    assert summary["terminal"] == 2
    assert summary["active"] == 1

    stale_meta = read_meta(stale)
    assert stale_meta["state"] == "LOST"
    assert stale_meta["lost_reason"] == "stale_heartbeat"
    assert "lost_at" in stale_meta
    assert read_meta(healthy)["state"] == "RUNNING"

    # Idempotent rerun.
    assert cluster_agent.main(argv) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["lost_materialized"] == []
    assert summary["already_lost"] == 1

    # Capacity accounting only reserves the healthy job's tasks now.
    assert cluster_agent._active_tasks_by_worker(tmp_path) == {"w1": 2}


def test_reconcile_prune_removes_only_old_terminal_jobs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    now = time.time()
    old_done = make_job(
        tmp_path,
        job_id="old-done",
        exit_code=0,
        finished_epoch=now - 40 * 86400,
        last_seen_epoch=now - 40 * 86400,
    )
    new_done = make_job(
        tmp_path,
        job_id="new-done",
        exit_code=0,
        finished_epoch=now - 86400,
        last_seen_epoch=now - 86400,
    )
    active = make_job(tmp_path, job_id="still-running", last_seen_epoch=now - 5)

    argv = [
        "--jobs-root",
        str(tmp_path),
        "reconcile",
        "--no-probe",
        "--prune",
        "--prune-age-days",
        "14",
    ]
    assert cluster_agent.main(argv) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["pruned"] == ["old-done"]
    assert not old_done.exists()
    assert new_done.exists()
    # Active jobs are never pruned regardless of age.
    assert active.exists()


def test_reconcile_counts_malformed_records_without_crashing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_job(tmp_path, job_id="good", last_seen_epoch=time.time() - 3)
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "meta.json").write_text("]]]")

    assert (
        cluster_agent.main(["--jobs-root", str(tmp_path), "reconcile", "--no-probe"]) == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["malformed"] == 1
    assert summary["scanned"] == 1


# ---------------------------------------------------------------------------
# worker wrapper script
# ---------------------------------------------------------------------------


def test_run_script_writes_worker_pid_and_heartbeat_loop() -> None:
    script = cluster_agent._build_run_script(Path("/jobs/j1"), Path("/work"), "true", 4)
    assert 'echo "$$" > "$WORKER_PID_FILE"' in script
    assert "NA49_CLUSTER_GW_HEARTBEAT_SECONDS" in script
    assert 'kill -0 "$PPID"' in script
    assert "trap 'kill \"$HEARTBEAT_PID\" 2>/dev/null' EXIT" in script
    assert 'mv -f "$LAST_SEEN_FILE.tmp" "$LAST_SEEN_FILE"' in script


def test_wrapper_script_produces_liveness_markers_locally(tmp_path: Path) -> None:
    """Execute the generated wrapper with local bash to verify its markers."""

    job_dir = tmp_path / "job"
    workdir = tmp_path / "work"
    job_dir.mkdir()
    workdir.mkdir()

    script = cluster_agent._build_run_script(job_dir, workdir, "echo hi > out.txt", 1)
    run_sh = job_dir / "run.sh"
    run_sh.write_text(script)

    env = dict(os.environ, NA49_CLUSTER_GW_HEARTBEAT_SECONDS="1")
    subprocess.run(["bash", str(run_sh)], check=True, env=env, timeout=60)

    assert (job_dir / "exit_code").read_text().strip() == "0"
    assert (job_dir / "finished_at").exists()
    assert (job_dir / "worker_pid").read_text().strip().isdigit()
    last_seen = int((job_dir / "last_seen").read_text().strip())
    assert abs(time.time() - last_seen) < 60


# ---------------------------------------------------------------------------
# SSH PID probe parsing (transport stubbed; no network access)
# ---------------------------------------------------------------------------


def _stub_subprocess(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    stub = types.SimpleNamespace(
        run=handler,
        TimeoutExpired=subprocess.TimeoutExpired,
    )
    monkeypatch.setattr(cluster_agent, "subprocess", stub)


def test_ssh_pid_probe_parses_remote_verdicts(monkeypatch: pytest.MonkeyPatch) -> None:
    class Alive:
        returncode = 0
        stdout = "ALIVE\n"
        stderr = ""

    class Dead:
        returncode = 0
        stdout = "DEAD\n"
        stderr = ""

    class Refused:
        returncode = 255
        stdout = ""
        stderr = "ssh: connect refused"

    seen_cmds: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        seen_cmds.append(cmd)
        if "w-alive" in cmd:
            return Alive()
        if "w-dead" in cmd:
            return Dead()
        if "w-refused" in cmd:
            return Refused()
        raise subprocess.TimeoutExpired(cmd="ssh", timeout=1)

    _stub_subprocess(monkeypatch, fake_run)

    assert cluster_agent._pid_alive_over_ssh(ssh_bin="ssh", ssh_options=[], worker="w-alive", pid=7) is True
    assert cluster_agent._pid_alive_over_ssh(ssh_bin="ssh", ssh_options=[], worker="w-dead", pid=7) is False
    assert cluster_agent._pid_alive_over_ssh(ssh_bin="ssh", ssh_options=[], worker="w-refused", pid=7) is None
    assert cluster_agent._pid_alive_over_ssh(ssh_bin="ssh", ssh_options=[], worker="w-hang", pid=7) is None

    probe_cmd = seen_cmds[0]
    assert "BatchMode=yes" in probe_cmd
    assert any(option.startswith("ConnectTimeout=") for option in probe_cmd)


# ---------------------------------------------------------------------------
# client terminal-state handling
# ---------------------------------------------------------------------------


def test_client_wait_treats_lost_as_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "LOST" in cluster_client.TERMINAL_STATES

    client = cluster_client.ClusterGatewayClient(host="unused")
    calls = {"count": 0}

    def fake_status(job_id: str) -> dict:
        calls["count"] += 1
        if calls["count"] < 3:
            return {"state": "RUNNING"}
        return {"state": "LOST"}

    monkeypatch.setattr(client, "status", fake_status)
    final = client.wait("some-job", poll_seconds=0.01)
    assert final["state"] == "LOST"
    assert calls["count"] == 3


# ---------------------------------------------------------------------------
# client-side threshold forwarding
# ---------------------------------------------------------------------------


def test_client_agent_args_put_thresholds_before_subcommand() -> None:
    client = cluster_client.ClusterGatewayClient(
        jobs_root="/jr",
        stale_heartbeat_seconds=3600,
        launch_grace_seconds=14400,
    )
    args = client._agent_args("status", "j1")
    assert args == [
        "--jobs-root",
        "/jr",
        "--stale-heartbeat-seconds",
        "3600",
        "--launch-grace-seconds",
        "14400",
        "status",
        "j1",
    ]
    # Omitted thresholds emit nothing.
    bare = cluster_client.ClusterGatewayClient()._agent_args("status", "j1")
    assert bare == ["status", "j1"]


def _make_fake_ssh(tmp_path: Path) -> Path:
    """A local stand-in for ssh: run the 'remote' command line verbatim."""

    fake = tmp_path / "fake-ssh"
    fake.write_text("#!/usr/bin/env bash\nexec bash -c \"$2\"\n")
    fake.chmod(0o755)
    return fake


def test_client_forwards_grace_to_every_subcommand(tmp_path: Path) -> None:
    """capacity/status/wait keep a 1000 s-old SUBMITTED job RUNNING.

    The client carries --launch-grace-seconds on every agent call; a job
    past the default 900 s grace would otherwise read as LOST. Transport
    is a local fake ssh script; no live connection is made.
    """

    now = time.time()
    make_job(
        tmp_path,
        job_id="20260826T000000Z-00000004",
        meta={"state": "SUBMITTED", "submitted_at": _iso(now - 1000), "tasks": 4},
    )

    agent_path = Path(cluster_agent.__file__).resolve()
    fake_ssh = _make_fake_ssh(tmp_path)
    extended = cluster_client.ClusterGatewayClient(
        host="fake-host",
        agent_command=f"python3 {agent_path}",
        ssh_bin=str(fake_ssh),
        jobs_root=str(tmp_path),
        launch_grace_seconds=14400,
    )
    default = cluster_client.ClusterGatewayClient(
        host="fake-host",
        agent_command=f"python3 {agent_path}",
        ssh_bin=str(fake_ssh),
        jobs_root=str(tmp_path),
    )

    # capacity reserves the slots under the extended grace...
    workers = extended.capacity(workers=["w1"])
    by_name = {entry["worker"]: entry for entry in workers}
    assert by_name["w1"]["active_tasks"] == 4

    # ...and status reports RUNNING...
    job = extended.status("20260826T000000Z-00000004")
    assert job["state"] == "RUNNING"

    # ...while wait keeps polling instead of returning a terminal verdict.
    with pytest.raises(TimeoutError):
        extended.wait(
            "20260826T000000Z-00000004", poll_seconds=0.01, timeout_seconds=0.05
        )

    # Control: without forwarded thresholds the same record reads LOST.
    assert default.status("20260826T000000Z-00000004")["state"] == "LOST"
    workers = default.capacity(workers=["w1"])
    by_name = {entry["worker"]: entry for entry in workers}
    assert by_name["w1"]["active_tasks"] == 0
