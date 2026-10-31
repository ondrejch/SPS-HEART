"""Job-ID confinement and SSH/subprocess timeout tests for the cluster gateway.

Job IDs supplied to ``status``, ``result``, and ``logs`` must match the
generated ``JOB_ID_PATTERN``; ``resolve_job_dir`` refuses symlinked job
directories and confines every resolved path to the jobs root before any
file access, and pruning never follows symlinks. The client's SSH calls and
the agent's job launch carry explicit timeouts overridable through
environment variables. No live SSH connection is made; slow transports are
local ``sleep`` processes.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from helpers.na49_gateway import cluster_agent
from helpers.na49_gateway import cluster_client


JOB_ID_PATTERN_TEXT = r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$"

# Job IDs that must be refused: traversal climbs, absolute paths, embedded
# separators, wrong formats, and wrong hex widths or case. None of them
# matches the generated job-ID syntax.
ADVERSARIAL_JOB_IDS = [
    "../escape",
    "../../etc",
    "/etc/passwd",  # absolute path outside any jobs root
    "sub/dir",
    "..",
    ".",
    "good-job",  # wrong format entirely
    "20260830T120000Z-abc",  # hex too short
    "20260830T120000Z-abcdef012",  # hex too long
    "20260830T120000Z-ABCDEF01",  # uppercase hex
]

VALID_JOB_ID = "20260826T000000Z-abcdef01"


def _iso(epoch: float) -> str:
    """Format *epoch* the way the gateway records timestamps."""

    return (
        datetime.fromtimestamp(epoch, tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def make_job(
    root: Path,
    job_id: str,
    *,
    worker: str = "w1",
    meta: dict | None = None,
    tasks: int = 1,
    exit_code: int | None = None,
    finished_epoch: float | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
) -> Path:
    """Create a fake job directory with marker files (generated-format ID)."""

    directory = root / job_id
    directory.mkdir(parents=True)
    record = {
        "job_id": job_id,
        "name": "",
        "worker": worker,
        "tasks": tasks,
        "submitted_at": _iso(time.time()),
        "state": "RUNNING",
    }
    if meta:
        record.update(meta)
    (directory / "meta.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    if exit_code is not None:
        (directory / "exit_code").write_text(f"{exit_code}\n")
    if finished_epoch is not None:
        (directory / "finished_at").write_text(_iso(finished_epoch) + "\n")
    if stdout is not None:
        (directory / "stdout.log").write_text(stdout)
    if stderr is not None:
        (directory / "stderr.log").write_text(stderr)
    return directory


def make_jobs_root(tmp_path: Path) -> Path:
    """Return a fresh jobs root directory."""

    jobs_root = tmp_path / "jobs"
    jobs_root.mkdir()
    return jobs_root


# ---------------------------------------------------------------------------
# shared job-ID constants
# ---------------------------------------------------------------------------


def test_job_id_constants_match_generated_format() -> None:
    """Minting and validation share one pattern; it accepts what minting emits."""

    assert cluster_agent.JOB_ID_PATTERN == JOB_ID_PATTERN_TEXT
    assert cluster_agent.JOB_ID_TIME_FORMAT == "%Y%m%dT%H%M%SZ"
    minted = cluster_agent._mint_job_id()
    assert re.match(cluster_agent.JOB_ID_PATTERN, minted)
    # The minted timestamp component decodes with the shared time format.
    stamp, hex_part = minted.split("-")
    time.strptime(stamp, cluster_agent.JOB_ID_TIME_FORMAT)
    assert re.fullmatch(r"[0-9a-f]{8}", hex_part)


# ---------------------------------------------------------------------------
# refusals: pattern and containment
# ---------------------------------------------------------------------------


def test_resolve_job_dir_refuses_adversarial_job_ids(tmp_path: Path) -> None:
    """Traversal shapes, absolute paths, and wrong formats fail the pattern check."""

    jobs_root = make_jobs_root(tmp_path)
    for candidate in ADVERSARIAL_JOB_IDS:
        with pytest.raises(ValueError, match="Invalid job_id"):
            cluster_agent.resolve_job_dir(jobs_root, candidate)
        with pytest.raises(ValueError, match="expected the generated format"):
            cluster_agent.resolve_job_dir(jobs_root, candidate)


def test_status_result_logs_refuse_adversarial_job_ids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """status, result, and logs reject bad IDs before touching the filesystem.

    The error message names the offending job_id, and files inside and
    outside the jobs root are left untouched.
    """

    jobs_root = make_jobs_root(tmp_path)
    sentinel = jobs_root / "sentinel.txt"
    sentinel.write_text("keep")
    outside = tmp_path / "outside_secret.txt"
    outside.write_text("unchanged")

    for candidate in ADVERSARIAL_JOB_IDS:
        for subcmd, expected_rc in (("status", 2), ("result", 2), ("logs", 1)):
            capsys.readouterr()
            rc = cluster_agent.main(["--jobs-root", str(jobs_root), subcmd, candidate])
            captured = capsys.readouterr()
            assert rc == expected_rc, (candidate, subcmd, rc, captured.err)
            assert "Invalid job_id" in captured.err
            assert candidate in captured.err
            if subcmd in ("status", "result"):
                error_payload = json.loads(captured.err)
                assert error_payload["ok"] is False

    assert sentinel.read_text() == "keep"
    assert outside.read_text() == "unchanged"


def test_containment_check_refuses_paths_that_sneak_past_the_pattern(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Defense in depth: the resolved-path containment check stands alone.

    The generated-syntax check already rejects separators and ``..``
    components; this test relaxes the pattern to prove the containment check
    is an independent second line of defense for candidates that resolve
    outside the root, both on ``resolve_job_dir`` and end to end through
    the ``status`` handler. (``sub/dir`` from the adversarial list resolves
    inside the root and is therefore refused by the pattern check alone.)
    """

    jobs_root = make_jobs_root(tmp_path)
    monkeypatch.setattr(cluster_agent, "_JOB_ID_RE", re.compile(r".*"))

    for candidate in ("../escape", "..", "/etc/passwd"):
        with pytest.raises(ValueError, match="outside the jobs root"):
            cluster_agent.resolve_job_dir(jobs_root, candidate)

        capsys.readouterr()
        rc = cluster_agent.main(["--jobs-root", str(jobs_root), "status", candidate])
        captured = capsys.readouterr()
        assert rc == 2
        assert "Invalid job_id" in captured.err
        assert "outside the jobs root" in captured.err


# ---------------------------------------------------------------------------
# symlinked job directories
# ---------------------------------------------------------------------------


def test_symlinked_job_directory_refused_for_status_result_logs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A symlinked job directory is refused even when its target is inside the root."""

    jobs_root = make_jobs_root(tmp_path)
    target = make_job(
        jobs_root, "20260826T000000Z-11111111", exit_code=0, finished_epoch=time.time() - 30
    )
    link_id = "20260826T000000Z-22222222"
    (jobs_root / link_id).symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        cluster_agent.resolve_job_dir(jobs_root, link_id)

    for subcmd, expected_rc in (("status", 2), ("result", 2), ("logs", 1)):
        capsys.readouterr()
        rc = cluster_agent.main(["--jobs-root", str(jobs_root), subcmd, link_id])
        captured = capsys.readouterr()
        assert rc == expected_rc, (subcmd, rc, captured.err)
        assert "Invalid job_id" in captured.err
        assert "symlink" in captured.err

    # The refusal is about the symlink entry: the target directory itself is
    # still reachable through its own generated ID.
    assert cluster_agent.main(["--jobs-root", str(jobs_root), "status", target.name]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True


def test_list_and_capacity_accounting_skip_symlinked_entries(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Listing and active-worker accounting never see through a symlinked entry."""

    jobs_root = make_jobs_root(tmp_path)
    make_job(jobs_root, "20260826T000000Z-33333333", tasks=1)
    linked_target = make_job(jobs_root, "20260826T000000Z-44444444", tasks=5)
    (jobs_root / "20260826T000000Z-55555555").symlink_to(linked_target, target_is_directory=True)

    assert cluster_agent.main(["--jobs-root", str(jobs_root), "list"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [job["job_id"] for job in payload["jobs"]] == [
        "20260826T000000Z-44444444",
        "20260826T000000Z-33333333",
    ]

    assert cluster_agent._active_jobs_by_worker(jobs_root) == {"w1": 2}
    assert cluster_agent._active_tasks_by_worker(jobs_root) == {"w1": 6}


# ---------------------------------------------------------------------------
# containment positive controls and pruning
# ---------------------------------------------------------------------------


def test_valid_job_id_resolves_strictly_inside_jobs_root(tmp_path: Path) -> None:
    """A legitimate generated ID resolves to a path beneath the resolved root."""

    jobs_root = make_jobs_root(tmp_path)
    make_job(jobs_root, VALID_JOB_ID)
    resolved = cluster_agent.resolve_job_dir(jobs_root, VALID_JOB_ID)
    assert resolved == jobs_root / VALID_JOB_ID
    assert resolved.resolve() == (jobs_root / VALID_JOB_ID).resolve()
    assert resolved.resolve().parent == jobs_root.resolve()


def test_status_result_logs_accept_generated_job_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Positive control: all three subcommands work with a generated ID."""

    jobs_root = make_jobs_root(tmp_path)
    make_job(
        jobs_root,
        VALID_JOB_ID,
        exit_code=0,
        finished_epoch=time.time() - 30,
        stdout="hello stdout\nsecond line\n",
        stderr="warned\n",
    )

    assert cluster_agent.main(["--jobs-root", str(jobs_root), "status", VALID_JOB_ID]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["job"]["job_id"] == VALID_JOB_ID
    assert payload["job"]["state"] == "COMPLETED"

    assert (
        cluster_agent.main(
            ["--jobs-root", str(jobs_root), "result", VALID_JOB_ID, "--tail-lines", "5"]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["job"]["stdout_tail"] == ["hello stdout\n", "second line\n"]
    assert payload["job"]["stderr_tail"] == ["warned\n"]

    assert cluster_agent.main(["--jobs-root", str(jobs_root), "logs", VALID_JOB_ID]) == 0
    assert "hello stdout" in capsys.readouterr().out


def test_prune_terminal_jobs_never_follows_symlinks(tmp_path: Path) -> None:
    """Pruning deletes a real terminal job but never a symlinked entry or its target."""

    now = time.time()
    jobs_root = make_jobs_root(tmp_path)

    # A real terminal job old enough to prune.
    real_old_id = "20260826T000000Z-aaaaaaaa"
    real_old = make_job(
        jobs_root, real_old_id, exit_code=0, finished_epoch=now - 3600
    )

    # A symlinked entry pointing at a terminal job inside the root; the
    # target itself is too young to prune, so it must survive the run.
    inside_target_id = "20260826T000000Z-bbbbbbbb"
    inside_target = make_job(
        jobs_root, inside_target_id, exit_code=0, finished_epoch=now - 10
    )
    inside_link_id = "20260826T000000Z-cccccccc"
    (jobs_root / inside_link_id).symlink_to(inside_target, target_is_directory=True)

    # A symlinked entry pointing outside the root, laid out as an old
    # terminal job with a sentinel file that deletion through the link
    # would destroy.
    outside_dir = tmp_path / "outside_job"
    outside_dir.mkdir()
    make_job(outside_dir, "00000000T000000Z-00000000", exit_code=0, finished_epoch=now - 7200)
    sentinel = outside_dir / "sentinel.txt"
    sentinel.write_text("must survive")
    outside_link_id = "20260826T000000Z-dddddddd"
    (jobs_root / outside_link_id).symlink_to(outside_dir, target_is_directory=True)

    pruned = cluster_agent._prune_terminal_jobs(jobs_root, older_than_epoch=now - 100)

    assert pruned == [real_old_id]
    assert not real_old.exists()
    assert (jobs_root / inside_link_id).is_symlink()
    assert inside_target.exists()
    assert (jobs_root / outside_link_id).is_symlink()
    assert sentinel.read_text() == "must survive"


# ---------------------------------------------------------------------------
# minted IDs satisfy the shared validation
# ---------------------------------------------------------------------------


def test_minted_job_ids_pass_resolve_job_dir_200_rounds(tmp_path: Path) -> None:
    """Minting and validation agree: every minted ID resolves without error."""

    jobs_root = make_jobs_root(tmp_path)
    for _ in range(200):
        minted = cluster_agent._mint_job_id()
        assert re.match(cluster_agent.JOB_ID_PATTERN, minted)
        resolved = cluster_agent.resolve_job_dir(jobs_root, minted)
        assert resolved == jobs_root / minted
        assert resolved.resolve().parent == jobs_root.resolve()


# ---------------------------------------------------------------------------
# SSH/subprocess timeouts
# ---------------------------------------------------------------------------


def test_timeout_defaults_env_names_and_invalid_env_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Documented defaults and environment overrides resolve as specified."""

    assert cluster_agent.DEFAULT_LAUNCH_TIMEOUT_SECONDS == 30.0
    assert cluster_agent.LAUNCH_TIMEOUT_SECONDS_ENV_VAR == "NA49_CLUSTER_GW_LAUNCH_TIMEOUT_SECONDS"
    assert cluster_client.DEFAULT_CLIENT_TIMEOUT_SECONDS == 60.0
    assert cluster_client.CLIENT_TIMEOUT_ENV_VAR == "NA49_GW_TIMEOUT"

    # Client: a valid value is honored; invalid or non-positive values fall
    # back to the default (a non-negative number of seconds is documented).
    monkeypatch.setenv("NA49_GW_TIMEOUT", "12.5")
    assert cluster_client._subprocess_timeout() == 12.5
    for bad in ("not-a-number", "", "0", "-3"):
        monkeypatch.setenv("NA49_GW_TIMEOUT", bad)
        assert cluster_client._subprocess_timeout() == cluster_client.DEFAULT_CLIENT_TIMEOUT_SECONDS

    # Agent: a valid value is honored; invalid values fall back to the default.
    monkeypatch.setenv("NA49_CLUSTER_GW_LAUNCH_TIMEOUT_SECONDS", "7.5")
    assert cluster_agent._launch_timeout_seconds() == 7.5
    for bad in ("not-a-number", ""):
        monkeypatch.setenv("NA49_CLUSTER_GW_LAUNCH_TIMEOUT_SECONDS", bad)
        assert (
            cluster_agent._launch_timeout_seconds() == cluster_agent.DEFAULT_LAUNCH_TIMEOUT_SECONDS
        )

    # Unset environment resolves to the documented defaults.
    monkeypatch.delenv("NA49_GW_TIMEOUT", raising=False)
    monkeypatch.delenv("NA49_CLUSTER_GW_LAUNCH_TIMEOUT_SECONDS", raising=False)
    assert cluster_client._subprocess_timeout() == cluster_client.DEFAULT_CLIENT_TIMEOUT_SECONDS
    assert cluster_agent._launch_timeout_seconds() == cluster_agent.DEFAULT_LAUNCH_TIMEOUT_SECONDS


class _SlowTransportClient(cluster_client.ClusterGatewayClient):
    """Stubbed transport: the SSH base command is a slow local process."""

    def _ssh_base_command(self) -> list[str]:
        return ["/bin/sh", "-c", "sleep 5"]


def test_client_timeout_kills_slow_calls_and_names_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NA49_GW_TIMEOUT bounds every client subprocess call and is named on expiry."""

    monkeypatch.setenv("NA49_GW_TIMEOUT", "1")
    client = _SlowTransportClient(host="unused", agent_command="agent")

    started = time.monotonic()
    with pytest.raises(cluster_client.ClusterGatewayError) as excinfo:
        client._run_json(["status", "x"])
    elapsed = time.monotonic() - started
    assert "timed out" in str(excinfo.value)
    assert "NA49_GW_TIMEOUT" in str(excinfo.value)
    assert elapsed < 4.0  # the stub sleeps for 5 s; the timeout fired first

    started = time.monotonic()
    with pytest.raises(cluster_client.ClusterGatewayError) as excinfo:
        client._run_text(["logs", "x"], capture_output=True)
    elapsed = time.monotonic() - started
    assert "timed out" in str(excinfo.value)
    assert "NA49_GW_TIMEOUT" in str(excinfo.value)
    assert elapsed < 4.0

    # The streaming path (logs --follow) is bounded by the same timeout.
    started = time.monotonic()
    with pytest.raises(cluster_client.ClusterGatewayError) as excinfo:
        client._run_text(["logs", "x", "--follow"], capture_output=False)
    elapsed = time.monotonic() - started
    assert "timed out" in str(excinfo.value)
    assert "NA49_GW_TIMEOUT" in str(excinfo.value)
    assert elapsed < 4.0


def test_agent_launch_timeout_kills_slow_ssh_and_names_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NA49_CLUSTER_GW_LAUNCH_TIMEOUT_SECONDS bounds the SSH job launch."""

    monkeypatch.setenv("NA49_CLUSTER_GW_LAUNCH_TIMEOUT_SECONDS", "1")
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="NA49_CLUSTER_GW_LAUNCH_TIMEOUT_SECONDS"):
        cluster_agent._launch_remote_job(
            ssh_bin="/bin/sh",
            ssh_options=["-c", "sleep 5"],
            worker="slowhost",
            run_script_path=Path("/dev/null"),
        )
    assert time.monotonic() - started < 4.0

    # An explicit timeout_seconds overrides the environment: with the env
    # variable back at the default, the explicit 1 s bound still fails the
    # slow launch fast.
    monkeypatch.setenv("NA49_CLUSTER_GW_LAUNCH_TIMEOUT_SECONDS", "60")
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="timed out after 1 s"):
        cluster_agent._launch_remote_job(
            ssh_bin="/bin/sh",
            ssh_options=["-c", "sleep 5"],
            worker="slowhost",
            run_script_path=Path("/dev/null"),
            timeout_seconds=1.0,
        )
    assert time.monotonic() - started < 4.0
