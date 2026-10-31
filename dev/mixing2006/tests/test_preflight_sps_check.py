"""Regression coverage for the preflight SPS hash-check verdicts.

Before this fix, ``preflight.sh`` counted failures with ``grep -c 'check
FAIL'`` over the full stage-00 CHECK_ONLY log. That substring also matches
the human-facing summary line ``SPS hash check FAILED`` (and every per-file
``check FAIL ... file absent`` line), so a fresh archive extraction with all
21 SPS files absent produced 22 counted failures against 21 absent files,
flagged the difference as a mismatch, and failed the preflight. A missing
input is a download-stage concern (WARN), not a hash mismatch (FAIL).

``sps_resonances_paper/00-download_spshadrons/run.sh`` now ends its CHECK_ONLY
mode with one machine-readable line,

    CHECK_ONLY_SUMMARY verified=N absent=N mismatch=N

and with distinct exit codes (0 all verified, 3 files absent only, 2 at
least one present file mismatches, 1 operational failure). ``preflight.sh``
parses only that line, so the human-facing lines can never be miscounted.
This module drives the real scripts through subprocesses with ``SPS_ROOT``,
``RUNS_ROOT``, ``EXTERNAL_ROOT``, and ``VENV_DIR`` pointed at scratch
directories:

1. all SPS files absent -> ``check WARN sps_hash_check`` and exit 0;
2. one present file with the wrong sha256 -> ``check FAIL sps_hash_check``
   and exit 1;
3. the real mirror (when this checkout carries it) -> ``check PASS
   sps_hash_check`` and exit 0.

Set ``PREFLIGHT_SPS_CHECK_PIPELINE_DIR`` to point this module at a different
copy of the pipeline scripts. The module skips cleanly when bash or the
pipeline scripts are absent (sdist checkouts).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
PIPELINE_DIR = Path(os.environ.get("PREFLIGHT_SPS_CHECK_PIPELINE_DIR", PKG_ROOT / "sps_resonances_paper"))
PREFLIGHT_SH = PIPELINE_DIR / "preflight.sh"
LOCK_FILE = PIPELINE_DIR / "reproducibility.lock.json"

pytestmark = pytest.mark.skipif(
    not (PREFLIGHT_SH.is_file() and LOCK_FILE.is_file()),
    reason="bash pipeline scripts absent (sdist checkout)",
)


def _sps_artifacts():
    """(relative-path, expected-sha256) pairs recorded in the manifest."""
    lock = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    artifacts = lock["downloads"]["sps_web"]["artifacts"]
    return [(str(entry["path"]), str(entry["sha256"])) for entry in artifacts]


def _run_preflight(tmp_path: Path, sps_root: Path) -> subprocess.CompletedProcess[str]:
    env = {
        "SPS_ROOT": str(sps_root),
        "RUNS_ROOT": str(tmp_path / "runs"),
        "EXTERNAL_ROOT": str(tmp_path / "external"),
        "VENV_DIR": str(tmp_path / "venv"),
        "PREFLIGHT_MIN_FREE_GB": "1",
        "PATH": os.environ["PATH"],
    }
    return subprocess.run(
        ["bash", str(PREFLIGHT_SH)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_preflight_warns_when_sps_files_absent(tmp_path):
    """A fresh archive extraction (no SPS files at all) must not fail the
    preflight: the files are downloadable, so the verdict stays WARN and the
    exit code stays 0."""
    sps_root = tmp_path / "sps"
    sps_root.mkdir()
    result = _run_preflight(tmp_path, sps_root)
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "check WARN sps_hash_check" in combined, combined
    n_artifacts = len(_sps_artifacts())
    assert f"{n_artifacts} of {n_artifacts} manifest files are absent" in combined, combined
    assert "preflight verdict: PASS WITH WARNINGS" in combined, combined


def test_preflight_fails_when_sps_file_hash_mismatches(tmp_path):
    """A present file whose sha256 contradicts the manifest is a mismatch,
    not an absence: the verdict must be FAIL and the exit code 1."""
    rel, expected = _sps_artifacts()[0]
    sps_root = tmp_path / "sps"
    target = sps_root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"corrupted content that cannot match the manifest")
    result = _run_preflight(tmp_path, sps_root)
    combined = result.stdout + result.stderr
    assert result.returncode == 1, combined
    assert "check FAIL sps_hash_check" in combined, combined
    assert "1 of" in combined and "does not match the manifest" in combined, combined
    assert "preflight verdict: FAIL" in combined, combined


def test_preflight_passes_when_sps_files_verified(tmp_path):
    """With the reference mirror in place the check passes: every recorded
    file verifies and the sps_hash_check verdict is PASS."""
    mirror = PKG_ROOT / "spshadrons.web.cern.ch"
    if not mirror.is_dir() or not any(mirror.iterdir()):
        pytest.skip("reference SPS mirror not present in this checkout")
    result = _run_preflight(tmp_path, mirror)
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "check PASS sps_hash_check" in combined, combined
