"""Standard-run gate: execute the benchmarking-helper test tree.

Keeps the analysis-layer tests (invariant kernels, closure, cache policy)
inside the normal `pytest` entry point of the package, per the 05-rev audit
recommendation.  Skipped when the monorepo helper tree is absent (standalone
package checkouts).

Data-dependent helper tests carry ``pytest.mark.external_data``; the gate
deselects them so the run stays green in hermetic source archives where
``spshadrons.web.cern.ch/`` is excluded. Run them explicitly with
``pytest helpers/benchmarking/tests -m external_data`` when the tables are
present.

The layout-sync check (Stage D of notes/opencode/agentic_framework_plan.md)
enforces dual-repo protocol rule 3: the two vendored benchmarking-helper
trees must stay logically in sync.
"""
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HELPERS = os.path.abspath(os.path.join(HERE, "..", "..", "..", "helpers", "benchmarking", "tests"))
SYNC_CHECK = os.path.abspath(
    os.path.join(HERE, "..", "..", "..", "helpers", "benchmarking", "check_helpers_layout_sync.py")
)


@pytest.mark.skipif(not os.path.isdir(HELPERS), reason="monorepo helper tree not present")
def test_benchmark_helper_suite():
    repo_root = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
    env = dict(os.environ, PYTHONPATH=repo_root + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-m", "not external_data", HELPERS],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=900,
    )
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-1000:]


@pytest.mark.skipif(not os.path.isdir(HELPERS), reason="monorepo helper tree not present")
def test_helpers_layout_sync_reports_no_logic_drift():
    """Run check_helpers_layout_sync.py; any logic drift fails the session."""

    assert os.path.isfile(SYNC_CHECK), f"missing sync checker: {SYNC_CHECK}"
    repo_root = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
    r = subprocess.run(
        [sys.executable, SYNC_CHECK],
        cwd=repo_root, capture_output=True, text=True, timeout=300,
    )
    output = r.stdout[-2000:] + r.stderr[-1000:]
    assert r.returncode == 0, output
    assert "logic_drift=0" in r.stdout, output
