"""Standard-run gate: execute the cluster-gateway test tree.

Brings ``helpers/na49_gateway/tests`` inside the normal ``pytest`` entry point
of the package, mirroring ``test_helpers_gate.py``.  Skipped when the monorepo
gateway tree is absent (standalone package checkouts).

All current gateway tests are hermetic: they exercise stale-job recovery,
client parsing, and agent reconciliation against fake jobs directories and
stubbed/faked SSH transport only.  No live SSH endpoint, cluster access, or
external table is required.  Any future live-endpoint gateway test must carry
a dedicated marker registered in both the root ``pytest.ini`` and this
package's ``pyproject.toml`` markers, and must stay outside this gate's
selection string.
"""
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
GATEWAY_TESTS = os.path.abspath(
    os.path.join(HERE, "..", "..", "..", "helpers", "na49_gateway", "tests")
)


@pytest.mark.skipif(not os.path.isdir(GATEWAY_TESTS), reason="monorepo gateway tree not present")
def test_gateway_suite():
    repo_root = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
    env = dict(os.environ, PYTHONPATH=repo_root + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-m", "not external_data", GATEWAY_TESTS],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=900,
    )
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-1000:]
