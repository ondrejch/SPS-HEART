"""Test-tier selection for the solve-and-verify test protocol (helper tree).

Companion to ``dev/mixing2006/conftest.py`` (nested tree). Both read the
single list at the monorepo root, ``slow_tests.txt``; entries are namespaced
by prefix (``tests/`` for the nested tree, ``helpers/`` for this tree), so
each conftest only ever matches its own tree's node ids.

When the environment variable ``NA49_TEST_TIER`` is set to ``fast``, the
node ids listed in ``slow_tests.txt`` (one per line, comments with ``#``)
are deselected, so in-flight development feedback runs only the fast tier.
A bare test file path selects the whole file; a full node id selects a
single test. With the variable unset (the default) the suite runs exactly
as it always did.

The full suite, including the slow tier, is the land-gate requirement
(see ``.opencode/protocol.md``). The list is maintained from
``--durations=0`` measurements; add a line whenever a test is re-measured
above the ~2-3 s threshold.
"""

from __future__ import annotations

import os
from pathlib import Path

_SLOW_LIST = Path(__file__).resolve().parent / "slow_tests.txt"


def _slow_entries() -> set[str]:
    if not _SLOW_LIST.is_file():
        return set()
    entries = set()
    for line in _SLOW_LIST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            entries.add(line)
    return entries


def pytest_collection_modifyitems(config, items) -> None:
    if os.environ.get("NA49_TEST_TIER") != "fast":
        return
    slow = _slow_entries()
    if not slow:
        return
    kept = []
    dropped = []
    for item in items:
        nodeid = item.nodeid
        if nodeid.startswith("dev/mixing2006/"):
            nodeid = nodeid[len("dev/mixing2006/") :]
        if any(nodeid == entry or nodeid.startswith(entry + "::") for entry in slow):
            dropped.append(item)
        else:
            kept.append(item)
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = kept
