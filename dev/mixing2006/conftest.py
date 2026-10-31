"""Test-tier selection for the solve-and-verify test protocol (nested tree).

Companion to the monorepo-root ``conftest.py`` (helper tree). Both read the
single list at the monorepo root, ``slow_tests.txt``.

When the environment variable ``NA49_TEST_TIER`` is set to ``fast``, the
node ids listed there (one per line, comments with ``#``) are deselected,
so in-flight development feedback runs only the fast tier. A bare test
file path selects the whole file; a full node id selects a single test.
With the variable unset (the default) the suite runs exactly as it
always did.

The full suite, including the slow tier, is the land-gate requirement
(see ``.opencode/protocol.md``). The list is maintained from
``--durations=0`` measurements; add a line whenever a test is re-measured
above the ~2-3 s threshold.
"""

from __future__ import annotations

import os
from pathlib import Path

_SLOW_LIST = Path(__file__).resolve().parents[2] / "slow_tests.txt"


def _slow_entries() -> set[str]:
    if not _SLOW_LIST.is_file():
        return set()
    entries = set()
    for line in _SLOW_LIST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            entries.add(line)
    return entries


#: Tests that reproduce or pin behavior sealed under the pre-2026-09-28
#: kernels (the Round 5 run drivers' smoke modes, the run registers'
#: substrate-versus-disk checks). They are listed, one node id or file
#: path per line, in ``tests/superseded_substrate_tests.txt`` with the
#: reason in the file header, and marked strict xfail here: when a
#: re-seal under the corrected kernels makes one pass, the strict mark
#: fails the run so the entry is removed deliberately.
_SUPERSEDED_LIST = Path(__file__).resolve().parent / "tests" / "superseded_substrate_tests.txt"
_SUPERSEDED_REASON = (
    "sealed under the pre-2026-09-28 kernels (FAB-03/07/08/09 corrections); "
    "the sealed substrate no longer matches the tree; re-seal under the "
    "corrected kernels and remove the entry from tests/superseded_substrate_tests.txt"
)


def _superseded_entries() -> set[str]:
    if not _SUPERSEDED_LIST.is_file():
        return set()
    return {
        line.strip()
        for line in _SUPERSEDED_LIST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def _mark_superseded(config, items) -> None:
    import pytest

    superseded = _superseded_entries()
    if not superseded:
        return
    for item in items:
        nodeid = item.nodeid
        if nodeid.startswith("dev/mixing2006/"):
            nodeid = nodeid[len("dev/mixing2006/") :]
        if any(nodeid == e or nodeid.startswith(e + "::") for e in superseded):
            item.add_marker(pytest.mark.xfail(reason=_SUPERSEDED_REASON, strict=True))


def pytest_collection_modifyitems(config, items) -> None:
    _mark_superseded(config, items)
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
