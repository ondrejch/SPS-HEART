"""The freeze-manifest inventory count, read from the manifest.

Replaces the hard-coded ``INVENTORY_COUNT = <n>`` literals and the
cross-file count-pin chain of the register tests (2026-09-28): the count
is the manifest's own ``source_tree_sha256`` size, so a landing that
adds files re-pins the manifest only. ``test_legacy_v1_freeze.py`` owns
the digest verification; this helper owns nothing but the count.
"""
from __future__ import annotations

import json
from pathlib import Path

MANIFEST_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "v2_singlet"
    / "legacy_v1_manifest.json"
)


def inventory_count() -> int:
    """The number of pinned regular files in the freeze manifest."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return len(manifest["source_tree_sha256"])
