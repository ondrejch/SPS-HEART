"""The re-hash of a sealed run directory against its ``SHA256SUMS.txt``.

The archive checks of the register tests verify every listed file at its
digest. The event files (``.npz``) of the superseded sealed runs were
deleted to free disk space (2026-10-22 and 2026-10-27 cleanups,
``_knowledge/operations/gateway-fit-fanout.md``); each cleaned directory
lists them in ``EVENTS_DELETED.txt``, and ``SHA256SUMS.txt`` and
``seal_manifest.json`` keep their digests. A listed ``.npz`` file is
accepted as absent; every other entry must be present at its digest.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

SUMS_NAME = "SHA256SUMS.txt"
DELETED_NAME = "EVENTS_DELETED.txt"


def deleted_events(run_dir: Path) -> frozenset[str]:
    """The ``.npz`` names that ``EVENTS_DELETED.txt`` lists (``<size>  <name>`` rows)."""
    marker = run_dir / DELETED_NAME
    if not marker.is_file():
        return frozenset()
    rows = (line.split() for line in marker.read_text(encoding="utf-8").splitlines())
    return frozenset(r[1] for r in rows if len(r) == 2 and r[0].isdigit() and r[1].endswith(".npz"))


def skip_unless_sealed(run_dir: Path) -> Path:
    """Skip the calling test when this checkout has no sealed run directory.

    The gateway suite and a source-only tree do not stage ``dev/00runs``.
    A missing archive is not a failure of the code under test.
    """
    import pytest

    run_dir = Path(run_dir)
    if not (run_dir / "seal_manifest.json").is_file():
        pytest.skip("the sealed run dir is not present")
    return run_dir


def sealed_mismatches(run_dir: Path) -> list[str]:
    """The ``SHA256SUMS.txt`` entries missing or off their digest, deleted event files excepted."""
    deleted = deleted_events(run_dir)
    mismatches = []
    for line in (run_dir / SUMS_NAME).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        path = run_dir / name
        if not path.is_file():
            if name not in deleted:
                mismatches.append(name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            mismatches.append(name)
    return mismatches
