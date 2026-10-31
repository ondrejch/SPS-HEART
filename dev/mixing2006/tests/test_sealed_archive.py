"""The sealed-archive re-hash accepts only listed, deleted event files as absent."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tests._sealed_archive import sealed_mismatches, skip_unless_sealed


def _sealed(run_dir: Path, files: dict[str, bytes]) -> None:
    run_dir.mkdir()
    for name, body in files.items():
        (run_dir / name).write_bytes(body)
    (run_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{hashlib.sha256(body).hexdigest()}  {name}\n" for name, body in sorted(files.items())))


def test_a_listed_deleted_event_file_is_accepted(tmp_path):
    run = tmp_path / "run"
    _sealed(run, {"report.json": b"{}", "slot_0.npz": b"events"})
    (run / "slot_0.npz").unlink()
    (run / "EVENTS_DELETED.txt").write_text("header line\nDeleted (1 files, 0.00 GiB):\n           6  slot_0.npz\n")
    assert sealed_mismatches(run) == []


def test_an_unlisted_missing_file_and_a_listed_compact_file_are_flagged(tmp_path):
    run = tmp_path / "run"
    _sealed(run, {"report.json": b"{}", "slot_0.npz": b"events", "slot_1.npz": b"more"})
    for name in ("report.json", "slot_1.npz"):
        (run / name).unlink()
    (run / "EVENTS_DELETED.txt").write_text("           2  report.json\n")
    assert sealed_mismatches(run) == ["report.json", "slot_1.npz"]


def test_skip_unless_sealed_requires_the_seal_manifest(tmp_path):
    with pytest.raises(pytest.skip.Exception):
        skip_unless_sealed(tmp_path)
    (tmp_path / "seal_manifest.json").write_text("{}", encoding="utf-8")
    assert skip_unless_sealed(tmp_path) == tmp_path


def test_a_changed_file_is_flagged_even_when_listed(tmp_path):
    run = tmp_path / "run"
    _sealed(run, {"report.json": b"{}", "slot_0.npz": b"events"})
    (run / "slot_0.npz").write_bytes(b"other")
    (run / "EVENTS_DELETED.txt").write_text("           6  slot_0.npz\n")
    assert sealed_mismatches(run) == ["slot_0.npz"]
