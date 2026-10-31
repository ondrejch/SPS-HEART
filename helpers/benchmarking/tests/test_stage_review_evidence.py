"""Compact M4-M13 evidence stager (review 9 archive recipe)."""

from __future__ import annotations

from pathlib import Path

from helpers.benchmarking.stage_review_evidence import (
    COMPACT_RUNS,
    stage_compact_evidence,
)


def test_stage_compact_evidence_copies_small_text_and_skips_large_and_npz(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "00runs"
    staging = tmp_path / "staging"
    keep = COMPACT_RUNS[0]
    src = runs / keep
    src.mkdir(parents=True)
    (src / "pre_registration.json").write_text("{}\n")
    (src / "REPORT.md").write_text("# report\n")
    (src / "events.jsonl").write_text("skip me\n")
    (src / "big.txt").write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    (src / "cache.npz").write_bytes(b"npz")
    copied = stage_compact_evidence(staging, runs_root=runs)
    names = {p.name for p in copied}
    assert names == {"pre_registration.json", "REPORT.md"}
    dest = staging / "evidence" / keep
    assert (dest / "pre_registration.json").is_file()
    assert not (dest / "events.jsonl").exists()
    assert not (dest / "big.txt").exists()
    assert not (dest / "cache.npz").exists()
    readme = (staging / "evidence" / "README.md").read_text()
    assert keep in readme
    assert "Missing run directories" in readme
