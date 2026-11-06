"""The slot merge and the seal's clean-validation check (2026-11-05): a slot's arrays do not depend on its chunk split
(the per-slot constant led_ssbar_scale was concatenated per chunk, so the clean validation's probe slot failed from R98
on), and the seal refuses a clean validation whose verdicts do not all hold."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

NESTED_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NESTED_ROOT / "helpers" / "benchmarking"))
import run_engine as E  # noqa: E402


@pytest.fixture(scope="module")
def engine():
    E.configure(NESTED_ROOT / "config" / "runs" / "r139.json")
    return E


@pytest.mark.parametrize("cls", ["sd_projectile", "nd0"])
def test_a_slot_s_arrays_do_not_depend_on_its_chunk_split(engine, cls):
    e = engine
    one = e.generate_slot(cls, 777, 12, 555000, e.GEN0_SHAPE, e.FLOOR_GRID[0], e.GEN_FLAVOR, chunks=1)
    three = e.generate_slot(cls, 777, 12, 555000, e.GEN0_SHAPE, e.FLOOR_GRID[0], e.GEN_FLAVOR, chunks=3)
    assert one["event_rolling_sha256"] == three["event_rolling_sha256"]
    assert e.arrays_digest(one["arrays"]) == e.arrays_digest(three["arrays"])
    for key in e.SLOT_CONSTANT_ARRAYS:
        if key in three["arrays"]:
            assert three["arrays"][key].shape == (1,)


def test_the_seal_needs_every_clean_validation_verdict():
    good = {"report_hash_match": True, "chosen_branch": {"branch_digest_match": True}, "probe_slot": {"arrays_match": True}}
    assert E.clean_validation_failures(good) == []
    assert E.clean_validation_failures({**good, "probe_slot": {"arrays_match": False}}) == ["probe_arrays_match"]
    assert E.clean_validation_failures({**good, "report_hash_match": False}) == ["report_hash_match"]
    assert set(E.clean_validation_failures({})) == {"report_hash_match", "branch_digest_match", "probe_arrays_match"}
