"""The R140 spec: the R139 contract without the line-emitted nucleon pairs (the pair completion back at the neutral
family, as R138), generated at the R139 optimum with r_bbar at the R138 value (2026-11-04; the removal probes at the
R139 point)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

NESTED_ROOT = Path(__file__).resolve().parents[1]
RUNS = NESTED_ROOT / "config" / "runs"


@pytest.fixture(scope="module")
def engine():
    spec = importlib.util.spec_from_file_location("run_engine_r140_spec_test", NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py")
    e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e)
    e.configure(RUNS / "r140.json")
    return e


def _specs():
    return tuple(json.loads((RUNS / f"{r}.json").read_text()) for r in ("r138", "r139", "r140"))


def test_the_r140_spec_is_r139_without_the_line_pairs():
    r138, r139, r140 = _specs()
    strip = lambda p: {k: v for k, v in p.items() if k != "gen0"}  # noqa: E731
    assert [strip(p) for p in r140["parameters"]] == [strip(p) for p in r139["parameters"]]
    assert r140["kernel"]["baryon_pair_completion"] == "strange_neutral" == r138["kernel"]["baryon_pair_completion"]
    assert "baryon_pair_share" not in r140["kernel"]
    assert {k: v for k, v in r140["kernel"].items() if k != "baryon_pair_completion"} == {
        k: v for k, v in r139["kernel"].items() if k not in ("baryon_pair_completion", "baryon_pair_share")}
    assert r140["kernel"]["closed_channel_reference"] is True and r140["kernel"]["leading_split_axis"] == "gottfried_jackson"
    assert r140["topology"] == r139["topology"] and r140["topology"]["nd_symmetric_remnants"] is True
    assert r140["fit"] == r139["fit"]
    for block in ("diffractive_split", "isovector", "fixed", "statistics", "central_component",
                  "single_diffraction_window", "single_diffraction_window_counts", "systematic_variants"):
        assert r140.get(block) == r139.get(block), block
    assert r140["generation_0"]["run"] == "r139_symmetric_lines_2026-11-04"
    assert r140["generation_0"]["from_fit"] == r139["generation_0"]["from_fit"]
    assert "r_bbar" in r140["generation_0"]["judgment"]
    assert r140["seeds"]["raw_first"] > r139["seeds"]["raw_first"] + 511
    assert r140["seeds"]["partition_base"] != r139["seeds"]["partition_base"]
    path = NESTED_ROOT / ".." / "00runs" / "r139_symmetric_lines_2026-11-04" / "branch_report_f0.json"
    if not path.is_file():
        return  # the gateway suite tree carries no run archive
    report = json.loads(path.read_text())
    r138_report = json.loads((NESTED_ROOT / ".." / "00runs" / "r138_counted_pairs_2026-11-04" / "report.json").read_text())
    for p in r140["parameters"]:
        if p["name"] == "r_bbar":
            assert p["gen0"] == r138_report["fitted"]["r_bbar"]
        elif p["name"] in r140["generation_0"]["from_fit"]:
            assert p["gen0"] == report["fitted"][p["name"]], p["name"]


def test_the_engine_carries_the_options(engine):
    e = engine
    assert e.N_FREE_PARAMETERS == 14 and e.COUNTED_DECAYS is True
    assert e.KERNEL.closed_channel_reference is True and e.KERNEL.leading_split_axis == "gottfried_jackson"
    assert e.TOPOLOGY_OPTIONS["nd_symmetric_remnants"] is True
    assert e.KERNEL.baryon_pair_completion == "strange_neutral" and e.KERNEL.baryon_pair_share is None
