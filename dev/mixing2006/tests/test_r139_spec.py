"""The R139 spec: the R138 contract with the symmetric non-diffractive remnants, the channels closed at the pole, the
Gottfried-Jackson split axis (2026-11-04; the code reviews muse13f M6 and grok47) and the line-emitted nucleon pairs
(Fischer's n+p pbar correlation); generated at the R138 optimum with r_bbar at a stated fraction of it."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

NESTED_ROOT = Path(__file__).resolve().parents[1]
RUNS = NESTED_ROOT / "config" / "runs"
ADDED = {"closed_channel_reference": True, "leading_split_axis": "gottfried_jackson",
         "baryon_pair_completion": "charged_nucleon", "baryon_pair_completion_members": "octet", "baryon_pair_share": "line"}


@pytest.fixture(scope="module")
def engine():
    spec = importlib.util.spec_from_file_location("run_engine_r139_spec_test", NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py")
    e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e)
    e.configure(RUNS / "r139.json")
    return e


def _specs():
    return tuple(json.loads((RUNS / f"{r}.json").read_text()) for r in ("r138", "r139"))


def test_the_r139_spec_is_r138_with_the_four_pieces():
    r138, r139 = _specs()
    strip = lambda p: {k: v for k, v in p.items() if k != "gen0"}  # noqa: E731
    assert [strip(p) for p in r139["parameters"]] == [strip(p) for p in r138["parameters"]]
    added = ADDED
    assert {k: v for k, v in r139["kernel"].items() if k not in added} == {k: v for k, v in r138["kernel"].items()
                                                                          if k not in added}
    assert {k: r139["kernel"][k] for k in added} == added
    assert r139["topology"] == {**r138["topology"], "nd_symmetric_remnants": True}
    assert r139["fit"] == r138["fit"]
    for block in ("diffractive_split", "isovector", "fixed", "statistics", "central_component",
                  "single_diffraction_window", "single_diffraction_window_counts", "systematic_variants"):
        assert r139.get(block) == r138.get(block), block
    assert r139["generation_0"]["run"] == "r138_counted_pairs_2026-11-04"
    assert r139["seeds"]["raw_first"] > r138["seeds"]["raw_first"] + 511
    assert r139["seeds"]["partition_base"] != r138["seeds"]["partition_base"]
    path = NESTED_ROOT / ".." / "00runs" / "r138_counted_pairs_2026-11-04" / "branch_report_f0.json"
    if not path.is_file():
        return  # the gateway suite tree carries no run archive
    report = json.loads(path.read_text())
    assert "r_bbar" not in r139["generation_0"]["from_fit"]
    assert set(r139["generation_0"]["from_fit"]) == set(r138["generation_0"]["from_fit"]) - {"r_bbar"}
    factor = float(r139["generation_0"]["judgment"]["r_bbar"].split(" times")[0])
    assert 0.3 <= factor < 1.0
    for p in r139["parameters"]:
        if p["name"] == "r_bbar":
            assert p["gen0"] == report["fitted"]["r_bbar"] * factor
        elif p["name"] in r139["generation_0"]["from_fit"]:
            assert p["gen0"] == report["fitted"][p["name"]], p["name"]


def test_the_engine_carries_the_options(engine):
    e = engine
    assert e.N_FREE_PARAMETERS == 14 and e.COUNTED_DECAYS is True
    assert e.KERNEL.closed_channel_reference is True and e.KERNEL.leading_split_axis == "gottfried_jackson"
    assert e.TOPOLOGY_OPTIONS["nd_symmetric_remnants"] is True
    assert e.KERNEL.baryon_pair_completion == "charged_nucleon" and e.KERNEL.baryon_pair_share == "line"
