"""The R146 spec (2026-11-06): the R141 model regenerated at the R142 optimum on a new seed block, the next
regeneration step after the R142-R144 replicas moved from the R141 optimum in one direction."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

NESTED_ROOT = Path(__file__).resolve().parents[1]
RUNS = NESTED_ROOT / "config" / "runs"
SAME_BLOCKS = ("kernel", "topology", "diffractive_split", "isovector", "fixed", "statistics", "central_component", "fit",
               "single_diffraction_window", "single_diffraction_window_counts", "systematic_variants", "floor_grid",
               "flavor_bbar_scope", "report_fixed", "frozen_extra")


def _spec(name):
    return json.loads((RUNS / f"{name}.json").read_text())


def test_the_r146_spec_is_r141_at_the_r142_optimum():
    r141, r146 = _spec("r141"), _spec("r146")
    for block in SAME_BLOCKS:
        assert r146.get(block) == r141.get(block), block
    strip = lambda p: {k: v for k, v in p.items() if k != "gen0"}  # noqa: E731
    assert [strip(p) for p in r146["parameters"]] == [strip(p) for p in r141["parameters"]]
    assert r146["generation_0"]["run"] == "r142_replica_a_2026-11-06"
    assert r146["generation_0"]["from_fit"] == r141["generation_0"]["from_fit"]
    others = [json.loads(p.read_text()).get("seeds", {}) for p in RUNS.glob("r*.json") if p.stem != "r146"]
    assert r146["seeds"]["raw_first"] not in {o.get("raw_first") for o in others}
    assert r146["seeds"]["partition_base"] not in {o.get("partition_base") for o in others}
    assert all(abs(r146["seeds"]["raw_first"] - o["raw_first"]) > 511 for o in others if "raw_first" in o)
    path = NESTED_ROOT / ".." / "00runs" / "r142_replica_a_2026-11-06" / "branch_report_f0.json"
    if not path.is_file():
        return  # the gateway suite tree carries no run archive
    report = json.loads(path.read_text())
    assert r146["generation_0"]["branch_digest"] == report["digest"]
    for p in r146["parameters"]:
        if p["name"] in r146["generation_0"]["from_fit"]:
            assert p["gen0"] == report["fitted"][p["name"]], p["name"]


def test_the_engine_reads_the_r146_spec():
    spec = importlib.util.spec_from_file_location("run_engine_r146_spec_test",
                                                  NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py")
    e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e)
    e.configure(RUNS / "r146.json")
    assert e.N_FREE_PARAMETERS == 14 and e.KERNEL.baryon_line_kappa == 1.5
