"""The R142 to R145 specs (2026-11-06): the R141 model regenerated at the R141 optimum on new seed blocks. R142, R143
and R144 are the replica checks, R145 frees the diffractive intercept epsilon."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

NESTED_ROOT = Path(__file__).resolve().parents[1]
RUNS = NESTED_ROOT / "config" / "runs"
NEW = ("r142", "r143", "r144", "r145")
SAME_BLOCKS = ("kernel", "topology", "diffractive_split", "isovector", "fixed", "statistics", "central_component", "fit",
               "single_diffraction_window", "single_diffraction_window_counts", "systematic_variants", "floor_grid",
               "flavor_bbar_scope", "report_fixed", "frozen_extra")


def _spec(name):
    return json.loads((RUNS / f"{name}.json").read_text())


@pytest.mark.parametrize("name", NEW)
def test_the_spec_is_r141_at_the_r141_optimum(name):
    r141, new = _spec("r141"), _spec(name)
    for block in SAME_BLOCKS:
        assert new.get(block) == r141.get(block), block
    strip = lambda p: {k: v for k, v in p.items() if k not in ("gen0", "free", "value")}  # noqa: E731
    assert [strip(p) for p in new["parameters"]] == [strip(p) for p in r141["parameters"]]
    free = {p["name"]: p["free"] for p in new["parameters"]}
    assert free == {**{p["name"]: p["free"] for p in r141["parameters"]}, **({"epsilon": True} if name == "r145" else {})}
    assert new["generation_0"]["run"] == "r141_aligned_lines_2026-11-05"
    assert new["generation_0"]["from_fit"] == r141["generation_0"]["from_fit"]
    path = NESTED_ROOT / ".." / "00runs" / "r141_aligned_lines_2026-11-05" / "branch_report_f0.json"
    if not path.is_file():
        return  # the gateway suite tree carries no run archive
    report = json.loads(path.read_text())
    assert new["generation_0"]["branch_digest"] == report["digest"]
    for p in new["parameters"]:
        if p["name"] in new["generation_0"]["from_fit"]:
            assert p["gen0"] == report["fitted"][p["name"]], p["name"]


def test_the_seed_blocks_are_new_and_disjoint():
    blocks = [(_spec(n)["seeds"]["raw_first"], _spec(n)["seeds"]["partition_base"]) for n in ("r141",) + NEW]
    raws = sorted(b[0] for b in blocks)
    assert all(b - a > 511 for a, b in zip(raws, raws[1:]))
    assert len({b[1] for b in blocks}) == len(blocks)
    others = [json.loads(p.read_text()).get("seeds", {}) for p in RUNS.glob("r*.json") if p.stem not in NEW]
    assert not {b[0] for b in blocks[1:]} & {o.get("raw_first") for o in others}
    assert not {b[1] for b in blocks[1:]} & {o.get("partition_base") for o in others}


@pytest.mark.parametrize("name", NEW)
def test_the_engine_reads_the_spec(name):
    spec = importlib.util.spec_from_file_location(f"run_engine_{name}_spec_test",
                                                  NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py")
    e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e)
    e.configure(RUNS / f"{name}.json")
    assert e.N_FREE_PARAMETERS == (15 if name == "r145" else 14)
    assert e.KERNEL.baryon_line_kappa == 1.5 and e.N_STARTS_VARIANT == 1
