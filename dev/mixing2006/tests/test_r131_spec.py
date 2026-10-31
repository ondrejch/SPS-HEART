"""The R131 spec: the R130 contract with the diffractive mass law's epsilon free (2026-10-31)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

NESTED_ROOT = Path(__file__).resolve().parents[1]
RUNS = NESTED_ROOT / "config" / "runs"


def test_the_r131_spec_is_r130_with_epsilon_free():
    r130, r131 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r130", "r131"))
    eps130, eps131 = ({p["name"]: p for p in r["parameters"]}["epsilon"] for r in (r130, r131))
    assert eps130["free"] is False and eps130["value"] == 0.1
    assert eps131["free"] is True and "value" not in eps131 and eps131["gen0"] == 0.1
    assert {k: v for k, v in eps131.items() if k != "free"} == {k: v for k, v in eps130.items() if k not in ("free", "value")}
    strip = lambda p: {k: v for k, v in p.items() if k != "gen0"}  # noqa: E731
    assert [strip(p) for p in r131["parameters"] if p["name"] != "epsilon"] == \
        [strip(p) for p in r130["parameters"] if p["name"] != "epsilon"]
    for block in ("kernel", "topology", "fit", "diffractive_split", "isovector", "fixed", "statistics", "central_component",
                  "single_diffraction_window", "systematic_variants", "flavor_bbar_scope"):
        assert r131[block] == r130[block], block
    assert r131["generation_0"]["run"] == "r130_xi_tower_2026-10-30"
    assert r131["seeds"]["raw_first"] > r130["seeds"]["raw_first"] + 511
    spec = importlib.util.spec_from_file_location("run_engine_r131_spec_test", NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py")
    e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
    e.configure(RUNS / "r131.json")
    assert e.N_FREE_PARAMETERS == 11 and e.PARAM_INDEX["epsilon"].free and "epsilon" in e.SHAPE_NAMES
