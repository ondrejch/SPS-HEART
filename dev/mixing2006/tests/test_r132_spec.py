"""The R132 spec: the R130 contract with the diffractive epsilon at the Donnachie-Landshoff 0.0808 (2026-10-31)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

NESTED_ROOT = Path(__file__).resolve().parents[1]
RUNS = NESTED_ROOT / "config" / "runs"


def test_the_r132_spec_is_r130_with_epsilon_at_the_dl_intercept():
    r130, r132 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r130", "r132"))
    eps130, eps132 = ({p["name"]: p for p in r["parameters"]}["epsilon"] for r in (r130, r132))
    assert eps132["free"] is False and eps132["value"] == 0.0808 and eps132["gen0"] == 0.0808
    assert {k: v for k, v in eps132.items() if k not in ("value", "gen0")} == \
        {k: v for k, v in eps130.items() if k not in ("value", "gen0")}
    strip = lambda p: {k: v for k, v in p.items() if k != "gen0"}  # noqa: E731
    assert [strip(p) for p in r132["parameters"] if p["name"] != "epsilon"] == \
        [strip(p) for p in r130["parameters"] if p["name"] != "epsilon"]
    for block in ("kernel", "topology", "fit", "diffractive_split", "isovector", "fixed", "statistics", "central_component",
                  "single_diffraction_window", "systematic_variants", "flavor_bbar_scope"):
        assert r132[block] == r130[block], block
    assert r132["generation_0"]["run"] == "r131_epsilon_free_2026-10-31"
    assert r132["seeds"]["raw_first"] > json.loads((RUNS / "r131.json").read_text())["seeds"]["raw_first"] + 511
    spec = importlib.util.spec_from_file_location("run_engine_r132_spec_test", NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py")
    e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
    e.configure(RUNS / "r132.json")
    assert e.N_FREE_PARAMETERS == 10 and not e.PARAM_INDEX["epsilon"].free and e.PARAM_INDEX["epsilon"].value == 0.0808
