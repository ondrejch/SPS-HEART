"""The R135 spec: the R133 contract with the single-diffraction anchor counting every intact proton below its window
edge (single_diffraction_window_counts = "intact_protons"; the review of 2026-11-01, item R6), at the R133 optimum."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

NESTED_ROOT = Path(__file__).resolve().parents[1]
RUNS = NESTED_ROOT / "config" / "runs"


def test_the_r135_spec_is_r133_with_the_anchor_on_every_intact_proton():
    r133, r135 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r133", "r135"))
    assert "single_diffraction_window_counts" not in r133
    assert r135["single_diffraction_window_counts"] == "intact_protons"
    strip = lambda p: {k: v for k, v in p.items() if k != "gen0"}  # noqa: E731
    assert [strip(p) for p in r135["parameters"]] == [strip(p) for p in r133["parameters"]]
    for block in ("kernel", "topology", "fit", "diffractive_split", "isovector", "fixed", "statistics", "central_component",
                  "single_diffraction_window", "systematic_variants", "flavor_bbar_scope", "frozen_extra"):
        assert r135.get(block) == r133.get(block), block
    g0 = r135["generation_0"]
    assert g0["run"] == "r133_pomeron_shrinkage_2026-11-01" and set(g0["judgment"]) == {"epsilon"}
    assert {"b_sd", "intact_vertex_factor"} <= set(g0["from_fit"])
    assert r135["seeds"]["raw_first"] > json.loads((RUNS / "r134.json").read_text())["seeds"]["raw_first"] + 511
    spec = importlib.util.spec_from_file_location("run_engine_r135_spec_test", NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py")
    e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e)
    e.configure(RUNS / "r135.json")
    assert e.N_FREE_PARAMETERS == 10 and not e.PARAM_INDEX["epsilon"].free
    assert e.SD_WINDOW_INTACT and 0.01 < e.NEUTRAL_OPE_WINDOW < 0.02
