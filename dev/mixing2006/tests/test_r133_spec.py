"""The R133 and R134 specs: the R132 contract with the Pomeron's shrinkage in the diffractive classes (topology
diffractive_pomeron_slope = 0.25 GeV^-2, the Donnachie-Landshoff slope; 2026-11-01), epsilon held at 0.0808 (R133) or
free (R134)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

NESTED_ROOT = Path(__file__).resolve().parents[1]
RUNS = NESTED_ROOT / "config" / "runs"


@pytest.mark.parametrize("tag, eps_free", [("r133", False), ("r134", True)])
def test_the_spec_is_r132_with_the_shrinkage(tag, eps_free):
    r132, spec = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r132", tag))
    extra = {"diffractive_pomeron_slope": 0.25}
    if "diffractive_pomeron_linear_t" in spec["topology"]:
        # the shrinkage held to the trajectory's linear range at the intact proton's break
        extra["diffractive_pomeron_linear_t"] = r132["topology"]["intact_t_break"][0]
    assert spec["topology"] == {**r132["topology"], **extra}
    by132 = {p["name"]: p for p in r132["parameters"]}
    for p in spec["parameters"]:
        q = by132[p["name"]]
        if p["name"] == "epsilon" and eps_free:
            assert p["free"] is True and "value" not in p and p["gen0"] == 0.0808
            assert {k: v for k, v in p.items() if k not in ("free", "gen0")} == \
                {k: v for k, v in q.items() if k not in ("free", "gen0", "value")}
            continue
        assert {k: v for k, v in p.items() if k != "gen0"} == {k: v for k, v in q.items() if k != "gen0"}, p["name"]
    for block in ("kernel", "fit", "diffractive_split", "isovector", "statistics", "central_component",
                  "single_diffraction_window", "systematic_variants", "flavor_bbar_scope", "frozen_extra"):
        assert spec.get(block) == r132.get(block), block
    assert spec["generation_0"]["run"] == "r132_epsilon_dl_2026-10-31"
    assert set(spec["generation_0"]["judgment"]) == {"epsilon", "b_sd", "intact_vertex_factor"}
    assert spec["seeds"]["raw_first"] > r132["seeds"]["raw_first"] + 511
    e_spec = importlib.util.spec_from_file_location(f"run_engine_{tag}_spec_test",
                                                    NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py")
    e = importlib.util.module_from_spec(e_spec)
    e_spec.loader.exec_module(e)
    e.configure(RUNS / f"{tag}.json")
    assert e.TOPOLOGY_OPTIONS["diffractive_pomeron_slope"] == 0.25
    assert e.N_FREE_PARAMETERS == (11 if eps_free else 10)
    assert e.PARAM_INDEX["epsilon"].free is eps_free


def test_the_two_specs_draw_apart():
    r133, r134 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r133", "r134"))
    assert r134["seeds"]["raw_first"] > r133["seeds"]["raw_first"] + 511
    assert r133["seeds"]["partition_base"] != r134["seeds"]["partition_base"]
    assert r133["identity"]["run_id"] != r134["identity"]["run_id"]
