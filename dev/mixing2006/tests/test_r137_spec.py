"""The R137 spec: the R136 contract with the four leading-split choices fitted (2026-11-03, the fourteen-parameter
fit), at the R136 optimum."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

NESTED_ROOT = Path(__file__).resolve().parents[1]
RUNS = NESTED_ROOT / "config" / "runs"
SPLIT = ("leading_split_fraction", "meson_split_fraction", "leading_split_kappa", "meson_split_slope")


@pytest.fixture(scope="module")
def engine():
    spec = importlib.util.spec_from_file_location("run_engine_r137_spec_test", NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py")
    e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e)
    e.configure(RUNS / "r137.json")
    return e


def test_the_r137_spec_is_r136_with_the_split_choices_free():
    r136, r137 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r136", "r137"))
    strip = lambda p: {k: v for k, v in p.items() if k != "gen0"}  # noqa: E731
    assert [strip(p) for p in r137["parameters"] if p["name"] not in SPLIT] == [strip(p) for p in r136["parameters"]]
    assert [p["name"] for p in r137["parameters"]][-1] == "f_central"
    by = {p["name"]: p for p in r137["parameters"]}
    for n in SPLIT:
        assert by[n]["free"] and by[n]["gen0"] == r136["kernel"][n]
    for block in ("kernel", "topology", "fit", "diffractive_split", "isovector", "fixed", "statistics", "central_component",
                  "single_diffraction_window", "single_diffraction_window_counts", "systematic_variants"):
        assert r137.get(block) == r136.get(block), block
    assert r137["generation_0"]["run"] == "r136_fixed_choices_2026-11-03"
    assert r137["seeds"]["raw_first"] > r136["seeds"]["raw_first"] + 511


def test_fourteen_free_parameters_round_trip(engine):
    e = engine
    assert e.N_FREE_PARAMETERS == 14 and e.SPLIT_FREE == SPLIT
    phys = {**{n: float(e.GEN0_SHAPE[e.shape_pos(n)]) for n in e.EXTRA_SHAPE}, "epsilon_nd": 0.26, "b_sd": 5.8,
            "gamma_s_had": 0.28, "r_v": 0.76, "r_bbar": 3.17, "assoc_norm": 1.0, "f_isovector": 1e-3,
            "leading_split_fraction": 0.23, "meson_split_fraction": 0.05, "leading_split_kappa": 4.5, "meson_split_slope": 3.5}
    x = e.x_from_physical(phys, 0.0)
    back = e.physical_from_x(x, 0.0, e.anchors())
    for n in SPLIT:
        assert back[n] == pytest.approx(phys[n], rel=1e-12), n
    share = e.tail_of(back)
    assert len(share) == 6 and share[2:] == tuple(back[n] for n in SPLIT)
