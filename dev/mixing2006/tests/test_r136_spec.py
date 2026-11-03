"""The R136 spec: the R135 contract with the three fixed discrete choices the paired rescan at R133 moved (2026-11-03):
the leading-meson split share 0.045, the non-diffractive duality mass 2.2 GeV and the line-shape redraw mass 1.2 GeV,
generated at the R135 optimum."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

NESTED_ROOT = Path(__file__).resolve().parents[1]
RUNS = NESTED_ROOT / "config" / "runs"


def test_the_r136_spec_is_r135_with_the_three_choices():
    r135, r136 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r135", "r136"))
    assert r136["kernel"] == {**r135["kernel"], "meson_split_fraction": 0.045}
    assert r136["topology"] == {**r135["topology"], "nd_duality_mass": 2.2, "nd_duality_shape_mass": 1.2}
    strip = lambda p: {k: v for k, v in p.items() if k != "gen0"}  # noqa: E731
    assert [strip(p) for p in r136["parameters"]] == [strip(p) for p in r135["parameters"]]
    for block in ("fit", "diffractive_split", "isovector", "fixed", "statistics", "central_component",
                  "single_diffraction_window", "single_diffraction_window_counts", "systematic_variants",
                  "flavor_bbar_scope", "frozen_extra"):
        assert r136.get(block) == r135.get(block), block
    assert r136["generation_0"]["run"] == "r135_sd_intact_protons_2026-11-02"
    assert r136["seeds"]["raw_first"] > r135["seeds"]["raw_first"] + 511
    spec = importlib.util.spec_from_file_location("run_engine_r136_spec_test", NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py")
    e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e)
    e.configure(RUNS / "r136.json")
    assert e.N_FREE_PARAMETERS == 10 and e.KERNEL.meson_split_fraction == 0.045
    assert e.TOPOLOGY_OPTIONS["nd_duality_mass"] == 2.2 and e.TOPOLOGY_OPTIONS["nd_duality_shape_mass"] == 1.2
