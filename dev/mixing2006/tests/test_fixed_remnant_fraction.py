"""The non-diffractive remnant fraction fixed by the run spec (2026-10-13):
with the low-mass non-diffractive lines through the N* and Delta records
(R94) the fit drives the low-mass record kind to zero, and a spec may fix
w_excited at 1 (``free: false``) with the generation weight of the
low-mass kind at 0.  The kind ratio of the reweighting then carries -inf
for that kind, and a fixed fraction is no fitted boundary."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np

NESTED_ROOT = Path(__file__).resolve().parents[1]
ENGINE = NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py"


def _engine(name):
    spec = importlib.util.spec_from_file_location(name, ENGINE)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    return engine


def test_the_kind_ratio():
    """The ratio of each drawn kind is its log weight ratio; the intact
    kind and a kind the generation never drew carry -inf."""
    e = _engine("run_engine_kind_ratio")
    out = e.remnant_log_ratio((0.0, 0.05, 0.95), (0.0, 0.049, 0.951))
    assert out[0] == -math.inf
    assert math.isclose(out[1], math.log(0.05 / 0.049)) and math.isclose(out[2], math.log(0.95 / 0.951))
    out = e.remnant_log_ratio((0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
    assert out[0] == -math.inf and out[1] == -math.inf and out[2] == 0.0


def test_a_spec_with_the_fraction_fixed(tmp_path):
    """w_excited fixed at 1: no shape parameter of its own, the physical
    point carries w_low_mass 0, and the generation draws no low-mass kind."""
    base = json.loads((NESTED_ROOT / "config" / "runs" / "r94.json").read_text())
    params = [dict(p, free=False, value=1.0, gen0=1.0) if p["name"] == "w_excited" else p for p in base["parameters"]]
    g0 = dict(base["generation_0"], from_fit=[n for n in base["generation_0"]["from_fit"] if n not in ("w_excited", "w_low_mass")],
              extra={"w_low_mass": 0.0})
    (tmp_path / "fixed.json").write_text(json.dumps(dict(base, parameters=params, generation_0=g0)))
    e = _engine("run_engine_fixed_fraction")
    e.configure(tmp_path / "fixed.json")
    assert "w_excited" not in e.SHAPE_NAMES and e.FIXED_VALUES["w_excited"] == 1.0
    assert e.ND_GENERATION_REMNANT_WEIGHTS == {"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.0,
                                               "high_mass_baryonic_continuum": 1.0}
    anchors = {"f_sd_total": {"value": 0.2045}, "f_dd": {"value": 0.0472}}
    point = {p["name"]: p["gen0"] for p in base["parameters"]}
    phys = e.physical_from_x(e.x_from_physical({**point, "w_excited": 1.0}, fc_fixed=0.0), 0.0, anchors)
    assert phys["w_excited"] == 1.0 and phys["w_low_mass"] == 0.0
    ratio = e.remnant_log_ratio((phys["w_intact"], phys["w_low_mass"], phys["w_excited"]),
                                [e.ND_GENERATION_REMNANT_WEIGHTS[k] for k in e.REMNANT_KINDS])
    assert ratio[2] == 0.0 and np.all(np.isinf(ratio[:2]))


def test_the_r95_spec_is_r94_without_the_low_mass_kind_at_the_r94_optimum():
    runs = NESTED_ROOT / "config" / "runs"
    r94, r95 = (json.loads((runs / f"{r}.json").read_text()) for r in ("r94", "r95"))
    assert r95["kernel"] == r94["kernel"] and r95["topology"] == r94["topology"]
    for block in ("fit", "diffractive_split", "isovector"):
        assert r95[block] == r94[block]
    w = next(p for p in r95["parameters"] if p["name"] == "w_excited")
    assert w["free"] is False and w["value"] == 1.0 and r95["generation_0"]["extra"] == {"w_low_mass": 0.0}
    assert [p["name"] for p in r95["parameters"]] == [p["name"] for p in r94["parameters"]]
    assert r95["generation_0"]["run"] == "r94_nd_duality_2026-10-12" and r95["generation_0"]["fit"] == "free"
    assert not {"w_excited", "w_low_mass"} & set(r95["generation_0"]["from_fit"])
    assert r95["seeds"]["raw_first"] > r94["seeds"]["raw_first"] + 511
    assert r95["seeds"]["partition_base"] != r94["seeds"]["partition_base"]
    # the generation-0 values are the R94 free optimum of its final branch report (with the sealed archive)
    branch = NESTED_ROOT.parent / "00runs" / "r94_nd_duality_2026-10-12" / "branch_report_f0.json"
    if branch.is_file():
        fitted = json.loads(branch.read_text())["fitted"]
        gen0 = {p["name"]: p["gen0"] for p in r95["parameters"]}
        for name in r95["generation_0"]["from_fit"]:
            assert float(gen0[name]) == float(fitted[name]), name
