"""The vector-to-pseudoscalar weight r_V free in the run spec (2026-10-13):
the probes at the R94 and R95 optima found r_V = 1.4 with r_BBbar raised to
hold the pbar yield 122 below the reference with the yields fixed; R96
frees r_V beside the other flavor weights."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

NESTED_ROOT = Path(__file__).resolve().parents[1]
ENGINE = NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py"


def _engine(name):
    spec = importlib.util.spec_from_file_location(name, ENGINE)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    return engine


def test_r_v_free_in_the_engine():
    """r_V free: a flavor shape parameter, no fixed value, carried into the
    physical point and the flavor point in FLAVOR_ORDER."""
    e = _engine("run_engine_free_rv")
    e.configure(NESTED_ROOT / "config" / "runs" / "r96.json")
    assert "r_v" in e.SHAPE_NAMES and "r_v" in e.FLAVOR_FREE and e.R_V_FIXED is None
    assert e.GEN_FLAVOR[e.FLAVOR_ORDER.index("r_v")] == 1.0
    anchors = {"f_sd_total": {"value": 0.2045}, "f_dd": {"value": 0.0472}}
    point = {p.name: p.gen0 for p in e.PARAMS}
    phys = e.physical_from_x(e.x_from_physical({**point, "r_v": 1.3}, fc_fixed=0.0), 0.0, anchors)
    assert abs(phys["r_v"] - 1.3) < 1e-12 and abs(e.flavor_of(phys)[1] - 1.3) < 1e-12


def test_r_v_fixed_or_free_is_required(tmp_path):
    base = json.loads((NESTED_ROOT / "config" / "runs" / "r96.json").read_text())
    bad = dict(base, parameters=[p for p in base["parameters"] if p["name"] != "r_v"], fixed={})
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    e = _engine("run_engine_rv_missing")
    with pytest.raises(ValueError, match="r_v"):
        e.configure(tmp_path / "bad.json")


def test_the_r96_spec_is_r95_with_r_v_free_at_the_r95_optimum():
    runs = NESTED_ROOT / "config" / "runs"
    r95, r96 = (json.loads((runs / f"{r}.json").read_text()) for r in ("r95", "r96"))
    assert r96["kernel"] == r95["kernel"] and r96["topology"] == r95["topology"]
    for block in ("fit", "diffractive_split", "isovector"):
        assert r96[block] == r95[block]
    assert "r_v" in r95["fixed"] and "r_v" not in r96["fixed"]
    rv = next(p for p in r96["parameters"] if p["name"] == "r_v")
    rb = next(p for p in r96["parameters"] if p["name"] == "r_bbar")
    assert rv["free"] is True and rv["gen0"] == 1.0 and rv["transform"] == "log" and rv["bounds"] == rb["bounds"]
    assert [p["name"] for p in r96["parameters"] if p["name"] != "r_v"] == [p["name"] for p in r95["parameters"]]
    assert r96["generation_0"]["run"] == "r95_no_low_mass_kind_2026-10-13"
    assert r96["seeds"]["raw_first"] > r95["seeds"]["raw_first"] + 511
    assert r96["seeds"]["partition_base"] != r95["seeds"]["partition_base"]
    branch = NESTED_ROOT.parent / "00runs" / "r95_no_low_mass_kind_2026-10-13" / "branch_report_f0.json"
    if branch.is_file():
        fitted = json.loads(branch.read_text())["fitted"]
        gen0 = {p["name"]: p["gen0"] for p in r96["parameters"]}
        for name in r96["generation_0"]["from_fit"]:
            if name in gen0:
                assert float(gen0[name]) == float(fitted[name]), name
