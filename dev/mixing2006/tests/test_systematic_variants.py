"""The spec's systematic variants of the run engine (2026-10-22, the
systematics campaign): extra refits on the run's own sample beside the three
committed variants, the model floor of the chi2 variance, the shared
normalization and the class-fraction anchors of the external cross sections
(sigma_SD, sigma_DD) replaced where a variant names them."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

NESTED_ROOT = Path(__file__).resolve().parents[1]
ENGINE = NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py"
COMMITTED = ("discrepancy_0.10", "normalization_+0.02", "normalization_-0.02")


def _engine(tmp_path, extra=None, name="engine_variants"):
    spec_dict = json.loads((NESTED_ROOT / "config" / "runs" / "r107.json").read_text())
    if extra is not None:
        spec_dict["systematic_variants"] = extra
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(spec_dict))
    spec = importlib.util.spec_from_file_location(name, ENGINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.configure(path)
    return module


def test_without_systematic_variants_the_committed_three(tmp_path):
    e = _engine(tmp_path, name="engine_plain")
    assert [v[0] for v in e.VARIANTS] == list(COMMITTED)
    assert all(len(v) == 3 for v in e.VARIANTS) and e.VARIANT_ANCHORS == {}


def test_the_spec_variants_follow_the_committed_ones(tmp_path):
    extra = [{"label": "discrepancy_0.03", "discrepancy": 0.03},
             {"label": "sigma_sd_5.0mb", "f_sd_total": 5.0 / 31.78},
             {"label": "sigma_dd_2.5mb", "f_dd": 2.5 / 31.78, "normalization": 0.01}]
    e = _engine(tmp_path, extra, name="engine_extra")
    assert [v[0] for v in e.VARIANTS] == list(COMMITTED) + [x["label"] for x in extra]
    assert e.VARIANTS[3] == ("discrepancy_0.03", 0.03, e.NORM_FIXED)
    assert e.VARIANTS[5] == ("sigma_dd_2.5mb", e.DISCREPANCY, 0.01)
    assert e.VARIANT_ANCHORS == {"sigma_sd_5.0mb": {"f_sd_total": pytest.approx(5.0 / 31.78)},
                                 "sigma_dd_2.5mb": {"f_dd": pytest.approx(2.5 / 31.78)}}
    a = e.anchors(e.VARIANT_ANCHORS["sigma_sd_5.0mb"])
    base = e.anchors()
    assert a["f_sd_total"]["value"] == pytest.approx(5.0 / 31.78) and a["f_dd"] == base["f_dd"]
    assert a["f_sd_total"]["sigma"] == base["f_sd_total"]["sigma"] and "override" not in base


def test_malformed_variants_are_refused(tmp_path):
    e = _engine(tmp_path, name="engine_refuse")
    with pytest.raises(ValueError, match="takes a label"):
        e.systematic_variants({"systematic_variants": [{"label": "x", "sigma_sd": 5.0}]})
    with pytest.raises(ValueError, match="distinct and new"):
        e.systematic_variants({"systematic_variants": [{"label": "discrepancy_0.10", "discrepancy": 0.1}]})
    with pytest.raises(ValueError, match="distinct and new"):
        e.systematic_variants({"systematic_variants": [{"label": "a"}, {"label": "a"}]})
    with pytest.raises(ValueError, match="not one of"):
        e.anchors({"f_nd": 0.7})


def test_the_r108_spec_is_r107_with_the_variants():
    runs = NESTED_ROOT / "config" / "runs"
    r107, r108 = (json.loads((runs / f"{r}.json").read_text()) for r in ("r107", "r108"))
    for block in ("kernel", "topology", "fit", "diffractive_split", "isovector", "fixed", "statistics"):
        assert r108[block] == r107[block], block
    assert [p["name"] for p in r108["parameters"]] == [p["name"] for p in r107["parameters"]]
    assert r108["generation_0"]["run"] == "r107_pair_s_wave_2026-10-22"
    assert r108["seeds"]["raw_first"] > r107["seeds"]["raw_first"] + 511
    assert [v["label"] for v in r108["systematic_variants"]] == [
        "discrepancy_0.03", "discrepancy_0.07", "sigma_sd_5.0mb", "sigma_sd_8.0mb", "sigma_dd_0.5mb", "sigma_dd_2.5mb"]
    branch = NESTED_ROOT.parent / "00runs" / "r107_pair_s_wave_2026-10-22" / "branch_report_f0.json"
    if branch.is_file():
        report = json.loads(branch.read_text())
        assert r108["generation_0"]["branch_digest"] == report["digest"]
        gen0 = {p["name"]: p["gen0"] for p in r108["parameters"]}
        for name in r108["generation_0"]["from_fit"]:
            assert float(gen0[name]) == float(report["fitted"][name]), name
