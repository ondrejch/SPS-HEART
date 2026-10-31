"""The single-diffraction anchor in its window (2026-10-23, after R108): the
measured single-diffraction cross section is defined for M^2/s < 0.05, while
the model's single-diffraction class draws its excited mass up to
(sqrt s - m_p)^2. Under the run spec's single_diffraction_window the engine
applies the measured share to the class's masses below the edge, and the class
share follows from the fitted mass law, f_sd_window / F(epsilon)."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.topology_mass_laws import EXCITATION_M_MIN, PROTON_MASS

NESTED_ROOT = Path(__file__).resolve().parents[1]
ENGINE = NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py"
RUNS = NESTED_ROOT / "config" / "runs"
SIGMA_INEL = 31.78


def _engine(tmp_path, run="r109", edit=None, name="engine_window"):
    spec_dict = json.loads((RUNS / f"{run}.json").read_text())
    if edit is not None:
        edit(spec_dict)
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(spec_dict))
    spec = importlib.util.spec_from_file_location(name, ENGINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.configure(path)
    return module


def _gen0_phys(e) -> dict:
    spec = e.SPEC
    phys = {p["name"]: float(p["gen0"] if p.get("gen0") is not None else p["value"]) for p in spec["parameters"]}
    return phys


def test_the_anchor_file_carries_the_measured_window():
    ext = json.loads((NESTED_ROOT / "config" / "v2_singlet" / "external_cross_sections.json").read_text())
    window = ext["sigma_sd_window_mb"]
    assert (window["value"], window["uncertainty"], window["xi_max"]) == (4.9, 0.8, 0.05)
    frac = ext["fractions_of_inelastic"]
    assert frac["f_sd_window"]["value"] == pytest.approx(4.9 / SIGMA_INEL, rel=1e-12)
    assert frac["f_sd_window"]["uncertainty"] == pytest.approx(0.8 / SIGMA_INEL, rel=1e-12)
    assert frac["f_sd_window"]["xi_max"] == window["xi_max"]
    # the anchor of R51 to R108 stays as it was, for the sealed runs
    assert (frac["f_sd_total"]["value"], frac["f_sd_total"]["uncertainty"]) == (0.2045, 0.0472)


def test_the_window_fraction_is_the_class_marginal(tmp_path):
    e = _engine(tmp_path, name="engine_fraction")
    lo, hi = EXCITATION_M_MIN ** 2, (e.SQRT_S - PROTON_MASS) ** 2
    cut = 0.05 * e.SQRT_S ** 2
    for eps in (0.0, 0.25, 0.5639, 0.74):
        # the trapezoid integral of (M^2)^(-1-eps) on a fine grid in ln M^2
        y = np.linspace(math.log(lo), math.log(hi), 400001)
        dens = np.exp(-eps * y)
        cell = 0.5 * (dens[1:] + dens[:-1]) * np.diff(y)
        inside = cell[y[1:] <= math.log(cut)].sum() / cell.sum()
        assert e.sd_window_fraction(eps, 0.05) == pytest.approx(inside, abs=5e-5), eps
    # the R108 sample: 80.3% of the class below M^2/s = 0.05 at epsilon = 0.567
    assert e.sd_window_fraction(0.5671, 0.05) == pytest.approx(0.803, abs=0.002)
    with pytest.raises(ValueError, match="outside the class window"):
        e.sd_window_fraction(0.5, 0.95)


def test_the_class_share_follows_epsilon(tmp_path):
    e = _engine(tmp_path, name="engine_share")
    a = e.anchors()
    assert a["f_sd_window"] == {"value": pytest.approx(4.9 / SIGMA_INEL), "sigma": pytest.approx(0.8 / SIGMA_INEL),
                                "xi_max": 0.05}
    phys0 = _gen0_phys(e)
    for eps in (0.40, 0.5639, 0.74):
        x = e.x_from_physical({**phys0, "epsilon": eps, "f_central": 0.0}, 0.0)
        phys = e.physical_from_x(x, 0.0, a)
        share = (4.9 / SIGMA_INEL) / e.sd_window_fraction(eps, 0.05)
        assert phys["f_sd_total"] == pytest.approx(share, rel=1e-12)
        assert phys["f_nd"] == pytest.approx(1.0 - share - a["f_dd"]["value"], rel=1e-12)
        assert phys["f_sd_window"] == pytest.approx(4.9 / SIGMA_INEL)
        frac = e.class_fractions(phys)
        assert frac["sd_projectile"] == frac["sd_target"] == pytest.approx(share / 2.0)
    # at the R108 optimum the class carries about 6.1 mb
    x = e.x_from_physical({**phys0, "epsilon": 0.5639, "f_central": 0.0}, 0.0)
    assert e.physical_from_x(x, 0.0, a)["f_sd_total"] * SIGMA_INEL == pytest.approx(6.116, abs=0.005)


def test_without_the_window_the_whole_class(tmp_path):
    e = _engine(tmp_path, run="r108", name="engine_legacy")
    assert e.SD_WINDOW is None
    a = e.anchors()
    assert "f_sd_window" not in a
    x = e.x_from_physical({**_gen0_phys(e), "f_central": 0.0}, 0.0)
    phys = e.physical_from_x(x, 0.0, a)
    assert phys["f_sd_total"] == 0.2045 and "f_sd_window" not in phys
    with pytest.raises(ValueError, match="not one of"):
        e.anchors({"f_sd_window": 0.15})


def test_the_window_variants(tmp_path):
    e = _engine(tmp_path, name="engine_window_variants")
    assert e.VARIANT_ANCHORS["sigma_sd_window_4.1mb"] == {"f_sd_window": pytest.approx(4.1 / SIGMA_INEL)}
    assert e.VARIANT_ANCHORS["sigma_sd_window_5.7mb"] == {"f_sd_window": pytest.approx(5.7 / SIGMA_INEL)}
    a = e.anchors(e.VARIANT_ANCHORS["sigma_sd_window_5.7mb"])
    assert a["f_sd_window"]["value"] == pytest.approx(5.7 / SIGMA_INEL)
    assert a["f_sd_window"]["xi_max"] == 0.05 and a["f_sd_window"]["sigma"] == pytest.approx(0.8 / SIGMA_INEL)
    with pytest.raises(ValueError, match="needs the single_diffraction_window"):
        e.systematic_variants({"systematic_variants": [{"label": "w", "f_sd_window": 0.15}]})


def test_a_malformed_window_is_refused(tmp_path):
    for bad in (True, 0.0, 1.5, "0.05"):
        def edit(s, bad=bad):
            s["single_diffraction_window"] = bad
        with pytest.raises(ValueError, match="single_diffraction_window"):
            _engine(tmp_path, edit=edit, name=f"engine_bad_{abs(hash(str(bad)))}")

    def other_edge(s):
        s["single_diffraction_window"] = 0.1
    e = _engine(tmp_path, edit=other_edge, name="engine_other_edge")
    with pytest.raises(ValueError, match="differs from the spec"):
        e.anchors()


def test_the_r109_spec_is_r108_with_the_window():
    r108, r109 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r108", "r109"))
    for block in ("kernel", "topology", "fit", "diffractive_split", "isovector", "fixed", "statistics", "central_component"):
        assert r109[block] == r108[block], block
    assert "single_diffraction_window" not in r108 and r109["single_diffraction_window"] == 0.05
    assert [p["name"] for p in r109["parameters"]] == [p["name"] for p in r108["parameters"]]
    assert r109["generation_0"]["run"] == "r108_systematics_2026-10-22"
    assert r109["seeds"]["raw_first"] > r108["seeds"]["raw_first"] + 511
    assert [v["label"] for v in r109["systematic_variants"]] == [
        "discrepancy_0.03", "discrepancy_0.07", "sigma_sd_window_4.1mb", "sigma_sd_window_5.7mb", "sigma_dd_0.5mb",
        "sigma_dd_2.5mb"]
    branch = NESTED_ROOT.parent / "00runs" / "r108_systematics_2026-10-22" / "branch_report_f0.json"
    if branch.is_file():
        report = json.loads(branch.read_text())
        assert r109["generation_0"]["branch_digest"] == report["digest"]
        gen0 = {p["name"]: p["gen0"] for p in r109["parameters"]}
        for name in r109["generation_0"]["from_fit"]:
            assert float(gen0[name]) == float(report["fitted"][name]), name


def test_the_r110_spec_is_r109_regenerated_at_its_optimum():
    r109, r110 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r109", "r110"))
    for block in ("kernel", "topology", "fit", "diffractive_split", "isovector", "fixed", "statistics", "central_component",
                  "single_diffraction_window", "systematic_variants"):
        assert r110[block] == r109[block], block
    assert [p["name"] for p in r110["parameters"]] == [p["name"] for p in r109["parameters"]]
    assert r110["generation_0"]["run"] == "r109_sd_window_2026-10-23"
    assert r110["seeds"]["raw_first"] > r109["seeds"]["raw_first"] + 511
    branch = NESTED_ROOT.parent / "00runs" / "r109_sd_window_2026-10-23" / "branch_report_f0.json"
    if branch.is_file():
        report = json.loads(branch.read_text())
        assert r110["generation_0"]["branch_digest"] == report["digest"]
        gen0 = {p["name"]: p["gen0"] for p in r110["parameters"]}
        for name in r110["generation_0"]["from_fit"]:
            assert float(gen0[name]) == float(report["fitted"][name]), name
