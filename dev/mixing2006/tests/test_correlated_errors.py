"""The correlated error model of the NA49 spectra (2026-10-28, spec fit.correlated_errors): the scale nuisances
profiled in closed form, the neutron table's common scale and per-point remainder, the spec block's validation, and
the committed objective bit for bit without the block."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize

NESTED_ROOT = Path(__file__).resolve().parents[1]
ENGINE = NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py"
RUNS = NESTED_ROOT / "config" / "runs"


def _engine(name="run_engine_correlated_test"):
    spec = importlib.util.spec_from_file_location(name, ENGINE)
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    return E


def test_the_profile_is_the_minimum_over_the_scales():
    E = _engine()
    rng = np.random.default_rng(3)
    y = rng.normal(size=40)
    design = np.zeros((40, 3))
    design[:20, 0] = rng.uniform(0.5, 2.0, 20)
    design[20:, 1] = rng.uniform(0.5, 2.0, 20)
    design[10:30, 2] = rng.uniform(0.5, 2.0, 20)
    priors = np.array([0.015, 0.08, 0.10])
    eta, residual, penalty = E.profile_scales(y, design, priors)
    value = float(residual @ residual) + penalty
    f = lambda e: float(np.sum((y - design @ e) ** 2)) + float(np.sum((e / priors) ** 2))  # noqa: E731
    brute = minimize(f, np.zeros(3), method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
    assert value == pytest.approx(brute.fun, rel=1e-9) and np.allclose(eta, brute.x, atol=1e-6)
    assert value <= f(np.zeros(3)) + 1e-12
    # a tight prior pins the scale, a loose one lets it fit
    eta_tight, _, _ = E.profile_scales(y, design, np.array([1e-6, 1e-6, 1e-6]))
    assert np.allclose(eta_tight, 0.0, atol=1e-6)


def test_the_neutron_remainder():
    E = _engine()
    data = np.array([0.481, 0.407, 0.378, 0.325, 0.325, 0.293, 0.286, 0.215])
    pct = np.array([20.8, 14.7, 13.2, 11.5, 12.3, 10.2, 10.5, 27.9])
    sigma = data * pct / 100.0
    rem = E.neutron_remainder(sigma, data, 0.10, 0.05)
    assert np.all(rem >= 0.05 * data - 1e-12) and np.all(rem <= sigma + 1e-12)
    assert rem[5] == pytest.approx(0.05 * data[5]) and rem[0] == pytest.approx(np.sqrt(sigma[0] ** 2 - (0.1 * data[0]) ** 2))


def test_the_block_validates():
    E = _engine()
    ok = E.parse_correlated_errors({"normalization": 0.015, "kaon_identification": {"k_plus": 0.08, "k_minus": 0.04, "xf_min": 0.25},
                                    "neutron_scale": 0.10, "neutron_floor": 0.05}, True)
    assert ok["normalization"] == 0.015 and ok["kaon_identification"]["xf_min"] == 0.25 and ok["neutron_floor"] == 0.05
    assert E.parse_correlated_errors(None, True) is None
    assert E.parse_correlated_errors({"normalization": 0.0}, False) == {"normalization": 0.0}
    for bad, neutron in (({}, True), ({"normalisation": 0.015}, True), ({"normalization": 1.5}, True),
                         ({"normalization": True}, True), ({"kaon_identification": {"k_plus": 0.08}}, True),
                         ({"neutron_scale": 0.1}, True), ({"neutron_scale": 0.1, "neutron_floor": 0.05}, False),
                         ({"neutron_scale": 0.0, "neutron_floor": 0.05}, True)):
        with pytest.raises(ValueError, match="correlated_errors"):
            E.parse_correlated_errors(bad, neutron)


def test_the_identification_knot_form():
    E = _engine("run_engine_knots_test")
    knots = {"k_plus": [[0.25, 0.04], [0.40, 0.12]], "k_minus": [[0.25, 0.0], [0.50, 0.06]]}
    ok = E.parse_correlated_errors({"normalization": 0.015, "kaon_identification": knots}, False)
    assert ok["kaon_identification"] == knots and "xf_min" not in ok["kaon_identification"]
    # the amplitude: zero below the first knot, linear between, held beyond the last
    amp = E.identification_amplitude(np.array([0.0, 0.2, 0.25, 0.3, 0.325, 0.4, 0.5]), knots["k_plus"])
    assert np.allclose(amp, [0.0, 0.0, 0.04, 0.04 + 0.08 / 3, 0.08, 0.12, 0.12])
    amp = E.identification_amplitude(np.array([0.25, 0.3, 0.4, 0.5]), knots["k_minus"])
    assert np.allclose(amp, [0.0, 0.012, 0.036, 0.06])
    for bad in ({"k_plus": [[0.25, 0.04]], "k_minus": [[0.25, 0.0], [0.5, 0.06]]},
                {"k_plus": [[0.40, 0.04], [0.25, 0.12]], "k_minus": [[0.25, 0.0], [0.5, 0.06]]},
                {"k_plus": [[0.25, 0.04], [0.40, 0.12]], "k_minus": [[0.25, 0.0], [0.5, 0.0]]},
                {"k_plus": [[0.25, 0.04], [0.40, 1.2]], "k_minus": [[0.25, 0.0], [0.5, 0.06]]},
                {"k_plus": [[0.25, 0.04], [0.40, 0.12]]},
                {"k_plus": 0.08, "k_minus": [[0.25, 0.0], [0.5, 0.06]]}):
        with pytest.raises(ValueError, match="correlated_errors"):
            E.parse_correlated_errors({"kaon_identification": bad}, False)
    # a constant amplitude on one scale of prior 1 is the R125 form with that prior: the profile is the same
    rng = np.random.default_rng(5)
    y = rng.normal(size=30); base = np.zeros((30, 1)); base[10:, 0] = rng.uniform(0.5, 2.0, 20)
    e1, r1, p1 = E.profile_scales(y, base, np.array([0.08]))
    e2, r2, p2 = E.profile_scales(y, 0.08 * base, np.array([1.0]))
    assert np.allclose(r1, r2) and p1 == pytest.approx(p2) and e1[0] == pytest.approx(0.08 * e2[0])


def test_the_r128_spec_is_r127_with_the_identification_knots():
    r127, r128 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r127", "r128"))
    c127, c128 = r127["fit"]["correlated_errors"], r128["fit"]["correlated_errors"]
    assert {k: v for k, v in c128.items() if k != "kaon_identification"} == {k: v for k, v in c127.items() if k != "kaon_identification"}
    assert c128["kaon_identification"] == {"k_plus": [[0.25, 0.04], [0.40, 0.12]], "k_minus": [[0.25, 0.0], [0.50, 0.06]]}
    assert {k: v for k, v in r128["fit"].items() if k != "correlated_errors"} == {k: v for k, v in r127["fit"].items() if k != "correlated_errors"}
    for block in ("kernel", "topology", "diffractive_split", "isovector", "fixed", "statistics", "central_component",
                  "single_diffraction_window", "systematic_variants", "flavor_bbar_scope"):
        assert r128[block] == r127[block], block
    strip = lambda p: {k: v for k, v in p.items() if k not in ("gen0",)}  # noqa: E731
    assert [strip(p) for p in r128["parameters"]] == [strip(p) for p in r127["parameters"]]
    assert r128["generation_0"]["run"] == "r127_excited_kaons_2026-10-28"
    assert r128["seeds"]["raw_first"] > r127["seeds"]["raw_first"] + 511
    E = _engine("run_engine_r128_spec_test")
    E.configure(RUNS / "r128.json")
    assert E.CORRELATED_ERRORS["kaon_identification"] == c128["kaon_identification"]


def test_the_r125_spec_is_r124_with_the_error_model(tmp_path):
    r124, r125 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r124", "r125"))
    assert {k: v for k, v in r125["fit"].items() if k != "correlated_errors"} == r124["fit"]
    assert r125["fit"]["correlated_errors"] == {"normalization": 0.015,
                                                "kaon_identification": {"k_plus": 0.08, "k_minus": 0.04, "xf_min": 0.25},
                                                "neutron_scale": 0.10, "neutron_floor": 0.05}
    for block in ("kernel", "topology", "diffractive_split", "isovector", "fixed", "statistics", "central_component",
                  "single_diffraction_window", "systematic_variants", "flavor_bbar_scope"):
        assert r125[block] == r124[block], block
    strip = lambda p: {k: v for k, v in p.items() if k not in ("gen0",)}  # noqa: E731
    assert [strip(p) for p in r125["parameters"]] == [strip(p) for p in r124["parameters"]]
    g0 = r125["generation_0"]
    assert g0["run"] == "r124_ope_absorption_2026-10-27"
    assert r125["seeds"]["raw_first"] > r124["seeds"]["raw_first"] + 511
    E = _engine("run_engine_r125_spec_test")
    E.configure(RUNS / "r125.json")
    assert E.CORRELATED_ERRORS == r125["fit"]["correlated_errors"]
    E2 = _engine("run_engine_r124_spec_test")
    E2.configure(RUNS / "r124.json")
    assert E2.CORRELATED_ERRORS is None
    s = json.loads((RUNS / "r125.json").read_text())
    s["fit"]["correlated_errors"] = {"normalization": 0.015, "neutron_scale": 0.1}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(s))
    with pytest.raises(ValueError, match="correlated_errors"):
        _engine("run_engine_r125_bad").configure(path)


def test_the_r126_spec_is_the_replica_of_r125():
    r125, r126 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r125", "r126"))
    for block in ("kernel", "topology", "diffractive_split", "isovector", "fixed", "statistics", "central_component",
                  "single_diffraction_window", "systematic_variants", "flavor_bbar_scope", "fit"):
        assert r126[block] == r125[block], block
    strip = lambda p: {k: v for k, v in p.items() if k not in ("gen0",)}  # noqa: E731
    assert [strip(p) for p in r126["parameters"]] == [strip(p) for p in r125["parameters"]]
    g0 = r126["generation_0"]
    assert g0["run"] == "r125_correlated_errors_2026-10-28"
    assert {k: v for k, v in g0.items() if k not in ("run", "branch_digest")} == \
        {k: v for k, v in r125["generation_0"].items() if k not in ("run", "branch_digest")}
    assert r126["seeds"]["raw_first"] > r125["seeds"]["raw_first"] + 511
    assert r126["seeds"]["partition_base"] != r125["seeds"]["partition_base"]
    assert r126["identity"]["run_id"] == "r126_correlated_errors_replica_2026-10-28"
    E = _engine("run_engine_r126_spec_test")
    E.configure(RUNS / "r126.json")
    assert E.CORRELATED_ERRORS == r125["fit"]["correlated_errors"]
