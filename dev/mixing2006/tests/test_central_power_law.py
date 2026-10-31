"""The power central mass law of the class with the central source
(2026-10-07, R81): ``TopologyNominalParams.epsilon_c`` draws M_c^2 from
(M_c^2)^(-1-epsilon_c) on the kinematic window, with no transfer coupling,
and the run engine's ``epsilon_c`` parameter reweights it exactly."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from na49_mixing.rng_substreams import derive_substreams
from na49_mixing.topology_mass_laws import PI0_MASS, log_density_ratio
from na49_mixing.topology_runner import NOMINAL_PARAMS, _committed_surfaces, run_topology

NESTED_ROOT = Path(__file__).resolve().parents[1]
ENGINE = NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py"
LABELS = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")


def _arrays(mc2, m1, m2):
    n = len(mc2)
    return {"t1": np.full(n, -0.3), "t2": np.full(n, -0.4), "t_c": np.full(n, -0.7), "mc2": np.asarray(mc2, float),
            "m1": np.asarray(m1, float), "m2": np.asarray(m2, float), "k1": np.zeros(n), "k2": np.zeros(n),
            "lo1": np.full(n, math.nan), "hi1": np.full(n, math.nan), "lo2": np.full(n, math.nan), "hi2": np.full(n, math.nan)}


def test_the_power_term_is_the_exact_density_ratio():
    sq = 17.270
    m1, m2 = np.array([1.2, 3.0]), np.array([2.0, 1.5])
    mc2 = np.array([0.8, 6.0])
    eps_c, eps_c0 = 0.35, -0.2
    out = log_density_ratio("nd", _arrays(mc2, m1, m2), 0.2, 3.0, eps0=0.2, b_sd0=3.0, sqrt_s=sq,
                            central_law="power", eps_c=eps_c, eps_c0=eps_c0)
    a = PI0_MASS * PI0_MASS
    for k in range(2):
        c = (sq - m1[k] - m2[k]) ** 2
        z = lambda e: quad(lambda x: x ** (-1.0 - e), a, c)[0]
        exact = (-1.0 - eps_c) * math.log(mc2[k]) - math.log(z(eps_c)) - ((-1.0 - eps_c0) * math.log(mc2[k]) - math.log(z(eps_c0)))
        assert out[k] == pytest.approx(exact, abs=1e-9)
    # the committed law is unchanged when the option is absent
    base = log_density_ratio("nd", _arrays(mc2, m1, m2), 0.25, 3.0, eps0=0.2, b_sd0=3.0, sqrt_s=sq)
    assert np.all(np.isfinite(base))
    with pytest.raises(ValueError):
        log_density_ratio("nd", _arrays(mc2, m1, m2), 0.2, 3.0, eps0=0.2, b_sd0=3.0, sqrt_s=sq, central_law="power")


def test_the_generator_draws_the_power_law_and_closes():
    catalog, fams = _committed_surfaces()
    masses = []
    for eps_c in (-0.5, 0.5):
        params = dataclasses.replace(NOMINAL_PARAMS, epsilon_c=eps_c)
        mcs = []
        for i in range(40):
            subs = derive_substreams(20261007, 3, i)
            rngs = {f"rng_{lab}": np.random.default_rng(subs.stream_for(lab)) for lab in LABELS}
            ev = run_topology("nd", params, catalog, fams, **rngs)
            tot = np.sum([h.four_momentum for h in ev.final_state], axis=0)
            assert abs(tot[1]) < 1e-7 and abs(tot[2]) < 1e-7
            mcs.append(ev.central_mass)
        masses.append(np.mean(np.log(np.asarray(mcs) ** 2)))
    # a smaller exponent puts the central systems higher on the window
    assert masses[0] > masses[1] + 0.5


def _engine(name):
    spec = importlib.util.spec_from_file_location(name, ENGINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_engine_parameter(tmp_path):
    spec = json.loads((NESTED_ROOT / "config" / "runs" / "r80.json").read_text())
    params = [p for p in spec["parameters"] if p["name"] != "f_central"]
    params.append({"name": "epsilon_c", "transform": "identity", "bounds": [-1.5, 1.5], "free": True, "gen0": 0.0,
                   "hessian_step": 0.02, "bound_check": True, "starts": ["gen0", -0.5, 0.5, 1.5, 0.2, -0.3, 1.0, 2.5]})
    params.append(next(p for p in spec["parameters"] if p["name"] == "f_central"))
    spec["parameters"] = params
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    e = _engine("run_engine_central_power")
    e.configure(path)
    assert e.CENTRAL_POWER and e.EXTRA_SHAPE == ("epsilon_c",) and e.GEN0_SHAPE[3] == 0.0
    nd = e._params_at("nd", e.GEN0_SHAPE, e.GEN_FLAVOR)
    assert nd.epsilon_c == 0.0 and nd.nd_central
    phys = {**{n: 1.0 for n in e.SHAPE_NAMES}, "w_excited": 0.8, "epsilon_c": 0.3, "f_central": 0.5}
    assert e.epsc_of(e.physical_from_x(e.x_from_physical(phys, None), None, e.anchors())) == pytest.approx(0.3)
    # the null fit has no central events: its epsilon_c never moves the regeneration point
    free = {**{n: 1.0 for n in e.SHAPE_NAMES + e.FLAVOR_ORDER}, "epsilon_c": 0.22}
    null = {**free, "epsilon": 0.6, "epsilon_c": 2.9999}
    for ok_free, ok_null, basis in ((True, True, "midpoint"), (False, True, "null only"), (True, False, "free only")):
        point, got = e.regeneration_point_of({"physical": free, "in_trust_region": ok_free},
                                             {"physical": null, "in_trust_region": ok_null})
        assert got == basis and point["epsilon_c"] == 0.22
    assert e.regeneration_point_of({"physical": free, "in_trust_region": True},
                                   {"physical": null, "in_trust_region": True})[0]["epsilon"] == pytest.approx(0.8)
    spec["central_component"] = False
    for p in spec["parameters"]:
        if p["name"] == "f_central":
            p.update({"free": False, "value": 0.0, "gen0": 0.0})
    spec["topology"].pop("central_recoil", None)
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="needs central_component true"):
        _engine("run_engine_central_power_bad").configure(path)
