"""The coherence region of diffraction (2026-10-23, after R109): the topology
option diffractive_xi_max ends the single- and double-diffraction excited mass
windows at M^2/s = xi_max (Goulianos and Montanha, PRD 59 (1999) 114017: the
coherence region xi < 0.1). The generators draw below the edge, the importance
reweighting normalizes on the capped windows (the double-diffraction joint mass
becomes the product of the two marginal masses), and the engine's
single-diffraction window fraction uses the same upper edge."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.rng_substreams import derive_substreams
from na49_mixing.spectral_density import families_from_json
from na49_mixing.topology_double_diffraction import NOMINAL_PARAMS as DD_PARAMS, generate_dd
from na49_mixing.topology_elastic import SQRT_S
from na49_mixing.topology_mass_laws import (EXCITATION_M_MIN, PROTON_MASS, _dd_joint, log_density_ratio,
                                            log_dd_joint_mass, log_power_integral)
from na49_mixing.topology_runner import TopologyNominalParams, run_topology
from na49_mixing.topology_single_diffraction import NOMINAL_PARAMS as SD_PARAMS, NOMINAL_SD_REMNANT_WEIGHTS, generate_sd

NESTED_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
CONTINUUM_PATH = NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json"
ENGINE = NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py"
RUNS = NESTED_ROOT / "config" / "runs"
XI = 0.1
CAP = XI * SQRT_S * SQRT_S


@pytest.fixture(scope="module")
def catalog():
    return HadronCatalog.from_json(CATALOG_PATH)


@pytest.fixture(scope="module")
def fams():
    return families_from_json(str(CONTINUUM_PATH))


def _rngs(i, labels=("spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")):
    subs = derive_substreams(20261023, 0, i)
    return tuple(np.random.default_rng(subs.stream_for(label)) for label in labels)


def test_single_diffraction_draws_below_the_edge(catalog, fams):
    capped = [generate_sd("projectile", SD_PARAMS, NOMINAL_SD_REMNANT_WEIGHTS, catalog, fams, *_rngs(i), xi_max=XI)
              for i in range(60)]
    assert max(e.excitation_mass ** 2 for e in capped) <= CAP * (1.0 + 1e-12)
    free = [generate_sd("projectile", SD_PARAMS, NOMINAL_SD_REMNANT_WEIGHTS, catalog, fams, *_rngs(i)) for i in range(60)]
    assert max(e.excitation_mass ** 2 for e in free) > CAP


def test_double_diffraction_draws_below_the_edge(catalog, fams):
    labels = ("exchange_kinematics", "decay_channel", "decay_kinematics")
    capped = [generate_dd(DD_PARAMS, catalog, fams, *_rngs(i, labels), xi_max=XI) for i in range(40)]
    assert max(max(e.m1, e.m2) ** 2 for e in capped) <= CAP * (1.0 + 1e-12)


def test_the_runner_validates_the_edge(catalog, fams):
    for bad in (True, 0.0, 0.3, float("nan")):
        params = TopologyNominalParams(diffractive_xi_max=bad)
        rngs = dict(zip(("rng_topology_selection", "rng_spectral_state", "rng_exchange_kinematics", "rng_decay_channel",
                         "rng_decay_kinematics"), _rngs(1, ("topology_selection", "spectral_state", "exchange_kinematics",
                                                            "decay_channel", "decay_kinematics"))))
        with pytest.raises(ValueError, match="diffractive_xi_max"):
            run_topology("sd_projectile", params, catalog, fams, **rngs)


def test_the_capped_single_diffraction_ratio_is_the_capped_power_law():
    rng = np.random.default_rng(7)
    m2 = np.exp(rng.uniform(math.log(EXCITATION_M_MIN ** 2), math.log(CAP), 500))
    arrays = {"m2": m2, "t": np.full(m2.size, -0.1), "m_rem": np.full(m2.size, PROTON_MASS)}
    eps, eps0 = 0.16, 0.56
    got = log_density_ratio("sd_projectile", arrays, eps, 6.0, eps0=eps0, b_sd0=6.0, sqrt_s=SQRT_S, xi_max=XI)
    lo = EXCITATION_M_MIN ** 2
    want = -(eps - eps0) * np.log(m2) + log_power_integral(eps0, lo, CAP) - log_power_integral(eps, lo, CAP)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
    # without the edge the window runs to (sqrt s - m_p)^2
    free = log_density_ratio("sd_projectile", arrays, eps, 6.0, eps0=eps0, b_sd0=6.0, sqrt_s=SQRT_S)
    hi = (SQRT_S - PROTON_MASS) ** 2
    want_free = -(eps - eps0) * np.log(m2) + log_power_integral(eps0, lo, hi) - log_power_integral(eps, lo, hi)
    np.testing.assert_allclose(free, want_free, rtol=0, atol=1e-12)


def test_the_capped_double_diffraction_joint_mass_factorizes():
    lo = EXCITATION_M_MIN
    for eps in (0.0, 0.16, 0.56):
        assert _dd_joint(eps, SQRT_S, lo, lo, XI) == pytest.approx(2.0 * log_power_integral(eps, lo * lo, CAP), abs=1e-12)
        assert _dd_joint(eps, SQRT_S, lo, lo, None) == log_dd_joint_mass(eps, SQRT_S, lo, lo)
    with pytest.raises(ValueError, match="closure"):
        _dd_joint(0.2, SQRT_S, lo, lo, 0.3)
    rng = np.random.default_rng(8)
    m12, m22 = (np.exp(rng.uniform(math.log(lo * lo), math.log(CAP), 300)) for _ in range(2))
    arrays = {"m12": m12, "m22": m22, "t1": np.full(300, -0.1), "t2": np.full(300, -0.1)}
    got = log_density_ratio("dd", arrays, 0.16, 6.0, eps0=0.56, b_sd0=6.0, sqrt_s=SQRT_S, xi_max=XI)
    want = (-(0.16 - 0.56) * (np.log(m12) + np.log(m22)) + _dd_joint(0.56, SQRT_S, lo, lo, XI)
            - _dd_joint(0.16, SQRT_S, lo, lo, XI))
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-10)


def _engine(tmp_path, edit, name):
    spec_dict = json.loads((RUNS / "r109.json").read_text())
    edit(spec_dict)
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(spec_dict))
    spec = importlib.util.spec_from_file_location(name, ENGINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.configure(path)
    return module


def test_the_window_fraction_ends_at_the_coherence_edge(tmp_path):
    def capped(s):
        s["topology"]["diffractive_xi_max"] = XI
    e = _engine(tmp_path, capped, "engine_coherence")
    lo = EXCITATION_M_MIN ** 2
    for eps in (0.1, 0.16, 0.56):
        want = math.exp(log_power_integral(eps, lo, 0.05 * SQRT_S ** 2) - log_power_integral(eps, lo, CAP))
        assert e.sd_window_fraction(eps, 0.05) == pytest.approx(want, rel=1e-6)
    assert e.sd_window_fraction(0.16, XI) == 1.0

    def window_beyond(s):
        s["topology"]["diffractive_xi_max"] = 0.04
    with pytest.raises(ValueError, match="beyond the coherence edge"):
        _engine(tmp_path, window_beyond, "engine_window_beyond")

    def bad_edge(s):
        s["topology"]["diffractive_xi_max"] = 0.5
    with pytest.raises(ValueError, match="diffractive_xi_max"):
        _engine(tmp_path, bad_edge, "engine_bad_edge")


def test_the_r111_spec_is_r109_with_the_coherence_edge():
    r109, r111 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r109", "r111"))
    for block in ("kernel", "fit", "diffractive_split", "isovector", "fixed", "statistics", "central_component",
                  "single_diffraction_window", "systematic_variants"):
        assert r111[block] == r109[block], block
    assert r111["topology"] == {**r109["topology"], "diffractive_xi_max": XI}
    assert [p["name"] for p in r111["parameters"]] == [p["name"] for p in r109["parameters"]]
    assert r111["generation_0"]["run"] == "r110_sd_window_replica_2026-10-23"
    assert r111["seeds"]["raw_first"] > json.loads((RUNS / "r110.json").read_text())["seeds"]["raw_first"] + 511
    branch = NESTED_ROOT.parent / "00runs" / "r110_sd_window_replica_2026-10-23" / "branch_report_f0.json"
    if branch.is_file():
        report = json.loads(branch.read_text())
        assert r111["generation_0"]["branch_digest"] == report["digest"]
        gen0 = {p["name"]: p["gen0"] for p in r111["parameters"]}
        for name in r111["generation_0"]["from_fit"]:
            assert float(gen0[name]) == float(report["fitted"][name]), name


def test_the_leading_proton_tail_draws_its_own_window(catalog, fams):
    """generate_sd's tail (2026-10-23, the probe form of the reggeized pion exchange beyond the coherence region): the
    excited mass on [xi_lo s, xi_hi s] with the marginal (M^2)^(-1-eps_tail), the transfer from exp(b t) on the
    exact window."""
    xis = []
    for i in range(40):
        e = generate_sd("projectile", SD_PARAMS, NOMINAL_SD_REMNANT_WEIGHTS, catalog, fams, *_rngs(100 + i), exact_range=True,
                        xi_max=XI, tail=(0.1, 0.5, -2.0, 4.0))
        xis.append(e.excitation_mass ** 2 / SQRT_S ** 2)
    assert 0.1 <= min(xis) and max(xis) <= 0.5
    assert np.mean(xis) > 0.25   # the marginal grows as M^2 (dsigma/dxi ~ xi)
    with pytest.raises(ValueError, match="tail needs"):
        generate_sd("projectile", SD_PARAMS, NOMINAL_SD_REMNANT_WEIGHTS, catalog, fams, *_rngs(1), tail=(0.1, 0.5, -2.0, 4.0))


# ----------------------------------------------------------------------
# the leading-proton tail as a component of the single-diffraction classes (2026-10-23, R112)
# ----------------------------------------------------------------------
TAIL = (0.3, 0.1, 0.5, 1.0, 7.0)


def test_the_tail_component_draws_its_share(catalog, fams):
    events = [generate_sd("projectile", SD_PARAMS, NOMINAL_SD_REMNANT_WEIGHTS, catalog, fams, *_rngs(300 + i), exact_range=True,
                          xi_max=XI, tail_component=TAIL) for i in range(80)]
    tails = [e for e in events if e.tail]
    assert all(e.tail is not None for e in events)
    assert 10 <= len(tails) <= 40   # the share 0.3 of 80
    assert all(0.1 <= e.excitation_mass ** 2 / SQRT_S ** 2 <= 0.5 for e in tails)
    assert all(e.excitation_mass ** 2 <= CAP * (1 + 1e-12) for e in events if not e.tail)
    plain = generate_sd("projectile", SD_PARAMS, NOMINAL_SD_REMNANT_WEIGHTS, catalog, fams, *_rngs(1), exact_range=True)
    assert plain.tail is None


def test_the_shape_arrays_carry_the_tail_flag(catalog, fams):
    import importlib.util as ilu
    spec = ilu.spec_from_file_location("obs_tail", NESTED_ROOT / "helpers" / "benchmarking" / "run_observables.py")
    ob = ilu.module_from_spec(spec)
    import sys
    sys.path.insert(0, str(NESTED_ROOT / "helpers" / "benchmarking"))
    spec.loader.exec_module(ob)
    with_tail = generate_sd("projectile", SD_PARAMS, NOMINAL_SD_REMNANT_WEIGHTS, catalog, fams, *_rngs(5), exact_range=True,
                            tail_component=TAIL)
    assert ob.shape_arrays(with_tail)["tail"] in (0.0, 1.0)
    plain = generate_sd("projectile", SD_PARAMS, NOMINAL_SD_REMNANT_WEIGHTS, catalog, fams, *_rngs(5), exact_range=True)
    assert "tail" not in ob.shape_arrays(plain)


def _tail_engine(tmp_path, name="engine_tail"):
    def edit(s):
        s["topology"]["diffractive_xi_max"] = XI
        s["topology"]["sd_tail"] = [0.1, 0.5, 1.0, 7.0]
        fc = s["parameters"].pop()
        s["parameters"].append({"name": "sd_tail_share", "transform": "identity", "bounds": [0.0, 0.6], "free": True,
                                "gen0": 0.18, "hessian_step": 0.01,
                                "starts": ["gen0", 0.1, 0.25, 0.05, 0.3, 0.15, 0.2, 0.12], "midpoint": "arithmetic",
                                "bound_check": True})
        s["parameters"].append(fc)
    return _engine(tmp_path, edit, name)


def test_the_engine_weights_the_tail_share(tmp_path):
    e = _tail_engine(tmp_path)
    assert e.TAIL_FREE and e.SD_TAIL_LAW == (0.1, 0.5, 1.0, 7.0) and e.TAIL_NAME in e.EXTRA_SHAPE
    params = e._params_at("sd_projectile", e.GEN0_SHAPE)
    assert params.sd_tail == (0.18, 0.1, 0.5, 1.0, 7.0)
    assert e._params_at("nd0", e.GEN0_SHAPE).sd_tail is None
    # the class share: the coherence-region part over (1 - s)
    a = e.anchors()
    phys0 = {p["name"]: float(p["gen0"] if p.get("gen0") is not None else p["value"]) for p in e.SPEC["parameters"]}
    x = e.x_from_physical({**phys0, "f_central": 0.0, "sd_tail_share": 0.25}, 0.0)
    phys = e.physical_from_x(x, 0.0, a)
    assert phys["f_sd_total"] == pytest.approx(e.sd_class_share(a, phys["epsilon"]) / 0.75, rel=1e-12)
    assert phys["sd_tail_share"] == pytest.approx(0.25)

    # the weights of a stand-in block: the tail events take s / s0, the others (1 - s) / (1 - s0)
    class _T:
        point = e.GEN0_SHAPE
    tail = np.array([1.0, 0.0, 0.0, 1.0, 0.0])
    m2 = np.array([60.0, 3.0, 5.0, 80.0, 10.0])
    block = {"n": 5, "shape": {"m2": m2, "t": np.full(5, -0.1), "m_rem": np.full(5, PROTON_MASS),
                               "m2_low": np.full(5, EXCITATION_M_MIN ** 2), "slope_factor": np.full(5, 1.0), "tail": tail}}
    eps0, b0, eps_nd0 = e.GEN0_SHAPE[:3]
    log_w = e.Templates.log_weights(_T(), "sd_projectile", block, eps0, b0, eps_nd0, f_tail=0.36)
    np.testing.assert_allclose(log_w[tail > 0.5], math.log(0.36 / 0.18))
    np.testing.assert_allclose(log_w[tail < 0.5], math.log(0.64 / 0.82))
    # at the generation point the share ratios are the only weights: the mean tail weight of a sample drawn at s0
    # is then s / s0 per tail event against (1 - s) / (1 - s0) per coherence-region event
    log_w0 = e.Templates.log_weights(_T(), "sd_projectile", block, eps0, b0, eps_nd0, f_tail=0.18)
    np.testing.assert_allclose(log_w0, 0.0, atol=1e-12)


def test_the_r112_spec_is_r111_with_the_tail():
    r111, r112 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r111", "r112"))
    for block in ("kernel", "fit", "diffractive_split", "isovector", "fixed", "statistics", "central_component",
                  "single_diffraction_window", "systematic_variants"):
        assert r112[block] == r111[block], block
    assert r112["topology"] == {**r111["topology"], "sd_tail": [0.1, 0.5, 1.0, 7.0]}
    names = [p["name"] for p in r112["parameters"]]
    assert names == [p["name"] for p in r111["parameters"]][:-1] + ["sd_tail_share", "f_central"]
    tail = next(p for p in r112["parameters"] if p["name"] == "sd_tail_share")
    assert (tail["transform"], tail["bounds"], tail["gen0"], tail["free"]) == ("identity", [0.0, 0.6], 0.18, True)
    assert r112["generation_0"]["run"] == "r111_coherence_2026-10-23"
    assert r112["seeds"]["raw_first"] > r111["seeds"]["raw_first"] + 511
    branch = NESTED_ROOT.parent / "00runs" / "r111_coherence_2026-10-23" / "branch_report_f0.json"
    if branch.is_file():
        report = json.loads(branch.read_text())
        assert r112["generation_0"]["branch_digest"] == report["digest"]
        gen0 = {p["name"]: p["gen0"] for p in r112["parameters"]}
        for name in r112["generation_0"]["from_fit"]:
            assert float(gen0[name]) == float(report["fitted"][name]), name


def test_every_event_of_the_tail_class_carries_the_same_shape_arrays(tmp_path):
    """R112 stopped on a block whose slope-factor array was shorter than the class (2026-10-24): the tail events
    carried no factor. Every single-diffraction event of a run with the tail now records the same shape keys."""
    import sys
    e = _tail_engine(tmp_path, name="engine_tail_keys")
    sys.path.insert(0, str(NESTED_ROOT / "helpers" / "benchmarking"))
    import run_observables as ob
    from na49_mixing.kernel_config import load_kernel
    import dataclasses
    cat, fam = load_kernel(dataclasses.replace(e.KERNEL, continuation_floor=float(e.FLOOR_GRID[0])), e.NESTED_ROOT)
    params = e._params_at("sd_target", e.GEN0_SHAPE)
    keys = set()
    tails = 0
    for i in range(60):
        rngs = dict(zip(("rng_topology_selection", "rng_spectral_state", "rng_exchange_kinematics", "rng_decay_channel",
                         "rng_decay_kinematics"), _rngs(600 + i, ("topology_selection", "spectral_state", "exchange_kinematics",
                                                                  "decay_channel", "decay_kinematics"))))
        ev = run_topology("sd_target", params, cat, fam, **rngs)
        tails += bool(ev.tail)
        keys.add(tuple(sorted(ob.shape_arrays(ev))))
    assert tails > 0 and len(keys) == 1, keys


def test_the_r113_spec_is_r112_on_a_new_seed_block():
    r112, r113 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r112", "r113"))
    for block in ("kernel", "topology", "fit", "diffractive_split", "isovector", "fixed", "statistics", "central_component",
                  "single_diffraction_window", "systematic_variants", "parameters", "generation_0"):
        assert r113[block] == r112[block], block
    assert r113["seeds"]["raw_first"] > r112["seeds"]["raw_first"] + 511
    assert r113["identity"]["run_id"] == "r113_sd_tail_2026-10-24"


def test_the_r114_spec_is_r113_regenerated_at_its_optimum():
    r113, r114 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r113", "r114"))
    for block in ("kernel", "topology", "fit", "diffractive_split", "isovector", "fixed", "statistics", "central_component",
                  "single_diffraction_window", "systematic_variants"):
        assert r114[block] == r113[block], block
    assert [p["name"] for p in r114["parameters"]] == [p["name"] for p in r113["parameters"]]
    assert r114["generation_0"]["run"] == "r113_sd_tail_2026-10-24"
    assert "sd_tail_share" in r114["generation_0"]["from_fit"]
    assert r114["seeds"]["raw_first"] > r113["seeds"]["raw_first"] + 511
    branch = NESTED_ROOT.parent / "00runs" / "r113_sd_tail_2026-10-24" / "branch_report_f0.json"
    if branch.is_file():
        report = json.loads(branch.read_text())
        assert r114["generation_0"]["branch_digest"] == report["digest"]
        gen0 = {p["name"]: p["gen0"] for p in r114["parameters"]}
        for name in r114["generation_0"]["from_fit"]:
            assert float(gen0[name]) == float(report["fitted"][name]), name
