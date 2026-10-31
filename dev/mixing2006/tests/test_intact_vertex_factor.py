"""The intact proton's vertex factor of the single-diffraction classes
(2026-10-15): under the factorized vertex slopes
``TopologyNominalParams.intact_vertex_factor`` f gives the intact line the
term f, B = b [f + (m_p/M_X)^kappa] / 2; a low-mass resonance remnant keeps
its committed term, and None is the committed event."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

import na49_mixing.topology_single_diffraction as SD
from na49_mixing.kernel_config import KernelConfig, load_kernel
from na49_mixing.topology_mass_laws import PROTON_MASS, vertex_slope_factor
from na49_mixing.topology_remnants import KIND_INTACT_PROTON
from na49_mixing.topology_runner import TopologyNominalParams, run_topology

NESTED_ROOT = Path(__file__).resolve().parents[1]
R96 = json.loads((NESTED_ROOT / "config" / "runs" / "r96.json").read_text())
LABELS = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")
KAPPA = 1.25
BASE = TopologyNominalParams(
    epsilon=0.50, b_sd=6.58, epsilon_nd=0.25, nd_central=False, exact_transfer_range=True, line_thresholds=True,
    transfer_slope_mass_exponent=KAPPA, low_mass_spin_isospin=True,
    remnant_weights={"intact_proton": 0.5, "low_mass_baryonic_resonance": 0.5, "high_mass_baryonic_continuum": 0.0})


@pytest.fixture(scope="module")
def kernel():
    return load_kernel(KernelConfig(**R96["kernel"]), NESTED_ROOT)


def _event(cls, params, kernel, seed):
    catalog, fams = kernel
    rngs = {f"rng_{lab}": np.random.default_rng([seed, 15, k]) for k, lab in enumerate(LABELS)}
    return run_topology(cls, params, catalog, fams, **rngs)


def test_off_and_unit_factor_are_the_committed_event(kernel):
    """None and f = 1 give the committed event, draw for draw."""
    for cls in ("sd_projectile", "sd_target"):
        for seed in range(8):
            ev = _event(cls, BASE, kernel, seed)
            one = _event(cls, dataclasses.replace(BASE, intact_vertex_factor=1.0), kernel, seed)
            assert one.final_state == ev.final_state and one.t == ev.t and one.slope_factor == ev.slope_factor


def test_the_intact_term(kernel):
    """A proton remnant's transfer follows B = b [f + (m_p/M_X)^kappa] / 2;
    a low-mass resonance remnant other than the proton keeps the committed
    factor (the R99 review, M4: the low-mass set includes the proton, which
    carries f as the reweighting reads it)."""
    f = 1.5
    seen = {"intact": 0, "resonance": 0}
    for seed in range(40):
        ev = _event("sd_projectile", dataclasses.replace(BASE, intact_vertex_factor=f), kernel, seed)
        base = vertex_slope_factor(ev.excitation_mass, ev.remnant.mass, KAPPA)
        if ev.remnant.pdg_id == 2212:
            seen["intact"] += 1
            assert ev.remnant.mass == PROTON_MASS
            assert math.isclose(ev.slope_factor, 0.5 * (f + (PROTON_MASS / max(ev.excitation_mass, PROTON_MASS)) ** KAPPA),
                                rel_tol=1e-12)
        else:
            seen["resonance"] += 1
            assert ev.slope_factor == base
    assert seen["intact"] > 0 and seen["resonance"] > 0


def test_the_probe_form(kernel, monkeypatch):
    """The option reproduces the probe's form (PROBE_SD_INTACT_SLOPE: the
    vertex factor of a line at the proton mass raised by (f - 1) / 2)
    event for event, so the probes of 2026-10-13/14 carry over."""
    f = 1.5
    ours = [_event(cls, dataclasses.replace(BASE, intact_vertex_factor=f), kernel, seed)
            for cls in ("sd_projectile", "sd_target") for seed in range(10)]
    orig = SD.vertex_slope_factor

    def probe_factor(m_excitation, m_remnant, kappa):
        base = orig(m_excitation, m_remnant, kappa)
        return base + 0.5 * (f - 1.0) if float(m_remnant) <= PROTON_MASS * (1.0 + 1e-9) else base
    monkeypatch.setattr(SD, "vertex_slope_factor", probe_factor)
    probe = [_event(cls, BASE, kernel, seed) for cls in ("sd_projectile", "sd_target") for seed in range(10)]
    for a, b in zip(ours, probe):
        assert a.final_state == b.final_state and a.t == b.t and a.slope_factor == b.slope_factor


def test_the_other_classes_ignore_it(kernel):
    """The non-diffractive and double-diffraction lines hold no intact
    proton: their events do not change."""
    for cls in ("nd", "dd"):
        for seed in range(6):
            ev = _event(cls, BASE, kernel, seed)
            on = _event(cls, dataclasses.replace(BASE, intact_vertex_factor=1.75), kernel, seed)
            assert on.final_state == ev.final_state


def test_the_refusals(kernel):
    catalog, fams = kernel
    rngs = {f"rng_{lab}": np.random.default_rng([0, 15, k]) for k, lab in enumerate(LABELS)}
    for bad in (dict(intact_vertex_factor=0.0), dict(intact_vertex_factor=-1.0), dict(intact_vertex_factor=float("nan")),
                dict(intact_vertex_factor=True), dict(intact_vertex_factor=1.5, transfer_slope_mass_exponent=None)):
        with pytest.raises(ValueError, match="intact_vertex_factor"):
            run_topology("sd_projectile", dataclasses.replace(BASE, **bad), catalog, fams, **rngs)


def _engine(name):
    path = NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py"
    spec = importlib.util.spec_from_file_location(name, path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    return engine


def test_the_run_spec_option(tmp_path):
    engine = _engine("run_engine_intact_vertex")
    good = dict(R96, topology={**R96["topology"], "intact_vertex_factor": 1.5})
    (tmp_path / "good.json").write_text(json.dumps(good))
    engine.configure(tmp_path / "good.json")
    assert engine.TOPOLOGY_OPTIONS["intact_vertex_factor"] == 1.5
    assert engine._params_at("sd_projectile", (0.5, 6.58, 0.25)).intact_vertex_factor == 1.5
    assert engine._params_at("nd0", (0.5, 6.58, 0.25)).intact_vertex_factor == 1.5
    assert engine._params_at("nd", (0.5, 6.58, 0.25)).intact_vertex_factor is None
    bare = {k: v for k, v in R96["topology"].items() if k != "transfer_slope_mass_exponent"}
    for topology in ({**R96["topology"], "intact_vertex_factor": 0.0}, {**bare, "intact_vertex_factor": 1.5}):
        (tmp_path / "bad.json").write_text(json.dumps(dict(R96, topology=topology)))
        with pytest.raises(ValueError, match="intact_vertex_factor"):
            _engine("run_engine_intact_vertex_bad").configure(tmp_path / "bad.json")


# ----------------------------------------------------------------------
# The free factor (R97): the reweighting of the intact vertex term
# ----------------------------------------------------------------------
from na49_mixing.topology_mass_laws import log_density_ratio, pp_t_window_vec  # noqa: E402

SQRT_S = 17.27
B0 = 6.58


def _sd_sample(n, f0, rng):
    """Single-diffraction arrays at the generation factor f0: half the events
    with the intact proton, half with a 1.232 GeV remnant, masses spread over
    1.2-6 GeV, each transfer drawn from exp(B t) on its exact window."""
    m_x = rng.uniform(1.2, 6.0, n)
    m_rem = np.where(np.arange(n) % 2 == 0, PROTON_MASS, 1.232)
    m2 = m_x * m_x
    lo, hi = pp_t_window_vec(m2, m_rem * m_rem)
    factor = np.array([vertex_slope_factor(a, b, KAPPA) for a, b in zip(m_x, m_rem)])
    factor = factor + 0.5 * (f0 - 1.0) * (m_rem <= PROTON_MASS)
    slope = B0 * factor
    u = rng.random(n)
    e = np.exp(slope * (lo - hi))
    t = hi + np.log(e + u * (1.0 - e)) / slope
    return {"m2": m2, "t": t, "m_rem": m_rem, "slope_factor": factor}, slope, lo, hi


def _mean_t(slope, lo, hi):
    """The mean of t under exp(B t) on [lo, hi]."""
    e = np.exp(slope * (lo - hi))
    return hi - 1.0 / slope + (hi - lo) * e / (1.0 - e)


def test_the_reweighting_of_the_intact_term():
    """Over a sample at f0 the weights to f average to one, the weighted
    mean transfer of the intact events is that of the slope at f, and the
    remnant events keep the weights of the committed reweighting."""
    rng = np.random.default_rng(97)
    f0, n = 1.3, 400000
    arrays, slope0, lo, hi = _sd_sample(n, f0, rng)
    intact = arrays["m_rem"] <= PROTON_MASS
    for f in (1.1, 1.5):
        for cls in ("sd_projectile", "sd_target"):
            log_w = log_density_ratio(cls, arrays, 0.5, B0, eps0=0.5, b_sd0=B0, sqrt_s=SQRT_S, exact_range=True,
                                      intact_factor=(f, f0))
            w = np.exp(log_w)
            assert abs(np.mean(w[intact]) - 1.0) < 6e-3 and np.all(w[~intact] == 1.0)
            slope = B0 * (arrays["slope_factor"] + 0.5 * (f - f0))
            expected = np.mean(_mean_t(slope[intact], lo[intact], hi[intact]))
            assert abs(np.mean(w[intact] * arrays["t"][intact]) - expected) < 3e-3 * abs(expected)
    # with the slope moved as well: the remnant events carry the committed term
    committed = log_density_ratio("sd_projectile", arrays, 0.45, 6.2, eps0=0.5, b_sd0=B0, sqrt_s=SQRT_S, exact_range=True)
    joint = log_density_ratio("sd_projectile", arrays, 0.45, 6.2, eps0=0.5, b_sd0=B0, sqrt_s=SQRT_S, exact_range=True,
                              intact_factor=(1.5, f0))
    assert np.array_equal(joint[~intact], committed[~intact]) and not np.allclose(joint[intact], committed[intact])
    # at f = f0 the term is the committed one to rounding
    same = log_density_ratio("sd_projectile", arrays, 0.45, 6.2, eps0=0.5, b_sd0=B0, sqrt_s=SQRT_S, exact_range=True,
                             intact_factor=(f0, f0))
    assert np.allclose(same, committed, rtol=0.0, atol=1e-12)


def test_the_intact_term_refusals():
    rng = np.random.default_rng(3)
    arrays, *_ = _sd_sample(10, 1.3, rng)
    with pytest.raises(ValueError, match="intact vertex factor"):
        log_density_ratio("dd", {"m12": arrays["m2"], "m22": arrays["m2"], "t1": arrays["t"], "t2": arrays["t"]}, 0.5, B0,
                          eps0=0.5, b_sd0=B0, sqrt_s=SQRT_S, exact_range=True, intact_factor=(1.5, 1.3))
    bare = {k: v for k, v in arrays.items() if k != "slope_factor"}
    with pytest.raises(ValueError, match="intact vertex factor"):
        log_density_ratio("sd_projectile", bare, 0.5, B0, eps0=0.5, b_sd0=B0, sqrt_s=SQRT_S, exact_range=True,
                          intact_factor=(1.5, 1.3))


def _free_spec(f0=1.3):
    params = []
    for p in R96["parameters"]:
        if p["name"] == "f_central":
            params.append({"name": "intact_vertex_factor", "transform": "log", "bounds": [0.5, 3.0], "free": True, "gen0": f0,
                           "hessian_step": 0.04, "starts": ["gen0", 1.0, 1.6, 1.15, 1.45, 2.0, 1.25, 1.8], "midpoint": "geometric",
                           "bound_check": True})
        params.append(p)
    return dict(R96, parameters=params)


def test_the_free_factor_in_the_engine(tmp_path):
    """The registry parameter is a coordinate of the generation point: the
    single-diffraction and nd0 generators carry it, and the weights route it
    to the single-diffraction classes only."""
    import types
    engine = _engine("run_engine_intact_free")
    (tmp_path / "free.json").write_text(json.dumps(_free_spec()))
    engine.configure(tmp_path / "free.json")
    assert engine.INTACT_FREE and engine.EXTRA_SHAPE == ("intact_vertex_factor",)
    assert "intact_vertex_factor" in engine.SHAPE_NAMES and engine.GEN0_SHAPE[3] == 1.3
    point = (0.5, 6.58, 0.25, 1.4)
    assert engine._params_at("sd_target", point).intact_vertex_factor == 1.4
    assert engine._params_at("nd0", point).intact_vertex_factor == 1.4
    assert engine._params_at("nd", point).intact_vertex_factor is None
    phys = engine.physical_from_x(engine.x_from_physical({**{p["name"]: p["gen0"] for p in _free_spec()["parameters"]
                                                            if p.get("gen0") is not None}, "w_excited": 1.0, "f_central": 0.0},
                                                         fc_fixed=0.0), 0.0, {"f_sd_total": {"value": 0.2045}, "f_dd": {"value": 0.0472}})
    assert math.isclose(engine.ivf_of(phys), 1.3)
    rng = np.random.default_rng(5)
    arrays, *_ = _sd_sample(2000, 1.3, rng)
    fake = types.SimpleNamespace(point=(0.5, 6.58, 0.25, 1.3))
    block = {"n": 2000, "shape": arrays}
    got = engine.Templates.log_weights(fake, "sd_projectile", block, 0.5, 6.58, 0.25, f_ivf=1.5)
    want = log_density_ratio("sd_projectile", arrays, 0.5, 6.58, eps0=0.5, b_sd0=6.58, sqrt_s=engine.SQRT_S, exact_range=True,
                             intact_factor=(1.5, 1.3))
    assert np.array_equal(got, want)
    assert np.all(engine.Templates.log_weights(fake, "sd_projectile", block, 0.5, 6.58, 0.25, f_ivf=1.3) == 0.0)


def test_the_free_factor_refusals(tmp_path):
    base = _free_spec()
    fixed = dict(base, parameters=[dict(p, free=False, value=1.3) if p["name"] == "intact_vertex_factor" else p
                                   for p in base["parameters"]])
    both = dict(base, topology={**base["topology"], "intact_vertex_factor": 1.3})
    bare = dict(base, topology={k: v for k, v in base["topology"].items() if k != "transfer_slope_mass_exponent"})
    for k, spec in enumerate((fixed, both, bare)):
        (tmp_path / f"bad{k}.json").write_text(json.dumps(spec))
        with pytest.raises(ValueError, match="intact_vertex_factor"):
            _engine(f"run_engine_intact_bad{k}").configure(tmp_path / f"bad{k}.json")
    short = dict(base, parameters=[dict(p, starts=p["starts"][:6]) if p["name"] == "intact_vertex_factor" else p
                                   for p in base["parameters"]])
    (tmp_path / "short.json").write_text(json.dumps(short))
    with pytest.raises(ValueError, match="dispersed starts"):
        _engine("run_engine_intact_short").configure(tmp_path / "short.json")


def test_the_r97_spec_is_r96_with_the_free_factor():
    """R97: the R96 contract with the intact vertex factor free, generated at
    the R96 optimum with that factor and the two exponents set by judgment."""
    runs = NESTED_ROOT / "config" / "runs"
    r96, r97 = (json.loads((runs / f"{r}.json").read_text()) for r in ("r96", "r97"))
    assert r97["kernel"] == r96["kernel"] and r97["topology"] == r96["topology"]
    for block in ("fit", "diffractive_split", "isovector", "fixed", "statistics"):
        assert r97[block] == r96[block]
    names = [p["name"] for p in r97["parameters"]]
    assert names == [p["name"] for p in r96["parameters"]][:-1] + ["intact_vertex_factor", "f_central"]
    ivf = next(p for p in r97["parameters"] if p["name"] == "intact_vertex_factor")
    assert ivf["free"] is True and ivf["transform"] == "log" and ivf["gen0"] == 1.3 and len(ivf["starts"]) == 8
    g0 = r97["generation_0"]
    assert g0["run"] == "r96_free_rv_2026-10-13" and g0["fit"] == "free"
    assert set(g0["judgment"]) == {"epsilon", "epsilon_nd", "intact_vertex_factor"}
    assert not set(g0["judgment"]) & set(g0["from_fit"]) and "r_v" in g0["from_fit"]
    gen0 = {p["name"]: p["gen0"] for p in r97["parameters"]}
    assert (gen0["epsilon"], gen0["epsilon_nd"]) == (0.5, 0.25)
    assert r97["seeds"]["raw_first"] > r96["seeds"]["raw_first"] + 511
    assert r97["seeds"]["partition_base"] != r96["seeds"]["partition_base"]
    # the generation-0 values are the R96 free optimum of its final branch report (with the sealed archive)
    branch = NESTED_ROOT.parent / "00runs" / "r96_free_rv_2026-10-13" / "branch_report_f0.json"
    if branch.is_file():
        fitted = json.loads(branch.read_text())["fitted"]
        for name in g0["from_fit"]:
            assert float(gen0[name]) == float(fitted[name]), name


def test_generation_matches_the_reweighting_on_isovector_lines(kernel):
    """Reviews 2026-10-16 (M1; M4 of the R99 review): only the proton carries
    the elastic vertex term, whatever its remnant kind (the low-mass set
    includes the ground state) and on vacuum and isovector lines alike, as
    the reweighting of ``topology_mass_laws.log_density_ratio`` reads it (a
    remnant at the proton mass).  Events of one seed generated at f0 and at
    f differ in their slope factor by (f - f0) / 2 on every proton remnant
    and by nothing on the neutron, the Delta(1232) and the other records."""
    f0, f = 1.37, 1.80
    seen = {"intact proton": 0, "low-mass proton": 0, "other": 0}
    for iv in (0.0, 1.0):
        for cls in ("sd_projectile", "sd_target"):
            for seed in range(60):
                p0 = dataclasses.replace(BASE, intact_vertex_factor=f0, isovector_fraction=iv)
                a = _event(cls, p0, kernel, seed)
                b = _event(cls, dataclasses.replace(p0, intact_vertex_factor=f), kernel, seed)
                # the mask of log_density_ratio: a remnant at the proton mass
                at_proton_mass = a.remnant.mass <= PROTON_MASS * (1.0 + 1e-9)
                assert at_proton_mass == (a.remnant.pdg_id == 2212)
                expected = a.slope_factor + (0.5 * (f - f0) if at_proton_mass else 0.0)
                assert math.isclose(b.slope_factor, expected, rel_tol=1e-12)
                if at_proton_mass:
                    seen["intact proton" if a.remnant.kind == KIND_INTACT_PROTON else "low-mass proton"] += 1
                else:
                    seen["other"] += 1
    assert all(v > 0 for v in seen.values()), seen
