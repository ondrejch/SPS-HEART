"""The intact proton's two-slope transfer (2026-10-24, after the probes at the R113 optimum): the x_F 0.95 protons of
p+p at 158 GeV/c fall as exp(-6.3 p_T^2) below 0.7 GeV/c and as exp(-2.5 p_T^2) from 0.9 to 1.5 GeV/c, and the model's
single exponential left them at 0.60, 0.25 and 0.09 of the data at 1.1, 1.3 and 1.5 GeV/c. The topology option
intact_t_break (t_b, rho) draws the intact proton's transfer from exp(B t) above -t_b and the continuous slope rho B
below it (draw_t_window_break, one uniform as draw_t_window), and the importance reweighting takes the ratio of the
normalized two-slope densities for the intact-proton events of the single-diffraction classes."""

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
from na49_mixing.topology_mass_laws import (PROTON_MASS, MassLawParams, _log_two_slope_norm, _vertex_transfer_term,
                                            _vertex_transfer_term_break, draw_t_window, draw_t_window_break,
                                            log_density_ratio, pp_t_window_vec)
from na49_mixing.topology_runner import TopologyNominalParams, run_topology

NESTED_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
CONTINUUM_PATH = NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json"
ENGINE = NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py"
RUNS = NESTED_ROOT / "config" / "runs"
BREAK = (1.0, 0.4)


def _two_slope_density(t, b, t_b, rho):
    return np.where(t >= -t_b, np.exp(b * t), np.exp(-b * t_b + rho * b * (t + t_b)))


def test_rho_one_and_a_window_above_the_break_are_draw_t_window():
    rng = np.random.default_rng(3)
    for _ in range(300):
        b = rng.uniform(2.0, 12.0)
        hi = -rng.uniform(0.0, 0.3)
        lo = hi - rng.uniform(0.05, 30.0)
        seed = int(rng.integers(1 << 30))
        ref = draw_t_window(lo, hi, MassLawParams(0.0, b), np.random.default_rng(seed))
        got = draw_t_window_break(lo, hi, MassLawParams(0.0, b), (0.8, 1.0), np.random.default_rng(seed))
        assert got == pytest.approx(ref, abs=1e-12)
        # a window that ends above the break draws the committed law bit for bit
        lo2 = max(lo, -0.7)
        assert (draw_t_window_break(lo2, hi, MassLawParams(0.0, b), (0.8, 0.4), np.random.default_rng(seed))
                == draw_t_window(lo2, hi, MassLawParams(0.0, b), np.random.default_rng(seed)))


def test_the_draw_follows_the_two_slope_density():
    b, lo, hi = 6.3, -6.0, -0.01
    rng = np.random.default_rng(11)
    ts = np.array([draw_t_window_break(lo, hi, MassLawParams(0.0, b), BREAK, rng) for _ in range(200_000)])
    assert ts.min() >= lo and ts.max() <= hi
    edges = np.linspace(lo, hi, 60)
    counts, _ = np.histogram(ts, edges)
    fine = np.linspace(lo, hi, 60_001)
    dens = _two_slope_density(fine, b, *BREAK)
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(fine))))
    expected = np.diff(np.interp(edges, fine, cdf)) / cdf[-1] * ts.size
    ok = expected > 400
    assert np.all(np.abs(counts[ok] - expected[ok]) < 5.0 * np.sqrt(expected[ok]))
    # the share below the break: the analytic one
    below = np.interp(-BREAK[0], fine, cdf) / cdf[-1]
    assert np.mean(ts < -BREAK[0]) == pytest.approx(below, abs=4.0 * math.sqrt(below / ts.size))


def test_the_normalization_is_the_integral():
    for b, lo, hi in ((6.0, -5.0, -0.02), (8.5, -0.9, -0.1), (5.5, -3.0, -1.2), (2.0, -40.0, -0.3)):
        fine = np.linspace(lo, hi, 400_001)
        dens = _two_slope_density(fine, b, *BREAK)
        numeric = math.log(float(np.sum(0.5 * (dens[1:] + dens[:-1]) * np.diff(fine))))
        assert float(_log_two_slope_norm(np.array([b]), np.array([lo]), np.array([hi]), *BREAK)[0]) == pytest.approx(
            numeric, abs=1e-6)


def test_the_reweighting_carries_the_draw_to_another_slope():
    """Events drawn at the slope b0 f, reweighted by the transfer term, reproduce the law drawn at b f."""
    rng = np.random.default_rng(5)
    n = 120_000
    m2 = np.exp(rng.uniform(math.log(1.2 ** 2), math.log(30.0), n))
    lo, hi = pp_t_window_vec(m2, np.full(n, PROTON_MASS ** 2))
    f = rng.uniform(1.0, 1.6, n)
    b0, b1 = 6.0, 5.2
    t0 = np.array([draw_t_window_break(a, c, MassLawParams(0.0, b0 * g), BREAK, rng) for a, c, g in zip(lo, hi, f)])
    t1 = np.array([draw_t_window_break(a, c, MassLawParams(0.0, b1 * g), BREAK, rng) for a, c, g in zip(lo, hi, f)])
    w = np.exp(_vertex_transfer_term_break(f, t0, lo, hi, b1, b0, BREAK))
    assert np.mean(w) == pytest.approx(1.0, abs=0.01)
    for cut in (-0.3, -1.0, -1.6):
        direct = np.mean(t1 < cut)
        reweighted = np.sum(w * (t0 < cut)) / np.sum(w)
        assert reweighted == pytest.approx(direct, abs=5.0 * math.sqrt(direct * (1 - direct) / n) + 2e-4)


def test_the_ratio_moves_only_the_intact_proton_and_rho_one_is_the_committed_term():
    rng = np.random.default_rng(9)
    n = 400
    m2 = np.exp(rng.uniform(math.log(1.3 ** 2), math.log(25.0), n))
    m_rem = np.where(np.arange(n) % 3 == 0, 1.232, PROTON_MASS)
    lo, hi = pp_t_window_vec(m2, m_rem * m_rem)
    t = lo + (hi - lo) * rng.uniform(0.0, 1.0, n) ** 6
    arrays = {"m2": m2, "t": t, "m_rem": m_rem, "slope_factor": rng.uniform(1.0, 1.4, n)}
    kw = dict(eps0=0.1, b_sd0=6.0, sqrt_s=17.2702, exact_range=True)
    plain = log_density_ratio("sd_projectile", arrays, 0.12, 5.6, **kw)
    broken = log_density_ratio("sd_projectile", arrays, 0.12, 5.6, t_break=BREAK, **kw)
    unit = log_density_ratio("sd_projectile", arrays, 0.12, 5.6, t_break=(1.0, 1.0), **kw)
    intact = m_rem <= PROTON_MASS * (1.0 + 1e-9)
    np.testing.assert_array_equal(broken[~intact], plain[~intact])
    assert not np.allclose(broken[intact], plain[intact])
    np.testing.assert_allclose(unit, plain, rtol=0.0, atol=1e-9)
    # the intact vertex factor moves with it
    both = log_density_ratio("sd_projectile", arrays, 0.12, 5.6, t_break=BREAK, intact_factor=(1.8, 1.6), **kw)
    g = arrays["slope_factor"] + 0.1 * intact
    expected = _vertex_transfer_term_break(arrays["slope_factor"][intact], t[intact], lo[intact], hi[intact], 5.6, 6.0,
                                           BREAK, g[intact])
    base = log_density_ratio("sd_projectile", arrays, 0.12, 5.6, intact_factor=(1.8, 1.6), **kw)
    committed = _vertex_transfer_term(arrays["slope_factor"], t, lo, hi, 5.6, 6.0, g)
    np.testing.assert_allclose(both[intact] - base[intact], expected - committed[intact], atol=1e-9)
    with pytest.raises(ValueError, match="transfer break"):
        log_density_ratio("dd", {"m12": m2, "m22": m2, "t1": t, "t2": t, "slope_factor": arrays["slope_factor"]},
                          0.12, 5.6, t_break=BREAK, **kw)


@pytest.fixture(scope="module")
def catalog():
    return HadronCatalog.from_json(CATALOG_PATH)


@pytest.fixture(scope="module")
def fams():
    return families_from_json(str(CONTINUUM_PATH))


def _rngs(i):
    labels = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")
    subs = derive_substreams(20261024, 0, i)
    return {f"rng_{label}": np.random.default_rng(subs.stream_for(label)) for label in labels}


def test_the_runner_validates_the_break(catalog, fams):
    for bad in ((0.0, 0.4), (1.0, 0.0), (1.0, 1.5), (True, 0.4), (1.0,), (float("nan"), 0.4)):
        params = TopologyNominalParams(exact_transfer_range=True, transfer_slope_mass_exponent=1.25, intact_t_break=bad)
        with pytest.raises(ValueError, match="intact_t_break"):
            run_topology("sd_projectile", params, catalog, fams, **_rngs(1))
    with pytest.raises(ValueError, match="intact_t_break"):
        run_topology("sd_projectile", TopologyNominalParams(intact_t_break=BREAK), catalog, fams, **_rngs(1))


def test_the_intact_proton_reaches_large_transfers(catalog, fams):
    base = TopologyNominalParams(exact_transfer_range=True, transfer_slope_mass_exponent=1.25,
                                 remnant_weights={"intact_proton": 1.0, "low_mass_baryonic_resonance": 0.0,
                                                  "high_mass_baryonic_continuum": 0.0})
    import dataclasses
    broken = dataclasses.replace(base, intact_t_break=(0.5, 0.4))
    n = 400
    plain = [run_topology("sd_target", base, catalog, fams, **_rngs(i)) for i in range(n)]
    tail = [run_topology("sd_target", broken, catalog, fams, **_rngs(i)) for i in range(n)]
    far = lambda evs: sum(1 for e in evs if e.t < -0.5)
    assert far(tail) > far(plain) + 10
    # the remnant draw and the excited mass keep their uniforms: the same events but for the transfer
    assert [e.excitation_mass for e in plain[:50]] == [e.excitation_mass for e in tail[:50]]


def _engine(tmp_path, edit, name):
    spec_dict = json.loads((RUNS / "r113.json").read_text())
    edit(spec_dict)
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(spec_dict))
    spec = importlib.util.spec_from_file_location(name, ENGINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.configure(path)
    return module


def test_the_engine_threads_the_break(tmp_path):
    e = _engine(tmp_path, lambda s: s["topology"].__setitem__("intact_t_break", [1.0, 0.4]), "engine_tbreak")
    assert e.T_BREAK == (1.0, 0.4)
    assert e._params_at("sd_projectile", e.GEN0_SHAPE).intact_t_break == (1.0, 0.4)
    assert e._params_at("dd", e.GEN0_SHAPE).intact_t_break is None
    assert e._params_at("nd0", e.GEN0_SHAPE).intact_t_break is None
    for bad in ([0.0, 0.4], [1.0, 1.2], [1.0], "1.0"):
        with pytest.raises(ValueError, match="intact_t_break"):
            _engine(tmp_path, lambda s, v=bad: s["topology"].__setitem__("intact_t_break", v), "engine_tbreak_bad")


def test_the_r115_spec_is_r114_with_the_break():
    r114, r115 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r114", "r115"))
    assert r115["topology"] == {**r114["topology"], "intact_t_break": [1.0, 0.4]}
    for block in ("kernel", "fit", "diffractive_split", "isovector", "fixed", "statistics", "central_component",
                  "single_diffraction_window", "systematic_variants"):
        assert r115[block] == r114[block], block
    assert [p["name"] for p in r115["parameters"]] == [p["name"] for p in r114["parameters"]]
    assert r115["generation_0"]["run"] == "r114_sd_tail_replica_2026-10-24"
    assert r115["generation_0"]["from_fit"] == r114["generation_0"]["from_fit"]
    assert r115["seeds"]["raw_first"] > r114["seeds"]["raw_first"] + 511
    branch = NESTED_ROOT.parent / "00runs" / "r114_sd_tail_replica_2026-10-24" / "branch_report_f0.json"
    if branch.is_file():
        report = json.loads(branch.read_text())
        assert r115["generation_0"]["branch_digest"] == report["digest"]
        gen0 = {p["name"]: p["gen0"] for p in r115["parameters"]}
        for name in r115["generation_0"]["from_fit"]:
            assert float(gen0[name]) == float(report["fitted"][name]), name
