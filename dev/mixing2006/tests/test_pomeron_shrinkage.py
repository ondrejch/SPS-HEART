"""The Pomeron's shrinkage in the diffractive classes (2026-11-01, the review of 2026-11-01 item R5).

The triple-Pomeron term at fixed transfer is d^2 sigma / dxi dt ~ G(t) xi^(-1 - eps - 2 alpha' t); the committed law is its
t = 0 form, (M^2)^(-1-eps), with the vertex law normalized per mass.  The topology option diffractive_pomeron_slope alpha'
multiplies the committed law by xi^(-2 alpha' t) (the coherence-region vacuum events of single and double diffraction):
the transfer slope becomes B + 2 alpha' L, L the rapidity gap, and the mass marginal gains Z_{B+c} / Z_B.  The draws
(draw_marginal_pomeron, draw_t_pomeron, draw_joint_dd_exact) and the importance reweighting (log_density_ratio's pomeron)
must describe the same law, and alpha' = 0 must give the committed ratio."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing import topology_mass_laws as T

NESTED_ROOT = Path(__file__).resolve().parents[1]
RUNS = NESTED_ROOT / "config" / "runs"
ENGINE = NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py"
MP = T.PROTON_MASS
SQ = 17.270
LO = T.EXCITATION_M_MIN ** 2
HI = min((SQ - MP) ** 2, 0.1 * SQ * SQ)
KAPPA, ALPHA, BREAK = 1.25, 0.25, (1.0, 0.4)


def _sd_sample(n, eps, b, f, seed, alpha=ALPHA, t_lin=None):
    rng = np.random.default_rng(seed)
    ft = 0.5 * (f - 1.0)
    rows = []
    for _ in range(n):
        m2 = T.draw_marginal_pomeron(LO, HI, eps, b, MP, KAPPA, ft, alpha, BREAK, rng, t_lin=t_lin)
        sf = T.vertex_slope_factor(math.sqrt(m2), MP, KAPPA) + ft
        t_lo, t_hi = T.pp_t_window(m2, MP * MP)
        c = 2.0 * alpha * float(T.pomeron_gap_log(m2, MP * MP))
        rows.append((m2, T.draw_t_pomeron(t_lo, t_hi, b * sf, c, BREAK, rng, t_lin=t_lin), sf))
    a = np.asarray(rows)
    return {"m2": a[:, 0], "t": a[:, 1], "m_rem": np.full(n, MP), "slope_factor": a[:, 2], "pomeron": np.ones(n),
            "m2_low": np.full(n, LO)}


def _engine():
    spec = importlib.util.spec_from_file_location("run_engine_pomeron_test", ENGINE)
    e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e)
    return e


def test_the_gap_and_the_zero_slope_limit():
    # an intact proton: L = ln(s / M^2); two heavy systems leave no gap
    s = T.PINNED_SQRT_S_EXACT ** 2
    assert float(T.pomeron_gap_log(4.0, MP * MP)) == pytest.approx(math.log(s / 4.0), rel=1e-12)
    assert float(T.pomeron_gap_log(29.0, 29.0)) == 0.0
    # alpha' = 0: the marginal's factor is one and the transfer draw is the committed one
    m2 = np.geomspace(LO, HI, 50)
    assert np.max(np.abs(T.sd_pomeron_log_factor(m2, MP, 5.67, KAPPA, 0.4, 0.0, BREAK))) < 1e-12
    for seed in range(50):
        ref = T.draw_t_window_break(-6.0, -0.01, T.MassLawParams(0.0, 6.3), BREAK, np.random.default_rng(seed))
        got = T.draw_t_pomeron(-6.0, -0.01, 6.3, 0.0, BREAK, np.random.default_rng(seed))
        assert got == pytest.approx(ref, abs=1e-12)


def test_alpha_zero_is_the_committed_ratio():
    arrays = _sd_sample(2000, 0.0808, 5.67, 1.75, 1, alpha=0.0)
    kw = dict(eps0=0.0808, b_sd0=5.67, sqrt_s=SQ, exact_range=True, intact_factor=(1.5, 1.75), xi_max=0.1, t_break=BREAK)
    committed = T.log_density_ratio("sd_projectile", {k: v for k, v in arrays.items() if k != "pomeron"}, 0.15, 4.5, **kw)
    shrunk = T.log_density_ratio("sd_projectile", arrays, 0.15, 4.5, pomeron=(0.0, KAPPA, None), **kw)
    assert np.max(np.abs(shrunk - committed)) < 1e-10
    # the flag at 0 keeps the committed ratio at any alpha'
    off = dict(arrays, pomeron=np.zeros(len(arrays["m2"])))
    assert np.array_equal(T.log_density_ratio("sd_projectile", off, 0.15, 4.5, pomeron=(ALPHA, KAPPA, None), **kw), committed)


def test_the_single_diffraction_draws_follow_the_reweighting():
    gen = _sd_sample(120_000, 0.0808, 5.67, 1.75, 2)
    trial = _sd_sample(120_000, 0.15, 4.5, 1.5, 3)
    w = np.exp(T.log_density_ratio("sd_projectile", gen, 0.15, 4.5, eps0=0.0808, b_sd0=5.67, sqrt_s=SQ, exact_range=True,
                                   intact_factor=(1.5, 1.75), xi_max=0.1, t_break=BREAK, pomeron=(ALPHA, KAPPA, None)))
    assert w.mean() == pytest.approx(1.0, abs=0.004)
    for key, edges in (("m2", np.geomspace(LO, HI, 7)), ("t", np.array([-4.0, -0.5, -0.3, -0.15, -0.06, 0.0]))):
        h_w, _ = np.histogram(gen[key], edges, weights=w)
        h_d, _ = np.histogram(trial[key], edges)
        assert np.all(np.abs(h_w / h_d - 1.0) < 0.03), (key, h_w / h_d)
    # the shrinkage steepens the transfer: the mean |t| falls against the committed law at the same point
    committed = _sd_sample(40_000, 0.0808, 5.67, 1.75, 4, alpha=0.0)
    assert np.mean(gen["t"]) > np.mean(committed["t"]) + 0.03


def test_the_linear_range():
    # t_lin at the intact proton's break: above -t_lin the slope B + c, below it the committed rho B; the draws and the
    # reweighting describe the same law
    gen = _sd_sample(120_000, 0.0808, 5.67, 1.75, 12, t_lin=1.0)
    trial = _sd_sample(120_000, 0.12, 5.0, 1.5, 13, t_lin=1.0)
    w = np.exp(T.log_density_ratio("sd_projectile", gen, 0.12, 5.0, eps0=0.0808, b_sd0=5.67, sqrt_s=SQ, exact_range=True,
                                   intact_factor=(1.5, 1.75), xi_max=0.1, t_break=BREAK,
                                   pomeron=(ALPHA, KAPPA, None, 1.0)))
    assert w.mean() == pytest.approx(1.0, abs=0.004)
    for key, edges in (("m2", np.geomspace(LO, HI, 7)), ("t", np.array([-4.0, -0.5, -0.25, -0.1, 0.0]))):
        h_w, _ = np.histogram(gen[key], edges, weights=w)
        h_d, _ = np.histogram(trial[key], edges)
        assert np.all(np.abs(h_w / h_d - 1.0) < 0.04), (key, h_w / h_d)
    # below the break the committed slope rho B (no raise), above it B + c; the single slope gains c above -t_lin only
    assert T._pomeron_slopes(6.0, 1.5, BREAK, 1.0) == (7.5, 0.4 * 6.0, 1.0)
    assert T._pomeron_slopes(6.0, 1.5, BREAK, None) == (7.5, 0.4 * 6.0 + 1.5, 1.0)
    assert T._pomeron_slopes(3.0, 1.0, None, 1.0) == (4.0, 3.0, 1.0)
    assert T._pomeron_slopes(3.0, 1.0, None, None) == (4.0, 4.0, None)
    # the linear range must meet the intact proton's break
    with pytest.raises(ValueError, match="linear range"):
        T.draw_t_pomeron(-5.0, -0.01, 6.0, 1.0, BREAK, np.random.default_rng(1), t_lin=0.8)


def test_the_window_fraction_integral():
    gen = _sd_sample(80_000, 0.0808, 5.67, 1.75, 5)
    args = (0.0808, 5.67, KAPPA, 0.5 * (1.75 - 1.0), ALPHA, BREAK)
    cut = 0.05 * SQ * SQ
    frac = math.exp(T.log_sd_pomeron_mass_norm(*args, LO, cut, MP) - T.log_sd_pomeron_mass_norm(*args, LO, HI, MP))
    assert frac == pytest.approx(float(np.mean(gen["m2"] <= cut)), abs=0.004)
    # the shrinkage moves the class toward high masses: the window holds less of it than the pure power
    pure = math.exp(T.log_power_integral(0.0808, LO, cut) - T.log_power_integral(0.0808, LO, HI))
    assert frac < pure - 0.005


def test_the_double_diffraction_draw_follows_the_reweighting():
    def sample(n, eps, b, seed):
        rng = np.random.default_rng(seed)
        rows = []
        for _ in range(n):
            a, d, t, t2 = T.draw_joint_dd_exact(LO, HI, LO, HI, SQ, T.MassLawParams(eps, b), rng, slope_mass_exponent=KAPPA,
                                               pomeron_slope=ALPHA)
            assert t == t2
            rows.append((a, d, t, T.vertex_slope_factor(math.sqrt(a), math.sqrt(d), KAPPA)))
        a = np.asarray(rows)
        n = len(rows)
        return {"m12": a[:, 0], "m22": a[:, 1], "t1": a[:, 2], "t2": a[:, 2], "slope_factor": a[:, 3],
                "pomeron": np.ones(n), "m_lo1": np.full(n, T.EXCITATION_M_MIN), "m_lo2": np.full(n, T.EXCITATION_M_MIN)}
    gen, trial = sample(6000, 0.0808, 5.67, 6), sample(6000, 0.15, 4.5, 7)
    w = np.exp(T.log_density_ratio("dd", gen, 0.15, 4.5, eps0=0.0808, b_sd0=5.67, sqrt_s=SQ, exact_range=True, xi_max=0.1,
                                   pomeron=(ALPHA, KAPPA, None)))
    assert w.mean() == pytest.approx(1.0, abs=0.015)
    edges = np.array([-4.0, -0.4, -0.15, 0.0])
    h_w, _ = np.histogram(gen["t1"], edges, weights=w)
    h_d, _ = np.histogram(trial["t1"], edges)
    assert np.all(np.abs(h_w / h_d - 1.0) < 0.08), h_w / h_d


def test_the_refusals():
    arrays = _sd_sample(10, 0.0808, 5.67, 1.75, 8)
    with pytest.raises(ValueError, match="exact transfer window"):
        T.log_density_ratio("sd_projectile", arrays, 0.1, 5.0, eps0=0.0808, b_sd0=5.67, sqrt_s=SQ, exact_range=False,
                            pomeron=(ALPHA, KAPPA, 1.75))
    dd = {"m12": np.array([4.0]), "m22": np.array([4.0]), "t1": np.array([-0.1]), "t2": np.array([-0.1]),
          "slope_factor": np.array([0.5]), "pomeron": np.ones(1)}
    with pytest.raises(ValueError, match="coherence edge"):
        T.log_density_ratio("dd", dd, 0.1, 5.0, eps0=0.0808, b_sd0=5.67, sqrt_s=SQ, exact_range=True,
                            pomeron=(ALPHA, KAPPA, None))


def test_a_shrinkage_request_without_the_flag_is_refused():
    # the second review round of 2026-11-01, C1: the missing flag used to fall back to the committed ratio, 4% off in
    # the weights at a trial point while exact at the generation point
    arrays = _sd_sample(10, 0.0808, 5.67, 1.75, 9)
    kw = dict(eps0=0.0808, b_sd0=5.67, sqrt_s=SQ, exact_range=True, intact_factor=(1.5, 1.75), xi_max=0.1, t_break=BREAK)
    bare = {k: v for k, v in arrays.items() if k != "pomeron"}
    with pytest.raises(ValueError, match="no 'pomeron' flag"):
        T.log_density_ratio("sd_projectile", bare, 0.15, 4.5, pomeron=(ALPHA, KAPPA, None), **kw)
    # without the request the same events take the committed ratio, as before
    assert np.all(np.isfinite(T.log_density_ratio("sd_projectile", bare, 0.15, 4.5, **kw)))


def test_the_run_spec_option(tmp_path):
    e = _engine()
    spec = json.loads((RUNS / "r132.json").read_text())
    spec["topology"]["diffractive_pomeron_slope"] = 0.25
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    e.configure(path)
    assert e.TOPOLOGY_OPTIONS["diffractive_pomeron_slope"] == 0.25
    params = e._params_at("sd_projectile", e.GEN0_SHAPE)
    assert params.diffractive_pomeron_slope == 0.25
    # the window anchor carries the shrinkage factor at the vertex slope and the intact factor
    f = float(e.GEN0_SHAPE[e.shape_pos(e.IVF_NAME)])
    b = float(e.GEN0_SHAPE[1])
    lo, hi = T.EXCITATION_M_MIN ** 2, 0.1 * e.SQRT_S ** 2
    args = (0.0808, b, 1.25, 0.5 * (f - 1.0), 0.25, (1.0, 0.4))
    expect = math.exp(T.log_sd_pomeron_mass_norm(*args, lo, 0.05 * e.SQRT_S ** 2, MP)
                      - T.log_sd_pomeron_mass_norm(*args, lo, hi, MP))
    assert e.sd_window_fraction(0.0808, 0.05, b, f) == pytest.approx(expect, rel=1e-12)
    with pytest.raises(ValueError, match="b_sd"):
        e.sd_window_fraction(0.0808, 0.05)
    # the linear range: beside the slope, at the intact proton's break
    spec["topology"]["diffractive_pomeron_linear_t"] = 0.8
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="intact proton's break"):
        e.configure(path)
    spec["topology"]["diffractive_pomeron_linear_t"] = 1.0
    path.write_text(json.dumps(spec))
    e.configure(path)
    assert e._params_at("dd", e.GEN0_SHAPE).diffractive_pomeron_linear_t == 1.0
    del spec["topology"]["diffractive_pomeron_linear_t"]
    # the option needs the exact window and the coherence edge
    del spec["topology"]["diffractive_xi_max"]
    spec.pop("single_diffraction_window")
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="diffractive_pomeron_slope"):
        e.configure(path)


def test_the_events_carry_the_law(tmp_path):
    # the R132 spec with the option: the coherence-region vacuum events are flagged and drawn from the law, the tail,
    # one-pion-exchange and isovector events keep their committed laws, and every event conserves four-momentum
    e = _engine()
    spec = json.loads((RUNS / "r132.json").read_text())
    spec["topology"]["diffractive_pomeron_slope"] = 0.25
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    e.configure(path)
    for cls in ("sd_projectile", "dd"):
        out = e.generate_slot(cls, 20261101, 40, 3990000000, e.GEN0_SHAPE, 2.0)
        a = out["arrays"]
        flag = a["shape_pomeron"]
        assert out["conservation"]["violations"] == 0
        assert set(np.unique(flag)) <= {0.0, 1.0} and flag.sum() > 0
        if cls == "sd_projectile":
            assert np.all(flag[(a["shape_tail"] > 0.5) | (a["shape_ope"] > 0.5)] == 0.0)
            vac = flag > 0.5
            # the vertex law under the shrinkage: the slope B + 2 alpha' ln(s / M^2) on each flagged event's window
            assert np.all(a["shape_t"][vac] <= 0.0)
        # the reweighting at the generation point is the identity
        arrays = {k[len("shape_"):]: v for k, v in a.items() if k.startswith("shape_")}
        f = float(e.GEN0_SHAPE[e.shape_pos(e.IVF_NAME)])
        b = float(e.GEN0_SHAPE[1])
        kw = dict(eps0=0.0808, b_sd0=b, sqrt_s=e.SQRT_S, exact_range=True, xi_max=0.1, pomeron=(0.25, 1.25, None))
        if cls == "sd_projectile":
            kw.update(intact_factor=(f, f), t_break=(1.0, 0.4))
        assert np.max(np.abs(T.log_density_ratio(cls, arrays, 0.0808, b, **kw))) < 1e-9
