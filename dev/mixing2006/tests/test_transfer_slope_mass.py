"""The factorized vertex slopes of the 2 -> 2 transfer (2026-10-08):
``TopologyNominalParams.transfer_slope_mass_exponent`` kappa draws each
exact-window transfer from exp(B t) at
B = b [(m_p/M_1)^kappa + (m_p/M_2)^kappa] / 2, records B / b on the event
(``slope_factor``), and the transfer reweighting of
``log_density_ratio`` follows the recorded factor; kappa = 0 is the single
slope b draw for draw."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.kernel_config import R74, load_kernel
from na49_mixing.topology_mass_laws import (
    PROTON_MASS,
    MassLawParams,
    draw_t_window,
    log_density_ratio,
    pp_t_window,
    vertex_slope_factor,
)
from na49_mixing.topology_runner import TopologyNominalParams, run_topology

NESTED_ROOT = Path(__file__).resolve().parents[1]
BASE = dataclasses.replace(R74, isospin_completion=("baryonic", "strange_baryonic", "strange_mesonic"), delta_lines=True,
                           three_body_phase_space=True)
LABELS = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")
#: the non-diffractive kind weights, and the diffractive classes' non-excited line (intact or low-mass)
WEIGHTS = {"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.25, "high_mass_baryonic_continuum": 0.75}
WEIGHTS_DIFFRACTIVE = {"intact_proton": 0.5, "low_mass_baryonic_resonance": 0.5, "high_mass_baryonic_continuum": 0.0}


@pytest.fixture(scope="module")
def kernel():
    return load_kernel(BASE, NESTED_ROOT)


def _params(kappa):
    return TopologyNominalParams(epsilon=0.4, b_sd=3.0, epsilon_nd=0.16, remnant_weights=WEIGHTS, nd_central=False,
                                 exact_transfer_range=True, transfer_slope_mass_exponent=kappa)


def _event(cls, params, kernel, seed):
    rngs = {f"rng_{lab}": np.random.default_rng([seed, k]) for k, lab in enumerate(LABELS)}
    if cls != "nd":
        params = dataclasses.replace(params, remnant_weights=WEIGHTS_DIFFRACTIVE)
    return run_topology(cls, params, *kernel, **rngs)


def _masses(ev):
    if ev.class_label == "dd":
        return ev.m1, ev.m2, ev.t1
    if ev.class_label == "nd":
        return ev.remnants[0].mass, ev.remnants[1].mass, ev.t1
    return ev.excitation_mass, ev.remnant.mass, ev.t


def test_the_vertex_factor():
    assert vertex_slope_factor(PROTON_MASS, PROTON_MASS, 0.75) == 1.0
    assert vertex_slope_factor(3.0, 5.0, 0.0) == 1.0
    assert vertex_slope_factor(0.5, PROTON_MASS, 2.0) == 1.0  # a line below m_p counts as m_p
    assert vertex_slope_factor(4.0, PROTON_MASS, 1.0) == pytest.approx(0.5 * (PROTON_MASS / 4.0 + 1.0))
    assert vertex_slope_factor(1e6, 1e6, 1.0) == pytest.approx(0.0, abs=1e-5)
    for bad in (-0.1, math.inf, math.nan):
        with pytest.raises(ValueError):
            vertex_slope_factor(2.0, 2.0, bad)


@pytest.mark.parametrize("cls", ["sd_projectile", "dd", "nd"])
def test_kappa_zero_is_the_single_slope(kernel, cls):
    """kappa = 0 draws every event as the single slope does (factor 1.0)."""
    for seed in range(6):
        a = _event(cls, _params(None), kernel, seed)
        b = _event(cls, _params(0.0), kernel, seed)
        assert a.slope_factor is None and b.slope_factor == 1.0
        assert _masses(a) == _masses(b)
        assert [h.four_momentum for h in a.final_state] == [h.four_momentum for h in b.final_state]


def test_the_option_needs_the_exact_window(kernel):
    params = dataclasses.replace(_params(0.75), exact_transfer_range=False)
    with pytest.raises(ValueError, match="exact_transfer_range"):
        _event("nd", params, kernel, 0)
    with pytest.raises(ValueError):
        _event("nd", _params(-1.0), kernel, 0)


@pytest.mark.parametrize("cls", ["sd_projectile", "dd", "nd"])
def test_the_transfer_follows_the_vertex_slope(kernel, cls):
    """Under kappa = 1 the recorded factor is that of the event's masses and
    the transfer's CDF at the slope b * factor on its window is uniform."""
    n, us = 300, []
    for seed in range(n):
        ev = _event(cls, _params(1.0), kernel, 1000 + seed)
        m1, m2, t = _masses(ev)
        f = vertex_slope_factor(m1, m2, 1.0)
        assert ev.slope_factor == f
        lo, hi = pp_t_window(m1 * m1, m2 * m2)
        b = 3.0 * f
        us.append((math.exp(b * (t - hi)) - math.exp(b * (lo - hi))) / (1.0 - math.exp(b * (lo - hi))))
    us = np.asarray(us)
    assert abs(us.mean() - 0.5) < 4.0 * math.sqrt(1.0 / 12.0 / n)
    assert abs(np.mean(us * us) - 1.0 / 3.0) < 4.0 * math.sqrt(4.0 / 45.0 / n)


def test_the_reweighting_is_exact():
    """Synthetic nd shapes at b0 = 3 with kappa = 0.75: the weights to
    b1 = 4.5 average to 1 and carry the transfer moments of a direct draw
    at b1; the recorded factor 1 reproduces the single-slope term."""
    rng = np.random.default_rng(5)
    n, b0, b1, kappa = 20000, 3.0, 4.5, 0.75
    m1 = rng.uniform(1.0, 6.0, n)
    m2 = rng.uniform(1.0, 6.0, n)
    f = np.asarray([vertex_slope_factor(a, b, kappa) for a, b in zip(m1, m2)])
    t0, t1 = np.empty(n), np.empty(n)
    for i in range(n):
        lo, hi = pp_t_window(m1[i] ** 2, m2[i] ** 2)
        t0[i] = draw_t_window(lo, hi, MassLawParams(0.0, b0 * f[i]), rng)
        t1[i] = draw_t_window(lo, hi, MassLawParams(0.0, b1 * f[i]), rng)
    arrays = {"t1": t0, "t2": t0, "t_c": np.zeros(n), "mc2": np.zeros(n), "m1": m1, "m2": m2,
              "k1": np.zeros(n), "k2": np.zeros(n), "lo1": np.full(n, np.nan), "hi1": np.full(n, np.nan),
              "lo2": np.full(n, np.nan), "hi2": np.full(n, np.nan), "slope_factor": f}
    w = np.exp(log_density_ratio("nd", arrays, 0.16, b1, eps0=0.16, b_sd0=b0, sqrt_s=17.27, exact_range=True))
    assert abs(w.mean() - 1.0) < 4.0 * w.std() / math.sqrt(n)
    est, direct = np.mean(w * t0), t1.mean()
    err = math.hypot(np.std(w * t0), np.std(t1)) / math.sqrt(n)
    assert abs(est - direct) < 4.0 * err
    with pytest.raises(ValueError, match="exact transfer window"):
        log_density_ratio("nd", arrays, 0.16, b1, eps0=0.16, b_sd0=b0, sqrt_s=17.27, exact_range=False)
    single = dict(arrays)
    del single["slope_factor"]
    ones = dict(arrays, slope_factor=np.ones(n))
    np.testing.assert_allclose(log_density_ratio("nd", ones, 0.16, b1, eps0=0.16, b_sd0=b0, sqrt_s=17.27, exact_range=True),
                               log_density_ratio("nd", single, 0.16, b1, eps0=0.16, b_sd0=b0, sqrt_s=17.27, exact_range=True),
                               rtol=1e-12, atol=1e-12)


def test_the_run_spec_option(tmp_path):
    """The engine's topology block takes the exponent as a number with the
    exact window; the central class (nd) keeps the single slope."""
    path = NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py"
    spec = importlib.util.spec_from_file_location("run_engine_slope_law", path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    base = json.loads((NESTED_ROOT / "config" / "runs" / "r82.json").read_text())
    assert base["topology"].get("exact_transfer_range") is True
    good = dict(base, topology={**base["topology"], "transfer_slope_mass_exponent": 0.75})
    (tmp_path / "good.json").write_text(json.dumps(good))
    engine.configure(tmp_path / "good.json")
    assert engine.TOPOLOGY_OPTIONS["transfer_slope_mass_exponent"] == 0.75
    shape = (0.4, 3.0, 0.16)
    flavor = tuple(1.0 for _ in engine.FLAVOR_ORDER)
    assert engine._params_at("nd0", shape, flavor).transfer_slope_mass_exponent == 0.75
    assert engine._params_at("nd", shape, flavor).transfer_slope_mass_exponent is None
    for topology in ({**base["topology"], "transfer_slope_mass_exponent": True},
                     {**base["topology"], "transfer_slope_mass_exponent": -0.5},
                     {**base["topology"], "exact_transfer_range": False, "transfer_slope_mass_exponent": 0.75}):
        (tmp_path / "bad.json").write_text(json.dumps(dict(base, topology=topology)))
        with pytest.raises(ValueError):
            engine.configure(tmp_path / "bad.json")
