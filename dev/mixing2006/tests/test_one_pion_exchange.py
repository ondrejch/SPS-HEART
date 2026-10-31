"""The one-pion-exchange component of the single-diffraction classes (2026-10-26, at the R120 optimum): the forward
soft pi+ and the forward neutrons ask for hard leading neutrons at normal multiplicity, the pion-exchange class the
two-line cascade does not make.

- the law: the flux density, the three-uniform joint draw, the transfer on the reachable window, the leading
  neutron's spectrum (its peak near x_L 0.7-0.8); the proton form factor and the Regge pi+ p cross section of the
  fixed pion term of Appleby et al. (2016), the component's form;
- generate_sd: a share of the vacuum events beside the tail turns into p + p -> n + X++ (the neutron remnant, the
  excited line at Q = 2), every other event the same bit for bit, conservation exact, malformed components refused;
- the runner reads the topology's sd_ope;
- the neutral partner (2026-10-27, the law's pion "isospin"): p + p -> p + X+ at half the squared coupling, with the
  neutral pion's mass and sigma_pi0 p, the share then the whole pion exchange.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.one_pion_exchange import (NEUTRAL_PION_MASS, PION_MASS, OPELaw, draw_ope, ope_component_of, ope_law_of,
                                           ope_log_absorption, ope_log_density, ope_marginal_cells, ope_neutral_fraction,
                                           ope_parts, pion_proton_cross_section)
from na49_mixing.topology_mass_laws import pp_t_window
from na49_mixing.topology_single_diffraction import SQRT_S

NESTED_ROOT = Path(__file__).resolve().parents[1]
ENGINE = NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py"
SPEC = NESTED_ROOT / "config" / "runs" / "r120.json"
S = SQRT_S * SQRT_S
M2_NEUTRON = 0.93956542 ** 2
M2_THRESHOLD = (0.93827231 + PION_MASS) ** 2   # p pi+
PROTON = OPELaw(0.9, 0.0, form_factor="proton")
APPLEBY = OPELaw(0.93, 0.0, form_factor="proton", pion_proton="regge")


def test_the_flux_vanishes_at_zero_transfer_and_peaks_near_the_pion_mass():
    law = OPELaw(0.9, 0.3)
    ts = -np.logspace(-5, 1, 2001)
    f = np.exp(ope_log_density(0.3, ts, law))
    assert f[0] < 1e-2 * f.max()
    # the spin flip: the flux falls linearly in |t| toward t = 0
    f6, f5 = np.exp(ope_log_density(0.3, np.array([-1e-6, -1e-5]), law))
    assert abs(f6 / f5 - 0.1) < 0.005
    t_peak = -ts[int(np.argmax(f))]
    assert 0.5 * PION_MASS ** 2 < t_peak < 3.0 * PION_MASS ** 2


def test_the_law_refuses_bad_parameters():
    for bad in ((-0.1, 0.3), (0.9, float("nan")), (0.9, 0.3, 0.0)):
        with pytest.raises(ValueError):
            OPELaw(*bad)


def test_the_draw_takes_three_uniforms_and_stays_on_the_window():
    law = OPELaw(0.9, 0.3)
    a, b = np.random.default_rng(3), np.random.default_rng(3)
    draw_ope(M2_THRESHOLD, 0.89 * S, M2_NEUTRON, S, law, a)
    for _ in range(3):
        b.random()
    assert a.random() == b.random()
    rng = np.random.default_rng(4)
    for _ in range(300):
        m2, t = draw_ope(M2_THRESHOLD, 0.89 * S, M2_NEUTRON, S, law, rng)
        lo, hi = pp_t_window(m2, M2_NEUTRON)
        assert M2_THRESHOLD <= m2 <= 0.89 * S and lo <= t <= hi < 0.0 and t >= -law.t_cut


@pytest.mark.parametrize("law", [OPELaw(0.9, 0.3), OPELaw(0.9, 1.0), PROTON, APPLEBY])
def test_the_leading_neutron_peaks_near_x_l_0_75(law):
    rng = np.random.default_rng(11)
    xl = np.array([1.0 - draw_ope(M2_THRESHOLD, 0.89 * S, M2_NEUTRON, S, law, rng)[0] / S
                   for _ in range(6000)])
    h, edges = np.histogram(xl, bins=np.linspace(0.0, 1.0, 21))
    peak = 0.5 * (edges[int(np.argmax(h))] + edges[int(np.argmax(h)) + 1])
    assert 0.65 < peak < 0.85
    assert np.mean(xl > 0.95) < 0.05 and np.mean(xl < 0.3) < 0.05


def test_the_proton_form_factor_is_appleby_eq_14_and_narrows_the_transfer():
    # F_1(t)^2 against the three exponentials of Appleby et al. (2016), Eq. 14, below 1 GeV^2
    ts = -np.linspace(0.0, 1.0, 41)[1:]
    f2 = np.exp(ope_log_density(0.3, ts, PROTON) - ope_log_density(0.3, ts, OPELaw(0.9, 0.0)))
    eq14 = 0.27 * np.exp(8.38 * ts) + 0.56 * np.exp(3.78 * ts) + 0.18 * np.exp(1.36 * ts)
    np.testing.assert_allclose(f2, eq14, rtol=0.03)
    # its slope at t = 0 (about 4.6 GeV^-2) against 2 R^2 = 0.6: the transfer about half as large
    ra, rb = np.random.default_rng(5), np.random.default_rng(5)
    t_exp = np.array([draw_ope(M2_THRESHOLD, 0.89 * S, M2_NEUTRON, S, OPELaw(0.9, 0.3), ra)[1] for _ in range(3000)])
    t_pro = np.array([draw_ope(M2_THRESHOLD, 0.89 * S, M2_NEUTRON, S, PROTON, rb)[1] for _ in range(3000)])
    assert np.mean(-t_pro) < 0.6 * np.mean(-t_exp)


def test_the_regge_cross_section_weights_the_masses():
    # Donnachie-Landshoff (1992) for pi+ p: 23.2 mb at s' = 100 GeV^2, rising toward the threshold
    assert pion_proton_cross_section(100.0) == pytest.approx(23.2, abs=0.05)
    assert pion_proton_cross_section(1.2) > 1.6 * pion_proton_cross_section(100.0)
    ys, flat = ope_marginal_cells(M2_THRESHOLD, 0.89 * S, M2_NEUTRON, S, PROTON)
    _, regge = ope_marginal_cells(M2_THRESHOLD, 0.89 * S, M2_NEUTRON, S, OPELaw(0.9, 0.0, form_factor="proton",
                                                                                     pion_proton="regge"))
    sigma = np.array([pion_proton_cross_section(math.exp(y)) for y in 0.5 * (ys[1:] + ys[:-1])])
    np.testing.assert_allclose(regge / flat, sigma, rtol=2e-3)


def test_the_component_names_its_form_factor():
    assert ope_law_of(0.9, 0.3) == OPELaw(0.9, 0.3)
    assert ope_law_of(0.93, "proton", "regge") == APPLEBY
    for bad in ((0.9, "exponential"), (0.9, "dipole"), (0.9, 0.3, "flat")):
        with pytest.raises(ValueError):
            ope_law_of(*bad)
    with pytest.raises(ValueError):
        OPELaw(0.9, 0.3, form_factor="proton")
    assert ope_component_of((0.1, 0.003, 0.89, 0.9, "proton")) == (0.1, 0.003, 0.89, PROTON)
    assert ope_component_of((0.1, 0.003, 0.89, 0.9, 0.3)) == (0.1, 0.003, 0.89, OPELaw(0.9, 0.3))
    for bad in ((0.1, 0.003, 0.89, 0.9), (0.1, 0.003, 0.89, 0.9, "proton", "regge", 1.0), (1.0, 0.003, 0.89, 0.9, 0.3),
                (0.1, 0.89, 0.003, 0.9, 0.3), (0.1, 0.003, 0.89, 0.9, "proton", "x"), (float("nan"), 0.003, 0.89, 0.9, 0.3)):
        with pytest.raises(ValueError):
            ope_component_of(bad)


def test_the_neutral_partner_takes_a_third_of_the_pion_exchange():
    """p -> n pi+ couples with twice the square of p -> p pi0; the neutral law differs in the pion's mass and in
    sigma_pi0 p (the pi+/pi- average), so its part of the whole exchange lies a little above one third."""
    iso = OPELaw(0.93, 0.0, form_factor="proton", pion_proton="regge", pion="isospin")
    charged, neutral = ope_parts(iso)
    assert charged == APPLEBY and neutral == dataclasses.replace(APPLEBY, pion="neutral")
    assert (charged.pion_mass, neutral.pion_mass, iso.pion_mass) == (PION_MASS, NEUTRAL_PION_MASS, PION_MASS)
    for m2 in (2.0, 30.0, 200.0):
        diff = pion_proton_cross_section(m2, "neutral") - pion_proton_cross_section(m2)
        assert diff == pytest.approx((31.79 - 27.56) * m2 ** -0.4525, rel=1e-12)
    ts = -np.array([0.005, 0.02, 0.1])
    assert np.all(ope_log_density(0.3, ts, neutral) > ope_log_density(0.3, ts, charged))   # the lighter propagator pole
    f0 = ope_neutral_fraction(M2_THRESHOLD, 0.89 * S, M2_NEUTRON, 0.93827231 ** 2, S, iso)
    assert 1.0 / 3.0 < f0 < 0.36
    assert f0 == pytest.approx(0.347, abs=0.002)   # the value quoted for Eq. 39 at 158 GeV/c
    for call in (lambda: ope_log_density(0.3, ts, iso), lambda: ope_marginal_cells(M2_THRESHOLD, 0.89 * S, M2_NEUTRON, S, iso)):
        with pytest.raises(ValueError):
            call()
    assert ope_component_of((0.3, 0.003, 0.89, 0.93, "proton", "regge", "isospin"))[3] == iso
    assert ope_component_of((0.3, 0.003, 0.89, 0.93, "proton", "regge", "charged"))[3] == APPLEBY
    with pytest.raises(ValueError):
        ope_component_of((0.3, 0.003, 0.89, 0.93, "proton", "regge", "both"))


@pytest.fixture(scope="module")
def physics():
    spec = importlib.util.spec_from_file_location("run_engine_ope_test", ENGINE)
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    E.configure(SPEC)
    catalog, fams = E._load_physics(E.FLOOR_GRID[0])
    report = json.loads(SPEC.read_text())
    gen0 = {p["name"]: float(p["gen0"]) for p in report["parameters"] if p.get("gen0") is not None}
    shape = (gen0["epsilon"], gen0["b_sd"], gen0["epsilon_nd"], gen0["intact_vertex_factor"], gen0["sd_tail_share"])
    flavor = tuple(gen0[n] for n in E.FLAVOR_ORDER)
    return E, catalog, fams, shape, flavor


def _events(E, catalog, fams, params, cls, n, seed=20261026):
    from na49_mixing.topology_runner import run_topology
    out = []
    for i in range(n):
        subs = E.derive_substreams(seed, 0, 5000 + i)
        rngs = {f"rng_{lab}": np.random.default_rng(subs.stream_for(lab))
                for lab in ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")}
        out.append(run_topology(cls, params, catalog, fams, **rngs))
    return out


@pytest.mark.parametrize("cls", ["sd_target", "sd_projectile"])
def test_a_share_of_the_vacuum_events_turns_into_one_pion_exchange(physics, cls):
    E, catalog, fams, shape, flavor = physics
    base = E._params_at(cls, shape, flavor)
    assert base.sd_tail is not None and base.sd_ope is None
    ope = dataclasses.replace(base, sd_ope=(0.25, 0.003, 0.89, 0.9, 0.3))
    a_events, b_events = _events(E, catalog, fams, base, cls, 120), _events(E, catalog, fams, ope, cls, 120)
    n_ope = 0
    for a, b in zip(a_events, b_events):
        assert bool(a.tail) == bool(b.tail)   # the tail events stay tail events
        if b.ope:
            n_ope += 1
            assert not a.ope and not a.tail
            remnant = [h for h in b.final_state if h.line == "remnant"]
            assert len(remnant) == 1 and remnant[0].pdg_id == 2112
            excited = [h for h in b.final_state if h.line != "remnant"]
            assert sum(int(h.qn[1]) for h in excited) == 2 and sum(int(h.qn[0]) for h in excited) == 1
            assert b.exchange_configuration is not None and not b.exchange_configuration.is_vacuum
            E.event_conservation(b)
        else:
            # every other event the same bit for bit
            assert a.excitation_mass == b.excitation_mass and a.t == b.t
            assert [h.four_momentum for h in a.final_state] == [h.four_momentum for h in b.final_state]
    assert 15 <= n_ope <= 50   # 0.25 of the about 0.74 vacuum coherence events (the tail takes 0.26)


def test_the_neutron_goes_forward_on_the_target_side(physics):
    E, catalog, fams, shape, flavor = physics
    ope = dataclasses.replace(E._params_at("sd_target", shape, flavor), sd_ope=(0.6, 0.003, 0.89, 0.9, 0.3))
    beta, gamma = E.OB.cm_boost(E.SQRT_S_CM)
    xs = []
    for ev in _events(E, catalog, fams, ope, "sd_target", 80):
        if ev.ope:
            n = [h for h in ev.final_state if h.line == "remnant"][0]
            xs.append(2.0 * E.OB.to_cm(n.four_momentum, beta, gamma)[3] / E.SQRT_S_CM)
    assert len(xs) >= 20 and min(xs) > 0.0 and 0.5 < float(np.median(xs)) < 0.85


def test_the_runner_takes_the_proton_form_factor(physics):
    E, catalog, fams, shape, flavor = physics
    base = dataclasses.replace(E._params_at("sd_target", shape, flavor), sd_ope=(0.6, 0.003, 0.89, 0.9, 0.3))
    proton = dataclasses.replace(base, sd_ope=(0.6, 0.003, 0.89, 0.93, "proton", "regge"))
    n_ope = n_moved = 0
    for a, b in zip(_events(E, catalog, fams, base, "sd_target", 60), _events(E, catalog, fams, proton, "sd_target", 60)):
        assert bool(a.ope) == bool(b.ope)
        if b.ope:
            n_ope += 1
            n_moved += a.excitation_mass != b.excitation_mass
            E.event_conservation(b)
        else:
            assert [h.four_momentum for h in a.final_state] == [h.four_momentum for h in b.final_state]
    assert n_ope >= 15 and n_moved == n_ope


def test_the_isospin_law_adds_the_neutral_exchange(physics):
    """Under the law's pion "isospin" the band's upper part f0 turns into p + p -> p + X+: a proton remnant and an
    excited line of charge one; the charged events of the band and every other event stay bit for bit."""
    E, catalog, fams, shape, flavor = physics
    charged = dataclasses.replace(E._params_at("sd_target", shape, flavor), sd_ope=(0.6, 0.003, 0.89, 0.93, "proton", "regge"),
                                  sd_two_sided=(2.0, 0.0, 4.0, "o"))
    iso = dataclasses.replace(charged, sd_ope=charged.sd_ope + ("isospin",))
    n_charged = n_neutral = 0
    for a, b in zip(_events(E, catalog, fams, charged, "sd_target", 160), _events(E, catalog, fams, iso, "sd_target", 160)):
        assert bool(a.ope) == bool(b.ope)
        remnant = [h for h in b.final_state if h.line == "remnant"]
        if b.ope and remnant[0].pdg_id == 2212:
            n_neutral += 1
            assert len(remnant) == 1
            excited = [h for h in b.final_state if h.line != "remnant"]
            assert sum(int(h.qn[1]) for h in excited) == 1 and sum(int(h.qn[0]) for h in excited) == 1
            assert b.exchange_configuration.i3_transfer == 0 and not b.exchange_configuration.is_vacuum
            E.event_conservation(b)
        else:
            n_charged += bool(b.ope)
            assert a.excitation_mass == b.excitation_mass and a.t == b.t
            assert [h.four_momentum for h in a.final_state] == [h.four_momentum for h in b.final_state]
    assert n_neutral >= 8 and 0.15 < n_neutral / (n_neutral + n_charged) < 0.6


def test_malformed_components_are_refused(physics):
    E, catalog, fams, shape, flavor = physics
    base = E._params_at("sd_target", shape, flavor)
    for bad in ((0.2, 0.5, 0.3, 0.9, 0.3), (0.2, 0.003, 0.89, -1.0, 0.3), (0.2, 0.003, 0.89, 0.9), (0.9, 0.003, 0.89, 0.9, 0.3),
                (0.2, 0.003, 0.89, 0.9, "dipole"), (0.2, 0.003, 0.89, 0.9, "proton", "flat")):
        with pytest.raises(ValueError):
            _events(E, catalog, fams, dataclasses.replace(base, sd_ope=bad), "sd_target", 40)


# ----------------------------------------------------------------------
# The engine (2026-10-26): the share free beside the tail
# ----------------------------------------------------------------------
RUNS = NESTED_ROOT / "config" / "runs"


def _engine(name, path=RUNS / "r121.json"):
    spec = importlib.util.spec_from_file_location(name, ENGINE)
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    E.configure(path)
    return E


def test_the_engine_weights_the_one_pion_exchange_share():
    e = _engine("run_engine_ope_share_test")
    assert e.OPE_FREE and e.SD_OPE_LAW == (0.003, 0.89, 0.9, 0.3) and e.SD_TWO_SIDED == (2.0, 0.0, 4.0, "o")
    assert e.EXTRA_SHAPE[-2:] == (e.TAIL_NAME, e.OPE_NAME)
    s0, so0 = (float(e.GEN0_SHAPE[e.shape_pos(n)]) for n in (e.TAIL_NAME, e.OPE_NAME))
    params = e._params_at("sd_projectile", e.GEN0_SHAPE, e.GEN_FLAVOR)
    assert params.sd_ope == (so0,) + e.SD_OPE_LAW and params.sd_two_sided == e.SD_TWO_SIDED
    assert e._params_at("nd0", e.GEN0_SHAPE, e.GEN_FLAVOR).sd_ope is None
    # the class share: the coherence-region part over (1 - s - s_pi)
    a = e.anchors()
    phys0 = {p["name"]: float(p["gen0"] if p.get("gen0") is not None else p["value"]) for p in e.SPEC["parameters"]}
    x = e.x_from_physical({**phys0, "sd_tail_share": 0.25, "sd_ope_share": 0.12}, None)
    phys = e.physical_from_x(np.append(x, 0.0) if len(x) == len(e.SHAPE_NAMES) else x, None, a)
    assert phys["f_sd_total"] == pytest.approx(e.sd_class_share(a, phys["epsilon"]) / (1.0 - 0.25 - 0.12), rel=1e-12)
    assert e.tail_of(phys) == pytest.approx((0.25, 0.12))

    # the weights of a stand-in block: tail s / s0, one-pion exchange s_pi / s_pi0, the rest (1 - s - s_pi) / (1 - s0 - s_pi0)
    class _T:
        point = e.GEN0_SHAPE
    tail = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    ope = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    m2 = np.array([60.0, 70.0, 5.0, 120.0, 10.0])
    block = {"n": 5, "shape": {"m2": m2, "t": np.full(5, -0.1), "m_rem": np.full(5, 0.93827231),
                               "m2_low": np.full(5, 1.07 ** 2), "slope_factor": np.full(5, 1.0), "tail": tail, "ope": ope}}
    eps0, b0, eps_nd0 = e.GEN0_SHAPE[:3]
    log_w = e.Templates.log_weights(_T(), "sd_projectile", block, eps0, b0, eps_nd0, f_tail=(0.30, 0.15))
    np.testing.assert_allclose(log_w[tail > 0.5], math.log(0.30 / s0))
    np.testing.assert_allclose(log_w[ope > 0.5], math.log(0.15 / so0))
    rest = (tail < 0.5) & (ope < 0.5)
    np.testing.assert_allclose(log_w[rest], math.log((1.0 - 0.30 - 0.15) / (1.0 - s0 - so0)))
    np.testing.assert_allclose(e.Templates.log_weights(_T(), "sd_projectile", block, eps0, b0, eps_nd0, f_tail=(s0, so0)),
                               0.0, atol=1e-12)
    # the share key of the caches
    assert e._share_key((0.3, 0.15)) == (0.3, 0.15) and e._share_key(0.3) == 0.3


def test_a_one_pion_exchange_event_is_no_isovector_event():
    e = _engine("run_engine_ope_iv_test")

    class _Ev:
        exchange_configuration = __import__("na49_mixing.topology_single_diffraction", fromlist=["_ope_configuration"])._ope_configuration("target")
        ope = True
    assert e.iv_config_index(_Ev()) == -1
    _Ev.ope = None
    assert e.iv_config_index(_Ev()) >= 0


def test_the_engine_refuses_a_lone_law_or_share(tmp_path):
    s = json.loads((RUNS / "r121.json").read_text())
    for edit in ("no_share", "no_law", "no_tail"):
        t = json.loads(json.dumps(s))
        if edit == "no_share":
            t["parameters"] = [p for p in t["parameters"] if p["name"] != "sd_ope_share"]
        elif edit == "no_law":
            del t["topology"]["sd_ope"]
        else:
            t["topology"].pop("sd_tail")
            t["parameters"] = [p for p in t["parameters"] if p["name"] != "sd_tail_share"]
        path = tmp_path / f"{edit}.json"
        path.write_text(json.dumps(t))
        with pytest.raises(ValueError):
            _engine(f"run_engine_ope_refuse_{edit}", path)


def test_the_engine_takes_the_proton_form_factor(tmp_path):
    s = json.loads((RUNS / "r121.json").read_text())
    s["topology"]["sd_ope"] = [0.003, 0.89, 0.93, "proton", "regge"]
    path = tmp_path / "proton.json"
    path.write_text(json.dumps(s))
    e = _engine("run_engine_ope_proton_test", path)
    assert e.SD_OPE_LAW == (0.003, 0.89, 0.93, "proton", "regge")
    params = e._params_at("sd_target", e.GEN0_SHAPE, e.GEN_FLAVOR)
    assert ope_component_of(params.sd_ope)[3] == APPLEBY
    s["topology"]["sd_ope"] = [0.003, 0.89, 0.93, "proton", "regge", "isospin"]
    path.write_text(json.dumps(s))
    e = _engine("run_engine_ope_isospin_test", path)
    assert e.SD_OPE_LAW == (0.003, 0.89, 0.93, "proton", "regge", "isospin")
    assert ope_component_of(e._params_at("sd_target", e.GEN0_SHAPE, e.GEN_FLAVOR).sd_ope)[3].pion == "isospin"
    for bad in ([0.003, 0.89, 0.93, "dipole"], [0.003, 0.89, 0.93, "proton", "flat"], [0.003, 0.89, 0.93, 0.3, "regge", 1],
                [0.003, 0.89, "proton", 0.3], [0.003, 0.89, 0.93, "proton", "regge", "neutral"],
                [0.003, 0.89, 0.93, "proton", "isospin"]):
        s["topology"]["sd_ope"] = bad
        path.write_text(json.dumps(s))
        with pytest.raises(ValueError):
            _engine("run_engine_ope_proton_refuse", path)


def test_the_r121_spec_is_r120_with_the_one_pion_exchange():
    r120, r121 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r120", "r121"))
    assert {k: v for k, v in r121["topology"].items() if k not in ("sd_ope", "sd_two_sided")} == r120["topology"]
    assert r121["topology"]["sd_ope"] == [0.003, 0.89, 0.9, 0.3] and r121["topology"]["sd_two_sided"] == [2.0, 0.0, 4.0, "o"]
    for block in ("kernel", "fit", "diffractive_split", "isovector", "fixed", "statistics", "central_component",
                  "single_diffraction_window", "systematic_variants", "flavor_bbar_scope"):
        assert r121[block] == r120[block], block
    names = [p["name"] for p in r121["parameters"]]
    assert names == [p["name"] for p in r120["parameters"]][:-1] + ["sd_ope_share", "f_central"]
    ope = r121["parameters"][names.index("sd_ope_share")]
    assert ope["free"] and ope["gen0"] == 0.10 and ope["bounds"] == [0.0, 0.35]
    g0 = r121["generation_0"]
    assert g0["run"] == "r120_epsilon_fixed_2026-10-26" and g0["from_fit"] == r120["generation_0"]["from_fit"]
    assert "sd_ope_share" in g0["judgment"] and "epsilon" in g0["judgment"]
    assert r121["seeds"]["raw_first"] > r120["seeds"]["raw_first"] + 511
    branch = NESTED_ROOT.parent / "00runs" / "r120_epsilon_fixed_2026-10-26" / "branch_report_f0.json"
    if branch.is_file():
        report = json.loads(branch.read_text())
        assert g0["branch_digest"] == report["digest"]
        gen0 = {p["name"]: p["gen0"] for p in r121["parameters"]}
        for name in g0["from_fit"]:
            assert float(gen0[name]) == float(report["fitted"][name]), name


def test_the_r122_spec_is_r121_with_the_literature_pion_term():
    r121, r122 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r121", "r122"))
    assert {k: v for k, v in r122["topology"].items() if k != "sd_ope"} == {
        k: v for k, v in r121["topology"].items() if k != "sd_ope"}
    assert r122["topology"]["sd_ope"] == [0.003, 0.89, 0.93, "proton", "regge"]
    for block in ("kernel", "fit", "diffractive_split", "isovector", "fixed", "statistics", "central_component",
                  "single_diffraction_window", "systematic_variants", "flavor_bbar_scope"):
        assert r122[block] == r121[block], block
    strip = lambda p: {k: v for k, v in p.items() if k != "gen0"}  # noqa: E731
    assert [strip(p) for p in r122["parameters"]] == [strip(p) for p in r121["parameters"]]
    g0 = r122["generation_0"]
    assert g0["run"] == "r121_one_pion_exchange_2026-10-26"
    assert g0["from_fit"] == r121["generation_0"]["from_fit"] + ["sd_ope_share"]
    assert "sd_ope_share" not in g0["judgment"] and "epsilon" in g0["judgment"]
    assert r122["seeds"]["raw_first"] > r121["seeds"]["raw_first"] + 511
    assert r122["seeds"]["partition_base"] > r121["seeds"]["partition_base"]
    branch = NESTED_ROOT.parent / "00runs" / "r121_one_pion_exchange_2026-10-26" / "branch_report_f0.json"
    if branch.is_file():
        report = json.loads(branch.read_text())
        assert g0["branch_digest"] == report["digest"]
        gen0 = {p["name"]: p["gen0"] for p in r122["parameters"]}
        for name in g0["from_fit"]:
            assert float(gen0[name]) == float(report["fitted"][name]), name


def test_the_r123_spec_is_r122_with_the_neutral_partner():
    r122, r123 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r122", "r123"))
    assert {k: v for k, v in r123["topology"].items() if k != "sd_ope"} == {
        k: v for k, v in r122["topology"].items() if k != "sd_ope"}
    assert r123["topology"]["sd_ope"] == r122["topology"]["sd_ope"] + ["isospin"]
    for block in ("kernel", "fit", "diffractive_split", "isovector", "fixed", "statistics", "central_component",
                  "single_diffraction_window", "systematic_variants", "flavor_bbar_scope"):
        assert r123[block] == r122[block], block
    strip = lambda p: {k: v for k, v in p.items() if k not in ("gen0", "bounds", "starts")}  # noqa: E731
    assert [strip(p) for p in r123["parameters"]] == [strip(p) for p in r122["parameters"]]
    by_name = {p["name"]: p for p in r123["parameters"]}
    # the share is the whole pion exchange now; with the tail it stays below 1
    assert by_name["sd_ope_share"]["bounds"][1] + by_name["sd_tail_share"]["bounds"][1] < 1.0
    assert all(isinstance(v, str) or v < by_name["sd_ope_share"]["bounds"][1] for v in by_name["sd_ope_share"]["starts"])
    g0 = r123["generation_0"]
    assert g0["run"] == "r122_ope_proton_form_factor_2026-10-27"
    assert r123["seeds"]["raw_first"] > r122["seeds"]["raw_first"] + 511
    assert r123["seeds"]["partition_base"] > r122["seeds"]["partition_base"]
    branch = NESTED_ROOT.parent / "00runs" / g0["run"] / "branch_report_f0.json"
    if branch.is_file():
        report = json.loads(branch.read_text())
        assert g0["branch_digest"] == report["digest"]
        for name in g0["from_fit"]:
            assert float(by_name[name]["gen0"]) == float(report["fitted"][name]), name


def test_block_minima_on_the_share_bound_are_counted():
    """R122 (2026-10-27): all 32 replica-block fits ended at the share's bound 0.35, the single-block objective biased
    along the share by its own Monte Carlo variance; the spread of zero read as stable.  The engine counts the minima
    at a bound and names a pile of them in its residual register."""
    e = _engine("run_engine_block_bound_test", RUNS / "r122.json")
    lo, hi = e.PARAM_INDEX[e.OPE_NAME].bounds
    assert e.blocks_on_bound(e.OPE_NAME, [hi - 6e-5] * 30 + [0.30, 0.29]) == 30
    assert e.blocks_on_bound(e.OPE_NAME, [lo + 5e-4, 0.2, 0.3]) == 1
    assert e.blocks_on_bound(e.OPE_NAME, [0.30, 0.32, 0.34]) == 0
    log_names = [n for n in e.SHAPE_NAMES if e.PARAM_INDEX[n].transform == "log"]
    assert log_names
    n = log_names[0]
    lo, hi = e.PARAM_INDEX[n].bounds
    assert e.blocks_on_bound(n, [hi * (1 - 5e-4), math.sqrt(lo * hi)]) == 1
    src = ENGINE.read_text()
    assert 'blocks_on_bound(name, s["block_values"])' in src and '"kind": "replica_block_minima_on_bound"' in src


def test_the_eikonal_absorption_is_a_survival_factor():
    """The absorption model's S^2 lies in (0, 1], falls with xi at fixed q (the pion's range 1 / mu shrinks with xi)
    and with q at fixed xi (the small impact parameters are screened); the law without it adds nothing."""
    law = dataclasses.replace(APPLEBY, absorption="eikonal")
    q = np.array([0.05, 0.2, 0.4, 0.6])
    rows = {}
    for xi in (0.01, 0.1, 0.25, 0.4):
        t = -(q * q + xi * xi * 0.938272 ** 2) / (1.0 - xi)
        rows[xi] = np.exp(ope_log_absorption(xi, t, law))
        assert np.all((rows[xi] > 0.0) & (rows[xi] <= 1.0))
        assert np.all(np.diff(rows[xi]) < 0.0)
    for a, b in ((0.01, 0.1), (0.1, 0.25), (0.25, 0.4)):
        assert np.all(rows[a][:3] > rows[b][:3])
    assert rows[0.1][0] == pytest.approx(0.74, abs=0.02) and rows[0.4][0] == pytest.approx(0.48, abs=0.02)
    ts = -np.array([0.005, 0.02, 0.1, 0.5])
    assert np.all(ope_log_absorption(0.1, ts, APPLEBY) == 0.0)
    assert np.allclose(ope_log_density(0.2, ts, law), ope_log_density(0.2, ts, APPLEBY) + ope_log_absorption(0.2, ts, law))
    assert np.exp(ope_log_absorption(0.2, -0.3, law)) == pytest.approx(np.exp(ope_log_absorption(0.2, -0.3, law)))
    with pytest.raises(ValueError):
        OPELaw(0.93, 0.0, form_factor="proton", absorption="glauber")


def test_without_screening_the_survival_factor_is_one(monkeypatch):
    import na49_mixing.one_pion_exchange as OPE
    law = dataclasses.replace(APPLEBY, absorption="eikonal")
    monkeypatch.setattr(OPE, "PP_TOTAL_CROSS_SECTION_MB", 0.0)
    OPE._absorption_table.cache_clear()
    try:
        ts = -np.array([0.005, 0.02, 0.1, 0.5, 2.0])
        assert np.allclose(np.exp(ope_log_absorption(0.2, ts, law)), 1.0, atol=3e-3)
    finally:
        OPE._absorption_table.cache_clear()


def test_the_absorption_suppresses_the_heavy_systems_most():
    """The absorbed marginal lies below the unabsorbed one at every mass, by a factor that falls with the mass, and
    integrates to about half of it; the neutral fraction barely moves; the component takes the eighth entry."""
    law = dataclasses.replace(APPLEBY, absorption="eikonal")
    ys, c_abs = ope_marginal_cells(M2_THRESHOLD, 0.89 * S, M2_NEUTRON, S, law)
    _, c_born = ope_marginal_cells(M2_THRESHOLD, 0.89 * S, M2_NEUTRON, S, APPLEBY)
    ratio = c_abs / c_born
    assert np.all(ratio < 1.0) and 0.45 < c_abs.sum() / c_born.sum() < 0.55
    xi = np.exp(0.5 * (ys[1:] + ys[:-1])) / S
    coarse = [ratio[(xi > lo) & (xi < hi)].mean() for lo, hi in ((0.05, 0.15), (0.2, 0.3), (0.35, 0.45))]
    assert coarse[0] > coarse[1] > coarse[2]
    iso = dataclasses.replace(law, pion="isospin")
    assert ope_parts(iso) == (law, dataclasses.replace(law, pion="neutral"))
    f0 = ope_neutral_fraction(M2_THRESHOLD, 0.89 * S, M2_NEUTRON, 0.93827231 ** 2, S, iso)
    assert f0 == pytest.approx(0.349, abs=0.003)
    assert ope_component_of((0.3, 0.003, 0.89, 0.93, "proton", "regge", "isospin", "eikonal"))[3] == iso
    assert ope_component_of((0.3, 0.003, 0.89, 0.93, "proton", "regge", "isospin", "none"))[3] == dataclasses.replace(iso, absorption="none")
    for bad in ((0.3, 0.003, 0.89, 0.93, "proton", "regge", "isospin", "glauber"),
                (0.3, 0.003, 0.89, 0.93, "proton", "regge", "isospin", "eikonal", 1)):
        with pytest.raises(ValueError):
            ope_component_of(bad)


def test_the_engine_takes_the_absorbed_law(tmp_path):
    s = json.loads((RUNS / "r123.json").read_text())
    s["topology"]["sd_ope"] = [0.003, 0.89, 0.93, "proton", "regge", "isospin", "eikonal"]
    path = tmp_path / "absorbed.json"
    path.write_text(json.dumps(s))
    e = _engine("run_engine_ope_absorbed_test", path)
    assert e.SD_OPE_LAW == (0.003, 0.89, 0.93, "proton", "regge", "isospin", "eikonal")
    assert ope_component_of(e._params_at("sd_target", e.GEN0_SHAPE, e.GEN_FLAVOR).sd_ope)[3].absorption == "eikonal"
    for bad in ([0.003, 0.89, 0.93, "proton", "regge", "isospin", "glauber"], [0.003, 0.89, 0.93, "proton", "regge", "eikonal"],
                [0.003, 0.89, 0.93, "proton", "regge", "isospin", "eikonal", "x"]):
        s["topology"]["sd_ope"] = bad
        path.write_text(json.dumps(s))
        with pytest.raises(ValueError):
            _engine("run_engine_ope_absorbed_refuse", path)


def test_the_r124_spec_is_r123_with_the_absorptive_correction():
    r123, r124 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r123", "r124"))
    assert {k: v for k, v in r124["topology"].items() if k != "sd_ope"} == {
        k: v for k, v in r123["topology"].items() if k != "sd_ope"}
    assert r124["topology"]["sd_ope"] == r123["topology"]["sd_ope"] + ["eikonal"]
    for block in ("kernel", "fit", "diffractive_split", "isovector", "fixed", "statistics", "central_component",
                  "single_diffraction_window", "systematic_variants", "flavor_bbar_scope"):
        assert r124[block] == r123[block], block
    strip = lambda p: {k: v for k, v in p.items() if k not in ("gen0",)}  # noqa: E731
    assert [strip(p) for p in r124["parameters"]] == [strip(p) for p in r123["parameters"]]
    by_name = {p["name"]: p for p in r124["parameters"]}
    g0 = r124["generation_0"]
    assert g0["run"] == "r123_ope_neutral_partner_2026-10-27"
    assert r124["seeds"]["raw_first"] > r123["seeds"]["raw_first"] + 511
    assert r124["seeds"]["partition_base"] > r123["seeds"]["partition_base"]
    branch = NESTED_ROOT.parent / "00runs" / g0["run"] / "branch_report_f0.json"
    if branch.is_file():
        report = json.loads(branch.read_text())
        assert g0["branch_digest"] == report["digest"]
        for name in g0["from_fit"]:
            assert float(by_name[name]["gen0"]) == float(report["fitted"][name]), name


def test_the_pdg_elastic_profile_gives_both_measured_cross_sections():
    """``absorption="eikonal_pdg"`` (2026-10-30): the profile takes the PDG compilation's p+p total cross section at
    158 GeV/c and the slope at which the purely absorptive Gaussian profile also gives the measured elastic cross
    section, sigma_tot = 2 int Gamma d^2b and sigma_el = int Gamma^2 d^2b; "eikonal" keeps the R124 inputs."""
    import na49_mixing.one_pion_exchange as OPE
    pdg = dataclasses.replace(APPLEBY, absorption="eikonal_pdg")
    assert OPE.elastic_profile(dataclasses.replace(APPLEBY, absorption="eikonal")) == (38.9, 11.6)
    sigma_tot, slope = OPE.elastic_profile(pdg)
    assert sigma_tot == pytest.approx(38.746) and slope == pytest.approx(11.01, abs=0.01)
    b = np.linspace(0.0, 40.0, 40001)
    gamma = sigma_tot * OPE._MB_TO_GEV2 / (4.0 * math.pi * slope) * np.exp(-b * b / (2.0 * slope))
    def area(f):
        g = 2.0 * math.pi * b * f
        return float((0.5 * (g[1:] + g[:-1]) * np.diff(b)).sum()) / OPE._MB_TO_GEV2
    assert area(2.0 * gamma) == pytest.approx(sigma_tot, rel=1e-6)
    assert area(gamma * gamma) == pytest.approx(OPE.PP_ELASTIC_CROSS_SECTION_MB_PDG, rel=1e-6)
    q = np.array([0.05, 0.2, 0.4, 0.6])
    for xi in (0.1, 0.4):
        t = -(q * q + xi * xi * 0.938272 ** 2) / (1.0 - xi)
        s_pdg = np.exp(ope_log_absorption(xi, t, pdg))
        s_r124 = np.exp(ope_log_absorption(xi, t, dataclasses.replace(APPLEBY, absorption="eikonal")))
        assert np.all((s_pdg > 0.0) & (s_pdg <= 1.0)) and np.all(np.diff(s_pdg) < 0.0)
        assert np.allclose(s_pdg, s_r124, rtol=0.15) and not np.allclose(s_pdg, s_r124, rtol=1e-3)
    assert ope_component_of((0.3, 0.003, 0.89, 0.93, "proton", "regge", "isospin", "eikonal_pdg"))[3].absorption == "eikonal_pdg"


def test_the_r129_spec_is_r128_with_the_pdg_elastic_profile():
    r128, r129 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r128", "r129"))
    assert {k: v for k, v in r129["topology"].items() if k != "sd_ope"} == {
        k: v for k, v in r128["topology"].items() if k != "sd_ope"}
    assert r129["topology"]["sd_ope"] == r128["topology"]["sd_ope"][:-1] + ["eikonal_pdg"]
    assert r128["topology"]["sd_ope"][-1] == "eikonal"
    for block in ("kernel", "fit", "diffractive_split", "isovector", "fixed", "statistics", "central_component",
                  "single_diffraction_window", "systematic_variants", "flavor_bbar_scope"):
        assert r129[block] == r128[block], block
    strip = lambda p: {k: v for k, v in p.items() if k not in ("gen0",)}  # noqa: E731
    assert [strip(p) for p in r129["parameters"]] == [strip(p) for p in r128["parameters"]]
    by_name = {p["name"]: p for p in r129["parameters"]}
    g0 = r129["generation_0"]
    assert g0["run"] == "r128_kaon_identification_knots_2026-10-28"
    e = _engine("run_engine_r129_spec_test", RUNS / "r129.json")
    assert e.SD_OPE_LAW == (0.003, 0.89, 0.93, "proton", "regge", "isospin", "eikonal_pdg")
    assert ope_component_of(e._params_at("sd_target", e.GEN0_SHAPE, e.GEN_FLAVOR).sd_ope)[3].absorption == "eikonal_pdg"
    assert r129["seeds"]["raw_first"] > r128["seeds"]["raw_first"] + 511
    assert r129["seeds"]["partition_base"] > r128["seeds"]["partition_base"]
    branch = NESTED_ROOT.parent / "00runs" / g0["run"] / "branch_report_f0.json"
    if branch.is_file():
        report = json.loads(branch.read_text())
        assert g0["branch_digest"] == report["digest"]
        for name in g0["from_fit"]:
            assert float(by_name[name]["gen0"]) == float(report["fitted"][name]), name
