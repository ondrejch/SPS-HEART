"""The measured Dalitz distributions of the light-meson three-body decays
(``KernelConfig.dalitz_matrix_elements``, the law ``phase_space_dalitz``,
2026-10-20): the omega's P-wave |p+ x p-|^2, the eta -> 3 pi and
eta' -> eta pi pi Dalitz parameters, drawn on the four uniforms of the flat
vertex."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.dalitz_matrix_elements import DALITZ_MATRIX_ELEMENTS, dalitz_element_for
from na49_mixing.hadron_catalog import THREE_BODY_LAWS
from na49_mixing.kernel_config import KernelConfig, load_kernel
from na49_mixing.resonance_decay import three_body_phase_space
from na49_mixing.topology_runner import TopologyNominalParams, run_topology

NESTED_ROOT = Path(__file__).resolve().parents[1]
RUNS = NESTED_ROOT / "config" / "runs"
LABELS = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")
MPI, MPI0, META = 0.13957039, 0.1349768, 0.547862


def test_the_option_and_the_law():
    assert "phase_space_dalitz" in THREE_BODY_LAWS
    assert KernelConfig().dalitz_matrix_elements is False and KernelConfig().as_dict()["dalitz_matrix_elements"] is False
    with pytest.raises(ValueError, match="three_body_phase_space"):
        KernelConfig(dalitz_matrix_elements=True)
    with pytest.raises(ValueError, match="dalitz_matrix_elements"):
        KernelConfig(three_body_phase_space=True, dalitz_matrix_elements=1)
    assert {k[0] for k in DALITZ_MATRIX_ELEMENTS} == {223, 221, 331}
    assert dalitz_element_for(223, (211, -211, 111)) is dalitz_element_for(223, (211, -211, 111))
    assert dalitz_element_for(223, (211, -211)) is None and dalitz_element_for(333, (211, -211, 111)) is None


def test_the_uniform_budget_and_the_flat_limit():
    """The element keeps the four uniforms; without one the vertex is the
    phase-space law bit for bit."""
    el = dalitz_element_for(223, (211, -211, 111))
    for law in ("phase_space", "phase_space_dalitz"):
        r1, r2 = np.random.default_rng(4), np.random.default_rng(4)
        for _ in range(50):
            a = three_body_phase_space(0.78, MPI, MPI, MPI0, r1, law="phase_space")
            b = three_body_phase_space(0.78, MPI, MPI, MPI0, r2, law=law)
            assert all(np.array_equal(x, y) for x, y in zip(a, b))
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    for _ in range(50):
        three_body_phase_space(0.78, MPI, MPI, MPI0, r1, law="phase_space")
        p = three_body_phase_space(0.78, MPI, MPI, MPI0, r2, law="phase_space", matrix_element=el)
        assert np.allclose(sum(p), (0.78, 0.0, 0.0, 0.0), atol=1e-12)
        for v, m in zip(p, (MPI, MPI, MPI0)):
            assert math.isclose(v[0] ** 2 - v[1:] @ v[1:], m * m, rel_tol=1e-9)
    assert r1.random() == r2.random()
    with pytest.raises(ValueError, match="phase_space law"):
        three_body_phase_space(0.78, MPI, MPI, MPI0, r1, law="flat_m2", matrix_element=el)


@pytest.mark.parametrize("parent, daughters, M, masses", [
    (223, (211, -211, 111), 0.78265, (MPI, MPI, MPI0)),
    (223, (111, -211, 211), 0.79, (MPI0, MPI, MPI)),
    (221, (211, -211, 111), META, (MPI, MPI, MPI0)),
    (331, (221, 211, -211), 0.95778, (META, MPI, MPI)),
])
def test_the_draw_follows_the_element(parent, daughters, M, masses):
    """The sampled Dalitz plot equals flat phase space reweighted by the
    element: the (0, 1) and (1, 2) mass-squared histograms agree."""
    el = dalitz_element_for(parent, daughters)
    rng_f, rng_d = np.random.default_rng(11), np.random.default_rng(12)
    n = 12000
    flat, w, drawn = [], [], []
    for _ in range(n):
        a = three_body_phase_space(M, *masses, rng_f, law="phase_space")
        t = tuple(float(v[0]) - m for v, m in zip(a, masses))
        cr = np.cross(a[0][1:], a[1][1:])
        w.append(el(t, masses, M - sum(masses), float(cr @ cr)))
        flat.append(a)
        drawn.append(three_body_phase_space(M, *masses, rng_d, law="phase_space", matrix_element=el))
    w = np.array(w)

    def m2(points, i, j):
        s = np.array([p[i] + p[j] for p in points])
        return s[:, 0] ** 2 - np.sum(s[:, 1:] ** 2, axis=1)

    for i, j in ((0, 1), (1, 2)):
        a, b = m2(flat, i, j), m2(drawn, i, j)
        edges = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 11)
        ha, _ = np.histogram(a, bins=edges, weights=w)
        hb, _ = np.histogram(b, bins=edges)
        expect = ha / ha.sum() * n
        assert np.all(np.abs(hb - expect) < 5.0 * np.sqrt(expect + 1.0) + 0.02 * expect), (i, j, hb, expect)


def test_the_omega_p_wave_empties_the_soft_end():
    """The P-wave vanishes on the Dalitz-plot boundary: the omega's charged
    pions below 0.10 GeV/c fall from about 6% to about 2% at the same mean."""
    el = dalitz_element_for(223, (211, -211, 111))
    r1, r2 = np.random.default_rng(21), np.random.default_rng(21)
    flat, me = [], []
    for _ in range(8000):
        a = three_body_phase_space(0.78265, MPI, MPI, MPI0, r1, law="phase_space")
        b = three_body_phase_space(0.78265, MPI, MPI, MPI0, r2, law="phase_space", matrix_element=el)
        flat += [np.linalg.norm(v[1:]) for v in a[:2]]
        me += [np.linalg.norm(v[1:]) for v in b[:2]]
    flat, me = np.array(flat), np.array(me)
    assert abs(me.mean() - flat.mean()) < 0.004
    assert (flat < 0.10).mean() > 0.05 and (me < 0.10).mean() < 0.032
    assert me.std() < 0.85 * flat.std()


def test_the_kernel_option_sets_the_law_and_events_conserve():
    r102 = json.loads((RUNS / "r102.json").read_text())
    assert r102["kernel"]["three_body_phase_space"] is True
    cat0, _ = load_kernel(KernelConfig(**r102["kernel"]), NESTED_ROOT)
    cat, fams = load_kernel(KernelConfig(**{**r102["kernel"], "dalitz_matrix_elements": True}), NESTED_ROOT)
    assert cat0.three_body_law == "phase_space" and cat.three_body_law == "phase_space_dalitz"
    params = TopologyNominalParams(epsilon=0.53, b_sd=6.44, epsilon_nd=0.247, nd_central=False,
                                   remnant_weights={"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.0,
                                                    "high_mass_baryonic_continuum": 1.0}, **r102["topology"])
    moved = 0
    for seed in range(20):
        rngs = {f"rng_{lab}": np.random.default_rng([seed, 16384, k]) for k, lab in enumerate(LABELS)}
        ev = run_topology("nd", params, cat, fams, **rngs)
        rngs0 = {f"rng_{lab}": np.random.default_rng([seed, 16384, k]) for k, lab in enumerate(LABELS)}
        ev0 = run_topology("nd", params, cat0, fams, **rngs0)
        for i in range(3):
            assert sum(int(h.qn[i]) for h in ev.final_state) == (2, 2, 0)[i]
        # the same hadrons: only the omega, eta and eta' decay kinematics move
        assert sorted(int(h.pdg_id) for h in ev.final_state) == sorted(int(h.pdg_id) for h in ev0.final_state)
        moved += any(not np.allclose(a.four_momentum, b.four_momentum)
                     for a, b in zip(ev.final_state, ev0.final_state))
    assert moved > 0


def test_the_r104_spec_is_r103_with_the_dalitz_law():
    r103, r104 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r103", "r104"))
    assert r104["kernel"] == {**r103["kernel"], "dalitz_matrix_elements": True}
    assert r104["topology"] == r103["topology"]
    for block in ("fit", "diffractive_split", "isovector", "fixed", "statistics"):
        assert r104[block] == r103[block]
    assert [p["name"] for p in r104["parameters"]] == [p["name"] for p in r103["parameters"]]
    assert r104["generation_0"]["run"] == "r103_redraw_shapes_2026-10-20"
    assert r104["seeds"]["raw_first"] > r103["seeds"]["raw_first"] + 511
    branch = NESTED_ROOT.parent / "00runs" / "r103_redraw_shapes_2026-10-20" / "branch_report_f0.json"
    if branch.is_file():
        report = json.loads(branch.read_text())
        assert r104["generation_0"]["branch_digest"] == report["digest"]
        gen0 = {p["name"]: p["gen0"] for p in r104["parameters"]}
        for name in r104["generation_0"]["from_fit"]:
            assert float(gen0[name]) == float(report["fitted"][name]), name


def test_the_omega_density_follows_the_besiii_variables():
    """The omega -> pi+ pi- pi0 density 1 + 2 alpha z + 2 beta z^(3/2) sin 3 phi in BESIII's variables (PRD 98 (2018)
    112007, Eq. (1)): x = (t - u)/(sqrt(3) R), y = (s - s0)/R + 2(m+- - m0)/(m_omega - 2 m+- - m0), with t = m^2(pi- pi0),
    u = m^2(pi0 pi+), s = m^2(pi+ pi-).  At points where the odd beta term is large the code's density agrees within the
    charged-neutral mass offset (1.2% at the edge), where the opposite beta sign misses by 3-11% (the code review of 2026-11-04,
    finding M4)."""
    import math as _m
    from na49_mixing import dalitz_matrix_elements as D
    mw, mc, m0 = 0.78266, 0.13957, 0.13498
    q = mw - 2 * mc - m0
    r = 2.0 / 3.0 * mw * q
    checked = 0
    for tp, tm in ((0.05, 0.25), (0.25, 0.05), (0.15, 0.03), (0.02, 0.18), (0.12, 0.12), (0.1, 0.2)):
        t0 = q - tp - tm
        if t0 <= 0.0:
            continue
        e = (tp + mc, tm + mc, t0 + m0)
        s = mw * mw + m0 * m0 - 2 * mw * e[2]
        t = mw * mw + mc * mc - 2 * mw * e[0]
        u = mw * mw + mc * mc - 2 * mw * e[1]
        s0 = (s + t + u) / 3.0
        x = (t - u) / (_m.sqrt(3.0) * r)
        y = (s - s0) / r + 2 * (mc - m0) / (mw - 2 * mc - m0)
        z = x * x + y * y
        odd = 3 * x * x * y - y ** 3
        want = 1 + 2 * D.OMEGA_ALPHA * z + 2 * D.OMEGA_BETA * odd
        wrong = 1 + 2 * D.OMEGA_ALPHA * z - 2 * D.OMEGA_BETA * odd
        got = D._omega_3pi((tp, tm, t0), mw, q, 1.0)
        assert abs(got / want - 1.0) < 0.015, (tp, tm, got, want)   # the charged-neutral offset of y
        if abs(wrong / want - 1.0) > 0.02:
            checked += 1
            assert abs(got / want - 1.0) < abs(got / wrong - 1.0)
    assert checked >= 2
