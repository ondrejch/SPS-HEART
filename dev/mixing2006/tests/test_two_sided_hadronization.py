"""The two-sided hadronization of a heavy excited system (2026-10-26): the one-pion-exchange X++ cascaded as one
isotropic line puts its baryon and its hard pions near the centre; split into a baryonic line along the incoming
proton and the exchanged object's line opposite, the baryon keeps its side.

- the kinematics: the mass draw's law and range, the fixed budgets (two plus two uniforms), the two lines summing to
  the excitation's four-momentum on their mass shells, the baryonic line along the flight direction;
- generate_sd: the split on the scope's systems only (the other events the same bit for bit), exact conservation, the
  X++ baryon away from the centre, the runner's refusals.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.two_sided_hadronization import draw_line_mass, split_kinematics, split_masses

NESTED_ROOT = Path(__file__).resolve().parents[1]
ENGINE = NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py"
SPEC = NESTED_ROOT / "config" / "runs" / "r120.json"


def test_the_line_mass_law_and_its_range():
    assert draw_line_mass(1.0, 4.0, 0.3, 0.0) == pytest.approx(1.0)
    assert draw_line_mass(1.0, 4.0, 0.3, 1.0) == pytest.approx(4.0)
    # eps = 0 is flat in ln M^2: the median at the geometric mean
    assert draw_line_mass(1.0, 4.0, 0.0, 0.5) == pytest.approx(2.0)
    # a steeper law puts the median lower
    assert draw_line_mass(1.0, 4.0, 0.5, 0.5) < draw_line_mass(1.0, 4.0, 0.0, 0.5)
    with pytest.raises(ValueError):
        draw_line_mass(2.0, 1.0, 0.0, 0.5)


def test_the_masses_take_two_uniforms_and_fit_below_m_x():
    a, b = np.random.default_rng(1), np.random.default_rng(1)
    m1, m2 = split_masses(8.0, 1.07, 0.27, 0.2, a)
    b.random(), b.random()
    assert a.random() == b.random()
    rng = np.random.default_rng(2)
    for _ in range(500):
        m1, m2 = split_masses(6.0, 1.07, 0.27, 0.0, rng)
        assert m1 >= 1.07 and m2 >= 0.27 and m1 + m2 <= 6.0 + 1e-12
    with pytest.raises(ValueError):
        split_masses(1.2, 1.07, 0.27, 0.0, rng)


def _p4(m, p, direction):
    e = math.sqrt(m * m + p * p)
    return (e,) + tuple(p * d for d in direction)


@pytest.mark.parametrize("direction", [(0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.6, 0.0, 0.8)])
def test_the_lines_sum_to_the_excitation_on_their_shells(direction):
    rng = np.random.default_rng(7)
    mx, m1, m2 = 9.0, 3.1, 2.2
    x = _p4(mx, 40.0, direction)
    proj = []
    for _ in range(400):
        p1, p2 = split_kinematics(x, mx, m1, m2, 4.0, rng)
        for k in range(4):
            assert p1[k] + p2[k] == pytest.approx(x[k], abs=1e-9)
        for p, m in ((p1, m1), (p2, m2)):
            assert p[0] ** 2 - p[1] ** 2 - p[2] ** 2 - p[3] ** 2 == pytest.approx(m * m, rel=1e-8, abs=1e-8)
        proj.append(sum(p1[k + 1] * direction[k] for k in range(3)) / p1[0] - sum(p2[k + 1] * direction[k] for k in range(3)) / p2[0])
    # the baryonic line runs ahead along the flight direction
    assert np.mean(np.asarray(proj) > 0.0) > 0.95


def test_the_kinematics_take_two_uniforms():
    a, b = np.random.default_rng(9), np.random.default_rng(9)
    split_kinematics(_p4(7.0, 20.0, (0.0, 0.0, 1.0)), 7.0, 2.0, 1.0, 4.0, a)
    b.random(), b.random()
    assert a.random() == b.random()


@pytest.fixture(scope="module")
def physics():
    spec = importlib.util.spec_from_file_location("run_engine_two_sided_test", ENGINE)
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    E.configure(SPEC)
    catalog, fams = E._load_physics(E.FLOOR_GRID[0])
    s = json.loads(SPEC.read_text())
    g0 = {p["name"]: float(p["gen0"]) for p in s["parameters"] if p.get("gen0") is not None}
    shape = (g0["epsilon"], g0["b_sd"], g0["epsilon_nd"], g0["intact_vertex_factor"], g0["sd_tail_share"])
    flavor = tuple(g0[n] for n in E.FLAVOR_ORDER)
    return E, catalog, fams, shape, flavor


def _events(E, catalog, fams, params, cls, n):
    from na49_mixing.topology_runner import run_topology
    out = []
    for i in range(n):
        subs = E.derive_substreams(20261026, 0, 7000 + i)
        rngs = {f"rng_{lab}": np.random.default_rng(subs.stream_for(lab))
                for lab in ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")}
        out.append(run_topology(cls, params, catalog, fams, **rngs))
    return out


def test_the_split_acts_on_the_scope_only_and_moves_the_baryon_out(physics):
    E, catalog, fams, shape, flavor = physics
    beta, gamma = E.OB.cm_boost(E.SQRT_S_CM)
    base = dataclasses.replace(E._params_at("sd_projectile", shape, flavor), sd_ope=(0.3, 0.003, 0.89, 0.9, 0.3))
    split = dataclasses.replace(base, sd_two_sided=(2.0, 0.0, 4.0, "o"))
    xb_single, xb_split, n_ope = [], [], 0
    for a, b in zip(_events(E, catalog, fams, base, "sd_projectile", 100), _events(E, catalog, fams, split, "sd_projectile", 100)):
        assert bool(a.ope) == bool(b.ope) and a.excitation_mass == b.excitation_mass and a.t == b.t
        if b.ope:
            n_ope += 1
            E.event_conservation(b)
            for ev, acc in ((a, xb_single), (b, xb_split)):
                acc += [2.0 * E.OB.to_cm(h.four_momentum, beta, gamma)[3] / E.SQRT_S_CM
                        for h in ev.final_state if h.line != "remnant" and int(h.qn[0]) == 1]
        else:
            # outside the scope every event is the same bit for bit
            assert [h.four_momentum for h in a.final_state] == [h.four_momentum for h in b.final_state]
    assert n_ope >= 15
    assert np.mean(np.abs(xb_split) < 0.1) < np.mean(np.abs(xb_single) < 0.1) - 0.15
    assert np.median(xb_split) > np.median(xb_single)


def test_the_tail_scope_splits_the_heavy_tail_systems(physics):
    E, catalog, fams, shape, flavor = physics
    base = E._params_at("sd_target", shape, flavor)
    split = dataclasses.replace(base, sd_two_sided=(3.0, 0.0, 4.0, "t"))
    n_tail = 0
    for a, b in zip(_events(E, catalog, fams, base, "sd_target", 80), _events(E, catalog, fams, split, "sd_target", 80)):
        if b.tail:
            n_tail += 1
            E.event_conservation(b)
        else:
            assert [h.four_momentum for h in a.final_state] == [h.four_momentum for h in b.final_state]
    assert n_tail >= 10


def test_the_runner_refuses_a_malformed_split(physics):
    E, catalog, fams, shape, flavor = physics
    base = E._params_at("sd_target", shape, flavor)
    for bad in ((2.0, 0.0, 4.0, "x"), (2.0, 0.0, 4.0, ""), (-1.0, 0.0, 4.0, "o"), (2.0, 0.0, 0.0, "o"), (2.0, 0.0, 4.0)):
        with pytest.raises(ValueError):
            _events(E, catalog, fams, dataclasses.replace(base, sd_two_sided=bad), "sd_target", 2)
