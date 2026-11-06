"""The symmetric non-diffractive remnant pair (``TopologyNominalParams.nd_symmetric_remnants``, 2026-11-04; the code
review of that date, finding M6): both lines draw on the projectile's window and a pair that leaves no neutral pion is
drawn again, so the two remnant masses share one law."""

from __future__ import annotations

import dataclasses
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.topology_non_diffractive import PI0_MASS, SQRT_S
from na49_mixing.topology_runner import TopologyNominalParams, run_topology

NESTED_ROOT = Path(__file__).resolve().parents[1]
LABELS = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")


@pytest.fixture(scope="module")
def setup():
    spec = importlib.util.spec_from_file_location("run_engine_sym_test", NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py")
    e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e)
    e.configure(NESTED_ROOT / "config" / "runs" / "r137.json")
    cat, fams = e._load_physics(e.FLOOR_GRID[0])
    params = dataclasses.replace(e._params_at("nd0", e.GEN0_SHAPE, e.GEN_FLAVOR), remnant_weights={
        "intact_proton": 0.0, "low_mass_baryonic_resonance": 0.0, "high_mass_baryonic_continuum": 1.0})
    return cat, fams, params


def _masses(setup, symmetric, n, seed):
    cat, fams, params = setup
    p = dataclasses.replace(params, nd_symmetric_remnants=symmetric)
    out = []
    for i in range(n):
        rngs = {f"rng_{lab}": np.random.default_rng([seed, i, k]) for k, lab in enumerate(LABELS)}
        ev = run_topology("nd", p, cat, fams, **rngs)
        out.append((float(ev.remnants[0].mass), float(ev.remnants[1].mass), ev.remnant_windows))
    return out


def test_the_default_is_the_ordered_draw():
    assert TopologyNominalParams.__dataclass_fields__["nd_symmetric_remnants"].default is False


def test_a_symmetric_pair_fits_and_records_the_full_windows(setup):
    for m1, m2, (w1, w2) in _masses(setup, True, 60, 3):
        assert m1 + m2 <= SQRT_S - PI0_MASS + 1e-12
        assert w1[1] == w2[1]                      # one upper edge for both lines
        assert w1[0] <= m1 <= w1[1] or m1 < w1[0]  # an intact or low-mass kind sits below the window


def test_the_two_lines_share_one_mass_law(setup):
    rows = _masses(setup, True, 1500, 5)
    m1 = np.array([r[0] for r in rows]); m2 = np.array([r[1] for r in rows])
    sigma = math.sqrt(m1.var() / m1.size + m2.var() / m2.size)
    assert abs(m1.mean() - m2.mean()) < 3.5 * sigma, (m1.mean(), m2.mean(), sigma)
