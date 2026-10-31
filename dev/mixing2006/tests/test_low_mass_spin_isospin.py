"""The low-mass record of a vacuum line under the isoscalar exchange
(2026-10-08): ``TopologyNominalParams.low_mass_spin_isospin`` draws the
record over the I = 1/2 records below the kernel boundary (the proton
included) with weight 2J + 1, by the same residual uniform; off, the draw
is the committed uniform pick over every (1, 1, 0) record."""

from __future__ import annotations

import collections
import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.kernel_config import R74, load_kernel
from na49_mixing.topology_remnants import (
    KIND_INTACT_PROTON,
    low_mass_records,
    low_mass_spin_isospin_records,
    select_remnant,
)
from na49_mixing.topology_runner import TopologyNominalParams, run_topology

NESTED_ROOT = Path(__file__).resolve().parents[1]
BASE = dataclasses.replace(R74, isospin_completion=("baryonic", "strange_baryonic", "strange_mesonic"), delta_lines=True,
                           three_body_phase_space=True)
LOW = {"intact_proton": 0.0, "low_mass_baryonic_resonance": 1.0, "high_mass_baryonic_continuum": 0.0}
LABELS = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")


class _Counting:
    def __init__(self, seed):
        self.rng, self.n = np.random.default_rng(seed), 0

    def random(self):
        self.n += 1
        return self.rng.random()


@pytest.fixture(scope="module")
def kernel():
    return load_kernel(BASE, NESTED_ROOT)


def _never(lo, hi):
    raise AssertionError("the low-mass kind takes no mass draw")


def test_the_set_and_its_weights(kernel):
    catalog, _fams = kernel
    pool = low_mass_spin_isospin_records(catalog)
    assert all(float(r.isospin) == 0.5 for r, _w in pool)
    assert {int(r.pdg_id) for r, _w in pool} == {int(r.pdg_id) for r in low_mass_records(catalog) if float(r.isospin) == 0.5}
    assert 2212 in {int(r.pdg_id) for r, _w in pool}
    assert all(w == 2.0 * float(r.spin) + 1.0 for r, w in pool)


def test_the_draw_follows_the_weights(kernel):
    """One uniform per selection; the record shares match 2J + 1 over the
    I = 1/2 set; no I = 3/2 record is drawn; off, the committed pick."""
    catalog, _fams = kernel
    pool = low_mass_spin_isospin_records(catalog)
    total = math.fsum(w for _r, w in pool)
    n = 6000
    on, off = collections.Counter(), collections.Counter()
    for seed in range(n):
        rng = _Counting(seed)
        rec = select_remnant(KIND_INTACT_PROTON, LOW, catalog, rng, _never, spin_isospin=True)
        assert rng.n == 1
        on[rec.pdg_id] += 1
        off[select_remnant(KIND_INTACT_PROTON, LOW, catalog, np.random.default_rng(seed), _never).pdg_id] += 1
    for record, weight in pool:
        p = weight / total
        assert abs(on[int(record.pdg_id)] / n - p) < 4.0 * math.sqrt(p * (1.0 - p) / n)
    assert not {2214, 202214, 122214} & set(on)
    assert {2214, 202214, 122214} <= set(off)
    # the default argument is the committed draw, uniform for uniform
    for seed in range(50):
        a = select_remnant(KIND_INTACT_PROTON, LOW, catalog, np.random.default_rng(seed), _never)
        b = select_remnant(KIND_INTACT_PROTON, LOW, catalog, np.random.default_rng(seed), _never, spin_isospin=False)
        assert a == b


@pytest.mark.parametrize("cls", ["sd_projectile", "nd"])
def test_the_runner_option(kernel, cls):
    """The vacuum lines of SD and ND draw no I = 3/2 low-mass record under
    the option; without it they do."""
    weights = LOW if cls == "nd" else {"intact_proton": 0.0, "low_mass_baryonic_resonance": 1.0,
                                       "high_mass_baryonic_continuum": 0.0}
    params = TopologyNominalParams(epsilon=0.4, b_sd=3.0, epsilon_nd=0.16, remnant_weights=weights, nd_central=False,
                                   exact_transfer_range=True)
    seen = {False: set(), True: set()}
    for flag in (False, True):
        p = dataclasses.replace(params, low_mass_spin_isospin=flag)
        for seed in range(60):
            rngs = {f"rng_{lab}": np.random.default_rng([seed, k]) for k, lab in enumerate(LABELS)}
            ev = run_topology(cls, p, *kernel, **rngs)
            remnants = ev.remnants if cls == "nd" else (ev.remnant,)
            seen[flag] |= {int(r.pdg_id) for r in remnants}
    assert seen[False] & {2214, 202214, 122214}
    assert not seen[True] & {2214, 202214, 122214}
