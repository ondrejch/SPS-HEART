"""The decays the NA49 tables count, performed in the observable (``na49_mixing.counted_decays``, 2026-11-04)."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest

from na49_mixing import counted_decays as CD
from na49_mixing.kernel_config import KernelConfig, load_kernel
from na49_mixing.topology_single_diffraction import _final_from_catalog

NESTED_ROOT = Path(__file__).resolve().parents[1]
R137 = KernelConfig(**json.loads((NESTED_ROOT / "config" / "runs" / "r137.json").read_text())["kernel"])


@pytest.fixture(scope="module")
def catalog():
    return load_kernel(dataclasses.replace(R137, omega_baryon=True), NESTED_ROOT)[0]


def _moving(catalog, pdg, p=(0.3, -0.2, 7.0)):
    m = float(catalog.get(pdg).pole_mass)
    return _final_from_catalog(catalog, pdg, (math.sqrt(m * m + sum(x * x for x in p)), *p), "projectile")


def test_the_table_holds_the_conjugates():
    table = CD.load_counted_decays(NESTED_ROOT)
    assert set(table) == {3212, -3212, 3322, -3322, 3312, -3312, 3334, -3334}
    assert table[-3312] == (((-3122, 211), 1.0),)
    assert table[3212] == (((3122, 22), 1.0),)
    assert table[-3334][-1][1] == pytest.approx(1.0)


@pytest.mark.parametrize("pdg", [3212, 3312, -3312, 3322, 3334, -3334])
def test_a_decay_conserves_and_ends_in_stable_hadrons(catalog, pdg):
    h = _moving(catalog, pdg)
    out = CD.counted_final_state([h], catalog, NESTED_ROOT)
    assert all(int(x.pdg_id) not in CD.load_counted_decays(NESTED_ROOT) for x in out)
    for j in range(4):
        assert math.fsum(float(x.four_momentum[j]) for x in out) == pytest.approx(float(h.four_momentum[j]), abs=1e-9)
    assert sum(int(x.qn[0]) for x in out) == int(h.qn[0]) and sum(int(x.qn[1]) for x in out) == int(h.qn[1])
    assert all(x.line == "projectile" for x in out)
    for x in out:
        e, px, py, pz = (float(v) for v in x.four_momentum)
        assert abs(e * e - px * px - py * py - pz * pz - float(x.mass) ** 2) < 1e-9 * max(e * e, 1.0)
    # deterministic: the same hadron decays the same way
    again = CD.counted_final_state([h], catalog, NESTED_ROOT)
    assert [x.four_momentum for x in again] == [x.four_momentum for x in out]


def test_other_hadrons_pass_unchanged(catalog):
    hs = [_moving(catalog, 2212), _moving(catalog, 3122), _moving(catalog, 211)]
    assert CD.counted_final_state(hs, catalog, NESTED_ROOT) == hs


def test_the_omega_branching(catalog):
    """Over many momenta the Omega- gives Lambda K- in 0.678 of its decays."""
    n, lam_k = 4000, 0
    for i in range(n):
        out = CD.counted_final_state([_moving(catalog, 3334, (0.0, 0.0, 1.0 + 0.001 * i))], catalog, NESTED_ROOT)
        lam_k += any(int(x.pdg_id) == -321 for x in out)
    assert lam_k / n == pytest.approx(0.678, abs=0.025)
