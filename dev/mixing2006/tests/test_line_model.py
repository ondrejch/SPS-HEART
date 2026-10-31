"""The general line model (TASK-20261004-refactor-run-engine-kernel-config,
P3): an excited line's root carries the line's (B, Q, S) and starts in the
continuum family of those quantum numbers.  The vacuum exchange keeps every
line at (1, 1, 0) (baryonic_np), bit for bit; a line of other quantum
numbers cascades in its own family (the base of the Round 8 isovector
exchange)."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.kernel_config import R74, load_kernel
from na49_mixing.rng_substreams import derive_substreams
from na49_mixing.topology_double_diffraction import _decay_system
from na49_mixing.topology_single_diffraction import _decay_excitation

NESTED_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def kernel():
    return load_kernel(R74, NESTED_ROOT)


def _rngs(i):
    subs = derive_substreams(20261004, 11, 9_950_000_000 + i)
    return (np.random.default_rng(subs.stream_for("decay_channel")), np.random.default_rng(subs.stream_for("decay_kinematics")))


def _p4(mass, pz=3.0, px=0.2, py=-0.1):
    return (math.sqrt(mass * mass + px * px + py * py + pz * pz), px, py, pz)


def _finals(hadrons):
    return [(h.pdg_id, h.four_momentum, h.line, h.qn) for h in hadrons]


@pytest.mark.parametrize("mass", [1.3, 2.4, 6.0])
def test_the_default_line_is_explicit_proton_quantum_numbers(kernel, mass):
    catalog, fams = kernel
    for i in range(10):
        default = _decay_excitation(_p4(mass), mass, catalog, fams, *_rngs(i))
        explicit = _decay_excitation(_p4(mass), mass, catalog, fams, *_rngs(i), line_qn=(1, 1, 0))
        assert _finals(default) == _finals(explicit)
        default = _decay_system(_p4(mass), mass, catalog, fams, *_rngs(i), "projectile")
        explicit = _decay_system(_p4(mass), mass, catalog, fams, *_rngs(i), "projectile", line_qn=(1, 1, 0))
        assert _finals(default) == _finals(explicit)


def test_a_line_without_a_family_is_refused(kernel):
    catalog, fams = kernel
    with pytest.raises(ValueError):
        _decay_excitation(_p4(3.0), 3.0, catalog, fams, *_rngs(0), line_qn=(1, 2, 0))
    with pytest.raises(ValueError):
        _decay_system(_p4(3.0), 3.0, catalog, fams, *_rngs(0), "target", line_qn=(1, 2, 0))


def test_a_neutral_line_cascades_in_its_own_family(kernel):
    """A (1, 0, 0) line on an in-memory neutral baryonic family (two
    charge-rotated classes): every final state carries B = 1 and Q = 0."""
    catalog, fams = kernel
    base = next(f for f in fams if f.family_id == "baryonic_np")
    neutral = dataclasses.replace(base, family_id="test_baryonic_n0", qn=(1, 0, 0),
                                  allowed_decay_classes=("(2112,111)", "(2212,-211)"), isospin=0.5, active=False)
    fams_n = list(fams) + [neutral]
    charges = set()
    for i in range(30):
        mass = 1.4 + 0.15 * i
        finals = _decay_excitation(_p4(mass), mass, catalog, fams_n, *_rngs(i), line_qn=(1, 0, 0))
        assert sum(int(h.qn[0]) for h in finals) == 1 and sum(int(h.qn[1]) for h in finals) == 0
        charges |= {int(h.pdg_id) for h in finals if h.pdg_id is not None}
    assert 2212 in charges and -211 in charges
