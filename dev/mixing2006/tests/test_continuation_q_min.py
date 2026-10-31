"""The minimum decay momentum of a continuation step (2026-10-10):
``KernelConfig.continuation_q_min`` opens a continuation channel only when
its sibling's decay momentum at the daughter's floor mass reaches the
value, so a state within reach of the floor decays into its direct pairs
(with the floor pools) instead of emitting a sibling at q* -> 0; off is the
committed rule bit for bit."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.cascade_decay import (CATALOG_FAMILY_PREFIX, ISOTROPIC_ALIGNMENT_LABEL, KIND_CONTINUUM, CascadeState,
                                       _continuation_floor, _daughter_pole_mass, _family_by_id, enumerate_channels)
from na49_mixing.kernel_config import R74, KernelConfig, load_kernel
from na49_mixing.resonance_widths import decay_momentum

NESTED_ROOT = Path(__file__).resolve().parents[1]
BASE = dataclasses.replace(R74, continuation_floor=2.0,
                           isospin_completion=("baryonic", "strange_baryonic", "strange_mesonic"), delta_lines=True,
                           three_body_phase_space=True, isoscalar_dipion=True, leading_split_fraction=0.2,
                           leading_split_kappa=4.0, floor_pools=("charged", "strange"), conjugate_towers=True)
ON = dataclasses.replace(BASE, continuation_q_min=0.25)


@pytest.fixture(scope="module")
def kernels():
    return load_kernel(BASE, NESTED_ROOT), load_kernel(ON, NESTED_ROOT)


def _state(family_id, qn, mass):
    return CascadeState(momentum=(float(mass), 0.0, 0.0, 0.0), mass=float(mass), qn=tuple(qn), quantum={},
                        family_id=str(family_id), axis=(0.0, 0.0, 1.0), alignment_label=ISOTROPIC_ALIGNMENT_LABEL,
                        parent=None, exchange_ancestry=("continuation_q_min_probe",), depth=1)


def _step_momentum(channel, catalog, fams, mass):
    sibling = next(d for d in channel.daughters if d.startswith(CATALOG_FAMILY_PREFIX))
    daughter = _family_by_id(fams, next(d for d in channel.daughters if not d.startswith(CATALOG_FAMILY_PREFIX)))
    m_sib = _daughter_pole_mass(catalog, int(sibling[len(CATALOG_FAMILY_PREFIX):]))
    return decay_momentum(mass, _continuation_floor(daughter, catalog), m_sib)


def test_the_option_validates_and_records():
    assert BASE.as_dict()["continuation_q_min"] is None and ON.as_dict()["continuation_q_min"] == 0.25
    assert KernelConfig(continuation_q_min=1).continuation_q_min == 1.0
    for bad in (-0.1, float("nan"), True, "0.25"):
        with pytest.raises(ValueError, match="continuation_q_min must be a non-negative finite momentum"):
            KernelConfig(continuation_q_min=bad)


def test_every_family_carries_the_option(kernels):
    (_c0, fams0), (_c1, fams1) = kernels
    assert all(f.continuation_q_min is None for f in fams0)
    assert all(f.continuation_q_min == 0.25 for f in fams1)
    assert [dataclasses.replace(f, continuation_q_min=None) for f in fams1] == fams0


@pytest.mark.parametrize("family_id", ["baryonic_np", "mesonic_neutral", "mesonic_charged", "anti_baryonic_np"])
@pytest.mark.parametrize("mass", [2.2, 2.4, 3.0, 4.5])
def test_the_slow_steps_close_and_the_open_ones_keep_their_weights(kernels, family_id, mass):
    (cat0, fams0), (cat1, fams1) = kernels
    qn = _family_by_id(fams0, family_id).qn
    off = enumerate_channels(_state(family_id, qn, mass), cat0, fams0)
    on = enumerate_channels(_state(family_id, qn, mass), cat1, fams1)
    cont_off = [c for c in off if c.kind == KIND_CONTINUUM]
    kept = [c for c in cont_off if _step_momentum(c, cat0, fams0, mass) >= 0.25]
    cont_on = [c for c in on if c.kind == KIND_CONTINUUM]
    assert [c.daughters for c in cont_on] == [c.daughters for c in kept]
    if kept:
        # the open continuation channels keep their relative weights
        t_off, t_on = sum(c.weight for c in kept), sum(c.weight for c in cont_on)
        for a, b in zip(kept, cont_on):
            assert b.weight / t_on == pytest.approx(a.weight / t_off, rel=1e-12)
    else:
        assert all(_step_momentum(c, cat0, fams0, mass) < 0.25 for c in cont_off)
        assert on and all(c.kind != KIND_CONTINUUM for c in on)


def test_a_state_within_reach_of_the_floor_becomes_a_floor_vertex(kernels):
    (cat0, fams0), (cat1, fams1) = kernels
    # a charged mesonic state at 2.2 GeV: the pion step reaches only 0.14 GeV/c at the 2.0 GeV floor
    st = _state("mesonic_charged", (0, 1, 0), 2.2)
    off = enumerate_channels(st, cat0, fams0)
    on = enumerate_channels(st, cat1, fams1)
    assert any(c.kind == KIND_CONTINUUM for c in off)
    assert not any(c.kind == KIND_CONTINUUM for c in on)
    floor_classes = set(_family_by_id(fams1, "mesonic_charged").floor_decay_classes)
    pools = {f"({c.daughters[0][len(CATALOG_FAMILY_PREFIX):]},{c.daughters[1][len(CATALOG_FAMILY_PREFIX):]})" for c in on}
    assert pools & floor_classes


def test_events_conserve_baryon_number_and_charge(kernels):
    from na49_mixing.rng_substreams import derive_substreams
    from na49_mixing.topology_runner import NOMINAL_PARAMS, run_topology
    _, (cat, fams) = kernels
    labels = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")
    params = dataclasses.replace(NOMINAL_PARAMS, exact_transfer_range=True, line_thresholds=True, nd_central=False)
    known = {int(r.pdg_id): r for r in cat.records()}
    for i in range(60):
        subs = derive_substreams(20261010, 90, 9_900_000_000 + i)
        rngs = {f"rng_{lab}": np.random.default_rng(subs.stream_for(lab)) for lab in labels}
        finals = run_topology("nd", params, cat, fams, **rngs).final_state
        assert sum(int(known[int(h.pdg_id)].B) for h in finals if int(h.pdg_id) in known) == 2
        assert sum(int(known[int(h.pdg_id)].Q) for h in finals if int(h.pdg_id) in known) == 2
