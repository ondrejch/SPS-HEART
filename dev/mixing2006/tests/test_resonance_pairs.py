"""The baryon-resonance pairs of the neutral mesonic continuum (2026-10-10):
``KernelConfig.baryon_resonance_pairs`` adds the pairs of equal isospin
among N, N*(1440), N(1520), N(1535) and Delta(1232) Deltabar to the neutral
mesonic family (``na49_mixing.resonance_pairs``), offered with the allowed
classes' weight law; the baryon-pair share of the continuation channels and
of the direct pairs each stays its value without them, and off is the
kernel with the charge-conjugate towers bit for bit."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.cascade_decay import (CATALOG_FAMILY_PREFIX, ISOTROPIC_ALIGNMENT_LABEL, KIND_CONTINUUM, CascadeState,
                                       enumerate_channels)
from na49_mixing.kernel_config import R74, KernelConfig, load_kernel
from na49_mixing.resonance_pairs import GROUPS, MULTIPLETS, PAIR_FAMILY, resonance_pair_classes

NESTED_ROOT = Path(__file__).resolve().parents[1]
TOWERS = dataclasses.replace(R74, continuation_floor=2.0,
                             isospin_completion=("baryonic", "strange_baryonic", "strange_mesonic"), delta_lines=True,
                             three_body_phase_space=True, isoscalar_dipion=True, leading_split_fraction=0.2,
                             leading_split_kappa=4.0, floor_pools=("charged", "strange"), conjugate_towers=True)
ON = dataclasses.replace(TOWERS, baryon_resonance_pairs=True)


@pytest.fixture(scope="module")
def kernels():
    return load_kernel(TOWERS, NESTED_ROOT), load_kernel(ON, NESTED_ROOT)


def _state(family_id, qn, mass):
    return CascadeState(momentum=(float(mass), 0.0, 0.0, 0.0), mass=float(mass), qn=tuple(qn), quantum={},
                        family_id=str(family_id), axis=(0.0, 0.0, 1.0), alignment_label=ISOTROPIC_ALIGNMENT_LABEL,
                        parent=None, exchange_ancestry=("resonance_pairs_probe",), depth=1)


def _abs_b(catalog, fams, daughter):
    if daughter.startswith(CATALOG_FAMILY_PREFIX):
        return abs(int(catalog.get(int(daughter[len(CATALOG_FAMILY_PREFIX):])).B))
    return abs(int(next(f for f in fams if f.family_id == daughter).qn[0]))


def test_the_option_validates_records_and_needs_the_towers():
    assert TOWERS.as_dict()["baryon_resonance_pairs"] is False and ON.as_dict()["baryon_resonance_pairs"] is True
    with pytest.raises(ValueError, match="baryon_resonance_pairs must be a bool"):
        KernelConfig(conjugate_towers=True, baryon_resonance_pairs=1)
    with pytest.raises(ValueError, match="need the charge-conjugate towers"):
        dataclasses.replace(TOWERS, conjugate_towers=False, baryon_resonance_pairs=True)


def test_off_keeps_the_towers_kernel_and_on_only_sets_the_neutral_pairs(kernels):
    (cat0, fams0), (cat1, fams1) = kernels
    assert cat1.records() == cat0.records()
    assert [f.family_id for f in fams1] == [f.family_id for f in fams0]
    for f0, f1 in zip(fams0, fams1):
        assert f0.resonance_pair_classes == ()
        if f1.family_id == PAIR_FAMILY:
            assert dataclasses.replace(f1, resonance_pair_classes=(), provenance=f0.provenance) == f0
        else:
            assert f1 == f0
    neutral = next(f for f in fams1 if f.family_id == PAIR_FAMILY)
    every = resonance_pair_classes(cat1)
    # 16 N-group multiplet pairs at two charges each, and Delta Deltabar at four
    assert len(every) == 2 * len(GROUPS[0]) ** 2 + len(MULTIPLETS["Delta(1232)"])
    assert neutral.resonance_pair_classes == tuple(c for c in every if c not in neutral.allowed_decay_classes)
    assert "(2212,-2212)" not in neutral.resonance_pair_classes and "(12212,-2214)" not in every
    assert {"(2216,-12212)", "(2224,-2224)", "(102112,-2112)"} <= set(neutral.resonance_pair_classes)


@pytest.mark.parametrize("mass", [3.1, 4.5, 6.0])
def test_the_vertex_holds_the_pair_share_of_each_channel_set(kernels, mass):
    (cat0, fams0), (cat1, fams1) = kernels
    st = _state(PAIR_FAMILY, (0, 0, 0), mass)
    off = enumerate_channels(st, cat0, fams0)
    on = enumerate_channels(st, cat1, fams1)
    off_by = {c.daughters: c for c in off}

    def pair(c, cat, fams):
        return sum(_abs_b(cat, fams, d) for d in c.daughters) >= 2

    new = [c for c in on if c.daughters not in off_by]
    assert new and all(pair(c, cat1, fams1) for c in new)
    # a resonance sibling beside a continuum daughter needs the floor plus its mass (N*(1440): about 3.4 GeV)
    assert any(c.kind != KIND_CONTINUUM for c in new)
    assert any(c.kind == KIND_CONTINUUM for c in new) == (mass >= 3.5)
    for continuing in (True, False):
        set_off = [c for c in off if (c.kind == KIND_CONTINUUM) == continuing]
        set_on = [c for c in on if (c.kind == KIND_CONTINUUM) == continuing]
        t_off, t_on = math.fsum(c.weight for c in set_off), math.fsum(c.weight for c in set_on)
        s_off = math.fsum(c.weight for c in set_off if pair(c, cat0, fams0)) / t_off
        s_on = math.fsum(c.weight for c in set_on if pair(c, cat1, fams1)) / t_on
        assert s_on == pytest.approx(s_off, rel=1e-12, abs=1e-15)
        # the channels that are not baryon pairs keep their relative weights within the set
        ratios = {c.weight / t_on / (off_by[c.daughters].weight / t_off) for c in set_on
                  if not pair(c, cat1, fams1)}
        assert max(ratios) == pytest.approx(min(ratios), rel=1e-12)


def test_the_charged_and_baryonic_vertices_do_not_move(kernels):
    (cat0, fams0), (cat1, fams1) = kernels
    for family_id, qn in (("mesonic_charged", (0, 1, 0)), ("baryonic_np", (1, 1, 0)), ("anti_baryonic_np", (-1, -1, 0))):
        st = _state(family_id, qn, 4.0)
        assert enumerate_channels(st, cat1, fams1) == enumerate_channels(st, cat0, fams0)


def test_events_conserve_baryon_number_and_charge(kernels):
    from na49_mixing.rng_substreams import derive_substreams
    from na49_mixing.topology_runner import NOMINAL_PARAMS, run_topology
    _, (cat, fams) = kernels
    labels = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")
    params = dataclasses.replace(NOMINAL_PARAMS, exact_transfer_range=True, line_thresholds=True, nd_central=False)
    known = {int(r.pdg_id): r for r in cat.records()}
    for i in range(60):
        subs = derive_substreams(20261010, 89, 9_800_000_000 + i)
        rngs = {f"rng_{lab}": np.random.default_rng(subs.stream_for(lab)) for lab in labels}
        finals = run_topology("nd", params, cat, fams, **rngs).final_state
        assert sum(int(known[int(h.pdg_id)].B) for h in finals if int(h.pdg_id) in known) == 2
        assert sum(int(known[int(h.pdg_id)].Q) for h in finals if int(h.pdg_id) in known) == 2


def test_the_r89_spec_is_r88_with_the_pairs_and_the_step_minimum():
    import json
    runs = NESTED_ROOT / "config" / "runs"
    r88, r89 = (json.loads((runs / f"{r}.json").read_text()) for r in ("r88", "r89"))
    assert r89["kernel"] == {**r88["kernel"], "baryon_resonance_pairs": True, "continuation_q_min": 0.25}
    assert r89["topology"] == r88["topology"] and r89["topology"]["transfer_slope_mass_exponent"] == 1.25
    assert r89["fit"] == r88["fit"] and r89["diffractive_split"] == r88["diffractive_split"]
    assert r89["generation_0"]["run"] == "r88_conjugate_towers_2026-10-09" and "judgment" not in r89["generation_0"]
    assert {p["name"] for p in r89["parameters"]} == {p["name"] for p in r88["parameters"]}
    config = KernelConfig(**r89["kernel"])
    assert config.baryon_resonance_pairs is True and config.continuation_q_min == 0.25
