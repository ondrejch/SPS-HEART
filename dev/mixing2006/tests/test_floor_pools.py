"""The floor pools of the charged and strange mesonic continuum (2026-10-09):
``KernelConfig.floor_pools`` sets ``ContinuumFamily.floor_decay_classes``
on the six mesonic families (``na49_mixing.floor_pools``), whose pairs join
the direct pairs of a vertex with no open continuation channel; a vertex
that continues keeps the committed channels, and off is the committed
kernel."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest

import na49_mixing.floor_pools as FP
from na49_mixing.cascade_decay import (CATALOG_FAMILY_PREFIX, ISOTROPIC_ALIGNMENT_LABEL, KIND_CONTINUUM, CascadeState,
                                       _daughter_pole_mass, _isospin_factor_direct, _parse_class, _spin_factor,
                                       enumerate_channels)
from na49_mixing.kernel_config import R74, KernelConfig, load_kernel
from na49_mixing.resonance_widths import decay_momentum

NESTED_ROOT = Path(__file__).resolve().parents[1]
BASE = dataclasses.replace(R74, continuation_floor=2.0, isospin_completion=("baryonic", "strange_baryonic", "strange_mesonic"),
                           delta_lines=True, three_body_phase_space=True, isoscalar_dipion=True)
ON = dataclasses.replace(BASE, floor_pools=("charged", "strange"))
POOL_FAMILIES = ("mesonic_charged", "mesonic_anticharged", "mesonic_strange", "mesonic_antistrange", "mesonic_strange_qm1",
                 "mesonic_antistrange_q1")


@pytest.fixture(scope="module")
def kernels():
    return load_kernel(BASE, NESTED_ROOT), load_kernel(ON, NESTED_ROOT)


def _state(family, mass):
    return CascadeState(momentum=(float(mass), 0.0, 0.0, 0.0), mass=float(mass), qn=tuple(family.qn), quantum={},
                        family_id=str(family.family_id), axis=(0.0, 0.0, 1.0), alignment_label=ISOTROPIC_ALIGNMENT_LABEL,
                        parent=None, exchange_ancestry=("floor_pool_probe",), depth=1)


def _by_id(fams):
    return {str(f.family_id): f for f in fams}


def test_off_is_the_committed_kernel(kernels):
    (_, fams_off), _ = kernels
    assert all(f.floor_decay_classes == () for f in fams_off)
    assert BASE.as_dict()["floor_pools"] == [] and ON.as_dict()["floor_pools"] == ["charged", "strange"]


@pytest.mark.parametrize("pools, isospin, message", [
    ("charged", ("strange_mesonic",), "sequence"),
    (("charged", "charged"), ("strange_mesonic",), "distinct"),
    (("neutral",), ("strange_mesonic",), "distinct"),
    (("strange",), (), "strange_mesonic"),
])
def test_the_option_validates(pools, isospin, message):
    with pytest.raises(ValueError, match=message):
        KernelConfig(isospin_completion=isospin, floor_pools=pools)


def test_the_six_families_carry_their_floor_classes(kernels):
    (_, fams_off), (catalog, fams_on) = kernels
    off, on = _by_id(fams_off), _by_id(fams_on)
    assert set(off) == set(on)
    for family_id, family in on.items():
        if family_id not in POOL_FAMILIES:
            assert family == off[family_id]
            continue
        assert family.allowed_decay_classes == off[family_id].allowed_decay_classes
        assert len(family.floor_decay_classes) in (11, 16)
        assert not set(family.floor_decay_classes) & set(family.allowed_decay_classes)
        assert family.provenance[-1].startswith("floor pool: the direct pairs")
        assert dataclasses.replace(family, floor_decay_classes=(), provenance=family.provenance[:-1]) == off[family_id]
    assert "(213,223)" in on["mesonic_charged"].floor_decay_classes
    assert "(-213,223)" in on["mesonic_anticharged"].floor_decay_classes
    assert "(323,-213)" in on["mesonic_antistrange"].floor_decay_classes


def test_a_floor_vertex_takes_the_heavier_pairs(kernels):
    (_, fams_off), (catalog, fams_on) = kernels
    family = _by_id(fams_on)["mesonic_charged"]
    mass = 1.9  # below the 2.0 GeV floor plus the lightest sibling: no continuation channel
    off = enumerate_channels(_state(family, mass), catalog, fams_off)
    on = enumerate_channels(_state(family, mass), catalog, fams_on)
    assert off and not any(c.kind == KIND_CONTINUUM for c in on)
    assert math.isclose(math.fsum(c.weight for c in on), 1.0, rel_tol=1e-12)
    raw = {}
    for class_id in family.allowed_decay_classes + family.floor_decay_classes:
        d0, d1 = _parse_class(class_id)
        m0, m1 = _daughter_pole_mass(catalog, d0), _daughter_pole_mass(catalog, d1)
        if mass >= m0 + m1:
            iso = 1.0 if family.isospin is None else _isospin_factor_direct(float(family.isospin), 2, d0, d1, catalog)
            if iso > 0.0:
                raw[(f"{CATALOG_FAMILY_PREFIX}{d0}", f"{CATALOG_FAMILY_PREFIX}{d1}")] = (
                    _spin_factor(catalog, d0) * _spin_factor(catalog, d1) * decay_momentum(mass, m0, m1) * iso)
    total = math.fsum(raw.values())
    assert {c.daughters: c.weight for c in on} == pytest.approx({k: v / total for k, v in raw.items()}, rel=1e-12)
    assert ("catalog:213", "catalog:223") in {c.daughters for c in on}
    assert len(on) > len(off)


@pytest.mark.parametrize("family_id", POOL_FAMILIES)
def test_a_continuing_vertex_keeps_the_committed_channels(kernels, family_id):
    (_, fams_off), (catalog, fams_on) = kernels
    family = _by_id(fams_on)[family_id]
    state = _state(family, 3.0)
    off = enumerate_channels(state, catalog, fams_off)
    on = enumerate_channels(state, catalog, fams_on)
    assert any(c.kind == KIND_CONTINUUM for c in on)
    assert on == off


def test_the_pools_are_refused_when_inconsistent(kernels, monkeypatch):
    (_, fams_off), (catalog, _) = kernels
    monkeypatch.setitem(FP.FLOOR_CLASSES, "charged", {"mesonic_charged": ("(213,-211)",)})
    with pytest.raises(ValueError, match="does not carry"):
        FP.floor_pool_families(fams_off, catalog, ("charged",))
    no_strange_partners = [f for f in fams_off if str(f.family_id) != "mesonic_strange_qm1"]
    with pytest.raises(ValueError, match="absent families"):
        FP.floor_pool_families(no_strange_partners, catalog, ("strange",))
    with pytest.raises(ValueError, match="unknown sectors"):
        FP.floor_pool_families(fams_off, catalog, ("neutral",))
