"""The charge-conjugate baryonic towers (2026-10-09):
``KernelConfig.conjugate_towers`` appends the conjugate of every baryonic
family, rebuilds every antibaryon record as the conjugate of its baryon,
and gives the two conjugate baryon-pair continuations of a mesonic vertex
half the weight each (``na49_mixing.conjugate_towers``); an antibaryonic
vertex mirrors its baryonic partner channel for channel, and off is the
committed kernel."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.cascade_decay import (CATALOG_FAMILY_PREFIX, CONJUGATE_FAMILY_PREFIX, ISOTROPIC_ALIGNMENT_LABEL,
                                       KIND_CONTINUUM, CascadeState, enumerate_channels)
from na49_mixing.conjugate_towers import charge_conjugate
from na49_mixing.flavor_weights import FlavorParams, flavor_exponents
from na49_mixing.kernel_config import R74, KernelConfig, load_kernel

NESTED_ROOT = Path(__file__).resolve().parents[1]
BASE = dataclasses.replace(R74, continuation_floor=2.0, isospin_completion=("baryonic", "strange_baryonic", "strange_mesonic"),
                           delta_lines=True, three_body_phase_space=True, isoscalar_dipion=True,
                           leading_split_fraction=0.2, leading_split_kappa=4.0, floor_pools=("charged", "strange"))
ON = dataclasses.replace(BASE, conjugate_towers=True)
FLAVOR = FlavorParams(gamma_s_had=0.284, r_v=1.0, r_bbar=1.587, assoc_norm=2.23)


@pytest.fixture(scope="module")
def kernels():
    return load_kernel(BASE, NESTED_ROOT), load_kernel(ON, NESTED_ROOT)


def _state(family_id, qn, mass, depth=1):
    return CascadeState(momentum=(float(mass), 0.0, 0.0, 0.0), mass=float(mass), qn=tuple(qn), quantum={},
                        family_id=str(family_id), axis=(0.0, 0.0, 1.0), alignment_label=ISOTROPIC_ALIGNMENT_LABEL,
                        parent=None, exchange_ancestry=("conjugate_towers_probe",), depth=depth)


def _baryon_number(catalog, fams, daughter):
    if daughter.startswith(CATALOG_FAMILY_PREFIX):
        return int(catalog.get(int(daughter[len(CATALOG_FAMILY_PREFIX):])).B)
    return int(next(f for f in fams if f.family_id == daughter).qn[0])


def _conjugate_daughter(daughter, catalog, fams):
    """The conjugate of a channel daughter: the conjugate catalog record, or
    the first family (config order, the kernel's daughter rule) with the
    negated quantum numbers."""
    if daughter.startswith(CATALOG_FAMILY_PREFIX):
        return f"{CATALOG_FAMILY_PREFIX}{charge_conjugate(int(daughter[len(CATALOG_FAMILY_PREFIX):]), catalog)}"
    qn = tuple(-int(v) for v in next(f for f in fams if f.family_id == daughter).qn)
    return next(f.family_id for f in fams if tuple(int(v) for v in f.qn) == qn)


def test_the_option_validates_and_records():
    assert BASE.as_dict()["conjugate_towers"] is False and ON.as_dict()["conjugate_towers"] is True
    with pytest.raises(ValueError, match="conjugate_towers must be a bool"):
        KernelConfig(conjugate_towers=1)


def test_off_keeps_the_committed_kernel_and_on_only_adds(kernels):
    (cat0, fams0), (cat1, fams1) = kernels
    assert not any(f.family_id.startswith(CONJUGATE_FAMILY_PREFIX) for f in fams0)
    assert fams1[:len(fams0)] == fams0
    added = fams1[len(fams0):]
    assert [f.family_id for f in added] == [CONJUGATE_FAMILY_PREFIX + f.family_id for f in fams0 if f.qn[0] == 1]
    for record in cat0.records():
        if int(record.B) >= 0:
            assert cat1.get(int(record.pdg_id)) == record


def test_every_antibaryon_is_the_conjugate_of_its_baryon(kernels):
    _, (cat, _fams) = kernels
    baryons = [r for r in cat.records() if int(r.B) == 1]
    assert len([r for r in cat.records() if int(r.B) == -1]) == len(baryons)
    for b in baryons:
        a = cat.get(-int(b.pdg_id))
        assert (a.B, a.Q, a.S, a.isospin3) == (-b.B, -b.Q, -b.S, -b.isospin3)
        assert (a.isospin, a.spin, a.parity, a.pole_mass, a.nominal_width, a.mass_range, a.stable_for_decay) == (
            b.isospin, b.spin, b.parity, b.pole_mass, b.nominal_width, b.mass_range, b.stable_for_decay)
        assert len(a.decay_channels) == len(b.decay_channels)
        for ca, cb in zip(a.decay_channels, b.decay_channels):
            assert ca.parent_pdg == -int(b.pdg_id)
            assert ca.daughters == tuple(charge_conjugate(d, cat) for d in cb.daughters)
            assert (ca.branching_fraction, ca.orbital_angular_momentum, ca.is_three_body) == (
                cb.branching_fraction, cb.orbital_angular_momentum, cb.is_three_body)
    report = cat.validate()
    assert not getattr(report, "errors", ())
    # the neutral non-strange mesons and the weak neutral kaons are their own conjugates
    for pdg in (111, 113, 221, 223, 225, 333, 310, 130):
        assert charge_conjugate(pdg, cat) == pdg
    for pdg in (211, 321, 311, 2212, 3122, 213, 323):
        assert charge_conjugate(pdg, cat) == -pdg


def test_the_conjugate_families_mirror_the_baryonic_ones(kernels):
    _, (cat, fams) = kernels
    by_id = {f.family_id: f for f in fams}
    for f in fams:
        if not f.family_id.startswith(CONJUGATE_FAMILY_PREFIX):
            continue
        b = by_id[f.family_id[len(CONJUGATE_FAMILY_PREFIX):]]
        assert tuple(f.qn) == tuple(-v for v in b.qn)
        for ca, cb in zip(f.allowed_decay_classes, b.allowed_decay_classes):
            d0, d1 = (int(x) for x in cb.strip("()").split(","))
            assert ca == f"({charge_conjugate(d0, cat)},{charge_conjugate(d1, cat)})"
        assert len(f.allowed_decay_classes) == len(b.allowed_decay_classes)
        same = {k: v for k, v in dataclasses.asdict(f).items() if k not in ("family_id", "qn", "allowed_decay_classes",
                                                                          "floor_decay_classes")}
        assert same == {k: v for k, v in dataclasses.asdict(b).items() if k in same}


@pytest.mark.parametrize("mass", [2.6, 3.0, 4.0, 6.0])
def test_the_mesonic_vertex_keeps_its_baryon_pair_share(kernels, mass):
    (cat0, fams0), (cat1, fams1) = kernels
    st = _state("mesonic_neutral", (0, 0, 0), mass)
    off = {c.daughters: c for c in enumerate_channels(st, cat0, fams0, FLAVOR)}
    on = {c.daughters: c for c in enumerate_channels(st, cat1, fams1, FLAVOR)}

    def pair(c, cat, fams):
        return sum(abs(_baryon_number(cat, fams, d)) for d in c.daughters) >= 2

    for subset in (True, False):
        s_off = math.fsum(c.weight for c in off.values() if pair(c, cat0, fams0) and (c.kind == KIND_CONTINUUM) == subset)
        s_on = math.fsum(c.weight for c in on.values() if pair(c, cat1, fams1) and (c.kind == KIND_CONTINUUM) == subset)
        assert s_on == pytest.approx(s_off, rel=1e-12, abs=1e-15)
    for daughters, c in off.items():
        if pair(c, cat0, fams0) and c.kind == KIND_CONTINUUM:
            partner = tuple(sorted(_conjugate_daughter(d, cat1, fams1) for d in daughters))
            mirror = next(o for d, o in on.items() if tuple(sorted(d)) == partner)
            assert on[daughters].weight == pytest.approx(0.5 * c.weight, rel=1e-12)
            assert mirror.weight == pytest.approx(0.5 * c.weight, rel=1e-12)
        else:
            assert on[daughters].weight == pytest.approx(c.weight, rel=1e-12)
    assert any(pair(c, cat1, fams1) and c.kind == KIND_CONTINUUM for c in on.values()) == (mass >= 3.0)


@pytest.mark.parametrize("family_id", ["baryonic_np", "baryonic_np_q0", "baryonic_strange", "baryonic_np_q2"])
@pytest.mark.parametrize("mass", [2.6, 3.5])
def test_the_antibaryonic_vertex_mirrors_its_partner(kernels, family_id, mass):
    _, (cat, fams) = kernels
    b = next(f for f in fams if f.family_id == family_id)
    chans_b = enumerate_channels(_state(family_id, b.qn, mass), cat, fams, FLAVOR)
    chans_a = enumerate_channels(_state(CONJUGATE_FAMILY_PREFIX + family_id, tuple(-v for v in b.qn), mass), cat, fams,
                                 FLAVOR)
    assert len(chans_a) == len(chans_b)
    for ca, cb in zip(chans_a, chans_b):
        assert ca.daughters == tuple(_conjugate_daughter(d, cat, fams) for d in cb.daughters)
        assert ca.kind == cb.kind
        assert ca.weight == pytest.approx(cb.weight, rel=1e-12)
        assert flavor_exponents(tuple(-v for v in b.qn), ca, cat, fams) == flavor_exponents(tuple(b.qn), cb, cat, fams)


def test_associated_production_counts_at_the_antibaryonic_vertex(kernels):
    _, (cat, fams) = kernels
    b = next(f for f in fams if f.family_id == "baryonic_np")
    st = _state("baryonic_np", b.qn, 3.0)
    assoc = [c for c in enumerate_channels(st, cat, fams) if f"{CATALOG_FAMILY_PREFIX}321" in c.daughters
             and any(d.startswith(CATALOG_FAMILY_PREFIX + "3") for d in c.daughters)]
    assert assoc
    for c in assoc:
        mirror = dataclasses.replace(c, daughters=tuple(_conjugate_daughter(d, cat, fams) for d in c.daughters))
        assert flavor_exponents(tuple(b.qn), c, cat, fams).n_assoc == 1
        assert flavor_exponents(tuple(-v for v in b.qn), mirror, cat, fams).n_assoc == 1


def test_events_conserve_baryon_number_and_charge(kernels):
    from na49_mixing.rng_substreams import derive_substreams
    from na49_mixing.topology_runner import NOMINAL_PARAMS, run_topology
    _, (cat, fams) = kernels
    labels = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")
    params = dataclasses.replace(NOMINAL_PARAMS, exact_transfer_range=True, line_thresholds=True, nd_central=False)
    known = {int(r.pdg_id): r for r in cat.records()}
    for i in range(60):
        subs = derive_substreams(20261009, 88, 9_700_000_000 + i)
        rngs = {f"rng_{lab}": np.random.default_rng(subs.stream_for(lab)) for lab in labels}
        finals = run_topology("nd", params, cat, fams, **rngs).final_state
        assert sum(int(known[int(h.pdg_id)].B) for h in finals if int(h.pdg_id) in known) == 2
        assert sum(int(known[int(h.pdg_id)].Q) for h in finals if int(h.pdg_id) in known) == 2


def test_the_r88_spec_is_r87_with_the_towers_and_the_slope_exponent():
    import json
    runs = NESTED_ROOT / "config" / "runs"
    r87, r88 = (json.loads((runs / f"{r}.json").read_text()) for r in ("r87", "r88"))
    assert r88["kernel"] == {**r87["kernel"], "conjugate_towers": True}
    assert r88["topology"] == {**r87["topology"], "transfer_slope_mass_exponent": 1.25}
    assert r87["topology"]["transfer_slope_mass_exponent"] == 0.75
    assert r88["fit"] == r87["fit"] and r88["diffractive_split"] == r87["diffractive_split"]
    gen0 = {p["name"]: p["gen0"] for p in r88["parameters"]}
    assert gen0["b_sd"] == 6.3 and "b_sd" not in r88["generation_0"]["from_fit"]
    assert "b_sd" in r88["generation_0"]["judgment"]
    assert KernelConfig(**r88["kernel"]).conjugate_towers is True
