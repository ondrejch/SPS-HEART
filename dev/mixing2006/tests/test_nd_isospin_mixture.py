"""The isospin mixture of the non-diffractive lines (2026-10-10):
``KernelConfig.non_diffractive_isospin_mixture`` marks the B != 0 families,
and the non-diffractive cascade
(``topology_non_diffractive.non_diffractive_families``) counts every charge
combination of their vertices equally, the equal mixture of I = 1/2 and
3/2.  The diffractive classes keep the isospin shares bit for bit, and the
option off is the committed kernel."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from na49_mixing import cascade_decay as C
from na49_mixing.kernel_config import KernelConfig, load_kernel
from na49_mixing.rng_substreams import derive_substreams
from na49_mixing.topology_non_diffractive import non_diffractive_families
from na49_mixing.topology_runner import NOMINAL_PARAMS, run_topology

NESTED_ROOT = Path(__file__).resolve().parents[1]
R89 = KernelConfig(**json.loads((NESTED_ROOT / "config" / "runs" / "r89.json").read_text())["kernel"])
LABELS = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")
PARAMS = dataclasses.replace(NOMINAL_PARAMS, exact_transfer_range=True, line_thresholds=True, nd_central=False)
# the R89 non-diffractive remnant split (w_excited 0.90): the excited lines cascade
PARAMS_ND = dataclasses.replace(PARAMS, remnant_weights={"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.1,
                                                         "high_mass_baryonic_continuum": 0.9})


@pytest.fixture(scope="module")
def kernels():
    return {name: load_kernel(dataclasses.replace(R89, non_diffractive_isospin_mixture=on), NESTED_ROOT)
            for name, on in (("off", False), ("on", True))}


def _state(fams, family_id, mass):
    family = C._family_by_id(fams, family_id)
    return C.CascadeState(momentum=(mass, 0.0, 0.0, 0.0), mass=mass, qn=tuple(family.qn), quantum={},
                          family_id=family_id, axis=(0.0, 0.0, 1.0), alignment_label=C.ISOTROPIC_ALIGNMENT_LABEL,
                          parent=None, exchange_ancestry=("nd_isospin_mixture",), depth=1)


def _events(class_label, cat, fams, n):
    out = []
    for i in range(n):
        subs = derive_substreams(20261010, 92, 9_920_000_000 + i)
        rngs = {f"rng_{lab}": np.random.default_rng(subs.stream_for(lab)) for lab in LABELS}
        params = PARAMS_ND if class_label == "nd" else PARAMS
        out.append(run_topology(class_label, params, cat, fams, **rngs).final_state)
    return out


def test_the_option_validates_and_marks_the_baryonic_families(kernels):
    assert R89.as_dict()["non_diffractive_isospin_mixture"] is False
    for bad in (1, "yes", None):
        with pytest.raises(ValueError, match="non_diffractive_isospin_mixture must be a bool"):
            dataclasses.replace(R89, non_diffractive_isospin_mixture=bad)
    cat_off, fams_off = kernels["off"]
    cat_on, fams_on = kernels["on"]
    assert not any(f.non_diffractive_isospin_mixture or f.isospin_mixture for f in fams_off)
    assert {str(f.family_id) for f in fams_on if f.non_diffractive_isospin_mixture} == \
        {str(f.family_id) for f in fams_on if int(f.qn[0]) != 0}
    assert any(int(f.qn[0]) == -1 for f in fams_on if f.non_diffractive_isospin_mixture)
    assert not any(f.isospin_mixture for f in fams_on)
    assert [dataclasses.replace(f, non_diffractive_isospin_mixture=False) for f in fams_on] == fams_off


def test_the_non_diffractive_view(kernels):
    _cat, fams_off = kernels["off"]
    _cat, fams_on = kernels["on"]
    assert non_diffractive_families(fams_off) is fams_off
    assert non_diffractive_families(None) is None
    view = non_diffractive_families(fams_on)
    assert non_diffractive_families(fams_on) is view and non_diffractive_families(view) is view
    assert [str(f.family_id) for f in view] == [str(f.family_id) for f in fams_on]
    for family, seen in zip(fams_on, view):
        assert seen.isospin_mixture == (int(family.qn[0]) != 0)
        assert dataclasses.replace(seen, isospin_mixture=False) == family


@pytest.mark.parametrize("mass", [1.25, 2.2, 4.0])
def test_a_mixture_vertex_counts_every_charge_combination_equally(kernels, mass, monkeypatch):
    """The vertex of a marked family in the non-diffractive view is the
    vertex with every isospin share set to one."""
    cat, fams_on = kernels["on"]
    view = non_diffractive_families(fams_on)
    for family_id in ("baryonic_np", "baryonic_np_q0", "anti_baryonic_np"):
        seen = C._enumerate_continuum_channels(_state(view, family_id, mass), cat, view, family_id)
        committed = C._enumerate_continuum_channels(_state(fams_on, family_id, mass), cat, fams_on, family_id)
        with monkeypatch.context() as m:
            m.setattr(C, "_isospin_factor_continuum", lambda *a, **k: 1.0)
            m.setattr(C, "_isospin_factor_direct", lambda *a, **k: 1.0)
            unit = C._enumerate_continuum_channels(_state(fams_on, family_id, mass), cat, fams_on, family_id)
        assert [(c.daughters, c.kind) for c in seen] == [(c.daughters, c.kind) for c in unit]
        assert [c.weight for c in seen] == pytest.approx([c.weight for c in unit], rel=1e-12)
        assert [c.weight for c in seen] != pytest.approx([c.weight for c in committed], rel=1e-6)


def test_the_nucleon_pion_pairs_of_a_charged_line(kernels):
    """At a Q = +1 baryonic vertex the isospin shares give n pi+ : p pi0 =
    2 : 1, and the mixture counts the two equally."""
    cat, fams_on = kernels["on"]
    view = non_diffractive_families(fams_on)
    ratios = {}
    for name, fams in (("shares", fams_on), ("mixture", view)):
        channels = C._enumerate_continuum_channels(_state(fams, "baryonic_np", 1.25), cat, fams, "baryonic_np")
        weight = {tuple(sorted(c.daughters)): c.weight for c in channels}
        ratios[name] = weight[tuple(sorted(("catalog:2112", "catalog:211")))] / weight[
            tuple(sorted(("catalog:2212", "catalog:111")))]
    assert ratios["shares"] / ratios["mixture"] == pytest.approx(2.0, rel=1e-12)


@pytest.mark.parametrize("class_label", ["sd_projectile", "sd_target", "dd"])
def test_the_diffractive_classes_keep_the_isospin_shares(kernels, class_label):
    cat_off, fams_off = kernels["off"]
    cat_on, fams_on = kernels["on"]
    assert _events(class_label, cat_on, fams_on, 40) == _events(class_label, cat_off, fams_off, 40)


def test_the_non_diffractive_events_change_and_conserve(kernels):
    cat_off, fams_off = kernels["off"]
    cat_on, fams_on = kernels["on"]
    on = _events("nd", cat_on, fams_on, 80)
    assert on != _events("nd", cat_off, fams_off, 80)
    known = {int(r.pdg_id): r for r in cat_on.records()}
    for finals in on:
        assert sum(int(known[int(h.pdg_id)].B) for h in finals if int(h.pdg_id) in known) == 2
        assert sum(int(known[int(h.pdg_id)].Q) for h in finals if int(h.pdg_id) in known) == 2


def test_the_r90_spec_is_r89_with_the_mixture_at_the_r89_optimum():
    runs = NESTED_ROOT / "config" / "runs"
    r89, r90 = (json.loads((runs / f"{r}.json").read_text()) for r in ("r89", "r90"))
    assert r90["kernel"] == {**r89["kernel"], "non_diffractive_isospin_mixture": True}
    for block in ("topology", "fit", "diffractive_split", "isovector"):
        assert r90[block] == r89[block]
    assert r90["generation_0"]["run"] == "r89_pairs_step_minimum_2026-10-10" and r90["generation_0"]["fit"] == "free"
    assert [p["name"] for p in r90["parameters"]] == [p["name"] for p in r89["parameters"]]
    assert r90["seeds"]["master"] == r89["seeds"]["master"]
    assert r90["seeds"]["raw_first"] > r89["seeds"]["raw_first"] + 511
    assert r90["seeds"]["partition_base"] != r89["seeds"]["partition_base"]
    assert KernelConfig(**r90["kernel"]).non_diffractive_isospin_mixture is True
