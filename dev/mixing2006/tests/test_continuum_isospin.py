"""The isospin completion of the continuum towers
(TASK-20261004-r8-isospin-complete-towers): the added families, their
charge-rotated classes, the KernelConfig option, the Clebsch-Gordan shares
it opens, and event conservation."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.cascade_decay import ISOTROPIC_ALIGNMENT_LABEL, KIND_CONTINUUM, CascadeState, _parse_class, _qn_of, enumerate_channels
from na49_mixing.continuum_isospin import SECTORS, isospin_partner_families
from na49_mixing.kernel_config import R74, KernelConfig, load_kernel
from na49_mixing.rng_substreams import derive_substreams
from na49_mixing.topology_runner import NOMINAL_PARAMS, run_topology

NESTED_ROOT = Path(__file__).resolve().parents[1]
ALL = ("baryonic", "strange_baryonic", "strange_mesonic")
EXPECTED = {
    "baryonic_np_qm1": ((1, -1, 0), 1.5), "baryonic_np_q0": ((1, 0, 0), 0.5), "baryonic_np_q2": ((1, 2, 0), 1.5),
    "baryonic_strange_qm1": ((1, -1, -1), 1.0), "baryonic_strange_q1": ((1, 1, -1), 1.0),
    "mesonic_strange_qm1": ((0, -1, -1), 0.5), "mesonic_antistrange_q1": ((0, 1, 1), 0.5),
}
#: the content pin of the added families' class lists (2026-10-04)
CLASSES_SHA256 = "b91146815af6db57d5e9ad35ea1dad62506046496b1225da35d4173830b4e7cd"


@pytest.fixture(scope="module")
def completed():
    return load_kernel(dataclasses.replace(R74, isospin_completion=ALL), NESTED_ROOT)


def _added(fams):
    return [f for f in fams if f.family_id in EXPECTED]


def test_the_added_families(completed):
    catalog, fams = completed
    added = _added(fams)
    assert [f.family_id for f in added] == list(EXPECTED)
    assert [f.family_id for f in fams[:7]] == ["baryonic_np", "baryonic_strange", "mesonic_neutral", "mesonic_charged",
                                               "mesonic_anticharged", "mesonic_strange", "mesonic_antistrange"]
    for f in added:
        qn, iso = EXPECTED[f.family_id]
        assert tuple(f.qn) == qn and f.isospin == iso and f.active is False and f.allowed_decay_classes
        # the run's options carry over from the source family
        assert (f.continuation_floor, f.continuation_ceiling, f.continuation_mass_law, f.continuation_barrier_l) == (1.7, 17.3, "density", 1)
        for cls in f.allowed_decay_classes:
            d0, d1 = _parse_class(cls)
            q0, q1 = _qn_of(catalog, d0), _qn_of(catalog, d1)
            assert (q0[0] + q1[0], q0[1] + q1[1], q0[2] + q1[2]) == qn
            assert d0 not in (310, 130) and d1 not in (310, 130)


def test_the_completion_is_deterministic_and_pinned(completed):
    catalog, fams = completed
    base = fams[:7]
    again = isospin_partner_families(base, catalog, ALL)
    assert again == _added(fams)
    blob = repr([(f.family_id, f.allowed_decay_classes) for f in again]).encode()
    digest = hashlib.sha256(blob).hexdigest()
    if CLASSES_SHA256 is not None:
        assert digest == CLASSES_SHA256


def test_sectors_select_and_nothing_is_exotic(completed):
    catalog, fams = completed
    base = fams[:7]
    assert [f.family_id for f in isospin_partner_families(base, catalog, ("baryonic",))] == \
        ["baryonic_np_qm1", "baryonic_np_q0", "baryonic_np_q2"]
    assert [f.family_id for f in isospin_partner_families(base, catalog, ("strange_mesonic",))] == \
        ["mesonic_strange_qm1", "mesonic_antistrange_q1"]
    charges = {sector: {q for _src, _bs, qs in SECTORS[sector] for q in qs} for sector in SECTORS}
    assert charges == {"baryonic": {-1, 0, 2}, "strange_baryonic": {-1, 1}, "strange_mesonic": {-1, 1}}
    # a declared charge state is never added twice
    assert isospin_partner_families(base + isospin_partner_families(base, catalog, ALL), catalog, ALL) == []


@pytest.mark.parametrize("bad", [("thermal",), ("baryonic", "baryonic"), "baryonic"])
def test_the_option_validates(bad):
    with pytest.raises(ValueError):
        KernelConfig(isospin_completion=bad)


def test_the_committed_kernel_is_unchanged():
    _c, plain = load_kernel(R74, NESTED_ROOT)
    _c2, completed = load_kernel(dataclasses.replace(R74, isospin_completion=ALL), NESTED_ROOT)
    assert completed[:7] == plain and len(plain) == 7


def test_the_clebsch_gordan_shares_of_a_p_star(completed):
    """A p* (I = 1/2) continuum state shedding a pion and continuing as a
    nucleon-like continuum: n* pi+ against p* pi0 in the ratio 2 : 1 (the
    shares 2/3 and 1/3), up to the small pi+/pi0 phase-space difference."""
    catalog, fams = completed
    state = CascadeState(momentum=(4.0, 0.0, 0.0, 0.0), mass=4.0, qn=(1, 1, 0), quantum={}, family_id="baryonic_np",
                         axis=(0.0, 0.0, 1.0), alignment_label=ISOTROPIC_ALIGNMENT_LABEL, parent=None,
                         exchange_ancestry=("isospin_test",), depth=0)
    weights = {tuple(c.daughters): c.weight for c in enumerate_channels(state, catalog, fams) if c.kind == KIND_CONTINUUM}
    n_pi_plus = next(w for d, w in weights.items() if set(d) == {"baryonic_np_q0", "catalog:211"})
    p_pi_zero = next(w for d, w in weights.items() if set(d) == {"baryonic_np", "catalog:111"})
    assert n_pi_plus / p_pi_zero == pytest.approx(2.0, rel=0.05)
    assert any("baryonic_np_q2" in d for d in weights)  # Delta++-like continuum with a pi-


@pytest.mark.parametrize("cls", ["sd_projectile", "dd", "nd"])
def test_events_conserve_with_the_complete_towers(completed, cls):
    catalog, fams = completed
    weights = ({"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.1, "high_mass_baryonic_continuum": 0.9}
               if cls == "nd" else dict(NOMINAL_PARAMS.remnant_weights))
    params = dataclasses.replace(NOMINAL_PARAMS, epsilon=0.63, epsilon_nd=0.23, remnant_weights=weights)
    for i in range(40):
        subs = derive_substreams(20261004, 13, 9_990_000_000 + i)
        rngs = {f"rng_{lab}": np.random.default_rng(subs.stream_for(lab)) for lab in
                ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")}
        event = run_topology(cls, params, catalog, fams, **rngs)
        assert tuple(sum(int(h.qn[j]) for h in event.final_state) for j in range(3)) == (2, 2, 0)


def test_the_isovector_meson_pairs():
    """The I = 1 charged mesonic continuum carries the charge-correlated
    baryon pairs (2026-10-05): M+ -> p nbar, M- -> n pbar; the other families
    and the default kernel are unchanged."""
    from na49_mixing.continuum_isospin import ISOVECTOR_PAIR_CLASSES, isovector_meson_pair_families
    base = dataclasses.replace(R74, isospin_completion=("baryonic",))
    _c, plain = load_kernel(base, NESTED_ROOT)
    catalog, paired = load_kernel(dataclasses.replace(base, isovector_meson_pairs=True), NESTED_ROOT)
    assert [f.family_id for f in paired] == [f.family_id for f in plain]
    for a, b in zip(plain, paired):
        extra = ISOVECTOR_PAIR_CLASSES.get(a.family_id, ())
        assert b.allowed_decay_classes == tuple(a.allowed_decay_classes) + extra
    assert isovector_meson_pair_families(paired) == paired  # idempotent
    state = CascadeState(momentum=(3.0, 0.0, 0.0, 0.0), mass=3.0, qn=(0, 1, 0), quantum={}, family_id="mesonic_charged",
                         axis=(0.0, 0.0, 1.0), alignment_label=ISOTROPIC_ALIGNMENT_LABEL, parent=None,
                         exchange_ancestry=("pair_test",), depth=0)
    pairs = [c for c in enumerate_channels(state, catalog, paired) if "catalog:-2112" in c.daughters]
    assert pairs and all(c.weight > 0 for c in pairs)
    assert not [c for c in enumerate_channels(state, catalog, plain) if "catalog:-2112" in c.daughters]
    with pytest.raises(ValueError):
        KernelConfig(isovector_meson_pairs=1)
    # the pairs continue into the baryonic families of every charge
    with pytest.raises(ValueError):
        dataclasses.replace(R74, isovector_meson_pairs=True)


def test_events_conserve_with_the_isovector_meson_pairs():
    catalog, fams = load_kernel(dataclasses.replace(R74, isospin_completion=ALL, delta_lines=True, isovector_meson_pairs=True),
                                NESTED_ROOT)
    params = dataclasses.replace(NOMINAL_PARAMS, epsilon=0.26, epsilon_nd=0.2, isovector_fraction=0.2,
                                 remnant_weights={"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.2,
                                                  "high_mass_baryonic_continuum": 0.8})
    for i in range(30):
        subs = derive_substreams(20261005, 13, 9_980_000_000 + i)
        rngs = {f"rng_{lab}": np.random.default_rng(subs.stream_for(lab)) for lab in
                ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")}
        event = run_topology("nd", params, catalog, fams, **rngs)
        assert tuple(sum(int(h.qn[j]) for h in event.final_state) for j in range(3)) == (2, 2, 0)
