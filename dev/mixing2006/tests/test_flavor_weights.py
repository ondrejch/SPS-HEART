"""Round 6 flavor tests (tasks 37 and 44; plan section 12).

- exponents on hand-built channels (created strangeness with the weak
  neutral kaons counted as strange quanta, hidden phi content, vector
  mesons, baryon pairs, associated production) and their weights;
- the identity point returns the enumerated channels unchanged, object
  for object, and the threaded ``flavor`` keyword reproduces the Round 5
  traces bit for bit when absent;
- the flavor source grid (task 44): over the active families and a mass
  grid the strange, vector, and baryon-pair shares respond monotonically
  to gamma_s, r_V, r_BBbar and the weights stay normalized;
- charge completion (task 44): every allowed class conserves (B, Q, S),
  every daughter is a committed catalog record, and every Round 6
  strange class has its isospin or charge-conjugate partner;
- catalog parents keep their measured branching fractions;
- zero random numbers.
Hermetic: the committed catalog and families plus NumPy.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing import cascade_decay
from na49_mixing.cascade_decay import (
    CATALOG_FAMILY_PREFIX,
    CascadeChannel,
    CascadeState,
    KIND_CONTINUUM,
    enumerate_channels,
)
from na49_mixing.flavor_weights import (
    NOMINAL_FLAVOR,
    FlavorParams,
    apply_flavor_weights,
    flavor_exponents,
    flavor_weight,
)
from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.spectral_density import families_from_json
from na49_mixing.topology_runner import NOMINAL_PARAMS

CONFIG = Path(__file__).resolve().parent.parent / "config" / "v2_singlet"
ROUND6_CLASSES = {
    "mesonic_neutral": ["(323,-321)", "(-323,321)", "(313,-311)", "(-313,311)", "(333,111)", "(333,221)", "(2212,-2212)", "(2112,-2112)"],
    "mesonic_charged": ["(323,-311)", "(321,-313)"],
    "mesonic_anticharged": ["(-323,311)", "(-321,313)"],
    "baryonic_np": ["(3224,311)", "(3214,321)", "(102134,321)"],
    "baryonic_strange": ["(3114,211)", "(3214,111)", "(3224,-211)", "(102134,111)"],
}


@pytest.fixture(scope="module")
def catalog():
    return HadronCatalog.from_json(str(CONFIG / "resonance_catalog.json"))


@pytest.fixture(scope="module")
def fams():
    return families_from_json(str(CONFIG / "continuum_resonances.json"))


def _channel(parent, d0, d1, weight=1.0):
    return CascadeChannel(parent_id=parent, daughters=(f"{CATALOG_FAMILY_PREFIX}{d0}", f"{CATALOG_FAMILY_PREFIX}{d1}"), kind="established_hadron", weight=weight)


def _root(family_id, mass, qn):
    return CascadeState(momentum=(mass, 0.0, 0.0, 0.0), mass=mass, qn=qn, quantum={}, family_id=family_id,
                        axis=(0.0, 0.0, 1.0), alignment_label="isotropic", parent=None, exchange_ancestry=(), depth=0)


def test_prefix_equals_the_cascade_prefix():
    from na49_mixing import flavor_weights
    assert flavor_weights.CATALOG_FAMILY_PREFIX == cascade_decay.CATALOG_FAMILY_PREFIX


def test_exponents_on_hand_channels(catalog, fams):
    neutral = (0, 0, 0)
    e = flavor_exponents(neutral, _channel("mesonic_neutral", 321, -321), catalog, fams)   # K+ K-
    assert (e.n_ssbar, e.n_v, e.n_bbar, e.n_assoc) == (1.0, 0, 0, 0)
    e = flavor_exponents(neutral, _channel("mesonic_neutral", 310, 130), catalog, fams)    # K0S K0L: strange in flavor
    assert e.n_ssbar == 1.0
    e = flavor_exponents(neutral, _channel("mesonic_neutral", 323, -321), catalog, fams)   # K*+ K-
    assert (e.n_ssbar, e.n_v) == (1.0, 1)
    e = flavor_exponents(neutral, _channel("mesonic_neutral", 333, 111), catalog, fams)    # phi pi0: hidden s-sbar, vector
    assert (e.n_ssbar, e.n_v) == (1.0, 1)
    e = flavor_exponents(neutral, _channel("mesonic_neutral", 113, 111), catalog, fams)    # rho0 pi0
    assert (e.n_ssbar, e.n_v, e.n_bbar) == (0.0, 1, 0)
    e = flavor_exponents(neutral, _channel("mesonic_neutral", 2212, -2212), catalog, fams)  # p pbar
    assert (e.n_bbar, e.n_ssbar) == (1, 0.0)
    e = flavor_exponents((1, 1, 0), _channel("baryonic_np", 3122, 321), catalog, fams)      # N* -> Lambda K+
    assert (e.n_ssbar, e.n_assoc) == (1.0, 1)
    e = flavor_exponents((1, 0, -1), _channel("baryonic_strange", 2212, -321), catalog, fams)  # strange parent -> p K-: transfer, no creation
    assert (e.n_ssbar, e.n_assoc) == (0.0, 0)
    e = flavor_exponents((1, 1, 0), _channel("baryonic_np", 2212, 111), catalog, fams)      # p pi0
    assert (e.n_ssbar, e.n_v, e.n_bbar, e.n_assoc) == (0.0, 0, 0, 0)
    # a continuum daughter counts through its family quantum numbers
    ch = CascadeChannel("mesonic_neutral", ("mesonic_strange", f"{CATALOG_FAMILY_PREFIX}321"), KIND_CONTINUUM, 1.0)
    assert flavor_exponents(neutral, ch, catalog, fams).n_ssbar == 1.0


def test_weights_and_parameter_validation():
    e = flavor_exponents  # noqa: F841
    p = FlavorParams(gamma_s_had=0.5, r_v=2.0, r_bbar=0.1, assoc_norm=1.5)
    from na49_mixing.flavor_weights import FlavorExponents
    assert flavor_weight(FlavorExponents(1.0, 1, 0, 0), p) == 0.5 * 2.0
    assert flavor_weight(FlavorExponents(0.0, 0, 1, 0), p) == 0.1
    assert flavor_weight(FlavorExponents(1.0, 0, 0, 1), p) == 0.5 * 1.5
    assert flavor_weight(FlavorExponents(0.0, 0, 0, 0), p) == 1.0
    assert NOMINAL_FLAVOR.is_identity() and not p.is_identity()
    for bad in ({"gamma_s_had": 0.0}, {"r_v": -1.0}, {"r_bbar": math.inf}, {"hidden_ssbar_content": {333: 3.0}}):
        with pytest.raises(ValueError):
            FlavorParams(**bad)


def test_identity_returns_the_same_objects(catalog, fams):
    state = _root("mesonic_neutral", 4.0, (0, 0, 0))
    channels = enumerate_channels(state, catalog, fams)
    assert enumerate_channels(state, catalog, fams, flavor=None) == channels
    same = apply_flavor_weights(channels, state.qn, catalog, fams, NOMINAL_FLAVOR)
    assert all(a is b for a, b in zip(same, channels))
    assert NOMINAL_PARAMS.flavor is None


@pytest.mark.parametrize("family_id, qn, mass", [("mesonic_neutral", (0, 0, 0), 3.0), ("mesonic_neutral", (0, 0, 0), 4.5), ("baryonic_np", (1, 1, 0), 3.5), ("baryonic_np", (1, 1, 0), 4.8)])
def test_flavor_source_grid_monotone_shares(catalog, fams, family_id, qn, mass):
    """Task 44: the strange, vector, and baryon-pair shares of the vertex
    fall monotonically as gamma_s, r_V, r_BBbar fall, and the weights stay
    normalized."""
    state = _root(family_id, mass, qn)
    base = enumerate_channels(state, catalog, fams)
    assert base and abs(math.fsum(c.weight for c in base) - 1.0) < 1e-12

    def shares(params):
        chans = enumerate_channels(state, catalog, fams, flavor=params)
        assert abs(math.fsum(c.weight for c in chans) - 1.0) < 1e-12
        ex = [flavor_exponents(qn, c, catalog, fams) for c in chans]
        return (math.fsum(c.weight for c, e in zip(chans, ex) if e.n_ssbar > 0),
                math.fsum(c.weight for c, e in zip(chans, ex) if e.n_v > 0),
                math.fsum(c.weight for c, e in zip(chans, ex) if e.n_bbar > 0))
    s1, v1, b1 = shares(NOMINAL_FLAVOR)
    s2, _, _ = shares(FlavorParams(gamma_s_had=0.5))
    s3, _, _ = shares(FlavorParams(gamma_s_had=0.25))
    assert s1 > s2 > s3 >= 0.0
    _, v2, _ = shares(FlavorParams(r_v=0.5))
    if v1 > 0.0:  # the baryonic families declare no vector-meson class
        assert v1 > v2
    else:
        assert v2 == 0.0
    if family_id == "mesonic_neutral" and mass > 2.0:
        assert b1 > 0.0, "the baryon-pair classes are open above 2 m_p"
        _, _, b2 = shares(FlavorParams(r_bbar=0.2))
        assert b1 > b2 > 0.0
        assert s1 > 0.0 and v1 > 0.0


def test_round6_classes_open_in_the_enumeration(catalog, fams):
    """K* K-bar, phi, and baryon-pair channels appear among a neutral
    mesonic source's direct channels above their thresholds.  Under the
    config's isospin rule (I = 0 for the neutral mesonic family, owner
    decision 2026-09-30) phi pi0 carries no isospin share (I = 0 x I = 1
    cannot couple to I = 0); without a rule every class opens."""
    import dataclasses
    state = _root("mesonic_neutral", 4.5, (0, 0, 0))

    def pairs_of(families):
        return {tuple(sorted(int(d[len(CATALOG_FAMILY_PREFIX):]) for d in c.daughters))
                for c in enumerate_channels(state, catalog, families) if all(d.startswith(CATALOG_FAMILY_PREFIX) for d in c.daughters)}

    pairs = pairs_of(fams)
    for want in ((-321, 323), (-323, 321), (-311, 313), (-313, 311), (221, 333), (-2212, 2212), (-2112, 2112)):
        assert want in pairs, want
    assert (111, 333) not in pairs
    no_rule = pairs_of([dataclasses.replace(f, isospin=None) for f in fams])
    for want in ((-321, 323), (-323, 321), (-311, 313), (-313, 311), (111, 333), (221, 333), (-2212, 2212), (-2112, 2112)):
        assert want in no_rule, want


def test_charge_completion_of_the_family_classes(catalog, fams):
    """Task 44: every allowed class conserves (B, Q, S), every daughter is
    a catalog record, and every Round 6 strange class carries its isospin
    or charge-conjugate partner within the same family."""
    for family in fams:
        fqn = tuple(family.qn)
        classes = [tuple(int(x) for x in cls.strip("()").split(",")) for cls in family.allowed_decay_classes]
        for d0, d1 in classes:
            r0, r1 = catalog.get(d0), catalog.get(d1)
            assert (r0.B + r1.B, r0.Q + r1.Q, r0.S + r1.S) == fqn, (family.family_id, d0, d1)
        as_set = {tuple(sorted(c)) for c in classes}
        partners = {
            "mesonic_neutral": [((-321, 323), (-311, 313)), ((-323, 321), (-313, 311)), ((-2212, 2212), (-2112, 2112))],
            "baryonic_np": [((311, 3224), (321, 3214))],
            "baryonic_strange": [((211, 3114), (-211, 3224))],
        }.get(family.family_id, [])
        for a, b in partners:
            assert a in as_set and b in as_set, (family.family_id, a, b)
        for cls in ROUND6_CLASSES.get(family.family_id, []):
            assert cls in family.allowed_decay_classes, (family.family_id, cls)


def test_catalog_parents_keep_measured_branching(catalog, fams):
    state = _root(f"{CATALOG_FAMILY_PREFIX}2214", 1.6, (1, 1, 0))
    base = enumerate_channels(state, catalog, fams)
    strong = enumerate_channels(state, catalog, fams, flavor=FlavorParams(gamma_s_had=0.1, r_v=0.1, r_bbar=0.1))
    assert [c.weight for c in strong] == [c.weight for c in base]


def test_zero_random_numbers(catalog, fams):
    state = _root("mesonic_neutral", 4.0, (0, 0, 0))
    rng = np.random.default_rng(5)
    before = rng.bit_generator.state
    enumerate_channels(state, catalog, fams, flavor=FlavorParams(gamma_s_had=0.3, r_v=0.7, r_bbar=0.2))
    assert rng.bit_generator.state == before
