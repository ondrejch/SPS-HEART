"""Tests for the 2026-09-30 final-state fixes (card
TASK-20260930-r5-gate5-followup-final-state-isospin).

(a) Every unstable established hadron decays before it reaches a final
state: the direct-channel daughters of the single-diffraction excitation,
the double-diffraction systems and the non-diffractive central sources,
and the single-diffraction remnant record.  (b) The continuum config's
``isospin_rule`` gives every family its minimal isospin I = |I3|, and the
continuum and direct vertices weight their channels by the
Clebsch-Gordan share.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.cascade_decay import (
    CATALOG_FAMILY_PREFIX,
    ISOTROPIC_ALIGNMENT_LABEL,
    KIND_CONTINUUM,
    CascadeState,
    _isospin_factor_continuum,
    _isospin_factor_direct,
    enumerate_channels,
)
from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.rng_substreams import derive_substreams
from na49_mixing.spectral_density import ISOSPIN_RULES, families_from_json, family_isospin
from na49_mixing.topology_runner import NOMINAL_PARAMS, run_topology
from na49_mixing.topology_single_diffraction import SDEvent, gap_size

NESTED_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
FAMILIES_PATH = NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json"
CLASSES = ("sd_projectile", "sd_target", "dd", "nd")


@pytest.fixture(scope="module")
def catalog() -> HadronCatalog:
    return HadronCatalog.from_json(CATALOG_PATH)


@pytest.fixture(scope="module")
def fams():
    return families_from_json(str(FAMILIES_PATH))


def _root(family_id, mass, qn):
    return CascadeState(momentum=(float(mass), 0.0, 0.0, 0.0), mass=float(mass), qn=tuple(qn), quantum={},
                        family_id=family_id, axis=(0.0, 0.0, 1.0), alignment_label=ISOTROPIC_ALIGNMENT_LABEL,
                        parent=None, exchange_ancestry=("isospin_probe",), depth=0)


def _direct_pairs(channels):
    out = {}
    for c in channels:
        if c.kind != KIND_CONTINUUM:
            out[tuple(sorted(int(d[len(CATALOG_FAMILY_PREFIX):]) for d in c.daughters))] = c.weight
    return out


# ----------------------------------------------------------------------
# (b) the isospin rule
# ----------------------------------------------------------------------
def test_committed_config_sets_the_minimal_isospin_rule(fams):
    config = json.loads(FAMILIES_PATH.read_text())
    assert config["isospin_rule"]["rule"] in ISOSPIN_RULES
    iso = {f.family_id: f.isospin for f in fams}
    assert iso["baryonic_np"] == 0.5 and iso["mesonic_neutral"] == 0.0 and iso["baryonic_strange"] == 0.0
    assert iso["mesonic_charged"] == iso["mesonic_anticharged"] == 1.0
    assert iso["mesonic_strange"] == iso["mesonic_antistrange"] == 0.5
    assert all(f.isospin == family_isospin(f.qn) for f in fams)


def test_loader_refuses_an_unknown_rule(tmp_path):
    config = json.loads(FAMILIES_PATH.read_text())
    config["isospin_rule"] = {"rule": "maximal"}
    path = tmp_path / "families.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        families_from_json(str(path))
    del config["isospin_rule"]
    path.write_text(json.dumps(config))
    assert all(f.isospin is None for f in families_from_json(str(path)))


def test_clebsch_gordan_shares(catalog):
    f = lambda i, m, a, b: _isospin_factor_direct(i, m, a, b, catalog)  # noqa: E731
    # I = 0 central mesonic source
    assert math.isclose(f(0.0, 0, 211, -211), 2 / 3) and math.isclose(f(0.0, 0, 111, 111), 1 / 3)
    assert f(0.0, 0, 223, 111) == 0.0 and f(0.0, 0, 221, 111) == 0.0      # omega pi0, eta pi0 forbidden
    assert math.isclose(f(0.0, 0, 321, -321), 0.5) and math.isclose(f(0.0, 0, 311, -311), 0.5)
    # the weak-eigenstate pair at its flavor states (the catalog records K0S
    # and K0L at 2 I3 = 0, a mixture)
    assert math.isclose(f(0.0, 0, 310, 130), 0.5)
    assert math.isclose(f(0.0, 0, 213, -211) + f(0.0, 0, -213, 211) + f(0.0, 0, 113, 111), 1.0)
    # I = 1/2 excited nucleon (2 I3 = +1)
    assert math.isclose(f(0.5, 1, 2212, 111), 1 / 3) and math.isclose(f(0.5, 1, 2112, 211), 2 / 3)
    assert math.isclose(f(0.5, 1, 2224, -211), 0.5) and math.isclose(f(0.5, 1, 2214, 111), 1 / 3)
    assert math.isclose(f(0.5, 1, 2114, 211), 1 / 6)
    assert math.isclose(f(0.5, 1, 2212, 223), 1.0)                        # p omega (I = 0 meson)


def test_isospin_rule_shapes_the_enumerated_channels(catalog, fams):
    no_rule = [dataclasses.replace(f, isospin=None) for f in fams]
    meson = _root("mesonic_neutral", 3.0, (0, 0, 0))
    with_rule = _direct_pairs(enumerate_channels(meson, catalog, fams))
    without = _direct_pairs(enumerate_channels(meson, catalog, no_rule))
    assert (111, 223) in without and (111, 223) not in with_rule        # omega pi0 removed by I = 0
    ratio_with = with_rule[(-211, 211)] / with_rule[(111, 111)]
    ratio_without = without[(-211, 211)] / without[(111, 111)]
    assert math.isclose(ratio_with / ratio_without, 2.0, rel_tol=1e-12)
    baryon = _root("baryonic_np", 2.5, (1, 1, 0))
    b_with = _direct_pairs(enumerate_channels(baryon, catalog, fams))
    b_without = _direct_pairs(enumerate_channels(baryon, catalog, no_rule))
    r_with = b_with[(211, 2112)] / b_with[(111, 2212)]
    r_without = b_without[(211, 2112)] / b_without[(111, 2212)]
    assert math.isclose(r_with / r_without, 2.0, rel_tol=1e-12)          # n pi+ : p pi0 = 2 : 1


def test_continuation_share_needs_a_daughter_isospin(catalog, fams):
    parent = next(f for f in fams if f.family_id == "mesonic_neutral")
    charged = next(f for f in fams if f.family_id == "mesonic_charged")
    # I = 0 -> (I = 1, I3 = +1) continuum + pi- : |<1 1; 1 -1 | 0 0>|^2 = 1/3
    assert math.isclose(_isospin_factor_continuum(0.0, 0, charged, -211, catalog), 1 / 3)
    # I = 0 -> (I = 0) continuum + pi0 is forbidden
    assert _isospin_factor_continuum(0.0, 0, parent, 111, catalog) == 0.0
    with pytest.raises(ValueError):
        _isospin_factor_continuum(0.0, 0, dataclasses.replace(charged, isospin=None), -211, catalog)


# ----------------------------------------------------------------------
# (a) no unstable hadron in any final state
# ----------------------------------------------------------------------
@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("floor", [None, 1.4])
def test_no_unstable_final_states(catalog, fams, cls, floor):
    families = fams if floor is None else [dataclasses.replace(f, continuation_floor=floor) for f in fams]
    stable = catalog.stable_states()
    for i in range(60):
        subs = derive_substreams(20260930, 0, 9_000_000_000 + i)
        rngs = {f"rng_{lab}": np.random.default_rng(subs.stream_for(lab)) for lab in
                ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")}
        event = run_topology(cls, NOMINAL_PARAMS, catalog, families, **rngs)
        bad = [h.pdg_id for h in event.final_state if h.pdg_id is not None and int(h.pdg_id) not in stable]
        assert bad == [], (cls, i, bad)
        assert tuple(sum(int(h.qn[j]) for h in event.final_state) for j in range(3)) == (2, 2, 0)


def test_low_mass_sd_remnant_decays_and_the_gap_holds(catalog, fams):
    stable = catalog.stable_states()
    seen_decayed = 0
    for i in range(120):
        subs = derive_substreams(20260930, 1, 9_100_000_000 + i)
        rngs = {f"rng_{lab}": np.random.default_rng(subs.stream_for(lab)) for lab in
                ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")}
        event = run_topology("sd_projectile", NOMINAL_PARAMS, catalog, fams, **rngs)
        assert isinstance(event, SDEvent)
        remnant_line = [h for h in event.final_state if h.line == "remnant"]
        if event.remnant.pdg_id is not None and int(event.remnant.pdg_id) not in stable:
            seen_decayed += 1
            assert len(remnant_line) >= 2
            total = [math.fsum(float(h.four_momentum[j]) for h in remnant_line) for j in range(4)]
            m2 = total[0] ** 2 - total[1] ** 2 - total[2] ** 2 - total[3] ** 2
            assert math.isclose(math.sqrt(m2), float(event.remnant.mass), rel_tol=1e-9)
        else:
            assert len(remnant_line) == 1
        assert math.isclose(gap_size(event), float(event.gap_size), abs_tol=1e-12)
    assert seen_decayed > 10
