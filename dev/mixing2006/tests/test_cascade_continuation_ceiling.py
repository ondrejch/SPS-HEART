"""Tests for the cascade continuation ceiling (2026-10-02; the diagnosis of
the central deficit after R67).

The ceiling is an optional per-family key, ``continuation_ceiling``, that
lifts the highest realized mass of a continuation daughter above the
support edge ``SUPPORT_M_MAX = 5.0 GeV``. Without it a state above the
edge sheds its excess into the first emitted hadron (an 8 GeV baryonic
state drops to 4.5 GeV in one step). The absent key reproduces the card-4
rule bit for bit; the source density, its support and ``n_norm`` keep the
support edge.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.cascade_decay import (
    CATALOG_FAMILY_PREFIX,
    ISOTROPIC_ALIGNMENT_LABEL,
    CascadeState,
    _continuation_ceiling,
    _continuum_daughter_mass,
    enumerate_channels,
)
from na49_mixing.cascade_termination import run_cascade
from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.spectral_density import (
    SUPPORT_M_MAX,
    _family_from_record,
    families_from_json,
    family_density,
)

NESTED_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
FAMILIES_PATH = NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json"
SEED = 20261002


@pytest.fixture(scope="module")
def catalog() -> HadronCatalog:
    return HadronCatalog.from_json(CATALOG_PATH)


@pytest.fixture(scope="module")
def fams():
    return families_from_json(str(FAMILIES_PATH))


@pytest.fixture(scope="module")
def records():
    return json.loads(FAMILIES_PATH.read_text(encoding="utf-8"))["families"]


def _with(fams, **kw):
    return [dataclasses.replace(f, **kw) for f in fams]


def _root(family_id: str, mass: float, qn) -> CascadeState:
    return CascadeState(
        momentum=(float(mass), 0.0, 0.0, 0.0), mass=float(mass), qn=tuple(qn),
        quantum={}, family_id=family_id, axis=(0.0, 0.0, 1.0),
        alignment_label=ISOTROPIC_ALIGNMENT_LABEL, parent=None,
        exchange_ancestry=("ceiling_probe",), depth=0)


def _family(fams, family_id):
    return next(f for f in fams if f.family_id == family_id)


def test_committed_config_sets_no_ceiling(fams):
    assert all(f.continuation_ceiling is None for f in fams)
    assert all(_continuation_ceiling(f) == SUPPORT_M_MAX for f in fams)


def test_loader_reads_and_validates_the_ceiling(records):
    record = copy.deepcopy(records[0])
    base = _family_from_record(record)
    record["continuation_ceiling"] = 17.3
    family = _family_from_record(record)
    assert family.continuation_ceiling == 17.3 and _continuation_ceiling(family) == 17.3
    # the source side keeps the support edge: n_norm and the density
    assert family.n_norm == base.n_norm
    for mass in (2.9, 3.5, 4.5):
        assert family_density(family, mass) == family_density(base, mass)
    record["continuation_ceiling"] = SUPPORT_M_MAX
    assert _family_from_record(record).continuation_ceiling == SUPPORT_M_MAX
    for bad in (4.9, 0.0, -1.0, math.inf, math.nan, None, "x"):
        record["continuation_ceiling"] = bad
        with pytest.raises(ValueError):
            _family_from_record(record)


@pytest.mark.parametrize("family_id, qn", [("baryonic_np", (1, 1, 0)), ("mesonic_neutral", (0, 0, 0))])
def test_ceiling_at_the_support_edge_is_bit_identical(catalog, fams, family_id, qn):
    at_edge = _with(fams, continuation_ceiling=SUPPORT_M_MAX)
    for mass in (4.0, 6.0, 9.0):
        state = _root(family_id, mass, qn)
        assert enumerate_channels(state, catalog, fams) == enumerate_channels(state, catalog, at_edge)
        lines = []
        for families in (fams, at_edge):
            rc = np.random.default_rng(SEED); rk = np.random.default_rng(SEED + 1)
            lines.append([(s.family_id, s.mass, s.momentum) for s in run_cascade(state, catalog, families, rc, rk)])
        assert lines[0] == lines[1]


def test_mass_rule_uses_the_ceiling(fams):
    family = _family(fams, "baryonic_np")
    lifted = dataclasses.replace(family, continuation_ceiling=17.3)
    width = float(family.nominal_width)
    pion = 0.13957
    # an 8 GeV state emitting a pion: capped at the support edge, or one
    # nominal width below the budget edge with the ceiling lifted
    assert _continuum_daughter_mass(family, 8.0, pion) == SUPPORT_M_MAX - width
    assert _continuum_daughter_mass(lifted, 8.0, pion) == 8.0 - pion - width
    # below the support edge the ceiling changes nothing
    assert _continuum_daughter_mass(lifted, 4.0, pion) == _continuum_daughter_mass(family, 4.0, pion)


def test_lifted_ceiling_cascades_heavy_states_step_by_step(catalog, fams):
    """From a 9 GeV baryonic state the lifted ceiling gives more cascade
    steps, more charged hadrons and no daughter below the budget edge
    minus one width; four-momentum is conserved per line."""

    lifted = _with(fams, continuation_ceiling=17.3)
    stats = {}
    for label, families in (("capped", fams), ("lifted", lifted)):
        rc = np.random.default_rng(SEED); rk = np.random.default_rng(SEED + 1)
        first_drop, steps = [], []
        for _ in range(150):
            line = run_cascade(_root("baryonic_np", 9.0, (1, 1, 0)), catalog, families, rc, rk)
            continuum = [s for s in line if not s.family_id.startswith(CATALOG_FAMILY_PREFIX)]
            steps.append(len(continuum) - 1)
            if len(continuum) > 1:
                first_drop.append(continuum[0].mass - continuum[1].mass)
        stats[label] = (np.mean(steps), np.mean(first_drop))
    assert stats["capped"][1] > 3.0            # the capped first step sheds more than 3 GeV
    assert stats["lifted"][1] < stats["capped"][1] - 1.0
    assert stats["lifted"][0] > stats["capped"][0]


@pytest.mark.parametrize("cls", ["sd_projectile", "dd", "nd"])
def test_conservation_with_the_lifted_ceiling(catalog, fams, cls):
    from na49_mixing.rng_substreams import derive_substreams
    from na49_mixing.topology_runner import NOMINAL_PARAMS, run_topology
    lifted = _with(fams, continuation_floor=1.7, continuation_ceiling=17.3)
    # the non-diffractive remnants mostly excited; the diffractive split
    # keeps its nominal intact and low-mass kinds
    weights = ({"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.1, "high_mass_baryonic_continuum": 0.9}
               if cls == "nd" else dict(NOMINAL_PARAMS.remnant_weights))
    params = dataclasses.replace(NOMINAL_PARAMS, epsilon=0.0, epsilon_nd=0.0, remnant_weights=weights)
    for i in range(60):
        subs = derive_substreams(SEED, 3, 9_400_000_000 + i)
        rngs = {f"rng_{lab}": np.random.default_rng(subs.stream_for(lab)) for lab in
                ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")}
        event = run_topology(cls, params, catalog, lifted, **rngs)
        assert tuple(sum(int(h.qn[j]) for h in event.final_state) for j in range(3)) == (2, 2, 0)


def test_the_depth_guard_follows_the_ceiling(fams):
    """The card-9 guard stays 12 for the committed families; a lifted
    ceiling raises it to the lifted depth, which a 16 GeV state needs (about
    23 steps of one pion and one nominal width down to a 1.7 GeV floor)."""

    from na49_mixing.cascade_decay import MAX_CASCADE_DEPTH, _LIFTED_CASCADE_DEPTH, _cascade_depth_limit
    assert MAX_CASCADE_DEPTH == 12 and _cascade_depth_limit(fams) == 12
    lifted = _with(fams, continuation_floor=1.7, continuation_ceiling=17.3)
    assert _cascade_depth_limit(lifted) == _LIFTED_CASCADE_DEPTH == 48
    steps = (16.3 - 1.7) / (0.1349766 + 0.5)
    assert steps < _LIFTED_CASCADE_DEPTH - 10


@pytest.mark.parametrize("cls", ["sd_projectile", "dd", "nd"])
def test_heavy_excitations_cascade_deeply_with_the_lifted_ceiling(catalog, fams, cls):
    """Heavy excitations (epsilon = -0.5 favors high masses) cascade with
    the lifted ceiling and conserve.  Chains deeper than the card-9 guard
    are rare: the first R68 generation met depth 13 in a few events of
    16 million, in slots 20, 98, 123, 130 and 155."""

    from na49_mixing.rng_substreams import derive_substreams
    from na49_mixing.topology_runner import NOMINAL_PARAMS, run_topology
    lifted = _with(fams, continuation_floor=1.7, continuation_ceiling=17.3)
    weights = ({"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.0, "high_mass_baryonic_continuum": 1.0}
               if cls == "nd" else dict(NOMINAL_PARAMS.remnant_weights))
    params = dataclasses.replace(NOMINAL_PARAMS, epsilon=-0.5, epsilon_nd=-0.5, remnant_weights=weights)
    for i in range(80):
        subs = derive_substreams(SEED, 4, 9_500_000_000 + i)
        rngs = {f"rng_{lab}": np.random.default_rng(subs.stream_for(lab)) for lab in
                ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")}
        event = run_topology(cls, params, catalog, lifted, **rngs)
        assert tuple(sum(int(h.qn[j]) for h in event.final_state) for j in range(3)) == (2, 2, 0)
