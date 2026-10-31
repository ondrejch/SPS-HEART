"""Tests for the cascade continuation floor (Round 5 follow-up Block B
stage 2, card ``TASK-20260929-r5-gate5-followup-cascade-multiplicity``).

The floor is an optional per-family config key, ``continuation_floor``,
that lowers the realized mass of a continuation daughter below the
family's ``m_min``. The absent key reproduces the card-4 rule bit for
bit; the source density, its ramp and ``n_norm`` keep ``m_min``.
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
    KIND_CONTINUUM,
    CascadeState,
    _continuum_daughter_mass,
    draw_decay,
    enumerate_channels,
)
from na49_mixing.cascade_termination import run_cascade
from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.spectral_density import (
    _family_from_record,
    families_from_json,
    family_density,
)

NESTED_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
FAMILIES_PATH = NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json"
SEED = 20260930


@pytest.fixture(scope="module")
def catalog() -> HadronCatalog:
    return HadronCatalog.from_json(CATALOG_PATH)


@pytest.fixture(scope="module")
def fams():
    return families_from_json(str(FAMILIES_PATH))


@pytest.fixture(scope="module")
def records():
    return json.loads(FAMILIES_PATH.read_text(encoding="utf-8"))["families"]


def _with_floor(fams, floor):
    return [dataclasses.replace(f, continuation_floor=floor) for f in fams]


def _root(family_id: str, mass: float, qn) -> CascadeState:
    return CascadeState(
        momentum=(float(mass), 0.0, 0.0, 0.0), mass=float(mass), qn=tuple(qn),
        quantum={}, family_id=family_id, axis=(0.0, 0.0, 1.0),
        alignment_label=ISOTROPIC_ALIGNMENT_LABEL, parent=None,
        exchange_ancestry=("floor_probe",), depth=0)


def _family(fams, family_id):
    return next(f for f in fams if f.family_id == family_id)


# ----------------------------------------------------------------------
# the loader
# ----------------------------------------------------------------------
def test_committed_config_sets_no_floor(fams):
    assert all(f.continuation_floor is None for f in fams)


def test_loader_reads_and_validates_the_floor(records):
    record = copy.deepcopy(records[0])
    base = _family_from_record(record)
    record["continuation_floor"] = 1.5
    family = _family_from_record(record)
    assert family.continuation_floor == 1.5
    # the source side keeps m_min: n_norm, the ramp and the density
    assert family.m_min == base.m_min and family.n_norm == base.n_norm
    for mass in (2.0, 2.7, 3.0, 4.5):
        assert family_density(family, mass) == family_density(base, mass)
    record["continuation_floor"] = base.m_min
    assert _family_from_record(record).continuation_floor == base.m_min
    for bad in (0.0, -1.0, base.m_min + 1e-9, math.inf, None, "1.5x"):
        record["continuation_floor"] = bad
        with pytest.raises(ValueError):
            _family_from_record(record)


# ----------------------------------------------------------------------
# bit identity at the default
# ----------------------------------------------------------------------
@pytest.mark.parametrize("family_id, qn", [("baryonic_np", (1, 1, 0)),
                                           ("mesonic_neutral", (0, 0, 0))])
def test_floor_at_m_min_is_bit_identical(catalog, fams, family_id, qn):
    """An explicit floor equal to m_min enumerates the same channels
    and draws the same cascade as the absent floor."""

    at_min = [dataclasses.replace(f, continuation_floor=f.m_min) for f in fams]
    for mass in (2.9, 3.4, 4.0, 4.9):
        state = _root(family_id, mass, qn)
        assert enumerate_channels(state, catalog, fams) == \
            enumerate_channels(state, catalog, at_min)
        lines = []
        for families in (fams, at_min):
            rc = np.random.default_rng(SEED); rk = np.random.default_rng(SEED + 1)
            lines.append([(s.family_id, s.mass, s.momentum)
                          for s in run_cascade(state, catalog, families, rc, rk)])
        assert lines[0] == lines[1]


# ----------------------------------------------------------------------
# the lowered floor
# ----------------------------------------------------------------------
def test_mass_rule_uses_the_floor(fams):
    family = _family(fams, "baryonic_np")
    lowered = dataclasses.replace(family, continuation_floor=1.5)
    width = float(family.nominal_width)
    # a 2.68 GeV state emitting a pion: below m_min, open above the floor
    parent, sibling = 2.68, 0.13957
    edge = parent - sibling
    with pytest.raises(ValueError):
        _continuum_daughter_mass(family, parent, sibling)
    assert _continuum_daughter_mass(lowered, parent, sibling) == max(1.5, edge - width)
    # the floor binds when the width step would cross it
    assert _continuum_daughter_mass(lowered, 2.0, sibling) == 1.5
    with pytest.raises(ValueError):
        _continuum_daughter_mass(lowered, 1.6, sibling)
    # above the routing boundary the floor changes nothing unless it binds
    assert _continuum_daughter_mass(lowered, 4.0, sibling) == \
        _continuum_daughter_mass(family, 4.0, sibling)


def test_lowered_floor_opens_continuation_below_m_min(catalog, fams):
    lowered = _with_floor(fams, 1.5)
    state = _root("baryonic_np", 2.6, (1, 1, 0))
    before = [c for c in enumerate_channels(state, catalog, fams) if c.kind == KIND_CONTINUUM]
    after = [c for c in enumerate_channels(state, catalog, lowered) if c.kind == KIND_CONTINUUM]
    assert before == [] and after
    rc = np.random.default_rng(SEED); rk = np.random.default_rng(SEED + 1)
    channel, daughters = draw_decay(state, after, catalog, lowered, rc, rk)
    line = [d for d in daughters if not d.family_id.startswith(CATALOG_FAMILY_PREFIX)]
    assert len(line) == 1 and 1.5 <= line[0].mass < 2.6
    assert sum(d.momentum[0] for d in daughters) == pytest.approx(2.6, abs=1e-9)


@pytest.mark.parametrize("family_id, qn", [("baryonic_np", (1, 1, 0)),
                                           ("mesonic_neutral", (0, 0, 0))])
def test_lowered_floor_ends_lines_below_the_boundary(catalog, fams, family_id, qn):
    """With the frozen floor every line from a 4 GeV source ends at the
    routing boundary; with a 1.5 GeV floor the lines end between the
    floor and the boundary after more steps, and conserve energy."""

    lowered = _with_floor(fams, 1.5)
    m_min = _family(fams, family_id).m_min
    ends = {"frozen": [], "lowered": []}; steps = {"frozen": [], "lowered": []}
    for label, families in (("frozen", fams), ("lowered", lowered)):
        rc = np.random.default_rng(SEED); rk = np.random.default_rng(SEED + 1)
        for _ in range(200):
            line = run_cascade(_root(family_id, 4.0, qn), catalog, families, rc, rk)
            continuum = [s for s in line if not s.family_id.startswith(CATALOG_FAMILY_PREFIX)]
            ends[label].append(continuum[-1].mass); steps[label].append(len(continuum) - 1)
    assert min(ends["frozen"]) >= m_min - 1e-12
    assert all(1.5 - 1e-12 <= m for m in ends["lowered"])
    assert np.mean(ends["lowered"]) < m_min - 0.5
    assert np.mean(steps["lowered"]) > np.mean(steps["frozen"])


# ----------------------------------------------------------------------
# the threshold-aware floor (2026-10-01)
# ----------------------------------------------------------------------
def test_floor_never_below_the_family_threshold(catalog, fams):
    """A continuation daughter is realized no lower than its family's
    lowest direct-channel threshold: a 1.1 GeV floor lifts the strange
    baryonic family to its lightest open pair (above the Lambda mass),
    leaves the nucleon family at 1.1, and the default (m_min) is untouched."""

    from na49_mixing.cascade_decay import _continuation_floor, _lowest_direct_threshold
    strange = _family(fams, "baryonic_strange")
    nucleon = _family(fams, "baryonic_np")
    thr = _lowest_direct_threshold(strange, catalog)
    assert thr > 1.1157
    assert _continuation_floor(dataclasses.replace(strange, continuation_floor=1.1), catalog) == thr
    assert _continuation_floor(dataclasses.replace(nucleon, continuation_floor=1.1), catalog) == 1.1
    assert _continuation_floor(strange, catalog) == strange.m_min
    # without the catalog the floor is the configured value (the Round 4
    # ablation module's callers)
    assert _continuation_floor(dataclasses.replace(strange, continuation_floor=1.1)) == 1.1


@pytest.mark.parametrize("cls", ["sd_projectile", "dd", "nd"])
def test_conservation_at_a_low_floor(catalog, fams, cls):
    """At a 1.1 GeV floor every event conserves (B, Q, S) exactly: no
    continuation daughter is stranded below its thresholds, and a
    below-threshold state carries its line's own quantum numbers."""

    from na49_mixing.rng_substreams import derive_substreams
    from na49_mixing.topology_runner import NOMINAL_PARAMS, run_topology
    low = _with_floor(fams, 1.1)
    for i in range(80):
        subs = derive_substreams(SEED, 3, 9_200_000_000 + i)
        rngs = {f"rng_{lab}": np.random.default_rng(subs.stream_for(lab)) for lab in
                ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")}
        event = run_topology(cls, NOMINAL_PARAMS, catalog, low, **rngs)
        assert tuple(sum(int(h.qn[j]) for h in event.final_state) for j in range(3)) == (2, 2, 0)
