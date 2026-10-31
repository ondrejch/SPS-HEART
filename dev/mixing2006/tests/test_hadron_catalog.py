"""Typed hadron/resonance schema validation (Round 2, card 1).

Task ``TASK-20260913-r2-hadron-resonance-schema`` (plan section 22,
task 9; record fields of plan section 8.1).  The seven deterministic
tests pin the ``na49_mixing.hadron_catalog`` contract:

- ``test_record_roundtrip``: a hand-constructed ``HadronRecord`` set
  serializes to JSON and parses back to an equal catalog;
- ``test_channel_conserves_bqs``: a channel with a wrong daughter sum
  (a missing charge state) makes ``validate()`` report an error naming
  the channel (plan section 2.5);
- ``test_branching_sum_guard``: a per-parent branching sum exceeding 1
  by more than ``1e-9`` is an error; a sum of ``0.9999999995`` is a
  warning only (the unrepresented channel budget);
- ``test_mass_range_consistency``: a ``mass_range`` lower bound below
  the lowest channel threshold is an error;
- ``test_stable_termination``: a ``stable_for_decay`` record with a
  non-empty ``decay_channels`` is an error (and the iff converse);
- ``test_isospin3_consistency``: a record with
  ``Q != I3 + (B + S)/2`` is an error;
- ``test_pdg_agreement``: a record whose ``pole_mass`` contradicts the
  vendored table by more than ``1e-6`` GeV is an error.

The probes are hand-constructed synthetic records built around the
vendored PDG table entries (rho(770), omega(782), pi, K) so that every
rule can be isolated; no physics values are transcribed by this card
(the catalog import with provenance is card 2).  Channel thresholds in
the probes are derived from the same vendored masses the catalog
reads, so the mass-range rule is satisfied by construction except
where a test violates it on purpose.

Hermetic: no event generation, no external tables.
"""

from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from na49_mixing.hadron_catalog import (
    STABLE_STATES,
    CatalogValidationReport,
    DecayChannel,
    HadronCatalog,
    HadronRecord,
    ProvenanceTag,
)
from na49_mixing.pdg_database import get_mass

MASS_RANGE_SPAN = 0.2  # GeV/c^2 headroom above the threshold in the probes


# ----------------------------------------------------------------------
# Hand-constructed record builders (synthetic schema probes)
# ----------------------------------------------------------------------
def _tag(field: str, value) -> ProvenanceTag:
    return ProvenanceTag(
        document="PDG 2024 review, sect. 27",
        section_or_table="table 27.1",
        field=field,
        accessed_on="2026-09-13",
        transcribed_value=value,
        transcriber="na49-res r2 card 2",
    )


def _channel(parent_pdg: int, daughters, branching_fraction: float = 1.0,
             orbital_angular_momentum: int = 0, is_three_body: bool = False,
             provenance: ProvenanceTag | None = None) -> DecayChannel:
    return DecayChannel(
        parent_pdg=parent_pdg,
        daughters=tuple(daughters),
        branching_fraction=branching_fraction,
        orbital_angular_momentum=orbital_angular_momentum,
        is_three_body=is_three_body,
        provenance=provenance,
    )


def _threshold(channel: DecayChannel) -> float:
    """Channel threshold from the vendored masses (the masses the
    catalog itself reads for non-record daughters)."""
    return math.fsum(get_mass(d) for d in channel.daughters)


def _resonance(pdg_id: int, name: str, B: int, Q: int, S: int,
               isospin: float, isospin3: int, channels,
               pole_mass: float | None = None,
               provenance: ProvenanceTag | None = None) -> HadronRecord:
    """A decayable resonance record consistent with every rule except
    the one a test violates on purpose."""
    channels = tuple(channels)
    lowest = min(_threshold(c) for c in channels)
    if pole_mass is None:
        pole_mass = get_mass(pdg_id)
    return HadronRecord(
        pdg_id=pdg_id,
        name=name,
        B=B,
        Q=Q,
        S=S,
        isospin=isospin,
        isospin3=isospin3,
        spin=1.0,
        parity="-",
        pole_mass=pole_mass,
        nominal_width=0.1491,
        mass_range=(lowest, lowest + MASS_RANGE_SPAN),
        decay_channels=channels,
        stable_for_decay=False,
        provenance=provenance,
    )


def _stable(pdg_id: int, name: str, B: int, Q: int, S: int,
            isospin: float, isospin3: int) -> HadronRecord:
    """A terminal stable record (no channels, ``stable_for_decay``)."""
    mass = get_mass(pdg_id)
    return HadronRecord(
        pdg_id=pdg_id,
        name=name,
        B=B,
        Q=Q,
        S=S,
        isospin=isospin,
        isospin3=isospin3,
        spin=0.0,
        parity="+",
        pole_mass=mass,
        nominal_width=0.0,
        mass_range=(mass, mass),
        decay_channels=(),
        stable_for_decay=True,
        provenance=None,
    )


# ----------------------------------------------------------------------
# 1. test_record_roundtrip
# ----------------------------------------------------------------------
def test_record_roundtrip(tmp_path):
    """A hand-constructed record set serializes to JSON and parses back
    to an equal catalog."""
    # The termination list is the pre-registered stable set (round 2
    # index, item 11): the vendored table's stable entries plus K0S/K0L.
    assert STABLE_STATES == frozenset({
        2212, 2112, -2212, -2112,      # p, n, pbar, nbar
        211, -211, 111,                # pi+, pi-, pi0
        321, -321,                     # K+, K-
        310, 130,                      # K0S, K0L
    })

    catalog = HadronCatalog([
        # pi+ and rho+ are isovector (I = 1): I3 = +1 -> isospin3 = 2,
        # which is what Q = I3 + (B + S)/2 requires for Q = +1 mesons.
        _stable(211, "pi+", 0, 1, 0, isospin=1.0, isospin3=2),
        _resonance(
            213, "rho+", 0, 1, 0, isospin=1.0, isospin3=2,
            channels=(
                _channel(213, (211, 111), 1.0, orbital_angular_momentum=1,
                         provenance=_tag("branching_fraction", 1.0)),
            ),
            provenance=_tag("pole_mass", get_mass(213)),
        ),
        _resonance(
            223, "omega", 0, 0, 0, isospin=0.0, isospin3=0,
            channels=(
                _channel(223, (211, -211, 111), 1.0, is_three_body=True),
            ),
        ),
    ])
    report = catalog.validate()
    assert isinstance(report, CatalogValidationReport)
    assert report.record_count == 3
    assert report.errors == (), report.errors
    assert report.warnings == (), report.warnings

    path = tmp_path / "catalog.json"
    catalog.to_json(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert len(payload["records"]) == 3

    loaded = HadronCatalog.from_json(path)
    assert isinstance(loaded, HadronCatalog)
    assert loaded.records() == catalog.records()
    assert loaded.get(213) == catalog.get(213)
    assert loaded.get(213).provenance == _tag("pole_mass", get_mass(213))
    assert loaded.get(213).decay_channels[0].provenance == _tag(
        "branching_fraction", 1.0
    )
    assert loaded.stable_states() == catalog.stable_states()
    assert 211 in loaded.stable_states()
    rho_channel = loaded.get(213).decay_channels[0]
    assert loaded.threshold(213, rho_channel) == _threshold(rho_channel)
    assert loaded.validate().errors == ()


# ----------------------------------------------------------------------
# 2. test_channel_conserves_bqs
# ----------------------------------------------------------------------
def test_channel_conserves_bqs():
    """A channel with a wrong daughter sum (a missing charge state)
    makes ``validate()`` report an error naming the channel."""
    # rho0 -> pi+ pi+: the pi- charge state is missing, so the daughter
    # sum (0, 2, 0) does not equal the parent (0, 0, 0).
    bad_channel = _channel(113, (211, 211), 1.0, orbital_angular_momentum=1)
    record = _resonance(
        113, "rho0", 0, 0, 0, isospin=1.0, isospin3=0,
        channels=(bad_channel,),
    )
    catalog = HadronCatalog([record])
    report = catalog.validate()

    assert report.record_count == 1
    assert report.errors, "a charge-violating channel must be an error"
    assert len(report.errors) == 1
    message = report.errors[0]
    assert "rho0" in message and "pdg 113" in message
    assert "(0, 2, 0)" in message and "(0, 0, 0)" in message
    # The violation is reported, never repaired: the channel keeps its
    # wrong daughter list.
    assert catalog.get(113).decay_channels[0].daughters == (211, 211)


# ----------------------------------------------------------------------
# 3. test_branching_sum_guard
# ----------------------------------------------------------------------
def test_branching_sum_guard():
    """A per-parent branching sum above 1 + 1e-9 is an error; a sum of
    0.9999999995 is a warning only."""
    # (a) 0.75 + 0.75 = 1.5, exceeding 1 by more than 1e-9.
    over = _resonance(
        113, "rho0", 0, 0, 0, isospin=1.0, isospin3=0,
        channels=(
            _channel(113, (211, -211), 0.75, orbital_angular_momentum=1),
            _channel(113, (111, 111), 0.75, orbital_angular_momentum=1),
        ),
    )
    report = HadronCatalog([over]).validate()
    assert len(report.errors) == 1
    assert "sum to 1.5" in report.errors[0]
    assert "rho0" in report.errors[0]
    assert not report.warnings

    # (b) a single channel at 0.9999999995 leaves an unrepresented
    # channel budget: warning, not an error (until card 16 tightens it).
    under = _resonance(
        113, "rho0", 0, 0, 0, isospin=1.0, isospin3=0,
        channels=(_channel(113, (211, -211), 0.9999999995,
                           orbital_angular_momentum=1),),
    )
    report = HadronCatalog([under]).validate()
    assert report.errors == (), report.errors
    assert len(report.warnings) == 1
    assert "unrepresented channel budget" in report.warnings[0]
    assert "rho0" in report.warnings[0]

    # (c) a saturated sum of exactly 1 is silent.
    saturated = _resonance(
        113, "rho0", 0, 0, 0, isospin=1.0, isospin3=0,
        channels=(_channel(113, (211, -211), 1.0, orbital_angular_momentum=1),),
    )
    report = HadronCatalog([saturated]).validate()
    assert report.errors == ()
    assert report.warnings == ()


# ----------------------------------------------------------------------
# 4. test_mass_range_consistency
# ----------------------------------------------------------------------
def test_mass_range_consistency():
    """A ``mass_range`` lower bound away from the lowest channel
    threshold is an error."""
    good_channel = _channel(113, (211, -211), 1.0, orbital_angular_momentum=1)
    highest = _threshold(good_channel)
    record = _resonance(
        113, "rho0", 0, 0, 0, isospin=1.0, isospin3=0,
        channels=(good_channel,),
    )
    # Sanity: the probe with the exact threshold lower bound is valid.
    assert HadronCatalog([record]).validate().errors == ()

    lowered = HadronRecord(
        **{
            **record.__dict__,
            "mass_range": (highest - 0.01, highest + MASS_RANGE_SPAN),
        }
    )
    report = HadronCatalog([lowered]).validate()
    assert len(report.errors) == 1
    assert "mass_range[0]" in report.errors[0]
    assert "rho0" in report.errors[0]

    # The upper bound must stay at or above the lower bound.
    inverted = HadronRecord(
        **{
            **record.__dict__,
            "mass_range": (highest, highest - 1e-9),
        }
    )
    report = HadronCatalog([inverted]).validate()
    assert any("mass_range[1]" in e for e in report.errors)


# ----------------------------------------------------------------------
# 5. test_stable_termination
# ----------------------------------------------------------------------
def test_stable_termination():
    """A ``stable_for_decay`` record with a non-empty ``decay_channels``
    is an error; so is the iff converse (channels demand instability)."""
    # (a) terminal flag with declared channels.  The record carries the
    # matching mass_range for its channel so that the termination rule
    # is the only violated one.
    channel = _channel(221, (211, -211), 1.0)
    thr = _threshold(channel)
    stable_with_channels = HadronRecord(
        pdg_id=221,
        name="eta",
        B=0,
        Q=0,
        S=0,
        isospin=0.0,
        isospin3=0,
        spin=0.0,
        parity="-",
        pole_mass=get_mass(221),
        nominal_width=0.0,
        mass_range=(thr, thr),
        decay_channels=(channel,),
        stable_for_decay=True,
        provenance=None,
    )
    report = HadronCatalog([stable_with_channels]).validate()
    assert len(report.errors) == 1
    assert "stable_for_decay is True" in report.errors[0]
    assert "eta" in report.errors[0]

    # (b) the iff converse: no channels but declared unstable.
    unstable_without_channels = HadronRecord(
        **{
            **_stable(221, "eta", 0, 0, 0, isospin=0.0, isospin3=0).__dict__,
            "stable_for_decay": False,
        }
    )
    report = HadronCatalog([unstable_without_channels]).validate()
    assert len(report.errors) == 1
    assert "stable_for_decay is False" in report.errors[0]


# ----------------------------------------------------------------------
# 6. test_isospin3_consistency
# ----------------------------------------------------------------------
def test_isospin3_consistency():
    """A record with ``Q != I3 + (B + S)/2`` is an error."""
    # pi+ with isospin3 = 1 (I3 = +1/2) instead of +2 (I3 = +1):
    # Q = 1 but I3 + (B + S)/2 = 1/2 under the stored 2*I3 convention.
    bad_isospin = HadronRecord(
        **{
            **_stable(211, "pi+", 0, 1, 0, isospin=1.0, isospin3=2).__dict__,
            "isospin3": 1,
        }
    )
    report = HadronCatalog([bad_isospin]).validate()
    assert len(report.errors) == 1
    assert "isospin3" in report.errors[0]
    assert "pi+" in report.errors[0]

    # The consistent pi+ (I = 1, I3 = +1, isospin3 = 2) carries no
    # isospin error.
    good = _stable(211, "pi+", 0, 1, 0, isospin=1.0, isospin3=2)
    assert HadronCatalog([good]).validate().errors == ()


# ----------------------------------------------------------------------
# 7. test_pdg_agreement
# ----------------------------------------------------------------------
def test_pdg_agreement():
    """A record whose ``pole_mass`` contradicts the vendored table by
    more than ``1e-6`` GeV is an error; within the tolerance it passes."""
    # (a) a 1e-3 GeV contradiction is an error.
    contradicted = _resonance(
        113, "rho0", 0, 0, 0, isospin=1.0, isospin3=0,
        channels=(_channel(113, (211, -211), 1.0, orbital_angular_momentum=1),),
        pole_mass=get_mass(113) + 1e-3,
    )
    report = HadronCatalog([contradicted]).validate()
    assert len(report.errors) == 1
    assert "pole_mass" in report.errors[0]
    assert "rho0" in report.errors[0]

    # (b) a 5e-7 refinement agrees within the 1e-6 tolerance.
    refined = _resonance(
        113, "rho0", 0, 0, 0, isospin=1.0, isospin3=0,
        channels=(_channel(113, (211, -211), 1.0, orbital_angular_momentum=1),),
        pole_mass=get_mass(113) + 5e-7,
    )
    report = HadronCatalog([refined]).validate()
    assert report.errors == (), report.errors
