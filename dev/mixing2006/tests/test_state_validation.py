"""State-type labeling and per-state color-singlet validation (Round 1,
card TASK-20260911-r1-allowed-state-type-validation).

The seven deterministic tests pin:

- the ``(pdg_id, role, metadata) -> state_type`` mapping of
  ``na49_mixing.state_validation.state_type_of`` over all five allowed
  labels plus three ``UnrecognizedStateType`` raises;
- the closed-set rejection of ``validate_state_type``;
- the integer ``(B, Q, S)`` rule of ``is_color_singlet``;
- construction-time stamping on a staged p+p sample (every
  ``FinalParticle`` and ``SourceRecord`` carries a label from the set,
  and ``ExcitationEvent.validate`` passes);
- construction-time stamping on the legacy ``Generator`` path (every
  ``Particle`` carries a label from the set, or ``None`` as a missing
  label, and the state-type check passes);
- the per-state singlet rejection: a hand-constructed ``FinalParticle``
  with a mutated ``(B, Q, S) = (1, 1.5, 0)`` fails ``validate`` with
  ``NonSingletState``;
- the hand-constructed ``SourceRecord`` stamp (REV-895f4b1-01 fix): a
  source whose metadata carries a named PDG resonance id stamps
  ``established_resonance``, continuum sources keep
  ``continuum_resonance`` (``sampled_mass``-first precedence), and an
  unresolvable source keeps ``None``.

Samples are deterministic: the staged sample comes from
``config/staged_excitation_smoke.json`` (``fit.enabled`` false) with the
config seed and 20 events; the legacy sample from the smallest bundled
generator card (``delta0.gen``) with seed 20260911 and 20 events,
cascaded through ``build_soft_resonance_decays``.

Hermetic: no external tables beyond the PDG mass table already vendored
in ``na49_mixing/``.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from na49_mixing.continuum_channels import ContinuumParameters
from na49_mixing.decays import build_soft_resonance_decays
from na49_mixing.event import Event
from na49_mixing.event_builder_pp import PPEventBuilder, PPExcitationConfig
from na49_mixing.excitation_kinematics import FourVector
from na49_mixing.exchange_model import (
    ReggeExcitationParameters,
    SimpleExcitationParameters,
)
from na49_mixing.generator import Generator
from na49_mixing.particle import Particle
from na49_mixing.pdg_database import get_mass, get_particle
from na49_mixing.quantum_numbers import QuantumNumbers, quantum_numbers
from na49_mixing.reaction import Reaction
from na49_mixing.source_event import ExcitationEvent, FinalParticle, SourceRecord
from na49_mixing.state_validation import (
    STATE_TYPES,
    NonSingletState,
    UnrecognizedStateType,
    is_color_singlet,
    state_type_of,
    validate_state_type,
)

NESTED_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE_CONFIG = os.path.join(NESTED_ROOT, "config", "staged_excitation_smoke.json")
LEGACY_CARD = os.path.join(
    NESTED_ROOT, "na49_mixing", "generators", "michal", "deltas-to-pion", "delta0.gen"
)

N_EVENTS = 20  # card bound: <= 20 events per builder sample
LEGACY_SEED = 20260911


# ----------------------------------------------------------------------
# 1. test_state_type_of_mapping
# ----------------------------------------------------------------------
def test_state_type_of_mapping():
    """Each of the five state types has at least one concrete mapping
    example, and invalid combinations raise ``UnrecognizedStateType``."""
    # The set is closed and holds exactly the plan section 6, task 7 labels.
    assert STATE_TYPES == frozenset({
        "hadron",
        "established_resonance",
        "continuum_resonance",
        "exchange",
        "nuclear_remnant",
    })

    cases = [
        # (pdg_id, role, metadata) -> expected state_type
        # stable hadrons of the PDG mass table with integer (B, Q, S)
        ((211, "final_particle", None), "hadron"),        # pi+
        ((-2212, "final_particle", None), "hadron"),      # pbar
        ((3122, "final_particle", None), "hadron"),       # Lambda
        ((-321, "final_particle", None), "hadron"),       # K-
        # a source with a continuum mass draw is a continuum resonance
        (
            (0, "source", {"sampled_mass": 2.7, "t": -1.0}),
            "continuum_resonance",
        ),
        # a named resonance of the established table (Delta(1600)++,
        # non-zero width) is an established resonance
        ((202224, "source", None), "established_resonance"),
        ((2214, "source", None), "established_resonance"),
        # a spectator nucleon of a p+A event is the nuclear remnant
        ((2212, "spectator", None), "nuclear_remnant"),
        ((2112, "spectator", None), "nuclear_remnant"),
        # an exchange record is labeled by role, not by a PDG identity
        ((0, "exchange_record", None), "exchange"),
    ]
    for (pdg_id, role, metadata), expected in cases:
        assert state_type_of(pdg_id, role, metadata) == expected, (
            f"state_type_of({pdg_id!r}, {role!r}, {metadata!r}) "
            f"!= {expected!r}"
        )

    invalid = [
        # unknown role
        (211, "decay_daughter", None),
        # PDG id outside the vendored mass table
        (999999, "final_particle", None),
        # source with neither a continuum mass draw nor a resonance id
        (0, "source", {"t": -1.0}),
    ]
    for pdg_id, role, metadata in invalid:
        with pytest.raises(UnrecognizedStateType):
            state_type_of(pdg_id, role, metadata)


# ----------------------------------------------------------------------
# 2. test_validate_state_type_rejects
# ----------------------------------------------------------------------
def test_validate_state_type_rejects():
    """``validate_state_type`` raises for every label outside the set."""
    for bad in ("gluon", "quark", "", "HADRON", "hadrons", "parton", "string"):
        with pytest.raises(UnrecognizedStateType):
            validate_state_type(bad)
    for good in sorted(STATE_TYPES):
        validate_state_type(good)  # does not raise


# ----------------------------------------------------------------------
# 3. test_is_color_singlet
# ----------------------------------------------------------------------
def test_is_color_singlet():
    """Integer ``(B, Q, S)`` triples are singlets; fractional ones are not.

    The fractional probes below pass floats where the signature annotates
    ``int`` on purpose: the runtime contract is "integer-valued number",
    and the colored-object probes need non-integer values to be rejected.
    """
    assert is_color_singlet(1, 1, 0) is True      # proton
    assert is_color_singlet(0, 1, 0) is True      # pi+
    assert is_color_singlet(0, -1, -1) is True    # K-
    assert is_color_singlet(1, 1, -1) is True     # integer triple (Lambda shape)
    assert is_color_singlet(1, 0, -1) is True     # Lambda: exact (B, Q, S)
    assert is_color_singlet(1 / 3, 2 / 3, 0) is False   # type: ignore[arg-type]  # colored quark
    assert is_color_singlet(0, 0.5, 0) is False   # non-integer charge
    assert is_color_singlet(1, 1.0, 0) is True    # integer-valued float charge
    assert is_color_singlet(2.0, 1, 0) is True    # type: ignore[arg-type]  # integer-valued float baryon


# ----------------------------------------------------------------------
# Shared sample builders
# ----------------------------------------------------------------------
def _staged_builder_and_events():
    """20 staged p+p events from the smoke config (fit disabled)."""
    config = json.loads(open(SMOKE_CONFIG, encoding="utf-8").read())
    assert config["fit"]["enabled"] is False
    builder = PPEventBuilder(
        PPExcitationConfig(
            beam_momentum=float(config["beam_momentum"]),
            production_mode="simple",
            simple=SimpleExcitationParameters(**config["simple"]),
            regge=ReggeExcitationParameters(**config["regge"]),
            continuum=ContinuumParameters(**config["continuum"]),
        ),
        np.random.default_rng(int(config["seed"])),
    )
    events = [builder.generate(i) for i in range(N_EVENTS)]
    return builder, events


def _legacy_events():
    """20 legacy events via the ``Generator`` path and the smallest
    legacy generator card, cascaded through the soft-resonance library."""
    rng = np.random.default_rng(LEGACY_SEED)
    generator = Generator(reaction=Reaction(158.0), rng=rng)
    generator.set_all(LEGACY_CARD)
    decay_chain = dict(build_soft_resonance_decays(rng))
    events = []
    for _ in range(N_EVENTS):
        primary = generator.generate()
        assert primary is not None, "legacy Generator failed to produce a primary"
        event = Event(rng=rng)
        event._decay_chain = dict(decay_chain)
        event.add_particle(primary)
        event.mark_primary(primary)
        event.decay_cascade(max_passes=8)
        events.append(event)
    return events


# ----------------------------------------------------------------------
# 4. test_stamped_staged_event
# ----------------------------------------------------------------------
def test_stamped_staged_event():
    """Every staged state carries a label from the set and validate passes."""
    _builder, events = _staged_builder_and_events()
    assert len(events) <= N_EVENTS
    n_states = 0
    for event in events:
        for particle in event.final_particles:
            assert particle.state_type in STATE_TYPES, (
                f"unstamped final particle PDG {particle.pdg_id}: "
                f"{particle.state_type!r}"
            )
            assert particle.state_type == "hadron"
            n_states += 1
        for source in event.sources:
            assert source.state_type in STATE_TYPES, (
                f"unstamped source {source.source_id}: {source.state_type!r}"
            )
            # Staged sources carry a continuum mass draw.
            assert source.state_type == "continuum_resonance"
            assert "sampled_mass" in source.metadata
            n_states += 1
        for spectator in event.spectators:
            assert spectator.state_type in STATE_TYPES
            assert spectator.state_type == "nuclear_remnant"
            n_states += 1
        assert n_states > 0
        event.validate()  # includes the per-state singlet audit


# ----------------------------------------------------------------------
# 5. test_stamped_legacy_event
# ----------------------------------------------------------------------
def test_stamped_legacy_event():
    """Every legacy ``Particle`` carries a label from the set or ``None``
    (missing label on a legacy record); the state-type check passes."""
    events = _legacy_events()
    n_states = 0
    n_missing = 0
    for event in events:
        for particle in event.particles():
            if particle.state_type is None:
                # A missing label is allowed only when the particle has no
                # table identity to stamp from; every particle of this
                # sample carries a real PDG id, so none may be missing.
                assert get_particle(particle.get_id()) is None, (
                    f"unstamped legacy particle PDG {particle.get_id()}"
                )
                n_missing += 1
            else:
                assert particle.state_type in STATE_TYPES
                assert particle.state_type == "hadron"
                validate_state_type(particle.state_type)
            qn = quantum_numbers(particle.get_id())
            assert is_color_singlet(qn.B, qn.Q, qn.S)
            n_states += 1
    assert n_states > 0
    assert n_missing == 0


# ----------------------------------------------------------------------
# 6. test_non_singlet_rejected
# ----------------------------------------------------------------------
class _MutatedSingletParticle(FinalParticle):
    """Hand-constructed final particle with a mutated ``(B, Q, S)``.

    ``FinalParticle.quantum_numbers`` derives from the PDG id and is
    read-only, so the probe subclasses and overrides the property to
    inject the non-integer triple the rejection test needs.  The label
    is passed explicitly as ``"hadron"`` so the constructor does not
    re-stamp it.
    """

    @property
    def quantum_numbers(self):
        # The mutated triple deliberately carries a non-integer charge.
        return QuantumNumbers(B=1, Q=1.5, S=0)  # type: ignore[arg-type]


def test_non_singlet_rejected():
    """A final particle with mutated (B, Q, S) = (1, 1.5, 0) fails
    ``validate`` with ``NonSingletState``."""
    p4 = FourVector(get_mass(2212), 0.0, 0.0, 0.0)
    mutated = _MutatedSingletParticle(
        particle_id=0,
        pdg_id=2212,
        p4=p4,
        source_id=-1,
        parent_id=None,
        channel_family="final",
        state_type="hadron",
    )
    assert mutated.state_type == "hadron"
    event = ExcitationEvent(
        event_id=0,
        stage="state_type_probe",
        incoming=[p4],
        incoming_quantum_numbers=QuantumNumbers(1, 1, 0),
        sources=[],
        final_particles=[mutated],
    )
    with pytest.raises(NonSingletState) as excinfo:
        event.validate()
    assert "(B, Q, S) = (1, 1.5, 0)" in str(excinfo.value)
    # The singlet rejection is a validation-time rejection, not a repair:
    # the particle keeps its mutated quantum numbers and its label.
    assert mutated.quantum_numbers == QuantumNumbers(B=1, Q=1.5, S=0)  # type: ignore[arg-type]
    assert mutated.state_type == "hadron"


# ----------------------------------------------------------------------
# 7. test_source_record_construction_stamping
# ----------------------------------------------------------------------
def test_source_record_construction_stamping():
    """A hand-constructed ``SourceRecord`` stamps from the PDG identity
    its metadata carries (REV-895f4b1-01): a named-resonance source
    stamps ``established_resonance``, continuum sources keep
    ``continuum_resonance`` (``sampled_mass``-first precedence), and a
    source with neither a resolvable identity nor a mass draw keeps the
    missing label ``None``."""
    # A source whose metadata names a table resonance (Delta(1232)++ and
    # N(1600)+, non-zero widths) stamps established_resonance; both
    # metadata keys are resolved.
    for pdg_id, key in ((2214, "pdg_id"), (202224, "resonance_pdg")):
        source = SourceRecord(
            source_id=0,
            role="source",
            p4=FourVector(get_mass(pdg_id), 0.0, 0.0, 0.0),
            quantum_numbers=quantum_numbers(pdg_id),
            metadata={key: pdg_id},
        )
        assert source.state_type == "established_resonance", (
            f"SourceRecord(metadata={{{key!r}: {pdg_id}}}) stamped "
            f"{source.state_type!r}"
        )

    # A continuum-drawn source keeps stamping continuum_resonance, with
    # and without a nucleon id beside the mass draw (the real p+A target
    # metadata shape; sampled_mass precedence is state_type_of's rule).
    continuum = SourceRecord(
        source_id=1,
        role="source",
        p4=FourVector(2.7, 0.0, 0.0, 0.0),
        quantum_numbers=QuantumNumbers(1, 1, 0),
        metadata={"sampled_mass": 2.7, "t": -1.0},
    )
    assert continuum.state_type == "continuum_resonance"
    pa_target = SourceRecord(
        source_id=2,
        role="target",
        p4=FourVector(1.1, 0.0, 0.0, 0.0),
        quantum_numbers=QuantumNumbers(1, 1, 0),
        metadata={"sampled_mass": 1.1, "target_pdg": 2212, "t": -0.5},
    )
    assert pa_target.state_type == "continuum_resonance"

    # A table id without a resonance width and without a mass draw
    # (a nucleon) leaves the label undeterminable: the missing label
    # None, never an invalid label.
    unresolvable = SourceRecord(
        source_id=3,
        role="source",
        p4=FourVector(2.0, 0.0, 0.0, 0.0),
        quantum_numbers=QuantumNumbers(1, 1, 0),
        metadata={"pdg_id": 2212},
    )
    assert unresolvable.state_type is None