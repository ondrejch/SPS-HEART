"""Coverage extension for state-type labeling and the per-state
color-singlet check (Round 1, card
TASK-20260911-r1-allowed-state-type-validation).

``tests/test_state_validation.py`` holds the six card-named deterministic
tests.  This file extends coverage of the same committed behavior without
touching that file:

- the closed-set assertion of ``STATE_TYPES`` (a ``frozenset`` with exactly
  the five plan section 6, task 7 labels);
- ``validate_state_type`` rejection of further outside labels (spacing
  variants of ``"hadron"``) and the ``ValueError`` hierarchy of both
  rejection exceptions;
- ``is_color_singlet`` boundary cases the six-test file leaves open:
  integer-valued float triples, fractional baryon number with integer
  charge, fractional strangeness, ``bool`` inputs, and non-finite values;
- the *missing label* versus *invalid label* distinction on staged records:
  a record that parses with ``state_type = None`` (the additive field's
  default) passes validation, while a record carrying a label outside the
  set fails with ``UnrecognizedStateType``;
- the second ``NonSingletState`` variants: a final particle with fractional
  baryon number ``(B, Q, S) = (1/3, 2/3, 0)`` and a ``SourceRecord`` with
  non-integer charge;
- construction stamping on a small staged p+p sample (hadron finals,
  ``continuum_resonance`` sources), a small staged p+A sample (the
  ``nuclear_remnant`` spectator path of the ``baryon_nucleus`` stage), and
  a small legacy sample, plus the unit-level ``__post_init__`` stamping
  rules for each stamped path;
- the legacy ``Particle`` label propagation: ``copy`` preserves the label,
  ``set_id`` (the decay-daughter identity registration) stamps it, and the
  four-momentum merge in ``add`` re-asserts or fills in the label.

Samples are deterministic: the staged p+p sample reuses the smoke config
seed (``config/staged_excitation_smoke.json``, ``fit.enabled`` false); the
staged p+A sample reuses the C12 fixture of ``test_excitation_stages.py``
with seed 44; the legacy sample reuses the committed six-test fixture
(``delta0.gen``, seed 20260911).  All samples stay within the card bound
of 20 events.  Hermetic: no external tables beyond the vendored PDG mass
table.
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
from na49_mixing.event_builder_pA import PAEventBuilder, PAExcitationConfig
from na49_mixing.event_builder_pp import PPEventBuilder, PPExcitationConfig
from na49_mixing.excitation_kinematics import FourVector
from na49_mixing.exchange_model import (
    ReggeExcitationParameters,
    SimpleExcitationParameters,
)
from na49_mixing.glauber import NucleusSpec
from na49_mixing.generator import Generator
from na49_mixing.particle import Particle
from na49_mixing.pdg_database import get_mass
from na49_mixing.quantum_numbers import QuantumNumbers
from na49_mixing.reaction import Reaction
from na49_mixing.source_event import ExcitationEvent, FinalParticle, SourceRecord
from na49_mixing.state_validation import (
    STATE_TYPES,
    NonSingletState,
    UnrecognizedStateType,
    is_color_singlet,
    validate_state_type,
)

NESTED_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE_CONFIG = os.path.join(NESTED_ROOT, "config", "staged_excitation_smoke.json")
LEGACY_CARD = os.path.join(
    NESTED_ROOT, "na49_mixing", "generators", "michal", "deltas-to-pion", "delta0.gen"
)

# Card bound: <= 20 events per builder sample; coverage samples are smaller
# so the full suite stays fast.
STAGED_PP_EVENTS = 10
STAGED_PA_EVENTS = 5
LEGACY_EVENTS = 10

# Fixed seeds: p+p reuses the smoke config seed; p+A and legacy reuse the
# seeds of test_excitation_stages.py / test_state_validation.py.
STAGED_PA_SEED = 44
LEGACY_SEED = 20260911


# ----------------------------------------------------------------------
# 1. Closed-set assertion
# ----------------------------------------------------------------------
def test_state_types_is_closed_five_value_frozenset():
    """``STATE_TYPES`` is a ``frozenset`` holding exactly the five
    plan section 6, task 7 labels; no sixth value is reachable."""
    assert isinstance(STATE_TYPES, frozenset)
    assert len(STATE_TYPES) == 5
    assert STATE_TYPES == frozenset({
        "hadron",
        "established_resonance",
        "continuum_resonance",
        "exchange",
        "nuclear_remnant",
    })
    # No sixth value is reachable: over a probe pool mixing the five
    # labels with strings outside the set, the set of accepted labels is
    # exactly the closed set.
    probe_labels = sorted(STATE_TYPES) + [
        "gluon", "quark", "", "hadron ", "HADRON", "hadrons", "parton",
        "string", "nuclear-remnant", "continuum_resonances",
    ]
    accepted: set = set()
    for label in probe_labels:
        try:
            validate_state_type(label)
            accepted.add(label)
        except UnrecognizedStateType:
            pass
    assert accepted == STATE_TYPES


# ----------------------------------------------------------------------
# 2. validate_state_type rejection + exception hierarchy
# ----------------------------------------------------------------------
def test_validate_state_type_rejection_boundaries():
    """Both rejection exceptions subclass ``ValueError``, and
    ``validate_state_type`` rejects spacing variants of a legal label."""
    assert issubclass(UnrecognizedStateType, ValueError)
    assert issubclass(NonSingletState, ValueError)
    # The six-test file covers "gluon", "quark", "", "HADRON", "hadrons",
    # "parton", "string"; pin the spacing variants here.
    for bad in ("gluon", "quark", "", "hadron ", " hadron"):
        with pytest.raises(UnrecognizedStateType):
            validate_state_type(bad)
    # A label outside the set fails even when spelled like one of the five.
    for bad in ("nuclear-remnant", "continuum_resonances"):
        with pytest.raises(UnrecognizedStateType):
            validate_state_type(bad)
    # And the error names the offending label.
    with pytest.raises(UnrecognizedStateType, match="nuclear-remnant"):
        validate_state_type("nuclear-remnant")


# ----------------------------------------------------------------------
# 3. is_color_singlet boundaries
# ----------------------------------------------------------------------
def test_is_color_singlet_boundaries():
    """Boundary inputs the six-test file leaves open.

    Integer-valued float triples are singlets (the runtime contract is
    "integer-valued number"); fractional baryon number, fractional
    strangeness, ``bool`` inputs, and non-finite values are not.
    """
    # The fractional/float/bool/non-finite probes pass floats and bools
    # where the signature annotates ``int`` on purpose: the runtime
    # contract is "integer-valued number", and the boundary probes need
    # non-integer and non-Integral inputs to be rejected.
    # Integer-valued float triples.
    assert is_color_singlet(1.0, 2.0, 0.0) is True     # type: ignore[arg-type]
    assert is_color_singlet(0.0, 1.0, -1.0) is True    # type: ignore[arg-type]
    assert is_color_singlet(-1.0, -1.0, 0.0) is True   # type: ignore[arg-type]
    # Fractional baryon number with integer charge (a quark probe).
    assert is_color_singlet(1 / 3, 1, 0) is False      # type: ignore[arg-type]
    assert is_color_singlet(-1 / 3, -1, 0) is False    # type: ignore[arg-type]
    # Fractional strangeness.
    assert is_color_singlet(1, 1, 0.5) is False        # type: ignore[arg-type]
    # bool is a bool, not an integer value, in this contract (either
    # literal; the check never treats bool as int).
    assert is_color_singlet(True, 1, 0) is False
    assert is_color_singlet(1, True, 0) is False
    assert is_color_singlet(1, 1, False) is False
    # Non-finite values are not integer values.
    assert is_color_singlet(float("nan"), 1, 0) is False   # type: ignore[arg-type]
    assert is_color_singlet(1, float("inf"), 0) is False   # type: ignore[arg-type]
    assert is_color_singlet(1, 1, float("-inf")) is False  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# 4. Missing label vs invalid label on staged records
# ----------------------------------------------------------------------
def _on_shell_proton() -> FinalParticle:
    """A final proton at rest: on shell, balanced, label ``hadron``."""
    return FinalParticle(
        particle_id=0,
        pdg_id=2212,
        p4=FourVector(get_mass(2212), 0.0, 0.0, 0.0),
        source_id=-1,
        parent_id=None,
        channel_family="final",
    )


def _probe_event(records_kind: str, final: FinalParticle, source: SourceRecord | None) -> ExcitationEvent:
    """A single-source/single-final probe event that balances exactly."""
    p4 = FourVector(get_mass(2212), 0.0, 0.0, 0.0)
    sources = [source] if source is not None else []
    return ExcitationEvent(
        event_id=0,
        stage=f"{records_kind}_label_probe",
        incoming=[p4],
        incoming_quantum_numbers=QuantumNumbers(1, 1, 0),
        sources=sources,
        final_particles=[final],
    )


def test_missing_label_is_not_a_validation_failure():
    """A record that parses with the additive field's ``None`` default is a
    *missing label*, not an *invalid label*: construction does not force a
    label, and validation passes such a record."""
    # A source without a continuum mass draw cannot be stamped and keeps
    # the missing label (``None``).
    source = SourceRecord(
        source_id=777,
        role="projectile",
        p4=FourVector(0.0, 0.0, 0.0, 0.0),
        quantum_numbers=QuantumNumbers(0, 1, 0),
    )
    assert source.state_type is None
    assert source.as_dict()["state_type"] is None
    # A final particle without a table identity parses the same way: the
    # construction does not force a label.  (Its full event validation is
    # a different failure, the documented unknown-PDG rejection of
    # ``quantum_numbers``; the missing-label semantics live in the
    # state-type check, which never raises on ``None``.)
    unresolvable = FinalParticle(
        particle_id=1,
        pdg_id=999999,
        p4=FourVector(1.0, 0.0, 0.0, 0.0),
        source_id=-1,
        parent_id=None,
        channel_family="final",
    )
    assert unresolvable.state_type is None
    # The committed model of a pre-field serialized record: the default
    # keeps old payloads parseable, and the serialized dict carries the
    # field with its resolved (here missing) value.
    # A record with a *resolvable* identity stamps instead of staying None,
    # so a missing label is always a genuinely unresolvable one.
    stamped = _on_shell_proton()
    assert stamped.state_type == "hadron"
    assert stamped.as_dict()["state_type"] == "hadron"
    # The missing label passes validation on the full event path: the
    # event balances and the state-type check skips the None label.
    event = _probe_event("missing", _on_shell_proton(), source)
    event.validate()
    # The final-particle path of the same distinction: a record parsed
    # back without the field (label ``None``) passes the state-type check.
    # The committed constructor re-stamps a resolvable identity, so the
    # parsed old record is modeled by clearing the label on a stamped one.
    parsed_old = _on_shell_proton()
    parsed_old.state_type = None
    event_final = _probe_event("missing_final", parsed_old, None)
    event_final.validate()


def test_invalid_label_is_a_validation_failure():
    """A record carrying a label outside the closed set fails
    ``validate`` with ``UnrecognizedStateType`` (final-particle and
    source paths)."""
    bad_final = _on_shell_proton()
    bad_final.state_type = "gluon"  # outside the closed set.
    event = _probe_event("invalid_final", bad_final, None)
    with pytest.raises(UnrecognizedStateType):
        event.validate()
    # The label survives the rejected validation (rejection, not repair).
    assert bad_final.state_type == "gluon"

    bad_source = SourceRecord(
        source_id=778,
        role="projectile",
        p4=FourVector(2.0, 0.0, 0.0, 0.0),
        quantum_numbers=QuantumNumbers(0, 1, 0),
        metadata={"sampled_mass": 2.7},
    )
    assert bad_source.state_type == "continuum_resonance"
    bad_source.state_type = "quark"  # outside the closed set.
    event2 = _probe_event("invalid_source", _on_shell_proton(), bad_source)
    with pytest.raises(UnrecognizedStateType):
        event2.validate()
    assert bad_source.state_type == "quark"


# ----------------------------------------------------------------------
# 5. NonSingletState variants
# ----------------------------------------------------------------------
class _FractionalBaryonParticle(FinalParticle):
    """A final particle whose ``(B, Q, S)`` carries fractional baryon
    number (the colored-object probe).  ``FinalParticle.quantum_numbers``
    is derived from the PDG id, so the probe subclasses and overrides the
    property; the label is passed explicitly as ``"hadron"`` so the
    constructor does not re-stamp it."""

    @property
    def quantum_numbers(self):
        return QuantumNumbers(B=1 / 3, Q=2 / 3, S=0)  # type: ignore[arg-type]


def test_non_singlet_fractional_baryon_rejected():
    """A final particle with valid label but fractional baryon number
    ``(B, Q, S) = (1/3, 2/3, 0)`` fails ``validate`` with
    ``NonSingletState`` (the second variant of the card probe)."""
    p4 = FourVector(get_mass(2112), 0.0, 0.0, 0.0)
    mutated = _FractionalBaryonParticle(
        particle_id=1,
        pdg_id=2112,
        p4=p4,
        source_id=-1,
        parent_id=None,
        channel_family="final",
        state_type="hadron",
    )
    assert mutated.state_type == "hadron"
    event = ExcitationEvent(
        event_id=1,
        stage="fractional_baryon_probe",
        incoming=[p4],
        incoming_quantum_numbers=QuantumNumbers(1, 1, 0),
        sources=[],
        final_particles=[mutated],
    )
    with pytest.raises(NonSingletState) as excinfo:
        event.validate()
    message = str(excinfo.value)
    assert "not a color singlet" in message
    assert "particle 1" in message
    assert "PDG 2112" in message
    # Rejection, not repair: the mutated quantum numbers survive.
    assert mutated.quantum_numbers == QuantumNumbers(B=1 / 3, Q=2 / 3, S=0)  # type: ignore[arg-type]


def test_non_singlet_source_rejected():
    """A ``SourceRecord`` whose own ``(B, Q, S)`` is not an integer triple
    fails ``validate`` with ``NonSingletState`` (the source path of the
    per-state audit)."""
    source = SourceRecord(
        source_id=5,
        role="projectile",
        p4=FourVector(2.0, 0.0, 0.0, 0.0),
        quantum_numbers=QuantumNumbers(0, 0.5, 0),  # type: ignore[arg-type]
    )
    event = ExcitationEvent(
        event_id=2,
        stage="source_singlet_probe",
        incoming=[FourVector(2.0, 0.0, 0.0, 0.0)],
        incoming_quantum_numbers=QuantumNumbers(1, 1, 0),
        sources=[source],
        final_particles=[],
    )
    with pytest.raises(NonSingletState) as excinfo:
        event.validate()
    message = str(excinfo.value)
    assert "source 5" in message
    assert "(0, 0.5, 0)" in message


# ----------------------------------------------------------------------
# Shared sample builders (module-scoped: each sample is built once)
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def staged_pp_events():
    """10 staged p+p events from the smoke config (fit disabled)."""
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
    return [builder.generate(i) for i in range(STAGED_PP_EVENTS)]


@pytest.fixture(scope="module")
def staged_pa_events():
    """5 staged p+A events from the C12 fixture of test_excitation_stages
    (the ``baryon_nucleus`` stage; seed 44)."""
    nucleus = NucleusSpec(
        name="C12",
        A=12,
        Z=6,
        profile="harmonic_oscillator",
        radius_fm=1.7,
        diffuseness_fm=0.5,
    )
    builder = PAEventBuilder(
        PAExcitationConfig(
            nucleus=nucleus,
            glauber_dr_fm=0.1,
            max_event_attempts=100,
        ),
        np.random.default_rng(STAGED_PA_SEED),
    )
    return [builder.generate(i) for i in range(STAGED_PA_EVENTS)]


@pytest.fixture(scope="module")
def legacy_events():
    """10 legacy events via the ``Generator`` path on the smallest bundled
    card (``delta0.gen``), cascaded through the soft-resonance library."""
    rng = np.random.default_rng(LEGACY_SEED)
    generator = Generator(reaction=Reaction(158.0), rng=rng)
    generator.set_all(LEGACY_CARD)
    decay_chain = dict(build_soft_resonance_decays(rng))
    events = []
    for _ in range(LEGACY_EVENTS):
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
# 6. Construction stamping
# ----------------------------------------------------------------------
def test_construction_stamping_unit_paths():
    """The ``__post_init__`` / constructor stamp rules for each stamped
    path: hadron, nuclear_remnant (spectator channel family),
    continuum_resonance (sampled mass draw), and the unresolvable
    ``None`` default."""
    # A decay daughter of a resolved channel stamps hadron.
    daughter = FinalParticle(
        particle_id=0,
        pdg_id=211,
        p4=FourVector(get_mass(211), 0.0, 0.0, 0.0),
        source_id=1,
        parent_id=None,
        channel_family="resolved_nonstrange",
    )
    assert daughter.state_type == "hadron"
    # A spectator nucleon stamps nuclear_remnant via the channel family.
    spectator = FinalParticle(
        particle_id=1,
        pdg_id=2212,
        p4=FourVector(get_mass(2212), 0.0, 0.0, 0.0),
        source_id=2,
        parent_id=None,
        channel_family="spectator",
    )
    assert spectator.state_type == "nuclear_remnant"
    # A source with a continuum mass draw stamps continuum_resonance.
    continuum_source = SourceRecord(
        source_id=3,
        role="projectile",
        p4=FourVector(2.7, 0.0, 0.0, 0.0),
        quantum_numbers=QuantumNumbers(1, 1, 0),
        metadata={"sampled_mass": 2.7, "t": -1.0},
    )
    assert continuum_source.state_type == "continuum_resonance"
    # A source with neither draw nor resonance id keeps the missing label.
    unresolved_source = SourceRecord(
        source_id=4,
        role="projectile",
        p4=FourVector(2.7, 0.0, 0.0, 0.0),
        quantum_numbers=QuantumNumbers(1, 1, 0),
    )
    assert unresolved_source.state_type is None
    # The legacy constructor stamps from the PDG identity the same way.
    assert Particle(0.0, 0.0, 0.0, 1.0, get_mass(211), 211).state_type == "hadron"
    assert Particle(0.0, 0.0, 0.0, 0.0, 0.0, 0).state_type is None


def test_staged_pp_sample_stamped(staged_pp_events):
    """Every staged p+p state carries a label from the set: hadron finals
    and continuum_resonance sources (the continuum source path)."""
    n_states = 0
    n_sources = 0
    for event in staged_pp_events:
        for particle in event.final_particles:
            assert particle.state_type in STATE_TYPES
            assert particle.state_type == "hadron"
            n_states += 1
        for source in event.sources:
            assert source.state_type in STATE_TYPES
            # Every staged source carries a continuum mass draw.
            assert source.state_type == "continuum_resonance"
            assert "sampled_mass" in source.metadata
            n_sources += 1
            n_states += 1
        assert len(event.final_particles) > 0
        assert n_sources >= 2  # two-source excitation per p+p event
        event.validate()  # includes the per-state singlet audit
    assert n_states > 0


def test_staged_pa_sample_stamps_nuclear_remnants(staged_pa_events):
    """Every p+A spectator (nuclear remnant) carries ``nuclear_remnant``,
    the finals carry ``hadron``, the sources carry ``continuum_resonance``,
    and the full per-event validation passes."""
    n_spectators = 0
    observed_spectator_labels = set()
    observed_final_labels = set()
    observed_source_labels = set()
    for event in staged_pa_events:
        for spectator in event.spectators:
            assert spectator.state_type in STATE_TYPES
            assert spectator.channel_family == "spectator"
            observed_spectator_labels.add(spectator.state_type)
            n_spectators += 1
        for particle in event.final_particles:
            assert particle.state_type in STATE_TYPES
            observed_final_labels.add(particle.state_type)
        for source in event.sources:
            assert source.state_type in STATE_TYPES
            assert "sampled_mass" in source.metadata
            observed_source_labels.add(source.state_type)
        event.validate()
    # The nuclear_remnant path is exercised: C12 events at seed 44 carry
    # remnant nucleons, and every one of them is stamped.
    assert n_spectators >= 1
    assert observed_spectator_labels == {"nuclear_remnant"}
    assert observed_final_labels == {"hadron"}
    assert observed_source_labels == {"continuum_resonance"}


def test_legacy_sample_stamped(legacy_events):
    """Every legacy ``Particle`` of the ``delta0.gen`` sample carries a
    label from the closed set (this sample resolves every PDG identity, so
    no missing label occurs)."""
    observed = set()
    n_states = 0
    for event in legacy_events:
        for particle in event.particles():
            assert particle.state_type is None or particle.state_type in STATE_TYPES
            observed.add(particle.state_type)
            n_states += 1
    assert n_states > 0
    assert observed == {"hadron"}


# ----------------------------------------------------------------------
# 7. Legacy Particle label propagation
# ----------------------------------------------------------------------
def test_particle_copy_preserves_state_type():
    """``Particle.copy`` carries the stamp across (stamped and missing
    labels alike)."""
    stamped = Particle(1.0, 0.0, 0.0, 1.0, get_mass(211), 211)
    assert stamped.state_type == "hadron"
    clone = Particle.copy(stamped)
    assert clone is not stamped
    assert clone.state_type == stamped.state_type == "hadron"

    missing = Particle(0.0, 0.0, 0.0, 0.0, 0.0, 0)
    assert missing.state_type is None
    clone_missing = Particle.copy(missing)
    assert clone_missing.state_type is None
    assert clone_missing is not missing


def test_set_id_stamps_decay_daughter_label():
    """A decay daughter with no identity yet parses with a missing label;
    registering the identity via ``set_id`` stamps it, and a non-table
    identity leaves it missing (not invalid)."""
    daughter = Particle(0.5, 0.0, 0.0, 0.0, 0.0, 0)
    assert daughter.state_type is None
    daughter.set_id(311)  # K0: table hadron with integer (B, Q, S).
    assert daughter.state_type == "hadron"
    daughter.set_id(999999)  # outside the vendored mass table.
    assert daughter.state_type is None


def test_add_reasserts_label_after_merge():
    """The four-momentum merge in ``Particle.add`` keeps an existing stamp
    and fills in a missing one once the object carries a table identity."""
    parent = Particle(1.0, 0.0, 0.0, 1.0, get_mass(2212), 2212)
    assert parent.state_type == "hadron"
    other = Particle(0.0, 1.0, 0.0, -1.0, get_mass(-211), -211)
    parent.add(other)
    assert parent.state_type == "hadron"

    # Fill-in branch: a record that lost its label (simulating a legacy
    # serialized record) is filled at the next merge.
    lost = Particle(0.0, 0.0, 0.0, 0.0, get_mass(2212), 2212)
    lost.state_type = None
    lost.add(Particle(0.0, 0.0, 0.1, 0.0, get_mass(211), 211))
    assert lost.state_type == "hadron"
