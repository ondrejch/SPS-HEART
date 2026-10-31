"""The R23 resonance-only p+p builder: pre-registered behavior pins.

Task ``TASK-20260913-r2-run-r23-resonance-only-pp-ablation`` (Round 2,
card 13; plan section 8 run R23, line 567).  Three hermetic,
small-N tests on :mod:`na49_mixing.resonance_pp_builder` (no
staged-pipeline generation at scale, no external tables):

- ``test_bit_identity_draws``: at a fixed seed, 1,000 events each from
  the legacy staged builder (``PPEventBuilder``) and the
  ``ResonancePPBuilder`` have identical topology selection, exchange
  masses, and spectral-state draws (the source layer is bit-identical
  by construction: the same per-event generator, the same draw
  sequence, exact float equality).

- ``test_kernel_switch``: a hand-configured source at exactly
  ``2.680194110727528 GeV`` (the M8-frozen ``continuum.match_mass``)
  decays through the catalog-driven kernel (the boundary is inclusive
  on the new-kernel side) and a source at ``2.680194110727529 GeV``
  through the legacy kernel; the legacy-kernel branch is bit-identical
  to the legacy builder at the same seed (the same ``SourceDecayer``
  construction on the same fresh per-event generator state), and the
  catalog-driven branch consumes no uniform from the shared generator.

- ``test_event_conservation``: 10,000 events (fixed seed) conserve
  ``B = 2`` and ``Q = 2`` at every event, and every source's final set
  conserves the source's ``(B, Q, S) = (1, 1, 0)``; both kernels are
  exercised (the mass sampler spans the boundary on both sides).

Hermetic: the committed catalog, the legacy_v1 manifest's M8 block, and
the vendored PDG table only; no NA49/SPS external tables, no run-dir
writes.  The generation budgets are small (1,000 + 1,000 + 10,000
events) and deterministic for the committed seeds and catalog.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from na49_mixing.continuum_channels import ContinuumParameters
from na49_mixing.event_builder_pp import PPEventBuilder, PPExcitationConfig
from na49_mixing.exchange_model import ReggeExcitationParameters
from na49_mixing.excitation_kinematics import FourVector
from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.quantum_numbers import QuantumNumbers
from na49_mixing.resonance_pp_builder import (
    KERNEL_BOUNDARY_M8,
    KERNEL_LABEL,
    LEGACY_LABEL,
    ROUND2_MASTER_SEED,
    STABLE_IDENTITY_FAMILY,
    ResonancePPBuilder,
)
from na49_mixing.source_event import SourceRecord

NESTED_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
MANIFEST_PATH = NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json"

#: The pre-registered test seeds (the R23 replica-0 seed convention).
BIT_IDENTITY_SEED = 20300913
BIT_IDENTITY_EVENTS = 1_000
CONSERVATION_SEED = 20300913
CONSERVATION_EVENTS = 10_000
KERNEL_TEST_SEED = 20300913
KERNEL_TEST_EVENT_ID = 777


def _m8_frozen_parameters() -> dict:
    """The M8 frozen ten-parameter set, verbatim from the manifest."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return dict(
        manifest["frozen_fit_outputs"]["m8_frozen_pp_parameters"]
    )


def _m8_config() -> PPExcitationConfig:
    """The M8-frozen p+p config (the R01 driver's application rule)."""
    values = _m8_frozen_parameters()
    assert values["continuum.match_mass"] == KERNEL_BOUNDARY_M8
    regge = ReggeExcitationParameters()
    continuum = ContinuumParameters()
    for name, value in values.items():
        group, field_name = name.split(".", 1)
        if group == "regge":
            regge = dataclasses.replace(regge, **{field_name: float(value)})
        else:
            continuum = dataclasses.replace(
                continuum, **{field_name: float(value)}
            )
    assert float(continuum.match_mass) == KERNEL_BOUNDARY_M8
    return PPExcitationConfig(
        production_mode="regge", regge=regge, continuum=continuum
    )


@pytest.fixture(scope="module")
def m8_config() -> PPExcitationConfig:
    return _m8_config()


@pytest.fixture(scope="module")
def catalog() -> HadronCatalog:
    return HadronCatalog.from_json(CATALOG_PATH)


def _event_rng(seed: int, event_id: int) -> np.random.Generator:
    """The legacy staged p+p per-event seed rule (the R01 convention)."""
    return np.random.default_rng(
        np.random.SeedSequence(int(seed), spawn_key=(int(event_id),))
    )


def _source_layer(event) -> dict:
    """The source-layer quantities of one generated event: the topology
    selection, the exchange masses, and the spectral-state realization
    (the momentum transfer and the source four-momenta)."""
    topology = [
        (s.source_id, s.role, s.quantum_numbers.as_tuple())
        for s in event.sources
    ]
    exchange_masses = [
        (float(s.metadata["sampled_mass"]),) for s in event.sources
    ]
    spectral_states = [
        (float(s.metadata["t"]), tuple(s.p4.as_dict().values()))
        for s in event.sources
    ]
    return {
        "topology": topology,
        "exchange_masses": exchange_masses,
        "spectral_states": spectral_states,
    }


def test_bit_identity_draws(m8_config, catalog):
    """1,000 events at a fixed seed: the legacy staged builder and the
    ``ResonancePPBuilder`` draw identical topology selections, exchange
    masses, and spectral-state realizations (the source layer is
    bit-identical by construction; exact float equality)."""
    legacy_layers = []
    new_layers = []
    for event_id in range(BIT_IDENTITY_EVENTS):
        legacy = PPEventBuilder(
            m8_config, _event_rng(BIT_IDENTITY_SEED, event_id)
        )
        legacy_event = legacy.generate(event_id)
        builder = ResonancePPBuilder(
            m8_config,
            catalog,
            seed=BIT_IDENTITY_SEED,
            event_id=event_id,
            master_seed=ROUND2_MASTER_SEED,
            replica_index=0,
        )
        new_event = builder.generate().event
        legacy_layers.append(_source_layer(legacy_event))
        new_layers.append(_source_layer(new_event))
    for layer in ("topology", "exchange_masses", "spectral_states"):
        assert legacy_layers[0][layer] == new_layers[0][layer]
        assert legacy_layers == new_layers, (
            f"the {layer} layer of the source layer is not bit-identical "
            "between the legacy staged builder and the "
            "ResonancePPBuilder at the same seed"
        )
    # Both builders carry two (1, 1, 0) sources per event.
    assert legacy_layers[0]["topology"] == [
        (0, "projectile", (1, 1, 0)), (1, "target", (1, 1, 0)),
    ]


def test_kernel_switch(m8_config, catalog):
    """The boundary is inclusive on the new-kernel side: a source at
    exactly 2.680194110727528 GeV decays through the catalog-driven
    kernel, a source at 2.680194110727529 GeV through the legacy
    kernel; the legacy branch is bit-identical to the legacy builder at
    the same seed and the catalog-driven branch consumes no uniform
    from the shared generator."""

    def make_source(mass: float) -> SourceRecord:
        return SourceRecord(
            0,
            "projectile",
            FourVector(float(mass), 0.0, 0.0, 0.0),
            QuantumNumbers(1, 1, 0),
            collision_index=1,
            metadata={"sampled_mass": float(mass), "t": 0.0},
        )

    boundary = KERNEL_BOUNDARY_M8
    at_boundary = make_source(boundary)
    above_boundary_mass = 2.680194110727529
    above_boundary = make_source(above_boundary_mass)
    assert above_boundary_mass > boundary

    # -- the new-kernel side (inclusive boundary) ------------------------
    builder_at = ResonancePPBuilder(
        m8_config,
        catalog,
        seed=KERNEL_TEST_SEED,
        event_id=KERNEL_TEST_EVENT_ID,
        master_seed=ROUND2_MASTER_SEED,
        replica_index=0,
    )
    shared_state_before = builder_at.rng.bit_generator.state
    finals_at, vertices_at, record_at = builder_at.decay_source(at_boundary)
    assert record_at.kernel == KERNEL_LABEL
    assert record_at.boundary_new_kernel is True
    assert record_at.record_pdg is not None
    assert record_at.n_open_records > 0
    assert finals_at, "the boundary source must decay into final particles"
    total = (0, 0, 0)
    for final in finals_at:
        qn = final.quantum_numbers
        total = (total[0] + qn.B, total[1] + qn.Q, total[2] + qn.S)
    assert total == (1, 1, 0)
    # the catalog-driven branch never touches the shared generator
    assert builder_at.rng.bit_generator.state == shared_state_before

    # -- the legacy-kernel side (above the boundary) ---------------------
    legacy = PPEventBuilder(
        m8_config, _event_rng(KERNEL_TEST_SEED, KERNEL_TEST_EVENT_ID)
    )
    expected = legacy.decayer.decay(above_boundary, 0)
    builder_above = ResonancePPBuilder(
        m8_config,
        catalog,
        seed=KERNEL_TEST_SEED,
        event_id=KERNEL_TEST_EVENT_ID,
        master_seed=ROUND2_MASTER_SEED,
        replica_index=0,
    )
    got, vertices_above, record_above = builder_above.decay_source(
        above_boundary
    )
    assert record_above.kernel == LEGACY_LABEL
    assert record_above.boundary_new_kernel is False
    assert record_above.record_pdg is None
    assert vertices_above is None
    assert [f.as_dict() for f in got] == [f.as_dict() for f in expected], (
        "the legacy-kernel branch is not bit-identical to the legacy "
        "builder at the same seed"
    )
    # the boundary semantics are exact float comparisons on the source
    # mass: mass == boundary takes the new kernel, mass > boundary the
    # legacy kernel
    assert boundary < above_boundary_mass


def test_event_conservation(m8_config, catalog):
    """10,000 events (fixed seed) conserve ``B = 2`` and ``Q = 2`` at
    every event and ``(B, Q, S) = (1, 1, 0)`` per source; both kernels
    are exercised."""
    n_resonance = n_legacy = 0
    for event_id in range(CONSERVATION_EVENTS):
        builder = ResonancePPBuilder(
            m8_config,
            catalog,
            seed=CONSERVATION_SEED,
            event_id=event_id,
            master_seed=ROUND2_MASTER_SEED,
            replica_index=0,
        )
        result = builder.generate()
        event = result.event
        total_b = total_q = total_s = 0
        for particle in event.final_particles:
            qn = particle.quantum_numbers
            total_b += qn.B
            total_q += qn.Q
            total_s += qn.S
        assert (total_b, total_q, total_s) == (2, 2, 0), (
            f"event {event_id}: finals sum to "
            f"{(total_b, total_q, total_s)}, expected (2, 2, 0)"
        )
        for record in result.source_records:
            if record.kernel == KERNEL_LABEL:
                n_resonance += 1
            else:
                n_legacy += 1
            assert (record.boundary_new_kernel, record.kernel) in (
                (True, KERNEL_LABEL), (False, LEGACY_LABEL),
            )
    assert n_resonance > 0 and n_legacy > 0, (
        "the pre-registered 10,000-event sample must exercise both "
        f"kernels (resonance {n_resonance}, legacy {n_legacy})"
    )
