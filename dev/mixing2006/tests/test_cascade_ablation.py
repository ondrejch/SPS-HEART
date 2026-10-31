"""Card tests for the isotropic-versus-aligned ablation runner.

Round 4, card 10 (``TASK-20260920-r4-isotropic-aligned-ablation-
runner``; plan section 22 task 27, line 1749, plan section 10 run R43,
lines 790-798).  The card's pre-registered deterministic tests against
the committed catalog (``config/v2_singlet/resonance_catalog.json``)
and the continuum family configuration
(``config/v2_singlet/continuum_resonances.json``):

- ``test_exchange_identity``: on a pinned seed and pinned exchange
  event, the exchange layer (topology, exchange kinematics, spectral
  state) is bit-identical across all three variants (the identical-
  exchange-events contract of plan line 798): the root state is the
  draw-free pure function of the exchange event, the exchange
  substreams are the house derivation shared across the variants, the
  event mapping is never mutated, and all three variants decay the
  same root.
- ``test_v2_pin``: the isotropic sequential variant is bit-identical
  to card 4's cascade at kappa = 0 with b_dec nominal (the reference
  loop drives card 4's own ``draw_decay`` over the attached pools
  through card 9's routing) on pinned seeds and pinned roots; at the
  off limit ``b_dec = 0`` the runner's line is bit-identical to card
  9's unattached ``run_cascade`` baseline.
- ``test_stream_disjointness``: the per-variant decay streams are
  disjoint (a draw on variant 0 never advances variant 1's stream) and
  deterministic (the same ``(master_seed, event_index, variant_id)``
  reproduces the stream byte-for-byte); the streams are keyed on the
  variant id and validated.
- ``test_support_boundary``: the support documentation is carried on
  the module (v1 single-step vs v2/v3 multi-vertex; v2/v3 support
  identical) in the machine-readable ``SUPPORT_BOUNDARY`` block (the
  pre-registration block shape card 14 consumes), with the
  pre-registered policy register and its validation edges.
- ``test_conservation``: every variant conserves (B, Q, S) exactly per
  event on the pinned fixtures (the v1 daughter sums, the v2/v3 line
  states against their records and families, and the open-channel
  sums at every line state).
- ``test_budget_per_variant``: v1 consumes the Round 2 single-step
  budget (1 channel + 2 kinematics + the kernel's documented 2 mass
  draws per resonance daughter); v2/v3 consume card 4's per-vertex
  budget (1 channel + 2 kinematics per line state), the alignment and
  form-factor attachments adding zero uniforms (the card 4 frozen
  budget, identical across the policies).
- ``test_layout_sync``: the consumed-surface register and the layout
  sync (the subprocess check from the monorepo root; ``logic_drift=0``).

plus the card's acceptance-(vi) freeze re-pin check
``test_freeze_manifest_repin``.

Hermetic: the committed config and catalog files, numpy, the standard
library, and the vendored ``na49_mixing`` modules only; no event
generation, no external tables, no Monte Carlo run.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.cascade_ablation import (
    ABLATION_POLICIES,
    ABLATION_VARIANTS,
    SUPPORT_BOUNDARY,
    W_DEC_REFERENCE_COS_THETA,
    W_DEC_REFERENCE_SIN_THETA,
    AblationPolicy,
    decay_variant,
    exchange_root_state,
    exchange_substreams,
    reference_direction,
    variant_stream,
)
from na49_mixing.cascade_alignment import (
    BARYONIC_KAPPA_DEFAULT,
    MESONIC_KAPPA_DEFAULT,
)
from na49_mixing.cascade_decay import (
    CATALOG_FAMILY_PREFIX,
    ISOTROPIC_ALIGNMENT_LABEL,
    KIND_CONTINUUM,
    KIND_ESTABLISHED_HADRON,
    KIND_ESTABLISHED_RESONANCE,
    MAX_CASCADE_DEPTH,
    CascadeChannel,
    channel_weight,
    draw_decay,
    enumerate_channels,
)
from na49_mixing.cascade_form_factor import (
    DECAY_FORM_FACTOR_B_DEFAULT,
    attach_to_channel_weight,
)
from na49_mixing.cascade_termination import (
    CascadeDepthGuardError,
    handoff_decay,
    is_catalog_handoff,
    run_cascade,
)
from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.resonance_widths import decay_momentum
from na49_mixing.rng_substreams import derive_substreams
from na49_mixing.spectral_density import SUPPORT_M_MAX, families_from_json
from tests._freeze_inventory import inventory_count as _inventory_count

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent          # dev/mixing2006
MONO_ROOT = NESTED_ROOT.parent.parent          # the monorepo root
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
FAMILIES_PATH = NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json"
MANIFEST_PATH = NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json"
SYNC_CHECKER = MONO_ROOT / "helpers" / "benchmarking" / "check_helpers_layout_sync.py"

#: The round 4 master seed (round 4 index item 9), fixed for the
#: deterministic tests.
MASTER_SEED = 20260920

BARYONIC_QN = (1, 1, 0)
STRANGE_BARYONIC_QN = (1, 0, -1)
MESONIC_QN = (0, 0, 0)

#: The pinned catalog roots: the Delta+ bit-identity fixture of card 4,
#: the pinned 2 GeV baryonic source record (the card-9 handoff
#: fixture), and the pinned 2 GeV mesonic source record.
CATALOG_ROOTS = (
    ("catalog:2214", 1.232, BARYONIC_QN),
    ("catalog:12212", 2.0, BARYONIC_QN),
    ("catalog:100113", 2.0, MESONIC_QN),
)

#: The pinned continuum roots of the accepted source-mass range (round
#: 4 index item 2), both families and the strange family.
CONTINUUM_ROOTS = (
    ("baryonic_np", 3.0, BARYONIC_QN),
    ("baryonic_np", 4.0, BARYONIC_QN),
    ("baryonic_np", 8.0, BARYONIC_QN),
    ("mesonic_neutral", 3.0, MESONIC_QN),
    ("mesonic_neutral", 4.0, MESONIC_QN),
    ("baryonic_strange", 4.0, STRANGE_BARYONIC_QN),
)

#: The pinned event indices of the per-variant stream keys.
PINNED_EVENTS = (7, 11, 23, 101)

#: The byte-unchanged frozen substrate of the card's provenance
#: section (verified on disk at this card's landing): cards 4-9's
#: cascade surfaces, the Round 2 kernel (the v1 handoff target and the
#: card-9 delegation), the builder, the card-6 blend routing, the
#: card-2 engine, the committed catalog, the continuum families
#: configuration, and the R32 driver in both trees.
FROZEN_SURFACES = {
    "dev/mixing2006/na49_mixing/cascade_decay.py":
        "d2a48049af3d6df968993c040033a74422ccd709f53a42bad0538ae5111bcc0c",
    "dev/mixing2006/na49_mixing/cascade_axes.py":
        "a7251a6b6c0ed4b9cfc56c0182c28a70efccb7a92a33b756090ff716a3ec3c61",
    "dev/mixing2006/na49_mixing/cascade_alignment.py":
        "0d0df347222423aaf6f02403e177867b57da588a5e40d641fd7b467e9c4a81f4",
    "dev/mixing2006/na49_mixing/cascade_form_factor.py":
        "b16077c9e3418b3f19b117aadf6fc8a8e838af5e9d37b1ab0bae89ccfb88f74c",
    "dev/mixing2006/na49_mixing/cascade_termination.py":
        "a62f6cce7ace4d6306b176bf12d61e28c22a0b5f1709a201d05a63fcf833da56",
    "dev/mixing2006/na49_mixing/resonance_decay.py":
        "6a1e8f0ad3c9e6e56645f919ba14060836ab27dd8e53235eca4434f9b1aa8210",
    "dev/mixing2006/na49_mixing/resonance_pp_builder.py":
        "4ab612609e1ee0bc1e037dcbb2815b6484306ad32402b7289e6636f8e829cdb1",
    "dev/mixing2006/na49_mixing/spectral_density.py":
        "382e048b9f66a35a6446e39d97e95a9fed885e0ac5ec297bfa19823e5acf20b7",
    "dev/mixing2006/na49_mixing/threshold_metric.py":
        "ee240f44645d2e592acfae462f7d1e9c3b3dd8cd78c96bd40b7b73509f7fcc63",
    "dev/mixing2006/config/v2_singlet/resonance_catalog.json":
        "cf9e79b99199c509f2e83def0a99c215f73ca7761f9f87fb01d1f779b80e32f3",
    "dev/mixing2006/config/v2_singlet/continuum_resonances.json":
        "33b2e597932013747f4781f83c027d18e1de794272717a1d1f18df4377f4cfe5",
    "dev/mixing2006/helpers/benchmarking/staged_r32_spectral_systematic.py":
        "99454facce856c93505e6c4b0d4e47b85eeb0ef54f72b1631e3d29fcc511cb90",
    "helpers/benchmarking/staged_r32_spectral_systematic.py":
        "99454facce856c93505e6c4b0d4e47b85eeb0ef54f72b1631e3d29fcc511cb90",
}

#: This card's freeze re-pin entries (the new module and the new test
#: file, plus the register chain's twelve count pins this card
#: re-pins 469 -> 471 with the naming comments).  The manifest itself
#: is excluded from its own inventory by path and appears only in the
#: freeze exception's file list.
FREEZE_CARD = "TASK-20260920-r4-isotropic-aligned-ablation-runner"
MODULE_REL = "dev/mixing2006/na49_mixing/cascade_ablation.py"
MY_TEST_REL = "dev/mixing2006/tests/test_cascade_ablation.py"
MANIFEST_REL = ("dev/mixing2006/config/v2_singlet/"
                "legacy_v1_manifest.json")
COUNT_PIN_FILES = (
    "dev/mixing2006/tests/test_cascade_alignment_baryonic.py",
    "dev/mixing2006/tests/test_cascade_alignment_mesonic.py",
    "dev/mixing2006/tests/test_cascade_axes.py",
    "dev/mixing2006/tests/test_cascade_form_factor.py",
    "dev/mixing2006/tests/test_cascade_termination.py",
    "dev/mixing2006/tests/test_r4_audit_blend_path_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_r4_cascade_alignment_baryonic_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_r4_cascade_alignment_mesonic_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_axes_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_decay_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_form_factor_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_termination_contract_pins.py",
)
FREEZE_INVENTORY_FILES = (MODULE_REL, MY_TEST_REL) + COUNT_PIN_FILES
FREEZE_EXCEPTION_FILES = FREEZE_INVENTORY_FILES + (MANIFEST_REL,)

# 469 entries at the card-9 register landing; this card adds its
# ablation runner module and test file (+2 entries), so the count pin
# of this test module is re-pinned (469 -> 471); the card-10
# test-author register (test_r4_cascade_ablation_contract_pins.py)
# re-pins the inventory additively for its own file (the constant read
# "INVENTORY_COUNT = 471" at the card-10 solver landing; 471 -> 472,
# +1 entry), so this count constant is re-pinned once more.
# card 11 of the round
# (TASK-20260920-r4-run-r40-fixed-source-alignment-scan) re-pins the
# inventory additively for its new dual-tree run driver and test file
# (helpers/benchmarking/staged_r40_alignment_scan.py,
# dev/mixing2006/helpers/benchmarking/staged_r40_alignment_scan.py,
# tests/test_staged_r40_alignment_scan.py; the constant read
# "INVENTORY_COUNT = 472" at the card-10 register landing;
# 472 -> 475, +3 entries), so this count constant is re-pinned once
# more. The card-11 test-author register
# (test_staged_r40_alignment_scan_contract_pins.py) re-pins the
# inventory additively for its own file (the constant read
# "INVENTORY_COUNT = 475" at the card-11 solver landing; 475 -> 476,
# +1 entry), so this count constant is re-pinned once more.
# re-pinned 476 -> 479 by card 12 (TASK-20260920-r4-run-r41-
# nonstrange-pilot-fit; +3 entries: the dual-tree R41 driver in
# both helper trees and the hermetic R41 test file).
# re-pinned 479 -> 480 by card 12 test-author (TASK-20260920-r4-run-
# r41-nonstrange-pilot-fit-test-author; +1 entry: the card-12
# test-author register file).
    # re-pinned 480 -> 483 by card 13 (TASK-20260920-r4-run-r42-
    # recoil-decay-separation; +3 entries: the dual-tree R42 driver in
    # both helper trees and the hermetic R42 test file).
    # re-pinned 483 -> 484 by card 13 test-author (TASK-20260920-r4-run-
    # r42-recoil-decay-separation-test-author; +1 entry: the card-13
    # test-author register file).
    # re-pinned 484 -> 487 by card 14 (TASK-20260920-r4-run-r43-
    # cascade-ablation; +3 entries: the dual-tree R43 driver in
    # both helper trees and the hermetic R43 test file).
# re-pinned 487 -> 488 by card 14 test-author (TASK-20260920-r4-run-
# r43-cascade-ablation-test-author; +1 entry: the card-14
# test-author register file).
# re-pinned 488 -> 491 by card 15 (TASK-20260920-r4-run-r44-
# frozen-nonstrange-validation; +3 entries: the dual-tree R44
# driver in both helper trees and the hermetic R44 test file).
# re-pinned 491 -> 495 by card 20 (TASK-20260920-r4-gate4-followup-simplified-alignment; +4 entries: the dual-tree R45 driver in both helper trees and the two new R45 test files).
# re-pinned 495 -> 499 by card 21 (TASK-20260920-r4-gate4-followup-recoil-decay-reprofile; +4 entries: the dual-tree R46 driver in both helper trees and the two new R46 test files).
# re-pinned 499 -> 501 by Round 5 card 1 (TASK-20260925-r5-elastic-reference; +2 entries: the elastic reference module na49_mixing/topology_elastic.py and the new elastic test file).
# re-pinned 501 -> 502 by Round 5 card 1 test-author (TASK-20260925-r5-elastic-reference-test-author; +1 entry: the card-1 test-author register file).
# re-pinned 502 -> 504 by Round 5 card 2 (TASK-20260925-r5-exchange-quantum-number-ledger; +2 entries: the exchange quantum-number ledger module na49_mixing/exchange_ledger.py and the new ledger test file tests/test_exchange_ledger.py).
# re-pinned 504 -> 505 by Round 5 card 2 test-author (TASK-20260925-r5-exchange-quantum-number-ledger-test-author; +1 entry: the card-2 test-author register file).
# re-pinned 505 -> 508 by the test-protocol tiering change (2026-09-25; +3 entries: helpers/benchmarking/select_tests_for_changes.py, its vendored copy, and helpers/benchmarking/tests/test_select_tests_for_changes.py).
# re-pinned 508 -> 510 by Round 5 card 3 (TASK-20260925-r5-intact-low-mass-remnants; +2 entries: the remnant module na49_mixing/topology_remnants.py and the new remnant test file tests/test_topology_remnants.py).
# re-pinned 510 -> 511 by Round 5 card 3 test-author (TASK-20260925-r5-intact-low-mass-remnants-test-author; +1 entry: the card-3 test-author register file tests/test_topology_remnants_contract_pins.py).
# re-pinned 511 -> 513 by Round 5 card 4 (TASK-20260925-r5-topology-mass-laws; +2 entries: the mass-laws module na49_mixing/topology_mass_laws.py and the new mass-laws test file tests/test_topology_mass_laws.py).
# re-pinned 513 -> 514 by Round 5 card 4 test-author (TASK-20260925-r5-topology-mass-laws-test-author; +1 entry: the card-4 test-author register file tests/test_topology_mass_laws_contract_pins.py).
# re-pinned 514 -> 516 by Round 5 card 5 (TASK-20260925-r5-single-diffraction; +2 entries: the single-diffraction module na49_mixing/topology_single_diffraction.py and the new single-diffraction test file tests/test_topology_single_diffraction.py).
# re-pinned 516 -> 517 by Round 5 card 5 test-author (TASK-20260925-r5-single-diffraction-test-author; +1 entry: the card-5 test-author register file tests/test_topology_single_diffraction_contract_pins.py).
# re-pinned 519 -> 520 by Round 5 card 6 test-author (TASK-20260925-r5-double-diffraction-test-author; +1 entry: the card-6 test-author register file tests/test_topology_double_diffraction_contract_pins.py).
# re-pinned 520 -> 522 by Round 5 card 7 (TASK-20260925-r5-non-diffrative-central-excitation; +2 entries: the module na49_mixing/topology_non_diffractive.py and the test file tests/test_topology_non_diffractive.py).
# re-pinned 522 -> 523 by Round 5 card 7 test-author (TASK-20260925-r5-non-diffrative-central-excitation-test-author; +1 entry: the card-7 test-author register file tests/test_topology_non_diffractive_contract_pins.py).
# re-pinned 523 -> 525 by Round 5 card 8 (TASK-20260925-r5-multiple-neutral-excitations; +2 entries: the module na49_mixing/topology_multi_neutral.py and the test file tests/test_topology_multi_neutral.py).
# re-pinned 525 -> 526 by Round 5 card 8 test-author (TASK-20260925-r5-multiple-neutral-excitations-test-author; +1 entry: the card-8 test-author register file tests/test_topology_multi_neutral_contract_pins.py).
# re-pinned 526 -> 528 by Round 5 card 9 (TASK-20260925-r5-topology-template-runner; +2 entries: the runner module na49_mixing/topology_runner.py and the test file tests/test_topology_runner.py).
# re-pinned 528 -> 529 by Round 5 card 9 test-author (TASK-20260925-r5-topology-template-runner-test-author; +1 entry: the card-9 test-author register file tests/test_topology_runner_contract_pins.py).
# re-pinned 529 -> 532 by Round 5 card 10 (TASK-20260925-r5-run-r50-pure-topology-templates; +3 entries: the dual-tree driver helpers/benchmarking/staged_r50_pure_topology_templates.py, its nested copy, and the test file tests/test_staged_r50_pure_topology_templates.py).
# re-pinned 532 -> 533 by Round 5 card 10 test-author (TASK-20260925-r5-run-r50-pure-topology-templates-test-author; +1 entry: the card-10 test-author register file tests/test_staged_r50_pure_topology_templates_contract_pins.py).
# re-pinned 533 -> 536 by Round 5 card 11 (TASK-20260925-r5-run-r51-topology-fraction-fit; +3 entries: the dual-tree driver helpers/benchmarking/staged_r51_topology_fraction_fit.py, its nested copy, and the test file tests/test_staged_r51_topology_fraction_fit.py).
# re-pinned 536 -> 537 by Round 5 card 11 test-author (TASK-20260925-r5-run-r51-topology-fraction-fit-test-author; +1 entry: the card-11 test-author register file tests/test_staged_r51_topology_fraction_fit_contract_pins.py).
# re-pinned 537 -> 540 by Round 5 card 12 (TASK-20260925-r5-run-r52-central-excitation-necessity; +3 entries: the dual-tree driver dev/mixing2006/helpers/benchmarking/staged_r52_central_excitation_necessity.py, its outer mirror helpers/benchmarking/staged_r52_central_excitation_necessity.py, and the test file dev/mixing2006/tests/test_staged_r52_central_excitation_necessity.py; the card's pre-registered 523 -> 526 literal is noted, the operative growth followed)
# re-pinned 540 -> 541 by Round 5 card 12 test-author (TASK-20260925-r5-run-r52-central-excitation-necessity-test-author; +1 entry: the card-12 test-author register file tests/test_staged_r52_central_excitation_necessity_contract_pins.py).
# re-pinned 541 -> 544 by Round 5 card 13 (TASK-20260925-r5-run-r53-multiple-neutral-excitation; +3 entries: the dual-tree driver dev/mixing2006/helpers/benchmarking/staged_r53_multiple_neutral_excitation.py, its outer mirror helpers/benchmarking/staged_r53_multiple_neutral_excitation.py, and the test file tests/test_staged_r53_multiple_neutral_excitation.py; the card's pre-registered 526 -> 529 literal is noted, the operative growth followed).
# re-pinned 544 -> 547 by Round 5 card 14 (TASK-20260925-r5-run-r54-local-ledger-audit; +3 entries: the dual-tree driver dev/mixing2006/helpers/benchmarking/staged_r54_local_ledger_audit.py, its outer mirror helpers/benchmarking/staged_r54_local_ledger_audit.py, and the test file tests/test_staged_r54_local_ledger_audit.py; the card's pre-registered 529 -> 532 literal is noted, the operative growth followed).
# re-pinned 547 -> 548 by Round 5 card 13 test-author (TASK-20260925-r5-run-r53-multiple-neutral-excitation-test-author; +1 entry: the card-13 test-author register file tests/test_staged_r53_multiple_neutral_excitation_contract_pins.py).
# re-pinned 548 -> 551 by Round 5 card 15 (TASK-20260925-r5-run-r55-frozen-topology-validation; +3 entries: the dual-tree driver dev/mixing2006/helpers/benchmarking/staged_r55_frozen_topology_validation.py, its outer mirror helpers/benchmarking/staged_r55_frozen_topology_validation.py, and the test file tests/test_staged_r55_frozen_topology_validation.py; the card's pre-registered 532 -> 535 literal is noted, the operative growth followed).
INVENTORY_COUNT = _inventory_count()
# ----------------------------------------------------------------------
# Fixtures and helpers
# ----------------------------------------------------------------------
@pytest.fixture(scope="session")
def catalog() -> HadronCatalog:
    return HadronCatalog.from_json(CATALOG_PATH)


@pytest.fixture(scope="session")
def fams():
    return families_from_json(str(FAMILIES_PATH))


def _exchange_event(family_id: str, mass: float,
                    qn: tuple[int, int, int]) -> dict:
    """A pinned at-rest exchange event of plan line 798: the topology
    label, the exchange kinematics (the source at rest at the realized
    mass), the spectral state (the family identifier and its quantum
    numbers), and the lab-z production axis (the isolated-source
    convention of round 4 index item 4)."""

    return {
        "topology": "ablation_probe",
        "momentum": (float(mass), 0.0, 0.0, 0.0),
        "family_id": family_id,
        "mass": float(mass),
        "qn": tuple(int(v) for v in qn),
        "axis": (0.0, 0.0, 1.0),
        "ancestry": ("exchange_probe",),
    }


class _CountingRng:
    """A counting proxy around one generator (the house draw-count
    pattern): every ``random()`` consumes exactly one uniform from the
    wrapped stream and advances the counter."""

    def __init__(self, source) -> None:
        self._source = source
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        return float(self._source.random())


def _wrap(streams) -> tuple[_CountingRng, _CountingRng]:
    return _CountingRng(streams[0]), _CountingRng(streams[1])


def _malformed(*parts):
    """A runtime-malformed argument handed to a validator (the
    checked-never-repaired edges; the declared call types are
    satisfied only at the type-checker's eye)."""

    return parts


def _variant_streams(event_index: int, variant_id: int):
    return variant_stream(MASTER_SEED, event_index, variant_id)


def _policies() -> tuple:
    """The three pre-registered policies in the register order."""

    return tuple(ABLATION_POLICIES[name] for name in ABLATION_VARIANTS)


def _line_family_qn(state, catalog: HadronCatalog, fams) -> tuple[int, int, int]:
    """The ``(B, Q, S)`` of a line state's record or family (the
    committed catalog and the family configuration are the resolution
    sources)."""

    family_id = str(state.family_id)
    if family_id.startswith(CATALOG_FAMILY_PREFIX):
        record = catalog.get(int(family_id[len(CATALOG_FAMILY_PREFIX):]))
        return (int(record.B), int(record.Q), int(record.S))
    family = next(f for f in fams if str(f.family_id) == family_id)
    return (int(family.qn[0]), int(family.qn[1]), int(family.qn[2]))


def _channel_qn_sum(channel: CascadeChannel, catalog: HadronCatalog,
                    fams) -> tuple[int, int, int]:
    """The ``(B, Q, S)`` sum of a channel's daughters (the card-4
    enumeration's conservation, resolved against the committed
    surfaces)."""

    total = [0, 0, 0]
    for daughter_id in channel.daughters:
        if daughter_id.startswith(CATALOG_FAMILY_PREFIX):
            record = catalog.get(
                int(daughter_id[len(CATALOG_FAMILY_PREFIX):]))
            values = (int(record.B), int(record.Q), int(record.S))
        else:
            family = next(
                f for f in fams if str(f.family_id) == daughter_id)
            values = (int(family.qn[0]), int(family.qn[1]),
                      int(family.qn[2]))
        for k in range(3):
            total[k] += values[k]
    return (total[0], total[1], total[2])


def _realized_masses(channel: CascadeChannel, state, catalog: HadronCatalog,
                     fams) -> tuple[float, float]:
    """The realized daughter masses of *channel* at *state* (the
    established daughters at their catalog pole masses, the continuum
    daughter at the pre-registered card-4 mass rule; the test-side
    mirror of card 4's rule, the same expressions)."""

    kinds: list[str] = []
    pole: list[float] = [0.0, 0.0]
    for daughter_id in channel.daughters:
        if daughter_id.startswith(CATALOG_FAMILY_PREFIX):
            pdg = int(daughter_id[len(CATALOG_FAMILY_PREFIX):])
            kinds.append("catalog")
            pole[len(kinds) - 1] = float(catalog.get(pdg).pole_mass)
        else:
            kinds.append("family")
    realized = [0.0, 0.0]
    for slot in (0, 1):
        if kinds[slot] == "catalog":
            realized[slot] = pole[slot]
    for slot in (0, 1):
        if kinds[slot] != "family":
            continue
        family = next(
            f for f in fams if str(f.family_id) == channel.daughters[slot])
        sibling = realized[1 - slot]
        edge = min(SUPPORT_M_MAX, float(state.mass) - sibling)
        realized[slot] = max(
            float(family.m_min), edge - float(family.nominal_width))
    return (realized[0], realized[1])


def _attached_pool(pool, state, b_dec: float, catalog: HadronCatalog,
                   fams) -> list[CascadeChannel]:
    """The attached channel list at *b_dec* (the reference loop's pool):
    every channel's weight is the card-4 hooked weight scaled by the
    card-8 W_dec factor at the channel's reference two-body
    configuration (the runner's pinned reference direction, the
    mirrored card-4 mass rule), at the raw scale of the enumeration."""

    wrapped = attach_to_channel_weight(channel_weight, b_dec)
    attached = []
    for channel in pool:
        mass0, mass1 = _realized_masses(channel, state, catalog, fams)
        q_star = decay_momentum(float(state.mass), mass0, mass1)
        direction = reference_direction(state.axis)
        daughter_p4 = (
            math.sqrt(mass0 * mass0 + q_star * q_star),
            q_star * direction[0],
            q_star * direction[1],
            q_star * direction[2],
        )
        attached.append(CascadeChannel(
            parent_id=channel.parent_id,
            daughters=channel.daughters,
            kind=channel.kind,
            weight=wrapped(channel, daughter_p4, state.axis),
        ))
    return attached


def _reference_isotropic_sequential_line(root, catalog: HadronCatalog,
                                         fams, rng_channel,
                                         rng_kinematics):
    """The reference loop of the v2 pin: card 9's driver routing with
    card 4's own ``draw_decay`` vertices over the attached pools (the
    independent machinery the runner's v2 is pinned against)."""

    trace = [root]
    current = root
    while True:
        if current.depth > MAX_CASCADE_DEPTH:
            raise CascadeDepthGuardError(
                f"reference loop: the line state of family "
                f"{current.family_id!r} exceeded depth {MAX_CASCADE_DEPTH}"
            )
        if is_catalog_handoff(current, catalog):
            handoff_decay(current, catalog, rng_channel, rng_kinematics)
            return trace
        channels = enumerate_channels(current, catalog, fams)
        continuum = [c for c in channels if c.kind == KIND_CONTINUUM]
        pool = continuum if continuum else [
            c for c in channels
            if c.kind in (KIND_ESTABLISHED_HADRON, KIND_ESTABLISHED_RESONANCE)
        ]
        if not pool:
            raise ValueError(
                "reference loop: no open channel at "
                f"{current.family_id!r}"
            )
        attached = _attached_pool(pool, current, DECAY_FORM_FACTOR_B_DEFAULT,
                                  catalog, fams)
        _channel, daughters = draw_decay(
            current, attached, catalog, fams, rng_channel, rng_kinematics)
        if continuum:
            successor = next(
                d for d in daughters
                if not d.family_id.startswith(CATALOG_FAMILY_PREFIX))
            trace.append(successor)
            current = successor
            continue
        return trace


# ----------------------------------------------------------------------
# test_exchange_identity
# ----------------------------------------------------------------------
def test_exchange_identity(catalog):
    """On a pinned seed and pinned exchange event, the exchange layer
    (topology, exchange kinematics, spectral state) is bit-identical
    across all three variants (plan line 798): the root state is the
    draw-free pure function of the event, the exchange substreams are
    the house derivation shared across the variants, the event mapping
    is never mutated, and every variant decays the same root."""

    event = _exchange_event("baryonic_np", 4.0, BARYONIC_QN)
    snapshot = copy.deepcopy(event)

    # the root state is draw-free and pure
    probe = np.random.default_rng(MASTER_SEED)
    state_before = probe.bit_generator.state
    root = exchange_root_state(event)
    root_again = exchange_root_state(event)
    assert root == root_again
    assert probe.bit_generator.state == state_before, (
        "exchange_root_state consumed random numbers")
    assert root.family_id == "baryonic_np"
    assert root.mass == 4.0
    assert root.qn == BARYONIC_QN
    assert root.axis == (0.0, 0.0, 1.0)
    assert root.alignment_label == ISOTROPIC_ALIGNMENT_LABEL
    assert root.depth == 0
    assert root.exchange_ancestry == ("exchange_probe",)

    # the exchange substreams are the house derivation, spawned once
    # per (master_seed, event_index) and shared across the variants
    subs = exchange_substreams(MASTER_SEED, 11)
    assert subs == derive_substreams(MASTER_SEED, 0, 11)
    assert exchange_substreams(MASTER_SEED, 11) == subs
    assert exchange_substreams(MASTER_SEED, 11, 3) == \
        derive_substreams(MASTER_SEED, 3, 11)

    # the event mapping is never mutated and every variant decays the
    # same root (bit-identical exchange layer across the variants)
    for policy in _policies():
        rc, rk = _variant_streams(11, ABLATION_VARIANTS.index(policy.variant))
        output = decay_variant(event, policy, rc, rk)
        assert event == snapshot, (
            f"{policy.variant}: the exchange event was mutated")
        if policy.variant == "isotropic_direct":
            assert output, "the v1 step has terminal daughters"
            for daughter in output:
                assert daughter.parent == root
                assert daughter.depth == root.depth + 1
        else:
            assert output[0] == root, (
                f"{policy.variant}: the line does not start at the "
                "shared exchange root")

    # the exchange event's spectral state is the runner's resolution
    # input: a catalog event resolves against the committed record
    catalog_event = _exchange_event("catalog:2214", 1.232, BARYONIC_QN)
    catalog_root = exchange_root_state(catalog_event)
    assert catalog_root.qn == BARYONIC_QN
    assert catalog_root.quantum["I"] == pytest.approx(
        float(catalog.get(2214).isospin))


# ----------------------------------------------------------------------
# test_v2_pin
# ----------------------------------------------------------------------
def test_v2_pin(catalog, fams):
    """The isotropic sequential variant is bit-identical to card 4's
    cascade at kappa = 0 with b_dec nominal on pinned seeds (the
    reference loop drives card 4's own ``draw_decay`` over the attached
    pools through card 9's routing); at the off limit ``b_dec = 0`` the
    runner's line is bit-identical to card 9's unattached
    ``run_cascade`` baseline."""

    v2 = ABLATION_POLICIES["isotropic_sequential"]
    assert v2.kappa_b == 0.0 and v2.kappa_m == 0.0
    assert v2.b_dec == DECAY_FORM_FACTOR_B_DEFAULT == 1.0

    lines = 0
    for family_id, mass, qn in CONTINUUM_ROOTS + CATALOG_ROOTS:
        event = _exchange_event(family_id, mass, qn)
        for event_index in PINNED_EVENTS:
            trace = decay_variant(
                event, v2, *_variant_streams(event_index, 1))
            reference = _reference_isotropic_sequential_line(
                exchange_root_state(event), catalog, fams,
                *_variant_streams(event_index, 1))
            assert trace == reference, (
                f"{family_id}@{mass} event {event_index}: the isotropic "
                "sequential variant drifted from card 4's cascade at "
                "kappa = 0 with b_dec nominal")
            lines += 1
    assert lines == (len(CONTINUUM_ROOTS) + len(CATALOG_ROOTS)) * len(
        PINNED_EVENTS)

    # the off limit: b_dec = 0 reproduces the unattached card-9
    # run_cascade baseline bit for bit (the attached weights are the
    # unattached ones, the dispatch is card 4's isotropic draw)
    off_policy = AblationPolicy("isotropic_sequential", 0.0, 0.0, 0.0)
    for family_id, mass, qn in (("baryonic_np", 4.0, BARYONIC_QN),
                                ("mesonic_neutral", 4.0, MESONIC_QN),
                                ("baryonic_strange", 4.0,
                                 STRANGE_BARYONIC_QN)):
        event = _exchange_event(family_id, mass, qn)
        for event_index in (7, 11):
            trace = decay_variant(
                event, off_policy, *_variant_streams(event_index, 1))
            baseline = run_cascade(
                exchange_root_state(event), catalog, fams,
                *_variant_streams(event_index, 1))
            assert trace == baseline, (
                f"{family_id}@{mass} event {event_index}: the off-limit "
                "line drifted from the unattached run_cascade baseline")


# ----------------------------------------------------------------------
# test_stream_disjointness
# ----------------------------------------------------------------------
def test_stream_disjointness():
    """The per-variant decay streams are child spawns keyed on the
    variant id, disjoint across variants (a draw on variant 0 never
    advances variant 1's stream), and deterministic on the pinned key
    (the same ``(master_seed, event_index, variant_id)`` reproduces the
    stream byte-for-byte)."""

    # determinism: the same key reproduces the streams byte-for-byte
    for event_index in PINNED_EVENTS:
        for variant_id in range(len(ABLATION_VARIANTS)):
            first = _variant_streams(event_index, variant_id)
            second = _variant_streams(event_index, variant_id)
            for gen_a, gen_b in zip(first, second):
                assert [gen_a.random() for _ in range(6)] == [
                    gen_b.random() for _ in range(6)]

    # disjointness: a draw on variant 0 never advances variant 1's or
    # variant 2's streams
    streams = {
        vid: _variant_streams(11, vid)
        for vid in range(len(ABLATION_VARIANTS))
    }
    states_before = {
        vid: (gen.bit_generator.state for gen in pair)
        for vid, pair in streams.items()
    }
    states_before = {
        vid: tuple(gen.bit_generator.state for gen in pair)
        for vid, pair in streams.items()
    }
    for _ in range(16):
        streams[0][0].random()
        streams[0][1].random()
    for vid in (1, 2):
        states_after = tuple(
            gen.bit_generator.state for gen in streams[vid])
        assert states_after == states_before[vid], (
            f"variant {vid}: its stream advanced while variant 0 drew")

    # the variant streams are keyed on the variant id: the three
    # channels streams (and the three kinematics streams) differ
    fresh = {vid: _variant_streams(11, vid)
             for vid in range(len(ABLATION_VARIANTS))}
    channel_sequences = [
        tuple(fresh[vid][0].random() for _ in range(8))
        for vid in range(len(ABLATION_VARIANTS))
    ]
    kinematics_sequences = [
        tuple(fresh[vid][1].random() for _ in range(8))
        for vid in range(len(ABLATION_VARIANTS))
    ]
    assert len(set(channel_sequences)) == len(ABLATION_VARIANTS)
    assert len(set(kinematics_sequences)) == len(ABLATION_VARIANTS)

    # the key is sensitive to the event index and the master seed
    seq_a = tuple(_variant_streams(11, 0)[0].random() for _ in range(4))
    seq_b = tuple(_variant_streams(12, 0)[0].random() for _ in range(4))
    seq_c = tuple(
        variant_stream(MASTER_SEED + 1, 11, 0)[0].random() for _ in range(4))
    assert seq_a != seq_b
    assert seq_a != seq_c

    # the arguments are validated (checked, never repaired; the
    # malformed values are runtime shapes the declared call types do
    # not carry -- the house checked-never-repaired edge)
    for bad in _malformed(-1, True, 1.0, "7"):
        with pytest.raises(ValueError):
            variant_stream(bad, 11, 0)
        with pytest.raises(ValueError):
            variant_stream(MASTER_SEED, bad, 0)
        with pytest.raises(ValueError):
            variant_stream(MASTER_SEED, 11, bad)
    with pytest.raises(ValueError, match="closed ablation register"):
        variant_stream(MASTER_SEED, 11, len(ABLATION_VARIANTS))


# ----------------------------------------------------------------------
# test_support_boundary
# ----------------------------------------------------------------------
def test_support_boundary():
    """The support boundary is carried on the module and machine-
    readable (the pre-registration block shape card 14 consumes): v1
    single-step vs v2/v3 multi-vertex, v2/v3 support identical, v1
    regenerated per plan sections 5.2 line 275 and 5.3 line 288; the
    pre-registered policy register and its validation edges."""

    assert ABLATION_VARIANTS == (
        "isotropic_direct", "isotropic_sequential",
        "aligned_sequential")

    # the machine-readable block: the plan citations, the per-variant
    # support and common-random-numbers contract, the boundary
    assert set(SUPPORT_BOUNDARY) >= {
        "plan_line_275", "plan_line_288", "variants",
        "v2_v3_support_identical", "regeneration_required",
    }
    assert "support does not change" in SUPPORT_BOUNDARY["plan_line_275"]
    assert "Regenerate when thresholds, allowed channels, or parent "
    "masses change." in SUPPORT_BOUNDARY["plan_line_288"]
    variants = SUPPORT_BOUNDARY["variants"]
    assert set(variants) == set(ABLATION_VARIANTS)
    direct = variants["isotropic_direct"]
    assert direct["decay_steps"] == "single_step"
    assert direct["support"] == "single_step"
    assert direct["common_random_numbers"] == "regenerate"
    assert direct["boundary"]
    for name in ("isotropic_sequential", "aligned_sequential"):
        entry = variants[name]
        assert entry["decay_steps"] == "multi_vertex"
        assert entry["support"] == "multi_vertex"
        assert entry["common_random_numbers"] == "shared"
    assert variants["aligned_sequential"]["boundary"] is not None
    assert "identical to isotropic_sequential" in \
        variants["aligned_sequential"]["boundary"]
    assert SUPPORT_BOUNDARY["v2_v3_support_identical"] is True
    assert SUPPORT_BOUNDARY["regeneration_required"] == ("isotropic_direct",)

    # the pre-registered policy register: the R43 variants at the
    # isotropic point and at the nominal point (round 4 index item 3)
    assert set(ABLATION_POLICIES) == set(ABLATION_VARIANTS)
    for name in ("isotropic_direct", "isotropic_sequential"):
        policy = ABLATION_POLICIES[name]
        assert policy.kappa_b == 0.0 and policy.kappa_m == 0.0
    aligned = ABLATION_POLICIES["aligned_sequential"]
    assert aligned.kappa_b == BARYONIC_KAPPA_DEFAULT == 1.0
    assert aligned.kappa_m == MESONIC_KAPPA_DEFAULT == 1.0
    for policy in ABLATION_POLICIES.values():
        assert policy.b_dec == DECAY_FORM_FACTOR_B_DEFAULT == 1.0

    # the policy validation edges (checked, never repaired)
    with pytest.raises(ValueError, match="closed register"):
        AblationPolicy("aligned_direct", 0.0, 0.0, 1.0)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="kappa"):
            AblationPolicy("aligned_sequential", bad, 0.0, 1.0)
        with pytest.raises(ValueError, match="kappa"):
            AblationPolicy("aligned_sequential", 0.0, bad, 1.0)
    for bad in (float("nan"), -1e-12, -0.5):
        with pytest.raises(ValueError, match="b_dec"):
            AblationPolicy("aligned_sequential", 1.0, 1.0, bad)

    # the reference emission configuration is pinned and consistent
    assert W_DEC_REFERENCE_COS_THETA == 0.3
    assert W_DEC_REFERENCE_SIN_THETA == pytest.approx(
        math.sqrt(1.0 - 0.3 * 0.3), rel=0, abs=1e-15)
    direction = reference_direction((0.0, 0.0, 1.0))
    norm = math.sqrt(sum(component * component for component in direction))
    assert norm == pytest.approx(1.0, abs=1e-12)
    assert direction[2] == pytest.approx(W_DEC_REFERENCE_COS_THETA,
                                         abs=1e-12)


# ----------------------------------------------------------------------
# test_conservation
# ----------------------------------------------------------------------
def test_conservation(catalog, fams):
    """Every variant conserves (B, Q, S) exactly per event on the
    pinned fixtures: the v1 daughter sums, the v2/v3 line states
    against their records and families, and the open-channel sums at
    every line state (the runner's checks are the enforcement; the
    completion of every pinned decay is the evidence)."""

    for family_id, mass, qn in CONTINUUM_ROOTS + CATALOG_ROOTS:
        event = _exchange_event(family_id, mass, qn)
        root = exchange_root_state(event)
        for policy in _policies():
            event_index = PINNED_EVENTS[0]
            output = decay_variant(
                event, policy, *_variant_streams(event_index,
                                                 ABLATION_VARIANTS.index(
                                                     policy.variant)))
            if policy.variant == "isotropic_direct":
                # the single step's terminal daughters sum to the root
                assert output
                total = tuple(
                    sum(daughter.qn[k] for daughter in output)
                    for k in range(3))
                assert total == tuple(int(v) for v in root.qn), (
                    f"{family_id}@{mass} v1: the daughters sum to "
                    f"{total}, the root carries {tuple(root.qn)}")
                for daughter in output:
                    assert daughter.parent == root
                    assert daughter.depth == root.depth + 1
                    closure = max(
                        abs(daughter.momentum[k] + 0.0) for k in (0,))
                    assert math.isfinite(closure)
                momentum_sum = tuple(
                    sum(daughter.momentum[k] for daughter in output)
                    for k in range(4))
                worst = max(
                    abs(momentum_sum[k] - float(root.momentum[k]))
                    for k in range(4))
                assert worst <= 1e-9, (
                    f"{family_id}@{mass} v1: the daughter four-momenta "
                    f"miss the root by {worst!r} GeV")
            else:
                # the cascade line: every line state carries its
                # record's / family's quantum numbers, starting at the
                # root's, and every open channel of the line states
                # conserves (B, Q, S) at the vertex
                assert output[0] == root
                for state in output:
                    expected = _line_family_qn(state, catalog, fams)
                    assert tuple(int(v) for v in state.qn) == expected, (
                        f"{policy.variant} {family_id}@{mass}: the line "
                        f"state {state.family_id!r} carries "
                        f"{tuple(state.qn)}, its surface carries "
                        f"{expected}")
                    channels = enumerate_channels(state, catalog, fams)
                    for channel in channels:
                        assert _channel_qn_sum(
                            channel, catalog, fams) == expected, (
                            f"{policy.variant} {family_id}@{mass}: "
                            f"channel {tuple(channel.daughters)} of "
                            f"{state.family_id!r} violates (B, Q, S)")

    # the pinned v1 resonance-daughter fixture: the queued resonance
    # daughter is terminal for the single step and its quantum numbers
    # still sum to the root (the ablation floor's conservation)
    event = _exchange_event("catalog:12212", 2.0, BARYONIC_QN)
    output = decay_variant(event, ABLATION_POLICIES["isotropic_direct"],
                           *_variant_streams(11, 0))
    total = tuple(sum(daughter.qn[k] for daughter in output)
                  for k in range(3))
    assert total == BARYONIC_QN


# ----------------------------------------------------------------------
# test_budget_per_variant
# ----------------------------------------------------------------------
def test_budget_per_variant(catalog, fams):
    """v1 consumes the Round 2 single-step budget (1 channel + 2
    kinematics, plus the kernel's documented 2 mass draws per resonance
    daughter); v2/v3 consume card 4's frozen per-vertex budget (1
    channel + 2 kinematics per line state), the alignment and
    form-factor attachments adding zero uniforms (identical per-vertex
    counts across the policies, the per-attachment draw-budget
    invariance).  The line lengths themselves may differ between
    policies: a different form-factor slope reweights the channel pool,
    so the same channel quantile can select a different channel and a
    different remainder mass (observed at the mesonic 4.0 GeV root
    between the nominal and the scan slope under the 2026-09-28
    kernels)."""

    stable = catalog.stable_states()

    # -- v1: the Round 2 single-step budget at the catalog fixtures ----
    event = _exchange_event("catalog:2214", 1.232, BARYONIC_QN)
    rc, rk = _wrap(_variant_streams(7, 0))
    output = decay_variant(event, ABLATION_POLICIES["isotropic_direct"],
                           rc, rk)
    assert rc.calls == 1 and rk.calls == 2, (
        "v1 Delta+: the single-step budget is 1 channel + 2 kinematics "
        f"(got {rc.calls} + {rk.calls})")
    assert len(output) == 2

    # the resonance-daughter clause: the kernel's documented 2 mass
    # draws per resonance daughter of the realized single vertex
    event = _exchange_event("catalog:12212", 2.0, BARYONIC_QN)
    for event_index in (7, 11, 13):
        rc, rk = _wrap(_variant_streams(event_index, 0))
        output = decay_variant(
            event, ABLATION_POLICIES["isotropic_direct"], rc, rk)
        n_resonance = sum(
            1 for daughter in output
            if int(daughter.family_id[len(CATALOG_FAMILY_PREFIX):])
            not in stable)
        assert rc.calls == 1
        assert rk.calls == 2 + 2 * n_resonance, (
            f"v1 12212@2.0 event {event_index}: the single-step budget "
            f"is 1 channel + 2 kinematics + 2 per resonance daughter "
            f"(got {rc.calls} + {rk.calls} with {n_resonance} resonance "
            "daughters)")

    # -- v1: the continuum single step is one card-4 vertex -------------
    event = _exchange_event("baryonic_np", 4.0, BARYONIC_QN)
    rc, rk = _wrap(_variant_streams(7, 0))
    output = decay_variant(event, ABLATION_POLICIES["isotropic_direct"],
                           rc, rk)
    assert rc.calls == 1 and rk.calls == 2
    assert len(output) == 2

    # -- v2/v3: card 4's frozen per-vertex budget per line state, and
    # the attachments add zero uniforms (the per-attachment invariance)
    scan_policy = AblationPolicy("aligned_sequential", 4.0, 4.0, 2.0)
    for family_id, mass, qn in CONTINUUM_ROOTS:
        event = _exchange_event(family_id, mass, qn)
        counts = {}
        for name, policy in (
            ("v2", ABLATION_POLICIES["isotropic_sequential"]),
            ("v3", ABLATION_POLICIES["aligned_sequential"]),
            ("scan", scan_policy),
        ):
            rc, rk = _wrap(_variant_streams(7, 1))
            trace = decay_variant(event, policy, rc, rk)
            assert rc.calls == len(trace), (
                f"{family_id}@{mass} {name}: the channel budget is one "
                f"uniform per line state (got {rc.calls} for "
                f"{len(trace)} states)")
            assert rk.calls == 2 * len(trace), (
                f"{family_id}@{mass} {name}: the kinematics budget is "
                f"two uniforms per line state (got {rk.calls})")
            counts[name] = (rc.calls / len(trace), rk.calls / len(trace))
        assert counts["v2"] == counts["v3"] == counts["scan"] == (1.0, 2.0), (
            f"{family_id}@{mass}: the attachments moved the per-vertex "
            f"draw budget ({counts})")

    # -- the catalog-routed roots: the handoff remainder is the Round 2
    # kernel's own budget (the card-9 delegation); the line is the root
    event = _exchange_event("catalog:12212", 2.0, BARYONIC_QN)
    for policy in (ABLATION_POLICIES["isotropic_sequential"],
                   ABLATION_POLICIES["aligned_sequential"]):
        rc, rk = _wrap(_variant_streams(7, 1))
        trace = decay_variant(event, policy, rc, rk)
        assert trace == [exchange_root_state(event)]
        assert rc.calls >= 1 and rk.calls >= 2, (
            "the handoff remainder consumed the kernel's budget")


# ----------------------------------------------------------------------
# test_layout_sync
# ----------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_layout_sync():
    """The consumed-surface register (the frozen substrate digests of
    the card's provenance section, verified on disk) and the layout
    sync (the subprocess check from the monorepo root; ``logic_drift
    = 0``)."""

    for rel, expected in FROZEN_SURFACES.items():
        path = MONO_ROOT / rel
        assert path.is_file(), f"missing frozen surface: {rel}"
        pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    probe = subprocess.run(
        [sys.executable, str(SYNC_CHECKER)],
        capture_output=True, text=True, cwd=str(MONO_ROOT))
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert "logic_drift=0" in probe.stdout


# ----------------------------------------------------------------------
# The acceptance-(vi) freeze re-pin check
# ----------------------------------------------------------------------
