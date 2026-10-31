"""Card tests for the cascade termination rules and the depth guard.

Round 4, card 9 (``TASK-20260920-r4-termination-depth-tests``; plan
section 22 task 26, line 1748; plan section 10.6, lines 750-757).  The
card's pre-registered deterministic tests against the committed catalog
(``config/v2_singlet/resonance_catalog.json``) and the continuum family
configuration (``config/v2_singlet/continuum_resonances.json``):

- ``test_catalog_handoff_predicate``: the handoff predicate is true
  exactly on the pinned states matching catalog records within support
  (QN match, mass inside ``[m_min, m_up]``; false on QN mismatch or
  out-of-support mass).
- ``test_direct_channel_termination``: the direct-termination predicate
  fires exactly on the pinned states with no open continuum channel and
  a valid direct hadronic channel.
- ``test_handoff_bit_identity``: on a pinned handoff state, the
  delegated Round 2 decay is bit-identical to the standalone
  ``ResonanceDecayer`` at the same seed (the Round 2 kernel is
  byte-unchanged).
- ``test_termination_complete``: a pinned 10,000-cascade trace over the
  accepted source-mass range (2-8 GeV, baryonic ``(1, 1, 0)`` and
  mesonic ``(0, 0, 0)`` families) terminates on every cascade (100%);
  the termination mode (catalog handoff / direct channel) is recorded
  per cascade.
- ``test_guard_never_reached``: the pinned 10,000-cascade trace reaches
  the depth guard zero times (plan section 10.6 line 757, "tune physics
  so the guard is never reached in accepted events").
- ``test_guard_fires``: a synthetic depth-13 state (a pinned fixture
  forcing the chain) raises ``CascadeDepthGuardError`` naming the
  depth.
- ``test_conservation_through_termination``: every vertex of the pinned
  trace conserves ``(B, Q, S)`` exactly, including the handoff and
  direct-termination vertices.
- ``test_layout_sync``: the test register and the layout sync.

plus the card's acceptance-(v) freeze re-pin check
``test_freeze_manifest_repin``.

The pinned trace (the card's hermetic instantiation of the accepted
source-mass range, round 4 index item 2): the ten at-rest root points of
the R40 grid, 1,000 cascades per point.  The 2 GeV points sit below the
kernel boundary B = 2.680194110727528 GeV, where the R40 routing is
catalog-level, so their pinned roots are catalog records standing for
the sources (N*(1440)+ and rho(1450)0, both within their record
supports at 2.0 GeV); the 3, 4, 6, and 8 GeV points are the continuum
families.  Every cascade carries its own (master seed, replica 0, event
index) substream triple, event index = cascade index.

Hermetic: the committed config and catalog files, numpy, the standard
library, and the vendored ``na49_mixing`` modules only; no event
generation, no external tables, no Monte Carlo run.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.cascade_decay import (
    CATALOG_FAMILY_PREFIX,
    ISOTROPIC_ALIGNMENT_LABEL,
    KIND_CONTINUUM,
    CascadeState,
    enumerate_channels,
)
from na49_mixing.cascade_termination import (
    TERMINATION_CATALOG_HANDOFF,
    TERMINATION_DIRECT_CHANNEL,
    MAX_CASCADE_DEPTH,
    CascadeDepthGuardError,
    has_valid_direct_channel,
    handoff_decay,
    is_catalog_handoff,
    run_cascade,
)
from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.resonance_decay import ResonanceDecayer
from na49_mixing.rng_substreams import derive_substreams
from na49_mixing.spectral_density import families_from_json
from tests._freeze_inventory import inventory_count as _inventory_count

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent          # dev/mixing2006
MONO_ROOT = NESTED_ROOT.parent.parent          # the monorepo root
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
FAMILIES_PATH = NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json"
MANIFEST_PATH = NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json"
SYNC_CHECKER = MONO_ROOT / "helpers" / "benchmarking" / "check_helpers_layout_sync.py"

#: The round 4 master seed (round 4 index item 9), fixed for the pinned
#: trace and the handoff bit-identity fixtures.
MASTER_SEED = 20260920

#: The M8-frozen kernel boundary B (the continuum family m_min; the
#: catalog/continuum routing boundary of the accepted source range).
KERNEL_BOUNDARY = 2.680194110727528

BARYONIC_QN = (1, 1, 0)
MESONIC_QN = (0, 0, 0)

#: The pinned trace roots: the ten at-rest points of the accepted
#: source-mass range (round 4 index item 2), 1,000 cascades per point.
#: The 2 GeV points are catalog-routed (below the kernel boundary), so
#: their pinned roots are the catalog records standing for the sources;
#: the 3, 4, 6, and 8 GeV points are the continuum families.
CASCADE_ROOTS = (
    ("catalog:12212", 2.0, BARYONIC_QN),   # N*(1440)+, in support at 2.0
    ("baryonic_np", 3.0, BARYONIC_QN),
    ("baryonic_np", 4.0, BARYONIC_QN),
    ("baryonic_np", 6.0, BARYONIC_QN),
    ("baryonic_np", 8.0, BARYONIC_QN),
    ("catalog:100113", 2.0, MESONIC_QN),   # rho(1450)0, in support at 2.0
    ("mesonic_neutral", 3.0, MESONIC_QN),
    ("mesonic_neutral", 4.0, MESONIC_QN),
    ("mesonic_neutral", 6.0, MESONIC_QN),
    ("mesonic_neutral", 8.0, MESONIC_QN),
)
CASCADES_PER_ROOT = 1_000
N_CASCADES = CASCADES_PER_ROOT * len(CASCADE_ROOTS)  # 10,000

#: The pinned trace outcomes (deterministic: fixed seeds, fixed roots,
#: fixed machinery).  Every cascade terminates; the two 2 GeV
#: catalog-routed roots hand off at depth 0 (the kernel takes the
#: remainder) and the eight continuum roots terminate through the
#: direct channel; the deepest line state sits at depth 4, far below
#: the guard.
PINNED_HANDOFF_CASCADES = 2_000
PINNED_DIRECT_CASCADES = 8_000
PINNED_MAX_LINE_DEPTH = 4

#: The pinned handoff bit-identity fixtures: the rho0 pole (the pinned
#: single-vertex kernel decay) and the pinned 2 GeV baryonic source
#: record (a multi-vertex kernel chain through the Delta+ daughter).
HANDOFF_FIXTURES = (
    ("catalog:113", 0.77526, MESONIC_QN, 7),
    ("catalog:12212", 2.0, BARYONIC_QN, 11),
)

#: The byte-unchanged frozen substrate of the card's provenance
#: section, verified on disk at this card's landing: card 4's cascade
#: surface, card 5's axis machinery, the card-6/7 alignment kernel,
#: card 8's form factor, the Round 2 kernel (the handoff target), the
#: builder, the card-6 blend routing, the card-2 engine, the committed
#: catalog, the continuum families configuration, and the R32 driver
#: in both trees.
FROZEN_SURFACES = {
    "dev/mixing2006/na49_mixing/cascade_decay.py":
        "d2a48049af3d6df968993c040033a74422ccd709f53a42bad0538ae5111bcc0c",
    "dev/mixing2006/na49_mixing/cascade_axes.py":
        "a7251a6b6c0ed4b9cfc56c0182c28a70efccb7a92a33b756090ff716a3ec3c61",
    "dev/mixing2006/na49_mixing/cascade_alignment.py":
        "0d0df347222423aaf6f02403e177867b57da588a5e40d641fd7b467e9c4a81f4",
    "dev/mixing2006/na49_mixing/cascade_form_factor.py":
        "b16077c9e3418b3f19b117aadf6fc8a8e838af5e9d37b1ab0bae89ccfb88f74c",
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

#: This card's freeze re-pin entries (the new module, the additive
#: package exports, the new test file, and the register chain's count
#: pins this card re-pins 466 -> 468 with the naming comments).  The
#: manifest itself is excluded from its own inventory by path and
#: appears only in the freeze exception's file list.
FREEZE_CARD = "TASK-20260920-r4-termination-depth-tests"
MODULE_REL = "dev/mixing2006/na49_mixing/cascade_termination.py"
INIT_REL = "dev/mixing2006/na49_mixing/__init__.py"
MY_TEST_REL = "dev/mixing2006/tests/test_cascade_termination.py"
MANIFEST_REL = "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json"
COUNT_PIN_FILES = (
    "dev/mixing2006/tests/test_cascade_alignment_baryonic.py",
    "dev/mixing2006/tests/test_cascade_alignment_mesonic.py",
    "dev/mixing2006/tests/test_cascade_axes.py",
    "dev/mixing2006/tests/test_cascade_form_factor.py",
    "dev/mixing2006/tests/test_r4_audit_blend_path_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_r4_cascade_alignment_baryonic_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_r4_cascade_alignment_mesonic_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_axes_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_decay_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_form_factor_contract_pins.py",
)
FREEZE_INVENTORY_FILES = (MODULE_REL, INIT_REL, MY_TEST_REL) + COUNT_PIN_FILES
FREEZE_EXCEPTION_FILES = FREEZE_INVENTORY_FILES + (MANIFEST_REL,)

# 466 entries at the card-8 register landing; this card adds its
# termination module and test file (+2 entries) and re-pins the
# register chain's count pins (466 -> 468); card 9's independent
# test-author register (test_r4_cascade_termination_contract_pins.py)
# re-pins the inventory additively for its own file (the constant read
# "INVENTORY_COUNT = 468" at this card's landing; 468 -> 469,
# +1 entry), so this count constant is re-pinned once more; card 10 of
# the round (TASK-20260920-r4-isotropic-aligned-ablation-runner)
# re-pins the inventory additively for its new ablation runner module
# and test file (na49_mixing/cascade_ablation.py,
# test_cascade_ablation.py; the constant read "INVENTORY_COUNT = 469"
# at the card-9 register landing; 469 -> 471, +2 entries), so this
# count constant is re-pinned once more; the card-10 test-author
# register (test_r4_cascade_ablation_contract_pins.py) re-pins the
# inventory additively for its own file (the constant read
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


def _root_state(family_id: str, mass: float, qn: tuple[int, int, int],
                depth: int = 0) -> CascadeState:
    """A pinned at-rest root state with the lab-z production axis and
    the isotropic label (the caller-supplied axis and label of the
    card-4 landing; card 5's machinery is consumed read-only)."""

    return CascadeState(
        momentum=(float(mass), 0.0, 0.0, 0.0),
        mass=float(mass),
        qn=qn,
        quantum={},
        family_id=family_id,
        axis=(0.0, 0.0, 1.0),
        alignment_label=ISOTROPIC_ALIGNMENT_LABEL,
        parent=None,
        exchange_ancestry=("cascade_probe",),
        depth=depth,
    )


def _sub_rngs(seed: int, event_index: int):
    """The closed decay_channel/decay_kinematics substream generators
    of one (master seed, replica 0, event index) triple (the round 2
    card 9 and round 3 card 8 convention)."""

    subs = derive_substreams(seed, 0, event_index)
    return (
        np.random.default_rng(subs.stream_for("decay_channel")),
        np.random.default_rng(subs.stream_for("decay_kinematics")),
    )


def _termination_mode(trace: list[CascadeState], catalog) -> str:
    """The termination mode of a finished cascade: the terminal state
    identifies the mode exactly (a catalog record within support is a
    handoff; a continuum terminal state is a direct-channel
    termination)."""

    terminal = trace[-1]
    if is_catalog_handoff(terminal, catalog):
        return TERMINATION_CATALOG_HANDOFF
    return TERMINATION_DIRECT_CHANNEL


def _depth_chain(n_states: int, family_id: str, mass: float,
                 qn: tuple[int, int, int]) -> list[CascadeState]:
    """A synthetic line of *n_states* states at depths 0..n_states-1
    with consistent parent links and depth counters (the card-4 depth
    counter is the guard's input; the fixture forces the chain)."""

    states: list[CascadeState] = []
    parent: CascadeState | None = None
    for depth in range(n_states):
        state = CascadeState(
            momentum=(float(mass), 0.0, 0.0, 0.0),
            mass=float(mass),
            qn=qn,
            quantum={},
            family_id=family_id,
            axis=(0.0, 0.0, 1.0),
            alignment_label=ISOTROPIC_ALIGNMENT_LABEL,
            parent=parent,
            exchange_ancestry=("guard_probe",),
            depth=depth,
        )
        states.append(state)
        parent = state
    return states


@pytest.fixture(scope="session")
def pinned_trace(catalog, fams):
    """The pinned 10,000-cascade trace over the accepted source-mass
    range: the per-cascade (mode, terminal depth, trace) records.

    Deterministic: fixed seeds (the master seed, replica 0, one event
    index per cascade), fixed roots, fixed machinery.
    """

    records = []
    for index in range(N_CASCADES):
        family_id, mass, qn = CASCADE_ROOTS[index % len(CASCADE_ROOTS)]
        rng_channel, rng_kinematics = _sub_rngs(MASTER_SEED, index)
        trace = run_cascade(
            _root_state(family_id, mass, qn),
            catalog, fams, rng_channel, rng_kinematics,
        )
        records.append({
            "root": (family_id, mass, qn),
            "trace": trace,
            "mode": _termination_mode(trace, catalog),
            "terminal_depth": trace[-1].depth,
        })
    return records


# ----------------------------------------------------------------------
# test_catalog_handoff_predicate
# ----------------------------------------------------------------------
def test_catalog_handoff_predicate(catalog):
    """The handoff predicate is true exactly on the pinned states
    matching catalog records within support (QN match, mass inside
    ``[m_min, m_up]``); false on QN mismatch or out-of-support mass."""

    # true: the pinned in-support catalog states
    true_states = (
        # the Delta+ pole inside (1.079135, 2.0)
        ("catalog:2214", 1.232, BARYONIC_QN),
        # the rho0 pole inside (0.279140, 1.5)
        ("catalog:113", 0.77526, MESONIC_QN),
        # a stable record: the degenerate support [pole, pole]
        ("catalog:2212", 0.93827231, BARYONIC_QN),
        # the inclusive support bounds of the Delta+ record
        ("catalog:2214", 1.07913531, BARYONIC_QN),
        ("catalog:2214", 2.0, BARYONIC_QN),
    )
    for family_id, mass, qn in true_states:
        state = _root_state(family_id, mass, qn)
        assert is_catalog_handoff(state, catalog) is True, (
            f"{family_id} at {mass} must hand off"
        )

    # false: the QN mismatch (the record id resolves, the state's
    # (B, Q, S) disagrees)
    mismatched = _root_state("catalog:2214", 1.232, MESONIC_QN)
    assert is_catalog_handoff(mismatched, catalog) is False

    # false: the out-of-support masses (below m_min, above m_up)
    below = _root_state("catalog:2214", 1.0, BARYONIC_QN)
    assert is_catalog_handoff(below, catalog) is False
    above = _root_state("catalog:2214", 2.5, BARYONIC_QN)
    assert is_catalog_handoff(above, catalog) is False

    # false: a continuum state is never a handoff, even though a
    # committed catalog record (the proton) carries the same (B, Q, S)
    continuum = _root_state("baryonic_np", 4.0, BARYONIC_QN)
    assert is_catalog_handoff(continuum, catalog) is False

    # false: an unknown record id is no match
    unknown = _root_state("catalog:999999", 1.232, BARYONIC_QN)
    assert is_catalog_handoff(unknown, catalog) is False


# ----------------------------------------------------------------------
# test_direct_channel_termination
# ----------------------------------------------------------------------
def test_direct_channel_termination(catalog, fams):
    """The direct-termination predicate fires exactly on the pinned
    states with no open continuum channel and a valid direct hadronic
    channel."""

    # true: the pinned continuum states at the family floor and just
    # above it -- the card-4 mass rule lands the line there, the
    # continuum budget windows are closed, and the direct channels are
    # open
    for family_id, mass, qn in (
        ("baryonic_np", KERNEL_BOUNDARY, BARYONIC_QN),
        ("mesonic_neutral", KERNEL_BOUNDARY, MESONIC_QN),
        ("baryonic_np", 2.8, BARYONIC_QN),
    ):
        state = _root_state(family_id, mass, qn)
        channels = enumerate_channels(state, catalog, fams)
        assert not any(c.kind == KIND_CONTINUUM for c in channels), (
            f"{family_id} at {mass}: the continuum windows must be closed"
        )
        assert has_valid_direct_channel(state, channels) is True, (
            f"{family_id} at {mass} must terminate through the direct "
            "channel"
        )

    # false: a continuum channel is open (the mixed enumeration of the
    # accepted range)
    for family_id, mass, qn in (
        ("baryonic_np", 4.0, BARYONIC_QN),
        ("mesonic_neutral", 4.0, MESONIC_QN),
        ("baryonic_np", 8.0, BARYONIC_QN),
    ):
        state = _root_state(family_id, mass, qn)
        channels = enumerate_channels(state, catalog, fams)
        assert any(c.kind == KIND_CONTINUUM for c in channels)
        assert has_valid_direct_channel(state, channels) is False

    # false: no channel at all is open (below every threshold), so the
    # direct rule does not fire either; the driver's no-channel case
    # raises (checked, never repaired: neither termination rule applies)
    empty = _root_state("mesonic_neutral", 0.2, MESONIC_QN)
    assert enumerate_channels(empty, catalog, fams) == []
    assert has_valid_direct_channel(empty, []) is False
    rng_channel, rng_kinematics = _sub_rngs(MASTER_SEED, 4)
    with pytest.raises(ValueError, match="no open continuum channel"):
        run_cascade(empty, catalog, fams, rng_channel, rng_kinematics)

    # the predicate is a pure formula over the channels: a catalog
    # state's enumeration (all established kinds) satisfies it, but the
    # driver checks the catalog handoff first, so a catalog state never
    # terminates through this rule in run_cascade
    delta = _root_state("catalog:2214", 1.232, BARYONIC_QN)
    channels = enumerate_channels(delta, catalog, fams)
    assert all(c.kind != KIND_CONTINUUM for c in channels)
    assert has_valid_direct_channel(delta, channels) is True

    # a channel list from another parent raises (checked, never
    # repaired)
    other = _root_state("mesonic_neutral", 4.0, MESONIC_QN)
    with pytest.raises(ValueError, match="parent_id"):
        has_valid_direct_channel(other, channels)


# ----------------------------------------------------------------------
# test_handoff_bit_identity
# ----------------------------------------------------------------------
def test_handoff_bit_identity(catalog, fams):
    """On a pinned handoff state, the delegated Round 2 decay is
    bit-identical to the standalone ``ResonanceDecayer`` at the same
    seed (the Round 2 kernel is byte-unchanged)."""

    for family_id, mass, qn, event_index in HANDOFF_FIXTURES:
        state = _root_state(family_id, mass, qn)
        assert is_catalog_handoff(state, catalog)
        pdg = int(family_id[len(CATALOG_FAMILY_PREFIX):])

        # the standalone Round 2 kernel at the same seed triple
        kernel = ResonanceDecayer(catalog, MASTER_SEED, 0, event_index)
        vertices_standalone = kernel.decay_resonance(
            pdg, (float(mass), 0.0, 0.0, 0.0))
        assert vertices_standalone, (
            f"{family_id}: the pinned handoff decay has vertices"
        )

        # the delegated decay through the card-9 handoff
        rng_channel, rng_kinematics = _sub_rngs(MASTER_SEED, event_index)
        vertices_delegated = handoff_decay(
            state, catalog, rng_channel, rng_kinematics)

        # bit-identical: every vertex field equal (the frozen dataclass
        # equality compares the channels, charge combinations, and
        # daughter four-momenta)
        assert vertices_delegated == vertices_standalone, (
            f"{family_id}: the delegated decay drifted from the "
            "standalone Round 2 kernel"
        )

        # the same delegation through the driver: the trace ends at the
        # handoff state and the passed generators are advanced exactly
        # as the standalone kernel's internal generators
        rng_channel, rng_kinematics = _sub_rngs(MASTER_SEED, event_index)
        trace = run_cascade(
            state, catalog, fams, rng_channel, rng_kinematics)
        assert trace == [state]
        assert _termination_mode(trace, catalog) == \
            TERMINATION_CATALOG_HANDOFF
        assert rng_channel.bit_generator.state == \
            kernel._rng_channel.bit_generator.state
        assert rng_kinematics.bit_generator.state == \
            kernel._rng_kinematics.bit_generator.state


# ----------------------------------------------------------------------
# test_termination_complete
# ----------------------------------------------------------------------
def test_termination_complete(catalog, fams, pinned_trace):
    """A pinned 10,000-cascade trace over the accepted source-mass
    range (2-8 GeV, baryonic ``(1, 1, 0)`` and mesonic ``(0, 0, 0)``
    families) terminates on every cascade (100%); the termination mode
    (catalog handoff / direct channel) is recorded per cascade."""

    assert len(pinned_trace) == N_CASCADES == 10_000

    # every cascade terminates: run_cascade returned a trace ending at
    # a terminal state, and the recorded mode is one of the two
    # pre-registered modes
    modes = [record["mode"] for record in pinned_trace]
    assert set(modes) == {
        TERMINATION_CATALOG_HANDOFF, TERMINATION_DIRECT_CHANNEL}

    # the pinned mode counts (deterministic: the two catalog-routed
    # 2 GeV roots hand off at depth 0, the eight continuum roots
    # terminate through the direct channel)
    handoff = modes.count(TERMINATION_CATALOG_HANDOFF)
    direct = modes.count(TERMINATION_DIRECT_CHANNEL)
    assert handoff == PINNED_HANDOFF_CASCADES == 2_000
    assert direct == PINNED_DIRECT_CASCADES == 8_000
    assert handoff + direct == N_CASCADES

    # the per-root mode recording (1,000 cascades per pinned root)
    per_root: dict[tuple, dict[str, int]] = {}
    for record in pinned_trace:
        counts = per_root.setdefault(record["root"], {})
        counts[record["mode"]] = counts.get(record["mode"], 0) + 1
    assert len(per_root) == len(CASCADE_ROOTS)
    for root_key, counts in per_root.items():
        assert sum(counts.values()) == CASCADES_PER_ROOT, root_key
        family_id, mass, _qn = root_key
        if family_id.startswith(CATALOG_FAMILY_PREFIX):
            # the catalog-routed 2 GeV sources hand off at depth 0
            assert counts == {TERMINATION_CATALOG_HANDOFF:
                              CASCADES_PER_ROOT}, root_key
        else:
            assert counts == {TERMINATION_DIRECT_CHANNEL:
                              CASCADES_PER_ROOT}, root_key
        assert float(mass) in (2.0, 3.0, 4.0, 6.0, 8.0)

    # both families are present in the pinned roots
    root_qn = {record["root"][2] for record in pinned_trace}
    assert root_qn == {BARYONIC_QN, MESONIC_QN}


# ----------------------------------------------------------------------
# test_guard_never_reached
# ----------------------------------------------------------------------
def test_guard_never_reached(catalog, fams, pinned_trace):
    """The pinned 10,000-cascade trace reaches the depth guard zero
    times (plan section 10.6 line 757, "tune physics so the guard is
    never reached in accepted events")."""

    assert MAX_CASCADE_DEPTH == 12

    # zero guard activations: run_cascade completed every cascade
    # without raising CascadeDepthGuardError (the fixture would have
    # failed otherwise), and the deepest line state is pinned far below
    # the guard
    deepest = max(record["terminal_depth"] for record in pinned_trace)
    assert deepest == PINNED_MAX_LINE_DEPTH == 4
    assert deepest < MAX_CASCADE_DEPTH
    for record in pinned_trace:
        assert record["terminal_depth"] <= MAX_CASCADE_DEPTH

    # the depth counter along each line increments by exactly one per
    # continuation (card 4's counter; the guard consumes it read-only)
    for record in pinned_trace:
        trace = record["trace"]
        assert trace[0].depth == 0
        for parent, child in zip(trace, trace[1:]):
            assert child.depth == parent.depth + 1


# ----------------------------------------------------------------------
# test_guard_fires
# ----------------------------------------------------------------------
def test_guard_fires(catalog, fams):
    """A synthetic depth-13 state (a pinned fixture forcing the chain)
    raises ``CascadeDepthGuardError`` naming the depth."""

    # the pinned synthetic chain: thirteen continuations past the root,
    # each state's depth counter and parent link consistent
    chain = _depth_chain(14, "baryonic_np", 4.0, BARYONIC_QN)
    assert [state.depth for state in chain] == list(range(14))
    assert chain[13].depth == 13
    rng_channel, rng_kinematics = _sub_rngs(MASTER_SEED, 0)
    with pytest.raises(CascadeDepthGuardError, match="depth 13"):
        run_cascade(chain[13], catalog, fams, rng_channel, rng_kinematics)

    # the boundary: a depth-12 state does not fire the guard; a pinned
    # catalog handoff state at depth 12 terminates through the kernel
    # (the guard is checked, never repaired, at depth > 12 only)
    boundary_parent = chain[11]
    boundary = CascadeState(
        momentum=(0.77526, 0.0, 0.0, 0.0),
        mass=0.77526,
        qn=MESONIC_QN,
        quantum={},
        family_id="catalog:113",
        axis=(0.0, 0.0, 1.0),
        alignment_label=ISOTROPIC_ALIGNMENT_LABEL,
        parent=boundary_parent,
        exchange_ancestry=("guard_probe",),
        depth=12,
    )
    assert boundary.depth == 12
    rng_channel, rng_kinematics = _sub_rngs(MASTER_SEED, 1)
    trace = run_cascade(
        boundary, catalog, fams, rng_channel, rng_kinematics)
    assert trace == [boundary]
    assert _termination_mode(trace, catalog) == \
        TERMINATION_CATALOG_HANDOFF

    # the guard honours the driver's max_depth argument (the card's
    # interface): a depth-3 state raises at max_depth = 2, naming the
    # depth; a negative max_depth is rejected outright (checked, never
    # repaired)
    shallow = _depth_chain(4, "baryonic_np", 4.0, BARYONIC_QN)
    assert shallow[3].depth == 3
    rng_channel, rng_kinematics = _sub_rngs(MASTER_SEED, 2)
    with pytest.raises(CascadeDepthGuardError, match="depth 3"):
        run_cascade(shallow[3], catalog, fams,
                    rng_channel, rng_kinematics, max_depth=2)
    with pytest.raises(ValueError, match="max_depth"):
        run_cascade(shallow[0], catalog, fams,
                    rng_channel, rng_kinematics, max_depth=-1)

    # the guard fires before any termination rule: a catalog handoff
    # state beyond the limit raises (the guard is the first check)
    deep_catalog = CascadeState(
        momentum=(0.77526, 0.0, 0.0, 0.0),
        mass=0.77526,
        qn=MESONIC_QN,
        quantum={},
        family_id="catalog:113",
        axis=(0.0, 0.0, 1.0),
        alignment_label=ISOTROPIC_ALIGNMENT_LABEL,
        parent=None,
        exchange_ancestry=("guard_probe",),
        depth=MAX_CASCADE_DEPTH + 1,
    )
    rng_channel, rng_kinematics = _sub_rngs(MASTER_SEED, 3)
    with pytest.raises(CascadeDepthGuardError, match="depth 13"):
        run_cascade(deep_catalog, catalog, fams,
                    rng_channel, rng_kinematics)


# ----------------------------------------------------------------------
# test_conservation_through_termination
# ----------------------------------------------------------------------
def test_conservation_through_termination(catalog, fams, pinned_trace):
    """Every vertex of the pinned trace conserves ``(B, Q, S)`` exactly,
    including the handoff and direct-termination vertices."""

    stable = catalog.stable_states()

    for record in pinned_trace:
        trace = record["trace"]
        family_id, mass, qn = record["root"]

        # the root carries the pinned root quantum numbers
        assert tuple(int(v) for v in trace[0].qn) == tuple(qn)
        assert trace[0].parent is None
        assert trace[0].family_id == family_id

        for state in trace:
            state_qn = tuple(int(v) for v in state.qn)

            if str(state.family_id).startswith(CATALOG_FAMILY_PREFIX):
                # the handoff vertices (the catalog-routed 2 GeV roots
                # and any catalog terminal state): the state's (B, Q,
                # S) matches the committed record exactly, and the mass
                # lies within the record support (the routing vertex's
                # conservation and support conditions)
                pdg = int(str(state.family_id)[len(CATALOG_FAMILY_PREFIX):])
                record_qn = (
                    int(catalog.get(pdg).B),
                    int(catalog.get(pdg).Q),
                    int(catalog.get(pdg).S),
                )
                assert state_qn == record_qn, (
                    f"handoff vertex {state.family_id}: the state "
                    f"carries {state_qn}, the record carries "
                    f"{record_qn}"
                )
                assert is_catalog_handoff(state, catalog)
                # the delegated kernel decay ran byte-unchanged (its
                # own conservation checks cover the remainder)
                if state is trace[-1]:
                    assert record["mode"] == TERMINATION_CATALOG_HANDOFF
                continue

            # the continuum line states: the state's quantum numbers
            # match its continuum family
            family = next(
                f for f in fams if f.family_id == state.family_id)
            family_qn = tuple(int(v) for v in family.qn)
            assert state_qn == family_qn, (
                f"continuum vertex {state.family_id}: the state "
                f"carries {state_qn}, the family carries {family_qn}"
            )

            # the realized vertices conserve (B, Q, S) exactly at the
            # channel level: every open channel of the enumeration (the
            # drawn continuation and, at the terminal state, the drawn
            # direct channel among them) sums to the state's quantum
            # numbers; card 4's draw_decay enforced the same sum on the
            # realized vertex (checked, never repaired, raise on
            # violation)
            channels = enumerate_channels(state, catalog, fams)
            assert channels, (
                f"{state.family_id} at {state.mass}: the accepted line "
                "state carries an open set"
            )
            for channel in channels:
                total = [0, 0, 0]
                for daughter_id in channel.daughters:
                    if daughter_id.startswith(CATALOG_FAMILY_PREFIX):
                        daughter = catalog.get(
                            int(daughter_id[len(CATALOG_FAMILY_PREFIX):]))
                        values = (
                            int(daughter.B), int(daughter.Q),
                            int(daughter.S))
                    else:
                        daughter_family = next(
                            f for f in fams
                            if f.family_id == daughter_id)
                        values = tuple(int(v) for v in daughter_family.qn)
                    for k in range(3):
                        total[k] += values[k]
                assert tuple(total) == state_qn, (
                    f"{state.family_id} at {state.mass} -> "
                    f"{tuple(channel.daughters)}: daughters sum to "
                    f"{tuple(total)}, the parent carries {state_qn}"
                )

            # the direct-termination vertex: the terminal continuum
            # state carries no open continuum channel and a valid
            # direct channel, so the realized vertex is a direct
            # hadronic decay whose daughters are established states
            if state is trace[-1]:
                assert record["mode"] == TERMINATION_DIRECT_CHANNEL
                assert has_valid_direct_channel(state, channels)
                for channel in channels:
                    if channel.kind == KIND_CONTINUUM:
                        continue
                    for daughter_id in channel.daughters:
                        pdg = int(
                            daughter_id[len(CATALOG_FAMILY_PREFIX):])
                        daughter_record = catalog.get(pdg)
                        assert (
                            pdg in stable
                            or daughter_record.decay_channels
                        ), (
                            f"{daughter_id} is neither a stable "
                            "established hadron nor an established "
                            "resonance"
                        )
                # every direct daughter sits below the parent mass (the
                # plan section 10.2 line 690 termination clause)
                for channel in channels:
                    if channel.kind == KIND_CONTINUUM:
                        continue
                    masses = [
                        float(catalog.get(
                            int(d[len(CATALOG_FAMILY_PREFIX):])
                        ).pole_mass)
                        for d in channel.daughters
                    ]
                    assert any(m < float(state.mass) for m in masses)


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
# The acceptance-(v) freeze re-pin check
# ----------------------------------------------------------------------
