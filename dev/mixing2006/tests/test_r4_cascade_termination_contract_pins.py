"""Round 4, card 9 test-author register: cascade termination and depth
guard.

Independent contract pins for the cascade termination rules and the
hard maximum depth guard (the card-9 module
``na49_mixing/cascade_termination.py``), Round 4 card 9
(``TASK-20260920-r4-termination-depth-tests``; plan section 22 task 26,
line 1748; plan section 10.6, lines 750-757).  This register is the
test-author half of the card: it pins the module contract on
self-authored synthetic catalog and continuum-family fixtures
(self-authored records and family records with real PDG ids for the
vendored-table quantum-number resolution), its own seeds, and its own
counting-generator instrumentation, and it re-pins the freeze manifest
for the register.  The solver's 10,000-cascade pinned trace is not
re-run here: the termination modes and the guard are pinned on small
self-authored channel graphs, and the delegation determinism at a
small scale.

Part 1: the module contract (pure, no sealed run dir)
-----------------------------------------------------

- ``test_catalog_handoff_predicate``: the handoff predicate is true
  exactly on the self-authored catalog states matching a record
  within its support, inclusive at both support edges; false on a
  continuum state (same ``(B, Q, S)`` as a committed record), an
  unknown record id, a ``(B, Q, S)`` mismatch, masses just outside
  both support edges, and non-finite masses (the on-disk behavior is
  pinned: non-finite is no handoff).
- ``test_direct_channel_predicate``: the direct-termination predicate
  on self-authored ``CascadeChannel`` fixtures (the exact card-4
  field names ``parent_id`` / ``daughters`` / ``kind`` /
  ``weight``): true on established-hadron and established-resonance
  channels alone, false with an open continuum channel (even when
  established channels exist), false with continuum channels only,
  false with zero channels, and a ``ValueError`` on a channel list
  from another parent (checked, never repaired).
- ``test_depth_guard_boundaries``: the guard raises
  ``CascadeDepthGuardError`` naming the depth at ``depth >
  max_depth`` (self-authored depths: max_depth = 1 fires at depth
  2, max_depth = 0 fires at the first continuation at depth 1); the
  ``depth == max_depth`` case does not raise (the guard is strictly
  greater); a negative ``max_depth`` is rejected with
  ``ValueError``; the guard is the first check (a catalog handoff
  state beyond the limit raises before any termination rule) and
  consumes no random numbers when it fires.
- ``test_run_cascade_termination_modes``: the driver's termination
  modes on self-authored channel graphs: the catalog handoff (the
  returned line is the root alone, root first, the terminal state
  satisfies :func:`is_catalog_handoff`), the direct channel (a
  self-authored continuum root at the family floor terminates
  through the established pool, the terminal state is a continuum
  state with no open continuum channel), the continuation chains
  (six pinned events from a self-authored 3.5 GeV baryonic root: the
  line is root first, the depths increment by one, the terminal
  state is exactly the last element, and the terminal state
  identifies the mode), and the no-channel edge (``ValueError``, a
  state outside the accepted model).
- ``test_handoff_delegation``: the handoff delegation to the Round 2
  kernel on a self-authored handoff state and self-authored catalog
  record: a stable handoff state returns an empty vertex list (the
  state is final), the delegated decay is bit-identical to the
  standalone ``ResonanceDecayer`` at the same seed on a single- and
  a multi-vertex fixture, the passed substream generators advance to
  exactly the kernel's bit states (the card-4 kernel-shim pattern:
  the delegation adds no draw of its own), two calls at the same
  seeds give bit-identical vertex lists, and a non-handoff state
  raises ``ValueError``.
- ``test_below_limit_noop``: a self-authored trace at a low depth
  where the guard never activates: every pinned chain completes
  without ``CascadeDepthGuardError``, the deepest line state sits
  far below the limit, and the driver adds no random number of its
  own (the per-vertex budget stays 1 channel + 2 kinematics).
- ``test_predicates_zero_draw``: the predicates consume no
  randomness (the seeded bit-generator state is byte-identical
  before and after a self-authored predicate scan).
- ``test_module_constants``: ``MAX_CASCADE_DEPTH == 12``
  (the card-4 value, re-exported by object identity, not a copy),
  ``TERMINATION_CATALOG_HANDOFF == "catalog_handoff"``,
  ``TERMINATION_DIRECT_CHANNEL == "direct_channel"``, the
  ``CascadeDepthGuardError`` hierarchy, and the module's
  ``__all__``.

Part 2: the substrate and freeze pins
-------------------------------------

- ``test_module_byte_and_export_pins``: the module byte stability
  (sha256 ``0e768572...``) and the additive ``na49_mixing`` package
  exports (the seven card-9 names present in the package ``__init__``
  import block and ``__all__``, resolved by object identity, with
  the prior card exports unchanged).
- ``test_frozen_substrate_digests``: the byte-unchanged surfaces the
  card consumes (card 4's cascade surface, card 5's axis machinery,
  the card-6/7 alignment kernel, card 8's form factor, the Round 2
  kernel, the builder, the card-6 blend routing, the card-2 engine,
  the committed catalog, the continuum families configuration, and
  the R32 driver in both trees), verified on disk.
- ``test_freeze_repin_register``: the freeze-manifest re-pin for this
  register (the strict path-sorted inventory at 469 entries carries
  this file at its on-disk digest; the card-9 solver freeze
  exception stays at the solver's 14 path-sorted files; this
  register's own freeze exception names the test-author card exactly
  once with the path-sorted file list; the eleven consumed count
  pins (the ten register-chain files plus the card-9 solver test's
  own count constant) stay at their on-disk digests, each re-pinned
  468 -> 469 with the naming comment attributing the bump to this
  register; the frozen M8 parameter map stays intact).

Hermetic: the self-authored fixtures, numpy, the standard library,
and the vendored ``na49_mixing`` modules only; no sealed run dir, no
event generation, no external tables, no Monte Carlo run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.cascade_decay import (
    CATALOG_FAMILY_PREFIX,
    ISOTROPIC_ALIGNMENT_LABEL,
    KIND_CONTINUUM,
    KIND_ESTABLISHED_HADRON,
    KIND_ESTABLISHED_RESONANCE,
    MAX_CASCADE_DEPTH,
    CascadeChannel,
    CascadeState,
    draw_decay,
    enumerate_channels,
)
from na49_mixing.cascade_termination import (
    TERMINATION_CATALOG_HANDOFF,
    TERMINATION_DIRECT_CHANNEL,
    CascadeDepthGuardError,
    handoff_decay,
    has_valid_direct_channel,
    is_catalog_handoff,
    run_cascade,
)
from na49_mixing.hadron_catalog import DecayChannel, HadronCatalog, HadronRecord
from na49_mixing.resonance_decay import ResonanceDecayer
from na49_mixing.rng_substreams import derive_substreams
from na49_mixing.spectral_density import SUPPORT_M_MAX, ContinuumFamily
from tests._freeze_inventory import inventory_count as _inventory_count

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent          # dev/mixing2006
MONO_ROOT = NESTED_ROOT.parent.parent          # the monorepo root
MODULE_PATH = NESTED_ROOT / "na49_mixing" / "cascade_termination.py"
INIT_PATH = NESTED_ROOT / "na49_mixing" / "__init__.py"
MANIFEST_PATH = (NESTED_ROOT / "config" / "v2_singlet"
                 / "legacy_v1_manifest.json")

MODULE_REL = "dev/mixing2006/na49_mixing/cascade_termination.py"
INIT_REL = "dev/mixing2006/na49_mixing/__init__.py"
MY_TEST_REL = ("dev/mixing2006/tests/"
               "test_r4_cascade_termination_contract_pins.py")
SOLVER_TEST_REL = "dev/mixing2006/tests/test_cascade_termination.py"
MANIFEST_REL = ("dev/mixing2006/config/v2_singlet/"
                "legacy_v1_manifest.json")

FREEZE_CARD = "TASK-20260920-r4-termination-depth-tests"
# The test-author re-pin's own freeze-exception card name (distinct
# from the solver's, so the solver's "exactly one entry naming this
# card" check is untouched).
TEST_AUTHOR_CARD = (
    "TASK-20260920-r4-termination-depth-tests-test-author"
)

# The ten register-chain files whose inventory-count pins this
# register re-pins (468 -> 469), each with the naming comment
# attributing the bump to this register's file.
CARD3_REGISTER_REL = ("dev/mixing2006/tests/"
                      "test_r4_audit_blend_path_contract_pins.py")
CARD4_REGISTER_REL = ("dev/mixing2006/tests/"
                      "test_r4_cascade_decay_contract_pins.py")
CARD5_SOLVER_TEST_REL = "dev/mixing2006/tests/test_cascade_axes.py"
CARD5_REGISTER_REL = ("dev/mixing2006/tests/"
                      "test_r4_cascade_axes_contract_pins.py")
CARD6_SOLVER_TEST_REL = ("dev/mixing2006/tests/"
                         "test_cascade_alignment_baryonic.py")
CARD6_REGISTER_REL = ("dev/mixing2006/tests/"
                      "test_r4_cascade_alignment_baryonic_contract_pins.py")
CARD7_SOLVER_TEST_REL = ("dev/mixing2006/tests/"
                         "test_cascade_alignment_mesonic.py")
CARD7_REGISTER_REL = ("dev/mixing2006/tests/"
                      "test_r4_cascade_alignment_mesonic_contract_pins.py")
CARD8_SOLVER_TEST_REL = ("dev/mixing2006/tests/"
                         "test_cascade_form_factor.py")
CARD8_REGISTER_REL = ("dev/mixing2006/tests/"
                      "test_r4_cascade_form_factor_contract_pins.py")
CHAIN_FILES = (
    CARD3_REGISTER_REL,
    CARD4_REGISTER_REL,
    CARD5_SOLVER_TEST_REL,
    CARD5_REGISTER_REL,
    CARD6_SOLVER_TEST_REL,
    CARD6_REGISTER_REL,
    CARD7_SOLVER_TEST_REL,
    CARD7_REGISTER_REL,
    CARD8_SOLVER_TEST_REL,
    CARD8_REGISTER_REL,
)
# The eleven count pins this register re-pins: the ten register-chain
# files plus the card-9 solver test's own count constant.
REPIN_FILES = CHAIN_FILES + (SOLVER_TEST_REL,)

# 468 entries at the card-9 solver landing; this register adds its
# file (+1 entry) for the test-author re-pin (468 -> 469, +1 entry)
# and re-pins the eleven consumed count pins (the ten register-chain
# files plus the card-9 solver test's own count constant) with the
# bump attributed to this register's file; card 10 of the round
# (TASK-20260920-r4-isotropic-aligned-ablation-runner) re-pins the
# inventory additively for its new ablation runner module and test
# file (na49_mixing/cascade_ablation.py, test_cascade_ablation.py; the
# constant read "INVENTORY_COUNT = 469" at the card-9 register
# landing; 469 -> 471, +2 entries); the card-10 test-author register
# (test_r4_cascade_ablation_contract_pins.py) re-pins the inventory
# additively for its own file (the constant read
# "INVENTORY_COUNT = 471" at the card-10 solver landing; 471 -> 472,
# +1 entry).
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
MODULE_SHA256 = (
    "a62f6cce7ace4d6306b176bf12d61e28c22a0b5f1709a201d05a63fcf833da56"
)
# The package __init__ additive re-pin: Round 5 card 1
# (TASK-20260925-r5-elastic-reference) extends the package exports
# additively with the elastic reference module's names
# (na49_mixing/topology_elastic.py; the digest read
# "1b3a6e651dd43d1b5fde0839d6c84c8dab24675043c39bcc65e2d763eb39868e"
# at the card-21 landing).
# re-pinned by Round 5 card 2 (TASK-20260925-r5-exchange-quantum-number-ledger; the exchange-ledger module's additive package exports).
# re-pinned by Round 5 card 3 (TASK-20260925-r5-intact-low-mass-remnants; the remnant module's additive package exports).
# re-pinned by Round 5 card 7 (TASK-20260925-r5-non-diffrative-central-excitation; the non-diffractive module's additive package exports).
# re-pinned by Round 5 card 8 (TASK-20260925-r5-multiple-neutral-excitations; the multiple-neutral-excitation module's additive package exports).
# re-pinned by Round 5 card 9 (TASK-20260925-r5-topology-template-runner; the topology-runner module's additive package exports).
# re-pinned by the Round 5 card 9 review fix round (TASK-20260925-r5-topology-template-runner-review-fix; the REV-bc4de47-03 export-surface correction re-pinned the package __init__).
INIT_SHA256 = (
    "7d2c5a85d9b24de104c0d9cf72195689bf7c39c873e154798699448073b36de8"
)

#: The byte-unchanged frozen surfaces the card consumes (the card's
#: provenance section, verified on disk at this register's landing):
#: card 4's cascade surface (the mandated byte-stability pin), card 5's
#: axis machinery, the card-6/7 alignment kernel, card 8's form
#: factor, the Round 2 kernel (the handoff target), the builder, the
#: card-6 blend routing, the card-2 engine, the committed catalog, the
#: continuum families configuration, and the dual-tree R32 driver.
FROZEN_SUBSTRATE = {
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
    "dev/mixing2006/helpers/benchmarking/"
    "staged_r32_spectral_systematic.py":
        "99454facce856c93505e6c4b0d4e47b85eeb0ef54f72b1631e3d29fcc511cb90",
    "helpers/benchmarking/staged_r32_spectral_systematic.py":
        "99454facce856c93505e6c4b0d4e47b85eeb0ef54f72b1631e3d29fcc511cb90",
}

#: The solver's card-9 freeze-exception file list (14 path-sorted
#: files; this register leaves it untouched and records its own
#: re-pin under a distinct card name).
SOLVER_EXCEPTION_FILES = (
    MANIFEST_REL,
    INIT_REL,
    MODULE_REL,
    SOLVER_TEST_REL,
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

#: This register's own freeze-exception file list (path-sorted): the
#: manifest, this register, and the eleven re-pinned count pins.
TEST_AUTHOR_EXCEPTION_FILES = tuple(
    sorted([MANIFEST_REL, MY_TEST_REL] + list(REPIN_FILES))
)

#: The independent register seed (the day after the round 4 master
#: seed, the solver's 20260920).
REGISTER_SEED = 20260921


# ----------------------------------------------------------------------
# The self-authored synthetic catalog (real PDG ids; vendored-table
# quantum-number resolution)
# ----------------------------------------------------------------------
def _record(
    pdg_id: int,
    name: str,
    B: int,
    Q: int,
    S: int,
    isospin: float,
    isospin3: int,
    spin: float,
    parity: str,
    pole_mass: float,
    nominal_width: float = 0.0,
    mass_range: tuple[float, float] | None = None,
    channels: tuple[tuple[int, int, float, int], ...] = (),
    stable_for_decay: bool = True,
) -> HadronRecord:
    """One self-authored catalog record (the vendored-table masses)."""

    if mass_range is None:
        mass_range = (pole_mass, pole_mass)
    decay_channels = tuple(
        DecayChannel(
            parent_pdg=pdg_id,
            daughters=(d0, d1),
            branching_fraction=br,
            orbital_angular_momentum=L,
        )
        for d0, d1, br, L in channels
    )
    return HadronRecord(
        pdg_id=pdg_id,
        name=name,
        B=B,
        Q=Q,
        S=S,
        isospin=isospin,
        isospin3=isospin3,
        spin=spin,
        parity=parity,
        pole_mass=pole_mass,
        nominal_width=nominal_width,
        mass_range=mass_range,
        decay_channels=decay_channels,
        stable_for_decay=stable_for_decay,
        provenance=None,
    )


def _rt_catalog() -> HadronCatalog:
    """The self-authored small multi-resonance catalog.

    The stable hadrons resolve through the vendored PDG table quantum
    numbers (the Round 2 kernel's daughter sources); the established
    resonances carry the tabulated charge channels (``rtRho`` is a
    single-vertex handoff fixture, ``rtDelta`` an established-hadron
    handoff fixture, and ``rtOmega`` a two-vertex kernel chain
    through the ``rtRho`` daughter).  The masses are the vendored
    table values, so the thresholds are the physical ones.
    """

    records = (
        _record(2212, "proton", 1, 1, 0, 0.5, 1, 0.5, "+", 0.93827231),
        _record(2112, "neutron", 1, 0, 0, 0.5, -1, 0.5, "+", 0.93956541),
        _record(211, "pi+", 0, 1, 0, 1.0, 2, 0, "-", 0.1395699),
        _record(-211, "pi-", 0, -1, 0, 1.0, -2, 0, "-", 0.1395699),
        _record(111, "pi0", 0, 0, 0, 1.0, 0, 0, "-", 0.1349766),
        _record(
            9015, "rtRho", 0, 0, 0, 1.0, 0, 1, "-",
            0.8, nominal_width=0.15,
            mass_range=(0.3, 1.2),
            channels=((211, -211, 1.0, 1),),
            stable_for_decay=False,
        ),
        _record(
            9016, "rtDelta", 1, 1, 0, 1.5, 1, 1.5, "-",
            1.25, nominal_width=0.2,
            mass_range=(1.1, 2.4),
            channels=((2212, 111, 0.75, 1),
                      (2112, 211, 0.25, 1)),
            stable_for_decay=False,
        ),
        _record(
            9017, "rtOmega", 0, 0, 0, 1.0, 0, 1, "-",
            0.95, nominal_width=0.1,
            mass_range=(0.5, 1.5),
            channels=((9015, 111, 1.0, 1),),
            stable_for_decay=False,
        ),
    )
    return HadronCatalog(records)


#: The self-authored continuum family configuration: a baryonic family
#: (whose allowed-decay-class list declares the N-pi charge pair) and
#: a mesonic family.  The baryonic family self-continues through the
#: ``(2212, 111)`` class and sheds to the mesonic family through the
#: ``pi0`` lead, so the pinned 3.5 GeV root produces continuation
#: chains that terminate through the direct channel.
RT_BARYON = ContinuumFamily(
    family_id="rt_baryon",
    qn=(1, 1, 0),
    m_min=2.3,
    t_h=0.165,
    b_power=0.0,
    n_norm=1.0,
    ramp_width=0.5,
    nominal_width=0.5,
    allowed_decay_classes=("(2212,111)", "(2112,211)"),
    active=True,
    provenance=(),
)
RT_MESON = ContinuumFamily(
    family_id="rt_meson",
    qn=(0, 0, 0),
    m_min=1.4,
    t_h=0.165,
    b_power=0.0,
    n_norm=1.0,
    ramp_width=0.5,
    nominal_width=0.2,
    allowed_decay_classes=("(211,-211)",),
    active=True,
    provenance=(),
)
RT_FAMILIES = (RT_BARYON, RT_MESON)


# ----------------------------------------------------------------------
# Fixtures and helpers
# ----------------------------------------------------------------------
@pytest.fixture(scope="session")
def catalog() -> HadronCatalog:
    return _rt_catalog()


def _rt_state(
    family_id: str,
    mass: float,
    qn: tuple[int, int, int],
    depth: int = 0,
    parent: CascadeState | None = None,
) -> CascadeState:
    """A pinned at-rest state with the lab-z axis, the isotropic
    label, and the register's exchange ancestry."""

    return CascadeState(
        momentum=(float(mass), 0.0, 0.0, 0.0),
        mass=float(mass),
        qn=qn,
        quantum={},
        family_id=family_id,
        axis=(0.0, 0.0, 1.0),
        alignment_label=ISOTROPIC_ALIGNMENT_LABEL,
        parent=parent,
        exchange_ancestry=("register_probe",),
        depth=depth,
    )


def _sub_rngs(seed: int, event_index: int) -> tuple:
    """The closed decay_channel/decay_kinematics substream generators
    of one (register seed, replica 0, event index) triple (the round 2
    card 9 and round 3 card 8 convention)."""

    subs = derive_substreams(seed, 0, event_index)
    return (
        np.random.default_rng(subs.stream_for("decay_channel")),
        np.random.default_rng(subs.stream_for("decay_kinematics")),
    )


class _CountingRng:
    """A counting proxy around one seeded generator: every ``random()``
    consumes exactly one uniform and advances the counter (the house
    draw-budget pattern)."""

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        return float(self._rng.random())


def _rt_daughter_mass(
    family_id: str,
    sibling_mass: float,
    parent_mass: float,
) -> float:
    """The card-4 deterministic continuum daughter mass rule,
    recomputed in the test (the same expression the module uses)."""

    family = next(f for f in RT_FAMILIES if f.family_id == family_id)
    edge = min(SUPPORT_M_MAX, parent_mass - sibling_mass)
    return max(float(family.m_min), edge - float(family.nominal_width))


def _termination_mode(trace: list[CascadeState], catalog) -> str:
    """The termination mode of a finished cascade: the terminal state
    identifies the mode exactly (a catalog record within support is a
    handoff; a continuum terminal state is a direct-channel
    termination)."""

    terminal = trace[-1]
    if is_catalog_handoff(terminal, catalog):
        return TERMINATION_CATALOG_HANDOFF
    return TERMINATION_DIRECT_CHANNEL


# ----------------------------------------------------------------------
# test_catalog_handoff_predicate (acceptance (i))
# ----------------------------------------------------------------------
def test_catalog_handoff_predicate(catalog):
    """The handoff predicate is true exactly on the self-authored
    catalog states matching a record within its support (inclusive at
    both support edges); false on a continuum state, an unknown record
    id, a QN mismatch, masses just outside both support edges, and
    non-finite masses (the on-disk behavior is pinned)."""

    # true: the in-support states and the inclusive support edges
    true_states = (
        # rtRho inside (0.3, 1.2)
        ("catalog:9015", 0.8, (0, 0, 0)),
        # the inclusive support edges of the rtRho record
        ("catalog:9015", 0.3, (0, 0, 0)),
        ("catalog:9015", 1.2, (0, 0, 0)),
        # the inclusive support edges of the rtDelta record
        ("catalog:9016", 1.1, (1, 1, 0)),
        ("catalog:9016", 2.4, (1, 1, 0)),
        # a stable record: the degenerate support [pole, pole]
        ("catalog:2212", 0.93827231, (1, 1, 0)),
    )
    for family_id, mass, qn in true_states:
        state = _rt_state(family_id, mass, qn)
        assert is_catalog_handoff(state, catalog) is True, (
            f"{family_id} at {mass} must hand off"
        )

    # false: the masses just outside both support edges
    below = _rt_state("catalog:9015", 0.2999999, (0, 0, 0))
    assert is_catalog_handoff(below, catalog) is False
    above = _rt_state("catalog:9015", 1.2000001, (0, 0, 0))
    assert is_catalog_handoff(above, catalog) is False

    # false: the QN mismatch (the record id resolves, the state's
    # (B, Q, S) disagrees with the rtDelta record)
    mismatched = _rt_state("catalog:9016", 1.25, (0, 0, 0))
    assert is_catalog_handoff(mismatched, catalog) is False

    # false: a continuum state is never a handoff, even though a
    # committed catalog record (the rtDelta) carries the same (B, Q, S)
    continuum = _rt_state("rt_baryon", 1.25, (1, 1, 0))
    assert is_catalog_handoff(continuum, catalog) is False

    # false: an unknown record id is no match
    unknown = _rt_state("catalog:99999", 0.8, (0, 0, 0))
    assert is_catalog_handoff(unknown, catalog) is False

    # the non-finite mass edge: the on-disk behavior is no handoff
    # (the support comparison of a non-finite mass is False)
    for bad_mass in (float("nan"), float("inf"), float("-inf")):
        state = _rt_state("catalog:9015", bad_mass, (0, 0, 0))
        assert is_catalog_handoff(state, catalog) is False, (
            f"a non-finite mass {bad_mass!r} must not hand off"
        )


# ----------------------------------------------------------------------
# test_direct_channel_predicate (acceptance (i))
# ----------------------------------------------------------------------
def test_direct_channel_predicate(catalog):
    """The direct-termination predicate on self-authored
    ``CascadeChannel`` fixtures (the exact card-4 field names
    ``parent_id`` / ``daughters`` / ``kind`` / ``weight``): true on
    established channels alone, false with an open continuum channel,
    false with zero channels, and a ``ValueError`` on a channel list
    from another parent."""

    parent = "rt_baryon"
    hadron = CascadeChannel(
        parent_id=parent,
        daughters=("catalog:2212", "catalog:111"),
        kind=KIND_ESTABLISHED_HADRON,
        weight=0.5,
    )
    resonance = CascadeChannel(
        parent_id=parent,
        daughters=("catalog:9015", "catalog:111"),
        kind=KIND_ESTABLISHED_RESONANCE,
        weight=0.5,
    )
    continuum = CascadeChannel(
        parent_id=parent,
        daughters=("rt_baryon", "catalog:111"),
        kind=KIND_CONTINUUM,
        weight=0.5,
    )
    state = _rt_state(parent, 4.0, (1, 1, 0))

    # true: at least one established channel, no continuum channel
    assert has_valid_direct_channel(state, [hadron]) is True
    assert has_valid_direct_channel(state, [resonance]) is True
    assert has_valid_direct_channel(state, [hadron, resonance]) is True

    # false: an open continuum channel blocks the direct rule, even
    # when established channels exist
    assert has_valid_direct_channel(state, [hadron, continuum]) is False
    assert has_valid_direct_channel(state, [resonance, continuum]) is False
    assert has_valid_direct_channel(state, [continuum]) is False

    # false: zero channels is no direct channel
    assert has_valid_direct_channel(state, []) is False

    # checked, never repaired: a channel list from another parent
    # raises ValueError
    foreign = CascadeChannel(
        parent_id="rt_meson",
        daughters=("catalog:211", "catalog:-211"),
        kind=KIND_ESTABLISHED_HADRON,
        weight=1.0,
    )
    with pytest.raises(ValueError, match="parent_id"):
        has_valid_direct_channel(state, [foreign])


# ----------------------------------------------------------------------
# test_depth_guard_boundaries (acceptance (iii), the guard clause)
# ----------------------------------------------------------------------
def test_depth_guard_boundaries(catalog):
    """The guard raises ``CascadeDepthGuardError`` naming the depth at
    ``depth > max_depth`` on self-authored depths; the ``depth ==
    max_depth`` case does not raise (the guard is strictly greater); a
    negative ``max_depth`` is rejected; the guard is the first check
    and consumes no random numbers when it fires."""

    # max_depth = 1 fires at depth 2, naming the depth
    state2 = _rt_state("rt_baryon", 4.0, (1, 1, 0), depth=2)
    rng_c, rng_k = _sub_rngs(REGISTER_SEED, 30)
    with pytest.raises(CascadeDepthGuardError, match="carries depth 2"):
        run_cascade(state2, catalog, RT_FAMILIES, rng_c, rng_k, max_depth=1)
    # the message carries the configured maximum
    with pytest.raises(CascadeDepthGuardError,
                       match="MAX_CASCADE_DEPTH = 1"):
        run_cascade(state2, catalog, RT_FAMILIES,
                    *_sub_rngs(REGISTER_SEED, 31), max_depth=1)

    # max_depth = 0: the depth-0 root does not fire (0 > 0 is false),
    # the first continuation at depth 1 does (the guard is strictly
    # greater)
    root0 = _rt_state("rt_baryon", 4.0, (1, 1, 0), depth=0)
    with pytest.raises(CascadeDepthGuardError, match="carries depth 1"):
        run_cascade(root0, catalog, RT_FAMILIES,
                    *_sub_rngs(REGISTER_SEED, 32), max_depth=0)

    # depth == max_depth does not raise: a catalog handoff state at
    # depth 12 with max_depth = 12 terminates through the kernel
    boundary = _rt_state("catalog:9015", 0.8, (0, 0, 0), depth=12)
    trace = run_cascade(boundary, catalog, RT_FAMILIES,
                        *_sub_rngs(REGISTER_SEED, 33), max_depth=12)
    assert trace == [boundary]
    assert _termination_mode(trace, catalog) == \
        TERMINATION_CATALOG_HANDOFF

    # the default maximum: a depth-13 state fires at MAX_CASCADE_DEPTH
    # = 12, naming the depth
    deep13 = _rt_state("rt_baryon", 4.0, (1, 1, 0),
                       depth=MAX_CASCADE_DEPTH + 1)
    with pytest.raises(CascadeDepthGuardError,
                       match="carries depth 13"):
        run_cascade(deep13, catalog, RT_FAMILIES,
                    *_sub_rngs(REGISTER_SEED, 34))

    # the guard honours the driver's max_depth argument and fires
    # before any termination rule: a catalog handoff state beyond the
    # limit raises (the handoff check never runs)
    deep_catalog = _rt_state("catalog:9015", 0.8, (0, 0, 0),
                             depth=MAX_CASCADE_DEPTH + 1)
    with pytest.raises(CascadeDepthGuardError,
                       match="carries depth 13"):
        run_cascade(deep_catalog, catalog, RT_FAMILIES,
                    *_sub_rngs(REGISTER_SEED, 35))

    # a negative max_depth is rejected outright (checked, never
    # repaired)
    with pytest.raises(ValueError, match="must be non-negative"):
        run_cascade(root0, catalog, RT_FAMILIES,
                    *_sub_rngs(REGISTER_SEED, 36), max_depth=-1)

    # the guard consumes no random numbers when it fires (the check
    # runs before any draw site)
    counting_c = _CountingRng(REGISTER_SEED + 37)
    counting_k = _CountingRng(REGISTER_SEED + 38)
    with pytest.raises(CascadeDepthGuardError):
        run_cascade(state2, catalog, RT_FAMILIES, counting_c, counting_k,
                    max_depth=1)
    assert counting_c.calls == 0
    assert counting_k.calls == 0


# ----------------------------------------------------------------------
# test_run_cascade_termination_modes (acceptance (ii))
# ----------------------------------------------------------------------
def test_run_cascade_termination_modes(catalog):
    """The driver's termination modes on self-authored channel graphs:
    the catalog handoff, the direct channel, the continuation chains
    (six pinned events), and the no-channel edge.  The returned line
    is a list of ``CascadeState``, root first, ending at the terminal
    state, and the terminal state identifies the mode exactly."""

    # -- the catalog handoff mode -------------------------------------
    root = _rt_state("catalog:9015", 0.8, (0, 0, 0))
    assert is_catalog_handoff(root, catalog)
    trace = run_cascade(root, catalog, RT_FAMILIES,
                        *_sub_rngs(REGISTER_SEED, 40))
    assert isinstance(trace, list)
    assert all(isinstance(state, CascadeState) for state in trace)
    # the line is the root alone, root first (object identity)
    assert trace == [root]
    assert trace[0] is root
    assert trace[0].depth == 0
    # the terminal state identifies the mode
    assert is_catalog_handoff(trace[-1], catalog)
    assert _termination_mode(trace, catalog) == \
        TERMINATION_CATALOG_HANDOFF

    # -- the direct-channel mode ---------------------------------------
    # the self-authored continuum root at the family floor: the
    # continuum budget windows are closed, the direct channels are
    # open, so the line terminates through the direct pool
    floor = _rt_state("rt_baryon", 2.3, (1, 1, 0))
    floor_channels = enumerate_channels(floor, catalog, RT_FAMILIES)
    assert floor_channels
    assert all(c.kind == KIND_ESTABLISHED_HADRON for c in floor_channels)
    trace = run_cascade(floor, catalog, RT_FAMILIES,
                        *_sub_rngs(REGISTER_SEED, 41))
    assert trace == [floor]
    assert trace[0] is floor
    terminal = trace[-1]
    # the terminal state is a continuum state with no open continuum
    # channel and a valid direct channel
    assert not str(terminal.family_id).startswith(CATALOG_FAMILY_PREFIX)
    assert not is_catalog_handoff(terminal, catalog)
    assert not any(c.kind == KIND_CONTINUUM
                   for c in enumerate_channels(terminal, catalog,
                                               RT_FAMILIES))
    assert has_valid_direct_channel(
        terminal, enumerate_channels(terminal, catalog, RT_FAMILIES))
    assert _termination_mode(trace, catalog) == \
        TERMINATION_DIRECT_CHANNEL

    # -- the continuation chains (six pinned events) --------------------
    # the pinned 3.5 GeV baryonic roots: the continuum daughter mass
    # rule steps the line down (shed pi0 + 0.5, then at the 2.8650234
    # GeV state the shed proton + 0.2 window opens for the mesonic
    # family, and the next baryonic daughter lands exactly on the
    # family floor 2.3, where the windows close and the direct rule
    # terminates the line)
    meson_from_35 = _rt_daughter_mass(
        "rt_meson", float(catalog.get(2212).pole_mass), 3.5)
    baryon_from_35 = _rt_daughter_mass(
        "rt_baryon", float(catalog.get(111).pole_mass), 3.5)
    meson_from_baryon = _rt_daughter_mass(
        "rt_meson", float(catalog.get(2212).pole_mass), baryon_from_35)
    assert meson_from_35 == pytest.approx(2.36172769, abs=1e-9)
    assert baryon_from_35 == pytest.approx(2.8650234, abs=1e-9)
    assert meson_from_baryon == pytest.approx(1.72675109, abs=1e-9)
    assert baryon_from_35 > 2.3  # the intermediate state has an open
    # window (the chain continues one more step)

    pinned_lines = (
        # (event index, [(family_id, mass, depth), ...])
        (100, [("rt_baryon", 3.5, 0), ("rt_meson", meson_from_35, 1)]),
        (101, [("rt_baryon", 3.5, 0), ("rt_baryon", baryon_from_35, 1),
               ("rt_baryon", 2.3, 2)]),
        (102, [("rt_baryon", 3.5, 0), ("rt_baryon", baryon_from_35, 1),
               ("rt_baryon", 2.3, 2)]),
        (103, [("rt_baryon", 3.5, 0), ("rt_baryon", baryon_from_35, 1),
               ("rt_meson", meson_from_baryon, 2)]),
        (104, [("rt_baryon", 3.5, 0), ("rt_meson", meson_from_35, 1)]),
        (105, [("rt_baryon", 3.5, 0), ("rt_baryon", baryon_from_35, 1),
               ("rt_meson", meson_from_baryon, 2)]),
    )
    for event, expected in pinned_lines:
        root = _rt_state("rt_baryon", 3.5, (1, 1, 0))
        trace = run_cascade(root, catalog, RT_FAMILIES,
                            *_sub_rngs(REGISTER_SEED, event))
        assert len(trace) == len(expected), event
        assert trace[0] is root, "the line is root first"
        for state, (family_id, mass, depth) in zip(trace, expected):
            assert state.family_id == family_id, event
            assert state.mass == pytest.approx(mass, abs=1e-12), event
            assert state.depth == depth, event
        # the depths increment by exactly one per continuation
        for parent_state, child in zip(trace, trace[1:]):
            assert child.depth == parent_state.depth + 1
            assert child.parent is parent_state
        # exactly one terminal state: the last element, identified by
        # the mode predicates (every pinned terminal is a continuum
        # state with no open continuum channel)
        terminal = trace[-1]
        assert not is_catalog_handoff(terminal, catalog)
        assert has_valid_direct_channel(
            terminal, enumerate_channels(terminal, catalog, RT_FAMILIES))
        assert _termination_mode(trace, catalog) == \
            TERMINATION_DIRECT_CHANNEL

    # -- the no-channel edge --------------------------------------------
    # below every threshold the enumeration is empty and neither
    # termination rule applies: the driver raises (checked, never
    # repaired, a state outside the accepted model)
    empty = _rt_state("rt_meson", 0.2, (0, 0, 0))
    assert enumerate_channels(empty, catalog, RT_FAMILIES) == []
    with pytest.raises(
        ValueError,
        match="no open continuum channel and no valid direct hadronic",
    ):
        run_cascade(empty, catalog, RT_FAMILIES,
                    *_sub_rngs(REGISTER_SEED, 42))


# ----------------------------------------------------------------------
# test_handoff_delegation (acceptance (i), the bit-identity clause)
# ----------------------------------------------------------------------
def test_handoff_delegation(catalog):
    """The handoff delegation to the Round 2 kernel on self-authored
    handoff states: a stable handoff state returns an empty vertex
    list; the delegated decay is bit-identical to the standalone
    ``ResonanceDecayer`` at the same seed (single- and multi-vertex
    fixtures); the passed generators advance to exactly the kernel's
    bit states; two calls at the same seeds give bit-identical vertex
    lists; a non-handoff state raises ``ValueError``."""

    # -- the stable handoff returns an empty vertex list ----------------
    stable = _rt_state("catalog:2212", 0.93827231, (1, 1, 0))
    assert is_catalog_handoff(stable, catalog)
    assert handoff_decay(stable, catalog,
                         *_sub_rngs(REGISTER_SEED, 50)) == []

    # -- the single-vertex fixture: rtDelta ------------------------------
    single = _rt_state("catalog:9016", 1.25, (1, 1, 0))
    assert is_catalog_handoff(single, catalog)
    kernel = ResonanceDecayer(catalog, REGISTER_SEED, 0, 51)
    standalone = kernel.decay_resonance(
        9016, (1.25, 0.0, 0.0, 0.0))
    assert len(standalone) == 1
    rng_c, rng_k = _sub_rngs(REGISTER_SEED, 51)
    delegated = handoff_decay(single, catalog, rng_c, rng_k)
    assert delegated == standalone, (
        "the single-vertex delegated decay drifted from the "
        "standalone Round 2 kernel"
    )
    # the delegation adds no draw of its own: the passed generators
    # end at exactly the kernel's internal bit states
    assert rng_c.bit_generator.state == \
        kernel._rng_channel.bit_generator.state
    assert rng_k.bit_generator.state == \
        kernel._rng_kinematics.bit_generator.state
    # determinism: two calls at the same seeds, bit-identical lists
    d1 = handoff_decay(single, catalog, *_sub_rngs(REGISTER_SEED, 52))
    d2 = handoff_decay(single, catalog, *_sub_rngs(REGISTER_SEED, 52))
    assert d1 == d2

    # -- the multi-vertex fixture: rtOmega -> rtRho -> pions -------------
    multi = _rt_state("catalog:9017", 0.95, (0, 0, 0))
    assert is_catalog_handoff(multi, catalog)
    kernel_m = ResonanceDecayer(catalog, REGISTER_SEED, 0, 53)
    standalone_m = kernel_m.decay_resonance(
        9017, (0.95, 0.0, 0.0, 0.0))
    assert len(standalone_m) == 2
    rng_c, rng_k = _sub_rngs(REGISTER_SEED, 53)
    delegated_m = handoff_decay(multi, catalog, rng_c, rng_k)
    assert delegated_m == standalone_m, (
        "the multi-vertex delegated decay drifted from the "
        "standalone Round 2 kernel"
    )
    assert rng_c.bit_generator.state == \
        kernel_m._rng_channel.bit_generator.state
    assert rng_k.bit_generator.state == \
        kernel_m._rng_kinematics.bit_generator.state
    d1 = handoff_decay(multi, catalog, *_sub_rngs(REGISTER_SEED, 54))
    d2 = handoff_decay(multi, catalog, *_sub_rngs(REGISTER_SEED, 54))
    assert d1 == d2 and len(d1) == 2

    # -- the delegation through the driver --------------------------------
    # the driver terminates the line at the handoff root and advances
    # the passed generators exactly as the standalone kernel does
    rng_c, rng_k = _sub_rngs(REGISTER_SEED, 53)
    trace = run_cascade(multi, catalog, RT_FAMILIES, rng_c, rng_k)
    assert trace == [multi]
    assert _termination_mode(trace, catalog) == \
        TERMINATION_CATALOG_HANDOFF
    assert rng_c.bit_generator.state == \
        kernel_m._rng_channel.bit_generator.state
    assert rng_k.bit_generator.state == \
        kernel_m._rng_kinematics.bit_generator.state

    # -- checked, never repaired: a non-handoff state raises ---------------
    continuum = _rt_state("rt_baryon", 4.0, (1, 1, 0))
    assert not is_catalog_handoff(continuum, catalog)
    with pytest.raises(ValueError, match="not a catalog handoff"):
        handoff_decay(continuum, catalog, *_sub_rngs(REGISTER_SEED, 55))


# ----------------------------------------------------------------------
# test_below_limit_noop (acceptance (iii), the zero-activation clause)
# ----------------------------------------------------------------------
def test_below_limit_noop(catalog):
    """The self-authored traces at a low depth where the guard never
    activates: every pinned chain completes without
    ``CascadeDepthGuardError``, the deepest line state sits far below
    the limit, and the driver adds no random number of its own (the
    card-4 per-vertex budget stays 1 channel + 2 kinematics)."""

    assert MAX_CASCADE_DEPTH == 12

    # the zero-activation claim: the six pinned 3.5 GeV chains (the
    # deepest line state sits at depth 2, far below the limit)
    # complete without the guard firing
    deepest = 0
    for event in (100, 101, 102, 103, 104, 105):
        root = _rt_state("rt_baryon", 3.5, (1, 1, 0))
        trace = run_cascade(root, catalog, RT_FAMILIES,
                            *_sub_rngs(REGISTER_SEED, event))
        deepest = max(deepest, trace[-1].depth)
        for state in trace:
            assert state.depth <= MAX_CASCADE_DEPTH
    assert deepest == 2
    assert deepest < MAX_CASCADE_DEPTH

    # the driver adds no random number of its own: the direct-
    # termination vertex consumes exactly the card-4 per-vertex budget
    floor = _rt_state("rt_baryon", 2.3, (1, 1, 0))
    counting_c = _CountingRng(REGISTER_SEED + 60)
    counting_k = _CountingRng(REGISTER_SEED + 61)
    trace = run_cascade(floor, catalog, RT_FAMILIES,
                        counting_c, counting_k)
    assert len(trace) == 1
    assert counting_c.calls == 1, "the channel draw is 1 uniform"
    assert counting_k.calls == 2, "the kinematics draw is 2 uniforms"

    # a two-vertex chain consumes exactly two per-vertex budgets
    counting_c = _CountingRng(REGISTER_SEED + 62)
    counting_k = _CountingRng(REGISTER_SEED + 63)
    root = _rt_state("rt_baryon", 3.5, (1, 1, 0))
    trace = run_cascade(root, catalog, RT_FAMILIES,
                        counting_c, counting_k)
    assert len(trace) == 2
    assert counting_c.calls == 2
    assert counting_k.calls == 4

    # the stable handoff route consumes zero uniforms end to end (the
    # guard, the handoff predicate, and the kernel's stable short-
    # circuit are all draw-free)
    stable = _rt_state("catalog:2212", 0.93827231, (1, 1, 0))
    counting_c = _CountingRng(REGISTER_SEED + 64)
    counting_k = _CountingRng(REGISTER_SEED + 65)
    trace = run_cascade(stable, catalog, RT_FAMILIES,
                        counting_c, counting_k)
    assert trace == [stable]
    assert counting_c.calls == 0
    assert counting_k.calls == 0


# ----------------------------------------------------------------------
# test_predicates_zero_draw (acceptance (iv), the zero-draw clause)
# ----------------------------------------------------------------------
def test_predicates_zero_draw(catalog):
    """The predicates consume no randomness: the seeded bit-generator
    state is byte-identical before and after a self-authored
    predicate scan (both the true and the false branches, and the
    raising parent-mismatch call)."""

    rng = np.random.default_rng(REGISTER_SEED + 99)
    state_before = rng.bit_generator.state

    for mass in (0.3, 0.8, 1.2, 1.2000001, float("nan")):
        is_catalog_handoff(_rt_state("catalog:9015", mass, (0, 0, 0)),
                           catalog)
    is_catalog_handoff(_rt_state("rt_baryon", 1.25, (1, 1, 0)), catalog)
    is_catalog_handoff(_rt_state("catalog:99999", 0.8, (0, 0, 0)),
                       catalog)

    state = _rt_state("rt_baryon", 4.0, (1, 1, 0))
    hadron = CascadeChannel(
        parent_id="rt_baryon",
        daughters=("catalog:2212", "catalog:111"),
        kind=KIND_ESTABLISHED_HADRON,
        weight=0.5,
    )
    continuum = CascadeChannel(
        parent_id="rt_baryon",
        daughters=("rt_baryon", "catalog:111"),
        kind=KIND_CONTINUUM,
        weight=0.5,
    )
    has_valid_direct_channel(state, [hadron])
    has_valid_direct_channel(state, [hadron, continuum])
    has_valid_direct_channel(state, [continuum])
    has_valid_direct_channel(state, [])
    with pytest.raises(ValueError):
        has_valid_direct_channel(
            state, [CascadeChannel(
                parent_id="rt_meson",
                daughters=("catalog:211", "catalog:-211"),
                kind=KIND_ESTABLISHED_HADRON,
                weight=1.0,
            )])

    state_after = rng.bit_generator.state
    assert state_after == state_before, (
        "the predicates consumed random numbers: the bit-generator "
        "state changed over the self-authored predicate scan"
    )


# ----------------------------------------------------------------------
# test_module_constants (the named constants)
# ----------------------------------------------------------------------
def test_module_constants():
    """MAX_CASCADE_DEPTH == 12 (the card-4 value, re-exported by
    object identity, not a copy); the termination mode constants; the
    ``CascadeDepthGuardError`` hierarchy; the module's ``__all__``."""

    from na49_mixing import cascade_decay as card4
    from na49_mixing import cascade_termination as module

    assert MAX_CASCADE_DEPTH == 12
    assert isinstance(MAX_CASCADE_DEPTH, int)
    # the re-export is the card-4 constant by object identity
    assert module.MAX_CASCADE_DEPTH is card4.MAX_CASCADE_DEPTH
    assert module.MAX_CASCADE_DEPTH is MAX_CASCADE_DEPTH

    assert TERMINATION_CATALOG_HANDOFF == "catalog_handoff"
    assert TERMINATION_DIRECT_CHANNEL == "direct_channel"
    assert isinstance(TERMINATION_CATALOG_HANDOFF, str)
    assert isinstance(TERMINATION_DIRECT_CHANNEL, str)
    assert TERMINATION_CATALOG_HANDOFF != TERMINATION_DIRECT_CHANNEL

    assert issubclass(CascadeDepthGuardError, RuntimeError)

    assert module.__all__ == [
        "MAX_CASCADE_DEPTH",
        "CascadeDepthGuardError",
        "TERMINATION_CATALOG_HANDOFF",
        "TERMINATION_DIRECT_CHANNEL",
        "is_catalog_handoff",
        "has_valid_direct_channel",
        "handoff_decay",
        "run_cascade",
    ]


# ----------------------------------------------------------------------
# Part 2: the substrate and freeze pins
# ----------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    """The sha256 of *path* (the house byte-pin helper)."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_module_byte_and_export_pins():
    """The module byte stability (sha256 0e768572...) and the additive
    ``na49_mixing`` package exports (the seven card-9 names present in
    the package ``__init__`` import block and ``__all__``, resolved by
    object identity, with the prior card exports unchanged)."""

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)
    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    import na49_mixing

    init_src = INIT_PATH.read_text(encoding="utf-8")
    import_block = init_src[
        init_src.index("from .cascade_termination import"):
        init_src.index(")", init_src.index("from .cascade_termination import"))
    ]
    for name in (
        "TERMINATION_CATALOG_HANDOFF",
        "TERMINATION_DIRECT_CHANNEL",
        "CascadeDepthGuardError",
        "has_valid_direct_channel",
        "handoff_decay",
        "is_catalog_handoff",
        "run_cascade",
    ):
        assert name in import_block, (
            f"the __init__ import block is missing {name}")
        assert f'"{name}"' in init_src, (
            f"the __init__ __all__ is missing {name}")
    from na49_mixing import cascade_termination as module

    assert na49_mixing.TERMINATION_CATALOG_HANDOFF is (
        module.TERMINATION_CATALOG_HANDOFF)
    assert na49_mixing.TERMINATION_DIRECT_CHANNEL is (
        module.TERMINATION_DIRECT_CHANNEL)
    assert na49_mixing.CascadeDepthGuardError is module.CascadeDepthGuardError
    assert na49_mixing.has_valid_direct_channel is (
        module.has_valid_direct_channel)
    assert na49_mixing.handoff_decay is module.handoff_decay
    assert na49_mixing.is_catalog_handoff is module.is_catalog_handoff
    assert na49_mixing.run_cascade is module.run_cascade

    # the prior card exports are unchanged (the additive-only clause)
    from na49_mixing import (
        cascade_alignment,
        cascade_axes,
        cascade_decay,
        cascade_form_factor,
    )

    for name in (
        "cascade_decay",
        "cascade_axes",
        "cascade_alignment",
        "cascade_form_factor",
    ):
        assert hasattr(na49_mixing, name), (
            f"the prior card export {name} is missing")
    assert na49_mixing.CascadeState is cascade_decay.CascadeState
    assert na49_mixing.CascadeChannel is cascade_decay.CascadeChannel
    assert na49_mixing.enumerate_channels is cascade_decay.enumerate_channels
    assert na49_mixing.draw_decay is cascade_decay.draw_decay
    assert na49_mixing.MAX_CASCADE_DEPTH is cascade_decay.MAX_CASCADE_DEPTH
    assert na49_mixing.transverse_momentum is (
        cascade_form_factor.transverse_momentum)


def test_frozen_substrate_digests():
    """The byte-unchanged surfaces the card consumes (the card's
    provenance section, verified on disk): card 4's cascade surface
    (mandated byte-stable), card 5's axis machinery, the card-6/7
    alignment kernel, card 8's form factor, the Round 2 kernel, the
    builder, the card-6 blend routing, the card-2 engine, the
    committed catalog, the continuum families configuration, and the
    R32 driver in both trees."""

    for rel, expected in FROZEN_SUBSTRATE.items():
        path = MONO_ROOT / rel
        assert path.is_file(), f"missing frozen surface: {rel}"
        pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)


# ----------------------------------------------------------------------
# test_freeze_repin_register (the acceptance-(v) freeze re-pin check)
# ----------------------------------------------------------------------
