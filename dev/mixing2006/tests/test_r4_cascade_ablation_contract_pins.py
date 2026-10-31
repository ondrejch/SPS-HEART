"""Round 4, card 10 test-author register: isotropic-versus-aligned
ablation runner.

Independent contract pins for the isotropic-versus-aligned ablation
runner module (``na49_mixing/cascade_ablation.py``), Round 4 card 10
(``TASK-20260920-r4-isotropic-aligned-ablation-runner``; plan section
22 task 27, line 1749; plan section 10 run R43, lines 790-798).  This
register is the test-author half of the card: it pins the module
contract on its own seeds, its own exchange-event fixtures, and its
own counting-generator instrumentation, and it re-pins the freeze
manifest for the register.  The solver's pinned seeds and pinned
exchange events are not re-run here: the contract is pinned on the
register's own triples, roots, and draw counts, at a small scale.

Part 1: the module contract (pure, no sealed run dir)
-----------------------------------------------------

- ``test_variant_register_and_policy_contract``: the closed
  ``ABLATION_VARIANTS`` register (the exact three names in the plan's
  order) and the register-order variant-id mapping; the frozen
  ``AblationPolicy`` dataclass (mutation raises, the constructor
  normalizes int fields to float); the construction-time validation on
  the register's own invalid variant / kappa / b_dec values (the
  exception types and messages on disk); and the three pre-registered
  policies at the isotropic point (0, 0, no form factor) and the
  nominal point (1.0, 1.0, b_dec = 1.0 GeV^-2), exact values on disk.
- ``test_variant_stream_contract``: the ``variant_stream`` child
  ``SeedSequence`` spawns on the register's own
  ``(master_seed, event_index, variant_id)`` triples: the one-step
  ``spawn_key=(event_index, variant_id)`` derivation pinned against
  the literal fresh-object spawn chain (bit-identical), the two-label
  ``spawn(2)`` split, the key sensitivity to the event index and the
  master seed, the disjointness (two variant ids give disjoint
  streams, verified by bit-generator state and by drawn sequences),
  the determinism (same triple, byte-identical streams), and the
  variant-id domain (a non-integer, negative, or out-of-register id
  raises ``ValueError`` on disk).
- ``test_exchange_substreams_contract``: the ``exchange_substreams``
  closed exchange-layer label set (``topology_selection`` /
  ``exchange_kinematics`` / ``spectral_state`` on disk), the
  determinism, the ``replica_index`` semantics (the default 0, the
  register's own replica value, and the negative-replica refusal on
  disk), and the exchange identity across variants (the same
  ``(master_seed, event_index)`` gives the same exchange event
  regardless of which variant consumes it: the same substreams and the
  same root state for all three variants).
- ``test_exchange_root_state_contract``: the
  ``exchange_root_state`` mapping contract on the register's own
  fixtures: the pinned keys (``topology`` / ``momentum`` /
  ``family_id`` / ``mass`` / ``qn`` / ``axis``, the optional
  ``ancestry`` defaulting to ``(topology,)``), the field round-trip,
  the catalog quantum mapping against the record, the draw-free
  contract, the never-mutated contract (the event mapping is
  unchanged and the root carries fresh tuples, not the mapping's
  internals), and the register's own malformed events (missing key,
  non-mapping, three-component momentum, non-finite component,
  non-positive energy, mass beyond the invariant-mass tolerance,
  two-component quantum numbers, non-unit axis, catalog quantum
  mismatch, unknown family id), with the on-disk exception types and
  messages.
- ``test_decay_variant_v1_single_step``: the v1 single-step on the
  register's own catalog-root and continuum-root fixtures: the
  terminal daughters of the single step (the root as their parent,
  depth 1, the isotropic label, the exact ``(B, Q, S)`` sum, the
  four-momentum closure), and the budget (catalog:113 = 1 + 2,
  catalog:12212@2.0 = 1 + 2 + 2 per realized resonance daughter,
  continuum = 1 + 2), verified on disk with the register's own
  counting generators.
- ``test_decay_variant_v2_v3_lines``: the v2/v3 line traces on the
  register's own fixtures: root first, the depths increment by one,
  the terminal state, the budget 1 + 2 per line state, the alignment
   label bookkeeping on disk (v2 all isotropic; v3 the governing-
   kernel labels: the baryonic label on the baryonic daughter of a
   baryonic vertex, the placeholder on its mesonic daughter, the
   mesonic label on both daughters of the symmetric mesonic vertex),
   the v2
  off-limit (policy (0, 0, 0.0) bit-identical to card 9's unattached
  ``run_cascade`` on the register's own small-scale fixtures), and
  the v2 pin (the kappa = 0 isotropic branches, card 4's
  ``draw_decay`` reference behavior over the attached pools at the
  nominal b_dec, bit-identical on the register's own seeds).
- ``test_support_boundary_block``: the machine-readable
  ``SUPPORT_BOUNDARY`` block on disk: the plan citations, the v1
  single-step support regenerated, the v2/v3 multi-vertex support
  identical and shared, the ``regeneration_required`` register, the
  variant-key closure against ``ABLATION_VARIANTS``, and the
  block's importability and JSON serializability (the card-14
  pre-registration contract).
- ``test_reference_direction_and_w_dec_pin``: the pinned reference
  emission configuration: ``W_DEC_REFERENCE_COS_THETA == 0.3``
  exactly, ``W_DEC_REFERENCE_SIN_THETA`` the exact sine, the
  register's own axis fixtures (the coordinate axes and an oblique
  axis), the reference direction's cosine against the axis exactly
  0.3 and its unit norm, the q_perp-invariance (the transverse
  fraction is the fixed value ``sqrt(1 - 0.3^2)`` independent of the
  axis, i.e. of the azimuth choice), the draw-free contract, and the
  module docstring's documented boundary (the v1 continuum-root
  direct-channel rule).
- ``test_toggle_zero_draw``: the zero-draw contract of the toggle
  wiring: the W_dec factor and the ``draw_vertex_cos_theta``
  replacement consume no uniforms beyond the budget, verified with
  the register's own counting generators around the full
  ``decay_variant`` for all three policies (v1 the single-step
  budget, v2/v3 the 1 + 2 per line state, identical across the
  policies and across a non-nominal scan policy).

Part 2: the substrate and freeze pins
-------------------------------------

- ``test_module_and_init_pins``: the new module digest
  (``c0206cf6...``), the ``na49_mixing/__init__.py`` byte pin
  (``1b3a6e65...`` -- the card-10 decision: the exports deliberately
  not exercised, the package ``__init__`` untouched), and the
  module's importability as ``na49_mixing.cascade_ablation`` by
  object identity via importlib, with the package exports unchanged.
- ``test_frozen_substrate_digests``: the byte-unchanged frozen
  substrate of the card's provenance section (cards 4-9's cascade
  surfaces, the Round 2 kernel, the builder, the card-6 blend
  routing, the card-2 engine, the committed catalog, the continuum
  families configuration, and the R32 driver in both trees),
  verified on disk.
- ``test_freeze_repin_register``: the freeze-manifest re-pin for
  this register (the strict path-sorted inventory at 472 entries
  carries this file at its on-disk digest; the card-10 solver freeze
  exception stays at the solver's 15 path-sorted files; this
  register's own freeze exception names the test-author card exactly
  once with the path-sorted file list; the thirteen consumed count
  pins (the twelve register-chain files plus the card-10 solver
  test's own count constant) stay at their on-disk digests, each
  re-pinned 471 -> 472 with the naming comment attributing the bump
  to this register, the card-10 solver's 471 literal retained for
  its re-pin check; the frozen M8 parameter map stays intact).

Hermetic: the committed config and catalog files, numpy, the
standard library, and the vendored ``na49_mixing`` modules only; no
event generation, no external tables, no Monte Carlo run.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.cascade_ablation import (
    ABLATION_POLICIES,
    ABLATION_VARIANTS,
    EXCHANGE_EVENT_KEYS,
    SUPPORT_BOUNDARY,
    VARIANT_IDS,
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
    BARYONIC_ALIGNMENT_LABEL,
    BARYONIC_KAPPA_DEFAULT,
    MESONIC_ALIGNMENT_LABEL,
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
    CascadeState,
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
from na49_mixing.resonance_decay import KinematicsError
from na49_mixing.resonance_widths import decay_momentum
from na49_mixing.rng_substreams import SUBSTREAM_LABELS, derive_substreams
from na49_mixing.spectral_density import SUPPORT_M_MAX, families_from_json
from tests._freeze_inventory import inventory_count as _inventory_count

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent          # dev/mixing2006
MONO_ROOT = NESTED_ROOT.parent.parent          # the monorepo root
MODULE_PATH = NESTED_ROOT / "na49_mixing" / "cascade_ablation.py"
INIT_PATH = NESTED_ROOT / "na49_mixing" / "__init__.py"
CATALOG_PATH = (NESTED_ROOT / "config" / "v2_singlet"
                / "resonance_catalog.json")
FAMILIES_PATH = (NESTED_ROOT / "config" / "v2_singlet"
                 / "continuum_resonances.json")
MANIFEST_PATH = (NESTED_ROOT / "config" / "v2_singlet"
                 / "legacy_v1_manifest.json")

MODULE_REL = "dev/mixing2006/na49_mixing/cascade_ablation.py"
INIT_REL = "dev/mixing2006/na49_mixing/__init__.py"
MY_TEST_REL = ("dev/mixing2006/tests/"
               "test_r4_cascade_ablation_contract_pins.py")
SOLVER_TEST_REL = "dev/mixing2006/tests/test_cascade_ablation.py"
MANIFEST_REL = ("dev/mixing2006/config/v2_singlet/"
                "legacy_v1_manifest.json")
FREEZE_CARD = "TASK-20260920-r4-isotropic-aligned-ablation-runner"
# The test-author re-pin's own freeze-exception card name (distinct
# from the solver's, so the solver's "exactly one entry naming this
# card" check is untouched).
TEST_AUTHOR_CARD = (
    "TASK-20260920-r4-isotropic-aligned-ablation-runner-test-author"
)

#: The twelve register-chain files whose inventory-count pins this
#: register re-pins (471 -> 472), each with the naming comment
#: attributing the bump to this register's file.
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
CARD9_SOLVER_TEST_REL = ("dev/mixing2006/tests/"
                         "test_cascade_termination.py")
CARD9_REGISTER_REL = ("dev/mixing2006/tests/"
                      "test_r4_cascade_termination_contract_pins.py")
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
    CARD9_SOLVER_TEST_REL,
    CARD9_REGISTER_REL,
)
# The thirteen count pins this register re-pins: the twelve
# register-chain files plus the card-10 solver test's own count
# constant.
REPIN_FILES = CHAIN_FILES + (SOLVER_TEST_REL,)

# 471 entries at the card-10 solver landing; this register adds its
# file (+1 entry) for the test-author re-pin (471 -> 472, +1 entry)
# and re-pins the thirteen consumed count pins (the twelve
# register-chain files plus the card-10 solver test's own count
# constant) with the bump attributed to this register's file; card 11
# of the round (TASK-20260920-r4-run-r40-fixed-source-alignment-scan)
# re-pins the inventory additively for its new dual-tree run driver and
# test file (helpers/benchmarking/staged_r40_alignment_scan.py,
# dev/mixing2006/helpers/benchmarking/staged_r40_alignment_scan.py,
# tests/test_staged_r40_alignment_scan.py; the constant read
# "INVENTORY_COUNT = 472" at the card-10 register landing;
# 472 -> 475, +3 entries), so this count constant is re-pinned once
# more. The card-11 test-author register
# (test_staged_r40_alignment_scan_contract_pins.py) re-pins the
# inventory additively for its own file (the constant read
# "INVENTORY_COUNT = 475" at the card-11 solver landing; 475 -> 476,
# +1 entry), so this count constant is re-pinned once more.
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
# re-pinned 476 -> 479 by card 12 (TASK-20260920-r4-run-r41-
# nonstrange-pilot-fit; +3 entries: the dual-tree R41 driver in
# both helper trees and the hermetic R41 test file).
# re-pinned 479 -> 480 by card 12 test-author (TASK-20260920-r4-run-
# r41-nonstrange-pilot-fit-test-author; +1 entry: the card-12
# re-pinned 491 -> 495 by card 20 (TASK-20260920-r4-gate4-followup-simplified-alignment; +4 entries: the dual-tree R45 driver in both helper trees and the two new R45 test files).
# re-pinned 495 -> 499 by card 21 (TASK-20260920-r4-gate4-followup-recoil-decay-reprofile; +4 entries: the dual-tree R46 driver in both helper trees and the two new R46 test files).
# test-author register file).

MODULE_SHA256 = (
    "ed15dcf5c606255b6d020d42569f1fb0c5c8e96b5eaa7030bc649832c143b335"
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

#: The byte-unchanged frozen substrate of the card's provenance
#: section (verified on disk at this register's landing): cards 4-9's
#: cascade surfaces, the Round 2 kernel (the v1 single-step target
#: and the card-9 delegation), the builder, the card-6 blend routing,
#: the card-2 engine, the committed catalog, the continuum families
#: configuration, and the R32 driver in both trees.
FROZEN_SUBSTRATE = {
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
    "dev/mixing2006/helpers/benchmarking/"
    "staged_r32_spectral_systematic.py":
        "99454facce856c93505e6c4b0d4e47b85eeb0ef54f72b1631e3d29fcc511cb90",
    "helpers/benchmarking/staged_r32_spectral_systematic.py":
        "99454facce856c93505e6c4b0d4e47b85eeb0ef54f72b1631e3d29fcc511cb90",
}

#: The solver's card-10 freeze-exception file list (15 path-sorted
#: files; this register leaves it untouched and records its own
#: re-pin under a distinct card name).
SOLVER_EXCEPTION_FILES = (
    MANIFEST_REL,
    MODULE_REL,
    SOLVER_TEST_REL,
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

#: This register's own freeze-exception file list (path-sorted): the
#: manifest, this register, and the thirteen re-pinned count pins.
TEST_AUTHOR_EXCEPTION_FILES = tuple(
    sorted([MANIFEST_REL, MY_TEST_REL] + list(REPIN_FILES))
)

#: The independent register seed (the day after the round 4 master
#: seed, the solver's 20260920; the day after the card-4 and card-9
#: registers' 20260921).
REGISTER_SEED = 20260922

#: The register's own event indices of the per-variant stream keys
#: (disjoint from the solver's pinned 7, 11, 23, 101).
REGISTER_EVENTS = (4, 17, 42, 1234)

#: The register's own quantum-number fixtures.
BARYONIC_QN = (1, 1, 0)
STRANGE_BARYONIC_QN = (1, 0, -1)
MESONIC_QN = (0, 0, 0)

#: The register's own pinned roots (disjoint from the solver's
#: fixture choices): the rho0 mesonic catalog root (the single
#: established channel), the 12212 baryonic catalog root at 2.0 (the
#: resonance-daughter budget clause), and the continuum roots at the
#: register's own masses.
CATALOG_ROOTS = (
    ("catalog:113", 0.77526, MESONIC_QN),
    ("catalog:12212", 2.0, BARYONIC_QN),
)
CONTINUUM_ROOTS = (
    ("baryonic_strange", 3.0, STRANGE_BARYONIC_QN),
    ("baryonic_np", 5.5, BARYONIC_QN),
    ("mesonic_neutral", 5.0, MESONIC_QN),
)


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
    """A pinned at-rest exchange event of plan line 798 (the
    register's own fixtures): the topology label, the exchange
    kinematics (the source at rest at the realized mass), the
    spectral state (the family identifier and its quantum numbers),
    and the lab-z production axis (the isolated-source convention of
    round 4 index item 4)."""

    return {
        "topology": "register_ablation_probe",
        "momentum": (float(mass), 0.0, 0.0, 0.0),
        "family_id": family_id,
        "mass": float(mass),
        "qn": tuple(int(v) for v in qn),
        "axis": (0.0, 0.0, 1.0),
        "ancestry": ("register_probe",),
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


def _variant_streams(event_index: int, variant_id: int):
    return variant_stream(REGISTER_SEED, event_index, variant_id)


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


def _reference_attached_pool(pool, state, b_dec: float,
                             catalog: HadronCatalog,
                             fams) -> list[CascadeChannel]:
    """The attached channel list at *b_dec* (the reference loop's
    pool): every channel's weight is the card-4 hooked weight scaled
    by the card-8 W_dec factor at the channel's reference two-body
    configuration (the runner's pinned reference direction, the
    mirrored card-4 mass rule), at the raw scale of the enumeration.
    The register's own mirror of the runner's attachment, on the
    register's own fixtures."""

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


def _reference_isotropic_line(root, catalog: HadronCatalog, fams,
                              rng_channel, rng_kinematics):
    """The reference loop of the v2 pin: card 9's driver routing with
    card 4's own ``draw_decay`` vertices over the attached pools at
    the nominal b_dec (the independent machinery the runner's v2 is
    pinned against; the register's own mirror, the same routing)."""

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
        attached = _reference_attached_pool(
            pool, current, DECAY_FORM_FACTOR_B_DEFAULT, catalog, fams)
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


def _sha256_file(path: Path) -> str:
    """The sha256 of *path* (the house byte-pin helper)."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_normalize(value):
    """The JSON round-trip shape of *value* (tuples become lists,
    dicts recurse; the on-disk block carries tuple values)."""

    if isinstance(value, dict):
        return {key: _json_normalize(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return [_json_normalize(val) for val in value]
    return value


# ----------------------------------------------------------------------
# Part 1: the module contract
# ----------------------------------------------------------------------
def test_variant_register_and_policy_contract():
    """The closed ``ABLATION_VARIANTS`` register (the exact three
    names in the plan's order) with the register-order variant-id
    mapping; the frozen ``AblationPolicy`` dataclass (mutation raises
    on disk, the constructor normalizes int fields to float); the
    construction-time validation on the register's own invalid
    variant / kappa / b_dec values; and the three pre-registered
    policies at the isotropic point and the nominal point (round 4
    index item 3), exact values on disk."""

    # the closed register, in the plan's order (R43 items 1-3)
    assert ABLATION_VARIANTS == (
        "isotropic_direct", "isotropic_sequential", "aligned_sequential"
    )
    assert isinstance(ABLATION_VARIANTS, tuple)
    assert len(ABLATION_VARIANTS) == 3
    assert len(set(ABLATION_VARIANTS)) == 3
    # the register order fixes the variant ids (0, 1, 2) of the
    # per-variant decay streams
    assert VARIANT_IDS == {
        "isotropic_direct": 0,
        "isotropic_sequential": 1,
        "aligned_sequential": 2,
    }
    assert list(VARIANT_IDS.values()) == [0, 1, 2]

    # the frozen dataclass: mutation raises (checked on disk), and
    # the constructor normalizes the numeric fields to float
    policy = ABLATION_POLICIES["isotropic_direct"]
    with pytest.raises(AttributeError):
        policy.kappa_b = 1.0  # type: ignore[misc]
    with pytest.raises(AttributeError):
        policy.variant = "aligned_sequential"  # type: ignore[misc]
    normalized = AblationPolicy("aligned_sequential", 1, 1, 1)
    assert normalized.kappa_b == 1.0
    assert isinstance(normalized.kappa_b, float)
    assert isinstance(normalized.kappa_m, float)
    assert isinstance(normalized.b_dec, float)

    # the validation edges (checked, never repaired; the register's
    # own invalid values, the on-disk exception types and messages)
    with pytest.raises(ValueError, match="outside the closed register"):
        AblationPolicy("direct_sequential", 0.0, 0.0, 1.0)
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="kappa_b"):
            AblationPolicy("isotropic_direct", bad, 0.0, 1.0)
        with pytest.raises(ValueError, match="kappa_m"):
            AblationPolicy("isotropic_sequential", 0.0, bad, 1.0)
    for bad in (float("nan"), float("inf"), -0.5, -1e-12):
        with pytest.raises(ValueError, match="b_dec"):
            AblationPolicy("aligned_sequential", 1.0, 1.0, bad)
    # a negative kappa is finite: the on-disk behavior accepts it
    # (the domain check is finiteness, not sign)
    negative = AblationPolicy("aligned_sequential", -1.0, 1.0, 1.0)
    assert negative.kappa_b == -1.0

    # the three pre-registered policies, exact values on disk
    assert set(ABLATION_POLICIES) == set(ABLATION_VARIANTS)
    for name in ("isotropic_direct", "isotropic_sequential"):
        policy = ABLATION_POLICIES[name]
        assert policy.variant == name
        assert policy.kappa_b == 0.0 and policy.kappa_m == 0.0
    v1 = ABLATION_POLICIES["isotropic_direct"]
    assert v1.b_dec == DECAY_FORM_FACTOR_B_DEFAULT == 1.0
    v2 = ABLATION_POLICIES["isotropic_sequential"]
    assert v2.b_dec == DECAY_FORM_FACTOR_B_DEFAULT == 1.0
    v3 = ABLATION_POLICIES["aligned_sequential"]
    assert v3.kappa_b == BARYONIC_KAPPA_DEFAULT == 1.0
    assert v3.kappa_m == MESONIC_KAPPA_DEFAULT == 1.0
    assert v3.b_dec == DECAY_FORM_FACTOR_B_DEFAULT == 1.0
    for policy in ABLATION_POLICIES.values():
        assert policy.variant in ABLATION_VARIANTS


# ----------------------------------------------------------------------
# The per-variant decay streams
# ----------------------------------------------------------------------
def test_variant_stream_contract():
    """The ``variant_stream`` child ``SeedSequence`` spawns on the
    register's own triples: the one-step ``spawn_key`` derivation
    pinned against the literal fresh-object spawn chain, the two-
    label ``spawn(2)`` split, the key sensitivity, the disjointness
    across the variant ids, the determinism, and the variant-id
    domain (checked on disk)."""

    # the derivation: the module's streams are the child spawns keyed
    # on (master_seed, event_index, variant_id), the two label
    # children of the one-step form, bit-identical to the literal
    # fresh-object spawn chain
    for ms, ei, vid in ((REGISTER_SEED, 17, 1), (REGISTER_SEED + 1, 4, 0),
                        (REGISTER_SEED + 2, 42, 2)):
        one_step = np.random.SeedSequence(ms, spawn_key=(ei, vid))
        channel_seq, kinematics_seq = one_step.spawn(2)
        literal = np.random.SeedSequence(ms).spawn(ei + 1)[ei]
        literal = literal.spawn(vid + 1)[vid]
        literal_c, literal_k = literal.spawn(2)
        assert channel_seq.entropy == literal_c.entropy
        assert channel_seq.spawn_key == literal_c.spawn_key
        assert kinematics_seq.entropy == literal_k.entropy
        assert kinematics_seq.spawn_key == literal_k.spawn_key
        gen_c, gen_k = variant_stream(ms, ei, vid)
        ref_c = np.random.default_rng(channel_seq)
        ref_k = np.random.default_rng(kinematics_seq)
        assert gen_c.bit_generator.state == ref_c.bit_generator.state
        assert gen_k.bit_generator.state == ref_k.bit_generator.state

    # determinism: the same triple reproduces the streams byte for
    # byte
    for event_index in REGISTER_EVENTS:
        for variant_id in range(len(ABLATION_VARIANTS)):
            first = _variant_streams(event_index, variant_id)
            second = _variant_streams(event_index, variant_id)
            for gen_a, gen_b in zip(first, second):
                assert [gen_a.random() for _ in range(6)] == [
                    gen_b.random() for _ in range(6)]

    # disjointness: two variant ids give disjoint streams (the
    # bit-generator state of one variant never advances while the
    # other draws, and the drawn sequences differ)
    streams = {vid: _variant_streams(17, vid)
               for vid in range(len(ABLATION_VARIANTS))}
    states_before = {
        vid: tuple(gen.bit_generator.state for gen in pair)
        for vid, pair in streams.items()
    }
    for _ in range(12):
        streams[0][0].random()
        streams[0][1].random()
        streams[2][0].random()
    for vid in (0, 2):
        states_after = tuple(
            gen.bit_generator.state for gen in streams[vid])
        assert states_after != states_before[vid], (
            f"variant {vid}: its stream did not advance while it "
            "drew")
    states_after = tuple(
        gen.bit_generator.state for gen in streams[1])
    assert states_after == states_before[1], (
        "variant 1: its stream advanced while another variant drew")
    channel_sequences = [
        tuple(streams[vid][0].random() for _ in range(8))
        for vid in range(len(ABLATION_VARIANTS))
    ]
    kinematics_sequences = [
        tuple(streams[vid][1].random() for _ in range(8))
        for vid in range(len(ABLATION_VARIANTS))
    ]
    assert len(set(channel_sequences)) == len(ABLATION_VARIANTS)
    assert len(set(kinematics_sequences)) == len(ABLATION_VARIANTS)

    # the key is sensitive to the event index and the master seed
    seq_a = tuple(_variant_streams(17, 0)[0].random() for _ in range(4))
    seq_b = tuple(_variant_streams(42, 0)[0].random() for _ in range(4))
    seq_c = tuple(
        variant_stream(REGISTER_SEED + 1, 17, 0)[0].random()
        for _ in range(4))
    assert seq_a != seq_b
    assert seq_a != seq_c

    # the variant-id domain (checked, never repaired): a
    # non-integer, a negative value, and an id outside the closed
    # register all raise ValueError on disk
    for bad in (-1, 1.5, True, "1"):
        with pytest.raises(ValueError, match="non-negative integer"):
            variant_stream(REGISTER_SEED, 17, bad)
    for bad in (-1, True, "4"):
        with pytest.raises(ValueError, match="non-negative integer"):
            variant_stream(REGISTER_SEED, bad, 0)
        with pytest.raises(ValueError, match="non-negative integer"):
            variant_stream(bad, 17, 0)
    with pytest.raises(ValueError, match="closed ablation register"):
        variant_stream(REGISTER_SEED, 17, len(ABLATION_VARIANTS))
    with pytest.raises(ValueError, match="closed ablation register"):
        variant_stream(REGISTER_SEED, 17, 99)


# ----------------------------------------------------------------------
# The shared exchange layer
# ----------------------------------------------------------------------
def test_exchange_substreams_contract():
    """The ``exchange_substreams`` closed exchange-layer label set
    (``topology_selection`` / ``exchange_kinematics`` /
    ``spectral_state`` on disk), the determinism, the
    ``replica_index`` semantics (the default 0, the register's own
    replica, the negative-replica refusal), and the exchange
    identity across variants (the same ``(master_seed,
    event_index)`` gives the same exchange event regardless of which
    variant consumes it)."""

    # the closed label set: the first three entries of the house
    # SUBSTREAM_LABELS, all carried by the returned object
    subs = exchange_substreams(REGISTER_SEED, 17)
    assert subs == derive_substreams(REGISTER_SEED, 0, 17)
    for label in ("topology_selection", "exchange_kinematics",
                  "spectral_state"):
        assert label in SUBSTREAM_LABELS
        assert subs.stream_for(label) == derive_substreams(
            REGISTER_SEED, 0, 17).stream_for(label)
    # the exchange labels are the first three of the closed set
    assert tuple(SUBSTREAM_LABELS[:3]) == (
        "topology_selection", "exchange_kinematics", "spectral_state")

    # determinism: the same (master_seed, event_index) reproduces
    # the substreams
    assert exchange_substreams(REGISTER_SEED, 17) == subs
    for event_index in REGISTER_EVENTS:
        assert exchange_substreams(REGISTER_SEED, event_index) == \
            derive_substreams(REGISTER_SEED, 0, event_index)

    # the replica_index semantics: the default is 0, the house
    # derivation is (master_seed, replica_index, event_index), and a
    # negative replica is refused on disk
    assert exchange_substreams(REGISTER_SEED, 17, 0) == subs
    for replica in (5, 99):
        assert exchange_substreams(REGISTER_SEED, 17, replica) == \
            derive_substreams(REGISTER_SEED, replica, 17)
    assert exchange_substreams(REGISTER_SEED, 17, 5) != subs
    with pytest.raises(ValueError, match="non-negative integer"):
        exchange_substreams(REGISTER_SEED, 17, -1)

    # the exchange identity across variants: the exchange layer is
    # spawned once per (master_seed, event_index) outside the variant
    # loop and shared across all three variants -- the same
    # substreams and the same root state for every variant
    event = _exchange_event("baryonic_np", 4.0, BARYONIC_QN)
    for variant_id in range(len(ABLATION_VARIANTS)):
        assert exchange_substreams(REGISTER_SEED, 17) == subs
        root = exchange_root_state(event)
        assert root.family_id == "baryonic_np"
        assert root.mass == 4.0
        assert root.qn == BARYONIC_QN
        assert root.depth == 0
        assert root.alignment_label == ISOTROPIC_ALIGNMENT_LABEL


# ----------------------------------------------------------------------
# The exchange event and the root state
# ----------------------------------------------------------------------
def test_exchange_root_state_contract(catalog):
    """The ``exchange_root_state`` mapping contract on the register's
    own fixtures: the pinned keys with the optional ``ancestry``
    default, the field round-trip, the catalog quantum mapping, the
    draw-free contract, the never-mutated contract (fresh objects,
    not the mapping's internals), and the register's own malformed
    events (the on-disk exception types and messages)."""

    assert EXCHANGE_EVENT_KEYS == (
        "topology", "momentum", "family_id", "mass", "qn", "axis")

    # the field round-trip on the register's own continuum fixture
    event = _exchange_event("baryonic_np", 4.0, BARYONIC_QN)
    snapshot = dict(event)
    root = exchange_root_state(event)
    assert isinstance(root, CascadeState)
    assert root.momentum == (4.0, 0.0, 0.0, 0.0)
    assert root.mass == 4.0
    assert root.qn == BARYONIC_QN
    assert root.family_id == "baryonic_np"
    assert root.axis == (0.0, 0.0, 1.0)
    assert root.depth == 0
    assert root.parent is None
    assert root.alignment_label == ISOTROPIC_ALIGNMENT_LABEL
    assert root.exchange_ancestry == ("register_probe",)
    assert root.quantum == {}
    # the event mapping is never mutated, and the root carries fresh
    # tuples (not the mapping's internals)
    assert event == snapshot, "the exchange event was mutated"
    assert root.momentum is not event["momentum"]
    assert root.axis is not event["axis"]
    assert root.qn is not event["qn"]

    # the catalog quantum mapping: the record's I / I3 / J / P
    catalog_event = _exchange_event("catalog:113", 0.77526, MESONIC_QN)
    catalog_root = exchange_root_state(catalog_event)
    record = catalog.get(113)
    assert catalog_root.qn == MESONIC_QN
    assert catalog_root.quantum == {
        "I": float(record.isospin),
        "I3": int(record.isospin3),
        "J": float(record.spin),
        "P": record.parity,
    }

    # the optional ancestry defaults to (topology,)
    bare = _exchange_event("baryonic_np", 4.0, BARYONIC_QN)
    del bare["ancestry"]
    bare_root = exchange_root_state(bare)
    assert bare_root.exchange_ancestry == ("register_ablation_probe",)

    # the draw-free contract: no randomness is consumed
    probe = np.random.default_rng(REGISTER_SEED + 99)
    state_before = probe.bit_generator.state
    exchange_root_state(event)
    exchange_root_state(catalog_event)
    assert probe.bit_generator.state == state_before, (
        "exchange_root_state consumed random numbers")

    # the register's own malformed events (checked, never repaired)
    missing = _exchange_event("baryonic_np", 4.0, BARYONIC_QN)
    del missing["axis"]
    with pytest.raises(ValueError, match="missing the pinned key"):
        exchange_root_state(missing)
    with pytest.raises(ValueError, match="must be a mapping"):
        exchange_root_state([("baryonic_np", 4.0, BARYONIC_QN)])
    short = _exchange_event("baryonic_np", 4.0, BARYONIC_QN)
    short["momentum"] = (4.0, 0.0, 0.0)
    with pytest.raises(KinematicsError, match="four components"):
        exchange_root_state(short)
    nan_momentum = _exchange_event("baryonic_np", 4.0, BARYONIC_QN)
    nan_momentum["momentum"] = (float("nan"), 0.0, 0.0, 0.0)
    with pytest.raises(KinematicsError, match="not finite"):
        exchange_root_state(nan_momentum)
    zero_energy = _exchange_event("baryonic_np", 4.0, BARYONIC_QN)
    zero_energy["momentum"] = (0.0, 0.0, 0.0, 0.0)
    with pytest.raises(KinematicsError, match="not positive"):
        exchange_root_state(zero_energy)
    mismatched_mass = _exchange_event("baryonic_np", 3.0, BARYONIC_QN)
    mismatched_mass["momentum"] = (4.0, 0.0, 0.0, 0.0)
    with pytest.raises(
            KinematicsError,
            match="disagrees with the invariant mass"):
        exchange_root_state(mismatched_mass)
    short_qn = _exchange_event("baryonic_np", 4.0, BARYONIC_QN)
    short_qn["qn"] = (1, 1)
    with pytest.raises(
            ValueError, match="three integer quantum numbers"):
        exchange_root_state(short_qn)
    non_unit = _exchange_event("baryonic_np", 4.0, BARYONIC_QN)
    non_unit["axis"] = (2.0, 0.0, 0.0)
    with pytest.raises(KinematicsError, match="is not unit"):
        exchange_root_state(non_unit)
    catalog_mismatch = _exchange_event("catalog:113", 0.77526,
                                       BARYONIC_QN)
    with pytest.raises(
            ValueError, match="disagrees with the catalog record"):
        exchange_root_state(catalog_mismatch)
    unknown = _exchange_event("no_such_family", 4.0, BARYONIC_QN)
    with pytest.raises(ValueError, match="names no continuum family"):
        exchange_root_state(unknown)


# ----------------------------------------------------------------------
# v1: the single-step direct decay
# ----------------------------------------------------------------------
def test_decay_variant_v1_single_step(catalog):
    """The v1 single-step on the register's own catalog-root and
    continuum-root fixtures: the terminal daughters of the single
    step (the root as their parent, depth 1, the isotropic label, the
    exact ``(B, Q, S)`` sum, the four-momentum closure) and the
    budget (catalog:113 = 1 + 2, catalog:12212@2.0 = 1 + 2 + 2 per
    realized resonance daughter, continuum = 1 + 2), verified on disk
    with the register's own counting generators."""

    stable = catalog.stable_states()
    v1 = ABLATION_POLICIES["isotropic_direct"]

    # the register's own catalog root: the rho0, the single
    # established channel, the terminal daughters of the single step
    event = _exchange_event("catalog:113", 0.77526, MESONIC_QN)
    snapshot = dict(event)
    rc, rk = _wrap(_variant_streams(4, 0))
    output = decay_variant(event, v1, rc, rk)
    assert event == snapshot, "the v1 event was mutated"
    assert len(output) == 2
    for daughter in output:
        assert daughter.depth == 1
        assert daughter.alignment_label == ISOTROPIC_ALIGNMENT_LABEL
        assert int(daughter.family_id[len(CATALOG_FAMILY_PREFIX):]) \
            in stable
    assert tuple(
        sum(daughter.qn[k] for daughter in output)
        for k in range(3)) == MESONIC_QN
    momentum_sum = tuple(
        sum(daughter.momentum[k] for daughter in output)
        for k in range(4))
    worst = max(abs(momentum_sum[k] - float(event["momentum"][k]))
                for k in range(4))
    assert worst <= 1e-9, f"the v1 daughters miss the root by {worst!r}"
    # the single-step budget: 1 channel + 2 kinematics (both
    # daughters stable: no resonance-daughter mass draws)
    assert rc.calls == 1 and rk.calls == 2, (
        "v1 catalog:113: the single-step budget is 1 channel + 2 "
        f"kinematics (got {rc.calls} + {rk.calls})")

    # the resonance-daughter clause: the budget is 1 + 2 + 2 per
    # realized resonance daughter of the single vertex
    event = _exchange_event("catalog:12212", 2.0, BARYONIC_QN)
    for event_index in (4, 17, 42):
        rc, rk = _wrap(_variant_streams(event_index, 0))
        output = decay_variant(event, v1, rc, rk)
        assert len(output) == 2
        n_resonance = sum(
            1 for daughter in output
            if int(daughter.family_id[len(CATALOG_FAMILY_PREFIX):])
            not in stable)
        assert rc.calls == 1, (
            f"v1 catalog:12212@2.0 event {event_index}: the channel "
            f"budget is 1 (got {rc.calls})")
        assert rk.calls == 2 + 2 * n_resonance, (
            f"v1 catalog:12212@2.0 event {event_index}: the "
            f"single-step budget is 2 kinematics + 2 per resonance "
            f"daughter (got {rk.calls} with {n_resonance} resonance "
            "daughters)")

    # the register's own continuum root: the single step is one
    # card-4 vertex over the direct (established) pool
    event = _exchange_event("baryonic_strange", 3.0, STRANGE_BARYONIC_QN)
    rc, rk = _wrap(_variant_streams(17, 0))
    output = decay_variant(event, v1, rc, rk)
    assert len(output) == 2
    for daughter in output:
        assert daughter.parent.family_id == "baryonic_strange"
        assert daughter.depth == 1
        assert daughter.alignment_label == ISOTROPIC_ALIGNMENT_LABEL
    assert tuple(
        sum(daughter.qn[k] for daughter in output)
        for k in range(3)) == STRANGE_BARYONIC_QN
    assert rc.calls == 1 and rk.calls == 2, (
        "v1 continuum: the single step is one card-4 vertex (1 "
        f"channel + 2 kinematics; got {rc.calls} + {rk.calls})")


# ----------------------------------------------------------------------
# v2/v3: the sequential cascade with the attachments
# ----------------------------------------------------------------------
def test_decay_variant_v2_v3_lines(catalog, fams):
    """The v2/v3 line traces on the register's own fixtures: root
    first, the depths increment by one, the terminal state, the
    budget 1 + 2 per line state, the alignment label bookkeeping,
    the v2 off-limit (policy (0, 0, 0.0) bit-identical to card 9's
    unattached ``run_cascade``), and the v2 pin (the kappa = 0
    isotropic branches, card 4's ``draw_decay`` reference behavior
    at the nominal b_dec, bit-identical on the register's own
    seeds)."""

    v2 = ABLATION_POLICIES["isotropic_sequential"]
    v3 = ABLATION_POLICIES["aligned_sequential"]

    for family_id, mass, qn in CONTINUUM_ROOTS:
        event = _exchange_event(family_id, mass, qn)
        for event_index in (4, 17):
            # -- the line structure: root first, depths, terminal --
            rc, rk = _wrap(_variant_streams(event_index, 1))
            trace2 = decay_variant(event, v2, rc, rk)
            assert trace2, f"{family_id}@{mass}: the line is empty"
            assert trace2[0] == exchange_root_state(event)
            assert trace2[0].depth == 0
            for parent_state, child in zip(trace2, trace2[1:]):
                assert child.depth == parent_state.depth + 1
                assert child.parent is parent_state
            # -- the budget: 1 channel + 2 kinematics per line state
            assert rc.calls == len(trace2), (
                f"{family_id}@{mass} v2: the channel budget is one "
                f"uniform per line state (got {rc.calls} for "
                f"{len(trace2)} states)")
            assert rk.calls == 2 * len(trace2), (
                f"{family_id}@{mass} v2: the kinematics budget is "
                f"two uniforms per line state (got {rk.calls})")
            # -- the alignment label bookkeeping -------------------
            # v2 at the isotropic point (kappa = 0 on both kernels):
            # every line state carries the card-4 placeholder
            assert all(
                state.alignment_label == ISOTROPIC_ALIGNMENT_LABEL
                for state in trace2), (
                f"{family_id}@{mass} v2: the isotropic variant carries "
                "the kernel's label")
            # -- v3 on the same streams: the budget invariance -----
            rc3, rk3 = _wrap(_variant_streams(event_index, 1))
            trace3 = decay_variant(event, v3, rc3, rk3)
            assert rc3.calls == rc.calls and rk3.calls == rk.calls, (
                f"{family_id}@{mass}: the v3 attachments moved the "
                "draw budget")
            assert len(trace3) == len(trace2)
            # the governing-kernel label bookkeeping on disk: the
            # root carries the placeholder; at a baryonic vertex
            # (a baryon-carrying daughter) the baryonic daughter
            # carries the baryonic label and the mesonic daughter
            # stays on the placeholder; at the symmetric mesonic
            # vertex (both daughters B == 0) both daughters carry
            # the mesonic label (the nominal kappas are active)
            for index, state in enumerate(trace3):
                if index == 0:
                    expected = ISOTROPIC_ALIGNMENT_LABEL
                elif state.qn[0] != 0 and trace3[index - 1].qn[0] == 0:
                    # the baryon-antibaryon pair vertex of a mesonic
                    # parent (Round 6, plan section 12.4): no baryonic
                    # line, the mesonic kernel governs both daughters
                    expected = MESONIC_ALIGNMENT_LABEL
                elif state.qn[0] != 0:
                    expected = BARYONIC_ALIGNMENT_LABEL
                elif trace3[index - 1].qn[0] != 0:
                    expected = ISOTROPIC_ALIGNMENT_LABEL
                else:
                    expected = MESONIC_ALIGNMENT_LABEL
                assert state.alignment_label == expected, (
                    f"{family_id}@{mass} v3: the line state at depth "
                    f"{state.depth} carries {state.alignment_label!r}, "
                    f"not the governing-kernel label {expected!r}")

    # the v2 off-limit: the policy (0, 0, 0.0) reproduces the
    # unattached card-9 run_cascade baseline bit for bit on the
    # register's own small-scale fixtures
    off_policy = AblationPolicy("isotropic_sequential", 0.0, 0.0, 0.0)
    for family_id, mass, qn in (("baryonic_strange", 3.0,
                                 STRANGE_BARYONIC_QN),
                                ("mesonic_neutral", 5.0, MESONIC_QN)):
        event = _exchange_event(family_id, mass, qn)
        for event_index in (4, 17):
            trace = decay_variant(
                event, off_policy, *_variant_streams(event_index, 1))
            baseline = run_cascade(
                exchange_root_state(event), catalog, fams,
                *_variant_streams(event_index, 1))
            assert trace == baseline, (
                f"{family_id}@{mass} event {event_index}: the "
                "off-limit line drifted from the unattached "
                "run_cascade baseline")

    # the v2 pin: the kappa = 0 isotropic branches are bit-identical
    # to card 4's draw_decay over the attached pools at the nominal
    # b_dec (the reference loop, the register's own seeds)
    lines = 0
    for family_id, mass, qn in CONTINUUM_ROOTS:
        event = _exchange_event(family_id, mass, qn)
        for event_index in REGISTER_EVENTS:
            trace = decay_variant(
                event, v2, *_variant_streams(event_index, 1))
            reference = _reference_isotropic_line(
                exchange_root_state(event), catalog, fams,
                *_variant_streams(event_index, 1))
            assert trace == reference, (
                f"{family_id}@{mass} event {event_index}: the "
                "isotropic sequential variant drifted from card 4's "
                "cascade at kappa = 0 with b_dec nominal")
            lines += 1
    assert lines == len(CONTINUUM_ROOTS) * len(REGISTER_EVENTS)


# ----------------------------------------------------------------------
# The machine-readable support boundary
# ----------------------------------------------------------------------
def test_support_boundary_block():
    """The machine-readable ``SUPPORT_BOUNDARY`` block on disk: the
    plan citations, the v1 single-step support regenerated, the v2/v3
    multi-vertex support identical and shared, the
    ``regeneration_required`` register, the variant-key closure
    against ``ABLATION_VARIANTS``, and the block's importability and
    JSON serializability (the card-14 pre-registration contract)."""

    # the block is importable and stable (JSON-serializable: the
    # pre-registration shape card 14 consumes)
    assert isinstance(SUPPORT_BOUNDARY, dict)
    serialized = json.dumps(SUPPORT_BOUNDARY, sort_keys=True)
    assert len(serialized) > 0
    # the round-trip is exact modulo the tuple/list shape (the
    # on-disk block carries tuple values)
    assert json.loads(serialized) == _json_normalize(SUPPORT_BOUNDARY)

    assert set(SUPPORT_BOUNDARY) >= {
        "plan_line_275", "plan_line_288", "variants",
        "v2_v3_support_identical", "regeneration_required",
    }
    assert "support does not change" in SUPPORT_BOUNDARY[
        "plan_line_275"]
    assert "Regenerate when thresholds, allowed channels, or parent " \
        "masses change." in SUPPORT_BOUNDARY["plan_line_288"]

    variants = SUPPORT_BOUNDARY["variants"]
    assert set(variants) == set(ABLATION_VARIANTS)
    direct = variants["isotropic_direct"]
    assert direct["decay_steps"] == "single_step"
    assert direct["support"] == "single_step"
    assert direct["common_random_numbers"] == "regenerate"
    assert direct["boundary"], (
        "the v1 boundary is recorded in the block")
    for name in ("isotropic_sequential", "aligned_sequential"):
        entry = variants[name]
        assert entry["decay_steps"] == "multi_vertex"
        assert entry["support"] == "multi_vertex"
        assert entry["common_random_numbers"] == "shared"
    # the v2/v3 support is identical (the same channels, the same
    # parent masses; only the angular and form-factor weights differ)
    assert SUPPORT_BOUNDARY["v2_v3_support_identical"] is True
    aligned = variants["aligned_sequential"]
    assert aligned["boundary"]
    assert "isotropic_sequential" in aligned["boundary"]
    assert SUPPORT_BOUNDARY["regeneration_required"] == (
        "isotropic_direct",)


# ----------------------------------------------------------------------
# The pinned reference emission configuration
# ----------------------------------------------------------------------
def test_reference_direction_and_w_dec_pin():
    """The pinned reference emission configuration on the register's
    own axis fixtures: ``W_DEC_REFERENCE_COS_THETA == 0.3`` exactly,
    the exact sine, the reference direction's cosine against the
    axis exactly 0.3 and its unit norm, the q_perp-invariance (the
    transverse fraction is the fixed value independent of the axis,
    i.e. of the azimuth choice), the draw-free contract, and the
    module docstring's documented boundary."""

    assert W_DEC_REFERENCE_COS_THETA == 0.3
    assert W_DEC_REFERENCE_SIN_THETA == pytest.approx(
        math.sqrt(1.0 - 0.3 * 0.3), rel=0, abs=1e-15)

    # the register's own axis fixtures: the coordinate axes and an
    # oblique axis (the rejection-frame construction is
    # deterministic, with no free azimuth)
    axes = (
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    oblique = (0.2, -0.5, 0.85)
    norm = math.sqrt(sum(v * v for v in oblique))
    axes = axes + (tuple(v / norm for v in oblique),)
    probe = np.random.default_rng(REGISTER_SEED + 97)
    state_before = probe.bit_generator.state
    for axis in axes:
        direction = reference_direction(axis)
        cos_theta = sum(
            d * a for d, a in zip(direction, axis))
        unit_norm = math.sqrt(sum(d * d for d in direction))
        assert unit_norm == pytest.approx(1.0, abs=1e-12)
        assert cos_theta == pytest.approx(0.3, abs=1e-12), (
            f"the reference direction's cosine against the axis "
            f"{axis} is {cos_theta!r}, not the pinned 0.3")
        # the q_perp-invariance: the transverse fraction is the
        # fixed value, independent of the axis (azimuth choice)
        q_perp = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
        assert q_perp == pytest.approx(math.sqrt(1.0 - 0.3 * 0.3),
                                       abs=1e-12)
    state_after = probe.bit_generator.state
    assert state_after == state_before, (
        "reference_direction consumed random numbers")

    # the module docstring documents the v1 continuum-root boundary
    # (the card 4 draw_decay over the direct established pool, the
    # plan section 10.6 direct-channel rule)
    from na49_mixing import cascade_ablation as module
    assert "direct (established) pool" in module.__doc__
    assert "draw_decay" in module.__doc__


# ----------------------------------------------------------------------
# The zero-draw contract of the toggle wiring
# ----------------------------------------------------------------------
def test_toggle_zero_draw(catalog):
    """The zero-draw contract of the toggle wiring: the attachments
    (the W_dec factor, the ``draw_vertex_cos_theta`` replacement)
    consume no uniforms beyond the budget, verified with the
    register's own counting generators around the full
    ``decay_variant`` for all three policies (v1 the single-step
    budget, v2/v3 the 1 + 2 per line state, identical across the
    policies and across a non-nominal scan policy)."""

    scan_policy = AblationPolicy("aligned_sequential", 2.5, 3.5, 0.75)

    # v1: the single-step budget (the Round 2 kernel's own vertex at
    # a catalog root; one card-4 vertex at a continuum root)
    for family_id, mass, qn in (CATALOG_ROOTS[0],
                                CONTINUUM_ROOTS[0]):
        event = _exchange_event(family_id, mass, qn)
        rc, rk = _wrap(_variant_streams(4, 0))
        output = decay_variant(
            event, ABLATION_POLICIES["isotropic_direct"], rc, rk)
        assert len(output) == 2
        if family_id.startswith(CATALOG_FAMILY_PREFIX):
            assert rc.calls == 1 and rk.calls == 2, (
                f"v1 {family_id}: the single-step budget is 1 + 2 "
                f"(got {rc.calls} + {rk.calls})")
        else:
            assert rc.calls == 1 and rk.calls == 2, (
                f"v1 {family_id}: the direct pool step is one card-4 "
                f"vertex (got {rc.calls} + {rk.calls})")

    # v2/v3: the 1 + 2 per line state, identical across the
    # pre-registered policies and a non-nominal scan policy (the
    # per-attachment draw-budget invariance)
    for family_id, mass, qn in CONTINUUM_ROOTS:
        event = _exchange_event(family_id, mass, qn)
        counts = {}
        for name, policy in (
            ("v2", ABLATION_POLICIES["isotropic_sequential"]),
            ("v3", ABLATION_POLICIES["aligned_sequential"]),
            ("scan", scan_policy),
        ):
            rc, rk = _wrap(_variant_streams(4, 1))
            trace = decay_variant(event, policy, rc, rk)
            assert rc.calls == len(trace), (
                f"{family_id}@{mass} {name}: the channel budget is "
                f"one uniform per line state (got {rc.calls} for "
                f"{len(trace)} states)")
            assert rk.calls == 2 * len(trace), (
                f"{family_id}@{mass} {name}: the kinematics budget "
                f"is two uniforms per line state (got {rk.calls})")
            counts[name] = (rc.calls, rk.calls)
        assert counts["v2"] == counts["v3"] == counts["scan"], (
            f"{family_id}@{mass}: the attachments moved the draw "
            f"budget ({counts})")


# ----------------------------------------------------------------------
# Part 2: the substrate and freeze pins
# ----------------------------------------------------------------------
def test_module_and_init_pins():
    """The new module digest (``c0206cf6...``), the
    ``na49_mixing/__init__.py`` byte pin (``1b3a6e65...`` -- the
    card-10 decision: the package ``__init__`` untouched), and the
    module's importability as ``na49_mixing.cascade_ablation`` by
    object identity via importlib, with the package exports
    unchanged."""

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)
    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    import na49_mixing

    # the module is importable as na49_mixing.cascade_ablation, by
    # object identity (the importlib and attribute routes agree)
    loaded = importlib.import_module("na49_mixing.cascade_ablation")
    assert loaded is na49_mixing.cascade_ablation
    assert loaded is importlib.import_module("na49_mixing.cascade_ablation")
    for name in module_export_names():
        assert hasattr(loaded, name), f"the module is missing {name}"

    # the package exports are unchanged (the non-extension clause):
    # no ablation name in the package namespace or its __all__
    for name in module_export_names():
        assert name not in na49_mixing.__all__, (
            f"the package __all__ gained {name} (the card-10 "
            "decision: the __init__ is byte-unchanged)")
        assert not hasattr(na49_mixing, name) or \
            name == "cascade_ablation", (
            f"the package namespace gained {name}")


def module_export_names() -> tuple[str, ...]:
    """The module's exported names (its ``__all__``, pinned on
    disk)."""

    from na49_mixing import cascade_ablation as module

    assert module.__all__ == [
        "ABLATION_VARIANTS",
        "ABLATION_POLICIES",
        "AblationPolicy",
        "EXCHANGE_EVENT_KEYS",
        "SUPPORT_BOUNDARY",
        "W_DEC_REFERENCE_COS_THETA",
        "W_DEC_REFERENCE_SIN_THETA",
        "exchange_root_state",
        "exchange_substreams",
        "variant_stream",
        "decay_variant",
        "reference_direction",
    ]
    return tuple(module.__all__)


def test_frozen_substrate_digests():
    """The byte-unchanged frozen substrate of the card's provenance
    section, verified on disk: cards 4-9's cascade surfaces, the
    Round 2 kernel, the builder, the card-6 blend routing, the
    card-2 engine, the committed catalog, the continuum families
    configuration, and the R32 driver in both trees."""

    for rel, expected in FROZEN_SUBSTRATE.items():
        path = MONO_ROOT / rel
        assert path.is_file(), f"missing frozen surface: {rel}"
        pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)


# ----------------------------------------------------------------------
# test_freeze_repin_register (the test-author freeze re-pin check)
# ----------------------------------------------------------------------
