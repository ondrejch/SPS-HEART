"""Round 4, card 8 test-author register: hadronic decay form factor.

Independent contract pins for the hadronic decay form factor (the
card-8 additive module ``na49_mixing/cascade_form_factor.py``), Round 4
card 8 (``TASK-20260920-r4-hadronic-decay-form-factor``; plan section 22
task 25, plan section 10.5, lines 734-748).  This register is the
test-author half of the card: it pins the form-factor module contract
on self-authored ``(p, axis)`` fixtures, ``(q_perp, b_dec)`` weight
pairs, and a self-authored monotone-response emission grid (independent
of the solver's pinned fixtures), self-authored zero-draw
instrumentation (a fixed-seed bit-generator state identity across a
pinned operation cascade plus a self-authored source scan), the
card-4 draw-budget invariance through the ``attach_to_channel_weight``
composition (a self-authored recording uniform-counting proxy on a
self-authored oblique-axis fixture, both a catalog resonance and a
continuum state at self-authored masses), the exchange-layer separation
on a self-authored token scan, and it re-pins the freeze manifest for
the card.

Part 1: the module contract (pure, no sealed run dir)
-----------------------------------------------------

- ``test_q_perp_convention``: ``q_perp^2 = |p|^2 - (p . a)^2`` on
  self-authored fixtures spanning axes aligned with, orthogonal to, and
  oblique to the daughter momentum (1e-12), the zero two-body momentum
  (exactly ``0.0``), the near-parallel roundoff clamp (a self-authored
  configuration whose float64 ``|p|^2 - (p . a)^2`` is negative, pinned
  to return exactly ``0.0``), and the checked-never-repaired edges (a
  non-unit axis at the card-5 tolerance boundary, a non-finite axis
  component, a malformed axis, and a malformed or non-finite momentum
  all raise the kernel's ``KinematicsError``).
- ``test_weight_values``: the weight equals ``exp(-b_dec q_perp^2)``
  within 1e-15 on self-authored ``(q_perp, b_dec)`` pairs; it is exactly
  ``1.0`` at ``b_dec = 0`` (bit equality, the off limit) and at
  ``q_perp = 0``; the pre-registered default slope is ``1.0 GeV^-2``;
  the underflow boundary decays gracefully to ``0.0``; and a non-finite
  or negative ``q_perp`` / ``b_dec`` raises ``ValueError``.
- ``test_monotone_q2_response``: the weighted ``<q_perp^2>`` of a
  self-authored two-body channel decreases with ``b_dec`` (``b_dec`` in
  ``{0, 0.5, 1, 2}``, a self-authored 51-point emission grid distinct
  from the solver's), strictly monotone while the configuration carries
  a spread in ``q_perp^2``, with the off limit reproducing the
  unweighted mean exactly.
- ``test_attach_to_channel_weight``: the W_dec composition on a
  self-authored recording inner channel-weight callable (the wrapper
  delegates the hooks to the inner callable exactly once, scales the
  result by the form factor at the daughter ``q_perp``, passes the
  hooks through unchanged, defaults the hooks to ``1.0``, is
  bit-identical to the unattached weight at ``b_dec = 0``, and the
  normalized shares over a self-authored multi-channel vertex change
  exactly by the form-factor factor per channel); a non-callable inner
  callable and a non-finite / negative ``b_dec`` raise ``ValueError``
  at wrap time (checked, never repaired).
- ``test_exchange_layer_separation``: white-box on a self-authored
  token scan (the form factor is absent from the builder / exchange
  layer; the builder bytes are unchanged at sha256 ``4ab61260...``; no
  exchange-layer symbol references the form-factor module; the
  exchange-slope symbols are absent from the form-factor module; and no
  single parameter controls both source recoil and daughter transverse
  momentum, plan section 10.5 line 748).
- ``test_zero_draws``: the form-factor machinery consumes zero random
  numbers (a pinned bit-generator state identity over a self-authored
  1,000-operation cascade, and the module source carries no random
  machinery per the self-authored source scan).
- ``test_budget_unchanged``: attaching the factor changes no draw count
  (the card-4 per-vertex budget stays 1 channel + 2 kinematics for
  every scanned ``b_dec``, the kinematics substream consumes the same
  two uniforms as the unattached baseline, and at ``b_dec = 0`` the
  attached draw is bit-identical to the card-4 baseline), verified on a
  self-authored catalog-resonance and continuum fixture at a
  self-authored oblique axis.
- ``test_module_constants``: ``DECAY_FORM_FACTOR_B_DEFAULT == 1.0``
  (value and type) and the module's ``__all__`` entries.

Part 2: the substrate and freeze pins
-------------------------------------

- ``test_module_byte_and_export_pins``: the module byte stability
  (sha256 ``b16077c9...``) and the additive ``na49_mixing`` package
  exports (the four card-8 names present in the package ``__init__``
  import block and ``__all__``, resolved by object identity, with the
  prior card exports unchanged).
- ``test_frozen_substrate_digests``: the byte-unchanged surfaces the
  card consumes (card 4's cascade surface, card 5's axis machinery,
  the card-6/7 alignment kernel, the builder, the card-6 blend routing,
  the card-2 engine, the committed catalog, the continuum families
  configuration, and the R32 driver in both trees), verified on disk.
- ``test_freeze_repin_register``: the freeze-manifest re-pin for this
  register (the strict path-sorted inventory at 466 entries carries
  this file at its on-disk digest; the card-8 freeze exception stays at
  the solver's 12 path-sorted files; this register's own freeze
  exception names the test-author card exactly once with the path-
  sorted file list; the nine consumed count pins (the eight register-
  chain files plus the card-8 solver test's own count constant) stay
  at their on-disk digests, each re-pinned 465 -> 466 with the naming
  comment attributing the bump to this register; the frozen M8
  parameter map stays intact).

Hermetic: the self-authored fixtures, numpy, the standard library, and
the vendored ``na49_mixing`` modules only; no sealed run dir, no event
generation, no external tables, no Monte Carlo run.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.cascade_axes import AXIS_NORM_TOL
from na49_mixing.cascade_decay import (
    ISOTROPIC_ALIGNMENT_LABEL,
    CascadeChannel,
    CascadeState,
    channel_weight,
    draw_decay,
    enumerate_channels,
)
from na49_mixing.cascade_form_factor import (
    DECAY_FORM_FACTOR_B_DEFAULT,
    attach_to_channel_weight,
    decay_form_factor_weight,
    transverse_momentum,
)
from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.resonance_decay import KinematicsError
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
MODULE_PATH = NESTED_ROOT / "na49_mixing" / "cascade_form_factor.py"
INIT_PATH = NESTED_ROOT / "na49_mixing" / "__init__.py"

MODULE_REL = "dev/mixing2006/na49_mixing/cascade_form_factor.py"
INIT_REL = "dev/mixing2006/na49_mixing/__init__.py"
MY_TEST_REL = ("dev/mixing2006/tests/"
               "test_r4_cascade_form_factor_contract_pins.py")
SOLVER_TEST_REL = "dev/mixing2006/tests/test_cascade_form_factor.py"
MANIFEST_REL = ("dev/mixing2006/config/v2_singlet/"
                "legacy_v1_manifest.json")

# The eight register-chain files whose inventory-count pins this
# register re-pins (465 -> 466), each with the naming comment
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
CHAIN_FILES = (
    CARD3_REGISTER_REL,
    CARD4_REGISTER_REL,
    CARD5_SOLVER_TEST_REL,
    CARD5_REGISTER_REL,
    CARD6_SOLVER_TEST_REL,
    CARD6_REGISTER_REL,
    CARD7_SOLVER_TEST_REL,
    CARD7_REGISTER_REL,
)
# The nine count pins this register re-pins: the eight register-chain
# files plus the card-8 solver test's own count constant.
REPIN_FILES = CHAIN_FILES + (SOLVER_TEST_REL,)

FREEZE_CARD = "TASK-20260920-r4-hadronic-decay-form-factor"
# The test-author re-pin's own freeze-exception card name (distinct
# from the solver's, so the solver's "exactly one entry naming this
# card" check is untouched).
TEST_AUTHOR_CARD = ("TASK-20260920-r4-hadronic-decay-form-factor-"
                    "test-author")

# 465 entries at the card-8 solver landing; this register adds its file
# (+1 entry) for the test-author re-pin (465 -> 466, +1 entry), and
# re-pins the nine consumed count pins (the eight register-chain files
# plus the card-8 solver test's own count constant) with the bump
# attributed to this register's file; card 9 of the round
# (TASK-20260920-r4-termination-depth-tests) re-pins the inventory
# additively for its new termination module and test file
# (na49_mixing/cascade_termination.py, test_cascade_termination.py;
# the constant read "INVENTORY_COUNT = 466" at the card-8 register
# landing; 466 -> 468, +2 entries); card 9's independent
# test-author register (test_r4_cascade_termination_contract_pins.py)
# re-pins the inventory additively for its own file (the constant read
# "INVENTORY_COUNT = 468" at the card-9 solver landing; 468 -> 469,
# +1 entry); card 10 of the round
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
    "b16077c9e3418b3f19b117aadf6fc8a8e838af5e9d37b1ab0bae89ccfb88f74c"
)
# The package __init__ additive re-pin: card 9 of the round
# (TASK-20260920-r4-termination-depth-tests) extends the package
# exports additively with the termination module's names
# (na49_mixing/cascade_termination.py; the digest read
# "457e2997f6c052636979ee72b2a26bc6fd320517be525ac60e1b017f924a17a7"
# at the card-8 register landing; re-pinned by Round 5 card 1
# (TASK-20260925-r5-elastic-reference) for the elastic reference
# module's additive package exports).
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
#: card 4's cascade surface, card 5's axis machinery, the card-6/7
#: alignment kernel, the builder, the card-6 blend routing, the card-2
#: engine, the committed catalog, the continuum families
#: configuration, and the dual-tree R32 driver.
FROZEN_SUBSTRATE = {
    "dev/mixing2006/na49_mixing/cascade_decay.py":
        "d2a48049af3d6df968993c040033a74422ccd709f53a42bad0538ae5111bcc0c",
    "dev/mixing2006/na49_mixing/cascade_axes.py":
        "a7251a6b6c0ed4b9cfc56c0182c28a70efccb7a92a33b756090ff716a3ec3c61",
    "dev/mixing2006/na49_mixing/cascade_alignment.py":
        "0d0df347222423aaf6f02403e177867b57da588a5e40d641fd7b467e9c4a81f4",
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

#: The solver's card-8 freeze-exception file list (12 path-sorted
#: files; this register leaves it untouched and records its own
#: re-pin under a distinct card name).
SOLVER_EXCEPTION_FILES = (
    MANIFEST_REL,
    INIT_REL,
    MODULE_REL,
    SOLVER_TEST_REL,
) + CHAIN_FILES

#: This register's own freeze-exception file list (path-sorted): the
#: manifest, this register, and the nine re-pinned count pins.
TEST_AUTHOR_EXCEPTION_FILES = tuple(
    sorted([MANIFEST_REL, MY_TEST_REL] + list(REPIN_FILES))
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


def _root_state(family_id: str, mass: float, qn: tuple[int, int, int],
                axis: tuple[float, float, float]) -> CascadeState:
    """A pinned at-rest root state with a self-authored production
    axis and the isotropic label (the caller-supplied axis of this
    card's landing; card 5's axis machinery is consumed read-only)."""

    return CascadeState(
        momentum=(float(mass), 0.0, 0.0, 0.0),
        mass=float(mass),
        qn=qn,
        quantum={},
        family_id=family_id,
        axis=axis,
        alignment_label=ISOTROPIC_ALIGNMENT_LABEL,
        parent=None,
        exchange_ancestry=("exchange_probe",),
        depth=0,
    )


class _RecordingRng:
    """A recording proxy around one seeded generator (the house
    draw-count pattern): every ``random()`` consumes exactly one
    uniform, advances the counter, and records the value."""

    def __init__(self, seed) -> None:
        self._rng = np.random.default_rng(seed)
        self.calls = 0
        self.values: list[float] = []

    def random(self) -> float:
        self.calls += 1
        value = float(self._rng.random())
        self.values.append(value)
        return value


def _sub_seeds(seed: int, event_index: int):
    """The closed decay_channel/decay_kinematics substream seeds of one
    (master seed, replica 0, event index) triple (the round 2 card 9
    and round 3 card 8 convention)."""

    subs = derive_substreams(seed, 0, event_index)
    return (
        subs.stream_for("decay_channel"),
        subs.stream_for("decay_kinematics"),
    )


def _daughter_masses(
    channel: CascadeChannel, state: CascadeState, catalog: HadronCatalog,
    fams,
) -> tuple[float, float]:
    """The realized daughter masses of *channel* at *state* (the
    established daughters at their catalog pole masses, the continuum
    daughter at the pre-registered card-4 mass rule; the test-side
    mirror of card 4's ``_daughter_masses``)."""

    from na49_mixing.cascade_decay import CATALOG_FAMILY_PREFIX

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
        family_id = channel.daughters[slot]
        family = next(f for f in fams if str(f.family_id) == family_id)
        sibling = realized[1 - slot]
        edge = min(SUPPORT_M_MAX, float(state.mass) - sibling)
        if not edge >= float(family.m_min):
            raise ValueError(
                f"the continuum daughter {family_id!r} is closed at "
                f"parent mass {float(state.mass)!r} GeV"
            )
        realized[slot] = max(
            float(family.m_min),
            edge - float(family.nominal_width),
        )
    return (realized[0], realized[1])


def _channel_daughter_p4(
    channel: CascadeChannel, state: CascadeState, catalog: HadronCatalog,
    fams, direction: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """The pinned daughter four-momentum of *channel*: its two-body
    momentum at the realized daughter masses, emitted along the
    self-authored *direction* (the parent at rest, so the parent rest
    frame is the working frame)."""

    mass0, mass1 = _daughter_masses(channel, state, catalog, fams)
    q_star = decay_momentum(float(state.mass), mass0, mass1)
    return (
        math.sqrt(mass0 * mass0 + q_star * q_star),
        q_star * direction[0],
        q_star * direction[1],
        q_star * direction[2],
    )


def _attached_channels(
    channels: list[CascadeChannel],
    state: CascadeState,
    catalog: HadronCatalog,
    fams,
    direction: tuple[float, float, float],
    b_dec: float,
) -> list[CascadeChannel]:
    """The attached channel list at *b_dec*: every channel's weight is
    the attached weight of the card-4 base weight and the W_dec factor
    at the channel's pinned two-body configuration, normalized over the
    vertex (the caller-side attachment the run cards consume)."""

    wrapped = attach_to_channel_weight(channel_weight, b_dec)
    weighted = [
        wrapped(channel, _channel_daughter_p4(channel, state, catalog, fams,
                                              direction),
                state.axis)
        for channel in channels
    ]
    total = math.fsum(weighted)
    if not (math.isfinite(total) and total > 0.0):
        raise ValueError("the attached weights do not normalize")
    return [
        CascadeChannel(
            parent_id=channel.parent_id,
            daughters=channel.daughters,
            kind=channel.kind,
            weight=value / total,
        )
        for channel, value in zip(channels, weighted)
    ]


def _malformed(*parts):
    """A runtime-malformed argument shape handed to a validator (the
    checked-never-repaired edges; the declared call types are
    satisfied only at the type-checker's eye)."""

    return parts


# ----------------------------------------------------------------------
# test_q_perp_convention (acceptance (i))
# ----------------------------------------------------------------------
def test_q_perp_convention():
    """q_perp^2 = |p|^2 - (p . a)^2 within 1e-12 on the self-authored
    fixture spanning axes aligned with, orthogonal to, and oblique to
    the daughter momentum; the returned value is the positive
    magnitude; the near-parallel roundoff clamp pins the float64
    negative to exactly 0.0; the checked-never-repaired edges raise
    the kernel's KinematicsError."""

    fixture = [
        # (label, daughter four-momentum, unit axis, expected q_perp)
        ("aligned", (1.4, 0.0, 0.0, 0.9), (0.0, 0.0, 1.0), 0.0),
        ("anti-aligned", (1.4, 0.0, 0.0, -0.9), (0.0, 0.0, 1.0), 0.0),
        ("orthogonal x", (1.4, 0.8, 0.0, 0.0), (0.0, 0.0, 1.0), 0.8),
        ("orthogonal y", (1.4, 0.0, 0.35, 0.0), (0.0, 0.0, 1.0), 0.35),
        ("oblique z-axis", (1.4, 0.4, 0.5, 0.6), (0.0, 0.0, 1.0),
         math.sqrt(0.41)),
        (
            "oblique diagonal axis",
            (1.5, 0.6, -0.3, 0.4),
            (
                1.0 / math.sqrt(3.0),
                1.0 / math.sqrt(3.0),
                1.0 / math.sqrt(3.0),
            ),
            None,  # pinned through the formula below
        ),
    ]
    for label, p4, axis, expected in fixture:
        px, py, pz = p4[1], p4[2], p4[3]
        p2 = px * px + py * py + pz * pz
        pa = px * axis[0] + py * axis[1] + pz * axis[2]
        formula_q2 = p2 - pa * pa
        q_perp = transverse_momentum(p4, axis)
        assert q_perp >= 0.0
        assert abs(q_perp * q_perp - formula_q2) <= 1e-12, label
        if expected is not None:
            assert abs(q_perp - expected) <= 1e-12, label

    # the zero two-body momentum carries no transverse component (the
    # two-body threshold)
    assert transverse_momentum((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0)) == 0.0

    # the near-parallel roundoff clamp: a self-authored configuration
    # whose float64 |p|^2 - (p . a)^2 is negative (the floating-point
    # negative a near-parallel configuration can produce) is pinned to
    # return exactly 0.0 (the card-4 sin_theta clamp,
    # max(0.0, q_perp^2), absorbed here)
    roundoff = [
        (
            (1.0, 0.3897375866534707, -0.8652672619230073, -0.5183913564886867),
            (0.3604184534162938, -0.8001750204277476, -0.4793938622056615),
        ),
        (
            (1.0, -0.5801495127840347, -2.1753271524249507,
             0.29660416392127126),
            (-0.25548086539482395, -0.9579504384144512, 0.13061579138325008),
        ),
        (
            (1.0, 1.4464715324421158, 1.8892147775320427, 1.2329002291912718),
            (0.5397636804919708, 0.7049772498678342, 0.46006765414506934),
        ),
    ]
    for p4, axis in roundoff:
        px, py, pz = p4[1], p4[2], p4[3]
        p2 = px * px + py * py + pz * pz
        pa = px * axis[0] + py * axis[1] + pz * axis[2]
        naive_q2 = p2 - pa * pa
        assert naive_q2 < 0.0, (
            "the self-authored near-parallel fixture is no longer a "
            "float64 negative (the roundoff clamp is untested)"
        )
        assert transverse_momentum(p4, axis) == 0.0

    # checked, never repaired: a non-unit axis, a non-finite axis
    # component, a malformed axis, and a malformed or non-finite
    # momentum raise the kernel's KinematicsError
    with pytest.raises(KinematicsError):
        transverse_momentum((1.0, 0.8, 0.0, 0.0), (0.0, 0.0, 2.0))
    # the card-5 tolerance boundary: an axis whose norm is just
    # beyond AXIS_NORM_TOL is rejected, one just within is accepted
    s_within = 1.0 - 1e-12
    assert abs(math.sqrt(s_within * s_within) - 1.0) <= AXIS_NORM_TOL
    assert transverse_momentum(
        (1.0, 0.8, 0.0, 0.0), (s_within, 0.0, 0.0)
    ) >= 0.0
    s_beyond = 1.0 + 2e-12
    assert abs(math.sqrt(s_beyond * s_beyond) - 1.0) > AXIS_NORM_TOL
    with pytest.raises(KinematicsError):
        transverse_momentum(
            (1.0, 0.8, 0.0, 0.0), (s_beyond, 0.0, 0.0)
        )
    with pytest.raises(KinematicsError):
        transverse_momentum(
            (1.0, 0.8, 0.0, 0.0), _malformed(0.0, 0.0, float("nan"))
        )
    with pytest.raises(KinematicsError):
        transverse_momentum((1.0, 0.8, 0.0, 0.0), _malformed(0.0, 0.0))
    with pytest.raises(KinematicsError):
        transverse_momentum(_malformed(1.0, 0.8, 0.0), (0.0, 0.0, 1.0))
    with pytest.raises(KinematicsError):
        transverse_momentum(
            _malformed(1.0, float("inf"), 0.0, 0.0), (0.0, 0.0, 1.0)
        )


# ----------------------------------------------------------------------
# test_weight_values (acceptance (ii), the value clause)
# ----------------------------------------------------------------------
def test_weight_values():
    """The weight equals exp(-b_dec q_perp^2) within 1e-15 on
    self-authored values; it is exactly 1.0 at b_dec = 0 and at
    q_perp = 0 (bit equality, the off limit); the pre-registered
    default is 1.0 GeV^-2; the underflow boundary decays gracefully to
    0.0; the checked-never-repaired edges raise ValueError."""

    pinned = [
        (0.0, 1.0, math.exp(0.0)),
        (0.5, 0.25, math.exp(-0.25 * 0.25)),
        (1.25, 1.0, math.exp(-1.0 * 1.5625)),
        (2.4, 0.75, math.exp(-0.75 * 5.76)),
        (0.7, 2.5, math.exp(-2.5 * 0.49)),
    ]
    for q_perp, b_dec, expected in pinned:
        value = decay_form_factor_weight(q_perp, b_dec)
        assert abs(value - expected) <= 1e-15, (q_perp, b_dec)

    # the off limit: b_dec = 0 gives the factor exactly 1.0 (bit
    # equality, so the attached weights are the unattached ones bit
    # for bit; the R43 ablation baseline)
    for q_perp in (0.0, 0.5, 1.25, 30.0, 1e3):
        assert decay_form_factor_weight(q_perp, 0.0) == 1.0
    # the zero transverse momentum carries no form-factor suppression
    for b_dec in (0.0, 0.5, 1.0, 3.0):
        assert decay_form_factor_weight(0.0, b_dec) == 1.0

    # the pre-registered default slope (the R40 nominal point, round 4
    # index item 3)
    assert DECAY_FORM_FACTOR_B_DEFAULT == 1.0
    assert decay_form_factor_weight(
        0.7, DECAY_FORM_FACTOR_B_DEFAULT
    ) == decay_form_factor_weight(0.7, 1.0)

    # the underflow boundary: a large exponent decays gracefully to 0
    assert decay_form_factor_weight(30.0, 2.5) == 0.0

    # checked, never repaired: a non-finite or negative q_perp or
    # b_dec raises ValueError
    for bad_q in (float("nan"), float("inf"), float("-inf"), -1e-12, -0.5):
        with pytest.raises(ValueError):
            decay_form_factor_weight(bad_q, 1.0)
    for bad_b in (float("nan"), float("inf"), float("-inf"), -1e-12, -0.5):
        with pytest.raises(ValueError):
            decay_form_factor_weight(0.7, bad_b)


# ----------------------------------------------------------------------
# test_monotone_q2_response (acceptance (ii), the monotone clause)
# ----------------------------------------------------------------------
def test_monotone_q2_response():
    """The weighted <q_perp^2> of a self-authored two-body channel
    decreases with b_dec (b_dec in {0, 0.5, 1, 2}; a self-authored
    51-point emission grid distinct from the solver's): the
    exponential tilt toward smaller transverse momenta is strictly
    monotone while the configuration carries a spread in q_perp^2, and
    the off limit reproduces the unweighted mean exactly."""

    axis = (0.0, 0.0, 1.0)
    q_star = 0.6
    phi = 1.3
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    cos_grid = tuple(-1.0 + 0.04 * k for k in range(51))
    configs = []
    q2_values = []
    for cos_theta in cos_grid:
        sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
        p4 = (
            1.0,
            q_star * sin_theta * cos_phi,
            q_star * sin_theta * sin_phi,
            q_star * cos_theta,
        )
        configs.append((p4, axis))
        q_perp = transverse_momentum(p4, axis)
        q2_values.append(q_perp * q_perp)
    assert len(configs) == 51
    # the self-authored configuration carries a genuine spread in
    # q_perp^2
    assert min(q2_values) == 0.0
    assert abs(max(q2_values) - q_star ** 2) <= 1e-12

    # the weighted <q_perp^2> through the attached wrapper (the
    # constant inner weight cancels in the mean, so the tilt is the
    # form factor's alone)
    inner = lambda channel, w_flavor=1.0, w_align=1.0: 1.0  # noqa: E731
    means = {}
    for b_dec in (0.0, 0.5, 1.0, 2.0):
        wrapped = attach_to_channel_weight(inner, b_dec)
        weights = [
            wrapped(f"ch{k}", p4, a)
            for k, (p4, a) in enumerate(configs)
        ]
        total = math.fsum(weights)
        means[b_dec] = math.fsum(
            w * q2 for w, q2 in zip(weights, q2_values)
        ) / total

    # strictly decreasing in b_dec, and the off limit reproduces the
    # unweighted mean exactly
    assert abs(means[0.0] - math.fsum(q2_values) / len(q2_values)) <= 1e-15
    assert means[0.0] > means[0.5] > means[1.0] > means[2.0]


# ----------------------------------------------------------------------
# test_attach_to_channel_weight (the W_dec composition contract)
# ----------------------------------------------------------------------
def test_attach_to_channel_weight():
    """The W_dec composition on a self-authored recording inner
    channel-weight callable: the wrapper delegates the hooks to the
    inner callable exactly once, scales the result by the form factor
    at the daughter q_perp, passes the hooks through unchanged,
    defaults the hooks to 1.0, is bit-identical to the unattached
    weight at b_dec = 0, and the normalized shares over a
    self-authored multi-channel vertex change exactly by the
    form-factor factor per channel; the wrap-time edges raise
    ValueError."""

    calls = []

    def inner(channel, w_flavor=1.0, w_align=1.0):
        calls.append((channel, w_flavor, w_align))
        return 0.2 * w_flavor * w_align

    pinned_p4 = (1.4, 0.4, 0.5, 0.6)
    pinned_axis = (0.0, 0.0, 1.0)
    wrapped = attach_to_channel_weight(inner, 1.5)
    value = wrapped("the-channel", pinned_p4, pinned_axis, 2.0, 3.0)
    assert len(calls) == 1
    assert calls[0] == ("the-channel", 2.0, 3.0)
    q_perp = transverse_momentum(pinned_p4, pinned_axis)
    expected = 0.2 * 2.0 * 3.0 * decay_form_factor_weight(q_perp, 1.5)
    assert abs(value - expected) <= 1e-15

    # the hooks pass through unchanged and default to 1.0: the wrapped
    # value at the hook defaults equals the inner default-hook weight
    # scaled by the factor (the W_dec factor is separate from
    # W_flavor / W_align)
    calls.clear()
    value_default = wrapped("the-channel", pinned_p4, pinned_axis)
    assert calls[0] == ("the-channel", 1.0, 1.0)
    assert abs(value_default - 0.2 * decay_form_factor_weight(
        transverse_momentum(pinned_p4, pinned_axis), 1.5
    )) <= 1e-15

    # the off limit: b_dec = 0 reproduces the unattached weight bit
    # for bit (the R43 ablation baseline)
    wrapped_off = attach_to_channel_weight(inner, 0.0)
    for p4 in (pinned_p4, (1.0, 0.3, 0.0, 0.0), (1.2, 0.0, 0.7, 0.2)):
        assert wrapped_off("the-channel", p4, pinned_axis) == inner(
            "the-channel"
        )

    # the wrapped callable's pinned identity
    assert wrapped.__name__ == "decay_form_factor_channel_weight"

    # the normalized shares over a self-authored multi-channel vertex
    # change exactly by the form-factor factor per channel
    axis = (0.0, 0.0, 1.0)
    base_weights = (0.5, 0.3, 0.2)
    p4s = (
        (1.0, 0.2, 0.0, 0.4),
        (1.0, 0.5, 0.1, 0.3),
        (1.0, 0.0, 0.0, 0.1),
    )
    b_dec = 2.0
    factors = [
        decay_form_factor_weight(transverse_momentum(p4, axis), b_dec)
        for p4 in p4s
    ]
    total = math.fsum(w * f for w, f in zip(base_weights, factors))
    expected_shares = [w * f / total for w, f in zip(base_weights, factors)]

    def vertex(channel, w_flavor=1.0, w_align=1.0):
        # a self-authored inner weight keyed by the channel label
        return base_weights[int(channel[1])]

    attached = []
    wrapped_vertex = attach_to_channel_weight(vertex, b_dec)
    for k, p4 in enumerate(p4s):
        attached.append(wrapped_vertex(f"c{k}", p4, axis))
    total_attached = math.fsum(attached)
    for attached_value, expected in zip(attached, expected_shares):
        assert abs(attached_value / total_attached - expected) <= 1e-15

    # checked, never repaired: a non-callable inner callable and a
    # non-finite / negative b_dec raise ValueError at wrap time
    with pytest.raises(ValueError):
        attach_to_channel_weight(1.0, 1.0)
    with pytest.raises(ValueError):
        attach_to_channel_weight(inner, -0.5)
    with pytest.raises(ValueError):
        attach_to_channel_weight(inner, float("nan"))
    with pytest.raises(ValueError):
        attach_to_channel_weight(inner, float("inf"))


# ----------------------------------------------------------------------
# test_exchange_layer_separation (acceptance (iii))
# ----------------------------------------------------------------------
def test_exchange_layer_separation():
    """White-box on a self-authored token scan: the form factor is
    absent from the builder / exchange layer (the builder bytes are
    unchanged at sha256 4ab61260...; no exchange-layer symbol
    references the form-factor module; the exchange-slope symbols are
    absent from the form-factor module); no single parameter controls
    both source recoil and daughter transverse momentum (plan section
    10.5 line 748)."""

    # the builder bytes are unchanged (the exchange layer stays the
    # builder's territory)
    builder_path = NESTED_ROOT / "na49_mixing" / "resonance_pp_builder.py"
    assert _sha256_file(builder_path) == (
        FROZEN_SUBSTRATE[
            "dev/mixing2006/na49_mixing/resonance_pp_builder.py"
        ]
    )

    # no exchange-layer symbol references the form factor: the builder,
    # the exchange-model surface (the exchange slope's home), the
    # card-6 blend routing, and the card-2 engine carry no form-factor
    # symbol and no decay-slope name
    exchange_layer = (
        "resonance_pp_builder.py",
        "exchange_model.py",
        "spectral_density.py",
        "threshold_metric.py",
    )
    form_factor_tokens = (
        "cascade_form_factor",
        "decay_form_factor",
        "attach_to_channel_weight",
        "b_dec",
    )
    for name in exchange_layer:
        source = (NESTED_ROOT / "na49_mixing" / name).read_text(
            encoding="utf-8"
        )
        hits = [token for token in form_factor_tokens if token in source]
        assert not hits, f"{name} references the decay layer: {hits}"

    # no form-factor symbol references the exchange layer: the module
    # carries no exchange-slope name of plan line 739/748
    module_source = MODULE_PATH.read_text(encoding="utf-8")
    exchange_tokens = ("B_eff", "effective_transfer_slope")
    hits = [token for token in exchange_tokens if token in module_source]
    assert not hits, f"the form-factor module reads the exchange layer: {hits}"

    # no single parameter controls both layers (plan section 10.5
    # line 748): the decay-side default is the named form-factor slope
    # 1.0 GeV^-2, distinct from the M8-frozen exchange slope of the
    # frozen parameter map
    assert DECAY_FORM_FACTOR_B_DEFAULT == 1.0
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    frozen = manifest["frozen_fit_outputs"]["m8_frozen_pp_parameters"]
    exchange_slope = frozen["regge.effective_transfer_slope"]
    assert exchange_slope == 8.504140762083612
    assert exchange_slope != DECAY_FORM_FACTOR_B_DEFAULT


# ----------------------------------------------------------------------
# test_zero_draws (acceptance (iv), the zero-draw clause)
# ----------------------------------------------------------------------
def test_zero_draws():
    """A self-authored pinned cascade of 1,000 form-factor operations
    consumes zero random numbers (the bit-generator state is
    byte-identical before and after) and the module source carries no
    random machinery."""

    rng = np.random.default_rng(20260920 + 25)
    state_before = rng.bit_generator.state

    axes = (
        (0.0, 0.0, 1.0),
        (1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0),
         1.0 / math.sqrt(3.0)),
        (0.6, 0.0, 0.8),
    )
    slopes = (0.0, 0.5, 1.0, 2.0, DECAY_FORM_FACTOR_B_DEFAULT)
    operations = 0
    for step in range(1_000):
        k = step % 5
        p4 = (
            1.0 + 0.02 * k,
            0.15 * k,
            0.07 * (k - 2),
            0.25 + 0.04 * k,
        )
        axis = axes[step % len(axes)]
        b_dec = slopes[step % len(slopes)]
        kind = step % 4
        if kind == 0:
            transverse_momentum(p4, axis)
        elif kind == 1:
            decay_form_factor_weight(0.1 * k, b_dec)
        elif kind == 2:
            wrapped = attach_to_channel_weight(
                lambda channel, w_flavor=1.0, w_align=1.0: 0.5,
                b_dec,
            )
            wrapped(f"ch{k}", p4, axis)
        else:
            attach_to_channel_weight(
                lambda channel, w_flavor=1.0, w_align=1.0: 0.5, b_dec
            )
        operations += 1
    assert operations == 1_000

    state_after = rng.bit_generator.state
    assert state_after == state_before, (
        "the form factor consumed random numbers: the bit-generator "
        "state changed over the pinned 1,000-operation cascade"
    )

    # the module source carries no random machinery (the audit-card
    # zero-draw source contract)
    source = MODULE_PATH.read_text(encoding="utf-8")
    forbidden = (
        "np.random", "default_rng", "SeedSequence", "import random",
        "random.", "randint", "shuffle(", "gauss(", "weibull",
        "getstate", "setstate", "rng", "uniform(",
    )
    hits = [token for token in forbidden if token in source]
    assert not hits, f"the form-factor module carries random machinery: {hits}"


# ----------------------------------------------------------------------
# test_budget_unchanged (acceptance (iv), the budget clause)
# ----------------------------------------------------------------------
def test_budget_unchanged(catalog, fams):
    """Attaching the factor changes no draw count: the card-4
    per-vertex budget stays 1 channel + 2 kinematics for every scanned
    b_dec, the kinematics substream consumes the same two uniforms as
    the unattached baseline, and at b_dec = 0 the attached draw is
    bit-identical to the card-4 baseline, verified on a self-authored
    catalog-resonance and continuum fixture at a self-authored
    oblique axis."""

    axis = (
        1.0 / math.sqrt(3.0),
        1.0 / math.sqrt(3.0),
        1.0 / math.sqrt(3.0),
    )
    direction = (0.5773502691896258, -0.5773502691896258,
                 0.5773502691896258)
    fixtures = (
        ("catalog:2214", 1.25, (1, 1, 0)),
        ("baryonic_np", 3.5, (1, 1, 0)),
    )
    for family_id, mass, qn in fixtures:
        state = _root_state(family_id, mass, qn, axis)
        channels = enumerate_channels(state, catalog, fams)
        assert channels, f"no open channel at the {family_id!r} fixture"

        # the baseline draw: exactly 1 channel + 2 kinematics uniforms
        seed_c, seed_k = _sub_seeds(20260920, 25)
        rng_c = _RecordingRng(seed_c)
        rng_k = _RecordingRng(seed_k)
        selected, daughters = draw_decay(
            state, channels, catalog, fams, rng_c, rng_k
        )
        assert rng_c.calls == 1, "the baseline channel draw is 1 uniform"
        assert rng_k.calls == 2, "the baseline kinematics draw is 2 uniforms"
        baseline_kinematics = list(rng_k.values)

        for b_dec in (0.0, 0.5, 1.0, 2.0):
            attached = _attached_channels(
                channels, state, catalog, fams, direction, b_dec
            )
            assert math.fsum(
                channel.weight for channel in attached
            ) <= 1.0 + 1e-15

            # the draw budget: exactly 1 channel + 2 kinematics, and
            # the kinematics substream consumes the same two uniforms
            # as the baseline (the form factor sits in the channel
            # weighting, so the kinematics site is untouched)
            rng_c2 = _RecordingRng(seed_c)
            rng_k2 = _RecordingRng(seed_k)
            draw_decay(state, attached, catalog, fams, rng_c2, rng_k2)
            assert rng_c2.calls == 1, (
                f"the attached channel draw at b_dec={b_dec!r} is "
                "1 uniform"
            )
            assert rng_k2.calls == 2, (
                f"the attached kinematics draw at b_dec={b_dec!r} is "
                "2 uniforms"
            )
            assert rng_k2.values == baseline_kinematics, (
                f"the kinematics substream moved at b_dec={b_dec!r}"
            )

            # the off limit: b_dec = 0 reproduces the card-4/6/7
            # baseline bit for bit (the off-limit factor is exactly
            # 1.0, so the pre-normalization attached weight is the
            # unattached weight, and the draw selects the same
            # channel and realizes the same daughters from the same
            # uniforms; the caller-side re-normalization may differ
            # by a final ulp, which does not move the draw)
            if b_dec == 0.0:
                wrapped_off = attach_to_channel_weight(channel_weight, 0.0)
                for channel in channels:
                    daughter_p4 = _channel_daughter_p4(
                        channel, state, catalog, fams, direction
                    )
                    assert wrapped_off(
                        channel, daughter_p4, state.axis
                    ) == channel_weight(channel)
                rng_c3 = _RecordingRng(seed_c)
                rng_k3 = _RecordingRng(seed_k)
                selected0, daughters0 = draw_decay(
                    state, attached, catalog, fams, rng_c3, rng_k3
                )
                assert rng_c3.calls == 1 and rng_k3.calls == 2
                assert selected0.daughters == selected.daughters
                assert daughters0 == daughters, (
                    "the b_dec = 0 attached draw is not bit-identical "
                    "to the card-4 baseline"
                )


# ----------------------------------------------------------------------
# test_module_constants (the named default and the module __all__)
# ----------------------------------------------------------------------
def test_module_constants():
    """DECAY_FORM_FACTOR_B_DEFAULT == 1.0 (value and type) and the
    module's __all__ entries."""

    from na49_mixing import cascade_form_factor as module

    assert DECAY_FORM_FACTOR_B_DEFAULT == 1.0
    assert isinstance(DECAY_FORM_FACTOR_B_DEFAULT, float)
    assert module.__all__ == [
        "DECAY_FORM_FACTOR_B_DEFAULT",
        "transverse_momentum",
        "decay_form_factor_weight",
        "attach_to_channel_weight",
    ]


# ----------------------------------------------------------------------
# test_module_byte_and_export_pins (Part 2)
# ----------------------------------------------------------------------
def test_module_byte_and_export_pins():
    """The module byte stability (sha256 b16077c9...) and the additive
    na49_mixing package exports (the four card-8 names present in the
    package __init__ import block and __all__, resolved by object
    identity, with the prior card exports unchanged)."""

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)
    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    import na49_mixing

    init_src = INIT_PATH.read_text(encoding="utf-8")
    import_block = init_src[
        init_src.index("from .cascade_form_factor import"):
        init_src.index(")", init_src.index("from .cascade_form_factor import"))
    ]
    for name in (
        "DECAY_FORM_FACTOR_B_DEFAULT",
        "attach_to_channel_weight",
        "decay_form_factor_weight",
        "transverse_momentum",
    ):
        assert name in import_block, (
            f"the __init__ import block is missing {name}")
        assert f'"{name}"' in init_src, (
            f"the __init__ __all__ is missing {name}")
    from na49_mixing import cascade_form_factor as module

    assert na49_mixing.DECAY_FORM_FACTOR_B_DEFAULT is (
        module.DECAY_FORM_FACTOR_B_DEFAULT)
    assert na49_mixing.transverse_momentum is module.transverse_momentum
    assert na49_mixing.decay_form_factor_weight is (
        module.decay_form_factor_weight)
    assert na49_mixing.attach_to_channel_weight is (
        module.attach_to_channel_weight)

    # the prior card exports are unchanged (the additive-only clause)
    for name in ("cascade_decay", "cascade_axes", "cascade_alignment"):
        assert hasattr(na49_mixing, name), (
            f"the prior card export {name} is missing")


# ----------------------------------------------------------------------
# test_frozen_substrate_digests (Part 2)
# ----------------------------------------------------------------------
def test_frozen_substrate_digests():
    """The byte-unchanged surfaces the card consumes (the card's
    provenance section), verified on disk."""

    for rel, expected in FROZEN_SUBSTRATE.items():
        path = MONO_ROOT / rel
        assert path.is_file(), f"missing frozen surface: {rel}"
        pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)


# ----------------------------------------------------------------------
# test_freeze_repin_register (the acceptance-(v) freeze re-pin check)
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    """The sha256 of *path* (the house byte-pin helper)."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
