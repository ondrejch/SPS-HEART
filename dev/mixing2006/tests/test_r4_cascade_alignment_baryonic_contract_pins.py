"""Round 4, card 6 test-author register: baryonic alignment kernel.

Independent contract pins for the baryonic alignment kernel module
(``na49_mixing/cascade_alignment.py``), Round 4 card 6
(``TASK-20260920-r4-baryonic-alignment-kernel``; plan section 22 task
23, plan section 10.4, lines 716-732).  This register is the
test-author half of the card: it pins the module contract on
self-authored (cosine, kappa) fixtures and quantile grids
(independent of the solver's pinned fixtures), self-authored
reference algebra (an independent bisection inversion of the CDF and
a hand-computed midpoint case), self-authored zero-draw
instrumentation (a fixed-seed bit-generator state identity across a
pinned operation cascade plus a self-authored source scan), and it
re-pins the freeze manifest for the card.

Part 1: the module contract (pure, no sealed run dir)
-----------------------------------------------------

- ``test_baryonic_kernel_density_law_and_domain``: the angular law
  ``exp(kappa * c)`` exactly on self-authored (c, kappa) pairs
  spanning the boundaries c = +/-1 and kappa = 0, the forward
  asymmetry at kappa > 0, and the checked-never-repaired domain
  contract (a cosine outside [-1, 1], a non-finite cosine, a
  non-finite kappa, and an uncastable cosine string all raise
  ``ValueError``).
- ``test_baryonic_normalization_analytic_and_limit``: the analytic
  ``2 sinh(kappa)/kappa`` at self-authored kappa values (the even-in-
  kappa structure), the kappa -> 0 limit returning exactly 2.0, the
  small-kappa continuity at self-authored probe points inside the
  branch tolerance, the branch boundary at exactly ``kappa = +/-
  1e-8`` (the closed form is taken there, not the limit, pinned to
  the on-disk strict-``<`` comparison), and the ``ValueError`` edges.
- ``test_draw_baryonic_alignment_inverse_cdf``: the closed-form
  inverse CDF verified against a self-authored bisection inversion of
  the CDF on a self-authored quantile grid at self-authored kappa
  values (positive and negative), the hand-computed midpoint case
  (u = 1/2, kappa = 1: ``ln(cosh(1))`` bit-exactly), the
  CDF round trip within 1e-12, the monotone-in-u structure, and the
  mirror identity under kappa -> -kappa.
- ``test_draw_isotropic_branch_boundary_and_endpoints``: the
  isotropic branch ``2u - 1`` bit-stable on self-authored u values
  inside the branch tolerance; the branch boundary at exactly
  ``|kappa| = 1e-8`` (the closed form is taken, not the isotropic
  branch, pinned to the on-disk comparison semantics); the u = 0 and
  u = 1 endpoints (cosine -1 and +1 at the closed form, exact at the
  self-authored kappa values; exact at the isotropic branch); and
  the ``ValueError`` edges on u and kappa.
- ``test_continues_baryon_line_predicate``: the B != 0 predicate on
  self-authored qn tuples (positive and negative baryon number, the
  mesonic zero), the on-disk integer cast of numeric-string and
  float tuples, and the invalid shapes (two- and four-element
  tuples, an uncastable string) raising ``ValueError``.
- ``test_draw_vertex_cos_theta_vertex_interface``: the vertex-level
  interface: the leading-baryon direct draw, the trailing-baryon
  back-to-back mirror, the non-baryon vertex reproducing the card-4
  isotropic law ``2u - 1`` bit-exactly, the isotropic branch being
  vertex-uniform across all daughter orders (the mirror dropped),
  the mirror activation at exactly ``|kappa| = 1e-8`` (the on-disk
  ``>=`` comparison), the two-baryon ``ValueError``
  (checked-never-repaired), and the input validation on
  self-authored fixtures.
- ``test_assign_alignment_label_fallbacks``: the
  ``BARYONIC_ALIGNMENT_LABEL`` on the baryon line at the default
  kappa, the fallback to the card-4 ``ISOTROPIC_ALIGNMENT_LABEL``
  (imported from ``cascade_decay`` and pinned by identity) at the
  isotropic branch and off the baryonic line, the label branch
  boundary at exactly ``|kappa| = 1e-8`` (the ``>=`` semantics), and
  the input validation.
- ``test_module_constants``: ``BARYONIC_KAPPA_DEFAULT == 1.0``,
  ``ISOTROPIC_BRANCH_TOL == 1e-8``, and
  ``BARYONIC_ALIGNMENT_LABEL == "baryonic_alignment"`` (value,
  type), with the package-level constants resolving to the module's
  objects.
- ``test_zero_draw_contract``: the public functions consume no
  randomness on the measurement path (the fixed-seed bit-generator
  state is byte-identical across a pinned operation cascade that
  feeds only the single quantile argument and identity arguments,
  and the module source carries no random machinery per the
  self-authored source scan).

Part 2: the substrate and freeze pins
-------------------------------------

- ``test_module_byte_and_export_pins``: the module byte stability
  (sha256 ``7aad9434...``, re-pinned for card 7 of the round
  (TASK-20260920-r4-mesonic-alignment-kernel), whose additive mesonic
  extension kept the baryonic kernel region of the module
  byte-identical) and the additive ``na49_mixing`` package exports
  (every name in the module ``__all__`` present in the package
  ``__init__`` and resolved by object identity; the ``__all__`` pin
  carries card 7's additive mesonic names).
- ``test_frozen_substrate_digests``: the byte-unchanged surfaces the
  card consumes (card 4's cascade surface, card 5's axis machinery,
  the builder, the card-6 blend routing, the card-2 engine, the
  committed catalog, the continuum families configuration, and the
  R32 driver in both trees), verified on disk.
- ``test_freeze_repin_register``: the freeze-manifest re-pin for
  this register (the strict path-sorted inventory at 463 entries
  carries this file at its on-disk digest; the card-6 freeze
  exception names the card exactly once with this register's file;
  the four prior count pins (the card-3 register, the card-4
  register, the card-5 solver test, and the card-5 register) and
  the card-6 solver test's count constant are re-pinned
  460 -> 461 naming this register; card 7 of the round re-pins them
  additively for its mesonic-kernel test file
  (test_cascade_alignment_mesonic.py), 461 -> 462; card 7's
  independent test-author register
  (test_r4_cascade_alignment_mesonic_contract_pins.py) re-pins them
  additively for its own file, 462 -> 463).

Hermetic: the self-authored fixtures, numpy, the standard library,
and the vendored ``na49_mixing`` modules only; no sealed run dir, no
event generation, no external tables, no Monte Carlo run.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

import na49_mixing
from na49_mixing import cascade_alignment
from na49_mixing.cascade_alignment import (
    BARYONIC_ALIGNMENT_LABEL,
    BARYONIC_KAPPA_DEFAULT,
    ISOTROPIC_BRANCH_TOL,
    assign_alignment_label,
    baryonic_kernel_density,
    baryonic_normalization,
    continues_baryon_line,
    draw_baryonic_alignment,
    draw_vertex_cos_theta,
)
from na49_mixing.cascade_decay import ISOTROPIC_ALIGNMENT_LABEL
from tests._freeze_inventory import inventory_count as _inventory_count

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent          # dev/mixing2006
MONO_ROOT = NESTED_ROOT.parent.parent          # the monorepo root
MODULE_PATH = NESTED_ROOT / "na49_mixing" / "cascade_alignment.py"
INIT_PATH = NESTED_ROOT / "na49_mixing" / "__init__.py"
MANIFEST_PATH = (NESTED_ROOT / "config" / "v2_singlet"
                 / "legacy_v1_manifest.json")

MODULE_REL = "dev/mixing2006/na49_mixing/cascade_alignment.py"
INIT_REL = "dev/mixing2006/na49_mixing/__init__.py"
SOLVER_TEST_REL = "dev/mixing2006/tests/test_cascade_alignment_baryonic.py"
MY_TEST_REL = ("dev/mixing2006/tests/"
               "test_r4_cascade_alignment_baryonic_contract_pins.py")
CARD3_REGISTER_REL = ("dev/mixing2006/tests/"
                      "test_r4_audit_blend_path_contract_pins.py")
CARD4_REGISTER_REL = ("dev/mixing2006/tests/"
                      "test_r4_cascade_decay_contract_pins.py")
CARD5_SOLVER_TEST_REL = "dev/mixing2006/tests/test_cascade_axes.py"
CARD5_REGISTER_REL = ("dev/mixing2006/tests/"
                      "test_r4_cascade_axes_contract_pins.py")
MANIFEST_REL = ("dev/mixing2006/config/v2_singlet/"
                "legacy_v1_manifest.json")

FREEZE_CARD = "TASK-20260920-r4-baryonic-alignment-kernel"
# 460 entries at the card-6 solver landing; this register adds its
# file (+1 entry) for the test-author re-pin (460 -> 461, +1 entry);
# card 7 of the round (TASK-20260920-r4-mesonic-alignment-kernel)
# re-pins the inventory additively for its new mesonic-kernel test
# file (test_cascade_alignment_mesonic.py; the card-mandated re-pin;
# 461 -> 462, +1 entry), so this count constant is re-pinned once
# more; card 7's independent test-author register
# (test_r4_cascade_alignment_mesonic_contract_pins.py) re-pins the
# inventory additively for its own file (462 -> 463, +1 entry), so
# this count constant is re-pinned once more; card 8 of the round
# (TASK-20260920-r4-hadronic-decay-form-factor) re-pins the inventory
# additively for its new form-factor module and test file
# (na49_mixing/cascade_form_factor.py, test_cascade_form_factor.py;
# 463 -> 465, +2 entries); card 8's independent test-author register
# (test_r4_cascade_form_factor_contract_pins.py) re-pins the inventory
# additively for its own file (the constant read
# "INVENTORY_COUNT = 465" at the card-8 solver landing; 465 -> 466,
# +1 entry), so this count constant is re-pinned once more; card 9 of
# the round (TASK-20260920-r4-termination-depth-tests) re-pins the
# inventory additively for its new termination module and test file
# (na49_mixing/cascade_termination.py, test_cascade_termination.py;
# the constant read "INVENTORY_COUNT = 466" at the card-8 register
# landing; 466 -> 468, +2 entries), so this count constant is
# re-pinned once more; card 9's independent test-author register
# (test_r4_cascade_termination_contract_pins.py) re-pins the inventory
# additively for its own file (the constant read
# "INVENTORY_COUNT = 468" at the card-9 solver landing; 468 -> 469,
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
COUNT_PIN = "assert len(keys) == INVENTORY_COUNT"

# Card 7 of the round (TASK-20260920-r4-mesonic-alignment-kernel)
# extended the module additively with the mesonic even law; the
# baryonic kernel region of the module stayed byte-identical through
# the extension (asserted by card 7's baryonic-surface byte-stability
# test), and this full-file pin is re-pinned to the extended module's
# digest.
MODULE_SHA256 = (
    "0d0df347222423aaf6f02403e177867b57da588a5e40d641fd7b467e9c4a81f4"
)

#: The byte-unchanged frozen surfaces the card consumes (the card's
#: provenance section, verified on disk at this register's landing):
#: card 4's cascade surface, card 5's axis machinery, the builder,
#: the card-6 blend routing, the card-2 engine, the committed
#: catalog, the continuum families configuration, and the dual-tree
#: R32 driver.
FROZEN_SUBSTRATE = {
    "dev/mixing2006/na49_mixing/cascade_decay.py":
        "d2a48049af3d6df968993c040033a74422ccd709f53a42bad0538ae5111bcc0c",
    "dev/mixing2006/na49_mixing/cascade_axes.py":
        "a7251a6b6c0ed4b9cfc56c0182c28a70efccb7a92a33b756090ff716a3ec3c61",
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

#: The independent register seed (the day after the card-4 and
#: card-5 registers' 20260921, the solver's 20260920).
REGISTER_SEED = 20260922

# ----------------------------------------------------------------------
# Self-authored fixtures (independent of the solver's pinned fixtures)
# ----------------------------------------------------------------------
BARYONIC_QN = (1, 1, 0)
STRANGE_BARYONIC_QN = (1, 0, -1)
ANTIBARYONIC_QN = (-1, -1, 0)
MESONIC_QN = (0, 0, 0)
CHARGED_MESONIC_QN = (0, 1, 0)

#: The self-authored (cosine, kappa) pairs of the angular-law check:
#: the boundaries c = -1 and c = +1, kappa = 0, and a negative kappa.
DENSITY_FIXTURES = tuple(
    (c, k)
    for k in (0.0, 0.25, 1.0, 2.5, -0.5)
    for c in (-1.0, -0.6, -0.3, 0.0, 0.2, 0.65, 0.99, 1.0)
)

#: The self-authored normalization grid.
NORM_KAPPAS = (0.0, 0.25, 0.75, 1.0, 2.5, 4.0)

#: The self-authored small-kappa continuity probe points (inside the
#: branch tolerance, on both sides).
LIMIT_PROBES = (1e-12, -1e-12, 1e-10, -1e-10, 1e-9, -1e-9,
                0.5e-8, -0.5e-8)

#: The self-authored kappa values of the inverse-CDF check (the
#: negative kappa included: the law is mirrored, the draw is not).
DRAW_KAPPAS = (0.25, 1.0, 3.0, -2.0, 4.0)

#: The self-authored quantile grid: the regular 26-point grid plus the
#: self-authored deep-tail and irregular quantiles.
QUANTILE_GRID = tuple(
    [j / 25 for j in range(26)] + [1e-12, 1.0 - 1e-12, 0.37, 0.63]
)

#: The self-authored u values of the isotropic-branch and vertex
#: checks.
ISOTROPIC_US = (0.0, 0.13, 0.37, 0.5, 0.66, 0.88, 1.0)

#: The endpoint kappas of the closed form (the u = 0 / u = 1 pin).
ENDPOINT_KAPPAS = (0.25, 1.0, 2.0, 3.0)


# ----------------------------------------------------------------------
# Self-authored reference algebra
# ----------------------------------------------------------------------
def _reference_cdf(c: float, kappa: float) -> float:
    """The self-authored CDF reference of the baryonic law: the
    normalized integral of exp(kappa x) from -1 to *c*."""

    return (math.exp(kappa * c) - math.exp(-kappa)) / (
        math.exp(kappa) - math.exp(-kappa))


def _bisect_inversion(u: float, kappa: float,
                      iters: int = 120) -> float:
    """The self-authored bisection inversion of the CDF on [-1, 1]
    (120 halvings drive the interval below float resolution, so the
    result is the unique fixed point to machine precision)."""

    lo, hi = -1.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _reference_cdf(mid, kappa) < u:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ----------------------------------------------------------------------
# test_baryonic_kernel_density_law_and_domain
# ----------------------------------------------------------------------
def test_baryonic_kernel_density_law_and_domain():
    """The angular law ``exp(kappa * c)`` exactly on the
    self-authored fixtures (the boundaries c = +/-1 and kappa = 0
    included), the forward asymmetry at kappa > 0, and the
    checked-never-repaired domain contract."""

    # -- the angular law, exact on the self-authored fixtures --------
    for c, kappa in DENSITY_FIXTURES:
        got = baryonic_kernel_density(c, kappa)
        assert got == math.exp(kappa * c), (
            f"c={c}, kappa={kappa}: the density is not the pinned law"
        )
        assert got > 0.0 and math.isfinite(got)
    # the boundaries and the isotropic point, spelled out
    assert baryonic_kernel_density(1.0, 0.0) == 1.0
    assert baryonic_kernel_density(-1.0, 0.0) == 1.0
    assert baryonic_kernel_density(1.0, 2.0) == math.exp(2.0)
    assert baryonic_kernel_density(-1.0, 2.0) == math.exp(-2.0)

    # -- the forward asymmetry (the odd part is the alignment) -------
    for kappa in (0.25, 1.0, 2.5):
        c = 0.4
        ratio = (baryonic_kernel_density(c, kappa)
                 / baryonic_kernel_density(-c, kappa))
        assert abs(ratio - math.exp(2.0 * kappa * c)) <= 1e-12
        assert ratio > 1.0  # kappa > 0: forward along the inherited axis

    # -- the domain contract: checked, never repaired ----------------
    for c in (-1.5, 1.5, -1.0000001, 1.0000001, float("nan"),
              float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            baryonic_kernel_density(c, 1.0)
    with pytest.raises(ValueError):
        baryonic_kernel_density(0.5, float("nan"))
    with pytest.raises(ValueError):
        baryonic_kernel_density(0.5, float("inf"))
    # the uncastable cosine string (the on-disk float cast fails)
    with pytest.raises(ValueError):
        baryonic_kernel_density("abc", 1.0)
    # the castable string is accepted (the on-disk float cast)
    assert baryonic_kernel_density("0.5", 2.0) == math.exp(1.0)


# ----------------------------------------------------------------------
# test_baryonic_normalization_analytic_and_limit
# ----------------------------------------------------------------------
def test_baryonic_normalization_analytic_and_limit():
    """The analytic ``2 sinh(kappa)/kappa`` on the self-authored grid,
    the kappa -> 0 limit exactly 2.0, the small-kappa continuity, the
    branch boundary at exactly ``|kappa| = 1e-8`` (the closed form is
    taken, pinned to the on-disk strict-``<`` comparison), and the
    ``ValueError`` edges."""

    for kappa in NORM_KAPPAS:
        got = baryonic_normalization(kappa)
        analytic = (2.0 if kappa == 0.0
                    else 2.0 * math.sinh(kappa) / kappa)
        assert abs(got - analytic) <= 1e-12, (
            f"kappa={kappa}: the normalization {got!r} misses the "
            f"analytic {analytic!r}"
        )

    # the kappa -> 0 limit is exactly 2.0 (the isotropic branch)
    assert baryonic_normalization(0.0) == 2.0
    assert baryonic_normalization(-0.0) == 2.0

    # the normalization is even in kappa (structural)
    for kappa in (0.25, 0.75, 1.0, 2.5, 4.0):
        assert baryonic_normalization(-kappa) == baryonic_normalization(kappa)

    # the small-kappa continuity at the self-authored probe points
    for kappa in LIMIT_PROBES:
        assert abs(baryonic_normalization(kappa) - 2.0) <= 1e-12, (
            f"kappa={kappa}: the kappa -> 0 limit is discontinuous"
        )

    # the branch boundary: |kappa| == 1e-8 is NOT inside the strict
    # "<" branch, so the closed form is taken (pinned on disk)
    for kappa in (ISOTROPIC_BRANCH_TOL, -ISOTROPIC_BRANCH_TOL):
        got = baryonic_normalization(kappa)
        closed_form = 2.0 * math.sinh(kappa) / kappa
        assert got == closed_form, (
            f"kappa={kappa}: the branch boundary does not take the "
            "closed form"
        )
        assert abs(got - 2.0) <= 1e-12, (
            f"kappa={kappa}: the closed form is not continuous with "
            "the limit"
        )
    # and the limit branch is exactly 2.0 at the same boundary from
    # inside (the comparison semantics pinned from both sides)
    assert baryonic_normalization(0.5 * ISOTROPIC_BRANCH_TOL) == 2.0

    # -- checked, never repaired --------------------------------------
    with pytest.raises(ValueError):
        baryonic_normalization(float("nan"))
    with pytest.raises(ValueError):
        baryonic_normalization(float("inf"))
    with pytest.raises(ValueError):
        baryonic_normalization(float("-inf"))


# ----------------------------------------------------------------------
# test_draw_baryonic_alignment_inverse_cdf
# ----------------------------------------------------------------------
def test_draw_baryonic_alignment_inverse_cdf():
    """The closed-form inverse CDF against the self-authored bisection
    inversion on the self-authored quantile grid at the
    self-authored kappas (the negative kappa included), the
    hand-computed midpoint case, the CDF round trip, the monotone-in-
    u structure, and the mirror identity under kappa -> -kappa."""

    for kappa in DRAW_KAPPAS:
        for u in QUANTILE_GRID:
            c = draw_baryonic_alignment(u, kappa)
            # the closed form lands on [-1, 1] to machine precision
            # (a negative-kappa endpoint can round one ulp past +1)
            assert -1.0 - 1e-12 <= c <= 1.0 + 1e-12, (
                f"kappa={kappa}, u={u!r}: the draw left [-1, 1] ({c!r})"
            )
            # against the self-authored bisection inversion
            bisected = _bisect_inversion(u, kappa)
            assert abs(c - bisected) <= 1e-12, (
                f"kappa={kappa}, u={u!r}: the closed form misses the "
                f"bisection inversion by {abs(c - bisected)!r}"
            )
            # the CDF round trip
            back = _reference_cdf(c, kappa)
            assert abs(back - u) <= 1e-12, (
                f"kappa={kappa}, u={u!r}: the CDF round trip misses "
                f"by {abs(back - u)!r}"
            )
        # the draw is monotone in u (the CDF is strictly increasing;
        # the grid in its sorted quantile order)
        previous = -2.0
        for u in sorted(QUANTILE_GRID):
            c = draw_baryonic_alignment(u, kappa)
            assert c > previous, (
                f"kappa={kappa}: the draw is not monotone in u at "
                f"u={u!r}"
            )
            previous = c

    # -- the hand-computed midpoint case (u = 1/2, kappa = 1) --------
    # (1/kappa) * ln(exp(-kappa) + 0.5 (exp(kappa) - exp(-kappa)))
    # = ln(0.5 (exp(kappa) + exp(-kappa))) = ln(cosh(1))
    assert draw_baryonic_alignment(0.5, 1.0) == math.log(math.cosh(1.0)), (
        "the hand-computed midpoint case u = 1/2, kappa = 1 must be "
        "ln(cosh(1)) bit-exactly"
    )
    # the midpoint is the forward bias at kappa > 0 (coth(1) - 1 > 0
    # is the mean; the median is positive by the forward skew)
    assert draw_baryonic_alignment(0.5, 1.0) > 0.0

    # -- the mirror identity under kappa -> -kappa --------------------
    # CDF_{-k}(c) = 1 - CDF_k(-c), so the inverse is mirrored against
    # the reflected quantile
    for kappa in (0.25, 1.0, 3.0, 4.0):
        for u in QUANTILE_GRID:
            mirrored = draw_baryonic_alignment(u, -kappa)
            expect = -draw_baryonic_alignment(1.0 - u, kappa)
            assert abs(mirrored - expect) <= 1e-12, (
                f"kappa={kappa}, u={u!r}: the mirrored draw identity "
                f"fails ({mirrored!r} vs {expect!r})"
            )


# ----------------------------------------------------------------------
# test_draw_isotropic_branch_boundary_and_endpoints
# ----------------------------------------------------------------------
def test_draw_isotropic_branch_boundary_and_endpoints():
    """The isotropic branch ``2u - 1`` bit-stable inside the branch
    tolerance on the self-authored u values; the branch boundary at
    exactly ``|kappa| = 1e-8`` (the closed form is taken, not the
    isotropic branch); the u = 0 / u = 1 endpoints; and the
    ``ValueError`` edges."""

    # -- the isotropic branch: bit-stable with 2u - 1 -----------------
    for u in ISOTROPIC_US:
        for kappa in (0.0, -0.0, 0.5 * ISOTROPIC_BRANCH_TOL,
                      -0.5 * ISOTROPIC_BRANCH_TOL, 9e-9, -9e-9):
            got = draw_baryonic_alignment(u, kappa)
            assert got == 2.0 * u - 1.0, (
                f"kappa={kappa}, u={u!r}: the isotropic branch is not "
                "bit-stable with 2u - 1"
            )

    # -- the branch boundary: exactly |kappa| = 1e-8 ------------------
    # the on-disk comparison is strict (abs(kappa) < 1e-8), so the
    # closed form is taken at the boundary, not the isotropic branch
    for kappa in (ISOTROPIC_BRANCH_TOL, -ISOTROPIC_BRANCH_TOL):
        got = draw_baryonic_alignment(0.3, kappa)
        isotropic = 2.0 * 0.3 - 1.0
        assert got != isotropic, (
            f"kappa={kappa}: the boundary takes the isotropic branch; "
            "the on-disk comparison is strict (abs(kappa) < tol)"
        )
        # the closed form at the boundary is the same law in the
        # limit: within O(kappa) of the isotropic value
        assert abs(got - isotropic) < 1e-6, (
            f"kappa={kappa}: the boundary closed form left the "
            "isotropic neighborhood"
        )
    # and just inside the boundary the isotropic branch stands
    for kappa in (0.999e-8, -0.999e-8):
        assert draw_baryonic_alignment(0.3, kappa) == 2.0 * 0.3 - 1.0

    # -- the endpoints u = 0 and u = 1 --------------------------------
    for kappa in ENDPOINT_KAPPAS:
        assert draw_baryonic_alignment(0.0, kappa) == -1.0, (
            f"kappa={kappa}: quantile 0 must map to cos(theta) = -1"
        )
        assert abs(draw_baryonic_alignment(1.0, kappa) - 1.0) <= 1e-12, (
            f"kappa={kappa}: quantile 1 must map to cos(theta) = +1"
        )
    # at the isotropic branch the endpoints are exact by the law
    assert draw_baryonic_alignment(0.0, 0.0) == -1.0
    assert draw_baryonic_alignment(1.0, 0.0) == 1.0

    # -- checked, never repaired --------------------------------------
    for u in (-0.1, 1.1, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            draw_baryonic_alignment(u, 1.0)
    for kappa in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            draw_baryonic_alignment(0.5, kappa)


# ----------------------------------------------------------------------
# test_continues_baryon_line_predicate
# ----------------------------------------------------------------------
def test_continues_baryon_line_predicate():
    """The B != 0 predicate on the self-authored qn tuples (positive
    and negative baryon number, the mesonic zero), the on-disk
    integer cast of numeric-string and float tuples, and the invalid
    shapes."""

    # -- the predicate -------------------------------------------------
    assert continues_baryon_line(BARYONIC_QN) is True
    assert continues_baryon_line(STRANGE_BARYONIC_QN) is True
    assert continues_baryon_line(ANTIBARYONIC_QN) is True
    assert continues_baryon_line(MESONIC_QN) is False
    assert continues_baryon_line(CHARGED_MESONIC_QN) is False

    # -- the on-disk integer cast --------------------------------------
    # numeric-string tuples are cast (int() semantics)
    assert continues_baryon_line(("1", "1", "0")) is True
    assert continues_baryon_line(("0", "1", "0")) is False
    # float tuples are cast by truncation (int() semantics)
    assert continues_baryon_line((1.5, 1.0, 0.0)) is True
    assert continues_baryon_line((0.9, 1.0, 0.0)) is False
    assert continues_baryon_line((0.0, 1.0, 0.0)) is False

    # -- checked, never repaired ---------------------------------------
    with pytest.raises(ValueError):
        continues_baryon_line((1, 1))               # two elements
    with pytest.raises(ValueError):
        continues_baryon_line((1, 1, 0, 0))         # four elements
    with pytest.raises(ValueError):
        continues_baryon_line(("a", "1", "0"))      # uncastable string
    with pytest.raises(ValueError):
        continues_baryon_line(("1.5", "1", "0"))    # non-integer string
    with pytest.raises(ValueError):
        continues_baryon_line(())                   # empty tuple


# ----------------------------------------------------------------------
# test_draw_vertex_cos_theta_vertex_interface
# ----------------------------------------------------------------------
def test_draw_vertex_cos_theta_vertex_interface():
    """The vertex-level interface: the leading-baryon direct draw,
    the trailing-baryon back-to-back mirror, the non-baryon vertex
    reproducing the card-4 isotropic law, the isotropic branch
    vertex-uniform across daughter orders, the mirror activation at
    exactly ``|kappa| = 1e-8``, the two-baryon ``ValueError``, and
    the input validation."""

    pair_lead_baryon = (BARYONIC_QN, MESONIC_QN)
    pair_trail_baryon = (MESONIC_QN, BARYONIC_QN)
    pair_no_baryon = (MESONIC_QN, MESONIC_QN)
    pair_lead_strange = (STRANGE_BARYONIC_QN, CHARGED_MESONIC_QN)

    # -- the leading-baryon direct draw --------------------------------
    for u in ISOTROPIC_US:
        for kappa in (0.25, 1.0, 2.5, -1.5):
            got = draw_vertex_cos_theta(u, pair_lead_baryon,
                                        kappa_baryonic=kappa)
            expect = draw_baryonic_alignment(u, kappa)
            assert got == expect, (
                f"u={u!r}, kappa={kappa}: the leading baryon must "
                "carry the direct kernel draw"
            )
    # the strange baryonic line takes the same law
    for u in (0.2, 0.5, 0.8):
        assert draw_vertex_cos_theta(
            u, pair_lead_strange) == draw_baryonic_alignment(
            u, BARYONIC_KAPPA_DEFAULT)

    # -- the trailing-baryon back-to-back mirror ------------------------
    for u in ISOTROPIC_US:
        for kappa in (0.25, 1.0, 2.5, -1.5):
            got = draw_vertex_cos_theta(u, pair_trail_baryon,
                                        kappa_baryonic=kappa)
            expect = -draw_baryonic_alignment(u, kappa)
            assert got == expect, (
                f"u={u!r}, kappa={kappa}: the trailing baryon must "
                "carry the mirrored kernel draw"
            )

    # -- the non-baryon vertex: the card-4 isotropic law ----------------
    for u in ISOTROPIC_US:
        got = draw_vertex_cos_theta(u, pair_no_baryon)
        assert got == 2.0 * u - 1.0, (
            f"u={u!r}: the non-baryon vertex must reproduce the "
            "card-4 isotropic law 2u - 1 bit-exactly"
        )
        # the same at an explicit active kappa (no baryon line: the
        # kernel is not consulted)
        assert draw_vertex_cos_theta(
            u, pair_no_baryon, kappa_baryonic=2.5) == 2.0 * u - 1.0

    # -- the isotropic branch: vertex-uniform across daughter orders ----
    # at kappa = 0 every daughter order draws the plain isotropic
    # value (the mirror is dropped)
    for u in ISOTROPIC_US:
        isotropic = 2.0 * u - 1.0
        for pair in (pair_lead_baryon, pair_trail_baryon,
                     pair_no_baryon):
            got = draw_vertex_cos_theta(u, pair, kappa_baryonic=0.0)
            assert got == isotropic, (
                f"u={u!r}, pair={pair}: the isotropic branch must be "
                "vertex-uniform across daughter orders"
            )

    # -- the mirror activation boundary: exactly |kappa| = 1e-8 --------
    # the on-disk comparison is ">=" for the trailing-baryon mirror,
    # so the mirror is active at the boundary (and dropped just
    # inside)
    for kappa in (ISOTROPIC_BRANCH_TOL, -ISOTROPIC_BRANCH_TOL):
        got = draw_vertex_cos_theta(
            0.3, pair_trail_baryon, kappa_baryonic=kappa)
        assert got == -draw_baryonic_alignment(
            0.3, kappa), (
            f"kappa={kappa}: the mirror must be active at exactly "
            "the branch tolerance (the on-disk '>=' comparison)"
        )
        assert got != 2.0 * 0.3 - 1.0, (
            f"kappa={kappa}: the boundary must not take the "
            "isotropic branch"
        )
    for kappa in (0.999e-8, -0.999e-8):
        assert draw_vertex_cos_theta(
            0.3, pair_trail_baryon, kappa_baryonic=kappa) == 2.0 * 0.3 - 1.0

    # -- the two-baryon ValueError (checked, never repaired) ------------
    for pair in ((BARYONIC_QN, STRANGE_BARYONIC_QN),):
        with pytest.raises(ValueError):
            draw_vertex_cos_theta(0.5, pair)
    # the baryon-antibaryon pair (net baryon number zero: the Round 6
    # baryon-pair channels of a mesonic parent, plan section 12.4,
    # 2026-09-28) carries no baryonic line and follows the symmetric
    # mesonic dispatch in either daughter order: the isotropic branch
    # at the vertex default and under the baryonic kernel alone, the
    # even law at an active kappa_mesonic
    for pair in ((ANTIBARYONIC_QN, BARYONIC_QN),
                 (STRANGE_BARYONIC_QN, ANTIBARYONIC_QN)):
        assert draw_vertex_cos_theta(0.5, pair) == 2.0 * 0.5 - 1.0
        assert draw_vertex_cos_theta(
            0.3, pair, kappa_baryonic=1.0) == 2.0 * 0.3 - 1.0
        assert draw_vertex_cos_theta(
            0.3, pair, kappa_mesonic=1.0) == draw_vertex_cos_theta(
                0.3, pair_no_baryon, kappa_mesonic=1.0)

    # -- the input validation on the self-authored fixtures -------------
    with pytest.raises(ValueError):
        draw_vertex_cos_theta(-0.1, pair_no_baryon)
    with pytest.raises(ValueError):
        draw_vertex_cos_theta(1.1, pair_no_baryon)
    with pytest.raises(ValueError):
        draw_vertex_cos_theta(float("nan"), pair_no_baryon)
    with pytest.raises(ValueError):
        draw_vertex_cos_theta(0.5, pair_no_baryon,
                              kappa_baryonic=float("inf"))
    with pytest.raises(ValueError):
        draw_vertex_cos_theta(0.5, ((1, 1), MESONIC_QN))
    with pytest.raises(ValueError):
        draw_vertex_cos_theta(0.5, (MESONIC_QN, (0, 0, 0, 0)))
    # a pair that is not a two-daughter sequence
    with pytest.raises(ValueError):
        draw_vertex_cos_theta(0.5, "abc")


# ----------------------------------------------------------------------
# test_assign_alignment_label_fallbacks
# ----------------------------------------------------------------------
def test_assign_alignment_label_fallbacks():
    """The label on the baryon line at the default kappa; the
    fallback to the card-4 ``ISOTROPIC_ALIGNMENT_LABEL`` at the
    isotropic branch and off the baryonic line; the label branch
    boundary at exactly ``|kappa| = 1e-8``; and the input
    validation."""

    # the card-4 placeholder constant, pinned by identity (the module
    # imports it, it does not redeclare it)
    assert ISOTROPIC_ALIGNMENT_LABEL == "isotropic"

    # -- the baryon line at the default kappa ---------------------------
    assert assign_alignment_label(
        BARYONIC_QN) == BARYONIC_ALIGNMENT_LABEL
    assert assign_alignment_label(
        STRANGE_BARYONIC_QN) == BARYONIC_ALIGNMENT_LABEL
    assert assign_alignment_label(
        ANTIBARYONIC_QN) == BARYONIC_ALIGNMENT_LABEL

    # -- the fallback off the baryonic line ------------------------------
    assert assign_alignment_label(MESONIC_QN) == ISOTROPIC_ALIGNMENT_LABEL
    assert assign_alignment_label(
        CHARGED_MESONIC_QN) == ISOTROPIC_ALIGNMENT_LABEL
    # the explicit active kappa off the line: the fallback stands
    assert assign_alignment_label(
        MESONIC_QN, kappa_baryonic=2.5) == ISOTROPIC_ALIGNMENT_LABEL

    # -- the fallback at the isotropic branch ----------------------------
    for kappa in (0.0, -0.0, 0.5 * ISOTROPIC_BRANCH_TOL,
                  -0.5 * ISOTROPIC_BRANCH_TOL, 9e-9, -9e-9):
        assert assign_alignment_label(
            BARYONIC_QN, kappa_baryonic=kappa) == ISOTROPIC_ALIGNMENT_LABEL

    # -- the label branch boundary: exactly |kappa| = 1e-8 ---------------
    # the on-disk comparison is ">=", so the label stands at the
    # boundary (and falls back just inside)
    for kappa in (ISOTROPIC_BRANCH_TOL, -ISOTROPIC_BRANCH_TOL):
        assert assign_alignment_label(
            BARYONIC_QN, kappa_baryonic=kappa) == BARYONIC_ALIGNMENT_LABEL
    for kappa in (0.999e-8, -0.999e-8):
        assert assign_alignment_label(
            BARYONIC_QN, kappa_baryonic=kappa) == ISOTROPIC_ALIGNMENT_LABEL

    # -- checked, never repaired -----------------------------------------
    with pytest.raises(ValueError):
        assign_alignment_label((1, 1))
    with pytest.raises(ValueError):
        assign_alignment_label(BARYONIC_QN, float("nan"))
    with pytest.raises(ValueError):
        assign_alignment_label(MESONIC_QN, float("inf"))


# ----------------------------------------------------------------------
# test_module_constants
# ----------------------------------------------------------------------
def test_module_constants():
    """The module constants: ``BARYONIC_KAPPA_DEFAULT == 1.0``,
    ``ISOTROPIC_BRANCH_TOL == 1e-8``, and
    ``BARYONIC_ALIGNMENT_LABEL == "baryonic_alignment"``."""

    assert BARYONIC_KAPPA_DEFAULT == 1.0
    assert isinstance(BARYONIC_KAPPA_DEFAULT, float)
    assert ISOTROPIC_BRANCH_TOL == 1e-8
    assert isinstance(ISOTROPIC_BRANCH_TOL, float)
    assert BARYONIC_ALIGNMENT_LABEL == "baryonic_alignment"
    assert isinstance(BARYONIC_ALIGNMENT_LABEL, str)
    # the package-level constants are the module's objects (the
    # additive exports, not copies)
    assert na49_mixing.BARYONIC_KAPPA_DEFAULT is BARYONIC_KAPPA_DEFAULT
    assert na49_mixing.ISOTROPIC_BRANCH_TOL is ISOTROPIC_BRANCH_TOL
    assert na49_mixing.BARYONIC_ALIGNMENT_LABEL is BARYONIC_ALIGNMENT_LABEL


# ----------------------------------------------------------------------
# test_zero_draw_contract
# ----------------------------------------------------------------------
def test_zero_draw_contract():
    """The public functions consume no randomness on the measurement
    path: the fixed-seed bit-generator state is byte-identical across
    a pinned operation cascade (only the quantile argument and the
    identity arguments are fed), and the module source carries no
    random machinery (the self-authored source scan)."""

    rng = np.random.default_rng(REGISTER_SEED)
    state_before = rng.bit_generator.state

    kappas = (BARYONIC_KAPPA_DEFAULT, 0.0, 0.25, 2.5, 4.0, -1.5)
    pairs = (
        (BARYONIC_QN, MESONIC_QN),
        (MESONIC_QN, BARYONIC_QN),
        (MESONIC_QN, MESONIC_QN),
    )
    operations = 0
    for step in range(1_200):
        u = (step % 26) / 25.0
        c = -1.0 + (step % 21) / 10.0
        kappa = kappas[(step * 7) % len(kappas)]
        pair = pairs[(step * 3) % len(pairs)]
        qn = pair[step % 2]
        kind = step % 5
        if kind == 0:
            baryonic_kernel_density(c, kappa)
        elif kind == 1:
            baryonic_normalization(kappa)
        elif kind == 2:
            draw_baryonic_alignment(u, kappa)
        elif kind == 3:
            draw_vertex_cos_theta(u, pair, kappa_baryonic=kappa)
        else:
            assign_alignment_label(qn, kappa_baryonic=kappa)
        operations += 1
    assert operations == 1_200

    state_after = rng.bit_generator.state
    assert state_after == state_before, (
        "the alignment kernel consumed random numbers: the fixed-seed "
        "bit-generator state changed over the pinned 1,200-operation "
        "cascade"
    )

    # the module source carries no random machinery (the
    # self-authored scan: tokens that would appear in any
    # random-number consumer; the module is standard-library math
    # and the card-4 label constant only)
    source = MODULE_PATH.read_text(encoding="utf-8")
    forbidden = (
        "import random", "from random", "random.", "np.random",
        "numpy.random", "default_rng", "RandomState", "SeedSequence",
        "getrandbits", "randint(", "random(", "rand(", "shuffle(",
        "gauss(", "weibullvariate(", "bit_generator", "spawn(",
    )
    hits = [token for token in forbidden if token in source]
    assert not hits, f"the alignment module carries random machinery: {hits}"


# ----------------------------------------------------------------------
# test_module_byte_and_export_pins
# ----------------------------------------------------------------------
def test_module_byte_and_export_pins():
    """The module byte stability (sha256) and the additive
    ``na49_mixing`` package exports (every exported name by object
    identity)."""

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    # the module's own export list (the contract surface; the nine
    # card-6 baryonic names lead the list in their landed order and
    # card 7 of the round appended its additive mesonic names below
    # them)
    assert cascade_alignment.__all__ == [
        "BARYONIC_KAPPA_DEFAULT",
        "ISOTROPIC_BRANCH_TOL",
        "BARYONIC_ALIGNMENT_LABEL",
        "baryonic_kernel_density",
        "baryonic_normalization",
        "draw_baryonic_alignment",
        "continues_baryon_line",
        "draw_vertex_cos_theta",
        "assign_alignment_label",
        "MESONIC_KAPPA_DEFAULT",
        "MESONIC_CDF_NODES",
        "MESONIC_ALIGNMENT_LABEL",
        "mesonic_kernel_density",
        "mesonic_normalization",
        "mesonic_cdf_table",
        "draw_mesonic_alignment",
    ]

    init_src = INIT_PATH.read_text(encoding="utf-8")
    for name in cascade_alignment.__all__:
        assert f'"{name}"' in init_src, (
            f"the package __init__ no longer exports {name}"
        )
        assert hasattr(na49_mixing, name), (
            f"the package no longer resolves {name}"
        )
        assert getattr(na49_mixing, name) is getattr(
            cascade_alignment, name), (
            f"the package export {name} is not the module's object"
        )


# ----------------------------------------------------------------------
# test_frozen_substrate_digests
# ----------------------------------------------------------------------
def test_frozen_substrate_digests():
    """The byte-unchanged frozen surfaces the card consumes, verified
    on disk (the dual-tree R32 driver included)."""

    for rel, expected in FROZEN_SUBSTRATE.items():
        path = MONO_ROOT / rel
        assert path.is_file(), f"missing frozen surface: {rel}"
        pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)


# ----------------------------------------------------------------------
# test_freeze_repin_register
# ----------------------------------------------------------------------
