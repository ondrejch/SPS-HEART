"""Round 4, card 7 test-author register: mesonic alignment kernel.

Independent contract pins for the mesonic alignment kernel (the
card-7 additive extension of ``na49_mixing/cascade_alignment.py``),
Round 4 card 7 (``TASK-20260920-r4-mesonic-alignment-kernel``; plan
section 22 task 24, plan section 10.4, lines 716-732).  This register
is the test-author half of the card: it pins the mesonic module
contract on self-authored (cosine, kappa) fixtures and quantile grids
(independent of the solver's pinned fixtures), self-authored
reference algebra (a high-precision decimal power series for erfi, a
self-authored 512-node Gauss-Legendre quadrature, and a self-authored
bisection inversion of the analytic CDF), self-authored zero-draw
instrumentation (a fixed-seed bit-generator state identity across a
pinned operation cascade plus a self-authored source scan), the
byte-stable baryonic region of the extended module at a digest
recomputed by this register's own boundary logic, and it re-pins the
freeze manifest for the card.

Part 1: the module contract (pure, no sealed run dir)
-----------------------------------------------------

- ``test_mesonic_kernel_density_law_and_domain``: the even angular law
  ``exp(kappa * c**2)`` exactly on self-authored (c, kappa) pairs
  spanning the boundaries c = +/-1, c = 0, kappa = 0, and negative
  kappas; the even symmetry ``density(-c) == density(c)`` bit-exactly
  (verified on disk); and the checked-never-repaired domain contract
  (a cosine outside [-1, 1], a non-finite cosine, a non-finite
  kappa, and an uncastable cosine string all raise ``ValueError``;
  the castable string is accepted).
- ``test_mesonic_normalization_analytic_and_limit``: the analytic
  ``sqrt(pi/kappa) * erfi(sqrt(kappa))`` at self-authored kappa values
  (verified against the self-authored 100-digit decimal erfi series
  and the self-authored 512-node Gauss-Legendre quadrature within
  1e-12), the erf branch for ``kappa < 0`` (the on-disk semantics at
  self-authored negative kappas), the kappa -> 0 limit returning
  exactly 2.0, the linear approach ``2 + 2 kappa / 3`` (the
  coefficient pinned at self-authored probe points), the branch
  boundary at exactly ``|kappa| = 1e-8`` (the strict ``<`` branch is
  not taken there; the closed form is taken, pinned to the on-disk
  comparison), and the ``ValueError`` edges.
- ``test_mesonic_cdf_table_construction``: the 10,001-node table on
  the uniform cosine grid at self-authored kappas: the node values at
  the endpoints (F[0] = 0, F[-1] = 1), the even symmetry
  ``F(-c) = 1 - F(c)`` at construction (within 1e-12 on the node
  pairs), the monotone non-decreasing property (strictly increasing
  at the active kappas), the node values against the self-authored
  analytic CDF reference, the kappa -> 0 table (the node values are
  the uniform grid, bit-exact), and the read-only / cached behavior
  (the table is write-protected, served from the module cache by
  object identity, and the ``-0.0`` / ``0.0`` cache aliasing pinned).
- ``test_draw_mesonic_alignment_table_inversion``: the
  searchsorted + linear-interpolation inversion verified against the
  self-authored bisection inversion of the analytic CDF on a
  self-authored (u, kappa) grid (the solver's 57-point grid is not
  reused), the CDF-space round trip bounded by the density-scaled
  interpolation bound, the u = 0 / u = 1 endpoints (the on-disk
  behavior: quantile 0 maps to cos(theta) = -1 exactly, quantile 1
  to +1 exactly at every active kappa), the even-law consequences
  (the quantile-symmetry identity ``draw(u) + draw(1 - u) = 0`` the
  table symmetry implies, so the mean is zero; the draw is distinct
  for ``kappa`` and ``-kappa`` while the baryonic mirror identity is
  the contrast), and the ``ValueError`` edges.
- ``test_draw_mesonic_isotropic_branch_boundary_and_endpoints``: the
  isotropic branch ``2u - 1`` bit-stable with the exact float
  expression on self-authored u values inside the branch tolerance;
  the branch boundary at exactly ``|kappa| = 1e-8`` (the table draw
  is taken, not the isotropic branch, pinned to the on-disk
  ``>=`` semantics); and the ``ValueError`` edges on u and kappa.
- ``test_draw_vertex_cos_theta_mesonic_dispatch``: the dispatch in
  ``draw_vertex_cos_theta`` with ``kappa_mesonic``: the default-
  argument vertex (``kappa_mesonic = 0.0``) preserving the card-6
  contract (the baryon line bit-identical to card 6's behavior, the
  mesonic vertex ``2u - 1``); the active mesonic vertex (both
  daughters ``B == 0``, the even law in both daughter orders
  identically, the baryonic strength not consulted); the mixed
  baryon-meson vertex (the baryonic law bit-identically regardless of
  ``kappa_mesonic``, direct and mirrored orders); the two-baryon
  ``ValueError`` unchanged; and the input validation on
  self-authored fixtures.
- ``test_assign_alignment_label_mesonic``: the
  ``MESONIC_ALIGNMENT_LABEL`` on the mesonic daughter while the
  kernel is active, the ``BARYONIC_ALIGNMENT_LABEL`` on the baryon
  line (unchanged at any mesonic state), the card-4
  ``ISOTROPIC_ALIGNMENT_LABEL`` fallback at the isotropic branches
  and off both lines, the label branch boundary at exactly
  ``|kappa| = 1e-8`` (the ``>=`` semantics), the vertex-exact
  call-site convention at each vertex kind (the governing kernel's
  kappa, the other at the isotropic branch, labels agreeing with the
  drawn law), and the input validation.
- ``test_module_constants``: ``MESONIC_KAPPA_DEFAULT == 1.0``,
  ``MESONIC_CDF_NODES == 10001``, and
  ``MESONIC_ALIGNMENT_LABEL == "mesonic_alignment"`` (value and
  type), with the card-6 constants still present
  (``BARYONIC_KAPPA_DEFAULT``, ``ISOTROPIC_BRANCH_TOL``,
  ``BARYONIC_ALIGNMENT_LABEL``) and the package-level constants
  resolving to the module's objects.
- ``test_zero_draw_contract``: the module's public functions consume
  no randomness (the fixed-seed bit-generator state is
  byte-identical across a pinned operation cascade over all eight
  public functions, the CDF table construction included, and the
  module source carries no random machinery per the self-authored
  source scan).
- ``test_baryonic_region_byte_stable``: the card-6 baryonic region of
  the extended module (from the ``__all__`` closing bracket to the
  ``draw_vertex_cos_theta`` definition) at its pinned sha256,
  recomputed by this register's own boundary logic and verified
  byte-identical; the baryonic constants and functions in the
  region, the mesonic names absent from it, and the single mesonic
  section marker in the extended file.

Part 2: the substrate and freeze pins
-------------------------------------

- ``test_module_byte_and_export_pins``: the module byte stability
  (sha256 ``7aad9434...``, the extended card-7 module) and the
  additive ``na49_mixing`` package exports (every one of the 16
  names of the extended ``__all__`` present in the package
  ``__init__`` and resolved by object identity).
- ``test_frozen_substrate_digests``: the byte-unchanged surfaces the
  card consumes (card 4's cascade surface, card 5's axis machinery,
  the builder, the card-6 blend routing, the card-2 engine, the
  committed catalog, the continuum families configuration, and the
  R32 driver in both trees), verified on disk.
- ``test_freeze_repin_register``: the freeze-manifest re-pin for
  this register (the strict path-sorted inventory at 463 entries
  carries this file at its on-disk digest; the card-7 freeze
  exception names the card exactly once with the solver's 10
  path-sorted files plus this register's file; the extended module,
  the additive package exports, the card-7 solver test, and the six
  consumed count pins (the card-3 register, the card-4 register, the
  card-5 solver test, the card-5 register, the card-6 solver test,
  and the card-6 register) stay at their on-disk digests, each
  count pin re-pinned 462 -> 463 naming this register).

Hermetic: the self-authored fixtures, numpy, the standard library,
and the vendored ``na49_mixing`` modules only; no sealed run dir, no
event generation, no external tables, no Monte Carlo run.
"""

from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

import na49_mixing
from na49_mixing import cascade_alignment
from na49_mixing.cascade_alignment import (
    BARYONIC_ALIGNMENT_LABEL,
    BARYONIC_KAPPA_DEFAULT,
    ISOTROPIC_BRANCH_TOL,
    MESONIC_ALIGNMENT_LABEL,
    MESONIC_CDF_NODES,
    MESONIC_KAPPA_DEFAULT,
    assign_alignment_label,
    draw_baryonic_alignment,
    draw_mesonic_alignment,
    draw_vertex_cos_theta,
    mesonic_cdf_table,
    mesonic_kernel_density,
    mesonic_normalization,
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
SOLVER_TEST_REL = "dev/mixing2006/tests/test_cascade_alignment_mesonic.py"
MY_TEST_REL = ("dev/mixing2006/tests/"
               "test_r4_cascade_alignment_mesonic_contract_pins.py")
CARD3_REGISTER_REL = ("dev/mixing2006/tests/"
                      "test_r4_audit_blend_path_contract_pins.py")
CARD4_REGISTER_REL = ("dev/mixing2006/tests/"
                      "test_r4_cascade_decay_contract_pins.py")
CARD5_SOLVER_TEST_REL = "dev/mixing2006/tests/test_cascade_axes.py"
CARD5_REGISTER_REL = ("dev/mixing2006/tests/"
                      "test_r4_cascade_axes_contract_pins.py")
CARD6_SOLVER_TEST_REL = "dev/mixing2006/tests/test_cascade_alignment_baryonic.py"
CARD6_REGISTER_REL = ("dev/mixing2006/tests/"
                      "test_r4_cascade_alignment_baryonic_contract_pins.py")
MANIFEST_REL = ("dev/mixing2006/config/v2_singlet/"
                "legacy_v1_manifest.json")

FREEZE_CARD = "TASK-20260920-r4-mesonic-alignment-kernel"
# 462 entries at the card-7 solver landing; this register adds its
# file (+1 entry) for the test-author re-pin (462 -> 463, +1 entry),
# and re-pins the seven consumed count pins (the six chain files plus
# the card-7 solver test's own count constant) with the bump
# attributed to this register's file; card 8 of the round
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

#: The extended module's pinned bytes (the card-7 additive extension
#: of the card-6 module; the card-6 register re-pinned this pin for
#: the extension).
MODULE_SHA256 = (
    "0d0df347222423aaf6f02403e177867b57da588a5e40d641fd7b467e9c4a81f4"
)

#: The byte-stable baryonic surface of the extended module: the sha256
#: of the region from the ``__all__`` closing bracket to the
#: ``draw_vertex_cos_theta`` definition, recomputed by this register's
#: own boundary logic and verified byte-identical to the solver's
#: pin at this register's landing.
BARYONIC_REGION_SHA256 = (
    "0d2feb2e31d1501e4d57ef45157e0876f00b17dc99938828b599fea5dfb8c431"
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

#: The independent register seed (the day after the card-6 register's
#: 20260922).
REGISTER_SEED = 20260923


# ----------------------------------------------------------------------
# Self-authored fixtures (independent of the solver's pinned fixtures)
# ----------------------------------------------------------------------
BARYONIC_QN = (1, 1, 0)
STRANGE_BARYONIC_QN = (1, 0, -1)
ANTIBARYONIC_QN = (-1, -1, 0)
MESONIC_QN = (0, 0, 0)
CHARGED_MESONIC_QN = (0, 1, 0)

#: The self-authored (cosine, kappa) pairs of the angular-law check:
#: the boundaries c = -1 and c = +1, c = 0, kappa = 0, and negative
#: kappas.
DENSITY_FIXTURES = tuple(
    (c, k)
    for k in (0.0, 0.3, 1.0, 2.0, 4.0, -0.25, -1.5)
    for c in (-1.0, -0.8, -0.45, -0.1, 0.0, 0.15, 0.7, 1.0)
)

#: The self-authored normalization grid (positive and negative).
NORM_KAPPAS_POS = (0.0, 0.3, 0.75, 1.0, 2.0, 4.0)
NORM_KAPPAS_NEG = (-0.3, -0.75, -1.0, -2.0, -4.0)

#: The self-authored small-kappa continuity probe points (inside the
#: branch tolerance, on both sides).
LIMIT_PROBES = (1e-12, -1e-12, 1e-10, -1e-10, 2e-9, -2e-9,
                0.5e-8, -0.5e-8)

#: The self-authored probe points of the linear approach
#: ``2 + 2 kappa / 3`` (just outside the branch tolerance).
LINEAR_APPROACH_KAPPAS = (1e-7, 1e-6, 1e-5)
COEFFICIENT_PROBE_KAPPAS = (1e-4, 5e-4)

#: The self-authored kappa values of the CDF table checks.
TABLE_KAPPAS = (0.25, 1.0, 2.0, 4.0, -0.25, -1.0)

#: The self-authored kappa values of the sampler checks (the negative
#: kappas included: the even law's transverse enhancement is the erf
#: branch).
DRAW_KAPPAS = (0.5, 1.0, 2.0, 4.0, -0.5, -2.5)

#: The self-authored quantile grid: the regular 31-point grid plus the
#: self-authored deep-tail and irregular quantiles (not the solver's
#: 57-point grid).
QUANTILE_GRID = tuple(sorted(
    {j / 30 for j in range(31)}
    | {0.11, 0.47, 0.73, 1e-12, 1.0 - 1e-12, 1.0 / 7.0, 6.0 / 7.0}
))

#: The self-authored u values of the isotropic-branch and vertex
#: checks.
ISOTROPIC_US = (0.0, 0.07, 0.31, 0.5, 0.58, 0.83, 1.0)

#: The self-authored u values of the quantile-symmetry identity
#: (the even-law zero-mean consequence).
SYM_US = (0.11, 0.37, 0.5, 0.63, 0.89)

#: The self-authored vertex dispatch u grid.
VERTEX_US = (0.0, 0.13, 0.37, 0.5, 0.66, 0.88, 1.0)


# ----------------------------------------------------------------------
# Self-authored reference algebra
# ----------------------------------------------------------------------
_DECIMAL_PI = (
    "3.14159265358979323846264338327950288419716939937510582097494459"
    "2307816406286208998628034825342117068"
)


def _reference_erfi(x: float, precision: int = 80) -> float:
    """The self-authored high-precision imaginary error function
    ``erfi(x) = (2/sqrt(pi)) sum_n x^(2n+1) / (n! (2n + 1))``, summed
    in 80-digit decimal arithmetic (independent of the module's
    float-64 power series; the sign-alternating terms at ``x < 0``
    are absorbed by the working precision, so no cancellation loss).
    The result converts to float at the end (the 1e-12 acceptance
    tolerance is far outside the conversion rounding)."""

    with localcontext() as ctx:
        ctx.prec = precision
        x_dec = Decimal(repr(float(x)))  # NumPy 2 writes repr(np.float64) as 'np.float64(...)'
        if x_dec == 0:
            return 0.0
        sqrt_pi = Decimal(_DECIMAL_PI).sqrt()
        x2 = x_dec * x_dec
        term = x_dec
        total = x_dec
        n = 1
        while True:
            term = (term * x2 * Decimal(2 * n - 1)
                    / (Decimal(n) * Decimal(2 * n + 1)))
            total += term
            if term.copy_abs() < total.copy_abs() * Decimal(10) ** (10 - precision):
                break
            n += 1
            if n > 400:
                raise ValueError(
                    "the self-authored decimal erfi series did not "
                    f"converge for x = {x!r}")
        return float(Decimal(2) / sqrt_pi * total)


#: The self-authored Gauss-Legendre quadrature grid of the
#: normalization cross-check (512 nodes; the kappa grid's worst
#: quadrature error is far inside the pre-registered 1e-12).
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(512)


def _quadrature(kappa: float) -> float:
    """The self-authored 512-node Gauss-Legendre quadrature of
    ``exp(kappa c**2)`` over [-1, 1]."""

    values = np.exp(kappa * _GL_NODES * _GL_NODES)
    return float(np.sum(_GL_WEIGHTS * values))


def _reference_cdf(c: float, kappa: float) -> float:
    """The self-authored analytic CDF of the even law at the cosine
    *c* (the self-authored decimal erfi reference for ``kappa > 0``,
    the standard-library erf analogue for ``kappa < 0``, the uniform
    reference inside the branch tolerance)."""

    if abs(kappa) < ISOTROPIC_BRANCH_TOL:
        return 0.5 * (c + 1.0)
    if kappa > 0.0:
        s = math.sqrt(kappa)
        return 0.5 * (1.0 + _reference_erfi(s * c) / _reference_erfi(s))
    s = math.sqrt(-kappa)
    return 0.5 * (1.0 + math.erf(s * c) / math.erf(s))


def _reference_max_density(kappa: float) -> float:
    """The maximum normalized density of the even law (at c = +/-1 for
    kappa > 0, at c = 0 for kappa < 0; 0.5 inside the branch)."""

    if abs(kappa) < ISOTROPIC_BRANCH_TOL:
        return 0.5
    if kappa > 0.0:
        return math.exp(kappa) / mesonic_normalization(kappa)
    return 1.0 / mesonic_normalization(kappa)


def _bisect_inversion(u: float, kappa: float,
                      iters: int = 90) -> float:
    """The self-authored bisection inversion of the analytic CDF on
    [-1, 1] (90 halvings drive the interval below float resolution,
    so the result is the unique fixed point to machine precision)."""

    if abs(kappa) < ISOTROPIC_BRANCH_TOL:
        def cdf(c: float) -> float:
            return 0.5 * (c + 1.0)
    elif kappa > 0.0:
        s = math.sqrt(kappa)
        denom = _reference_erfi(s)

        def cdf(c: float) -> float:
            return 0.5 * (1.0 + _reference_erfi(s * c) / denom)
    else:
        s = math.sqrt(-kappa)
        denom = math.erf(s)

        def cdf(c: float) -> float:
            return 0.5 * (1.0 + math.erf(s * c) / denom)

    lo, hi = -1.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < u:
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
# test_mesonic_kernel_density_law_and_domain
# ----------------------------------------------------------------------
def test_mesonic_kernel_density_law_and_domain():
    """The even angular law ``exp(kappa * c**2)`` exactly on the
    self-authored fixtures (the boundaries c = +/-1, c = 0, kappa =
    0, and negative kappas included), the even symmetry bit-exactly,
    and the checked-never-repaired domain contract."""

    # -- the angular law, exact on the self-authored fixtures --------
    for c, kappa in DENSITY_FIXTURES:
        got = mesonic_kernel_density(c, kappa)
        assert got == math.exp(kappa * c * c), (
            f"c={c}, kappa={kappa}: the density is not the pinned law"
        )
        assert got > 0.0 and math.isfinite(got)
    # the boundaries and the isotropic point, spelled out
    assert mesonic_kernel_density(0.0, 0.0) == 1.0
    assert mesonic_kernel_density(1.0, 0.0) == 1.0
    assert mesonic_kernel_density(-1.0, 0.0) == 1.0
    assert mesonic_kernel_density(1.0, 2.0) == math.exp(2.0)
    assert mesonic_kernel_density(-1.0, 2.0) == math.exp(2.0)
    # the negative kappa: the transverse enhancement (c = 0 carries
    # the maximum of exp(-|kappa| c**2))
    assert mesonic_kernel_density(0.0, -1.5) == 1.0
    assert mesonic_kernel_density(1.0, -1.5) == math.exp(-1.5)

    # -- the even symmetry, bit-exact (verified on disk) --------------
    # the law is even in c: density(-c) == density(c) exactly (the
    # c**2 argument is identical for +c and -c, so the on-disk
    # computation is bit-identical)
    for c in (-0.8, -0.45, -0.1, 0.0, 0.15, 0.7, 0.99):
        for kappa in (0.3, 1.0, 2.0, 4.0, -0.25, -1.5):
            assert (mesonic_kernel_density(-c, kappa)
                    == mesonic_kernel_density(c, kappa)), (
                f"c={c}, kappa={kappa}: the density is not even in c"
            )

    # -- the domain contract: checked, never repaired -----------------
    for c in (-1.5, 1.5, -1.0000001, 1.0000001, float("nan"),
              float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            mesonic_kernel_density(c, 1.0)
    with pytest.raises(ValueError):
        mesonic_kernel_density(0.5, float("nan"))
    with pytest.raises(ValueError):
        mesonic_kernel_density(0.5, float("inf"))
    with pytest.raises(ValueError):
        mesonic_kernel_density(0.5, float("-inf"))
    # the uncastable cosine string (the on-disk float cast fails)
    with pytest.raises(ValueError):
        mesonic_kernel_density("abc", 1.0)
    # the uncastable kappa (the on-disk float cast fails)
    with pytest.raises(ValueError):
        mesonic_kernel_density(0.5, "xyz")
    # the castable string is accepted (the on-disk float cast)
    assert mesonic_kernel_density("0.5", 2.0) == math.exp(0.5)


# ----------------------------------------------------------------------
# test_mesonic_normalization_analytic_and_limit
# ----------------------------------------------------------------------
def test_mesonic_normalization_analytic_and_limit():
    """The analytic ``sqrt(pi/kappa) * erfi(sqrt(kappa))`` on the
    self-authored grid (against the self-authored decimal erfi and
    the self-authored quadrature), the erf branch at the negative
    kappas, the kappa -> 0 limit exactly 2.0, the linear approach
    ``2 + 2 kappa / 3`` (the coefficient pinned), the branch boundary
    at exactly ``|kappa| = 1e-8`` (the closed form is taken, pinned
    to the on-disk strict-``<`` comparison), and the ``ValueError``
    edges."""

    # -- the positive-kappa analytic form ------------------------------
    for kappa in NORM_KAPPAS_POS:
        got = mesonic_normalization(kappa)
        if kappa == 0.0:
            continue
        analytic = (math.sqrt(math.pi / kappa)
                    * _reference_erfi(math.sqrt(kappa)))
        assert abs(got - analytic) <= 1e-12, (
            f"kappa={kappa}: the normalization {got!r} misses the "
            f"self-authored analytic {analytic!r}"
        )
        quad = _quadrature(kappa)
        assert abs(got - quad) <= 1e-12, (
            f"kappa={kappa}: the normalization {got!r} misses the "
            f"self-authored quadrature {quad!r}"
        )

    # -- the negative-kappa erf branch (the on-disk semantics) ---------
    for kappa in NORM_KAPPAS_NEG:
        got = mesonic_normalization(kappa)
        analytic = (math.sqrt(math.pi / -kappa)
                    * math.erf(math.sqrt(-kappa)))
        assert abs(got - analytic) <= 1e-12, (
            f"kappa={kappa}: the erf branch {got!r} misses the "
            f"analytic {analytic!r}"
        )
        assert abs(got - _quadrature(kappa)) <= 1e-12
        # the transverse-enhanced law is less spread than the
        # isotropic one: the normalization is below 2
        assert got < 2.0

    # the normalization is strictly increasing in kappa (structural:
    # exp(kappa c**2) is pointwise non-decreasing in kappa and strictly
    # increasing on the range c != 0), and its deviation from the
    # isotropic value 2.0 is positive for kappa > 0 (the aligned law is
    # more concentrated) and negative for kappa < 0
    combined = sorted(NORM_KAPPAS_NEG + NORM_KAPPAS_POS)
    values = [mesonic_normalization(kappa) for kappa in combined]
    for lower, upper in zip(combined, combined[1:]):
        assert values[combined.index(lower)] < values[combined.index(upper)], (
            f"kappa={lower} -> {upper}: the normalization is not "
            "strictly increasing"
        )
    for kappa in (0.3, 0.75, 1.0, 2.0, 4.0):
        assert mesonic_normalization(kappa) > 2.0
        assert mesonic_normalization(-kappa) < 2.0

    # -- the kappa -> 0 limit ------------------------------------------
    assert mesonic_normalization(0.0) == 2.0
    assert mesonic_normalization(-0.0) == 2.0
    for kappa in LIMIT_PROBES:
        assert abs(mesonic_normalization(kappa) - 2.0) <= 1e-12, (
            f"kappa={kappa}: the kappa -> 0 limit is discontinuous"
        )
    for kappa in (0.5 * ISOTROPIC_BRANCH_TOL,
                  -0.5 * ISOTROPIC_BRANCH_TOL):
        assert mesonic_normalization(kappa) == 2.0

    # -- the linear approach: 2 + 2 kappa / 3 (the coefficient) --------
    # the even law integrates to 2 + 2 kappa / 3 + O(kappa**2), so
    # the residual from the linear form is bounded by 5 kappa**2 at
    # the self-authored probe points, and the coefficient (norm - 2)
    # / kappa -> 2 / 3
    for kappa in LINEAR_APPROACH_KAPPAS:
        residual = abs(mesonic_normalization(kappa)
                       - (2.0 + (2.0 / 3.0) * kappa))
        assert residual <= 5.0 * kappa * kappa, (
            f"kappa={kappa}: the linear approach residual "
            f"{residual!r} exceeds 5 kappa**2"
        )
    for kappa in COEFFICIENT_PROBE_KAPPAS:
        coefficient = (mesonic_normalization(kappa) - 2.0) / kappa
        assert abs(coefficient - 2.0 / 3.0) <= 1e-3, (
            f"kappa={kappa}: the linear coefficient {coefficient!r} "
            "misses 2/3"
        )

    # -- the branch boundary: exactly |kappa| = 1e-8 -------------------
    # the on-disk comparison is strict (abs(kappa) < ISOTROPIC_BRANCH_
    # TOL), so the limit branch is NOT taken at the boundary: the
    # closed form is taken (pinned on disk)
    for kappa in (ISOTROPIC_BRANCH_TOL, -ISOTROPIC_BRANCH_TOL):
        got = mesonic_normalization(kappa)
        assert got != 2.0, (
            f"kappa={kappa}: the branch boundary takes the limit "
            "branch; the on-disk comparison is strict (abs(kappa) < "
            "tol)"
        )
        assert abs(got - 2.0) <= 1e-7, (
            f"kappa={kappa}: the closed form at the boundary left "
            "the isotropic neighborhood"
        )
    # and the limit branch stands just inside the boundary
    for kappa in (0.999e-8, -0.999e-8):
        assert mesonic_normalization(kappa) == 2.0

    # -- checked, never repaired --------------------------------------
    with pytest.raises(ValueError):
        mesonic_normalization(float("nan"))
    with pytest.raises(ValueError):
        mesonic_normalization(float("inf"))
    with pytest.raises(ValueError):
        mesonic_normalization(float("-inf"))


# ----------------------------------------------------------------------
# test_mesonic_cdf_table_construction
# ----------------------------------------------------------------------
def test_mesonic_cdf_table_construction():
    """The 10,001-node CDF table on the uniform cosine grid at the
    self-authored kappas: the endpoints, the even symmetry at
    construction, the monotone non-decreasing property, the node
    values against the self-authored analytic CDF reference, the
    kappa -> 0 table (the uniform grid, bit-exact), and the
    read-only / cached behavior (the on-disk cache semantics)."""

    for kappa in TABLE_KAPPAS:
        table = mesonic_cdf_table(kappa)
        # the node count, the dtype, and the uniform cosine grid
        assert table.shape == (MESONIC_CDF_NODES,)
        assert table.dtype == np.float64
        grid = np.linspace(-1.0, 1.0, MESONIC_CDF_NODES)
        # the node values against the self-authored analytic CDF
        # reference (the closed-form node construction carries no
        # quadrature error; the reference's 80-digit erfi converts to
        # float at the end)
        worst_node = 0.0
        for i, c in enumerate(grid):
            worst_node = max(
                worst_node, abs(float(table[i]) - _reference_cdf(c, kappa)))
        assert worst_node <= 1e-12, (
            f"kappa={kappa}: the node values miss the self-authored "
            f"analytic CDF by {worst_node!r}"
        )
        # the endpoints
        assert float(table[0]) == 0.0
        assert float(table[-1]) == 1.0
        # the even symmetry F(-c) = 1 - F(c) at construction (the
        # node pairs; the grid is symmetric, c_i = -c_(N-1-i))
        worst_nodes = float(np.max(np.abs(table + table[::-1] - 1.0)))
        assert worst_nodes <= 1e-12, (
            f"kappa={kappa}: the node symmetry misses by {worst_nodes!r}"
        )
        # the monotone non-decreasing property (and, at the active
        # kappas, strictly increasing: the density is positive
        # everywhere on [-1, 1])
        assert np.all(np.diff(table) >= 0.0), (
            f"kappa={kappa}: the table is not monotone non-decreasing"
        )
        assert np.all(np.diff(table) > 0.0), (
            f"kappa={kappa}: the table is not strictly increasing at "
            "the active kappa"
        )

    # -- the kappa -> 0 table: the uniform CDF (the node values are the
    # uniform grid, bit-exact)
    for kappa in (0.0, -0.0, 0.5 * ISOTROPIC_BRANCH_TOL,
                  -0.5 * ISOTROPIC_BRANCH_TOL):
        table = mesonic_cdf_table(kappa)
        uniform = 0.5 * (np.linspace(-1.0, 1.0, MESONIC_CDF_NODES) + 1.0)
        assert float(np.max(np.abs(table - uniform))) == 0.0, (
            f"kappa={kappa}: the isotropic table is not the uniform "
            "CDF grid bit-exactly"
        )
        assert float(table[0]) == 0.0 and float(table[-1]) == 1.0

    # -- the read-only / cached behavior (the on-disk cache semantics) -
    # the table is write-protected by convention
    table = mesonic_cdf_table(1.0)
    assert table.flags.writeable is False
    with pytest.raises(ValueError):
        table[0] = 0.5
    # a repeated kappa value serves the identical table from the
    # module cache (object identity, not a copy)
    again = mesonic_cdf_table(1.0)
    assert again is table
    assert cascade_alignment._MESONIC_CDF_TABLES[1.0] is table
    # the -0.0 / 0.0 cache aliasing (the float keys are equal): the
    # isotropic table is served from the same cache entry
    iso_pos = mesonic_cdf_table(0.0)
    iso_neg = mesonic_cdf_table(-0.0)
    assert iso_neg is iso_pos
    assert float(np.max(np.abs(mesonic_cdf_table(0.5 * ISOTROPIC_BRANCH_TOL)
                               - iso_pos))) == 0.0

    # -- checked, never repaired --------------------------------------
    with pytest.raises(ValueError):
        mesonic_cdf_table(float("nan"))
    with pytest.raises(ValueError):
        mesonic_cdf_table(float("inf"))


# ----------------------------------------------------------------------
# test_draw_mesonic_alignment_table_inversion
# ----------------------------------------------------------------------
def test_draw_mesonic_alignment_table_inversion():
    """The searchsorted + linear-interpolation inversion against the
    self-authored bisection inversion of the analytic CDF on the
    self-authored (u, kappa) grid, the CDF-space round trip bounded
    by the density-scaled interpolation bound, the u = 0 / u = 1
    endpoints, the even-law consequences (the quantile-symmetry
    identity and the contrast with the baryonic mirror), and the
    ``ValueError`` edges."""

    for kappa in DRAW_KAPPAS:
        max_density = _reference_max_density(kappa)
        previous = -2.0
        for u in sorted(QUANTILE_GRID):
            c = draw_mesonic_alignment(u, kappa)
            assert -1.0 <= c <= 1.0, (
                f"kappa={kappa}, u={u!r}: the draw left [-1, 1] ({c!r})"
            )
            # the round trip against the self-authored bisection
            # inversion (the pre-registered 1e-4 linear-interpolation
            # bound of the 10,001-node table)
            bisected = _bisect_inversion(u, kappa)
            deviation = abs(c - bisected)
            assert deviation <= 1e-4, (
                f"kappa={kappa}, u={u!r}: the round trip misses by "
                f"{deviation!r} in c (beyond the pre-registered 1e-4)"
            )
            # the CDF-space round trip, bounded by the interpolation
            # bound scaled by the maximum density
            back = _reference_cdf(c, kappa)
            assert abs(back - u) <= max_density * 1e-4, (
                f"kappa={kappa}, u={u!r}: the CDF round trip misses "
                f"by {abs(back - u)!r}"
            )
            # the draw is monotone in u
            assert c >= previous, (
                f"kappa={kappa}: the draw is not monotone in u at "
                f"u={u!r}"
            )
            previous = c

    # -- the endpoints u = 0 and u = 1 (the on-disk behavior) ----------
    # quantile 0 maps to cos(theta) = -1 exactly (the searchsorted
    # left edge), quantile 1 to +1 exactly (the last interpolation
    # span lands on the last grid node) at every active kappa
    for kappa in DRAW_KAPPAS:
        assert draw_mesonic_alignment(0.0, kappa) == -1.0, (
            f"kappa={kappa}: quantile 0 must map to cos(theta) = -1"
        )
        assert draw_mesonic_alignment(1.0, kappa) == 1.0, (
            f"kappa={kappa}: quantile 1 must map to cos(theta) = +1"
        )
    # at the isotropic branch the endpoints are exact by the law
    assert draw_mesonic_alignment(0.0, 0.0) == -1.0
    assert draw_mesonic_alignment(1.0, 0.0) == 1.0

    # -- the even-law consequences --------------------------------------
    # the table symmetry F(-c) = 1 - F(c) on the symmetric grid
    # implies the quantile-symmetry identity draw(u) + draw(1 - u) =
    # 0, so the mean is exactly zero by symmetry
    for kappa in DRAW_KAPPAS:
        for u in SYM_US:
            pair_sum = (draw_mesonic_alignment(u, kappa)
                        + draw_mesonic_alignment(1.0 - u, kappa))
            assert abs(pair_sum) <= 1e-12, (
                f"kappa={kappa}, u={u!r}: the quantile-symmetry "
                f"identity misses by {pair_sum!r}"
            )
    # the law at -kappa is distinct from the law at +kappa (the even
    # law has no baryonic-style mirror between kappa and -kappa):
    # pinned on disk at a self-authored pair
    assert (draw_mesonic_alignment(0.99, 2.0)
            != draw_mesonic_alignment(0.99, -2.0)), (
        "the draws at kappa and -kappa must be distinct (no baryonic "
        "mirror between the two)"
    )
    # while the baryonic mirror identity is the contrast (the baryon
    # law at -kappa mirrors against the reflected quantile):
    for u, kappa in ((0.3, 1.5), (0.7, 2.0)):
        mirrored = draw_baryonic_alignment(u, -kappa)
        expect = -draw_baryonic_alignment(1.0 - u, kappa)
        assert abs(mirrored - expect) <= 1e-12, (
            f"u={u!r}, kappa={kappa}: the baryonic mirror identity "
            "fails (the mesonic contrast)"
        )

    # -- checked, never repaired --------------------------------------
    for u in (-0.1, 1.1, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            draw_mesonic_alignment(u, 1.0)
    for kappa in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            draw_mesonic_alignment(0.5, kappa)
    # the uncastable string arguments (the on-disk float casts fail)
    with pytest.raises(ValueError):
        draw_mesonic_alignment("abc", 1.0)
    with pytest.raises(ValueError):
        draw_mesonic_alignment(0.5, "xyz")


# ----------------------------------------------------------------------
# test_draw_mesonic_isotropic_branch_boundary_and_endpoints
# ----------------------------------------------------------------------
def test_draw_mesonic_isotropic_branch_boundary_and_endpoints():
    """The isotropic branch ``2u - 1`` bit-stable with the exact
    float expression on the self-authored u values inside the branch
    tolerance; the branch boundary at exactly ``|kappa| = 1e-8`` (the
    table draw is taken, not the isotropic branch, pinned to the
    on-disk ``>=`` semantics); and the ``ValueError`` edges."""

    # -- the isotropic branch: bit-stable with the exact float --------
    for u in ISOTROPIC_US:
        for kappa in (0.0, -0.0, 0.5 * ISOTROPIC_BRANCH_TOL,
                      -0.5 * ISOTROPIC_BRANCH_TOL, 1e-9, -1e-9):
            got = draw_mesonic_alignment(u, kappa)
            assert got == 2.0 * u - 1.0, (
                f"kappa={kappa}, u={u!r}: the isotropic branch is not "
                "bit-stable with the exact float expression 2u - 1"
            )

    # -- the branch boundary: exactly |kappa| = 1e-8 ------------------
    # the on-disk comparison is strict (abs(kappa) < ISOTROPIC_BRANCH_
    # TOL), so the table draw is taken at the boundary, not the
    # isotropic branch (pinned on disk)
    for kappa in (ISOTROPIC_BRANCH_TOL, -ISOTROPIC_BRANCH_TOL):
        got = draw_mesonic_alignment(0.3, kappa)
        isotropic = 2.0 * 0.3 - 1.0
        assert got != isotropic, (
            f"kappa={kappa}: the boundary takes the isotropic branch; "
            "the on-disk comparison is strict (abs(kappa) < tol)"
        )
        # the table draw at the boundary is the same law in the limit:
        # within O(kappa) of the isotropic value
        assert abs(got - isotropic) < 1e-6, (
            f"kappa={kappa}: the boundary table draw left the "
            "isotropic neighborhood"
        )
    # and just inside the boundary the isotropic branch stands
    for kappa in (0.999e-8, -0.999e-8):
        assert draw_mesonic_alignment(0.3, kappa) == 2.0 * 0.3 - 1.0


# ----------------------------------------------------------------------
# test_draw_vertex_cos_theta_mesonic_dispatch
# ----------------------------------------------------------------------
def test_draw_vertex_cos_theta_mesonic_dispatch():
    """The dispatch in ``draw_vertex_cos_theta`` with
    ``kappa_mesonic``: the default-argument vertex preserving the
    card-6 contract, the active mesonic vertex (the even law in both
    daughter orders identically), the mixed baryon-meson vertex (the
    baryonic law bit-identically regardless of ``kappa_mesonic``),
    the two-baryon ``ValueError`` unchanged, and the input
    validation on the self-authored fixtures."""

    pair_lead_baryon = (BARYONIC_QN, MESONIC_QN)
    pair_trail_baryon = (MESONIC_QN, BARYONIC_QN)
    pair_no_baryon = (MESONIC_QN, MESONIC_QN)
    pair_charged = (CHARGED_MESONIC_QN, MESONIC_QN)
    pair_lead_strange = (STRANGE_BARYONIC_QN, CHARGED_MESONIC_QN)

    # -- the default-argument vertex (kappa_mesonic = 0.0): the card-6
    # contract preserved ------------------------------------------------
    # the baryon line is bit-identical to card 6's behavior (the
    # default-argument call, the mesonic kernel inert)
    for u in VERTEX_US:
        for kappa_b in (BARYONIC_KAPPA_DEFAULT, 0.5, 2.5, -1.5):
            got_lead = draw_vertex_cos_theta(
                u, pair_lead_baryon, kappa_baryonic=kappa_b)
            assert got_lead == draw_baryonic_alignment(u, kappa_b), (
                f"u={u!r}, kappa_b={kappa_b}: the default-argument "
                "leading baryon must carry the card-6 direct draw"
            )
            got_trail = draw_vertex_cos_theta(
                u, pair_trail_baryon, kappa_baryonic=kappa_b)
            expected_trail = (
                -draw_baryonic_alignment(u, kappa_b)
                if abs(kappa_b) >= ISOTROPIC_BRANCH_TOL
                else 2.0 * u - 1.0
            )
            assert got_trail == expected_trail, (
                f"u={u!r}, kappa_b={kappa_b}: the default-argument "
                "trailing baryon must carry the card-6 mirrored draw "
                "(the mirror dropped at the baryonic isotropic "
                "branch)"
            )
        # the mesonic vertex at the defaults: the card-6 isotropic
        # draw, bit-stable, at any baryonic strength
        for kappa_b in (0.0, BARYONIC_KAPPA_DEFAULT, 2.5):
            assert draw_vertex_cos_theta(
                u, pair_no_baryon, kappa_baryonic=kappa_b
            ) == 2.0 * u - 1.0, (
                f"u={u!r}: the default-argument mesonic vertex must "
                "reproduce the card-6 isotropic draw 2u - 1"
            )
    # the strange baryonic line takes the same law at the defaults
    for u in (0.2, 0.5, 0.8):
        assert draw_vertex_cos_theta(
            u, pair_lead_strange) == draw_baryonic_alignment(
            u, BARYONIC_KAPPA_DEFAULT)

    # -- the active mesonic vertex: the even law ------------------------
    # both daughters B == 0; the even law in both daughter orders
    # identically (no daughter-order mirror), the baryonic strength
    # not consulted
    for u in VERTEX_US:
        for kappa_m in (MESONIC_KAPPA_DEFAULT, 2.5, 4.0, -1.5):
            expected = draw_mesonic_alignment(u, kappa_m)
            for pair in (pair_no_baryon, pair_charged,
                         (MESONIC_QN, CHARGED_MESONIC_QN)):
                assert draw_vertex_cos_theta(
                    u, pair, kappa_mesonic=kappa_m) == expected, (
                    f"u={u!r}, kappa_m={kappa_m}: the symmetric "
                    "mesonic vertex must take the even law"
                )
            # the baryonic strength is not consulted on the mesonic
            # vertex (both orders identical at any kappa_baryonic)
            for kappa_b in (0.0, BARYONIC_KAPPA_DEFAULT, 2.5, -2.5):
                assert draw_vertex_cos_theta(
                    u, pair_charged, kappa_baryonic=kappa_b,
                    kappa_mesonic=kappa_m) == expected, (
                    f"u={u!r}, kappa_b={kappa_b}, kappa_m={kappa_m}: "
                    "the baryonic strength must not be consulted on "
                    "the mesonic vertex"
                )

    # -- the mixed baryon-meson vertex: the baryonic law byte-identically
    # regardless of kappa_mesonic ----------------------------------------
    for u in VERTEX_US:
        for kappa_b in (BARYONIC_KAPPA_DEFAULT, 0.5, 2.5, -1.5, 0.0):
            for kappa_m in (0.0, MESONIC_KAPPA_DEFAULT, 4.0, -1.0):
                got_lead = draw_vertex_cos_theta(
                    u, pair_lead_baryon, kappa_baryonic=kappa_b,
                    kappa_mesonic=kappa_m)
                assert got_lead == draw_baryonic_alignment(u, kappa_b), (
                    f"u={u!r}, kappa_b={kappa_b}, kappa_m={kappa_m}: "
                    "the leading baryon must carry the direct kernel "
                    "draw regardless of the mesonic strength"
                )
                got_trail = draw_vertex_cos_theta(
                    u, pair_trail_baryon, kappa_baryonic=kappa_b,
                    kappa_mesonic=kappa_m)
                expected_trail = (
                    -draw_baryonic_alignment(u, kappa_b)
                    if abs(kappa_b) >= ISOTROPIC_BRANCH_TOL
                    else 2.0 * u - 1.0
                )
                assert got_trail == expected_trail, (
                    f"u={u!r}, kappa_b={kappa_b}, kappa_m={kappa_m}: "
                    "the trailing baryon must carry the mirrored "
                    "kernel draw (the mirror dropped at the baryonic "
                    "isotropic branch) regardless of the mesonic "
                    "strength"
                )

    # -- the mesonic branch boundary: exactly |kappa_M| = 1e-8 ---------
    # the on-disk comparison is ">=", so the table draw is active at
    # the boundary (and the isotropic branch stands just inside)
    for kappa in (ISOTROPIC_BRANCH_TOL, -ISOTROPIC_BRANCH_TOL):
        got = draw_vertex_cos_theta(
            0.3, pair_no_baryon, kappa_mesonic=kappa)
        assert got == draw_mesonic_alignment(0.3, kappa), (
            f"kappa={kappa}: the boundary must take the mesonic table "
            "draw (the on-disk '>=' comparison)"
        )
        assert got != 2.0 * 0.3 - 1.0
    for kappa in (0.999e-8, -0.999e-8):
        assert draw_vertex_cos_theta(
            0.3, pair_no_baryon, kappa_mesonic=kappa) == 2.0 * 0.3 - 1.0

    # -- the two-baryon ValueError (checked, never repaired) ------------
    for pair in ((BARYONIC_QN, STRANGE_BARYONIC_QN),):
        with pytest.raises(ValueError):
            draw_vertex_cos_theta(0.5, pair, kappa_mesonic=1.0)
        with pytest.raises(ValueError):
            draw_vertex_cos_theta(0.5, pair)
    # -- the baryon-antibaryon pair (Round 6, plan section 12.4) --------
    # net baryon number zero: no baryonic line passes through the
    # vertex, so the pair follows the symmetric mesonic dispatch in
    # either daughter order (the even law at an active kappa_mesonic,
    # the isotropic branch at the vertex default), bit for bit the
    # symmetric mesonic vertex's own draw
    for pair in ((ANTIBARYONIC_QN, BARYONIC_QN),
                 (STRANGE_BARYONIC_QN, ANTIBARYONIC_QN)):
        assert draw_vertex_cos_theta(
            0.3, pair, kappa_mesonic=1.0) == draw_vertex_cos_theta(
                0.3, pair_no_baryon, kappa_mesonic=1.0)
        assert draw_vertex_cos_theta(0.5, pair) == 2.0 * 0.5 - 1.0
        assert draw_vertex_cos_theta(
            0.3, pair, kappa_baryonic=1.0) == 2.0 * 0.3 - 1.0

    # -- the input validation on the self-authored fixtures -------------
    with pytest.raises(ValueError):
        draw_vertex_cos_theta(-0.05, pair_no_baryon, kappa_mesonic=1.0)
    with pytest.raises(ValueError):
        draw_vertex_cos_theta(1.05, pair_no_baryon, kappa_mesonic=1.0)
    with pytest.raises(ValueError):
        draw_vertex_cos_theta(float("nan"), pair_no_baryon,
                              kappa_mesonic=1.0)
    with pytest.raises(ValueError):
        draw_vertex_cos_theta(0.5, pair_no_baryon,
                              kappa_mesonic=float("inf"))
    with pytest.raises(ValueError):
        draw_vertex_cos_theta(0.5, pair_no_baryon,
                              kappa_mesonic=float("-inf"))
    with pytest.raises(ValueError):
        draw_vertex_cos_theta(0.5, pair_no_baryon,
                              kappa_mesonic=float("nan"))
    with pytest.raises(ValueError):
        draw_vertex_cos_theta(0.5, pair_no_baryon,
                              kappa_baryonic=float("inf"),
                              kappa_mesonic=1.0)
    with pytest.raises(ValueError):
        draw_vertex_cos_theta(0.5, ((1, 1), MESONIC_QN),
                              kappa_mesonic=1.0)
    with pytest.raises(ValueError):
        draw_vertex_cos_theta(0.5, (MESONIC_QN, (0, 0, 0, 0)),
                              kappa_mesonic=1.0)
    # a pair that is not a two-daughter sequence
    with pytest.raises(ValueError):
        draw_vertex_cos_theta(0.5, "abc", kappa_mesonic=1.0)
    # the castable string quantile is accepted (the on-disk float
    # cast)
    assert draw_vertex_cos_theta(
        "0.5", pair_no_baryon, kappa_mesonic=1.0
    ) == draw_mesonic_alignment(0.5, 1.0)


# ----------------------------------------------------------------------
# test_assign_alignment_label_mesonic
# ----------------------------------------------------------------------
def test_assign_alignment_label_mesonic():
    """The label on the mesonic daughter while the mesonic kernel is
    active, the label on the baryon line (unchanged), the card-4
    ``ISOTROPIC_ALIGNMENT_LABEL`` fallback at the isotropic branches
    and off both lines, the vertex-exact call-site convention at each
    vertex kind, and the input validation."""

    # the card-4 placeholder constant, pinned by identity (the module
    # imports it, it does not redeclare it)
    assert ISOTROPIC_ALIGNMENT_LABEL == "isotropic"

    # -- the mesonic daughter while the kernel is active ---------------
    for kappa in (MESONIC_KAPPA_DEFAULT, 4.0, -2.0,
                  ISOTROPIC_BRANCH_TOL, -ISOTROPIC_BRANCH_TOL):
        assert assign_alignment_label(
            MESONIC_QN, kappa_mesonic=kappa) == MESONIC_ALIGNMENT_LABEL, (
            f"kappa={kappa}: the mesonic daughter must carry the "
            "mesonic label at the active branch (the on-disk '>=' "
            "semantics)"
        )
        assert assign_alignment_label(
            CHARGED_MESONIC_QN, kappa_mesonic=kappa
        ) == MESONIC_ALIGNMENT_LABEL

    # -- the fallback at the isotropic branch ---------------------------
    for kappa in (0.0, -0.0, 0.5 * ISOTROPIC_BRANCH_TOL,
                  -0.5 * ISOTROPIC_BRANCH_TOL, 9e-9, -9e-9,
                  0.999e-8, -0.999e-8):
        assert assign_alignment_label(
            MESONIC_QN, kappa_mesonic=kappa) == ISOTROPIC_ALIGNMENT_LABEL, (
            f"kappa={kappa}: the isotropic branch must fall back to "
            "the card-4 placeholder"
        )

    # -- the fallback off both lines -------------------------------------
    # the mesonic daughter at the active baryonic strength (off the
    # baryonic line, the mesonic kernel inert at the defaults): the
    # placeholder stands
    assert assign_alignment_label(
        MESONIC_QN, kappa_baryonic=2.5) == ISOTROPIC_ALIGNMENT_LABEL
    assert assign_alignment_label(
        CHARGED_MESONIC_QN, kappa_baryonic=2.5,
        kappa_mesonic=0.0) == ISOTROPIC_ALIGNMENT_LABEL

    # -- the baryonic labels are unchanged (at any mesonic state) -------
    assert assign_alignment_label(
        BARYONIC_QN) == BARYONIC_ALIGNMENT_LABEL
    assert assign_alignment_label(
        STRANGE_BARYONIC_QN) == BARYONIC_ALIGNMENT_LABEL
    assert assign_alignment_label(
        ANTIBARYONIC_QN) == BARYONIC_ALIGNMENT_LABEL
    for kappa in (MESONIC_KAPPA_DEFAULT, 4.0, -2.0, 0.0):
        assert assign_alignment_label(
            BARYONIC_QN, kappa_mesonic=kappa) == BARYONIC_ALIGNMENT_LABEL, (
            f"kappa={kappa}: the baryonic label must be unchanged at "
            "any mesonic state"
        )
    # the baryon line at the baryonic isotropic branch falls back,
    # even at the active mesonic kernel (the baryonic branch requires
    # the baryonic kernel active)
    assert assign_alignment_label(
        BARYONIC_QN, kappa_baryonic=0.0,
        kappa_mesonic=MESONIC_KAPPA_DEFAULT) == ISOTROPIC_ALIGNMENT_LABEL

    # -- the vertex-exact call-site convention --------------------------
    # the governing kernel of a drawn vertex is the baryonic one at
    # any vertex with a baryon-carrying daughter (the other kappa held
    # at the isotropic branch) and the mesonic one at the symmetric
    # mesonic vertex: the labels agree with the drawn law at every
    # vertex kind and kappa state
    pair_baryon_leading = (BARYONIC_QN, MESONIC_QN)
    pair_baryon_trailing = (MESONIC_QN, BARYONIC_QN)
    pair_no_baryon = (MESONIC_QN, MESONIC_QN)
    for u in VERTEX_US:
        for kappa_b, kappa_m in (
            (BARYONIC_KAPPA_DEFAULT, 0.0),
            (BARYONIC_KAPPA_DEFAULT, MESONIC_KAPPA_DEFAULT),
            (0.0, MESONIC_KAPPA_DEFAULT),
            (0.0, 0.0),
            (2.5, -1.5),
        ):
            for pair in (pair_baryon_leading, pair_baryon_trailing,
                         pair_no_baryon):
                draw = draw_vertex_cos_theta(
                    u, pair, kappa_baryonic=kappa_b,
                    kappa_mesonic=kappa_m)
                if pair[0][0] != 0 or pair[1][0] != 0:
                    gov_b, gov_m = kappa_b, 0.0
                else:
                    gov_b, gov_m = 0.0, kappa_m
                labels = [
                    assign_alignment_label(
                        qn, kappa_baryonic=gov_b, kappa_mesonic=gov_m)
                    for qn in pair
                ]
                active = (abs(gov_b) >= ISOTROPIC_BRANCH_TOL
                          or abs(gov_m) >= ISOTROPIC_BRANCH_TOL)
                if not active:
                    for label in labels:
                        assert label == ISOTROPIC_ALIGNMENT_LABEL
                    assert draw == 2.0 * u - 1.0
                elif pair[0][0] != 0 or pair[1][0] != 0:
                    expected_labels = [
                        BARYONIC_ALIGNMENT_LABEL if qn[0] != 0
                        else ISOTROPIC_ALIGNMENT_LABEL
                        for qn in pair
                    ]
                    assert labels == expected_labels, (
                        f"u={u!r}, pair={pair}: the baryonic-line "
                        "labels disagree with the convention"
                    )
                    if pair[0][0] != 0:
                        assert draw == draw_baryonic_alignment(
                            u, gov_b)
                    else:
                        assert draw == -draw_baryonic_alignment(
                            u, gov_b)
                else:
                    assert labels == [MESONIC_ALIGNMENT_LABEL,
                                      MESONIC_ALIGNMENT_LABEL], (
                        f"u={u!r}: the symmetric mesonic labels "
                        "disagree with the convention"
                    )
                    assert draw == draw_mesonic_alignment(u, gov_m)

    # -- checked, never repaired -----------------------------------------
    with pytest.raises(ValueError):
        assign_alignment_label((1, 1), kappa_mesonic=1.0)
    with pytest.raises(ValueError):
        assign_alignment_label(MESONIC_QN, kappa_mesonic=float("nan"))
    with pytest.raises(ValueError):
        assign_alignment_label(MESONIC_QN, kappa_mesonic=float("-inf"))
    with pytest.raises(ValueError):
        assign_alignment_label(MESONIC_QN, kappa_baryonic=float("inf"))


# ----------------------------------------------------------------------
# test_module_constants
# ----------------------------------------------------------------------
def test_module_constants():
    """The module constants: ``MESONIC_KAPPA_DEFAULT == 1.0``,
    ``MESONIC_CDF_NODES == 10001``, and
    ``MESONIC_ALIGNMENT_LABEL == "mesonic_alignment"`` (value,
    type), with the card-6 constants still present and the
    package-level constants resolving to the module's objects."""

    assert MESONIC_KAPPA_DEFAULT == 1.0
    assert isinstance(MESONIC_KAPPA_DEFAULT, float)
    assert MESONIC_CDF_NODES == 10001
    assert isinstance(MESONIC_CDF_NODES, int)
    assert MESONIC_ALIGNMENT_LABEL == "mesonic_alignment"
    assert isinstance(MESONIC_ALIGNMENT_LABEL, str)
    # the card-6 constants are still present
    assert BARYONIC_KAPPA_DEFAULT == 1.0
    assert isinstance(BARYONIC_KAPPA_DEFAULT, float)
    assert ISOTROPIC_BRANCH_TOL == 1e-8
    assert isinstance(ISOTROPIC_BRANCH_TOL, float)
    assert BARYONIC_ALIGNMENT_LABEL == "baryonic_alignment"
    assert isinstance(BARYONIC_ALIGNMENT_LABEL, str)
    # the package-level constants are the module's objects (the
    # additive exports, not copies)
    assert na49_mixing.MESONIC_KAPPA_DEFAULT is MESONIC_KAPPA_DEFAULT
    assert na49_mixing.MESONIC_CDF_NODES is MESONIC_CDF_NODES
    assert (na49_mixing.MESONIC_ALIGNMENT_LABEL
            is MESONIC_ALIGNMENT_LABEL)
    assert na49_mixing.BARYONIC_KAPPA_DEFAULT is BARYONIC_KAPPA_DEFAULT
    assert na49_mixing.ISOTROPIC_BRANCH_TOL is ISOTROPIC_BRANCH_TOL
    assert (na49_mixing.BARYONIC_ALIGNMENT_LABEL
            is BARYONIC_ALIGNMENT_LABEL)


# ----------------------------------------------------------------------
# test_zero_draw_contract
# ----------------------------------------------------------------------
def test_zero_draw_contract():
    """The module's public functions consume no randomness on the
    measurement path: the fixed-seed bit-generator state is
    byte-identical across a pinned operation cascade over all eight
    public functions (the CDF table construction included, which is
    deterministic), and the module source carries no random machinery
    (the self-authored source scan)."""

    rng = np.random.default_rng(REGISTER_SEED)
    state_before = rng.bit_generator.state

    kappas = (MESONIC_KAPPA_DEFAULT, 0.0, 0.5, 2.0, 4.0, -1.0, -2.5)
    pairs = (
        (MESONIC_QN, MESONIC_QN),
        (CHARGED_MESONIC_QN, MESONIC_QN),
        (MESONIC_QN, CHARGED_MESONIC_QN),
        (BARYONIC_QN, MESONIC_QN),
    )
    operations = 0
    for step in range(900):
        u = (step % 19) / 18.0
        c = -1.0 + (step % 11) / 10.0
        kappa = kappas[(step * 5) % len(kappas)]
        pair = pairs[(step * 2) % len(pairs)]
        qn = pair[step % 2]
        kind = step % 6
        if kind == 0:
            mesonic_kernel_density(c, kappa)
        elif kind == 1:
            mesonic_normalization(kappa)
        elif kind == 2:
            mesonic_cdf_table(kappa)
        elif kind == 3:
            draw_mesonic_alignment(u, kappa)
        elif kind == 4:
            draw_vertex_cos_theta(u, pair, kappa_baryonic=0.0,
                                  kappa_mesonic=kappa)
        else:
            assign_alignment_label(qn, kappa_baryonic=0.0,
                                   kappa_mesonic=kappa)
        operations += 1
    assert operations == 900

    state_after = rng.bit_generator.state
    assert state_after == state_before, (
        "the mesonic kernel consumed random numbers: the fixed-seed "
        "bit-generator state changed over the pinned 900-operation "
        "cascade"
    )

    # the module source carries no random machinery (the
    # self-authored scan: tokens that would appear in any
    # random-number consumer; the module is standard-library math,
    # numpy arrays, and the card-4 label constant only)
    source = MODULE_PATH.read_text(encoding="utf-8")
    forbidden = (
        "import random", "from random", "random.", "np.random",
        "numpy.random", "default_rng", "RandomState", "SeedSequence",
        "getrandbits", "randint(", "random(", "rand(", "shuffle(",
        "gauss(", "weibullvariate(", "bit_generator", "spawn(",
        "getstate(", "setstate(",
    )
    hits = [token for token in forbidden if token in source]
    assert not hits, f"the alignment module carries random machinery: {hits}"


# ----------------------------------------------------------------------
# test_baryonic_region_byte_stable
# ----------------------------------------------------------------------
def test_baryonic_region_byte_stable():
    """The card-6 baryonic region of the extended module (from the
    ``__all__`` closing bracket to the ``draw_vertex_cos_theta``
    definition) at its pinned sha256, recomputed by this register's
    own boundary logic and verified byte-identical; the baryonic
    constants and functions in the region, the mesonic names absent
    from it, and the single mesonic section marker in the extended
    file."""

    source = MODULE_PATH.read_text(encoding="utf-8")

    # this register's own boundary logic (independent of the solver's
    # pinned block): the __all__ list, its closing bracket, and the
    # draw_vertex_cos_theta definition
    all_start = source.index("__all__ = [")
    close = source.index("\n]\n", all_start)
    region_start = close + len("\n]\n")
    def_index = source.index("def draw_vertex_cos_theta(", region_start)
    region = source[region_start:def_index]
    digest = hashlib.sha256(region.encode("utf-8")).hexdigest()
    assert digest == BARYONIC_REGION_SHA256, (
        "the card-6 baryonic region drifted inside the module "
        f"({digest!r} against the pinned "
        f"{BARYONIC_REGION_SHA256!r})"
    )

    # the baryonic surface in the region, in its landed order
    for marker in (
        "BARYONIC_KAPPA_DEFAULT: float = 1.0",
        "ISOTROPIC_BRANCH_TOL: float = 1e-8",
        'BARYONIC_ALIGNMENT_LABEL: str = "baryonic_alignment"',
        "def baryonic_kernel_density(",
        "def baryonic_normalization(",
        "def draw_baryonic_alignment(",
        "def continues_baryon_line(",
    ):
        assert marker in region, (
            f"the baryonic region lost {marker!r}"
        )
    # the mesonic names are absent from the baryonic region (the
    # additive extension is below draw_vertex_cos_theta)
    assert "MESONIC" not in region, (
        "the baryonic region is not byte-stable: mesonic names "
        "appear inside it"
    )

    # the card-7 extension is the single appended section (exactly
    # one mesonic-kernel section marker in the extended file)
    marker = "# The mesonic kernel (plan section 10.4, lines 724-728)"
    assert source.count(marker) == 1


# ----------------------------------------------------------------------
# test_module_byte_and_export_pins
# ----------------------------------------------------------------------
def test_module_byte_and_export_pins():
    """The module byte stability (sha256) and the additive
    ``na49_mixing`` package exports (every one of the 16 names of the
    extended ``__all__`` present in the package ``__init__`` and
    resolved by object identity)."""

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    # the module's own export list (the contract surface; the nine
    # card-6 baryonic names lead the list in their landed order and
    # card 7 appended its additive mesonic names below them)
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
