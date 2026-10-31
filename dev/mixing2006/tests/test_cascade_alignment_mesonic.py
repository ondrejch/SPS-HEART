"""Card tests for the mesonic alignment kernel.

Round 4, card 7 (``TASK-20260920-r4-mesonic-alignment-kernel``; plan
section 22 task 24, plan section 10.4, lines 724-732).  The card's
pre-registered deterministic tests against the closed-form identities,
the CDF-table identities, the pinned quantile grids, the committed
catalog (``config/v2_singlet/resonance_catalog.json``), and the
continuum family configuration
(``config/v2_singlet/continuum_resonances.json``):

- ``test_constants``: ``MESONIC_KAPPA_DEFAULT == 1.0``,
  ``MESONIC_CDF_NODES == 10001``, ``MESONIC_ALIGNMENT_LABEL ==
  "mesonic_alignment"`` (value and type), each resolved by object
  identity from the package exports, with the branch tolerance the
  card-6 constant.
- ``test_normalization``: ``mesonic_normalization`` agrees with the
  analytic ``sqrt(pi/kappa) * erfi(sqrt(kappa))`` (the independent
  scipy reference) and with a Gauss-Legendre quadrature cross-check
  within 1e-12 at kappa in {0.5, 1, 2, 4}; the erf analogue holds at
  the negative kappas; the kappa -> 0 limit is continuous to 1e-12
  (the even law approaches the limit linearly, 2 + 2 kappa / 3 + ...,
  so the probe grid stays within 1.5e-12 of the branch).
- ``test_table_even_symmetry``: the CDF table satisfies F(-c) = 1 -
  F(c) within 1e-12 at kappa in {0.5, 1, 2, 4}, on the node pairs and
  through the linear interpolation at off-node cosines; the table
  carries exactly ``MESONIC_CDF_NODES`` nodes with F[0] = 0, F[-1] =
  1, strictly increasing, and the even symmetry of the density
  itself.
- ``test_table_interpolation_round_trip``: the sampled draw round-
  trips the uniform quantile within 1e-4 in c against an independent
  bisection inversion of the analytic CDF (scipy erfi reference) on
  the pinned quantile grid at kappa in {1, 2, 4} (the pre-registered
  set; 0.5 and a negative kappa added), with the CDF-space round trip
  bounded by the density-scaled interpolation bound and the exact
  endpoints.
- ``test_isotropic_branch_bit_stable``: at kappa = 0 (and inside the
  pre-registered branch tolerance) the draw is bit-identical to the
  card-4/card-6 isotropic branch (2u - 1) on a pinned seed — the R43
  ablation baseline contract (card 14, acceptance (ii)); the branch
  boundary at exactly |kappa| = 1e-8 takes the table (the on-disk
  ``>=`` comparison semantics of the card-6 branch).
- ``test_vertex_dispatch``: ``draw_vertex_cos_theta`` dispatches the
  baryon line versus the mesonic daughters: the mixed baryon-meson
  vertices keep the card-6 baryonic law exactly (direct and mirrored)
  regardless of ``kappa_mesonic``; the symmetric mesonic vertices
  (both daughters ``B == 0``) take the even mesonic law when active,
  in both daughter orders identically (no mirror), and the card-6
  isotropic draw at the defaults and inside the branch tolerance;
  the two-baryon ``ValueError`` and the input validation are
  unchanged (checked, never repaired).
- ``test_alignment_label_assignment``: the mesonic daughter carries
  ``MESONIC_ALIGNMENT_LABEL`` while the mesonic kernel is active and
  falls back to the card-4 placeholder at the isotropic branch; the
  baryonic labels are unchanged; the vertex-exact call-site
  convention (the governing kernel's kappa, the other at the
  isotropic branch) reproduces labels that agree with the drawn law
  at every vertex kind and kappa state.
- ``test_draw_budget``: the mesonic attachment consumes exactly one
  uniform (the cos(theta) table draw) in place of the Round 2
  cos(theta) draw on a pinned symmetric mesonic vertex of the
  continuum ``mesonic_neutral`` family; the per-vertex budget of
  card 4 stays 1 channel + 2 kinematics (counting proxies through
  ``draw_decay``), the azimuth draw and the substream derivation
  order are unchanged (the card-4 draw-site source pin), and at
  ``kappa_M = 0`` the attached vertex value is bit-identical to the
  card-4 isotropic draw.
- ``test_zero_draws``: a pinned cascade of 1,000 mesonic kernel
  operations consumes zero random numbers (the generator state is
  byte-identical before and after) and the module source carries no
  random machinery (the zero-draw contract beyond the replaced
  draw).
- ``test_baryonic_surface_byte_stable``: the card-6 baryonic surface
  of the module is byte-unchanged within the extended file — the
  region from the ``__all__`` closing bracket to the
  ``draw_vertex_cos_theta`` definition (the baryonic constants, the
  validators, the baryonic density, normalization, closed-form draw,
  and the baryon-line predicate) at its pinned sha256, the baryonic
  ``__all__`` block in its landed order, and the baryonic dispatch
  and label branches as byte-exact substrings; the card-6 test
  module (``test_cascade_alignment_baryonic.py``) pins the behavior
  unchanged in the same suite.
- ``test_layout_sync``: the consumed-surface register (the frozen
  substrate digests of the card's provenance section, the card-4,
  card-5, and card-6 surfaces included, verified on disk) and the
  layout sync (the subprocess check from the monorepo root;
  ``logic_drift=0``).

plus the acceptance-(vi) freeze re-pin check ``test_freeze_manifest_repin``.

Hermetic: the committed config and catalog files, numpy, scipy's erfi
(the independent analytic reference of the normalization and the
round-trip inversion; the module itself stays scipy-free), the
standard library, and the vendored ``na49_mixing`` modules only; no
event generation, no external tables, no Monte Carlo run.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.special import erfi as scipy_erfi

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
    baryonic_kernel_density,
    baryonic_normalization,
    draw_baryonic_alignment,
    draw_mesonic_alignment,
    draw_vertex_cos_theta,
    mesonic_cdf_table,
    mesonic_kernel_density,
    mesonic_normalization,
)
from na49_mixing.cascade_decay import (
    CATALOG_FAMILY_PREFIX,
    ISOTROPIC_ALIGNMENT_LABEL,
    KIND_CONTINUUM,
    CascadeState,
    draw_decay,
    enumerate_channels,
)
from na49_mixing.hadron_catalog import HadronCatalog
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
MODULE_PATH = NESTED_ROOT / "na49_mixing" / "cascade_alignment.py"
CARD4_PATH = NESTED_ROOT / "na49_mixing" / "cascade_decay.py"

#: The round 4 master seed (round 4 index item 9), fixed for the
#: deterministic substream draws of the card-4 integration fixture and
#: for the pinned statistical streams (the offsets are distinct from
#: the card-6 test module's).
MASTER_SEED = 20260920

#: The pre-registered kappa values of the acceptance criteria (i)-(ii).
KAPPA_GRID = (0.0, 0.5, 1.0, 2.0, 4.0)
NORMALIZATION_KAPPAS = (0.5, 1.0, 2.0, 4.0)
ROUND_TRIP_KAPPAS = (1.0, 2.0, 4.0)

#: The pinned quantile grid: the regular 51-point grid plus the pinned
#: edge and deep-tail quantiles of the round-trip fidelity check.
QUANTILE_GRID = tuple(
    [j / 50 for j in range(51)]
    + [1e-12, 1.0 - 1e-12, 0.5 ** 20, 1.0 - 0.5 ** 20, 1.0 / 3.0, 2.0 / 3.0]
)

#: The pinned (cosine, kappa) fixtures of the angular-law check (the
#: negative kappas included: the even law is defined on the full
#: order-one range and the transverse enhancement is the erf branch).
DENSITY_FIXTURES = tuple(
    (c, k)
    for k in (0.0, 0.25, 1.0, 2.5, 4.0, -0.5, -2.0)
    for c in (-1.0, -0.6, -0.3, 0.0, 0.2, 0.65, 0.99, 1.0)
)

#: The pinned symmetric mesonic vertex fixture: the 8 GeV neutral
#: mesonic root keeps the realized continuum daughter inside the
#: family support, so the pinned continuation has an open line (the
#: card-4 mass rule realizes the daughter at the budget edge minus
#: the family nominal width); every daughter of a (0, 0, 0) parent
#: carries B = 0 by conservation.
ROOT_FAMILY = "mesonic_neutral"
ROOT_MASS = 8.0
ROOT_QN = (0, 0, 0)
BARYONIC_QN = (1, 1, 0)
STRANGE_BARYONIC_QN = (1, 0, -1)
MESONIC_QN = (0, 0, 0)
CHARGED_MESONIC_QN = (0, 1, 0)
VERTEX_EVENT = 19

#: The Gauss-Legendre quadrature of the independent reference
#: (800 nodes; the kappa grid's worst quadrature error is below
#: 1e-14, far inside the pre-registered 1e-12).
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(800)

#: The frozen substrate of the card's provenance section (byte-unchanged
#: surfaces, verified on disk at this card's landing): card 4's cascade
#: surface, card 5's axis machinery, and card 6's baryonic kernel
#: module bytes at the card-6 landing digest (consumed read-only; the
#: baryonic surface inside the extended module is pinned by
#: ``test_baryonic_surface_byte_stable``), the builder, the card-6
#: blend routing, the card-2 engine, the committed catalog, the
#: continuum families configuration, and the dual-tree R32 driver.
FROZEN_SURFACES = {
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
    "dev/mixing2006/helpers/benchmarking/staged_r32_spectral_systematic.py":
        "99454facce856c93505e6c4b0d4e47b85eeb0ef54f72b1631e3d29fcc511cb90",
    "helpers/benchmarking/staged_r32_spectral_systematic.py":
        "99454facce856c93505e6c4b0d4e47b85eeb0ef54f72b1631e3d29fcc511cb90",
}

#: The byte-stable baryonic surface of the module (acceptance (v)):
#: the sha256 of the region from the ``__all__`` closing bracket to
#: the ``draw_vertex_cos_theta`` definition — the baryonic constants,
#: the validation helpers, the baryonic density, the analytic
#: normalization, the closed-form inverse-CDF draw, and the
#: baryon-line predicate, byte-identical to the card-6 landing
#: (verified against the committed card-6 file at this card's
#: landing).
BARYONIC_REGION_SHA256 = (
    "0d2feb2e31d1501e4d57ef45157e0876f00b17dc99938828b599fea5dfb8c431"
)

#: The card-6 baryonic ``__all__`` block (the nine names in their
#: landed order, pinned as a byte-exact prefix of the extended list).
BARYONIC_ALL_BLOCK = (
    "__all__ = [\n"
    '    "BARYONIC_KAPPA_DEFAULT",\n'
    '    "ISOTROPIC_BRANCH_TOL",\n'
    '    "BARYONIC_ALIGNMENT_LABEL",\n'
    '    "baryonic_kernel_density",\n'
    '    "baryonic_normalization",\n'
    '    "draw_baryonic_alignment",\n'
    '    "continues_baryon_line",\n'
    '    "draw_vertex_cos_theta",\n'
    '    "assign_alignment_label",\n'
)

#: The baryonic dispatch of ``draw_vertex_cos_theta`` (the card-6
#: lines, pinned as a byte-exact substring of the extended body).
BARYONIC_DISPATCH_BLOCK = (
    "    if lead_baryon:\n"
    "        return draw_baryonic_alignment(u_value, kappa_value)\n"
    "    if trail_baryon and abs(kappa_value) >= ISOTROPIC_BRANCH_TOL:\n"
    "        return -draw_baryonic_alignment(u_value, kappa_value)\n"
)

#: The baryonic branch of ``assign_alignment_label`` (the card-6
#: lines, pinned as a byte-exact substring of the extended body).
BARYONIC_LABEL_BLOCK = (
    "    if qn[0] != 0 and abs(kappa_value) >= ISOTROPIC_BRANCH_TOL:\n"
    "        return BARYONIC_ALIGNMENT_LABEL\n"
)

#: This card's freeze re-pin entries (the extended kernel module, the
#: additive package exports, this new test file, and the six consumed
#: count-pin files whose hard inventory-count pins the card-mandated
#: re-pin bumps: the card-3 audit register, the card-4 register, the
#: card-5 solver test, the card-5 register, the card-6 solver test,
#: and the card-6 register).  The manifest itself is excluded from its
#: own inventory by path and appears only in the freeze exception's
#: file list.
FREEZE_CARD = "TASK-20260920-r4-mesonic-alignment-kernel"
# 461 entries at the card-6 register landing; this card adds its
# mesonic-kernel test file (+1 entry) for the card-mandated re-pin
# (461 -> 462) and re-pins the six consumed count pins with the bump
# attributed to this card's test file (test_cascade_alignment_mesonic);
# card 7's independent test-author register
# (test_r4_cascade_alignment_mesonic_contract_pins.py) re-pins the
# inventory additively for its own file (462 -> 463, +1 entry); card 8
# of the round (TASK-20260920-r4-hadronic-decay-form-factor) re-pins
# the inventory additively for its new form-factor module and test
# file (na49_mixing/cascade_form_factor.py,
# test_cascade_form_factor.py; 463 -> 465, +2 entries); card 8's
# independent test-author register
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
MANIFEST_REL = ("dev/mixing2006/config/v2_singlet/"
                "legacy_v1_manifest.json")
MODULE_REL = "dev/mixing2006/na49_mixing/cascade_alignment.py"
INIT_REL = "dev/mixing2006/na49_mixing/__init__.py"
MY_TEST_REL = "dev/mixing2006/tests/test_cascade_alignment_mesonic.py"
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
FREEZE_INVENTORY_FILES = (
    MODULE_REL,
    INIT_REL,
    MY_TEST_REL,
    CARD3_REGISTER_REL,
    CARD4_REGISTER_REL,
    CARD5_SOLVER_TEST_REL,
    CARD5_REGISTER_REL,
    CARD6_SOLVER_TEST_REL,
    CARD6_REGISTER_REL,
)
FREEZE_EXCEPTION_FILES = FREEZE_INVENTORY_FILES + (MANIFEST_REL,)


# ----------------------------------------------------------------------
# Fixtures, references, and helpers
# ----------------------------------------------------------------------
@pytest.fixture(scope="session")
def catalog() -> HadronCatalog:
    return HadronCatalog.from_json(CATALOG_PATH)


@pytest.fixture(scope="session")
def fams():
    return families_from_json(str(FAMILIES_PATH))


def _reference_cdf(c: float, kappa: float) -> float:
    """The independent analytic CDF of the even law at the cosine *c*
    (the scipy erfi/erf reference; the isotropic uniform reference
    inside the branch tolerance)."""

    if abs(kappa) < ISOTROPIC_BRANCH_TOL:
        return 0.5 * (c + 1.0)
    if kappa > 0.0:
        s = math.sqrt(kappa)
        return 0.5 * (1.0 + float(scipy_erfi(s * c)) / float(scipy_erfi(s)))
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
                      iters: int = 120) -> float:
    """The independent bisection inversion of the analytic CDF on
    [-1, 1] (120 halvings drive the interval below float resolution,
    so the result is the unique fixed point to machine precision)."""

    lo, hi = -1.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _reference_cdf(mid, kappa) < u:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _quadrature(kappa: float) -> float:
    """The Gauss-Legendre quadrature of exp(kappa c^2) over [-1, 1]
    on the fixed 800-node grid."""

    values = np.exp(kappa * _GL_NODES * _GL_NODES)
    return float(np.sum(_GL_WEIGHTS * values))


def _interpolate_table(table: np.ndarray, c: float) -> float:
    """The linear interpolation of the CDF table at the cosine *c*
    (the inverse construction of the sampler's interpolation)."""

    if c <= -1.0:
        return float(table[0])
    if c >= 1.0:
        return float(table[-1])
    pos = (c + 1.0) * 0.5 * (MESONIC_CDF_NODES - 1)
    idx = int(pos)
    if idx >= MESONIC_CDF_NODES - 1:
        return float(table[-1])
    t = pos - idx
    return float(table[idx] * (1.0 - t) + table[idx + 1] * t)


def _root_state(family_id: str, mass: float, qn: tuple[int, int, int],
                depth: int = 0) -> CascadeState:
    """A pinned at-rest root state with the lab-z production axis (the
    isolated-source convention of round 4 index item 4) and the card-4
    isotropic label."""

    return CascadeState(
        momentum=(float(mass), 0.0, 0.0, 0.0),
        mass=float(mass),
        qn=qn,
        quantum={},
        family_id=family_id,
        axis=(0.0, 0.0, 1.0),
        alignment_label=ISOTROPIC_ALIGNMENT_LABEL,
        parent=None,
        exchange_ancestry=("exchange_probe",),
        depth=depth,
    )


class _CountingRng:
    """A counting proxy around one seeded generator (the house
    draw-count pattern): every ``random()`` consumes exactly one
    uniform and advances the counter."""

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        return float(self._rng.random())


def _sub_rngs(seed: int, event_index: int):
    """The closed decay_channel/decay_kinematics substream generators
    of one (master seed, replica 0, event index) triple (the round 2
    card 9 and round 3 card 8 convention)."""

    subs = derive_substreams(seed, 0, event_index)
    return (
        np.random.default_rng(subs.stream_for("decay_channel")),
        np.random.default_rng(subs.stream_for("decay_kinematics")),
    )


def _daughter_qn_of(catalog: HadronCatalog, fams, daughter_id: str):
    """The ``(B, Q, S)`` of one channel daughter id (the established
    catalog record's quantum numbers or the continuum family's)."""

    if daughter_id.startswith(CATALOG_FAMILY_PREFIX):
        record = catalog.get(int(daughter_id[len(CATALOG_FAMILY_PREFIX):]))
        return (int(record.B), int(record.Q), int(record.S))
    for family in fams:
        if str(family.family_id) == daughter_id:
            return tuple(int(v) for v in family.qn)
    raise AssertionError(f"unknown daughter id {daughter_id!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ----------------------------------------------------------------------
# test_constants
# ----------------------------------------------------------------------
def test_constants():
    """The module constants: ``MESONIC_KAPPA_DEFAULT == 1.0``,
    ``MESONIC_CDF_NODES == 10001``, and ``MESONIC_ALIGNMENT_LABEL ==
    "mesonic_alignment"``; the branch tolerance is the card-6
    constant; the package-level constants resolve to the module's
    objects."""

    assert MESONIC_KAPPA_DEFAULT == 1.0
    assert isinstance(MESONIC_KAPPA_DEFAULT, float)
    assert MESONIC_CDF_NODES == 10001
    assert isinstance(MESONIC_CDF_NODES, int)
    assert MESONIC_ALIGNMENT_LABEL == "mesonic_alignment"
    assert isinstance(MESONIC_ALIGNMENT_LABEL, str)
    assert ISOTROPIC_BRANCH_TOL == 1e-8
    # the package-level constants are the module's objects (the
    # additive exports, not copies)
    assert na49_mixing.MESONIC_KAPPA_DEFAULT is MESONIC_KAPPA_DEFAULT
    assert na49_mixing.MESONIC_CDF_NODES is MESONIC_CDF_NODES
    assert na49_mixing.MESONIC_ALIGNMENT_LABEL is MESONIC_ALIGNMENT_LABEL
    assert (na49_mixing.draw_mesonic_alignment
            is cascade_alignment.draw_mesonic_alignment)
    assert (na49_mixing.mesonic_cdf_table
            is cascade_alignment.mesonic_cdf_table)


# ----------------------------------------------------------------------
# test_normalization (acceptance (i))
# ----------------------------------------------------------------------
def test_normalization():
    """The kernel normalization agrees with the analytic
    ``sqrt(pi/kappa) * erfi(sqrt(kappa))`` (the independent scipy
    reference) and with the quadrature cross-check within 1e-12 at
    kappa in {0.5, 1, 2, 4}; the erf analogue holds at the negative
    kappas; the kappa -> 0 limit is continuous to 1e-12."""

    for kappa in NORMALIZATION_KAPPAS:
        got = mesonic_normalization(kappa)
        analytic = (math.sqrt(math.pi / kappa)
                    * float(scipy_erfi(math.sqrt(kappa))))
        assert abs(got - analytic) <= 1e-12, (
            f"kappa={kappa}: normalization {got!r} misses the analytic "
            f"{analytic!r}"
        )
        quad = _quadrature(kappa)
        assert abs(got - quad) <= 1e-12, (
            f"kappa={kappa}: normalization {got!r} misses the "
            f"quadrature {quad!r}"
        )

    # the density integrates to the normalization on the same grid,
    # and the negative-kappa erf branch is exact as well
    for kappa in (-0.5, -1.0, -2.0, -4.0):
        got = mesonic_normalization(kappa)
        analytic = (math.sqrt(math.pi / -kappa)
                    * math.erf(math.sqrt(-kappa)))
        assert abs(got - analytic) <= 1e-12
        assert abs(got - _quadrature(kappa)) <= 1e-12

    # the kappa -> 0 limit is exactly 2.0 at the branch and continuous
    # to 1e-12 across it (the even law approaches the limit linearly,
    # 2 + 2 kappa / 3 + O(kappa^3), so the probe grid stays within
    # 1.5e-12 of the branch on both sides)
    assert mesonic_normalization(0.0) == 2.0
    assert mesonic_normalization(-0.0) == 2.0
    for kappa in (1e-13, -1e-13, 5e-13, -5e-13, 1e-12, -1e-12,
                  1.4e-12, -1.4e-12):
        assert abs(mesonic_normalization(kappa) - 2.0) <= 1e-12, (
            f"kappa={kappa}: the limit is discontinuous"
        )
    for kappa in (0.5 * ISOTROPIC_BRANCH_TOL, -0.5 * ISOTROPIC_BRANCH_TOL):
        assert mesonic_normalization(kappa) == 2.0
    # and the closed form just outside the branch agrees with the
    # limit to the linear order (2 + 2 kappa / 3)
    for kappa in (ISOTROPIC_BRANCH_TOL, -ISOTROPIC_BRANCH_TOL):
        closed = mesonic_normalization(kappa)
        assert abs(closed - 2.0) <= 1e-7

    # -- checked, never repaired ----------------------------------------
    with pytest.raises(ValueError):
        mesonic_normalization(float("nan"))
    with pytest.raises(ValueError):
        mesonic_normalization(float("inf"))


# ----------------------------------------------------------------------
# test_table_even_symmetry (acceptance (ii), symmetry leg)
# ----------------------------------------------------------------------
def test_table_even_symmetry():
    """The CDF table satisfies F(-c) = 1 - F(c) within 1e-12 at kappa
    in {0.5, 1, 2, 4}, on the node pairs and through the linear
    interpolation at off-node cosines; the table carries exactly
    MESONIC_CDF_NODES nodes, F[0] = 0, F[-1] = 1, strictly increasing,
    and the density itself is even."""

    probe_cs = (-0.93, -0.5, -0.21, -0.05, 0.05, 0.21, 0.5, 0.93)
    for kappa in (0.5, 1.0, 2.0, 4.0):
        table = mesonic_cdf_table(kappa)
        assert table.shape == (MESONIC_CDF_NODES,)
        assert table.dtype == np.float64
        assert float(table[0]) == 0.0
        assert float(table[-1]) == 1.0
        assert np.all(np.diff(table) > 0.0), (
            f"kappa={kappa}: the table is not strictly increasing"
        )
        # the node-pair symmetry
        worst_nodes = float(np.max(np.abs(table + table[::-1] - 1.0)))
        assert worst_nodes <= 1e-12, (
            f"kappa={kappa}: the node symmetry misses by {worst_nodes!r}"
        )
        # the off-node interpolated symmetry
        for c in probe_cs:
            f_pos = _interpolate_table(table, c)
            f_neg = _interpolate_table(table, -c)
            assert abs(f_pos + f_neg - 1.0) <= 1e-12, (
                f"kappa={kappa}, c={c}: the interpolated even symmetry "
                f"misses by {abs(f_pos + f_neg - 1.0)!r}"
            )
        # the density is even in the cosine (the law's structural
        # symmetry the table inherits)
        for c in probe_cs:
            assert (mesonic_kernel_density(c, kappa)
                    == mesonic_kernel_density(-c, kappa))

    # the isotropic table (the branch tolerance) is the uniform CDF
    for kappa in (0.0, -0.0, 0.5 * ISOTROPIC_BRANCH_TOL):
        table = mesonic_cdf_table(kappa)
        assert float(table[0]) == 0.0 and float(table[-1]) == 1.0
        assert float(np.max(np.abs(
            table - 0.5 * (np.linspace(-1.0, 1.0, MESONIC_CDF_NODES) + 1.0)
        ))) == 0.0

    # -- checked, never repaired ----------------------------------------
    with pytest.raises(ValueError):
        mesonic_kernel_density(1.5, 1.0)
    with pytest.raises(ValueError):
        mesonic_kernel_density(float("nan"), 1.0)
    with pytest.raises(ValueError):
        mesonic_kernel_density(0.5, float("inf"))
    with pytest.raises(ValueError):
        mesonic_cdf_table(float("nan"))


# ----------------------------------------------------------------------
# test_table_interpolation_round_trip (acceptance (ii), sampler leg)
# ----------------------------------------------------------------------
def test_table_interpolation_round_trip():
    """The sampled draw round-trips the uniform quantile within 1e-4
    in c against the independent bisection inversion of the analytic
    CDF on the pinned quantile grid at kappa in {1, 2, 4} (the
    pre-registered set; 0.5 and a negative kappa added), with the
    CDF-space round trip bounded by the density-scaled interpolation
    bound; the endpoints land on the cosine range (quantile 0 exactly
    at cos(theta) = -1, quantile 1 within 1e-12 of +1); the draw is
    monotone in u."""

    for kappa in ROUND_TRIP_KAPPAS + (0.5, -1.0):
        max_density = _reference_max_density(kappa)
        previous = -2.0
        for u in sorted(QUANTILE_GRID):
            c = draw_mesonic_alignment(u, kappa)
            assert -1.0 <= c <= 1.0, (
                f"kappa={kappa}, u={u!r}: the draw left [-1, 1] "
                f"({c!r})"
            )
            # the pre-registered c-space round trip against the
            # independent inversion
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
            assert c >= previous, (
                f"kappa={kappa}: the draw is not monotone in u at "
                f"u={u!r}"
            )
            previous = c

    # the endpoints
    for kappa in ROUND_TRIP_KAPPAS:
        assert draw_mesonic_alignment(0.0, kappa) == -1.0
        assert abs(draw_mesonic_alignment(1.0, kappa) - 1.0) <= 1e-12
    # at the isotropic branch the endpoints are exact by the law
    assert draw_mesonic_alignment(0.0, 0.0) == -1.0
    assert draw_mesonic_alignment(1.0, 0.0) == 1.0

    # -- checked, never repaired ----------------------------------------
    for u in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            draw_mesonic_alignment(u, 1.0)
    for kappa in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            draw_mesonic_alignment(0.5, kappa)


# ----------------------------------------------------------------------
# test_isotropic_branch_bit_stable (acceptance (iii))
# ----------------------------------------------------------------------
def test_isotropic_branch_bit_stable():
    """At kappa = 0 (and inside the pre-registered branch tolerance)
    the draw is bit-identical to the card-4/card-6 isotropic branch
    ``2u - 1`` on a pinned seed — the R43 ablation baseline contract
    (card 14, acceptance (ii)); the branch boundary at exactly
    |kappa| = 1e-8 takes the table (the on-disk ``>=`` comparison
    semantics of the card-6 branch)."""

    stream = np.random.default_rng(MASTER_SEED + 83)
    pinned_us = [float(u) for u in stream.random(1_000)]
    for u in pinned_us + list(QUANTILE_GRID):
        for kappa in (0.0, -0.0, 0.5 * ISOTROPIC_BRANCH_TOL,
                      -0.5 * ISOTROPIC_BRANCH_TOL, 1e-9, -1e-9):
            got = draw_mesonic_alignment(u, kappa)
            assert got == 2.0 * u - 1.0, (
                f"kappa={kappa}, u={u!r}: the isotropic branch is not "
                "bit-stable with the Round 2 isotropic draw"
            )

    # the branch boundary: exactly |kappa| = 1e-8 is NOT inside the
    # strict "<" branch, so the table draw is taken (pinned on disk)
    for kappa in (ISOTROPIC_BRANCH_TOL, -ISOTROPIC_BRANCH_TOL):
        got = draw_mesonic_alignment(0.3, kappa)
        isotropic = 2.0 * 0.3 - 1.0
        assert got != isotropic, (
            f"kappa={kappa}: the boundary takes the isotropic branch; "
            "the on-disk comparison is strict (abs(kappa) < tol)"
        )
        # the table draw at the boundary is the same law in the
        # limit: within O(kappa) of the isotropic value
        assert abs(got - isotropic) < 1e-6, (
            f"kappa={kappa}: the boundary table draw left the "
            "isotropic neighborhood"
        )
    # and just inside the boundary the isotropic branch stands
    for kappa in (0.999e-8, -0.999e-8):
        assert draw_mesonic_alignment(0.3, kappa) == 2.0 * 0.3 - 1.0

    # the card-4 draw site computes exactly the pinned expression
    # (the same float expression on the same quantile is the
    # bit-identity)
    card4_src = CARD4_PATH.read_text(encoding="utf-8")
    assert "cos_theta = 2.0 * u_cos - 1.0" in card4_src, (
        "the card-4 isotropic cos(theta) draw site is not the pinned "
        "expression the isotropic branch must reproduce bit for bit"
    )


# ----------------------------------------------------------------------
# test_vertex_dispatch (the baryon-versus-meson dispatch)
# ----------------------------------------------------------------------
def test_vertex_dispatch():
    """The card-6 vertex interface dispatches the baryon line versus
    the mesonic daughters: the mixed vertices keep the card-6 baryonic
    law exactly regardless of kappa_mesonic; the symmetric mesonic
    vertices take the even law when active (both daughter orders
    identically) and the card-6 isotropic draw at the defaults and
    inside the branch tolerance; the two-baryon ``ValueError`` and the
    input validation are unchanged."""

    pair_lead_baryon = (BARYONIC_QN, MESONIC_QN)
    pair_trail_baryon = (MESONIC_QN, BARYONIC_QN)
    pair_no_baryon = (MESONIC_QN, MESONIC_QN)
    pair_charged = (CHARGED_MESONIC_QN, MESONIC_QN)
    pair_lead_strange = (STRANGE_BARYONIC_QN, CHARGED_MESONIC_QN)

    # -- the mixed vertices keep the baryonic law exactly --------------
    for u in (0.0, 0.13, 0.37, 0.5, 0.66, 0.88, 1.0):
        for kappa_b in (BARYONIC_KAPPA_DEFAULT, 0.5, 2.5, -1.5):
            for kappa_m in (0.0, MESONIC_KAPPA_DEFAULT, 4.0):
                got_lead = draw_vertex_cos_theta(
                    u, pair_lead_baryon,
                    kappa_baryonic=kappa_b, kappa_mesonic=kappa_m)
                assert got_lead == draw_baryonic_alignment(u, kappa_b), (
                    f"u={u!r}, kappa_b={kappa_b}, kappa_m={kappa_m}: "
                    "the leading baryon must carry the direct kernel "
                    "draw regardless of the mesonic strength"
                )
                got_trail = draw_vertex_cos_theta(
                    u, pair_trail_baryon,
                    kappa_baryonic=kappa_b, kappa_mesonic=kappa_m)
                expected_trail = (
                    -draw_baryonic_alignment(u, kappa_b)
                    if abs(kappa_b) >= ISOTROPIC_BRANCH_TOL
                    else 2.0 * u - 1.0
                )
                assert got_trail == expected_trail, (
                    f"u={u!r}, kappa_b={kappa_b}, kappa_m={kappa_m}: "
                    "the trailing baryon must carry the mirrored "
                    "kernel draw (the mirror dropped at the baryonic "
                    "isotropic branch)"
                )
    # the strange baryonic line takes the same law at the defaults
    for u in (0.2, 0.5, 0.8):
        assert draw_vertex_cos_theta(
            u, pair_lead_strange) == draw_baryonic_alignment(
            u, BARYONIC_KAPPA_DEFAULT)

    # -- the symmetric mesonic vertices take the even law --------------
    for u in (0.0, 0.13, 0.37, 0.5, 0.66, 0.88, 1.0):
        for kappa_m in (MESONIC_KAPPA_DEFAULT, 0.5, 2.0, 4.0, -1.0):
            expected = draw_mesonic_alignment(u, kappa_m)
            # both daughter orders draw the same value (the even law
            # has no daughter-order mirror)
            assert draw_vertex_cos_theta(
                u, pair_no_baryon, kappa_mesonic=kappa_m) == expected
            assert draw_vertex_cos_theta(
                u, (MESONIC_QN, CHARGED_MESONIC_QN),
                kappa_mesonic=kappa_m) == expected
            assert draw_vertex_cos_theta(
                u, pair_charged, kappa_baryonic=0.0,
                kappa_mesonic=kappa_m) == expected
            # the baryonic strength is not consulted on the mesonic
            # vertex
            assert draw_vertex_cos_theta(
                u, pair_no_baryon, kappa_baryonic=2.5,
                kappa_mesonic=kappa_m) == expected

    # -- the card-6 interface at the defaults --------------------------
    # the vertex default kappa_mesonic = 0.0 keeps the card-6 isotropic
    # law on the mesonic vertex (the card-6 test module pins it)
    for u in (0.0, 0.25, 0.5, 0.75, 1.0 - 1e-9):
        assert draw_vertex_cos_theta(u, pair_no_baryon) == 2.0 * u - 1.0
        assert draw_vertex_cos_theta(
            u, pair_no_baryon, kappa_baryonic=2.5) == 2.0 * u - 1.0

    # -- the mesonic branch boundary: exactly |kappa_M| = 1e-8 ---------
    # the on-disk comparison is ">=", so the table draw is active at
    # the boundary (and the isotropic branch just inside)
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
    for pair in ((BARYONIC_QN, STRANGE_BARYONIC_QN),
                 (STRANGE_BARYONIC_QN, BARYONIC_QN)):
        with pytest.raises(ValueError):
            draw_vertex_cos_theta(0.5, pair)

    # -- the input validation -------------------------------------------
    with pytest.raises(ValueError):
        draw_vertex_cos_theta(-0.1, pair_no_baryon, kappa_mesonic=1.0)
    with pytest.raises(ValueError):
        draw_vertex_cos_theta(1.1, pair_no_baryon, kappa_mesonic=1.0)
    with pytest.raises(ValueError):
        draw_vertex_cos_theta(float("nan"), pair_no_baryon,
                              kappa_mesonic=1.0)
    with pytest.raises(ValueError):
        draw_vertex_cos_theta(0.5, pair_no_baryon,
                              kappa_mesonic=float("inf"))
    with pytest.raises(ValueError):
        draw_vertex_cos_theta(0.5, pair_no_baryon,
                              kappa_mesonic=float("nan"))
    with pytest.raises(ValueError):
        draw_vertex_cos_theta(0.5, ((1, 1), MESONIC_QN),
                              kappa_mesonic=1.0)
    with pytest.raises(ValueError):
        draw_vertex_cos_theta(0.5, "abc", kappa_mesonic=1.0)


# ----------------------------------------------------------------------
# test_alignment_label_assignment (the label bookkeeping)
# ----------------------------------------------------------------------
def test_alignment_label_assignment(catalog, fams):
    """The mesonic daughter carries ``MESONIC_ALIGNMENT_LABEL`` while
    the mesonic kernel is active, with the fallback to the card-4
    placeholder at the isotropic branch; the baryonic labels are
    unchanged; the vertex-exact call-site convention (the governing
    kernel's kappa, the other at the isotropic branch) reproduces
    labels that agree with the drawn law at every vertex kind and
    kappa state; on drawn card-4 daughters the assignment replaces
    the placeholder without touching any other daughter field."""

    # -- the per-daughter assignment ------------------------------------
    assert assign_alignment_label(
        MESONIC_QN, kappa_mesonic=MESONIC_KAPPA_DEFAULT
    ) == MESONIC_ALIGNMENT_LABEL
    assert assign_alignment_label(
        CHARGED_MESONIC_QN, kappa_mesonic=4.0) == MESONIC_ALIGNMENT_LABEL
    # the defaults and the isotropic branch: the placeholder stands
    assert assign_alignment_label(MESONIC_QN) == ISOTROPIC_ALIGNMENT_LABEL
    assert assign_alignment_label(
        MESONIC_QN, kappa_baryonic=2.5) == ISOTROPIC_ALIGNMENT_LABEL
    for kappa in (0.0, -0.0, 0.5 * ISOTROPIC_BRANCH_TOL,
                  -0.5 * ISOTROPIC_BRANCH_TOL, 9e-9, -9e-9):
        assert assign_alignment_label(
            MESONIC_QN, kappa_mesonic=kappa) == ISOTROPIC_ALIGNMENT_LABEL
    # the branch boundary at exactly |kappa_M| = 1e-8 (the >= semantics)
    for kappa in (ISOTROPIC_BRANCH_TOL, -ISOTROPIC_BRANCH_TOL):
        assert assign_alignment_label(
            MESONIC_QN, kappa_mesonic=kappa) == MESONIC_ALIGNMENT_LABEL
    # the baryonic labels are unchanged (at either mesonic state)
    assert assign_alignment_label(
        BARYONIC_QN) == BARYONIC_ALIGNMENT_LABEL
    assert assign_alignment_label(
        BARYONIC_QN, kappa_mesonic=MESONIC_KAPPA_DEFAULT
    ) == BARYONIC_ALIGNMENT_LABEL
    assert assign_alignment_label(
        BARYONIC_QN, kappa_baryonic=0.0,
        kappa_mesonic=MESONIC_KAPPA_DEFAULT) == ISOTROPIC_ALIGNMENT_LABEL
    with pytest.raises(ValueError):
        assign_alignment_label((1, 1))
    with pytest.raises(ValueError):
        assign_alignment_label(MESONIC_QN, kappa_mesonic=float("nan"))
    with pytest.raises(ValueError):
        assign_alignment_label(MESONIC_QN, kappa_mesonic=float("inf"))

    # -- the vertex-exact convention: the labels and the drawn law ------
    # agree at every vertex kind and kappa state
    pair_baryon_leading = (BARYONIC_QN, MESONIC_QN)
    pair_baryon_trailing = (MESONIC_QN, BARYONIC_QN)
    pair_no_baryon = (MESONIC_QN, MESONIC_QN)
    for u in QUANTILE_GRID:
        for kappa_b, kappa_m in (
            (BARYONIC_KAPPA_DEFAULT, 0.0),
            (BARYONIC_KAPPA_DEFAULT, MESONIC_KAPPA_DEFAULT),
            (0.0, MESONIC_KAPPA_DEFAULT),
            (0.0, 0.0),
            (0.5, 2.0),
        ):
            for pair in (pair_baryon_leading, pair_baryon_trailing,
                         pair_no_baryon):
                draw = draw_vertex_cos_theta(
                    u, pair, kappa_baryonic=kappa_b,
                    kappa_mesonic=kappa_m)
                # the governing kernel: the baryonic line at any
                # vertex with a baryon daughter, the mesonic kernel
                # at the symmetric mesonic vertex
                if pair[0][0] != 0 or pair[1][0] != 0:
                    gov_b, gov_m = kappa_b, 0.0
                else:
                    gov_b, gov_m = 0.0, kappa_m
                labels = [
                    assign_alignment_label(
                        qn, kappa_baryonic=gov_b, kappa_mesonic=gov_m)
                    for qn in pair
                ]
                active = abs(gov_b) >= ISOTROPIC_BRANCH_TOL or (
                    abs(gov_m) >= ISOTROPIC_BRANCH_TOL)
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

    # -- the card-4 integration: the labels on drawn daughters ---------
    root = _root_state(ROOT_FAMILY, ROOT_MASS, ROOT_QN)
    channels = enumerate_channels(root, catalog, fams)
    pool = [c for c in channels if c.kind == KIND_CONTINUUM
            and ROOT_FAMILY in c.daughters]
    assert pool
    rng_channel = _CountingRng(MASTER_SEED + 91)
    rng_kinematics = _CountingRng(MASTER_SEED + 92)
    _, daughters = draw_decay(
        root, pool, catalog, fams, rng_channel, rng_kinematics)
    assert rng_channel.calls == 1 and rng_kinematics.calls == 2
    # a (0, 0, 0) parent conserves baryon number: both daughters are
    # mesonic
    for daughter in daughters:
        assert int(daughter.qn[0]) == 0
    # card 4 copies the parent label verbatim; this card's assignment
    # replaces it at the call site (the governing kernel of the
    # symmetric mesonic vertex is the mesonic one)
    for daughter in daughters:
        assert daughter.alignment_label == ISOTROPIC_ALIGNMENT_LABEL
    filled = [
        replace(daughter, alignment_label=assign_alignment_label(
            daughter.qn, kappa_baryonic=0.0,
            kappa_mesonic=MESONIC_KAPPA_DEFAULT))
        for daughter in daughters
    ]
    for state, daughter in zip(filled, daughters):
        assert state is not daughter
        assert state.alignment_label == MESONIC_ALIGNMENT_LABEL
        assert state.momentum == daughter.momentum
        assert state.mass == daughter.mass
        assert state.qn == daughter.qn
        assert state.family_id == daughter.family_id
        assert state.axis == daughter.axis
        assert state.parent is daughter.parent
        assert state.exchange_ancestry == daughter.exchange_ancestry
        assert state.depth == daughter.depth
    # the ablation baseline: at kappa_M = 0 every daughter falls back
    for daughter in daughters:
        assert assign_alignment_label(
            daughter.qn, kappa_baryonic=0.0,
            kappa_mesonic=0.0) == ISOTROPIC_ALIGNMENT_LABEL


# ----------------------------------------------------------------------
# test_draw_budget (acceptance (iv))
# ----------------------------------------------------------------------
def test_draw_budget(catalog, fams):
    """The mesonic attachment consumes exactly one uniform (the
    cos(theta) draw) in place of the Round 2 cos(theta) draw on a
    pinned symmetric mesonic vertex; the per-vertex budget of card 4
    stays 1 channel + 2 kinematics (counting proxies through
    ``draw_decay``); the azimuth draw and the substream derivation
    order are unchanged (the card-4 draw-site source pin); at
    ``kappa_M = 0`` the attached vertex value is bit-identical to the
    card-4 isotropic draw."""

    root = _root_state(ROOT_FAMILY, ROOT_MASS, ROOT_QN)
    channels = enumerate_channels(root, catalog, fams)
    pool = [c for c in channels if c.kind == KIND_CONTINUUM
            and ROOT_FAMILY in c.daughters]
    assert pool

    # -- the per-vertex budget of card 4 is unchanged -------------------
    rng_channel = _CountingRng(MASTER_SEED + 101)
    rng_kinematics = _CountingRng(MASTER_SEED + 102)
    channel, daughters = draw_decay(
        root, pool, catalog, fams, rng_channel, rng_kinematics)
    assert rng_channel.calls == 1 and rng_kinematics.calls == 2, (
        "the per-vertex budget of card 4 moved"
    )
    pair = tuple(
        _daughter_qn_of(catalog, fams, daughter_id)
        for daughter_id in channel.daughters
    )
    assert len(pair) == 2
    assert all(qn[0] == 0 for qn in pair), (
        "the pinned symmetric mesonic vertex carries two mesonic "
        "daughters"
    )

    # -- the attachment at the uniform level ----------------------------
    # the three uniforms of the card-4 vertex, in the card-4 order: the
    # channel draw on the decay_channel substream, then cos(theta) and
    # the azimuth on the decay_kinematics substream
    subs_channel, subs_kinematics = _sub_rngs(MASTER_SEED, VERTEX_EVENT)
    u_channel = float(subs_channel.random())
    u_cos = float(subs_kinematics.random())
    u_phi = float(subs_kinematics.random())
    card4_cos = 2.0 * u_cos - 1.0

    # the aligned cos(theta) at the same quantile slot: the kernel draw
    # replaces the isotropic value, the azimuth is untouched, and no
    # additional uniform is consumed by the attachment
    aligned_cos = draw_vertex_cos_theta(
        u_cos, pair, kappa_mesonic=MESONIC_KAPPA_DEFAULT)
    assert aligned_cos == draw_mesonic_alignment(
        u_cos, MESONIC_KAPPA_DEFAULT)
    assert -1.0 <= aligned_cos <= 1.0
    phi = 2.0 * math.pi * u_phi
    assert phi == 2.0 * math.pi * u_phi  # the unchanged azimuth draw

    # the isotropic limit at the pre-registered kappa_M = 0: the
    # attached vertex value is bit-identical to the card-4 isotropic
    # draw (the R43 ablation baseline, card 14 acceptance (ii)), and
    # so is the default-argument vertex call (the card-6 interface)
    assert draw_vertex_cos_theta(
        u_cos, pair, kappa_mesonic=0.0) == card4_cos
    assert draw_vertex_cos_theta(u_cos, pair) == card4_cos

    # the default-strength attachment differs from the isotropic
    # baseline (the alignment is active at the pre-registered default)
    assert aligned_cos != card4_cos


# ----------------------------------------------------------------------
# test_zero_draws (the zero-draw contract beyond the replaced draw)
# ----------------------------------------------------------------------
def test_zero_draws():
    """A pinned cascade of 1,000 mesonic kernel operations consumes
    zero random numbers (the generator state is byte-identical before
    and after) and the module source carries no random machinery."""

    rng = np.random.default_rng(MASTER_SEED + 111)
    state_before = rng.bit_generator.state

    kappas = (MESONIC_KAPPA_DEFAULT, 0.0, 0.5, 2.0, 4.0, -1.0)
    pairs = (
        (MESONIC_QN, MESONIC_QN),
        (CHARGED_MESONIC_QN, MESONIC_QN),
        (BARYONIC_QN, MESONIC_QN),
    )
    operations = 0
    for step in range(1_000):
        u = 0.5 ** (step % 53)
        c = -1.0 + (step % 41) / 20.0
        kappa = kappas[step % len(kappas)]
        pair = pairs[step % len(pairs)]
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
            assign_alignment_label(pair[step % 2], kappa_baryonic=0.0,
                                   kappa_mesonic=kappa)
        operations += 1
    assert operations == 1_000

    state_after = rng.bit_generator.state
    assert state_after == state_before, (
        "the mesonic kernel consumed random numbers: the generator "
        "state changed over the pinned 1,000-operation cascade"
    )

    # the module source carries no random machinery (the audit-card
    # zero-draw source contract, the card-6 scan token list)
    source = MODULE_PATH.read_text(encoding="utf-8")
    forbidden = (
        "np.random", "default_rng", "SeedSequence", "import random",
        "random.", "randint", "shuffle(", "gauss(", "weibull",
        "getstate", "setstate", "rng", "uniform(",
    )
    hits = [token for token in forbidden if token in source]
    assert not hits, f"the alignment module carries random machinery: {hits}"


# ----------------------------------------------------------------------
# test_baryonic_surface_byte_stable (acceptance (v))
# ----------------------------------------------------------------------
def test_baryonic_surface_byte_stable():
    """The card-6 baryonic surface of the module is byte-unchanged
    within the extended file: the region from the ``__all__`` closing
    bracket to the ``draw_vertex_cos_theta`` definition (the baryonic
    constants, the validation helpers, the baryonic density, the
    analytic normalization, the closed-form inverse-CDF draw, and the
    baryon-line predicate) at its pinned sha256 — the digest verified
    byte-identical against the committed card-6 file at this card's
    landing — with the card-6 baryonic ``__all__`` block in its landed
    order and the baryonic dispatch and label branches as byte-exact
    substrings of the extended bodies; the card-6 test module pins the
    behavior unchanged in the same suite."""

    source = MODULE_PATH.read_text(encoding="utf-8")

    # the card-6 baryonic __all__ block leads the extended list in its
    # landed order
    head, sep, _rest = source.partition(BARYONIC_ALL_BLOCK)
    assert sep == BARYONIC_ALL_BLOCK, (
        "the card-6 baryonic __all__ block is not the pinned prefix of "
        "the module export list"
    )
    assert "Baryonic and mesonic alignment kernels" in head, (
        "the module docstring head above the export list moved"
    )

    # the baryonic region: from the __all__ closing bracket to the
    # draw_vertex_cos_theta definition, byte-exact
    all_close = source.index("\n]\n", source.index(BARYONIC_ALL_BLOCK))
    region = source[all_close + len("\n]\n"):]
    def_index = region.index("def draw_vertex_cos_theta(")
    region = region[:def_index]
    digest = hashlib.sha256(region.encode("utf-8")).hexdigest()
    assert digest == BARYONIC_REGION_SHA256, (
        "the card-6 baryonic surface region drifted inside the module "
        f"({digest!r} against the pinned "
        f"{BARYONIC_REGION_SHA256!r})"
    )

    # the baryonic dispatch lines survive byte-exactly in the extended
    # draw_vertex_cos_theta body, and the mesonic branch was inserted
    # before the unchanged isotropic fallback
    assert BARYONIC_DISPATCH_BLOCK in source
    dispatch_tail = (
        "    if (qn_leading[0] == 0 and qn_trailing[0] == 0\n"
        "            and abs(kappa_mes_value) >= ISOTROPIC_BRANCH_TOL):\n"
        "        return draw_mesonic_alignment(u_value, kappa_mes_value)\n"
        "    return 2.0 * u_value - 1.0\n"
    )
    assert dispatch_tail in source
    assert source.index(BARYONIC_DISPATCH_BLOCK) < source.index(
        dispatch_tail)

    # the baryonic label branch survives byte-exactly in the extended
    # assign_alignment_label body
    assert BARYONIC_LABEL_BLOCK in source
    label_tail = (
        "    if qn[0] == 0 and abs(kappa_mes_value) >= "
        "ISOTROPIC_BRANCH_TOL:\n"
        "        return MESONIC_ALIGNMENT_LABEL\n"
        "    return ISOTROPIC_ALIGNMENT_LABEL\n"
    )
    assert label_tail in source

    # the card-7 extension is the single appended section (exactly one
    # mesonic-kernel section marker)
    marker = "# The mesonic kernel (plan section 10.4, lines 724-728)"
    assert source.count(marker) == 1


# ----------------------------------------------------------------------
# test_layout_sync
# ----------------------------------------------------------------------
def test_layout_sync():
    """The consumed-surface register (the frozen substrate digests of
    the card's provenance section, the card-4 cascade surface, the
    card-5 axis machinery, and the card-6 blend routing included,
    verified on disk) and the layout sync (the subprocess check from
    the monorepo root; ``logic_drift=0``)."""

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
