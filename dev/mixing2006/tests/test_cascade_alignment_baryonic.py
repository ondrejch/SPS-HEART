"""Card tests for the baryonic alignment kernel.

Round 4, card 6 (``TASK-20260920-r4-baryonic-alignment-kernel``; plan
section 22 task 23, plan section 10.4, lines 716-732).  The card's
pre-registered deterministic tests against the closed-form identities,
the pinned quantile grids, the committed catalog
(``config/v2_singlet/resonance_catalog.json``), and the continuum
family configuration (``config/v2_singlet/continuum_resonances.json``):

- ``test_normalization``: the kernel normalization agrees with the
  analytic ``2 sinh(kappa)/kappa`` and with a Gauss-Legendre
  quadrature cross-check within 1e-12 at kappa in {0, 0.5, 1, 2, 4};
  the kappa -> 0 limit is continuous to 1e-12.
- ``test_inverse_cdf_round_trip``: the CDF of a sampled cos(theta)
  reproduces the uniform quantile within 1e-9 on the pinned quantile
  grid at kappa in {0.5, 1, 2, 4}.
- ``test_moments``: the analytic mean ``coth(kappa) - 1/kappa``
  matches quadrature within 1e-12; the sample mean of 100,000 draws at
  kappa = 1 (pinned seed) agrees within 3 sigma of ``1/sqrt(N)``.
- ``test_isotropic_branch_bit_stable``: at kappa = 0 (and inside the
  pre-registered branch tolerance) the draw is bit-identical to the
  Round 2 isotropic cos(theta) draw ``2u - 1`` — the card-4 baseline
  the R43 ablation (card 14) stands on — on a pinned seed.
- ``test_kernel_selection``: the angular law on pinned fixtures, the
  even/odd selection (the baryonic law carries the odd forward part
  along the inherited axis; the axis-even mesonic law is card 7's
  territory and non-baryonic daughters keep the isotropic draw), and
  the checked-never-repaired error paths.
- ``test_alignment_label_assignment``: the card-4
  ``ISOTROPIC_ALIGNMENT_LABEL`` placeholder is replaced by the kernel's
  ``BARYONIC_ALIGNMENT_LABEL`` on the baryonic line at the default
  kappa, with the fallback to the placeholder at the isotropic branch
  and off the baryonic line; the label and the drawn law agree at the
  vertex, and on drawn card-4 daughters the assignment replaces the
  placeholder without touching any other daughter field.
- ``test_draw_budget``: the baryonic attachment consumes exactly one
  uniform (the cos(theta) draw) in place of the Round 2 cos(theta)
  draw; the per-vertex budget of card 4 stays 1 channel + 2 kinematics
  (counting proxies through ``draw_decay``), the azimuth draw and the
  substream derivation order are unchanged (the card-4 draw-site
  source pin), and at ``kappa_B = 0`` the attached vertex value is
  bit-identical to the card-4 isotropic draw.
- ``test_zero_draws``: a pinned cascade of 1,000 kernel operations
  consumes zero random numbers (the generator state is byte-identical
  before and after) and the module source carries no random machinery
  (the zero-draw contract beyond the replaced draw).
- ``test_layout_sync``: the consumed-surface register (the frozen
  substrate digests of the card's provenance section, the card-4 and
  card-5 surfaces included, verified on disk) and the layout sync (the
  subprocess check from the monorepo root; ``logic_drift=0``).

plus the acceptance-(v) freeze re-pin check ``test_freeze_manifest_repin``.

Hermetic: the committed config and catalog files, numpy, the standard
library, and the vendored ``na49_mixing`` modules only; no event
generation, no external tables, no Monte Carlo run.
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
#: for the pinned statistical streams.
MASTER_SEED = 20260920

#: The pre-registered kappa values of the acceptance criteria (i)-(ii).
KAPPA_GRID = (0.0, 0.5, 1.0, 2.0, 4.0)
ROUND_TRIP_KAPPAS = (0.5, 1.0, 2.0, 4.0)

#: The pinned quantile grid: the regular 65-point grid plus the pinned
#: edge and deep-tail quantiles of the inverse-CDF fidelity check.
QUANTILE_GRID = tuple(
    [j / 64 for j in range(65)]
    + [1e-9, 1.0 - 1e-9, 0.5 ** 20, 1.0 - 0.5 ** 20, 1.0 / 3.0, 2.0 / 3.0]
)

#: The pinned (cosine, kappa) fixtures of the angular-law check.
DENSITY_FIXTURES = tuple(
    (c, k)
    for k in (0.0, 0.5, 1.0, 2.0, 4.0, -1.0)
    for c in (-1.0, -0.7, -0.25, 0.0, 0.1, 0.5, 0.9, 1.0)
)

#: The pinned baryonic-line vertex fixture: the 8 GeV baryonic root
#: keeps the realized continuum daughter inside the family support, so
#: the pinned continuation has an open line (the card-4 mass rule
#: realizes the daughter at the budget edge minus the family nominal
#: width).
ROOT_FAMILY = "baryonic_np"
ROOT_MASS = 8.0
ROOT_QN = (1, 1, 0)
BARYONIC_QN = (1, 1, 0)
STRANGE_BARYONIC_QN = (1, 0, -1)
MESONIC_QN = (0, 0, 0)
VERTEX_EVENT = 7

#: The Gauss-Legendre quadrature of the independent reference
#: (800 nodes; the kappa grid's worst quadrature error is below 1e-14,
#: far inside the pre-registered 1e-12).
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(800)

#: The frozen substrate of the card's provenance section (byte-unchanged
#: surfaces, verified on disk at this card's landing): card 4's cascade
#: surface and card 5's axis machinery (consumed read-only), the
#: builder, the card-6 blend routing, the card-2 engine, the committed
#: catalog, the continuum families configuration, and the dual-tree
#: R32 driver.
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

#: This card's freeze re-pin entries (the new kernel module, the
#: additive package exports, the new test file, and the four consumed
#: files whose hard inventory-count pins the card-mandated re-pin
#: bumps: the card-3 audit register, the card-4 register, the card-5
#: solver test, and the card-5 register).  The manifest itself is
#: excluded from its own inventory by path and appears only in the
#: freeze exception's file list.
FREEZE_CARD = "TASK-20260920-r4-baryonic-alignment-kernel"
# 458 entries at the card-5 register landing; this card adds its kernel
# module and its test file (+2 entries) for the card-mandated re-pin
# (458 -> 460) and re-pins the four consumed count pins with the bump
# attributed to this card's test file (test_cascade_alignment_baryonic);
# card 6's independent test-author register
# (test_r4_cascade_alignment_baryonic_contract_pins.py) re-pins the
# inventory additively for its own file (460 -> 461, +1 entry), so
# this count constant is re-pinned again; card 7 of the round
# (TASK-20260920-r4-mesonic-alignment-kernel) re-pins the inventory
# additively for its new mesonic-kernel test file
# (test_cascade_alignment_mesonic.py; the card-mandated re-pin;
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
MANIFEST_REL = ("dev/mixing2006/config/v2_singlet/"
                "legacy_v1_manifest.json")
MODULE_REL = "dev/mixing2006/na49_mixing/cascade_alignment.py"
INIT_REL = "dev/mixing2006/na49_mixing/__init__.py"
MY_TEST_REL = "dev/mixing2006/tests/test_cascade_alignment_baryonic.py"
CARD3_REGISTER_REL = ("dev/mixing2006/tests/"
                      "test_r4_audit_blend_path_contract_pins.py")
CARD4_REGISTER_REL = ("dev/mixing2006/tests/"
                      "test_r4_cascade_decay_contract_pins.py")
CARD5_SOLVER_TEST_REL = "dev/mixing2006/tests/test_cascade_axes.py"
CARD5_REGISTER_REL = ("dev/mixing2006/tests/"
                      "test_r4_cascade_axes_contract_pins.py")
FREEZE_INVENTORY_FILES = (
    MODULE_REL,
    INIT_REL,
    MY_TEST_REL,
    CARD3_REGISTER_REL,
    CARD4_REGISTER_REL,
    CARD5_SOLVER_TEST_REL,
    CARD5_REGISTER_REL,
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
    """The independent CDF reference of the baryonic law: the
    normalized integral of exp(kappa x) from -1 to *c* (the isotropic
    uniform reference inside the branch tolerance)."""

    if abs(kappa) < ISOTROPIC_BRANCH_TOL:
        return 0.5 * (c + 1.0)
    return (math.exp(kappa * c) - math.exp(-kappa)) / (
        math.exp(kappa) - math.exp(-kappa))


def _reference_mean(kappa: float) -> float:
    """The independent analytic mean of the baryonic law,
    ``coth(kappa) - 1/kappa`` (0 at the isotropic branch)."""

    if abs(kappa) < ISOTROPIC_BRANCH_TOL:
        return 0.0
    return 1.0 / math.tanh(kappa) - 1.0 / kappa


def _quadrature(kappa: float, weight_at_node) -> float:
    """The Gauss-Legendre quadrature of ``weight_at_node(c)`` over
    [-1, 1] on the fixed 800-node grid."""

    values = weight_at_node(_GL_NODES)
    return float(np.sum(_GL_WEIGHTS * values))


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
    """The closed decay_channel/decay_kinematics substream generators of
    one (master seed, replica 0, event index) triple (the round 2 card 9
    and round 3 card 8 convention)."""

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
# test_normalization (acceptance (i))
# ----------------------------------------------------------------------
def test_normalization():
    """The kernel normalization agrees with the analytic
    ``2 sinh(kappa)/kappa`` and with the quadrature cross-check within
    1e-12 at kappa in {0, 0.5, 1, 2, 4}; the kappa -> 0 limit is
    continuous to 1e-12."""

    for kappa in KAPPA_GRID:
        got = baryonic_normalization(kappa)
        if kappa == 0.0:
            analytic = 2.0
        else:
            analytic = 2.0 * math.sinh(kappa) / kappa
        assert abs(got - analytic) <= 1e-12, (
            f"kappa={kappa}: normalization {got!r} misses the analytic "
            f"{analytic!r}"
        )
        quad = _quadrature(kappa, lambda c, k=kappa: np.exp(k * c))
        assert abs(got - quad) <= 1e-12, (
            f"kappa={kappa}: normalization {got!r} misses the "
            f"quadrature {quad!r}"
        )

    # the density integrates to the normalization on the same grid
    for kappa in KAPPA_GRID:
        quad = _quadrature(kappa, lambda c, k=kappa: np.exp(k * c))
        assert abs(quad - baryonic_normalization(kappa)) <= 1e-12

    # the kappa -> 0 limit is continuous to 1e-12 across the branch
    assert baryonic_normalization(0.0) == 2.0
    for kappa in (1e-12, -1e-12, 1e-10, -1e-10, 1e-9, 9.999e-9, 1e-8):
        assert abs(baryonic_normalization(kappa) - 2.0) <= 1e-12, (
            f"kappa={kappa}: the limit is discontinuous"
        )
    # the normalization is even in kappa (structural sanity)
    for kappa in (0.5, 1.0, 2.0, 4.0):
        assert baryonic_normalization(-kappa) == baryonic_normalization(kappa)


# ----------------------------------------------------------------------
# test_inverse_cdf_round_trip (acceptance (ii), CDF leg)
# ----------------------------------------------------------------------
def test_inverse_cdf_round_trip():
    """The CDF of a sampled cos(theta) reproduces the uniform quantile
    within 1e-9 on the pinned quantile grid at kappa in
    {0.5, 1, 2, 4}; the endpoints land on the cosine range."""

    for kappa in ROUND_TRIP_KAPPAS:
        for u in QUANTILE_GRID:
            c = draw_baryonic_alignment(u, kappa)
            assert -1.0 <= c <= 1.0, (
                f"kappa={kappa}, u={u!r}: the draw left [-1, 1] "
                f"({c!r})"
            )
            back = _reference_cdf(c, kappa)
            assert abs(back - u) <= 1e-9, (
                f"kappa={kappa}, u={u!r}: the round trip misses by "
                f"{abs(back - u)!r}"
            )
    # the endpoints: quantile 0 maps to cos(theta) = -1, quantile 1 to
    # +1 (within the same fidelity bound)
    for kappa in ROUND_TRIP_KAPPAS:
        assert abs(draw_baryonic_alignment(0.0, kappa) + 1.0) <= 1e-12
        assert abs(draw_baryonic_alignment(1.0, kappa) - 1.0) <= 1e-12


# ----------------------------------------------------------------------
# test_moments (acceptance (ii), sample-mean leg)
# ----------------------------------------------------------------------
def test_moments():
    """The analytic mean ``coth(kappa) - 1/kappa`` matches the
    quadrature within 1e-12; the sample mean of 100,000 draws at
    kappa = 1 (pinned seed) agrees within 3 sigma of ``1/sqrt(N)``,
    the bounded-variable bound of the pre-registered acceptance."""

    for kappa in ROUND_TRIP_KAPPAS:
        analytic = _reference_mean(kappa)
        quad = _quadrature(
            kappa, lambda c, k=kappa: c * np.exp(k * c)) / _quadrature(
            kappa, lambda c, k=kappa: np.exp(k * c))
        assert abs(analytic - quad) <= 1e-12, (
            f"kappa={kappa}: the analytic mean {analytic!r} misses the "
            f"quadrature {quad!r}"
        )
        # the baryonic line is biased forward along the inherited axis
        assert analytic > 0.0

    # the isotropic limit of the mean is 0 (coth(kappa) - 1/kappa -> 0)
    assert _reference_mean(0.0) == 0.0
    assert abs(_reference_mean(1e-6)) < 1e-6

    n_draws = 100_000
    stream = np.random.default_rng(MASTER_SEED)
    quantiles = stream.random(n_draws)
    sample_mean = math.fsum(
        draw_baryonic_alignment(float(u), BARYONIC_KAPPA_DEFAULT)
        for u in quantiles
    ) / n_draws
    analytic = _reference_mean(BARYONIC_KAPPA_DEFAULT)
    bound = 3.0 / math.sqrt(n_draws)
    assert abs(sample_mean - analytic) <= bound, (
        f"the 100,000-draw sample mean {sample_mean!r} misses the "
        f"analytic {analytic!r} beyond the pre-registered 3 sigma of "
        f"1/sqrt(N) = {bound!r}"
    )


# ----------------------------------------------------------------------
# test_isotropic_branch_bit_stable (acceptance (iii))
# ----------------------------------------------------------------------
def test_isotropic_branch_bit_stable():
    """At kappa = 0 (and inside the pre-registered branch tolerance)
    the draw is bit-identical to the Round 2 isotropic cos(theta) draw
    ``2u - 1`` on a pinned seed — the card-4 baseline the R43 ablation
    (card 14, acceptance (ii)) stands on."""

    stream = np.random.default_rng(MASTER_SEED + 62)
    pinned_us = [float(u) for u in stream.random(1_000)]
    for u in pinned_us + list(QUANTILE_GRID):
        for kappa in (0.0, -0.0, 0.5 * ISOTROPIC_BRANCH_TOL,
                      -0.5 * ISOTROPIC_BRANCH_TOL, 1e-9, -1e-9):
            got = draw_baryonic_alignment(u, kappa)
            assert got == 2.0 * u - 1.0, (
                f"kappa={kappa}, u={u!r}: the isotropic branch is not "
                "bit-stable with the Round 2 isotropic draw"
            )

    # the card-4 draw site computes exactly this expression (the same
    # float expression on the same quantile is the bit-identity)
    card4_src = CARD4_PATH.read_text(encoding="utf-8")
    assert "cos_theta = 2.0 * u_cos - 1.0" in card4_src, (
        "the card-4 isotropic cos(theta) draw site is not the pinned "
        "expression the isotropic branch must reproduce bit for bit"
    )


# ----------------------------------------------------------------------
# test_kernel_selection (the even/odd selection and the angular law)
# ----------------------------------------------------------------------
def test_kernel_selection():
    """The angular law on the pinned fixtures (exp(kappa c) exactly);
    the even/odd selection (the baryonic law carries the odd forward
    part along the inherited axis, the normalization is even in kappa,
    and the draw is mirrored under kappa -> -kappa); the daughter
    selection follows baryon number; the error paths are
    checked-never-repaired."""

    # -- the angular law, exact on the pinned fixtures -----------------
    for c, kappa in DENSITY_FIXTURES:
        got = baryonic_kernel_density(c, kappa)
        assert got == math.exp(kappa * c), (
            f"c={c}, kappa={kappa}: the density is not the pinned law"
        )
        assert got > 0.0 and math.isfinite(got)
    assert baryonic_kernel_density(0.3, 0.0) == 1.0

    # -- the even/odd selection ----------------------------------------
    # the baryonic law is not even in cos(theta): the forward/backward
    # ratio at +/-c carries exp(2 kappa c) (the odd part is the
    # alignment)
    for kappa in (0.5, 1.0, 2.0, 4.0):
        c = 0.5
        ratio = (baryonic_kernel_density(c, kappa)
                 / baryonic_kernel_density(-c, kappa))
        assert abs(ratio - math.exp(2.0 * kappa * c)) <= 1e-12
        assert ratio != 1.0
    # the mean is forward along the inherited axis (checked in
    # test_moments); the normalization is even in kappa
    for kappa in (0.5, 1.0, 2.0, 4.0):
        assert baryonic_normalization(-kappa) == baryonic_normalization(kappa)
    # the draw is mirrored under kappa -> -kappa against the reflected
    # quantile (the CDF identity CDF_{-k}(c) = 1 - CDF_k(-c))
    for kappa in (0.5, 1.0, 2.0, 4.0):
        for u in QUANTILE_GRID:
            mirrored = draw_baryonic_alignment(u, -kappa)
            expect = -draw_baryonic_alignment(1.0 - u, kappa)
            assert abs(mirrored - expect) <= 1e-12, (
                f"kappa={kappa}, u={u!r}: the mirrored draw identity "
                f"fails ({mirrored!r} vs {expect!r})"
            )
    # a daughter off the baryonic line keeps the isotropic draw (card
    # 7's even-law territory replaces it, not this card)
    for u in (0.0, 0.25, 0.5, 0.75, 1.0 - 1e-9):
        assert draw_vertex_cos_theta(
            u, (MESONIC_QN, MESONIC_QN)) == 2.0 * u - 1.0

    # -- the daughter selection follows baryon number -------------------
    assert continues_baryon_line(BARYONIC_QN) is True
    assert continues_baryon_line(STRANGE_BARYONIC_QN) is True
    assert continues_baryon_line((-1, -1, 0)) is True
    assert continues_baryon_line(MESONIC_QN) is False
    assert continues_baryon_line((0, 1, 0)) is False

    # -- checked, never repaired ----------------------------------------
    with pytest.raises(ValueError):
        baryonic_kernel_density(1.5, 1.0)  # outside [-1, 1]
    with pytest.raises(ValueError):
        baryonic_kernel_density(float("nan"), 1.0)
    with pytest.raises(ValueError):
        baryonic_kernel_density(0.5, float("inf"))
    with pytest.raises(ValueError):
        baryonic_normalization(float("nan"))
    with pytest.raises(ValueError):
        draw_baryonic_alignment(-0.1, 1.0)
    with pytest.raises(ValueError):
        draw_baryonic_alignment(1.1, 1.0)
    with pytest.raises(ValueError):
        draw_baryonic_alignment(float("nan"), 1.0)
    with pytest.raises(ValueError):
        draw_baryonic_alignment(0.5, float("nan"))
    with pytest.raises(ValueError):
        continues_baryon_line((1, 1))
    with pytest.raises(ValueError):
        draw_vertex_cos_theta(0.5, (BARYONIC_QN, (1, 0, 0)))
    with pytest.raises(ValueError):
        draw_vertex_cos_theta(0.5, (MESONIC_QN, MESONIC_QN), float("nan"))


# ----------------------------------------------------------------------
# test_alignment_label_assignment (the label bookkeeping)
# ----------------------------------------------------------------------
def test_alignment_label_assignment(catalog, fams):
    """The card-4 ``ISOTROPIC_ALIGNMENT_LABEL`` placeholder is replaced
    by the kernel's ``BARYONIC_ALIGNMENT_LABEL`` on the baryonic line
    at the default kappa, with the fallback to the placeholder at the
    isotropic branch and off the baryonic line; the label and the drawn
    law agree at the vertex; on drawn card-4 daughters the assignment
    replaces the placeholder without touching any other daughter
    field."""

    # -- the per-daughter assignment ------------------------------------
    assert assign_alignment_label(
        BARYONIC_QN) == BARYONIC_ALIGNMENT_LABEL
    assert assign_alignment_label(
        STRANGE_BARYONIC_QN) == BARYONIC_ALIGNMENT_LABEL
    assert assign_alignment_label(MESONIC_QN) == ISOTROPIC_ALIGNMENT_LABEL
    # the isotropic branch: the label falls back with the drawn law
    assert assign_alignment_label(
        BARYONIC_QN, kappa_baryonic=0.0) == ISOTROPIC_ALIGNMENT_LABEL
    assert assign_alignment_label(
        BARYONIC_QN, kappa_baryonic=-0.0) == ISOTROPIC_ALIGNMENT_LABEL
    assert assign_alignment_label(
        BARYONIC_QN, kappa_baryonic=0.5 * ISOTROPIC_BRANCH_TOL
    ) == ISOTROPIC_ALIGNMENT_LABEL
    assert assign_alignment_label(
        BARYONIC_QN, kappa_baryonic=ISOTROPIC_BRANCH_TOL
    ) == BARYONIC_ALIGNMENT_LABEL
    with pytest.raises(ValueError):
        assign_alignment_label((1, 1))
    with pytest.raises(ValueError):
        assign_alignment_label(BARYONIC_QN, float("nan"))

    # -- the label and the drawn law agree at the vertex ----------------
    pair_baryon_leading = (BARYONIC_QN, MESONIC_QN)
    pair_baryon_trailing = (MESONIC_QN, BARYONIC_QN)
    pair_no_baryon = (MESONIC_QN, MESONIC_QN)
    for u in QUANTILE_GRID:
        for pair, kappa in (
            (pair_baryon_leading, BARYONIC_KAPPA_DEFAULT),
            (pair_baryon_trailing, BARYONIC_KAPPA_DEFAULT),
            (pair_no_baryon, BARYONIC_KAPPA_DEFAULT),
            (pair_baryon_leading, 0.0),
            (pair_baryon_trailing, 0.0),
        ):
            labels = [
                assign_alignment_label(qn, kappa_baryonic=kappa)
                for qn in pair
            ]
            # the per-daughter labels follow the line selection and the
            # branch state
            for qn, label in zip(pair, labels):
                expect = (BARYONIC_ALIGNMENT_LABEL
                          if qn[0] != 0 and abs(kappa) >= ISOTROPIC_BRANCH_TOL
                          else ISOTROPIC_ALIGNMENT_LABEL)
                assert label == expect, (
                    f"qn={qn}, kappa={kappa}: the label bookkeeping "
                    f"{label!r} disagrees with the selection"
                )
            # the vertex cos(theta) follows the baryonic daughter when
            # its label stands (mirrored back-to-back when it trails),
            # the isotropic draw otherwise
            draw = draw_vertex_cos_theta(u, pair, kappa_baryonic=kappa)
            if BARYONIC_ALIGNMENT_LABEL in labels:
                if pair[0][0] != 0:
                    assert draw == draw_baryonic_alignment(u, kappa), (
                        f"u={u!r}, kappa={kappa}: the leading baryonic "
                        "label stands without the forward kernel draw"
                    )
                else:
                    assert draw == -draw_baryonic_alignment(u, kappa), (
                        f"u={u!r}, kappa={kappa}: the trailing baryonic "
                        "label stands without the mirrored kernel draw"
                    )
            else:
                assert draw == 2.0 * u - 1.0, (
                    f"u={u!r}, kappa={kappa}: the isotropic label "
                    "stands without the isotropic draw"
                )

    # -- the card-4 integration: the labels on drawn daughters ---------
    root = _root_state(ROOT_FAMILY, ROOT_MASS, ROOT_QN)
    channels = enumerate_channels(root, catalog, fams)
    pool = [c for c in channels if c.kind == KIND_CONTINUUM
            and ROOT_FAMILY in c.daughters]
    assert pool
    rng_channel = _CountingRng(MASTER_SEED + 11)
    rng_kinematics = _CountingRng(MASTER_SEED + 12)
    _, daughters = draw_decay(
        root, pool, catalog, fams, rng_channel, rng_kinematics)
    assert rng_channel.calls == 1 and rng_kinematics.calls == 2
    baryon = next(d for d in daughters if int(d.qn[0]) != 0)
    meson = next(d for d in daughters if int(d.qn[0]) == 0)
    # card 4 copies the parent label verbatim; this card's assignment
    # replaces it at the call site
    for daughter in daughters:
        assert daughter.alignment_label == ISOTROPIC_ALIGNMENT_LABEL
    filled_baryon = replace(
        baryon, alignment_label=assign_alignment_label(baryon.qn))
    filled_meson = replace(
        meson, alignment_label=assign_alignment_label(meson.qn))
    assert filled_baryon.alignment_label == BARYONIC_ALIGNMENT_LABEL
    assert filled_meson.alignment_label == ISOTROPIC_ALIGNMENT_LABEL
    # the replaced states keep every other field of the card-4 daughters
    for filled, daughter in ((filled_baryon, baryon),
                             (filled_meson, meson)):
        assert filled.momentum == daughter.momentum
        assert filled.mass == daughter.mass
        assert filled.qn == daughter.qn
        assert filled.family_id == daughter.family_id
        assert filled.axis == daughter.axis
        assert filled.parent is daughter.parent
        assert filled.exchange_ancestry == daughter.exchange_ancestry
        assert filled.depth == daughter.depth
    # the ablation baseline: at kappa_B = 0 every daughter falls back
    for daughter in daughters:
        assert assign_alignment_label(
            daughter.qn, kappa_baryonic=0.0) == ISOTROPIC_ALIGNMENT_LABEL


# ----------------------------------------------------------------------
# test_draw_budget (acceptance (iv))
# ----------------------------------------------------------------------
def test_draw_budget(catalog, fams):
    """The baryonic attachment consumes exactly one uniform (the
    cos(theta) draw) in place of the Round 2 cos(theta) draw; the
    per-vertex budget of card 4 stays 1 channel + 2 kinematics
    (counting proxies through ``draw_decay``); the azimuth draw and the
    substream derivation order are unchanged (the card-4 draw-site
    source pin); at ``kappa_B = 0`` the attached vertex value is
    bit-identical to the card-4 isotropic draw."""

    root = _root_state(ROOT_FAMILY, ROOT_MASS, ROOT_QN)
    channels = enumerate_channels(root, catalog, fams)
    pool = [c for c in channels if c.kind == KIND_CONTINUUM
            and ROOT_FAMILY in c.daughters]
    assert pool

    # -- the per-vertex budget of card 4 is unchanged -------------------
    rng_channel = _CountingRng(MASTER_SEED + 21)
    rng_kinematics = _CountingRng(MASTER_SEED + 22)
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
    baryon_slots = sum(1 for qn in pair if qn[0] != 0)
    assert baryon_slots == 1, (
        "the pinned baryonic-line vertex carries exactly one "
        "baryon-carrying daughter"
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
    # replaces the isotropic value, the azimuth is untouched
    aligned_cos = draw_vertex_cos_theta(u_cos, pair)
    assert aligned_cos == draw_baryonic_alignment(
        u_cos, BARYONIC_KAPPA_DEFAULT)
    assert -1.0 <= aligned_cos <= 1.0
    phi = 2.0 * math.pi * u_phi
    assert phi == 2.0 * math.pi * u_phi  # the unchanged azimuth draw

    # the isotropic limit at the pre-registered kappa_B = 0: the
    # attached vertex value is bit-identical to the card-4 isotropic
    # draw (the R43 ablation baseline, card 14 acceptance (ii))
    assert draw_vertex_cos_theta(
        u_cos, pair, kappa_baryonic=0.0) == card4_cos

    # the default-kappa attachment differs from the isotropic baseline
    # (the alignment is active at the pre-registered default)
    assert aligned_cos != card4_cos

    # -- the card-4 draw site (the source pin of the attachment) -------
    card4_src = CARD4_PATH.read_text(encoding="utf-8")
    pin_cos = "u_cos = float(rng_kinematics.random())"
    pin_phi = "u_phi = float(rng_kinematics.random())"
    pin_law = "cos_theta = 2.0 * u_cos - 1.0"
    pin_azimuth = "phi = 2.0 * math.pi * u_phi"
    for fragment in (pin_cos, pin_phi, pin_law, pin_azimuth):
        assert fragment in card4_src, (
            f"the card-4 draw-site pin {fragment!r} is missing"
        )
    assert card4_src.index(pin_cos) < card4_src.index(pin_phi) < \
        card4_src.index(pin_azimuth), (
        "the substream derivation order moved: the cos(theta) draw "
        "must stay the first kinematics uniform and the azimuth the "
        "second"
    )


# ----------------------------------------------------------------------
# test_zero_draws (the zero-draw contract beyond the replaced draw)
# ----------------------------------------------------------------------
def test_zero_draws():
    """A pinned cascade of 1,000 kernel operations consumes zero random
    numbers (the generator state is byte-identical before and after)
    and the module source carries no random machinery."""

    rng = np.random.default_rng(MASTER_SEED + 71)
    state_before = rng.bit_generator.state

    kappas = (BARYONIC_KAPPA_DEFAULT, 0.0, 0.5, 2.0, 4.0, -1.0)
    pairs = (
        (BARYONIC_QN, MESONIC_QN),
        (MESONIC_QN, BARYONIC_QN),
        (MESONIC_QN, MESONIC_QN),
    )
    operations = 0
    for step in range(1_000):
        u = 0.5 ** (step % 53)
        c = -1.0 + (step % 41) / 20.0
        kappa = kappas[step % len(kappas)]
        pair = pairs[step % len(pairs)]
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
            assign_alignment_label(pair[step % 2], kappa_baryonic=kappa)
        operations += 1
    assert operations == 1_000

    state_after = rng.bit_generator.state
    assert state_after == state_before, (
        "the alignment kernel consumed random numbers: the generator "
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
    assert not hits, f"the alignment module carries random machinery: {hits}"


# ----------------------------------------------------------------------
# test_layout_sync
# ----------------------------------------------------------------------
def test_layout_sync():
    """The consumed-surface register (the frozen substrate digests of
    the card's provenance section, the card-4 cascade surface and the
    card-5 axis machinery included, verified on disk) and the layout
    sync (the subprocess check from the monorepo root;
    ``logic_drift=0``)."""

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
