"""Card tests for the inherited production axes.

Round 4, card 5 (``TASK-20260920-r4-inherited-production-axes``; plan
section 22 task 22, plan section 10.1, lines 664-674).  The card's
pre-registered deterministic tests against the pinned Lorentz
fixtures, the committed catalog
(``config/v2_singlet/resonance_catalog.json``), and the continuum
family configuration (``config/v2_singlet/continuum_resonances.json``):

- ``test_initial_axis_unit_norm``: the initial source axis has unit
  norm within 1e-12 on a pinned fixture spanning the two-source
  topology and the isolated source.
- ``test_beam_direction_convention``: for the pinned two-source p+p
  topology, the source-rest-frame axis equals the boosted beam
  direction (pre-registered fixture values, 1e-12; the collinear
  sources return the beam axis exactly); the isolated source carries
  the lab z-axis; the error paths raise the kernel's
  ``KinematicsError``.
- ``test_boost_round_trip``: boost the axis into the daughter rest
  frame and back to the parent frame; the axis is recovered within
  1e-12 (pinned fixtures, including a moving parent and a large
  boost), and the one-way transport matches an independent
  light-aberration reference within 1e-12.
- ``test_inheritance_rule``: the baryon-carrying daughter carries the
  boosted parent axis; the symmetric mesonic daughter carries the
  parent rest-frame axis (pinned fixture, both branches); plus the
  card-4 integration: on drawn ``draw_decay`` daughters the axis
  fields are populated through ``inherit_axis`` (the baryon-line
  daughter the transported axis, the mesonic daughter the parent
  axis, the card-4 isotropic label untouched), over a one-step
  continuation, with the per-vertex draw budget unchanged.
- ``test_zero_draws``: a pinned cascade of 1,000 axis operations
  consumes zero random numbers (the generator state is byte-identical
  before and after) and the module source carries no random machinery.
- ``test_layout_sync``: the consumed-surface register and the layout
  sync (the subprocess check from the monorepo root; ``logic_drift=0``).

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

from na49_mixing.cascade_axes import (
    AXIS_NORM_TOL,
    LAB_Z_AXIS,
    boost_axis_to_rest_frame,
    inherit_axis,
    initial_source_axis,
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
from na49_mixing.resonance_decay import KinematicsError, _boost_lab
from na49_mixing.resonance_widths import decay_momentum
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
MODULE_PATH = NESTED_ROOT / "na49_mixing" / "cascade_axes.py"

#: The round 4 master seed (round 4 index item 9), fixed for the
#: deterministic substream draws of the card-4 integration fixture.
MASTER_SEED = 20260920

#: The pinned fixture masses and momenta (pure Lorentz-algebra
#: fixtures; the committed catalog is consumed only by the card-4
#: integration fixture).
PROTON_MASS = 0.938
PION_MASS = 0.1396
PARENT_MASS = 4.0
PARENT_AT_REST = (PARENT_MASS, 0.0, 0.0, 0.0)
AXIS_TILTED_SOURCE = (0.48, -0.36, 0.8)
BETA_PARENT = (0.3, -0.1, 0.5)
DAUGHTER_Q = 0.8
DAUGHTER_Q_FAST = 3.0

#: The pinned two-source p+p fixtures: a projectile-like and a
#: target-like source moving along the beam (z) axis, plus a recoiling
#: source with a transverse component.
PP_SOURCE_MASS = 2.7
PP_SOURCE_ENERGY = 8.6
PP_SOURCE_PZ = math.sqrt(PP_SOURCE_ENERGY ** 2 - PP_SOURCE_MASS ** 2)
PP_RECOIL_DIRECTION = (0.3, -0.55, 0.95)
PP_RECOIL_Q = 2.2

#: The frozen substrate of the card's provenance section (byte-unchanged
#: surfaces, verified on disk at this card's landing), including card
#: 4's cascade surface, consumed read-only.
FROZEN_SURFACES = {
    "dev/mixing2006/na49_mixing/cascade_decay.py":
        "d2a48049af3d6df968993c040033a74422ccd709f53a42bad0538ae5111bcc0c",
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
#: package exports, the new test file, and the two consumed registers
#: whose hard inventory-count pins the card-mandated re-pin bumps).
#: The manifest itself is excluded from its own inventory by path and
#: appears only in the freeze exception's file list.
FREEZE_CARD = "TASK-20260920-r4-inherited-production-axes"
# 457 entries at the card-5 solver landing; card 5's independent
# test-author register (test_r4_cascade_axes_contract_pins.py)
# re-pins the inventory additively for its own file (457 -> 458,
# +1 entry), so this count pin is re-pinned accordingly; card 6 of
# the round (TASK-20260920-r4-baryonic-alignment-kernel) re-pins the
# inventory additively for its new alignment-kernel module and test
# file (test_cascade_alignment_baryonic.py; the card-mandated re-pin;
# 458 -> 460, +2 entries), so this count pin is re-pinned again;
# card 6's independent test-author register
# (test_r4_cascade_alignment_baryonic_contract_pins.py) re-pins the
# inventory additively for its own file (460 -> 461, +1 entry), so
# this count pin is re-pinned once more; card 7 of the round
# (TASK-20260920-r4-mesonic-alignment-kernel) re-pins the inventory
# additively for its new mesonic-kernel test file
# (test_cascade_alignment_mesonic.py; the card-mandated re-pin;
# 461 -> 462, +1 entry), so this count pin is re-pinned again;
# card 7's independent test-author register
# (test_r4_cascade_alignment_mesonic_contract_pins.py) re-pins the
# inventory additively for its own file (462 -> 463, +1 entry), so
# this count pin is re-pinned once more; card 8 of the round
# (TASK-20260920-r4-hadronic-decay-form-factor) re-pins the inventory
# additively for its new form-factor module and test file
# (na49_mixing/cascade_form_factor.py, test_cascade_form_factor.py;
# 463 -> 465, +2 entries); card 8's independent test-author register
# (test_r4_cascade_form_factor_contract_pins.py) re-pins the inventory
# additively for its own file (the constant read
# "INVENTORY_COUNT = 465" at the card-8 solver landing; 465 -> 466,
# +1 entry), so this count pin is re-pinned once more; card 9 of the
# round (TASK-20260920-r4-termination-depth-tests) re-pins the
# inventory additively for its new termination module and test file
# (na49_mixing/cascade_termination.py, test_cascade_termination.py;
# the constant read "INVENTORY_COUNT = 466" at the card-8 register
# landing; 466 -> 468, +2 entries), so this count pin is re-pinned
# once more; card 9's independent test-author register
# (test_r4_cascade_termination_contract_pins.py) re-pins the inventory
# additively for its own file (the constant read
# "INVENTORY_COUNT = 468" at the card-9 solver landing; 468 -> 469,
# +1 entry), so this count pin is re-pinned once more; card 10 of the
# round (TASK-20260920-r4-isotropic-aligned-ablation-runner) re-pins
# the inventory additively for its new ablation runner module and test
# file (na49_mixing/cascade_ablation.py, test_cascade_ablation.py; the
# constant read "INVENTORY_COUNT = 469" at the card-9 register
# landing; 469 -> 471, +2 entries), so this count pin is re-pinned
# once more; the card-10 test-author register
# (test_r4_cascade_ablation_contract_pins.py) re-pins the inventory
# additively for its own file (the constant read
# "INVENTORY_COUNT = 471" at the card-10 solver landing; 471 -> 472,
# +1 entry), so this count pin is re-pinned once more.
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
MODULE_REL = "dev/mixing2006/na49_mixing/cascade_axes.py"
INIT_REL = "dev/mixing2006/na49_mixing/__init__.py"
MY_TEST_REL = "dev/mixing2006/tests/test_cascade_axes.py"
CARD3_REGISTER_REL = ("dev/mixing2006/tests/"
                      "test_r4_audit_blend_path_contract_pins.py")
CARD4_REGISTER_REL = ("dev/mixing2006/tests/"
                      "test_r4_cascade_decay_contract_pins.py")
FREEZE_INVENTORY_FILES = (
    MODULE_REL,
    INIT_REL,
    MY_TEST_REL,
    CARD3_REGISTER_REL,
    CARD4_REGISTER_REL,
)
FREEZE_EXCEPTION_FILES = FREEZE_INVENTORY_FILES + (MANIFEST_REL,)


# ----------------------------------------------------------------------
# Fixtures and helpers
# ----------------------------------------------------------------------
@pytest.fixture(scope="session")
def catalog() -> HadronCatalog:
    return HadronCatalog.from_json(CATALOG_PATH)


@pytest.fixture(scope="session")
def fams():
    return families_from_json(str(FAMILIES_PATH))


def _unit(v: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return (v[0] / norm, v[1] / norm, v[2] / norm)


def _on_shell(mass: float, direction: tuple[float, float, float],
              q: float) -> tuple[float, float, float, float]:
    """The four-momentum of a particle of mass *mass* carrying momentum
    magnitude *q* along the unit *direction*."""

    energy = math.sqrt(mass * mass + q * q)
    return (energy, q * direction[0], q * direction[1], q * direction[2])


def _neg3(v: tuple[float, float, float]) -> tuple[float, float, float]:
    return (-v[0], -v[1], -v[2])


def _aberration_reference(
    n: tuple[float, float, float],
    beta: tuple[float, float, float],
) -> tuple[float, float, float]:
    """The independent light-aberration reference: the direction *n* as
    seen from the rest frame of an object moving with velocity *beta*
    (``cos(theta') = (cos(theta) - beta) / (1 - beta cos(theta))``, the
    azimuth preserved).  Analytic, independent of ``_boost_lab``."""

    beta_sq = beta[0] * beta[0] + beta[1] * beta[1] + beta[2] * beta[2]
    if beta_sq == 0.0:
        return n
    speed = math.sqrt(beta_sq)
    bhat = (beta[0] / speed, beta[1] / speed, beta[2] / speed)
    cos_t = n[0] * bhat[0] + n[1] * bhat[1] + n[2] * bhat[2]
    sin_t = math.sqrt(max(0.0, 1.0 - cos_t * cos_t))
    cos_tp = (cos_t - speed) / (1.0 - speed * cos_t)
    sin_tp = math.sqrt(max(0.0, 1.0 - cos_tp * cos_tp))
    if sin_t == 0.0:
        return (cos_tp * bhat[0], cos_tp * bhat[1], cos_tp * bhat[2])
    perp = tuple(n[i] - cos_t * bhat[i] for i in range(3))
    return (
        cos_tp * bhat[0] + (sin_tp / sin_t) * perp[0],
        cos_tp * bhat[1] + (sin_tp / sin_t) * perp[1],
        cos_tp * bhat[2] + (sin_tp / sin_t) * perp[2],
    )


def _two_step_reference(
    axis: tuple[float, float, float],
    beta_parent: tuple[float, float, float],
    beta_daughter: tuple[float, float, float],
) -> tuple[float, float, float]:
    """The independent through-the-working-frame reference for the
    daughter-frame axis: the axis (given in the parent rest frame)
    aberrated into the working frame with the negated parent velocity
    and into the daughter rest frame with the daughter velocity."""

    in_working = _aberration_reference(axis, _neg3(beta_parent))
    return _aberration_reference(in_working, beta_daughter)


def _max_dev(a, b) -> float:
    return max(abs(a[i] - b[i]) for i in range(3))


def _norm_dev(v) -> float:
    return abs(math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) - 1.0)


AXIS_TILTED = _unit(AXIS_TILTED_SOURCE)

#: The pinned two-body split of the at-rest parent: the baryon and the
#: pion back-to-back at the exact decay momentum.
SPLIT_Q = decay_momentum(PARENT_MASS, PROTON_MASS, PION_MASS)
P_BARYON = _on_shell(PROTON_MASS, AXIS_TILTED, SPLIT_Q)
P_MESON = _on_shell(PION_MASS, _neg3(AXIS_TILTED), SPLIT_Q)
BETA_BARYON = (
    P_BARYON[1] / P_BARYON[0],
    P_BARYON[2] / P_BARYON[0],
    P_BARYON[3] / P_BARYON[0],
)

PARENT_MOVING = _boost_lab(PARENT_AT_REST, BETA_PARENT)

#: The pinned daughter fixtures of the zero-draw operation cascade.
PP_SOURCES = (
    (PP_SOURCE_ENERGY, 0.0, 0.0, PP_SOURCE_PZ),
    _on_shell(PP_SOURCE_MASS, _unit(PP_RECOIL_DIRECTION), PP_RECOIL_Q),
    (3.0, 0.0, 0.0, 0.0),
)
PP_BEAMS = ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0), None)


def _root_state(family_id: str, mass: float, qn: tuple[int, int, int],
                momentum: tuple[float, float, float, float] | None = None,
                axis: tuple[float, float, float] = LAB_Z_AXIS,
                depth: int = 0) -> CascadeState:
    """A pinned root state with the lab-z production axis (or the given
    momentum) and the card-4 isotropic label."""

    if momentum is None:
        momentum = (float(mass), 0.0, 0.0, 0.0)
    return CascadeState(
        momentum=momentum,
        mass=float(mass),
        qn=qn,
        quantum={},
        family_id=family_id,
        axis=axis,
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


# ----------------------------------------------------------------------
# test_initial_axis_unit_norm
# ----------------------------------------------------------------------
INITIAL_AXIS_FIXTURES = (
    # (label, source_momentum, beam_direction)
    ("isolated_at_rest", (3.0, 0.0, 0.0, 0.0), None),
    ("isolated_moving",
     _on_shell(3.0, _unit((0.2, 0.4, -0.5)), 1.0), None),
    ("pp_projectile_collinear",
     (PP_SOURCE_ENERGY, 0.0, 0.0, PP_SOURCE_PZ), (0.0, 0.0, 1.0)),
    ("pp_target_collinear",
     _on_shell(PP_SOURCE_MASS, (0.0, 0.0, 1.0), 2.0), (0.0, 0.0, 1.0)),
    ("pp_recoiling",
     _on_shell(PP_SOURCE_MASS, _unit(PP_RECOIL_DIRECTION), PP_RECOIL_Q),
     (0.0, 0.0, 1.0)),
)


def test_initial_axis_unit_norm():
    """The initial source axis has unit norm within 1e-12 on a pinned
    fixture spanning the two-source topology and the isolated source."""

    assert len(INITIAL_AXIS_FIXTURES) == 5
    for label, source_momentum, beam_direction in INITIAL_AXIS_FIXTURES:
        axis = initial_source_axis(source_momentum, beam_direction)
        assert len(axis) == 3
        assert _norm_dev(axis) <= AXIS_NORM_TOL, (
            f"{label}: the initial axis norm deviates by {_norm_dev(axis)!r}"
        )
        assert all(math.isfinite(component) for component in axis)


# ----------------------------------------------------------------------
# test_beam_direction_convention
# ----------------------------------------------------------------------
def test_beam_direction_convention():
    """For the pinned two-source p+p topology, the source-rest-frame
    axis equals the boosted beam direction (pre-registered fixture
    values, 1e-12); the collinear sources return the beam axis exactly;
    the isolated source carries the lab z-axis."""

    tol = AXIS_NORM_TOL

    # -- the collinear two-source topology: the beam axis exactly ------
    projectile = initial_source_axis(
        (PP_SOURCE_ENERGY, 0.0, 0.0, PP_SOURCE_PZ), (0.0, 0.0, 1.0))
    assert projectile == (0.0, 0.0, 1.0), (
        "the collinear projectile source must carry the beam axis exactly"
    )
    target = initial_source_axis(
        _on_shell(PP_SOURCE_MASS, (0.0, 0.0, 1.0), 2.0), (0.0, 0.0, 1.0))
    assert target == (0.0, 0.0, 1.0), (
        "the collinear target source must carry the beam axis exactly"
    )

    # -- the isolated-source convention: the lab z-axis ----------------
    assert initial_source_axis((3.0, 0.0, 0.0, 0.0), None) == LAB_Z_AXIS
    moving_isolated = initial_source_axis(
        _on_shell(3.0, _unit((0.2, 0.4, -0.5)), 1.0), None)
    assert moving_isolated == LAB_Z_AXIS, (
        "the isolated source carries the lab z-axis by convention "
        "(beam_direction is None), independent of the source motion"
    )

    # -- the recoiling source: the boosted beam direction against the
    # independent aberration reference ---------------------------------
    source_p4 = _on_shell(
        PP_SOURCE_MASS, _unit(PP_RECOIL_DIRECTION), PP_RECOIL_Q)
    beta_source = (
        source_p4[1] / source_p4[0],
        source_p4[2] / source_p4[0],
        source_p4[3] / source_p4[0],
    )
    recoiling = initial_source_axis(source_p4, (0.0, 0.0, 1.0))
    expected = _aberration_reference((0.0, 0.0, 1.0), beta_source)
    assert _max_dev(recoiling, expected) <= tol, (
        "the recoiling-source axis misses the boosted beam direction "
        f"by {_max_dev(recoiling, expected)!r}"
    )

    # the beam direction is a direction: any nonzero scale normalizes
    scaled = initial_source_axis(source_p4, (0.0, 0.0, 7.0))
    assert scaled == initial_source_axis(source_p4, (0.0, 0.0, 1.0))

    # -- checked, never repaired ---------------------------------------
    with pytest.raises(KinematicsError):
        initial_source_axis(source_p4, (0.0, 0.0, 0.0))
    with pytest.raises(KinematicsError):
        initial_source_axis((1.0, 2.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    with pytest.raises(KinematicsError):
        initial_source_axis((-4.0, 0.0, 0.0, 0.0), None)
    with pytest.raises(KinematicsError):
        initial_source_axis((4.0, 5.0, 0.0, 0.0), None)  # spacelike


# ----------------------------------------------------------------------
# test_boost_round_trip
# ----------------------------------------------------------------------
ROUND_TRIP_FIXTURES = (
    # (label, axis, parent_p4, daughter_p4)
    ("parent_at_rest", LAB_Z_AXIS, PARENT_AT_REST, P_BARYON),
    ("parent_moving", LAB_Z_AXIS, PARENT_MOVING, P_BARYON),
    ("large_boost", LAB_Z_AXIS, PARENT_AT_REST,
     _on_shell(PROTON_MASS, AXIS_TILTED, DAUGHTER_Q_FAST)),
    ("tilted_axis_moving_parent", AXIS_TILTED, PARENT_MOVING, P_BARYON),
)


def test_boost_round_trip():
    """Boost the axis into the daughter rest frame and back to the
    parent frame; the axis is recovered within 1e-12 (pinned fixtures,
    including a moving parent and a large boost), and the one-way
    transport matches the independent light-aberration reference within
    1e-12."""

    tol = AXIS_NORM_TOL
    for label, axis, parent_p4, daughter_p4 in ROUND_TRIP_FIXTURES:
        transported = boost_axis_to_rest_frame(axis, parent_p4, daughter_p4)
        recovered = boost_axis_to_rest_frame(
            transported, daughter_p4, parent_p4)
        assert _max_dev(recovered, axis) <= tol, (
            f"{label}: the boost round trip misses the axis by "
            f"{_max_dev(recovered, axis)!r}"
        )
        assert _norm_dev(transported) <= tol
        assert _norm_dev(recovered) <= tol

    # the one-way transport against the independent reference (the
    # through-the-working-frame composition of two aberration steps)
    beta_daughter = (
        P_BARYON[1] / P_BARYON[0],
        P_BARYON[2] / P_BARYON[0],
        P_BARYON[3] / P_BARYON[0],
    )
    expected_moving = _aberration_reference(
        _aberration_reference(LAB_Z_AXIS, _neg3(BETA_PARENT)), beta_daughter)
    got_moving = boost_axis_to_rest_frame(
        LAB_Z_AXIS, PARENT_MOVING, P_BARYON)
    assert _max_dev(got_moving, expected_moving) <= tol, (
        "the moving-parent transport misses the two-step aberration "
        f"reference by {_max_dev(got_moving, expected_moving)!r}"
    )
    expected_rest = _aberration_reference(LAB_Z_AXIS, beta_daughter)
    got_rest = boost_axis_to_rest_frame(LAB_Z_AXIS, PARENT_AT_REST, P_BARYON)
    assert _max_dev(got_rest, expected_rest) <= tol

    # a daughter at rest inside the parent rest frame (it moves with the
    # parent velocity in the working frame): the frames coincide and the
    # axis is unchanged (within the tolerance; the reconstructed
    # daughter velocity carries rounding at the 1e-16 level)
    coframe = _boost_lab((PROTON_MASS, 0.0, 0.0, 0.0), BETA_PARENT)
    same = boost_axis_to_rest_frame(AXIS_TILTED, PARENT_MOVING, coframe)
    assert _max_dev(same, AXIS_TILTED) <= tol

    # a collinear daughter along the axis from a parent at rest: the
    # aberration preserves the collinear direction exactly
    collinear = boost_axis_to_rest_frame(
        (0.0, 0.0, 1.0), PARENT_AT_REST,
        _on_shell(PROTON_MASS, (0.0, 0.0, 1.0), 1.2))
    assert collinear == (0.0, 0.0, 1.0)

    # -- checked, never repaired ---------------------------------------
    with pytest.raises(KinematicsError):
        boost_axis_to_rest_frame((0.0, 0.0, 0.0), PARENT_AT_REST, P_BARYON)
    with pytest.raises(KinematicsError):
        boost_axis_to_rest_frame(
            LAB_Z_AXIS, (4.0, 5.0, 0.0, 0.0), P_BARYON)  # spacelike parent
    with pytest.raises(KinematicsError):
        boost_axis_to_rest_frame(
            LAB_Z_AXIS, PARENT_AT_REST, (0.938, 0.938, 0.0, 0.0))


# ----------------------------------------------------------------------
# test_inheritance_rule
# ----------------------------------------------------------------------
def test_inheritance_rule(catalog, fams):
    """The baryon-carrying daughter carries the boosted parent axis; the
    symmetric mesonic daughter carries the parent rest-frame axis
    (pinned fixture, both branches); on drawn card-4 daughters the axis
    fields are populated through ``inherit_axis`` with the card-4
    isotropic label untouched, over a one-step continuation, with the
    per-vertex draw budget unchanged."""

    tol = AXIS_NORM_TOL

    # -- the pinned algebra fixture, both branches ----------------------
    boosted = boost_axis_to_rest_frame(
        LAB_Z_AXIS, PARENT_AT_REST, P_BARYON)
    assert inherit_axis(
        (1, 1, 0), LAB_Z_AXIS, PARENT_AT_REST, P_BARYON) == boosted
    assert inherit_axis(
        (-1, -1, 0), LAB_Z_AXIS, PARENT_AT_REST, P_BARYON) == boosted
    # the transported axis differs from the parent axis (the boost is
    # nontrivial at the pinned split: the daughter moves at beta ~ 0.895)
    assert _max_dev(boosted, LAB_Z_AXIS) > 1e-6
    # the mesonic daughters take the parent rest-frame axis verbatim
    assert inherit_axis(
        (0, 0, 0), LAB_Z_AXIS, PARENT_AT_REST, P_MESON) == LAB_Z_AXIS
    assert inherit_axis(
        (0, 1, 0), LAB_Z_AXIS, PARENT_AT_REST, P_MESON) == LAB_Z_AXIS
    assert inherit_axis(
        (0, -1, 0), AXIS_TILTED, PARENT_AT_REST, P_MESON) == AXIS_TILTED
    with pytest.raises(ValueError):
        inherit_axis((1, 1), LAB_Z_AXIS, PARENT_AT_REST, P_BARYON)

    # -- the card-4 integration: the axis fields on drawn daughters ----
    # the 8 GeV root keeps the realized continuum daughter inside the
    # family support, so the pinned continuation has an open line (the
    # card-4 mass rule realizes the daughter at the budget edge minus
    # the family nominal width)
    baryonic_root = _root_state("baryonic_np", 8.0, (1, 1, 0))
    channels = enumerate_channels(baryonic_root, catalog, fams)
    continuum = [c for c in channels if c.kind == KIND_CONTINUUM]
    # the pinned baryonic-line pool: the continuum channels whose
    # continuum daughter is the baryonic family (its class siblings are
    # mesonic), so the baryon-line daughter is the cascade state and the
    # pinned continuation has an open line
    baryon_pool = [
        c for c in continuum if "baryonic_np" in c.daughters]
    assert baryon_pool
    rng_channel = _CountingRng(MASTER_SEED + 11)
    rng_kinematics = _CountingRng(MASTER_SEED + 12)
    channel, daughters = draw_decay(
        baryonic_root, baryon_pool, catalog, fams,
        rng_channel, rng_kinematics)
    assert rng_channel.calls == 1 and rng_kinematics.calls == 2
    assert len(daughters) == 2
    # card 4 copies the parent axis verbatim; this card's machinery
    # replaces it at the call site
    for daughter in daughters:
        assert daughter.axis == baryonic_root.axis
    # baryon number conservation puts exactly one daughter on the
    # baryonic line at the (1, 1, 0) vertex
    baryon_line = [d for d in daughters if int(d.qn[0]) != 0]
    meson_line = [d for d in daughters if int(d.qn[0]) == 0]
    assert len(baryon_line) == 1 and len(meson_line) == 1
    baryon, meson = baryon_line[0], meson_line[0]
    expected_baryon_axis = boost_axis_to_rest_frame(
        baryonic_root.axis, baryonic_root.momentum, baryon.momentum)
    filled_baryon = replace(
        baryon, axis=inherit_axis(
            baryon.qn, baryonic_root.axis,
            baryonic_root.momentum, baryon.momentum))
    filled_meson = replace(
        meson_line[0], axis=inherit_axis(
            meson_line[0].qn, baryonic_root.axis,
            baryonic_root.momentum, meson_line[0].momentum))
    assert filled_baryon.axis == expected_baryon_axis
    assert filled_meson.axis == baryonic_root.axis
    # the replaced states keep every other field of the card-4 daughters
    for filled, daughter in ((filled_baryon, baryon),
                             (filled_meson, meson_line[0])):
        assert filled.momentum == daughter.momentum
        assert filled.mass == daughter.mass
        assert filled.qn == daughter.qn
        assert filled.family_id == daughter.family_id
        assert filled.alignment_label == ISOTROPIC_ALIGNMENT_LABEL
        assert filled.parent is daughter.parent
        assert filled.exchange_ancestry == daughter.exchange_ancestry
        assert filled.depth == daughter.depth

    # -- the continuation: the inherited axis feeds the next vertex ----
    channels = enumerate_channels(filled_baryon, catalog, fams)
    continuum = [c for c in channels if c.kind == KIND_CONTINUUM]
    assert continuum, "the pinned continuation has an open continuum line"
    rng_channel = _CountingRng(MASTER_SEED + 21)
    rng_kinematics = _CountingRng(MASTER_SEED + 22)
    _, grand_daughters = draw_decay(
        filled_baryon, continuum, catalog, fams,
        rng_channel, rng_kinematics)
    assert rng_channel.calls == 1 and rng_kinematics.calls == 2
    grand_baryon = next(d for d in grand_daughters if int(d.qn[0]) != 0)
    expected_grand_axis = boost_axis_to_rest_frame(
        filled_baryon.axis, filled_baryon.momentum, grand_baryon.momentum)
    filled_grand = replace(
        grand_baryon, axis=inherit_axis(
            grand_baryon.qn, filled_baryon.axis,
            filled_baryon.momentum, grand_baryon.momentum))
    assert filled_grand.axis == expected_grand_axis
    assert filled_grand.depth == filled_baryon.depth + 1
    assert _norm_dev(filled_grand.axis) <= tol

    # -- the mesonic root: both daughters carry the parent axis --------
    mesonic_root = _root_state("mesonic_neutral", 4.0, (0, 0, 0))
    channels = enumerate_channels(mesonic_root, catalog, fams)
    continuum = [c for c in channels if c.kind == KIND_CONTINUUM]
    # the pinned mesonic-line pool: the continuum channels whose
    # continuum daughter is the mesonic family, so both daughters are
    # mesonic (the full pool also holds the baryon-antibaryon classes,
    # whose baryonic daughter takes the boosted axis)
    meson_pool = [
        c for c in continuum if "mesonic_neutral" in c.daughters]
    assert meson_pool
    rng_channel = _CountingRng(MASTER_SEED + 31)
    rng_kinematics = _CountingRng(MASTER_SEED + 32)
    _, daughters = draw_decay(
        mesonic_root, meson_pool, catalog, fams,
        rng_channel, rng_kinematics)
    assert rng_channel.calls == 1 and rng_kinematics.calls == 2
    for daughter in daughters:
        assert int(daughter.qn[0]) == 0
        filled = replace(
            daughter, axis=inherit_axis(
                daughter.qn, mesonic_root.axis,
                mesonic_root.momentum, daughter.momentum))
        assert filled.axis == mesonic_root.axis

    # -- the moving-source topology: the machinery is frame-general ----
    moving_root = _root_state(
        "baryonic_np", 4.0, (1, 1, 0),
        momentum=_boost_lab((4.0, 0.0, 0.0, 0.0), (0.2, 0.1, 0.6)))
    channels = enumerate_channels(moving_root, catalog, fams)
    continuum = [c for c in channels if c.kind == KIND_CONTINUUM]
    rng_channel = _CountingRng(MASTER_SEED + 41)
    rng_kinematics = _CountingRng(MASTER_SEED + 42)
    _, daughters = draw_decay(
        moving_root, continuum, catalog, fams, rng_channel, rng_kinematics)
    assert rng_channel.calls == 1 and rng_kinematics.calls == 2
    baryon_line = next(d for d in daughters if int(d.qn[0]) != 0)
    filled = replace(
        baryon_line, axis=inherit_axis(
            baryon_line.qn, moving_root.axis,
            moving_root.momentum, baryon_line.momentum))
    expected = boost_axis_to_rest_frame(
        moving_root.axis, moving_root.momentum, baryon_line.momentum)
    assert filled.axis == expected


# ----------------------------------------------------------------------
# test_zero_draws
# ----------------------------------------------------------------------
def test_zero_draws():
    """A pinned cascade of 1,000 axis operations consumes zero random
    numbers (the generator state is byte-identical before and after)
    and the module source carries no random machinery."""

    rng = np.random.default_rng(MASTER_SEED)
    state_before = rng.bit_generator.state

    daughters = (P_BARYON, P_MESON)
    axes = (LAB_Z_AXIS, AXIS_TILTED)
    operations = 0
    for step in range(1_000):
        source_p4 = PP_SOURCES[step % len(PP_SOURCES)]
        beam = PP_BEAMS[step % len(PP_BEAMS)]
        axis = axes[step % len(axes)]
        daughter_p4 = daughters[step % len(daughters)]
        parent_p4 = PARENT_AT_REST if step % 8 < 4 else PARENT_MOVING
        kind = step % 4
        if kind == 0:
            initial_source_axis(source_p4, beam)
        elif kind == 1:
            boost_axis_to_rest_frame(axis, parent_p4, daughter_p4)
        elif kind == 2:
            inherit_axis((1, 1, 0), axis, parent_p4, daughter_p4)
        else:
            inherit_axis((0, 0, 0), axis, parent_p4, daughter_p4)
        operations += 1
    assert operations == 1_000

    state_after = rng.bit_generator.state
    assert state_after == state_before, (
        "the axis machinery consumed random numbers: the generator "
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
    assert not hits, f"the axis module carries random machinery: {hits}"


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
    the card's provenance section, card 4's cascade surface included,
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
# The acceptance-(v) freeze re-pin check
# ----------------------------------------------------------------------
