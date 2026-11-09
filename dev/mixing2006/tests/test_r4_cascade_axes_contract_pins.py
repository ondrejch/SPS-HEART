"""Round 4, card 5 test-author register: inherited production axes.

Independent contract pins for the inherited production axes module
(``na49_mixing/cascade_axes.py``), Round 4 card 5
(``TASK-20260920-r4-inherited-production-axes``; plan section 22 task
22, plan section 10.1, lines 664-674; the round 4 index item 4 axis
convention).  This register is the test-author half of the card: it
pins the module contract on self-authored momentum fixtures
(independent of the solver's pinned fixtures), self-authored
reference algebra (an independent light-aberration implementation),
self-authored zero-draw instrumentation, and it re-pins the freeze
manifest for the card.

Part 1: the module contract (pure, no sealed run dir)
-----------------------------------------------------

- ``test_initial_source_axis_unit_norm_and_convention``: unit norm
  within ``AXIS_NORM_TOL`` on self-authored momentum fixtures
  spanning the two-source topology (collinear projectile-like and
  target-like sources, a recoiling source) and the isolated source,
  including a highly boosted source (beta ~ 0.99); the collinear
  two-source convention (the beam direction in the source rest frame
  for a source moving along the beam, exact ``(0, 0, 1)``); the
  isolated-source lab z-axis (``LAB_Z_AXIS`` exactly, independent of
  the source motion); and the beam-direction normalization contract
  (any nonzero scale of the beam direction gives the same axis
  within the tolerance).
- ``test_initial_source_axis_input_validation``: the checked-never-
  repaired contract on self-authored invalid inputs (a zero beam
  direction, a non-finite beam direction, a non-3-tuple beam
  direction, a spacelike source, a negative-energy source, a
  non-finite source component, and a non-4-vector source: the
  kernel's ``KinematicsError`` in every case, nothing repaired).
- ``test_boost_round_trip_and_boundary``: unit-norm preservation on
  self-authored fixtures; the round-trip self-inverse property
  (boost into the daughter rest frame and back within
  ``AXIS_NORM_TOL``) on a resting parent, a high-velocity parent
  (beta ~ 0.95), a tilted axis, and a fast daughter; the exact
  boundary (parent and daughter four-momenta equal: the axis returns
  unchanged, bit-exact for a resting parent, within the tolerance
  for a moving one); the collinear exact preservation.
- ``test_wigner_free_single_boost_agreement``: the Wigner-rotation-
  free contract on its degenerate case: for a parent at rest in the
  working frame the transported axis equals the axis transformed by
  the single daughter boost (the self-authored light-aberration
  reference, within ``AXIS_NORM_TOL``); the two-step through-the-
  working-frame reference on a moving parent within the same bound.
- ``test_inherit_axis_branches_and_validation``: the baryon branch
  (``B != 0``: the transported parent axis, verified against the
  self-authored hand computation through
  ``boost_axis_to_rest_frame`` and against the independent
  aberration reference; positive and negative baryon number); the
  mesonic branch (``B == 0``: the parent rest-frame axis verbatim,
  exact equality); the input validation on self-authored fixtures
  (a non-3-tuple ``daughter_qn`` raises ``ValueError``, the
  on-disk integer cast of a numeric-string tuple, a zero axis
  raising ``KinematicsError`` on both branches, a spacelike
  daughter momentum raising it on the baryon branch, and the
  on-disk mesonic branch's four-momentum non-validation, since it
  returns the parent axis verbatim without touching the momenta).
- ``test_zero_draw_contract``: the public functions consume no
  randomness on self-authored instrumentation (sentinel monkey-
  patches of the standard-library and numpy random machinery that
  raise on any call, a fixed-seed bit-generator state identity
  across a pinned operation cascade, and a self-authored source
  scan for random machinery).
- ``test_module_constants``: ``AXIS_NORM_TOL == 1e-12`` and
  ``LAB_Z_AXIS == (0.0, 0.0, 1.0)`` (value, type, and tuple arity;
  the package-level constants are the module's objects).

Part 2: the substrate and freeze pins
-------------------------------------

- ``test_module_byte_and_export_pins``: the module byte stability
  (sha256 ``a7251a6b...``) and the additive ``na49_mixing`` package
  exports (``initial_source_axis``, ``boost_axis_to_rest_frame``,
  ``inherit_axis``, ``LAB_Z_AXIS``, ``AXIS_NORM_TOL`` in the
  package ``__all__`` and as runtime attributes).
- ``test_frozen_substrate_digests``: the byte-unchanged surfaces the
  card consumes (card 4's cascade surface, the builder, the card-6
  surface, the card-2 engine, the Round 2 kernel, the committed
  catalog, the continuum families configuration, and the R32 driver
  in both trees), verified on disk.
- ``test_freeze_repin_register``: the freeze-manifest re-pin for
  this register (the strict path-sorted inventory at 461 entries
  carries this file at its on-disk digest; the card-5 freeze
  exception names this card exactly once with this register's file;
  the two prior registers' inventory-count pins are re-pinned
  457 -> 458 naming this register, as is the solver test's count
  constant; card 6 re-pins them again 458 -> 460 naming the
  baryonic-alignment-kernel test, and card 6's independent
  test-author register re-pins them once more 460 -> 461 naming
  itself).

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
from na49_mixing.cascade_axes import (
    AXIS_NORM_TOL,
    LAB_Z_AXIS,
    boost_axis_to_rest_frame,
    inherit_axis,
    initial_source_axis,
)
from na49_mixing.resonance_decay import KinematicsError, _boost_lab
from na49_mixing.resonance_widths import decay_momentum
from tests._freeze_inventory import inventory_count as _inventory_count

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent          # dev/mixing2006
MONO_ROOT = NESTED_ROOT.parent.parent          # the monorepo root
MODULE_PATH = NESTED_ROOT / "na49_mixing" / "cascade_axes.py"
INIT_PATH = NESTED_ROOT / "na49_mixing" / "__init__.py"
MANIFEST_PATH = (NESTED_ROOT / "config" / "v2_singlet"
                 / "legacy_v1_manifest.json")

MODULE_REL = "dev/mixing2006/na49_mixing/cascade_axes.py"
INIT_REL = "dev/mixing2006/na49_mixing/__init__.py"
SOLVER_TEST_REL = "dev/mixing2006/tests/test_cascade_axes.py"
MY_TEST_REL = ("dev/mixing2006/tests/"
               "test_r4_cascade_axes_contract_pins.py")
CARD3_REGISTER_REL = ("dev/mixing2006/tests/"
                      "test_r4_audit_blend_path_contract_pins.py")
CARD4_REGISTER_REL = ("dev/mixing2006/tests/"
                      "test_r4_cascade_decay_contract_pins.py")
MANIFEST_REL = ("dev/mixing2006/config/v2_singlet/"
                "legacy_v1_manifest.json")

FREEZE_CARD = "TASK-20260920-r4-inherited-production-axes"
# 458 entries at this register's landing; card 6 of the round
# (TASK-20260920-r4-baryonic-alignment-kernel) re-pins the inventory
# additively for its new alignment-kernel module and test file
# (test_cascade_alignment_baryonic.py; the card-mandated re-pin;
# 458 -> 460, +2 entries), so this count pin is re-pinned again;
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
COUNT_PIN = "assert len(keys) == INVENTORY_COUNT"

MODULE_SHA256 = (
    "a7251a6b6c0ed4b9cfc56c0182c28a70efccb7a92a33b756090ff716a3ec3c61"
)

#: The byte-unchanged frozen surfaces the card consumes (the card's
#: provenance section, verified on disk at this register's landing):
#: card 4's cascade surface, the builder, the card-6 surface, the
#: card-2 engine, the Round 2 kernel, the committed catalog, the
#: continuum families configuration, and the dual-tree R32 driver.
FROZEN_SUBSTRATE = {
    "dev/mixing2006/na49_mixing/cascade_decay.py":
        "d2a48049af3d6df968993c040033a74422ccd709f53a42bad0538ae5111bcc0c",
    "dev/mixing2006/na49_mixing/resonance_pp_builder.py":
        "4ab612609e1ee0bc1e037dcbb2815b6484306ad32402b7289e6636f8e829cdb1",
    "dev/mixing2006/na49_mixing/spectral_density.py":
        "382e048b9f66a35a6446e39d97e95a9fed885e0ac5ec297bfa19823e5acf20b7",
    "dev/mixing2006/na49_mixing/threshold_metric.py":
        "ee240f44645d2e592acfae462f7d1e9c3b3dd8cd78c96bd40b7b73509f7fcc63",
    "dev/mixing2006/na49_mixing/resonance_decay.py":
        "6a1e8f0ad3c9e6e56645f919ba14060836ab27dd8e53235eca4434f9b1aa8210",
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


# ----------------------------------------------------------------------
# Self-authored fixtures and helpers
# ----------------------------------------------------------------------
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
    """The self-authored light-aberration reference: the direction *n*
    as seen from the rest frame of an object moving with velocity
    *beta* (``cos(theta') = (cos(theta) - beta) / (1 - beta
    cos(theta))``, the azimuth preserved).  Analytic; independent of
    the kernel's ``_boost_lab``."""

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


def _beta(p4: tuple[float, float, float, float]) -> tuple[float, float, float]:
    return (p4[1] / p4[0], p4[2] / p4[0], p4[3] / p4[0])


def _max_dev(a, b) -> float:
    return max(abs(a[i] - b[i]) for i in range(3))


def _norm_dev(v) -> float:
    return abs(math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) - 1.0)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


#: The self-authored masses and momenta (independent of the solver's
#: pinned fixtures; the split momentum is the kernel's exact two-body
#: decay momentum).
PROTON_MASS = 0.93827
PION_MASS = 0.13957
PARENT_MASS = 5.0
PARENT_AT_REST = (PARENT_MASS, 0.0, 0.0, 0.0)

TILTED_AXIS = _unit((0.55, 0.25, -0.75))
_FAST_BETA_UNIT = _unit((0.5, -0.1, 0.75))
FAST_BETA: tuple[float, float, float] = (
    0.95 * _FAST_BETA_UNIT[0],
    0.95 * _FAST_BETA_UNIT[1],
    0.95 * _FAST_BETA_UNIT[2],
)
PARENT_FAST = _boost_lab(PARENT_AT_REST, FAST_BETA)

#: A highly boosted source on the beam axis (beta ~ 0.99).
SOURCE_MASS = 1.0
SOURCE_ENERGY = 7.0
SOURCE_Q = math.sqrt(SOURCE_ENERGY ** 2 - SOURCE_MASS ** 2)
SOURCE_BOOSTED = (SOURCE_ENERGY, 0.0, 0.0, SOURCE_Q)

#: The self-authored two-source p+p fixtures: the projectile-like and
#: target-like sources moving along the beam (z) axis, and a recoiling
#: source with a transverse component.
PROJECTILE_SOURCE = (8.0, 0.0, 0.0, math.sqrt(8.0 ** 2 - 2.5 ** 2))
TARGET_SOURCE = _on_shell(2.5, (0.0, 0.0, 1.0), 3.0)
RECOIL_DIRECTION = _unit((-0.2, 0.6, 0.8))
RECOIL_SOURCE = _on_shell(2.5, RECOIL_DIRECTION, 2.5)

#: The self-authored two-body split of the resting parent: the baryon
#: and the pion back-to-back along the tilted axis at the exact decay
#: momentum.
SPLIT_Q = decay_momentum(PARENT_MASS, PROTON_MASS, PION_MASS)
P_BARYON = _on_shell(PROTON_MASS, TILTED_AXIS, SPLIT_Q)
P_MESON = _on_shell(PION_MASS, _neg3(TILTED_AXIS), SPLIT_Q)


# ----------------------------------------------------------------------
# test_initial_source_axis_unit_norm_and_convention
# ----------------------------------------------------------------------
INITIAL_AXIS_FIXTURES = (
    # (label, source_momentum, beam_direction)
    ("isolated_at_rest", (3.5, 0.0, 0.0, 0.0), None),
    ("isolated_moving",
     _on_shell(3.5, _unit((0.3, -0.4, 0.5)), 2.0), None),
    ("projectile_collinear", PROJECTILE_SOURCE, (0.0, 0.0, 1.0)),
    ("target_collinear", TARGET_SOURCE, (0.0, 0.0, 1.0)),
    ("recoiling", RECOIL_SOURCE, (0.0, 0.0, 1.0)),
    ("highly_boosted", SOURCE_BOOSTED, (0.0, 0.0, 1.0)),
)


def test_initial_source_axis_unit_norm_and_convention():
    """Unit norm within the tolerance on self-authored momentum
    fixtures; the collinear two-source convention exact; the
    isolated-source lab z-axis exact; the beam-direction
    normalization contract."""

    tol = AXIS_NORM_TOL
    for label, source_momentum, beam_direction in INITIAL_AXIS_FIXTURES:
        axis = initial_source_axis(source_momentum, beam_direction)
        assert len(axis) == 3
        assert all(math.isfinite(component) for component in axis)
        assert _norm_dev(axis) <= tol, (
            f"{label}: the initial axis norm deviates by "
            f"{_norm_dev(axis)!r}"
        )

    # -- the collinear two-source convention: the beam axis exactly --
    for label, source in (("projectile", PROJECTILE_SOURCE),
                          ("target", TARGET_SOURCE)):
        axis = initial_source_axis(source, (0.0, 0.0, 1.0))
        assert axis == (0.0, 0.0, 1.0), (
            f"{label}: the collinear source must carry the beam axis "
            "exactly (the source moves along the beam)"
        )

    # -- the isolated-source convention: the lab z-axis exactly ------
    assert initial_source_axis((3.5, 0.0, 0.0, 0.0), None) == LAB_Z_AXIS
    moving_isolated = initial_source_axis(
        _on_shell(3.5, _unit((0.3, -0.4, 0.5)), 2.0), None)
    assert moving_isolated == LAB_Z_AXIS, (
        "the isolated source carries the lab z-axis by convention "
        "(beam_direction is None), independent of the source motion"
    )

    # -- the off-axis transports against the self-authored reference --
    expected_recoil = _aberration_reference(
        (0.0, 0.0, 1.0), _beta(RECOIL_SOURCE))
    got_recoil = initial_source_axis(RECOIL_SOURCE, (0.0, 0.0, 1.0))
    assert _max_dev(got_recoil, expected_recoil) <= tol, (
        "the recoiling-source axis misses the boosted beam direction "
        f"by {_max_dev(got_recoil, expected_recoil)!r}"
    )
    expected_boosted = _aberration_reference(
        (0.0, 0.0, 1.0), _beta(SOURCE_BOOSTED))
    got_boosted = initial_source_axis(SOURCE_BOOSTED, (0.0, 0.0, 1.0))
    assert _max_dev(got_boosted, expected_boosted) <= tol

    # the beam direction is a direction: any nonzero scale normalizes
    # to the same axis (within the floating-point tolerance)
    scaled = initial_source_axis(RECOIL_SOURCE, (0.0, 0.0, 3.5))
    assert _max_dev(scaled, got_recoil) <= tol, (
        "a rescaled beam direction must give the same axis (the "
        "direction is normalized)"
    )


# ----------------------------------------------------------------------
# test_initial_source_axis_input_validation
# ----------------------------------------------------------------------
def test_initial_source_axis_input_validation():
    """Checked, never repaired: every invalid self-authored input
    raises the kernel's ``KinematicsError``."""

    valid_p4 = PARENT_AT_REST
    cases = (
        ("zero beam direction",
         lambda: initial_source_axis(valid_p4, (0.0, 0.0, 0.0))),
        ("nan beam component",
         lambda: initial_source_axis(valid_p4, (float("nan"), 0.0, 1.0))),
        ("inf beam component",
         lambda: initial_source_axis(valid_p4, (0.0, 1.0, float("inf")))),
        ("two-component beam",
         lambda: initial_source_axis(valid_p4, (0.0, 1.0))),
        ("spacelike source",
         lambda: initial_source_axis((4.0, 5.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
        ("negative-energy source",
         lambda: initial_source_axis((-4.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
        ("nan source component",
         lambda: initial_source_axis(
             (4.0, float("nan"), 0.0, 0.0), None)),
        ("three-component source",
         lambda: initial_source_axis((4.0, 0.0, 0.0), None)),
    )
    for label, call in cases:
        with pytest.raises(KinematicsError):
            call()


# ----------------------------------------------------------------------
# test_boost_round_trip_and_boundary
# ----------------------------------------------------------------------
ROUND_TRIP_FIXTURES = (
    # (label, axis, parent_p4, daughter_p4)
    ("rest_parent_z_axis", LAB_Z_AXIS, PARENT_AT_REST, P_BARYON),
    ("rest_parent_tilted", TILTED_AXIS, PARENT_AT_REST, P_BARYON),
    ("fast_parent_z_axis", LAB_Z_AXIS, PARENT_FAST, P_BARYON),
    ("fast_parent_tilted", TILTED_AXIS, PARENT_FAST, P_BARYON),
    ("rest_parent_pion", LAB_Z_AXIS, PARENT_AT_REST, P_MESON),
)


def test_boost_round_trip_and_boundary():
    """Unit-norm preservation; the round-trip self-inverse property
    within the tolerance on self-authored momentum pairs (a resting
    parent, a high-velocity parent, a tilted axis, a fast daughter);
    the exact boundary (parent == daughter momentum)."""

    tol = AXIS_NORM_TOL
    for label, axis, parent_p4, daughter_p4 in ROUND_TRIP_FIXTURES:
        transported = boost_axis_to_rest_frame(axis, parent_p4, daughter_p4)
        assert _norm_dev(transported) <= tol, (
            f"{label}: the transported axis norm deviates by "
            f"{_norm_dev(transported)!r}"
        )
        recovered = boost_axis_to_rest_frame(
            transported, daughter_p4, parent_p4)
        assert _norm_dev(recovered) <= tol
        assert _max_dev(recovered, axis) <= tol, (
            f"{label}: the boost round trip misses the axis by "
            f"{_max_dev(recovered, axis)!r}"
        )

    # -- the exact boundary: parent and daughter coincide -----------
    same = boost_axis_to_rest_frame(TILTED_AXIS, PARENT_FAST, PARENT_FAST)
    assert _max_dev(same, TILTED_AXIS) <= tol, (
        "a parent and a daughter with equal four-momenta share a rest "
        "frame; the axis must return unchanged"
    )
    same_rest = boost_axis_to_rest_frame(
        LAB_Z_AXIS, PARENT_AT_REST, PARENT_AT_REST)
    assert same_rest == LAB_Z_AXIS, (
        "the coincident resting parent/daughter boundary is bit-exact"
    )

    # a collinear daughter along the beam from a resting parent: the
    # aberration preserves the collinear direction exactly
    collinear = boost_axis_to_rest_frame(
        (0.0, 0.0, 1.0), PARENT_AT_REST,
        _on_shell(PROTON_MASS, (0.0, 0.0, 1.0), 1.5))
    assert collinear == (0.0, 0.0, 1.0)

    # -- checked, never repaired -------------------------------------
    with pytest.raises(KinematicsError):
        boost_axis_to_rest_frame((0.0, 0.0, 0.0), PARENT_AT_REST, P_BARYON)
    with pytest.raises(KinematicsError):
        boost_axis_to_rest_frame(
            TILTED_AXIS, (4.0, 5.0, 0.0, 0.0), P_BARYON)  # spacelike parent
    with pytest.raises(KinematicsError):
        boost_axis_to_rest_frame(
            LAB_Z_AXIS, PARENT_AT_REST, (PION_MASS, 1.0, 0.0, 0.0))
    with pytest.raises(KinematicsError):
        boost_axis_to_rest_frame(
            (float("nan"), 0.0, 0.0), PARENT_AT_REST, P_BARYON)


# ----------------------------------------------------------------------
# test_wigner_free_single_boost_agreement
# ----------------------------------------------------------------------
def test_wigner_free_single_boost_agreement():
    """The Wigner-rotation-free contract on its degenerate case: for a
    parent at rest in the working frame the transported axis equals
    the axis transformed by the single daughter boost (the
    self-authored light-aberration reference); the two-step through-
    the-working-frame reference on a moving parent."""

    tol = AXIS_NORM_TOL
    daughter_beta = _beta(P_BARYON)
    for axis in (LAB_Z_AXIS, TILTED_AXIS):
        got = boost_axis_to_rest_frame(axis, PARENT_AT_REST, P_BARYON)
        expected = _aberration_reference(axis, daughter_beta)
        assert _max_dev(got, expected) <= tol, (
            "the resting-parent transport misses the single daughter "
            f"boost by {_max_dev(got, expected)!r} (both conventions "
            "must agree in the degenerate case)"
        )
        assert _norm_dev(got) <= tol

    # the moving parent: the through-the-working-frame composition of
    # the two aberration steps
    expected_moving = _aberration_reference(
        _aberration_reference(TILTED_AXIS, _neg3(FAST_BETA)), daughter_beta)
    got_moving = boost_axis_to_rest_frame(TILTED_AXIS, PARENT_FAST, P_BARYON)
    assert _max_dev(got_moving, expected_moving) <= tol, (
        "the moving-parent transport misses the two-step aberration "
        f"reference by {_max_dev(got_moving, expected_moving)!r}"
    )


# ----------------------------------------------------------------------
# test_inherit_axis_branches_and_validation
# ----------------------------------------------------------------------
def test_inherit_axis_branches_and_validation():
    """The baryon branch carries the transported parent axis (against
    the self-authored hand computation); the mesonic branch takes
    the parent rest-frame axis verbatim; the input validation on
    self-authored fixtures."""

    tol = AXIS_NORM_TOL

    # -- the baryon branch (B != 0): the transported parent axis -----
    boosted = boost_axis_to_rest_frame(
        LAB_Z_AXIS, PARENT_AT_REST, P_BARYON)
    got = inherit_axis((1, 1, 0), LAB_Z_AXIS, PARENT_AT_REST, P_BARYON)
    assert got == boosted, (
        "the baryon-carrying daughter must inherit the parent axis "
        "transported to its rest frame (the hand computation through "
        "boost_axis_to_rest_frame)"
    )
    # the same hand computation through the independent aberration
    # reference (resting parent: the single daughter boost)
    expected = _aberration_reference(LAB_Z_AXIS, _beta(P_BARYON))
    assert _max_dev(got, expected) <= tol

    # the transported axis differs from the parent axis (the boost is
    # nontrivial at the pinned split: the baryon moves at beta ~ 0.93)
    assert _max_dev(boosted, LAB_Z_AXIS) > 1e-6

    # negative baryon number on a moving parent, tilted axis
    boosted_fast = boost_axis_to_rest_frame(
        TILTED_AXIS, PARENT_FAST, P_MESON)
    got_fast = inherit_axis(
        (-1, -1, 0), TILTED_AXIS, PARENT_FAST, P_MESON)
    assert got_fast == boosted_fast
    expected_fast = _aberration_reference(
        _aberration_reference(TILTED_AXIS, _neg3(FAST_BETA)), _beta(P_MESON))
    assert _max_dev(got_fast, expected_fast) <= tol
    assert _norm_dev(got_fast) <= tol

    # -- the mesonic branch (B == 0): the parent axis verbatim -------
    assert inherit_axis(
        (0, 0, 0), TILTED_AXIS, PARENT_FAST, P_MESON) == TILTED_AXIS
    assert inherit_axis(
        (0, 1, 0), LAB_Z_AXIS, PARENT_AT_REST, P_MESON) == LAB_Z_AXIS
    assert inherit_axis(
        (0, -1, 0), TILTED_AXIS, PARENT_AT_REST, P_BARYON) == TILTED_AXIS

    # -- checked, never repaired --------------------------------------
    with pytest.raises(ValueError):
        inherit_axis((1, 1), LAB_Z_AXIS, PARENT_AT_REST, P_BARYON)
    with pytest.raises(ValueError):
        inherit_axis((1, 1, 0, 0), LAB_Z_AXIS, PARENT_AT_REST, P_BARYON)
    # the on-disk integer cast: a numeric-string qn tuple is accepted
    got_str = inherit_axis(("1", "1", "0"), LAB_Z_AXIS, PARENT_AT_REST, P_BARYON)
    assert got_str == boosted
    # zero axis on both branches
    with pytest.raises(KinematicsError):
        inherit_axis((0, 0, 0), (0.0, 0.0, 0.0), PARENT_AT_REST, P_MESON)
    with pytest.raises(KinematicsError):
        inherit_axis((1, 1, 0), (0.0, 0.0, 0.0), PARENT_AT_REST, P_BARYON)
        # a spacelike daughter momentum on the baryon branch
    with pytest.raises(KinematicsError):
        inherit_axis(
            (1, 1, 0), LAB_Z_AXIS, PARENT_AT_REST, (1.0, 2.0, 0.0, 0.0))
    # the on-disk mesonic branch consumes no four-momentum at all (it
    # returns the parent axis verbatim): a malformed momentum is not
    # validated there, it is simply unused
    assert inherit_axis(
        (0, 0, 0), LAB_Z_AXIS, (4.0, 5.0, 0.0, 0.0), P_MESON) == LAB_Z_AXIS


# ----------------------------------------------------------------------
# test_zero_draw_contract
# ----------------------------------------------------------------------
def _sentinel(name: str):
    def _raise(*args, **kwargs):
        raise AssertionError(f"random machinery {name} was invoked")

    return _raise


@pytest.fixture
def no_random(monkeypatch):
    """Sentinel monkey-patches of the standard-library and numpy random
    machinery: any call raises (the self-authored zero-draw guard).
    Returns the unpatched ``np.random.default_rng`` so the test can
    still seed its own state-identity generator."""

    import random as _random

    # The returned factory is built from classes captured before the patches: NumPy 2's default_rng checks its seed
    # against np.random.RandomState, which the sentinels below replace. Generator(PCG64(SeedSequence(seed))) has the
    # same state as default_rng(seed).
    _generator, _pcg64, _seed_sequence = np.random.Generator, np.random.PCG64, np.random.SeedSequence

    def original_default_rng(seed):
        return _generator(_pcg64(_seed_sequence(seed)))
    for name in ("random", "randint", "uniform", "choice", "choices",
                 "getrandbits", "seed", "SystemRandom", "Random",
                 "shuffle", "sample", "triangular", "gauss", "beta",
                 "expovariate", "normalvariate", "vonmises",
                 "weibullvariate", "lognormvariate", "gammavariate"):
        monkeypatch.setattr(
            _random, name, _sentinel(f"random.{name}"), raising=False)
    for name in ("random", "rand", "randint", "uniform", "normal",
                 "choice", "permutation", "shuffle", "sample", "seed",
                 "default_rng", "RandomState", "SeedSequence"):
        monkeypatch.setattr(
            np.random, name, _sentinel(f"numpy.random.{name}"), raising=False)
    return original_default_rng


def test_zero_draw_contract(no_random):
    """The public functions consume no randomness (the sentinel random
    machinery stays uncalled, the fixed-seed bit-generator state is
    byte-identical across a pinned operation cascade, and the module
    source carries no random machinery)."""

    rng = no_random(20260921)
    state_before = rng.bit_generator.state

    sources = (PARENT_AT_REST, RECOIL_SOURCE, SOURCE_BOOSTED,
               PROJECTILE_SOURCE, TARGET_SOURCE)
    beams = ((0.0, 0.0, 1.0), None, (0.0, 0.0, 1.0))
    axes = (LAB_Z_AXIS, TILTED_AXIS)
    momenta = (PARENT_AT_REST, PARENT_FAST, P_BARYON, P_MESON)
    qns = ((1, 1, 0), (0, 0, 0), (-1, -1, 0), (0, 1, 0))
    operations = 0
    for step in range(400):
        source = sources[step % len(sources)]
        beam = beams[(step * 3) % len(beams)]
        axis = axes[(step * 7) % len(axes)]
        parent = momenta[(step * 5) % len(momenta)]
        daughter = momenta[(step * 5 + 1) % len(momenta)]
        qn = qns[(step * 11) % len(qns)]
        if step % 4 == 0:
            initial_source_axis(source, beam)
        elif step % 4 == 1:
            boost_axis_to_rest_frame(axis, parent, daughter)
        else:
            inherit_axis(qn, axis, parent, daughter)
        operations += 1
    assert operations == 400

    state_after = rng.bit_generator.state
    assert state_after == state_before, (
        "the axis machinery consumed random numbers: the fixed-seed "
        "bit-generator state changed over the pinned 400-operation "
        "cascade"
    )

    # the module source carries no random machinery (self-authored
    # scan: tokens that would appear in any random-number consumer)
    source = MODULE_PATH.read_text(encoding="utf-8")
    forbidden = (
        "import random", "from random", "np.random", "numpy.random",
        "default_rng", "RandomState", "SeedSequence", "getrandbits",
        "randint", "random(", "rand(", "shuffle(", "gauss(", "uniform(",
        "tri(", "exponential(", "entropy(", "bit_generator",
    )
    hits = [token for token in forbidden if token in source]
    assert not hits, f"the axis module carries random machinery: {hits}"


# ----------------------------------------------------------------------
# test_module_constants
# ----------------------------------------------------------------------
def test_module_constants():
    """The module constants: ``AXIS_NORM_TOL == 1e-12`` and
    ``LAB_Z_AXIS == (0.0, 0.0, 1.0)``."""

    assert AXIS_NORM_TOL == 1e-12
    assert isinstance(AXIS_NORM_TOL, float)
    assert LAB_Z_AXIS == (0.0, 0.0, 1.0)
    assert isinstance(LAB_Z_AXIS, tuple)
    assert len(LAB_Z_AXIS) == 3
    assert all(isinstance(component, float) for component in LAB_Z_AXIS)
    # the package-level constants are the module's objects (the
    # additive exports, not copies)
    assert na49_mixing.AXIS_NORM_TOL is AXIS_NORM_TOL
    assert na49_mixing.LAB_Z_AXIS is LAB_Z_AXIS


# ----------------------------------------------------------------------
# test_module_byte_and_export_pins
# ----------------------------------------------------------------------
def test_module_byte_and_export_pins():
    """The module byte stability (sha256) and the additive
    ``na49_mixing`` package exports."""

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    init_src = INIT_PATH.read_text(encoding="utf-8")
    for name in ("initial_source_axis", "boost_axis_to_rest_frame",
                 "inherit_axis", "LAB_Z_AXIS", "AXIS_NORM_TOL"):
        assert f'"{name}"' in init_src, (
            f"the package __init__ no longer exports {name}"
        )
    for name in ("initial_source_axis", "boost_axis_to_rest_frame",
                 "inherit_axis", "LAB_Z_AXIS", "AXIS_NORM_TOL"):
        assert hasattr(na49_mixing, name), (
            f"the package no longer resolves {name}"
        )
    from na49_mixing.cascade_axes import (
        boost_axis_to_rest_frame as module_boost,
        inherit_axis as module_inherit,
        initial_source_axis as module_initial,
    )
    assert na49_mixing.boost_axis_to_rest_frame is module_boost
    assert na49_mixing.inherit_axis is module_inherit
    assert na49_mixing.initial_source_axis is module_initial


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
