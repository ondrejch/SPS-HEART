"""Round 4, card 13 test-author register: the R42 recoil/decay
separation contract pins.

Card ``TASK-20260920-r4-run-r42-recoil-decay-separation`` (plan section
10 run R42, line 786; the Round 4 card set, card 13), re-pointed by
the card 17 relitigation
(``TASK-20260920-r4-run-r42-relitigation``) to the corrected driver
and the sealed run
``dev/00runs/r42_recoil_decay_separation_2026-09-23/`` (the rework of
the recording to the R01 accumulate-dense-then-pick convention).
This register is the test-author half of the card: independent pins
of the dual-tree run driver
(``helpers/benchmarking/staged_r42_recoil_decay_separation.py``,
byte-identical in the nested tree and the monorepo mirror), of the
hermetic solver test
(``tests/test_staged_r42_recoil_decay_separation.py``), and of the
sealed run ``dev/00runs/r42_recoil_decay_separation_2026-09-23/``.
The two superseded sealed archives
(``dev/00runs/r42_recoil_decay_separation_2026-09-22/`` and
``dev/00runs/r42_recoil_decay_separation_2026-09-22r2/``) stay intact
and are pinned as the supersession reference via the provenance's
superseded_r42_archives block.
The driver is loaded once via importlib (its top level is the
pre-registration identity; no events are generated), and the sealed
JSON is read from disk.  The solver's pinned seed-map and pinned
algebra traces are not re-run here; the contract is pinned on the
committed bytes and the sealed on-disk values.

- ``test_dual_tree_driver_byte_identity``: both driver copies exist,
  are byte-identical, and sit at the committed digest; the driver's
  own self-computed identity digest agrees; the freeze manifest
  inventory pins both at that digest.
- ``test_driver_identity_constants``: the driver's run identity, the
  file-name constants, the substrate and sealed-dir rel paths, the card
  chain, the R32 p+p collision configuration, the two-axis profile grid
  (the 9 x 9 pre-registered choice), the nominal point held fixed, the
  pinned reweighting-validity grid points, the nuisance structure, the
  obligation 4 carry, the distinguishability bound, and the stability
  statistics.
- ``test_seed_map_partition``: the master seed, the raw block
  20301020-20301027, the event-index partition 500_000_000 + i, the
  kernel seed rule, the hermetic smoke block (+9000), and the seed
  functions' refusal discipline.
- ``test_cli_surface``: the exact argparse command surface on disk
  (the program name, exactly the seven mode flags, the three auxiliary
  flags and their defaults) plus the on-disk exit-code behavior (help
  exits 0, no mode refuses with 2, two modes refuse with 2).
- ``test_pre_registration_ordering_and_drift``: the
  pre-registration-before-generation ordering invariant (a non-empty
  run dir is refused before the substrate is touched) and the
  pre-registration-vs-card drift probes (grid, statistics, seed,
  carry, and written_before_generation drift are refused; the smoke
  mode is exempt; the sealed pre-registration passes).
- ``test_pre_registration_on_disk``: the pre-registration's schema,
  the T2 statistics, the seed map, the 9 x 9 profile grid, the
  collision configuration, the conservation and observables blocks,
  the likelihood form, the acceptance check set with its five
  register triggers, the blend-band-sensitive bins, the obligation 4
  carry with the audit outcomes, the substrate verification, and the
  sealed run dirs.
- ``test_profile_report_on_disk``: the report's schema, the internal
  digest, the full 9 x 9 surface at its pinned values, the grid argmin
   (the boundary minimum), the pooled Hessian block (not positive
   definite), the undefined correlation, the per-block curvature
   stability (seven of eight blocks positive definite; block 4
   alone failing), the five-entry named-residual register, the
   per-block ESS, and the pooled objective at the minimum.
- ``test_gate4_evidence``: the gate record's identity, its clause
  (the string-like rank law), its conservation summary (zero
  violations over 800,000 events), the profile-report digest
  cross-reference, and the equality of its minimum, Hessian,
  curvature-stability, carry, and named-residual blocks with the
  report's.
- ``test_seal_manifest_and_checksums``: the seal manifest's 13 sealed
  files (8 replica records plus the five top-level files) path-sorted,
  the checksum file's 13 entries, and the full re-hash of all 13
  sealed files against the checksum digests.
- ``test_clean_environment_record``: the clean-environment
  validation's command list, the byte-identical report re-hash
  (digest equality), the probe replica record, and the pending seal
  state.
- ``test_provenance_on_disk``: the provenance's schema, the substrate
   digests, the M8 frozen set hash, the recorded driver digest (the
   committed dual-tree digest: the driver was not amended after the
   pre-registration), the nominal point, the superseded-archives
   supersession block, and the sealed run dir verification.
- ``test_frozen_substrate_stability``: the driver's fifteen frozen
  substrate files byte-stable on disk at their pinned digests.
- ``test_solver_test_file_digest``: the hermetic solver test file at
  its committed digest, pinned in the freeze manifest inventory, with
  the card-13 solver freeze exception intact (the four path-sorted
  files).
- ``test_freeze_repin_register``: the freeze-manifest re-pin for this
  register (the strict path-sorted inventory at 488 entries carries
  this file at its on-disk digest; the card-13 solver freeze exception
  stays at the solver's 4 path-sorted files; this register's own
  freeze exception names the test-author card exactly once with the 19
  path-sorted files; the seventeen consumed count pins -- the sixteen
  chain files plus the card-13 solver test's own count constant -- are
  re-pinned 483 -> 484 with the naming comment attributing the bump to
  this register, the card-13 solver's 480 -> 483 comment, the card-12
  solver's 476 -> 479 comment, the card-12 test-author's 479 -> 480
  comment, and the card-10 solver's 471 literal retained for their
  re-pin checks).

Hermetic: standard library and numpy, the committed config and
catalog, and the sealed run dir (read-only) only; no SPS tables, no
event generation, no Monte Carlo run.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
from tests._freeze_inventory import inventory_count as _inventory_count
from tests._sealed_archive import sealed_mismatches, skip_unless_sealed

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent          # dev/mixing2006
MONO_ROOT = NESTED_ROOT.parent.parent          # the monorepo root
NESTED_DRIVER = (
    NESTED_ROOT / "helpers" / "benchmarking"
    / "staged_r42_recoil_decay_separation.py"
)
OUTER_DRIVER = (
    MONO_ROOT / "helpers" / "benchmarking"
    / "staged_r42_recoil_decay_separation.py"
)
MANIFEST_PATH = (NESTED_ROOT / "config" / "v2_singlet"
                 / "legacy_v1_manifest.json")
RUN_DIR = (MONO_ROOT / "dev" / "00runs"
            / "r42_recoil_decay_separation_2026-09-23")

#: The superseded sealed run dirs (the card-13 first seal and its
#: axis-swap re-seal; both stay intact and are pinned as the
#: supersession reference via the provenance's
#: superseded_r42_archives block, not read directly by this register).
SUPERSEDED_RUN_DIRS = (
    "dev/00runs/r42_recoil_decay_separation_2026-09-22",
    "dev/00runs/r42_recoil_decay_separation_2026-09-22r2",
)

DRIVER_REL_NESTED = (
    "dev/mixing2006/helpers/benchmarking/"
    "staged_r42_recoil_decay_separation.py"
)
DRIVER_REL_OUTER = (
    "helpers/benchmarking/staged_r42_recoil_decay_separation.py"
)
SOLVER_TEST_REL = (
    "dev/mixing2006/tests/test_staged_r42_recoil_decay_separation.py"
)

#: The committed corrected dual-tree driver digest (byte-identical in
#: both trees; the freeze manifest inventory pins both copies here;
#: the card-17 relitigation rework of the recording to the R01
#: accumulate-dense-then-pick convention, superseding the card-13
#: driver).
DRIVER_SHA256 = (
    "a1917e578dfcc56b8f8c73652bc893d3aee21b870cdafeca4b46dfd7ad4961fd"
)

#: The superseded card-13 driver digest (the published-row flat index
#: recording; the driver bytes were superseded by the relitigation).
#: Pinned as the supersession reference in the pre-registration's
#: recording_convention block and the provenance's
#: superseded_r42_archives block.
SUPERSEDED_DRIVER_SHA256 = (
    "30e97f18653129ba3e7f1e5d8af782c819a1a7b189413032925ec35b26b0afa0"
)

#: The hermetic solver test file's committed digest (22 hermetic
#: tests, including the grid-hessian boundary-corner regression test
#: and the four mirrored hole-geometry tests; the card-17
#: relitigation landing; the card-13 test-author re-pin of its own
#: count constant 483 -> 484 and the card-14 test-author re-pin of
#: the same count constant 487 -> 488; the card-20 count re-pin
#: 491 -> 495 touched the same count constant, so the digest moved
#: from the card-17 landing digest 4217c029f3107c4a...; the Round 5
#: card 1 count re-pin 499 -> 501 touched the same constant, so the
#: digest moved from the card-21 landing digest e3f112a3ec0dc5c4...;
#: the Round 5 card 1 test-author count re-pin 501 -> 502 touched the
#: same constant, so the digest moved from the Round 5 card 1 landing
#: digest 92e0a39a07635250...).
#: the Round 5 card 2 count re-pin 502 -> 504 touched the same constant (TASK-20260925-r5-exchange-quantum-number-ledger).
#: the Round 5 card 2 test-author count re-pin 504 -> 505 touched the same constant (TASK-20260925-r5-exchange-quantum-number-ledger-test-author).
#: the Round 5 card 3 count re-pin 508 -> 510 touched the same constant (TASK-20260925-r5-intact-low-mass-remnants).
#: the Round 5 card 5 test-author count re-pin 516 -> 517 touched the same constant (TASK-20260925-r5-single-diffraction-test-author).
#: re-pinned by the Round 5 card 7 test-author 522 -> 523 count re-pin, which
#: re-touched this solver test file (TASK-20260925-r5-non-diffrative-central-excitation-test-author).
#: re-pinned by the Round 5 card 8 test-author 525 -> 526 count re-pin, which
#: re-touched this solver test file (TASK-20260925-r5-multiple-neutral-excitations-test-author).
#: re-pinned by the Round 5 card 9 526 -> 528 count re-pin, which
#: re-touched this solver test file (TASK-20260925-r5-topology-template-runner).
#: re-pinned by the Round 5 card 9 test-author 528 -> 529 count re-pin, which
#: re-touched this solver test file (TASK-20260925-r5-topology-template-runner-test-author).
#: re-pinned 2026-11-08: the solver test uses np.trapezoid where NumPy 2 removed np.trapz
#: (TASK-20261108-sps-heart-numpy2-compat).
SOLVER_TEST_SHA256 = (
    "db7f9814b226ee04ec505235b14c9d28f9466ea1d53312498a02222596b56a0e"
)

# ----------------------------------------------------------------------
# Freeze re-pin constants (the register chain's inventory count pins)
# ----------------------------------------------------------------------
MY_TEST_REL = (
    "dev/mixing2006/tests/"
    "test_staged_r42_recoil_decay_separation_contract_pins.py"
)
MANIFEST_REL = (
    "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json"
)
SOLVER_CARD = "TASK-20260920-r4-run-r42-recoil-decay-separation"
# The test-author re-pin's own freeze-exception card name (distinct
# from the solver's, so the solver's "exactly one entry naming this
# card" check is untouched).
TEST_AUTHOR_CARD = (
    "TASK-20260920-r4-run-r42-recoil-decay-separation-test-author"
)

#: The sixteen register-chain files whose inventory-count pins this
#: register re-pins (483 -> 484, +1 entry): the twelve register-chain
#: files (cards 1, 4, 5, 6, 7, 8, 9, 10) plus the card-10 solver
#: test's own count constant (test_cascade_ablation.py), the card-11
#: test-author register's own count constant
#: (test_staged_r40_alignment_scan_contract_pins.py), and the card-12
#: test-author register's own count constant
#: (test_staged_r41_nonstrange_pilot_fit_contract_pins.py).
CHAIN_COUNT_PIN_FILES = (
    "dev/mixing2006/tests/test_r4_audit_blend_path_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_decay_contract_pins.py",
    "dev/mixing2006/tests/test_cascade_axes.py",
    "dev/mixing2006/tests/test_r4_cascade_axes_contract_pins.py",
    "dev/mixing2006/tests/test_cascade_alignment_baryonic.py",
    "dev/mixing2006/tests/"
    "test_r4_cascade_alignment_baryonic_contract_pins.py",
    "dev/mixing2006/tests/test_cascade_alignment_mesonic.py",
    "dev/mixing2006/tests/"
    "test_r4_cascade_alignment_mesonic_contract_pins.py",
    "dev/mixing2006/tests/test_cascade_form_factor.py",
    "dev/mixing2006/tests/test_r4_cascade_form_factor_contract_pins.py",
    "dev/mixing2006/tests/test_cascade_termination.py",
    "dev/mixing2006/tests/test_r4_cascade_termination_contract_pins.py",
    "dev/mixing2006/tests/test_cascade_ablation.py",
    "dev/mixing2006/tests/test_r4_cascade_ablation_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_staged_r40_alignment_scan_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_staged_r41_nonstrange_pilot_fit_contract_pins.py",
)

#: The seventeen count pins this register re-pins: the sixteen
#: register-chain files plus the card-13 solver test's own count
#: constant (test_staged_r42_recoil_decay_separation.py).
REPIN_FILES = CHAIN_COUNT_PIN_FILES + (SOLVER_TEST_REL,)

#: The card-13 solver freeze-exception file list (4 path-sorted files;
#: this register leaves it untouched).
SOLVER_EXCEPTION_FILES = (
    MANIFEST_REL,
    DRIVER_REL_NESTED,
    SOLVER_TEST_REL,
    DRIVER_REL_OUTER,
)

#: This register's own freeze-exception file list (path-sorted): the
#: manifest, this register, the card-13 solver test, and the sixteen
#: re-pinned count pins.
MY_EXCEPTION_FILES = tuple(
    sorted([MANIFEST_REL, MY_TEST_REL, SOLVER_TEST_REL]
           + list(CHAIN_COUNT_PIN_FILES))
)

# 483 entries at the card-13 solver landing (the dual-tree R42 driver
# in both helper trees and the hermetic R42 test file); this register
# adds its file (+1 entry) for the test-author re-pin (483 -> 484,
# +1 entry) and re-pins the seventeen consumed count pins (the sixteen
# register-chain files plus the card-13 solver test's own count
# constant) with the bump attributed to this register's file.
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
# Pinned frozen substrate (rel, digest) -- the driver's own table
# ----------------------------------------------------------------------
SUBSTRATE: dict[str, tuple[str, str]] = {
    "cascade_decay": (
        "dev/mixing2006/na49_mixing/cascade_decay.py",
        "f63b3649039f12c1b65c87b82b64433e8c746679b90050d9e72eb0110fd923e2",
    ),
    "cascade_axes": (
        "dev/mixing2006/na49_mixing/cascade_axes.py",
        "a7251a6b6c0ed4b9cfc56c0182c28a70efccb7a92a33b756090ff716a3ec3c61",
    ),
    "cascade_alignment": (
        "dev/mixing2006/na49_mixing/cascade_alignment.py",
        "7aad9434e70efaf2424d6b66fbeaede9419ea869b4968b11d1466961de5912b2",
    ),
    "cascade_form_factor": (
        "dev/mixing2006/na49_mixing/cascade_form_factor.py",
        "b16077c9e3418b3f19b117aadf6fc8a8e838af5e9d37b1ab0bae89ccfb88f74c",
    ),
    "cascade_termination": (
        "dev/mixing2006/na49_mixing/cascade_termination.py",
        "65eadd60c3e06007218d83f6baa5f12f9088caacd24512b3de48c594d4a3063e",
    ),
    "cascade_ablation": (
        "dev/mixing2006/na49_mixing/cascade_ablation.py",
        "c0206cf6e75b883de345a788c0246ea7bd7f48bf03102ff7e811e2cc5af9d1c0",
    ),
    "round2_kernel": (
        "dev/mixing2006/na49_mixing/resonance_decay.py",
        "0b675ffef0b17ec0a6d8e93f593a15adda03dc8dca40e5a789f5f3285d384bc6",
    ),
    "builder_blend_path": (
        "dev/mixing2006/na49_mixing/resonance_pp_builder.py",
        "4ab612609e1ee0bc1e037dcbb2815b6484306ad32402b7289e6636f8e829cdb1",
    ),
    "card6_routing": (
        "dev/mixing2006/na49_mixing/spectral_density.py",
        "561c65ee9a77a1b7f7eb07a7619dafb584779a51ccb23d7fe5651c0605e12563",
    ),
    "card2_engine": (
        "dev/mixing2006/na49_mixing/threshold_metric.py",
        "ee240f44645d2e592acfae462f7d1e9c3b3dd8cd78c96bd40b7b73509f7fcc63",
    ),
    "catalog": (
        "dev/mixing2006/config/v2_singlet/resonance_catalog.json",
        "6e4de7372a5bbf99affd6fa3ccc295c4cef09404dab3b17acd8816cda762bdcd",
    ),
    "continuum_config": (
        "dev/mixing2006/config/v2_singlet/continuum_resonances.json",
        "242f906cf040eadbe3796dfcfb61d0d51535277399858c901a162e6653d635e1",
    ),
    "r32_driver": (
        "helpers/benchmarking/staged_r32_spectral_systematic.py",
        "99454facce856c93505e6c4b0d4e47b85eeb0ef54f72b1631e3d29fcc511cb90",
    ),
    "r40_driver": (
        "helpers/benchmarking/staged_r40_alignment_scan.py",
        "7366a55e0e15032f6d4c9194adce7537ea5239075e34f921900b75be64787354",
    ),
    # The committed R41 driver (the card-16 relitigation rework; the
    # at-run-time value is pinned by R41_DRIVER_SHA256_AT_RUN_TIME)
    "r41_driver": (
        "helpers/benchmarking/staged_r41_nonstrange_pilot_fit.py",
        "abf2a8c30344ebeecab4413f5d1c4045b091ac3467226c6257e09d501697cdfc",
    ),
}

# ----------------------------------------------------------------------
# Pinned sealed-run values (read from disk, verified on disk)
# ----------------------------------------------------------------------

#: The seal manifest's file list (8 replica records plus the five
#: top-level files), path-sorted.
SEALED_FILES = tuple(
    f"replica_{r}.json" for r in range(8)
) + (
    "clean_environment_commands.json",
    "gate4_r42.json",
    "pre_registration.json",
    "profile_report.json",
    "provenance.json",
)

#: The on-disk digest of every sealed file (the checksum file's
#: entries plus the seal manifest's own digest), the sealed
#: 2026-09-23 archive of the card-17 relitigation.
SEALED_FILE_DIGESTS = {
    "clean_environment_commands.json": (
        "09e024a81af0d50215ff9265492a536e15cad780d3f9318170fe4c6d258a8fed"
    ),
    "gate4_r42.json": (
        "09f1db626813fac8fcc86429e6df82c089bec4717f506ea82e33a580f7a3c13e"
    ),
    "pre_registration.json": (
        "8e00ce7aa05694f7e4ee0dc261ed93fc3bceba3aecf8c675d740fd7eb624705c"
    ),
    "profile_report.json": (
        "430be97de52ba84b1519559b35c5758b2c4d0b08e8a890056ab5c05b78037d62"
    ),
    "provenance.json": (
        "1fb6c2a17dfba4d020863dfb5c9c8eb2d4a43055fb72a0b21f2d1178cd62f4eb"
    ),
    "replica_0.json": (
        "7b50ddf4ecec877a11b813932922354dfa94f9080b87c07248f4b7a78ad36a04"
    ),
    "replica_1.json": (
        "8fc58cf739464aa89e740f59e6d6984d1ed59364c1b0ebb0008e049cc6f90438"
    ),
    "replica_2.json": (
        "82e11dcc74a3ea788574ea9fa2627839031cdef7d3ab83ee8bcfdb67fcf9e3ee"
    ),
    "replica_3.json": (
        "c224e8515e734825a82f1a40e8ec4ca02303476491f3061ecb4cabe7661c2720"
    ),
    "replica_4.json": (
        "8cfc4a3e828e56d830c13ecaf47001daca0af8c9f186848dab7a6168161dd805"
    ),
    "replica_5.json": (
        "6f79e455dfcb7401116bb6320a0f02f3d10341f63152e0c376c253f601749068"
    ),
    "replica_6.json": (
        "f85b2b405c27fb8ea4a299c7f69f09693182c725c662315d40b93b22b6d47420"
    ),
    "replica_7.json": (
        "038857da10856c22fd62999b57b0071540962f05f8e4accdcc46e753e85f7f69"
    ),
    "seal_manifest.json": (
        "cc297241131f59bf1778cd1b89c95df2dfac364cf495fd0d1e0f0dfa79308a5b"
    ),
}

#: The checksum file's own digest (excluded from its own entries).
CHECKSUMS_SHA256 = (
    "be111df3ff5e38226f50de10b8837c9bd4db612e35be15eadd5766531b2594f0"
)

#: The profile report's internal digest (the report bytes carry no
#: wall-clock fields).
PROFILE_REPORT_DIGEST = (
    "e18a1e275dda9b72c86ce4f63ea216f250b3a378626f63497a8d252d52755af5"
)

#: The frozen R32 p+p collision configuration.
M8_PARAMETER_HASH = (
    "358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1"
)
BEAM_MOMENTUM_GEV_C = 158.0
SQRT_S_GEV = 17.2701885890706
BLEND_RAMP_START_GEV = 2.680194110727528
BLEND_RAMP_WIDTH_GEV = 0.5

#: The pre-registered seed map.
MASTER_SEED = 20260920
RAW_REPLICA_SEEDS = tuple(20301020 + r for r in range(8))
EVENT_INDEX_BASE = 500_000_000
EVENT_INDEX_RULE = "500_000_000 + i, i < 100,000"
SMOKE_RAW_SEED = 20309920
SMOKE_KERNEL_REPLICA_SLOT = 9000

#: The pre-registered 9 x 9 profile grid (index item 10).
B_EFF_MULTIPLIERS = (
    0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5,
)
B_DEC_GRID = (0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6)
B_EFF_FROZEN_GEV2 = 8.504140762083612

#: The pinned reweighting-validity grid points.
VALIDITY_GRID_POINTS = ((1.5, 1.7), (2.0, 0.5))

#: The pre-registered obligation 4 carry (the sealed R32 spread,
#: verbatim from the card).
OBLIGATION4_R32_SPREAD = {
    "bin_spread_p95": 4.028426,
    "feeddown_spread_max_sigma": 31.1773,
    "persisting_residuals": 32,
}

#: The sealed profile minimum (the grid argmin, on the boundary).
MINIMUM = {
    "b_dec": 2.6,
    "b_eff_gev2": 4.252070381041806,
    "b_eff_multiplier": 0.5,
    "i": 0,
    "j": 8,
    "minus2lnl": 16078.399851141561,
}

#: The pooled 9 x 9 profiled -2lnL surface (81 pinned values, row
#: i = multiplier index, column j = b_dec index), evaluated on the
#: corrected dense-lattice-aligned spectra.
SURFACE = [
    [20993.03255098508, 20957.32577294224, 20765.81646475778,
     20458.270932873613, 19969.788439678494, 19231.76387226819,
     18292.523372488, 17227.56824950981, 16078.399851141561],
    [35417.50210739121, 37045.44877735863, 38120.61529227261,
     38654.55049839032, 38620.87125609978, 38033.904472092676,
     36929.15355686841, 35318.62435760557, 33187.74389198897],
    [39986.78162717613, 41805.776427858276, 42903.635450535156,
     43476.056287616004, 43594.09294666548, 43270.0258486501,
     42474.499228538385, 41106.6749716961, 38946.69351432155],
    [38536.94980877143, 40188.86968534118, 41158.30416946519,
     41646.51719340787, 41743.09116647974, 41470.447887470924,
     40792.35714943298, 39583.83738643558, 37568.86113199638],
    [36066.543581970356, 37508.548773410104, 38306.00452830238,
     38660.443391872854, 38675.55626566977, 38386.89041123561,
     37764.05132232052, 36684.58377539103, 34888.53621827156],
    [33554.0018980913, 34760.02632254493, 35373.36842181287,
     35593.91770098798, 35530.514607699195, 35220.86605300642,
     34631.80381013653, 33639.2301270411, 32005.029184978455],
    [31242.277354776757, 32240.536134560636, 32700.647306407725,
     32815.576514643835, 32691.665026557654, 32362.704092999462,
     31790.19948068096, 30848.773022008907, 29321.552872457025],
    [29200.080282891326, 30033.220771653716, 30376.83838661936,
     30415.13483610898, 30248.80201008028, 29906.181555522082,
     29344.282046924538, 28438.24887440284, 26990.072014492544],
    [27417.961095103165, 28125.796817593087, 28384.49837636101,
     28370.05493341824, 28176.817608299887, 27827.820160832485,
     27276.886967692484, 26401.091306824634, 25018.76356121667],
]

#: The pooled Hessian at the minimum (grid second differences; not
#: positive definite; the one-sided boundary triplets at the corner
#: on the corrected aligned spectra).
HESSIAN_EIGENVALUES = [-182549.71181584388, 7.698059717158571]
HESSIAN = [
    [-181606.31069623717, -13089.494229978034],
    [-13089.494229978034, -935.7030598895411],
]

#: The per-block curvature eigenvalues at the pooled minimum (seven
#: of eight replica blocks positive definite; block 4 alone failing
#: the positive-definite clause).
BLOCK_CURVATURE_EIGENVALUES = {
    "0": [5343.577894705166, 220029.11325967722],
    "1": [6582.411465697041, 358667.2353051157],
    "2": [1544.2858550274427, 25567.053910870403],
    "3": [3530.247065655907, 286894.46652847773],
    "4": [-1999.037806083929, 15699.862514911674],
    "5": [1641.0754892203815, 54135.52186734246],
    "6": [711.7443042717218, 236492.78426237762],
    "7": [2378.35088676504, 241611.33905700978],
}

#: The per-replica-block effective sample size fractions at the
#: minimum.
PER_BLOCK_ESS = [
    0.29830065087237756,
    0.09004369554519313,
    0.15922756436907845,
    0.31843127822833167,
    0.22541444462898516,
    0.2779550302332313,
    0.294159631612186,
    0.24078749362562524,
]

#: The pooled objective at the minimum.
OBJECTIVE = {
    "ess_min": 0.09004369554519313,
    "minus2lnl_profiled": 16078.399851141561,
    "n_eff_total": 803388.0819337294,
    "nuisance_columns": ["norm_b_transport_spectrum", "carry"],
    "nuisances": [-0.005252426383737569, -0.010484078705604201],
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "staged_r42_recoil_decay_separation_under_test", NESTED_DRIVER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before exec_module: the fork worker pool resolves the
    # worker functions by module name (the sys.modules inheritance of
    # forked children makes the pickle reference resolvable).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    return _load_driver()


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pre_registration():
    skip_unless_sealed(RUN_DIR)
    return json.loads(
        (RUN_DIR / "pre_registration.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def profile_report():
    skip_unless_sealed(RUN_DIR)
    return json.loads(
        (RUN_DIR / "profile_report.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def gate_record():
    skip_unless_sealed(RUN_DIR)
    return json.loads(
        (RUN_DIR / "gate4_r42.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def seal_manifest():
    skip_unless_sealed(RUN_DIR)
    return json.loads(
        (RUN_DIR / "seal_manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def clean_environment():
    skip_unless_sealed(RUN_DIR)
    return json.loads(
        (RUN_DIR / "clean_environment_commands.json")
        .read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def provenance():
    skip_unless_sealed(RUN_DIR)
    return json.loads(
        (RUN_DIR / "provenance.json").read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# Driver identity, byte identity, and the CLI surface
# ----------------------------------------------------------------------
def test_dual_tree_driver_byte_identity(driver, manifest):
    """The dual-tree driver is byte-identical in both trees at the
    committed digest, the driver's own self-computed identity digest
    agrees, and the freeze manifest pins both copies."""
    assert NESTED_DRIVER.is_file(), f"missing nested driver: {NESTED_DRIVER}"
    assert OUTER_DRIVER.is_file(), f"missing outer driver: {OUTER_DRIVER}"
    nested_digest = _sha256_file(NESTED_DRIVER)
    outer_digest = _sha256_file(OUTER_DRIVER)
    assert nested_digest == outer_digest, (
        "the dual-tree driver is not byte-identical")
    assert nested_digest == DRIVER_SHA256, (
        "the committed dual-tree driver digest drifted")
    assert driver.DRIVER_SHA256 == DRIVER_SHA256, (
        "the driver's self-computed identity digest does not match "
        "the committed digest")
    inventory = manifest["source_tree_sha256"]
    assert inventory[DRIVER_REL_NESTED] == DRIVER_SHA256, (
        "the freeze manifest does not pin the nested driver at the "
        "committed digest")
    assert inventory[DRIVER_REL_OUTER] == DRIVER_SHA256, (
        "the freeze manifest does not pin the outer driver at the "
        "committed digest")


def test_driver_identity_constants(driver):
    """The driver's run identity, the file-name constants, the
    substrate and sealed-dir rel paths, the card chain, the R32 p+p
    collision configuration, the two-axis profile grid, the nominal
    point, the pinned reweighting-validity points, the nuisance
    structure, the obligation 4 carry, the distinguishability bound,
    and the stability statistics."""
    assert driver.RUN_ID == "R42"
    assert driver.CARD == SOLVER_CARD
    assert driver.RUN_NAME == "r42_recoil_decay_separation_2026-09-23"
    assert driver.DEFAULT_RUN_DIR == (
        "dev/00runs/r42_recoil_decay_separation_2026-09-23")
    # The superseded-archive provenance constants (the card-13 seals)
    assert driver.SUPERSEDED_DRIVER_SHA256 == SUPERSEDED_DRIVER_SHA256
    assert driver.SUPERSEDED_RUN_DIRS == SUPERSEDED_RUN_DIRS
    assert driver.GATE_RECORD_NAME == "gate4_r42.json"
    assert driver.PROFILE_REPORT_NAME == "profile_report.json"
    assert driver.PREREGISTRATION_NAME == "pre_registration.json"
    assert driver.PROVENANCE_NAME == "provenance.json"
    assert driver.CLEAN_ENV_NAME == "clean_environment_commands.json"
    assert driver.SHA256SUMS_NAME == "SHA256SUMS.txt"
    assert driver.SEAL_NAME == "seal_manifest.json"
    assert driver.RECORD_FMT == "replica_{}.json"
    assert driver.SMOKE_PREFIX == "smoke_"
    assert driver.CATALOG_REL == (
        "config/v2_singlet/resonance_catalog.json")
    assert driver.FAMILIES_REL == (
        "config/v2_singlet/continuum_resonances.json")
    # The driver's self-referential rel path (nested-tree relative, the
    # _locate_roots convention)
    assert driver.DRIVER_REL == (
        "helpers/benchmarking/staged_r42_recoil_decay_separation.py")
    assert driver.ENGINE_REL == (
        "dev/mixing2006/na49_mixing/threshold_metric.py")
    assert driver.BUILDER_REL == (
        "dev/mixing2006/na49_mixing/resonance_pp_builder.py")
    assert driver.ROUTING_REL == (
        "dev/mixing2006/na49_mixing/spectral_density.py")
    assert driver.KERNEL_REL == (
        "dev/mixing2006/na49_mixing/resonance_decay.py")
    assert driver.R32_DRIVER_REL == (
        "helpers/benchmarking/staged_r32_spectral_systematic.py")
    assert driver.R40_DRIVER_REL == (
        "helpers/benchmarking/staged_r40_alignment_scan.py")
    assert driver.R41_DRIVER_REL == (
        "helpers/benchmarking/staged_r41_nonstrange_pilot_fit.py")
    assert driver.CARD_CHAIN == (
        "TASK-20260920-r4-audit-transition-routing-formulation",
        "TASK-20260920-r4-audit-strange-sector-continuum-families",
        "TASK-20260920-r4-audit-builder-blend-path-routing-weight",
        "TASK-20260920-r4-rework-transition-routing-formulation",
        "TASK-20260920-r4-sequential-color-singlet-resonance-decay",
        "TASK-20260920-r4-inherited-production-axes",
        "TASK-20260920-r4-baryonic-alignment-kernel",
        "TASK-20260920-r4-mesonic-alignment-kernel",
        "TASK-20260920-r4-hadronic-decay-form-factor",
        "TASK-20260920-r4-termination-depth-tests",
        "TASK-20260920-r4-isotropic-aligned-ablation-runner",
        "TASK-20260920-r4-run-r40-fixed-source-alignment-scan",
        "TASK-20260920-r4-run-r41-nonstrange-pilot-fit",
        "TASK-20260920-r4-run-r42-recoil-decay-separation",
    )
    # The R32 p+p collision configuration (index item 7)
    assert driver.BEAM_MOMENTUM_GEV_C == BEAM_MOMENTUM_GEV_C
    assert driver.SQRT_S_GEV == SQRT_S_GEV
    assert driver.BLEND_RAMP_START_GEV == BLEND_RAMP_START_GEV
    assert driver.BLEND_RAMP_WIDTH_GEV == BLEND_RAMP_WIDTH_GEV
    assert driver.M8_PARAMETER_HASH == M8_PARAMETER_HASH
    # The frozen exchange-layer slope (the M8-frozen value)
    assert driver.B_EFF_FROZEN_GEV2 == B_EFF_FROZEN_GEV2
    # The nominal point held fixed (not a profile axis of this card)
    assert driver.NOMINAL_KAPPA_B == 1.0
    assert driver.NOMINAL_KAPPA_M == 1.0
    assert driver.NOMINAL_WIDTH_GEV == 0.5
    assert driver.NOMINAL_POINT_NOTE.startswith(
        "the alignment kernels stay at the nominal point")
    # The 9 x 9 profile grid (index item 10, the pre-registered
    # minimal choice)
    assert driver.B_EFF_MULTIPLIERS == B_EFF_MULTIPLIERS
    assert driver.B_DEC_GRID == B_DEC_GRID
    assert driver.B_DEC_NOMINAL_GEV2 == 1.0
    assert driver.B_DEC_NOMINAL_GEV2 not in driver.B_DEC_GRID
    assert driver.B_DEC_MULT_STEP == 0.25
    assert driver.B_DEC_STEP_GEV2 == 0.3
    assert driver.PROFILE_GRID_NOTE.startswith(
        "the 9 x 9 profile grid")
    # The pinned reweighting-validity grid points and events
    assert driver.VALIDITY_GRID_POINTS == VALIDITY_GRID_POINTS
    assert driver.VALIDITY_EVENTS == 2_000
    assert driver.VALIDITY_BOUND == 1e-6
    # The likelihood's nuisance structure (plan section 5.4)
    assert driver.NUISSANCE_COLUMNS == (
        "norm_b_transport_spectrum", "carry")
    assert driver.NORM_NUISANCE_PRIOR == 0.35
    assert driver.CARRY_NUISANCE_PRIOR == 0.25
    assert driver.CARRY_SPECIES == ("proton", "antiproton")
    # The obligation 4 model-uncertainty carry
    assert driver.OBLIGATION4_R32_SPREAD == OBLIGATION4_R32_SPREAD
    # The robust-statistics cross-reference
    assert driver.SIGMA_BAND == 3.0
    assert driver.SE_FLOOR == 1e-12
    assert driver.ESS_FLOOR_FRACTION == 0.01
    # The pre-registered distinguishability bound (acceptance (ii))
    assert driver.CORRELATION_BOUND == 0.9
    assert driver.CORRELATION_NOTE.startswith(
        "the parameter correlation rho")
    assert driver.EVENT_CLOSURE_TOL == 1e-9
    assert driver.DEGENERATE_WINDOW_TOL == 1e-12
    # Generation chunking and the stage label
    assert driver.CHUNK_SIZE == 2500
    assert driver.DEFAULT_WORKERS == 8
    assert driver.STAGE_LABEL == "r42_recoil_decay_separation"
    # The clean-environment validation's pre-registered probe replica
    assert driver.PROBE_REPLICA == 0
    # The species conventions (the R01/R32 driver conventions)
    assert driver.SPECIES == (
        ("pi_plus", 211), ("pi_minus", -211), ("k_plus", 321),
        ("k_minus", -321), ("proton", 2212), ("antiproton", -2212))
    assert driver.SPECIES_KEYS == (
        "pi_plus", "pi_minus", "k_plus", "k_minus", "proton",
        "antiproton")
    # The sealed archives consumed read-only
    assert driver.SEALED_RUN_DIRS == (
        "dev/00runs/r40_fixed_source_alignment_scan_2026-09-21",
        "dev/00runs/r41_nonstrange_pilot_fit_2026-09-23",
        "dev/00runs/r30_spectral_normalization_2026-09-17",
        "dev/00runs/r31_catalog_continuum_matching_scan_2026-09-17",
        "dev/00runs/r32_spectral_systematic_pp_2026-09-18",
        "dev/00runs/r32_spectral_systematic_pp_confirmation_2026-09-19",
    )
    assert driver.R01_RUN_DIR == (
        "dev/00runs/r01_legacy_pp_reference_2026-09-11")
    assert driver.R40_RUN_DIR == (
        "dev/00runs/r40_fixed_source_alignment_scan_2026-09-21")
    assert driver.R41_RUN_DIR == (
        "dev/00runs/r41_nonstrange_pilot_fit_2026-09-23")
    assert driver.R4_AUDIT_CARD1_DIR == (
        "dev/00runs/r4_audit_transition_routing_formulation_2026-09-20")
    # The frozen substrate table (the r41_driver entry pins the
    # corrected card-16 R41 driver, which was already committed at
    # the time this run executed)
    assert driver.FROZEN_SUBSTRATE == SUBSTRATE


def test_seed_map_partition(driver):
    """The master seed, the raw block 20301020-20301027, the
    event-index partition 500_000_000 + i, the kernel seed rule, the
    hermetic smoke block (+9000), and the seed functions' refusal
    discipline."""
    assert driver.MASTER_SEED == MASTER_SEED
    assert driver.SEED_BASE == MASTER_SEED
    assert driver.SEED_OFFSET == 40_000
    assert driver.SEED_STRIDE == 100
    assert driver.T2_REPLICAS == 8
    assert driver.T2_EVENTS_PER_REPLICA == 100_000
    assert driver.RAW_REPLICA_SEEDS == RAW_REPLICA_SEEDS
    assert driver.RAW_SEED_RULE.startswith(
        "raw replica seed = 20260920 + 40000 + 100 + r = 20301020 + r")
    assert "20301020-20301027" in driver.RAW_SEED_RULE
    assert driver.KERNEL_SEED_RULE.startswith(
        "derive_substreams(20260920, r, 500_000_000 + i)")
    assert driver.EVENT_INDEX_RULE == EVENT_INDEX_RULE
    assert driver.EVENT_INDEX_BASE == EVENT_INDEX_BASE
    for r in range(8):
        assert driver.raw_replica_seed(r) == 20301020 + r
        assert driver.kernel_replica_slot(r) == r
    for r in (-1, 8):
        with pytest.raises(SystemExit) as exc:
            driver.raw_replica_seed(r)
        assert exc.value.code == 2
    assert driver.event_index(0) == EVENT_INDEX_BASE
    assert driver.event_index(99_999) == EVENT_INDEX_BASE + 99_999
    # The hermetic smoke block (+9000; never sealed)
    assert driver.SMOKE_REPLICAS == 1
    assert driver.SMOKE_EVENTS == 2_000
    assert driver.SMOKE_SEED_STRIDE == SMOKE_KERNEL_REPLICA_SLOT
    assert driver.SMOKE_RAW_SEED == SMOKE_RAW_SEED
    assert driver.smoke_raw_seed() == SMOKE_RAW_SEED
    assert driver.SMOKE_VALIDITY_EVENTS == 200
    assert driver.SMOKE_SYNTHETIC_TRUTH == {
        "b_eff_multiplier": 1.3,
        "b_dec": 1.2,
    }


def test_cli_surface(driver):
    """The exact argparse command surface on disk: the program name,
    exactly the seven mode flags, the three auxiliary flags and their
    defaults, and the on-disk exit-code behavior."""
    parser = driver._build_parser()
    # The driver does not pin an explicit program name on its parser
    # (the R40 driver does): argparse falls back to the invoked
    # script's basename.
    assert parser.prog == os.path.basename(sys.argv[0])
    modes = {
        "--pre-registration", "--smoke", "--generate", "--profile",
        "--gate", "--seal", "--clean-validate",
    }
    auxiliary = {"--replica", "--run-dir", "--workers"}
    seen = set()
    for action in parser._actions:
        seen.update(action.option_strings)
    # Exactly the mode flags, the auxiliary flags, and argparse's own
    # help flag (the surface on disk, no inferred extras).
    assert seen == modes | auxiliary | {"-h", "--help"}, (
        f"the CLI surface drifted: extra or missing flags "
        f"{sorted(seen - (modes | auxiliary | {'-h', '--help'}))}")
    mode_actions = [
        a for a in parser._actions if a.option_strings
        and set(a.option_strings) & modes
    ]
    assert len(mode_actions) == 7
    for action in mode_actions:
        assert action.const is True
        assert action.default is False
    defaults = {}
    for action in parser._actions:
        if action.option_strings:
            defaults[action.option_strings[0]] = action.default
    assert defaults["--workers"] == 8
    assert defaults["--run-dir"] is None
    assert defaults["--replica"] is None
    # The on-disk exit-code behavior
    with pytest.raises(SystemExit) as exc:
        driver.main(["--help"])
    assert exc.value.code == 0
    with pytest.raises(SystemExit) as exc:
        driver.main([])
    assert exc.value.code == 2, (
        "no mode must refuse with exit code 2")
    with pytest.raises(SystemExit) as exc:
        driver.main(["--pre-registration", "--smoke"])
    assert exc.value.code == 2, (
        "two modes must refuse with exit code 2")


# ----------------------------------------------------------------------
# The pre-registration ordering invariant and the drift probes
# ----------------------------------------------------------------------
def test_pre_registration_ordering_and_drift(driver, pre_registration,
                                             capsys):
    """The pre-registration-before-generation ordering invariant (a
    non-empty run dir is refused before the substrate is touched) and
    the pre-registration-vs-card drift probes (grid, statistics,
    seed, carry, and written_before_generation drift are refused; the
    smoke mode is exempt; the sealed pre-registration passes)."""
    # The ordering invariant: pre-registration refuses a non-empty run
    # dir before verifying anything (the run dir must be fresh).
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        probe_dir = Path(tmp) / "r42_ordering_probe"
        probe_dir.mkdir()
        (probe_dir / "sentinel.txt").write_text("pre-existing content")
        with pytest.raises(SystemExit) as exc:
            driver.run_pre_registration(probe_dir)
        assert exc.value.code == 2
        assert "not empty" in capsys.readouterr().err, (
            "the non-empty run dir refusal message is missing")
    # The sealed pre-registration passes the card's assertion
    driver._assert_pre_registration(pre_registration, RUN_DIR)
    # In-memory drift probes: every mutation below must be refused
    # (the refusal message lands on stderr; SystemExit carries the
    # code only)
    def _refused(mutate, message_part):
        mutated = copy.deepcopy(pre_registration)
        mutate(mutated)
        with pytest.raises(SystemExit) as exc:
            driver._assert_pre_registration(mutated, RUN_DIR)
        assert exc.value.code == 2
        assert message_part in capsys.readouterr().err

    _refused(
        lambda p: p["profile_grid"]["b_eff_multipliers"].pop(),
        "multipliers")
    _refused(
        lambda p: p["profile_grid"].__setitem__(
            "b_dec_gev2", list(B_DEC_GRID) + [2.9]),
        "b_dec grid")
    _refused(
        lambda p: p["profile_grid"].__setitem__("b_eff_frozen_gev2", 8.0),
        "frozen B_eff")
    _refused(
        lambda p: p["profile_grid"]["reweighting_validity"]
        ["pinned_grid_points"].__setitem__(
            0, {"b_eff_multiplier": 1.5, "b_dec_gev2": 1.6}),
        "pinned grid points")
    _refused(
        lambda p: p["statistics"].__setitem__("replicas", 4),
        "statistics")
    _refused(
        lambda p: p["seed_map"]["raw_replica_seeds"].__setitem__(
            0, 20301021 + 1),
        "seeds")
    _refused(
        lambda p: p["seed_map"].__setitem__(
            "event_index_rule", "400_000_000 + i, i < 100,000"),
        "event-index rule")
    _refused(
        lambda p: p["obligation_4_carry"]["r32_spread"].__setitem__(
            "bin_spread_p95", 5.0),
        "obligation 4")
    _refused(
        lambda p: p.__setitem__("written_before_generation", False),
        "written_before_generation")
    _refused(
        lambda p: p.pop("conservation"),
        "conservation")
    # The smoke mode is exempt from the card assertions
    smoke = copy.deepcopy(pre_registration)
    smoke["mode"] = "smoke"
    smoke["profile_grid"] = {"b_eff_multipliers": [0.1]}
    driver._assert_pre_registration(smoke, RUN_DIR)


# ----------------------------------------------------------------------
# The sealed pre-registration
# ----------------------------------------------------------------------
def test_pre_registration_on_disk(driver, pre_registration):
    """The pre-registration's identity, T2 statistics, seed map, the
    9 x 9 profile grid, collision configuration, conservation and
    observables blocks, likelihood form, acceptance check set with its
    five register triggers, blend-band-sensitive bins, obligation 4
    carry, substrate verification, and sealed run dirs."""
    assert pre_registration["schema"] == (
        "na49_mixing.staged_r42_recoil_decay_separation.pre_registration/1")
    assert pre_registration["run"] == "R42"
    assert pre_registration["card"] == SOLVER_CARD
    assert pre_registration["mode"] == "production"
    assert pre_registration["written_before_generation"] is True
    # The T2 statistics (plan section 5.1 line 254)
    stats = pre_registration["statistics"]
    assert stats["replicas"] == 8
    assert stats["events_per_replica"] == 100_000
    assert stats["total_events"] == 800_000
    assert stats["tier"].startswith("T2: 8 replicas x 100,000")
    # The pre-registered seed map
    seeds = pre_registration["seed_map"]
    assert seeds["master_seed"] == MASTER_SEED
    assert seeds["raw_replica_seeds"] == list(RAW_REPLICA_SEEDS)
    assert seeds["event_index_rule"] == EVENT_INDEX_RULE
    assert seeds["kernel_seed_rule"].startswith(
        "derive_substreams(20260920, r, 500_000_000 + i)")
    assert seeds["raw_seed_rule"].startswith(
        "raw replica seed = 20260920 + 40000 + 100 + r = 20301020 + r")
    assert "500_000_000 + i" in seeds["disjointness"]
    assert seeds["smoke"]["raw_seed"] == SMOKE_RAW_SEED
    assert seeds["smoke"]["kernel_replica_slot"] == (
        SMOKE_KERNEL_REPLICA_SLOT)
    assert seeds["smoke"]["synthetic_truth"] == {
        "b_eff_multiplier": 1.3,
        "b_dec": 1.2,
    }
    # The pre-registered 9 x 9 profile grid (index item 10)
    grid = pre_registration["profile_grid"]
    assert grid["index_item"] == 10
    assert [float(m) for m in grid["b_eff_multipliers"]] == (
        list(B_EFF_MULTIPLIERS))
    assert [float(b) for b in grid["b_dec_gev2"]] == list(B_DEC_GRID)
    assert grid["b_eff_frozen_gev2"] == B_EFF_FROZEN_GEV2
    assert grid["b_dec_nominal_gev2"] == 1.0
    assert grid["b_dec_nominal_on_grid"] is False
    assert grid["b_dec_step_gev2"] == 0.3
    assert grid["b_eff_multiplier_step"] == 0.25
    validity = grid["reweighting_validity"]
    assert [tuple(map(float, (p["b_eff_multiplier"], p["b_dec_gev2"])))
            for p in validity["pinned_grid_points"]] == list(
        VALIDITY_GRID_POINTS)
    assert validity["bound_relative"] == 1e-06
    assert validity["pinned_subsample_events"] == 2_000
    # The collision configuration (round 4 index item 7)
    collision = pre_registration["collision"]
    assert collision["index_item"] == 7
    assert collision["w_flavor"] == 1.0
    assert collision["system"].startswith(
        "p+p at sqrt(s) = 17.270 GeV (beam 158 GeV/c)")
    assert "358989d8" in collision["configuration"]
    assert collision["frozen_b_eff_gev2"] == B_EFF_FROZEN_GEV2
    assert pre_registration["conservation"]["per_event"].startswith(
        "every p+p event conserves total (B, Q, S) = (2, 2, 0)")
    # The observables and likelihood
    assert pre_registration["observables"]["training"].startswith(
        "the leading-baryon transfer observable")
    assert "-2lnL" in pre_registration["likelihood"]["form"]
    assert pre_registration["likelihood"]["nuisances"]["carry_prior"] == 0.25
    assert pre_registration["likelihood"][
        "profile_is_not_fit_set"].startswith(
            "the nuisance profile is part of the objective definition")
    # The acceptance check set (plan section 24; index item 11)
    acceptance = pre_registration["acceptance"]
    assert set(acceptance) == {
        "i_stable_curvature", "ii_distinguishable_effects",
        "iii_reweighting_validity", "iv_conservation",
        "named_residual_register", "precedent", "register_triggers",
        "v_determinism", "vi_no_refit_beyond_profile",
        "vi_seal_and_clean_validation", "vii_pins_and_freeze"}
    assert acceptance["register_triggers"] == [
        "the profile minimum on the grid boundary of either axis "
        "(a boundary finding: the profile range does not extend)",
        "a parameter correlation at or above the pre-registered 0.9 "
        "bound (the distinguishability failure; the Gate 4 string-like "
        "rank law clause consumes it directly)",
        "a pooled Hessian at the minimum that is not positive "
        "definite, or any per-block Hessian failure (the curvature "
        "finding)",
        "an effective sample size fraction below 0.01 at the profile "
        "minimum (the weight-degeneracy diagnosis)",
        "a non-finite pooled surface value at any grid point "
        "(recorded; the minimum is taken over the finite points)",
    ]
    # The blend-band-sensitive bins (the sealed R32 persisting
    # residuals)
    bins = pre_registration["blend_band_sensitive_bins"]
    assert bins["entries"] == 32
    assert bins["sigma_range"][0] == 2.0001998039779263
    assert bins["sigma_range"][1] == 7.532929574244998
    assert bins["intersecting_species"] == {"proton": 9, "antiproton": 0}
    assert "persisting_bin_attribution.json" in bins["source"]
    # The obligation 4 carry with the audit outcomes
    carry = pre_registration["obligation_4_carry"]
    assert carry["obligation"] == 4
    assert carry["carry_prior"] == 0.25
    assert carry["r32_spread"] == OBLIGATION4_R32_SPREAD
    assert len(carry["blend_band_sensitive_bins"]) == 32
    assert carry["audit_outcome"]["card_2"].startswith("PASS")
    assert carry["audit_outcome"]["card_3"].startswith("PASS")
    assert carry["audit_outcome"]["card_1"].startswith("RECOMMENDATION")
    # The frozen substrate digests, verified on disk (the r41_driver
    # entry records the corrected card-16 R41 driver, committed at
    # the time this run executed)
    assert pre_registration["frozen_substrate"] == {
        key: digest for key, (_rel, digest) in SUBSTRATE.items()
    }
    for key, status in pre_registration["substrate"].items():
        assert status["verified"] is True
        assert status["sha256"] == SUBSTRATE[key][1]
    # The sealed run dirs, verified
    assert pre_registration["sealed_run_dirs"] == list(
        driver.SEALED_RUN_DIRS)
    status = pre_registration["sealed_run_dirs_status"]
    assert status["dev/00runs/r30_spectral_normalization_2026-09-17"] == (
        {"entries": 3, "mismatches": 0, "verified": True})
    assert status[
        "dev/00runs/r31_catalog_continuum_matching_scan_2026-09-17"] == (
            {"entries": 79, "mismatches": 0, "verified": True})
    assert status[
        "dev/00runs/r32_spectral_systematic_pp_2026-09-18"] == (
            {"entries": 29, "mismatches": 0, "verified": True})
    assert status[
        "dev/00runs/r32_spectral_systematic_pp_confirmation_2026-09-19"] == (
            {"entries": 55, "mismatches": 0, "verified": True})
    assert status[
        "dev/00runs/r40_fixed_source_alignment_scan_2026-09-21"] == (
            {"entries": 364, "mismatches": 0, "verified": True})
    assert status[
        "dev/00runs/r41_nonstrange_pilot_fit_2026-09-23"] == (
            {"entries": 13, "mismatches": 0, "verified": True})
    # The determinism block
    assert pre_registration["determinism"]["probe_replica"][
        "replica"] == 0
    # The plan anchors
    assert pre_registration["plan_anchors"]["run"] == (
        "plan section 10 run R42, line 786")
    assert pre_registration["plan_anchors"]["statistics_tier"] == (
        "plan section 5.1 line 254")
    assert pre_registration["plan_anchors"]["b_transport"] == (
        "plan section 7.3 line 431 (B_transport = p - pbar)")
    # The recording convention (the relitigation clause; acceptance
    # (viii) hole-geometry compliance): dense accumulation with the
    # pick map, holes dropped, and the superseded/corrected driver
    # digests recorded
    rec = pre_registration["recording_convention"]
    assert rec["convention"].startswith(
        "the R01 accumulate-dense-then-pick recording convention")
    assert rec["holes_dropped"] is True
    assert rec["pick_map"] == "bin_index_pt * n_x + bin_index_xf"
    assert rec["superseded_driver_sha256"] == SUPERSEDED_DRIVER_SHA256
    assert rec["corrected_driver_sha256"] == DRIVER_SHA256
    geo = rec["geometry"]
    assert geo["lattice_cells_total"] == 1296
    assert geo["holes_total"] == 279
    assert geo["published_rows_total"] == 1017
    assert sorted(geo["per_species"]) == [
        "antiproton", "k_minus", "k_plus", "pi_minus", "pi_plus",
        "proton"]
    # The pre-registration-time provenance (the committed driver: the
    # driver was not amended after the pre-registration)
    assert pre_registration["provenance"]["driver_sha256"] == (
        DRIVER_SHA256)
    # The pre-registration's on-disk bytes at the pinned digest
    assert _sha256_file(RUN_DIR / "pre_registration.json") == (
        SEALED_FILE_DIGESTS["pre_registration.json"])


# ----------------------------------------------------------------------
# The sealed profile report
# ----------------------------------------------------------------------
def test_profile_report_on_disk(driver, profile_report):
    """The report's schema, the internal digest, the full 9 x 9
    surface at its pinned values, the grid argmin (the boundary
    minimum), the pooled Hessian block (not positive definite), the
    undefined correlation, the per-block curvature stability (seven
    of eight blocks positive definite; block 4 alone failing), the
    five-entry named-residual register, the per-block ESS, and the
    pooled objective at the minimum."""
    assert profile_report["schema"] == (
        "na49_mixing.staged_r42_recoil_decay_separation.report/1")
    assert profile_report["run"] == "R42"
    assert profile_report["card"] == SOLVER_CARD
    assert profile_report["mode"] == "production"
    assert profile_report["digest"] == PROFILE_REPORT_DIGEST
    # The pinned 9 x 9 surface (81 values; the argmin at row 0, col 8)
    assert profile_report["surface"] == SURFACE
    assert len(profile_report["surface"]) == 9
    for row in profile_report["surface"]:
        assert len(row) == 9
    assert profile_report["nonfinite_surface_points"] == []
    # The profile grid
    grid = profile_report["profile_grid"]
    assert [float(m) for m in grid["b_eff_multipliers"]] == (
        list(B_EFF_MULTIPLIERS))
    assert [float(b) for b in grid["b_dec_gev2"]] == list(B_DEC_GRID)
    assert grid["b_eff_frozen_gev2"] == B_EFF_FROZEN_GEV2
    # The grid argmin (on the boundary of both axes)
    assert profile_report["minimum"] == MINIMUM
    # The pooled Hessian block (not positive definite; boundary
    # curvature)
    hessian = profile_report["hessian"]
    assert hessian["positive_definite"] is False
    assert hessian["eigenvalues"] == HESSIAN_EIGENVALUES
    assert hessian["hessian"] == HESSIAN
    assert hessian["interior_flags"] == {
        "all_interior": False,
        "i0": 0,
        "interior_bb": False,
        "interior_kk": False,
        "j0": 8,
    }
    # The correlation: undefined at the boundary minimum
    assert profile_report["correlation_b_eff_b_dec"] is None
    assert profile_report["distinguishability_bound"] == 0.9
    # The per-block curvature stability (acceptance (i); seven of
    # eight blocks positive definite, block 4 alone failing)
    stability = profile_report["curvature_stability"]
    assert set(stability) == {str(b) for b in range(8)}
    for block, entry in stability.items():
        assert entry["positive_definite"] is (block != "4")
        assert entry["eigenvalues"] == (
            BLOCK_CURVATURE_EIGENVALUES[block])
    # The five-entry named-residual register
    register = profile_report["named_residual_register"]
    assert len(register) == 5
    assert [entry["index"] for entry in register] == [1, 2, 3, 4, 5]
    assert [entry["kind"] for entry in register] == [
        "profile_minimum_on_boundary", "profile_minimum_on_boundary",
        "correlation_undefined", "curvature_not_positive",
        "block_curvature_failure"]
    assert register[0]["axis"] == "b_eff_multiplier"
    assert register[0]["observed"] == 0.5
    assert register[0]["bound"] == [0.5, 2.5]
    assert register[1]["axis"] == "b_dec"
    assert register[1]["observed"] == 2.6
    assert register[1]["bound"] == [0.2, 2.6]
    assert register[2]["kind"] == "correlation_undefined"
    assert register[2]["observed"] is None
    assert register[2]["bound"] == 0.9
    assert register[3]["observed"]["eigenvalues"] == HESSIAN_EIGENVALUES
    assert register[3]["observed"]["hessian"] == HESSIAN
    # The block-curvature entry names the single failing block
    assert register[4]["observed"] == {
        "4": BLOCK_CURVATURE_EIGENVALUES["4"]}
    for entry in register:
        assert entry["robust_statistics_cross_reference"].startswith(
            "the between-replica standard error")
    # The per-block ESS
    assert profile_report["per_block_ess"] == PER_BLOCK_ESS
    # The pooled objective at the minimum
    objective = profile_report["objective_at_minimum"]
    assert objective["ess"]["min"] == OBJECTIVE["ess_min"]
    assert objective["ess"]["per_block"] == PER_BLOCK_ESS
    assert objective["minus2lnl_profiled"] == (
        OBJECTIVE["minus2lnl_profiled"])
    assert objective["n_eff_total"] == OBJECTIVE["n_eff_total"]
    assert objective["nuisance_columns"] == OBJECTIVE["nuisance_columns"]
    assert objective["nuisances"] == OBJECTIVE["nuisances"]
    assert len(objective["nuisances"]) == 2
    # The optimizer contract (the round's second and final use)
    assert profile_report["optimizer_invoked"] is True
    assert profile_report["optimizer_uses"] == 2
    assert profile_report["optimizer_note"].startswith(
        "the profile is the round's second and final optimizer use")
    # The reweighting validity (the pinned sub-sample, both points)
    validity = profile_report["reweighting_validity"]
    assert validity["pass"] is True
    assert validity["bound_relative"] == 1e-06
    assert validity["max_relative_deviation"] == (
        1.4875428853980945e-14)
    assert validity["pinned_grid_points"] == 2
    assert validity["pinned_subsample_events"] == 2_000
    assert len(validity["points"]) == 2
    for point in validity["points"]:
        assert point["transfer_law"]["pass"] is True
        assert point["form_factor"]["pass"] is True
    # The model-uncertainty carry (obligation 4)
    carry = profile_report["model_uncertainty_carry"]
    assert carry["obligation_4"] is True
    assert carry["carry_prior"] == 0.25
    assert carry["r32_spread"] == OBLIGATION4_R32_SPREAD
    # The dropped bins and the no-refit clause
    assert profile_report["dropped_bins"] == []
    assert profile_report["n_dropped_bins"] == 0
    assert profile_report["no_refit"].startswith(
        "the profile axes are exactly the two pre-registered "
        "parameters")
    assert profile_report["determinism"]["report_note"].startswith(
        "no wall-clock fields")
    # The report's on-disk bytes at the pinned digest
    assert _sha256_file(RUN_DIR / "profile_report.json") == (
        SEALED_FILE_DIGESTS["profile_report.json"])


def test_gate4_evidence(driver, gate_record, profile_report):
    """The gate record's identity, its clause (the string-like rank
    law), its conservation summary (zero violations over 800,000
    events), the profile-report digest cross-reference, and the
    equality of its Hessian, curvature-stability, carry, and
    named-residual blocks with the report's (its minimum and
    reweighting-validity blocks are the report's minus the
    report-level profile fields)."""
    assert gate_record["schema"] == (
        "na49_mixing.staged_r42_recoil_decay_separation.gate/1")
    assert gate_record["run"] == "R42"
    assert gate_record["card"] == SOLVER_CARD
    assert gate_record["gate"] == "gate 4 (plan section 10, line 806)"
    assert gate_record["clause"] == (
        "string-like rank law (recoil/decay separation)")
    assert gate_record["sealed"] is True
    # The conservation summary (zero violations over 800,000 events)
    assert gate_record["events_checked"] == 800_000
    conservation = gate_record["conservation"]
    assert conservation["violations"] == 0
    assert conservation["pass"] is True
    assert conservation["tolerance"] == 1e-09
    assert conservation["events_checked"] == 800_000
    assert conservation["worst_energy_closure_gev"] == (
        3.481659405224491e-13)
    # The profile-report cross-references
    assert gate_record["profile_report_digest"] == PROFILE_REPORT_DIGEST
    assert gate_record["profile_report_digest"] == (
        profile_report["digest"])
    # The gate's minimum block is the report's minimum minus the
    # report-level grid-index and -2lnL fields
    assert gate_record["minimum"] == {
        k: v for k, v in profile_report["minimum"].items()
        if k in ("b_dec", "b_eff_gev2", "b_eff_multiplier")}
    # The gate's reweighting-validity block is the report's minus the
    # report-level pinned-subsample and points fields
    assert gate_record["reweighting_validity"] == {
        k: v for k, v in profile_report["reweighting_validity"].items()
        if k in ("bound_relative", "max_relative_deviation", "pass",
                 "pinned_grid_points")}
    assert gate_record["hessian_eigenvalues"] == HESSIAN_EIGENVALUES
    assert gate_record["optimizer_invoked"] is True
    assert gate_record["optimizer_uses"] == 2
    assert gate_record["correlation_b_eff_b_dec"] is None
    assert gate_record["distinguishability_bound"] == 0.9
    # The equality blocks
    assert gate_record["hessian"] == profile_report["hessian"]
    assert gate_record["curvature_stability"] == (
        profile_report["curvature_stability"])
    # The gate's carry block is the report's carry block minus the
    # carry_prior (the prior is a report-level field)
    assert gate_record["model_uncertainty_carry"] == {
        k: v for k, v in profile_report["model_uncertainty_carry"].items()
        if k != "carry_prior"}
    assert gate_record["named_residual_register"] == (
        profile_report["named_residual_register"])
    assert gate_record["sealed_note"].startswith(
        "the run seals after this record")
    # The gate record's on-disk bytes at the pinned digest
    assert _sha256_file(RUN_DIR / "gate4_r42.json") == (
        SEALED_FILE_DIGESTS["gate4_r42.json"])


# ----------------------------------------------------------------------
# The seal manifest and the checksum file
# ----------------------------------------------------------------------
def test_seal_manifest_and_checksums(seal_manifest):
    """The seal manifest's 13 sealed files (8 replica records plus the
    five top-level files) path-sorted, the checksum file's 13 entries,
    and the full re-hash of all 13 sealed files against the checksum
    digests."""
    assert seal_manifest["schema"] == (
        "na49_mixing.staged_r42_recoil_decay_separation.seal/1")
    assert seal_manifest["run"] == "R42"
    assert seal_manifest["card"] == SOLVER_CARD
    assert seal_manifest["record_count"] == 8
    assert seal_manifest["sha256sums_entries"] == 13
    assert seal_manifest["files"] == list(sorted(SEALED_FILES))
    assert len(seal_manifest["files"]) == 13
    # The run continued past UTC midnight, so the seal timestamp is
    # dated one day after the run dir
    assert seal_manifest["sealed_utc"].startswith("2026-09-24")
    assert seal_manifest["run_dir"].endswith(
        "dev/00runs/r42_recoil_decay_separation_2026-09-23")
    # The checksum file: 13 entries, run-dir-relative names, the
    # self-exclusion, and the full re-hash
    sums_path = RUN_DIR / "SHA256SUMS.txt"
    lines = [
        line for line in sums_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 13
    assert sealed_mismatches(RUN_DIR) == []
    names = []
    for line in lines:
        digest, name = line.split("  ", 1)
        names.append(name)
        assert digest == SEALED_FILE_DIGESTS[name]
    assert names == sorted(names), "the checksum entries are name-sorted"
    assert "SHA256SUMS.txt" not in names
    assert "seal_manifest.json" not in names
    # The seal manifest's own on-disk digest
    assert _sha256_file(RUN_DIR / "seal_manifest.json") == (
        SEALED_FILE_DIGESTS["seal_manifest.json"])
    assert _sha256_file(sums_path) == CHECKSUMS_SHA256


# ----------------------------------------------------------------------
# The clean-environment validation record
# ----------------------------------------------------------------------
def test_clean_environment_record(clean_environment, profile_report):
    """The clean-environment validation's command list, the
    byte-identical report re-hash (digest equality), the probe replica
    record, and the pending seal state."""
    assert clean_environment["schema"] == (
        "na49_mixing.staged_r42_recoil_decay_separation.clean_env/1")
    assert clean_environment["run"] == "R42"
    assert clean_environment["card"] == SOLVER_CARD
    commands = clean_environment["commands"]
    assert len(commands) == 3
    assert commands[0] == (
        "cd dev/mixing2006 && python -m pytest "
        "tests/test_staged_r42_recoil_decay_separation.py -q")
    assert commands[1] == (
        "python -m pytest tests/test_legacy_v1_freeze.py "
        "-W error::UserWarning -q")
    assert commands[2].startswith(
        "python helpers/benchmarking/staged_r42_recoil_decay_separation.py "
        "--clean-validate --run-dir ")
    assert "r42_recoil_decay_separation_2026-09-23" in commands[2]
    # The byte-identical report re-hash
    assert clean_environment["report_hash_match"] is True
    assert clean_environment["profile_report_digest"] == (
        PROFILE_REPORT_DIGEST)
    assert clean_environment["profile_report_rerun_digest"] == (
        PROFILE_REPORT_DIGEST)
    assert clean_environment["profile_report_digest"] == (
        profile_report["digest"])
    # The probe replica (setting-independent full replica re-run)
    probe = clean_environment["probe_replica"]
    assert probe["replica"] == 0
    assert probe["matches_sealed_record"] is True
    # The pending seal state (the validation runs before the seal)
    assert clean_environment["seal"]["state"].startswith("pending")
    assert clean_environment["seal"]["entries"] == 0
    assert clean_environment["seal"]["mismatches"] == []
    assert clean_environment["verified_utc"].startswith("2026-09-24")


# ----------------------------------------------------------------------
# The provenance
# ----------------------------------------------------------------------
def test_provenance_on_disk(provenance, driver):
    """The provenance's schema, the substrate digests, the M8 frozen
    set hash, the recorded driver digest (the committed dual-tree
    digest: the driver was not amended after the pre-registration),
    the nominal point, and the sealed run dir verification."""
    assert provenance["schema"] == (
        "na49_mixing.staged_r42_recoil_decay_separation.provenance/1")
    assert provenance["run"] == "R42"
    assert provenance["card"] == SOLVER_CARD
    # The substrate digests, against the frozen table
    assert provenance["builder_sha256"] == (
        SUBSTRATE["builder_blend_path"][1])
    assert provenance["card6_surface_sha256"] == SUBSTRATE["card6_routing"][1]
    assert provenance["engine_sha256"] == SUBSTRATE["card2_engine"][1]
    assert provenance["catalog_sha256"] == SUBSTRATE["catalog"][1]
    assert provenance["continuum_config_sha256"] == (
        SUBSTRATE["continuum_config"][1])
    assert provenance["round2_kernel_sha256"] == (
        SUBSTRATE["round2_kernel"][1])
    assert provenance["r32_driver_sha256"] == SUBSTRATE["r32_driver"][1]
    assert provenance["r40_driver_sha256"] == SUBSTRATE["r40_driver"][1]
    # The corrected card-16 R41 driver was already committed at the
    # time this run executed (the card-17 relitigation re-pointed the
    # sibling pin before generation), so the recorded digest is the
    # committed dual-tree digest
    assert provenance["r41_driver_sha256"] == SUBSTRATE["r41_driver"][1]
    assert provenance["m8_frozen_set_hash"] == M8_PARAMETER_HASH
    assert provenance["frozen_b_eff_gev2"] == B_EFF_FROZEN_GEV2
    assert provenance["cascade_module_sha256"] == {
        name: SUBSTRATE[name][1]
        for name in ("cascade_decay", "cascade_axes", "cascade_alignment",
                     "cascade_form_factor", "cascade_termination",
                     "cascade_ablation")
    }
    # The recorded driver digest: the committed dual-tree digest (the
    # driver was not amended after the pre-registration, unlike R41)
    assert provenance["new_driver_sha256_both_trees"] == DRIVER_SHA256
    assert provenance["driver_note"].startswith(
        "byte-identical in the nested tree")
    assert provenance["nominal_point"].startswith(
        "the R40 pre-registered nominal alignment point")
    assert provenance["r41_predecessor"].startswith(
        "the sealed R41 pilot fit archive")
    # The superseded archives (the card-13 seals, intact and verified
    # as the supersession reference)
    superseded = provenance["superseded_r42_archives"]
    assert [entry["run_dir"] for entry in superseded] == (
        list(SUPERSEDED_RUN_DIRS))
    for entry in superseded:
        assert entry["seal"] == {"entries": 13, "mismatches": 0,
                                 "verified": True}
        assert entry["state"].startswith(
            "intact; the seal verifies (verify_sealed_dir)")
        assert entry["superseded_driver_sha256"] == (
            SUPERSEDED_DRIVER_SHA256)
    # The sealed run dir verification
    assert provenance["sealed_run_dirs"] == {
        "dev/00runs/r30_spectral_normalization_2026-09-17":
            {"entries": 3, "mismatches": 0, "verified": True},
        "dev/00runs/r31_catalog_continuum_matching_scan_2026-09-17":
            {"entries": 79, "mismatches": 0, "verified": True},
        "dev/00runs/r32_spectral_systematic_pp_2026-09-18":
            {"entries": 29, "mismatches": 0, "verified": True},
        "dev/00runs/r32_spectral_systematic_pp_confirmation_2026-09-19":
            {"entries": 55, "mismatches": 0, "verified": True},
        "dev/00runs/r40_fixed_source_alignment_scan_2026-09-21":
            {"entries": 364, "mismatches": 0, "verified": True},
        "dev/00runs/r41_nonstrange_pilot_fit_2026-09-23":
            {"entries": 13, "mismatches": 0, "verified": True},
    }
    assert provenance["written_utc"] == "2026-09-23T23:53:12Z"
    # The pre-registration carries the same pre-registration-time
    # driver digest
    pre = json.loads(
        (RUN_DIR / "pre_registration.json").read_text(encoding="utf-8"))
    assert pre["provenance"]["driver_sha256"] == DRIVER_SHA256


# ----------------------------------------------------------------------
# The frozen substrate, byte-stable on disk
# ----------------------------------------------------------------------
def test_frozen_substrate_stability(driver):
    """The driver's fifteen frozen substrate files byte-stable on
    disk at their pinned digests (verified independently of the
    driver's table)."""
    for name, (rel, digest) in SUBSTRATE.items():
        path = MONO_ROOT / rel
        assert path.is_file(), f"missing frozen surface: {rel}"
        assert _sha256_file(path) == digest, f"drifted surface: {rel}"


# ----------------------------------------------------------------------
# The hermetic solver test file
# ----------------------------------------------------------------------
def test_solver_test_file_digest(manifest):
    """The hermetic solver test file at its committed digest, pinned
    in the freeze manifest inventory, with the card-13 solver freeze
    exception intact (the four path-sorted files)."""
    solver_path = MONO_ROOT / SOLVER_TEST_REL
    assert solver_path.is_file(), f"missing solver test: {SOLVER_TEST_REL}"
    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)
    inventory = manifest["source_tree_sha256"]
    assert inventory[SOLVER_TEST_REL] == SOLVER_TEST_SHA256, (
        "the freeze manifest does not pin the solver test at its "
        "committed digest")
    # The card-13 solver freeze exception stays untouched: exactly one
    # entry naming the solver's card, at the solver's 4 path-sorted
    # files
    exceptions = manifest["freeze_exceptions"]
    solver_entries = [
        e for e in exceptions if e.get("card") == SOLVER_CARD]
    assert len(solver_entries) == 1, (
        "the card-13 solver freeze exception must name its card once")
    solver_files = solver_entries[0]["files"]
    assert solver_files == sorted(solver_files), (
        "the card-13 solver freeze-exception files are path-sorted")
    assert solver_files == list(SOLVER_EXCEPTION_FILES), (
        "the card-13 solver freeze-exception files drifted")


# ----------------------------------------------------------------------
# The freeze re-pin register (the test-author re-pin)
# ----------------------------------------------------------------------
