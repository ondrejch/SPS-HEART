"""Round 4, cards 14/18 test-author register: the R43 cascade-ablation
contract pins.

Card ``TASK-20260920-r4-run-r43-cascade-ablation`` (plan section 10
run R43, lines 790-798; the Round 4 card set, card 14), re-pointed by
the card-18 relitigation (``TASK-20260920-r4-run-r43-relitigation``;
the corrected dense-lattice recording convention, mirroring the
card-16/card-17 re-point precedent).  This register is the
test-author half of the card: independent pins of the dual-tree run
driver (``helpers/benchmarking/staged_r43_cascade_ablation.py``,
byte-identical in the nested tree and the monorepo mirror), of the
hermetic solver test
(``tests/test_staged_r43_cascade_ablation.py``), and of the sealed run
``dev/00runs/r43_cascade_ablation_2026-09-23/``.
The superseded sealed run ``dev/00runs/r43_cascade_ablation_2026-09-22/``
stays intact and read-only; its driver digest, solver-test digest,
seal state, and two-entry named-residual register are retained below
as superseded constants (provenance only, never read from disk here).
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
  file-name constants, the substrate and sealed-dir rel paths, the
  card chain, the R32 p+p collision configuration, the three-variant
  register and policies (the nominal point held fixed, not the R41
  fitted values), the support boundary, the beyond-isotropic factor,
  the seed map, and the obligation 4 carry.
- ``test_seed_map_partition``: the master seed, the raw block
  20301020-20301027, the event-index partition 600_000_000 + i, the
  variant-stream rule, the hermetic smoke block (+9000), and the
  seed functions' refusal discipline.
- ``test_cli_surface``: the exact argparse command surface on disk
  (the seven mode flags, the three auxiliary flags) plus the on-disk
  exit-code behavior (no mode refuses with 2, two modes refuse with 2).
- ``test_pre_registration_on_disk``: the pre-registration's schema,
  the T2 statistics, the seed map, the collision configuration, the
  three-variant register and policies, the nominal point, the support
  boundary, the beyond-isotropic metric (the 2.0 threshold), the
  conservation block, the obligation 4 carry, the frozen substrate,
  the sealed run dirs, and the no-fitting clause.
- ``test_ablation_report_on_disk``: the report's schema, the internal
  digest, the per-variant deviance table, the beyond-isotropic
  verdict (all_pass false at the pre-registered 2.0 x pooled SE
  threshold; three of the four observables pass), the depth
  statistics, the retention, the per-variant multiplicity means, the
  exchange bit-identity, the one-entry named-residual register, the
  stream pin, and the no-fitting clause.
- ``test_gate4_evidence``: the gate record's identity, its clause
  (beyond isotropic sequential), its conservation summary (zero
  violations over 800,000 events), the exchange bit-identity block,
  the deviance table, the beyond-isotropic verdict (pass false with
  the one named finding), the support boundary, the
  model-uncertainty carry, the named-residual register, and the
  no-fitting clause.
- ``test_seal_manifest_and_checksums``: the seal manifest's 13 sealed
  files (8 replica records plus the five top-level files) path-sorted,
  the checksum file's 13 entries, the 15-file run dir (13 payload plus
  the two meta files), and the full re-hash of all 13 sealed files
  against the checksum digests.
- ``test_clean_environment_record``: the clean-environment
  validation's command list, the byte-identical report re-hash
  (digest equality), the probe replica record, and the pending seal
  state.
- ``test_provenance_on_disk``: the provenance's schema, the substrate
  digests, the M8 frozen set hash, the recorded driver digest (the
  committed dual-tree digest: the relitigation pre-registered
  against the committed corrected driver), the superseded archive's
  verified seal state as the supersession reference, the nominal
  point, and the sealed run dir verification.
- ``test_frozen_substrate_stability``: the driver's sixteen frozen
  substrate files byte-stable on disk at their pinned digests
  (the r41_driver and r42_driver entries are the corrected
  card-16/card-17 digests).
- ``test_solver_test_file_digest``: the hermetic solver test file at
  its committed digest, pinned in the freeze manifest inventory, with
  the card-14 solver freeze exception intact (the solver landing's 22
  path-sorted files).
- ``test_freeze_repin_register``: the freeze-manifest re-pin for this
  register (the strict path-sorted inventory at 488 entries carries
  this file at its on-disk digest; the card-14 solver freeze
  exception stays at the solver landing's 22 path-sorted files; this
  register's own freeze exception names the test-author card exactly
  once with the 21 path-sorted files; the nineteen consumed count
  pins -- the sixteen chain files plus the card-13 solver test's, the
  card-13 test-author register's, and the card-14 solver test's own
  count constants -- are re-pinned 487 -> 488 with the naming comment
  attributing the bump to this register, the card-14 solver's 484 ->
  487 comment, the card-13 test-author's 483 -> 484 comment, the
  card-13 solver's 480 -> 483 comment, the card-12 and card-10
  comments, and the card-10 solver's 471 literal retained for their
  re-pin checks).

Hermetic: standard library and numpy, the committed config and
catalog, and the sealed run dir (read-only) only; no SPS tables, no
event generation, no Monte Carlo run.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
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
    / "staged_r43_cascade_ablation.py"
)
OUTER_DRIVER = (
    MONO_ROOT / "helpers" / "benchmarking"
    / "staged_r43_cascade_ablation.py"
)
MANIFEST_PATH = (NESTED_ROOT / "config" / "v2_singlet"
                 / "legacy_v1_manifest.json")
RUN_DIR = (MONO_ROOT / "dev" / "00runs"
           / "r43_cascade_ablation_2026-09-23")

#: The superseded sealed run dir (the card-14 archive; stays intact
#: and is pinned as the supersession reference via the provenance's
#: superseded_r43_archive block, not read directly by this register).
SUPERSEDED_RUN_DIR_NAME = (
    "dev/00runs/r43_cascade_ablation_2026-09-22"
)

DRIVER_REL_NESTED = (
    "dev/mixing2006/helpers/benchmarking/"
    "staged_r43_cascade_ablation.py"
)
DRIVER_REL_OUTER = (
    "helpers/benchmarking/staged_r43_cascade_ablation.py"
)
SOLVER_TEST_REL = (
    "dev/mixing2006/tests/test_staged_r43_cascade_ablation.py"
)

#: The committed corrected dual-tree driver digest (byte-identical in
#: both trees; the freeze manifest inventory pins both copies here;
#: the card-18 relitigation rework of the recording convention).
DRIVER_SHA256 = (
    "dfb5ee3f4ee6598698faaf3da641fbf315ba21d95f2203792a2414a72e11801c"
)

#: The superseded card-14 driver digest (the published-row flat index
#: recording; the driver bytes were superseded by the relitigation).
#: Pinned as the supersession reference in the pre-registration's
#: recording_convention block, the driver's own SUPERSEDED_DRIVER_
#: SHA256 constant, and the provenance's superseded_r43_archive
#: block.
SUPERSEDED_DRIVER_SHA256 = (
    "4c6bc0f32430577ff80e607e19f2512bf2ed1c40701f1f115ff82a88c975bf3a"
)

#: The driver digest the sealed provenance records at pre-registration
#: time.  The relitigation pre-registered against the committed
#: corrected driver, so this run carries no distinct
#: pre-registration-time digest (unlike the card-14 landing, where
#: the driver was amended after the pre-registration).
PROVENANCE_DRIVER_SHA256 = (
    "dfb5ee3f4ee6598698faaf3da641fbf315ba21d95f2203792a2414a72e11801c"
)

#: The hermetic solver test file's committed digest (17 hermetic
#: tests: the card-14 13 plus the four mirrored hole-geometry tests;
#: the card-18 relitigation landing; the card-20 count re-pin
#: 491 -> 495 touched the INVENTORY_COUNT constant, so the digest
#: moved from the card-18 landing digest 864611ac374017c1...; the
#: Round 5 card 1 count re-pin 499 -> 501 touched the same constant,
#: so the digest moved from the card-21 landing digest 605d11b9be65...;
#: the Round 5 card 1 test-author count re-pin 501 -> 502 touched the
#: same constant, so the digest moved from the Round 5 card 1 landing
#: digest b192096e31b46a63...).
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
SOLVER_TEST_SHA256 = (
    "f53771a8183ec3deb6b0723afd7b4938d322e77f357f55a05d6d679299660c0c"
)

#: The superseded card-14 solver test file's digest (13 hermetic
#: tests; the card-14 landing).  Retained as the supersession
#: reference; the committed test file was extended by the
#: relitigation's four mirrored hole-geometry tests.
SUPERSEDED_SOLVER_TEST_SHA256 = (
    "e6e2c0c6c822c1603fa50da8c647b81d5a4f42fa792945b04d8b7e60df6faf11"
)

# ----------------------------------------------------------------------
# Freeze re-pin constants (the register chain's inventory count pins)
# ----------------------------------------------------------------------
MY_TEST_REL = (
    "dev/mixing2006/tests/"
    "test_staged_r43_cascade_ablation_contract_pins.py"
)
MANIFEST_REL = (
    "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json"
)
SOLVER_CARD = "TASK-20260920-r4-run-r43-cascade-ablation"
# The test-author re-pin's own freeze-exception card name (distinct
# from the solver's, so the solver's "exactly one entry naming this
# card" check is untouched).
TEST_AUTHOR_CARD = (
    "TASK-20260920-r4-run-r43-cascade-ablation-test-author"
)

#: The nineteen register-chain files whose inventory-count pins this
#: register re-pins (487 -> 488, +1 entry): the sixteen
#: register-chain files (cards 1, 4, 5, 6, 7, 8, 9, 10, 11, 12) plus
#: the card-13 solver test's own count constant
#: (test_staged_r42_recoil_decay_separation.py), the card-13
#: test-author register's own count constant
#: (test_staged_r42_recoil_decay_separation_contract_pins.py), and the
#: card-14 solver test's own count constant
#: (test_staged_r43_cascade_ablation.py).
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
    "dev/mixing2006/tests/"
    "test_staged_r42_recoil_decay_separation.py",
    "dev/mixing2006/tests/"
    "test_staged_r42_recoil_decay_separation_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_staged_r43_cascade_ablation.py",
)

#: The card-14 solver freeze-exception file list (22 path-sorted
#: files; the solver landing plus its chain re-pins; this register
#: leaves it untouched).
SOLVER_EXCEPTION_FILES = (
    MANIFEST_REL,
    DRIVER_REL_NESTED,
    "dev/mixing2006/tests/test_cascade_ablation.py",
    "dev/mixing2006/tests/test_cascade_alignment_baryonic.py",
    "dev/mixing2006/tests/test_cascade_alignment_mesonic.py",
    "dev/mixing2006/tests/test_cascade_axes.py",
    "dev/mixing2006/tests/test_cascade_form_factor.py",
    "dev/mixing2006/tests/test_cascade_termination.py",
    "dev/mixing2006/tests/test_r4_audit_blend_path_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_ablation_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_alignment_baryonic_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_alignment_mesonic_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_axes_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_decay_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_form_factor_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_termination_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r40_alignment_scan_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r41_nonstrange_pilot_fit_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r42_recoil_decay_separation.py",
    "dev/mixing2006/tests/test_staged_r42_recoil_decay_separation_contract_pins.py",
    SOLVER_TEST_REL,
    DRIVER_REL_OUTER,
)

#: This register's own freeze-exception file list (path-sorted): the
#: manifest, this register, and the nineteen re-pinned count pins
#: (the card-14 solver test is among them).
MY_EXCEPTION_FILES = tuple(
    sorted(set([MANIFEST_REL, MY_TEST_REL]
               + list(CHAIN_COUNT_PIN_FILES)))
)

# 487 entries at the card-14 solver landing (the dual-tree R43 driver
# in both helper trees and the hermetic R43 test file); this register
# adds its file (+1 entry) for the test-author re-pin (487 -> 488,
# +1 entry) and re-pins the nineteen consumed count pins (the sixteen
# register-chain files plus the card-13 solver test's, the card-13
# test-author register's, and the card-14 solver test's own count
# constants) with the bump attributed to this register's file.
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
    # The committed R41 driver (the card-16 relitigation rework)
    "r41_driver": (
        "helpers/benchmarking/staged_r41_nonstrange_pilot_fit.py",
        "abf2a8c30344ebeecab4413f5d1c4045b091ac3467226c6257e09d501697cdfc",
    ),
    # The committed R42 driver (the card-17 relitigation rework)
    "r42_driver": (
        "helpers/benchmarking/staged_r42_recoil_decay_separation.py",
        "a1917e578dfcc56b8f8c73652bc893d3aee21b870cdafeca4b46dfd7ad4961fd",
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
    "ablation_report.json",
    "clean_environment_commands.json",
    "gate4_r43.json",
    "pre_registration.json",
    "provenance.json",
)

#: The on-disk digest of every sealed file (the checksum file's
#: entries plus the seal manifest's own digest).
SEALED_FILE_DIGESTS = {
    "ablation_report.json": (
        "ac2b8846703ff4f459209d03f3283df61868c93e41d4c20749a1f2112fc607c7"
    ),
    "clean_environment_commands.json": (
        "725f338ffc0097985f2c896cf9a34619f03a465b2d3c0cc0abf1bc5676c25c06"
    ),
    "gate4_r43.json": (
        "6541159fefe410574d27a8fc0b0d1fd4f0f625beee007ccb8ad1b3efe6df4532"
    ),
    "pre_registration.json": (
        "f3d647a2406edf7b0fc585c7355f8cf3b9ecf0626c906d7f24af3fd3d4d092d3"
    ),
    "provenance.json": (
        "38206d920288a73fdb149fc5f0ac4331baac8cda7ee983c1e767e0371b22d9d1"
    ),
    "replica_0.json": (
        "b442e168e509863fbde60c72d2f6a05493d3bb76df24ccedb7bb56092b69a82a"
    ),
    "replica_1.json": (
        "07cffe780569f1e57893c86b60db54dd3f97eb4d1c50b45f32201ec65161d61c"
    ),
    "replica_2.json": (
        "eb54636612d9a97d24fc6628a5a2d91874b4313e8435255e21631867fe452ff2"
    ),
    "replica_3.json": (
        "fbfd9e91242a1cc786d69a1625059cb9e5f902ed03dee679e30721dd539b4b6e"
    ),
    "replica_4.json": (
        "444e0cabe0e15bba67bfef0a3e93c1d97191b781a9abd035fe3dd5ed2b809d87"
    ),
    "replica_5.json": (
        "46085cbcb79b78ae3f2a61618c2f32f083569292a726229040a40550a407159b"
    ),
    "replica_6.json": (
        "324b410ba501e6369a369def3a6c3a69038a66a3984a21f44bf74f1269c4e03f"
    ),
    "replica_7.json": (
        "d0d892475dc23a2c3ebe6d7e9498205958f83a63943dc377b3813a49d90570b7"
    ),
    "seal_manifest.json": (
        "234347fc082c21bf0e242d58cb04bc1bfe41ce3b7436d5cfcc7b04bee8e47a02"
    ),
}

#: The checksum file's own digest (excluded from its own entries).
CHECKSUMS_SHA256 = (
    "d0ef45ea7cf7c693e57c4bb713d8f28d3225749a2e900a0e3f7fb34a764b99d5"
)

#: The ablation report's internal digest (the report bytes carry no
#: wall-clock fields).
ABLATION_REPORT_DIGEST = (
    "6995b640c58648893ab4ee11635a1bee6dc8b9c3b5b775a9d79708b305069d40"
)

#: The superseded sealed archive's verified seal state (the card-14
#: archive stays intact and read-only; pinned here as the
#: supersession reference via the provenance's superseded_r43_archive
#: block, never re-hashed by this register).
SUPERSEDED_SEAL_STATE = {"entries": 13, "mismatches": 0, "verified": True}

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
EVENT_INDEX_BASE = 600_000_000
EVENT_INDEX_RULE = "600_000_000 + i, i < 100,000"
SMOKE_RAW_SEED = 20309920
SMOKE_KERNEL_REPLICA_SLOT = 9_000

#: The pre-registered nominal point (v3's alignment point, not the
#: R41 fitted values).
NOMINAL_KAPPA_B = 1.0
NOMINAL_KAPPA_M = 1.0
NOMINAL_B_DEC_GEV2 = 1.0

#: The pre-registered beyond-isotropic threshold factor.
BEYOND_ISOTROPIC_FACTOR = 2.0

#: The pre-registered obligation 4 carry (the sealed R32 spread,
#: verbatim from the card).
OBLIGATION4_R32_SPREAD = {
    "bin_spread_p95": 4.028426,
    "feeddown_spread_max_sigma": 31.1773,
    "persisting_residuals": 32,
}

#: The per-variant deviance table (v1, v2, v3) at the sealed values.
DEVIANCE_TABLE = {
    "b_transport_seagull": {
        "aligned_sequential": 0.0004913296781584034,
        "isotropic_direct": 0.0009704591017265884,
        "isotropic_sequential": 0.00046736991233405415,
    },
    "b_transport_spectrum": {
        "aligned_sequential": 79880.86043563992,
        "isotropic_direct": 112182.03191424314,
        "isotropic_sequential": 86285.96471204117,
    },
    "pi_sigma_seagull": {
        "aligned_sequential": 0.10525693002236887,
        "isotropic_direct": 0.10386800817506077,
        "isotropic_sequential": 0.11588313527413957,
    },
    "pi_sigma_spectrum": {
        "aligned_sequential": 502002.22845741047,
        "isotropic_direct": 3119509.226308894,
        "isotropic_sequential": 592105.1930505438,
    },
}

#: The beyond-isotropic verdict (v3 minus v2) at the sealed values.
BEYOND_ISOTROPIC_V3_MINUS_V2 = {
    "b_transport_seagull": -2.3959765824349295e-05,
    "b_transport_spectrum": 6405.104276401253,
    "pi_sigma_seagull": 0.010626205251770701,
    "pi_sigma_spectrum": 90102.9645931333,
}

#: The pooled MC standard error of the per-observable difference at
#: the sealed values.
BEYOND_ISOTROPIC_POOLED_SE = {
    "b_transport_seagull": 2.1194362999537304e-06,
    "b_transport_spectrum": 352.33629310146193,
    "pi_sigma_seagull": 0.00037382845557904496,
    "pi_sigma_spectrum": 1196.9897278966466,
}

#: The per-variant cascade-depth convention at the sealed values.
DEPTH_CONVENTION = (
    "the cascade depth of the variant's decay machinery per source "
    "decay (0 for a stable-identity source and for a catalog-handoff "
    "root, 1 for a v1 decay step, the number of cascade vertices for "
    "a v2/v3 line); the 95th percentile is the smallest histogram "
    "depth whose cumulative count reaches 0.95 x n_decays (no "
    "interpolation); the guard activations are zero by construction "
    "(a guard raises and stops the run)"
)

#: The per-variant cascade depth statistics at the sealed values.
DEPTH_STATISTICS = {
    "aligned_sequential": {
        "convention": DEPTH_CONVENTION,
        "guard_activations": 0, "max": 5, "mean": 2.5713625,
        "n_decays": 1_600_000, "p95": 4,
    },
    "isotropic_direct": {
        "convention": DEPTH_CONVENTION,
        "guard_activations": 0, "max": 1, "mean": 0.965064375,
        "n_decays": 1_600_000, "p95": 1,
    },
    "isotropic_sequential": {
        "convention": DEPTH_CONVENTION,
        "guard_activations": 0, "max": 5, "mean": 2.570289375,
        "n_decays": 1_600_000, "p95": 4,
    },
}

#: The pooled spectral retention per variant at the sealed values.
RETENTION_POOLED = {
    "aligned_sequential": {
        "b_transport": 0.23143030755370328,
        "pi_sigma": 0.06112799530840281,
    },
    "isotropic_direct": {
        "b_transport": 0.20244142817601157,
        "pi_sigma": 0.09048653135766668,
    },
    "isotropic_sequential": {
        "b_transport": 0.19640881328695864,
        "pi_sigma": 0.060729897276197785,
    },
}

#: The per-variant mean multiplicities at the sealed values.
MULTIPLICITY_MEANS = {
    "aligned_sequential": {
        "antiproton": 0.0,
        "k_minus": 0.14749125,
        "k_plus": 0.36949875,
        "pi_minus": 2.076765,
        "pi_plus": 2.632815,
        "proton": 1.07061125,
    },
    "isotropic_direct": {
        "antiproton": 0.0,
        "k_minus": 0.0,
        "k_plus": 0.2820475,
        "pi_minus": 0.45416,
        "pi_plus": 1.04177625,
        "proton": 0.98986375,
    },
    "isotropic_sequential": {
        "antiproton": 0.0,
        "k_minus": 0.1472425,
        "k_plus": 0.36936375,
        "pi_minus": 2.07570625,
        "pi_plus": 2.63326,
        "proton": 1.06861125,
    },
}

#: The conservation summary at the sealed values (zero violations
#: over 800,000 events x 3 variants, zero guard activations).
CONSERVATION_WORST_ENERGY_CLOSURE_GEV = 2.5934809855243657e-13

#: The superseded card-14 run's two-entry named-residual register
#: (the two failing beyond-isotropic observables of the 2026-09-22
#: archive).  Retained as the supersession reference; the
#: relitigation seals a one-entry register (the b_transport_seagull
#: non-positive finding only).
SUPERSEDED_NAMED_RESIDUAL_REGISTER = (
    {
        "observable": "b_transport_spectrum",
        "kind": "beyond_isotropic_within_band",
        "observed": 1135.421858498681,
    },
    {
        "observable": "b_transport_seagull",
        "kind": "beyond_isotropic_not_positive",
        "observed": -1.7985961442749292e-05,
    },
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "staged_r43_cascade_ablation_under_test", NESTED_DRIVER
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
def ablation_report():
    skip_unless_sealed(RUN_DIR)
    return json.loads(
        (RUN_DIR / "ablation_report.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def gate_record():
    skip_unless_sealed(RUN_DIR)
    return json.loads(
        (RUN_DIR / "gate4_r43.json").read_text(encoding="utf-8"))


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
    collision configuration, the three-variant register and policies,
    the nominal point, the support boundary, the beyond-isotropic
    factor, the seed map, and the obligation 4 carry."""
    assert driver.RUN_ID == "R43"
    assert driver.CARD == SOLVER_CARD
    assert driver.RUN_NAME == "r43_cascade_ablation_2026-09-23"
    assert driver.DEFAULT_RUN_DIR == (
        "dev/00runs/r43_cascade_ablation_2026-09-23")
    assert driver.GATE_RECORD_NAME == "gate4_r43.json"
    assert driver.REPORT_NAME == "ablation_report.json"
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
        "helpers/benchmarking/staged_r43_cascade_ablation.py")
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
    assert driver.R42_DRIVER_REL == (
        "helpers/benchmarking/staged_r42_recoil_decay_separation.py")
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
        "TASK-20260920-r4-run-r43-cascade-ablation",
    )
    # The R32 p+p collision configuration (index item 7)
    assert driver.BEAM_MOMENTUM_GEV_C == BEAM_MOMENTUM_GEV_C
    assert driver.SQRT_S_GEV == SQRT_S_GEV
    assert driver.BLEND_RAMP_START_GEV == BLEND_RAMP_START_GEV
    assert driver.BLEND_RAMP_WIDTH_GEV == BLEND_RAMP_WIDTH_GEV
    assert driver.M8_PARAMETER_HASH == M8_PARAMETER_HASH
    # The frozen exchange-layer slope (the M8-frozen value)
    assert driver.B_EFF_FROZEN_GEV2 == 8.504140762083612
    # The three-variant register and the card-10 policies: v1/v2 at the
    # isotropic point, v3 at the pre-registered nominal point (not the
    # R41 fitted values)
    assert driver.ABLATION_VARIANTS == (
        "isotropic_direct", "isotropic_sequential", "aligned_sequential")
    for name, (kb, km) in {
        "isotropic_direct": (0.0, 0.0),
        "isotropic_sequential": (0.0, 0.0),
        "aligned_sequential": (1.0, 1.0),
    }.items():
        policy = driver.ABLATION_POLICIES[name]
        assert policy.variant == name
        assert policy.kappa_b == kb and policy.kappa_m == km
        assert policy.b_dec == NOMINAL_B_DEC_GEV2
    # The R41 fitted tuple (0.0, 4.0) is not consumed
    assert (driver.ABLATION_POLICIES["aligned_sequential"].kappa_b,
            driver.ABLATION_POLICIES["aligned_sequential"].kappa_m) != (
                0.0, 4.0)
    # The support boundary (card 10's block, byte-identical)
    assert driver.SUPPORT_BOUNDARY["v2_v3_support_identical"] is True
    assert driver.SUPPORT_BOUNDARY["regeneration_required"] == (
        ("isotropic_direct",))
    # The pre-registered beyond-isotropic threshold factor
    assert driver.BEYOND_ISOTROPIC_FACTOR == BEYOND_ISOTROPIC_FACTOR
    # The seed map (index item 9)
    assert driver.MASTER_SEED == MASTER_SEED
    assert driver.SEED_BASE == MASTER_SEED
    assert driver.SEED_OFFSET == 40_000
    assert driver.SEED_STRIDE == 100
    assert driver.T2_REPLICAS == 8
    assert driver.T2_EVENTS_PER_REPLICA == 100_000
    assert driver.RAW_REPLICA_SEEDS == RAW_REPLICA_SEEDS
    assert driver.EVENT_INDEX_RULE == EVENT_INDEX_RULE
    assert driver.EVENT_INDEX_BASE == EVENT_INDEX_BASE
    assert driver.SMOKE_RAW_SEED == SMOKE_RAW_SEED
    assert driver.SMOKE_KERNEL_REPLICA_SLOT == SMOKE_KERNEL_REPLICA_SLOT
    assert driver.SMOKE_EVENT_INDEX_BASE == 690_000_000
    # The training observables and the combination pairs
    assert driver.TRAINING_OBSERVABLES == (
        "pi_sigma_spectrum", "b_transport_spectrum", "pi_sigma_seagull",
        "b_transport_seagull")
    # The obligation 4 model-uncertainty carry
    assert driver.OBLIGATION4_R32_SPREAD == OBLIGATION4_R32_SPREAD
    # The robust-statistics and closure conventions
    assert driver.SE_FLOOR == 1e-12
    assert driver.DETERMINISTIC_SE_FLOOR == 1e-12
    assert driver.EVENT_CLOSURE_TOL == 1e-9
    assert driver.MAX_CASCADE_DEPTH == 12
    # Generation chunking and the stage label
    assert driver.CHUNK_SIZE == 2500
    assert driver.DEFAULT_WORKERS == 8
    assert driver.STAGE_LABEL == "r43_cascade_ablation"
    # The clean-environment validation's pre-registered probe replica
    assert driver.PROBE_REPLICA == 0
    # The species conventions
    assert driver.SPECIES == (
        ("pi_plus", 211), ("pi_minus", -211), ("k_plus", 321),
        ("k_minus", -321), ("proton", 2212), ("antiproton", -2212))
    assert driver.SPECIES_KEYS == (
        "pi_plus", "pi_minus", "k_plus", "k_minus", "proton",
        "antiproton")
    # The sealed archives consumed read-only
    assert driver.SEALED_RUN_DIRS == (
        "dev/00runs/r30_spectral_normalization_2026-09-17",
        "dev/00runs/r31_catalog_continuum_matching_scan_2026-09-17",
        "dev/00runs/r32_spectral_systematic_pp_2026-09-18",
        "dev/00runs/r32_spectral_systematic_pp_confirmation_2026-09-19",
    )
    assert driver.R01_RUN_DIR == (
        "dev/00runs/r01_legacy_pp_reference_2026-09-11")
    # The superseded driver digest and run dir, recorded by the
    # corrected driver as the supersession reference
    assert driver.SUPERSEDED_DRIVER_SHA256 == SUPERSEDED_DRIVER_SHA256
    assert driver.SUPERSEDED_RUN_DIR == SUPERSEDED_RUN_DIR_NAME
    # The frozen substrate table (the relitigation pre-registered
    # against the committed corrected drivers, so the r41_driver and
    # r42_driver entries are the corrected card-16/card-17 digests)
    assert driver.FROZEN_SUBSTRATE == {
        key: (rel, digest) for key, (rel, digest) in SUBSTRATE.items()
    }


def test_seed_map_partition(driver):
    """The master seed, the raw block 20301020-20301027, the
    event-index partition 600_000_000 + i, the variant-stream rule, the
    hermetic smoke block (+9000), and the seed functions' refusal
    discipline."""
    assert driver.MASTER_SEED == MASTER_SEED
    assert driver.RAW_REPLICA_SEEDS == RAW_REPLICA_SEEDS
    assert driver.RAW_SEED_RULE.startswith(
        "raw replica seed = 20260920 + 40000 + 100 + r = 20301020 + r")
    assert "20301020-20301027" in driver.RAW_SEED_RULE
    assert driver.EVENT_INDEX_RULE == EVENT_INDEX_RULE
    assert driver.EVENT_INDEX_BASE == EVENT_INDEX_BASE
    assert "600_000_000 + i" in driver.VARIANT_STREAM_RULE
    assert "20301040 + k" in driver.RAW_SEED_RULE
    for r in range(8):
        assert driver.raw_replica_seed(r) == 20301020 + r
    for r in (-1, 8):
        with pytest.raises(SystemExit):
            driver.raw_replica_seed(r)
    # The hermetic smoke block (+9000; never sealed)
    assert driver.SMOKE_REPLICAS == 1
    assert driver.SMOKE_EVENTS == 2_000
    assert driver.SMOKE_SEED_STRIDE == SMOKE_KERNEL_REPLICA_SLOT
    assert driver.SMOKE_RAW_SEED == SMOKE_RAW_SEED
    assert driver.smoke_raw_seed() == SMOKE_RAW_SEED
    assert driver.SMOKE_EVENT_INDEX_RULE == "690_000_000 + i, i < 2,000"


def test_cli_surface(driver):
    """The exact argparse command surface on disk: exactly the seven
    mode flags, the three auxiliary flags, and the on-disk exit-code
    behavior (no mode refuses with 2, two modes refuse with 2)."""
    parser = driver._build_parser()
    modes = {
        "--pre-registration", "--smoke", "--generate", "--evaluate",
        "--gate", "--seal", "--clean-validate",
    }
    auxiliary = {"--replica", "--run-dir", "--workers"}
    seen = set()
    for action in parser._actions:
        seen.update(action.option_strings)
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
        driver.main([])
    assert exc.value.code == 2, (
        "no mode must refuse with exit code 2")
    with pytest.raises(SystemExit) as exc:
        driver.main(["--evaluate", "--gate"])
    assert exc.value.code == 2, (
        "two modes must refuse with exit code 2")


# ----------------------------------------------------------------------
# The sealed pre-registration
# ----------------------------------------------------------------------
def test_pre_registration_on_disk(driver, pre_registration):
    """The pre-registration's identity, T2 statistics, seed map,
    collision configuration, the three-variant register and policies,
    the nominal point, the support boundary, the beyond-isotropic
    metric (the 2.0 threshold), the conservation block, the obligation
    4 carry, the frozen substrate, the sealed run dirs, and the
    no-fitting clause."""
    assert pre_registration["schema"] == (
        "na49_mixing.staged_r43_cascade_ablation.pre_registration/1")
    assert pre_registration["run"] == "R43"
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
    assert "600_000_000 + i" in seeds["exchange_seed_rule"]
    assert "600_000_000 + i" in seeds["variant_stream_rule"]
    assert "20301020-20301027" in seeds["raw_seed_rule"]
    assert seeds["smoke"]["raw_seed"] == SMOKE_RAW_SEED
    assert seeds["smoke"]["kernel_replica_slot"] == (
        SMOKE_KERNEL_REPLICA_SLOT)
    assert seeds["smoke"]["event_index_base"] == 690_000_000
    # The collision configuration (round 4 index item 7)
    collision = pre_registration["collision"]
    assert collision["index_item"] == 7
    assert collision["w_flavor"] == 1.0
    assert collision["system"].startswith(
        "p+p at sqrt(s) = 17.270 GeV (beam 158 GeV/c)")
    assert "358989d8" in collision["configuration"]
    assert collision["frozen_b_eff_gev2"] == 8.504140762083612
    assert pre_registration["conservation"]["per_event"].startswith(
        "every p+p event conserves total (B, Q, S) = (2, 2, 0)")
    # The three-variant register and the card-10 policies
    variants = pre_registration["variants"]
    assert variants["index_item"] == 8
    assert variants["register"] == [
        "isotropic_direct", "isotropic_sequential", "aligned_sequential"]
    for name, (kb, km, vid) in {
        "isotropic_direct": (0.0, 0.0, 0),
        "isotropic_sequential": (0.0, 0.0, 1),
        "aligned_sequential": (1.0, 1.0, 2),
    }.items():
        pol = variants["policies"][name]
        assert pol["kappa_b"] == kb and pol["kappa_m"] == km
        assert pol["b_dec_gev2"] == NOMINAL_B_DEC_GEV2
        assert pol["variant_id"] == vid
    # The pre-registered nominal point (index item 3, not a fit
    # parameter)
    nominal = pre_registration["nominal_point"]
    assert nominal["kappa_b"] == NOMINAL_KAPPA_B
    assert nominal["kappa_m"] == NOMINAL_KAPPA_M
    assert nominal["b_dec_gev2"] == NOMINAL_B_DEC_GEV2
    # The support boundary (the documented v1/v2/v3 boundary)
    boundary = pre_registration["support_boundary"]
    assert boundary["v2_v3_support_identical"] is True
    assert boundary["regeneration_required"] == ["isotropic_direct"]
    assert boundary["variants"]["isotropic_direct"][
        "common_random_numbers"] == "regenerate"
    assert boundary["variants"]["isotropic_direct"]["decay_steps"] == (
        "single_step")
    assert boundary["variants"]["isotropic_sequential"][
        "common_random_numbers"] == "shared"
    assert boundary["variants"]["aligned_sequential"][
        "common_random_numbers"] == "shared"
    # The pre-registered beyond-isotropic metric (the 2.0 threshold)
    metric = pre_registration["beyond_isotropic_metric"]
    assert metric["factor"] == BEYOND_ISOTROPIC_FACTOR
    assert metric["bound"] == "pre-registered, never relaxed"
    # The no-fitting clause (plan section 24)
    assert pre_registration["no_fitting"] is True
    assert pre_registration["optimizer_invoked"] is False
    # The obligation 4 carry with the audit outcomes
    carry = pre_registration["obligation_4_carry"]
    assert carry["obligation"] == 4
    assert carry["r32_spread"] == OBLIGATION4_R32_SPREAD
    assert carry["audit_outcome"]["card_1"].startswith("RECOMMENDATION")
    assert carry["audit_outcome"]["card_2"].startswith("PASS")
    assert carry["audit_outcome"]["card_3"].startswith("PASS")
    # The frozen substrate digests, verified on disk (the relitigation
    # pre-registered against the committed corrected drivers, so the
    # r41_driver and r42_driver entries are the corrected
    # card-16/card-17 digests)
    frozen = {key: digest for key, (_rel, digest) in SUBSTRATE.items()}
    assert pre_registration["frozen_substrate"] == frozen
    for key, status in pre_registration["substrate"].items():
        assert status["verified"] is True
        assert status["sha256"] == frozen[key]
    # The sealed run dirs, verified
    assert pre_registration["sealed_run_dirs"] == list(
        driver.SEALED_RUN_DIRS)
    status = pre_registration["sealed_run_dirs_status"]
    assert status[
        "dev/00runs/r30_spectral_normalization_2026-09-17"] == (
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
    # The recording convention (the relitigation clause): dense
    # accumulation with the pick map, holes dropped, the
    # variant-invariant pick, and the superseded/corrected driver
    # digests recorded
    rec = pre_registration["recording_convention"]
    assert rec["convention"].startswith(
        "the R01 accumulate-dense-then-pick recording convention")
    assert rec["holes_dropped"] is True
    assert rec["pick_map"] == "bin_index_pt * n_x + bin_index_xf"
    assert rec["variant_invariant"] is True
    assert rec["superseded_driver_sha256"] == SUPERSEDED_DRIVER_SHA256
    assert rec["corrected_driver_sha256"] == DRIVER_SHA256
    assert rec["superseded_run_dir"] == SUPERSEDED_RUN_DIR_NAME
    geo = rec["geometry"]
    assert geo["lattice_cells_total"] == 1296
    assert geo["holes_total"] == 279
    assert geo["published_rows_total"] == 1017
    assert sorted(geo["per_species"]) == [
        "antiproton", "k_minus", "k_plus", "pi_minus", "pi_plus",
        "proton"]
    # The pre-registration's on-disk bytes at the pinned digest
    assert _sha256_file(RUN_DIR / "pre_registration.json") == (
        SEALED_FILE_DIGESTS["pre_registration.json"])


# ----------------------------------------------------------------------
# The sealed ablation report
# ----------------------------------------------------------------------
def test_ablation_report_on_disk(driver, ablation_report):
    """The report's schema, the internal digest, the per-variant
    deviance table, the beyond-isotropic verdict (all_pass false at
    the pre-registered 2.0 x pooled SE threshold; three of the four
    observables pass), the depth statistics, the retention, the
    per-variant multiplicity means, the exchange bit-identity, the
    one-entry named-residual register, the stream pin, and the
    no-fitting clause."""
    assert ablation_report["schema"] == (
        "na49_mixing.staged_r43_cascade_ablation.report/1")
    assert ablation_report["run"] == "R43"
    assert ablation_report["card"] == SOLVER_CARD
    assert ablation_report["mode"] == "production"
    assert ablation_report["digest"] == ABLATION_REPORT_DIGEST
    assert ablation_report["all_recorded_observables_finite"] is True
    assert ablation_report["events_checked"] == 800_000
    assert ablation_report["replicas"] == 8
    # The per-variant deviance table
    assert ablation_report["deviance_table"] == DEVIANCE_TABLE
    # The beyond-isotropic verdict (all_pass false at the 2.0 x pooled
    # SE threshold; three of the four observables pass -- only the
    # b_transport seagull fails, non-positive)
    beyond = ablation_report["beyond_isotropic"]
    assert beyond["all_pass"] is False
    assert beyond["threshold_factor"] == BEYOND_ISOTROPIC_FACTOR
    rows = beyond["rows"]
    assert set(rows) == set(TRAINING_OBSERVABLE_SET)
    assert rows["b_transport_seagull"]["pass"] is False
    assert rows["b_transport_seagull"]["positive"] is False
    assert rows["b_transport_seagull"]["beyond_2x_se"] is False
    assert rows["b_transport_spectrum"]["pass"] is True
    assert rows["b_transport_spectrum"]["positive"] is True
    assert rows["b_transport_spectrum"]["beyond_2x_se"] is True
    assert rows["pi_sigma_seagull"]["pass"] is True
    assert rows["pi_sigma_spectrum"]["pass"] is True
    for obs in TRAINING_OBSERVABLE_SET:
        assert rows[obs]["delta_v2_minus_v3"] == (
            BEYOND_ISOTROPIC_V3_MINUS_V2[obs])
        assert rows[obs]["pooled_se"] == BEYOND_ISOTROPIC_POOLED_SE[obs]
    # The cascade depth statistics
    assert ablation_report["depth_statistics"] == DEPTH_STATISTICS
    # The pooled spectral retention per variant
    retention = ablation_report["retention"]
    assert set(retention) == set(RETENTION_POOLED)
    for variant, groups in RETENTION_POOLED.items():
        for group, pooled in groups.items():
            assert retention[variant][group]["pooled"] == pooled
    # The per-variant mean multiplicities
    multiplicity = ablation_report["multiplicity"]
    assert set(multiplicity) == set(MULTIPLICITY_MEANS)
    for variant, species in MULTIPLICITY_MEANS.items():
        for name, mean in species.items():
            assert multiplicity[variant][name][
                "mean_multiplicity"] == mean
    # The exchange bit-identity (verified at run scale, all 8
    # replicas)
    bit = ablation_report["exchange_bit_identity"]
    assert bit["pass"] is True
    assert bit["variants"] == 3
    assert bit["events"] == 800_000
    assert bit["layers"] == ["topology", "exchange", "spectral_state"]
    assert len(bit["per_replica"]) == 8
    for entry in bit["per_replica"]:
        assert entry["variants_digests_equal"] is True
        assert entry["n_events"] == 100_000
    # The one-entry named-residual register (the failing
    # beyond-isotropic observable; the superseded run's two-entry
    # register stays as SUPERSEDED_NAMED_RESIDUAL_REGISTER)
    register = ablation_report["named_residual_register"]
    assert len(register) == 1
    assert [entry["index"] for entry in register] == [1]
    assert [entry["kind"] for entry in register] == [
        "beyond_isotropic_not_positive"]
    assert register[0]["observable"] == "b_transport_seagull"
    assert register[0]["observed"] == (
        BEYOND_ISOTROPIC_V3_MINUS_V2["b_transport_seagull"])
    for entry in register:
        assert entry["robust_statistics_cross_reference"].startswith(
            "the between-replica standard error")
    # The stream pin (the pinned-key replay of card 10's
    # variant_stream contract)
    pin = ablation_report["stream_pin"]
    assert pin["key"]["master_seed"] == MASTER_SEED
    assert pin["key"]["event_index"] == EVENT_INDEX_BASE
    assert pin["key"]["variant_ids"] == [0, 1, 2]
    assert len(pin["first_channel_uniform_per_variant"]) == 3
    # The training observables and the variants
    assert ablation_report["training_observables"] == list(
        driver.TRAINING_OBSERVABLES)
    assert ablation_report["variants"] == [
        "isotropic_direct", "isotropic_sequential", "aligned_sequential"]
    # The no-fitting clause
    assert ablation_report["no_fitting"] is True
    assert ablation_report["optimizer_invoked"] is False
    # The report's on-disk bytes at the pinned digest
    assert _sha256_file(RUN_DIR / "ablation_report.json") == (
        SEALED_FILE_DIGESTS["ablation_report.json"])


TRAINING_OBSERVABLE_SET = (
    "pi_sigma_spectrum", "b_transport_spectrum", "pi_sigma_seagull",
    "b_transport_seagull")


# ----------------------------------------------------------------------
# The Gate 4 evidence
# ----------------------------------------------------------------------
def test_gate4_evidence(driver, gate_record, ablation_report):
    """The gate record's identity, its clause (beyond isotropic
    sequential), its conservation summary (zero violations over
    800,000 events), the exchange bit-identity block, the deviance
    table, the beyond-isotropic verdict (pass false with the one named
    finding), the support boundary, the model-uncertainty carry, the
    named-residual register, and the no-fitting clause."""
    assert gate_record["schema"] == (
        "na49_mixing.staged_r43_cascade_ablation.gate/1")
    assert gate_record["run"] == "R43"
    assert gate_record["card"] == SOLVER_CARD
    assert gate_record["gate"] == "gate 4 (plan section 10, line 806)"
    assert gate_record["clause"] == "beyond isotropic sequential"
    assert gate_record["sealed"] is True
    # The conservation summary (zero violations over 800,000 events)
    assert gate_record["events_checked"] == 800_000
    conservation = gate_record["conservation"]
    assert conservation["violations"] == 0
    assert conservation["guard_activations"] == 0
    assert conservation["pass"] is True
    assert conservation["events_checked"] == 800_000
    assert conservation["tolerance"] == 1e-09
    assert conservation["worst_energy_closure_gev"] == (
        CONSERVATION_WORST_ENERGY_CLOSURE_GEV)
    # The exchange bit-identity block (verified at run scale)
    bit = gate_record["exchange_bit_identity"]
    assert bit["pass"] is True
    assert bit["variants"] == 3
    assert bit["events"] == 800_000
    assert bit["layers"] == ["topology", "exchange", "spectral_state"]
    # The support boundary
    boundary = gate_record["support_boundary"]
    assert boundary["regeneration_required"] is False
    assert boundary["v1"] == "single-step"
    assert boundary["v2_v3"] == "multi-vertex (identical)"
    # The per-variant deviance table
    assert gate_record["deviance_table"] == DEVIANCE_TABLE
    # The cascade depth statistics (equal to the report's)
    assert gate_record["depth_statistics"] == DEPTH_STATISTICS
    # The pooled spectral retention per variant (equal to the
    # report's)
    gate_retention = gate_record["retention"]
    for variant, groups in RETENTION_POOLED.items():
        for group, pooled in groups.items():
            assert gate_retention[variant][group]["pooled"] == pooled
    # The per-variant mean multiplicities (equal to the report's)
    gate_multiplicity = gate_record["multiplicity"]
    for variant, species in MULTIPLICITY_MEANS.items():
        for name, mean in species.items():
            assert gate_multiplicity[variant][name][
                "mean_multiplicity"] == mean
    # The beyond-isotropic verdict (pass false; the one named finding)
    beyond = gate_record["beyond_isotropic"]
    assert beyond["pass"] is False
    assert beyond["threshold"] == "2.0 x pooled SE"
    assert beyond["v3_minus_v2_deviance"] == BEYOND_ISOTROPIC_V3_MINUS_V2
    assert beyond["pooled_se"] == BEYOND_ISOTROPIC_POOLED_SE
    # The model-uncertainty carry (obligation 4)
    carry = gate_record["model_uncertainty_carry"]
    assert carry["obligation_4"] is True
    assert carry["r32_spread"] == OBLIGATION4_R32_SPREAD
    # The named-residual register (the failing beyond-isotropic
    # observable, equal to the report's)
    register = gate_record["named_residual_register"]
    assert register == ablation_report["named_residual_register"]
    assert len(register) == 1
    assert [entry["kind"] for entry in register] == [
        "beyond_isotropic_not_positive"]
    # The report digest cross-reference (the gate's digest equals the
    # ablation report's internal digest)
    assert gate_record["report_digest"] == ABLATION_REPORT_DIGEST
    assert gate_record["report_digest"] == ablation_report["digest"]
    # The nominal point
    nominal = gate_record["nominal_point"]
    assert nominal["kappa_B"] == NOMINAL_KAPPA_B
    assert nominal["kappa_m"] == NOMINAL_KAPPA_M
    assert nominal["b_dec"] == NOMINAL_B_DEC_GEV2
    # The no-fitting clause
    assert gate_record["no_fitting"] is True
    assert gate_record["optimizer_invoked"] is False
    # The gate record's on-disk bytes at the pinned digest
    assert _sha256_file(RUN_DIR / "gate4_r43.json") == (
        SEALED_FILE_DIGESTS["gate4_r43.json"])


# ----------------------------------------------------------------------
# The seal manifest and the checksum file
# ----------------------------------------------------------------------
def test_seal_manifest_and_checksums(seal_manifest):
    """The seal manifest's 13 sealed files (8 replica records plus the
    five top-level files) path-sorted, the checksum file's 13 entries,
    the 15-file run dir (13 payload plus the two meta files), and the
    full re-hash of all 13 sealed files against the checksum
    digests."""
    assert seal_manifest["schema"] == (
        "na49_mixing.staged_r43_cascade_ablation.seal/1")
    assert seal_manifest["run"] == "R43"
    assert seal_manifest["card"] == SOLVER_CARD
    assert seal_manifest["record_count"] == 8
    assert seal_manifest["sha256sums_entries"] == 13
    assert seal_manifest["files"] == list(sorted(SEALED_FILES))
    assert len(seal_manifest["files"]) == 13
    assert seal_manifest["sealed_utc"].startswith("2026-09-24")
    assert seal_manifest["run_dir"].endswith(
        "dev/00runs/r43_cascade_ablation_2026-09-23")
    # The run dir is 15 files: the 13 payload plus the two meta files
    # (seal_manifest.json and SHA256SUMS.txt)
    dir_entries = sorted(p.name for p in RUN_DIR.iterdir()
                         if p.is_file())
    assert len(dir_entries) == 15
    assert "seal_manifest.json" in dir_entries
    assert "SHA256SUMS.txt" in dir_entries
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
def test_clean_environment_record(clean_environment, ablation_report):
    """The clean-environment validation's command list, the
    byte-identical report re-hash (digest equality), the probe replica
    record, and the pending seal state."""
    assert clean_environment["schema"] == (
        "na49_mixing.staged_r43_cascade_ablation.clean_env/1")
    assert clean_environment["run"] == "R43"
    assert clean_environment["card"] == SOLVER_CARD
    commands = clean_environment["commands"]
    assert len(commands) == 3
    assert commands[0] == (
        "cd dev/mixing2006 && python -m pytest "
        "tests/test_staged_r43_cascade_ablation.py -q")
    assert commands[1] == (
        "python -m pytest tests/test_legacy_v1_freeze.py "
        "-W error::UserWarning -q")
    assert commands[2].startswith(
        "python helpers/benchmarking/staged_r43_cascade_ablation.py "
        "--clean-validate --run-dir ")
    assert "r43_cascade_ablation_2026-09-23" in commands[2]
    # The byte-identical report re-hash
    assert clean_environment["report_hash_match"] is True
    assert clean_environment["report_digest"] == ABLATION_REPORT_DIGEST
    assert clean_environment["report_rerun_digest"] == (
        ABLATION_REPORT_DIGEST)
    assert clean_environment["report_digest"] == (
        ablation_report["digest"])
    # The probe replica (setting-independent full replica re-run)
    probe = clean_environment["probe_replica"]
    assert probe["replica"] == 0
    assert probe["matches_sealed_record"] is True
    assert probe["per_variant_matches"] == {
        "isotropic_direct": True, "isotropic_sequential": True,
        "aligned_sequential": True}
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
    set hash, the recorded driver digest (equal to the committed
    corrected dual-tree digest: the relitigation pre-registered
    against the committed corrected driver), the superseded archive's
    verified seal state as the supersession reference, the nominal
    point, and the sealed run dir verification."""
    assert provenance["schema"] == (
        "na49_mixing.staged_r43_cascade_ablation.provenance/1")
    assert provenance["run"] == "R43"
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
    # The R41 driver digest recorded at run time (the corrected
    # card-16 driver; the relitigation pre-registered against the
    # committed corrected driver)
    assert provenance["r41_driver_sha256"] == (
        SUBSTRATE["r41_driver"][1])
    # The R42 driver digest recorded at run time (the corrected
    # card-17 driver; the relitigation pre-registered against the
    # committed corrected driver)
    assert provenance["r42_driver_sha256"] == (
        SUBSTRATE["r42_driver"][1])
    assert provenance["m8_frozen_set_hash"] == M8_PARAMETER_HASH
    assert provenance["frozen_b_eff_gev2"] == 8.504140762083612
    assert provenance["cascade_module_sha256"] == {
        name: SUBSTRATE[name][1]
        for name in ("cascade_decay", "cascade_axes", "cascade_alignment",
                     "cascade_form_factor", "cascade_termination",
                     "cascade_ablation")
    }
    # The recorded driver digest: the relitigation pre-registered
    # against the committed corrected driver, so the recorded digest
    # equals the committed dual-tree digest (unlike the card-14
    # landing, where the driver was amended after the
    # pre-registration)
    assert provenance["new_driver_sha256_both_trees"] == (
        PROVENANCE_DRIVER_SHA256)
    assert provenance["new_driver_sha256_both_trees"] == DRIVER_SHA256
    assert provenance["driver_note"].startswith(
        "byte-identical in the nested tree")
    assert provenance["nominal_point"].startswith(
        "v3 runs at the pre-registered nominal point")
    # The superseded archive (the card-14 seal, intact and verified as
    # the supersession reference)
    superseded = provenance["superseded_r43_archive"]
    assert superseded["run_dir"] == SUPERSEDED_RUN_DIR_NAME
    assert superseded["seal"] == SUPERSEDED_SEAL_STATE
    assert superseded["state"].startswith(
        "intact; the seal verifies (verify_sealed_dir)")
    assert superseded["superseded_driver_sha256"] == (
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
    }
    # The pre-registration carries the same pre-registration-time
    # driver digest
    pre = json.loads(
        (RUN_DIR / "pre_registration.json").read_text(encoding="utf-8"))
    assert pre["provenance"]["driver_sha256"] == DRIVER_SHA256


# ----------------------------------------------------------------------
# The frozen substrate, byte-stable on disk
# ----------------------------------------------------------------------
def test_frozen_substrate_stability(driver):
    """The driver's sixteen frozen substrate files byte-stable on
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
    in the freeze manifest inventory, with the card-14 solver freeze
    exception intact (the solver landing's 22 path-sorted files)."""
    solver_path = MONO_ROOT / SOLVER_TEST_REL
    assert solver_path.is_file(), f"missing solver test: {SOLVER_TEST_REL}"
    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)
    inventory = manifest["source_tree_sha256"]
    assert inventory[SOLVER_TEST_REL] == SOLVER_TEST_SHA256, (
        "the freeze manifest does not pin the solver test at its "
        "committed digest")
    # The card-14 solver freeze exception stays untouched: exactly one
    # entry naming the solver's card, at the solver landing's 22
    # path-sorted files
    exceptions = manifest["freeze_exceptions"]
    solver_entries = [
        e for e in exceptions if e.get("card") == SOLVER_CARD]
    assert len(solver_entries) == 1, (
        "the card-14 solver freeze exception must name its card once")
    solver_files = solver_entries[0]["files"]
    assert solver_files == sorted(solver_files), (
        "the card-14 solver freeze-exception files are path-sorted")
    assert solver_files == list(SOLVER_EXCEPTION_FILES), (
        "the card-14 solver freeze-exception files drifted")


# ----------------------------------------------------------------------
# The freeze re-pin register (the test-author re-pin)
# ----------------------------------------------------------------------
