"""Round 4, card 11 test-author register: the R40 fixed-source alignment
scan contract pins.

Card ``TASK-20260920-r4-run-r40-fixed-source-alignment-scan`` (plan
section 10 run R40, line 761).  This register is the test-author half
of the card: independent pins of the dual-tree run driver
(``helpers/benchmarking/staged_r40_alignment_scan.py``, byte-identical
in the nested tree and the monorepo mirror) and of the sealed run
``dev/00runs/r40_fixed_source_alignment_scan_2026-09-21/``.  The
driver is loaded once via importlib (its top level is the
pre-registration identity; no events are generated), and the sealed
JSON is read from disk.  The solver's pinned seed-map and pinned
traces are not re-run here; the contract is pinned on the committed
bytes and the sealed on-disk values.

- ``test_dual_tree_driver_byte_identity``: both driver copies exist,
  are byte-identical, and sit at the committed digest; the freeze
  manifest inventory pins both at that digest.
- ``test_driver_identity_constants``: the driver's run identity, the
  file-name constants, the T1 statistics, the seed map (master seed,
  raw block, event-index partition, smoke block, probe chunk), the
  source grid, the scan grid, the species register, the routing
  regimes, and the frozen-substrate table.
- ``test_cli_surface``: the exact argparse command surface on disk
  (the program name, the required mutually exclusive mode group with
  exactly the seven mode flags, the ten auxiliary flags and their
  defaults) plus the on-disk exit-code behavior (help exits 0, no
  mode refuses with 2, two modes refuse with 2).
- ``test_pre_registration_on_disk``: the pre-registration's schema,
  the source grid (ten at-rest points with route regimes and blend
  weights), the scan grid (nine settings, the three slices), the T1
  statistics, the seed map, the routing surface, the acceptance
  check-set pre-registration, the no-fitting clause, the substrate
  verification, and the sealed run dirs.
- ``test_scan_report_headline_census``: the report's conservation
  summary (zero violations, the exact worst closures), the depth
  summary, and the full 90-entry routing census table with the
  per-entry structure (species, partition, conservation, depth).
- ``test_check_set_sigma_algebra``: the 58-row check set; the
  endpoint-contrast sigma algebra reproduced bit for bit on every
  row; the verdict rule; the two named residuals at their exact
  sigmas with the 1-based row-index consistency.
- ``test_gate4_evidence``: the gate record's identity, its
  conservation and depth summaries against the report, the model-
  uncertainty carry, the seal clause, and the named-residual
  register's equality with the report's.
- ``test_seal_manifest_and_checksums``: the seal manifest's 365
  entries (360 setting records plus the five top-level files)
  path-sorted, the checksum file's 364 entries, the sums file's own
  digest, and a sampled re-hash of ten sealed files.
- ``test_clean_environment_record``: the clean-environment
  validation's seal verification (364 files, no mismatches), the
  byte-identical report re-hash, the probe-chunk checks, and the
  probe instrument's digest.
- ``test_provenance_on_disk``: the provenance's schema, the card
  chain, the recorded pre-registration driver digests (distinct
  from the committed dual-tree digest), the substrate digests, and
  the seed-map fields.
- ``test_frozen_substrate_stability``: the driver's thirteen frozen
  substrate files byte-stable on disk at their pinned digests.
- ``test_freeze_repin_register``: the freeze-manifest re-pin for
  this register (the strict path-sorted inventory at 476 entries
  carries this file at its on-disk digest; the card-11 solver freeze
  exception stays at the solver's 18 path-sorted files; this
  register's own freeze exception names the test-author card exactly
  once with the 16 path-sorted files; the fourteen consumed count
  pins are re-pinned 475 -> 476 with the naming comment attributing
  the bump to this register, the card-10 solver's 471 literal
  retained for its re-pin check; the frozen M8 parameter map stays
  intact).

Hermetic: standard library and numpy, the committed config and
catalog, and the sealed run dir (read-only) only; no SPS tables, no
event generation, no Monte Carlo run.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
from tests._freeze_inventory import inventory_count as _inventory_count
from tests._sealed_archive import skip_unless_sealed

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent          # dev/mixing2006
MONO_ROOT = NESTED_ROOT.parent.parent          # the monorepo root
NESTED_DRIVER = (
    NESTED_ROOT / "helpers" / "benchmarking" / "staged_r40_alignment_scan.py"
)
OUTER_DRIVER = (
    MONO_ROOT / "helpers" / "benchmarking" / "staged_r40_alignment_scan.py"
)
MANIFEST_PATH = (NESTED_ROOT / "config" / "v2_singlet"
                 / "legacy_v1_manifest.json")
RUN_DIR = (MONO_ROOT / "dev" / "00runs"
           / "r40_fixed_source_alignment_scan_2026-09-21")

DRIVER_REL_NESTED = (
    "dev/mixing2006/helpers/benchmarking/staged_r40_alignment_scan.py"
)
DRIVER_REL_OUTER = "helpers/benchmarking/staged_r40_alignment_scan.py"

#: The committed dual-tree driver digest (byte-identical in both
#: trees; the freeze manifest inventory pins both copies here).
DRIVER_SHA256 = (
    "7366a55e0e15032f6d4c9194adce7537ea5239075e34f921900b75be64787354"
)

#: The driver digest the provenance records at pre-registration time
#: (the value the run pre-registered against; the committed driver
#: carries the distinct digest above, which the clean-environment
#: validation used as its probe instrument).
PROVENANCE_DRIVER_SHA256 = (
    "094b51f187dc438352b6ad0f8ef7e13e91419e1de6afa40092ba1afd7249c8f6"
)

# ----------------------------------------------------------------------
# Freeze re-pin constants (the register chain's inventory count pins)
# ----------------------------------------------------------------------
MY_TEST_REL = (
    "dev/mixing2006/tests/test_staged_r40_alignment_scan_contract_pins.py"
)
MANIFEST_REL = (
    "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json"
)
SOLVER_CARD = "TASK-20260920-r4-run-r40-fixed-source-alignment-scan"
# The test-author re-pin's own freeze-exception card name (distinct
# from the solver's, so the solver's "exactly one entry naming this
# card" check is untouched).
TEST_AUTHOR_CARD = (
    "TASK-20260920-r4-run-r40-fixed-source-alignment-scan-test-author"
)

#: The fourteen register-chain files whose inventory-count pins this
#: register re-pins (475 -> 476, +1 entry): the twelve register-chain
#: files plus the card-10 solver test's own count constant
#: (test_cascade_ablation.py) and the card-10 register's own count
#: constant (test_r4_cascade_ablation_contract_pins.py).
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
)

#: The card-11 solver freeze-exception file list (18 path-sorted
#: files; this register leaves it untouched).
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
    "dev/mixing2006/tests/"
    "test_r4_cascade_alignment_baryonic_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_r4_cascade_alignment_mesonic_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_axes_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_decay_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_form_factor_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_termination_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r40_alignment_scan.py",
    DRIVER_REL_OUTER,
)

#: This register's own freeze-exception file list (path-sorted): the
#: manifest, this register, and the fourteen re-pinned count pins.
MY_EXCEPTION_FILES = tuple(
    sorted([MANIFEST_REL, MY_TEST_REL] + list(CHAIN_COUNT_PIN_FILES))
)

# 475 entries at the card-11 solver landing; this register adds its
# file (+1 entry) for the test-author re-pin (475 -> 476, +1 entry)
# and re-pins the fourteen consumed count pins (the twelve
# register-chain files plus the card-10 solver test's own count
# constant and the card-10 register's own count constant) with the
# bump attributed to this register's file.
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
# ----------------------------------------------------------------------
# Pinned sealed-run values (read from disk, verified on disk)
# ----------------------------------------------------------------------

#: The full 90-entry routing census of the scan report:
#: (setting, point) -> (n_catalog, n_continuum, fraction_continuum).
ROUTING_CENSUS = {
    (0, 0): (100000, 0, 0.0),
    (0, 1): (35988, 64012, 0.64012),
    (0, 2): (0, 100000, 1.0),
    (0, 3): (0, 100000, 1.0),
    (0, 4): (0, 100000, 1.0),
    (0, 5): (100000, 0, 0.0),
    (0, 6): (36299, 63701, 0.63701),
    (0, 7): (0, 100000, 1.0),
    (0, 8): (0, 100000, 1.0),
    (0, 9): (0, 100000, 1.0),
    (1, 0): (100000, 0, 0.0),
    (1, 1): (35804, 64196, 0.64196),
    (1, 2): (0, 100000, 1.0),
    (1, 3): (0, 100000, 1.0),
    (1, 4): (0, 100000, 1.0),
    (1, 5): (100000, 0, 0.0),
    (1, 6): (36080, 63920, 0.6392),
    (1, 7): (0, 100000, 1.0),
    (1, 8): (0, 100000, 1.0),
    (1, 9): (0, 100000, 1.0),
    (2, 0): (100000, 0, 0.0),
    (2, 1): (36150, 63850, 0.6385),
    (2, 2): (0, 100000, 1.0),
    (2, 3): (0, 100000, 1.0),
    (2, 4): (0, 100000, 1.0),
    (2, 5): (100000, 0, 0.0),
    (2, 6): (36481, 63519, 0.63519),
    (2, 7): (0, 100000, 1.0),
    (2, 8): (0, 100000, 1.0),
    (2, 9): (0, 100000, 1.0),
    (3, 0): (100000, 0, 0.0),
    (3, 1): (36032, 63968, 0.63968),
    (3, 2): (0, 100000, 1.0),
    (3, 3): (0, 100000, 1.0),
    (3, 4): (0, 100000, 1.0),
    (3, 5): (100000, 0, 0.0),
    (3, 6): (36139, 63861, 0.63861),
    (3, 7): (0, 100000, 1.0),
    (3, 8): (0, 100000, 1.0),
    (3, 9): (0, 100000, 1.0),
    (4, 0): (100000, 0, 0.0),
    (4, 1): (36086, 63914, 0.63914),
    (4, 2): (0, 100000, 1.0),
    (4, 3): (0, 100000, 1.0),
    (4, 4): (0, 100000, 1.0),
    (4, 5): (100000, 0, 0.0),
    (4, 6): (36071, 63929, 0.63929),
    (4, 7): (0, 100000, 1.0),
    (4, 8): (0, 100000, 1.0),
    (4, 9): (0, 100000, 1.0),
    (5, 0): (100000, 0, 0.0),
    (5, 1): (36164, 63836, 0.63836),
    (5, 2): (0, 100000, 1.0),
    (5, 3): (0, 100000, 1.0),
    (5, 4): (0, 100000, 1.0),
    (5, 5): (100000, 0, 0.0),
    (5, 6): (36105, 63895, 0.63895),
    (5, 7): (0, 100000, 1.0),
    (5, 8): (0, 100000, 1.0),
    (5, 9): (0, 100000, 1.0),
    (6, 0): (100000, 0, 0.0),
    (6, 1): (35967, 64033, 0.64033),
    (6, 2): (0, 100000, 1.0),
    (6, 3): (0, 100000, 1.0),
    (6, 4): (0, 100000, 1.0),
    (6, 5): (100000, 0, 0.0),
    (6, 6): (36109, 63891, 0.63891),
    (6, 7): (0, 100000, 1.0),
    (6, 8): (0, 100000, 1.0),
    (6, 9): (0, 100000, 1.0),
    (7, 0): (100000, 0, 0.0),
    (7, 1): (36175, 63825, 0.63825),
    (7, 2): (0, 100000, 1.0),
    (7, 3): (0, 100000, 1.0),
    (7, 4): (0, 100000, 1.0),
    (7, 5): (100000, 0, 0.0),
    (7, 6): (36185, 63815, 0.63815),
    (7, 7): (0, 100000, 1.0),
    (7, 8): (0, 100000, 1.0),
    (7, 9): (0, 100000, 1.0),
    (8, 0): (100000, 0, 0.0),
    (8, 1): (36054, 63946, 0.63946),
    (8, 2): (0, 100000, 1.0),
    (8, 3): (0, 100000, 1.0),
    (8, 4): (0, 100000, 1.0),
    (8, 5): (100000, 0, 0.0),
    (8, 6): (36189, 63811, 0.63811),
    (8, 7): (0, 100000, 1.0),
    (8, 8): (0, 100000, 1.0),
    (8, 9): (0, 100000, 1.0),
}

#: The exact worst-closure values of the sealed run (GeV).
WORST_ENERGY_CLOSURE_GEV = 2.1316282072803006e-14
WORST_LIGHTCONE_IDENTITY_GEV = 3.907985046680551e-14
CLOSURE_TOLERANCE = 1e-09

#: The two named residuals of the pre-registered acceptance-(v)
#: check set (exact sigmas; the 1-based row positions in the
#: 58-row check set).
NAMED_RESIDUAL_SIGMAS = {
    ("b_dec", 0, "mean_pt_x", "pi_minus"): -3.6074494378515953,
    ("b_dec", 2, "mean_pt_x", "k_plus"): -3.603361450793928,
}

#: The committed substrate digests (the driver's FROZEN_SUBSTRATE
#: table, re-verified on disk).
SUBSTRATE = {
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
}

#: The six paper-window species by exact pdg identity.
SPECIES = {
    "pi_plus": 211,
    "pi_minus": -211,
    "k_plus": 321,
    "k_minus": -321,
    "proton": 2212,
    "antiproton": -2212,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "staged_r40_alignment_scan_under_test", NESTED_DRIVER
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
def scan_report():
    skip_unless_sealed(RUN_DIR)
    return json.loads(
        (RUN_DIR / "scan_report.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def gate_record():
    skip_unless_sealed(RUN_DIR)
    return json.loads(
        (RUN_DIR / "gate4_r40.json").read_text(encoding="utf-8"))


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
# Driver identity, byte identity, and CLI surface
# ----------------------------------------------------------------------
def test_dual_tree_driver_byte_identity(manifest):
    """The dual-tree driver is byte-identical in both trees at the
    committed digest, and the freeze manifest pins both copies."""
    assert NESTED_DRIVER.is_file(), f"missing nested driver: {NESTED_DRIVER}"
    assert OUTER_DRIVER.is_file(), f"missing outer driver: {OUTER_DRIVER}"
    nested_digest = _sha256_file(NESTED_DRIVER)
    outer_digest = _sha256_file(OUTER_DRIVER)
    assert nested_digest == DRIVER_SHA256, (
        f"nested driver drifted: {nested_digest}")
    assert outer_digest == DRIVER_SHA256, (
        f"outer driver drifted: {outer_digest}")
    assert nested_digest == outer_digest, (
        "the dual-tree driver is not byte-identical")
    inventory = manifest["source_tree_sha256"]
    assert inventory[DRIVER_REL_NESTED] == DRIVER_SHA256
    assert inventory[DRIVER_REL_OUTER] == DRIVER_SHA256


def test_driver_identity_constants(driver):
    """The driver's run identity, file names, T1 statistics, seed
    map, source grid, scan grid, species register, and routing
    regimes."""
    assert driver.RUN_ID == "R40"
    assert driver.CARD == (
        "TASK-20260920-r4-run-r40-fixed-source-alignment-scan")
    assert driver.RUN_NAME == (
        "r40_fixed_source_alignment_scan_2026-09-21")
    assert driver.DEFAULT_RUN_DIR == (
        "dev/00runs/r40_fixed_source_alignment_scan_2026-09-21")
    assert driver.GATE_RECORD_NAME == "gate4_r40.json"
    assert driver.SCAN_REPORT_NAME == "scan_report.json"
    assert driver.PREREGISTRATION_NAME == "pre_registration.json"
    assert driver.PROVENANCE_NAME == "provenance.json"
    assert driver.CLEAN_ENV_NAME == "clean_environment_commands.json"
    assert driver.SHA256SUMS_NAME == "SHA256SUMS.txt"
    assert driver.SEAL_NAME == "seal_manifest.json"
    assert driver.RECORD_FMT == "setting_{}_point_{}_replica_{}.json"
    # T1 statistics (plan section 5.1, line 253)
    assert driver.N_SETTINGS == 9
    assert driver.N_POINTS == 10
    assert driver.T1_REPLICAS == 4
    assert driver.T1_DECAYS_PER_REPLICA == 25_000
    assert driver.TOTAL_DECAYS == 9_000_000
    # The seed map
    assert driver.MASTER_SEED == 20260920
    assert driver.SEED_BASE == 20260920
    assert driver.SEED_OFFSET == 40_000
    assert driver.SEED_STRIDE == 100
    assert driver.EVENT_INDEX_BASE == 300_000_000
    assert driver.EVENT_INDEX_SETTING_STRIDE == 10_000_000
    assert driver.EVENT_INDEX_POINT_STRIDE == 1_000_000
    assert driver.event_index(0, 0, 0) == 300_000_000
    assert driver.event_index(8, 9, 24_999) == (
        300_000_000 + 8 * 10_000_000 + 9 * 1_000_000 + 24_999)
    assert driver.raw_replica_seed(0) == 20301020
    assert driver.raw_replica_seed(3) == 20301023
    # The smoke block (+7000) and the probe chunk
    assert driver.SMOKE_SETTING == 1
    assert driver.SMOKE_POINT == 0
    assert driver.SMOKE_REPLICAS == 1
    assert driver.SMOKE_EVENTS == 2_000
    assert driver.SMOKE_SEED_STRIDE == 7000
    assert driver.SMOKE_RAW_SEED == 20307920
    assert driver.SMOKE_KERNEL_REPLICA_SLOT == 7000
    assert driver.PROBE_SETTING == 1
    assert driver.PROBE_POINT == 0
    assert driver.PROBE_REPLICA == 0
    # The source grid (round 4 index item 2)
    assert [p["point"] for p in driver.POINTS] == list(range(10))
    for p in driver.POINTS:
        if p["point"] < 5:
            assert p["family"] == "baryonic"
            assert p["qn"] == (1, 1, 0)
        else:
            assert p["family"] == "mesonic"
            assert p["qn"] == (0, 0, 0)
    assert [p["mass_gev"] for p in driver.POINTS] == [
        2.0, 3.0, 4.0, 6.0, 8.0, 2.0, 3.0, 4.0, 6.0, 8.0]
    # The scan grid (round 4 index item 3)
    assert driver.NOMINAL_POINT == (1.0, 1.0, 1.0)
    assert [s["setting"] for s in driver.SETTINGS] == list(range(9))
    assert driver.KAPPA_B_SLICE == (0, 1, 2, 3)
    assert driver.KAPPA_M_SLICE == (4, 1, 5, 6)
    assert driver.B_DEC_SLICE == (7, 1, 8)
    assert (driver.SETTINGS[1]["kappa_b"], driver.SETTINGS[1]["kappa_m"],
            driver.SETTINGS[1]["b_dec"]) == (1.0, 1.0, 1.0)
    # The species register and the acceptance constants
    assert dict(driver.SPECIES) == SPECIES
    assert driver.SIGMA_BAND == 3.0
    assert driver.SE_FLOOR == 1e-12
    assert driver.PT_CHECK_MIN_MEAN_MULTIPLICITY == 0.01
    assert driver.CHUNK_WORKERS_DEFAULT == 4
    # The routing regimes (the card-6 blend at the frozen source mass)
    assert driver.BLEND_WEIGHT_AT_3 == 0.6396117785449444
    assert set(driver.ROUTE_REGIMES) == {2.0, 3.0, 4.0, 6.0, 8.0}
    assert "catalog" in driver.ROUTE_REGIMES[2.0]
    assert "blend band" in driver.ROUTE_REGIMES[3.0]
    assert driver.ROUTE_REGIMES[4.0].startswith("continuum")
    # The frozen substrate table
    assert driver.FROZEN_SUBSTRATE == SUBSTRATE
    assert driver.SEALED_RUN_DIRS == (
        "dev/00runs/r30_spectral_normalization_2026-09-17",
        "dev/00runs/r31_catalog_continuum_matching_scan_2026-09-17",
        "dev/00runs/r32_spectral_systematic_pp_2026-09-18",
        "dev/00runs/r32_spectral_systematic_pp_confirmation_2026-09-19",
    )
    assert driver.CARD_CHAIN == (
        "TASK-20260920-r4-audit-transition-routing-formulation",
        "TASK-20260920-r4-audit-strange-sector-continuum-families",
        "TASK-20260920-r4-audit-builder-blend-path-routing-weight",
        "TASK-20260920-r4-sequential-color-singlet-resonance-decay",
        "TASK-20260920-r4-inherited-production-axes",
        "TASK-20260920-r4-baryonic-alignment-kernel",
        "TASK-20260920-r4-mesonic-alignment-kernel",
        "TASK-20260920-r4-hadronic-decay-form-factor",
        "TASK-20260920-r4-termination-depth-tests",
        "TASK-20260920-r4-isotropic-aligned-ablation-runner",
    )


def test_cli_surface(driver):
    """The exact argparse command surface on disk: the program name,
    the required mutually exclusive mode group with exactly the seven
    mode flags, the ten auxiliary flags and their defaults, and the
    on-disk exit-code behavior."""
    parser = driver._build_parser()
    assert parser.prog == "staged_r40_alignment_scan.py"
    modes = {
        "--pre-registration", "--smoke", "--generate", "--evaluate",
        "--gate", "--seal", "--clean-validate",
    }
    auxiliary = {
        "--setting", "--point", "--replica", "--settings", "--points",
        "--replicas", "--decays-per-replica", "--master-seed",
        "--workers", "--run-dir",
    }
    seen = set()
    for action in parser._actions:
        seen.update(action.option_strings)
    # Exactly the mode flags, the auxiliary flags, and argparse's own
    # help flag (the surface on disk, no inferred extras).
    assert seen == modes | auxiliary | {"-h", "--help"}, (
        f"the CLI surface drifted: extra or missing flags "
        f"{sorted(seen - (modes | auxiliary | {'-h', '--help'}))}")
    # The mode group is required and mutually exclusive
    mode_actions = [
        a for a in parser._actions if a.option_strings and
        set(a.option_strings) & modes
    ]
    assert len(mode_actions) == 7
    for action in mode_actions:
        assert action.const is True
        assert action.default is False
    # The auxiliary defaults on disk
    defaults = {}
    for action in parser._actions:
        if action.option_strings:
            defaults[action.option_strings[0]] = action.default
    assert defaults["--workers"] == 4
    assert defaults["--run-dir"] == (
        "dev/00runs/r40_fixed_source_alignment_scan_2026-09-21")
    for flag in ("--setting", "--point", "--replica", "--settings",
                 "--points", "--replicas", "--decays-per-replica",
                 "--master-seed"):
        assert defaults[flag] is None, flag
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
# The sealed pre-registration
# ----------------------------------------------------------------------
def test_pre_registration_on_disk(driver, pre_registration):
    """The pre-registration's identity, source grid, scan grid,
    statistics, seed map, routing surface, acceptance pre-
    registration, no-fitting clause, and substrate verification."""
    assert pre_registration["schema"] == (
        "na49_mixing.staged_r40_alignment_scan.pre_registration/1")
    assert pre_registration["run"] == "R40"
    assert pre_registration["card"] == (
        "TASK-20260920-r4-run-r40-fixed-source-alignment-scan")
    assert pre_registration["written_before_generation"] is True
    assert pre_registration["no_fitting"] is True
    assert pre_registration["optimizer_invoked"] is False
    assert pre_registration["total_decays"] == 9_000_000
    # The source grid: ten at-rest points with the route regimes and
    # the blend weights on disk
    points = pre_registration["source_grid"]["points"]
    assert len(points) == 10
    for p in points:
        assert p["at_rest"] is True
        assert p["axis"] == [0.0, 0.0, 1.0]
        if p["point"] in (0, 5):
            assert p["route_regime"] == "catalog"
            assert p["blend_weight"] == 0.0
        elif p["point"] in (1, 6):
            assert p["route_regime"] == "blend_band"
            assert p["blend_weight"] == 0.6396117785449444
        else:
            assert p["route_regime"] == "continuum"
            assert p["blend_weight"] == 1.0
    assert [p["point"] for p in points] == list(range(10))
    assert points[0]["qn"] == [1, 1, 0]
    assert points[5]["qn"] == [0, 0, 0]
    assert [p["mass_gev"] for p in points] == [
        2.0, 3.0, 4.0, 6.0, 8.0, 2.0, 3.0, 4.0, 6.0, 8.0]
    # The scan grid: the nominal point, nine settings, the slices
    scan = pre_registration["scan_grid"]
    assert scan["nominal_point"] == {
        "kappa_b": 1.0, "kappa_m": 1.0,
        "b_dec_per_gev_squared": 1.0,
    }
    assert len(scan["settings"]) == 9
    assert [s["setting"] for s in scan["settings"]] == list(range(9))
    assert scan["slices"] == {
        "kappa_B": [0, 1, 2, 3],
        "kappa_M": [4, 1, 5, 6],
        "b_dec": [7, 1, 8],
    }
    # The T1 statistics
    stats = pre_registration["statistics"]
    assert stats["replicas"] == 4
    assert stats["decays_per_replica"] == 25_000
    assert stats["decays_per_point"] == 100_000
    assert stats["settings"] == 9
    assert stats["points"] == 10
    assert stats["total_decays"] == 9_000_000
    # The seed map
    seed = pre_registration["seed_map"]
    assert seed["master_seed"] == 20260920
    assert seed["raw_replica_seeds"] == [
        20301020, 20301021, 20301022, 20301023]
    assert seed["event_index_rule"] == (
        "300_000_000 + s*10_000_000 + p*1_000_000 + i, i < 25,000")
    assert seed["smoke"] == {
        "setting": 1,
        "point": 0,
        "replicas": 1,
        "decays_per_replica": 2000,
        "raw_seed_rule": (
            "raw smoke seed = 20260920 + 40000 + 7000 = 20307920 "
            "(the +7000 smoke block, one replica); the smoke kernel "
            "layer draws at derive_substreams(20260920, 7000, "
            "event_index) on the production event-index partition of "
            "(setting 1, point 0), disjoint from every production "
            "space (replica indices 0..3), from the R32 +5000 smoke "
            "block, from the R32 confirmation blocks, and from every "
            "previously used block"
        ),
        "raw_seed": 20307920,
        "kernel_replica_slot": 7000,
    }
    # The routing surface (the card-6 blend, the audited surface of
    # card 1)
    routing = pre_registration["routing"]
    assert routing["surface"] == (
        "na49_mixing/spectral_density.py (the card-6 blend, the "
        "audited surface of card 1)")
    assert routing["boundary_gev"] == 2.680194110727528
    assert routing["ramp_width_gev"] == 0.5
    assert routing["weight_at_point_masses"] == {
        "2.0": 0.0,
        "3.0": 0.6396117785449444,
        "4.0": 1.0,
        "6.0": 1.0,
        "8.0": 1.0,
    }
    # The acceptance-(v) check-set pre-registration
    acceptance = pre_registration["acceptance"]["v_kernel_verification"]
    assert acceptance["kappa_B"]["slice"] == [0, 1, 2, 3]
    assert acceptance["kappa_B"]["observable"] == (
        "baryon_line_fraction (the baryon-line longitudinal "
        "retention)")
    assert acceptance["kappa_B"]["points_predicted_increasing"] == [
        1, 2, 3, 4]
    assert acceptance["kappa_B"]["points_predicted_flat"] == [0]
    assert acceptance["kappa_M"]["slice"] == [4, 1, 5, 6]
    assert acceptance["kappa_M"]["observable"].startswith(
        "central multiplicity")
    assert acceptance["kappa_M"]["points_predicted_flat"] == [
        5, 6, 7, 8, 9]
    assert acceptance["b_dec"]["slice"] == [7, 1, 8]
    assert acceptance["b_dec"]["observable"] == (
        "mean_pt_x per paper-window species")
    assert acceptance["b_dec"]["points_predicted_decreasing"] == [
        1, 2, 3, 4, 6, 7, 8, 9]
    assert acceptance["b_dec"]["points_predicted_flat"] == [0, 5]
    assert acceptance["b_dec"]["min_mean_multiplicity"] == 0.01
    assert "DETERMINISTIC_SE_FLOOR = 1e-12" in acceptance["se_convention"]
    # The substrate verification against the committed bytes
    substrate = pre_registration["substrate"]
    assert substrate["verified"] is True
    for name, (rel, digest) in SUBSTRATE.items():
        assert substrate["digests"][name] == {
            "path": rel, "sha256": digest,
        }, name
    # The sealed run dirs
    sealed = pre_registration["sealed_run_dirs"]
    assert [d["path"] for d in sealed] == list(driver.SEALED_RUN_DIRS)
    assert [d["entries"] for d in sealed] == [3, 79, 29, 55]
    assert all(d["verified"] is True for d in sealed)
    # The obligation-4 model-uncertainty carry (the R32 spread, the
    # band-only scope, the audit outcomes)
    carry = pre_registration["obligation_4_carry"]
    assert carry["obligation"] == 4
    assert carry["r32_spread"] == {
        "bin_spread_p95": 4.028426,
        "feeddown_spread_max_sigma": 31.1773,
        "persisting_residuals": 32,
    }
    assert carry["applies_to"] == (
        "3 GeV blend-band points (w = 0.6396)")
    assert carry["not_applicable_to"] == (
        "the 2 GeV catalog points and the 4, 6, 8 GeV continuum "
        "points are off-band and the term is recorded as not "
        "applicable to them")
    assert carry["audit_outcome"]["card_2"] == (
        "PASS (dev/00runs/"
        "r4_audit_strange_sector_continuum_families_2026-09-20/"
        "verdict.json)")
    assert carry["audit_outcome"]["card_3"] == (
        "PASS (dev/00runs/"
        "r4_audit_builder_blend_path_routing_weight_2026-09-20/"
        "verdict.json)")
    assert carry["audit_outcome"]["card_1"].startswith(
        "RECOMMENDATION (sealed dev/00runs/"
        "r4_audit_transition_routing_formulation_2026-09-20/"
        "verdict.json)")
    assert ("TASK-20260920-r4-rework-transition-routing-"
            "formulation") in carry["audit_outcome"]["card_1"]


# ----------------------------------------------------------------------
# The sealed scan report: conservation, depth, routing census
# ----------------------------------------------------------------------
def test_scan_report_headline_census(scan_report):
    """The report's identity, conservation summary, depth summary,
    and the full 90-entry routing census with the per-entry
    structure."""
    assert scan_report["schema"] == (
        "na49_mixing.staged_r40_alignment_scan.report/1")
    assert scan_report["run"] == "R40"
    assert scan_report["card"] == (
        "TASK-20260920-r4-run-r40-fixed-source-alignment-scan")
    assert scan_report["mode"] == "production"
    assert scan_report["settings"] == 9
    assert scan_report["points"] == 10
    assert scan_report["decays_checked"] == 9_000_000
    assert scan_report["no_fitting"] is True
    assert scan_report["optimizer_invoked"] is False
    # Conservation: zero violations, the exact worst closures
    conservation = scan_report["conservation"]
    assert conservation == {
        "events_checked": 9_000_000,
        "qn_violations": 0,
        "worst_energy_closure_gev": WORST_ENERGY_CLOSURE_GEV,
        "worst_lightcone_identity_gev": WORST_LIGHTCONE_IDENTITY_GEV,
        "tolerance": CLOSURE_TOLERANCE,
        "pass": True,
    }
    # The depth summary
    depth = scan_report["depth"]
    assert depth["guard_activations"] == 0
    assert depth["max_over_points"] == 4
    assert depth["p95_max_over_points"] == 4
    assert depth["mean_max_over_points"] == 3.09539
    assert depth["guard_bound"] == 12
    assert len(depth["per_setting_point"]) == 90
    # The per-(setting, point) entries: the full routing census
    psp = scan_report["observables"]["per_setting_point"]
    assert len(psp) == 90
    for setting in range(9):
        for point in range(10):
            entry = psp[f"setting_{setting}_point_{point}"]
            assert set(entry) == {
                "mass_gev", "n_decays", "replicas", "species",
                "partition", "alignment_moment", "depth",
                "routing_census", "conservation",
            }
            n_dec, n_cont, frac = ROUTING_CENSUS[(setting, point)]
            census = entry["routing_census"]
            assert set(census) == {
                "n_catalog", "n_continuum", "n_band_catalog",
                "n_band_continuum", "fraction_continuum",
                "blend_weight_at_mass",
            }
            assert census["n_catalog"] == n_dec, (setting, point)
            assert census["n_continuum"] == n_cont, (setting, point)
            assert census["fraction_continuum"] == frac, (setting, point)
            # The band counters: on-band points (1, 6) carry the whole
            # census in the band; off-band points record zero
            if point in (1, 6):
                assert census["n_band_catalog"] == n_dec
                assert census["n_band_continuum"] == n_cont
            else:
                assert census["n_band_catalog"] == 0
                assert census["n_band_continuum"] == 0
            assert census["n_catalog"] + census["n_continuum"] == (
                entry["n_decays"])
            if point in (0, 5):
                assert census["blend_weight_at_mass"] == 0.0
            elif point in (1, 6):
                assert census["blend_weight_at_mass"] == (
                    0.6396117785449444)
            else:
                assert census["blend_weight_at_mass"] == 1.0
            assert entry["n_decays"] == 100_000
            assert entry["replicas"] == 4
            expected_mass = [2.0, 3.0, 4.0, 6.0, 8.0,
                             2.0, 3.0, 4.0, 6.0, 8.0][point]
            assert entry["mass_gev"] == expected_mass
            assert set(entry["species"]) == set(SPECIES)
            assert set(entry["conservation"]) == {
                "events_checked", "qn_violations",
                "worst_energy_closure_gev",
                "worst_lightcone_identity_gev",
            }
            assert entry["conservation"]["qn_violations"] == 0
            assert entry["conservation"]["events_checked"] == 100_000
            assert set(entry["depth"]) == {
                "all", "catalog_route", "continuum_route",
                "guard_activations"}
            assert entry["depth"]["guard_activations"] == 0
            assert entry["depth"]["all"]["n_decays"] == 100_000
            assert entry["depth"]["all"]["guard_activations"] == 0
            # The route blocks: the route with no decays at a (setting,
            # point) is null, the other carries its census
            assert (entry["depth"]["catalog_route"] is None) == (
                n_dec == 0)
            assert (entry["depth"]["continuum_route"] is None) == (
                n_cont == 0)
            if entry["depth"]["catalog_route"] is not None:
                assert entry["depth"]["catalog_route"]["n_decays"] == n_dec
            if entry["depth"]["continuum_route"] is not None:
                assert entry["depth"]["continuum_route"]["n_decays"] == (
                    n_cont)
    # The report's determinism and obligation-4 blocks
    assert scan_report["determinism"]["records_with_digest"] == 360
    assert scan_report["determinism"]["records_total"] == 360
    assert scan_report["obligation_4_carry"]["obligation_4"] is True
    assert scan_report["obligation_4_carry"]["r32_spread"] == {
        "bin_spread_p95": 4.028426,
        "feeddown_spread_max_sigma": 31.1773,
        "persisting_residuals": 32,
    }


# ----------------------------------------------------------------------
# The check set: the endpoint-contrast sigma algebra and the named
# residuals
# ----------------------------------------------------------------------
def _endpoint_contr_sigma(row) -> float:
    """The pre-registered endpoint-contrast sigma: the raw contrast
    ``(x_last - x_first) / sqrt(se_first^2 + se_last^2)`` with the
    predicted direction applied for the direction verdicts (a flat
    prediction carries the raw contrast)."""
    values = row["values"]
    ses = row["endpoint_ses"]
    se = math.sqrt(ses[0] * ses[0] + ses[-1] * ses[-1])
    raw = (values[-1] - values[0]) / se
    d = row["predicted_direction"]
    return raw if d == 0 else d * raw


def test_check_set_sigma_algebra(scan_report):
    """The 58-row check set: the endpoint-contrast sigma algebra
    reproduced bit for bit on every row, the verdict rule, and the
    two named residuals at their exact sigmas."""
    check_set = scan_report["check_set"]
    assert check_set["status"] == "evaluated"
    assert check_set["sigma_band"] == 3.0
    rows = check_set["rows"]
    assert len(rows) == 58
    row_keys = {
        "response", "point", "observable", "species",
        "settings_order", "predicted_direction", "values",
        "endpoint_ses", "sigma", "verdict",
        "robust_statistics_reference",
    }
    for row in rows:
        # Failed rows carry a diagnosis string; passing rows do not
        if row["verdict"] == "fail":
            assert set(row) == row_keys | {"diagnosis"}
        else:
            assert set(row) == row_keys
        # The sigma algebra, bit for bit (the on-disk floats are the
        # same IEEE operations over the same endpoints)
        assert _endpoint_contr_sigma(row) == row["sigma"], row
        # The verdict rule: a direction verdict fails at
        # sigma < -3.0, a flat verdict at |sigma| > 3.0
        d = row["predicted_direction"]
        expected = "fail" if (
            d == 0 and abs(row["sigma"]) > 3.0
        ) or (d != 0 and row["sigma"] < -3.0) else "pass"
        assert row["verdict"] == expected, row
    # Exactly two named residuals, at the exact sigmas, with the
    # 1-based row positions consistent with the check set rows
    register = check_set["named_residual_register"]
    assert len(register) == 2
    assert scan_report["named_residual_register"] == register
    for entry in register:
        key = (entry["response"], entry["point"], entry["observable"],
               entry["species"])
        assert key in NAMED_RESIDUAL_SIGMAS, key
        assert entry["sigma"] == NAMED_RESIDUAL_SIGMAS[key], key
        row = rows[entry["index"] - 1]
        assert (row["response"], row["point"], row["observable"],
                row["species"], row["sigma"],
                row["predicted_direction"]) == (
            entry["response"], entry["point"], entry["observable"],
            entry["species"], entry["sigma"],
            entry["predicted_direction"],
        )
    assert {e["index"] for e in register} == {17, 26}
    # The two failed rows and their exact on-disk diagnoses (the
    # named residuals: b_dec point 0 pi_minus, b_dec point 2 k_plus)
    failed = [r for r in rows if r["verdict"] == "fail"]
    assert [(r["response"], r["point"], r["observable"], r["species"])
            for r in failed] == [
        ("b_dec", 0, "mean_pt_x", "pi_minus"),
        ("b_dec", 2, "mean_pt_x", "k_plus"),
    ]
    assert failed[0]["diagnosis"] == (
        "the b_dec endpoint contrast at point 0 moves beyond the "
        "3.0-sigma band (|sigma| = 3.61) against the pre-registered "
        "zero-response prediction: a driver defect or a physics "
        "finding, diagnosed in this register, not refit here "
        "(robust-statistics cross-reference: the between-replica SE "
        "convention with the deterministic SE floor; the R32 "
        "finite-MC-scatter envelope applies at |sigma| <= 4.0)")
    assert failed[1]["diagnosis"] == (
        "the b_dec endpoint contrast at point 2 moved against the "
        "pre-registered direction beyond the 3.0-sigma band "
        "(sigma = -3.60): a driver defect or a physics finding, "
        "diagnosed in this register, not refit here "
        "(robust-statistics cross-reference: the between-replica SE "
        "convention with the deterministic SE floor; the R32 "
        "finite-MC-scatter envelope applies at |sigma| <= 4.0)")
    # The slice orders on disk
    kappa_b_rows = [r for r in rows if r["response"] == "kappa_B"]
    kappa_m_rows = [r for r in rows if r["response"] == "kappa_M"]
    b_dec_rows = [r for r in rows if r["response"] == "b_dec"]
    assert [r["point"] for r in kappa_b_rows] == [0, 1, 2, 3, 4]
    assert all(r["settings_order"] == [0, 1, 2, 3]
               for r in kappa_b_rows)
    assert [r["point"] for r in kappa_m_rows] == list(range(10))
    assert all(r["settings_order"] == [4, 1, 5, 6]
               for r in kappa_m_rows)
    assert len(b_dec_rows) == 43
    assert all(r["observable"] == "mean_pt_x" for r in b_dec_rows)
    assert all(r["settings_order"] == [7, 1, 8] for r in b_dec_rows)


# ----------------------------------------------------------------------
# The gate 4 evidence
# ----------------------------------------------------------------------
def test_gate4_evidence(gate_record, scan_report):
    """The gate record's identity, its conservation and depth
    summaries against the report, the model-uncertainty carry, and
    the named-residual register's equality with the report's."""
    assert gate_record["schema"] == (
        "na49_mixing.staged_r40_alignment_scan.gate/1")
    assert gate_record["gate"] == "gate 4 (plan section 10, line 806)"
    assert gate_record["clause"] == (
        "shape (training-side scan evidence)")
    assert gate_record["run"] == "R40"
    assert gate_record["card"] == (
        "TASK-20260920-r4-run-r40-fixed-source-alignment-scan")
    assert gate_record["settings"] == 9
    assert gate_record["points"] == 10
    assert gate_record["decays_checked"] == 9_000_000
    assert gate_record["no_fitting"] is True
    assert gate_record["optimizer_invoked"] is False
    assert gate_record["sealed"] is True
    # Conservation against the report
    assert gate_record["conservation"] == {
        "violations": 0,
        "pass": True,
        "worst_energy_closure_gev": WORST_ENERGY_CLOSURE_GEV,
        "tolerance": CLOSURE_TOLERANCE,
    }
    # Depth against the report's summary
    assert gate_record["depth"]["guard_activations"] == 0
    assert gate_record["depth"]["guard_bound"] == 12
    assert gate_record["depth"]["max_over_points"] == 4
    assert gate_record["depth"]["p95_max_over_points"] == 4
    assert gate_record["depth"]["mean_max_over_points"] == 3.09539
    assert len(gate_record["depth"]["per_setting_point"]) == 90
    # The model-uncertainty carry
    assert gate_record["model_uncertainty_carry"] == {
        "obligation_4": True,
        "r32_spread": {
            "bin_spread_p95": 4.028426,
            "feeddown_spread_max_sigma": 31.1773,
            "persisting_residuals": 32,
        },
        "applies_to": "3 GeV blend-band points (w = 0.6396)",
    }
    # The named-residual register is the report's register
    assert gate_record["named_residual_register"] == (
        scan_report["check_set"]["named_residual_register"])
    # The observables section covers all 90 (setting, point) entries
    assert len(
        gate_record["observables"]["per_setting_point"]
    ) == 90


# ----------------------------------------------------------------------
# The seal manifest and the checksum file
# ----------------------------------------------------------------------
def test_seal_manifest_and_checksums(seal_manifest, clean_environment,
                                     scan_report):
    """The seal manifest's 365 entries path-sorted (360 setting
    records plus the five top-level files), the checksum file's 364
    entries, the sums file's own digest, and a sampled re-hash of ten
    sealed files."""
    assert seal_manifest["schema"] == (
        "na49_mixing.staged_r40_alignment_scan.seal_manifest/1")
    assert seal_manifest["run"] == "R40"
    assert seal_manifest["card"] == (
        "TASK-20260920-r4-run-r40-fixed-source-alignment-scan")
    assert seal_manifest["file_count"] == 365
    assert seal_manifest["total_bytes"] == 2_981_861
    assert seal_manifest["sha256_sums_file"] == "SHA256SUMS.txt"
    assert seal_manifest["excluded_from_sums"] == [
        "seal_manifest.json", "SHA256SUMS.txt"]
    files = seal_manifest["files"]
    assert len(files) == 365
    paths = [f["path"] for f in files]
    assert paths == sorted(paths), "the seal entries are not path-sorted"
    # 360 setting records plus the five top-level files
    setting_paths = [p for p in paths if p.startswith("setting_")]
    top_paths = sorted(p for p in paths if not p.startswith("setting_"))
    assert len(setting_paths) == 360
    assert top_paths == [
        "SHA256SUMS.txt", "gate4_r40.json", "pre_registration.json",
        "provenance.json", "scan_report.json",
    ]
    for setting in range(9):
        for point in range(10):
            for replica in range(4):
                name = f"setting_{setting}_point_{point}_replica_{replica}.json"
                assert name in paths, name
    # The sums file's own digest (the self-pin of SHA256SUMS.txt)
    sums_path = RUN_DIR / "SHA256SUMS.txt"
    assert _sha256_file(sums_path) == seal_manifest["sha256_sums_sha256"]
    by_name = {f["path"]: f for f in files}
    assert by_name["SHA256SUMS.txt"]["sha256"] == (
        seal_manifest["sha256_sums_sha256"])
    assert by_name["SHA256SUMS.txt"]["bytes"] == 55_630
    # The checksum file: 364 entries (the seal manifest cannot carry
    # its own digest), monorepo-root-relative paths
    sums_lines = sums_path.read_text(encoding="utf-8").splitlines()
    assert len(sums_lines) == 364
    for line in sums_lines:
        digest, rel = line.split("  ")
        assert rel.startswith(
            "dev/00runs/r40_fixed_source_alignment_scan_2026-09-21/")
        assert len(digest) == 64
    # A sampled re-hash of ten sealed files (the five top-level files
    # plus five setting records spread over the setting/point/replica
    # space), against both the seal manifest and the checksum file
    sampled = [
        "pre_registration.json",
        "scan_report.json",
        "gate4_r40.json",
        "provenance.json",
        "SHA256SUMS.txt",
        "setting_0_point_0_replica_0.json",
        "setting_1_point_0_replica_0.json",
        "setting_4_point_5_replica_2.json",
        "setting_7_point_8_replica_1.json",
        "setting_8_point_9_replica_3.json",
    ]
    for name in sampled:
        on_disk = _sha256_file(RUN_DIR / name)
        assert by_name[name]["sha256"] == on_disk, name
        # SHA256SUMS.txt excludes itself from the sums file (it is
        # named in excluded_from_sums); every other sampled file is
        # in the checksum file
        if name == "SHA256SUMS.txt":
            assert name in seal_manifest["excluded_from_sums"]
            assert all(not line.endswith(
                "/SHA256SUMS.txt") for line in sums_lines)
            continue
        rel = f"dev/00runs/r40_fixed_source_alignment_scan_2026-09-21/{name}"
        assert f"{on_disk}  {rel}" in sums_lines, name
    # The sealed report's digest is the on-disk report's digest and
    # the clean-environment record's sealed hash
    report_digest = _sha256_file(RUN_DIR / "scan_report.json")
    assert by_name["scan_report.json"]["sha256"] == report_digest
    assert clean_environment["sealed_report_sha256"]["scan_report.json"] == (
        report_digest)
    assert scan_report is not None  # the fixture loaded the report


# ----------------------------------------------------------------------
# The clean-environment validation record
# ----------------------------------------------------------------------
def test_clean_environment_record(clean_environment, seal_manifest):
    """The clean-environment validation: the seal verification, the
    byte-identical report re-hash, the probe-chunk checks, and the
    probe instrument's digest."""
    assert clean_environment["schema"] == (
        "na49_mixing.staged_r40_alignment_scan.clean_environment/1")
    assert clean_environment["run"] == "R40"
    assert clean_environment["card"] == (
        "TASK-20260920-r4-run-r40-fixed-source-alignment-scan")
    assert clean_environment["verified_utc"].startswith("2026-09-22")
    # The seal verification: all 364 sums-file entries verified, no
    # mismatches
    assert clean_environment["seal_verification"] == {
        "files_checked": 364,
        "mismatches": [],
    }
    # The byte-identical report re-hash
    assert clean_environment["report_hashes_match"] is True
    sealed = clean_environment["sealed_report_sha256"]
    recomputed = clean_environment["recomputed_report_sha256"]
    assert sealed == recomputed
    assert sealed["scan_report.json"] == _sha256_file(
        RUN_DIR / "scan_report.json")
    # The probe chunk (setting 1, point 0, replica 0) re-ran cleanly
    assert clean_environment["probe_returncode"] == 0
    assert clean_environment["probe_chunk_checks"] == {
        "n_decays": True,
        "rolling_digest": True,
        "route_census": True,
        "depth_histograms": True,
        "draw_budget": True,
    }
    probe_digest = clean_environment["probe_chunk_rolling_digest"]
    assert probe_digest["sealed"] == probe_digest["recomputed"]
    record = json.loads(
        (RUN_DIR / "setting_1_point_0_replica_0.json")
        .read_text(encoding="utf-8"))
    assert record["rolling_digest"] == probe_digest["sealed"]
    assert record["setting"] == 1
    assert record["point"] == 0
    assert record["replica"] == 0
    # The probe instrument: the committed dual-tree driver
    assert clean_environment["probe_tool"]["sha256"] == DRIVER_SHA256
    assert clean_environment["probe_tool"]["path"].endswith(
        "helpers/benchmarking/staged_r40_alignment_scan.py")
    # The clean-environment record is the post-seal validation record:
    # it is not part of the seal
    assert "clean_environment_commands.json" not in [
        f["path"] for f in seal_manifest["files"]]


# ----------------------------------------------------------------------
# The provenance
# ----------------------------------------------------------------------
def test_provenance_on_disk(driver, provenance):
    """The provenance's schema, the card chain, the recorded
    pre-registration driver digests, the substrate digests, and the
    seed-map fields."""
    assert provenance["schema"] == (
        "na49_mixing.staged_r40_alignment_scan.provenance/1")
    assert provenance["run"] == "R40"
    assert provenance["card"] == (
        "TASK-20260920-r4-run-r40-fixed-source-alignment-scan")
    assert provenance["written_before_generation"] is True
    # The card chain (the driver's, on disk)
    assert provenance["card_chain"] == list(driver.CARD_CHAIN)
    # The driver digest recorded at pre-registration time (both trees
    # byte-identical; the committed driver carries the distinct
    # digest DRIVER_SHA256, pinned in test_dual_tree_driver_byte_
    # identity and used as the clean-environment probe instrument)
    recorded = provenance["driver_sha256"]
    assert recorded["nested"] == PROVENANCE_DRIVER_SHA256
    assert recorded["outer"] == PROVENANCE_DRIVER_SHA256
    assert recorded["nested"] == recorded["outer"]
    # The substrate digests against the committed bytes
    assert provenance["cascade_modules_sha256"] == {
        name: SUBSTRATE[name][1]
        for name in ("cascade_decay", "cascade_axes", "cascade_alignment",
                     "cascade_form_factor", "cascade_termination",
                     "cascade_ablation")
    }
    assert provenance["round2_kernel_sha256"] == (
        SUBSTRATE["round2_kernel"][1])
    assert provenance["builder_blend_path_sha256"] == (
        SUBSTRATE["builder_blend_path"][1])
    assert provenance["card6_routing_sha256"] == SUBSTRATE["card6_routing"][1]
    assert provenance["card2_engine_sha256"] == SUBSTRATE["card2_engine"][1]
    assert provenance["catalog"]["sha256"] == SUBSTRATE["catalog"][1]
    assert provenance["continuum_config"]["sha256"] == (
        SUBSTRATE["continuum_config"][1])
    assert provenance["r32_driver_sha256"] == SUBSTRATE["r32_driver"][1]
    # The routing surface fields
    assert provenance["kernel_boundary_gev"] == 2.680194110727528
    assert provenance["ramp_width_gev"] == 0.5
    assert provenance["blend_weight_at_3gev"] == 0.6396117785449444
    # The seed map
    assert provenance["master_seed"] == 20260920
    assert provenance["raw_replica_seeds"] == [
        20301020, 20301021, 20301022, 20301023]
    assert provenance["sealed_run_dirs"] == [
        {"path": "dev/00runs/r30_spectral_normalization_2026-09-17",
         "entries": 3, "verified": True},
        {"path": (
            "dev/00runs/r31_catalog_continuum_matching_scan_2026-09-17"),
         "entries": 79, "verified": True},
        {"path": (
            "dev/00runs/r32_spectral_systematic_pp_2026-09-18"),
         "entries": 29, "verified": True},
        {"path": (
            "dev/00runs/r32_spectral_systematic_pp_confirmation_"
            "2026-09-19"),
         "entries": 55, "verified": True},
    ]


# ----------------------------------------------------------------------
# The frozen substrate, byte-stable on disk
# ----------------------------------------------------------------------
def test_frozen_substrate_stability():
    """The driver's thirteen frozen substrate files byte-stable on
    disk at their pinned digests (verified independently of the
    driver's table)."""
    for name, (rel, digest) in SUBSTRATE.items():
        path = MONO_ROOT / rel
        assert path.is_file(), f"missing frozen surface: {rel}"
        assert _sha256_file(path) == digest, f"drifted surface: {rel}"


# ----------------------------------------------------------------------
# The freeze re-pin register (the test-author re-pin)
# ----------------------------------------------------------------------
