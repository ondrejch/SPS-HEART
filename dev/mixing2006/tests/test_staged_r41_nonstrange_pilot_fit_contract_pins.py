"""Round 4, card 16 test-author register: the R41 nonstrange pilot fit
relitigation contract pins.

Card ``TASK-20260920-r4-run-r41-relitigation`` (plan section 10 run R41,
line 768; the Round 4 card set, card 16; supersedes card 12 and the
sealed ``dev/00runs/r41_nonstrange_pilot_fit_2026-09-22/`` archive,
which stays intact and is the supersession reference this register
pins via the provenance).  This register is the test-author half of
the card: independent pins of the corrected dual-tree run driver
(``helpers/benchmarking/staged_r41_nonstrange_pilot_fit.py``,
byte-identical in the nested tree and the monorepo mirror, reworked to
the R01 accumulate-dense-then-pick recording convention), of the
hermetic solver test
(``tests/test_staged_r41_nonstrange_pilot_fit.py``), and of the
relitigated sealed run
``dev/00runs/r41_nonstrange_pilot_fit_2026-09-23/``.  This register
verifies the relitigated archive; the superseded 2026-09-22 archive is
read-only and byte-unchanged.  The driver is loaded once via importlib
(its top level is the pre-registration identity; no events are
generated), and the sealed JSON is read from disk.  The solver's
pinned seed-map and pinned traces are not re-run here; the contract is
pinned on the committed bytes and the sealed on-disk values.

- ``test_dual_tree_driver_byte_identity``: both driver copies exist,
  are byte-identical, and sit at the committed digest; the driver's own
  self-computed identity digest agrees; the freeze manifest inventory
  pins both at that digest.
- ``test_driver_identity_constants``: the driver's run identity, the
  file-name constants, the substrate and sealed-dir rel paths, the card
  chain, the R32 p+p collision configuration, the four-parameter fit
  set (parameters, the obligation 4 exclusion, the initial values, the
  bounds, the width grid, the reweightable directions), the optimizer
  contract, the nuisance structure, the obligation 4 carry, and the
  stability statistics.
- ``test_seed_map_partition``: the master seed, the raw block
  20301020-20301027, the event-index partition 400_000_000 + i, the
  kernel seed rule, the hermetic smoke block (+8000), and the seed
  functions' refusal discipline.
- ``test_cli_surface``: the exact argparse command surface on disk
  (the program name, exactly the seven mode flags, the three auxiliary
  flags and their defaults) plus the on-disk exit-code behavior (help
  exits 0, no mode refuses with 2, two modes refuse with 2).
- ``test_pre_registration_ordering_and_drift``: the
  pre-registration-before-generation ordering invariant (a non-empty
  run dir is refused before the substrate is touched) and the
  pre-registration-vs-card drift probes (fit-set, statistics, seed,
  width-grid, and written_before_generation drift are refused; the
  smoke mode is exempt; the sealed pre-registration passes).
- ``test_pre_registration_on_disk``: the pre-registration's schema,
  the T2 statistics, the seed map, the fit set, the collision
  configuration, the conservation and observables blocks, the
  likelihood form, the hessian discipline, the acceptance check set
  with its five register triggers, the blend-band-sensitive bins, the
  obligation 4 carry with the audit outcomes, the recording
  convention (dense accumulation, the pick map, the hole geometry,
  the superseded and corrected driver digests), the substrate
  verification, and the sealed run dirs.
- ``test_fit_report_on_disk``: the report's schema, the internal
  digest, the boundary-constrained optimum on all four parameters
  (kappa_B 0.0, kappa_m 4.0, b_dec 2.0 GeV^-2, width 0.3 at the
  pre-registered grid edge), the eight per-block fits (interior
  per-block optima; the block minima do not agree), the optimizer
  profile on the five-point width grid (the argmin on the grid edge,
  -2lnL 7866.301937844129), the replica-block stability table (kappa_B
  and kappa_m fail the pre-registered 20% pooled-error bound; b_dec
  passes degenerate on the infinite bound), the not-quoted Hessian
  block (the replica-block minima do not agree), the eight-entry
  named-residual register, and the pooled objective at the optimum.
- ``test_gate4_evidence``: the gate record's identity, its conservation
  summary (zero violations over 800,000 events), the fit-report digest
  cross-reference, and the equality of its stability, Hessian, carry,
  and named-residual blocks with the report's (the width_curvature
  NaN is not self-equal under == and is asserted separately).
- ``test_seal_manifest_and_checksums``: the seal manifest's 13 sealed
  files (8 replica records plus the five top-level files) path-sorted,
  the checksum file's 13 entries, and the full re-hash of all 13
  sealed files against the checksum digests.
- ``test_clean_environment_record``: the clean-environment
  validation's command list, the byte-identical report re-hash
  (digest equality), the probe replica record, and the pending seal
  state.
- ``test_provenance_on_disk``: the provenance's schema, the substrate
  digests, the M8 frozen set hash, the recorded driver digest (equal
  to the committed dual-tree digest this time: the relitigation
  pre-registered against the committed corrected driver), the
  superseded archive's verified seal state as the supersession
  reference, and the sealed run dir verification.
- ``test_frozen_substrate_stability``: the driver's fourteen frozen
  substrate files byte-stable on disk at their pinned digests.
- ``test_solver_test_file_digest``: the hermetic solver test file at
  its committed digest, pinned in the freeze manifest inventory, with
  the card-12 solver freeze exception intact (the four path-sorted
  files).
- ``test_freeze_repin_register``: the freeze-manifest re-pin for this
  register (the strict path-sorted inventory at 480 entries carries
  this file at its on-disk digest; the card-12 solver freeze exception
  stays at the solver's 4 path-sorted files; this register's own
  freeze exception names the test-author card exactly once with the 17
  path-sorted files; the fifteen consumed count pins are re-pinned
  479 -> 480 with the naming comment attributing the bump to this
  register, the card-12 solver's 476 -> 479 comment and the card-10
  solver's 471 literal retained for their re-pin checks).

Hermetic: standard library and numpy, the committed config and
catalog, and the sealed run dir (read-only) only; no SPS tables, no
event generation, no Monte Carlo run.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
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
    / "staged_r41_nonstrange_pilot_fit.py"
)
OUTER_DRIVER = (
    MONO_ROOT / "helpers" / "benchmarking"
    / "staged_r41_nonstrange_pilot_fit.py"
)
MANIFEST_PATH = (NESTED_ROOT / "config" / "v2_singlet"
                 / "legacy_v1_manifest.json")
RUN_DIR = (MONO_ROOT / "dev" / "00runs"
            / "r41_nonstrange_pilot_fit_2026-09-23")

#: The superseded sealed run dir (the card-12 archive; stays intact
#: and is pinned as the supersession reference via the provenance's
#: superseded_r41_archive block, not read directly by this register).
SUPERSEDED_RUN_DIR_NAME = (
    "dev/00runs/r41_nonstrange_pilot_fit_2026-09-22"
)

DRIVER_REL_NESTED = (
    "dev/mixing2006/helpers/benchmarking/"
    "staged_r41_nonstrange_pilot_fit.py"
)
DRIVER_REL_OUTER = (
    "helpers/benchmarking/staged_r41_nonstrange_pilot_fit.py"
)
SOLVER_TEST_REL = "dev/mixing2006/tests/test_staged_r41_nonstrange_pilot_fit.py"

#: The committed corrected dual-tree driver digest (byte-identical in
#: both trees; the freeze manifest inventory pins both copies here;
#: the card-16 relitigation rework of the recording convention).
DRIVER_SHA256 = (
    "abf2a8c30344ebeecab4413f5d1c4045b091ac3467226c6257e09d501697cdfc"
)

#: The superseded card-12 driver digest (the published-row flat index
#: recording; the driver bytes were superseded by the relitigation).
#: Pinned as the supersession reference in the pre-registration's
#: recording_convention block and the provenance's
#: superseded_r41_archive block.
SUPERSEDED_DRIVER_SHA256 = (
    "573d833bfbc659885fb83e6c0794207947fbe9e5a6dafe6159650bd45097172b"
)

#: The driver digest the sealed provenance records at pre-registration
#: time.  The relitigation pre-registered against the committed
#: corrected driver, so this run carries no distinct
#: pre-registration-time digest (unlike the card-12 landing, where
#: the driver was amended after the pre-registration).
PROVENANCE_DRIVER_SHA256 = (
    "abf2a8c30344ebeecab4413f5d1c4045b091ac3467226c6257e09d501697cdfc"
)

#: The hermetic solver test file's committed digest (18 hermetic
#: tests: the card-12 14 plus the four mirrored hole-geometry tests;
#: the card-16 relitigation landing).
SOLVER_TEST_SHA256 = (
    "0fe8a88a2ff9b6c6a814f79f3f6b2c2fe08ccb22aa80addeb8c4b9c8b6ec106d"
)

# ----------------------------------------------------------------------
# Freeze re-pin constants (the register chain's inventory count pins)
# ----------------------------------------------------------------------
MY_TEST_REL = (
    "dev/mixing2006/tests/"
    "test_staged_r41_nonstrange_pilot_fit_contract_pins.py"
)
MANIFEST_REL = (
    "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json"
)
SOLVER_CARD = "TASK-20260920-r4-run-r41-nonstrange-pilot-fit"
# The test-author re-pin's own freeze-exception card name (distinct
# from the solver's, so the solver's "exactly one entry naming this
# card" check is untouched).
TEST_AUTHOR_CARD = (
    "TASK-20260920-r4-run-r41-nonstrange-pilot-fit-test-author"
)

#: The fifteen register-chain files whose inventory-count pins this
#: register re-pins (479 -> 480, +1 entry): the twelve register-chain
#: files (cards 1, 4, 5, 6, 7, 8, 9, 10) plus the card-10 solver
#: test's own count constant (test_cascade_ablation.py), the card-11
#: test-author register's own count constant
#: (test_staged_r40_alignment_scan_contract_pins.py), and the two
#: card-1/card-4 registers whose pin is the in-test count assertion.
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
)

#: The card-12 solver freeze-exception file list (4 path-sorted files;
#: this register leaves it untouched).
SOLVER_EXCEPTION_FILES = (
    MANIFEST_REL,
    DRIVER_REL_NESTED,
    SOLVER_TEST_REL,
    DRIVER_REL_OUTER,
)

#: This register's own freeze-exception file list (path-sorted): the
#: manifest, this register, and the fifteen re-pinned count pins.
MY_EXCEPTION_FILES = tuple(
    sorted([MANIFEST_REL, MY_TEST_REL] + list(CHAIN_COUNT_PIN_FILES))
)

# 479 entries at the card-12 solver landing (the dual-tree R41 driver
# in both helper trees and the hermetic R41 test file); this register
# adds its file (+1 entry) for the test-author re-pin (479 -> 480,
# +1 entry) and re-pins the fifteen consumed count pins (the twelve
# register-chain files plus the card-10 solver test's own count
# constant, the card-11 test-author register's own count constant, and
# the card-1/card-4 registers' in-test count assertions) with the bump
# attributed to this register's file.
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
    "fit_report.json",
    "gate4_r41.json",
    "pre_registration.json",
    "provenance.json",
)

#: The on-disk digest of every sealed file of the relitigated archive
#: (the checksum file's entries plus the seal manifest's own digest).
SEALED_FILE_DIGESTS = {
    "clean_environment_commands.json": (
        "af435eff43091ff72e66c4b2af56f9db009482dd72ec45525423647bb5a8fb06"
    ),
    "fit_report.json": (
        "bc419ab16488f3219917654cba3d6657c1ef35d1222a5a0a656371df7849e5bb"
    ),
    "gate4_r41.json": (
        "c7d36cb27697c13b0767234e4a80471d95eaaa9c608afcd275a0d428c4b4a27c"
    ),
    "pre_registration.json": (
        "2b292a3500fa3466068fd5dfd8c94dad3b6d6de9c1f7c1dff0f0a5cb39b7dede"
    ),
    "provenance.json": (
        "d24d3cd774307daaff9f2ff23ea773764c3b5c0f80b7e267213c8715309e927b"
    ),
    "replica_0.json": (
        "80349325e54f44f960c6c4f093780fad66933ef9a57bdd8556ac3fb16515004d"
    ),
    "replica_1.json": (
        "4dd5a688dca152c85fa562d2bad3da34abe8bc0649ce3ec7cc681db22073afcb"
    ),
    "replica_2.json": (
        "520eb5db448c72671d53678d26f76da5c253d49d68b874ce37437cbb566fd79d"
    ),
    "replica_3.json": (
        "4a5ba6a7576c9cfc9fccf355caae3f7c47b0a28897d4c03a4334567af62bb9cb"
    ),
    "replica_4.json": (
        "31ee143e2b55deb0d880ce2fdd727f7b8d5c9c3c2d0e7c398e01e5dd1b1a3211"
    ),
    "replica_5.json": (
        "dc3d89f61755419eb989a238a68ece17320c0e915d8a1e02c596bd2dbbfa2641"
    ),
    "replica_6.json": (
        "0ff01a93ed6f838353c7a96a5bcf818bafd67d8779de60b285c36a5d0ac85f92"
    ),
    "replica_7.json": (
        "9c3466dbb6b8b020229eca6f7251e59d246d2698d422fa379f5ea6bd95832f4f"
    ),
    "seal_manifest.json": (
        "03c609100fc87aa3992b22c2f04dff052fd51b7bdb3a5b9ba26612ed9609cb9f"
    ),
}

#: The checksum file's own digest (excluded from its own entries).
CHECKSUMS_SHA256 = (
    "e761e182eb00cd5f252f907f905fbde8b0b94283732f73429938d2c3c1b65231"
)

#: The fit report's internal digest (the card's 76a40d34b5ccf376...
#: prefix; the report bytes carry no wall-clock fields).
FIT_REPORT_DIGEST = (
    "76a40d34b5ccf376127577fd2bc5de370d37a95d46379a32db8096f8dcd1d145"
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
EVENT_INDEX_BASE = 400_000_000
EVENT_INDEX_RULE = "400_000_000 + i, i < 100,000"
SMOKE_RAW_SEED = 20308920
SMOKE_KERNEL_REPLICA_SLOT = 8000

#: The pre-registered fit set and bounds (the four-parameter pilot
#: fit; the blend ramp start excluded by obligation 4).
FIT_SET = ("kappa_B", "kappa_m", "b_dec", "baryonic_nominal_width")
INITIAL_VALUES = {
    "kappa_B": 1.0,
    "kappa_m": 1.0,
    "b_dec": 1.0,
    "baryonic_nominal_width": 0.5,
}
PARAM_BOUNDS = {
    "kappa_B": (0.0, 4.0),
    "kappa_m": (0.0, 4.0),
    "b_dec": (0.0, 2.0),
    "baryonic_nominal_width": (0.3, 0.7),
}
W_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)

#: The pre-registered obligation 4 carry (the sealed R32 spread,
#: verbatim from the card).
OBLIGATION4_R32_SPREAD = {
    "bin_spread_p95": 4.028426,
    "feeddown_spread_max_sigma": 31.1773,
    "persisting_residuals": 32,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nan_equal(left, right) -> bool:
    """Recursive equality that treats NaN as equal to NaN.  The sealed
    JSON round-trip leaves the width_curvature field as NaN, which is
    not self-equal under ==; the gate/report block comparisons must
    not hinge on that comparison."""
    if isinstance(left, float) and math.isnan(left):
        return isinstance(right, float) and math.isnan(right)
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(_nan_equal(left[key], right[key]) for key in left))
    if isinstance(left, list):
        return (isinstance(right, list) and len(left) == len(right)
                and all(_nan_equal(a, b) for a, b in zip(left, right)))
    return left == right


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "staged_r41_nonstrange_pilot_fit_under_test", NESTED_DRIVER
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
def fit_report():
    skip_unless_sealed(RUN_DIR)
    return json.loads(
        (RUN_DIR / "fit_report.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def gate_record():
    skip_unless_sealed(RUN_DIR)
    return json.loads(
        (RUN_DIR / "gate4_r41.json").read_text(encoding="utf-8"))


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
    collision configuration, the four-parameter fit set, the optimizer
    contract, the nuisance structure, and the stability statistics."""
    assert driver.RUN_ID == "R41"
    assert driver.CARD == SOLVER_CARD
    assert driver.RUN_NAME == "r41_nonstrange_pilot_fit_2026-09-23"
    assert driver.DEFAULT_RUN_DIR == (
        "dev/00runs/r41_nonstrange_pilot_fit_2026-09-23")
    assert driver.GATE_RECORD_NAME == "gate4_r41.json"
    assert driver.FIT_REPORT_NAME == "fit_report.json"
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
        "helpers/benchmarking/staged_r41_nonstrange_pilot_fit.py")
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
    )
    # The R32 p+p collision configuration (index item 7)
    assert driver.BEAM_MOMENTUM_GEV_C == BEAM_MOMENTUM_GEV_C
    assert driver.SQRT_S_GEV == SQRT_S_GEV
    assert driver.BLEND_RAMP_START_GEV == BLEND_RAMP_START_GEV
    assert driver.BLEND_RAMP_WIDTH_GEV == BLEND_RAMP_WIDTH_GEV
    assert driver.M8_PARAMETER_HASH == M8_PARAMETER_HASH
    # The four-parameter fit set (plan lines 770-775)
    assert driver.FIT_SET == FIT_SET
    assert driver.FIT_SET_EXCLUDED == {
        "blend_ramp_start": "obligation 4 (fixed at B = "
        "2.680194110727528 GeV; decision card lines 305-310)",
    }
    assert driver.INITIAL_VALUES == INITIAL_VALUES
    assert driver.PARAM_BOUNDS == PARAM_BOUNDS
    assert driver.BOUNDS_NOTE.startswith(
        "kappa_B and kappa_M within the R40 scan range [0, 4]")
    assert driver.W_GRID == W_GRID
    assert driver.W_GRID_NOMINAL == 0.5
    assert driver.W_GRID_INDEX_NOMINAL == 2
    assert driver.REWEIGHT_PARAMS == ("kappa_B", "kappa_m", "b_dec")
    # The optimizer contract (pre-registered, deterministic)
    assert driver.NM_XATOL == 1e-3
    assert driver.NM_FATOL == 1e-6
    assert driver.NM_MAXITER == 400
    assert driver.NM_MAXFEV == 800
    assert driver.HESSIAN_STEPS == {
        "kappa_B": 0.1, "kappa_m": 0.1, "b_dec": 0.1}
    # The pre-registered nuisance structure (plan section 5.4)
    assert driver.NUISSANCE_BLOCK_KEYS == (
        "pi_sigma_spectrum", "b_transport_spectrum", "seagulls",
        "multiplicity")
    assert driver.NORM_NUISANCE_PRIOR == 0.35
    assert driver.CARRY_NUISANCE_PRIOR == 0.25
    assert driver.ESS_FLOOR_FRACTION == 0.01
    assert driver.OBLIGATION4_R32_SPREAD == OBLIGATION4_R32_SPREAD
    assert driver.CARRY_SPECIES == ("pi_plus", "pi_minus", "proton")
    # The stability statistics
    assert driver.SIGMA_BAND == 3.0
    assert driver.STABILITY_SPREAD_BOUND == 0.20
    assert driver.EVENT_CLOSURE_TOL == 1e-9
    assert driver.CHUNK_SIZE == 2500
    assert driver.DEFAULT_WORKERS == 8
    assert driver.STAGE_LABEL == "r41_nonstrange_pilot_fit"
    # The sealed archives consumed read-only
    assert driver.SEALED_RUN_DIRS == (
        "dev/00runs/r40_fixed_source_alignment_scan_2026-09-21",
        "dev/00runs/r30_spectral_normalization_2026-09-17",
        "dev/00runs/r31_catalog_continuum_matching_scan_2026-09-17",
        "dev/00runs/r32_spectral_systematic_pp_2026-09-18",
        "dev/00runs/r32_spectral_systematic_pp_confirmation_2026-09-19",
    )
    assert driver.R01_RUN_DIR == (
        "dev/00runs/r01_legacy_pp_reference_2026-09-11")
    assert driver.R40_RUN_DIR == (
        "dev/00runs/r40_fixed_source_alignment_scan_2026-09-21")
    assert driver.R4_AUDIT_CARD1_DIR == (
        "dev/00runs/r4_audit_transition_routing_formulation_2026-09-20")
    # The frozen substrate table
    assert driver.FROZEN_SUBSTRATE == SUBSTRATE


def test_seed_map_partition(driver):
    """The master seed, the raw block 20301020-20301027, the
    event-index partition 400_000_000 + i, the kernel seed rule, the
    hermetic smoke block (+8000), and the seed functions' refusal
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
        "derive_substreams(20260920, r, 400_000_000 + i)")
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
    # The hermetic smoke block (+8000; never sealed)
    assert driver.SMOKE_REPLICAS == 1
    assert driver.SMOKE_EVENTS == 2_000
    assert driver.SMOKE_SEED_STRIDE == SMOKE_KERNEL_REPLICA_SLOT
    assert driver.SMOKE_RAW_SEED == SMOKE_RAW_SEED
    assert driver.smoke_raw_seed() == SMOKE_RAW_SEED
    assert driver.SMOKE_W_GRID == (0.5,)
    assert driver.SMOKE_SYNTHETIC_TRUTH == {
        "kappa_B": 1.3,
        "kappa_m": 0.8,
        "b_dec": 1.2,
        "baryonic_nominal_width": 0.5,
    }
    # The clean-environment validation's pre-registered probe replica
    assert driver.PROBE_REPLICA == 0


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
        "--pre-registration", "--smoke", "--generate", "--fit", "--gate",
        "--seal", "--clean-validate",
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
    the pre-registration-vs-card drift probes (fit-set, statistics,
    seed, width-grid, and written_before_generation drift are
    refused; the smoke mode is exempt; the sealed pre-registration
    passes)."""
    # The ordering invariant: pre-registration refuses a non-empty run
    # dir before verifying anything (the run dir must be fresh).
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        probe_dir = Path(tmp) / "r41_ordering_probe"
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
        lambda p: p["fit_set"]["parameters"].pop(),
        "fit set")
    _refused(
        lambda p: p["fit_set"].__setitem__("count", 3),
        "fit-set count")
    _refused(
        lambda p: p["fit_set"]["excluded"].popitem(),
        "blend ramp start")
    _refused(
        lambda p: p["statistics"].__setitem__("replicas", 4),
        "statistics")
    _refused(
        lambda p: p["seed_map"]["raw_replica_seeds"].__setitem__(
            0, 20301021 + 1),
        "seeds")
    _refused(
        lambda p: p["seed_map"].__setitem__(
            "event_index_rule", "300_000_000 + i, i < 100,000"),
        "event-index rule")
    _refused(
        lambda p: p["fit_set"].__setitem__(
            "width_grid", [0.3, 0.4, 0.5, 0.6]),
        "width grid")
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
    smoke["fit_set"] = {"parameters": ["placeholder"], "count": 1}
    driver._assert_pre_registration(smoke, RUN_DIR)


# ----------------------------------------------------------------------
# The sealed pre-registration
# ----------------------------------------------------------------------
def test_pre_registration_on_disk(driver, pre_registration):
    """The pre-registration's identity, T2 statistics, seed map, fit
    set, collision configuration, conservation and observables blocks,
    likelihood form, hessian discipline, acceptance check set with its
    register triggers, blend-band-sensitive bins, obligation 4 carry,
    substrate verification, and sealed run dirs."""
    assert pre_registration["schema"] == (
        "na49_mixing.staged_r41_nonstrange_pilot_fit.pre_registration/1")
    assert pre_registration["run"] == "R41"
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
        "derive_substreams(20260920, r, 400_000_000 + i)")
    assert seeds["raw_seed_rule"].startswith(
        "raw replica seed = 20260920 + 40000 + 100 + r = 20301020 + r")
    assert "20301040 + k" in seeds["disjointness"]
    assert seeds["smoke"]["raw_seed"] == SMOKE_RAW_SEED
    assert seeds["smoke"]["kernel_replica_slot"] == (
        SMOKE_KERNEL_REPLICA_SLOT)
    # The pre-registered fit set
    fit = pre_registration["fit_set"]
    assert fit["count"] == 4
    assert fit["parameters"] == list(FIT_SET)
    assert "blend_ramp_start" in fit["excluded"]
    assert fit["initial_values"] == INITIAL_VALUES
    assert {k: tuple(v) for k, v in fit["parameter_bounds"].items()} == (
        PARAM_BOUNDS)
    assert [float(w) for w in fit["width_grid"]] == list(W_GRID)
    assert fit["reweightable"] == list(driver.REWEIGHT_PARAMS)
    assert fit["r40_initial_values"].startswith(
        "the nominal point (kappa_B, kappa_m, b_dec) = (1, 1, 1.0")
    assert "scan_report.json" in fit["r40_gradient_information"]["source"]
    # The collision configuration (round 4 index item 7)
    collision = pre_registration["collision"]
    assert collision["index_item"] == 7
    assert collision["w_flavor"] == 1.0
    assert collision["system"].startswith(
        "p+p at sqrt(s) = 17.270 GeV (beam 158 GeV/c)")
    assert "358989d8" in collision["configuration"]
    assert pre_registration["conservation"]["per_event"].startswith(
        "every p+p event conserves total (B, Q, S) = (2, 2, 0)")
    # The observables and likelihood
    assert pre_registration["observables"]["training"].startswith(
        "the pooled combinations of plan section 7.3")
    assert "V_data" in pre_registration["likelihood"]["form"]
    assert pre_registration["likelihood"]["nuisances"]["carry_prior"] == 0.25
    assert pre_registration["likelihood"]["profile_is_not_fit_set"].startswith(
        "the nuisance profile is part of the objective definition")
    # The hessian discipline (plan section 5.4 line 311)
    assert pre_registration["hessian_discipline"]["steps"] == {
        "kappa_B": 0.1, "kappa_m": 0.1, "b_dec": 0.1}
    # The determinism block
    assert pre_registration["determinism"]["probe_replica"]["replica"] == 0
    # The optimizer contract
    opt = pre_registration["optimizer"]
    assert opt["algorithm"] == "scipy.optimize.minimize, method Nelder-Mead"
    assert opt["xatol"] == 0.001
    assert opt["fatol"] == 1e-06
    assert opt["maxiter"] == 400
    assert opt["maxfev"] == 800
    assert opt["optimizer_uses"] == 1
    assert opt["start_point"] == {
        "kappa_B": 1.0, "kappa_m": 1.0, "b_dec": 1.0}
    # The acceptance check set (plan section 24; index item 11)
    acceptance = pre_registration["acceptance"]
    assert set(acceptance) == {
        "i_fit_convergence", "ii_replica_block_stability",
        "iii_conservation", "iv_determinism", "named_residual_register",
        "precedent", "register_triggers", "v_no_refit_beyond_fit_set",
        "vi_seal_and_clean_validation", "vii_pins_and_freeze"}
    assert acceptance["register_triggers"] == [
        "any acceptance (ii) stability failure (per parameter)",
        "a pooled optimum on any pre-registered parameter bound "
        "(a boundary finding: the fit range does not extend)",
        "an effective sample size fraction below 0.01 at the pooled "
        "optimum (the weight-degeneracy diagnosis)",
        "a convergence failure of any pre-registered optimizer run "
        "(recorded; a convergence failure is a valid run outcome)",
        "the width argmin on the grid edge (the width is "
        "edge-constrained by the pre-registered bracket)",
    ]
    # The blend-band-sensitive bins (the sealed R32 persisting
    # residuals)
    bins = pre_registration["blend_band_sensitive_bins"]
    assert bins["entries"] == 32
    assert bins["sigma_range"][0] == 2.0001998039779263
    assert bins["sigma_range"][1] == 7.532929574244998
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
    # The frozen substrate digests, verified on disk
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
    # The plan anchors
    assert pre_registration["plan_anchors"]["fit_set"] == (
        "plan lines 770-775")
    assert pre_registration["plan_anchors"]["run"] == (
        "plan section 10 run R41, line 768")
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


# ----------------------------------------------------------------------
# The sealed fit report
# ----------------------------------------------------------------------
def test_fit_report_on_disk(driver, fit_report):
    """The report's schema, the internal digest, the boundary-
    constrained optimum on all four parameters, the eight per-block
    fits (interior per-block optima), the optimizer profile on the
    five-point width grid (the argmin on the grid edge), the
    replica-block stability table, the not-quoted Hessian block, the
    eight-entry named-residual register, and the pooled objective at
    the optimum."""
    assert fit_report["schema"] == (
        "na49_mixing.staged_r41_nonstrange_pilot_fit.report/1")
    assert fit_report["run"] == "R41"
    assert fit_report["card"] == SOLVER_CARD
    assert fit_report["mode"] == "production"
    assert fit_report["digest"] == FIT_REPORT_DIGEST
    assert fit_report["fit_set"] == list(FIT_SET)
    # The boundary-constrained optimum on all four parameters (the
    # pre-registered bounds; the width on the grid edge)
    assert fit_report["fitted"] == {
        "kappa_B": 0.0,
        "kappa_m": 4.0,
        "b_dec": 2.0,
        "baryonic_nominal_width": 0.3,
    }
    assert fit_report["initial_values"] == INITIAL_VALUES
    assert fit_report["no_refit"].startswith(
        "the fit set is exactly the four pre-registered parameters")
    # The eight per-block fits (converged; interior per-block optima,
    # so the block minima do not agree)
    blocks = fit_report["block_fits"]
    assert len(blocks) == 8
    assert [b["replica"] for b in blocks] == list(range(8))
    for block in blocks:
        assert block["converged"] is True
    assert [(b["kappa_B"], b["kappa_m"], b["b_dec"]) for b in blocks] == [
        (2.4709417482237743, 3.5262887584954132, 2.0),
        (2.2071674583431227, 3.274944444978746, 2.0),
        (2.5108368958876808, 3.4510621429700854, 2.0),
        (2.1609081935552847, 2.877649249551528, 2.0),
        (2.4478691496566576, 3.285665674757591, 1.3725886180966542),
        (2.5461927288175845, 3.723073080376501, 0.0),
        (2.5751021792725743, 3.522174020184462, 1.7662013322907453),
        (2.4744959110380567, 3.500604455880298, 1.3401573272010112),
    ]
    assert [b["nfev"] for b in blocks] == [225, 299, 260, 257, 239, 199,
                                           261, 272]
    # The optimizer contract (one fit procedure per objective)
    assert fit_report["optimizer_invoked"] is True
    assert fit_report["optimizer_uses"] == 1
    opt = fit_report["optimizer"]
    assert opt["invoked"] is True
    assert opt["converged"] is True
    assert opt["uses"] == 1
    assert opt["width_grid"] == list(W_GRID)
    assert [p["width_gev"] for p in opt["profile"]] == list(W_GRID)
    for entry in opt["profile"]:
        assert entry["converged"] is True
        assert entry["kappa_B"] == 0.0
        assert entry["kappa_m"] == 4.0
        assert entry["b_dec"] == 2.0
    assert [p["minus2lnl"] for p in opt["profile"]] == [
        7866.301937844129, 8615.312536512807, 9477.442498527183,
        9583.528869393678, 9562.180854847275]
    assert [p["nfev"] for p in opt["profile"]] == [41, 41, 41, 41, 41]
    # The width profile (the grid argmin on the pre-registered lower
    # edge; the width argmin register trigger)
    assert [p["width_gev"] for p in fit_report["width_profile"]] == (
        list(W_GRID))
    assert fit_report["width_profile"] == opt["profile"]
    # The replica-block stability (acceptance (ii)): kappa_B and
    # kappa_m fail the pre-registered 20% pooled-error bound; b_dec
    # passes degenerate (the per-block values span the bound, so the
    # pooled error is infinite and the bound degenerates with it)
    stability = fit_report["replica_block_stability"]
    assert set(stability) == {"kappa_B", "kappa_m", "b_dec"}
    assert stability["kappa_B"]["pass"] is False
    assert stability["kappa_B"]["spread"] == 0.4141939857172896
    assert stability["kappa_B"]["pooled_error"] == 0.021207663136679735
    assert stability["kappa_B"]["bound"] == 0.004241532627335947
    assert stability["kappa_B"]["per_block"] == [
        2.4709417482237743, 2.2071674583431227, 2.5108368958876808,
        2.1609081935552847, 2.4478691496566576, 2.5461927288175845,
        2.5751021792725743, 2.4744959110380567]
    assert stability["kappa_m"]["pass"] is False
    assert stability["kappa_m"]["spread"] == 0.8454238308249731
    assert stability["kappa_m"]["pooled_error"] == 0.007873938251432193
    assert stability["kappa_m"]["bound"] == 0.0015747876502864386
    assert stability["kappa_m"]["per_block"] == [
        3.5262887584954132, 3.274944444978746, 3.4510621429700854,
        2.877649249551528, 3.285665674757591, 3.723073080376501,
        3.522174020184462, 3.500604455880298]
    assert stability["b_dec"]["pass"] is True
    assert stability["b_dec"]["spread"] == 2.0
    assert math.isinf(stability["b_dec"]["pooled_error"])
    assert math.isinf(stability["b_dec"]["bound"])
    assert stability["b_dec"]["per_block"] == [
        2.0, 2.0, 2.0, 2.0, 1.3725886180966542, 0.0, 1.7662013322907453,
        1.3401573272010112]
    # The Hessian block (not quoted: the replica-block minima do not
    # agree; not positive definite; the width curvature undefined on
    # the grid)
    hessian = fit_report["hessian"]
    assert hessian["quoted"] is False
    assert hessian["positive_definite"] is False
    assert hessian["minima_agree"] is False
    assert hessian["eigenvalues"] == [
        -351.5285821457218, 1719.1749778337683, 16962.558520448147]
    assert math.isnan(hessian["width_curvature"])
    assert hessian["steps"] == {"kappa_B": 0.1, "kappa_m": 0.1,
                                "b_dec": 0.1}
    # The eight-entry named-residual register (the two acceptance (ii)
    # stability failures, the four boundary optima, the not-quoted
    # Hessian, and the weight degeneracy)
    register = fit_report["named_residual_register"]
    assert len(register) == 8
    assert [entry["index"] for entry in register] == [1, 2, 3, 4, 5, 6,
                                                      7, 8]
    assert [entry["kind"] for entry in register] == [
        "replica_block_stability", "replica_block_stability",
        "boundary_optimum", "boundary_optimum", "boundary_optimum",
        "boundary_optimum", "hessian_not_quoted", "weight_degeneracy"]
    assert register[0]["parameter"] == "kappa_B"
    assert register[0]["spread"] == 0.4141939857172896
    assert register[0]["pooled_error"] == 0.021207663136679735
    assert register[0]["bound"] == 0.004241532627335947
    assert register[0]["per_block"] == stability["kappa_B"]["per_block"]
    assert register[1]["parameter"] == "kappa_m"
    assert register[1]["spread"] == 0.8454238308249731
    assert register[1]["pooled_error"] == 0.007873938251432193
    assert register[1]["bound"] == 0.0015747876502864386
    assert register[1]["per_block"] == stability["kappa_m"]["per_block"]
    assert register[2]["parameter"] == "kappa_B"
    assert register[2]["observed"] == 0.0
    assert register[2]["bound"] == [0.0, 4.0]
    assert register[3]["parameter"] == "kappa_m"
    assert register[3]["observed"] == 4.0
    assert register[3]["bound"] == [0.0, 4.0]
    assert register[4]["parameter"] == "b_dec"
    assert register[4]["observed"] == 2.0
    assert register[4]["bound"] == [0.0, 2.0]
    assert register[5]["parameter"] == "baryonic_nominal_width"
    assert register[5]["observed"] == 0.3
    assert register[5]["bound"] == [0.3, 0.7]
    assert register[6]["observed"]["minima_agree"] is False
    assert register[6]["observed"]["positive_definite"] is False
    assert math.isnan(register[6]["observed"]["width_curvature"])
    assert register[7]["kind"] == "weight_degeneracy"
    assert register[7]["observed"] == 0.0058986913530268146
    assert register[7]["floor"] == 0.01
    for entry in register:
        assert entry["robust_statistics_cross_reference"].startswith(
            "the between-replica standard error")
    # The pooled objective at the optimum
    objective = fit_report["objective_at_optimum"]
    assert objective["ess"]["min"] == 0.0058986913530268146
    assert objective["ess"]["per_block"][3] == 0.0058986913530268146
    assert objective["ess"]["per_block"] == [
        0.019534286468347668, 0.011696349209294367, 0.016284341455345214,
        0.0058986913530268146, 0.019929059199401522, 0.012834833223055793,
        0.015036463935381265, 0.014492634260263117]
    assert objective["n_eff_total"] == 2166006.3538750205
    assert objective["minus2lnl_profiled"] == 7866.301937844129
    assert objective["minus2lnl_profiled"] == (
        opt["profile"][0]["minus2lnl"])
    assert objective["nuisance_columns"] == [
        "norm_pi_sigma_spectrum", "norm_b_transport_spectrum",
        "norm_multiplicity", "carry"]
    assert len(objective["nuisances"]) == 4
    # The model-uncertainty carry (obligation 4)
    carry = fit_report["model_uncertainty_carry"]
    assert carry["obligation_4"] is True
    assert carry["carry_prior"] == 0.25
    assert carry["r32_spread"] == OBLIGATION4_R32_SPREAD
    # The dropped bins and the determinism note
    assert fit_report["dropped_bins"] == []
    assert fit_report["n_dropped_bins"] == 0
    assert fit_report["determinism"]["report_note"].startswith(
        "no wall-clock fields")
    # The report's on-disk bytes at the pinned digest
    assert _sha256_file(RUN_DIR / "fit_report.json") == (
        SEALED_FILE_DIGESTS["fit_report.json"])


def test_gate4_evidence(driver, gate_record, fit_report):
    """The gate record's identity, its conservation summary, the
    fit-report digest cross-reference, and the equality of its
    stability, Hessian, carry, and named-residual blocks with the
    report's (the width_curvature NaN is not self-equal under == and
    is asserted separately)."""
    assert gate_record["schema"] == (
        "na49_mixing.staged_r41_nonstrange_pilot_fit.gate/1")
    assert gate_record["run"] == "R41"
    assert gate_record["card"] == SOLVER_CARD
    assert gate_record["gate"] == "gate 4 (plan section 10, line 806)"
    assert gate_record["clause"] == "unconstrained"
    assert gate_record["sealed"] is True
    assert gate_record["fit_set"] == list(FIT_SET)
    # The conservation summary (zero violations over 800,000 events)
    assert gate_record["events_checked"] == 800_000
    conservation = gate_record["conservation"]
    assert conservation["violations"] == 0
    assert conservation["pass"] is True
    assert conservation["tolerance"] == 1e-09
    assert conservation["events_checked"] == 800_000
    assert conservation["worst_energy_closure_gev"] == 3.446132268436486e-13
    # The fit-report cross-references
    assert gate_record["fit_report_digest"] == FIT_REPORT_DIGEST
    assert gate_record["fit_report_digest"] == fit_report["digest"]
    assert gate_record["fitted"] == fit_report["fitted"]
    assert gate_record["optimizer_invoked"] is True
    assert gate_record["optimizer_uses"] == 1
    assert gate_record["excluded"] == driver.FIT_SET_EXCLUDED
    # The equality blocks (the width_curvature NaN is not self-equal
    # under ==, so it is asserted separately)
    assert gate_record["replica_block_stability"] == (
        fit_report["replica_block_stability"])
    assert _nan_equal(gate_record["hessian"], fit_report["hessian"])
    assert math.isnan(gate_record["hessian"]["width_curvature"])
    assert math.isnan(fit_report["hessian"]["width_curvature"])
    # The gate's carry block is the report's carry block minus the
    # carry_prior (the prior is a report-level field)
    assert gate_record["model_uncertainty_carry"] == {
        k: v for k, v in fit_report["model_uncertainty_carry"].items()
        if k != "carry_prior"}
    assert _nan_equal(gate_record["named_residual_register"],
                      fit_report["named_residual_register"])
    assert gate_record["no_fitting_note"].startswith(
        "the fit set is the card's pre-registered four-parameter pilot "
        "fit")
    assert gate_record["sealed_note"].startswith(
        "the run seals after this record")


# ----------------------------------------------------------------------
# The seal manifest and the checksum file
# ----------------------------------------------------------------------
def test_seal_manifest_and_checksums(seal_manifest):
    """The seal manifest's 13 sealed files (8 replica records plus the
    five top-level files) path-sorted, the checksum file's 13 entries,
    and the full re-hash of all 13 sealed files against the checksum
    digests."""
    assert seal_manifest["schema"] == (
        "na49_mixing.staged_r41_nonstrange_pilot_fit.seal/1")
    assert seal_manifest["run"] == "R41"
    assert seal_manifest["card"] == SOLVER_CARD
    assert seal_manifest["record_count"] == 8
    assert seal_manifest["sha256sums_entries"] == 13
    assert seal_manifest["files"] == list(sorted(SEALED_FILES))
    assert len(seal_manifest["files"]) == 13
    assert seal_manifest["sealed_utc"].startswith("2026-09-23")
    assert seal_manifest["run_dir"].endswith(
        "dev/00runs/r41_nonstrange_pilot_fit_2026-09-23")
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
def test_clean_environment_record(clean_environment, fit_report):
    """The clean-environment validation's command list, the
    byte-identical report re-hash (digest equality), the probe replica
    record, and the pending seal state."""
    assert clean_environment["schema"] == (
        "na49_mixing.staged_r41_nonstrange_pilot_fit.clean_env/1")
    assert clean_environment["run"] == "R41"
    assert clean_environment["card"] == SOLVER_CARD
    commands = clean_environment["commands"]
    assert len(commands) == 3
    assert commands[0] == (
        "cd dev/mixing2006 && python -m pytest "
        "tests/test_staged_r41_nonstrange_pilot_fit.py -q")
    assert commands[1] == (
        "python -m pytest tests/test_legacy_v1_freeze.py "
        "-W error::UserWarning -q")
    assert commands[2].startswith(
        "python helpers/benchmarking/staged_r41_nonstrange_pilot_fit.py "
        "--clean-validate --run-dir ")
    assert "r41_nonstrange_pilot_fit_2026-09-23" in commands[2]
    # The byte-identical report re-hash
    assert clean_environment["report_hash_match"] is True
    assert clean_environment["fit_report_digest"] == FIT_REPORT_DIGEST
    assert clean_environment["fit_report_rerun_digest"] == (
        FIT_REPORT_DIGEST)
    assert clean_environment["fit_report_digest"] == fit_report["digest"]
    # The probe replica (setting-independent full replica re-run)
    probe = clean_environment["probe_replica"]
    assert probe["replica"] == 0
    assert probe["matches_sealed_record"] is True
    # The pending seal state (the validation runs before the seal)
    assert clean_environment["seal"]["state"].startswith("pending")
    assert clean_environment["seal"]["entries"] == 0
    assert clean_environment["seal"]["mismatches"] == []
    assert clean_environment["verified_utc"].startswith("2026-09-23")


# ----------------------------------------------------------------------
# The provenance
# ----------------------------------------------------------------------
def test_provenance_on_disk(provenance, driver):
    """The provenance's schema, the substrate digests, the M8 frozen
    set hash, the recorded driver digest (equal to the committed
    dual-tree digest: the relitigation pre-registered against the
    committed corrected driver), the superseded archive's verified
    seal state as the supersession reference, and the sealed run dir
    verification."""
    assert provenance["schema"] == (
        "na49_mixing.staged_r41_nonstrange_pilot_fit.provenance/1")
    assert provenance["run"] == "R41"
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
    assert provenance["m8_frozen_set_hash"] == M8_PARAMETER_HASH
    assert provenance["cascade_module_sha256"] == {
        name: SUBSTRATE[name][1]
        for name in ("cascade_decay", "cascade_axes", "cascade_alignment",
                     "cascade_form_factor", "cascade_termination",
                     "cascade_ablation")
    }
    # The recorded driver digest: the relitigation pre-registered
    # against the committed corrected driver, so the recorded digest
    # equals the committed dual-tree digest (unlike the card-12
    # landing, where the driver was amended after the
    # pre-registration)
    assert provenance["new_driver_sha256_both_trees"] == (
        PROVENANCE_DRIVER_SHA256)
    assert provenance["new_driver_sha256_both_trees"] == DRIVER_SHA256
    assert provenance["driver_note"].startswith(
        "byte-identical in the nested tree")
    assert provenance["r40_initial_values"].startswith(
        "the sealed R40 scan archive")
    # The superseded archive (the card-12 seal, intact and verified as
    # the supersession reference)
    superseded = provenance["superseded_r41_archive"]
    assert superseded["run_dir"] == SUPERSEDED_RUN_DIR_NAME
    assert superseded["seal"] == {"entries": 13, "mismatches": 0,
                                  "verified": True}
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
        "dev/00runs/r40_fixed_source_alignment_scan_2026-09-21":
            {"entries": 364, "mismatches": 0, "verified": True},
    }
    assert provenance["written_utc"] == "2026-09-23T18:33:54Z"
    # The pre-registration carries the same pre-registration-time
    # driver digest
    # (checked in test_pre_registration_on_disk's provenance block via
    #  the sealed JSON below)
    pre = json.loads(
        (RUN_DIR / "pre_registration.json").read_text(encoding="utf-8"))
    assert pre["provenance"]["driver_sha256"] == (
        PROVENANCE_DRIVER_SHA256)


# ----------------------------------------------------------------------
# The frozen substrate, byte-stable on disk
# ----------------------------------------------------------------------
def test_frozen_substrate_stability(driver):
    """The driver's fourteen frozen substrate files byte-stable on
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
    in the freeze manifest inventory, with the card-12 solver freeze
    exception intact (the four path-sorted files)."""
    solver_path = MONO_ROOT / SOLVER_TEST_REL
    assert solver_path.is_file(), f"missing solver test: {SOLVER_TEST_REL}"
    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)
    inventory = manifest["source_tree_sha256"]
    assert inventory[SOLVER_TEST_REL] == SOLVER_TEST_SHA256, (
        "the freeze manifest does not pin the solver test at its "
        "committed digest")
    # The card-12 solver freeze exception stays untouched: exactly one
    # entry naming the solver's card, at the solver's 4 path-sorted
    # files
    exceptions = manifest["freeze_exceptions"]
    solver_entries = [
        e for e in exceptions if e.get("card") == SOLVER_CARD]
    assert len(solver_entries) == 1, (
        "the card-12 solver freeze exception must name its card once")
    solver_files = solver_entries[0]["files"]
    assert solver_files == sorted(solver_files), (
        "the card-12 solver freeze-exception files are path-sorted")
    assert solver_files == list(SOLVER_EXCEPTION_FILES), (
        "the card-12 solver freeze-exception files drifted")


# ----------------------------------------------------------------------
# The freeze re-pin register (the test-author re-pin)
# ----------------------------------------------------------------------
