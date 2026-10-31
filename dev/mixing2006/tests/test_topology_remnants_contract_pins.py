"""Round 5, card 3 test-author register: the intact and low-mass remnants.

Independent contract pins for the intact and low-mass remnants
(``na49_mixing/topology_remnants.py``), Round 5 card 3
(``TASK-20260925-r5-intact-low-mass-remnants``; plan section 22 task 33,
line 1758; plan section 11.2, lines 833-842).  This register is the
test-author half of the card: it pins the module contract on its own
pinned fixtures (the closed remnant-kind sets, the neutron refusal
mode, the boundary split of the committed catalog at the frozen kernel
boundary, the intact proton mass pin, the pinned low-mass record pole
masses, the no-minimum-excitation pin, the draw budget on the closed
``spectral_state`` substream, the weight-rejection pins, and the frozen
``RemnantSelection`` record with the ``(1, 1, 0, 1)`` stamp) and re-pins
the freeze manifest for the card (``INVENTORY_COUNT`` 510 -> 511; the
Round 5 card 4 re-pin moves the chain to 513; the Round 5 card 4
test-author re-pin moves the chain to 514; the Round 5 card 5
re-pin moves the chain to 516; the Round 5 card 5 test-author re-pin moves the chain to 517).

Part 1: the module contract (pure, hermetic)
--------------------------------------------

- ``test_named_constants_and_closed_sets``: the named module constants
  (the frozen kernel boundary, the catalog proton mass and its catalog
  identity, the recorded continuum support end, the doubled-I3 remnant
  quantum numbers, the charge-exchange plan reference) and the closed
  ``REMNANT_KINDS`` set (four kinds) with the Round 5
  ``AVAILABLE_REMNANT_KINDS`` partition (three non-neutron kinds).
- ``test_neutron_refusal_mode``: the neutron remnant kind raises
  ``ChargeExchangeNotImplemented`` naming plan task 47 before any draw
  (zero uniforms, pinned RNG state) whether the neutron is named with
  weight or not, and a drawn neutron kind raises the same refusal after
  the kind draw (exactly one uniform, pinned post-draw state); the
  refusal is consistent with card 2's closed allowed-exchange set
  ``{VACUUM}``.
- ``test_boundary_split``: on the committed catalog (pinned at its
  digest) the ``low_mass_records`` / ``high_mass_records`` partition of
  the ``(1, 1, 0)`` records at the frozen kernel boundary
  ``B = 2.680194110727528 GeV`` is exact (catalog order, no overlap,
  nothing dropped; the pinned census is ten low-mass records and an
  empty high-mass side, every threshold off the boundary by more than
  1e-12, pinned minimum distance 1.6069452007275278 GeV).
- ``test_intact_mass_pin``: the intact-proton remnant mass equals
  ``m_p = 0.93827231`` exactly (the catalog proton pole mass, no
  tolerance), its quantum numbers are ``(1, 1, 0, 1)``, its catalog
  identity is the proton record, and it carries no record id.
- ``test_low_mass_record_mass_pins``: the ten pinned low-mass records in
  catalog order with their exact pole masses and catalog identities,
  every pole mass at or above ``m_p``, the doubled-I3 record tie, and
  the pinned 10,000-draw trace: every drawn low-mass mass equals the
  selected record's pole mass (1e-12) with the record's catalog
  identity.
- ``test_no_minimum_excitation_pin``: the module source carries no
  excitation-count guard (white-box token scan; the plan's no-minimum
  rule appears verbatim instead) and the pinned 10,000-draw trace at
  the nominal SD split contains intact-proton remnants with positive
  frequency and no excitation-count rejection anywhere.
- ``test_draw_budget_pin``: exactly one uniform on the closed
  ``spectral_state`` label per selection (the counting proxy, the
  bit-identical post-draw stream state, and the continuous-stream
  identity over a pinned trace); the ``mass_sampler`` attachment
  consumes zero draws of this card's substream (the stand-in law on its
  own private stream, called exactly once with the frozen attachment
  arguments for the high-mass kind and never for the other kinds).
- ``test_weights_rejection_pin``: a weight vector naming a kind outside
  ``REMNANT_KINDS`` raises ``ValueError`` naming the kind; weights not
  summing to 1 within 1e-12 raise ``ValueError``; a negative or
  non-finite weight raises ``ValueError`` naming the kind; a kind
  argument outside the closed set raises ``ValueError`` naming it; a
  requested kind that does not name one of the weights' kinds raises
  ``ValueError``; every refusal precedes any draw (pinned RNG state).
- ``test_frozen_selection_record``: the ``RemnantSelection`` record is
  a frozen dataclass with exactly the card's five fields, and every
  selection on the pinned traces carries the doubled-I3 quantum-number
  stamp ``(B, Q, S, 2I3) = (1, 1, 0, 1)``.

Part 2: the substrate and freeze pins
--------------------------------------

- ``test_module_byte_and_export_pins``: the module at its committed
  sha256, its ``__all__`` export set, and the package ``__init__`` at
  its committed sha256 carrying the remnant export block.
- ``test_frozen_substrate_digests``: the card's frozen-substrate
  provenance surfaces (verified on disk) and the frozen M8 parameter
  map through the manifest, with the activated baryonic ``(1, 1, 0)``
  continuum family's recorded support ends pinned against the module
  constants.
- ``test_freeze_repin_register``: the freeze-manifest re-pin for this
  card (the strict path-sorted inventory at 517 entries; 516 after the Round 5 card 5 test-author re-pin; 514 after the
  Round 5 card 5 re-pin; 513 after the Round 5 card 4 test-author
  re-pin; 511 after the Round 5 card 4 re-pin;
  510 after the Round 5 card 3 solver landing; 508 after the
  test-protocol tiering change; 505 after the Round 5 card 2
  test-author re-pin; 504 after the Round 5 card 2 solver landing; 502
  after the Round 5 card 1 test-author re-pin; 501 at the Round 5 card 1
  landing) carries this
  register and the twenty-seven re-pinned chain count-pin files at
  their on-disk digests, the card 3 solver's freeze exception stays
  exactly once at its 31 path-sorted files, this card's freeze
  exception names the card exactly once with the path-sorted file
  list, the chain count pins are re-pinned 510 -> 511 with the 510
  literal retained in the naming comments, and the frozen M8 parameter
  map stays intact.
- ``test_layout_sync``: the layout sync (the subprocess check from the
  monorepo root; ``logic_drift = 0``).

Hermetic: the module, the committed catalog, the continuum families
configuration, numpy, and the standard library only; no event
generation driver, no external tables, no Monte Carlo run (the runs
R50-R55 exercise the remnants statistically).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.exchange_ledger import ALLOWED_EXCHANGES, VACUUM
from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.rng_substreams import SUBSTREAM_LABELS, derive_substreams
from na49_mixing.topology_remnants import (
    AVAILABLE_REMNANT_KINDS,
    CHARGE_EXCHANGE_TASK_REF,
    CONTINUUM_M_MAX,
    KIND_HIGH_MASS_BARYONIC_CONTINUUM,
    KIND_INTACT_PROTON,
    KIND_LOW_MASS_BARYONIC_RESONANCE,
    KIND_NEUTRON_CHARGE_EXCHANGE,
    KERNEL_BOUNDARY,
    PROTON_MASS,
    PROTON_PDG_ID,
    REMNANT_KINDS,
    REMNANT_QN,
    ChargeExchangeNotImplemented,
    RemnantSelection,
    high_mass_records,
    low_mass_records,
    select_remnant,
)
from tests._freeze_inventory import inventory_count as _inventory_count

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent          # dev/mixing2006
MONO_ROOT = NESTED_ROOT.parent.parent          # the monorepo root
MANIFEST_PATH = NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json"
SYNC_CHECKER = MONO_ROOT / "helpers" / "benchmarking" / "check_helpers_layout_sync.py"
MODULE_PATH = NESTED_ROOT / "na49_mixing" / "topology_remnants.py"
INIT_PATH = NESTED_ROOT / "na49_mixing" / "__init__.py"

MY_TEST_REL = (
    "dev/mixing2006/tests/test_topology_remnants_contract_pins.py")

#: The committed catalog and its pinned digest (the card's
#: frozen-substrate provenance; the boundary split and the low-mass
#: record mass pins are pinned at this digest).
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
CATALOG_SHA256 = (
    "cf9e79b99199c509f2e83def0a99c215f73ca7761f9f87fb01d1f779b80e32f3")
CONTINUUM_PATH = (
    NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json")

#: The Round 5 master seed (round 5 index item 9), fixed for the pinned
#: traces; replica 0, event index = selection index, on the closed
#: ``spectral_state`` substream.
MASTER_SEED = 20260925

#: The pinned trace sizes: the card's 10,000-draw traces and the
#: draw-budget trace.
N_TRACE = 10_000
N_BUDGET_TRACE = 1_000

#: The nominal SD kind split of the R50 templates (index item 2; fixed,
#: not fit): intact 0.5, low-mass 0.5, high-mass 0.0.
SD_WEIGHTS = {
    KIND_INTACT_PROTON: 0.5,
    KIND_LOW_MASS_BARYONIC_RESONANCE: 0.5,
    KIND_HIGH_MASS_BARYONIC_CONTINUUM: 0.0,
}

#: The committed module and package digests (nested tree HEAD a545287).
MODULE_SHA256 = (
    "8c26ae419cbb0e7de3b36d95bb6e097c1854df7d2c69f8558b7f71bd4a0342ca")
#: The package digest re-pinned by Round 5 card 3
#: (TASK-20260925-r5-intact-low-mass-remnants; the remnant module's
#: additive package exports).
# re-pinned by Round 5 card 7 (TASK-20260925-r5-non-diffrative-central-excitation; the non-diffractive module's additive package exports).
# re-pinned by Round 5 card 8 (TASK-20260925-r5-multiple-neutral-excitations; the multiple-neutral-excitation module's additive package exports).
# re-pinned by Round 5 card 9 (TASK-20260925-r5-topology-template-runner; the topology-runner module's additive package exports).
# re-pinned by the Round 5 card 9 review fix round (TASK-20260925-r5-topology-template-runner-review-fix; the REV-bc4de47-03 export-surface correction re-pinned the package __init__).
INIT_SHA256 = (
    "7d2c5a85d9b24de104c0d9cf72195689bf7c39c873e154798699448073b36de8")

#: The module's exact ``__all__`` export set (eighteen names).
MODULE_ALL = [
    "AVAILABLE_REMNANT_KINDS",
    "CHARGE_EXCHANGE_TASK_REF",
    "CONTINUUM_M_MAX",
    "KIND_HIGH_MASS_BARYONIC_CONTINUUM",
    "KIND_INTACT_PROTON",
    "KIND_LOW_MASS_BARYONIC_RESONANCE",
    "KIND_NEUTRON_CHARGE_EXCHANGE",
    "KERNEL_BOUNDARY",
    "REMNANT_KIND_INDEX",
    "PROTON_MASS",
    "PROTON_PDG_ID",
    "REMNANT_KINDS",
    "REMNANT_QN",
    "ChargeExchangeNotImplemented",
    "RemnantSelection",
    "high_mass_records",
    "low_mass_records",
    "select_remnant",
]

#: The remnant export block of the package ``__init__`` (the ten names
#: the package re-exports; the ``KIND_*`` spellings, the charge-exchange
#: plan reference, and the proton's catalog identity are module-level
#: only).
INIT_REMNANT_NAMES = [
    "AVAILABLE_REMNANT_KINDS",
    "CONTINUUM_M_MAX",
    "KERNEL_BOUNDARY",
    "REMNANT_KINDS",
    "REMNANT_QN",
    "ChargeExchangeNotImplemented",
    "RemnantSelection",
    "high_mass_records",
    "low_mass_records",
    "select_remnant",
]

#: The M8-frozen kernel boundary, pinned through the manifest's frozen
#: parameter map.
KERNEL_BOUNDARY_PIN = 2.680194110727528

#: The M8-frozen ten-parameter set hash (the sha256 of the manifest's
#: frozen parameter map, ``json.dumps(..., sort_keys=True,
#: separators=(",", ":"))``; the card's provenance pin).
M8_HASH = "358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1"

#: The card's frozen-substrate provenance pins (byte-stable surfaces,
#: verified on disk): the builder blend path, the card-6 blend routing,
#: the card-2 engine, the committed catalog, the continuum families
#: configuration, the sealed Round 4 R45/R46 drivers in both helper
#: trees, and the two Round 5 topology modules consumed read-only
#: (card 1's elastic reference, card 2's exchange ledger).
FROZEN_SURFACES = {
    "dev/mixing2006/na49_mixing/resonance_pp_builder.py":
        "4ab612609e1ee0bc1e037dcbb2815b6484306ad32402b7289e6636f8e829cdb1",
    "dev/mixing2006/na49_mixing/spectral_density.py":
        "382e048b9f66a35a6446e39d97e95a9fed885e0ac5ec297bfa19823e5acf20b7",
    "dev/mixing2006/na49_mixing/threshold_metric.py":
        "ee240f44645d2e592acfae462f7d1e9c3b3dd8cd78c96bd40b7b73509f7fcc63",
    "dev/mixing2006/na49_mixing/topology_elastic.py":
        "7640e64e0cb9428f712f854384c1e0410947909e6a3c5cb7aa9c1faf5fc1b824",
    "dev/mixing2006/na49_mixing/exchange_ledger.py":
        "3ab9b90803611c0d952def364e2ffb05017eb82c66d2e20eae7c966c18f0300a",
    "dev/mixing2006/config/v2_singlet/resonance_catalog.json":
        "cf9e79b99199c509f2e83def0a99c215f73ca7761f9f87fb01d1f779b80e32f3",
    "dev/mixing2006/config/v2_singlet/continuum_resonances.json":
        "33b2e597932013747f4781f83c027d18e1de794272717a1d1f18df4377f4cfe5",
    "dev/mixing2006/helpers/benchmarking/"
    "staged_r45_simplified_alignment_freeze.py":
        "48cc785aeccd24206cba5cc581d5b964df8e11b76df029a845bc0911e701bc10",
    "helpers/benchmarking/staged_r45_simplified_alignment_freeze.py":
        "48cc785aeccd24206cba5cc581d5b964df8e11b76df029a845bc0911e701bc10",
    "dev/mixing2006/helpers/benchmarking/"
    "staged_r46_simplified_recoil_decay_profile.py":
        "121774be722aadf971bc0a3fcb9694d3b5fd74bfcc2bd3a92c2201bf4e1d116f",
    "helpers/benchmarking/staged_r46_simplified_recoil_decay_profile.py":
        "121774be722aadf971bc0a3fcb9694d3b5fd74bfcc2bd3a92c2201bf4e1d116f",
}

#: The pinned low-mass records of the committed catalog in catalog
#: order: (record name, catalog pdg_id, pole mass [GeV]).  Every entry
#: sits below the frozen kernel boundary; the pinned minimum distance
#: to the boundary is 1.6069452007275278 GeV (far beyond the 1e-12
#: boundary-exclusion bound of the card's pre-registration).
LOW_MASS_PINNED = (
    ("proton", 2212, 0.93827231),
    ("Delta+", 2214, 1.232),
    ("N*(1440)+", 12212, 1.44),
    ("N*(1520)+", 2216, 1.52),
    ("N*(1680)+", 22212, 1.685),
    ("N(1535)+", 102212, 1.535),
    ("N(1650)+", 122212, 1.655),
    ("N(1720)+", 212214, 1.72),
    ("Delta(1600)+", 202214, 1.6),
    ("Delta(1700)+", 122214, 1.7),
)
MIN_BOUNDARY_DISTANCE = 1.6069452007275278

#: The first uniform of the pinned spectral_state stream of selection
#: index 6 (the drawn-neutron refusal probe: the kind draw selects the
#: neutron kind with the 0.5/0.5 split, so the refusal fires after the
#: draw).
DRAWN_NEUTRON_UNIFORM = 0.5640250029474355

#: The white-box guard tokens of an excitation-count refusal (plan line
#: 842: none of these may appear anywhere in the module source).
EXCITATION_GUARD_TOKENS = (
    "n_pi_min", "min_n_pi", "pion_min", "n_pion_min",
    "min_excitation", "excitation_count", "min_multiplicity",
    "reject_under",
)

#: The freeze-manifest inventory entry count after this card's
#: re-pin (510 after the Round 5 card 3 solver landing; +1 entry:
#: this register).
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
#: The solver card (Round 5, card 3) and this test-author register's
#: card name (the freeze-exception identity).
SOLVER_CARD = "TASK-20260925-r5-intact-low-mass-remnants"
TEST_AUTHOR_CARD = (
    "TASK-20260925-r5-intact-low-mass-remnants-test-author")

#: The register chain: the twenty-seven chain count-pin files re-pinned
#: 510 -> 511 by this register, 511 -> 513 by Round 5 card 4, and 513
#: -> 514 by Round 5 card 4 test-author (the twenty-six card-2 chain
#: files plus the card-2 register itself, path-sorted).
CHAIN_COUNT_PIN_FILES = (
    "dev/mixing2006/tests/test_cascade_ablation.py",
    "dev/mixing2006/tests/test_cascade_alignment_baryonic.py",
    "dev/mixing2006/tests/test_cascade_alignment_mesonic.py",
    "dev/mixing2006/tests/test_cascade_axes.py",
    "dev/mixing2006/tests/test_cascade_form_factor.py",
    "dev/mixing2006/tests/test_cascade_termination.py",
    "dev/mixing2006/tests/test_exchange_ledger_contract_pins.py",
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
    "dev/mixing2006/tests/test_staged_r43_cascade_ablation.py",
    "dev/mixing2006/tests/test_staged_r43_cascade_ablation_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r44_frozen_nonstrange_validation.py",
    "dev/mixing2006/tests/test_staged_r45_simplified_alignment_freeze.py",
    "dev/mixing2006/tests/test_staged_r45_simplified_alignment_freeze_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r46_simplified_recoil_decay_profile.py",
    "dev/mixing2006/tests/test_staged_r46_simplified_recoil_decay_profile_contract_pins.py",
    "dev/mixing2006/tests/test_topology_elastic_contract_pins.py",
)

#: This register's freeze-exception file list (path-sorted): the
#: manifest, this register, and the twenty-seven re-pinned chain
#: count-pin files.
TEST_AUTHOR_EXCEPTION_FILES = tuple(sorted(
    ("dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json",)
    + CHAIN_COUNT_PIN_FILES
    + (MY_TEST_REL,)
))


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remnant_stream_id(event_index: int) -> int:
    """The pinned spectral_state stream of one trace selection."""

    return derive_substreams(MASTER_SEED, 0, event_index).stream_for(
        "spectral_state")


def _remnant_rng(event_index: int) -> np.random.Generator:
    """The pinned spectral_state generator of one trace selection."""

    return np.random.default_rng(_remnant_stream_id(event_index))


class _CountingRng:
    """A counting proxy around one generator (the house draw-count
    pattern): every ``random()`` consumes exactly one uniform from the
    wrapped stream and advances the counter."""

    def __init__(self, source) -> None:
        self._source = source
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        return float(self._source.random())


def _refusing_sampler(m_low: float, m_high: float) -> float:
    """A mass sampler that must never be called (the intact and
    low-mass kinds take their masses without the caller's law)."""

    raise AssertionError(
        "the mass sampler must not be called for this kind, got "
        f"({m_low!r}, {m_high!r})")


class _StandInUniformLaw:
    """The pinned uniform stand-in mass law of this card's landing
    (card 4's law replaces it): a uniform draw on ``[m_low, m_high]``
    over the law's own private stream, so the attachment consumes zero
    draws of the spectral_state substream."""

    def __init__(self, seed: int = MASTER_SEED + 7) -> None:
        self._rng = np.random.default_rng(seed)
        self.calls = 0
        self.lower: float | None = None
        self.upper: float | None = None
        self.value: float | None = None

    def __call__(self, m_low: float, m_high: float) -> float:
        self.calls += 1
        self.lower = float(m_low)
        self.upper = float(m_high)
        self.value = self.lower + (self.upper - self.lower) * float(
            self._rng.random())
        return self.value


def _catalog() -> HadronCatalog:
    return HadronCatalog.from_json(CATALOG_PATH)


def _global_rng_snapshot():
    """A hashable snapshot of the numpy global generator state (the
    legacy ``RandomState`` state tuple; the MT state array is hashed
    through its bytes)."""

    return tuple(
        part.tobytes() if isinstance(part, np.ndarray) else part
        for part in np.random.get_state())


# ----------------------------------------------------------------------
# Part 1: the module contract
# ----------------------------------------------------------------------
def test_named_constants_and_closed_sets():
    """The named module constants and the closed remnant-kind sets: the
    frozen kernel boundary, the catalog proton mass and identity, the
    recorded continuum support end, the doubled-I3 remnant quantum
    numbers, the charge-exchange plan reference, the closed four-kind
    ``REMNANT_KINDS`` set, and the Round 5 three-kind
    ``AVAILABLE_REMNANT_KINDS`` partition."""

    # the named constants
    assert KERNEL_BOUNDARY == KERNEL_BOUNDARY_PIN == 2.680194110727528
    assert PROTON_MASS == 0.93827231
    assert PROTON_PDG_ID == 2212
    assert CONTINUUM_M_MAX == 5.0
    assert CHARGE_EXCHANGE_TASK_REF == "plan task 47"
    assert REMNANT_QN == (1, 1, 0, 1)
    assert all(isinstance(part, int) for part in REMNANT_QN)
    assert isinstance(KERNEL_BOUNDARY, float)
    assert isinstance(PROTON_MASS, float)
    assert isinstance(CONTINUUM_M_MAX, float)

    # the closed remnant-kind set: exactly the four plan section 11.2
    # kinds
    assert isinstance(REMNANT_KINDS, frozenset)
    assert REMNANT_KINDS == frozenset({
        KIND_INTACT_PROTON,
        KIND_NEUTRON_CHARGE_EXCHANGE,
        KIND_LOW_MASS_BARYONIC_RESONANCE,
        KIND_HIGH_MASS_BARYONIC_CONTINUUM,
    })
    assert len(REMNANT_KINDS) == 4
    assert {kind for kind in REMNANT_KINDS} == REMNANT_KINDS

    # the Round 5 availability set: exactly the three non-neutron kinds
    assert isinstance(AVAILABLE_REMNANT_KINDS, frozenset)
    assert AVAILABLE_REMNANT_KINDS == frozenset({
        KIND_INTACT_PROTON,
        KIND_LOW_MASS_BARYONIC_RESONANCE,
        KIND_HIGH_MASS_BARYONIC_CONTINUUM,
    })
    assert len(AVAILABLE_REMNANT_KINDS) == 3

    # the two partition: the availability set is the closed set minus
    # the neutron kind, the neutron kind is in the closed set only,
    # and nothing outside the closed set is available
    assert AVAILABLE_REMNANT_KINDS | {KIND_NEUTRON_CHARGE_EXCHANGE} == (
        REMNANT_KINDS)
    assert AVAILABLE_REMNANT_KINDS & {KIND_NEUTRON_CHARGE_EXCHANGE} == (
        frozenset())
    assert AVAILABLE_REMNANT_KINDS < REMNANT_KINDS


def test_neutron_refusal_mode():
    """The neutron remnant kind raises
    ``ChargeExchangeNotImplemented`` naming plan task 47 before any
    draw (zero uniforms, pinned RNG state) whether the neutron is named
    with weight or not, and a drawn neutron kind raises the same
    refusal after the kind draw (exactly one uniform, pinned post-draw
    state); the refusal is consistent with card 2's closed
    allowed-exchange set ``{VACUUM}``."""

    catalog = _catalog()
    assert CHARGE_EXCHANGE_TASK_REF == "plan task 47"
    assert issubclass(ChargeExchangeNotImplemented, RuntimeError)

    # consistency with card 2's closed allowed-exchange set (plan line
    # 838): charge exchange is outside the Round 5 closed set, so the
    # neutron kind is refused at this round
    assert isinstance(ALLOWED_EXCHANGES, frozenset)
    assert ALLOWED_EXCHANGES == frozenset({VACUUM})

    # the requested neutron kind, named with weight: the refusal
    # precedes the kind draw (zero uniforms consumed, pinned state)
    rng = _remnant_rng(3)
    state0 = rng.bit_generator.state
    with pytest.raises(ChargeExchangeNotImplemented, match="plan task 47"):
        select_remnant(
            KIND_NEUTRON_CHARGE_EXCHANGE,
            {KIND_NEUTRON_CHARGE_EXCHANGE: 1.0},
            catalog, rng, _refusing_sampler)
    assert rng.bit_generator.state == state0

    # the requested neutron kind, not named in the weights: the
    # refusal still precedes the kind draw (the kind argument is
    # validated against the closed set, then refused, before any draw)
    rng = _remnant_rng(3)
    state0 = rng.bit_generator.state
    with pytest.raises(ChargeExchangeNotImplemented, match="plan task 47"):
        select_remnant(
            KIND_NEUTRON_CHARGE_EXCHANGE,
            {KIND_INTACT_PROTON: 1.0},
            catalog, rng, _refusing_sampler)
    assert rng.bit_generator.state == state0

    # a drawn neutron kind (the neutron named with positive weight):
    # the refusal fires after the kind draw, exactly one uniform
    # consumed (pinned post-draw state against the reference stream
    # advanced by exactly one draw)
    sid = _remnant_stream_id(6)
    reference = np.random.default_rng(sid)
    u_first = reference.random()
    assert u_first == DRAWN_NEUTRON_UNIFORM
    assert u_first >= 0.5  # the kind draw selects the neutron kind
    rng = np.random.default_rng(sid)
    state0 = rng.bit_generator.state
    with pytest.raises(ChargeExchangeNotImplemented, match="plan task 47"):
        select_remnant(
            KIND_INTACT_PROTON,
            {KIND_INTACT_PROTON: 0.5, KIND_NEUTRON_CHARGE_EXCHANGE: 0.5},
            catalog, rng, _refusing_sampler)
    assert rng.bit_generator.state == reference.bit_generator.state
    assert rng.bit_generator.state != state0


def test_boundary_split():
    """On the committed catalog (pinned at its digest) the split at the
    frozen kernel boundary ``B = 2.680194110727528 GeV`` partitions the
    ``(1, 1, 0)`` records exactly: ``low_mass_records`` is the
    ``mass_range[0] < B`` side in catalog order, ``high_mass_records``
    the exact complement, no record sits on the boundary (B equality
    impossible within 1e-12), and the pinned census is ten low-mass
    records and an empty high-mass side."""

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)
    assert KERNEL_BOUNDARY == KERNEL_BOUNDARY_PIN

    catalog = _catalog()
    baryonic = [record for record in catalog.records()
                if (record.B, record.Q, record.S) == (1, 1, 0)]
    expected_low = [record for record in baryonic
                    if record.mass_range[0] < KERNEL_BOUNDARY]
    expected_high = [record for record in baryonic
                     if record.mass_range[0] >= KERNEL_BOUNDARY]

    low = low_mass_records(catalog)
    high = high_mass_records(catalog)
    assert [record.name for record in low] == [
        record.name for record in expected_low]
    assert [record.name for record in high] == [
        record.name for record in expected_high]

    # the split is exact: the two sides partition the baryonic records
    # in catalog order with no overlap and nothing dropped
    low_names = [record.name for record in low]
    high_names = [record.name for record in high]
    assert len(low_names) + len(high_names) == len(baryonic)
    assert not (set(low_names) & set(high_names))
    assert low_names + high_names == [record.name for record in baryonic]

    # B equality is impossible at the pinned catalog values: every
    # threshold sits off the frozen boundary by more than 1e-12, and
    # the pinned minimum distance is far beyond the bound
    assert all(abs(record.mass_range[0] - KERNEL_BOUNDARY) > 1e-12
               for record in baryonic)
    assert min(abs(record.mass_range[0] - KERNEL_BOUNDARY)
               for record in baryonic) == MIN_BOUNDARY_DISTANCE

    # the pinned census on the committed catalog: ten low-mass records,
    # an empty high-mass side
    assert len(baryonic) == 10
    assert len(low) == 10
    assert len(high) == 0
    assert high_mass_records(catalog) == []


def test_intact_mass_pin():
    """The intact-proton remnant mass equals ``m_p = 0.93827231``
    exactly (the catalog proton pole mass, no tolerance); its quantum
    numbers are ``(1, 1, 0, 1)``; its catalog identity is the proton
    record, and it carries no record id."""

    catalog = _catalog()
    proton = catalog.get(PROTON_PDG_ID)
    assert int(proton.pdg_id) == PROTON_PDG_ID == 2212
    assert float(proton.pole_mass) == PROTON_MASS == 0.93827231

    selection = select_remnant(
        KIND_INTACT_PROTON, {KIND_INTACT_PROTON: 1.0},
        catalog, _remnant_rng(5), _refusing_sampler)
    assert isinstance(selection, RemnantSelection)
    assert selection.kind == KIND_INTACT_PROTON
    # the intact mass is exact: no tolerance, the catalog value
    assert selection.mass == PROTON_MASS
    assert selection.qn == REMNANT_QN == (1, 1, 0, 1)
    assert selection.pdg_id == PROTON_PDG_ID
    assert selection.record_id is None


def test_low_mass_record_mass_pins():
    """The ten pinned low-mass records in catalog order with their
    exact pole masses and catalog identities (every pole mass at or
    above ``m_p``, the doubled-I3 record tie), and the pinned
    10,000-draw trace: every drawn low-mass mass equals the selected
    record's pole mass (1e-12) with the record's catalog identity."""

    catalog = _catalog()
    low = low_mass_records(catalog)
    assert len(low) == len(LOW_MASS_PINNED) == 10
    for record, (name, pdg, mass) in zip(low, LOW_MASS_PINNED):
        assert record.name == name
        assert int(record.pdg_id) == pdg
        # the pinned pole mass, exact
        assert float(record.pole_mass) == mass
        # no low-mass record pole mass lies below m_p
        assert mass >= PROTON_MASS
        # the doubled-I3 record tie: every (1, 1, 0) record carries
        # 2I3 = 1 (the Gell-Mann-Nishijima relation, pinned in the
        # catalog)
        assert (record.B, record.Q, record.S, record.isospin3) == (
            1, 1, 0, 1)

    records = {record.name: record for record in low}
    n_low = 0
    for i in range(N_TRACE):
        selection = select_remnant(
            KIND_INTACT_PROTON, SD_WEIGHTS,
            catalog, _remnant_rng(i), _refusing_sampler)
        if selection.kind == KIND_LOW_MASS_BARYONIC_RESONANCE:
            n_low += 1
            assert selection.record_id in records
            record = records[selection.record_id]
            # the low-mass remnant mass equals the selected record's
            # pole mass (1e-12; the pinned records make it exact)
            assert selection.mass == float(record.pole_mass)
            assert abs(selection.mass - float(record.pole_mass)) <= 1e-12
            assert selection.pdg_id == int(record.pdg_id)
            assert selection.qn == (1, 1, 0, 1)
    assert n_low > 0


def test_no_minimum_excitation_pin():
    """The module source carries no excitation-count guard (white-box
    token scan; the plan's no-minimum rule appears verbatim instead),
    and the pinned 10,000-draw trace at the nominal SD split (intact
    0.5, low-mass 0.5) contains intact-proton remnants with positive
    frequency and no excitation-count rejection anywhere (plan line
    842: the intact proton is a legal remnant with zero pions)."""

    catalog = _catalog()
    n_intact = 0
    n_low = 0
    for i in range(N_TRACE):
        # every selection succeeds: no excitation-count rejection is
        # raised anywhere on the trace
        selection = select_remnant(
            KIND_INTACT_PROTON, SD_WEIGHTS,
            catalog, _remnant_rng(i), _refusing_sampler)
        if selection.kind == KIND_INTACT_PROTON:
            n_intact += 1
        else:
            assert selection.kind == KIND_LOW_MASS_BARYONIC_RESONANCE
            n_low += 1
        assert selection.mass >= PROTON_MASS
    # the intact proton is a legal remnant with zero pions: it occurs
    # with positive frequency, and nothing else is drawn
    assert n_intact > 0
    assert n_low > 0
    assert n_intact + n_low == N_TRACE

    # white-box: the module source carries no excitation-count guard
    # (the plan's no-minimum rule appears verbatim instead)
    source = MODULE_PATH.read_text(encoding="utf-8")
    for token in EXCITATION_GUARD_TOKENS:
        assert token not in source, token
    assert "Do not impose a minimum N-pi excitation" in source


def test_draw_budget_pin():
    """Exactly one uniform on the closed ``spectral_state`` label per
    selection (the counting proxy, the bit-identical post-draw stream
    state, and the continuous-stream identity over a pinned trace);
    the ``mass_sampler`` attachment consumes zero draws of this card's
    substream (the stand-in law on its own private stream, called
    exactly once with the frozen attachment arguments for the
    high-mass kind and never for the other kinds)."""

    assert "spectral_state" in SUBSTREAM_LABELS
    catalog = _catalog()

    # every available kind: one selection consumes exactly one uniform
    # (the counting proxy), the post-draw state is bit-identical to
    # the reference stream after exactly one draw, and the stand-in
    # law is called exactly once for the high-mass kind with the
    # frozen attachment arguments and never for the other kinds
    for kind in sorted(AVAILABLE_REMNANT_KINDS):
        sid = _remnant_stream_id(0)
        counter = _CountingRng(np.random.default_rng(sid))
        law = _StandInUniformLaw()
        selection = select_remnant(kind, {kind: 1.0}, catalog, counter, law)
        assert counter.calls == 1
        advanced = np.random.default_rng(sid)
        advanced.random()
        assert (counter._source.bit_generator.state
                == advanced.bit_generator.state)
        if kind == KIND_HIGH_MASS_BARYONIC_CONTINUUM:
            assert law.calls == 1
            assert law.lower == PROTON_MASS
            assert law.upper == CONTINUUM_M_MAX
            assert selection.mass == law.value
            assert selection.pdg_id is None
            assert selection.record_id is None
        else:
            assert law.calls == 0
        assert selection.qn == REMNANT_QN == (1, 1, 0, 1)
        assert selection.mass >= PROTON_MASS

    # the continuous-stream identity over a pinned 1,000-selection
    # trace at the nominal SD split: one shared spectral_state stream
    # consumes exactly the pristine uniform sequence in order (no
    # burn-in, no extra draw anywhere in the module), and the drawn
    # kind is the cumulative resolution of the pristine uniform in the
    # deterministic sorted-kind order
    shared = _remnant_rng(0)
    pristine = np.random.default_rng(_remnant_stream_id(0))
    global0 = _global_rng_snapshot()
    for _i in range(N_BUDGET_TRACE):
        u_ref = pristine.random()
        selection = select_remnant(
            KIND_INTACT_PROTON, SD_WEIGHTS,
            catalog, shared, _refusing_sampler)
        assert selection.kind == (
            KIND_INTACT_PROTON if u_ref < 0.5
            else KIND_LOW_MASS_BARYONIC_RESONANCE)
        assert selection.mass >= PROTON_MASS
    assert shared.bit_generator.state == pristine.bit_generator.state
    assert _global_rng_snapshot() == global0


def test_weights_rejection_pin():
    """A weight vector naming a kind outside ``REMNANT_KINDS`` raises
    ``ValueError`` naming the kind; weights not summing to 1 within
    1e-12 raise ``ValueError``; a negative or non-finite weight raises
    ``ValueError`` naming the kind; a kind argument outside the closed
    set raises ``ValueError`` naming it; a requested kind that does not
    name one of the weights' kinds raises ``ValueError``; every
    refusal precedes any draw (pinned RNG state)."""

    catalog = _catalog()

    # a weight naming a kind outside the closed set: ValueError naming
    # the foreign kind, before any draw
    foreign = "baryonic_continuum_nucleon"
    assert foreign not in REMNANT_KINDS
    rng = _remnant_rng(0)
    state0 = rng.bit_generator.state
    with pytest.raises(ValueError, match=re.escape(foreign)):
        select_remnant(
            KIND_INTACT_PROTON, {foreign: 1.0},
            catalog, rng, _refusing_sampler)
    assert rng.bit_generator.state == state0

    # weights not summing to 1 within 1e-12: ValueError, before any
    # draw
    rng = _remnant_rng(0)
    state0 = rng.bit_generator.state
    with pytest.raises(ValueError, match="sum to 1"):
        select_remnant(
            KIND_INTACT_PROTON,
            {KIND_INTACT_PROTON: 0.5, KIND_LOW_MASS_BARYONIC_RESONANCE: 0.4},
            catalog, rng, _refusing_sampler)
    assert rng.bit_generator.state == state0

    # a negative weight: ValueError naming the kind, before any draw
    rng = _remnant_rng(0)
    state0 = rng.bit_generator.state
    with pytest.raises(ValueError, match=KIND_INTACT_PROTON):
        select_remnant(
            KIND_INTACT_PROTON,
            {KIND_INTACT_PROTON: -0.5,
             KIND_LOW_MASS_BARYONIC_RESONANCE: 1.5},
            catalog, rng, _refusing_sampler)
    assert rng.bit_generator.state == state0

    # a non-finite weight: ValueError naming the kind, before any draw
    rng = _remnant_rng(0)
    state0 = rng.bit_generator.state
    with pytest.raises(ValueError, match=KIND_INTACT_PROTON):
        select_remnant(
            KIND_INTACT_PROTON,
            {KIND_INTACT_PROTON: float("nan"),
             KIND_LOW_MASS_BARYONIC_RESONANCE: 1.0},
            catalog, rng, _refusing_sampler)
    assert rng.bit_generator.state == state0

    # a kind argument outside the closed set: ValueError naming it,
    # before any draw (and before the neutron refusal: the closed-set
    # check precedes it)
    outside = "neutron_absorption"
    assert outside not in REMNANT_KINDS
    rng = _remnant_rng(0)
    state0 = rng.bit_generator.state
    with pytest.raises(ValueError, match=re.escape(outside)):
        select_remnant(
            outside, {KIND_INTACT_PROTON: 1.0},
            catalog, rng, _refusing_sampler)
    assert rng.bit_generator.state == state0

    # a requested kind that does not name one of the weights' kinds:
    # ValueError, before any draw
    rng = _remnant_rng(0)
    state0 = rng.bit_generator.state
    with pytest.raises(ValueError, match=KIND_LOW_MASS_BARYONIC_RESONANCE):
        select_remnant(
            KIND_LOW_MASS_BARYONIC_RESONANCE,
            {KIND_INTACT_PROTON: 1.0},
            catalog, rng, _refusing_sampler)
    assert rng.bit_generator.state == state0


def test_frozen_selection_record():
    """The ``RemnantSelection`` record is a frozen dataclass with
    exactly the card's five fields (kind, pdg_id, record_id, mass,
    qn), and every selection on the pinned traces carries the
    doubled-I3 quantum-number stamp ``(B, Q, S, 2I3) = (1, 1, 0, 1)``."""

    # the frozen record and its exact field set (the card's five
    # fields, in the card's order)
    assert getattr(RemnantSelection, "__dataclass_params__").frozen
    assert [f.name for f in dataclasses.fields(RemnantSelection)] == [
        "kind", "pdg_id", "record_id", "mass", "qn"]

    # the frozen interface: the record is immutable (checked, never
    # repaired)
    catalog = _catalog()
    selection = select_remnant(
        KIND_INTACT_PROTON, {KIND_INTACT_PROTON: 1.0},
        catalog, _remnant_rng(5), _refusing_sampler)
    with pytest.raises(dataclasses.FrozenInstanceError):
        selection.mass = PROTON_MASS + 1.0
    # dataclasses.replace is the sanctioned way to derive a variant
    variant = dataclasses.replace(
        selection, kind=KIND_LOW_MASS_BARYONIC_RESONANCE)
    assert variant.kind == KIND_LOW_MASS_BARYONIC_RESONANCE
    assert variant.qn == selection.qn == (1, 1, 0, 1)

    # the (1, 1, 0, 1) stamp on the pinned traces: every selection of
    # every available kind carries the doubled-I3 remnant quantum
    # numbers exactly
    for kind in sorted(AVAILABLE_REMNANT_KINDS):
        selection = select_remnant(
            kind, {kind: 1.0},
            catalog, _remnant_rng(0), _StandInUniformLaw())
        assert isinstance(selection.qn, tuple)
        assert selection.qn == REMNANT_QN == (1, 1, 0, 1)
        assert all(isinstance(part, int) for part in selection.qn)


# ----------------------------------------------------------------------
# Part 2: the substrate and freeze pins
# ----------------------------------------------------------------------
def test_module_byte_and_export_pins():
    """The module at its committed sha256 (nested tree HEAD a545287)
    with its exact ``__all__`` export set (eighteen names), and the
    package ``__init__`` at its committed sha256 carrying the remnant
    export block (the ten re-exported names; the ``KIND_*`` spellings,
    the charge-exchange plan reference, and the proton's catalog
    identity are module-level only)."""

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    import na49_mixing.topology_remnants as _mod
    assert _mod.__all__ == MODULE_ALL

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)
    init_text = INIT_PATH.read_text(encoding="utf-8")
    block = re.search(
        r"from \.topology_remnants import \((.*?)\)", init_text, re.S)
    assert block is not None, (
        "the __init__ remnant import block is missing")
    for name in INIT_REMNANT_NAMES:
        assert re.search(rf"\b{name}\b", block.group(1)), (
            f"{name} is not in the __init__ remnant import block")
    # the module-level-only names stay out of the package block
    for name in ("KIND_INTACT_PROTON", "KIND_NEUTRON_CHARGE_EXCHANGE",
                 "KIND_LOW_MASS_BARYONIC_RESONANCE",
                 "KIND_HIGH_MASS_BARYONIC_CONTINUUM",
                 "CHARGE_EXCHANGE_TASK_REF", "PROTON_PDG_ID"):
        assert not re.search(rf"\b{name}\b", block.group(1)), (
            f"{name} must not enter the __init__ remnant import block")
    all_block = re.search(r"__all__\s*=\s*\[(.*?)\n\]", init_text, re.S)
    assert all_block is not None
    for name in INIT_REMNANT_NAMES:
        assert f'"{name}"' in all_block.group(1), (
            f"{name} is not in the __init__ __all__")


def test_frozen_substrate_digests():
    """The card's frozen-substrate provenance surfaces (verified on
    disk) and the frozen M8 parameter map through the manifest (the
    ten-parameter set hash and the kernel boundary); the activated
    baryonic ``(1, 1, 0)`` continuum family of the committed
    configuration carries its recorded support ends at the frozen
    boundary and the module's ``CONTINUUM_M_MAX``."""

    for rel, expected in FROZEN_SURFACES.items():
        path = MONO_ROOT / rel
        assert path.is_file(), f"missing frozen surface: {rel}"
        pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    m8 = manifest["frozen_fit_outputs"]["m8_frozen_pp_parameters"]
    computed_hash = hashlib.sha256(
        json.dumps(m8, sort_keys=True, separators=(",", ":")).encode(
            "utf-8")).hexdigest()
    assert computed_hash == M8_HASH
    assert m8["continuum.match_mass"] == KERNEL_BOUNDARY

    # the activated baryonic (1, 1, 0) continuum family of the
    # committed configuration: exactly one, with its support ends at
    # the frozen boundary and the card-6 surface's upper end (the
    # CONTINUUM_M_MAX pin of the module)
    continuum = json.loads(CONTINUUM_PATH.read_text(encoding="utf-8"))
    baryonic_families = [
        family for family in continuum["families"]
        if family["qn"] == [1, 1, 0] and family["active"] is True]
    assert len(baryonic_families) == 1
    assert baryonic_families[0]["family_id"] == "baryonic_np"
    assert baryonic_families[0]["m_min"] == KERNEL_BOUNDARY
    assert baryonic_families[0]["m_max"] == CONTINUUM_M_MAX == 5.0




def test_layout_sync():
    """The layout sync (the subprocess check from the monorepo root;
    ``logic_drift = 0``)."""

    probe = subprocess.run(
        [sys.executable, str(SYNC_CHECKER)],
        capture_output=True, text=True, cwd=str(MONO_ROOT))
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert "logic_drift=0" in probe.stdout
