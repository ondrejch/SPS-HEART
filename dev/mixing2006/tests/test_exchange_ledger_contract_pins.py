"""Round 5, card 2 test-author register: the exchange quantum-number ledger.

Independent contract pins for the explicit exchange quantum-number
ledger (``na49_mixing/exchange_ledger.py``), Round 5 card 2
(``TASK-20260925-r5-exchange-quantum-number-ledger``; plan section 22
task 35, line 1760; plan section 11.4, lines 871-881).  This register is
the test-author half of the card: it pins the module contract on its
own pinned fixtures (the doubled-I3 integer convention, the pinned
inheritance fixture, the closed allowed-exchange set and the
``NonSingletTransferError`` refusal modes, the decay-closure pin on the
87 committed catalog channels at the committed catalog digest, the
five-rule audit at 100% named detection on the pinned exact violation
tuples, the central-neutrality pin, and the zero-draw interface pin)
and re-pins the freeze manifest for the card (``INVENTORY_COUNT`` 504 ->
505).

Part 1: the module contract (pure, hermetic)
--------------------------------------------

- ``test_named_constants_and_records``: the named module constants (the
  doubled-I3 labels, the zero transfer, the vacuum exchange record, the
  closed allowed-exchange set) and the frozen dataclasses with their
  exact field sets, tied to card 1's doubled-I3 proton quadruple.
- ``test_inheritance_arithmetic``: the exact integer transfer map on
  the pinned fixture (``(1, 1, 0, 1)`` + ``(0, -1, 0, -1)`` =
  ``(1, 0, 0, 0)``), the vacuum identity, the bit-identical
  ``apply_transfer`` on the proton line with the four-momentum field
  carried through unchanged, and the checked quadruple inputs.
- ``test_nonsinglet_refusal_modes``: ``apply_transfer`` and
  ``register_exchange`` raise ``NonSingletTransferError`` on every
  exchange outside the closed set, naming it, and the closed set stays
  unchanged; the record-construction and argument-type guards raise
  ``ValueError``.
- ``test_decay_closure_catalog_channels``: ``close_decay`` returns True
  on the committed catalog channels of the ``(1, 1, 0)`` and
  ``(0, 0, 0)`` parents (87 base channels: 50 baryonic, 37 mesonic,
  pinned at the committed catalog digest) cycled to the 10,000-vertex
  fixture, and False on each planted one-unit violation.
- ``test_audit_five_violation_classes``: the pinned compliant event
  returns zero violations, and each of the five violation classes,
  planted one at a time, is detected with the named rule at 100%
  detection on the pinned exact violation tuples (rules (a)-(e)).
- ``test_central_neutrality``: a neutral central source passes rule
  (e), and each of the three planted nonzero central quantum numbers
  produces exactly one named violation.
- ``test_zero_draw_interface``: no ledger callable accepts an
  RNG-shaped parameter, the module imports no random source, and the
  pinned RNG-state identity holds over a full audit trace (the fresh
  generator and the global bit generator are both untouched).

Part 2: the substrate and freeze pins
--------------------------------------

- ``test_module_byte_and_export_pins``: the module at its committed
  sha256, its ``__all__`` export set, and the package ``__init__`` at
  its committed sha256 carrying the exchange-ledger export block.
- ``test_frozen_substrate_digests``: the card's frozen-substrate
  provenance surfaces (verified on disk) and the frozen M8 parameter
  map through the manifest.
- ``test_freeze_repin_register``: the freeze-manifest re-pin for this
  card (the strict path-sorted inventory at 517 entries; 516 after the Round 5 card 5 test-author re-pin; 514 after the
  Round 5 card 5 re-pin; 513 after the Round 5 card 4 test-author
  re-pin; 511 after the Round 5 card 4 re-pin;
  510 after the Round 5 card 3 solver landing;
  508 after the test-protocol tiering change; 505 after the Round 5
  card 2 test-author re-pin; 504 after the Round 5 card 2 solver
  landing; 502 after the Round 5 card 1 test-author re-pin; 501 at
  the Round 5 card 1 landing) carries this
  register and the twenty-six re-pinned chain count-pin files at their
  on-disk digests, the card 2 solver's freeze exception stays exactly
  once, this card's freeze exception names the card exactly once with
  the path-sorted file list, the chain count pins are re-pinned 504 ->
  505 with the 504 literal retained in the naming comments, and the
  frozen M8 parameter map stays intact.

Hermetic: the module, the committed catalog, numpy, and the standard
library only; no event generation driver, no external tables, no Monte
Carlo run (the ledger consumes zero random numbers; runs R50-R55
exercise the ledger, and R54 asserts zero violations across at least
one million events).
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.exchange_ledger import (
    ALLOWED_EXCHANGES,
    Exchange,
    LedgerAuditReport,
    LineState,
    NonSingletTransferError,
    QN_LABELS,
    VACUUM,
    ZERO_TRANSFER,
    apply_transfer,
    apply_transfer_qn,
    audit_event,
    close_decay,
    register_exchange,
)
from na49_mixing.hadron_catalog import HadronCatalog
from tests._freeze_inventory import inventory_count as _inventory_count

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent          # dev/mixing2006
MONO_ROOT = NESTED_ROOT.parent.parent          # the monorepo root
MANIFEST_PATH = NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json"
SYNC_CHECKER = MONO_ROOT / "helpers" / "benchmarking" / "check_helpers_layout_sync.py"
MODULE_PATH = NESTED_ROOT / "na49_mixing" / "exchange_ledger.py"
INIT_PATH = NESTED_ROOT / "na49_mixing" / "__init__.py"

MY_TEST_REL = (
    "dev/mixing2006/tests/test_exchange_ledger_contract_pins.py")

#: The committed catalog and its pinned digest (the card's
#: frozen-substrate provenance; the decay-closure fixture is pinned at
#: this digest).
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
CATALOG_SHA256 = (
    "cf9e79b99199c509f2e83def0a99c215f73ca7761f9f87fb01d1f779b80e32f3")

#: The Round 5 master seed (round 5 index item 9), fixed for the pinned
#: RNG-state identity trace.
MASTER_SEED = 20260925

#: The pinned closure-fixture sizes: the 87 committed base channels of
#: the two parent classes (50 baryonic, 37 mesonic) cycled to the
#: card's 10,000-vertex fixture.
# 87 + the 13 channels of the radiative revision (2026-10-02) - the seven of the
# eta-prime at 377 that the orphan-record revision removed (2026-10-29)
N_BASE_CATALOG_CHANNELS = 100 - 7
N_BARYONIC_BASE = 50
N_MESONIC_BASE = 50 - 7
N_CLOSURE_VERTICES = 10_000

#: A pinned four-momentum fixture (the optional LineState field is
#: carried through apply_transfer unchanged; the ledger performs no
#: kinematics).
FOUR_MOMENTUM_FIXTURE = (2.0, 0.1, -0.2, 1.9)

#: The committed module and package digests (nested tree HEAD bf62e5e).
MODULE_SHA256 = (
    "3ab9b90803611c0d952def364e2ffb05017eb82c66d2e20eae7c966c18f0300a")
#: The package digest re-pinned by Round 5 card 3
#: (TASK-20260925-r5-intact-low-mass-remnants; the remnant module's
#: additive package exports).
# re-pinned by Round 5 card 7 (TASK-20260925-r5-non-diffrative-central-excitation; the non-diffractive module's additive package exports).
# re-pinned by Round 5 card 8 (TASK-20260925-r5-multiple-neutral-excitations; the multiple-neutral-excitation module's additive package exports).
# re-pinned by Round 5 card 9 (TASK-20260925-r5-topology-template-runner; the topology-runner module's additive package exports).
# re-pinned by the Round 5 card 9 review fix round (TASK-20260925-r5-topology-template-runner-review-fix; the REV-bc4de47-03 export-surface correction re-pinned the package __init__).
INIT_SHA256 = (
    "7d2c5a85d9b24de104c0d9cf72195689bf7c39c873e154798699448073b36de8")

#: The module's exact ``__all__`` export set (fifteen names; the two Round 8 sets added 2026-10-04).
MODULE_ALL = [
    "ALLOWED_EXCHANGES",
    "ISOVECTOR_EXCHANGES",  # Round 8 2026-10-04 (TASK-20261004-r8-isovector-exchange-class)
    "ROUND8_EXCHANGES",  # Round 8 2026-10-04
    "Exchange",
    "LedgerAuditReport",
    "LineState",
    "NonSingletTransferError",
    "QN_LABELS",
    "VACUUM",
    "ZERO_TRANSFER",
    "apply_transfer",
    "apply_transfer_qn",
    "audit_event",
    "close_decay",
    "register_exchange",
]

#: The exchange-ledger export block of the package ``__init__`` (the
#: ten names the package re-exports; ``LineState`` stays card 1's
#: elastic record, and ``QN_LABELS`` / ``apply_transfer_qn`` are
#: module-level only).
INIT_LEDGER_NAMES = [
    "ALLOWED_EXCHANGES",
    "Exchange",
    "LedgerAuditReport",
    "NonSingletTransferError",
    "VACUUM",
    "ZERO_TRANSFER",
    "apply_transfer",
    "audit_event",
    "close_decay",
    "register_exchange",
]

#: The M8-frozen kernel boundary, pinned through the manifest's frozen
#: parameter map.
KERNEL_BOUNDARY = 2.680194110727528

#: The M8-frozen ten-parameter set hash (the sha256 of the manifest's
#: frozen parameter map, ``json.dumps(..., sort_keys=True,
#: separators=(",", ":"))``; the card's provenance pin).
M8_HASH = "358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1"

#: The card's frozen-substrate provenance pins (byte-stable surfaces,
#: verified on disk): the builder blend path, the card-6 blend routing,
#: the card-2 engine, the committed catalog, the continuum families
#: configuration, and the sealed Round 4 R45/R46 drivers in both
#: helper trees.
FROZEN_SURFACES = {
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

#: The freeze-manifest inventory entry count after this card's
#: re-pin (504 after the Round 5 card 2 solver landing; +1 entry:
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
#: The solver card (Round 5, card 2) and this test-author register's
#: card name (the freeze-exception identity).
SOLVER_CARD = "TASK-20260925-r5-exchange-quantum-number-ledger"
TEST_AUTHOR_CARD = (
    "TASK-20260925-r5-exchange-quantum-number-ledger-test-author")

#: The register chain: the twenty-six chain count-pin files re-pinned
#: 504 -> 505 by this register (the twenty-five card-1 chain files
#: plus the card-1 register itself, path-sorted).
CHAIN_COUNT_PIN_FILES = (
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
#: manifest, this register, and the twenty-six re-pinned chain
#: count-pin files.
TEST_AUTHOR_EXCEPTION_FILES = tuple(sorted(
    ("dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json",)
    + CHAIN_COUNT_PIN_FILES
    + (MY_TEST_REL,)
))

#: The pinned exact violation tuples of the five-rule audit (the card's
#: 100% named-detection acceptance on the pinned fixtures; rule ids
#: (a)-(e) of ``audit_event``).
VIOLATION_A = (
    "(a) closed-set membership: exchange 2 ('charge-exchange', "
    "transfer (0, -1, 0, -1)) is outside the closed allowed-exchange "
    "set {vacuum}",
)
VIOLATIONS_AB = (
    "(a) closed-set membership: exchange 2 ('vacuum', transfer "
    "(0, 1, 0, 0)) is outside the closed allowed-exchange set "
    "{vacuum}",
    "(b) vacuum zero transfer: exchange 2 ('vacuum') carries the "
    "nonzero transfer (0, 1, 0, 0)",
)
VIOLATION_C = (
    "(c) inheritance after the declared transfer: outgoing line 1 "
    "carries (B, Q, S, 2I3) (1, 0, 0, 1), expected (1, 1, 0, 1) "
    "(incoming line 1 with (1, 1, 0, 1) after the transfer "
    "(0, 0, 0, 0) of exchange 1 ('vacuum'))",
)
VIOLATION_D = (
    "(d) local decay closure: decay vertex 9 daughters sum to "
    "(B, Q, S, 2I3) (0, 2, 0, 4), the parent carries (0, 0, 0, 0)",
)
VIOLATION_E = (
    "(e) central-source neutrality: the central source carries "
    "B = 1 (plan line 849: B = Q = S = 0)",
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _proton_line(four_momentum=None) -> LineState:
    """The proton line ``(B, Q, S, 2I3) = (1, 1, 0, 1)`` (the card's
    pre-registered instantiation)."""

    return LineState(1, 1, 0, 1, four_momentum=four_momentum)


def _compliant_event():
    """The pinned compliant event: two proton lines, two vacuum
    exchanges, two outgoing excitations ``(1, 1, 0, 1)`` (the intact
    inheritance of the zero transfer), a neutral central source, and
    ten closed decay vertices (the pinned two-body channels pi+ pi-,
    pi0 pi0, and K+ K- of the neutral central system, cycled 4 + 3 +
    3)."""

    incoming = (_proton_line(), _proton_line())
    exchanges = (VACUUM, VACUUM)
    outgoing = (LineState(1, 1, 0, 1), LineState(1, 1, 0, 1))
    central = LineState(0, 0, 0, 0)
    pi_pair = (LineState(0, 1, 0, 2), LineState(0, -1, 0, -2))
    pi0_pair = (LineState(0, 0, 0, 0), LineState(0, 0, 0, 0))
    k_pair = (LineState(0, 1, 1, 1), LineState(0, -1, -1, -1))
    closed = ((central, pi_pair), (central, pi0_pair), (central, k_pair))
    decay_vertices = tuple(closed[i % 3] for i in range(10))
    return incoming, exchanges, outgoing, central, decay_vertices


def _catalog_vertices():
    """The committed catalog decay vertices of the ``(1, 1, 0)`` and
    ``(0, 0, 0)`` parents, in committed record/channel order; each
    daughter's ``(B, Q, S, 2I3)`` comes from the committed catalog
    record (the doubled ``isospin3`` field)."""

    catalog = HadronCatalog.from_json(CATALOG_PATH)
    records = {record.pdg_id: record for record in catalog.records()}
    vertices = []
    for record in catalog.records():
        if (record.B, record.Q, record.S) not in ((1, 1, 0), (0, 0, 0)):
            continue
        parent = LineState(record.B, record.Q, record.S, record.isospin3)
        for channel in (record.decay_channels or ()):
            # the photon of a radiative channel (2026-10-02) is no record:
            # it carries (0, 0, 0, 0)
            daughters = tuple(
                LineState(0, 0, 0, 0) if int(pdg) == 22 else
                LineState(records[pdg].B, records[pdg].Q, records[pdg].S,
                          records[pdg].isospin3)
                for pdg in channel.daughters)
            vertices.append((parent, daughters))
    return vertices


def _planted_vertex(vertices, component: str):
    """The first pinned vertex (in committed order) with a nonzero
    *component* on some daughter, and that daughter's index: the
    deterministic host of the planted one-unit violation."""

    for parent, daughters in vertices:
        for index, daughter in enumerate(daughters):
            if getattr(daughter, component) != 0:
                return parent, daughters, index
    raise AssertionError(f"no daughter with nonzero {component}")


def _off_by_one(state: LineState, component: str) -> LineState:
    """The state with one component lowered by one (the planted
    violation)."""

    fields = {"B": state.B, "Q": state.Q, "S": state.S,
              "I3x2": state.I3x2}
    fields[component] -= 1
    return LineState(fields["B"], fields["Q"], fields["S"], fields["I3x2"],
                     four_momentum=state.four_momentum)


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
def test_named_constants_and_records():
    """The named module constants and the doubled-I3 integer
    convention: the quantum-number labels, the zero transfer, the
    vacuum exchange record, the closed allowed-exchange set, and the
    frozen dataclasses with their exact field sets (the record
    convention tie to card 1's doubled-I3 proton quadruple)."""

    assert QN_LABELS == ("B", "Q", "S", "2I3")
    assert ZERO_TRANSFER == (0, 0, 0, 0)
    assert all(isinstance(part, int) for part in ZERO_TRANSFER)

    # the vacuum exchange record and the closed allowed-exchange set
    assert VACUUM.name == "vacuum"
    assert VACUUM.transfer == (0, 0, 0, 0)
    assert isinstance(ALLOWED_EXCHANGES, frozenset)
    assert ALLOWED_EXCHANGES == frozenset({VACUUM})
    assert len(ALLOWED_EXCHANGES) == 1

    # the frozen records and their exact field sets
    assert getattr(LineState, "__dataclass_params__").frozen
    assert getattr(Exchange, "__dataclass_params__").frozen
    assert getattr(LedgerAuditReport, "__dataclass_params__").frozen
    assert [f.name for f in dataclasses.fields(LineState)] == [
        "B", "Q", "S", "I3x2", "four_momentum"]
    assert [f.name for f in dataclasses.fields(Exchange)] == [
        "name", "transfer"]
    assert [f.name for f in dataclasses.fields(LedgerAuditReport)] == [
        "violations"]

    # the record-convention tie to card 1 (consumed read-only): the
    # same doubled-I3 proton quadruple
    from na49_mixing.topology_elastic import PROTON_QN
    assert PROTON_QN == (1, 1, 0, 1)

    # the LineState quadruple accessor
    proton = _proton_line()
    assert proton.qn == (1, 1, 0, 1)

    # checked construction: the quantum numbers are exact integers
    with pytest.raises(ValueError, match="exact integer"):
        LineState(1, 1.0, 0, 1)
    with pytest.raises(ValueError, match="exact integer"):
        Exchange("vacuum", (0, 0, 0, 1.0))
    with pytest.raises(ValueError, match="quadruple"):
        Exchange("vacuum", (0, 0, 0))


def test_inheritance_arithmetic():
    """The exact integer transfer map on the pinned fixture (outgoing =
    incoming + transfer, no tolerance): the proton line under the
    transfer ``(0, -1, 0, -1)`` yields ``(1, 0, 0, 0)`` (the neutron
    ``(B, Q, S)`` on the arithmetic fixture), the vacuum transfer is
    the exact identity, and ``apply_transfer`` on the proton line
    returns the proton line bit-identical with the four-momentum field
    carried through unchanged."""

    # the arithmetic fixture: the exact integer map
    assert apply_transfer_qn((1, 1, 0, 1), (0, -1, 0, -1)) == (1, 0, 0, 0)
    assert apply_transfer_qn(_proton_line().qn, (0, -1, 0, -1)) == (
        1, 0, 0, 0)
    # the vacuum transfer is the exact identity on the same map
    assert apply_transfer_qn(_proton_line().qn, ZERO_TRANSFER) == (1, 1, 0, 1)
    assert apply_transfer_qn((0, 0, 0, 0), (0, 0, 0, 0)) == (0, 0, 0, 0)

    # apply_transfer on the proton line under the vacuum exchange: the
    # line is unchanged, bit-identical fields, four-momentum carried
    # through
    line = _proton_line(four_momentum=FOUR_MOMENTUM_FIXTURE)
    result = apply_transfer(line, VACUUM)
    assert result == line
    assert (result.B, result.Q, result.S, result.I3x2) == (1, 1, 0, 1)
    assert result.four_momentum == FOUR_MOMENTUM_FIXTURE
    assert result.qn == line.qn == (1, 1, 0, 1)

    # checked, never repaired: the transfer map takes quadruples
    with pytest.raises(ValueError, match="quadruples"):
        apply_transfer_qn((1, 1, 0), (0, -1, 0, -1))
    with pytest.raises(ValueError, match="quadruples"):
        apply_transfer_qn((1, 1, 0, 1), (0, -1, 0))


def test_nonsinglet_refusal_modes():
    """``apply_transfer`` and ``register_exchange`` raise
    ``NonSingletTransferError`` on every exchange outside the closed
    set, naming it (the charge exchange and the zero-transfer
    secondary singlet, both Round 8), and the closed set stays
    unchanged; the argument-type guards raise ``ValueError``."""

    proton = _proton_line()

    # the charge exchange is outside the closed set: apply_transfer
    # raises, naming the exchange
    charge_exchange = Exchange("charge-exchange", (0, -1, 0, -1))
    with pytest.raises(NonSingletTransferError,
                       match=re.escape("charge-exchange")):
        apply_transfer(proton, charge_exchange)

    # a zero-transfer record that is not the vacuum exchange is also
    # outside the closed set (record equality is by name and transfer)
    secondary_singlet = Exchange("secondary-singlet", (0, 0, 0, 0))
    assert secondary_singlet not in ALLOWED_EXCHANGES
    assert secondary_singlet != VACUUM
    with pytest.raises(NonSingletTransferError,
                       match=re.escape("secondary-singlet")):
        apply_transfer(proton, secondary_singlet)

    # register_exchange is the Round 8 interface: in Round 5 the set
    # is closed, so it raises on any exchange, naming it
    for candidate in (charge_exchange, secondary_singlet, VACUUM):
        with pytest.raises(NonSingletTransferError,
                           match=re.escape(candidate.name)):
            register_exchange(candidate)

    # the closed set is unchanged after the refused registrations
    assert ALLOWED_EXCHANGES == frozenset({VACUUM})
    assert len(ALLOWED_EXCHANGES) == 1

    # checked arguments: the record and line types are validated
    with pytest.raises(ValueError, match="Exchange"):
        register_exchange("charge-exchange")
    with pytest.raises(ValueError, match="LineState"):
        apply_transfer(proton.qn, VACUUM)
    with pytest.raises(ValueError, match="Exchange"):
        apply_transfer(proton, "vacuum")

    # NonSingletTransferError is a ValueError (the refusal is checked,
    # not repaired)
    assert issubclass(NonSingletTransferError, ValueError)


def test_decay_closure_catalog_channels():
    """``close_decay`` returns True on the pinned 10,000-vertex
    fixture (the committed catalog channels of the ``(1, 1, 0)`` and
    ``(0, 0, 0)`` parents, pinned at the committed catalog digest) and
    False on each planted one-unit violation (one daughter charge,
    strangeness, and baryon number off by 1)."""

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    base = _catalog_vertices()
    assert len(base) == N_BASE_CATALOG_CHANNELS
    n_baryonic = sum(1 for parent, _ in base if parent.B == 1)
    assert n_baryonic == N_BARYONIC_BASE
    assert len(base) - n_baryonic == N_MESONIC_BASE

    # the pinned 10,000-vertex fixture: the committed channels cycle
    vertices = [base[i % len(base)] for i in range(N_CLOSURE_VERTICES)]
    for parent, daughters in vertices:
        assert close_decay(parent, daughters) is True

    # the three planted violations, each on a pinned committed vertex
    for component in ("Q", "S", "B"):
        parent, daughters, index = _planted_vertex(base, component)
        bad = (daughters[:index]
               + (_off_by_one(daughters[index], component),)
               + daughters[index + 1:])
        assert close_decay(parent, bad) is False
        # the unmutated vertex closes (the violation is the one unit)
        assert close_decay(parent, daughters) is True

    # checked arguments: the parent must be a LineState
    with pytest.raises(ValueError, match="LineState parent"):
        close_decay((0, 0, 0, 0), ())


def test_audit_five_violation_classes():
    """The pinned compliant event returns zero violations, and each of
    the five violation classes, planted one at a time on the compliant
    event, is detected with the named rule (100% detection on the
    pinned fixture; the exact violation tuples are pinned)."""

    incoming, exchanges, outgoing, central, decay_vertices = (
        _compliant_event())

    # the compliant event: zero violations
    report = audit_event(incoming, exchanges, outgoing, central,
                         decay_vertices)
    assert isinstance(report, LedgerAuditReport)
    assert report.violations == ()

    # (a) an exchange record outside the closed set, appended beyond
    # the two outgoing lines' pairing indices (rule (c) is untouched)
    planted = exchanges + (Exchange("charge-exchange", (0, -1, 0, -1)),)
    report = audit_event(incoming, planted, outgoing, central,
                         decay_vertices)
    assert report.violations == VIOLATION_A

    # (b) a vacuum-named exchange with a nonzero transfer (also outside
    # the closed set; both rules name the record)
    planted = exchanges + (Exchange("vacuum", (0, 1, 0, 0)),)
    report = audit_event(incoming, planted, outgoing, central,
                         decay_vertices)
    assert report.violations == VIOLATIONS_AB

    # (c) an outgoing excitation that does not inherit the line quantum
    # numbers after the declared transfer
    bad_outgoing = (outgoing[0], LineState(1, 0, 0, 1))
    report = audit_event(incoming, exchanges, bad_outgoing, central,
                         decay_vertices)
    assert report.violations == VIOLATION_C

    # (d) a decay vertex that does not close locally (vertex 9 planted:
    # pi+ pi+ instead of the pinned K+ K-)
    bad_vertices = list(decay_vertices)
    bad_vertices[9] = (central,
                       (LineState(0, 1, 0, 2), LineState(0, 1, 0, 2)))
    report = audit_event(incoming, exchanges, outgoing, central,
                         tuple(bad_vertices))
    assert report.violations == VIOLATION_D

    # (e) a central source carrying baryon number
    report = audit_event(incoming, exchanges, outgoing,
                         LineState(1, 0, 0, 0), decay_vertices)
    assert report.violations == VIOLATION_E


def test_central_neutrality():
    """A central source with ``(B, Q, S) = (0, 0, 0)`` passes rule (e);
    each of the three planted nonzero central quantum numbers produces
    exactly one named violation (plan line 849: B = Q = S = 0)."""

    incoming, exchanges, outgoing, _central, decay_vertices = (
        _compliant_event())

    report = audit_event(incoming, exchanges, outgoing,
                         LineState(0, 0, 0, 0), decay_vertices)
    assert report.violations == ()

    for name, bad_source in (
            ("B", LineState(1, 0, 0, 0)),
            ("Q", LineState(0, 1, 0, 0)),
            ("S", LineState(0, 0, 1, 0)),
    ):
        report = audit_event(incoming, exchanges, outgoing, bad_source,
                             decay_vertices)
        assert len(report.violations) == 1, (name, report.violations)
        violation = report.violations[0]
        assert violation.startswith("(e) central-source neutrality")
        assert f"carries {name} = 1" in violation
        assert "plan line 849: B = Q = S = 0" in violation


def test_zero_draw_interface():
    """The ledger machinery consumes zero random numbers: no ledger
    callable accepts an RNG-shaped parameter, the module imports no
    random source, and the pinned RNG-state identity holds over a full
    audit trace (the fresh generator and the global bit generator are
    both untouched)."""

    # structural: no ledger callable accepts an RNG-shaped parameter
    for func in (apply_transfer, apply_transfer_qn, audit_event,
                 close_decay, register_exchange):
        signature = inspect.signature(func)
        for name, parameter in signature.parameters.items():
            assert parameter.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD), (func, name)
            assert not re.search(r"rng|random|generator|seed|draw",
                                 name, re.IGNORECASE), (func, name)

    # structural: the module imports no random source at all (standard
    # library bookkeeping only)
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "numpy" not in source
    assert "import random" not in source
    assert "np.random" not in source

    # the pinned RNG-state identity over a full audit trace
    rng = np.random.default_rng(MASTER_SEED)
    state0 = rng.bit_generator.state
    global0 = _global_rng_snapshot()

    incoming, exchanges, outgoing, central, decay_vertices = (
        _compliant_event())
    assert apply_transfer(_proton_line(), VACUUM).qn == (1, 1, 0, 1)
    assert apply_transfer_qn(_proton_line().qn, (0, -1, 0, -1)) == (1, 0, 0, 0)
    for parent, daughters in decay_vertices:
        assert close_decay(parent, daughters)
    assert audit_event(incoming, exchanges, outgoing, central,
                       decay_vertices).violations == ()
    for planted in (
            exchanges + (Exchange("charge-exchange", (0, -1, 0, -1)),),
            exchanges + (Exchange("vacuum", (0, 1, 0, 0)),),
    ):
        assert audit_event(incoming, planted, outgoing, central,
                           decay_vertices).violations != ()
    with pytest.raises(NonSingletTransferError):
        register_exchange(Exchange("charge-exchange", (0, -1, 0, -1)))

    assert rng.bit_generator.state == state0
    assert _global_rng_snapshot() == global0


# ----------------------------------------------------------------------
# Part 2: the substrate and freeze pins
# ----------------------------------------------------------------------
def test_module_byte_and_export_pins():
    """The module at its committed sha256 with its exact ``__all__``
    export set (fifteen names since Round 8), and the package ``__init__`` at its
    committed sha256 carrying the exchange-ledger export block (the ten
    re-exported names; ``LineState`` stays card 1's elastic record, and
    ``QN_LABELS`` / ``apply_transfer_qn`` are module-level only)."""

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    import na49_mixing.exchange_ledger as _mod
    assert _mod.__all__ == MODULE_ALL

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)
    init_text = INIT_PATH.read_text(encoding="utf-8")
    block = re.search(
        r"from \.exchange_ledger import \((.*?)\)", init_text, re.S)
    assert block is not None, (
        "the __init__ exchange-ledger import block is missing")
    for name in INIT_LEDGER_NAMES:
        assert re.search(rf"\b{name}\b", block.group(1)), (
            f"{name} is not in the __init__ exchange-ledger import block")
    # the package keeps card 1's LineState; the ledger's record names
    # that would collide are not in the import block
    for name in ("LineState", "QN_LABELS", "apply_transfer_qn"):
        assert not re.search(rf"\b{name}\b", block.group(1)), (
            f"{name} must not enter the __init__ exchange-ledger block")
    all_block = re.search(r"__all__\s*=\s*\[(.*?)\n\]", init_text, re.S)
    assert all_block is not None
    for name in INIT_LEDGER_NAMES:
        assert f'"{name}"' in all_block.group(1), (
            f"{name} is not in the __init__ __all__")


def test_frozen_substrate_digests():
    """The card's frozen-substrate provenance surfaces (verified on
    disk) and the frozen M8 parameter map through the manifest (the
    ten-parameter set hash and the kernel boundary)."""

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




def test_layout_sync():
    """The layout sync (the subprocess check from the monorepo root;
    ``logic_drift = 0``)."""

    probe = subprocess.run(
        [sys.executable, str(SYNC_CHECKER)],
        capture_output=True, text=True, cwd=str(MONO_ROOT))
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert "logic_drift=0" in probe.stdout
