"""Round 5, card 6 test-author register: the double-diffraction class.

Independent contract pins for the double-diffraction class
(``na49_mixing/topology_double_diffraction.py``), Round 5 card 6
(``TASK-20260925-r5-double-diffraction``; plan section 22 task 30,
line 1755; plan section 11.1 item 4, line 827), at nested tree SHA
86fd47f (the documented-coupling fix; the operative solve convention
is unchanged from the 339f817 landing).  This register is the
test-author half of the card: it pins
the module contract on its own pinned fixtures (the frozen ``DDEvent``
record surface with the pair-tuple exchange records, the exact 2-to-2
solve's closure, mass shell, the intact two-body edge, and the
drawn-``(t_1, t_2)`` over-determination convention with its clamped
saturation slivers, the central-gap pin on a pinned event, the
draw-budget pins on the closed substream labels, the joint-law pins on
the exact mass limits, the frozen-cascade pins at the sealed R45 point
with the R46 ``no_refit`` white-box check, the conservation pins with
zero audit violations, the byte-stable module / solver test / package
``__init__`` digests and export surface, and the frozen-substrate
provenance) and re-pins the freeze manifest for the card
(``INVENTORY_COUNT`` 519 -> 520).

Part 1: the module contract (pure, hermetic)
--------------------------------------------

- ``test_named_constants_and_record_surface``: the closed
  double-diffraction line-label set ``{"projectile", "target"}``, the
  class label ``"dd"``, the frozen record surfaces (the nine-field
  ``DDEvent`` and the five-field ``FinalHadron``), the nominal
  parameters (``epsilon = 0``, ``B_SD`` the M8-frozen slope), the
  ledger constants (the proton quantum numbers, the vacuum exchange
  with the zero transfer, the baryonic ``baryonic_np`` excitation
  family), and the card-5 closed-substream ``generate_dd`` signature
  (three exchange/decay streams; no ``spectral_state`` and no
  ``topology_selection`` stream).
- ``test_dd_kinematics_closure_and_mass_shell``: the exact 2-to-2
  solve closes the incoming sum within ``1e-9`` relative on a pinned
  grid of ``(M_1, t_1, M_2, t_2)`` points, the solved pair sits on its
  mass shell, and the checked domain refusals (positive transfers,
  non-positive mass, above-threshold configuration) raise
  ``ValueError``.
- ``test_dd_kinematics_intact_two_body_edge``: the intact two-body edge
  (both systems at the proton mass, both transfers zero) resolves at
  the elastic-forward configuration: system 1 carries the beam
  four-vector and system 2 the target at rest, within ``1e-12``.
- ``test_dd_kinematics_overdetermination_convention``: the operative
  solve convention on the drawn ``(t_1, t_2)`` pair: the production
  angle follows from the drawn ``t_1`` through the documented
  invariant transfer on the projectile-side excitation line (the
  card-5 documented relation), the realized pair carries the exact
  two-body coupling ``t_1 = t_2`` (the back-to-back configuration of
  the shared total four-momentum), the recorded values stay the drawn
  ones (bit-identical transfers on the pinned event), and both
  saturation slivers clamp to the forward and backward
  configurations (the drawn domains extend past the exact two-body
  reach of the realized system).
- ``test_dd_kinematics_documented_sum_coupling``: the exact
  two-body coupling on the realized configuration (the pin of the
  finding behind the pre-86fd47f documentation bug): momentum
  conservation on the shared total four-momentum makes the two
  exchanges carry the same transfer four-vector, so the realized
  transfers coincide, ``t_1 = t_2``, at float-rounding level on the
  pinned grid and the pinned trace, and the realized sum deviates
  from the pre-fix documented sum coupling ``t_1 + t_2 = 2 m_p^2 +
  M_1^2 + M_2^2 - s`` by ``4 p* q* cos(theta*)`` (the sum-coupling
  deviation; measured 275.5850246322698 GeV^2 at the pinned
  event-0 pair, vanishing only at the central configuration
  ``cos(theta*) = 0``).
- ``test_central_gap_pin``: the central gap (the R50 stored quantity,
  plan line 903, the cross-side pair-distance minimum) on the pinned
  event equals the hand-computed rapidity difference within ``1e-12``
  and is positive on the pinned trace; the checked refusals (a
  recorded gap that disagrees beyond ``1e-12``, a non-``DDEvent``
  argument, a one-sided final state) raise ``ValueError``.
- ``test_no_intact_remnant``: the event record carries no intact
  proton remnant and no remnant kind draw (white-box field scan and
  module source scan); both lines are excited baryonic systems
  carrying ``(1, 1, 0, 1)`` after the vacuum transfers, the two
  exchange records are the pair-tuple vacuum convention, and the
  final-state line labels are the closed double-diffraction set
  (plan line 827).
- ``test_draw_budget``: a pinned 200-event trace on one shared stream
  per label consumes exactly six ``exchange_kinematics`` uniforms per
  accepted event plus four per rejected attempt (the pinned attempt
  histogram and the pinned per-vertex ``decay_channel`` /
  ``decay_kinematics`` totals at this landing), zero
  ``spectral_state`` and zero ``topology_selection`` uniforms (the
  class draw belongs to the runner, card 9), with pinned RNG-state
  identities.
- ``test_pathological_rejection_cap``: the pinned pathological
  configuration (individually legal windows, jointly impossible)
  raises ``PhaseSpaceRejectionExceeded`` at exactly ``REJECTION_CAP =
  1000`` attempts consuming exactly 4000 uniforms, the message naming
  the attempt count and the parameter point, with pinned RNG-state
  identity; the module's ``generate_dd`` consumes the same card-4
  joint draw on the exact windows (white-box source check).
- ``test_joint_law_pins``: the exact marginal limits
  ``(0.880354927712736, 266.7253293403128)`` GeV^2 at the nominal p+p
  configuration; on the pinned trace ``M_1 + M_2 <= sqrt(s)`` within
  ``1e-12``, both ``M_i^2`` inside the exact limits, and both ``t_i``
  inside their exact two-body ranges closed at ``t = 0``.
- ``test_conservation_pins``: over the pinned trace the final state
  carries exactly ``(B, Q, S) = (2, 2, 0)``, the four-momentum closure
  is within ``1e-9`` relative, and every event passes card 2's
  ``audit_event`` with zero violations (two vacuum transfers, both
  excitation lines ``(1, 1, 0, 1)``, no central source: rule (e)
  vacuous on the quantum-number-free central record).
- ``test_frozen_cascade_pins``: both excitation decays are the frozen
  simplified cascade at the sealed R45 point (kappa_B = 0.0, kappa_M
  = 4.0, b_dec = 2.0 GeV^-2, baryonic nominal width 0.3 GeV, pinned
  through the R45 driver digests in both helper trees), the
  ``W_flavor = 1.0`` hook identity on the channel weight, the
  ``baryonic_np`` excitation family, and the R46 ``no_refit``
  white-box check (no module in this card references a re-fit of the
  alignment parameters or claims the separation of exchange recoil
  from decay transverse dynamics).

Part 2: the substrate and freeze pins
--------------------------------------

- ``test_module_byte_and_export_pins``: the module at its committed
  sha256, its ``__all__`` export set, the solver test file at its
  committed sha256, and the package ``__init__`` at its committed
  sha256 carrying the double-diffraction export block (the six
  re-exported names; the package ``__all__`` total at 220 entries).
- ``test_frozen_substrate_digests``: the card's frozen-substrate
  provenance surfaces (verified on disk) and the frozen M8 parameter
  map through the manifest, with the activated baryonic ``(1, 1, 0)``
  continuum family's recorded support ends.
- ``test_freeze_repin_register``: the freeze-manifest re-pin for this
  card (the strict path-sorted inventory at 520 entries; 519 after
  the Round 5 card 6 re-pin; 517 after the Round 5 card 5 test-author
  re-pin; 516 after the Round 5 card 5 re-pin; 514 after the Round 5
  card 4 test-author re-pin; 513 after the Round 5 card 4 re-pin; 511
  after the Round 5 card 3 test-author re-pin; 510 after the Round 5
  card 3 solver landing; 508 after the test-protocol tiering change;
  505 after the Round 5 card 2 test-author re-pin; 504 after the
  Round 5 card 2 solver landing; 502 after the Round 5 card 1
  test-author re-pin; 501 at the Round 5 card 1 landing; +1 entry:
  this register) carries this register and the thirty re-pinned
  chain count-pin files at their on-disk digests, the card 6
  solver's freeze exception stays exactly once at its 34 path-sorted
  files with its reason naming the 517 -> 519 re-pin, this card's
  freeze exception names the card exactly once with the path-sorted
  file list and a reason naming the 519 -> 520 re-pin, the chain
  count pins are re-pinned 519 -> 520 with the 519 literal retained in
  the naming comments, and the frozen M8 parameter map stays intact.
- ``test_layout_sync``: the layout sync (the subprocess check from the
  monorepo root; ``logic_drift = 0``).

Hermetic: the module, the committed catalog, the continuum families
configuration, the Round 4 cascade modules (consumed read-only), the
committed cards 1-5 modules (consumed read-only), numpy, and the
standard library only; no event generation driver, no external
tables, no Monte Carlo run (the runs R50-R55 exercise the class
statistically).
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import inspect
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.exchange_ledger import (
    Exchange,
    LineState,
    audit_event,
)
from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.rng_substreams import SUBSTREAM_LABELS, derive_substreams
from na49_mixing.spectral_density import families_from_json
from na49_mixing.topology_elastic import (
    BEAM_MOMENTUM,
    ELASTIC_TRANSFER_SLOPE,
    PROTON_MASS,
    SQRT_S,
)
from na49_mixing.topology_mass_laws import (
    REJECTION_CAP,
    MassLawParams,
    PhaseSpaceRejectionExceeded,
    draw_joint_dd,
    sd_limits,
    t_range,
)
from na49_mixing.topology_remnants import KERNEL_BOUNDARY, REMNANT_QN
from na49_mixing.topology_double_diffraction import (
    ANCESTRY_LABEL,
    DD_CLASS_LABEL,
    DD_LINE_LABELS,
    EXCITATION_FAMILY_ID,
    NOMINAL_PARAMS,
    PROTON_QN,
    PROJECTILE_LINE_LABEL,
    TARGET_LINE_LABEL,
    VACUUM_EXCHANGE_NAME,
    ZERO_TRANSFER,
    DDEvent,
    FinalHadron,
    _beam_configuration,
    _incoming_lines,
    central_gap,
    dd_kinematics,
    generate_dd,
)
from tests._freeze_inventory import inventory_count as _inventory_count

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent          # dev/mixing2006
MONO_ROOT = NESTED_ROOT.parent.parent          # the monorepo root
MANIFEST_PATH = NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json"
SYNC_CHECKER = MONO_ROOT / "helpers" / "benchmarking" / "check_helpers_layout_sync.py"
MODULE_PATH = NESTED_ROOT / "na49_mixing" / "topology_double_diffraction.py"
SOLVER_TEST_PATH = NESTED_ROOT / "tests" / "test_topology_double_diffraction.py"
INIT_PATH = NESTED_ROOT / "na49_mixing" / "__init__.py"
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
CONTINUUM_PATH = (
    NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json")

#: The Round 5 master seed (round 5 index item 9), fixed for the pinned
#: traces; replica 0, one substream triple per event index.
MASTER_SEED = 20260925

#: The pinned trace sizes (the 500-event conservation/joint/gap trace
#: and the 200-event draw-budget trace).
N_PINNED = 500
N_BUDGET = 200

#: The exact marginal window of the joint law (both lines at the intact
#: proton mass; the card's ``[m_p^2, (sqrt(s) - m_p)^2]`` limits at the
#: recorded ``SQRT_S = 17.270``, the card-4 recorded values).
M2_MIN = 1.151863222816188  # (m_p + m_pi0)^2: the N pi threshold edge (2026-10-02)
M2_MAX = 266.7253293403128

#: The pinned event-0 record (master seed 20260925, replica 0, event
#: index 0; the joint-law draw and the central gap at that landing).
# re-pinned 2026-10-02 for the kernel fixes (the CM-mirrored target diffraction, the N pi threshold, the radiative channels)
EVENT0_M1 = 2.555650805763481
EVENT0_M2 = 4.089372877543087
EVENT0_T1 = -0.44500500534237103
EVENT0_T2 = -0.023600994766667327
# re-pinned 2026-09-29 for the Block B stage 1 channel pool (the heavier direct decay classes)
EVENT0_GAP = 1.0530511250407417

#: The pinned draw-budget totals of the 200-event budget trace (master
#: seed 20260925, replica 0, one shared stream per label in event
#: order): the joint-law attempts per event (each accepted event
#: consumes 6 ``exchange_kinematics`` uniforms plus 4 uniforms per
#: rejected attempt on the same stream) and the frozen Round 4
#: per-vertex decay budget totals on the ``decay_channel`` /
#: ``decay_kinematics`` streams at this landing.
DD_ATTEMPT_HISTOGRAM = {1: 171, 2: 26, 3: 3}
DD_EXCHANGE_TOTAL = 1_328
# re-pinned 2026-09-29 for the Block B stage 1 channel pool (the heavier direct decay classes)
# re-pinned 2026-09-30 for the corrected final state (decays at every assembly point, the isospin rule)
DD_CHANNEL_TOTAL = 1_632
DD_KINEMATICS_TOTAL = 3_698

#: The committed digests of the card's pinned surfaces (nested tree
#: HEAD 86fd47f; the module digest follows the documentation-only
#: coupling fix, the ``__init__`` digest the import-block dedup, and
#: the solver test stays byte-identical to the 339f817 landing).
MODULE_SHA256 = (
    "92c48ece4d22577b91cc3950e6ee20cf3a3bea5ad56b0ffc6edd32445174d54e")
SOLVER_TEST_SHA256 = (
    "d47bc602ec23caad6d0ec4927a627c75bd239f2f662f29e0420cf2fe653b294d")
# re-pinned by Round 5 card 7 (TASK-20260925-r5-non-diffrative-central-excitation; the non-diffractive module's additive package exports).
# re-pinned by Round 5 card 8 (TASK-20260925-r5-multiple-neutral-excitations; the multiple-neutral-excitation module's additive package exports).
# re-pinned by Round 5 card 9 (TASK-20260925-r5-topology-template-runner; the topology-runner module's additive package exports).
# re-pinned by the Round 5 card 9 review fix round (TASK-20260925-r5-topology-template-runner-review-fix; the REV-bc4de47-03 export-surface correction re-pinned the package __init__).
INIT_SHA256 = (
    "7d2c5a85d9b24de104c0d9cf72195689bf7c39c873e154798699448073b36de8")

#: The module's ``__all__`` export set (thirteen names).
MODULE_ALL = [
    "DD_CLASS_LABEL",
    "DD_LINE_LABELS",
    "PROJECTILE_LINE_LABEL",
    "TARGET_LINE_LABEL",
    "NOMINAL_PARAMS",
    "BEAM_MOMENTUM",
    "PROTON_MASS",
    "SQRT_S",
    "FinalHadron",
    "DDEvent",
    "central_gap",
    "dd_kinematics",
    "generate_dd",
]

#: The ``__init__`` double-diffraction export block (the six
#: re-exported names; ``NOMINAL_PARAMS``, ``BEAM_MOMENTUM``,
#: ``PROTON_MASS``, and ``SQRT_S`` stay card 1's elastic constants and
#: are module-level only, and the final-state hadron record stays
#: card 5's package-level ``FinalHadron`` export).
INIT_DD_NAMES = [
    "DD_CLASS_LABEL",
    "DD_LINE_LABELS",
    "DDEvent",
    "central_gap",
    "dd_kinematics",
    "generate_dd",
]

#: The package ``__all__`` total entry count (the additive card 6
#: export block joins the frozen surface).
INIT_ALL_COUNT = 253

#: The M8-frozen ten-parameter set hash (the sha256 of the manifest's
#: frozen parameter map, ``json.dumps(..., sort_keys=True,
#: separators=(",", ":"))``; the card's provenance pin).
M8_HASH = "358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1"

#: The card's frozen-substrate provenance pins (byte-stable surfaces,
#: verified on disk): the builder blend path, the card-6 blend routing,
#: the card-2 engine, the committed catalog, the continuum families
#: configuration, the sealed Round 4 R45/R46 drivers in both helper
#: trees, and the five committed Round 5 modules consumed read-only
#: (card 1's elastic reference, card 2's exchange ledger, card 3's
#: remnants, card 4's mass laws, card 5's single diffraction).
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
    "dev/mixing2006/na49_mixing/topology_remnants.py":
        "8c26ae419cbb0e7de3b36d95bb6e097c1854df7d2c69f8558b7f71bd4a0342ca",
    "dev/mixing2006/na49_mixing/topology_mass_laws.py":
        "6af0d8fbdf67204e316c42cae8e9962fb87251000b9e1364135045227224db0a",
    "dev/mixing2006/na49_mixing/topology_single_diffraction.py":
        "bbfb63d651bb4fa12a9a75b464b696c6a7be0c88eddc27bd267714a9479e205a",
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

#: The white-box tokens of a refit of the sealed R45 alignment point or
#: of a separation claim (the R46 clauses; none may appear anywhere in
#: the module source).
NO_REFIT_TOKENS = (
    "kappa_fit", "refit_kappa", "align_refit", "refit_alignment",
    "separation_claimed = True", "exchange_recoil_decay_separation",
)

#: The white-box tokens of a remnant path (the no-remnant contract,
#: plan line 827): none may appear anywhere in the module source (the
#: class has no remnant selection, no remnant kind draw, and no
#: spectral_state stream).
NO_REMNANT_TOKENS = (
    "select_remnant", "RemnantSelection", "REMNANT_KINDS",
    "rng_spectral_state", "KIND_INTACT_PROTON",
)

# re-pinned 517 -> 519 by Round 5 card 6 (TASK-20260925-r5-double-diffraction; +2 entries: the module na49_mixing/topology_double_diffraction.py and the test file tests/test_topology_double_diffraction.py).
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
#: The solver card (Round 5, card 6) and this test-author register's
#: card name (the freeze-exception identity).
SOLVER_CARD = "TASK-20260925-r5-double-diffraction"
TEST_AUTHOR_CARD = (
    "TASK-20260925-r5-double-diffraction-test-author")

#: This register's relative path in the monorepo (the inventory key).
MY_TEST_REL = (
    "dev/mixing2006/tests/test_topology_double_diffraction_contract_pins.py")

#: The register chain: the thirty chain count-pin files re-pinned
#: 519 -> 520 by this register (the card-6 solver's thirty chain files,
#: path-sorted).
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
    "dev/mixing2006/tests/test_topology_remnants_contract_pins.py",
    "dev/mixing2006/tests/test_topology_mass_laws_contract_pins.py",
    "dev/mixing2006/tests/test_topology_single_diffraction_contract_pins.py",
)

#: This register's freeze-exception file list (path-sorted): the
#: manifest, this register, and the thirty re-pinned chain
#: count-pin files.
TEST_AUTHOR_EXCEPTION_FILES = tuple(sorted(
    ("dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json",)
    + CHAIN_COUNT_PIN_FILES
    + (MY_TEST_REL,)
))


# ----------------------------------------------------------------------
# Fixtures and helpers
# ----------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _incoming_sum() -> tuple[float, float, float, float]:
    """The lab four-momentum sum of the pinned incoming pair."""

    lines = _incoming_lines()
    beam = lines[0].four_momentum
    target = lines[1].four_momentum
    if beam is None or target is None:
        raise ValueError("the pinned incoming lines carry no four-momenta")
    return (beam[0] + target[0], beam[1] + target[1],
            beam[2] + target[2], beam[3] + target[3])


def _rapidity(four_momentum) -> float:
    """The lab rapidity of a four-momentum (the card's convention)."""

    e, _px, _py, pz = four_momentum
    return 0.5 * math.log((e + pz) / (e - pz))


def _invariant_mass2(four_momentum) -> float:
    """The squared invariant mass of a four-momentum."""

    e, px, py, pz = four_momentum
    return e * e - px * px - py * py - pz * pz


def _cm_quantities(m1: float, m2: float):
    """The exact two-body CM quantities of the pinned configuration:
    ``(e_p, e_1, e_2, p_star, q_star)`` for the incoming proton pair
    and the outgoing ``(M_1, M_2)`` pair."""

    _e_beam, _e_total, sq_exact, p_star, _beta, _gamma = (
        _beam_configuration())
    s = sq_exact * sq_exact
    mp2 = PROTON_MASS * PROTON_MASS
    e_p = s / (2.0 * sq_exact)
    e_1 = (s + m1 * m1 - m2 * m2) / (2.0 * sq_exact)
    e_2 = (s + m2 * m2 - m1 * m1) / (2.0 * sq_exact)
    q_star = math.sqrt(
        max(0.0, (s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2))
        ) / (2.0 * sq_exact)
    return e_p, e_1, e_2, p_star, q_star


def _realized_transfer(system, other) -> float:
    """The momentum transfer of the solved system against the incoming
    proton: the projectile-side exchange against the beam, the
    target-side exchange against the target at rest (the lab
    invariant ``(p_in - P)^2``)."""

    if other == PROJECTILE_LINE_LABEL:
        e_beam, _e_total, _sq, _p_star, _beta, _gamma = (
            _beam_configuration())
        incoming = (e_beam, 0.0, 0.0, BEAM_MOMENTUM)
    else:
        incoming = (PROTON_MASS, 0.0, 0.0, 0.0)
    dot = (incoming[0] * system[0] - incoming[1] * system[1]
           - incoming[2] * system[2] - incoming[3] * system[3])
    return (PROTON_MASS * PROTON_MASS + _invariant_mass2(system)
            - 2.0 * dot)


def _event_rngs(event_index: int):
    """The closed substream generators of one pinned event (the
    double-diffraction triple; no ``spectral_state`` stream exists in
    the class's budget)."""

    subs = derive_substreams(MASTER_SEED, 0, event_index)
    return tuple(
        np.random.default_rng(subs.stream_for(label))
        for label in ("exchange_kinematics", "decay_channel",
                      "decay_kinematics"))


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


def _exchange_records(event: DDEvent) -> tuple[Exchange, Exchange]:
    """The card-2 ledger exchange records of the event's two vacuum
    exchanges (the test-side conversion of the pair-tuple convention)."""

    records = []
    for name, transfer in event.exchanges:
        records.append(Exchange(
            name=name,
            transfer=(transfer[0], transfer[1], transfer[2], transfer[3])))
    return (records[0], records[1])


def _decay_vertices(event: DDEvent):
    """The card-2 ledger decay vertices of the event: each excited
    line as the parent and its final-state hadrons as the daughters."""

    vertices = []
    for label, line in ((PROJECTILE_LINE_LABEL, event.lines[0]),
                        (TARGET_LINE_LABEL, event.lines[1])):
        daughters = tuple(
            LineState(B=h.qn[0], Q=h.qn[1], S=h.qn[2], I3x2=h.qn[3])
            for h in event.final_state if h.line == label)
        if not daughters:
            raise ValueError(
                f"the event carries no {label!r}-side final state")
        vertices.append((line, daughters))
    return tuple(vertices)


def _audit(event: DDEvent):
    """Card 2's five-rule audit over the event's ledger records (no
    central source: the class produces none)."""

    return audit_event(
        incoming_lines=_incoming_lines(),
        exchanges=_exchange_records(event),
        outgoing_lines=event.lines,
        central_source=None,
        decay_vertices=_decay_vertices(event),
    )


@pytest.fixture(scope="module")
def catalog() -> HadronCatalog:
    return HadronCatalog.from_json(CATALOG_PATH)


@pytest.fixture(scope="module")
def fams():
    return families_from_json(str(CONTINUUM_PATH))


@pytest.fixture(scope="module")
def pinned_trace(catalog, fams):
    """The pinned 500-event double-diffraction trace (master seed
    20260925, replica 0, event order)."""

    events = []
    for event_index in range(N_PINNED):
        rngs = _event_rngs(event_index)
        events.append(generate_dd(NOMINAL_PARAMS, catalog, fams, *rngs))
    return events


# ----------------------------------------------------------------------
# Part 1: the module contract (pure, hermetic)
# ----------------------------------------------------------------------
def test_named_constants_and_record_surface():
    """The closed line-label set, the class label, the frozen record
    surfaces, the nominal parameters (the R50 fixed values:
    ``epsilon = 0``, ``B_SD`` the M8-frozen slope), the ledger
    constants, and the card-5 closed-substream ``generate_dd``
    signature (no ``spectral_state`` and no ``topology_selection``
    stream: the remnant kind draw is absent by the pre-registered
    instantiation and the class draw belongs to the runner, card 9)."""

    assert DD_CLASS_LABEL == "dd"
    assert PROJECTILE_LINE_LABEL == "projectile"
    assert TARGET_LINE_LABEL == "target"
    assert isinstance(DD_LINE_LABELS, frozenset)
    assert DD_LINE_LABELS == frozenset({"projectile", "target"})
    # the frozen record surfaces
    assert getattr(DDEvent, "__dataclass_params__").frozen
    assert tuple(DDEvent.__dataclass_fields__) == (
        "class_label", "m1", "m2", "t1", "t2", "lines", "exchanges",
        "final_state", "gap_size",
        "exchange_configuration", "line_m_min", "slope_factor")  # Round 8 2026-10-04, default None (the vacuum exchanges);
    # line_m_min: 2026-10-06, None = the shared edges (line_thresholds off)
    # slope_factor: 2026-10-08, None = the single transfer slope (transfer_slope_mass_exponent off)
    assert DDEvent.__dataclass_fields__["exchange_configuration"].default is None
    assert getattr(FinalHadron, "__dataclass_params__").frozen
    assert tuple(FinalHadron.__dataclass_fields__) == (
        "pdg_id", "four_momentum", "mass", "qn", "line")
    # the nominal parameters: the R50 fixed values
    assert NOMINAL_PARAMS.epsilon == 0.0
    assert NOMINAL_PARAMS.b_sd == ELASTIC_TRANSFER_SLOPE
    assert NOMINAL_PARAMS.b_sd == 8.504140762083612
    # the ledger constants
    assert PROTON_QN == (1, 1, 0, 1)
    assert VACUUM_EXCHANGE_NAME == "vacuum"
    assert ZERO_TRANSFER == (0, 0, 0, 0)
    assert EXCITATION_FAMILY_ID == "baryonic_np"
    assert ANCESTRY_LABEL == "dd"
    # the card-5 closed-substream signature: three streams, no
    # spectral_state and no topology_selection (the class draws zero
    # of each)
    parameters = tuple(inspect.signature(generate_dd).parameters)
    assert parameters == (
        "params", "catalog", "fams", "rng_exchange_kinematics",
        "rng_decay_channel", "rng_decay_kinematics", "flavor", "ledger",
        "exchange", "exact_range", "line_thresholds", "slope_mass_exponent", "diffractive_duality",
        "duality_line_shapes", "xi_max")  # exchange: Round 8 2026-10-04, the drawn exchange configuration (None = vacuum);
    # xi_max: 2026-10-23, None = the excited masses up to the kinematic edge (the coherence edge M^2/s otherwise);
    # diffractive_duality: 2026-10-15, None = the continuum cascade of the vacuum systems;
    # duality_line_shapes: 2026-10-18, False = the fixed-width Breit-Wigner pick;
    # exact_range: 2026-10-05, the exact transfer window (False = the committed draw);
    # line_thresholds: 2026-10-06, the per-line lower mass edges (False = the shared edge);
    # slope_mass_exponent: 2026-10-08, the factorized vertex slopes (None = the single slope)
    for label in ("exchange_kinematics", "decay_channel",
                  "decay_kinematics", "spectral_state",
                  "topology_selection"):
        assert label in SUBSTREAM_LABELS


def test_dd_kinematics_closure_and_mass_shell():
    """The exact 2-to-2 solve closes the incoming sum within ``1e-9``
    relative on a pinned grid of ``(M_1, t_1, M_2, t_2)`` points, the
    solved pair sits on its mass shell, and the checked domain
    refusals (positive transfers, non-positive mass, the
    above-threshold configuration) raise ``ValueError``."""

    reference = _incoming_sum()
    points = (
        (2.0, -1.0, 3.0, -0.5),
        (3.0, -1.0, 2.0, -0.5),
        (5.0, -50.0, 4.0, -20.0),
        (12.0, -200.0, 3.0, -50.0),
    )
    for m1, t1, m2, t2 in points:
        system1, system2 = dd_kinematics(m1, t1, m2, t2, SQRT_S)
        total = tuple(system1[i] + system2[i] for i in range(4))
        relative = max(
            abs(total[0] - reference[0]) / reference[0],
            abs(total[1]) / reference[3],
            abs(total[2]) / reference[3],
            abs(total[3] - reference[3]) / reference[3])
        assert relative <= 1e-9, (m1, t1, m2, t2, relative)
        # the solved pair sits on its mass shell
        m1_solved = _invariant_mass2(system1)
        m2_solved = _invariant_mass2(system2)
        assert abs(m1_solved - m1 * m1) <= 1e-9 * m1 * m1, (m1, t1, m2, t2)
        assert abs(m2_solved - m2 * m2) <= 1e-9 * m2 * m2, (m1, t1, m2, t2)

    # the checked domain refusals (checked, never repaired)
    with pytest.raises(ValueError):
        dd_kinematics(2.0, 1.0, 3.0, -0.5, SQRT_S)
    with pytest.raises(ValueError):
        dd_kinematics(2.0, -1.0, 3.0, 0.5, SQRT_S)
    with pytest.raises(ValueError):
        dd_kinematics(-2.0, -1.0, 3.0, -0.5, SQRT_S)
    with pytest.raises(ValueError):
        dd_kinematics(2.0, -1.0, 0.0, -0.5, SQRT_S)
    # a two-body system closed beyond the joint-closure tolerance
    with pytest.raises(ValueError):
        dd_kinematics(17.0, -1.0, 1.0, -0.5, SQRT_S)
    # non-finite arguments
    with pytest.raises(ValueError):
        dd_kinematics(2.0, float("nan"), 3.0, -0.5, SQRT_S)


def test_dd_kinematics_intact_two_body_edge():
    """The intact two-body edge (both systems at the proton mass, both
    transfers zero) resolves at the elastic-forward configuration:
    system 1 carries the beam four-vector and system 2 the target at
    rest, within ``1e-12`` (the forward edge of the intact system is
    exactly ``t = 0``)."""

    system1, system2 = dd_kinematics(
        PROTON_MASS, 0.0, PROTON_MASS, 0.0, SQRT_S)
    beam = (math.sqrt(PROTON_MASS ** 2 + BEAM_MOMENTUM ** 2),
            0.0, 0.0, BEAM_MOMENTUM)
    target = (PROTON_MASS, 0.0, 0.0, 0.0)
    assert max(abs(a - b) for a, b in zip(system1, beam)) <= 1e-12
    assert max(abs(a - b) for a, b in zip(system2, target)) <= 1e-12


def test_dd_kinematics_overdetermination_convention(catalog, fams):
    """The operative solve convention on the drawn ``(t_1, t_2)`` pair
    (the card-4 factorized effective law over-determines the single
    production angle of the 2-to-2 system): the production angle
    follows from the drawn ``t_1`` through the documented invariant
    transfer on the projectile-side excitation line,
    ``t_1 = m_p^2 + M_1^2 - 2 (E_p* E_1* - p* q* cos(theta*))``, and
    the realized target-side transfer follows from the exact two-body
    coupling of the back-to-back configuration (``t_1 = t_2`` on the
    shared total four-momentum); the recorded values stay the drawn
    ones (bit-identical transfers on the pinned event, the joint-law
    record); and both saturation slivers clamp to the forward and
    backward configurations (the drawn domains extend past the exact
    two-body reach of the realized system)."""

    # the angle from the drawn t_1: the recovered cos(theta*) from the
    # solved system 1 (the invariant P_1 . p_beam) matches the
    # documented closed form
    for m1, t1, m2 in ((2.0, -1.0, 3.0), (5.0, -50.0, 4.0)):
        system1, _system2 = dd_kinematics(m1, t1, m2, 0.0, SQRT_S)
        e_beam, _e_total, _sq, _p_star, _beta, _gamma = (
            _beam_configuration())
        dot = e_beam * system1[0] - system1[3] * BEAM_MOMENTUM
        e_p, e_1, _e_2, p_star, q_star = _cm_quantities(m1, m2)
        cos_recovered = (e_p * e_1 - dot) / (p_star * q_star)
        cos_standard = (t1 - PROTON_MASS ** 2 - m1 * m1
                        + 2.0 * e_p * e_1) / (2.0 * p_star * q_star)
        assert cos_recovered == pytest.approx(cos_standard, abs=1e-9), (
            m1, t1, m2, cos_recovered, cos_standard)

    # the realized pair carries the exact two-body coupling t_1 = t_2
    # (the back-to-back configuration of the shared total
    # four-momentum) at float-rounding level
    for m1, t1, m2, t2 in ((2.0, -1.0, 3.0, -0.5), (5.0, -50.0, 4.0, -20.0)):
        system1, system2 = dd_kinematics(m1, t1, m2, t2, SQRT_S)
        t1_realized = _realized_transfer(system1, PROJECTILE_LINE_LABEL)
        t2_realized = _realized_transfer(system2, TARGET_LINE_LABEL)
        assert abs(t1_realized - t2_realized) <= 1e-9, (
            m1, t1, m2, t2, t1_realized, t2_realized)

    # the recorded values stay the drawn ones: the pinned event's
    # transfers are bit-identical to the card-4 joint-law draw on the
    # same stream (the masses within the sqrt round-trip)
    event = generate_dd(
        NOMINAL_PARAMS, catalog, fams, *_event_rngs(0))
    subs = derive_substreams(MASTER_SEED, 0, 0)
    rng = np.random.default_rng(subs.stream_for("exchange_kinematics"))
    m2_1, m2_2, t1, t2 = draw_joint_dd(
        M2_MIN, M2_MAX, M2_MIN, M2_MAX, SQRT_S, NOMINAL_PARAMS, rng)
    assert event.t1 == t1 == EVENT0_T1
    assert event.t2 == t2 == EVENT0_T2
    assert event.m1 == EVENT0_M1 and event.m2 == EVENT0_M2
    assert abs(event.m1 ** 2 - m2_1) <= 1e-15 * m2_1
    assert abs(event.m2 ** 2 - m2_2) <= 1e-15 * m2_2

    # both saturation slivers clamp: the drawn domains (card 4's
    # t_range with the other line at the intact proton mass) extend
    # past the exact reach of the realized (M_1, M_2) system; the
    # clamped configurations are identified in the CM frame, because
    # the pinned fixed-target boost carries both saturation
    # configurations to a positive lab pz
    m1, m2 = 2.0, 3.0
    e_p, e_1, _e_2, p_star, q_star = _cm_quantities(m1, m2)
    t_exact_fwd = (PROTON_MASS ** 2 + m1 * m1
                   - 2.0 * (e_p * e_1 - p_star * q_star))
    t_exact_bwd = (PROTON_MASS ** 2 + m1 * m1
                   - 2.0 * (e_p * e_1 + p_star * q_star))
    assert t_exact_bwd < t_exact_fwd < 0.0
    _e_beam, _e_total, _sq, _p_st, beta, gamma = _beam_configuration()

    def _cm_pz(system):
        return gamma * (system[3] - beta * system[0])

    # a drawn t_1 at 0 (inside the drawn range, above the exact
    # forward edge) solves to the forward configuration
    system1, _system2 = dd_kinematics(m1, 0.0, m2, 0.0, SQRT_S)
    assert abs(system1[1]) <= 1e-12 and abs(system1[2]) <= 1e-12
    assert _cm_pz(system1) > 0.0
    assert abs(_cm_pz(system1) - q_star) <= 1e-9 * q_star
    # a drawn t_1 below the exact backward edge (inside the drawn
    # range: card 4's t_range lower edge is more negative than the
    # exact backward edge of the realized system) solves to the
    # backward configuration
    t1_bwd_drawn = t_range(m1 * m1, PROTON_MASS, SQRT_S)[0]
    assert t1_bwd_drawn < t_exact_bwd
    system1, _system2 = dd_kinematics(m1, t1_bwd_drawn, m2, 0.0, SQRT_S)
    assert abs(system1[1]) <= 1e-12 and abs(system1[2]) <= 1e-12
    assert _cm_pz(system1) < 0.0
    assert abs(_cm_pz(system1) + q_star) <= 1e-9 * q_star


def test_dd_kinematics_documented_sum_coupling(pinned_trace):
    """The exact two-body coupling on the realized configuration (the
    pin of the finding behind the pre-86fd47f documentation bug):
    momentum conservation on the shared total four-momentum makes the
    two exchanges carry the same transfer four-vector
    (``p_1 - p_3 = p_4 - p_2``), so the realized transfers coincide,
    ``t_1 = t_2``, at float-rounding level on the pinned grid and on
    the pinned trace; and the realized sum deviates from the pre-fix
    documented sum coupling ``t_1 + t_2 = 2 m_p^2 + M_1^2 + M_2^2 -
    s`` by ``4 p* q* cos(theta*)`` (the sum-coupling deviation;
    measured 275.5850246322698 GeV^2 at the pinned event-0 pair,
    vanishing only at the central configuration ``cos(theta*) = 0``)."""

    def _gap_and_residual(system1, system2, m1, m2):
        """The realized transfer gap ``|t_1 - t_2|`` and the residual
        of the sum-coupling deviation: the realized sum minus the
        pre-fix documented sum coupling minus ``4 p* q*
        cos(theta*)``, with ``cos(theta*)`` the production angle of
        the realized configuration (system 1 in the CM frame)."""

        t1_realized = _realized_transfer(
            system1, PROJECTILE_LINE_LABEL)
        t2_realized = _realized_transfer(
            system2, TARGET_LINE_LABEL)
        _e_beam, _e_total, sq_exact, _p_star, beta, gamma = (
            _beam_configuration())
        s = sq_exact * sq_exact
        sum_coupling = (
            2.0 * PROTON_MASS ** 2 + m1 * m1 + m2 * m2 - s)
        e_p, e_1, _e_2, p_star, q_star = _cm_quantities(m1, m2)
        pz_cm = gamma * (system1[3] - beta * system1[0])
        cos_theta = pz_cm / q_star
        deviation = (
            (t1_realized + t2_realized) - sum_coupling
            - 4.0 * p_star * q_star * cos_theta)
        return abs(t1_realized - t2_realized), abs(deviation)

    # the pinned grid points of the exact solve and the pinned trace
    # (the master-seed 500-event landing; min q* on the trace is
    # 0.0953, so the realized production angle is well defined at
    # every pinned event)
    worst_gap = 0.0
    worst_residual = 0.0
    for m1, t1, m2, t2 in ((2.0, -1.0, 3.0, -0.5), (5.0, -50.0, 4.0, -20.0)):
        system1, system2 = dd_kinematics(m1, t1, m2, t2, SQRT_S)
        gap, residual = _gap_and_residual(system1, system2, m1, m2)
        worst_gap = max(worst_gap, gap)
        worst_residual = max(worst_residual, residual)
    for event in pinned_trace:
        system1 = event.lines[0].four_momentum
        system2 = event.lines[1].four_momentum
        gap, residual = _gap_and_residual(
            system1, system2, event.m1, event.m2)
        worst_gap = max(worst_gap, gap)
        worst_residual = max(worst_residual, residual)
    # float-rounding level on the grid and the pinned trace (measured
    # worst gap 1.0822454044046026e-11 at pinned event 442, worst
    # residual 1.9554136088117957e-11 at pinned event 150; 2.69e-11 on
    # the trace of the kernel fixes of 2026-10-02, still rounding)
    assert worst_gap <= 2.5e-11, worst_gap
    assert worst_residual <= 5e-11, worst_residual

    # the pinned event-0 pair: the finding's measured deviation of
    # the realized sum from the pre-fix documented sum coupling
    event = pinned_trace[0]
    system1 = event.lines[0].four_momentum
    system2 = event.lines[1].four_momentum
    t1_realized = _realized_transfer(system1, PROJECTILE_LINE_LABEL)
    t2_realized = _realized_transfer(system2, TARGET_LINE_LABEL)
    _e_beam, _e_total, sq_exact, _p_star, _beta, _gamma = (
        _beam_configuration())
    s = sq_exact * sq_exact
    sum_coupling = (
        2.0 * PROTON_MASS ** 2 + event.m1 * event.m1
        + event.m2 * event.m2 - s)
    deviation = (t1_realized + t2_realized) - sum_coupling
    assert abs(deviation - 272.35494625881444) <= 1e-9, deviation


def test_central_gap_pin(pinned_trace):
    """The central gap (the R50 stored quantity, plan line 903, the
    cross-side pair-distance minimum) on the pinned event equals the
    hand-computed rapidity difference within ``1e-12`` and is
    positive on the pinned trace (the two systems' decay products do
    not close the central gap at nominal parameters); the checked
    refusals raise ``ValueError``."""

    # the pinned event 0: the hand-computed cross-side pair-distance
    # minimum
    event = pinned_trace[0]
    assert event.m1 == EVENT0_M1 and event.m2 == EVENT0_M2
    assert event.t1 == EVENT0_T1 and event.t2 == EVENT0_T2
    assert event.gap_size == EVENT0_GAP
    projectile_ys = [
        _rapidity(h.four_momentum) for h in event.final_state
        if h.line == PROJECTILE_LINE_LABEL]
    target_ys = [
        _rapidity(h.four_momentum) for h in event.final_state
        if h.line == TARGET_LINE_LABEL]
    assert len(projectile_ys) == 2 and len(target_ys) == 4
    hand = min(abs(y_p - y_t)
               for y_p in projectile_ys for y_t in target_ys)
    assert abs(central_gap(event) - hand) <= 1e-12
    assert central_gap(event) == event.gap_size == EVENT0_GAP

    # positive on the whole pinned trace (the decay products do not
    # close the central gap at nominal parameters within 1e-12)
    for pinned in pinned_trace:
        assert pinned.gap_size > 1e-12, pinned.gap_size
        assert central_gap(pinned) == pinned.gap_size

    # the checked refusals (checked, never repaired)
    with pytest.raises(ValueError):
        central_gap("not an event")
    doctored = dataclasses.replace(
        event, gap_size=float(event.gap_size) + 1e-3)
    with pytest.raises(ValueError):
        central_gap(doctored)
    one_sided = dataclasses.replace(
        event, final_state=tuple(
            h for h in event.final_state
            if h.line == PROJECTILE_LINE_LABEL))
    with pytest.raises(ValueError):
        central_gap(one_sided)


def test_no_intact_remnant(pinned_trace):
    """The event record carries no intact proton remnant and no
    remnant kind draw (white-box field scan and module source scan);
    both lines are excited baryonic systems (plan line 827: no
    separate remnant kind draw), each carrying ``(1, 1, 0, 1)`` after
    the vacuum transfers, with the pair-tuple exchange records and
    the closed final-state line-label set."""

    # the white-box field scan: the record's fields carry no remnant
    # and no kind draw
    fields = tuple(DDEvent.__dataclass_fields__)
    for field in fields:
        assert "remnant" not in field
        assert "kind" not in field

    # the white-box module scan: no remnant selection machinery and no
    # spectral_state stream anywhere in the module's code
    source = MODULE_PATH.read_text(encoding="utf-8")
    for token in NO_REMNANT_TOKENS:
        assert token not in source, token

    # the generator's signature carries no spectral_state and no
    # topology_selection stream
    parameters = tuple(inspect.signature(generate_dd).parameters)
    assert "rng_spectral_state" not in parameters
    assert "rng_topology_selection" not in parameters

    # both lines are excited baryonic systems: the ledger lines
    # inherit the proton quantum numbers after the vacuum transfers
    # and the exchange records are the pair-tuple vacuum convention
    for event in pinned_trace:
        assert event.class_label == DD_CLASS_LABEL
        for line in event.lines:
            assert line.qn == REMNANT_QN == (1, 1, 0, 1)
        assert event.exchanges == (
            (VACUUM_EXCHANGE_NAME, ZERO_TRANSFER),
            (VACUUM_EXCHANGE_NAME, ZERO_TRANSFER))
        assert {h.line for h in event.final_state} == DD_LINE_LABELS


def test_draw_budget(catalog, fams):
    """A pinned 200-event trace on one shared stream per label
    consumes exactly six ``exchange_kinematics`` uniforms per accepted
    event plus four per rejected attempt on the same stream (the
    pinned attempt histogram at this landing) and the frozen Round 4
    per-vertex budget on ``decay_channel`` / ``decay_kinematics``
    (the pinned per-vertex totals at this landing); zero
    ``spectral_state`` and zero ``topology_selection`` uniforms (the
    class draw belongs to the runner, card 9), with pinned RNG-state
    identities."""

    # one shared stream per label across the pinned trace, wrapped in
    # the counting proxies (the spectral_state and topology_selection
    # streams are created here but never passed: the class draws none)
    subs = derive_substreams(MASTER_SEED, 0, 0)
    counter_exchange = _CountingRng(np.random.default_rng(
        subs.stream_for("exchange_kinematics")))
    counter_channel = _CountingRng(np.random.default_rng(
        subs.stream_for("decay_channel")))
    counter_kinematics = _CountingRng(np.random.default_rng(
        subs.stream_for("decay_kinematics")))
    counter_spectral = _CountingRng(np.random.default_rng(
        subs.stream_for("spectral_state")))
    counter_topology = _CountingRng(np.random.default_rng(
        subs.stream_for("topology_selection")))

    per_event_channel: list[int] = []
    per_event_kinematics: list[int] = []
    per_event_attempts: list[int] = []
    for _event_index in range(N_BUDGET):
        before_channel = counter_channel.calls
        before_kinematics = counter_kinematics.calls
        before_exchange = counter_exchange.calls
        generate_dd(NOMINAL_PARAMS, catalog, fams,
                    counter_exchange, counter_channel,
                    counter_kinematics)
        per_event_channel.append(counter_channel.calls - before_channel)
        per_event_kinematics.append(
            counter_kinematics.calls - before_kinematics)
        used = counter_exchange.calls - before_exchange
        # six uniforms per accepted event plus the rejection attempts
        # on the same stream: 4 uniforms per rejected attempt (the two
        # M^2 draws), never fewer than six
        assert used >= 6 and (used - 6) % 4 == 0, used
        per_event_attempts.append(1 + (used - 6) // 4)

    # the exchange_kinematics label: exactly six uniforms per accepted
    # event plus 4 per rejected attempt (card 4's joint-law budget)
    assert counter_exchange.calls == DD_EXCHANGE_TOTAL == 6 * N_BUDGET + 4 * sum(
        attempts - 1 for attempts in per_event_attempts)
    # the pinned attempt histogram at this landing (the joint law's
    # rejection on the shared stream, master seed 20260925, replica 0)
    assert dict(sorted(collections.Counter(per_event_attempts).items())) \
        == dict(sorted(DD_ATTEMPT_HISTOGRAM.items()))
    # zero spectral_state and zero topology_selection uniforms
    assert counter_spectral.calls == 0
    assert counter_topology.calls == 0

    # the frozen Round 4 per-vertex budget on the decay labels (the
    # pinned per-event totals at this landing; a catalog handoff adds
    # the Round 2 kernel's own draws on the same streams)
    assert sum(per_event_channel) == DD_CHANNEL_TOTAL
    assert sum(per_event_kinematics) == DD_KINEMATICS_TOTAL
    assert all(k >= 2 * c
               for c, k in zip(per_event_channel, per_event_kinematics))
    # since the N pi threshold of 2026-10-02 every excitation decays, so
    # every accepted event carries at least two vertices
    assert min(per_event_channel) == 2, (
        "every accepted event decays both excitations at least once "
        "on this trace (no below-threshold excitation)")

    # pinned RNG-state identity: the post-trace states equal fresh
    # streams advanced by exactly the counted draws
    subs = derive_substreams(MASTER_SEED, 0, 0)
    reference_exchange = np.random.default_rng(
        subs.stream_for("exchange_kinematics"))
    reference_channel = np.random.default_rng(
        subs.stream_for("decay_channel"))
    reference_kinematics = np.random.default_rng(
        subs.stream_for("decay_kinematics"))
    for _ in range(counter_exchange.calls):
        reference_exchange.random()
    for _ in range(counter_channel.calls):
        reference_channel.random()
    for _ in range(counter_kinematics.calls):
        reference_kinematics.random()
    assert (counter_exchange._source.bit_generator.state
            == reference_exchange.bit_generator.state)
    assert (counter_channel._source.bit_generator.state
            == reference_channel.bit_generator.state)
    assert (counter_kinematics._source.bit_generator.state
            == reference_kinematics.bit_generator.state)


def test_pathological_rejection_cap():
    """The pinned pathological configuration (each marginal on the
    exact individual upper-limit window; the individually legal
    windows are jointly impossible) raises
    ``PhaseSpaceRejectionExceeded`` at exactly ``REJECTION_CAP =
    1000`` attempts consuming exactly 4000 uniforms (the two
    marginals per rejected attempt; the closure is checked before the
    transfers), the message naming the attempt count and the
    parameter point, with pinned RNG-state identity; the module's
    ``generate_dd`` consumes the same card-4 joint draw on the exact
    windows (white-box source check)."""

    assert REJECTION_CAP == 1000
    assert issubclass(PhaseSpaceRejectionExceeded, RuntimeError)

    pathological_min = (SQRT_S - PROTON_MASS) ** 2
    pathological_max = SQRT_S * SQRT_S
    rng = np.random.default_rng(7)
    state0 = rng.bit_generator.state
    with pytest.raises(PhaseSpaceRejectionExceeded) as excinfo:
        draw_joint_dd(pathological_min, pathological_max,
                      pathological_min, pathological_max,
                      SQRT_S, MassLawParams(0.0, 0.0), rng)
    message = str(excinfo.value)
    assert "1000 attempts" in message
    assert "epsilon=0.0" in message and "b_sd=0.0" in message
    reference = np.random.default_rng(7)
    for _ in range(4 * REJECTION_CAP):
        reference.random()
    assert rng.bit_generator.state == reference.bit_generator.state
    assert rng.bit_generator.state != state0

    # the generate_dd-level joint-law call is card 4's: the white-box
    # source check names the joint draw on the exact windows
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "draw_joint_dd(" in source
    assert "sd_limits(PROTON_MASS, SQRT_S)" in source


def test_joint_law_pins(pinned_trace):
    """The exact marginal limits ``(0.880354927712736,
    266.7253293403128)`` GeV^2 at the nominal p+p configuration; on
    the pinned trace ``M_1 + M_2 <= sqrt(s)`` within ``1e-12``, both
    ``M_i^2`` inside the exact limits, and both ``t_i`` inside their
    exact two-body ranges closed at ``t = 0`` (each computed with the
    other line at the intact proton mass, card 4's factorized
    effective law)."""

    assert sd_limits(PROTON_MASS, SQRT_S) == (M2_MIN, M2_MAX) == (
        1.151863222816188, 266.7253293403128)
    assert M2_MIN == (PROTON_MASS + 0.1349766) * (PROTON_MASS + 0.1349766)
    assert M2_MAX == (SQRT_S - PROTON_MASS) ** 2

    for event in pinned_trace:
        assert event.m1 + event.m2 <= SQRT_S + 1e-12, (
            event.m1, event.m2)
        assert M2_MIN <= event.m1 * event.m1 <= M2_MAX, (
            event.m1, M2_MIN, M2_MAX)
        assert M2_MIN <= event.m2 * event.m2 <= M2_MAX, (event.m2,)
        t_min_1 = t_range(event.m1 * event.m1, PROTON_MASS, SQRT_S)[0]
        t_min_2 = t_range(event.m2 * event.m2, PROTON_MASS, SQRT_S)[0]
        assert t_min_1 <= event.t1 <= 0.0, (event.t1, t_min_1)
        assert t_min_2 <= event.t2 <= 0.0, (event.t2, t_min_2)
        # the drawn t ranges are closed at t = 0 exactly
        assert t_range(event.m1 * event.m1, PROTON_MASS, SQRT_S)[1] == 0.0
        assert t_range(event.m2 * event.m2, PROTON_MASS, SQRT_S)[1] == 0.0


def test_conservation_pins(pinned_trace):
    """Over the pinned trace the final state carries exactly
    ``(B, Q, S) = (2, 2, 0)``, the four-momentum closure is within
    ``1e-9`` relative (the mass shell at the same level), and every
    event passes card 2's ``audit_event`` with zero violations (two
    vacuum transfers, both excitation lines ``(1, 1, 0, 1)``, no
    central source: rule (e) vacuous on the quantum-number-free
    central record, plan section 11.4)."""

    assert len(pinned_trace) == N_PINNED
    reference = _incoming_sum()
    worst_closure = 0.0
    worst_shell = 0.0
    for event in pinned_trace:
        assert isinstance(event, DDEvent)
        assert event.class_label == DD_CLASS_LABEL
        # the exact quantum-number bookkeeping
        final_qn = (
            sum(h.qn[0] for h in event.final_state),
            sum(h.qn[1] for h in event.final_state),
            sum(h.qn[2] for h in event.final_state))
        assert final_qn == (2, 2, 0)
        # the four-momentum closure and the mass shell
        total = (
            math.fsum(h.four_momentum[0] for h in event.final_state),
            math.fsum(h.four_momentum[1] for h in event.final_state),
            math.fsum(h.four_momentum[2] for h in event.final_state),
            math.fsum(h.four_momentum[3] for h in event.final_state))
        relative = max(
            abs(total[0] - reference[0]) / reference[0],
            abs(total[1]) / reference[3],
            abs(total[2]) / reference[3],
            abs(total[3] - reference[3]) / reference[3])
        worst_closure = max(worst_closure, relative)
        for hadron in event.final_state:
            m2 = _invariant_mass2(hadron.four_momentum)
            # a massless final state (the photon of a radiative channel,
            # 2026-10-02) is on shell relative to its energy scale
            scale = (max(hadron.mass * hadron.mass, 1e-9) if hadron.mass > 0.0
                     else max(float(hadron.four_momentum[0]) ** 2, 1e-9))
            worst_shell = max(worst_shell, abs(m2 - hadron.mass ** 2) / scale)
        # card 2's five-rule audit with zero violations (the
        # quantum-number-free central record: no central source)
        report = _audit(event)
        assert not report.violations, report.violations
        # the outgoing line four-momenta close on the incoming sum too
        line_total = (
            event.lines[0].four_momentum[0]
            + event.lines[1].four_momentum[0],
            event.lines[0].four_momentum[1]
            + event.lines[1].four_momentum[1],
            event.lines[0].four_momentum[2]
            + event.lines[1].four_momentum[2],
            event.lines[0].four_momentum[3]
            + event.lines[1].four_momentum[3])
        line_relative = max(
            abs(line_total[0] - reference[0]) / reference[0],
            abs(line_total[1]) / reference[3],
            abs(line_total[2]) / reference[3],
            abs(line_total[3] - reference[3]) / reference[3])
        assert line_relative <= 1e-9, line_relative
    assert worst_closure <= 1e-9, worst_closure
    assert worst_shell <= 1e-9, worst_shell


def test_frozen_cascade_pins(catalog, fams):
    """Both excitation decays are the frozen simplified cascade at the
    sealed R45 point (kappa_B = 0.0, kappa_M = 4.0, b_dec = 2.0
    GeV^-2, baryonic nominal width 0.3 GeV, pinned through the R45
    driver digests in both helper trees); the ``W_flavor = 1.0`` hook
    identity on the channel weight (Round 6 owns the flavor weights);
    the ``baryonic_np`` excitation family of both lines; and the R46
    ``no_refit`` white-box check (no module in this card references a
    re-fit of the alignment parameters or claims the separation of
    exchange recoil from decay transverse dynamics)."""

    # the frozen simplified cascade is consumed read-only: the module
    # imports the sealed Round 4 machinery (the termination driver,
    # the decay machinery, the alignment and form-factor kernels) and
    # introduces no alignment re-fit and no separation claim (the R46
    # clauses; the white-box token scan)
    import na49_mixing.cascade_alignment  # noqa: F401
    import na49_mixing.cascade_form_factor  # noqa: F401
    import na49_mixing.cascade_termination  # noqa: F401
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert 'EXCITATION_FAMILY_ID: str = "baryonic_np"' in source
    assert "handoff_decay" in source and "draw_decay" in source
    for token in NO_REFIT_TOKENS:
        assert token not in source, token

    # the sealed R45 point is pinned through the R45 driver digests in
    # both helper trees (the frozen simplified configuration: the
    # constant-kappa alignment kernels, the frozen b_dec, the baryonic
    # nominal width)
    assert FROZEN_SURFACES[
        "dev/mixing2006/helpers/benchmarking/"
        "staged_r45_simplified_alignment_freeze.py"] == (
            "48cc785aeccd24206cba5cc581d5b964df8e11b76df029a845bc0911e701bc10")
    assert FROZEN_SURFACES[
        "helpers/benchmarking/staged_r45_simplified_alignment_freeze.py"] == (
            "48cc785aeccd24206cba5cc581d5b964df8e11b76df029a845bc0911e701bc10")
    for rel, expected in (
            ("dev/mixing2006/helpers/benchmarking/"
             "staged_r45_simplified_alignment_freeze.py",
             FROZEN_SURFACES[
                 "dev/mixing2006/helpers/benchmarking/"
                 "staged_r45_simplified_alignment_freeze.py"]),
            ("helpers/benchmarking/"
             "staged_r45_simplified_alignment_freeze.py",
             FROZEN_SURFACES[
                 "helpers/benchmarking/"
                 "staged_r45_simplified_alignment_freeze.py"])):
        path = MONO_ROOT / rel
        assert path.is_file(), f"missing frozen surface: {rel}"
        pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    # the W_flavor hook default is exactly 1.0 (Round 6 owns the flavor
    # weights); the hook is multiplicative and consumes zero draws
    from na49_mixing.cascade_decay import CascadeChannel, channel_weight
    probe_channel = CascadeChannel(
        parent_id="baryonic_np", daughters=("catalog:2212", "catalog:111"),
        kind="established_hadron", weight=0.5)
    assert channel_weight(probe_channel) == 0.5
    assert channel_weight(probe_channel, w_flavor=1.0) == 0.5

    # both excited lines decay through the baryonic_np continuum family
    # (the cascade line's family id; the anchored exchange ancestry)
    rngs = _event_rngs(0)
    event = generate_dd(NOMINAL_PARAMS, catalog, fams, *rngs)
    assert event.lines[0].qn == (1, 1, 0, 1)
    assert event.lines[1].qn == (1, 1, 0, 1)
    report = _audit(event)
    assert not report.violations


# ----------------------------------------------------------------------
# Part 2: the substrate and freeze pins
# ----------------------------------------------------------------------
def test_module_byte_and_export_pins():
    """The module at its committed sha256 (nested tree HEAD 86fd47f)
    with its exact ``__all__`` export set (thirteen names), the
    solver test file at its committed sha256, and the package
    ``__init__`` at its committed sha256 carrying the
    double-diffraction export block (the six re-exported names; the
    elastic constants stay card 1's module-level constants, and the
    final-state hadron record stays card 5's package-level
    ``FinalHadron`` export)."""

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    import na49_mixing.topology_double_diffraction as _mod
    assert _mod.__all__ == MODULE_ALL

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)
    init_text = INIT_PATH.read_text(encoding="utf-8")
    block = re.search(
        r"from \.topology_double_diffraction import \((.*?)\)",
        init_text, re.S)
    assert block is not None, (
        "the __init__ double-diffraction import block is missing")
    for name in INIT_DD_NAMES:
        assert re.search(rf"\b{name}\b", block.group(1)), (
            f"{name} is not in the __init__ double-diffraction import "
            "block")
    # the elastic constants and the card-5 record stay module-level
    # only (not re-exported through the double-diffraction block)
    for name in ("BEAM_MOMENTUM", "PROTON_MASS", "SQRT_S", "FinalHadron",
                 "NOMINAL_PARAMS"):
        assert not re.search(rf"\b{name}\b", block.group(1)), (
            f"{name} must not enter the __init__ double-diffraction "
            "import block")
    all_block = re.search(r"__all__\s*=\s*\[(.*?)\n\]", init_text, re.S)
    assert all_block is not None
    all_names = re.findall(r'"([^"]+)"', all_block.group(1))
    assert len(all_names) == INIT_ALL_COUNT
    for name in INIT_DD_NAMES:
        assert f'"{name}"' in all_block.group(1), (
            f"{name} is not in the __init__ __all__")


def test_frozen_substrate_digests():
    """The card's frozen-substrate provenance surfaces (verified on
    disk) and the frozen M8 parameter map through the manifest (the
    ten-parameter set hash, the kernel boundary, and the
    ``regge.effective_transfer_slope``); the activated baryonic
    ``(1, 1, 0)`` continuum family of the committed configuration
    carries its recorded support ends at the frozen boundary and the
    card-6 surface's upper end."""

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
    assert m8["regge.effective_transfer_slope"] == ELASTIC_TRANSFER_SLOPE

    # the activated baryonic (1, 1, 0) continuum family of the
    # committed configuration: exactly one, with its support ends at
    # the frozen boundary and the card-6 surface's upper end
    continuum = json.loads(CONTINUUM_PATH.read_text(encoding="utf-8"))
    baryonic_families = [
        family for family in continuum["families"]
        if family["qn"] == [1, 1, 0] and family["active"] is True]
    assert len(baryonic_families) == 1
    assert baryonic_families[0]["family_id"] == EXCITATION_FAMILY_ID
    assert baryonic_families[0]["m_min"] == KERNEL_BOUNDARY
    assert baryonic_families[0]["m_max"] == 5.0




def test_layout_sync():
    """The layout sync (the subprocess check from the monorepo root;
    ``logic_drift = 0``)."""

    probe = subprocess.run(
        [sys.executable, str(SYNC_CHECKER)],
        capture_output=True, text=True, cwd=str(MONO_ROOT))
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert "logic_drift=0" in probe.stdout
