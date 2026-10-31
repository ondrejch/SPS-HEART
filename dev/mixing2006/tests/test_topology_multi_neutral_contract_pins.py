"""Round 5, card 8 test-author register: the multiple-neutral-excitation
class.

Independent contract pins for the bounded multi-neutral-excitation
class (``na49_mixing/topology_multi_neutral.py``), Round 5 card 8
(``TASK-20260925-r5-multiple-neutral-excitations``; plan section 22
task 32, line 1757; plan section 11.1 item 6, line 829), at nested tree
SHA bb43481.  This register is the test-author half of the card: it
pins the module contract on its own pinned fixtures (the frozen
``NDMultiEvent`` record surface with the class label ``"nd_multi"``,
the exact residual limits of the second-source law, the exact
four-system kinematics solve with the central pair at rest in the
overall CM and the two baryonic systems back-to-back against it at the
pair invariant ``sqrt_s - M_c1 - M_c2`` with the production angle from
the drawn projectile-side transfer, the pinned event-0 record, the
draw-budget pins on the closed substream labels with the bounded count
draw on ``topology_selection`` and the pinned rejection cap, the
central-neutrality pins over every central source with the
planted-violation detection, the conservation pins with the pinned
central-channel census and zero audit violations, the frozen-cascade
pins at the sealed R45 point with the R46 ``no_refit`` white-box
check, the byte-stable module / solver test / package ``__init__``
digests and export surface, and the frozen-substrate provenance) and
re-pins the freeze manifest for the card (``INVENTORY_COUNT`` 525 ->
526).

Part 1: the module contract (pure, hermetic)
--------------------------------------------

- ``test_named_constants_and_record_surface``: the class label
  ``"nd_multi"``, the bounded-count constants (``COUNT_BND = 0.5``,
  ``DEFAULT_P2 = 0.0``, ``P_N2_TOLERANCE = 1e-12``, the rejection cap
  ``REJECTION_CAP = 1000``), the unretained registry entry, the frozen
  record surfaces (the thirteen-field ``NDMultiEvent`` and the
  five-field ``FinalHadron``), the nominal parameters (``epsilon = 0``,
  ``B_SD`` the M8-frozen slope, the nominal R50 remnant split per
  line), the closed-substream ``generate_nd_multi`` signature (the
  count draw on ``topology_selection`` first, then card 7's four
  exchange/decay streams, plus the bounded ``p_n2`` parameter), and
  the ``validate_p_n2`` bound check with its refusal and clamp edges.
- ``test_nd_multi_limits_pins``: the exact residual limits
  ``(0.01821868254756, 207.17155777605097)`` GeV^2 of the intact
  configuration at a first-source mass of 1.0 GeV, the lower limit at
  ``m_pi0^2`` exactly and the upper limit at
  ``(sqrt_s - M_1 - M_2 - M_c1)^2`` within ``1e-12`` relative, and the
  checked refusal of an empty residual window and of a non-positive
  mass.
- ``test_nd_multi_kinematics_pins``: the exact four-system solve on a
  pinned grid of ``(M_1, t_1, M_2, t_2, M_c1, M_c2)`` points: the four
  momenta at their committed values, the incoming-sum closure within
  ``1e-9`` relative, the mass shells, the central pair at rest in the
  overall CM, the pair invariant at ``sqrt_s - M_c1 - M_c2``, the
  back-to-back coupling in the CM, the realized projectile-side
  transfer at the drawn value within ``1e-9``, and the checked
  refusal of a positive drawn ``t_2``.
- ``test_event_record_pins``: the pinned event-0 record at its
  committed values (the count ``n = 1``, the remnant kinds and masses,
  the single central mass, the two drawn transfers, the exact
  ``t_c = t_1 + t_2`` identity, ``t_2c = 0.0``), the ledger lines at
  ``(1, 1, 0, 1)``, the central source at ``(0, 0, 0, 0)``, the two
  vacuum exchange records with the zero transfer, and the bit
  identity of the shared fields against card 7's ``generate_nd`` on
  the same trace.
- ``test_draw_budget_pins``: a pinned 200-event trace on one substream
  quintuple per event index (450,000 + i, ``P(n = 2) = 0.0``) consumes
  exactly one ``topology_selection`` uniform (the count draw), two
  ``spectral_state`` uniforms, and four ``exchange_kinematics``
  uniforms per event, plus the frozen Round 4 per-vertex budget on
  ``decay_channel`` / ``decay_kinematics`` (the pinned per-event
  totals and min/max at this landing); a pinned 200-event trace at
  ``P(n = 2) = 0.5`` (460,000 + i) consumes exactly
  ``4 + 1 + 2 * 1`` ``exchange_kinematics`` uniforms per ``n = 2``
  event (the ``t_2c`` draw plus one two-uniform ``M_2c^2`` attempt)
  with pinned RNG-state identity across two runs, and the pinned
  rejection cap fires at exactly ``REJECTION_CAP = 1000`` consecutive
  rejected attempts with the pinned exchange count.
- ``test_central_neutrality_pins``: on the pinned 10,000-event trace
  every central source of every event carries ``(B, Q, S, 2I3) =
  (0, 0, 0, 0)`` exactly and the central line's joint closure holds;
  ``check_multi_central_neutrality`` raises on each planted violation
  (central source ``B = 1``, one open central joint vertex), naming
  the offending source or the open vertex, and refuses a foreign
  record type.
- ``test_conservation_pins``: over the pinned 10,000-event trace at
  ``P(n = 2) = 0.25`` the final state carries exactly ``(B, Q, S) =
  (2, 2, 0)``, the four-momentum closure is within ``1e-9`` relative
  at the pinned worst value, ``t_c = t_1 + t_2`` exactly on every
  event, the ``n = 2`` census is pinned, and every event passes card
  2's ``audit_event`` with zero violations.
- ``test_central_channel_pins``: on the pinned trace the
  below-threshold quantum-number-free central states are pinned (the
  ``n = 1`` count and the ``n = 2`` event and state counts), the
  hyperons appear only on the baryonic lines (the pinned census), the
  central kaon events are net neutral in charge and strangeness (the
  pinned count), and the K K-bar direct channel events are pinned.
- ``test_frozen_cascade_pins``: every central source decays through
  the frozen simplified cascade at the sealed R45 point (kappa_B =
  0.0, kappa_M = 4.0, b_dec = 2.0 GeV^-2, baryonic nominal width
  0.3 GeV, pinned through the R45 driver digests in both helper
  trees), the ``W_flavor = 1.0`` hook identity on the channel weight,
  and the R46 ``no_refit`` white-box check (no module in this card
  references a re-fit of the alignment parameters or claims the
  separation of exchange recoil from decay transverse dynamics).

Part 2: the substrate and freeze pins
--------------------------------------

- ``test_module_byte_and_export_pins``: the module at its committed
  sha256, its ``__all__`` export set (eleven names), the solver test
  file at its committed sha256 after the 525 -> 526 count re-pin, and
  the package ``__init__`` at its committed sha256 carrying the
  multiple-neutral-excitation export block (the eleven re-exported
  names; the package ``__all__`` total at 242 entries).
- ``test_frozen_substrate_digests``: the card's frozen-substrate
  provenance surfaces (verified on disk) and the frozen M8 parameter
  map through the manifest, with the activated neutral-mesonic
  ``(0, 0, 0)`` continuum family's recorded support ends.
- ``test_freeze_repin_register``: the freeze-manifest re-pin for
  this card (the strict path-sorted inventory at 526 entries; 525 at
  the card 8 solver landing; 523 at the card 7 test-author landing;
  522 after the Round 5 card 7 landing; 520 after the Round 5 card 6
  landing; 519 after the Round 5 card 6 test-author re-pin; 517 after
  the Round 5 card 5 test-author re-pin; 516 after the Round 5 card 5
  re-pin; 514 after the Round 5 card 4 test-author re-pin; 513 after
  the Round 5 card 4 re-pin; 511 after the Round 5 card 3
  test-author re-pin; 510 after the Round 5 card 3 solver landing;
  508 after the test-protocol tiering change; 505 after the Round 5
  card 2 test-author re-pin; 504 after the Round 5 card 2 solver
  landing; 502 after the Round 5 card 1 test-author re-pin; 501 at
  the Round 5 card 1 landing; +1 entry: this register) carries this
  register and the thirty-four re-pinned chain count-pin files at
  their on-disk digests, the card 8 solver's freeze exception stays
  exactly once at its 37 path-sorted files with its reason naming the
  523 -> 525 re-pin, this card's test-author freeze exception names
  the card exactly once with the path-sorted file list and a reason
  naming the 525 -> 526 re-pin, the chain count pins are re-pinned
  525 -> 526 with the 525 literal retained in the naming comments,
  and the frozen M8 parameter map stays intact.
- ``test_layout_sync``: the layout sync (the subprocess check from
  the monorepo root; ``logic_drift = 0``).

Hermetic: the module, the committed catalog, the continuum families
configuration, the Round 4 cascade modules (consumed read-only), the
committed cards 1-7 modules (consumed read-only), numpy, and the
standard library only; no event generation driver, no external
tables, no Monte Carlo run (the runs R50-R55 exercise the class
statistically).
"""

from __future__ import annotations

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

from na49_mixing.cascade_decay import CascadeChannel, channel_weight
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
    ExchangeRecord,
)
from na49_mixing.topology_mass_laws import (
    PI0_MASS,
    MassLawParams,
    PhaseSpaceRejectionExceeded,
    REJECTION_CAP,
    t_range,
)
from na49_mixing.topology_remnants import (
    KIND_INTACT_PROTON,
    KIND_LOW_MASS_BARYONIC_RESONANCE,
    KERNEL_BOUNDARY,
)
from na49_mixing.topology_non_diffractive import (
    CENTRAL_LINE_LABEL,
    CENTRAL_QN,
    ND_LINE_LABELS,
    NOMINAL_ND_REMNANT_WEIGHTS,
    NOMINAL_PARAMS,
    PROJECTILE_LINE_LABEL,
    TARGET_LINE_LABEL,
    CentralNeutralityViolation,
    _beam_configuration,
    _incoming_lines,
    generate_nd,
)
from na49_mixing.topology_multi_neutral import (
    COUNT_BND,
    DEFAULT_P2,
    ND_MULTI_CLASS_LABEL,
    NDMultiEvent,
    P_N2_TOLERANCE,
    REGISTRY_ENTRY,
    check_multi_central_neutrality,
    generate_nd_multi,
    nd_multi_kinematics,
    nd_multi_limits,
    validate_p_n2,
)
from na49_mixing.topology_single_diffraction import FinalHadron
from tests._freeze_inventory import inventory_count as _inventory_count

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent          # dev/mixing2006
MONO_ROOT = NESTED_ROOT.parent.parent          # the monorepo root
MANIFEST_PATH = (
    NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json")
SYNC_CHECKER = (
    MONO_ROOT / "helpers" / "benchmarking" / "check_helpers_layout_sync.py")
MODULE_PATH = NESTED_ROOT / "na49_mixing" / "topology_multi_neutral.py"
SOLVER_TEST_PATH = (
    NESTED_ROOT / "tests" / "test_topology_multi_neutral.py")
INIT_PATH = NESTED_ROOT / "na49_mixing" / "__init__.py"
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
CONTINUUM_PATH = (
    NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json")

#: The Round 5 master seed (round 5 index item 9), fixed for the pinned
#: traces; replica 0, one substream quintuple per event index on the
#: closed labels (``topology_selection``, ``spectral_state``,
#: ``exchange_kinematics``, ``decay_channel``, ``decay_kinematics``).
MASTER_SEED = 20260925

#: The pinned trace sizes (the 10,000-event conservation/neutrality/
#: channel trace at P(n = 2) = 0.25 on the event indices 0-9,999, the
#: 200-event n = 1 budget trace on the dedicated event indices
#: 450,000 + i at P(n = 2) = 0.0, the 200-event n = 2 budget trace on
#: the dedicated event indices 460,000 + i at P(n = 2) = 0.5, and the
#: pinned cap event index 470,002, all disjoint from the solver
#: test's class traces).
N_PINNED = 10_000
N_N1_BUDGET = 200
N_N2_BUDGET = 200
CAP_EVENT_INDEX = 470_002
P_N2_PINNED = 0.25

#: The exact residual limits of the intact configuration at a
#: first-source mass of 1.0 GeV (the pinned grid configuration of the
#: second-source law at ``SQRT_S = 17.270``).
RESIDUAL_M2_MIN = 0.01821868254756
RESIDUAL_M2_MAX = 207.17155777605097

#: The pinned kinematics grid point ``(M_1, t_1, M_2, t_2, M_c1,
#: M_c2)`` of the four-system solve (both lines at the intact proton
#: mass, the drawn transfers inside their exact two-body ranges, the
#: central sources at 0.9 and 0.8 GeV) and its solved lab-frame
#: four-momenta (projectile system, target system, central source 1,
#: central source 2).
GRID_T1 = -0.35
GRID_T2 = -0.42
GRID_M_C1 = 0.9
GRID_M_C2 = 0.8

#: The pinned first-source mass of the card's exact-residual-limits
#: configuration (the solver test's ``nd_multi_limits(PROTON_MASS,
#: PROTON_MASS, 1.0, SQRT_S)`` call).
LIMITS_M_C1 = 1.0
GRID_SYSTEM1 = (142.17084217254157, 0.55325239349136, 0.0,
                142.16666951473226)
GRID_SYSTEM2 = (1.1247853277400235, -0.55325239349136, 0.0,
                0.28053323311606543)
GRID_CENTRAL1 = (8.28287508625037, 0.0, 0.0, 8.233833839374405)
GRID_CENTRAL2 = (7.362555632222551, 0.0, 0.0, 7.31896341277725)

#: The pinned event-0 record (master seed 20260925, replica 0, event
#: index 0 at P(n = 2) = 0.25; the count draw selects n = 1 and the
#: rest of the record is card 7's event-0 ND record).
EVENT0_COUNT = 1
EVENT0_REMNANT_0_KIND = "intact_proton"
EVENT0_REMNANT_0_MASS = 0.93827231
EVENT0_REMNANT_1_KIND = "low_mass_baryonic_resonance"
EVENT0_REMNANT_1_MASS = 1.535
EVENT0_CENTRAL_MASS = 0.2431513153059654
EVENT0_T1 = -0.13445599813262105
EVENT0_T2 = -0.35553908007869217
EVENT0_T_C = -0.4899950782113132
EVENT0_N_FINAL = 4

#: The pinned census of the 10,000-event trace at P(n = 2) = 0.25
#: (the n = 2 event count and the first n = 2 event index, the
#: below-threshold quantum-number-free central states at n = 1 and on
#: the n = 2 central lines, the total hyperons and the central
#: hyperons, the events carrying central kaons, and the events
#: carrying a K K-bar pair on the central line).
N_N2_EVENTS = 2_494
FIRST_N2_INDEX = 6
N1_BELOW_THRESHOLD = 3_391
N2_BELOW_EVENTS = 1_692
N2_BELOW_STATES = 2_186
HYPERON_TOTAL = 225
CENTRAL_HYPERONS = 0
# re-pinned 2026-09-29 for the Block B stage 1 channel pool (the heavier direct decay classes)
# re-pinned 2026-10-02 for the kernel fixes (the CM-mirrored target diffraction, the N pi threshold, the radiative channels)
EVENTS_CENTRAL_KAONS = 92
EVENTS_KKBAR = 66

#: The pinned draw-budget totals of the 200-event n = 1 budget trace
#: (master seed 20260925, replica 0, one substream quintuple per event
#: index 450,000 + i in event order, P(n = 2) = 0.0): the frozen Round
#: 4 per-vertex decay budget totals on the ``decay_channel`` /
#: ``decay_kinematics`` streams at this landing (a catalog handoff and
#: the unstable-sibling kernels add the Round 2 kernel's own
#: documented draws on the same streams; a stable remnant record and a
#: below-threshold central source consume zero).
# re-pinned 2026-09-30 for the corrected final state (decays at every assembly point, the isospin rule)
N1_BUDGET_CHANNEL_TOTAL = 389
N1_BUDGET_KINEMATICS_TOTAL = 948
N1_BUDGET_ZERO_DRAW_EVENTS = 30
N1_BUDGET_MIN_CHANNEL = 0
N1_BUDGET_MAX_CHANNEL = 5
N1_BUDGET_MIN_KINEMATICS = 0
N1_BUDGET_MAX_KINEMATICS = 16

#: The pinned draw-budget totals of the 200-event n = 2 budget trace
#: (event indices 460,000 + i, P(n = 2) = 0.5): the n = 2 event count
#: (every n = 2 event draws exactly one accepted M_2c^2 attempt, so
#: its exchange budget is 4 + 1 + 2 * 1 = 7 uniforms) and the
#: exchange-kinematics total of the trace.
N2_BUDGET_N2_EVENTS = 97
N2_BUDGET_EXCHANGE_TOTAL = 1_091

#: The committed digests of the card's pinned surfaces (nested tree
#: HEAD bb43481; the solver test digest is the post-re-pin value, the
#: 525 -> 526 count re-pin of this register's landing).
MODULE_SHA256 = (
    "51c3bfd8898bf155c4f3d32138f4b59ccd72cb5110a5178c0ed9bf899da3bb15")
#: re-pinned by the Round 5 card 9 526 -> 528 count re-pin, which
#: re-touched this solver test file (TASK-20260925-r5-topology-template-runner).
#: re-pinned by the Round 5 card 9 test-author 528 -> 529 count re-pin, which
#: re-touched this solver test file (TASK-20260925-r5-topology-template-runner-test-author).
SOLVER_TEST_SHA256 = (
    "261f845e5a46f6275c0a408d8418ff216f54a9cffbc70a66235aba01f82b4253")
# re-pinned by Round 5 card 9 (TASK-20260925-r5-topology-template-runner; the topology-runner module's additive package exports).
# re-pinned by the Round 5 card 9 review fix round (TASK-20260925-r5-topology-template-runner-review-fix; the REV-bc4de47-03 export-surface correction re-pinned the package __init__).
INIT_SHA256 = (
    "7d2c5a85d9b24de104c0d9cf72195689bf7c39c873e154798699448073b36de8")

#: The module's ``__all__`` export set (eleven names).
MODULE_ALL = [
    "ND_MULTI_CLASS_LABEL",
    "COUNT_BND",
    "DEFAULT_P2",
    "P_N2_TOLERANCE",
    "NDMultiEvent",
    "REGISTRY_ENTRY",
    "check_multi_central_neutrality",
    "generate_nd_multi",
    "nd_multi_kinematics",
    "nd_multi_limits",
    "validate_p_n2",
]

#: The ``__init__`` multiple-neutral-excitation export block (the
#: eleven re-exported names; ``NOMINAL_PARAMS``, ``BEAM_MOMENTUM``,
#: ``PROTON_MASS``, ``SQRT_S``, and the mass-law parameters stay
#: card 1/card 4's package-level exports and are module-level only,
#: and the final-state hadron record stays card 5's package-level
#: ``FinalHadron`` export).
INIT_MULTI_NAMES = [
    "COUNT_BND",
    "DEFAULT_P2",
    "ND_MULTI_CLASS_LABEL",
    "NDMultiEvent",
    "P_N2_TOLERANCE",
    "REGISTRY_ENTRY",
    "check_multi_central_neutrality",
    "generate_nd_multi",
    "nd_multi_kinematics",
    "nd_multi_limits",
    "validate_p_n2",
]

#: The package ``__all__`` total entry count (the additive card 8
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
#: trees, and the six committed Round 5 modules consumed read-only
#: (card 1's elastic reference, card 2's exchange ledger, card 3's
#: remnants, card 4's mass laws, card 5's single diffraction, card 7's
#: non-diffractive class).
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
    "dev/mixing2006/na49_mixing/topology_non_diffractive.py":
        "a62d578f4c7c6f0179cb48f4b92670ff631089e73b9d64592e52a93453d367e1",
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
#: The solver card (Round 5, card 8) and this test-author register's
#: card name (the freeze-exception identity).
SOLVER_CARD = "TASK-20260925-r5-multiple-neutral-excitations"
TEST_AUTHOR_CARD = (
    "TASK-20260925-r5-multiple-neutral-excitations-test-author")

#: This register's relative path in the monorepo (the inventory key).
MY_TEST_REL = (
    "dev/mixing2006/tests/test_topology_multi_neutral_contract_pins.py")

#: The register chain: the thirty-four chain count-pin files re-pinned
#: 525 -> 526 by this register (the card-8 solver's thirty-three chain
#: files plus the card-8 solver test file, path-sorted).
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
    "dev/mixing2006/tests/test_topology_double_diffraction_contract_pins.py",
    "dev/mixing2006/tests/test_topology_elastic_contract_pins.py",
    "dev/mixing2006/tests/test_topology_mass_laws_contract_pins.py",
    "dev/mixing2006/tests/test_topology_multi_neutral.py",
    "dev/mixing2006/tests/test_topology_non_diffractive.py",
    "dev/mixing2006/tests/test_topology_non_diffractive_contract_pins.py",
    "dev/mixing2006/tests/test_topology_remnants_contract_pins.py",
    "dev/mixing2006/tests/test_topology_single_diffraction_contract_pins.py",
)

#: This register's freeze-exception file list (path-sorted): the
#: manifest, the thirty-four re-pinned chain count-pin files, and this
#: register.
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


def _invariant_mass(four_momentum) -> float:
    """The invariant mass of a four-momentum."""

    e, px, py, pz = four_momentum
    return math.sqrt(max(0.0, e * e - px * px - py * py - pz * pz))


def _cm_boost() -> float:
    """The exact fixed-target CM velocity of the pinned beam
    configuration (the boost from the lab to the overall CM)."""

    e_beam, _e_total, _sq, _p_star, beta, _gamma = _beam_configuration()
    return beta


def _to_cm(four_momentum) -> tuple[float, float, float, float]:
    """The four-momentum in the overall CM frame (the exact forward
    boost with the pinned CM velocity)."""

    e, px, py, pz = four_momentum
    beta = _cm_boost()
    gamma = 1.0 / math.sqrt(1.0 - beta * beta)
    return (gamma * (e - beta * pz), px, py, gamma * (pz - beta * e))


def _event_rngs(event_index: int):
    """The closed substream generators of one pinned event (the
    five-label quintuple of this card)."""

    subs = derive_substreams(MASTER_SEED, 0, event_index)
    return tuple(
        np.random.default_rng(subs.stream_for(label))
        for label in ("topology_selection", "spectral_state",
                      "exchange_kinematics", "decay_channel",
                      "decay_kinematics"))


def _nd_event_rngs(event_index: int):
    """The card-7 four-label generators of one pinned event (the same
    event index; the n = 1 bit-identity comparison's reference side)."""

    subs = derive_substreams(MASTER_SEED, 0, event_index)
    return tuple(
        np.random.default_rng(subs.stream_for(label))
        for label in ("spectral_state", "exchange_kinematics",
                      "decay_channel", "decay_kinematics"))


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


def _exchange_records(event: NDMultiEvent) -> tuple[Exchange, Exchange]:
    """The card-2 ledger exchange records of the event's two vacuum
    exchanges (the test-side conversion of the card-1 record form)."""

    records = []
    for record in event.exchanges:
        records.append(Exchange(
            name=record.name,
            transfer=(record.transfer[0], record.transfer[1],
                      record.transfer[2], record.transfer[3])))
    return (records[0], records[1])


def _line_daughters(event: NDMultiEvent, line: str) -> tuple[LineState, ...]:
    """The card-2 ledger daughters of one line of origin."""

    return tuple(
        LineState(B=h.qn[0], Q=h.qn[1], S=h.qn[2], I3x2=h.qn[3])
        for h in event.final_state if h.line == line)


def _decay_vertices(event: NDMultiEvent):
    """The card-2 ledger decay vertices of the event: each baryonic
    line as the parent and its final-state hadrons as the daughters,
    plus the joint central closure over the central sources with the
    central-line daughters (the per-source cascade vertex closes
    locally through the cascade machinery's own conservation check
    during generation)."""

    vertices = []
    for line, source in ((PROJECTILE_LINE_LABEL, event.lines[0]),
                         (TARGET_LINE_LABEL, event.lines[1])):
        daughters = _line_daughters(event, line)
        if not daughters:
            raise ValueError(
                f"the event carries no {line!r}-side final state")
        vertices.append((source, daughters))
    central_daughters = _line_daughters(event, CENTRAL_LINE_LABEL)
    if not central_daughters:
        raise ValueError("the event carries no central final state")
    joint = LineState(B=0, Q=0, S=0, I3x2=0)
    vertices.append((joint, central_daughters))
    return tuple(vertices)


def _audit(event: NDMultiEvent):
    """Card 2's five-rule audit over the event's ledger records."""

    return audit_event(
        incoming_lines=_incoming_lines(),
        exchanges=_exchange_records(event),
        outgoing_lines=event.lines,
        central_source=event.central_sources[0],
        decay_vertices=_decay_vertices(event),
    )


def _relative_closure(event: NDMultiEvent) -> float:
    """The worst relative four-momentum closure component of one event
    against the pinned incoming lab sum."""

    total = [math.fsum(h.four_momentum[j] for h in event.final_state)
             for j in range(4)]
    reference = _incoming_sum()
    e_total = reference[0]
    p_beam = reference[3]
    return max(abs(total[0] - reference[0]) / e_total,
               abs(total[1]) / p_beam,
               abs(total[2]) / p_beam,
               abs(total[3] - reference[3]) / p_beam)


@pytest.fixture(scope="module")
def catalog() -> HadronCatalog:
    return HadronCatalog.from_json(CATALOG_PATH)


@pytest.fixture(scope="module")
def fams():
    return families_from_json(str(CONTINUUM_PATH))


@pytest.fixture(scope="module")
def pinned_trace(catalog, fams):
    """The pinned 10,000-event multi-neutral-excitation trace (master
    seed 20260925, replica 0, P(n = 2) = 0.25, one substream quintuple
    per event index 0-9,999)."""

    events = []
    for event_index in range(N_PINNED):
        rngs = _event_rngs(event_index)
        events.append(generate_nd_multi(
            NOMINAL_PARAMS, NOMINAL_ND_REMNANT_WEIGHTS, P_N2_PINNED,
            catalog, fams, *rngs))
    return events


# ----------------------------------------------------------------------
# Part 1: the module contract (pure, hermetic)
# ----------------------------------------------------------------------
def test_named_constants_and_record_surface():
    """The class label, the bounded-count constants, the unretained
    registry entry, the frozen record surfaces, the nominal parameters
    (the R50 fixed values: ``epsilon = 0``, ``B_SD`` the M8-frozen
    slope, the nominal R50 remnant split per line), the closed
    substream labels, the ``validate_p_n2`` bound check with its
    refusal and clamp edges, and the closed-substream
    ``generate_nd_multi`` signature."""

    assert ND_MULTI_CLASS_LABEL == "nd_multi"
    assert COUNT_BND == 0.5
    assert DEFAULT_P2 == 0.0
    assert P_N2_TOLERANCE == 1e-12
    assert REJECTION_CAP == 1000
    # the unretained registry entry (the card's pre-registered form)
    assert REGISTRY_ENTRY == {
        "class": "nd_multi",
        "module": "na49_mixing.topology_multi_neutral",
        "retained": False,
        "count_bounded": True,
        "p_n2_bound": COUNT_BND,
        "default_p_n2": DEFAULT_P2,
    }
    # the frozen record surfaces
    assert getattr(NDMultiEvent, "__dataclass_params__").frozen
    assert tuple(NDMultiEvent.__dataclass_fields__) == (
        "class_label", "count", "p_n2", "remnants", "central_masses",
        "t_c", "t1", "t2", "t_2c", "lines", "central_sources",
        "exchanges", "final_state")
    assert getattr(FinalHadron, "__dataclass_params__").frozen
    assert tuple(FinalHadron.__dataclass_fields__) == (
        "pdg_id", "four_momentum", "mass", "qn", "line")
    # the nominal parameters: the R50 fixed values (epsilon = 0, B_SD
    # the M8-frozen slope; the nominal remnant split per line)
    assert NOMINAL_PARAMS.epsilon == 0.0
    assert NOMINAL_PARAMS.b_sd == ELASTIC_TRANSFER_SLOPE
    assert NOMINAL_ND_REMNANT_WEIGHTS == {
        KIND_INTACT_PROTON: 0.5,
        KIND_LOW_MASS_BARYONIC_RESONANCE: 0.5,
        "high_mass_baryonic_continuum": 0.0,
    }
    # the closed substream labels (the count draw on
    # topology_selection, then card 7's four exchange/decay streams)
    for label in ("topology_selection", "spectral_state",
                  "exchange_kinematics", "decay_channel",
                  "decay_kinematics"):
        assert label in SUBSTREAM_LABELS
    # the closed-substream generator signature (the p_n2 parameter and
    # the five per-label substream generators)
    signature = inspect.signature(generate_nd_multi)
    assert tuple(signature.parameters) == (
        "params", "weights", "p_n2", "catalog", "fams",
        "rng_topology_selection", "rng_spectral_state",
        "rng_exchange_kinematics", "rng_decay_channel",
        "rng_decay_kinematics", "flavor", "ledger")
    # the bound check: the pinned edges (0.5 and 0.5 + 1e-12 legal,
    # above the edge refused, the negative-side float dust clamped to
    # 0.0, below the clamp refused, non-finite refused)
    assert validate_p_n2(0.5) == 0.5
    assert validate_p_n2(0.5 + 1e-12) == 0.5 + 1e-12
    assert validate_p_n2(0.0) == 0.0
    assert validate_p_n2(-1e-13) == 0.0
    with pytest.raises(ValueError):
        validate_p_n2(0.5 + 2e-12)
    with pytest.raises(ValueError):
        validate_p_n2(1.0)
    with pytest.raises(ValueError):
        validate_p_n2(-1e-9)
    with pytest.raises(ValueError):
        validate_p_n2(float("nan"))


def test_nd_multi_limits_pins():
    """The exact residual limits of the intact configuration at a
    first-source mass of 1.0 GeV (the pinned grid configuration), the
    lower limit at ``m_pi0^2`` exactly and the upper limit at
    ``(sqrt_s - M_1 - M_2 - M_c1)^2`` within ``1e-12`` relative, and
    the checked refusals of an empty residual window, a non-positive
    mass, and a non-finite ``sqrt_s``."""

    m2_min, m2_max = nd_multi_limits(PROTON_MASS, PROTON_MASS,
                                     LIMITS_M_C1, SQRT_S)
    assert m2_min == RESIDUAL_M2_MIN
    assert m2_max == RESIDUAL_M2_MAX
    assert m2_min == PI0_MASS * PI0_MASS
    upper = (SQRT_S - PROTON_MASS - PROTON_MASS - LIMITS_M_C1) ** 2
    assert abs(m2_max - upper) / upper <= 1e-12
    # the checked refusals: an empty residual window (the message names
    # the window), a non-positive mass, and a non-finite sqrt_s
    with pytest.raises(ValueError) as excinfo:
        nd_multi_limits(PROTON_MASS, PROTON_MASS, 16.0, SQRT_S)
    assert ("nd_multi_limits: the residual window is empty"
            in str(excinfo.value))
    with pytest.raises(ValueError):
        nd_multi_limits(0.0, PROTON_MASS, GRID_M_C1, SQRT_S)
    with pytest.raises(ValueError):
        nd_multi_limits(PROTON_MASS, -PROTON_MASS, GRID_M_C1, SQRT_S)
    with pytest.raises(ValueError):
        nd_multi_limits(PROTON_MASS, PROTON_MASS, GRID_M_C1,
                        float("nan"))


def test_nd_multi_kinematics_pins():
    """The exact four-system solve on the pinned grid point: the four
    momenta at their committed lab values, the incoming-sum closure
    within ``1e-9`` relative, the mass shells, the central pair at
    rest in the overall CM, the pair invariant at
    ``sqrt_s - M_c1 - M_c2``, the back-to-back coupling in the CM, the
    realized projectile-side transfer at the drawn value within
    ``1e-9``, and the checked refusals of a positive drawn transfer,
    a non-positive mass, and a closed four-system configuration."""

    system1, system2, central1, central2 = nd_multi_kinematics(
        PROTON_MASS, GRID_T1, PROTON_MASS, GRID_T2, GRID_M_C1,
        GRID_M_C2, SQRT_S)
    for got, want in ((system1, GRID_SYSTEM1), (system2, GRID_SYSTEM2),
                      (central1, GRID_CENTRAL1),
                      (central2, GRID_CENTRAL2)):
        for i in range(4):
            assert abs(got[i] - want[i]) <= 1e-12, (i, got, want)

    # the incoming-sum closure within 1e-9 relative
    reference = _incoming_sum()
    total = [system1[i] + system2[i] + central1[i] + central2[i]
             for i in range(4)]
    e_total = reference[0]
    p_beam = reference[3]
    worst = max(abs(total[0] - reference[0]) / e_total,
                abs(total[1]) / p_beam,
                abs(total[2]) / p_beam,
                abs(total[3] - reference[3]) / p_beam)
    assert worst <= 1e-9

    # the mass shells (both baryonic systems at the proton mass, the
    # central sources at their drawn masses)
    assert abs(_invariant_mass(system1) - PROTON_MASS) <= 1e-9
    assert abs(_invariant_mass(system2) - PROTON_MASS) <= 1e-9
    assert abs(_invariant_mass(central1) - GRID_M_C1) <= 1e-9
    assert abs(_invariant_mass(central2) - GRID_M_C2) <= 1e-9

    # the central pair at rest in the overall CM (each source carries
    # zero momentum there; the lab relation is the exact forward boost
    # of the at-rest four-momentum (M, 0, 0, 0))
    _e_beam, _e_total, _sq, _p_star, beta, _gamma = _beam_configuration()
    gamma = 1.0 / math.sqrt(1.0 - beta * beta)
    for momentum, mass in ((central1, GRID_M_C1), (central2, GRID_M_C2)):
        assert momentum[1] == 0.0
        assert momentum[2] == 0.0
        assert abs(momentum[0] - gamma * mass) <= 1e-9
        assert abs(momentum[3] - gamma * beta * mass) <= 1e-9

    # the pair invariant at the exact CM energy of the pinned beam
    # configuration minus M_c1 + M_c2 (the module's operative
    # convention: the exact four-system solve runs at the exact
    # fixed-target CM energy sq_exact, the SQRT_S constant being its
    # rounded quote) and the back-to-back coupling in the CM (the exact
    # two-body kinematics of the (M_1, M_2) pair)
    _e_beam, _e_total, sq_exact, _p_star, _beta, _gamma = (
        _beam_configuration())
    pair = [system1[i] + system2[i] for i in range(4)]
    assert abs(_invariant_mass(pair)
               - (sq_exact - GRID_M_C1 - GRID_M_C2)) <= 1e-9
    system1_cm = _to_cm(system1)
    system2_cm = _to_cm(system2)
    for i in (1, 2, 3):
        assert abs(system1_cm[i] + system2_cm[i]) <= 1e-9

    # the realized projectile-side transfer at the drawn value
    # (the documented invariant relation on the first baryonic line)
    e_beam, _e_total, _sq, _p_star, _beta, _gamma = _beam_configuration()
    incoming = (e_beam, 0.0, 0.0, BEAM_MOMENTUM)
    dot = (incoming[0] * system1[0] - incoming[1] * system1[1]
           - incoming[2] * system1[2] - incoming[3] * system1[3])
    m2_1 = system1[0] ** 2 - system1[1] ** 2 - system1[2] ** 2 \
        - system1[3] ** 2
    t1_realized = (PROTON_MASS * PROTON_MASS + m2_1 - 2.0 * dot)
    assert abs(t1_realized - GRID_T1) <= 1e-9

    # the checked refusals: a positive drawn transfer, a non-positive
    # mass, and a four-system configuration closed beyond the
    # joint-closure tolerance
    with pytest.raises(ValueError):
        nd_multi_kinematics(PROTON_MASS, GRID_T1, PROTON_MASS, 0.42,
                            GRID_M_C1, GRID_M_C2, SQRT_S)
    with pytest.raises(ValueError):
        nd_multi_kinematics(-PROTON_MASS, GRID_T1, PROTON_MASS, GRID_T2,
                            GRID_M_C1, GRID_M_C2, SQRT_S)
    with pytest.raises(ValueError):
        nd_multi_kinematics(PROTON_MASS, GRID_T1, PROTON_MASS, GRID_T2,
                            8.0, 8.0, SQRT_S)


def test_event_record_pins(catalog, fams):
    """The pinned event-0 record at its committed values (the count
    ``n = 1``, the remnant kinds and masses, the single central mass,
    the two drawn transfers, and the exact ``t_c = t_1 + t_2``
    identity with ``t_2c = 0.0``), the ledger lines at ``(1, 1, 0,
    1)``, the central source at ``(0, 0, 0, 0)``, the two vacuum
    exchange records with the zero transfer, and the bit identity of
    the shared fields against card 7's ``generate_nd`` on the same
    trace."""

    event = generate_nd_multi(NOMINAL_PARAMS, NOMINAL_ND_REMNANT_WEIGHTS,
                              P_N2_PINNED, catalog, fams,
                              *_event_rngs(0))
    assert event.class_label == ND_MULTI_CLASS_LABEL == "nd_multi"
    assert event.count == EVENT0_COUNT == 1
    assert event.p_n2 == P_N2_PINNED
    assert event.remnants[0].kind == EVENT0_REMNANT_0_KIND
    assert event.remnants[0].mass == EVENT0_REMNANT_0_MASS
    assert event.remnants[1].kind == EVENT0_REMNANT_1_KIND
    assert event.remnants[1].mass == EVENT0_REMNANT_1_MASS
    assert event.central_masses == (EVENT0_CENTRAL_MASS,)
    assert event.t1 == EVENT0_T1
    assert event.t2 == EVENT0_T2
    assert event.t_c == EVENT0_T_C
    assert event.t_c == event.t1 + event.t2
    assert event.t_2c == 0.0
    # the ledger lines: both baryonic lines at (1, 1, 0, 1) after the
    # vacuum exchanges' zero transfers; the central source at (0, 0, 0, 0)
    assert event.lines[0].qn == (1, 1, 0, 1)
    assert event.lines[1].qn == (1, 1, 0, 1)
    assert event.central_sources[0].qn == CENTRAL_QN == (0, 0, 0, 0)
    # the two vacuum exchange records in card 1's form
    for record in event.exchanges:
        assert isinstance(record, ExchangeRecord)
        assert record.name == "vacuum"
        assert tuple(record.transfer) == (0, 0, 0, 0)
    # the final-state line labels on the closed set
    assert len(event.final_state) == EVENT0_N_FINAL
    assert all(h.line in ND_LINE_LABELS for h in event.final_state)
    assert any(h.line == PROJECTILE_LINE_LABEL for h in event.final_state)
    assert any(h.line == TARGET_LINE_LABEL for h in event.final_state)
    assert any(h.line == CENTRAL_LINE_LABEL for h in event.final_state)
    # the n = 1 bit identity: the shared fields of the record are
    # field-for-field card 7's ND record on the same trace
    nd = generate_nd(NOMINAL_PARAMS, NOMINAL_ND_REMNANT_WEIGHTS,
                     catalog, fams, *_nd_event_rngs(0))
    assert event.remnants == nd.remnants
    assert event.central_masses == (nd.central_mass,)
    assert event.t_c == nd.t_c
    assert event.t1 == nd.t1
    assert event.t2 == nd.t2
    assert event.lines == nd.lines
    assert event.central_sources == (nd.central_source,)
    assert event.exchanges == nd.exchanges
    assert event.final_state == nd.final_state


def test_draw_budget_pins(catalog, fams, monkeypatch):
    """A pinned 200-event trace on one substream quintuple per event
    index (450,000 + i, P(n = 2) = 0.0) consumes exactly the
    pre-registered counts on the closed labels: 1 ``topology_selection``
    uniform (the count draw, drawn before card 7's draw order), 2
    ``spectral_state`` uniforms per event (one remnant selection per
    baryonic line), and 4 ``exchange_kinematics`` uniforms per event
    (the ``t_1`` draw, the ``t_2`` draw, and the two-uniform ``M_c1^2``
    central draw), plus the frozen Round 4 per-vertex budget on
    ``decay_channel`` / ``decay_kinematics`` (the pinned per-event
    totals and min/max at this landing); a pinned 200-event trace at
    P(n = 2) = 0.5 (460,000 + i) consumes exactly ``4 + 1 + 2 * 1``
    ``exchange_kinematics`` uniforms per n = 2 event (the ``t_2c`` draw
    plus one accepted two-uniform ``M_2c^2`` attempt) with pinned
    RNG-state identity across two runs, and the pinned rejection cap
    fires at exactly ``REJECTION_CAP = 1000`` consecutive rejected
    attempts with the pinned exchange count (``4 + 1 + 2 * 1000``)."""

    # the n = 1 budget trace: the closed-label counts per event and the
    # frozen Round 4 per-vertex decay budget
    per_event_channel: list[int] = []
    per_event_kinematics: list[int] = []
    for event_index in range(N_N1_BUDGET):
        subs = derive_substreams(
            MASTER_SEED, 0, 450_000 + event_index)
        counter_topology = _CountingRng(np.random.default_rng(
            subs.stream_for("topology_selection")))
        counter_spectral = _CountingRng(np.random.default_rng(
            subs.stream_for("spectral_state")))
        counter_exchange = _CountingRng(np.random.default_rng(
            subs.stream_for("exchange_kinematics")))
        counter_channel = _CountingRng(np.random.default_rng(
            subs.stream_for("decay_channel")))
        counter_kinematics = _CountingRng(np.random.default_rng(
            subs.stream_for("decay_kinematics")))
        generate_nd_multi(NOMINAL_PARAMS, NOMINAL_ND_REMNANT_WEIGHTS,
                          0.0, catalog, fams, counter_topology,
                          counter_spectral, counter_exchange,
                          counter_channel, counter_kinematics)
        assert counter_topology.calls == 1
        assert counter_spectral.calls == 2
        assert counter_exchange.calls == 4
        per_event_channel.append(counter_channel.calls)
        per_event_kinematics.append(counter_kinematics.calls)
    # the pinned per-event counts of the first event of the trace
    subs = derive_substreams(MASTER_SEED, 0, 450_000)
    first = [_CountingRng(np.random.default_rng(
        subs.stream_for(label)))
             for label in ("topology_selection", "spectral_state",
                           "exchange_kinematics", "decay_channel",
                           "decay_kinematics")]
    generate_nd_multi(NOMINAL_PARAMS, NOMINAL_ND_REMNANT_WEIGHTS, 0.0,
                      catalog, fams, *first)
    assert [counter.calls for counter in first] == [1, 2, 4, 2, 4]

    # the frozen Round 4 per-vertex budget on the decay labels (the
    # pinned per-event totals and min/max at this landing; a catalog
    # handoff and the unstable-sibling kernels add the Round 2 kernel's
    # own documented draws on the same streams; a stable remnant record
    # and a below-threshold central source consume zero)
    assert sum(per_event_channel) == N1_BUDGET_CHANNEL_TOTAL
    assert sum(per_event_kinematics) == N1_BUDGET_KINEMATICS_TOTAL
    assert min(per_event_channel) == N1_BUDGET_MIN_CHANNEL
    assert max(per_event_channel) == N1_BUDGET_MAX_CHANNEL
    assert min(per_event_kinematics) == N1_BUDGET_MIN_KINEMATICS
    assert max(per_event_kinematics) == N1_BUDGET_MAX_KINEMATICS
    assert all(k >= 2 * c
               for c, k in zip(per_event_channel, per_event_kinematics))
    assert sum(1 for c in per_event_channel if c == 0) \
        == N1_BUDGET_ZERO_DRAW_EVENTS

    # the n = 2 budget trace: the exchange budget per event (4 for the
    # n = 1 events, 4 + 1 + 2 * 1 = 7 for the n = 2 events, every n = 2
    # event with exactly one accepted M_2c^2 attempt) and the pinned
    # RNG-state identity across two runs on all five streams
    n2_events = 0
    exchange_total = 0
    run_a = []
    for event_index in range(N_N2_BUDGET):
        subs = derive_substreams(
            MASTER_SEED, 0, 460_000 + event_index)
        counters = tuple(
            _CountingRng(np.random.default_rng(subs.stream_for(label)))
            for label in ("topology_selection", "spectral_state",
                          "exchange_kinematics", "decay_channel",
                          "decay_kinematics"))
        event = generate_nd_multi(
            NOMINAL_PARAMS, NOMINAL_ND_REMNANT_WEIGHTS, 0.5, catalog,
            fams, *counters)
        assert counters[0].calls == 1
        assert counters[1].calls == 2
        if event.count == 1:
            assert counters[2].calls == 4
        else:
            n2_events += 1
            assert counters[2].calls == 7
        exchange_total += counters[2].calls
        run_a.append(tuple(c._source.bit_generator.state for c in counters))
    assert n2_events == N2_BUDGET_N2_EVENTS
    assert exchange_total == N2_BUDGET_EXCHANGE_TOTAL
    run_b = []
    for event_index in range(N_N2_BUDGET):
        subs = derive_substreams(
            MASTER_SEED, 0, 460_000 + event_index)
        rngs = [np.random.default_rng(subs.stream_for(label))
                for label in ("topology_selection", "spectral_state",
                              "exchange_kinematics", "decay_channel",
                              "decay_kinematics")]
        generate_nd_multi(NOMINAL_PARAMS, NOMINAL_ND_REMNANT_WEIGHTS,
                          0.5, catalog, fams, *rngs)
        run_b.append(tuple(rng.bit_generator.state for rng in rngs))
    for state_a, state_b in zip(run_a, run_b):
        assert state_a == state_b

    # the pinned rejection cap: with the joint closure refused on
    # every attempt the generator raises
    # PhaseSpaceRejectionExceeded after exactly REJECTION_CAP attempts
    # (the exchange count is 4 + 1 + 2 * 1000 = 2005)
    import na49_mixing.topology_multi_neutral as _tmn
    monkeypatch.setattr(_tmn, "joint_closure",
                        lambda *args, **kwargs: False)
    subs = derive_substreams(MASTER_SEED, 0, CAP_EVENT_INDEX)
    counters = tuple(
        _CountingRng(np.random.default_rng(subs.stream_for(label)))
        for label in ("topology_selection", "spectral_state",
                      "exchange_kinematics", "decay_channel",
                      "decay_kinematics"))
    with pytest.raises(PhaseSpaceRejectionExceeded) as excinfo:
        generate_nd_multi(NOMINAL_PARAMS, NOMINAL_ND_REMNANT_WEIGHTS,
                          0.5, catalog, fams, *counters)
    assert [counter.calls for counter in counters] == [1, 2, 2005, 0, 0]
    assert "exceeded the cap of 1000" in str(excinfo.value)


def test_central_neutrality_pins(pinned_trace):
    """On the pinned 10,000-event trace every central source of every
    event carries ``(B, Q, S, 2I3) = (0, 0, 0, 0)`` exactly and the
    central line's joint closure holds on every event;
    ``check_multi_central_neutrality`` raises on each planted
    violation (central source ``B = 1``, one open central joint
    vertex), naming the offending source or the open vertex, and
    refuses a foreign record type."""

    assert len(pinned_trace) == N_PINNED
    for event in pinned_trace:
        assert len(event.central_sources) == event.count
        for source in event.central_sources:
            assert source.qn == CENTRAL_QN == (0, 0, 0, 0)
        check_multi_central_neutrality(event)  # raises on any violation

    # the planted violations: each is detected, naming the offending
    # source component or the open joint vertex (the first n = 2 event
    # of the trace, whose central line carries two hadrons)
    event = pinned_trace[FIRST_N2_INDEX]
    assert event.count == 2
    source = event.central_sources[0]
    bad_source = LineState(B=1, Q=0, S=0, I3x2=0,
                           four_momentum=source.four_momentum)
    bad_sources = list(event.central_sources)
    bad_sources[0] = bad_source
    bad_event = dataclasses.replace(event, central_sources=tuple(bad_sources))
    with pytest.raises(CentralNeutralityViolation) as excinfo:
        check_multi_central_neutrality(bad_event)
    assert ("central source 0 carries B = 1 against the central "
            "quantum numbers (0, 0, 0, 0)" in str(excinfo.value))

    # one open central joint vertex: a central-line daughter doctored
    # to carry baryon number breaks the joint local closure
    doctored = list(event.final_state)
    for index, hadron in enumerate(doctored):
        if hadron.line == CENTRAL_LINE_LABEL:
            doctored[index] = dataclasses.replace(hadron, qn=(1, 0, 0, 0))
            break
    open_event = dataclasses.replace(event, final_state=tuple(doctored))
    with pytest.raises(CentralNeutralityViolation) as excinfo:
        check_multi_central_neutrality(open_event)
    assert ("the central decay vertices are open: the central line's "
            "2 final-state hadrons sum to (B, Q, S, 2I3) (1, 0, 0, 0) "
            "against" in str(excinfo.value))

    # the check refuses a foreign record type
    with pytest.raises(ValueError):
        check_multi_central_neutrality("not an event")


def test_conservation_pins(pinned_trace):
    """Over the pinned 10,000-event trace at P(n = 2) = 0.25 the final
    state carries exactly ``(B, Q, S) = (2, 2, 0)``, the four-momentum
    closure is within ``1e-9`` relative at the pinned worst value
    (1.33e-14), the mass shells are within ``1e-9`` relative,
    ``t_c = t_1 + t_2`` exactly on every event, the ``n = 2`` census is
    pinned, and every event passes card 2's ``audit_event`` with zero
    violations (two vacuum transfers, both baryonic lines ``(1, 1, 0,
    1)``, the central sources ``(0, 0, 0, 0)``, the local decay
    closures)."""

    assert len(pinned_trace) == N_PINNED
    worst_closure = 0.0
    worst_shell = 0.0
    n2 = 0
    for event in pinned_trace:
        assert isinstance(event, NDMultiEvent)
        assert event.class_label == ND_MULTI_CLASS_LABEL == "nd_multi"
        assert event.count in (1, 2)
        assert event.p_n2 == P_N2_PINNED
        assert (sum(h.qn[0] for h in event.final_state),
                sum(h.qn[1] for h in event.final_state),
                sum(h.qn[2] for h in event.final_state)) == (2, 2, 0)
        assert event.t_c == event.t1 + event.t2
        worst_closure = max(worst_closure, _relative_closure(event))
        for hadron in event.final_state:
            e, px, py, pz = hadron.four_momentum
            m2 = e * e - px * px - py * py - pz * pz
            # a massless final state (the photon of a radiative channel,
            # 2026-10-02) is on shell relative to its energy scale
            scale = max(hadron.mass * hadron.mass, 1e-9) if hadron.mass > 0.0 else max(float(hadron.four_momentum[0]) ** 2, 1e-9)
            worst_shell = max(worst_shell,
                              abs(m2 - hadron.mass * hadron.mass) / scale)
        if event.count == 2:
            n2 += 1
        assert not _audit(event).violations
        check_multi_central_neutrality(event)
    assert worst_closure <= 1e-9
    assert worst_shell <= 1e-9
    assert n2 == N_N2_EVENTS
    assert pinned_trace[FIRST_N2_INDEX].count == 2
    assert all(event.count == 1
               for event in pinned_trace[:FIRST_N2_INDEX])


def test_central_channel_pins(pinned_trace):
    """On the pinned trace the below-threshold quantum-number-free
    central states are pinned (the n = 1 central-decay count and the
    n = 2 central-line event and state counts), the hyperons appear
    only on the baryonic lines (the pinned census), the central kaon
    events are net neutral in charge and strangeness (the pinned
    count), and the K K-bar direct channel events are pinned."""

    assert len(pinned_trace) == N_PINNED
    n1_below = 0
    n2_below_events = 0
    n2_below_states = 0
    n_hyperon = 0
    n_central_hyperon = 0
    n_central_kaon_events = 0
    n_kkbar_events = 0
    hyperon_pdgs = (3122, 3124, 3222, 3224, 3312, 3322, 3334)
    kaon_pdgs = (321, -321, 310, 130)
    for event in pinned_trace:
        central = [h for h in event.final_state
                   if h.line == CENTRAL_LINE_LABEL]
        central_pdgs = [h.pdg_id for h in central]
        if (321 in central_pdgs and -321 in central_pdgs) \
                or (310 in central_pdgs and 130 in central_pdgs):
            n_kkbar_events += 1
        if any(p in kaon_pdgs for p in central_pdgs):
            n_central_kaon_events += 1
            # the central kaons are net neutral in charge and
            # strangeness on every central line
            assert sum(h.qn[1] for h in central) == 0
            assert sum(h.qn[2] for h in central) == 0
        singles = [h for h in central if h.pdg_id is None]
        if event.count == 1:
            if len(central) == 1 and central[0].pdg_id is None:
                # the below-threshold branch: a single
                # quantum-number-free final state at the drawn mass
                # with no vertex
                n1_below += 1
                state = central[0]
                assert state.qn == (0, 0, 0, 0)
        else:
            if singles:
                n2_below_events += 1
            for state in singles:
                assert state.qn == (0, 0, 0, 0)
            n2_below_states += len(singles)
        for hadron in event.final_state:
            if hadron.pdg_id in hyperon_pdgs:
                n_hyperon += 1
                assert hadron.line in (PROJECTILE_LINE_LABEL,
                                       TARGET_LINE_LABEL)
                if hadron.line == CENTRAL_LINE_LABEL:
                    n_central_hyperon += 1
    assert n1_below == N1_BELOW_THRESHOLD
    assert n2_below_events == N2_BELOW_EVENTS
    assert n2_below_states == N2_BELOW_STATES
    assert n_hyperon == HYPERON_TOTAL
    assert n_central_hyperon == CENTRAL_HYPERONS
    assert n_central_kaon_events == EVENTS_CENTRAL_KAONS
    assert n_kkbar_events == EVENTS_KKBAR


def test_frozen_cascade_pins(catalog, fams, pinned_trace):
    """Every central source decays through the frozen simplified
    cascade (kappa_B = 0.0, kappa_M = 4.0, b_dec = 2.0 GeV^-2, baryonic
    nominal width 0.3 GeV, pinned through the R45 driver digests in
    both helper trees); the ``W_flavor = 1.0`` hook identity on the
    channel weight, and the R46 ``no_refit`` white-box check (no
    module in this card references a re-fit of the alignment
    parameters or claims the separation of exchange recoil from decay
    transverse dynamics)."""

    # the frozen simplified cascade is consumed read-only: the module
    # routes both central sources through card 7's sealed decay
    # machinery (the cascade routing of the R45 point) and introduces
    # no alignment re-fit and no separation claim (the R46 clauses;
    # the white-box token scan)
    import na49_mixing.cascade_alignment  # noqa: F401
    import na49_mixing.cascade_form_factor  # noqa: F401
    import na49_mixing.cascade_termination  # noqa: F401
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "_decay_central" in source and "_decay_remnant" in source
    for token in NO_REFIT_TOKENS:
        assert token not in source, token

    # the sealed R45 point is pinned through the R45 driver digests in
    # both helper trees (the frozen simplified configuration: the
    # constant-kappa alignment kernels, the frozen b_dec, the baryonic
    # nominal width)
    for rel in ("dev/mixing2006/helpers/benchmarking/"
                "staged_r45_simplified_alignment_freeze.py",
                "helpers/benchmarking/"
                "staged_r45_simplified_alignment_freeze.py"):
        path = MONO_ROOT / rel
        assert path.is_file(), f"missing frozen surface: {rel}"
        assert _sha256_file(path) == FROZEN_SURFACES[rel], (
            f"drifted surface: {rel}")

    # the W_flavor hook default is exactly 1.0 (Round 6 owns the flavor
    # weights); the hook is multiplicative and consumes zero draws
    probe_channel = CascadeChannel(
        parent_id="baryonic_np", daughters=("catalog:2212", "catalog:111"),
        kind="established_hadron", weight=0.5)
    assert channel_weight(probe_channel) == 0.5
    assert channel_weight(probe_channel, w_flavor=1.0) == 0.5

    # the first n = 2 event of the trace: all four systems close
    # locally through the ledger's local decay closure and both central
    # sources stay neutral
    event = pinned_trace[FIRST_N2_INDEX]
    assert event.lines[0].qn == (1, 1, 0, 1)
    assert event.lines[1].qn == (1, 1, 0, 1)
    assert all(s.qn == CENTRAL_QN for s in event.central_sources)
    report = _audit(event)
    assert not report.violations
    check_multi_central_neutrality(event)


# ----------------------------------------------------------------------
# Part 2: the substrate and freeze pins
# ----------------------------------------------------------------------
def test_module_byte_and_export_pins():
    """The module at its committed sha256 (nested tree HEAD bb43481)
    with its exact ``__all__`` export set (eleven names), the solver
    test file at its committed sha256 (the post-re-pin value, the
    525 -> 526 count re-pin of this register's landing), and the
    package ``__init__`` at its committed sha256 carrying the
    multiple-neutral-excitation export block (the eleven re-exported
    names; the elastic and mass-law constants stay card 1/card 4's
    package-level exports, and the final-state hadron record stays
    card 5's package-level ``FinalHadron`` export)."""

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    import na49_mixing.topology_multi_neutral as _mod
    assert _mod.__all__ == MODULE_ALL

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)
    init_text = INIT_PATH.read_text(encoding="utf-8")
    block = re.search(
        r"from \.topology_multi_neutral import \((.*?)\)",
        init_text, re.S)
    assert block is not None, (
        "the __init__ multiple-neutral-excitation import block is "
        "missing")
    for name in INIT_MULTI_NAMES:
        assert re.search(rf"\b{name}\b", block.group(1)), (
            f"{name} is not in the __init__ multi-neutral import "
            "block")
    # the elastic and mass-law constants and the card-5 record stay
    # module-level only (not re-exported through the multi-neutral
    # block)
    for name in ("BEAM_MOMENTUM", "PROTON_MASS", "SQRT_S", "FinalHadron",
                 "NOMINAL_PARAMS", "MassLawParams"):
        assert not re.search(rf"\b{name}\b", block.group(1)), (
            f"{name} must not enter the __init__ multi-neutral import "
            "block")
    all_block = re.search(r"__all__\s*=\s*\[(.*?)\n\]", init_text, re.S)
    assert all_block is not None
    all_names = re.findall(r'"([^"]+)"', all_block.group(1))
    assert len(all_names) == INIT_ALL_COUNT
    for name in INIT_MULTI_NAMES:
        assert f'"{name}"' in all_block.group(1), (
            f"{name} is not in the __init__ __all__")


def test_frozen_substrate_digests():
    """The card's frozen-substrate provenance surfaces (verified on
    disk) and the frozen M8 parameter map through the manifest (the
    ten-parameter set hash, the kernel boundary, and the
    ``regge.effective_transfer_slope``); the activated neutral-mesonic
    ``(0, 0, 0)`` continuum family of the committed configuration
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

    # the activated neutral-mesonic (0, 0, 0) continuum family of the
    # committed configuration: exactly one, with its support ends at
    # the frozen boundary and the card-6 surface's upper end
    continuum = json.loads(CONTINUUM_PATH.read_text(encoding="utf-8"))
    neutral = [family for family in continuum["families"]
               if family["qn"] == [0, 0, 0] and family["active"] is True]
    assert len(neutral) == 1
    assert neutral[0]["family_id"] == "mesonic_neutral"
    assert neutral[0]["m_min"] == KERNEL_BOUNDARY
    assert neutral[0]["m_max"] == 5.0




def test_layout_sync():
    """The layout sync (the subprocess check from the monorepo root;
    ``logic_drift = 0``)."""

    probe = subprocess.run(
        [sys.executable, str(SYNC_CHECKER)],
        capture_output=True, text=True, cwd=str(MONO_ROOT))
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert "logic_drift=0" in probe.stdout
