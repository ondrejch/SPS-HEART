"""Round 5, card 7 test-author register: the non-diffractive class.

Independent contract pins for the one-neutral-excitation
non-diffractive class (``na49_mixing/topology_non_diffractive.py``),
Round 5 card 7 (``TASK-20260925-r5-non-diffrative-central-excitation``;
plan section 22 task 31, line 1756; plan section 11.1 item 5, line
828, with plan section 11.3, lines 844-870), at nested tree SHA
a4f89ed.  This register is the test-author half of the card: it pins
the module contract on its own pinned fixtures (the frozen ``NDEvent``
record surface with the class label ``"nd"``, the exact central limits
of the central-production law, the exact three-system kinematics solve
with the central system at rest in the overall CM and the two baryonic
systems back-to-back against it at the pair invariant ``sqrt_s - M_c``
with the production angle from the drawn projectile-side transfer, the
pinned event-0 record, the draw-budget pins on the closed substream
labels, the central-neutrality pins with the planted-violation
detection, the conservation pins with the pinned remnant census at the
nominal R50 split and zero audit violations, the central-channel pins
of plan section 11.3 (the below-threshold quantum-number-free states,
the K K-bar direct channels, the hyperons only on the baryonic lines,
the net-neutral central kaons), the frozen-cascade pins at the sealed
R45 point with the R46 ``no_refit`` white-box check, the byte-stable
module / solver test / package ``__init__`` digests and export
surface, and the frozen-substrate provenance) and re-pins the freeze
manifest for the card (``INVENTORY_COUNT`` 522 -> 523).

Part 1: the module contract (pure, hermetic)
--------------------------------------------

- ``test_named_constants_and_record_surface``: the closed
  non-diffractive line-label set ``{"projectile", "target",
  "central"}``, the class label ``"nd"``, the frozen record surfaces
  (the ten-field ``NDEvent`` and the five-field ``FinalHadron``), the
  nominal parameters (``epsilon = 0``, ``B_SD`` the M8-frozen slope,
  the nominal R50 remnant split per line), the central quantum numbers
  ``CENTRAL_QN = (0, 0, 0, 0)``, the pion mass at card 4's committed
  catalog value, and the closed-substream ``generate_nd`` signature
  (four exchange/decay streams; no ``topology_selection`` stream).
- ``test_nd_limits_pins``: the exact central limits
  ``(0.01821868254756, 236.95846853605096)`` GeV^2 at the pinned
  intact-pair configuration, the lower limit at ``m_pi0^2`` exactly
  and the upper limit at ``(sqrt_s - M_1 - M_2)^2`` within ``1e-12``
  relative, and the checked refusal of a non-positive remnant mass.
- ``test_nd_kinematics_pins``: the exact three-system solve on a
  pinned grid of ``(M_1, t_1, M_2, t_2, M_c)`` points: the four
  momenta at their committed values, the incoming-sum closure within
  ``1e-9`` relative, the mass shells, the central system at rest in
  the overall CM, the pair invariant at ``sqrt_s - M_c``, the
  back-to-back coupling in the CM, the realized projectile-side
  transfer at the drawn value within ``1e-9``, and the checked
  refusal of a positive drawn ``t_2``.
- ``test_event_record_pins``: the pinned event-0 record at its
  committed values (the remnant kinds and masses, the central mass,
  the two drawn transfers, and the exact ``t_c = t_1 + t_2``
  identity), the ledger lines at ``(1, 1, 0, 1)``, the two vacuum
  exchange records with the zero transfer, and the final-state line
  labels on the closed set.
- ``test_draw_budget_pins``: a pinned 200-event trace on one
  substream quadruple per event index consumes exactly two
  ``spectral_state`` uniforms per event (one remnant selection per
  baryonic line) and the ``exchange_kinematics`` budget of the card-4
  decomposition (the ``t_1`` draw, the ``t_2`` draw, and the
  two-uniform ``M_c^2`` central draw: 4 uniforms per event; the
  card's "5" literal is the recorded operative resolution), plus the
  frozen Round 4 per-vertex budget on ``decay_channel`` /
  ``decay_kinematics`` (the pinned per-event totals and min/max at
  this landing, with pinned RNG-state identity; the
  ``topology_selection`` draw belongs to the runner, card 9).
- ``test_central_neutrality_pins``: on the pinned 10,000-event trace
  the central source carries ``(B, Q, S, 2I3) = (0, 0, 0, 0)``
  exactly and every central decay vertex closes locally to
  ``(0, 0, 0)``; ``check_central_neutrality`` raises on each planted
  violation (central ``B = 1``, central ``Q = 1``, central ``S = 1``,
  one open central vertex), naming the offending component or the
  open vertex, and refuses a foreign record type.
- ``test_conservation_pins``: over the pinned trace the final state
  carries exactly ``(B, Q, S) = (2, 2, 0)``, the four-momentum
  closure is within ``1e-9`` relative at the pinned worst value, the
  remnant census is pinned at the nominal R50 split (intact/intact
  2558, mixed 4951, low/low 2491), ``t_c = t_1 + t_2`` exactly on
  every event, and every event passes card 2's ``audit_event`` with
  zero violations (two vacuum transfers, both baryonic lines
  ``(1, 1, 0, 1)``, the central source ``(0, 0, 0, 0)``, the local
  decay closures).
- ``test_central_channel_pins``: on the pinned central-decay trace
  the channel daughters sum to ``(0, 0, 0)`` exactly, the
  below-threshold central sources stay single quantum-number-free
  final states at their drawn mass with zero decay draws (the pinned
  count), and the ``K K-bar`` direct channels occur; on the pinned
  event trace the hyperons appear only on the baryonic lines (the
  pinned census), the central kaon events are net neutral in charge
  and strangeness (the pinned count), and the ``K K-bar`` direct
  channel events are pinned; the committed ``(0, 0, 0)`` records
  f2(1270), rho3(1690)0, f4(2050), and phi(1020) carry the recorded
  K K-bar pair classes, and the phi record carries both the ``K+ K-``
  and the ``K0S K0L`` channels.
- ``test_frozen_cascade_pins``: all three systems decay through the
  frozen simplified cascade at the sealed R45 point (kappa_B = 0.0,
  kappa_M = 4.0, b_dec = 2.0 GeV^-2, baryonic nominal width 0.3 GeV,
  pinned through the R45 driver digests in both helper trees), the
  ``W_flavor = 1.0`` hook identity on the channel weight, the
  neutral-mesonic ``mesonic_neutral`` continuum family of the central
  source, and the R46 ``no_refit`` white-box check (no module in this
  card references a re-fit of the alignment parameters or claims the
  separation of exchange recoil from decay transverse dynamics).

Part 2: the substrate and freeze pins
--------------------------------------

- ``test_module_byte_and_export_pins``: the module at its committed
  sha256, its ``__all__`` export set (nineteen names), the solver
  test file at its committed sha256 after the 522 -> 523 count
  re-pin, and the package ``__init__`` at its committed sha256
  carrying the non-diffractive export block (the eleven re-exported
  names; the package ``__all__`` total at 231 entries).
- ``test_frozen_substrate_digests``: the card's frozen-substrate
  provenance surfaces (verified on disk) and the frozen M8 parameter
  map through the manifest, with the activated neutral-mesonic
  ``(0, 0, 0)`` continuum family's recorded support ends.
- ``test_freeze_repin_register``: the freeze-manifest re-pin for
  this card (the strict path-sorted inventory at 523 entries; 522 at
  the card 7 solver landing; 520 after the Round 5 card 6 landing;
  519 after the Round 5 card 6 test-author re-pin; 517 after the
  Round 5 card 5 test-author re-pin; 516 after the Round 5 card 5
  re-pin; 514 after the Round 5 card 4 test-author re-pin; 513 after
  the Round 5 card 4 re-pin; 511 after the Round 5 card 3
  test-author re-pin; 510 after the Round 5 card 3 solver landing;
  508 after the test-protocol tiering change; 505 after the Round 5
  card 2 test-author re-pin; 504 after the Round 5 card 2 solver
  landing; 502 after the Round 5 card 1 test-author re-pin; 501 at
  the Round 5 card 1 landing; +1 entry: this register) carries this
  register and the thirty-two re-pinned chain count-pin files at
  their on-disk digests, the card 7 solver's freeze exception stays
  exactly once at its 35 path-sorted files with its reason naming the
  520 -> 522 re-pin, this card's test-author freeze exception names
  the card exactly once with the path-sorted file list and a reason
  naming the 522 -> 523 re-pin, the chain count pins are re-pinned
  522 -> 523 with the 522 literal retained in the naming comments,
  and the frozen M8 parameter map stays intact.
- ``test_layout_sync``: the layout sync (the subprocess check from
  the monorepo root; ``logic_drift = 0``).

Hermetic: the module, the committed catalog, the continuum families
configuration, the Round 4 cascade modules (consumed read-only), the
committed cards 1-6 modules (consumed read-only), numpy, and the
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

from na49_mixing.cascade_decay import (
    ISOTROPIC_ALIGNMENT_LABEL,
    CascadeChannel,
    CascadeState,
    channel_weight,
    enumerate_channels,
)
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
    draw_central,
)
from na49_mixing.topology_remnants import (
    KIND_INTACT_PROTON,
    KIND_LOW_MASS_BARYONIC_RESONANCE,
    KERNEL_BOUNDARY,
    RemnantSelection,
)
from na49_mixing.topology_non_diffractive import (
    CENTRAL_LINE_LABEL,
    CENTRAL_QN,
    ND_CLASS_LABEL,
    ND_LINE_LABELS,
    NOMINAL_ND_REMNANT_WEIGHTS,
    NOMINAL_PARAMS,
    PROJECTILE_LINE_LABEL,
    TARGET_LINE_LABEL,
    CentralNeutralityViolation,
    NDEvent,
    _beam_configuration,
    _incoming_lines,
    _decay_central,
    check_central_neutrality,
    generate_nd,
    nd_kinematics,
    nd_limits,
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
MODULE_PATH = NESTED_ROOT / "na49_mixing" / "topology_non_diffractive.py"
SOLVER_TEST_PATH = NESTED_ROOT / "tests" / "test_topology_non_diffractive.py"
INIT_PATH = NESTED_ROOT / "na49_mixing" / "__init__.py"
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
CONTINUUM_PATH = (
    NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json")

#: The Round 5 master seed (round 5 index item 9), fixed for the pinned
#: traces; replica 0, one substream quadruple per event index.
MASTER_SEED = 20260925

#: The pinned trace sizes (the 10,000-event conservation/census/channel
#: trace, the 200-event draw-budget trace on the dedicated event
#: indices 50,000 + i, and the 1,000-decay central-decay trace on the
#: dedicated streams: the mass stream at event index 70,000 and the
#: decay substreams at the event indices 60,000 + i, all disjoint from
#: the solver test's class traces).
N_PINNED = 10_000
N_BUDGET = 200
N_CENTRAL = 1_000

#: The exact central limits of the pinned intact-pair configuration
#: (card 4's recorded values at ``SQRT_S = 17.270``).
M2_MIN_PINNED = 0.01821868254756
M2_MAX_PINNED = 236.95846853605096

#: The pinned kinematics grid point ``(M_1, t_1, M_2, t_2, M_c)`` of
#: the three-system solve (both lines at the intact proton mass, the
#: drawn transfers inside their exact two-body ranges, the central
#: mass at 0.9 GeV) and its solved lab-frame four-momenta
#: (projectile system, target system, central system).
GRID_T1 = -0.35
GRID_T2 = -0.42
GRID_M_C = 0.9
GRID_SYSTEM1 = (149.53339780476412, 0.5733302398126688, 0.0,
                149.52935496599403)
GRID_SYSTEM2 = (1.1247853277400153, -0.5733302398126688, 0.0,
                0.2368111946315701)
GRID_CENTRAL = (8.28287508625037, 0.0, 0.0, 8.233833839374405)

#: The pinned event-0 record (master seed 20260925, replica 0, event
#: index 0; the nominal R50 split selection and the central draw at
#: that landing).
EVENT0_REMNANT_0_KIND = "intact_proton"
EVENT0_REMNANT_0_MASS = 0.93827231
EVENT0_REMNANT_1_KIND = "low_mass_baryonic_resonance"
EVENT0_REMNANT_1_MASS = 1.535
EVENT0_CENTRAL_MASS = 0.2431513153059654
EVENT0_T1 = -0.13445599813262105
EVENT0_T2 = -0.35553908007869217
EVENT0_T_C = -0.4899950782113132

#: The pinned remnant census of the 10,000-event trace at the nominal
#: R50 split (intact/intact, mixed, low/low) and the pinned strangeness
#: and K K-bar structure of the same trace (total hyperons, central
#: hyperons, events carrying central kaons, events carrying a
#: K K-bar pair on the central line).
CENSUS_INTACT_INTACT = 2_558
CENSUS_MIXED = 4_951
CENSUS_LOW_LOW = 2_491
HYPERON_TOTAL = 225
CENTRAL_HYPERONS = 0
# re-pinned 2026-09-29 for the Block B stage 1 channel pool (the heavier direct decay classes)
# re-pinned 2026-10-02 for the kernel fixes (the CM-mirrored target diffraction, the N pi threshold, the radiative channels)
EVENTS_CENTRAL_KAONS = 77
EVENTS_KKBAR = 55

#: The pinned draw-budget totals of the 200-event budget trace (master
#: seed 20260925, replica 0, one substream quadruple per event index
#: 50,000 + i in event order): the frozen Round 4 per-vertex decay
#: budget totals on the ``decay_channel`` / ``decay_kinematics``
#: streams at this landing (a catalog handoff and the unstable-sibling
#: kernels add the Round 2 kernel's own documented draws on the same
#: streams; a stable remnant record and a below-threshold central
#: source consume zero).
# re-pinned 2026-09-30 for the corrected final state (decays at every assembly point, the isospin rule)
ND_CHANNEL_TOTAL = 350
ND_KINEMATICS_TOTAL = 850
ND_ZERO_DRAW_EVENTS = 31
ND_MIN_CHANNEL = 0
ND_MAX_CHANNEL = 5
ND_MIN_KINEMATICS = 0
ND_MAX_KINEMATICS = 14

#: The pinned central-decay totals of the 1,000-decay trace (the mass
#: stream at the dedicated event index 70,000, the decay substreams at
#: the event indices 60,000 + i): the below-threshold count (the
#: quantum-number-free single states at zero decay draws), the
#: ``decay_channel`` / ``decay_kinematics`` totals of the decaying
#: sources, and the K K-bar pair count.
CENTRAL_BELOW_THRESHOLD = 421
CENTRAL_CHANNEL_TOTAL = 520
CENTRAL_KINEMATICS_TOTAL = 1_040
CENTRAL_KKBAR_EVENTS = 4

#: The committed digests of the card's pinned surfaces (nested tree
#: HEAD a4f89ed; the solver test digest is the post-re-pin value, the
#: 522 -> 523 count re-pin of this register's landing).
MODULE_SHA256 = (
    "a62d578f4c7c6f0179cb48f4b92670ff631089e73b9d64592e52a93453d367e1")
#: re-pinned by the Round 5 card 8 test-author 525 -> 526 count re-pin, which
#: re-touched this solver test file (TASK-20260925-r5-multiple-neutral-excitations-test-author).
#: re-pinned by the Round 5 card 9 526 -> 528 count re-pin, which
#: re-touched this solver test file (TASK-20260925-r5-topology-template-runner).
#: re-pinned by the Round 5 card 9 test-author 528 -> 529 count re-pin, which
#: re-touched this solver test file (TASK-20260925-r5-topology-template-runner-test-author).
SOLVER_TEST_SHA256 = (
    "0629fe376fef27883a105b78cd169a23124f27a28a067eada08d1de93ef4285d")
# re-pinned by Round 5 card 8 (TASK-20260925-r5-multiple-neutral-excitations; the multiple-neutral-excitation module's additive package exports).
# re-pinned by Round 5 card 9 (TASK-20260925-r5-topology-template-runner; the topology-runner module's additive package exports).
# re-pinned by the Round 5 card 9 review fix round (TASK-20260925-r5-topology-template-runner-review-fix; the REV-bc4de47-03 export-surface correction re-pinned the package __init__).
INIT_SHA256 = (
    "7d2c5a85d9b24de104c0d9cf72195689bf7c39c873e154798699448073b36de8")

#: The module's ``__all__`` export set (nineteen names).
MODULE_ALL = [
    "ND_CLASS_LABEL",
    "ND_LINE_LABELS",
    "CENTRAL_LINE_LABEL",
    "PROJECTILE_LINE_LABEL",
    "TARGET_LINE_LABEL",
    "CENTRAL_QN",
    "PI0_MASS",
    "NOMINAL_PARAMS",
    "NOMINAL_ND_REMNANT_WEIGHTS",
    "BEAM_MOMENTUM",
    "PROTON_MASS",
    "SQRT_S",
    "CentralNeutralityViolation",
    "NDEvent",
    "check_central_neutrality",
    "remnant_excitation_window",
    "generate_nd",
    "nd_kinematics",
    "nd_limits",
]

#: The ``__init__`` non-diffractive export block (the eleven
#: re-exported names; ``NOMINAL_PARAMS``, ``BEAM_MOMENTUM``,
#: ``PROTON_MASS``, and ``SQRT_S`` stay card 1's elastic constants and
#: are module-level only, and the final-state hadron record stays
#: card 5's package-level ``FinalHadron`` export).
INIT_ND_NAMES = [
    "CENTRAL_LINE_LABEL",
    "CENTRAL_QN",
    "ND_CLASS_LABEL",
    "ND_LINE_LABELS",
    "NOMINAL_ND_REMNANT_WEIGHTS",
    "CentralNeutralityViolation",
    "NDEvent",
    "check_central_neutrality",
    "generate_nd",
    "nd_kinematics",
    "nd_limits",
]

#: The package ``__all__`` total entry count (the additive card 7
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
#: The solver card (Round 5, card 7) and this test-author register's
#: card name (the freeze-exception identity).
SOLVER_CARD = "TASK-20260925-r5-non-diffrative-central-excitation"
TEST_AUTHOR_CARD = (
    "TASK-20260925-r5-non-diffrative-central-excitation-test-author")

#: This register's relative path in the monorepo (the inventory key).
MY_TEST_REL = (
    "dev/mixing2006/tests/test_topology_non_diffractive_contract_pins.py")

#: The register chain: the thirty-two chain count-pin files re-pinned
#: 522 -> 523 by this register (the card-7 solver's thirty-one chain
#: files plus the card-7 solver test file, path-sorted).
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
    "dev/mixing2006/tests/test_topology_non_diffractive.py",
    "dev/mixing2006/tests/test_topology_remnants_contract_pins.py",
    "dev/mixing2006/tests/test_topology_single_diffraction_contract_pins.py",
)

#: This register's freeze-exception file list (path-sorted): the
#: manifest, this register, the card-7 solver test file, and the
#: thirty-one re-pinned chain count-pin files.
TEST_AUTHOR_EXCEPTION_FILES = tuple(sorted(
    ("dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json",)
    + tuple(rel for rel in CHAIN_COUNT_PIN_FILES
            if rel != "dev/mixing2006/tests/test_topology_non_diffractive.py")
    + ("dev/mixing2006/tests/test_topology_non_diffractive.py",)
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
    non-diffractive quadruple)."""

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


def _exchange_records(event: NDEvent) -> tuple[Exchange, Exchange]:
    """The card-2 ledger exchange records of the event's two vacuum
    exchanges (the test-side conversion of the card-1 record form)."""

    records = []
    for record in event.exchanges:
        records.append(Exchange(
            name=record.name,
            transfer=(record.transfer[0], record.transfer[1],
                      record.transfer[2], record.transfer[3])))
    return (records[0], records[1])


def _line_daughters(event: NDEvent, line: str) -> tuple[LineState, ...]:
    """The card-2 ledger daughters of one line of origin."""

    return tuple(
        LineState(B=h.qn[0], Q=h.qn[1], S=h.qn[2], I3x2=h.qn[3])
        for h in event.final_state if h.line == line)


def _decay_vertices(event: NDEvent):
    """The card-2 ledger decay vertices of the event: each baryonic
    line as the parent and its final-state hadrons as the daughters,
    plus the central source with its central-line daughters."""

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
    vertices.append((event.central_source, central_daughters))
    return tuple(vertices)


def _audit(event: NDEvent):
    """Card 2's five-rule audit over the event's ledger records."""

    return audit_event(
        incoming_lines=_incoming_lines(),
        exchanges=_exchange_records(event),
        outgoing_lines=event.lines,
        central_source=event.central_source,
        decay_vertices=_decay_vertices(event),
    )


def _relative_closure(event: NDEvent) -> float:
    """The worst relative four-momentum closure component of one event
    against the pinned incoming lab sum."""

    total = [sum(h.four_momentum[j] for h in event.final_state)
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
    """The pinned 10,000-event non-diffractive trace (master seed
    20260925, replica 0, one substream quadruple per event index)."""

    events = []
    for event_index in range(N_PINNED):
        rngs = _event_rngs(event_index)
        events.append(generate_nd(NOMINAL_PARAMS,
                                  NOMINAL_ND_REMNANT_WEIGHTS,
                                  catalog, fams, *rngs))
    return events


@pytest.fixture(scope="module")
def central_trace(catalog, fams):
    """The pinned central-decay trace: 1,000 central-source decays at
    the drawn masses of card 4's central-production law (one shared
    mass stream at the dedicated event index 70,000; one decay
    substream pair per decay at the event indices 60,000 + i,
    disjoint from the solver test's class traces).

    Returns ``(drawn, below)``: the per-decay final-state tuples and
    the per-decay draw counts ``(channel, kinematics)``.
    """

    m2_min, m2_max = nd_limits(PROTON_MASS, PROTON_MASS, SQRT_S)
    mass_rng = np.random.default_rng(derive_substreams(
        MASTER_SEED, 0, 70_000).stream_for("exchange_kinematics"))
    drawn = []
    below = []
    for index in range(N_CENTRAL):
        m2_c = draw_central(m2_min, m2_max, NOMINAL_PARAMS, -0.2, mass_rng)
        mass = math.sqrt(m2_c)
        subs = derive_substreams(MASTER_SEED, 0, 60_000 + index)
        rng_channel = np.random.default_rng(subs.stream_for("decay_channel"))
        rng_kinematics = np.random.default_rng(
            subs.stream_for("decay_kinematics"))
        finals = _decay_central(
            (mass, 0.0, 0.0, 0.0), mass, catalog, fams,
            rng_channel, rng_kinematics)
        drawn.append(finals)
        below.append((mass, rng_channel.bit_generator.state,
                      rng_kinematics.bit_generator.state, len(finals)))
    return drawn, below


# ----------------------------------------------------------------------
# Part 1: the module contract (pure, hermetic)
# ----------------------------------------------------------------------
def test_named_constants_and_record_surface():
    """The closed line-label set, the class label, the frozen record
    surfaces, the nominal parameters (the R50 fixed values:
    ``epsilon = 0``, ``B_SD`` the M8-frozen slope, the nominal R50
    remnant split per line), the central quantum numbers, the pion
    mass, and the closed-substream ``generate_nd`` signature."""

    assert ND_CLASS_LABEL == "nd"
    assert isinstance(ND_LINE_LABELS, frozenset)
    assert ND_LINE_LABELS == frozenset(
        {"projectile", "target", "central"})
    assert PROJECTILE_LINE_LABEL == "projectile"
    assert TARGET_LINE_LABEL == "target"
    assert CENTRAL_LINE_LABEL == "central"
    assert CENTRAL_QN == (0, 0, 0, 0)
    assert PI0_MASS == 0.1349766
    assert getattr(NDEvent, "__dataclass_params__").frozen
    assert tuple(NDEvent.__dataclass_fields__) == (
        "class_label", "remnants", "central_mass", "t_c", "t1", "t2",
        "lines", "central_source", "exchanges", "final_state",
        "remnant_windows", "central_component",
        "exchange_configuration", "slope_factor", "duality_edges")  # remnant_windows: 2026-09-29, default nan pair;
    # exchange_configuration: Round 8 2026-10-04, default None (the vacuum exchanges)
    # slope_factor: 2026-10-08, None = the single transfer slope (transfer_slope_mass_exponent off)
    # duality_edges: 2026-10-12, None = the continuum lines (the duality option off)
    assert NDEvent.__dataclass_fields__["duality_edges"].default is None
    assert NDEvent.__dataclass_fields__["exchange_configuration"].default is None
    # central_component: 2026-10-02, default True (the card's event)
    assert NDEvent.__dataclass_fields__["central_component"].default is True
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
    # the closed substream labels and the closed-substream generator
    # signature (no topology_selection stream: the class draw belongs
    # to the runner, card 9)
    for label in ("spectral_state", "exchange_kinematics",
                  "decay_channel", "decay_kinematics"):
        assert label in SUBSTREAM_LABELS
    assert "topology_selection" in SUBSTREAM_LABELS
    signature = inspect.signature(generate_nd)
    assert tuple(signature.parameters) == (
        "params", "weights", "catalog", "fams", "rng_spectral_state",
        "rng_exchange_kinematics", "rng_decay_channel",
        "rng_decay_kinematics", "flavor", "ledger", "central",
        "exchange", "exact_range", "line_thresholds", "central_recoil", "central_epsilon",
        "slope_mass_exponent", "low_mass_spin_isospin", "duality",
        "duality_line_shapes", "duality_redraw_shapes")  # duality_redraw_shapes: 2026-10-20, False = the fixed-width redraw  # central: 2026-10-02; exchange: Round 8 2026-10-04 (None = vacuum);
    # duality_line_shapes: 2026-10-18, False = the fixed-width Breit-Wigner pick;
    # low_mass_spin_isospin: 2026-10-08, the I = 1/2 low-mass records with weight 2J + 1 (False = the uniform pick);
    # exact_range: 2026-10-05, the exact transfer window (False = the committed draw);
    # line_thresholds: 2026-10-06, the per-line lower mass edges (False = the shared edge);
    # slope_mass_exponent: 2026-10-08, the factorized vertex slopes (None = the single slope)
    # duality: 2026-10-12, the low-mass lines through the N* and Delta records (None = the continuum lines)
    assert signature.parameters["duality"].default is None


def test_nd_limits_pins():
    """The exact central limits of the pinned intact-pair
    configuration (card 4's recorded values at ``SQRT_S = 17.270``),
    the lower limit at ``m_pi0^2`` exactly and the upper limit at
    ``(sqrt_s - M_1 - M_2)^2`` within ``1e-12`` relative, and the
    checked refusal of a non-positive remnant mass."""

    m2_min, m2_max = nd_limits(PROTON_MASS, PROTON_MASS, SQRT_S)
    assert m2_min == M2_MIN_PINNED
    assert m2_max == M2_MAX_PINNED
    assert m2_min == PI0_MASS * PI0_MASS
    upper = (SQRT_S - PROTON_MASS - PROTON_MASS) ** 2
    assert abs(m2_max - upper) / upper <= 1e-12
    # the checked refusal: a non-positive remnant mass
    with pytest.raises(ValueError):
        nd_limits(-PROTON_MASS, PROTON_MASS, SQRT_S)
    with pytest.raises(ValueError):
        nd_limits(PROTON_MASS, -PROTON_MASS, SQRT_S)


def test_nd_kinematics_pins():
    """The exact three-system solve on the pinned grid point: the
    four momenta at their committed lab values, the incoming-sum
    closure within ``1e-9`` relative, the mass shells, the central
    system at rest in the overall CM, the pair invariant at
    ``sqrt_s - M_c``, the back-to-back coupling in the CM, the
    realized projectile-side transfer at the drawn value within
    ``1e-9``, and the checked refusal of a positive drawn ``t_2``."""

    system1, system2, central = nd_kinematics(
        PROTON_MASS, GRID_T1, PROTON_MASS, GRID_T2, GRID_M_C, SQRT_S)
    for got, want in ((system1, GRID_SYSTEM1), (system2, GRID_SYSTEM2),
                      (central, GRID_CENTRAL)):
        for i in range(4):
            assert abs(got[i] - want[i]) <= 1e-12, (i, got, want)

    # the incoming-sum closure within 1e-9 relative
    reference = _incoming_sum()
    total = [system1[i] + system2[i] + central[i] for i in range(4)]
    e_total = reference[0]
    p_beam = reference[3]
    worst = max(abs(total[0] - reference[0]) / e_total,
                abs(total[1]) / p_beam,
                abs(total[2]) / p_beam,
                abs(total[3] - reference[3]) / p_beam)
    assert worst <= 1e-9

    # the mass shells (both baryonic systems at the proton mass, the
    # central system at the drawn mass)
    assert abs(_invariant_mass(system1) - PROTON_MASS) <= 1e-9
    assert abs(_invariant_mass(system2) - PROTON_MASS) <= 1e-9
    assert abs(_invariant_mass(central) - GRID_M_C) <= 1e-9

    # the central system at rest in the overall CM
    central_cm = _to_cm(central)
    assert abs(central_cm[1]) <= 1e-12
    assert abs(central_cm[2]) <= 1e-12
    assert abs(central_cm[3]) <= 1e-12
    assert abs(central_cm[0] - GRID_M_C) <= 1e-9

    # the pair invariant at the exact CM energy of the pinned beam
    # configuration minus M_c (the module's operative convention: the
    # exact three-system solve runs at the exact fixed-target CM energy
    # sq_exact, the SQRT_S constant being its rounded quote) and the
    # back-to-back coupling in the CM (the exact two-body kinematics of
    # the (M_1, M_2) pair)
    _e_beam, _e_total, sq_exact, _p_star, _beta, _gamma = (
        _beam_configuration())
    pair = [system1[i] + system2[i] for i in range(4)]
    assert abs(_invariant_mass(pair) - (sq_exact - GRID_M_C)) <= 1e-9
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

    # the checked refusal: a positive drawn t2
    with pytest.raises(ValueError):
        nd_kinematics(PROTON_MASS, GRID_T1, PROTON_MASS, 0.42, GRID_M_C,
                      SQRT_S)


def test_event_record_pins(catalog, fams):
    """The pinned event-0 record at its committed values (the remnant
    kinds and masses, the central mass, the two drawn transfers, and
    the exact ``t_c = t_1 + t_2`` identity), the ledger lines at
    ``(1, 1, 0, 1)``, the two vacuum exchange records with the zero
    transfer, and the final-state line labels on the closed set."""

    event = generate_nd(NOMINAL_PARAMS, NOMINAL_ND_REMNANT_WEIGHTS,
                        catalog, fams, *_event_rngs(0))
    assert event.class_label == ND_CLASS_LABEL == "nd"
    assert event.remnants[0].kind == EVENT0_REMNANT_0_KIND
    assert event.remnants[0].mass == EVENT0_REMNANT_0_MASS
    assert event.remnants[1].kind == EVENT0_REMNANT_1_KIND
    assert event.remnants[1].mass == EVENT0_REMNANT_1_MASS
    assert event.central_mass == EVENT0_CENTRAL_MASS
    assert event.t1 == EVENT0_T1
    assert event.t2 == EVENT0_T2
    assert event.t_c == EVENT0_T_C
    assert event.t_c == event.t1 + event.t2
    # the ledger lines: both baryonic lines at (1, 1, 0, 1) after the
    # vacuum exchanges' zero transfers; the central source at (0, 0, 0, 0)
    assert event.lines[0].qn == (1, 1, 0, 1)
    assert event.lines[1].qn == (1, 1, 0, 1)
    assert event.central_source.qn == CENTRAL_QN == (0, 0, 0, 0)
    # the two vacuum exchange records in card 1's form
    for record in event.exchanges:
        assert isinstance(record, ExchangeRecord)
        assert record.name == "vacuum"
        assert tuple(record.transfer) == (0, 0, 0, 0)
    # the final-state line labels on the closed set
    assert event.final_state
    assert all(h.line in ND_LINE_LABELS for h in event.final_state)
    assert any(h.line == PROJECTILE_LINE_LABEL for h in event.final_state)
    assert any(h.line == TARGET_LINE_LABEL for h in event.final_state)
    assert any(h.line == CENTRAL_LINE_LABEL for h in event.final_state)


def test_draw_budget_pins(catalog, fams):
    """A pinned 200-event trace on one substream quadruple per event
    index (50,000 + i) consumes exactly the pre-registered counts on
    the closed labels: 2 ``spectral_state`` uniforms per event (one
    remnant selection per baryonic line) and the
    ``exchange_kinematics`` budget of the card-4 decomposition (the
    ``t_1`` draw, the ``t_2`` draw, and the two-uniform ``M_c^2``
    central draw: 4 uniforms per event; the card's "5" literal is the
    recorded operative resolution), plus the frozen Round 4
    per-vertex budget on ``decay_channel`` / ``decay_kinematics``
    (the pinned per-event totals and min/max at this landing; pinned
    RNG-state identity; the ``topology_selection`` draw belongs to the
    runner, card 9)."""

    per_event_channel: list[int] = []
    per_event_kinematics: list[int] = []
    spectral_total = 0
    exchange_total = 0
    for event_index in range(N_BUDGET):
        subs = derive_substreams(MASTER_SEED, 0, 50_000 + event_index)
        counter_spectral = _CountingRng(np.random.default_rng(
            subs.stream_for("spectral_state")))
        counter_exchange = _CountingRng(np.random.default_rng(
            subs.stream_for("exchange_kinematics")))
        counter_channel = _CountingRng(np.random.default_rng(
            subs.stream_for("decay_channel")))
        counter_kinematics = _CountingRng(np.random.default_rng(
            subs.stream_for("decay_kinematics")))
        generate_nd(NOMINAL_PARAMS, NOMINAL_ND_REMNANT_WEIGHTS, catalog,
                    fams, counter_spectral, counter_exchange,
                    counter_channel, counter_kinematics)
        assert counter_spectral.calls == 2
        assert counter_exchange.calls == 4
        spectral_total += counter_spectral.calls
        exchange_total += counter_exchange.calls
        per_event_channel.append(counter_channel.calls)
        per_event_kinematics.append(counter_kinematics.calls)

    # the kinematic labels: exactly the pre-registered counts per event
    assert spectral_total == 2 * N_BUDGET
    assert exchange_total == 4 * N_BUDGET

    # the frozen Round 4 per-vertex budget on the decay labels (the
    # pinned per-event totals and min/max at this landing; a catalog
    # handoff and the unstable-sibling kernels add the Round 2
    # kernel's own documented draws on the same streams; a stable
    # remnant record and a below-threshold central source consume zero)
    assert sum(per_event_channel) == ND_CHANNEL_TOTAL
    assert sum(per_event_kinematics) == ND_KINEMATICS_TOTAL
    assert min(per_event_channel) == ND_MIN_CHANNEL
    assert max(per_event_channel) == ND_MAX_CHANNEL
    assert min(per_event_kinematics) == ND_MIN_KINEMATICS
    assert max(per_event_kinematics) == ND_MAX_KINEMATICS
    assert all(k >= 2 * c
               for c, k in zip(per_event_channel, per_event_kinematics))
    assert sum(1 for c in per_event_channel if c == 0) \
        == ND_ZERO_DRAW_EVENTS


def test_central_neutrality_pins(pinned_trace):
    """On the pinned 10,000-event trace the central source carries
    ``(B, Q, S, 2I3) = (0, 0, 0, 0)`` exactly and every central decay
    vertex closes locally to ``(0, 0, 0)``; ``check_central_neutrality``
    raises on each planted violation (central ``B = 1``, central
    ``Q = 1``, central ``S = 1``, one open central vertex), naming the
    offending component or the open vertex, and refuses a foreign
    record type."""

    assert len(pinned_trace) == N_PINNED
    for event in pinned_trace:
        assert event.central_source.qn == CENTRAL_QN == (0, 0, 0, 0)
        check_central_neutrality(event)  # raises on any violation

    # the planted violations: each is detected, naming the offending
    # component or the open vertex
    event = pinned_trace[0]
    source = event.central_source
    for label, planted in (("B", (1, 0, 0, 0)),
                           ("Q", (0, 1, 0, 0)),
                           ("S", (0, 0, 1, 0))):
        bad_source = LineState(B=planted[0], Q=planted[1], S=planted[2],
                               I3x2=planted[3],
                               four_momentum=source.four_momentum)
        bad_event = dataclasses.replace(event, central_source=bad_source)
        with pytest.raises(CentralNeutralityViolation) as excinfo:
            check_central_neutrality(bad_event)
        assert f"{label} = 1" in str(excinfo.value)

    # one open central vertex: a central-line daughter doctored to
    # carry baryon number breaks the local closure
    central = [h for h in event.final_state if h.line == CENTRAL_LINE_LABEL]
    assert central
    doctored = list(event.final_state)
    for index, hadron in enumerate(doctored):
        if hadron.line == CENTRAL_LINE_LABEL:
            doctored[index] = dataclasses.replace(hadron, qn=(1, 0, 0, 0))
            break
    open_event = dataclasses.replace(event, final_state=tuple(doctored))
    with pytest.raises(CentralNeutralityViolation) as excinfo:
        check_central_neutrality(open_event)
    assert "open" in str(excinfo.value)

    # the check refuses a foreign record type
    with pytest.raises(ValueError):
        check_central_neutrality("not an event")


def test_conservation_pins(pinned_trace):
    """Over the pinned 10,000-event trace the final state carries
    exactly ``(B, Q, S) = (2, 2, 0)``, the four-momentum closure is
    within ``1e-9`` relative at the pinned worst value, the remnant
    census is pinned at the nominal R50 split, ``t_c = t_1 + t_2``
    exactly on every event, and every event passes card 2's
    ``audit_event`` with zero violations (two vacuum transfers, both
    baryonic lines ``(1, 1, 0, 1)``, the central source ``(0, 0, 0,
    0)``, the local decay closures)."""

    assert len(pinned_trace) == N_PINNED
    census = {"intact_intact": 0, "mixed": 0, "low_low": 0}
    worst_closure = 0.0
    for event in pinned_trace:
        b = sum(h.qn[0] for h in event.final_state)
        q = sum(h.qn[1] for h in event.final_state)
        s = sum(h.qn[2] for h in event.final_state)
        assert (b, q, s) == (2, 2, 0)
        assert event.t_c == event.t1 + event.t2
        worst_closure = max(worst_closure, _relative_closure(event))
        assert not _audit(event).violations
        i0 = event.remnants[0].kind == KIND_INTACT_PROTON
        i1 = event.remnants[1].kind == KIND_INTACT_PROTON
        if i0 and i1:
            census["intact_intact"] += 1
        elif i0 != i1:
            census["mixed"] += 1
        else:
            census["low_low"] += 1
    assert worst_closure <= 1e-9
    assert census == {
        "intact_intact": CENSUS_INTACT_INTACT,
        "mixed": CENSUS_MIXED,
        "low_low": CENSUS_LOW_LOW,
    }


def test_central_channel_pins(catalog, fams, pinned_trace, central_trace):
    """On the pinned central-decay trace the channel daughters sum to
    ``(0, 0, 0)`` exactly, the below-threshold central sources stay
    single quantum-number-free final states at their drawn mass with
    zero decay draws (the pinned count), and the K K-bar direct
    channels occur (the pinned count); on the pinned event trace the
    hyperons appear only on the baryonic lines (the pinned census),
    the central kaon events are net neutral in charge and strangeness
    (the pinned count), and the K K-bar direct channel events are
    pinned; the committed ``(0, 0, 0)`` records f2(1270),
    rho3(1690)0, f4(2050), and phi(1020) carry the recorded K K-bar
    pair classes, and the phi record carries both the ``K+ K-`` and
    the ``K0S K0L`` channels (plan section 11.3)."""

    drawn, below = central_trace
    assert len(drawn) == N_CENTRAL
    n_below = 0
    n_kkbar = 0
    for index, finals in enumerate(drawn):
        assert (sum(h.qn[0] for h in finals),
                sum(h.qn[1] for h in finals),
                sum(h.qn[2] for h in finals)) == (0, 0, 0)
        mass = below[index][0]
        if len(finals) == 1 and finals[0].pdg_id is None:
            # the below-threshold branch: a single quantum-number-free
            # final state at the drawn mass with no vertex
            n_below += 1
            state = finals[0]
            assert state.qn == (0, 0, 0, 0)
            assert state.mass == mass
            assert state.four_momentum == (mass, 0.0, 0.0, 0.0)
        pdgs = [h.pdg_id for h in finals]
        if (321 in pdgs and -321 in pdgs) or (310 in pdgs and 130 in pdgs):
            n_kkbar += 1
    assert n_below == CENTRAL_BELOW_THRESHOLD
    assert n_kkbar == CENTRAL_KKBAR_EVENTS

    # the committed (0, 0, 0) records carry the K K-bar pair classes
    # (plan section 11.3) and the phi record carries both recorded
    # channels
    for pdg_id in (225, 117, 229, 333):
        record = catalog.get(pdg_id)
        assert (record.B, record.Q, record.S) == (0, 0, 0)
        pairs = {tuple(c.daughters) for c in record.decay_channels
                 if not c.is_three_body}
        assert (321, -321) in pairs or (310, 130) in pairs, record.name
    phi_pairs = {tuple(c.daughters) for c in catalog.get(333)
                 .decay_channels if not c.is_three_body}
    assert (321, -321) in phi_pairs
    assert (310, 130) in phi_pairs

    # the K K-bar direct channels of the family are open above the
    # two-kaon threshold at a mass above it
    root = CascadeState(
        momentum=(2.7, 0.0, 0.0, 0.0), mass=2.7, qn=(0, 0, 0),
        quantum={}, family_id="mesonic_neutral", axis=(0.0, 0.0, 1.0),
        alignment_label=ISOTROPIC_ALIGNMENT_LABEL, parent=None,
        exchange_ancestry=("nd",), depth=0)
    direct_pairs = {
        tuple(channel.daughters)
        for channel in enumerate_channels(root, catalog, fams)
        if channel.kind != "continuum"}
    assert ("catalog:321", "catalog:-321") in direct_pairs
    assert ("catalog:310", "catalog:130") in direct_pairs

    # on the pinned event trace: the hyperons only on the baryonic
    # lines, the central kaon events net neutral, the K K-bar direct
    # channel events
    hyperon_pdgs = (3122, 3124, 3222, 3224, 3312, 3322, 3334)
    kaon_pdgs = (321, -321, 310, 130)
    n_hyperon = 0
    n_central_hyperon = 0
    n_central_kaon_events = 0
    n_kkbar_events = 0
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
        for hadron in event.final_state:
            if hadron.pdg_id in hyperon_pdgs:
                n_hyperon += 1
                assert hadron.line in (PROJECTILE_LINE_LABEL,
                                       TARGET_LINE_LABEL)
                if hadron.line == CENTRAL_LINE_LABEL:
                    n_central_hyperon += 1
    assert n_hyperon == HYPERON_TOTAL
    assert n_central_hyperon == CENTRAL_HYPERONS
    assert n_central_kaon_events == EVENTS_CENTRAL_KAONS
    assert n_kkbar_events == EVENTS_KKBAR


def test_frozen_cascade_pins(catalog, fams):
    """All three systems decay through the frozen simplified cascade
    (kappa_B = 0.0, kappa_M = 4.0, b_dec = 2.0 GeV^-2, baryonic nominal
    width 0.3 GeV, pinned through the R45 driver digests in both
    helper trees); the ``W_flavor = 1.0`` hook identity on the channel
    weight, the neutral-mesonic ``mesonic_neutral`` continuum family
    of the central source, and the R46 ``no_refit`` white-box check
    (no module in this card references a re-fit of the alignment
    parameters or claims the separation of exchange recoil from decay
    transverse dynamics)."""

    # the frozen simplified cascade is consumed read-only: the module
    # imports the sealed Round 4 machinery (the termination driver,
    # the decay machinery, the alignment and form-factor kernels) and
    # introduces no alignment re-fit and no separation claim (the R46
    # clauses; the white-box token scan)
    import na49_mixing.cascade_alignment  # noqa: F401
    import na49_mixing.cascade_form_factor  # noqa: F401
    import na49_mixing.cascade_termination  # noqa: F401
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert 'CENTRAL_FAMILY_ID: str = "mesonic_neutral"' in source
    assert "handoff_decay" in source and "draw_decay" in source
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

    # one event's three decays close locally through the ledger's local
    # decay closure and the central source stays neutral
    event = generate_nd(NOMINAL_PARAMS, NOMINAL_ND_REMNANT_WEIGHTS,
                        catalog, fams, *_event_rngs(0))
    assert event.lines[0].qn == (1, 1, 0, 1)
    assert event.lines[1].qn == (1, 1, 0, 1)
    assert event.central_source.qn == CENTRAL_QN
    report = _audit(event)
    assert not report.violations
    check_central_neutrality(event)


# ----------------------------------------------------------------------
# Part 2: the substrate and freeze pins
# ----------------------------------------------------------------------
def test_module_byte_and_export_pins():
    """The module at its committed sha256 (nested tree HEAD a4f89ed)
    with its exact ``__all__`` export set (nineteen names), the
    solver test file at its committed sha256 (the post-re-pin value,
    the 522 -> 523 count re-pin of this register's landing), and the
    package ``__init__`` at its committed sha256 carrying the
    non-diffractive export block (the eleven re-exported names; the
    elastic constants stay card 1's module-level constants, and the
    final-state hadron record stays card 5's package-level
    ``FinalHadron`` export)."""

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    import na49_mixing.topology_non_diffractive as _mod
    assert _mod.__all__ == MODULE_ALL

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)
    init_text = INIT_PATH.read_text(encoding="utf-8")
    block = re.search(
        r"from \.topology_non_diffractive import \((.*?)\)",
        init_text, re.S)
    assert block is not None, (
        "the __init__ non-diffractive import block is missing")
    for name in INIT_ND_NAMES:
        assert re.search(rf"\b{name}\b", block.group(1)), (
            f"{name} is not in the __init__ non-diffractive import "
            "block")
    # the elastic constants and the card-5 record stay module-level
    # only (not re-exported through the non-diffractive block)
    for name in ("BEAM_MOMENTUM", "PROTON_MASS", "SQRT_S", "FinalHadron",
                 "NOMINAL_PARAMS"):
        assert not re.search(rf"\b{name}\b", block.group(1)), (
            f"{name} must not enter the __init__ non-diffractive "
            "import block")
    all_block = re.search(r"__all__\s*=\s*\[(.*?)\n\]", init_text, re.S)
    assert all_block is not None
    all_names = re.findall(r'"([^"]+)"', all_block.group(1))
    assert len(all_names) == INIT_ALL_COUNT
    for name in INIT_ND_NAMES:
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
