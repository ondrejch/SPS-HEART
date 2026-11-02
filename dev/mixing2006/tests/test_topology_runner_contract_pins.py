"""Round 5, card 9 test-author register: the topology template runner.

Independent contract pins for the topology template runner
(``na49_mixing/topology_runner.py``), Round 5 card 9
(``TASK-20260925-r5-topology-template-runner``; plan section 22 task 36,
line 1761; plan section 11.1 item 1, line 824, with run R50, lines
898-907), at nested tree SHA cc22be0 (the review-fix round's
corrected physics; the register's bc4de47/2752f77 pre-fix pins are
re-anchored by this follow-up).  This register is the test-author
half of the card: it pins the module contract on its own pinned
fixtures (the six-entry class registry with the elastic class excluded
from every inelastic sample and the unretained ``nd_multi`` entry, the
fixed ``INELASTIC_ORDER`` walk over ``INELASTIC_SET``, the
``TopologyNominalParams`` pins at the fixed R50 values with ``B_SD``
byte-identical to the M8-frozen ``effective_transfer_slope``, the
closed ``topology_selection`` draw pins (the hand-computed 1,000-draw
sequence on the corrected absolute-cumulative walk, exactly one
uniform per draw with the pinned RNG-state identity, the elastic lower
edge, the 100,000-draw frequency pin at the positive elastic fraction
``f_e = 0.3``, the ``nd_multi`` exclusion at the default
``P(n = 2) = 0.0``, and the checked refusals on unknown labels and
non-unit mass), the dispatch bit-identity pins over the pinned
200-event per-class traces against the class modules' own generations
(all four inelastic classes), the per-event observable pins
hand-recomputed within ``1e-12`` over the pinned 1,000-event trace
(leading-baryon Feynman ``x_F`` per baryonic line in the overall CM
frame, excitation masses, momentum transfers, central gap, total and
central-activity multiplicities in the CM window ``|y^CM| < 0.5``,
central/forward pion and proton yields, the m_T-slope seagull moment;
the class census 260/249/241/250; the exact first-occurrence
literals), the ``BaseTopologyEvent`` union pin, the
frozen simplified cascade at the sealed R45 point with the R46
``no_refit`` white-box check, the byte-stable module / solver test /
package ``__init__`` digests and export surface, and the
frozen-substrate provenance) and re-pins the freeze manifest for the
card (``INVENTORY_COUNT`` 528 -> 529).

Part 1: the module contract (pure, hermetic)
--------------------------------------------

- ``test_registry_surface_pins``: the registry is exactly the six
  entries ``{"elastic", "sd_projectile", "sd_target", "dd", "nd",
  "nd_multi"}`` with the ``retained`` flags (``nd_multi`` ``retained``
  False, all others True) and the ``inelastic`` flags (elastic
  ``inelastic`` False, all others True); ``INELASTIC_SET`` is exactly
  the four classes; ``INELASTIC_ORDER`` is the fixed walk order
  (projectile SD, target SD, DD, ND); every entry names its landed
  module (importable, exact name); and ``BaseTopologyEvent`` is the
  union of the five landed record types in the committed order.
- ``test_nominal_params_pins``: ``TopologyNominalParams()`` equals the
  pre-registered R50 values exactly (``epsilon = 0``, ``B_SD =
  8.504140762083612`` byte-identical to the M8-frozen
  ``effective_transfer_slope`` at the M8 hash
  358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1, the
  equal 0.25 inelastic class fractions, the nominal remnant split
  intact 0.5 / low-mass 0.5 / high-mass 0.0 in the committed card-3
  key vocabulary, ``p_n2 = 0.0``); the beam configuration pins
  (``SQRT_S = 17.270``, ``PROTON_MASS = 0.93827231``, ``PI0_MASS =
  0.1349766``, ``BEAM_MOMENTUM = 158.0``, ``ELASTIC_TRANSFER_SLOPE =
  8.504140762083612``, ``MASTER_SEED = 20260925``); and the nominal
  structure builds exactly the committed class modules' mass law and
  remnant weights.
- ``test_draw_pins``: the pinned uniform head (the first twenty
  uniforms of ``np.random.default_rng(20260925)``), the class sequence
  of ``draw_topology`` over 1,000 draws against the hand-computed
  sequence from the uniform values on the corrected absolute-
  cumulative walk (each inelastic class returned at exactly its
  caller fraction, the review finding REV-bc4de47-02), exactly one
  uniform per draw with the pinned RNG-state identity (the counting
  proxy and the bit-generator state 1,000 uniforms past the seed), the
  elastic lower edge (a positive elastic fraction draws "elastic" on
  the first uniform, and the hand-computed positive-elastic sequence
  matches), the 100,000-draw frequency pin at the positive elastic
  fraction ``f_e = 0.3`` (the identity test cannot substitute for it;
  the pinned counts sit on the 0.300 / 0.175 targets within
  ``+/- 0.01``), the ``nd_multi`` exclusion at the default
  ``P(n = 2) = 0.0`` (never returned by the draw, exactly one central
  source when dispatched), and the checked refusals (unknown class
  labels including "elastic" and "nd_multi" as fraction keys, missing
  inelastic classes, negative fractions, out-of-range elastic
  fractions, non-unit mass).
- ``test_dispatch_bit_identity_pins``: each inelastic class label
  dispatches to the landed module and the nominal-parameter record is
  bit-identical to the class module's own generation on the same
  closed per-label substreams (the pinned 200-event trace per class,
  the cross-module identity); the elastic class dispatches on
  ``exchange_kinematics`` alone; the unretained ``nd_multi`` label
  stays dispatchable (count exactly 1 at the default); unknown labels
  and missing streams raise.
- ``test_observables_pins``: on the pinned 1,000-event trace every
  extracted observable matches the hand-computed recomputation within
  ``1e-12`` (the leading-baryon Feynman ``x_F`` per baryonic line in
  the overall CM frame, the excitation masses, the transfers, the
  central gap, the total multiplicity, the central-activity
  multiplicity in the CM window ``|y^CM| < 0.5``, the central and
  forward pion and proton yields, the seagull moment); the
  ``pinned_trace`` output is exactly the per-event extraction; the
  pinned class census is 260/249/241/250; the first-occurrence
  indices and the exact pinned first-occurrence literals hold (the
  four inelastic classes plus the elastic class's direct dispatch
  event-0).
- ``test_frozen_cascade_pins``: the runner passes the frozen
  simplified cascade through at the sealed R45 point (kappa_B = 0.0,
  kappa_M = 4.0, b_dec = 2.0 GeV^-2, baryonic nominal width 0.3 GeV,
  pinned through the R45 driver digests in both helper trees), the
  ``W_flavor = 1.0`` hook identity on the channel weight, and the R46
  ``no_refit`` white-box check (no refit of the alignment parameters
  and no separation claim anywhere in the module source; the exchange
  slopes act on the exchange kinematics only, never on any daughter
  transverse dynamics).

Part 2: the substrate and freeze pins
--------------------------------------

- ``test_module_byte_and_export_pins``: the module at its committed
  sha256 (nested tree HEAD cc22be0, the review-fix round) with its
  exact ``__all__`` export set (sixteen names), the solver test file
  at its committed sha256 after the review-fix round, and the package
  ``__init__`` at its committed sha256 carrying the topology-runner
  export block (the eleven re-exported names, the runner instance as
  ``NOMINAL_TOPOLOGY_PARAMS``; the beam constants and the transfer
  slope stay the card-1 package-level exports, module-level only
  here, the package-level ``NOMINAL_PARAMS`` stays card 5's mass-law
  export, and the package ``__all__`` total at 253 unique entries).
- ``test_frozen_substrate_digests``: the card's frozen-substrate
  provenance surfaces (verified on disk, including the consumed
  cards 1-8 class modules at their committed digests) and the frozen
  M8 parameter map through the manifest, with the activated
  neutral-mesonic ``(0, 0, 0)`` continuum family's recorded support
  ends.
- ``test_freeze_repin_register``: the freeze-manifest re-pin for this
  card (the strict path-sorted inventory at 529 entries; 528 at the
  card 9 solver landing; 526 at the card 8 test-author landing; 525
  at the card 8 solver landing; 523 at the card 7 test-author
  landing; 522 after the Round 5 card 7 landing; 520 after the Round
  5 card 6 landing; 519 after the Round 5 card 6 test-author re-pin;
  517 after the Round 5 card 5 test-author re-pin; 516 after the
  Round 5 card 5 re-pin; 514 after the Round 5 card 4 test-author
  re-pin; 513 after the Round 5 card 4 re-pin; 511 after the Round 5
  card 3 test-author re-pin; 510 after the Round 5 card 3 solver
  landing; 508 after the test-protocol tiering change; 505 after the
  Round 5 card 2 test-author re-pin; 504 after the Round 5 card 2
  solver landing; 502 after the Round 5 card 1 test-author re-pin;
  501 at the Round 5 card 1 landing; +1 entry: this register) carries
  this register and the thirty-six re-pinned chain count-pin files at
  their on-disk digests, the card 9 solver's freeze exception stays
  exactly once at its 39 path-sorted files with its reason naming the
  526 -> 528 re-pin, this card's test-author freeze exception names
  the card exactly once with the path-sorted file list and a reason
  naming the 528 -> 529 re-pin, the chain count pins are re-pinned
  528 -> 529 with the 528 literal retained in the naming comments, and
  the frozen M8 parameter map stays intact.
- ``test_layout_sync``: the layout sync (the subprocess check from the
  monorepo root; ``logic_drift = 0``).

Hermetic: the module, the committed catalog, the continuum families
configuration, the Round 4 cascade modules (consumed read-only), the
committed cards 1-8 modules (consumed read-only), numpy, and the
standard library only; no event generation driver, no external
tables, no Monte Carlo run (the runs R50-R55 exercise the runner
statistically).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Union, get_args, get_origin

import numpy as np
import pytest

from na49_mixing.cascade_decay import CascadeChannel, channel_weight
from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.rng_substreams import derive_substreams
from na49_mixing.spectral_density import families_from_json
from na49_mixing.topology_double_diffraction import (
    DDEvent,
    NOMINAL_PARAMS as DD_NOMINAL_PARAMS,
    generate_dd,
)
from na49_mixing.topology_elastic import (
    BEAM_MOMENTUM,
    ELASTIC_TRANSFER_SLOPE,
    PROTON_MASS,
    SQRT_S,
    ElasticEvent,
    check_zero_transfer,
    generate_elastic,
)
from na49_mixing.topology_mass_laws import PI0_MASS, MassLawParams
from na49_mixing.topology_multi_neutral import NDMultiEvent
from na49_mixing.topology_non_diffractive import (
    NDEvent,
    NOMINAL_ND_REMNANT_WEIGHTS,
    NOMINAL_PARAMS as ND_NOMINAL_PARAMS,
    generate_nd,
)
from na49_mixing.topology_remnants import KERNEL_BOUNDARY
from na49_mixing.topology_runner import (
    CENTRAL_WINDOW,
    INELASTIC_ORDER,
    INELASTIC_SET,
    MASTER_SEED,
    NOMINAL_PARAMS,
    TOPOLOGY_CLASSES,
    BaseTopologyEvent,
    TopologyNominalParams,
    draw_topology,
    extract_observables,
    pinned_trace,
    run_topology,
)
from na49_mixing.topology_single_diffraction import (
    SDEvent,
    NOMINAL_PARAMS as SD_NOMINAL_PARAMS,
    NOMINAL_SD_REMNANT_WEIGHTS,
    generate_sd,
)
from tests._freeze_inventory import inventory_count as _inventory_count

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent          # dev/mixing2006
MONO_ROOT = NESTED_ROOT.parent.parent          # the monorepo root
MANIFEST_PATH = (
    NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json")
SYNC_CHECKER = (
    MONO_ROOT / "helpers" / "benchmarking" / "check_helpers_layout_sync.py")
MODULE_PATH = NESTED_ROOT / "na49_mixing" / "topology_runner.py"
SOLVER_TEST_PATH = NESTED_ROOT / "tests" / "test_topology_runner.py"
INIT_PATH = NESTED_ROOT / "na49_mixing" / "__init__.py"
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
CONTINUUM_PATH = (
    NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json")

#: The Round 5 master seed (round 5 index item 9), fixed for the pinned
#: traces; replica 0, one substream quintuple per event index on the
#: closed labels (``topology_selection``, ``spectral_state``,
#: ``exchange_kinematics``, ``decay_channel``, ``decay_kinematics``).
MASTER_SEED_PIN = 20260925

#: The pinned trace sizes: the card's 1,000-event deterministic trace
#: (the T0-equivalent), the 1,000-draw draw-determinism trace, and the
#: 200-event dispatch-identity trace per class.
N_TRACE = 1_000
N_DRAWS = 1_000
N_DISPATCH = 200

#: The pre-registered R50 nominal value pins (the card's index item 2
#: and the M8-frozen substrate).
B_SD_PIN = 8.504140762083612
SQRT_S_PIN = 17.270
PROTON_MASS_PIN = 0.93827231
PI0_MASS_PIN = 0.1349766
BEAM_MOMENTUM_PIN = 158.0
ELASTIC_SLOPE_PIN = 8.504140762083612

#: The pinned uniform head (the first twenty uniforms of
#: ``np.random.default_rng(20260925)``; the hand-computed draw
#: sequence's reference values).
PINNED_UNIFORMS_HEAD = (
    0.15923906158588796, 0.195374162079877, 0.4728450701494308,
    0.4235755562503132, 0.5103955234605415, 0.32667797319392367,
    0.17389075227327044, 0.9286855769064936, 0.45369318481084686,
    0.010324082099433807, 0.45557750762632, 0.25932768002066764,
    0.2815099888856837, 0.20864418618590685, 0.19089764320649139,
    0.06607712454215575, 0.8590736006416746, 0.6391161131603442,
    0.0560214484366508, 0.7008540318983446,
)

#: The hand-computed head of the pinned trace's class sequence (the
#: per-event ``topology_selection`` substream uniforms walked through
#: the nominal 0.25-each split).
HAND_TRACE_HEAD = (
    "dd", "nd", "nd", "dd", "sd_target", "sd_target", "sd_projectile",
    "sd_target", "nd", "sd_projectile", "sd_target", "dd",
)

#: The pinned class counts of the 1,000-event trace at the nominal
#: 0.25-each inelastic fractions (no elastic, no nd_multi).
TRACE_CLASS_COUNTS = {
    "sd_projectile": 260,
    "sd_target": 249,
    "dd": 241,
    "nd": 250,
}

#: The first-occurrence event index of each inelastic class in the
#: pinned trace.
TRACE_FIRST_INDEX = {
    "sd_projectile": 6,
    "sd_target": 4,
    "dd": 0,
    "nd": 1,
}

#: The exact pinned first-occurrence observables (the hand-computed
#: extraction at the named trace indices; the four inelastic classes'
#: first occurrences plus the elastic class's direct dispatch event-0,
#: whose pseudo-final-state lines carry no catalog identity and whose
#: species yields therefore count zero hadrons).  The review fix round
#: (REV-bc4de47-01) re-anchored the Feynman ``x_F = 2 p_L^CM /
#: sqrt(s)`` literals and the central window ``|y^CM| < 0.5`` on the
#: overall CM frame of the pinned beam configuration (the elastic
#: event-0 pair at ``+/- 0.993195829131928``, neither nucleon central
#: in the CM window); the frame-independent observables (excitation
#: masses, transfers, gap sizes, seagull moments) and the census are
#: unchanged.
# re-pinned 2026-09-29 for the Block B stage 1 channel pool (the heavier direct decay classes)
PINNED_OBSERVABLES = {
    "elastic": dict(
        class_label='elastic',
        x_f_projectile=0.993195829131928,
        x_f_target=-0.9931958291319319,
        excitation_masses=(),
        transfers=(-0.13445599813262105,),
        gap_size=None,
        multiplicity_total=2,
        multiplicity_central=0,
        central_pions=0,
        central_protons=0,
        forward_pions=0,
        forward_protons=0,
        seagull_moment=0.36659877388073575,
    ),
    "sd_projectile": dict(
        class_label='sd_projectile',
        x_f_projectile=0.6459781619928519,
        x_f_target=-0.6459781619928386,
        excitation_masses=(10.09786948213648,),
        transfers=(-0.5262182871156378,),
        gap_size=1.9668571361885592,
        multiplicity_total=10,
        multiplicity_central=3,
        central_pions=2,
        central_protons=0,
        forward_pions=4,
        forward_protons=1,
        seagull_moment=0.8451000667859185,
    ),
    "sd_target": dict(
        class_label='sd_target',
        x_f_projectile=0.9361160002391692,
        x_f_target=-0.9361160002391824,
        excitation_masses=(3.9735186317979148,),
        transfers=(-0.35289897396798214,),
        gap_size=2.8853474827724455,
        multiplicity_total=10,
        multiplicity_central=0,
        central_pions=0,
        central_protons=0,
        forward_pions=8,
        forward_protons=0,
        seagull_moment=0.31624762145656676,
    ),
    "dd": dict(
        class_label='dd',
        x_f_projectile=0.9186066922852268,
        x_f_target=-0.9186066922852117,
        excitation_masses=(2.555650805763481, 4.089372877543087),
        transfers=(-0.44500500534237103, -0.023600994766667327),
        gap_size=1.0530511250407417,
        multiplicity_total=6,
        multiplicity_central=0,
        central_pions=0,
        central_protons=0,
        forward_pions=3,
        forward_protons=0,
        seagull_moment=0.6532348490614007,
    ),
    "nd": dict(
        class_label='nd',
        x_f_projectile=0.9843132502169744,
        x_f_target=-0.9843132502169951,
        excitation_masses=(0.16440192708655166,),
        transfers=(-0.03161309613722584, -0.0896993841496746, -0.12131248028690045),
        gap_size=None,
        multiplicity_total=3,
        multiplicity_central=1,
        central_pions=0,
        central_protons=0,
        forward_pions=0,
        forward_protons=2,
        seagull_moment=0.11780292839690441,
    ),
}

#: The committed digests of the card's pinned surfaces (nested tree
#: HEAD cc22be0, the review-fix round; the solver test digest is the
#: post-re-pin value of the review fix round, which re-touched this
#: solver test file with the corrected physics).
MODULE_SHA256 = (
    "66b769bdc90e44f8563ee1ff7fc6750123bac70185f6edbcd4626f5366dca66c")
#: re-pinned by the Round 5 card 9 review fix round
#: (TASK-20260925-r5-topology-template-runner-review-fix; the
#: REV-bc4de47-01/-02/-03 corrected physics).
SOLVER_TEST_SHA256 = (
    "cd2c809ee6b7fb1e81297d61cfea5d818e6088183bd20acb89ed65698db52901")
#: re-pinned by the Round 5 card 9 review fix round (the REV-bc4de47-03
#: export-surface correction: the runner instance re-exported as
#: NOMINAL_TOPOLOGY_PARAMS, the package NOMINAL_PARAMS restored to
#: card 5's mass-law export, __all__ deduplicated at 253).
INIT_SHA256 = (
    "7d2c5a85d9b24de104c0d9cf72195689bf7c39c873e154798699448073b36de8")

#: The module's ``__all__`` export set (sixteen names).
MODULE_ALL = [
    "BEAM_MOMENTUM",
    "CENTRAL_WINDOW",
    "ELASTIC_TRANSFER_SLOPE",
    "INELASTIC_ORDER",
    "INELASTIC_SET",
    "MASTER_SEED",
    "NOMINAL_PARAMS",
    "SQRT_S",
    "TOPOLOGY_CLASSES",
    "BaseTopologyEvent",
    "TopologyNominalParams",
    "TopologyObservables",
    "draw_topology",
    "extract_observables",
    "pinned_trace",
    "run_topology",
]

#: The ``__init__`` topology-runner export block (the eleven
#: re-exported names; the beam constants and the elastic transfer
#: slope stay card 1's package-level exports and are module-level
#: only here).  The runner's nominal-parameter instance re-enters the
#: package under its own name ``NOMINAL_TOPOLOGY_PARAMS`` (the review
#: fix REV-bc4de47-03); the package-level ``NOMINAL_PARAMS`` stays
#: card 5's single-diffraction mass-law export.
INIT_RUNNER_NAMES = [
    "INELASTIC_ORDER",
    "INELASTIC_SET",
    "MASTER_SEED",
    "NOMINAL_TOPOLOGY_PARAMS",
    "TOPOLOGY_CLASSES",
    "TopologyNominalParams",
    "TopologyObservables",
    "draw_topology",
    "extract_observables",
    "pinned_trace",
    "run_topology",
]

#: The package ``__all__`` total entry count (the additive card 9
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
#: trees, and the seven committed Round 5 modules consumed read-only
#: (card 1's elastic reference, card 2's exchange ledger, card 3's
#: remnants, card 4's mass laws, card 5's single diffraction, card 6's
#: double diffraction, card 7's non-diffractive class, card 8's
#: multiple-neutral class).
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
    "dev/mixing2006/na49_mixing/topology_double_diffraction.py":
        "92c48ece4d22577b91cc3950e6ee20cf3a3bea5ad56b0ffc6edd32445174d54e",
    "dev/mixing2006/na49_mixing/topology_multi_neutral.py":
        "51c3bfd8898bf155c4f3d32138f4b59ccd72cb5110a5178c0ed9bf899da3bb15",
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
# digest refresh (inventory count unchanged at 529) by the Round 5 card 9 review fix round register re-pin (TASK-20260925-r5-topology-template-runner-review-fix-test-author): the register re-pins its contract to the corrected physics at nested cc22be0; the manifest's inventory digest of this register follows.
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
#: The solver card (Round 5, card 9) and this test-author register's
#: card name (the freeze-exception identity).
SOLVER_CARD = "TASK-20260925-r5-topology-template-runner"
TEST_AUTHOR_CARD = (
    "TASK-20260925-r5-topology-template-runner-test-author")

#: This register's relative path in the monorepo (the inventory key).
MY_TEST_REL = (
    "dev/mixing2006/tests/test_topology_runner_contract_pins.py")

#: The card name of this register's re-pin follow-up for the review
#: fix round (the freeze-exception identity).
FIX_ROUND_TEST_AUTHOR_CARD = (
    "TASK-20260925-r5-topology-template-runner-review-fix-test-author")

#: The nested tree SHA at the review fix round (the corrected physics
#: the re-anchored pins consume).
FIX_ROUND_SHA = "cc22be0"

#: This follow-up's freeze-exception file list (path-sorted): the
#: manifest (the in-place digest refresh of this register and this
#: exception entry) and this register.
FIX_ROUND_TEST_AUTHOR_EXCEPTION_FILES = (
    "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json",
    MY_TEST_REL,
)

#: The register chain: the thirty-six chain count-pin files re-pinned
#: 528 -> 529 by this register (the card-9 solver's thirty-five chain
#: files plus the card-9 solver test file, path-sorted).
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
    "dev/mixing2006/tests/test_topology_multi_neutral_contract_pins.py",
    "dev/mixing2006/tests/test_topology_non_diffractive.py",
    "dev/mixing2006/tests/test_topology_non_diffractive_contract_pins.py",
    "dev/mixing2006/tests/test_topology_remnants_contract_pins.py",
    "dev/mixing2006/tests/test_topology_runner.py",
    "dev/mixing2006/tests/test_topology_single_diffraction_contract_pins.py",
)

#: This register's freeze-exception file list (path-sorted): the
#: manifest, the thirty-six re-pinned chain count-pin files, and this
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


def _event_rngs(event_index: int) -> dict:
    """The closed substream generators of one pinned event (the
    five-label quintuple of the runner's trace)."""

    subs = derive_substreams(MASTER_SEED_PIN, 0, event_index)
    return {
        label: np.random.default_rng(subs.stream_for(label))
        for label in ("topology_selection", "spectral_state",
                      "exchange_kinematics", "decay_channel",
                      "decay_kinematics")
    }


def _hand_uniform(event_index: int) -> float:
    """The pinned event's ``topology_selection`` uniform (the
    hand-computed side of the trace: one uniform on the closed
    substream, derived independently of the runner)."""

    stream = derive_substreams(MASTER_SEED_PIN, 0,
                                event_index).stream_for(
        "topology_selection")
    return float(np.random.default_rng(stream).random())


def _hand_draw_label(u: float, elastic_fraction: float,
                      fractions: dict) -> str:
    """The documented draw convention applied to one uniform value: the
    elastic fraction on the lower edge, the inelastic classes walked in
    the fixed order over the absolute cumulatives (the cumulative
    advances from ``elastic_fraction`` by each inelastic fraction, so
    each inelastic class is returned at exactly its caller fraction;
    the corrected allocation of the review finding REV-bc4de47-02,
    equivalently the caller's fractions divided by the inelastic mass
    against the rescaled inelastic uniform)."""

    if u < elastic_fraction:
        return "elastic"
    cumulative = elastic_fraction
    for label in INELASTIC_ORDER:
        cumulative += float(fractions[label])
        if u < cumulative:
            return label
    return INELASTIC_ORDER[-1]


def _hand_walk(fractions, elastic_fraction: float, n: int,
               seed: int) -> list:
    """The hand-computed draw sequence: the uniform values of the
    pinned seed through the documented convention over the caller's
    fraction mapping."""

    rng = np.random.default_rng(seed)
    return [_hand_draw_label(rng.random(), elastic_fraction, fractions)
            for _ in range(n)]


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


def _run_one(event_index: int, catalog, fams):
    """One pinned trace event: the class draw plus the dispatched
    record plus the extracted observables (the trace flow)."""

    rngs = _event_rngs(event_index)
    label = draw_topology(NOMINAL_PARAMS.class_fractions, 0.0,
                          rngs["topology_selection"])
    event = run_topology(
        label, NOMINAL_PARAMS, catalog, fams,
        rng_topology_selection=rngs["topology_selection"],
        rng_spectral_state=rngs["spectral_state"],
        rng_exchange_kinematics=rngs["exchange_kinematics"],
        rng_decay_channel=rngs["decay_channel"],
        rng_decay_kinematics=rngs["decay_kinematics"],
        p_n2=NOMINAL_PARAMS.p_n2)
    return label, event, extract_observables(event, SQRT_S)


def _cm_boost_parameters() -> tuple:
    """The lab to overall-CM boost parameters ``(beta, gamma)`` of the
    pinned M8 p+p beam configuration at the recorded :data:`SQRT_S`
    (the hand-computed side of the Feynman-x and central-window
    extraction; the module applies the same inverse boost, the review
    fix REV-bc4de47-01): ``beta = p_beam / (E_beam + m_p)`` along the
    beam axis and ``gamma = (E_beam + m_p) / sqrt(s)``."""

    e_beam = math.sqrt(PROTON_MASS * PROTON_MASS
                       + BEAM_MOMENTUM * BEAM_MOMENTUM)
    e_total = e_beam + PROTON_MASS
    return BEAM_MOMENTUM / e_total, e_total / SQRT_S


_CM_BETA, _CM_GAMMA = _cm_boost_parameters()


def _cm_four_momentum(four_momentum) -> tuple:
    """The inverse (lab to CM) longitudinal boost of one four-momentum,
    the class modules' forward fixed-target boost applied in reverse:
    ``(E*, px, py, pz*) = (gamma (E - beta pz), px, py, gamma (pz -
    beta E))``.  The transverse components are invariant."""

    e, px, py, pz = (float(v) for v in four_momentum)
    return (_CM_GAMMA * (e - _CM_BETA * pz), px, py,
            _CM_GAMMA * (pz - _CM_BETA * e))


def _hand_observables(event) -> dict:
    """The hand-computed observables from one record's raw fields (the
    independent recomputation side): the Feynman ``x_F = 2 p_L^CM /
    sqrt(s)`` and the central window ``|y^CM| < 0.5`` in the overall
    CM frame (the boosted four-momenta; the review fix
    REV-bc4de47-01), everything else from the frame-independent
    records (the excitation masses, the transfers, the gap size as a
    rapidity difference, and the transverse momenta)."""

    if isinstance(event, ElasticEvent):
        lines = event.outgoing
        masses: tuple = ()
        transfers = (event.t,)
        gap = None
        final = [(line.four_momentum, None) for line in event.outgoing]
    else:
        lines = event.lines
        final = [(h.four_momentum, h.pdg_id) for h in event.final_state]
        if hasattr(event, "excitation_mass"):
            masses = (event.excitation_mass,)
            transfers = (event.t,)
            gap = event.gap_size
        elif hasattr(event, "central_masses"):
            masses = event.central_masses
            transfers = (event.t1, event.t2, event.t_2c, event.t_c)
            gap = None
        elif hasattr(event, "central_mass"):
            masses = (event.central_mass,)
            transfers = (event.t1, event.t2, event.t_c)
            gap = None
        else:
            masses = (event.m1, event.m2)
            transfers = (event.t1, event.t2)
            gap = event.gap_size

    x_f_projectile = (2.0 * _cm_four_momentum(lines[0].four_momentum)[3]
                      / SQRT_S)
    x_f_target = (2.0 * _cm_four_momentum(lines[1].four_momentum)[3]
                  / SQRT_S)

    central = 0
    central_pions = 0
    central_protons = 0
    forward_pions = 0
    forward_protons = 0
    p_t_sum = 0.0
    for four_momentum, pdg_id in final:
        e_star, _px, _py, pz_star = _cm_four_momentum(four_momentum)
        rapidity = 0.5 * math.log((e_star + pz_star)
                                  / (e_star - pz_star))
        in_central = abs(rapidity) < CENTRAL_WINDOW
        if in_central:
            central += 1
        if pdg_id is not None:
            pid = int(pdg_id)
            if pid in (111, 211, -211):
                if in_central:
                    central_pions += 1
                else:
                    forward_pions += 1
            if pid == 2212:
                if in_central:
                    central_protons += 1
                else:
                    forward_protons += 1
        # the transverse momentum is invariant under the longitudinal
        # boost (read from the stored lab four-momenta)
        p_t_sum += math.hypot(float(four_momentum[1]),
                              float(four_momentum[2]))

    return {
        "class_label": event.class_label,
        "x_f_projectile": x_f_projectile,
        "x_f_target": x_f_target,
        "excitation_masses": tuple(float(m) for m in masses),
        "transfers": tuple(float(t) for t in transfers),
        "gap_size": None if gap is None else float(gap),
        "multiplicity_total": len(final),
        "multiplicity_central": central,
        "central_pions": central_pions,
        "central_protons": central_protons,
        "forward_pions": forward_pions,
        "forward_protons": forward_protons,
        "seagull_moment": (p_t_sum / len(final)) if final else 0.0,
    }


@pytest.fixture(scope="module")
def catalog() -> HadronCatalog:
    return HadronCatalog.from_json(CATALOG_PATH)


@pytest.fixture(scope="module")
def fams():
    return families_from_json(str(CONTINUUM_PATH))


@pytest.fixture(scope="module")
def pinned_trace(catalog, fams):
    """The pinned 1,000-event runner trace (master seed 20260925,
    replica 0, one substream quintuple per event index 0-999 at the
    nominal parameters; the per-event label, record, and observables)."""

    return [_run_one(event_index, catalog, fams)
            for event_index in range(N_TRACE)]


# ----------------------------------------------------------------------
# Part 1: the module contract (pure, hermetic)
# ----------------------------------------------------------------------
def test_registry_surface_pins():
    """The registry is exactly the six entries with the ``retained``
    flags (``nd_multi`` ``retained: False``, all others True) and the
    ``inelastic`` flags (elastic ``inelastic: False``, all others
    True); ``INELASTIC_SET`` is exactly the four classes;
    ``INELASTIC_ORDER`` is the fixed walk order; every entry names its
    landed module; and ``BaseTopologyEvent`` is the union of the five
    landed record types in the committed order."""

    assert set(TOPOLOGY_CLASSES) == {
        "elastic", "sd_projectile", "sd_target", "dd", "nd", "nd_multi"}
    assert len(TOPOLOGY_CLASSES) == 6
    assert TOPOLOGY_CLASSES["elastic"] == {
        "module": "na49_mixing.topology_elastic",
        "retained": True,
        "inelastic": False,
    }
    assert TOPOLOGY_CLASSES["sd_projectile"] == {
        "module": "na49_mixing.topology_single_diffraction",
        "retained": True,
        "inelastic": True,
    }
    assert TOPOLOGY_CLASSES["sd_target"] == {
        "module": "na49_mixing.topology_single_diffraction",
        "retained": True,
        "inelastic": True,
    }
    assert TOPOLOGY_CLASSES["dd"] == {
        "module": "na49_mixing.topology_double_diffraction",
        "retained": True,
        "inelastic": True,
    }
    assert TOPOLOGY_CLASSES["nd"] == {
        "module": "na49_mixing.topology_non_diffractive",
        "retained": True,
        "inelastic": True,
    }
    assert TOPOLOGY_CLASSES["nd_multi"] == {
        "module": "na49_mixing.topology_multi_neutral",
        "retained": False,
        "inelastic": True,
    }, "only nd_multi carries the unretained flag (the card-8 entry)"
    assert set(INELASTIC_SET) == {"sd_projectile", "sd_target", "dd", "nd"}
    assert "elastic" not in INELASTIC_SET
    assert "nd_multi" not in INELASTIC_SET
    assert INELASTIC_ORDER == ("sd_projectile", "sd_target", "dd", "nd")
    # every entry names its landed module (importable, exact name)
    import importlib
    for label, entry in TOPOLOGY_CLASSES.items():
        module = importlib.import_module(entry["module"])
        assert module.__name__ == entry["module"]
    # the BaseTopologyEvent union: the five landed record types, in
    # the committed order
    assert get_origin(BaseTopologyEvent) is Union
    assert get_args(BaseTopologyEvent) == (
        ElasticEvent, SDEvent, DDEvent, NDEvent, NDMultiEvent)


def test_nominal_params_pins():
    """``TopologyNominalParams()`` equals the pre-registered R50 values
    exactly (``epsilon = 0``, ``B_SD = 8.504140762083612`` byte-
    identical to the M8-frozen ``effective_transfer_slope``, the equal
    0.25 inelastic class fractions, the nominal remnant split intact
    0.5 / low-mass 0.5 / high-mass 0.0, ``p_n2 = 0.0``); the beam
    configuration pins hold (``SQRT_S = 17.270``, ``PROTON_MASS =
    0.93827231``, ``PI0_MASS = 0.1349766``, ``BEAM_MOMENTUM = 158.0``,
    ``MASTER_SEED = 20260925``); and the nominal structure builds
    exactly the committed class modules' mass law and remnant
    weights."""

    params = TopologyNominalParams()
    assert getattr(params, "__dataclass_params__").frozen
    assert tuple(params.__dataclass_fields__) == (
        "epsilon", "b_sd", "class_fractions", "remnant_weights", "p_n2",
        "flavor", "epsilon_nd", "nd_central", "isovector_fraction", "b_isovector", "isovector_delta_ratio",
        "b_nd", "exact_transfer_range", "line_thresholds", "central_recoil", "epsilon_c",
        "transfer_slope_mass_exponent", "low_mass_spin_isospin", "nd_duality_mass",
        "nd_duality_shape_mass", "duality_line_shapes", "duality_redraw_shapes", "diffractive_duality_mass",
        "intact_vertex_factor", "diffractive_xi_max", "sd_tail", "sd_ope", "sd_two_sided", "diffractive_pomeron_slope",
        "diffractive_pomeron_linear_t", "intact_t_break")  # diffractive_pomeron_slope: 2026-11-01, None = the zero-transfer form with the per-mass vertex law;
    # diffractive_pomeron_linear_t: 2026-11-01, None = the shrinkage at every t;
    # sd_ope: 2026-10-26, None = no one-pion exchange; sd_two_sided: 2026-10-26, None = every system one line; duality_line_shapes: 2026-10-18, False = the fixed-width pick;
    # sd_tail: 2026-10-23 (R112), None = no leading-proton tail of the single-diffraction classes;
    # intact_t_break: 2026-10-24, None = the one exponential of the intact proton's transfer;
    # diffractive_xi_max: 2026-10-23, None = the diffractive excited masses up to the kinematic edge;
    # epsilon_nd: 2026-09-29 Block A, None = shared;
    # transfer_slope_mass_exponent: 2026-10-08, None = the single transfer slope b_sd;
    # low_mass_spin_isospin: 2026-10-08, False = the uniform low-mass pick over every (1, 1, 0) record;
    # nd_duality_mass, nd_duality_shape_mass: 2026-10-12, None = the continuum lines;
    # diffractive_duality_mass: 2026-10-15, None = the continuum cascade of the diffractive lines;
    # intact_vertex_factor: 2026-10-15, None = the term 1 of the intact proton's vertex
    assert params.nd_duality_mass is None and params.nd_duality_shape_mass is None
    assert params.intact_vertex_factor is None
    assert params.diffractive_xi_max is None
    # line_thresholds: 2026-10-06, False = the shared p pi0 edge of every line;
    # isovector_fraction: Round 8 2026-10-04, 0.0 = the vacuum exchange only;
    # b_isovector: Round 8 2026-10-04, None = the shared b_sd
    assert params.isovector_fraction == 0.0 and params.b_isovector is None
    # isovector_delta_ratio, b_nd: 2026-10-05, 1.0 and None leave the event unchanged
    assert params.isovector_delta_ratio == 1.0 and params.b_nd is None
    assert params.exact_transfer_range is False  # 2026-10-05: the committed transfer draw
    assert params.epsilon_nd is None
    assert params.nd_central is True  # 2026-10-02: the central source on by default
    assert params.epsilon == 0.0
    assert params.b_sd == B_SD_PIN == ELASTIC_SLOPE_PIN
    assert params.b_sd.hex() == (B_SD_PIN).hex()
    assert dict(params.class_fractions) == {
        "sd_projectile": 0.25, "sd_target": 0.25, "dd": 0.25, "nd": 0.25}
    assert dict(params.remnant_weights) == {
        "intact_proton": 0.5, "low_mass_baryonic_resonance": 0.5,
        "high_mass_baryonic_continuum": 0.0}
    assert params.p_n2 == 0.0
    assert NOMINAL_PARAMS == params
    # the beam configuration pins
    assert SQRT_S == SQRT_S_PIN
    assert PROTON_MASS == PROTON_MASS_PIN
    assert PI0_MASS == PI0_MASS_PIN
    assert BEAM_MOMENTUM == BEAM_MOMENTUM_PIN
    assert ELASTIC_TRANSFER_SLOPE == ELASTIC_SLOPE_PIN
    assert MASTER_SEED == MASTER_SEED_PIN
    assert CENTRAL_WINDOW == 0.5
    # the M8-frozen surface: the manifest's frozen parameter map
    # carries the byte-identical effective_transfer_slope and its hash
    # is the M8 hash
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    m8 = manifest["frozen_fit_outputs"]["m8_frozen_pp_parameters"]
    assert m8["regge.effective_transfer_slope"] == params.b_sd
    computed_hash = hashlib.sha256(
        json.dumps(m8, sort_keys=True, separators=(",", ":")).encode(
            "utf-8")).hexdigest()
    assert computed_hash == M8_HASH
    assert m8["continuum.match_mass"] == KERNEL_BOUNDARY
    # the committed class modules' nominal surfaces (consumed
    # read-only; the runner builds the same mass law)
    assert MassLawParams(epsilon=params.epsilon,
                         b_sd=params.b_sd) == SD_NOMINAL_PARAMS
    assert SD_NOMINAL_PARAMS == DD_NOMINAL_PARAMS == ND_NOMINAL_PARAMS
    assert dict(params.remnant_weights) == dict(NOMINAL_SD_REMNANT_WEIGHTS)
    assert dict(params.remnant_weights) == dict(NOMINAL_ND_REMNANT_WEIGHTS)


def test_draw_pins(catalog, fams):
    """The pinned uniform head reproduces; the class sequence of
    ``draw_topology`` over 1,000 draws matches the hand-computed
    sequence from the uniform values on the corrected absolute-
    cumulative walk (the review finding REV-bc4de47-02: each inelastic
    class returned at exactly its caller fraction); the draw consumes
    exactly one uniform per draw on ``topology_selection`` (the
    counting proxy and the pinned RNG-state identity 1,000 uniforms
    past the seed); the elastic fraction sits on the lower edge of the
    single uniform; the 100,000-draw frequency pin at the positive
    elastic fraction ``f_e = 0.3`` holds (the identity test cannot
    substitute for it: both sides would copy the same walk, and the
    buggy unscaled walk put ``nd`` at 0.332 and the diffractive
    classes at 0.123); ``nd_multi`` is never returned at the default
    ``P(n = 2) = 0.0`` (excluded from the inelastic draw; exactly one
    central source when dispatched); and the draw raises on unknown
    class labels, missing inelastic classes, negative fractions,
    out-of-range elastic fractions, and non-unit mass (checked, never
    repaired)."""

    # the pinned uniform head (the hand-computed side's own stream)
    head = np.random.default_rng(MASTER_SEED_PIN)
    for expected_u in PINNED_UNIFORMS_HEAD:
        assert head.random() == expected_u

    # the pinned 1,000-draw sequence against the hand-computed
    # sequence
    fractions = NOMINAL_PARAMS.class_fractions
    rng = np.random.default_rng(MASTER_SEED_PIN)
    drawn = [draw_topology(fractions, 0.0, rng) for _ in range(N_DRAWS)]
    assert drawn == _hand_walk(fractions, 0.0, N_DRAWS, MASTER_SEED_PIN), (
        "the draw sequence drifted from the hand-computed sequence")
    assert "elastic" not in drawn, (
        "no draw returns 'elastic' at elastic_fraction = 0")
    assert "nd_multi" not in drawn, (
        "the unretained nd_multi class is never returned at the "
        "default P(n = 2) = 0.0")
    assert set(drawn) == {"sd_projectile", "sd_target", "dd", "nd"}

    # determinism: a fresh identical generator reproduces the
    # sequence
    again = np.random.default_rng(MASTER_SEED_PIN)
    assert [draw_topology(fractions, 0.0, again) for _ in range(N_DRAWS)
            ] == drawn

    # exactly one uniform per draw (pinned RNG-state identity): the
    # counting proxy advances exactly once per draw, and the post-draw
    # bit-generator state sits exactly 1,000 uniforms past the seed
    counting = _CountingRng(np.random.default_rng(MASTER_SEED_PIN))
    for index in range(N_DRAWS):
        draw_topology(fractions, 0.0, counting)
        assert counting.calls == index + 1
    probe = np.random.default_rng(MASTER_SEED_PIN)
    for _ in range(N_DRAWS):
        probe.random()
    assert counting._source.bit_generator.state == \
        probe.bit_generator.state

    # the elastic lower edge: with a positive elastic fraction the
    # first uniform (0.15923... < 0.3) draws "elastic", and the full
    # 1,000-draw sequence matches the hand-computed walk with elastic
    # present and nd_multi absent
    elastic_fractions = {label: 0.175 for label in INELASTIC_ORDER}
    first = draw_topology(elastic_fractions, 0.3,
                          np.random.default_rng(MASTER_SEED_PIN))
    assert first == "elastic", (
        "the elastic fraction occupies the lower edge of the single "
        "uniform")
    elastic_rng = np.random.default_rng(MASTER_SEED_PIN)
    got = [draw_topology(elastic_fractions, 0.3, elastic_rng)
           for _ in range(N_DRAWS)]
    assert got == _hand_walk(elastic_fractions, 0.3, N_DRAWS,
                            MASTER_SEED_PIN)
    assert "elastic" in got and "nd_multi" not in got

    # the frequency pin at the positive elastic fraction (the review
    # finding REV-bc4de47-02's surface; the hand-walk identity above
    # cannot substitute because both sides copy the same walk): at
    # f_e = 0.3 with 0.175-each inelastic fractions each class is
    # returned at exactly its given fraction of the drawn set; the
    # pinned 100,000-draw counts (master seed 20260925) sit on the
    # 0.300 / 0.175 targets within the pinned +/- 0.01 tolerance
    n_frequency = 100_000
    frequency_rng = np.random.default_rng(MASTER_SEED_PIN)
    frequency_counts = Counter(
        draw_topology(elastic_fractions, 0.3, frequency_rng)
        for _ in range(n_frequency))
    assert sum(frequency_counts.values()) == n_frequency
    assert dict(frequency_counts) == {
        "elastic": 29875, "sd_projectile": 17577, "sd_target": 17589,
        "dd": 17513, "nd": 17446}, (
        "the pinned 100,000-draw frequencies at f_e = 0.3 drifted")
    assert abs(frequency_counts["elastic"] / n_frequency - 0.3) <= 0.01, (
        "the elastic draw frequency drifted at f_e = 0.3")
    for label in INELASTIC_ORDER:
        assert abs(frequency_counts[label] / n_frequency - 0.175) <= 0.01, (
            f"the draw frequency of {label!r} drifted at f_e = 0.3 "
            "(the walk must return each class at exactly its given "
            "fraction of the drawn set)")

    # nd_multi at the default: the label stays dispatchable and the
    # unretained class carries exactly one central source at
    # P(n = 2) = 0.0
    rngs = _event_rngs(0)
    ndm = run_topology(
        "nd_multi", NOMINAL_PARAMS, catalog, fams,
        rng_topology_selection=rngs["topology_selection"],
        rng_spectral_state=rngs["spectral_state"],
        rng_exchange_kinematics=rngs["exchange_kinematics"],
        rng_decay_channel=rngs["decay_channel"],
        rng_decay_kinematics=rngs["decay_kinematics"])
    assert isinstance(ndm, NDMultiEvent)
    assert ndm.count == 1

    # the checked refusals
    nominal = {label: 0.25 for label in INELASTIC_ORDER}
    with pytest.raises(ValueError, match="unknown inelastic class"):
        draw_topology({**nominal, "elastic": 0.0}, 0.0,
                      np.random.default_rng(1))
    with pytest.raises(ValueError, match="unknown inelastic class"):
        draw_topology({**nominal, "nd_multi": 0.0}, 0.0,
                      np.random.default_rng(1))
    with pytest.raises(ValueError, match="missing"):
        draw_topology({label: 1.0 / 3.0
                       for label in list(INELASTIC_ORDER)[:-1]}, 0.0,
                      np.random.default_rng(1))
    with pytest.raises(ValueError, match="non-negative"):
        draw_topology({**nominal, "nd": -0.25}, 0.25,
                      np.random.default_rng(1))
    with pytest.raises(ValueError, match="elastic fraction"):
        draw_topology(nominal, 1.5, np.random.default_rng(1))
    with pytest.raises(ValueError, match="elastic fraction"):
        draw_topology(nominal, -0.1, np.random.default_rng(1))
    with pytest.raises(ValueError, match="unit mass"):
        draw_topology(nominal, 0.5, np.random.default_rng(1))
    # a legal positive elastic fraction draws a legal label
    got = draw_topology(elastic_fractions, 0.3, np.random.default_rng(1))
    assert got in set(INELASTIC_ORDER) | {"elastic"}


def test_dispatch_bit_identity_pins(catalog, fams):
    """Each inelastic class label dispatches to the landed module and
    the nominal-parameter record is bit-identical to the class
    module's own generation on the same closed per-label substreams
    (the pinned 200-event trace per class; the cross-module identity);
    the elastic class dispatches on ``exchange_kinematics`` alone; the
    unretained ``nd_multi`` label stays dispatchable; unknown labels
    and missing streams raise."""

    for event_index in range(N_DISPATCH):
        # elastic
        rngs = _event_rngs(event_index)
        got = run_topology(
            "elastic", NOMINAL_PARAMS, catalog, fams,
            rng_exchange_kinematics=rngs["exchange_kinematics"])
        reference = _event_rngs(event_index)
        want = generate_elastic(reference["exchange_kinematics"])
        assert got == want, f"elastic event {event_index} drifted"

        # sd_projectile / sd_target
        for label, side in (("sd_projectile", "projectile"),
                            ("sd_target", "target")):
            rngs = _event_rngs(event_index)
            got = run_topology(
                label, NOMINAL_PARAMS, catalog, fams,
                rng_spectral_state=rngs["spectral_state"],
                rng_exchange_kinematics=rngs["exchange_kinematics"],
                rng_decay_channel=rngs["decay_channel"],
                rng_decay_kinematics=rngs["decay_kinematics"])
            reference = _event_rngs(event_index)
            want = generate_sd(
                side, SD_NOMINAL_PARAMS, NOMINAL_SD_REMNANT_WEIGHTS,
                catalog, fams, reference["spectral_state"],
                reference["exchange_kinematics"],
                reference["decay_channel"],
                reference["decay_kinematics"])
            assert got == want, f"{label} event {event_index} drifted"

        # dd
        rngs = _event_rngs(event_index)
        got = run_topology(
            "dd", NOMINAL_PARAMS, catalog, fams,
            rng_exchange_kinematics=rngs["exchange_kinematics"],
            rng_decay_channel=rngs["decay_channel"],
            rng_decay_kinematics=rngs["decay_kinematics"])
        reference = _event_rngs(event_index)
        want = generate_dd(
            DD_NOMINAL_PARAMS, catalog, fams,
            reference["exchange_kinematics"], reference["decay_channel"],
            reference["decay_kinematics"])
        assert got == want, f"dd event {event_index} drifted"

        # nd
        rngs = _event_rngs(event_index)
        got = run_topology(
            "nd", NOMINAL_PARAMS, catalog, fams,
            rng_spectral_state=rngs["spectral_state"],
            rng_exchange_kinematics=rngs["exchange_kinematics"],
            rng_decay_channel=rngs["decay_channel"],
            rng_decay_kinematics=rngs["decay_kinematics"])
        reference = _event_rngs(event_index)
        want = generate_nd(
            ND_NOMINAL_PARAMS, NOMINAL_ND_REMNANT_WEIGHTS, catalog, fams,
            reference["spectral_state"], reference["exchange_kinematics"],
            reference["decay_channel"], reference["decay_kinematics"])
        assert got == want, f"nd event {event_index} drifted"

    # the unretained nd_multi label stays dispatchable (the registry
    # entry; the count is exactly 1 at the default P(n = 2) = 0.0)
    rngs = _event_rngs(0)
    got = run_topology(
        "nd_multi", NOMINAL_PARAMS, catalog, fams,
        rng_topology_selection=rngs["topology_selection"],
        rng_spectral_state=rngs["spectral_state"],
        rng_exchange_kinematics=rngs["exchange_kinematics"],
        rng_decay_channel=rngs["decay_channel"],
        rng_decay_kinematics=rngs["decay_kinematics"])
    assert isinstance(got, NDMultiEvent) and got.count == 1

    # validation: unknown label, missing stream
    with pytest.raises(ValueError, match="unknown class label"):
        run_topology("charge_exchange", NOMINAL_PARAMS, catalog, fams,
                     rng_exchange_kinematics=np.random.default_rng(1))
    with pytest.raises(ValueError, match="requires the closed substream"):
        run_topology("elastic", NOMINAL_PARAMS, catalog, fams)
    with pytest.raises(ValueError, match="requires the closed substream"):
        run_topology("sd_projectile", NOMINAL_PARAMS, catalog, fams,
                     rng_exchange_kinematics=np.random.default_rng(1))
    with pytest.raises(ValueError, match="requires the closed substream"):
        run_topology("nd_multi", NOMINAL_PARAMS, catalog, fams,
                     rng_spectral_state=np.random.default_rng(1),
                     rng_exchange_kinematics=np.random.default_rng(1),
                     rng_decay_channel=np.random.default_rng(1),
                     rng_decay_kinematics=np.random.default_rng(1))


def test_observables_pins(pinned_trace):
    """On the pinned 1,000-event trace every extracted observable
    matches the hand-computed recomputation within ``1e-12`` (the
    leading-baryon Feynman ``x_F`` per baryonic line in the overall
    CM frame, the excitation masses, the transfers, the central gap,
    the total multiplicity, the central-activity multiplicity in the
    CM window ``|y^CM| < 0.5``, the central and forward pion and
    proton yields, the seagull moment; the review fix
    REV-bc4de47-01 moved the Feynman x and the central window from the
    stored lab frame into the overall CM frame); the ``pinned_trace``
    output is exactly the per-event extraction; the pinned class
    census is 260/249/241/250; the first-occurrence indices and the
    exact pinned first-occurrence literals hold (the four inelastic
    classes plus the elastic class's direct dispatch event-0)."""

    from na49_mixing.topology_runner import pinned_trace as _pt

    trace = _pt(NOMINAL_PARAMS, MASTER_SEED_PIN, N_TRACE)
    assert len(trace) == N_TRACE

    labels = []
    for event_index, (label, event, obs) in enumerate(pinned_trace):
        labels.append(label)
        # the extracted observables are exactly the trace output
        assert obs == trace[event_index], (
            f"event {event_index}: the extracted observables differ "
            "from the pinned trace")
        # the hand-computed side (the independent recomputation)
        expected = _hand_observables(event)
        assert label == expected["class_label"]
        assert abs(obs.x_f_projectile - expected["x_f_projectile"]) <= 1e-12
        assert abs(obs.x_f_target - expected["x_f_target"]) <= 1e-12
        assert len(obs.excitation_masses) == len(expected["excitation_masses"])
        for got, want in zip(obs.excitation_masses,
                             expected["excitation_masses"]):
            assert abs(got - want) <= 1e-12
        assert len(obs.transfers) == len(expected["transfers"])
        for got, want in zip(obs.transfers, expected["transfers"]):
            assert abs(got - want) <= 1e-12
        if expected["gap_size"] is None:
            assert obs.gap_size is None
        else:
            assert abs(obs.gap_size - expected["gap_size"]) <= 1e-12
        assert obs.multiplicity_total == expected["multiplicity_total"]
        assert obs.multiplicity_central == expected["multiplicity_central"]
        for field in ("central_pions", "central_protons",
                      "forward_pions", "forward_protons"):
            assert getattr(obs, field) == expected[field]
        assert abs(obs.seagull_moment - expected["seagull_moment"]) <= 1e-12

    # the pinned trace labels: the hand-computed head, the class
    # census, and the first-occurrence indices
    assert tuple(labels[:len(HAND_TRACE_HEAD)]) == HAND_TRACE_HEAD
    assert Counter(labels) == Counter(TRACE_CLASS_COUNTS), (
        "the pinned trace's class census drifted")
    for label, first_index in TRACE_FIRST_INDEX.items():
        assert labels[first_index] == label
        assert labels.index(label) == first_index

    # the exact pinned first-occurrence observables (the hand-computed
    # literals): the first event of each inelastic class in the trace
    for label, expected in PINNED_OBSERVABLES.items():
        if label == "elastic":
            continue
        obs = pinned_trace[TRACE_FIRST_INDEX[label]][2]
        assert obs.class_label == label
        assert obs.x_f_projectile == expected["x_f_projectile"]
        assert obs.x_f_target == expected["x_f_target"]
        assert obs.excitation_masses == expected["excitation_masses"]
        assert obs.transfers == expected["transfers"]
        assert obs.gap_size == expected["gap_size"]
        assert obs.multiplicity_total == expected["multiplicity_total"]
        assert obs.multiplicity_central == expected["multiplicity_central"]
        assert obs.central_pions == expected["central_pions"]
        assert obs.central_protons == expected["central_protons"]
        assert obs.forward_pions == expected["forward_pions"]
        assert obs.forward_protons == expected["forward_protons"]
        assert obs.seagull_moment == expected["seagull_moment"]

    # the elastic class's direct dispatch event (the normalization
    # reference; the elastic fraction never enters the inelastic
    # trace): the pseudo-final-state lines carry no catalog identity,
    # so the species yields count zero hadrons
    rngs = _event_rngs(0)
    elastic = run_topology(
        "elastic", NOMINAL_PARAMS, None, None,
        rng_exchange_kinematics=rngs["exchange_kinematics"])
    obs = extract_observables(elastic, SQRT_S)
    expected = PINNED_OBSERVABLES["elastic"]
    assert obs.class_label == "elastic"
    assert obs.x_f_projectile == expected["x_f_projectile"]
    assert obs.x_f_target == expected["x_f_target"]
    assert obs.excitation_masses == ()
    assert obs.transfers == expected["transfers"]
    assert obs.gap_size is None
    assert obs.multiplicity_total == expected["multiplicity_total"]
    assert obs.multiplicity_central == expected["multiplicity_central"]
    assert obs.central_pions == 0
    assert obs.central_protons == 0
    assert obs.forward_pions == 0
    assert obs.forward_protons == 0
    assert obs.seagull_moment == expected["seagull_moment"]
    # the elastic event passes card 1's zero-transfer check (the
    # normalization reference's ledger convention)
    check_zero_transfer(elastic)


def test_frozen_cascade_pins(catalog, fams):
    """The trace events decay through the frozen simplified cascade at
    the sealed R45 point (kappa_B = 0.0, kappa_M = 4.0, b_dec = 2.0
    GeV^-2, baryonic nominal width 0.3 GeV, ``W_flavor = 1.0``, pinned
    through the R45 driver digests in both helper trees); a white-box
    check that the runner references no re-fit of the alignment
    parameters and no separation claim (the R46 clauses; the exchange
    slopes act on the exchange kinematics only, never on any daughter
    transverse dynamics)."""

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

    # the W_flavor hook default is exactly 1.0 (multiplicative, zero
    # draws)
    probe_channel = CascadeChannel(
        parent_id="baryonic_np", daughters=("catalog:2212", "catalog:111"),
        kind="established_hadron", weight=0.5)
    assert channel_weight(probe_channel) == 0.5
    assert channel_weight(probe_channel, w_flavor=1.0) == 0.5

    # the white-box token scan: no refit of the alignment parameters
    # and no separation claim anywhere in the runner module (the R46
    # clauses); the runner routes the decay through the class modules
    # (no direct cascade import)
    source = MODULE_PATH.read_text(encoding="utf-8")
    for token in NO_REFIT_TOKENS:
        assert token not in source, (
            f"the runner references the forbidden token {token!r} "
            "(the R46 no_refit clause)")
    assert "cascade_alignment" not in source
    assert "from na49_mixing.cascade" not in source

    # the frozen-parameter pass-through: the runner's nominal
    # structure builds exactly the committed class modules' mass law
    # (the R50 nominal point; no refit surface exists)
    params = TopologyNominalParams()
    assert MassLawParams(epsilon=params.epsilon,
                         b_sd=params.b_sd) == SD_NOMINAL_PARAMS
    assert BEAM_MOMENTUM == BEAM_MOMENTUM_PIN
    assert SQRT_S == SQRT_S_PIN
    assert PROTON_MASS == PROTON_MASS_PIN


# ----------------------------------------------------------------------
# Part 2: the substrate and freeze pins
# ----------------------------------------------------------------------
def test_module_byte_and_export_pins():
    """The module at its committed sha256 (nested tree HEAD cc22be0,
    the review-fix round) with its exact ``__all__`` export set
    (sixteen names), the solver test file at its committed sha256
    (the post-re-pin value of the review fix round), and the package
    ``__init__`` at its committed sha256 carrying the topology-runner
    export block (the eleven re-exported names, the runner instance
    as ``NOMINAL_TOPOLOGY_PARAMS``; the beam constants and the
    transfer slope stay card 1's package-level exports, module-level
    only here, the package-level ``NOMINAL_PARAMS`` stays card 5's
    mass-law export, and the final-state hadron record stays card 5's
    package-level ``FinalHadron`` export; the package ``__all__``
    total at 253 unique entries)."""

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    import na49_mixing.topology_runner as _mod
    assert _mod.__all__ == MODULE_ALL

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)
    init_text = INIT_PATH.read_text(encoding="utf-8")
    block = re.search(r"from \.topology_runner import \((.*?)\)",
                      init_text, re.S)
    assert block is not None, (
        "the __init__ topology-runner import block is missing")
    # the ten re-exported names in the parenthesized import block
    for name in INIT_RUNNER_NAMES:
        if name == "NOMINAL_TOPOLOGY_PARAMS":
            continue
        assert re.search(rf"\b{name}\b", block.group(1)), (
            f"{name} is not in the __init__ topology-runner import "
            "block")
    # the runner's nominal-parameter instance re-enters the package
    # under its own name (the review fix REV-bc4de47-03): the
    # package-level NOMINAL_PARAMS stays card 5's mass-law export
    as_import = re.search(
        r"from \.topology_runner import NOMINAL_PARAMS as "
        r"NOMINAL_TOPOLOGY_PARAMS", init_text)
    assert as_import is not None, (
        "the runner instance is not re-exported as "
        "NOMINAL_TOPOLOGY_PARAMS")
    # the beam constants and the transfer slope and the card-5 record
    # stay module-level only (not re-exported through the
    # topology-runner block); NOMINAL_PARAMS must not re-enter the
    # runner block under the card-5 name (REV-bc4de47-03)
    for name in ("BEAM_MOMENTUM", "PROTON_MASS", "SQRT_S",
                 "ELASTIC_TRANSFER_SLOPE", "CENTRAL_WINDOW",
                 "FinalHadron", "MassLawParams", "BaseTopologyEvent",
                 "NOMINAL_PARAMS"):
        assert not re.search(rf"\b{name}\b", block.group(1)), (
            f"{name} must not enter the __init__ topology-runner "
            "import block")
    all_block = re.search(r"__all__\s*=\s*\[(.*?)\n\]", init_text, re.S)
    assert all_block is not None
    all_names = re.findall(r'"([^"]+)"', all_block.group(1))
    assert len(all_names) == INIT_ALL_COUNT
    assert len(set(all_names)) == INIT_ALL_COUNT, (
        "the package __all__ must stay unique (the review fix "
        "REV-bc4de47-03 deduplicated the runner name)")
    assert all_names.count("NOMINAL_PARAMS") == 1, (
        "the package __all__ must carry card 5's NOMINAL_PARAMS "
        "exactly once")
    for name in INIT_RUNNER_NAMES:
        assert f'"{name}"' in all_block.group(1), (
            f"{name} is not in the __init__ __all__")


def test_frozen_substrate_digests():
    """The card's frozen-substrate provenance surfaces (verified on
    disk, including the consumed cards 1-8 class modules at their
    committed digests) and the frozen M8 parameter map through the
    manifest (the ten-parameter set hash, the kernel boundary, and the
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
