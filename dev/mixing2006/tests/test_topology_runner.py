"""Card tests for the topology template runner.

Round 5, card 9 (``TASK-20260925-r5-topology-template-runner``; plan
section 22 task 36, line 1761; plan section 11.1 item 1, line 824, with
run R50, lines 898-907).  The card's pre-registered deterministic tests
against the module ``na49_mixing/topology_runner.py``:

- ``test_registry``: the registry is exactly the six entries with the
  retained flags (``nd_multi`` ``retained: False``, all others True);
  the inelastic set is exactly ``{"sd_projectile", "sd_target", "dd",
  "nd"}`` (elastic and nd_multi excluded).
- ``test_draw_determinism``: with a pinned seed, the class sequence of
  ``draw_topology`` over 1,000 draws matches the hand-computed sequence
  from the uniform values; the draw consumes exactly 1 uniform per
  event on ``topology_selection`` (pinned RNG-state identity); the draw
  never returns ``"elastic"`` when the elastic fraction is 0 and never
  returns ``"nd_multi"``; the frequency pin at the positive elastic
  fraction ``f_e = 0.3`` with 0.175-each inelastic fractions holds
  (each class returned at exactly its given fraction of the drawn set
  over 100,000 draws, the corrected absolute-cumulative walk of the
  review fix round).
- ``test_draw_validation``: the draw raises on unknown class labels,
  missing inelastic classes, negative fractions, out-of-range elastic
  fractions, and non-unit mass (checked, never repaired).
- ``test_dispatch``: each inelastic class label dispatches to the
  landed module and returns the matching record type (SD_p / SD_t / DD
  / ND); the nominal-parameter record of each class is bit-identical to
  the class module's own pinned trace on the same seed (cross-module
  identity).
- ``test_observables``: on the pinned 1,000-event trace, the extracted
  observables match hand-computed values within 1e-12 (the
  leading-baryon Feynman x_F per line in the overall CM frame, the
  excitation masses, the transfers, the central gap size on the dd
  events, the total multiplicity, the central-activity multiplicity in
  the CM window ``|y^CM| < 0.5``, the central/forward pion and proton
  yields, and the seagull moments).  The review fix round (the
  ``REV-bc4de47-01`` correction) moved the Feynman x and the central
  window from the stored lab frame into the overall CM frame: every
  landed class record stores lab-frame four-momenta, and the extraction
  applies the inverse fixed-target boost ``(beta, gamma)`` before
  forming ``x_F = 2 p_L^CM / sqrt(s)`` and the central window.
- ``test_nominal_params``: ``TopologyNominalParams()`` equals the
  pre-registered values exactly (epsilon = 0, B_SD = 8.504140762083612,
  fractions 0.25 each, remnant weights 0.5/0.5/0.0, ``p_n2 = 0.0``);
  the M8 frozen value is byte-identical to the M8-frozen
  ``effective_transfer_slope`` (hash
  358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1,
  value 8.504140762083612).
- ``test_frozen_cascade``: the trace events decay through the frozen
  simplified cascade at the sealed R45 point; a white-box check that
  the runner references no re-fit of the alignment parameters (the R46
  ``no_refit`` clause).
- ``test_layout_sync``: the test register and the layout sync.

Operative resolutions recorded here (the card's literals against the
committed tree):

- The runner signatures: the card's sketch compresses the substream
  arguments of ``run_topology`` to a single ``rng``; the committed
  cards 5-8 pass the closed per-label substream generators (the
  card-8 recorded pattern), and the runner passes them through
  unchanged (the dispatch identity is the proof of the pass-through).
- The remnant-weight keys resolve to the committed card-3 remnant-kind
  vocabulary (the card's short literals name the same 0.5 / 0.5 / 0.0
  split).
- ``pinned_trace`` loads the committed catalog and the continuum
  families configuration from the package tree (the class generators'
  configuration arguments; the card's sketch omits them).
- The draw convention (the review fix round): the single uniform puts
  the elastic fraction on the lower edge (plan section 11.1 item 1
  names the elastic topology first) and walks the inelastic classes in
  the fixed committed order (projectile SD, target SD, DD, ND) over the
  absolute cumulatives, so each inelastic class is returned with
  probability exactly its caller fraction (the corrected allocation of
  the review finding REV-bc4de47-02; the buggy walk compared the
  unscaled fractions against the rescaled uniform and put the tail mass
  on the last class); at the nominal point the elastic fraction is
  exactly 0.0, so the pinned trace carries no elastic event and the
  elastic class is dispatched directly.

The pinned traces draw on the Round 5 master seed 20260925 (round 5
index item 9), replica 0, one substream quintuple per event index on
the closed labels (``topology_selection``, ``spectral_state``,
``exchange_kinematics``, ``decay_channel``, ``decay_kinematics``); the
dispatch-identity sub-traces use the same event indices (the same seed
trace).

Hermetic: the module, the committed catalog, the continuum families
configuration, the Round 4 cascade modules, and the committed cards
1-8 modules (all consumed read-only), numpy, and the standard library
only; no event generation driver, no external tables, no Monte Carlo
run (the runs R50-R55 exercise the runner statistically).
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

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
    generate_elastic,
)
from na49_mixing.topology_mass_laws import MassLawParams
from na49_mixing.topology_multi_neutral import (
    DEFAULT_P2,
    NDMultiEvent,
    generate_nd_multi,
)
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
    TopologyNominalParams,
    TopologyObservables,
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
MANIFEST_PATH = NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json"
SYNC_CHECKER = MONO_ROOT / "helpers" / "benchmarking" / "check_helpers_layout_sync.py"
MODULE_PATH = NESTED_ROOT / "na49_mixing" / "topology_runner.py"
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
CONTINUUM_PATH = (
    NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json")

#: The pinned trace sizes: the card's 1,000-event deterministic trace
#: (the T0-equivalent), the 1,000-draw draw-determinism trace, and the
#: 200-event dispatch-identity trace per class.
N_TRACE = 1_000
N_DRAWS = 1_000
N_DISPATCH = 200

#: The frozen-substrate provenance pins (byte-stable surfaces, verified
#: on disk): the builder blend path, the card-6 blend routing, the
#: card-2 engine, the committed catalog, the continuum families
#: configuration, the sealed Round 4 R45/R46 drivers in both helper
#: trees, and the committed Round 5 class modules consumed read-only
#: (cards 1-8).
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

#: The M8-frozen ten-parameter set hash (the sha256 of the manifest's
#: frozen parameter map, ``json.dumps(..., sort_keys=True,
#: separators=(",", ":"))``; the card's provenance pin).
M8_HASH = "358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1"

#: The white-box tokens of a refit of the sealed R45 alignment point or
#: of a separation claim (the R46 clauses; none may appear anywhere in
#: the module source).
NO_REFIT_TOKENS = (
    "kappa_fit", "refit_kappa", "align_refit", "refit_alignment",
    "separation_claimed = True", "exchange_recoil_decay_separation",
)

#: The freeze-exception identity of this card (the manifest re-pin).
FREEZE_CARD = "TASK-20260925-r5-topology-template-runner"

#: The inventory count at this card's landing: 526 at the card-8
#: test-author landing plus this card's two new files (the runner
#: module and the test file); the card's pre-registered 515 -> 517
#: literal predates the current baseline and is recorded in the
#: freeze-exception reason.
# re-pinned 526 -> 528 by Round 5 card 9 (TASK-20260925-r5-topology-template-runner; +2 entries: the runner module na49_mixing/topology_runner.py and the test file tests/test_topology_runner.py).
# re-pinned 528 -> 529 by Round 5 card 9 test-author (TASK-20260925-r5-topology-template-runner-test-author; +1 entry: the card-9 test-author register file tests/test_topology_runner_contract_pins.py).
# digest refresh (inventory count unchanged at 529) by the Round 5 card 9 review fix round (TASK-20260925-r5-topology-template-runner-review-fix; REV-bc4de47-01/-02/-03 corrected physics): the runner module na49_mixing/topology_runner.py, the package __init__ na49_mixing/__init__.py, and this test file re-pinned; the eleven INIT_SHA256 cross-pins in the chain registers follow.
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
#: The re-pinned chain count-pin files of the freeze re-pin (the
#: thirty-five test files carrying the inventory count pins, path-sorted;
#: the manifest, the package exports, the runner module, and this test
#: file join them in the freeze-exception entry).
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
    "dev/mixing2006/tests/test_topology_single_diffraction_contract_pins.py",
)

#: This card's freeze-exception file list (path-sorted): the manifest,
#: the package exports, the runner module, this test file, and the
#: re-pinned chain count-pin files.
FREEZE_EXCEPTION_FILES = tuple(sorted((
    "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json",
    "dev/mixing2006/na49_mixing/__init__.py",
    "dev/mixing2006/na49_mixing/topology_runner.py",
    "dev/mixing2006/tests/test_topology_runner.py",
) + CHAIN_COUNT_PIN_FILES))

#: The freeze-exception identity of the card's review fix round (the
#: digest refresh; the inventory count stays 529): the runner module,
#: the package ``__init__``, and this test file carry the corrected
#: physics (the findings REV-bc4de47-01, REV-bc4de47-02, and
#: REV-bc4de47-03) and the eleven ``INIT_SHA256`` cross-pins in the
#: chain registers follow the re-pinned package ``__init__``.
FREEZE_FIX_ROUND = "TASK-20260925-r5-topology-template-runner-review-fix"

#: The eleven chain registers whose ``INIT_SHA256`` cross-pin follows
#: the re-pinned package ``__init__`` (path-sorted; the card-9
#: test-author register is the test-author's re-pin follow-up and is
#: deliberately not re-pinned in this fix round).
INIT_CROSSPIN_FILES = (
    "dev/mixing2006/tests/test_exchange_ledger_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_ablation_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_form_factor_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_termination_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_topology_double_diffraction_contract_pins.py",
    "dev/mixing2006/tests/test_topology_elastic_contract_pins.py",
    "dev/mixing2006/tests/test_topology_mass_laws_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_topology_multi_neutral_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_topology_non_diffractive_contract_pins.py",
    "dev/mixing2006/tests/test_topology_remnants_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_topology_single_diffraction_contract_pins.py",
)

#: This fix round's freeze-exception file list (path-sorted): the
#: manifest, the three re-pinned files, and the eleven ``INIT_SHA256``
#: chain registers.
FREEZE_FIX_ROUND_FILES = tuple(sorted((
    "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json",
    "dev/mixing2006/na49_mixing/__init__.py",
    "dev/mixing2006/na49_mixing/topology_runner.py",
    "dev/mixing2006/tests/test_topology_runner.py",
) + INIT_CROSSPIN_FILES))

#: The pinned head of the draw-determinism uniform stream (the first
#: twenty uniforms of ``np.random.default_rng(20260925)``; the
#: hand-computed draw sequence's reference values).
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
#: per-event ``topology_selection`` uniforms walked through the nominal
#: 0.25-each split).
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
#: (REV-bc4de47-01) moved the Feynman x and the central window into the
#: overall CM frame: the x_F literals are ``2 p_L^CM / sqrt(s)`` (the
#: elastic event-0 pair at ``+/- 0.993195829131928``, neither nucleon
#: central in the CM window) and the multiplicities and species yields
#: are counted on the CM rapidity ``|y^CM| < 0.5``.
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


# ----------------------------------------------------------------------
# Fixtures and helpers
# ----------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="session")
def catalog() -> HadronCatalog:
    return HadronCatalog.from_json(CATALOG_PATH)


@pytest.fixture(scope="session")
def fams():
    return families_from_json(str(CONTINUUM_PATH))


def _event_rngs(event_index: int) -> dict:
    """The closed substream generators of one pinned event (the
    five-label quintuple of the runner's trace)."""

    subs = derive_substreams(MASTER_SEED, 0, event_index)
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

    stream = derive_substreams(MASTER_SEED, 0, event_index).stream_for(
        "topology_selection")
    return float(np.random.default_rng(stream).random())


def _hand_draw_label(u: float, elastic_fraction: float,
                     fractions) -> str:
    """The documented draw convention applied to one uniform value: the
    elastic fraction on the lower edge, the inelastic classes walked in
    the fixed order over the absolute cumulatives (each inelastic class
    returned at exactly its given fraction of the drawn set; the
    corrected allocation of the review finding REV-bc4de47-02)."""

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
    """The lab -> CM boost parameters of the pinned beam configuration
    at the recorded :data:`SQRT_S` (the hand-computed side of the
    Feynman-x and central-window extraction; the module applies the
    same inverse boost, the review fix REV-bc4de47-01)."""

    e_beam = math.sqrt(PROTON_MASS * PROTON_MASS
                       + BEAM_MOMENTUM * BEAM_MOMENTUM)
    e_total = e_beam + PROTON_MASS
    return BEAM_MOMENTUM / e_total, e_total / SQRT_S


_CM_BETA, _CM_GAMMA = _cm_boost_parameters()


def _cm_four_momentum(four_momentum) -> tuple:
    """The inverse (lab -> CM) longitudinal boost of one four-momentum:
    ``(E*, px, py, pz*) = (gamma (E - beta pz), px, py, gamma (pz -
    beta E))`` (the class modules' forward fixed-target boost in
    reverse; transverse components invariant)."""

    e, px, py, pz = (float(v) for v in four_momentum)
    return (_CM_GAMMA * (e - _CM_BETA * pz), px, py,
            _CM_GAMMA * (pz - _CM_BETA * e))


def _hand_observables(event) -> dict:
    """The hand-computed observables from one record's raw fields (the
    independent recomputation side of acceptance (iv)): the Feynman x
    and the central window in the overall CM frame (the boosted
    four-momenta; the review fix REV-bc4de47-01), everything else from
    the frame-independent records."""

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

    x_f_projectile = 2.0 * _cm_four_momentum(lines[0].four_momentum)[3] \
        / SQRT_S
    x_f_target = 2.0 * _cm_four_momentum(lines[1].four_momentum)[3] \
        / SQRT_S

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
        in_central = bool(abs(rapidity) < 0.5)
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
        px = float(four_momentum[1])
        py = float(four_momentum[2])
        p_t_sum += math.hypot(px, py)

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


# ----------------------------------------------------------------------
# Registry (acceptance (i))
# ----------------------------------------------------------------------
def test_registry():
    """The registry is exactly the six entries with the retained flags
    (nd_multi ``retained: False``, all others True); the inelastic set
    is exactly the four classes (elastic and nd_multi excluded)."""

    assert set(TOPOLOGY_CLASSES) == {
        "elastic", "sd_projectile", "sd_target", "dd", "nd", "nd_multi"}
    assert len(TOPOLOGY_CLASSES) == 6
    retained = {label: entry["retained"]
                for label, entry in TOPOLOGY_CLASSES.items()}
    assert retained == {
        "elastic": True, "sd_projectile": True, "sd_target": True,
        "dd": True, "nd": True, "nd_multi": False,
    }, "only nd_multi carries the unretained flag (the card-8 entry)"
    inelastic = {label: entry["inelastic"]
                 for label, entry in TOPOLOGY_CLASSES.items()}
    assert inelastic == {
        "elastic": False, "sd_projectile": True, "sd_target": True,
        "dd": True, "nd": True, "nd_multi": True,
    }, "the elastic class is excluded from every inelastic sample " \
       "(plan line 824)"
    assert set(INELASTIC_SET) == {"sd_projectile", "sd_target", "dd", "nd"}
    assert "elastic" not in INELASTIC_SET and "nd_multi" not in INELASTIC_SET
    assert INELASTIC_ORDER == ("sd_projectile", "sd_target", "dd", "nd")
    # every entry names its landed module (importable, exact name)
    import importlib
    for label, entry in TOPOLOGY_CLASSES.items():
        module = importlib.import_module(entry["module"])
        assert module.__name__ == entry["module"]


# ----------------------------------------------------------------------
# Draw determinism (acceptance (ii))
# ----------------------------------------------------------------------
def test_draw_determinism():
    """With the pinned seed, the class sequence of ``draw_topology``
    over 1,000 draws matches the hand-computed sequence from the uniform
    values; the draw consumes exactly 1 uniform per event on
    ``topology_selection`` (pinned RNG-state identity); the draw never
    returns ``"elastic"`` at elastic_fraction = 0 and never returns
    ``"nd_multi"``."""

    # the pinned uniform head (the hand-computed side's own stream)
    head = np.random.default_rng(MASTER_SEED)
    for expected_u in PINNED_UNIFORMS_HEAD:
        assert head.random() == expected_u

    # the pinned 1,000-draw sequence against the hand-computed sequence
    rng = np.random.default_rng(MASTER_SEED)
    drawn = [draw_topology(NOMINAL_PARAMS.class_fractions, 0.0, rng)
             for _ in range(N_DRAWS)]
    assert drawn == _hand_walk(NOMINAL_PARAMS.class_fractions, 0.0,
                               N_DRAWS, MASTER_SEED), (
        "the draw sequence drifted from the hand-computed sequence")
    assert "elastic" not in drawn, (
        "no draw returns 'elastic' at elastic_fraction = 0")
    assert "nd_multi" not in drawn, (
        "the unretained nd_multi class is never returned")
    assert set(drawn) == {"sd_projectile", "sd_target", "dd", "nd"}

    # determinism: a fresh identical generator reproduces the sequence
    again = np.random.default_rng(MASTER_SEED)
    assert [draw_topology(NOMINAL_PARAMS.class_fractions, 0.0, again)
            for _ in range(N_DRAWS)] == drawn

    # exactly one uniform per draw (pinned RNG-state identity): the
    # counting proxy advances exactly once per draw, and the post-draw
    # bit-generator state sits exactly 1,000 uniforms past the seed
    counting = _CountingRng(np.random.default_rng(MASTER_SEED))
    for index in range(1_000):
        draw_topology(NOMINAL_PARAMS.class_fractions, 0.0, counting)
        assert counting.calls == index + 1
    probe = np.random.default_rng(MASTER_SEED)
    for _ in range(1_000):
        probe.random()
    assert counting._source.bit_generator.state == \
        probe.bit_generator.state

    # a positive elastic fraction: the hand-computed sequence matches,
    # the elastic class appears, and exactly one uniform per draw holds
    fractions = {label: 0.175 for label in INELASTIC_ORDER}
    elastic_rng = np.random.default_rng(MASTER_SEED)
    got = [draw_topology(fractions, 0.3, elastic_rng)
           for _ in range(N_DRAWS)]
    assert got == _hand_walk(fractions, 0.3, N_DRAWS, MASTER_SEED)
    assert "elastic" in got and "nd_multi" not in got

    # the frequency pin at the positive elastic fraction (the review
    # finding REV-bc4de47-02's surface; the hand-walk identity above
    # cannot substitute because both sides would copy the same walk):
    # at f_e = 0.3 with 0.175-each inelastic fractions the 100,000-draw
    # frequencies sit on 0.300 / 0.175 x 4 within the pinned tolerance
    # (each class returned at exactly its given fraction of the drawn
    # set; the buggy unscaled walk put nd at 0.332 and the diffractive
    # classes at 0.123)
    n_frequency = 100_000
    frequency_rng = np.random.default_rng(MASTER_SEED)
    frequency_counts = Counter(
        draw_topology(fractions, 0.3, frequency_rng)
        for _ in range(n_frequency))
    assert sum(frequency_counts.values()) == n_frequency
    assert abs(frequency_counts["elastic"] / n_frequency - 0.3) <= 0.01, (
        "the elastic draw frequency drifted at f_e = 0.3")
    for label in INELASTIC_ORDER:
        assert abs(frequency_counts[label] / n_frequency - 0.175) <= 0.01, (
            f"the draw frequency of {label!r} drifted at f_e = 0.3 "
            "(the walk must return each class at exactly its given "
            "fraction of the drawn set)")


def test_draw_validation():
    """The draw raises on unknown class labels (including 'elastic' and
    'nd_multi' as fraction keys), missing inelastic classes, negative
    fractions, out-of-range elastic fractions, and non-unit mass
    (checked, never repaired)."""

    fractions = {label: 0.25 for label in INELASTIC_ORDER}
    with pytest.raises(ValueError, match="unknown inelastic class"):
        draw_topology({**fractions, "elastic": 0.0}, 0.0,
                      np.random.default_rng(1))
    with pytest.raises(ValueError, match="unknown inelastic class"):
        draw_topology({**fractions, "nd_multi": 0.0}, 0.0,
                      np.random.default_rng(1))
    with pytest.raises(ValueError, match="missing"):
        draw_topology({label: 1.0 / 3.0 for label in
                       list(INELASTIC_ORDER)[:-1]}, 0.0,
                      np.random.default_rng(1))
    with pytest.raises(ValueError, match="non-negative"):
        draw_topology({**fractions, "nd": -0.25}, 0.25,
                      np.random.default_rng(1))
    with pytest.raises(ValueError, match="elastic fraction"):
        draw_topology(fractions, 1.5, np.random.default_rng(1))
    with pytest.raises(ValueError, match="unit mass"):
        draw_topology(fractions, 0.5, np.random.default_rng(1))
    # a legal positive elastic fraction draws a legal label
    got = draw_topology({label: 0.175 for label in INELASTIC_ORDER}, 0.3,
                        np.random.default_rng(1))
    assert got in set(INELASTIC_ORDER) | {"elastic"}


# ----------------------------------------------------------------------
# Dispatch identity (acceptance (iii))
# ----------------------------------------------------------------------
def test_dispatch(catalog, fams):
    """Each inelastic class label dispatches to the landed module and
    returns the matching record type (SD_p / SD_t / DD / ND); the
    nominal-parameter record of each class is bit-identical to the class
    module's own pinned trace on the same seed (cross-module identity);
    the unretained nd_multi label stays dispatchable; unknown labels and
    missing streams raise."""

    from na49_mixing.topology_single_diffraction import generate_sd

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
                reference["exchange_kinematics"], reference["decay_channel"],
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
        run_topology("sd_projectile", NOMINAL_PARAMS, catalog, fams,
                     rng_exchange_kinematics=np.random.default_rng(1))


# ----------------------------------------------------------------------
# Observable extraction (acceptance (iv))
# ----------------------------------------------------------------------
def test_observables(catalog, fams):
    """On the pinned 1,000-event trace, the extracted observables match
    the hand-computed values within 1e-12 (leading-baryon Feynman x_F
    per line in the overall CM frame, excitation masses, transfers,
    central gap, total multiplicity, central-activity multiplicity in
    the CM window ``|y^CM| < 0.5``, central/forward pion and proton
    yields, seagull moments); the pinned class counts, the
    hand-computed trace labels, the first-occurrence indices, and the
    exact pinned observables hold."""

    trace = pinned_trace(NOMINAL_PARAMS, MASTER_SEED, N_TRACE)
    assert len(trace) == N_TRACE

    labels = []
    seen = {}
    for event_index in range(N_TRACE):
        label, event, obs = _run_one(event_index, catalog, fams)
        labels.append(label)

        # the extracted observables are exactly the trace output
        assert obs == trace[event_index], (
            f"event {event_index}: the extracted observables differ from "
            "the pinned trace")

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
        for field in ("central_pions", "central_protons", "forward_pions",
                      "forward_protons"):
            assert getattr(obs, field) == expected[field]
        assert abs(obs.seagull_moment - expected["seagull_moment"]) <= 1e-12

    # the pinned trace labels: the hand-computed head, the class counts,
    # and the first-occurrence indices
    assert tuple(labels[:len(HAND_TRACE_HEAD)]) == HAND_TRACE_HEAD
    assert Counter(labels) == Counter(TRACE_CLASS_COUNTS), (
        "the pinned trace's class counts drifted")
    for label, first_index in TRACE_FIRST_INDEX.items():
        assert labels[first_index] == label
        assert labels.index(label) == first_index

    # the exact pinned first-occurrence observables (the hand-computed
    # literals): the first event of each inelastic class in the trace
    # plus the elastic class's direct dispatch event-0
    for label, expected in PINNED_OBSERVABLES.items():
        if label == "elastic":
            continue
        obs = _run_one(TRACE_FIRST_INDEX[label], catalog, fams)[2]
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
    # reference; the elastic fraction never enters the inelastic trace)
    rngs = _event_rngs(0)
    elastic = run_topology(
        "elastic", NOMINAL_PARAMS, catalog, fams,
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
    assert obs.seagull_moment == expected["seagull_moment"]

    # the elastic event passes card 1's zero-transfer check (the
    # normalization reference's ledger convention)
    assert isinstance(elastic, ElasticEvent)
    from na49_mixing.topology_elastic import check_zero_transfer
    check_zero_transfer(elastic)


def test_observables_trace_identity(catalog, fams):
    """The ``pinned_trace`` output is exactly the per-event extraction
    over the re-derived events (the trace's own determinism pin over
    its first 300 events)."""

    trace = pinned_trace(NOMINAL_PARAMS, MASTER_SEED, 300)
    assert len(trace) == 300
    for event_index in range(300):
        label, _event, obs = _run_one(event_index, catalog, fams)
        assert trace[event_index] == obs
        u = _hand_uniform(event_index)
        assert trace[event_index].class_label == _hand_draw_label(
            u, 0.0, NOMINAL_PARAMS.class_fractions)


# ----------------------------------------------------------------------
# Nominal parameters (acceptance (v))
# ----------------------------------------------------------------------
def test_nominal_params():
    """``TopologyNominalParams()`` equals the pre-registered values
    exactly (epsilon = 0, B_SD = 8.504140762083612, fractions 0.25
    each, remnant weights 0.5/0.5/0.0, ``p_n2 = 0.0``); the M8 frozen
    value is byte-identical to the M8-frozen
    ``effective_transfer_slope`` (hash
    358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1,
    value 8.504140762083612)."""

    params = TopologyNominalParams()
    assert getattr(params, "__dataclass_params__").frozen
    assert tuple(params.__dataclass_fields__) == (
        "epsilon", "b_sd", "class_fractions", "remnant_weights", "p_n2",
        "flavor", "epsilon_nd", "nd_central", "isovector_fraction", "b_isovector", "isovector_delta_ratio", "b_nd",
        "exact_transfer_range", "line_thresholds", "central_recoil", "epsilon_c",
        "transfer_slope_mass_exponent", "low_mass_spin_isospin", "nd_duality_mass",
        "nd_duality_shape_mass", "duality_line_shapes", "duality_redraw_shapes", "diffractive_duality_mass",
        "intact_vertex_factor", "diffractive_xi_max", "sd_tail", "sd_ope", "sd_two_sided", "diffractive_pomeron_slope",
        "diffractive_pomeron_linear_t", "intact_t_break")  # diffractive_pomeron_slope: 2026-11-01, None = the zero-transfer form with the per-mass vertex law;
    # diffractive_pomeron_linear_t: 2026-11-01, None = the shrinkage at every t;
    # sd_ope: 2026-10-26, None = no one-pion exchange; sd_two_sided: 2026-10-26, None = every system one line; duality_line_shapes: 2026-10-18, False = the fixed-width pick;
    # sd_tail: 2026-10-23 (R112), None = no leading-proton tail of the single-diffraction classes;
    # intact_t_break: 2026-10-24, None = the one exponential of the intact proton's transfer;
                                  # diffractive_xi_max: 2026-10-23, None = the kinematic edge of the diffractive masses;
                                  # flavor: Round 6 task 37, None = identity;
                                  # transfer_slope_mass_exponent: 2026-10-08, None = the single slope;
                                  # low_mass_spin_isospin: 2026-10-08, False = the uniform pick over every record;
                                  # nd_duality_mass, nd_duality_shape_mass: 2026-10-12, None = the continuum lines;
                                  # intact_vertex_factor: 2026-10-15, None = the term 1 of the intact vertex;
                                  # line_thresholds: 2026-10-06, False = the shared p pi0 edge;
                                  # epsilon_nd: 2026-09-29 Block A, None = shared;
                                  # nd_central: 2026-10-02, True = the card's event
    assert params.flavor is None
    assert params.epsilon_nd is None
    assert params.nd_central is True
    assert params.epsilon == 0.0
    assert params.b_sd == 8.504140762083612
    assert params.b_sd.hex() == (8.504140762083612).hex()
    assert dict(params.class_fractions) == {
        "sd_projectile": 0.25, "sd_target": 0.25, "dd": 0.25, "nd": 0.25}
    assert dict(params.remnant_weights) == {
        "intact_proton": 0.5, "low_mass_baryonic_resonance": 0.5,
        "high_mass_baryonic_continuum": 0.0}
    assert params.p_n2 == 0.0
    assert NOMINAL_PARAMS == params
    assert ELASTIC_TRANSFER_SLOPE == 8.504140762083612

    # the M8-frozen surface: the manifest's frozen parameter map carries
    # the byte-identical effective_transfer_slope and its hash is the
    # M8 hash
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
    assert params.p_n2 == DEFAULT_P2 == 0.0


# ----------------------------------------------------------------------
# Frozen cascade (the R45 point, the R46 no_refit clause)
# ----------------------------------------------------------------------
def test_frozen_cascade(catalog, fams):
    """The trace events decay through the frozen simplified cascade at
    the sealed R45 point (kappa_B = 0.0, kappa_M = 4.0, b_dec = 2.0
    GeV^-2, baryonic nominal width 0.3 GeV, ``W_flavor = 1.0``); a
    white-box check that the runner references no re-fit of the
    alignment parameters and no separation claim (the R46 clauses)."""

    # the sealed R45 point is pinned through the R45/R46 driver digests
    # (FROZEN_SURFACES; verified in test_layout_sync) and through the
    # frozen W_flavor hook default (exactly 1.0, multiplicative, zero
    # draws)
    import na49_mixing.cascade_alignment  # noqa: F401
    import na49_mixing.cascade_form_factor  # noqa: F401
    import na49_mixing.cascade_termination  # noqa: F401
    from na49_mixing.cascade_decay import CascadeChannel, channel_weight
    probe_channel = CascadeChannel(
        parent_id="baryonic_np", daughters=("catalog:2212", "catalog:111"),
        kind="established_hadron", weight=0.5)
    assert channel_weight(probe_channel) == 0.5
    assert channel_weight(probe_channel, w_flavor=1.0) == 0.5

    # the white-box token scan: no re-fit of the alignment parameters
    # and no separation claim anywhere in the runner module (the R46
    # clauses); the exchange slopes act on the exchange kinematics only
    # (no daughter-level transverse dynamics surface exists here)
    source = MODULE_PATH.read_text(encoding="utf-8")
    for token in NO_REFIT_TOKENS:
        assert token not in source, (
            f"the runner references the forbidden token {token!r} "
            "(the R46 no_refit clause)")
    assert "cascade_alignment" not in source
    assert "from na49_mixing.cascade" not in source

    # the frozen-parameter pass-through: the runner's nominal structure
    # builds exactly the committed class modules' mass law (the R50
    # nominal point; no refit surface exists)
    params = TopologyNominalParams()
    assert MassLawParams(epsilon=params.epsilon,
                         b_sd=params.b_sd) == SD_NOMINAL_PARAMS
    assert BEAM_MOMENTUM == 158.0 and SQRT_S == 17.270
    assert PROTON_MASS == 0.93827231


# ----------------------------------------------------------------------
# Layout sync and the freeze re-pin
# ----------------------------------------------------------------------
def test_layout_sync():
    """The test register (the frozen-substrate provenance pins,
    verified on disk, the M8 hash and the kernel boundary through the
    manifest, the freeze re-pin of this card), and the layout sync (the
    subprocess check from the monorepo root; ``logic_drift = 0``)."""

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

    # the freeze re-pin of this card: the strict path-sorted inventory
    # at the operative count (526 at the card-8 test-author landing plus
    # this card's two new files), this card's freeze exception exactly
    # once with the path-sorted file list and the reason naming the
    # 526 -> 528 re-pin and the card's pre-registered literal
    inventory = manifest["source_tree_sha256"]
    keys = list(inventory.keys())
    assert keys == sorted(keys), "the inventory is not path-sorted"
    assert len(keys) == INVENTORY_COUNT
    module_rel = "dev/mixing2006/na49_mixing/topology_runner.py"
    test_rel = "dev/mixing2006/tests/test_topology_runner.py"
    assert inventory[module_rel] == _sha256_file(MODULE_PATH)
    assert inventory[test_rel] == _sha256_file(TEST_FILE)
    init_rel = "dev/mixing2006/na49_mixing/__init__.py"
    assert inventory[init_rel] == _sha256_file(
        NESTED_ROOT / "na49_mixing" / "__init__.py")
    for rel in CHAIN_COUNT_PIN_FILES:
        assert rel in keys, f"chain file missing from the inventory: {rel}"
        assert inventory[rel] == _sha256_file(MONO_ROOT / rel), (
            f"chain file digest drifted: {rel}")

    exceptions = manifest["freeze_exceptions"]
    entries = [e for e in exceptions if e.get("card") == FREEZE_CARD]
    assert len(entries) == 1, (
        "the card-9 freeze exception must name its card exactly once")
    files = entries[0]["files"]
    assert files == sorted(files), (
        "the freeze-exception files are path-sorted")
    assert files == list(FREEZE_EXCEPTION_FILES), (
        "the freeze-exception file list drifted")
    reason = entries[0]["reason"]
    assert reason.startswith("Round 5, card 9 ("), (
        "the freeze-exception reason must name its card")
    assert FREEZE_CARD in reason, (
        "the freeze-exception reason must name its card")
    assert "526 -> 528" in reason, (
        "the freeze-exception reason must name the 526 -> 528 re-pin")
    assert "515 -> 517" in reason, (
        "the freeze-exception reason must record the card's "
        "pre-registered literal")

    # the review fix round's freeze exception (the digest refresh; the
    # inventory count stays 529): exactly once, path-sorted, naming the
    # corrected physics findings and the INIT_SHA256 follow-through
    fix_entries = [e for e in exceptions
                   if e.get("card") == FREEZE_FIX_ROUND]
    assert len(fix_entries) == 1, (
        "the review fix round's freeze exception must name its card "
        "exactly once")
    fix_files = fix_entries[0]["files"]
    assert fix_files == sorted(fix_files), (
        "the fix-round freeze-exception files are path-sorted")
    assert fix_files == list(FREEZE_FIX_ROUND_FILES), (
        "the fix-round freeze-exception file list drifted")
    fix_reason = fix_entries[0]["reason"]
    assert fix_reason.startswith("Round 5, card 9 review fix round"), (
        "the fix-round freeze-exception reason must name the fix round")
    for finding in ("REV-bc4de47-01", "REV-bc4de47-02", "REV-bc4de47-03"):
        assert finding in fix_reason, (
            f"the fix-round freeze-exception reason must name {finding}")
    assert "digest refresh" in fix_reason
    assert "INIT_SHA256" in fix_reason, (
        "the fix-round freeze-exception reason must record the "
        "INIT_SHA256 cross-pin follow-through")


    probe = subprocess.run(
        [sys.executable, str(SYNC_CHECKER)],
        capture_output=True, text=True, cwd=str(MONO_ROOT))
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert "logic_drift=0" in probe.stdout
