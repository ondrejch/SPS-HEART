"""Card tests for the one-neutral-excitation non-diffractive class.

Round 5, card 7 (``TASK-20260925-r5-non-diffrative-central-excitation``;
plan section 22 task 31, line 1756; plan section 11.1 item 5, line 828,
with plan section 11.3, lines 844-870).  The card's pre-registered
deterministic tests against the module
``na49_mixing/topology_non_diffractive.py``:

- ``test_central_neutrality``: on a pinned 10,000-event trace the
  central source carries ``(B, Q, S, 2I3) = (0, 0, 0, 0)`` exactly and
  every central decay vertex closes locally to ``(0, 0, 0)``;
  ``check_central_neutrality`` raises on each planted violation
  (central ``B = 1``, central ``Q = 1``, central ``S = 1``, one open
  central vertex).
- ``test_conservation``: over the pinned trace the final state carries
  exactly ``(B, Q, S) = (2, 2, 0)`` and the four-momentum closure is
  within ``1e-9`` relative; every event passes card 2's
  ``audit_event`` with zero violations.
- ``test_channel_legality``: every central decay parent is a
  committed-catalog ``(0, 0, 0)`` record or a neutral-mesonic continuum
  state; on a pinned 10,000-vertex trace over the central decays, the
  channel daughters sum to ``(0, 0, 0)`` exactly; baryon-antibaryon
  channels appear only above threshold (``p p-bar`` at ``M >= 2 m_p``,
  hyperon pairs at their two-body thresholds, ``1e-12``); the ``K
  K-bar`` direct channel and the ``K*0 K-bar*0`` and phi channels are
  present in the pinned catalog enumeration (plan section 11.3).
- ``test_strangeness_locality``: on the pinned trace, associated
  strangeness (channels with ``S != 0``, e.g. ``K Lambda``) occurs only
  in the baryonic excitation decays (``(1, 1, 0)`` parents); the net
  strangeness of the central source and of every central decay vertex
  is 0 (the ``phi -> K+ K-`` channel is legal centrally with net
  ``S = 0``).
- ``test_phase_space``: ``M_c^2`` in ``[m_pi0^2, (sqrt(s) - M_1 - M_2)^2]``
  and the three-system closure ``M_1 + M_2 + M_c <= sqrt(s)`` within
  ``1e-12`` on the pinned trace; ``t_c = t_1 + t_2`` exactly.
- ``test_draw_budget``: a pinned 1,000-event trace consumes exactly 2
  ``spectral_state`` + the ``exchange_kinematics`` uniforms per event
  plus the frozen Round 4 per-vertex budget (pinned RNG-state identity;
  the ``topology_selection`` draw belongs to the runner, card 9).
- ``test_frozen_cascade``: all three systems decay through the frozen
  simplified cascade (kappa_B = 0.0, kappa_M = 4.0, b_dec = 2.0
  GeV^-2, nominal width 0.3 GeV, ``W_flavor = 1.0``); a white-box check
  that no module in this card references a re-fit of the alignment
  parameters (the R46 ``no_refit`` clause).
- ``test_layout_sync``: the test register and the layout sync.

Operative resolutions recorded here (the card's literals against the
committed tree):

- The exchange-kinematics budget: the card's objective names "5 on
  exchange_kinematics (t_1, t_2, M_c^2)" while its own decomposition at
  card 4's documented budgets is exactly 1 + 1 + 2 = 4 uniforms (card
  4: "Exactly 2 uniforms per marginal or central M^2 draw and 1 uniform
  per t draw"; no committed surface consumes a fifth uniform, and the
  closure is exact by construction, so no rejection attempt exists).
  The operative budget is the card-4 decomposition: 4 uniforms, pinned
  here with RNG-state identity.
- The pion threshold: the card's interface carried the 0.1349768 GeV
  quote; the committed catalog record pdg_id 111 carries
  0.1349766 GeV and card 4 resolved its own interface literal to the
  committed catalog value (the card-4 resolution precedent).  The
  operative ``m_pi0`` is card 4's
  :data:`~na49_mixing.topology_mass_laws.PI0_MASS`.
- The generator signature: the card's sketch compresses the substream
  arguments to a single ``rng``; the card-5/card-6 pattern passes the
  closed per-label substream generators (the draw-budget contract runs
  on the closed labels), together with the continuum families
  configuration the cascade enumeration consumes.  The record sketch's
  ``t_c`` field is carried verbatim and the two per-exchange transfers
  ``t1``/``t2`` (the summands the card's own ``t_c = t_1 + t_2``
  identity consumes) are carried additively.
- The three-system solve convention: the central system is produced at
  rest in the overall center-of-mass frame, the two baryonic systems
  recoil back-to-back against it at the pair invariant
  ``sqrt(s) - M_c``, and the production angle follows from the drawn
  projectile-side transfer through the documented invariant relation
  (the committed single- and double-diffraction pattern; the drawn
  ``t_2`` acts through ``t_c`` of the central law).  The exact
  three-system solve is over-determined by one in the drawn transfers
  (the 2-to-3 configuration's realized invariants trace a curve, the
  double-diffraction card's over-determination fact at the
  three-system extension), and the recorded drawn values stay the
  exchange-kinematics record.
- The baryonic decay routing: the two baryonic systems are card 3
  remnants; the frozen simplified cascade's established-record route is
  the card-9 handoff's delegation target (the byte-unchanged Round 2
  kernel, the card-5/card-6 established-sibling assembly convention),
  with a stable record (the intact proton and the proton's own catalog
  record) final with zero decay draws.  Two of the ten low-mass
  catalog records (Delta(1600)+ and Delta(1700)+) carry pole masses
  below their own recorded support windows, so the handoff predicate
  refuses them at their pole masses and the delegation runs through the
  kernel directly (identical consumption and bit-identical decay
  vertices).
- The plan section 11.3 channel-set literals against the committed
  enumeration: the ``K K-bar`` direct channels are live (the family's
  ``(321, -321)`` and ``(310, 130)`` classes, open above the two-kaon
  threshold, and recorded channels of the committed ``(0, 0, 0)``
  records f2(1270), rho3(1690)0, f4(2050), and phi(1020)).  The phi
  record carries the plan's phi route (its recorded ``K+ K-`` and
  ``K0S K0L`` channels) and the ``K*`` classes are recorded on the
  declared strange mesonic families' allowed-decay classes; no
  committed ``(0, 0, 0)`` record carries a ``K*0 K-bar*0`` channel or a
  baryon-antibaryon channel, so the ``K*0 K-bar*0`` presence is
  asserted at its committed declared-family surface and the
  baryon-antibaryon clause binds its exact two-body thresholds
  vacuously at this substrate (the enumeration opens no such channel;
  the threshold discipline is asserted structurally).

The pinned traces draw on the Round 5 master seed 20260925 (round 5
index item 9), replica 0, one substream quadruple per event index on
the closed labels (``spectral_state``, ``exchange_kinematics``,
``decay_channel``, ``decay_kinematics``); the ``topology_selection``
label is never touched by the module (the runner's draw, card 9).  The
pinned central-decay trace draws its masses on card 4's
central-production law at a dedicated stream (event index 30,000) and
its decay substreams at the event indices 20,000 + i (disjoint from
the class traces); the recursion sub-trace walks the pinned mass grid
above the continuum opening.

Hermetic: the module, the committed catalog, the continuum families
configuration, the Round 4 cascade modules, and the committed cards 1-5
modules (all consumed read-only), numpy, and the standard library only;
no event generation driver, no external tables, no Monte Carlo run
(the runs R50-R55 exercise the class statistically).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.cascade_decay import (
    ISOTROPIC_ALIGNMENT_LABEL,
    CascadeState,
    enumerate_channels,
)
from na49_mixing.exchange_ledger import (
    Exchange,
    LineState,
    audit_event,
)
from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.quantum_numbers import quantum_numbers
from na49_mixing.rng_substreams import SUBSTREAM_LABELS, derive_substreams
from na49_mixing.spectral_density import families_from_json
from na49_mixing.topology_elastic import (
    BEAM_MOMENTUM,
    ELASTIC_TRANSFER_SLOPE,
    PROTON_MASS,
    SQRT_S,
)
from na49_mixing.topology_mass_laws import (
    PI0_MASS,
    draw_central,
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
    ND_CLASS_LABEL,
    ND_LINE_LABELS,
    NOMINAL_ND_REMNANT_WEIGHTS,
    NOMINAL_PARAMS,
    PROJECTILE_LINE_LABEL as PROJECTILE_LINE,
    TARGET_LINE_LABEL as TARGET_LINE,
    CentralNeutralityViolation,
    NDEvent,
    _beam_configuration,
    _cm_triplet,
    _decay_central,
    _incoming_lines,
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
MANIFEST_PATH = NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json"
SYNC_CHECKER = MONO_ROOT / "helpers" / "benchmarking" / "check_helpers_layout_sync.py"
MODULE_PATH = NESTED_ROOT / "na49_mixing" / "topology_non_diffractive.py"
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
CONTINUUM_PATH = (
    NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json")

#: The Round 5 master seed (round 5 index item 9), fixed for the pinned
#: traces; replica 0, one substream quadruple per event index.
MASTER_SEED = 20260925

#: The pinned trace sizes (the card's 10,000-event neutrality and
#: conservation traces and the 1,000-event draw-budget trace).
N_CONSERVATION = 10_000
N_BUDGET = 1_000

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

#: The exact central limits of the pinned intact-pair configuration
#: (card 4's recorded values at ``SQRT_S = 17.270``).
M2_MIN_PINNED = 0.01821868254756
M2_MAX_PINNED = 236.95846853605096

#: The pinned draw-budget totals of the 1,000-event budget trace
#: (master seed 20260925, replica 0, one shared stream per label in
#: event order): the operative exchange-kinematics budget is card 4's
#: decomposition 1 + 1 + 2 = 4 uniforms per event (the card's "5"
#: literal resolution, recorded in the module and test docstrings), and
#: the frozen Round 4 per-vertex decay budget totals on the
#: ``decay_channel`` / ``decay_kinematics`` streams at this landing
#: (a catalog handoff and the unstable-sibling kernels add the Round 2
#: kernel's own documented draws on the same streams).
# re-pinned 2026-09-30 for the corrected final state (decays at every assembly point, the isospin rule)
ND_CHANNEL_TOTAL = 1_834
ND_KINEMATICS_TOTAL = 4_500
ND_ZERO_DRAW_EVENTS = 129

#: The freeze-exception identity of this card (the manifest re-pin).
FREEZE_CARD = "TASK-20260925-r5-non-diffrative-central-excitation"

#: The inventory count at this card's landing: 520 at the card-6
#: landing plus the card's two new files (the module and the test
#: file); the card's pre-registered 511 -> 513 literal predates the
#: current baseline and is recorded in the freeze-exception reason.
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
#: The re-pinned chain count-pin files of the freeze re-pin (the
#: thirty-one test files carrying the inventory count pins, path-sorted;
#: the manifest, the package exports, the module, and this test file
#: join them in the freeze-exception entry).
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
    "dev/mixing2006/tests/test_topology_remnants_contract_pins.py",
    "dev/mixing2006/tests/test_topology_single_diffraction_contract_pins.py",
)

#: This card's freeze-exception file list (path-sorted): the manifest,
#: the package exports, the module, this test file, and the
#: thirty-one re-pinned chain count-pin files.
FREEZE_EXCEPTION_FILES = tuple(sorted((
    "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json",
    "dev/mixing2006/na49_mixing/__init__.py",
    "dev/mixing2006/na49_mixing/topology_non_diffractive.py",
    "dev/mixing2006/tests/test_topology_non_diffractive.py",
) + CHAIN_COUNT_PIN_FILES))

#: The pinned recursion sub-trace masses (the deterministic grid above
#: the continuum opening of the neutral-mesonic family, plus the deep
#: probes across the drawn window's upper reach; the central cascade's
#: recursion, guard margin, and multiplicity structure).
RECURSION_GRID = tuple(
    [3.0 + 0.05 * k for k in range(40)]
    + [6.0, 8.0, 10.0, 12.0, 15.0]
)


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


def _event_rngs(event_index: int):
    """The closed substream generators of one pinned event."""

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


def _incoming_sum() -> tuple[float, float, float, float]:
    """The lab four-momentum sum of the pinned incoming pair."""

    lines = _incoming_lines()
    beam = lines[0].four_momentum
    target = lines[1].four_momentum
    if beam is None or target is None:
        raise ValueError("the pinned incoming lines carry no four-momenta")
    return (beam[0] + target[0], beam[1] + target[1],
            beam[2] + target[2], beam[3] + target[3])


def _final_sum(event: NDEvent) -> tuple[float, float, float, float]:
    """The lab four-momentum sum of the event's final state."""

    return (
        math.fsum(h.four_momentum[0] for h in event.final_state),
        math.fsum(h.four_momentum[1] for h in event.final_state),
        math.fsum(h.four_momentum[2] for h in event.final_state),
        math.fsum(h.four_momentum[3] for h in event.final_state),
    )


def _final_qn(event: NDEvent) -> tuple[int, int, int]:
    """The exact ``(B, Q, S)`` sum of the event's final state."""

    return (
        sum(h.qn[0] for h in event.final_state),
        sum(h.qn[1] for h in event.final_state),
        sum(h.qn[2] for h in event.final_state),
    )


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
    for line, source in ((PROJECTILE_LINE, event.lines[0]),
                         (TARGET_LINE, event.lines[1])):
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


def _check_conservation(event: NDEvent) -> float:
    """The exact ``(B, Q, S)`` bookkeeping and the relative
    four-momentum closure of one event; returns the worst relative
    closure component."""

    assert _final_qn(event) == (2, 2, 0)
    total = _final_sum(event)
    reference = _incoming_sum()
    e_total = reference[0]
    p_beam = reference[3]
    return max(
        abs(total[0] - reference[0]) / e_total,
        abs(total[1]) / p_beam,
        abs(total[2]) / p_beam,
        abs(total[3] - reference[3]) / p_beam,
    )


def _check_mass_shell(event: NDEvent) -> float:
    """The worst relative mass-shell deviation of the final-state
    hadrons (the below-threshold central state included)."""

    worst = 0.0
    for hadron in event.final_state:
        e, px, py, pz = hadron.four_momentum
        m2 = e * e - px * px - py * py - pz * pz
        # a massless final state (the photon of a radiative channel,
        # 2026-10-02) is on shell relative to its energy scale
        scale = (max(hadron.mass * hadron.mass, 1e-9) if hadron.mass > 0.0
                 else max(float(hadron.four_momentum[0]) ** 2, 1e-9))
        worst = max(worst, abs(m2 - hadron.mass * hadron.mass) / scale)
    return worst


@pytest.fixture(scope="session")
def pinned_trace(catalog, fams):
    """The pinned 10,000-event non-diffractive trace."""

    events = []
    for event_index in range(N_CONSERVATION):
        rngs = _event_rngs(event_index)
        events.append(generate_nd(NOMINAL_PARAMS,
                                  NOMINAL_ND_REMNANT_WEIGHTS,
                                  catalog, fams, *rngs))
    return events


@pytest.fixture(scope="session")
def central_trace(catalog, fams):
    """The pinned central-decay trace: 10,000 central-source decays at
    the drawn masses of card 4's central-production law (one shared
    mass stream at the dedicated event index 30,000; one decay
    substream pair per decay at the event indices 20,000 + i), plus the
    pinned recursion grid above the continuum opening.

    Returns ``(drawn, recursion)``: the per-decay final-state tuples of
    the two sub-traces.
    """

    m2_min, m2_max = nd_limits(PROTON_MASS, PROTON_MASS, SQRT_S)
    mass_rng = np.random.default_rng(derive_substreams(
        MASTER_SEED, 0, 30_000).stream_for("exchange_kinematics"))
    drawn = []
    for index in range(10_000):
        m2_c = draw_central(m2_min, m2_max, NOMINAL_PARAMS, -0.2, mass_rng)
        mass = math.sqrt(m2_c)
        subs = derive_substreams(MASTER_SEED, 0, 20_000 + index)
        rng_channel = np.random.default_rng(subs.stream_for("decay_channel"))
        rng_kinematics = np.random.default_rng(
            subs.stream_for("decay_kinematics"))
        drawn.append(_decay_central(
            (mass, 0.0, 0.0, 0.0), mass, catalog, fams,
            rng_channel, rng_kinematics))
    recursion = []
    for index, mass in enumerate(RECURSION_GRID):
        subs = derive_substreams(MASTER_SEED, 0, 40_000 + index)
        rng_channel = np.random.default_rng(subs.stream_for("decay_channel"))
        rng_kinematics = np.random.default_rng(
            subs.stream_for("decay_kinematics"))
        recursion.append(_decay_central(
            (mass, 0.0, 0.0, 0.0), mass, catalog, fams,
            rng_channel, rng_kinematics))
    return drawn, recursion


# ----------------------------------------------------------------------
# test_central_neutrality
# ----------------------------------------------------------------------
def test_central_neutrality(catalog, fams, pinned_trace):
    """On the pinned 10,000-event trace the central source carries
    ``(B, Q, S, 2I3) = (0, 0, 0, 0)`` exactly and every central decay
    vertex closes locally to ``(0, 0, 0)``; ``check_central_neutrality``
    raises on each planted violation (central ``B = 1``, central
    ``Q = 1``, central ``S = 1``, one open central vertex)."""

    assert len(pinned_trace) == N_CONSERVATION
    for event in pinned_trace:
        assert event.central_source.qn == CENTRAL_QN == (0, 0, 0, 0)
        check_central_neutrality(event)  # raises on any violation

    # the planted violations: each is detected, naming the offending
    # component or the open vertex
    event = pinned_trace[0]
    source = event.central_source
    for index, (label, planted) in enumerate((
            ("B", (1, 0, 0, 0)),
            ("Q", (0, 1, 0, 0)),
            ("S", (0, 0, 1, 0)),
    )):
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


# ----------------------------------------------------------------------
# test_conservation
# ----------------------------------------------------------------------
def test_conservation(catalog, fams, pinned_trace):
    """Over the pinned 10,000-event trace the final state carries
    exactly ``(B, Q, S) = (2, 2, 0)`` and the four-momentum closure is
    within ``1e-9`` relative; every event passes card 2's
    ``audit_event`` with zero violations (two vacuum transfers, both
    baryonic lines ``(1, 1, 0, 1)``, the central source ``(0, 0, 0,
    0)``, the local decay closures)."""

    assert len(pinned_trace) == N_CONSERVATION
    worst_closure = 0.0
    worst_shell = 0.0
    for event in pinned_trace:
        assert isinstance(event, NDEvent)
        assert event.class_label == ND_CLASS_LABEL == "nd"
        assert event.final_state
        assert {h.line for h in event.final_state} == ND_LINE_LABELS
        worst_closure = max(worst_closure, _check_conservation(event))
        worst_shell = max(worst_shell, _check_mass_shell(event))
        report = _audit(event)
        assert not report.violations, report.violations
    assert worst_closure <= 1e-9, worst_closure
    assert worst_shell <= 1e-9, worst_shell


# ----------------------------------------------------------------------
# test_channel_legality
# ----------------------------------------------------------------------
def test_channel_legality(catalog, fams, pinned_trace, central_trace):
    """Every central decay parent is a committed-catalog ``(0, 0, 0)``
    record or a neutral-mesonic continuum state; on the pinned
    central-decay trace the channel daughters sum to ``(0, 0, 0)``
    exactly; baryon-antibaryon channels appear only above threshold
    (the clause binds at its exact two-body thresholds; the committed
    enumeration opens none); the ``K K-bar`` direct channel and the
    ``K*0 K-bar*0`` and phi channels are present in the pinned catalog
    enumeration (plan section 11.3)."""

    drawn, recursion = central_trace
    assert len(drawn) == 10_000
    assert len(recursion) == len(RECURSION_GRID)

    # (1) every central decay's collected final state closes to
    # (0, 0, 0) exactly (the per-vertex closure is card 4's
    # draw_decay check, which raised on any violation during the
    # trace); the same exact closure on every event of the pinned
    # event trace's central line
    for finals in drawn:
        assert (sum(h.qn[0] for h in finals),
                sum(h.qn[1] for h in finals),
                sum(h.qn[2] for h in finals)) == (0, 0, 0)
    for finals in recursion:
        assert (sum(h.qn[0] for h in finals),
                sum(h.qn[1] for h in finals),
                sum(h.qn[2] for h in finals)) == (0, 0, 0)
    for event in pinned_trace:
        central = [h for h in event.final_state
                   if h.line == CENTRAL_LINE_LABEL]
        assert (sum(h.qn[0] for h in central),
                sum(h.qn[1] for h in central),
                sum(h.qn[2] for h in central)) == (0, 0, 0)

    # (2) at every realized and pinned mass, every enumerated channel's
    # daughters sum to (0, 0, 0) exactly and every established channel
    # is open above its daughter pole-mass threshold (1e-12); the
    # baryon-antibaryon clause: the committed enumeration opens no
    # baryon-antibaryon channel, so the clause binds vacuously at this
    # substrate, and the threshold guard is asserted structurally for
    # the pairs the enumeration would open
    n_baryon_pairs = 0
    masses = [math.sqrt(nd_limits(PROTON_MASS, PROTON_MASS, SQRT_S)[0])]
    masses += list(RECURSION_GRID)
    for event in pinned_trace[:200]:
        masses.append(event.central_mass)
    for mass in masses:
        root = CascadeState(
            momentum=(mass, 0.0, 0.0, 0.0), mass=mass, qn=(0, 0, 0),
            quantum={}, family_id="mesonic_neutral", axis=(0.0, 0.0, 1.0),
            alignment_label=ISOTROPIC_ALIGNMENT_LABEL, parent=None,
            exchange_ancestry=("nd",), depth=0)
        channels = enumerate_channels(root, catalog, fams)
        for channel in channels:
            daughter_sum = [0, 0, 0]
            pdgs = []
            for daughter_id in channel.daughters:
                if daughter_id.startswith("catalog:"):
                    record = catalog.get(int(daughter_id[len("catalog:"):]))
                    daughter_sum[0] += int(record.B)
                    daughter_sum[1] += int(record.Q)
                    daughter_sum[2] += int(record.S)
                    pdgs.append(int(record.pdg_id))
                else:
                    family = next(f for f in fams
                                  if f.family_id == daughter_id)
                    daughter_sum[0] += int(family.qn[0])
                    daughter_sum[1] += int(family.qn[1])
                    daughter_sum[2] += int(family.qn[2])
            assert tuple(daughter_sum) == (0, 0, 0), (
                channel.daughters, tuple(daughter_sum), mass)
            if channel.kind != "continuum":
                d0 = int(channel.daughters[0][len("catalog:"):])
                d1 = int(channel.daughters[1][len("catalog:"):])
                r0, r1 = catalog.get(d0), catalog.get(d1)
                threshold = float(r0.pole_mass) + float(r1.pole_mass)
                assert mass >= threshold - 1e-12, (mass, threshold)
                if (r0.B, r1.B) in ((1, -1), (-1, 1)):
                    n_baryon_pairs += 1
    # Round 6 (task 43, 2026-09-28): the (0, 0, 0) surfaces carry the
    # net-zero baryon-pair classes p pbar and n nbar above their
    # thresholds; every such channel conserves B exactly (checked above),
    # so the clause's threshold binding is live and the pairs are counted
    assert n_baryon_pairs > 0
    for record in catalog.records():
        if (record.B, record.Q, record.S) == (0, 0, 0):
            for channel in record.decay_channels:
                if channel.is_three_body or channel.is_radiative:
                    continue
                d0, d1 = channel.daughters
                b0 = catalog.get(int(d0)).B
                b1 = catalog.get(int(d1)).B
                assert not ((b0, b1) == (1, -1) or (b0, b1) == (-1, 1)), (
                    record.name, channel.daughters)

    # (3) the K K-bar direct channels of plan line 855: the family's
    # (321, -321) and (310, 130) classes are open at a mass above the
    # two-kaon threshold, the same pair classes are recorded channels of
    # the committed (0, 0, 0) records (f2(1270), rho3(1690)0, f4(2050),
    # phi(1020)), and the direct channels occur on the pinned event
    # trace's central line
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
    for record in catalog.records():
        if (record.B, record.Q, record.S) == (0, 0, 0):
            pairs = {tuple(c.daughters) for c in record.decay_channels
                     if not c.is_three_body}
            if record.pdg_id in (225, 117, 229, 333):
                assert (321, -321) in pairs or (310, 130) in pairs, \
                    record.name
    n_kkbar = 0
    for event in pinned_trace:
        pdgs = [h.pdg_id for h in event.final_state
                if h.line == CENTRAL_LINE_LABEL]
        if (321 in pdgs and -321 in pdgs) or (310 in pdgs and 130 in pdgs):
            n_kkbar += 1
    assert n_kkbar > 0, n_kkbar

    # (4) the phi route of plan line 855: the committed catalog record
    # phi(1020) carries the recorded K+ K- and K0S K0L channels with net
    # strangeness 0
    phi = catalog.get(333)
    assert phi.name == "phi(1020)"
    phi_pairs = {tuple(c.daughters) for c in phi.decay_channels
                 if not c.is_three_body}
    assert (321, -321) in phi_pairs
    assert (310, 130) in phi_pairs

    # (5) the K* route of plan line 855: the K*(892) classes are
    # recorded on the declared strange mesonic families'
    # allowed-decay classes; no committed (0, 0, 0) record carries a
    # K*0 K-bar*0 channel, so the presence is asserted at the committed
    # declared-family surface (the route enters the live enumeration
    # when the substrate declares it)
    continuum = json.loads(CONTINUUM_PATH.read_text(encoding="utf-8"))
    strange_classes: set[str] = set()
    for family in continuum["families"]:
        if tuple(family["qn"]) in ((0, 0, -1), (0, 0, 1)):
            strange_classes.update(family["allowed_decay_classes"])
    assert "(313,111)" in strange_classes
    assert "(-313,111)" in strange_classes
    assert "(323,-211)" in strange_classes
    assert "(-323,211)" in strange_classes
    for record in catalog.records():
        if (record.B, record.Q, record.S) == (0, 0, 0):
            for channel in record.decay_channels:
                assert 313 not in channel.daughters and \
                    -313 not in channel.daughters, record.name


# ----------------------------------------------------------------------
# test_strangeness_locality
# ----------------------------------------------------------------------
def test_strangeness_locality(catalog, fams, pinned_trace):
    """On the pinned trace, associated strangeness (channels with
    ``S != 0``, e.g. ``K Lambda``) occurs only in the baryonic
    excitation decays (the ``(1, 1, 0)`` parents: a hyperon appears
    only on a baryonic line); the net strangeness of the central source
    and of every central decay vertex is 0 (the ``phi -> K+ K-``
    channel is legal centrally with net ``S = 0``: the central line's
    ``S != 0`` hadrons are the net-neutral kaon pairs)."""

    n_hyperon_baryonic = 0
    n_strange_central = 0
    for event in pinned_trace:
        # the central source and every central decay vertex carry net
        # strangeness 0 exactly
        assert event.central_source.qn[2] == 0
        central = [h for h in event.final_state
                   if h.line == CENTRAL_LINE_LABEL]
        assert sum(h.qn[2] for h in central) == 0
        # Round 6 (task 43, 2026-09-28): the central line's net baryon
        # number stays 0 exactly; a central hyperon may now appear through
        # a resonance-mediated baryon-pair channel (the baryonic continuum
        # daughter of a p pbar class decaying to K Lambda), so the
        # per-hadron "no baryon number" clause is replaced by the net
        # closure of the (0, 0, 0) vertices
        assert sum(h.qn[0] for h in central) == 0
        for hadron in central:
            if hadron.qn[2] != 0:
                n_strange_central += 1
        for hadron in event.final_state:
            if hadron.line == CENTRAL_LINE_LABEL:
                continue
            if hadron.qn[2] != 0 and hadron.qn[0] != 0:
                # a hyperon: associated strangeness on a baryonic line,
                # a (1, 1, 0) parent's decay (the K Lambda / K Sigma
                # catalog channels of the N(1650)+ and N(1720)+
                # records)
                n_hyperon_baryonic += 1
    # both behaviors occur on the pinned trace: the associated
    # strangeness appears in the baryonic decays and never carries a
    # baryonic line's hyperon centrally
    assert n_hyperon_baryonic > 0, n_hyperon_baryonic

    # the phi -> K+ K- channel is a committed (0, 0, 0) record channel
    # with net strangeness 0 (legal centrally)
    phi = catalog.get(333)
    for channel in phi.decay_channels:
        net_s = sum(quantum_numbers(int(d)).S for d in channel.daughters)
        assert net_s == 0, (phi.name, channel.daughters)

    # the strangeness-carrying baryonic channels are (1, 1, 0) catalog
    # channels: every low-mass record's S != 0 channel sums to (1, 1, 0)
    for record in catalog.records():
        if (record.B, record.Q, record.S) != (1, 1, 0):
            continue
        for channel in record.decay_channels:
            if channel.is_three_body or channel.is_radiative:
                continue
            net = (sum(catalog.get(int(d)).B for d in channel.daughters),
                   sum(catalog.get(int(d)).Q for d in channel.daughters),
                   sum(catalog.get(int(d)).S for d in channel.daughters))
            if any(catalog.get(int(d)).S != 0 for d in channel.daughters):
                assert net == (1, 1, 0), (record.name, channel.daughters)


# ----------------------------------------------------------------------
# test_phase_space
# ----------------------------------------------------------------------
def test_phase_space(catalog, fams, pinned_trace):
    """``M_c^2`` in ``[m_pi0^2, (sqrt(s) - M_1 - M_2)^2]`` and the
    three-system closure ``M_1 + M_2 + M_c <= sqrt(s)`` within ``1e-12``
    on the pinned trace; ``t_c = t_1 + t_2`` exactly; the drawn ``t``
    ranges exact; the solve's closure and limits."""

    for event in pinned_trace:
        m1 = event.remnants[0].mass
        m2 = event.remnants[1].mass
        m2_min, m2_max = nd_limits(m1, m2, SQRT_S)
        mc2 = event.central_mass ** 2
        assert m2_min <= mc2 <= m2_max, (event.central_mass, m2_min, m2_max)
        assert m1 + m2 + event.central_mass <= SQRT_S + 1e-12
        assert event.t_c == event.t1 + event.t2
        t1_min = t_range(m1 * m1, PROTON_MASS, SQRT_S)[0]
        t2_min = t_range(m2 * m2, PROTON_MASS, SQRT_S)[0]
        assert t1_min <= event.t1 <= 0.0, (event.t1, t1_min)
        assert t2_min <= event.t2 <= 0.0, (event.t2, t2_min)

    # the exact central limits of the intact-pair configuration (card
    # 4's recorded values at the recorded SQRT_S)
    assert nd_limits(PROTON_MASS, PROTON_MASS, SQRT_S) == (
        M2_MIN_PINNED, M2_MAX_PINNED)
    assert M2_MIN_PINNED == PI0_MASS * PI0_MASS

    # the exact three-system solve closes the incoming sum within 1e-9
    # relative at a generic point
    system1, system2, central = nd_kinematics(
        2.0, -1.0, 3.0, -0.5, 1.0, SQRT_S)
    total = tuple(system1[i] + system2[i] + central[i] for i in range(4))
    reference = _incoming_sum()
    relative = max(
        abs(total[0] - reference[0]) / reference[0],
        abs(total[1]) / reference[3],
        abs(total[2]) / reference[3],
        abs(total[3] - reference[3]) / reference[3])
    assert relative <= 1e-9, relative

    # the drawn-closure at-rest resolution: a triplet at the exact
    # closure edge resolves at the at-rest configuration (all three
    # systems comoving at the CM velocity); the solve runs on the exact
    # invariant mass of the pinned configuration (the caller's sqrt_s
    # argument is the draw-domain bookkeeping value, the committed
    # card-5/card-6 convention)
    _e_beam, _e_total, sq_exact, _p_star, beta, gamma = _beam_configuration()
    m_c_edge = sq_exact - 2.0 * PROTON_MASS + 5e-13
    s1, s2, sc = _cm_triplet(PROTON_MASS, 0.0, PROTON_MASS,
                             m_c_edge, sq_exact)
    for momentum, mass in ((s1, PROTON_MASS), (s2, PROTON_MASS),
                           (sc, m_c_edge)):
        # the at-rest configuration in the overall CM (the central
        # frame; the lab boost is the caller's assembly step)
        assert momentum[1] == 0.0 and momentum[2] == 0.0
        assert momentum[3] == 0.0
        assert abs(momentum[0] - mass) <= 1e-12, momentum

    # the vanishing-central-mass forward configuration at t = 0: the
    # projectile system approaches the beam four-vector (the
    # elastic-forward limit of the solve; the residual is the central
    # system's own lab energy gamma * M_c)
    s1, s2, sc = nd_kinematics(PROTON_MASS, 0.0, PROTON_MASS, 0.0, 1e-6,
                               SQRT_S)
    beam = (math.sqrt(PROTON_MASS ** 2 + BEAM_MOMENTUM ** 2),
            0.0, 0.0, BEAM_MOMENTUM)
    assert max(abs(a - b) for a, b in zip(s1, beam)) <= 1e-4, s1
    assert abs(sc[0] - 1e-6 / math.sqrt(1.0 - beta * beta)) <= 1e-9

    # the drawn-domain refusals: a positive transfer, non-positive
    # masses, and a closed three-system configuration are refused
    # (checked, never repaired)
    with pytest.raises(ValueError):
        nd_kinematics(2.0, -1.0, 3.0, 0.5, 1.0, SQRT_S)
    with pytest.raises(ValueError):
        nd_kinematics(-2.0, -1.0, 3.0, -0.5, 1.0, SQRT_S)
    with pytest.raises(ValueError):
        nd_kinematics(2.0, -1.0, 3.0, -0.5, -1.0, SQRT_S)
    with pytest.raises(ValueError):
        nd_kinematics(17.0, -1.0, 1.0, -0.5, 0.5, SQRT_S)
    with pytest.raises(ValueError):
        nd_limits(PROTON_MASS, PROTON_MASS, 2.0)
    with pytest.raises(ValueError):
        nd_limits(0.0, PROTON_MASS, SQRT_S)
    with pytest.raises(ValueError):
        nd_limits(PROTON_MASS, PROTON_MASS, float("nan"))


# ----------------------------------------------------------------------
# test_draw_budget
# ----------------------------------------------------------------------
def test_draw_budget(catalog, fams):
    """A pinned 1,000-event trace consumes exactly the pre-registered
    counts on the closed labels: 2 ``spectral_state`` uniforms per event
    (one remnant selection per baryonic line) and the
    ``exchange_kinematics`` budget of the card-4 decomposition (the
    ``t_1`` draw, the ``t_2`` draw, and the two-uniform ``M_c^2``
    central draw: 4 uniforms per event; the card's "5" literal is the
    recorded operative resolution), plus the frozen Round 4 per-vertex
    budget on ``decay_channel`` / ``decay_kinematics`` (pinned RNG-state
    identity; the ``topology_selection`` draw belongs to the runner,
    card 9)."""

    for label in ("spectral_state", "exchange_kinematics",
                  "decay_channel", "decay_kinematics"):
        assert label in SUBSTREAM_LABELS
    assert "topology_selection" in SUBSTREAM_LABELS

    # one shared stream per label across the pinned 1,000-event trace,
    # wrapped in the counting proxies
    subs = derive_substreams(MASTER_SEED, 0, 0)
    counter_spectral = _CountingRng(np.random.default_rng(
        subs.stream_for("spectral_state")))
    counter_exchange = _CountingRng(np.random.default_rng(
        subs.stream_for("exchange_kinematics")))
    counter_channel = _CountingRng(np.random.default_rng(
        subs.stream_for("decay_channel")))
    counter_kinematics = _CountingRng(np.random.default_rng(
        subs.stream_for("decay_kinematics")))

    per_event_channel: list[int] = []
    per_event_kinematics: list[int] = []
    for _event_index in range(N_BUDGET):
        before_channel = counter_channel.calls
        before_kinematics = counter_kinematics.calls
        generate_nd(NOMINAL_PARAMS, NOMINAL_ND_REMNANT_WEIGHTS, catalog,
                    fams, counter_spectral, counter_exchange,
                    counter_channel, counter_kinematics)
        per_event_channel.append(counter_channel.calls - before_channel)
        per_event_kinematics.append(
            counter_kinematics.calls - before_kinematics)

    # the kinematic labels: exactly the pre-registered counts per event
    assert counter_spectral.calls == 2 * N_BUDGET
    assert counter_exchange.calls == 4 * N_BUDGET

    # the frozen Round 4 per-vertex budget on the decay labels (the
    # pinned per-event totals at this landing; a catalog handoff and
    # the unstable-sibling kernels add the Round 2 kernel's own
    # documented draws on the same streams; a stable remnant record and
    # a below-threshold central source consume zero)
    assert sum(per_event_channel) == ND_CHANNEL_TOTAL
    assert sum(per_event_kinematics) == ND_KINEMATICS_TOTAL
    assert all(k >= 2 * c
               for c, k in zip(per_event_channel, per_event_kinematics))
    assert sum(1 for c in per_event_channel if c == 0) == ND_ZERO_DRAW_EVENTS

    # pinned RNG-state identity: the post-trace states equal fresh
    # streams advanced by exactly the counted draws
    subs = derive_substreams(MASTER_SEED, 0, 0)
    reference_spectral = np.random.default_rng(
        subs.stream_for("spectral_state"))
    reference_exchange = np.random.default_rng(
        subs.stream_for("exchange_kinematics"))
    reference_channel = np.random.default_rng(subs.stream_for("decay_channel"))
    reference_kinematics = np.random.default_rng(
        subs.stream_for("decay_kinematics"))
    for _ in range(2 * N_BUDGET):
        reference_spectral.random()
    for _ in range(4 * N_BUDGET):
        reference_exchange.random()
    for _ in range(counter_channel.calls):
        reference_channel.random()
    for _ in range(counter_kinematics.calls):
        reference_kinematics.random()
    assert (counter_spectral._source.bit_generator.state
            == reference_spectral.bit_generator.state)
    assert (counter_exchange._source.bit_generator.state
            == reference_exchange.bit_generator.state)
    assert (counter_channel._source.bit_generator.state
            == reference_channel.bit_generator.state)
    assert (counter_kinematics._source.bit_generator.state
            == reference_kinematics.bit_generator.state)


# ----------------------------------------------------------------------
# test_frozen_cascade
# ----------------------------------------------------------------------
def test_frozen_cascade(catalog, fams):
    """All three systems decay through the frozen simplified cascade
    (kappa_B = 0.0, kappa_M = 4.0, b_dec = 2.0 GeV^-2, baryonic nominal
    width 0.3 GeV, ``W_flavor = 1.0``); a white-box check that no module
    in this card references a re-fit of the alignment parameters (the
    R46 ``no_refit`` clause) or claims the separation of exchange recoil
    from decay transverse dynamics."""

    # the frozen simplified cascade is consumed read-only: the module
    # imports the sealed Round 4 machinery (the constant-kappa alignment
    # kernels, the decay form factor, the termination driver) and
    # introduces no alignment re-fit and no separation claim (the R46
    # clauses; the white-box token scan).  The sealed R45 point
    # (kappa_B = 0.0, kappa_M = 4.0, b_dec = 2.0 GeV^-2, baryonic
    # nominal width 0.3 GeV) is the R45 driver's frozen configuration,
    # pinned through the R45 driver digest in FROZEN_SURFACES.
    import na49_mixing.cascade_alignment  # noqa: F401
    import na49_mixing.cascade_form_factor  # noqa: F401
    import na49_mixing.cascade_termination  # noqa: F401
    from na49_mixing.cascade_decay import CascadeChannel, channel_weight
    # the W_flavor hook default is exactly 1.0 (Round 6 owns the flavor
    # weights); the hook is multiplicative and consumes zero draws
    probe_channel = CascadeChannel(
        parent_id="baryonic_np", daughters=("catalog:2212", "catalog:111"),
        kind="established_hadron", weight=0.5)
    assert channel_weight(probe_channel) == 0.5
    assert channel_weight(probe_channel, w_flavor=1.0) == 0.5

    # the central source decays through the activated neutral-mesonic
    # (0, 0, 0) continuum family (the cascade line's family id) and the
    # baryonic systems through the established-record routing
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert 'CENTRAL_FAMILY_ID: str = "mesonic_neutral"' in source
    assert "handoff_decay" in source and "draw_decay" in source
    for token in NO_REFIT_TOKENS:
        assert token not in source, token

    # one event's three decays close locally through the ledger's local
    # decay closure and the central source stays neutral
    subs = derive_substreams(MASTER_SEED, 0, 0)
    rngs = [np.random.default_rng(subs.stream_for(label)) for label in
            ("spectral_state", "exchange_kinematics", "decay_channel",
             "decay_kinematics")]
    event = generate_nd(NOMINAL_PARAMS, NOMINAL_ND_REMNANT_WEIGHTS,
                        catalog, fams, *rngs)
    report = _audit(event)
    assert not report.violations
    check_central_neutrality(event)


# ----------------------------------------------------------------------
# test_layout_sync
# ----------------------------------------------------------------------
def test_layout_sync():
    """The test register (the card's frozen-substrate provenance pins,
    verified on disk: the builder, the card-6 routing, the card-2
    engine, the committed catalog, the continuum families
    configuration, the sealed R45/R46 drivers in both helper trees, and
    the five committed Round 5 modules consumed read-only; the
    M8-frozen ten-parameter set hash and the kernel boundary through
    the manifest; the activated continuum families with the
    neutral-mesonic ``(0, 0, 0)`` family's recorded support ends; the
    freeze re-pin of this card), and the layout sync (the subprocess
    check from the monorepo root; ``logic_drift = 0``)."""

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

    # the activated continuum families of the committed configuration:
    # exactly three (the two baryonic families and the neutral mesonic
    # family), with the neutral-mesonic family's recorded support ends
    # at the frozen boundary and the card-6 surface's upper end
    continuum = json.loads(CONTINUUM_PATH.read_text(encoding="utf-8"))
    active = [family for family in continuum["families"]
              if family["active"] is True]
    assert sorted(f["family_id"] for f in active) == [
        "baryonic_np", "baryonic_strange", "mesonic_neutral"]
    neutral = [family for family in continuum["families"]
               if family["qn"] == [0, 0, 0] and family["active"] is True]
    assert len(neutral) == 1
    assert neutral[0]["family_id"] == "mesonic_neutral"
    assert neutral[0]["m_min"] == KERNEL_BOUNDARY
    assert neutral[0]["m_max"] == 5.0

    # the freeze re-pin of this card: the strict path-sorted inventory
    # at the operative count (520 at the card-6 landing plus the card's
    # two new files), this card's freeze exception exactly once with the
    # path-sorted file list and the reason naming the 520 -> 522 re-pin
    # and the card's pre-registered literal
    inventory = manifest["source_tree_sha256"]
    keys = list(inventory.keys())
    assert keys == sorted(keys), "the inventory is not path-sorted"
    assert len(keys) == INVENTORY_COUNT
    module_rel = "dev/mixing2006/na49_mixing/topology_non_diffractive.py"
    test_rel = "dev/mixing2006/tests/test_topology_non_diffractive.py"
    assert inventory[module_rel] == _sha256_file(MODULE_PATH)
    assert inventory[test_rel] == _sha256_file(TEST_FILE)
    for rel in CHAIN_COUNT_PIN_FILES:
        assert rel in keys, f"chain file missing from the inventory: {rel}"
        assert inventory[rel] == _sha256_file(MONO_ROOT / rel), (
            f"chain file digest drifted: {rel}")
    exceptions = manifest["freeze_exceptions"]
    entries = [e for e in exceptions if e.get("card") == FREEZE_CARD]
    assert len(entries) == 1, (
        "the card-7 freeze exception must name its card exactly once")
    files = entries[0]["files"]
    assert files == sorted(files), (
        "the freeze-exception files are path-sorted")
    assert files == list(FREEZE_EXCEPTION_FILES), (
        "the freeze-exception file list drifted")
    reason = entries[0]["reason"]
    assert reason.startswith("Round 5, card 7 ("), (
        "the freeze-exception reason must name its card")
    assert FREEZE_CARD in reason, (
        "the freeze-exception reason must name its card")
    assert "520 -> 522" in reason, (
        "the freeze-exception reason must name the 520 -> 522 re-pin")
    assert "511 -> 513" in reason, (
        "the freeze-exception reason must record the card's "
        "pre-registered literal")


    probe = subprocess.run(
        [sys.executable, str(SYNC_CHECKER)],
        capture_output=True, text=True, cwd=str(MONO_ROOT))
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert "logic_drift=0" in probe.stdout


# ----------------------------------------------------------------------
# Closed-set guards (the card's record surface and nominal parameters)
# ----------------------------------------------------------------------
def test_closed_sets():
    """The frozen record surface, the class label, the closed
    line-label set, the central quantum numbers, and the nominal
    parameters."""

    assert ND_CLASS_LABEL == "nd"
    assert isinstance(ND_LINE_LABELS, frozenset)
    assert ND_LINE_LABELS == frozenset({"projectile", "target", "central"})
    assert CENTRAL_QN == (0, 0, 0, 0)
    assert CENTRAL_LINE_LABEL == "central"
    assert getattr(NDEvent, "__dataclass_params__").frozen
    assert tuple(NDEvent.__dataclass_fields__) == (
        "class_label", "remnants", "central_mass", "t_c", "t1", "t2",
        "lines", "central_source", "exchanges", "final_state",
        "remnant_windows", "central_component",
        "exchange_configuration", "slope_factor", "duality_edges")  # remnant_windows: 2026-09-29, default nan pair
    # slope_factor: 2026-10-08, None = the single transfer slope (transfer_slope_mass_exponent off)
    # duality_edges: 2026-10-12, None = the continuum lines (the duality option off)
    # central_component: 2026-10-02, default True (the card's event)
    assert NDEvent.__dataclass_fields__["central_component"].default is True
    # exchange_configuration: Round 8 2026-10-04, default None (the vacuum exchanges)
    assert NDEvent.__dataclass_fields__["exchange_configuration"].default is None
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
