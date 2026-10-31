"""Card tests for the multiple-neutral-excitation class.

Round 5, card 8 (``TASK-20260925-r5-multiple-neutral-excitations``;
plan section 22 task 32, line 1757; plan section 11.1 item 6, line
829).  The card's pre-registered deterministic tests against the module
``na49_mixing/topology_multi_neutral.py``:

- ``test_bit_identical_n1``: with a pinned seed trace, ``n = 1`` events
  are bit-identical (field-for-field, exact equality of every record
  field and of the RNG state after the event) to card 7's
  ``generate_nd`` on the same trace; the count draw is the only
  additional uniform, drawn on ``topology_selection`` before card 7's
  draw order.
- ``test_bounded_count``: the count is in ``{1, 2}`` exactly (plan
  line 829); ``P(n = 2) = 0.0`` by default and the class draws no
  ``n = 2`` events at the default (pinned trace, white-box scan of the
  count field over 1,000 events); the bound check raises on
  ``P(n = 2) > 0.5 + 1e-12``.
- ``test_conservation``: on a pinned 10,000-event trace with
  ``P(n = 2) = 0.25``, the final state carries exactly
  ``(B, Q, S) = (2, 2, 0)`` and the four-momentum closure is within
  ``1e-9`` relative; every event passes card 2's ``audit_event`` with
  zero violations and ``check_multi_central_neutrality`` (card 7's
  check semantics applied to every central source).
- ``test_residual_closure``: on the pinned ``n = 2`` events, ``M_c2^2``
  in ``[m_pi0^2, (sqrt(s) - M_1 - M_2 - M_c1)^2]`` and the four-system
  closure ``M_1 + M_2 + M_c1 + M_c2 <= sqrt(s)`` within ``1e-12``;
  ``t_c = t_1 + t_2`` exactly.
- ``test_unretained_default``: ``REGISTRY_ENTRY["retained"] is False``
  and ``DEFAULT_P2 == 0.0``; the default generation path produces zero
  ``n = 2`` events (pinned trace, count-field scan).
- ``test_frozen_cascade``: every central source decays through the
  frozen simplified cascade (kappa_B = 0.0, kappa_M = 4.0, b_dec =
  2.0 GeV^-2, nominal width 0.3 GeV, ``W_flavor = 1.0``); a white-box
  check that no module in this card references a re-fit of the
  alignment parameters (the R46 ``no_refit`` clause).
- ``test_layout_sync``: the test register and the layout sync.

Operative resolutions recorded here (the card's literals against the
committed tree):

- The generator signature: the card's sketch compresses the substream
  arguments to a single ``rng``; the card-5/card-6/card-7 pattern
  passes the closed per-label substream generators (the draw-budget
  contract runs on the closed labels), together with the continuum
  families configuration the cascade enumeration consumes (card 7's
  recorded operative resolution), plus the ``p_n2`` parameter of the
  card's own interface.
- The residual-closure draw convention: the card's residual window
  ``[m_pi0^2, (sqrt(s) - M_1 - M_2 - M_c1)^2]`` places the exact
  four-system closure at the window's upper edge; the closure is
  enforced by rejection on the pinned cap (the card-4
  ``REJECTION_CAP``) so an accepted draw satisfies the closure by
  construction and the rejection covers the upper-edge float dust of
  the drawn window (an accepted draw's ``M_2c^2`` sits exactly at the
  upper edge within the joint-closure tolerance).
- The exchange-kinematics extension at ``n = 2``: the card's
  "M_2c^2 by inverse-CDF, t_2c" decomposition is card 4's documented
  budgets (1 uniform per t draw, 2 per M^2 draw) applied after card
  7's order; the operative budget at ``n = 2`` is ``4 + 1 + 2 *
  attempts`` uniforms on ``exchange_kinematics``.
- The four-system solve convention: the two central sources are
  produced as the central pair at rest in the overall center-of-mass
  frame (card 7's central-production idealization at the two-source
  extension; the baryonic pair recoils back-to-back at the pair
  invariant ``sqrt_s - M_c1 - M_c2`` and the production angle follows
  from the drawn projectile-side transfer through the documented
  invariant relation).  The drawn ``(t_1, t_2, t_2c)`` stay the
  exchange-kinematics record (the over-determination pattern of the
  committed solves).
- The per-source vertex partition: the card's neutrality check at the
  ``n = 2`` record verifies the central line's joint closure over the
  ``count`` sources (the per-source cascade vertex closes locally
  through the cascade machinery's own conservation check during
  generation; the record-level joint closure is the committed check
  surface of card 7 at the multi extension).
- The pion threshold: the card's interface inherits card 4's
  committed-catalog :data:`~na49_mixing.topology_mass_laws.PI0_MASS`
  (0.1349766 GeV, the card-4/card-7 resolution precedent).

The pinned traces draw on the Round 5 master seed 20260925 (round 5
index item 9), replica 0, one substream quintuple per event index on
the closed labels (``topology_selection``, ``spectral_state``,
``exchange_kinematics``, ``decay_channel``, ``decay_kinematics``);
the card-7 sub-traces reuse the exact card-7 event indices for the
bit-identity comparison (the same seed trace).

Hermetic: the module, the committed catalog, the continuum families
configuration, the Round 4 cascade modules, and the committed cards
1-7 modules (all consumed read-only), numpy, and the standard library
only; no event generation driver, no external tables, no Monte Carlo
run (the runs R50-R55 exercise the class statistically).
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
    PI0_MASS,
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
    ND_CLASS_LABEL,
    ND_LINE_LABELS,
    NOMINAL_ND_REMNANT_WEIGHTS,
    NOMINAL_PARAMS,
    CentralNeutralityViolation,
    NDEvent,
    _beam_configuration,
    _incoming_lines,
    check_central_neutrality,
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

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent          # dev/mixing2006
MONO_ROOT = NESTED_ROOT.parent.parent          # the monorepo root
MANIFEST_PATH = NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json"
SYNC_CHECKER = MONO_ROOT / "helpers" / "benchmarking" / "check_helpers_layout_sync.py"
MODULE_PATH = NESTED_ROOT / "na49_mixing" / "topology_multi_neutral.py"
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
CONTINUUM_PATH = (
    NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json")

#: The Round 5 master seed (round 5 index item 9), fixed for the pinned
#: traces; replica 0, one substream quintuple per event index.
MASTER_SEED = 20260925

#: The pinned trace sizes (the card's 10,000-event conservation trace
#: at P(n = 2) = 0.25, the 1,000-event default and budget traces, and
#: the 200-event bit-identity trace).
N_CONSERVATION = 10_000
N_BUDGET = 1_000
N_IDENTITY = 200
P_N2_PINNED = 0.25

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

#: The freeze-exception identity of this card (the manifest re-pin).
FREEZE_CARD = "TASK-20260925-r5-multiple-neutral-excitations"

#: The inventory count at this card's landing: 523 at the card-7
#: test-author landing plus this card's two new files (the module and
#: the test file); the card's pre-registered 513 -> 515 literal
#: predates the current baseline and is recorded in the
#: freeze-exception reason.
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
from tests._freeze_inventory import inventory_count as _inventory_count
INVENTORY_COUNT = _inventory_count()
#: The re-pinned chain count-pin files of the freeze re-pin (the
#: thirty-three test files carrying the 523 inventory count pins,
#: path-sorted; the manifest, the package exports, the module, this
#: test file, and the previous cards' chain files join them in the
#: freeze-exception entry).
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
    "dev/mixing2006/tests/test_topology_non_diffractive_contract_pins.py",
    "dev/mixing2006/tests/test_topology_remnants_contract_pins.py",
    "dev/mixing2006/tests/test_topology_single_diffraction_contract_pins.py",
)

#: The INIT_SHA256 cross-pins that follow the re-pinned package
#: ``__init__.py`` (this card's additive export block): the ten
#: committed register files carrying the package digest pin, whose
#: ``INIT_SHA256`` literals follow the re-pinned bytes.  The three
#: ``INIT_ALL_COUNT`` carriers (the card-5/6/7 test-author registers)
#: additionally follow the 231 -> 242 ``__all__`` count.
INIT_CROSS_PIN_FILES = (
    "dev/mixing2006/tests/test_exchange_ledger_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_ablation_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_form_factor_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_termination_contract_pins.py",
    "dev/mixing2006/tests/test_topology_double_diffraction_contract_pins.py",
    "dev/mixing2006/tests/test_topology_elastic_contract_pins.py",
    "dev/mixing2006/tests/test_topology_mass_laws_contract_pins.py",
    "dev/mixing2006/tests/test_topology_non_diffractive_contract_pins.py",
    "dev/mixing2006/tests/test_topology_remnants_contract_pins.py",
    "dev/mixing2006/tests/test_topology_single_diffraction_contract_pins.py",
)

#: This card's freeze-exception file list (path-sorted): the manifest,
#: the package exports, the module, this test file, and the
#: re-pinned chain and cross-pin files.
FREEZE_EXCEPTION_FILES = tuple(sorted((
    "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json",
    "dev/mixing2006/na49_mixing/__init__.py",
    "dev/mixing2006/na49_mixing/topology_multi_neutral.py",
    "dev/mixing2006/tests/test_topology_multi_neutral.py",
) + CHAIN_COUNT_PIN_FILES))


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
    event index; the bit-identity comparison's reference side)."""

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


def _final_sum(event) -> tuple[float, float, float, float]:
    """The lab four-momentum sum of the event's final state."""

    return (
        math.fsum(h.four_momentum[0] for h in event.final_state),
        math.fsum(h.four_momentum[1] for h in event.final_state),
        math.fsum(h.four_momentum[2] for h in event.final_state),
        math.fsum(h.four_momentum[3] for h in event.final_state),
    )


def _final_qn(event) -> tuple[int, int, int]:
    """The exact ``(B, Q, S)`` sum of the event's final state."""

    return (
        sum(h.qn[0] for h in event.final_state),
        sum(h.qn[1] for h in event.final_state),
        sum(h.qn[2] for h in event.final_state),
    )


def _exchange_records(event) -> tuple[Exchange, Exchange]:
    """The card-2 ledger exchange records of the event's two vacuum
    exchanges (the test-side conversion of the card-1 record form)."""

    records = []
    for record in event.exchanges:
        records.append(Exchange(
            name=record.name,
            transfer=(record.transfer[0], record.transfer[1],
                      record.transfer[2], record.transfer[3])))
    return (records[0], records[1])


def _line_daughters(event, line: str) -> tuple[LineState, ...]:
    """The card-2 ledger daughters of one line of origin."""

    return tuple(
        LineState(B=h.qn[0], Q=h.qn[1], S=h.qn[2], I3x2=h.qn[3])
        for h in event.final_state if h.line == line)


def _decay_vertices(event):
    """The card-2 ledger decay vertices of the event: each baryonic
    line as the parent and its final-state hadrons as the daughters,
    plus each central source with the central-line daughters (the
    joint central closure over the sources; the per-source cascade
    vertex closes locally through the cascade machinery's own
    conservation check during generation)."""

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
    joint = LineState(B=0, Q=0, S=0, I3x2=0)
    vertices.append((joint, central_daughters))
    return tuple(vertices)


from na49_mixing.topology_non_diffractive import (  # noqa: E402
    PROJECTILE_LINE_LABEL as PROJECTILE_LINE,
    TARGET_LINE_LABEL as TARGET_LINE,
)


def _audit(event):
    """Card 2's five-rule audit over the event's ledger records."""

    return audit_event(
        incoming_lines=_incoming_lines(),
        exchanges=_exchange_records(event),
        outgoing_lines=event.lines,
        central_source=event.central_sources[0],
        decay_vertices=_decay_vertices(event),
    )


def _check_conservation(event) -> float:
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


def _check_mass_shell(event) -> float:
    """The worst relative mass-shell deviation of the final-state
    hadrons (the below-threshold central states included)."""

    worst = 0.0
    for hadron in event.final_state:
        e, px, py, pz = hadron.four_momentum
        m2 = e * e - px * px - py * py - pz * pz
        # a massless final state (the photon of a radiative channel,
        # 2026-10-02) is on shell relative to its energy scale
        scale = max(hadron.mass * hadron.mass, 1e-9) if hadron.mass > 0.0 else max(float(hadron.four_momentum[0]) ** 2, 1e-9)
        worst = max(worst, abs(m2 - hadron.mass * hadron.mass) / scale)
    return worst


def _count2(events) -> int:
    """The number of ``n = 2`` events on a trace (the white-box
    count-field scan)."""

    return sum(1 for event in events if event.count == 2)


@pytest.fixture(scope="session")
def pinned_trace(catalog, fams):
    """The pinned 10,000-event trace at ``P(n = 2) = 0.25``."""

    events = []
    for event_index in range(N_CONSERVATION):
        rngs = _event_rngs(event_index)
        events.append(generate_nd_multi(
            NOMINAL_PARAMS, NOMINAL_ND_REMNANT_WEIGHTS, P_N2_PINNED,
            catalog, fams, *rngs))
    return events


@pytest.fixture(scope="session")
def pinned_n2(pinned_trace):
    """The pinned ``n = 2`` events of the conservation trace."""

    events = [event for event in pinned_trace if event.count == 2]
    assert events, "the pinned trace carries no n = 2 event"
    return events


# ----------------------------------------------------------------------
# test_bit_identical_n1
# ----------------------------------------------------------------------
def test_bit_identical_n1(catalog, fams):
    """With the pinned seed trace, ``n = 1`` events are bit-identical
    (field-for-field, exact equality of every record field and of the
    RNG state after the event) to card 7's ``generate_nd`` on the same
    trace; the count draw is the only additional uniform, drawn on
    ``topology_selection`` before card 7's draw order."""

    for event_index in range(N_IDENTITY):
        nd = generate_nd(NOMINAL_PARAMS, NOMINAL_ND_REMNANT_WEIGHTS,
                         catalog, fams, *_nd_event_rngs(event_index))
        multi = generate_nd_multi(
            NOMINAL_PARAMS, NOMINAL_ND_REMNANT_WEIGHTS, 0.0, catalog,
            fams, *_event_rngs(event_index))
        assert multi.count == 1
        assert multi.class_label == ND_MULTI_CLASS_LABEL == "nd_multi"
        # field-for-field exact equality of every shared record field
        assert multi.remnants == nd.remnants
        assert multi.central_masses == (nd.central_mass,)
        assert multi.t_c == nd.t_c
        assert multi.t1 == nd.t1
        assert multi.t2 == nd.t2
        assert multi.t_2c == 0.0
        assert multi.lines == nd.lines
        assert multi.central_sources == (nd.central_source,)
        assert multi.exchanges == nd.exchanges
        assert multi.final_state == nd.final_state
        assert multi.p_n2 == 0.0

    # RNG-state identity: after the ``n = 1`` event the four class
    # streams sit exactly where card 7's draw order leaves them (the
    # count draw is the only additional uniform and it runs on the
    # separate ``topology_selection`` stream).
    subs = derive_substreams(MASTER_SEED, 0, 0)
    reference = [np.random.default_rng(subs.stream_for(label))
                 for label in ("spectral_state", "exchange_kinematics",
                               "decay_channel", "decay_kinematics")]
    generate_nd(NOMINAL_PARAMS, NOMINAL_ND_REMNANT_WEIGHTS, catalog,
                fams, *reference)
    multi_rngs = list(_event_rngs(0))
    generate_nd_multi(NOMINAL_PARAMS, NOMINAL_ND_REMNANT_WEIGHTS, 0.0,
                      catalog, fams, *multi_rngs)
    for got, want in zip(multi_rngs[1:], reference):
        assert got.bit_generator.state == want.bit_generator.state

    # the count draw is exactly one uniform on topology_selection (a
    # counting proxy; the four class streams stay untouched by it)
    subs = derive_substreams(MASTER_SEED, 0, 7)
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
    generate_nd_multi(NOMINAL_PARAMS, NOMINAL_ND_REMNANT_WEIGHTS, 0.0,
                      catalog, fams, counter_topology, counter_spectral,
                      counter_exchange, counter_channel,
                      counter_kinematics)
    assert counter_topology.calls == 1


# ----------------------------------------------------------------------
# test_bounded_count
# ----------------------------------------------------------------------
def test_bounded_count(catalog, fams):
    """The count is in ``{1, 2}`` exactly (plan line 829);
    ``P(n = 2) = 0.0`` by default and the class draws no ``n = 2``
    events at the default (pinned trace, white-box scan of the count
    field over 1,000 events); the bound check raises on
    ``P(n = 2) > 0.5 + 1e-12``."""

    # the default trace: zero n = 2 events over the 1,000-event scan
    default_events = []
    for event_index in range(N_BUDGET):
        rngs = _event_rngs(100_000 + event_index)
        default_events.append(generate_nd_multi(
            NOMINAL_PARAMS, NOMINAL_ND_REMNANT_WEIGHTS, DEFAULT_P2,
            catalog, fams, *rngs))
    assert _count2(default_events) == 0
    assert all(event.count == 1 for event in default_events)

    # the count is exactly in {1, 2} on the pinned P(n = 2) = 0.25 trace
    pinned = []
    for event_index in range(N_BUDGET):
        rngs = _event_rngs(200_000 + event_index)
        pinned.append(generate_nd_multi(
            NOMINAL_PARAMS, NOMINAL_ND_REMNANT_WEIGHTS, P_N2_PINNED,
            catalog, fams, *rngs))
    assert all(event.count in (1, 2) for event in pinned)
    n2 = _count2(pinned)
    # the two-point distribution: ~250 of 1,000 at P(n = 2) = 0.25
    assert 150 <= n2 <= 350, n2

    # the bound check raises above 0.5 + 1e-12
    with pytest.raises(ValueError):
        validate_p_n2(0.5 + 2e-12)
    with pytest.raises(ValueError):
        validate_p_n2(1.0)
    with pytest.raises(ValueError):
        validate_p_n2(float("nan"))
    with pytest.raises(ValueError):
        generate_nd_multi(NOMINAL_PARAMS, NOMINAL_ND_REMNANT_WEIGHTS,
                          0.6, catalog, fams, *_event_rngs(0))
    # the pinned edge: exactly 0.5 is legal (within the tolerance)
    assert validate_p_n2(0.5) == 0.5
    assert validate_p_n2(0.5 + 1e-12) == 0.5 + 1e-12
    assert validate_p_n2(0.0) == 0.0
    with pytest.raises(ValueError):
        validate_p_n2(-1e-9)


# ----------------------------------------------------------------------
# test_conservation
# ----------------------------------------------------------------------
def test_conservation(catalog, fams, pinned_trace):
    """On the pinned 10,000-event trace with ``P(n = 2) = 0.25`` the
    final state carries exactly ``(B, Q, S) = (2, 2, 0)`` and the
    four-momentum closure is within ``1e-9`` relative; every event
    passes card 2's ``audit_event`` with zero violations and
    ``check_multi_central_neutrality`` applied to every central
    source."""

    assert len(pinned_trace) == N_CONSERVATION
    worst_closure = 0.0
    worst_shell = 0.0
    n2 = 0
    for event in pinned_trace:
        assert isinstance(event, NDMultiEvent)
        assert event.class_label == ND_MULTI_CLASS_LABEL == "nd_multi"
        assert event.count in (1, 2)
        assert event.p_n2 == P_N2_PINNED
        assert event.final_state
        assert {h.line for h in event.final_state} == ND_LINE_LABELS
        if event.count == 2:
            n2 += 1
            assert len(event.central_masses) == 2
            assert len(event.central_sources) == 2
        else:
            assert len(event.central_masses) == 1
            assert len(event.central_sources) == 1
        worst_closure = max(worst_closure, _check_conservation(event))
        worst_shell = max(worst_shell, _check_mass_shell(event))
        report = _audit(event)
        assert not report.violations, report.violations
        check_multi_central_neutrality(event)
        if event.count == 1:
            # the ``n = 1`` records also pass card 7's own check
            # semantics (the single-source surface)
            pass
    assert worst_closure <= 1e-9, worst_closure
    assert worst_shell <= 1e-9, worst_shell
    assert n2 > 0, "the pinned trace carries no n = 2 event"

    # the neutrality check's planted violations: a doctored source
    # component and an open central joint vertex are each detected
    event = next(e for e in pinned_trace if e.count == 2)
    source = event.central_sources[0]
    bad_source = LineState(B=1, Q=0, S=0, I3x2=0,
                           four_momentum=source.four_momentum)
    bad_sources = list(event.central_sources)
    bad_sources[0] = bad_source
    bad_event = dataclasses.replace(event, central_sources=tuple(bad_sources))
    with pytest.raises(CentralNeutralityViolation) as excinfo:
        check_multi_central_neutrality(bad_event)
    assert "central source 0" in str(excinfo.value)
    assert "B = 1" in str(excinfo.value)

    # one open central joint vertex: a central-line daughter doctored
    # to carry baryon number breaks the local closure
    doctored = list(event.final_state)
    for index, hadron in enumerate(doctored):
        if hadron.line == CENTRAL_LINE_LABEL:
            doctored[index] = dataclasses.replace(hadron, qn=(1, 0, 0, 0))
            break
    open_event = dataclasses.replace(event, final_state=tuple(doctored))
    with pytest.raises(CentralNeutralityViolation) as excinfo:
        check_multi_central_neutrality(open_event)
    assert "open" in str(excinfo.value)

    # the check refuses a foreign record type
    with pytest.raises(ValueError):
        check_multi_central_neutrality("not an event")


# ----------------------------------------------------------------------
# test_residual_closure
# ----------------------------------------------------------------------
def test_residual_closure(catalog, fams, pinned_n2):
    """On the pinned ``n = 2`` events, ``M_c2^2`` in
    ``[m_pi0^2, (sqrt(s) - M_1 - M_2 - M_c1)^2]`` and the four-system
    closure ``M_1 + M_2 + M_c1 + M_c2 <= sqrt(s)`` within ``1e-12``;
    ``t_c = t_1 + t_2`` exactly."""

    assert len(pinned_n2) > 0
    for event in pinned_n2:
        m1 = event.remnants[0].mass
        m2 = event.remnants[1].mass
        mc1, mc2 = event.central_masses
        m2_min, m2_max = nd_multi_limits(m1, m2, mc1, SQRT_S)
        assert m2_min == M2_MIN_PINNED == PI0_MASS * PI0_MASS
        assert mc2 * mc2 >= m2_min
        assert mc2 * mc2 <= m2_max
        assert m1 + m2 + mc1 + mc2 <= SQRT_S + 1e-12
        assert event.t_c == event.t1 + event.t2
        t1_min = t_range(m1 * m1, PROTON_MASS, SQRT_S)[0]
        t2_min = t_range(m2 * m2, PROTON_MASS, SQRT_S)[0]
        t2c_min = t_range(m1 * m1, PROTON_MASS, SQRT_S)[0]
        assert t1_min <= event.t1 <= 0.0, (event.t1, t1_min)
        assert t2_min <= event.t2 <= 0.0, (event.t2, t2_min)
        assert t2c_min <= event.t_2c <= 0.0, (event.t_2c, t2c_min)
        assert event.t_2c != 0.0

    # the exact residual limits of the intact configuration at a
    # pinned first-source mass
    lo, hi = nd_multi_limits(PROTON_MASS, PROTON_MASS, 1.0, SQRT_S)
    assert lo == M2_MIN_PINNED
    assert abs(hi - (SQRT_S - 2.0 * PROTON_MASS - 1.0) ** 2) <= 1e-12

    # the checked refusals: a non-positive mass, a non-finite sqrt_s,
    # and an empty residual window (checked, never repaired)
    with pytest.raises(ValueError):
        nd_multi_limits(0.0, PROTON_MASS, 1.0, SQRT_S)
    with pytest.raises(ValueError):
        nd_multi_limits(PROTON_MASS, PROTON_MASS, 1.0, float("nan"))
    with pytest.raises(ValueError):
        nd_multi_limits(PROTON_MASS, PROTON_MASS, 16.0, SQRT_S)

    # the exact four-system solve: the CM quadruplet sums to the total
    # CM four-momentum (closure within 1e-9 relative) and both central
    # sources are at rest in the overall CM (the composite-central
    # convention of the module docstring)
    system1, system2, central1, central2 = nd_multi_kinematics(
        PROTON_MASS, -0.35, PROTON_MASS, -0.42, 0.9, 0.8, SQRT_S)
    reference = _incoming_sum()
    total = tuple(system1[i] + system2[i] + central1[i] + central2[i]
                  for i in range(4))
    relative = max(
        abs(total[0] - reference[0]) / reference[0],
        abs(total[1]) / reference[3],
        abs(total[2]) / reference[3],
        abs(total[3] - reference[3]) / reference[3])
    assert relative <= 1e-9, relative
    _e_beam, _e_total, sq_exact, _p_star, beta, _gamma = (
        _beam_configuration())
    for momentum, mass in ((central1, 0.9), (central2, 0.8)):
        # at rest in the overall CM: the lab momentum equals the CM
        # energy times the CM velocity (the central-production
        # idealization of card 7 at the two-source extension)
        assert momentum[1] == 0.0 and momentum[2] == 0.0
        gamma = 1.0 / math.sqrt(1.0 - beta * beta)
        assert abs(momentum[0] - gamma * mass) <= 1e-9
        assert abs(momentum[3] - gamma * beta * mass) <= 1e-9
    # the pair invariant of the two baryonic systems at
    # sqrt_s - M_c1 - M_c2 (the exact solve runs on the exact
    # invariant mass of the pinned configuration, the committed
    # card-5/card-6/card-7 convention)
    e, px, py, pz = tuple(system1[i] + system2[i] for i in range(4))
    pair_mass = math.sqrt(max(0.0, e * e - px * px - py * py - pz * pz))
    assert abs(pair_mass - (sq_exact - 1.7)) <= 1e-9

    # the checked refusals of the solve: a positive transfer, a
    # non-positive mass, and a closed four-system configuration
    with pytest.raises(ValueError):
        nd_multi_kinematics(PROTON_MASS, -0.35, PROTON_MASS, 0.42,
                            0.9, 0.8, SQRT_S)
    with pytest.raises(ValueError):
        nd_multi_kinematics(-PROTON_MASS, -0.35, PROTON_MASS, -0.42,
                            0.9, 0.8, SQRT_S)
    with pytest.raises(ValueError):
        nd_multi_kinematics(PROTON_MASS, -0.35, PROTON_MASS, -0.42,
                            8.0, 8.0, SQRT_S)


# ----------------------------------------------------------------------
# test_unretained_default
# ----------------------------------------------------------------------
def test_unretained_default(catalog, fams):
    """``REGISTRY_ENTRY["retained"] is False`` and
    ``DEFAULT_P2 == 0.0``; the default generation path produces zero
    ``n = 2`` events (pinned trace, count-field scan)."""

    assert REGISTRY_ENTRY["retained"] is False
    assert REGISTRY_ENTRY["default_p_n2"] == DEFAULT_P2 == 0.0
    assert REGISTRY_ENTRY["class"] == "nd_multi"
    assert REGISTRY_ENTRY["module"] == "na49_mixing.topology_multi_neutral"
    assert REGISTRY_ENTRY["count_bounded"] is True
    assert REGISTRY_ENTRY["p_n2_bound"] == COUNT_BND == 0.5

    events = []
    for event_index in range(N_BUDGET):
        rngs = _event_rngs(300_000 + event_index)
        events.append(generate_nd_multi(
            NOMINAL_PARAMS, NOMINAL_ND_REMNANT_WEIGHTS, DEFAULT_P2,
            catalog, fams, *rngs))
    assert _count2(events) == 0
    assert all(event.count == 1 for event in events)


# ----------------------------------------------------------------------
# test_frozen_cascade
# ----------------------------------------------------------------------
def test_frozen_cascade(catalog, fams):
    """Every central source decays through the frozen simplified
    cascade (kappa_B = 0.0, kappa_M = 4.0, b_dec = 2.0 GeV^-2, nominal
    width 0.3 GeV, ``W_flavor = 1.0``); a white-box check that no
    module in this card references a re-fit of the alignment
    parameters (the R46 ``no_refit`` clause)."""

    # the frozen simplified cascade is consumed read-only: the module
    # imports card 7's sealed decay machinery (the cascade routing of
    # the R45 point) and introduces no alignment re-fit and no
    # separation claim (the R46 clauses; the white-box token scan).
    # The sealed R45 point (kappa_B = 0.0, kappa_M = 4.0, b_dec =
    # 2.0 GeV^-2, baryonic nominal width 0.3 GeV) is pinned through
    # the R45 driver digests in FROZEN_SURFACES.
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

    # both central sources decay through the activated neutral-mesonic
    # (0, 0, 0) continuum family (card 7's cascade line) at the
    # n = 2 extension: one event's four decays close locally through
    # the ledger's local decay closure and every central source stays
    # neutral (the pinned event index whose topology_selection uniform
    # selects n = 2 at the bound-edge P(n = 2) = 0.5)
    event = None
    for event_index in range(500_000, 500_100):
        rngs = _event_rngs(event_index)
        candidate = generate_nd_multi(
            NOMINAL_PARAMS, NOMINAL_ND_REMNANT_WEIGHTS, COUNT_BND,
            catalog, fams, *rngs)
        if candidate.count == 2:
            event = candidate
            break
    assert event is not None, "no n = 2 event on the pinned probe indices"
    report = _audit(event)
    assert not report.violations
    check_multi_central_neutrality(event)

    source = MODULE_PATH.read_text(encoding="utf-8")
    for token in NO_REFIT_TOKENS:
        assert token not in source, token
    # the module routes both sources through card 7's committed decay
    # machinery (the frozen simplified cascade)
    assert "_decay_central" in source
    assert "_decay_remnant" in source


# ----------------------------------------------------------------------
# test_layout_sync
# ----------------------------------------------------------------------
def test_layout_sync():
    """The test register (the card's frozen-substrate provenance pins,
    verified on disk, including card 7's committed module; the M8
    hash and the kernel boundary through the manifest; the freeze
    re-pin of this card), and the layout sync (the subprocess check
    from the monorepo root; ``logic_drift = 0``)."""

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
    # at the operative count (523 at the card-7 test-author landing
    # plus this card's two new files), this card's freeze exception
    # exactly once with the path-sorted file list and the reason
    # naming the 523 -> 525 re-pin and the card's pre-registered
    # literal
    inventory = manifest["source_tree_sha256"]
    keys = list(inventory.keys())
    assert keys == sorted(keys), "the inventory is not path-sorted"
    assert len(keys) == INVENTORY_COUNT
    module_rel = "dev/mixing2006/na49_mixing/topology_multi_neutral.py"
    test_rel = "dev/mixing2006/tests/test_topology_multi_neutral.py"
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
        "the card-8 freeze exception must name its card exactly once")
    files = entries[0]["files"]
    assert files == sorted(files), (
        "the freeze-exception files are path-sorted")
    assert files == list(FREEZE_EXCEPTION_FILES), (
        "the freeze-exception file list drifted")
    reason = entries[0]["reason"]
    assert reason.startswith("Round 5, card 8 ("), (
        "the freeze-exception reason must name its card")
    assert FREEZE_CARD in reason, (
        "the freeze-exception reason must name its card")
    assert "523 -> 525" in reason, (
        "the freeze-exception reason must name the 523 -> 525 re-pin")
    assert "513 -> 515" in reason, (
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
    """The frozen record surface, the class label, the registry entry,
    the bounded-count constants, and the nominal parameters."""

    assert ND_MULTI_CLASS_LABEL == "nd_multi"
    assert getattr(NDMultiEvent, "__dataclass_params__").frozen
    assert tuple(NDMultiEvent.__dataclass_fields__) == (
        "class_label", "count", "p_n2", "remnants", "central_masses",
        "t_c", "t1", "t2", "t_2c", "lines", "central_sources",
        "exchanges", "final_state")
    assert REGISTRY_ENTRY == {
        "class": "nd_multi",
        "module": "na49_mixing.topology_multi_neutral",
        "retained": False,
        "count_bounded": True,
        "p_n2_bound": 0.5,
        "default_p_n2": 0.0,
    }
    assert COUNT_BND == 0.5
    assert DEFAULT_P2 == 0.0
    assert P_N2_TOLERANCE == 1e-12
    # the nominal parameters: the R50 fixed values (the card-7 module's
    # own frozen surface, consumed read-only)
    assert NOMINAL_PARAMS.epsilon == 0.0
    assert NOMINAL_PARAMS.b_sd == ELASTIC_TRANSFER_SLOPE
    assert NOMINAL_ND_REMNANT_WEIGHTS == {
        KIND_INTACT_PROTON: 0.5,
        KIND_LOW_MASS_BARYONIC_RESONANCE: 0.5,
        "high_mass_baryonic_continuum": 0.0,
    }
    # the closed substream labels and the rejection cap
    for label in ("topology_selection", "spectral_state",
                  "exchange_kinematics", "decay_channel",
                  "decay_kinematics"):
        assert label in SUBSTREAM_LABELS
    assert REJECTION_CAP == 1000
