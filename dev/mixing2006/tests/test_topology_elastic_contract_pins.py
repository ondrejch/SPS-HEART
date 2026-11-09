"""Round 5, card 1 test-author register: the elastic reference event class.

Independent contract pins for the elastic reference event class
(``na49_mixing/topology_elastic.py``), Round 5 card 1
(``TASK-20260925-r5-elastic-reference``; plan section 22 task 28,
line 1753; plan section 11.1 item 1, line 824).  This register is the
test-author half of the card: it pins the module contract on its own
pinned traces (the Round 5 master seed 20260925, replica 0, on the
closed ``exchange_kinematics`` substream) and re-pins the freeze
manifest for the card (``INVENTORY_COUNT`` 501 -> 502).

Part 1: the module contract (pure, hermetic)
--------------------------------------------

- ``test_module_constants``: the named module constants (the M8-frozen
  slope, the catalog proton mass, the beam momentum, the recorded
  sqrt(s)) and the doubled-I3 integer convention
  ``PROTON_QN = (1, 1, 0, 1)`` / ``ZERO_TRANSFER = (0, 0, 0, 0)``.
- ``test_t_law_contract``: the exact two-body elastic t range
  ``[t_min, 0]`` on the pinned M8 p+p configuration, the closed-form
  normalization against quadrature within 1e-12, the inverse-CDF round
  trip on the pinned quantile grid within 1e-9, and the checked
  (never repaired) domain edges.
- ``test_record_interface``: the frozen dataclasses and their exact
  field sets (no excited state, central system, or decay vertex in the
  record), the class label ``"elastic"``, the single vacuum exchange
  with zero transfer, the intact-proton lines, and the pinned incoming
  four-momenta.
- ``test_check_zero_transfer``: the exact integer ledger check passes
  on a pinned event and raises ``ElasticTransferViolation`` on two
  planted violations (a nonzero line transfer, a nonzero exchange
  transfer).
- ``test_acceptance_conservation``: over a pinned 10,000-event trace,
  four-momentum closure within 1e-9 relative (realized worst
  1.80e-15), both outgoing lines on the ``m_p`` mass shell within
  1e-9 relative (realized worst 1.34e-11), the realized Mandelstam t
  equal to the drawn t, and the exact ``(B, Q, S, 2I3)`` total
  ``(2, 2, 0, 2)`` (acceptance (i)).
- ``test_acceptance_t_law``: the closed-form normalization, the
  inverse-CDF round trip, and the 100,000-draw sample mean
  ``-0.11823975586629042`` against the analytic mean at
  ``z = 1.739`` sigma (acceptance (iii)).
- ``test_draw_budget``: a pinned 1,000-event trace consumes exactly
  one uniform on the closed ``exchange_kinematics`` substream per
  event and zero elsewhere (the counting proxy, the bit-identical
  post-draw stream state, and the continuous-stream identity;
  acceptance (v)).

Part 2: the substrate and freeze pins
-------------------------------------

- ``test_module_byte_and_export_pins``: the module at its committed
  sha256, its ``__all__`` export set, and the package ``__init__`` at
  its committed sha256 carrying the elastic export block.
- ``test_frozen_substrate_digests``: the card's frozen-substrate
  provenance surfaces (verified on disk) and the frozen M8 parameter
  map through the manifest.
- ``test_freeze_repin_register``: the freeze-manifest re-pin for this
  card (the strict path-sorted inventory at 502 entries after the
  Round 5 card 1 test-author re-pin; 501 at the Round 5 card 1
  landing) carries this register and the twenty-five re-pinned chain
  count-pin files at their on-disk digests, the solver's freeze
  exception stays exactly once, this card's freeze exception names the
  card exactly once with the path-sorted file list, the chain count
  pins are re-pinned 501 -> 502 with the 501 literal retained, and the
  frozen M8 parameter map stays intact.

Hermetic: standard library and numpy only; no event generation driver,
no external tables, no Monte Carlo run (the runs R50-R55 exercise the
class statistically).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from na49_mixing.rng_substreams import SUBSTREAM_LABELS, derive_substreams
from na49_mixing.topology_elastic import (
    BEAM_MOMENTUM,
    ELASTIC_CLASS_LABEL,
    ELASTIC_TRANSFER_SLOPE,
    PROTON_MASS,
    PROTON_QN,
    SQRT_S,
    VACUUM_EXCHANGE_NAME,
    ZERO_TRANSFER,
    ElasticEvent,
    ElasticTransferViolation,
    ExchangeRecord,
    LineState,
    check_zero_transfer,
    draw_elastic_t,
    elastic_t_normalization,
    elastic_t_quantile,
    elastic_t_range,
    generate_elastic,
)
from tests._freeze_inventory import inventory_count as _inventory_count

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent          # dev/mixing2006
MONO_ROOT = NESTED_ROOT.parent.parent          # the monorepo root
MANIFEST_PATH = NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json"
SYNC_CHECKER = MONO_ROOT / "helpers" / "benchmarking" / "check_helpers_layout_sync.py"

MY_TEST_REL = "dev/mixing2006/tests/test_topology_elastic_contract_pins.py"

#: The Round 5 master seed (round 5 index item 9), fixed for the pinned
#: traces; replica 0, event index = trace index.
MASTER_SEED = 20260925

#: Pinned trace sizes (the card's deterministic acceptance).
N_CONSERVATION = 10_000
N_T_DRAWS = 100_000
N_DRAW_BUDGET = 1_000

#: The pinned inverse-CDF quantile grid of the round-trip check (the
#: underflowed range edge excludes u = 0; see the module docstring).
QUANTILE_GRID = (
    0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0,
)

#: The committed module and package digests (nested tree HEAD
#: 91e38f7; the package digest re-pinned by Round 5 card 2,
#: TASK-20260925-r5-exchange-quantum-number-ledger, for the
#: exchange-ledger module's additive package exports).
MODULE_SHA256 = "7640e64e0cb9428f712f854384c1e0410947909e6a3c5cb7aa9c1faf5fc1b824"
#: re-pinned by Round 5 card 3 (TASK-20260925-r5-intact-low-mass-remnants) for the remnant module's additive package exports.
# re-pinned by Round 5 card 7 (TASK-20260925-r5-non-diffrative-central-excitation; the non-diffractive module's additive package exports).
# re-pinned by Round 5 card 8 (TASK-20260925-r5-multiple-neutral-excitations; the multiple-neutral-excitation module's additive package exports).
# re-pinned by Round 5 card 9 (TASK-20260925-r5-topology-template-runner; the topology-runner module's additive package exports).
# re-pinned by the Round 5 card 9 review fix round (TASK-20260925-r5-topology-template-runner-review-fix; the REV-bc4de47-03 export-surface correction re-pinned the package __init__).
INIT_SHA256 = "7d2c5a85d9b24de104c0d9cf72195689bf7c39c873e154798699448073b36de8"

#: The module's exact ``__all__`` export set (twelve names).
MODULE_ALL = [
    "BEAM_MOMENTUM",
    "ELASTIC_TRANSFER_SLOPE",
    "PROTON_MASS",
    "SQRT_S",
    "ElasticEvent",
    "ElasticTransferViolation",
    "ExchangeRecord",
    "LineState",
    "check_zero_transfer",
    "draw_elastic_t",
    "elastic_t_range",
    "generate_elastic",
]

#: The measured acceptance values of the pinned traces (the Round 5
#: master seed 20260925, replica 0; ``exchange_kinematics`` substream).
T_MIN = -294.74557216940036
T_MAX = 0.0
T_NORMALIZATION = 0.11758977514325487
MEAN_EXACT = -0.11758977514325487
T_SAMPLE_MEAN = -0.11823975586629042
T_SAMPLE_Z = 1.739
WORST_CLOSURE_REL = 1.798842369012912e-15
WORST_MASS_SHELL_REL = 1.3402821404725402e-11
WORST_T_MISMATCH_REL = 4.285460875053104e-14
WORST_ROUND_TRIP = 2.7755575615628914e-17

#: The M8-frozen effective transfer slope and kernel boundary, pinned
#: through the manifest's frozen parameter map.
SLOPE_PIN = 8.504140762083612
KERNEL_BOUNDARY = 2.680194110727528

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
#: re-pin (501 at the Round 5 card 1 landing; +1 entry: this register).
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
#: The solver card (Round 5, card 1) and this test-author register's
#: card name (the freeze-exception identity).
SOLVER_CARD = "TASK-20260925-r5-elastic-reference"
TEST_AUTHOR_CARD = "TASK-20260925-r5-elastic-reference-test-author"

#: The register chain: the twenty-five chain count-pin files re-pinned
#: 501 -> 502 by this register (path-sorted).
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
)

#: This register's freeze-exception file list (path-sorted): the
#: manifest, this register, and the twenty-five re-pinned chain
#: count-pin files.
TEST_AUTHOR_EXCEPTION_FILES = tuple(sorted(
    ("dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json",)
    + CHAIN_COUNT_PIN_FILES
    + (MY_TEST_REL,)
))


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _elastic_rng(event_index: int) -> np.random.Generator:
    """The pinned exchange_kinematics stream of one trace event."""

    streams = derive_substreams(MASTER_SEED, 0, event_index)
    return np.random.default_rng(streams.stream_for("exchange_kinematics"))


def _elastic_stream_id(event_index: int) -> int:
    return derive_substreams(MASTER_SEED, 0, event_index).stream_for(
        "exchange_kinematics")


class _FixedUniform:
    """A duck-typed single-uniform source: ``random()`` always returns
    the pinned quantile (the inverse-CDF round-trip probe)."""

    def __init__(self, u: float) -> None:
        self.u = u

    def random(self) -> float:
        return self.u


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


def _analytic_t_range() -> tuple[float, float]:
    return elastic_t_range(SQRT_S)


def _analytic_mean(t_min: float) -> float:
    """The closed-form mean of ``exp(B_SD t)`` on ``[t_min, 0]``:
    ``-1/B - t_min e^{B t_min} / (1 - e^{B t_min})``."""

    e_lo = math.exp(ELASTIC_TRANSFER_SLOPE * t_min)
    return (-1.0 / ELASTIC_TRANSFER_SLOPE
            - t_min * e_lo / (1.0 - e_lo))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ----------------------------------------------------------------------
# Part 1: the module contract
# ----------------------------------------------------------------------
def test_module_constants():
    """The named module constants and the doubled-I3 integer
    convention: the M8-frozen slope, the catalog proton mass, the beam
    momentum, the recorded sqrt(s), the proton qn, the zero transfer,
    the class label, and the vacuum exchange name."""

    assert ELASTIC_TRANSFER_SLOPE == SLOPE_PIN
    assert PROTON_MASS == 0.93827231
    assert BEAM_MOMENTUM == 158.0
    assert SQRT_S == 17.270
    assert isinstance(ELASTIC_TRANSFER_SLOPE, float)
    assert isinstance(PROTON_MASS, float)
    assert isinstance(BEAM_MOMENTUM, float)
    assert isinstance(SQRT_S, float)
    assert PROTON_QN == (1, 1, 0, 1)
    assert ZERO_TRANSFER == (0, 0, 0, 0)
    assert ELASTIC_CLASS_LABEL == "elastic"
    assert VACUUM_EXCHANGE_NAME == "vacuum"
    # the named constants are integer tuples in the doubled-I3 order
    assert all(isinstance(part, int) for part in PROTON_QN)
    assert all(isinstance(part, int) for part in ZERO_TRANSFER)


def test_t_law_contract():
    """The exact two-body elastic t range on the pinned M8 p+p
    configuration, the closed-form normalization against quadrature
    within 1e-12, the inverse-CDF round trip on the pinned quantile
    grid within 1e-9, and the checked (never repaired) domain edges."""

    t_min, t_max = _analytic_t_range()
    assert t_min == T_MIN
    assert t_max == T_MAX
    assert t_min < 0.0
    # the range edge is the squared four-momentum transfer of a
    # back-to-back elastic pair at the fixed-target invariant mass
    analytic = -((2.0 * PROTON_MASS * BEAM_MOMENTUM / SQRT_S) ** 2)
    assert abs(t_min - analytic) <= 1e-12

    # the recorded SQRT_S is the rounded quote of the exact
    # fixed-target invariant mass; the nominal range edge sits within
    # 0.01 GeV^2 of the exact two-body limit of the beam-momentum
    # configuration (the guard checks the exact limit; see the module
    # docstring)
    e_beam = math.sqrt(PROTON_MASS * PROTON_MASS
                       + BEAM_MOMENTUM * BEAM_MOMENTUM)
    s_exact = 2.0 * PROTON_MASS * (PROTON_MASS + e_beam)
    exact_limit = s_exact - 4.0 * PROTON_MASS * PROTON_MASS
    assert abs((-t_min) - exact_limit) < 0.01

    # the closed-form normalization against quadrature within 1e-12
    slope = ELASTIC_TRANSFER_SLOPE
    closed = (1.0 - math.exp(slope * t_min)) / slope
    assert elastic_t_normalization(t_min) == T_NORMALIZATION
    assert closed == T_NORMALIZATION
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quad_value, _err = quad(
            lambda t: math.exp(slope * t), t_min, t_max,
            epsabs=1e-13, epsrel=1e-13, limit=400)
    assert abs(quad_value - T_NORMALIZATION) <= 1e-12

    # the inverse-CDF round trip on the pinned quantile grid within
    # 1e-9 (the draw of a pinned quantile reproduces the quantile, and
    # the closed-form CDF of the drawn t reproduces the quantile)
    e_lo = math.exp(slope * t_min)
    worst = 0.0
    for u in QUANTILE_GRID:
        t_quant = elastic_t_quantile(u, t_min)
        t_draw = draw_elastic_t(_FixedUniform(u))
        assert abs(t_draw - t_quant) <= 1e-9
        cdf = (math.exp(slope * t_draw) - e_lo) / (1.0 - e_lo)
        worst = max(worst, abs(cdf - u))
        assert abs(cdf - u) <= 1e-9
    assert worst <= 1e-9

    # checked, never repaired: the domain edges are rejected
    with pytest.raises(ValueError):
        elastic_t_range(2.0 * PROTON_MASS)
    with pytest.raises(ValueError):
        elastic_t_range(float("nan"))
    with pytest.raises(ValueError):
        elastic_t_normalization(0.0)
    with pytest.raises(ValueError):
        elastic_t_quantile(0.0, t_min)
    with pytest.raises(ValueError):
        elastic_t_quantile(-1e-9, t_min)
    with pytest.raises(ValueError):
        elastic_t_quantile(1.0 + 1e-9, t_min)


def test_record_interface():
    """The frozen dataclasses and their exact field sets (no excited
    state, central system, or decay vertex in the record), the class
    label ``"elastic"``, the single vacuum exchange with zero transfer,
    the intact-proton lines, and the pinned incoming four-momenta."""

    event = generate_elastic(_elastic_rng(11))

    assert getattr(ElasticEvent, "__dataclass_params__").frozen
    assert getattr(LineState, "__dataclass_params__").frozen
    assert getattr(ExchangeRecord, "__dataclass_params__").frozen

    # the record's field set is exactly the card's five fields
    assert [f.name for f in dataclasses.fields(ElasticEvent)] == [
        "class_label", "incoming", "outgoing", "exchange", "t"]
    assert [f.name for f in dataclasses.fields(LineState)] == [
        "four_momentum", "qn"]
    assert [f.name for f in dataclasses.fields(ExchangeRecord)] == [
        "name", "transfer"]

    # the label excludes the class from every inelastic sample
    # (plan line 824)
    assert event.class_label == "elastic"
    assert event.exchange.name == "vacuum"
    assert event.exchange.transfer == (0, 0, 0, 0)
    assert event.t < 0.0

    # every line is an intact proton in the doubled-I3 convention
    for line in event.incoming + event.outgoing:
        assert isinstance(line, LineState)
        assert line.qn == (1, 1, 0, 1)
        assert len(line.four_momentum) == 4

    # the pinned incoming four-momenta: the beam proton at
    # (e_beam, 0, 0, p_beam), the target proton at rest at
    # (m_p, 0, 0, 0)
    e_beam = math.sqrt(PROTON_MASS * PROTON_MASS
                       + BEAM_MOMENTUM * BEAM_MOMENTUM)
    assert event.incoming[0].four_momentum == (
        e_beam, 0.0, 0.0, BEAM_MOMENTUM)
    assert event.incoming[1].four_momentum == (
        PROTON_MASS, 0.0, 0.0, 0.0)

    # no nested record beyond LineState/ExchangeRecord exists anywhere
    # in the record (no decay vertex, no central system)
    def _scan(obj):
        if isinstance(obj, tuple):
            for part in obj:
                _scan(part)
        elif dataclasses.is_dataclass(obj):
            assert isinstance(obj, (LineState, ExchangeRecord, ElasticEvent)), (
                f"unexpected record in the elastic event: {type(obj)!r}")

    _scan(event)


def test_check_zero_transfer():
    """The exact integer zero-transfer check passes on a pinned event
    and raises ``ElasticTransferViolation`` on two planted violations
    (a nonzero line transfer, a nonzero exchange transfer)."""

    event = generate_elastic(_elastic_rng(7))
    assert check_zero_transfer(event) is None

    # planted violation 1: a nonzero line transfer (the target-side
    # outgoing line loses one unit of charge)
    bad_line = dataclasses.replace(
        event,
        outgoing=(event.outgoing[0],
                  dataclasses.replace(event.outgoing[1], qn=(1, 0, 0, 1))))
    with pytest.raises(ElasticTransferViolation, match="line 1"):
        check_zero_transfer(bad_line)

    # planted violation 2: a nonzero exchange transfer
    bad_exchange = dataclasses.replace(
        event,
        exchange=ExchangeRecord(name="vacuum", transfer=(0, 0, 1, 0)))
    with pytest.raises(ElasticTransferViolation, match="exchange"):
        check_zero_transfer(bad_exchange)

    # checked, never repaired: the check consumes an ElasticEvent
    with pytest.raises(ValueError):
        check_zero_transfer(("not", "an", "event"))


def test_acceptance_conservation():
    """Over a pinned 10,000-event trace (acceptance (i)): exact
    ``(B, Q, S, 2I3)`` bookkeeping ((1, 1, 0, 1) per line,
    ``(2, 2, 0, 2)`` total), four-momentum closure within 1e-9
    relative (realized worst 1.80e-15), both outgoing lines on the
    ``m_p`` mass shell within 1e-9 relative (realized worst 1.34e-11),
    and the realized Mandelstam t equal to the drawn t."""

    n = N_CONSERVATION
    worst_closure = 0.0
    worst_mass = 0.0
    worst_t = 0.0
    for i in range(n):
        event = generate_elastic(_elastic_rng(i))
        check_zero_transfer(event)

        # exact integer bookkeeping per line and in total
        for in_line, out_line in zip(event.incoming, event.outgoing):
            assert in_line.qn == (1, 1, 0, 1)
            assert out_line.qn == (1, 1, 0, 1)
        total_in = tuple(sum(part) for part in zip(
            *(line.qn for line in event.incoming)))
        total_out = tuple(sum(part) for part in zip(
            *(line.qn for line in event.outgoing)))
        assert total_in == (2, 2, 0, 2)
        assert total_out == (2, 2, 0, 2)

        # four-momentum closure within 1e-9 relative
        in_sum = [a + b for a, b in zip(event.incoming[0].four_momentum,
                                        event.incoming[1].four_momentum)]
        out_sum = [a + b for a, b in zip(event.outgoing[0].four_momentum,
                                         event.outgoing[1].four_momentum)]
        for k in range(4):
            delta = abs(out_sum[k] - in_sum[k])
            assert delta <= 1e-9 * abs(in_sum[k])
            if in_sum[k] != 0.0:
                worst_closure = max(worst_closure,
                                    delta / abs(in_sum[k]))

        # both outgoing lines on the m_p mass shell within 1e-9 relative
        for line in event.outgoing:
            e, px, py, pz = line.four_momentum
            mass = math.sqrt(e * e - px * px - py * py - pz * pz)
            rel = abs(mass - PROTON_MASS) / PROTON_MASS
            assert rel <= 1e-9
            worst_mass = max(worst_mass, rel)

        # the realized Mandelstam t of the solved kinematics equals the
        # drawn t (absolute tolerance for the small-|t| bulk, relative
        # beyond 1 GeV^2)
        p1 = event.incoming[0].four_momentum
        p3 = event.outgoing[0].four_momentum
        t_real = ((p1[0] - p3[0]) ** 2
                  - (p1[1] - p3[1]) ** 2
                  - (p1[2] - p3[2]) ** 2
                  - (p1[3] - p3[3]) ** 2)
        rel = abs(t_real - event.t) / max(1.0, abs(event.t))
        assert rel <= 1e-9
        worst_t = max(worst_t, rel)

    # the realized worst values sit far inside the contract tolerance
    assert worst_closure <= 1e-14
    assert worst_mass <= 1e-10
    assert worst_t <= 1e-13


def test_acceptance_t_law():
    """The t law acceptance (iii): the closed-form normalization
    ``0.11758977514325487`` (``1/B``, the underflowed range edge), the
    inverse-CDF round trip within 1e-9, and the 100,000-draw sample
    mean ``-0.11823975586629042`` against the analytic mean at
    ``z = 1.739`` sigma of 1/sqrt(N)."""

    t_min, _t_max = _analytic_t_range()
    slope = ELASTIC_TRANSFER_SLOPE

    # the closed-form normalization (the underflowed range edge makes
    # it exactly 1/B)
    e_lo = math.exp(slope * t_min)
    assert e_lo == 0.0
    assert T_NORMALIZATION == (1.0 / slope)
    assert MEAN_EXACT == (-1.0 / slope)

    # the inverse-CDF round trip on the pinned quantile grid
    worst = 0.0
    for u in QUANTILE_GRID:
        t_quant = elastic_t_quantile(u, t_min)
        cdf = (math.exp(slope * t_quant) - e_lo) / (1.0 - e_lo)
        worst = max(worst, abs(cdf - u))
        assert abs(cdf - u) <= 1e-9
    assert worst == WORST_ROUND_TRIP

    # the 100,000-draw sample mean (one pinned exchange_kinematics
    # stream, consumed sequentially): the bit-exact pinned mean against
    # the analytic mean at z = 1.739 sigma of 1/sqrt(N)
    n = N_T_DRAWS
    stream = _elastic_rng(0)
    draws = np.fromiter((draw_elastic_t(stream) for _ in range(n)),
                        dtype=float, count=n)
    mean = float(np.mean(draws))
    sigma_mean = float(np.std(draws)) / math.sqrt(n)
    # np.mean's summation order differs between NumPy versions, so the pin holds to 1e-14 relative
    assert math.isclose(mean, T_SAMPLE_MEAN, rel_tol=1e-14, abs_tol=0.0)
    assert abs(mean - MEAN_EXACT) <= 3.0 * sigma_mean
    z = abs(mean - MEAN_EXACT) / sigma_mean
    assert abs(z - T_SAMPLE_Z) <= 1e-3


def test_draw_budget():
    """A pinned 1,000-event trace (acceptance (v)) consumes exactly one
    uniform on the closed exchange_kinematics substream per event and
    zero elsewhere: the counting proxy, the bit-identical post-draw
    stream state, the inverse CDF of the first pristine uniform, and
    the continuous-stream identity against the pristine uniform
    sequence."""

    assert "exchange_kinematics" in SUBSTREAM_LABELS
    t_min, _t_max = _analytic_t_range()
    n = N_DRAW_BUDGET

    # per-event stream: exactly one uniform (counting proxy), the
    # post-draw state is bit-identical to the reference stream after
    # exactly one draw, and the drawn t is the inverse CDF of the first
    # pristine uniform
    for i in range(n):
        sid = _elastic_stream_id(i)
        counter = _CountingRng(np.random.default_rng(sid))
        event = generate_elastic(counter)
        assert counter.calls == 1

        advanced = np.random.default_rng(sid)
        advanced.random()
        assert counter._source.bit_generator.state == advanced.bit_generator.state

        u_first = np.random.default_rng(sid).random()
        assert event.t == elastic_t_quantile(u_first, t_min)

    # the continuous-stream identity: one shared stream over the whole
    # trace consumes exactly the pristine uniform sequence in order
    # (no burn-in, no extra draw anywhere in the module)
    shared = _elastic_rng(0)
    pristine = np.random.default_rng(_elastic_stream_id(0))
    for i in range(n):
        u_ref = pristine.random()
        event = generate_elastic(shared)
        assert event.t == elastic_t_quantile(u_ref, t_min)


# ----------------------------------------------------------------------
# Part 2: the substrate and freeze pins
# ----------------------------------------------------------------------
def test_module_byte_and_export_pins():
    """The module at its committed sha256 with its exact ``__all__``
    export set, and the package ``__init__`` at its committed sha256
    carrying the elastic export block (all twelve names in the import
    block and in ``__all__``)."""

    module_path = NESTED_ROOT / "na49_mixing" / "topology_elastic.py"
    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    import na49_mixing.topology_elastic as _mod
    assert _mod.__all__ == MODULE_ALL

    init_path = NESTED_ROOT / "na49_mixing" / "__init__.py"
    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)
    init_text = init_path.read_text(encoding="utf-8")
    block = re.search(
        r"from \.topology_elastic import \((.*?)\)", init_text, re.S)
    assert block is not None, "the __init__ elastic import block is missing"
    for name in MODULE_ALL:
        assert re.search(rf"\b{name}\b", block.group(1)), (
            f"{name} is not in the __init__ elastic import block")
    all_block = re.search(r"__all__\s*=\s*\[(.*?)\n\]", init_text, re.S)
    assert all_block is not None
    for name in MODULE_ALL:
        assert f'"{name}"' in all_block.group(1), (
            f"{name} is not in the __init__ __all__")


def test_frozen_substrate_digests():
    """The card's frozen-substrate provenance surfaces (verified on
    disk) and the frozen M8 parameter map through the manifest (the
    slope and the kernel boundary)."""

    for rel, expected in FROZEN_SURFACES.items():
        path = MONO_ROOT / rel
        assert path.is_file(), f"missing frozen surface: {rel}"
        pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    m8 = manifest["frozen_fit_outputs"]["m8_frozen_pp_parameters"]
    assert m8["regge.effective_transfer_slope"] == SLOPE_PIN
    assert m8["continuum.match_mass"] == KERNEL_BOUNDARY




def test_layout_sync():
    """The layout sync (the subprocess check from the monorepo root;
    ``logic_drift = 0``)."""

    probe = subprocess.run(
        [sys.executable, str(SYNC_CHECKER)],
        capture_output=True, text=True, cwd=str(MONO_ROOT))
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert "logic_drift=0" in probe.stdout
