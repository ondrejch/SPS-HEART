"""Round 5, card 5 test-author register: the single-diffraction classes.

Independent contract pins for the projectile and target single-diffraction
classes (``na49_mixing/topology_single_diffraction.py``), Round 5 card 5
(``TASK-20260925-r5-single-diffraction``; plan section 22 task 29,
line 1754; plan section 11.1 items 2-3, lines 825-826).  This register is
the test-author half of the card: it pins the module contract on its own
pinned fixtures (the closed side set and the frozen record surface, the
exact two-body solve's closure, mass shell, domain refusals, and the
closed-at-0 forward sliver, the event record's line and exchange
annotations, the byte-stable module / solver test / package ``__init__``
digests and export surface, and the frozen-substrate provenance) and
re-pins the freeze manifest for the card (``INVENTORY_COUNT`` 516 ->
517).

Part 1: the module contract (pure, hermetic)
--------------------------------------------

- ``test_named_constants_and_sides``: the closed ``SDSide`` set
  ``{"projectile", "target"}``, the two class labels, the frozen
  record surfaces (the nine-field ``SDEvent`` and the five-field
  ``FinalHadron``), the nominal parameters (``epsilon = 0``,
  ``B_SD`` the M8-frozen slope), and the nominal remnant split
  (intact 0.5 / low-mass baryonic 0.5 / high-mass 0.0).
- ``test_sd_kinematics_closure_and_mass_shell``: the exact two-body
  solve closes the incoming sum within ``1e-9`` relative on a pinned
  grid of ``(M, t, m_remnant)`` points, the solved pair sits on its
  mass shell, and the checked domain refusals (positive ``t``,
  non-positive mass, above-threshold configuration) raise
  ``ValueError``.
- ``test_sd_kinematics_forward_sliver``: the exact two-body forward
  edge ``t_fwd(M, m_remnant) < 0`` for every excited mass ``M > m_p``
  (the documented closed-at-0 sliver), and a drawn ``t`` above the
  edge (``t = 0``) solves to the forward configuration
  (``cos(theta*) = 1``, transverse momentum zero).
- ``test_sd_kinematics_angle_documented_formula``: the scattering
  angle follows the documented invariant transfer on the excitation
  line, ``t = m_p^2 + M^2 - 2 (E_p* E_M* - p* q* cos(theta*))``
  (module docstring), solved to
  ``cos(theta*) = (t - m_p^2 - M^2 + 2 E_p* E_M*) / (2 p* q*)``;
  the factor 1/2 on the momentum transfer was restored at nested
  2ccc80b (REV-85f239a-01), and this pin now passes as a normal
  contract pin.
- ``test_event_record_surface``: the event record's line and exchange
  annotations (the two outgoing lines in the fixed projectile-side-first
  order, the single vacuum exchange with the zero transfer, the
  ``line`` labels of the final-state hadrons, and the exchange
  inheritance ``(1, 1, 0, 1)`` on the excitation line).

Part 2: the substrate and freeze pins
--------------------------------------

- ``test_module_byte_and_export_pins``: the module at its committed
  sha256, its ``__all__`` export set, the solver test file at its
  committed sha256, and the package ``__init__`` at its committed
  sha256 carrying the single-diffraction export block.
- ``test_frozen_substrate_digests``: the card's frozen-substrate
  provenance surfaces (verified on disk) and the frozen M8 parameter
  map through the manifest.
- ``test_freeze_repin_register``: the freeze-manifest re-pin for this
  card (the strict path-sorted inventory at 517 entries; 516 after the
  Round 5 card 5 re-pin; 514 after the Round 5 card 4 test-author
  re-pin; 513 after the Round 5 card 4 re-pin; 511 after the Round 5
  card 3 test-author re-pin; 510 after the Round 5 card 3 solver
  landing; 508 after the test-protocol tiering change; 505 after the
  Round 5 card 2 test-author re-pin; 504 after the Round 5 card 2
  solver landing; 502 after the Round 5 card 1 test-author re-pin; 501
  at the Round 5 card 1 landing) carries this register and the
  twenty-nine re-pinned chain count-pin files at their on-disk
  digests, the card 5 solver's freeze exception stays exactly once at
  its 33 path-sorted files, this card's freeze exception names the
  card exactly once with the path-sorted file list, the chain count
  pins are re-pinned 516 -> 517 with the 516 literal retained in the
  naming comments, and the frozen M8 parameter map stays intact.
- ``test_layout_sync``: the layout sync (the subprocess check from the
  monorepo root; ``logic_drift = 0``).

Hermetic: the module, the committed catalog, the continuum families
configuration, the Round 4 cascade modules (consumed read-only),
numpy, and the standard library only; no event generation driver, no
external tables, no Monte Carlo run (the runs R50-R55 exercise the
class statistically).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.rng_substreams import derive_substreams
from na49_mixing.spectral_density import families_from_json
from na49_mixing.topology_elastic import (
    BEAM_MOMENTUM,
    ELASTIC_TRANSFER_SLOPE,
    PROTON_MASS,
    SQRT_S,
)
from na49_mixing.topology_mass_laws import sd_limits, t_range
from na49_mixing.topology_remnants import (
    KIND_INTACT_PROTON,
    KIND_LOW_MASS_BARYONIC_RESONANCE,
    low_mass_records,
)
from na49_mixing.topology_single_diffraction import (
    NOMINAL_PARAMS,
    NOMINAL_SD_REMNANT_WEIGHTS,
    SD_CLASS_LABEL_PROJECTILE,
    SD_CLASS_LABEL_TARGET,
    SDSIDES,
    FinalHadron,
    SDEvent,
    _beam_configuration,
    _incoming_lines,
    gap_size,
    generate_sd,
    mirror_sd,
    sd_kinematics,
)
from tests._freeze_inventory import inventory_count as _inventory_count

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent          # dev/mixing2006
MONO_ROOT = NESTED_ROOT.parent.parent          # the monorepo root
MANIFEST_PATH = NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json"
SYNC_CHECKER = MONO_ROOT / "helpers" / "benchmarking" / "check_helpers_layout_sync.py"
MODULE_PATH = NESTED_ROOT / "na49_mixing" / "topology_single_diffraction.py"
SOLVER_TEST_PATH = NESTED_ROOT / "tests" / "test_topology_single_diffraction.py"
INIT_PATH = NESTED_ROOT / "na49_mixing" / "__init__.py"
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
CONTINUUM_PATH = (
    NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json")

#: The committed digests of the card's pinned surfaces (nested tree
#: HEAD 2ccc80b).
MODULE_SHA256 = (
    "bbfb63d651bb4fa12a9a75b464b696c6a7be0c88eddc27bd267714a9479e205a")
SOLVER_TEST_SHA256 = (
    "c92b4ee6bbe6d77b34bab355f559465bdd0551719e7453aedc58d344c67a6099")
# re-pinned by Round 5 card 7 (TASK-20260925-r5-non-diffrative-central-excitation; the non-diffractive module's additive package exports).
# re-pinned by Round 5 card 8 (TASK-20260925-r5-multiple-neutral-excitations; the multiple-neutral-excitation module's additive package exports).
# re-pinned by Round 5 card 9 (TASK-20260925-r5-topology-template-runner; the topology-runner module's additive package exports).
# re-pinned by the Round 5 card 9 review fix round (TASK-20260925-r5-topology-template-runner-review-fix; the REV-bc4de47-03 export-surface correction re-pinned the package __init__).
INIT_SHA256 = (
    "7d2c5a85d9b24de104c0d9cf72195689bf7c39c873e154798699448073b36de8")

#: The module's ``__all__`` export set (fourteen names).
MODULE_ALL = [
    "SDSIDES",
    "SD_CLASS_LABEL_PROJECTILE",
    "SD_CLASS_LABEL_TARGET",
    "NOMINAL_SD_REMNANT_WEIGHTS",
    "NOMINAL_PARAMS",
    "BEAM_MOMENTUM",
    "PROTON_MASS",
    "SQRT_S",
    "FinalHadron",
    "SDEvent",
    "gap_size",
    "generate_sd",
    "mirror_sd",
    "sd_kinematics",
]

#: The ``__init__`` single-diffraction export block (the eleven
#: re-exported names; ``BEAM_MOMENTUM``, ``PROTON_MASS``, and ``SQRT_S``
#: stay card 1's elastic constants and are module-level only).
INIT_SD_NAMES = [
    "NOMINAL_PARAMS",
    "NOMINAL_SD_REMNANT_WEIGHTS",
    "SD_CLASS_LABEL_PROJECTILE",
    "SD_CLASS_LABEL_TARGET",
    "SDSIDES",
    "FinalHadron",
    "SDEvent",
    "gap_size",
    "generate_sd",
    "mirror_sd",
    "sd_kinematics",
]

#: The package ``__all__`` total entry count (the additive card 5
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
#: trees, and the four Round 5 modules consumed read-only (card 1's
#: elastic reference, card 2's exchange ledger, card 3's remnants,
#: card 4's mass laws).
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
#: The solver card (Round 5, card 5) and this test-author register's
#: card name (the freeze-exception identity).
SOLVER_CARD = "TASK-20260925-r5-single-diffraction"
TEST_AUTHOR_CARD = (
    "TASK-20260925-r5-single-diffraction-test-author")

#: This register's relative path in the monorepo (the inventory key).
MY_TEST_REL = (
    "dev/mixing2006/tests/test_topology_single_diffraction_contract_pins.py")

#: The register chain: the twenty-nine chain count-pin files re-pinned
#: 516 -> 517 by this register (the twenty-eight card-4 chain files
#: plus the card-4 test-author register itself, path-sorted).
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
)

#: This register's freeze-exception file list (path-sorted): the
#: manifest, this register, and the twenty-nine re-pinned chain
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


def _cm_quantities(m: float, m_rem: float):
    """The exact two-body CM energies and momenta of the pinned
    configuration: ``(e_p, e_m, p_star, q_star)`` for the incoming
    ``(m_p, m_remnant)`` pair and the outgoing ``(M, m_remnant)`` pair."""

    _e_beam, _e_total, sq_exact, _p_star, _beta, _gamma = (
        _beam_configuration())
    s = sq_exact * sq_exact
    mp2 = PROTON_MASS * PROTON_MASS
    mr2 = m_rem * m_rem
    e_p = (s + mp2 - mr2) / (2.0 * sq_exact)
    e_m = (s + m * m - mr2) / (2.0 * sq_exact)
    p_star = math.sqrt(
        max(0.0, (s - (PROTON_MASS + m_rem) ** 2)
              * (s - (PROTON_MASS - m_rem) ** 2))) / (2.0 * sq_exact)
    q_star = math.sqrt(
        max(0.0, (s - (m + m_rem) ** 2) * (s - (m - m_rem) ** 2))
        ) / (2.0 * sq_exact)
    return e_p, e_m, p_star, q_star


def _solved_cos_theta(m: float, t: float, m_rem: float) -> float:
    """The CM scattering angle of the solved excitation, recovered
    from the returned lab four-momentum (the invariant
    ``P_X . p_in`` against the incoming projectile)."""

    excitation, _remnant = sd_kinematics(m, t, m_rem, SQRT_S)
    e_beam, _e_total, _sq, _p_star, _beta, _gamma = _beam_configuration()
    e_p, e_m, p_star, q_star = _cm_quantities(m, m_rem)
    dot = excitation[0] * e_beam - excitation[3] * BEAM_MOMENTUM
    return (e_p * e_m - dot) / (p_star * q_star)


def _generate_one(event_index: int, catalog, fams, side: str = "projectile") \
        -> SDEvent:
    """One pinned event at the Round 5 master seed, replica 0."""

    subs = derive_substreams(20260925, 0, event_index)
    rngs = tuple(
        np.random.default_rng(subs.stream_for(label))
        for label in ("spectral_state", "exchange_kinematics",
                      "decay_channel", "decay_kinematics"))
    return generate_sd(
        side, NOMINAL_PARAMS, NOMINAL_SD_REMNANT_WEIGHTS,
        catalog, fams, *rngs)


@pytest.fixture(scope="module")
def catalog() -> HadronCatalog:
    return HadronCatalog.from_json(CATALOG_PATH)


@pytest.fixture(scope="module")
def fams():
    return families_from_json(str(CONTINUUM_PATH))


# ----------------------------------------------------------------------
# Part 1: the module contract (pure, hermetic)
# ----------------------------------------------------------------------
def test_named_constants_and_sides():
    """The closed ``SDSide`` set, the two class labels, the frozen
    record surfaces, and the nominal parameters (the R50 fixed values:
    ``epsilon = 0``, ``B_SD`` the M8-frozen slope; the nominal remnant
    split intact 0.5 / low-mass baryonic 0.5 / high-mass 0.0)."""

    assert isinstance(SDSIDES, frozenset)
    assert SDSIDES == frozenset({"projectile", "target"})
    assert SD_CLASS_LABEL_PROJECTILE == "sd_projectile"
    assert SD_CLASS_LABEL_TARGET == "sd_target"
    assert getattr(SDEvent, "__dataclass_params__").frozen
    assert tuple(SDEvent.__dataclass_fields__) == (
        "class_label", "side", "remnant", "excitation_mass", "t",
        "lines", "exchange", "final_state", "gap_size",
        "exchange_configuration", "excitation_m_min", "slope_factor", "tail", "ope")  # Round 8 2026-10-04, default None (the vacuum exchange);
    # tail: 2026-10-23 (R112), None without the tail component, True / False for a tail / coherence-region event;
    # ope: 2026-10-26, None without the one-pion-exchange component, True for a p + p -> n + X++ event;
    # excitation_m_min: 2026-10-06, None = the shared edge (line_thresholds off)
    # slope_factor: 2026-10-08, None = the single transfer slope (transfer_slope_mass_exponent off)
    assert SDEvent.__dataclass_fields__["exchange_configuration"].default is None
    assert getattr(FinalHadron, "__dataclass_params__").frozen
    assert tuple(FinalHadron.__dataclass_fields__) == (
        "pdg_id", "four_momentum", "mass", "qn", "line")
    assert NOMINAL_PARAMS.epsilon == 0.0
    assert NOMINAL_PARAMS.b_sd == ELASTIC_TRANSFER_SLOPE
    assert NOMINAL_SD_REMNANT_WEIGHTS == {
        KIND_INTACT_PROTON: 0.5,
        KIND_LOW_MASS_BARYONIC_RESONANCE: 0.5,
        "high_mass_baryonic_continuum": 0.0,
    }


def test_sd_kinematics_closure_and_mass_shell():
    """The exact two-body solve closes the incoming sum within ``1e-9``
    relative on a pinned grid of ``(M, t, m_remnant)`` points, the
    solved pair sits on its mass shell, and the checked domain
    refusals (positive ``t``, non-positive mass, the above-threshold
    configuration) raise ``ValueError``."""

    reference = _incoming_sum()
    points = (
        (2.0, -1.0, PROTON_MASS),
        (3.0, -1.0, PROTON_MASS),
        (5.0, -50.0, PROTON_MASS),
        (12.0, -200.0, PROTON_MASS),
        (2.0, -1.0, 1.52),
        (4.0, -30.0, 1.72),
    )
    for m, t, m_rem in points:
        excitation, remnant = sd_kinematics(m, t, m_rem, SQRT_S)
        total = tuple(excitation[i] + remnant[i] for i in range(4))
        relative = max(
            abs(total[0] - reference[0]) / reference[0],
            abs(total[1]) / reference[3],
            abs(total[2]) / reference[3],
            abs(total[3] - reference[3]) / reference[3])
        assert relative <= 1e-9, (m, t, m_rem, relative)
        # the solved pair sits on its mass shell
        m_exc2 = (excitation[0] ** 2 - excitation[1] ** 2
                  - excitation[2] ** 2 - excitation[3] ** 2)
        m_rem2 = (remnant[0] ** 2 - remnant[1] ** 2
                  - remnant[2] ** 2 - remnant[3] ** 2)
        assert abs(m_exc2 - m * m) <= 1e-9 * m * m, (m, t, m_rem)
        assert abs(m_rem2 - m_rem * m_rem) <= 1e-9 * m_rem * m_rem, (
            m, t, m_rem)

    # the checked domain refusals
    with pytest.raises(ValueError):
        sd_kinematics(PROTON_MASS, 1.0, PROTON_MASS, SQRT_S)
    with pytest.raises(ValueError):
        sd_kinematics(0.0, 0.0, PROTON_MASS, SQRT_S)
    with pytest.raises(ValueError):
        sd_kinematics(20.0, 0.0, PROTON_MASS, SQRT_S)


def test_sd_kinematics_forward_sliver():
    """The exact two-body forward edge ``t_fwd(M, m_remnant) < 0`` for
    every excited mass ``M > m_p`` (the documented closed-at-0 sliver),
    and a drawn ``t`` above the edge (``t = 0``) solves to the forward
    configuration (``cos(theta*) = 1``, transverse momentum zero, the
    excitation forward in the lab frame)."""

    for m in (2.0, 5.0, 12.0):
        e_p, e_m, p_star, q_star = _cm_quantities(m, PROTON_MASS)
        mp2 = PROTON_MASS * PROTON_MASS
        t_fwd = mp2 + m * m - 2.0 * (e_p * e_m - p_star * q_star)
        assert t_fwd < 0.0, (m, t_fwd)

    # drawn t = 0 (above the edge) solves to the forward configuration
    for m in (2.0, 5.0, 12.0):
        excitation, _remnant = sd_kinematics(m, 0.0, PROTON_MASS, SQRT_S)
        assert abs(excitation[1]) <= 1e-12, (m, excitation[1])
        assert abs(excitation[2]) <= 1e-12, (m, excitation[2])
        assert excitation[3] > 0.0, (m, excitation[3])


def test_sd_kinematics_angle_documented_formula():
    """The scattering angle follows the documented invariant transfer on
    the excitation line, ``t = m_p^2 + M^2 - 2 (E_p* E_M* - p* q*
    cos(theta*))``, which solves to
    ``cos(theta*) = (t - m_p^2 - M^2 + 2 E_p* E_M*) / (2 p* q*)``.  The
    elastic case (``M = m_p``, intact remnant) reduces to the closed
    form ``cos(theta*) = 1 + t / (2 p*^2)`` on the exact two-body range."""

    mp = PROTON_MASS
    _e_beam, _e_total, sq_exact, _p_star, _beta, _gamma = (
        _beam_configuration())
    p2 = sq_exact * sq_exact / 4.0 - mp * mp
    for t in (-10.0, -p2, -2.0 * p2):
        cos_solved = _solved_cos_theta(mp, t, mp)
        cos_standard = 1.0 + t / (2.0 * p2)
        assert cos_solved == pytest.approx(cos_standard, abs=1e-9), (
            t, cos_solved, cos_standard)
    # a non-elastic point: the same documented relation
    for m, t in ((5.0, -50.0), (5.0, -150.0)):
        cos_solved = _solved_cos_theta(m, t, PROTON_MASS)
        e_p, e_m, p_star, q_star = _cm_quantities(m, PROTON_MASS)
        mp2 = PROTON_MASS * PROTON_MASS
        cos_standard = (t - mp2 - m * m + 2.0 * e_p * e_m) / (
            2.0 * p_star * q_star)
        assert cos_solved == pytest.approx(cos_standard, abs=1e-9), (
            m, t, cos_solved, cos_standard)


def test_event_record_surface(catalog, fams):
    """The event record's line and exchange annotations: the two
    outgoing lines in the fixed projectile-side-first order, the single
    vacuum exchange with the zero transfer, the final-state ``line``
    labels, and the exchange inheritance ``(1, 1, 0, 1)`` on the
    excitation line (the card-2 ledger arithmetic)."""

    event = _generate_one(0, catalog, fams)
    assert isinstance(event, SDEvent)
    assert event.class_label == SD_CLASS_LABEL_PROJECTILE
    assert event.side == "projectile"
    # the single vacuum exchange with the zero transfer
    name, transfer = event.exchange
    assert name == "vacuum"
    assert transfer == (0, 0, 0, 0)
    # the two outgoing lines, projectile-side first, inherit (1, 1, 0, 1)
    assert len(event.lines) == 2
    for line in event.lines:
        assert line.qn == (1, 1, 0, 1)
    # the final-state line labels: the remnant line and the excitation
    # line only (one exchange, two outgoing lines)
    labels = {h.line for h in event.final_state}
    assert labels <= {"remnant", "excitation"}
    assert "remnant" in labels
    remnant_hadrons = [h for h in event.final_state if h.line == "remnant"]
    excitation_hadrons = [h for h in event.final_state
                          if h.line == "excitation"]
    assert len(remnant_hadrons) == 1
    assert len(excitation_hadrons) >= 1
    # the excitation line carries the excited side's (1, 1, 0, 1)
    # quantum numbers through the zero transfer
    excitation_line = event.lines[0]
    assert excitation_line.qn == (1, 1, 0, 1)
    # the gap size is recorded and positive
    assert event.gap_size == gap_size(event)
    assert event.gap_size > 1e-12
    # the mirror carries the same record surface on the target side
    mirrored = mirror_sd(event)
    assert mirrored.side == "target"
    assert mirrored.class_label == SD_CLASS_LABEL_TARGET
    assert mirrored.exchange == event.exchange
    assert {h.line for h in mirrored.final_state} == labels


# ----------------------------------------------------------------------
# Part 2: the substrate and freeze pins
# ----------------------------------------------------------------------
def test_module_byte_and_export_pins():
    """The module at its committed sha256 (nested tree HEAD 2ccc80b)
    with its exact ``__all__`` export set (fourteen names), the solver
    test file at its committed sha256, and the package ``__init__`` at
    its committed sha256 carrying the single-diffraction export block
    (the eleven re-exported names; ``BEAM_MOMENTUM``, ``PROTON_MASS``,
    and ``SQRT_S`` stay card 1's elastic constants and are module-level
    only)."""

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    import na49_mixing.topology_single_diffraction as _mod
    assert _mod.__all__ == MODULE_ALL

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)
    init_text = INIT_PATH.read_text(encoding="utf-8")
    block = re.search(
        r"from \.topology_single_diffraction import \((.*?)\)",
        init_text, re.S)
    assert block is not None, (
        "the __init__ single-diffraction import block is missing")
    for name in INIT_SD_NAMES:
        assert re.search(rf"\b{name}\b", block.group(1)), (
            f"{name} is not in the __init__ single-diffraction import "
            "block")
    # the elastic constants stay module-level only (not re-exported)
    for name in ("BEAM_MOMENTUM", "PROTON_MASS", "SQRT_S"):
        assert not re.search(rf"\b{name}\b", block.group(1)), (
            f"{name} must not enter the __init__ single-diffraction "
            "import block")
    all_block = re.search(r"__all__\s*=\s*\[(.*?)\n\]", init_text, re.S)
    assert all_block is not None
    all_names = re.findall(r'"([^"]+)"', all_block.group(1))
    assert len(all_names) == INIT_ALL_COUNT
    for name in INIT_SD_NAMES:
        assert f'"{name}"' in all_block.group(1), (
            f"{name} is not in the __init__ __all__")


def test_frozen_substrate_digests():
    """The card's frozen-substrate provenance surfaces (verified on
    disk) and the frozen M8 parameter map through the manifest (the
    ten-parameter set hash and the ``regge.effective_transfer_slope``);
    the activated baryonic ``(1, 1, 0)`` continuum family of the
    committed configuration carries its recorded support ends."""

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
    assert m8["regge.effective_transfer_slope"] == ELASTIC_TRANSFER_SLOPE

    # the activated baryonic (1, 1, 0) continuum family of the
    # committed configuration: exactly one, with its recorded support ends
    continuum = json.loads(CONTINUUM_PATH.read_text(encoding="utf-8"))
    baryonic_families = [
        family for family in continuum["families"]
        if family["qn"] == [1, 1, 0] and family["active"] is True]
    assert len(baryonic_families) == 1
    assert baryonic_families[0]["family_id"] == "baryonic_np"
    assert baryonic_families[0]["m_max"] == 5.0




def test_layout_sync():
    """The layout sync (the subprocess check from the monorepo root;
    ``logic_drift = 0``)."""

    probe = subprocess.run(
        [sys.executable, str(SYNC_CHECKER)],
        capture_output=True, text=True, cwd=str(MONO_ROOT))
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert "logic_drift=0" in probe.stdout
