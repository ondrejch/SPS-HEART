"""Round 5, card 4 test-author register: the topology-specific mass laws.

Independent contract pins for the topology-specific mass laws
(``na49_mixing/topology_mass_laws.py``), Round 5 card 4
(``TASK-20260925-r5-topology-mass-laws``; plan section 22 task 34,
line 1759; plan section 11.5, lines 883-894).  This register is the
test-author half of the card: it pins the module contract on its own
pinned fixtures (the named constants with the effective-combination
parameter surface, the exact phase-space limits, the exact two-body t
range closed at zero, the marginal density's closed-form
``epsilon = 0`` law and its grid mass, the draw budgets on the closed
``exchange_kinematics`` substream, the closed-form t draw and the
flat-slope limit, the joint closure with the pinned pathological
rejection cap, and the analytic means) and re-pins the freeze manifest
for the card (``INVENTORY_COUNT`` 513 -> 514).

Part 1: the module contract (pure, hermetic)
--------------------------------------------

- ``test_named_constants_and_effective_surface``: the named module
  constants (the catalog proton mass, the catalog pion mass resolved
  against the committed catalog record, the pinned rejection cap) and
  the effective-combination-only parameter surface (the frozen
  two-field ``MassLawParams`` record with exactly ``(epsilon,
  b_sd)``, the nominal point tied to card 1's elastic slope and the
  M8-frozen ``regge.effective_transfer_slope``, and the white-box scan
  of the module source for the shape-only parameters and the Round 4
  daughter-transverse machinery).
- ``test_sd_limits_pin``: the exact marginal limits
  ``(m_p^2, (sqrt_s - m_remnant)^2)`` at the nominal p+p configuration
  (pinned ``(0.880354927712736, 266.7253293403128)`` GeV^2), the intact
  lower edge at ``m_p^2`` exactly (no minimum N-pi excitation), and
  the checked two-body threshold and finitude refusals.
- ``test_t_range_pin``: the exact two-body t range at the intact limit,
  bit-identical to the elastic two-body limit ``4 m_p^2 - s`` (pinned
  ``-294.7314802891491`` GeV^2) and closed at ``t = 0`` exactly for
  every excitation, with the checked closed-system refusal.
- ``test_marginal_density_closed_form``: the plan-form marginal
  density at ``epsilon = 0``, ``t = 0`` is exactly ``1 / M^2``, the
  grid mass of the closed form equals ``ln(m2_max / m2_min)`` within
  1e-12 on the pinned narrow window, and the grid draw round-trips the
  closed-form CDF within the documented 5e-3 grid-bias bound; the
  checked domain refusals.
- ``test_draw_budget_pins``: exactly 2 uniforms per marginal or
  central ``M^2`` draw, 1 per t draw, and 6 per accepted joint
  double-excitation draw on the closed ``exchange_kinematics`` label
  (the counting proxy, the bit-identical post-draw stream state, and
  the continuous-stream identity over a pinned trace).
- ``test_draw_t_closed_form_and_flat_limit``: the t draw reproduces
  the closed-form quantile of ``exp(b_sd t)`` on ``[t_min, 0]``
  within 1e-9, the flat-slope limit at ``|b_sd| <= 1e-14`` is the
  exact uniform quantile, and the normalization of the pinned intact
  range equals ``1 / B_SD = 0.11758977514325487`` exactly, tied to
  card 1's ``elastic_t_normalization``.
- ``test_joint_closure_pins``: the exact joint closure
  ``M_1 + M_2 + M_central <= sqrt_s`` with the pinned 1e-12
  tolerance at the checked edges, and the pinned pathological
  configuration (individually legal windows, jointly impossible) that
  raises ``PhaseSpaceRejectionExceeded`` at exactly
  ``REJECTION_CAP = 1000`` attempts with exactly 4 uniforms per
  rejected attempt, the message naming the attempt count and the
  parameter point.
- ``test_analytic_mean_pins``: the analytic means
  ``E[M^2] = 46.528050629693524`` GeV^2 on the nominal SD limits and
  ``E[t] = -0.11758977514325486`` GeV^2 on the intact two-body range,
  and the pinned 10,000-draw sample traces within 3 sigma of
  ``1 / sqrt(N)`` at the nominal parameters.

Part 2: the substrate and freeze pins
--------------------------------------

- ``test_module_byte_and_export_pins``: the module at its committed
  sha256, its ``__all__`` export set, the solver test file at its
  committed sha256, and the package ``__init__`` at its committed
  sha256 carrying the mass-laws export block.
- ``test_frozen_substrate_digests``: the card's frozen-substrate
  provenance surfaces (verified on disk) and the frozen M8 parameter
  map through the manifest, with the activated baryonic ``(1, 1, 0)``
  continuum family's recorded support ends.
- ``test_freeze_repin_register``: the freeze-manifest re-pin for this
  card (the strict path-sorted inventory at 516 entries; 514 after
  the Round 5 card 5 re-pin; 513 after the Round 5 card 4 re-pin;
  511 after the Round 5 card 3
  test-author re-pin; 510 after the Round 5 card 3 solver landing;
  508 after the test-protocol tiering change; 505 after the Round 5
  card 2 test-author re-pin; 504 after the Round 5 card 2 solver
  landing; 502 after the Round 5 card 1 test-author re-pin; 501 at
  the Round 5 card 1 landing) carries this
  register and the twenty-eight re-pinned chain count-pin files at
  their on-disk digests, the card 4 solver's freeze exception stays
  exactly once at its 32 path-sorted files, this card's freeze
  exception names the card exactly once with the path-sorted file
  list, the chain count pins are re-pinned 513 -> 514 with the 513
  literal retained in the naming comments, and the frozen M8 parameter
  map stays intact.
- ``test_layout_sync``: the layout sync (the subprocess check from the
  monorepo root; ``logic_drift = 0``).

Hermetic: the module, the committed catalog, the continuum families
configuration, the Round 5 card 1 elastic reference module (read-only),
numpy, and the standard library only; no event generation driver, no
external tables, no Monte Carlo run (the runs R50-R55 exercise the
laws statistically).
"""

from __future__ import annotations

import dataclasses
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
from na49_mixing.rng_substreams import SUBSTREAM_LABELS, derive_substreams
from na49_mixing.topology_elastic import (
    BEAM_MOMENTUM,
    ELASTIC_TRANSFER_SLOPE,
    SQRT_S,
    elastic_t_normalization,
)
from na49_mixing.topology_mass_laws import (
    PI0_MASS,
    PROTON_MASS,
    REJECTION_CAP,
    MassLawParams,
    PhaseSpaceRejectionExceeded,
    draw_central,
    draw_joint_dd,
    draw_marginal,
    draw_t,
    joint_closure,
    marginal_density,
    marginal_grid_mass,
    sd_limits,
    t_range,
)
from tests._freeze_inventory import inventory_count as _inventory_count

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent          # dev/mixing2006
MONO_ROOT = NESTED_ROOT.parent.parent          # the monorepo root
MANIFEST_PATH = NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json"
SYNC_CHECKER = MONO_ROOT / "helpers" / "benchmarking" / "check_helpers_layout_sync.py"
MODULE_PATH = NESTED_ROOT / "na49_mixing" / "topology_mass_laws.py"
INIT_PATH = NESTED_ROOT / "na49_mixing" / "__init__.py"
SOLVER_TEST_PATH = NESTED_ROOT / "tests" / "test_topology_mass_laws.py"

MY_TEST_REL = (
    "dev/mixing2006/tests/test_topology_mass_laws_contract_pins.py")

#: The committed catalog and its pinned digest (the card's
#: frozen-substrate provenance; the pion mass pin is pinned at this
#: digest).
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
CATALOG_SHA256 = (
    "cf9e79b99199c509f2e83def0a99c215f73ca7761f9f87fb01d1f779b80e32f3")
CONTINUUM_PATH = (
    NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json")

#: The Round 5 master seed (round 5 index item 9), fixed for the pinned
#: traces; replica 0, on the closed ``exchange_kinematics`` substream.
MASTER_SEED = 20260925

#: The pinned trace sizes (the 10,000-draw sample-mean traces).
N_TRACE = 10_000

#: The nominal effective parameters (the M8-frozen
#: ``regge.effective_transfer_slope`` as the R50 nominal B_SD, index
#: item 2; nominal epsilon 0.0).
B_SD_NOMINAL = 8.504140762083612
NOMINAL_PARAMS = MassLawParams(0.0, B_SD_NOMINAL)

#: The nominal M8 p+p collision energy (the elastic module's recorded
#: quote; the register's phase-space configuration).
SQRT_S_PIN = 17.27

#: The exact marginal limits at the nominal p+p configuration and the
#: intact other line (pinned at the nominal configuration).
M2_MIN_PIN = 1.151863222816188  # (m_p + m_pi0)^2: the N pi threshold edge (2026-10-02)
M2_MAX_PIN = 266.7253293403128

#: The exact two-body t range at the intact limit (the elastic
#: two-body limit ``4 m_p^2 - s``, bit-identical).
T_MIN_INTACT_PIN = -294.7314802891491

#: The t-law normalization of the pinned intact range (both the
#: closed-form normalizer and card 1's ``elastic_t_normalization``
#: reduce to ``1 / B_SD`` at the underflowed edge).
Z_PIN = 0.11758977514325487

#: The analytic means pinned on the nominal windows: the flat-in-ln
#: M^2 mean of ``[M2_MIN_PIN, M2_MAX_PIN]`` and the mean of
#: ``exp(b_sd t)`` on ``[T_MIN_INTACT_PIN, 0]``.
E_M2_PIN = 48.775268994477855  # on the N pi threshold limits (2026-10-02)
E_T_PIN = -0.11758977514325486

#: The pinned epsilon = 0, c = 0 closed-form window: a 5e-3 GeV^2-wide
#: slice at the physical lower edge (the grid mass equals the
#: logarithmic integral within 1e-12 there).
CLOSED_FORM_WINDOW = (M2_MIN_PIN, M2_MIN_PIN + 5e-3)

#: The pinned closed-form round-trip quantile grid (the documented
#: 5e-3 grid-bias bound, mirroring the solver test's grid).
ROUND_TRIP_QUANTILES = tuple(np.linspace(0.02, 0.98, 33))

#: The committed module, solver test, and package digests (nested tree
#: HEAD b6275d0).
MODULE_SHA256 = (
    "6af0d8fbdf67204e316c42cae8e9962fb87251000b9e1364135045227224db0a")
SOLVER_TEST_SHA256 = (
    "9249e30d9547222397b25ff47e8af9bf3d2d8e3d8396a675ffd53ceb3b260b45")
#: The package digest re-pinned by Round 5 card 4
#: (TASK-20260925-r5-topology-mass-laws; the mass-laws module's
#: additive package exports).
# re-pinned by Round 5 card 7 (TASK-20260925-r5-non-diffrative-central-excitation; the non-diffractive module's additive package exports).
# re-pinned by Round 5 card 8 (TASK-20260925-r5-multiple-neutral-excitations; the multiple-neutral-excitation module's additive package exports).
# re-pinned by Round 5 card 9 (TASK-20260925-r5-topology-template-runner; the topology-runner module's additive package exports).
# re-pinned by the Round 5 card 9 review fix round (TASK-20260925-r5-topology-template-runner-review-fix; the REV-bc4de47-03 export-surface correction re-pinned the package __init__).
INIT_SHA256 = (
    "7d2c5a85d9b24de104c0d9cf72195689bf7c39c873e154798699448073b36de8")

#: The module's exact ``__all__`` export set (sixteen names; the N pi
#: threshold edge EXCITATION_M_MIN joined on 2026-10-02, the tail's
#: smooth-edge draw draw_marginal_edge on 2026-10-25).
MODULE_ALL = [
    "PROTON_MASS",
    "PI0_MASS",
    "EXCITATION_M_MIN",
    "REJECTION_CAP",
    "PhaseSpaceRejectionExceeded",
    "MassLawParams",
    "sd_limits",
    "t_range",
    "marginal_density",
    "marginal_grid_mass",
    "draw_marginal",
    "draw_marginal_edge",
    "draw_t",
    "joint_closure",
    "draw_joint_dd",
    "draw_central",
]

#: The mass-laws export block of the package ``__init__`` (the thirteen
#: names the package re-exports; ``PROTON_MASS`` stays card 1's elastic
#: constant and is module-level only in this module, additive exports
#: only).
INIT_MASS_LAWS_NAMES = [
    "PI0_MASS",
    "REJECTION_CAP",
    "MassLawParams",
    "PhaseSpaceRejectionExceeded",
    "draw_central",
    "draw_joint_dd",
    "draw_marginal",
    "draw_t",
    "joint_closure",
    "marginal_density",
    "marginal_grid_mass",
    "sd_limits",
    "t_range",
]

#: The M8-frozen kernel boundary, pinned through the manifest's frozen
#: parameter map.
KERNEL_BOUNDARY_PIN = 2.680194110727528

#: The M8-frozen ten-parameter set hash (the sha256 of the manifest's
#: frozen parameter map, ``json.dumps(..., sort_keys=True,
#: separators=(",", ":"))``; the card's provenance pin).
M8_HASH = "358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1"

#: The card's frozen-substrate provenance pins (byte-stable surfaces,
#: verified on disk): the builder blend path, the card-6 blend routing,
#: the card-2 engine, the committed catalog, the continuum families
#: configuration, the sealed Round 4 R45/R46 drivers in both helper
#: trees, and the three Round 5 modules consumed read-only (card 1's
#: elastic reference, card 2's exchange ledger, card 3's remnants).
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

#: The white-box tokens of the parameters that enter only through one
#: effective shape and of the Round 4 daughter-transverse machinery
#: (plan line 894; plan section 24, line 1866: none of these may appear
#: anywhere in the module source).
SHAPE_ONLY_TOKENS = (
    "intercept", "trajectory", "residue", "daughter_pt", "daughter p_T",
    "b_dec", "form_factor", "cascade", "alpha_exc", "edge_power",
    "kappa_b", "kappa_m",
)
# "kappa" alone left the scan on 2026-10-08: the factorized vertex slopes
# (vertex_slope_factor) take a pre-registered exponent kappa as a kernel
# option of the topology runner.  MassLawParams keeps exactly epsilon and
# b_sd, so the fitted surface is unchanged.

#: The freeze-manifest inventory entry count after this card's
#: re-pin (513 after the Round 5 card 4 solver landing; +1 entry:
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
#: The solver card (Round 5, card 4) and this test-author register's
#: card name (the freeze-exception identity).
SOLVER_CARD = "TASK-20260925-r5-topology-mass-laws"
TEST_AUTHOR_CARD = (
    "TASK-20260925-r5-topology-mass-laws-test-author")

#: The register chain: the twenty-eight chain count-pin files re-pinned
#: 513 -> 514 by this register, 514 -> 516 by Round 5 card 5, and 516 -> 517 by the Round 5 card 5 test-author (the
#: twenty-six card-2 chain files plus
#: the card-2 register itself plus the card-3 register itself,
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
)

#: This register's freeze-exception file list (path-sorted): the
#: manifest, this register, and the twenty-eight re-pinned chain
#: count-pin files.
TEST_AUTHOR_EXCEPTION_FILES = tuple(sorted(
    ("dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json",)
    + CHAIN_COUNT_PIN_FILES
    + (MY_TEST_REL,)
))


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exchange_stream_id(event_index: int) -> int:
    """The pinned exchange_kinematics stream of one trace."""

    return derive_substreams(MASTER_SEED, 0, event_index).stream_for(
        "exchange_kinematics")


def _exchange_rng(event_index: int) -> np.random.Generator:
    """The pinned exchange_kinematics generator of one trace."""

    return np.random.default_rng(_exchange_stream_id(event_index))


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


class _ScriptedUniforms:
    """A scripted-uniform stub: ``random()`` returns the scripted
    values in order (the round-trip probes drive the draws with the
    pinned quantile grid this way)."""

    def __init__(self, values) -> None:
        self._values = [float(v) for v in values]
        self._next = 0

    def random(self) -> float:
        if self._next >= len(self._values):
            raise AssertionError("the scripted uniform sequence is exhausted")
        value = self._values[self._next]
        self._next += 1
        return value


def _eps0_cdf(x: float, a: float, b: float) -> float:
    """The closed-form CDF of the epsilon = 0, c = 0 law (the
    flat-in-ln M^2 law)."""

    return math.log(x / a) / math.log(b / a)


# ----------------------------------------------------------------------
# Part 1: the module contract
# ----------------------------------------------------------------------
def test_named_constants_and_effective_surface():
    """The named module constants (the catalog proton mass, the catalog
    pion mass resolved against the committed catalog record, the pinned
    rejection cap) and the effective-combination-only parameter surface
    (the frozen two-field ``MassLawParams`` record with exactly
    ``(epsilon, b_sd)``, the nominal point tied to card 1's elastic
    slope and the M8-frozen ``regge.effective_transfer_slope``, and
    the white-box scan of the module source for the shape-only
    parameters and the Round 4 daughter-transverse machinery)."""

    # the named constants
    assert PROTON_MASS == 0.93827231
    assert isinstance(PROTON_MASS, float)
    assert REJECTION_CAP == 1000
    assert isinstance(REJECTION_CAP, int)
    assert isinstance(PI0_MASS, float)

    # the pion mass is the committed catalog value (the card's
    # interface carried the 0.1349768 quote, resolved to the committed
    # catalog record pdg_id 111)
    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)
    catalog = HadronCatalog.from_json(CATALOG_PATH)
    pi0 = catalog.get(111)
    assert int(pi0.pdg_id) == 111
    assert float(pi0.pole_mass) == PI0_MASS == 0.1349766

    # the effective parameter surface: exactly the two fields
    assert getattr(MassLawParams, "__dataclass_params__").frozen
    assert tuple(MassLawParams.__dataclass_fields__) == ("epsilon", "b_sd")
    assert isinstance(MassLawParams(0.0, B_SD_NOMINAL).epsilon, float)
    assert isinstance(MassLawParams(0.0, B_SD_NOMINAL).b_sd, float)

    # the nominal point: the M8-frozen effective_transfer_slope, tied
    # to card 1's elastic slope (one slope serves all classes, plan
    # section 13.1 block A)
    assert B_SD_NOMINAL == 8.504140762083612
    assert B_SD_NOMINAL == ELASTIC_TRANSFER_SLOPE
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    m8 = manifest["frozen_fit_outputs"]["m8_frozen_pp_parameters"]
    assert m8["regge.effective_transfer_slope"] == B_SD_NOMINAL

    # white-box: no shape-only parameter, no Round 4
    # daughter-transverse machinery, only the shared exact two-body
    # arithmetic (plan line 894; plan section 24, line 1866)
    source = MODULE_PATH.read_text(encoding="utf-8")
    for token in SHAPE_ONLY_TOKENS:
        assert token not in source, token
    assert "from .excitation_kinematics import kallen" in source


def test_sd_limits_pin():
    """The exact marginal limits ``(m_p^2, (sqrt_s - m_remnant)^2)`` at
    the nominal p+p configuration (pinned ``(0.880354927712736,
    266.7253293403128)`` GeV^2), the intact lower edge at ``m_p^2``
    exactly (no minimum N-pi excitation, plan line 842), and the
    checked two-body threshold and finitude refusals."""

    assert SQRT_S == SQRT_S_PIN == 17.27
    assert sd_limits(PROTON_MASS, SQRT_S) == (M2_MIN_PIN, M2_MAX_PIN) == (
        1.151863222816188, 266.7253293403128)

    # the lower edge: the N pi threshold (m_p + m_pi0)^2 exactly, for
    # every remnant mass (it was the intact m_p^2 until 2026-10-02)
    limits = sd_limits(PROTON_MASS, SQRT_S)
    assert limits[0] == (PROTON_MASS + PI0_MASS) * (PROTON_MASS + PI0_MASS) == M2_MIN_PIN
    assert limits[1] == (SQRT_S - PROTON_MASS) ** 2 == M2_MAX_PIN
    # the lower edge is the intact mass for a non-proton remnant as
    # well (no minimum N-pi excitation; the marginal reaches down to
    # the intact mass of the other line)
    limits_pi = sd_limits(PI0_MASS, SQRT_S)
    assert limits_pi[0] == M2_MIN_PIN
    assert limits_pi[1] == (SQRT_S - PI0_MASS) ** 2

    # the checked refusals: the two-body threshold (checked, never
    # repaired) and finitude
    with pytest.raises(ValueError, match="two-body threshold"):
        sd_limits(PROTON_MASS, PROTON_MASS + 1e-9)
    with pytest.raises(ValueError, match="two-body threshold"):
        sd_limits(PROTON_MASS, PROTON_MASS)
    with pytest.raises(ValueError, match="finite GeV value"):
        sd_limits(PROTON_MASS, float("nan"))
    with pytest.raises(ValueError, match="positive finite"):
        sd_limits(float("inf"), SQRT_S)


def test_t_range_pin():
    """The exact two-body t range at the intact limit, bit-identical to
    the elastic two-body limit ``4 m_p^2 - s`` (pinned
    ``-294.7314802891491`` GeV^2) and closed at ``t = 0`` exactly for
    every excitation, with the checked closed-system refusal."""

    # the intact limit: the elastic two-body limit, bit-identical,
    # closed at t = 0 exactly
    # the intact limit sits at M = m_p (below the N pi threshold edge of
    # the marginal since 2026-10-02)
    assert t_range(PROTON_MASS * PROTON_MASS, PROTON_MASS, SQRT_S) == (
        T_MIN_INTACT_PIN, 0.0) == (-294.7314802891491, 0.0)
    assert T_MIN_INTACT_PIN == 4.0 * PROTON_MASS * PROTON_MASS - SQRT_S_PIN ** 2

    # the closed-at-0 upper edge for every excitation: the second
    # element is exactly 0.0, and the lower edge stays inside
    # [t_min_intact, 0)
    for excitation in (1.1, 2.5, 8.0, 16.0):
        t_low, t_high = t_range(excitation * excitation, PROTON_MASS,
                                SQRT_S)
        assert t_high == 0.0
        assert T_MIN_INTACT_PIN <= t_low < 0.0

    # the limits contract: the closed system is refused (checked,
    # never repaired), naming the values
    with pytest.raises(ValueError, match="two-body system is closed"):
        t_range(SQRT_S_PIN * SQRT_S_PIN, PROTON_MASS, SQRT_S)
    with pytest.raises(ValueError, match="positive finite"):
        t_range(float("nan"), PROTON_MASS, SQRT_S)
    # below threshold the closure check fires before the range is
    # computed (checked, never repaired), naming the values
    with pytest.raises(ValueError, match="two-body system is closed"):
        t_range(M2_MIN_PIN, PROTON_MASS, -1.0)


def test_marginal_density_closed_form():
    """The plan-form marginal density at ``epsilon = 0``, ``t = 0`` is
    exactly ``1 / M^2`` (the flat-in-ln M^2 law), the grid mass of the
    closed form equals ``ln(m2_max / m2_min)`` within 1e-12 on the
    pinned narrow window, and the grid draw round-trips the closed-form
    CDF within the documented 5e-3 grid-bias bound; the checked domain
    refusals."""

    # the exact 1 / M^2 law at epsilon = 0, t = 0
    for x in (M2_MIN_PIN, 1.0, 2.0, 17.0, 100.0, M2_MAX_PIN):
        assert marginal_density(x, 0.0, B_SD_NOMINAL, 0.0) == pytest.approx(
            1.0 / x, rel=4e-16)
    # the t = 0 factor is exactly exp(0) = 1 for every slope
    assert marginal_density(2.0, 0.0, B_SD_NOMINAL, 0.0) == pytest.approx(
        0.5, rel=4e-16)

    # the closed-form grid mass: the logarithmic integral within 1e-12
    # on the pinned narrow window
    a, b = CLOSED_FORM_WINDOW
    grid = marginal_grid_mass(a, b, 0.0, B_SD_NOMINAL, 0.0)
    closed = math.log(b / a)
    assert abs(grid - closed) <= 1e-12 * closed, (grid, closed)

    # the grid draw round-trips the closed-form CDF within the
    # documented 5e-3 grid-bias bound (the segment uniform driven by
    # the pinned quantile grid, the cell center as the within-cell
    # uniform)
    for u in ROUND_TRIP_QUANTILES:
        stub = _ScriptedUniforms([u, 0.5])
        value = draw_marginal(a, b, NOMINAL_PARAMS, 0.0, stub)
        cdf = _eps0_cdf(value, a, b)
        assert abs(cdf - u) <= 5e-3, (u, value, cdf)

    # the checked domain refusals: positive finite m2, finite
    # parameters, a strict window
    with pytest.raises(ValueError, match="positive finite"):
        marginal_density(0.0, 0.0, B_SD_NOMINAL, 0.0)
    with pytest.raises(ValueError, match="positive finite"):
        marginal_density(float("-inf"), 0.0, B_SD_NOMINAL, 0.0)
    with pytest.raises(ValueError, match="finite"):
        marginal_density(1.0, float("nan"), B_SD_NOMINAL, 0.0)
    with pytest.raises(ValueError, match="0 < m2_min"):
        marginal_grid_mass(1.0, 1.0, 0.0, B_SD_NOMINAL, 0.0)
    with pytest.raises(ValueError, match="0 < m2_min"):
        marginal_grid_mass(2.0, 1.0, 0.0, B_SD_NOMINAL, 0.0)


def test_draw_budget_pins():
    """Exactly 2 uniforms per marginal or central ``M^2`` draw, 1 per
    t draw, and 6 per accepted joint double-excitation draw on the
    closed ``exchange_kinematics`` label (the counting proxy, the
    bit-identical post-draw stream state, and the continuous-stream
    identity over a pinned trace)."""

    assert "exchange_kinematics" in SUBSTREAM_LABELS

    # the marginal draw: exactly 2 uniforms (the segment uniform plus
    # the within-cell uniform)
    sid = _exchange_stream_id(2)
    counter = _CountingRng(np.random.default_rng(sid))
    draw_marginal(M2_MIN_PIN, M2_MAX_PIN, NOMINAL_PARAMS, 0.0, counter)
    assert counter.calls == 2
    advanced = np.random.default_rng(sid)
    advanced.random()
    advanced.random()
    assert counter._source.bit_generator.state == advanced.bit_generator.state

    # the t draw: exactly 1 uniform
    sid = _exchange_stream_id(3)
    counter = _CountingRng(np.random.default_rng(sid))
    draw_t(T_MIN_INTACT_PIN, NOMINAL_PARAMS, counter)
    assert counter.calls == 1
    advanced = np.random.default_rng(sid)
    advanced.random()
    assert counter._source.bit_generator.state == advanced.bit_generator.state

    # the central draw: exactly 2 uniforms (the same functional form
    # with the total transfer t_c)
    sid = _exchange_stream_id(4)
    counter = _CountingRng(np.random.default_rng(sid))
    draw_central(M2_MIN_PIN, 2.0 * M2_MIN_PIN, NOMINAL_PARAMS, -1.0, counter)
    assert counter.calls == 2
    advanced = np.random.default_rng(sid)
    advanced.random()
    advanced.random()
    assert counter._source.bit_generator.state == advanced.bit_generator.state

    # the joint DD draw: exactly 6 uniforms per accepted draw (two
    # marginals and two transfers).  The half-energy windows are
    # always accepted (M_1 + M_2 <= sqrt_s by construction), so the
    # first attempt is accepted deterministically and a 50-draw trace
    # consumes exactly 6 per draw with nothing in between.
    half_window = (SQRT_S_PIN / 2.0) ** 2
    sid = _exchange_stream_id(5)
    counter = _CountingRng(np.random.default_rng(sid))
    m2_1, m2_2, t_1, t_2 = draw_joint_dd(
        M2_MIN_PIN, half_window, M2_MIN_PIN, half_window,
        SQRT_S_PIN, NOMINAL_PARAMS, counter)
    assert counter.calls == 6
    assert math.sqrt(m2_1) + math.sqrt(m2_2) <= SQRT_S_PIN + 1e-12
    assert 0.0 >= t_1 >= t_range(m2_1, PROTON_MASS, SQRT_S)[0]
    assert 0.0 >= t_2 >= t_range(m2_2, PROTON_MASS, SQRT_S)[0]
    advanced = np.random.default_rng(sid)
    for _ in range(6):
        advanced.random()
    assert counter._source.bit_generator.state == advanced.bit_generator.state
    total_calls = 6
    for _ in range(50):
        m2_1, m2_2, _t_1, _t_2 = draw_joint_dd(
            M2_MIN_PIN, half_window, M2_MIN_PIN, half_window,
            SQRT_S_PIN, NOMINAL_PARAMS, counter)
        assert math.sqrt(m2_1) + math.sqrt(m2_2) <= SQRT_S_PIN + 1e-12
    total_calls += 50 * 6
    assert counter.calls == total_calls


def test_draw_t_closed_form_and_flat_limit():
    """The t draw reproduces the closed-form quantile of ``exp(b_sd
    t)`` on ``[t_min, 0]`` within 1e-9, the flat-slope limit at
    ``|b_sd| <= 1e-14`` is the exact uniform quantile, and the
    normalization of the pinned intact range equals ``1 / B_SD =
    0.11758977514325487`` exactly, tied to card 1's
    ``elastic_t_normalization``."""

    # the closed-form round trip on the pinned intact two-body range
    e_lo = math.exp(B_SD_NOMINAL * T_MIN_INTACT_PIN)
    for u in ROUND_TRIP_QUANTILES:
        stub = _ScriptedUniforms([u])
        value = draw_t(T_MIN_INTACT_PIN, NOMINAL_PARAMS, stub)
        expected = math.log(e_lo + u * (1.0 - e_lo)) / B_SD_NOMINAL
        assert abs(value - expected) <= 1e-9, (u, value, expected)

    # the flat-slope limit at |b_sd| <= 1e-14: the uniform quantile on
    # the range, exactly (the zero-slope law and a slope at the
    # threshold)
    for slope in (0.0, 1e-15, -1e-15):
        flat = MassLawParams(0.0, slope)
        for u in (0.0, 0.25, 0.5, 0.999999):
            stub = _ScriptedUniforms([u])
            assert draw_t(T_MIN_INTACT_PIN, flat, stub) == (
                T_MIN_INTACT_PIN + u * (0.0 - T_MIN_INTACT_PIN))

    # the closed-form normalization of the pinned range: both the t
    # draw's normalizer and card 1's elastic_t_normalization reduce to
    # 1 / B_SD at the underflowed edge (the t factor acts on the
    # exchange kinematics only, with the identifiable constant
    # effective slope)
    z_intact = (1.0 - math.exp(B_SD_NOMINAL * T_MIN_INTACT_PIN)) / B_SD_NOMINAL
    assert z_intact == Z_PIN == 1.0 / B_SD_NOMINAL == 0.11758977514325487
    t_min_elastic_nominal = -((2.0 * PROTON_MASS * BEAM_MOMENTUM
                               / SQRT_S_PIN) ** 2)
    assert elastic_t_normalization(t_min_elastic_nominal) == Z_PIN

    # the checked domain refusal: t_min must be a finite non-positive
    # value
    with pytest.raises(ValueError, match="finite non-positive"):
        draw_t(1.0, NOMINAL_PARAMS, _exchange_rng(0))
    with pytest.raises(ValueError, match="MassLawParams"):
        draw_t(T_MIN_INTACT_PIN, (0.0, B_SD_NOMINAL), _exchange_rng(0))


def test_joint_closure_pins():
    """The exact joint closure ``M_1 + M_2 + M_central <= sqrt_s`` with
    the pinned 1e-12 tolerance at the checked edges, and the pinned
    pathological configuration (individually legal windows, jointly
    impossible) that raises ``PhaseSpaceRejectionExceeded`` at exactly
    ``REJECTION_CAP = 1000`` attempts with exactly 4 uniforms per
    rejected attempt, the message naming the attempt count and the
    parameter point."""

    assert REJECTION_CAP == 1000
    assert issubclass(PhaseSpaceRejectionExceeded, RuntimeError)

    # the closure's exact edges (the 1e-12 tolerance at the edges)
    assert joint_closure(1.0, 1.0, SQRT_S_PIN) is True
    assert joint_closure(10.0, 10.0, SQRT_S_PIN) is False
    assert joint_closure(SQRT_S_PIN, 0.0, SQRT_S_PIN) is True
    assert joint_closure(SQRT_S_PIN + 2e-12, 0.0, SQRT_S_PIN) is False
    # the central-production pass-through: the central mass enters the
    # same closure
    assert joint_closure(8.0, 8.0, SQRT_S_PIN, m_central=1.0) is True
    assert joint_closure(8.0, 8.0, SQRT_S_PIN, m_central=2.0) is False

    # the checked finitude refusal
    with pytest.raises(ValueError, match="finite"):
        joint_closure(float("nan"), 1.0, SQRT_S_PIN)

    # the pinned pathological configuration: each marginal on the
    # window [(sqrt_s - m_p)^2, s] -- the individually legal windows
    # are jointly impossible (M_1 + M_2 >= 2 (sqrt_s - m_p) > sqrt_s),
    # so the cap fires at exactly REJECTION_CAP attempts with exactly
    # 4 uniforms per rejected attempt (two marginals; the closure is
    # checked before the transfers)
    pathological_min = (SQRT_S_PIN - PROTON_MASS) ** 2
    pathological_max = SQRT_S_PIN * SQRT_S_PIN
    rng = _exchange_rng(6)
    state0 = rng.bit_generator.state
    with pytest.raises(PhaseSpaceRejectionExceeded) as excinfo:
        draw_joint_dd(pathological_min, pathological_max,
                      pathological_min, pathological_max,
                      SQRT_S_PIN, MassLawParams(0.0, 0.0), rng)
    message = str(excinfo.value)
    assert "1000 attempts" in message
    assert "epsilon=0.0" in message and "b_sd=0.0" in message
    reference = _exchange_rng(6)
    for _ in range(4 * REJECTION_CAP):
        reference.random()
    assert rng.bit_generator.state == reference.bit_generator.state
    assert rng.bit_generator.state != state0


def test_analytic_mean_pins():
    """The analytic means ``E[M^2] = 46.528050629693524`` GeV^2 on the
    nominal SD limits and ``E[t] = -0.11758977514325486`` GeV^2 on the
    intact two-body range, and the pinned 10,000-draw sample traces
    within 3 sigma of ``1 / sqrt(N)`` at the nominal parameters."""

    # the closed-form means
    e_mean_m2 = (M2_MAX_PIN - M2_MIN_PIN) / math.log(M2_MAX_PIN / M2_MIN_PIN)
    assert e_mean_m2 == E_M2_PIN == 48.775268994477855
    e_second = (M2_MAX_PIN * M2_MAX_PIN - M2_MIN_PIN * M2_MIN_PIN) / (
        2.0 * math.log(M2_MAX_PIN / M2_MIN_PIN))
    e_std_m2 = math.sqrt(e_second - e_mean_m2 * e_mean_m2)

    e_lo = math.exp(B_SD_NOMINAL * T_MIN_INTACT_PIN)
    z_norm = (1.0 - e_lo) / B_SD_NOMINAL
    first_moment = ((0.0 - 1.0 / B_SD_NOMINAL ** 2)
                    - e_lo * (T_MIN_INTACT_PIN / B_SD_NOMINAL
                              - 1.0 / B_SD_NOMINAL ** 2))
    e_mean_t = first_moment / z_norm
    assert e_mean_t == E_T_PIN == -0.11758977514325486
    e_std_t = 1.0 / B_SD_NOMINAL

    # the pinned 10,000-draw marginal trace (event index 0) within 3
    # sigma of 1 / sqrt(N)
    window = 3.0 * e_std_m2 / math.sqrt(N_TRACE)
    rng = _exchange_rng(0)
    total = 0.0
    for _ in range(N_TRACE):
        total += draw_marginal(M2_MIN_PIN, M2_MAX_PIN, NOMINAL_PARAMS, 0.0,
                               rng)
    sample_mean = total / N_TRACE
    deviation = sample_mean - e_mean_m2
    assert abs(deviation) <= window, (
        f"the nominal M^2 sample mean deviates by {deviation!r} GeV^2 "
        f"({deviation / (e_std_m2 / math.sqrt(N_TRACE)):.3f} sigma), "
        f"outside the 3-sigma window {window!r}")

    # the pinned 10,000-draw t trace (event index 1) within 3 sigma of
    # 1 / sqrt(N)
    window = 3.0 * e_std_t / math.sqrt(N_TRACE)
    rng = _exchange_rng(1)
    total = 0.0
    for _ in range(N_TRACE):
        total += draw_t(T_MIN_INTACT_PIN, NOMINAL_PARAMS, rng)
    sample_mean = total / N_TRACE
    deviation = sample_mean - e_mean_t
    assert abs(deviation) <= window, (
        f"the nominal t sample mean deviates by {deviation!r} GeV^2 "
        f"({deviation / (e_std_t / math.sqrt(N_TRACE)):.3f} sigma), "
        f"outside the 3-sigma window {window!r}")


# ----------------------------------------------------------------------
# Part 2: the substrate and freeze pins
# ----------------------------------------------------------------------
def test_module_byte_and_export_pins():
    """The module at its committed sha256 (nested tree HEAD b6275d0)
    with its exact ``__all__`` export set (fourteen names), the solver
    test file at its committed sha256, and the package ``__init__`` at
    its committed sha256 carrying the mass-laws export block (the
    thirteen re-exported names; ``PROTON_MASS`` stays card 1's elastic
    constant and is module-level only)."""

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    import na49_mixing.topology_mass_laws as _mod
    assert _mod.__all__ == MODULE_ALL

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)
    init_text = INIT_PATH.read_text(encoding="utf-8")
    block = re.search(
        r"from \.topology_mass_laws import \((.*?)\)", init_text, re.S)
    assert block is not None, (
        "the __init__ mass-laws import block is missing")
    for name in INIT_MASS_LAWS_NAMES:
        assert re.search(rf"\b{name}\b", block.group(1)), (
            f"{name} is not in the __init__ mass-laws import block")
    # the package keeps card 1's PROTON_MASS; the mass-laws module
    # constant stays module-level only
    assert not re.search(r"\bPROTON_MASS\b", block.group(1)), (
        "PROTON_MASS must not enter the __init__ mass-laws import block")
    all_block = re.search(r"__all__\s*=\s*\[(.*?)\n\]", init_text, re.S)
    assert all_block is not None
    for name in INIT_MASS_LAWS_NAMES:
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
    assert m8["continuum.match_mass"] == KERNEL_BOUNDARY_PIN
    assert m8["regge.effective_transfer_slope"] == B_SD_NOMINAL

    # the activated baryonic (1, 1, 0) continuum family of the
    # committed configuration: exactly one, with its support ends at
    # the frozen boundary and the card-6 surface's upper end
    continuum = json.loads(CONTINUUM_PATH.read_text(encoding="utf-8"))
    baryonic_families = [
        family for family in continuum["families"]
        if family["qn"] == [1, 1, 0] and family["active"] is True]
    assert len(baryonic_families) == 1
    assert baryonic_families[0]["family_id"] == "baryonic_np"
    assert baryonic_families[0]["m_min"] == KERNEL_BOUNDARY_PIN
    assert baryonic_families[0]["m_max"] == 5.0




def test_layout_sync():
    """The layout sync (the subprocess check from the monorepo root;
    ``logic_drift = 0``)."""

    probe = subprocess.run(
        [sys.executable, str(SYNC_CHECKER)],
        capture_output=True, text=True, cwd=str(MONO_ROOT))
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert "logic_drift=0" in probe.stdout
