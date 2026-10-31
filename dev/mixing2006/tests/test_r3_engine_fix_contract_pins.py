"""Round 3 card 8a phase 2: engine-fix contract pins (test-author
register).

Task ``TASK-20260918-r3-card2-engine-three-body-moment-fix`` (phase 2,
the live re-evaluation phase; nested P1 commit ``95096d2``).  This is
the test-author's independent pin register for the sealed run evidence
under ``dev/00runs/r3_threshold_metric_engine_fix_2026-09-18/`` and for
the fixed engine's contract, an independent cross-check of the solver's
register in
``tests/test_staged_r3_threshold_metric_engine_fix_pins.py`` (the same
sealed values, pinned from an independently authored test).

Part 1 (sealed-run artifact pins, read-only; skipped when the
gitignored run dir is absent on a fresh checkout):

- ``test_sealed_run_dir_inventory``: exactly the nine top-level files.
- ``test_sha256_sums``: six monorepo-root-relative entries, the pinned
  digests, lexicographic order, every digest verifies against disk from
  the monorepo root.
- ``test_seal_manifest``: six sealed payloads totaling 1,901,206 bytes,
  the sums-file digest, the documented exclusions.
- ``test_pre_registration_structure``: master seed 20260918 with
  ``draws: 0``, the fixed engine sha256 and P1 commit, the metric
  contract (``delta_analytic_full`` at the unchanged 3.0-sigma
  tolerance, the deterministic-SE floor 1e-12 equal to the vacuous-zero
  threshold), the sealed baseline (11 residual failures, pass rate
  0.9659442724458205, max 4.984844257802041), the coverage (834 applied
  over the 323 original R21/R22 failures, 422 applied over the 14
  original R31 failures), the T1/T2 acceptance thresholds,
  ``written_before_evaluation``, the nine declared outputs.
- ``test_provenance_structure``: the engine/catalog/archives/R31/
  baseline/driver digests, the seal states, the git identity, the
  environment fingerprint.
- ``test_live_reeval_r21_summary``: 351 applied / 342 pass / 9 fail,
  169 original failures with 160 now passing at rate
  0.9467455621301775, the per-pair-class breakdown, max sigma
  3.5992208640500682, the engine-versus-sealed-stored drift, the named
  residual census (four genuine open-set-change, three
  moment-approximation-floor, three marginal), and the floor-residual
  rows (the live payload records the ORIGINAL sealed sigmas, 174.22 /
  85.70 / 7.15).
- ``test_live_reeval_r22_summary``: 483 applied / 480 pass / 3 fail over
  the 1458-row check census, 154 original failures with 151 now
  passing at rate 0.9805194805194806, max sigma 4.973562263179045.
- ``test_live_reeval_r31_targeted_summary``: 422 applied = 299 pass +
  113 vacuous zero + 10 fail, the 14-row original-failure table pinned
  row by row (the two Lambda(1520) three-body checks resolved to
  +0.9201361176029509 and +0.36971803795576264, the two float-precision
  SE artifacts vacuous at sigma 0.0, ten persisting at their sealed
  sigmas), max sigma 7.01270490790188, the boundary-straddling pairs
  [2, 6, 10], the boundary-interior split.
- ``test_gate2_revised_clause``: the tolerance block (3.0, unchanged,
  floor 1e-12, relaxed false), the sealed baseline alongside the
  post-fix values (combined 834 applied / 822 pass / 12 fail, pass rate
  0.9628482972136223), the acceptance verdicts (``pass == false`` for
  T1 and T2 with every sub-verdict false; the four floor residuals at
  their sealed RE-EVALUATION sigmas 3.3622721646606926 /
  3.594992035740298 / 3.0846781101915357 / 4.984844257802041 against the
  live 3.362272164659226 / 3.5992208640500682 / 3.0407611711227656 /
  4.973562263179045), the twelve named residual findings,
  ``no_fitting``.
- ``test_clean_environment_record``: the seal re-verified (6 files, no
  mismatches), the clean-interpreter re-run (``python -I``), the four
  sealed report digests equal to the recomputed digests
  (``hashes_match true``).

Part 2 (pure engine contract, no sealed artifacts, always runs):

- ``test_engine_contract_constants``: ``SIGMA_TOLERANCE == 3.0``,
  ``DETERMINISTIC_SE_FLOOR == 1e-12``, ``ZERO_RESIDUAL_TOL == 1e-12``,
  and the module digest of the fixed engine.
- ``test_kallen_unequal_mass_slot_kinematics``: the exact Kallen
  leading-pair kinematics on an unequal-mass three-body system (every
  slot-0 node on-shell at mb, every slot-1 node on-shell at mc, the
  partner slot on-shell at md, the per-slot weights summing to 1, and
  the committed clamping of q_bc to 0 below 2*mb demonstrably gone).
- ``test_equal_mass_slot_byte_identity``: the equal-mass branch is
  byte-identical to the pre-fix committed formulas (an independent
  in-test re-implementation of the committed code).
- ``test_partner_slot_unchanged``: the partner slot (2) reproduces the
  committed two-body partner kinematics on any mass set.
- ``test_se_floor_evaluate_pair``: a collapsed between-replica SE is
  floored at 1e-12 with the vacuous-zero rule applied at the floor
  (residual at or below 1e-12 is a vacuous pass at sigma 0.0, a larger
  residual evaluates at the floored SE under the unchanged 3.0-sigma
  band), and a genuine SE is untouched.
- ``test_se_floor_reevaluated_status``: the re-evaluation mirror of the
  floor rule on sealed-check rows (the R31 window-0 K0/Sigma+
  float-precision SE artifact row resolves to a vacuous pass).
- ``test_deep_level_quadrature_noop_catalog_content``: the
  mechanism-(b) deep-level daughter-mass integration is a no-op on the
  committed catalog: the full record content of the five real-catalog
  equal-mass three-body cases (omega, eta, rho3 at the
  pinned evaluation masses) is byte-identical to the pre-P1 engine
  output (the sha256 pins captured from the committed code before the
  engine fix and carried in the phase-1 test file
  ``tests/test_threshold_metric.py``, reused here, not re-derived).

Hermetic: standard library, numpy, and the vendored ``na49_mixing``
modules; the Part 1 pins are read-only over the sealed run dir and skip
on a fresh checkout.  No Monte Carlo, no generation, no writes.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.threshold_metric import (
    DETERMINISTIC_SE_FLOOR,
    SIGMA_TOLERANCE,
    ZERO_RESIDUAL_TOL,
    RecordSet,
    evaluate_pair,
)

_tm = importlib.import_module("na49_mixing.threshold_metric")

NESTED_ROOT = Path(__file__).resolve().parent.parent
MONO_ROOT = NESTED_ROOT.parent.parent
RUN_NAME = "r3_threshold_metric_engine_fix_2026-09-18"
RUN_DIR = MONO_ROOT / "dev" / "00runs" / RUN_NAME
SUMS_PREFIX = f"dev/00runs/{RUN_NAME}/"

CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
METRIC_MODULE_PATH = NESTED_ROOT / "na49_mixing" / "threshold_metric.py"
DRIVER_PATH = NESTED_ROOT / "helpers" / "benchmarking" / (
    "rereval_threshold_checks.py"
)
DRIVER_OUTER_PATH = (
    MONO_ROOT / "helpers" / "benchmarking" / "rereval_threshold_checks.py"
)
BASELINE_DIR = MONO_ROOT / "dev" / "00runs" / (
    "r3_threshold_metric_reregistration_2026-09-17"
)
R31_DIR = MONO_ROOT / "dev" / "00runs" / (
    "r31_catalog_continuum_matching_scan_2026-09-17"
)

# ---------------------------------------------------------------------------
# The committed run contract (the sealed run realizes it).
# ---------------------------------------------------------------------------
RUN_ID = "R3-THRESHOLD-METRIC-ENGINE-FIX"
CARD_ID = "TASK-20260918-r3-card2-engine-three-body-moment-fix"
MASTER_SEED = 20260918
DRAWS = 0
PHASE_1_COMMIT = "95096d2"
TOLERANCE_SIGMA = 3.0
FLOOR = 1e-12
N_APPLIED_R21 = 351
N_APPLIED_R22 = 483
N_APPLIED_R21R22 = 834
N_ORIGINAL_FAILURES_R21 = 169
N_ORIGINAL_FAILURES_R22 = 154
N_ORIGINAL_FAILURES_R21R22 = 323
N_R31_APPLIED = 422
N_R31_ORIGINAL_FAILURES = 14
SEALED_RESIDUAL_FAILURES = 11
SEALED_PASS_RATE = 0.9659442724458205
SEALED_MAX_ABS_SIGMA = 4.984844257802041

PRE_SCHEMA = (
    "na49_mixing.rereval_threshold_checks.live_pre_registration/1"
)
PROV_SCHEMA = "na49_mixing.rereval_threshold_checks.live_provenance/1"
REREVAL_SCHEMA = "na49_mixing.rereval_threshold_checks.live_reeval/1"
REPORT_SCHEMA = (
    "na49_mixing.rereval_threshold_checks.revised_clause_v2/1"
)
SEAL_SCHEMA = "na49_mixing.rereval_threshold_checks.seal_manifest/1"
CLEAN_SCHEMA = (
    "na49_mixing.rereval_threshold_checks.clean_environment/1"
)

# ---------------------------------------------------------------------------
# The pinned digests (the sealed run's own inputs and payloads).
# ---------------------------------------------------------------------------
CATALOG_SHA256 = (
    "6e4de7372a5bbf99affd6fa3ccc295c4cef09404dab3b17acd8816cda762bdcd"
)
ENGINE_SHA256 = (
    "ee240f44645d2e592acfae462f7d1e9c3b3dd8cd78c96bd40b7b73509f7fcc63"
)
REREVAL_DRIVER_SHA256 = (
    "95f3ac6f0077b69c074eb701265d3a302fff09010f41641baf9c73cb6002b024"
)
R21_GATE_SHA256 = (
    "82af88d50e0b2be4da4e552a568181d1b6919f421190e7a54bea835556b07ac3"
)
R22_GATE_SHA256 = (
    "a2e81dbdb48d76c4a3c7e87436e9fb2e17f4a999918b261677ccbd2f16340865"
)
R31_VERIFICATION_SHA256 = (
    "de01f62f4907b3acbb8b4c5959148708d474d9c22df925d6b6d89dd188cde629"
)
#: The card-8b resume (TASK-20260918-r3-run-r31-verify-resume, under the
#: acceptance rescope TASK-20260918-r3-acceptance-rescope-card8a-8b)
#: regenerated the R31 matching verification with the fixed card-2
#: engine; this is the resumed report's digest.  The stopped digest
#: above stays the sealed stop-state record (kept by digest in the
#: resume block of the R31 provenance and pinned by
#: tests/test_staged_r31_matching_scan_gate_pins.py).
R31_RESUMED_VERIFICATION_SHA256 = (
    "9170426043399dc954c08ec255c7a7592f63ffbdfdcc3ab655cc1c0dd8e499af"
)
R31_PREREGISTRATION_SHA256 = (
    "3f5ed7d8599aeaa53d38de897f731125fb5d50e0e9f8d546058ea3ae9bbc1a14"
)
BASELINE_REPORT_SHA256 = (
    "e9609d9d1819148e818651723e37006ef77a53aa8a75c4eb3f4941c592eb0267"
)
SEAL_MANIFEST_SHA256 = (
    "6b7506cc8a98a0a2e6458b8ba54bcf241b755cd8559e7f45fce4ee79a7ae89ba"
)
SUMS_GATE2_V2_SHA256 = (
    "c45e4c9d28ba7c9986fce61b69f140ff016506d7c52afa7aa654d2e4eb3e6a88"
)
SUMS_R21_SHA256 = (
    "56f2ecf64bfb497c6dd67ceb77b290cf039aafbc059b047dfc4f46ef4b1e901f"
)
SUMS_R22_SHA256 = (
    "15213086e6203a4e623a32160fe9954f29ddca6b55c48dbf1e19e48512394f7c"
)
SUMS_R31_SHA256 = (
    "e7170a390b0455badc559d51807d3063a48b1ba0b230434b0d513121b04d029b"
)
SUMS_PREREG_SHA256 = (
    "08e4014b7545de2c07f86b7062ff4fdcdb059c9ceef7bc517b29b8b2d13b857e"
)
SUMS_PROV_SHA256 = (
    "9b6b2d88e391fef9752267e02ce4e1f1861397b5ce2d1db629367f92b7606416"
)

#: The exact SHA256SUMS.txt content (six monorepo-root-relative
#: entries, lexicographically sorted by relative path).
EXPECTED_SUMS_LINES = [
    f"{SUMS_GATE2_V2_SHA256}  {SUMS_PREFIX}gate2_revised_threshold_v2.json",
    f"{SUMS_R21_SHA256}  {SUMS_PREFIX}live_reeval_r21.json",
    f"{SUMS_R22_SHA256}  {SUMS_PREFIX}live_reeval_r22.json",
    f"{SUMS_R31_SHA256}  {SUMS_PREFIX}live_reeval_r31_targeted.json",
    f"{SUMS_PREREG_SHA256}  {SUMS_PREFIX}pre_registration.json",
    f"{SUMS_PROV_SHA256}  {SUMS_PREFIX}provenance.json",
]

#: The run dir holds exactly these nine top-level files.
EXPECTED_RUN_FILES = frozenset({
    "pre_registration.json",
    "provenance.json",
    "live_reeval_r21.json",
    "live_reeval_r22.json",
    "live_reeval_r31_targeted.json",
    "gate2_revised_threshold_v2.json",
    "clean_environment_commands.json",
    "seal_manifest.json",
    "SHA256SUMS.txt",
})

# ---------------------------------------------------------------------------
# The pinned live re-evaluation summaries (the committed verdict).
# ---------------------------------------------------------------------------
R21_SUMMARY = {
    "n_checks": 1026,
    "n_checks_applied": N_APPLIED_R21,
    "n_pass": 342,
    "n_fail": 9,
    "n_vacuous_zero_live": 0,
    "n_not_evaluable_live": 0,
    "original": {
        "n_checks_failed": N_ORIGINAL_FAILURES_R21,
        "failed_at_open_set_change": 4,
        "failed_at_unchanged": 165,
    },
    "reeval": {
        "original_failures_total": N_ORIGINAL_FAILURES_R21,
        "original_failures_now_passing": 160,
        "pass_rate_among_original_failures": 0.9467455621301775,
        "by_pair_class": {
            "open_set_change": {
                "applied": 72,
                "pass": 68,
                "fail": 4,
                "vacuous_zero_live": 0,
                "original_failures": 4,
                "original_failures_now_passing": 0,
                "pass_rate_among_original_failures": 0.0,
            },
            "open_set_unchanged": {
                "applied": 279,
                "pass": 274,
                "fail": 5,
                "vacuous_zero_live": 0,
                "original_failures": 165,
                "original_failures_now_passing": 160,
                "pass_rate_among_original_failures": 0.9696969696969697,
            },
        },
    },
    "max_abs_sigma": 3.5992208640500682,
    "max_abs_sigma_check": "pair(3,4) pi_0 mean_pt_y",
    "engine_vs_sealed_stored": {
        "multiplicity": 1.0470213190894029e-07,
        "mean_p": 0.0010346239650864675,
        "mean_pt_y": 0.00115295674763502,
    },
}

#: The R21 named residual census: (pair, species, observable,
#: pair_class, sigma).  Five genuine open-set-change, three
#: moment-approximation-floor, one marginal per the report (nine
#: failing checks).
R21_RESIDUALS = [
    ([0, 1], "proton", "mean_p", "open_set_change", 3.5987080275610572),
    ([0, 1], "proton", "mean_pt_y", "open_set_change", 3.445790227925448),
    ([1, 2], "pi_0", "mean_pt_y", "open_set_unchanged",
     3.362272164659226),
    ([3, 4], "pi_0", "mean_pt_y", "open_set_unchanged",
     3.5992208640500682),
    ([4, 5], "pi_minus", "mean_pt_y", "open_set_change", 3.4881623612596617),
    ([5, 6], "pi_0", "mean_pt_y", "open_set_change", 3.316931264858482),
    ([14, 15], "pi_plus", "mean_p", "open_set_unchanged",
     3.077997188838093),
    ([15, 16], "pi_0", "mean_p", "open_set_unchanged", 3.0407611711227656),
    ([18, 19], "pi_0", "multiplicity", "open_set_unchanged",
     3.014046219928031),
]

#: The R21 floor-residual rows as recorded in the live payload
#: (``sealed_sigma`` is the ORIGINAL sealed sigma of the pre-fix
#: engine, not the sealed re-evaluation sigma).
R21_FLOOR_RESIDUALS = [
    ("R21", [1, 2], "pi_0", "mean_pt_y", 174.22048958405568,
     3.362272164659226, False),
    ("R21", [3, 4], "pi_0", "mean_pt_y", 85.70171349014377,
     3.5992208640500682, False),
    ("R21", [15, 16], "pi_0", "mean_p", 7.1484024833977715,
     3.0407611711227656, False),
]

R22_SUMMARY = {
    "n_checks": 1458,
    "n_checks_applied": N_APPLIED_R22,
    "n_pass": 480,
    "n_fail": 3,
    "n_vacuous_zero_live": 0,
    "n_not_evaluable_live": 0,
    "original": {
        "n_checks_failed": N_ORIGINAL_FAILURES_R22,
        "failed_at_open_set_change": 1,
        "failed_at_unchanged": 153,
    },
    "reeval": {
        "original_failures_total": N_ORIGINAL_FAILURES_R22,
        "original_failures_now_passing": 151,
        "pass_rate_among_original_failures": 0.9805194805194806,
        "by_pair_class": {
            "open_set_change": {
                "applied": 83,
                "pass": 82,
                "fail": 1,
                "vacuous_zero_live": 0,
                "original_failures": 1,
                "original_failures_now_passing": 0,
                "pass_rate_among_original_failures": 0.0,
            },
            "open_set_unchanged": {
                "applied": 400,
                "pass": 398,
                "fail": 2,
                "vacuous_zero_live": 0,
                "original_failures": 153,
                "original_failures_now_passing": 151,
                "pass_rate_among_original_failures": 0.9869281045751634,
            },
        },
    },
    "max_abs_sigma": 4.973562263179045,
    "max_abs_sigma_check": "pair(23,24) pi_0 mean_pt_y",
    "engine_vs_sealed_stored": {
        "multiplicity": 8.520139052170528e-06,
        "mean_p": 0.00031546392915315913,
        "mean_pt_y": 0.0005148188409500876,
    },
}

R22_RESIDUALS = [
    ([7, 8], "pi_minus", "mean_pt_y", "open_set_change",
     3.8763921987053918),
    ([15, 16], "pi_plus", "mean_pt_y", "open_set_unchanged",
     3.110100410337241),
    ([23, 24], "pi_0", "mean_pt_y", "open_set_unchanged",
     4.973562263179045),
]

R22_FLOOR_RESIDUALS = [
    ("R22", [23, 24], "pi_0", "mean_pt_y", 6.390868888257438,
     4.973562263179045, False),
]

R31_SUMMARY = {
    "n_checks_applied": N_R31_APPLIED,
    "n_pass": 299,
    "n_fail": 10,
    "n_vacuous_zero_live": 113,
    "n_not_evaluable_live": 0,
    "n_absent_carried": 226,
    "original": {"checks_failed": N_R31_ORIGINAL_FAILURES},
    "max_abs_sigma": 7.01270490790188,
    "max_abs_sigma_check": "w2 pair(1,2) pi_0 mean_p",
    "boundary_straddling_pairs": [2, 6, 10],
}

#: The 14-row targeted R31 original-failure table: (window, pair,
#: global_pair, species, observable, sigma_original, sigma, status,
#: pass).  The two window-1 Lambda(1520) three-body checks resolve to
#: sub-3-sigma, the two float-precision SE artifacts vacuous, and ten
#: persist at their sealed sigmas.
R31_ORIGINAL_FAILURES = [
    (0, [0, 1], 0, "proton", "mean_pt_y", 3.2123466532077045,
     3.2123466532077045, "fail", False),
    (0, [0, 1], 0, "pi_minus", "mean_pt_y", 3.3254504430919427,
     3.3254504430919427, "fail", False),
    (0, [3, 4], 3, "neutron", "mean_p", 3.151740704856282,
     3.151740704856282, "fail", False),
    (0, [3, 4], 3, "K_0", "mean_p", 25.98076211353316, 0.0,
     "vacuous_zero", True),
    (0, [3, 4], 3, "Sigma_plus", "mean_p", 25.98076211353316, 0.0,
     "vacuous_zero", True),
    (1, [2, 3], 6, "pi_plus", "mean_pt_y", 7.225811435009375,
     0.9201361176029509, "pass", True),
    (1, [3, 4], 7, "pi_plus", "mean_pt_y", -5.610935929060075,
     0.36971803795576264, "pass", True),
    (2, [0, 1], 8, "pi_0", "mean_pt_y", -3.0280248998333104,
     3.0280248998333104, "fail", False),
    (2, [0, 1], 8, "K0L", "mean_p", 3.064256426739259, 3.064256426739259,
     "fail", False),
    (2, [0, 1], 8, "K0L", "mean_pt_y", 3.069041367744035, 3.069041367744035,
     "fail", False),
    (2, [1, 2], 9, "pi_0", "mean_p", 7.01270490790188, 7.01270490790188,
     "fail", False),
    (2, [3, 4], 11, "K0S", "mean_pt_y", 3.4011873510462327,
     3.4011873510462327, "fail", False),
    (2, [3, 4], 11, "K0L", "mean_p", 3.7365731963210447,
     3.7365731963210447, "fail", False),
    (2, [3, 4], 11, "K0L", "mean_pt_y", 4.547781556513862,
     4.547781556513862, "fail", False),
]

R31_BOUNDARY_INTERIOR = {
    "boundary_pair_applied": 104,
    "boundary_pair_failed_live": 0,
    "interior_pairs_applied": 318,
    "interior_pairs_failed_live": 10,
}

# ---------------------------------------------------------------------------
# The pinned gate-2 clause report (the sealed baseline alongside the
# post-fix values).
# ---------------------------------------------------------------------------
GATE2_TOLERANCE = {
    "sigma": 3.0,
    "unchanged": True,
    "se_combined": "sqrt(SE_k^2 + SE_{k+1}^2)",
    "deterministic_se_floor": 1e-12,
    "relaxed": False,
}

GATE2_SEALED_BASELINE_COMBINED = {
    "applied": 834,
    "pass": 823,
    "fail": 11,
    "residual_failures": 11,
    "pass_rate_among_original_failures": 0.9659442724458205,
    "max_abs_sigma": 4.984844257802041,
    "max_abs_sigma_check": "R22 pair(23,24) pi_0 mean_pt_y",
}

GATE2_COMBINED = {
    "applied": 834,
    "pass": 822,
    "fail": 12,
    "original_failures": 323,
    "original_failures_now_passing": 311,
    "residual_failures": 12,
    "max_abs_sigma": 4.973562263179045,
    "max_abs_sigma_check": "R22 pair(23,24) pi_0 mean_pt_y",
    "pass_rate_among_original_failures": 0.9628482972136223,
}

#: The four acceptance floor-residual checks with their sealed
#: RE-EVALUATION sigmas (the sealed card-2 re-evaluation's values)
#: against the post-fix live sigmas.
GATE2_FLOOR_RESIDUALS = [
    ("R21", [1, 2], "pi_0", "mean_pt_y", 3.3622721646606926,
     3.362272164659226, False),
    ("R21", [3, 4], "pi_0", "mean_pt_y", 3.594992035740298,
     3.5992208640500682, False),
    ("R21", [15, 16], "pi_0", "mean_p", 3.0846781101915357,
     3.0407611711227656, False),
    ("R22", [23, 24], "pi_0", "mean_pt_y", 4.984844257802041,
     4.973562263179045, False),
]

GATE2_ACCEPTANCE = {
    "t1_targeted_r31": {
        "n_original_failures": 14,
        "original_failures_passing": 4,
        "original_failures_persisting": 10,
        "pass": False,
    },
    "t2_card2_reeval": {
        "failure_count": {
            "sealed": 11,
            "achieved": 12,
            "required": "strictly below the sealed 11",
            "pass": False,
        },
        "pass_rate_among_original_failures": {
            "sealed": 0.9659442724458205,
            "threshold": 0.9659442724458205,
            "achieved": 0.9628482972136223,
            "pass": False,
        },
        "pass": False,
    },
    "pass": False,
}

N_RESIDUAL_FINDINGS = 12

# ---------------------------------------------------------------------------
# Part 2: the pre-P1 record-content sha256 pins (the deep-level
# quadrature no-op).  These values were captured from the committed
# code before the engine fix (phase-1 pins in
# ``tests/test_threshold_metric.py``) and are reused, not re-derived.
# ---------------------------------------------------------------------------
CONTENT_PIN_CASES = (
    (223, 0.7),      # omega(782), below the pole
    (223, 0.8251),   # omega(782), at the window top
    (221, 0.5479),   # eta, at the window top
    # the eta-prime(958) at 377, at the window top, left with its record in the
    # orphan-record revision of 2026-10-29
    (117, 1.6888),   # rho3(1690)0, at the pole
    (117, 2.4938),   # rho3(1690)0, at the window top
)
CONTENT_PIN_SHA256 = {
    "223|0.7": "7d672f3333a32ce1aa6840f95748483f"
               "cbe6ceddaf946c44dc781f46db4530ec",
    "223|0.8251": "f39bea338d8026b393044c7840c1bd2c"
                  "b5568b46a54177f5cdcef4ee755cbb72",
    "221|0.5479": "06dc94dddd6fb49fba8ec71b5b7896df"
                  "8ce010f08f198bb87568969ddf117dbf",
    "117|1.6888": "8c6abc296bedd1d0826dfdf01398d4ff"
                  "1598b87b8133a06bf5b07a51e29ddd0a",
    "117|2.4938": "e908175266ca59d0ae80d65ac744552c"
                  "70f0e4c543d2556f3b4527f15f60dd45",
}


def _content_digest(content: dict) -> str:
    """The canonical repr-dump sha256 of a record content block map
    (the phase-1 pin convention)."""
    blob = json.dumps(
        {skey: {f: repr(float(v)) for f, v in sorted(block.items())}
         for skey, block in sorted(content.items())},
        sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Part 2: the pre-fix committed three-body slot formulas (the
# independent byte-identity reference for the equal-mass branch and the
# partner slot, written out independently of the engine).
# ---------------------------------------------------------------------------
def _committed_slot_nodes(m_parent, masses, slot):
    """The committed engine's equal-mass three-body slot nodes, written
    out independently (the pre-fix formulas)."""
    mb, mc, md = (float(m) for m in masses)
    M = float(m_parent)
    lo2 = (mb + mc) ** 2
    hi2 = (M - md) ** 2
    mbc2 = 0.5 * (hi2 - lo2) * _tm._MBC_X + 0.5 * (hi2 + lo2)
    Mbc = np.sqrt(mbc2)
    q_bc = np.sqrt(np.maximum(Mbc * Mbc / 4.0 - mb * mb, 0.0))
    e_pair = np.maximum((M * M + mbc2 - md * md) / (2.0 * M), 1e-300)
    q_pair = np.sqrt(np.maximum(
        ((M * M - mbc2 - md * md) ** 2 - 4.0 * mbc2 * md * md)
        / (4.0 * M * M), 0.0))
    gamma_p = e_pair / Mbc
    beta_p = q_pair / e_pair
    if slot <= 1:
        sign = 1.0 if slot == 0 else -1.0
        e_rest = gamma_p[:, None] * (
            0.5 * Mbc[:, None] + beta_p[:, None] * q_bc[:, None] * sign
            * _tm._COS3_X[None, :])
        p_par = gamma_p[:, None] * (
            sign * q_bc[:, None] * _tm._COS3_X[None, :]
            + beta_p[:, None] * 0.5 * Mbc[:, None])
        p_rest = np.sqrt(np.maximum(
            p_par * p_par
            + (q_bc * q_bc)[:, None] * (1.0 - _tm._COS3_X[None, :] ** 2),
            0.0))
        weights = (_tm._MBC_W[:, None] * _tm._COS3_W[None, :]
                   * np.ones_like(e_rest))
        return (weights.ravel(), e_rest.ravel(), p_rest.ravel())
    e_rest = (M * M - mbc2 + md * md) / (2.0 * M)
    return (_tm._MBC_W.copy(), e_rest, q_pair)


# ---------------------------------------------------------------------------
# Part 1 helpers
# ---------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def run_dir():
    if not RUN_DIR.is_dir():
        pytest.skip(
            f"sealed run dir {RUN_NAME} absent on this checkout "
            "(the gitignored archive is not present); the Part 1 "
            "artifact pins skip with a recorded reason"
        )
    return RUN_DIR


def _load(run_dir, name: str) -> dict:
    with (run_dir / name).open(encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Part 1: the sealed-run artifact pins
# ---------------------------------------------------------------------------
def test_sealed_run_dir_inventory(run_dir):
    """The run dir holds exactly the nine declared top-level files
    (the six sealed payloads, the seal manifest, the sums file, the
    post-seal clean-environment record) and no sub-trees."""
    entries = sorted(p.name for p in run_dir.iterdir())
    assert entries == sorted(EXPECTED_RUN_FILES), entries
    for name in EXPECTED_RUN_FILES:
        assert (run_dir / name).is_file(), name


def test_sha256_sums(run_dir):
    """``SHA256SUMS.txt`` carries exactly the six pinned
    monorepo-root-relative entries, lexicographically sorted by
    relative path, and every digest verifies against the file on disk
    from the monorepo root."""
    text = (run_dir / "SHA256SUMS.txt").read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    assert lines == EXPECTED_SUMS_LINES
    paths = [line.split("  ", 1)[1] for line in lines]
    assert paths == sorted(paths)
    for line in lines:
        digest, rel = line.split("  ", 1)
        assert _sha256_file(MONO_ROOT / rel) == digest, rel
    assert _sha256_file(run_dir / "SHA256SUMS.txt") == SEAL_MANIFEST_SHA256


def test_seal_manifest(run_dir):
    """The seal manifest records the six sealed payloads (1,901,206
    bytes), the sums-file digest, and the documented exclusions."""
    seal = _load(run_dir, "seal_manifest.json")
    assert seal["schema"] == SEAL_SCHEMA
    assert seal["run"] == RUN_ID
    assert seal["card"] == CARD_ID
    assert seal["file_count"] == 6
    assert seal["total_bytes"] == 1_901_206
    assert seal["sha256_sums_file"] == "SHA256SUMS.txt"
    assert seal["sha256_sums_sha256"] == SEAL_MANIFEST_SHA256
    assert seal["excluded_from_sums"] == [
        "seal_manifest.json", "SHA256SUMS.txt"]
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$",
                    seal["created_utc"])


def test_pre_registration_structure(run_dir):
    """The sealed pre-registration carries the committed contract."""
    pre = _load(run_dir, "pre_registration.json")
    assert pre["schema"] == PRE_SCHEMA
    assert pre["run"] == RUN_ID
    assert pre["card"] == CARD_ID
    assert pre["phase"] == 2
    assert pre["master_seed"] == MASTER_SEED
    assert pre["draws"] == DRAWS
    assert pre["written_before_evaluation"] is True
    assert sorted(pre["outputs"]) == sorted(EXPECTED_RUN_FILES)

    engine = pre["engine"]
    assert engine["module"] == "na49_mixing.threshold_metric"
    assert engine["module_sha256_at_registration"] == ENGINE_SHA256
    assert engine["sha256"] == ENGINE_SHA256
    assert engine["phase_1_commit"] == PHASE_1_COMMIT
    assert set(engine["mechanisms"]) == {
        "a_exact_kallen_three_body_slots",
        "b_deep_level_daughter_mass_integration",
        "c_deterministic_se_floor",
    }

    metric = pre["metric"]
    assert metric["id"] == "delta_analytic_full"
    assert metric["tolerance_sigma"] == TOLERANCE_SIGMA
    assert metric["tolerance_unchanged"] is True
    floor = metric["deterministic_se_floor"]
    assert floor["value"] == FLOOR
    assert floor["vacuous_zero_threshold"] == FLOOR
    assert floor["tolerance_unchanged"] is True

    baseline = pre["baseline"]
    assert baseline["run"] == "R3-THRESHOLD-METRIC-REREGISTRATION"
    assert baseline["path"] == (
        "dev/00runs/r3_threshold_metric_reregistration_2026-09-17")
    assert baseline["sealed_residual_failures"] == SEALED_RESIDUAL_FAILURES
    assert baseline["pass_rate_among_original_failures"] == SEALED_PASS_RATE
    assert baseline["max_abs_sigma"] == SEALED_MAX_ABS_SIGMA
    assert baseline["read_only"] is True

    coverage = pre["coverage"]
    assert coverage["r21"] == {
        "adjacent_pairs": 19,
        "applied_checks_original": N_APPLIED_R21,
        "failed_original": N_ORIGINAL_FAILURES_R21,
    }
    assert coverage["r22"] == {
        "adjacent_pairs": 27,
        "applied_checks_original": N_APPLIED_R22,
        "failed_original": N_ORIGINAL_FAILURES_R22,
    }
    assert coverage["r31"] == {
        "adjacent_pairs": 12,
        "applied_checks": N_R31_APPLIED,
        "failed_original": N_R31_ORIGINAL_FAILURES,
    }
    assert coverage["observables"] == [
        "multiplicity", "mean_p", "mean_pt_y"]
    assert coverage["pair_classes"] == [
        "open_set_change", "open_set_unchanged"]
    assert (coverage["total_original_failures_r21_r22"]
            == N_ORIGINAL_FAILURES_R21R22)
    assert coverage["applied_total_r21_r22"] == N_APPLIED_R21R22

    acceptance = pre["acceptance"]
    t1 = acceptance["t1_targeted_r31"]
    assert t1["n_original_failures"] == N_R31_ORIGINAL_FAILURES
    assert t1["required"] == "n_fail_live == 0"
    t2 = acceptance["t2_card2_reeval"]
    assert t2["n_original_failures"] == N_ORIGINAL_FAILURES_R21R22
    floor_checks = t2["floor_residual_checks"]
    assert len(floor_checks) == 4
    assert [(c["run"], tuple(c["pair"]), c["species"], c["observable"])
            for c in floor_checks] == [
        (r[0], tuple(r[1]), r[2], r[3]) for r in GATE2_FLOOR_RESIDUALS]

    constraints = pre["constraints"]
    assert constraints["no_fitting"] is True
    assert constraints["sealed_archives_read_only"] is True
    assert constraints["tolerance_relaxed"] is False
    assert constraints["no_pair_special_casing"] is True

    for run, gate in (("r21", "gate2_r21.json"),
                      ("r22", "gate2_r22.json")):
        archive = pre["archives"][run]
        assert archive["gate_file"] == gate
        assert archive["read_only"] is True
    r31 = pre["r31_targeted"]
    assert r31["verification_file"] == "matching_verification.json"
    assert r31["read_only"] is True


def test_provenance_structure(run_dir):
    """The provenance records the fixed engine, the dual-tree driver,
    the sealed archive digests, the git identity, and the environment
    fingerprint."""
    prov = _load(run_dir, "provenance.json")
    assert prov["schema"] == PROV_SCHEMA
    assert prov["run"] == RUN_ID
    assert prov["card"] == CARD_ID
    assert prov["phase"] == 2
    assert prov["master_seed"] == MASTER_SEED
    assert prov["draws"] == DRAWS
    assert prov["written_utc"] == "2026-09-19T16:54:45Z"

    assert prov["engine"]["module"] == (
        "dev/mixing2006/na49_mixing/threshold_metric.py")
    assert prov["engine"]["module_sha256"] == ENGINE_SHA256
    assert prov["engine"]["phase_1_commit"] == PHASE_1_COMMIT
    assert prov["catalog"]["sha256"] == CATALOG_SHA256

    r21 = prov["sealed_archives"]["r21"]
    assert r21["gate_file"] == "gate2_r21.json"
    assert r21["gate_sha256"] == R21_GATE_SHA256
    assert r21["seal"] == {
        "files_checked": 103,
        "sha256_sums_sha256": (
            "92f4070758ceac252e9209b01efa076c3351cbc54ed270d347b8be3fb6164537"
        ),
    }
    r22 = prov["sealed_archives"]["r22"]
    assert r22["gate_file"] == "gate2_r22.json"
    assert r22["gate_sha256"] == R22_GATE_SHA256
    assert r22["seal"] == {
        "files_checked": 143,
        "sha256_sums_sha256": (
            "350c869f4e13a303e8bdb91ab8ad3d267bb60fe9a36348df79bee839f52c6fbc"
        ),
    }
    r31 = prov["r31_artifacts"]
    assert r31["verification_sha256"] == R31_VERIFICATION_SHA256
    assert r31["pre_registration_sha256"] == R31_PREREGISTRATION_SHA256
    assert r31["seal"] is None
    baseline = prov["baseline"]
    assert baseline["report_file"] == "gate2_revised_threshold.json"
    assert baseline["report_sha256"] == BASELINE_REPORT_SHA256
    assert prov["drivers"]["rereval_driver_sha256"] == (
        REREVAL_DRIVER_SHA256)

    git = prov["git"]
    assert len(git["nested_head"]) == 40
    assert len(git["outer_head"]) == 40
    env = prov["environment"]
    assert env["python_implementation"] == "CPython"
    assert env["machine"] == "x86_64"
    assert env["numpy_version"] == "1.26.4"

    # The sealed inputs verify against disk on this checkout.  The R31
    # matching verification is the card-8b resumed report (see the
    # R31_RESUMED_VERIFICATION_SHA256 note above); the stopped digest
    # recorded in this provenance is the sealed stop-state record.
    assert _sha256_file(METRIC_MODULE_PATH) == ENGINE_SHA256
    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)
    assert _sha256_file(BASELINE_DIR / "gate2_revised_threshold.json") \
        == BASELINE_REPORT_SHA256
    assert _sha256_file(R31_DIR / "matching_verification.json") \
        == R31_RESUMED_VERIFICATION_SHA256


def test_live_reeval_r21_summary(run_dir):
    """The R21 live re-evaluation carries the pinned summary, the
    named residual census, and the floor-residual rows (original
    sealed sigmas)."""
    payload = _load(run_dir, "live_reeval_r21.json")
    assert payload["schema"] == REREVAL_SCHEMA
    assert payload["card"] == CARD_ID
    assert payload["phase"] == 2
    assert payload["run"] == "R21"
    assert payload["metric"]["id"] == "delta_analytic_full"
    summary = payload["summary"]
    for key, value in R21_SUMMARY.items():
        assert summary[key] == value, key
    residuals = summary["residuals"]
    assert len(residuals) == len(R21_RESIDUALS)
    for row, (pair, species, observable, pair_class, sigma) in zip(
            residuals, R21_RESIDUALS):
        assert row["pair"] == pair
        assert row["species"] == species
        assert row["observable"] == observable
        assert row["pair_class"] == pair_class
        assert row["sigma"] == sigma
        assert abs(row["residual"]) == pytest.approx(
            abs(sigma) * row["se_combined"], rel=1e-12, abs=1e-18)
    floors = summary["floor_residuals"]
    assert len(floors) == len(R21_FLOOR_RESIDUALS)
    for row, (run, pair, species, observable, sealed, live, passed) in zip(
            floors, R21_FLOOR_RESIDUALS):
        assert row["run"] == run
        assert row["pair"] == pair
        assert row["species"] == species
        assert row["observable"] == observable
        assert row["sealed_sigma"] == sealed
        assert row["live_sigma"] == live
        assert row["pass"] is passed
    # The check census is the full R21 gate payload (1026 rows), of
    # which 351 checks carry the applied scope.
    assert len(payload["checks"]) == summary["n_checks"]
    applied = [c for c in payload["checks"] if c["scope"] == "applied"]
    assert len(applied) == summary["n_checks_applied"]


def test_live_reeval_r22_summary(run_dir):
    """The R22 live re-evaluation carries the pinned summary (the full
    1458-row check census, 483 applied), the named residual census, and
    the single floor-residual row."""
    payload = _load(run_dir, "live_reeval_r22.json")
    assert payload["schema"] == REREVAL_SCHEMA
    assert payload["run"] == "R22"
    summary = payload["summary"]
    for key, value in R22_SUMMARY.items():
        assert summary[key] == value, key
    residuals = summary["residuals"]
    assert len(residuals) == len(R22_RESIDUALS)
    for row, (pair, species, observable, pair_class, sigma) in zip(
            residuals, R22_RESIDUALS):
        assert row["pair"] == pair
        assert row["species"] == species
        assert row["observable"] == observable
        assert row["pair_class"] == pair_class
        assert row["sigma"] == sigma
    floors = summary["floor_residuals"]
    assert len(floors) == len(R22_FLOOR_RESIDUALS)
    for row, (run, pair, species, observable, sealed, live, passed) in zip(
            floors, R22_FLOOR_RESIDUALS):
        assert row["run"] == run
        assert row["pair"] == pair
        assert row["species"] == species
        assert row["observable"] == observable
        assert row["sealed_sigma"] == sealed
        assert row["live_sigma"] == live
        assert row["pass"] is passed
    assert len(payload["checks"]) == summary["n_checks"]
    applied = [c for c in payload["checks"] if c["scope"] == "applied"]
    assert len(applied) == summary["n_checks_applied"]


def test_live_reeval_r31_targeted_summary(run_dir):
    """The targeted R31 re-evaluation carries the pinned summary, the
    14-row original-failure table (the two Lambda(1520) three-body
    checks resolved, the two SE artifacts vacuous, ten persisting), and
    the boundary-interior split."""
    payload = _load(run_dir, "live_reeval_r31_targeted.json")
    assert payload["schema"] == REREVAL_SCHEMA
    assert payload["run"] == "R31"
    summary = payload["summary"]
    for key, value in R31_SUMMARY.items():
        assert summary[key] == value, key
    assert summary["original_failures_passing"] == 4
    assert summary["original_failures_persisting"] == 10
    assert summary["boundary_interior"] == R31_BOUNDARY_INTERIOR
    rows = summary["original_failures"]
    assert len(rows) == len(R31_ORIGINAL_FAILURES)
    for row, (window, pair, global_pair, species, observable,
              sigma_original, sigma, status, passed) in zip(
                  rows, R31_ORIGINAL_FAILURES):
        assert row["window"] == window
        assert row["pair"] == pair
        assert row["global_pair"] == global_pair
        assert row["species"] == species
        assert row["observable"] == observable
        assert row["sigma_original"] == sigma_original
        assert row["sigma"] == sigma
        assert row["status"] == status
        assert row["pass"] is passed
    residuals = summary["residuals"]
    assert len(residuals) == 10
    for row in residuals:
        # The fix may flip a residual's sign (the w2 pi_0 mean_pt_y row)
        # but never its magnitude.
        assert abs(row["sigma"]) == abs(row["sigma_original"])
        assert abs(row["sigma"]) > TOLERANCE_SIGMA
        assert isinstance(row["diagnosis"], str)
        assert len(row["diagnosis"]) > 0


def test_gate2_revised_clause(run_dir):
    """The revised clause report pins the tolerance block, the sealed
    baseline alongside the post-fix values, the acceptance verdicts
    (T1 and T2 both unmet, recorded as findings), and the twelve named
    residual findings."""
    report = _load(run_dir, "gate2_revised_threshold_v2.json")
    assert report["schema"] == REPORT_SCHEMA
    assert report["card"] == CARD_ID
    assert report["phase"] == 2
    assert report["gate"] == "gate 2 (plan section 8, line 575)"
    assert report["clause"] == "threshold_discontinuity"
    assert report["no_fitting"] is True

    assert report["tolerance"] == GATE2_TOLERANCE
    assert report["sealed_baseline"]["combined"] == (
        GATE2_SEALED_BASELINE_COMBINED)
    assert report["combined"] == GATE2_COMBINED

    targeted = report["targeted_r31"]
    assert targeted["n_checks_applied"] == N_R31_APPLIED
    assert targeted["n_pass"] == 299
    assert targeted["n_vacuous_zero_live"] == 113
    assert targeted["n_fail"] == 10
    assert targeted["max_abs_sigma"] == 7.01270490790188
    assert targeted["boundary_interior"] == R31_BOUNDARY_INTERIOR
    assert targeted["original_failures_passing"] == 4
    assert targeted["original_failures_persisting"] == 10

    acceptance = report["acceptance"]
    for key, value in GATE2_ACCEPTANCE["t1_targeted_r31"].items():
        assert acceptance["t1_targeted_r31"][key] == value, key
    t2 = acceptance["t2_card2_reeval"]
    for key, value in GATE2_ACCEPTANCE["t2_card2_reeval"].items():
        if key == "floor_residuals":
            continue
        assert t2[key] == value, key
    floors = t2["floor_residuals"]
    assert floors["required"] == "all four pass"
    assert floors["pass"] is False
    checks = floors["checks"]
    assert len(checks) == len(GATE2_FLOOR_RESIDUALS)
    for row, (run, pair, species, observable, sealed, live, passed) in zip(
            checks, GATE2_FLOOR_RESIDUALS):
        assert row["run"] == run
        assert row["pair"] == pair
        assert row["species"] == species
        assert row["observable"] == observable
        assert row["sealed_rereval_sigma"] == sealed
        assert row["live_sigma"] == live
        assert row["pass"] is passed
    assert acceptance["pass"] is False

    findings = report["residual_findings"]
    assert len(findings) == N_RESIDUAL_FINDINGS
    expected_keys = {
        ("R21", (0, 1), "proton", "mean_p"),
        ("R21", (0, 1), "proton", "mean_pt_y"),
        ("R21", (1, 2), "pi_0", "mean_pt_y"),
        ("R21", (3, 4), "pi_0", "mean_pt_y"),
        ("R21", (4, 5), "pi_minus", "mean_pt_y"),
        ("R21", (5, 6), "pi_0", "mean_pt_y"),
        ("R21", (14, 15), "pi_plus", "mean_p"),
        ("R21", (15, 16), "pi_0", "mean_p"),
        ("R21", (18, 19), "pi_0", "multiplicity"),
        ("R22", (7, 8), "pi_minus", "mean_pt_y"),
        ("R22", (15, 16), "pi_plus", "mean_pt_y"),
        ("R22", (23, 24), "pi_0", "mean_pt_y"),
    }
    assert {(f["run"], tuple(f["pair"]), f["species"], f["observable"])
            for f in findings} == expected_keys
    for finding in findings:
        assert finding["sigma"] > TOLERANCE_SIGMA
        assert isinstance(finding["diagnosis"], str)


def test_clean_environment_record(run_dir):
    """The post-seal clean-environment record re-verified the seal (6
    files, no mismatches), re-ran the live re-evaluations in the clean
    interpreter (``python -I``), and the four sealed report digests
    match the recomputed digests."""
    clean = _load(run_dir, "clean_environment_commands.json")
    assert clean["schema"] == CLEAN_SCHEMA
    assert clean["run"] == RUN_ID
    assert clean["card"] == CARD_ID
    assert clean["verified_utc"] == "2026-09-19T17:20:26Z"
    assert clean["seal_verification"] == {
        "files_checked": 6, "mismatches": []}
    assert clean["hashes_match"] is True
    assert len(clean["commands"]) == 1
    command = clean["commands"][0]
    assert command["argv"][0] == "/usr/bin/python"
    assert "-I" in command["argv"]
    assert command["cwd"] == str(MONO_ROOT)
    sealed = clean["sealed_report_sha256"]
    recomputed = clean["recomputed_report_sha256"]
    assert sealed == recomputed
    assert sealed["live_reeval_r21.json"] == SUMS_R21_SHA256
    assert sealed["live_reeval_r22.json"] == SUMS_R22_SHA256
    assert sealed["live_reeval_r31_targeted.json"] == SUMS_R31_SHA256
    assert sealed["gate2_revised_threshold_v2.json"] == SUMS_GATE2_V2_SHA256
    # The clean record verifies the sealed payloads it re-computes.
    for name, digest in sealed.items():
        assert _sha256_file(run_dir / name) == digest, name


# ---------------------------------------------------------------------------
# Part 2: the pure engine contract (no sealed artifacts)
# ---------------------------------------------------------------------------
def test_engine_contract_constants():
    """The tolerance and the deterministic-SE floor are the
    pre-registered values, the floor equals the vacuous-zero
    threshold, and the on-disk engine module is the fixed P2 engine."""
    assert SIGMA_TOLERANCE == TOLERANCE_SIGMA
    assert DETERMINISTIC_SE_FLOOR == FLOOR
    assert ZERO_RESIDUAL_TOL == FLOOR
    assert DETERMINISTIC_SE_FLOOR == ZERO_RESIDUAL_TOL
    assert _sha256_file(METRIC_MODULE_PATH) == ENGINE_SHA256
    # The dual-tree driver stays byte-identical at the pinned digest.
    assert _sha256_file(DRIVER_PATH) == REREVAL_DRIVER_SHA256
    assert _sha256_file(DRIVER_OUTER_PATH) == REREVAL_DRIVER_SHA256


def test_kallen_unequal_mass_slot_kinematics():
    """The exact Kallen leading-pair kinematics on an
    unequal-mass three-body system: every slot-0 node is on-shell at
    mb, every slot-1 node on-shell at mc, the partner slot on-shell at
    md, the per-slot weights sum to 1, and on the Mbc^2 nodes where
    the committed formula clamps q_bc to zero (Mbc^2 <= 4*mb^2) the
    exact pair-rest kinematics replaces the committed 0.5*Mbc."""
    M = 1.0
    mb, mc, md = 0.3, 0.1, 0.2
    masses = (mb, mc, md)
    for slot, m_d in ((0, mb), (1, mc), (2, md)):
        weights, e, p = _tm._three_body_slot_nodes(M, masses, slot)
        assert weights.shape == e.shape == p.shape
        assert weights.size == 576 if slot <= 1 else weights.size == 24
        assert weights.sum() == pytest.approx(1.0, abs=1e-15)
        m2 = e * e - p * p
        assert np.allclose(m2, m_d * m_d, rtol=1e-12, atol=1e-18), (
            slot, float(m2.min()), float(m2.max()))
        assert np.all(e >= m_d * (1.0 - 1e-12))
    # The independent exact Kallen reference on the Mbc^2 grid.
    lo2, hi2 = (mb + mc) ** 2, (M - md) ** 2
    mbc2 = 0.5 * (hi2 - lo2) * _tm._MBC_X + 0.5 * (hi2 + lo2)
    Mbc = np.sqrt(mbc2)
    q_ref = np.sqrt(np.maximum(
        (mbc2 - (mb + mc) ** 2) * (mbc2 - (mb - mc) ** 2), 0.0)
        ) / (2.0 * Mbc)
    e_b_ref = (mbc2 + mb * mb - mc * mc) / (2.0 * Mbc)
    e_pair = (M * M + mbc2 - md * md) / (2.0 * M)
    q_pair = np.sqrt((M * M - mbc2 - md * md) ** 2
                     - 4.0 * mbc2 * md * md) / (2.0 * M)
    gamma_p, beta_p = e_pair / Mbc, q_pair / e_pair
    clamped = mbc2 <= 4.0 * mb * mb
    assert int(np.sum(clamped)) > 0
    # The clamp is gone: below 2*mb (away from the lo endpoint) the
    # exact pair momentum is strictly positive, and the exact
    # pair-rest energy exceeds the committed 0.5*Mbc (mb > mc).
    assert np.all(q_ref[clamped & (mbc2 > lo2)] > 0.0)
    assert np.all(e_b_ref[clamped] > 0.5 * Mbc[clamped])
    w_f, e_f, p_f = _tm._three_body_slot_nodes(M, masses, 0)
    w_c, e_c, _ = _committed_slot_nodes(M, masses, 0)
    assert not np.array_equal(e_f, e_c)
    for i in np.where(clamped)[0]:
        for j, c in enumerate(_tm._COS3_X):
            e = gamma_p[i] * (e_b_ref[i] + beta_p[i] * q_ref[i] * c)
            p_par = gamma_p[i] * (q_ref[i] * c + beta_p[i] * e_b_ref[i])
            p = np.sqrt(p_par * p_par
                        + q_ref[i] ** 2 * (1.0 - c * c))
            k = len(_tm._COS3_X) * int(i) + j
            assert e_f[k] == pytest.approx(e, rel=1e-12, abs=1e-15)
            assert p_f[k] == pytest.approx(p, rel=1e-12, abs=1e-15)


def test_equal_mass_slot_byte_identity():
    """The equal-mass branch evaluates byte-identically to the
    committed pre-fix formulas on all three slots."""
    for M, masses in ((1.0, (0.2, 0.2, 0.3)),
                      (1.5, (0.3, 0.3, 0.4)),
                      (0.9, (0.15, 0.15, 0.25))):
        for slot in (0, 1, 2):
            got = _tm._three_body_slot_nodes(M, masses, slot)
            ref = _committed_slot_nodes(M, masses, slot)
            for g, r in zip(got, ref):
                assert np.array_equal(g, r), (M, masses, slot)


def test_partner_slot_unchanged():
    """The partner slot (2) reproduces the committed two-body partner
    kinematics (the leading-pair direction does not enter it) on an
    unequal-mass system as well."""
    for M, masses in ((1.0, (0.3, 0.1, 0.2)), (1.2, (0.2, 0.2, 0.3))):
        w, e, p = _tm._three_body_slot_nodes(M, masses, 2)
        w_ref, e_ref, p_ref = _committed_slot_nodes(M, masses, 2)
        assert np.array_equal(w, w_ref)
        assert np.array_equal(e, e_ref)
        assert np.array_equal(p, p_ref)


def test_se_floor_evaluate_pair():
    """The deterministic-observable floor in ``evaluate_pair``: a
    collapsed between-replica SE is floored at 1e-12 with the
    vacuous-zero rule applied at the floor; a genuine SE is
    untouched; the tolerance never relaxes."""
    # Collapsed SE, zero residual: vacuous pass at sigma 0.0.
    r = evaluate_pair(0.05, 0.05, 1.3e-16, 1.3e-16)
    assert r["se_combined"] == pytest.approx(1.838e-16, rel=1e-3)
    assert r["se_floor_applied"] is True
    assert r["se_effective"] == FLOOR
    assert r["sigma"] == 0.0
    assert r["pass"] is True
    # Collapsed SE, residual at the vacuous threshold: vacuous pass.
    r = evaluate_pair(0.05, 0.05 - 1e-12, 1.3e-16, 1.3e-16)
    assert r["residual"] == pytest.approx(1e-12, abs=1e-15)
    assert r["se_floor_applied"] is True
    assert r["sigma"] == 0.0
    assert r["pass"] is True
    # Collapsed SE, larger residual: sigma at the floored SE, failing
    # the unchanged 3.0-sigma band.
    r = evaluate_pair(0.05, 0.05 - 1e-10, 1.3e-16, 1.3e-16)
    assert r["se_floor_applied"] is True
    assert r["se_effective"] == FLOOR
    assert r["sigma"] == pytest.approx(100.0, rel=1e-6)
    assert r["pass"] is False
    # Collapsed SE, residual inside the band at the floored SE: pass.
    r = evaluate_pair(0.05, 0.05 - 2.5e-12, 1.3e-16, 1.3e-16)
    assert r["se_floor_applied"] is True
    assert r["sigma"] == pytest.approx(2.5, rel=1e-6)
    assert r["pass"] is True
    # Genuine SE: no floor, sigma at the combined SE.
    r = evaluate_pair(1.001, 1.0, 0.001, 0.001)
    assert r["se_floor_applied"] is False
    assert r["se_effective"] == r["se_combined"]
    assert r["se_combined"] == pytest.approx(
        math.sqrt(2.0) * 1e-3, rel=1e-15)
    assert r["sigma"] == pytest.approx(
        1e-3 / (math.sqrt(2.0) * 1e-3), rel=1e-12)
    assert r["pass"] is True
    assert r["tolerance_sigma"] == TOLERANCE_SIGMA


def test_se_floor_reevaluated_status():
    """The re-evaluation mirror of the floor rule on sealed-check
    rows: the R31 window-0 K0/Sigma+ float-precision SE artifact row
    (between-replica SE 1.28e-16, residual 3.3e-15) resolves to a
    vacuous pass, a larger residual at the collapsed SE fails at the
    floored SE, genuine SEs are untouched, and absent species carry
    no verdict."""
    reeval = _tm._reevaluated_status
    artifact = {
        "status": "fail",
        "se_combined": 1.2819751242557092e-16,
        "delta_observed": 0.06286598634487905,
        "delta_analytic_full": 0.06286598634487572,
    }
    status, sigma, passed = reeval(artifact)
    assert status == "vacuous_zero"
    assert sigma == 0.0
    assert passed is True
    # Larger residual at the collapsed SE: fail at the floored SE.
    big = dict(artifact)
    big["delta_analytic_full"] = artifact["delta_observed"] - 1e-10
    status, sigma, passed = reeval(big)
    assert status == "fail"
    assert sigma == pytest.approx(100.0, rel=1e-6)
    assert passed is False
    # Residual inside the band at the floored SE: pass.
    mid = dict(artifact)
    mid["delta_analytic_full"] = artifact["delta_observed"] - 2.5e-12
    status, sigma, passed = reeval(mid)
    assert status == "pass"
    assert sigma == pytest.approx(2.5, rel=1e-6)
    assert passed is True
    # Genuine SE, inside the band: pass; outside: fail.
    ok = {
        "status": "fail",
        "se_combined": 0.001,
        "delta_observed": 0.5,
        "delta_analytic_full": 0.5 - 0.002,
    }
    status, sigma, passed = reeval(ok)
    assert status == "pass"
    assert sigma == pytest.approx(2.0, rel=1e-12)
    assert passed is True
    bad = dict(ok)
    bad["delta_analytic_full"] = 0.5 - 0.004
    status, sigma, passed = reeval(bad)
    assert status == "fail"
    assert sigma == pytest.approx(4.0, rel=1e-12)
    assert passed is False
    # Absent species carries no verdict.
    status, sigma, passed = reeval({"status": "absent_species"})
    assert status == "absent_species"
    assert sigma is None
    assert passed is None


def test_deep_level_quadrature_noop_catalog_content():
    """The mechanism-(b) deep-level daughter-mass integration is a
    no-op on the committed catalog: the full record content of the
    five real-catalog equal-mass three-body cases is byte-identical
    to the pre-P1 engine output (the phase-1 pins, reused, not
    re-derived)."""
    catalog = HadronCatalog.from_json(CATALOG_PATH)
    for pdg, mass in CONTENT_PIN_CASES:
        record = catalog.get(pdg)
        record_set = RecordSet(catalog, [record])
        content = record_set.record_content(record, float(mass))
        assert _content_digest(content) == CONTENT_PIN_SHA256[
            f"{pdg}|{mass!r}"], (pdg, mass)
