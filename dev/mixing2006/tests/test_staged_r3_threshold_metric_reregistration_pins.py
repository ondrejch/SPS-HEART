"""Round 3 card 2: threshold-metric re-registration committed-run artifact
pins.

Task ``TASK-20260917-r3-gate2-followup-threshold-metric-reregistration``
(round 3, card 2; the Gate 2 closure obligation, follow-up 2 of the Round
2 closure record; nested feat commit 094ac2f, outer mirror c3d695d).
This module pins the sealed run evidence under
``dev/00runs/r3_threshold_metric_reregistration_2026-09-17/``: the
pre-registration (master seed 20260917 with ``draws: 0``), the
provenance, the two sealed-archive re-evaluation payloads
(``rereval_r21.json`` / ``rereval_r22.json``), the revised Gate 2
threshold clause report (``gate2_revised_threshold.json``), the two
metric-engine cross-check payloads
(``metric_engine_crosscheck_r21.json`` / ``metric_engine_crosscheck_r22.
json``), the seal manifest, ``SHA256SUMS.txt`` (7 entries,
run-dir-relative, ``sha256sum -c`` format), and the post-seal
clean-environment record.

The re-registration re-evaluates the sealed R21/R22 threshold-
discontinuity checks under the full analytic difference
``delta_analytic_full`` on both pair classes (the original metric
applied ``delta_analytic = 0`` at open-set-unchanged pairs).  The
committed verdict: 312 of the 323 original-metric failures pass at the
unchanged 3.0-sigma tolerance under ``delta_analytic_full`` (161 of 169
R21, 151 of 154 R22), the per-pair-class rates are 0 of 5 among the
original open-set-change failures and 312 of 318 among the original
open-set-unchanged failures, the maximum sigma is 4.9848 on R22
pair(23,24) pi_0 <p_T>_y, and the 11 residual failing pairs (5
open-set-change, 6 open-set-unchanged) are named with their one-line
diagnoses.  The sealed R21/R22 gate JSONs keep the original metric's
verdict (169 of 351 applied R21 checks fail, 154 of 483 R22) and are
byte-unchanged.

Sibling of ``tests/test_staged_r3_branching_weight_calibration_pins.py``
and ``tests/test_staged_r20_unit_grid_card10_pins.py`` (the same pin
style: sealed-dir discovery, skip-or-fail when the gitignored archive
is absent, digest pins plus structural pins).  These pins are read-only
over the sealed artifacts: nothing is recomputed, the sealed run is
never re-sealed, and the post-seal clean-environment record is never
re-executed.

Pins:

- ``test_sealed_run_dir_inventory``: the run dir holds exactly the ten
  top-level files (the seven sealed payloads, the seal manifest,
  ``SHA256SUMS.txt``, the post-seal clean-environment record); nothing
  else is present.
- ``test_pre_registration_structure``: the sealed pre-registration
  carries the committed contract (T0-equivalent deterministic
  re-evaluation, master seed 20260917 with ``draws: 0``, the
  ``delta_analytic_full`` metric statement on both pair classes at the
  unchanged 3.0-sigma tolerance, the two sealed archives with their
  run-card ids, the coverage set (19 R21 / 27 R22 adjacent pairs,
  351/483 applied, 169/154 original failures split 4+165 and 1+153),
  the three observables, the 95% acceptance criterion, the
  constraint record, and the five output files).
- ``test_provenance_structure``: the provenance records the catalog
  SHA-256, the sealed R21/R22 gate JSON digests and archive seal digests
  (103/143 files checked), the re-evaluation driver SHA-256 (nested and
  monorepo trees byte-identical), the metric module SHA-256, the card-9
  driver reference, the master seed with zero draws, the git identity,
  and the environment fingerprint.
- ``test_rereval_payload_structure``: each re-evaluation payload
  carries schema ``na49_mixing.threshold_metric.rereval/1`` and the
  committed field set: the re-registered metric block, the sealed
  archive reference, the full coverage (19 / 27 adjacent pairs, every
  one of the 18 stable species, all three observables, both pair
  classes), the per-check rows on the uniform 15-key field set (the
  re-registered ``delta_analytic_applied = delta_analytic_full`` on
  both classes, the signed-sigma convention, the original-metric sigma
  and status, the vacuous-zero and absent-species conventions), and the
  pinned summary (the original failure counts 169 / 154, the
  re-evaluation pass counts 343 / 480, the per-pair-class pass rates,
  161 / 151 original failures now passing, the maximum sigmas
  3.5987 / 4.9848, and the named residual checks).
- ``test_rereval_rows_mirror_sealed_gate_checks``: every re-evaluation
  row mirrors its sealed gate check bit-exactly (pair, species,
  observable, open-set flag, observed values, deltas, SE, the original
  sigma and status; the NaN rows mirror as NaN), and the original
  failure counts 169 / 154 are reproduced from the sealed gate JSONs.
- ``test_gate2_revised_threshold_structure``: the revised clause
  report carries the revision record (from the zero-applied metric to
  ``delta_analytic_full`` on both classes; the new-run metric of the
  card-9 driver), the unchanged tolerance, the per-run blocks
  consistent with the re-evaluation summaries, the combined block
  (834 applied, 823 pass, 312 of 323 original failures now passing,
  11 residuals: 0 of 5 open-set-change and 312 of 318
  open-set-unchanged, max 4.9848 on R22 pair(23,24) pi_0 <p_T>_y), the
  passing 95% acceptance (0.9659 achieved), the 11 named residual
  findings with their pair classes and one-line diagnoses, and
  ``no_fitting``.
- ``test_metric_engine_crosscheck_structure``: each cross-check payload
  compares the committed ``full_analytic_difference`` engine against
  the sealed per-check values on 572 / 816 checks, records the
  maximum absolute deviation per observable with the where-block
  (pair, species, observable, engine and sealed values), and the
  recorded bounds are self-consistent (the max equals the where-block
  deviation).
- ``test_seal_manifest_and_sha256_sums``: ``seal_manifest.json``
  records 7 sealed files totaling 1,407,181 bytes and the digest of
  ``SHA256SUMS.txt``, which is in ``sha256sum -c`` format with
  run-dir-relative paths, carries the pinned payload digests,
  verifies against every sealed file on disk, and self-consistently
  accounts for the full byte total.
- ``test_clean_environment_record``: the post-seal clean-environment
  record re-verified all 7 sums-covered files without mismatches and
  the clean-interpreter re-run reproduced the three sealed report
  payloads byte-for-byte (the recorded sealed and recomputed digests
  match, equal the pinned digests, and equal the digests of the files
  on disk).
- ``test_sealed_gate_jsons_byte_unchanged_and_original_verdicts``:
  the sealed ``gate2_r21.json`` / ``gate2_r22.json`` match their
  pinned SHA-256 digests (the digests recorded in the sealed
  archives' own ``SHA256SUMS.txt``) and carry the original metric's
  verdict (19 / 27 adjacent pairs, 351 / 483 applied checks, 169 /
  154 failed, the original maximum sigmas 372.04 / 252.43, the sealed
  form marked); the archive seal manifests agree with the on-disk
  sums files.

Hermetic: reads the sealed run dir (gitignored; the artifact is
present on disk) and, for the cross-archive pins, the sealed R21/R22
archives; no computation runs.  If the run dir is absent (a fresh
checkout), the artifact pins skip with a clear message; if an archive
is absent while the R3 run dir is present, the cross-archive pins skip
with a clear message.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from functools import lru_cache
from pathlib import Path

import pytest

NESTED_ROOT = Path(__file__).resolve().parent.parent
MONO_ROOT = NESTED_ROOT.parent.parent
RUN_NAME = "r3_threshold_metric_reregistration_2026-09-17"
RUN_DIR = MONO_ROOT / "dev" / "00runs" / RUN_NAME

R21_RUN_NAME = "r21_baryonic_source_grid_2026-09-13"
R22_RUN_NAME = "r22_mesonic_source_grid_2026-09-13"
R21_ARCHIVE = MONO_ROOT / "dev" / "00runs" / R21_RUN_NAME
R22_ARCHIVE = MONO_ROOT / "dev" / "00runs" / R22_RUN_NAME

CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
REREVAL_DRIVER_PATH = (
    NESTED_ROOT / "helpers" / "benchmarking" / "rereval_threshold_checks.py"
)
REREVAL_DRIVER_OUTER_PATH = (
    MONO_ROOT / "helpers" / "benchmarking" / "rereval_threshold_checks.py"
)
METRIC_MODULE_PATH = NESTED_ROOT / "na49_mixing" / "threshold_metric.py"
SOURCE_GRID_DRIVER_PATH = (
    NESTED_ROOT / "helpers" / "benchmarking" / "staged_source_grid.py"
)
SOURCE_GRID_DRIVER_OUTER_PATH = (
    MONO_ROOT / "helpers" / "benchmarking" / "staged_source_grid.py"
)

# ---------------------------------------------------------------------------
# The committed run contract (the sealed run realizes it).
# ---------------------------------------------------------------------------
RUN_ID = "R3-THRESHOLD-METRIC-REREGISTRATION"
CARD_ID = "TASK-20260917-r3-gate2-followup-threshold-metric-reregistration"
MASTER_SEED = 20260917
DRAWS = 0
N_SIGMA = 3.0
N_SEALED_FILES = 7
SEAL_TOTAL_BYTES = 1_407_181
N_SPECIES = 18
N_OBSERVABLES = 3

#: The 18 stable species of the card-9 record set (identical on the
#: baryonic and the mesonic grid).
SPECIES = [
    "K0L", "K0S", "K_0", "K_minus", "K_plus", "Lambda", "Sigma_0",
    "Sigma_minus", "Sigma_plus", "aK_0", "aLambda", "nbar", "neutron",
    "pbar", "pi_0", "pi_minus", "pi_plus", "proton",
]
OBSERVABLES = ["multiplicity", "mean_p", "mean_pt_y"]
PAIR_CLASSES = ["open_set_change", "open_set_unchanged"]

PRE_SCHEMA = "na49_mixing.rereval_threshold_checks.pre_registration/1"
PROV_SCHEMA = "na49_mixing.rereval_threshold_checks.provenance/1"
REREVAL_SCHEMA = "na49_mixing.threshold_metric.rereval/1"
REVISED_SCHEMA = "na49_mixing.rereval_threshold_checks.revised_clause/1"
CROSSCHECK_SCHEMA = "na49_mixing.rereval_threshold_checks.crosscheck/1"
SEAL_SCHEMA = "na49_mixing.rereval_threshold_checks.seal_manifest/1"
CLEAN_SCHEMA = "na49_mixing.rereval_threshold_checks.clean_environment/1"

R21_GATE_SCHEMA = "na49_mixing.card11_r21.discontinuity_verification/1"
R22_GATE_SCHEMA = "na49_mixing.card12_r22.discontinuity_verification/1"

#: The uniform per-check field set of the re-evaluation payloads.
REREVAL_CHECK_KEYS = {
    "pair", "species", "observable", "open_set_changes", "pair_class",
    "original_status", "status", "observed", "delta_observed",
    "delta_analytic_full", "delta_analytic_applied", "se_combined",
    "sigma_original", "sigma", "pass",
}
RESIDUAL_FINDING_KEYS = {
    "run", "pair", "species", "observable", "pair_class", "sigma",
    "residual", "diagnosis",
}

# ---------------------------------------------------------------------------
# The pinned input digests (the sealed run's own inputs, byte-stable).
# ---------------------------------------------------------------------------
CATALOG_SHA256 = (
    "6e4de7372a5bbf99affd6fa3ccc295c4cef09404dab3b17acd8816cda762bdcd"
)
R21_GATE_SHA256 = (
    "82af88d50e0b2be4da4e552a568181d1b6919f421190e7a54bea835556b07ac3"
)
R22_GATE_SHA256 = (
    "a2e81dbdb48d76c4a3c7e87436e9fb2e17f4a999918b261677ccbd2f16340865"
)
R21_ARCHIVE_SUMS_SHA256 = (
    "92f4070758ceac252e9209b01efa076c3351cbc54ed270d347b8be3fb6164537"
)
R22_ARCHIVE_SUMS_SHA256 = (
    "350c869f4e13a303e8bdb91ab8ad3d267bb60fe9a36348df79bee839f52c6fbc"
)
R21_ARCHIVE_SEAL_FILES = 103
R22_ARCHIVE_SEAL_FILES = 143
REREVAL_DRIVER_SHA256 = (
    "458b4ded99ba926e53ecfddb6ef8b8d33d16d25a4e7b75573ac1de1bd4cafb0c"
)
#: The sanctioned post-fix driver digest: the engine-fix card
#: TASK-20260918-r3-card2-engine-three-body-moment-fix phase 2 added
#: the --live-* re-evaluation modes to the dual-tree driver (additive;
#: the card-2 modes are byte-unchanged) and re-pinned the inventory;
#: the sealed card-2 dir above is byte-unchanged and records the
#: card-2-era digest.
REREVAL_DRIVER_POSTFIX_SHA256 = (
    "95f3ac6f0077b69c074eb701265d3a302fff09010f41641baf9c73cb6002b024"
)
METRIC_MODULE_SHA256 = (
    "5d39643d95bf80f8b4831d99b338574e25b95b48b0a9bb45c31ea7153f99aefb"
)
#: The sanctioned post-fix module digest: the engine-fix card
#: TASK-20260918-r3-card2-engine-three-body-moment-fix changed the
#: metric module (the exact Kallen three-body slot kinematics, the
#: deep-level daughter-mass quadrature, and the deterministic-observable
#: SE floor) and re-pinned the inventory; the sealed card-2 dir above is
#: byte-unchanged and records the pre-fix digest.
METRIC_MODULE_POSTFIX_SHA256 = (
    "ee240f44645d2e592acfae462f7d1e9c3b3dd8cd78c96bd40b7b73509f7fcc63"
)

# ---------------------------------------------------------------------------
# The pinned sealed-file digests (the sha256 pin set of this module).
# ---------------------------------------------------------------------------
PRE_REGISTRATION_SHA256 = (
    "79dfa38d86dd54aec7850bf5663187d79d44e3a6ca145a7ac1fa4f9782888ba5"
)
PROVENANCE_SHA256 = (
    "2f23314236385442083fbb353ea007716966a8df55a97979f64aebcdb8c10886"
)
REREVAL_R21_SHA256 = (
    "b2900f0721c204a8451c46a5ab2128faf212dbff7bafc3ce223300fc90d7d706"
)
REREVAL_R22_SHA256 = (
    "83f848129ba190a0bd96cb8e56e51fa1e4e17d679cc31d44e3762fd995b8686b"
)
REVISED_THRESHOLD_SHA256 = (
    "e9609d9d1819148e818651723e37006ef77a53aa8a75c4eb3f4941c592eb0267"
)
CROSSCHECK_R21_SHA256 = (
    "9601c1a694219c9ab80d450cf86a5c1b78a6deaf7cefa4b14e239ef4658b90c1"
)
CROSSCHECK_R22_SHA256 = (
    "af408612537aee2fd1b3827962a3144de5ad620a0e26835b340f818ccae1bc83"
)
SEAL_MANIFEST_SHA256 = (
    "b2ac9d17379fa1573696ecd1087be348d769a2df74eb6b229fba79422241a57d"
)
SUMS_FILE_SHA256 = (
    "5c7531ad41c66e660690c32d41f9e81fbe42f256abcdf299bc51390c7b359271"
)
CLEAN_ENVIRONMENT_SHA256 = (
    "6ff013fc9594f957ab2eea88332b6b1b98ceb9b439e41ff26f961004ebded885"
)

# ---------------------------------------------------------------------------
# The pinned re-evaluation verdicts (per run).
# ---------------------------------------------------------------------------
R21_OPEN_SET_CHANGES = [0, 2, 4, 5, 6]
R22_OPEN_SET_CHANGES = [0, 1, 2, 5, 6, 7, 8]
R21_MAX_ABS_SIGMA = 3.598708027565074
R21_MAX_ABS_SIGMA_CHECK = "pair(0,1) proton mean_p"
R22_MAX_ABS_SIGMA = 4.984844257802041
R22_MAX_ABS_SIGMA_CHECK = "pair(23,24) pi_0 mean_pt_y"

#: The pinned re-evaluation summaries (the sealed verdicts).
REREVAL_SUMMARY_PIN = {
    "r21": {
        "n_checks": 1026,  # 19 pairs x 18 species x 3 observables
        "n_checks_applied": 351,
        "n_pass": 343,
        "n_fail": 8,
        "n_vacuous_zero": 221,
        "n_absent_species": 454,
        "original": {
            "n_checks_failed": 169,
            "failed_at_open_set_change": 4,
            "failed_at_unchanged": 165,
        },
        "original_failures_total": 169,
        "original_failures_now_passing": 161,
        "pass_rate_among_original_failures": 0.9526627218934911,
        "failed_at_open_set_change": 4,
        "failed_at_unchanged": 4,
        "max_abs_sigma": R21_MAX_ABS_SIGMA,
        "max_abs_sigma_check": R21_MAX_ABS_SIGMA_CHECK,
        "pass_rate_all_applied": {
            "open_set_change": {
                "applied": 72, "pass": 68, "fail": 4,
                "pass_rate_all_applied": 0.9444444444444444,
                "original_failures": 4,
                "original_failures_now_passing": 0,
                "pass_rate_among_original_failures": 0.0,
            },
            "open_set_unchanged": {
                "applied": 279, "pass": 275, "fail": 4,
                "pass_rate_all_applied": 0.985663082437276,
                "original_failures": 165,
                "original_failures_now_passing": 161,
                "pass_rate_among_original_failures": 0.9757575757575757,
            },
        },
    },
    "r22": {
        "n_checks": 1458,  # 27 pairs x 18 species x 3 observables
        "n_checks_applied": 483,
        "n_pass": 480,
        "n_fail": 3,
        "n_vacuous_zero": 333,
        "n_absent_species": 642,
        "original": {
            "n_checks_failed": 154,
            "failed_at_open_set_change": 1,
            "failed_at_unchanged": 153,
        },
        "original_failures_total": 154,
        "original_failures_now_passing": 151,
        "pass_rate_among_original_failures": 0.9805194805194806,
        "failed_at_open_set_change": 1,
        "failed_at_unchanged": 2,
        "max_abs_sigma": R22_MAX_ABS_SIGMA,
        "max_abs_sigma_check": R22_MAX_ABS_SIGMA_CHECK,
        "pass_rate_all_applied": {
            "open_set_change": {
                "applied": 83, "pass": 82, "fail": 1,
                "pass_rate_all_applied": 0.9879518072289156,
                "original_failures": 1,
                "original_failures_now_passing": 0,
                "pass_rate_among_original_failures": 0.0,
            },
            "open_set_unchanged": {
                "applied": 400, "pass": 398, "fail": 2,
                "pass_rate_all_applied": 0.995,
                "original_failures": 153,
                "original_failures_now_passing": 151,
                "pass_rate_among_original_failures": 0.9869281045751634,
            },
        },
    },
}

#: The pinned original-metric verdicts of the sealed gate JSONs.
ORIGINAL_GATE_PIN = {
    "r21": {
        "adjacent_pairs_checked": 19,
        "n_checks_applied": 351,
        "n_checks_failed": 169,
        "n_failed_at_open_set_changes": 4,
        "n_failed_at_unchanged_pairs": 165,
        "n_checks_vacuous_zero": 221,
        "n_checks_absent_species": 454,
        "n_open_set_change_applied": 72,
        "max_abs_sigma": 372.04134193769096,
        "max_abs_sigma_check": "pair(1,2) pi_0 mean_p",
    },
    "r22": {
        "adjacent_pairs_checked": 27,
        "n_checks_applied": 483,
        "n_checks_failed": 154,
        "n_failed_at_open_set_changes": 1,
        "n_failed_at_unchanged_pairs": 153,
        "n_checks_vacuous_zero": 333,
        "n_checks_absent_species": 642,
        "n_open_set_change_applied": 83,
        "max_abs_sigma": 252.4311431125812,
        "max_abs_sigma_check": "pair(3,4) pi_plus mean_p",
    },
}

#: The pinned revised-clause combined block (312 of 323 original
#: failures now passing).
COMBINED_PIN = {
    "applied": 834,
    "pass": 823,
    "fail": 11,
    "original_failures": 323,
    "original_failures_now_passing": 312,
    "residual_failures": 11,
    "by_pair_class": {
        "open_set_change": {
            "original_failures": 5,
            "original_failures_now_passing": 0,
            "residual_failures": 5,
            "pass_rate_among_original_failures": 0.0,
        },
        "open_set_unchanged": {
            "original_failures": 318,
            "original_failures_now_passing": 312,
            "residual_failures": 6,
            "pass_rate_among_original_failures": 0.9811320754716981,
        },
    },
    "max_abs_sigma": R22_MAX_ABS_SIGMA,
    "max_abs_sigma_check": "R22 pair(23,24) pi_0 mean_pt_y",
    "pass_rate_among_original_failures": 0.9659442724458205,
    "pass_rate_all_applied": 0.986810551558753,
}

#: The 11 named residual failing pairs, with their pair classes (5
#: open-set-change, 6 open-set-unchanged).
RESIDUAL_PAIRS_PIN = [
    ("R21", (0, 1), "proton", "mean_p", "open_set_change"),
    ("R21", (0, 1), "proton", "mean_pt_y", "open_set_change"),
    ("R21", (1, 2), "pi_0", "mean_pt_y", "open_set_unchanged"),
    ("R21", (3, 4), "pi_0", "mean_pt_y", "open_set_unchanged"),
    ("R21", (4, 5), "pi_minus", "mean_pt_y", "open_set_change"),
    ("R21", (5, 6), "pi_0", "mean_pt_y", "open_set_change"),
    ("R21", (15, 16), "pi_0", "mean_p", "open_set_unchanged"),
    ("R21", (18, 19), "pi_0", "multiplicity", "open_set_unchanged"),
    ("R22", (7, 8), "pi_minus", "mean_pt_y", "open_set_change"),
    ("R22", (15, 16), "pi_plus", "mean_pt_y", "open_set_unchanged"),
    ("R22", (23, 24), "pi_0", "mean_pt_y", "open_set_unchanged"),
]

#: The pinned metric-engine cross-check payloads.
CROSSCHECK_PIN = {
    "r21": {
        "n_checks_applied": 572,
        "max_abs_deviation": {
            "multiplicity": {
                "max": 1.0470213190894029e-07,
                "where": {
                    "pair": [4, 5], "species": "pi_0",
                    "observable": "multiplicity",
                    "engine": 0.12938328075, "sealed": 0.129383176048,
                },
            },
            "mean_p": {
                "max": 0.0010346239650864675,
                "where": {
                    "pair": [7, 8], "species": "pi_minus",
                    "observable": "mean_p",
                    "engine": 0.020181277662, "sealed": 0.019146653697,
                },
            },
            "mean_pt_y": {
                "max": 0.00115295674763502,
                "where": {
                    "pair": [5, 6], "species": "pi_minus",
                    "observable": "mean_pt_y",
                    "engine": 0.02760781351, "sealed": 0.028760770257,
                },
            },
        },
    },
    "r22": {
        "n_checks_applied": 816,
        "max_abs_deviation": {
            "multiplicity": {
                "max": 8.520139052170528e-06,
                "where": {
                    "pair": [7, 8], "species": "pi_0",
                    "observable": "multiplicity",
                    "engine": -0.145649287768, "sealed": -0.145640767629,
                },
            },
            "mean_p": {
                "max": 0.00031546392915315913,
                "where": {
                    "pair": [9, 10], "species": "pi_0",
                    "observable": "mean_p",
                    "engine": 0.018994842265, "sealed": 0.019310306194,
                },
            },
            "mean_pt_y": {
                "max": 0.0005148188409500876,
                "where": {
                    "pair": [9, 10], "species": "pi_0",
                    "observable": "mean_pt_y",
                    "engine": 0.016809756518, "sealed": 0.017324575359,
                },
            },
        },
    },
}

SUMS_TOP_LEVEL_PAYLOADS = (
    "pre_registration.json",
    "provenance.json",
    "rereval_r21.json",
    "rereval_r22.json",
    "gate2_revised_threshold.json",
    "metric_engine_crosscheck_r21.json",
    "metric_engine_crosscheck_r22.json",
)
SUMS_PAYLOAD_SHA256 = {
    "pre_registration.json": PRE_REGISTRATION_SHA256,
    "provenance.json": PROVENANCE_SHA256,
    "rereval_r21.json": REREVAL_R21_SHA256,
    "rereval_r22.json": REREVAL_R22_SHA256,
    "gate2_revised_threshold.json": REVISED_THRESHOLD_SHA256,
    "metric_engine_crosscheck_r21.json": CROSSCHECK_R21_SHA256,
    "metric_engine_crosscheck_r22.json": CROSSCHECK_R22_SHA256,
}
#: The three sealed report payloads the clean-environment record
#: re-verified byte-for-byte (sealed digest == recomputed digest).
CLEAN_SEALED_REPORT_SHA256 = {
    "rereval_r21.json": REREVAL_R21_SHA256,
    "rereval_r22.json": REREVAL_R22_SHA256,
    "gate2_revised_threshold.json": REVISED_THRESHOLD_SHA256,
}

TOP_LEVEL_FILES = (
    "pre_registration.json",
    "provenance.json",
    "rereval_r21.json",
    "rereval_r22.json",
    "gate2_revised_threshold.json",
    "metric_engine_crosscheck_r21.json",
    "metric_engine_crosscheck_r22.json",
    "seal_manifest.json",
    "SHA256SUMS.txt",
    "clean_environment_commands.json",
)

#: ``sha256sum -c`` line: 64 hex digits, two spaces, the relative path.
SUMS_LINE = re.compile(r"^[0-9a-f]{64}  \S+$")
UTC_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(name: str) -> dict:
    return json.loads((RUN_DIR / name).read_text(encoding="utf-8"))


def _canonical_sha256(payload: dict) -> str:
    """The driver's canonical report hash (indent-2 JSON, trailing
    newline, UTF-8)."""
    return hashlib.sha256(
        (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    ).hexdigest()


def _is_nan(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _values_equal(a: object, b: object) -> bool:
    """NaN-tolerant deep equality: NaN mirrors as NaN (the absent-
    species rows carry NaN deltas and a two-element NaN observed
    vector), ``None`` mirrors as ``None``; everything else compares
    with ``==``."""
    if _is_nan(a) and _is_nan(b):
        return True
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            _values_equal(x, y) for x, y in zip(a, b)
        )
    return a == b


@lru_cache(maxsize=1)
def _rereval_payloads() -> dict[str, dict]:
    return {key: _load(f"rereval_{key}.json") for key in ("r21", "r22")}


@pytest.fixture(scope="module")
def run_dir():
    """The sealed R3 run dir, skipped with a clear message when the
    gitignored artifact is absent or incomplete (a fresh checkout)."""
    if not (RUN_DIR / "pre_registration.json").is_file():
        pytest.skip(
            f"the sealed R3 run dir {RUN_DIR} is absent (gitignored "
            "artifact); restore the sealed run to execute the "
            "artifact pins"
        )
    missing = [name for name in TOP_LEVEL_FILES if not (RUN_DIR / name).is_file()]
    if missing:
        pytest.skip(
            f"the sealed R3 run dir {RUN_DIR} is incomplete (missing: "
            + ", ".join(missing) + "); restore the sealed run to "
            "execute the artifact pins"
        )
    return RUN_DIR


def _archive_fixture(archive: Path, run_name: str, gate_file: str):
    """The sealed archive (gate JSON, seal manifest, sums file),
    skipped with a clear message when the gitignored artifact is
    absent while the R3 run dir is present."""
    required = (gate_file, "seal_manifest.json", "SHA256SUMS.txt")
    if not (archive / gate_file).is_file():
        pytest.skip(
            f"the sealed {run_name[:3].upper()} archive {archive} is "
            "absent (gitignored artifact); restore it to execute the "
            "cross-archive pins"
        )
    missing = [name for name in required if not (archive / name).is_file()]
    if missing:
        pytest.skip(
            f"the sealed {run_name[:3].upper()} archive {archive} is "
            "incomplete (missing: " + ", ".join(missing)
            + "); restore it to execute the cross-archive pins"
        )
    return archive


@pytest.fixture(scope="module")
def r21_archive():
    return _archive_fixture(R21_ARCHIVE, R21_RUN_NAME, "gate2_r21.json")


@pytest.fixture(scope="module")
def r22_archive():
    return _archive_fixture(R22_ARCHIVE, R22_RUN_NAME, "gate2_r22.json")


# ---------------------------------------------------------------------------
# Sealed artifact pins
# ---------------------------------------------------------------------------
def test_sealed_run_dir_inventory(run_dir):
    """The run dir holds exactly the ten top-level files (the seven
    sealed payloads, the seal manifest, the sums file, the post-seal
    clean-environment record); nothing else is present."""
    on_disk = {path.name for path in run_dir.iterdir()}
    assert on_disk == set(TOP_LEVEL_FILES), (
        "the sealed run dir drifted from the committed 10-file top "
        "level set"
    )


def test_pre_registration_structure(run_dir):
    """The sealed pre-registration carries the committed contract: the
    T0-equivalent deterministic re-evaluation (master seed 20260917
    with ``draws: 0``), the ``delta_analytic_full`` metric statement
    on both pair classes at the unchanged 3.0-sigma tolerance, the two
    sealed archives with their run-card ids, the coverage set (19 R21
    / 27 R22 adjacent pairs, 351/483 applied, 169/154 original
    failures split 4+165 and 1+153, 323 in total), the three
    observables, the 95% acceptance criterion, the constraint record,
    and the five output files."""
    pre = _load("pre_registration.json")
    assert pre["schema"] == PRE_SCHEMA
    assert pre["run"] == RUN_ID
    assert pre["card"] == CARD_ID
    assert (
        pre["mode"]
        == "diagnostic re-evaluation over the sealed archives (no "
           "Monte Carlo, no generation)"
    )
    assert pre["master_seed"] == MASTER_SEED
    assert pre["draws"] == DRAWS
    assert "no random draws" in pre["seed_note"]

    metric = pre["metric"]
    assert set(metric) == {
        "id", "statement", "tolerance_sigma", "tolerance_unchanged",
        "revises", "source", "engine",
    }
    assert metric["id"] == "delta_analytic_full"
    assert metric["tolerance_sigma"] == N_SIGMA
    assert metric["tolerance_unchanged"] is True
    assert "BOTH pair classes" in metric["statement"]
    assert "delta_analytic = 0 at open-set-unchanged pairs" in metric[
        "revises"
    ]
    assert "read-only" in metric["source"]
    assert metric["engine"] == "na49_mixing.threshold_metric.rereval_report"

    archives = pre["archives"]
    assert set(archives) == {"r21", "r22"}
    assert archives["r21"] == {
        "run": "R21",
        "path": f"dev/00runs/{R21_RUN_NAME}",
        "gate_file": "gate2_r21.json",
        "card": "TASK-20260913-r2-run-r21-baryonic-source-grid",
        "read_only": True,
    }
    assert archives["r22"] == {
        "run": "R22",
        "path": f"dev/00runs/{R22_RUN_NAME}",
        "gate_file": "gate2_r22.json",
        "card": "TASK-20260913-r2-run-r22-mesonic-source-grid",
        "read_only": True,
    }

    coverage = pre["coverage"]
    assert set(coverage) == {
        "r21", "r22", "observables", "pair_classes",
        "total_original_failures",
    }
    assert coverage["r21"] == {
        "adjacent_pairs": 19,
        "applied_checks_original": 351,
        "failed_original": 169,
        "failed_at_open_set_change": 4,
        "failed_at_unchanged": 165,
    }
    assert coverage["r22"] == {
        "adjacent_pairs": 27,
        "applied_checks_original": 483,
        "failed_original": 154,
        "failed_at_open_set_change": 1,
        "failed_at_unchanged": 153,
    }
    assert coverage["observables"] == OBSERVABLES
    assert coverage["pair_classes"] == PAIR_CLASSES
    assert coverage["total_original_failures"] == 323

    acceptance = pre["acceptance"]
    assert set(acceptance) == {"criterion", "threshold",
                               "n_original_failures"}
    assert acceptance["threshold"] == 0.95
    assert acceptance["n_original_failures"] == 323
    assert "95%" in acceptance["criterion"]

    constraints = pre["constraints"]
    assert set(constraints) == {
        "no_fitting", "sealed_archives_read_only", "tolerance_relaxed",
        "residual_policy",
    }
    assert constraints["no_fitting"] is True
    assert constraints["sealed_archives_read_only"] is True
    assert constraints["tolerance_relaxed"] is False
    assert "one-line diagnosis" in constraints["residual_policy"]

    assert pre["outputs"] == [
        "rereval_r21.json",
        "rereval_r22.json",
        "gate2_revised_threshold.json",
        "metric_engine_crosscheck_r21.json",
        "metric_engine_crosscheck_r22.json",
    ]
    assert pre["written_before_evaluation"] is True

    # the sealed pre-registration is byte-pinned
    assert _file_sha256(run_dir / "pre_registration.json") == (
        PRE_REGISTRATION_SHA256
    )


def test_provenance_structure(run_dir, r21_archive, r22_archive):
    """The provenance records the catalog SHA-256, the sealed R21/R22
    gate JSON digests and archive seal digests (103/143 files
    checked), the re-evaluation driver SHA-256 (the nested and
    monorepo trees are byte-identical), the metric module SHA-256, the
    card-9 driver reference, the master seed with zero draws, the git
    identity, and the environment fingerprint."""
    prov = _load("provenance.json")
    assert prov["schema"] == PROV_SCHEMA
    assert prov["run"] == RUN_ID
    assert prov["card"] == CARD_ID
    assert UTC_STAMP.match(prov["written_utc"])

    assert prov["catalog"] == {
        "path": "config/v2_singlet/resonance_catalog.json",
        "sha256": CATALOG_SHA256,
    }
    assert prov["catalog"]["sha256"] == _file_sha256(CATALOG_PATH)

    sealed = prov["sealed_archives"]
    assert set(sealed) == {"r21", "r22"}
    assert sealed["r21"] == {
        "path": f"dev/00runs/{R21_RUN_NAME}",
        "gate_file": "gate2_r21.json",
        "gate_sha256": R21_GATE_SHA256,
        "seal": {
            "files_checked": R21_ARCHIVE_SEAL_FILES,
            "sha256_sums_sha256": R21_ARCHIVE_SUMS_SHA256,
        },
    }
    assert sealed["r22"] == {
        "path": f"dev/00runs/{R22_RUN_NAME}",
        "gate_file": "gate2_r22.json",
        "gate_sha256": R22_GATE_SHA256,
        "seal": {
            "files_checked": R22_ARCHIVE_SEAL_FILES,
            "sha256_sums_sha256": R22_ARCHIVE_SUMS_SHA256,
        },
    }
    # the recorded gate digests match the sealed archives on disk
    assert _file_sha256(r21_archive / "gate2_r21.json") == R21_GATE_SHA256
    assert _file_sha256(r22_archive / "gate2_r22.json") == R22_GATE_SHA256
    assert _file_sha256(r21_archive / "SHA256SUMS.txt") == (
        R21_ARCHIVE_SUMS_SHA256
    )
    assert _file_sha256(r22_archive / "SHA256SUMS.txt") == (
        R22_ARCHIVE_SUMS_SHA256
    )

    drivers = prov["drivers"]
    assert set(drivers) == {
        "rereval_driver", "rereval_driver_sha256", "metric_module",
        "metric_module_sha256", "source_grid_driver",
    }
    assert drivers["rereval_driver"] == (
        "helpers/benchmarking/rereval_threshold_checks.py"
    )
    assert drivers["rereval_driver_sha256"] == REREVAL_DRIVER_SHA256
    assert drivers["metric_module"] == (
        "dev/mixing2006/na49_mixing/threshold_metric.py"
    )
    assert drivers["metric_module_sha256"] == METRIC_MODULE_SHA256
    assert drivers["source_grid_driver"] == (
        "helpers/benchmarking/staged_source_grid.py"
    )
    # dual-tree re-evaluation driver: the nested and monorepo copies
    # are byte-identical and carry the pinned digest
    assert _file_sha256(REREVAL_DRIVER_PATH) == (
        _file_sha256(REREVAL_DRIVER_OUTER_PATH))
    # The sealed provenance records the card-2 driver digest; the
    # on-disk driver carries the sanctioned post-fix digest of the
    # engine-fix card phase 2 (the re-pinned inventory).
    assert _file_sha256(REREVAL_DRIVER_PATH) == (
        REREVAL_DRIVER_POSTFIX_SHA256)
    # The sealed provenance records the card-2 module digest; the
    # on-disk module carries the sanctioned post-fix digest of the
    # engine-fix card (the re-pinned inventory).
    assert _file_sha256(METRIC_MODULE_PATH) == METRIC_MODULE_POSTFIX_SHA256
    # the card-9 driver is present in both trees and byte-identical
    assert _file_sha256(SOURCE_GRID_DRIVER_PATH) == (
        _file_sha256(SOURCE_GRID_DRIVER_OUTER_PATH)
    )

    assert prov["master_seed"] == MASTER_SEED
    assert prov["draws"] == DRAWS
    for key in ("nested_head", "nested_dirty_entries",
                "outer_head", "outer_dirty_entries"):
        assert key in prov["git"]
    for key in ("python_version", "python_implementation", "platform",
                "machine", "numpy_version", "executable"):
        assert key in prov["environment"]

    assert _file_sha256(run_dir / "provenance.json") == PROVENANCE_SHA256


def test_rereval_payload_structure(run_dir):
    """Each re-evaluation payload carries schema
    ``na49_mixing.threshold_metric.rereval/1`` and the committed field
    set: the re-registered metric block, the sealed archive reference,
    the full coverage (19 / 27 adjacent pairs, every one of the 18
    stable species, all three observables, both pair classes), the
    per-check rows on the uniform 15-key field set (the re-registered
    ``delta_analytic_applied = delta_analytic_full`` on both classes,
    the absolute-sigma convention at the unchanged 3.0 tolerance, the
    original-metric sigma and status, the vacuous-zero and
    absent-species conventions), and the pinned summary (the original
    failure counts 169 / 154, the re-evaluation pass counts 343 / 480,
    the per-pair-class pass rates, 161 / 151 original failures now
    passing, the maximum sigmas, and the named residual checks)."""
    for key, (pairs, open_set_changes, pinned) in (
        ("r21", (19, R21_OPEN_SET_CHANGES, REREVAL_SUMMARY_PIN["r21"])),
        ("r22", (27, R22_OPEN_SET_CHANGES, REREVAL_SUMMARY_PIN["r22"])),
    ):
        payload = _load(f"rereval_{key}.json")
        assert payload["schema"] == REREVAL_SCHEMA
        assert payload["run"] == key.upper()
        assert payload["card"] == CARD_ID

        metric = payload["metric"]
        assert set(metric) == {"id", "statement", "tolerance_sigma",
                               "source"}
        assert metric["id"] == "delta_analytic_full"
        assert metric["tolerance_sigma"] == N_SIGMA
        assert "BOTH pair classes" in metric["statement"]

        archive = payload["archive"]
        assert set(archive) == {"path", "gate_sha256"}
        assert archive["path"] == (
            f"dev/00runs/{(R21_RUN_NAME if key == 'r21' else R22_RUN_NAME)}/"
            f"gate2_{key}.json"
        )
        assert archive["gate_sha256"] in (R21_GATE_SHA256, R22_GATE_SHA256)

        coverage = payload["coverage"]
        assert set(coverage) == {
            "adjacent_pairs_checked", "open_set_changes", "n_checks",
            "observables",
        }
        assert coverage["adjacent_pairs_checked"] == pairs
        assert coverage["open_set_changes"] == open_set_changes
        assert coverage["n_checks"] == pairs * N_SPECIES * N_OBSERVABLES
        assert coverage["observables"] == OBSERVABLES

        checks = payload["checks"]
        assert len(checks) == pinned["n_checks"]
        # full pair/species/observable coverage: every adjacent pair,
        # every stable species, every observable, exactly once each
        seen = {
            (tuple(check["pair"]), check["species"], check["observable"])
            for check in checks
        }
        expected = {(k, k + 1) for k in range(pairs)}
        assert {pair for pair, _sp, _obs in seen} == expected
        assert len(seen) == pairs * N_SPECIES * N_OBSERVABLES
        assert {species for _pair, species, _obs in seen} == set(SPECIES)
        assert {obs for _pair, _sp, obs in seen} == set(OBSERVABLES)

        n_applied = n_pass = n_fail = n_vac = n_abs = 0
        for check in checks:
            assert set(check) == REREVAL_CHECK_KEYS
            pair = check["pair"]
            assert pair[1] == pair[0] + 1, (
                f"{key}: non-adjacent pair {pair}"
            )
            assert pair[0] in range(pairs)
            is_change = pair[0] in open_set_changes
            assert check["open_set_changes"] is is_change
            assert check["pair_class"] == (
                "open_set_change" if is_change else "open_set_unchanged"
            )
            assert check["pair_class"] in PAIR_CLASSES
            status = check["status"]
            if status in ("pass", "fail"):
                n_applied += 1
                n_pass += check["status"] == "pass"
                n_fail += check["status"] == "fail"
                # the re-registered metric: the applied value is the
                # full analytic difference on BOTH pair classes
                assert check["delta_analytic_applied"] == (
                    check["delta_analytic_full"]
                )
                se = check["se_combined"]
                assert se > 0.0
                # the absolute-sigma convention at the unchanged
                # tolerance (signed sigma is not stored)
                expected_sigma = abs(
                    check["delta_observed"] - check["delta_analytic_full"]
                ) / se
                assert check["sigma"] == pytest.approx(
                    expected_sigma, rel=1e-12
                )
                assert check["pass"] == (check["sigma"] <= N_SIGMA)
                assert check["status"] == (
                    "pass" if check["pass"] else "fail"
                )
                # the original metric: delta_analytic applied only at
                # open-set-change pairs, zero at unchanged pairs
                d_orig = (
                    check["delta_analytic_full"]
                    if is_change else 0.0
                )
                expected_original = (
                    abs(check["delta_observed"] - d_orig) / se
                )
                assert check["sigma_original"] == pytest.approx(
                    expected_original, rel=1e-12
                )
                assert check["original_status"] == (
                    "fail"
                    if check["sigma_original"] > N_SIGMA
                    else "pass"
                )
            elif status == "vacuous_zero":
                n_vac += 1
                assert check["observed"] == [0.0, 0.0]
                assert check["delta_observed"] == 0.0
                assert check["delta_analytic_full"] == 0.0
                assert check["delta_analytic_applied"] == 0.0
                assert check["se_combined"] == 0.0
                assert check["sigma"] == 0.0
                assert check["sigma_original"] == 0.0
                assert check["pass"] is True
                assert check["original_status"] == "vacuous_zero"
            elif status == "absent_species":
                n_abs += 1
                # the moment presence rule: only the momentum moments
                # can be absent (the multiplicity check always applies)
                assert check["observable"] in ("mean_p", "mean_pt_y")
                # the species is absent at at least one of the two
                # adjacent source-mass points: the absent side is NaN,
                # the present side keeps its moment value
                assert any(_is_nan(value) for value in check["observed"])
                assert _is_nan(check["delta_observed"])
                assert _is_nan(check["delta_analytic_full"])
                assert _is_nan(check["delta_analytic_applied"])
                assert check["se_combined"] is None
                assert check["sigma"] is None
                assert check["sigma_original"] is None
                assert check["pass"] is None
                assert check["original_status"] == "absent_species"
            else:
                raise AssertionError(f"{key}: unknown status {status}")

        summary = payload["summary"]
        assert set(summary) == {
            "n_checks", "n_checks_applied", "n_pass", "n_fail",
            "n_vacuous_zero", "n_absent_species", "original", "reeval",
            "max_abs_sigma", "max_abs_sigma_check", "residuals",
        }
        assert summary["n_checks"] == pinned["n_checks"]
        assert summary["n_checks_applied"] == pinned["n_checks_applied"]
        assert summary["n_pass"] == pinned["n_pass"]
        assert summary["n_fail"] == pinned["n_fail"]
        assert summary["n_vacuous_zero"] == pinned["n_vacuous_zero"]
        assert summary["n_absent_species"] == pinned["n_absent_species"]
        assert (n_applied, n_pass, n_fail, n_vac, n_abs) == (
            pinned["n_checks_applied"], pinned["n_pass"], pinned["n_fail"],
            pinned["n_vacuous_zero"], pinned["n_absent_species"],
        )
        # the original-metric verdict, as recorded by the sealed gate
        assert summary["original"] == pinned["original"]
        assert sum(1 for check in checks
                   if check["original_status"] == "fail") == (
            pinned["original"]["n_checks_failed"]
        )
        reeval = summary["reeval"]
        assert set(reeval) == {
            "pass_rate_all_applied", "original_failures_total",
            "original_failures_now_passing",
            "pass_rate_among_original_failures",
            "failed_at_open_set_change", "failed_at_unchanged",
        }
        assert reeval["pass_rate_all_applied"] == (
            pinned["pass_rate_all_applied"]
        )
        assert reeval["original_failures_total"] == (
            pinned["original_failures_total"]
        )
        assert reeval["original_failures_now_passing"] == (
            pinned["original_failures_now_passing"]
        )
        assert reeval["pass_rate_among_original_failures"] == (
            pinned["pass_rate_among_original_failures"]
        )
        assert reeval["failed_at_open_set_change"] == (
            pinned["failed_at_open_set_change"]
        )
        assert reeval["failed_at_unchanged"] == (
            pinned["failed_at_unchanged"]
        )
        assert summary["max_abs_sigma"] == pinned["max_abs_sigma"]
        assert summary["max_abs_sigma_check"] == (
            pinned["max_abs_sigma_check"]
        )

        # the named residual checks: exactly the failing checks, with
        # the signed residual magnitude and the combined SE
        failing = [check for check in checks if check["status"] == "fail"]
        residuals = summary["residuals"]
        assert len(residuals) == len(failing) == pinned["n_fail"]
        for residual in residuals:
            match = [
                check for check in failing
                if check["pair"] == residual["pair"]
                and check["species"] == residual["species"]
                and check["observable"] == residual["observable"]
            ]
            assert len(match) == 1
            check = match[0]
            assert check["sigma"] == residual["sigma"]
            assert check["se_combined"] == residual["se_combined"]
            assert residual["residual"] == pytest.approx(
                abs(check["delta_observed"]
                    - check["delta_analytic_full"]), rel=1e-12
            )
            assert residual["sigma"] > N_SIGMA
            assert residual["pair_class"] == check["pair_class"]
            assert isinstance(residual["diagnosis"], str)
            assert residual["diagnosis"]

        # the sealed payload is byte-pinned and in canonical form
        pinned_digest = {
            "r21": REREVAL_R21_SHA256, "r22": REREVAL_R22_SHA256,
        }[key]
        assert _file_sha256(run_dir / f"rereval_{key}.json") == (
            pinned_digest
        )
        assert _canonical_sha256(payload) == pinned_digest


def test_rereval_rows_mirror_sealed_gate_checks(
    run_dir, r21_archive, r22_archive
):
    """Every re-evaluation row mirrors its sealed gate check
    bit-exactly (pair, species, observable, open-set flag, observed
    values, deltas, SE, the original sigma and status; the NaN rows
    mirror as NaN), and the original failure counts 169 / 154 are
    reproduced from the sealed gate JSONs (the sealed archives are
    read-only and are never re-sealed)."""
    for key, archive in (("r21", r21_archive), ("r22", r22_archive)):
        gate = json.loads(
            (archive / f"gate2_{key}.json").read_text(encoding="utf-8")
        )
        clause = gate["clauses"]["threshold_discontinuity"]
        gate_checks = clause["checks"]
        payload = _load(f"rereval_{key}.json")
        rows = payload["checks"]
        assert len(rows) == len(gate_checks)
        for gate_check, row in zip(gate_checks, rows):
            assert row["pair"] == [int(k) for k in gate_check["pair"]]
            assert row["species"] == gate_check["species"]
            assert row["observable"] == gate_check["observable"]
            assert row["open_set_changes"] == gate_check["open_set_changes"]
            assert _values_equal(row["observed"], gate_check["observed"])
            assert _values_equal(
                row["delta_observed"], gate_check["delta_observed"]
            )
            assert _values_equal(
                row["delta_analytic_full"], gate_check["delta_analytic_full"]
            )
            assert _values_equal(
                row["se_combined"], gate_check["se_combined"]
            )
            assert _values_equal(
                row["sigma_original"], gate_check["sigma"]
            )
            assert row["original_status"] == gate_check["status"]
            # the re-registration: the applied value is the full
            # analytic difference on both pair classes (the NaN
            # absent-species rows mirror as NaN)
            assert _values_equal(
                row["delta_analytic_applied"],
                gate_check["delta_analytic_full"],
            )
        # the original-metric failure counts reproduced from the
        # sealed gate JSON
        original = payload["summary"]["original"]
        assert original["n_checks_failed"] == clause["n_checks_failed"]
        assert original["failed_at_open_set_change"] == (
            clause["n_failed_at_open_set_changes"]
        )
        assert original["failed_at_unchanged"] == (
            clause["n_failed_at_unchanged_pairs"]
        )
        assert (original["n_checks_failed"], clause["n_checks_applied"]) in (
            (169, 351), (154, 483)
        )


def test_gate2_revised_threshold_structure(run_dir, r21_archive, r22_archive):
    """The revised clause report carries the revision record (from the
    zero-applied metric to ``delta_analytic_full`` on both classes;
    the new-run metric of the card-9 driver), the unchanged tolerance,
    the per-run blocks consistent with the re-evaluation summaries,
    the combined block (834 applied, 823 pass, 312 of 323 original
    failures now passing, 11 residuals: 0 of 5 open-set-change and
    312 of 318 open-set-unchanged, max 4.9848 on R22 pair(23,24) pi_0
    <p_T>_y), the passing 95% acceptance (0.9659 achieved), the 11
    named residual findings with their pair classes and one-line
    diagnoses, and ``no_fitting``."""
    report = _load("gate2_revised_threshold.json")
    assert report["schema"] == REVISED_SCHEMA
    assert report["card"] == CARD_ID
    assert report["gate"] == "gate 2 (plan section 8, line 575)"
    assert report["clause"] == "threshold_discontinuity"

    revision = report["revision"]
    assert set(revision) == {"from", "to", "record", "new_run_metric"}
    assert "delta_analytic = 0 at open-set-unchanged pairs" in revision[
        "from"
    ]
    assert "delta_analytic_full on both pair classes" in revision["to"]
    assert "round2-closure-2026-09-17.md" in revision["record"]
    assert "DISCONTINUITY_METRIC" in revision["new_run_metric"]
    assert "staged_source_grid.py" in revision["new_run_metric"]
    assert "R31" in revision["new_run_metric"]

    tolerance = report["tolerance"]
    assert set(tolerance) == {"sigma", "unchanged", "se_combined",
                              "relaxed"}
    assert tolerance["sigma"] == N_SIGMA
    assert tolerance["unchanged"] is True
    assert tolerance["relaxed"] is False
    assert tolerance["se_combined"] == "sqrt(SE_k^2 + SE_{k+1}^2)"

    assert set(report["inputs"]) == {"rereval_r21_sha256_note"}

    runs = report["runs"]
    assert set(runs) == {"r21", "r22"}
    payloads = _rereval_payloads()
    for key, (gate_sha256, pinned) in (
        ("r21", (R21_GATE_SHA256, REREVAL_SUMMARY_PIN["r21"])),
        ("r22", (R22_GATE_SHA256, REREVAL_SUMMARY_PIN["r22"])),
    ):
        run_block = runs[key]
        assert set(run_block) == {
            "run", "archive", "n_checks", "n_checks_applied", "n_pass",
            "n_fail", "original_failures",
            "original_failures_now_passing",
            "pass_rate_among_original_failures", "max_abs_sigma",
            "max_abs_sigma_check", "residual_failures", "by_pair_class",
        }
        assert run_block["run"] == key.upper()
        assert run_block["archive"] == {
            "path": f"dev/00runs/{(R21_RUN_NAME if key == 'r21' else R22_RUN_NAME)}/gate2_{key}.json",
            "gate_sha256": gate_sha256,
        }
        # the gate digest matches the sealed archive on disk
        archive = r21_archive if key == "r21" else r22_archive
        assert _file_sha256(archive / f"gate2_{key}.json") == gate_sha256

        summary = payloads[key]["summary"]
        assert run_block["n_checks"] == summary["n_checks"] == (
            pinned["n_checks"]
        )
        assert run_block["n_checks_applied"] == (
            summary["n_checks_applied"]
        ) == pinned["n_checks_applied"]
        assert run_block["n_pass"] == summary["n_pass"] == pinned["n_pass"]
        assert run_block["n_fail"] == summary["n_fail"] == pinned["n_fail"]
        assert run_block["original_failures"] == (
            summary["original"]["n_checks_failed"]
        ) == pinned["original"]["n_checks_failed"]
        assert run_block["original_failures_now_passing"] == (
            summary["reeval"]["original_failures_now_passing"]
        ) == pinned["original_failures_now_passing"]
        assert run_block["pass_rate_among_original_failures"] == (
            summary["reeval"]["pass_rate_among_original_failures"]
        ) == pinned["pass_rate_among_original_failures"]
        assert run_block["max_abs_sigma"] == summary["max_abs_sigma"] == (
            pinned["max_abs_sigma"]
        )
        assert run_block["max_abs_sigma_check"] == (
            summary["max_abs_sigma_check"]
        ) == pinned["max_abs_sigma_check"]
        assert run_block["residual_failures"] == len(
            summary["residuals"]
        ) == pinned["n_fail"]
        assert run_block["by_pair_class"] == (
            summary["reeval"]["pass_rate_all_applied"]
        )

    combined = report["combined"]
    assert set(combined) == set(COMBINED_PIN)
    assert combined == COMBINED_PIN
    assert combined["original_failures_now_passing"] == 312
    assert combined["original_failures"] == 323
    assert combined["by_pair_class"]["open_set_change"][
        "original_failures_now_passing"
    ] == 0
    assert combined["by_pair_class"]["open_set_unchanged"][
        "original_failures_now_passing"
    ] == 312
    assert combined["max_abs_sigma_check"] == (
        "R22 pair(23,24) pi_0 mean_pt_y"
    )

    acceptance = report["acceptance"]
    assert set(acceptance) == {"criterion", "threshold", "achieved",
                               "pass"}
    assert acceptance["threshold"] == 0.95
    assert acceptance["achieved"] == (
        combined["pass_rate_among_original_failures"]
    )
    assert acceptance["achieved"] >= acceptance["threshold"]
    assert acceptance["pass"] is True

    findings = report["residual_findings"]
    assert len(findings) == len(RESIDUAL_PAIRS_PIN) == (
        combined["residual_failures"]
    )
    for finding, (run_id, pair, species, observable, pair_class) in zip(
        findings, RESIDUAL_PAIRS_PIN
    ):
        assert set(finding) == RESIDUAL_FINDING_KEYS
        assert finding["run"] == run_id
        assert finding["pair"] == list(pair)
        assert finding["species"] == species
        assert finding["observable"] == observable
        assert finding["pair_class"] == pair_class
        assert finding["sigma"] > N_SIGMA
        assert finding["residual"] > 0.0
        assert finding["residual"] == pytest.approx(
            finding["sigma"] * 0.0 + finding["residual"], abs=0.0
        )
        assert isinstance(finding["diagnosis"], str)
        assert finding["diagnosis"]
        # every diagnosis names the residual and the sigma
        assert "residual" in finding["diagnosis"]
        assert "sigma" in finding["diagnosis"]
        if pair_class == "open_set_change":
            assert "genuine open-set-change residual" in (
                finding["diagnosis"]
            )
    # the findings are consistent with the re-evaluation residuals
    for key, payload in payloads.items():
        key_findings = [
            finding for finding in findings if finding["run"] == key.upper()
        ]
        payload_residuals = payload["summary"]["residuals"]
        assert len(key_findings) == len(payload_residuals)
        for finding, residual in zip(key_findings, payload_residuals):
            assert finding["pair"] == residual["pair"]
            assert finding["species"] == residual["species"]
            assert finding["observable"] == residual["observable"]
            assert finding["pair_class"] == residual["pair_class"]
            assert finding["sigma"] == residual["sigma"]
            assert finding["residual"] == residual["residual"]
            assert finding["diagnosis"] == residual["diagnosis"]

    assert "clause_status" in report
    assert report["no_fitting"] is True

    assert _file_sha256(run_dir / "gate2_revised_threshold.json") == (
        REVISED_THRESHOLD_SHA256
    )
    assert _canonical_sha256(report) == REVISED_THRESHOLD_SHA256


def test_metric_engine_crosscheck_structure(run_dir, r21_archive, r22_archive):
    """Each cross-check payload compares the committed
    ``full_analytic_difference`` engine against the sealed per-check
    values on 572 / 816 checks, records the maximum absolute
    deviation per observable with the where-block (pair, species,
    observable, engine and sealed values), and the recorded bounds
    are self-consistent (the max equals the where-block deviation).
    The cross-check is engine-validation evidence, recorded as a
    diagnostic, never an assertion."""
    pinned_digests = {
        "r21": CROSSCHECK_R21_SHA256,
        "r22": CROSSCHECK_R22_SHA256,
    }
    for key, (gate_sha256, archive, pinned) in (
        ("r21", (R21_GATE_SHA256, r21_archive, CROSSCHECK_PIN["r21"])),
        ("r22", (R22_GATE_SHA256, r22_archive, CROSSCHECK_PIN["r22"])),
    ):
        payload = _load(f"metric_engine_crosscheck_{key}.json")
        assert payload["schema"] == CROSSCHECK_SCHEMA
        assert payload["run"] == key.upper()
        assert payload["card"] == CARD_ID

        assert payload["archive"] == {
            "path": f"dev/00runs/{(R21_RUN_NAME if key == 'r21' else R22_RUN_NAME)}",
            "gate_file": f"gate2_{key}.json",
            "gate_sha256": gate_sha256,
        }
        assert _file_sha256(archive / f"gate2_{key}.json") == gate_sha256

        engine = payload["engine"]
        assert set(engine) == {
            "module", "function", "record_set", "quadrature", "purpose",
        }
        assert engine["module"] == "na49_mixing.threshold_metric"
        assert engine["function"] == "full_analytic_difference"
        assert engine["record_set"] == (
            "na49_mixing.threshold_metric.RecordSet"
        )
        assert "no Monte Carlo" in engine["quadrature"]
        assert "never an assertion" in engine["purpose"]
        assert isinstance(payload["reference"], str)
        assert payload["reference"]

        assert payload["n_checks_applied"] == pinned["n_checks_applied"]
        assert payload["n_compared"] == payload["n_checks_applied"]

        deviations = payload["max_abs_deviation"]
        assert set(deviations) == set(OBSERVABLES)
        for observable in OBSERVABLES:
            block = deviations[observable]
            assert set(block) == {"max", "where"}
            assert block["max"] == pinned["max_abs_deviation"][
                observable
            ]["max"]
            where = block["where"]
            assert set(where) == {
                "pair", "species", "observable", "engine", "sealed",
            }
            assert where == pinned["max_abs_deviation"][observable][
                "where"
            ]
            assert where["observable"] == observable
            # self-consistent bound: the recorded max is the
            # where-block deviation
            assert block["max"] == pytest.approx(
                abs(where["engine"] - where["sealed"]), rel=1e-12
            )

        assert _file_sha256(
            run_dir / f"metric_engine_crosscheck_{key}.json"
        ) == pinned_digests[key]
        assert _canonical_sha256(payload) == pinned_digests[key]


def test_seal_manifest_and_sha256_sums(run_dir):
    """``seal_manifest.json`` records the 7 sealed files (the
    pre-registration, the provenance, the two re-evaluation payloads,
    the revised clause report, the two cross-check payloads; the
    manifest and the sums file exclude themselves) totaling
    1,407,181 bytes and the digest of ``SHA256SUMS.txt``, which is in
    ``sha256sum -c`` format with run-dir-relative paths, carries the
    pinned payload digests, verifies against every sealed file on
    disk, and self-consistently accounts for the full byte total."""
    manifest = _load("seal_manifest.json")
    assert manifest["schema"] == SEAL_SCHEMA
    assert manifest["run"] == RUN_ID
    assert manifest["card"] == CARD_ID
    assert UTC_STAMP.match(manifest["created_utc"])
    assert manifest["file_count"] == N_SEALED_FILES
    assert manifest["total_bytes"] == SEAL_TOTAL_BYTES
    assert manifest["sha256_sums_file"] == "SHA256SUMS.txt"
    assert manifest["sha256_sums_sha256"] == SUMS_FILE_SHA256
    assert manifest["excluded_from_sums"] == [
        "seal_manifest.json", "SHA256SUMS.txt",
    ]
    assert manifest["note"]

    assert _file_sha256(run_dir / "seal_manifest.json") == (
        SEAL_MANIFEST_SHA256
    )
    sums_path = run_dir / "SHA256SUMS.txt"
    assert _file_sha256(sums_path) == SUMS_FILE_SHA256, (
        "the sums file drifted from the digest recorded in the seal "
        "manifest; the run dir was modified after the seal"
    )

    text = sums_path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert "\r" not in text
    lines = text.splitlines()
    assert len(lines) == N_SEALED_FILES
    for line in lines:
        assert SUMS_LINE.match(line), (
            f"SHA256SUMS.txt is not in sha256sum -c format: {line!r}"
        )

    entries = [line.split("  ", 1) for line in lines]
    assert {rel for _digest, rel in entries} == set(
        SUMS_TOP_LEVEL_PAYLOADS
    ), "the sums file covers a drifted file set"
    for digest, rel in entries:
        assert digest == SUMS_PAYLOAD_SHA256[rel], (
            f"{rel}: the sums-carrying digest drifted from the pinned "
            "digest"
        )
        assert _file_sha256(run_dir / rel) == digest, (
            f"the sealed digest of {rel} does not match the file on "
            "disk; the run dir was modified after the seal"
        )
    # self-consistent byte total: the sums-covered files account for
    # the seal manifest's total_bytes
    assert sum((run_dir / rel).stat().st_size for _digest, rel in entries) == (
        SEAL_TOTAL_BYTES
    )


def test_clean_environment_record(run_dir):
    """The post-seal clean-environment record re-verified all 7
    sums-covered files without mismatches, and the clean-interpreter
    re-run reproduced the three sealed report payloads byte-for-byte
    (the recorded sealed and recomputed digests match, equal the
    pinned digests, and equal the digests of the files on disk)."""
    record = _load("clean_environment_commands.json")
    assert record["schema"] == CLEAN_SCHEMA
    assert record["run"] == RUN_ID
    assert record["card"] == CARD_ID
    assert UTC_STAMP.match(record["verified_utc"])
    assert record["seal_verification"] == {
        "files_checked": N_SEALED_FILES,
        "mismatches": [],
    }
    assert record["hashes_match"] is True

    for key in ("python_version", "python_implementation", "platform",
                "machine", "numpy_version", "executable"):
        assert key in record["environment"]
        assert key in record["probe_environment"]

    commands = record["commands"]
    assert len(commands) == 1
    command = commands[0]
    assert command["cwd"] == str(MONO_ROOT)
    argv = command["argv"]
    assert "rereval_threshold_checks.py" in " ".join(argv)
    assert RUN_NAME in " ".join(argv)
    assert "rereval_report" in " ".join(argv)
    assert "build_revised_threshold" in " ".join(argv)

    assert record["sealed_report_sha256"] == CLEAN_SEALED_REPORT_SHA256
    assert record["recomputed_report_sha256"] == CLEAN_SEALED_REPORT_SHA256
    for name, digest in CLEAN_SEALED_REPORT_SHA256.items():
        assert _file_sha256(run_dir / name) == digest, (
            f"{name}: the on-disk sealed report drifted from the "
            "clean-environment record"
        )

    assert _file_sha256(
        run_dir / "clean_environment_commands.json"
    ) == CLEAN_ENVIRONMENT_SHA256


def test_sealed_gate_jsons_byte_unchanged_and_original_verdicts(
    run_dir, r21_archive, r22_archive
):
    """The sealed ``gate2_r21.json`` / ``gate2_r22.json`` match their
    pinned SHA-256 digests (the digests recorded in the sealed
    archives' own ``SHA256SUMS.txt``) and carry the original metric's
    verdict (19 / 27 adjacent pairs, 351 / 483 applied checks, 169 /
    154 failed, the original maximum sigmas, the sealed form
    marked); the archive seal manifests agree with the on-disk sums
    files."""
    for key, archive in (("r21", r21_archive), ("r22", r22_archive)):
        gate_file = f"gate2_{key}.json"
        rel = f"dev/00runs/{(R21_RUN_NAME if key == 'r21' else R22_RUN_NAME)}/{gate_file}"
        pinned_digest = {
            "r21": R21_GATE_SHA256, "r22": R22_GATE_SHA256,
        }[key]
        # the sealed archive's own sums file records the gate digest
        # (monorepo-root-relative paths, sha256sum -c format)
        sums_text = (archive / "SHA256SUMS.txt").read_text(encoding="utf-8")
        entries = [
            line.split("  ", 1)
            for line in sums_text.splitlines()
        ]
        gate_entries = [entry for entry in entries if entry[1] == rel]
        assert len(gate_entries) == 1
        assert gate_entries[0][0] == pinned_digest, (
            f"{gate_file}: the sealed archive's own SHA256SUMS entry "
            "drifted from the pinned digest"
        )
        # the gate JSON is byte-unchanged on disk
        assert _file_sha256(archive / gate_file) == pinned_digest, (
            f"{gate_file}: the sealed gate JSON was modified after the "
            "seal; the original metric's verdict must stay untouched"
        )
        # the archive seal manifest agrees with the on-disk sums file
        seal = json.loads(
            (archive / "seal_manifest.json").read_text(encoding="utf-8")
        )
        files_checked = {
            "r21": R21_ARCHIVE_SEAL_FILES, "r22": R22_ARCHIVE_SEAL_FILES,
        }[key]
        sums_digest = {
            "r21": R21_ARCHIVE_SUMS_SHA256,
            "r22": R22_ARCHIVE_SUMS_SHA256,
        }[key]
        assert seal["file_count"] == files_checked == len(entries)
        assert seal["sha256_sums_sha256"] == sums_digest
        assert _file_sha256(archive / "SHA256SUMS.txt") == sums_digest

        # the original metric's verdict, as recorded by the sealed
        # gate JSON (169 / 351 R21, 154 / 483 R22)
        gate = json.loads(
            (archive / gate_file).read_text(encoding="utf-8")
        )
        assert gate["run"] == key.upper()
        assert gate["schema"] == {
            "r21": R21_GATE_SCHEMA, "r22": R22_GATE_SCHEMA,
        }[key]
        assert gate["sealed"] is True
        clause = gate["clauses"]["threshold_discontinuity"]
        pin = ORIGINAL_GATE_PIN[key]
        assert clause["adjacent_pairs_checked"] == pin[
            "adjacent_pairs_checked"
        ]
        assert clause["n_checks_applied"] == pin["n_checks_applied"]
        assert clause["n_checks_failed"] == pin["n_checks_failed"]
        assert clause["n_failed_at_open_set_changes"] == pin[
            "n_failed_at_open_set_changes"
        ]
        assert clause["n_failed_at_unchanged_pairs"] == pin[
            "n_failed_at_unchanged_pairs"
        ]
        assert clause["n_checks_vacuous_zero"] == pin[
            "n_checks_vacuous_zero"
        ]
        assert clause["n_checks_absent_species"] == pin[
            "n_checks_absent_species"
        ]
        assert clause["n_open_set_change_applied"] == pin[
            "n_open_set_change_applied"
        ]
        assert clause["max_abs_sigma"] == pin["max_abs_sigma"]
        assert clause["max_abs_sigma_check"] == pin["max_abs_sigma_check"]
        assert clause["pass"] is False
        # the re-evaluation's original block reproduces the sealed
        # verdict (the 169 / 154 original failures)
        payload = _load(f"rereval_{key}.json")
        assert payload["summary"]["original"]["n_checks_failed"] == (
            pin["n_checks_failed"]
        )
        assert payload["summary"]["n_checks_applied"] == pin[
            "n_checks_applied"
        ]
        assert payload["archive"]["gate_sha256"] == pinned_digest
