"""Round 3 card 8c: run R3-SE-AUDIT sealed-run artifact pins.

Task ``TASK-20260918-r3-se-audit-confirmation-mc`` (round 3, card 8c).  This
module pins the sealed run evidence under
``dev/00runs/r3_se_audit_2026-09-18/``: the pre-registration (the Phase A
hermetic SE-estimator and seed-map audit over the sealed R31/R21/R22
artifacts, the Phase B T1-scaled confirmation MC contract of 15 nodes x
50 replicas x 20,000 decays at the new master seed 20260918 in 750
gateway chunks, and the card-2 re-registered metric ``delta_analytic_full``
at the unchanged 3.0-sigma tolerance), the provenance, the Phase A
diagnostics (the node-1 tight-SE verdict, the ten named residuals with
their seed-issue / estimator-artifact classifications, the seed-map
audit, the replica-correlation audit, and the bit-identical replays),
the confirmation verification (422/422 checks passing at the
50-replica statistics, the 14 original R31 failures all resolved, and
the routing fractions), the aggregation state, the pre-investigation
inventory (the 78 sealed R31 files re-verified byte for byte), the seal
manifest (1537 sealed files totaling 16,990,201 bytes),
``SHA256SUMS.txt`` (1535 monorepo-root-relative lines), the post-seal
clean-environment record (1535 files re-verified, three clean-interpreter
re-runs), and the gateway submission/fetch records (750 jobs, 1,500
superseded jobs, three retries).

Sibling of ``tests/test_staged_r30_spectral_normalization_pins.py`` and
``tests/test_staged_r3_threshold_metric_reregistration_pins.py`` (the
same pin style: sealed-dir discovery, skip-or-fail when the gitignored
archive is absent, digest pins plus structural pins).  These pins are
read-only over the sealed artifacts: nothing is recomputed, the sealed
run is never re-sealed, and the post-seal clean-environment record is
never re-executed.  The only documented post-seal state change is the
clean-environment record itself: the seal manifest carries the record's
pre-seal digest (130,010 bytes) while the final on-disk record is the
post-seal verification (130,008 bytes), and the record is excluded from
``SHA256SUMS.txt`` for that reason.

Pins:

- ``test_sealed_run_dir_inventory``: the run dir holds exactly the nine
  top-level files (the six sealed payloads, the seal manifest,
  ``SHA256SUMS.txt``, the post-seal clean-environment record) plus the
  ``phase_b/`` and ``phase_b_smoke/`` trees.
- ``test_seal_manifest``: ``seal_manifest.json`` records 1,537 sealed
  files totaling 16,990,201 bytes, every entry verifies against disk
  except the documented pre-seal clean-environment entry, and the sums
  file digest is pinned.
- ``test_sha256_sums``: ``SHA256SUMS.txt`` carries 1,535
  monorepo-root-relative lines covering the top level, ``phase_b``,
  and ``phase_b_smoke``, lexicographically sorted by relative path, and
  every digest verifies against the file on disk from the monorepo root.
- ``test_pre_registration_structure``: the sealed pre-registration
  carries the committed contract (the draws, the unchanged 3.0-sigma
  tolerance with the engine-floor note, the ten input digests, the open
  sets of all three windows, the ten named residuals, the Phase A
  classification cutoffs and estimator definitions, and the Phase B
  windows/nodes/statistics/seed map/decay routing/chunk plan/metric).
- ``test_provenance_structure``: the provenance records the dual-tree
  driver digests (the pre-registration digest 9ccf36c5 and the
  post-seal digest 2c72fa7e, the frozen R31 driver), the fixed card-2
  engine, the git identity, the environment fingerprint, the seeds,
  the catalog and continuum-config digests, and the input hashes.
- ``test_phase_a_diagnostics_structure``: the Phase A report carries
  the node-1 tight-SE verdict (ratio 9.691627288965723,
  ``estimator_artifact``), the ten named residuals (nine
  ``seed_issue_flag``, one ``estimator_artifact``), the R21/R22 named
  residual classifications, the seed-map audit (the confirmation map
  disjoint from all others with zero internal duplicates), the
  replica-correlation audit (17 flagged pairs), the bit-identical
  replays, and the estimator cells.
- ``test_confirmation_verification_structure``: the re-evaluation
  carries the 422/422 verdict (14 old failures, 0 new failures, max
  absolute sigma 2.959754302712887), the 14 original failures all
  passing at the new statistics, the ten named residuals all
  ``genuine_finite_mc_scatter``, the routing fractions of all 15 nodes,
  and the 10 delta drift rows of the card-8a P1 engine fix.
- ``test_phase_b_aggregation_state``: the aggregation state records
  750/750 completed chunks (each 20,000 decays) over the full
  (window, node, replica) coverage with all 15 nodes merged.
- ``test_pre_investigation_inventory``: the inventory re-verified all
  78 sealed R31 files byte for byte (the recorded R31 pre-registration
  digest is cross-checked against the R31 run dir) and records the seal
  states of the R31/R21/R22/card-2 archives.
- ``test_clean_environment_record``: the post-seal record re-verified
  the 1535 sums-covered files without mismatches, re-ran
  ``--phase-a`` / ``--aggregate`` / ``--reeval`` in the clean
  interpreter (``python -I``, return code 0), and carries the 768
  sealed-report digests (a pinned subset cross-checked against disk).
- ``test_gateway_submission_and_fetch``: the submission manifest
  carries 750 jobs (chunk names, job ids, the nine gateway workers),
  1,500 superseded jobs, the three retry counts, and the fetch log
  records all 750 chunks ``COMPLETED``.

Hermetic: reads the sealed run dir (gitignored; the artifact is
present on disk) and the committed config, catalog, package, and
drivers; no computation runs.  If the run dir is absent (a fresh
checkout), the artifact pins skip with a clear message.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

NESTED_ROOT = Path(__file__).resolve().parent.parent
MONO_ROOT = NESTED_ROOT.parent.parent
RUN_NAME = "r3_se_audit_2026-09-18"
RUN_DIR = MONO_ROOT / "dev" / "00runs" / RUN_NAME
SUMS_PREFIX = f"dev/00runs/{RUN_NAME}/"

CONFIG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json"
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
DRIVER_PATH = NESTED_ROOT / "helpers" / "benchmarking" / "staged_r3_se_audit.py"
DRIVER_OUTER_PATH = (
    MONO_ROOT / "helpers" / "benchmarking" / "staged_r3_se_audit.py"
)
R31_DRIVER_PATH = NESTED_ROOT / "helpers" / "benchmarking" / "staged_r31_matching_scan.py"
R31_DRIVER_OUTER_PATH = (
    MONO_ROOT / "helpers" / "benchmarking" / "staged_r31_matching_scan.py"
)
ENGINE_PATH = NESTED_ROOT / "na49_mixing" / "threshold_metric.py"
SPECTRAL_DENSITY_MODULE_PATH = (
    NESTED_ROOT / "na49_mixing" / "spectral_density.py"
)
RESONANCE_DECAY_MODULE_PATH = (
    NESTED_ROOT / "na49_mixing" / "resonance_decay.py"
)
RESONANCE_WIDTHS_MODULE_PATH = (
    NESTED_ROOT / "na49_mixing" / "resonance_widths.py"
)
RNG_SUBSTREAMS_MODULE_PATH = NESTED_ROOT / "na49_mixing" / "rng_substreams.py"
R31_RUN_DIR = MONO_ROOT / "dev" / "00runs" / (
    "r31_catalog_continuum_matching_scan_2026-09-17"
)

# ---------------------------------------------------------------------------
# The committed run contract (the sealed run realizes it).
# ---------------------------------------------------------------------------
RUN_ID = "R3-SE-AUDIT"
CARD_ID = "TASK-20260918-r3-se-audit-confirmation-mc"
MASTER_SEED = 20260918
N_REPLICAS = 50
DECAYS_PER_REPLICA = 20_000
N_WINDOWS = 3
N_NODES = 5
N_CHUNKS = 750
CHUNK_SIZE_DECAYS = 20_000
TOTAL_DECAYS = 15_000_000
N_CHECKS = 422
N_OLD_FAILURES = 14
TOLERANCE_SIGMA = 3.0
BOUNDARY_GEV = 2.680194110727528
NODE_MASSES_GEV = [2.48, 2.58, 2.68, 2.78, 2.88]

#: The card-6 decay-routing expected continuum weight per node (the
#: ramp is active at nodes 3 and 4).
EXPECTED_W_AT_NODES = [
    0.0, 0.0, 0.0, 0.19961177854494405, 0.39961177854494423,
]

#: The pre-registered transition windows (window, (B, Q, S), label,
#: family expectation).
WINDOWS = [
    (0, [1, 1, 0], "baryonic (1, 1, 0)", "baryonic_np"),
    (1, [1, 0, -1], "strange baryonic (1, 0, -1)", "baryonic_strange"),
    (2, [0, 0, 0], "neutral mesonic (0, 0, 0)", "mesonic_neutral"),
]

#: The per-window open sets (identical on all five nodes of a window).
OPEN_SETS = {
    "0": [2212, 2214, 12212, 2216, 22212, 102212, 122212, 212214, 202214, 122214],
    "1": [3122, 3212, 102134, 3214],
    "2": [
        111, 310, 130, 113, 100113, 30113, 223, 100223, 30223, 221,
        377, 10113, 20113, 115, 225, 117, 229, 333,
    ],
}

PRE_SCHEMA = "na49_mixing.staged_r3_se_audit.pre_registration/1"
PROV_SCHEMA = "na49_mixing.staged_r3_se_audit.provenance/1"
PHASE_A_SCHEMA = "na49_mixing.staged_r3_se_audit.phase_a/1"
REEVAL_SCHEMA = "na49_mixing.staged_r3_se_audit.reeval/1"
AGGREGATION_SCHEMA = "na49_mixing.staged_r3_se_audit.aggregation_state/1"
INVENTORY_SCHEMA = "na49_mixing.staged_r3_se_audit.pre_investigation/1"
SEAL_SCHEMA = "na49_mixing.staged_r3_se_audit.seal_manifest/1"
CLEAN_SCHEMA = "na49_mixing.staged_r3_se_audit.clean_environment/1"
SUBMISSION_SCHEMA = "na49_mixing.staged_r3_se_audit.submission_manifest/1"
FETCH_SCHEMA = "na49_mixing.staged_r3_se_audit.fetch_log/1"

# ---------------------------------------------------------------------------
# The pinned input digests (the sealed run's own inputs, byte-stable).
# ---------------------------------------------------------------------------
CONFIG_SHA256 = (
    "242f906cf040eadbe3796dfcfb61d0d51535277399858c901a162e6653d635e1"
)
CATALOG_SHA256 = (
    "6e4de7372a5bbf99affd6fa3ccc295c4cef09404dab3b17acd8816cda762bdcd"
)
SPECTRAL_DENSITY_MODULE_SHA256 = (
    "561c65ee9a77a1b7f7eb07a7619dafb584779a51ccb23d7fe5651c0605e12563"
)
RESONANCE_DECAY_MODULE_SHA256 = (
    "0b675ffef0b17ec0a6d8e93f593a15adda03dc8dca40e5a789f5f3285d384bc6"
)
RESONANCE_WIDTHS_MODULE_SHA256 = (
    "1f87b8108cf4e7b0fa8a57b5b4e2747f8665eb549c77d34d624e54b082e53e1e"
)
RNG_SUBSTREAMS_MODULE_SHA256 = (
    "ed8f05f151b338a8ac7a4c395637bf71e68fce5b4f4b1ff09721d119c8a5bfb0"
)
ENGINE_SHA256 = (
    "ee240f44645d2e592acfae462f7d1e9c3b3dd8cd78c96bd40b7b73509f7fcc63"
)
R31_DRIVER_SHA256 = (
    "b47b2a858c3880ecf69e143035e8c209b6e6e5e0708ca2b9058c6c384340afe6"
)
#: The card-8b resume (TASK-20260918-r3-run-r31-verify-resume, under the
#: acceptance rescope TASK-20260918-r3-acceptance-rescope-card8a-8b)
#: regenerated the R31 matching verification with the fixed card-2
#: engine and updated the R31 provenance (the fixed engine sha256 entry
#: plus the resume block with the 76-artifact bit-identity record); the
#: two resumed digests below are pinned by
#: tests/test_staged_r31_matching_scan_gate_pins.py, and the pre-fix
#: stop-state digests recorded in the pre-investigation inventory stay
#: the sealed stop-state record.  The inventory's other 76 entries are
#: the resume's bit-identity set.
R31_RESUMED_VERIFICATION_SHA256 = (
    "9170426043399dc954c08ec255c7a7592f63ffbdfdcc3ab655cc1c0dd8e499af"
)
R31_RESUMED_PROVENANCE_SHA256 = (
    "95a03075facd687b0f74daf88bb18c0ef26c29ae9674d34df502c3b6f46a99b2"
)
CARD9_DRIVER_SHA256 = (
    "92733b66a861296d7a946d22ef4a6f3169c20f921671797bb4b0034704e1a0df"
)

#: The driver digest recorded at pre-registration (before the dual-tree
#: sync commit) and the post-seal committed digest.
DRIVER_SHA256_AT_PREREGISTRATION = (
    "9ccf36c51c0489bceba50955595de5fd9bd33ad13799b95c2ca7251275ffb466"
)
DRIVER_SHA256_CURRENT = (
    "2c72fa7e7b5a86a54076f6cc686bb33d165d2b280092df7ca6fd75d3b5635081"
)

# ---------------------------------------------------------------------------
# The pinned seal state.
# ---------------------------------------------------------------------------
N_SEAL_MANIFEST_ENTRIES = 1537
SEAL_TOTAL_BYTES = 16_990_201
N_SUMS_LINES = 1535
#: The on-disk byte total of the 1537 sealed files (the clean record's
#: final size is 2 bytes below its pre-seal size).
DISK_TOTAL_BYTES = 16_990_199
SUMS_FILE_SHA256 = (
    "c49dbb6a4dc150928a3295015008c0bd035c89acfc2953d8d75760b1d5b4cfa3"
)
EXCLUDED_FROM_SUMS = [
    "seal_manifest.json", "SHA256SUMS.txt", "clean_environment_commands.json",
]
#: The seal manifest's pre-seal entry for the clean-environment record
#: (the record was finalized by the post-seal verification).
CLEAN_SEALED_STALE_SHA256 = (
    "6145ef230e431b270da3b8c67d88a3fa94dd2170d959f0cd8d85e7e20037f7e6"
)
CLEAN_SEALED_STALE_BYTES = 130_010
CLEAN_ONDISK_SHA256 = (
    "1c98a9f7df3524a70f1fbb8328c3627d9c51ee37f83766df7d96a192ed8c359f"
)
CLEAN_ONDISK_BYTES = 130_008
SEAL_CREATED_UTC = "2026-09-19T10:51:22Z"
CLEAN_VERIFIED_UTC = "2026-09-19T11:29:18Z"

#: The 768-entry sealed-report digest map of the clean-environment
#: record; a pinned subset is cross-checked against disk.
N_SEALED_REPORT_HASHES = 768
SEALED_REPORT_HASH_PIN = {
    "confirmation_verification.json": (
        "98a3bff11184f93d16d17dbebee65dc2fd6834fac505aad7670ee7063ebd204d"
    ),
    "phase_a_diagnostics.json": (
        "993b1d29538e11869cad4e44f2cc0a0ab4a1838b1af5b95263174b6be26b5f8e"
    ),
    "phase_b_aggregation_state.json": (
        "8da5885897c3aa435a56024d1d643e3df33303d1b575329c824224cb1c199c4b"
    ),
}

TOP_LEVEL_FILES = (
    "SHA256SUMS.txt",
    "clean_environment_commands.json",
    "confirmation_verification.json",
    "phase_a_diagnostics.json",
    "phase_b_aggregation_state.json",
    "pre_investigation_inventory.json",
    "pre_registration.json",
    "provenance.json",
    "seal_manifest.json",
)
SUMS_TOP_LEVEL_FILES = (
    "confirmation_verification.json",
    "phase_a_diagnostics.json",
    "phase_b_aggregation_state.json",
    "pre_investigation_inventory.json",
    "pre_registration.json",
    "provenance.json",
)
N_PHASE_B_FILES = 1517
N_PHASE_B_SMOKE_FILES = 12

#: ``sha256sum -c`` line: 64 hex digits, two spaces, the
#: monorepo-root-relative path.
SUMS_LINE = re.compile(r"^[0-9a-f]{64}  \S+$")
UTC_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
JOB_ID = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{8}$")

#: The gateway worker fleet (the nine pinned hosts).
GATEWAY_WORKERS = {
    "c0801", "c0802", "c0803", "c0804", "c0805",
    "c0806", "c0807", "c0808", "c0809",
}

# ---------------------------------------------------------------------------
# The pinned Phase A verdicts.
# ---------------------------------------------------------------------------

#: The node-1 tight-SE target (window 2 node 1, 2.58 GeV, pi_0 mean_p):
#: per-node (E1, E4 b=32, ratio).
NODE1_TIGHT_SE_NODES = {
    "0": (0.001569919346256475, 0.001373011265241678, 0.87457439677756),
    "1": (0.00014970995013351702, 0.001450933038143691, 9.691627288965723),
    "2": (0.0006015599036712832, 0.00138845351317031, 2.3080885290004596),
    "3": (0.0027833759816500467, 0.0016527043654136063, 0.5937769012556638),
    "4": (0.0008727758730759142, 0.001745385626310474, 1.9998096649478074),
}
NODE1_SIBLING_RATIOS = [
    0.87457439677756, 2.3080885290004596, 0.5937769012556638, 1.9998096649478074,
]
NODE1_RATIO_TARGET = 9.691627288965723

#: The ten pre-registered named residuals (index, window, pair,
#: species, observable, pre-fix sigma on the card, Phase A
#: classification, honest E4 b=32 sigma).
NAMED_RESIDUALS_PHASE_A = [
    (1, 0, [0, 1], "proton", "mean_pt_y", 3.21, "seed_issue_flag", 3.1864926679232832),
    (2, 0, [0, 1], "pi_minus", "mean_pt_y", 3.33, "seed_issue_flag", 1.8071125157708587),
    (3, 0, [3, 4], "neutron", "mean_p", 3.15, "seed_issue_flag", 2.6895441772719093),
    (4, 2, [0, 1], "pi_0", "mean_pt_y", -3.03, "seed_issue_flag", -2.0348763223673374),
    (5, 2, [0, 1], "K0L", "mean_p", 3.06, "seed_issue_flag", 1.8090802179921976),
    (6, 2, [0, 1], "K0L", "mean_pt_y", 3.07, "seed_issue_flag", 1.9045269025323264),
    (7, 2, [1, 2], "pi_0", "mean_p", 7.01, "estimator_artifact", 2.164706337762149),
    (8, 2, [3, 4], "K0S", "mean_pt_y", 3.4, "seed_issue_flag", 1.7466364773436784),
    (9, 2, [3, 4], "K0L", "mean_p", 3.74, "seed_issue_flag", 1.9043264302163536),
    (10, 2, [3, 4], "K0L", "mean_pt_y", 4.55, "seed_issue_flag", 1.9489409877412907),
]

#: The 11 R21/R22 named residuals (run, pair, species, observable,
#: pair class, sealed re-evaluation sigma, Phase A classification).
NAMED_RESIDUALS_R21R22 = [
    ("R21", [0, 1], "proton", "mean_p", "open_set_change", 3.598708027565074, "mixed"),
    ("R21", [0, 1], "proton", "mean_pt_y", "open_set_change", 3.4457902279287818, "mixed"),
    ("R21", [1, 2], "pi_0", "mean_pt_y", "open_set_unchanged", 3.3622721646606926, "genuine_low_variance"),
    ("R21", [3, 4], "pi_0", "mean_pt_y", "open_set_unchanged", 3.594992035740298, "mixed"),
    ("R21", [4, 5], "pi_minus", "mean_pt_y", "open_set_change", 3.0555852180458674, "genuine_low_variance"),
    ("R21", [5, 6], "pi_0", "mean_pt_y", "open_set_change", 3.314280796269175, "mixed"),
    ("R21", [15, 16], "pi_0", "mean_p", "open_set_unchanged", 3.0846781101915357, "mixed"),
    ("R21", [18, 19], "pi_0", "multiplicity", "open_set_unchanged", 3.014046219969753, "genuine_low_variance"),
    ("R22", [7, 8], "pi_minus", "mean_pt_y", "open_set_change", 3.8051287494598878, "mixed"),
    ("R22", [15, 16], "pi_plus", "mean_pt_y", "open_set_unchanged", 3.018186264093622, "genuine_low_variance"),
    ("R22", [23, 24], "pi_0", "mean_pt_y", "open_set_unchanged", 4.984844257802041, "mixed"),
]

#: The per-cell replica-correlation audit (components, components
#: used, flagged pairs, flag, max |r|).
REPLICA_CORRELATION_CELLS = {
    "w0_j0": (21, 21, 2, True, 0.9270486875178369),
    "w0_j1": (21, 21, 1, True, 0.8447224407330114),
    "w0_j2": (21, 21, 2, True, 0.9198406879045561),
    "w0_j3": (30, 29, 2, True, 0.7052701821500643),
    "w0_j4": (30, 29, 0, False, 0.6040897699553563),
    "w1_j0": (33, 33, 1, True, 0.643042142518792),
    "w1_j1": (33, 33, 1, True, 0.6923898236050117),
    "w1_j2": (33, 33, 0, False, 0.649832055946629),
    "w1_j3": (33, 33, 0, False, 0.5759613899551853),
    "w1_j4": (33, 33, 0, False, 0.714579407813936),
    "w2_j0": (21, 21, 0, False, 0.81630858936322),
    "w2_j1": (21, 21, 2, True, 0.872662849989045),
    "w2_j2": (21, 21, 2, True, 0.8876871427041446),
    "w2_j3": (21, 21, 3, True, 0.8687861856929432),
    "w2_j4": (21, 21, 1, True, 0.912937010061032),
}
REPLICA_CORRELATION_FLAGGED_PAIRS = 17

# ---------------------------------------------------------------------------
# The pinned confirmation verdicts.
# ---------------------------------------------------------------------------
MAX_ABS_SIGMA_NEW = 2.959754302712887
METRIC_STATEMENT = (
    "delta_analytic_full at 3.0 sigma with SE_combined = "
    "sqrt(SE_j^2 + SE_{j+1}^2) at the confirmation statistics; "
    "per-check sigma at the old SE versus the new SE"
)

#: The 14 original R31 failures (window, pair, species, observable,
#: sealed old-se sigma, new-se sigma; all pass at the new statistics).
ORIGINAL_14_FAILURES = [
    (0, [0, 1], "pi_minus", "mean_pt_y", 3.3254504430919427, 1.7448633070905266, True),
    (0, [0, 1], "proton", "mean_pt_y", 3.2123466532077045, 0.24924759788716583, True),
    (0, [3, 4], "K_0", "mean_p", 25.98076211353316, 0.0, True),
    (0, [3, 4], "Sigma_plus", "mean_p", 25.98076211353316, 0.0, True),
    (0, [3, 4], "neutron", "mean_p", 3.151740704856282, -0.7953962610929308, True),
    (1, [2, 3], "pi_plus", "mean_pt_y", 7.225811435009375, 0.5552285837350331, False),
    (1, [3, 4], "pi_plus", "mean_pt_y", -5.610935929060075, -0.5240345496564799, False),
    (2, [0, 1], "K0L", "mean_p", 3.064256426739259, 0.5083062555597769, True),
    (2, [0, 1], "K0L", "mean_pt_y", 3.069041367744035, 0.6011345842962964, True),
    (2, [0, 1], "pi_0", "mean_pt_y", -3.0280248998333104, -0.9554659365490366, True),
    (2, [1, 2], "pi_0", "mean_p", 7.01270490790188, 0.815841577316934, True),
    (2, [3, 4], "K0L", "mean_p", 3.7365731963210447, -0.7620440622202165, True),
    (2, [3, 4], "K0L", "mean_pt_y", 4.547781556513862, -1.4888713594117697, True),
    (2, [3, 4], "K0S", "mean_pt_y", 3.4011873510462327, -0.24295175245811504, True),
]

#: The 10 named residuals at the confirmation statistics (all
#: ``genuine_finite_mc_scatter``): index, window, pair, species,
#: observable, sealed old-se sigma, new-se sigma.
NAMED_RESIDUALS_PHASE_B = [
    (1, 0, [0, 1], "proton", "mean_pt_y", 3.2123466532077045, 0.24924759788716583),
    (2, 0, [0, 1], "pi_minus", "mean_pt_y", 3.3254504430919427, 1.7448633070905266),
    (3, 0, [3, 4], "neutron", "mean_p", 3.151740704856282, -0.7953962610929308),
    (4, 2, [0, 1], "pi_0", "mean_pt_y", -3.0280248998333104, -0.9554659365490366),
    (5, 2, [0, 1], "K0L", "mean_p", 3.064256426739259, 0.5083062555597769),
    (6, 2, [0, 1], "K0L", "mean_pt_y", 3.069041367744035, 0.6011345842962964),
    (7, 2, [1, 2], "pi_0", "mean_p", 7.01270490790188, 0.815841577316934),
    (8, 2, [3, 4], "K0S", "mean_pt_y", 3.4011873510462327, -0.24295175245811504),
    (9, 2, [3, 4], "K0L", "mean_p", 3.7365731963210447, -0.7620440622202165),
    (10, 2, [3, 4], "K0L", "mean_pt_y", 4.547781556513862, -1.4888713594117697),
]

#: The card-8a P1 delta drift rows (all window 1): the 10 checks whose
#: ``delta_analytic_full`` moved when the three-body moment was fixed.
DELTA_DRIFT_ROWS = [
    (1, [0, 1], "Lambda", "mean_pt_y", -0.0015237986323975161, -0.001578019131541951),
    (1, [1, 2], "Lambda", "mean_pt_y", -0.0014408841430535932, -0.0014930945547485877),
    (1, [2, 3], "pi_plus", "mean_p", 0.3236426973580705, 0.31690375568578066),
    (1, [2, 3], "pi_plus", "mean_pt_y", 0.46075359018078454, 0.4942290824933768),
    (1, [2, 3], "Lambda", "mean_p", 0.08424629093351052, 0.08381516101842008),
    (1, [2, 3], "Lambda", "mean_pt_y", 0.05573349975785842, 0.054955350469866784),
    (1, [3, 4], "pi_plus", "mean_p", 0.25842033403489806, 0.2537623083099363),
    (1, [3, 4], "pi_plus", "mean_pt_y", 0.193265776839895, 0.15869379146207008),
    (1, [3, 4], "Lambda", "mean_p", 0.12500146652685643, 0.12441785051166057),
    (1, [3, 4], "Lambda", "mean_pt_y", 0.08626337981889529, 0.08530206747887153),
]

#: The per-node routing fractions (fraction continuum, expected w,
#: binomial sigma, n catalog, n continuum).
ROUTING_FRACTIONS = {
    "w0_j0": (0.0, 0.0, 0.0, 1_000_000, 0),
    "w0_j1": (0.0, 0.0, 0.0, 1_000_000, 0),
    "w0_j2": (0.0, 0.0, 0.0, 1_000_000, 0),
    "w0_j3": (0.199623, 0.19961177854494405, 0.02807409387558212, 800377, 199623),
    "w0_j4": (0.399035, 0.39961177854494423, -1.1775351367877205, 600965, 399035),
    "w1_j0": (0.0, 0.0, 0.0, 1_000_000, 0),
    "w1_j1": (0.0, 0.0, 0.0, 1_000_000, 0),
    "w1_j2": (0.0, 0.0, 0.0, 1_000_000, 0),
    "w1_j3": (0.199579, 0.19961177854494405, -0.08200611625466077, 800421, 199579),
    "w1_j4": (0.400073, 0.39961177854494423, 0.9416169757511204, 599927, 400073),
    "w2_j0": (0.0, 0.0, 0.0, 1_000_000, 0),
    "w2_j1": (0.0, 0.0, 0.0, 1_000_000, 0),
    "w2_j2": (0.0, 0.0, 0.0, 1_000_000, 0),
    "w2_j3": (0.199943, 0.19961177854494405, 0.8286574402775948, 800057, 199943),
    "w2_j4": (0.400038, 0.39961177854494423, 0.8701619430547118, 599962, 400038),
}

#: The inventory seal states of the read-only archives.
SEAL_STATES = {
    "r31": {"sealed": False, "note": "no SHA256SUMS.txt"},
    "r21": {"sealed": True, "sha256_sums_entries": 103, "mismatches": 0, "verify": "pass"},
    "r22": {"sealed": True, "sha256_sums_entries": 143, "mismatches": 0, "verify": "pass"},
    "card2_reregistration": {
        "sealed": True, "sha256_sums_entries": 7, "mismatches": 7, "verify": "FAIL",
    },
}

#: The R31 inventory's recorded driver sha (the sha256 of the sealed R31
#: pre_registration.json; the field name is historical).
R31_PRE_REGISTRATION_SHA256 = (
    "3f5ed7d8599aeaa53d38de897f731125fb5d50e0e9f8d546058ea3ae9bbc1a14"
)

#: The gateway submission pins.
CAPACITY_AT_SUBMIT_FREE_TASKS = 576
RETRY_COUNTS = {
    "r3se_w0_n1_r47": 1, "r3se_w0_n4_r16": 1, "r3se_w1_n0_r40": 1,
}
N_SUPERSEDED_JOBS = 1500


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(name: str) -> dict:
    return json.loads((RUN_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def run_dir():
    """The sealed R3-SE-AUDIT run dir, skipped with a clear message
    when the gitignored artifact is absent or incomplete (a fresh
    checkout)."""
    missing = [name for name in TOP_LEVEL_FILES if not (RUN_DIR / name).is_file()]
    if missing:
        pytest.skip(
            f"the sealed R3-SE-AUDIT run dir {RUN_DIR} is incomplete "
            "(missing: " + ", ".join(missing)
            + "); restore the sealed run to execute the artifact pins"
        )
    for tree in ("phase_b", "phase_b_smoke"):
        if not (RUN_DIR / tree).is_dir():
            pytest.skip(
                f"the sealed R3-SE-AUDIT run dir {RUN_DIR} is incomplete "
                f"(missing: {tree}/); restore the sealed run to execute "
                "the artifact pins"
            )
    return RUN_DIR


# ---------------------------------------------------------------------------
# Sealed artifact pins
# ---------------------------------------------------------------------------
def test_sealed_run_dir_inventory(run_dir):
    """The run dir holds exactly the nine top-level files (the six
    sealed payloads, the seal manifest, the sums file, the post-seal
    clean-environment record) plus the ``phase_b/`` and
    ``phase_b_smoke/`` trees; nothing else is present."""
    on_disk = {path.name for path in run_dir.iterdir()}
    assert on_disk == set(TOP_LEVEL_FILES) | {"phase_b", "phase_b_smoke"}, (
        "the sealed run dir drifted from the committed top-level set"
    )


def test_seal_manifest(run_dir):
    """``seal_manifest.json`` records the 1,537 sealed files (every file
    of the run dir except the manifest itself) totaling 16,990,201
    bytes, and every entry verifies against disk except the documented
    pre-seal clean-environment entry (the record was finalized by the
    post-seal verification; its on-disk digest is pinned separately)."""
    manifest = _load("seal_manifest.json")
    assert set(manifest) == {
        "schema", "run", "card", "created_utc", "file_count",
        "total_bytes", "sha256_sums_file", "sha256_sums_sha256",
        "excluded_from_sums", "path_convention", "note", "files",
    }
    assert manifest["schema"] == SEAL_SCHEMA
    assert manifest["run"] == RUN_ID
    assert manifest["card"] == CARD_ID
    assert manifest["created_utc"] == SEAL_CREATED_UTC
    assert manifest["file_count"] == N_SEAL_MANIFEST_ENTRIES
    assert manifest["total_bytes"] == SEAL_TOTAL_BYTES
    assert manifest["sha256_sums_file"] == "SHA256SUMS.txt"
    assert manifest["sha256_sums_sha256"] == SUMS_FILE_SHA256
    assert manifest["excluded_from_sums"] == EXCLUDED_FROM_SUMS
    assert "monorepo-root-relative" in manifest["path_convention"]

    files = manifest["files"]
    assert len(files) == N_SEAL_MANIFEST_ENTRIES
    paths = [entry["path"] for entry in files]
    assert paths == sorted(paths), (
        "the seal manifest file list is not in sorted relative-path "
        "order"
    )
    assert "seal_manifest.json" not in paths
    assert "SHA256SUMS.txt" in paths
    assert "clean_environment_commands.json" in paths

    for entry in files:
        assert set(entry) == {"path", "sha256", "bytes"}
        path = entry["path"]
        if path == "clean_environment_commands.json":
            # the documented pre-seal entry (stale by design)
            assert entry["sha256"] == CLEAN_SEALED_STALE_SHA256
            assert entry["bytes"] == CLEAN_SEALED_STALE_BYTES
            assert _file_sha256(run_dir / path) == CLEAN_ONDISK_SHA256, (
                "the post-seal clean-environment record drifted from "
                "the pinned on-disk digest"
            )
            assert (run_dir / path).stat().st_size == CLEAN_ONDISK_BYTES
        else:
            assert _file_sha256(run_dir / path) == entry["sha256"], (
                f"{path}: the sealed digest does not match the file on "
                "disk; the run dir was modified after the seal"
            )
            assert (run_dir / path).stat().st_size == entry["bytes"]

    # self-consistent byte totals: the recorded bytes account for the
    # sealed total; the on-disk bytes account for the post-seal total
    assert sum(entry["bytes"] for entry in files) == SEAL_TOTAL_BYTES
    assert (
        sum((run_dir / entry["path"]).stat().st_size for entry in files)
        == DISK_TOTAL_BYTES
    )

    sums_entry = next(
        entry for entry in files if entry["path"] == "SHA256SUMS.txt"
    )
    assert sums_entry["sha256"] == SUMS_FILE_SHA256
    assert _file_sha256(run_dir / "SHA256SUMS.txt") == SUMS_FILE_SHA256


def test_sha256_sums(run_dir):
    """``SHA256SUMS.txt`` carries the 1,535 monorepo-root-relative lines
    (the three excluded files are absent), covering the top level,
    ``phase_b``, and ``phase_b_smoke``, lexicographically sorted by
    relative path, and every digest verifies by recomputed SHA-256
    against the file on disk from the monorepo root."""
    sums_path = run_dir / "SHA256SUMS.txt"
    assert _file_sha256(sums_path) == SUMS_FILE_SHA256, (
        "the sums file drifted from the digest recorded in the seal "
        "manifest; the run dir was modified after the seal"
    )

    raw = sums_path.read_bytes()
    assert raw.endswith(b"\n")
    assert b"\r" not in raw
    lines = raw.decode("utf-8").splitlines()
    assert len(lines) == N_SUMS_LINES, (
        "SHA256SUMS.txt covers the drifted file set"
    )
    for line in lines:
        assert SUMS_LINE.match(line), (
            f"SHA256SUMS.txt is not in sha256sum -c format "
            f"(64 hex digits, two spaces, relative path): {line!r}"
        )
        assert SUMS_PREFIX in line, (
            f"SHA256SUMS.txt must carry monorepo-root-relative paths: "
            f"{line!r}"
        )

    entries = [line.split("  ", 1) for line in lines]
    rels = [rel for _digest, rel in entries]
    stripped = [rel[len(SUMS_PREFIX):] for rel in rels]
    top = [r for r in stripped if "/" not in r]
    phase_b = [r for r in stripped if r.startswith("phase_b/")]
    smoke = [r for r in stripped if r.startswith("phase_b_smoke/")]
    assert top == list(SUMS_TOP_LEVEL_FILES), (
        "the sums top-level group drifted from the committed set"
    )
    assert len(phase_b) == N_PHASE_B_FILES
    assert len(smoke) == N_PHASE_B_SMOKE_FILES
    # the entries are lexicographically sorted by relative path (the
    # "/" in the phase_b paths sorts before the "_" of the top-level
    # phase_b_aggregation_state.json)
    assert stripped == sorted(stripped)
    for name in EXCLUDED_FROM_SUMS:
        assert name not in stripped
        assert f"phase_b/{name}" not in stripped

    for digest, rel in entries:
        assert _file_sha256(MONO_ROOT / rel) == digest, (
            f"{rel}: the sealed digest does not match the file on "
            "disk; the run dir was modified after the seal"
        )


def test_pre_registration_structure(run_dir):
    """The sealed pre-registration carries the committed contract: the
    draws (0 for Phase A, 15,000,000 for Phase B), the unchanged
    3.0-sigma tolerance with the engine-floor note, ``no_fitting`` with
    no optimizer, the ten pinned input digests (catalog, continuum
    config, the five physics modules, the R31/card-9 drivers, the
    pre-registration driver digest), the per-window open sets, the
    ten named residuals and the 11 R21/R22 named residuals, the Phase A
    classification cutoffs, the E1-E4 estimator definitions, and the
    Phase B windows/nodes/statistics/seed map/decay routing/chunk
    plan/metric."""
    pre = _load("pre_registration.json")
    assert pre["schema"] == PRE_SCHEMA
    assert pre["run"] == RUN_ID
    assert pre["card"] == CARD_ID
    assert pre["written_before_evaluation"] is True
    assert pre["no_fitting"] is True
    assert "no optimizer" in pre["no_fitting_note"]
    assert pre["optimizer_invoked"] is False

    draws = pre["draws"]
    assert draws["phase_a"] == 0
    assert draws["phase_b"] == TOTAL_DECAYS

    tol = pre["no_tolerance_change"]
    assert tol["sigma_tolerance"] == TOLERANCE_SIGMA
    assert tol["unchanged"] is True
    assert "DETERMINISTIC_SE_FLOOR" in tol["engine_floor_note"]

    inputs = pre["input_hashes"]
    assert inputs["catalog_sha256"] == CATALOG_SHA256
    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)
    assert inputs["families_config_sha256"] == CONFIG_SHA256
    assert _file_sha256(CONFIG_PATH) == CONFIG_SHA256
    assert inputs["spectral_density_module_sha256"] == (
        SPECTRAL_DENSITY_MODULE_SHA256
    )
    assert _file_sha256(SPECTRAL_DENSITY_MODULE_PATH) == (
        SPECTRAL_DENSITY_MODULE_SHA256
    )
    assert inputs["resonance_decay_module_sha256"] == (
        RESONANCE_DECAY_MODULE_SHA256
    )
    assert _file_sha256(RESONANCE_DECAY_MODULE_PATH) == (
        RESONANCE_DECAY_MODULE_SHA256
    )
    assert inputs["resonance_widths_module_sha256"] == (
        RESONANCE_WIDTHS_MODULE_SHA256
    )
    assert _file_sha256(RESONANCE_WIDTHS_MODULE_PATH) == (
        RESONANCE_WIDTHS_MODULE_SHA256
    )
    assert inputs["rng_substreams_module_sha256"] == (
        RNG_SUBSTREAMS_MODULE_SHA256
    )
    assert _file_sha256(RNG_SUBSTREAMS_MODULE_PATH) == (
        RNG_SUBSTREAMS_MODULE_SHA256
    )
    assert inputs["threshold_metric_module_sha256"] == ENGINE_SHA256
    assert _file_sha256(ENGINE_PATH) == ENGINE_SHA256
    assert inputs["r31_driver_sha256"] == R31_DRIVER_SHA256
    assert _file_sha256(R31_DRIVER_PATH) == R31_DRIVER_SHA256
    assert _file_sha256(R31_DRIVER_OUTER_PATH) == R31_DRIVER_SHA256
    assert inputs["card9_driver_sha256"] == CARD9_DRIVER_SHA256
    # the driver digest at pre-registration (the dual-tree sync commit
    # moved it to DRIVER_SHA256_CURRENT; the committed copies are
    # cross-checked in the provenance test)
    assert inputs["driver_sha256"] == DRIVER_SHA256_AT_PREREGISTRATION

    open_sets = pre["open_sets"]
    assert set(open_sets) == set(OPEN_SETS)
    for window, pinned_ids in OPEN_SETS.items():
        nodes = open_sets[window]
        assert set(nodes) == {"0", "1", "2", "3", "4"}
        for node_ids in nodes.values():
            assert node_ids == pinned_ids, (
                f"window {window}: the open set drifted from the "
                "pre-registration (identical on all five nodes)"
            )

    phase_a = pre["phase_a"]
    assert phase_a["deliverable"] == "phase_a_diagnostics.json"
    assert phase_a["replay"]["draws"] == 0

    named = phase_a["named_residuals"]
    assert len(named) == len(NAMED_RESIDUALS_PHASE_A)
    for row, pinned in zip(named, NAMED_RESIDUALS_PHASE_A):
        assert (
            row["index"], row["window"], row["pair"], row["species"],
            row["observable"], row["pre_fix_sigma"],
        ) == pinned[:6]

    r21r22 = phase_a["r21r22_named_residuals"]
    assert len(r21r22) == len(NAMED_RESIDUALS_R21R22)
    for row, pinned in zip(r21r22, NAMED_RESIDUALS_R21R22):
        assert (
            row["run"], row["pair"], row["species"], row["observable"],
            row["pair_class"], row["sigma"],
        ) == pinned[:6]

    cutoffs = phase_a["classification_thresholds"]
    assert cutoffs["ratio_artifact"] == 3.0
    assert cutoffs["ratio_genuine"] == 1.5
    assert cutoffs["correlation_flag"] == 0.2
    assert cutoffs["no_tolerance_change"] is True
    assert "window 2 node 1 (2.58 GeV)" in cutoffs["node1_tight_se_target"]

    est = phase_a["estimator_definitions"]
    assert est["block_lengths"] == [32, 128]
    assert est["bootstrap_resamples"] == 2000
    assert est["bootstrap_resampler_seed"] == MASTER_SEED
    for key in ("E1", "E2", "E3", "E4"):
        assert key in est

    assert phase_a["seed_map_audit"]["correlation_flag_cutoff"] == 0.2
    assert "all 15 grid nodes" in phase_a["sibling_scope"]["statement"]

    phase_b = pre["phase_b"]
    windows = phase_b["windows"]
    assert len(windows) == N_WINDOWS
    for row, pinned in zip(windows, WINDOWS):
        assert (
            row["window"], row["qn"], row["label"],
            row["family_expectation"],
        ) == pinned
    assert [row["window"] for row in windows] == [0, 1, 2]

    nodes = phase_b["nodes"]
    assert nodes["masses_gev"] == NODE_MASSES_GEV
    assert nodes["boundary_gev"] == BOUNDARY_GEV

    stats = phase_b["statistics"]
    assert stats["replicas"] == N_REPLICAS
    assert stats["decays_per_replica"] == DECAYS_PER_REPLICA
    assert stats["decays_per_node"] == N_REPLICAS * DECAYS_PER_REPLICA
    assert stats["windows"] == N_WINDOWS
    assert stats["nodes"] == N_NODES
    assert stats["total_decays"] == TOTAL_DECAYS

    seed_map = phase_b["seed_map"]
    assert seed_map["master_seed"] == MASTER_SEED
    assert seed_map["substream_labels"] == [
        "decay_channel", "decay_kinematics",
    ]
    assert "20260918" in seed_map["rule"]

    routing = phase_b["decay_routing"]
    assert routing["boundary_gev"] == BOUNDARY_GEV
    assert routing["ramp_width_gev"] == 0.5
    assert routing["expected_w_at_nodes"] == EXPECTED_W_AT_NODES

    chunk_plan = phase_b["chunk_plan"]
    assert chunk_plan["chunks"] == N_CHUNKS
    assert chunk_plan["chunk_size_decays"] == CHUNK_SIZE_DECAYS
    assert chunk_plan["one_job_per_chunk"] is True
    assert chunk_plan["gateway_tasks"] == 1
    assert chunk_plan["single_math_thread"] is True
    assert chunk_plan["no_local_process_pool"] is True
    assert chunk_plan["retry_cap"] == 3

    metric = phase_b["metric"]
    assert metric["id"] == "delta_analytic_full"
    assert metric["tolerance_sigma"] == TOLERANCE_SIGMA
    assert "422" in metric["statement"]
    assert "never relaxed" in metric["tolerance_statement"]

    gateway = phase_b["gateway"]
    assert gateway["host"] == "cl"
    assert len(gateway["workers"]) == 9
    assert gateway["task_slots_per_worker"] == 32
    assert gateway["max_concurrent_slots"] == 288


def test_provenance_structure(run_dir):
    """The provenance records the dual-tree driver digests (the
    pre-registration digest 9ccf36c5 and the post-seal committed digest
    2c72fa7e, the nested and monorepo copies byte-identical), the frozen
    R31 driver, the fixed card-2 engine (cross-checked against disk),
    the git identity, the environment fingerprint, the seeds, the
    catalog and continuum-config digests, and the input hashes."""
    prov = _load("provenance.json")
    assert prov["schema"] == PROV_SCHEMA
    assert prov["run"] == RUN_ID
    assert prov["card"] == CARD_ID
    assert UTC_STAMP.match(prov["written_utc"])
    assert prov["written_before_evaluation"] is True

    assert prov["driver"] == (
        "helpers/benchmarking/staged_r3_se_audit.py"
    )
    assert prov["driver_sha256"] == DRIVER_SHA256_AT_PREREGISTRATION
    assert prov["r31_driver"] == (
        "helpers/benchmarking/staged_r31_matching_scan.py"
    )
    assert prov["r31_driver_sha256"] == R31_DRIVER_SHA256
    assert _file_sha256(R31_DRIVER_PATH) == R31_DRIVER_SHA256
    assert _file_sha256(R31_DRIVER_OUTER_PATH) == R31_DRIVER_SHA256
    assert "frozen" in prov["r31_driver_byte_identity_note"]

    engine = prov["fixed_engine"]
    assert engine["path"] == "dev/mixing2006/na49_mixing/threshold_metric.py"
    assert engine["sha256"] == ENGINE_SHA256
    assert _file_sha256(ENGINE_PATH) == ENGINE_SHA256
    assert "card-2 engine after the 8a P1 fix" in engine["note"]

    assert prov["git"] == {
        "nested_head": "95096d2df4e0a10fd1c6573388138dea0c3fe20c",
        "nested_dirty_entries": 10,
        "outer_head": "855635c5e5b83d44223a43558978440605b5dd83",
        "outer_dirty_entries": 4,
    }
    assert prov["environment"] == {
        "python_version": "3.12.3",
        "python_executable": "/usr/bin/python",
        "platform": "linux",
        "numpy_version": "1.26.4",
    }

    assert prov["catalog"] == {
        "path": "config/v2_singlet/resonance_catalog.json",
        "sha256": CATALOG_SHA256,
    }
    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)
    assert prov["continuum_config"] == {
        "path": "config/v2_singlet/continuum_resonances.json",
        "sha256": CONFIG_SHA256,
    }
    assert _file_sha256(CONFIG_PATH) == CONFIG_SHA256

    seeds = prov["seeds"]
    assert seeds["confirmation_master_seed"] == MASTER_SEED
    assert seeds["r31_master_seed"] == 20260917
    assert seeds["r20_r22_master_seed"] == 20260913
    assert seeds["r23_raw_replica_seeds"] == "20300913..20300920"

    sealed_inputs = prov["sealed_inputs"]
    assert sealed_inputs["inventory"] == "pre_investigation_inventory.json"
    for key in ("r31", "r21", "r22", "card2_reregistration"):
        assert key in sealed_inputs

    inputs = prov["input_hashes"]
    assert inputs["catalog_sha256"] == CATALOG_SHA256
    assert inputs["families_config_sha256"] == CONFIG_SHA256
    assert inputs["threshold_metric_module_sha256"] == ENGINE_SHA256
    assert inputs["r31_driver_sha256"] == R31_DRIVER_SHA256
    assert inputs["card9_driver_sha256"] == CARD9_DRIVER_SHA256
    assert inputs["driver_sha256"] == DRIVER_SHA256_AT_PREREGISTRATION
    assert inputs["spectral_density_module_sha256"] == (
        SPECTRAL_DENSITY_MODULE_SHA256
    )
    assert inputs["resonance_decay_module_sha256"] == (
        RESONANCE_DECAY_MODULE_SHA256
    )
    assert inputs["resonance_widths_module_sha256"] == (
        RESONANCE_WIDTHS_MODULE_SHA256
    )
    assert inputs["rng_substreams_module_sha256"] == (
        RNG_SUBSTREAMS_MODULE_SHA256
    )

    gateway = prov["gateway"]
    assert gateway["host"] == "cl"
    assert len(gateway["workers"]) == 9
    assert gateway["task_slots_per_worker"] == 32
    assert gateway["max_concurrent_slots"] == 288

    remote = prov["remote_checkout"]
    assert remote["path"].startswith("~/NA49/2026/01-NA49-res")
    assert "not a git repository" in remote["git"]

    # the committed dual-tree driver carries the post-seal digest
    assert _file_sha256(DRIVER_PATH) == DRIVER_SHA256_CURRENT
    assert _file_sha256(DRIVER_OUTER_PATH) == DRIVER_SHA256_CURRENT


def test_phase_a_diagnostics_structure(run_dir):
    """The Phase A report carries the node-1 tight-SE verdict (ratio
    9.691627288965723, ``estimator_artifact``, the window-2 node-1
    correlation flag), the ten named residuals (nine
    ``seed_issue_flag``, one ``estimator_artifact``) with their honest
    E4 b=32 sigmas, the 11 R21/R22 named residual classifications,
    the seed-map audit (the confirmation map disjoint from all others
    with zero internal duplicates, the 100 shared archive rows), the
    replica-correlation audit (17 flagged pairs), the bit-identical
    replays (R31 15 cells, R21 11 points, R22 6 points), the
    classification cutoffs, and the estimator cells."""
    payload = _load("phase_a_diagnostics.json")
    assert payload["schema"] == PHASE_A_SCHEMA
    assert payload["run"] == RUN_ID
    assert payload["card"] == CARD_ID
    assert payload["pre_registration"] == {
        "path": "pre_registration.json",
    }
    assert payload["no_fitting"] is True
    assert payload["optimizer_invoked"] is False
    assert payload["no_tolerance_change"] == {
        "sigma_tolerance": TOLERANCE_SIGMA,
        "unchanged": True,
    }

    cutoffs = payload["classification_thresholds"]
    assert cutoffs["ratio_artifact"] == 3.0
    assert cutoffs["ratio_genuine"] == 1.5
    assert cutoffs["correlation_flag"] == 0.2

    node1 = payload["node1_tight_se"]
    assert node1["target"] == "window 2 node 1 (2.58 GeV) pi_0 mean_p"
    for node, (e1, e4, ratio) in NODE1_TIGHT_SE_NODES.items():
        entry = node1["nodes"][node]
        assert entry["e1"] == e1
        assert entry["e4_b32"] == e4
        assert entry["ratio_e4_b32_over_e1"] == ratio
    assert node1["sibling_ratios"] == NODE1_SIBLING_RATIOS
    assert node1["ratio_target_node"] == NODE1_RATIO_TARGET
    assert node1["verdict"] == "estimator_artifact"
    assert node1["correlation_flag_window2_node1"] is True

    named = payload["named_residuals"]
    assert len(named) == len(NAMED_RESIDUALS_PHASE_A)
    for row, pinned in zip(named, NAMED_RESIDUALS_PHASE_A):
        assert (
            row["index"], row["window"], row["pair"], row["species"],
            row["observable"], row["pre_fix_sigma_card"],
            row["classification"], row["sigma_honest_e4_b32"],
        ) == pinned
    assert sum(
        row["classification"] == "seed_issue_flag" for row in named
    ) == 9
    assert sum(
        row["classification"] == "estimator_artifact" for row in named
    ) == 1

    r21r22 = payload["r21r22_named_residuals"]
    assert len(r21r22) == len(NAMED_RESIDUALS_R21R22)
    for row, pinned in zip(r21r22, NAMED_RESIDUALS_R21R22):
        assert (
            row["run"], row["pair"], row["species"], row["observable"],
            row["pair_class"], row["sigma_rereval_sealed"],
            row["classification"],
        ) == pinned

    audit = payload["seed_map_audit"]
    assert audit["substream_labels"] == [
        "decay_channel", "decay_kinematics",
    ]
    covered = audit["covered_maps"]
    assert covered["confirmation_full"] == {
        "cells": 15, "triples": TOTAL_DECAYS,
        "unique_id_rows": TOTAL_DECAYS, "duplicate_id_rows": 0,
        "zero_duplicates": True,
    }
    assert covered["r20_r23_sampled"] == {
        "cells": 2, "triples": 784, "unique_id_rows": 784,
        "duplicate_id_rows": 0, "zero_duplicates": True,
    }
    assert covered["r21r22_replayed_cells"] == {
        "cells": 17, "triples": 1_700_000, "unique_id_rows": 1_500_000,
        "duplicate_id_rows": 200_000, "zero_duplicates": False,
    }
    assert covered["r31_full"] == {
        "cells": 15, "triples": 1_500_000, "unique_id_rows": 1_500_000,
        "duplicate_id_rows": 0, "zero_duplicates": True,
    }
    assert audit["pairwise_shared_rows"] == {
        "confirmation_full|r20_r23_sampled": 0,
        "confirmation_full|r21r22_replayed_cells": 0,
        "confirmation_full|r31_full": 0,
        "r20_r23_sampled|r21r22_replayed_cells": 100,
        "r20_r23_sampled|r31_full": 0,
        "r21r22_replayed_cells|r31_full": 0,
    }
    assert audit["confirmation_map_disjoint_from_all_others"] is True
    assert audit["confirmation_map_zero_internal_duplicates"] is True
    assert audit["archive_group_overlaps"] == {
        "r20_r23_sampled|r21r22_replayed_cells": 100,
    }

    corr = payload["replica_correlation"]
    assert corr["correlation_flag_cutoff"] == 0.2
    assert corr["flagged_pairs"] == REPLICA_CORRELATION_FLAGGED_PAIRS
    assert corr["flag"] is True
    assert set(corr["cells"]) == set(REPLICA_CORRELATION_CELLS)
    for cell, (components, used, flagged, flag, max_abs_r) in (
        REPLICA_CORRELATION_CELLS.items()
    ):
        entry = corr["cells"][cell]
        assert entry["components"] == components
        assert entry["components_used"] == used
        assert entry["flagged_pairs"] == flagged
        assert entry["flag"] is flag
        assert entry["max_abs_r"] == max_abs_r
        assert sum(pair["flagged"] for pair in entry["pairs"]) == flagged

    replay = payload["replay"]
    assert replay["zero_new_random_numbers"] is True
    r31 = replay["r31"]
    assert r31["cells"] == 15
    assert r31["replicas_per_cell"] == 4
    assert r31["decays_per_replica"] == 25_000
    assert r31["chunk_size"] == 2500
    assert r31["replicas_checked"] == 60
    assert r31["nodes_checked"] == 15
    assert r31["all_exact"] is True
    assert len(r31["cells_checked"]) == 15
    assert all(
        entry["exact"] is True for entry in r31["cells_checked"].values()
    )
    for run_name in ("r21", "r22"):
        entry = replay[run_name]
        assert entry["master_seed"] == 20260913
        assert entry["seed_rule"] == (
            "derive_substreams(20260913, r, k * 1_000_000 + i)"
        )
        assert entry["all_exact"] is True
        assert all(cell["exact"] for cell in entry["cells"])
    assert len(replay["r21"]["cells"]) == 11
    assert replay["r21"]["replicas_checked"] == 44
    assert len(replay["r22"]["cells"]) == 6
    assert replay["r22"]["replicas_checked"] == 24

    cells = payload["estimator_cells"]
    assert set(cells["R31"]) == set(REPLICA_CORRELATION_CELLS)
    assert set(cells["R21"]) == {
        "k00", "k01", "k02", "k03", "k04", "k05", "k06",
        "k15", "k16", "k18", "k19",
    }
    assert set(cells["R22"]) == {"k07", "k08", "k15", "k16", "k23", "k24"}


def test_confirmation_verification_structure(run_dir):
    """The re-evaluation carries the 422/422 verdict (14 old failures,
    0 new failures, max absolute sigma 2.959754302712887, all 422 rows
    matched), the driver digest drift (9ccf36c5 to 2c72fa7e, one field),
    the 14 original failures all passing at the new statistics, the
    ten named residuals all ``genuine_finite_mc_scatter``, the routing
    fractions of all 15 nodes, the 10 delta drift rows of the card-8a
    P1 engine fix, and the full per-window check table."""
    payload = _load("confirmation_verification.json")
    assert payload["schema"] == REEVAL_SCHEMA
    assert payload["run"] == RUN_ID
    assert payload["card"] == CARD_ID
    assert payload["no_fitting"] is True
    assert payload["optimizer_invoked"] is False

    metric = payload["metric"]
    assert metric["id"] == "delta_analytic_full"
    assert metric["tolerance_sigma"] == TOLERANCE_SIGMA
    assert metric["tolerance_unchanged"] is True
    assert metric["statement"] == METRIC_STATEMENT

    stats = payload["statistics"]
    assert stats == {
        "replicas": N_REPLICAS,
        "decays_per_replica": DECAYS_PER_REPLICA,
        "master_seed": MASTER_SEED,
    }

    drift = payload["driver_sha256"]
    assert drift["at_preregistration"] == DRIVER_SHA256_AT_PREREGISTRATION
    assert drift["at_reeval"] == DRIVER_SHA256_CURRENT
    assert drift["drift"] == [
        {
            "field": "input_hashes.driver_sha256",
            "at_preregistration": DRIVER_SHA256_AT_PREREGISTRATION,
            "current": DRIVER_SHA256_CURRENT,
        }
    ]

    summary = payload["summary"]
    assert summary["checks_applied_old"] == N_CHECKS
    assert summary["checks_failed_old"] == N_OLD_FAILURES
    assert summary["checks_applied_new"] == N_CHECKS
    assert summary["checks_failed_new"] == 0
    assert summary["max_abs_sigma_new"] == MAX_ABS_SIGMA_NEW
    assert summary["matched_old_rows"] == N_CHECKS
    assert summary["old_failures_pass_new"] == N_OLD_FAILURES
    assert "never relaxed" in summary["tolerance_note"]
    assert len(summary["delta_drift_rows"]) == len(DELTA_DRIFT_ROWS)
    for row, pinned in zip(summary["delta_drift_rows"], DELTA_DRIFT_ROWS):
        assert (
            row["window"], row["pair"], row["species"], row["observable"],
            row["delta_sealed"], row["delta_fixed_engine"],
        ) == pinned

    failures = payload["original_14_failures"]
    assert len(failures) == N_OLD_FAILURES
    for row, pinned in zip(failures, ORIGINAL_14_FAILURES):
        assert (
            row["window"], row["pair"], row["species"], row["observable"],
            row["sigma_old_sealed"], row["sigma_new"],
            row["status_new"], row["delta_equal_sealed"],
        ) == (*pinned[:6], "pass", pinned[6])

    named = payload["named_residuals_phase_b"]
    assert len(named) == len(NAMED_RESIDUALS_PHASE_B)
    for row, pinned in zip(named, NAMED_RESIDUALS_PHASE_B):
        assert (
            row["index"], row["window"], row["pair"], row["species"],
            row["observable"], row["sigma_old_sealed"], row["sigma_new"],
        ) == pinned
        assert row["classification_phase_b"] == "genuine_finite_mc_scatter"

    routing = payload["routing_fractions"]
    assert set(routing) == set(ROUTING_FRACTIONS)
    for cell, (frac, w, binom, n_cat, n_cont) in ROUTING_FRACTIONS.items():
        entry = routing[cell]
        assert entry["fraction_continuum"] == frac
        assert entry["expected_w"] == w
        assert entry["binomial_sigma"] == binom
        assert entry["n_catalog"] == n_cat
        assert entry["n_continuum"] == n_cont
        assert entry["n_catalog"] + entry["n_continuum"] == 1_000_000

    windows = payload["windows"]
    assert set(windows) == {"0", "1", "2"}
    qn_by_window = {0: [1, 1, 0], 1: [1, 0, -1], 2: [0, 0, 0]}
    n_checks_total = 0
    drift_rows_seen = set()
    for window, entry in windows.items():
        assert set(entry) == {"pairs", "qn"}
        assert entry["qn"] == qn_by_window[int(window)]
        assert set(entry["pairs"]) == {"0", "1", "2", "3"}
        for pair_key, pair in entry["pairs"].items():
            assert set(pair) == {
                "pair", "global_pair", "masses_gev", "w_at_nodes",
                "boundary_straddling", "catalog_only", "checks",
            }
            j = int(pair_key)
            assert pair["w_at_nodes"] == EXPECTED_W_AT_NODES[j : j + 2]
            for check in pair["checks"]:
                if check["status"] == "absent_species":
                    assert check["pass"] is None
                    continue
                n_checks_total += 1
                assert check["status"] == "pass"
                assert check["pass"] is True
                assert check["tolerance_sigma"] == TOLERANCE_SIGMA
                assert abs(check["sigma"]) <= TOLERANCE_SIGMA
                if check["delta_equal_sealed"] is False:
                    drift_rows_seen.add(
                        (
                            int(window), tuple(check["pair"]),
                            check["species"], check["observable"],
                        )
                    )
    assert n_checks_total == N_CHECKS
    assert drift_rows_seen == {
        (w, tuple(p), s, o) for w, p, s, o, _d1, _d2 in DELTA_DRIFT_ROWS
    }


def test_phase_b_aggregation_state(run_dir):
    """The aggregation state records 750/750 completed chunks (each
    20,000 decays) over the full (window, node, replica) coverage with
    all 15 nodes merged in replica order."""
    payload = _load("phase_b_aggregation_state.json")
    assert payload["schema"] == AGGREGATION_SCHEMA
    assert payload["run"] == RUN_ID
    assert payload["card"] == CARD_ID
    assert payload["all_nodes_merged"] is True
    assert payload["chunk_dir"] == "phase_b/"
    assert payload["expected_chunks"] == N_CHUNKS
    assert payload["n_completed"] == N_CHUNKS
    assert payload["smoke"] is False
    assert payload["nodes_merged"] == [
        f"w{w}_j{j}" for w in range(N_WINDOWS) for j in range(N_NODES)
    ]
    assert "replica order" in payload["merge_order"]

    completed = payload["completed_chunks"]
    assert len(completed) == N_CHUNKS
    coverage = set()
    for name, entry in completed.items():
        assert set(entry) == {"window", "node", "replica", "n_decays"}
        assert entry["n_decays"] == CHUNK_SIZE_DECAYS
        w, j, r = entry["window"], entry["node"], entry["replica"]
        assert name == f"chunk_w{w}_n{j}_r{r}.json"
        coverage.add((w, j, r))
    assert coverage == {
        (w, j, r)
        for w in range(N_WINDOWS)
        for j in range(N_NODES)
        for r in range(N_REPLICAS)
    }


def test_pre_investigation_inventory(run_dir):
    """The inventory re-verified all 78 sealed R31 files byte for byte
    (every recorded digest and size cross-checked against the R31 run
    dir on disk), records the sealed R31 pre-registration digest
    (cross-checked against the R31 dir), and carries the seal states of
    the R31/R21/R22/card-2 archives."""
    payload = _load("pre_investigation_inventory.json")
    assert payload["schema"] == INVENTORY_SCHEMA
    assert payload["run"] == RUN_ID
    assert payload["card"] == CARD_ID
    assert payload["r31_run_dir"] == (
        "dev/00runs/r31_catalog_continuum_matching_scan_2026-09-17"
    )
    assert payload["r31_file_count"] == 78
    assert len(payload["r31_files"]) == 78

    # The card-8b resume regenerated matching_verification.json and
    # updated provenance.json in the R31 run dir (see the
    # R31_RESUMED_* note above); the inventory's other 76 entries are
    # the resume's bit-identity set and must still match byte for byte.
    resumed = {
        "matching_verification.json": R31_RESUMED_VERIFICATION_SHA256,
        "provenance.json": R31_RESUMED_PROVENANCE_SHA256,
    }
    for entry in payload["r31_files"]:
        assert set(entry) == {"path", "sha256", "bytes"}
        assert entry["path"].startswith(payload["r31_run_dir"] + "/")
        name = entry["path"].rsplit("/", 1)[1]
        assert _file_sha256(MONO_ROOT / entry["path"]) == resumed.get(
            name, entry["sha256"]
        ), (
            f"{entry['path']}: the sealed R31 file drifted; the "
            "read-only archive was modified"
        )
        if name not in resumed:
            assert (MONO_ROOT / entry["path"]).stat().st_size == (
                entry["bytes"]
            )

    # the recorded "driver" digest is the sealed R31 pre-registration
    # (the field name is historical)
    assert payload["r31_driver_sha256_in_inventory"] == (
        R31_PRE_REGISTRATION_SHA256
    )
    assert _file_sha256(R31_RUN_DIR / "pre_registration.json") == (
        R31_PRE_REGISTRATION_SHA256
    )

    assert payload["seal_states"] == SEAL_STATES
    assert "read-only" in payload["read_only_inputs_note"]


def test_clean_environment_record(run_dir):
    """The post-seal clean-environment record re-verified the 1,535
    sums-covered files without mismatches, re-ran ``--phase-a`` /
    ``--aggregate`` / ``--reeval`` in the clean interpreter
    (``python -I``, return code 0) against the committed dual-tree
    driver, and carries the 768 sealed-report digests (a pinned subset
    cross-checked against disk).  The on-disk record carries the
    pinned post-seal digest (the seal manifest keeps the pre-seal
    entry)."""
    record = _load("clean_environment_commands.json")
    assert set(record) == {
        "schema", "run", "card", "verified_utc", "seal_verification",
        "commands", "environment", "hashes_match", "probe_tool",
        "sealed_report_hashes",
    }
    assert record["schema"] == CLEAN_SCHEMA
    assert record["run"] == RUN_ID
    assert record["card"] == CARD_ID
    assert record["verified_utc"] == CLEAN_VERIFIED_UTC
    assert record["hashes_match"] is True
    assert record["seal_verification"] == {
        "files_checked": N_SUMS_LINES,
        "mismatches": [],
    }

    commands = record["commands"]
    assert len(commands) == 3
    purposes = [command["purpose"] for command in commands]
    assert purposes == [
        "clean-interpreter re-run of --phase-a",
        "clean-interpreter re-run of --aggregate",
        "clean-interpreter re-run of --reeval",
    ]
    modes = ["--phase-a", "--aggregate", "--reeval"]
    for command, mode in zip(commands, modes):
        assert set(command) == {"purpose", "argv", "cwd", "returncode"}
        assert command["cwd"] == str(MONO_ROOT)
        assert command["returncode"] == 0
        argv = command["argv"]
        assert argv[0] == "/usr/bin/python"
        assert argv[1] == "-I"
        assert argv[2] == str(DRIVER_PATH)
        assert mode in argv
        assert "--run-dir" in argv
        assert argv[argv.index("--run-dir") + 1] == str(RUN_DIR)

    assert record["environment"] == {
        "python_version": "3.12.3",
        "python_executable": "/usr/bin/python",
        "platform": "linux",
        "numpy_version": "1.26.4",
    }

    probe = record["probe_tool"]
    assert probe["path"] == str(DRIVER_PATH)
    assert probe["sha256"] == DRIVER_SHA256_CURRENT
    assert _file_sha256(DRIVER_PATH) == DRIVER_SHA256_CURRENT
    assert _file_sha256(DRIVER_OUTER_PATH) == DRIVER_SHA256_CURRENT
    assert "probe" in probe["note"]

    sealed = record["sealed_report_hashes"]
    assert len(sealed) == N_SEALED_REPORT_HASHES
    for name, entry in sealed.items():
        assert set(entry) == {"hashes_match", "sha256"}
        assert entry["hashes_match"] is True
    for name, digest in SEALED_REPORT_HASH_PIN.items():
        entry = sealed[name]
        assert set(entry) == {"hashes_match", "sha256"}
        assert entry["hashes_match"] is True
        assert entry["sha256"] == digest
        assert _file_sha256(run_dir / name) == digest, (
            f"{name}: the sealed report drifted from the "
            "clean-environment record"
        )

    assert _file_sha256(run_dir / "clean_environment_commands.json") == (
        CLEAN_ONDISK_SHA256
    )
    assert (
        run_dir / "clean_environment_commands.json"
    ).stat().st_size == CLEAN_ONDISK_BYTES


def test_gateway_submission_and_fetch(run_dir):
    """The submission manifest carries the 750 jobs (the chunk names,
    the job ids, the nine gateway workers), the 1,500 superseded jobs,
    the three retry counts, and the resubmission note; the fetch log
    records all 750 chunks ``COMPLETED`` with byte counts and digests
    (the digests verify against the chunk files on disk)."""
    submission = _load("phase_b/submission_manifest.json")
    assert submission["schema"] == SUBMISSION_SCHEMA
    assert submission["run"] == RUN_ID
    assert submission["card"] == CARD_ID
    assert submission["capacity_at_submit_free_tasks"] == (
        CAPACITY_AT_SUBMIT_FREE_TASKS
    )
    assert submission["remote_project"] == "NA49/2026/01-NA49-res"
    assert submission["remote_driver_sha256_verified"] is None
    assert submission["retry_counts"] == RETRY_COUNTS
    assert "wave 1 failed remotely" in submission["resubmission_note"]
    assert submission["chunk_command_template"].startswith(
        "python3 dev/mixing2006/helpers/benchmarking/"
        "staged_r3_se_audit.py --chunk"
    )

    jobs = submission["jobs"]
    assert len(jobs) == N_CHUNKS
    expected_chunks = {
        f"chunk_w{w}_n{j}_r{r}.json"
        for w in range(N_WINDOWS)
        for j in range(N_NODES)
        for r in range(N_REPLICAS)
    }
    assert {entry["chunk"] for entry in jobs.values()} == expected_chunks
    for key, entry in jobs.items():
        assert set(entry) == {"chunk", "job_id", "worker"}
        w, j, r = entry["chunk"].split("_")[1:]
        assert key == f"r3se_{w}_{j}_r{int(r[1:-5]):02d}"
        assert JOB_ID.match(entry["job_id"])
        assert entry["worker"] in GATEWAY_WORKERS

    superseded = submission["superseded_jobs"]
    assert len(superseded) == N_CHUNKS
    total_superseded = 0
    for entry_list in superseded.values():
        total_superseded += len(entry_list)
        for entry in entry_list:
            assert set(entry) == {"chunk", "job_id", "worker"}
            assert JOB_ID.match(entry["job_id"])
            assert entry["worker"] in GATEWAY_WORKERS
    assert total_superseded == N_SUPERSEDED_JOBS

    fetch = _load("phase_b/fetch_log.json")
    assert fetch["schema"] == FETCH_SCHEMA
    assert fetch["run"] == RUN_ID
    assert fetch["card"] == CARD_ID
    fetched = fetch["fetched"]
    assert len(fetched) == N_CHUNKS
    assert set(fetched) == expected_chunks
    for name, entry in fetched.items():
        assert set(entry) == {"bytes", "job_state", "sha256"}
        assert entry["job_state"] == "COMPLETED"
        assert _file_sha256(run_dir / "phase_b" / name) == entry["sha256"], (
            f"{name}: the fetched chunk digest does not match the "
            "file on disk"
        )
        assert (run_dir / "phase_b" / name).stat().st_size == entry["bytes"]
