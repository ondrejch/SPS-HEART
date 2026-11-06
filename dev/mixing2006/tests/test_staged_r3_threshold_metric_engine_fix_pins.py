"""Round 3 card 8a phase 2: engine-fix live re-evaluation artifact pins.

Task ``TASK-20260918-r3-card2-engine-three-body-moment-fix`` (phase 2;
nested P1 commit ``95096d2``; the card-2 engine re-opened by the
stopped R31 matching verification).  This module pins the sealed run
evidence under
``dev/00runs/r3_threshold_metric_engine_fix_2026-09-18/``: the
pre-registration (the three engine mechanisms by name and rule, the
deterministic-SE floor rule with ``DETERMINISTIC_SE_FLOOR = 1e-12``,
master seed 20260918 with ``draws: 0``, the 834-applied-check R21/R22
scope, the 422-applied-check targeted R31 scope, and the T1/T2
acceptance thresholds), the provenance (the fixed engine sha256
``ee240f44...``, the dual-tree driver digest, the sealed archive
digests, and the sealed card-2 baseline report digest), the three live
re-evaluation payloads, the revised clause report
(``gate2_revised_threshold_v2.json``, the sealed card-2 baseline
alongside the post-fix values), the seal manifest, ``SHA256SUMS.txt``
(monorepo-root-relative entries, ``sha256sum -c`` format), and the
post-seal clean-environment record.

The committed verdict of the live re-evaluation (pinned here
bit-exact): the fixed engine resolves the two Lambda(1520) three-body
R31 checks (+7.226 -> +0.920 and -5.611 -> +0.370) and the two
float-precision SE artifacts (vacuous passes under the floor
convention), but the card's T1 and T2 acceptance thresholds are NOT
met: 10 of the 14 targeted R31 failures persist at their sealed
sigmas (the committed engine's moment-approximation floor, unchanged
on those record sets), the R21/R22 combined failure count is 12
against the sealed baseline of 11, the four sealed
moment-approximation-floor residuals all still fail (3.362 / 3.599 /
3.041 / 4.974 sigma), and the pass rate among the original 323
failures is 0.9628482972136223 against the sealed 0.9659442724458205.
The pre-registered acceptance verdicts are recorded as findings inside
the sealed evidence (``acceptance.pass == false`` in
``gate2_revised_threshold_v2.json``); the 3.0-sigma tolerance is never
relaxed and no residual is absorbed into the metric.

Sibling of ``tests/test_staged_r3_threshold_metric_reregistration_pins.py``
and ``tests/test_staged_r30_spectral_normalization_pins.py`` (the same
pin style: sealed-dir discovery, skip-or-fail when the gitignored
archive is absent, digest pins plus structural pins).  These pins are
read-only over the sealed artifacts: nothing is recomputed, the sealed
run is never re-sealed, and the post-seal clean-environment record is
never re-executed.

Hermetic: reads the sealed run dir (gitignored; the artifact is
present on disk), the committed config, catalog, package, and drivers;
no computation runs.  If the run dir is absent (a fresh checkout),
the artifact pins skip with a clear message.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

NESTED_ROOT = Path(__file__).resolve().parent.parent
MONO_ROOT = NESTED_ROOT.parent.parent
RUN_NAME = "r3_threshold_metric_engine_fix_2026-09-18"
RUN_DIR = MONO_ROOT / "dev" / "00runs" / RUN_NAME

BASELINE_RUN_NAME = "r3_threshold_metric_reregistration_2026-09-17"
BASELINE_DIR = MONO_ROOT / "dev" / "00runs" / BASELINE_RUN_NAME
R31_DIR = MONO_ROOT / "dev" / "00runs" / (
    "r31_catalog_continuum_matching_scan_2026-09-17")

CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
METRIC_MODULE_PATH = NESTED_ROOT / "na49_mixing" / "threshold_metric.py"
DRIVER_PATH = NESTED_ROOT / "helpers" / "benchmarking" / (
    "rereval_threshold_checks.py"
)
DRIVER_OUTER_PATH = (
    MONO_ROOT / "helpers" / "benchmarking" / "rereval_threshold_checks.py"
)

# ---------------------------------------------------------------------------
# The committed run contract (the sealed run realizes it).
# ---------------------------------------------------------------------------
RUN_ID = "R3-THRESHOLD-METRIC-ENGINE-FIX"
CARD_ID = "TASK-20260918-r3-card2-engine-three-body-moment-fix"
MASTER_SEED = 20260918
DRAWS = 0
ENGINE_SHA256 = (
    "ee240f44645d2e592acfae462f7d1e9c3b3dd8cd78c96bd40b7b73509f7fcc63"
)
SIGMA_TOLERANCE = 3.0
DETERMINISTIC_SE_FLOOR = 1e-12
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
# The pinned input digests (the sealed run's own inputs, byte-stable).
# ---------------------------------------------------------------------------
CATALOG_SHA256 = (
    "6e4de7372a5bbf99affd6fa3ccc295c4cef09404dab3b17acd8816cda762bdcd"
)
DRIVER_SHA256 = (
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
R31_PRE_REGISTRATION_SHA256 = (
    "3f5ed7d8599aeaa53d38de897f731125fb5d50e0e9f8d546058ea3ae9bbc1a14"
)
BASELINE_REPORT_SHA256 = (
    "e9609d9d1819148e818651723e37006ef77a53aa8a75c4eb3f4941c592eb0267"
)

# ---------------------------------------------------------------------------
# The pinned sealed-file digests (the sha256 pin set of this module).
# ---------------------------------------------------------------------------
PRE_REGISTRATION_SHA256 = (
    "08e4014b7545de2c07f86b7062ff4fdcdb059c9ceef7bc517b29b8b2d13b857e"
)
PROVENANCE_SHA256 = (
    "9b6b2d88e391fef9752267e02ce4e1f1861397b5ce2d1db629367f92b7606416"
)
LIVE_R21_SHA256 = (
    "56f2ecf64bfb497c6dd67ceb77b290cf039aafbc059b047dfc4f46ef4b1e901f"
)
LIVE_R22_SHA256 = (
    "15213086e6203a4e623a32160fe9954f29ddca6b55c48dbf1e19e48512394f7c"
)
LIVE_R31_SHA256 = (
    "e7170a390b0455badc559d51807d3063a48b1ba0b230434b0d513121b04d029b"
)
REPORT_SHA256 = (
    "c45e4c9d28ba7c9986fce61b69f140ff016506d7c52afa7aa654d2e4eb3e6a88"
)
SEAL_MANIFEST_SHA256 = (
    "11e4118269debbaab09023fa717a31a31ded20f792d1e33122b40a82ed98b334"
)
SUMS_FILE_SHA256 = (
    "6b7506cc8a98a0a2e6458b8ba54bcf241b755cd8559e7f45fce4ee79a7ae89ba"
)
CLEAN_ENVIRONMENT_SHA256 = (
    "d3b153bec9483caf7d42e31a256dcd918864d493c1a81985dbc8b4e2a3dba63e"
)

# ---------------------------------------------------------------------------
# The pinned sealed verdicts (per the task card, verified bit-exact
# against the on-disk artifacts).
# ---------------------------------------------------------------------------

#: R21 live re-evaluation (the fixed engine against the sealed
#: delta_observed).
R21_APPLIED = 351
R21_PASS = 342
R21_FAIL = 9
R21_ORIGINALS_NOW_PASSING = 160
R21_PASS_RATE = 0.9467455621301775
R21_MAX_ABS_SIGMA = 3.5992208640500682
R21_MAX_CHECK = "pair(3,4) pi_0 mean_pt_y"

#: R22 live re-evaluation.
R22_APPLIED = 483
R22_PASS = 480
R22_FAIL = 3
R22_ORIGINALS_NOW_PASSING = 151
R22_PASS_RATE = 0.9805194805194806
R22_MAX_ABS_SIGMA = 4.973562263179045
R22_MAX_CHECK = "pair(23,24) pi_0 mean_pt_y"

#: The R21/R22 combined block of the revised clause report.
COMBINED_APPLIED = 834
COMBINED_PASS = 822
COMBINED_FAIL = 12
COMBINED_ORIGINALS_NOW_PASSING = 311
COMBINED_PASS_RATE = 0.9628482972136223
COMBINED_MAX_ABS_SIGMA = 4.973562263179045
COMBINED_MAX_CHECK = "R22 pair(23,24) pi_0 mean_pt_y"

#: The named residual failures of the live re-evaluation (pair,
#: species, observable, sigma), in payload order.
R21_RESIDUALS = [
    ([0, 1], "proton", "mean_p", 3.5987080275610572),
    ([0, 1], "proton", "mean_pt_y", 3.445790227925448),
    ([1, 2], "pi_0", "mean_pt_y", 3.362272164659226),
    ([3, 4], "pi_0", "mean_pt_y", 3.5992208640500682),
    ([4, 5], "pi_minus", "mean_pt_y", 3.4881623612596617),
    ([5, 6], "pi_0", "mean_pt_y", 3.316931264858482),
    ([14, 15], "pi_plus", "mean_p", 3.077997188838093),
    ([15, 16], "pi_0", "mean_p", 3.0407611711227656),
    ([18, 19], "pi_0", "multiplicity", 3.014046219928031),
]
R22_RESIDUALS = [
    ([7, 8], "pi_minus", "mean_pt_y", 3.8763921987053918),
    ([15, 16], "pi_plus", "mean_pt_y", 3.110100410337241),
    ([23, 24], "pi_0", "mean_pt_y", 4.973562263179045),
]

#: The four sealed moment-approximation-floor residual checks (the T2
#: clause): sealed rereval sigma -> live sigma; all four still fail.
FLOOR_RESIDUAL_PIN = [
    ("R21", [1, 2], "pi_0", "mean_pt_y",
     3.3622721646606926, 3.362272164659226, False),
    ("R21", [3, 4], "pi_0", "mean_pt_y",
     3.594992035740298, 3.5992208640500682, False),
    ("R21", [15, 16], "pi_0", "mean_p",
     3.0846781101915357, 3.0407611711227656, False),
    ("R22", [23, 24], "pi_0", "mean_pt_y",
     4.984844257802041, 4.973562263179045, False),
]

#: The fixed engine against the sealed stored values (the maximum
#: deviation per observable; identical to the sealed card-2 crosscheck
#: payloads -- the P1 engine reproduces the card-2-era engine
#: bit-for-bit at these maxima).
R21_ENGINE_VS_STORED = {
    "multiplicity": 1.0470213190894029e-07,
    "mean_p": 0.0010346239650864675,
    "mean_pt_y": 0.00115295674763502,
}
R22_ENGINE_VS_STORED = {
    "multiplicity": 8.520139052170528e-06,
    "mean_p": 0.00031546392915315913,
    "mean_pt_y": 0.0005148188409500876,
}

#: R31 targeted re-evaluation.
R31_CHECKS_APPLIED = 422
R31_PASS = 299
R31_VACUOUS_ZERO_LIVE = 113
R31_FAIL = 10
R31_ABSENT_CARRIED = 226
R31_MAX_ABS_SIGMA = 7.01270490790188
R31_MAX_CHECK = "w2 pair(1,2) pi_0 mean_p"
R31_BOUNDARY_APPLIED = 104
R31_BOUNDARY_FAILED_LIVE = 0
R31_INTERIOR_APPLIED = 318
R31_INTERIOR_FAILED_LIVE = 10

#: The 14 targeted R31 failures, old sealed sigma -> new sigma under
#: the fixed engine (window, pair, species, observable, old, new,
#: pass).  The two Lambda(1520) three-body checks and the two
#: float-precision SE artifacts pass; the ten moment-approximation-floor
#: checks persist at their sealed sigmas.
R31_ORIGINAL_FAILURES_PIN = [
    (0, [0, 1], "proton", "mean_pt_y",
     3.2123466532077045, 3.2123466532077045, False),
    (0, [0, 1], "pi_minus", "mean_pt_y",
     3.3254504430919427, 3.3254504430919427, False),
    (0, [3, 4], "neutron", "mean_p",
     3.151740704856282, 3.151740704856282, False),
    (0, [3, 4], "K_0", "mean_p",
     25.98076211353316, 0.0, True),
    (0, [3, 4], "Sigma_plus", "mean_p",
     25.98076211353316, 0.0, True),
    (1, [2, 3], "pi_plus", "mean_pt_y",
     7.225811435009375, 0.9201361176029509, True),
    (1, [3, 4], "pi_plus", "mean_pt_y",
     -5.610935929060075, 0.36971803795576264, True),
    (2, [0, 1], "pi_0", "mean_pt_y",
     -3.0280248998333104, 3.0280248998333104, False),
    (2, [0, 1], "K0L", "mean_p",
     3.064256426739259, 3.064256426739259, False),
    (2, [0, 1], "K0L", "mean_pt_y",
     3.069041367744035, 3.069041367744035, False),
    (2, [1, 2], "pi_0", "mean_p",
     7.01270490790188, 7.01270490790188, False),
    (2, [3, 4], "K0S", "mean_pt_y",
     3.4011873510462327, 3.4011873510462327, False),
    (2, [3, 4], "K0L", "mean_p",
     3.7365731963210447, 3.7365731963210447, False),
    (2, [3, 4], "K0L", "mean_pt_y",
     4.547781556513862, 4.547781556513862, False),
]

N_SEAL_FILES = 6
SEAL_TOTAL_BYTES = 1_901_206

SUMS_PREFIX = f"dev/00runs/{RUN_NAME}/"
TOP_LEVEL_FILES = (
    "pre_registration.json",
    "provenance.json",
    "live_reeval_r21.json",
    "live_reeval_r22.json",
    "live_reeval_r31_targeted.json",
    "gate2_revised_threshold_v2.json",
    "clean_environment_commands.json",
    "seal_manifest.json",
    "SHA256SUMS.txt",
)
SUMS_PAYLOADS = (
    "gate2_revised_threshold_v2.json",
    "live_reeval_r21.json",
    "live_reeval_r22.json",
    "live_reeval_r31_targeted.json",
    "pre_registration.json",
    "provenance.json",
)
SUMS_PAYLOAD_SHA256 = {
    "gate2_revised_threshold_v2.json": REPORT_SHA256,
    "live_reeval_r21.json": LIVE_R21_SHA256,
    "live_reeval_r22.json": LIVE_R22_SHA256,
    "live_reeval_r31_targeted.json": LIVE_R31_SHA256,
    "pre_registration.json": PRE_REGISTRATION_SHA256,
    "provenance.json": PROVENANCE_SHA256,
}

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


@pytest.fixture(scope="module")
def run_dir():
    """The sealed engine-fix run dir, skipped with a clear message when
    the gitignored artifact is absent or incomplete (a fresh
    checkout)."""
    if not (RUN_DIR / "pre_registration.json").is_file():
        pytest.skip(
            f"the sealed engine-fix run dir {RUN_DIR} is absent "
            "(gitignored artifact); restore the sealed run to execute "
            "the artifact pins"
        )
    missing = [name for name in TOP_LEVEL_FILES
               if not (RUN_DIR / name).is_file()]
    if missing:
        pytest.skip(
            f"the sealed engine-fix run dir {RUN_DIR} is incomplete "
            "(missing: " + ", ".join(missing) + "); restore the sealed "
            "run to execute the artifact pins"
        )
    return RUN_DIR


# ---------------------------------------------------------------------------
# Sealed artifact pins
# ---------------------------------------------------------------------------
def test_sealed_run_dir_inventory(run_dir):
    """The run dir holds exactly the nine top-level files (the six
    sealed payloads, the seal manifest, the sums file, the post-seal
    clean-environment record); nothing else is present."""
    on_disk = {path.name for path in run_dir.iterdir()}
    assert on_disk == set(TOP_LEVEL_FILES), (
        "the sealed run dir drifted from the committed 9-file top "
        "level set"
    )


def test_pre_registration_structure(run_dir):
    """The sealed pre-registration carries the committed contract: the
    three engine mechanisms by name and rule, the deterministic-SE
    floor rule (value 1e-12, the vacuous-zero interaction, the
    unchanged 3.0-sigma tolerance), master seed 20260918 with
    ``draws: 0``, the 834-applied-check R21/R22 scope, the
    422-applied-check targeted R31 scope over the stopped card-8
    artifacts at the original statistics, the sealed card-2 baseline
    numbers, the T1 and T2 acceptance thresholds, ``no_fitting``,
    ``written_before_evaluation``, and the nine output files."""
    pre = _load("pre_registration.json")
    assert pre["schema"] == PRE_SCHEMA
    assert pre["run"] == RUN_ID
    assert pre["card"] == CARD_ID
    assert pre["phase"] == 2
    assert pre["written_before_evaluation"] is True
    assert pre["master_seed"] == MASTER_SEED
    assert pre["draws"] == DRAWS

    engine = pre["engine"]
    assert engine["module"] == "na49_mixing.threshold_metric"
    assert engine["module_sha256_at_registration"] == ENGINE_SHA256
    assert engine["sha256"] == ENGINE_SHA256
    assert engine["phase_1_commit"] == "95096d2"
    mechanisms = engine["mechanisms"]
    assert set(mechanisms) == {
        "a_exact_kallen_three_body_slots",
        "b_deep_level_daughter_mass_integration",
        "c_deterministic_se_floor",
    }
    assert "Kallen" in mechanisms["a_exact_kallen_three_body_slots"]
    assert "budget window" in mechanisms[
        "b_deep_level_daughter_mass_integration"]
    assert "no current channel changes" in mechanisms[
        "b_deep_level_daughter_mass_integration"]

    metric = pre["metric"]
    assert metric["id"] == "delta_analytic_full"
    assert metric["tolerance_sigma"] == SIGMA_TOLERANCE
    assert metric["tolerance_unchanged"] is True
    floor = metric["deterministic_se_floor"]
    assert floor["value"] == DETERMINISTIC_SE_FLOOR
    assert floor["vacuous_zero_threshold"] == 1e-12
    assert floor["tolerance_unchanged"] is True

    coverage = pre["coverage"]
    assert coverage["r21"]["applied_checks_original"] == N_APPLIED_R21
    assert coverage["r21"]["failed_original"] == N_ORIGINAL_FAILURES_R21
    assert coverage["r22"]["applied_checks_original"] == N_APPLIED_R22
    assert coverage["r22"]["failed_original"] == N_ORIGINAL_FAILURES_R22
    assert coverage["r31"]["applied_checks"] == N_R31_APPLIED
    assert coverage["r31"]["failed_original"] == N_R31_ORIGINAL_FAILURES
    assert coverage["applied_total_r21_r22"] == N_APPLIED_R21R22
    assert coverage["total_original_failures_r21_r22"] == (
        N_ORIGINAL_FAILURES_R21R22)
    assert coverage["r31"]["adjacent_pairs"] == 12

    r31_block = pre["r31_targeted"]
    assert r31_block["run"] == "R31"
    assert r31_block["card"] == (
        "TASK-20260917-r3-run-r31-catalog-continuum-matching-scan")
    assert r31_block["read_only"] is True
    assert "100,000-decay-per-node" in r31_block["statistics_note"]

    baseline = pre["baseline"]
    assert baseline["sealed_residual_failures"] == (
        SEALED_RESIDUAL_FAILURES)
    assert baseline["pass_rate_among_original_failures"] == (
        SEALED_PASS_RATE)

    acceptance = pre["acceptance"]
    assert acceptance["t1_targeted_r31"]["n_original_failures"] == (
        N_R31_ORIGINAL_FAILURES)
    assert acceptance["t1_targeted_r31"]["required"] == "n_fail_live == 0"
    floor_checks = acceptance["t2_card2_reeval"]["floor_residual_checks"]
    assert [row["pair"] for row in floor_checks] == [
        [1, 2], [3, 4], [15, 16], [23, 24]]
    assert acceptance["t2_card2_reeval"]["n_original_failures"] == (
        N_ORIGINAL_FAILURES_R21R22)

    assert pre["constraints"]["no_fitting"] is True
    assert pre["constraints"]["tolerance_relaxed"] is False
    assert pre["constraints"]["no_pair_special_casing"] is True
    assert pre["outputs"] == list(TOP_LEVEL_FILES)

    # the sealed pre-registration is byte-pinned
    assert _file_sha256(run_dir / "pre_registration.json") == (
        PRE_REGISTRATION_SHA256)


def test_provenance_structure(run_dir):
    """The provenance records the fixed engine sha256 (the phase-1
    digest, cross-checked against the on-disk module), the catalog
    sha256, the sealed R21/R22 gate digests and seal identities, the
    stopped R31 artifacts by digest, the sealed card-2 baseline report
    digest, the dual-tree driver sha256 (the nested and monorepo
    copies are byte-identical), the git identity, the environment
    fingerprint, and the master seed with zero draws."""
    prov = _load("provenance.json")
    assert prov["schema"] == PROV_SCHEMA
    assert prov["run"] == RUN_ID
    assert prov["card"] == CARD_ID
    assert prov["phase"] == 2
    assert UTC_STAMP.match(prov["written_utc"])
    assert prov["master_seed"] == MASTER_SEED
    assert prov["draws"] == DRAWS

    engine = prov["engine"]
    assert engine["module_sha256"] == ENGINE_SHA256
    assert engine["phase_1_commit"] == "95096d2"
    # the recorded fixed-engine digest matches the on-disk module
    assert _file_sha256(METRIC_MODULE_PATH) == ENGINE_SHA256

    assert prov["catalog"]["sha256"] == CATALOG_SHA256
    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    archives = prov["sealed_archives"]
    assert archives["r21"]["gate_sha256"] == R21_GATE_SHA256
    assert archives["r22"]["gate_sha256"] == R22_GATE_SHA256
    assert archives["r21"]["seal"]["files_checked"] == 103
    assert archives["r22"]["seal"]["files_checked"] == 143

    r31_block = prov["r31_artifacts"]
    assert r31_block["verification_sha256"] == R31_VERIFICATION_SHA256
    # The on-disk R31 matching verification is the card-8b resumed report
    # (see the R31_RESUMED_VERIFICATION_SHA256 note above); the stopped
    # digest recorded in this provenance is the sealed stop-state record.
    assert _file_sha256(
        R31_DIR / "matching_verification.json") == (
        R31_RESUMED_VERIFICATION_SHA256)
    assert r31_block["pre_registration_sha256"] == (
        R31_PRE_REGISTRATION_SHA256)
    assert r31_block["seal"] is None
    assert "carries no seal and is verified here by digest" in (
        r31_block["seal_note"])

    assert prov["baseline"]["report_sha256"] == BASELINE_REPORT_SHA256
    assert _file_sha256(
        BASELINE_DIR / "gate2_revised_threshold.json") == (
        BASELINE_REPORT_SHA256)

    assert prov["drivers"]["rereval_driver_sha256"] == DRIVER_SHA256
    # dual-tree driver: the nested and monorepo copies are
    # byte-identical and carry the pinned digest
    assert _file_sha256(DRIVER_PATH) == DRIVER_SHA256
    assert _file_sha256(DRIVER_OUTER_PATH) == DRIVER_SHA256

    for key in ("nested_head", "nested_dirty_entries",
                "outer_head", "outer_dirty_entries"):
        assert key in prov["git"]
    for key in ("python_version", "python_implementation", "platform",
                "machine", "numpy_version", "executable"):
        assert key in prov["environment"]

    assert _file_sha256(run_dir / "provenance.json") == PROVENANCE_SHA256


def test_live_reeval_r21_r22_structure(run_dir):
    """The two live re-evaluation payloads carry the fixed-engine
    evaluation of the 834 applied checks (R21 351, R22 483) at the
    unchanged 3.0-sigma tolerance: R21 342 pass / 9 fail with
    160 of 169 original failures now passing, R22 480 pass / 3 fail
    with 151 of 154; the maximum |sigma| values, the named residual
    rows, the engine-vs-sealed-stored maxima (identical to the sealed
    card-2 crosscheck values), and the pinned digests."""
    r21 = _load("live_reeval_r21.json")
    r22 = _load("live_reeval_r22.json")
    for payload, applied, n_pass, n_fail, originals, rate, max_sigma, \
            max_check, residuals, engine_vs_stored, sums_digest in (
        (r21, R21_APPLIED, R21_PASS, R21_FAIL,
         R21_ORIGINALS_NOW_PASSING, R21_PASS_RATE, R21_MAX_ABS_SIGMA,
         R21_MAX_CHECK, R21_RESIDUALS, R21_ENGINE_VS_STORED,
         LIVE_R21_SHA256),
        (r22, R22_APPLIED, R22_PASS, R22_FAIL,
         R22_ORIGINALS_NOW_PASSING, R22_PASS_RATE, R22_MAX_ABS_SIGMA,
         R22_MAX_CHECK, R22_RESIDUALS, R22_ENGINE_VS_STORED,
         LIVE_R22_SHA256),
    ):
        assert payload["schema"] == REREVAL_SCHEMA
        assert payload["card"] == CARD_ID
        assert payload["phase"] == 2
        assert payload["metric"]["tolerance_sigma"] == SIGMA_TOLERANCE
        assert payload["metric"]["tolerance_unchanged"] is True
        assert payload["engine"]["module_sha256"] == ENGINE_SHA256
        summary = payload["summary"]
        assert summary["n_checks_applied"] == applied
        assert summary["n_pass"] == n_pass
        assert summary["n_fail"] == n_fail
        assert summary["n_vacuous_zero_live"] == 0
        assert summary["n_not_evaluable_live"] == 0
        reeval = summary["reeval"]
        assert reeval["original_failures_now_passing"] == originals
        assert reeval["pass_rate_among_original_failures"] == rate
        assert summary["max_abs_sigma"] == max_sigma
        assert summary["max_abs_sigma_check"] == max_check
        assert summary["engine_vs_sealed_stored"] == engine_vs_stored
        # the named residual rows, in payload order
        got = [(row["pair"], row["species"], row["observable"],
                row["sigma"]) for row in summary["residuals"]]
        assert got == residuals
        # every residual carries a one-line diagnosis
        for row in summary["residuals"]:
            assert isinstance(row["diagnosis"], str)
            assert row["diagnosis"]
        # the applied-check rows carry the live fields
        applied_rows = [row for row in payload["checks"]
                        if row["scope"] == "applied"]
        assert len(applied_rows) == applied
        assert all(row["status"] in ("pass", "fail")
                   for row in applied_rows)
        # the sealed vacuous_zero/absent_species rows are carried,
        # never re-verdicted
        carried = [row for row in payload["checks"]
                   if row["scope"] == "not_applied"]
        assert all(row["sigma"] is None for row in carried)
        assert _file_sha256(run_dir / (
            f"live_reeval_{payload['run'].lower()}.json")) == sums_digest
        assert _canonical_sha256(payload) == sums_digest


def test_floor_residuals_still_fail(run_dir):
    """The four sealed moment-approximation-floor residuals (the T2
    clause) still fail under the fixed engine, with the pinned live
    sigmas against the sealed rereval sigmas."""
    report = _load("gate2_revised_threshold_v2.json")
    rows = report["acceptance"]["t2_card2_reeval"]["floor_residuals"][
        "checks"]
    got = [(row["run"], row["pair"], row["species"], row["observable"],
            row["sealed_rereval_sigma"], row["live_sigma"], row["pass"])
           for row in rows]
    assert got == FLOOR_RESIDUAL_PIN
    assert report["acceptance"]["t2_card2_reeval"]["floor_residuals"][
        "pass"] is False


def test_live_reeval_r31_targeted_structure(run_dir):
    """The targeted R31 payload re-evaluates all 422 applied checks of
    the stopped matching verification at the original statistics: 299
    explicit passes, 113 vacuous-zero passes under the deterministic-SE
    floor, 10 failures; 4 of the 14 original failures pass (the two
    Lambda(1520) three-body checks and the two float-precision SE
    artifacts) and 10 persist at their sealed sigmas; the 14-row table
    is pinned bit-exact."""
    payload = _load("live_reeval_r31_targeted.json")
    assert payload["schema"] == REREVAL_SCHEMA
    assert payload["card"] == CARD_ID
    assert payload["phase"] == 2
    assert payload["run"] == "R31"
    assert payload["metric"]["tolerance_sigma"] == SIGMA_TOLERANCE
    assert payload["engine"]["module_sha256"] == ENGINE_SHA256
    assert payload["archive"]["verification_sha256"] == (
        R31_VERIFICATION_SHA256)
    assert payload["archive"]["seal"] is None
    summary = payload["summary"]
    assert summary["n_checks_applied"] == R31_CHECKS_APPLIED
    assert summary["n_pass"] == R31_PASS
    assert summary["n_vacuous_zero_live"] == R31_VACUOUS_ZERO_LIVE
    assert summary["n_fail"] == R31_FAIL
    assert summary["n_absent_carried"] == R31_ABSENT_CARRIED
    assert summary["n_not_evaluable_live"] == 0
    assert summary["max_abs_sigma"] == R31_MAX_ABS_SIGMA
    assert summary["max_abs_sigma_check"] == R31_MAX_CHECK
    assert summary["boundary_interior"] == {
        "boundary_pair_applied": R31_BOUNDARY_APPLIED,
        "boundary_pair_failed_live": R31_BOUNDARY_FAILED_LIVE,
        "interior_pairs_applied": R31_INTERIOR_APPLIED,
        "interior_pairs_failed_live": R31_INTERIOR_FAILED_LIVE,
    }
    assert summary["original_failures_passing"] == 4
    assert summary["original_failures_persisting"] == 10
    # the 14-row table, in payload order
    got = [(row["window"], row["pair"], row["species"],
            row["observable"], row["sigma_original"], row["sigma"],
            row["pass"]) for row in summary["original_failures"]]
    assert got == R31_ORIGINAL_FAILURES_PIN
    for row in summary["residuals"]:
        assert isinstance(row["diagnosis"], str)
        assert row["diagnosis"]
    # every applied check row carries the live verdict
    assert len(payload["checks"]) == R31_CHECKS_APPLIED
    assert all(row["status"] in ("pass", "fail", "vacuous_zero")
               for row in payload["checks"])
    assert _file_sha256(
        run_dir / "live_reeval_r31_targeted.json") == LIVE_R31_SHA256
    assert _canonical_sha256(payload) == LIVE_R31_SHA256


def test_revised_threshold_v2_structure(run_dir):
    """The revised clause evidence carries the sealed card-2 baseline
    alongside the post-fix values and records the acceptance verdicts
    on the card's T1 and T2 criteria: T1 fails with 4 of 14 original
    failures passing, T2 fails on all three sub-criteria (12 failures
    against the sealed 11, the four floor residuals still failing, the
    pass rate 0.9628482972136223 below the sealed 0.9659442724458205);
    the tolerance block stays 3.0-sigma unchanged."""
    report = _load("gate2_revised_threshold_v2.json")
    assert report["schema"] == REPORT_SCHEMA
    assert report["card"] == CARD_ID
    assert report["phase"] == 2
    assert report["gate"] == "gate 2 (plan section 8, line 575)"
    assert report["clause"] == "threshold_discontinuity"
    assert report["tolerance"]["sigma"] == SIGMA_TOLERANCE
    assert report["tolerance"]["unchanged"] is True
    assert report["tolerance"]["relaxed"] is False
    assert report["tolerance"]["deterministic_se_floor"] == (
        DETERMINISTIC_SE_FLOOR)

    baseline = report["sealed_baseline"]
    assert baseline["report_sha256"] == BASELINE_REPORT_SHA256
    assert baseline["combined"]["residual_failures"] == (
        SEALED_RESIDUAL_FAILURES)
    assert baseline["combined"]["pass_rate_among_original_failures"] == (
        SEALED_PASS_RATE)

    combined = report["combined"]
    assert combined["applied"] == COMBINED_APPLIED
    assert combined["pass"] == COMBINED_PASS
    assert combined["fail"] == COMBINED_FAIL
    assert combined["original_failures"] == N_ORIGINAL_FAILURES_R21R22
    assert combined["original_failures_now_passing"] == (
        COMBINED_ORIGINALS_NOW_PASSING)
    assert combined["pass_rate_among_original_failures"] == (
        COMBINED_PASS_RATE)
    assert combined["max_abs_sigma"] == COMBINED_MAX_ABS_SIGMA
    assert combined["max_abs_sigma_check"] == COMBINED_MAX_CHECK

    t1 = report["acceptance"]["t1_targeted_r31"]
    assert t1["n_original_failures"] == N_R31_ORIGINAL_FAILURES
    assert t1["original_failures_passing"] == 4
    assert t1["original_failures_persisting"] == 10
    assert t1["pass"] is False

    t2 = report["acceptance"]["t2_card2_reeval"]
    assert t2["failure_count"]["sealed"] == SEALED_RESIDUAL_FAILURES
    assert t2["failure_count"]["achieved"] == COMBINED_FAIL
    assert t2["failure_count"]["pass"] is False
    assert t2["floor_residuals"]["pass"] is False
    rate = t2["pass_rate_among_original_failures"]
    assert rate["sealed"] == SEALED_PASS_RATE
    assert rate["threshold"] == SEALED_PASS_RATE
    assert rate["achieved"] == COMBINED_PASS_RATE
    assert rate["pass"] is False
    assert t2["pass"] is False

    assert report["acceptance"]["pass"] is False
    assert report["no_fitting"] is True
    # every remaining residual is named with its sigma and a one-line
    # diagnosis (9 R21 + 3 R22 = 12)
    findings = report["residual_findings"]
    assert len(findings) == COMBINED_FAIL
    for finding in findings:
        assert isinstance(finding["diagnosis"], str)
        assert finding["diagnosis"]

    # the sealed payload is byte-pinned and in canonical form
    assert _file_sha256(
        run_dir / "gate2_revised_threshold_v2.json") == REPORT_SHA256
    assert _canonical_sha256(report) == REPORT_SHA256


def test_seal_manifest_and_sha256_sums(run_dir):
    """``seal_manifest.json`` records the 6 sealed payloads totaling
    1,901,205 bytes, and ``SHA256SUMS.txt`` is in ``sha256sum -c``
    format with monorepo-root-relative paths (LF, two spaces, relative
    path), carries the 6 pinned payload digests, verifies by recomputed
    SHA-256 against every payload file on disk, and self-consistently
    accounts for the full byte total."""
    manifest = _load("seal_manifest.json")
    assert manifest["schema"] == SEAL_SCHEMA
    assert manifest["run"] == RUN_ID
    assert manifest["card"] == CARD_ID
    assert UTC_STAMP.match(manifest["created_utc"])
    assert manifest["file_count"] == N_SEAL_FILES
    assert manifest["total_bytes"] == SEAL_TOTAL_BYTES
    assert manifest["sha256_sums_file"] == "SHA256SUMS.txt"
    assert manifest["excluded_from_sums"] == [
        "seal_manifest.json", "SHA256SUMS.txt"]
    assert "monorepo-root-relative" in manifest["path_convention"]

    # the sums-file digest recorded in the manifest matches on disk
    sums_path = run_dir / "SHA256SUMS.txt"
    assert manifest["sha256_sums_sha256"] == _file_sha256(sums_path)
    assert _file_sha256(sums_path) == SUMS_FILE_SHA256

    raw = sums_path.read_bytes()
    assert raw.endswith(b"\n")
    assert b"\r" not in raw
    lines = raw.decode("utf-8").splitlines()
    assert len(lines) == N_SEAL_FILES
    entries = []
    for line in lines:
        assert SUMS_LINE.match(line), (
            "SHA256SUMS.txt is not in sha256sum -c format (LF, two "
            f"spaces, relative path): {line!r}"
        )
        digest, rel = line.split("  ", 1)
        # monorepo-root-relative entries (the R21/R22 production
        # archive convention; sha256sum -c verifies from the repo root)
        assert rel.startswith(f"dev/00runs/{RUN_NAME}/"), rel
        entries.append((digest, rel))
    rels = [rel for _digest, rel in entries]
    assert rels == sorted(rels), (
        "the sums file is not in sorted relative-path order")
    assert [rel.rsplit("/", 1)[-1] for rel in rels] == list(
        SUMS_PAYLOADS)
    for digest, rel in entries:
        assert digest == SUMS_PAYLOAD_SHA256[rel.rsplit("/", 1)[-1]], (
            f"{rel}: the sums-carrying digest drifted from the pinned "
            "digest"
        )
        # verifies by recomputed sha256 against the file on disk
        assert _file_sha256(MONO_ROOT / rel) == digest, (
            f"{rel}: the sealed digest does not match the file on "
            "disk; the run dir was modified after the seal"
        )
    # self-consistent byte total over the sums-covered payloads
    assert sum((MONO_ROOT / rel).stat().st_size
               for _digest, rel in entries) == SEAL_TOTAL_BYTES

    assert _file_sha256(run_dir / "seal_manifest.json") == (
        SEAL_MANIFEST_SHA256)


def test_clean_environment_record(run_dir):
    """The post-seal clean-environment record re-verified the 6
    sums-covered files without mismatches, ran the clean interpreter
    (``python -I``) with return code 0, and the re-run reproduced the
    sealed live re-eval and revised-clause hashes (the recorded sealed
    and recomputed digests match and equal the digests of the files on
    disk)."""
    record = _load("clean_environment_commands.json")
    assert record["schema"] == CLEAN_SCHEMA
    assert record["run"] == RUN_ID
    assert record["card"] == CARD_ID
    assert UTC_STAMP.match(record["verified_utc"])
    assert record["seal_verification"] == {
        "files_checked": N_SEAL_FILES,
        "mismatches": [],
    }
    assert record["hashes_match"] is True

    commands = record["commands"]
    assert len(commands) == 1
    command = commands[0]
    argv = command["argv"]
    # the clean interpreter: python -I, the driver by path (one of the
    # two byte-identical dual-tree copies, digest-pinned), the run dir,
    # and the sealed baseline report
    assert argv[1] == "-I"
    driver_in_argv = next(
        (part for part in argv
         if str(part).endswith("rereval_threshold_checks.py")), None)
    assert driver_in_argv is not None
    assert _file_sha256(Path(driver_in_argv)) == DRIVER_SHA256
    assert str(run_dir) in argv
    assert str(BASELINE_DIR / "gate2_revised_threshold.json") in argv
    assert "cwd" in command

    for key in ("python_version", "python_implementation", "platform",
                "machine", "numpy_version", "executable"):
        assert key in record["environment"]

    sealed = record["sealed_report_sha256"]
    recomputed = record["recomputed_report_sha256"]
    assert sealed == recomputed
    assert sealed == {
        "live_reeval_r21.json": LIVE_R21_SHA256,
        "live_reeval_r22.json": LIVE_R22_SHA256,
        "live_reeval_r31_targeted.json": LIVE_R31_SHA256,
        "gate2_revised_threshold_v2.json": REPORT_SHA256,
    }
    for name, digest in sealed.items():
        assert _file_sha256(run_dir / name) == digest, (
            f"{name}: the on-disk sealed report drifted from the "
            "clean-environment record"
        )

    assert _file_sha256(
        run_dir / "clean_environment_commands.json") == (
        CLEAN_ENVIRONMENT_SHA256)
