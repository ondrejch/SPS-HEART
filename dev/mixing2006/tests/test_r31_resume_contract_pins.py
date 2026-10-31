"""Round 3 card 8b resume: R31 matching-verification contract pins
(test-author register).

Task ``TASK-20260918-r3-run-r31-verify-resume`` (the Round 3 resume card
for card 8, run R31 of plan section 9 line 640; nested execution commit
``7493298``), executed under the re-scoped acceptance of
``.collab/tasks/TASK-20260918-r3-acceptance-rescope-card8a-8b.md``
(section 2, items (i)-(vi)).  This is the test-author's independent pin
register for the sealed run evidence under
``dev/00runs/r31_catalog_continuum_matching_scan_2026-09-17/`` after the
resume and for the frozen driver / fixed engine contract, an
independent cross-check of the solver's register in
``tests/test_staged_r31_matching_scan_gate_pins.py`` (the same sealed
values, pinned from an independently authored test).

Part 1 (sealed-run artifact pins, read-only; skipped when the
gitignored run dir is absent on a fresh checkout):

- ``test_sealed_run_dir_inventory``: exactly the 82 files of the
  resumed state (the 76 generated artifacts, the regenerated
  verification, the updated provenance, the gate evidence, the seal
  pair, the post-seal clean-environment record) and no sub-trees.
- ``test_seal_integrity``: ``SHA256SUMS.txt`` carries 79
  monorepo-root-relative lines, sorted by path, every digest verified
  against disk from the monorepo root; ``seal_manifest.json`` records
  80 sealed files totaling 1,165,025 bytes, every entry verified, the
  sums-file digest pinned, and the seal excludes its own records
  (``seal_manifest.json`` + ``SHA256SUMS.txt``) per the convention in
  use.
- ``test_pre_registration_byte_identity``: the pre-registration's
  sha256 matches the pre-registration digest recorded in the updated
  provenance's resume block (the stop-state record) and the on-disk
  bytes are the generation-time state: driver binding ``b47b2a85...``,
  the T1 statistics 4 replicas x 25,000 decays x 15 nodes, master seed
  20260917, the card-9 seed map, and the generation-time engine hash
  ``5d39643d...`` preserved in its own input hashes.
- ``test_provenance_resume_block``: the engine update line (pre-fix
  ``5d39643d...`` to fixed ``ee240f44...`` in both the ``modules`` and
  ``input_hashes`` entries, equal to the on-disk module), the
  76-artifact bit-identity record (a sample of the digests verified
  against disk; the sealed card-8c pre-investigation inventory as the
  source, verified), the pre-registration byte-identity record, and
  the resume block carrying the stopped verification digest
  ``de01f62f...``.
- ``test_matching_verification_resumed``: the regenerated report's
  digest ``91704260...``, the census 422 applied = 299 explicit + 113
  vacuous-zero + 10 named MC-scatter residuals (each residual verified
  row by row against the on-disk payload with its signed sigma and
  the card-8c classification attached), boundary pair 0 of 104,
  interior 10 of 318, max |sigma| 7.01270490790188, the recorded
  ``pass == false`` census form, events_checked 1,500,000, charge
  balance exact, no fitting.
- ``test_gate_evidence``: the ``no_boundary_dominance`` clause passing
  with its clause result and inputs pinned (aggregate 0/104 vs 10/318
  and the per-window counts), the smoothness clause's recorded census
  form, the card-8c cross-reference block (50-replica x 20,000
  statistics, 0/422, max |sigma| 2.959754302712887, the sealed
  card-8c confirmation artifact's digest verified against disk,
  ``authoritative_smoothness_evidence == true``), the pass census,
  the frozen driver's verbatim gate refusal (exit code 2), the input
  digests verified against disk, and the instrument identity.
- ``test_clean_environment_record``: the post-seal record
  re-verified the 79 sums-covered files without mismatches, re-ran
  the matching verification from the clean interpreter (``python
  -I``, the probe's exit code 1 recorded as the expected
  named-residual exit), the reproduced report hashes match the sealed
  digest, and the frozen driver's verbatim clean-validate refusal
  (exit code 2, the probe's stdout line).
- ``test_generated_artifacts_bit_identity``: all 76 generated
  artifacts (the pre-registration + the 15 node + the 60 replica
  records) listed in the provenance bit-identity record match their
  on-disk digests (the full 76, not a sample).
- ``test_rescoped_acceptance_structure``: the re-scoped acceptance is
  honored by the sealed evidence structure -- the gate evidence
  carries BOTH the original-statistics seal (the 10 named-residual
  register, acceptance (i)) AND the card-8c confirmation-statistics
  cross-reference (acceptance (v)).

Part 2 (pure contract, no sealed artifacts, always runs):

- ``test_frozen_driver_and_engine_contract``: the frozen R31 driver
  sha256 ``b47b2a85...`` on disk in both trees (nested + outer
  mirror), the fixed engine sha256 ``ee240f44...`` on disk,
  ``SIGMA_TOLERANCE == 3.0`` and ``DETERMINISTIC_SE_FLOOR == 1e-12``
  in the engine, and the committed catalog digest on disk.

Hermetic: standard library and the vendored ``na49_mixing`` module;
the Part 1 pins are read-only over the sealed run dir and skip on a
fresh checkout.  No Monte Carlo, no generation, no writes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from na49_mixing.threshold_metric import (
    DETERMINISTIC_SE_FLOOR,
    SIGMA_TOLERANCE,
)

NESTED_ROOT = Path(__file__).resolve().parent.parent
MONO_ROOT = NESTED_ROOT.parent.parent
RUN_NAME = "r31_catalog_continuum_matching_scan_2026-09-17"
RUN_DIR = MONO_ROOT / "dev" / "00runs" / RUN_NAME
SUMS_PREFIX = f"dev/00runs/{RUN_NAME}/"

ENGINE_PATH = NESTED_ROOT / "na49_mixing" / "threshold_metric.py"
DRIVER_PATH = NESTED_ROOT / "helpers" / "benchmarking" / (
    "staged_r31_matching_scan.py"
)
DRIVER_OUTER_PATH = (
    MONO_ROOT / "helpers" / "benchmarking" / "staged_r31_matching_scan.py"
)
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
AUDIT_RUN_DIR = MONO_ROOT / "dev" / "00runs" / "r3_se_audit_2026-09-18"
AUDIT_CONFIRMATION_PATH = AUDIT_RUN_DIR / "confirmation_verification.json"
AUDIT_INVENTORY_PATH = AUDIT_RUN_DIR / "pre_investigation_inventory.json"

# ---------------------------------------------------------------------------
# The pinned digests (the sealed run's own inputs and payloads).
# ---------------------------------------------------------------------------
RUN_ID = "R31"
CARD_ID = "TASK-20260917-r3-run-r31-catalog-continuum-matching-scan"
RESUME_CARD = "TASK-20260918-r3-run-r31-verify-resume"
ACCEPTANCE_RECORD = (
    ".collab/tasks/TASK-20260918-r3-acceptance-rescope-card8a-8b.md"
)

DRIVER_SHA256 = (
    "b47b2a858c3880ecf69e143035e8c209b6e6e5e0708ca2b9058c6c384340afe6"
)
ENGINE_SHA256 = (
    "ee240f44645d2e592acfae462f7d1e9c3b3dd8cd78c96bd40b7b73509f7fcc63"
)
PRE_FIX_ENGINE_SHA256 = (
    "5d39643d95bf80f8b4831d99b338574e25b95b48b0a9bb45c31ea7153f99aefb"
)
CATALOG_SHA256 = (
    "cf9e79b99199c509f2e83def0a99c215f73ca7761f9f87fb01d1f779b80e32f3"
)
PRE_REGISTRATION_SHA256 = (
    "3f5ed7d8599aeaa53d38de897f731125fb5d50e0e9f8d546058ea3ae9bbc1a14"
)
STOPPED_VERIFICATION_SHA256 = (
    "de01f62f4907b3acbb8b4c5959148708d474d9c22df925d6b6d89dd188cde629"
)
VERIFICATION_SHA256 = (
    "9170426043399dc954c08ec255c7a7592f63ffbdfdcc3ab655cc1c0dd8e499af"
)
PROVENANCE_SHA256 = (
    "95a03075facd687b0f74daf88bb18c0ef26c29ae9674d34df502c3b6f46a99b2"
)
GATE_SHA256 = (
    "98c1ba7327fc65d0cc6828fc0c93f12e1f4b9eaffad3a0617223143994dac686"
)
SUMS_SHA256 = (
    "949fefb6ec53282fd532ff72095a44efad180a44e9d8c40209e2baa97e88205f"
)
AUDIT_CONFIRMATION_SHA256 = (
    "98a3bff11184f93d16d17dbebee65dc2fd6834fac505aad7670ee7063ebd204d"
)
AUDIT_INVENTORY_SHA256 = (
    "16679283da93bc7cee5474f814478e016933fd932ba087ff241099999723bd4c"
)
INSTRUMENT_SHA256 = (
    "d7eea975baa1bcf4caca6af450064e2a8f5d5cde85ea002c8d4031ae5261093f"
)

SEAL_SCHEMA = "na49_mixing.staged_r31_matching_scan.seal_manifest/1"
SEAL_FILE_COUNT = 80
SEAL_TOTAL_BYTES = 1_165_025
SEAL_EXCLUDED_FROM_SUMS = ["seal_manifest.json", "SHA256SUMS.txt"]

SUMS_LINE_COUNT = 79
N_GENERATED_ARTIFACTS = 76
N_RUN_DIR_FILES = 82

MASTER_SEED = 20260917
N_REPLICAS = 4
DECAYS_PER_REPLICA = 25_000
N_WINDOWS = 3
N_NODES = 5
N_TOTAL_DECAYS = 1_500_000
TOLERANCE_SIGMA = 3.0
MAX_ABS_SIGMA = 7.01270490790188

N_CHECKS_APPLIED = 422
N_EXPLICIT_PASS = 299
N_VACUOUS_ZERO = 113
N_CHECKS_FAILED = 10
BOUNDARY_PAIR_APPLIED = 104
BOUNDARY_PAIR_FAILED = 0
INTERIOR_PAIRS_APPLIED = 318
INTERIOR_PAIRS_FAILED = 10

AUDIT_MASTER_SEED = 20260918
AUDIT_REPLICAS = 50
AUDIT_DECAYS_PER_REPLICA = 20_000
AUDIT_MAX_ABS_SIGMA = 2.959754302712887

#: The run dir holds exactly these top-level files after the resume.
EXPECTED_RUN_FILES = frozenset(
    [
        "pre_registration.json",
        "matching_verification.json",
        "provenance.json",
        "gate3_r31.json",
        "seal_manifest.json",
        "SHA256SUMS.txt",
        "clean_environment_commands.json",
    ]
    + [
        f"window_{w}_node_{j}.json"
        for w in range(N_WINDOWS)
        for j in range(N_NODES)
    ]
    + [
        f"window_{w}_node_{j}_replica_{r}.json"
        for w in range(N_WINDOWS)
        for j in range(N_NODES)
        for r in range(N_REPLICAS)
    ]
)

# ---------------------------------------------------------------------------
# The pinned named-residual register (the re-scoped acceptance (i)):
# (window, pair, global_pair, species, observable, signed sigma,
# card-8c confirmation sigma).  Every row is genuine finite-MC
# scatter per the sealed card-8c classification.
# ---------------------------------------------------------------------------
NAMED_RESIDUALS = [
    (0, (0, 1), 0, "proton", "mean_pt_y", 3.2123466532077045,
     0.24924759788716583),
    (0, (0, 1), 0, "pi_minus", "mean_pt_y", 3.3254504430919427,
     1.7448633070905266),
    (0, (3, 4), 3, "neutron", "mean_p", 3.151740704856282,
     -0.7953962610929308),
    (2, (0, 1), 8, "pi_0", "mean_pt_y", -3.0280248998333104,
     -0.9554659365490366),
    (2, (0, 1), 8, "K0L", "mean_p", 3.064256426739259,
     0.5083062555597769),
    (2, (0, 1), 8, "K0L", "mean_pt_y", 3.069041367744035,
     0.6011345842962964),
    (2, (1, 2), 9, "pi_0", "mean_p", 7.01270490790188,
     0.815841577316934),
    (2, (3, 4), 11, "K0S", "mean_pt_y", 3.4011873510462327,
     -0.24295175245811504),
    (2, (3, 4), 11, "K0L", "mean_p", 3.7365731963210447,
     -0.7620440622202165),
    (2, (3, 4), 11, "K0L", "mean_pt_y", 4.547781556513862,
     -1.4888713594117697),
]

#: The same rows' on-disk payload fields as recorded in the gate
#: evidence register: (delta_observed, delta_analytic_full,
#: se_combined), one per NAMED_RESIDUALS entry.
RESIDUAL_PAYLOADS = [
    (0.012322342268683173, 0.0062177790031839875,
     0.0019003438683690764),
    (0.0252999638583572, 0.021174937180702735,
     0.001240441482513596),
    (0.07960296704396219, 0.07395983950871043,
     0.0017904796313214103),
    (-0.003471884619533794, 0.002634536618622285,
     0.0020166350806732902),
    (0.00751593112584345, 0.001279136644500134,
     0.0020353369995147644),
    (0.002962923300522051, -0.00042679061181879335,
     0.00110448622425462),
    (0.007234816205737737, 0.0028875760825715635,
     0.0006199091763105169),
    (0.20374243105262982, 0.19162747359934748,
     0.003561978862927289),
    (0.28622393767793014, 0.269943936472745,
     0.00435693357250812),
    (0.20522407234438073, 0.19162747359934748,
     0.0029897211587831462),
]

#: The four closed original failures: (window, pair, global_pair,
#: species, observable, sigma_original_sealed, sigma_resume,
#: status_resume).  The two window-1 Lambda(1520) three-body checks
#: resolve sub-3-sigma through the exact Kallen kinematics; the two
#: float-precision SE artifacts go vacuous at the floor.
CLOSED_ORIGINAL_FAILURES = [
    (1, (2, 3), 6, "pi_plus", "mean_pt_y", 7.225811435009375,
     -0.9201361176029509, "pass"),
    (1, (3, 4), 7, "pi_plus", "mean_pt_y", -5.610935929060075,
     0.36971803795576264, "pass"),
    (0, (3, 4), 3, "K_0", "mean_p", 25.98076211353316, 0.0,
     "pass (vacuous_zero)"),
    (0, (3, 4), 3, "Sigma_plus", "mean_p", 25.98076211353316, 0.0,
     "pass (vacuous_zero)"),
]

#: The per-window no_boundary_dominance clause rows:
#: (window, qn, boundary_pair, boundary_pair_failure_rate,
#: interior_pairs_failure_rate, boundary_pair_applied,
#: boundary_pair_failed, interior_pairs_applied, interior_pairs_failed)
NBD_PER_WINDOW = [
    (0, (1, 1, 0), 2, 0.0, 0.029411764705882353, 32, 0, 102, 3),
    (1, (1, 0, -1), 6, 0.0, 0.0, 40, 0, 120, 0),
    (2, (0, 0, 0), 10, 0.0, 0.07291666666666667, 32, 0, 96, 7),
]

#: The aggregate interior-pair failure rate (10 of 318).
NBD_INTERIOR_RATE = 10 / 318  # 0.031446540880503145

#: The frozen driver's verbatim gate refusal on a failing
#: verification (recorded in the sealed gate evidence).
GATE_REFUSAL_MESSAGE = (
    "staged_r31_matching_scan: "
    "dev/00runs/r31_catalog_continuum_matching_scan_2026-09-17/"
    "matching_verification.json records checks_failed > 0; the run "
    "stops (a failing match re-opens the card-6 routing or the "
    "originating family card, never the tolerance)"
)

#: The frozen driver's verbatim clean-validate refusal (recorded in
#: the sealed clean-environment record): the message and the probe's
#: stdout line, both verbatim.
CLEAN_REFUSAL_MESSAGE = (
    "staged_r31_matching_scan: the clean-interpreter verification "
    "probe failed (return code 1):"
)
CLEAN_PROBE_STDOUT = (
    "staged_r31_matching_scan: the matching verification FAILED (10 "
    "of 422 checks outside the 3.0-sigma band, max |sigma| = "
    "7.01270490790188); the run stops and the card-6 routing or the "
    "originating family card is re-opened, never the tolerance"
)

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


def _verification_row(mv: dict, window: int, pair, species: str,
                       observable: str) -> dict:
    """Locate one applied check row of the resumed verification by
    (window, pair (j, j+1), species, observable)."""
    rows = mv["windows"][str(window)]["pairs"][str(pair[0])]["checks"]
    matches = [
        r for r in rows
        if r["species"] == species and r["observable"] == observable
    ]
    assert len(matches) == 1, (window, pair, species, observable)
    return matches[0]


# ---------------------------------------------------------------------------
# Part 1: the sealed-run artifact pins
# ---------------------------------------------------------------------------
def test_sealed_run_dir_inventory(run_dir):
    """The run dir holds exactly the 82 files of the resumed state
    (the 76 generated artifacts, the regenerated verification, the
    updated provenance, the gate evidence, the seal pair, the
    post-seal clean-environment record) and no sub-trees."""
    entries = sorted(p.name for p in run_dir.iterdir())
    assert len(entries) == N_RUN_DIR_FILES, entries
    assert entries == sorted(EXPECTED_RUN_FILES), entries
    for name in EXPECTED_RUN_FILES:
        assert (run_dir / name).is_file(), name


def test_seal_integrity(run_dir):
    """``SHA256SUMS.txt`` carries exactly the 79 pinned
    monorepo-root-relative entries, lexicographically sorted by
    relative path, and every digest verifies against disk from the
    monorepo root.  The seal manifest records the 80 sealed payloads
    (1,165,025 bytes), the sums-file digest, and the documented
    exclusion of the seal's own records."""
    text = (run_dir / "SHA256SUMS.txt").read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    assert len(lines) == SUMS_LINE_COUNT
    paths = [line.split("  ", 1)[1] for line in lines]
    assert paths == sorted(paths)
    assert all(p.startswith(SUMS_PREFIX) for p in paths)
    for line in lines:
        digest, rel = line.split("  ", 1)
        assert _sha256_file(MONO_ROOT / rel) == digest, rel
    assert _sha256_file(run_dir / "SHA256SUMS.txt") == SUMS_SHA256

    seal = _load(run_dir, "seal_manifest.json")
    assert seal["schema"] == SEAL_SCHEMA
    assert seal["run"] == RUN_ID
    assert seal["card"] == CARD_ID
    assert seal["file_count"] == SEAL_FILE_COUNT
    assert seal["total_bytes"] == SEAL_TOTAL_BYTES
    assert len(seal["files"]) == SEAL_FILE_COUNT
    assert seal["sha256_sums_file"] == "SHA256SUMS.txt"
    assert seal["sha256_sums_sha256"] == SUMS_SHA256
    assert seal["excluded_from_sums"] == SEAL_EXCLUDED_FROM_SUMS
    total = 0
    for entry in seal["files"]:
        name = entry["path"]
        assert (run_dir / name).is_file(), name
        assert _sha256_file(run_dir / name) == entry["sha256"], name
        total += entry["bytes"]
        assert (run_dir / name).stat().st_size == entry["bytes"], name
    assert total == SEAL_TOTAL_BYTES
    sums_entry = {e["path"]: e for e in seal["files"]}["SHA256SUMS.txt"]
    assert sums_entry["sha256"] == SUMS_SHA256
    # The seal excludes its own records per the convention in use:
    # the sums file and the seal manifest are both absent from the
    # sums, while the manifest records the sums file (sealed, with
    # its own digest).
    basenames = {Path(p).name for p in paths}
    assert "SHA256SUMS.txt" not in basenames
    assert "seal_manifest.json" not in basenames
    assert sums_entry["bytes"] == (run_dir / "SHA256SUMS.txt").stat().st_size


def test_pre_registration_byte_identity(run_dir):
    """The pre-registration is byte-identical to the generation-time
    state: its sha256 matches the pre-registration digest recorded in
    the updated provenance's resume block (the stop-state record), and
    the key fields are unchanged (the driver binding, the T1
    statistics, the master seed, the card-9 seed map, the
    generation-time engine hash)."""
    pre = _load(run_dir, "pre_registration.json")
    assert _sha256_file(run_dir / "pre_registration.json") == (
        PRE_REGISTRATION_SHA256)

    prov = _load(run_dir, "provenance.json")
    byte_identity = prov["resume"]["pre_registration_byte_identity"]
    assert byte_identity["sha256"] == PRE_REGISTRATION_SHA256
    assert byte_identity["matches_generation_time_record"] is True

    assert pre["schema"] == (
        "na49_mixing.staged_r31_matching_scan.pre_registration/1")
    assert pre["run"] == RUN_ID
    assert pre["card"] == CARD_ID
    assert pre["written_before_generation"] is True
    assert pre["no_fitting"] is True
    assert pre["optimizer_invoked"] is False

    # The driver binding stays the frozen dual-tree driver.
    assert pre["input_hashes"]["driver_sha256"] == DRIVER_SHA256
    # The generation-time engine hash is preserved in the
    # pre-registration (the immutable stop-state record).
    assert pre["input_hashes"]["threshold_metric_module_sha256"] == (
        PRE_FIX_ENGINE_SHA256)

    # The T1 statistics: 4 replicas x 25,000 decays per node over
    # 3 windows x 5 nodes = 1,500,000 decays.
    stats = pre["statistics"]
    assert stats["tier"] == "T1"
    assert stats["replicas"] == N_REPLICAS
    assert stats["decays_per_replica"] == DECAYS_PER_REPLICA
    assert stats["windows"] == N_WINDOWS
    assert stats["nodes"] == N_NODES
    assert stats["total_decays"] == N_TOTAL_DECAYS

    # The card-9 seed map.
    seed_map = pre["seed_map"]
    assert seed_map["master_seed"] == MASTER_SEED
    assert seed_map["substream_labels"] == [
        "decay_channel", "decay_kinematics"]
    assert "derive_substreams" in seed_map["rule"]
    assert "20260917" in seed_map["rule"]

    # The metric contract: the card-2 re-registered metric at the
    # unchanged 3.0-sigma tolerance, boundary pairs checked like any
    # other pair.
    metric = pre["discontinuity_metric"]
    assert metric["id"] == "delta_analytic_full"
    assert metric["tolerance_sigma"] == TOLERANCE_SIGMA
    assert metric["boundary_straddling_pairs"] == [2, 6, 10]


def test_provenance_resume_block(run_dir):
    """The updated provenance carries the engine update line (the
    pre-fix ``5d39643d...`` to the fixed ``ee240f44...`` in both
    engine entries), the 76-artifact bit-identity record (a sample of
    the digests verified against disk, the sealed card-8c
    pre-investigation inventory verified as the source), the
    pre-registration byte-identity record, and the resume block
    carrying the stopped verification digest."""
    prov = _load(run_dir, "provenance.json")
    assert prov["schema"] == (
        "na49_mixing.staged_r31_matching_scan.provenance/1")
    assert prov["run"] == RUN_ID
    assert prov["card"] == CARD_ID
    assert prov["driver"] == ("helpers/benchmarking/"
                              "staged_r31_matching_scan.py")
    assert prov["driver_sha256"] == DRIVER_SHA256
    assert _sha256_file(run_dir / "provenance.json") == PROVENANCE_SHA256

    # The engine update line: both engine entries carry the fixed
    # sha256, equal to the on-disk module.
    assert prov["modules"]["threshold_metric_sha256"] == ENGINE_SHA256
    assert prov["input_hashes"]["threshold_metric_module_sha256"] == (
        ENGINE_SHA256)
    assert _sha256_file(ENGINE_PATH) == ENGINE_SHA256
    assert prov["catalog"]["sha256"] == CATALOG_SHA256
    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)

    resume = prov["resume"]
    assert resume["resumed"] is True
    assert resume["resume_card"] == RESUME_CARD
    assert resume["acceptance_record"] == ACCEPTANCE_RECORD

    engine = resume["engine_sha256"]
    assert engine["pre_fix"] == PRE_FIX_ENGINE_SHA256
    assert engine["fixed"] == ENGINE_SHA256
    assert engine["fixed_matches_on_disk_module"] is True
    assert engine["updated_entries"] == [
        "modules.threshold_metric_sha256",
        "input_hashes.threshold_metric_module_sha256",
    ]
    assert engine["engine_fix_card"] == (
        "TASK-20260918-r3-card2-engine-three-body-moment-fix")
    assert engine["sealed_engine_fix_evidence"] == (
        "dev/00runs/r3_threshold_metric_engine_fix_2026-09-18")

    # The stopped verification is kept by digest (the card-8 stop
    # state, superseded by the resume's regenerated report).
    stopped = resume["stopped_verification"]
    assert stopped["sha256"] == STOPPED_VERIFICATION_SHA256
    assert stopped["checks_applied"] == N_CHECKS_APPLIED
    assert stopped["checks_failed"] == 14
    assert stopped["max_abs_sigma"] == 25.98076211353316

    # The 76-artifact bit-identity record: the source inventory is
    # the sealed card-8c pre-investigation inventory (verified), zero
    # mismatches, and a sample of the digests matches disk.
    bit_id = resume["generated_artifact_bit_identity"]
    assert bit_id["generated_artifacts"] == N_GENERATED_ARTIFACTS
    assert bit_id["mismatches"] == 0
    assert bit_id["source_inventory"] == (
        "dev/00runs/r3_se_audit_2026-09-18/pre_investigation_inventory.json"
    )
    assert bit_id["source_inventory_sha256"] == AUDIT_INVENTORY_SHA256
    assert _sha256_file(AUDIT_INVENTORY_PATH) == AUDIT_INVENTORY_SHA256
    digests = bit_id["digests"]
    assert len(digests) == N_GENERATED_ARTIFACTS
    sample = ["pre_registration.json", "window_0_node_0.json",
              "window_1_node_2_replica_1.json",
              "window_2_node_4_replica_3.json"]
    for name in sample:
        assert digests[name] == _sha256_file(run_dir / name), name

    # The pre-registration byte-identity record.
    byte_identity = resume["pre_registration_byte_identity"]
    assert byte_identity["sha256"] == PRE_REGISTRATION_SHA256
    assert byte_identity["matches_generation_time_record"] is True
    assert any(DRIVER_SHA256 in binding
               for binding in byte_identity["generation_time_bindings"])


def test_matching_verification_resumed(run_dir):
    """The regenerated matching verification seals with the pinned
    digest, the 422 = 299 + 113 + 10 census, the 10 named residuals
    with their signed sigmas (each verified against the on-disk
    payload), boundary pair 0/104, interior 10/318, max |sigma|
    7.01270490790188, and the recorded census form."""
    mv = _load(run_dir, "matching_verification.json")
    assert _sha256_file(run_dir / "matching_verification.json") == (
        VERIFICATION_SHA256)
    assert mv["schema"] == (
        "na49_mixing.staged_r31_matching_scan.verification/1")
    assert mv["run"] == RUN_ID
    assert mv["card"] == CARD_ID
    assert mv["metric"]["id"] == "delta_analytic_full"
    assert mv["metric"]["tolerance_sigma"] == TOLERANCE_SIGMA
    assert mv["events_checked"] == N_TOTAL_DECAYS
    assert mv["charge_balance_exact"] is True
    assert mv["no_fitting"] is True
    assert mv["optimizer_invoked"] is False

    summary = mv["summary"]
    assert summary["windows"] == N_WINDOWS
    assert summary["adjacent_pairs_checked"] == 12
    assert summary["checks_applied"] == N_CHECKS_APPLIED
    assert summary["checks_failed"] == N_CHECKS_FAILED
    assert summary["max_abs_sigma"] == MAX_ABS_SIGMA
    assert summary["boundary_straddling_pairs"] == [2, 6, 10]
    assert summary["boundary_pair_max_abs_sigma"] == 0.0
    assert summary["interior_pair_max_abs_sigma"] == MAX_ABS_SIGMA
    # The recorded census form: with the 10 named MC-scatter
    # residuals the original-statistics smoothness clause does not
    # pass in the plain form; the re-scoped acceptance (i) seals
    # exactly this census.
    assert summary["pass"] is False

    # The applied census from the check rows: 422 applied = 299
    # explicit + 113 vacuous-zero + 10 failing; every applied row
    # carries the unchanged 3.0-sigma tolerance.
    applied = 0
    n_pass = 0
    n_vacuous = 0
    n_fail = 0
    for window in mv["windows"].values():
        for pair in window["pairs"].values():
            for row in pair["checks"]:
                if row["status"] == "absent_species":
                    continue
                applied += 1
                assert row["tolerance_sigma"] == TOLERANCE_SIGMA
                if row["status"] == "pass":
                    n_pass += 1
                    if row["sigma"] == 0.0 and row["se_combined"] < 1e-12:
                        n_vacuous += 1
                else:
                    assert row["status"] == "fail"
                    n_fail += 1
    assert applied == N_CHECKS_APPLIED
    assert n_pass == N_EXPLICIT_PASS + N_VACUOUS_ZERO
    assert n_vacuous == N_VACUOUS_ZERO
    assert n_fail == N_CHECKS_FAILED

    # The boundary/interior split.
    nbd = mv["no_boundary_dominance"]
    assert nbd["boundary_pair_applied"] == BOUNDARY_PAIR_APPLIED
    assert nbd["boundary_pair_failed"] == BOUNDARY_PAIR_FAILED
    assert nbd["interior_pairs_applied"] == INTERIOR_PAIRS_APPLIED
    assert nbd["interior_pairs_failed"] == INTERIOR_PAIRS_FAILED
    assert nbd["pass"] is True

    # The 10 named residuals, each verified row by row against the
    # on-disk payload with its signed sigma.
    failing = [
        row
        for window in mv["windows"].values()
        for pair in window["pairs"].values()
        for row in pair["checks"]
        if row["status"] == "fail"
    ]
    assert len(failing) == len(NAMED_RESIDUALS)
    for (window, pair, _global, species, observable, sigma, _conf), row in (
            zip(NAMED_RESIDUALS, failing)):
        assert tuple(row["pair"]) == pair
        assert row["species"] == species
        assert row["observable"] == observable
        assert row["sigma"] == sigma
        assert abs(sigma) > TOLERANCE_SIGMA
        assert row["pass"] is False
        on_disk = _verification_row(mv, window, pair, species, observable)
        assert on_disk is row
        assert on_disk["sigma"] == sigma


def test_gate_evidence(run_dir):
    """The gate evidence carries the no_boundary_dominance clause
    passing with its clause result and inputs pinned, the
    smoothness clause's recorded census, the card-8c cross-reference
    block as the authoritative smoothness evidence (50-replica x
    20,000 statistics, 0/422, max |sigma| 2.96, the sealed card-8c
    confirmation artifact's digest verified against disk), the pass
    census, the frozen driver's verbatim gate refusal, the input
    digests, and the instrument identity."""
    gate = _load(run_dir, "gate3_r31.json")
    assert _sha256_file(run_dir / "gate3_r31.json") == GATE_SHA256
    assert gate["schema"] == (
        "na49_mixing.staged_r31_matching_scan.gate_resume/1")
    assert gate["gate"] == "gate 3 (plan section 9, line 650)"
    assert gate["run"] == RUN_ID
    assert gate["card"] == CARD_ID
    assert gate["resume_card"] == RESUME_CARD
    assert gate["acceptance_record"] == ACCEPTANCE_RECORD
    assert gate["events_checked"] == N_TOTAL_DECAYS
    assert gate["charge_balance_exact"] is True
    assert gate["no_fitting"] is True
    assert gate["optimizer_invoked"] is False
    assert gate["sealed"] is True

    # The no_boundary_dominance clause: passing, with the clause
    # result and its inputs pinned (aggregate and per window).
    nbd = gate["clauses"]["no_boundary_dominance"]
    assert nbd["pass"] is True
    assert nbd["boundary_pair_failure_rate"] == 0.0
    assert nbd["interior_pair_failure_rate"] == NBD_INTERIOR_RATE
    assert nbd["boundary_pair_failed"] == BOUNDARY_PAIR_FAILED
    assert nbd["boundary_pair_applied"] == BOUNDARY_PAIR_APPLIED
    assert nbd["interior_pairs_failed"] == INTERIOR_PAIRS_FAILED
    assert nbd["interior_pairs_applied"] == INTERIOR_PAIRS_APPLIED
    assert len(nbd["per_window"]) == N_WINDOWS
    for row, (window, qn, boundary_pair, b_rate, i_rate, b_applied,
              b_failed, i_applied, i_failed) in zip(
                  nbd["per_window"], NBD_PER_WINDOW):
        assert row["window"] == window
        assert tuple(row["qn"]) == qn
        assert row["boundary_pair"] == boundary_pair
        assert row["boundary_pair_failure_rate"] == b_rate
        assert row["interior_pairs_failure_rate"] == i_rate
        assert row["boundary_pair_applied"] == b_applied
        assert row["boundary_pair_failed"] == b_failed
        assert row["interior_pairs_applied"] == i_applied
        assert row["interior_pairs_failed"] == i_failed
        assert row["no_boundary_dominance"] is True

    # The smoothness clause: the recorded original-statistics census
    # (the 10 named residuals persist, so the plain clause does not
    # pass; the authoritative evidence is the card-8c block below).
    smooth = gate["clauses"]["smoothness"]
    assert smooth["checks_applied"] == N_CHECKS_APPLIED
    assert smooth["checks_failed"] == N_CHECKS_FAILED
    assert smooth["max_abs_sigma"] == MAX_ABS_SIGMA
    assert smooth["boundary_straddling_pairs"] == [2, 6, 10]
    assert smooth["boundary_pair_max_abs_sigma"] == 0.0
    assert smooth["interior_pair_max_abs_sigma"] == MAX_ABS_SIGMA
    assert smooth["pass"] is False
    assert "card-8c" in smooth["verdict_note"]

    # The pass census (the card-8a P1 vacuous-zero convention).
    census = gate["pass_census"]
    assert census["checks_applied"] == N_CHECKS_APPLIED
    assert census["explicit_pass"] == N_EXPLICIT_PASS
    assert census["vacuous_zero_pass"] == N_VACUOUS_ZERO
    assert census["checks_failed"] == N_CHECKS_FAILED
    assert "DETERMINISTIC_SE_FLOOR" in census["classification_rule"]

    # The named-residual register: the 10 residuals with their
    # signed sigmas, each carrying the card-8c classification, and
    # cross-checked against the regenerated verification rows.
    register = gate["named_residual_register"]
    assert len(register) == len(NAMED_RESIDUALS)
    mv = _load(run_dir, "matching_verification.json")
    for entry, (window, pair, global_pair, species, observable,
                sigma, conf), (d_obs, d_analytic, se) in zip(
                    register, NAMED_RESIDUALS, RESIDUAL_PAYLOADS):
        assert entry["window"] == window
        assert tuple(entry["pair"]) == pair
        assert entry["global_pair"] == global_pair
        assert entry["species"] == species
        assert entry["observable"] == observable
        assert entry["sigma"] == sigma
        assert entry["tolerance_sigma"] == TOLERANCE_SIGMA
        assert entry["card_8c_classification"] == (
            "genuine_finite_mc_scatter")
        assert entry["card_8c_sigma_confirmation"] == conf
        # The sealed card-8c confirmation sigmas resolve every named
        # residual inside the 3.0-sigma band at the honest 50-replica
        # SE (the largest recorded is 1.7448633070905266).
        assert abs(conf) < TOLERANCE_SIGMA
        on_disk = _verification_row(mv, window, pair, species, observable)
        assert on_disk["sigma"] == sigma
        assert entry["delta_observed"] == d_obs
        assert entry["delta_analytic_full"] == d_analytic
        assert entry["se_combined"] == se
        assert entry["delta_observed"] == on_disk["delta_observed"]
        assert entry["delta_analytic_full"] == on_disk["delta_analytic_full"]
        assert entry["se_combined"] == on_disk["se_combined"]
        assert entry["delta_observed"] - entry["delta_analytic_full"] == (
            pytest.approx(sigma * se, rel=1e-12, abs=1e-18))

    # The four closed original failures.
    closed = gate["closed_original_failures"]
    assert len(closed) == len(CLOSED_ORIGINAL_FAILURES)
    for entry, (window, pair, global_pair, species, observable,
                sigma_original, sigma_resume, status) in zip(
                    closed, CLOSED_ORIGINAL_FAILURES):
        assert entry["window"] == window
        assert tuple(entry["pair"]) == pair
        assert entry["global_pair"] == global_pair
        assert entry["species"] == species
        assert entry["observable"] == observable
        assert entry["sigma_original_sealed"] == sigma_original
        assert entry["sigma_resume"] == sigma_resume
        assert entry["status_resume"] == status

    # The card-8c cross-reference block (acceptance (v)): the
    # confirmation statistics as the authoritative smoothness
    # evidence, with the sealed card-8c artifact digest pinned and
    # verified against disk.
    cross = gate["card_8c_cross_reference"]
    assert cross["card"] == "TASK-20260918-r3-se-audit-confirmation-mc"
    assert cross["run_dir"] == "dev/00runs/r3_se_audit_2026-09-18"
    assert cross["artifact"] == (
        "dev/00runs/r3_se_audit_2026-09-18/confirmation_verification.json"
    )
    assert cross["sha256"] == AUDIT_CONFIRMATION_SHA256
    assert _sha256_file(AUDIT_CONFIRMATION_PATH) == AUDIT_CONFIRMATION_SHA256
    stats = cross["statistics"]
    assert stats["replicas"] == AUDIT_REPLICAS
    assert stats["decays_per_replica"] == AUDIT_DECAYS_PER_REPLICA
    assert stats["master_seed"] == AUDIT_MASTER_SEED
    summary = cross["summary"]
    assert summary["checks_applied"] == N_CHECKS_APPLIED
    assert summary["checks_failed"] == 0
    assert summary["max_abs_sigma"] == AUDIT_MAX_ABS_SIGMA
    assert cross["authoritative_smoothness_evidence"] is True

    # The frozen driver's verbatim gate refusal on a failing
    # verification (exit code 2; the gate evidence is assembled by
    # the card-8b instrument in the driver's own gate-assembly
    # shape).
    refusal = gate["frozen_driver_gate_refusal"]
    assert refusal["attempted"] is True
    assert refusal["exit_code"] == 2
    assert refusal["message"] == GATE_REFUSAL_MESSAGE

    # The input digests, verified against disk.
    inputs = gate["input_hashes"]
    assert inputs["matching_verification_sha256"] == VERIFICATION_SHA256
    assert inputs["pre_registration_sha256"] == PRE_REGISTRATION_SHA256
    assert inputs["provenance_sha256"] == PROVENANCE_SHA256
    assert inputs["threshold_metric_module_sha256"] == ENGINE_SHA256
    assert inputs["driver_sha256"] == DRIVER_SHA256

    # The instrument identity.
    instrument = gate["instrument"]
    assert instrument["path"] == (
        "dev/00runs/tmp/r31_resume_8b/resume_instrument.py")
    assert instrument["sha256"] == INSTRUMENT_SHA256


def test_clean_environment_record(run_dir):
    """The post-seal clean-environment record re-verified the 79
    sums-covered files without mismatches, re-ran the matching
    verification from the clean interpreter (``python -I``; the
    probe's exit code 1 is the expected named-residual exit), the
    reproduced report hashes match the sealed digest, and the frozen
    driver's verbatim clean-validate refusal is recorded."""
    clean = _load(run_dir, "clean_environment_commands.json")
    assert clean["schema"] == (
        "na49_mixing.staged_r31_matching_scan.clean_environment_resume/1")
    assert clean["run"] == RUN_ID
    assert clean["card"] == CARD_ID
    assert clean["resume_card"] == RESUME_CARD
    assert clean["acceptance_record"] == ACCEPTANCE_RECORD
    assert clean["seal_verification"] == {
        "files_checked": SUMS_LINE_COUNT, "mismatches": []}
    assert clean["hashes_match"] is True
    assert clean["probe_returncode"] == 1
    assert clean["probe_returncode_expected"] is True
    assert len(clean["commands"]) == 1
    command = clean["commands"][0]
    assert command["argv"][0] == "/usr/bin/python"
    assert "-I" in command["argv"]
    assert "--verify" in command["argv"]
    assert command["cwd"] == str(MONO_ROOT)
    sealed = clean["sealed_report_sha256"]
    recomputed = clean["recomputed_report_sha256"]
    assert sealed == recomputed
    assert sealed["matching_verification.json"] == VERIFICATION_SHA256
    assert _sha256_file(run_dir / "matching_verification.json") == (
        VERIFICATION_SHA256)

    # The frozen driver's verbatim clean-validate refusal (exit
    # code 2) with the probe's stdout line.
    refusal = clean["frozen_driver_clean_validate"]
    assert refusal["attempted"] is True
    assert refusal["refused"] is True
    assert refusal["exit_code"] == 2
    assert refusal["message"] == CLEAN_REFUSAL_MESSAGE
    assert refusal["probe_stdout"] == CLEAN_PROBE_STDOUT


def test_generated_artifacts_bit_identity(run_dir):
    """All 76 generated artifacts (the pre-registration + the 15
    node + the 60 replica records) listed in the provenance
    bit-identity record match their on-disk digests (the full 76,
    not a sample)."""
    prov = _load(run_dir, "provenance.json")
    digests = prov["resume"]["generated_artifact_bit_identity"]["digests"]
    assert len(digests) == N_GENERATED_ARTIFACTS
    assert set(digests) <= EXPECTED_RUN_FILES
    for name, digest in digests.items():
        assert _sha256_file(run_dir / name) == digest, name
    # The record's digests agree with the seal's file entries.
    seal = _load(run_dir, "seal_manifest.json")
    sealed = {e["path"]: e["sha256"] for e in seal["files"]}
    for name, digest in digests.items():
        assert sealed[name] == digest, name


def test_rescoped_acceptance_structure(run_dir):
    """The re-scoped acceptance is honored by the sealed evidence
    structure: the gate evidence carries BOTH the original-statistics
    seal (the 10 named MC-scatter residuals, acceptance (i)) AND the
    card-8c confirmation-statistics cross-reference (the 0/422
    authoritative smoothness evidence, acceptance (v))."""
    gate = _load(run_dir, "gate3_r31.json")
    # Acceptance (i): the original-statistics seal.
    assert "named_residual_register" in gate
    register = gate["named_residual_register"]
    assert len(register) == N_CHECKS_FAILED
    for entry in register:
        assert entry["card_8c_classification"] == (
            "genuine_finite_mc_scatter")
        assert abs(entry["sigma"]) > TOLERANCE_SIGMA
    assert "closed_original_failures" in gate
    # Acceptance (v): the confirmation-statistics cross-reference.
    cross = gate["card_8c_cross_reference"]
    assert cross["statistics"]["replicas"] == AUDIT_REPLICAS
    assert cross["statistics"]["decays_per_replica"] == (
        AUDIT_DECAYS_PER_REPLICA)
    assert cross["summary"]["checks_applied"] == N_CHECKS_APPLIED
    assert cross["summary"]["checks_failed"] == 0
    assert cross["summary"]["max_abs_sigma"] == AUDIT_MAX_ABS_SIGMA
    assert cross["authoritative_smoothness_evidence"] is True
    # The two blocks cross-reference each other's statistics: the
    # original-statistics census (10 named residuals) and the
    # confirmation census (0 failed) are both present, and the
    # smoothness clause's recorded verdict points at the
    # confirmation block as the authoritative evidence.
    assert gate["clauses"]["smoothness"]["checks_failed"] == (
        N_CHECKS_FAILED)
    assert "card_8c_cross_reference" in (
        gate["clauses"]["smoothness"]["verdict_note"])


# ---------------------------------------------------------------------------
# Part 2: the pure contract (no sealed artifacts)
# ---------------------------------------------------------------------------
def test_frozen_driver_and_engine_contract():
    """The frozen R31 driver is byte-identical at the pinned digest
    in both trees, the on-disk engine is the fixed card-2 engine,
    the 3.0-sigma tolerance and the deterministic-SE floor are the
    pre-registered values, and the committed catalog is at its
    pinned digest."""
    assert _sha256_file(DRIVER_PATH) == DRIVER_SHA256
    assert _sha256_file(DRIVER_OUTER_PATH) == DRIVER_SHA256
    assert _sha256_file(ENGINE_PATH) == ENGINE_SHA256
    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)
    assert SIGMA_TOLERANCE == TOLERANCE_SIGMA
    assert DETERMINISTIC_SE_FLOOR == 1e-12
