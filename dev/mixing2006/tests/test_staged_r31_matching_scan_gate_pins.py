"""Round 3 card 8b: run R31 verify-resume sealed-run artifact pins.

Task ``TASK-20260918-r3-run-r31-verify-resume`` (round 3, card 8b), under
the re-scoped acceptance of
``.collab/tasks/TASK-20260918-r3-acceptance-rescope-card8a-8b.md`` (its
section 2).  This module pins the sealed run evidence under
``dev/00runs/r31_catalog_continuum_matching_scan_2026-09-17/`` after the
card-8b resume: the regenerated matching verification (422 applied
checks at the original T1 statistics with the fixed card-2 engine — 299
explicit passes + 113 vacuous-zero passes under the
``DETERMINISTIC_SE_FLOOR`` convention + the 10 named MC-scatter
residuals, max |sigma| 7.01270490790188, the boundary-straddling pairs
clean at 0 of 104 and the interior pairs at 10 of 318), the updated
provenance (the fixed engine sha256 ``ee240f44...`` replacing the
pre-fix ``5d39643d...`` in both engine entries, with the resume block
carrying the 76-artifact bit-identity digest record and the
pre-registration byte-identity record), the gate evidence
``gate3_r31.json`` (the driver's gate-assembly shape assembled by the
card-8b instrument because the frozen driver's ``--gate`` refuses any
verification with ``checks_failed > 0``; the named-residual register
with the card-8c classification, the no-boundary-dominance clause, and
the card-8c cross-reference block as the authoritative smoothness
evidence), the seal (``seal_manifest.json`` + ``SHA256SUMS.txt``), and
the post-seal clean-environment record (the clean-interpreter
``python -I`` re-run reproducing the verification byte for byte; the
probe's exit code 1 is the expected named-residual exit under the
re-scoped acceptance).

Sibling of ``tests/test_staged_r3_se_audit_pins.py`` and
``tests/test_staged_r3_threshold_metric_engine_fix_pins.py`` (the same
pin style: sealed-dir discovery, skip when the gitignored archive is
absent, digest pins plus structural pins).  These pins are read-only
over the sealed artifacts: nothing is recomputed, the sealed run is
never re-sealed, and the clean-environment record is never re-executed.

Pins:

- ``test_sealed_run_dir_inventory``: the run dir holds exactly the 82
  files of the resumed state (the 76 generated artifacts, the
  regenerated verification, the updated provenance, the gate evidence,
  the seal pair, the post-seal clean-environment record).
- ``test_seal_integrity``: ``SHA256SUMS.txt`` carries 79
  monorepo-root-relative lines, sorted by path, every digest verified
  from the monorepo root; ``seal_manifest.json`` records 80 sealed
  files (the sums file included), every entry verifies, and the sums
  digest is pinned; the clean-environment record is the documented
  post-seal state and is excluded from the sums.
- ``test_pre_registration_byte_identity``: the pre-registration is
  byte-identical to the generation-time state (sha256 ``3f5ed7d8...``,
  the value pinned by the card-8a archive block and the card-8c
  pre-investigation inventory); its driver binding ``b47b2a85...`` is
  intact and its generation-time engine hash ``5d39643d...`` is
  preserved inside the pre-registration.
- ``test_generated_artifacts_bit_identity``: the 76 generated artifacts
  (pre-registration + 15 node + 60 replica records) match the digest
  record in ``provenance.json``'s resume block, which itself matches
  the sealed card-8c pre-investigation inventory; 0 mismatches.
- ``test_provenance_engine_update``: both engine entries carry the
  fixed card-8a sha256 ``ee240f44...`` (equal to the on-disk module in
  the nested tree), the pre-fix value is recorded in the resume block
  with the updated entry names, and the frozen driver digest matches
  both trees on disk.
- ``test_verification_census``: 422 applied, 299 explicit + 113
  vacuous-zero passes + 10 failures, max |sigma| 7.01270490790188
  (interior; the boundary pairs' failing-check maximum is 0.0), the
  tolerance 3.0 on every applied row, ``events_checked`` 1,500,000,
  charge balance exact, ``no_fitting`` true, and the
  ``no_boundary_dominance`` clause passing in aggregate (0 of 104
  boundary checks failed against 10 of 318 interior checks) and in
  every window.
- ``test_named_residual_register``: the gate evidence's register lists
  exactly the 10 failing checks with their signed sigmas, each
  cross-identified with the sealed card-8c register (classification
  ``genuine_finite_mc_scatter`` and the sealed original-statistics
  sigma), and the four closed original failures carry their pre-fix
  sigmas and their resume statuses.
- ``test_card_8c_cross_reference``: the gate evidence cross-references
  the card-8c confirmation statistics (50 replicas x 20,000 decays,
  master seed 20260918; 422 applied, 0 failed, max |sigma| 2.96) as the
  authoritative smoothness evidence, with the artifact digest pinned to
  the on-disk sealed file.
- ``test_gate_evidence_structure``: the gate evidence carries the
  driver's clause shapes, the pass census, the frozen-driver refusal
  record (exit code 2, the verbatim refusal), the input digests (all
  verified against disk), and the instrument identity.
- ``test_clean_environment_record``: the post-seal record re-verified
  the 79 sums-covered files without mismatches, re-ran the matching
  verification in the clean interpreter (``python -I``, return code 1 =
  the expected named-residual exit), and reproduced the sealed report
  bytes (pinned digest, equal to the on-disk file).

Hermetic: reads the sealed run dir (gitignored; the artifact is present
on disk) and the committed config, package, and drivers; no computation
runs.  If the run dir is absent (a fresh checkout), the artifact pins
skip with a clear message.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

NESTED_ROOT = Path(__file__).resolve().parent.parent
MONO_ROOT = NESTED_ROOT.parent.parent
RUN_NAME = "r31_catalog_continuum_matching_scan_2026-09-17"
RUN_DIR = MONO_ROOT / "dev" / "00runs" / RUN_NAME
SUMS_PREFIX = f"dev/00runs/{RUN_NAME}/"

ENGINE_PATH = NESTED_ROOT / "na49_mixing" / "threshold_metric.py"
DRIVER_PATH = NESTED_ROOT / "helpers" / "benchmarking" / "staged_r31_matching_scan.py"
DRIVER_OUTER_PATH = MONO_ROOT / "helpers" / "benchmarking" / "staged_r31_matching_scan.py"
AUDIT_CONFIRMATION_PATH = (
    MONO_ROOT / "dev" / "00runs" / "r3_se_audit_2026-09-18"
    / "confirmation_verification.json"
)
AUDIT_INVENTORY_PATH = (
    MONO_ROOT / "dev" / "00runs" / "r3_se_audit_2026-09-18"
    / "pre_investigation_inventory.json"
)
INSTRUMENT_PATH = (
    MONO_ROOT / "dev" / "00runs" / "tmp" / "r31_resume_8b"
    / "resume_instrument.py"
)

#: The identity digests of the resumed run state.
ENGINE_SHA = "ee240f44645d2e592acfae462f7d1e9c3b3dd8cd78c96bd40b7b73509f7fcc63"
PRE_FIX_ENGINE_SHA = (
    "5d39643d95bf80f8b4831d99b338574e25b95b48b0a9bb45c31ea7153f99aefb"
)
DRIVER_SHA = "b47b2a858c3880ecf69e143035e8c209b6e6e5e0708ca2b9058c6c384340afe6"
PRE_REGISTRATION_SHA = (
    "3f5ed7d8599aeaa53d38de897f731125fb5d50e0e9f8d546058ea3ae9bbc1a14"
)
STOPPED_VERIFICATION_SHA = (
    "de01f62f4907b3acbb8b4c5959148708d474d9c22df925d6b6d89dd188cde629"
)
VERIFICATION_SHA = "9170426043399dc954c08ec255c7a7592f63ffbdfdcc3ab655cc1c0dd8e499af"
GATE_SHA = "98c1ba7327fc65d0cc6828fc0c93f12e1f4b9eaffad3a0617223143994dac686"
CLEAN_ENV_SHA = "e2f0104e7094fc3846068eada12fd2ae9bf5bc794c0a8fbb4f98688b72624594"
SUMS_SHA = "949fefb6ec53282fd532ff72095a44efad180a44e9d8c40209e2baa97e88205f"
AUDIT_CONFIRMATION_SHA = (
    "98a3bff11184f93d16d17dbebee65dc2fd6834fac505aad7670ee7063ebd204d"
)
AUDIT_INVENTORY_SHA = (
    "16679283da93bc7cee5474f814478e016933fd932ba087ff241099999723bd4c"
)
INSTRUMENT_SHA = "d7eea975baa1bcf4caca6af450064e2a8f5d5cde85ea002c8d4031ae5261093f"

#: The 76 generated artifacts (pre-registration + 15 node + 60 replica
#: records) are bit-identical to the card-8 stop state.
N_GENERATED = 76
N_SUMS_ENTRIES = 79
N_SEAL_FILES = 80
N_RUN_FILES = 82

#: The 10 named MC-scatter residuals (re-scoped acceptance (i)): the
#: signed sigma of the regenerated verification, which equals the sealed
#: card-8c original-statistics sigma for every entry.
NAMED_RESIDUALS = (
    (0, (0, 1), 0, "proton", "mean_pt_y", 3.2123466532077045),
    (0, (0, 1), 0, "pi_minus", "mean_pt_y", 3.3254504430919427),
    (0, (3, 4), 3, "neutron", "mean_p", 3.151740704856282),
    (2, (0, 1), 8, "pi_0", "mean_pt_y", -3.0280248998333104),
    (2, (0, 1), 8, "K0L", "mean_p", 3.064256426739259),
    (2, (0, 1), 8, "K0L", "mean_pt_y", 3.069041367744035),
    (2, (1, 2), 9, "pi_0", "mean_p", 7.01270490790188),
    (2, (3, 4), 11, "K0S", "mean_pt_y", 3.4011873510462327),
    (2, (3, 4), 11, "K0L", "mean_p", 3.7365731963210447),
    (2, (3, 4), 11, "K0L", "mean_pt_y", 4.547781556513862),
)

#: The four closed original failures (re-scoped acceptance (i): "the 4
#: engine + 2 SE-artifact failures closed" - the closed set is the four
#: checks 14 - 10 = 4: the two Lambda(1520) pi+ three-body checks and
#: the two deterministic-SE artifacts).
CLOSED_ORIGINAL_FAILURES = (
    (1, (2, 3), 6, "pi_plus", "mean_pt_y", 7.225811435009375,
     -0.9201361176029509),
    (1, (3, 4), 7, "pi_plus", "mean_pt_y", -5.610935929060075,
     0.36971803795576264),
    (0, (3, 4), 3, "K_0", "mean_p", 25.98076211353316, 0.0),
    (0, (3, 4), 3, "Sigma_plus", "mean_p", 25.98076211353316, 0.0),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_run_dir() -> None:
    if not RUN_DIR.is_dir():
        pytest.skip(
            f"the sealed R31 run dir is absent on this checkout: {RUN_DIR}"
        )


def _load(name: str):
    return json.loads((RUN_DIR / name).read_text(encoding="utf-8"))


def _verification_checks():
    verification = _load("matching_verification.json")
    rows = []
    for wd in verification["windows"].values():
        for pd in wd["pairs"].values():
            rows.extend(pd["checks"])
    return verification, rows


# ---------------------------------------------------------------------------
# Inventory and seal
# ---------------------------------------------------------------------------


def test_sealed_run_dir_inventory():
    _require_run_dir()
    names = sorted(path.name for path in RUN_DIR.iterdir() if path.is_file())
    assert len(names) == N_RUN_FILES, names
    expected_generated = (
        ["pre_registration.json"]
        + [f"window_{w}_node_{j}.json" for w in range(3) for j in range(5)]
        + [
            f"window_{w}_node_{j}_replica_{r}.json"
            for w in range(3)
            for j in range(5)
            for r in range(4)
        ]
    )
    assert len(expected_generated) == N_GENERATED
    expected = sorted(
        expected_generated
        + [
            "matching_verification.json",
            "provenance.json",
            "gate3_r31.json",
            "SHA256SUMS.txt",
            "seal_manifest.json",
            "clean_environment_commands.json",
        ]
    )
    assert names == expected, sorted(set(names) ^ set(expected))


def test_seal_integrity():
    _require_run_dir()
    sums_lines = [
        line
        for line in (RUN_DIR / "SHA256SUMS.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    assert len(sums_lines) == N_SUMS_ENTRIES
    paths = [line.split("  ", 1)[1] for line in sums_lines]
    assert paths == sorted(paths), "the sums file is not path-sorted"
    assert all(path.startswith(SUMS_PREFIX) for path in paths)
    for line in sums_lines:
        digest, rel = line.split("  ", 1)
        payload = MONO_ROOT / rel
        assert payload.is_file(), rel
        assert _sha256(payload) == digest, rel
    assert _sha256(RUN_DIR / "SHA256SUMS.txt") == SUMS_SHA

    seal = _load("seal_manifest.json")
    assert seal["file_count"] == N_SEAL_FILES
    assert seal["sha256_sums_file"] == "SHA256SUMS.txt"
    assert seal["sha256_sums_sha256"] == SUMS_SHA
    assert seal["excluded_from_sums"] == [
        "seal_manifest.json",
        "SHA256SUMS.txt",
    ]
    sealed_names = sorted(entry["path"] for entry in seal["files"])
    on_disk = sorted(
        path.name for path in RUN_DIR.iterdir() if path.is_file()
    )
    on_disk.remove("seal_manifest.json")  # the manifest cannot list itself
    on_disk.remove(
        "clean_environment_commands.json"
    )  # the documented post-seal record
    assert sealed_names == on_disk
    for entry in seal["files"]:
        payload = RUN_DIR / entry["path"]
        assert _sha256(payload) == entry["sha256"], entry["path"]
        assert payload.stat().st_size == entry["bytes"], entry["path"]
    # the clean-environment record is the documented post-seal state
    clean_names = {entry["path"] for entry in seal["files"]}
    assert "clean_environment_commands.json" not in clean_names


# ---------------------------------------------------------------------------
# Bit identity: the pre-registration and the 76 generated artifacts
# ---------------------------------------------------------------------------


def test_pre_registration_byte_identity():
    _require_run_dir()
    assert _sha256(RUN_DIR / "pre_registration.json") == PRE_REGISTRATION_SHA
    provenance = _load("provenance.json")
    identity = provenance["resume"]["pre_registration_byte_identity"]
    assert identity["sha256"] == PRE_REGISTRATION_SHA
    assert identity["matches_generation_time_record"] is True
    # the generation-time bindings are intact inside the pre-registration
    pre = _load("pre_registration.json")
    assert pre["input_hashes"]["driver_sha256"] == DRIVER_SHA
    assert pre["input_hashes"][
        "threshold_metric_module_sha256"
    ] == PRE_FIX_ENGINE_SHA
    assert pre["written_before_generation"] is True
    # the stopped verification digest is recorded in the resume block
    stopped = provenance["resume"]["stopped_verification"]
    assert stopped["sha256"] == STOPPED_VERIFICATION_SHA
    assert stopped["checks_applied"] == 422
    assert stopped["checks_failed"] == 14


def test_generated_artifacts_bit_identity():
    _require_run_dir()
    provenance = _load("provenance.json")
    record = provenance["resume"]["generated_artifact_bit_identity"]
    assert record["generated_artifacts"] == N_GENERATED
    assert record["mismatches"] == 0
    assert record["source_inventory_sha256"] == AUDIT_INVENTORY_SHA
    assert _sha256(AUDIT_INVENTORY_PATH) == AUDIT_INVENTORY_SHA
    digests = record["digests"]
    assert len(digests) == N_GENERATED
    inventory = json.loads(AUDIT_INVENTORY_PATH.read_text(encoding="utf-8"))
    inventory_map = {
        Path(entry["path"]).name: entry["sha256"]
        for entry in inventory["r31_files"]
    }
    for name, digest in digests.items():
        payload = RUN_DIR / name
        assert payload.is_file(), name
        assert _sha256(payload) == digest, name
        assert inventory_map[name] == digest, name


# ---------------------------------------------------------------------------
# Provenance: the engine sha256 update
# ---------------------------------------------------------------------------


def test_provenance_engine_update():
    _require_run_dir()
    provenance = _load("provenance.json")
    assert provenance["modules"]["threshold_metric_sha256"] == ENGINE_SHA
    assert provenance["input_hashes"][
        "threshold_metric_module_sha256"
    ] == ENGINE_SHA
    assert provenance["driver_sha256"] == DRIVER_SHA
    assert provenance["input_hashes"]["driver_sha256"] == DRIVER_SHA
    assert _sha256(ENGINE_PATH) == ENGINE_SHA
    assert _sha256(DRIVER_PATH) == DRIVER_SHA
    assert _sha256(DRIVER_OUTER_PATH) == DRIVER_SHA
    engine = provenance["resume"]["engine_sha256"]
    assert engine["pre_fix"] == PRE_FIX_ENGINE_SHA
    assert engine["fixed"] == ENGINE_SHA
    assert engine["fixed_matches_on_disk_module"] is True
    assert engine["updated_entries"] == [
        "modules.threshold_metric_sha256",
        "input_hashes.threshold_metric_module_sha256",
    ]
    assert provenance["resume"]["resume_card"] == (
        "TASK-20260918-r3-run-r31-verify-resume"
    )
    assert provenance["resume"]["acceptance_record"] == (
        ".collab/tasks/TASK-20260918-r3-acceptance-rescope-card8a-8b.md"
    )


# ---------------------------------------------------------------------------
# The regenerated matching verification
# ---------------------------------------------------------------------------


def test_verification_census():
    _require_run_dir()
    assert _sha256(RUN_DIR / "matching_verification.json") == VERIFICATION_SHA
    verification, rows = _verification_checks()
    summary = verification["summary"]
    assert summary["checks_applied"] == 422
    assert summary["checks_failed"] == 10
    assert summary["max_abs_sigma"] == 7.01270490790188
    assert summary["interior_pair_max_abs_sigma"] == 7.01270490790188
    assert summary["boundary_pair_max_abs_sigma"] == 0.0
    assert summary["boundary_straddling_pairs"] == [2, 6, 10]
    assert summary["pass"] is False
    assert verification["metric"]["id"] == "delta_analytic_full"
    assert verification["metric"]["tolerance_sigma"] == 3.0
    assert verification["events_checked"] == 1_500_000
    assert verification["charge_balance_exact"] is True
    assert verification["no_fitting"] is True
    assert verification["optimizer_invoked"] is False

    explicit = vacuous = failed = 0
    for row in rows:
        status = row["status"]
        if status == "pass":
            assert row["tolerance_sigma"] == 3.0
            if row["se_combined"] < 1e-12 and row["sigma"] == 0.0:
                vacuous += 1
            else:
                explicit += 1
        elif status == "fail":
            assert row["tolerance_sigma"] == 3.0
            failed += 1
        else:
            assert status in ("absent_species", "absent_species_analytic")
    assert (explicit, vacuous, failed) == (299, 113, 10)

    dominance = verification["no_boundary_dominance"]
    assert dominance["pass"] is True
    assert dominance["boundary_pair_failed"] == 0
    assert dominance["boundary_pair_applied"] == 104
    assert dominance["interior_pairs_failed"] == 10
    assert dominance["interior_pairs_applied"] == 318
    assert dominance["boundary_pair_failure_rate"] == 0.0
    assert dominance["interior_pair_failure_rate"] == 0.031446540880503145
    for row in dominance["per_window"]:
        assert row["no_boundary_dominance"] is True
        assert row["boundary_pair_failed"] == 0


def test_named_residual_register():
    _require_run_dir()
    gate = _load("gate3_r31.json")
    register = gate["named_residual_register"]
    assert len(register) == 10
    verification, rows = _verification_checks()
    fail_rows = {
        (row["global_pair"], row["species"], row["observable"]): row
        for row in rows
        if row["status"] == "fail"
    }
    assert len(fail_rows) == 10
    confirmation = json.loads(
        AUDIT_CONFIRMATION_PATH.read_text(encoding="utf-8")
    )
    conf_named = {
        (int(r["window"]), tuple(r["pair"]), r["species"], r["observable"]): r
        for r in confirmation["named_residuals_phase_b"]
    }
    for entry in register:
        key = (entry["window"], tuple(entry["pair"]), entry["species"],
               entry["observable"])
        expected_sigma = {
            (w, p, s, o): sig
            for w, p, g, s, o, sig in NAMED_RESIDUALS
        }[key]
        assert entry["global_pair"] == 4 * entry["window"] + entry["pair"][0]
        assert entry["boundary_straddling"] is False
        assert entry["sigma"] == expected_sigma, key
        row = fail_rows[(entry["global_pair"], entry["species"],
                         entry["observable"])]
        assert row["sigma"] == entry["sigma"], key
        assert row["se_combined"] == entry["se_combined"], key
        conf_row = conf_named[key]
        assert conf_row["classification_phase_b"] == (
            "genuine_finite_mc_scatter"
        )
        assert entry["card_8c_classification"] == "genuine_finite_mc_scatter"
        assert conf_row["sigma_old_sealed"] == entry["sigma"], key
        assert entry["card_8c_sigma_confirmation"] == conf_row["sigma_new"]
    indices = sorted(entry["index"] for entry in register)
    assert indices == list(range(1, 11))

    closed = gate["closed_original_failures"]
    assert len(closed) == 4
    for w, pair, gp, species, observable, old, new in (
        CLOSED_ORIGINAL_FAILURES
    ):
        match = [
            entry
            for entry in closed
            if (entry["window"], tuple(entry["pair"]), entry["species"],
                entry["observable"]) == (w, pair, species, observable)
        ]
        assert len(match) == 1, (w, pair, species, observable)
        entry = match[0]
        assert entry["global_pair"] == gp
        assert entry["sigma_original_sealed"] == old
        assert abs(entry["sigma_resume"]) == abs(new)
        assert entry["status_resume"].startswith("pass")


# ---------------------------------------------------------------------------
# The gate evidence and the card-8c cross-reference
# ---------------------------------------------------------------------------


def test_card_8c_cross_reference():
    _require_run_dir()
    gate = _load("gate3_r31.json")
    block = gate["card_8c_cross_reference"]
    assert block["authoritative_smoothness_evidence"] is True
    assert block["card"] == "TASK-20260918-r3-se-audit-confirmation-mc"
    assert block["run_dir"] == "dev/00runs/r3_se_audit_2026-09-18"
    assert block["statistics"] == {
        "replicas": 50,
        "decays_per_replica": 20000,
        "master_seed": 20260918,
    }
    assert block["summary"] == {
        "checks_applied": 422,
        "checks_failed": 0,
        "max_abs_sigma": 2.959754302712887,
    }
    assert block["sha256"] == AUDIT_CONFIRMATION_SHA
    assert _sha256(AUDIT_CONFIRMATION_PATH) == AUDIT_CONFIRMATION_SHA


def test_gate_evidence_structure():
    _require_run_dir()
    assert _sha256(RUN_DIR / "gate3_r31.json") == GATE_SHA
    gate = _load("gate3_r31.json")
    assert gate["schema"] == (
        "na49_mixing.staged_r31_matching_scan.gate_resume/1"
    )
    assert gate["gate"] == "gate 3 (plan section 9, line 650)"
    assert gate["run"] == "R31"
    assert gate["card"] == (
        "TASK-20260917-r3-run-r31-catalog-continuum-matching-scan"
    )
    assert gate["resume_card"] == "TASK-20260918-r3-run-r31-verify-resume"
    assert gate["acceptance_record"] == (
        ".collab/tasks/TASK-20260918-r3-acceptance-rescope-card8a-8b.md"
    )
    smoothness = gate["clauses"]["smoothness"]
    assert smoothness["checks_applied"] == 422
    assert smoothness["checks_failed"] == 10
    assert smoothness["max_abs_sigma"] == 7.01270490790188
    assert smoothness["pass"] is False
    dominance = gate["clauses"]["no_boundary_dominance"]
    assert dominance["pass"] is True
    assert dominance["boundary_pair_failed"] == 0
    assert dominance["boundary_pair_applied"] == 104
    assert dominance["interior_pairs_failed"] == 10
    assert dominance["interior_pairs_applied"] == 318
    census = gate["pass_census"]
    assert census == {
        "checks_applied": 422,
        "explicit_pass": 299,
        "vacuous_zero_pass": 113,
        "checks_failed": 10,
        "classification_rule": census["classification_rule"],
    }
    assert gate["events_checked"] == 1_500_000
    assert gate["charge_balance_exact"] is True
    assert gate["no_fitting"] is True
    assert gate["optimizer_invoked"] is False
    assert gate["sealed"] is True
    refusal = gate["frozen_driver_gate_refusal"]
    assert refusal["attempted"] is True
    assert refusal["exit_code"] == 2
    assert "records checks_failed > 0" in refusal["message"]
    hashes = gate["input_hashes"]
    assert hashes["matching_verification_sha256"] == VERIFICATION_SHA
    assert hashes["pre_registration_sha256"] == PRE_REGISTRATION_SHA
    assert hashes["threshold_metric_module_sha256"] == ENGINE_SHA
    assert hashes["driver_sha256"] == DRIVER_SHA
    assert hashes["provenance_sha256"] == _sha256(RUN_DIR / "provenance.json")
    assert gate["instrument"]["sha256"] == INSTRUMENT_SHA
    if INSTRUMENT_PATH.is_file():
        assert _sha256(INSTRUMENT_PATH) == INSTRUMENT_SHA


# ---------------------------------------------------------------------------
# The clean-environment record
# ---------------------------------------------------------------------------


def test_clean_environment_record():
    _require_run_dir()
    assert _sha256(
        RUN_DIR / "clean_environment_commands.json"
    ) == CLEAN_ENV_SHA
    record = _load("clean_environment_commands.json")
    assert record["schema"] == (
        "na49_mixing.staged_r31_matching_scan.clean_environment_resume/1"
    )
    assert record["resume_card"] == "TASK-20260918-r3-run-r31-verify-resume"
    assert record["seal_verification"] == {
        "files_checked": N_SUMS_ENTRIES,
        "mismatches": [],
    }
    assert record["probe_returncode"] == 1
    assert record["probe_returncode_expected"] is True
    assert record["hashes_match"] is True
    assert record["sealed_report_sha256"] == {
        "matching_verification.json": VERIFICATION_SHA
    }
    assert record["sealed_report_sha256"] == record["recomputed_report_sha256"]
    assert _sha256(
        RUN_DIR / "matching_verification.json"
    ) == VERIFICATION_SHA
    command = record["commands"][0]
    assert command["argv"][1] == "-I"
    assert command["argv"][3] == "--verify"
    assert "python" in Path(command["argv"][0]).name
    refusal = record["frozen_driver_clean_validate"]
    assert refusal["attempted"] is True
    assert refusal["refused"] is True
    assert refusal["exit_code"] == 2
