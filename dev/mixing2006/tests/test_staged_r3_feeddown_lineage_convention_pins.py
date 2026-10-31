"""Round 3 card 3: feed-down lineage convention committed-run artifact pins.

Task ``TASK-20260917-r3-gate2-followup-feeddown-lineage-convention``
(round 3, card 3; the Gate 2 closure obligation, follow-up 3 of the Round
2 closure record; nested feat commit 9e0c337, outer mirror 9240c0e).
This module pins the sealed run evidence under
``dev/00runs/r3_feeddown_lineage_convention_2026-09-17/``: the
pre-registration (master seed 20260917 with ``draws: 0``), the
provenance, the corrected seven-species feed-down table
(``feeddown_corrected.json``), the transition-mismatch trigger record
(``subcard_trigger.json``), the seal manifest, ``SHA256SUMS.txt`` (4
entries, run-dir-relative, ``sha256sum -c`` format), and the post-seal
clean-environment record.

The re-evaluation recomputes the sealed R23 seven-species feed-down
diagnostic read-only over the sealed R23 archive
(``r23_resonance_only_pp_ablation_2026-09-13``, 800,000 events, 1,600,000
realized sources, 18.65% below the kernel boundary) under the corrected
lineage convention: resonant parentage is recorded only for non-root
cascade decays, so the corrected analytic expectation is
``f_res = 1 - (direct-production fraction)`` (the original analytic
chain assumed all final particles carry resonant parentage,
``f_ana = 1`` exactly, zero sigma, ``|z| = Infinity`` across all seven
species).  The committed verdict: all seven species are finite under the
corrected convention, the maximum ``|z|`` is 2.6639608326007873 on
pi_0, every species is within the pre-registered 3.0-sigma tolerance,
the trigger is false, the conditional sub-card
``TASK-20260917-r3-r23-feeddown-transition-nuisance-fit`` stays closed,
and no nuisance fit is executed (``no_fitting``, no optimizer
invocation).  The sealed R23 archive is byte-unchanged.

Sibling of ``tests/test_staged_r3_branching_weight_calibration_pins.py``
and ``tests/test_staged_r3_threshold_metric_reregistration_pins.py``
(the same pin style: sealed-dir discovery, skip-or-fail when the
gitignored archive is absent, digest pins plus structural pins).  These
pins are read-only over the sealed artifacts: nothing is recomputed,
the sealed run is never re-sealed, and the post-seal
clean-environment record is never re-executed.

Pins:

- ``test_sealed_run_dir_inventory``: the run dir holds exactly the seven
  top-level files (the four sealed payloads, the seal manifest,
  ``SHA256SUMS.txt``, the post-seal clean-environment record); nothing
  else is present.
- ``test_pre_registration_structure``: the sealed pre-registration
  carries the committed contract (T0-equivalent deterministic
  re-evaluation, master seed 20260917 with ``draws: 0``, the corrected
  lineage convention with its degenerate-convention correction record,
  the sealed R23 archive reference with its run-card id, the
  pre-registered transition-mismatch criterion (any of the seven
  species with ``|observed - corrected_analytic| > 3.0 * SE_combined``,
  the combined MC/analytic standard error, 3.0-sigma tolerance
  unchanged, ``min_n_per_species = 1000``), the conditional sub-card
  id, the constraint record (``no_fitting``, no optimizer, sealed
  archive read-only, no new Monte Carlo, no draw-sequence change), and
  the two output files).
- ``test_provenance_structure``: the provenance records the catalog
  SHA-256, the sealed R23 archive pre-registration digest and seal
  digest (12 payload files checked), the dual-tree re-evaluation driver
  and R23 driver SHA-256s (the nested and monorepo trees are
  byte-identical), the master seed with zero draws, the git identity,
  and the environment fingerprint.
- ``test_feeddown_corrected_structure``: the corrected table carries
  schema ``na49_mixing.rereval_feeddown_lineage.corrected_feeddown/1``
  and the committed field set: the convention block (the corrected
  analytic expectation, the unchanged tracker, the fix location in the
  R23 driver's analytic chain), the combined SE definition, the pooled
  statistics (8 replicas, 800,000 events, 1,600,000 sources, the
  298,395 below-boundary sources of the sealed mass histogram), the
  seven per-species rows on the uniform 12-key field set (every z
  finite, every ``|z| <= 3.0``, the pass flags, the per-row
  self-consistency: ``z = (observed - analytic) / SE_combined`` at
  ``SE_combined = sqrt(se_mc^2 + se_analytic^2)``), the pinned summary
  (7 checked, max ``|z|`` 2.6639608326007873 on pi_0, empty
  ``species_at_threshold``, ``all_finite``, ``pass``), the
  degenerate-reference block (the sealed R23 ``|z| = Infinity``
  verdict, read-only), and ``no_fitting`` / ``optimizer_invoked``.
- ``test_subcard_trigger_structure``: the trigger record carries the
  pre-registered criterion, the committed decision (trigger false,
  max ``|z|`` 2.6639608326007873 on pi_0, ``species_at_threshold``
  empty, ``subcard_unblocked`` false, the sub-card stays closed and no
  nuisance fit is executed), consistent with the corrected table's
  summary, and ``no_fitting`` / ``optimizer_invoked``.
- ``test_seal_manifest_and_sha256_sums``: ``seal_manifest.json``
  records 4 sealed files totaling 11,920 bytes and the digest of
  ``SHA256SUMS.txt``, which is in ``sha256sum -c`` format with
  run-dir-relative paths, carries the pinned payload digests,
  verifies against every sealed file on disk, and self-consistently
  accounts for the full byte total.
- ``test_clean_environment_record``: the post-seal
  clean-environment record re-verified all 4 sums-covered files
  without mismatches and the clean-interpreter re-run reproduced the
  two sealed report payloads byte-for-byte (the recorded sealed and
  recomputed digests match, equal the pinned digests, and equal the
  digests of the files on disk).
- ``test_sealed_r23_gate_json_byte_unchanged``: the sealed
  ``gate2_r23.json`` matches its pinned SHA-256 digest (recorded in
  the sealed archive's own seal manifest), carries the sealed form
  marked, the degenerate feed-down verdict
  (``feed_down_max_abs_sigma = "Infinity"``, ``feed_down_pass`` false,
  ``pass`` false), and the ``no_branching_fit`` constraint record.
- ``test_sealed_r23_degenerate_state_not_reproducible``: the sealed
  R23 feed-down degenerate state is byte-unchanged (the
  ``comparison_r01.json`` digest pinned; the ``gate2_r23.json``
  ``input_hashes`` agree) and records ``z = "-Infinity"`` with
  ``fraction_resonant_analytic = 1.0`` on all seven species; on the
  same sealed archive the corrected table is finite where the sealed
  report records Infinity (the corrected analytic fraction is not 1.0
  on any species, the observed fractions and final counts mirror the
  sealed degenerate rows bit-exactly, the corrected ``|z|`` maxima
  are finite at 2.6639608326007873, and the trigger stays false).

Hermetic: reads the sealed run dir (gitignored; the artifact is
present on disk) and, for the cross-archive pins, the sealed R23
archive; no computation runs.  If the run dir is absent (a fresh
checkout), the artifact pins skip with a clear message; if the R23
archive is absent while the R3 run dir is present, the cross-archive
pins skip with a clear message.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import pytest

NESTED_ROOT = Path(__file__).resolve().parent.parent
MONO_ROOT = NESTED_ROOT.parent.parent
RUN_NAME = "r3_feeddown_lineage_convention_2026-09-17"
RUN_DIR = MONO_ROOT / "dev" / "00runs" / RUN_NAME

R23_RUN_NAME = "r23_resonance_only_pp_ablation_2026-09-13"
R23_ARCHIVE = MONO_ROOT / "dev" / "00runs" / R23_RUN_NAME

CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
REREVAL_DRIVER_PATH = (
    NESTED_ROOT / "helpers" / "benchmarking" / "rereval_feeddown_lineage.py"
)
REREVAL_DRIVER_OUTER_PATH = (
    MONO_ROOT / "helpers" / "benchmarking" / "rereval_feeddown_lineage.py"
)
R23_DRIVER_PATH = (
    NESTED_ROOT / "helpers" / "benchmarking" / "staged_r23_resonance_ablation.py"
)
R23_DRIVER_OUTER_PATH = (
    MONO_ROOT / "helpers" / "benchmarking" / "staged_r23_resonance_ablation.py"
)

# ---------------------------------------------------------------------------
# The committed run contract (the sealed run realizes it).
# ---------------------------------------------------------------------------
RUN_ID = "R3-FEEDDOWN-LINEAGE-CONVENTION"
CARD_ID = "TASK-20260917-r3-gate2-followup-feeddown-lineage-convention"
R23_CARD_ID = "TASK-20260913-r2-run-r23-resonance-only-pp-ablation"
SUBCARD_ID = "TASK-20260917-r3-r23-feeddown-transition-nuisance-fit"
MASTER_SEED = 20260917
DRAWS = 0
N_SIGMA = 3.0
N_SPECIES = 7
N_SEALED_FILES = 4
SEAL_TOTAL_BYTES = 11_920

#: The seven stable species of the R23 feed-down diagnostic (in table
#: order; the sealed mass-histogram species set).
SPECIES = ["K_plus", "Lambda", "neutron", "pi_0", "pi_minus", "pi_plus",
           "proton"]

#: The committed verdict: all seven species finite under the corrected
#: convention, max |z| = 2.6639608326007873 on pi_0, trigger false.
MAX_ABS_SIGMA = 2.6639608326007873
MAX_ABS_SIGMA_SPECIES = "pi_0"

PRE_SCHEMA = "na49_mixing.rereval_feeddown_lineage.pre_registration/1"
PROV_SCHEMA = "na49_mixing.rereval_feeddown_lineage.provenance/1"
CORRECTED_SCHEMA = "na49_mixing.rereval_feeddown_lineage.corrected_feeddown/1"
TRIGGER_SCHEMA = "na49_mixing.rereval_feeddown_lineage.subcard_trigger/1"
SEAL_SCHEMA = "na49_mixing.rereval_feeddown_lineage.seal_manifest/1"
CLEAN_SCHEMA = "na49_mixing.rereval_feeddown_lineage.clean_environment/1"

R23_GATE_SCHEMA = "na49_mixing.staged_r23_resonance_ablation.gate/1"
R23_COMPARISON_SCHEMA = (
    "na49_mixing.staged_r23_resonance_ablation.comparison/1"
)

#: The uniform per-species field set of the corrected table.
ROW_KEYS = {
    "species", "n_finals", "n_resonant", "fraction_resonant_observed",
    "expected_finals", "expected_resonant", "fraction_resonant_analytic",
    "se_mc", "se_analytic", "se_combined", "z", "within_3_sigma",
}

# ---------------------------------------------------------------------------
# The pinned input digests (the sealed run's own inputs, byte-stable).
# ---------------------------------------------------------------------------
CATALOG_SHA256 = (
    "6e4de7372a5bbf99affd6fa3ccc295c4cef09404dab3b17acd8816cda762bdcd"
)
R23_GATE_SHA256 = (
    "c312c0c47cf6aafb0f1e1f742a6f4491100942e06793158b95370fa1b868834a"
)
R23_COMPARISON_SHA256 = (
    "4c0401f449317851896476600c3f665df57962a3068884e914b159ac7868132f"
)
R23_PRE_REGISTRATION_SHA256 = (
    "52ff0d105aaf52374c61882804fc0d0d5acabcfcd1fc55ea9a5557ffc9c50530"
)
R23_ARCHIVE_SUMS_SHA256 = (
    "def579ffecccc8875140c70b5e68108e4fe010a65f4ca5f13b2e63318d28d1c2"
)
R23_ARCHIVE_SEAL_FILES = 12
REREVAL_DRIVER_SHA256 = (
    "0b500862cdf87be28b2da5066ac4a04cfbb181acda503a1108b46f795a56721d"
)
R23_DRIVER_SHA256 = (
    "b15db02b6260fc2618ad3cb536c4f8ad6502c3cb5efb8dfa62110ea561d808d3"
)

# ---------------------------------------------------------------------------
# The pinned sealed-file digests (the sha256 pin set of this module).
# ---------------------------------------------------------------------------
PRE_REGISTRATION_SHA256 = (
    "b7166493bc4a96bc7fce668ae7b76c83778bdb6888f29b500ddb9b5d71a17f7b"
)
PROVENANCE_SHA256 = (
    "283dbc1b1f641cae9631672a2cfee58a36a737e2d8f665da3acf6713a2bb5fdc"
)
FEEDDOWN_CORRECTED_SHA256 = (
    "f884eda7c9ceba9003f84f3ec6a29517faff564ec46507ecb78c37190e42da09"
)
SUBCARD_TRIGGER_SHA256 = (
    "e946240a74f53ab86418ecaa03c14800f67d17241d63436b93ea9fd90f70daa3"
)
SEAL_MANIFEST_SHA256 = (
    "0f5b6bc2afb50a2eaae73333fd7104632e2746a0ed5a27697ceadb633c6a120b"
)
SUMS_FILE_SHA256 = (
    "6266d3bed7dd3fa5f738c4f1de2b3b7e09d1c0b70dceb23717f4fb56ff441137"
)
CLEAN_ENVIRONMENT_SHA256 = (
    "0ff57a9c9abf2751e0afec32c594f4fad74d81e7e7e277ed9df46ebade4f3156"
)

#: The pinned per-species rows of the corrected table (the sealed
#: verdict, bit-exact).
ROWS_PIN = {
    "K_plus": {
        "n_finals": 1590,
        "n_resonant": 0,
        "fraction_resonant_observed": 0.0,
        "expected_finals": 0.005309673508832923,
        "expected_resonant": 0.0,
        "fraction_resonant_analytic": 0.0,
        "se_mc": 0.0,
        "se_analytic": 0.0,
        "se_combined": 0.0,
        "z": 0.0,
    },
    "Lambda": {
        "n_finals": 1590,
        "n_resonant": 0,
        "fraction_resonant_observed": 0.0,
        "expected_finals": 0.005309673508832923,
        "expected_resonant": 0.0,
        "fraction_resonant_analytic": 0.0,
        "se_mc": 0.0,
        "se_analytic": 0.0,
        "se_combined": 0.0,
        "z": 0.0,
    },
    "neutron": {
        "n_finals": 107288,
        "n_resonant": 18927,
        "fraction_resonant_observed": 0.17641301916337335,
        "expected_finals": 0.3583583731696809,
        "expected_resonant": 0.06206732890901644,
        "fraction_resonant_analytic": 0.1731990475345413,
        "se_mc": 0.0011637092272322224,
        "se_analytic": 0.0011553076765796362,
        "se_combined": 0.0016398033397664652,
        "z": 1.9599738278919312,
    },
    "pi_0": {
        "n_finals": 158081,
        "n_resonant": 74017,
        "fraction_resonant_observed": 0.46822198746212385,
        "expected_finals": 0.5287823992543261,
        "expected_resonant": 0.24508817739501382,
        "fraction_resonant_analytic": 0.4634953389912943,
        "se_mc": 0.001255021751971742,
        "se_analytic": 0.0012542080905918196,
        "se_combined": 0.0017742935305152298,
        "z": 2.6639608326007873,
    },
    "pi_minus": {
        "n_finals": 50800,
        "n_resonant": 20474,
        "fraction_resonant_observed": 0.4030314960629921,
        "expected_finals": 0.16879505187984292,
        "expected_resonant": 0.06807645235300801,
        "fraction_resonant_analytic": 0.4033083410612555,
        "se_mc": 0.0021762728615225563,
        "se_analytic": 0.0021765153246640625,
        "se_combined": 0.0030778860482963116,
        "z": -0.08994647427465856,
    },
    "pi_plus": {
        "n_finals": 158088,
        "n_resonant": 62234,
        "fraction_resonant_observed": 0.3936668184808461,
        "expected_finals": 0.5271534250495239,
        "expected_resonant": 0.2063412689870111,
        "fraction_resonant_analytic": 0.39142545449197486,
        "se_mc": 0.001228770058704334,
        "se_analytic": 0.0012275295946934117,
        "se_combined": 0.001736866363027514,
        "z": 1.2904642732353686,
    },
    "proton": {
        "n_finals": 189517,
        "n_resonant": 47139,
        "fraction_resonant_observed": 0.24873230369834895,
        "expected_finals": 0.6363319533214862,
        "expected_resonant": 0.15652517779995215,
        "fraction_resonant_analytic": 0.2459803833249795,
        "se_mc": 0.0009929774068301665,
        "se_analytic": 0.0009892759992368779,
        "se_combined": 0.0014016672690554221,
        "z": 1.9633192799200807,
    },
}

#: The pinned pooled block (the sealed R23 archive statistics).
POOLED_PIN = {
    "n_replicas": 8,
    "n_events": 800000,
    "n_sources": 1600000,
    "n_new_kernel": 298395,
    "n_finals_new_kernel": 666954,
    "expected_finals_per_new_kernel_source": 2.230040549692526,
    "observed_finals_per_new_kernel_source": 2.235137988237068,
    "histogram_total": 298395,
}

SUMS_TOP_LEVEL_PAYLOADS = (
    "feeddown_corrected.json",
    "pre_registration.json",
    "provenance.json",
    "subcard_trigger.json",
)
SUMS_PAYLOAD_SHA256 = {
    "feeddown_corrected.json": FEEDDOWN_CORRECTED_SHA256,
    "pre_registration.json": PRE_REGISTRATION_SHA256,
    "provenance.json": PROVENANCE_SHA256,
    "subcard_trigger.json": SUBCARD_TRIGGER_SHA256,
}
#: The two sealed report payloads the clean-environment record
#: re-verified byte-for-byte (sealed digest == recomputed digest).
CLEAN_SEALED_REPORT_SHA256 = {
    "feeddown_corrected.json": FEEDDOWN_CORRECTED_SHA256,
    "subcard_trigger.json": SUBCARD_TRIGGER_SHA256,
}

TOP_LEVEL_FILES = (
    "pre_registration.json",
    "provenance.json",
    "feeddown_corrected.json",
    "subcard_trigger.json",
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


def _r23_archive_file(name: str) -> dict:
    return json.loads(
        (R23_ARCHIVE / name).read_text(encoding="utf-8")
    )


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


@pytest.fixture(scope="module")
def r23_archive():
    """The sealed R23 archive (the degenerate-convention reference),
    skipped with a clear message when the gitignored artifact is
    absent while the R3 run dir is present."""
    required = (
        "gate2_r23.json", "comparison_r01.json", "pre_registration.json",
        "seal_manifest.json", "SHA256SUMS.txt",
    )
    missing = [name for name in required if not (R23_ARCHIVE / name).is_file()]
    if missing:
        pytest.skip(
            f"the sealed R23 archive {R23_ARCHIVE} is incomplete "
            f"(missing: " + ", ".join(missing)
            + "); restore it to execute the cross-archive pins"
        )
    return R23_ARCHIVE


# ---------------------------------------------------------------------------
# Sealed artifact pins
# ---------------------------------------------------------------------------
def test_sealed_run_dir_inventory(run_dir):
    """The run dir holds exactly the seven top-level files (the four
    sealed payloads, the seal manifest, the sums file, the post-seal
    clean-environment record); nothing else is present."""
    on_disk = {path.name for path in run_dir.iterdir()}
    assert on_disk == set(TOP_LEVEL_FILES), (
        "the sealed run dir drifted from the committed 7-file top "
        "level set"
    )


def test_pre_registration_structure(run_dir):
    """The sealed pre-registration carries the committed contract: the
    T0-equivalent deterministic re-evaluation (master seed 20260917
    with ``draws: 0``), the corrected lineage convention with its
    degenerate-convention correction record, the sealed R23 archive
    reference with its run-card id, the pre-registered
    transition-mismatch criterion (any of the seven species with
    ``|observed - corrected_analytic| > 3.0 * SE_combined``, the
    combined MC/analytic standard error, 3.0-sigma tolerance
    unchanged, ``min_n_per_species = 1000``), the conditional
    sub-card id, the constraint record, and the two output files."""
    pre = _load("pre_registration.json")
    assert set(pre) == {
        "schema", "run", "card", "mode", "master_seed", "draws",
        "seed_note", "archive", "convention", "criterion", "subcard",
        "constraints", "outputs", "written_before_evaluation",
    }
    assert pre["schema"] == PRE_SCHEMA
    assert pre["run"] == RUN_ID
    assert pre["card"] == CARD_ID
    assert pre["mode"] == (
        "diagnostic re-evaluation over the sealed R23 archive (no "
        "Monte Carlo, no generation)"
    )
    assert pre["master_seed"] == MASTER_SEED
    assert pre["draws"] == DRAWS
    assert "no random draws" in pre["seed_note"]

    assert pre["archive"] == {
        "run": "R23",
        "path": f"dev/00runs/{R23_RUN_NAME}",
        "card": R23_CARD_ID,
        "read_only": True,
    }

    convention = pre["convention"]
    assert set(convention) == {
        "lineage", "corrects", "corrected_expectation", "tracker",
    }
    assert "non-root cascade decays" in convention["lineage"]
    assert "f_ana = 1" in convention["corrects"]
    assert "Infinity" in convention["corrects"]
    assert "f_res = 1 - (direct-production fraction)" in (
        convention["corrected_expectation"]
    )
    assert "unchanged" in convention["tracker"]

    criterion = pre["criterion"]
    assert set(criterion) == {
        "statement", "se_combined", "n_sigma", "tolerance_unchanged",
        "min_n_per_species", "n_species",
    }
    assert "seven species" in criterion["statement"]
    assert "3.0" in criterion["statement"]
    assert "sqrt(se_mc^2 + se_analytic^2)" in criterion["se_combined"]
    assert criterion["n_sigma"] == N_SIGMA
    assert criterion["tolerance_unchanged"] is True
    assert criterion["min_n_per_species"] == 1000
    assert criterion["n_species"] == N_SPECIES

    assert pre["subcard"] == SUBCARD_ID

    constraints = pre["constraints"]
    assert set(constraints) == {
        "no_fitting", "optimizer_invoked", "sealed_archive_read_only",
        "no_new_monte_carlo", "no_draw_sequence_change",
    }
    assert constraints["no_fitting"] is True
    assert constraints["optimizer_invoked"] is False
    assert constraints["sealed_archive_read_only"] is True
    assert constraints["no_new_monte_carlo"] is True
    assert "diagnostic-only" in constraints["no_draw_sequence_change"]

    assert pre["outputs"] == [
        "feeddown_corrected.json",
        "subcard_trigger.json",
    ]
    assert pre["written_before_evaluation"] is True

    # the sealed pre-registration is byte-pinned
    assert _file_sha256(run_dir / "pre_registration.json") == (
        PRE_REGISTRATION_SHA256
    )


def test_provenance_structure(run_dir):
    """The provenance records the catalog SHA-256, the sealed R23
    archive pre-registration digest and seal digest (12 payload files
    checked), the dual-tree re-evaluation driver and R23 driver
    SHA-256s (the nested and monorepo trees are byte-identical), the
    master seed with zero draws, the git identity, and the
    environment fingerprint."""
    prov = _load("provenance.json")
    assert set(prov) == {
        "schema", "run", "card", "written_utc", "master_seed", "draws",
        "catalog", "sealed_archive", "drivers", "git", "environment",
    }
    assert prov["schema"] == PROV_SCHEMA
    assert prov["run"] == RUN_ID
    assert prov["card"] == CARD_ID
    assert UTC_STAMP.match(prov["written_utc"])

    assert prov["catalog"] == {
        "path": "config/v2_singlet/resonance_catalog.json",
        "sha256": CATALOG_SHA256,
    }
    assert prov["catalog"]["sha256"] == _file_sha256(CATALOG_PATH)

    assert prov["sealed_archive"] == {
        "path": f"dev/00runs/{R23_RUN_NAME}",
        "pre_registration_sha256": R23_PRE_REGISTRATION_SHA256,
        "seal": {
            "files_checked": R23_ARCHIVE_SEAL_FILES,
            "sha256_sums_sha256": R23_ARCHIVE_SUMS_SHA256,
        },
    }

    drivers = prov["drivers"]
    assert set(drivers) == {
        "rereval_driver", "rereval_driver_sha256", "r23_driver",
        "r23_driver_sha256",
    }
    assert drivers["rereval_driver"] == (
        "helpers/benchmarking/rereval_feeddown_lineage.py"
    )
    assert drivers["rereval_driver_sha256"] == REREVAL_DRIVER_SHA256
    assert drivers["r23_driver"] == (
        "helpers/benchmarking/staged_r23_resonance_ablation.py"
    )
    assert drivers["r23_driver_sha256"] == R23_DRIVER_SHA256
    # dual-tree drivers: the nested and monorepo copies are
    # byte-identical and carry the pinned digests
    assert _file_sha256(REREVAL_DRIVER_PATH) == REREVAL_DRIVER_SHA256
    assert _file_sha256(REREVAL_DRIVER_OUTER_PATH) == REREVAL_DRIVER_SHA256
    assert _file_sha256(R23_DRIVER_PATH) == R23_DRIVER_SHA256
    assert _file_sha256(R23_DRIVER_OUTER_PATH) == R23_DRIVER_SHA256

    assert prov["master_seed"] == MASTER_SEED
    assert prov["draws"] == DRAWS
    for key in ("nested_head", "nested_dirty_entries",
                "outer_head", "outer_dirty_entries"):
        assert key in prov["git"]
    for key in ("python_version", "python_implementation", "platform",
                "machine", "numpy_version", "executable"):
        assert key in prov["environment"]

    assert _file_sha256(run_dir / "provenance.json") == PROVENANCE_SHA256


def test_feeddown_corrected_structure(run_dir):
    """The corrected table carries schema
    ``na49_mixing.rereval_feeddown_lineage.corrected_feeddown/1`` and
    the committed field set: the convention block (the corrected
    analytic expectation, the unchanged tracker, the fix location in
    the R23 driver's analytic chain), the combined SE definition, the
    pooled statistics, the seven per-species rows on the uniform 12-key
    field set (every z finite, every ``|z| <= 3.0``, the pass flags,
    the per-row self-consistency), the pinned summary (max ``|z|``
    2.6639608326007873 on pi_0, empty ``species_at_threshold``,
    ``all_finite``, ``pass``), the degenerate-reference block, and
    ``no_fitting`` / ``optimizer_invoked``."""
    payload = _load("feeddown_corrected.json")
    assert set(payload) == {
        "schema", "card", "run", "archive", "master_seed", "draws",
        "convention", "se_combined_definition", "pooled", "rows",
        "summary", "degenerate_reference", "no_fitting",
        "optimizer_invoked",
    }
    assert payload["schema"] == CORRECTED_SCHEMA
    assert payload["card"] == CARD_ID
    assert payload["run"] == "R23"
    assert payload["archive"] == {
        "path": f"dev/00runs/{R23_RUN_NAME}",
        "card": R23_CARD_ID,
        "read_only": True,
    }
    assert payload["master_seed"] == MASTER_SEED
    assert payload["draws"] == DRAWS

    convention = payload["convention"]
    assert set(convention) == {
        "lineage", "tracker", "analytic", "fix_location",
    }
    assert "non-root cascade decays" in convention["lineage"]
    assert "unchanged" in convention["tracker"]
    assert "f_res = 1 - (direct-production fraction)" in (
        convention["analytic"]
    )
    assert "staged_r23_resonance_ablation.py" in convention["fix_location"]
    assert "_ChainExpectation" in convention["fix_location"]
    assert "byte-identical" in convention["fix_location"]

    assert "sqrt(se_mc^2 + se_analytic^2)" in payload[
        "se_combined_definition"
    ]

    assert payload["pooled"] == POOLED_PIN

    rows = payload["rows"]
    assert len(rows) == N_SPECIES
    assert [row["species"] for row in rows] == SPECIES
    for row in rows:
        assert set(row) == ROW_KEYS
        species = row["species"]
        pinned = ROWS_PIN[species]
        # the sealed values are bit-exact
        for key, value in pinned.items():
            assert row[key] == value, (
                f"{species}: the sealed {key} drifted from the pinned "
                "value"
            )
        # all z are finite (the acceptance criterion of the card)
        assert math.isfinite(row["z"]), (
            f"{species}: the corrected z is not finite; the "
            "degenerate convention would have reproduced"
        )
        # the pre-registered 3.0-sigma tolerance: every species within
        assert abs(row["z"]) <= N_SIGMA, (
            f"{species}: |z| = {row['z']!r} exceeds the pre-registered "
            "3.0-sigma tolerance"
        )
        assert row["within_3_sigma"] is (abs(row["z"]) <= N_SIGMA)
        # the observed fraction mirrors the raw counts
        assert row["fraction_resonant_observed"] == pytest.approx(
            row["n_resonant"] / row["n_finals"], rel=1e-12, abs=0.0
        )
        # the corrected analytic fraction is the resonant share of the
        # expected finals
        if row["expected_finals"] > 0.0:
            assert row["fraction_resonant_analytic"] == pytest.approx(
                row["expected_resonant"] / row["expected_finals"],
                rel=1e-12,
            )
        # the combined SE and the signed z are self-consistent
        assert row["se_combined"] == pytest.approx(
            math.hypot(row["se_mc"], row["se_analytic"]),
            rel=1e-12, abs=0.0,
        )
        if row["se_combined"] > 0.0:
            assert row["z"] == pytest.approx(
                (row["fraction_resonant_observed"]
                 - row["fraction_resonant_analytic"])
                / row["se_combined"],
                rel=1e-12,
            )
        else:
            assert row["z"] == 0.0

    summary = payload["summary"]
    assert set(summary) == {
        "n_checked", "min_n_per_species", "n_sigma", "max_abs_sigma",
        "max_abs_sigma_species", "species_at_threshold", "all_finite",
        "pass",
    }
    assert summary["n_checked"] == N_SPECIES
    assert summary["min_n_per_species"] == 1000
    assert summary["n_sigma"] == N_SIGMA
    assert summary["max_abs_sigma"] == MAX_ABS_SIGMA
    assert summary["max_abs_sigma_species"] == MAX_ABS_SIGMA_SPECIES
    # self-consistent: the recorded maximum is the maximum of the rows
    row_max = max(abs(row["z"]) for row in rows)
    assert summary["max_abs_sigma"] == row_max
    assert summary["max_abs_sigma_species"] == (
        next(row["species"] for row in rows
             if abs(row["z"]) == row_max)
    )
    assert summary["species_at_threshold"] == []
    assert summary["all_finite"] is True
    assert summary["pass"] is True

    degenerate = payload["degenerate_reference"]
    assert set(degenerate) == {
        "sealed_report", "sealed_feed_down_max_abs_sigma", "note",
    }
    assert degenerate["sealed_report"] == (
        f"dev/00runs/{R23_RUN_NAME}/comparison_r01.json"
    )
    assert degenerate["sealed_feed_down_max_abs_sigma"] == "Infinity"
    assert "read-only" in degenerate["note"]

    assert payload["no_fitting"] is True
    assert payload["optimizer_invoked"] is False

    # the sealed payload is byte-pinned and in canonical form
    assert _file_sha256(run_dir / "feeddown_corrected.json") == (
        FEEDDOWN_CORRECTED_SHA256
    )
    assert _canonical_sha256(payload) == FEEDDOWN_CORRECTED_SHA256


def test_subcard_trigger_structure(run_dir):
    """The trigger record carries the pre-registered criterion, the
    committed decision (trigger false, max ``|z|``
    2.6639608326007873 on pi_0, ``species_at_threshold`` empty,
    ``subcard_unblocked`` false, the sub-card stays closed and no
    nuisance fit is executed), consistent with the corrected table's
    summary, and ``no_fitting`` / ``optimizer_invoked``."""
    trigger = _load("subcard_trigger.json")
    assert set(trigger) == {
        "schema", "card", "criterion", "trigger", "max_abs_sigma",
        "max_abs_sigma_species", "species_at_threshold",
        "n_species_checked", "n_sigma", "subcard", "subcard_unblocked",
        "disposition", "no_fitting", "optimizer_invoked",
    }
    assert trigger["schema"] == TRIGGER_SCHEMA
    assert trigger["card"] == CARD_ID
    assert "seven species" in trigger["criterion"]
    assert "3.0" in trigger["criterion"]

    assert trigger["trigger"] is False
    assert trigger["max_abs_sigma"] == MAX_ABS_SIGMA
    assert trigger["max_abs_sigma_species"] == MAX_ABS_SIGMA_SPECIES
    assert trigger["species_at_threshold"] == []
    assert trigger["n_species_checked"] == N_SPECIES
    assert trigger["n_sigma"] == N_SIGMA
    assert trigger["subcard"] == SUBCARD_ID
    assert trigger["subcard_unblocked"] is False
    assert "stays closed" in trigger["disposition"]
    assert "no nuisance fit is executed" in trigger["disposition"]
    assert trigger["no_fitting"] is True
    assert trigger["optimizer_invoked"] is False

    # consistent with the corrected table's summary: the trigger is
    # the decision rule applied to the recorded maximum
    summary = _load("feeddown_corrected.json")["summary"]
    assert trigger["max_abs_sigma"] == summary["max_abs_sigma"]
    assert trigger["species_at_threshold"] == summary[
        "species_at_threshold"
    ]
    assert trigger["trigger"] is (summary["max_abs_sigma"] > N_SIGMA)

    # the sealed payload is byte-pinned and in canonical form
    assert _file_sha256(run_dir / "subcard_trigger.json") == (
        SUBCARD_TRIGGER_SHA256
    )
    assert _canonical_sha256(trigger) == SUBCARD_TRIGGER_SHA256


def test_seal_manifest_and_sha256_sums(run_dir):
    """``seal_manifest.json`` records the 4 sealed files (the
    pre-registration, the provenance, the corrected table, the
    trigger record; the manifest and the sums file exclude
    themselves) totaling 11,920 bytes and the digest of
    ``SHA256SUMS.txt``, which is in ``sha256sum -c`` format with
    run-dir-relative paths, carries the pinned payload digests,
    verifies against every sealed file on disk, and self-consistently
    accounts for the full byte total."""
    manifest = _load("seal_manifest.json")
    assert set(manifest) == {
        "schema", "run", "card", "created_utc", "file_count",
        "total_bytes", "sha256_sums_file", "sha256_sums_sha256",
        "excluded_from_sums", "note",
    }
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
    """The post-seal clean-environment record re-verified all 4
    sums-covered files without mismatches, and the clean-interpreter
    re-run reproduced the two sealed report payloads byte-for-byte
    (the recorded sealed and recomputed digests match, equal the
    pinned digests, and equal the digests of the files on disk)."""
    record = _load("clean_environment_commands.json")
    assert set(record) == {
        "schema", "run", "card", "verified_utc", "seal_verification",
        "commands", "environment", "probe_environment",
        "sealed_report_sha256", "recomputed_report_sha256",
        "hashes_match",
    }
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
    # the recorded argv names the nested-tree driver and the run dir
    # (the sealed R23 archive is resolved inside the driver)
    assert str(REREVAL_DRIVER_PATH) in argv
    assert str(RUN_DIR) in argv

    assert record["sealed_report_sha256"] == CLEAN_SEALED_REPORT_SHA256
    assert record["recomputed_report_sha256"] == CLEAN_SEALED_REPORT_SHA256
    for name, digest in CLEAN_SEALED_REPORT_SHA256.items():
        assert _file_sha256(run_dir / name) == digest, (
            f"{name}: the on-disk sealed report drifted from the "
            "clean-environment record"
        )

    assert _file_sha256(run_dir / "clean_environment_commands.json") == (
        CLEAN_ENVIRONMENT_SHA256
    )


# ---------------------------------------------------------------------------
# Cross-archive pins (the sealed R23 archive is read-only)
# ---------------------------------------------------------------------------
def test_sealed_r23_gate_json_byte_unchanged(run_dir, r23_archive):
    """The sealed ``gate2_r23.json`` matches its pinned SHA-256 digest
    (recorded in the sealed archive's own seal manifest), carries the
    sealed form marked, the degenerate feed-down verdict
    (``feed_down_max_abs_sigma = "Infinity"``, ``feed_down_pass``
    false, the clause pass false), and the ``no_branching_fit``
    constraint record."""
    gate_file = "gate2_r23.json"
    # the sealed archive's own seal manifest records the gate digest
    seal = _r23_archive_file("seal_manifest.json")
    entries = {entry["path"]: entry for entry in seal["files"]}
    assert entries[gate_file]["sha256"] == R23_GATE_SHA256, (
        f"{gate_file}: the sealed archive's seal manifest entry "
        "drifted from the pinned digest"
    )
    assert entries[gate_file]["bytes"] == (
        (r23_archive / gate_file).stat().st_size
    )
    # the gate JSON is byte-unchanged on disk
    assert _file_sha256(r23_archive / gate_file) == R23_GATE_SHA256, (
        f"{gate_file}: the sealed gate JSON was modified after the "
        "seal; the degenerate-convention verdict must stay untouched"
    )

    gate = _r23_archive_file(gate_file)
    assert gate["schema"] == R23_GATE_SCHEMA
    assert gate["run"] == "R23"
    assert gate["card"] == R23_CARD_ID
    assert gate["sealed"] is True

    # the degenerate feed-down verdict, as recorded by the sealed gate
    # report (the |z| = Infinity state this card corrects)
    observables = gate["clauses"]["observables"]
    assert observables["feed_down_max_abs_sigma"] == "Infinity"
    assert observables["feed_down_pass"] is False
    assert observables["pass"] is False

    assert gate["clauses"]["no_branching_fit"] == {
        "no_fitting": True,
        "optimizer_invoked": False,
    }


def test_sealed_r23_degenerate_state_not_reproducible(run_dir, r23_archive):
    """The sealed R23 feed-down degenerate state is byte-unchanged
    (the ``comparison_r01.json`` digest pinned, the
    ``gate2_r23.json`` ``input_hashes`` agree, the archive sums and
    seal manifest agree) and records ``z = "-Infinity"`` with
    ``fraction_resonant_analytic = 1.0`` on all seven species; on the
    same sealed archive the corrected table is finite where the
    sealed report records Infinity (the corrected analytic fraction
    is not 1.0 on any species, the observed fractions and final
    counts mirror the sealed degenerate rows bit-exactly, the
    corrected ``|z|`` maxima are finite at
    2.6639608326007873, and the trigger stays false)."""
    # the sealed comparison report is byte-unchanged
    comparison_file = "comparison_r01.json"
    assert _file_sha256(r23_archive / comparison_file) == (
        R23_COMPARISON_SHA256
    ), (
        f"{comparison_file}: the sealed comparison report was "
        "modified after the seal"
    )
    # the gate report's input hashes record the same digest
    gate = _r23_archive_file("gate2_r23.json")
    assert gate["input_hashes"]["comparison_r01_sha256"] == (
        R23_COMPARISON_SHA256
    )
    # the archive sums file and the seal manifest agree
    sums_entries = {}
    for line in (r23_archive / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        digest, rel = line.split("  ", 1)
        sums_entries[rel] = digest
    assert sums_entries[comparison_file] == R23_COMPARISON_SHA256
    seal = _r23_archive_file("seal_manifest.json")
    entries = {entry["path"]: entry for entry in seal["files"]}
    assert entries[comparison_file]["sha256"] == R23_COMPARISON_SHA256
    assert entries["SHA256SUMS.txt"]["sha256"] == R23_ARCHIVE_SUMS_SHA256
    assert _file_sha256(r23_archive / "SHA256SUMS.txt") == (
        R23_ARCHIVE_SUMS_SHA256
    )
    # the sealed archive pre-registration digest recorded in the R3
    # provenance matches on disk
    assert _file_sha256(r23_archive / "pre_registration.json") == (
        R23_PRE_REGISTRATION_SHA256
    )

    # the sealed degenerate state: |z| = Infinity on all seven species
    comparison = _r23_archive_file(comparison_file)
    assert comparison["schema"] == R23_COMPARISON_SCHEMA
    feed = comparison["metrics"]["feed_down"]
    assert set(feed) == {"summary", "rows"}
    assert feed["summary"] == {
        "n_checked": N_SPECIES,
        "min_n_per_species": 1000,
        "histogram_total": POOLED_PIN["histogram_total"],
        "max_abs_sigma": "Infinity",
        "n_sigma": N_SIGMA,
        "pass": False,
    }
    degenerate_rows = feed["rows"]
    assert len(degenerate_rows) == N_SPECIES
    assert [row["species"] for row in degenerate_rows] == SPECIES
    for row in degenerate_rows:
        assert set(row) == {
            "species", "n_finals", "fraction_resonant_observed",
            "fraction_resonant_analytic", "sigma", "z",
        }
        assert row["z"] == "-Infinity"
        assert row["fraction_resonant_analytic"] == 1.0
        assert row["sigma"] == 0.0

    # on the same archive, the corrected table is finite where the
    # sealed report records Infinity
    corrected = _load("feeddown_corrected.json")
    corrected_rows = {row["species"]: row for row in corrected["rows"]}
    for degenerate in degenerate_rows:
        species = degenerate["species"]
        row = corrected_rows[species]
        # the tracker side is unchanged: the observed fractions and
        # final counts mirror the sealed degenerate rows bit-exactly
        assert row["n_finals"] == degenerate["n_finals"]
        assert row["fraction_resonant_observed"] == (
            degenerate["fraction_resonant_observed"]
        )
        # the corrected z is finite (the sealed state is not
        # reproducible under the corrected convention)
        assert math.isfinite(row["z"]), (
            f"{species}: the corrected z is infinite; the degenerate "
            "state was reproduced under the corrected convention"
        )
        # the corrected analytic fraction is not the degenerate
        # f_ana = 1
        assert row["fraction_resonant_analytic"] != 1.0
    # the corrected maximum is the pinned finite value (the sealed
    # maximum is the string "Infinity")
    assert max(abs(row["z"]) for row in corrected["rows"]) == (
        MAX_ABS_SIGMA
    )
    assert corrected["summary"]["max_abs_sigma_species"] == (
        MAX_ABS_SIGMA_SPECIES
    )
    assert corrected["summary"]["all_finite"] is True
    assert _load("subcard_trigger.json")["trigger"] is False
