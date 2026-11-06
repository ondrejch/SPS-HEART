"""Round 3 card 1: open-weight branching weight calibration committed-run
artifact pins.

Task ``TASK-20260917-r3-gate2-followup-branching-weight-calibration``
(round 3, card 1; the Gate 2 closure obligation, follow-up 1 of the
Round 2 closure record; nested feat commit 812cec8, outer mirror
15917ec).  This module pins the sealed run evidence under
``dev/00runs/r3_branching_weight_calibration_2026-09-17/``: the
pre-registration (master seed 20260917 with ``draws: 0``), the
provenance, the 80 recomputed open-weight expectation files under
``expectations/``, the before- and after-sigma tables, the calibration
report, the disposition of the 39 residual states, the seal manifest,
``SHA256SUMS.txt`` (86 entries, run-dir-relative, ``sha256sum -c``
format), and the post-seal clean-environment record.

The calibration recomputed the per-state, per-channel open-weight
expectations from the committed catalog through the committed
``staged_r20_unit_grid.py`` expected-value mirror and compared them
read-only against the sealed R20 sampled channel shares at the
pre-registered 3.0-sigma per-cell binomial tolerance.  The committed
verdict: the recomputed expectations reproduce the sealed R20
per-state sigma table bit-exactly on all 80 states, so scope clause (d)
does not trigger, no convention fix lands, and the 39 failing states
stay named in ``disposition.json`` as findings (the before- and
after-tables are identical modulo their ``tag`` field).

Sibling of ``tests/test_staged_r20_unit_grid_card10_pins.py`` and
``tests/test_staged_r23_resonance_ablation_pins.py`` (the same pin
style: sealed-dir discovery, skip-or-fail when the gitignored
archive is absent, digest pins plus structural pins).  These pins are
read-only over the sealed artifacts: nothing is recomputed, the sealed
run is never re-sealed, and the post-seal clean-environment record is
never re-executed.

Pins:

- ``test_sealed_run_dir_inventory``: the run dir holds exactly the nine
  top-level files (pre-registration, provenance, the before/after
  sigma tables, the calibration report, the disposition, the seal
  manifest, ``SHA256SUMS.txt``, the clean-environment record) plus the
  ``expectations/`` subdirectory with exactly the 80 recomputed
  ``state_<pdg>.json`` files of the pre-registered state set; nothing
  else is present.
- ``test_pre_registration_structure``: the sealed pre-registration
  carries the committed contract (T0-equivalent deterministic
  diagnostic, master seed 20260917 with ``draws: 0``, the six-step
  scope, the 80-state coverage set, the 3.0-sigma binomial comparison
  convention, the committed catalog / sealed R20 archive / dual-tree
  driver inputs with their digests, and the output list).
- ``test_provenance_structure``: the provenance records the catalog
  SHA-256, the sealed R20 archive sums digest, the dual-tree driver
  and expected-value-mirror digests (the nested and monorepo trees are
  byte-identical), the master seed with zero draws, the git identity,
  and the environment fingerprint.
- ``test_sigma_table_sha256_pins``: the before- and after-sigma tables
  match their pinned SHA-256 digests and are exactly the canonical
  indent-2 JSON bytes (the driver's canonical report form).
- ``test_sigma_table_structure``: each table carries schema
  ``na49_mixing.calibrate_branching_weights.sigma_table/1``, the 80
  per-state entries on the committed field set (ranks 0..79, the
  59 important / 21 minor split with the 10,000/2,000 per-point
  counts, the sealed R20 gate criterion recomputed from the three
  per-state maxima, the all-true ``matches_sealed_gate2`` flags, the
  per-state diagnostics block), and the pinned totals (41 passing,
  39 failing, worst 4.913494961192409 sigma on pdg -122112, the 28,019
  cells, the 40.229479945611864 expected failing states of the exact
  binomial null, and the branching-maximum state tail).
- ``test_tables_identical_no_fix``: the before- and after-tables are
  identical modulo their ``tag`` field (no convention fix was
  applied, so the calibration moved nothing).
- ``test_expectation_file_schema``: every one of the 80 recomputed
  expectation files carries the committed field set: the state block
  byte-identical to the sealed R20 state file, the 100-point R20 grid
  with identical masses, the degenerate-point inventory, and the
  per-point open weights (the expected shares are the normalized
  weights summing to 1, the card-5/6 class shares are within-class
  normalized over their member charge combinations, and the
  open-channel cell count feeds the sigma-table diagnostics).
- ``test_sigma_tables_match_sealed_r20_gate2``: for all 80 states the
  recomputed branching, charge, and angular maxima agree bit-exactly
  with the sealed R20 Gate 2 report, the pass verdicts, the ranks,
  the classifications, the pole points, the degenerate-point
  inventories, and the channel counts agree with the sealed R20
  pre-registration and state files.
- ``test_disposition_structure``: ``disposition.json`` carries
  ``fix_applied: false`` on every entry, the 39 named residual states
  exactly the failing states of the sigma tables, every entry
  recorded as a finding (``repaired_states`` empty, no branching-
  fraction change), and the PDG -2224 entry with its angular-
  kernel-only diagnosis (branching residual identically zero, the
  single channel has no branching cell with variance, the 3.946
  sigma angular-kernel residual is outside the open-weight scope).
- ``test_calibration_report_structure``: the report carries the four
  checked diagnosis dimensions, the statistical verdict consistent
  with the sigma-table totals (28,019 cells, mean square 0.990895,
  the 40.2295 expected failing states of the binomial null against
  the 39 observed), the clause-(d) no-fix decision, and the five
  constraint-record flags (no branching fit, catalog byte-unchanged,
  R20 archive read-only, nominal widths and barrier polynomial
  unchanged).
- ``test_seal_manifest_and_sha256_sums``: ``seal_manifest.json``
  records 86 sealed files totaling 7,103,071 bytes and the digest of
  ``SHA256SUMS.txt``, which is in ``sha256sum -c`` format with
  run-dir-relative paths, carries the six pinned payload digests,
  verifies against every sealed file on disk, and self-consistently
  accounts for the full byte total.
- ``test_clean_environment_record``: the post-seal clean-environment
  record re-verified all 86 sums-covered files without mismatches and
  the clean-interpreter re-run reproduced the four sealed report
  payloads byte-for-byte (the recorded sealed and recomputed digests
  match, equal the pinned digests, and equal the digests of the files
  on disk).

Hermetic: reads the sealed run dir (gitignored; the artifact is
present on disk) and, for the cross-archive pins, the sealed R20
archive; no computation runs.  If the run dir is absent (a fresh
checkout), the artifact pins skip with a clear message; if the R20
archive is absent while the R3 run dir is present, the cross-archive
pins skip with a clear message.
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

import pytest

NESTED_ROOT = Path(__file__).resolve().parent.parent
MONO_ROOT = NESTED_ROOT.parent.parent
RUN_NAME = "r3_branching_weight_calibration_2026-09-17"
RUN_DIR = MONO_ROOT / "dev" / "00runs" / RUN_NAME

R20_RUN_NAME = "r20_single_resonance_unit_grid_2026-09-13"
R20_ARCHIVE = MONO_ROOT / "dev" / "00runs" / R20_RUN_NAME

CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
DRIVER_PATH = NESTED_ROOT / "helpers" / "benchmarking" / "calibrate_branching_weights.py"
MIRROR_PATH = NESTED_ROOT / "helpers" / "benchmarking" / "staged_r20_unit_grid.py"
MIRROR_OUTER_PATH = MONO_ROOT / "helpers" / "benchmarking" / "staged_r20_unit_grid.py"
DRIVER_OUTER_PATH = MONO_ROOT / "helpers" / "benchmarking" / "calibrate_branching_weights.py"

# ---------------------------------------------------------------------------
# The committed run contract (the sealed run realizes it).
# ---------------------------------------------------------------------------
RUN_ID = "R3-BRANCHING-WEIGHT-CALIBRATION"
CARD_ID = "TASK-20260917-r3-gate2-followup-branching-weight-calibration"
MASTER_SEED = 20260917
DRAWS = 0
N_SIGMA = 3.0
N_STATES = 80
N_IMPORTANT = 59
N_MINOR = 21
N_FAILING = 39
N_PASSING = N_STATES - N_FAILING
N_SEALED_FILES = 86  # 80 expectation files + 6 payloads
SEAL_TOTAL_BYTES = 7_103_071
N_SUMS_COVERED_TOP_LEVEL = 6  # everything in the sums except expectations/
TOTAL_BUDGET_DECAYS = "63,110,000"

#: The worst state of the sealed run (the card's own citation):
#: Nbar(1650)0, PDG -122112, worst 4.913 sigma on the branching cells.
WORST_STATE_PD_G = -122112
WORST_BRANCHING_ABS_SIGMA = 4.913494961192409
#: The PDG -2224 (aDelta--) residual: angular-kernel-only diagnosis.
ANGULAR_ONLY_STATE_PD_G = -2224
ANGULAR_ONLY_STATE_NAME = "aDelta--"
ANGULAR_ONLY_MAX_ABS_SIGMA = 3.9464484879178423

PRE_SCHEMA = "na49_mixing.calibrate_branching_weights.pre_registration/1"
PROV_SCHEMA = "na49_mixing.calibrate_branching_weights.provenance/1"
EXPECT_SCHEMA = "na49_mixing.calibrate_branching_weights.expectation/1"
TABLE_SCHEMA = "na49_mixing.calibrate_branching_weights.sigma_table/1"
REPORT_SCHEMA = "na49_mixing.calibrate_branching_weights.calibration_report/1"
DISPOSITION_SCHEMA = "na49_mixing.calibrate_branching_weights.disposition/1"
SEAL_SCHEMA = "na49_mixing.calibrate_branching_weights.seal_manifest/1"
CLEAN_SCHEMA = "na49_mixing.calibrate_branching_weights.clean_environment/1"

# ---------------------------------------------------------------------------
# The pinned input digests (the sealed run's own inputs, byte-stable).
# ---------------------------------------------------------------------------
CATALOG_SHA256 = (
    "6e4de7372a5bbf99affd6fa3ccc295c4cef09404dab3b17acd8816cda762bdcd"
)
R20_PRE_REGISTRATION_SHA256 = (
    "1fe966cab3926fa2c4accac510b62bfe428f1253f6c9259059abf0f985c7515a"
)
R20_ARCHIVE_SUMS_SHA256 = (
    "1d74383d962098ca7c6a9078f36cca3230c6972131774aae2b8fc29572b9f596"
)
DRIVER_SHA256 = (
    "b40b17b5fb28e448bc49e65cac112a862b3b2cf3753be41bf7ff3638a0efd214"
)
MIRROR_SHA256 = (
    "ad992d21b222be1e6a819f76ff15984ed3dde9003a7528cefba57fc193fa3f6a"
)

# ---------------------------------------------------------------------------
# The pinned sealed-file digests (the sha256 pin set of this module).
# ---------------------------------------------------------------------------
PRE_REGISTRATION_SHA256 = (
    "129f71874b0a02625ce773ec71f36022c0d193275a157b785f522ff6549b5c67"
)
PROVENANCE_SHA256 = (
    "3533ad213524c0930f0d18fabd52395d99241d43c7c43588cc5a4835cba3cd42"
)
SIGMA_TABLE_BEFORE_SHA256 = (
    "95ed3d3b61f1076758bcea5a8a96424f505644be728964b08416f57f1a6ef900"
)
SIGMA_TABLE_AFTER_SHA256 = (
    "212944fc99c73b8ac9951b5fece2736190b833a29575e304a9c28ba113069ee4"
)
CALIBRATION_REPORT_SHA256 = (
    "f917db0f7161aedce26c8d1d9488be48790c02b9cd8ac0c805bd7afc28201614"
)
DISPOSITION_SHA256 = (
    "c1405521b32c5d335e83857dee80ee986e5db7eeeb9dd9533cac95467709fd75"
)
SEAL_MANIFEST_SHA256 = (
    "fda08c8775facbb03012d9107c12e0bb83a95f273d450772ef5c15e4d7fedbd4"
)
SUMS_FILE_SHA256 = (
    "bc88db819b8d97446d28ca70b8250b4468b8166e77e38f4b32a5fe72233214bb"
)
CLEAN_ENVIRONMENT_SHA256 = (
    "430ece94cfea107cbd16ac45856a7e39bce3d30c2865a013cef829c115faad1c"
)

#: The 80 pre-registered state pdg ids (the sealed R20 decaying set,
#: every catalog record with ``stable_for_decay = false``).
PDG_IDS = [
    2224, 2214, 2114, 1114, -1114, -2114, -2214, -2224, 12112, 12212,
    -12112, 2116, 2216, -2116, 22112, 22212, -22112, 102112, 102212,
    -102112, 122112, 122212, -122112, 212114, 212214, -212114, -212214,
    202224, 202214, 202114, 201114, 122224, 122214, 122114, 121114, 113,
    213, -213, 100113, 100213, -100213, 30113, 30213, -30213, 223, 100223,
    30223, 221, 377, 10113, 10213, -10213, 20113, 20213, -20213, 115, 215,
    -215, 225, 117, -217, 217, 229, 323, -323, 313, -313, 10323, -10323,
    10313, -10313, 325, -325, 315, -315, 333, 102134, 3224, 3214, 3114,
]

# The sigma-table per-state field sets (uniform across all 80 entries).
PER_STATE_KEYS = {
    "name", "rank", "classification", "pole_point", "n_open_points",
    "n_degenerate_points", "n_channels", "decays_per_point",
    "branching_pole_max_abs_sigma", "branching_open_max_abs_sigma",
    "branching_max_abs_sigma", "charge_max_abs_sigma",
    "angular_max_abs_sigma", "gate2_pass", "matches_sealed_gate2",
    "diagnostics",
}
DIAGNOSTICS_KEYS = {
    "n_cells", "n_cells_with_variance", "sum_abs_z", "sum_z2",
    "mean_abs_z", "mean_z2", "n_small_count_cells", "worst_cell",
    "channel_pooled_max_abs_t", "channel_pooled_t",
    "expected_state_fail_probability",
}
WORST_CELL_KEYS = {
    "point_j", "channel", "expected_share", "sampled_fraction",
    "abs_sigma", "expected_count_np", "small_count",
}
MATCHES_SEALED_GATE2_KEYS = {
    "branching_max_abs_sigma", "charge_max_abs_sigma",
    "angular_max_abs_sigma",
}
#: The pinned sigma-table totals (identical in the before- and
#: after-tables: no fix was applied).
TOTALS_PIN = {
    "n_states": N_STATES,
    "n_states_pass": N_PASSING,
    "n_states_fail": N_FAILING,
    "worst_branching_abs_sigma": WORST_BRANCHING_ABS_SIGMA,
    "worst_state_pdg": WORST_STATE_PD_G,
    "n_cells": 28_019,
    "n_cells_with_variance": 26_418,
    "mean_abs_z_cells": 0.7936519531827061,
    "mean_z2_cells": 0.9908950417650979,
    "n_small_count_cells": 2_131,
    "expected_failing_states_binomial_null": 40.229479945611864,
    "state_maxima_tail": {
        "states_max_abs_sigma_ge_3": 34,
        "states_max_abs_sigma_ge_3.5": 15,
        "states_max_abs_sigma_ge_4": 7,
        "states_max_abs_sigma_ge_4.5": 2,
    },
}

# The disposition per-state field set (uniform across all 39 entries).
DISPOSITION_PER_STATE_KEYS = {
    "name", "classification", "n_channels", "worst_branching_abs_sigma",
    "worst_charge_abs_sigma", "worst_angular_abs_sigma", "worst_cell",
    "channel_pooled_max_abs_t", "expected_state_fail_probability",
    "fix_applied", "disposition", "diagnosis",
}

#: The six payloads of the run dir that carry a digest in
#: ``SHA256SUMS.txt`` (everything sealed except the 80 expectation
#: files, the seal manifest, and the sums file itself; the
#: clean-environment record is written after the seal and is not in
#: the sums).
SUMS_TOP_LEVEL_PAYLOADS = (
    "calibration_report.json",
    "disposition.json",
    "pre_registration.json",
    "provenance.json",
    "sigma_table_after.json",
    "sigma_table_before.json",
)
#: The pinned digest of each sums-covered top-level payload.
SUMS_PAYLOAD_SHA256 = {
    "calibration_report.json": CALIBRATION_REPORT_SHA256,
    "disposition.json": DISPOSITION_SHA256,
    "pre_registration.json": PRE_REGISTRATION_SHA256,
    "provenance.json": PROVENANCE_SHA256,
    "sigma_table_after.json": SIGMA_TABLE_AFTER_SHA256,
    "sigma_table_before.json": SIGMA_TABLE_BEFORE_SHA256,
}
#: The four sealed report payloads the clean-environment record
#: re-verified byte-for-byte (sealed digest == recomputed digest).
CLEAN_SEALED_REPORT_SHA256 = {
    "sigma_table_before.json": SIGMA_TABLE_BEFORE_SHA256,
    "sigma_table_after.json": SIGMA_TABLE_AFTER_SHA256,
    "calibration_report.json": CALIBRATION_REPORT_SHA256,
    "disposition.json": DISPOSITION_SHA256,
}

TOP_LEVEL_FILES = (
    "pre_registration.json",
    "provenance.json",
    "sigma_table_before.json",
    "sigma_table_after.json",
    "calibration_report.json",
    "disposition.json",
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


def _expectation_path(pd_g: int) -> Path:
    return RUN_DIR / "expectations" / f"state_{pd_g}.json"


@lru_cache(maxsize=1)
def _open_channel_cells() -> dict[int, int]:
    """Per state, the number of branching cells: one per open point per
    channel that is threshold-open at the point mass (channels below
    threshold carry no cell), summed over the open grid points of the
    recomputed expectation file."""
    out: dict[int, int] = {}
    for pd_g in PDG_IDS:
        expectation = json.loads(
            _expectation_path(pd_g).read_text(encoding="utf-8")
        )
        out[pd_g] = sum(
            len(point["expected_shares"])
            for point in expectation["points"].values()
            if not point["degenerate"]
        )
    return out


@lru_cache(maxsize=1)
def _sigma_tables() -> dict[str, dict]:
    return {
        tag: _load(f"sigma_table_{tag}.json")
        for tag in ("before", "after")
    }


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
    pdg_ids = json.loads(
        (RUN_DIR / "pre_registration.json").read_text(encoding="utf-8")
    )["state_coverage"]["pdg_ids"]
    missing += [
        f"expectations/state_{pd_g}.json"
        for pd_g in pdg_ids
        if not _expectation_path(pd_g).is_file()
    ]
    if missing:
        pytest.skip(
            f"the sealed R3 run dir {RUN_DIR} is incomplete (missing: "
            + ", ".join(missing[:4]) + f" and {len(missing) - 4} more); "
            "restore the sealed run to execute the artifact pins"
        )
    return RUN_DIR


@pytest.fixture(scope="module")
def pre_registration(run_dir) -> dict:
    return _load("pre_registration.json")


@pytest.fixture(scope="module")
def r20_archive():
    """The sealed R20 archive (the sealed sampled-share input),
    skipped with a clear message when the gitignored artifact is
    absent while the R3 run dir is present."""
    if not (R20_ARCHIVE / "gate2_r20.json").is_file():
        pytest.skip(
            f"the sealed R20 archive {R20_ARCHIVE} is absent "
            "(gitignored artifact); restore it to execute the "
            "cross-archive pins"
        )
    return R20_ARCHIVE


# ---------------------------------------------------------------------------
# Sealed artifact pins
# ---------------------------------------------------------------------------
def test_sealed_run_dir_inventory(run_dir):
    """The run dir holds exactly the nine top-level files plus the
    ``expectations/`` subdirectory with exactly the 80 recomputed
    ``state_<pdg>.json`` files of the pre-registered state set;
    nothing else is present."""
    on_disk = {path.name for path in run_dir.iterdir()}
    assert on_disk == set(TOP_LEVEL_FILES) | {"expectations"}, (
        "the sealed run dir drifted from the committed 9-file top "
        "level set plus expectations/"
    )
    expectations = sorted(
        path.name for path in (run_dir / "expectations").iterdir()
    )
    assert len(expectations) == N_STATES
    assert expectations == sorted(
        f"state_{pd_g}.json" for pd_g in PDG_IDS
    ), (
        "the expectations/ inventory drifted from the pre-registered "
        "80-state set"
    )


def test_pre_registration_structure(run_dir, pre_registration):
    """The sealed pre-registration carries the committed contract: the
    T0-equivalent deterministic diagnostic (master seed 20260917 with
    ``draws: 0``), the six-step scope, the 80-state coverage set, the
    3.0-sigma binomial comparison convention against the sealed R20
    shares, the committed catalog / sealed R20 archive / dual-tree
    driver inputs with their digests, and the output list."""
    pre = pre_registration
    assert pre["schema"] == PRE_SCHEMA
    assert pre["run"] == RUN_ID
    assert pre["card"] == CARD_ID
    assert pre["written_before_computation"] is True

    stats = pre["statistics"]
    assert set(stats) == {"tier", "master_seed", "draws", "note"}
    assert stats["tier"] == "T0-equivalent deterministic diagnostic"
    assert stats["master_seed"] == MASTER_SEED
    assert stats["draws"] == DRAWS
    assert "no new Monte Carlo" in stats["note"]
    assert TOTAL_BUDGET_DECAYS in stats["note"]

    assert set(pre["scope"]) == {
        "recompute", "compare", "diagnose", "fix_if_identified",
        "re_verify", "report",
    }
    assert "plan section 8.2" in pre["objective"]
    assert "plan section 24" in pre["objective"]
    assert "external PDG inputs" in pre["objective"]

    coverage = pre["state_coverage"]
    assert set(coverage) == {"rule", "n_states", "pdg_ids"}
    assert "stable_for_decay = false" in coverage["rule"]
    assert coverage["n_states"] == N_STATES
    assert len(coverage["pdg_ids"]) == N_STATES
    assert len(set(coverage["pdg_ids"])) == N_STATES
    assert coverage["pdg_ids"] == PDG_IDS

    comparison = pre["comparison"]
    assert set(comparison) == {
        "n_sigma", "sigma_convention", "gate_criterion",
        "sampled_inputs",
    }
    assert comparison["n_sigma"] == N_SIGMA
    assert "sqrt(p (1 - p) / N)" in comparison["sigma_convention"]
    assert "branching, charge, and angular" in comparison["gate_criterion"]
    assert TOTAL_BUDGET_DECAYS in comparison["sampled_inputs"]
    assert "read-only" in comparison["sampled_inputs"]

    inputs = pre["inputs"]
    assert set(inputs) == {"catalog", "r20_archive", "drivers"}
    assert inputs["catalog"]["path"] == (
        "config/v2_singlet/resonance_catalog.json"
    )
    assert inputs["catalog"]["sha256"] == CATALOG_SHA256
    assert inputs["catalog"]["sha256"] == _file_sha256(CATALOG_PATH)
    assert inputs["r20_archive"]["path"] == (
        f"dev/00runs/{R20_RUN_NAME}"
    )
    assert inputs["r20_archive"]["pre_registration_sha256"] == (
        R20_PRE_REGISTRATION_SHA256
    )
    assert inputs["drivers"] == {
        "calibration_driver":
            "helpers/benchmarking/calibrate_branching_weights.py",
        "expected_value_mirror": "staged_r20_unit_grid.py",
    }

    assert pre["outputs"] == [
        "expectations/state_<pdg>.json (80 files)",
        "sigma_table_before.json",
        "sigma_table_after.json",
        "calibration_report.json",
        "disposition.json",
    ]

    # the sealed pre-registration is byte-pinned
    assert _file_sha256(run_dir / "pre_registration.json") == (
        PRE_REGISTRATION_SHA256
    )


def test_provenance_structure(run_dir, pre_registration):
    """The provenance records the catalog SHA-256, the sealed R20
    archive sums digest, the dual-tree driver and expected-value-mirror
    digests (the nested tree and the monorepo mirror are
    byte-identical), the master seed with zero draws, the git
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
    assert prov["r20_archive"] == {
        "path": f"dev/00runs/{R20_RUN_NAME}",
        "sha256_sums": R20_ARCHIVE_SUMS_SHA256,
    }

    drivers = prov["drivers"]
    assert set(drivers) == {
        "calibration_driver", "calibration_driver_sha256",
        "expected_value_mirror", "expected_value_mirror_sha256",
    }
    assert drivers["calibration_driver"] == (
        "helpers/benchmarking/calibrate_branching_weights.py"
    )
    assert drivers["calibration_driver_sha256"] == DRIVER_SHA256
    assert drivers["expected_value_mirror"] == "staged_r20_unit_grid.py"
    assert drivers["expected_value_mirror_sha256"] == MIRROR_SHA256
    # dual-tree drivers: the nested and monorepo copies are
    # byte-identical and carry the pinned digests
    assert _file_sha256(DRIVER_PATH) == DRIVER_SHA256
    assert _file_sha256(DRIVER_OUTER_PATH) == DRIVER_SHA256
    assert _file_sha256(MIRROR_PATH) == MIRROR_SHA256
    assert _file_sha256(MIRROR_OUTER_PATH) == MIRROR_SHA256

    assert prov["master_seed"] == MASTER_SEED
    assert prov["draws"] == DRAWS
    for key in ("nested_head", "nested_dirty_entries",
                "outer_head", "outer_dirty_entries"):
        assert key in prov["git"]
    for key in ("python_version", "python_implementation", "platform",
                "machine", "numpy_version", "executable"):
        assert key in prov["environment"]

    assert _file_sha256(run_dir / "provenance.json") == PROVENANCE_SHA256


def test_sigma_table_sha256_pins(run_dir):
    """The before- and after-sigma tables match their pinned SHA-256
    digests and are exactly the canonical indent-2 JSON bytes (the
    driver's canonical report form)."""
    for name, pinned in (
        ("sigma_table_before.json", SIGMA_TABLE_BEFORE_SHA256),
        ("sigma_table_after.json", SIGMA_TABLE_AFTER_SHA256),
    ):
        path = run_dir / name
        assert _file_sha256(path) == pinned, (
            f"{name}: the sealed file drifted from the pinned "
            "SHA-256; the run dir was modified after the seal"
        )
        assert _canonical_sha256(_load(name)) == pinned, (
            f"{name}: the sealed file is not the canonical "
            "indent-2 JSON form"
        )


def test_sigma_table_structure(run_dir):
    """Each sigma table carries the committed field set: the 80
    per-state entries (ranks 0..79, the 59 important / 21 minor split
    with the 10,000/2,000 per-point counts, the sealed R20 gate
    criterion recomputed from the three per-state maxima, the
    all-true ``matches_sealed_gate2`` flags, the per-state
    diagnostics) and the pinned totals (41 passing, 39 failing,
    worst 4.913 sigma on pdg -122112, the 28,019 cells, the 40.2295
    expected failing states of the exact binomial null, and the
    branching-maximum state tail)."""
    for tag, pinned in (
        ("before", SIGMA_TABLE_BEFORE_SHA256),
        ("after", SIGMA_TABLE_AFTER_SHA256),
    ):
        table = _load(f"sigma_table_{tag}.json")
        assert table["schema"] == TABLE_SCHEMA
        assert table["run"] == RUN_ID
        assert table["card"] == CARD_ID
        assert table["tag"] == tag
        assert table["n_sigma"] == N_SIGMA
        assert "binomial" in table["comparison_convention"]
        assert "3.0 sigma per cell" in table["comparison_convention"]

        sources = table["sources"]
        assert set(sources) == {"catalog", "r20_archive",
                                "expectations_dir"}
        assert sources["catalog"]["sha256"] == CATALOG_SHA256
        assert sources["r20_archive"]["sha256_sums"] == (
            R20_ARCHIVE_SUMS_SHA256
        )
        assert sources["expectations_dir"] == "expectations"

        per_state = table["per_state"]
        assert len(per_state) == N_STATES
        assert set(per_state) == {str(pd_g) for pd_g in PDG_IDS}
        ranks = [entry["rank"] for entry in per_state.values()]
        assert sorted(ranks) == list(range(N_STATES))
        n_important = 0
        for str_pdg, entry in per_state.items():
            assert set(entry) == PER_STATE_KEYS
            assert entry["classification"] in ("important", "minor")
            expected_dpp = {
                "important": 10_000, "minor": 2_000,
            }[entry["classification"]]
            assert entry["decays_per_point"] == expected_dpp
            n_important += entry["classification"] == "important"
            assert entry["n_open_points"] + entry["n_degenerate_points"] == 100
            assert entry["n_open_points"] > 0
            assert entry["n_channels"] >= 1
            for key in ("branching_pole_max_abs_sigma",
                        "branching_open_max_abs_sigma",
                        "branching_max_abs_sigma", "charge_max_abs_sigma",
                        "angular_max_abs_sigma"):
                assert 0.0 <= entry[key]
            # the sealed R20 per-state gate criterion: the maxima of
            # the branching, charge, and angular cells against 3.0
            # sigma
            assert entry["gate2_pass"] == bool(
                entry["branching_max_abs_sigma"] <= N_SIGMA
                and entry["charge_max_abs_sigma"] <= N_SIGMA
                and entry["angular_max_abs_sigma"] <= N_SIGMA
            )
            # the recomputed maxima are bit-identical to the sealed
            # R20 gate table on every state
            assert entry["matches_sealed_gate2"] == {
                key: True for key in MATCHES_SEALED_GATE2_KEYS
            }
            diagnostics = entry["diagnostics"]
            assert set(diagnostics) == DIAGNOSTICS_KEYS
            assert set(diagnostics["worst_cell"]) == WORST_CELL_KEYS
            # the branching cells: one per open point per channel that is
            # threshold-open at the point mass (channels below threshold
            # carry no cell), so the count sits between the per-point
            # minimum and the full per-point maximum and equals the
            # recomputed expectation files' open-channel count
            n_open = entry["n_open_points"]
            n_chan = entry["n_channels"]
            assert n_open <= diagnostics["n_cells"] <= n_open * n_chan
            assert diagnostics["n_cells"] == _open_channel_cells()[
                int(str_pdg)
            ]
            assert 0 <= diagnostics["n_cells_with_variance"] <= (
                diagnostics["n_cells"]
            )
            if diagnostics["n_cells_with_variance"]:
                assert diagnostics["mean_z2"] == pytest.approx(
                    diagnostics["sum_z2"]
                    / diagnostics["n_cells_with_variance"],
                    rel=1e-12,
                )
            else:
                assert diagnostics["mean_z2"] == 0.0
                assert diagnostics["sum_z2"] == 0.0
            assert 0.0 <= diagnostics["expected_state_fail_probability"]
        assert n_important == N_IMPORTANT
        assert N_STATES - n_important == N_MINOR

        totals = table["totals"]
        assert set(totals) == set(TOTALS_PIN)
        assert totals == TOTALS_PIN
        # the worst state carries the pinned worst branching maximum
        worst = per_state[str(WORST_STATE_PD_G)]
        assert worst["branching_max_abs_sigma"] == (
            WORST_BRANCHING_ABS_SIGMA
        )
        assert worst["gate2_pass"] is False
        # the state tail is the branching-maximum convention (the
        # committed per-cell max criterion of the per-state branching
        # maxima)
        branching_maxima = [
            entry["branching_max_abs_sigma"] for entry in per_state.values()
        ]
        for threshold, key in (
            (3.0, "states_max_abs_sigma_ge_3"),
            (3.5, "states_max_abs_sigma_ge_3.5"),
            (4.0, "states_max_abs_sigma_ge_4"),
            (4.5, "states_max_abs_sigma_ge_4.5"),
        ):
            assert totals["state_maxima_tail"][key] == sum(
                1 for value in branching_maxima if value >= threshold
            )


def test_tables_identical_no_fix(run_dir):
    """The before- and after-tables are identical modulo their
    ``tag`` field: the diagnosis identified no convention defect, so
    no fix was applied and the calibration moved nothing (the two
    files carry distinct SHA-256 digests solely because of the tag)."""
    before = _load("sigma_table_before.json")
    after = _load("sigma_table_after.json")
    assert before["tag"] == "before"
    assert after["tag"] == "after"
    assert {key: value for key, value in before.items()
            if key != "tag"} == {key: value for key, value in after.items()
                                 if key != "tag"}, (
        "the before- and after-sigma tables differ beyond their tag "
        "field; a convention fix would have landed"
    )
    assert before["per_state"] == after["per_state"]
    assert before["totals"] == after["totals"]
    assert (SIGMA_TABLE_BEFORE_SHA256 != SIGMA_TABLE_AFTER_SHA256), (
        "the two tables pin distinct digests (tag-differing canonical "
        "bytes)"
    )


def test_expectation_file_schema(run_dir, r20_archive):
    """Every one of the 80 recomputed expectation files carries schema
    ``na49_mixing.calibrate_branching_weights.expectation/1`` and the
    committed field set: the state block byte-identical to the sealed
    R20 state file, the 100-point R20 grid (identical masses), the
    degenerate-point inventory, and the per-point open weights (the
    expected shares are the normalized weights, summing to 1, and the
    card-5/6 class shares are within-class normalized over their member
    charge combinations)."""
    cells = _open_channel_cells()
    for pd_g in PDG_IDS:
        expectation = json.loads(
            _expectation_path(pd_g).read_text(encoding="utf-8")
        )
        assert expectation["schema"] == EXPECT_SCHEMA
        assert expectation["run"] == RUN_ID
        assert expectation["card"] == CARD_ID

        state = expectation["state"]
        r20_state = json.loads(
            (r20_archive / f"state_{pd_g}.json").read_text(encoding="utf-8")
        )
        assert set(state) == {
            "pdg_id", "name", "rank", "classification", "decays_per_point",
            "B", "Q", "S", "pole_mass", "nominal_width", "mass_range",
            "channels",
        }
        assert state == r20_state["state"], (
            f"state_{pd_g}.json: the state block drifted from the "
            "sealed R20 state file"
        )
        assert state["pdg_id"] == pd_g

        grid = expectation["grid"]
        assert set(grid) == {"points", "formula", "masses_gev"}
        assert grid["points"] == 100
        assert "j / 99" in grid["formula"]
        assert [float(x) for x in grid["masses_gev"]] == (
            [float(x) for x in r20_state["grid"]["masses_gev"]]
        )

        r20_degenerate = sorted(
            int(point["j"])
            for point in r20_state["degenerate_points"]
        )
        degenerate = sorted(
            int(entry["j"]) for entry in expectation["degenerate_points"]
        )
        assert degenerate == r20_degenerate

        # the pole point is the closest open grid point to the pole mass
        masses = r20_state["grid"]["masses_gev"]
        pole_mass = r20_state["state"]["pole_mass"]
        expected_pole = min(
            (j for j in range(100) if j not in degenerate),
            key=lambda j: abs(masses[j] - pole_mass),
        )
        assert expectation["pole_point"] == expected_pole

        points = expectation["points"]
        assert sorted(int(key) for key in points) == list(range(100))
        n_cells = 0
        for j in range(100):
            point = points[str(j)]
            if j in degenerate:
                assert set(point) == {"j", "mass_gev", "degenerate"}
                assert point["degenerate"] is True
            else:
                assert set(point) == {
                    "j", "mass_gev", "degenerate", "channel_weights",
                    "expected_shares", "class_shares",
                }
                assert point["degenerate"] is False
                weights = point["channel_weights"]
                shares = point["expected_shares"]
                assert set(weights) == set(shares)
                assert all(weight >= 0.0 for weight in weights.values())
                assert any(weight > 0.0 for weight in weights.values())
                total = sum(weights.values())
                for key in shares:
                    assert shares[key] == pytest.approx(
                        weights[key] / total, rel=1e-15
                    )
                assert sum(shares.values()) == pytest.approx(1.0, abs=1e-12)
                # the card-5/6 class convention: each class is recorded
                # over its member charge combinations (a subset of the
                # open channels; three-body channels are one open object
                # without a class), normalized within the class
                for class_share in point["class_shares"].values():
                    assert set(class_share) <= set(weights)
                    assert sum(
                        combo[1] for combo in class_share.values()
                    ) == pytest.approx(1.0, abs=1e-12)
                n_cells += len(weights)
        assert n_cells == cells[pd_g]


def test_sigma_tables_match_sealed_r20_gate2(run_dir, r20_archive):
    """For all 80 states the recomputed branching, charge, and angular
    maxima agree bit-exactly with the sealed R20 Gate 2 report, and
    the pass verdicts, ranks, classifications, pole points,
    degenerate-point inventories, and channel counts agree with the
    sealed R20 pre-registration and state files (the sealed R20
    archive is read-only and never re-sealed)."""
    gate = json.loads(
        (r20_archive / "gate2_r20.json").read_text(encoding="utf-8")
    )
    r20_per_state = gate["clauses"]["branching_test"]["per_state"]
    assert len(r20_per_state) == N_STATES

    r20_pre = json.loads(
        (r20_archive / "pre_registration.json").read_text(encoding="utf-8")
    )
    r20_by_pdg = {
        str(entry["pdg_id"]): entry
        for entry in r20_pre["state_coverage"]["states"]
    }

    for tag in ("before", "after"):
        table = _load(f"sigma_table_{tag}.json")
        for str_pdg, entry in table["per_state"].items():
            r20_entry = r20_per_state[str_pdg]
            assert set(r20_entry) == {
                "branching_max_abs_sigma", "charge_max_abs_sigma",
                "angular_max_abs_sigma", "degenerate_points", "pass",
            }
            for key in ("branching_max_abs_sigma", "charge_max_abs_sigma",
                        "angular_max_abs_sigma"):
                assert entry[key] == r20_entry[key], (
                    f"{tag} state {str_pdg}: the recomputed {key} does "
                    "not match the sealed R20 gate table bit-exactly"
                )
            assert entry["gate2_pass"] == r20_entry["pass"]
            assert entry["rank"] == r20_by_pdg[str_pdg]["rank"]
            assert entry["classification"] == (
                r20_by_pdg[str_pdg]["classification"]
            )
            state_file = json.loads(
                (r20_archive / f"state_{str_pdg}.json")
                .read_text(encoding="utf-8")
            )
            assert entry["n_channels"] == len(state_file["state"]["channels"])
            assert entry["n_degenerate_points"] == (
                len(state_file["degenerate_points"])
            )
            assert entry["n_open_points"] == (
                100 - entry["n_degenerate_points"]
            )
            # the pole point is the closest open grid point to the
            # pole mass (the committed R20 pole-point rule)
            masses = state_file["grid"]["masses_gev"]
            pole_mass = state_file["state"]["pole_mass"]
            degenerate = {
                int(point["j"]) for point in state_file["degenerate_points"]
            }
            expected_pole = min(
                (j for j in range(100) if j not in degenerate),
                key=lambda j: abs(masses[j] - pole_mass),
            )
            assert entry["pole_point"] == expected_pole


def test_disposition_structure(run_dir, pre_registration):
    """``disposition.json`` carries ``fix_applied: false`` on every
    entry, the 39 named residual states exactly the failing states of
    the sigma tables, every entry recorded as a finding
    (``repaired_states`` empty, no branching-fraction change), and the
    PDG -2224 entry with its angular-kernel-only diagnosis."""
    disposition = _load("disposition.json")
    assert disposition["schema"] == DISPOSITION_SCHEMA
    assert disposition["run"] == RUN_ID
    assert disposition["card"] == CARD_ID
    assert disposition["fix_applied"] is False
    assert disposition["tables_identical_no_fix"] is True
    assert disposition["n_failing_before"] == N_FAILING
    assert disposition["n_failing_after"] == N_FAILING
    assert disposition["repaired_states"] == []

    before = _load("sigma_table_before.json")
    after = _load("sigma_table_after.json")
    failing = {
        str(pd_g) for pd_g, entry in before["per_state"].items()
        if not entry["gate2_pass"]
    }
    assert len(failing) == N_FAILING
    assert {
        str(pd_g) for pd_g, entry in after["per_state"].items()
        if not entry["gate2_pass"]
    } == failing

    residual = disposition["residual_states"]
    per_state = disposition["per_state"]
    assert len(residual) == N_FAILING
    assert set(residual) == failing
    assert set(per_state) == failing
    for entry in per_state.values():
        assert set(entry) == DISPOSITION_PER_STATE_KEYS
        assert entry["fix_applied"] is False
        assert entry["disposition"] == "residual recorded as a finding"
        assert isinstance(entry["diagnosis"], str)
        assert entry["diagnosis"]
        assert set(entry["worst_cell"]) == WORST_CELL_KEYS

    # the PDG -2224 (aDelta--) residual: the angular-kernel-only
    # diagnosis (single channel, no branching cell with variance)
    angular_only = per_state[str(ANGULAR_ONLY_STATE_PD_G)]
    assert angular_only["name"] == ANGULAR_ONLY_STATE_NAME
    assert angular_only["classification"] == "important"
    assert angular_only["n_channels"] == 1
    assert angular_only["worst_branching_abs_sigma"] == 0.0
    assert angular_only["worst_charge_abs_sigma"] == 0.0
    assert angular_only["worst_angular_abs_sigma"] == (
        ANGULAR_ONLY_MAX_ABS_SIGMA
    )
    assert angular_only["channel_pooled_max_abs_t"] == 0.0
    assert angular_only["expected_state_fail_probability"] == 0.0
    cell = angular_only["worst_cell"]
    assert cell["channel"] == "-2212,-211"
    assert cell["expected_share"] == 1.0
    assert cell["sampled_fraction"] == 1.0
    assert cell["abs_sigma"] == 0.0
    assert "single-open-channel state" in angular_only["diagnosis"]
    assert "angular-kernel check" in angular_only["diagnosis"]
    assert "outside the open-weight calibration scope" in (
        angular_only["diagnosis"]
    )
    assert "not an open weight" in angular_only["diagnosis"]

    # the module-level note names the angular-kernel-only case
    assert "-2224 fails only on the angular kernel" in disposition["note"]

    assert _file_sha256(run_dir / "disposition.json") == DISPOSITION_SHA256


def test_calibration_report_structure(run_dir):
    """The report carries the four checked diagnosis dimensions, the
    statistical verdict consistent with the sigma-table totals (the
    28,019 cells, the mean square 0.990895, the 40.2295 expected
    failing states of the exact binomial null against the 39
    observed), the clause-(d) no-fix decision, and the five
    constraint-record flags."""
    report = _load("calibration_report.json")
    assert report["schema"] == REPORT_SCHEMA
    assert report["run"] == RUN_ID
    assert report["card"] == CARD_ID
    assert report["n_sigma"] == N_SIGMA

    dimensions = report["diagnosis_dimensions"]
    assert set(dimensions) == {
        "barrier evaluation point", "quadrature grid",
        "channel summation order", "degenerate-weight handling",
    }
    for dimension, text in dimensions.items():
        assert isinstance(text, str) and text.startswith("checked:"), (
            f"the {dimension!r} diagnosis is not recorded as checked"
        )

    verdict = report["statistical_verdict"]
    assert set(verdict) == {
        "n_cells", "n_cells_with_variance", "mean_abs_z_cells",
        "mean_z2_cells", "mean_abs_z_half_normal",
        "n_small_count_cells",
        "expected_failing_states_binomial_null", "n_failing_observed",
        "reading",
    }
    before = _load("sigma_table_before.json")
    totals = before["totals"]
    for key in ("n_cells", "n_cells_with_variance", "mean_abs_z_cells",
                "mean_z2_cells", "n_small_count_cells",
                "expected_failing_states_binomial_null"):
        assert verdict[key] == totals[key], (
            f"the report {key} drifted from the sigma-table totals"
        )
    assert verdict["n_failing_observed"] == N_FAILING
    assert verdict["expected_failing_states_binomial_null"] == pytest.approx(
        40.229479945611864, abs=1e-9,
    )
    assert "reading" in verdict and verdict["reading"]

    assert "no fix is applied" in report["clause_d_decision"]
    assert "resonance_widths.py" in report["clause_d_decision"]
    assert report["n_failing_before"] == N_FAILING
    assert report["n_failing_after"] == N_FAILING
    assert report["n_failing_before"] == report["n_failing_after"]

    constraint = report["constraint_record"]
    assert constraint == {
        "no_branching_fit": True,
        "catalog_byte_unchanged": True,
        "r20_archive_read_only": True,
        "nominal_widths_unchanged": True,
        "barrier_polynomial_unchanged": True,
    }

    assert _file_sha256(run_dir / "calibration_report.json") == (
        CALIBRATION_REPORT_SHA256
    )


def test_seal_manifest_and_sha256_sums(run_dir):
    """``seal_manifest.json`` records the 86 sealed files (80
    expectation files plus the 6 payloads; the manifest and the sums
    file exclude themselves) totaling 7,103,071 bytes and the digest
    of ``SHA256SUMS.txt``, which is in ``sha256sum -c`` format with
    run-dir-relative paths, carries the pinned payload digests,
    verifies against every sealed file on disk, and self-consistently
    accounts for the full byte total."""
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

    expected_rels = (
        {name for name in SUMS_TOP_LEVEL_PAYLOADS}
        | {f"expectations/state_{pd_g}.json" for pd_g in PDG_IDS}
    )
    assert len(expected_rels) == N_SEALED_FILES
    entries = [line.split("  ", 1) for line in lines]
    assert {rel for _digest, rel in entries} == expected_rels, (
        "the sums file covers a drifted file set"
    )
    for digest, rel in entries:
        if rel in SUMS_PAYLOAD_SHA256:
            assert digest == SUMS_PAYLOAD_SHA256[rel], (
                f"{rel}: the sums-carrying digest drifted from the "
                "pinned digest"
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
    """The post-seal clean-environment record re-verified all 86
    sums-covered files without mismatches, and the clean-interpreter
    re-run reproduced the four sealed report payloads byte-for-byte
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
    assert "calibrate_branching_weights.py" in " ".join(argv)
    assert RUN_NAME in " ".join(argv)
    assert R20_RUN_NAME in " ".join(argv)

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
