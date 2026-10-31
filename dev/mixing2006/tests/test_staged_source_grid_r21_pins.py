"""Card-11 R21 fixed baryonic-source grid: committed-run-artifact pins.

Task ``TASK-20260913-r2-run-r21-baryonic-source-grid`` (round 2, card 11;
the run R21 of plan section 8, line 559).  This module pins the sealed
run R21 evidence under
``dev/00runs/r21_baryonic_source_grid_2026-09-13/``: the pre-registration,
provenance, the 20 point files and the 80 per-replica files on the
committed baryonic grid (source (B, Q, S) = (1, 1, 0) at rest,
``M_k = 1.07324891 + 0.1 k`` for ``k = 0..19``, 4 replicas x 25,000
decays per point, master seed 20260913), the Gate 2 discontinuity
verification report ``gate2_r21.json``, the seal manifest, and
``SHA256SUMS.txt`` (repo-root-relative paths, ``sha256sum -c`` format).

Sibling of ``tests/test_staged_source_grid_card9_pins.py`` (which pins
the driver's pre-registered contract on freshly generated grids) and of
``tests/test_staged_r20_unit_grid_card10_pins.py`` (the same pin style
on the sealed R20 run).  These pins are read-only over the sealed
artifacts: no decays are generated, the sealed run is never re-sealed,
and the post-seal clean-environment record
``clean_environment_commands.json`` is never re-executed.

Pins:

- ``test_sealed_run_dir_inventory``: the run dir holds exactly the
  103-file sealed set (pre-registration, provenance, the Gate 2
  report, the 20 ``point_<k:02d>.json`` files, and the 80
  ``replica_<r>_point_<k:02d>.json`` files) plus the three post-seal
  files (seal manifest, sums file, clean-environment record).
- ``test_pre_registration_structure``: the sealed pre-registration
  carries the committed contract (schema, the card-9 driver card id,
  mode, the baryonic source (1, 1, 0) at rest, the 20-point grid
  ``M_k = 1.07324891 + 0.1 k`` with the threshold start, T1
  statistics 4 replicas x 25,000 decays, master seed 20260913 with the
  closed substream labels, the per-point open sets recomputed
  independently from the committed catalog, the card-9 output schema
  key list, and ``written_before_generation``).
- ``test_point_file_schema``: every ``point_<k:02d>.json`` carries the
  card-9 schema (multiplicities over the catalog's stable termination
  set in catalog record order, momentum moments, feed-down ancestry,
  the exact per-event charge balance (Q, B) = (1, 1) over 100,000
  events, the charge-conjugate co-occurrence pairs), sits on its grid
  mass with its pre-registered open set, and its multiplicities are
  the mean of the four replica multiplicities.
- ``test_replica_file_schema``: every ``replica_<r>_point_<k:02d>.json``
  carries the same card-9 schema at the per-replica level (25,000
  decays, the same species / moments / feed-down / pairs key sets, the
  exact (1, 1) charge balance over its own events), and the four
  replicas' multiplicities average to the point's multiplicities.
- ``test_gate2_report_structure``: ``gate2_r21.json`` carries the Gate
  2 clauses with the committed verdict: ``local_conservation`` passes
  over 2,000,000 events, ``branching_test`` is the null placeholder,
  ``threshold_discontinuity`` fails (19 adjacent pairs, 351 applied
  checks, 169 failed, max |sigma| = 372.04134193769096),
  ``no_branching_fit`` records no fitting, and the sealed form is
  marked.
- ``test_seal_manifest_and_sha256_sums``: ``seal_manifest.json``
  records the 103-file sealed set and its digest, and
  ``SHA256SUMS.txt`` is in ``sha256sum -c`` format with
  repo-root-relative paths and verifies against every sealed file on
  disk.
- ``test_clean_environment_validation_record``: the post-seal
  validation record re-verified all 103 files without mismatches and
  the clean-interpreter re-run reproduced the sealed Gate 2 report
  byte-for-byte (hashes match).

Hermetic: reads the sealed run dir (gitignored; the artifact is
present on disk) and the committed catalog; no generation runs.  If
the run dir is absent (a fresh checkout), the pins skip with a clear
message.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from na49_mixing.hadron_catalog import HadronCatalog

NESTED_ROOT = Path(__file__).resolve().parent.parent
MONO_ROOT = NESTED_ROOT.parent.parent
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
RUN_NAME = "r21_baryonic_source_grid_2026-09-13"
RUN_DIR = MONO_ROOT / "dev" / "00runs" / RUN_NAME

# ---------------------------------------------------------------------------
# The committed run contract (the pre-registration realizes it).
# ---------------------------------------------------------------------------
RUN_ID = "R21"
CARD_ID = "TASK-20260913-r2-run-r21-baryonic-source-grid"
#: The card-9 driver card: the pre-registration is written by the
#: driver and records the driver's card id, while the card-11/12
#: records (gate report, seal manifest, clean-environment record)
#: record the run card id.
DRIVER_CARD_ID = "TASK-20260913-r2-source-grid-runner"
MASTER_SEED = 20260913
SEED_RULE = "derive_substreams(20260913, r, k * 1_000_000 + i)"
SEED_LABELS = ("decay_channel", "decay_kinematics")
N_REPLICAS = 4
DECAYS_PER_REPLICA = 25_000
DECAYS_PER_POINT = N_REPLICAS * DECAYS_PER_REPLICA
N_POINTS = 20
GRID_START_GEV = 1.07324891
GRID_STEP_GEV = 0.1
SOURCE_QN = (1, 1, 0)
T1_LABEL = (
    "T1: 4 replicas x 25,000 decays per grid point (plan section 5.1)"
)
N_SEALED_FILES = 103  # pre-registration, provenance, gate report, 20+80
SEAL_TOTAL_BYTES = 916_240
SUMS_SHA256 = "92f4070758ceac252e9209b01efa076c3351cbc54ed270d347b8be3fb6164537"

PRE_SCHEMA = "na49_mixing.staged_source_grid.pre_registration/1"
SEAL_SCHEMA = "na49_mixing.card11_r21.seal_manifest/1"
GATE_SCHEMA = "na49_mixing.card11_r21.discontinuity_verification/1"
CLEAN_SCHEMA = "na49_mixing.card11_r21.clean_environment/1"

#: The card-9 per-point output schema key list (verbatim order).
SCHEMA_KEYS = [
    "point", "source_mass_gev", "source_qn", "open_records", "n_decays",
    "multiplicities", "momentum_moments", "feed_down", "charge_balance",
    "pairs",
]
PAIR_KEYS = (
    "pi_plus_pi_minus_per_event", "p_pbar_per_event", "n_nbar_per_event",
)
MOMENT_KEYS = ("mean_p_gev", "mean_pt_y_gev", "mean_e_gev")
CHECK_KEYS = {
    "pair", "species", "observable", "open_set_changes", "status",
    "observed", "delta_observed", "delta_analytic_full",
    "delta_analytic_applied", "se_combined", "sigma", "pass",
}
OBSERVABLES = ("multiplicity", "mean_p", "mean_pt_y")

#: The committed Gate 2 verdict pins.
N_ADJACENT_PAIRS = 19
OPEN_SET_CHANGES = (0, 2, 4, 5, 6)
OPEN_SET_CHANGE_RECORDS = {
    "0": [2214],
    "2": [12212, 2216, 22212],
    "4": [102212],
    "5": [122212],
    "6": [212214, 202214, 122214],
}
N_CHECKS_APPLIED = 351
N_CHECKS_FAILED = 169
N_CHECKS_VACUOUS_ZERO = 221
N_CHECKS_ABSENT_SPECIES = 454
N_OPEN_SET_CHANGE_APPLIED = 72
N_FAILED_AT_OPEN_SET_CHANGES = 4
N_FAILED_AT_UNCHANGED_PAIRS = 165
MAX_ABS_SIGMA = 372.04134193769096
MAX_ABS_SIGMA_CHECK = "pair(1,2) pi_0 mean_p"
N_LOCAL_EVENTS_CHECKED = 2_000_000

#: ``sha256sum -c`` line: 64 hex digits, two spaces, the relative path.
SUMS_LINE = re.compile(r"^[0-9a-f]{64}  \S+$")


@lru_cache(maxsize=1)
def _catalog() -> HadronCatalog:
    return HadronCatalog.from_json(CATALOG_PATH)


def _species_key(name: str) -> str:
    """The driver's species-key transliteration (pi+ -> pi_plus, pi- ->
    pi_minus, a trailing 0 -> _0, otherwise verbatim)."""
    if name.endswith("+"):
        return name[:-1] + "_plus"
    if name.endswith("-"):
        return name[:-1] + "_minus"
    if name.endswith("0"):
        return name[:-1] + "_0"
    return name


def _stable_species_keys_ordered() -> list[str]:
    """The multiplicity key list of the card-9 schema: the catalog's
    stable termination set, all listed, in catalog record order."""
    stable = _catalog().stable_states()
    return [
        _species_key(str(record.name))
        for record in _catalog().records()
        if int(record.pdg_id) in stable
    ]


def _expected_open_set(k: int) -> list[int]:
    """The card's open set at baryon point *k*, computed independently
    from the committed catalog with the truncated-spectral-window rule:
    the records with source (1, 1, 0) whose window
    ``[mass_range[0], min(mass_range[1], M_k)]`` is non-empty, in
    catalog record order."""
    catalog = _catalog()
    mass = GRID_START_GEV + GRID_STEP_GEV * k
    out: list[int] = []
    for record in catalog.records():
        if (int(record.B), int(record.Q), int(record.S)) != SOURCE_QN:
            continue
        lo = float(record.mass_range[0])
        hi = float(record.mass_range[1])
        if min(hi, mass) >= lo:
            out.append(int(record.pdg_id))
    return out


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sealed_file_names() -> set[str]:
    """The 103-file sealed set recorded in SHA256SUMS.txt (the sums
    exclude the seal manifest, the sums file itself, and the post-seal
    clean-environment record)."""
    return (
        {"pre_registration.json", "provenance.json", "gate2_r21.json"}
        | {f"point_{k:02d}.json" for k in range(N_POINTS)}
        | {
            f"replica_{r}_point_{k:02d}.json"
            for r in range(N_REPLICAS)
            for k in range(N_POINTS)
        }
    )


def _load(name: str) -> dict:
    return json.loads((RUN_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def run_dir():
    """The sealed R21 run dir, skipped with a clear message when the
    gitignored artifact is absent or incomplete (a fresh checkout)."""
    if not (RUN_DIR / "pre_registration.json").is_file():
        pytest.skip(
            f"the sealed R21 run dir {RUN_DIR} is absent (gitignored "
            "artifact); restore the sealed run to execute the "
            "artifact pins"
        )
    expected = _sealed_file_names() | {
        "seal_manifest.json", "SHA256SUMS.txt",
        "clean_environment_commands.json",
    }
    missing = sorted(name for name in expected
                     if not (RUN_DIR / name).is_file())
    if missing:
        pytest.skip(
            f"the sealed R21 run dir {RUN_DIR} is incomplete (missing: "
            + ", ".join(missing) + "); restore the sealed run to "
            "execute the artifact pins"
        )
    return RUN_DIR


@pytest.fixture(scope="module")
def pre_registration(run_dir) -> dict:
    return _load("pre_registration.json")


# ---------------------------------------------------------------------------
# Sealed artifact pins
# ---------------------------------------------------------------------------
def test_sealed_run_dir_inventory(run_dir):
    """The run dir holds exactly the 103-file sealed set plus the three
    post-seal files (seal manifest, sums file, clean-environment
    record); nothing else is present."""
    on_disk = {path.name for path in run_dir.iterdir()}
    sealed = _sealed_file_names()
    assert len(sealed) == N_SEALED_FILES
    assert on_disk == sealed | {
        "seal_manifest.json", "SHA256SUMS.txt",
        "clean_environment_commands.json",
    }, (
        "the sealed run dir drifted from the committed 103-file sealed "
        "set plus the post-seal files"
    )


def test_pre_registration_structure(run_dir, pre_registration):
    """The sealed pre-registration carries the committed contract: the
    baryonic source (1, 1, 0) at rest on the 20-point grid
    M_k = 1.07324891 + 0.1 k, T1 statistics (4 replicas x 25,000
    decays per point), master seed 20260913 with the closed substream
    labels, the per-point open sets (recomputed independently from the
    committed catalog), the card-9 output schema, and
    written_before_generation."""
    pre = pre_registration
    assert pre["schema"] == PRE_SCHEMA
    assert pre["card"] == DRIVER_CARD_ID
    assert pre["mode"] == "production"
    assert pre["source_type"] == "baryonic"
    assert list(pre["source_qn"]) == list(SOURCE_QN)
    assert pre["written_before_generation"] is True

    baryonic = pre["grids"]["baryonic"]
    assert baryonic["source_qn"] == list(SOURCE_QN)
    assert baryonic["start_gev"] == GRID_START_GEV
    assert baryonic["step_gev"] == GRID_STEP_GEV
    assert baryonic["points"] == N_POINTS
    assert baryonic["label"] == (
        "20 points, 1.07324891 + 0.1 k, (1,1,0) at rest"
    )
    # the grid start is the charged/baryonic threshold
    assert "1.07324891" in baryonic["threshold_note"]
    assert "threshold" in baryonic["threshold_note"]

    assert pre["grid_points"] == list(range(N_POINTS))
    for k in range(N_POINTS):
        assert pre["grid_masses_gev"][k] == pytest.approx(
            GRID_START_GEV + GRID_STEP_GEV * k, abs=1e-12
        )
    # both committed grids are recorded (the run executes the baryonic
    # one; the mesonic grid is carried alongside for R22)
    mesonic = pre["grids"]["mesonic"]
    assert mesonic["source_qn"] == [0, 0, 0]
    assert mesonic["start_gev"] == 0.2699532
    assert mesonic["step_gev"] == GRID_STEP_GEV
    assert mesonic["points"] == 28
    assert mesonic["label"] == (
        "28 points, 0.2699532 + 0.1 k, (0,0,0) at rest"
    )
    assert "2 m_pi0" in mesonic["threshold_note"]

    stats = pre["statistics"]
    assert stats["standard"] == T1_LABEL
    assert stats["replicas"] == N_REPLICAS
    assert stats["decays_per_replica"] == DECAYS_PER_REPLICA

    seed_map = pre["seed_map"]
    assert seed_map["master_seed"] == MASTER_SEED
    assert seed_map["rule"] == SEED_RULE
    assert seed_map["substream_labels"] == list(SEED_LABELS)

    for key in (
        "open_set_rule", "open_set_nonempty_note", "record_draw",
        "mass_rule", "decay", "kernels",
    ):
        assert key in pre["mixture_model"]

    assert set(pre["open_sets"]) == {str(k) for k in range(N_POINTS)}
    for k in range(N_POINTS):
        assert pre["open_sets"][str(k)] == _expected_open_set(k), (
            f"point {k}: the pre-registered open set does not match the "
            "committed catalog's truncated-spectral-window rule"
        )

    schema = pre["output_schema"]
    assert schema["point_file"] == "point_<k:02d>.json"
    assert schema["replica_file"] == "replica_<r>_point_<k:02d>.json"
    assert schema["keys"] == list(SCHEMA_KEYS)
    assert "catalog stable termination set" in (
        schema["multiplicity_species"]
    )
    assert schema["mean_pt_y_convention"].startswith(
        "the density-p_T f convention"
    )
    assert "min(n, nbar)" in schema["pairs_convention"]


def test_point_file_schema(run_dir, pre_registration):
    """Every point_<k:02d>.json carries the card-9 schema:
    multiplicities over the catalog's stable termination set (all
    listed, in catalog record order), momentum moments and feed-down
    ancestry over exactly the present species, the exact per-event
    charge balance (Q, B) = (1, 1) over 100,000 events, the
    charge-conjugate co-occurrence pairs, the point's grid mass and
    pre-registered open set, and multiplicities equal to the mean of
    the four replicas' multiplicities."""
    species_keys = _stable_species_keys_ordered()
    for k in range(N_POINTS):
        payload = _load(f"point_{k:02d}.json")
        assert list(payload) == SCHEMA_KEYS, (
            f"point_{k:02d}.json: the card-9 schema key set"
        )
        assert payload["point"] == k
        assert payload["source_mass_gev"] == pytest.approx(
            GRID_START_GEV + GRID_STEP_GEV * k, abs=1e-12
        )
        assert list(payload["source_qn"]) == list(SOURCE_QN)
        assert payload["open_records"] == (
            pre_registration["open_sets"][str(k)]
        )
        assert payload["n_decays"] == DECAYS_PER_POINT

        multiplicities = payload["multiplicities"]
        assert list(multiplicities) == species_keys, (
            f"point_{k:02d}.json: the multiplicity keys must be the "
            "catalog's stable termination set in catalog record order"
        )
        for species, entry in multiplicities.items():
            assert set(entry) == {"mean", "se"}, (
                f"point_{k:02d}.json {species}: multiplicity entry"
            )
            assert 0.0 <= entry["mean"] < 2.0
            assert 0.0 <= entry["se"]
        present = [
            species for species, entry in multiplicities.items()
            if entry["mean"] > 0.0
        ]

        moments = payload["momentum_moments"]
        assert list(moments) == present, (
            f"point_{k:02d}.json: momentum moments must cover exactly "
            "the present species"
        )
        for species, entry in moments.items():
            assert set(entry) == set(MOMENT_KEYS)
            for key in MOMENT_KEYS:
                assert entry[key] >= 0.0

        feed_down = payload["feed_down"]
        assert list(feed_down) == present, (
            f"point_{k:02d}.json: feed-down ancestry must cover exactly "
            "the present species"
        )
        for species, entry in feed_down.items():
            assert set(entry) == {"fraction_resonant", "by_parent"}
            assert 0.0 <= entry["fraction_resonant"] <= 1.0
            assert sum(entry["by_parent"].values()) == pytest.approx(
                entry["fraction_resonant"], abs=1e-9
            )

        balance = payload["charge_balance"]
        assert balance == {
            "total_Q_exact": SOURCE_QN[1],
            "total_B_exact": SOURCE_QN[0],
            "events_checked": DECAYS_PER_POINT,
        }

        pairs = payload["pairs"]
        assert set(pairs) == set(PAIR_KEYS)
        for key in PAIR_KEYS:
            assert set(pairs[key]) == {"mean", "se"}
            assert 0.0 <= pairs[key]["mean"]
            assert 0.0 <= pairs[key]["se"]

        # the point aggregate is the mean of the four replicas
        for species in species_keys:
            replica_means = [
                _load(f"replica_{r}_point_{k:02d}.json")
                ["multiplicities"][species]["mean"]
                for r in range(N_REPLICAS)
            ]
            assert sum(replica_means) / N_REPLICAS == pytest.approx(
                multiplicities[species]["mean"], abs=1e-12
            ), (
                f"point_{k:02d}.json {species}: the point multiplicity "
                "is not the mean of the four replicas"
            )


def test_replica_file_schema(run_dir, pre_registration):
    """Every replica_<r>_point_<k:02d>.json carries the card-9 schema
    at the per-replica level: 25,000 decays, the same species /
    moments / feed-down / pairs key sets as its point file, the exact
    (Q, B) = (1, 1) charge balance over the replica's own events, and
    the point's grid mass and pre-registered open set."""
    species_keys = _stable_species_keys_ordered()
    for r in range(N_REPLICAS):
        for k in range(N_POINTS):
            name = f"replica_{r}_point_{k:02d}.json"
            payload = _load(name)
            point = _load(f"point_{k:02d}.json")
            assert list(payload) == SCHEMA_KEYS, f"{name}: schema keys"
            assert payload["point"] == k
            assert payload["source_mass_gev"] == (
                point["source_mass_gev"]
            )
            assert list(payload["source_qn"]) == list(SOURCE_QN)
            assert payload["open_records"] == (
                pre_registration["open_sets"][str(k)]
            )
            assert payload["n_decays"] == DECAYS_PER_REPLICA

            assert list(payload["multiplicities"]) == species_keys
            for species, entry in payload["multiplicities"].items():
                assert set(entry) == {"mean", "se"}
                assert 0.0 <= entry["mean"] < 2.0
                assert 0.0 <= entry["se"]
            present = [
                species for species, entry
                in payload["multiplicities"].items()
                if entry["mean"] > 0.0
            ]
            assert list(payload["momentum_moments"]) == present, (
                f"{name}: the replica moments must cover exactly the "
                "present species"
            )
            assert list(payload["feed_down"]) == present, (
                f"{name}: the replica feed-down must cover exactly the "
                "present species"
            )

            assert payload["charge_balance"] == {
                "total_Q_exact": SOURCE_QN[1],
                "total_B_exact": SOURCE_QN[0],
                "events_checked": DECAYS_PER_REPLICA,
            }
            assert set(payload["pairs"]) == set(PAIR_KEYS)


def test_gate2_report_structure(run_dir):
    """gate2_r21.json carries the Gate 2 clauses with the committed
    verdict: local_conservation passes over 2,000,000 events,
    branching_test is the null placeholder, threshold_discontinuity
    fails (19 adjacent pairs, 351 applied checks, 169 failed, max
    |sigma| = 372.04134193769096 at pair(1,2) pi_0 mean_p),
    no_branching_fit records no fitting, and the sealed form is
    marked."""
    gate = _load("gate2_r21.json")
    assert gate["schema"] == GATE_SCHEMA
    assert gate["gate"] == "gate 2 (plan section 8, line 575)"
    assert gate["run"] == RUN_ID
    assert gate["card"] == CARD_ID
    assert gate["sealed"] is True
    assert "note" in gate

    clauses = gate["clauses"]
    assert set(clauses) == {
        "local_conservation", "branching_test",
        "threshold_discontinuity", "observables", "no_branching_fit",
    }

    local = clauses["local_conservation"]
    assert set(local) == {
        "charge_balance_exact", "baryon_balance_exact",
        "events_checked", "pass",
    }
    assert local["charge_balance_exact"] is True
    assert local["baryon_balance_exact"] is True
    assert local["events_checked"] == N_LOCAL_EVENTS_CHECKED
    assert local["pass"] is True

    assert clauses["branching_test"] is None

    threshold = clauses["threshold_discontinuity"]
    assert threshold["adjacent_pairs_checked"] == N_ADJACENT_PAIRS
    assert threshold["open_set_changes"] == list(OPEN_SET_CHANGES)
    assert threshold["open_set_change_records"] == (
        OPEN_SET_CHANGE_RECORDS
    )
    assert threshold["pass"] is False

    metric = threshold["metric"]
    for key in (
        "statement", "delta_analytic", "se_multiplicity", "se_moments",
        "moment_presence_rule", "quadrature", "numeric_approximations",
        "engine_validation",
    ):
        assert key in metric
    assert metric["quadrature"] == {
        "root_m_nodes": 384,
        "daughter_m_nodes": 96,
        "angle_nodes": 48,
        "eta_mbc_nodes": 24,
        "eta_cos_nodes": 24,
        "eta_lab_angle_nodes": 64,
    }
    assert "3 SE_combined" in metric["statement"]

    checks = threshold["checks"]
    # one entry per adjacent pair, per stable species, per observable
    assert len(checks) == N_ADJACENT_PAIRS \
        * len(_stable_species_keys_ordered()) * len(OBSERVABLES)
    assert all(set(check) == CHECK_KEYS for check in checks)
    pairs = sorted({tuple(check["pair"]) for check in checks})
    assert pairs == [(k, k + 1) for k in range(N_POINTS - 1)]
    species = sorted({check["species"] for check in checks})
    assert species == sorted(_stable_species_keys_ordered())
    assert all(
        check["observable"] in OBSERVABLES for check in checks
    )

    n_applied = n_failed = n_vacuous = n_absent = 0
    for check in checks:
        assert check["status"] in (
            "pass", "fail", "vacuous_zero", "absent_species",
        )
        if check["status"] in ("pass", "fail"):
            n_applied += 1
            n_failed += check["status"] == "fail"
            # the pre-registered 3-sigma metric, pinned per check
            assert check["pass"] == (
                abs(check["sigma"]) <= 3.0
            ), f"{check['pair']} {check['species']} " \
               f"{check['observable']}: the sigma rule"
            if check["open_set_changes"]:
                assert check["delta_analytic_applied"] == (
                    check["delta_analytic_full"]
                )
            else:
                assert check["delta_analytic_applied"] == 0.0
        elif check["status"] == "vacuous_zero":
            n_vacuous += 1
            assert check["pass"] is True
            assert check["observed"] == [0.0, 0.0]
            assert check["sigma"] == 0.0
        else:  # absent_species
            n_absent += 1
            assert check["pass"] is None
            assert check["sigma"] is None
            assert check["se_combined"] is None

    assert threshold["n_checks_applied"] == n_applied == N_CHECKS_APPLIED
    assert threshold["n_checks_failed"] == n_failed == N_CHECKS_FAILED
    assert threshold["n_checks_vacuous_zero"] == n_vacuous == (
        N_CHECKS_VACUOUS_ZERO
    )
    assert threshold["n_checks_absent_species"] == n_absent == (
        N_CHECKS_ABSENT_SPECIES
    )
    assert threshold["n_open_set_change_applied"] == (
        N_OPEN_SET_CHANGE_APPLIED
    )
    assert threshold["n_failed_at_open_set_changes"] == (
        N_FAILED_AT_OPEN_SET_CHANGES
    )
    assert threshold["n_failed_at_unchanged_pairs"] == (
        N_FAILED_AT_UNCHANGED_PAIRS
    )
    assert threshold["max_abs_sigma"] == MAX_ABS_SIGMA
    assert threshold["max_abs_sigma_check"] == MAX_ABS_SIGMA_CHECK

    assert clauses["observables"] is None
    assert clauses["no_branching_fit"] == {"no_fitting": True}


def test_seal_manifest_and_sha256_sums(run_dir):
    """seal_manifest.json records the 103-file sealed set and the
    digest of SHA256SUMS.txt, which is in sha256sum -c format with
    repo-root-relative paths and verifies against every sealed file on
    disk."""
    manifest = _load("seal_manifest.json")
    assert manifest["schema"] == SEAL_SCHEMA
    assert manifest["run"] == RUN_ID
    assert manifest["card"] == CARD_ID
    assert manifest["file_count"] == N_SEALED_FILES
    assert manifest["total_bytes"] == SEAL_TOTAL_BYTES
    assert manifest["sha256_sums_file"] == "SHA256SUMS.txt"
    assert manifest["excluded_from_sums"] == [
        "seal_manifest.json", "SHA256SUMS.txt",
    ]
    assert "monorepo-root-relative" in manifest["path_convention"]
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$",
                    manifest["created_utc"])

    sums_path = run_dir / "SHA256SUMS.txt"
    assert manifest["sha256_sums_sha256"] == _file_sha256(sums_path)
    assert manifest["sha256_sums_sha256"] == SUMS_SHA256
    text = sums_path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert "\r" not in text
    lines = text.splitlines()
    assert len(lines) == N_SEALED_FILES
    prefix = f"dev/00runs/{RUN_NAME}/"
    entries = []
    for line in lines:
        assert SUMS_LINE.match(line), (
            f"SHA256SUMS.txt is not in sha256sum -c format: {line!r}"
        )
        digest, rel = line.split("  ", 1)
        assert rel.startswith(prefix), (
            f"SHA256SUMS.txt paths must be repo-root-relative: {rel!r}"
        )
        entries.append((digest, rel))
    assert {rel for _digest, rel in entries} == {
        prefix + name for name in _sealed_file_names()
    }
    for digest, rel in entries:
        assert _file_sha256(MONO_ROOT / rel) == digest, (
            f"the sealed digest of {rel} does not match the file on "
            "disk; the run dir was modified after the seal"
        )
    assert sum((MONO_ROOT / rel).stat().st_size
               for _digest, rel in entries) == SEAL_TOTAL_BYTES


def test_clean_environment_validation_record(run_dir):
    """The post-seal clean-environment record re-verified all 103
    sealed files without mismatches, and the clean-interpreter
    re-run (python -I, no local packages) reproduced the sealed Gate 2
    report byte-for-byte (the recorded sealed and recomputed digests
    match and equal the digest of the report on disk)."""
    record = _load("clean_environment_commands.json")
    assert record["schema"] == CLEAN_SCHEMA
    assert record["run"] == RUN_ID
    assert record["card"] == CARD_ID
    assert record["seal_verification"] == {
        "files_checked": N_SEALED_FILES,
        "mismatches": [],
    }
    assert record["probe_returncode"] == 0
    assert record["hashes_match"] is True
    for key in (
        "python_version", "python_implementation", "platform",
        "machine", "numpy_version", "executable",
    ):
        assert key in record["environment"]
    gate_digest = _file_sha256(run_dir / "gate2_r21.json")
    assert record["sealed_report_sha256"] == {
        "gate2_r21.json": gate_digest
    }
    assert record["recomputed_report_sha256"] == {
        "gate2_r21.json": gate_digest
    }
    # the re-run is the clean-interpreter probe of the run-level
    # verification script, run from the monorepo root
    command = record["commands"][0]
    assert command["cwd"] == str(MONO_ROOT)
    assert "-I" in command["argv"]
