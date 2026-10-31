"""Card-13 R23 resonance-only p+p ablation: committed-run-artifact pins.

Task ``TASK-20260913-r2-run-r23-resonance-only-pp-ablation`` (round 2, card
13; the run R23 of plan section 8, line 561).  This module pins the sealed
run R23 evidence under
``dev/00runs/r23_resonance_only_pp_ablation_2026-09-13/``: the pre-
registration, the provenance, the bit-identity report, the eight replica
files (``replica_<seed>.json`` for seed ``20300913 + k``, ``k = 0..7``),
the replica-seeds manifest, the Gate 2 verdict report ``gate2_r23.json``,
the R01 comparison report ``comparison_r01.json``, the seal manifest, and
``SHA256SUMS.txt`` (run-dir-relative paths, ``sha256sum -c`` format).

The run is a resonance-only p+p ablation: sources at or below the M8 kernel
boundary (``continuum.match_mass`` = 2.680194110727528 GeV) decay through
the catalog-driven card-6/7 kernel while sources above it keep the legacy
``SourceDecayer`` (H3), bit-identical to the legacy builder at the same
seed.  The first 10,000 events of replica 0 are regenerated on the legacy
side and compared event by event across four layers before any replica is
generated.

Sibling of ``tests/test_staged_source_grid_r22_pins.py`` (the same pin
style on the sealed R22 mesonic run) and of
``tests/test_staged_source_grid_r21_pins.py`` (the R21 baryonic run).  These
pins are read-only over the sealed artifacts: no events are generated, the
sealed run is never re-sealed, and the post-seal clean-environment record
``clean_environment_commands.json`` is never re-executed.

Pins:

- ``test_sealed_run_dir_inventory``: the run dir holds exactly the 17-file
  sealed set (pre-registration, provenance, bit-identity report, the Gate 2
  report, the R01 comparison, the replica-seeds manifest, the eight
  ``replica_<seed>.json`` files) plus the three post-seal files (seal
  manifest, sums file, clean-environment record).
- ``test_pre_registration_structure``: the sealed pre-registration carries
  the committed contract (T2 statistics 8 replicas x 100,000 events, the
  replica seeds ``20300913 + k``, the M8 frozen parameter set and its hash,
  the kernel boundary 2.680194110727528 GeV, the observable set, the
  identified top-5 set from the sealed R11 ranked residual map, the bit-
  identity check config, the comparison-metrics clauses (i)-(iii), and
  ``no_fitting``).
- ``test_bit_identity_report_structure``: ``bit_identity_report.json``
  passes all four layers (topology, exchange, spectral_state,
  legacy_kernel_finals) over 10,000 events at the replica-0 seed, with the
  below-boundary source-0 skip accounting.
- ``test_replica_file_schema``: every ``replica_<seed>.json`` carries the
  committed per-replica schema (the per-species binned cross sections and
  counts over the 1017-bin six-species paper window, the feed-down
  fractions, the boundary statistics, and the per-event conservation check)
  at 100,000 events, and its seed matches the pre-registration and the
  replica-seeds manifest.
- ``test_replica_seeds_manifest``: ``replica_seeds.json`` records the eight
  completed replicas, the T2 event budget, and the production mode.
- ``test_gate2_report_structure``: ``gate2_r23.json`` carries the Gate 2
  clauses with the committed verdict: ``local_conservation`` passes over
  800,000 events, ``branching_test`` fails (folded from R20),
  ``threshold_discontinuity`` fails (folded from R20/R21/R22),
  ``observables`` fails with the (i)-(iii) metrics, ``no_branching_fit``
  records no fitting, ``bit_identity`` passes all four layers, and the
  sealed form is marked.
- ``test_comparison_report_structure``: ``comparison_r01.json`` carries the
  R01 comparison (the normalization convention, the 1017-bin pooled
  predictions, the (i)-(iii) metrics against the sealed R01 archive, the
  boundary and feed-down blocks, the conservation check, and the input
  hashes).
- ``test_seal_manifest_and_sha256_sums``: ``seal_manifest.json`` records the
  15-file sealed set and its digest, and ``SHA256SUMS.txt`` is in
  ``sha256sum -c`` format with run-dir-relative paths and verifies against
  every sealed file on disk.
- ``test_clean_environment_validation_record``: the post-seal
  clean-environment record re-verified all 12 sums-covered files without
  mismatches and the clean-interpreter re-run reproduced the sealed R01
  comparison byte-for-byte (hashes match).

Hermetic: reads the sealed run dir (gitignored; the artifact is present on
disk); no generation runs.  If the run dir is absent (a fresh checkout), the
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
RUN_NAME = "r23_resonance_only_pp_ablation_2026-09-13"
RUN_DIR = MONO_ROOT / "dev" / "00runs" / RUN_NAME

# ---------------------------------------------------------------------------
# The committed run contract (the pre-registration realizes it).
# ---------------------------------------------------------------------------
RUN_ID = "R23"
CARD_ID = "TASK-20260913-r2-run-r23-resonance-only-pp-ablation"
DRIVER = "helpers/benchmarking/staged_r23_resonance_ablation.py"
STAGE_LABEL = "r23_resonance_ablation"

N_REPLICAS = 8
EVENTS_PER_REPLICA = 100_000
TOTAL_EVENTS = N_REPLICAS * EVENTS_PER_REPLICA  # 800,000
REPLICA_SEED_BASE = 20300913
REPLICA_SEEDS = [REPLICA_SEED_BASE + k for k in range(N_REPLICAS)]
KERNEL_MASTER_SEED = 20260913

#: The M8 frozen p+p parameter set: its hash and its kernel boundary.
M8_PARAMETER_HASH = (
    "358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1"
)
KERNEL_BOUNDARY_GEV = 2.680194110727528

#: The sealed input-hash pins (shared across the pre-registration,
#: provenance, replicas, bit-identity report, and comparison report).
CONFIG_SHA256 = "154e41028fc68d63daf74af73ab812789ae11915fcdabb3b1a6d32deabf6693f"
TABLES_SHA256 = "0f144334eff08022716b3d811b9f746a54e59aa97053fc2b9094d1bcdb81f81c"
TABLE_DATA_DIGEST = "e3df7b4141f761d5574e957aae8e5e1bc29bd034e6ea1ac0116d1cd25583824e"
CATALOG_SHA256 = "6e4de7372a5bbf99affd6fa3ccc295c4cef09404dab3b17acd8816cda762bdcd"
BUILDER_MODULE_SHA256 = (
    "bde0fa9a228f3e9b713cc646bddfc64e2b66b7968a5db3e0c8050d169e81e036"
)
DRIVER_SHA256 = "fcf4c37ff7ca1c165ea52e24602297bb0b91ac1b48ff48494c2a0a7e93d3df8a"
R11_SUMS_SHA256 = (
    "e15234adca12a5c5b7e5b73781c81f1ea68764132bd506ccca0c898d404a14bb"
)
R01_ARCHIVE_SEAL_SHA256 = (
    "e52fbb7fe2abc516d94a456e57e62a2700bdb6ef9979fa78871baebdf1c36b02"
)

# The bit-identity check (first 10,000 events of replica 0, four layers).
BIT_IDENTITY_SEED = REPLICA_SEED_BASE
BIT_IDENTITY_N_EVENTS = 10_000
BIT_IDENTITY_LAYERS = (
    "topology", "exchange", "spectral_state", "legacy_kernel_finals",
)

# The observable set: the 1017-bin six-species paper window (R01 convention).
SIGMA_INEL_MB = 31.78
OBSERVABLE_WINDOW = {"xF": (0.0, 0.4), "pT": (0.05, 1.5)}
#: The pre-registration ``observable_set.species`` keys and their bin counts.
OBSERVABLE_SPECIES = {
    "pi-": 229, "pi+": 230, "K+": 153, "K-": 148, "p": 135, "pbar": 122,
}
TOTAL_BINS = 1017
#: The driver's species-key transliteration (pre-registration -> replica
#: sums/counts keys, verbatim order in the replica files).
SPECIES_TRANSLIT = {
    "pi-": "pi_minus", "pi+": "pi_plus", "K+": "k_plus",
    "K-": "k_minus", "p": "proton", "pbar": "antiproton",
}
REPLICA_SUMS_ORDER = [
    "pi_minus", "pi_plus", "k_plus", "k_minus", "proton", "antiproton",
]

# The identified set: the top-5 entries of the sealed R11 ranked residual
# map, (rank, bin_index) in rank order.
IDENTIFIED_SET = ((1, 18), (2, 31), (3, 0), (4, 32), (5, 49))

# The Gate 2 / comparison verdict pins.
N_LOCAL_EVENTS_CHECKED = TOTAL_EVENTS  # 800,000
INCLUSIVE_N_COMPARED = TOTAL_BINS
INCLUSIVE_N_FLAGGED = 132
INCLUSIVE_MAX_ABS_SIGMA = 28.466888372909363
FEED_DOWN_N_CHECKED = 7
COMPARISON_FEED_DOWN_MAX_ABS_SIGMA = "Infinity"

#: The seal manifest records the 15-file sealed set (everything on disk
#: except the three post-seal files' own digests: the seal manifest, the
#: clean-environment record, and the sums file which cannot carry its own
#: digest beyond what the manifest records).
N_SEAL_MANIFEST_FILES = 15
SEAL_TOTAL_BYTES = 637_225
#: The SHA256SUMS.txt covers the 12 payloads that can carry a digest
#: (everything in the seal manifest except SHA256SUMS.txt itself, the
#: provenance, and the gate report, which the manifest records separately).
N_SUMS_LINES = 12

PRE_SCHEMA = "na49_mixing.staged_r23_resonance_ablation.pre_registration/1"
PROV_SCHEMA = "na49_mixing.staged_r23_resonance_ablation.provenance/1"
BIT_ID_SCHEMA = "na49_mixing.staged_r23_resonance_ablation.bit_identity/1"
GATE_SCHEMA = "na49_mixing.staged_r23_resonance_ablation.gate/1"
COMPARISON_SCHEMA = "na49_mixing.staged_r23_resonance_ablation.comparison/1"
REPLICA_SEEDS_SCHEMA = (
    "na49_mixing.staged_r23_resonance_ablation.replica_manifest/1"
)
CLEAN_SCHEMA = "na49_mixing.staged_r23_resonance_ablation.clean_environment/1"

#: ``sha256sum -c`` line: 64 hex digits, two spaces, the relative path.
SUMS_LINE = re.compile(r"^[0-9a-f]{64}  \S+$")


def _on_disk_file_names() -> set[str]:
    """The 17-file sealed set present in the run dir on disk (payloads plus
    the three post-seal files: seal manifest, sums file, clean-environment
    record)."""
    return (
        {
            "pre_registration.json", "provenance.json",
            "bit_identity_report.json", "gate2_r23.json",
            "comparison_r01.json", "replica_seeds.json",
            "seal_manifest.json", "SHA256SUMS.txt",
            "clean_environment_commands.json",
        }
        | {f"replica_{seed}.json" for seed in REPLICA_SEEDS}
    )


def _seal_manifest_file_names() -> set[str]:
    """The 15-file sealed set recorded in ``seal_manifest.json``'s ``files``
    list (the payloads plus ``SHA256SUMS.txt``, the provenance, and the gate
    report, but not the seal manifest's own digest nor the clean-environment
    record)."""
    return (
        {
            "SHA256SUMS.txt", "bit_identity_report.json", "comparison_r01.json",
            "gate2_r23.json", "pre_registration.json", "provenance.json",
            "replica_seeds.json",
        }
        | {f"replica_{seed}.json" for seed in REPLICA_SEEDS}
    )


def _sums_file_names() -> set[str]:
    """The 12 payloads listed in ``SHA256SUMS.txt`` (the seal-manifest set
    minus the sums file itself, the provenance, and the gate report)."""
    return (
        {
            "bit_identity_report.json", "comparison_r01.json",
            "pre_registration.json", "replica_seeds.json",
        }
        | {f"replica_{seed}.json" for seed in REPLICA_SEEDS}
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(name: str) -> dict:
    return json.loads((RUN_DIR / name).read_text(encoding="utf-8"))


def _replica_path(seed: int) -> Path:
    return RUN_DIR / f"replica_{seed}.json"


@lru_cache(maxsize=1)
def _replicas_loaded() -> dict[int, dict]:
    return {
        seed: json.loads(_replica_path(seed).read_text(encoding="utf-8"))
        for seed in REPLICA_SEEDS
    }


@pytest.fixture(scope="module")
def run_dir():
    """The sealed R23 run dir, skipped with a clear message when the
    gitignored artifact is absent or incomplete (a fresh checkout)."""
    if not (RUN_DIR / "pre_registration.json").is_file():
        pytest.skip(
            f"the sealed R23 run dir {RUN_DIR} is absent (gitignored "
            "artifact); restore the sealed run to execute the "
            "artifact pins"
        )
    missing = sorted(
        name for name in _on_disk_file_names()
        if not (RUN_DIR / name).is_file()
    )
    if missing:
        pytest.skip(
            f"the sealed R23 run dir {RUN_DIR} is incomplete (missing: "
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
    """The run dir holds exactly the 17-file sealed set (payloads plus the
    three post-seal files: seal manifest, sums file, clean-environment
    record); nothing else is present."""
    on_disk = {path.name for path in run_dir.iterdir()}
    expected = _on_disk_file_names()
    assert len(expected) == 17
    assert on_disk == expected, (
        "the sealed run dir drifted from the committed 17-file set"
    )


def test_pre_registration_structure(run_dir, pre_registration):
    """The sealed pre-registration carries the committed contract: the T2
    statistics (8 replicas x 100,000 events), the replica seeds
    ``20300913 + k``, the M8 frozen parameter set and its hash with the
    kernel boundary 2.680194110727528 GeV, the observable set, the
    identified top-5 set from the sealed R11 ranked residual map, the bit-
    identity check config, the comparison-metrics clauses (i)-(iii), and
    ``no_fitting``.  The provenance carries the same driver and catalog
    hashes."""
    pre = pre_registration
    assert pre["schema"] == PRE_SCHEMA
    assert pre["run"] == RUN_ID
    assert pre["card"] == CARD_ID
    assert pre["written_before_generation"] is True

    stats = pre["statistics"]
    assert stats["tier"] == "T2"
    assert "8 replicas x 100,000" in stats["label"]
    assert stats["replicas"] == N_REPLICAS
    assert stats["events_per_replica"] == EVENTS_PER_REPLICA
    assert stats["total_events"] == TOTAL_EVENTS

    # the replica seeds are the M8 seed rule with the Round 2 master seed:
    # replica k (k = 0..7) runs at seed 20300913 + k
    assert pre["replica_seeds"] == REPLICA_SEEDS
    assert pre["replica_seeds"] == [
        REPLICA_SEED_BASE + k for k in range(N_REPLICAS)
    ]
    for key in ("replica_seed_rule", "legacy_event_seed_rule",
                "kernel_seed_rule"):
        assert key in pre["seed_rule"]
    assert str(KERNEL_MASTER_SEED) in pre["seed_rule"]["kernel_seed_rule"]

    params = pre["parameters"]
    assert params["source"].startswith("config/v2_singlet/legacy_v1_manifest")
    assert params["m8_parameter_hash"] == M8_PARAMETER_HASH
    assert pre["m8_parameter_hash"] == M8_PARAMETER_HASH
    m8 = params["m8_frozen_pp_parameters"]
    assert m8["continuum.match_mass"] == KERNEL_BOUNDARY_GEV
    assert params["kernel_boundary_gev"] == KERNEL_BOUNDARY_GEV
    assert params["kernel_boundary_gev"] == (
        params["m8_frozen_pp_parameters"]["continuum.match_mass"]
    )

    # the bit-identity check config pins the replica-0 seed and 10,000
    # events across the four named layers
    bit = pre["bit_identity_check"]
    assert bit["seed"] == BIT_IDENTITY_SEED
    assert bit["n_events"] == BIT_IDENTITY_N_EVENTS
    assert list(bit["layers_checked"]) == list(BIT_IDENTITY_LAYERS)
    assert set(bit["layer_definitions"]) == set(BIT_IDENTITY_LAYERS)

    # the observable set: the six-species paper window, 1017 bins total
    obs = pre["observable_set"]
    assert obs["sigma_inel_mb"] == SIGMA_INEL_MB
    assert list(obs["window"]["xF"]) == list(OBSERVABLE_WINDOW["xF"])
    assert list(obs["window"]["pT"]) == list(OBSERVABLE_WINDOW["pT"])
    assert obs["species"] == OBSERVABLE_SPECIES
    assert obs["total_bins"] == TOTAL_BINS
    assert sum(obs["species"].values()) == TOTAL_BINS
    for key in ("regge.alpha_exc0", "regge.effective_transfer_slope",
                "regge.edge_power", "continuum.pion_multiplicity_slope",
                "continuum.strange_lambda_strength",
                "continuum.strange_sigma_strength",
                "continuum.strange_pair_strength", "continuum.pair_strength",
                "continuum.pair_threshold_power"):
        assert key in m8

    # the identified set: the top-5 of the sealed R11 ranked residual map
    ident = pre["identified_set"]
    assert "r11_observable_decomposition" in ident["source"]
    assert "ranked_residual_map.json" in ident["source"]
    assert ident["sha256_r11_sums"] == R11_SUMS_SHA256
    assert [(entry["rank"], entry["bin_index"])
            for entry in ident["entries"]] == list(IDENTIFIED_SET)
    for entry in ident["entries"]:
        for key in ("sector", "observable", "region", "bin_index", "d_r01",
                    "m_r01", "sigma_r01", "delta_over_sigma_r01"):
            assert key in entry

    # the comparison-metrics clauses (i)-(iii)
    metrics = pre["comparison_metrics"]
    assert set(metrics) == {"inclusive_stability", "identified_set",
                            "feed_down"}
    assert metrics["inclusive_stability"]["n_sigma"] == 4.0
    assert "combined 4-sigma" in (
        metrics["inclusive_stability"]["clause"]
    )
    assert metrics["identified_set"]["max_abs_pull"] == 2.5
    assert metrics["identified_set"]["sum_n_sigma"] == 2.0
    assert metrics["feed_down"]["n_sigma"] == 3.0
    assert metrics["feed_down"]["min_n_per_species"] == 1000
    assert metrics["feed_down"]["mass_histogram_bins"] == 32

    # no fitting: the plan-section-8.2 external inputs, no optimizer
    assert pre["no_fitting"] is True
    assert pre["optimizer_invoked"] is False
    for key in ("config_hash", "tables_sha256", "builder_module_sha256",
                "driver_sha256", "catalog_sha256"):
        assert key in pre
    assert pre["config_hash"] == CONFIG_SHA256
    assert pre["tables_sha256"] == TABLES_SHA256
    assert pre["builder_module_sha256"] == BUILDER_MODULE_SHA256
    assert pre["driver_sha256"] == DRIVER_SHA256
    assert pre["catalog_sha256"] == CATALOG_SHA256

    # the provenance carries the same driver and catalog identity
    prov = _load("provenance.json")
    assert prov["schema"] == PROV_SCHEMA
    assert prov["run"] == RUN_ID
    assert prov["card"] == CARD_ID
    assert prov["driver"] == DRIVER
    assert prov["driver_sha256"] == DRIVER_SHA256
    assert prov["catalog"] == {
        "path": "config/v2_singlet/resonance_catalog.json",
        "sha256": CATALOG_SHA256,
    }
    assert prov["m8_parameter_hash"] == M8_PARAMETER_HASH
    assert prov["kernel_boundary_gev"] == KERNEL_BOUNDARY_GEV
    assert prov["written_before_generation"] is True
    assert prov["verification"]["config_sha256"] == CONFIG_SHA256
    assert prov["verification"]["tables_sha256"] == TABLES_SHA256
    assert prov["verification"]["catalog_sha256"] == CATALOG_SHA256
    assert prov["verification"]["r11_sums_sha256"] == R11_SUMS_SHA256
    assert prov["verification"]["r01_archive_seal_sha256"] == (
        R01_ARCHIVE_SEAL_SHA256
    )
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$",
                    prov["written_utc"])


def test_bit_identity_report_structure(run_dir, pre_registration):
    """``bit_identity_report.json`` passes all four layers (topology,
    exchange, spectral_state, legacy_kernel_finals) over 10,000 events at
    the replica-0 seed, with empty mismatch lists and the below-boundary
    source-0 skip accounting on the legacy-kernel-final layer."""
    bit = _load("bit_identity_report.json")
    assert bit["schema"] == BIT_ID_SCHEMA
    assert bit["run"] == RUN_ID
    assert bit["card"] == CARD_ID
    assert bit["seed"] == BIT_IDENTITY_SEED
    assert bit["n_events"] == BIT_IDENTITY_N_EVENTS
    assert list(bit["layer_definitions"]) == list(BIT_IDENTITY_LAYERS)
    assert bit["pass"] is True
    assert bit["legacy_stage_label"] == "r01_reference"

    # the layer set and the seed / event count agree with the
    # pre-registered bit-identity check
    pre_bit = pre_registration["bit_identity_check"]
    assert list(bit["layer_definitions"]) == list(pre_bit["layers_checked"])
    assert bit["seed"] == pre_bit["seed"]
    assert bit["n_events"] == pre_bit["n_events"]

    layers = bit["layers"]
    assert set(layers) == set(BIT_IDENTITY_LAYERS)
    for name in BIT_IDENTITY_LAYERS:
        entry = layers[name]
        assert entry["mismatches"] == [], (
            f"layer {name}: a non-empty mismatch list means the "
            "bit-identity check did not pass"
        )
        assert entry["n_matched"] == entry["n_compared"]
    # the first three layers compare all 10,000 events exactly
    for name in ("topology", "exchange", "spectral_state"):
        assert layers[name]["n_compared"] == BIT_IDENTITY_N_EVENTS
        assert layers[name]["n_matched"] == BIT_IDENTITY_N_EVENTS
    # the legacy-kernel-final layer: every one of the 2 sources per event is
    # either compared, skipped (below-boundary new-kernel side), or skipped
    # (source-0-perturbed), so the three counts partition 2 * n_events; the
    # perturbed-event count is recorded consistently
    legacy = layers["legacy_kernel_finals"]
    assert legacy["n_skipped_new_kernel"] >= 0
    assert legacy["n_skipped_perturbed"] >= 0
    assert bit["n_events_with_perturbed_source0"] == (
        legacy["n_skipped_perturbed"]
    )
    assert (
        legacy["n_compared"]
        + legacy["n_skipped_new_kernel"]
        + legacy["n_skipped_perturbed"]
    ) == 2 * BIT_IDENTITY_N_EVENTS
    # the input hashes agree with the pre-registration
    assert bit["config_sha256"] == CONFIG_SHA256
    assert bit["m8_parameter_hash"] == M8_PARAMETER_HASH
    assert bit["catalog_sha256"] == CATALOG_SHA256


def test_replica_file_schema(run_dir, pre_registration):
    """Every ``replica_<seed>.json`` carries the committed per-replica
    schema at 100,000 events: the per-species binned cross sections
    (``sums``) and counts over the 1017-bin six-species paper window, the
    feed-down fractions, the boundary statistics, and the per-event
    conservation check.  The seed matches the pre-registration and the
    replica-seeds manifest, and the binned-table key set is exactly the
    pre-registration observable set (transliterated)."""
    obs = pre_registration["observable_set"]
    replicas = _replicas_loaded()
    assert set(replicas) == set(REPLICA_SEEDS)
    for seed in REPLICA_SEEDS:
        payload = replicas[seed]
        assert payload["seed"] == seed, (
            f"replica_{seed}.json: the seed does not match the file name"
        )
        assert payload["n_events"] == EVENTS_PER_REPLICA
        assert payload["mode"] == "production"
        assert payload["stage_label"] == STAGE_LABEL
        assert payload["kernel_master_seed"] == KERNEL_MASTER_SEED
        assert payload["kernel_boundary_gev"] == KERNEL_BOUNDARY_GEV
        # the sealed input hashes agree with the pre-registration
        assert payload["config_sha256"] == CONFIG_SHA256
        assert payload["tables_sha256"] == TABLES_SHA256
        assert payload["builder_module_sha256"] == BUILDER_MODULE_SHA256
        assert payload["driver_sha256"] == DRIVER_SHA256

        # the per-species binned tables: sums and counts over exactly the
        # pre-registration observable set, in the committed transliterated
        # order, with the per-species bin counts
        for block in ("sums", "counts"):
            assert list(payload[block]) == REPLICA_SUMS_ORDER, (
                f"replica_{seed}.json: the {block} species order"
            )
            for label, bins in obs["species"].items():
                key = SPECIES_TRANSLIT[label]
                assert len(payload[block][key]) == bins, (
                    f"replica_{seed}.json: the {block} bin count for "
                    f"{key} ({len(payload[block][key])}) must equal the "
                    f"pre-registered {bins}"
                )
        assert sum(len(payload["sums"][key]) for key in
                   REPLICA_SUMS_ORDER) == TOTAL_BINS
        assert sum(len(payload["counts"][key]) for key in
                   REPLICA_SUMS_ORDER) == TOTAL_BINS
        for key in REPLICA_SUMS_ORDER:
            # the binned cross-section sums are non-negative, and the
            # binned counts are non-negative integers (the two tables are
            # distinct: ``sums`` carries the f * dxF * dpT integrands,
            # ``counts`` the per-bin event counts)
            assert all(value >= 0.0 for value in payload["sums"][key]), (
                f"replica_{seed}.json: the {key} cross-section sums must "
                "be non-negative"
            )
            assert all(
                isinstance(count, int) and count >= 0
                for count in payload["counts"][key]
            ), (
                f"replica_{seed}.json: the {key} binned counts must be "
                "non-negative integers"
            )

        # the feed-down block: per-species resonant fractions over the new-
        # kernel finals
        feed = payload["feed_down"]
        assert feed["n_finals_new_kernel"] >= 0
        assert len(feed["per_species"]) == len(
            feed["fractions_resonant"]
        )
        for species, entry in feed["per_species"].items():
            assert set(entry) == {"n", "n_resonant", "fraction_resonant",
                                  "by_parent"}
            assert entry["n"] >= 0
            assert 0 <= entry["n_resonant"] <= entry["n"]
            assert sum(entry["by_parent"].values()) == entry["n_resonant"]
            if entry["n"] == 0:
                assert entry["fraction_resonant"] == 0.0
            else:
                assert entry["fraction_resonant"] == pytest.approx(
                    entry["n_resonant"] / entry["n"], abs=1e-9
                )
            assert feed["fractions_resonant"][species] == (
                entry["fraction_resonant"]
            )

        # the boundary statistics
        boundary = payload["boundary"]
        assert boundary["n_sources"] == 2 * EVENTS_PER_REPLICA
        assert boundary["n_new_kernel"] + boundary["n_legacy_kernel"] == (
            boundary["n_sources"]
        )
        assert 0.0 <= boundary["fraction_new_kernel"] <= 1.0
        assert boundary["fraction_new_kernel"] == pytest.approx(
            boundary["n_new_kernel"] / boundary["n_sources"], abs=1e-9
        )
        assert len(boundary["mass_histogram_edges"]) == 33
        assert len(boundary["mass_histogram_counts"]) == 32
        assert sum(boundary["mass_histogram_counts"]) == (
            boundary["n_new_kernel"]
        )
        assert boundary["mass_histogram_edges"][0] < (
            boundary["mass_histogram_edges"][-1]
        )
        assert boundary["mass_histogram_edges"][-1] == KERNEL_BOUNDARY_GEV
        assert 0.0 < boundary["mean_mass_new_kernel_gev"] < (
            KERNEL_BOUNDARY_GEV
        )

        # the per-event conservation check
        conservation = payload["conservation"]
        assert conservation["events_checked"] == EVENTS_PER_REPLICA
        assert conservation["event_b_q_conserved"] is True
        assert "B = 2 and Q = 2" in conservation["convention"]


def test_replica_seeds_manifest(run_dir, pre_registration):
    """``replica_seeds.json`` records the eight completed replicas, the T2
    event budget (8 x 100,000 = 800,000 events), and the production mode;
    the seeds agree with the pre-registration."""
    manifest = _load("replica_seeds.json")
    assert manifest["schema"] == REPLICA_SEEDS_SCHEMA
    assert manifest["card"] == CARD_ID
    assert manifest["mode"] == "production"
    assert manifest["declared_replicas"] == N_REPLICAS
    assert manifest["events_per_replica"] == EVENTS_PER_REPLICA
    assert manifest["replicas_completed"] == N_REPLICAS
    assert manifest["events_completed"] == TOTAL_EVENTS

    seeds = manifest["seeds"]
    assert len(seeds) == N_REPLICAS
    for index, entry in enumerate(seeds):
        assert entry["replica_index"] == index
        assert entry["seed"] == REPLICA_SEEDS[index]
        assert entry["status"] == "completed"
        assert entry["n_events"] == EVENTS_PER_REPLICA
        assert entry["mode"] == "production"
    assert [entry["seed"] for entry in seeds] == pre_registration[
        "replica_seeds"
    ]


def test_gate2_report_structure(run_dir, pre_registration):
    """``gate2_r23.json`` carries the Gate 2 clauses with the committed
    verdict: ``local_conservation`` passes over 800,000 events,
    ``branching_test`` fails (folded from R20), ``threshold_discontinuity``
    fails (folded from R20/R21/R22), ``observables`` fails with the (i)-(iii)
    metrics, ``no_branching_fit`` records no fitting, ``bit_identity``
    passes all four layers, and the sealed form is marked."""
    gate = _load("gate2_r23.json")
    assert gate["schema"] == GATE_SCHEMA
    assert gate["gate"] == "gate 2 (plan section 8, line 575)"
    assert gate["run"] == RUN_ID
    assert gate["card"] == CARD_ID
    assert gate["sealed"] is True
    assert "decision_note" in gate
    assert "seal_note" in gate

    clauses = gate["clauses"]
    assert set(clauses) == {
        "local_conservation", "branching_test",
        "threshold_discontinuity", "observables", "no_branching_fit",
    }

    # local conservation passes over the full 800,000-event budget
    local = clauses["local_conservation"]
    assert local["event_b_q_conserved"] is True
    assert local["events_checked"] == N_LOCAL_EVENTS_CHECKED == TOTAL_EVENTS
    assert local["pass"] is True
    assert local["folded_from"] == [
        "dev/00runs/r20_single_resonance_unit_grid_2026-09-13/"
        "gate2_r20.json",
        "dev/00runs/r21_baryonic_source_grid_2026-09-13/gate2_r21.json",
        "dev/00runs/r22_mesonic_source_grid_2026-09-13/gate2_r22.json",
    ]

    # branching_test fails, folded from R20
    branching = clauses["branching_test"]
    assert branching["pass"] is False
    assert branching["folded_from"] == [
        "dev/00runs/r20_single_resonance_unit_grid_2026-09-13/"
        "gate2_r20.json",
    ]
    assert branching["fold"]["r20"]["value"] is False

    # threshold_discontinuity fails, folded from R20/R21/R22
    threshold = clauses["threshold_discontinuity"]
    assert threshold["pass"] is False
    assert threshold["folded_from"] == local["folded_from"]
    for run in ("r20", "r21", "r22"):
        assert threshold["fold"][run]["value"] is False

    # the no-branching-fit clause records no fitting
    assert clauses["no_branching_fit"] == {
        "no_fitting": True, "optimizer_invoked": False,
    }

    # observables fail, with the (i)-(iii) metrics and the identified set
    obs = clauses["observables"]
    assert obs["pass"] is False
    assert obs["identified_set_pass"] is False
    assert obs["inclusive_pass"] is False
    assert obs["feed_down_pass"] is False
    assert obs["bit_identity_pass"] is True
    assert [(entry["rank"], entry["bin_index"])
            for entry in obs["identified_set"]] == list(IDENTIFIED_SET)
    assert obs["inclusive_stability_max_abs_sigma"] == (
        INCLUSIVE_MAX_ABS_SIGMA
    )
    assert obs["feed_down_max_abs_sigma"] == COMPARISON_FEED_DOWN_MAX_ABS_SIGMA
    assert obs["r01_comparison"]["species"] == REPLICA_SUMS_ORDER
    assert obs["r01_comparison"]["max_abs_delta_over_sigma_vs_r01"] == (
        INCLUSIVE_MAX_ABS_SIGMA
    )

    # the bit-identity block agrees with the bit-identity report
    bit_block = gate["bit_identity"]
    report = _load("bit_identity_report.json")
    assert bit_block["layers_checked"] == list(BIT_IDENTITY_LAYERS)
    assert bit_block["seed"] == BIT_IDENTITY_SEED == report["seed"]
    assert bit_block["n_events"] == BIT_IDENTITY_N_EVENTS == report["n_events"]
    assert bit_block["pass"] is True == report["pass"]

    # the input hashes: the comparison and bit-identity report digests
    # match the files on disk, and the folded-gate digests match the
    # folded Gate 2 reports on disk
    assert gate["input_hashes"]["comparison_r01_sha256"] == _file_sha256(
        run_dir / "comparison_r01.json"
    )
    assert gate["input_hashes"]["bit_identity_report_sha256"] == (
        _file_sha256(run_dir / "bit_identity_report.json")
    )
    for run, entry in gate["input_hashes"]["folded_gates"].items():
        assert entry["path"].startswith(str(MONO_ROOT))
        assert _file_sha256(Path(entry["path"])) == entry["sha256"], (
            f"the folded gate digest for {run} does not match the "
            "on-disk Gate 2 report"
        )


def test_comparison_report_structure(run_dir, pre_registration):
    """``comparison_r01.json`` carries the R01 comparison: the f/sigma_inel
    normalization convention, the 1017-bin six-species pooled predictions,
    the (i)-(iii) metrics against the sealed R01 archive, the boundary and
    feed-down blocks, the conservation check, and the input hashes (which
    agree with the pre-registration)."""
    comp = _load("comparison_r01.json")
    assert comp["schema"] == COMPARISON_SCHEMA
    assert comp["run"] == RUN_ID
    assert comp["card"] == CARD_ID
    assert comp["n_replicas"] == N_REPLICAS
    assert comp["replica_seeds"] == pre_registration["replica_seeds"]
    assert comp["n_events_total"] == TOTAL_EVENTS
    assert comp["no_fitting"] is True
    assert comp["optimizer_invoked"] is False

    # the normalization convention (f / sigma_inel, per inelastic event)
    norm = comp["normalization"]
    assert "f / sigma_inel" in norm["convention"]
    assert norm["sigma_inel_mb"] == SIGMA_INEL_MB
    assert list(norm["window"]["xF"]) == list(OBSERVABLE_WINDOW["xF"])
    assert list(norm["window"]["pT"]) == list(OBSERVABLE_WINDOW["pT"])

    # the pooled predictions: one row per paper-window bin, 1017 total
    preds = comp["pooled_predictions"]
    assert len(preds) == TOTAL_BINS
    for row in preds:
        assert set(row) == {"species", "bin_index", "xF_mid", "pT_mid",
                            "value", "stat_error"}
        assert row["species"] in REPLICA_SUMS_ORDER
        assert row["stat_error"] >= 0.0

    # the (i)-(iii) metrics
    metrics = comp["metrics"]
    assert set(metrics) == {"inclusive_stability", "identified_set",
                            "feed_down"}

    inclusive = metrics["inclusive_stability"]
    summary = inclusive["summary"]
    assert summary["n_compared"] == INCLUSIVE_N_COMPARED == TOTAL_BINS
    assert summary["n_flagged"] == INCLUSIVE_N_FLAGGED
    assert summary["max_abs_delta_over_sigma"] == INCLUSIVE_MAX_ABS_SIGMA
    assert summary["n_sigma"] == 4.0
    assert summary["pass"] is False
    for row in inclusive["flagged_rows"]:
        assert row["within_4_sigma"] is False
        assert abs(row["delta_over_sigma"]) > 4.0
        assert row["sigma_combined"] > 0.0
        assert abs(row["delta_over_sigma"]) == pytest.approx(
            abs(row["r23_value"] - row["r01_value"]) / row["sigma_combined"],
            abs=1e-9,
        )

    identified = metrics["identified_set"]
    ident_summary = identified["summary"]
    assert [(entry["rank"], entry["bin_index"])
            for entry in ident_summary["entries"]] == list(IDENTIFIED_SET)
    assert ident_summary["max_abs_pull"] == 2.5
    assert ident_summary["sum_n_sigma"] == 2.0
    assert ident_summary["gross_violations"] == 5
    assert ident_summary["pass"] is False
    assert len(identified["cells"]) == len(IDENTIFIED_SET)

    feed = metrics["feed_down"]
    feed_summary = feed["summary"]
    assert feed_summary["n_checked"] == FEED_DOWN_N_CHECKED
    assert feed_summary["min_n_per_species"] == 1000
    assert feed_summary["max_abs_sigma"] == (
        COMPARISON_FEED_DOWN_MAX_ABS_SIGMA
    )
    assert feed_summary["n_sigma"] == 3.0
    assert feed_summary["pass"] is False

    # the boundary and feed-down blocks
    boundary = comp["boundary"]
    assert boundary["n_sources"] == 2 * TOTAL_EVENTS
    assert boundary["n_new_kernel"] + boundary["n_legacy_kernel"] == (
        boundary["n_sources"]
    )
    assert boundary["boundary_gev"] == KERNEL_BOUNDARY_GEV
    assert boundary["boundary_inclusive"] is True
    assert len(boundary["mass_histogram_edges"]) == 33
    assert len(boundary["mass_histogram_counts"]) == 32
    assert sum(boundary["mass_histogram_counts"]) == boundary["n_new_kernel"]

    feed_block = comp["feed_down"]
    assert feed_block["n_finals_new_kernel"] >= 0
    assert len(feed_block["per_species"]) == len(
        feed_block["fractions_resonant"]
    )

    # the conservation check over the full budget
    assert comp["conservation"]["events_checked"] == TOTAL_EVENTS
    assert comp["conservation"]["event_b_q_conserved"] is True

    # the R01 archive reference and the R11 ranked map
    assert comp["r01_archive"]["archive_seal_sha256"] == (
        R01_ARCHIVE_SEAL_SHA256
    )
    assert comp["r11_ranked_map"]["top5"][0]["bin_index"] == IDENTIFIED_SET[0][1]
    assert [(entry["rank"], entry["bin_index"])
            for entry in comp["r11_ranked_map"]["top5"]] == (
        list(IDENTIFIED_SET)
    )

    # the input hashes agree with the pre-registration
    inputs = comp["input_hashes"]
    assert inputs["config_sha256"] == CONFIG_SHA256
    assert inputs["tables_sha256"] == TABLES_SHA256
    assert inputs["table_data_digest"] == TABLE_DATA_DIGEST
    assert inputs["catalog_sha256"] == CATALOG_SHA256
    assert inputs["r11_sums_sha256"] == R11_SUMS_SHA256
    assert inputs["r01_sums_sha256"] == R01_ARCHIVE_SEAL_SHA256
    assert inputs["driver_sha256"] == DRIVER_SHA256
    assert inputs["builder_module_sha256"] == BUILDER_MODULE_SHA256


def test_seal_manifest_and_sha256_sums(run_dir):
    """``seal_manifest.json`` records the 15-file sealed set and its
    digest, and ``SHA256SUMS.txt`` is in ``sha256sum -c`` format with
    run-dir-relative paths and verifies against every sealed file on disk.
    The sums cover the 12 payloads that can carry a digest; the digest of
    the sums file matches the seal manifest's recorded value."""
    manifest = _load("seal_manifest.json")
    assert manifest["run_directory"] == RUN_NAME
    assert manifest["card"] == CARD_ID
    assert "sealed_at_utc" in manifest
    re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$",
             manifest["sealed_at_utc"])

    files = manifest["files"]
    assert len(files) == N_SEAL_MANIFEST_FILES
    assert {entry["path"] for entry in files} == _seal_manifest_file_names()
    assert sum(entry["bytes"] for entry in files) == SEAL_TOTAL_BYTES
    for entry in files:
        assert _file_sha256(run_dir / entry["path"]) == entry["sha256"], (
            f"the seal-manifest digest of {entry['path']} does not "
            "match the file on disk; the run dir was modified after "
            "the seal"
        )

    sums_path = run_dir / "SHA256SUMS.txt"
    text = sums_path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert "\r" not in text
    lines = text.splitlines()
    assert len(lines) == N_SUMS_LINES
    entries = []
    for line in lines:
        assert SUMS_LINE.match(line), (
            f"SHA256SUMS.txt is not in sha256sum -c format: {line!r}"
        )
        digest, rel = line.split("  ", 1)
        # the sums paths are run-dir-relative (no directory prefix)
        assert "/" not in rel, (
            f"SHA256SUMS.txt paths must be run-dir-relative: {rel!r}"
        )
        entries.append((digest, rel))
    assert {rel for _digest, rel in entries} == _sums_file_names()
    for digest, rel in entries:
        assert _file_sha256(run_dir / rel) == digest, (
            f"the sealed digest of {rel} does not match the file on "
            "disk; the run dir was modified after the seal"
        )
    # the sums file's own digest is recorded in the seal manifest and is
    # self-consistent
    assert _file_sha256(sums_path) == (
        next(entry["sha256"] for entry in files
             if entry["path"] == "SHA256SUMS.txt")
    )


def test_clean_environment_validation_record(run_dir):
    """The post-seal clean-environment record re-verified all 12
    sums-covered files without mismatches, and the clean-interpreter
    re-run (``python -I``) reproduced the sealed R01 comparison
    byte-for-byte (the recorded sealed and recomputed digests match and
    equal the digest of the comparison on disk)."""
    record = _load("clean_environment_commands.json")
    assert record["schema"] == CLEAN_SCHEMA
    assert record["run"] == RUN_ID
    assert record["card"] == CARD_ID
    assert record["seal_verification"] == {
        "files_checked": N_SUMS_LINES,
        "mismatches": [],
    }
    assert record["probe_returncode"] == 0
    assert record["hashes_match"] is True
    for key in ("python_version", "python_executable", "platform",
                "numpy_version"):
        assert key in record["environment"]

    comp_digest = _file_sha256(run_dir / "comparison_r01.json")
    assert record["sealed_report_sha256"] == {
        "comparison_r01.json": comp_digest
    }
    assert record["recomputed_report_sha256"] == {
        "comparison_r01.json": comp_digest
    }
    # the re-run is the clean-interpreter probe of the committed dual-tree
    # driver, run from the monorepo root
    command = record["commands"][0]
    assert command["cwd"] == str(MONO_ROOT)
    assert "-I" in command["argv"]
    assert DRIVER in " ".join(command["argv"])
    assert record["probe_tool"]["sha256"] == DRIVER_SHA256
