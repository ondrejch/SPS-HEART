"""Coverage pins for the R02 legacy p+A reference driver.

Companion to ``tests/test_r02_reference_harness.py`` (the six card-named
hermetic checks of card ``TASK-20260911-r1-run-r02-legacy-pa-reference``);
this module pins the committed behavior around them.  All checks are
hermetic and small-N: the only event generation is two 10-event driver
subprocesses (the card's subprocess ceiling is 100 events); the
hand-constructed batches are built from plain arrays and stub particles
with no generation at all; the 4,800,000-event T2 run is the R02 run
itself, not a test.  The read-only cross-checks against the sealed
4.8M-event run directory and the M13 pre-registration file skip with a
recorded reason when the artifact store is absent from the checkout.

CLI surface (subprocess, all refusals fire before the ``--out`` directory
is created or a single event is generated):

- ``test_cli_unknown_arg_refused`` and
  ``test_cli_missing_required_arg_refused``: an unknown flag or a
  missing required argument is an argparse refusal (exit 2).
- ``test_cli_nonpositive_replicas_refused_before_generation`` and
  ``test_cli_nonpositive_events_refused_before_generation``: 0 and
  negative ``--replicas`` / ``--events-per-replica`` are refused with
  the exit-2 refusal text.
- ``test_cli_unavailable_target_refused_before_generation``: the
  recorded unavailable targets C and W are refused with the "is
  unavailable" text (a new target files its own card).
- ``test_cli_class_list_refusals``: an unknown class, a duplicated
  class, and an empty class list are each refused with the committed
  refusal text.
- ``test_cli_pa_config_name_resolves_onto_bundled_config``: the card's
  ``config/staged_excitation_pA.json`` CLI path (the file does not exist
  in the frozen tree) resolves onto the bundled production config: the
  driver announces the resolution, the written pre-registration equals
  the card's payload, and the run manifest records the production
  config sha256 pin.

Config resolution and verification (in-process, no generation):

- ``test_config_pins_match_card1_manifest``: the driver's embedded
  config sha256 pins equal card 1's manifest ``source_tree_sha256``
  inventory entries and the actual file digests; the pristine
  production and smoke configs verify in-process; the card's pA CLI
  name maps onto the production pin.
- ``test_verify_preregistration_mutations_refused``: a mutated
  pre-registration ``config_hash`` or ``m8_parameter_hash`` refuses
  with exit code 2 before any generation.

Seed derivation (in-process, no generation):

- ``test_class_replica_seed_rule_pinned``: ``class_replica_seeds``
  applies ``seed_base + 50000 + 100*i + k`` (the hand-built lists for
  the default base and the full T2 tier are pinned), and the offset
  base 50000 is provably disjoint from R01's 40000 stream (the gap
  10000 exceeds the per-class spread 5*100 + 7).
- ``test_default_pre_registration_identity``: the default
  pre-registration payload carries the card's verbatim identity
  (target, classes, E_BE values, seed rule, T2 tier, hashes,
  normalization, hemisphere rule).

Hemisphere rule (in-process, no generation):

- ``test_hemisphere_slot_boundaries``: the pre-registered xF split at
  the boundaries themselves (xF = 0.2 and xF = 0.0 belong to central;
  negative zero is not a target-side track).
- ``test_hemisphere_payload_zero_and_error_contract``: an absent
  hemisphere region carries exactly ``0.0`` (serialized ``0.0``, not
  ``-0.0`` or NaN), and the payload refuses a non-positive event
  count, a missing species, and a wrong region count.

Data-side matching and model-only species (in-process over the loaded
tables and the committed ``pa_data`` files; no generation):

- ``test_pa_data_cell_match_key_convention``: the minimum-bias class
  carries exactly the two published pion tables (K+, K-, p, pbar are
  model-only); the match key is the (xF_mid, pT_mid) pair rounded to
  1e-9 (a 1e-11 query offset still matches, an off-grid cell does
  not); the loader's sha256 field re-hashes the file.
- ``test_residuals_matched_null_and_chi2_convention``: with a
  zero-signal hand batch, every matched bin carries the table value
  ``d`` and an infinite combined sigma (pull 0.0), every unmatched
  bin carries all-null data fields, model-only species carry null
  chi2 with the recorded reason, and the per-species chi2 DOF is
  ``data-matched bins - 1``.
- ``test_empty_bins_infinite_mc_encoding_matches_r01``: a batch with
  one populated bin keeps all 1017 bins; the committed JSON encoding
  of the infinite MC error is the string sentinel ``"Infinity"`` (the
  R01 convention, RFC 8259 safe; the R02 writer's artifact
  strict-parses and ``float()`` restores ``inf``), and the R02
  combined covariance's ``V_mc`` agrees entry for entry with the R01
  driver's over the same arrays (the cross-driver convention pin).

Centrality mapping (in-process, no generation):

- ``test_centrality_mapping_constants_pinned``: the embedded M13
  mapping constants (the five wounded-K bands, the calibrated mean
  target-wound numbers, the measured thesis nu) are pinned verbatim.
- ``test_centrality_mapping_cross_checks_m13_preregistration``: when
  the M13 pre-registration file exists, the driver's runtime
  cross-check verifies it, and the closed-form slice fractions of the
  pinned geometry match the pre-registered ``slice_fraction_model``
  values.

Gate-0 assembly (hand-built run directories, no generation):

- ``test_gate0_tier_status_complete_and_deferred``: 6/6 classes with
  8/8 replicas assemble ``tier_status = "complete"`` with a null
  deferral reason; 5/6 classes (the sixth not started) and 6/6
  classes at 7/8 replicas each assemble ``"deferred"`` with the
  recorded reason; the run manifest and the acceptance record
  classify through the single strict predicate and cannot disagree;
  a deferral does not block the verdict.
- ``test_gate0_immutability_flips_on_payload_mutation``: the
  acceptance record re-verifies every ``SHA256SUMS.txt`` entry at
  assembly time; a payload mutation after sealing flips ``immutable``
  to false and the verdict to ``fail``.

Seal (subprocess, 10 events, the card's subprocess ceiling):

- ``test_cli_seal_small_run``: the ``--seal`` path on a fresh 10-event
  minimum-bias smoke run writes the per-class artifacts plus the
  immutable-archive triple; ``SHA256SUMS.txt`` is in ``sha256sum -c``
  format over every payload file and re-hashes cleanly,
  ``seal_manifest.json`` excludes itself, ``provenance.json`` carries
  the run identity (pre-registered seed base, the actual run seeds,
  the E_BE values, the salted code fingerprint), and the per-class
  chi2 recorded in the run manifest equals the per-species sum of
  squared pulls recomputed from the sealed ``residuals`` file.  No
  ``gate0_r02.json`` is written for the non-canonical smoke run.

Sealed-run cross-checks (read-only, skip with a recorded reason when
the artifact store is absent):

- ``test_sealed_r02_run_pins_seed_rule_and_tier``: the sealed 4.8M
  event archive pins the committed seed rule (every per-class seed
  list equals ``20260911 + 50000 + 100*i + k``, k = 0..7), the T2
  completion (6 classes x 8 replicas x 100,000 events), the
  minimum-bias matched-bin count (118 per pion species, chi2 dof
  117), and the gate0 acceptance record (verdict pass, complete
  tier, the recorded flagged-cell and empty-bin counts).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from helpers.benchmarking import sps_core
from helpers.benchmarking import staged_r01_reference as r01
from helpers.benchmarking import staged_r02_reference as r02
from helpers.benchmarking.benchmark_sps_species import RESONANCE_DECAY_LIB_VERSION
from na49_mixing.reaction import Reaction
from na49_mixing.glauber import NucleusSpec, WoundedNucleonSampler

NESTED_ROOT = Path(r02.NESTED_ROOT)
MONO_ROOT = Path(r02.MONO_ROOT)
DRIVER_PATH = NESTED_ROOT / "helpers" / "benchmarking" / "staged_r02_reference.py"
SMOKE_CONFIG = NESTED_ROOT / "config" / "staged_excitation_smoke.json"
PRODUCTION_CONFIG = NESTED_ROOT / "config" / "staged_excitation.json"
MANIFEST = NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json"

# Card seed base and the smoke-run seed-base convention (shared with the
# R01 coverage module).
SEED_BASE = 20260911
SMOKE_SEED_BASE = 20311000
# The smoke run's minimum-bias replica seed: 20311000 + 50000 + 100*0 + 0.
SMOKE_MB_SEED = SMOKE_SEED_BASE + r02.SEED_OFFSET_BASE
SEAL_N_EVENTS = 10  # well under the card's 100-event subprocess ceiling
SUBPROCESS_TIMEOUT = 600

# The sealed T2 run directory (gitignored artifact store, read-only).
SEALED_R02_DIR = MONO_ROOT / "dev" / "00runs" / "r02_legacy_pa_reference_2026-09-11"
_SEALED_R02_PRESENT = (SEALED_R02_DIR / "run_manifest.json").is_file() and (
    SEALED_R02_DIR / "gate0_r02.json"
).is_file()

# The 1017-bin grid is read from the bundled SPS tables; the tests that
# need the loaded tables skip with a recorded reason when the reference
# mirror is absent from the checkout.
SPS_DATA = NESTED_ROOT / "spshadrons.web.cern.ch" / "data"
_SPS_TABLE_NAMES = (
    "xsect_piminus.txt",
    "xsect_piplus.txt",
    "xsect_kaplus_pp.txt",
    "xsect_kaminus_pp.txt",
    "xsect_proton_pp.txt",
    "xsect_antiproton_pp.txt",
)
SPS_TABLES_PRESENT = all((SPS_DATA / name).is_file() for name in _SPS_TABLE_NAMES)
requires_sps_tables = pytest.mark.skipif(
    not SPS_TABLES_PRESENT,
    reason=(
        "reference SPS mirror not present in this checkout; the 1017-bin "
        "grid cannot be loaded"
    ),
)

# The M13 pre-registration (gitignored artifact store; the driver's
# embedded constants stand alone when it is absent).
M13_PREREG = MONO_ROOT / r02.M13_PREREG_RELPATH
requires_m13_prereg = pytest.mark.skipif(
    not M13_PREREG.is_file(),
    reason=(
        f"M13 pre-registration file absent from this checkout ({r02.M13_PREREG_RELPATH}); "
        "the embedded centrality constants stand alone"
    ),
)
requires_sealed_r02 = pytest.mark.skipif(
    not _SEALED_R02_PRESENT,
    reason=(
        "sealed R02 T2 run directory not present on this checkout "
        f"({SEALED_R02_DIR}); read-only cross-check skipped"
    ),
)

SPECIES_KEYS = ("pi_minus", "pi_plus", "k_plus", "k_minus", "proton", "antiproton")
MODEL_ONLY_SPECIES = ("k_plus", "k_minus", "proton", "antiproton")
SPECIES_BIN_SCHEMA = {
    "pi_minus": 229,
    "pi_plus": 230,
    "k_plus": 153,
    "k_minus": 148,
    "proton": 135,
    "antiproton": 122,
}


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _run_r02(args, out_dir: Path):
    """One driver subprocess invocation from the nested tree root."""
    return subprocess.run(
        [sys.executable, str(DRIVER_PATH)]
        + [str(a) for a in args]
        + ["--out", str(out_dir)],
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
        cwd=str(NESTED_ROOT),
    )


def _base_args() -> list:
    """The required arguments shared by the refusal-path subprocess tests."""
    return [
        "--config",
        str(SMOKE_CONFIG),
        "--replicas",
        "1",
        "--events-per-replica",
        "1",
        "--seed-base",
        str(SMOKE_SEED_BASE),
    ]


def _harness_sqrt_s() -> float:
    return math.sqrt(Reaction(158.0, 2212, 2212).s())


def _interior_target(table) -> tuple[int, int, int, float, float]:
    """A table bin whose tabulated center sits strictly inside its edges.

    The driver bins at the ``sps_core.grid_indices`` (digitize) edges of
    ``edges_from_centers``; a particle placed at the tabulated center is
    placed deterministically in that bin.  The first such bin with
    ``ix >= 1 and ip >= 1`` is returned, with the placement re-verified
    through the committed ``grid_indices`` convention at run time.
    """
    xedges = sps_core.edges_from_centers(table.xf_centers)
    pedges = sps_core.edges_from_centers(table.pt_centers)
    for i in range(int(table.data.size)):
        ix = int(table.bin_index_xf[i])
        ip = int(table.bin_index_pt[i])
        if ix < 1 or ip < 1:
            continue
        x = float(table.xf_centers[ix])
        p = float(table.pt_centers[ip])
        if not (xedges[ix] < x < xedges[ix + 1] and pedges[ip] < p < pedges[ip + 1]):
            continue
        got_ix, got_ip = sps_core.grid_indices(
            np.array([x]), np.array([p]), xedges, pedges
        )
        if int(got_ix[0]) == ix and int(got_ip[0]) == ip:
            return i, ix, ip, x, p
    raise AssertionError(f"no strictly interior bin found for {table.species}")


def _matched_bin_count(table, species_data: dict | None) -> int:
    """The model bins whose tabulated centers match a published cell."""
    return sum(
        1
        for i in range(int(table.data.size))
        if r02._match_pa_cell(
            species_data,
            float(table.xf_centers[table.bin_index_xf[i]]),
            float(table.pt_centers[table.bin_index_pt[i]]),
        )
        is not None
    )


@pytest.fixture(scope="module")
def cov_tables() -> list:
    """The windowed six-species tables (the 1017-bin grid)."""
    raw_config = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    return r02.load_binned_tables(raw_config)


# ---------------------------------------------------------------------------
# CLI surface (subprocess refusals, no generation)
# ---------------------------------------------------------------------------


def test_cli_unknown_arg_refused(tmp_path):
    """An unknown flag is an argparse refusal before --out is created."""
    out_dir = tmp_path / "unknown_arg"
    assert not out_dir.exists()
    proc = _run_r02(_base_args() + ["--bogus-flag"], out_dir)
    assert proc.returncode == 2
    assert "unrecognized arguments" in proc.stderr
    assert not out_dir.exists()


def test_cli_missing_required_arg_refused(tmp_path):
    """A missing required argument is an argparse refusal (exit 2)."""
    out_dir = tmp_path / "missing_arg"
    proc = subprocess.run(
        [
            sys.executable,
            str(DRIVER_PATH),
            "--config",
            str(SMOKE_CONFIG),
            "--replicas",
            "1",
            "--events-per-replica",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
        cwd=str(NESTED_ROOT),
    )
    assert proc.returncode == 2
    assert "--out" in proc.stderr
    assert not out_dir.exists()


@pytest.mark.parametrize("replicas", [0, -1])
def test_cli_nonpositive_replicas_refused_before_generation(tmp_path, replicas):
    """0 and negative --replicas are refused before --out is created."""
    out_dir = tmp_path / f"replicas_{replicas}"
    args = [
        "--config",
        str(SMOKE_CONFIG),
        "--replicas",
        str(replicas),
        "--events-per-replica",
        "1",
        "--seed-base",
        str(SMOKE_SEED_BASE),
    ]
    proc = _run_r02(args, out_dir)
    assert proc.returncode == 2
    assert "must be a positive integer" in proc.stderr
    assert not out_dir.exists()


@pytest.mark.parametrize("events", [0, -1])
def test_cli_nonpositive_events_refused_before_generation(tmp_path, events):
    """0 and negative --events-per-replica are refused before generation."""
    out_dir = tmp_path / f"events_{events}"
    args = [
        "--config",
        str(SMOKE_CONFIG),
        "--replicas",
        "1",
        "--events-per-replica",
        str(events),
        "--seed-base",
        str(SMOKE_SEED_BASE),
    ]
    proc = _run_r02(args, out_dir)
    assert proc.returncode == 2
    assert "must be a positive integer" in proc.stderr
    assert not out_dir.exists()


@pytest.mark.parametrize("target", ["C", "W"])
def test_cli_unavailable_target_refused_before_generation(tmp_path, target):
    """The recorded unavailable targets C and W are refused (exit 2)."""
    out_dir = tmp_path / f"target_{target}"
    proc = _run_r02(_base_args() + ["--target", target], out_dir)
    assert proc.returncode == 2
    assert "is unavailable" in proc.stderr
    assert "Pb208" in proc.stderr
    assert not out_dir.exists()


@pytest.mark.parametrize(
    "raw, message",
    [
        ("minimum_bias,set_6", "unknown centrality classes"),
        ("minimum_bias,minimum_bias", "duplicated centrality classes"),
        (" , ", "must list at least one class"),
    ],
)
def test_cli_class_list_refusals(tmp_path, raw, message):
    """An unknown, duplicated, or empty class list is refused (exit 2)."""
    out_dir = tmp_path / "classes"
    proc = _run_r02(_base_args() + ["--centrality-classes", raw], out_dir)
    assert proc.returncode == 2
    assert message in proc.stderr
    assert not out_dir.exists()


# ---------------------------------------------------------------------------
# Config resolution and verification (in-process, no generation)
# ---------------------------------------------------------------------------


def test_config_pins_match_card1_manifest():
    """The driver's embedded pins equal card 1's manifest inventory."""
    inventory = _load(MANIFEST)["source_tree_sha256"]
    assert inventory[r02.PRODUCTION_CONFIG_RELPATH] == r02.PRODUCTION_CONFIG_SHA256
    assert inventory[r02.SMOKE_CONFIG_RELPATH] == r02.SMOKE_CONFIG_SHA256
    assert r02.sha256_file(PRODUCTION_CONFIG) == r02.PRODUCTION_CONFIG_SHA256
    assert r02.sha256_file(SMOKE_CONFIG) == r02.SMOKE_CONFIG_SHA256

    pristine = r02.verify_config_and_m8(
        PRODUCTION_CONFIG, r02.default_pre_registration()
    )
    assert pristine["config_sha256"] == r02.PRODUCTION_CONFIG_SHA256
    assert pristine["production_config"] is True
    assert pristine["config_name"] == r02.PRODUCTION_CONFIG_NAME
    assert pristine["manifest_config_pin"] == r02.PRODUCTION_CONFIG_SHA256
    assert pristine["m8_parameter_hash"] == r02.M8_PARAMETER_HASH

    smoke = r02.verify_config_and_m8(SMOKE_CONFIG, r02.default_pre_registration())
    assert smoke["config_sha256"] == r02.SMOKE_CONFIG_SHA256
    assert smoke["production_config"] is False
    assert smoke["manifest_config_pin"] == r02.SMOKE_CONFIG_SHA256

    # The card's pA CLI name resolves onto the pinned production bytes
    # (no separate p+A config file exists in the frozen tree).
    assert r02._config_pin(r02.PA_CONFIG_NAME) == (
        r02.PRODUCTION_CONFIG_SHA256,
        r02.PRODUCTION_CONFIG_RELPATH,
    )
    assert r02._config_pin(r02.PRODUCTION_CONFIG_NAME) == (
        r02.PRODUCTION_CONFIG_SHA256,
        r02.PRODUCTION_CONFIG_RELPATH,
    )
    assert r02._config_pin(r02.SMOKE_CONFIG_NAME) == (
        r02.SMOKE_CONFIG_SHA256,
        r02.SMOKE_CONFIG_RELPATH,
    )
    with pytest.raises(SystemExit) as excinfo:
        r02._config_pin("some_other_config.json")
    assert excinfo.value.code == 2


def test_verify_preregistration_mutations_refused():
    """A mutated pre-registration config_hash or M8 hash refuses (exit 2)."""
    drifted = r02.default_pre_registration()
    drifted["config_hash"] = "0" * 64
    with pytest.raises(SystemExit) as excinfo:
        r02.verify_config_and_m8(PRODUCTION_CONFIG, drifted)
    assert excinfo.value.code == 2

    drifted = r02.default_pre_registration()
    drifted["m8_parameter_hash"] = "f" * 64
    with pytest.raises(SystemExit) as excinfo:
        r02.verify_config_and_m8(PRODUCTION_CONFIG, drifted)
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# Seed derivation (in-process, no generation)
# ---------------------------------------------------------------------------


def test_class_replica_seed_rule_pinned():
    """Replica seeds are seed_base + 50000 + 100*i + k; disjoint from R01."""
    # The card's canonical minimum-bias seeds, k = 0..2.
    assert r02.class_replica_seeds(SEED_BASE, 0, 3) == [
        20310911,
        20310912,
        20310913,
    ]
    # A smoke seed base and a class index one.
    assert r02.class_replica_seeds(SMOKE_SEED_BASE, 1, 1) == [
        SMOKE_SEED_BASE + 50000 + 100
    ]
    # The full declared T2 tier (6 classes x 8 replicas).
    t2_seeds = [
        r02.class_replica_seeds(SEED_BASE, i, r02.T2_REPLICAS)
        for i in range(len(r02.CENTRALITY_CLASSES))
    ]
    for i, seeds in enumerate(t2_seeds):
        assert seeds == [
            SEED_BASE + 50000 + 100 * i + k for k in range(r02.T2_REPLICAS)
        ]
    # Disjointness from the R01 stream: the offset-base gap 10000 exceeds
    # the R02 per-class spread (5 * 100) plus the replica span (7), so
    # no R02 T2 seed collides with any R01 T3 seed at the same base.
    assert r02.SEED_OFFSET_BASE - r01.SEED_OFFSET == 10000
    assert (
        (len(r02.CENTRALITY_CLASSES) - 1) * r02.SEED_OFFSET_PER_CLASS
        + (r02.T2_REPLICAS - 1)
        < r02.SEED_OFFSET_BASE - r01.SEED_OFFSET
    )
    r01_stream = set(r01.replica_seeds(r01.SEED_BASE, 20))
    assert not (set(sum(t2_seeds, [])) & r01_stream)


def test_default_pre_registration_identity():
    """The default pre-registration carries the card's verbatim identity."""
    p = r02.default_pre_registration()
    assert p["target"] == "Pb208"
    assert p["unavailable_targets"] == ["C", "W"]
    assert p["centrality_classes"] == list(r02.CENTRALITY_CLASSES)
    assert p["e_be_per_class"] == {
        name: r02.E_BE_K_BAND_MEAN[name] for name in r02.CENTRALITY_CLASSES
    }
    assert p["seed_base"] == r02.SEED_BASE == 20260911
    assert p["seed_offset_base"] == r02.SEED_OFFSET_BASE == 50000
    assert p["seed_offset_per_class"] == r02.SEED_OFFSET_PER_CLASS == 100
    assert p["replicas_t2"] == r02.T2_REPLICAS == 8
    assert p["events_per_replica_t2"] == r02.T2_EVENTS_PER_REPLICA == 100000
    assert p["config_hash"] == r02.PRODUCTION_CONFIG_SHA256
    assert p["m8_parameter_hash"] == r02.M8_PARAMETER_HASH
    assert p["sigma_inel_mb"] == r02.SIGMA_INEL_MB == 31.78
    assert p["hemisphere_rule"] == dict(r02.HEMISPHERE_RULE)
    assert p["tier"] == {
        "declared": "T2",
        "declared_replicas": 8,
        "declared_events_per_replica": 100000,
    }


# ---------------------------------------------------------------------------
# Config resolution (subprocess, 10 events)
# ---------------------------------------------------------------------------


@requires_sps_tables
def test_cli_pa_config_name_resolves_onto_bundled_config(tmp_path):
    """The card's pA CLI path resolves onto the pinned production bytes."""
    out_dir = tmp_path / "pa_name_resolution"
    assert not out_dir.exists()
    proc = _run_r02(
        [
            # The relative name: no such file exists in the frozen tree,
            # so the driver's committed resolution path is exercised.
            "--config",
            "config/" + r02.PA_CONFIG_NAME,
            "--centrality-classes",
            "minimum_bias",
            "--replicas",
            "1",
            "--events-per-replica",
            str(SEAL_N_EVENTS),
            "--seed-base",
            str(SMOKE_SEED_BASE),
        ],
        out_dir,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"resolves onto the bundled {r02.PRODUCTION_CONFIG_NAME}" in proc.stdout
    # The written pre-registration equals the card's payload exactly.
    assert _load(out_dir / r02.PREREGISTRATION_NAME) == r02.default_pre_registration()
    manifest = _load(out_dir / r02.RUN_MANIFEST_NAME)
    assert manifest["config_sha256"] == r02.PRODUCTION_CONFIG_SHA256
    assert manifest["m8_parameter_hash"] == r02.M8_PARAMETER_HASH
    assert manifest["classes"]["minimum_bias"]["seeds"] == [SMOKE_MB_SEED]
    predictions = _load(out_dir / r02.bin_predictions_name("minimum_bias"))
    assert len(predictions) == 1017


# ---------------------------------------------------------------------------
# Hemisphere rule (in-process, no generation)
# ---------------------------------------------------------------------------


def test_hemisphere_slot_boundaries():
    """The pre-registered xF split at the boundaries themselves."""
    # The forward threshold belongs to central (0.0 <= xF <= 0.2).
    assert r02._hemisphere_slot(0.2) == 1
    assert r02._hemisphere_slot(0.2 + 1e-12) == 0
    # The xF = 0.0 boundary belongs to central (target is xF < 0.0).
    assert r02._hemisphere_slot(0.0) == 1
    assert r02._hemisphere_slot(-1e-12) == 2
    # Negative zero is not a target-side track.
    assert r02._hemisphere_slot(-0.0) == 1
    # Interior points.
    assert r02._hemisphere_slot(0.4) == 0
    assert r02._hemisphere_slot(0.1) == 1
    assert r02._hemisphere_slot(-0.5) == 2
    # The pre-registration records the rule verbatim.
    assert r02.default_pre_registration()["hemisphere_rule"] == {
        "projectile_side": "xF > 0.2",
        "central": "0.0 <= xF <= 0.2",
        "target_side": "xF < 0.0",
    }


def test_hemisphere_payload_zero_and_error_contract():
    """Absent regions carry exactly 0.0; bad inputs are refused."""
    sqrt_s = _harness_sqrt_s()
    sums = {s: [1.0, 2.0, 0.0] for s in r02.SPECIES_DISPLAY}
    entries = r02.hemisphere_yields_payload(sums, 4, sqrt_s)
    norm = math.pi * sqrt_s * 4
    assert [e["species"] for e in entries] == list(r02.SPECIES_DISPLAY)
    for entry in entries:
        assert set(entry) == {"species", "projectile_side", "central", "target_side"}
        assert math.isfinite(entry["projectile_side"])
        assert math.isfinite(entry["central"])
        assert entry["projectile_side"] == pytest.approx(1.0 / norm, rel=0.0, abs=1e-15)
        assert entry["central"] == pytest.approx(2.0 / norm, rel=0.0, abs=1e-15)
        # An absent region is exactly 0.0: the serialization must not
        # carry the negative-zero or NaN form.
        assert entry["target_side"] == 0.0
        assert json.dumps(entry["target_side"]) == "0.0"
    with pytest.raises(ValueError):
        r02.hemisphere_yields_payload(sums, 0, sqrt_s)
    with pytest.raises(ValueError):
        r02.hemisphere_yields_payload({"pi_minus": [1.0, 2.0, 0.0]}, 4, sqrt_s)
    with pytest.raises(ValueError):
        r02.hemisphere_yields_payload(
            {s: [1.0, 2.0] for s in r02.SPECIES_DISPLAY}, 4, sqrt_s
        )


# ---------------------------------------------------------------------------
# Data-side matching and model-only species (in-process, no generation)
# ---------------------------------------------------------------------------


@requires_sps_tables
def test_pa_data_cell_match_key_convention(cov_tables):
    """The (xF_mid, pT_mid) match key rounded to 1e-9; two data species."""
    pa = r02.load_pa_data("minimum_bias")
    # Only the two published pion tables exist; K+, K-, p, pbar are
    # model-only species (absent from the mapping).
    assert set(pa) == {"pi_minus", "pi_plus"}
    for species, entry in pa.items():
        assert set(entry) == {
            "file",
            "sha256",
            "cells",
            "n_rows",
            "n_unique_cells",
            "n_pt_zero_cells",
        }
        assert entry["sha256"] == r02.sha256_file(
            r02.pa_data_dir() / entry["file"]
        )
        assert 0 < entry["n_unique_cells"] <= entry["n_rows"]
        assert entry["n_pt_zero_cells"] >= 0

    table = next(t for t in cov_tables if t.species == "pi_minus")
    xf_mid = float(table.xf_centers[table.bin_index_xf[0]])
    pt_mid = float(table.pt_centers[table.bin_index_pt[0]])
    synthetic = {"cells": {(round(xf_mid, 9), round(pt_mid, 9)): [0.5, 0.1, 2]}}
    assert r02._match_pa_cell(synthetic, xf_mid, pt_mid) == [0.5, 0.1, 2]
    # A query offset far below the 1e-9 rounding unit still rounds onto
    # the same key.
    assert r02._match_pa_cell(synthetic, xf_mid + 1e-11, pt_mid) == [0.5, 0.1, 2]
    # Off-grid queries do not match.
    assert r02._match_pa_cell(synthetic, xf_mid + 1e-4, pt_mid) is None
    assert r02._match_pa_cell(synthetic, xf_mid, pt_mid + 1e-4) is None
    assert r02._match_pa_cell({}, xf_mid, pt_mid) is None
    assert r02._match_pa_cell(None, xf_mid, pt_mid) is None
    # The real minimum-bias table covers some, but not all, of the
    # model bins.
    n_matched = _matched_bin_count(table, pa["pi_minus"])
    assert 0 < n_matched < int(table.data.size)


@requires_sps_tables
def test_residuals_matched_null_and_chi2_convention(cov_tables):
    """Matched bins carry d and an infinite sigma; unmatched carry nulls."""
    tables = cov_tables
    pa = r02.load_pa_data("minimum_bias")
    # A zero-signal hand batch: no model value anywhere, so every
    # matched bin carries an infinite combined sigma (pull 0.0).
    values = {t.species: np.zeros(int(t.data.size)) for t in tables}
    counts = {t.species: np.zeros(int(t.data.size), dtype=int) for t in tables}
    errors = {
        t.species: r01.mc_error_array(values[t.species], counts[t.species])
        for t in tables
    }
    residuals = r02.residuals_payload(tables, values, errors, pa)
    assert len(residuals) == 1017

    by_species = {t.species: t for t in tables}
    for species in SPECIES_KEYS:
        table = by_species[species]
        entries = [e for e in residuals if e["species"] == species]
        assert len(entries) == int(table.data.size)
        n_matched = _matched_bin_count(table, pa.get(species))
        matched = [e for e in entries if e["d"] is not None]
        assert len(matched) == n_matched
        if species in pa:
            for e in matched:
                assert math.isfinite(e["d"]) and e["d"] > 0.0
                assert e["m"] == 0.0
                assert math.isinf(e["sigma"])  # empty MC bin: infinite sigma
                assert e["delta_over_sigma"] == 0.0
            for e in entries:
                if e["d"] is None:
                    assert e["sigma"] is None
                    assert e["delta_over_sigma"] is None
        else:
            # Model-only species: every bin is null (no published table).
            assert n_matched == 0
            assert all(e["d"] is None and e["delta_over_sigma"] is None for e in entries)

    chi2 = r02.chi2_payload(residuals)
    assert chi2["dof_convention"] == r02.CHI2_DOF_CONVENTION
    for species in SPECIES_KEYS:
        per = chi2["per_species"][species]
        if species in pa:
            n_matched = _matched_bin_count(by_species[species], pa[species])
            assert per["model_only"] is False
            assert per["n_bins"] == n_matched
            assert per["dof"] == n_matched - 1  # data-matched bins - 1
            assert per["n_empty_bins"] == n_matched  # all empty in this batch
            assert per["chi2"] == pytest.approx(0.0, abs=0.0)
        else:
            assert per["model_only"] is True
            assert per["chi2"] is None
            assert per["dof"] is None
            assert per["n_bins"] == 0
            assert "model-only" in per["reason"]
    total_matched = sum(
        _matched_bin_count(by_species[s], pa[s]) for s in ("pi_minus", "pi_plus")
    )
    assert chi2["total_data_matched_bins"] == total_matched
    assert chi2["total_chi2"] == pytest.approx(0.0, abs=0.0)


@requires_sps_tables
def test_empty_bins_infinite_mc_encoding_matches_r01(cov_tables, tmp_path):
    """N_bin == 0 bins: the R01 sentinel convention, cross-driver pin."""
    tables = cov_tables
    sqrt_s = _harness_sqrt_s()
    counts = {t.species: np.zeros(int(t.data.size), dtype=int) for t in tables}
    populated = _interior_target(next(t for t in tables if t.species == "k_plus"))
    counts["k_plus"][populated[0]] = 50
    sums = {t.species: np.zeros(int(t.data.size)) for t in tables}

    entries, values, errors = r01.bin_predictions_payload(
        tables, sums, counts, 1, sqrt_s
    )
    assert len(entries) == 1017
    # Exactly one bin carries a finite stat error: the populated one.
    finite = [
        (e["species"], e["bin_index"])
        for e in entries
        if math.isfinite(e["stat_error"])
    ]
    assert finite == [("k_plus", populated[0])]
    assert entries[0]["value"] == 0.0  # zero sums: m = 0 everywhere
    for e in entries:
        if (e["species"], e["bin_index"]) != finite[0]:
            assert math.isinf(e["stat_error"])
    # The committed JSON encoding of the infinite MC error is the string
    # sentinel "Infinity" (the R01 convention, RFC 8259 safe): the R02
    # writer's artifact strict-parses with no bare Infinity token, and
    # float() restores the infinite stat error on read.
    path = tmp_path / "bin_predictions.json"
    r02._write_json(path, entries[:2])
    text = path.read_text(encoding="utf-8")

    def _reject_constant(constant):
        raise AssertionError(f"bare JSON constant {constant!r} is not RFC 8259")

    loaded = json.loads(text, parse_constant=_reject_constant)
    assert loaded[0]["stat_error"] == "Infinity"
    assert math.isinf(float(loaded[0]["stat_error"]))

    residuals = r02.residuals_payload(tables, values, errors, {})
    failure = r02.build_failure_map(residuals, counts)
    expected_empty = {
        (t.species, i)
        for t in tables
        for i in range(int(t.data.size))
        if counts[t.species][i] == 0
    }
    assert {(e["species"], e["bin_index"]) for e in failure["empty_bins"]} == (
        expected_empty
    )
    assert failure["n_empty_bins"] == 1017 - 1
    assert failure["n_flagged"] == 0
    assert failure["model_only"] is True
    assert failure["model_only_species"] == list(r02.SPECIES_DISPLAY)

    # Cross-driver pin: the R02 combined covariance's V_mc agrees entry
    # for entry with the R01 driver's over the same arrays.
    cov_r02 = r02.covariance_payload(tables, errors, r01.read_sigma_eta(), {})
    cov_r01 = r01.covariance_payload(tables, errors, r01.read_sigma_eta())
    assert len(cov_r02["V_mc"]) == len(cov_r01["V_mc"]) == 1017
    for a, b in zip(cov_r02["V_mc"], cov_r01["V_mc"]):
        assert a["species"] == b["species"]
        assert a["bin_index"] == b["bin_index"]
        if math.isinf(b["variance"]):
            assert math.isinf(a["variance"])
        else:
            assert a["variance"] == pytest.approx(b["variance"], rel=0.0, abs=0.0)
    assert "Infinity" in json.dumps(cov_r02["V_mc"][:1])
    # With no data cells, V_data is 0.0 everywhere and the combined
    # variance carries the infinite MC variance unchanged.
    assert all(e["variance"] == 0.0 for e in cov_r02["V_data"])
    for a, b in zip(cov_r02["V_combined"], cov_r02["V_mc"]):
        if math.isinf(b["variance"]):
            assert math.isinf(a["variance"])
        else:
            assert a["variance"] == pytest.approx(b["variance"], rel=0.0, abs=0.0)


# ---------------------------------------------------------------------------
# Centrality mapping (in-process, no generation)
# ---------------------------------------------------------------------------


def test_centrality_mapping_constants_pinned():
    """The embedded M13 mapping constants are pinned verbatim."""
    assert r02.WOUND_BANDS == {
        "minimum_bias": None,
        "set_1": (2, 4),
        "set_2": (4, 4),
        "set_3": (4, 7),
        "set_4": (6, 6),
        "set_5": (7, 7),
    }
    assert r02.E_BE_K_BAND_MEAN == {
        "minimum_bias": None,
        "set_1": 2.9370128803339877,
        "set_2": 4.0,
        "set_3": 5.247809226460652,
        "set_4": 6.0,
        "set_5": 7.0,
    }
    assert r02.NU_MEASURED == {
        "minimum_bias": None,
        "set_1": 2.9,
        "set_2": 4.1,
        "set_3": 5.2,
        "set_4": 6.0,
        "set_5": 6.9,
    }
    # The pre-registration and the run identity carry the same mapping.
    assert r02.default_pre_registration()["e_be_per_class"] == {
        name: r02.E_BE_K_BAND_MEAN[name] for name in r02.CENTRALITY_CLASSES
    }


@requires_m13_prereg
def test_centrality_mapping_cross_checks_m13_preregistration():
    """The driver's runtime cross-check verifies the M13 file; slice fractions."""
    result = r02.verify_centrality_constants()
    assert result["cross_checked"] is True
    assert result["sha256"] == r02.sha256_file(M13_PREREG)

    sets = _load(M13_PREREG)["mapping_table"]["sets"]
    spec = NucleusSpec(**r02.GEOMETRY_PB208)
    glauber = WoundedNucleonSampler(spec, np.random.default_rng(SEED_BASE), dr_fm=0.05)
    for name in r02.CENTRALITY_CLASSES:
        if name == "minimum_bias":
            continue
        record = sets[name.split("_", 1)[1]]
        lo, hi = r02.WOUND_BANDS[name]
        assert [lo, hi] == list(record["band"])
        assert r02.E_BE_K_BAND_MEAN[name] == float(record["k_band_mean"])
        assert r02.NU_MEASURED[name] == float(record["nu_measured"])
        # The closed-form slice fraction of the pinned geometry matches
        # the pre-registered model fraction (the same sampler arithmetic).
        assert glauber.fraction_for_wound_range(lo, hi) == pytest.approx(
            float(record["slice_fraction_model"]), rel=1e-9
        )


# ---------------------------------------------------------------------------
# Gate-0 assembly (hand-built run directories, no generation)
# ---------------------------------------------------------------------------


def _gate0_fixture(
    out_dir: Path, completed_classes: int, completed_replicas: int
) -> dict:
    """Hand-built run directory at the committed seal ordering.

    Returns the per-class report payload written into the run manifest.
    """
    out_dir.mkdir(parents=True)
    (out_dir / r02.PREREGISTRATION_NAME).write_text(
        json.dumps(r02.default_pre_registration(), indent=2) + "\n",
        encoding="utf-8",
    )
    class_reports = {}
    for i, name in enumerate(r02.CENTRALITY_CLASSES):
        if i < completed_classes:
            requested = r02.T2_REPLICAS
            done = completed_replicas
            events = completed_replicas * r02.T2_EVENTS_PER_REPLICA
        else:
            requested, done, events = 0, 0, 0
        class_reports[name] = {
            "class_index": i,
            "band": (
                list(r02.WOUND_BANDS[name])
                if r02.WOUND_BANDS[name] is not None
                else None
            ),
            "replicas_requested": requested,
            "replicas_completed": done,
            "events_completed": events,
        }
    (
        classes_completed,
        replicas_per_class,
        events_completed,
        tier_status,
        tier_deferral_reason,
    ) = r02._tier_fields(
        class_reports, r02.T2_REPLICAS, r02.T2_EVENTS_PER_REPLICA
    )
    (out_dir / r02.RUN_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "classes": class_reports,
                "tier": {
                    "classes_completed": classes_completed,
                    "replicas_per_class_completed": replicas_per_class,
                    "events_completed": events_completed,
                },
                "tier_status": tier_status,
                "tier_deferral_reason": tier_deferral_reason,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    for name in r02.CENTRALITY_CLASSES:
        (out_dir / r02.failure_map_name(name)).write_text(
            json.dumps({"n_total_bins": 1017, "n_flagged": 0, "n_empty_bins": 0})
            + "\n",
            encoding="utf-8",
        )
    (out_dir / r02.PROVENANCE_NAME).write_text("{}\n", encoding="utf-8")
    (out_dir / r02.CLEAN_ENV_NAME).write_text(
        json.dumps(
            {
                r02.CLEAN_ENV_PYTEST_KEY: "pass",
                r02.CLEAN_ENV_SMOKE_KEY: "pass",
            }
        ),
        encoding="utf-8",
    )
    # Sealing happens after every payload file: the sums inventory is
    # the last write before the acceptance record.
    r02.write_sha256sums(out_dir)
    return class_reports


def _production_verification() -> dict:
    return r02.verify_config_and_m8(
        config_path=PRODUCTION_CONFIG,
        pre_registration=r02.default_pre_registration(),
    )


@pytest.mark.parametrize(
    "completed_classes, completed_replicas, status",
    [
        (6, 8, "complete"),
        (5, 8, "deferred"),
        (6, 7, "deferred"),
    ],
)
def test_gate0_tier_status_complete_and_deferred(
    tmp_path, completed_classes, completed_replicas, status
):
    """The strict tier predicate, with the two records in agreement."""
    out_dir = tmp_path / f"gate0_{completed_classes}x{completed_replicas}"
    _gate0_fixture(out_dir, completed_classes, completed_replicas)
    gate = r02.assemble_gate0(
        out_dir, _production_verification(), r02.default_pre_registration()
    )
    manifest = _load(out_dir / r02.RUN_MANIFEST_NAME)
    assert gate["gate"] == "gate0"
    assert gate["clause"] == r02.GATE_CLAUSE
    assert gate["config_hash_verified"] is True
    assert gate["m8_parameter_hash_verified"] is True
    assert gate["archive_sealed"] is True
    assert gate["immutable"] is True
    assert gate["target"] == "Pb208"
    assert gate["unavailable_targets"] == ["C", "W"]
    assert gate["n_total_bins_per_class"] == 1017
    assert gate["n_classes"] == 6
    assert gate["n_flagged_cells"] == 0
    assert gate["empty_bins"] == 0
    assert gate["clean_environment_commands"] == {
        r02.CLEAN_ENV_PYTEST_KEY: "pass",
        r02.CLEAN_ENV_SMOKE_KEY: "pass",
    }
    assert gate["archive_seal_sha256"] == r02.sha256_file(
        out_dir / r02.SHA256SUMS_NAME
    )
    # The single strict predicate: the gate record and the run manifest
    # classify through it and cannot disagree.
    expected = r02.tier_status_decision(
        completed_classes,
        completed_replicas if completed_classes == 6 else 0,
        len(r02.CENTRALITY_CLASSES),
        r02.T2_REPLICAS,
    )
    assert expected == status
    assert gate["tier_status"] == status
    assert manifest["tier_status"] == status
    assert gate["tier"]["classes_completed"] == manifest["tier"]["classes_completed"]
    assert (
        gate["tier"]["replicas_per_class_completed"]
        == manifest["tier"]["replicas_per_class_completed"]
    )
    assert gate["tier"]["events_completed"] == manifest["tier"]["events_completed"]
    if status == "complete":
        assert gate["tier_deferral_reason"] is None
        assert manifest["tier_deferral_reason"] is None
        assert gate["tier"]["classes_completed"] == 6
        assert gate["tier"]["replicas_per_class_completed"] == 8
        assert gate["tier"]["events_completed"] == 6 * 8 * 100000
    else:
        assert gate["tier_deferral_reason"] is not None
        assert manifest["tier_deferral_reason"] is not None
        assert "of 6 classes" in gate["tier_deferral_reason"]
        assert "follow-up obligation" in gate["tier_deferral_reason"]
        if (completed_classes, completed_replicas) == (5, 8):
            assert "5 of 6 classes" in gate["tier_deferral_reason"]
            assert "4000000 of 4800000" in gate["tier_deferral_reason"]
        else:  # (6, 7): no class completed, 7 of 8 replicas per class
            assert "7 of 8" in gate["tier_deferral_reason"]
            assert "4200000 of 4800000" in gate["tier_deferral_reason"]
    # A deferral does not block Gate 0: the verdict is pass when the
    # hashes verify, the archive seals, and the commands recorded pass.
    assert gate["verdict"] == "pass"
    # The acceptance record is written under the card's file name and
    # matches the returned payload.
    assert _load(out_dir / r02.GATE_RECORD_NAME) == gate


def test_gate0_immutability_flips_on_payload_mutation(tmp_path):
    """A post-seal payload mutation flips immutable and the verdict."""
    out_dir = tmp_path / "gate0_mutation"
    _gate0_fixture(out_dir, 6, 8)
    verification = _production_verification()
    gate = r02.assemble_gate0(
        out_dir, verification, r02.default_pre_registration()
    )
    assert gate["immutable"] is True
    assert gate["verdict"] == "pass"

    # Mutate a payload file after sealing (kept as valid JSON, since the
    # acceptance record re-parses it): the record re-verifies every
    # SHA256SUMS.txt entry and must see the digest drift.
    payload = out_dir / r02.failure_map_name("minimum_bias")
    payload.write_text(
        json.dumps({"n_total_bins": 1017, "n_flagged": 1, "n_empty_bins": 0}) + "\n",
        encoding="utf-8",
    )
    mutated = r02.assemble_gate0(
        out_dir, verification, r02.default_pre_registration()
    )
    assert mutated["immutable"] is False
    assert mutated["verdict"] == "fail"
    assert mutated["archive_sealed"] is True  # the files still exist


# ---------------------------------------------------------------------------
# Seal: the --seal path on a small run directory (10 events)
# ---------------------------------------------------------------------------


@requires_sps_tables
def test_cli_seal_small_run(tmp_path):
    """--seal on a 10-event smoke run writes the immutable-archive triple."""
    out_dir = tmp_path / "seal_run"
    assert not out_dir.exists()
    proc = _run_r02(
        [
            "--config",
            str(SMOKE_CONFIG),
            "--centrality-classes",
            "minimum_bias",
            "--replicas",
            "1",
            "--events-per-replica",
            str(SEAL_N_EVENTS),
            "--seed-base",
            str(SMOKE_SEED_BASE),
            "--seal",
        ],
        out_dir,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # --out is created fresh by the driver.
    assert out_dir.is_dir()
    for name in (
        r02.PREREGISTRATION_NAME,
        r02.replica_record_name("minimum_bias", SMOKE_MB_SEED),
        r02.RUN_MANIFEST_NAME,
        r02.bin_predictions_name("minimum_bias"),
        r02.hemisphere_yields_name("minimum_bias"),
        r02.covariance_name("minimum_bias"),
        r02.residuals_name("minimum_bias"),
        r02.failure_map_name("minimum_bias"),
        r02.PROVENANCE_NAME,
        r02.SHA256SUMS_NAME,
        r02.SEAL_NAME,
    ):
        assert (out_dir / name).is_file(), name
    # The smoke run is not the canonical production run: no gate record.
    assert not (out_dir / r02.GATE_RECORD_NAME).is_file()
    assert "no gate record (not the canonical production run)" in proc.stdout

    # The pre-registration equals the card's payload exactly.
    assert _load(out_dir / r02.PREREGISTRATION_NAME) == r02.default_pre_registration()

    # The binned artifacts carry the 1017-bin set.
    predictions = _load(out_dir / r02.bin_predictions_name("minimum_bias"))
    assert len(predictions) == 1017
    assert {
        s: sum(1 for e in predictions if e["species"] == s) for s in SPECIES_KEYS
    } == SPECIES_BIN_SCHEMA

    # SHA256SUMS.txt: sha256sum -c format over every regular file except
    # the files that cannot carry their own digests.
    sums_text = (out_dir / r02.SHA256SUMS_NAME).read_text(encoding="utf-8")
    lines = [line for line in sums_text.splitlines() if line.strip()]
    pattern = re.compile(r"^[0-9a-f]{64}  \S+$")
    assert all(pattern.match(line) for line in lines)
    digests = {
        name: digest for digest, name in (line.split("  ", 1) for line in lines)
    }
    every_file = {
        p.relative_to(out_dir).as_posix()
        for p in out_dir.rglob("*")
        if p.is_file() and not p.is_symlink()
    }
    assert set(digests) == every_file - {
        r02.SHA256SUMS_NAME,
        r02.SEAL_NAME,
        r02.PROVENANCE_NAME,
        r02.GATE_RECORD_NAME,
    }
    for line in lines:
        digest, name = line.split("  ", 1)
        payload = (out_dir / name).read_bytes()
        assert digest == hashlib.sha256(payload).hexdigest()

    # The seal manifest records every file except itself, digests match.
    seal = _load(out_dir / r02.SEAL_NAME)
    sealed = {entry["path"] for entry in seal["files"]}
    assert sealed == every_file - {r02.SEAL_NAME}
    for entry in seal["files"]:
        path = out_dir / entry["path"]
        assert entry["sha256"] == r02.sha256_file(path)
        assert entry["bytes"] == path.stat().st_size

    # The provenance record carries the run identity and the code
    # fingerprint (RESONANCE_DECAY_LIB_VERSION salt).
    provenance = _load(out_dir / r02.PROVENANCE_NAME)
    assert provenance["producer"] == "staged_r02_reference.py"
    assert provenance["config_sha256"] == r02.SMOKE_CONFIG_SHA256
    assert provenance["m8_parameter_hash"] == r02.M8_PARAMETER_HASH
    assert provenance["sigma_inel_mb"] == 31.78
    # seed_base is recorded from the pre-registration (the card's T2
    # base), while replica_seeds_per_class carries the actual run seeds:
    # a smoke seed base is reported alongside, not in place of, the
    # pre-registered one.
    assert provenance["seed_base"] == SEED_BASE
    assert provenance["seed_offset_base"] == r02.SEED_OFFSET_BASE
    assert provenance["seed_offset_per_class"] == r02.SEED_OFFSET_PER_CLASS
    assert provenance["replica_seeds_per_class"] == {"minimum_bias": [SMOKE_MB_SEED]}
    assert provenance["events_per_replica"] == SEAL_N_EVENTS
    assert provenance["events_completed"] == SEAL_N_EVENTS
    assert provenance["tier_status"] == "deferred"
    assert provenance["e_be_per_class"]["minimum_bias"] is None
    assert provenance["hemisphere_rule"] == dict(r02.HEMISPHERE_RULE)
    fingerprint = provenance["code_fingerprint"]
    assert fingerprint["resonance_decay_lib_version"] == RESONANCE_DECAY_LIB_VERSION
    assert re.fullmatch(r"[0-9a-f]{64}", fingerprint["staged_generation_code_sha256"])

    # The per-replica record: the class, the run seed, the stamped config
    # hash and table digest, and the wound bookkeeping.
    replica = _load(out_dir / r02.replica_record_name("minimum_bias", SMOKE_MB_SEED))
    manifest = _load(out_dir / r02.RUN_MANIFEST_NAME)
    assert replica["class"] == "minimum_bias"
    assert replica["seed"] == SMOKE_MB_SEED
    assert replica["band"] is None
    assert replica["n_events"] == SEAL_N_EVENTS
    assert replica["config_sha256"] == r02.SMOKE_CONFIG_SHA256
    assert replica["tables_sha256"] == manifest["tables_sha256"]
    assert re.fullmatch(r"[0-9a-f]{64}", replica["event_hash_set_sha256"])
    assert 0.0 < replica["mean_target_wounds"] < 208.0

    # The run manifest: the smoke config pin, the smoke run's seed base,
    # and the declared T2 seed lists (independent of the run's seed base).
    assert manifest["config_sha256"] == r02.SMOKE_CONFIG_SHA256
    assert manifest["m8_parameter_hash"] == r02.M8_PARAMETER_HASH
    assert manifest["seed_base"] == SMOKE_SEED_BASE
    assert manifest["seed_offset_base"] == r02.SEED_OFFSET_BASE
    assert manifest["seed_offset_per_class"] == r02.SEED_OFFSET_PER_CLASS
    assert manifest["replica_seeds_per_class"] == {"minimum_bias": [SMOKE_MB_SEED]}
    assert manifest["declared_t2"]["seeds_per_class"] == {
        name: r02.class_replica_seeds(SEED_BASE, i, r02.T2_REPLICAS)
        for i, name in enumerate(r02.CENTRALITY_CLASSES)
    }
    assert manifest["tier_status"] == "deferred"
    assert manifest["tier_deferral_reason"] is not None
    class_report = manifest["classes"]["minimum_bias"]
    assert class_report["band"] is None
    assert class_report["e_be_k_band_mean"] is None
    assert class_report["nu_measured"] is None
    assert class_report["replicas_requested"] == 1
    assert class_report["replicas_completed"] == 1
    assert class_report["events_completed"] == SEAL_N_EVENTS
    assert class_report["model_only"] is False
    assert class_report["model_only_species"] == list(MODEL_ONLY_SPECIES)
    assert class_report["seeds"] == [SMOKE_MB_SEED]
    assert len(class_report["replicas"]) == 1
    assert class_report["replicas"][0]["status"] == "completed"
    assert class_report["replicas"][0]["seed"] == SMOKE_MB_SEED

    # The per-class chi2 recorded in the run manifest equals the
    # per-species sum of squared pulls recomputed from the sealed
    # residuals file (the committed DOF convention, bins - 1).
    residuals = _load(out_dir / r02.residuals_name("minimum_bias"))
    assert len(residuals) == 1017
    for species in SPECIES_KEYS:
        entries = [e for e in residuals if e["species"] == species]
        matched = [e for e in entries if e["d"] is not None]
        per = class_report["chi2_per_species"][species]
        assert class_report["n_data_matched_bins"][species] == len(matched)
        if species in ("pi_minus", "pi_plus"):
            assert per["model_only"] is False
            assert per["n_bins"] == len(matched)
            assert per["dof"] == len(matched) - 1
            assert per["chi2"] == pytest.approx(
                sum(float(e["delta_over_sigma"]) ** 2 for e in matched), rel=1e-9
            )
        else:
            assert per["model_only"] is True
            assert per["chi2"] is None
            assert len(matched) == 0

    # The failure map's empty_bins agree with the bin predictions'
    # infinite stat errors (recorded, never dropped).
    failure = _load(out_dir / r02.failure_map_name("minimum_bias"))
    assert failure["n_total_bins"] == 1017
    assert failure["model_only"] is False
    assert failure["data_species"] == ["pi_minus", "pi_plus"]
    empty_from_predictions = {
        (e["species"], e["bin_index"])
        for e in predictions
        if not math.isfinite(float(e["stat_error"]))
    }
    empty_from_map = {
        (e["species"], e["bin_index"]) for e in failure["empty_bins"]
    }
    assert empty_from_map == empty_from_predictions
    assert failure["n_empty_bins"] == len(empty_from_map)

    # The hemisphere yields: six species in the display order, every
    # region finite and non-negative; the pion yields are positive where
    # every p+A sample certainly produces pions in the nucleon-nucleon CM
    # frame (the central region around x_F = 0 and the target side x_F < 0,
    # which carries the K target sources). The projectile side x_F > 0.2
    # holds a few pions per hundred events and may be empty at 10 events
    # (card TASK-20261005-pa-frame-nn-cm-xf: the lab-frame ratio
    # 2 p_z^lab / sqrt(s_NN) had filled it spuriously).
    hemisphere = _load(out_dir / r02.hemisphere_yields_name("minimum_bias"))
    assert [e["species"] for e in hemisphere] == list(SPECIES_KEYS)
    for entry in hemisphere:
        for key in ("projectile_side", "central", "target_side"):
            assert math.isfinite(entry[key])
            assert entry[key] >= 0.0
    for species in ("pi_minus", "pi_plus"):
        entry = next(e for e in hemisphere if e["species"] == species)
        assert entry["central"] > 0.0
        assert entry["target_side"] > 0.0


# ---------------------------------------------------------------------------
# Sealed-run cross-checks (read-only, the artifact store)
# ---------------------------------------------------------------------------


@requires_sps_tables
@requires_sealed_r02
def test_sealed_r02_run_pins_seed_rule_and_tier():
    """The sealed 4.8M-event archive pins the seed rule, tier, and gate0."""
    manifest = _load(SEALED_R02_DIR / r02.RUN_MANIFEST_NAME)
    # The per-class seed lists equal the committed rule, and every
    # replica record exists under its seed name.
    assert manifest["declared_t2"]["seeds_per_class"] == {
        name: r02.class_replica_seeds(SEED_BASE, i, r02.T2_REPLICAS)
        for i, name in enumerate(r02.CENTRALITY_CLASSES)
    }
    for i, name in enumerate(r02.CENTRALITY_CLASSES):
        seeds = manifest["classes"][name]["seeds"]
        assert seeds == r02.class_replica_seeds(SEED_BASE, i, r02.T2_REPLICAS)
        for seed in seeds:
            assert (
                SEALED_R02_DIR / r02.replica_record_name(name, seed)
            ).is_file()
        report = manifest["classes"][name]
        assert report["band"] == (
            list(r02.WOUND_BANDS[name]) if r02.WOUND_BANDS[name] is not None else None
        )
        assert report["e_be_k_band_mean"] == r02.E_BE_K_BAND_MEAN[name]
        assert report["nu_measured"] == r02.NU_MEASURED[name]
        assert report["replicas_completed"] == r02.T2_REPLICAS
        assert report["events_completed"] == (
            r02.T2_REPLICAS * r02.T2_EVENTS_PER_REPLICA
        )
    # The declared T2 tier completed.
    assert manifest["tier_status"] == "complete"
    assert manifest["tier"]["events_completed"] == (
        6 * r02.T2_REPLICAS * r02.T2_EVENTS_PER_REPLICA
    )

    # The minimum-bias matched-bin count is a pure function of the
    # committed data tables and the model binning: 118 per pion species
    # (dof 117), as the sealed run records.
    tables = r02.load_binned_tables(
        json.loads(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    )
    pa = r02.load_pa_data("minimum_bias")
    by_species = {t.species: t for t in tables}
    for species in ("pi_minus", "pi_plus"):
        assert _matched_bin_count(by_species[species], pa[species]) == 118
    mb = manifest["classes"]["minimum_bias"]
    assert mb["n_data_matched_bins"] == {
        "pi_minus": 118,
        "pi_plus": 118,
        "k_plus": 0,
        "k_minus": 0,
        "proton": 0,
        "antiproton": 0,
    }
    pi_chi2 = mb["chi2_per_species"]["pi_minus"]
    assert pi_chi2["n_bins"] == 118
    assert pi_chi2["dof"] == 117
    assert pi_chi2["chi2"] == pytest.approx(1381489.136798142, rel=1e-9)

    # The gate0 acceptance record of the canonical production run.
    gate = _load(SEALED_R02_DIR / r02.GATE_RECORD_NAME)
    assert gate["gate"] == "gate0"
    assert gate["clause"] == r02.GATE_CLAUSE
    assert gate["verdict"] == "pass"
    assert gate["config_hash_verified"] is True
    assert gate["m8_parameter_hash_verified"] is True
    assert gate["archive_sealed"] is True
    assert gate["immutable"] is True
    assert gate["target"] == "Pb208"
    assert gate["unavailable_targets"] == ["C", "W"]
    assert gate["centrality_classes"] == list(r02.CENTRALITY_CLASSES)
    assert gate["tier_status"] == "complete"
    assert gate["tier_deferral_reason"] is None
    assert gate["tier"]["classes_completed"] == 6
    assert gate["tier"]["replicas_per_class_completed"] == 8
    assert gate["tier"]["events_completed"] == 4800000
    assert gate["n_total_bins_per_class"] == 1017
    assert gate["n_flagged_cells"] == 1151
    assert gate["empty_bins"] == 386
    assert gate["archive_seal_sha256"] == r02.sha256_file(
        SEALED_R02_DIR / r02.SHA256SUMS_NAME
    )
