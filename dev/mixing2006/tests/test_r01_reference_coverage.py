"""Coverage pins for the R01 legacy p+p reference driver.

Companion to ``tests/test_r01_reference_harness.py`` (the four card-named
hermetic checks of card ``TASK-20260911-r1-run-r01-legacy-pp-reference``);
this module pins the committed behavior around them.  All checks are
hermetic and small-N: the only event generation is a single 100-event
driver subprocess (the card's subprocess ceiling); the hand-constructed
binning batch is built from stub particles with no generation at all;
the 5,000,000-event T3 run is the R01 run itself, not a test.  The one
read-only cross-check against the sealed 5M-event run directory skips
with a recorded reason when the artifact store is absent.

CLI surface (subprocess, the card's smoke-config path):

- ``test_cli_unknown_arg_refused`` and
  ``test_cli_missing_required_arg_refused``: an unknown flag or a
  missing required argument is an argparse refusal (exit 2) before the
  ``--out`` directory is created.
- ``test_cli_nonpositive_replicas_refused_before_generation`` and
  ``test_cli_nonpositive_events_refused_before_generation``: 0 and
  negative ``--replicas`` / ``--events-per-replica`` are refused with
  the exit-2 refusal text before the ``--out`` directory is created or
  a single event is generated.
- ``test_cli_preregistration_mismatch_refused_before_generation``: a
  run directory whose ``pre_registration.json`` carries a mutated
  ``config_hash`` is refused (exit 2) before any generation; the
  pre-registration is written fresh by the driver and must equal the
  card's payload exactly.

Pre-registration enforcement (in-process, no generation):

- ``test_verify_config_hash_preregistration_mismatch``: the pristine
  production config verifies, and a mutated ``config_hash`` in the
  pre-registration refuses with exit code 2.
- ``test_verify_m8_parameter_hash_mismatch``: a mutated
  ``m8_parameter_hash`` in the pre-registration refuses with exit code
  2; the manifest's frozen block is re-verified against the driver's
  verbatim copy on the pristine path.
- ``test_replica_seed_rule_pinned``: ``replica_seeds`` applies
  ``seed_base + 40000 + k`` (the hand-built three-seed list for
  ``seed_base = 20260911`` is ``[20300911, 20300912, 20300913]``), and
  the default pre-registration carries the full 20-seed T3 archive.

Binning correctness (hand-constructed stub batch, no generation):

- ``test_hand_constructed_bin_counts_and_midpoints``: particles placed
  at tabulated (xF, pT) bin centers of two species land in exactly the
  intended table bins; the pooled binning reproduces the placed counts
  exactly, every other bin carries 0, the per-species bin-count schema
  (229/230/153/148/135/122, total 1017) holds, and the entry
  ``xF_mid``/``pT_mid`` values match the ``sps_core.edges_from_centers``
  midpoint convention (each tabulated center is the midpoint of its bin
  edges).
- ``test_empty_bins_recorded_and_infinite_mc_encoding``: a batch with
  ``N_bin == 0`` bins is never silently dropped: the bin stays in the
  residuals with an infinite combined sigma and pull 0.0, the failure
  map lists exactly the zero-count bins under ``empty_bins`` (species
  plus bin index), and the committed JSON encoding of the infinite MC
  variance is the string sentinel ``"Infinity"`` (RFC 8259 safe; the
  written artifact strict-parses and ``float()`` restores ``inf``).

Covariance assembly (hand-constructed MC errors over the loaded tables):

- ``test_covariance_assembly_and_card7_nuisance``: ``V_combined =
  V_data + V_mc`` element-wise (diagonal) over all 1017 bins, and the
  normalization-nuisance block carries ``eta = 0.0``, the card-7 width
  ``sigma_eta = 0.01`` (the shared width of all six table nuisances in
  ``config/v2_singlet/likelihood_nuisances.json``, read by
  ``read_sigma_eta``), and scope ``all_six_tables``.

Chi2 (hand payload plus the sealed 5M-event run, read-only):

- ``test_chi2_dof_convention_hand_payload``: the per-species chi2 is
  ``sum(delta_over_sigma**2)`` over the species' bins with the
  committed DOF convention ``bins - 1`` (R01 fits no parameter),
  including ``chi2_dof`` and the total ``dof = total_bins - 1``.
- ``test_chi2_matches_sealed_r01_run``: the chi2 recorded in the sealed
  T3 run equals the per-species sum of squared pulls recomputed from
  the sealed ``residuals.json`` (pi- 1329031.99, pi+ 1450782.79,
  K+ 325707.33, K- 15710.76, p 154567.09, pbar 9004.71), pinning the
  ``bins - 1`` DOF formula against the committed 5M-event archive.

Failure-map boundary (hand-constructed pulls, no tables):

- ``test_failure_map_threshold_boundary_ranking_and_tie_break``: a bin
  at exactly ``|pull| = 3.0`` is not flagged (strict ``>``), one at
  3.0001 is; ranking is ``|pull|`` descending with rank 1 at the
  largest and contiguous ranks; the committed tie-break is the config
  species display order (pi-, pi+, K+, K-, p, pbar) and then the lower
  bin index, with the signed pull preserved in the ``pull`` field.

Gate-0 assembly (hand-built run directories, no generation):

- ``test_gate0_tier_status_complete_and_deferred``: 20/20 completed
  replicas assemble ``tier_status = "complete"`` with a null deferral
  reason; 1/20 assembles ``"deferred"`` with the recorded reason; the
  deferral does not block the verdict (both assemble ``pass`` when the
  hashes verify, the archive seals, and the clean-environment commands
  recorded ``pass``).
- ``test_gate0_seal_reverified_after_payload_mutation``: the acceptance
  record re-verifies every ``SHA256SUMS.txt`` entry at assembly time; a
  payload mutation after sealing flips ``immutable`` to false and the
  verdict to ``fail``.

Seal (subprocess, 100 events, the card's subprocess ceiling):

- ``test_cli_seal_small_run``: the ``--seal`` path on a fresh 100-event
  smoke run writes the binned artifacts plus the immutable-archive
  triple; ``SHA256SUMS.txt`` is in ``sha256sum -c`` format over every
  regular payload file except the self-referential files and re-hashes
  cleanly, ``seal_manifest.json`` excludes itself, and
  ``provenance.json`` carries the config hash, the M8 parameter hash,
  the seed list, and the ``RESONANCE_DECAY_LIB_VERSION`` salted code
  fingerprint; no ``gate0_r01.json`` is written for the non-production
  smoke config.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from helpers.benchmarking import sps_core
from helpers.benchmarking import staged_r01_reference as r01
from helpers.benchmarking.benchmark_sps_species import RESONANCE_DECAY_LIB_VERSION
from na49_mixing.reaction import Reaction

NESTED_ROOT = Path(r01.NESTED_ROOT)
MONO_ROOT = Path(r01.MONO_ROOT)
DRIVER_PATH = NESTED_ROOT / "helpers" / "benchmarking" / "staged_r01_reference.py"
SMOKE_CONFIG = NESTED_ROOT / "config" / "staged_excitation_smoke.json"
PRODUCTION_CONFIG = NESTED_ROOT / "config" / "staged_excitation.json"

# Card-seeded T0 replica seeds: 20260911 + 40000 + k, k = 0..2.
CARD_SEED_BASE = 20260911
CARD_SEEDS_3 = [20300911, 20300912, 20300913]
# Subprocess seal run: the card's seed-base convention for smoke runs.
SMOKE_SEED_BASE = 20311000
SMOKE_SEED = SMOKE_SEED_BASE + 40000  # 20351000
SEAL_N_EVENTS = 100  # the card's subprocess generation ceiling
SUBPROCESS_TIMEOUT = 600

# The committed DOF convention string (chi2.json, the sealed run).
CHI2_DOF_CONVENTION = "bins - 1 (card R01; R01 fits no parameter)"

# The sealed T3 run directory (gitignored artifact store, read-only).
SEALED_RUN_DIR = MONO_ROOT / "dev" / "00runs" / "r01_legacy_pp_reference_2026-09-11"
_SEALED_PRESENT = (SEALED_RUN_DIR / "chi2.json").is_file() and (
    SEALED_RUN_DIR / "residuals.json"
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

SPECIES_KEYS = ("pi_minus", "pi_plus", "k_plus", "k_minus", "proton", "antiproton")
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


def _run_driver(extra_args, out_dir: Path):
    """One driver subprocess invocation from the nested tree root."""
    return subprocess.run(
        [sys.executable, str(DRIVER_PATH), "--config", str(SMOKE_CONFIG)]
        + [str(a) for a in extra_args]
        + ["--out", str(out_dir)],
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
        cwd=str(NESTED_ROOT),
    )


def _harness_sqrt_s() -> float:
    return math.sqrt(Reaction(158.0, 2212, 2212).s())


@pytest.fixture(scope="module")
def cov_tables() -> list:
    """The windowed six-species tables (the 1017-bin grid)."""
    raw_config = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    return r01.load_binned_tables(raw_config)


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_unknown_arg_refused(tmp_path):
    """An unknown flag is an argparse refusal before --out is created."""
    out_dir = tmp_path / "unknown_arg"
    assert not out_dir.exists()
    proc = _run_driver(
        [
            "--replicas",
            "1",
            "--events-per-replica",
            "1",
            "--seed-base",
            str(SMOKE_SEED_BASE),
            "--bogus-flag",
        ],
        out_dir,
    )
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
    proc = _run_driver(
        ["--replicas", str(replicas), "--events-per-replica", "1",
         "--seed-base", str(SMOKE_SEED_BASE)],
        out_dir,
    )
    assert proc.returncode == 2
    assert "must be a positive integer" in proc.stderr
    # The refusal fires before the out directory is created: nothing,
    # not even a pre-registration, exists.
    assert not out_dir.exists()


@pytest.mark.parametrize("events", [0, -1])
def test_cli_nonpositive_events_refused_before_generation(tmp_path, events):
    """0 and negative --events-per-replica are refused before generation."""
    out_dir = tmp_path / f"events_{events}"
    proc = _run_driver(
        ["--replicas", "1", "--events-per-replica", str(events),
         "--seed-base", str(SMOKE_SEED_BASE)],
        out_dir,
    )
    assert proc.returncode == 2
    assert "must be a positive integer" in proc.stderr
    assert not out_dir.exists()


def test_cli_preregistration_mismatch_refused_before_generation(tmp_path):
    """A run dir whose pre-registration config_hash drifted is refused."""
    out_dir = tmp_path / "prereg_drift"
    out_dir.mkdir()
    drifted = r01.default_pre_registration()
    drifted["config_hash"] = "0" * 64
    (out_dir / r01.PREREGISTRATION_NAME).write_text(
        json.dumps(drifted, indent=2) + "\n", encoding="utf-8"
    )
    proc = _run_driver(
        ["--replicas", "1", "--events-per-replica", "1",
         "--seed-base", str(SMOKE_SEED_BASE)],
        out_dir,
    )
    assert proc.returncode == 2
    assert "pre-registration mismatch" in proc.stderr
    # The refusal fires before any generation: no replica record and no
    # binned artifact can exist.
    assert not any(out_dir.glob("replica_*.json"))
    assert not (out_dir / r01.BIN_PREDICTIONS_NAME).is_file()
    assert not (out_dir / r01.FAILURE_MAP_NAME).is_file()


# ---------------------------------------------------------------------------
# Pre-registration enforcement (in-process)
# ---------------------------------------------------------------------------


def test_verify_config_hash_preregistration_mismatch():
    """A mutated pre-registration config_hash refuses with exit code 2."""
    pristine = r01.verify_config_and_m8(
        PRODUCTION_CONFIG, r01.default_pre_registration()
    )
    assert pristine["production_config"] is True
    assert pristine["config_sha256"] == r01.PRODUCTION_CONFIG_SHA256
    assert pristine["m8_parameter_hash"] == r01.M8_PARAMETER_HASH

    drifted = r01.default_pre_registration()
    drifted["config_hash"] = "0" * 64
    with pytest.raises(SystemExit) as excinfo:
        r01.verify_config_and_m8(PRODUCTION_CONFIG, drifted)
    assert excinfo.value.code == 2


def test_verify_m8_parameter_hash_mismatch():
    """A mutated pre-registration m8_parameter_hash refuses with exit 2."""
    drifted = r01.default_pre_registration()
    drifted["m8_parameter_hash"] = "f" * 64
    with pytest.raises(SystemExit) as excinfo:
        r01.verify_config_and_m8(PRODUCTION_CONFIG, drifted)
    assert excinfo.value.code == 2


def test_replica_seed_rule_pinned():
    """replica seeds are seed_base + 40000 + k; the card's T0 list is pinned."""
    assert r01.replica_seeds(CARD_SEED_BASE, 3) == CARD_SEEDS_3
    assert r01.replica_seeds(SMOKE_SEED_BASE, 1) == [SMOKE_SEED]
    assert r01.replica_seeds(CARD_SEED_BASE, 20) == [20300911 + k for k in range(20)]

    prereg = r01.default_pre_registration()
    assert prereg["seed_base"] == CARD_SEED_BASE
    assert prereg["seed_offset"] == 40000
    assert prereg["replica_seeds"] == [20300911 + k for k in range(20)]
    assert prereg["sigma_inel_mb"] == 31.78
    assert prereg["observable_set"]["species"] == {
        "pi-": 229,
        "pi+": 230,
        "K+": 153,
        "K-": 148,
        "p": 135,
        "pbar": 122,
    }
    assert prereg["observable_set"]["total_bins"] == 1017
    assert prereg["tier"] == {
        "declared": "T3",
        "declared_replicas": 20,
        "declared_events_per_replica": 250000,
    }


# ---------------------------------------------------------------------------
# Binning correctness (hand-constructed stub batch, no generation)
# ---------------------------------------------------------------------------


def _stub_event(pdg: int, pT: float, xF: float, sqrt_s: float):
    """A single-particle event stub carrying only the fields the driver reads."""
    pz = xF * sqrt_s / 2.0
    particle = types.SimpleNamespace(
        pdg_id=int(pdg),
        p4=types.SimpleNamespace(e=1.0, px=float(pT), py=0.0, pz=float(pz)),
    )
    return types.SimpleNamespace(final_particles=[particle])


def _interior_target(table) -> tuple[int, int, int, float, float]:
    """A table bin whose tabulated center sits strictly inside its edges.

    The driver bins at the ``sps_core.grid_indices`` (digitize) edges of
    ``edges_from_centers``; a particle at the tabulated center is placed
    deterministically in that bin.  The first such bin with
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


@requires_sps_tables
def test_hand_constructed_bin_counts_and_midpoints(cov_tables):
    """Placed particles bin exactly at their tabulated (xF, pT) centers."""
    tables = cov_tables
    assert {t.species: int(t.data.size) for t in tables} == SPECIES_BIN_SCHEMA
    assert sum(int(t.data.size) for t in tables) == 1017

    sqrt_s = _harness_sqrt_s()
    by_species = {t.species: t for t in tables}
    plan = {
        "pi_minus": (3, _interior_target(by_species["pi_minus"])),
        "k_plus": (2, _interior_target(by_species["k_plus"])),
    }
    events = []
    for species, (n, (i, ix, ip, x, p)) in plan.items():
        pdg = int(by_species[species].pdg_id)
        for _ in range(n):
            events.append(_stub_event(pdg, p, x, sqrt_s))
    assert len(events) == 5

    sums, counts = r01._bin_track_sums(events, tables, sqrt_s)
    # Per-species count arrays follow the loaded table order and schema.
    for table in tables:
        arr = np.asarray(counts[table.species])
        assert arr.size == int(table.data.size)
        expected = plan.get(table.species, (0, None))[0]
        assert int(arr.sum()) == expected
        if table.species in plan:
            i = plan[table.species][1][0]
            assert int(arr[i]) == expected
            assert int(np.sum(arr != 0)) == 1  # every other bin is empty
        else:
            assert bool(np.all(arr == 0))

    entries, values, errors = r01.bin_predictions_payload(
        tables, sums, counts, len(events), sqrt_s
    )
    assert len(entries) == 1017
    for species, (n, (i, ix, ip, x, p)) in plan.items():
        table = by_species[species]
        entry = [e for e in entries if e["species"] == species][i]
        # The entry carries the tabulated reference-grid center of the
        # bin exactly: the committed midpoint is the grid center as
        # recorded in the SPS table (not a recomputed bin-edge mean),
        # and it equals the center the stub particle was placed at.
        assert entry["bin_index"] == i
        assert entry["xF_mid"] == float(table.xf_centers[ix])
        assert entry["pT_mid"] == float(table.pt_centers[ip])
        assert entry["xF_mid"] == pytest.approx(x, abs=0.0)
        assert entry["pT_mid"] == pytest.approx(p, abs=0.0)
        # Placed weight sums: value = (n * (E/pT)) / (pi * sqrt(s) * N *
        # dxF * dpT) with the stub E = 1.0, strictly positive and finite.
        assert entry["value"] > 0.0 and math.isfinite(entry["stat_error"])
        assert values[species][i] == pytest.approx(entry["value"])
        assert math.isfinite(errors[species][i])
    # Every unpopulated bin has value 0 and the infinite MC error.
    for table in tables:
        populated = plan.get(table.species, (0, None))[1]
        for i in range(int(table.data.size)):
            if populated is not None and i == populated[0]:
                continue
            entry = [e for e in entries if e["species"] == table.species][i]
            assert entry["value"] == 0.0
            assert math.isinf(entry["stat_error"])


# ---------------------------------------------------------------------------
# Empty bins: recorded, never dropped; the committed JSON encoding
# ---------------------------------------------------------------------------


@requires_sps_tables
def test_empty_bins_recorded_and_infinite_mc_encoding(cov_tables, tmp_path):
    """N_bin == 0 bins keep their entry and are listed under empty_bins."""
    tables = cov_tables
    # All bins empty except one k_plus bin: the tabulated data value with
    # 20 tracks, so the populated bin's pull is exactly 0.
    table_kp = next(t for t in tables if t.species == "k_plus")
    counts = {t.species: np.zeros(int(t.data.size), dtype=int) for t in tables}
    values = {t.species: np.zeros(int(t.data.size)) for t in tables}
    populated = _interior_target(table_kp)
    d = float(table_kp.data[populated[0]])
    counts["k_plus"][populated[0]] = 20
    values["k_plus"][populated[0]] = d
    mc_err = {
        t.species: r01.mc_error_array(values[t.species], counts[t.species])
        for t in tables
    }
    # The populated bin follows value * sqrt(1/N_bin); every other bin
    # carries the infinite MC error.
    assert mc_err["k_plus"][populated[0]] == pytest.approx(
        d * math.sqrt(1.0 / 20.0)
    )
    assert math.isinf(mc_err["k_plus"][0])

    residuals = r01.residuals_payload(tables, values, mc_err)
    # Nothing is silently dropped: every species keeps every bin.
    assert len(residuals) == 1017
    for table in tables:
        for i in range(int(table.data.size)):
            entry = [e for e in residuals if e["species"] == table.species][i]
            if counts[table.species][i] == 0:
                assert math.isinf(entry["sigma"])
                # d is the table value (finite), m = 0; the pull of a bin
                # with infinite sigma is 0.0 by the committed convention.
                assert math.isfinite(entry["d"])
                assert entry["m"] == 0.0
                assert entry["delta_over_sigma"] == 0.0
            else:
                assert math.isfinite(entry["sigma"])

    failure_map = r01.build_failure_map(residuals, counts)
    assert failure_map["n_flagged"] == 0
    assert failure_map["n_empty_bins"] == 1017 - 1
    assert failure_map["empty_bins"][0] == {
        "species": "pi_minus",
        "bin_index": 0,
    }
    empty_set = {(e["species"], e["bin_index"]) for e in failure_map["empty_bins"]}
    expected = {
        (t.species, i)
        for t in tables
        for i in range(int(t.data.size))
        if counts[t.species][i] == 0
    }
    assert empty_set == expected
    assert ("k_plus", populated[0]) not in empty_set

    # The committed JSON encoding of the infinite variance is the
    # string sentinel "Infinity" (RFC 8259 safe): the written artifact
    # strict-parses with no bare Infinity token, and float() restores
    # the infinite variance on read.
    covariance = r01.covariance_payload(tables, mc_err, r01.read_sigma_eta())
    path = tmp_path / "v_mc.json"
    r01._write_json(path, covariance["V_mc"])
    text = path.read_text(encoding="utf-8")

    def _reject_constant(constant):
        raise AssertionError(f"bare JSON constant {constant!r} is not RFC 8259")

    loaded = json.loads(text, parse_constant=_reject_constant)
    assert loaded[0]["variance"] == "Infinity"
    assert math.isinf(float(loaded[0]["variance"]))
    inf_mc = [
        e["variance"]
        for e in covariance["V_mc"]
        if e["species"] == "pi_minus"
    ]
    assert all(math.isinf(v) for v in inf_mc)
    inf_combined = [
        e["variance"]
        for e in covariance["V_combined"]
        if e["species"] == "k_plus" and e["bin_index"] != populated[0]
    ]
    assert all(math.isinf(v) for v in inf_combined)


# ---------------------------------------------------------------------------
# Covariance assembly (card 7 combined covariance, diagonal)
# ---------------------------------------------------------------------------


@requires_sps_tables
def test_covariance_assembly_and_card7_nuisance(cov_tables):
    """V_combined = V_data + V_mc element-wise; the card-7 nuisance block."""
    tables = cov_tables
    # Hand-constructed MC errors: 0.1 everywhere, 0.5 at one bin.
    mc_err = {
        t.species: np.full(int(t.data.size), 0.1) for t in tables
    }
    target = _interior_target(tables[0])
    mc_err[tables[0].species][target[0]] = 0.5

    # sigma_eta is read from the card-7 config; all six table nuisances
    # share the width 0.01 (config/v2_singlet/likelihood_nuisances.json).
    assert r01.read_sigma_eta() == pytest.approx(0.01)
    covariance = r01.covariance_payload(tables, mc_err, r01.read_sigma_eta())
    assert covariance["normalization_nuisance"] == {
        "eta": 0.0,
        "sigma_eta": 0.01,
        "table_scope": "all_six_tables",
    }

    data = {(e["species"], e["bin_index"]): e["variance"] for e in covariance["V_data"]}
    mc = {(e["species"], e["bin_index"]): e["variance"] for e in covariance["V_mc"]}
    combined = {
        (e["species"], e["bin_index"]): e["variance"]
        for e in covariance["V_combined"]
    }
    assert len(data) == len(mc) == len(combined) == 1017
    # V_data is the squared table error; V_mc the squared hand error.
    for table in tables:
        for i in range(int(table.data.size)):
            key = (table.species, i)
            assert data[key] == pytest.approx(
                float(table.sigma[i]) ** 2, rel=1e-12, abs=0.0
            )
            expected_mc = 0.5 if (table is tables[0] and i == target[0]) else 0.1
            assert mc[key] == pytest.approx(expected_mc**2, rel=0.0, abs=0.0)
            # Diagonal sum, element-wise.
            assert combined[key] == pytest.approx(
                data[key] + mc[key], rel=1e-12, abs=0.0
            )


# ---------------------------------------------------------------------------
# Chi2: the recorded DOF convention and the sealed 5M-event run
# ---------------------------------------------------------------------------


def _pull_entry(species: str, index: int, pull: float) -> dict:
    return {
        "species": species,
        "bin_index": index,
        "d": float(pull),
        "m": 0.0,
        "sigma": 1.0,
        "delta_over_sigma": float(pull),
    }


def test_chi2_dof_convention_hand_payload():
    """chi2 = sum(pull**2) per species with the committed DOF bins - 1."""
    residuals = [_pull_entry("pi_minus", i, v) for i, v in enumerate(
        [0.5, -2.0, 0.0, 4.0, 0.0]
    )]
    residuals += [_pull_entry("k_plus", i, v) for i, v in enumerate([1.0, -1.0])]
    counts = {
        "pi_minus": np.array([0, 1, 1, 1, 1], dtype=int),
        "k_plus": np.ones(2, dtype=int),
    }
    chi2 = r01.chi2_payload(residuals, counts)
    assert chi2["dof_convention"] == CHI2_DOF_CONVENTION
    pi = chi2["per_species"]["pi_minus"]
    assert pi["chi2"] == pytest.approx(0.25 + 4.0 + 0.0 + 16.0 + 0.0)
    assert pi["dof"] == 4  # bins - 1
    assert pi["chi2_dof"] == pytest.approx(20.25 / 4.0)
    assert pi["n_bins"] == 5
    assert pi["n_empty_bins"] == 1
    kp = chi2["per_species"]["k_plus"]
    assert kp["chi2"] == pytest.approx(2.0)
    assert kp["dof"] == 1
    assert chi2["total_chi2"] == pytest.approx(22.25)
    assert chi2["total_bins"] == 7
    assert chi2["total_dof"] == 6  # total_bins - 1


@pytest.mark.skipif(
    not _SEALED_PRESENT,
    reason=(
        "sealed R01 T3 run directory not present on this checkout "
        f"({SEALED_RUN_DIR}); read-only cross-check skipped"
    ),
)
def test_chi2_matches_sealed_r01_run():
    """The sealed run's chi2.json equals sum(pull**2) over its residuals."""
    residuals = _load(SEALED_RUN_DIR / "residuals.json")
    chi2 = _load(SEALED_RUN_DIR / "chi2.json")
    assert len(residuals) == 1017
    assert chi2["dof_convention"] == CHI2_DOF_CONVENTION
    assert set(chi2["per_species"]) == set(SPECIES_KEYS)
    recomputed: dict[str, float] = {}
    n_bins: dict[str, int] = {}
    for entry in residuals:
        recomputed[entry["species"]] = (
            recomputed.get(entry["species"], 0.0)
            + float(entry["delta_over_sigma"]) ** 2
        )
        n_bins[entry["species"]] = n_bins.get(entry["species"], 0) + 1
    for species, per in chi2["per_species"].items():
        assert per["n_bins"] == SPECIES_BIN_SCHEMA[species]
        assert per["dof"] == per["n_bins"] - 1  # the bins - 1 convention
        assert per["chi2"] == pytest.approx(recomputed[species], rel=1e-9)
    assert chi2["total_bins"] == 1017
    assert chi2["total_dof"] == 1016
    # The run evidence of the card's 5M-event run (recorded, not gated).
    assert chi2["per_species"]["pi_minus"]["chi2"] == pytest.approx(
        1329031.99, rel=1e-6
    )
    assert chi2["per_species"]["pi_plus"]["chi2"] == pytest.approx(
        1450782.79, rel=1e-6
    )


# ---------------------------------------------------------------------------
# Failure map: threshold boundary, ranking, tie-break
# ---------------------------------------------------------------------------


def test_failure_map_threshold_boundary_ranking_and_tie_break():
    """Strict > 3.0 flagging, |pull| ranking, and the committed tie-break."""
    counts = {s: np.ones(10, dtype=int) for s in SPECIES_KEYS}
    residuals = [
        # |pull| 5.0 three-way tie across species: the committed
        # tie-break is the config species display order (pi-, pi+, K+,
        # K-, p, pbar) and then the lower bin index, sign-agnostic.
        _pull_entry("k_plus", 7, 5.0),
        _pull_entry("pi_minus", 2, -5.0),
        _pull_entry("antiproton", 0, -5.0),
        # 4.5: below the tie, rank 6.
        _pull_entry("proton", 5, 4.5),
        # 3.0001: just above the threshold, rank 7.
        _pull_entry("pi_minus", 9, 3.0001),
        # Exactly 3.0: NOT flagged (strict >).
        _pull_entry("pi_plus", 3, 3.0),
        # Same-species |pull| 6.0 tie: the lower bin index ranks first.
        _pull_entry("pi_plus", 3, 6.0),
        _pull_entry("pi_plus", 1, -6.0),
    ]
    failure_map = r01.build_failure_map(residuals, counts, threshold=3.0)
    assert failure_map["threshold"] == 3.0
    assert failure_map["n_total_bins"] == 1017
    # 8 candidates minus the exact-3.0 bin: 7 flagged.
    assert failure_map["n_flagged"] == 7
    flagged = failure_map["flagged_cells"]
    assert [c["rank"] for c in flagged] == [1, 2, 3, 4, 5, 6, 7]
    # |pull| descending; the 6.0 pair ranks above the 5.0 tie.
    assert (flagged[0]["species"], flagged[0]["bin_index"], flagged[0]["pull"]) == (
        "pi_plus", 1, -6.0,
    )
    assert (flagged[1]["species"], flagged[1]["bin_index"], flagged[1]["pull"]) == (
        "pi_plus", 3, 6.0,
    )
    # The 5.0 tie breaks to species display order (pi- before K+ before
    # pbar), then bin index; the signed pull is preserved.
    assert (
        flagged[2]["species"],
        flagged[2]["bin_index"],
        flagged[2]["pull"],
    ) == ("pi_minus", 2, -5.0)
    assert (
        flagged[3]["species"],
        flagged[3]["bin_index"],
        flagged[3]["pull"],
    ) == ("k_plus", 7, 5.0)
    assert (
        flagged[4]["species"],
        flagged[4]["bin_index"],
        flagged[4]["pull"],
    ) == ("antiproton", 0, -5.0)
    assert (
        flagged[5]["species"],
        flagged[5]["bin_index"],
        flagged[5]["pull"],
    ) == ("proton", 5, 4.5)
    assert flagged[6]["pull"] == pytest.approx(3.0001)
    # The exact-3.0 bin (pi_plus, 3 at pull 3.0) is not flagged; the
    # same bin at pull 6.0 is.
    assert all(
        not (c["species"] == "pi_plus" and c["bin_index"] == 3
             and abs(c["pull"]) == pytest.approx(3.0))
        for c in flagged
    )


# ---------------------------------------------------------------------------
# Gate-0 assembly (hand-built run directories, no generation)
# ---------------------------------------------------------------------------


def _build_gate0_fixture(out_dir: Path, n_completed: int) -> None:
    """Hand-built run directory at the committed seal ordering."""
    out_dir.mkdir(parents=True)
    (out_dir / r01.PREREGISTRATION_NAME).write_text(
        json.dumps(r01.default_pre_registration(), indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / r01.BIN_PREDICTIONS_NAME).write_text(
        json.dumps(
            [{"species": "pi_minus", "bin_index": i} for i in range(1017)]
        ),
        encoding="utf-8",
    )
    (out_dir / r01.FAILURE_MAP_NAME).write_text(
        json.dumps({"n_flagged": 3, "n_empty_bins": 0}) + "\n", encoding="utf-8"
    )
    (out_dir / r01.PROVENANCE_NAME).write_text("{}\n", encoding="utf-8")
    (out_dir / r01.CLEAN_ENV_NAME).write_text(
        json.dumps(
            {
                r01.CLEAN_ENV_PYTEST_KEY: "pass",
                r01.CLEAN_ENV_SMOKE_KEY: "pass",
            }
        ),
        encoding="utf-8",
    )
    (out_dir / r01.REPLICA_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "seed_base": CARD_SEED_BASE,
                "seeds": [
                    {
                        "seed": 20300911 + k,
                        "replica_index": k,
                        "status": "completed" if k < n_completed else "pending",
                        "n_events": 250000 if k < n_completed else None,
                    }
                    for k in range(20)
                ],
            }
        ),
        encoding="utf-8",
    )
    # Sealing happens after every payload file: the sums inventory is
    # the last write before the acceptance record.
    r01.write_sha256sums(out_dir)


def _production_verification() -> dict:
    return {
        "config_sha256": r01.PRODUCTION_CONFIG_SHA256,
        "m8_parameter_hash": r01.M8_PARAMETER_HASH,
    }


@pytest.mark.parametrize("n_completed, status", [(20, "complete"), (1, "deferred")])
def test_gate0_tier_status_complete_and_deferred(tmp_path, n_completed, status):
    """20/20 completes the tier; 1/20 defers it without blocking the verdict."""
    out_dir = tmp_path / f"gate0_{n_completed}_of_20"
    _build_gate0_fixture(out_dir, n_completed)
    gate = r01.assemble_gate0(
        out_dir, _production_verification(), r01.default_pre_registration()
    )
    assert gate["gate"] == "gate0"
    assert gate["clause"] == r01.GATE_CLAUSE
    assert gate["config_hash_verified"] is True
    assert gate["m8_parameter_hash_verified"] is True
    assert gate["archive_sealed"] is True
    assert gate["immutable"] is True
    assert gate["n_total_bins"] == 1017
    assert gate["n_flagged_cells"] == 3
    assert gate["empty_bins"] == 0
    assert gate["clean_environment_commands"] == {
        r01.CLEAN_ENV_PYTEST_KEY: "pass",
        r01.CLEAN_ENV_SMOKE_KEY: "pass",
    }
    assert gate["archive_seal_sha256"] == r01.sha256_file(
        out_dir / r01.SHA256SUMS_NAME
    )
    assert gate["tier_status"] == status
    if status == "complete":
        assert gate["tier_deferral_reason"] is None
        assert gate["tier"]["replicas_completed"] == 20
        assert gate["tier"]["events_completed"] == 20 * 250000
    else:
        assert gate["tier_deferral_reason"] is not None
        assert "1 of 20" in gate["tier_deferral_reason"]
        assert "follow-up obligation" in gate["tier_deferral_reason"]
        assert gate["tier"]["replicas_completed"] == 1
        assert gate["tier"]["events_completed"] == 250000
    # A deferral does not block Gate 0: the verdict is pass when the
    # hashes verify, the archive seals, and the commands recorded pass.
    assert gate["verdict"] == "pass"
    # The acceptance record is written under the card's file name and
    # matches the returned payload.
    assert _load(out_dir / r01.GATE_RECORD_NAME) == gate


def test_gate0_seal_reverified_after_payload_mutation(tmp_path):
    """A post-seal payload mutation flips immutable and the verdict."""
    out_dir = tmp_path / "gate0_mutation"
    _build_gate0_fixture(out_dir, 20)
    verification = _production_verification()
    gate = r01.assemble_gate0(
        out_dir, verification, r01.default_pre_registration()
    )
    assert gate["immutable"] is True
    assert gate["verdict"] == "pass"

    # Mutate a payload file after sealing (kept as valid JSON, since the
    # acceptance record re-parses it): the record re-verifies every
    # SHA256SUMS.txt entry and must see the digest drift.
    payload = out_dir / r01.FAILURE_MAP_NAME
    payload.write_text(
        json.dumps({"n_flagged": 4, "n_empty_bins": 0}) + "\n", encoding="utf-8"
    )
    mutated = r01.assemble_gate0(
        out_dir, verification, r01.default_pre_registration()
    )
    assert mutated["immutable"] is False
    assert mutated["verdict"] == "fail"
    assert mutated["archive_sealed"] is True  # the files still exist


# ---------------------------------------------------------------------------
# Seal: the --seal path on a small run directory (100 events)
# ---------------------------------------------------------------------------


def test_cli_seal_small_run(tmp_path):
    """--seal on a 100-event smoke run writes the immutable-archive triple."""
    out_dir = tmp_path / "seal_run"
    assert not out_dir.exists()
    proc = _run_driver(
        [
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
        r01.PREREGISTRATION_NAME,
        f"replica_{SMOKE_SEED}.json",
        r01.REPLICA_MANIFEST_NAME,
        r01.BIN_PREDICTIONS_NAME,
        r01.COVARIANCE_NAME,
        r01.RESIDUALS_NAME,
        r01.FAILURE_MAP_NAME,
        r01.CHI2_NAME,
        r01.PROVENANCE_NAME,
        r01.SHA256SUMS_NAME,
        r01.SEAL_NAME,
    ):
        assert (out_dir / name).is_file(), name
    # The smoke config is not the production config: no gate record.
    assert not (out_dir / r01.GATE_RECORD_NAME).is_file()
    assert "no gate record (non-production config)" in proc.stdout

    # The binned artifacts carry the 1017-bin set.
    predictions = _load(out_dir / r01.BIN_PREDICTIONS_NAME)
    assert len(predictions) == 1017
    assert {
        s: sum(1 for e in predictions if e["species"] == s) for s in SPECIES_KEYS
    } == SPECIES_BIN_SCHEMA

    # SHA256SUMS.txt: sha256sum -c format over every regular file except
    # the files that cannot carry their own digests.
    sums_text = (out_dir / r01.SHA256SUMS_NAME).read_text(encoding="utf-8")
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
        r01.SHA256SUMS_NAME,
        r01.SEAL_NAME,
        r01.PROVENANCE_NAME,
        r01.GATE_RECORD_NAME,
    }
    # Re-hash one payload file and confirm the recorded digest matches
    # (unit level; no driver re-generation).
    assert digests[r01.BIN_PREDICTIONS_NAME] == r01.sha256_file(
        out_dir / r01.BIN_PREDICTIONS_NAME
    )
    for line in lines:
        digest, name = line.split("  ", 1)
        payload = (out_dir / name).read_bytes()
        assert digest == hashlib.sha256(payload).hexdigest()

    # The seal manifest records every file except itself, digests match.
    seal = _load(out_dir / r01.SEAL_NAME)
    sealed = {entry["path"] for entry in seal["files"]}
    assert sealed == every_file - {r01.SEAL_NAME}
    for entry in seal["files"]:
        path = out_dir / entry["path"]
        assert entry["sha256"] == r01.sha256_file(path)
        assert entry["bytes"] == path.stat().st_size

    # The provenance record carries the run identity and the code
    # fingerprint (RESONANCE_DECAY_LIB_VERSION salt).
    provenance = _load(out_dir / r01.PROVENANCE_NAME)
    assert provenance["producer"] == "staged_r01_reference.py"
    assert provenance["config_sha256"] == r01.SMOKE_CONFIG_SHA256
    assert provenance["m8_parameter_hash"] == r01.M8_PARAMETER_HASH
    assert provenance["sigma_inel_mb"] == 31.78
    # seed_base is recorded from the pre-registration (the card's
    # declared T3 base), while replica_seeds carries the actual run
    # seeds: a smoke seed base is reported alongside, not in place of,
    # the pre-registered one.
    assert provenance["seed_base"] == CARD_SEED_BASE
    assert provenance["seed_offset"] == 40000
    assert provenance["replica_seeds"] == [SMOKE_SEED]
    assert provenance["replicas_completed"] == 1
    assert provenance["events_completed"] == SEAL_N_EVENTS
    assert provenance["tier_status"] == "deferred"
    fingerprint = provenance["code_fingerprint"]
    assert fingerprint["resonance_decay_lib_version"] == RESONANCE_DECAY_LIB_VERSION
    assert re.fullmatch(r"[0-9a-f]{64}", fingerprint["staged_generation_code_sha256"])

    # The per-replica seed archive records the T3 tier incrementally:
    # the requested seed is completed, the remaining 19 are pending.
    manifest = _load(out_dir / r01.REPLICA_MANIFEST_NAME)
    assert manifest["requested_replicas"] == 1
    assert manifest["tier_status"] == "deferred"
    assert manifest["seeds"] == [
        {
            "seed": SMOKE_SEED,
            "replica_index": 0,
            "status": "completed",
            "n_events": SEAL_N_EVENTS,
            "wall_seconds": manifest["seeds"][0]["wall_seconds"],
            "event_hash_set_sha256": (
                manifest["seeds"][0]["event_hash_set_sha256"]
            ),
            "record": f"replica_{SMOKE_SEED}.json",
        }
    ]
    # The archive tracks the declared T3 tier incrementally: a smoke
    # seed base is not part of the declared seed list, so every
    # declared entry stays pending (with no record) even though the
    # requested-seed record above completes.
    archive = manifest["t3_seed_archive"]
    assert len(archive) == 20
    assert [e["status"] for e in archive] == ["pending"] * 20
    assert [e["seed"] for e in archive] == manifest["declared_t3"]["seeds"]
    assert [e["replica_index"] for e in archive] == list(range(20))
    assert all(e["record"] is None for e in archive)
    assert all(e["n_events"] is None for e in archive)
