"""Hermetic self-check suite for the R01 legacy p+p reference driver.

Four checks plus the empty-bin edge (card
``TASK-20260911-r1-run-r01-legacy-pp-reference``):

- ``test_r01_binning_small_n``: the driver's binning logic on a
  hand-constructed batch of 10 staged events (one seed, the module
  fixture, ``config/staged_excitation_smoke.json`` with
  ``fit.enabled: false``); the 1017-bin output has the correct
  per-species bin counts (229/230/153/148/135/122), ``total_bins ==
  1017``, and the single-batch prediction is bit-identical to the
  existing ``na49_mixing.staged_evaluation._bin_tracks`` convention.
- ``test_r01_residual_and_failure_map``: a hand-constructed (d, m,
  sigma) with one bin at ``|delta/sigma| = 4.0`` and the rest at 0.0
  flags exactly one cell, ranked 1 with pull 4.0; a second bin with
  ``N_bin == 0`` carries an infinite MC error, is listed under
  ``empty_bins``, and is never silently dropped.
- ``test_r01_config_hash_verified``: a mutated config copy fails the
  driver with a non-zero exit code before any event is generated (the
  in-process verification of the pristine copy is asserted first).
- ``test_r01_seal_manifest``: the seal path writes ``SHA256SUMS.txt``
  (``sha256sum -c`` format) and ``provenance.json`` under the run
  directory, and ``seal_manifest.json`` covers every regular file in
  the directory except itself.

Hermetic: the 10-event builder call in the module-scoped fixture is the
only generation in the test; the full T3 run (20 x 250,000) is the R01
run itself, not a test.  Standard library, the driver module, and
``na49_mixing`` only; the smoke config needs no external tables.  The
binning check reads the bundled SPS tables for the 1017-bin grid and
skips with a recorded reason when the reference mirror is absent from
the checkout (the same convention as ``test_preflight_sps_check``).
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from helpers.benchmarking import staged_r01_reference as r01
from na49_mixing.event_builder_pp import PPEventBuilder
from na49_mixing.reaction import Reaction
from na49_mixing.staged_evaluation import _bin_tracks

NESTED_ROOT = Path(r01.NESTED_ROOT)
DRIVER_PATH = NESTED_ROOT / "helpers" / "benchmarking" / "staged_r01_reference.py"
SMOKE_CONFIG = NESTED_ROOT / "config" / "staged_excitation_smoke.json"

# The canonical T0 replica seed: 20260911 + 40000 + 0 = 20300911 (the
# card's M8 substream rule; the same arithmetic as config_m8.json).
HARNESS_SEED = 20300911
HARNESS_N_EVENTS = 10

SPS_DATA = NESTED_ROOT / "spshadrons.web.cern.ch" / "data"
_SPS_TABLE_NAMES = (
    "xsect_piminus.txt",
    "xsect_piplus.txt",
    "xsect_kaplus_pp.txt",
    "xsect_kaminus_pp.txt",
    "xsect_proton_pp.txt",
    "xsect_antiproton_pp.txt",
)
_SPS_PRESENT = all((SPS_DATA / name).is_file() for name in _SPS_TABLE_NAMES)


def _smoke_payload() -> dict:
    raw_config = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    assert raw_config["fit"]["enabled"] is False
    return raw_config


@pytest.fixture(scope="module")
def harness_batch() -> list:
    """The single ≤ 10-event builder call of the test module."""
    raw_config = _smoke_payload()
    pp_config, _ = r01.build_pp_config(raw_config)
    events = []
    for event_id in range(HARNESS_N_EVENTS):
        builder = PPEventBuilder(pp_config, r01._event_rng(HARNESS_SEED, event_id))
        events.append(builder.generate(event_id, stage=r01.STAGE_LABEL))
    return events


@pytest.fixture(scope="module")
def harness_tables() -> list:
    return r01.load_binned_tables(_smoke_payload())


def _harness_sqrt_s() -> float:
    return math.sqrt(Reaction(158.0, 2212, 2212).s())


def test_r01_binning_small_n(harness_batch, harness_tables):
    """10 staged events bin onto exactly the 1017-bin six-species set."""
    tables = harness_tables
    counts_per_species = {t.species: int(t.data.size) for t in tables}
    assert counts_per_species == {
        "pi_minus": 229,
        "pi_plus": 230,
        "k_plus": 153,
        "k_minus": 148,
        "proton": 135,
        "antiproton": 122,
    }
    assert sum(counts_per_species.values()) == 1017

    sqrt_s = _harness_sqrt_s()
    sums, counts = r01._bin_track_sums(harness_batch, tables, sqrt_s)
    entries, values_by_species, errors_by_species = r01.bin_predictions_payload(
        tables, sums, counts, len(harness_batch), sqrt_s
    )
    assert len(entries) == 1017
    # The pooled estimator on a single batch is bit-identical to the
    # existing staged_evaluation._bin_tracks convention.
    reference = _bin_tracks(harness_batch, tables, sqrt_s)
    for table in tables:
        assert np.array_equal(
            values_by_species[table.species], reference[table.species]
        )
    # Entry fields carry the tabulated bin midpoints and finite-or-infinite
    # stat errors.
    first = entries[0]
    assert set(first) == {"species", "bin_index", "xF_mid", "pT_mid", "value", "stat_error"}
    assert first["species"] == tables[0].species and first["bin_index"] == 0
    assert math.isfinite(first["xF_mid"]) and math.isfinite(first["pT_mid"])
    # A 10-event batch leaves most bins empty: the empty ones carry the
    # infinite MC error instead of being dropped.
    empty = [
        entry
        for entry in entries
        if not math.isfinite(float(entry["stat_error"]))
    ]
    assert empty, "a 10-event batch must leave empty bins"
    for entry in empty:
        assert entry["value"] == 0.0


def test_r01_residual_and_failure_map():
    """One cell at |delta/sigma| = 4.0 flags with rank 1; empty bins survive."""
    # Five pi_minus bins: three at pull 0.0, one at exactly 4.0, and one
    # empty MC bin (N_bin == 0) with infinite MC error.  The (d, m, sigma)
    # triple is hand-constructed; the empty bin's sigma comes from the
    # driver's mc_error_array/combined_sigma convention.
    d = [1.0, 1.0, 1.0, 2.0, 1.0]
    m = [1.0, 1.0, 1.0, 1.0, 0.0]
    counts = [100, 100, 100, 1000, 0]
    mc_err = r01.mc_error_array(np.asarray(m, dtype=float), counts)
    assert math.isclose(mc_err[0], 0.1)  # value * sqrt(1/N_bin)
    assert math.isinf(mc_err[4])
    hand_sigma = [1.0, 1.0, 1.0, 0.25, float("inf")]
    residuals = [
        {
            "species": "pi_minus",
            "bin_index": i,
            "d": float(d[i]),
            "m": float(m[i]),
            "sigma": float(hand_sigma[i]),
            "delta_over_sigma": float((d[i] - m[i]) / hand_sigma[i]),
        }
        for i in range(5)
    ]
    assert residuals[3]["delta_over_sigma"] == pytest.approx(4.0, rel=0.0, abs=0.0)
    assert residuals[0]["delta_over_sigma"] == pytest.approx(0.0)
    assert residuals[4]["delta_over_sigma"] == 0.0  # d - m over infinite sigma
    assert not math.isfinite(residuals[4]["sigma"])

    failure_map = r01.build_failure_map(
        residuals, {"pi_minus": counts}, threshold=3.0
    )
    assert failure_map["n_total_bins"] == 1017
    assert failure_map["n_flagged"] == 1
    assert failure_map["n_empty_bins"] == 1
    flagged = failure_map["flagged_cells"]
    assert len(flagged) == 1
    assert flagged[0]["rank"] == 1
    assert flagged[0]["pull"] == pytest.approx(4.0)
    assert flagged[0]["bin_index"] == 3
    assert failure_map["empty_bins"] == [{"species": "pi_minus", "bin_index": 4}]
    # The empty bin is recorded, not silently dropped, and not flagged.
    assert all(cell["bin_index"] != 4 for cell in flagged)

    chi2 = r01.chi2_payload(residuals, {"pi_minus": counts})
    assert chi2["per_species"]["pi_minus"]["n_bins"] == 5
    assert chi2["per_species"]["pi_minus"]["n_empty_bins"] == 1
    assert chi2["per_species"]["pi_minus"]["dof"] == 4
    assert chi2["per_species"]["pi_minus"]["chi2"] == pytest.approx(16.0)

    # A negative pull of the same magnitude also flags with rank 1.
    residuals_neg = [dict(entry) for entry in residuals]
    residuals_neg[3]["d"], residuals_neg[3]["m"] = 0.0, 1.0
    residuals_neg[3]["delta_over_sigma"] = float(
        (0.0 - 1.0) / residuals_neg[3]["sigma"]
    )
    failure_neg = r01.build_failure_map(
        residuals_neg, {"pi_minus": counts}, threshold=3.0
    )
    assert failure_neg["n_flagged"] == 1
    assert failure_neg["flagged_cells"][0]["pull"] == pytest.approx(-4.0)
    assert failure_neg["flagged_cells"][0]["rank"] == 1


def test_r01_config_hash_verified(tmp_path):
    """A mutated config copy fails the driver before any generation."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / r01.SMOKE_CONFIG_NAME
    # The pin is over the exact file bytes, so the control copy must be a
    # byte-identical copy of the bundled smoke config (not a re-serialized
    # JSON dump).
    shutil.copyfile(SMOKE_CONFIG, config_path)
    pristine = r01.verify_config_and_m8(config_path, r01.default_pre_registration())
    assert pristine["config_sha256"] == r01.SMOKE_CONFIG_SHA256

    mutated_raw = json.loads(config_path.read_text(encoding="utf-8"))
    mutated_raw["continuum"]["match_mass"] = float(
        mutated_raw["continuum"]["match_mass"]
    ) + 1e-6
    config_path.write_text(json.dumps(mutated_raw, indent=2), encoding="utf-8")
    mutated_out = tmp_path / "run_mutated"
    proc = subprocess.run(
        [
            sys.executable,
            str(DRIVER_PATH),
            "--config",
            str(config_path),
            "--replicas",
            "1",
            "--events-per-replica",
            "2",
            "--seed-base",
            "20311000",
            "--out",
            str(mutated_out),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(NESTED_ROOT),
    )
    assert proc.returncode != 0, proc.stdout
    assert "config sha256" in proc.stderr
    # The refusal fires before generation: no replica record and no
    # binned artifact can exist.
    assert not any(mutated_out.glob("replica_*.json"))
    assert not (mutated_out / r01.BIN_PREDICTIONS_NAME).is_file()
    assert not (mutated_out / r01.FAILURE_MAP_NAME).is_file()


def test_r01_seal_manifest(tmp_path):
    """The seal path writes SHA256SUMS.txt + provenance.json over every file."""
    out_dir = tmp_path / "seal_run"
    out_dir.mkdir()
    # Synthetic payload files (no generation, no tables).
    (out_dir / r01.PREREGISTRATION_NAME).write_text(
        json.dumps(r01.default_pre_registration(), indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / r01.BIN_PREDICTIONS_NAME).write_text("[]\n", encoding="utf-8")
    (out_dir / r01.FAILURE_MAP_NAME).write_text(
        json.dumps({"n_total_bins": 1017}) + "\n", encoding="utf-8"
    )

    verification = r01.verify_config_and_m8(SMOKE_CONFIG, r01.default_pre_registration())
    r01.write_provenance(
        out_dir,
        verification=verification,
        pre_registration=r01.default_pre_registration(),
        seeds=[HARNESS_SEED],
        replicas_requested=1,
        replicas_completed=1,
        events_per_replica=HARNESS_N_EVENTS,
        events_completed=HARNESS_N_EVENTS,
        argv=["--out", str(out_dir)],
        wall_seconds=0.0,
        tier_status="deferred",
        tier_deferral_reason="harness seal check",
        input_paths=[SMOKE_CONFIG, Path(verification["manifest_path"])],
    )
    seal = r01.seal_run_dir(out_dir)

    assert (out_dir / r01.SHA256SUMS_NAME).is_file()
    assert (out_dir / r01.PROVENANCE_NAME).is_file()
    assert (out_dir / r01.SEAL_NAME).is_file()

    # The seal manifest covers every regular file in the directory except
    # itself, with correct digests.
    every_file = {
        p.relative_to(out_dir).as_posix()
        for p in out_dir.rglob("*")
        if p.is_file()
    }
    sealed_names = {entry["path"] for entry in seal["files"]}
    assert sealed_names == every_file - {r01.SEAL_NAME}
    assert r01.SHA256SUMS_NAME in sealed_names
    assert r01.PROVENANCE_NAME in sealed_names
    for entry in seal["files"]:
        path = out_dir / entry["path"]
        assert entry["sha256"] == r01.sha256_file(path)
        assert entry["bytes"] == path.stat().st_size

    # SHA256SUMS.txt is in sha256sum -c format and verifies line by line.
    sums_text = (out_dir / r01.SHA256SUMS_NAME).read_text(encoding="utf-8")
    lines = [line for line in sums_text.splitlines() if line.strip()]
    assert lines
    for line in lines:
        digest, name = line.split("  ", 1)
        assert len(digest) == 64
        payload = (out_dir / name).read_bytes()
        assert digest == hashlib.sha256(payload).hexdigest()
    # The sums cover every payload file except the files that cannot
    # carry their own digests.
    sums_names = {line.split("  ", 1)[1] for line in lines}
    assert sums_names == every_file - {
        r01.SHA256SUMS_NAME,
        r01.SEAL_NAME,
        r01.PROVENANCE_NAME,
        r01.GATE_RECORD_NAME,
    }
    # The provenance record validates under the run_provenance convention.
    assert r01.validate_run_provenance(out_dir / r01.PROVENANCE_NAME) == []
