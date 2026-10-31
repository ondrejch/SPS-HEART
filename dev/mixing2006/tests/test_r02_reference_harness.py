"""Hermetic self-check suite for the R02 legacy p+A reference driver.

The five card-named checks plus the empty-bin edge (card
``TASK-20260911-r1-run-r02-legacy-pa-reference``):

- ``test_r02_binning_small_n``: the driver's binning logic on a
  hand-constructed batch of 10 p+A staged events (one seed, the module
  fixture, ``config/staged_excitation_smoke.json`` with
  ``fit.enabled: false``); the 1017-bin output has the correct
  per-species bin counts (229/230/153/148/135/122) and
  ``total_bins == 1017``, and the single-batch prediction is
  bit-identical to the existing ``na49_mixing.staged_evaluation._bin_tracks``
  convention.
- ``test_r02_hemisphere_yields``: for a hand-constructed event batch with
  known xF values, the hemisphere yields (projectile-side ``xF > 0.2``,
  central ``0.0 <= xF <= 0.2``, target-side ``xF < 0.0``) match the
  hand-computed sums to < 1e-12.
- ``test_r02_centrality_slice``: for a hand-constructed
  ``NucleusSpec(Pb208)`` at the pinned geometry, the pre-registered
  wounded-K slice of each M13-mapped centrality set selects the correct
  wounded-nucleon band (every drawn K inside [lo, hi], the sampled mean K
  against the closed-form band mean, and the closed-form slice fraction
  against the M13 pre-registered slice mass).
- ``test_r02_config_hash_verified``: a mutated config copy fails the
  driver with a non-zero exit code before any event is generated (the
  in-process verification of the pristine copy is asserted first).
- ``test_r02_seal_manifest``: the ``--seal`` path writes a
  ``SHA256SUMS.txt`` (sha256sum -c format) and a ``provenance.json``
  under the run directory, and ``seal_manifest.json`` covers every
  regular file in the directory except itself.
- ``test_r02_empty_bins_recorded``: a bin with ``N_bin == 0`` appears in
  the per-class failure map's ``empty_bins`` key with the infinite-MC-error
  encoding of the R01 driver (``stat_error`` serialized as literal JSON
  ``Infinity``, pull 0.0 by the infinite-sigma convention, never dropped),
  and model-only species carry null residuals with no flagging.

Hermetic: the 10-event builder call in the module-scoped fixture is the
only p+A generation in the test module; the full T2 run
(8 x 100,000 x 6 classes) is the R02 run itself, not a test.  The
centrality check draws wound counts only (closed-form sampler
arithmetic, no event construction).  Standard library, the driver
module, and ``na49_mixing`` only; the smoke config needs no external
tables.  The binning check reads the bundled SPS tables for the 1017-bin
grid and skips with a recorded reason when the reference mirror is
absent from the checkout (the same convention as
``test_preflight_sps_check``).
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from helpers.benchmarking import staged_r01_reference as r01
from helpers.benchmarking import staged_r02_reference as r02
from na49_mixing.event_builder_pA import PAEventBuilder
from na49_mixing.excitation_kinematics import FourVector
from na49_mixing.glauber import NucleusSpec, WoundedNucleonSampler
from na49_mixing.staged_evaluation import _bin_tracks

NESTED_ROOT = Path(r02.NESTED_ROOT)
DRIVER_PATH = NESTED_ROOT / "helpers" / "benchmarking" / "staged_r02_reference.py"
SMOKE_CONFIG = NESTED_ROOT / "config" / "staged_excitation_smoke.json"

# The canonical minimum-bias replica seed k = 0: 20260911 + 50000 + 0
# (the card's replica_{i,k} rule; R01's offset 40000 is disjoint).
HARNESS_SEED = 20310911
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
    """The single ≤ 10-event p+A builder call of the test module."""
    raw_config = _smoke_payload()
    pa_config, _applied = r02.build_pa_config(raw_config)
    builder = PAEventBuilder(pa_config, np.random.default_rng(HARNESS_SEED))
    return [builder.generate(event_id) for event_id in range(HARNESS_N_EVENTS)]


@pytest.fixture(scope="module")
def harness_tables() -> list:
    return r02.load_binned_tables(_smoke_payload())


def _harness_sqrt_s() -> float:
    from na49_mixing.reaction import Reaction

    return math.sqrt(Reaction(158.0, 2212, 2212).s())


def test_r02_binning_small_n(harness_batch, harness_tables):
    """10 p+A staged events bin onto exactly the 1017-bin six-species set."""
    if not _SPS_PRESENT:
        pytest.skip(
            "the bundled SPS tables are absent from this checkout "
            "(the 1017-bin grid needs them); recorded skip"
        )
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
    assert sum(counts_per_species.values()) == r02.TOTAL_BINS == 1017

    sqrt_s = _harness_sqrt_s()
    assert math.isclose(sqrt_s, r02.SQRT_S_NN_GEV_PIN, rel_tol=0, abs_tol=1e-9)
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
    assert set(first) == {
        "species",
        "bin_index",
        "xF_mid",
        "pT_mid",
        "value",
        "stat_error",
    }
    assert first["species"] == tables[0].species and first["bin_index"] == 0
    assert math.isfinite(first["xF_mid"]) and math.isfinite(first["pT_mid"])
    # A 10-event batch leaves most bins empty: the empty ones carry the
    # infinite MC error instead of being dropped (checked further in
    # test_r02_empty_bins_recorded).
    empty = [entry for entry in entries if not math.isfinite(float(entry["stat_error"]))]
    assert empty, "a 10-event batch must leave empty bins"
    for entry in empty:
        assert entry["value"] == 0.0


def test_r02_hemisphere_yields():
    """Hand-constructed tracks reproduce the hemisphere sums to < 1e-12."""
    sqrt_s = _harness_sqrt_s()
    # Hand-set tracks: one projectile-side pion (xF = 0.4), one central
    # pion (xF = 0.1), one target-side pion (xF = -0.2), one boundary
    # track exactly at xF = 0.2 (central by the pre-registered rule), one
    # boundary track exactly at xF = 0.0 (central), one kaon at
    # xF = 0.3 (projectile), and one pT = 0 spectator-like track at
    # xF = -0.5 (excluded, infinite weight).
    m = 0.1395699  # the pion table mass
    pt = 0.5

    def track(xf: float, pdg: int, p_t: float = pt):
        pz = xf * sqrt_s / 2.0
        energy = math.sqrt(m * m + p_t * p_t + pz * pz)
        return SimpleNamespace(pdg_id=pdg, p4=FourVector(energy, 0.0, p_t, pz))

    events = [
        SimpleNamespace(
            final_particles=[
                track(0.4, 211),   # pi_plus, projectile side
                track(0.1, -211),  # pi_minus, central
                track(-0.2, 211),  # pi_plus, target side (xF < 0)
                track(0.2, -211),  # pi_minus, boundary xF = 0.2 -> central
                track(0.0, 211),   # pi_plus, boundary xF = 0.0 -> central
                SimpleNamespace(   # a pT = 0 target-side track: excluded
                    pdg_id=211, p4=FourVector(m, 0.0, 0.0, -1.0)
                ),
            ]
        ),
        SimpleNamespace(
            final_particles=[
                track(0.3, 321),   # k_plus, projectile side
                track(-0.5, -211), # pi_minus, target side
            ]
        ),
    ]
    # The species list keyed exactly as the driver's table species (all
    # six species, the production shape; the synthetic tracks touch only
    # three of them).
    species_pdgs = {
        "pi_minus": (-211,),
        "pi_plus": (211,),
        "k_plus": (321,),
        "k_minus": (-321,),
        "proton": (2212,),
        "antiproton": (-2212,),
    }
    sums, counts = r02._hemisphere_track_sums(events, species_pdgs, sqrt_s)

    # Hand-computed sums per region (independent inline arithmetic; the
    # synthetic tracks all carry the same rest mass and pT).
    def hand_weight(xf: float) -> float:
        pz = xf * sqrt_s / 2.0
        energy = math.sqrt(m * m + pt * pt + pz * pz)
        return energy / pt

    hand = {
        # pi_minus: xF = +0.1 and the xF = 0.2 boundary track (central);
        # xF = -0.5 (target).
        "pi_minus": [0.0, hand_weight(0.1) + hand_weight(0.2), hand_weight(0.5)],
        # pi_plus: xF = +0.4 (projectile); xF = 0.0 boundary (central);
        # xF = -0.2 (target).
        "pi_plus": [hand_weight(0.4), hand_weight(0.0), hand_weight(0.2)],
        "k_plus": [hand_weight(0.3), 0.0, 0.0],
        "k_minus": [0.0, 0.0, 0.0],
        "proton": [0.0, 0.0, 0.0],
        "antiproton": [0.0, 0.0, 0.0],
    }
    hand_counts = {
        "pi_minus": [0, 2, 1],
        "pi_plus": [1, 1, 1],
        "k_plus": [1, 0, 0],
        "k_minus": [0, 0, 0],
        "proton": [0, 0, 0],
        "antiproton": [0, 0, 0],
    }
    for species, regions in hand.items():
        for slot, expected in enumerate(regions):
            assert sums[species][slot] == pytest.approx(expected, rel=0.0, abs=1e-12)
            assert int(counts[species][slot]) == hand_counts[species][slot]

    # The payload normalization: sum / (pi * sqrt(s) * N_events).
    n_events = 5
    entries = r02.hemisphere_yields_payload(sums, n_events, sqrt_s)
    norm = math.pi * sqrt_s * n_events
    assert set(entries[0]) == {"species", "projectile_side", "central", "target_side"}
    assert [e["species"] for e in entries] == list(species_pdgs)
    for species, regions in hand.items():
        record = next(e for e in entries if e["species"] == species)
        assert record["projectile_side"] == pytest.approx(
            regions[0] / norm, rel=0.0, abs=1e-12
        )
        assert record["central"] == pytest.approx(regions[1] / norm, rel=0.0, abs=1e-12)
        assert record["target_side"] == pytest.approx(regions[2] / norm, rel=0.0, abs=1e-12)
    # The target-side yields are nonzero (xF < 0 tracks exist in p+A
    # events even though the 1017-bin window starts at xF = 0).
    assert next(e for e in entries if e["species"] == "pi_minus")["target_side"] > 0.0
    assert next(e for e in entries if e["species"] == "pi_plus")["target_side"] > 0.0


def test_r02_centrality_slice():
    """The M13 pre-registered E_BE-to-K bands select the correct wounded K."""
    spec = r02.NucleusSpec(**r02.GEOMETRY_PB208)
    glauber = WoundedNucleonSampler(
        spec, np.random.default_rng(HARNESS_SEED), dr_fm=0.05
    )
    for set_no in (1, 2, 3, 4, 5):
        name = f"set_{set_no}"
        band = r02.WOUND_BANDS[name]
        lo, hi = band
        draws = [glauber.sample_event_in_range(lo, hi) for _ in range(3000)]
        ks = [int(k) for _b, k in draws]
        assert all(lo <= k <= hi for k in ks), name
        mean_k = sum(ks) / len(ks)
        closed_form = r02.E_BE_K_BAND_MEAN[name]
        # 4-sigma band of the sample mean; the widest mapped band's
        # variance (1.17) bounds the spread, the point bands have none.
        tolerance = max(0.06, 4.0 * math.sqrt(1.2 / len(ks)))
        assert abs(mean_k - closed_form) < tolerance, name
    # The closed-form slice masses match the M13 pre-registration's
    # slice masses to machine precision (the same sampler arithmetic).
    m13_slice_mass = {
        "set_1": 0.40967344939571354,
        "set_2": 0.12451622826546226,
        "set_3": 0.3886609204099082,
        "set_4": 0.08876227346491845,
        "set_5": 0.06603385842140419,
    }
    for name, expected in m13_slice_mass.items():
        lo, hi = r02.WOUND_BANDS[name]
        fraction = glauber.fraction_for_wound_range(lo, hi)
        assert fraction == pytest.approx(expected, rel=1e-12)
    # Minimum bias keeps the unconditioned draw (the M13 geometry anchor).
    assert r02.WOUND_BANDS["minimum_bias"] is None
    assert glauber.mean_target_wounds_minbias() == pytest.approx(
        3.937103537381027, rel=1e-12
    )


def test_r02_config_hash_verified(tmp_path):
    """A mutated config copy fails the driver before any generation."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / r02.PA_CONFIG_NAME
    # The pin is over the exact file bytes, so the control copy must be a
    # byte-identical copy of the bundled staged-excitation config (the
    # card's staged_excitation_pA.json resolves onto these bytes).
    shutil.copyfile(
        NESTED_ROOT / "config" / "staged_excitation.json", config_path
    )
    pristine = r02.verify_config_and_m8(config_path, r02.default_pre_registration())
    assert pristine["config_sha256"] == r02.PRODUCTION_CONFIG_SHA256

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
            "--target",
            "Pb208",
            "--centrality-classes",
            "minimum_bias",
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
    assert not (mutated_out / r02.bin_predictions_name("minimum_bias")).is_file()
    assert not (mutated_out / r02.failure_map_name("minimum_bias")).is_file()


def test_r02_seal_manifest(tmp_path):
    """The seal path writes SHA256SUMS.txt + provenance.json over every file."""
    out_dir = tmp_path / "seal_run"
    out_dir.mkdir()
    # Synthetic payload files (no generation, no tables).
    (out_dir / r02.PREREGISTRATION_NAME).write_text(
        json.dumps(r02.default_pre_registration(), indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / r02.bin_predictions_name("minimum_bias")).write_text(
        "[]\n", encoding="utf-8"
    )
    (out_dir / r02.failure_map_name("minimum_bias")).write_text(
        json.dumps({"n_total_bins": 1017}) + "\n", encoding="utf-8"
    )

    verification = r02.verify_config_and_m8(
        config_path=NESTED_ROOT / "config" / "staged_excitation.json",
        pre_registration=r02.default_pre_registration(),
    )
    r02.write_provenance(
        out_dir,
        verification=verification,
        pre_registration=r02.default_pre_registration(),
        seeds_per_class={"minimum_bias": [HARNESS_SEED]},
        replicas_per_class={"minimum_bias": 1},
        events_per_replica=10,
        events_completed=10,
        argv=["--out", str(out_dir)],
        wall_seconds=0.0,
        tier_status="deferred",
        tier_deferral_reason="harness seal check",
        input_paths=[Path(verification["config_path"]), Path(verification["manifest_path"])],
    )
    seal = r02.seal_run_dir(out_dir)

    assert (out_dir / r02.SHA256SUMS_NAME).is_file()
    assert (out_dir / r02.PROVENANCE_NAME).is_file()
    assert (out_dir / r02.SEAL_NAME).is_file()

    # The seal manifest covers every regular file in the directory except
    # itself, with correct digests.
    every_file = {
        p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*") if p.is_file()
    }
    sealed_names = {entry["path"] for entry in seal["files"]}
    assert sealed_names == every_file - {r02.SEAL_NAME}
    assert r02.SHA256SUMS_NAME in sealed_names
    assert r02.PROVENANCE_NAME in sealed_names
    for entry in seal["files"]:
        path = out_dir / entry["path"]
        assert entry["sha256"] == r02.sha256_file(path)
        assert entry["bytes"] == path.stat().st_size

    # SHA256SUMS.txt is in sha256sum -c format and verifies line by line.
    sums_text = (out_dir / r02.SHA256SUMS_NAME).read_text(encoding="utf-8")
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
        r02.SHA256SUMS_NAME,
        r02.SEAL_NAME,
        r02.PROVENANCE_NAME,
        r02.GATE_RECORD_NAME,
    }
    # The provenance record validates under the run_provenance convention.
    assert r02.validate_run_provenance(out_dir / r02.PROVENANCE_NAME) == []


def test_r02_empty_bins_recorded():
    """Empty MC bins land in empty_bins with the R01 Infinity convention."""
    # Five pi_minus bins: one flagged cell at pull 4.0, one empty MC bin
    # (N_bin == 0) with an infinite MC error, two model-only bins (no
    # matched data), and one ordinary matched bin at pull 0.
    m = [1.0, 0.0, 1.0, 1.0, 1.0]
    counts = [100, 0, 100, 100, 100]
    mc_err = r01.mc_error_array(np.asarray(m, dtype=float), counts)
    assert math.isclose(mc_err[0], 0.1)
    assert math.isinf(mc_err[1])
    residuals = [
        {
            "species": "pi_minus",
            "bin_index": 0,
            "d": 1.0,
            "m": 1.0,
            "sigma": 1.0,
            "delta_over_sigma": 0.0,
        },
        {
            "species": "pi_minus",
            "bin_index": 1,
            "d": 1.0,
            "m": 0.0,
            "sigma": float("inf"),
            "delta_over_sigma": 0.0,  # (d - m) / inf -> 0.0 by convention
        },
        {
            "species": "pi_minus",
            "bin_index": 2,
            "d": None,
            "m": 0.5,
            "sigma": None,
            "delta_over_sigma": None,  # model-only: no published p+Pb cell
        },
        {
            "species": "pi_minus",
            "bin_index": 3,
            "d": 2.0,
            "m": 1.0,
            "sigma": 0.25,
            "delta_over_sigma": 4.0,
        },
        {
            "species": "k_plus",
            "bin_index": 0,
            "d": None,
            "m": 0.25,
            "sigma": None,
            "delta_over_sigma": None,  # model-only species
        },
    ]
    # The literal-JSON-Infinity encoding: the driver's own writer keeps
    # the R01 convention (json.dumps with the default allow_nan).
    serialized = json.dumps({"stat_error": r01.mc_error_array([1.0], [0])[0]})
    assert "Infinity" in serialized

    failure_map = r02.build_failure_map(
        residuals, {"pi_minus": counts, "k_plus": [5]}, threshold=3.0
    )
    assert failure_map["n_total_bins"] == 1017
    assert failure_map["n_flagged"] == 1
    assert failure_map["n_empty_bins"] == 1
    assert failure_map["empty_bins"] == [
        {"species": "pi_minus", "bin_index": 1}
    ]
    assert failure_map["flagged_cells"][0]["rank"] == 1
    assert failure_map["flagged_cells"][0]["pull"] == pytest.approx(4.0)
    assert failure_map["model_only"] is False
    assert failure_map["data_species"] == ["pi_minus"]
    # The partial unit-test residual list covers only pi_minus and
    # k_plus; every other species is model-only in this hand-built map.
    assert failure_map["model_only_species"] == [
        "pi_plus",
        "k_plus",
        "k_minus",
        "proton",
        "antiproton",
    ]

    # A class whose species all lack published tables: model-only with
    # n_flagged 0 (no data-side pulls exist).
    residuals_model_only = [
        {
            "species": "k_plus",
            "bin_index": 0,
            "d": None,
            "m": 0.25,
            "sigma": None,
            "delta_over_sigma": None,
        }
    ]
    failure_model_only = r02.build_failure_map(
        residuals_model_only, {"k_plus": [5]}
    )
    assert failure_model_only["model_only"] is True
    assert failure_model_only["n_flagged"] == 0
    # The hand-built map carries only k_plus residuals; the other five
    # species are absent from the list and model-only by construction.
    assert failure_model_only["model_only_species"] == [
        "pi_minus",
        "pi_plus",
        "k_plus",
        "k_minus",
        "proton",
        "antiproton",
    ]

    chi2 = r02.chi2_payload(residuals)
    assert chi2["per_species"]["pi_minus"]["n_bins"] == 3
    assert chi2["per_species"]["pi_minus"]["n_empty_bins"] == 1
    assert chi2["per_species"]["pi_minus"]["dof"] == 2
    assert chi2["per_species"]["pi_minus"]["chi2"] == pytest.approx(16.0)
    assert chi2["per_species"]["k_plus"]["chi2"] is None
    assert chi2["per_species"]["k_plus"]["model_only"] is True

    # The strict tier predicate: complete only at exactly 6 x 8.
    assert r02.tier_status_decision(6, 8, 6, 8) == "complete"
    assert r02.tier_status_decision(6, 1, 6, 8) == "deferred"
    assert r02.tier_status_decision(6, 9, 6, 8) == "deferred"
    assert r02.tier_status_decision(5, 8, 6, 8) == "deferred"
