"""Hermetic small-N self-check suite for the R11 observable-decomposition
driver (Round 1, card 13,
``TASK-20260911-r1-run-r11-legacy-observable-decomposition``).

Two card-named tests:

- ``test_r11_driver_small_n``: the driver runs against a hand-constructed
  small R01-like archive (10 events, one seed, the six-species 1017-bin
  set from ``config/staged_excitation_smoke.json`` with
  ``fit.enabled: false``); the ``combinations.json`` carries all six
  pre-registered combinations, the integrals, the seagull moments, the
  multiplicity moments, the gap fraction, and the leading-baryon
  inelasticity; the ``ranked_residual_map.json`` has ``n_flagged >= 0``
  and rows sorted by ``|pull|`` descending; the per-replica digest
  self-check passes (the fixture's per-replica
  ``event_hash_set_sha256`` is the digest of the same seeded 10-event
  stream the driver re-runs, so the re-run verification accepts it).
- ``test_r11_no_fitting``: the driver and the ``fit_observables`` module
  import no optimizer (an AST source-inspection check over the committed
  sources), and the acceptance record's ``no_fitting`` field is ``true``.

Hermeticity note: the module-scoped fixture runs the SAME seeded
10-event builder call the driver's re-run will execute (the driver's
digest self-check can only accept the true digest of the tiny stream);
there is no other generation in the test, and the full R11 run (20 x
250,000 events on the sealed R01 archive) is the run itself, not a
test.  The smoke config needs no external tables; the binned-artifact
checks skip with a recorded reason when the SPS mirror is absent from
the checkout (the same convention as ``test_r01_reference_harness``).
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from helpers.benchmarking import staged_observable_decomposition as r11
from helpers.benchmarking import staged_r01_reference as r01
from na49_mixing.reaction import Reaction

NESTED_ROOT = Path(r01.NESTED_ROOT)
DRIVER_PATH = (
    NESTED_ROOT / "helpers" / "benchmarking" / "staged_observable_decomposition.py"
)
MODULE_PATH = NESTED_ROOT / "na49_mixing" / "fit_observables.py"
SMOKE_CONFIG = NESTED_ROOT / "config" / "staged_excitation_smoke.json"

# The canonical T0 replica seed: 20260911 + 40000 + 0 = 20300911.
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

pytestmark = pytest.mark.skipif(
    not _SPS_PRESENT,
    reason="the NA49/SPS table mirror is absent from this checkout",
)

OPTIMIZER_ROOTS = {
    "scipy",
    "iminuit",
    "lmfit",
    "cma",
    "nlopt",
    "optuna",
    "pymc",
    "emcee",
    "sklearn",
    "statsmodels",
}


def _smoke_payload() -> dict:
    raw_config = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    assert raw_config["fit"]["enabled"] is False
    return raw_config


def _harness_sqrt_s() -> float:
    return math.sqrt(Reaction(158.0, 2212, 2212).s())


@pytest.fixture(scope="module")
def r11_small_run(tmp_path_factory) -> dict:
    """Build the 10-event R01-like archive, then run the driver once."""
    raw_config = _smoke_payload()
    sqrt_s = _harness_sqrt_s()
    tables = r01.load_binned_tables(raw_config)
    tables_sha256 = r01.tables_digest(tables)

    # The seeded 10-event stream the driver will re-run (the fixture's
    # only generation; sub-second at 10 events).
    pp_config, _ = r01.build_pp_config(raw_config)
    hashes: list[str] = []
    sums: dict[str, list] = {}
    counts: dict[str, list] = {}
    for event_id in range(HARNESS_N_EVENTS):
        builder = r01.PPEventBuilder(
            pp_config, r01._event_rng(HARNESS_SEED, event_id)
        )
        event = builder.generate(event_id, stage=r01.STAGE_LABEL)
        hashes.append(event.as_dict()["event_hash"])
        chunk_sums, chunk_counts = r01._bin_track_sums(
            [event], tables, sqrt_s
        )
        for species, values in chunk_sums.items():
            sums[species] = (
                list(map(float, values))
                if species not in sums
                else list(map(float, np.asarray(sums[species]) + values))
            )
        for species, values in chunk_counts.items():
            counts[species] = (
                [int(v) for v in values]
                if species not in counts
                else [
                    int(a) + int(b)
                    for a, b in zip(counts[species], values)
                ]
            )
    digest = hashlib.sha256(
        json.dumps(hashes, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    sums_arrays = {s: np.asarray(v, dtype=float) for s, v in sums.items()}
    counts_arrays = {s: np.asarray(v, dtype=np.int64) for s, v in counts.items()}
    entries, _values, _errors = r01.bin_predictions_payload(
        tables, sums_arrays, counts_arrays, HARNESS_N_EVENTS, sqrt_s
    )
    sigma_eta = r01.read_sigma_eta()
    residuals = r01.residuals_payload(tables, _values, _errors)
    covariance = r01.covariance_payload(tables, _errors, sigma_eta)

    archive_dir = tmp_path_factory.mktemp("r01_like_archive")
    r01._write_json(
        archive_dir / r01.PREREGISTRATION_NAME, r01.default_pre_registration()
    )
    r01._write_json(
        archive_dir / r01.REPLICA_MANIFEST_NAME,
        {
            "seed_base": 20260911,
            "seed_offset": 40000,
            "requested_replicas": 1,
            "requested_events_per_replica": HARNESS_N_EVENTS,
            "tier_status": "deferred",
            "tier_deferral_reason": "harness fixture (10 events)",
            "seeds": [
                {
                    "seed": HARNESS_SEED,
                    "replica_index": 0,
                    "status": "completed",
                    "n_events": HARNESS_N_EVENTS,
                    "wall_seconds": 0.0,
                    "event_hash_set_sha256": digest,
                    "record": f"replica_{HARNESS_SEED}.json",
                }
            ],
            "config_sha256": r01.SMOKE_CONFIG_SHA256,
            "tables_sha256": tables_sha256,
        },
    )
    r01._write_json(
        archive_dir / f"replica_{HARNESS_SEED}.json",
        {
            "seed": HARNESS_SEED,
            "n_events": HARNESS_N_EVENTS,
            "chunk_size": r01.CHUNK_SIZE,
            "chunks": 1,
            "worker_count": 1,
            "event_seed_rule": r01.EVENT_SEED_RULE,
            "event_hash_set_sha256": digest,
            "config_sha256": r01.SMOKE_CONFIG_SHA256,
            "tables_sha256": tables_sha256,
            "wall_seconds": 0.0,
            "started_utc": "2026-09-12T00:00:00+00:00",
            "finished_utc": "2026-09-12T00:00:00+00:00",
            "sums": sums,
            "counts": counts,
            "reused": False,
        },
    )
    r01._write_json(
        archive_dir / r01.BIN_PREDICTIONS_NAME,
        entries,
    )
    r01._write_json(archive_dir / r01.COVARIANCE_NAME, covariance)
    r01._write_json(archive_dir / r01.RESIDUALS_NAME, residuals)
    r01._write_json(
        archive_dir / r01.FAILURE_MAP_NAME,
        r01.build_failure_map(residuals, counts_arrays),
    )
    r01._write_json(
        archive_dir / r01.CHI2_NAME, r01.chi2_payload(residuals, counts_arrays)
    )
    # The seal inventory over every payload file (the driver verifies it).
    skip = {r01.SHA256SUMS_NAME}
    lines = []
    for path in sorted(archive_dir.rglob("*")):
        if path.is_file() and path.name not in skip:
            rel = path.relative_to(archive_dir).as_posix()
            lines.append(f"{r01.sha256_file(path)}  {rel}\n")
    (archive_dir / r01.SHA256SUMS_NAME).write_text(
        "".join(lines), encoding="utf-8", newline="\n"
    )

    out_dir = tmp_path_factory.mktemp("r11_small_run")
    exit_code = r11.main(
        [
            "--r01-archive",
            str(archive_dir),
            "--out",
            str(out_dir),
            "--workers",
            "1",
            "--top-n",
            "50",
        ]
    )
    return {
        "exit_code": exit_code,
        "archive_dir": archive_dir,
        "out_dir": out_dir,
        "digest": digest,
    }


def test_r11_driver_small_n(r11_small_run):
    """The driver's small-N run writes every card-required artifact."""
    assert r11_small_run["exit_code"] == 0
    out_dir = r11_small_run["out_dir"]
    combinations = json.loads(
        (out_dir / r11.COMBINATIONS_NAME).read_text(encoding="utf-8")
    )
    # All six pre-registered combinations, with the card's names.
    assert set(combinations["combinations"]) == {
        "pi_sigma",
        "pi_delta",
        "k_pair",
        "k_associated",
        "b_pair",
        "b_transport",
    }
    for name, block in combinations["combinations"].items():
        assert block["n_cells"] > 0
        assert len(block["per_cell"]) == block["n_cells"]
        for cell in block["per_cell"]:
            assert cell["region"] in {
                "xF_central",
                "xF_transition",
                "xF_forward",
            }
    # The integrals per species, with the card's three regions.
    assert set(combinations["integrals"]) == {
        "pi_minus",
        "pi_plus",
        "k_plus",
        "k_minus",
        "proton",
        "antiproton",
    }
    for block in combinations["integrals"].values():
        assert set(block["model"]) == {"forward", "central", "backward"}
        assert block["model"]["backward"] == 0.0  # the p+p paper window
        assert block["data"]["backward"] == 0.0
        assert math.isfinite(block["model"]["forward"])
        assert block["model"]["central"] >= 0.0
    # The seagull moments carry the mandatory subscripts.
    for block in combinations["seagull_moments"].values():
        assert set(block["model"]) == {"pT_x_per_xF", "pT_y_per_xF"}
        assert set(block["data"]) == {"pT_x_per_xF", "pT_y_per_xF"}
        assert len(block["xF_mid_per_bin"]) == len(block["model"]["pT_x_per_xF"])
    # The multiplicity moments, gap fraction, and leading-baryon inelasticity.
    assert set(combinations["multiplicity_moments"]) == {
        "pi_minus",
        "pi_plus",
        "k_plus",
        "k_minus",
        "proton",
        "antiproton",
    }
    for moments in combinations["multiplicity_moments"].values():
        assert moments["n_mean"] >= 0.0
        assert moments["n2_mean"] >= moments["n_mean"] ** 2 - 1e-12
        assert moments["n_events_with_species"] > 0
    assert combinations["empty_species"] == []
    gap = combinations["rapidity_gap_fraction"]
    assert gap["threshold"] == 1.0
    assert 0.0 <= gap["fraction"] <= 1.0
    assert gap["n_events"] == HARNESS_N_EVENTS
    inelasticity = combinations["leading_baryon_inelasticity"]
    assert inelasticity["n_events_with_proton"] + inelasticity[
        "n_events_without_proton"
    ] == HARNESS_N_EVENTS
    assert inelasticity["n_events_with_proton"] > 0
    assert math.isfinite(inelasticity["mean"])
    assert inelasticity["rms"] >= inelasticity["std"] - 1e-12

    # The re-run verification block: the digest self-check passed.
    gate = json.loads((out_dir / r11.GATE_RECORD_NAME).read_text("utf-8"))
    rerun = gate["rerun_verification"]
    assert rerun["all_digests_match"] is True
    assert rerun["digest_matches"] == 1
    assert rerun["events_total"] == HARNESS_N_EVENTS
    assert rerun["pooled_prediction_agrees"] is True

    # The ranked residual map: n_flagged >= 0, rows sorted by |pull| desc.
    ranked = json.loads((out_dir / r11.RANKED_MAP_NAME).read_text("utf-8"))
    assert ranked["n_flagged"] >= 0
    assert ranked["n_total_cells"] > 0
    assert ranked["threshold"] == 3.0
    pulls = [abs(row["pull"]) for row in ranked["rows"]]
    assert pulls == sorted(pulls, reverse=True)
    assert [row["rank"] for row in ranked["rows"]] == list(
        range(1, len(ranked["rows"]) + 1)
    )


def test_r11_no_fitting(r11_small_run, tmp_path):
    """The driver imports no optimizer; the acceptance field is true."""
    # AST source inspection over the committed driver and module sources:
    # no imported module may root into an optimizer package.
    optimizer_roots = OPTIMIZER_ROOTS
    for source_path in (DRIVER_PATH, MODULE_PATH):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        for name in imported:
            root = name.split(".")[0]
            assert root not in optimizer_roots, (
                f"{source_path.name} imports {name!r}: an R11 driver must "
                "not import an optimizer (no fitting)"
            )
        source_text = source_path.read_text(encoding="utf-8")
        for keyword in ("minimize(", "least_squares(", "Minuit("):
            assert keyword not in source_text
    # The acceptance output of the small run attests no_fitting itself.
    out_dir = r11_small_run["out_dir"]
    gate = json.loads((out_dir / r11.GATE_RECORD_NAME).read_text("utf-8"))
    assert gate["no_fitting"] is True
    assert gate["verdict"] == "pass"
