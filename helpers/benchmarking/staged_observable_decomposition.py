#!/usr/bin/env python3
"""R11 legacy observable decomposition driver (Round 1, card 13).

Evaluates the sealed R01 p+p archive
(``dev/00runs/r01_legacy_pp_reference_2026-09-11/``, card 10) through the
physics-separated combinations of plan section 7.3 and writes a ranked
residual map.  No fitting: no optimizer is imported, no parameter is
changed, and no model is altered; the driver is a pure evaluation of the
sealed negative control.

Evaluated observables (plan section 7.3; the six pre-registered
combinations of :mod:`na49_mixing.fit_observables` plus the moment and
integral observables):

- the six combinations per bin: ``pi_sigma = f(pi-) + f(pi+)``,
  ``pi_delta = f(pi-) - f(pi+)``, ``k_pair = 2 f(K-)``,
  ``k_associated = f(K+) - f(K-)``, ``b_pair = 2 f(pbar)``,
  ``b_transport = f(p) - f(pbar)``;
- the forward / central / backward xF integrals per species (forward
  ``xF > 0.2``, central ``[0.0, 0.2]``, backward ``< 0.0``; the backward
  integral is 0 on the p+p paper window and is recorded for the p+A
  extension);
- the seagull moments per species, both ``<pT>`` conventions per xF bin
  (``pT_x_per_xF`` from the density ``(pT/E) f``, ``pT_y_per_xF`` from
  the density ``pT f``; the subscripts are mandatory);
- the multiplicity moments ``<n>`` and ``<n^2>`` per species;
- the rapidity-gap fraction at the pre-registered threshold
  ``dy_gap > 1.0``;
- the leading-baryon inelasticity ``1 - xF_leading_proton`` (mean and
  RMS);
- the ranked residual map: cells (sector x observable x region x bin)
  ranked by ``|pull| = |(d - m) / sigma|`` descending, with the combined
  data + MC uncertainty of card 7 (``V = V_data + V_mc``; the
  ``na49_mixing.fit_likelihood.combined_covariance`` rule applied to the
  independent species cells), threshold 3.0, top 100 rows.

Inter-card gap and resolution (documented method choice).  The card's
public interface names an event-level track NPZ from card 10 as the
source of the multiplicity moments, the rapidity-gap fraction, and the
leading-baryon inelasticity.  The sealed R01 archive carries no
event-level track file: card 10's driver wrote only binned artifacts
(``bin_predictions.json``, ``covariance.json``, ``residuals.json``,
``failure_map.json``, ``chi2.json``, per-replica manifests with
``event_hash_set_sha256`` digests, ``pre_registration.json``, seal
files, ``gate0_r01.json``).  The driver therefore:

1. verifies the sealed R01 archive first (every ``SHA256SUMS.txt`` entry
   must re-hash correctly; the inventory's own sha256 is recorded as
   ``r01_archive_sha256`` in ``gate1_r11.json``);
2. evaluates every binned quantity (the six combinations, the
   integrals, the seagull moments, the ranked residual map) from the
   sealed binned artifacts only;
3. obtains the event-level quantities from a DETERMINISTIC RE-RUN of
   the R01 seeded stream: the same ``PPEventBuilder``, the same frozen
   config (verified against the sealed replica records' config hash and
   the card-1 manifest through the R01 driver's own verification), the
   same per-replica seeds and event counts as the sealed archive,
   generated with the R01 driver's own generation logic (imported from
   ``staged_r01_reference.py``: the per-event seed rule,
   ``build_pp_config``, the R01 stage label, ``_bin_track_sums``, the
   chunk size) with one added event-level aggregation pass;
4. SELF-CHECK: per replica, the re-run's ``event_hash_set_sha256`` must
   equal the sealed replica record's digest.  Any mismatch fails the
   run (exit 1): the re-run is trusted only when it *is* the R01
   stream.  The per-replica binned weight sums and track counts are in
   addition compared against the sealed replica records, and the pooled
   re-run prediction against the sealed ``bin_predictions.json``.

The re-run runs replica by replica and writes an
``r11_replica_<seed>.json`` event-level summary as each replica
completes, so progress survives interruption; a re-invocation reuses
completed R11 replica records (re-verifying their digests against the
sealed archive) and continues.

Documented choices where the card is silent:

- Combination cells.  A two-species combination is evaluated on the
  intersection of the two species' tabulated ``(xF, pT)`` cells (the
  sealed tables are sparse and their cell sets differ, e.g. pi- 229
  bins vs pi+ 230); the count of unmatched cells is recorded per
  combination, never silently dropped.  A combination cell's
  ``bin_index`` is its position in the sorted common-cell list (the
  combination's own bin enumeration).  The combination values are
  produced by :func:`na49_mixing.fit_observables.evaluate_combination`
  on the aligned per-cell tuples.
- Combined uncertainty of a combination.  For a two-species sum or
  difference the independent species variances add (the card-7 rule
  applied pairwise on the diagonal Round-1 covariance, through
  ``fit_likelihood.combined_covariance``); a single-species ``2 f``
  combination scales the variance by 4.  A cell whose variance is
  infinite (an empty MC bin) keeps an infinite combined sigma and a
  pull of 0.0 (the R01 convention): recorded, never dropped.
- Sector naming.  ``f_xsect`` cells use the sealed archive's species
  key (``pi_minus`` ...); a combination's sector is its species keys
  joined with ``+`` (e.g. ``pi_minus+pi_plus``).
- Region naming.  Cells use card 8's ``SECTOR_REGIONS`` xF regions
  (``xF_central``, ``xF_transition``, ``xF_forward``), half-open
  ``[lo, hi)`` membership on the bin xF midpoints; the forward /
  central / backward integrals use the card's own coarser split at
  ``xF = 0.2`` (documented in :mod:`na49_mixing.fit_observables`).
- Event-level uncertainty.  The standard ``1 / sqrt(N_events)``
  convention (the existing ``sps_core.py`` convention); a species with
  ``N_events == 0`` gets an infinite uncertainty and is listed under
  ``empty_species`` (never silently dropped).
- dy_gap.  The largest difference between the sorted lab rapidities of
  the event's final-state particles (the event's rapidity ordering);
  threshold pre-registered at ``dy_gap > 1.0`` (see
  :data:`GAP_DEFINITION`).

Output files under ``--out``: ``pre_registration.json`` (written before
the re-run), one ``r11_replica_<seed>.json`` per replica,
``combinations.json``, ``ranked_residual_map.json``, and, written last
in the R01/R02/R10 seal order, ``provenance.json`` (the
``run_provenance.py`` convention), ``SHA256SUMS.txt`` (``sha256sum -c``
format over every payload file), ``gate1_r11.json`` (the acceptance
record; it carries the sums digest, so the sums inventory excludes it),
and ``seal_manifest.json`` (covers every other file; it cannot contain
its own digest).

Exit codes: 0 when every assertion holds (``gate1_r11.json`` verdict
``pass``), 2 when an archive/config verification refuses the run before
the re-run, 1 when a post-re-run assertion fails (a per-replica digest
mismatch, a pooled-prediction disagreement, a seal mismatch, or a
failing verdict).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import json
import math
import multiprocessing
import shutil
import sys
import time
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

# ---------------------------------------------------------------------------
# Tree location: this driver runs from the nested tree
# (dev/mixing2006/helpers/benchmarking/) and from its byte-identical
# monorepo mirror (helpers/benchmarking/).  Both must find the
# na49_mixing package, the R01 driver, and card 1's manifest.
# ---------------------------------------------------------------------------


def _locate_roots() -> tuple[Path, Path]:
    here = Path(__file__).resolve()
    nested = None
    for base in (here.parents[2], here.parents[2] / "dev" / "mixing2006"):
        if (base / "na49_mixing").is_dir():
            nested = base.resolve()
            break
    if nested is None:
        _refuse(
            "cannot locate the na49_mixing package next to "
            f"{here}; the driver runs from the nested tree "
            "(dev/mixing2006/helpers/benchmarking/) or its monorepo mirror "
            "(helpers/benchmarking/)"
        )
    mono = nested.parent.parent if nested.parent.name == "dev" else nested
    return nested, mono


def _refuse(message: str) -> NoReturn:
    """Print a refusal to stderr and exit with code 2 before the re-run."""
    print(
        f"staged_observable_decomposition: {message}", file=sys.stderr, flush=True
    )
    raise SystemExit(2)


NESTED_ROOT, MONO_ROOT = _locate_roots()
if str(NESTED_ROOT) not in sys.path:
    sys.path.insert(0, str(NESTED_ROOT))

from helpers.benchmarking import sps_core  # noqa: E402
from helpers.benchmarking.benchmark_sps_species import (  # noqa: E402
    RESONANCE_DECAY_LIB_VERSION,
)
from helpers.benchmarking import staged_r01_reference as r01  # noqa: E402
from helpers.benchmarking.run_provenance import (  # noqa: E402
    RUN_PROVENANCE_FILENAME,
    collect_source_record,
    validate_run_provenance,
    write_run_provenance,
)
from na49_mixing import fit_observables as fo  # noqa: E402
from na49_mixing.event_builder_pp import PPEventBuilder  # noqa: E402
from na49_mixing.fit_likelihood import (  # noqa: E402
    SECTOR_REGIONS,
    SECTOR_XF_REGION_ORDER,
    combined_covariance,
)
from na49_mixing.reaction import Reaction  # noqa: E402

# ---------------------------------------------------------------------------
# Pre-registered run identity (card TASK-20260911-r1-run-r11-legacy-
# observable-decomposition).
# ---------------------------------------------------------------------------

CARD = "TASK-20260911-r1-run-r11-legacy-observable-decomposition"
RUN_DESCRIPTION = "r11_observable_decomposition_2026-09-11"
GATE_RECORD_NAME = "gate1_r11.json"
GATE_CLAUSE = "physics_separated_observable_decomposition"

PREREGISTRATION_NAME = "pre_registration.json"
COMBINATIONS_NAME = "combinations.json"
RANKED_MAP_NAME = "ranked_residual_map.json"
R11_REPLICA_FMT = "r11_replica_{}.json"
SHA256SUMS_NAME = r01.SHA256SUMS_NAME
SEAL_NAME = r01.SEAL_NAME
PROVENANCE_NAME = r01.PROVENANCE_NAME

# The R01 artifact names the driver consumes (all read-only inputs).
R01_SUMS_NAME = r01.SHA256SUMS_NAME
R01_PREREGISTRATION_NAME = r01.PREREGISTRATION_NAME
R01_REPLICA_MANIFEST_NAME = r01.REPLICA_MANIFEST_NAME
R01_BIN_PREDICTIONS_NAME = r01.BIN_PREDICTIONS_NAME
R01_COVARIANCE_NAME = r01.COVARIANCE_NAME
R01_RESIDUALS_NAME = r01.RESIDUALS_NAME

# The R01 stage label is part of the event payload and therefore of the
# event hash: the re-run must reuse it bit for bit.
STAGE_LABEL = r01.STAGE_LABEL
CHUNK_SIZE = r01.CHUNK_SIZE
DEFAULT_WORKERS = r01.DEFAULT_WORKERS

DY_GAP_THRESHOLD = 1.0
RANKED_TOP_N = 100
RANKED_THRESHOLD = fo.RANKED_RESIDUAL_THRESHOLD
F_XSECT_OBSERVABLE = "f_xsect"
PREDICTION_AGREEMENT_TOL = 1e-9

GAP_DEFINITION = (
    "dy_gap is the largest difference between the sorted lab rapidities "
    "y = 0.5*ln((E + pz)/(E - pz)) of the event's final-state particles "
    "(the event's rapidity ordering); an event counts when dy_gap > 1.0. "
    "Events with fewer than two final particles cannot exceed the "
    "threshold. The sealed R01 archive carries no event-level track file "
    "and no source-side rapidity metadata field on an event track, so the "
    "rapidity gap is implemented from the event's rapidity ordering "
    "(documented method choice; threshold pre-registered at 1.0)."
)
INELASTICITY_DEFINITION = (
    "1 - xF_leading_proton per event, with xF_leading_proton = "
    "2 * pz / sqrt(s) of the most forward final-state proton (PDG 2212); "
    "the mean and the RMS are over the events that carry at least one "
    "final-state proton, and events without one are counted under "
    "n_events_without_proton (recorded, never silently dropped)."
)
EVENT_LEVEL_RESOLUTION = (
    "the sealed R01 archive carries no event-level track NPZ (card 10 "
    "wrote only binned artifacts); the event-level quantities come from "
    "the deterministic re-run of the R01 seeded stream (same "
    "PPEventBuilder, same frozen config, same replica seeds and event "
    "counts, R01 stage label), verified per replica by "
    "event_hash_set_sha256 equality with the sealed per-replica "
    "manifests; a digest mismatch fails the run"
)
VARIANCE_RULE = (
    "for a two-species sum or difference the independent species "
    "variances add (na49_mixing.fit_likelihood.combined_covariance, the "
    "card-7 V rule applied pairwise on the diagonal Round-1 covariance); "
    "a single-species 2*f combination scales the variance by 4"
)

# The two bundled legacy_v1 configs (pins from the R01 driver).
DISPLAY_TO_CONFIG = {
    display: config_key for config_key, display in r01.SPECIES_DISPLAY.items()
}


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds"
    )


def _write_json(path: Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def canonical_json_sha256(obj: Any) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _registration_signature(payload: dict) -> str:
    """Canonical form without the creation timestamp (incremental re-runs)."""
    stable = {k: v for k, v in payload.items() if k != "created_utc"}
    return canonical_json_sha256(stable)


# ---------------------------------------------------------------------------
# Sealed R01 archive verification
# ---------------------------------------------------------------------------


def verify_r01_archive(archive_dir: Path) -> dict:
    """Verify the sealed R01 archive before anything else runs.

    ``SHA256SUMS.txt`` must exist and every entry must re-hash to its
    file (the archive is the immutable input).  The pre-registration and
    the replica manifest must exist, and every completed replica must
    carry its per-replica record with the event-hash digest, event
    count, config hash, and table digest; all replicas must agree on
    the config and table digests.  Any problem refuses the run (exit 2)
    before the re-run starts.
    """
    archive_dir = Path(archive_dir)
    if not archive_dir.is_dir():
        _refuse(f"R01 archive not found: {archive_dir}")
    sums_path = archive_dir / R01_SUMS_NAME
    if not sums_path.is_file():
        _refuse(f"sealed R01 archive carries no {R01_SUMS_NAME}: {archive_dir}")
    mismatches: list[str] = []
    n_entries = 0
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        n_entries += 1
        payload = archive_dir / name
        if not payload.is_file() or r01.sha256_file(payload) != digest:
            mismatches.append(name)
    if mismatches:
        _refuse(
            "sealed R01 archive verification failed; "
            f"{len(mismatches)} of {n_entries} SHA256SUMS.txt entries do "
            f"not re-hash correctly (first: {mismatches[0]!r}); the archive "
            "is the read-only input and must be unmodified"
        )
    archive_seal_sha256 = r01.sha256_file(sums_path)

    prereg_path = archive_dir / R01_PREREGISTRATION_NAME
    if not prereg_path.is_file():
        _refuse(f"R01 pre-registration missing: {prereg_path}")
    archive_prereg = json.loads(prereg_path.read_text(encoding="utf-8"))

    manifest_path = archive_dir / R01_REPLICA_MANIFEST_NAME
    if not manifest_path.is_file():
        _refuse(f"R01 replica manifest missing: {manifest_path}")
    replica_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completed = [
        entry for entry in replica_manifest.get("seeds", [])
        if entry.get("status") == "completed"
    ]
    if not completed:
        _refuse(
            f"R01 replica manifest {manifest_path} lists no completed "
            "replica; nothing to evaluate"
        )
    replicas: dict[int, dict] = {}
    for entry in completed:
        seed = int(entry["seed"])
        record_path = archive_dir / f"replica_{seed}.json"
        if not record_path.is_file():
            _refuse(f"completed replica record missing: {record_path}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        for field in (
            "n_events",
            "event_hash_set_sha256",
            "config_sha256",
            "tables_sha256",
            "sums",
            "counts",
        ):
            if field not in record:
                _refuse(f"replica record {record_path} carries no {field}")
        replicas[seed] = record
    config_shas = {record["config_sha256"] for record in replicas.values()}
    tables_shas = {record["tables_sha256"] for record in replicas.values()}
    if len(config_shas) != 1 or len(tables_shas) != 1:
        _refuse(
            "sealed R01 replicas disagree on the config or table digest "
            f"(config {sorted(config_shas)!r}, tables {sorted(tables_shas)!r})"
        )
    npz_tracks = sorted(p.name for p in archive_dir.glob("*.npz"))
    return {
        "archive_path": archive_dir,
        "archive_seal_sha256": archive_seal_sha256,
        "sums_entries": n_entries,
        "sums_mismatches": mismatches,
        "archive_prereg": archive_prereg,
        "seeds": sorted(replicas),
        "replicas": replicas,
        "config_sha256": config_shas.pop(),
        "tables_sha256": tables_shas.pop(),
        "events_per_replica": sorted(
            {int(record["n_events"]) for record in replicas.values()}
        ),
        "sealed_track_npz": npz_tracks,
    }


def resolve_bundled_config(config_sha256: str) -> Path:
    """Map a replica record's config hash to the bundled legacy_v1 config."""
    for sha, name in (
        (r01.PRODUCTION_CONFIG_SHA256, r01.PRODUCTION_CONFIG_NAME),
        (r01.SMOKE_CONFIG_SHA256, r01.SMOKE_CONFIG_NAME),
    ):
        if config_sha256 == sha:
            return NESTED_ROOT / "config" / name
    _refuse(
        f"sealed R01 config sha256 {config_sha256!r} matches neither "
        "bundled legacy_v1 config; R11 evaluates only R01 archives built "
        "from the two bundled configs"
    )


# ---------------------------------------------------------------------------
# R11 pre-registration (written before the re-run)
# ---------------------------------------------------------------------------


def r11_pre_registration_payload(
    verification: dict, seeds: list[int]
) -> dict:
    """The pre-registered R11 payload, written before the re-run."""
    return {
        "card": CARD,
        "run": RUN_DESCRIPTION,
        "r01_archive_path": str(verification["archive_path"]),
        "r01_archive_sha256": verification["archive_seal_sha256"],
        "r01_sums_entries": int(verification["sums_entries"]),
        "r01_pre_registration_config_hash": verification["archive_prereg"].get(
            "config_hash"
        ),
        "r01_m8_parameter_hash": verification["archive_prereg"].get(
            "m8_parameter_hash"
        ),
        "config_sha256": verification["config_sha256"],
        "tables_sha256": verification["tables_sha256"],
        "replica_seeds": [int(s) for s in seeds],
        "events_per_replica": sorted(
            {int(verification["replicas"][s]["n_events"]) for s in seeds}
        ),
        "dy_gap_threshold": DY_GAP_THRESHOLD,
        "dy_gap_definition": GAP_DEFINITION,
        "leading_baryon_inelasticity_definition": INELASTICITY_DEFINITION,
        "event_level_resolution": EVENT_LEVEL_RESOLUTION,
        "sealed_track_npz": list(verification["sealed_track_npz"]),
        "combinations": list(fo.COMBINATION_NAMES),
        "ranked_residual_map": {
            "threshold": RANKED_THRESHOLD,
            "top_n": RANKED_TOP_N,
            "sector_region_naming": (
                "card 8's SECTOR_REGIONS xF regions (xF_central, "
                "xF_transition, xF_forward), half-open [lo, hi) membership "
                "on the bin xF midpoints"
            ),
        },
        "no_fitting": True,
    }


def write_r11_pre_registration(out_dir: Path, payload: dict) -> None:
    """Write the R11 pre-registration before the re-run (refuse a mismatch).

    An existing registration must equal the payload up to its creation
    timestamp, so an interrupted run can resume; anything else refuses.
    """
    prereg_path = Path(out_dir) / PREREGISTRATION_NAME
    if prereg_path.is_file():
        loaded = json.loads(prereg_path.read_text(encoding="utf-8"))
        if _registration_signature(loaded) != _registration_signature(payload):
            _refuse(
                f"pre-registration mismatch: {prereg_path} does not equal "
                "the R11 registration (R01 archive identity, seeds, event "
                "counts, dy_gap threshold, combination set, and ranked-map "
                "settings are pre-registered; a different registration "
                "files its own card)"
            )
        return
    stamped = dict(payload)
    stamped["created_utc"] = _utc_now_iso()
    _write_json(prereg_path, stamped)


# ---------------------------------------------------------------------------
# Deterministic re-run of the R01 seeded stream (event-level aggregation)
# ---------------------------------------------------------------------------


def _event_aggregates_zero(species_pdgs: dict[str, int]) -> dict:
    return {
        "multiplicity": {species: [0, 0, 0] for species in species_pdgs},
        "gap": {"n_events": 0, "n_with_gap": 0, "sum_max_gap": 0.0},
        "leading_proton": {
            "n_with_proton": 0,
            "n_without_proton": 0,
            "sum_inelasticity": 0.0,
            "sum_inelasticity2": 0.0,
        },
    }


def _merge_chunk_aggregates(target: dict, chunk: dict) -> None:
    for species in target["multiplicity"]:
        for i in range(3):
            target["multiplicity"][species][i] += chunk["multiplicity"][
                species
            ][i]
    for key in ("n_events", "n_with_gap"):
        target["gap"][key] += chunk["gap"][key]
    target["gap"]["sum_max_gap"] += chunk["gap"]["sum_max_gap"]
    for key in ("n_with_proton", "n_without_proton"):
        target["leading_proton"][key] += chunk["leading_proton"][key]
    for key in ("sum_inelasticity", "sum_inelasticity2"):
        target["leading_proton"][key] += chunk["leading_proton"][key]


def _r11_generate_chunk(task: tuple):
    """Generate one R01 chunk and aggregate the event-level observables.

    The generation is the R01 driver's own logic (``staged_r01_reference``
    build/rng/stage/``_bin_track_sums``), so the per-replica event-hash
    digest reproduces the sealed archive bit for bit.  The added
    event-level pass reads only the already-generated event: per-species
    multiplicities, the rapidity ordering (the dy_gap definition), and
    the leading proton.  Chunk-local float sums accumulate in event
    order; the parent combines chunks in event order, so the result is
    independent of the worker count.
    """
    seed, event_ids, payload, tables, sqrt_s_gev, species_pdgs = task
    proton_pdg = species_pdgs["proton"]
    pp_config, _ = r01.build_pp_config(payload)
    hashes: list[str] = []
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    aggregates = _event_aggregates_zero(species_pdgs)
    mult_by_pdg = {pdg: 0 for pdg in species_pdgs.values()}
    for event_id in event_ids:
        builder = PPEventBuilder(pp_config, r01._event_rng(seed, event_id))
        event = builder.generate(int(event_id), stage=STAGE_LABEL)
        hashes.append(event.as_dict()["event_hash"])

        for pdg in mult_by_pdg:
            mult_by_pdg[pdg] = 0
        lead_pz = None
        rapidities: list[float] = []
        for particle in event.final_particles:
            pdg = int(particle.pdg_id)
            if pdg in mult_by_pdg:
                mult_by_pdg[pdg] += 1
            p4 = particle.p4
            if pdg == proton_pdg and (lead_pz is None or p4.pz > lead_pz):
                lead_pz = p4.pz
            e = float(p4.e)
            pz = float(p4.pz)
            rapidities.append(0.5 * math.log((e + pz) / (e - pz)))
        for species, pdg in species_pdgs.items():
            n = mult_by_pdg[pdg]
            slot = aggregates["multiplicity"][species]
            slot[0] += n
            slot[1] += n * n
            if n > 0:
                slot[2] += 1
        if len(rapidities) >= 2:
            rapidities.sort()
            dy_gap = max(b - a for a, b in zip(rapidities, rapidities[1:]))
            aggregates["gap"]["n_events"] += 1
            aggregates["gap"]["sum_max_gap"] += dy_gap
            if dy_gap > DY_GAP_THRESHOLD:
                aggregates["gap"]["n_with_gap"] += 1
        if lead_pz is not None:
            inelasticity = 1.0 - 2.0 * float(lead_pz) / sqrt_s_gev
            aggregates["leading_proton"]["n_with_proton"] += 1
            aggregates["leading_proton"]["sum_inelasticity"] += inelasticity
            aggregates["leading_proton"]["sum_inelasticity2"] += (
                inelasticity * inelasticity
            )
        else:
            aggregates["leading_proton"]["n_without_proton"] += 1

        chunk_sums, chunk_counts = r01._bin_track_sums(
            [event], tables, sqrt_s_gev
        )
        for species, values in chunk_sums.items():
            sums[species] = (
                values.astype(float, copy=True)
                if species not in sums
                else sums[species] + values
            )
        for species, values in chunk_counts.items():
            counts[species] = (
                values.astype(np.int64, copy=True)
                if species not in counts
                else counts[species] + values
            )
    return (
        int(event_ids[0]),
        hashes,
        aggregates,
        sums,
        counts,
        len(event_ids),
    )


def _map_chunks(tasks: list, workers: int, progress_label: str):
    """Yield worker results in completion order (fork pool, or inline)."""
    if int(workers) <= 1 or len(tasks) == 1:
        completed = 0
        for task in tasks:
            completed += 1
            yield _r11_generate_chunk(task)
            print(
                f"[r11] {progress_label}: chunk {completed}/{len(tasks)}",
                flush=True,
            )
        return
    context = multiprocessing.get_context("fork")
    completed = 0
    with context.Pool(processes=int(workers)) as pool:
        for result in pool.imap_unordered(
            _r11_generate_chunk, tasks, chunksize=1
        ):
            completed += 1
            yield result
            if completed % 5 == 0 or completed == len(tasks):
                print(
                    f"[r11] {progress_label}: chunk {completed}/{len(tasks)}",
                    flush=True,
                )


def run_r11_replica(
    seed: int,
    n_events: int,
    payload: dict,
    tables: list,
    sqrt_s_gev: float,
    workers: int,
    out_dir: Path,
    config_sha256: str,
    tables_sha256: str,
    sealed_record: dict,
    species_pdgs: dict[str, int],
) -> dict:
    """One deterministic re-run replica with its event-level aggregates.

    An existing ``r11_replica_<seed>.json`` that matches the seed, event
    count, config hash, and table digest is reused unchanged after
    re-verifying its digest against the sealed record; anything else is
    regenerated.  The digest equality against the sealed
    ``event_hash_set_sha256`` is the run's core verification.
    """
    out_dir = Path(out_dir)
    record_path = out_dir / f"r11_replica_{int(seed)}.json"
    sealed_digest = str(sealed_record["event_hash_set_sha256"])
    if record_path.is_file():
        try:
            stored = json.loads(record_path.read_text(encoding="utf-8"))
        except ValueError:
            stored = None
        if (
            isinstance(stored, dict)
            and int(stored.get("seed", -1)) == int(seed)
            and int(stored.get("n_events", -1)) == int(n_events)
            and stored.get("config_sha256") == config_sha256
            and stored.get("tables_sha256") == tables_sha256
            and isinstance(stored.get("aggregates"), dict)
            and stored.get("event_hash_set_sha256") is not None
        ):
            stored["reused"] = True
            print(
                f"[r11] replica {seed}: reusing completed event-level "
                f"record ({int(n_events)} events)",
                flush=True,
            )
            return stored

    event_ids = list(range(int(n_events)))
    chunks = [
        event_ids[start : start + CHUNK_SIZE]
        for start in range(0, len(event_ids), CHUNK_SIZE)
    ]
    tasks = [
        (int(seed), chunk, payload, tables, sqrt_s_gev, species_pdgs)
        for chunk in chunks
    ]
    started = time.perf_counter()
    parts = sorted(
        _map_chunks(tasks, int(workers), f"replica {seed}"),
        key=lambda part: part[0],
    )
    hashes = [h for _, chunk_hashes, _, _, _, _ in parts for h in chunk_hashes]
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    aggregates = _event_aggregates_zero(species_pdgs)
    for _, _, part_aggregates, part_sums, part_counts, _ in parts:
        _merge_chunk_aggregates(aggregates, part_aggregates)
        for species, values in part_sums.items():
            values = np.asarray(values, dtype=float)
            sums[species] = (
                values if species not in sums else sums[species] + values
            )
        for species, values in part_counts.items():
            values = np.asarray(values, dtype=np.int64)
            counts[species] = (
                values if species not in counts else counts[species] + values
            )
    wall = time.perf_counter() - started
    payload_digest = json.dumps(hashes, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(payload_digest).hexdigest()
    sums_match = {
        species: [float(v) for v in sums[species]]
        == [float(v) for v in sealed_record["sums"][species]]
        for species in sums
    }
    counts_match = {
        species: [int(v) for v in counts[species]]
        == [int(v) for v in sealed_record["counts"][species]]
        for species in counts
    }
    record = {
        "seed": int(seed),
        "n_events": int(n_events),
        "chunk_size": CHUNK_SIZE,
        "chunks": len(chunks),
        "worker_count": int(workers),
        "event_seed_rule": r01.EVENT_SEED_RULE,
        "stage": STAGE_LABEL,
        "event_hash_set_sha256": digest,
        "sealed_event_hash_set_sha256": sealed_digest,
        "digest_match": digest == sealed_digest,
        "sums_exact_match": all(sums_match.values()),
        "counts_exact_match": all(counts_match.values()),
        "config_sha256": config_sha256,
        "tables_sha256": tables_sha256,
        "wall_seconds": round(wall, 3),
        "aggregates": aggregates,
        "sums": {
            s: [float(v) for v in values] for s, values in sums.items()
        },
        "counts": {
            s: [int(v) for v in values] for s, values in counts.items()
        },
        "reused": False,
    }
    _write_json(record_path, record)
    return record


# ---------------------------------------------------------------------------
# Sealed binned artifacts (per-species arrays)
# ---------------------------------------------------------------------------


def parse_sealed_binned(archive_dir: Path, tables: list) -> dict:
    """Read ``bin_predictions`` / ``covariance`` / ``residuals`` into arrays.

    Returns per-species arrays in each species' sealed bin order:
    ``d`` (the SPS table value in f/sigma_inel units), ``m`` (the pooled
    model prediction), ``v_data`` / ``v_mc`` / ``v_combined`` (the
    card-7 variances), ``sigma`` and ``delta_over_sigma`` (the sealed
    residuals), and the per-bin ``(xF_mid, pT_mid)`` midpoints.  The
    sealed ``d`` must equal the loaded table data bit for bit (the same
    conversion) and the sealed ``sigma`` must match ``sqrt(V_combined)``
    to rounding; a deviation refuses the run.
    """
    predictions = json.loads(
        (Path(archive_dir) / R01_BIN_PREDICTIONS_NAME).read_text(
            encoding="utf-8"
        )
    )
    covariance = json.loads(
        (Path(archive_dir) / R01_COVARIANCE_NAME).read_text(encoding="utf-8")
    )
    residuals = json.loads(
        (Path(archive_dir) / R01_RESIDUALS_NAME).read_text(encoding="utf-8")
    )
    sealed: dict[str, dict[str, list[float]]] = {}
    for table in tables:
        sealed[table.species] = {
            "m": [],
            "v_data": [],
            "v_mc": [],
            "v_combined": [],
            "d": [],
            "sigma": [],
            "delta": [],
            "xF_mid": [],
            "pT_mid": [],
        }
    for entry in predictions:
        slot = sealed[entry["species"]]
        slot["m"].append(float(entry["value"]))
        slot["xF_mid"].append(float(entry["xF_mid"]))
        slot["pT_mid"].append(float(entry["pT_mid"]))
    for entry in covariance["V_data"]:
        sealed[entry["species"]]["v_data"].append(float(entry["variance"]))
    for entry in covariance["V_mc"]:
        sealed[entry["species"]]["v_mc"].append(float(entry["variance"]))
    for entry in covariance["V_combined"]:
        sealed[entry["species"]]["v_combined"].append(float(entry["variance"]))
    for entry in residuals:
        sealed[entry["species"]]["d"].append(float(entry["d"]))
        sealed[entry["species"]]["sigma"].append(float(entry["sigma"]))
        sealed[entry["species"]]["delta"].append(
            float(entry["delta_over_sigma"])
        )
    for table in tables:
        n_bins = int(table.data.size)
        record = sealed[table.species]
        for field, values in sealed[table.species].items():
            if len(values) != n_bins:
                _refuse(
                    f"sealed {table.species} {field} carries "
                    f"{len(values)} entries for {n_bins} bins"
                )
        for i in range(n_bins):
            if sealed[table.species]["d"][i] != float(table.data[i]):
                _refuse(
                    f"sealed {table.species} residual d[{i}] differs from "
                    "the loaded SPS table value; the sealed archive and "
                    "the local tables disagree"
                )
            sealed_sigma = sealed[table.species]["sigma"][i]
            variance = sealed[table.species]["v_combined"][i]
            if math.isinf(sealed_sigma) or math.isinf(variance):
                if not (math.isinf(sealed_sigma) and math.isinf(variance)):
                    _refuse(
                        f"sealed {table.species} bin {i}: infinite sigma and "
                        "variance disagree"
                    )
            else:
                rebuilt = math.sqrt(max(variance, 0.0))
                if not math.isclose(
                    rebuilt, sealed_sigma, rel_tol=1e-12, abs_tol=0.0
                ):
                    _refuse(
                        f"sealed {table.species} bin {i}: sqrt(V_combined) "
                        f"{rebuilt!r} != sealed sigma {sealed_sigma!r}"
                    )
    return sealed


def xf_sector_region(xf_mid: float) -> str:
    """The card-8 xF region of a bin midpoint (``ValueError`` outside)."""
    last = len(SECTOR_XF_REGION_ORDER) - 1
    for position, name in enumerate(SECTOR_XF_REGION_ORDER):
        lo, hi = SECTOR_REGIONS[name]
        if lo <= xf_mid < hi:
            return name
        if position == last and xf_mid == hi:
            return name
    raise ValueError(
        f"xF midpoint {xf_mid!r} lies outside the SECTOR_REGIONS xF grid"
    )


def compare_pooled_prediction(
    tables: list, rerun_values: dict[str, np.ndarray], sealed: dict
) -> dict:
    """The re-run's pooled prediction against the sealed ``bin_predictions``.

    The re-run is only trusted when it *is* the R01 stream; the digest
    equality is the binding check, and this cross-check binds the pooled
    binned content.  The comparison is over every sealed bin: relative
    deviation for non-zero sealed values, absolute for zero bins.
    """
    max_rel = 0.0
    max_abs = 0.0
    n_compared = 0
    worst_key = None
    for table in tables:
        rerun = np.asarray(rerun_values[table.species], dtype=float)
        model = np.asarray(sealed[table.species]["m"], dtype=float)
        for i, (mine, theirs) in enumerate(zip(rerun, model)):
            n_compared += 1
            if theirs != 0.0:
                rel = abs(mine - theirs) / abs(theirs)
                if rel > max_rel:
                    max_rel = rel
                    worst_key = (table.species, i, rel)
            else:
                abs_diff = abs(mine - theirs)
                if abs_diff > max_abs:
                    max_abs = abs_diff
                    worst_key = (table.species, i, abs_diff)
    return {
        "n_compared": n_compared,
        "max_rel_diff": max_rel,
        "max_abs_diff_zero_bins": max_abs,
        "worst": (
            None
            if worst_key is None
            else {
                "species": worst_key[0],
                "bin_index": int(worst_key[1]),
                "value": float(worst_key[2]),
            }
        ),
        "agrees": max_rel <= PREDICTION_AGREEMENT_TOL,
    }


# ---------------------------------------------------------------------------
# Combination / integral / moment builders (pure functions over the
# sealed arrays and the re-run pooled prediction; unit-tested)
# ---------------------------------------------------------------------------


def aligned_species_cells(
    sealed: dict, species: str
) -> tuple[list[tuple[float, float]], dict[tuple[float, float], int]]:
    """The species' tabulated ``(xF, pT)`` cells in sealed bin order."""
    coords = [
        (float(x), float(p))
        for x, p in zip(
            sealed[species]["xF_mid"], sealed[species]["pT_mid"]
        )
    ]
    index = {coord: i for i, coord in enumerate(coords)}
    return coords, index


def _finite_tuple(values, what: str) -> list[float]:
    """Narrow a combination result to floats (sum/difference are total).

    The six pre-registered combinations are sums and differences, whose
    element-wise results are defined on every aligned bin; a ``None``
    entry (the ratio guard) would mean a broken input and refuses the
    run rather than silently dropping the bin.
    """
    out = []
    for i, value in enumerate(values):
        if value is None:
            _refuse(
                f"{what} bin {i} is undefined (None); the pre-registered "
                "sum/difference combinations are total on aligned cells"
            )
        out.append(float(value))
    return out


def build_combination_cells(
    combo: fo.ObservableCombination,
    sealed: dict,
    rerun_values: dict[str, np.ndarray],
    tables_by_species: dict[str, Any],
    d_cells: dict,
    m_cells: dict,
    v_cells: dict,
) -> dict:
    """One combination's cells into the residual maps and a summary dict.

    Two-species combinations are evaluated on the intersection of the
    two species' tabulated cells (sorted by (xF, pT)); the combination's
    combined variance is the card-7 pairwise sum of the two independent
    species variances (``combined_covariance``); a single-species ``2 f``
    combination scales the variance by 4.  The ``m`` cell value is the
    sealed model prediction; the re-run pooled prediction is compared
    against it per combination cell as verification evidence.
    """
    config_species = [DISPLAY_TO_CONFIG[name] for name in combo.species]
    for name in combo.species:
        if name not in DISPLAY_TO_CONFIG:
            _refuse(
                f"combination {combo.name!r} names unknown species {name!r}"
            )
    per_cell: list[dict] = []
    if len(combo.species) == 1:
        sp = config_species[0]
        display = combo.species[0]
        coords, _ = aligned_species_cells(sealed, sp)
        d_combo = _finite_tuple(
            fo.evaluate_combination(
                combo, {display: tuple(sealed[sp]["d"])}
            ),
            f"{combo.name} data",
        )
        m_combo = _finite_tuple(
            fo.evaluate_combination(
                combo, {display: tuple(sealed[sp]["m"])}
            ),
            f"{combo.name} model",
        )
        v_combo = [4.0 * v for v in sealed[sp]["v_combined"]]
        unmatched = 0
        for k, (d_val, m_val, v_val) in enumerate(
            zip(d_combo, m_combo, v_combo)
        ):
            xF, pT = coords[k]
            region = xf_sector_region(xF)
            key = fo.residual_cell_key(sp, combo.name, region, k)
            d_cells[key] = (float(d_val),)
            m_cells[key] = (float(m_val),)
            v_cells[key] = (float(v_val),)
            sigma = math.sqrt(v_val) if math.isfinite(v_val) else math.inf
            pull = 0.0 if math.isinf(sigma) else (d_val - m_val) / sigma
            per_cell.append(
                {
                    "bin_index": k,
                    "xF_mid": xF,
                    "pT_mid": pT,
                    "region": region,
                    "d": float(d_val),
                    "m": float(m_val),
                    "variance": float(v_val),
                    "sigma": sigma,
                    "delta_over_sigma": float(pull),
                }
            )
        rerun_combo = _finite_tuple(
            fo.evaluate_combination(
                combo,
                {
                    display: tuple(
                        float(v) for v in rerun_values[sp]
                    )
                },
            ),
            f"{combo.name} rerun",
        )
        rerun_max_rel = max(
            (
                abs(float(a) - float(b)) / abs(float(b))
                for a, b in zip(rerun_combo, m_combo)
                if float(b) != 0.0
            ),
            default=0.0,
        )
        return {
            "name": combo.name,
            "operator": combo.operator,
            "species": list(combo.species),
            "species_keys": [
                DISPLAY_TO_CONFIG[name] for name in combo.species
            ],
            "definition": combo.definition,
            "variance_rule": VARIANCE_RULE,
            "n_cells": len(per_cell),
            "n_unmatched_cells": unmatched,
            "per_cell": per_cell,
            "rerun_max_rel_diff": rerun_max_rel,
        }
    sp_a, sp_b = config_species[0], config_species[1]
    coords_a, index_a = aligned_species_cells(sealed, sp_a)
    coords_b, index_b = aligned_species_cells(sealed, sp_b)
    common = sorted(set(index_a) & set(index_b))
    n_unmatched = (len(index_a) - len(common)) + (len(index_b) - len(common))
    d_a = [sealed[sp_a]["d"][index_a[c]] for c in common]
    d_b = [sealed[sp_b]["d"][index_b[c]] for c in common]
    m_a = [sealed[sp_a]["m"][index_a[c]] for c in common]
    m_b = [sealed[sp_b]["m"][index_b[c]] for c in common]
    v_a = [sealed[sp_a]["v_combined"][index_a[c]] for c in common]
    v_b = [sealed[sp_b]["v_combined"][index_b[c]] for c in common]
    d_combo = _finite_tuple(
        fo.evaluate_combination(
            combo,
            {
                combo.species[0]: tuple(d_a),
                combo.species[1]: tuple(d_b),
            },
        ),
        f"{combo.name} data",
    )
    m_combo = _finite_tuple(
        fo.evaluate_combination(
            combo,
            {
                combo.species[0]: tuple(m_a),
                combo.species[1]: tuple(m_b),
            },
        ),
        f"{combo.name} model",
    )
    rerun_values_combo = fo.evaluate_combination(
        combo,
        {
            combo.species[0]: tuple(
                float(rerun_values[sp_a][index_a[c]]) for c in common
            ),
            combo.species[1]: tuple(
                float(rerun_values[sp_b][index_b[c]]) for c in common
            ),
        },
    )
    rerun_aligned = [
        _finite_tuple((rerun_values_combo[k],), f"{combo.name} rerun")[0]
        for k in range(len(common))
    ]
    rerun_max_rel = 0.0
    for k, (xF, pT) in enumerate(common):
        v_val = float(
            combined_covariance((v_a[k],), (v_b[k],))[0]
        )
        d_val = float(d_combo[k])
        m_val = float(m_combo[k])
        sigma = math.sqrt(v_val) if math.isfinite(v_val) else math.inf
        pull = 0.0 if math.isinf(sigma) else (d_val - m_val) / sigma
        region = xf_sector_region(xF)
        # The combination cell's sector joins the two species keys; the
        # bin_index is the combination's own sorted common-cell position.
        key = fo.residual_cell_key(f"{sp_a}+{sp_b}", combo.name, region, k)
        d_cells[key] = (d_val,)
        m_cells[key] = (m_val,)
        v_cells[key] = (v_val,)
        per_cell.append(
            {
                "bin_index": k,
                "xF_mid": xF,
                "pT_mid": pT,
                "region": region,
                "d": d_val,
                "m": m_val,
                "variance": v_val,
                "sigma": sigma,
                "delta_over_sigma": float(pull),
            }
        )
        if m_val != 0.0:
            rel = abs(float(rerun_aligned[k]) - m_val) / abs(m_val)
            rerun_max_rel = max(rerun_max_rel, rel)
    return {
        "name": combo.name,
        "operator": combo.operator,
        "species": list(combo.species),
        "species_keys": [sp_a, sp_b],
        "definition": combo.definition,
        "variance_rule": VARIANCE_RULE,
        "n_cells": len(per_cell),
        "n_unmatched_cells": n_unmatched,
        "n_unmatched_cells_a": len(index_a) - len(common),
        "n_unmatched_cells_b": len(index_b) - len(common),
        "per_cell": per_cell,
        "rerun_max_rel_diff": rerun_max_rel,
    }


# ---------------------------------------------------------------------------
# Integrals, seagull moments, event-level quantities
# ---------------------------------------------------------------------------


def species_bin_edges_and_indices(
    table: Any,
) -> tuple[tuple[tuple[float, ...], tuple[float, ...]], list[tuple[int, int]]]:
    """The nested ``(xF_edges, pT_edges)`` pair and per-bin ``(ix, ip)``.

    The edges are the midpoint-edge grid of the species' windowed table
    (``sps_core.edges_from_centers``), and ``bin_indices[i]`` is bin
    ``i``'s ``(ix, ip)`` position on that grid.
    """
    x_edges = sps_core.edges_from_centers(table.xf_centers)
    p_edges = sps_core.edges_from_centers(table.pt_centers)
    indices = [
        (int(ix), int(ip))
        for ix, ip in zip(table.bin_index_xf, table.bin_index_pt)
    ]
    return (
        tuple(float(v) for v in x_edges),
        tuple(float(v) for v in p_edges),
    ), indices


def build_integrals(
    table: Any, sealed: dict, rerun_values: dict[str, np.ndarray]
) -> dict:
    """Forward / central / backward integrals of one species (model + data).

    The model integrals use the sealed pooled prediction (the archive is
    the evaluated object) and the data integrals the sealed SPS table
    values; both go through
    :func:`na49_mixing.fit_observables.forward_central_backward_integral`
    with the species' own bin grid.
    """
    sp = table.species
    edges, indices = species_bin_edges_and_indices(table)
    model = fo.forward_central_backward_integral(
        {sp: tuple(float(v) for v in sealed[sp]["m"])},
        edges,
        sp,
        bin_indices=indices,
    )
    data = fo.forward_central_backward_integral(
        {sp: tuple(float(v) for v in sealed[sp]["d"])},
        edges,
        sp,
        bin_indices=indices,
    )
    rerun = fo.forward_central_backward_integral(
        {sp: tuple(float(v) for v in rerun_values[sp])},
        edges,
        sp,
        bin_indices=indices,
    )
    return {
        "species": sp,
        "n_bins": int(table.data.size),
        "model": model,
        "data": data,
        "rerun": rerun,
        "note": (
            "midpoint-rectangle sum over the bin (xF, pT) cells of the "
            "species' windowed table; the backward region is empty on the "
            "p+p paper window xF in [0, 0.4] and its integral is 0 "
            "(documented; the region exists for the p+A extension)"
        ),
    }


def build_seagull(
    table: Any, sealed: dict, rerun_values: dict[str, np.ndarray], sqrt_s: float
) -> dict:
    """The two ``<pT>`` conventions per xF bin of one species (model + data).

    The xF midpoints of the species' xF bins are recorded next to the
    moments; a ``None`` entry marks a bin whose density weight vanished
    (recorded, never silently dropped).
    """
    sp = table.species
    edges, indices = species_bin_edges_and_indices(table)
    x_edges, _p_edges = edges
    x_mids = sorted(
        {
            0.5 * (x_edges[int(ix)] + x_edges[int(ix) + 1])
            for ix, _ip in indices
        }
    )
    model = fo.seagull_moment(
        {sp: tuple(float(v) for v in sealed[sp]["m"])},
        edges,
        sp,
        bin_indices=indices,
        sqrt_s_gev=sqrt_s,
        mass_gev=float(table.mass_gev),
    )
    data = fo.seagull_moment(
        {sp: tuple(float(v) for v in sealed[sp]["d"])},
        edges,
        sp,
        bin_indices=indices,
        sqrt_s_gev=sqrt_s,
        mass_gev=float(table.mass_gev),
    )
    return {
        "species": sp,
        "species_display": r01.SPECIES_DISPLAY[sp],
        "mass_gev": float(table.mass_gev),
        "xF_mid_per_bin": x_mids,
        "model": model,
        "data": data,
        "n_undefined_model": sum(
            1 for v in model["pT_x_per_xF"] + model["pT_y_per_xF"]
            if v is None
        ),
        "n_undefined_data": sum(
            1 for v in data["pT_x_per_xF"] + data["pT_y_per_xF"] if v is None
        ),
    }


def pool_event_level(replica_records: list, species_order: list[str]) -> dict:
    """Pool the per-replica event-level aggregates into the R11 quantities.

    Multiplicity moments ``<n>`` and ``<n^2>`` per species, the
    rapidity-gap fraction at the pre-registered threshold, and the
    leading-baryon inelasticity (mean and RMS).  Statistical
    uncertainties follow the standard ``1 / sqrt(N_events)`` convention;
    standard errors are recorded additionally.  A species observed in no
    event gets an infinite uncertainty and is listed under
    ``empty_species``.
    """
    pooled = _event_aggregates_zero(
        {species: 0 for species in species_order}
    )
    pooled_species_pdgs: dict[str, int] = {}
    for record in sorted(replica_records, key=lambda r: int(r["seed"])):
        _merge_chunk_aggregates(pooled, record["aggregates"])
    n_total = sum(int(r["n_events"]) for r in replica_records)
    if n_total <= 0:
        _refuse("the re-run completed zero events; nothing to evaluate")
    multiplicity: dict[str, dict] = {}
    empty_species: list[str] = []
    for species in species_order:
        sum_n, sum_n2, n_with_species = pooled["multiplicity"][species]
        n_mean = sum_n / n_total
        n2_mean = sum_n2 / n_total
        empty = n_with_species == 0
        if empty:
            empty_species.append(species)
        multiplicity[species] = {
            "n_mean": n_mean,
            "n2_mean": n2_mean,
            "stat_uncertainty": (
                float("inf") if empty else 1.0 / math.sqrt(n_total)
            ),
            "n_events_with_species": int(n_with_species),
            "n_events_total": int(n_total),
            "empty": empty,
        }
    gap = pooled["gap"]
    fraction = gap["n_with_gap"] / n_total
    gap_fraction = {
        "threshold": DY_GAP_THRESHOLD,
        "definition": GAP_DEFINITION,
        "fraction": fraction,
        "n_events_with_gap": int(gap["n_with_gap"]),
        "n_events_with_two_or_more_particles": int(gap["n_events"]),
        "n_events": int(n_total),
        "stat_uncertainty": 1.0 / math.sqrt(n_total),
        "binomial_std": math.sqrt(
            fraction * max(1.0 - fraction, 0.0) / n_total
        ),
        "mean_max_gap": gap["sum_max_gap"] / max(int(gap["n_events"]), 1),
    }
    proton_block = pooled["leading_proton"]
    n_with = int(proton_block["n_with_proton"])
    n_without = int(proton_block["n_without_proton"])
    if n_with > 0:
        mean = proton_block["sum_inelasticity"] / n_with
        rms = math.sqrt(max(proton_block["sum_inelasticity2"] / n_with, 0.0))
        variance = max(
            proton_block["sum_inelasticity2"] / n_with - mean * mean, 0.0
        )
        std = math.sqrt(variance)
    else:
        mean = float("nan")
        rms = float("nan")
        std = float("nan")
    inelasticity = {
        "definition": INELASTICITY_DEFINITION,
        "mean": mean,
        "rms": rms,
        "std": std,
        "n_events_with_proton": n_with,
        "n_events_without_proton": n_without,
        "fraction_without_proton": n_without / n_total,
        "stat_uncertainty": (
            float("inf") if n_with == 0 else 1.0 / math.sqrt(n_total)
        ),
        "standard_error_of_mean": (
            float("inf") if n_with == 0 else std / math.sqrt(n_with)
        ),
    }
    return {
        "n_events_total": int(n_total),
        "multiplicity": multiplicity,
        "empty_species": empty_species,
        "rapidity_gap_fraction": gap_fraction,
        "leading_baryon_inelasticity": inelasticity,
    }


# ---------------------------------------------------------------------------
# Ranked residual map assembly and the f_xsect cross-check
# ---------------------------------------------------------------------------


def build_f_xsect_cells(
    tables: list, sealed: dict, d_cells: dict, m_cells: dict, v_cells: dict
) -> None:
    """The raw per-species ``f_xsect`` cells into the residual maps."""
    for table in tables:
        sp = table.species
        for i in range(int(table.data.size)):
            xF = float(sealed[sp]["xF_mid"][i])
            region = xf_sector_region(xF)
            key = fo.residual_cell_key(sp, F_XSECT_OBSERVABLE, region, i)
            d_cells[key] = (float(sealed[sp]["d"][i]),)
            m_cells[key] = (float(sealed[sp]["m"][i]),)
            v_cells[key] = (float(sealed[sp]["v_combined"][i]),)


def cross_check_f_xsect_pulls(
    d_cells: dict, m_cells: dict, v_cells: dict, sealed: dict
) -> dict:
    """Every ``f_xsect`` cell against the sealed residual convention.

    For every ``f_xsect`` cell the pull must equal the sealed
    ``residuals.json`` ``delta_over_sigma`` (same ``d - m`` over the
    combined sigma; the sqrt-then-square round-trip differs only in the
    last ulps).  Infinite-sigma cells carry pull 0.0 in both conventions
    and are compared as such.
    """
    worst = 0.0
    n_compared = 0
    for key, d_vals in d_cells.items():
        sector, observable, _region, bin_index = fo.parse_residual_cell_key(
            key
        )
        if observable != F_XSECT_OBSERVABLE:
            continue
        sp = sector
        i = int(bin_index)
        sealed_pull = sealed[sp]["delta"][i]
        sealed_sigma = sealed[sp]["sigma"][i]
        d_val = float(d_vals[0])
        m_val = float(m_cells[key][0])
        variance = float(v_cells[key][0])
        if math.isfinite(variance):
            mine = (d_val - m_val) / math.sqrt(variance)
        else:
            mine = 0.0
        if math.isinf(sealed_sigma) or math.isinf(sealed_pull):
            if mine != 0.0:
                _refuse(
                    f"f_xsect cross-check failed for {sp} bin {i}: infinite "
                    f"sealed sigma must give pull 0.0, got {mine!r}"
                )
            n_compared += 1
            continue
        diff = abs(mine - sealed_pull)
        scale = max(abs(sealed_pull), 1.0)
        worst = max(worst, diff / scale)
        n_compared += 1
    return {
        "n_compared": n_compared,
        "max_rel_diff": worst,
        "agrees": worst <= PREDICTION_AGREEMENT_TOL,
    }


def assemble_ranked_map(
    tables: list,
    sealed: dict,
    rerun_values: dict[str, np.ndarray],
    d_cells: dict,
    m_cells: dict,
    v_cells: dict,
    top_n: int,
) -> tuple[dict, list[dict], dict]:
    """The f_xsect cells, the six combination cells, and the ranked map."""
    build_f_xsect_cells(tables, sealed, d_cells, m_cells, v_cells)
    combination_blocks: list[dict] = []
    for combo in fo.OBSERVABLE_COMBINATIONS:
        block = build_combination_cells(
            combo,
            sealed,
            rerun_values,
            {t.species: t for t in tables},
            d_cells,
            m_cells,
            v_cells,
        )
        combination_blocks.append(block)
    ranked = fo.ranked_residual_map(
        d_cells, m_cells, v_cells, top_n=int(top_n), threshold=RANKED_THRESHOLD
    )
    pull_check = cross_check_f_xsect_pulls(d_cells, m_cells, v_cells, sealed)
    return ranked, combination_blocks, pull_check


# ---------------------------------------------------------------------------
# Code fingerprint, provenance, seal (the R01/R02/R10 convention)
# ---------------------------------------------------------------------------


def r11_code_fingerprint() -> str:
    """sha256 over the staged-generation sources, the salt, and both drivers.

    The same tagged-bytes recipe as the R01 driver's fingerprint, with
    this driver's source in place of the R01 driver's (the R11 record
    covers the generation logic it actually re-runs plus itself).
    """
    chunks: list[tuple[str, bytes]] = []
    for modname in r01.FINGERPRINT_MODULES:
        mod = importlib.import_module(modname)
        origin = getattr(mod, "__file__", None)
        if not origin:
            raise RuntimeError(f"cannot resolve source file of {modname}")
        with open(origin, "rb") as handle:
            chunks.append((f"source:{modname}", handle.read()))
    chunks.append(
        (
            "salt:RESONANCE_DECAY_LIB_VERSION",
            RESONANCE_DECAY_LIB_VERSION.encode("utf-8"),
        )
    )
    chunks.append(
        ("r01_driver", Path(r01.__file__).resolve().read_bytes())
    )
    chunks.append(("r11_driver", Path(__file__).resolve().read_bytes()))
    digest = hashlib.sha256()
    for tag, payload in chunks:
        tag_b = tag.encode("utf-8")
        digest.update(len(tag_b).to_bytes(8, "big"))
        digest.update(tag_b)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def write_provenance_r11(
    out_dir: Path,
    verification: dict,
    seeds: list[int],
    n_events_total: int,
    rerun_wall_seconds: float,
    argv: list[str],
) -> dict:
    """Write ``provenance.json`` via the ``run_provenance.py`` convention."""
    record = write_run_provenance(
        Path(out_dir),
        input_paths=[
            Path(out_dir) / PREREGISTRATION_NAME,
            Path(verification["archive_path"]) / R01_SUMS_NAME,
            Path(verification["archive_path"]) / R01_PREREGISTRATION_NAME,
        ],
        argv=[str(a) for a in argv],
        env_var_names=["NA49_RUN_SOURCE_ARCHIVE"],
        rng={
            "type": "deterministic re-run of the R01 seeded stream: "
            + r01.EVENT_SEED_RULE,
            "seeds": {
                "r01_seed_base": int(
                    verification["archive_prereg"]["seed_base"]
                ),
                "r01_seed_offset": int(
                    verification["archive_prereg"]["seed_offset"]
                ),
                "rerun_replica_seeds": [int(s) for s in seeds],
            },
        },
        source_record=collect_source_record(
            outer_repo=MONO_ROOT, nested_repo=NESTED_ROOT
        ),
        producer="staged_observable_decomposition.py",
        extra={
            "card": CARD,
            "run": RUN_DESCRIPTION,
            "gate_clause": GATE_CLAUSE,
            "no_fitting": True,
            "r01_archive_path": str(verification["archive_path"]),
            "r01_archive_sha256": verification["archive_seal_sha256"],
            "r01_sums_entries": int(verification["sums_entries"]),
            "r01_sealed_track_npz": list(verification["sealed_track_npz"]),
            "config_sha256": verification["config_sha256"],
            "tables_sha256": verification["tables_sha256"],
            "dy_gap_threshold": DY_GAP_THRESHOLD,
            "combinations": list(fo.COMBINATION_NAMES),
            "events_total": int(n_events_total),
            "event_level_resolution": EVENT_LEVEL_RESOLUTION,
            "code_fingerprint": {
                "resonance_decay_lib_version": RESONANCE_DECAY_LIB_VERSION,
                "r11_code_sha256": r11_code_fingerprint(),
                "fingerprint_modules": list(r01.FINGERPRINT_MODULES),
            },
            "rerun_wall_seconds": round(float(rerun_wall_seconds), 3),
        },
    )
    problems = validate_run_provenance(Path(out_dir) / RUN_PROVENANCE_FILENAME)
    if problems:
        _refuse(
            "written provenance record failed validation:\n  "
            + "\n  ".join(problems)
        )
    written = Path(out_dir) / RUN_PROVENANCE_FILENAME
    target = Path(out_dir) / PROVENANCE_NAME
    if target.exists():
        target.unlink()
    shutil.move(str(written), str(target))
    return record


def write_sha256sums(out_dir: Path) -> Path:
    """``SHA256SUMS.txt`` over every payload file (the R01 convention).

    Excludes the files that cannot carry their own digests: the
    inventory itself, ``seal_manifest.json``, ``provenance.json``, and
    ``gate1_r11.json`` (it carries the inventory's own digest).
    """
    out_dir = Path(out_dir)
    skip = {SHA256SUMS_NAME, SEAL_NAME, PROVENANCE_NAME, GATE_RECORD_NAME}
    lines = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.name not in skip:
            rel = path.relative_to(out_dir).as_posix()
            lines.append(f"{r01.sha256_file(path)}  {rel}\n")
    sums_path = out_dir / SHA256SUMS_NAME
    sums_path.write_text("".join(lines), encoding="utf-8", newline="\n")
    return sums_path


def write_seal_manifest(out_dir: Path) -> dict:
    """``seal_manifest.json`` over every other file plus the sums."""
    out_dir = Path(out_dir)
    entries = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.name != SEAL_NAME:
            entries.append(
                {
                    "path": path.relative_to(out_dir).as_posix(),
                    "sha256": r01.sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    seal = {
        "sealed_at_utc": _utc_now_iso(),
        "run_directory": out_dir.name,
        "card": CARD,
        "format": (
            "SHA256SUMS.txt covers every payload file in sha256sum -c "
            "format (sorted relative paths), excluding the files that "
            "cannot carry their own digests (SHA256SUMS.txt itself, "
            "seal_manifest.json, provenance.json, gate1_r11.json); "
            "seal_manifest.json records the payload files plus "
            "SHA256SUMS.txt, provenance.json, and gate1_r11.json; "
            "seal_manifest.json cannot contain its own digest"
        ),
        "files": entries,
    }
    _write_json(out_dir / SEAL_NAME, seal)
    return seal


# ---------------------------------------------------------------------------
# Acceptance record
# ---------------------------------------------------------------------------


def assemble_gate1_r11(
    out_dir: Path,
    verification: dict,
    ranked: dict,
    combinations_blocks: list[dict],
    integrals_blocks: list[dict],
    seagull_blocks: list[dict],
    event_level: dict,
    pull_check: dict,
    prediction_check: dict,
    rerun_records: list[dict],
    top_n: int,
    n_species: int,
    rerun_wall_seconds: float,
) -> dict:
    """Assemble ``gate1_r11.json`` from the run's artifacts.

    The record matches the card's acceptance schema; the additive
    ``rerun_verification`` block carries the card-13 gap resolution (the
    per-replica digest self-check and the pooled-prediction agreement).
    The verdict is ``pass`` only when every assertion holds.
    """
    out_dir = Path(out_dir)
    sums_path = out_dir / SHA256SUMS_NAME
    if not sums_path.is_file():
        _refuse(f"seal inventory missing: {sums_path}")
    r11_seal_sha256 = r01.sha256_file(sums_path)

    names_written = {block["name"] for block in combinations_blocks}
    combinations_ok = names_written == set(fo.COMBINATION_NAMES)
    integrals_ok = (
        all(
            set(block["model"]) == {"forward", "central", "backward"}
            and set(block["data"]) == {"forward", "central", "backward"}
            for block in integrals_blocks
        )
        and len(integrals_blocks) == int(n_species)
    )
    seagull_ok = all(
        set(block["model"]) == {"pT_x_per_xF", "pT_y_per_xF"}
        and set(block["data"]) == {"pT_x_per_xF", "pT_y_per_xF"}
        for block in seagull_blocks
    )
    event_level_ok = (
        int(event_level["n_events_total"]) > 0
        and bool(event_level["rapidity_gap_fraction"])
        and bool(event_level["leading_baryon_inelasticity"])
        and bool(event_level["multiplicity"])
    )
    digests_ok = all(bool(record["digest_match"]) for record in rerun_records)
    gate = {
        "gate": "gate1",
        "clause": GATE_CLAUSE,
        "r01_archive_sha256": verification["archive_seal_sha256"],
        "no_fitting": True,
        "combinations_evaluated": sorted(names_written),
        "integrals_evaluated": ["forward", "central", "backward"],
        "seagull_moments_evaluated": ["pT_x_per_xF", "pT_y_per_xF"],
        "multiplicity_moments_evaluated": ["<n>", "<n^2>"],
        "gap_fraction_evaluated": True,
        "leading_baryon_inelasticity_evaluated": True,
        "ranked_residual_map": {
            "top_n": int(top_n),
            "threshold": float(ranked["threshold"]),
            "n_flagged": int(ranked["n_flagged"]),
            "n_total_cells": int(ranked["n_total_cells"]),
        },
        "empty_species": len(event_level["empty_species"]),
        "verdict": "pass",
        "rerun_verification": {
            "r01_archive": str(verification["archive_path"]),
            "r01_seal_verified": True,
            "r01_sums_entries": int(verification["sums_entries"]),
            "r01_sealed_track_npz": list(verification["sealed_track_npz"]),
            "event_level_resolution": EVENT_LEVEL_RESOLUTION,
            "replica_seeds": [int(r["seed"]) for r in rerun_records],
            "events_per_replica": sorted(
                {int(r["n_events"]) for r in rerun_records}
            ),
            "events_total": int(
                sum(int(r["n_events"]) for r in rerun_records)
            ),
            "replicas_rerun": sum(
                1 for r in rerun_records if not r.get("reused", False)
            ),
            "replicas_reused": sum(
                1 for r in rerun_records if r.get("reused", False)
            ),
            "digest_matches": sum(
                1 for r in rerun_records if r["digest_match"]
            ),
            "digest_mismatches": sum(
                1 for r in rerun_records if not r["digest_match"]
            ),
            "all_digests_match": bool(digests_ok),
            "per_replica_sums_exact_match": all(
                bool(r["sums_exact_match"]) for r in rerun_records
            ),
            "config_sha256": verification["config_sha256"],
            "tables_sha256": verification["tables_sha256"],
            "pooled_prediction_max_rel_diff_vs_sealed": prediction_check[
                "max_rel_diff"
            ],
            "pooled_prediction_agrees": bool(prediction_check["agrees"]),
            "f_xsect_pulls_vs_sealed_residuals": {
                "n_compared": pull_check["n_compared"],
                "max_rel_diff": pull_check["max_rel_diff"],
                "agrees": bool(pull_check["agrees"]),
            },
            "dy_gap_threshold": DY_GAP_THRESHOLD,
            "rerun_wall_seconds": round(float(rerun_wall_seconds), 3),
            "stage": STAGE_LABEL,
        },
        "r11_seal_sha256": r11_seal_sha256,
    }
    checks_pass = (
        combinations_ok
        and integrals_ok
        and seagull_ok
        and event_level_ok
        and digests_ok
        and bool(prediction_check["agrees"])
        and bool(pull_check["agrees"])
        and int(ranked["n_total_cells"]) > 0
    )
    gate["combinations_complete"] = bool(combinations_ok)
    gate["integrals_complete"] = bool(integrals_ok)
    gate["seagull_complete"] = bool(seagull_ok)
    gate["event_level_complete"] = bool(event_level_ok)
    gate["verdict"] = "pass" if checks_pass else "fail"
    _write_json(out_dir / GATE_RECORD_NAME, gate)
    return gate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "R11 legacy observable decomposition driver: evaluate the "
            "sealed R01 p+p archive through the physics-separated "
            "combinations of plan section 7.3 and build a ranked residual "
            "map; no fitting (the event-level quantities come from the "
            "deterministic re-run of the R01 seeded stream, verified "
            "per replica by event-hash digest equality)"
        )
    )
    parser.add_argument(
        "--r01-archive",
        required=True,
        help="path to the sealed R01 run directory (read-only input)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="R11 run directory (dev/00runs/... resolves against the "
        "monorepo root)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="worker processes for the per-replica re-run (default 4)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=RANKED_TOP_N,
        help=f"rows returned by the ranked residual map (default "
        f"{RANKED_TOP_N})",
    )
    return parser


def main(argv: list | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if int(args.workers) <= 0:
        _refuse(
            f"--workers must be a positive integer, got {args.workers}"
        )
    if int(args.top_n) < 0:
        _refuse(f"--top-n must be non-negative, got {args.top_n}")
    out_dir = r01.resolve_out_dir(args.out)
    archive_dir = r01.resolve_out_dir(args.r01_archive)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify the sealed R01 archive (read-only input) before anything.
    verification = verify_r01_archive(archive_dir)
    print(
        f"[r11] R01 archive {verification['archive_path']} verified: "
        f"{verification['sums_entries']} SHA256SUMS.txt entries re-hash; "
        f"seal sha256 {verification['archive_seal_sha256']}",
        flush=True,
    )
    print(
        f"[r11] sealed track files found: "
        f"{verification['sealed_track_npz'] or 'none (documented gap)'}",
        flush=True,
    )

    # 2. Resolve and verify the frozen config from the replica records.
    config_path = resolve_bundled_config(verification["config_sha256"])
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    config_verification = r01.verify_config_and_m8(
        config_path, verification["archive_prereg"]
    )
    r01.build_pp_config(raw_config)  # refuse on a bad application, pre-run
    print(
        f"[r11] config {config_path} sha256 "
        f"{config_verification['config_sha256']}; M8 hash "
        f"{config_verification['m8_parameter_hash']} verified",
        flush=True,
    )

    # 3. The observable set and sqrt(s), pinned to the archive's digest.
    tables = r01.load_binned_tables(raw_config)
    tables_sha256 = r01.tables_digest(tables)
    if tables_sha256 != verification["tables_sha256"]:
        _refuse(
            "the local SPS tables hash to "
            f"{tables_sha256!r} but the sealed R01 replica records pin "
            f"{verification['tables_sha256']!r}; R11 evaluates the archive "
            "under its own observable set"
        )
    sqrt_s = math.sqrt(
        Reaction(float(raw_config.get("beam_momentum", 158.0)), 2212, 2212).s()
    )
    species_order = [t.species for t in tables]
    species_pdgs = {t.species: int(t.pdg_id) for t in tables}
    if "proton" not in species_pdgs:
        _refuse(
            "the sealed archive's tables carry no 'proton' species; the "
            "leading-baryon inelasticity needs the final-state proton"
        )
    print(
        f"[r11] observable set: {sum(int(t.data.size) for t in tables)} bins "
        f"over {len(tables)} species; sqrt(s) {sqrt_s:.6f} GeV; "
        f"dy_gap threshold {DY_GAP_THRESHOLD}",
        flush=True,
    )

    # 4. R11 pre-registration, written BEFORE the re-run.
    seeds = verification["seeds"]
    payload = r11_pre_registration_payload(verification, seeds)
    write_r11_pre_registration(out_dir, payload)
    print(
        f"[r11] pre-registration written: {len(seeds)} seeds "
        f"({payload['events_per_replica']} events each), "
        f"{len(payload['combinations'])} combinations",
        flush=True,
    )

    # 5. The deterministic re-run, replica by replica; per-replica
    #    event-level summaries land as each replica completes.
    payload_config = r01.config_payload(raw_config)
    rerun_records: list[dict] = []
    rerun_started = time.perf_counter()
    for index, seed in enumerate(seeds, start=1):
        n_events = int(verification["replicas"][seed]["n_events"])
        print(
            f"[r11] replica {seed} ({index}/{len(seeds)}), "
            f"{n_events} events",
            flush=True,
        )
        record = run_r11_replica(
            int(seed),
            int(n_events),
            payload_config,
            tables,
            sqrt_s,
            int(args.workers),
            out_dir,
            config_verification["config_sha256"],
            verification["tables_sha256"],
            verification["replicas"][int(seed)],
            species_pdgs,
        )
        if not record["digest_match"]:
            print(
                f"[r11] DIGEST MISMATCH for replica {seed}: re-run "
                f"{record['event_hash_set_sha256']} != sealed "
                f"{record['sealed_event_hash_set_sha256']}; the re-run is "
                "not the R01 stream and the run fails",
                flush=True,
            )
            return 1
        print(
            f"[r11] replica {seed} ({index}/{len(seeds)}): digest match, "
            f"sums_exact_match {record['sums_exact_match']}, "
            f"wall {record['wall_seconds']:.1f}s",
            flush=True,
        )
        rerun_records.append(record)
    rerun_wall = time.perf_counter() - rerun_started
    events_total = sum(int(r["n_events"]) for r in rerun_records)
    print(
        f"[r11] re-run complete: {len(rerun_records)}/{len(seeds)} replicas, "
        f"{events_total} events, all digests match "
        f"({rerun_wall:.1f}s wall)",
        flush=True,
    )

    # 6. The pooled re-run prediction against the sealed bin_predictions.
    totals: dict[str, np.ndarray] = {}
    count_totals: dict[str, np.ndarray] = {}
    for record in rerun_records:  # already in ascending-seed order
        for species, values in record["sums"].items():
            values = np.asarray(values, dtype=float)
            totals[species] = (
                values if species not in totals else totals[species] + values
            )
        for species, values in record["counts"].items():
            values = np.asarray(values, dtype=np.int64)
            count_totals[species] = (
                values
                if species not in count_totals
                else count_totals[species] + values
            )
    _, rerun_values, _ = r01.bin_predictions_payload(
        tables, totals, count_totals, events_total, sqrt_s
    )
    sealed = parse_sealed_binned(archive_dir, tables)
    prediction_check = compare_pooled_prediction(tables, rerun_values, sealed)
    print(
        f"[r11] pooled prediction vs sealed bin_predictions: max rel diff "
        f"{prediction_check['max_rel_diff']:.3e} over "
        f"{prediction_check['n_compared']} bins",
        flush=True,
    )
    if not prediction_check["agrees"]:
        print(
            "[r11] POOLED PREDICTION DISAGREEMENT; the re-run does not "
            "reproduce the sealed binned content and the run fails",
            flush=True,
        )
        return 1

    # 7. The physics-separated combinations, integrals, seagull moments,
    #    and event-level quantities; then the ranked residual map.
    d_cells: dict[str, tuple[float, ...]] = {}
    m_cells: dict[str, tuple[float, ...]] = {}
    v_cells: dict[str, tuple[float, ...]] = {}
    ranked, combination_blocks, pull_check = assemble_ranked_map(
        tables,
        sealed,
        rerun_values,
        d_cells,
        m_cells,
        v_cells,
        int(args.top_n),
    )
    if not pull_check["agrees"]:
        print(
            f"[r11] F_XSECT PULL CROSS-CHECK FAILED "
            f"(max rel diff {pull_check['max_rel_diff']:.3e}); the map's "
            "raw-spectra pulls must reproduce the sealed residuals",
            flush=True,
        )
        return 1
    integrals_blocks = [
        build_integrals(table, sealed, rerun_values) for table in tables
    ]
    seagull_blocks = [
        build_seagull(table, sealed, rerun_values, sqrt_s) for table in tables
    ]
    event_level = pool_event_level(rerun_records, species_order)
    combinations_payload = {
        "run": RUN_DESCRIPTION,
        "card": CARD,
        "r01_archive_path": str(verification["archive_path"]),
        "r01_archive_sha256": verification["archive_seal_sha256"],
        "config_sha256": config_verification["config_sha256"],
        "m8_parameter_hash": config_verification["m8_parameter_hash"],
        "tables_sha256": verification["tables_sha256"],
        "sqrt_s_gev": sqrt_s,
        "replica_seeds": [int(s) for s in seeds],
        "events_total": int(events_total),
        "dy_gap_threshold": DY_GAP_THRESHOLD,
        "no_fitting": True,
        "combinations": {
            block["name"]: block for block in combination_blocks
        },
        "integrals": {block["species"]: block for block in integrals_blocks},
        "seagull_moments": {
            block["species"]: block for block in seagull_blocks
        },
        "multiplicity_moments": event_level["multiplicity"],
        "empty_species": event_level["empty_species"],
        "rapidity_gap_fraction": event_level["rapidity_gap_fraction"],
        "leading_baryon_inelasticity": event_level[
            "leading_baryon_inelasticity"
        ],
        "conventions": {
            "combinations": (
                "plan section 7.3; element-wise operations on the aligned "
                "per-cell values (two-species combinations on the "
                "intersection of the sealed tables' tabulated (xF, pT) "
                "cells; unmatched cells counted, never dropped)"
            ),
            "variance_rule": VARIANCE_RULE,
            "seagull": (
                "<pT>_x from the density (pT/E) f, <pT>_y from the density "
                "pT f (Fischer 2022 equations 48 and 49), per xF bin; the "
                "subscripts are mandatory"
            ),
            "event_level_uncertainty": (
                "standard 1 / sqrt(N_events) convention; a species with "
                "N_events == 0 carries an infinite uncertainty and is "
                "listed under empty_species"
            ),
            "event_level_resolution": EVENT_LEVEL_RESOLUTION,
        },
        "rerun_wall_seconds": round(rerun_wall, 3),
    }
    _write_json(out_dir / COMBINATIONS_NAME, combinations_payload)
    ranked_payload = dict(ranked)
    ranked_payload["convention"] = (
        "cells (sector x observable x region x bin) ranked by |pull| = "
        "|(d - m) / sigma| descending; sigma = sqrt(V) is the combined "
        "data + MC uncertainty of card 7 (V = V_data + V_mc, diagonal in "
        "Round 1; a combination's variance follows the card-7 pairwise "
        "rule); region naming per card 8's SECTOR_REGIONS xF split; a "
        "cell with infinite V (an empty MC bin) carries an infinite "
        "sigma and a pull of 0.0 (recorded, never dropped); f_xsect "
        "sectors use the sealed archive's species keys, combination "
        "sectors join them with '+'"
    )
    _write_json(out_dir / RANKED_MAP_NAME, ranked_payload)
    print(
        f"[r11] ranked residual map: {ranked['n_total_cells']} cells, "
        f"{ranked['n_flagged']} flagged (|pull| > {RANKED_THRESHOLD}), "
        f"top {len(ranked['rows'])} rows written",
        flush=True,
    )
    print(
        "[r11] top-5 flagged cells: "
        + "; ".join(
            f"{row['sector']}|{row['observable']}|{row['region']}|"
            f"{row['bin_index']} pull {row['pull']:.3f}"
            for row in ranked["rows"][:5]
        ),
        flush=True,
    )

    # 8. Provenance, seal inventory, acceptance record, seal manifest
    #    (the R01/R02/R10 order; gate1 carries the sums digest, so the
    #    sums inventory excludes it, and the seal manifest covers gate1).
    write_provenance_r11(
        out_dir,
        verification,
        seeds,
        events_total,
        rerun_wall,
        [str(a) for a in sys.argv[1:]],
    )
    write_sha256sums(out_dir)
    gate = assemble_gate1_r11(
        out_dir,
        verification,
        ranked,
        combination_blocks,
        integrals_blocks,
        seagull_blocks,
        event_level,
        pull_check,
        prediction_check,
        rerun_records,
        int(args.top_n),
        len(tables),
        rerun_wall,
    )
    seal_entries = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.name != SEAL_NAME:
            seal_entries.append(
                {
                    "path": path.relative_to(out_dir).as_posix(),
                    "sha256": r01.sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    seal = {
        "sealed_at_utc": _utc_now_iso(),
        "run_directory": out_dir.name,
        "card": CARD,
        "format": (
            "SHA256SUMS.txt covers every payload file in sha256sum -c "
            "format (sorted relative paths), excluding the files that "
            "cannot carry their own digests (SHA256SUMS.txt itself, "
            "seal_manifest.json, provenance.json, gate1_r11.json); "
            "seal_manifest.json records the payload files plus "
            "SHA256SUMS.txt, provenance.json, and gate1_r11.json; "
            "seal_manifest.json cannot contain its own digest"
        ),
        "files": seal_entries,
    }
    _write_json(out_dir / SEAL_NAME, seal)

    # Re-verify every seal and sums entry; any mismatch fails the run.
    seal_ok = True
    for entry in seal["files"]:
        path = out_dir / entry["path"]
        if not path.is_file() or r01.sha256_file(path) != entry["sha256"]:
            seal_ok = False
            break
    sums_ok = True
    sums_path = out_dir / SHA256SUMS_NAME
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        file_path = out_dir / name
        if not file_path.is_file() or r01.sha256_file(file_path) != digest:
            sums_ok = False
            break
    if not seal_ok or not sums_ok:
        print(
            "[r11] seal verification FAILED; the run directory is not "
            "immutable as written",
            flush=True,
        )
        return 1
    print(
        f"[r11] run directory sealed: {out_dir} "
        f"({len(seal['files'])} sealed files); gate1_r11.json verdict "
        f"{gate['verdict']}",
        flush=True,
    )
    summary = combinations_payload["combinations"]
    for name in fo.COMBINATION_NAMES:
        block = summary[name]
        values = [cell["m"] for cell in block["per_cell"]]
        mean_value = sum(values) / len(values) if values else float("nan")
        print(
            f"[r11] combination {name}: {block['n_cells']} cells "
            f"(unmatched {block['n_unmatched_cells']}), "
            f"mean model value {mean_value:.6g}",
            flush=True,
        )
    for species, moments in event_level["multiplicity"].items():
        print(
            f"[r11] multiplicity {species}: <n> {moments['n_mean']:.6g}, "
            f"<n^2> {moments['n2_mean']:.6g}",
            flush=True,
        )
    gap = event_level["rapidity_gap_fraction"]
    print(
        f"[r11] gap fraction (dy_gap > {DY_GAP_THRESHOLD}): "
        f"{gap['fraction']:.6g} over {event_level['n_events_total']} events",
        flush=True,
    )
    inelasticity = event_level["leading_baryon_inelasticity"]
    print(
        f"[r11] leading-baryon inelasticity: mean {inelasticity['mean']:.6g}, "
        f"RMS {inelasticity['rms']:.6g} "
        f"(events without a proton {inelasticity['n_events_without_proton']})",
        flush=True,
    )
    return 0 if gate["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
