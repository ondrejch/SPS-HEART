#!/usr/bin/env python3
"""R00 deterministic-correctness run driver (Round 1, card 9).

Runs the pre-registered deterministic-correctness benchmark R00 of plan
section 6 and writes its acceptance record ``gate0_r00.json``.  Two
stages, both on the staged p+p two-source builder (``PPEventBuilder``)
at the frozen M8 ten-parameter set of the ``legacy_v1`` manifest:

- the deterministic core: 10 pre-registered seeds
  (``seed_k = 20311000 + k``), 5,000 events per seed.  Every event is
  audited with ``na49_mixing.event_conservation.audit_staged_event``
  and the per-event reports reduce through ``vertex_audit_summary``.
  Each seed is additionally generated twice through independent
  multiprocessing worker pools (worker count 1 and worker count 4);
  the ``ExcitationEvent.event_hash`` sequence and the final-particle
  four-vectors must be bit-identical across the two runs.
- the conservation audit: 10 further pre-registered seeds
  (``audit_k = 20312000 + k``), 100,000 events per seed, exactly
  1,000,000 events in total.  The audit streams chunk by chunk through
  a worker pool and keeps only the aggregated counters, so memory stays
  flat over the full million events.

Verification before any generation: the config file sha256 must match
the pin recorded in card 1's manifest
(``config/v2_singlet/legacy_v1_manifest.json``) and the canonical sha256
of the manifest's ``m8_frozen_pp_parameters`` block must match the
pre-registered ``m8_parameter_hash``.  A mismatch refuses the run with
a non-zero exit code before a single event is generated.  The driver
also verifies the ``pre_registration.json`` found in the output
directory (it writes the canonical one when absent).

Per-event seeding contract: event ``e`` of seed ``s`` draws from
``numpy.random.default_rng(numpy.random.SeedSequence(s,
spawn_key=(e,)))``.  The rule is a pure function of ``(s, e)`` and
consumes no shared generator state, so any worker count produces the
same event bit for bit; the worker pools exist to spread the wall clock,
not to define the stream.

Output files under ``--out``: ``pre_registration.json``,
``deterministic_core.json`` with one ``deterministic_seed_<seed>.json``
per core seed, ``conservation_audit_1m.json`` with one
``audit_seed_<seed>.json`` per audit seed, ``gate0_r00.json`` (written
when both stage results are present), and the seal pair
``SHA256SUMS.txt`` plus ``seal_manifest.json`` written after every other
file.  The seal is the manifest-plus-digest inventory of the finished
run directory; it is written last and never cleaned.

Exit codes: 0 when every executed assertion holds, 1 when a stage
assertion fails (the acceptance record still carries the numbers), 2
when a verification refusal stops the run before any generation (the
refusal text goes to stderr).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

# ---------------------------------------------------------------------------
# Tree location: this driver runs from the nested tree
# (dev/mixing2006/helpers/benchmarking/) and from its byte-identical
# monorepo mirror (helpers/benchmarking/).  Both must find the
# na49_mixing package and card 1's manifest.
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


NESTED_ROOT, MONO_ROOT = _locate_roots()
if str(NESTED_ROOT) not in sys.path:
    sys.path.insert(0, str(NESTED_ROOT))

from na49_mixing.continuum_channels import ContinuumParameters  # noqa: E402
from na49_mixing.event_builder_pp import PPEventBuilder, PPExcitationConfig  # noqa: E402
from na49_mixing.event_conservation import (  # noqa: E402
    VertexAuditReport,
    audit_staged_event,
    vertex_audit_summary,
)
from na49_mixing.exchange_model import (  # noqa: E402
    ReggeExcitationParameters,
    SimpleExcitationParameters,
)


def _refuse(message: str) -> NoReturn:
    """Print a refusal to stderr and exit with code 2 before generation."""
    print(f"staged_r00_correctness: {message}", file=sys.stderr, flush=True)
    raise SystemExit(2)

# ---------------------------------------------------------------------------
# Pre-registered run identity (card TASK-20260911-r1-run-r00-deterministic-
# correctness; the seeds, the event counts, and the tolerances are verbatim
# from the card, the two hashes are read off the frozen tree).
# ---------------------------------------------------------------------------

CARD = "TASK-20260911-r1-run-r00-deterministic-correctness"
RUN_DESCRIPTION = "r00_deterministic_correctness_2026-09-11"
GATE_RECORD_NAME = "gate0_r00.json"
GATE_CLAUSE = "deterministic_correctness_and_conservation_1m"

STAGE_CORE_NAME = "deterministic_core.json"
STAGE_AUDIT_NAME = "conservation_audit_1m.json"
PREREGISTRATION_NAME = "pre_registration.json"
SHA256SUMS_NAME = "SHA256SUMS.txt"
SEAL_NAME = "seal_manifest.json"

CORE_STAGE_LABEL = "r00_deterministic"
AUDIT_STAGE_LABEL = "r00_audit"

DETERMINISTIC_SEEDS = tuple(20311000 + k for k in range(10))
AUDIT_SEEDS = tuple(20312000 + k for k in range(10))
EVENTS_PER_SEED_DETERMINISTIC = 5000
EVENTS_PER_SEED_AUDIT = 100000

# Frozen M8 p+p calibration set: verbatim copy of the manifest's
# frozen_fit_outputs.m8_frozen_pp_parameters block (combined-error optimum
# of dev/00runs/staged_binned_fit_mc_2026-09-06/, decay hypothesis H3).
# M8_PARAMETER_HASH is the sha256 of the canonical JSON
# (json.dumps(sort_keys=True, separators=(",", ":"))); the freeze report
# dev/00runs/r1_legacy_v1_freeze_2026-09-11/freeze_report.json records the
# same digest as m8_parameters_hash.
M8_FROZEN_PP_PARAMETERS = {
    "regge.alpha_exc0": 1.1418864566363858,
    "regge.effective_transfer_slope": 8.504140762083612,
    "regge.edge_power": 2.3769493784466396,
    "continuum.match_mass": 2.680194110727528,
    "continuum.pion_multiplicity_slope": 1.5226475471033067,
    "continuum.strange_lambda_strength": 0.30184606498191385,
    "continuum.strange_sigma_strength": 0.3777282734799746,
    "continuum.strange_pair_strength": 0.19187855210738686,
    "continuum.pair_strength": 0.08411858717003319,
    "continuum.pair_threshold_power": 1.7610429093564994,
}
M8_PARAMETER_HASH = "358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1"

# Config pins from the legacy_v1 manifest source_tree_sha256 inventory.
PRODUCTION_CONFIG_NAME = "staged_excitation.json"
PRODUCTION_CONFIG_RELPATH = "dev/mixing2006/config/staged_excitation.json"
PRODUCTION_CONFIG_SHA256 = (
    "154e41028fc68d63daf74af73ab812789ae11915fcdabb3b1a6d32deabf6693f"
)
SMOKE_CONFIG_NAME = "staged_excitation_smoke.json"
SMOKE_CONFIG_RELPATH = "dev/mixing2006/config/staged_excitation_smoke.json"
SMOKE_CONFIG_SHA256 = (
    "7d36656a128a42da9f0c2342680befe6c81b500c432df97be689d61131881035"
)
MANIFEST_RELPATH = "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json"

EVENT_SEED_RULE = (
    "numpy.random.default_rng(numpy.random.SeedSequence(seed, "
    "spawn_key=(event_id,))); a pure function of (seed, event_id), no "
    "shared generator state, so any worker count is bit-identical"
)

CORE_CHUNK_SIZE = 500
AUDIT_CHUNK_SIZE = 2500
CORE_WORKERS_RUN_A = 1
CORE_WORKERS_RUN_B = 4
DEFAULT_AUDIT_WORKERS = 4


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(obj: Any) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Pre-registration and verification
# ---------------------------------------------------------------------------


def default_pre_registration() -> dict:
    """The R00 pre-registration payload, exactly as the card defines it."""
    return {
        "deterministic_seeds": list(DETERMINISTIC_SEEDS),
        "audit_seeds": list(AUDIT_SEEDS),
        "events_per_seed_deterministic": EVENTS_PER_SEED_DETERMINISTIC,
        "events_per_seed_audit": EVENTS_PER_SEED_AUDIT,
        "config_hash": PRODUCTION_CONFIG_SHA256,
        "m8_parameter_hash": M8_PARAMETER_HASH,
        "tolerances": {
            "four_momentum_relative_max": 1e-10,
            "mass_shell_relative": 1e-10,
            "bqs_exact": True,
            "reproducibility": "bit_identical_event_hash_and_p4",
        },
    }


def _config_pin(name: str) -> tuple[str, str]:
    """Map a bundled config file name to (pinned sha256, manifest key)."""
    if name == PRODUCTION_CONFIG_NAME:
        return PRODUCTION_CONFIG_SHA256, PRODUCTION_CONFIG_RELPATH
    if name == SMOKE_CONFIG_NAME:
        return SMOKE_CONFIG_SHA256, SMOKE_CONFIG_RELPATH
    _refuse(
        f"refusing config {name!r}; R00 consumes only the two bundled "
        f"legacy_v1 configs ({PRODUCTION_CONFIG_NAME!r}, {SMOKE_CONFIG_NAME!r})"
    )


def verify_config_and_m8(config_path: Path, pre_registration: dict) -> dict:
    """Verify the config sha256 and the M8 parameter hash against card 1.

    The config file bytes must hash to the manifest pin (and, for the
    production config, to the pre-registration's ``config_hash``).  The
    manifest's ``m8_frozen_pp_parameters`` block must hash canonically to
    the pre-registered ``m8_parameter_hash`` and equal the driver's
    verbatim copy.  Any mismatch raises ``SystemExit`` (non-zero exit)
    before generation starts.
    """
    path = Path(config_path)
    if not path.is_file():
        _refuse(f"config not found: {path}")
    name = path.name
    pinned_sha, manifest_key = _config_pin(name)
    actual_sha = sha256_file(path)

    problems: list[str] = []
    if name == PRODUCTION_CONFIG_NAME:
        registered = pre_registration.get("config_hash")
        if registered != actual_sha:
            problems.append(
                f"pre-registration config_hash {registered!r} != config file "
                f"sha256 {actual_sha!r} ({path})"
            )
    if actual_sha != pinned_sha:
        problems.append(
            f"config sha256 {actual_sha!r} != pinned {pinned_sha!r} "
            f"({manifest_key})"
        )

    manifest_path = NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json"
    if not manifest_path.is_file():
        _refuse(f"card 1 manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    m8_block = manifest.get("frozen_fit_outputs", {}).get(
        "m8_frozen_pp_parameters"
    )
    if m8_block is None:
        problems.append("manifest carries no frozen_fit_outputs."
                        "m8_frozen_pp_parameters block")
    else:
        block_sha = canonical_json_sha256(m8_block)
        if block_sha != M8_PARAMETER_HASH:
            problems.append(
                f"manifest m8_frozen_pp_parameters canonical sha256 "
                f"{block_sha!r} != pre-registered {M8_PARAMETER_HASH!r}"
            )
        if m8_block != M8_FROZEN_PP_PARAMETERS:
            problems.append(
                "manifest m8_frozen_pp_parameters values differ from the "
                "driver's verbatim frozen copy"
            )
        registered_m8 = pre_registration.get("m8_parameter_hash")
        if registered_m8 != M8_PARAMETER_HASH:
            problems.append(
                f"pre-registration m8_parameter_hash {registered_m8!r} != "
                f"{M8_PARAMETER_HASH!r}"
            )
    inventory = manifest.get("source_tree_sha256", {})
    manifest_pin = inventory.get(manifest_key)
    if manifest_pin is None:
        problems.append(f"manifest does not pin {manifest_key}")
    elif manifest_pin != actual_sha:
        problems.append(
            f"manifest pin for {manifest_key} is {manifest_pin!r} but the "
            f"config file hashes to {actual_sha!r}"
        )

    if problems:
        _refuse(
            "config/M8 verification failed; refusing to run:\n  "
            + "\n  ".join(problems)
        )
    return {
        "config_path": str(path),
        "config_sha256": actual_sha,
        "manifest_path": str(manifest_path),
        "manifest_config_pin": manifest_pin,
        "m8_parameter_hash": M8_PARAMETER_HASH,
        "production_config": name == PRODUCTION_CONFIG_NAME,
    }


def build_pp_config(raw_config: dict) -> tuple[PPExcitationConfig, dict]:
    """Parameter dataclasses from the config with the frozen M8 set applied.

    Mirrors the M13 ``build_parameter_set`` convention: the bundled
    config blocks build the dataclasses, then each of the ten M8 values
    overrides its field, the application is asserted, and the H3 decay
    kernel must hold as the dataclass default.
    """
    simple = SimpleExcitationParameters(**(raw_config.get("simple") or {}))
    regge = ReggeExcitationParameters(**(raw_config.get("regge") or {}))
    continuum = ContinuumParameters(**(raw_config.get("continuum") or {}))
    for name, value in M8_FROZEN_PP_PARAMETERS.items():
        group, field_name = name.split(".", 1)
        if group == "regge":
            regge = dataclasses.replace(regge, **{field_name: float(value)})
        else:
            continuum = dataclasses.replace(continuum, **{field_name: float(value)})
    applied = {
        "regge": {k: float(v) for k, v in regge.__dict__.items()},
        "continuum": {
            k: (v if isinstance(v, str) else float(v))
            for k, v in continuum.__dict__.items()
        },
    }
    for name, value in M8_FROZEN_PP_PARAMETERS.items():
        group, field_name = name.split(".", 1)
        if applied[group][field_name] != float(value):
            _refuse(f"parameter application failed for {name}")
    if str(applied["continuum"]["decay_hypothesis"]).strip().upper() != "H3":
        _refuse("the H3 decay kernel must apply as the dataclass default")
    config = PPExcitationConfig(
        beam_momentum=float(raw_config.get("beam_momentum", 158.0)),
        production_mode="regge",
        simple=simple,
        regge=regge,
        continuum=continuum,
    )
    return config, applied


def config_payload(raw_config: dict) -> dict:
    """Plain-dict projection of the config for worker processes."""
    return {
        "beam_momentum": float(raw_config.get("beam_momentum", 158.0)),
        "simple": dict(raw_config.get("simple") or {}),
        "regge": dict(raw_config.get("regge") or {}),
        "continuum": dict(raw_config.get("continuum") or {}),
    }


# ---------------------------------------------------------------------------
# Generation and audit machinery (worker side)
# ---------------------------------------------------------------------------


def _event_rng(seed: int, event_id: int):
    return np.random.default_rng(
        np.random.SeedSequence(int(seed), spawn_key=(int(event_id),))
    )


def _generate_chunk(task: tuple):
    """Generate and audit one chunk of events in a worker process.

    Returns ``(records, chunk_summary)``.  ``records`` holds the
    per-event reproducibility records (event hash plus final-particle
    four-vectors, ordered by event id within the chunk) and is empty
    when the task asks for audit-only collection; the 1M audit streams
    counters only, so no event payload crosses the process boundary.
    ``chunk_summary`` is the ``vertex_audit_summary`` of the chunk's
    per-event reports.
    """
    seed, event_ids, payload, stage_label, collect_records = task
    pp_config, _ = build_pp_config(payload)
    records = []
    reports: list[VertexAuditReport] = []
    for event_id in event_ids:
        builder = PPEventBuilder(pp_config, _event_rng(seed, event_id))
        event = builder.generate(int(event_id), stage=stage_label)
        report = audit_staged_event(
            event,
            relative_tolerance=pp_config.conservation_tolerance,
            event_index=int(event_id),
        )
        if collect_records:
            records.append(
                {
                    "event_id": int(event_id),
                    "event_hash": event.as_dict()["event_hash"],
                    "p4": [
                        [particle.p4.e, particle.p4.px, particle.p4.py, particle.p4.pz]
                        for particle in event.final_particles
                    ],
                }
            )
        reports.append(report)
    return records, vertex_audit_summary(reports)


def _map_chunks(tasks: list, workers: int, progress_label: str):
    """Yield worker results as they complete over a fork worker pool."""
    context = multiprocessing.get_context("fork")
    completed = 0
    started = time.perf_counter()
    with context.Pool(processes=workers) as pool:
        for result in pool.imap_unordered(_generate_chunk, tasks, chunksize=1):
            completed += 1
            yield result
            if completed % 10 == 0 or completed == len(tasks):
                elapsed = time.perf_counter() - started
                print(
                    f"[r00] {progress_label}: chunk {completed}/{len(tasks)} "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )


def combine_vertex_audit_summaries(summaries: list) -> dict:
    """Reduce per-chunk ``vertex_audit_summary`` dicts to the run summary.

    Each chunk summary is the ``vertex_audit_summary`` of its reports;
    the reduction sums the counters, takes the maxima, and ANDs
    ``all_pass``, so the combined dict carries the same fields and
    semantics as a direct ``vertex_audit_summary`` over all reports.
    """
    summaries = list(summaries)
    if not summaries:
        return vertex_audit_summary([])
    return {
        "n_events": int(sum(s["n_events"] for s in summaries)),
        "n_vertices": int(sum(s["n_vertices"] for s in summaries)),
        "conservation_failures": int(
            sum(s["conservation_failures"] for s in summaries)
        ),
        "offshell_daughters": int(sum(s["offshell_daughters"] for s in summaries)),
        "bqs_residual_nonzero": int(sum(s["bqs_residual_nonzero"] for s in summaries)),
        "ancestry_violations": int(sum(s["ancestry_violations"] for s in summaries)),
        "max_four_momentum_relative_residual": float(
            max(s["max_four_momentum_relative_residual"] for s in summaries)
        ),
        "max_mass_shell_relative_deviation": float(
            max(s["max_mass_shell_relative_deviation"] for s in summaries)
        ),
        "all_pass": bool(all(s["all_pass"] for s in summaries)),
    }


def _event_hash_digest(records: list) -> str:
    ordered = [record["event_hash"] for record in records]
    payload = json.dumps(ordered, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _p4_digest(records: list) -> str:
    flattened = [
        component for record in records for particle in record["p4"] for component in particle
    ]
    payload = json.dumps(flattened, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _records_sorted(parts: list) -> list:
    records = [record for part_records, _ in parts for record in part_records]
    records.sort(key=lambda record: record["event_id"])
    return records


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------


def run_deterministic_seed(
    seed: int,
    n_events: int,
    payload: dict,
    out_dir: Path | None = None,
) -> dict:
    """One deterministic-core seed: two worker-count runs plus comparison."""
    event_ids = list(range(int(n_events)))
    chunks = [
        event_ids[start : start + CORE_CHUNK_SIZE]
        for start in range(0, len(event_ids), CORE_CHUNK_SIZE)
    ]
    tasks = [
        (int(seed), chunk, payload, CORE_STAGE_LABEL, True) for chunk in chunks
    ]

    started_a = time.perf_counter()
    parts_a = list(
        _map_chunks(tasks, CORE_WORKERS_RUN_A, f"core seed {seed} run A (1 worker)")
    )
    seconds_a = time.perf_counter() - started_a
    started_b = time.perf_counter()
    parts_b = list(
        _map_chunks(tasks, CORE_WORKERS_RUN_B, f"core seed {seed} run B (4 workers)")
    )
    seconds_b = time.perf_counter() - started_b

    records_a = _records_sorted(parts_a)
    records_b = _records_sorted(parts_b)
    digest_a = _event_hash_digest(records_a)
    digest_b = _event_hash_digest(records_b)
    p4_digest_a = _p4_digest(records_a)
    p4_digest_b = _p4_digest(records_b)
    bit_identical = (
        records_a == records_b
        and digest_a == digest_b
        and p4_digest_a == p4_digest_b
    )
    summary_a = combine_vertex_audit_summaries([summary for _, summary in parts_a])
    summary_b = combine_vertex_audit_summaries([summary for _, summary in parts_b])
    summaries_identical = summary_a == summary_b

    result = {
        "seed": int(seed),
        "n_events": int(n_events),
        "stage_file": STAGE_CORE_NAME,
        "event_seed_rule": EVENT_SEED_RULE,
        "chunk_size": CORE_CHUNK_SIZE,
        "chunks": len(chunks),
        "worker_counts": {"run_a": CORE_WORKERS_RUN_A, "run_b": CORE_WORKERS_RUN_B},
        "run_a": {
            "worker_count": CORE_WORKERS_RUN_A,
            "wall_seconds": round(seconds_a, 3),
            "repro_run_a_sha256": digest_a,
            "p4_set_sha256": p4_digest_a,
            "audit_summary": summary_a,
        },
        "run_b": {
            "worker_count": CORE_WORKERS_RUN_B,
            "wall_seconds": round(seconds_b, 3),
            "repro_run_b_sha256": digest_b,
            "p4_set_sha256": p4_digest_b,
            "audit_summary": summary_b,
        },
        "bit_identical": bool(bit_identical),
        "audit_summaries_identical": bool(summaries_identical),
        "conservation_failures": summary_a["conservation_failures"],
        "offshell_daughters": summary_a["offshell_daughters"],
        "bqs_residual_nonzero": summary_a["bqs_residual_nonzero"],
        "ancestry_violations": summary_a["ancestry_violations"],
        "max_four_momentum_relative_residual": summary_a[
            "max_four_momentum_relative_residual"
        ],
        "max_mass_shell_relative_deviation": summary_a[
            "max_mass_shell_relative_deviation"
        ],
        "all_pass": bool(
            summary_a["all_pass"]
            and summary_b["all_pass"]
            and bit_identical
            and summaries_identical
        ),
    }
    if out_dir is not None:
        _write_json(Path(out_dir) / f"deterministic_seed_{int(seed)}.json", result)
    return result


def gate0_per_seed_fields(result: dict) -> dict:
    """The card's per-seed schema for the acceptance record, exactly."""
    return {
        "seed": result["seed"],
        "n_events": result["n_events"],
        "conservation_failures": result["conservation_failures"],
        "offshell_daughters": result["offshell_daughters"],
        "bqs_residual_nonzero": result["bqs_residual_nonzero"],
        "ancestry_violations": result["ancestry_violations"],
        "repro_run_a_sha256": result["run_a"]["repro_run_a_sha256"],
        "repro_run_b_sha256": result["run_b"]["repro_run_b_sha256"],
        "bit_identical": result["bit_identical"],
    }


def run_deterministic_core(
    out_dir: Path,
    seeds: list,
    events_per_seed: int,
    raw_config: dict,
    verification: dict,
    pre_registration: dict,
) -> dict:
    """The deterministic core stage over its seed list."""
    payload = config_payload(raw_config)
    started = time.perf_counter()
    started_iso = _utc_now_iso()
    per_seed = []
    for index, seed in enumerate(seeds, start=1):
        print(
            f"[r00] deterministic core: seed {seed} ({index}/{len(seeds)}), "
            f"{events_per_seed} events x 2 worker-count runs",
            flush=True,
        )
        per_seed.append(
            run_deterministic_seed(seed, events_per_seed, payload, out_dir=out_dir)
        )
    wall = time.perf_counter() - started
    finished_iso = _utc_now_iso()
    summary = combine_vertex_audit_summaries(
        [entry["run_a"]["audit_summary"] for entry in per_seed]
    )
    stage = {
        "stage": "r00_deterministic_core",
        "card": CARD,
        "seeds": [int(seed) for seed in seeds],
        "events_per_seed": int(events_per_seed),
        "total_events": int(events_per_seed) * len(seeds),
        "per_seed": [gate0_per_seed_fields(entry) for entry in per_seed],
        "all_pass": bool(all(entry["all_pass"] for entry in per_seed)),
        "max_four_momentum_relative_residual": float(
            max(
                entry["max_four_momentum_relative_residual"] for entry in per_seed
            )
        ),
        "max_mass_shell_relative_deviation": float(
            max(entry["max_mass_shell_relative_deviation"] for entry in per_seed)
        ),
        "conservation_failures": summary["conservation_failures"],
        "offshell_daughters": summary["offshell_daughters"],
        "bqs_residual_nonzero": summary["bqs_residual_nonzero"],
        "ancestry_violations": summary["ancestry_violations"],
        "config_path": verification["config_path"],
        "config_sha256": verification["config_sha256"],
        "pre_registered_config_hash": pre_registration["config_hash"],
        "m8_parameter_hash": verification["m8_parameter_hash"],
        "production_mode": "regge",
        "event_seed_rule": EVENT_SEED_RULE,
        "chunk_size": CORE_CHUNK_SIZE,
        "worker_counts": {
            "run_a": CORE_WORKERS_RUN_A,
            "run_b": CORE_WORKERS_RUN_B,
        },
        "wall_seconds": round(wall, 3),
        "started_utc": started_iso,
        "finished_utc": finished_iso,
    }
    _write_json(Path(out_dir) / STAGE_CORE_NAME, stage)
    return stage


def run_audit_seed(
    seed: int,
    n_events: int,
    payload: dict,
    workers: int,
    out_dir: Path | None = None,
) -> dict:
    """One conservation-audit seed: streaming generation plus aggregation."""
    event_ids = list(range(int(n_events)))
    chunks = [
        event_ids[start : start + AUDIT_CHUNK_SIZE]
        for start in range(0, len(event_ids), AUDIT_CHUNK_SIZE)
    ]
    tasks = [
        (int(seed), chunk, payload, AUDIT_STAGE_LABEL, False) for chunk in chunks
    ]
    started = time.perf_counter()
    partials = []
    events_done = 0
    for records, summary in _map_chunks(
        tasks, int(workers), f"audit seed {seed}"
    ):
        del records
        partials.append(summary)
        events_done += summary["n_events"]
    wall = time.perf_counter() - started
    combined = combine_vertex_audit_summaries(partials)
    result = {
        "seed": int(seed),
        "n_events": combined["n_events"],
        "stage_file": STAGE_AUDIT_NAME,
        "event_seed_rule": EVENT_SEED_RULE,
        "worker_count": int(workers),
        "chunk_size": AUDIT_CHUNK_SIZE,
        "chunks": len(chunks),
        "wall_seconds": round(wall, 3),
        "audit_summary": combined,
    }
    if out_dir is not None:
        _write_json(Path(out_dir) / f"audit_seed_{int(seed)}.json", result)
    return result


def run_conservation_audit(
    out_dir: Path,
    seeds: list,
    events_per_seed: int,
    raw_config: dict,
    verification: dict,
    workers: int,
    pre_registration: dict,
) -> dict:
    """The 1,000,000-event conservation audit stage over its seed list."""
    payload = config_payload(raw_config)
    started = time.perf_counter()
    started_iso = _utc_now_iso()
    per_seed = []
    for index, seed in enumerate(seeds, start=1):
        print(
            f"[r00] conservation audit: seed {seed} ({index}/{len(seeds)}), "
            f"{events_per_seed} events",
            flush=True,
        )
        per_seed.append(
            run_audit_seed(seed, events_per_seed, payload, workers, out_dir=out_dir)
        )
    wall = time.perf_counter() - started
    finished_iso = _utc_now_iso()
    summary = combine_vertex_audit_summaries(
        [entry["audit_summary"] for entry in per_seed]
    )
    stage = {
        "stage": "r00_conservation_audit_1m",
        "card": CARD,
        "seeds": [int(seed) for seed in seeds],
        "events_per_seed": int(events_per_seed),
        "total_events": int(summary["n_events"]),
        "conservation_failures": summary["conservation_failures"],
        "offshell_daughters": summary["offshell_daughters"],
        "bqs_residual_nonzero": summary["bqs_residual_nonzero"],
        "ancestry_violations": summary["ancestry_violations"],
        "max_four_momentum_relative_residual": summary[
            "max_four_momentum_relative_residual"
        ],
        "max_mass_shell_relative_deviation": summary[
            "max_mass_shell_relative_deviation"
        ],
        "all_pass": summary["all_pass"],
        "per_seed": [
            {
                "seed": entry["seed"],
                "n_events": entry["n_events"],
                "conservation_failures": entry["audit_summary"][
                    "conservation_failures"
                ],
                "offshell_daughters": entry["audit_summary"]["offshell_daughters"],
                "bqs_residual_nonzero": entry["audit_summary"][
                    "bqs_residual_nonzero"
                ],
                "ancestry_violations": entry["audit_summary"]["ancestry_violations"],
                "wall_seconds": entry["wall_seconds"],
            }
            for entry in per_seed
        ],
        "config_path": verification["config_path"],
        "config_sha256": verification["config_sha256"],
        "pre_registered_config_hash": pre_registration["config_hash"],
        "m8_parameter_hash": verification["m8_parameter_hash"],
        "production_mode": "regge",
        "event_seed_rule": EVENT_SEED_RULE,
        "worker_count": int(workers),
        "chunk_size": AUDIT_CHUNK_SIZE,
        "wall_seconds": round(wall, 3),
        "started_utc": started_iso,
        "finished_utc": finished_iso,
    }
    _write_json(Path(out_dir) / STAGE_AUDIT_NAME, stage)
    return stage


# ---------------------------------------------------------------------------
# Acceptance record and seal
# ---------------------------------------------------------------------------


def _load_stage(out_dir: Path, name: str) -> dict:
    path = Path(out_dir) / name
    if not path.is_file():
        _refuse(f"stage result missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def assemble_gate0(out_dir: Path) -> dict:
    """Assemble ``gate0_r00.json`` from both stage results.

    The record matches the card's acceptance schema exactly.  The
    verdict is ``pass`` only when the hashes verified, every per-seed
    entry has zero counters and ``bit_identical`` true, and the 1M audit
    holds exactly 1,000,000 events with zero counters and both max
    residuals within 1e-10.
    """
    out_dir = Path(out_dir)
    core = _load_stage(out_dir, STAGE_CORE_NAME)
    audit = _load_stage(out_dir, STAGE_AUDIT_NAME)
    prereg_path = out_dir / PREREGISTRATION_NAME
    if not prereg_path.is_file():
        _refuse(f"pre-registration missing: {prereg_path}")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))

    config_hash_verified = bool(
        core.get("config_sha256") == prereg.get("config_hash")
        and audit.get("config_sha256") == prereg.get("config_hash")
        and core.get("config_sha256") == PRODUCTION_CONFIG_SHA256
    )
    m8_verified = bool(
        core.get("m8_parameter_hash") == prereg.get("m8_parameter_hash")
        and audit.get("m8_parameter_hash") == prereg.get("m8_parameter_hash")
        and core.get("m8_parameter_hash") == M8_PARAMETER_HASH
    )
    registration_ok = bool(
        core.get("seeds") == prereg.get("deterministic_seeds")
        and core.get("events_per_seed") == prereg.get("events_per_seed_deterministic")
        and audit.get("seeds") == prereg.get("audit_seeds")
        and audit.get("events_per_seed") == prereg.get("events_per_seed_audit")
    )

    per_seed = core.get("per_seed", [])
    core_ok = bool(
        core.get("total_events")
        == core.get("events_per_seed", 0) * len(core.get("seeds", []))
        and core.get("events_per_seed") == EVENTS_PER_SEED_DETERMINISTIC
        and core.get("all_pass")
        and per_seed
        and all(
            entry["conservation_failures"] == 0
            and entry["offshell_daughters"] == 0
            and entry["bqs_residual_nonzero"] == 0
            and entry["ancestry_violations"] == 0
            and entry["bit_identical"] is True
            for entry in per_seed
        )
    )
    audit_ok = bool(
        audit.get("total_events") == EVENTS_PER_SEED_AUDIT * len(AUDIT_SEEDS)
        and audit.get("total_events") == 1000000
        and audit.get("conservation_failures") == 0
        and audit.get("offshell_daughters") == 0
        and audit.get("bqs_residual_nonzero") == 0
        and audit.get("ancestry_violations") == 0
        and float(audit.get("max_four_momentum_relative_residual", 1.0)) <= 1e-10
        and float(audit.get("max_mass_shell_relative_deviation", 1.0)) <= 1e-10
        and audit.get("all_pass")
    )
    verdict = (
        "pass"
        if (
            config_hash_verified
            and m8_verified
            and registration_ok
            and core_ok
            and audit_ok
        )
        else "fail"
    )
    gate = {
        "gate": "gate0",
        "clause": GATE_CLAUSE,
        "config_hash_verified": config_hash_verified,
        "m8_parameter_hash_verified": m8_verified,
        "deterministic_core": {
            "seeds": core.get("seeds", []),
            "events_per_seed": core.get("events_per_seed"),
            "total_events": core.get("total_events"),
            "per_seed": per_seed,
            "all_pass": core.get("all_pass"),
        },
        "conservation_audit_1m": {
            "seeds": audit.get("seeds", []),
            "events_per_seed": audit.get("events_per_seed"),
            "total_events": audit.get("total_events"),
            "conservation_failures": audit.get("conservation_failures"),
            "offshell_daughters": audit.get("offshell_daughters"),
            "bqs_residual_nonzero": audit.get("bqs_residual_nonzero"),
            "ancestry_violations": audit.get("ancestry_violations"),
            "max_four_momentum_relative_residual": audit.get(
                "max_four_momentum_relative_residual"
            ),
            "max_mass_shell_relative_deviation": audit.get(
                "max_mass_shell_relative_deviation"
            ),
        },
        "verdict": verdict,
    }
    _write_json(out_dir / GATE_RECORD_NAME, gate)
    return gate


def seal_run_dir(out_dir: Path) -> dict:
    """Seal the finished run directory: sha256 inventory plus seal manifest.

    ``SHA256SUMS.txt`` lists every payload file (everything except the
    two seal files) in ``sha256sum -c`` format with sorted relative
    paths.  ``seal_manifest.json`` then records every payload file plus
    ``SHA256SUMS.txt`` itself; the seal manifest cannot contain its own
    digest and says so.
    """
    out_dir = Path(out_dir)
    skip = {SHA256SUMS_NAME, SEAL_NAME}
    payload = sorted(
        p
        for p in out_dir.iterdir()
        if p.is_file() and p.name not in skip
    )
    lines = []
    for path in payload:
        lines.append(f"{sha256_file(path)}  {path.name}")
    (out_dir / SHA256SUMS_NAME).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    entries = []
    for path in payload + [out_dir / SHA256SUMS_NAME]:
        entries.append(
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    seal = {
        "sealed_at_utc": _utc_now_iso(),
        "run_directory": out_dir.name,
        "card": CARD,
        "format": (
            "SHA256SUMS.txt covers every payload file in sha256sum -c "
            "format (sorted file names); seal_manifest.json records the "
            "payload files plus SHA256SUMS.txt; seal_manifest.json cannot "
            "contain its own digest"
        ),
        "files": entries,
    }
    _write_json(out_dir / SEAL_NAME, seal)
    return seal


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def resolve_out_dir(raw: str) -> Path:
    """Resolve ``--out``; ``dev/00runs/...`` paths are monorepo-relative.

    The card's commands invoke this driver from the nested tree with
    ``--out dev/00runs/r00_deterministic_correctness_2026-09-11``, so an
    artifact-store-shaped path resolves against the monorepo root even
    though the process working directory is the nested tree.  Absolute
    paths and every other relative path resolve normally.
    """
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    if path.parts[:2] == ("dev", "00runs"):
        return MONO_ROOT / path
    return Path.cwd() / path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "R00 deterministic-correctness run driver: the pre-registered "
            "10 x 5,000-event deterministic core, the 1,000,000-event "
            "conservation audit, and the gate0_r00.json acceptance record"
        )
    )
    parser.add_argument("--config", required=True, help="bundled legacy_v1 config JSON")
    parser.add_argument(
        "--mode",
        choices=("deterministic", "audit", "all"),
        required=True,
        help="deterministic core, conservation audit, or both in order",
    )
    parser.add_argument(
        "--out", required=True, help="run directory (dev/00runs/... resolves against the monorepo root)"
    )
    parser.add_argument(
        "--events-per-seed",
        type=int,
        default=None,
        help=(
            "ad-hoc event count per seed; omit for the pre-registered "
            "counts (5,000 core / 100,000 audit)"
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "ad-hoc seed list (one seed for the card's smoke command); "
            "omit for the pre-registered seed lists"
        ),
    )
    parser.add_argument(
        "--audit-workers",
        type=int,
        default=DEFAULT_AUDIT_WORKERS,
        help="worker processes for the conservation audit stage (default 4)",
    )
    return parser


def main(argv: list | None = None) -> int:
    args = _build_parser().parse_args(argv)
    # The ad-hoc event count must be a positive integer. This single
    # pre-generation check covers every code path that consumes
    # --events-per-seed (deterministic, audit, all) and fires before the
    # --out directory is created. Omitting the flag selects the
    # pre-registered counts (5,000 core / 100,000 audit), which are
    # positive by construction.
    if args.events_per_seed is not None and int(args.events_per_seed) <= 0:
        _refuse(
            f"--events-per-seed must be a positive integer, got "
            f"{args.events_per_seed}; 0 and negative counts are refused "
            "before any generation (omit the flag for the pre-registered "
            "counts: 5,000 core / 100,000 audit)"
        )
    out_dir = resolve_out_dir(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload_expected = default_pre_registration()
    prereg_path = out_dir / PREREGISTRATION_NAME
    if prereg_path.is_file():
        loaded = json.loads(prereg_path.read_text(encoding="utf-8"))
        if loaded != payload_expected:
            _refuse(
                f"pre-registration mismatch: {prereg_path} does not equal the "
                "R00 pre-registration (seeds, event counts, hashes, and "
                "tolerances are pre-registered; a different registration "
                "files its own card)"
            )
    else:
        _write_json(prereg_path, payload_expected)
    pre_registration = payload_expected

    config_path = Path(args.config)
    verification = verify_config_and_m8(config_path, pre_registration)
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    build_pp_config(raw_config)  # refuse before any generation on a bad application

    print(
        f"[r00] config {config_path} sha256 {verification['config_sha256']}",
        flush=True,
    )
    print(
        f"[r00] M8 parameter hash {verification['m8_parameter_hash']} verified "
        f"against {verification['manifest_path']}",
        flush=True,
    )

    exit_code = 0
    if args.mode in ("deterministic", "all"):
        seeds = (
            [int(s) for s in args.seeds]
            if args.seeds is not None
            else list(pre_registration["deterministic_seeds"])
        )
        events = (
            int(args.events_per_seed)
            if args.events_per_seed is not None
            else int(pre_registration["events_per_seed_deterministic"])
        )
        core = run_deterministic_core(
            out_dir, seeds, events, raw_config, verification, pre_registration
        )
        core_ok = bool(
            core["all_pass"]
            and all(
                entry["conservation_failures"] == 0
                and entry["offshell_daughters"] == 0
                and entry["bqs_residual_nonzero"] == 0
                and entry["ancestry_violations"] == 0
                and entry["bit_identical"] is True
                for entry in core["per_seed"]
            )
        )
        print(
            f"[r00] deterministic core: total_events {core['total_events']} "
            f"conservation_failures {core['conservation_failures']} "
            f"offshell_daughters {core['offshell_daughters']} "
            f"bqs_residual_nonzero {core['bqs_residual_nonzero']} "
            f"ancestry_violations {core['ancestry_violations']} "
            f"all_pass {core['all_pass']}",
            flush=True,
        )
        if not core_ok:
            exit_code = 1
    if args.mode in ("audit", "all"):
        seeds = (
            [int(s) for s in args.seeds]
            if args.seeds is not None
            else list(pre_registration["audit_seeds"])
        )
        events = (
            int(args.events_per_seed)
            if args.events_per_seed is not None
            else int(pre_registration["events_per_seed_audit"])
        )
        audit = run_conservation_audit(
            out_dir,
            seeds,
            events,
            raw_config,
            verification,
            int(args.audit_workers),
            pre_registration,
        )
        audit_ok = bool(
            audit["all_pass"]
            and audit["conservation_failures"] == 0
            and audit["offshell_daughters"] == 0
            and audit["bqs_residual_nonzero"] == 0
            and audit["ancestry_violations"] == 0
            and audit["max_four_momentum_relative_residual"] <= 1e-10
            and audit["max_mass_shell_relative_deviation"] <= 1e-10
        )
        print(
            f"[r00] conservation audit: total_events {audit['total_events']} "
            f"conservation_failures {audit['conservation_failures']} "
            f"offshell_daughters {audit['offshell_daughters']} "
            f"bqs_residual_nonzero {audit['bqs_residual_nonzero']} "
            f"max_four_momentum_relative_residual "
            f"{audit['max_four_momentum_relative_residual']:.3e} "
            f"max_mass_shell_relative_deviation "
            f"{audit['max_mass_shell_relative_deviation']:.3e} "
            f"all_pass {audit['all_pass']}",
            flush=True,
        )
        if not audit_ok:
            exit_code = 1

    if (out_dir / STAGE_CORE_NAME).is_file() and (
        out_dir / STAGE_AUDIT_NAME
    ).is_file():
        gate = assemble_gate0(out_dir)
        seal_run_dir(out_dir)
        print(
            f"[r00] gate0_r00.json verdict {gate['verdict']}; run directory "
            f"sealed: {out_dir}",
            flush=True,
        )
        if gate["verdict"] != "pass":
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
