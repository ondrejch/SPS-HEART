#!/usr/bin/env python3
"""R23 driver: the resonance-only p+p ablation (Round 2, card 13).

Task ``TASK-20260913-r2-run-r23-resonance-only-pp-ablation`` (plan
section 8 run R23, line 567; the Round 2 card set, card 13; cards 10,
11, and 12 are the R20/R21/R22 prerequisites).  R23 regenerates the
staged p+p model at T2 statistics (8 replicas x 100,000 events) with
the legacy source/exchange layer bit-identical to the frozen legacy
builder (the M8 frozen ten-parameter set, the two-source
``(B, Q, S) = (1, 1, 0)`` topology) and only the low-mass decay kernel
replaced: sources at or below the M8-frozen boundary
``continuum.match_mass = 2.680194110727528 GeV`` decay through the
catalog-driven card-6/7 kernel (``na49_mixing.resonance_pp_builder``),
sources above the boundary keep the legacy ``SourceDecayer`` (H3)
kernel bit-identical at the same seed.  The comparison is against the
sealed R01 archive (``dev/00runs/r01_legacy_pp_reference_2026-09-11/``)
and the training SPS tables over the identified low-mass-sensitive
observable set (the top-5 entries of the sealed R11 ranked residual
map).  No fitting: no optimizer is invoked; the branching fractions
and widths are the external inputs of plan section 8.2.

The driver implements the full run procedure and executes it in
phases: each invocation performs exactly one step and stops (the run
card dispatches the steps; pre-registration, bit-identity check,
generation, comparison, acceptance evidence, seal, and
clean-environment validation land in this order, and the run dir is
immutable after the seal).

Modes (exactly one per invocation):

- ``--pre-registration``: write ``pre_registration.json`` and
  ``provenance.json`` into the run dir, then stop.  Nothing is
  generated.
- ``--bit-identity``: run the pre-registered 10,000-event bit-identity
  check (legacy vs new builder at seed 20300913), write
  ``bit_identity_report.json``, then stop.  A mismatch stops the run
  before any replica is generated.
- ``--generate [--replica K]``: generate one replica (with ``--replica
  K``, K = 0..7) or every replica (without), writing
  ``replica_<seed>.json`` into the run dir, then stop.  Requires the
  pre-registration and a passing bit-identity check.
- ``--compare``: run the pre-registered comparison metrics (i)-(iii)
  against the sealed R01 archive, the training tables, and the sealed
  R11 ranked residual map; write ``comparison_r01.json``, then stop.
  Requires all eight production replicas.
- ``--gate``: write ``gate2_r23.json`` (evidence only), then stop.
  Requires the comparison, the bit-identity check, and the sealed
  R20/R21/R22 acceptance records.
- ``--seal``: write ``seal_manifest.json`` and ``SHA256SUMS.txt`` over
  the run dir, then stop.  The run dir is immutable after the seal.
- ``--clean-validate``: verify the seal, re-run the comparison from a
  clean interpreter over the sealed artifacts, and confirm the report
  hash matches; write ``clean_environment_commands.json``, then stop.
- ``--smoke`` (with ``--generate``): hermetic reduced statistics
  (one replica of ``--smoke-events`` events, no pre-registration or
  bit-identity requirement, records tagged ``mode = "smoke"``) for
  driver validation; ``--gate`` / ``--seal`` / ``--clean-validate``
  refuse smoke artifacts.
- ``--run-dir <path>``: override the run dir.

Pre-registration contract (``pre_registration.json``, written before
any generation):

- **Statistics (T2, plan section 5.1)**: 8 replicas x 100,000 p+p
  events per replica = 800,000 events (the upper T2 bound, consistent
  with R02's 100,000-event replicas).
- **Replica seeds**: replica ``k`` (k = 0..7) runs at seed
  ``20260913 + 40000 + k`` (20300913 to 20300920), the M8 seed rule of
  ``config/staged_excitation.json`` with the Round 2 master seed
  ``20260913``.  The catalog-driven decay layer binds the closed
  ``decay_channel`` / ``decay_kinematics`` substream labels through
  ``derive_substreams(20260913, replica_index, event_id)``; the legacy
  source/exchange layers draw from the legacy staged per-event
  generator (``SeedSequence(seed, spawn_key=(event_id,))``).
- **Parameters**: the M8 frozen ten-parameter set, verbatim from the
  ``m8_frozen_pp_parameters`` block of
  ``config/v2_singlet/legacy_v1_manifest.json`` (its canonical sha256
  recorded in ``provenance.json``); in particular
  ``continuum.match_mass = 2.680194110727528 GeV`` is the kernel
  boundary (round 2 index item 7), inclusive on the new-kernel side.
- **Bit-identity check**: the first 10,000 events of replica 0
  regenerated with the frozen legacy staged builder at seed 20300913
  and compared event by event against the new builder: identical
  topology, exchange masses, spectral states, and (for sources above
  the boundary) identical final-particle lists;
  ``bit_identity_report.json`` records the per-layer match counts.
- **Identified observable set**: the top-5 entries of the sealed R11
  ranked residual map
  (``dev/00runs/r11_observable_decomposition_2026-09-11/
  ranked_residual_map.json``), read from the sealed archive at
  pre-registration (the set and its SHA-256 are recorded).
- **Comparison metrics** (cross sections are the invariant cross
  sections normalized to the inelastic cross section, ``f /
  sigma_inel``, in the paper windows of ``helpers/benchmarking/
  sps_core.py``; ``sigma_inel = 31.78 mb``):
  - (i) inclusive stability: per species and paper window, the R23
    pooled value agrees with the sealed R01 value within the combined
    4-sigma of the two archives.
  - (ii) identified set: the top-5 identified observables' normalized
    residuals against the training tables: every bin has
    ``|Delta/sigma| <= 2.5`` (no gross violation introduced), and the
    sum of ``|Delta/sigma|`` over the set is no larger than R01's sum
    by more than the combined 2-sigma (not worse, at the 2-sigma
    level; the combined uncertainty of a pull difference is the
    MC-noise part of the two archives' per-cell errors).
  - (iii) feed-down diagnostics: the per-species resonant feed-down
    fraction of the new-kernel events matches the catalog analytic
    expectation (the catalog BRs and width law evaluated at the
    recorded boundary mass distribution) within 3-sigma.
- **No fitting**: ``no_fitting = true``; no optimizer is invoked; the
  branching fractions and widths are the external inputs of plan
  section 8.2 (plan line 572: "no refit in the first pass").

Per-replica records (``replica_<seed>.json``): the pooled per-species
binned weight sums and track counts (the ``sps_core`` paper-window
tables, the R01 binning), the per-species resonant feed-down fractions
of the new-kernel events (the card-6 lineage: every final particle
carries its direct parent), the boundary statistics (the per-source
kernel-switch decisions and the fraction of sources at or below the
boundary, plus the below-boundary mass histogram the analytic
feed-down expectation is evaluated over), and the conservation record
(exact per-event ``B = 2`` and ``Q = 2``).

Acceptance evidence (``gate2_r23.json``): evidence only.  The
``local_conservation`` / ``branching_test`` /
``threshold_discontinuity`` clauses fold the sealed R20/R21/R22
acceptance records (their exact values are recorded in each clause's
``fold`` block) with R23's own conservation evidence; the
``observables`` clause carries R23's metrics (i)-(iii) and requires a
passing bit-identity check.  A clause's ``pass`` is the conjunction of
its folded values and R23's own contribution, so the written
retain / revise / reject decision (the Gate 2 follow-up obligation,
round 2 index card) interprets truthful evidence, never a template.
``no_branching_fit`` records ``no_fitting = true`` and
``optimizer_invoked = false``.

``comparison_r01.json`` and ``gate2_r23.json`` carry no wall-clock
fields: their bytes are a pure function of the sealed inputs, which is
what the clean-environment validation re-hashes.

Dual-tree driver: byte-identical in the nested tree
(``dev/mixing2006/helpers/benchmarking/``) and the monorepo mirror
(``helpers/benchmarking/``); both locate the package with the
established ``_locate_roots`` pattern.

Hermetic: the committed catalog, the two bundled legacy_v1 configs,
the training SPS tables, and the sealed R01/R11 archives; no event
generation outside the declared modes, no external services.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import multiprocessing
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

# ---------------------------------------------------------------------------
# Tree location: this driver runs from the nested tree
# (dev/mixing2006/helpers/benchmarking/) and from its byte-identical
# monorepo mirror (helpers/benchmarking/).  Both must find the
# na49_mixing package, the resonance catalog, and the R01/R11 drivers.
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
    """Print a refusal to stderr and exit with code 2 before the step."""
    print(f"staged_r23_resonance_ablation: {message}", file=sys.stderr, flush=True)
    raise SystemExit(2)


NESTED_ROOT, MONO_ROOT = _locate_roots()
if str(NESTED_ROOT) not in sys.path:
    sys.path.insert(0, str(NESTED_ROOT))

from helpers.benchmarking import staged_observable_decomposition as r11  # noqa: E402
from helpers.benchmarking import staged_r01_reference as r01  # noqa: E402
from na49_mixing.continuum_channels import minimum_excited_mass  # noqa: E402
from na49_mixing.event_builder_pp import PPEventBuilder  # noqa: E402
from na49_mixing.fit_likelihood import combined_covariance  # noqa: E402
from na49_mixing.fit_observables import (  # noqa: E402
    combination_by_name,
    evaluate_combination,
)
from na49_mixing.hadron_catalog import HadronCatalog  # noqa: E402
from na49_mixing.pdg_database import get_particle  # noqa: E402
from na49_mixing.quantum_numbers import QuantumNumbers, quantum_numbers  # noqa: E402
from na49_mixing.reaction import Reaction  # noqa: E402
from na49_mixing.resonance_decay import ResonanceDecayer  # noqa: E402
from na49_mixing.resonance_pp_builder import (  # noqa: E402
    KERNEL_BOUNDARY_M8,
    KERNEL_LABEL,
    LEGACY_LABEL,
    ROUND2_MASTER_SEED,
    ResonancePPBuilder,
)

# ---------------------------------------------------------------------------
# Pre-registered run identity (card TASK-20260913-r2-run-r23-resonance-
# only-pp-ablation; the seed rule, statistics, boundary, and comparison
# metrics are verbatim from the card and the round 2 index items 7 and 12).
# ---------------------------------------------------------------------------

RUN_ID = "R23"
CARD = "TASK-20260913-r2-run-r23-resonance-only-pp-ablation"
RUN_NAME = "r23_resonance_only_pp_ablation_2026-09-13"
DEFAULT_RUN_DIR = "dev/00runs/r23_resonance_only_pp_ablation_2026-09-13"
GATE_RECORD_NAME = "gate2_r23.json"
GATE_CLAUSE = "observables_and_no_branching_fit_round2"

PREREGISTRATION_NAME = "pre_registration.json"
PROVENANCE_NAME = "provenance.json"
BIT_IDENTITY_NAME = "bit_identity_report.json"
REPLICA_SEEDS_NAME = "replica_seeds.json"
COMPARISON_NAME = "comparison_r01.json"
CLEAN_ENV_NAME = "clean_environment_commands.json"
SHA256SUMS_NAME = "SHA256SUMS.txt"
SEAL_NAME = "seal_manifest.json"
REPLICA_FMT = "replica_{}.json"

CATALOG_REL = "config/v2_singlet/resonance_catalog.json"
DRIVER_REL = "helpers/benchmarking/staged_r23_resonance_ablation.py"

#: The R01/R11 sealed archives and the R20-R22 sealed acceptance
#: records (read-only inputs).
R01_RUN_DIR = "dev/00runs/r01_legacy_pp_reference_2026-09-11"
R11_RUN_DIR = "dev/00runs/r11_observable_decomposition_2026-09-11"
R20_GATE_REL = "dev/00runs/r20_single_resonance_unit_grid_2026-09-13/gate2_r20.json"
R21_GATE_REL = "dev/00runs/r21_baryonic_source_grid_2026-09-13/gate2_r21.json"
R22_GATE_REL = "dev/00runs/r22_mesonic_source_grid_2026-09-13/gate2_r22.json"

#: T2 statistics (plan section 5.1; round 2 index item 7): 8 replicas x
#: 100,000 p+p events per replica = 800,000 events (the upper T2 bound,
#: consistent with R02's 100,000-event replicas).
T2_REPLICAS = 8
T2_EVENTS_PER_REPLICA = 100_000
T2_LABEL = "T2: 8 replicas x 100,000 events (plan section 5.1, upper bound)"

#: The replica seeds: the M8 seed rule ``seed + seed_offset + k`` with
#: the Round 2 master seed 20260913 (20300913 to 20300920).
SEED_BASE = 20260913
SEED_OFFSET = 40000
REPLICA_SEEDS = [SEED_BASE + SEED_OFFSET + k for k in range(T2_REPLICAS)]

#: The kernel-layer seed triple (the Round 2 seed convention; the
#: legacy source/exchange layers carry the replica seed above).
KERNEL_MASTER_SEED = ROUND2_MASTER_SEED

#: The M8-frozen kernel boundary (inclusive on the new-kernel side).
KERNEL_BOUNDARY = KERNEL_BOUNDARY_M8

SIGMA_INEL_MB = 31.78
PAPER_WINDOW = {"xF": [0.0, 0.4], "pT": [0.05, 1.5]}

#: The bit-identity check: the first 10,000 events of replica 0 at the
#: replica-0 seed, legacy vs new builder, event by event.
BIT_IDENTITY_SEED = REPLICA_SEEDS[0]
BIT_IDENTITY_EVENTS = 10_000

#: The comparison tolerances (the card's pre-registered metrics).
INCLUSIVE_N_SIGMA = 4.0
IDENTIFIED_MAX_ABS_PULL = 2.5
IDENTIFIED_SUM_N_SIGMA = 2.0
FEED_DOWN_N_SIGMA = 3.0
FEED_DOWN_MIN_N = 1_000

#: The below-boundary mass histogram (the analytic feed-down input).
MASS_HISTOGRAM_BINS = 32

CHUNK_SIZE = 2500
DEFAULT_WORKERS = 4

#: The clean-environment command keys recorded verbatim.
CLEAN_ENV_PYTEST_KEY = "cd dev/mixing2006 && python -m pytest"
CLEAN_ENV_BUILDER_KEY = (
    "cd dev/mixing2006 && python -m pytest tests/test_resonance_pp_builder.py -q"
)
CLEAN_ENV_SMOKE_KEY = (
    "na49-staged-excitation --config config/staged_excitation_smoke.json"
)

#: The event-seed rules recorded in the pre-registration.
LEGACY_EVENT_SEED_RULE = (
    "numpy.random.default_rng(numpy.random.SeedSequence(seed, "
    "spawn_key=(event_id,))); the legacy staged p+p convention (the R01 "
    "driver's rule); one shared generator serves the topology selection, "
    "exchange kinematics, and spectral-state layers and the legacy "
    "decay branch"
)
KERNEL_SEED_RULE = (
    "derive_substreams(20260913, replica_index, event_id) on the closed "
    "decay_channel / decay_kinematics labels for the card-6/7 kernel "
    "and its spawned mixture child (card-9 scope decision #4); no "
    "extension of SUBSTREAM_LABELS"
)

#: The bit-identity layer definitions (recorded verbatim).
LAYER_DEFINITIONS = {
    "topology": (
        "the per-event source list (source_id, role, (B, Q, S)); the "
        "two-source (B, Q, S) = (1, 1, 0) topology of PPEventBuilder, "
        "exact equality"
    ),
    "exchange": (
        "the sampled exchange masses and the momentum transfer t (the "
        "exchange_kinematics layer), exact float equality"
    ),
    "spectral_state": (
        "the realized source four-momenta (the spectral-state draws "
        "propagated through the exact two-body scattering), exact "
        "component-wise float equality"
    ),
    "legacy_kernel_finals": (
        "the final-particle lists of the sources above the boundary "
        "(pdg_id, four-momentum, channel family), exact equality; the "
        "legacy branch is bit-identical to the legacy builder at the "
        "same seed"
    ),
}

#: The smoke mode defaults (driver validation; never sealed).
SMOKE_EVENTS = 2_000


def _now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _write_json(path: Path, payload: dict) -> None:
    """UTF-8 JSON with the R01 sentinel convention (identical inputs
    produce identical bytes; non-finite floats become string
    sentinels)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(r01._json_safe(payload), indent=2, allow_nan=False) + "\n"
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def _canonical_json_bytes(payload: dict) -> bytes:
    """The exact byte sequence :func:`_write_json` writes for *payload*."""
    return (
        json.dumps(r01._json_safe(payload), indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(payload)


def _sha256_file(path: Path) -> str:
    return r01.sha256_file(path)


def _canonical_json_sha256(obj: Any) -> str:
    return r01.canonical_json_sha256(obj)


def _read_json(path: Path) -> dict:
    """Load a JSON artifact, restoring the non-finite string sentinels."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _restore_sentinels(payload)


def _restore_sentinels(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _restore_sentinels(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore_sentinels(v) for v in value]
    if isinstance(value, str):
        if value == "Infinity":
            return math.inf
        if value == "-Infinity":
            return -math.inf
        if value == "NaN":
            return math.nan
    return value


def _git_info() -> dict:
    """The recorded git identities (defensive: a checkout without a
    working git records ``None``)."""
    info: dict = {}
    for name, cwd in (("nested_head", NESTED_ROOT), ("outer_head", MONO_ROOT)):
        try:
            head = subprocess.run(
                ["git", "-C", str(cwd), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "-C", str(cwd), "status", "--porcelain"],
                capture_output=True, text=True, check=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            info[name] = None
            info[name.replace("head", "dirty_entries")] = None
        else:
            info[name] = head
            info[name.replace("head", "dirty_entries")] = len(
                dirty.splitlines()
            )
    return info


def environment_fingerprint() -> dict:
    return {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": sys.platform,
        "numpy_version": np.__version__,
    }


def _load_catalog() -> HadronCatalog:
    return HadronCatalog.from_json(NESTED_ROOT / CATALOG_REL)


def _load_raw_config() -> dict:
    path = NESTED_ROOT / "config" / r01.PRODUCTION_CONFIG_NAME
    if not path.is_file():
        _refuse(f"production config missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _species_key(name: str) -> str:
    """The card-9 species-key transliteration (``+`` -> ``_plus``, ``-``
    -> ``_minus``, a trailing ``0`` -> ``_0``), otherwise verbatim."""
    if name.endswith("+"):
        return name[:-1] + "_plus"
    if name.endswith("-"):
        return name[:-1] + "_minus"
    if name.endswith("0"):
        return name[:-1] + "_0"
    return name


def _record_name(catalog: HadronCatalog, pdg_id: int) -> str:
    """The species name of a final particle: the catalog record's name,
    else the vendored PDG table's, the bare id when neither resolves."""
    try:
        return str(catalog.get(int(pdg_id)).name)
    except KeyError:
        entry = get_particle(int(pdg_id))
        if entry is None:
            return str(int(pdg_id))
        return str(entry.name)


#: Cache of the on-shell reference masses of stable daughters.
_STABLE_MASS_CACHE: dict[int, float] = {}


def _stable_reference_mass(catalog: HadronCatalog, pdg: int) -> float:
    """The on-shell reference mass of a stable daughter (the catalog
    pole mass, else the vendored PDG table's mass)."""
    cached = _STABLE_MASS_CACHE.get(int(pdg))
    if cached is not None:
        return cached
    try:
        mass = float(catalog.get(int(pdg)).pole_mass)
    except KeyError:
        entry = get_particle(int(pdg))
        if entry is None:
            raise KeyError(
                f"daughter pdg_id {pdg} is neither a catalog record nor "
                "in the vendored PDG table"
            ) from None
        mass = float(entry.mass)
    _STABLE_MASS_CACHE[int(pdg)] = mass
    return mass


# ---------------------------------------------------------------------------
# The worker context (fork-inherited; no event payload crosses the
# process boundary) and the chunked generation
# ---------------------------------------------------------------------------

#: Module-level worker context, set once in the parent before the fork
#: pool is created (the R01 ``_generate_chunk`` convention).  ``None``
#: in the parent main path.
_WORKER_CONTEXT: dict[str, Any] | None = None


def _set_worker_context(context: dict[str, Any]) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


def _mass_histogram_edges() -> np.ndarray:
    """The below-boundary mass histogram edges: the baryonic source
    threshold (``minimum_excited_mass((1, 1, 0))``) to the M8-frozen
    kernel boundary."""
    minimum = minimum_excited_mass(QuantumNumbers(1, 1, 0))
    return np.linspace(float(minimum), float(KERNEL_BOUNDARY),
                       MASS_HISTOGRAM_BINS + 1)


def _empty_boundary() -> dict:
    return {
        "n_sources": 0,
        "n_new_kernel": 0,
        "n_legacy_kernel": 0,
        "mass_sum_new_kernel": 0.0,
        "mass_histogram": [0] * MASS_HISTOGRAM_BINS,
    }


def _empty_feeddown(species_keys) -> dict:
    return {
        "n_finals": 0,
        "per_species": {
            key: {"n": 0, "n_resonant": 0, "by_parent": {}}
            for key in species_keys
        },
    }


def _accumulate_feeddown(
    feeddown: dict,
    catalog: HadronCatalog,
    event,
    record,
    vertices,
) -> None:
    """One new-kernel source's finals into the feed-down accumulators.

    The card-6 lineage: ``final.parent_id`` is the index of the
    producing vertex in the source's vertex list; a final particle is
    *resonant feed-down* when its producing vertex's parent is a
    resonance other than the drawn root record.  A final with
    ``parent_id is None`` is a direct source product (the stable
    identity path), never a resonance feed-down.
    """
    record_pdg = int(record.record_pdg)
    for final in event.final_particles:
        if int(final.source_id) != int(record.source_id):
            continue
        key = _species_key(_record_name(catalog, int(final.pdg_id)))
        slot = feeddown["per_species"][key]
        slot["n"] += 1
        feeddown["n_finals"] += 1
        if final.parent_id is None:
            continue
        parent_pdg = int(vertices[int(final.parent_id)].parent_pdg)
        if parent_pdg != record_pdg:
            slot["n_resonant"] += 1
            bucket = slot["by_parent"]
            bucket[str(parent_pdg)] = bucket.get(str(parent_pdg), 0) + 1


def _r23_generate_chunk(task: tuple):
    """Generate one chunk of events in a worker process and bin it.

    Returns ``(start_event_id, event_hashes, sums, counts, boundary,
    feeddown, n_events)``.  ``sums``/``counts`` hold the per-species
    binned weight sums and track counts at the table bins (the R01
    ``_bin_track_sums`` convention); ``boundary`` holds the kernel-switch
    statistics and the below-boundary mass histogram; ``feeddown`` holds
    the per-species resonant feed-down counts of the new-kernel events.
    """
    assert _WORKER_CONTEXT is not None
    seed, event_ids = task
    tables = _WORKER_CONTEXT["tables"]
    catalog = _WORKER_CONTEXT["catalog"]
    pp_config = _WORKER_CONTEXT["pp_config"]
    hist_edges: np.ndarray = _WORKER_CONTEXT["hist_edges"]
    n_bins_hist = int(hist_edges.size - 1)
    hist_lo = float(hist_edges[0])
    hist_width = float(hist_edges[1] - hist_edges[0])

    hashes: list[str] = []
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    boundary = _empty_boundary()
    feeddown = _empty_feeddown(_WORKER_CONTEXT["species_keys"])
    for event_id in event_ids:
        builder = ResonancePPBuilder(
            pp_config,
            catalog,
            seed=int(seed),
            event_id=int(event_id),
            master_seed=KERNEL_MASTER_SEED,
            replica_index=0,
        )
        result = builder.generate(stage="r23_resonance_ablation")
        event = result.event
        hashes.append(event.as_dict()["event_hash"])
        chunk_sums, chunk_counts = r01._bin_track_sums(
            [event], tables, builder.sqrt_s
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
        for record, vertices in zip(result.source_records,
                                    result.kernel_vertices):
            boundary["n_sources"] += 1
            if record.kernel == KERNEL_LABEL:
                boundary["n_new_kernel"] += 1
                mass = float(record.mass)
                boundary["mass_sum_new_kernel"] += mass
                width = float(hist_edges[1] - hist_edges[0])
                index = int((mass - float(hist_edges[0])) / width)
                index = min(max(index, 0), n_bins_hist - 1)
                boundary["mass_histogram"][index] += 1
            else:
                boundary["n_legacy_kernel"] += 1
            if record.kernel != KERNEL_LABEL or vertices is None:
                continue
            _accumulate_feeddown(feeddown, catalog, event, record, vertices)
    return (
        int(event_ids[0]),
        hashes,
        sums,
        counts,
        boundary,
        feeddown,
        len(event_ids),
    )


def _map_chunks(tasks: list, workers: int, progress_label: str):
    """Yield worker results as they complete over a fork worker pool."""
    context = multiprocessing.get_context("fork")
    completed = 0
    started = time.perf_counter()
    with context.Pool(processes=workers) as pool:
        for result in pool.imap_unordered(
            _r23_generate_chunk, tasks, chunksize=1
        ):
            completed += 1
            yield result
            if completed % 10 == 0 or completed == len(tasks):
                elapsed = time.perf_counter() - started
                print(
                    f"[r23] {progress_label}: chunk {completed}/{len(tasks)} "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )


# ---------------------------------------------------------------------------
# Pre-registration
# ---------------------------------------------------------------------------


def _r11_ranked_rows() -> tuple[list[dict], str]:
    """The sealed R11 ranked residual map rows and the archive's
    ``SHA256SUMS.txt`` digest."""
    map_path = MONO_ROOT / R11_RUN_DIR / "ranked_residual_map.json"
    if not map_path.is_file():
        _refuse(f"sealed R11 ranked residual map missing: {map_path}")
    payload = _read_json(map_path)
    rows = payload.get("rows") or []
    if len(rows) < 5:
        _refuse(
            f"the sealed R11 ranked residual map carries {len(rows)} "
            "rows; the identified set needs the top-5 entries"
        )
    sums_path = MONO_ROOT / R11_RUN_DIR / "SHA256SUMS.txt"
    if not sums_path.is_file():
        _refuse(f"sealed R11 archive carries no {SHA256SUMS_NAME}: {sums_path}")
    return [dict(rows[i]) for i in range(5)], _sha256_file(sums_path)


def build_pre_registration(verification: dict) -> dict:
    """The R23 pre-registration payload, written before any generation."""
    rows, r11_seal = _r11_ranked_rows()
    return {
        "schema": "na49_mixing.staged_r23_resonance_ablation.pre_registration/1",
        "run": RUN_ID,
        "card": CARD,
        "written_before_generation": True,
        "statistics": {
            "tier": "T2",
            "label": T2_LABEL,
            "replicas": T2_REPLICAS,
            "events_per_replica": T2_EVENTS_PER_REPLICA,
            "total_events": T2_REPLICAS * T2_EVENTS_PER_REPLICA,
        },
        "replica_seeds": [int(s) for s in REPLICA_SEEDS],
        "seed_rule": {
            "replica_seed_rule": (
                "replica k (k = 0..7) runs at seed "
                f"{SEED_BASE} + {SEED_OFFSET} + k = "
                f"{REPLICA_SEEDS[0]} to {REPLICA_SEEDS[-1]}, the M8 seed "
                "rule of config/staged_excitation.json with the Round 2 "
                "master seed 20260913"
            ),
            "legacy_event_seed_rule": LEGACY_EVENT_SEED_RULE,
            "kernel_seed_rule": KERNEL_SEED_RULE,
        },
        "parameters": {
            "source": (
                "config/v2_singlet/legacy_v1_manifest.json, "
                "frozen_fit_outputs.m8_frozen_pp_parameters (verbatim)"
            ),
            "m8_parameter_hash": r01.M8_PARAMETER_HASH,
            "m8_frozen_pp_parameters": dict(r01.M8_FROZEN_PP_PARAMETERS),
            "kernel_boundary_gev": float(KERNEL_BOUNDARY),
            "boundary_note": (
                "continuum.match_mass of the M8 frozen set is the kernel "
                "boundary (round 2 index item 7); sources at or below "
                "the boundary decay through the catalog-driven card-6/7 "
                "kernel (inclusive), sources above it keep the legacy "
                "SourceDecayer (H3) bit-identical at the same seed; the "
                "current config's continuum.match_mass = 2.4 is the "
                "pre-M8 value and is not used by R23"
            ),
        },
        "bit_identity_check": {
            "seed": int(BIT_IDENTITY_SEED),
            "n_events": int(BIT_IDENTITY_EVENTS),
            "layers_checked": list(LAYER_DEFINITIONS),
            "layer_definitions": dict(LAYER_DEFINITIONS),
            "convention": (
                "the first 10,000 events of replica 0 are regenerated "
                "with the frozen legacy staged builder at the replica-0 "
                "seed and compared event by event against the new "
                "builder; a mismatch stops the run before any replica "
                "is generated"
            ),
        },
        "observable_set": {
            "window": dict(PAPER_WINDOW),
            "sigma_inel_mb": SIGMA_INEL_MB,
            "species": dict(r01.SPECIES_BIN_COUNTS),
            "total_bins": r01.TOTAL_BINS,
            "binning": (
                "the 1017-bin six-species paper-window binned tables "
                "(the R01 convention; the pooled estimator M_bin = "
                "sum_sums / (pi * sqrt(s) * N_total * dxF * dpT))"
            ),
        },
        "identified_set": {
            "source": (
                f"{R11_RUN_DIR}/ranked_residual_map.json (sealed), the "
                "top-5 entries by |pull|"
            ),
            "sha256_r11_sums": r11_seal,
            "entries": [
                {
                    "rank": int(row["rank"]),
                    "sector": str(row["sector"]),
                    "observable": str(row["observable"]),
                    "region": str(row["region"]),
                    "bin_index": int(row["bin_index"]),
                    "d_r01": float(row["d"]),
                    "m_r01": float(row["m"]),
                    "sigma_r01": float(row["sigma"]),
                    "delta_over_sigma_r01": float(row["delta_over_sigma"]),
                }
                for row in rows
            ],
        },
        "comparison_metrics": {
            "inclusive_stability": {
                "clause": (
                    "(i) inclusive stability: per species and paper "
                    "window the R23 value agrees with the sealed R01 "
                    "value within the combined 4-sigma of the two "
                    "archives (sigma = sqrt(stat_r01^2 + stat_r23^2))"
                ),
                "n_sigma": INCLUSIVE_N_SIGMA,
            },
            "identified_set": {
                "clause": (
                    "(ii) identified set: every top-5 bin has "
                    "|Delta/sigma| <= 2.5 against the training tables "
                    "(no gross violation introduced), and the sum of "
                    "|Delta/sigma| over the set is no larger than R01's "
                    "sum by more than the combined 2-sigma (not worse, "
                    "at the 2-sigma level; the combined uncertainty of "
                    "a pull difference is the MC-noise part of the two "
                    "archives' per-cell errors)"
                ),
                "max_abs_pull": IDENTIFIED_MAX_ABS_PULL,
                "sum_n_sigma": IDENTIFIED_SUM_N_SIGMA,
            },
            "feed_down": {
                "clause": (
                    "(iii) feed-down diagnostics: the per-species "
                    "resonant feed-down fraction of the new-kernel "
                    "events matches the catalog analytic expectation "
                    "(the catalog BRs and the card-3/4 width law "
                    "evaluated at the recorded boundary mass "
                    "distribution) within 3-sigma"
                ),
                "n_sigma": FEED_DOWN_N_SIGMA,
                "min_n_per_species": FEED_DOWN_MIN_N,
                "mass_histogram_bins": MASS_HISTOGRAM_BINS,
                "analytic_note": (
                    "the analytic chain expectation walks the catalog "
                    "from the drawn root record at the bin mass (the "
                    "card-6 open-weight normalization over the open "
                    "records at that mass) through the daughter chain "
                    "at pole masses; the observed side counts the "
                    "card-6 lineage of the new-kernel events (a final "
                    "particle is resonant when its producing vertex's "
                    "parent is a resonance other than the drawn root "
                    "record)"
                ),
            },
        },
        "no_fitting": True,
        "optimizer_invoked": False,
        "no_fitting_note": (
            "no optimizer is invoked; the branching fractions and "
            "widths are the external inputs of plan section 8.2 (plan "
            "line 572: no refit in the first pass)"
        ),
        "m8_parameter_hash": r01.M8_PARAMETER_HASH,
        "config_hash": verification["config_sha256"],
        "tables_sha256": verification["tables_sha256"],
        "builder_module_sha256": verification["builder_module_sha256"],
        "driver_sha256": verification["driver_sha256"],
        "catalog_sha256": verification["catalog_sha256"],
    }


def run_pre_registration(run_dir: Path) -> None:
    """Write ``pre_registration.json`` and ``provenance.json``, stop."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    pre_path = run_dir / PREREGISTRATION_NAME
    if pre_path.is_file() and any(run_dir.glob(REPLICA_FMT.format("*"))):
        _refuse(
            f"{pre_path} exists and replica records are present; the "
            "pre-registration is immutable after generation has started "
            "(a change invalidates the run; re-seal under a new date)"
        )
    config_path = NESTED_ROOT / "config" / r01.PRODUCTION_CONFIG_NAME
    if not config_path.is_file():
        _refuse(f"production config missing: {config_path}")
    config_sha = _sha256_file(config_path)
    raw_config = _load_raw_config()
    tables = r01.load_binned_tables(raw_config)
    tables_sha = r01.tables_digest(tables)
    catalog_sha = _sha256_file(NESTED_ROOT / CATALOG_REL)
    builder_sha = _sha256_file(
        NESTED_ROOT / "na49_mixing" / "resonance_pp_builder.py"
    )
    driver_sha = _sha256_file(Path(__file__).resolve())
    _rows, r11_sums = _r11_ranked_rows()
    verification = {
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "tables_sha256": r01.tables_digest(
            r01.load_binned_tables(json.loads(
                config_path.read_text(encoding="utf-8")
            ))
        ),
        "m8_parameter_hash": r01.M8_PARAMETER_HASH,
        "catalog_sha256": catalog_sha,
        "builder_module_sha256": builder_sha,
        "driver_sha256": driver_sha,
        "r11_sums_sha256": r11_sums,
    }
    # The sealed R01 archive is a verified read-only input.
    r01_archive = r11.verify_r01_archive(MONO_ROOT / R01_RUN_DIR)
    verification["r01_archive_seal_sha256"] = r01_archive["archive_seal_sha256"]
    verification["r01_replica_seeds"] = r01_archive["seeds"]
    verification["r01_config_sha256"] = r01_archive["config_sha256"]
    verification["r01_events_per_replica"] = r01_archive["events_per_replica"]
    pre = build_pre_registration(verification)
    _write_json(pre_path, pre)
    provenance = {
        "schema": "na49_mixing.staged_r23_resonance_ablation.provenance/1",
        "run": RUN_ID,
        "card": CARD,
        "driver": DRIVER_REL,
        "driver_sha256": driver_sha,
        "git": _git_info(),
        "environment": environment_fingerprint(),
        "written_utc": _now_utc(),
        "written_before_generation": True,
        "verification": verification,
        "catalog": {
            "path": CATALOG_REL,
            "sha256": catalog_sha,
        },
        "m8_parameter_hash": r01.M8_PARAMETER_HASH,
        "kernel_boundary_gev": float(KERNEL_BOUNDARY),
        "verification": verification,
    }
    _write_json(run_dir / PROVENANCE_NAME, provenance)
    print(
        "staged_r23_resonance_ablation: pre-registration written to "
        f"{pre_path} (8 replicas x 100,000 events, seeds "
        f"{REPLICA_SEEDS[0]}..{REPLICA_SEEDS[-1]}, kernel boundary "
        f"{KERNEL_BOUNDARY!r} GeV, identified set from the sealed R11 "
        "top-5); nothing generated"
    )


def _load_pre_registration(run_dir: Path) -> dict:
    path = Path(run_dir) / PREREGISTRATION_NAME
    if not path.is_file():
        _refuse(
            f"{path} is missing; the run requires the pre-registration "
            "(run --pre-registration first)"
        )
    return _read_json(path)


def _assert_pre_registration(run_dir: Path) -> dict:
    """Assert the written pre-registration against this driver (a
    mismatch invalidates the run)."""
    pre = _load_pre_registration(run_dir)
    problems: list[str] = []
    if int(pre.get("statistics", {}).get("replicas", -1)) != T2_REPLICAS:
        problems.append("statistics.replicas mismatch")
    if int(pre.get("statistics", {}).get(
            "events_per_replica", -1)) != T2_EVENTS_PER_REPLICA:
        problems.append("statistics.events_per_replica mismatch")
    if [int(s) for s in pre.get("replica_seeds", [])] != [
            int(s) for s in REPLICA_SEEDS]:
        problems.append("replica_seeds mismatch")
    if pre.get("seed_rule", {}).get(
            "legacy_event_seed_rule") != LEGACY_EVENT_SEED_RULE:
        problems.append("legacy_event_seed_rule mismatch")
    if pre.get("seed_rule", {}).get("kernel_seed_rule") != KERNEL_SEED_RULE:
        problems.append("kernel_seed_rule mismatch")
    if float(pre.get("parameters", {}).get(
            "kernel_boundary_gev", -1.0)) != float(KERNEL_BOUNDARY):
        problems.append("kernel_boundary mismatch")
    if pre.get("parameters", {}).get(
            "m8_frozen_pp_parameters") != r01.M8_FROZEN_PP_PARAMETERS:
        problems.append("m8 parameter block mismatch")
    if int(pre.get("bit_identity_check", {}).get("seed", -1)) != (
            BIT_IDENTITY_SEED):
        problems.append("bit_identity seed mismatch")
    if int(pre.get("bit_identity_check", {}).get("n_events", -1)) != (
            BIT_IDENTITY_EVENTS):
        problems.append("bit_identity n_events mismatch")
    entries = pre.get("identified_set", {}).get("entries", [])
    rows, _ = _r11_ranked_rows()
    if len(entries) != 5:
        problems.append("identified_set needs exactly 5 entries")
    else:
        for entry, row in zip(entries, rows):
            if (int(entry["rank"]) != int(row["rank"])
                    or str(entry["sector"]) != str(row["sector"])
                    or str(entry["observable"]) != str(row["observable"])
                    or str(entry["region"]) != str(row["region"])
                    or int(entry["bin_index"]) != int(row["bin_index"])):
                problems.append(
                    f"identified_set entry rank {entry['rank']} does not "
                    "match the sealed R11 top-5"
                )
    if problems:
        raise RuntimeError(
            "staged_r23_resonance_ablation: the written pre-registration "
            "does not match this driver (the run is invalidated):\n  "
            + "\n  ".join(problems)
        )
    return pre


# ---------------------------------------------------------------------------
# Mode: --bit-identity
# ---------------------------------------------------------------------------


def _event_payload(event) -> dict:
    """A plain-dict projection of one event for the comparisons."""
    return {
        "sources": [
            {
                "source_id": int(s.source_id),
                "role": str(s.role),
                "quantum_numbers": s.quantum_numbers.as_dict(),
                "metadata": dict(s.metadata),
                "p4": s.p4.as_dict(),
            }
            for s in event.sources
        ],
        "final_particles": [
            {
                "pdg_id": int(p.pdg_id),
                "p4": p.p4.as_dict(),
                "channel_family": str(p.channel_family),
                "source_id": int(p.source_id),
            }
            for p in event.final_particles
        ],
    }


def _finals_of_source(event_payload: dict, source_id: int) -> list[tuple]:
    """The final particles of one source as comparable tuples."""
    return [
        (
            int(p["pdg_id"]),
            tuple(p["p4"].values()),
            str(p["channel_family"]),
        )
        for p in event_payload["final_particles"]
        if int(p["source_id"]) == int(source_id)
    ]


def run_bit_identity(run_dir: Path) -> dict:
    """The pre-registered 10,000-event bit-identity check.

    The first 10,000 events of replica 0 are regenerated with the
    frozen legacy staged builder (the R01 stream: the same builder,
    config, per-event seed rule, and stage label) and compared event by
    event against the new builder.  A mismatch stops the run before
    any replica is generated (exit 1; a mismatch re-opens the builder
    card, not the tolerance).
    """
    pre = _load_pre_registration(run_dir)
    if int(pre["bit_identity_check"]["seed"]) != BIT_IDENTITY_SEED or int(
            pre["bit_identity_check"]["n_events"]) != BIT_IDENTITY_EVENTS:
        _refuse(
            "the written pre-registration's bit-identity block does not "
            "match this driver; a change after generation invalidates "
            "the run"
        )
    raw_config = _load_raw_config()
    pp_config, _applied = r01.build_pp_config(raw_config)
    catalog = _load_catalog()
    config_path = NESTED_ROOT / "config" / r01.PRODUCTION_CONFIG_NAME
    verification = r01.verify_config_and_m8(config_path, pre)

    layers = {
        "topology": {"n_compared": 0, "n_matched": 0, "mismatches": []},
        "exchange": {"n_compared": 0, "n_matched": 0, "mismatches": []},
        "spectral_state": {"n_compared": 0, "n_matched": 0, "mismatches": []},
        "legacy_kernel_finals": {
            "n_compared": 0,
            "n_matched": 0,
            "n_skipped_new_kernel": 0,
            "n_skipped_perturbed": 0,
            "mismatches": [],
        },
    }
    started = _now_utc()
    t0 = time.perf_counter()
    n_perturbed = 0
    for event_id in range(BIT_IDENTITY_EVENTS):
        legacy_builder = PPEventBuilder(
            pp_config,
            np.random.default_rng(
                np.random.SeedSequence(BIT_IDENTITY_SEED, spawn_key=(event_id,))
            ),
        )
        legacy_event = legacy_builder.generate(event_id,
                                               stage=r01.STAGE_LABEL)
        builder = ResonancePPBuilder(
            pp_config,
            catalog,
            seed=BIT_IDENTITY_SEED,
            event_id=event_id,
            master_seed=KERNEL_MASTER_SEED,
            replica_index=0,
        )
        new_result = builder.generate(stage="r23_resonance_ablation")
        new_event = new_result.event

        legacy_topology = [
            (s.source_id, s.role, s.quantum_numbers.as_tuple())
            for s in legacy_event.sources
        ]
        new_topology = [
            (s.source_id, s.role, s.quantum_numbers.as_tuple())
            for s in new_event.sources
        ]
        layers["topology"]["n_compared"] += 1
        if legacy_topology == new_topology:
            layers["topology"]["n_matched"] += 1
        else:
            layers["topology"]["mismatches"].append({"event": event_id})

        legacy_exchange = [
            (float(s.metadata["sampled_mass"]), float(s.metadata["t"]))
            for s in legacy_event.sources
        ]
        new_exchange = [
            (float(s.metadata["sampled_mass"]), float(s.metadata["t"]))
            for s in new_event.sources
        ]
        layers["exchange"]["n_compared"] += 1
        if legacy_exchange == new_exchange:
            layers["exchange"]["n_matched"] += 1
        else:
            layers["exchange"]["mismatches"].append({"event": event_id})

        legacy_spectral = [
            tuple(s.p4.as_dict().values()) for s in legacy_event.sources
        ]
        new_spectral = [
            tuple(s.p4.as_dict().values()) for s in new_event.sources
        ]
        layers["spectral_state"]["n_compared"] += 1
        if legacy_spectral == new_spectral:
            layers["spectral_state"]["n_matched"] += 1
        else:
            layers["spectral_state"]["mismatches"].append({"event": event_id})

        # The legacy-kernel final-particle layer.  Source 0 is always
        # eligible (the shared generator state at its decay is the
        # legacy builder's state); source 1 is eligible only when
        # source 0 also took the legacy branch (a below-boundary
        # source 0 consumes no shared uniform, so the legacy builder's
        # source-1 decay reads a different generator state).
        perturbed = False
        for i in range(2):
            if i == 1 and new_result.source_records[0].kernel == KERNEL_LABEL:
                perturbed = True
                layers["legacy_kernel_finals"]["n_skipped_perturbed"] += 1
                continue
            record = new_result.source_records[i]
            if record.kernel != LEGACY_LABEL:
                layers["legacy_kernel_finals"]["n_skipped_new_kernel"] += 1
                continue
            mine = _finals_of_source(_event_payload(new_event), i)
            theirs = _finals_of_source(_event_payload(legacy_event), i)
            layers["legacy_kernel_finals"]["n_compared"] += 1
            if mine == theirs:
                layers["legacy_kernel_finals"]["n_matched"] += 1
            else:
                layers["legacy_kernel_finals"]["mismatches"].append(
                    {"event": event_id, "source": i}
                )
        if perturbed:
            n_perturbed += 1

    wall = time.perf_counter() - t0
    pass_all = (
        layers["topology"]["n_matched"] == layers["topology"]["n_compared"]
        and layers["exchange"]["n_matched"] == layers["exchange"]["n_compared"]
        and layers["spectral_state"]["n_matched"]
        == layers["spectral_state"]["n_compared"]
        and layers["legacy_kernel_finals"]["n_compared"]
        == layers["legacy_kernel_finals"]["n_matched"]
    )
    report = {
        "schema": "na49_mixing.staged_r23_resonance_ablation.bit_identity/1",
        "run": RUN_ID,
        "card": CARD,
        "seed": int(BIT_IDENTITY_SEED),
        "n_events": int(BIT_IDENTITY_EVENTS),
        "layer_definitions": dict(LAYER_DEFINITIONS),
        "layers": layers,
        "n_events_with_perturbed_source0": int(n_perturbed),
        "perturbation_note": (
            "a below-boundary source 0 consumes no uniform from the "
            "shared generator, so the legacy builder's source-1 decay "
            "reads a different generator state than the new builder's; "
            "those source-1 final lists are skipped "
            "(n_skipped_perturbed) and the topology / exchange / "
            "spectral layers remain fully comparable (they precede the "
            "decays)"
        ),
        "pass": bool(pass_all),
        "config_sha256": verification["config_sha256"],
        "m8_parameter_hash": verification["m8_parameter_hash"],
        "catalog_sha256": _sha256_file(NESTED_ROOT / CATALOG_REL),
        "legacy_stage_label": r01.STAGE_LABEL,
        "legacy_stage_note": (
            "the legacy side regenerates the R01 stream (the same "
            "builder, config, per-event seed rule, and stage label), so "
            "its event hashes equal the sealed R01 replica "
            f"{BIT_IDENTITY_SEED} stream's first {BIT_IDENTITY_EVENTS} "
            "event hashes"
        ),
        "wall_seconds": round(wall, 3),
        "started_utc": started,
        "finished_utc": _now_utc(),
    }
    _write_json(run_dir / BIT_IDENTITY_NAME, report)
    if not pass_all:
        print(
            "staged_r23_resonance_ablation: the bit-identity check "
            "FAILED; the run stops before any replica is generated "
            "(a mismatch re-opens the builder card, not the tolerance)",
            flush=True,
        )
        raise SystemExit(1)
    print(
        "staged_r23_resonance_ablation: bit-identity check passed over "
        f"{BIT_IDENTITY_EVENTS} events at seed {BIT_IDENTITY_SEED} "
        f"({layers['legacy_kernel_finals']['n_compared']} "
        "legacy-kernel sources compared bit for bit, "
        f"{layers['legacy_kernel_finals']['n_skipped_new_kernel']} "
        "new-kernel sources skipped, "
        f"{layers['legacy_kernel_finals']['n_skipped_perturbed']} "
        "perturbed-source-1 skips)",
        flush=True,
    )
    return report


def _assert_bit_identity(run_dir: Path) -> dict:
    path = Path(run_dir) / BIT_IDENTITY_NAME
    if not path.is_file():
        _refuse(
            f"{path} is missing; the run requires the bit-identity check "
            "(run --bit-identity first; a mismatch stops the run before "
            "any replica is generated)"
        )
    report = _read_json(path)
    if not report.get("pass"):
        _refuse(
            f"{path} records pass = false; the run stops (a mismatch "
            "re-opens the builder card, not the tolerance)"
        )
    return report


# ---------------------------------------------------------------------------
# Mode: --generate
# ---------------------------------------------------------------------------


def _table_data_digest(tables: list) -> str:
    """Digest binding the pooled binning to the loaded table data."""
    payload = [
        {
            "species": t.species,
            "pdg_id": int(t.pdg_id),
            "path": str(t.path),
            "sha256": _sha256_file(t.path),
            "n_bins": int(t.data.size),
        }
        for t in tables
    ]
    return _canonical_json_sha256(payload)


def _aggregate_chunks(parts: list, species_keys: tuple):
    """Merge the chunk results into one replica's pooled state."""
    hashes = [
        h for _s, chunk_hashes, _su, _co, _b, _f, _n in parts for h in chunk_hashes
    ]
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    boundary = _empty_boundary()
    feeddown = _empty_feeddown(species_keys)
    for _start, _hashes, part_sums, part_counts, part_boundary, part_feeddown, _n in parts:
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
        for key in ("n_sources", "n_new_kernel", "n_legacy_kernel"):
            boundary[key] += int(part_boundary[key])
        boundary["mass_sum_new_kernel"] += float(
            part_boundary["mass_sum_new_kernel"]
        )
        for i, value in enumerate(part_boundary["mass_histogram"]):
            boundary["mass_histogram"][i] += int(value)
        feeddown["n_finals"] += int(part_feeddown["n_finals"])
        for key, slot in part_feeddown["per_species"].items():
            target = feeddown["per_species"][key]
            target["n"] += int(slot["n"])
            target["n_resonant"] += int(slot["n_resonant"])
            for pdg, cnt in slot["by_parent"].items():
                bucket = target["by_parent"]
                bucket[pdg] = bucket.get(pdg, 0) + int(cnt)
    return hashes, sums, counts, boundary, feeddown


def _replica_record(
    seed: int,
    replica_index: int,
    n_events: int,
    hashes: list[str],
    sums: dict,
    counts: dict,
    boundary: dict,
    feeddown: dict,
    catalog: HadronCatalog,
    workers: int,
    n_chunks: int,
    config_sha: str,
    tables_sha: str,
    table_data_digest: str,
    builder_sha: str,
    driver_sha: str,
    wall: float,
    started_iso: str,
    smoke: bool,
) -> dict:
    """Assemble and write one replica record."""
    n_new = boundary["n_new_kernel"]
    boundary_fraction = (
        float(n_new) / float(boundary["n_sources"])
        if boundary["n_sources"] > 0 else 0.0
    )
    mean_mass = (
        boundary["mass_sum_new_kernel"] / float(n_new) if n_new > 0 else 0.0
    )
    per_species_payload = {}
    fractions = {}
    for key, slot in feeddown["per_species"].items():
        fraction = (
            float(slot["n_resonant"]) / float(slot["n"])
            if slot["n"] > 0 else 0.0
        )
        fractions[key] = fraction
        per_species_payload[key] = {
            "n": int(slot["n"]),
            "n_resonant": int(slot["n_resonant"]),
            "fraction_resonant": fraction,
            "by_parent": {
                str(pdg): int(cnt)
                for pdg, cnt in sorted(slot["by_parent"].items())
            },
        }
    payload_digest = json.dumps(hashes, separators=(",", ":")).encode("utf-8")
    return {
        "seed": int(seed),
        "replica_index": int(replica_index),
        "mode": "smoke" if smoke else "production",
        "n_events": int(n_events),
        "chunk_size": CHUNK_SIZE,
        "chunks": int(n_chunks),
        "worker_count": int(workers),
        "legacy_event_seed_rule": LEGACY_EVENT_SEED_RULE,
        "kernel_seed_rule": KERNEL_SEED_RULE,
        "kernel_master_seed": int(KERNEL_MASTER_SEED),
        "stage_label": "r23_resonance_ablation",
        "event_hash_set_sha256": hashlib.sha256(payload_digest).hexdigest(),
        "config_sha256": config_sha,
        "tables_sha256": tables_sha,
        "table_data_digest": table_data_digest,
        "builder_module_sha256": builder_sha,
        "driver_sha256": driver_sha,
        "kernel_boundary_gev": float(KERNEL_BOUNDARY),
        "boundary": {
            "fraction_new_kernel": boundary_fraction,
            "mean_mass_new_kernel_gev": mean_mass,
            "mass_histogram_edges": [
                float(v) for v in _mass_histogram_edges()
            ],
            "mass_histogram_counts": [
                int(v) for v in boundary["mass_histogram"]
            ],
            "mass_sum_new_kernel": float(boundary["mass_sum_new_kernel"]),
            "n_sources": int(boundary["n_sources"]),
            "n_new_kernel": int(n_new),
            "n_legacy_kernel": int(boundary["n_legacy_kernel"]),
            "fraction_definition": (
                "the fraction of realized sources at or below the kernel "
                "boundary (the new-kernel side, inclusive)"
            ),
        },
        "feed_down": {
            "n_finals_new_kernel": int(feeddown["n_finals"]),
            "fractions_resonant": {
                key: float(v) for key, v in fractions.items()
            },
            "per_species": per_species_payload,
            "lineage_convention": (
                "every final particle of a catalog-driven source carries "
                "its direct parent (parent_id is the index of the "
                "producing vertex; the full ancestry is recoverable from "
                "the vertex list); a final particle is resonant feed-down "
                "when its producing vertex's parent is a resonance other "
                "than the drawn root record; the legacy-branch finals "
                "keep the legacy parent convention and are excluded from "
                "this block"
            ),
        },
        "conservation": {
            "events_checked": int(n_events),
            "event_b_q_conserved": True,
            "convention": (
                "every p+p event conserves total B = 2 and Q = 2 exactly "
                "and every source's final set conserves the source's "
                "(B, Q, S) = (1, 1, 0); checked per event, never "
                "repaired; a violation raises and stops the run"
            ),
        },
        "wall_seconds": round(wall, 3),
        "started_utc": started_iso,
        "finished_utc": _now_utc(),
        "mode": "smoke" if smoke else "production",
        "sums": {s: [float(v) for v in values] for s, values in sums.items()},
        "counts": {
            s: [int(v) for v in values] for s, values in counts.items()
        },
        "reused": False,
    }


def _generate_replica(
    seed: int,
    replica_index: int,
    n_events: int,
    run_dir: Path,
    tables: list,
    catalog: HadronCatalog,
    pp_config,
    workers: int,
    config_sha: str,
    tables_sha: str,
    table_data_digest: str,
    builder_sha: str,
    driver_sha: str,
    smoke: bool,
) -> dict:
    """One seeded replica: chunked generation, pooled per-bin sums,
    boundary statistics, and the feed-down lineage.

    An existing ``replica_<seed>.json`` that matches the event count,
    the config hash, the table digest, the driver identity, and the mode
    is reused unchanged so the declared tier can be completed
    incrementally; anything else is regenerated.
    """
    record_path = Path(run_dir) / REPLICA_FMT.format(int(seed))
    if record_path.is_file():
        try:
            stored = _read_json(record_path)
        except ValueError:
            stored = None
        if (
            isinstance(stored, dict)
            and int(stored.get("seed", -1)) == int(seed)
            and int(stored.get("n_events", -1)) == int(n_events)
            and stored.get("config_sha256") == config_sha
            and stored.get("tables_sha256") == tables_sha
            and stored.get("builder_module_sha256") == builder_sha
            and stored.get("driver_sha256") == driver_sha
            and stored.get("mode") == ("smoke" if smoke else "production")
            and isinstance(stored.get("sums"), dict)
            and isinstance(stored.get("counts"), dict)
            and isinstance(stored.get("feed_down"), dict)
            and isinstance(stored.get("boundary"), dict)
        ):
            stored["reused"] = True
            print(
                f"[r23] replica {seed}: reusing completed record "
                f"({int(stored['n_events'])} events)",
                flush=True,
            )
            return stored

    stable = catalog.stable_states()
    species_keys = tuple(
        _species_key(str(record.name))
        for record in catalog.records()
        if int(record.pdg_id) in stable
    )
    hist_edges = _mass_histogram_edges()
    event_ids = list(range(int(n_events)))
    chunks = [
        event_ids[start : start + CHUNK_SIZE]
        for start in range(0, len(event_ids), CHUNK_SIZE)
    ]
    tasks = [(int(seed), chunk) for chunk in chunks]
    _set_worker_context({
        "tables": tables,
        "catalog": catalog,
        "species_keys": species_keys,
        "stable": frozenset(stable),
        "pp_config": pp_config,
        "hist_edges": hist_edges,
    })
    started = time.perf_counter()
    started_iso = _now_utc()
    parts = sorted(
        _map_chunks(tasks, int(workers), f"replica {seed}"),
        key=lambda part: part[0],
    )
    hashes, sums, counts, boundary, feeddown = _aggregate_chunks(
        parts, species_keys
    )
    wall = time.perf_counter() - started
    record = _replica_record(
        seed,
        replica_index,
        n_events,
        hashes,
        sums,
        counts,
        boundary,
        feeddown,
        catalog,
        workers,
        len(chunks),
        config_sha,
        tables_sha,
        table_data_digest,
        builder_sha,
        driver_sha,
        wall,
        started_iso,
        smoke,
    )
    _write_json(Path(run_dir) / REPLICA_FMT.format(int(seed)), record)
    return record


# ---------------------------------------------------------------------------
# Replica manifest, generation dispatch
# ---------------------------------------------------------------------------


def _update_replica_manifest(run_dir: Path, smoke: bool = False) -> None:
    entries = []
    for index, seed in enumerate(REPLICA_SEEDS):
        path = Path(run_dir) / REPLICA_FMT.format(int(seed))
        if path.is_file():
            record = _read_json(path)
            entries.append({
                "replica_index": int(index),
                "seed": int(seed),
                "status": "completed",
                "n_events": int(record.get("n_events", 0)),
                "mode": str(record.get("mode", "production")),
            })
        else:
            entries.append({
                "replica_index": int(index),
                "seed": int(seed),
                "status": "absent",
            })
    completed = [e for e in entries if e["status"] == "completed"]
    payload = {
        "schema": "na49_mixing.staged_r23_resonance_ablation.replica_manifest/1",
        "card": CARD,
        "declared_replicas": T2_REPLICAS,
        "events_per_replica": (
            int(completed[0]["n_events"]) if completed else T2_EVENTS_PER_REPLICA
        ),
        "mode": "smoke" if smoke else "production",
        "replicas_completed": len(completed),
        "events_completed": int(sum(e["n_events"] for e in completed)),
        "seeds": entries,
    }
    _write_json(Path(run_dir) / REPLICA_SEEDS_NAME, payload)


def run_generation(run_dir: Path, args) -> None:
    """Generate one replica (``--replica K``) or every replica."""
    run_dir = Path(run_dir)
    pre = _load_pre_registration(run_dir)
    smoke = bool(args.smoke)
    if not smoke:
        _assert_bit_identity(run_dir)
    raw_config = _load_raw_config()
    config_path = NESTED_ROOT / "config" / r01.PRODUCTION_CONFIG_NAME
    verification = r01.verify_config_and_m8(config_path, pre)
    tables = r01.load_binned_tables(raw_config)
    tables_sha = r01.tables_digest(tables)
    table_data_digest = _table_data_digest(tables)
    catalog = _load_catalog()
    pp_config, _applied = r01.build_pp_config(raw_config)
    builder_sha = _sha256_file(
        NESTED_ROOT / "na49_mixing" / "resonance_pp_builder.py"
    )
    driver_sha = _sha256_file(Path(__file__).resolve())
    if not smoke:
        if pre.get("builder_module_sha256") != builder_sha:
            _refuse(
                "the committed resonance_pp_builder module sha256 "
                f"{builder_sha!r} differs from the pre-registered "
                f"{pre.get('builder_module_sha256')!r}; the run is "
                "invalidated (re-register under a new date)"
            )
        if pre.get("driver_sha256") != driver_sha:
            _refuse(
                "the committed driver sha256 differs from the "
                "pre-registered driver_sha256; the run is invalidated "
                "(re-register under a new date)"
            )

    n_events = int(args.smoke_events) if smoke else T2_EVENTS_PER_REPLICA
    if args.replica is None:
        indices = list(range(T2_REPLICAS))
    else:
        index = int(args.replica)
        if not 0 <= index < T2_REPLICAS:
            _refuse(
                f"--replica {index} outside the pre-registered set "
                f"0..{T2_REPLICAS - 1}"
            )
        indices = [index]
    for index in indices:
        seed = int(REPLICA_SEEDS[index])
        _generate_replica(
            seed, index, n_events, run_dir, tables, catalog, pp_config,
            int(args.workers), verification["config_sha256"], tables_sha,
            table_data_digest, builder_sha, driver_sha, smoke,
        )
    _update_replica_manifest(run_dir, smoke)
    print(
        f"[r23] generation step complete (run dir {run_dir}, mode "
        f"{'smoke' if smoke else 'production'})",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Pooled comparison inputs (boundary, feed-down, predictions)
# ---------------------------------------------------------------------------


def _load_production_replicas(
    run_dir: Path, pre: dict, config_sha: str, tables_sha: str,
    table_data_digest: str,
) -> dict[int, dict]:
    """The eight production replica records, each asserted against the
    pre-registration."""
    replicas: dict[int, dict] = {}
    for seed in REPLICA_SEEDS:
        path = Path(run_dir) / REPLICA_FMT.format(int(seed))
        if not path.is_file():
            _refuse(
                f"replica record missing: {path}; --compare requires all "
                f"{T2_REPLICAS} production replicas"
            )
        record = _read_json(path)
        identity_ok = (
            int(record.get("seed", -1)) == int(seed)
            and int(record.get("n_events", -1)) == T2_EVENTS_PER_REPLICA
            and record.get("config_sha256") == config_sha
            and record.get("tables_sha256") == tables_sha
            and record.get("table_data_digest") == table_data_digest
            and record.get("builder_module_sha256")
            == pre.get("builder_module_sha256")
            and record.get("driver_sha256") == pre.get("driver_sha256")
            and record.get("mode") == "production"
        )
        if not identity_ok:
            _refuse(
                f"replica record {path} does not match the "
                "pre-registration (seed, event count, config/tables "
                "digests, builder/driver identity, or a smoke record in "
                "a production comparison)"
            )
        replicas[int(seed)] = record
    return replicas


def _pooled_boundary(replicas: dict[int, dict]) -> dict:
    """The pooled boundary statistics and mass histogram."""
    pooled = {
        "n_sources": 0,
        "n_new_kernel": 0,
        "n_legacy_kernel": 0,
        "mass_sum_new_kernel": 0.0,
    }
    histogram = [0] * MASS_HISTOGRAM_BINS
    edges: list[float] | None = None
    for seed in sorted(replicas):
        block = replicas[seed]["boundary"]
        for key in ("n_sources", "n_new_kernel", "n_legacy_kernel"):
            pooled[key] += int(block[key])
        pooled["mass_sum_new_kernel"] += float(block["mass_sum_new_kernel"])
        block_edges = [float(v) for v in block["mass_histogram_edges"]]
        if edges is None:
            edges = block["mass_histogram_edges"]
        elif [float(v) for v in edges] != [
                float(v) for v in block["mass_histogram_edges"]]:
            _refuse("replica boundary histograms disagree on their edges")
        for i, value in enumerate(block["mass_histogram_counts"]):
            histogram[i] += int(value)
    if edges is None:
        edges = [float(v) for v in _mass_histogram_edges()]
    n_new = pooled["n_new_kernel"]
    return {
        "n_sources": int(pooled["n_sources"]),
        "n_new_kernel": int(n_new),
        "n_legacy_kernel": int(pooled["n_legacy_kernel"]),
        "fraction_new_kernel": (
            float(n_new) / float(pooled["n_sources"])
            if pooled["n_sources"] > 0 else 0.0
        ),
        "mean_mass_new_kernel_gev": (
            pooled["mass_sum_new_kernel"] / float(n_new) if n_new else 0.0
        ),
        "mass_histogram_edges": [float(v) for v in edges],
        "mass_histogram_counts": histogram,
        "boundary_gev": float(KERNEL_BOUNDARY),
        "boundary_inclusive": True,
    }


def _pooled_feeddown(replicas: dict[int, dict]) -> dict:
    """The pooled per-species feed-down counts and fractions."""
    per_species: dict[str, dict] = {}
    n_finals = 0
    for seed in sorted(replicas):
        block = replicas[seed]["feed_down"]
        n_finals += int(block["n_finals_new_kernel"])
        for key, slot in block["per_species"].items():
            target = per_species.setdefault(
                key, {"n": 0, "n_resonant": 0, "by_parent": {}}
            )
            target["n"] += int(slot["n"])
            target["n_resonant"] += int(slot["n_resonant"])
            for pdg, cnt in slot.get("by_parent", {}).items():
                bucket = target["by_parent"]
                bucket[str(pdg)] = bucket.get(str(pdg), 0) + int(cnt)
    fractions = {}
    for key, slot in per_species.items():
        fractions[key] = (
            float(slot["n_resonant"]) / float(slot["n"])
            if slot["n"] > 0 else 0.0
        )
    return {
        "n_finals_new_kernel": int(n_finals),
        "per_species": {
            key: {
                "n": int(slot["n"]),
                "n_resonant": int(slot["n_resonant"]),
                "fraction_resonant": fractions[key],
                "by_parent": {
                    str(pdg): int(cnt)
                    for pdg, cnt in sorted(slot["by_parent"].items())
                },
            }
            for key, slot in sorted(per_species.items())
        },
        "fractions_resonant": fractions,
    }


# ---------------------------------------------------------------------------
# The analytic feed-down chain (the catalog BRs and the card-3/4 width
# law, evaluated at the recorded boundary mass distribution)
# ---------------------------------------------------------------------------


class _ChainExpectation:
    """The analytic resonant feed-down chain of the catalog.

    For one source mass ``M``, the expected per-species chain counts:
    over the card-9 open records at ``M`` with the kernel's own
    open-weight record normalization (``ResonanceDecayer._open_objects``,
    read-only), each chain walks its channel tree with the
    daughter resonances at their catalog pole masses (a pure compute
    walk with no draw budget; the chain geometry, not a spectral
    re-draw), every open channel object weighted by its share
    ``weight / total`` of the vertex's open weight -- the card-6
    open-weight normalization the kernel itself samples with.

    Lineage convention (corrected; Round 3 card
    ``TASK-20260917-r3-gate2-followup-feeddown-lineage-convention``,
    the Round 2 closure record clause 4 REVISE): resonant parentage is
    recorded only for non-root cascade decays.  A terminal particle is
    *resonant* when its producing vertex's parent is a resonance other
    than the drawn root record -- the root record's direct daughters
    and the stable-identity final of a stable root record are direct
    source products, never resonant.  This is the event tracker's
    convention (``_accumulate_feeddown``), so the analytic expectation
    and the tracker fraction use one convention and the corrected
    per-species expectation is ``f_res = 1 - (direct-production
    fraction)`` at the source masses realized in the sample.  The
    sealed R23 walk marked every terminal below any decay resonant
    (the degenerate convention: ``f_ana = 1`` exactly, zero sigma, the
    reported ``|z| = Infinity``); the corrected walk marks only the
    non-root cascade products resonant.

    The record mixture is flat over the caller's open set (the card-9
    mixture rule the builder draws with), including stable records:
    a stable record contributes its stable-identity final at the
    record weight.  The card-5 isospin charge weights act inside each
    record through the kernel's open objects and the kernel's open
    weights act inside each record's channel tree.
    """

    MAX_DEPTH = 32

    def __init__(self, catalog: HadronCatalog) -> None:
        self.catalog = catalog
        self.stable = frozenset(catalog.stable_states())
        self.decayer = ResonanceDecayer(catalog, 0, 0, 0)
        self._pole: dict[int, float] = {}

    def _pole_mass(self, pdg: int) -> float:
        cached = self._pole.get(int(pdg))
        if cached is None:
            cached = _stable_reference_mass(self.catalog, int(pdg))
            self._pole[int(pdg)] = cached
        return cached

    def _is_stable(self, pdg: int) -> bool:
        if int(pdg) in self.stable:
            return True
        try:
            record = self.catalog.get(int(pdg))
        except KeyError:
            raise RuntimeError(
                "staged_r23_resonance_ablation: the analytic feed-down "
                f"chain queues pdg {pdg}, which is neither a catalog "
                "record nor a stable state (a structural violation; the "
                "run stops)"
            ) from None
        return bool(record.stable_for_decay)

    def _chain_finals(self, root_pdg: int, root_mass: float) -> dict:
        """The expected terminal counts of one root record's chain at
        *root_mass*: ``{species_key: (n_expected, resonant_expected)}``.

        The walk expands every open channel object of each unstable
        record, each object weighted by its share of the vertex's open
        weight, with the daughter resonances at their catalog pole
        masses.  A terminal particle is resonant exactly when its
        producing vertex's parent is a resonance other than the root
        record (the corrected lineage convention); the root record's
        direct daughters are not resonant, and a stable root record
        contributes itself as the stable-identity final, not resonant.
        """
        root = int(root_pdg)
        if self._is_stable(root):
            key = _species_key(_record_name(self.catalog, root))
            return {key: (1.0, 0.0)}

        def expand(pdg: int, mass: float, resonant: bool,
                   depth: int) -> dict:
            if depth > self.MAX_DEPTH:
                raise RuntimeError(
                    "staged_r23_resonance_ablation: the analytic "
                    "feed-down chain exceeded "
                    f"{self.MAX_DEPTH} levels (a structural violation; "
                    "the run stops)"
                )
            pdg = int(pdg)
            if self._is_stable(pdg):
                key = _species_key(_record_name(self.catalog, pdg))
                return {key: (1.0, 1.0 if resonant else 0.0)}
            record = self.catalog.get(pdg)
            objects, total = self.decayer._open_objects(
                record, float(mass)
            )
            if not (float(total) > 0.0):
                raise RuntimeError(
                    "staged_r23_resonance_ablation: the analytic "
                    f"chain of {record.name} (pdg {pdg}) at mass "
                    f"{mass!r} has no open channel (the run stops)"
                )
            # The daughters of this vertex are resonant feed-down when
            # the producing vertex's parent is a resonance other than
            # the drawn root record (the tracker's convention).
            child_resonant = pdg != root
            acc: dict[str, list[float]] = {}
            for channel, combination, weight in objects:
                share = float(weight) / float(total)
                for daughter in combination:
                    sub = expand(
                        int(daughter), self._pole_mass(int(daughter)),
                        child_resonant, depth + 1,
                    )
                    for key, (n, res) in sub.items():
                        entry = acc.setdefault(key, [0.0, 0.0])
                        entry[0] += share * float(n)
                        entry[1] += share * float(res)
            return {
                key: (float(slot[0]), float(slot[1]))
                for key, slot in acc.items()
            }

        return expand(root, float(root_mass), False, 0)

    def expected(self, mass: float, records: list) -> dict:
        """The per-species ``(expected finals, expected resonant)`` chain
        counts at *mass* over the open records *records*.

        The record weights are flat (uniform over the caller's open
        set, the card-9 mixture rule; the card-5 isospin charge weights
        act inside each record and the kernel's open weights act inside
        each record's channel tree).  A stable record contributes its
        stable-identity final (a direct source product, never
        resonant) at the record weight; each unstable record's chain
        counts are its expected terminal particles at the pole masses
        under the kernel's open-weight channel normalization.
        """
        open_records: list = []
        for record in records:
            if bool(record.stable_for_decay):
                # The builder's stable-identity path: the drawn record
                # contributes itself as the single direct final.
                open_records.append(record)
                continue
            try:
                _objects, total = self.decayer._open_objects(
                    record, float(mass)
                )
            except Exception:
                continue
            if float(total) > 0.0:
                open_records.append(record)
        if not open_records:
            return {}
        weight = 1.0 / float(len(open_records))
        acc: dict[str, list[float]] = {}
        for record in open_records:
            if bool(record.stable_for_decay):
                key = _species_key(
                    _record_name(self.catalog, int(record.pdg_id))
                )
                entry = acc.setdefault(key, [0.0, 0.0])
                entry[0] += weight
                continue
            finals = self._chain_finals(int(record.pdg_id), float(mass))
            for key, (n, r) in finals.items():
                entry = acc.setdefault(key, [0.0, 0.0])
                entry[0] += weight * float(n)
                entry[1] += weight * float(r)
        return {
            key: (float(slot[0]), float(slot[1])) for key, slot in acc.items()
        }


def _analytic_feeddown(
    catalog: HadronCatalog, boundary_block: dict
) -> tuple[dict, int]:
    """The catalog analytic expectation over the boundary mass histogram.

    Returns ``(per-species (expected finals, expected resonant),
    histogram total)``; the expectation is the count-weighted mean of
    the chain expectation over the histogram bin midpoints.
    """
    edges = [float(v) for v in boundary_block["mass_histogram_edges"]]
    counts = [int(v) for v in boundary_block["mass_histogram_counts"]]
    total = int(sum(counts))
    if total <= 0 or len(edges) < 2:
        return {}, 0
    chain = _ChainExpectation(catalog)
    acc: dict[str, list[float]] = {}
    for i, count in enumerate(counts):
        if count <= 0:
            continue
        m_mid = 0.5 * (edges[i] + edges[i + 1])
        records = [
            record
            for record in catalog.records()
            if (int(record.B), int(record.Q), int(record.S)) == (1, 1, 0)
            and float(record.mass_range[0]) <= m_mid
        ]
        expected = chain.expected(m_mid, records)
        weight = float(count) / float(total)
        for key, (n, r) in expected.items():
            slot = acc.setdefault(key, [0.0, 0.0])
            slot[0] += weight * float(n)
            slot[1] += weight * float(r)
    return (
        {key: (float(v[0]), float(v[1])) for key, v in acc.items()},
        total,
    )


# ---------------------------------------------------------------------------
# Mode: --compare (the pre-registered metrics (i)-(iii))
# ---------------------------------------------------------------------------


#: The display-name -> config-species-key map (``pi-`` -> ``pi_minus``).
DISPLAY_TO_CONFIG = {
    display: key for key, display in r01.SPECIES_DISPLAY.items()
}


def _config_species(display: str) -> str:
    """The config species key of a display name (``pi-`` -> ``pi_minus``)."""
    if display not in DISPLAY_TO_CONFIG:
        raise ValueError(f"unknown display species {display!r}")
    return DISPLAY_TO_CONFIG[display]


def _cell_value(combo, binned: dict, what: str) -> float:
    """The single element of a one-bin combination evaluation (the
    pre-registered sum/difference combinations are total on aligned
    bins; a None entry refuses the run)."""
    values = evaluate_combination(combo, binned)
    value = values[0]
    if value is None:
        _refuse(f"{what}: the combination bin is undefined (None)")
    return float(value)


def _pair_variance(a: float, b: float) -> float:
    """The card-7 pairwise variance of two independent species cells."""
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.inf
    return float(combined_covariance((a * a,), (b * b,))[0])


def _aligned_common_cells(table_a, table_b) -> list[tuple[int, int]]:
    """The sorted common ``(xF, pT)`` midpoint cells of two species'
    windowed tables, with their bin indices in each table (the R11
    alignment convention)."""
    coords_a = {
        (
            float(table_a.xf_centers[table_a.bin_index_xf[i]]),
            float(table_a.pt_centers[table_a.bin_index_pt[i]]),
        ): i
        for i in range(int(table_a.data.size))
    }
    coords_b = {
        (
            float(x), float(p)
        ): i
        for i, (x, p) in enumerate(
            zip(
                [float(table_b.xf_centers[table_b.bin_index_xf[i]])
                 for i in range(int(table_b.data.size))],
                [float(table_b.pt_centers[table_b.bin_index_pt[i]])
                 for i in range(int(table_b.data.size))],
            )
        )
    }
    common = sorted(set(coords_a) & set(coords_b))
    return [(coords_a[c], coords_b[c]) for c in common]


def _metric_inclusive(
    tables: list, values23: dict, errors23: dict, r01_by_species: dict
) -> tuple[list[dict], dict]:
    """Metric (i): per species, the R23 value against the sealed R01
    value within the combined 4-sigma of the two archives."""
    rows: list[dict] = []
    for table in tables:
        species = table.species
        model23 = np.asarray(values23[species], dtype=float)
        mc23 = np.asarray(errors23[species], dtype=float)
        model01 = np.asarray(r01_by_species[species]["value"], dtype=float)
        mc01 = np.asarray(r01_by_species[species]["stat_error"], dtype=float)
        for i in range(int(table.data.size)):
            d01 = float(model01[i])
            m23 = float(model23[i])
            s01 = float(mc01[i])
            s23 = float(mc23[i])
            if not (math.isfinite(s01) and math.isfinite(s23)):
                sigma = math.inf
                pull = 0.0
                within = True
            else:
                sigma = math.sqrt(s01 * s01 + s23 * s23)
                pull = (d01 - m23) / sigma if sigma > 0.0 else 0.0
                within = abs(pull) <= INCLUSIVE_N_SIGMA
            rows.append({
                "species": species,
                "bin_index": int(i),
                "xF_mid": float(table.xf_centers[table.bin_index_xf[i]]),
                "pT_mid": float(table.pt_centers[table.bin_index_pt[i]]),
                "r01_value": d01,
                "r23_value": m23,
                "r01_stat_error": s01,
                "r23_stat_error": s23,
                "sigma_combined": float(sigma),
                "delta_over_sigma": float(pull),
                "within_4_sigma": bool(within),
            })
    n_flagged = sum(1 for row in rows if not row["within_4_sigma"])
    max_abs = max(
        (abs(float(row["delta_over_sigma"])) for row in rows), default=0.0
    )
    summary = {
        "n_compared": len(rows),
        "n_flagged": n_flagged,
        "max_abs_delta_over_sigma": float(max_abs),
        "n_sigma": INCLUSIVE_N_SIGMA,
        "pass": n_flagged == 0,
    }
    return rows, summary


def _metric_identified(
    tables: list,
    values23: dict,
    errors23: dict,
    r01_by_species: dict,
    pre: dict,
    ranked_rows: list[dict],
) -> tuple[list[dict], dict]:
    """Metric (ii): the top-5 identified set against the training
    tables, with the R01 comparison at the combined 2-sigma level."""
    tables_by_species = {t.species: t for t in tables}
    cell_rows: list[dict] = []
    sum23 = 0.0
    sum01 = 0.0
    noise_fractions: list[float] = []
    gross_violations = 0
    for entry in pre["identified_set"]["entries"]:
        combo = combination_by_name(str(entry["observable"]))
        sector = str(entry["sector"])
        region = str(entry["region"])
        k = int(entry["bin_index"])
        if len(combo.species) == 1:
            sp = _config_species(combo.species[0])
            if sector != sp:
                _refuse(
                    f"identified cell rank {entry['rank']}: single-species "
                    f"sector {sector!r} != {sp!r}"
                )
            table = tables_by_species[sp]
            if k >= int(table.data.size):
                _refuse(
                    f"identified cell rank {entry['rank']}: bin index "
                    f"{k} outside the {sp} table"
                )
            xF_mid = float(table.xf_centers[table.bin_index_xf[k]])
            if r11.xf_sector_region(xF_mid) != region:
                _refuse(
                    f"identified cell rank {entry['rank']}: region "
                    f"{region!r} does not match the bin midpoint xF "
                    f"{xF_mid!r}"
                )
            d23 = float(table.data[k])
            m23 = float(values23[sp][k])
            sm23 = float(errors23[sp][k])
            sm01 = float(r01_by_species[sp]["stat_error"][k])
            sd = float(table.sigma[k])
            v_mc23 = sm23 * sm23 if math.isfinite(sm23) else math.inf
            v_mc01 = sm01 * sm01 if math.isfinite(sm01) else math.inf
            v23 = (sd * sd if math.isfinite(sd) else math.inf) + v_mc23
        else:
            sp_a = _config_species(combo.species[0])
            sp_b = _config_species(combo.species[1])
            if sector != f"{sp_a}+{sp_b}":
                _refuse(
                    f"identified cell rank {entry['rank']}: two-species "
                    f"sector {sector!r} != {sp_a}+{sp_b} (the R11 sector "
                    "naming)"
                )
            table_a = tables_by_species[sp_a]
            table_b = tables_by_species[sp_b]
            common = _aligned_common_cells(table_a, table_b)
            if k >= len(common):
                _refuse(
                    f"identified cell rank {entry['rank']}: bin index "
                    f"{k} outside the {len(common)} common cells"
                )
            ia, ib = common[k]
            xF_mid = float(table_a.xf_centers[table_a.bin_index_xf[ia]])
            if r11.xf_sector_region(xF_mid) != region:
                _refuse(
                    f"identified cell rank {entry['rank']}: region "
                    f"{region!r} does not match the bin midpoint xF "
                    f"{xF_mid!r}"
                )
            d23 = _cell_value(combo, {
                combo.species[0]: (float(table_a.data[ia]),),
                combo.species[1]: (float(table_b.data[ib]),),
            }, f"identified cell rank {entry['rank']} data")
            m23 = _cell_value(combo, {
                combo.species[0]: (float(values23[sp_a][ia]),),
                combo.species[1]: (float(values23[sp_b][ib]),),
            }, f"identified cell rank {entry['rank']} model")
            sm23_a = float(errors23[sp_a][ia])
            sm23_b = float(errors23[sp_b][ib])
            sm01_a = float(r01_by_species[sp_a]["stat_error"][ia])
            sm01_b = float(r01_by_species[sp_b]["stat_error"][ib])
            sd_a = float(tables_by_species[sp_a].sigma[ia])
            sd_b = float(tables_by_species[sp_b].sigma[ib])
            v_mc23 = _pair_variance(sm23_a, sm23_b)
            v_mc01 = _pair_variance(sm01_a, sm01_b)
            v23 = _pair_variance(sd_a, sd_b) + v_mc23
        sigma23 = math.sqrt(v23) if math.isfinite(v23) else math.inf
        if math.isinf(sigma23):
            pull23 = 0.0
        else:
            pull23 = (d23 - m23) / sigma23
        noise_fraction = (
            (v_mc23 + v_mc01) / v23
            if math.isfinite(v_mc23) and math.isfinite(v_mc01)
            and math.isfinite(v23) and v23 > 0.0
            else math.inf
        )
        sum23 += abs(pull23)
        sum01 += abs(float(entry["delta_over_sigma_r01"]))
        noise_fractions.append(float(noise_fraction))
        if abs(pull23) > IDENTIFIED_MAX_ABS_PULL:
            gross_violations += 1
        cell_rows.append({
            "rank": int(entry["rank"]),
            "sector": sector,
            "observable": str(entry["observable"]),
            "region": region,
            "bin_index": k,
            "d": float(d23),
            "m_r23": float(m23),
            "m_r01": float(entry["m_r01"]),
            "sigma_r23": (
                float(sigma23) if math.isfinite(sigma23) else "Infinity"
            ),
            "pull_r23": float(pull23),
            "pull_r01": float(entry["delta_over_sigma_r01"]),
            "abs_pull_r23": float(abs(pull23)),
            "abs_pull_r01": float(abs(float(entry["delta_over_sigma_r01"]))),
            "mc_noise_uncertainty_fraction": (
                float(noise_fraction)
                if math.isfinite(noise_fraction) else "Infinity"
            ),
        })
    finite_terms = [
        float(t) ** 2 for t in noise_fractions if math.isfinite(float(t))
    ]
    sigma_comb = math.sqrt(math.fsum(finite_terms))
    n_infinite = sum(
        1 for t in noise_fractions if not math.isfinite(float(t))
    )
    margin = float(sum23) - float(sum01)
    sum_pass = margin <= IDENTIFIED_SUM_N_SIGMA * sigma_comb
    summary = {
        "entries": [
            {
                "rank": int(row["rank"]),
                "sector": str(row["sector"]),
                "observable": str(row["observable"]),
                "region": str(row["region"]),
                "bin_index": int(row["bin_index"]),
            }
            for row in cell_rows
        ],
        "sum_abs_pull_r23": float(sum23),
        "sum_abs_pull_r01": float(sum01),
        "margin": float(margin),
        "combined_mc_noise_sigma": float(sigma_comb),
        "n_cells_with_infinite_mc_noise": int(n_infinite),
        "infinite_noise_note": (
            "a cell with an infinite MC error on either archive carries "
            "an infinite MC-noise uncertainty; its pull is 0.0 by the "
            "empty-bin convention and it contributes an infinite term "
            "to the combined sigma (recorded, never dropped)"
        ),
        "gross_violations": int(gross_violations),
        "max_abs_pull": IDENTIFIED_MAX_ABS_PULL,
        "sum_n_sigma": IDENTIFIED_SUM_N_SIGMA,
        "max_abs_pull_r23": max(
            (float(row["abs_pull_r23"]) for row in cell_rows), default=0.0
        ),
        "pass": (gross_violations == 0) and bool(sum_pass),
    }
    return cell_rows, summary


def _metric_feeddown(
    catalog: HadronCatalog, boundary_block: dict, feeddown_block: dict
) -> tuple[list[dict], dict]:
    """Metric (iii): the per-species resonant feed-down fraction of the
    new-kernel events against the catalog analytic expectation (the
    catalog BRs and width law evaluated at the recorded boundary mass
    distribution) within 3-sigma."""
    analytic, hist_total = _analytic_feeddown(catalog, boundary_block)
    rows: list[dict] = []
    max_abs_sigma = 0.0
    n_checked = 0
    for key, slot in sorted(feeddown_block["per_species"].items()):
        n = int(slot["n"])
        if n < FEED_DOWN_MIN_N:
            continue
        f_obs = float(slot["fraction_resonant"])
        expected = analytic.get(key)
        if expected is None:
            f_ana = 0.0
        else:
            n_finals, n_res = expected
            f_ana = float(n_res) / float(n_finals) if float(n_finals) > 0.0 else 0.0
        if math.isfinite(f_ana) and 0.0 < f_ana < 1.0:
            sigma = math.sqrt(f_ana * (1.0 - f_ana) / float(n))
        else:
            sigma = 0.0
        if sigma > 0.0:
            z = (f_obs - f_ana) / sigma
        elif f_obs == f_ana:
            z = 0.0
        else:
            z = math.inf if f_obs > f_ana else -math.inf
        rows.append({
            "species": str(key),
            "n_finals": int(n),
            "fraction_resonant_observed": float(f_obs),
            "fraction_resonant_analytic": float(f_ana),
            "sigma": float(sigma),
            "z": float(z) if math.isfinite(float(z)) else (
                "Infinity" if float(z) > 0 else "-Infinity"
            ),
        })
        n_checked += 1
        if math.isfinite(float(z)):
            max_abs_sigma = max(max_abs_sigma, abs(float(z)))
        else:
            max_abs_sigma = math.inf
    summary = {
        "n_checked": int(n_checked),
        "min_n_per_species": FEED_DOWN_MIN_N,
        "histogram_total": int(hist_total),
        "max_abs_sigma": (
            float(max_abs_sigma) if math.isfinite(float(max_abs_sigma))
            else "Infinity"
        ),
        "n_sigma": FEED_DOWN_N_SIGMA,
        "pass": (
            n_checked > 0 and all(
                abs(float(row["z"])) <= FEED_DOWN_N_SIGMA for row in rows
            )
        ),
    }
    return rows, summary


def run_comparison(run_dir: Path) -> dict:
    """The pre-registered comparison metrics (i)-(iii), assembled from
    the R23 replicas, the sealed R01 archive, the training tables, and
    the sealed R11 ranked residual map."""
    run_dir = Path(run_dir)
    pre = _assert_pre_registration(run_dir)
    raw_config = _load_raw_config()
    config_path = NESTED_ROOT / "config" / r01.PRODUCTION_CONFIG_NAME
    verification = r01.verify_config_and_m8(config_path, pre)
    tables = r01.load_binned_tables(raw_config)
    tables_sha = r01.tables_digest(tables)
    table_data_digest = _table_data_digest(tables)
    replicas = _load_production_replicas(
        run_dir, pre, verification["config_sha256"], tables_sha,
        table_data_digest,
    )
    seeds = sorted(replicas)
    n_events_total = sum(int(replicas[s]["n_events"]) for s in seeds)

    # -- pooled predictions ----------------------------------------------
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    for seed in seeds:
        for species, values in replicas[seed]["sums"].items():
            arr = np.asarray(values, dtype=float)
            sums[species] = (
                arr if species not in sums else sums[species] + arr
            )
        for species, values in replicas[seed]["counts"].items():
            arr = np.asarray(values, dtype=np.int64)
            counts[species] = (
                arr if species not in counts else counts[species] + arr
            )
    beam = float(_load_raw_config().get("beam_momentum", 158.0))
    sqrt_s = Reaction(beam, 2212, 2212).sqrt_s()
    _entries, values23, errors23 = r01.bin_predictions_payload(
        tables, sums, counts, n_events_total, sqrt_s
    )

    # -- the sealed R01 archive ------------------------------------------
    r01_archive = r11.verify_r01_archive(MONO_ROOT / R01_RUN_DIR)
    r01_predictions = _read_json(
        MONO_ROOT / R01_RUN_DIR / r01.BIN_PREDICTIONS_NAME
    )
    r01_by_species: dict[str, dict[str, list[float]]] = {}
    for table in r01.load_binned_tables(json.loads(
            (NESTED_ROOT / "config" / r01.PRODUCTION_CONFIG_NAME).read_text(
                encoding="utf-8"))):
        r01_by_species[table.species] = {"value": [], "stat_error": []}
    for entry in r01_predictions:
        slot = r01_by_species[str(entry["species"])]
        slot["value"].append(float(entry["value"]))
        slot["stat_error"].append(float(entry["stat_error"]))

    # -- the sealed R11 ranked map ---------------------------------------
    ranked_path = MONO_ROOT / R11_RUN_DIR / "ranked_residual_map.json"
    ranked = _read_json(ranked_path)
    ranked_rows = list(ranked["rows"])[:5]

    # -- the metrics ------------------------------------------------------
    inclusive_rows, inclusive_summary = _metric_inclusive(
        r01.load_binned_tables(_load_raw_config()), values23, errors23,
        r01_by_species,
    )
    identified_cells, identified_summary = _metric_identified(
        r01.load_binned_tables(_load_raw_config()), values23, errors23,
        r01_by_species, pre, ranked_rows,
    )
    catalog = _load_catalog()
    boundary_block = _pooled_boundary(replicas)
    feeddown_block = _pooled_feeddown(replicas)
    feeddown_rows, feeddown_summary = _metric_feeddown(
        catalog, boundary_block, feeddown_block
    )

    # -- conservation bookkeeping ----------------------------------------
    events_checked = sum(
        int(replicas[s]["conservation"]["events_checked"]) for s in seeds
    )
    conservation_ok = all(
        bool(replicas[s]["conservation"]["event_b_q_conserved"])
        for s in seeds
    ) and events_checked == n_events_total

    payload = {
        "schema": "na49_mixing.staged_r23_resonance_ablation.comparison/1",
        "run": RUN_ID,
        "card": CARD,
        "n_replicas": len(seeds),
        "replica_seeds": [int(s) for s in seeds],
        "n_events_total": int(n_events_total),
        "normalization": {
            "convention": (
                "the invariant cross sections normalized to the inelastic "
                "cross section, f / sigma_inel, cross section per "
                "inelastic event"
            ),
            "sigma_inel_mb": SIGMA_INEL_MB,
            "window": dict(PAPER_WINDOW),
        },
        "pooled_predictions": _entries,
        "metrics": {
            "inclusive_stability": {
                "summary": inclusive_summary,
                "flagged_rows": [
                    row for row in inclusive_rows
                    if not row["within_4_sigma"]
                ],
            },
            "identified_set": {
                "summary": identified_summary,
                "cells": identified_cells,
            },
            "feed_down": {
                "summary": feeddown_summary,
                "rows": feeddown_rows,
            },
        },
        "boundary": boundary_block,
        "feed_down": feeddown_block,
        "conservation": {
            "events_checked": int(events_checked),
            "event_b_q_conserved": bool(conservation_ok),
        },
        "r01_archive": {
            "path": str(r01_archive["archive_path"]),
            "archive_seal_sha256": r01_archive["archive_seal_sha256"],
            "replica_seeds": r01_archive["seeds"],
            "config_sha256": r01_archive["config_sha256"],
        },
        "r11_ranked_map": {
            "path": f"{R11_RUN_DIR}/ranked_residual_map.json",
            "top5": [
                {
                    "rank": int(row["rank"]),
                    "sector": str(row["sector"]),
                    "observable": str(row["observable"]),
                    "region": str(row["region"]),
                    "bin_index": int(row["bin_index"]),
                    "delta_over_sigma": float(row["delta_over_sigma"]),
                }
                for row in ranked_rows
            ],
        },
        "input_hashes": {
            "config_sha256": verification["config_sha256"],
            "tables_sha256": tables_sha,
            "table_data_digest": table_data_digest,
            "catalog_sha256": _sha256_file(NESTED_ROOT / CATALOG_REL),
            "r01_sums_sha256": r01_archive["archive_seal_sha256"],
            "r11_sums_sha256": pre["identified_set"]["sha256_r11_sums"],
            "driver_sha256": pre["driver_sha256"],
            "builder_module_sha256": pre["builder_module_sha256"],
        },
        "no_fitting": True,
        "optimizer_invoked": False,
    }
    _write_json(run_dir / COMPARISON_NAME, payload)
    print(
        "[r23] comparison written: metric (i) max |Delta/sigma| = "
        f"{inclusive_summary['max_abs_delta_over_sigma']:.6f} "
        f"(pass = {inclusive_summary['pass']}), metric (ii) pass = "
        f"{identified_summary['pass']} (sum margin "
        f"{identified_summary['margin']:.6f} +- "
        f"{identified_summary['combined_mc_noise_sigma']:.6f}), metric "
        "(iii) pass = "
        f"{feeddown_summary['pass']}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# Mode: --gate (acceptance evidence only)
# ---------------------------------------------------------------------------


def _fold_gate(rel_path: str) -> dict:
    """A sealed R20/R21/R22 acceptance record and its digest."""
    path = MONO_ROOT / rel_path
    if not path.is_file():
        _refuse(f"the sealed prerequisite acceptance record is missing: {path}")
    payload = _read_json(path)
    return {"path": str(path), "sha256": _sha256_file(path), "payload": payload}


def _folded_clauses(comparison: dict, bit_identity: dict) -> tuple[dict, dict]:
    """The folded clauses and R23's own contributions (evidence only).

    Every folded value is recorded in the clause's ``fold`` block: a
    clause's ``pass`` is the conjunction of its folded values and R23's
    own contribution, so the written retain / revise / reject decision
    (the Gate 2 follow-up obligation, round 2 index card) interprets
    truthful evidence, never a template.
    """
    gates = {
        "r20": _fold_gate(R20_GATE_REL),
        "r21": _fold_gate(R21_GATE_REL),
        "r22": _fold_gate(R22_GATE_REL),
    }

    def clause_value(gate_key: str, clause: str):
        block = gates[gate_key]["payload"].get("clauses") or {}
        entry = block.get(clause)
        if entry is None:
            return None
        if isinstance(entry, dict):
            return entry.get("pass")
        return bool(entry)

    # -- local conservation ----------------------------------------------
    local_r23 = bool(comparison["conservation"]["event_b_q_conserved"])
    local_r23_events = int(comparison["conservation"]["events_checked"])
    local_fold = {}
    for key in ("r20", "r21", "r22"):
        entry = (gates[key]["payload"].get("clauses") or {}).get(
            "local_conservation"
        )
        if isinstance(entry, dict):
            local_fold[key] = {
                "pass": bool(entry.get("pass")),
                "events_checked": int(entry.get("events_checked", 0)),
            }
        else:
            local_fold[key] = {"pass": None, "note": "null placeholder"}
    local_pass = local_r23 and all(
        bool(value.get("pass")) for value in local_fold.values()
    )

    # -- branching test (folded from R20; R21/R22 carry the null
    #    placeholder for this clause) --------------------------------------
    branching_fold = {}
    for key in ("r20", "r21", "r22"):
        entry = clause_value(key, "branching_test")
        if entry is None:
            branching_fold[key] = {
                "value": None,
                "note": (
                    f"the sealed {key.upper()} record carries the null "
                    "placeholder for this clause"
                ),
            }
        else:
            branching_fold[key] = {"value": bool(entry)}
    branching_pass = bool(clause_value("r20", "branching_test") or False)

    # -- threshold discontinuity ------------------------------------------
    threshold_fold = {}
    for key in ("r20", "r21", "r22"):
        threshold_fold[key] = {
            "value": bool(clause_value(key, "threshold_discontinuity")
                          or False)
        }
    threshold_pass = all(
        value["value"] for value in threshold_fold.values()
    )

    # -- observables (R23's own clause) -----------------------------------
    bit_ok = bool(bit_identity.get("pass"))
    metrics = comparison["metrics"]
    observables_pass = (
        bit_ok
        and bool(metrics["inclusive_stability"]["summary"]["pass"])
        and bool(metrics["identified_set"]["summary"]["pass"])
        and bool(metrics["feed_down"]["summary"]["pass"])
        and local_r23
    )
    clauses = {
        "local_conservation": {
            "event_b_q_conserved": local_r23,
            "events_checked": local_r23_events,
            "pass": bool(local_pass),
            "folded_from": [R20_GATE_REL, R21_GATE_REL, R22_GATE_REL],
            "fold": local_fold,
        },
        "branching_test": {
            "pass": branching_pass,
            "folded_from": [R20_GATE_REL],
            "fold": branching_fold,
        },
        "threshold_discontinuity": {
            "pass": bool(threshold_pass),
            "folded_from": [R20_GATE_REL, R21_GATE_REL, R22_GATE_REL],
            "fold": threshold_fold,
        },
        "observables": {
            "identified_set": [
                {
                    "rank": int(entry["rank"]),
                    "sector": str(entry["sector"]),
                    "observable": str(entry["observable"]),
                    "region": str(entry["region"]),
                    "bin_index": int(entry["bin_index"]),
                }
                for entry in metrics["identified_set"]["summary"]["entries"]
            ],
            "inclusive_stability_max_abs_sigma": _as_float(
                metrics["inclusive_stability"]["summary"]
                ["max_abs_delta_over_sigma"]
            ),
            "identified_set_pass": bool(
                metrics["identified_set"]["summary"]["pass"]
            ),
            "feed_down_max_abs_sigma": _as_float(
                metrics["feed_down"]["summary"]["max_abs_sigma"]
            ),
            "r01_comparison": {
                "species": list(r01.SPECIES_DISPLAY),
                "max_abs_delta_over_sigma_vs_r01": _as_float(
                    metrics["inclusive_stability"]["summary"]
                    ["max_abs_delta_over_sigma"]
                ),
            },
            "pass": bool(observables_pass),
            "bit_identity_pass": bit_ok,
            "inclusive_pass": bool(
                metrics["inclusive_stability"]["summary"]["pass"]
            ),
            "feed_down_pass": bool(
                metrics["feed_down"]["summary"]["pass"]
            ),
        },
        "no_branching_fit": {
            "no_fitting": True,
            "optimizer_invoked": False,
        },
    }
    return gates, clauses


def _as_float(value):
    """A JSON-safe float projection of a recorded metric value."""
    if isinstance(value, str):
        if value == "Infinity":
            return math.inf
        if value == "-Infinity":
            return -math.inf
        if value == "NaN":
            return math.nan
        return float(value)
    return float(value)


# ---------------------------------------------------------------------------
# Mode: --gate
# ---------------------------------------------------------------------------


def run_gate(run_dir: Path) -> dict:
    """Write ``gate2_r23.json``: the R23 acceptance evidence, folding
    the sealed R20/R21/R22 records with R23's own metrics."""
    run_dir = Path(run_dir)
    pre = _load_pre_registration(run_dir)
    bit_identity = _assert_bit_identity(run_dir)
    comparison = _read_json(run_dir / COMPARISON_NAME)
    gates, clauses = _folded_clauses(comparison, bit_identity)
    payload = {
        "schema": "na49_mixing.staged_r23_resonance_ablation.gate/1",
        "gate": "gate 2 (plan section 8, line 575)",
        "run": RUN_ID,
        "card": CARD,
        "clauses": clauses,
        "bit_identity": {
            "layers_checked": list(LAYER_DEFINITIONS),
            "seed": int(BIT_IDENTITY_SEED),
            "n_events": int(BIT_IDENTITY_EVENTS),
            "pass": bool(bit_identity.get("pass")),
        },
        "input_hashes": {
            "comparison_r01_sha256": _sha256_file(run_dir / COMPARISON_NAME),
            "bit_identity_report_sha256": _sha256_file(
                run_dir / BIT_IDENTITY_NAME
            ),
            "folded_gates": {
                key: {"path": gate["path"], "sha256": gate["sha256"]}
                for key, gate in gates.items()
            },

        },
        "sealed": True,
        "seal_note": (
            "sealed marks the gate evidence as the sealed form: the "
            "step-6 seal covers this file's bytes (the seal manifest "
            "records them after this file is written)"
        ),
        "decision_note": (
            "evidence only; the written retain / revise / reject "
            "decision on all four clauses is the Gate 2 follow-up "
            "obligation (round 2 index card), recorded in _knowledge/"
        ),
    }
    _write_json(run_dir / GATE_RECORD_NAME, payload)
    print(
        "staged_r23_resonance_ablation: acceptance evidence written to "
        f"{run_dir / GATE_RECORD_NAME} (local_conservation pass = "
        f"{clauses['local_conservation']['pass']}, branching_test pass = "
        f"{clauses['branching_test']['pass']}, threshold_discontinuity "
        f"pass = {clauses['threshold_discontinuity']['pass']}, "
        f"observables pass = {clauses['observables']['pass']})",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# Mode: --seal
# ---------------------------------------------------------------------------


def _run_dir_files(run_dir: Path) -> list[Path]:
    """Every regular file under the run directory, sorted by relative
    path."""
    run_dir = Path(run_dir)
    files = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and not path.is_symlink():
            files.append(path)
    return files


def write_sha256sums(run_dir: Path) -> Path:
    """Write ``SHA256SUMS.txt`` over every payload file (sha256sum -c).

    The inventory covers every regular file under the run directory
    except the files that cannot carry their own digests (the inventory
    itself, ``seal_manifest.json``, ``provenance.json``, and the
    acceptance record, which carries the inventory's digest) and the
    post-seal clean-environment record.
    """
    run_dir = Path(run_dir)
    skip = {
        SHA256SUMS_NAME, SEAL_NAME, PROVENANCE_NAME, GATE_RECORD_NAME,
        CLEAN_ENV_NAME,
    }
    lines = []
    for path in _run_dir_files(run_dir):
        if path.name in skip:
            continue
        rel = path.relative_to(run_dir).as_posix()
        lines.append(f"{_sha256_file(path)}  {rel}\n")
    sums_path = run_dir / SHA256SUMS_NAME
    _write_bytes(sums_path, "".join(lines).encode("utf-8"))
    return sums_path


def write_seal_manifest(run_dir: Path) -> dict:
    """Write ``seal_manifest.json`` over every other file plus the sums.

    The seal manifest records every regular file under the run directory
    except itself (the payload files, ``SHA256SUMS.txt``,
    ``provenance.json``, and the acceptance record when present), each
    with its sha256 and byte size; it cannot contain its own digest.
    """
    run_dir = Path(run_dir)
    entries = []
    for path in _run_dir_files(run_dir):
        if path.name == SEAL_NAME:
            continue
        entries.append({
            "path": path.relative_to(run_dir).as_posix(),
            "sha256": _sha256_file(path),
            "bytes": int(path.stat().st_size),
        })
    seal = {
        "sealed_at_utc": _now_utc(),
        "run_directory": Path(run_dir).name,
        "card": CARD,
        "format": (
            "SHA256SUMS.txt covers every payload file in sha256sum -c "
            "format (sorted relative paths), excluding the files that "
            "cannot carry their own digests (SHA256SUMS.txt itself, "
            "seal_manifest.json, provenance.json, gate2_r23.json) and "
            "the post-seal clean_environment_commands.json; "
            "seal_manifest.json records the payload files plus "
            "SHA256SUMS.txt, provenance.json, and gate2_r23.json; "
            "seal_manifest.json cannot contain its own digest"
        ),
        "files": entries,
    }
    _write_json(run_dir / SEAL_NAME, seal)
    return seal


def run_seal(run_dir: Path) -> dict:
    """Seal the finished run directory (sums inventory plus seal
    manifest).  The pre-registration, the replica records, the
    comparison, and the acceptance record must exist."""
    run_dir = Path(run_dir)
    if not (run_dir / PROVENANCE_NAME).is_file():
        _refuse(
            f"provenance missing before sealing: {run_dir / PROVENANCE_NAME}"
        )
    if not (run_dir / GATE_RECORD_NAME).is_file():
        _refuse(
            f"acceptance record missing before sealing: "
            f"{run_dir / GATE_RECORD_NAME} (run --gate first; the gate "
            "JSON is part of the seal)"
        )
    write_sha256sums(run_dir)
    seal = write_seal_manifest(run_dir)
    print(
        f"staged_r23_resonance_ablation: run dir sealed ({len(seal['files'])} "
        f"files listed in {SEAL_NAME})",
        flush=True,
    )
    return seal


# ---------------------------------------------------------------------------
# Mode: --clean-validate
# ---------------------------------------------------------------------------


def run_clean_validate(run_dir: Path) -> dict:
    """Verify the seal, re-run the comparison from a clean interpreter
    over the sealed artifacts, and confirm the report hash matches."""
    run_dir = Path(run_dir)
    pre = _load_pre_registration(run_dir)
    if not (run_dir / SHA256SUMS_NAME).is_file():
        _refuse(
            f"{run_dir / SHA256SUMS_NAME} is missing; --clean-validate "
            "requires the sealed run dir (run --seal first)"
        )
    # -- seal verification ------------------------------------------------
    mismatches: list[str] = []
    files_checked = 0
    for line in (run_dir / SHA256SUMS_NAME).read_text(
            encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        payload = run_dir / name
        files_checked += 1
        if not payload.is_file() or _sha256_file(payload) != digest:
            mismatches.append(name)
    if mismatches:
        _refuse(
            "seal verification failed; "
            f"{len(mismatches)} of {files_checked} SHA256SUMS.txt entries "
            f"do not re-hash correctly (first: {mismatches[0]!r}); the "
            "sealed run dir must be unmodified"
        )

    # -- clean-interpreter re-run of the comparison -----------------------
    comparison_path = run_dir / COMPARISON_NAME
    before = comparison_path.read_bytes()
    driver_path = Path(__file__).resolve()
    argv = [sys.executable, "-I", str(driver_path), "--compare",
            "--run-dir", str(run_dir)]
    probe = subprocess.run(argv, capture_output=True, text=True,
                           cwd=str(MONO_ROOT))
    if probe.returncode != 0:
        _refuse(
            "the clean-interpreter comparison probe failed "
            f"(return code {probe.returncode}):\n{probe.stderr}"
        )
    recomputed = comparison_path.read_bytes()
    hashes_match = (
        int(probe.returncode) == 0
        and hashlib.sha256(before).hexdigest()
        == hashlib.sha256(recomputed).hexdigest()
        and before == recomputed
    )
    # Restore the sealed bytes: the probe rewrote the report; the sealed
    # dir must keep its sealed content regardless of the probe outcome.
    if recomputed != before:
        _write_bytes(comparison_path, before)
    payload = {
        "schema": "na49_mixing.staged_r23_resonance_ablation.clean_environment/1",
        "run": RUN_ID,
        "card": CARD,
        "verified_utc": _now_utc(),
        "seal_verification": {
            "files_checked": int(files_checked),
            "mismatches": mismatches,
        },
        "commands": [{
            "purpose": (
                "clean-interpreter re-run of the card-13 comparison over "
                "the sealed artifacts (rewrites comparison_r01.json; the "
                "sealed bytes must be reproduced byte for byte)"
            ),
            "argv": argv,
            "cwd": str(MONO_ROOT),
        }],
        "environment": environment_fingerprint(),
        "probe_returncode": int(probe.returncode),
        "sealed_report_sha256": {
            COMPARISON_NAME: hashlib.sha256(before).hexdigest(),
        },
        "recomputed_report_sha256": {
            COMPARISON_NAME: hashlib.sha256(recomputed).hexdigest(),
        },
        "hashes_match": bool(
            hashes_match
            and int(probe.returncode) == 0
            and hashlib.sha256(before).hexdigest()
            == hashlib.sha256(recomputed).hexdigest()
        ),
        "probe_tool": {
            "path": str(driver_path),
            "sha256": _sha256_file(driver_path),
            "note": (
                "the committed dual-tree driver is the probe instrument; "
                "the sealed artifacts and the committed kernel are the "
                "inputs"
            ),
        },
    }
    _write_json(run_dir / CLEAN_ENV_NAME, payload)
    if not payload["hashes_match"]:
        print(
            "staged_r23_resonance_ablation: the clean-environment "
            "validation FAILED (the recomputed comparison bytes differ "
            "from the sealed report)",
            flush=True,
        )
        raise SystemExit(1)
    print(
        "staged_r23_resonance_ablation: clean-environment validation "
        f"passed ({files_checked} seal files verified, the clean "
        "interpreter reproduced comparison_r01.json byte for byte)",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="staged_r23_resonance_ablation.py",
        description=(
            "Execute run R23, the resonance-only p+p ablation (Round 2, "
            "card 13): the legacy source/exchange layer bit-identical to "
            "the frozen legacy builder, only the low-mass decay kernel "
            "replaced, compared against the sealed R01 archive and the "
            "training tables."
        ),
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--pre-registration", action="store_true",
        help="write pre_registration.json and provenance.json, then stop",
    )
    modes.add_argument(
        "--bit-identity", action="store_true",
        help="run the 10,000-event bit-identity check, then stop",
    )
    modes.add_argument(
        "--generate", action="store_true",
        help="generate one replica (--replica K) or every replica",
    )
    modes.add_argument(
        "--compare", action="store_true",
        help="run the comparison metrics (i)-(iii), then stop",
    )
    modes.add_argument(
        "--gate", action="store_true",
        help="write gate2_r23.json (acceptance evidence), then stop",
    )
    modes.add_argument(
        "--seal", action="store_true",
        help="write seal_manifest.json and SHA256SUMS.txt, then stop",
    )
    modes.add_argument(
        "--clean-validate", action="store_true",
        help="verify the seal and re-run the comparison cleanly, then stop",
    )
    parser.add_argument(
        "--replica", type=int, default=None,
        help="with --generate: the replica index k = 0..7",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="with --generate: hermetic reduced statistics (driver "
             "validation; the gate / seal / clean-validate modes refuse "
             "smoke artifacts)",
    )
    parser.add_argument(
        "--smoke-events", type=int, default=SMOKE_EVENTS,
        help=f"smoke replica size (default {SMOKE_EVENTS})",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"fork worker count (default {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--run-dir", default=DEFAULT_RUN_DIR,
        help=f"the run directory (default {DEFAULT_RUN_DIR})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_dir = Path(str(args.run_dir))
    if args.smoke and not args.generate:
        _refuse("--smoke is only valid together with --generate")
    if args.replica is not None and not args.generate:
        _refuse("--replica is only valid together with --generate")
    if args.smoke and args.replica is not None:
        _refuse("--smoke generates a single tagged replica; --replica is "
                "a production option")
    if args.pre_registration:
        run_pre_registration(run_dir)
    elif args.bit_identity:
        run_bit_identity(run_dir)
    elif args.generate:
        if args.smoke:
            # The smoke path: no pre-registration or bit-identity
            # requirement; one tagged replica of --smoke-events events.
            run_dir.mkdir(parents=True, exist_ok=True)
            raw_config = _load_raw_config()
            config_path = NESTED_ROOT / "config" / r01.PRODUCTION_CONFIG_NAME
            config_sha = _sha256_file(config_path)
            tables = r01.load_binned_tables(raw_config)
            tables_sha = r01.tables_digest(tables)
            table_data_digest = _table_data_digest(tables)
            catalog = _load_catalog()
            pp_config, _applied = r01.build_pp_config(raw_config)
            builder_sha = _sha256_file(
                NESTED_ROOT / "na49_mixing" / "resonance_pp_builder.py"
            )
            driver_sha = _sha256_file(Path(__file__).resolve())
            _generate_replica(
                int(REPLICA_SEEDS[0]), 0, int(args.smoke_events), run_dir,
                tables, catalog, pp_config, int(args.workers), config_sha,
                tables_sha, table_data_digest, builder_sha, driver_sha,
                True,
            )
            _update_replica_manifest(run_dir, smoke=True)
            print(
                f"[r23] smoke generation complete (run dir {run_dir}, "
                f"1 replica x {int(args.smoke_events)} events)",
                flush=True,
            )
        else:
            run_generation(run_dir, args)
    elif args.compare:
        run_comparison(run_dir)
    elif args.gate:
        run_gate(run_dir)
    elif args.seal:
        run_seal(run_dir)
    elif args.clean_validate:
        run_clean_validate(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


