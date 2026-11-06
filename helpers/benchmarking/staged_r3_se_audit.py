#!/usr/bin/env python3
"""R3 SE audit + gateway confirmation driver (Round 3, card 8c).

Task ``TASK-20260918-r3-se-audit-confirmation-mc`` (plan section 23,
line 1817, "Round 3 -- continuum color-singlet states"; run R31,
plan section 9 "Required runs", line 640; the round-3 index card 8c).
Two workstreams, one run, one run dir, one pre-registration:

- **Phase A** (hermetic, analysis-only, no new random numbers): a
  deterministic bit-identical replay of the sealed R31 sample (the
  committed R31 driver code at the original master seed 20260917) and
  of the R21/R22 named-residual subsets (the committed card-9 driver
  code at the seeds recorded in each sealed run's pre-registration),
  with the replayed stored sums cross-checked exactly against the
  sealed per-replica aggregates before any estimator is computed; the
  SE-estimator battery E1-E4 on the 10 named R31 momentum-moment
  residuals plus the pre-registered sibling scope; the R31 seed-map
  audit (stream-collision, replica-correlation, degenerate-stream
  checks); the node-1 tight-SE classification; and the same estimator
  audit on the sealed card-2 re-evaluation's 11 named R21/R22
  residuals.  Deliverable: ``phase_a_diagnostics.json`` (no wall-clock
  fields; its bytes are a pure function of the sealed artifacts and the
  committed code, which is what the clean-environment validation
  re-hashes).

- **Phase B** (gateway confirmation MC): regenerate the 15 R31 nodes at
  the confirmation statistics, 50 replicas x 20,000 decays =
  1,000,000 decays per node, 15,000,000 total, with a NEW master seed
  20260918 disjoint from R20-R23 and the original R31 seeds; 750 chunks
  of 20,000 decays, one chunk per gateway job (``--tasks 1``, single
  math thread, no local process pool in the chunk command); chunk
  outputs land in ``phase_b/`` and are aggregated in fixed replica
  order; then all 422 R31 checks are re-evaluated with the fixed
  card-2 engine (8a P1) at the confirmation statistics, reporting
  per-check sigma at the old SE versus the new SE, with the 14 original
  failures named.  The 3.0-sigma tolerance is never relaxed; no
  special-casing of any pair; no fitting of any kind.

Seed-parameterized per-decay function: this driver imports the
committed R31 driver module (``staged_r31_matching_scan``, nested
``ce80619``, sha256-pinned in the pre-registration; byte-identical
across the dual tree) and the committed card-9 driver module
(``staged_source_grid``) and reuses their building blocks (the card-6
``route_source_decay`` routing, the card-3 ``draw_mass``,
``ResonanceDecayer``, ``_open_set``, ``_precompute_kernel_maps``,
``_Accumulator``, ``_accumulate_event``).  The R31 driver is NOT
modified: its ``_r31_generate_chunk`` is coupled to a local fork pool
and the hard-coded R31 ``MASTER_SEED = 20260917`` and is not reused
across a gateway process boundary.  This driver carries the
seed-parameterized ``_decay_particles_se(master_seed, replica,
event_index)`` wrapper; a hermetic parity test asserts that at the
original seed (20260917) the wrapper's per-decay output is bit-identical
to the R31 driver's ``_decay_particles`` on a pinned (window, node,
replica, event) sample, and that a reduced-scale replay reproduces the
sealed R31 replica diagnostics exactly.

Phase A replay determinism: the R31 replay accumulates each replica
through the committed R31 chunk convention (2,500-decay chunks merged
in start order, the R31 ``_generate_node`` flow), so the replayed
per-replica payload equals the sealed replica file exactly (every
stored sum matches bit for bit); the R21/R22 replay accumulates each
replica sequentially in one accumulator (the card-9 driver's
convention) the same way.  The bootstrap resampler is a fixed analysis
seed (``np.random.default_rng(20260918)``) consumed in the
pre-registered estimator order; it is a bootstrap resampler, not an
event-generation substream, and draws zero event-generation random
numbers.

Modes (exactly one per invocation; each performs one step and stops):

- ``--pre-registration``: write ``pre_registration.json``,
  ``provenance.json``, and ``pre_investigation_inventory.json`` into
  the run dir, then stop.  Nothing is evaluated or generated.
- ``--phase-a``: the Phase A replay, cross-checks, estimator battery,
  seed-map audit, and classifications; write
  ``phase_a_diagnostics.json``, then stop.
- ``--chunk [--window W --node J --replica R] [--decays N]``: generate
  one Phase B chunk (20,000 decays, production) or, with ``--smoke``,
  one smoke chunk (2,000 decays by default) into ``phase_b_smoke/``;
  then stop.  A chunk file that already exists and parses with the
  matching identity is left unchanged (a completed chunk is never
  regenerated); the cluster output is the canonical chunk record.
- ``--aggregate``: verify and merge the fetched chunk outputs into the
  per-replica and per-node card-9-schema diagnostics (with the routing
  fractions), writing the checkpoint state; then stop.  Nodes are
  merged only when all 50 of their chunks are present, so a dropped
  session resumes aggregation from the chunk outputs.
- ``--reeval``: re-evaluate all 422 R31 checks with the fixed engine at
  the confirmation statistics, per-check sigma at the old SE versus the
  new SE, the 14 original failures named individually; write
  ``confirmation_verification.json``, then stop.
- ``--seal``: write ``seal_manifest.json`` and ``SHA256SUMS.txt`` over
  the run dir, then stop.  Smoke artifacts are refused.
- ``--clean-validate``: verify the seal, re-run ``--phase-a``,
  ``--aggregate``, and ``--reeval`` from clean interpreters over the
  sealed artifacts, confirm the report hashes match; write
  ``clean_environment_commands.json``, then stop.  Smoke artifacts are
  refused.
- ``--smoke`` (with ``--chunk`` / ``--aggregate`` / ``--reeval``):
  hermetic reduced statistics for driver validation; smoke artifacts
  live under ``phase_b_smoke/`` and are refused by ``--seal`` and
  ``--clean-validate``.

Dual-tree driver: byte-identical in the nested tree
(``dev/mixing2006/helpers/benchmarking/``) and the monorepo mirror
(``helpers/benchmarking/``); both locate the package with the
established ``_locate_roots`` pattern.

Hermetic: the committed catalog, the committed continuum config, the
sealed R31/R21/R22/card-2 run dirs, and the vendored ``na49_mixing``
modules only; no external tables; no event generation outside the
declared modes.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import multiprocessing
import os
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
# na49_mixing package, the resonance catalog, and the continuum config.
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
    print(f"staged_r3_se_audit: {message}", file=sys.stderr, flush=True)
    raise SystemExit(2)


NESTED_ROOT, MONO_ROOT = _locate_roots()
if str(NESTED_ROOT) not in sys.path:
    sys.path.insert(0, str(NESTED_ROOT))

from helpers.benchmarking import staged_r01_reference as r01  # noqa: E402
from helpers.benchmarking import staged_r31_matching_scan as r31  # noqa: E402
from helpers.benchmarking import staged_source_grid as sgrid  # noqa: E402
from na49_mixing.excitation_kinematics import FourVector  # noqa: E402
from na49_mixing.hadron_catalog import HadronCatalog  # noqa: E402
from na49_mixing.pdg_database import get_particle  # noqa: E402
from na49_mixing.quantum_numbers import quantum_numbers  # noqa: E402
from na49_mixing.resonance_decay import ResonanceDecayer  # noqa: E402
from na49_mixing.resonance_decay import SourceRecord as KernelSourceRecord  # noqa: E402
from na49_mixing.source_event import FinalParticle  # noqa: E402
from na49_mixing.resonance_widths import decay_momentum, draw_mass  # noqa: E402
from na49_mixing.rng_substreams import SUBSTREAM_LABELS, derive_substreams  # noqa: E402
from na49_mixing.spectral_density import (  # noqa: E402
    families_from_json,
    route_source_decay,
)
from na49_mixing.threshold_metric import (  # noqa: E402
    DETERMINISTIC_SE_FLOOR,
    SIGMA_TOLERANCE,
    THRESHOLD_METRIC_ID,
    ZERO_RESIDUAL_TOL,
    RecordSet,
    evaluate_pair,
    full_analytic_difference,
)

# ---------------------------------------------------------------------------
# Run identity (card TASK-20260918-r3-se-audit-confirmation-mc; the card
# fixes the Phase A estimator definitions and cutoffs, the named
# residual lists, the confirmation statistics, the new seed scheme, and
# the chunk plan; the 3.0-sigma tolerance is never relaxed).
# ---------------------------------------------------------------------------

RUN_ID = "R3-SE-AUDIT"
CARD = "TASK-20260918-r3-se-audit-confirmation-mc"
RUN_NAME = "r3_se_audit_2026-09-18"
DEFAULT_RUN_DIR = "dev/00runs/r3_se_audit_2026-09-18"
PHASE_B_NAME = "phase_b"
SMOKE_DIR_NAME = "phase_b_smoke"
PREREGISTRATION_NAME = "pre_registration.json"
PROVENANCE_NAME = "provenance.json"
PRE_INVENTORY_NAME = "pre_investigation_inventory.json"
PHASE_A_NAME = "phase_a_diagnostics.json"
REEVAL_NAME = "confirmation_verification.json"
AGG_STATE_NAME = "phase_b_aggregation_state.json"
SUBMISSION_MANIFEST_REL = "phase_b/submission_manifest.json"
FETCH_LOG_REL = "phase_b/fetch_log.json"
SHA256SUMS_NAME = "SHA256SUMS.txt"
SEAL_NAME = "seal_manifest.json"
CLEAN_ENV_NAME = "clean_environment_commands.json"

CATALOG_REL = "config/v2_singlet/resonance_catalog.json"
FAMILIES_REL = "config/v2_singlet/continuum_resonances.json"
NEW_DRIVER_REL = "helpers/benchmarking/staged_r3_se_audit.py"
R31_DRIVER_REL = "helpers/benchmarking/staged_r31_matching_scan.py"
CARD9_DRIVER_REL = "helpers/benchmarking/staged_source_grid.py"

#: The sealed run dirs (read-only inputs; byte-unchanged by this card).
R31_RUN_DIR_REL = "dev/00runs/r31_catalog_continuum_matching_scan_2026-09-17"
R21_RUN_DIR_REL = "dev/00runs/r21_baryonic_source_grid_2026-09-13"
R22_RUN_DIR_REL = "dev/00runs/r22_mesonic_source_grid_2026-09-13"
REREVAL_RUN_DIR_REL = "dev/00runs/r3_threshold_metric_reregistration_2026-09-17"

#: The confirmed engine of record (card 8a P1, nested 95096d2):
#: exact Kallen three-body slots, deep-level daughter quadrature, the
#: deterministic SE floor; SIGMA_TOLERANCE stays 3.0.
ENGINE_MODULE_REL = "dev/mixing2006/na49_mixing/threshold_metric.py"

#: Phase B statistics (the card): T1-scaled confirmation, 15 nodes x
#: 1,000,000 decays as 50 replicas x 20,000 decays; 15,000,000 total.
CONFIRMATION_REPLICAS = 50
CHUNK_DECAYS = 20_000
DECAYS_PER_NODE = CONFIRMATION_REPLICAS * CHUNK_DECAYS
TOTAL_DECAYS = 3 * 5 * DECAYS_PER_NODE

#: The confirmation seed map (the card): master seed 20260918, disjoint
#: from R20-R23 (master 20260913 / raw replica seeds 20300913..20300920)
#: and the original R31 master 20260917.
CONFIRMATION_MASTER_SEED = 20260918
SEED_RULE = (
    "derive_substreams(20260918, r', (w*5 + j) * 100_000_000 + r'*20_000 "
    "+ i) per (window w, node j, replica r' < 50, decay i < 20,000) on "
    "the closed decay_channel/decay_kinematics labels; the confirmation "
    "master seed 20260918 is disjoint from R20-R23 (R20-R22 master "
    "20260913, R23 raw replica seeds 20300913..20300920) and from the "
    "original R31 master 20260917"
)

#: Phase A estimator definitions (the card's pre-registered battery).
BOOTSTRAP_SEED = 20260918
BLOCK_LENGTHS = (32, 128)
BOOTSTRAP_RESAMPLES = 2000
#: E2: within-replica moving-block bootstrap over the 25,000 decays of
#: each sealed replica; the node-level value is the mean of the four
#: per-replica SEs over sqrt(n_replicas).  E4: pooled single-sample
#: moving-block bootstrap over all 4 x 25,000 decays of a node as one
#: sample (the SE of the node mean directly).
E1_DEFINITION = (
    "between-replica SE: std ddof=1 of the stored replica moments over "
    "sqrt(n_replicas); for the multiplicity the per-node pooled stored "
    "'se' (the card-9 convention, the existing estimator)"
)
E2_DEFINITION = (
    "within-replica moving-block bootstrap: block lengths b = 32 and "
    "b = 128 decays, B = 2,000 resamples, event-level resampling of the "
    "per-decay contribution tuples (count, p_sum, e_sum, ept_sum) of "
    "the bit-identical deterministic replay, L = ceil(N/b) blocks per "
    "resample, the resampler RNG np.random.default_rng(20260918) "
    "consumed in the pre-registered estimator order (a bootstrap "
    "resampler and not an event-generation substream); the node-level "
    "value is the mean of the per-replica SEs over sqrt(n_replicas)"
)
E3_DEFINITION = (
    "replica jackknife (leave-one-out over the stored replica moments): "
    "SE = sqrt((n-1)/n * sum (m_i - m_bar)^2)"
)
E4_DEFINITION = (
    "pooled single-sample moving-block bootstrap: all 4 x 25,000 decays "
    "of a node as one sample in replica order, same b and B as E2; the "
    "value is the SE of the node mean directly"
)
CLASSIFICATION_RULE = (
    "ratio cutoffs on the pair-combined SEs: E4_combined/E1_combined > 3 "
    "= estimator artifact, < 1.5 = genuine low variance, in between = "
    "mixed; replica pairwise correlation |r| > 0.2 for n = 4 = seed "
    "issue flag; the classification is a diagnostic pre-registration "
    "value, not a metric tolerance, and the 3.0-sigma band is never "
    "touched"
)
RATIO_ARTIFACT = 3.0
RATIO_GENUINE = 1.5
CORRELATION_FLAG = 0.2
SUBSTREAM_LABELS_AUDIT = ("decay_channel", "decay_kinematics")

#: The 10 named R31 momentum-moment residuals (the card table; the
#: pre-fix sigma values are the card's, delta_analytic_full is
#: bit-identical pre/post engine fix).
NAMED_RESIDUALS: tuple[dict, ...] = (
    {"index": 1, "window": 0, "pair": [0, 1], "species": "proton",
     "observable": "mean_pt_y", "pre_fix_sigma": 3.21},
    {"index": 2, "window": 0, "pair": [0, 1], "species": "pi_minus",
     "observable": "mean_pt_y", "pre_fix_sigma": 3.33},
    {"index": 3, "window": 0, "pair": [3, 4], "species": "neutron",
     "observable": "mean_p", "pre_fix_sigma": 3.15},
    {"index": 4, "window": 2, "pair": [0, 1], "species": "pi_0",
     "observable": "mean_pt_y", "pre_fix_sigma": -3.03},
    {"index": 5, "window": 2, "pair": [0, 1], "species": "K0L",
     "observable": "mean_p", "pre_fix_sigma": 3.06},
    {"index": 6, "window": 2, "pair": [0, 1], "species": "K0L",
     "observable": "mean_pt_y", "pre_fix_sigma": 3.07},
    {"index": 7, "window": 2, "pair": [1, 2], "species": "pi_0",
     "observable": "mean_p", "pre_fix_sigma": 7.01},
    {"index": 8, "window": 2, "pair": [3, 4], "species": "K0S",
     "observable": "mean_pt_y", "pre_fix_sigma": 3.40},
    {"index": 9, "window": 2, "pair": [3, 4], "species": "K0L",
     "observable": "mean_p", "pre_fix_sigma": 3.74},
    {"index": 10, "window": 2, "pair": [3, 4], "species": "K0L",
     "observable": "mean_pt_y", "pre_fix_sigma": 4.55},
)

#: The four mechanism-attributable failures closed by the card-8a P1
#: engine fix (named in the card; Phase B reports all 14 at the
#: confirmation statistics).
MECHANISM_RESIDUALS: tuple[dict, ...] = (
    {"window": 1, "pair": [2, 3], "species": "pi_plus",
     "observable": "mean_pt_y", "pre_fix_sigma": 7.23,
     "note": "window 1 pair (2, 3) pi_plus mean_pt_y, engine 0.184825 "
             "vs kernel-exact MC 0.18536 after the exact Kallen slots"},
    {"window": 1, "pair": [3, 4], "species": "pi_plus",
     "observable": "mean_pt_y", "pre_fix_sigma": -5.61,
     "note": "window 1 pair (3, 4) pi_plus mean_pt_y, same mechanism"},
    {"window": 0, "pair": [3, 4], "species": "K_0",
     "observable": "mean_p", "pre_fix_sigma": 25.98,
     "note": "float-precision SE artifact under the pre-registered SE "
             "floor (deterministic realization, collapsed SE)"},
    {"window": 0, "pair": [3, 4], "species": "Sigma_plus",
     "observable": "mean_p", "pre_fix_sigma": 25.98,
     "note": "float-precision SE artifact under the pre-registered SE "
             "floor, same mechanism as K_0"},
)

#: The 11 named R21/R22 residuals of the sealed card-2 re-evaluation
#: (gate2_revised_threshold.json residual_findings; run, pair, species,
#: observable, pair class, re-evaluated sigma).
R21R22_NAMED_RESIDUALS: tuple[dict, ...] = (
    {"run": "R21", "pair": [0, 1], "species": "proton",
     "observable": "mean_p", "pair_class": "open_set_change",
     "sigma": 3.598708027565074},
    {"run": "R21", "pair": [0, 1], "species": "proton",
     "observable": "mean_pt_y", "pair_class": "open_set_change",
     "sigma": 3.4457902279287818},
    {"run": "R21", "pair": [1, 2], "species": "pi_0",
     "observable": "mean_pt_y", "pair_class": "open_set_unchanged",
     "sigma": 3.3622721646606926},
    {"run": "R21", "pair": [3, 4], "species": "pi_0",
     "observable": "mean_pt_y", "pair_class": "open_set_unchanged",
     "sigma": 3.594992035740298},
    {"run": "R21", "pair": [4, 5], "species": "pi_minus",
     "observable": "mean_pt_y", "pair_class": "open_set_change",
     "sigma": 3.0555852180458674},
    {"run": "R21", "pair": [5, 6], "species": "pi_0",
     "observable": "mean_pt_y", "pair_class": "open_set_change",
     "sigma": 3.314280796269175},
    {"run": "R21", "pair": [15, 16], "species": "pi_0",
     "observable": "mean_p", "pair_class": "open_set_unchanged",
     "sigma": 3.0846781101915357},
    {"run": "R21", "pair": [18, 19], "species": "pi_0",
     "observable": "multiplicity", "pair_class": "open_set_unchanged",
     "sigma": 3.014046219969753},
    {"run": "R22", "pair": [7, 8], "species": "pi_minus",
     "observable": "mean_pt_y", "pair_class": "open_set_change",
     "sigma": 3.8051287494598878},
    {"run": "R22", "pair": [15, 16], "species": "pi_plus",
     "observable": "mean_pt_y", "pair_class": "open_set_unchanged",
     "sigma": 3.018186264093622},
    {"run": "R22", "pair": [23, 24], "species": "pi_0",
     "observable": "mean_pt_y", "pair_class": "open_set_unchanged",
     "sigma": 4.984844257802041},
)

#: The sibling scope (the card): all 15 grid nodes for each affected
#: species+observable, and all species at the affected nodes.
AFFECTED_CELLS: tuple[tuple[int, int], ...] = (
    (0, 0), (0, 1), (0, 3), (0, 4), (2, 0), (2, 1), (2, 3), (2, 4),
)
AFFECTED_SPECIES_OBS: tuple[tuple[str, str], ...] = (
    ("proton", "mean_pt_y"),
    ("pi_minus", "mean_pt_y"),
    ("neutron", "mean_p"),
    ("pi_0", "mean_pt_y"),
    ("K0L", "mean_p"),
    ("K0L", "mean_pt_y"),
    ("pi_0", "mean_p"),
    ("K0S", "mean_pt_y"),
)
AFFECTED_SPECIES = tuple(sorted({s for s, _ in AFFECTED_SPECIES_OBS}))

#: Phase A replay conventions (the committed drivers').
REPLAY_CHUNK_SIZE = 2500
DEFAULT_AUDIT_WORKERS = 12

#: Smoke mode (driver validation; never sealed): 2 nodes x 2 replicas x
#: 2,000 decays, replaying pre-registered confirmation triples at
#: reduced counts into phase_b_smoke/ (no draw exists outside the
#: pre-registered confirmation map).  The two nodes form the
#: boundary-straddling pair (j = 3, 4) so the smoke exercises the
#: continuum-route chunk generation and the routed-observable
#: re-evaluation path end to end.
SMOKE_NODES = ((2, 3), (2, 4))
SMOKE_REPLICAS = 2
SMOKE_DECAYS = 2_000

OBSERVABLES = ("multiplicity", "mean_p", "mean_pt_y")
MOMENT_FIELD = {"mean_p": "mean_p_gev", "mean_pt_y": "mean_pt_y_gev"}

#: The pinned parity sample (window, node, replica, event_index) for
#: the hermetic bit-identity test against the frozen R31 driver's
#: ``_decay_particles`` at the original master seed 20260917.
PARITY_WINDOWS = (0, 1, 2)
PARITY_NODES = (0, 2, 4)
PARITY_REPLICAS = (0, 1, 2, 3)
PARITY_EVENTS = (0, 1, 7, 1234, 24999)


def parity_sample() -> list[tuple[int, int, int, int]]:
    """The pinned (window, node, replica, event) parity sample."""
    return [
        (w, j, r, i)
        for w in PARITY_WINDOWS
        for j in PARITY_NODES
        for r in PARITY_REPLICAS
        for i in PARITY_EVENTS
    ]


# ---------------------------------------------------------------------------
# Shared helpers (the R31 driver's conventions, imported; the R31
# driver is the committed building-block source and stays
# byte-unchanged).
# ---------------------------------------------------------------------------

_now_utc = r31._now_utc
_write_json = r31._write_json
_read_json = r31._read_json
_sha256_file = r31._sha256_file
_write_bytes = r31._write_bytes
_git_info = r31._git_info
environment_fingerprint = r31.environment_fingerprint
_run_dir_files = r31._run_dir_files
_species_key = r31._species_key
_record_name = r31._record_name
_bqs_of = r31._bqs_of
_open_set = r31._open_set
_stable_species_keys = r31._stable_species_keys
_precompute_kernel_maps = r31._precompute_kernel_maps
_build_context = r31._build_context
DEGENERATE_WINDOW_TOL = r31.DEGENERATE_WINDOW_TOL
CONTINUUM_ROOT_SENTINEL = r31.CONTINUUM_ROOT_SENTINEL

WINDOWS = r31.WINDOWS
N_WINDOWS = r31.N_WINDOWS
NODE_MASSES = r31.NODE_MASSES
N_NODES = r31.N_NODES
SPEC = r31.SPEC
BOUNDARY = r31.BOUNDARY
RAMP_WIDTH = r31.RAMP_WIDTH
EXPECTED_W = r31.EXPECTED_W
METRIC_ID = r31.METRIC_ID
METRIC_N_SIGMA = r31.METRIC_N_SIGMA
OBSERVABLE_KEYS_R31 = r31.OBSERVABLES


def _git_state() -> dict:
    """The nested + outer git identities (defensive)."""
    return _git_info()


# ---------------------------------------------------------------------------
# Input hashes and the pre-investigation inventory
# ---------------------------------------------------------------------------


def _input_hashes() -> dict:
    catalog_path = NESTED_ROOT / CATALOG_REL
    families_path = NESTED_ROOT / FAMILIES_REL
    return {
        "catalog_sha256": _sha256_file(catalog_path),
        "families_config_sha256": _sha256_file(families_path),
        "spectral_density_module_sha256": _sha256_file(
            NESTED_ROOT / "na49_mixing" / "spectral_density.py"
        ),
        "threshold_metric_module_sha256": _sha256_file(
            NESTED_ROOT / "na49_mixing" / "threshold_metric.py"
        ),
        "resonance_decay_module_sha256": _sha256_file(
            NESTED_ROOT / "na49_mixing" / "resonance_decay.py"
        ),
        "resonance_widths_module_sha256": _sha256_file(
            NESTED_ROOT / "na49_mixing" / "resonance_widths.py"
        ),
        "rng_substreams_module_sha256": _sha256_file(
            NESTED_ROOT / "na49_mixing" / "rng_substreams.py"
        ),
        "r31_driver_sha256": _sha256_file(NESTED_ROOT / R31_DRIVER_REL),
        "card9_driver_sha256": _sha256_file(NESTED_ROOT / CARD9_DRIVER_REL),
        "driver_sha256": _sha256_file(Path(__file__).resolve()),
    }


SEALED_DIRS = (
    ("r31", R31_RUN_DIR_REL),
    ("r21", R21_RUN_DIR_REL),
    ("r22", R22_RUN_DIR_REL),
    ("card2_reregistration", REREVAL_RUN_DIR_REL),
)


def build_pre_investigation_inventory() -> dict:
    """The sha256 inventory of every file in the R31 run dir and the
    seal state of the R21/R22/card-2 dirs (run procedure step 1;
    read-only inputs)."""
    entries = []
    for path in _run_dir_files(MONO_ROOT / R31_RUN_DIR_REL):
        entries.append({
            "path": path.relative_to(MONO_ROOT).as_posix(),
            "sha256": _sha256_file(path),
            "bytes": int(path.stat().st_size),
        })
    seals = {}
    for name, rel in SEALED_DIRS:
        sums_path = MONO_ROOT / rel / SHA256SUMS_NAME
        if not sums_path.is_file():
            seals[name] = {"sealed": False, "note": "no SHA256SUMS.txt"}
            continue
        mismatches = 0
        checked = 0
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, rel_path = line.split("  ", 1)
            payload = MONO_ROOT / rel_path
            checked += 1
            if not payload.is_file() or _sha256_file(payload) != digest:
                mismatches += 1
        seals[name] = {
            "sealed": True,
            "sha256_sums_entries": int(checked),
            "mismatches": int(mismatches),
            "verify": "pass" if mismatches == 0 else "FAIL",
        }
    return {
        "schema": "na49_mixing.staged_r3_se_audit.pre_investigation/1",
        "run": RUN_ID,
        "card": CARD,
        "r31_run_dir": R31_RUN_DIR_REL,
        "r31_files": entries,
        "r31_file_count": len(entries),
        "r31_driver_sha256_in_inventory": _sha256_file(
            MONO_ROOT / R31_RUN_DIR_REL / "pre_registration.json"
        ),
        "seal_states": seals,
        "read_only_inputs_note": (
            "the sealed R31, R21, R22, and card-2 re-evaluation dirs are "
            "read-only; every R31 run-dir file hash recorded here must "
            "still match at the end of the run"
        ),
    }


# ---------------------------------------------------------------------------
# Pre-registration (written before any Phase A evaluation and before any
# Phase B chunk generation; asserted against this card).
# ---------------------------------------------------------------------------


def build_pre_registration(catalog: HadronCatalog, families: list) -> dict:
    """The card-8c pre-registration payload, written before any
    evaluation (Phase A) and before any chunk generation (Phase B)."""
    context = _build_context(catalog, families)
    open_sets_payload = {}
    for spec_window in r31.WINDOWS:
        qn = tuple(int(v) for v in spec_window["qn"])
        open_sets_payload[str(spec_window["window"])] = {
            str(j): [int(rec.pdg_id) for rec in context["open_sets"][qn][mass]]
            for j, mass in enumerate(NODE_MASSES)
        }
    return {
        "schema": "na49_mixing.staged_r3_se_audit.pre_registration/1",
        "run": RUN_ID,
        "card": CARD,
        "written_before_evaluation": True,
        "phase_a": {
            "scope": (
                "hermetic SE-estimator and seed-map audit over the "
                "sealed R31 artifacts and the sealed R21/R22 grids; no "
                "new random numbers (deterministic replay of the sealed "
                "samples at their recorded seeds, zero new generation)"
            ),
            "named_residuals": [dict(r) for r in NAMED_RESIDUALS],
            "sibling_scope": {
                "species_observable_at_all_nodes": [
                    {"species": s, "observable": o}
                    for s, o in AFFECTED_SPECIES_OBS
                ],
                "all_species_at_affected_nodes": [
                    {"window": w, "node": j} for w, j in AFFECTED_CELLS
                ],
                "statement": (
                    "all 15 grid nodes for each affected species+"
                    "observable, and all species at the affected nodes"
                ),
            },
            "r21r22_named_residuals": [dict(r) for r in R21R22_NAMED_RESIDUALS],
            "r21r22_replay_scope": (
                "the 11 named residuals' point cells (both nodes of each "
                "residual pair), replayed at the seeds recorded in each "
                "sealed run's pre_registration.json (master 20260913, "
                "derive_substreams(20260913, r, k * 1_000_000 + i)), "
                "cross-checked against the sealed per-replica aggregates "
                "before any estimator is computed"
            ),
            "replay": {
                "r31": (
                    "bit-identical deterministic replay of the sealed "
                    "R31 sample with the committed R31 driver code "
                    "(helpers/benchmarking/staged_r31_matching_scan.py) "
                    "at the original master seed 20260917 and the R31 "
                    "seed map derive_substreams(20260917, r, (w*5 + j) "
                    "* 1_000_000 + i); per-replica accumulation through "
                    "the R31 2,500-decay chunk convention merged in "
                    "chunk order; every stored sum of every sealed "
                    "replica and node record must match exactly before "
                    "any estimator is computed"
                ),
                "r21r22": (
                    "bit-identical deterministic replay of the "
                    "named-residual subsets with the committed card-9 "
                    "driver code at the seeds recorded in each sealed "
                    "run's pre_registration.json (master 20260913, "
                    "derive_substreams(20260913, r, k * 1_000_000 + i)); "
                    "the same exact-match cross-check applies"
                ),
                "draws": 0,
            },
            "estimator_definitions": {
                "E1": E1_DEFINITION,
                "E2": E2_DEFINITION,
                "E3": E3_DEFINITION,
                "E4": E4_DEFINITION,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "block_lengths": list(BLOCK_LENGTHS),
                "bootstrap_resampler_seed": BOOTSTRAP_SEED,
                "resampler_seed_note": (
                    "the bootstrap resampler is a fixed analysis seed, "
                    "not an event-generation substream; it draws no "
                    "event-generation random numbers; one "
                    "np.random.default_rng(20260918) instance is consumed "
                    "in the documented estimator order: archives in the "
                    "order R31, R21, R22; cells ascending (window, node) "
                    "or point index; species in catalog stable order "
                    "within a cell's capture set; b ascending (32, 128); "
                    "per b: E2 replicas 0..3 then the pooled E4"
                ),
                "statistic_conventions": (
                    "per resample: multiplicity = total count / "
                    "resampled events; mean_p = total p_sum / total "
                    "count (undefined where the count is 0); mean_pt_y = "
                    "total ept_sum / total e_sum (undefined where the "
                    "energy sum is 0); SE = std ddof=1 over the B "
                    "resample values (undefined resamples excluded and "
                    "counted)"
                ),
            },
            "seed_map_audit": {
                "stream_collision_test": (
                    "derive_substreams ids on the two closed labels "
                    "(decay_channel, decay_kinematics) over the full "
                    "pre-registered event sample per node: the full R31 "
                    "map (15 cells x 4 replicas x 25,000), the full "
                    "confirmation map (15 cells x 50 replicas x 20,000), "
                    "and the R21/R22 replayed cells (17 cells x 4 "
                    "replicas x 25,000); a duplicate stream-id pair "
                    "within or across the maps is reported (zero "
                    "expected)"
                ),
                "replica_correlation_test": (
                    "per R31 node, the pairwise Pearson correlation of "
                    "the four sealed per-replica moment vectors' "
                    "deviations from the per-node mean vector (centering "
                    "removes the shared species-scale structure, so the "
                    "correlation measures replica-noise coherence; raw "
                    "moment vectors correlate at ~0.999 across every "
                    "replica pair by construction and flag nothing "
                    "selectively); |r| > 0.2 for n = 4 flags a seed issue"
                ),
                "degenerate_stream_check": (
                    "no two (replica, event) pairs of a map share a "
                    "stream id on either label (covered by the "
                    "stream-collision test) and no confirmation triple "
                    "shares an id with the R20-R23 / original-R31 maps "
                    "(the cross-map union uniqueness); R20-R22 derive "
                    "substreams from master 20260913, R23 uses raw "
                    "replica seeds 20300913..20300920, R31 uses master "
                    "20260917, all distinct from 20260918; sampled "
                    "R20-style and R23-style triples are included in "
                    "the empirical collision set"
                ),
                "correlation_flag_cutoff": CORRELATION_FLAG,
            },
            "classification_thresholds": {
                "ratio_artifact": RATIO_ARTIFACT,
                "ratio_genuine": RATIO_GENUINE,
                "correlation_flag": CORRELATION_FLAG,
                "rule": CLASSIFICATION_RULE,
                "precedence_note": (
                    "the E4/E1 ratio takes precedence when it exceeds "
                    "the artifact cutoff (direct SE-honesty evidence); "
                    "the positive-side replica-noise coherence flag is "
                    "recorded alongside as a modifier"
                ),
                "node1_tight_se_target": (
                    "window 2 node 1 (2.58 GeV) pi_0 mean_p, "
                    "between-replica SE 0.00015 versus sibling ~0.0015: "
                    "classified with the node-level E4/E1 ratio and the "
                    "same thresholds"
                ),
                "no_tolerance_change": True,
            },
            "deliverable": "phase_a_diagnostics.json",
        },
        "phase_b": {
            "windows": [
                {
                    "window": int(spec_window["window"]),
                    "qn": [int(v) for v in spec_window["qn"]],
                    "label": spec_window["label"],
                    "family_expectation": spec_window["family_expectation"],
                }
                for spec_window in r31.WINDOWS
            ],
            "nodes": {
                "masses_gev": [float(M) for M in NODE_MASSES],
                "rule": (
                    "identical to R31: M_j = 2.48 + 0.1 j, j = 0..4, at "
                    "rest (p = 0, E = M_j); the isolated source decay"
                ),
                "boundary_gev": BOUNDARY,
            },
            "statistics": {
                "tier": "T1-scaled confirmation of the R31 sample",
                "replicas": CONFIRMATION_REPLICAS,
                "decays_per_replica": CHUNK_DECAYS,
                "decays_per_node": DECAYS_PER_NODE,
                "windows": N_WINDOWS,
                "nodes": N_NODES,
                "total_decays": TOTAL_DECAYS,
                "estimator_uncertainty_note": (
                    "the between-replica SE estimator uncertainty at 4 "
                    "replicas is ~50% of the SE (sqrt(1/(2*(n-1))) ~ "
                    "0.41); at 50 replicas it is < ~10% (~0.10)"
                ),
            },
            "seed_map": {
                "master_seed": CONFIRMATION_MASTER_SEED,
                "rule": SEED_RULE,
                "substream_labels": ["decay_channel", "decay_kinematics"],
                "disjointness_note": (
                    "master seeds 20260918 (this run), 20260917 (R31), "
                    "and 20260913 (R20-R22) are distinct, and R23's raw "
                    "replica seeds 20300913..20300920 are distinct from "
                    "20260918; the empirical collision test covers the "
                    "full confirmation map against the full R31 map and "
                    "the R21/R22 replayed cells"
                ),
            },
            "decay_routing": {
                "source": "card 6, identical to R31 per node mass",
                "boundary_gev": BOUNDARY,
                "ramp_width_gev": RAMP_WIDTH,
                "expected_w_at_nodes": [float(w) for w in EXPECTED_W],
            },
            "chunk_plan": {
                "chunks": 750,
                "chunk_size_decays": CHUNK_DECAYS,
                "one_job_per_chunk": True,
                "gateway_tasks": 1,
                "single_math_thread": True,
                "no_local_process_pool": True,
                "chunk_output": "phase_b/chunk_w<w>_n<j>_r<r'>.json",
                "merge_order": (
                    "the 50 replica files per node merged in replica "
                    "order r' = 0..49 (exact integer counts, float sums "
                    "in fixed order) into the 15 per-node diagnostics"
                ),
                "retry_cap": 3,
                "output_rule": "chunk outputs are appended only to the new run dir",
            },
            "metric": {
                "id": METRIC_ID,
                "tolerance_sigma": METRIC_N_SIGMA,
                "se_conventions": {
                    "multiplicity": (
                        "the per-node pooled standard error of the mean "
                        "over the 1,000,000 decays (the card-9 stored "
                        "'se')"
                    ),
                    "moments": (
                        "the between-replica standard error: std ddof=1 "
                        "over the 50 replica moments / sqrt(50)"
                    ),
                },
                "statement": (
                    "delta_analytic_full at 3.0 sigma with "
                    "SE_combined = sqrt(SE_j^2 + SE_{j+1}^2) over all "
                    "422 R31 checks (the same species, pairs, and "
                    "observables; the presence rule re-evaluated on the "
                    "new sample); per-check sigma at the old SE versus "
                    "the new SE"
                ),
                "tolerance_statement": (
                    "the 3.0-sigma tolerance is never relaxed; this run "
                    "is a T1-scaled statistics upgrade of the R31 "
                    "sample (same windows, grid, routing, metric), not "
                    "a new run tier and not a band change"
                ),
            },
            "gateway": {
                "host": "cl",
                "workers": ["c0801", "c0802", "c0803", "c0804", "c0805",
                            "c0806", "c0807", "c0808", "c0809"],
                "task_slots_per_worker": 32,
                "max_concurrent_slots": 288,
                "jobs_root": "~/.na49_cluster_gw/jobs",
                "remote_project_relpath": "NA49/2026/01-NA49-res",
            },
        },
        "draws": {
            "phase_a": 0,
            "phase_b": TOTAL_DECAYS,
            "note": (
                "Phase A: 0 new (deterministic replay of the sealed "
                "samples; the bootstrap resampler seed is recorded). "
                "Phase B: 15,000,000 decays (master 20260918); the "
                "smoke mode replays pre-registered confirmation triples "
                "at reduced counts, so no draw exists outside the "
                "pre-registered map"
            ),
        },
        "no_fitting": True,
        "optimizer_invoked": False,
        "no_fitting_note": (
            "no fitting of any kind (plan section 24, line 1865): no "
            "optimizer, no parameter adjustment, no tolerance change; a "
            "persistent mismatch at the confirmation statistics "
            "re-opens the engine-fix card (or the originating family "
            "card), never the tolerance"
        ),
        "no_tolerance_change": {
            "sigma_tolerance": METRIC_N_SIGMA,
            "unchanged": True,
            "engine_floor_note": (
                "DETERMINISTIC_SE_FLOOR = 1e-12 and the vacuous-zero "
                "rule are the card-8a P1 engine contract, applied "
                "unchanged by evaluate_pair"
            ),
        },
        "open_sets": open_sets_payload,
        "input_hashes": _input_hashes(),
    }


def run_pre_registration(run_dir: Path) -> None:
    """Write the pre-registration, provenance, and pre-investigation
    inventory, then stop.  Nothing is evaluated or generated."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    pre_path = run_dir / PREREGISTRATION_NAME
    if pre_path.exists():
        _refuse(
            f"{pre_path} already exists; the pre-registration is written "
            "once, before any evaluation or chunk generation (a rewrite "
            "would invalidate the run)"
        )
    if (run_dir / PHASE_B_NAME).is_dir() and any(
            (run_dir / PHASE_B_NAME).glob("chunk_*.json")):
        _refuse("phase_b chunk outputs already exist")
    catalog = HadronCatalog.from_json(NESTED_ROOT / CATALOG_REL)
    families = families_from_json(str(NESTED_ROOT / FAMILIES_REL))
    pre = build_pre_registration(catalog, families)
    _write_json(pre_path, pre)
    _write_json(
        run_dir / PRE_INVENTORY_NAME, build_pre_investigation_inventory()
    )
    hashes = _input_hashes()
    provenance = {
        "schema": "na49_mixing.staged_r3_se_audit.provenance/1",
        "run": RUN_ID,
        "card": CARD,
        "driver": NEW_DRIVER_REL,
        "driver_sha256": hashes["driver_sha256"],
        "r31_driver": R31_DRIVER_REL,
        "r31_driver_sha256": hashes["r31_driver_sha256"],
        "r31_driver_byte_identity_note": (
            "the R31 driver is frozen for this card (card 8b pins it); "
            "the nested and outer copies are byte-identical and the "
            "sealed R31 pre-registration binds its sha256 b47b2a85..."
        ),
        "fixed_engine": {
            "path": ENGINE_MODULE_REL,
            "sha256": _sha256_file(MONO_ROOT / ENGINE_MODULE_REL),
            "note": (
                "the card-2 engine after the 8a P1 fix (nested 95096d2): "
                "exact Kallen three-body slots, deep-level daughter "
                "quadrature, DETERMINISTIC_SE_FLOOR = 1e-12; "
                "SIGMA_TOLERANCE = 3.0 unchanged"
            ),
        },
        "git": _git_state(),
        "environment": environment_fingerprint(),
        "written_utc": _now_utc(),
        "written_before_evaluation": True,
        "catalog": {"path": CATALOG_REL, "sha256": hashes["catalog_sha256"]},
        "continuum_config": {
            "path": FAMILIES_REL,
            "sha256": hashes["families_config_sha256"],
        },
        "sealed_inputs": {
            "r31": {"path": R31_RUN_DIR_REL},
            "r21": {"path": R21_RUN_DIR_REL},
            "r22": {"path": R22_RUN_DIR_REL},
            "card2_reregistration": {"path": REREVAL_RUN_DIR_REL},
            "inventory": PRE_INVENTORY_NAME,
        },
        "gateway": {
            "identity": "sanitized pinned baseline "
                        "helpers/na49_gateway/config.na49_project.json",
            "host": "cl",
            "workers": ["c0801", "c0802", "c0803", "c0804", "c0805",
                        "c0806", "c0807", "c0808", "c0809"],
            "task_slots_per_worker": 32,
            "max_concurrent_slots": 288,
            "capacity_snapshot_at_preregistration": {
                "workers_free": 9,
                "free_tasks": 288,
                "active_tasks": 0,
                "note": (
                    "python3 -m helpers.na49_gateway.cluster_client "
                    "capacity, recorded before any submission"
                ),
            },
            "config_note": (
                "the pinned sanitized baseline "
                "helpers/na49_gateway/config.na49_project.json; the real "
                "site-specific settings live in the per-user config "
                "outside the repository"
            ),
        },
        "remote_checkout": {
            "path": "~/NA49/2026/01-NA49-res (remote_project_relpath "
                    "NA49/2026/01-NA49-res)",
            "git": "not a git repository (plain checkout; no VCS "
                   "metadata on the remote home mount)",
            "synced": [
                "dev/mixing2006/na49_mixing",
                "dev/mixing2006/helpers",
                "dev/mixing2006/config",
                "helpers/na49_gateway",
            ],
            "sync_note": (
                "the remote tree is synced from the local nested tree "
                "before submission; phase_b/submission_manifest.json "
                "records the remote driver sha256 verified at "
                "submission time"
            ),
        },
        "seeds": {
            "confirmation_master_seed": CONFIRMATION_MASTER_SEED,
            "r31_master_seed": 20260917,
            "r20_r22_master_seed": 20260913,
            "r23_raw_replica_seeds": "20300913..20300920",
            "disjointness_note": (
                "the confirmation master seed 20260918 is distinct from "
                "20260917 (R31), 20260913 (R20-R22), and the R23 raw "
                "replica seeds; the pre-registered empirical collision "
                "test covers the full maps"
            ),
        },
        "input_hashes": hashes,
    }
    _write_json(run_dir / PROVENANCE_NAME, provenance)
    print(
        f"staged_r3_se_audit: pre-registration written to {pre_path} "
        f"(Phase A hermetic replay + estimators; Phase B "
        f"{TOTAL_DECAYS} decays = 15 nodes x 50 x 20,000, master seed "
        f"{CONFIRMATION_MASTER_SEED}, 750 chunks; nothing evaluated or "
        "generated)"
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
    problems = []
    phase_b = pre.get("phase_b", {})
    stats = phase_b.get("statistics", {})
    if int(stats.get("replicas", -1)) != CONFIRMATION_REPLICAS:
        problems.append("statistics.replicas mismatch")
    if int(stats.get("decays_per_replica", -1)) != CHUNK_DECAYS:
        problems.append("statistics.decays_per_replica mismatch")
    if int(stats.get("decays_per_node", -1)) != DECAYS_PER_NODE:
        problems.append("statistics.decays_per_node mismatch")
    if pre.get("draws", {}).get("phase_b") != TOTAL_DECAYS:
        problems.append("draws.phase_b mismatch")
    if int(pre.get("draws", {}).get("phase_a", -1)) != 0:
        problems.append("draws.phase_a must be 0")
    seed_map = phase_b.get("seed_map", {})
    if int(seed_map.get("master_seed", -1)) != CONFIRMATION_MASTER_SEED:
        problems.append("master seed mismatch")
    if str(seed_map.get("rule", "")) != SEED_RULE:
        problems.append("seed rule mismatch")
    est = pre.get("phase_a", {}).get("estimator_definitions", {})
    if int(est.get("bootstrap_resampler_seed", -1)) != BOOTSTRAP_SEED:
        problems.append("bootstrap resampler seed mismatch")
    if [int(b) for b in est.get("block_lengths", [])] != list(BLOCK_LENGTHS):
        problems.append("block lengths mismatch")
    if int(est.get("bootstrap_resamples", -1)) != BOOTSTRAP_RESAMPLES:
        problems.append("bootstrap resamples mismatch")
    thresholds = pre.get("phase_a", {}).get("classification_thresholds", {})
    if float(thresholds.get("ratio_artifact", -1)) != RATIO_ARTIFACT:
        problems.append("ratio_artifact cutoff mismatch")
    if float(thresholds.get("ratio_genuine", -1)) != RATIO_GENUINE:
        problems.append("ratio_genuine cutoff mismatch")
    if float(thresholds.get("correlation_flag", -1)) != CORRELATION_FLAG:
        problems.append("correlation flag cutoff mismatch")
    named = pre.get("phase_a", {}).get("named_residuals", [])
    if len(named) != 10:
        problems.append("the named-residual list must carry the 10 "
                        "post-engine-fix failures")
    if len(pre.get("phase_a", {}).get("r21r22_named_residuals", [])) != 11:
        problems.append("the R21/R22 named set must carry 11 residuals")
    metric = pre.get("phase_b", {}).get("metric", {})
    if metric.get("id") != METRIC_ID:
        problems.append("metric id mismatch (the re-registered card-2 "
                        "metric is the contract)")
    if float(metric.get("tolerance_sigma", -1.0)) != METRIC_N_SIGMA:
        problems.append("metric tolerance mismatch")
    if pre.get("no_fitting") is not True:
        problems.append("no_fitting mismatch")
    if pre.get("no_tolerance_change", {}).get(
            "sigma_tolerance") != SIGMA_TOLERANCE:
        problems.append("no_tolerance_change.sigma_tolerance mismatch")
    hashes = pre.get("input_hashes", {})
    drift = []
    current_driver = _sha256_file(Path(__file__).resolve())
    if hashes.get("driver_sha256") != current_driver:
        drift.append({
            "field": "input_hashes.driver_sha256",
            "at_preregistration": hashes.get("driver_sha256"),
            "current": current_driver,
        })
    if problems:
        raise RuntimeError(
            "staged_r3_se_audit: the written pre-registration does not "
            "match this driver (the run is invalidated):\n  "
            + "\n  ".join(problems)
        )
    pre["_driver_sha256_drift"] = drift
    return pre


# ---------------------------------------------------------------------------
# The seed-parameterized per-decay function (design decision 1 of the
# card): a faithful wrapper of the committed R31 realization with the
# master seed carried as a parameter.  The R31 driver is NOT modified;
# the committed building blocks (route_source_decay, draw_mass,
# ResonanceDecayer, the kernel) are the contract.  The hermetic parity
# test asserts bit-identity against r31._decay_particles at the
# original master seed 20260917 on the pinned (window, node, replica,
# event) sample.
# ---------------------------------------------------------------------------


def _decay_particles_se(
    catalog: HadronCatalog,
    qn: tuple[int, int, int],
    mass: float,
    master_seed: int,
    replica: int,
    event_index: int,
    context: dict,
) -> tuple[str, list[tuple[int, tuple[float, float, float, float], int | None]]]:
    """One confirmation-map decay of an at-rest source ``(qn, mass)``:
    the R31 driver's per-decay realization with the master seed
    parameterized (the card-6 routing and the pre-registered per-route
    realization, unchanged).  Returns the route and the final particles
    as ``(pdg_id, four_momentum, parent_pdg_or_None)`` with the root
    parent the drawn record's pdg (catalog route) or the continuum
    sentinel 0 (continuum route)."""
    catalog_obj = context["catalog"]
    spec = context["spec"]
    fams = context["families"]
    mass_maps = context["mass_maps"]
    class_maps = context["class_maps"]
    stable = context["stable"]
    subs = derive_substreams(master_seed, replica, event_index)
    rng_record = np.random.default_rng(
        np.random.SeedSequence(subs.stream_for("decay_channel")).spawn(1)[0]
    )
    rng_mass = np.random.default_rng(
        np.random.SeedSequence(subs.stream_for("decay_kinematics")).spawn(1)[0]
    )
    decayer = ResonanceDecayer(catalog_obj, master_seed, replica, event_index)
    decayer._mass_maps.update(mass_maps)
    decayer._class_maps.update(class_maps)

    route = route_source_decay(
        tuple(int(v) for v in qn), float(mass), spec, catalog_obj, fams,
        rng_record,
    )

    if route == "catalog":
        open_records = context["open_sets"][(int(qn[0]), int(qn[1]),
                                             int(qn[2]))][float(mass)]
        u = float(rng_record.random())
        record = open_records[min(int(u * len(open_records)),
                                  len(open_records) - 1)]
        root_pdg = int(record.pdg_id)
        if root_pdg in stable or bool(record.stable_for_decay):
            return route, [(root_pdg, (float(mass), 0.0, 0.0, 0.0), None)]
        lo = float(record.mass_range[0])
        hi = min(float(record.mass_range[1]), float(mass))
        if hi - lo <= DEGENERATE_WINDOW_TOL:
            realized = float(mass)
        else:
            drawn = draw_mass(
                (lo, hi), record, mass_maps[root_pdg], rng_mass
            )
            if drawn is None:
                raise RuntimeError(
                    "staged_r3_se_audit: record "
                    f"{record.name} (pdg {root_pdg}) has no line-shape "
                    f"mass in the window {(lo, hi)!r} at node mass "
                    f"{mass!r} GeV; no fallback mass is implemented"
                )
            realized = float(drawn)
        vertices, finals = decayer.decay_source(KernelSourceRecord(
            pdg_id=root_pdg,
            four_momentum=(realized, 0.0, 0.0, 0.0),
            B=int(qn[0]), Q=int(qn[1]), S=int(qn[2]),
            mass=realized,
            source_id=int(event_index),
        ))
        if not finals:
            raise RuntimeError(
                "staged_r3_se_audit: the catalog-driven decay of "
                f"{record.name} (pdg {root_pdg}) at mass {realized!r} GeV "
                "produced no final particles (a non-stable record must "
                "decay; the run stops)"
            )
        out = []
        for final in finals:
            parent_pdg = (
                int(vertices[int(final.parent_id)].parent_pdg)
                if final.parent_id is not None else None
            )
            p4 = final.p4
            out.append((
                int(final.pdg_id),
                (float(p4.e), float(p4.px), float(p4.py), float(p4.pz)),
                parent_pdg,
            ))
        return route, out

    # -- the continuum route ---------------------------------------------
    family = context["family_of"][tuple(int(v) for v in qn)]
    n_classes = context["n_classes_of"][tuple(int(v) for v in qn)]
    realized = float(mass)
    u_cls = float(rng_record.random())
    classes = family.allowed_decay_classes
    cls = classes[min(int(u_cls * n_classes), n_classes - 1)]
    d0, d1 = (int(v) for v in str(cls).strip("()").split(","))
    u_cos = float(rng_mass.random())
    u_phi = float(rng_mass.random())
    cos_theta = 2.0 * u_cos - 1.0
    phi = 2.0 * math.pi * u_phi
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    direction = (
        sin_theta * math.cos(phi),
        sin_theta * math.sin(phi),
        cos_theta,
    )
    rec0 = catalog_obj.get(d0)
    rec1 = catalog_obj.get(d1)
    res0 = d0 not in context["stable"]
    res1 = d1 not in context["stable"]
    m0_pole = float(rec0.pole_mass)
    m1_pole = float(rec1.pole_mass)
    if res0:
        lo0 = float(rec0.mass_range[0])
        hi0 = min(float(rec0.mass_range[1]), realized - m1_pole)
        drawn0 = draw_mass((lo0, hi0), rec0, mass_maps[d0],
                           rng_mass)
        if drawn0 is None:
            raise RuntimeError(
                "staged_r3_se_audit: the continuum root daughter "
                f"{rec0.name} (pdg {d0}) has an empty on-shell budget "
                f"window {(lo0, hi0)!r} at realized mass {realized!r}; "
                "no fallback mass is implemented"
            )
        mass0 = float(drawn0)
    else:
        mass0 = m0_pole
    if res1:
        lo1 = float(rec1.mass_range[0])
        hi1 = min(float(rec1.mass_range[1]), realized - mass0)
        drawn1 = draw_mass((lo1, hi1), rec1, mass_maps[d1],
                           rng_mass)
        if drawn1 is None:
            raise RuntimeError(
                "staged_r3_se_audit: the continuum root daughter "
                f"{rec1.name} (pdg {d1}) has an empty on-shell budget "
                f"window {(lo1, hi1)!r} at realized mass {realized!r}; "
                "no fallback mass is implemented"
            )
        mass1 = float(drawn1)
    else:
        mass1 = m1_pole
    q_star = float(decay_momentum(realized, mass0, mass1))
    energy0 = (realized * realized + mass0 * mass0 - mass1 * mass1) \
        / (2.0 * realized)
    energy1 = (realized * realized - mass0 * mass0 + mass1 * mass1) \
        / (2.0 * realized)
    out = []

    def _emit(pdg: int, is_res: bool, energy: float,
              momentum: tuple[float, float, float]) -> None:
        rest = (energy, momentum[0], momentum[1], momentum[2])
        if is_res:
            vertices = decayer.decay_resonance(pdg, rest)
            for vertex in vertices:
                for daughter in vertex.daughters:
                    if daughter.is_resonance:
                        continue
                    p4 = daughter.four_momentum
                    out.append((
                        int(daughter.pdg_id),
                        (float(p4[0]), float(p4[1]), float(p4[2]),
                         float(p4[3])),
                        int(vertex.parent_pdg),
                    ))
        else:
            out.append((int(pdg), rest, CONTINUUM_ROOT_SENTINEL))

    _emit(d0, res0, energy0,
          (q_star * direction[0], q_star * direction[1],
           q_star * direction[2]))
    _emit(d1, res1, energy1,
          (-q_star * direction[0], -q_star * direction[1],
           -q_star * direction[2]))
    if not out:
        raise RuntimeError(
            "staged_r3_se_audit: the continuum decay of "
            f"{family.family_id} class {cls!r} at realized mass "
            f"{realized!r} produced no final particles (the run stops)"
        )
    return route, out


#: Module-level worker context for the fork pool (set once in the
#: parent before the pool is created; ``None`` on the main path).
_AUDIT_CONTEXT: dict[str, Any] | None = None
_AUDIT_CAPTURE: tuple[str, ...] = ()


def _set_audit_context(context: dict, capture: tuple[str, ...]) -> None:
    global _AUDIT_CONTEXT, _AUDIT_CAPTURE
    _AUDIT_CONTEXT = context
    _AUDIT_CAPTURE = tuple(capture)


def _skey_of(catalog: HadronCatalog, pdg: int) -> str:
    """The species key of a final particle (the card-9 transliteration;
    cached per worker process, the same values the committed
    accumulator derives)."""
    key = _AUDIT_SKEY_CACHE.get(pdg)
    if key is None:
        key = _species_key(_record_name(catalog, pdg))
        _AUDIT_SKEY_CACHE[pdg] = key
    return key


_AUDIT_SKEY_CACHE: dict[int, str] = {}


def _capture_event(particles: list, capture_index: dict,
                   capture_arrays: dict, row: int,
                   catalog: HadronCatalog) -> None:
    """One event's per-species contribution tuple
    ``(count, p_sum, e_sum, ept_sum)`` into the capture arrays."""
    touched: dict[str, list] = {}
    for pdg, p4, _parent in particles:
        e, px, py, pz = p4
        key = _skey_of(catalog, pdg)
        if key not in capture_index:
            continue
        bucket = touched.get(key)
        if bucket is None:
            bucket = touched[key] = [0, 0.0, 0.0, 0.0]
        bucket[0] += 1
        bucket[1] += math.sqrt(px * px + py * py + pz * pz)
        bucket[2] += e
        bucket[3] += e * math.hypot(px, py)
    for key, values in touched.items():
        arr = capture_arrays[key]
        arr[row, 0] += values[0]
        arr[row, 1] += values[1]
        arr[row, 2] += values[2]
        arr[row, 3] += values[3]


# ---------------------------------------------------------------------------
# Phase A: the bit-identical deterministic replay of the sealed samples
# (zero new random numbers) with per-event capture, the cross-check
# against the sealed aggregates (exact, before any estimator), the
# E1-E4 battery, the seed-map audit, and the classifications.
# ---------------------------------------------------------------------------


def _audit_r31_chunk(task: tuple):
    """One R31 replay chunk in a worker process: the committed R31
    per-decay function (``r31._decay_particles``, the original master
    seed) and the committed accumulator, plus the per-event capture for
    the cell's species set.  Returns ``(start, acc, captures)``."""
    assert _AUDIT_CONTEXT is not None
    window, node, replica, start, stop = task
    context = _AUDIT_CONTEXT
    catalog = context["catalog"]
    acc = r31._Accumulator(context["species_keys"])
    qn = tuple(int(v) for v in r31.WINDOWS[window]["qn"])
    mass = float(r31.NODE_MASSES[node])
    base = (window * r31.N_NODES + node) * 1_000_000
    capture_arrays = {
        key: np.zeros((stop - start, 4), dtype=np.float64)
        for key in _AUDIT_CAPTURE
    }
    capture_index = {key: i for i, key in enumerate(_AUDIT_CAPTURE)}
    for offset in range(start, stop):
        event_index = base + offset
        route, particles = r31._decay_particles(
            catalog, qn, mass, replica, event_index, context
        )
        r31._accumulate_event(acc, catalog, context["species_index"], route,
                              particles, qn)
        _capture_event(particles, capture_index, capture_arrays,
                       offset - start, catalog)
    return window, node, replica, start, acc, capture_arrays


def _audit_grid_chunk(task: tuple):
    """One card-9 (point, replica) replay chunk in a worker process:
    the committed R21/R22 driver's per-decay realization (the master
    seed recorded in the sealed run's pre-registration), the committed
    ``staged_source_grid`` accumulation, plus the per-event capture.
    The replica is accumulated sequentially in one accumulator (the
    card-9 convention) so the replayed sums are bit-identical to the
    sealed per-replica aggregates.  Returns ``(point, replica, acc,
    captures)``."""
    assert _AUDIT_CONTEXT is not None
    run_label, point, replica, decays = task
    context = _AUDIT_CONTEXT
    catalog = context["catalog"]
    cell = context["grid_cells"][(run_label, point)]
    records = cell["records"]
    mass = float(cell["mass"])
    source_qn = tuple(int(v) for v in cell["source_qn"])
    master_seed = int(cell["master_seed"])
    daughter_maps = cell["daughter_maps"]
    acc = sgrid._Accumulator(cell["species_keys"])
    capture_arrays = {
        key: np.zeros((decays, 4), dtype=np.float64)
        for key in _AUDIT_CAPTURE
    }
    capture_index = {key: i for i, key in enumerate(_AUDIT_CAPTURE)}
    for i in range(decays):
        event_index = point * 1_000_000 + i
        subs = derive_substreams(master_seed, replica, event_index)
        rng_record = np.random.default_rng(
            np.random.SeedSequence(
                subs.stream_for("decay_channel")
            ).spawn(1)[0]
        )
        rng_mass = np.random.default_rng(
            np.random.SeedSequence(
                subs.stream_for("decay_kinematics")
            ).spawn(1)[0]
        )
        record, realized = sgrid._realize_record(
            catalog, records, mass, rng_record, rng_mass, daughter_maps,
        )
        decayer = ResonanceDecayer(catalog, master_seed, replica, event_index)
        vertices, finals = decayer.decay_source(sgrid.SourceRecord(
            pdg_id=int(record.pdg_id),
            four_momentum=(realized, 0.0, 0.0, 0.0),
            B=int(source_qn[0]),
            Q=int(source_qn[1]),
            S=int(source_qn[2]),
            mass=float(realized),
            source_id=int(event_index),
        ))
        if not finals:
            finals = [FinalParticle(
                particle_id=0,
                pdg_id=int(record.pdg_id),
                p4=FourVector(float(realized), 0.0, 0.0, 0.0),
                source_id=int(event_index),
                parent_id=None,
                channel_family="source_stable",
            )]
        sgrid._accumulate_event(
            acc, catalog, vertices, finals,
            int(record.pdg_id), source_qn,
        )
        flat = [
            (int(final.pdg_id),
             (float(final.p4.e), float(final.p4.px), float(final.p4.py),
              float(final.p4.pz)),
             None)
            for final in finals
        ]
        _capture_event(flat, capture_index, capture_arrays, i, catalog)
    return run_label, point, replica, acc, capture_arrays


def _map_replay_chunks(tasks: list, workers: int, label: str, worker_fn):
    """Yield replay worker results as they complete over a fork pool
    (the committed R31 chunk-pool convention; analysis-only)."""
    context = multiprocessing.get_context("fork")
    completed = 0
    started = time.perf_counter()
    with context.Pool(processes=workers) as pool:
        for result in pool.imap_unordered(worker_fn, tasks, chunksize=1):
            completed += 1
            yield result
            if completed % 20 == 0 or completed == len(tasks):
                elapsed = time.perf_counter() - started
                print(
                    f"[r3-se-audit] {label}: {completed}/{len(tasks)} "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )


# ---------------------------------------------------------------------------
# The E1-E4 battery (the card's pre-registered estimator definitions).
# ---------------------------------------------------------------------------


def _e1_between_replica(values: list) -> float:
    """E1 for a moment: std ddof=1 of the stored replica values over
    sqrt(n) (the existing convention)."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(max(variance, 0.0) / n)


def _e3_jackknife(values: list) -> float:
    """E3: leave-one-out replica jackknife,
    SE = sqrt((n-1)/n * sum (m_i - m_bar)^2)."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(max((n - 1) / n * sum((v - mean) ** 2 for v in values),
                         0.0))


def _moving_block_bootstrap(contrib: np.ndarray, b: int, n_resamples: int,
                            rng: np.random.Generator) -> dict:
    """The moving-block bootstrap of one contribution sample *contrib*
    (shape (N, 4): count, p_sum, e_sum, ept_sum per event): L = ceil(N/b)
    blocks per resample drawn uniformly from the N - b + 1 legal starts,
    the per-resample statistics (multiplicity, mean_p, mean_pt_y) from
    the block sums, SE = std ddof=1 over the B resample values."""
    contrib = np.ascontiguousarray(contrib, dtype=np.float64)
    n = int(contrib.shape[0])
    n_blocks = -(-n // b)
    total_len = n_blocks * b
    csum = np.zeros((n + 1, 4), dtype=np.float64)
    np.cumsum(contrib, axis=0, out=csum[1:])
    stats = np.empty((n_resamples, 3), dtype=np.float64)
    slice_size = max(1, int(2_000_000 // max(n_blocks, 1)))
    for i0 in range(0, n_resamples, slice_size):
        i1 = min(i0 + slice_size, n_resamples)
        starts = rng.integers(0, n - b + 1, size=(i1 - i0, n_blocks))
        blocks = csum[starts + b] - csum[starts]
        totals = blocks.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            count = totals[:, 0]
            stats[i0:i1, 0] = count / float(total_len)
            stats[i0:i1, 1] = np.where(
                count > 0.0,
                totals[:, 1] / np.where(count > 0.0, count, 1.0),
                np.nan,
            )
            energy = totals[:, 2]
            stats[i0:i1, 2] = np.where(
                energy > 0.0,
                totals[:, 3] / np.where(energy > 0.0, energy, 1.0),
                np.nan,
            )
    out: dict[str, Any] = {"b": int(b), "n_resamples": int(n_resamples),
                           "resampled_events": int(total_len)}
    undefined = {}
    for col, name in ((0, "multiplicity"), (1, "mean_p"), (2, "mean_pt_y")):
        vals = stats[:, col]
        finite = np.isfinite(vals)
        n_finite = int(finite.sum())
        undefined[name] = n_resamples - n_finite
        out["se_" + name] = (
            float(np.std(vals[finite], ddof=1)) if n_finite >= 2
            else float("nan")
        )
    out["undefined_resamples"] = undefined
    return out


def _observable_se(boot: dict, observable: str) -> float:
    return float(boot["se_" + observable])


def _estimator_entry(replica_records: list, node_record: dict | None,
                     skey: str, replica_captures: list,
                     pooled: np.ndarray | None,
                     rng: np.random.Generator, n_replicas: int) -> dict:
    """The full E1-E4 entry for one (cell, species): E1/E3 from the
    sealed per-replica values, E2 (b = 32, 128) per replica with the
    node-level mean over sqrt(n), E4 pooled single-sample (b = 32, 128)
    from the replayed per-event contribution tuples."""
    entry: dict[str, Any] = {}
    # -- E1 (the stored convention) and E3 -------------------------------
    e1: dict[str, Any] = {}
    e3: dict[str, Any] = {}
    for observable in OBSERVABLES:
        if observable == "multiplicity":
            block = (node_record or {}).get("multiplicities", {}).get(skey)
            e1[observable] = float(block["se"]) if block is not None else None
            values = []
            missing = False
            for record in replica_records:
                mblock = record.get("multiplicities", {}).get(skey)
                if mblock is None:
                    missing = True
                    break
                values.append(float(mblock["mean"]))
            e3[observable] = (
                None if (missing or len(values) != n_replicas)
                else _e3_jackknife(values)
            )
        else:
            field = MOMENT_FIELD[observable]
            values = []
            missing = False
            for record in replica_records:
                block = record.get("momentum_moments", {}).get(skey)
                if block is None:
                    missing = True
                    break
                values.append(float(block[field]))
            if missing or len(values) != n_replicas:
                e1[observable] = None
                e3[observable] = None
            else:
                e1[observable] = _e1_between_replica(values)
                e3[observable] = _e3_jackknife(values)
    entry["e1"] = e1
    entry["e3"] = e3
    # -- E2 (per replica + node level) and E4 (pooled) --------------------
    e2: dict[str, Any] = {}
    e4: dict[str, Any] = {}
    for b in BLOCK_LENGTHS:
        per_replica = []
        for contrib in replica_captures:
            boot = _moving_block_bootstrap(contrib, int(b),
                                           BOOTSTRAP_RESAMPLES, rng)
            per_replica.append(boot)
        node_level = {}
        for observable in OBSERVABLES:
            vals = [float(boot["se_" + observable]) for boot in per_replica]
            finite = [v for v in vals if math.isfinite(v)]
            node_level[observable] = (
                float(sum(finite) / len(finite) / math.sqrt(n_replicas))
                if finite else float("nan")
            )
        e2["b%d" % b] = {
            "per_replica": {
                observable: [
                    float(v) if math.isfinite(v) else "NaN"
                    for v in (float(boot["se_" + observable])
                              for boot in per_replica)
                ]
                for observable in OBSERVABLES
            },
            "node_level": node_level,
        }
        if pooled is not None:
            boot = _moving_block_bootstrap(pooled, int(b), BOOTSTRAP_RESAMPLES,
                                           rng)
            e4["b%d" % b] = {
                "se": {
                    observable: float(boot["se_" + observable])
                    for observable in OBSERVABLES
                },
                "undefined_resamples": boot["undefined_resamples"],
                "resampled_events": int(boot["resampled_events"]),
            }
    entry["e2"] = e2
    entry["e4"] = e4
    ratios: dict[str, Any] = {}
    b0 = "b%d" % BLOCK_LENGTHS[0]
    for observable in OBSERVABLES:
        e1_v = e1.get(observable)
        e4_v = (e4.get(b0) or {}).get("se", {}).get(observable)
        if e1_v is None or e4_v is None or not (float(e1_v) > 0.0) \
                or not math.isfinite(float(e4_v)):
            ratios[observable] = None
        else:
            ratios[observable] = float(e4_v) / float(e1_v)
    entry["ratio_e4_b32_over_e1"] = ratios
    return entry


def _payload_of_r31_acc(acc, window: int, node: int, record_pdgs: list,
                        mode: str = "production") -> dict:
    """The card-9 payload of one replayed R31 sample (the committed
    builder, the R31 run identity) for the sealed cross-check."""
    qn = (int(r31.WINDOWS[window]["qn"][0]),
          int(r31.WINDOWS[window]["qn"][1]),
          int(r31.WINDOWS[window]["qn"][2]))
    return r31._diagnostics_payload(
        acc, window, node, float(r31.NODE_MASSES[node]), qn, record_pdgs,
        mode,
    )


def is_exact_payload(replayed: dict, sealed: dict) -> bool:
    """Exact payload equality (bit for bit, every stored sum)."""
    return replayed == sealed


def _refuse_replay_mismatch(cell: str, kind: str, replica: int) -> NoReturn:
    raise RuntimeError(
        "staged_r3_se_audit: the deterministic replay of the sealed "
        f"sample at {cell} ({kind}, replica {replica}) does not match "
        "the sealed aggregates bit for bit; the cross-check precedes "
        "every estimator computation and the run stops (the replay "
        "uses the same committed driver code and the original seeds)"
    )


def _cell_estimators(sealed_replicas: list, sealed_node: dict,
                     capture: list, cell_captures: dict,
                     rng: np.random.Generator, n_replicas: int,
                     species_keys: tuple) -> dict:
    """The per-cell estimator entries: one E1-E4 entry per captured
    species, in catalog stable order (the pre-registered rng order)."""
    out: dict[str, Any] = {"species": {}}
    for skey in species_keys:
        if skey not in capture:
            continue
        replica_captures = cell_captures.get(skey, [])
        if len(replica_captures) != n_replicas:
            replica_captures = [
                np.zeros((r31.T1_DECAYS_PER_REPLICA, 4)) for _ in
                range(n_replicas)
            ]
        pooled = np.concatenate(replica_captures, axis=0)
        out["species"][skey] = _estimator_entry(
            sealed_replicas, sealed_node, skey, replica_captures, pooled,
            rng, n_replicas,
        )
    out["capture"] = list(capture)
    return out


def _replay_r31(context: dict, workers: int, capture_sets: dict,
                rng: np.random.Generator,
                sealed_replicas: dict, sealed_nodes: dict,
                species_keys: tuple) -> tuple[dict, dict]:
    """Replay the sealed R31 sample (all 15 cells, 4 replicas x 25,000,
    zero new random numbers) through the committed R31 code, cross-check
    every per-replica and per-node payload against the sealed
    aggregates exactly, capture the per-event contribution tuples for
    the cell's species set, and compute the estimator entries in the
    pre-registered rng order.  Returns ``(estimator_cells, replay_
    report)``."""
    replay_report: dict[str, Any] = {
        "cells": N_WINDOWS * N_NODES,
        "replicas_per_cell": r31.T1_REPLICAS,
        "decays_per_replica": r31.T1_DECAYS_PER_REPLICA,
        "chunk_size": REPLAY_CHUNK_SIZE,
        "cross_check": "exact payload equality (bit for bit)",
        "cells_checked": {},
        "replicas_checked": 0, "nodes_checked": 0, "all_exact": True,
    }
    estimator_cells: dict[str, Any] = {}
    for w in range(N_WINDOWS):
        for j in range(N_NODES):
            capture = capture_sets[(w, j)]
            _set_audit_context(context, tuple(capture))
            tasks = []
            for replica in range(r31.T1_REPLICAS):
                for start in range(0, r31.T1_DECAYS_PER_REPLICA,
                                   REPLAY_CHUNK_SIZE):
                    stop = min(start + REPLAY_CHUNK_SIZE,
                               r31.T1_DECAYS_PER_REPLICA)
                    tasks.append((w, j, replica, start, stop))
            replica_accs: dict[int, Any] = {}
            replica_captures: dict[int, list] = {
                r: [] for r in range(r31.T1_REPLICAS)
            }
            for (rw, rj, rr, start, acc, captures) in _map_replay_chunks(
                    tasks, workers, "R31 replay w%d n%d" % (w, j),
                    _audit_r31_chunk):
                replica_accs.setdefault(rr, []).append((start, acc))
                # the capture blocks are stored WITH their chunk start and
                # concatenated in chunk order below (the same fixed
                # float-addition order as the accumulator merge); an
                # arrival-order concatenation would permute the per-event
                # rows run-to-run and break the estimator reproducibility
                replica_captures[rr].append((start, captures))
            record_pdgs = [int(p) for p in
                           sealed_replicas[(w, j)][0]["open_records"]]
            exact = True
            cell_captures: dict[str, list] = {}
            for replica in range(r31.T1_REPLICAS):
                parts = sorted(replica_accs[replica], key=lambda part: part[0])
                merged = r31._Accumulator(context["species_keys"])
                for _start, part_acc in parts:
                    merged.merge(part_acc)
                payload = _payload_of_r31_acc(merged, w, j, record_pdgs)
                sealed = sealed_replicas[(w, j)][replica]
                if not is_exact_payload(payload, sealed):
                    exact = False
                replay_report["replicas_checked"] += 1
                blocks_by_key = {
                    key: [caps[key] for _start, caps in
                          sorted(replica_captures[replica],
                                 key=lambda part: part[0])]
                    for key in capture
                }
                replica_captures[replica] = None
                for key in capture:
                    blocks = blocks_by_key[key]
                    cell_captures.setdefault(key, []).append(
                        np.concatenate(blocks, axis=0) if blocks
                        else np.zeros((r31.T1_DECAYS_PER_REPLICA, 4))
                    )
                    del blocks
            estimator_cells["w%d_j%d" % (w, j)] = _cell_estimators(
                sealed_replicas[(w, j)], sealed_nodes[(w, j)],
                list(capture), cell_captures, rng,
                r31.T1_REPLICAS, species_keys,
            )
            node_acc = r31._Accumulator(context["species_keys"])
            for replica in range(r31.T1_REPLICAS):
                rep_acc = r31._Accumulator(context["species_keys"])
                for _start, part_acc in sorted(replica_accs[replica],
                                               key=lambda part: part[0]):
                    rep_acc.merge(part_acc)
                node_acc.merge(rep_acc)
            node_payload = _payload_of_r31_acc(node_acc, w, j, record_pdgs)
            node_payload["replicas"] = int(r31.T1_REPLICAS)
            node_payload["decays_per_replica"] = int(
                r31.T1_DECAYS_PER_REPLICA)
            if not is_exact_payload(node_payload, sealed_nodes[(w, j)]):
                exact = False
            replay_report["cells_checked"]["w%d_j%d" % (w, j)] = {
                "exact": bool(exact),
            }
            replay_report["all_exact"] = bool(
                replay_report["all_exact"] and exact
            )
            replay_report["nodes_checked"] = (
                replay_report.get("nodes_checked", 0) + 1
            )
    return estimator_cells, replay_report


def _grid_cell_key(run_label: str, point: int) -> str:
    return "%s_k%02d" % (run_label, point)


def _replay_grid(run_label: str, run_dir_rel: str, catalog: HadronCatalog,
                 context: dict, workers: int, capture_sets: dict,
                 rng: np.random.Generator, species_keys: tuple,
                 estimator_cells: dict) -> dict:
    """Replay the named-residual point cells of one sealed card-9 grid
    (R21 or R22) at the seeds recorded in the sealed run's
    pre_registration.json, cross-check every per-replica and per-point
    payload against the sealed aggregates exactly, capture the
    per-event contribution tuples for the residual species, and compute
    the estimator entries in the pre-registered rng order.  Returns the
    replay report; the estimator entries land in *estimator_cells*."""
    run_dir = MONO_ROOT / run_dir_rel
    sealed_pre = _read_json(run_dir / PREREGISTRATION_NAME)
    master_seed = int(sealed_pre["seed_map"]["master_seed"])
    n_points = len(sealed_pre["grid_masses_gev"])
    replay_report: dict[str, Any] = {
        "run": run_label,
        "pre_registration": sealed_pre.get("schema"),
        "master_seed": int(master_seed),
        "seed_rule": sealed_pre["seed_map"]["rule"],
        "cells": [], "replicas_checked": 0, "points_checked": 0,
        "all_exact": True,
    }
    for k in range(n_points):
        capture = capture_sets.get((run_label, k))
        if capture is None:
            continue
        mass = float(sealed_pre["grid_masses_gev"][k])
        source_qn = tuple(int(v) for v in sealed_pre["source_qn"])
        records = sgrid._open_set(catalog, source_qn, mass)
        if not records:
            _refuse("the card-9 open set at point %d is empty" % k)
        record_pdgs = [int(r.pdg_id) for r in records]
        replica_records = []
        for replica in range(sgrid.T1_REPLICAS):
            path = run_dir / ("replica_%d_point_%02d.json" % (replica, k))
            replica_records.append(_read_json(path))
        if [int(p) for p in replica_records[0]["open_records"]] != record_pdgs:
            _refuse(
                "the replayed card-9 open set at %s point %d differs "
                "from the sealed open_records" % (run_label, k)
            )
        sealed_node = _read_json(run_dir / ("point_%02d.json" % k))
        context["grid_cells"][(run_label, k)] = {
            "records": records,
            "mass": mass,
            "source_qn": source_qn,
            "master_seed": master_seed,
            "daughter_maps": {},
            "species_keys": species_keys,
        }
        _set_audit_context(context, tuple(capture))
        tasks = [(run_label, k, replica, sgrid.T1_DECAYS_PER_REPLICA)
                 for replica in range(sgrid.T1_REPLICAS)]
        replica_accs: dict[int, Any] = {}
        replica_captures: dict[int, dict] = {
            r: {key: None for key in capture}
            for r in range(sgrid.T1_REPLICAS)
        }
        for (rl, rk, rr, acc, captures) in _map_replay_chunks(
                tasks, workers, "%s replay k%d" % (run_label, k),
                _audit_grid_chunk):
            replica_accs[rr] = acc
            replica_captures[rr] = captures
        cell_captures: dict[str, list] = {}
        exact = True
        for replica in range(sgrid.T1_REPLICAS):
            acc = replica_accs[replica]
            payload = sgrid._diagnostics_payload(
                acc, k, mass, source_qn, record_pdgs
            )
            if not is_exact_payload(payload, replica_records[replica]):
                exact = False
            replay_report["replicas_checked"] += 1
            for key in capture:
                arr = replica_captures[replica][key]
                cell_captures.setdefault(key, []).append(
                    arr if arr is not None
                    else np.zeros((sgrid.T1_DECAYS_PER_REPLICA, 4))
                )
        point_acc = sgrid._Accumulator(species_keys)
        for replica in range(sgrid.T1_REPLICAS):
            point_acc.merge(replica_accs[replica])
        point_payload = sgrid._diagnostics_payload(
            point_acc, k, mass, source_qn, record_pdgs,
        )
        if not is_exact_payload(point_payload, sealed_node):
            exact = False
        replay_report["cells"].append({"point": int(k), "exact": bool(exact)})
        replay_report["all_exact"] = bool(
            replay_report.get("all_exact", True) and exact
        )
        estimator_cells["k%02d" % k] = _cell_estimators(
            replica_records, sealed_node, list(capture), cell_captures,
            rng, sgrid.T1_REPLICAS, species_keys,
        )
    replay_report["points_checked"] = len(replay_report["cells"])
    return replay_report


    replay_report["points_checked"] = len(replay_report["cells"])
    return replay_report


# ---------------------------------------------------------------------------
# Seed-map audit (the pre-registered procedure): stream-collision test,
# degenerate-stream check, and the replica-correlation test.
# ---------------------------------------------------------------------------

_MAP_SEED_R31 = 20260917
_GRID_SEED_R21R22 = 20260913
_R23_SAMPLE_SEEDS = (20300913, 20300914, 20300917, 20300920)


def _derive_id_rows(triples: list) -> np.ndarray:
    """The (N, 2) uint64 rows of the two audited stream ids per triple."""
    rows = np.empty((len(triples), 2), dtype=np.uint64)
    for i, (ms, ri, ei) in enumerate(triples):
        subs = derive_substreams(ms, ri, ei)
        rows[i, 0] = subs.stream_for("decay_channel")
        rows[i, 1] = subs.stream_for("decay_kinematics")
    return rows


def _rows_unique(rows: np.ndarray) -> tuple[np.ndarray, int]:
    unique = np.unique(rows, axis=0)
    return unique, int(unique.shape[0])


def _cell_triples_r31(w: int, j: int) -> list:
    base = (w * 5 + j) * 1_000_000
    return [
        (_MAP_SEED_R31, replica, base + i)
        for replica in range(4)
        for i in range(r31.T1_DECAYS_PER_REPLICA)
    ]


def _cell_triples_confirmation(w: int, j: int) -> list:
    base = (w * 5 + j) * 100_000_000
    return [
        (CONFIRMATION_MASTER_SEED, rp, base + rp * 20_000 + i)
        for rp in range(CONFIRMATION_REPLICAS)
        for i in range(CHUNK_DECAYS)
    ]


def _cell_triples_grid(k: int) -> list:
    return [
        (_GRID_SEED_R21R22, replica, k * 1_000_000 + i)
        for replica in range(4)
        for i in range(r31.T1_DECAYS_PER_REPLICA)
    ]


def _seed_audit_worker(task: tuple) -> tuple:
    """Derive one cell's id rows (the pool task for the seed audit)."""
    label, triples = task
    return label, _derive_id_rows(triples)


def _r21r22_audit_cells() -> dict:
    """The R21/R22 replayed-cell set of the 11 named residuals: both
    nodes of each residual pair."""
    cells: dict = {"R21": set(), "R22": set()}
    for residual in R21R22_NAMED_RESIDUALS:
        cells[residual["run"]].add(int(residual["pair"][0]))
        cells[residual["run"]].add(int(residual["pair"][1]))
    return {run: sorted(keys) for run, keys in cells.items()}


_R21R22_AUDIT_CELLS: dict = {}


def _prepare_r21r22_audit_cells() -> None:
    global _R21R22_AUDIT_CELLS
    _R21R22_AUDIT_CELLS = _r21r22_audit_cells()


def run_seed_map_audit(workers: int) -> dict:
    """The pre-registered seed-map audit: the stream-collision test of
    the two closed labels over the full pre-registered event sample per
    node (the full R31 map, the full confirmation map, the R21/R22
    replayed cells, sampled R20-style and R23-style triples), the
    degenerate-stream check (duplicate triples), and the cross-map
    disjointness (zero shared id rows expected across the union)."""
    print("[r3-se-audit] seed-map audit: deriving stream ids ...",
          flush=True)
    maps: dict = {
        "r31_full": [],
        "confirmation_full": [],
        "r21r22_replayed_cells": [],
        "r20_r23_sampled": [],
    }
    for w in range(N_WINDOWS):
        for j in range(N_NODES):
            maps["r31_full"].append(("r31_w%d_j%d" % (w, j),
                                     _cell_triples_r31(w, j)))
            maps["confirmation_full"].append(
                ("conf_w%d_j%d" % (w, j), _cell_triples_confirmation(w, j))
            )
    for run_label, cells in sorted(_R21R22_AUDIT_CELLS.items()):
        for k in cells:
            maps["r21r22_replayed_cells"].append(
                ("%s_k%02d" % (run_label, k), _cell_triples_grid(k))
            )
    # Sampled R20-style triples (master 20260913, event_index =
    # s * 1_000_000 + j * 10_000 + i, the R20 unit-grid map) and
    # R23-style triples (the raw replica seeds 20300913..20300920 as
    # master seeds of their own map).
    r20_sample = [
        (_GRID_SEED_R21R22, replica, s * 1_000_000 + j * 10_000 + i)
        for s in range(2)
        for j in range(5)
        for replica in range(2)
        for i in range(0, 10_000, 1_000)
    ]
    r23_sample = [
        (seed, 0, i) for seed in _R23_SAMPLE_SEEDS
        for i in range(0, 20_000, 137)
    ]
    maps["r20_r23_sampled"] = [
        ("r20_style", r20_sample), ("r23_style", r23_sample),
    ]
    tasks = [(label, triples)
             for group in maps.values() for label, triples in group]
    id_rows: dict = {}
    for label, rows in _map_replay_chunks(
            tasks, workers, "seed-map audit", _seed_audit_worker):
        id_rows[label] = rows
    maps_payload: dict = {}
    group_rows: dict = {}
    for group in sorted(maps):
        entries = maps[group]
        n_triples = sum(id_rows[label].shape[0] for label, _r in entries)
        combined = (
            np.concatenate([id_rows[label] for label, _r in entries], axis=0)
            if entries else np.empty((0, 2), dtype=np.uint64)
        )
        _unique, n_unique = _rows_unique(combined)
        group_rows[group] = _unique
        maps_payload[group] = {
            "cells": len(entries),
            "triples": int(n_triples),
            "unique_id_rows": int(n_unique),
            "duplicate_id_rows": int(n_triples - n_unique),
            "zero_duplicates": bool(n_triples == n_unique),
        }
    # pairwise shared-row counts between the covered map groups (the
    # shared-row count of a pair is n1 + n2 - unique(union))
    group_names = sorted(group_rows)
    pairwise: dict = {}
    for i, ga in enumerate(group_names):
        for gb in group_names[i + 1:]:
            rows_a = group_rows[ga]
            rows_b = group_rows[gb]
            _u, n_union = _rows_unique(np.concatenate([rows_a, rows_b],
                                                      axis=0))
            shared = int(rows_a.shape[0] + rows_b.shape[0] - n_union)
            pairwise["%s|%s" % (ga, gb)] = shared
    conf_rows = group_rows["confirmation_full"]
    others = [group_rows[g] for g in group_names if g != "confirmation_full"]
    conf_disjoint = True
    for rows in others:
        _u, n_union = _rows_unique(np.concatenate([conf_rows, rows], axis=0))
        if int(conf_rows.shape[0] + rows.shape[0] - n_union) != 0:
            conf_disjoint = False
    archive_overlaps = {
        key: value for key, value in pairwise.items()
        if value and "confirmation_full" not in key
    }
    audit = {
        "substream_labels": list(SUBSTREAM_LABELS_AUDIT),
        "covered_maps": maps_payload,
        "pairwise_shared_rows": pairwise,
        "confirmation_map_disjoint_from_all_others": bool(conf_disjoint),
        "confirmation_map_zero_internal_duplicates": bool(
            maps_payload["confirmation_full"]["zero_duplicates"]),
        "archive_group_overlaps": archive_overlaps,
        "archive_overlap_note": (
            "the R20-R22 archives share the master seed 20260913 and "
            "overlapping event-index spaces (R20: s * 1_000_000 + "
            "j * 10_000 + i; R21/R22: k * 1_000_000 + i with overlapping "
            "k ranges), so sealed-archive triples can coincide across "
            "those runs; this is a named property of the sealed "
            "Round 2 conventions and involves no confirmation-map row "
            "(the confirmation map is empirically disjoint from every "
            "other covered map)"
        ),
        "r23_note": (
            "R23 uses raw replica seeds 20300913..20300920 as the master "
            "seeds of its own map (no derive_substreams map from "
            "20260918); the sampled R23-style probe is included in the "
            "empirical union above"
        ),
    }
    return audit


def run_replica_correlation(sealed_replicas: dict) -> dict:
    """The pre-registered replica-correlation test: per R31 node, the
    pairwise Pearson correlation of the four sealed per-replica moment
    vectors' DEVIATIONS from the per-node mean vector (centering removes
    the shared species-scale structure, so the correlation measures
    replica-noise coherence; raw moment vectors correlate at ~0.999
    across every replica pair by construction and flag nothing
    selectively); |r| > 0.2 for n = 4 flags a seed issue.  Zero-variance
    components are dropped from the correlation and counted."""
    from math import sqrt as _sqrt

    def _pearson(a: list, b: list) -> float:
        n = len(a)
        ma = sum(a) / n
        mb = sum(b) / n
        da = [v - ma for v in a]
        db = [v - mb for v in b]
        denom = _sqrt(sum(x * x for x in da) * sum(x * x for x in db))
        if denom <= 0.0:
            return float("nan")
        return sum(x * y for x, y in zip(da, db)) / denom

    cells = {}
    flagged_pairs = 0
    for w in range(N_WINDOWS):
        for j in range(N_NODES):
            records = sealed_replicas[(w, j)]
            present = list(records[0].get("momentum_moments", {}).keys())
            vecs = []
            for record in records:
                vec = []
                for skey in present:
                    vec.append(float(record["multiplicities"][skey]["mean"]))
                    block = record["momentum_moments"].get(skey)
                    if block is not None:
                        vec.append(float(block["mean_p_gev"]))
                        vec.append(float(block["mean_pt_y_gev"]))
                vecs.append(vec)
            width = min(len(vec) for vec in vecs)
            matrix = np.array([vec[:width] for vec in vecs],
                              dtype=np.float64)
            center = matrix.mean(axis=0)
            centered = matrix - center
            keep = [c for c in range(width)
                    if float(np.std(centered[:, c])) > 0.0]
            dropped = width - len(keep)
            pairs = []
            for a in range(len(records)):
                for b in range(a + 1, len(records)):
                    if len(keep) >= 2:
                        r = _pearson(list(centered[a, keep]),
                                     list(centered[b, keep]))
                    else:
                        r = float("nan")
                    flagged = bool(math.isfinite(r)
                                   and r > CORRELATION_FLAG)
                    if flagged:
                        flagged_pairs += 1
                    pairs.append({
                        "replicas": [a, b],
                        "pearson_r": (float(r) if math.isfinite(r)
                                      else "NaN"),
                        "flagged": flagged,
                    })
            cells["w%d_j%d" % (w, j)] = {
                "components": int(width),
                "components_used": int(len(keep)),
                "zero_variance_components_dropped": int(dropped),
                "pairs": pairs,
                "flagged_pairs": int(sum(1 for pair in pairs
                                         if pair["flagged"])),
                "flag": bool(any(pair["flagged"] for pair in pairs)),
                "max_abs_r": float(max(
                    (abs(pair["pearson_r"]) for pair in pairs
                     if isinstance(pair["pearson_r"], float)
                     and math.isfinite(pair["pearson_r"])),
                    default=0.0,
                )),
            }
    return {
        "correlation_flag_cutoff": CORRELATION_FLAG,
        "centering_note": (
            "the correlation is computed on the replica deviations from "
            "the per-node mean moment vector (the shared species-scale "
            "structure is removed by centering; raw moment vectors "
            "correlate at ~0.999 across every replica pair by "
            "construction).  For independent replicas the centered "
            "noise vectors carry a mechanical negative baseline of "
            "about -1/3 at n = 4, so the seed-issue flag applies to "
            "the POSITIVE side only (r > +0.2, replica-noise "
            "coherence); the negative side is the independence "
            "signature and is reported, not flagged"
        ),
        "cells": cells,
        "flagged_pairs": int(flagged_pairs),
        "flag": bool(flagged_pairs > 0),
    }


def _r31_sealed_records() -> tuple[dict, dict]:
    """The sealed R31 per-replica and per-node records (read-only)."""
    run_dir = MONO_ROOT / R31_RUN_DIR_REL
    sealed_replicas: dict = {}
    sealed_nodes: dict = {}
    for w in range(N_WINDOWS):
        for j in range(N_NODES):
            records = []
            for replica in range(r31.T1_REPLICAS):
                path = run_dir / (
                    "window_%d_node_%d_replica_%d.json" % (w, j, replica)
                )
                records.append(_read_json(path))
            sealed_replicas[(w, j)] = records
            sealed_nodes[(w, j)] = _read_json(
                run_dir / ("window_%d_node_%d.json" % (w, j))
            )
    return sealed_replicas, sealed_nodes


def _r31_capture_sets(species_keys: tuple, sealed_nodes: dict) -> dict:
    """The pre-registered capture sets: all species at the affected
    nodes, and the affected species at every other node."""
    capture_sets: dict = {}
    for w in range(N_WINDOWS):
        for j in range(N_NODES):
            present = set(sealed_nodes[(w, j)].get("momentum_moments", {}))
            if (w, j) in AFFECTED_CELLS:
                capture = [s for s in species_keys if s in present]
            else:
                capture = [s for s in species_keys
                           if s in present and s in AFFECTED_SPECIES]
            capture_sets[(w, j)] = capture
    return capture_sets


def _r21r22_capture_sets(species_keys: tuple) -> dict:
    """The card-9 grid capture sets: the residual species of every
    named residual, at both nodes of its pair."""
    wanted: dict = {}
    for residual in R21R22_NAMED_RESIDUALS:
        run_label = residual["run"]
        for k in (int(residual["pair"][0]), int(residual["pair"][1])):
            wanted.setdefault((run_label, k), set()).add(residual["species"])
    capture_sets: dict = {}
    for (run_label, k), species in sorted(wanted.items()):
        capture = [s for s in species_keys if s in species]
        if not capture:
            capture = sorted(species)
        capture_sets[(run_label, k)] = capture
    return capture_sets


def _residual_evidence(entry_a: dict, entry_b: dict, observable: str,
                       residual_value):
    """The combined pair-level estimator evidence of one residual."""
    b32 = "b%d" % BLOCK_LENGTHS[0]
    b128 = "b%d" % BLOCK_LENGTHS[1]
    e1_a = entry_a.get("e1", {}).get(observable)
    e1_b = entry_b.get("e1", {}).get(observable)
    e4_a = (entry_a.get("e4", {}).get(b32) or {}).get("se", {}).get(observable)
    e4_b = (entry_b.get("e4", {}).get(b32) or {}).get("se", {}).get(observable)
    e4_a128 = (entry_a.get("e4", {}).get(b128) or {}).get("se", {}).get(
        observable)
    e4_b128 = (entry_b.get("e4", {}).get(b128) or {}).get("se", {}).get(
        observable)
    combined = {
        "e1_combined": None,
        "e4_b32_combined": None,
        "e4_b128_combined": None,
        "ratio_e4_b32_over_e1": None,
        "ratio_e4_b128_over_e1": None,
        "sigma_honest_e4_b32": None,
    }
    if e1_a is not None and e1_b is not None:
        e1c = math.sqrt(float(e1_a) ** 2 + float(e1_b) ** 2)
        combined["e1_combined"] = float(e1c)
        if e4_a is not None and e4_b is not None \
                and math.isfinite(float(e4_a)) and math.isfinite(float(e4_b)):
            e4c = math.sqrt(float(e4_a) ** 2 + float(e4_b) ** 2)
            combined["e4_b32_combined"] = float(e4c)
            if e1c > 0.0:
                combined["ratio_e4_b32_over_e1"] = float(e4c / e1c)
                if residual_value is not None and e4c > 0.0:
                    combined["sigma_honest_e4_b32"] = float(
                        float(residual_value) / e4c
                    )
        if e4_a128 is not None and e4_b128 is not None \
                and math.isfinite(float(e4_a128)) \
                and math.isfinite(float(e4_b128)) and e1c > 0.0:
            e4c128 = math.sqrt(float(e4_a128) ** 2 + float(e4_b128) ** 2)
            combined["e4_b128_combined"] = float(e4c128)
            combined["ratio_e4_b128_over_e1"] = float(e4c128 / e1c)
    return combined


def _classify_residual(ratio: float | None, correlation_flagged: bool,
                       sigma_honest: float | None) -> str:
    """The pre-registered classification rule (diagnostic values, never
    a metric change): the E4/E1 ratio is the direct SE-honesty
    evidence and takes precedence when it exceeds the artifact cutoff;
    the replica-noise coherence flag (positive centered correlation) is
    a modifier recorded alongside, not an override of a strong ratio
    classification."""
    if ratio is None:
        return "undefined"
    if float(ratio) > RATIO_ARTIFACT:
        return "estimator_artifact"
    if correlation_flagged:
        return "seed_issue_flag"
    if float(ratio) < RATIO_GENUINE:
        return "genuine_low_variance"
    return "mixed"


def _sealed_verification_rows() -> dict:
    """The sealed R31 verification rows keyed by
    (window, pair index, species, observable)."""
    report = _read_json(MONO_ROOT / R31_RUN_DIR_REL / "matching_verification.json")
    rows: dict = {}
    fails: dict = {}
    for wkey, window in report.get("windows", {}).items():
        for jkey, pair in window.get("pairs", {}).items():
            for row in pair.get("checks", []):
                status = row.get("status")
                if status not in ("pass", "fail"):
                    continue
                key = (int(wkey), int(jkey), str(row["species"]),
                       str(row["observable"]))
                rows[key] = row
                if status == "fail":
                    fails[key] = row
    return report, rows, fails


def _rereval_rows() -> dict:
    """The sealed card-2 re-evaluation rows keyed by
    (run, pair, species, observable)."""
    base = MONO_ROOT / REREVAL_RUN_DIR_REL
    out: dict = {}
    for run_label, name in (("R21", "rereval_r21.json"),
                            ("R22", "rereval_r22.json")):
        payload = _read_json(base / name)
        for row in payload.get("checks", []):
            key = (str(payload.get("run")), tuple(int(v) for v in row["pair"]),
                   str(row["species"]), str(row["observable"]))
            out[key] = row
    return out


def run_phase_a(run_dir: Path, workers: int) -> dict:
    """Phase A: the hermetic replay, cross-checks, estimator battery,
    seed-map audit, and classifications; write
    ``phase_a_diagnostics.json`` (no wall-clock fields)."""
    run_dir = Path(run_dir)
    pre = _assert_pre_registration(run_dir)
    catalog = HadronCatalog.from_json(NESTED_ROOT / CATALOG_REL)
    families = families_from_json(str(NESTED_ROOT / FAMILIES_REL))
    context = _build_context(catalog, families,
                             _precompute_kernel_maps(catalog))
    context["grid_cells"] = {}
    species_keys = context["species_keys"]
    # the card-9 species-key transliteration is shared by the two
    # committed drivers; assert it on every catalog pdg before the
    # replays (the capture keys must match the sealed record keys)
    for record in catalog.records():
        pdg = int(record.pdg_id)
        a = _species_key(_record_name(catalog, pdg))
        b = sgrid._species_key(sgrid._record_name(catalog, pdg))
        if a != b:
            _refuse(
                "the R31 and card-9 species-key conventions differ for "
                f"pdg {pdg} ({a!r} vs {b!r}); the capture sets would "
                "mis-align"
            )
    sealed_replicas, sealed_nodes = _r31_sealed_records()
    capture_sets = _r31_capture_sets(species_keys, sealed_nodes)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimator_cells: dict = {"R31": {}, "R21": {}, "R22": {}}
    r31_entries, r31_report = _replay_r31(
        context, int(workers), capture_sets, rng, sealed_replicas,
        sealed_nodes, species_keys,
    )
    estimator_cells["R31"] = r31_entries
    if not r31_report["all_exact"]:
        bad = [k for k, v in r31_report["cells_checked"].items()
               if not v["exact"]]
        _refuse(
            "the R31 replay cross-check failed (the replayed stored "
            "sums must match the sealed aggregates exactly before any "
            "estimator is computed); mismatched cells: "
            + ", ".join(bad)
        )
    grid_capture_sets = _r21r22_capture_sets(species_keys)
    r21_report = _replay_grid(
        "R21", R21_RUN_DIR_REL, catalog, context, int(workers),
        grid_capture_sets, rng, species_keys, estimator_cells["R21"],
    )
    r22_report = _replay_grid(
        "R22", R22_RUN_DIR_REL, catalog, context, int(workers),
        grid_capture_sets, rng, species_keys, estimator_cells["R22"],
    )
    for report in (r21_report, r22_report):
        if not report["all_exact"]:
            bad = [c["point"] for c in report["cells"] if not c["exact"]]
            _refuse(
                "the %s replay cross-check failed at points %s (the "
                "replayed stored sums must match the sealed aggregates "
                "exactly)" % (report["run"], bad)
            )
    _prepare_r21r22_audit_cells()
    seed_audit = run_seed_map_audit(int(workers))
    correlation = run_replica_correlation(sealed_replicas)
    # -- the 10 named residual classifications ---------------------------
    verification_report, _rows_all, fails = _sealed_verification_rows()
    named_rows = []
    for residual in NAMED_RESIDUALS:
        row = _classify_named_residual(residual, fails, estimator_cells,
                                       correlation)
        named_rows.append(row)
    node1 = _node1_tight_se(estimator_cells, correlation)
    rereval = _rereval_rows()
    r21r22_rows = []
    for residual in R21R22_NAMED_RESIDUALS:
        r21r22_rows.append(_audit_r21r22_residual(residual, rereval,
                                                  estimator_cells))
    payload = {
        "schema": "na49_mixing.staged_r3_se_audit.phase_a/1",
        "run": RUN_ID,
        "card": CARD,
        "pre_registration": {
            "path": PREREGISTRATION_NAME,
        },
        "environment": environment_fingerprint(),
        "replay": {
            "r31": r31_report,
            "r21": r21_report,
            "r22": r22_report,
            "zero_new_random_numbers": True,
        },
        "estimator_cells": estimator_cells,
        "seed_map_audit": seed_audit,
        "replica_correlation": correlation,
        "named_residuals": named_rows,
        "node1_tight_se": node1,
        "r21r22_named_residuals": r21r22_rows,
        "classification_thresholds": {
            "ratio_artifact": RATIO_ARTIFACT,
            "ratio_genuine": RATIO_GENUINE,
            "correlation_flag": CORRELATION_FLAG,
            "rule": CLASSIFICATION_RULE,
        },
        "no_tolerance_change": {
            "sigma_tolerance": SIGMA_TOLERANCE,
            "unchanged": True,
        },
        "no_fitting": True,
        "optimizer_invoked": False,
    }
    path = run_dir / PHASE_A_NAME
    _write_json(path, payload)
    print(
        "staged_r3_se_audit: phase A written to %s (replay exact: R31 "
        "%s, R21 %s, R22 %s; confirmation map disjoint from all other "
        "maps: %s; correlation flag: %s)"
        % (path, r31_report["all_exact"], r21_report["all_exact"],
           r22_report["all_exact"],
           seed_audit["confirmation_map_disjoint_from_all_others"],
           correlation["flag"])
    )
    return payload


def _classify_named_residual(residual: dict, fails: dict,
                             estimator_cells: dict,
                             correlation: dict) -> dict:
    """The Phase A classification row of one named residual: the
    E1-E4 evidence from its two cells plus the sealed verification
    sigma (cross-checked against the card's pre-fix sigma)."""
    w = int(residual["window"])
    j, jp1 = int(residual["pair"][0]), int(residual["pair"][1])
    skey = str(residual["species"])
    observable = str(residual["observable"])
    key = (w, j, skey, observable)
    sealed = fails.get(key)
    cell_a = estimator_cells["R31"].get("w%d_j%d" % (w, j), {})
    cell_b = estimator_cells["R31"].get("w%d_j%d" % (w, jp1), {})
    entry_a = cell_a.get("species", {}).get(skey)
    entry_b = estimator_cells["R31"].get("w%d_j%d" % (w, jp1), {}).get(
        "species", {}).get(skey)
    residual_value = None
    sigma_old = None
    if sealed is not None:
        sigma_old = float(sealed["sigma"])
        residual_value = float(sealed["delta_observed"]) - float(
            sealed["delta_analytic_full"])
        card_sigma = float(residual["pre_fix_sigma"])
        if abs(sigma_old - card_sigma) > 0.011:
            _refuse(
                "the sealed verification sigma for named residual %d "
                "disagrees with the card's pre-fix value"
                % int(residual["index"])
            )
    if entry_a is None or entry_b is None:
        evidence = _residual_evidence({}, {}, observable, None)
        evidence["absent"] = True
        classification = "undefined"
        sigma_honest = None
        ratio = None
    else:
        residual_value = (
            float(sealed["delta_observed"]) - float(sealed["delta_analytic_full"])
        ) if sealed is not None else None
        evidence = _residual_evidence(entry_a, entry_b, observable,
                                      residual_value)
        ratio = evidence.get("ratio_e4_b32_over_e1")
        sigma_honest = evidence.get("sigma_honest_e4_b32")
        cell_flags = [
            correlation["cells"]["w%d_j%d" % (w, jj)]["flag"]
            for jj in (j, jp1)
        ]
        correlation_flagged = bool(any(cell_flags))
        classification = _classify_residual(
            evidence.get("ratio_e4_b32_over_e1"), correlation_flagged,
            evidence.get("sigma_honest_e4_b32"),
        )
        evidence["correlation_flagged"] = correlation_flagged
        evidence["classification"] = classification
        evidence["classification_note"] = (
            "a diagnostic classification; the confirmation verdict at "
            "the 50-replica statistics is recorded in "
            "confirmation_verification.json"
        )
    return {
        "index": int(residual["index"]),
        "window": w,
        "pair": [j, jp1],
        "species": skey,
        "observable": observable,
        "pre_fix_sigma_card": float(residual["pre_fix_sigma"]),
        "sigma_old_sealed": sigma_old,
        "residual_sealed": (
            float(sealed["delta_observed"]) - float(sealed["delta_analytic_full"])
            if sealed is not None else None
        ),
        "se_combined_sealed": (
            float(sealed["se_combined"]) if sealed is not None else None
        ),
        "cells": {
            "node_j": {"cell": "w%d_j%d" % (w, j),
                       "e1": None if entry_a is None else entry_a["e1"].get(
                           observable),
                       "e4_b32": None if entry_a is None else
                       (entry_a.get("e4", {}).get(
                           "b%d" % BLOCK_LENGTHS[0]) or {}).get("se", {}).get(
                           observable)},
            "node_jp1": {"cell": "w%d_j%d" % (w, jp1),
                         "e1": None if entry_b is None else entry_b["e1"].get(
                             observable),
                         "e4_b32": None if entry_b is None else
                         (entry_b.get("e4", {}).get(
                             "b%d" % BLOCK_LENGTHS[0]) or {}).get(
                             "se", {}).get(observable)},
        },
        "evidence": evidence,
        "classification": classification,
        "sigma_honest_e4_b32": evidence.get("sigma_honest_e4_b32"),
    }


def card_sigma_of(residual: dict) -> float:
    return float(residual["pre_fix_sigma"])


def _node1_tight_se(estimator_cells: dict, correlation: dict) -> dict:
    """The node-1 tight-SE classification (window 2, node 1, pi_0,
    mean_p): the node-level E4/E1 ratio and the sibling ratios at the
    other nodes of window 2, classified with the pre-registered ratio
    cutoffs."""
    skey = "pi_0"
    observable = "mean_p"
    nodes = {}
    for j in range(N_NODES):
        entry = estimator_cells["R31"].get("w2_j%d" % j, {}).get(
            "species", {}).get(skey)
        if entry is None:
            nodes[str(j)] = {"absent": True}
            continue
        e1 = entry["e1"].get(observable)
        b0 = "b%d" % BLOCK_LENGTHS[0]
        e4 = (entry.get("e4", {}).get(b0) or {}).get("se", {}).get(
            observable)
        ratio = None
        if e1 is not None and e4 is not None and float(e1) > 0.0 \
                and math.isfinite(float(e4)):
            ratio = float(e4) / float(e1)
        nodes[str(j)] = {
            "e1": None if e1 is None else float(e1),
            "e4_b32": None if e4 is None else float(e4),
            "ratio_e4_b32_over_e1": ratio,
        }
    target = nodes.get("1", {})
    ratio_target = target.get("ratio_e4_b32_over_e1")
    sibling_ratios = [v.get("ratio_e4_b32_over_e1") for kk, v in
                      sorted(nodes.items()) if kk != "1"
                      and v.get("ratio_e4_b32_over_e1") is not None]
    if ratio_target is not None:
        if float(ratio_target) > RATIO_ARTIFACT:
            verdict = "estimator_artifact"
        elif float(ratio_target) < RATIO_GENUINE:
            verdict = "genuine_low_variance"
        else:
            verdict = "mixed"
    else:
        verdict = "undefined"
    return {
        "target": "window 2 node 1 (2.58 GeV) pi_0 mean_p",
        "nodes": nodes,
        "sibling_ratios": sibling_ratios,
        "ratio_target_node": ratio_target,
        "verdict": verdict,
        "correlation_flag_window2_node1": bool(
            correlation["cells"].get("w2_j1", {}).get("flag")
        ),
        "rule_note": (
            "ratio cutoffs E4/E1 > 3 = estimator artifact, < 1.5 = "
            "genuine low variance, in between = mixed (the "
            "pre-registered diagnostic rule)"
        ),
    }


def _audit_r21r22_residual(residual: dict, rereval: dict,
                           estimator_cells: dict) -> dict:
    """The estimator audit row of one R21/R22 named residual."""
    run_label = str(residual["run"])
    pair = (int(residual["pair"][0]), int(residual["pair"][1]))
    skey = str(residual["species"])
    observable = str(residual["observable"])
    rereval_key = (run_label, pair, skey, observable)
    sealed = rereval.get(rereval_key)
    if sealed is None:
        _refuse(
            "the named R21/R22 residual %s pair %s %s %s is missing "
            "from the sealed re-evaluation rows"
            % (run_label, pair, skey, observable)
        )
    residual_value = float(sealed["delta_observed"]) - float(
        sealed["delta_analytic_full"])
    if abs(abs(float(residual_value) / float(sealed["se_combined"]))
           - abs(float(sealed["sigma"]))) > 1e-6 * max(
                1.0, abs(float(sealed["sigma"]))):
        _refuse(
            "the sealed re-evaluation row for %s pair %s %s %s is "
            "internally inconsistent" % (run_label, pair, skey, observable)
        )
    entry_a = estimator_cells.get(run_label, {}).get(
        "k%02d" % pair[0], {}).get("species", {}).get(skey)
    entry_b = estimator_cells.get(run_label, {}).get(
        "k%02d" % pair[1], {}).get("species", {}).get(skey)
    evidence = _residual_evidence(entry_a or {}, entry_b or {}, observable,
                                  residual_value)
    classification = _classify_residual(
        evidence.get("ratio_e4_b32_over_e1"), False,
        evidence.get("sigma_honest_e4_b32"),
    )
    return {
        "run": run_label,
        "pair": [pair[0], pair[1]],
        "species": skey,
        "observable": observable,
        "pair_class": residual["pair_class"],
        "sigma_rereval_sealed": float(sealed["sigma"]),
        "residual_sealed": float(residual_value),
        "se_combined_sealed": float(sealed["se_combined"]),
        "e1_recomputed_combined": evidence.get("e1_combined"),
        "e1_cross_check_note": (
            "the between-replica SE convention of the sealed "
            "R21/R22 archives is recomputed from the sealed per-replica "
            "records inside the estimator entries (e1); a disagreement "
            "with se_combined_sealed beyond 1e-9 relative would refuse "
            "the run"
        ),
        "evidence": evidence,
        "classification": classification,
    }


# ---------------------------------------------------------------------------
# Phase B: the chunk mode (one chunk per gateway job), the aggregation,
# the re-evaluation, the seal, and the clean-environment validation.
# ---------------------------------------------------------------------------


def chunk_file_name(window: int, node: int, replica: int) -> str:
    return "chunk_w%d_n%d_r%d.json" % (window, node, replica)


def _acc_from_block(block: dict, species_keys: tuple, pair_keys: tuple = ()):
    """Rebuild a committed ``_Accumulator`` from a stored chunk block
    (exact integer counts, float sums; JSON round-trips floats
    exactly)."""
    acc = r31._Accumulator(species_keys)
    acc.n_events = int(block["n_events"])
    acc.n_catalog = int(block["n_catalog"])
    acc.n_continuum = int(block["n_continuum"])
    for key in species_keys:
        acc.count[key] = int(block["count"][key])
        acc.count_sq[key] = int(block["count_sq"][key])
        acc.p_sum[key] = float(block["p_sum"][key])
        acc.e_sum[key] = float(block["e_sum"][key])
        acc.ept_sum[key] = float(block["ept_sum"][key])
        acc.n_species[key] = int(block["n_species"][key])
        acc.n_resonant[key] = int(block["n_resonant"][key])
        for pdg_text, cnt in block["parent_counts"][key].items():
            acc.parent_counts[key][int(pdg_text)] = int(cnt)
    for key in acc.pair_keys:
        acc.pair[key] = int(block["pair"][key])
        acc.pair_sq[key] = int(block["pair_sq"][key])
    return acc


def _acc_to_block(acc) -> dict:
    """The stored accumulator block of one chunk (all sums, the merge
    contract of the aggregation)."""
    return {
        "n_events": int(acc.n_events),
        "n_catalog": int(acc.n_catalog),
        "n_continuum": int(acc.n_continuum),
        "count": {key: int(v) for key, v in acc.count.items()},
        "count_sq": {key: int(v) for key, v in acc.count_sq.items()},
        "p_sum": {key: float(v) for key, v in acc.p_sum.items()},
        "e_sum": {key: float(v) for key, v in acc.e_sum.items()},
        "ept_sum": {key: float(v) for key, v in acc.ept_sum.items()},
        "n_species": {key: int(v) for key, v in acc.n_species.items()},
        "n_resonant": {key: int(v) for key, v in acc.n_resonant.items()},
        "parent_counts": {
            key: {str(pdg): int(cnt) for pdg, cnt in sorted(bucket.items())}
            for key, bucket in acc.parent_counts.items()
        },
        "pair": {key: int(v) for key, v in acc.pair.items()},
        "pair_sq": {key: int(v) for key, v in acc.pair_sq.items()},
    }


def _diagnostics_payload_se(acc, window: int, node: int, source_qn: tuple,
                            record_pdgs: list, mode: str,
                            replicas: int | None = None,
                            decays_per_replica: int | None = None) -> dict:
    """The card-9 output schema for one Phase B sample with this run's
    identity (the R31 payload builder's field set, the confirmation run
    identity, master seed, and seed rule)."""
    n = acc.n_events
    if n <= 0:
        raise RuntimeError(
            "staged_r3_se_audit: an empty event sample cannot produce "
            "the diagnostics payload"
        )
    multiplicities = {}
    for key in acc.count:
        mean, se = r31._mean_se(acc.count[key], acc.count_sq[key], n)
        multiplicities[key] = {"mean": mean, "se": se}
    momentum_moments: dict = {}
    feed_down: dict = {}
    for key in acc.count:
        n_species = acc.n_species[key]
        if n_species <= 0:
            continue
        e_sum = acc.e_sum[key]
        momentum_moments[key] = {
            "mean_p_gev": acc.p_sum[key] / n_species,
            "mean_pt_y_gev": (
                acc.ept_sum[key] / e_sum if e_sum > 0.0 else 0.0
            ),
            "mean_e_gev": e_sum / n_species,
            "n_particles": int(n_species),
        }
        feed_down[key] = {
            "fraction_resonant": acc.n_resonant[key] / n_species,
            "by_parent": {
                str(pdg): cnt / n_species
                for pdg, cnt in sorted(acc.parent_counts[key].items())
            },
        }
    pairs: dict = {}
    for name in acc.pair_keys:
        mean, se = r31._mean_se(acc.pair[name], acc.pair_sq[name], n)
        pairs[name] = {"mean": mean, "se": se}
    payload = {
        "schema": "na49_mixing.staged_r3_se_audit.diagnostics/1",
        "run": RUN_ID,
        "card": CARD,
        "mode": mode,
        "window": int(window),
        "node": int(node),
        "point": int(node),
        "source_mass_gev": float(NODE_MASSES[int(node)]),
        "source_qn": [int(q) for q in source_qn],
        "open_records": [int(p) for p in record_pdgs],
        "n_decays": int(n),
        "master_seed": int(CONFIRMATION_MASTER_SEED),
        "seed_rule": SEED_RULE,
        "routing": {
            "n_catalog": int(acc.n_catalog),
            "n_continuum": int(acc.n_continuum),
            "fraction_continuum": float(acc.n_continuum) / float(n),
            "expected_w_at_node": float(EXPECTED_W[int(node)]),
            "boundary_gev": BOUNDARY,
            "ramp_width_gev": RAMP_WIDTH,
            "rule": (
                "the card-6 routing evaluated at the node mass: catalog "
                "for M < B, one uniform on the spawned decay_channel "
                "child with continuum iff u < w(M) for B <= M < B + W, "
                "continuum for M >= B + W"
            ),
        },
        "multiplicities": multiplicities,
        "momentum_moments": momentum_moments,
        "feed_down": feed_down,
        "charge_balance": {
            "total_Q_exact": int(source_qn[1]),
            "total_B_exact": int(source_qn[0]),
            "events_checked": int(n),
        },
        "pairs": pairs,
    }
    if replicas is not None:
        payload["replicas"] = int(replicas)
        payload["decays_per_replica"] = int(decays_per_replica)
    return payload


def _chunk_path(run_dir: Path, window: int, node: int, replica: int,
                smoke: bool) -> Path:
    sub = SMOKE_DIR_NAME if smoke else PHASE_B_NAME
    return Path(run_dir) / sub / chunk_file_name(window, node, replica)


def run_chunk(run_dir: Path, window: int, node: int, replica: int,
              smoke: bool, decays: int | None) -> None:
    """Generate one Phase B chunk (production: 20,000 decays under the
    pre-registered confirmation map; smoke: reduced counts into
    phase_b_smoke/).  No local process pool: one chunk per process, the
    gateway discipline."""
    run_dir = Path(run_dir)
    if window is None or node is None or replica is None:
        _refuse("--chunk requires --window W --node J --replica R")
    if not 0 <= int(window) < N_WINDOWS:
        _refuse("--window %d outside 0..%d" % (window, N_WINDOWS - 1))
    if not 0 <= int(node) < N_NODES:
        _refuse("--node %d outside 0..%d" % (node, N_NODES - 1))
    if smoke:
        chunk_decays = int(decays) if decays is not None else SMOKE_DECAYS
        out_dir = Path(run_dir) / SMOKE_DIR_NAME
        mode = "smoke"
    else:
        pre = _assert_pre_registration(run_dir)
        if decays is not None and int(decays) != CHUNK_DECAYS:
            _refuse(
                "--decays %s does not match the pre-registered chunk "
                "size %d" % (decays, CHUNK_DECAYS)
            )
        chunk_decays = CHUNK_DECAYS
        out_dir = Path(run_dir) / PHASE_B_NAME
        mode = "production"
    out_path = out_dir / chunk_file_name(window, node, replica)
    if out_path.is_file():
        try:
            stored = _read_json(out_path)
        except ValueError:
            stored = None
        if isinstance(stored, dict) and int(
                stored.get("window", -1)) == int(window) \
                and int(stored.get("node", -1)) == int(node) \
                and int(stored.get("replica", -1)) == int(replica) \
                and int(stored.get("n_decays", -1)) == chunk_decays:
            print(
                "staged_r3_se_audit: chunk %s already present and "
                "verified; a completed chunk is never regenerated "
                "(the cluster output is the canonical chunk record)"
                % out_path.name
            )
            return
    catalog = HadronCatalog.from_json(NESTED_ROOT / CATALOG_REL)
    families = families_from_json(str(NESTED_ROOT / FAMILIES_REL))
    context = _build_context(catalog, families,
                             _precompute_kernel_maps(catalog))
    qn = tuple(int(v) for v in WINDOWS[int(window)]["qn"])
    mass = float(NODE_MASSES[int(node)])
    open_records = context["open_sets"][qn][float(mass)]
    record_pdgs = [int(rec.pdg_id) for rec in open_records]
    base = (int(window) * N_NODES + int(node)) * 100_000_000 \
        + int(replica) * CHUNK_DECAYS
    acc = r31._Accumulator(context["species_keys"])
    started = time.perf_counter()
    for offset in range(chunk_decays):
        event_index = base + offset
        route, particles = _decay_particles_se(
            catalog, qn, float(mass), CONFIRMATION_MASTER_SEED,
            int(replica), event_index, context,
        )
        r31._accumulate_event(acc, catalog, context["species_index"], route,
                              particles, qn)
    elapsed = time.perf_counter() - started
    payload = _diagnostics_payload_se(acc, window, node, qn, record_pdgs,
                                      mode)
    chunk_record = {
        "schema": "na49_mixing.staged_r3_se_audit.chunk/1",
        "run": RUN_ID,
        "card": CARD,
        "mode": mode,
        "window": int(window),
        "node": int(node),
        "replica": int(replica),
        "n_decays": int(chunk_decays),
        "event_index_range": [int(base), int(base + chunk_decays)],
        "master_seed": int(CONFIRMATION_MASTER_SEED),
        "seed_rule": SEED_RULE,
        "smoke": bool(smoke),
        "accumulator": _acc_to_block(acc),
        "diagnostics": payload,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".json.tmp")
    _write_json(tmp_path, chunk_record)
    os.replace(tmp_path, out_path)
    print(
        "staged_r3_se_audit: chunk %s written (%d decays in %.1fs; "
        "routing catalog %d / continuum %d, fraction %.6f, expected w "
        "%.12f)"
        % (out_path.name, chunk_decays, elapsed, acc.n_catalog,
           acc.n_continuum, acc.n_continuum / max(acc.n_events, 1),
           float(EXPECTED_W[int(node)]))
    )


def _production_chunk_list() -> list:
    return [
        (w, j, r)
        for w in range(N_WINDOWS)
        for j in range(N_NODES)
        for r in range(CONFIRMATION_REPLICAS)
    ]


def _production_chunk_list() -> list:
    return [
        (w, j, r)
        for w in range(N_WINDOWS)
        for j in range(N_NODES)
        for r in range(CONFIRMATION_REPLICAS)
    ]


SMOKE_CHUNK_LIST = [(2, 3, r) for r in range(SMOKE_REPLICAS)] + [
    (2, 4, r) for r in range(SMOKE_REPLICAS)
]


def _agg_state_path(run_dir: Path, smoke: bool) -> Path:
    """The aggregation-state file path (the smoke dir for smoke runs)."""
    run_dir = Path(run_dir)
    if smoke:
        return run_dir / SMOKE_DIR_NAME / AGG_STATE_NAME
    return run_dir / AGG_STATE_NAME


def run_aggregate(run_dir: Path, smoke: bool) -> None:
    """Verify and merge the fetched chunk outputs into the per-replica
    and per-node diagnostics (with the routing fractions), writing the
    checkpoint state.  Nodes merge only when all of their chunks are
    present; a dropped session resumes aggregation from the chunk
    outputs (a verified chunk is never regenerated)."""
    run_dir = Path(run_dir)
    sub = SMOKE_DIR_NAME if smoke else PHASE_B_NAME
    base_dir = Path(run_dir) / sub
    chunks = SMOKE_CHUNK_LIST if smoke else _production_chunk_list()
    if not smoke:
        _assert_pre_registration(run_dir)
    state = {
        "schema": "na49_mixing.staged_r3_se_audit.aggregation_state/1",
        "run": RUN_ID,
        "card": CARD,
        "smoke": bool(smoke),
        "chunk_dir": sub + "/",
        "expected_chunks": len(chunks),
        "completed_chunks": {},
        "nodes_merged": [],
        "merge_order": ("replica order r' = 0..49 (exact integer "
                        "counts, float sums in fixed order)"),
    }
    missing = []
    replicas_per_node = len(chunks) // (N_WINDOWS * N_NODES)
    for (w, j, r) in chunks:
        path = base_dir / chunk_file_name(w, j, r)
        if not path.is_file():
            missing.append(path.name)
            continue
        record = _read_json(path)
        expected_mode = "smoke" if smoke else "production"
        if int(record.get("window", -1)) != int(w) \
                or int(record.get("node", -1)) != int(j) \
                or int(record.get("replica", -1)) != int(r) \
                or str(record.get("mode")) != expected_mode:
            _refuse("chunk %s carries the wrong identity or mode"
                    % path.name)
        state["completed_chunks"][path.name] = {
            "window": int(w), "node": int(j), "replica": int(r),
            "n_decays": int(record["n_decays"]),
        }
    chunk_decays_seen = {
        int(v["n_decays"]) for v in state["completed_chunks"].values()
    }
    if len(chunk_decays_seen) != 1:
        _refuse("the chunk set carries inconsistent chunk sizes: %s"
                % sorted(chunk_decays_seen))
    chunk_decays = chunk_decays_seen.pop()
    state["n_completed"] = len(state["completed_chunks"])
    if missing:
        state["missing_chunks"] = len(missing)
        _write_json(_agg_state_path(run_dir, smoke), state)
        print(
            "staged_r3_se_audit: aggregation checkpoint updated (%d/%d "
            "chunks present; %d pending); rerun --aggregate after "
            "fetching more chunk outputs"
            % (state["n_completed"], len(chunks), len(missing))
        )
        return
    n_cells = len({(w, j) for (w, j, _r) in chunks})
    replicas_per_node = len(chunks) // n_cells
    cells = sorted({(w, j) for (w, j, _r) in chunks})
    catalog = HadronCatalog.from_json(NESTED_ROOT / CATALOG_REL)
    context = _build_context(
        catalog, families_from_json(str(NESTED_ROOT / FAMILIES_REL)),
        _precompute_kernel_maps(catalog),
    )
    species_keys = context["species_keys"]
    open_sets = context["open_sets"]
    merged_nodes = 0
    for (w, j) in cells:
            qn = tuple(int(v) for v in WINDOWS[w]["qn"])
            open_records = open_sets[qn][float(NODE_MASSES[j])]
            record_pdgs = [int(rec.pdg_id) for rec in open_records]
            node_acc = r31._Accumulator(species_keys)
            replica_payloads = []
            for r in range(replicas_per_node):
                record = _read_json(base_dir / chunk_file_name(w, j, r))
                acc = _acc_from_block(record["accumulator"],
                                      tuple(species_keys))
                replica_payloads.append(record["diagnostics"])
                node_acc.merge(acc)
            node_payload = _diagnostics_payload_se(
                node_acc, w, j, qn, record_pdgs,
                "smoke" if smoke else "production",
                replicas=replicas_per_node,
                decays_per_replica=chunk_decays,
            )
            _write_json(base_dir / ("window_%d_node_%d.json" % (w, j)),
                        node_payload)
            for r, replica_payload in enumerate(replica_payloads):
                _write_json(
                    base_dir / ("window_%d_node_%d_replica_%d.json"
                                % (w, j, r)),
                    replica_payload,
                )
            merged_nodes += 1
            state["nodes_merged"].append("w%d_j%d" % (w, j))
    state["all_nodes_merged"] = True
    _write_json(_agg_state_path(run_dir, smoke), state)
    print(
        "staged_r3_se_audit: aggregation complete (%d chunks, %d nodes; "
        "run dir %s)" % (len(chunks), merged_nodes, run_dir)
    )


def _moment_se_n(replica_records: list, skey: str, field: str):
    """The between-replica standard error of one moment over the stored
    replica values (std ddof=1 over sqrt(n); ``None`` when any replica
    lacks the species block)."""
    values = []
    for record in replica_records:
        block = record.get("momentum_moments", {}).get(skey)
        if block is None:
            return None
        values.append(float(block[field]))
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(max(variance, 0.0) / n)


def run_reeval(run_dir: Path, smoke: bool) -> None:
    """Re-evaluate all R31 checks with the fixed engine at the
    confirmation statistics: per-check sigma at the old SE versus the
    new SE, the 14 original failures named individually, and the 10
    named residuals classified at the confirmation statistics; write
    ``confirmation_verification.json``, then stop."""
    run_dir = Path(run_dir)
    sub = SMOKE_DIR_NAME if smoke else PHASE_B_NAME
    base_dir = Path(run_dir) / sub
    replicas = SMOKE_REPLICAS if smoke else CONFIRMATION_REPLICAS
    rows_all: dict = {}
    fails: dict = {}
    pre = None
    if not smoke:
        pre = _assert_pre_registration(run_dir)
        _sealed_report, rows_all, fails = _sealed_verification_rows()
    catalog = HadronCatalog.from_json(NESTED_ROOT / CATALOG_REL)
    families = families_from_json(str(NESTED_ROOT / FAMILIES_REL))
    species_keys = _stable_species_keys(catalog)
    nodes: dict = {}
    replica_map: dict = {}
    required_cells = sorted({(w, j) for (w, j, _r) in (
        SMOKE_CHUNK_LIST if smoke else _production_chunk_list())})
    for w in range(N_WINDOWS):
        for j in range(N_NODES):
            if (w, j) not in required_cells:
                continue
            node_path = base_dir / ("window_%d_node_%d.json" % (w, j))
            if not node_path.is_file():
                _refuse("node record missing: %s (run --aggregate first)"
                        % node_path)
            node = _read_json(node_path)
            decays_per_replica = int(node.get("decays_per_replica",
                                              CHUNK_DECAYS))
            if int(node.get("n_decays", -1)) != replicas * decays_per_replica:
                _refuse("%s carries the wrong event count" % node_path)
            nodes[(w, j)] = node
            records = []
            for r in range(replicas):
                records.append(_read_json(base_dir / (
                    "window_%d_node_%d_replica_%d.json" % (w, j, r))))
            replica_map[(w, j)] = records
    record_set_cache: dict = {}
    windows_payload: dict = {}
    drift_rows: list = []
    total_applied = 0
    total_failed = 0
    max_abs_sigma = 0.0
    matched_old = 0
    new_status_by_key: dict = {}
    for spec_window in r31.WINDOWS:
        w = int(spec_window["window"])
        mu_catalog, mu_continuum, weights, record_set = \
            r31._routed_payloads(w, catalog, families, record_set_cache)
        pairs_payload = {}
        for j in range(N_NODES - 1):
            if (w, j) not in nodes or (w, j + 1) not in nodes:
                continue
            global_pair = 4 * w + j
            masses = (float(NODE_MASSES[j]), float(NODE_MASSES[j + 1]))
            w_j, w_jp1 = weights[j], weights[j + 1]
            catalog_only = (w_j <= 0.0) and (w_jp1 <= 0.0)
            catalog_delta = {}
            for skey in species_keys:
                full = full_analytic_difference(
                    record_set, masses[0], masses[1], skey
                )
                catalog_delta[skey] = full
            pair_rows = []
            for skey in species_keys:
                node_j = nodes[(w, j)]
                node_jp1 = nodes[(w, j + 1)]
                mult_j = node_j["multiplicities"][skey]
                mult_jp1 = node_jp1["multiplicities"][skey]
                observed_j = {"multiplicity": float(mult_j["mean"])}
                observed_jp1 = {"multiplicity": float(mult_jp1["mean"])}
                se = {"multiplicity": math.sqrt(
                    float(mult_j["se"]) ** 2 + float(mult_jp1["se"]) ** 2)}
                present_j = _moment_se_n(replica_map[(w, j)], skey,
                                         "mean_p_gev")
                present_jp1 = _moment_se_n(replica_map[(w, j + 1)], skey,
                                           "mean_p_gev")
                if present_j is not None and present_jp1 is not None:
                    block_j = node_j["momentum_moments"][skey]
                    block_jp1 = node_jp1["momentum_moments"][skey]
                    observed_j["mean_p"] = float(block_j["mean_p_gev"])
                    observed_jp1["mean_p"] = float(block_jp1["mean_p_gev"])
                    se["mean_p"] = math.sqrt(
                        float(present_j) ** 2 + float(present_jp1) ** 2)
                    pt_j = _moment_se_n(replica_map[(w, j)], skey,
                                        "mean_pt_y_gev")
                    pt_jp1 = _moment_se_n(replica_map[(w, j + 1)], skey,
                                          "mean_pt_y_gev")
                    if (pt_j is not None and pt_jp1 is not None
                            and int(block_j["n_particles"]) > 0
                            and int(block_jp1["n_particles"]) > 0):
                        observed_j["mean_pt_y"] = float(
                            block_j["mean_pt_y_gev"])
                        observed_jp1["mean_pt_y"] = float(
                            block_jp1["mean_pt_y_gev"])
                        se["mean_pt_y"] = math.sqrt(
                            float(pt_j) ** 2 + float(pt_jp1) ** 2)
                for observable in OBSERVABLES:
                    if observable not in observed_j:
                        pair_rows.append({
                            "pair": [int(j), int(j + 1)],
                            "global_pair": int(global_pair),
                            "species": skey,
                            "observable": observable,
                            "status": "absent_species",
                            "pass": None,
                        })
                        continue
                    observed = (observed_jp1[observable]
                                - observed_j[observable])
                    if catalog_only:
                        delta_value = catalog_delta[skey][
                            "delta_analytic_full"].get(observable)
                        if delta_value is None:
                            pair_rows.append({
                                "pair": [int(j), int(j + 1)],
                                "global_pair": int(global_pair),
                                "species": skey,
                                "observable": observable,
                                "status": "absent_species_analytic",
                                "pass": None,
                            })
                            continue
                        delta_full = float(delta_value)
                    else:
                        ana_j = r31._routed_observable(
                            mu_catalog, mu_continuum, weights, j, skey,
                            observable,
                        )
                        ana_jp1 = r31._routed_observable(
                            mu_catalog, mu_continuum, weights, j + 1, skey,
                            observable,
                        )
                        if ana_j is None or ana_jp1 is None:
                            pair_rows.append({
                                "pair": [int(j), int(j + 1)],
                                "global_pair": int(global_pair),
                                "species": skey,
                                "observable": observable,
                                "status": "absent_species_analytic",
                                "pass": None,
                            })
                            continue
                        delta_full = ana_jp1 - ana_j
                    if observable == "multiplicity":
                        se_k = float(mult_j["se"])
                        se_kp1 = float(mult_jp1["se"])
                    else:
                        se_k = se[observable] / math.sqrt(2.0)
                        se_kp1 = se[observable] / math.sqrt(2.0)
                    verdict = evaluate_pair(
                        float(observed), float(delta_full), se_k, se_kp1
                    )
                    row = {
                        "pair": [int(j), int(j + 1)],
                        "global_pair": int(global_pair),
                        "masses_gev": list(masses),
                        "species": skey,
                        "observable": observable,
                        "observed": [observed_j[observable],
                                     observed_jp1[observable]],
                        "delta_observed": float(observed),
                        "delta_analytic_full": float(delta_full),
                        "se_combined": float(verdict["se_combined"]),
                        "sigma": float(verdict["sigma"]),
                        "tolerance_sigma": float(METRIC_N_SIGMA),
                        "status": ("pass" if verdict["pass"] else "fail"),
                        "pass": bool(verdict["pass"]),
                        "se_floor_applied": bool(verdict["se_floor_applied"]),
                    }
                    old_row = (None if smoke else rows_all.get(
                        (w, j, skey, observable)))
                    if old_row is not None:
                        matched_old += 1
                        old_delta = float(old_row["delta_analytic_full"])
                        row["delta_analytic_full_sealed"] = old_delta
                        row["delta_equal_sealed"] = bool(
                            float(delta_full) == old_delta)
                        if not row["delta_equal_sealed"]:
                            drift_rows.append({
                                "window": w, "pair": [int(j), int(j + 1)],
                                "species": skey,
                                "observable": observable,
                                "delta_sealed": old_delta,
                                "delta_fixed_engine": float(delta_full),
                            })
                        row["sigma_old_sealed"] = float(old_row["sigma"])
                        row["se_combined_old"] = float(
                            old_row["se_combined"])
                        old_se = float(old_row["se_combined"])
                        postfix = evaluate_pair(
                            float(old_row["delta_observed"]),
                            float(delta_full),
                            old_se / math.sqrt(2.0), old_se / math.sqrt(2.0),
                        )
                        row["sigma_postfix_old_se"] = float(
                            postfix["sigma"])
                        row["se_floor_applied_old_se"] = bool(
                            postfix["se_floor_applied"])
                    pair_rows.append(row)
                    total_applied += 1
                    value = abs(float(verdict["sigma"]))
                    if math.isfinite(value) and value > max_abs_sigma:
                        max_abs_sigma = float(value)
                    if not verdict["pass"]:
                        total_failed += 1
            pairs_payload = windows_payload.setdefault(str(w), {})
            pairs_all = pairs_payload.setdefault("pairs", {})
            pairs_all[str(j)] = {
                "pair": [int(j), int(j + 1)],
                "global_pair": int(global_pair),
                "masses_gev": list(masses),
                "w_at_nodes": [float(w_j), float(w_jp1)],
                "boundary_straddling": bool(j == 2),
                "catalog_only": bool(catalog_only),
                "checks": pair_rows,
            }
        if str(w) in windows_payload:
            windows_payload[str(w)]["qn"] = [
                int(v) for v in spec_window["qn"]]
    routing_block: dict = {}
    for w in range(N_WINDOWS):
        for j in range(N_NODES):
            if (w, j) not in nodes:
                continue
            routing = nodes[(w, j)]["routing"]
            frac = float(routing["fraction_continuum"])
            expected = float(routing["expected_w_at_node"])
            n = int(nodes[(w, j)]["n_decays"])
            if 0.0 < expected < 1.0:
                z = (frac - expected) / math.sqrt(
                    expected * (1.0 - expected) / float(n))
            elif abs(frac - expected) < 1e-12:
                z = 0.0
            else:
                z = float("inf")
            routing_block["w%d_j%d" % (w, j)] = {
                "fraction_continuum": frac,
                "expected_w": expected,
                "binomial_sigma": float(z) if math.isfinite(z)
                else "Infinity",
                "n_catalog": int(routing["n_catalog"]),
                "n_continuum": int(routing["n_continuum"]),
            }
    if not smoke:
        applied_new_keys = set()
        pass_new_keys = set()
        for w in range(N_WINDOWS):
            for pair in windows_payload[str(w)]["pairs"].values():
                for row in pair["checks"]:
                    key = (w, row["pair"][0], row["species"],
                           row["observable"])
                    if row.get("status") in ("pass", "fail"):
                        applied_new_keys.add(key)
                        if row["status"] == "pass":
                            pass_new_keys.add(key)
        old_fail_keys = sorted(fails)
        old_fail_new = []
        for key in old_fail_keys:
            w, jj, skey, observable = key
            new_row = None
            for pair in windows_payload[str(w)]["pairs"].values():
                for row in pair["checks"]:
                    if (row["pair"][0] == jj and row["species"] == skey
                            and row["observable"] == observable
                            and row.get("status") in ("pass", "fail")):
                        new_row = row
            if new_row is None:
                _refuse("the original failure (%d, %d, %s, %s) has no "
                        "new row" % (w, jj, skey, observable))
            old_fail_new.append({
                "window": w,
                "pair": [int(jj), int(jj + 1)],
                "global_pair": 4 * w + int(jj),
                "species": skey,
                "observable": observable,
                "sigma_old_sealed": float(fails[key]["sigma"]),
                "se_combined_old": float(fails[key]["se_combined"]),
                "sigma_postfix_old_se": (
                    None if new_row.get("sigma_postfix_old_se") is None
                    else float(new_row["sigma_postfix_old_se"])),
                "sigma_new": float(new_row["sigma"]),
                "se_combined_new": float(new_row["se_combined"]),
                "status_new": new_row["status"],
                "delta_equal_sealed": new_row.get("delta_equal_sealed"),
            })
        named_rows = []
        for block in NAMED_RESIDUALS:
            w = int(block["window"])
            jj = int(block["pair"][0])
            skey = str(block["species"])
            observable = str(block["observable"])
            key = (w, jj, skey, observable)
            new_row = None
            for pair in windows_payload[str(w)]["pairs"].values():
                for row in pair["checks"]:
                    if (row["pair"][0] == jj and row["species"] == skey
                            and row["observable"] == observable
                            and row.get("status") in ("pass", "fail")):
                        new_row = row
            if new_row is None:
                _refuse("named residual %d has no new row"
                        % int(block["index"]))
            if abs(float(new_row["sigma"])) <= METRIC_N_SIGMA:
                verdict = "genuine_finite_mc_scatter"
                note = ("resolved at the confirmation statistics "
                        "(|sigma_new| <= 3.0 at the honest 50-replica SE)")
            else:
                verdict = "persistent_mismatch"
                note = ("re-opens the engine-fix card (or the originating "
                        "family card), never the tolerance")
            named_rows.append({
                "index": int(block["index"]),
                "window": w,
                "pair": [int(jj), int(jj + 1)],
                "species": skey,
                "observable": observable,
                "pre_fix_sigma_card": float(block["pre_fix_sigma"]),
                "sigma_old_sealed": float(fails[key]["sigma"]),
                "sigma_postfix_old_se": new_row.get(
                    "sigma_postfix_old_se"),
                "sigma_new": float(new_row["sigma"]),
                "se_combined_new": float(new_row["se_combined"]),
                "classification_phase_b": verdict,
                "classification_note": note,
            })
        fail_pass_count = sum(1 for key in fails if key in pass_new_keys)
    else:
        old_fail_new, named_rows, fail_pass_count = [], [], 0
    summary = {
        "checks_applied_old": (422 if not smoke else None),
        "checks_failed_old": (14 if not smoke else None),
        "checks_applied_new": int(total_applied),
        "checks_failed_new": int(total_failed),
        "max_abs_sigma_new": (max_abs_sigma if max_abs_sigma > 0.0 else 0.0),
        "matched_old_rows": int(matched_old),
        "old_failures_pass_new": int(fail_pass_count),
        "delta_drift_rows": drift_rows,
        "tolerance_note": ("the 3.0-sigma band is never relaxed; a "
                           "persistent mismatch re-opens the engine-fix "
                           "card or the originating family card"),
    }
    payload = {
        "schema": "na49_mixing.staged_r3_se_audit.reeval/1",
        "run": RUN_ID,
        "card": CARD,
        "metric": {
            "id": METRIC_ID,
            "tolerance_sigma": float(METRIC_N_SIGMA),
            "tolerance_unchanged": True,
            "statement": (
                "delta_analytic_full at 3.0 sigma with "
                "SE_combined = sqrt(SE_j^2 + SE_{j+1}^2) at the "
                "confirmation statistics; per-check sigma at the old SE "
                "versus the new SE"
            ),
        },
        "statistics": {
            "replicas": int(replicas),
            "decays_per_replica": (None if smoke else int(CHUNK_DECAYS)),
            "master_seed": (None if smoke
                            else int(CONFIRMATION_MASTER_SEED)),
        },
        "driver_sha256": {
            "at_preregistration": (None if pre is None else
                                   pre.get("input_hashes", {}).get(
                                       "driver_sha256")),
            "at_reeval": _sha256_file(Path(__file__).resolve()),
            "drift": ([] if pre is None
                      else pre.get("_driver_sha256_drift", [])),
            "note": (
                "the pre-registration binds the statistics, seeds, "
                "estimator definitions, and cutoffs (re-asserted "
                "unchanged); a driver-code fix between the "
                "pre-registration and this re-evaluation is recorded "
                "here, the pre-registration file itself is immutable"
            ),
        },
        "summary": summary,
        "original_14_failures": old_fail_new,
        "named_residuals_phase_b": named_rows,
        "routing_fractions": routing_block,
        "windows": windows_payload,
        "no_fitting": True,
        "optimizer_invoked": False,
    }
    if smoke:
        out_path = run_dir / SMOKE_DIR_NAME / "confirmation_verification_smoke.json"
    else:
        out_path = run_dir / REEVAL_NAME
    _write_json(out_path, payload)
    print(
        "staged_r3_se_audit: re-evaluation written to %s (%d checks "
        "applied, %d failed, %d sealed rows matched)"
        % (out_path, total_applied, total_failed, matched_old)
    )


def _windows_key_of(w: int, j: int, skey: str, observable: str) -> tuple:
    return (w, j, skey, observable)


# ---------------------------------------------------------------------------
# Seal and clean-environment validation (the R31 conventions).
# ---------------------------------------------------------------------------


def write_sha256sums(run_dir: Path) -> Path:
    """Write ``SHA256SUMS.txt`` over every payload file (sha256sum -c);
    entries carry monorepo-root-relative paths (the R21 convention)."""
    run_dir = Path(run_dir).resolve()
    skip = {SHA256SUMS_NAME, SEAL_NAME, CLEAN_ENV_NAME}
    lines = []
    for path in _run_dir_files(run_dir):
        if path.name in skip:
            continue
        rel = path.relative_to(MONO_ROOT).as_posix()
        lines.append("%s  %s\n" % (_sha256_file(path), rel))
    sums_path = run_dir / SHA256SUMS_NAME
    _write_bytes(sums_path, "".join(lines).encode("utf-8"))
    return sums_path


def write_seal_manifest(run_dir: Path) -> dict:
    """Write ``seal_manifest.json`` over every other file plus the sums
    (the seal manifest cannot contain its own digest)."""
    run_dir = Path(run_dir)
    entries = []
    total_bytes = 0
    for path in _run_dir_files(run_dir):
        if path.name == SEAL_NAME:
            continue
        size = int(path.stat().st_size)
        total_bytes += size
        entries.append({
            "path": path.relative_to(run_dir).as_posix(),
            "sha256": _sha256_file(path),
            "bytes": size,
        })
    sums_path = run_dir / SHA256SUMS_NAME
    seal = {
        "schema": "na49_mixing.staged_r3_se_audit.seal_manifest/1",
        "run": RUN_ID,
        "card": CARD,
        "created_utc": _now_utc(),
        "file_count": len(entries),
        "total_bytes": int(total_bytes),
        "sha256_sums_file": SHA256SUMS_NAME,
        "sha256_sums_sha256": _sha256_file(sums_path),
        "excluded_from_sums": [SEAL_NAME, SHA256SUMS_NAME,
                               CLEAN_ENV_NAME],
        "path_convention": (
            "the SHA256SUMS.txt entries carry monorepo-root-relative "
            "paths, so the run card's validation command `sha256sum -c "
            "dev/00runs/%s/SHA256SUMS.txt` verifies from the repo root "
            "(the R21 convention)" % run_dir.name
        ),
        "note": (
            "the run dir is immutable after the seal; "
            "clean_environment_commands.json is the post-seal validation "
            "record, written after this manifest and never edited"
        ),
        "files": entries,
    }
    _write_json(run_dir / SEAL_NAME, seal)
    return seal


SEAL_REQUIRED = (
    PREREGISTRATION_NAME,
    PROVENANCE_NAME,
    PRE_INVENTORY_NAME,
    PHASE_A_NAME,
    REEVAL_NAME,
    AGG_STATE_NAME,
)


def run_seal(run_dir: Path) -> None:
    """Seal the finished run directory (sums inventory plus seal
    manifest).  Smoke artifacts are refused; the full production set is
    required."""
    run_dir = Path(run_dir).resolve()
    for name in SEAL_REQUIRED:
        if not (run_dir / name).is_file():
            _refuse("%s is missing before sealing" % (run_dir / name))
    for name in (SUBMISSION_MANIFEST_REL, FETCH_LOG_REL):
        if not (run_dir / name).is_file():
            _refuse("%s is missing before sealing" % name)
    phase_b = run_dir / PHASE_B_NAME
    chunk_files = chunk_files_ok(phase_b)
    if len(chunk_files) != N_WINDOWS * N_NODES * CONFIRMATION_REPLICAS:
        _refuse("the production chunk set is incomplete or carries "
                "smoke artifacts; %d chunk files found" % len(chunk_files))
    n_nodes = len(list(phase_b.glob("window_*_node_*.json"))) - len(
        list(phase_b.glob("window_*_node_*_replica_*.json")))
    if n_nodes != N_WINDOWS * N_NODES:
        _refuse("expected %d node records before sealing, found %d"
                % (N_WINDOWS * N_NODES, n_nodes))
    n_replicas = len(list(phase_b.glob("window_*_node_*_replica_*.json")))
    if n_replicas != N_WINDOWS * N_NODES * CONFIRMATION_REPLICAS:
        _refuse("expected %d replica records before sealing, found %d"
                % (N_WINDOWS * N_NODES * CONFIRMATION_REPLICAS, n_replicas))
    for path in chunk_files:
        record = _read_json(path)
        if str(record.get("mode")) != "production":
            _refuse("smoke artifacts are refused by --seal: %s" % path.name)
    write_sha256sums(run_dir)
    seal = write_seal_manifest(run_dir)
    print(
        "staged_r3_se_audit: run dir sealed (%d files listed in %s)"
        % (len(seal["files"]), SEAL_NAME)
    )


def chunk_files_ok(phase_b: Path) -> list:
    return sorted(phase_b.glob("chunk_w*_n*_r*.json"))


def run_clean_validate(run_dir: Path) -> dict:
    """Verify the seal, re-run --phase-a, --aggregate, and --reeval from
    clean interpreters over the sealed artifacts, and confirm the
    report hashes match; write ``clean_environment_commands.json``."""
    run_dir = Path(run_dir).resolve()
    _load_pre_registration(run_dir)
    sums_path = run_dir / SHA256SUMS_NAME
    if not sums_path.is_file():
        _refuse("%s is missing; --clean-validate requires the sealed run "
                "dir (run --seal first)" % sums_path)
    mismatches: list = []
    files_checked = 0
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        payload = MONO_ROOT / name
        files_checked += 1
        if not payload.is_file() or _sha256_file(payload) != digest:
            mismatches.append(name)
    if mismatches:
        _refuse("seal verification failed; %d of %d SHA256SUMS.txt "
                "entries do not re-hash correctly (first: %r)"
                % (len(mismatches), files_checked, mismatches[0]))
    phase_b = run_dir / PHASE_B_NAME
    targets = [run_dir / PHASE_A_NAME, run_dir / AGG_STATE_NAME,
               run_dir / REEVAL_NAME]
    for w in range(N_WINDOWS):
        for j in range(N_NODES):
            targets.append(phase_b / ("window_%d_node_%d.json" % (w, j)))
            for r in range(CONFIRMATION_REPLICAS):
                targets.append(phase_b / (
                    "window_%d_node_%d_replica_%d.json" % (w, j, r)))
    snapshots = {}
    for path in targets:
        snapshots[path] = path.read_bytes()
    driver_path = Path(__file__).resolve()
    commands = []
    probe_results = {}
    for mode in ("--phase-a", "--aggregate", "--reeval"):
        argv = [sys.executable, "-I", str(driver_path), mode,
                "--run-dir", str(run_dir)]
        probe = subprocess.run(argv, capture_output=True, text=True,
                               cwd=str(MONO_ROOT))
        commands.append({"purpose": "clean-interpreter re-run of %s" % mode,
                         "argv": argv, "cwd": str(MONO_ROOT),
                         "returncode": int(probe.returncode)})
        if probe.returncode != 0:
            _refuse("the clean-interpreter %s probe failed (return code "
                    "%d):\n%s" % (mode, probe.returncode, probe.stderr))
    hash_results = {}
    for path in targets:
        before = snapshots[path]
        after = path.read_bytes()
        same = before == after
        if not same:
            _write_bytes(path, before)
        hash_results[path.relative_to(run_dir).as_posix()] = {
            "hashes_match": bool(same),
            "sha256": hashlib.sha256(before).hexdigest(),
        }
    hashes_match = all(result["hashes_match"]
                       for result in hash_results.values())
    payload = {
        "schema": "na49_mixing.staged_r3_se_audit.clean_environment/1",
        "run": RUN_ID,
        "card": CARD,
        "verified_utc": _now_utc(),
        "seal_verification": {
            "files_checked": int(files_checked),
            "mismatches": mismatches,
        },
        "commands": commands,
        "environment": environment_fingerprint(),
        "sealed_report_hashes": hash_results,
        "hashes_match": bool(hashes_match),
        "probe_tool": {
            "path": str(driver_path),
            "sha256": _sha256_file(driver_path),
            "note": ("the committed dual-tree driver is the probe "
                     "instrument; the sealed artifacts and the committed "
                     "kernel are the inputs"),
        },
    }
    _write_json(run_dir / CLEAN_ENV_NAME, payload)
    if not hashes_match:
        print(
            "staged_r3_se_audit: the clean-environment validation FAILED "
            "(a recomputed report differs from the sealed bytes)",
            flush=True,
        )
        raise SystemExit(1)
    print(
        "staged_r3_se_audit: clean-environment validation passed (%d "
        "seal files verified; Phase A, aggregation, and re-evaluation "
        "reproduced byte for byte from clean interpreters)" % files_checked
    )
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="staged_r3_se_audit.py",
        description=(
            "Execute the Round 3 card 8c SE audit and gateway "
            "confirmation (TASK-20260918-r3-se-audit-confirmation-mc): "
            "Phase A, the hermetic SE-estimator and seed-map audit over "
            "the sealed R31/R21/R22/card-2 artifacts; Phase B, the "
            "T1-scaled confirmation MC on the cluster gateway (15 nodes "
            "x 50 replicas x 20,000 decays, master seed 20260918) and "
            "the re-evaluation of all 422 R31 checks with the fixed "
            "engine at the old SE versus the new SE.  The 3.0-sigma "
            "tolerance is never relaxed."
        ),
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--pre-registration", action="store_true",
        help="write pre_registration.json, provenance.json, and "
             "pre_investigation_inventory.json, then stop",
    )
    modes.add_argument(
        "--phase-a", action="store_true",
        help="run the hermetic Phase A replay, estimators, seed-map "
             "audit, and classifications, then stop",
    )
    modes.add_argument(
        "--chunk", action="store_true",
        help="generate one Phase B chunk (--window W --node J "
             "--replica R; 20,000 decays in production)",
    )
    modes.add_argument(
        "--aggregate", action="store_true",
        help="verify and merge the fetched chunk outputs, then stop",
    )
    modes.add_argument(
        "--reeval", action="store_true",
        help="re-evaluate all R31 checks at the confirmation "
             "statistics, then stop",
    )
    modes.add_argument(
        "--seal", action="store_true",
        help="write seal_manifest.json and SHA256SUMS.txt, then stop",
    )
    modes.add_argument(
        "--clean-validate", action="store_true",
        help="verify the seal and re-run Phase A, the aggregation, and "
             "the re-evaluation cleanly, then stop",
    )
    parser.add_argument("--window", type=int, default=None,
                        help="with --chunk: the window index w = 0..2")
    parser.add_argument("--node", type=int, default=None,
                        help="with --chunk: the node index j = 0..4")
    parser.add_argument("--replica", type=int, default=None,
                        help="with --chunk: the replica index r' = 0..49")
    parser.add_argument("--decays", type=int, default=None,
                        help="with --chunk --smoke: the smoke chunk size "
                             "(default %d)" % SMOKE_DECAYS)
    parser.add_argument(
        "--smoke", action="store_true",
        help="hermetic reduced statistics (driver validation); smoke "
             "artifacts live under phase_b_smoke/ and are refused by "
             "--seal and --clean-validate",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_AUDIT_WORKERS,
                        help="Phase A fork worker count (default %d)"
                             % DEFAULT_AUDIT_WORKERS)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR,
                        help="the run directory (default %s)"
                             % DEFAULT_RUN_DIR)
    return parser


def main(argv: list | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_dir_text = str(args.run_dir)
    if run_dir_text.rstrip("/").endswith(PHASE_B_NAME) or \
            run_dir_text.rstrip("/").endswith(SMOKE_DIR_NAME):
        run_dir = Path(run_dir_text).parent
    else:
        run_dir = Path(run_dir_text)
    if args.smoke and not (args.chunk or args.aggregate or args.reeval):
        _refuse("--smoke is only valid with --chunk / --aggregate / "
                "--reeval")
    if args.pre_registration:
        run_pre_registration(run_dir)
    elif args.phase_a:
        run_phase_a(run_dir, int(args.workers))
    elif args.chunk:
        run_chunk(run_dir, args.window, args.node, args.replica,
                  bool(args.smoke), args.decays)
    elif args.aggregate:
        run_aggregate(run_dir, bool(args.smoke))
    elif args.reeval:
        run_reeval(run_dir, bool(args.smoke))
    elif args.seal:
        run_seal(run_dir)
    elif args.clean_validate:
        run_clean_validate(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
