#!/usr/bin/env python3
"""R32 driver: the p+p spectral systematic (Round 3, card 9).

Task ``TASK-20260918-r3-run-r32-spectral-systematic-pp`` (plan section 9
run R32, line 644; the Round 3 card set, card 9; the plan section 23
line 1817 Round 3 continuum card set).  R32 runs the full p+p collision
at ``sqrt(s) = 17.2 GeV`` for each of the three pre-registered
transition-window variants (round-3 index item 5), replacing the R23
hard kernel switch at the boundary with the card-6 blend routing, and
records the spread across variants as the model uncertainty of the
catalog/continuum transition.  No topology refit, no refit of the M8
frozen ten-parameter set, no optimizer (plan section 24): the spread is
treated as model uncertainty unless data clearly select one option, and
the first pass performs no data selection (plan section 9, line 646).

Variants (round 3 index item 5): three variants of the card-6 blend
with the ramp width ``W = 0.5 GeV`` fixed and every other decay
parameter (catalog, branching fractions, family parameters) identical:

- v0 (nominal): ramp start ``B = 2.680194110727528 GeV`` (the
  M8-frozen ``continuum.match_mass``);
- v1 (early): ramp start ``B - 0.2 GeV = 2.480194110727528 GeV``;
- v2 (late): ramp start ``B + 0.2 GeV = 2.880194110727528 GeV``.

Statistics (T2, plan section 5.1): 8 replicas x 100,000 p+p events per
variant (the upper T2 bound, consistent with R23's 800,000-event T2
bound); 3 x 800,000 = 2,400,000 events total.

Seeds: a NEW master seed ``20260919`` (the run execution date), disjoint
from ``20260913`` (R20-R23), ``20260917`` (R31 and the retired card 9
spec), and ``20260918`` (card 8c).  Per (variant ``v``, replica
``r < 8``): raw replica seed ``20260919 + 40000 + 600 + 100*v + r =
20301519 + 100*v + r`` (v0: 20301519-20301526, v1: 20301619-20301626,
v2: 20301719-20301726).  The kernel-layer substreams derive from the
same master with the variant folded into the replica index:
``derive_substreams(20260919, 100*v + r, event_id)`` on the closed
``decay_channel`` / ``decay_kinematics`` labels, so all 24
(variant, replica) kernel-layer spaces are pairwise disjoint.  The
legacy source/exchange layers draw from the legacy staged per-event
generator (``SeedSequence(seed, spawn_key=(event_id,))``).  The
pre-registered robust-statistics confirmation blocks (acceptance (v),
triggered only by finite-MC-scatter diagnostics) use the raw-seed rule
``20260919 + 40000 + 900 + 100*v + r`` (blocks 20301819-20301834 /
20301919-20301934 / 20302019-20302034, 16 replicas per variant,
gateway).  The hermetic smoke mode uses the ``+5000`` block
(raw seeds 20305919 + 100*v + r, kernel replica index 5000 + 100*v +
r), disjoint from every production and confirmation space.

Builder: the R23 ``ResonancePPBuilder``
(``na49_mixing.resonance_pp_builder.py``, card 13) extended additively
with the ``blend`` parameter (``None`` keeps the R23 behavior
byte-stable, asserted by the card-13 pins and by the R32 regression
pin over the sealed R23 replica-0 stream's first 1,000 event hashes; a
``BlendSpec`` routes the source decay through the card-6 blend with
the variant ramp start).  The card-6 blend surface
(``na49_mixing/spectral_density.py``) is committed and byte-unchanged.

Bit-identity check (variant v0): 10,000 events generated at the R23
replica-0 seed ``20300913``; the topology, exchange kinematics, and
spectral-state four-momenta layers are bit-identical between R32 v0,
the frozen R23 builder (``blend = None``), and the legacy staged
builder, event by event (the R23 convention).  The above-boundary
finals differ from R23 by design (the legacy H3 kernel is replaced by
the continuum routing); that layer is recorded, not asserted.  A layer
mismatch stops the run before any replica is generated and re-opens
the builder extension.

Comparisons (per variant, on the pooled 8-replica statistics): the
pooled 1017-bin six-species paper-window predictions (the ``sps_core``
windows, the R01 binning), the five-cell low-mass-sensitive set (the
R11 ranked-residual cells, the R23 convention), and the per-species
feed-down fractions under the corrected lineage convention of card 3
(a final particle is resonant feed-down when its producing vertex's
parent is a resonance other than the drawn root: the catalog route's
drawn record, the continuum route's root sentinel 0).  Cross-variant
spread: per bin, per cell, and per species, the min/max and the median
absolute deviation across variants.

Spread bounds (pre-registered, the model-uncertainty contract): the
95th percentile of the absolute bin-level cross-variant deviation is
< 2.0 in units of the pooled MC standard error of the comparison, and
the per-species feed-down-fraction spread is < 3.0 sigma of its
combined SE.  The pooled MC standard error of a bin combines the three
variants' pooled per-bin MC errors (the R01 convention
``value * sqrt(1/N_bin)`` on the pooled counts) as
``sqrt(SE_0^2 + SE_1^2 + SE_2^2)``; the absolute bin-level deviation is
the maximum pairwise deviation ``(max - min)`` across variants; the
median variant per bin is recorded as the reference (its value and its
own pooled MC standard error are recorded beside the pooled unit).
Under pure Monte-Carlo scatter the 95th percentile of the normalized
three-variant range sits near 1.8, inside the 2.0 bound; a systematic
model shift between variants pushes it above.  The feed-down spread is
the ``(max - min)`` across variants over the combined SE
``sqrt(SE_0^2 + SE_1^2 + SE_2^2)`` of the three variants'
between-replica standard errors (the R21/R22 convention, std ddof=1
over sqrt(8)).  Exceeding a bound is a Gate 3 finding, not a reason to
refit: the failing verdicts are named in the ``named_residual_register``
(the register shape of the sealed ``gate3_r31.json``), each with a
one-line first-pass diagnosis (finite MC scatter vs systematic vs
SE-estimator artifact under the ``DETERMINISTIC_SE_FLOOR`` convention).

Conservation: every p+p event conserves total ``B = 2`` and ``Q = 2``
exactly; every source conserves its ``(B, Q, S)`` exactly (the Round 2
kernel conservation, checked per event, never repaired, a violation
raises and stops the run).

Routing census: per variant and replica, the fraction of sources
routed catalog vs continuum family (recorded; v1/v2 shift the census,
the shifts are recorded, not asserted).

No refit: ``no_fitting`` is ``true``; no optimizer is invoked; no
topology, exchange-mass, decay, or branching parameter is refit (plan
section 24).

Modes (exactly one per invocation; each performs one step and stops):

- ``--pre-registration``: write ``pre_registration.json`` and
  ``provenance.json`` into the run dir, then stop.  Nothing is
  generated.  Verifies the sealed R23/R30/R31/card-2/card-8a/card-8c
  seals and the frozen substrate digests first.
- ``--bit-identity``: run the pre-registered 10,000-event bit-identity
  check at the R23 replica-0 seed 20300913 (v0 blend vs the frozen R23
  builder vs the legacy staged builder), write
  ``bit_identity_v0.json``, then stop.  A layer mismatch stops the run
  (exit 1) after the report is written.
- ``--generate [--variant V] [--replica R]``: generate one (variant,
  replica) chunk file, one whole variant, or every variant, writing
  ``variant_<v>_replica_<r>.json``, then stop.  Requires the
  pre-registration and a passing bit-identity check; the statistics
  flags must match the pre-registration in production mode.  A
  completed record that matches the pre-registered identity is reused
  unchanged, so the declared tier can be completed replica by replica
  (the pre-registered gateway fallback runs one job per replica).
- ``--compare``: run the spread evaluation (the per-variant pooled
  comparisons and the cross-variant spread metrics), write
  ``spread_report.json``, then stop.  Requires all 24 production
  records.  A spread bound failure is recorded and named; the step
  still exits 0 (the run seals with the named-residual register; a
  bound failure is a Gate 3 finding, never a refit).
- ``--gate``: write ``gate3_r32.json`` (evidence only), then stop.
  Requires the spread report and the passing bit-identity check.
- ``--seal``: write ``seal_manifest.json`` and ``SHA256SUMS.txt`` over
  the run dir, then stop.  The run dir is immutable after the seal.
- ``--clean-validate``: verify the seal, re-run the spread evaluation
  from a clean interpreter over the sealed artifacts, and confirm the
  report hash matches; write ``clean_environment_commands.json``, then
  stop.
- ``--smoke``: the hermetic end-to-end driver validation (never
  sealed): a smoke pre-registration, 2 variants x 1 replica x 2,000
  events at the pre-registered smoke seeds on a synthetic hermetic
  binning grid, and the smoke spread evaluation, into an explicit
  ``--run-dir`` (the default production run dir is refused).  Two
  identical smoke invocations produce byte-identical artifacts.  The
  ``--gate`` / ``--seal`` / ``--clean-validate`` modes refuse smoke
  artifacts.

``spread_report.json`` and ``gate3_r32.json`` carry no wall-clock
fields: their bytes are a pure function of the sealed artifacts and the
committed code, which is what the clean-environment validation
re-hashes.

Dual-tree driver: byte-identical in the nested tree
(``dev/mixing2006/helpers/benchmarking/``) and the monorepo mirror
(``helpers/benchmarking/``); both locate the package with the
established ``_locate_roots`` pattern.

Hermetic in every mode except production ``--generate`` / ``--compare``
table loading: the committed catalog, the committed continuum config,
the committed production config declarations, and the sealed archives
only.  The production comparison binning loads the six-species SPS
tables (the ``spshadrons.web.cern.ch/`` archive) like the R23 driver;
the smoke mode builds a synthetic hermetic grid from the committed
table declarations instead.
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
# na49_mixing package, the resonance catalog, the continuum config, and
# the sealed reference drivers.
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
    print(
        f"staged_r32_spectral_systematic: {message}",
        file=sys.stderr, flush=True,
    )
    raise SystemExit(2)


NESTED_ROOT, MONO_ROOT = _locate_roots()
if str(NESTED_ROOT) not in sys.path:
    sys.path.insert(0, str(NESTED_ROOT))

from helpers.benchmarking import staged_observable_decomposition as r11  # noqa: E402
from helpers.benchmarking import staged_r01_reference as r01  # noqa: E402
from helpers.benchmarking import staged_r23_resonance_ablation as r23drv  # noqa: E402
from helpers.benchmarking import sps_core  # noqa: E402
from na49_mixing.event_builder_pp import PPEventBuilder  # noqa: E402
from na49_mixing.hadron_catalog import HadronCatalog  # noqa: E402
from na49_mixing.pdg_database import get_particle  # noqa: E402
from na49_mixing.reaction import Reaction  # noqa: E402
from na49_mixing.resonance_pp_builder import (  # noqa: E402
    CONTINUUM_LABEL,
    CONTINUUM_ROOT_SENTINEL,
    KERNEL_BOUNDARY_M8,
    KERNEL_LABEL,
    ROUND2_MASTER_SEED,
    ResonancePPBuilder,
)
from na49_mixing.spectral_density import (  # noqa: E402
    BlendSpec,
    families_from_json,
)
from na49_mixing.staged_evaluation import BinnedFitTable  # noqa: E402
from na49_mixing.threshold_metric import DETERMINISTIC_SE_FLOOR  # noqa: E402

# ---------------------------------------------------------------------------
# Pre-registered run identity (card TASK-20260918-r3-run-r32-spectral-
# systematic-pp; the variants, statistics, seeds, spread bounds, and
# bit-identity clause are verbatim from the card and the round 3 index
# items 4, 5 and 7).
# ---------------------------------------------------------------------------

RUN_ID = "R32"
CARD = "TASK-20260918-r3-run-r32-spectral-systematic-pp"
RUN_NAME = "r32_spectral_systematic_pp_2026-09-18"
DEFAULT_RUN_DIR = "dev/00runs/r32_spectral_systematic_pp_2026-09-18"
CONFIRMATION_RUN_NAME = "r32_spectral_systematic_pp_confirmation_2026-09-19"
GATE_RECORD_NAME = "gate3_r32.json"
SPREAD_REPORT_NAME = "spread_report.json"
PREREGISTRATION_NAME = "pre_registration.json"
PROVENANCE_NAME = "provenance.json"
BIT_IDENTITY_NAME = "bit_identity_v0.json"
CLEAN_ENV_NAME = "clean_environment_commands.json"
SHA256SUMS_NAME = "SHA256SUMS.txt"
SEAL_NAME = "seal_manifest.json"
VARIANT_FMT = "variant_{}_replica_{}.json"
SMOKE_PREFIX = "smoke_"

CATALOG_REL = "config/v2_singlet/resonance_catalog.json"
FAMILIES_REL = "config/v2_singlet/continuum_resonances.json"
DRIVER_REL = "helpers/benchmarking/staged_r32_spectral_systematic.py"
ENGINE_REL = "dev/mixing2006/na49_mixing/threshold_metric.py"
BUILDER_REL = "dev/mixing2006/na49_mixing/resonance_pp_builder.py"

#: The card chain this run sits on (the Round 3 prerequisites).
CARD_CHAIN = (
    "TASK-20260917-r3-gate2-followup-feeddown-lineage-convention",
    "TASK-20260917-r3-run-r30-spectral-normalization",
    "TASK-20260917-r3-run-r31-catalog-continuum-matching-scan",
    "TASK-20260918-r3-card2-engine-three-body-moment-fix",
    "TASK-20260918-r3-se-audit-confirmation-mc",
    "TASK-20260918-r3-acceptance-rescope-card8a-8b",
    "TASK-20260918-r3-run-r32-spectral-systematic-pp",
)

#: The three transition-window variants (round 3 index item 5): the
#: card-6 blend with the ramp width fixed, only the ramp start varied.
RAMP_WIDTH_GEV = 0.5
VARIANTS: tuple[dict, ...] = (
    {"variant": 0, "label": "v0 (nominal)",
     "ramp_start_gev": 2.680194110727528},
    {"variant": 1, "label": "v1 (early)",
     "ramp_start_gev": 2.480194110727528},
    {"variant": 2, "label": "v2 (late)",
     "ramp_start_gev": 2.880194110727528},
)
N_VARIANTS = len(VARIANTS)

#: T2 statistics (plan section 5.1): 8 replicas x 100,000 p+p events
#: per variant; 3 x 800,000 = 2,400,000 events total.
T2_REPLICAS = 8
T2_EVENTS_PER_REPLICA = 100_000
T2_LABEL = (
    "T2: 8 replicas x 100,000 p+p events per variant (plan section 5.1, "
    "upper bound, consistent with R23's 800,000-event T2 bound)"
)
TOTAL_EVENTS = N_VARIANTS * T2_REPLICAS * T2_EVENTS_PER_REPLICA

#: Master seed and the per-replica seed map (the card's seed scheme).
MASTER_SEED = 20260919
SEED_BASE = 20260919
SEED_OFFSET = 40_000
SEED_VARIANT_STRIDE = 600
RAW_SEED_RULE = (
    "raw replica seed = 20260919 + 40000 + 600 + 100*v + r = "
    "20301519 + 100*v + r per (variant v, replica r < 8); v0: "
    "20301519-20301526, v1: 20301619-20301626, v2: 20301719-20301726"
)
KERNEL_SEED_RULE = (
    "derive_substreams(20260919, 100*v + r, event_id) per (variant v, "
    "replica r < 8) on the closed decay_channel/decay_kinematics labels "
    "(the card-9 seed map with the variant folded into the replica "
    "index; all 24 (variant, replica) kernel-layer spaces are pairwise "
    "disjoint)"
)
LEGACY_EVENT_SEED_RULE = (
    "numpy.random.default_rng(numpy.random.SeedSequence(seed, "
    "spawn_key=(event_id,))); the legacy staged p+p convention (the R01 "
    "driver's rule); one shared generator serves the topology "
    "selection, exchange kinematics, and spectral-state layers and the "
    "legacy decay branch"
)
DISJOINTNESS_NOTE = (
    "the R32 master seed 20260919 is disjoint from 20260913 (R20-R22 "
    "substream space; the R23 raw replica seeds 20300913-20300920), "
    "from 20260917 (the R31 substream space; the retired card 9 spec's "
    "planned raw seeds 20301217-20301424, never drawn), and from "
    "20260918 (the card-8c substream space); the R32 raw-seed blocks "
    "(20301519-20301526, 20301619-20301626, 20301719-20301726) are "
    "disjoint from the R23 raw seeds, from the retired card 9's planned "
    "seeds, from every substream-derived space of masters "
    "20260913/20260917/20260918, and from the pre-registered "
    "confirmation (+900) and smoke (+5000) blocks"
)

#: The bit-identity check (variant v0): 10,000 events at the R23
#: replica-0 seed, the three source layers asserted, the finals
#: recorded.
BIT_IDENTITY_SEED = 20300913
BIT_IDENTITY_EVENTS = 10_000
BIT_IDENTITY_MASTER_SEED = ROUND2_MASTER_SEED
BIT_IDENTITY_REPLICA_INDEX = 0
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
}

#: The spread bounds (the pre-registered model-uncertainty contract).
BIN_SPREAD_BOUND = 2.0
FEEDDOWN_SPREAD_BOUND = 3.0
SE_FLOOR = float(DETERMINISTIC_SE_FLOOR)
#: The first-pass finite-MC-scatter envelope of the named-residual
#: diagnosis: a failing verdict with |sigma| <= 4.0 is provisionally
#: finite-MC scatter (the confirmation-eligible class; under pure
#: Monte-Carlo scatter the maximum normalized three-variant range over
#: the 1017-bin check set reaches about 3.4 sigma), a larger one a
#: systematic candidate (the responsible card re-opens, no confirmation
#: trigger).  The 3.0-sigma metric band is never relaxed anywhere.
FINITE_MC_SCATTER_MAX_SIGMA = 4.0

#: The robust-statistics confirmation sub-run (acceptance (v),
#: triggered only by finite-MC-scatter diagnostics): 16 replicas x
#: 100,000 events per variant (4,800,000 events), the same master seed,
#: the +900 raw-seed blocks, gateway.
CONFIRMATION_REPLICAS = 16
CONFIRMATION_SEED_STRIDE = 900
CONFIRMATION_RAW_SEED_RULE = (
    "raw replica seed = 20260919 + 40000 + 900 + 100*v + r per "
    "(variant v, confirmation replica r < 16); blocks "
    "20301819-20301834 (v0), 20301919-20301934 (v1), 20302019-20302034 "
    "(v2); 4,800,000 events; gateway, 48 jobs of 100,000 events, "
    "--tasks 1"
)
CONFIRMATION_KERNEL_REPLICA_RULE = (
    "derive_substreams(20260919, 100*v + 900 + r, event_id) per "
    "(variant v, confirmation replica r < 16); the confirmation "
    "kernel-layer spaces are disjoint from the production spaces"
)

#: The hermetic smoke mode (driver validation; never sealed): 2
#: variants x 1 replica x 2,000 events at the pre-registered smoke
#: seeds on a synthetic hermetic binning grid.
SMOKE_VARIANTS = (0, 2)
SMOKE_REPLICAS = 1
SMOKE_EVENTS = 2_000
SMOKE_SEED_STRIDE = 5000
#: The smoke raw-seed base: 20260919 + 40000 + 5000 = 20305919.
SMOKE_RAW_SEED_BASE = SEED_BASE + SEED_OFFSET + SMOKE_SEED_STRIDE
SMOKE_RAW_SEED_RULE = (
    "raw replica seed = 20260919 + 40000 + 5000 + 100*v + r per "
    "(smoke variant v, smoke replica r < 1); the +5000 smoke block "
    "(20305919 + 100*v + r) is disjoint from every production, "
    "confirmation, and prior-run seed space"
)
SMOKE_KERNEL_REPLICA_RULE = (
    "derive_substreams(20260919, 5000 + 100*v + r, event_id); the "
    "smoke kernel-layer spaces are disjoint from the production and "
    "confirmation spaces"
)
SMOKE_GRID_XF_BINS = 10
SMOKE_GRID_PT_BINS = 10

#: The five-cell low-mass-sensitive set (the R11 ranked-residual cells,
#: the R23 convention), read from the sealed archive at pre-registration.
R11_RUN_DIR = "dev/00runs/r11_observable_decomposition_2026-09-11"
R01_RUN_DIR = "dev/00runs/r01_legacy_pp_reference_2026-09-11"

#: The sealed run dirs whose seals the pre-registration verifies
#: (the card's read-only substrate).
SEALED_RUN_DIRS: tuple[str, ...] = (
    "dev/00runs/r23_resonance_only_pp_ablation_2026-09-13",
    "dev/00runs/r30_spectral_normalization_2026-09-17",
    "dev/00runs/r31_catalog_continuum_matching_scan_2026-09-17",
    "dev/00runs/r3_threshold_metric_reregistration_2026-09-17",
    "dev/00runs/r3_threshold_metric_engine_fix_2026-09-18",
    "dev/00runs/r3_se_audit_2026-09-18",
)

#: The frozen substrate digests (the card's substrate list).
ENGINE_SHA256 = (
    "ee240f44645d2e592acfae462f7d1e9c3b3dd8cd78c96bd40b7b73509f7fcc63"
)
BUILDER_R23_PRE_EXTENSION_SHA256 = (
    "bde0fa9a228f3e9b713cc646bddfc64e2b66b7968a5db3e0c8050d169e81e036"
)
R31_DRIVER_SHA256 = (
    "b47b2a858c3880ecf69e143035e8c209b6e6e5e0708ca2b9058c6c384340afe6"
)
R3_SE_AUDIT_DRIVER_SHA256 = (
    "2c72fa7e7b5a86a54076f6cc686bb33d165d2b280092df7ca6fd75d3b5635081"
)
CATALOG_SHA256 = (
    "e42a5da7137d789ba90221478c9bb404898f4fe8a7ee4604cd85b5623b751752"
)
FAMILIES_SHA256 = (
    "6da57d5709b94061bc178fe94781dfe5eddcc597c5dde46bfcd5db56985cde7d"
)

#: The pinned blend=None regression sample: the sealed R23 replica-0
#: stream's first 1,000 event hashes (regenerated pre-extension and
#: verified against the sealed replica record's event_hash_set_sha256
#: 503ab64e3ce23be92050e7e4e657a4aa4e276c2b9c267de7a14f86b7521221ec;
#: embedded verbatim in the R32 hermetic test).
BLEND_NONE_SAMPLE_N = 1_000
BLEND_NONE_SAMPLE_SEED = 20300913
SEALED_R23_EVENT_HASH_SET_SHA256 = (
    "503ab64e3ce23be92050e7e4e657a4aa4e276c2b9c267de7a14f86b7521221ec"
)

#: The kernel boundary (the M8-frozen continuum.match_mass; the v0
#: blend ramp start is the same value).
KERNEL_BOUNDARY = KERNEL_BOUNDARY_M8

#: Generation chunking (the R01/R23 chunk convention).
CHUNK_SIZE = 2500
DEFAULT_WORKERS = 4

#: The event-seed stage label of the R32 stream.
STAGE_LABEL = "r32_spectral_systematic"


def _now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _write_json(path: Path, payload: dict) -> None:
    """UTF-8 JSON with the R01 sentinel convention (identical inputs
    produce identical bytes; non-finite floats become string
    sentinels)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(r01._json_safe(payload), indent=2, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


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


def _sha256_file(path: Path) -> str:
    return r01.sha256_file(path)


def _write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


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


# ---------------------------------------------------------------------------
# Seed map helpers (the card's seed scheme, verbatim)
# ---------------------------------------------------------------------------


def raw_replica_seed(variant: int, replica: int) -> int:
    """The production raw replica seed ``20301519 + 100*v + r``."""
    return SEED_BASE + SEED_OFFSET + SEED_VARIANT_STRIDE + 100 * int(
        variant
    ) + int(replica)


def kernel_replica_index(variant: int, replica: int) -> int:
    """The kernel-layer substream replica index ``100*v + r``."""
    return 100 * int(variant) + int(replica)


def confirmation_raw_seed(variant: int, replica: int) -> int:
    """The confirmation (+900) raw seed (acceptance (v))."""
    return (
        SEED_BASE + SEED_OFFSET + CONFIRMATION_SEED_STRIDE
        + 100 * int(variant) + int(replica)
    )


def confirmation_kernel_replica_index(variant: int, replica: int) -> int:
    """The confirmation kernel-layer replica index ``100*v + 900 + r``."""
    return 100 * int(variant) + CONFIRMATION_SEED_STRIDE + int(replica)


def smoke_raw_seed(variant: int, replica: int) -> int:
    """The smoke (+5000) raw seed (the hermetic smoke block)."""
    return (
        SEED_BASE + SEED_OFFSET + SMOKE_SEED_STRIDE
        + 100 * int(variant) + int(replica)
    )


def smoke_kernel_replica_index(variant: int, replica: int) -> int:
    """The smoke kernel-layer replica index ``5000 + 100*v + r``."""
    return SMOKE_SEED_STRIDE + 100 * int(variant) + int(replica)


def variant_blend(variant: int) -> BlendSpec:
    """The card-6 blend of the pre-registered variant (the ramp width
    fixed at :data:`RAMP_WIDTH_GEV`; only the ramp start varies)."""
    for entry in VARIANTS:
        if int(entry["variant"]) == int(variant):
            return BlendSpec(
                boundary=float(entry["ramp_start_gev"]),
                ramp_width=RAMP_WIDTH_GEV,
            )
    raise ValueError(f"unknown variant index {variant!r}")


def _variant_record_name(variant: int, replica: int) -> str:
    return VARIANT_FMT.format(int(variant), int(replica))


# ---------------------------------------------------------------------------
# Shared input loaders and species conventions (the card-9/R23 driver
# conventions)
# ---------------------------------------------------------------------------


def _load_catalog() -> HadronCatalog:
    return HadronCatalog.from_json(NESTED_ROOT / CATALOG_REL)


def _load_families() -> list:
    return families_from_json(str(NESTED_ROOT / FAMILIES_REL))


def _load_raw_config() -> dict:
    path = NESTED_ROOT / "config" / r01.PRODUCTION_CONFIG_NAME
    if not path.is_file():
        _refuse(f"production config missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _m8_config():
    """The M8-frozen p+p config (the R01 driver's application rule)."""
    raw_config = _load_raw_config()
    config, _applied = r01.build_pp_config(raw_config)
    return config


def _sqrt_s() -> float:
    beam = float(_load_raw_config().get("beam_momentum", 158.0))
    return float(Reaction(beam, 2212, 2212).sqrt_s())


def _species_key(name: str) -> str:
    """The card-9 species-key transliteration (``+`` -> ``_plus``, ``-``
    -> ``_minus``, a trailing ``0`` -> ``_0``), otherwise verbatim."""
    name = str(name)
    if name.endswith("+"):
        return name[:-1] + "_plus"
    if name.endswith("-"):
        return name[:-1] + "_minus"
    if name.endswith("0"):
        return name[:-1] + "_0"
    return name


def _record_name(catalog: HadronCatalog, pdg_id: int) -> str:
    """The species name of a particle: the catalog record's name for
    records, the vendored PDG table's name otherwise, the bare id when
    neither resolves (the card-9 driver's convention)."""
    try:
        return str(catalog.get(int(pdg_id)).name)
    except KeyError:
        entry = get_particle(int(pdg_id))
        if entry is None:
            return str(int(pdg_id))
        return str(entry.name)


def _stable_species_keys(catalog: HadronCatalog) -> tuple[str, ...]:
    """The per-species key set of the diagnostics: the catalog's stable
    termination set, in catalog record order, with the charge-symbol
    transliteration (the card-9 multiplicity species set)."""
    stable = catalog.stable_states()
    return tuple(
        _species_key(record.name)
        for record in catalog.records()
        if int(record.pdg_id) in stable
    )


def _input_hashes() -> dict:
    return {
        "catalog_sha256": _sha256_file(NESTED_ROOT / CATALOG_REL),
        "families_config_sha256": _sha256_file(NESTED_ROOT / FAMILIES_REL),
        "engine_module_sha256": _sha256_file(
            MONO_ROOT / ENGINE_REL
        ),
        "builder_module_sha256": _sha256_file(MONO_ROOT / BUILDER_REL),
        "r23_driver_sha256": _sha256_file(
            NESTED_ROOT / "helpers" / "benchmarking"
            / "staged_r23_resonance_ablation.py"
        ),
        "r31_driver_sha256": _sha256_file(
            NESTED_ROOT / "helpers" / "benchmarking"
            / "staged_r31_matching_scan.py"
        ),
        "driver_sha256": _sha256_file(Path(__file__).resolve()),
    }


# ---------------------------------------------------------------------------
# Frozen substrate and sealed-archive verification (read-only)
# ---------------------------------------------------------------------------


def _verify_sealed_dir(rel: str) -> dict:
    """Verify one sealed run dir's ``SHA256SUMS.txt`` (read-only; the
    sealed archives are byte-unchanged inputs).  The entries carry
    either monorepo-root-relative paths (the R21/R31 convention) or
    run-dir-relative paths (the R23/R30 convention); both resolve."""
    sums_path = MONO_ROOT / rel / SHA256SUMS_NAME
    if not sums_path.is_file():
        _refuse(f"sealed run dir carries no {SHA256SUMS_NAME}: {sums_path}")
    run_dir = MONO_ROOT / rel
    mismatches: list[str] = []
    entries = 0
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        payload = MONO_ROOT / name
        if not payload.is_file():
            payload = run_dir / name
        entries += 1
        if not payload.is_file() or _sha256_file(payload) != digest:
            mismatches.append(name)
    if mismatches:
        _refuse(
            f"sealed run dir {rel}: {len(mismatches)} of {entries} "
            f"{SHA256SUMS_NAME} entries do not re-hash correctly (first: "
            f"{mismatches[0]!r}); the sealed archive must be unmodified"
        )
    return {"path": rel, "entries": int(entries), "verified": True}


def _verify_frozen_substrate() -> dict:
    """The frozen substrate digest checks (the card's substrate list)."""
    problems: list[str] = []
    hashes = _input_hashes()
    if hashes["engine_module_sha256"] != ENGINE_SHA256:
        problems.append(
            "the fixed card-2 engine sha256 "
            f"{hashes['engine_module_sha256']!r} != the pre-registered "
            f"{ENGINE_SHA256!r}"
        )
    if hashes["catalog_sha256"] != CATALOG_SHA256:
        problems.append(
            "the committed catalog sha256 "
            f"{hashes['catalog_sha256']!r} != the pre-registered "
            f"{CATALOG_SHA256!r}"
        )
    if hashes["families_config_sha256"] != FAMILIES_SHA256:
        problems.append(
            "the committed continuum config sha256 "
            f"{hashes['families_config_sha256']!r} != the pre-registered "
            f"{FAMILIES_SHA256!r}"
        )
    if hashes["r31_driver_sha256"] != R31_DRIVER_SHA256:
        problems.append(
            "the frozen R31 driver sha256 "
            f"{hashes['r31_driver_sha256']!r} != the pre-registered "
            f"{R31_DRIVER_SHA256!r}"
        )
    audit_path = (
        NESTED_ROOT / "helpers" / "benchmarking" / "staged_r3_se_audit.py"
    )
    audit_sha = _sha256_file(audit_path)
    if audit_sha != R3_SE_AUDIT_DRIVER_SHA256:
        problems.append(
            "the frozen R3-SE audit driver sha256 "
            f"{audit_sha!r} != the pre-registered "
            f"{R3_SE_AUDIT_DRIVER_SHA256!r}"
        )
    # The R23 builder's pre-extension digest, as pinned by the sealed
    # R23 replica records (the byte-unchanged reference).
    r23_dir = MONO_ROOT / SEALED_RUN_DIRS[0]
    for replica_seed in (
        20300913, 20300914, 20300915, 20300916,
        20300917, 20300918, 20300919, 20300920,
    ):
        record_path = r23_dir / f"replica_{replica_seed}.json"
        if not record_path.is_file():
            _refuse(f"sealed R23 replica record missing: {record_path}")
        record = _read_json(record_path)
        recorded = str(record.get("builder_module_sha256", ""))
        if recorded != BUILDER_R23_PRE_EXTENSION_SHA256:
            problems.append(
                f"sealed R23 replica {replica_seed} pins builder sha256 "
                f"{recorded!r} != the pre-extension reference "
                f"{BUILDER_R23_PRE_EXTENSION_SHA256!r}"
            )
    # The M8 frozen block (the manifest is the single source of truth).
    manifest_path = (
        NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    m8_block = manifest.get("frozen_fit_outputs", {}).get(
        "m8_frozen_pp_parameters"
    )
    if m8_block != r01.M8_FROZEN_PP_PARAMETERS:
        problems.append(
            "the manifest m8_frozen_pp_parameters block differs from the "
            "driver's verbatim frozen copy"
        )
    if r01.canonical_json_sha256(m8_block) != r01.M8_PARAMETER_HASH:
        problems.append(
            "the manifest m8_frozen_pp_parameters canonical sha256 != "
            "the pre-registered M8 parameter hash"
        )
    if problems:
        _refuse(
            "frozen substrate verification failed; refusing to run:\n  "
            + "\n  ".join(problems)
        )
    return {
        "sealed_run_dirs": [
            _verify_sealed_dir(rel) for rel in SEALED_RUN_DIRS
        ],
        "engine_sha256": hashes["engine_module_sha256"],
        "builder_pre_extension_r23_sha256": (
            BUILDER_R23_PRE_EXTENSION_SHA256
        ),
        "builder_post_extension_sha256": hashes["builder_module_sha256"],
        "r31_driver_sha256": hashes["r31_driver_sha256"],
        "r3_se_audit_driver_sha256": audit_sha,
        "r23_driver_sha256": hashes["r23_driver_sha256"],
        "catalog_sha256": hashes["catalog_sha256"],
        "families_config_sha256": hashes["families_config_sha256"],
        "m8_parameter_hash": r01.M8_PARAMETER_HASH,
    }


# ---------------------------------------------------------------------------
# Mode: --pre-registration
# ---------------------------------------------------------------------------


def _r11_top5() -> tuple[list[dict], str]:
    """The sealed R11 ranked residual map's top-5 rows and the
    archive's ``SHA256SUMS.txt`` digest (the R23 convention)."""
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
    sums_path = MONO_ROOT / R11_RUN_DIR / SHA256SUMS_NAME
    if not sums_path.is_file():
        _refuse(f"sealed R11 archive carries no {SHA256SUMS_NAME}: {sums_path}")
    return [dict(rows[i]) for i in range(5)], _sha256_file(sums_path)


def build_pre_registration(catalog: HadronCatalog, families: list) -> dict:
    """The R32 pre-registration payload, written before any generation."""
    rows, r11_seal = _r11_top5()
    raw_config = _load_raw_config()
    binned = (raw_config.get("fit") or {}).get("binned") or {}
    return {
        "schema": "na49_mixing.staged_r32_spectral_systematic"
                  ".pre_registration/1",
        "run": RUN_ID,
        "card": CARD,
        "written_before_generation": True,
        "collision": {
            "system": "p+p",
            "sqrt_s_gev": _sqrt_s(),
            "beam_momentum_gev_c": float(
                raw_config.get("beam_momentum", 158.0)
            ),
            "sqrt_s_note": (
                "the M8 beam momentum 158 GeV/c gives "
                "sqrt(s) = 17.270 GeV (the plan section 9 "
                "'sqrt(s) = 17.2 GeV' reference)"
            ),
            "topology": (
                "the two-source (B, Q, S) = (1, 1, 0) topology of the "
                "frozen legacy builder (the R23 convention)"
            ),
            "source_mass_law": (
                "the legacy source-mass law of the M8 frozen set "
                "(round-3 index item 4; the Regge exchange parameters "
                "applied unchanged)"
            ),
        },
        "parameters": {
            "source": (
                "config/v2_singlet/legacy_v1_manifest.json, "
                "frozen_fit_outputs.m8_frozen_pp_parameters (verbatim)"
            ),
            "m8_parameter_hash": r01.M8_PARAMETER_HASH,
            "m8_frozen_pp_parameters": dict(r01.M8_FROZEN_PP_PARAMETERS),
            "kernel_boundary_gev": float(KERNEL_BOUNDARY),
            "kernel_boundary_note": (
                "the M8-frozen continuum.match_mass is the R23 kernel "
                "boundary; with a blend the per-source kernel switch at "
                "the boundary is replaced by the card-6 blend routing "
                "at the variant ramp start"
            ),
            "no_refit": (
                "no optimizer is invoked; no topology, exchange-mass, "
                "decay, or branching parameter is refit (plan section 24)"
            ),
        },
        "variants": [
            {
                "variant": int(entry["variant"]),
                "label": entry["label"],
                "ramp_start_gev": float(entry["ramp_start_gev"]),
                "ramp_width_gev": RAMP_WIDTH_GEV,
            }
            for entry in VARIANTS
        ],
        "variants_note": (
            "three variants of the card-6 blend (round-3 index item 5): "
            "the ramp width W = 0.5 GeV fixed, every other decay "
            "parameter (catalog, branching fractions, family "
            "parameters) identical; the only difference is the blend "
            "ramp start (plan section 9, line 646)"
        ),
        "statistics": {
            "tier": "T2",
            "label": T2_LABEL,
            "replicas_per_variant": T2_REPLICAS,
            "events_per_replica": T2_EVENTS_PER_REPLICA,
            "events_per_variant": T2_REPLICAS * T2_EVENTS_PER_REPLICA,
            "variants": N_VARIANTS,
            "total_events": TOTAL_EVENTS,
            "sizing_note": (
                "sized to the check set: the cross-variant spread "
                "metrics are evaluated on the pooled 8-replica "
                "per-variant statistics, and the 8-replica per-variant "
                "tier is the pre-registered contract"
            ),
        },
        "seed_map": {
            "master_seed": MASTER_SEED,
            "raw_seed_rule": RAW_SEED_RULE,
            "raw_replica_seeds": {
                str(entry["variant"]): [
                    raw_replica_seed(int(entry["variant"]), r)
                    for r in range(T2_REPLICAS)
                ]
                for entry in VARIANTS
            },
            "kernel_seed_rule": KERNEL_SEED_RULE,
            "legacy_event_seed_rule": LEGACY_EVENT_SEED_RULE,
            "substream_labels": ["decay_channel", "decay_kinematics"],
            "disjointness_note": DISJOINTNESS_NOTE,
        },
        "builder": {
            "module": BUILDER_REL,
            "pre_extension_r23_sha256": BUILDER_R23_PRE_EXTENSION_SHA256,
            "post_extension_sha256": _input_hashes()[
                "builder_module_sha256"
            ],
            "extension": (
                "the R23 ResonancePPBuilder extended additively with the "
                "blend parameter: blend = None keeps the R23 behavior "
                "byte-stable (asserted by the card-13 pins and by the "
                "R32 regression pin over the sealed R23 replica-0 "
                "stream's first 1,000 event hashes); a BlendSpec routes "
                "the source decay through the card-6 blend with the "
                "variant ramp start"
            ),
            "blend_surface": (
                "na49_mixing/spectral_density.py (card 6, committed, "
                "byte-unchanged): BlendSpec, route_source_decay, and the "
                "boundary-matching factor of the double-counting "
                "identity; the routed Monte Carlo never consults the "
                "factor (the routing weight is w(M) alone)"
            ),
        },
        "bit_identity_check": {
            "variant": 0,
            "seed": int(BIT_IDENTITY_SEED),
            "n_events": int(BIT_IDENTITY_EVENTS),
            "kernel_master_seed": int(BIT_IDENTITY_MASTER_SEED),
            "kernel_replica_index": int(BIT_IDENTITY_REPLICA_INDEX),
            "layers_checked": list(LAYER_DEFINITIONS),
            "layer_definitions": dict(LAYER_DEFINITIONS),
            "comparison_builders": (
                "the R32 v0 blend builder, the frozen R23 builder "
                "(blend = None), and the legacy staged PPEventBuilder; "
                "the three source layers are asserted identical across "
                "all three, event by event (the R23 convention)"
            ),
            "finals_layer": (
                "the above-boundary finals differ from R23 by design "
                "(the legacy H3 kernel is replaced by the continuum "
                "routing); that layer is recorded, not asserted; a "
                "source-layer mismatch stops the run before any "
                "replica is generated and re-opens the builder "
                "extension"
            ),
        },
        "comparisons": {
            "paper_window": {
                "window": dict(r01.PAPER_WINDOW),
                "sigma_inel_mb": float(binned.get("sigma_inel_mb", 31.78)),
                "species": dict(r01.SPECIES_BIN_COUNTS),
                "total_bins": r01.TOTAL_BINS,
                "binning": (
                    "the 1017-bin six-species paper-window binned "
                    "tables (the R01 convention; the pooled estimator "
                    "M_bin = sum_sums / (pi * sqrt(s) * N_total * dxF * "
                    "dpT)); the production comparison binning loads the "
                    "six-species SPS tables, the smoke mode a synthetic "
                    "hermetic grid from the committed table "
                    "declarations"
                ),
            },
            "identified_set": {
                "source": (
                    f"{R11_RUN_DIR}/ranked_residual_map.json (sealed), "
                    "the top-5 entries by |pull| (the R11 "
                    "ranked-residual cells, the R23 convention)"
                ),
                "sha256_r11_sums": r11_seal,
                "entries": [
                    {
                        "rank": int(row["rank"]),
                        "sector": str(row["sector"]),
                        "observable": str(row["observable"]),
                        "region": str(row["region"]),
                        "bin_index": int(row["bin_index"]),
                    }
                    for row in rows
                ],
            },
            "feed_down": {
                "lineage_convention": (
                    "the corrected lineage convention of Round 3 card 3 "
                    "(the degenerate infinity convention of R23 is not "
                    "used): a final particle is resonant feed-down when "
                    "its producing vertex's parent is a resonance other "
                    "than the drawn root - the catalog route's drawn "
                    "mixture record, the continuum route's root sentinel "
                    "0 (the continuum state is not a catalog resonance, "
                    "so the root-split daughters are direct source "
                    "products); the fractions cover every blend-routed "
                    "source's finals"
                ),
            },
        },
        "cross_variant_spread": {
            "per_bin": (
                "per bin, per cell, and per species: the min/max and the "
                "median absolute deviation of the three variants' "
                "pooled values"
            ),
            "bin_spread_bound": BIN_SPREAD_BOUND,
            "bin_spread_unit": (
                "the pooled MC standard error of the comparison: "
                "sqrt(SE_0^2 + SE_1^2 + SE_2^2) of the three variants' "
                "pooled per-bin MC errors (the R01 convention "
                "value * sqrt(1/N_bin) on the pooled counts); the "
                "median variant per bin is recorded as the reference "
                "with its own pooled MC standard error; the absolute "
                "bin-level cross-variant deviation is the maximum "
                "pairwise deviation (max - min) across variants"
            ),
            "bin_spread_calibration_note": (
                "under pure Monte-Carlo scatter the 95th percentile of "
                "the normalized three-variant range sits near 1.8, "
                "inside the 2.0 bound; a systematic model shift between "
                "variants pushes it above"
            ),
            "bin_spread_aggregate": (
                "the 95th percentile of the absolute bin-level "
                "cross-variant deviation over the 1017-bin check set "
                "is < 2.0 (in units of the pooled MC standard error of "
                "the comparison)"
            ),
            "feeddown_spread_bound": FEEDDOWN_SPREAD_BOUND,
            "feeddown_spread_unit": (
                "the combined SE of the three variants' per-species "
                "feed-down fractions: each variant's between-replica "
                "standard error (std, ddof=1, over sqrt(8), the "
                "R21/R22 convention), combined as "
                "sqrt(SE_0^2 + SE_1^2 + SE_2^2); the spread is the "
                "(max - min) across variants"
            ),
            "se_floor": SE_FLOOR,
            "se_floor_convention": (
                "the card-8a deterministic-SE floor: a pass row with "
                "the unit SE at or below the floor and sigma exactly "
                "0.0 is a vacuous-zero pass (recorded in the census, "
                "never treated as evidence of agreement)"
            ),
        },
        "named_residual_precedent": {
            "register_shape": (
                "the register shape of the sealed "
                "dev/00runs/r31_catalog_continuum_matching_scan_2026-09-17"
                "/gate3_r31.json: every failing verdict named (kind, "
                "species, bin/cell index, variant pair, observed "
                "spread, sigma, bound) with a one-line first-pass "
                "diagnosis (finite_mc_scatter vs systematic_candidate "
                "vs se_estimator_artifact)"
            ),
            "diagnosis_rule": (
                "a failing verdict with an infinite or sub-floor unit "
                "SE is an se_estimator_artifact; |sigma| <= "
                f"{FINITE_MC_SCATTER_MAX_SIGMA} is provisionally "
                "finite_mc_scatter (the confirmation-eligible class); a "
                "larger sigma is a systematic_candidate (the "
                "responsible card re-opens, no confirmation trigger)"
            ),
        },
        "robust_statistics_cross_reference": {
            "trigger": (
                "triggered only by finite-MC-scatter diagnostics of the "
                "named-residual register (acceptance (v))"
            ),
            "statistics": {
                "replicas_per_variant": CONFIRMATION_REPLICAS,
                "events_per_replica": T2_EVENTS_PER_REPLICA,
                "total_events": (
                    N_VARIANTS * CONFIRMATION_REPLICAS
                    * T2_EVENTS_PER_REPLICA
                ),
                "master_seed": MASTER_SEED,
            },
            "raw_seed_rule": CONFIRMATION_RAW_SEED_RULE,
            "kernel_replica_rule": CONFIRMATION_KERNEL_REPLICA_RULE,
            "raw_replica_seeds": {
                str(entry["variant"]): [
                    confirmation_raw_seed(int(entry["variant"]), r)
                    for r in range(CONFIRMATION_REPLICAS)
                ]
                for entry in VARIANTS
            },
            "execution": (
                "gateway, 48 jobs of 100,000 events, --tasks 1, single "
                "math thread (NA49_CLUSTER_GW_TASKS forces thread "
                "counts to 1), submission paced by the capacity check"
            ),
            "run_dir": f"dev/00runs/{CONFIRMATION_RUN_NAME}",
            "cross_reference_block": (
                "sealed in its own run dir and cross-referenced from "
                "gate3_r32.json in the card_8c_cross_reference block "
                "shape of the sealed R31 gate evidence; a named "
                "residual that passes at the confirmation statistics is "
                "classified finite-MC scatter, one that persists "
                "re-opens the responsible card (card 6 routing, the "
                "originating family card, or the builder extension)"
            ),
        },
        "conservation": {
            "statement": (
                "every p+p event conserves total B = 2 and Q = 2 "
                "exactly, and every source conserves its (B, Q, S) = "
                "(1, 1, 0) exactly (the Round 2 kernel conservation, "
                "checked per event, never repaired; a violation raises "
                "and stops the run)"
            ),
        },
        "routing_census": {
            "statement": (
                "per variant and replica, the fraction of sources "
                "routed catalog vs continuum family (recorded; v1/v2 "
                "shift the census, the shifts are recorded, not "
                "asserted)"
            ),
            "boundaries_gev": [
                float(entry["ramp_start_gev"]) for entry in VARIANTS
            ],
        },
        "no_fitting": True,
        "optimizer_invoked": False,
        "no_fitting_note": (
            "no optimizer is invoked (plan section 24); the spread is "
            "treated as model uncertainty in the first pass; no data "
            "selection among the variants (the Gate 3 decision records "
            "whether data select one option)"
        ),
        "gate_evidence_shape": {
            "file": GATE_RECORD_NAME,
            "gate": "gate 3 (plan section 9, line 650)",
            "clauses": ["smoothness", "no_boundary_dominance"],
            "smoothness_cross_references": [
                "dev/00runs/r31_catalog_continuum_matching_scan_2026-09-17"
                "/gate3_r31.json",
                "dev/00runs/r3_se_audit_2026-09-18/"
                "confirmation_verification.json (0/422, max 2.96)",
            ],
            "handoff_note": (
                "R32 delivers the final Round 3 evidence to the Gate 3 "
                "decision; the written proceed / revise / reject "
                "decision on the two Gate 3 clauses is the Gate 3 "
                "follow-up obligation of the round-3 index card, "
                "written after R32 lands (no Round 4 card may start "
                "before it)"
            ),
        },
        "input_hashes": _input_hashes(),
    }


def run_pre_registration(run_dir: Path) -> None:
    """Write ``pre_registration.json`` and ``provenance.json``, stop."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    pre_path = run_dir / PREREGISTRATION_NAME
    if pre_path.is_file() and any(
        run_dir.glob("variant_*_replica_*.json")
    ):
        _refuse(
            f"{pre_path} exists and variant records are present; the "
            "pre-registration is immutable after generation has started "
            "(a change invalidates the run; re-seal under a new date)"
        )
    substrate = _verify_frozen_substrate()
    catalog = _load_catalog()
    families = _load_families()
    pre = build_pre_registration(catalog, families)
    _write_json(pre_path, pre)
    hashes = _input_hashes()
    driver_path = Path(__file__).resolve()
    provenance = {
        "schema": "na49_mixing.staged_r32_spectral_systematic.provenance/1",
        "run": RUN_ID,
        "card": CARD,
        "card_chain": list(CARD_CHAIN),
        "driver": DRIVER_REL,
        "driver_sha256": {
            "nested": hashes["driver_sha256"],
            "outer": _sha256_file(
                MONO_ROOT / DRIVER_REL
            ) if (MONO_ROOT / DRIVER_REL).is_file() else None,
            "note": (
                "the dual-tree driver is byte-identical in both trees; "
                "the outer entry is the on-disk mirror's digest"
            ),
        },
        "engine": {
            "path": ENGINE_REL,
            "sha256": hashes["engine_module_sha256"],
            "role": (
                "the fixed card-2 engine (8a P1); R32 does not "
                "re-verify the threshold metric (that was R31/8a/8c) "
                "and consumes the card-6 routing"
            ),
        },
        "builder": {
            "path": BUILDER_REL,
            "pre_extension_r23_sha256": (
                BUILDER_R23_PRE_EXTENSION_SHA256
            ),
            "post_extension_sha256": hashes["builder_module_sha256"],
            "byte_stability_note": (
                "the pre-extension digest is the byte-unchanged R23 "
                "reference pinned by the sealed R23 replica records; "
                "the post-extension digest is the on-disk module with "
                "the additive blend parameter; blend = None keeps the "
                "R23 behavior byte-stable"
            ),
        },
        "frozen_drivers": {
            "r23_driver_sha256": hashes["r23_driver_sha256"],
            "r31_driver_sha256": hashes["r31_driver_sha256"],
            "r3_se_audit_driver_sha256": substrate[
                "r3_se_audit_driver_sha256"
            ],
        },
        "catalog": {
            "path": CATALOG_REL,
            "sha256": hashes["catalog_sha256"],
        },
        "continuum_config": {
            "path": FAMILIES_REL,
            "sha256": hashes["families_config_sha256"],
        },
        "m8_parameter_hash": r01.M8_PARAMETER_HASH,
        "kernel_boundary_gev": float(KERNEL_BOUNDARY),
        "blend": {
            "ramp_width_gev": RAMP_WIDTH_GEV,
            "ramp_starts_gev": [
                float(entry["ramp_start_gev"]) for entry in VARIANTS
            ],
        },
        "sealed_run_dirs": substrate["sealed_run_dirs"],
        "r11_ranked_residual_map": {
            "path": f"{R11_RUN_DIR}/ranked_residual_map.json",
            "sha256_sums": _r11_top5()[1],
        },
        "git": _git_info(),
        "environment": environment_fingerprint(),
        "written_utc": _now_utc(),
        "written_before_generation": True,
        "input_hashes": hashes,
    }
    _write_json(run_dir / PROVENANCE_NAME, provenance)
    print(
        "staged_r32_spectral_systematic: pre-registration written to "
        f"{pre_path} (3 variants x 8 replicas x 100,000 events = "
        f"{TOTAL_EVENTS}, master seed {MASTER_SEED}, ramp starts "
        f"{[float(e['ramp_start_gev']) for e in VARIANTS]} GeV, W = "
        f"{RAMP_WIDTH_GEV} GeV); nothing generated"
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
    stats = pre.get("statistics", {})
    if int(stats.get("replicas_per_variant", -1)) != T2_REPLICAS:
        problems.append("statistics.replicas_per_variant mismatch")
    if int(stats.get("events_per_replica", -1)) != T2_EVENTS_PER_REPLICA:
        problems.append("statistics.events_per_replica mismatch")
    if int(stats.get("variants", -1)) != N_VARIANTS:
        problems.append("statistics.variants mismatch")
    if int(stats.get("total_events", -1)) != TOTAL_EVENTS:
        problems.append("statistics.total_events mismatch")
    variants = pre.get("variants", [])
    if [
        (int(v.get("variant", -1)), float(v.get("ramp_start_gev", -1.0)))
        for v in variants
    ] != [
        (int(entry["variant"]), float(entry["ramp_start_gev"]))
        for entry in VARIANTS
    ]:
        problems.append("variant ramp starts mismatch")
    if any(
        float(v.get("ramp_width_gev", -1.0)) != RAMP_WIDTH_GEV
        for v in variants
    ):
        problems.append("variant ramp width mismatch (W = 0.5 GeV is "
                        "fixed)")
    seed_map = pre.get("seed_map", {})
    if int(seed_map.get("master_seed", -1)) != MASTER_SEED:
        problems.append("master seed mismatch")
    if str(seed_map.get("raw_seed_rule", "")) != RAW_SEED_RULE:
        problems.append("raw seed rule mismatch")
    if str(seed_map.get("kernel_seed_rule", "")) != KERNEL_SEED_RULE:
        problems.append("kernel seed rule mismatch")
    expected_seeds = {
        str(entry["variant"]): [
            raw_replica_seed(int(entry["variant"]), r)
            for r in range(T2_REPLICAS)
        ]
        for entry in VARIANTS
    }
    if seed_map.get("raw_replica_seeds") != expected_seeds:
        problems.append("raw replica seed blocks mismatch")
    bit = pre.get("bit_identity_check", {})
    if int(bit.get("seed", -1)) != BIT_IDENTITY_SEED:
        problems.append("bit_identity seed mismatch")
    if int(bit.get("n_events", -1)) != BIT_IDENTITY_EVENTS:
        problems.append("bit_identity n_events mismatch")
    spread = pre.get("cross_variant_spread", {})
    if float(spread.get("bin_spread_bound", -1.0)) != BIN_SPREAD_BOUND:
        problems.append("bin spread bound mismatch")
    if float(spread.get("feeddown_spread_bound", -1.0)) != (
        FEEDDOWN_SPREAD_BOUND
    ):
        problems.append("feed-down spread bound mismatch")
    confirmation = pre.get("robust_statistics_cross_reference", {})
    if int(confirmation.get("statistics", {}).get(
            "replicas_per_variant", -1)) != CONFIRMATION_REPLICAS:
        problems.append("confirmation replica count mismatch")
    if str(confirmation.get("raw_seed_rule", "")) != (
        CONFIRMATION_RAW_SEED_RULE
    ):
        problems.append("confirmation raw seed rule mismatch")
    if pre.get("no_fitting") is not True:
        problems.append("no_fitting mismatch")
    if pre.get("optimizer_invoked") is not False:
        problems.append("optimizer_invoked mismatch")
    entries = pre.get("comparisons", {}).get("identified_set", {}).get(
        "entries", []
    )
    rows, _ = _r11_top5()
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
            "staged_r32_spectral_systematic: the written pre-registration "
            "does not match this driver (the run is invalidated):\n  "
            + "\n  ".join(problems)
        )
    return pre


# ---------------------------------------------------------------------------
# Mode: --bit-identity (variant v0, the R23 seed conventions)
# ---------------------------------------------------------------------------

#: Module-level worker context, set once in the parent before the fork
#: pool is created (the R01/R23 ``_generate_chunk`` convention).  ``None``
#: in the parent main path.
_WORKER_CONTEXT: dict[str, Any] | None = None


def _set_worker_context(context: dict[str, Any]) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


def _event_layers(event) -> dict:
    """The three source-layer payloads of one event (the R23 layer
    conventions, exact float equality)."""
    return {
        "topology": [
            (s.source_id, s.role, s.quantum_numbers.as_tuple())
            for s in event.sources
        ],
        "exchange": [
            (float(s.metadata["sampled_mass"]), float(s.metadata["t"]))
            for s in event.sources
        ],
        "spectral_state": [
            tuple(s.p4.as_dict().values()) for s in event.sources
        ],
    }


def _finals_of_source(event, source_id: int) -> list[tuple]:
    """The final particles of one source as comparable tuples."""
    return [
        (
            int(p.pdg_id),
            tuple(p.p4.as_dict().values()),
            str(p.channel_family),
        )
        for p in event.final_particles
        if int(p.source_id) == int(source_id)
    ]


def _bit_identity_chunk(task: list):
    """One chunk of the bit-identity comparison: the legacy staged
    builder, the frozen R23 builder (blend = None), and the R32 v0
    blend builder at the same (seed, event_id); the three source layers
    compared across all three, the finals layers recorded."""
    assert _WORKER_CONTEXT is not None
    event_ids = task
    context = _WORKER_CONTEXT
    catalog = context["catalog"]
    pp_config = context["pp_config"]
    families = context["families"]
    blend = context["blend"]
    layers = {
        name: {"n_compared": 0, "n_matched": 0, "mismatches": []}
        for name in ("legacy_vs_builder", "builder_vs_v0")
    }
    for layer in layers.values():
        for key in LAYER_DEFINITIONS:
            layer[key] = {"n_compared": 0, "n_matched": 0, "mismatches": []}
    census = {
        "n_below_boundary_sources": 0,
        "n_in_band_sources": 0,
        "n_above_boundary_sources": 0,
        "v0_catalog_routed": 0,
        "v0_continuum_routed": 0,
        "below_boundary_finals_compared": 0,
        "below_boundary_finals_matched": 0,
        "below_boundary_finals_skipped_perturbed": 0,
        "legacy_finals_compared": 0,
        "legacy_finals_matched": 0,
        "legacy_finals_skipped_new_kernel": 0,
        "legacy_finals_skipped_perturbed": 0,
    }
    for event_id in event_ids:
        legacy_builder = PPEventBuilder(
            pp_config,
            np.random.default_rng(
                np.random.SeedSequence(BIT_IDENTITY_SEED,
                                       spawn_key=(event_id,))
            ),
        )
        legacy_event = legacy_builder.generate(
            event_id, stage="r23_resonance_ablation"
        )
        builder = ResonancePPBuilder(
            pp_config, catalog,
            seed=BIT_IDENTITY_SEED, event_id=event_id,
            master_seed=BIT_IDENTITY_MASTER_SEED,
            replica_index=BIT_IDENTITY_REPLICA_INDEX,
        )
        builder_result = builder.generate(stage=STAGE_LABEL)
        v0 = ResonancePPBuilder(
            pp_config, catalog,
            seed=BIT_IDENTITY_SEED, event_id=event_id,
            master_seed=BIT_IDENTITY_MASTER_SEED,
            replica_index=BIT_IDENTITY_REPLICA_INDEX,
            blend=blend, families=families,
        )
        v0_result = v0.generate(stage=STAGE_LABEL)
        legacy_layers = _event_layers(legacy_event)
        builder_layers = _event_layers(builder_result.event)
        v0_layers = _event_layers(v0_result.event)
        for key in LAYER_DEFINITIONS:
            for name, mine, theirs in (
                ("legacy_vs_builder", legacy_layers, builder_layers),
                ("builder_vs_v0", builder_layers, v0_layers),
            ):
                slot = layers[name][key]
                slot["n_compared"] += 1
                if mine[key] == theirs[key]:
                    slot["n_matched"] += 1
                else:
                    slot["mismatches"].append({"event": event_id})
        # The finals layers (recorded, not asserted), under the R23
        # perturbation-skip convention: a source's finals are comparable
        # only when every earlier source took the same-consumption
        # branch in both builders.  The below-boundary v0 catalog
        # finals are comparable against the frozen builder's while all
        # earlier sources lie below the blend boundary (the catalog
        # route consumes one spawned-record uniform there, exactly the
        # builder's catalog branch); the frozen builder's legacy finals
        # are comparable against the legacy builder's while all earlier
        # sources took the legacy branch (the R23
        # ``legacy_kernel_finals`` layer).
        comparable_v0 = True
        comparable_legacy = True
        for record, v0_record in zip(
            builder_result.source_records, v0_result.source_records
        ):
            mass = float(record.mass)
            below_v0 = mass < blend.boundary
            in_band = (not below_v0) and (
                mass < blend.boundary + blend.ramp_width
            )
            if below_v0:
                census["n_below_boundary_sources"] += 1
            elif in_band:
                census["n_in_band_sources"] += 1
            else:
                census["n_above_boundary_sources"] += 1
            if v0_record.route == "catalog":
                census["v0_catalog_routed"] += 1
            else:
                census["v0_continuum_routed"] += 1
            if below_v0 and v0_record.kernel == KERNEL_LABEL:
                if comparable_v0:
                    census["below_boundary_finals_compared"] += 1
                    if _finals_of_source(
                        v0_result.event, record.source_id
                    ) == _finals_of_source(
                        builder_result.event, record.source_id
                    ):
                        census["below_boundary_finals_matched"] += 1
                else:
                    census["below_boundary_finals_skipped_perturbed"] += 1
            comparable_v0 = comparable_v0 and below_v0
            if record.kernel == "legacy":
                if comparable_legacy:
                    census["legacy_finals_compared"] += 1
                    if _finals_of_source(
                        builder_result.event, record.source_id
                    ) == _finals_of_source(legacy_event, record.source_id):
                        census["legacy_finals_matched"] += 1
                else:
                    census["legacy_finals_skipped_perturbed"] += 1
            else:
                census["legacy_finals_skipped_new_kernel"] += 1
                comparable_legacy = False
    return event_ids[0], layers, census


def bit_identity_report(n_events: int, workers: int) -> dict:
    """The pre-registered bit-identity comparison payload (in-process;
    the CLI ``--bit-identity`` mode writes it to the run dir)."""
    if int(n_events) <= 0:
        _refuse("--bit-identity-events must be positive")
    catalog = _load_catalog()
    families = _load_families()
    pp_config = _m8_config()
    blend = variant_blend(0)
    event_ids = list(range(int(n_events)))
    chunks = [
        event_ids[start:start + CHUNK_SIZE]
        for start in range(0, len(event_ids), CHUNK_SIZE)
    ]
    _set_worker_context({
        "catalog": catalog,
        "pp_config": pp_config,
        "families": families,
        "blend": blend,
    })
    layers = {
        name: {
            key: {"n_compared": 0, "n_matched": 0, "mismatches": []}
            for key in list(LAYER_DEFINITIONS)
        }
        for name in ("legacy_vs_builder", "builder_vs_v0")
    }
    census = {
        "n_below_boundary_sources": 0,
        "n_in_band_sources": 0,
        "n_above_boundary_sources": 0,
        "v0_catalog_routed": 0,
        "v0_continuum_routed": 0,
        "below_boundary_finals_compared": 0,
        "below_boundary_finals_matched": 0,
        "below_boundary_finals_skipped_perturbed": 0,
        "legacy_finals_compared": 0,
        "legacy_finals_matched": 0,
        "legacy_finals_skipped_new_kernel": 0,
        "legacy_finals_skipped_perturbed": 0,
    }
    if int(workers) > 1 and len(chunks) > 1:
        context = multiprocessing.get_context("fork")
        with context.Pool(processes=int(workers)) as pool:
            parts = pool.map(_bit_identity_chunk, chunks, chunksize=1)
    else:
        parts = [_bit_identity_chunk(chunk) for chunk in chunks]
    for _start, part_layers, part_census in parts:
        for name in layers:
            for key in layers[name]:
                layers[name][key]["n_compared"] += int(
                    part_layers[name][key]["n_compared"]
                )
                layers[name][key]["n_matched"] += int(
                    part_layers[name][key]["n_matched"]
                )
                layers[name][key]["mismatches"].extend(
                    part_layers[name][key]["mismatches"]
                )
        for key, value in part_census.items():
            census[key] += int(value)
    pass_all = all(
        slot["n_matched"] == slot["n_compared"]
        for name in layers
        for slot in layers[name].values()
    )
    return {
        "schema": "na49_mixing.staged_r32_spectral_systematic.bit_identity/1",
        "run": RUN_ID,
        "card": CARD,
        "variant": 0,
        "blend": {
            "boundary_gev": float(blend.boundary),
            "ramp_width_gev": float(blend.ramp_width),
        },
        "seed": int(BIT_IDENTITY_SEED),
        "n_events": int(n_events),
        "kernel_master_seed": int(BIT_IDENTITY_MASTER_SEED),
        "kernel_replica_index": int(BIT_IDENTITY_REPLICA_INDEX),
        "layer_definitions": dict(LAYER_DEFINITIONS),
        "asserted_layers": {
            name: {
                key: {
                    "n_compared": int(slot["n_compared"]),
                    "n_matched": int(slot["n_matched"]),
                    "mismatches": list(slot["mismatches"]),
                }
                for key, slot in per_name.items()
            }
            for name, per_name in layers.items()
        },
        "assertion": (
            "the topology, exchange, and spectral_state layers are "
            "bit-identical across the legacy staged builder, the frozen "
            "R23 builder (blend = None), and the R32 v0 blend builder, "
            "event by event; a mismatch stops the run before any "
            "replica is generated"
        ),
        "finals_census": {
            **census,
            "note": (
                "recorded, not asserted: the above-boundary finals "
                "differ from R23 by design (the legacy H3 kernel is "
                "replaced by the continuum routing); the below-boundary "
                "v0 catalog finals and the frozen builder's legacy "
                "finals above the boundary are compared under the R23 "
                "perturbation-skip convention"
            ),
        },
        "pass": bool(pass_all),
    }


def run_bit_identity(run_dir: Path, n_events: int) -> dict:
    """The pre-registered 10,000-event bit-identity check, written to
    the run dir.  A layer mismatch stops the run (exit 1)."""
    run_dir = Path(run_dir)
    pre = _assert_pre_registration(run_dir)
    bit = pre.get("bit_identity_check", {})
    if int(bit.get("seed", -1)) != BIT_IDENTITY_SEED or int(
            bit.get("n_events", -1)) != BIT_IDENTITY_EVENTS:
        _refuse(
            "the written pre-registration's bit-identity block does not "
            "match this driver; a change after generation invalidates "
            "the run"
        )
    if int(n_events) != BIT_IDENTITY_EVENTS:
        _refuse(
            f"--bit-identity-events {n_events} does not match the "
            f"pre-registered {BIT_IDENTITY_EVENTS}; the production "
            "bit-identity check is fixed"
        )
    report = bit_identity_report(int(n_events), DEFAULT_WORKERS)
    _write_json(run_dir / BIT_IDENTITY_NAME, report)
    if not report["pass"]:
        print(
            "staged_r32_spectral_systematic: the bit-identity check "
            "FAILED; the run stops before any replica is generated "
            "(a mismatch re-opens the builder extension, not the "
            "tolerance)",
            flush=True,
        )
        raise SystemExit(1)
    print(
        "staged_r32_spectral_systematic: bit-identity check passed over "
        f"{n_events} events at seed {BIT_IDENTITY_SEED} "
        "(the topology / exchange / spectral_state layers bit-identical "
        "across the legacy staged builder, the frozen R23 builder, and "
        "the R32 v0 blend builder; the finals layers recorded)",
        flush=True,
    )
    return report


def _assert_bit_identity(run_dir: Path) -> dict:
    path = Path(run_dir) / BIT_IDENTITY_NAME
    if not path.is_file():
        _refuse(
            f"{path} is missing; the run requires the bit-identity "
            "check (run --bit-identity first; a mismatch stops the run "
            "before any replica is generated)"
        )
    report = _read_json(path)
    if not report.get("pass"):
        _refuse(
            f"{path} records pass = false; the run stops (a mismatch "
            "re-opens the builder extension, not the tolerance)"
        )
    return report


# ---------------------------------------------------------------------------
# Mode: --generate (the per-(variant, replica) chunked generation)
# ---------------------------------------------------------------------------


def _empty_routing() -> dict:
    return {
        "n_sources": 0,
        "n_catalog": 0,
        "n_continuum": 0,
        "mass_sum_catalog": 0.0,
        "mass_sum_continuum": 0.0,
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
    """One blend-routed source's finals into the feed-down accumulators.

    The corrected lineage convention of Round 3 card 3: a final
    particle is resonant feed-down when its producing vertex's parent
    is a resonance other than the drawn root - the catalog route's
    drawn mixture record (``record.record_pdg``), the continuum route's
    root sentinel 0 (the continuum state is not a catalog resonance).
    A final with ``parent_id is None`` is a direct source product (the
    stable-identity path), never a resonance feed-down.
    """
    root_pdg = (
        int(record.record_pdg)
        if record.route == "catalog"
        else CONTINUUM_ROOT_SENTINEL
    )
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
        if parent_pdg != root_pdg:
            slot["n_resonant"] += 1
            bucket = slot["by_parent"]
            bucket[str(parent_pdg)] = bucket.get(str(parent_pdg), 0) + 1


def _r32_generate_chunk(task: tuple):
    """Generate one chunk of events in a worker process and bin it.

    Returns ``(start_event_id, event_hashes, sums, counts, routing,
    feeddown, n_events)``.  ``sums``/``counts`` hold the per-species
    binned weight sums and track counts at the table bins (the R01
    ``_bin_track_sums`` convention); ``routing`` holds the blend
    routing census; ``feeddown`` holds the per-species resonant
    feed-down counts of the blend-routed events under the corrected
    lineage convention of card 3.  The builder's kernel replica index
    follows the mode: production draws at ``100*v + r``, smoke at the
    pre-registered smoke block ``5000 + 100*v + r`` (the stamped
    record field).
    """
    assert _WORKER_CONTEXT is not None
    variant, replica, seed, event_ids, blend_boundary = task
    context = _WORKER_CONTEXT
    catalog = context["catalog"]
    pp_config = context["pp_config"]
    tables = context["tables"]
    families = context["families"]
    kernel_replica = (
        smoke_kernel_replica_index(variant, replica)
        if context["smoke"] else kernel_replica_index(variant, replica)
    )
    blend = BlendSpec(boundary=float(blend_boundary),
                      ramp_width=RAMP_WIDTH_GEV)
    hashes: list[str] = []
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    routing = _empty_routing()
    feeddown = _empty_feeddown(context["species_keys"])
    for event_id in event_ids:
        builder = ResonancePPBuilder(
            pp_config,
            catalog,
            seed=int(seed),
            event_id=int(event_id),
            master_seed=MASTER_SEED,
            replica_index=kernel_replica,
            blend=blend,
            families=families,
        )
        result = builder.generate(stage=STAGE_LABEL)
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
            routing["n_sources"] += 1
            if record.route == "catalog":
                routing["n_catalog"] += 1
                routing["mass_sum_catalog"] += float(record.mass)
            else:
                routing["n_continuum"] += 1
                routing["mass_sum_continuum"] += float(record.mass)
            if vertices is None:
                continue
            _accumulate_feeddown(feeddown, catalog, event, record, vertices)
    return (
        int(event_ids[0]),
        hashes,
        sums,
        counts,
        routing,
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
            _r32_generate_chunk, tasks, chunksize=1
        ):
            completed += 1
            yield result
            if completed % 4 == 0 or completed == len(tasks):
                elapsed = time.perf_counter() - started
                print(
                    f"[r32] {progress_label}: chunk {completed}/"
                    f"{len(tasks)} ({elapsed:.1f}s)",
                    flush=True,
                )


def _aggregate_chunks(parts: list, species_keys: tuple):
    """Merge the chunk results into one replica's pooled state (exact
    integer counts; float sums in generation order)."""
    hashes = [
        h for _s, chunk_hashes, _su, _co, _r, _f, _n in parts
        for h in chunk_hashes
    ]
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    routing = _empty_routing()
    feeddown = _empty_feeddown(species_keys)
    for _start, _hs, part_sums, part_counts, part_routing, part_fd, _n in parts:
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
        for key in ("n_sources", "n_catalog", "n_continuum"):
            routing[key] += int(part_routing[key])
        routing["mass_sum_catalog"] += float(part_routing[
            "mass_sum_catalog"])
        routing["mass_sum_continuum"] += float(part_routing[
            "mass_sum_continuum"])
        feeddown["n_finals"] += int(part_fd["n_finals"])
        for key, slot in part_fd["per_species"].items():
            target = feeddown["per_species"][key]
            target["n"] += int(slot["n"])
            target["n_resonant"] += int(slot["n_resonant"])
            for pdg, cnt in slot["by_parent"].items():
                bucket = target["by_parent"]
                bucket[pdg] = bucket.get(pdg, 0) + int(cnt)
    return hashes, sums, counts, routing, feeddown


def _replica_record(
    variant: int,
    replica: int,
    seed: int,
    kernel_replica: int,
    n_events: int,
    hashes: list,
    sums: dict,
    counts: dict,
    routing: dict,
    feeddown: dict,
    workers: int,
    n_chunks: int,
    smoke: bool,
) -> dict:
    """Assemble one per-(variant, replica) record (the smoke records
    carry no wall-clock fields: two identical smoke invocations must
    produce byte-identical artifacts)."""
    n_sources = routing["n_sources"]
    n_catalog = routing["n_catalog"]
    n_continuum = routing["n_continuum"]
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
    blend = variant_blend(variant)
    record = {
        "schema": "na49_mixing.staged_r32_spectral_systematic.replica/1",
        "run": RUN_ID,
        "card": CARD,
        "variant": int(variant),
        "replica": int(replica),
        "seed": int(seed),
        "kernel_replica_index": int(kernel_replica),
        "mode": "smoke" if smoke else "production",
        "n_events": int(n_events),
        "chunk_size": CHUNK_SIZE,
        "chunks": int(n_chunks),
        "worker_count": int(workers),
        "stage_label": STAGE_LABEL,
        "master_seed": int(MASTER_SEED),
        "legacy_event_seed_rule": LEGACY_EVENT_SEED_RULE,
        "kernel_seed_rule": KERNEL_SEED_RULE,
        "event_hash_set_sha256": hashlib.sha256(
            payload_digest
        ).hexdigest(),
        "builder_module_sha256": _input_hashes()["builder_module_sha256"],
        "driver_sha256": _sha256_file(Path(__file__).resolve()),
        "catalog_sha256": _input_hashes()["catalog_sha256"],
        "kernel_boundary_gev": float(KERNEL_BOUNDARY),
        "blend": {
            "boundary_gev": float(blend.boundary),
            "ramp_width_gev": float(blend.ramp_width),
            "label": VARIANTS[int(variant)]["label"],
        },
        "routing": {
            "n_sources": int(n_sources),
            "n_catalog": int(n_catalog),
            "n_continuum": int(n_continuum),
            "fraction_catalog": (
                float(n_catalog) / float(n_sources)
                if n_sources > 0 else 0.0
            ),
            "fraction_continuum": (
                float(n_continuum) / float(n_sources)
                if n_sources > 0 else 0.0
            ),
            "mean_mass_catalog_gev": (
                float(routing["mass_sum_catalog"]) / float(n_catalog)
                if n_catalog > 0 else 0.0
            ),
            "mean_mass_continuum_gev": (
                float(routing["mass_sum_continuum"]) / float(n_continuum)
                if n_continuum > 0 else 0.0
            ),
            "census_note": (
                "the fraction of realized sources routed catalog vs "
                "continuum family by the card-6 blend routing at the "
                "variant ramp start (recorded; v1/v2 shift the census, "
                "the shifts are recorded, not asserted)"
            ),
        },
        "feed_down": {
            "n_finals_blend_routed": int(feeddown["n_finals"]),
            "fractions_resonant": {
                key: float(v) for key, v in fractions.items()
            },
            "per_species": per_species_payload,
            "lineage_convention": (
                "the corrected lineage convention of Round 3 card 3: a "
                "final particle is resonant feed-down when its "
                "producing vertex's parent is a resonance other than "
                "the drawn root (the catalog route's drawn record, the "
                "continuum route's root sentinel 0); the fractions "
                "cover every blend-routed source's finals"
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
        "sums": {s: [float(v) for v in values] for s, values in sums.items()},
        "counts": {
            s: [int(v) for v in values] for s, values in counts.items()
        },
        "reused": False,
    }
    if not smoke:
        record["wall_seconds"] = None
        record["started_utc"] = None
        record["finished_utc"] = None
    return record


def _generate_variant_replica(
    run_dir: Path,
    variant: int,
    replica: int,
    n_events: int,
    workers: int,
    context: dict,
    smoke: bool,
    record_name: str,
    raw_seed: int,
) -> dict:
    """One (variant, replica) sample: chunked generation, pooled per-bin
    sums, the routing census, and the feed-down lineage.

    An existing record that matches the event count, the mode, the
    builder identity, and the driver identity is reused unchanged, so
    the declared tier can be completed replica by replica (the
    pre-registered gateway fallback runs one job per replica)."""
    record_path = Path(run_dir) / record_name
    if record_path.is_file():
        try:
            stored = _read_json(record_path)
        except ValueError:
            stored = None
        if (
            isinstance(stored, dict)
            and int(stored.get("variant", -1)) == int(variant)
            and int(stored.get("replica", -1)) == int(replica)
            and int(stored.get("seed", -1)) == int(raw_seed)
            and int(stored.get("n_events", -1)) == int(n_events)
            and stored.get("mode") == ("smoke" if smoke else "production")
            and stored.get("builder_module_sha256")
            == _input_hashes()["builder_module_sha256"]
            and stored.get("driver_sha256")
            == _sha256_file(Path(__file__).resolve())
            and isinstance(stored.get("sums"), dict)
            and isinstance(stored.get("counts"), dict)
            and isinstance(stored.get("feed_down"), dict)
            and isinstance(stored.get("routing"), dict)
        ):
            stored["reused"] = True
            print(
                f"[r32] variant {variant} replica {replica}: reusing "
                f"completed record ({int(stored['n_events'])} events)",
                flush=True,
            )
            return stored
    species_keys = context["species_keys"]
    event_ids = list(range(int(n_events)))
    chunks = [
        event_ids[start:start + CHUNK_SIZE]
        for start in range(0, len(event_ids), CHUNK_SIZE)
    ]
    tasks = [
        (int(variant), int(replica), int(raw_seed), chunk,
         float(variant_blend(variant).boundary))
        for chunk in chunks
    ]
    _set_worker_context(context)
    started = time.perf_counter()
    started_iso = _now_utc()
    parts = sorted(
        _map_chunks(
            tasks, int(workers),
            f"variant {variant} replica {replica}",
        ),
        key=lambda part: part[0],
    )
    hashes, sums, counts, routing, feeddown = _aggregate_chunks(
        parts, species_keys
    )
    wall = time.perf_counter() - started
    record = _replica_record(
        variant, replica, raw_seed,
        smoke_kernel_replica_index(variant, replica) if smoke
        else kernel_replica_index(variant, replica),
        n_events, hashes, sums, counts,
        routing, feeddown, int(workers), len(chunks), smoke,
    )
    if not smoke:
        record["wall_seconds"] = round(wall, 3)
        record["started_utc"] = started_iso
        record["finished_utc"] = _now_utc()
    _write_json(record_path, record)
    return record


def _build_generation_context(tables: list, smoke: bool) -> dict:
    catalog = _load_catalog()
    species_keys = _stable_species_keys(catalog)
    return {
        "catalog": catalog,
        "pp_config": _m8_config(),
        "families": _load_families(),
        "tables": tables,
        "species_keys": species_keys,
        "smoke": bool(smoke),
    }


def _generation_tables(smoke: bool) -> list:
    """The binning tables of the generation step: the six-species SPS
    tables in production (the R01 loader), the synthetic hermetic grid
    in smoke mode."""
    if not smoke:
        return r01.load_binned_tables(_load_raw_config())
    return _smoke_tables()


def _smoke_tables() -> list:
    """The synthetic hermetic binning grid of the smoke mode: a uniform
    paper-window grid built from the committed table declarations
    (species, pdg, mass, normalization) only; no SPS table is read."""
    raw_config = _load_raw_config()
    binned = (raw_config.get("fit") or {}).get("binned") or {}
    if not binned or not binned.get("tables"):
        _refuse("config carries no fit.binned tables block; the smoke "
                "grid needs the committed table declarations")
    n_x = SMOKE_GRID_XF_BINS
    n_p = SMOKE_GRID_PT_BINS
    window = dict(binned.get("window") or {})
    xf_lo = float(window.get("xf_min", r01.PAPER_WINDOW["xF"][0]))
    xf_hi = float(window.get("xf_max", r01.PAPER_WINDOW["xF"][1]))
    pt_lo = float(window.get("pt_min", r01.PAPER_WINDOW["pT"][0]))
    pt_hi = float(window.get("pt_max", r01.PAPER_WINDOW["pT"][1]))
    xf_centers = np.linspace(xf_lo, xf_hi, n_x + 2)[1:-1]
    pt_centers = np.linspace(pt_lo, pt_hi, n_p + 2)[1:-1]
    tables: list[BinnedFitTable] = []
    for entry in binned["tables"]:
        xf_pts = np.tile(xf_centers, n_p)
        pt_pts = np.repeat(pt_centers, n_x)
        ones = np.ones(xf_pts.size, dtype=float)
        tables.append(BinnedFitTable.from_arrays(
            species=str(entry["species"]),
            pdg_id=int(entry["pdg"]),
            mass_gev=float(entry["mass_gev"]),
            normalization=str(entry["normalization"]),
            sigma_inel_mb=float(binned.get("sigma_inel_mb", 31.78)),
            path="smoke://synthetic-grid",
            xf=xf_pts,
            pt=pt_pts,
            value=ones,
            error=ones,
            window=window,
        ))
    return tables


def _assert_generation_identity(run_dir: Path, pre: dict) -> None:
    hashes = _input_hashes()
    if pre.get("input_hashes", {}).get("driver_sha256") != hashes[
            "driver_sha256"]:
        _refuse(
            "the committed driver sha256 differs from the "
            "pre-registered driver_sha256; the run is invalidated "
            "(re-register under a new date)"
        )
    if pre.get("input_hashes", {}).get("catalog_sha256") != hashes[
            "catalog_sha256"]:
        _refuse(
            "the committed catalog sha256 differs from the "
            "pre-registered catalog_sha256; the run is invalidated"
        )
    if hashes["builder_module_sha256"] != pre.get("builder", {}).get(
            "post_extension_sha256"):
        _refuse(
            "the builder module sha256 differs from the pre-registered "
            "post_extension_sha256; the run is invalidated"
        )


def run_generation(
    run_dir: Path,
    variant: int | None,
    replica: int | None,
    workers: int,
    statistics: dict,
) -> None:
    """Generate the requested slice: one (variant, replica), one whole
    variant, or every variant; each record lands on disk before the
    next.  Requires the pre-registration and a passing bit-identity
    check; the statistics flags must match the pre-registration."""
    run_dir = Path(run_dir)
    pre = _assert_pre_registration(run_dir)
    _assert_bit_identity(run_dir)
    _assert_generation_identity(run_dir, pre)
    declared = {
        "variants": N_VARIANTS,
        "replicas": T2_REPLICAS,
        "events_per_replica": T2_EVENTS_PER_REPLICA,
        "master_seed": MASTER_SEED,
    }
    for key, value in declared.items():
        given = statistics.get(key)
        if given is not None and int(given) != int(value):
            _refuse(
                f"--{key} {given} does not match the pre-registered "
                f"{value}; the production statistics are fixed"
            )
    if variant is None:
        variants = list(range(N_VARIANTS))
    else:
        if not 0 <= int(variant) < N_VARIANTS:
            _refuse(
                f"--variant {variant} outside the pre-registered set "
                f"0..{N_VARIANTS - 1}"
            )
        variants = [int(variant)]
    if replica is None:
        replicas = list(range(T2_REPLICAS))
    else:
        if not 0 <= int(replica) < T2_REPLICAS:
            _refuse(
                f"--replica {replica} outside the pre-registered set "
                f"0..{T2_REPLICAS - 1}"
            )
        replicas = [int(replica)]
    tables = _generation_tables(smoke=False)
    context = _build_generation_context(tables, smoke=False)
    for v in variants:
        for r in replicas:
            record = _generate_variant_replica(
                run_dir, v, r, T2_EVENTS_PER_REPLICA, int(workers),
                context, False,
                _variant_record_name(v, r), raw_replica_seed(v, r),
            )
            routing = record["routing"]
            print(
                f"[r32] variant {v} replica {r}: {record['n_events']} "
                f"events, routing catalog {routing['n_catalog']} / "
                f"continuum {routing['n_continuum']} "
                f"(fraction catalog {routing['fraction_catalog']:.6f}, "
                f"ramp start {record['blend']['boundary_gev']!r} GeV)",
                flush=True,
            )
    print(
        f"[r32] generation step complete (run dir {run_dir}, mode "
        "production)",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Mode: --smoke (the hermetic end-to-end driver validation)
# ---------------------------------------------------------------------------


def run_smoke(run_dir: Path, workers: int) -> None:
    """The hermetic smoke validation: a smoke pre-registration, 2
    variants x 1 replica x 2,000 events at the pre-registered smoke
    seeds on the synthetic hermetic binning grid, and the smoke spread
    evaluation.  Two identical smoke invocations produce byte-identical
    artifacts (no wall-clock fields in the smoke records)."""
    run_dir = Path(run_dir)
    if Path(run_dir).resolve() == (
        (MONO_ROOT / DEFAULT_RUN_DIR).resolve()
    ):
        _refuse(
            "--smoke refuses the default production run dir "
            f"({DEFAULT_RUN_DIR}); pass an explicit --run-dir for the "
            "smoke artifacts (they are never sealed)"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    catalog = _load_catalog()
    families = _load_families()
    pre = build_pre_registration(catalog, families)
    pre["mode"] = "smoke"
    pre["smoke"] = {
        "variants": list(SMOKE_VARIANTS),
        "replicas": SMOKE_REPLICAS,
        "events_per_replica": SMOKE_EVENTS,
        "raw_seed_rule": SMOKE_RAW_SEED_RULE,
        "kernel_replica_rule": SMOKE_KERNEL_REPLICA_RULE,
        "raw_replica_seeds": {
            str(v): [smoke_raw_seed(v, r) for r in range(SMOKE_REPLICAS)]
            for v in SMOKE_VARIANTS
        },
        "binning": (
            "the synthetic hermetic grid built from the committed table "
            "declarations (uniform "
            f"{SMOKE_GRID_XF_BINS} x {SMOKE_GRID_PT_BINS} paper-window "
            "grid per species; no SPS table is read)"
        ),
        "determinism_note": (
            "the smoke records carry no wall-clock fields; two "
            "identical smoke invocations produce byte-identical "
            "artifacts"
        ),
    }
    _write_json(run_dir / f"{SMOKE_PREFIX}pre_registration.json", pre)
    tables = _smoke_tables()
    context = _build_generation_context(tables, smoke=True)
    for v in SMOKE_VARIANTS:
        for r in range(SMOKE_REPLICAS):
            record = _generate_variant_replica(
                run_dir, v, r, SMOKE_EVENTS, int(workers), context, True,
                f"{SMOKE_PREFIX}{_variant_record_name(v, r)}",
                smoke_raw_seed(v, r),
            )
            routing = record["routing"]
            print(
                f"[r32-smoke] variant {v} replica {r}: "
                f"{record['n_events']} events, routing catalog "
                f"{routing['n_catalog']} / continuum "
                f"{routing['n_continuum']}",
                flush=True,
            )
    report = run_spread_evaluation(run_dir, smoke=True)
    print(
        "[r32-smoke] smoke validation complete (bin p95 "
        f"{report['summary']['bin_spread_p95']}, feed-down max sigma "
        f"{report['summary']['feeddown_spread_max_sigma']}; smoke "
        "artifacts are never sealed)",
        flush=True,
    )


def _refuse_smoke_artifacts(run_dir: Path) -> None:
    """The R31/8c convention: the gate / seal / clean-validate modes
    refuse smoke artifacts."""
    run_dir = Path(run_dir)
    smoke_files = sorted(
        p.name for p in run_dir.iterdir()
        if p.is_file() and p.name.startswith(SMOKE_PREFIX)
    )
    if smoke_files:
        _refuse(
            f"{run_dir} carries smoke artifacts ({', '.join(smoke_files)}"
            "); the gate / seal / clean-validate modes refuse smoke "
            "artifacts (the R31/8c convention)"
        )
    for path in sorted(run_dir.glob("variant_*_replica_*.json")):
        try:
            record = _read_json(path)
        except ValueError:
            continue
        if record.get("mode") == "smoke":
            _refuse(
                f"{path} is a smoke record; the gate / seal / "
                "clean-validate modes refuse smoke artifacts"
            )


# ---------------------------------------------------------------------------
# Mode: --compare (the cross-variant spread evaluation)
# ---------------------------------------------------------------------------


def _median_variant(values: list) -> int:
    """The index of the median variant value (ties broken by the lowest
    index; deterministic)."""
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    return int(order[(len(order) - 1) // 2])


def _pooled_se(values: list) -> float:
    """The pooled MC standard error of the comparison over the
    variants' finite per-variant standard errors."""
    finite = [float(se) for se in values if math.isfinite(float(se))]
    if len(finite) < len(values):
        return math.inf
    return math.sqrt(math.fsum(se * se for se in finite))


def _between_replica_se(fractions: list) -> float:
    """The between-replica standard error of one variant's per-species
    feed-down fraction: the sample standard deviation (ddof=1) of the
    stored replica fractions over sqrt(k) (the R21/R22 convention)."""
    values = [float(v) for v in fractions]
    n = len(values)
    if n == 0:
        return 0.0
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(max(variance, 0.0) / n)


def _load_variant_records(
    run_dir: Path,
    smoke: bool,
    n_replicas: int,
    n_events_per_replica: int,
) -> dict:
    """The per-variant replica records, each asserted against the
    pre-registered identity."""
    prefix = SMOKE_PREFIX if smoke else ""
    expected_replicas = SMOKE_REPLICAS if smoke else T2_REPLICAS
    if int(n_replicas) != expected_replicas:
        _refuse("replica count mismatch in the record loader")
    del n_replicas
    variant_entries: list[dict] = (
        [{"variant": v, "ramp_start_gev": float(variant_blend(v).boundary)}
         for v in SMOKE_VARIANTS] if smoke
        else [dict(entry) for entry in VARIANTS]
    )
    variants: dict[int, list[dict]] = {}
    for entry in variant_entries:
        v = int(entry["variant"])
        records = []
        for r in range(int(expected_replicas)):
            name = f"{prefix}{_variant_record_name(v, r)}"
            path = Path(run_dir) / name
            if not path.is_file():
                _refuse(
                    f"variant record missing: {path}; --compare requires "
                    f"all {N_VARIANTS} variants x {expected_replicas} "
                    "replicas"
                )
            record = _read_json(path)
            if record.get("mode") != ("smoke" if smoke else "production"):
                _refuse(f"{path} is not a "
                        f"{'smoke' if smoke else 'production'} record")
            if int(record.get("variant", -1)) != v or int(
                    record.get("replica", -1)) != r:
                _refuse(f"{path} carries the wrong variant/replica "
                        "identity")
            if int(record.get("n_events", -1)) != int(
                    n_events_per_replica):
                _refuse(f"{path} carries the wrong event count")
            if int(record.get("master_seed", -1)) != MASTER_SEED:
                _refuse(f"{path} carries the wrong master seed")
            expected_seed = (
                smoke_raw_seed(v, r) if smoke else raw_replica_seed(v, r)
            )
            if int(record.get("seed", -1)) != expected_seed:
                _refuse(f"{path} carries the wrong raw replica seed")
            if int(record.get("kernel_replica_index", -1)) != (
                smoke_kernel_replica_index(v, r) if smoke
                else kernel_replica_index(v, r)
            ):
                _refuse(f"{path} carries the wrong kernel replica index")
            blend = record.get("blend", {})
            if float(blend.get("ramp_width_gev", -1.0)) != RAMP_WIDTH_GEV:
                _refuse(f"{path} carries the wrong blend ramp width")
            if float(blend.get("boundary_gev", -1.0)) != float(
                    entry["ramp_start_gev"]):
                _refuse(f"{path} carries the wrong blend ramp start")
            if record.get("builder_module_sha256") != _input_hashes()[
                    "builder_module_sha256"]:
                _refuse(f"{path} carries the wrong builder identity")
            if record.get("driver_sha256") != _sha256_file(
                    Path(__file__).resolve()):
                _refuse(f"{path} carries the wrong driver identity")
            records.append(record)
        variants[v] = records
    return variants


def _pooled_bin_payload(
    tables: list, sums: dict, counts: dict, n_events_total: int,
    sqrt_s: float,
) -> tuple[list, dict, dict]:
    """The per-species binned predictions without the fixed 1017-bin
    total check (the smoke grid's reduced total; the R01 conventions
    verbatim).  The production path uses ``r01.bin_predictions_payload``
    (which asserts the 1017-bin total)."""
    if int(n_events_total) <= 0:
        raise ValueError("the pooled prediction needs a positive event count")
    entries: list[dict] = []
    values_by_species: dict[str, np.ndarray] = {}
    errors_by_species: dict[str, np.ndarray] = {}
    for table in tables:
        species = table.species
        xedges = sps_core.edges_from_centers(table.xf_centers)
        pedges = sps_core.edges_from_centers(table.pt_centers)
        n_x = xedges.size - 1
        dx = np.diff(xedges)
        dp = np.diff(pedges)
        area = (dp[:, None] * dx[None, :]).ravel()
        pick = table.bin_index_pt * n_x + table.bin_index_xf
        area_sel = area[pick]
        m = np.asarray(sums[species], dtype=float) / (
            math.pi * float(sqrt_s) * float(n_events_total) * area_sel
        )
        n_bin = np.asarray(counts[species], dtype=np.int64)
        mc_err = r01.mc_error_array(m, n_bin)
        values_by_species[species] = m
        errors_by_species[species] = mc_err
        for i in range(table.data.size):
            entries.append(
                {
                    "species": species,
                    "bin_index": int(i),
                    "xF_mid": float(table.xf_centers[table.bin_index_xf[i]]),
                    "pT_mid": float(table.pt_centers[table.bin_index_pt[i]]),
                    "value": float(m[i]),
                    "stat_error": float(mc_err[i]),
                }
            )
    return entries, values_by_species, errors_by_species


def _pooled_variant_stats(
    records: list[dict], tables: list, sqrt_s: float, smoke: bool,
) -> dict:
    """One variant's pooled statistics: the 1017-bin (or smoke-grid)
    paper-window values and pooled MC errors (the R01 conventions), the
    per-species feed-down fractions with their between-replica standard
    errors, and the pooled routing census."""
    n_events_total = sum(int(record["n_events"]) for record in records)
    if n_events_total <= 0:
        _refuse("the pooled prediction needs a positive event count")
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    for record in records:
        for species, values in record["sums"].items():
            arr = np.asarray(values, dtype=float)
            sums[species] = (
                arr if species not in sums else sums[species] + arr
            )
        for species, values in record["counts"].items():
            arr = np.asarray(values, dtype=np.int64)
            counts[species] = (
                arr if species not in counts else counts[species] + arr
            )
    entries, values_by_species, errors_by_species = (
        _pooled_bin_payload(
            tables, sums, counts, n_events_total, sqrt_s
        )
        if smoke else r01.bin_predictions_payload(
            tables, sums, counts, n_events_total, sqrt_s
        )
    )
    feeddown_fractions: dict[str, float] = {}
    feeddown_pooled: dict[str, dict] = {}
    pooled_n: dict[str, int] = {}
    pooled_n_res: dict[str, int] = {}
    replica_fractions: dict[str, list[float]] = {}
    for key in sorted({k for record in records
                       for k in record["feed_down"]["per_species"]}):
        replica_fractions[key] = [
            float(
                record["feed_down"]["per_species"].get(key, {}).get(
                    "fraction_resonant", 0.0)
            )
            for record in records
        ]
        pooled_n[key] = sum(
            int(record["feed_down"]["per_species"].get(key, {}).get("n", 0))
            for record in records
        )
        pooled_n_res[key] = sum(
            int(record["feed_down"]["per_species"].get(key, {}).get(
                "n_resonant", 0))
            for record in records
        )
        fraction = (
            float(pooled_n_res[key]) / float(pooled_n[key])
            if pooled_n[key] > 0 else 0.0
        )
        feeddown_fractions[key] = fraction
        feeddown_pooled[key] = {
            "n_finals": int(pooled_n[key]),
            "n_resonant": int(pooled_n_res[key]),
            "fraction_resonant": fraction,
            "between_replica_se": _between_replica_se(
                replica_fractions[key]
            ),
        }
    routing: dict[str, Any] = {
        "n_sources": sum(
            int(record["routing"]["n_sources"]) for record in records
        ),
        "n_catalog": sum(
            int(record["routing"]["n_catalog"]) for record in records
        ),
        "n_continuum": sum(
            int(record["routing"]["n_continuum"]) for record in records
        ),
    }
    routing["fraction_catalog"] = (
        float(routing["n_catalog"]) / float(routing["n_sources"])
        if routing["n_sources"] > 0 else 0.0
    )
    routing["fraction_continuum"] = (
        float(routing["n_continuum"]) / float(routing["n_sources"])
        if routing["n_sources"] > 0 else 0.0
    )
    return {
        "n_events": int(n_events_total),
        "entries": entries,
        "values_by_species": values_by_species,
        "errors_by_species": errors_by_species,
        "feeddown": feeddown_pooled,
        "routing": routing,
    }


def _cell_pooled_values(
    stats: dict, tables: list, pre: dict, check_regions: bool,
) -> list[dict]:
    """The five-cell identified set values of one variant (the R23
    cell convention: the aligned common cells, the R11
    ``combination_by_name`` evaluation, the pair-variance MC noise)."""
    tables_by_species = {t.species: t for t in tables}
    rows = []
    for entry in pre["comparisons"]["identified_set"]["entries"]:
        combo = r23drv.combination_by_name(str(entry["observable"]))
        sector = str(entry["sector"])
        k = int(entry["bin_index"])
        if len(combo.species) == 1:
            sp = r23drv._config_species(combo.species[0])
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
            ia = ib = k
            xF_mid = float(table.xf_centers[table.bin_index_xf[k]])
            m = float(stats["values_by_species"][sp][k])
            sm = float(stats["errors_by_species"][sp][k])
            v_mc = sm * sm if math.isfinite(sm) else math.inf
        else:
            sp_a = r23drv._config_species(combo.species[0])
            sp_b = r23drv._config_species(combo.species[1])
            if sector != f"{sp_a}+{sp_b}":
                _refuse(
                    f"identified cell rank {entry['rank']}: two-species "
                    f"sector {sector!r} != {sp_a}+{sp_b} (the R11 sector "
                    "naming)"
                )
            table_a = tables_by_species[sp_a]
            table_b = tables_by_species[sp_b]
            common = r23drv._aligned_common_cells(table_a, table_b)
            if k >= len(common):
                _refuse(
                    f"identified cell rank {entry['rank']}: bin index "
                    f"{k} outside the {len(common)} common cells"
                )
            ia, ib = common[k]
            xF_mid = float(
                table_a.xf_centers[table_a.bin_index_xf[ia]]
            )
            m = r23drv._cell_value(combo, {
                combo.species[0]: (
                    float(stats["values_by_species"][sp_a][ia]),
                ),
                combo.species[1]: (
                    float(stats["values_by_species"][sp_b][ib]),
                ),
            }, f"identified cell rank {entry['rank']} pooled value")
            sm_a = float(stats["errors_by_species"][sp_a][ia])
            sm_b = float(stats["errors_by_species"][sp_b][ib])
            v_mc = r23drv._pair_variance(sm_a, sm_b)
        if check_regions and r11.xf_sector_region(xF_mid) != str(
                entry["region"]):
            _refuse(
                f"identified cell rank {entry['rank']}: region "
                f"{entry['region']!r} does not match the bin midpoint "
                f"xF {xF_mid!r}"
            )
        rows.append({
            "rank": int(entry["rank"]),
            "sector": sector,
            "observable": str(entry["observable"]),
            "region": str(entry["region"]),
            "bin_index": k,
            "xF_mid": float(xF_mid),
            "value": float(m),
            "stat_error": (
                float(math.sqrt(v_mc)) if math.isfinite(v_mc)
                else math.inf
            ),
        })
    return rows


def _diagnose(sigma: float, unit_se: float) -> tuple[str, str]:
    """The pre-registered first-pass diagnosis of a failing verdict."""
    if not math.isfinite(float(unit_se)) or float(unit_se) <= SE_FLOOR:
        return (
            "se_estimator_artifact",
            "the unit standard error sits at or below the "
            f"DETERMINISTIC_SE_FLOOR {SE_FLOOR!r} (an empty-bin or "
            "degenerate-SE artifact; the sigma is a division artifact, "
            "not a physics deviation)",
        )
    if abs(float(sigma)) <= FINITE_MC_SCATTER_MAX_SIGMA:
        return (
            "finite_mc_scatter",
            "the failing sigma is inside the pre-registered finite-MC "
            f"envelope (|sigma| <= {FINITE_MC_SCATTER_MAX_SIGMA}); "
            "confirmation-eligible under acceptance (v)",
        )
    return (
        "systematic_candidate",
        "the failing sigma exceeds the finite-MC envelope; a "
        "systematic model shift between variants is indicated (the "
        "responsible card re-opens, no confirmation trigger)",
    )


def _named_residual_entry(index: int, kind: str, row: dict) -> dict:
    """One named-residual register entry (the sealed gate3_r31.json
    register shape, adapted to the R32 verdict fields)."""
    variant_values = row.get("values", [])
    max_i = int(row.get("argmax_variant", 0))
    min_i = int(row.get("argmin_variant", 0))
    diagnosis, note = _diagnose(row["sigma"], row["unit_se"])
    return {
        "index": int(index),
        "kind": kind,
        "species": str(row["species"]),
        "bin_index": (
            int(row["bin_index"]) if row.get("bin_index") is not None
            else None
        ),
        "cell_rank": (
            int(row["rank"]) if row.get("rank") is not None else None
        ),
        "variant_pair": [int(min_i), int(max_i)],
        "observed_spread": float(row["spread"]),
        "sigma": float(row["sigma"]),
        "bound": float(row["bound"]),
        "unit_se": float(row["unit_se"]),
        "values": [float(v) for v in variant_values],
        "classification": diagnosis,
        "diagnosis": note,
    }


def _spread_rows(
    per_variant_values: dict[int, list[float]],
    per_variant_ses: dict[int, list[float]],
    identity_rows: list[dict],
    bound: float,
    row_kind: str,
) -> tuple[list[dict], dict]:
    """The cross-variant spread verdict rows for one check family
    (bins or cells): per row, the min/max and the median absolute
    deviation across variants, the deviation in units of the pooled MC
    standard error, and the verdict at the bound."""
    rows: list[dict] = []
    n_fail = 0
    n_vacuous = 0
    for position, identity in enumerate(identity_rows):
        values = [
            float(per_variant_values[v][position])
            for v in sorted(per_variant_values)
        ]
        ses = [
            float(per_variant_ses[v][position])
            for v in sorted(per_variant_values)
        ]
        spread = max(values) - min(values)
        median_i = _median_variant(values)
        unit_se = _pooled_se(ses)
        mad = float(np.median([
            abs(v - values[median_i]) for v in values
        ]))
        # The informational literal reading of the card's unit phrase
        # ("the pooled MC standard error of the median variant"): the
        # spread in units of the median variant's own pooled MC
        # standard error.  The pre-registered bound verdicts are
        # evaluated on the pooled comparison unit above; this field is
        # recorded so the P2 evidence is reviewable under both
        # readings.
        median_se = ses[median_i]
        if math.isfinite(median_se) and median_se > SE_FLOOR:
            sigma_median_unit = float(spread) / float(median_se)
        elif spread == 0.0:
            sigma_median_unit = 0.0
        else:
            sigma_median_unit = math.inf
        row = dict(identity)
        row.update({
            "values": values,
            "stat_errors": ses,
            "median_variant": median_i,
            "median_variant_value": values[median_i],
            "median_variant_stat_error": ses[median_i],
            "min_value": min(values),
            "max_value": max(values),
            "argmin_variant": int(values.index(min(values))),
            "argmax_variant": int(values.index(max(values))),
            "median_absolute_deviation": mad,
            "spread": float(spread),
            "unit_se": (
                float(unit_se) if math.isfinite(unit_se) else math.inf
            ),
            "sigma_median_unit_informational": sigma_median_unit,
            "bound": float(bound),
        })
        if math.isinf(unit_se):
            row["sigma"] = 0.0
            row["status"] = "vacuous_empty_bin"
            row["pass"] = None
            n_vacuous += 1
        elif unit_se <= SE_FLOOR:
            if spread == 0.0:
                row["sigma"] = 0.0
                row["status"] = "vacuous_zero"
                row["pass"] = None
                n_vacuous += 1
            else:
                row["sigma"] = math.inf
                row["status"] = "fail"
                row["pass"] = False
                n_fail += 1
        else:
            sigma = float(spread) / float(unit_se)
            row["sigma"] = sigma
            row["status"] = "pass" if sigma < bound else "fail"
            row["pass"] = bool(sigma < bound)
            if not row["pass"]:
                n_fail += 1
        rows.append(row)
    census = {
        "rows": len(rows),
        "pass": sum(1 for r in rows if r["status"] == "pass"),
        "fail": n_fail,
        "vacuous": n_vacuous,
    }
    return rows, census


def run_spread_evaluation(run_dir: Path, smoke: bool = False) -> dict:
    """The pre-registered spread evaluation: the per-variant pooled
    comparisons (the paper-window bins, the five identified cells, the
    per-species feed-down fractions) and the cross-variant spread
    metrics with their pre-registered bounds."""
    run_dir = Path(run_dir)
    prefix = SMOKE_PREFIX if smoke else ""
    pre_name = f"{prefix}{PREREGISTRATION_NAME}"
    pre_path = run_dir / pre_name
    if not pre_path.is_file():
        _refuse(
            f"{pre_path} is missing; the spread evaluation requires the "
            f"{'smoke ' if smoke else ''}pre-registration"
        )
    pre = _read_json(pre_path)
    if not smoke:
        _assert_pre_registration(run_dir)
        _assert_generation_identity(run_dir, pre)
    n_replicas = SMOKE_REPLICAS if smoke else T2_REPLICAS
    n_events = SMOKE_EVENTS if smoke else T2_EVENTS_PER_REPLICA
    variants = _load_variant_records(
        run_dir, smoke, n_replicas, n_events
    )
    del n_replicas
    if smoke:
        tables = _smoke_tables()
    else:
        tables = r01.load_binned_tables(_load_raw_config())
    sqrt_s = _sqrt_s()
    stats = {}
    for v in sorted(variants):
        stats[v] = _pooled_variant_stats(
            variants[v], tables, sqrt_s, smoke=smoke
        )

    # -- the bin-level spread (the 1017-bin six-species check set) ------
    values_by_species = {
        v: stats[v]["values_by_species"] for v in sorted(stats)
    }
    errors_by_species = {
        v: stats[v]["errors_by_species"] for v in sorted(stats)
    }
    species_order = [t.species for t in tables]
    tables_by_species = {t.species: t for t in tables}
    bin_identity = []
    bin_values: dict[int, list[float]] = {v: [] for v in sorted(stats)}
    bin_ses: dict[int, list[float]] = {v: [] for v in sorted(stats)}
    for species in species_order:
        table = tables_by_species[species]
        for i in range(int(table.data.size)):
            bin_identity.append({
                "species": species,
                "bin_index": int(i),
                "xF_mid": float(table.xf_centers[table.bin_index_xf[i]]),
                "pT_mid": float(table.pt_centers[table.bin_index_pt[i]]),
            })
            for v in sorted(stats):
                bin_values[v].append(
                    float(values_by_species[v][species][i])
                )
                bin_ses[v].append(float(errors_by_species[v][species][i]))
    bin_rows, bin_census = _spread_rows(
        bin_values, bin_ses, bin_identity, BIN_SPREAD_BOUND, "bin"
    )
    finite_sigmas = [
        float(r["sigma"]) for r in bin_rows
        if r["status"] in ("pass", "fail")
    ]
    bin_p95 = float(np.percentile(np.asarray(finite_sigmas), 95.0)) \
        if finite_sigmas else 0.0
    finite_median_unit = [
        float(r["sigma_median_unit_informational"]) for r in bin_rows
        if r["status"] in ("pass", "fail")
        and math.isfinite(
            float(r["sigma_median_unit_informational"])
        )
    ]
    bin_p95_median_unit = (
        float(np.percentile(np.asarray(finite_median_unit), 95.0))
        if finite_median_unit else 0.0
    )
    bin_p95_per_species = {}
    for species in species_order:
        sig = [
            float(r["sigma"]) for r in bin_rows
            if r["species"] == species and r["status"] in ("pass", "fail")
        ]
        bin_p95_per_species[species] = (
            float(np.percentile(np.asarray(sig), 95.0)) if sig else 0.0
        )
    bin_pass = bool(bin_p95 < BIN_SPREAD_BOUND)

    # -- the cell-level spread (the five identified cells) ---------------
    cell_rows_per_variant = {
        v: _cell_pooled_values(
            stats[v], tables, pre, check_regions=not smoke
        )
        for v in sorted(stats)
    }
    cell_identity = [
        {
            "rank": int(row["rank"]),
            "sector": row["sector"],
            "observable": row["observable"],
            "region": row["region"],
            "bin_index": int(row["bin_index"]),
            "xF_mid": row["xF_mid"],
        }
        for row in cell_rows_per_variant[sorted(stats)[0]]
    ]
    cell_values = {
        v: [float(row["value"]) for row in cell_rows_per_variant[v]]
        for v in sorted(stats)
    }
    cell_ses = {
        v: [float(row["stat_error"]) for row in cell_rows_per_variant[v]]
        for v in sorted(stats)
    }
    cell_rows, cell_census = _spread_rows(
        cell_values, cell_ses, cell_identity, BIN_SPREAD_BOUND, "cell"
    )
    cell_pass = all(r["status"] == "pass" for r in cell_rows)

    # -- the species-level feed-down spread (the corrected lineage) ------
    replicas_for_se = SMOKE_REPLICAS if smoke else T2_REPLICAS
    species_keys = sorted({
        key for v in sorted(stats) for key in stats[v]["feeddown"]
    })
    feeddown_rows = []
    feeddown_fail = 0
    feeddown_vacuous = 0
    for key in species_keys:
        values = [
            float(stats[v]["feeddown"][key]["fraction_resonant"])
            for v in sorted(stats)
        ]
        ses = [
            float(stats[v]["feeddown"][key]["between_replica_se"])
            for v in sorted(stats)
        ]
        ns = [
            int(stats[v]["feeddown"][key]["n_finals"])
            for v in sorted(stats)
        ]
        spread = max(values) - min(values)
        median_i = _median_variant(values)
        unit_se = _pooled_se(ses)
        mad = float(np.median([abs(v - values[median_i]) for v in values]))
        row = {
            "kind": "species_feeddown",
            "species": key,
            "n_finals": ns,
            "values": values,
            "stat_errors": ses,
            "median_variant": median_i,
            "median_variant_value": values[median_i],
            "median_variant_stat_error": ses[median_i],
            "min_value": min(values),
            "max_value": max(values),
            "argmin_variant": int(values.index(min(values))),
            "argmax_variant": int(values.index(max(values))),
            "median_absolute_deviation": mad,
            "spread": float(spread),
            "unit_se": (
                float(unit_se) if math.isfinite(unit_se) else math.inf
            ),
            "bound": float(FEEDDOWN_SPREAD_BOUND),
        }
        if all(n == 0 for n in ns):
            row["sigma"] = 0.0
            row["status"] = "absent_species"
            row["pass"] = None
        elif replicas_for_se < 2:
            # The smoke's single-replica tier carries no between-replica
            # standard error (structurally zero); the feed-down spread
            # verdict does not apply there (recorded, never asserted).
            row["sigma"] = 0.0
            row["status"] = "vacuous_single_replica"
            row["pass"] = None
            feeddown_vacuous += 1
        elif math.isinf(unit_se):
            row["sigma"] = 0.0
            row["status"] = "vacuous_empty_bin"
            row["pass"] = None
            feeddown_vacuous += 1
        elif unit_se <= SE_FLOOR:
            if spread == 0.0:
                row["sigma"] = 0.0
                row["status"] = "vacuous_zero"
                row["pass"] = None
                feeddown_vacuous += 1
            else:
                row["sigma"] = math.inf
                row["status"] = "fail"
                row["pass"] = False
                feeddown_fail += 1
        else:
            sigma = float(spread) / float(unit_se)
            row["sigma"] = sigma
            row["status"] = "pass" if sigma < FEEDDOWN_SPREAD_BOUND else (
                "fail"
            )
            row["pass"] = bool(sigma < FEEDDOWN_SPREAD_BOUND)
            if not row["pass"]:
                feeddown_fail += 1
        feeddown_rows.append(row)
    applied_feeddown = [
        r for r in feeddown_rows if r["status"] in ("pass", "fail")
    ]
    feeddown_max_sigma = (
        max(abs(float(r["sigma"])) for r in applied_feeddown)
        if applied_feeddown else 0.0
    )
    feeddown_pass = feeddown_fail == 0

    # -- the named-residual register (acceptance (iv)) --------------------
    register: list[dict] = []
    for row in bin_rows:
        if row["status"] == "fail":
            register.append(
                _named_residual_entry(len(register) + 1, "bin", row)
            )
    for row in cell_rows:
        if row["status"] == "fail":
            register.append(
                _named_residual_entry(len(register) + 1, "cell", row)
            )
    for row in feeddown_rows:
        if row["status"] == "fail":
            register.append(
                _named_residual_entry(
                    len(register) + 1, "species_feeddown", row
                )
            )
    finite_scatter = [
        e for e in register if e["classification"] == "finite_mc_scatter"
    ]
    confirmation_triggered = bool(finite_scatter) and not smoke

    # -- the report -------------------------------------------------------
    events_checked = sum(stats[v]["n_events"] for v in sorted(stats))
    overall_pass = bool(
        bin_pass and feeddown_pass and cell_pass and not smoke
    )
    if smoke:
        overall_pass = bool(bin_pass and feeddown_pass and cell_pass)
    summary = {
        "variants": N_VARIANTS if not smoke else len(SMOKE_VARIANTS),
        "replicas_per_variant": (
            SMOKE_REPLICAS if smoke else T2_REPLICAS
        ),
        "events_checked": int(events_checked),
        "bin_spread_p95": bin_p95,
        "bin_spread_bound": BIN_SPREAD_BOUND,
        "bin_spread_p95_median_unit_informational": bin_p95_median_unit,
        "bin_spread_unit_note": (
            "the pre-registered unit is the pooled MC standard error of "
            "the comparison (sqrt of the summed variant SEs); the "
            "median variant per bin is the recorded reference, and "
            "bin_spread_p95_median_unit_informational carries the "
            "literal 'standard error of the median variant' reading "
            "for review (informational; the bound verdicts are "
            "evaluated on the pre-registered unit)"
        ),
        "bin_spread_p95_per_species": bin_p95_per_species,
        "bin_checks": bin_census,
        "bin_pass": bool(bin_pass),
        "cell_checks": cell_census,
        "cell_pass": bool(cell_pass),
        "feeddown_spread_max_sigma": (
            float(feeddown_max_sigma)
            if math.isfinite(float(feeddown_max_sigma)) else "Infinity"
        ),
        "feeddown_spread_bound": FEEDDOWN_SPREAD_BOUND,
        "feeddown_checks": {
            "rows": len(feeddown_rows),
            "pass": sum(
                1 for r in feeddown_rows if r["status"] == "pass"
            ),
            "fail": feeddown_fail,
            "vacuous": feeddown_vacuous,
            "absent_species": sum(
                1 for r in feeddown_rows
                if r["status"] == "absent_species"
            ),
        },
        "feeddown_pass": bool(feeddown_pass),
        "pass": bool(overall_pass),
    }
    report = {
        "schema": "na49_mixing.staged_r32_spectral_systematic.spread/1",
        "run": RUN_ID,
        "card": CARD,
        "mode": "smoke" if smoke else "production",
        "statistics": {
            "tier": "T2" if not smoke else "smoke",
            "replicas_per_variant": (
                SMOKE_REPLICAS if smoke else T2_REPLICAS
            ),
            "events_per_replica": (
                SMOKE_EVENTS if smoke else T2_EVENTS_PER_REPLICA
            ),
            "variants": {
                str(v): {
                    "ramp_start_gev": float(
                        next(
                            e["ramp_start_gev"] for e in VARIANTS
                            if int(e["variant"]) == v
                        )
                    ) if not smoke else None,
                    "n_events": int(stats[v]["n_events"]),
                    "routing": stats[v]["routing"],
                }
                for v in sorted(stats)
            },
            "events_checked": int(events_checked),
        },
        "spread_bounds": {
            "bin_spread_bound": BIN_SPREAD_BOUND,
            "bin_spread_unit": (
                "the pooled MC standard error of the comparison "
                "sqrt(SE_0^2 + SE_1^2 + SE_2^2); the absolute bin-level "
                "cross-variant deviation is the maximum pairwise "
                "deviation (max - min) across variants; the median "
                "variant per bin is recorded as the reference"
            ),
            "feeddown_spread_bound": FEEDDOWN_SPREAD_BOUND,
            "feeddown_spread_unit": (
                "the combined between-replica SE "
                "sqrt(SE_0^2 + SE_1^2 + SE_2^2) of the three variants' "
                "per-species feed-down fractions; the spread is the "
                "(max - min) across variants"
            ),
            "se_floor": SE_FLOOR,
        },
        "bin_checks": [
            {
                key: row[key]
                for key in (
                    "species", "bin_index", "xF_mid", "pT_mid",
                    "values", "stat_errors", "median_variant",
                    "median_variant_value", "median_variant_stat_error",
                    "min_value", "max_value", "argmin_variant",
                    "argmax_variant", "median_absolute_deviation",
                    "spread", "unit_se",
                    "sigma_median_unit_informational",
                    "sigma", "bound", "status", "pass",
                )
            }
            for row in bin_rows
        ],
        "cell_checks": [
            {
                key: row[key]
                for key in (
                    "rank", "sector", "observable", "region", "bin_index",
                    "xF_mid", "values", "stat_errors", "median_variant",
                    "median_variant_value", "median_variant_stat_error",
                    "min_value", "max_value", "argmin_variant",
                    "argmax_variant", "median_absolute_deviation",
                    "spread", "unit_se", "sigma", "bound", "status",
                    "pass",
                )
            }
            for row in cell_rows
        ],
        "feeddown_checks": [
            {
                key: row[key]
                for key in (
                    "kind", "species", "n_finals", "values",
                    "stat_errors", "median_variant",
                    "median_variant_value", "median_variant_stat_error",
                    "min_value", "max_value", "argmin_variant",
                    "argmax_variant", "median_absolute_deviation",
                    "spread", "unit_se", "sigma", "bound", "status",
                    "pass",
                )
            }
            for row in feeddown_rows
        ],
        "routing_census": {
            str(v): stats[v]["routing"] for v in sorted(stats)
        },
        "summary": summary,
        "named_residual_register": register,
        "confirmation_trigger": {
            "triggered": bool(confirmation_triggered),
            "rule": (
                "triggered only by finite-MC-scatter diagnostics of the "
                "named-residual register (acceptance (v)); the "
                "confirmation re-evaluation runs at 16 replicas x "
                "100,000 events per variant (4,800,000 events), raw "
                "seeds 20260919 + 40000 + 900 + 100*v + r, gateway, and "
                "is sealed in its own run dir "
                f"dev/00runs/{CONFIRMATION_RUN_NAME}"
            ),
            "finite_mc_scatter_entries": len(finite_scatter),
        },
        "no_fitting": True,
        "optimizer_invoked": False,
    }
    _write_json(run_dir / f"{prefix}{SPREAD_REPORT_NAME}", report)
    if not smoke:
        print(
            "staged_r32_spectral_systematic: spread evaluation written "
            f"to {run_dir / SPREAD_REPORT_NAME} (bin p95 {bin_p95:.6f} "
            f"vs bound {BIN_SPREAD_BOUND}, feed-down max sigma "
            f"{summary['feeddown_spread_max_sigma']} vs bound "
            f"{FEEDDOWN_SPREAD_BOUND}, named residuals "
            f"{len(register)}, confirmation triggered "
            f"{confirmation_triggered})",
            flush=True,
        )
    return report


# ---------------------------------------------------------------------------
# Mode: --gate
# ---------------------------------------------------------------------------


def _assert_spread_report(run_dir: Path, smoke: bool) -> dict:
    path = Path(run_dir) / (
        f"{SMOKE_PREFIX}{SPREAD_REPORT_NAME}" if smoke else SPREAD_REPORT_NAME
    )
    if not path.is_file():
        _refuse(
            f"{path} is missing; the gate requires the spread evaluation "
            "(run --compare first)"
        )
    return _read_json(path)


def run_gate(run_dir: Path) -> dict:
    """Write ``gate3_r32.json``: the Gate 3 evidence, assembled from the
    spread evaluation, the bit-identity check, and the sealed R31 /
    card-8c cross-references (evidence only)."""
    run_dir = Path(run_dir)
    _refuse_smoke_artifacts(run_dir)
    pre = _assert_pre_registration(run_dir)
    bit = _assert_bit_identity(run_dir)
    report = _assert_spread_report(run_dir, smoke=False)
    summary = report["summary"]
    if report.get("mode") != "production":
        _refuse("the spread report is not a production record")
    routed = report["routing_census"]
    clauses = {
        "smoothness": {
            "p_plus_p_consistency": "pass" if summary["pass"] else "fail",
            "r31_cross_reference": {
                "artifact": (
                    "dev/00runs/r31_catalog_continuum_matching_scan_"
                    "2026-09-17/gate3_r31.json"
                ),
                "card_8c_authoritative": (
                    "dev/00runs/r3_se_audit_2026-09-18/"
                    "confirmation_verification.json (0/422, max 2.96)"
                ),
            },
        },
        "no_boundary_dominance": {
            "variants": int(summary["variants"]),
            "events_checked": int(summary["events_checked"]),
            "bit_identity_v0_vs_r23": {
                "layers": list(LAYER_DEFINITIONS),
                "events": int(bit["n_events"]),
                "pass": bool(bit["pass"]),
            },
            "bin_spread_p95_abs_se": float(summary["bin_spread_p95"]),
            "bin_spread_bound": float(BIN_SPREAD_BOUND),
            "feeddown_spread_max_sigma": summary[
                "feeddown_spread_max_sigma"
            ],
            "feeddown_spread_bound": float(FEEDDOWN_SPREAD_BOUND),
            "routing_census": {
                f"v{v}": float(routed[str(v)]["fraction_catalog"])
                for v in sorted(routed, key=lambda x: int(x))
            },
            "pass": bool(summary["pass"]),
        },
    }
    payload = {
        "schema": "na49_mixing.staged_r32_spectral_systematic.gate/1",
        "gate": "gate 3 (plan section 9, line 650)",
        "run": RUN_ID,
        "card": CARD,
        "clauses": clauses,
        "named_residual_register": list(report["named_residual_register"]),
        "confirmation_cross_reference": (
            report["confirmation_trigger"] if report[
                "confirmation_trigger"]["triggered"] else None
        ),
        "events_checked": int(summary["events_checked"]),
        "no_fitting": True,
        "optimizer_invoked": False,
        "sealed": True,
        "seal_note": (
            "sealed marks the gate evidence as the sealed form: the "
            "seal covers this file's bytes (the seal manifest records "
            "them after this file is written)"
        ),
        "decision_note": (
            "evidence only; the written proceed / revise / reject "
            "decision on the two Gate 3 clauses is the Gate 3 "
            "follow-up obligation (round 3 index card), recorded in "
            "_knowledge/"
        ),
        "input_hashes": {
            "spread_report_sha256": _sha256_file(
                run_dir / SPREAD_REPORT_NAME
            ),
            "bit_identity_sha256": _sha256_file(
                run_dir / BIT_IDENTITY_NAME
            ),
            "pre_registration_sha256": _sha256_file(
                run_dir / PREREGISTRATION_NAME
            ),
        },
    }
    del pre
    _write_json(run_dir / GATE_RECORD_NAME, payload)
    print(
        "staged_r32_spectral_systematic: gate evidence written to "
        f"{run_dir / GATE_RECORD_NAME} (p_plus_p_consistency = "
        f"{clauses['smoothness']['p_plus_p_consistency']}, "
        "no_boundary_dominance pass = "
        f"{clauses['no_boundary_dominance']['pass']}, named residuals "
        f"{len(payload['named_residual_register'])})",
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
    itself and ``seal_manifest.json``); the entries carry
    monorepo-root-relative paths, so the card's validation command
    ``sha256sum -c dev/00runs/<run>/SHA256SUMS.txt`` verifies from the
    repository root (the R21 convention)."""
    run_dir = Path(run_dir)
    skip = {SHA256SUMS_NAME, SEAL_NAME}
    lines = []
    for path in _run_dir_files(run_dir):
        if path.name in skip:
            continue
        rel = path.relative_to(MONO_ROOT).as_posix()
        lines.append(f"{_sha256_file(path)}  {rel}\n")
    sums_path = run_dir / SHA256SUMS_NAME
    _write_bytes(sums_path, "".join(lines).encode("utf-8"))
    return sums_path


def write_seal_manifest(run_dir: Path) -> dict:
    """Write ``seal_manifest.json`` over every other file plus the
    sums.  The seal manifest records every regular file under the run
    directory except itself, each with its sha256 and byte size; it
    cannot contain its own digest."""
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
        "schema": "na49_mixing.staged_r32_spectral_systematic"
                  ".seal_manifest/1",
        "run": RUN_ID,
        "card": CARD,
        "created_utc": _now_utc(),
        "file_count": len(entries),
        "total_bytes": int(total_bytes),
        "sha256_sums_file": SHA256SUMS_NAME,
        "sha256_sums_sha256": _sha256_file(sums_path),
        "excluded_from_sums": [SEAL_NAME, SHA256SUMS_NAME],
        "path_convention": (
            "the SHA256SUMS.txt entries carry monorepo-root-relative "
            f"paths (dev/00runs/{run_dir.name}/ prefixed), so the run "
            f"card's validation command `sha256sum -c dev/00runs/"
            f"{run_dir.name}/SHA256SUMS.txt` verifies from the repo "
            "root (the R21 convention)"
        ),
        "note": (
            "the run dir is immutable after the seal (the Round 1 "
            "sealing discipline); clean_environment_commands.json is "
            "the post-seal validation record, written after this "
            "manifest and never edited"
        ),
        "files": entries,
    }
    _write_json(run_dir / SEAL_NAME, seal)
    return seal


def run_seal(run_dir: Path) -> dict:
    """Seal the finished run directory (sums inventory plus seal
    manifest).  The pre-registration, the provenance, the bit-identity
    report, the 24 production variant records, the spread report, and
    the gate record must exist; smoke artifacts are refused."""
    run_dir = Path(run_dir)
    _refuse_smoke_artifacts(run_dir)
    for name in (
        PREREGISTRATION_NAME,
        PROVENANCE_NAME,
        BIT_IDENTITY_NAME,
        SPREAD_REPORT_NAME,
        GATE_RECORD_NAME,
    ):
        if not (run_dir / name).is_file():
            _refuse(
                f"{run_dir / name} is missing before sealing (the gate "
                "JSON is part of the seal; run --compare and --gate "
                "first)"
            )
    n_records = len(list(run_dir.glob("variant_*_replica_*.json")))
    if n_records != N_VARIANTS * T2_REPLICAS:
        _refuse(
            f"expected {N_VARIANTS * T2_REPLICAS} variant records before "
            f"sealing, found {n_records}"
        )
    write_sha256sums(run_dir)
    seal = write_seal_manifest(run_dir)
    print(
        f"staged_r32_spectral_systematic: run dir sealed "
        f"({len(seal['files'])} files listed in {SEAL_NAME})",
        flush=True,
    )
    return seal


# ---------------------------------------------------------------------------
# Mode: --clean-validate
# ---------------------------------------------------------------------------


def run_clean_validate(run_dir: Path) -> dict:
    """Verify the seal, re-run the spread evaluation from a clean
    interpreter over the sealed artifacts, and confirm the report hash
    matches."""
    run_dir = Path(run_dir)
    _refuse_smoke_artifacts(run_dir)
    _load_pre_registration(run_dir)
    if not (run_dir / SHA256SUMS_NAME).is_file():
        _refuse(
            f"{run_dir / SHA256SUMS_NAME} is missing; --clean-validate "
            "requires the sealed run dir (run --seal first)"
        )
    # -- seal verification (root-relative entries) ------------------------
    mismatches: list[str] = []
    files_checked = 0
    for line in (run_dir / SHA256SUMS_NAME).read_text(
            encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        payload = MONO_ROOT / name
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

    # -- clean-interpreter re-run of the spread evaluation ----------------
    report_path = run_dir / SPREAD_REPORT_NAME
    before = report_path.read_bytes()
    driver_path = Path(__file__).resolve()
    argv = [sys.executable, "-I", str(driver_path), "--compare",
            "--run-dir", str(run_dir)]
    probe = subprocess.run(argv, capture_output=True, text=True,
                           cwd=str(MONO_ROOT))
    if probe.returncode != 0:
        _refuse(
            "the clean-interpreter spread-evaluation probe failed "
            f"(return code {probe.returncode}):\n{probe.stderr}"
        )
    recomputed = report_path.read_bytes()
    hashes_match = (
        int(probe.returncode) == 0
        and hashlib.sha256(before).hexdigest()
        == hashlib.sha256(recomputed).hexdigest()
        and before == recomputed
    )
    # Restore the sealed bytes: the probe rewrote the report; the sealed
    # dir must keep its sealed content regardless of the probe outcome.
    if recomputed != before:
        _write_bytes(report_path, before)
    payload = {
        "schema": "na49_mixing.staged_r32_spectral_systematic"
                  ".clean_environment/1",
        "run": RUN_ID,
        "card": CARD,
        "verified_utc": _now_utc(),
        "seal_verification": {
            "files_checked": int(files_checked),
            "mismatches": mismatches,
        },
        "commands": [{
            "purpose": (
                "clean-interpreter re-run of the spread evaluation over "
                "the sealed artifacts (rewrites spread_report.json; the "
                "sealed bytes must be reproduced byte for byte)"
            ),
            "argv": argv,
            "cwd": str(MONO_ROOT),
        }],
        "environment": environment_fingerprint(),
        "probe_returncode": int(probe.returncode),
        "sealed_report_sha256": {
            SPREAD_REPORT_NAME: hashlib.sha256(before).hexdigest(),
        },
        "recomputed_report_sha256": {
            SPREAD_REPORT_NAME: hashlib.sha256(recomputed).hexdigest(),
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
            "staged_r32_spectral_systematic: the clean-environment "
            "validation FAILED (the recomputed spread-report bytes "
            "differ from the sealed report)",
            flush=True,
        )
        raise SystemExit(1)
    print(
        "staged_r32_spectral_systematic: clean-environment validation "
        f"passed ({files_checked} seal files verified, the clean "
        f"interpreter reproduced {SPREAD_REPORT_NAME} byte for byte)",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="staged_r32_spectral_systematic.py",
        description=(
            "Execute run R32, the p+p spectral systematic (Round 3, "
            "card 9): three pre-registered transition-window variants "
            "of the card-6 blend at sqrt(s) = 17.2 GeV, the card-13 "
            "builder's additive blend routing, no refit, and the "
            "cross-variant spread recorded as the model uncertainty "
            "(plan section 9, line 644)."
        ),
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--pre-registration", action="store_true",
        help="write pre_registration.json and provenance.json, then stop",
    )
    modes.add_argument(
        "--bit-identity", action="store_true",
        help="run the pre-registered 10,000-event bit-identity check "
             "(variant v0 at the R23 replica-0 seed 20300913), then stop",
    )
    modes.add_argument(
        "--generate", action="store_true",
        help="generate one (variant, replica) (--variant V --replica R), "
             "one whole variant (--variant V), or every variant",
    )
    modes.add_argument(
        "--compare", action="store_true",
        help="run the spread evaluation over the sealed statistics, "
             "then stop",
    )
    modes.add_argument(
        "--gate", action="store_true",
        help="write gate3_r32.json (acceptance evidence), then stop",
    )
    modes.add_argument(
        "--seal", action="store_true",
        help="write seal_manifest.json and SHA256SUMS.txt, then stop",
    )
    modes.add_argument(
        "--clean-validate", action="store_true",
        help="verify the seal and re-run the spread evaluation cleanly, "
             "then stop",
    )
    modes.add_argument(
        "--smoke", action="store_true",
        help="the hermetic end-to-end driver validation (a smoke "
             "pre-registration, 2 variants x 1 replica x 2,000 events on "
             "the synthetic hermetic grid, and the smoke spread "
             "evaluation) into an explicit --run-dir; never sealed",
    )
    parser.add_argument(
        "--variant", type=int, default=None,
        help="with --generate: the variant index v = 0..2",
    )
    parser.add_argument(
        "--replica", type=int, default=None,
        help="with --generate: the replica index r = 0..7",
    )
    parser.add_argument(
        "--variants", type=int, default=None,
        help="the declared variant count (must match the "
             "pre-registration in production mode)",
    )
    parser.add_argument(
        "--replicas", type=int, default=None,
        help="the declared replica count per variant (must match the "
             "pre-registration in production mode)",
    )
    parser.add_argument(
        "--events-per-replica", type=int, default=None,
        help="the declared events per replica (must match the "
             "pre-registration in production mode)",
    )
    parser.add_argument(
        "--master-seed", type=int, default=None,
        help="the declared master seed (must match the "
             "pre-registration in production mode)",
    )
    parser.add_argument(
        "--bit-identity-events", type=int, default=BIT_IDENTITY_EVENTS,
        help=(f"the bit-identity event count (default "
              f"{BIT_IDENTITY_EVENTS}; the production check is fixed)"),
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
    statistics = {
        "variants": args.variants,
        "replicas": args.replicas,
        "events_per_replica": args.events_per_replica,
        "master_seed": args.master_seed,
    }
    if args.pre_registration:
        run_pre_registration(run_dir)
    elif args.bit_identity:
        run_bit_identity(run_dir, int(args.bit_identity_events))
    elif args.generate:
        run_generation(
            run_dir, args.variant, args.replica, args.workers, statistics
        )
    elif args.compare:
        run_spread_evaluation(run_dir, smoke=False)
    elif args.gate:
        run_gate(run_dir)
    elif args.seal:
        run_seal(run_dir)
    elif args.clean_validate:
        run_clean_validate(run_dir)
    elif args.smoke:
        run_smoke(run_dir, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

