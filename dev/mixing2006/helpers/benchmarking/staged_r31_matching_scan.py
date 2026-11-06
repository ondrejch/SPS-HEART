#!/usr/bin/env python3
"""R31 driver: the catalog/continuum matching scan (Round 3, card 8).

Task ``TASK-20260917-r3-run-r31-catalog-continuum-matching-scan`` (plan
section 9 run R31, line 642; the Round 3 card set, card 8; cards 2, 6,
and 7 are the prerequisites).  R31 scans the source-level behavior of
the card-6 blend routing through the three pre-registered transition
windows: T1 isolated-source runs on a five-node at-rest grid per
activated continuum family, straddling the catalog-completeness
boundary ``B = 2.680194110727528 GeV``, storing the card-9 diagnostic
set (species multiplicities, per-species momentum moments, feed-down
ancestry, charge balance) plus the per-node routing fraction, and
checking the Gate 3 smoothness clause (plan section 9, line 650) with
the re-registered threshold metric of card 2 at the unchanged
3.0-sigma tolerance.  No fitting of any kind (``no_fitting = true``);
a failing match re-opens the card-6 routing or the originating family
card, never the tolerance.

Windows (round 3 index item 1): three windows, one per activated
continuum family of ``config/v2_singlet/continuum_resonances.json``,
each a five-node at-rest grid ``M_j = 2.48 + 0.1 j GeV``, ``j = 0..4``
(2.48, 2.58, 2.68, 2.78, 2.88 GeV).  Node 2.68 sits 0.00194 GeV below
the boundary, node 2.78 sits 0.09981 GeV above it.  Window 0 is the
baryonic family ``(1, 1, 0)`` (``baryonic_np``), window 1 the strange
baryonic family ``(1, 0, -1)`` (``baryonic_strange``), window 2 the
neutral mesonic family ``(0, 0, 0)`` (``mesonic_neutral``).  The event
is the isolated source decay (no target, no exchange record), the
R21/R22 convention (round 2 index item 10), with the source at rest
(``p = 0``, ``E = M_j``).

Statistics (T1, plan section 5.1): 4 replicas x 25,000 decays per
node; 3 x 5 x 100,000 = 1,500,000 decays total.

Seeds (round 3 index item 7): master seed 20260917; per (window ``w``,
node ``j``, replica ``r < 4``, decay ``i < 25,000``)
``derive_substreams(20260917, r, (w*5 + j) * 1_000_000 + i)`` on the
closed ``decay_channel`` / ``decay_kinematics`` labels (the card-9 seed
map with the window folded into the point index).  The R31 seeds are
disjoint from the R20-R23 seeds (those derive from master 20260913;
R23 uses raw replica seeds).

Decay routing (card 6, round 3 index item 4), applied per grid node:

- ``M_j < B``: the catalog mixture, the R23 builder convention.  The
  routing consumes zero uniforms, so the mixture layer's draw sequence
  is byte-identical to the no-blend convention at the same seed.
- ``B <= M_j < B + W``: route to the continuum family mixture with
  probability ``w(M_j)`` and to the catalog mixture with probability
  ``1 - w(M_j)``; the routing draw is a single uniform on the spawned
  ``decay_channel`` child, consumed before any record draw, so the
  lineage is traceable from the route.
- ``M_j >= B + W``: the continuum family mixture; zero routing
  uniforms.  No R31 node reaches this branch (the top node 2.88 lies
  inside the transition band), and the rule is pre-registered in full
  nevertheless.

Per-route realization (pre-registered; the R31 fixed-mass-grid
conventions, asserted per decay by the kernel's own checks):

- **Catalog route**: the card-9 mixture convention at the node mass.
  The open set at ``M_j`` is the catalog records with the source
  ``(B, Q, S)`` whose truncated spectral window
  ``[mass_range[0], min(mass_range[1], M_j)]`` is non-empty; the
  record draw is the R23 builder's expression
  ``min(int(u * len(open)), len(open) - 1)`` on the spawned
  ``decay_channel`` child; the record's mass is realized from the
  card-3 spectral function restricted to the truncated window (the
  card-3 ``draw_mass`` on the spawned ``decay_kinematics`` child; a
  degenerate window, width at most 1e-12, fixes the mass at the node
  mass ``M_j``); a drawn record already in the stable set contributes
  itself as the single identity final at ``(M_j, 0, 0, 0)`` with no
  producing vertex; any other drawn record decays through the Round 2
  kernel (``ResonanceDecayer.decay_source``) at the at-rest
  four-momentum of the realized mass.  This is the realization the
  card-2 metric engine integrates, so the metric is a consistency
  test of the same convention on both sides.
- **Continuum route**: the family-mixture analogue of the card-9
  realization.  The activated families with the source ``(B, Q, S)``
  are exactly one per window (asserted); the continuum state sits at
  rest at the node mass (the plan section 9.3 blend evaluated at the
  source mass: at mass ``M`` the produced state is the continuum state
  at ``M``, so there is no family-mass redraw); one allowed decay
  class of the family is drawn uniformly over the family's
  ``allowed_decay_classes`` (one uniform on the spawned
  ``decay_channel`` child; the card-9 flat-mixture rule over the
  pre-registered class list); the root split is the kernel's own
  two-body convention at rest at the node mass (the isotropic
  direction from two uniforms, then the daughter masses in channel
  order, each resonance daughter from the card-3 ``draw_mass`` inside
  its on-shell budget window with the partner at its pole mass, each
  stable daughter at its pole mass, the Kaellen decay momentum and the
  exact energy split); every resonance daughter cascades through the
  Round 2 kernel (``ResonanceDecayer.decay_resonance``) on the labeled
  substreams.  The node mass is the source mass of the event on both
  routes and the continuum contribution is a smooth function of the source mass, which is what
  the Gate 3 smoothness clause tests.  Draw budget per decay: at most
  one routing uniform, one class/record uniform, two family-mass
  uniforms, and the root split's direction and daughter-mass uniforms
  on the two spawned children; the kernel keeps the labeled ids
  themselves (no extension of ``SUBSTREAM_LABELS``).

Discontinuity metric (the re-registered metric of card 2,
``na49_mixing.threshold_metric``, pre-registered): for every adjacent
node pair ``(j, j+1)`` in each window (four pairs per window, twelve
total) and every species of the catalog stable termination set, with
``SE_combined = sqrt(SE_j^2 + SE_{j+1}^2)``, the observed difference of
the species multiplicity and of the per-species momentum moments
(``<p>`` and ``<p_T>_y``) satisfies

    |observed - delta_analytic_full| <= 3 * SE_combined,

where ``delta_analytic_full`` is the full analytic difference of the
routed flat-mixture expectation between the two node masses,

    mu_routed(M) = (1 - w(M)) * mu_catalog(M) + w(M) * mu_continuum(M),

``mu_catalog(M)`` is the card-9 flat record mixture payload of the
card-2 engine (``RecordSet.mixture_payload`` over the open set at
``M``, the card-9 truncated-window rule at the point mass) and
``mu_continuum(M)`` the continuum-family payload of this driver's
analytic engine (the family density integrated over the truncated
support on a fixed Gauss-Legendre grid, the class shares uniform, the
root split and daughter budget windows integrated with the card-2
engine's own kinematics scheme, no Monte Carlo).  At the pairs where
both nodes lie below the boundary (``w = 0`` at both masses) the routed
difference reduces exactly to the card-2
``full_analytic_difference`` value with its open-set-change and
momentum-drift decomposition, which the driver applies verbatim and
cross-checks against its own block arithmetic.  The pair straddling
the boundary (``j = 2``, 2.68 to 2.78) is checked like any other pair;
no special-casing of the boundary pair is permitted.  The multiplicity
standard error is the per-node pooled standard error of the mean (the
card-9 stored ``se``); the moment standard errors are the
between-replica standard errors of the four stored replica moments
(std, ddof=1, over sqrt(4)), the R21/R22 convention.  A moment check
applies only where the species is present at both nodes of the pair
(``absent_species`` otherwise; the multiplicity check always applies).
The tolerance is the pre-registered 3.0-sigma band, never relaxed.

Balance: every event conserves the source ``(B, Q, S)`` exactly at
every vertex (the Round 2 kernel conservation, checked per event,
never repaired; a violation raises and stops the run).

Gate evidence (``gate3_r31.json``): evidence only.  The smoothness
clause carries the checked/failed counts, the maximum |sigma| overall,
on the boundary-straddling pairs (global pair indices 2, 6, 10) and on
the interior pairs; the no-boundary-dominance clause compares the
boundary pair's failure rate against the interior pairs' failure rate
of the same window.  ``no_fitting`` is true; ``events_checked`` is
1,500,000.

Modes (exactly one per invocation; each performs one step and stops):

- ``--pre-registration``: write ``pre_registration.json`` and
  ``provenance.json`` into the run dir, then stop.  Nothing is
  generated.
- ``--generate [--window W] [--node J]``: generate one node
  (``--window W --node J``), one whole window (``--window W``), or
  every window, writing ``window_<w>_node_<j>.json`` and
  ``window_<w>_node_<j>_replica_<r>.json`` per node, then stop.  The
  statistics flags (``--windows/--nodes/--decays-per-replica/
  --replicas/--master-seed``) must match the pre-registration in
  production mode.  A completed node record that matches the
  pre-registration is reused unchanged, so the declared tier can be
  completed window by window.
- ``--verify``: run the matching verification (the re-registered
  metric over all twelve adjacent pairs and all species), write
  ``matching_verification.json``, then stop.  A failing check stops
  the run (exit 1) after the report is written.
- ``--gate``: write ``gate3_r31.json`` (evidence only), then stop.
  Requires the passing verification.
- ``--seal``: write ``seal_manifest.json`` and ``SHA256SUMS.txt`` over
  the run dir, then stop.  The run dir is immutable after the seal.
- ``--clean-validate``: verify the seal, re-run the matching
  verification from a clean interpreter over the sealed artifacts, and
  confirm the report hash matches; write
  ``clean_environment_commands.json``, then stop.
- ``--smoke`` (with ``--generate``): hermetic reduced statistics for
  driver validation; ``--verify`` / ``--gate`` / ``--seal`` /
  ``--clean-validate`` refuse smoke artifacts.

``matching_verification.json`` and ``gate3_r31.json`` carry no
wall-clock fields: their bytes are a pure function of the sealed
artifacts and the committed code, which is what the clean-environment
validation re-hashes.

Dual-tree driver: byte-identical in the nested tree
(``dev/mixing2006/helpers/benchmarking/``) and the monorepo mirror
(``helpers/benchmarking/``); both locate the package with the
established ``_locate_roots`` pattern.

Hermetic: the committed catalog, the committed continuum config, and
the vendored ``na49_mixing`` modules only; no external tables, no
event generation outside the declared modes.
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
    print(f"staged_r31_matching_scan: {message}", file=sys.stderr, flush=True)
    raise SystemExit(2)


NESTED_ROOT, MONO_ROOT = _locate_roots()
if str(NESTED_ROOT) not in sys.path:
    sys.path.insert(0, str(NESTED_ROOT))

from helpers.benchmarking import staged_r01_reference as r01  # noqa: E402
from na49_mixing.hadron_catalog import HadronCatalog  # noqa: E402
from na49_mixing.pdg_database import get_particle  # noqa: E402
from na49_mixing.quantum_numbers import quantum_numbers  # noqa: E402
from na49_mixing.resonance_decay import (  # noqa: E402
    ResonanceDecayer,
    derive_charge_weights,
    class_pair_of,
)
from na49_mixing.resonance_decay import SourceRecord as KernelSourceRecord  # noqa: E402
from na49_mixing.resonance_widths import decay_momentum, draw_mass  # noqa: E402
from na49_mixing.rng_substreams import derive_substreams  # noqa: E402
from na49_mixing.spectral_density import (  # noqa: E402
    BlendSpec,
    _daughter_masses,
    active_families,
    families_from_json,
    route_source_decay,
)
from na49_mixing.threshold_metric import (  # noqa: E402
    ISOTROPIC_SIN_THETA_MEAN,
    SIGMA_TOLERANCE,
    THRESHOLD_METRIC_ID,
    RecordSet,
    evaluate_pair,
    full_analytic_difference,
)

# ---------------------------------------------------------------------------
# Pre-registered run identity (card TASK-20260917-r3-run-r31-catalog-
# continuum-matching-scan; the windows, statistics, seeds, routing, and
# metric are verbatim from the card and the round 3 index items 1, 4
# and 7).
# ---------------------------------------------------------------------------

RUN_ID = "R31"
CARD = "TASK-20260917-r3-run-r31-catalog-continuum-matching-scan"
RUN_NAME = "r31_catalog_continuum_matching_scan_2026-09-17"
DEFAULT_RUN_DIR = "dev/00runs/r31_catalog_continuum_matching_scan_2026-09-17"
GATE_RECORD_NAME = "gate3_r31.json"
VERIFICATION_NAME = "matching_verification.json"
PREREGISTRATION_NAME = "pre_registration.json"
PROVENANCE_NAME = "provenance.json"
CLEAN_ENV_NAME = "clean_environment_commands.json"
SHA256SUMS_NAME = "SHA256SUMS.txt"
SEAL_NAME = "seal_manifest.json"

CATALOG_REL = "config/v2_singlet/resonance_catalog.json"
FAMILIES_REL = "config/v2_singlet/continuum_resonances.json"
DRIVER_REL = "helpers/benchmarking/staged_r31_matching_scan.py"

#: The card chain this run sits on (the Round 3 prerequisites).
CARD_CHAIN = (
    "TASK-20260917-r3-gate2-followup-threshold-metric-reregistration",
    "TASK-20260917-r3-baryonic-continuum-spectral-family",
    "TASK-20260917-r3-mesonic-continuum-spectral-family",
    "TASK-20260917-r3-smooth-catalog-continuum-matching",
    "TASK-20260917-r3-run-r30-spectral-normalization",
    "TASK-20260917-r3-run-r31-catalog-continuum-matching-scan",
)

#: The three transition windows (round 3 index item 1): one per
#: activated continuum family, in the pre-registered order.
WINDOWS: tuple[dict, ...] = (
    {"window": 0, "qn": (1, 1, 0), "label": "baryonic (1, 1, 0)",
     "family_expectation": "baryonic_np"},
    {"window": 1, "qn": (1, 0, -1), "label": "strange baryonic (1, 0, -1)",
     "family_expectation": "baryonic_strange"},
    {"window": 2, "qn": (0, 0, 0), "label": "neutral mesonic (0, 0, 0)",
     "family_expectation": "mesonic_neutral"},
)
N_WINDOWS = len(WINDOWS)

#: The five-node at-rest grid per window (round 3 index item 1):
#: ``M_j = 2.48 + 0.1 j``, ``j = 0..4``.
NODE_START_GEV = 2.48
NODE_STEP_GEV = 0.1
N_NODES = 5
#: The pre-registered node masses (exact decimal values; the grid rule
#: ``M_j = 2.48 + 0.1 j`` evaluated exactly).
NODE_MASSES = [2.48, 2.58, 2.68, 2.78, 2.88]

#: T1 statistics (plan section 5.1): 4 replicas x 25,000 decays per
#: node; 3 x 5 x 100,000 = 1,500,000 decays total.
T1_REPLICAS = 4
T1_DECAYS_PER_REPLICA = 25_000
T1_LABEL = "T1: 4 replicas x 25,000 decays per node (plan section 5.1)"
TOTAL_DECAYS = N_WINDOWS * N_NODES * T1_REPLICAS * T1_DECAYS_PER_REPLICA

#: Master seed and the per-decay seed map (round 3 index item 7): the
#: card-9 seed map with the window folded into the point index.
MASTER_SEED = 20260917
SEED_RULE = (
    "derive_substreams(20260917, r, (w*5 + j) * 1_000_000 + i) per "
    "(window w, node j, replica r < 4, decay i < 25,000) on the closed "
    "decay_channel/decay_kinematics labels (the card-9 seed map)"
)
SEED_LABELS = ("decay_channel", "decay_kinematics")

#: The blend (card 6; round 3 index item 4).
SPEC = BlendSpec()
BOUNDARY = float(SPEC.boundary)
RAMP_WIDTH = float(SPEC.ramp_width)
#: The routing weights at the five nodes (the pre-registered ramp):
#: the three below-boundary nodes route to the catalog with w = 0
#: exactly; node 2.78 and node 2.88 sit inside the transition band.
EXPECTED_W = [float(SPEC.weight(M)) for M in NODE_MASSES]

#: The degenerate-window bound (the card-9 mixture item 3 rule).
DEGENERATE_WINDOW_TOL = 1e-12

#: The re-registered metric (card 2) and the tolerance (never relaxed).
METRIC_ID = THRESHOLD_METRIC_ID
METRIC_N_SIGMA = SIGMA_TOLERANCE

#: Global pair indices of the boundary-straddling pairs: window w
#: carries the four pairs 4w + j (j = 0..3); the boundary pair is
#: j = 2 (2.68 -> 2.78).
BOUNDARY_PAIR_INDICES = [4 * w + 2 for w in range(N_WINDOWS)]

#: Generation chunking (the R01/R23 chunk convention).
CHUNK_SIZE = 2500
DEFAULT_WORKERS = 8

#: Smoke mode defaults (driver validation; never sealed).
SMOKE_DECAYS = 200
SMOKE_REPLICAS = 1

#: The pair diagnostics (the card-9 output schema's co-occurrence set).
PAIR_DEFINITIONS = (
    ("pi_plus_pi_minus_per_event", 211, -211),
    ("p_pbar_per_event", 2212, -2212),
    ("n_nbar_per_event", 2112, -2112),
)

#: The continuum-source root sentinel in the vertex lineage: the root
#: vertex of a continuum-routed decay carries parent_pdg 0 (never a
#: particle id), so a final particle is resonant feed-down exactly when
#: its producing vertex's parent is a resonance other than the drawn
#: root (the corrected lineage convention of Round 3 card 3; the root
#: record's direct daughters are direct source products).
CONTINUUM_ROOT_SENTINEL = 0

#: The kernel boundary (the M8-frozen continuum.match_mass; the blend
#: boundary B is the same value).
KERNEL_BOUNDARY = BOUNDARY


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


def _bqs_of(catalog: HadronCatalog, pdg_id: int) -> tuple[int, int, int]:
    """The ``(B, Q, S)`` of a particle: the catalog record fields for
    records, the exact PDG quantum numbers otherwise (the card-9
    driver's convention)."""
    try:
        record = catalog.get(int(pdg_id))
        return (int(record.B), int(record.Q), int(record.S))
    except KeyError:
        qn = quantum_numbers(int(pdg_id))
        return (int(qn.B), int(qn.Q), int(qn.S))


def _is_stable_state(catalog: HadronCatalog, pdg_id: int) -> bool:
    """The kernel's stable convention: a stable termination state (the
    catalog's ``stable_states()`` set)."""
    return int(pdg_id) in catalog.stable_states()


def _open_set(
    catalog: HadronCatalog, qn: tuple[int, int, int], mass: float
) -> list:
    """The card-9 open set at *mass*: the catalog records with the
    source ``(B, Q, S)`` whose truncated spectral window
    ``[mass_range[0], min(mass_range[1], mass)]`` is non-empty, in
    catalog record order."""
    out = []
    for record in catalog.records():
        if (int(record.B), int(record.Q), int(record.S)) != tuple(qn):
            continue
        lo = float(record.mass_range[0])
        hi = float(record.mass_range[1])
        if min(hi, float(mass)) >= lo:
            out.append(record)
    return out


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


# ---------------------------------------------------------------------------
# Pure-compute precomputation shared across the decays (no draws): the
# daughter-mass maps, the card-5 class maps, the per-node open sets,
# and the per-(family, top) truncated family draw tables.
# ---------------------------------------------------------------------------


def _precompute_kernel_maps(catalog: HadronCatalog) -> tuple[dict, dict]:
    """The kernel's own pure-compute caches, precomputed once and
    primed into every per-decay ``ResonanceDecayer`` (the caches are
    documented pure compute caches; priming them with the identical
    values changes no draw)."""
    mass_maps: dict[int, dict[int, float]] = {}
    for record in catalog.records():
        mass_maps[int(record.pdg_id)] = _daughter_masses(catalog, record)
    class_maps: dict[tuple[int, tuple[int, int]], dict] = {}
    for record in catalog.records():
        for channel in record.decay_channels:
            if channel.is_three_body or len(channel.daughters) != 2:
                continue
            key = (int(record.pdg_id), class_pair_of(channel))
            if key not in class_maps:
                class_maps[key] = derive_charge_weights(
                    record, class_pair_of(channel), catalog
                )
    return mass_maps, class_maps


def continuum_payload(record_set: RecordSet, family, mass: float) -> dict:
    """The analytic continuum-family payload at the node mass *mass*
    (no draws): the continuum state sits at rest at the node mass (the
    plan section 9.3 blend evaluated at the source mass), one allowed
    decay class is drawn uniformly over the family's
    ``allowed_decay_classes``, and the root split and daughter budget
    windows are integrated with the card-2 engine's own kinematics
    scheme (the ``RecordSet`` two-body object at depth 0).  Returns the
    card-9 payload shape ``{species: {"n", "p_sum", "e_sum",
    "ept_sum"}}`` with the isotropic ``E[sin theta] = pi/4`` factor
    applied once at the final aggregation (the
    ``RecordSet.record_content`` convention)."""
    share = 1.0 / len(family.allowed_decay_classes)
    class_pairs = [
        tuple(int(v) for v in cls.strip("()").split(","))
        for cls in family.allowed_decay_classes
    ]
    acc: dict[str, list] = {}
    out: dict[str, list] = {}
    for pair in class_pairs:
        record_set._two_body_object(
            out, None, pair, float(mass), share, 0
        )
    for skey, (count, nodes) in out.items():
        bucket = acc.get(skey)
        if bucket is None:
            bucket = [0.0, 0.0, 0.0, 0.0]
            acc[skey] = bucket
        weights = nodes[:, 0]
        bucket[0] += float(count)
        bucket[1] += float(np.sum(weights * nodes[:, 2]))
        bucket[2] += float(np.sum(weights * nodes[:, 1]))
        bucket[3] += float(np.sum(weights * nodes[:, 1] * nodes[:, 2]))
    return {
        skey: {
            "n": float(n),
            "p_sum": float(p),
            "e_sum": float(e),
            "ept_sum": float(ep) * ISOTROPIC_SIN_THETA_MEAN,
        }
        for skey, (n, p, e, ep) in acc.items()
    }


def _payload_observables(payload: dict, skey: str) -> dict:
    """The three checked observables of one payload block for *skey*:
    ``{"multiplicity", "mean_p", "mean_pt_y"}`` with ``None`` where the
    moment is undefined (the card-2 ``_observable_value`` convention)."""
    block = payload.get(skey)
    out = {"multiplicity": None, "mean_p": None, "mean_pt_y": None}
    if block is None or float(block["n"]) <= 0.0:
        out["multiplicity"] = 0.0
        return out
    out["multiplicity"] = float(block["n"])
    out["mean_p"] = float(block["p_sum"]) / float(block["n"])
    e_sum = float(block["e_sum"])
    if e_sum > 0.0:
        out["mean_pt_y"] = float(block["ept_sum"]) / e_sum
    return out


def _observable_of(payload: dict, skey: str, observable: str):
    values = _payload_observables(payload, skey)
    return values[observable]


# ---------------------------------------------------------------------------
# Per-decay generation (the worker context and the chunked loop)
# ---------------------------------------------------------------------------

#: Module-level worker context, set once in the parent before the fork
#: pool is created (the R01 ``_generate_chunk`` convention).  ``None``
#: in the parent main path.
_WORKER_CONTEXT: dict[str, Any] | None = None


def _set_worker_context(context: dict[str, Any]) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


class _Accumulator:
    """Per-event diagnostic accumulators of one sample (the card-9
    accumulator fields plus the routing counts).  All sums are
    accumulated in generation order, so the aggregated payload is
    bit-reproducible for the same event sequence."""

    def __init__(self, species_keys: tuple[str, ...]) -> None:
        self.n_events = 0
        self.count = {key: 0 for key in species_keys}
        self.count_sq = {key: 0 for key in species_keys}
        self.p_sum = {key: 0.0 for key in species_keys}
        self.e_sum = {key: 0.0 for key in species_keys}
        self.ept_sum = {key: 0.0 for key in species_keys}
        self.n_species = {key: 0 for key in species_keys}
        self.n_resonant = {key: 0 for key in species_keys}
        self.parent_counts: dict[str, dict[int, int]] = {
            key: {} for key in species_keys
        }
        self.pair_keys = tuple(name for name, _, _ in PAIR_DEFINITIONS)
        self.pair = {key: 0 for key in self.pair_keys}
        self.pair_sq = {key: 0 for key in self.pair_keys}
        self.n_catalog = 0
        self.n_continuum = 0

    def merge(self, other: "_Accumulator") -> None:
        """Merge *other* into ``self`` (exact integer counts; float sums
        in generation order)."""
        self.n_events += other.n_events
        self.n_catalog += other.n_catalog
        self.n_continuum += other.n_continuum
        for key in self.count:
            self.count[key] += other.count[key]
            self.count_sq[key] += other.count_sq[key]
            self.p_sum[key] += other.p_sum[key]
            self.e_sum[key] += other.e_sum[key]
            self.ept_sum[key] += other.ept_sum[key]
            self.n_species[key] += other.n_species[key]
            self.n_resonant[key] += other.n_resonant[key]
            for pdg, cnt in other.parent_counts[key].items():
                bucket = self.parent_counts[key]
                bucket[pdg] = bucket.get(pdg, 0) + cnt
        for key in self.pair:
            self.pair[key] += other.pair[key]
            self.pair_sq[key] += other.pair_sq[key]


def _mean_se(total: int, total_sq: int, n: int) -> tuple[float, float]:
    """Mean and standard error of the mean over *n* events (the sample
    standard deviation with the ``n - 1`` convention; 0.0 for fewer
    than two events)."""
    if n <= 0:
        return 0.0, 0.0
    mean = total / n
    if n < 2:
        return mean, 0.0
    variance = (total_sq - total * total / n) / (n - 1)
    return mean, math.sqrt(max(variance / n, 0.0))


def _decay_particles(
    catalog: HadronCatalog,
    qn: tuple[int, int, int],
    mass: float,
    replica: int,
    event_index: int,
    context: dict,
) -> tuple[str, list[tuple[int, tuple[float, float, float, float], int | None]]]:
    """One R31 decay of an at-rest source ``(qn, mass)``: the card-6
    routing and the pre-registered per-route realization.  Returns the
    route and the final particles as
    ``(pdg_id, four_momentum, parent_pdg_or_None)`` with the root
    parent the drawn record's pdg (catalog route) or the continuum
    sentinel 0 (continuum route)."""
    catalog_obj = context["catalog"]
    spec = context["spec"]
    fams = context["families"]
    mass_maps = context["mass_maps"]
    class_maps = context["class_maps"]
    stable = context["stable"]
    subs = derive_substreams(MASTER_SEED, replica, event_index)
    rng_record = np.random.default_rng(
        np.random.SeedSequence(subs.stream_for("decay_channel")).spawn(1)[0]
    )
    rng_mass = np.random.default_rng(
        np.random.SeedSequence(subs.stream_for("decay_kinematics")).spawn(1)[0]
    )
    decayer = ResonanceDecayer(catalog_obj, MASTER_SEED, replica, event_index)
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
            # The card-9 stable-identity semantics: the source
            # contributes itself as the single final particle at the
            # node four-momentum, with no producing vertex.
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
                    "staged_r31_matching_scan: record "
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
                "staged_r31_matching_scan: the catalog-driven decay of "
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
    # The continuum state sits at rest at the node mass (the plan
    # section 9.3 blend evaluated at the source mass: at mass M the
    # produced state is the continuum state at M; no family-mass
    # redraw).
    realized = float(mass)
    u_cls = float(rng_record.random())
    classes = family.allowed_decay_classes
    cls = classes[min(int(u_cls * n_classes), n_classes - 1)]
    d0, d1 = (int(v) for v in str(cls).strip("()").split(","))
    # The root split (the kernel's own two-body convention at rest):
    # the isotropic direction first (two uniforms), then the daughter
    # masses in channel order (each resonance daughter from the card-3
    # draw_mass inside its on-shell budget window with the partner at
    # its pole mass; each stable daughter at its pole mass).
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
    res0 = d0 not in stable
    res1 = d1 not in stable
    m0_pole = float(rec0.pole_mass)
    m1_pole = float(rec1.pole_mass)
    if res0:
        lo0 = float(rec0.mass_range[0])
        hi0 = min(float(rec0.mass_range[1]), realized - m1_pole)
        drawn0 = draw_mass((lo0, hi0), rec0, mass_maps[d0], rng_mass)
        if drawn0 is None:
            raise RuntimeError(
                "staged_r31_matching_scan: the continuum root daughter "
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
        drawn1 = draw_mass((lo1, hi1), rec1, mass_maps[d1], rng_mass)
        if drawn1 is None:
            raise RuntimeError(
                "staged_r31_matching_scan: the continuum root daughter "
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
            "staged_r31_matching_scan: the continuum decay of "
            f"{family.family_id} class {cls!r} at realized mass "
            f"{realized!r} produced no final particles (the run stops)"
        )
    return route, out


def _accumulate_event(
    acc: _Accumulator,
    catalog: HadronCatalog,
    species_index: dict[str, int],
    route: str,
    particles: list,
    source_qn: tuple[int, int, int],
) -> None:
    """One event into *acc*: the per-event ``(B, Q, S)`` balance against
    the source (checked, never repaired), the species counts, the
    momentum sums, the feed-down lineage, and the pair co-occurrence."""
    acc.n_events += 1
    if route == "catalog":
        acc.n_catalog += 1
    else:
        acc.n_continuum += 1
    event_pdg_counts: dict[int, int] = {}
    for pdg, _p4, _parent in particles:
        event_pdg_counts[pdg] = event_pdg_counts.get(pdg, 0) + 1
    total_b = total_q = total_s = 0
    for pdg, cnt in event_pdg_counts.items():
        b, q, s = _bqs_of(catalog, pdg)
        total_b += b * cnt
        total_q += q * cnt
        total_s += s * cnt
    if (total_b, total_q, total_s) != tuple(source_qn):
        raise RuntimeError(
            "staged_r31_matching_scan: an event's final particles sum to "
            f"(B, Q, S) = {(total_b, total_q, total_s)} but the source "
            f"carries {tuple(source_qn)}; conservation is checked, never "
            "repaired (the run stops)"
        )
    event_counts: dict[str, int] = {}
    for pdg, cnt in event_pdg_counts.items():
        key = _species_key(_record_name(catalog, pdg))
        event_counts[key] = event_counts.get(key, 0) + cnt
    for key, c in event_counts.items():
        acc.count[key] += c
        acc.count_sq[key] += c * c
    pair_counts = {
        name: min(event_pdg_counts.get(pdga, 0),
                  event_pdg_counts.get(pdgb, 0))
        for name, pdga, pdgb in PAIR_DEFINITIONS
    }
    for name in acc.pair_keys:
        v = pair_counts[name]
        acc.pair[name] += v
        acc.pair_sq[name] += v * v
    for pdg, p4, parent_pdg in particles:
        key = _species_key(_record_name(catalog, pdg))
        e, px, py, pz = p4
        acc.n_species[key] += 1
        p = math.sqrt(px * px + py * py + pz * pz)
        acc.p_sum[key] += p
        acc.e_sum[key] += e
        acc.ept_sum[key] += e * math.hypot(px, py)
        if parent_pdg is not None and parent_pdg != CONTINUUM_ROOT_SENTINEL:
            acc.n_resonant[key] += 1
            bucket = acc.parent_counts[key]
            bucket[int(parent_pdg)] = bucket.get(int(parent_pdg), 0) + 1


def _r31_generate_chunk(task: tuple):
    """Generate one chunk of decays in a worker process and accumulate
    it.  Returns ``(chunk_start, acc)``."""
    assert _WORKER_CONTEXT is not None
    window, node, replica, start, stop = task
    context = _WORKER_CONTEXT
    catalog = context["catalog"]
    acc = _Accumulator(context["species_keys"])
    qn = tuple(int(v) for v in WINDOWS[window]["qn"])
    mass = float(NODE_MASSES[node])
    base = (window * N_NODES + node) * 1_000_000
    for offset in range(start, stop):
        event_index = base + offset
        route, particles = _decay_particles(
            catalog, qn, mass, replica, event_index, context
        )
        _accumulate_event(acc, catalog, context["species_index"], route,
                          particles, qn)
    return start, acc


def _map_chunks(tasks: list, workers: int, progress_label: str):
    """Yield worker results as they complete over a fork worker pool."""
    context = multiprocessing.get_context("fork")
    completed = 0
    started = time.perf_counter()
    with context.Pool(processes=workers) as pool:
        for result in pool.imap_unordered(
            _r31_generate_chunk, tasks, chunksize=1
        ):
            completed += 1
            yield result
            if completed % 4 == 0 or completed == len(tasks):
                elapsed = time.perf_counter() - started
                print(
                    f"[r31] {progress_label}: chunk {completed}/"
                    f"{len(tasks)} ({elapsed:.1f}s)",
                    flush=True,
                )


def _diagnostics_payload(
    acc: _Accumulator,
    window: int,
    node: int,
    mass: float,
    source_qn: tuple[int, int, int],
    open_records: list[int],
    mode: str,
) -> dict:
    """The card-9 per-point output schema for one accumulated sample,
    with the R31 window/node identity and the routing block."""
    n = acc.n_events
    if n <= 0:
        raise RuntimeError(
            "staged_r31_matching_scan: an empty event sample cannot "
            "produce the per-node diagnostics"
        )
    multiplicities: dict[str, dict[str, float]] = {}
    for key in acc.count:
        mean, se = _mean_se(acc.count[key], acc.count_sq[key], n)
        multiplicities[key] = {"mean": mean, "se": se}
    momentum_moments: dict[str, dict] = {}
    feed_down: dict[str, dict] = {}
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
    pairs: dict[str, dict[str, float]] = {}
    for name in acc.pair_keys:
        mean, se = _mean_se(acc.pair[name], acc.pair_sq[name], n)
        pairs[name] = {"mean": mean, "se": se}
    n_decays = n
    return {
        "schema": "na49_mixing.staged_r31_matching_scan.diagnostics/1",
        "run": RUN_ID,
        "card": CARD,
        "mode": mode,
        "window": int(window),
        "node": int(node),
        "point": int(node),
        "source_mass_gev": float(mass),
        "source_qn": [int(q) for q in source_qn],
        "open_records": [int(p) for p in open_records],
        "n_decays": int(n_decays),
        "master_seed": int(MASTER_SEED),
        "seed_rule": SEED_RULE,
        "routing": {
            "n_catalog": int(acc.n_catalog),
            "n_continuum": int(acc.n_continuum),
            "fraction_continuum": (
                float(acc.n_continuum) / float(n) if n > 0 else 0.0
            ),
            "expected_w_at_node": float(EXPECTED_W[node]),
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


def _build_context(
    catalog: HadronCatalog,
    families: list,
    pre_mass_maps: tuple[dict, dict] | None = None,
) -> dict:
    """The worker context: the committed inputs, the per-node open
    sets, the per-window family, and the precomputed kernel maps."""
    mass_maps, class_maps = pre_mass_maps or _precompute_kernel_maps(catalog)
    open_sets: dict[tuple, dict[float, list]] = {}
    family_of: dict[tuple, Any] = {}
    n_classes_of: dict[tuple, int] = {}
    for spec_window in WINDOWS:
        qn = tuple(int(v) for v in spec_window["qn"])
        actives = active_families(families, qn)
        if len(actives) != 1:
            _refuse(
                f"window {spec_window['window']} expects exactly one "
                f"activated family for {qn}, found "
                f"{[f.family_id for f in actives]}"
            )
        expected = spec_window["family_expectation"]
        if actives[0].family_id != expected:
            _refuse(
                f"window {spec_window['window']}: the activated family "
                f"{actives[0].family_id!r} does not match the "
                f"pre-registered {expected!r}"
            )
        family_of[qn] = actives[0]
        n_classes_of[qn] = len(actives[0].allowed_decay_classes)
        per_mass: dict[float, list] = {}
        for mass in NODE_MASSES:
            records = _open_set(catalog, qn, mass)
            if not records:
                _refuse(
                    f"the open set at {qn} and node mass {mass!r} is "
                    "empty; the pre-registered mixture model has no "
                    "record to draw and no fallback path is implemented"
                )
            per_mass[float(mass)] = records
        open_sets[qn] = per_mass
    species_keys = _stable_species_keys(catalog)
    return {
        "catalog": catalog,
        "families": families,
        "spec": SPEC,
        "mass_maps": mass_maps,
        "class_maps": class_maps,
        "stable": frozenset(catalog.stable_states()),
        "open_sets": open_sets,
        "family_of": family_of,
        "n_classes_of": n_classes_of,
        "species_keys": species_keys,
        "species_index": {key: i for i, key in enumerate(species_keys)},
    }


def _generate_node(
    run_dir: Path,
    window: int,
    node: int,
    decays_per_replica: int,
    replicas: int,
    workers: int,
    context: dict,
    mode: str,
) -> dict:
    """Generate one (window, node) sample: one
    ``window_<w>_node_<j>_replica_<r>.json`` per replica with the raw
    per-replica diagnostics and one ``window_<w>_node_<j>.json`` with
    the replica-merged aggregates.  A completed node record that
    matches the current inputs is reused unchanged."""
    run_dir = Path(run_dir)
    node_name = f"window_{window}_node_{node}.json"
    node_path = run_dir / node_name
    qn = tuple(int(v) for v in WINDOWS[window]["qn"])
    mass = float(NODE_MASSES[node])
    open_records = context["open_sets"][qn][mass]
    record_pdgs = [int(r.pdg_id) for r in open_records]
    if node_path.is_file() and mode == "production":
        try:
            stored = _read_json(node_path)
        except ValueError:
            stored = None
        if (
            isinstance(stored, dict)
            and int(stored.get("window", -1)) == int(window)
            and int(stored.get("node", -1)) == int(node)
            and int(stored.get("n_decays", -1))
            == decays_per_replica * replicas
            and stored.get("mode") == "production"
            and stored.get("master_seed") == MASTER_SEED
            and stored.get("open_records") == record_pdgs
        ):
            stored["reused"] = True
            print(
                f"[r31] window {window} node {node}: reusing completed "
                f"record ({int(stored['n_decays'])} decays)",
                flush=True,
            )
            return stored

    species_keys = context["species_keys"]
    node_acc = _Accumulator(species_keys)
    replica_accs: list[_Accumulator] = []
    _set_worker_context(context)
    for replica in range(replicas):
        acc = _Accumulator(species_keys)
        chunks = [
            (window, node, replica, start,
             min(start + CHUNK_SIZE, decays_per_replica))
            for start in range(0, decays_per_replica, CHUNK_SIZE)
        ]
        parts = sorted(
            _map_chunks(
                chunks, workers,
                f"window {window} node {node} replica {replica}",
            ),
            key=lambda part: part[0],
        )
        for _start, part_acc in parts:
            acc.merge(part_acc)
        _write_json(
            run_dir / f"window_{window}_node_{node}_replica_{replica}.json",
            _diagnostics_payload(
                acc, window, node, mass, qn, record_pdgs, mode
            ),
        )
        replica_accs.append(acc)
        node_acc.merge(acc)
    record = _diagnostics_payload(
        node_acc, window, node, mass, qn, record_pdgs, mode
    )
    record["replicas"] = int(replicas)
    record["decays_per_replica"] = int(decays_per_replica)
    _write_json(node_path, record)
    return record


# ---------------------------------------------------------------------------
# Mode: --pre-registration
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
        "driver_sha256": _sha256_file(Path(__file__).resolve()),
    }


def build_pre_registration(catalog: HadronCatalog, families: list) -> dict:
    """The R31 pre-registration payload, written before any generation."""
    context = _build_context(catalog, families)
    open_sets_payload = {}
    for spec_window in WINDOWS:
        qn = tuple(int(v) for v in spec_window["qn"])
        open_sets_payload[str(spec_window["window"])] = {
            str(j): [int(r.pdg_id) for r in context["open_sets"][qn][mass]]
            for j, mass in enumerate(NODE_MASSES)
        }
    return {
        "schema": "na49_mixing.staged_r31_matching_scan.pre_registration/1",
        "run": RUN_ID,
        "card": CARD,
        "written_before_generation": True,
        "windows": [
            {
                "window": int(spec_window["window"]),
                "qn": [int(v) for v in spec_window["qn"]],
                "label": spec_window["label"],
                "family_expectation": spec_window["family_expectation"],
            }
            for spec_window in WINDOWS
        ],
        "nodes": {
            "masses_gev": [float(M) for M in NODE_MASSES],
            "rule": "M_j = 2.48 + 0.1 j, j = 0..4, at rest (p = 0, "
                    "E = M_j); the isolated source decay, the R21/R22 "
                    "convention (round 2 index item 10)",
            "boundary_gev": BOUNDARY,
            "straddle_note": (
                "node 2.68 lies below the boundary B by "
                f"{BOUNDARY - 2.68!r} GeV and node 2.78 lies above it by "
                f"{2.78 - BOUNDARY!r} GeV; the boundary-straddling pair "
                "is (j = 2, j+1 = 3) in every window"
            ),
        },
        "statistics": {
            "tier": "T1",
            "label": T1_LABEL,
            "replicas": T1_REPLICAS,
            "decays_per_replica": T1_DECAYS_PER_REPLICA,
            "windows": N_WINDOWS,
            "nodes": N_NODES,
            "total_decays": TOTAL_DECAYS,
        },
        "seed_map": {
            "master_seed": MASTER_SEED,
            "rule": SEED_RULE,
            "substream_labels": list(SEED_LABELS),
            "disjointness_note": (
                "the R31 seeds are disjoint from the R20-R23 seeds "
                "(R20-R22 derive substreams from master 20260913; R23 "
                "uses raw replica seeds 20300913..20300920)"
            ),
        },
        "decay_routing": {
            "source": "card 6 (round 3 index item 4), applied per grid "
                      "node",
            "boundary_gev": BOUNDARY,
            "ramp_width_gev": RAMP_WIDTH,
            "expected_w_at_nodes": [float(w) for w in EXPECTED_W],
            "expected_w_note": (
                "w(M) = 0 at and below B, (M - B)/W on the transition "
                "band, 1 at and above B + W (the card-6 BlendSpec ramp); "
                "w(2.48) = w(2.58) = w(2.68) = 0 exactly (node 2.68 is "
                "below B), w(2.78) = 0.199611778544944, "
                "w(2.88) = 0.399611778544944"
            ),
            "routing_draw": (
                "one uniform on the spawned decay_channel child iff "
                "B <= M_j < B + W, consumed before any record draw; "
                "zero routing uniforms otherwise"
            ),
        },
        "realization": {
            "catalog_route": (
                "the card-9 mixture convention at the node mass: the "
                "open set at M_j, the R23 builder's record draw "
                "expression on the spawned decay_channel child, the "
                "card-3 draw_mass on the record support truncated at "
                "M_j on the spawned decay_kinematics child (a "
                "degenerate window fixes the mass at M_j), the "
                "stable-identity final at (M_j, 0, 0, 0) for a drawn "
                "stable record, otherwise the Round 2 kernel "
                "decay_source at the at-rest four-momentum of the "
                "realized mass"
            ),
            "continuum_route": (
                "the family-mixture analogue: the single activated "
                "family per window (asserted), the continuum state at "
                "rest at the node mass (the plan section 9.3 blend "
                "evaluated at the source mass: at mass M the produced "
                "state is the continuum state at M; no family-mass "
                "redraw), one allowed class uniform over the family's "
                "allowed_decay_classes (one uniform on the spawned "
                "decay_channel child; equal class weights, the card-9 "
                "flat-mixture rule over the pre-registered class "
                "list), the root split in the kernel's own two-body "
                "convention at rest at the node mass (isotropic "
                "direction, budget-window daughter masses in channel "
                "order, the Kaellen decay momentum), and the resonance "
                "daughters' cascades through the Round 2 kernel "
                "decay_resonance on the labeled substreams"
            ),
            "truncation_note": (
                "the catalog route realizes the record mass inside the "
                "node-mass-truncated record window (the card-9 "
                "convention the card-2 metric engine integrates) and "
                "the continuum route decays the continuum state at the "
                "node mass exactly, so the node mass stays the source "
                "mass of the event on both routes and the continuum "
                "contribution is a smooth function of the source mass, "
                "which is what the Gate 3 smoothness clause tests"
            ),
            "kernel_caches_note": (
                "the kernel's daughter-mass maps and card-5 class maps "
                "are precomputed once (pure compute caches) and primed "
                "into every per-decay ResonanceDecayer unchanged; no "
                "draw depends on them"
            ),
        },
        "discontinuity_metric": {
            "id": METRIC_ID,
            "engine_catalog": (
                "na49_mixing.threshold_metric.full_analytic_difference "
                "(the card-2 re-registered metric; the card-9 flat "
                "record mixture payload of RecordSet.mixture_payload)"
            ),
            "engine_continuum": (
                "the driver's continuum analytic engine "
                "(continuum_payload): the continuum state at rest at "
                "the node mass, uniform class shares over the family's "
                "allowed classes, and the root split and daughter "
                "budget windows integrated with the card-2 engine's "
                "own kinematics scheme (RecordSet._two_body_object at "
                "depth 0), no Monte Carlo"
            ),
            "routed_expectation": (
                "mu_routed(M) = (1 - w(M)) mu_catalog(M) + w(M) "
                "mu_continuum(M); delta_analytic_full is the routed "
                "difference between the two node masses; at pairs with "
                "w = 0 at both nodes it reduces exactly to the card-2 "
                "full_analytic_difference value (applied verbatim, with "
                "its open-set-change and momentum-drift decomposition)"
            ),
            "statement": (
                "for every adjacent node pair (j, j+1) in each window "
                "and every species, with SE_combined = sqrt(SE_j^2 + "
                "SE_{j+1}^2): |observed - delta_analytic_full| <= "
                "3 SE_combined on the species multiplicity and the "
                "per-species momentum moments <p> and <p_T>_y; the "
                "boundary-straddling pair (j = 2) is checked like any "
                "other pair, no special-casing"
            ),
            "tolerance_sigma": METRIC_N_SIGMA,
            "se_conventions": {
                "multiplicity": (
                    "the per-node pooled standard error of the mean "
                    "(the card-9 stored 'se')"
                ),
                "moments": (
                    "the between-replica standard error of the four "
                    "stored replica moments (std, ddof=1, over sqrt(4))"
                ),
            },
            "moment_presence_rule": (
                "a moment check applies only where the species is "
                "present at both nodes of the pair (absent_species "
                "otherwise; the multiplicity check always applies)"
            ),
            "boundary_straddling_pairs": BOUNDARY_PAIR_INDICES,
        },
        "balance": {
            "statement": (
                "every event conserves the source (B, Q, S) exactly at "
                "every vertex (the Round 2 kernel conservation, checked "
                "per event, never repaired)"
            ),
        },
        "open_sets": open_sets_payload,
        "output_schema": {
            "node_file": "window_<w>_node_<j>.json",
            "replica_file": "window_<w>_node_<j>_replica_<r>.json",
            "keys": [
                "window", "node", "point", "source_mass_gev", "source_qn",
                "open_records", "n_decays", "routing", "multiplicities",
                "momentum_moments", "feed_down", "charge_balance", "pairs",
            ],
            "multiplicity_species": (
                "the catalog stable termination set, all listed, in "
                "catalog record order"
            ),
            "mean_pt_y_convention": (
                "the density-p_T f convention (<p_T>_y): sum(E * pT) / "
                "sum(E) over the species' particles"
            ),
            "pairs_convention": (
                "the charge-conjugate co-occurrence min(n, nbar) per "
                "event, mean and standard error over the node's events"
            ),
            "feed_down_lineage": (
                "a final particle is resonant feed-down when its "
                "producing vertex's parent is a resonance other than "
                "the drawn root (the drawn record on the catalog route, "
                "the continuum-source sentinel 0 on the continuum "
                "route); the root's direct daughters are direct source "
                "products (the corrected lineage convention of Round 3 "
                "card 3)"
            ),
        },
        "no_fitting": True,
        "optimizer_invoked": False,
        "no_fitting_note": (
            "no optimizer is invoked; a failing match re-opens the "
            "card-6 routing or the originating family card, never the "
            "3-sigma tolerance"
        ),
        "input_hashes": _input_hashes(),
    }


def run_pre_registration(run_dir: Path) -> None:
    """Write ``pre_registration.json`` and ``provenance.json``, stop."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    pre_path = run_dir / PREREGISTRATION_NAME
    if pre_path.is_file() and any(run_dir.glob("window_*_node_*.json")):
        _refuse(
            f"{pre_path} exists and node records are present; the "
            "pre-registration is immutable after generation has started "
            "(a change invalidates the run; re-seal under a new date)"
        )
    catalog = HadronCatalog.from_json(NESTED_ROOT / CATALOG_REL)
    families = families_from_json(str(NESTED_ROOT / FAMILIES_REL))
    pre = build_pre_registration(catalog, families)
    _write_json(pre_path, pre)
    hashes = _input_hashes()
    provenance = {
        "schema": "na49_mixing.staged_r31_matching_scan.provenance/1",
        "run": RUN_ID,
        "card": CARD,
        "card_chain": list(CARD_CHAIN),
        "driver": DRIVER_REL,
        "driver_sha256": hashes["driver_sha256"],
        "git": _git_info(),
        "environment": environment_fingerprint(),
        "written_utc": _now_utc(),
        "written_before_generation": True,
        "catalog": {
            "path": CATALOG_REL,
            "sha256": hashes["catalog_sha256"],
        },
        "continuum_config": {
            "path": FAMILIES_REL,
            "sha256": hashes["families_config_sha256"],
        },
        "modules": {
            "spectral_density_sha256": hashes[
                "spectral_density_module_sha256"
            ],
            "threshold_metric_sha256": hashes[
                "threshold_metric_module_sha256"
            ],
            "resonance_decay_sha256": hashes[
                "resonance_decay_module_sha256"
            ],
            "resonance_widths_sha256": hashes[
                "resonance_widths_module_sha256"
            ],
        },
        "kernel_boundary_gev": KERNEL_BOUNDARY,
        "blend": {
            "boundary_gev": BOUNDARY,
            "ramp_width_gev": RAMP_WIDTH,
        },
        "input_hashes": hashes,
    }
    _write_json(run_dir / PROVENANCE_NAME, provenance)
    print(
        "staged_r31_matching_scan: pre-registration written to "
        f"{pre_path} (3 windows x 5 nodes x 4 replicas x 25,000 decays "
        f"= {TOTAL_DECAYS}, master seed {MASTER_SEED}, blend boundary "
        f"{BOUNDARY!r} GeV, expected w at the nodes "
        f"{[round(w, 12) for w in EXPECTED_W]}); nothing generated"
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
    if int(stats.get("replicas", -1)) != T1_REPLICAS:
        problems.append("statistics.replicas mismatch")
    if int(stats.get("decays_per_replica", -1)) != T1_DECAYS_PER_REPLICA:
        problems.append("statistics.decays_per_replica mismatch")
    if int(stats.get("windows", -1)) != N_WINDOWS:
        problems.append("statistics.windows mismatch")
    if int(stats.get("nodes", -1)) != N_NODES:
        problems.append("statistics.nodes mismatch")
    if int(stats.get("total_decays", -1)) != TOTAL_DECAYS:
        problems.append("statistics.total_decays mismatch")
    if [float(M) for M in pre.get("nodes", {}).get("masses_gev", [])] != [
            float(M) for M in NODE_MASSES]:
        problems.append("node masses mismatch")
    if float(pre.get("nodes", {}).get("boundary_gev", -1.0)) != BOUNDARY:
        problems.append("blend boundary mismatch")
    if [float(w) for w in pre.get("decay_routing", {}).get(
            "expected_w_at_nodes", [])] != [float(w) for w in EXPECTED_W]:
        problems.append("expected routing weights mismatch")
    if str(pre.get("seed_map", {}).get("rule", "")) != SEED_RULE:
        problems.append("seed rule mismatch")
    if int(pre.get("seed_map", {}).get("master_seed", -1)) != MASTER_SEED:
        problems.append("master seed mismatch")
    if pre.get("discontinuity_metric", {}).get("id") != METRIC_ID:
        problems.append("metric id mismatch (the re-registered card-2 "
                        "metric is the contract)")
    if float(pre.get("discontinuity_metric", {}).get(
            "tolerance_sigma", -1.0)) != METRIC_N_SIGMA:
        problems.append("metric tolerance mismatch")
    if [int(v) for v in pre.get("discontinuity_metric", {}).get(
            "boundary_straddling_pairs", [])] != [
            int(v) for v in BOUNDARY_PAIR_INDICES]:
        problems.append("boundary_straddling_pairs mismatch")
    if pre.get("no_fitting") is not True:
        problems.append("no_fitting mismatch")
    windows = pre.get("windows", [])
    if [tuple(int(v) for v in w.get("qn", ())) for w in windows] != [
            tuple(int(v) for v in w["qn"]) for w in WINDOWS]:
        problems.append("window quantum numbers mismatch")
    if problems:
        raise RuntimeError(
            "staged_r31_matching_scan: the written pre-registration does "
            "not match this driver (the run is invalidated):\n  "
            + "\n  ".join(problems)
        )
    return pre


# ---------------------------------------------------------------------------
# Mode: --generate
# ---------------------------------------------------------------------------


def run_generation(
    run_dir: Path,
    window: int | None,
    node: int | None,
    workers: int,
    smoke: bool,
    smoke_decays: int,
    smoke_replicas: int,
    statistics: dict,
) -> None:
    """Generate the requested slice: one node, one window, or every
    window; each node's artifacts land on disk before the next."""
    run_dir = Path(run_dir)
    catalog = HadronCatalog.from_json(NESTED_ROOT / CATALOG_REL)
    families = families_from_json(str(NESTED_ROOT / FAMILIES_REL))
    context = _build_context(
        catalog, families, _precompute_kernel_maps(catalog)
    )
    if smoke:
        decays = int(smoke_decays)
        replicas = int(smoke_replicas)
        mode = "smoke"
    else:
        pre = _assert_pre_registration(run_dir)
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
        declared = {
            "windows": N_WINDOWS,
            "nodes": N_NODES,
            "decays_per_replica": T1_DECAYS_PER_REPLICA,
            "replicas": T1_REPLICAS,
            "master_seed": MASTER_SEED,
        }
        for key, value in declared.items():
            given = statistics.get(key)
            if given is not None and int(given) != int(value):
                _refuse(
                    f"--{key} {given} does not match the pre-registered "
                    f"{value}; the production statistics are fixed"
                )
        decays = T1_DECAYS_PER_REPLICA
        replicas = T1_REPLICAS
        mode = "production"
    if window is None:
        windows = list(range(N_WINDOWS))
    else:
        if not 0 <= int(window) < N_WINDOWS:
            _refuse(
                f"--window {window} outside the pre-registered set "
                f"0..{N_WINDOWS - 1}"
            )
        windows = [int(window)]
    for w in windows:
        if node is None:
            nodes = list(range(N_NODES))
        else:
            if not 0 <= int(node) < N_NODES:
                _refuse(
                    f"--node {node} outside the pre-registered set "
                    f"0..{N_NODES - 1}"
                )
            nodes = [int(node)]
        for j in nodes:
            record = _generate_node(
                run_dir, w, j, decays, replicas, int(workers), context, mode
            )
            routing = record["routing"]
            print(
                f"[r31] window {w} node {j} (M = {NODE_MASSES[j]:.2f} "
                f"GeV): {record['n_decays']} decays, routing "
                f"catalog {routing['n_catalog']} / continuum "
                f"{routing['n_continuum']} "
                f"(fraction {routing['fraction_continuum']:.6f}, "
                f"expected w = {routing['expected_w_at_node']:.12f})",
                flush=True,
            )
    print(
        f"[r31] generation step complete (run dir {run_dir}, mode "
        f"{mode})",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Mode: --verify (the matching verification)
# ---------------------------------------------------------------------------


def _load_node_records(run_dir: Path, pre: dict) -> dict:
    """The fifteen production node records and their replica records,
    each asserted against the pre-registration."""
    nodes: dict[tuple[int, int], dict] = {}
    replicas: dict[tuple[int, int], list[dict]] = {}
    for spec_window in WINDOWS:
        w = int(spec_window["window"])
        for j in range(N_NODES):
            path = Path(run_dir) / f"window_{w}_node_{j}.json"
            if not path.is_file():
                _refuse(
                    f"node record missing: {path}; --verify requires all "
                    f"{N_WINDOWS * N_NODES} production nodes"
                )
            record = _read_json(path)
            if record.get("mode") != "production":
                _refuse(f"{path} is not a production record")
            if int(record.get("window", -1)) != w or int(
                    record.get("node", -1)) != j:
                _refuse(f"{path} carries the wrong window/node identity")
            if int(record.get("n_decays", -1)) != (
                    T1_REPLICAS * T1_DECAYS_PER_REPLICA):
                _refuse(f"{path} carries the wrong event count")
            if int(record.get("master_seed", -1)) != MASTER_SEED:
                _refuse(f"{path} carries the wrong master seed")
            replica_records = []
            for r in range(T1_REPLICAS):
                rpath = Path(run_dir) / (
                    f"window_{w}_node_{j}_replica_{r}.json"
                )
                if not rpath.is_file():
                    _refuse(f"replica record missing: {rpath}")
                rrec = _read_json(rpath)
                if rrec.get("mode") != "production":
                    _refuse(f"{rpath} is not a production record")
                if int(rrec.get("n_decays", -1)) != T1_DECAYS_PER_REPLICA:
                    _refuse(f"{rpath} carries the wrong event count")
                replica_records.append(rrec)
            nodes[(w, j)] = record
            replicas[(w, j)] = replica_records
    del pre
    return {"nodes": nodes, "replicas": replicas}


def _moment_se(replica_records: list[dict], skey: str, field: str) -> float:
    """The between-replica standard error of one moment: the sample
    standard deviation (ddof=1) of the four stored replica values over
    sqrt(4) (the R21/R22 convention).  ``None`` when any replica lacks
    the species (the moment is undefined there)."""
    values = []
    for record in replica_records:
        block = record.get("momentum_moments", {}).get(skey)
        if block is None:
            return None
        values.append(float(block[field]))
    if len(values) != T1_REPLICAS:
        return None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(max(variance, 0.0) / len(values))


def _routed_payloads(
    window: int,
    catalog: HadronCatalog,
    families: list,
    record_set_cache: dict,
) -> tuple[dict, dict, dict, dict]:
    """The analytic payloads of one window: the catalog mixture payload
    at every node mass, the continuum payload at every node mass with a
    positive routing weight, and the blend weights.  Returns
    ``(mu_catalog, mu_continuum, weights, record_set)``."""
    qn = tuple(int(v) for v in WINDOWS[window]["qn"])
    open_records = _open_set(catalog, qn, NODE_MASSES[0])
    if not open_records:
        _refuse(f"the open set of window {window} is empty")
    union: dict[int, Any] = {}
    for mass in NODE_MASSES:
        for record in _open_set(catalog, qn, mass):
            union.setdefault(int(record.pdg_id), record)
    record_set = record_set_cache.get(window)
    if record_set is None:
        record_set = RecordSet(
            catalog, [union[pdg] for pdg in sorted(union)]
        )
        record_set_cache[window] = record_set
    mu_catalog: dict[int, dict] = {}
    for j, mass in enumerate(NODE_MASSES):
        mu_catalog[j] = record_set.mixture_payload(
            record_set.open_records(float(mass)), float(mass)
        )
    weights = {j: float(SPEC.weight(M)) for j, M in enumerate(NODE_MASSES)}
    mu_continuum: dict[int, dict] = {}
    family = active_families(families, qn)[0]
    for j, mass in enumerate(NODE_MASSES):
        if weights[j] <= 0.0:
            continue
        mu_continuum[j] = continuum_payload(record_set, family, float(mass))
    return mu_catalog, mu_continuum, weights, record_set


def _routed_observable(
    mu_catalog: dict, mu_continuum: dict, weights: dict, node: int,
    skey: str, observable: str,
):
    """One observable of the routed flat-mixture expectation at *node*:
    the blended payload block first, then the observable (``None``
    where the blended moment is undefined)."""
    w = weights[node]
    cat_block = mu_catalog[node].get(skey)
    cont_block = mu_continuum.get(node, {}).get(skey)
    if cat_block is None and (cont_block is None or w <= 0.0):
        block = None
    else:
        zero = {"n": 0.0, "p_sum": 0.0, "e_sum": 0.0, "ept_sum": 0.0}
        cat_block = zero if cat_block is None else cat_block
        cont_block = zero if cont_block is None else cont_block
        block = {
            field: (1.0 - w) * float(cat_block[field])
            + w * float(cont_block[field])
            for field in ("n", "p_sum", "e_sum", "ept_sum")
        }
    if block is None:
        return 0.0 if observable == "multiplicity" else None
    if observable == "multiplicity":
        return float(block["n"])
    if observable == "mean_p":
        if not float(block["n"]) > 0.0:
            return None
        return float(block["p_sum"]) / float(block["n"])
    if observable == "mean_pt_y":
        if not float(block["e_sum"]) > 0.0:
            return None
        return float(block["ept_sum"]) / float(block["e_sum"])
    raise ValueError(f"unknown observable {observable!r}")


OBSERVABLES = ("multiplicity", "mean_p", "mean_pt_y")


def run_verification(run_dir: Path) -> dict:
    """The pre-registered matching verification: the re-registered
    metric over all four adjacent pairs per window (twelve total) and
    all species, the boundary-straddling pair included, no
    special-casing."""
    run_dir = Path(run_dir)
    pre = _assert_pre_registration(run_dir)
    loaded = _load_node_records(run_dir, pre)
    nodes = loaded["nodes"]
    replicas = loaded["replicas"]
    catalog = HadronCatalog.from_json(NESTED_ROOT / CATALOG_REL)
    families = families_from_json(str(NESTED_ROOT / FAMILIES_REL))
    species_keys = _stable_species_keys(catalog)
    record_set_cache: dict[int, RecordSet] = {}

    windows_payload: dict[str, dict] = {}
    total_applied = 0
    total_failed = 0
    max_abs_sigma = 0.0
    boundary_max = 0.0
    interior_max = 0.0
    boundary_failures = 0
    boundary_applied = 0
    interior_failures = 0
    interior_applied = 0
    per_window_rates = []
    for spec_window in WINDOWS:
        w = int(spec_window["window"])
        mu_catalog, mu_continuum, weights, record_set = _routed_payloads(
            w, catalog, families, record_set_cache
        )
        pairs_payload = {}
        w_boundary_fail = w_boundary_applied = 0
        w_interior_fail = w_interior_applied = 0
        for j in range(N_NODES - 1):
            global_pair = 4 * w + j
            masses = (float(NODE_MASSES[j]), float(NODE_MASSES[j + 1]))
            w_j, w_jp1 = weights[j], weights[j + 1]
            catalog_only = (w_j <= 0.0) and (w_jp1 <= 0.0)
            # The card-2 catalog-side delta (the contract at
            # catalog-only pairs; the catalog block at blend pairs).
            catalog_delta = {}
            open_set_change = {}
            momentum_drift = {}
            for skey in species_keys:
                full = full_analytic_difference(
                    record_set, masses[0], masses[1], skey
                )
                catalog_delta[skey] = full["delta_analytic_full"]
                open_set_change[skey] = full["open_set_change"]
                momentum_drift[skey] = full["momentum_drift"]
            pair_rows = []
            for skey in species_keys:
                observed_j = {}
                observed_jp1 = {}
                se = {}
                node_j = nodes[(w, j)]
                node_jp1 = nodes[(w, j + 1)]
                mult_j = node_j["multiplicities"][skey]
                mult_jp1 = node_jp1["multiplicities"][skey]
                observed_j["multiplicity"] = float(mult_j["mean"])
                observed_jp1["multiplicity"] = float(mult_jp1["mean"])
                se["multiplicity"] = math.sqrt(
                    float(mult_j["se"]) ** 2 + float(mult_jp1["se"]) ** 2
                )
                present_j = _moment_se(
                    replicas[(w, j)], skey, "mean_p_gev"
                )
                present_jp1 = _moment_se(
                    replicas[(w, j + 1)], skey, "mean_p_gev"
                )
                obs_ok = present_j is not None and present_jp1 is not None
                if obs_ok:
                    block_j = node_j["momentum_moments"][skey]
                    block_jp1 = node_jp1["momentum_moments"][skey]
                    observed_j["mean_p"] = float(block_j["mean_p_gev"])
                    observed_jp1["mean_p"] = float(block_jp1["mean_p_gev"])
                    se["mean_p"] = math.sqrt(
                        float(present_j) ** 2 + float(present_jp1) ** 2
                    )
                    pt_j = _moment_se(
                        replicas[(w, j)], skey, "mean_pt_y_gev"
                    )
                    pt_jp1 = _moment_se(
                        replicas[(w, j + 1)], skey, "mean_pt_y_gev"
                    )
                    if (
                        pt_j is not None
                        and pt_jp1 is not None
                        and int(block_j["n_particles"]) > 0
                        and int(block_jp1["n_particles"]) > 0
                    ):
                        observed_j["mean_pt_y"] = float(
                            block_j["mean_pt_y_gev"]
                        )
                        observed_jp1["mean_pt_y"] = float(
                            block_jp1["mean_pt_y_gev"]
                        )
                        se["mean_pt_y"] = math.sqrt(
                            float(pt_j) ** 2 + float(pt_jp1) ** 2
                        )
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
                    observed = (
                        observed_jp1[observable] - observed_j[observable]
                    )
                    if catalog_only:
                        catalog_value = catalog_delta[skey][observable]
                        if catalog_value is None:
                            # The card-2 analytic moment is undefined at
                            # one node of the pair (the species' moment
                            # is absent there); the check does not
                            # apply.
                            pair_rows.append({
                                "pair": [int(j), int(j + 1)],
                                "global_pair": int(global_pair),
                                "species": skey,
                                "observable": observable,
                                "status": "absent_species_analytic",
                                "pass": None,
                            })
                            continue
                        delta_full = float(catalog_value)
                        components = {
                            "open_set_change": open_set_change[skey][
                                observable],
                            "momentum_drift": momentum_drift[skey][
                                observable],
                            "delta_catalog": float(catalog_value),
                            "catalog_block_payload": None,
                            "continuum_block_payload": None,
                            "w_at_nodes": [float(w_j), float(w_jp1)],
                        }
                    else:
                        ana_j = _routed_observable(
                            mu_catalog, mu_continuum, weights, j, skey,
                            observable,
                        )
                        ana_jp1 = _routed_observable(
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
                        # The routed decomposition at the payload level:
                        # the catalog block is the weighted catalog
                        # payload difference, the continuum block the
                        # weighted continuum payload difference; the
                        # two blocks add to the routed payload exactly.
                        zero = {"n": 0.0, "p_sum": 0.0, "e_sum": 0.0,
                                "ept_sum": 0.0}
                        cat_blk_j = mu_catalog[j].get(skey) or zero
                        cat_blk_jp1 = mu_catalog[j + 1].get(skey) or zero
                        cont_blk_j = mu_continuum.get(j, {}).get(skey) \
                            or zero
                        cont_blk_jp1 = mu_continuum.get(j + 1, {}).get(
                            skey) or zero
                        fields = ("n", "p_sum", "e_sum", "ept_sum")
                        catalog_block_payload = {
                            field: (
                                (1.0 - w_jp1) * float(cat_blk_jp1[field])
                                - (1.0 - w_j) * float(cat_blk_j[field])
                            )
                            for field in fields
                        }
                        continuum_block_payload = {
                            field: (
                                w_jp1 * float(cont_blk_jp1[field])
                                - w_j * float(cont_blk_j[field])
                            )
                            for field in fields
                        }
                        components = {
                            "open_set_change": open_set_change[skey][
                                observable],
                            "momentum_drift": momentum_drift[skey][
                                observable],
                            "delta_catalog": catalog_delta[skey][
                                observable],
                            "catalog_block_payload": catalog_block_payload,
                            "continuum_block_payload":
                                continuum_block_payload,
                            "w_at_nodes": [float(w_j), float(w_jp1)],
                        }
                    # The SE convention: the multiplicity uses the pooled
                    # per-node standard errors; the moments use the
                    # between-replica standard errors (the R21/R22
                    # convention).  evaluate_pair combines them as
                    # SE_combined = sqrt(SE_j^2 + SE_{j+1}^2).
                    if observable == "multiplicity":
                        se_k = float(mult_j["se"])
                        se_kp1 = float(mult_jp1["se"])
                    else:
                        se_k = se[observable] / math.sqrt(2.0)
                        se_kp1 = se[observable] / math.sqrt(2.0)
                    verdict = evaluate_pair(
                        observed, delta_full, se_k, se_kp1
                    )
                    sigma = verdict["sigma"]
                    row = {
                        "pair": [int(j), int(j + 1)],
                        "global_pair": int(global_pair),
                        "masses_gev": list(masses),
                        "species": skey,
                        "observable": observable,
                        "observed": [
                            observed_j[observable], observed_jp1[observable]
                        ],
                        "delta_observed": float(observed),
                        "delta_analytic_full": float(delta_full),
                        "delta_components": components,
                        "se_combined": float(verdict["se_combined"]),
                        "sigma": float(sigma),
                        "tolerance_sigma": float(METRIC_N_SIGMA),
                        "status": "pass" if verdict["pass"] else "fail",
                        "pass": bool(verdict["pass"]),
                    }
                    pair_rows.append(row)
                    total_applied += 1
                    is_boundary = (j == 2)
                    if is_boundary:
                        boundary_applied += 1
                        w_boundary_applied += 1
                    else:
                        interior_applied += 1
                        w_interior_applied += 1
                    if verdict["pass"]:
                        continue
                    total_failed += 1
                    if is_boundary:
                        boundary_failures += 1
                        w_boundary_fail += 1
                    else:
                        interior_failures += 1
                        w_interior_fail += 1
                    value = abs(float(sigma))
                    if math.isfinite(value):
                        if value > max_abs_sigma:
                            max_abs_sigma = value
                        if is_boundary:
                            if value > boundary_max:
                                boundary_max = value
                        else:
                            if value > interior_max:
                                interior_max = value
                    else:
                        max_abs_sigma = math.inf
                        if is_boundary:
                            boundary_max = math.inf
                        else:
                            interior_max = math.inf
            pairs_payload[str(j)] = {
                "pair": [int(j), int(j + 1)],
                "global_pair": int(global_pair),
                "masses_gev": list(masses),
                "w_at_nodes": [float(w_j), float(w_jp1)],
                "boundary_straddling": bool(j == 2),
                "catalog_only": bool(catalog_only),
                "checks": pair_rows,
            }
        b_rate = (
            w_boundary_fail / w_boundary_applied
            if w_boundary_applied > 0 else 0.0
        )
        i_rate = (
            w_interior_fail / w_interior_applied
            if w_interior_applied > 0 else 0.0
        )
        per_window_rates.append({
            "window": int(w),
            "qn": [int(v) for v in spec_window["qn"]],
            "boundary_pair": 4 * w + 2,
            "boundary_pair_failure_rate": float(b_rate),
            "interior_pairs_failure_rate": float(i_rate),
            "boundary_pair_applied": int(w_boundary_applied),
            "boundary_pair_failed": int(w_boundary_fail),
            "interior_pairs_applied": int(w_interior_applied),
            "interior_pairs_failed": int(w_interior_fail),
            "no_boundary_dominance": bool(b_rate <= i_rate),
        })
        windows_payload[str(w)] = {
            "qn": [int(v) for v in spec_window["qn"]],
            "pairs": pairs_payload,
        }
    boundary_rate = (
        boundary_failures / boundary_applied
        if boundary_applied > 0 else 0.0
    )
    interior_rate = (
        interior_failures / interior_applied
        if interior_applied > 0 else 0.0
    )
    events_checked = sum(
        int(nodes[(w, j)]["charge_balance"]["events_checked"])
        for w, j in nodes
    )
    balance_ok = all(
        int(nodes[(w, j)]["charge_balance"]["events_checked"])
        == T1_REPLICAS * T1_DECAYS_PER_REPLICA
        for w, j in nodes
    )
    summary = {
        "windows": N_WINDOWS,
        "adjacent_pairs_checked": N_WINDOWS * (N_NODES - 1),
        "checks_applied": int(total_applied),
        "checks_failed": int(total_failed),
        "max_abs_sigma": (
            float(max_abs_sigma)
            if math.isfinite(float(max_abs_sigma)) else "Infinity"
        ),
        "boundary_straddling_pairs": [
            int(v) for v in BOUNDARY_PAIR_INDICES
        ],
        "boundary_pair_max_abs_sigma": (
            float(boundary_max)
            if math.isfinite(float(boundary_max)) else "Infinity"
        ),
        "interior_pair_max_abs_sigma": (
            float(interior_max)
            if math.isfinite(float(interior_max)) else "Infinity"
        ),
        "pass": bool(total_failed == 0),
    }
    dominance = {
        "boundary_pair_failure_rate": float(boundary_rate),
        "interior_pair_failure_rate": float(interior_rate),
        "boundary_pair_failed": int(boundary_failures),
        "boundary_pair_applied": int(boundary_applied),
        "interior_pairs_failed": int(interior_failures),
        "interior_pairs_applied": int(interior_applied),
        "per_window": per_window_rates,
        "pass": bool(all(row["no_boundary_dominance"]
                         for row in per_window_rates)),
    }
    payload = {
        "schema": "na49_mixing.staged_r31_matching_scan.verification/1",
        "run": RUN_ID,
        "card": CARD,
        "metric": {
            "id": METRIC_ID,
            "tolerance_sigma": float(METRIC_N_SIGMA),
            "statement": pre["discontinuity_metric"]["statement"],
        },
        "summary": summary,
        "no_boundary_dominance": dominance,
        "windows": windows_payload,
        "events_checked": int(events_checked),
        "charge_balance_exact": bool(balance_ok),
        "no_fitting": True,
        "optimizer_invoked": False,
    }
    _write_json(run_dir / VERIFICATION_NAME, payload)
    if total_failed > 0:
        print(
            "staged_r31_matching_scan: the matching verification FAILED "
            f"({total_failed} of {total_applied} checks outside the "
            f"3.0-sigma band, max |sigma| = {summary['max_abs_sigma']}); "
            "the run stops and the card-6 routing or the originating "
            "family card is re-opened, never the tolerance",
            flush=True,
        )
        raise SystemExit(1)
    print(
        "staged_r31_matching_scan: matching verification passed "
        f"({total_applied} checks over "
        f"{N_WINDOWS * (N_NODES - 1)} adjacent pairs, max |sigma| = "
        f"{summary['max_abs_sigma']:.6f}, boundary pair max "
        f"{summary['boundary_pair_max_abs_sigma']:.6f}, interior max "
        f"{summary['interior_pair_max_abs_sigma']:.6f})",
        flush=True,
    )
    return payload


def _assert_verification(run_dir: Path) -> dict:
    path = Path(run_dir) / VERIFICATION_NAME
    if not path.is_file():
        _refuse(
            f"{path} is missing; the run requires the matching "
            "verification (run --verify first)"
        )
    report = _read_json(path)
    if not report.get("summary", {}).get("pass"):
        _refuse(
            f"{path} records checks_failed > 0; the run stops (a "
            "failing match re-opens the card-6 routing or the "
            "originating family card, never the tolerance)"
        )
    if not report.get("no_boundary_dominance", {}).get("pass"):
        _refuse(
            f"{path} records no_boundary_dominance pass = false; the "
            "run stops"
        )
    return report


# ---------------------------------------------------------------------------
# Mode: --gate
# ---------------------------------------------------------------------------


def run_gate(run_dir: Path) -> dict:
    """Write ``gate3_r31.json``: the Gate 3 evidence, assembled from the
    matching verification (evidence only)."""
    run_dir = Path(run_dir)
    pre = _assert_pre_registration(run_dir)
    verification = _assert_verification(run_dir)
    summary = verification["summary"]
    dominance = verification["no_boundary_dominance"]
    clauses = {
        "smoothness": {
            "windows": int(summary["windows"]),
            "adjacent_pairs_checked": int(summary["adjacent_pairs_checked"]),
            "checks_applied": int(summary["checks_applied"]),
            "checks_failed": int(summary["checks_failed"]),
            "max_abs_sigma": summary["max_abs_sigma"],
            "boundary_straddling_pairs": [
                int(v) for v in summary["boundary_straddling_pairs"]
            ],
            "boundary_pair_max_abs_sigma": summary[
                "boundary_pair_max_abs_sigma"
            ],
            "interior_pair_max_abs_sigma": summary[
                "interior_pair_max_abs_sigma"
            ],
            "pass": bool(summary["pass"]),
        },
        "no_boundary_dominance": {
            "boundary_pair_failure_rate": float(
                dominance["boundary_pair_failure_rate"]
            ),
            "interior_pair_failure_rate": float(
                dominance["interior_pair_failure_rate"]
            ),
            "per_window": dominance["per_window"],
            "pass": bool(dominance["pass"]),
        },
    }
    payload = {
        "schema": "na49_mixing.staged_r31_matching_scan.gate/1",
        "gate": "gate 3 (plan section 9, line 650)",
        "run": RUN_ID,
        "card": CARD,
        "clauses": clauses,
        "events_checked": int(verification["events_checked"]),
        "charge_balance_exact": bool(
            verification["charge_balance_exact"]
        ),
        "input_hashes": {
            "matching_verification_sha256": _sha256_file(
                run_dir / VERIFICATION_NAME
            ),
            "pre_registration_sha256": _sha256_file(
                run_dir / PREREGISTRATION_NAME
            ),
        },
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
    }
    _write_json(run_dir / GATE_RECORD_NAME, payload)
    print(
        "staged_r31_matching_scan: gate evidence written to "
        f"{run_dir / GATE_RECORD_NAME} (smoothness pass = "
        f"{clauses['smoothness']['pass']}, no_boundary_dominance pass = "
        f"{clauses['no_boundary_dominance']['pass']})",
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
        "schema": "na49_mixing.staged_r31_matching_scan.seal_manifest/1",
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
            f"paths ({run_dir.name}/<file> prefixed by dev/00runs/), so "
            "the run card's validation command `sha256sum -c "
            f"dev/00runs/{run_dir.name}/SHA256SUMS.txt` verifies from "
            "the repo root (the R21 convention)"
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
    manifest).  The pre-registration, the node/replica records, the
    verification, and the gate record must exist."""
    run_dir = Path(run_dir)
    for name in (
        PREREGISTRATION_NAME,
        PROVENANCE_NAME,
        VERIFICATION_NAME,
        GATE_RECORD_NAME,
    ):
        if not (run_dir / name).is_file():
            _refuse(
                f"{run_dir / name} is missing before sealing (the gate "
                "JSON is part of the seal; run --verify and --gate first)"
            )
    n_nodes = len(list(run_dir.glob("window_*_node_*.json"))) - len(
        list(run_dir.glob("window_*_node_*_replica_*.json"))
    )
    if n_nodes != N_WINDOWS * N_NODES:
        _refuse(
            f"expected {N_WINDOWS * N_NODES} node records before "
            f"sealing, found {n_nodes}"
        )
    write_sha256sums(run_dir)
    seal = write_seal_manifest(run_dir)
    print(
        f"staged_r31_matching_scan: run dir sealed ({len(seal['files'])} "
        f"files listed in {SEAL_NAME})",
        flush=True,
    )
    return seal


# ---------------------------------------------------------------------------
# Mode: --clean-validate
# ---------------------------------------------------------------------------


def run_clean_validate(run_dir: Path) -> dict:
    """Verify the seal, re-run the matching verification from a clean
    interpreter over the sealed artifacts, and confirm the report hash
    matches."""
    run_dir = Path(run_dir)
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

    # -- clean-interpreter re-run of the verification ---------------------
    report_path = run_dir / VERIFICATION_NAME
    before = report_path.read_bytes()
    driver_path = Path(__file__).resolve()
    argv = [sys.executable, "-I", str(driver_path), "--verify",
            "--run-dir", str(run_dir)]
    probe = subprocess.run(argv, capture_output=True, text=True,
                           cwd=str(MONO_ROOT))
    if probe.returncode != 0:
        _refuse(
            "the clean-interpreter verification probe failed "
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
        "schema": "na49_mixing.staged_r31_matching_scan.clean_environment/1",
        "run": RUN_ID,
        "card": CARD,
        "verified_utc": _now_utc(),
        "seal_verification": {
            "files_checked": int(files_checked),
            "mismatches": mismatches,
        },
        "commands": [{
            "purpose": (
                "clean-interpreter re-run of the matching verification "
                "(step 3) over the sealed artifacts (rewrites "
                "matching_verification.json; the sealed bytes must be "
                "reproduced byte for byte)"
            ),
            "argv": argv,
            "cwd": str(MONO_ROOT),
        }],
        "environment": environment_fingerprint(),
        "probe_returncode": int(probe.returncode),
        "sealed_report_sha256": {
            VERIFICATION_NAME: hashlib.sha256(before).hexdigest(),
        },
        "recomputed_report_sha256": {
            VERIFICATION_NAME: hashlib.sha256(recomputed).hexdigest(),
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
            "staged_r31_matching_scan: the clean-environment validation "
            "FAILED (the recomputed verification bytes differ from the "
            "sealed report)",
            flush=True,
        )
        raise SystemExit(1)
    print(
        "staged_r31_matching_scan: clean-environment validation passed "
        f"({files_checked} seal files verified, the clean interpreter "
        f"reproduced {VERIFICATION_NAME} byte for byte)",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="staged_r31_matching_scan.py",
        description=(
            "Execute run R31, the catalog/continuum matching scan "
            "(Round 3, card 8): T1 isolated-source runs on three "
            "five-node transition windows around the catalog-"
            "completeness boundary, the card-6 routing, and the "
            "re-registered threshold metric at 3.0 sigma (plan section "
            "9, line 642)."
        ),
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--pre-registration", action="store_true",
        help="write pre_registration.json and provenance.json, then stop",
    )
    modes.add_argument(
        "--generate", action="store_true",
        help="generate one node (--window W --node J), one window "
             "(--window W), or every window",
    )
    modes.add_argument(
        "--verify", action="store_true",
        help="run the matching verification over the sealed statistics, "
             "then stop",
    )
    modes.add_argument(
        "--gate", action="store_true",
        help="write gate3_r31.json (acceptance evidence), then stop",
    )
    modes.add_argument(
        "--seal", action="store_true",
        help="write seal_manifest.json and SHA256SUMS.txt, then stop",
    )
    modes.add_argument(
        "--clean-validate", action="store_true",
        help="verify the seal and re-run the verification cleanly, "
             "then stop",
    )
    parser.add_argument(
        "--window", type=int, default=None,
        help="with --generate: the window index w = 0..2",
    )
    parser.add_argument(
        "--node", type=int, default=None,
        help="with --generate: the node index j = 0..4",
    )
    parser.add_argument(
        "--windows", type=int, default=None,
        help="the declared window count (must match the "
             "pre-registration in production mode)",
    )
    parser.add_argument(
        "--nodes", type=int, default=None,
        help="the declared node count per window (must match the "
             "pre-registration in production mode)",
    )
    parser.add_argument(
        "--decays-per-replica", type=int, default=None,
        help="the declared decays per replica (must match the "
             "pre-registration in production mode)",
    )
    parser.add_argument(
        "--replicas", type=int, default=None,
        help="the declared replica count (must match the "
             "pre-registration in production mode)",
    )
    parser.add_argument(
        "--master-seed", type=int, default=None,
        help="the declared master seed (must match the "
             "pre-registration in production mode)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="with --generate: hermetic reduced statistics (driver "
             "validation; the verify / gate / seal / clean-validate "
             "modes refuse smoke artifacts)",
    )
    parser.add_argument(
        "--smoke-decays", type=int, default=SMOKE_DECAYS,
        help=f"smoke decays per replica (default {SMOKE_DECAYS})",
    )
    parser.add_argument(
        "--smoke-replicas", type=int, default=SMOKE_REPLICAS,
        help=f"smoke replica count (default {SMOKE_REPLICAS})",
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
    statistics = {
        "windows": args.windows,
        "nodes": args.nodes,
        "decays_per_replica": args.decays_per_replica,
        "replicas": args.replicas,
        "master_seed": args.master_seed,
    }
    if args.pre_registration:
        run_pre_registration(run_dir)
    elif args.generate:
        run_generation(
            run_dir,
            args.window,
            args.node,
            args.workers,
            bool(args.smoke),
            int(args.smoke_decays),
            int(args.smoke_replicas),
            statistics,
        )
    elif args.verify:
        run_verification(run_dir)
    elif args.gate:
        run_gate(run_dir)
    elif args.seal:
        run_seal(run_dir)
    elif args.clean_validate:
        run_clean_validate(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
