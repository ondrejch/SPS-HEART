#!/usr/bin/env python3
"""R4 audit driver: the builder blend-path routing-weight audit (Round 4, card 3).

Task ``TASK-20260920-r4-audit-builder-blend-path-routing-weight``
(Gate 3 decision card follow-up obligation 3; plan section 9.3, lines
625-632; the Round 4 index card, cards 1-3).  The audit confirms on the
byte-stable builder surface (``na49_mixing/resonance_pp_builder.py``,
sha256 ``4ab61260...``) that the ``ResonancePPBuilder`` blend path
contributes no routing artifact beyond the pre-registered w(M) routing
weight: the routed Monte Carlo consumes exactly the record draw and the
w(M) routing uniform per region and never consults the double-counting
boundary-matching factor of the card-6 blend.  The bit-identity leg
already passes (sealed R32 v0), so this is a routing-weight audit, not
a rebuild.  The outcome is a PASS verdict (the surface is adequate
as-is) or a RECOMMENDATION that spawns a separate rework card (the
cards 8a/8b precedent of Round 3).  The audit itself never edits the
pinned surface.

Zero new random numbers (hermetic): the audit consumes no new random
number.  The probes are fixed-seed re-evaluations of pinned fixtures:
both the 1,000-event draw-budget probe and the 200-event isolation spy
fixture re-run the sealed R32 v0 bit-identity stream configuration (raw
seed 20300913, kernel master 20260913, kernel replica index 0, the
first event ids of the sealed 10,000-event stream), so every consumed
uniform is a uniform the sealed R32/R23 runs already consumed.  The
direct route-contract probes are hand-fed stub streams (pre-chosen
uniform values, no generator at all).  No event is generated beyond
these re-evaluations.  ``draws`` is 0 (no new random numbers anywhere
in the run); the reserved audit confirmation block 20301040 + k,
k = 0..15, under the Round 4 master seed 20260920 is pre-registered for
the exceptional case and must not fire on the first pass.

Probes (the card's pre-registered measurement contract, index item 6):

- **Draw-budget trace** (``--draw-budget``): the 1,000-event fixed-seed
  probe across the three routing regions.  Per probe event, three
  builders generate the same event: the legacy staged builder, the
  frozen R23 builder (``blend = None``), and the traced v0 blend
  builder.  The source-layer bit-identity (topology / exchange /
  spectral_state, exact float equality) is re-verified at probe scale,
  which pins the probe's realized source masses to the sealed stream's
  masses (no new masses drawn; the masses are pinned fixtures).  The
  routed sampling path is traced white-box: the spawned
  ``decay_channel`` child generator is wrapped with a transparent
  counter, the builder module's ``route_source_decay`` binding is
  wrapped with a call recorder, and the per-source draw sequence is
  asserted against the card's contract: catalog region (M < B) zero
  routing uniforms and exactly one path draw (the record draw) with
  byte-identical finals to the R23 builder convention under the sealed
  perturbation-skip convention; transition (B <= M < B + W) exactly one
  routing uniform consumed before any record draw with the route
  ``'continuum'`` exactly when the draw ``u < w(M)``, then exactly one
  further path draw (the record draw on a catalog-routed source, the
  class draw on a continuum-routed source); continuum region
  (M >= B + W) zero routing uniforms and one path draw (the class
  draw).  Any extra uniform, missing uniform, or ordering change is a
  named finding.

- **Boundary-matching-factor isolation** (``--isolation``): the white-
  box assertion, on a pinned 200-event call fixture, that the routed
  sampling path consumes only the record draw and the routing uniform.
  Three legs: (i) a call spy on the card-6 blend-normalization surface
  (``blended_density``, ``blended_normalization_report``,
  ``_blended_pieces``, ``_blended_support``) records zero consultations
  during the fixture; (ii) the blend pieces cache
  (``_BLEND_PIECES_CACHE``, the only store of the boundary-matching
  factor) is asserted empty before and after the fixture (no side
  channel); (iii) an AST assertion over the builder sampling functions
  (``decay_source``, ``_decay_blended``, ``_decay_resonance``,
  ``_decay_continuum``) and the card-6 routing functions
  (``route_source_decay``, ``_require_open_catalog``,
  ``_require_active_family``) finds no reference to the factor and no
  call into the blend-normalization surface (no read, no argument).
  The routing-weight consistency leg completes the probe: every
  source's stamped ``blend_weight`` equals ``BlendSpec.weight(mass)``
  exactly (the pre-registered w(M) the card-6 surface specifies for the
  same (M, B, W)), and a hand-fed stub contract re-derives the
  three-region rule and the ``u < w(M)`` outcome on pinned masses.  A
  read on the sampling path is a named finding.

- **Census consistency** (``--census``): the probe re-derives the
  sealed R32 v0 routing census, and the sealed archive is the
  reference.  Leg A (the pre-registered 1e-6 relative check): the
  sealed v0 routing census is re-derived from the sealed per-replica
  payload routing counts (production 8, confirmation 16; pure integer
  pooling, zero draws) and compared with the sealed census records (the
  spread report, the gate record, the confirmation verification); the
  comparison span is the full realized source-mass span of the sealed
  production, identical on both sides by construction.  Leg B: the
  probe's own 1,000-event routing census from the draw-budget trace,
  re-derived independently from the traced (mass, u, w, route) rows and
  asserted internally exact.  Leg C (informational): the sealed
  production census fraction is recorded inside the w(M)-implied
  bracket ``[P(M < B), P(M < B) + P(B <= M < B + W)]`` computed from the
  sealed R32 v0 bit-identity stream's band census (the routing-weight
  consistency at the census level; informational, never the 1e-6
  gate).

- **Verdict** (``--verdict``): PASS if the draw budget is exact in all
  three regions, the below-boundary path re-verifies bit-identity, the
  boundary-matching factor is never consulted on the sampling path, and
  the census is consistent within 1e-6; or a RECOMMENDATION (separate
  rework card) with named findings (each finding: region, event
  fixture, observed draw sequence, diagnosis, the proposed rework).
  This card edits nothing pinned.

New spread bounds (pre-registered, per the decision card): any new
spread diagnostic produced by this audit carries the R32 pre-registered
bounds (bin spread p95 < 2.0 pooled-SE units, feed-down spread max
< 3.0 sigma, decision card lines 106-109).  This audit produces no
spread diagnostic (no per-bin spread is measured); the bounds are
recorded in the pre-registration and carried for any rework card this
audit spawns; they are never relaxed.

Modes (exactly one per invocation; each performs one step and stops):

- ``--pre-registration``: write ``pre_registration.json`` and
  ``provenance.json`` into the run dir, then stop.  Nothing is
  measured.  Verifies the sealed R32 production, confirmation, and R23
  seals, the sealed anchor digests, and the frozen substrate digests
  (including the builder bytes) first.
- ``--draw-budget``: the 1,000-event fixed-seed probe and the
  per-region budget table, write ``draw_budget_trace.json``, then
  stop.  Requires the pre-registration.
- ``--isolation``: the white-box factor isolation and the routing-
  weight consistency fixture, write ``isolation_assertion.json``, then
  stop.  Requires the pre-registration.
- ``--census``: the sealed-census re-derivation and the probe census,
  write ``census_consistency.json``, then stop.  Requires the
  pre-registration and the draw-budget step.
- ``--verdict``: write ``verdict.json`` (PASS or RECOMMENDATION with
  named findings), then stop.  Requires the three measurement steps.
- ``--seal``: write ``seal_manifest.json`` and ``SHA256SUMS.txt`` over
  the run dir, then stop.  The run dir is immutable after the seal.
- ``--clean-validate``: verify the seal, re-run the four deterministic
  measurement steps from a clean interpreter over the sealed artifacts,
  confirm every artifact hash matches, write
  ``clean_environment_commands.json``, then stop.

The four measurement artifacts (draw-budget, isolation, census,
verdict) carry no wall-clock fields: their bytes are a pure function of
the sealed archives and the committed driver, which is what the
clean-environment validation re-hashes.

Dual-tree driver: byte-identical in the nested tree
(``dev/mixing2006/helpers/benchmarking/``) and the monorepo mirror
(``helpers/benchmarking/``); both locate the package and the sealed
archives with the established ``_locate_roots`` pattern.  Hermetic in
every mode: standard library, numpy, and the pinned ``na49_mixing``
surfaces only (the audit object itself, read-only at the pinned
digests); no external SPS tables, no event generation beyond the
pre-registered fixed-seed re-evaluations, no optimizer, no new random
numbers.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

# ---------------------------------------------------------------------------
# Tree location: this driver runs from the nested tree
# (dev/mixing2006/helpers/benchmarking/) and from its byte-identical
# monorepo mirror (helpers/benchmarking/).  Both must find the
# na49_mixing package and the sealed archives under the monorepo root.
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
        f"audit_r4_builder_blend_path: {message}",
        file=sys.stderr, flush=True,
    )
    raise SystemExit(2)


NESTED_ROOT, MONO_ROOT = _locate_roots()
if str(NESTED_ROOT) not in sys.path:
    sys.path.insert(0, str(NESTED_ROOT))

from helpers.benchmarking import staged_r01_reference as r01  # noqa: E402
from na49_mixing.event_builder_pp import PPEventBuilder  # noqa: E402
from na49_mixing.hadron_catalog import HadronCatalog  # noqa: E402
from na49_mixing.resonance_pp_builder import (  # noqa: E402
    CONTINUUM_LABEL,
    KERNEL_LABEL,
    LEGACY_LABEL,
    ResonancePPBuilder,
)
import na49_mixing.resonance_pp_builder as builder_module  # noqa: E402
import na49_mixing.spectral_density as spectral_density  # noqa: E402
from na49_mixing.spectral_density import (  # noqa: E402
    BLEND_BOUNDARY,
    BLEND_RAMP_WIDTH,
    BlendSpec,
    families_from_json,
)
from na49_mixing.source_event import SourceRecord  # noqa: E402

# ---------------------------------------------------------------------------
# Pre-registered audit identity (card TASK-20260920-r4-audit-builder-
# blend-path-routing-weight; the probe set, the bounds, and the verdict
# contract are verbatim from the card and the round 4 index item 6).
# ---------------------------------------------------------------------------

CARD = "TASK-20260920-r4-audit-builder-blend-path-routing-weight"
RUN_ID = "R4-audit-builder-blend-path-routing-weight"
RUN_LABEL = "r4_audit_builder_blend_path_routing_weight"

PREREGISTRATION_NAME = "pre_registration.json"
PROVENANCE_NAME = "provenance.json"
TRACE_NAME = "draw_budget_trace.json"
ISOLATION_NAME = "isolation_assertion.json"
CENSUS_NAME = "census_consistency.json"
VERDICT_NAME = "verdict.json"
CLEAN_ENV_NAME = "clean_environment_commands.json"
SHA256SUMS_NAME = "SHA256SUMS.txt"
SEAL_NAME = "seal_manifest.json"

# The byte-stable pinned surfaces and the frozen substrate (read-only).
BUILDER_REL = "dev/mixing2006/na49_mixing/resonance_pp_builder.py"
BUILDER_SHA256 = "4ab612609e1ee0bc1e037dcbb2815b6484306ad32402b7289e6636f8e829cdb1"
CARD6_REL = "dev/mixing2006/na49_mixing/spectral_density.py"
CARD6_SHA256 = "6f889f6d72968de53e7b27e32ba64d124d1b4e782a73606e6f6794eef9f156a9"
ENGINE_REL = "dev/mixing2006/na49_mixing/threshold_metric.py"
ENGINE_SHA256 = "ee240f44645d2e592acfae462f7d1e9c3b3dd8cd78c96bd40b7b73509f7fcc63"
CATALOG_REL = "dev/mixing2006/config/v2_singlet/resonance_catalog.json"
CATALOG_SHA256 = "e42a5da7137d789ba90221478c9bb404898f4fe8a7ee4604cd85b5623b751752"
FAMILIES_REL = "dev/mixing2006/config/v2_singlet/continuum_resonances.json"
FAMILIES_SHA256 = "6da57d5709b94061bc178fe94781dfe5eddcc597c5dde46bfcd5db56985cde7d"
R32_DRIVER_NAME = "staged_r32_spectral_systematic.py"
R32_DRIVER_SHA256 = "99454facce856c93505e6c4b0d4e47b85eeb0ef54f72b1631e3d29fcc511cb90"

# The sealed archives (the probe inputs and the references) and the
# sealed anchor artifacts (read-only; their seals verify).
R32_PROD_DIR = "dev/00runs/r32_spectral_systematic_pp_2026-09-18"
R32_CONF_DIR = "dev/00runs/r32_spectral_systematic_pp_confirmation_2026-09-19"
R23_DIR = "dev/00runs/r23_resonance_only_pp_ablation_2026-09-13"

SPREAD_REPORT_NAME = "spread_report.json"
GATE_RECORD_NAME = "gate3_r32.json"
BIT_IDENTITY_NAME = "bit_identity_v0.json"
CONFIRMATION_VERIFICATION_NAME = "confirmation_verification.json"
R23_BIT_IDENTITY_NAME = "bit_identity_report.json"

# Sealed anchor digests (asserted by the pre-registration; the
# measurement modes re-hash every input they read against the
# pre-registered values).
SEALED_INPUT_DIGESTS = {
    f"{R32_PROD_DIR}/{SPREAD_REPORT_NAME}":
        "ed102b10e9c99d973a9c3e9dd262536eaca08802aeb8f313ea1894f7cefe022e",
    f"{R32_PROD_DIR}/{GATE_RECORD_NAME}":
        "64f104a8f0fdb71fdf506a3eb6a369091a2f3eed5047e499f1d344ad5668bab6",
    f"{R32_PROD_DIR}/{BIT_IDENTITY_NAME}":
        "9ec4c6077642422283ae68fd3b3ae6bd00fa8c9b30172610ab259013bb8e5c5c",
    f"{R32_CONF_DIR}/{CONFIRMATION_VERIFICATION_NAME}":
        "0553fed06a9b59e9f2020c0c53bbad7d75a11b1f2baf21f0469ee8b3e48d275f",
    f"{R23_DIR}/{R23_BIT_IDENTITY_NAME}":
        "5c168540d3269c0f315cd47fba652ad4f1721f023fb310944d40f0f9b548e857",
}

# Sealed R32 anchor numbers (asserted against the sealed artifacts).
PRODUCTION_CENSUS_V0 = {
    "n_sources": 1600000,
    "n_catalog": 357274,
    "fraction_catalog": 0.22329625,
    "fraction_continuum": 0.77670375,
}
GATE_CENSUS_V0 = 0.22329625
CONFIRMATION_CENSUS_V0 = {
    "n_sources": 3200000,
    "n_catalog": 715424,
    "fraction_catalog": 0.22357,
    "fraction_continuum": 0.77643,
}
# The sealed R32 v0 bit-identity stream (the probe configuration, the
# sealed band census, and the below-boundary bit-identity reference).
BIT_IDENTITY_SEED = 20300913
BIT_IDENTITY_EVENTS = 10000
BIT_IDENTITY_MASTER_SEED = 20260913
BIT_IDENTITY_REPLICA_INDEX = 0
BIT_IDENTITY_BAND_CENSUS = {
    "n_below_boundary_sources": 3760,
    "n_in_band_sources": 1488,
    "n_above_boundary_sources": 14752,
    "v0_catalog_routed": 4520,
    "v0_continuum_routed": 15480,
}
# The sealed R23 bit-identity reference (the below-boundary convention).
R23_BIT_IDENTITY_ANCHOR = {
    "seed": 20300913,
    "n_events": 10000,
    "layers_topology_matched": 10000,
    "layers_topology_compared": 10000,
    "layers_exchange_matched": 10000,
    "layers_spectral_state_matched": 10000,
    "legacy_kernel_finals_matched": 14537,
}

# The blend (the R32 v0 nominal variant, verbatim).
KERNEL_BOUNDARY_GEV = 2.680194110727528
RAMP_WIDTH_GEV = 0.5

# The probe (hermetic, fixed seed): the sealed R32 v0 bit-identity
# stream's first 1,000 events.  The probe reuses the sealed stream's
# realized source-mass values (no new masses drawn; the masses are
# pinned fixtures re-verified by the source-layer bit-identity leg).
PROBE_EVENTS = 1000
PROBE_STAGE = "r4_audit_builder_blend_probe"
LEGACY_STAGE = "r23_resonance_ablation"
SPY_EVENTS = 200

# Pre-registered bounds and contracts.
CENSUS_DISAGREEMENT_BOUND = 1e-6
BIN_SPREAD_BOUND = 2.0
FEEDDOWN_SPREAD_BOUND = 3.0

# The reserved audit confirmation block (the exceptional case; must not
# fire on the first pass).  Master seed 20260920 (the Round 4 master),
# raw seeds 20301040 + k, k = 0..15, disjoint from every previously used
# raw block and from the Round 4 run blocks 20301020-20301039.
RESERVED_MASTER_SEED = 20260920
RESERVED_BLOCK_FIRST = 20301040
RESERVED_BLOCK_LAST = 20301055

# The three routing regions (the card's probe-set regions; the exact
# draw-budget contract of the card-6 routing surface).
REGION_RULES = {
    "catalog_region": (
        "M < B: zero routing uniforms, the R23 catalog convention (one "
        "record draw on the spawned decay_channel child; the catalog-"
        "path draw sequence is byte-identical to the no-blend "
        "convention at the same seed)"
    ),
    "transition": (
        "B <= M < B + W: exactly one routing uniform consumed before "
        "any record draw; continuum exactly when the draw u < w(M)"
    ),
    "continuum_region": (
        "M >= B + W: zero routing uniforms; the continuum family "
        "mixture (one class draw on the spawned decay_channel child "
        "follows)"
    ),
}

# The blend-normalization surface (the boundary-matching factor's only
# homes) and the sampling-path functions of the two pinned surfaces
# (the isolation assertion's scope).
FACTOR_SURFACE = (
    "blended_density",
    "blended_normalization_report",
    "_blended_pieces",
    "_blended_support",
)
FACTOR_NAME = "matching_factor"
BUILDER_SAMPLING_FUNCTIONS = (
    "decay_source",
    "_decay_blended",
    "_decay_resonance",
    "_decay_continuum",
)
CARD6_ROUTING_FUNCTIONS = (
    "route_source_decay",
    "_require_open_catalog",
    "_require_active_family",
)

PROBE_SOURCE_QN = (1, 1, 0)


# ---------------------------------------------------------------------------
# Small shared helpers (the R32 driver conventions, local copies)
# ---------------------------------------------------------------------------


def _now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _json_safe(value: Any) -> Any:
    """The R01 sentinel convention: non-finite floats become strings so
    identical inputs produce identical bytes (RFC 8259 compatibility)."""
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


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


def _write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _read_json(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _restore_sentinels(payload)


def _write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def default_run_dir() -> Path:
    """The dated run dir under ``dev/00runs`` (gitignored, not under a
    pinned root), named with the execution date at run time."""
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    return MONO_ROOT / "dev" / "00runs" / f"{RUN_LABEL}_{today}"


# ---------------------------------------------------------------------------
# Sealed-archive and frozen-substrate verification (read-only)
# ---------------------------------------------------------------------------


def verify_sealed_dir(rel: str) -> dict:
    """Verify one sealed run dir's ``SHA256SUMS.txt`` (read-only; the
    sealed archives are byte-unchanged inputs).  The entries carry
    monorepo-root-relative paths (the R21/R32 convention) or
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


def substrate_digests() -> dict:
    """The on-disk digests of the frozen substrate (computed, not
    asserted)."""
    return {
        "builder_module_sha256": _sha256_file(MONO_ROOT / BUILDER_REL),
        "card6_surface_sha256": _sha256_file(MONO_ROOT / CARD6_REL),
        "engine_module_sha256": _sha256_file(MONO_ROOT / ENGINE_REL),
        "catalog_sha256": _sha256_file(MONO_ROOT / CATALOG_REL),
        "families_config_sha256": _sha256_file(MONO_ROOT / FAMILIES_REL),
        "r32_driver_nested_sha256": _sha256_file(
            NESTED_ROOT / "helpers" / "benchmarking" / R32_DRIVER_NAME
        ),
        "r32_driver_outer_sha256": _sha256_file(
            MONO_ROOT / "helpers" / "benchmarking" / R32_DRIVER_NAME
        ),
    }


def verify_frozen_substrate() -> dict:
    """The frozen substrate digest checks (the card's substrate list)."""
    problems: list[str] = []
    hashes = substrate_digests()
    expected = {
        "card6_surface_sha256": (CARD6_SHA256, "the card-6 blend routing"),
        "builder_module_sha256": (BUILDER_SHA256, "the builder blend path"),
        "engine_module_sha256": (ENGINE_SHA256, "the fixed card-2 engine"),
        "catalog_sha256": (CATALOG_SHA256, "the committed catalog"),
        "families_config_sha256": (
            FAMILIES_SHA256, "the committed continuum config"),
        "r32_driver_nested_sha256": (
            R32_DRIVER_SHA256, "the frozen R32 driver (nested copy)"),
        "r32_driver_outer_sha256": (
            R32_DRIVER_SHA256, "the frozen R32 driver (outer mirror)"),
    }
    for key, (want, role) in expected.items():
        if hashes[key] != want:
            problems.append(
                f"{role} sha256 {hashes[key]!r} != the pre-registered {want!r}"
            )
    if hashes["r32_driver_nested_sha256"] != hashes["r32_driver_outer_sha256"]:
        problems.append(
            "the R32 driver copies are not byte-identical across the two "
            "helper trees"
        )
    if problems:
        _refuse(
            "frozen substrate verification failed; refusing to run:\n  "
            + "\n  ".join(problems)
        )
    return hashes


def verify_sealed_inputs() -> dict:
    """Re-hash the sealed anchor inputs against the pre-registered
    digests (the measurement modes call this before every step)."""
    problems: list[str] = []
    for rel, want in SEALED_INPUT_DIGESTS.items():
        path = MONO_ROOT / rel
        if not path.is_file():
            problems.append(f"sealed input missing: {rel}")
            continue
        got = _sha256_file(path)
        if got != want:
            problems.append(
                f"sealed input {rel} sha256 {got!r} != the pre-registered "
                f"{want!r}"
            )
    if problems:
        _refuse(
            "sealed input verification failed; refusing to run:\n  "
            + "\n  ".join(problems)
        )
    return dict(SEALED_INPUT_DIGESTS)


# ---------------------------------------------------------------------------
# The audit fixture state (the committed catalog, the committed
# continuum families, the M8-frozen config, the v0 blend)
# ---------------------------------------------------------------------------

_CATALOG: HadronCatalog | None = None
_FAMILIES: list | None = None
_PP_CONFIG: Any = None


def audit_catalog() -> HadronCatalog:
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = HadronCatalog.from_json(MONO_ROOT / CATALOG_REL)
    return _CATALOG


def audit_families() -> list:
    global _FAMILIES
    if _FAMILIES is None:
        _FAMILIES = families_from_json(str(MONO_ROOT / FAMILIES_REL))
    return _FAMILIES


def audit_config() -> Any:
    """The M8-frozen p+p config (the R32 driver's application rule)."""
    global _PP_CONFIG
    if _PP_CONFIG is None:
        raw_path = NESTED_ROOT / "config" / r01.PRODUCTION_CONFIG_NAME
        if not raw_path.is_file():
            _refuse(f"production config missing: {raw_path}")
        raw_config = json.loads(raw_path.read_text(encoding="utf-8"))
        config, _applied = r01.build_pp_config(raw_config)
        _PP_CONFIG = config
    return _PP_CONFIG


def audit_blend() -> BlendSpec:
    """The pre-registered v0 nominal blend (the R32 variant 0)."""
    return BlendSpec(KERNEL_BOUNDARY_GEV, RAMP_WIDTH_GEV)


def _region_of(mass: float) -> str:
    """The routing region of a source mass (the card's probe-set
    regions: M < B, B <= M < B + W, M >= B + W)."""
    if mass < KERNEL_BOUNDARY_GEV:
        return "catalog_region"
    if mass < KERNEL_BOUNDARY_GEV + RAMP_WIDTH_GEV:
        return "transition"
    return "continuum_region"


def expected_uniforms_of(mass: float) -> int:
    """The pre-registered routing-uniform budget of a source mass
    (0 / 1 / 0 across the three regions)."""
    return 1 if _region_of(mass) == "transition" else 0


def _expected_route(u: float | None, mass: float) -> str:
    """The hand-derived route of the card-6 contract: the three-region
    rule plus the ``u < w(M)`` band outcome (the audit's independent
    reimplementation of the pre-registered routing weight)."""
    region = _region_of(mass)
    if region == "catalog_region":
        return "catalog"
    if region == "continuum_region":
        return "continuum"
    if u is None:
        _refuse("a transition-region source carries no routing uniform")
    weight = (mass - KERNEL_BOUNDARY_GEV) / RAMP_WIDTH_GEV
    return "continuum" if float(u) < weight else "catalog"


def _expected_weight(mass: float) -> float:
    """The pre-registered w(M) ramp, re-derived from the card's
    decimals (the audit's independent copy of the card-6 ramp)."""
    if not mass > KERNEL_BOUNDARY_GEV:
        return 0.0
    if mass >= KERNEL_BOUNDARY_GEV + RAMP_WIDTH_GEV:
        return 1.0
    return (mass - KERNEL_BOUNDARY_GEV) / RAMP_WIDTH_GEV


# ---------------------------------------------------------------------------
# The white-box tracing machinery
# ---------------------------------------------------------------------------

#: The original ``route_source_decay`` binding of the builder module,
#: captured at import (the tracer wraps and restores it).
_ORIGINAL_ROUTE = builder_module.route_source_decay

#: The per-event trace records of the patched builder binding: one
#: (mass, route, child_calls_before, child_calls_after) per call.
_ROUTE_TRACE: list[list] = []

#: The current event's traced ``decay_channel`` child (set per event,
#: read by the tracer to segment the child call log at the routing
#: call boundaries).
_ROUTE_CHILD_HOLDER: dict = {"rng": None}


class _TracingRNG:
    """A transparent wrapper that records every ``random()`` draw on
    the wrapped spawned child (the draw-budget accounting probe)."""

    def __init__(self, inner):
        self._inner = inner
        self.calls: list[float] = []

    def random(self):
        value = self._inner.random()
        self.calls.append(value)
        return value

    @property
    def bit_generator(self):
        return self._inner.bit_generator


class _StubRNG:
    """A hand-fed uniform stream (no generator): the direct route-
    contract probes consume exactly the pre-chosen values, so no new
    random number is drawn anywhere in the audit."""

    def __init__(self, values: list[float]):
        self._values = list(values)
        self.consumed: list[float] = []

    def random(self):
        if not self._values:
            raise AssertionError(
                "the hand-fed uniform stream was drained: the routing "
                "consumed more draws than the pre-registered budget"
            )
        value = self._values.pop(0)
        self.consumed.append(value)
        return value


def _install_route_tracer() -> None:
    """Wrap the builder module's ``route_source_decay`` binding (the
    sampling path's routing entry point) with the call recorder."""

    def _traced(qn, M, spec, catalog_, fams, rng) -> str:
        child = _ROUTE_CHILD_HOLDER["rng"]
        before = len(child.calls) if child is not None else None
        route = _ORIGINAL_ROUTE(qn, M, spec, catalog_, fams, rng)
        after = len(child.calls) if child is not None else None
        _ROUTE_TRACE.append([float(M), str(route), before, after])
        return route

    builder_module.route_source_decay = _traced


def _restore_route_tracer() -> None:
    """Restore the original binding and clear the trace state."""
    builder_module.route_source_decay = _ORIGINAL_ROUTE
    _ROUTE_TRACE.clear()
    _ROUTE_CHILD_HOLDER["rng"] = None


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


# ---------------------------------------------------------------------------
# Mode: --pre-registration
# ---------------------------------------------------------------------------


def build_pre_registration(substrate: dict, seals: list) -> dict:
    """The audit pre-registration payload, written before any
    measurement: the probe set, the draw-budget contract, the isolation
    assertion, the census bound, the verdict contract, and the
    zero-draw assertion."""
    return {
        "schema": "na49_mixing.audit_r4_builder_blend_path"
                  ".pre_registration/1",
        "run": RUN_ID,
        "card": CARD,
        "written_before_measurement": True,
        "surface": {
            "builder_blend_path": {
                "path": BUILDER_REL,
                "sha256": BUILDER_SHA256,
                "note": (
                    "the builder surface is byte-unchanged by this "
                    "audit; a surface change is a separate rework card, "
                    "never an in-audit edit"
                ),
            },
            "card6_blend_routing": {
                "path": CARD6_REL,
                "sha256": CARD6_SHA256,
            },
            "engine_module_sha256": ENGINE_SHA256,
            "catalog_sha256": CATALOG_SHA256,
            "families_config_sha256": FAMILIES_SHA256,
            "r32_driver_sha256": R32_DRIVER_SHA256,
        },
        "blend": {
            "boundary_gev": KERNEL_BOUNDARY_GEV,
            "ramp_width_gev": RAMP_WIDTH_GEV,
            "note": (
                "the R32 v0 nominal variant, verbatim: the card-6 "
                "BlendSpec evaluated at the frozen source mass; w(M) = 0 "
                "at and below B, (M - B)/W on the open transition "
                "interval, 1 from B + W upward"
            ),
        },
        "regions": REGION_RULES,
        "probe_set": {
            "events": PROBE_EVENTS,
            "seed": BIT_IDENTITY_SEED,
            "kernel_master_seed": BIT_IDENTITY_MASTER_SEED,
            "kernel_replica_index": BIT_IDENTITY_REPLICA_INDEX,
            "event_ids": [0, PROBE_EVENTS - 1],
            "source": (
                "the sealed R32 v0 bit-identity stream "
                f"({R32_PROD_DIR}/{BIT_IDENTITY_NAME}: seed "
                f"{BIT_IDENTITY_SEED}, kernel master "
                f"{BIT_IDENTITY_MASTER_SEED}, replica index "
                f"{BIT_IDENTITY_REPLICA_INDEX}, 10,000 sealed events) at "
                f"the probe scale of the first {PROBE_EVENTS} events; "
                "the probe reuses the sealed stream's realized "
                "source-mass values (no new masses drawn; the masses "
                "are pinned fixtures re-verified by the source-layer "
                "bit-identity leg)"
            ),
            "builders_per_event": [
                "the legacy staged PPEventBuilder (the R01 convention)",
                "the frozen R23 builder (blend = None)",
                "the traced v0 blend builder (the audit object)",
            ],
            "spy_fixture_events": SPY_EVENTS,
        },
        "draw_budget_contract": {
            "catalog_region": (
                "zero routing uniforms and a byte-identical record / "
                "mass draw sequence to the R23 builder convention (the "
                "sealed R32 v0 bit-identity leg, re-verified at probe "
                "scale: the source layers and the below-boundary "
                "catalog-route finals under the R23 perturbation-skip "
                "convention)"
            ),
            "transition": (
                "exactly one routing uniform with the routing outcome "
                "matching u < w(M), consumed before any record draw"
            ),
            "continuum_region": (
                "zero routing uniforms and the family draw follows"
            ),
            "named_finding_rule": (
                "any extra uniform, missing uniform, or ordering change "
                "is a named finding"
            ),
        },
        "isolation_assertion": {
            "spy_surface": [
                "blended_density",
                "blended_normalization_report",
                "_blended_pieces",
                "_blended_support",
            ],
            "spy_fixture_events": SPY_EVENTS,
            "cache_side_channel": (
                "the blend pieces cache (_BLEND_PIECES_CACHE, the only "
                "store of the boundary-matching factor) is asserted "
                "empty before and after the fixture"
            ),
            "static_assertion": (
                "an AST assertion over the builder sampling functions "
                "(decay_source, _decay_blended, _decay_resonance, "
                "_decay_continuum) and the card-6 routing functions "
                "(route_source_decay, _require_open_catalog, "
                "_require_active_family): no reference to the boundary-"
                "matching factor and no call into the blend-"
                "normalization surface (no read, no argument, no side "
                "channel)"
            ),
            "routing_weight_consistency": (
                "every source's stamped blend_weight equals "
                "BlendSpec.weight(mass) exactly (the pre-registered "
                "w(M) the card-6 surface specifies for the same "
                "(M, B, W)), and a hand-fed stub-stream contract "
                "re-derives the three-region rule and the u < w(M) "
                "outcome on pinned masses"
            ),
            "named_finding_rule": (
                "a read on the sampling path is a named finding"
            ),
        },
        "census_consistency": {
            "leg_a": {
                "method": (
                    "the probe re-derives the sealed R32 v0 routing "
                    "census from the sealed per-replica payload routing "
                    "counts (production 8, confirmation 16; pure "
                    "integer pooling, zero draws) and compares it with "
                    "the sealed census records (the spread report, the "
                    "gate record, the confirmation verification); the "
                    "sealed archive is the reference and the comparison "
                    "span is the full realized source-mass span of the "
                    "sealed production, identical on both sides by "
                    "construction"
                ),
                "disagreement_bound": CENSUS_DISAGREEMENT_BOUND,
                "bound_note": (
                    "a disagreement beyond 1e-6 relative is a driver "
                    "defect (verdict criterion (d)); rework the driver, "
                    "never the surface"
                ),
            },
            "leg_b": (
                "the probe's own 1,000-event routing census from the "
                "draw-budget trace, re-derived independently from the "
                "traced (mass, u, w, route) rows and asserted "
                "internally exact"
            ),
            "leg_c_informational": (
                "the sealed production census fraction is recorded "
                "inside the w(M)-implied bracket [P(M < B), P(M < B) + "
                "P(B <= M < B + W)] computed from the sealed R32 v0 "
                "bit-identity stream's band census (the routing-weight "
                "consistency at the census level; informational, never "
                "the 1e-6 gate)"
            ),
        },
        "new_spread_bounds": {
            "bin_spread_p95_bound": BIN_SPREAD_BOUND,
            "feeddown_spread_max_bound": FEEDDOWN_SPREAD_BOUND,
            "note": (
                "any new spread diagnostic produced by this audit "
                "carries the R32 pre-registered bounds (decision card "
                "lines 106-109); this audit produces no spread "
                "diagnostic (no per-bin spread is measured), and the "
                "bounds are carried for any rework card this audit "
                "spawns; the bounds are never relaxed and a bound "
                "failure is a recorded finding, never a refit"
            ),
        },
        "verdict_contract": {
            "pass": (
                "the draw budget is exact in all three regions (0 / 1 / "
                "0 routing uniforms), the below-boundary path "
                "re-verifies bit-identity, the boundary-matching factor "
                "is never consulted on the sampling path, and the "
                "census is consistent within 1e-6"
            ),
            "recommendation": (
                "RECOMMENDATION (separate rework card) with named "
                "findings (each finding: region, event fixture, "
                "observed draw sequence, diagnosis, the proposed "
                "rework) if any of (a) the draw budget, (b) the "
                "below-boundary bit-identity leg, (c) the isolation "
                "assertion, or (d) the census consistency fails"
            ),
            "criteria": ["a", "b", "c", "d"],
            "scope_note": (
                "this card owns the builder-extension share only: the "
                "routing-formulation share is card 1's scope "
                "(TASK-20260920-r4-audit-transition-routing-formulation)"
                " and the strange-sector family share is card 2's scope "
                "(TASK-20260920-r4-audit-strange-sector-continuum-"
                "families); the three audits are independent inputs to "
                "the Round 4 run cards"
            ),
            "rework_precedent": (
                "the Round 3 cards 8a/8b precedent: a RECOMMENDATION "
                "spawns a separate rework card; this audit edits "
                "nothing pinned"
            ),
        },
        "zero_draw": {
            "draws": 0,
            "no_fitting": True,
            "optimizer_invoked": False,
            "method": (
                "the audit consumes no new random number: the probes "
                "are fixed-seed re-evaluations of the sealed R32 v0 "
                "bit-identity stream configuration (raw seed "
                f"{BIT_IDENTITY_SEED}, kernel master "
                f"{BIT_IDENTITY_MASTER_SEED}, kernel replica index "
                f"{BIT_IDENTITY_REPLICA_INDEX}, the first {PROBE_EVENTS} "
                f"and {SPY_EVENTS} event ids), so every consumed uniform "
                "is a uniform the sealed runs already consumed, and the "
                "direct contract probes are hand-fed stub streams with "
                "no generator at all; no event is generated beyond "
                "these re-evaluations and no optimizer runs"
            ),
            "reserved_audit_block": {
                "master_seed": RESERVED_MASTER_SEED,
                "raw_seeds": [RESERVED_BLOCK_FIRST, RESERVED_BLOCK_LAST],
                "raw_seed_rule": (
                    "20301040 + k, k = 0..15, under the Round 4 master "
                    "seed 20260920; disjoint from every previously used "
                    "raw block and from the Round 4 run blocks "
                    "20301020-20301039"
                ),
                "fired": False,
                "note": (
                    "the reserved confirmation block is pre-registered "
                    "for the exceptional case and must not fire on the "
                    "first pass"
                ),
            },
        },
        "sealed_inputs": dict(SEALED_INPUT_DIGESTS),
        "sealed_run_dirs_verified": seals,
        "substrate": substrate,
    }


def run_pre_registration(run_dir: Path) -> dict:
    """Write ``pre_registration.json`` and ``provenance.json``, then
    stop.  Nothing is measured."""
    run_dir = Path(run_dir)
    substrate = verify_frozen_substrate()
    if substrate["builder_module_sha256"] != BUILDER_SHA256:
        _refuse("the builder surface is not byte-stable at the pinned digest")
    if substrate["card6_surface_sha256"] != CARD6_SHA256:
        _refuse("the card-6 surface is not byte-stable at the pinned digest")
    seals = [
        verify_sealed_dir(rel)
        for rel in (R32_PROD_DIR, R32_CONF_DIR, R23_DIR)
    ]
    verify_sealed_inputs()
    pre = build_pre_registration(substrate, seals)
    _write_json(run_dir / PREREGISTRATION_NAME, pre)
    driver_path = Path(__file__).resolve()
    provenance = {
        "schema": "na49_mixing.audit_r4_builder_blend_path.provenance/1",
        "run": RUN_ID,
        "card": CARD,
        "card_chain": [
            "TASK-20260920-r3-gate3-decision",
            "TASK-20260920-round4-index",
            CARD,
        ],
        "written_utc": _now_utc(),
        "driver": "helpers/benchmarking/audit_r4_builder_blend_path.py",
        "driver_sha256": {
            "nested": _sha256_file(
                NESTED_ROOT / "helpers" / "benchmarking"
                / "audit_r4_builder_blend_path.py"
            ),
            "outer": _sha256_file(driver_path),
            "note": (
                "the dual-tree driver is byte-identical in both trees; "
                "the outer entry is the on-disk mirror's digest"
            ),
        },
        "frozen_substrate": substrate,
        "sealed_run_dirs_verified": seals,
        "sealed_inputs": dict(SEALED_INPUT_DIGESTS),
        "zero_draw": {
            "draws": 0,
            "no_fitting": True,
            "optimizer_invoked": False,
            "reserved_audit_block_fired": False,
        },
        "environment": environment_fingerprint(),
        "git": _git_info(),
    }
    _write_json(run_dir / PROVENANCE_NAME, provenance)
    print(
        f"audit_r4_builder_blend_path: pre-registration written to "
        f"{run_dir} ({len(seals)} sealed archives verified, substrate "
        "digests pinned)",
        flush=True,
    )
    return pre


def assert_pre_registration(run_dir: Path) -> dict:
    """Load the pre-registration, re-verify the substrate digests and
    the sealed input digests, and refuse to run without it."""
    run_dir = Path(run_dir)
    path = run_dir / PREREGISTRATION_NAME
    if not path.is_file():
        _refuse(
            f"{path} is missing; every measurement mode requires the "
            "pre-registration (run --pre-registration first)"
        )
    pre = _read_json(path)
    if pre.get("card") != CARD:
        _refuse(f"{path} carries the wrong card identity")
    substrate = verify_frozen_substrate()
    recorded = pre.get("substrate", {})
    for key, want in (
        ("card6_surface_sha256", CARD6_SHA256),
        ("builder_module_sha256", BUILDER_SHA256),
        ("engine_module_sha256", ENGINE_SHA256),
        ("catalog_sha256", CATALOG_SHA256),
        ("families_config_sha256", FAMILIES_SHA256),
        ("r32_driver_nested_sha256", R32_DRIVER_SHA256),
        ("r32_driver_outer_sha256", R32_DRIVER_SHA256),
    ):
        if recorded.get(key) != want or substrate[key] != want:
            _refuse(
                f"substrate digest drift on {key}: the pre-registration and "
                f"the on-disk file must both carry {want!r}"
            )
    verify_sealed_inputs()
    zero = pre.get("zero_draw", {})
    if zero.get("draws") != 0 or not zero.get("no_fitting") or zero.get(
            "optimizer_invoked"):
        _refuse("the pre-registration zero-draw contract is not intact")
    if zero.get("reserved_audit_block", {}).get("fired") is not False:
        _refuse("the reserved audit block must not fire on the first pass")
    return pre


# ---------------------------------------------------------------------------
# Mode: --draw-budget (the 1,000-event fixed-seed traced probe)
# ---------------------------------------------------------------------------


def _probe_event(event_id: int, blend: BlendSpec, catalog: HadronCatalog,
                 config: Any) -> dict:
    """One probe event: the three builders at the same (seed, event_id),
    the source-layer bit-identity leg, and the white-box draw-budget
    trace of the v0 blend path (the draw sequence on the spawned
    ``decay_channel`` child, segmented per source)."""
    legacy_builder = PPEventBuilder(
        config,
        np.random.default_rng(
            np.random.SeedSequence(BIT_IDENTITY_SEED,
                                   spawn_key=(event_id,))
        ),
    )
    legacy_event = legacy_builder.generate(event_id, stage=LEGACY_STAGE)
    r23_builder = ResonancePPBuilder(
        audit_config(), catalog,
        seed=BIT_IDENTITY_SEED, event_id=event_id,
        master_seed=BIT_IDENTITY_MASTER_SEED,
        replica_index=BIT_IDENTITY_REPLICA_INDEX,
    )
    r23_result = r23_builder.generate(stage=PROBE_STAGE)
    v0_builder = ResonancePPBuilder(
        audit_config(), catalog,
        seed=BIT_IDENTITY_SEED, event_id=event_id,
        master_seed=BIT_IDENTITY_MASTER_SEED,
        replica_index=BIT_IDENTITY_REPLICA_INDEX,
        blend=blend, families=audit_families(),
    )
    child = _TracingRNG(v0_builder._rng_record)
    kin = _TracingRNG(v0_builder._rng_kinematics)
    v0_builder._rng_record = child
    v0_builder._rng_kinematics = kin
    _ROUTE_TRACE.clear()
    _ROUTE_CHILD_HOLDER["rng"] = child
    v0_result = v0_builder.generate(stage=PROBE_STAGE)
    _ROUTE_CHILD_HOLDER["rng"] = None

    # -- the source-layer bit-identity leg (re-verified at probe scale) --
    legacy_layers = _event_layers(legacy_event)
    r23_layers = _event_layers(r23_result.event)
    v0_layers = _event_layers(v0_result.event)
    layer_rows = []
    for key in ("topology", "exchange", "spectral_state"):
        layer_rows.append({
            "layer": key,
            "legacy_vs_r23": bool(legacy_layers[key] == r23_layers[key]),
            "r23_vs_v0": bool(r23_layers[key] == v0_layers[key]),
        })

    # -- the draw-budget segmentation -----------------------------------
    trace = list(_ROUTE_TRACE)
    if len(trace) < 2:
        _refuse(
            f"event {event_id}: the routed sampling path issued "
            f"{len(trace)} routing calls; the final attempt decays both "
            "sources and must issue at least two"
        )
    final_routes = trace[-2:]
    attempts = int(v0_result.event.metadata["generation_attempt"])
    source_rows: list[dict] = []
    comparable_v0 = True
    comparable_legacy = True
    n_sources = len(v0_result.source_records)
    for index, (record, route_call) in enumerate(
            zip(v0_result.source_records, final_routes)):
        mass, route, n_before, n_after = (
            float(route_call[0]), str(route_call[1]),
            int(route_call[2]), int(route_call[3]),
        )
        if abs(mass - float(record.mass)) > 0.0:
            _refuse(
                f"event {event_id}: the routing call mass {mass!r} does "
                f"not match the source record mass {float(record.mass)!r}"
            )
        region = _region_of(mass)
        routing_uniforms = n_after - n_before
        expected_uniforms = 1 if region == "transition" else 0
        next_before = (
            final_routes[index + 1][2] if index + 1 < len(final_routes)
            else len(child.calls)
        )
        path_draws = next_before - n_after
        u = child.calls[n_before] if routing_uniforms == 1 else None
        weight = float(blend.weight(mass))
        expected_route = _expected_route(u, mass)
        row = {
            "event_id": int(event_id),
            "source_id": int(record.source_id),
            "role": str(record.role),
            "attempt": attempts,
            "mass": mass,
            "region": region,
            "routing_uniforms": int(routing_uniforms),
            "routing_uniform_expected": int(expected_uniforms),
            "path_draws": int(path_draws),
            "route": route,
            "route_expected": expected_route,
            "u": u,
            "w": weight,
            "blend_weight_stamped": (
                float(record.blend_weight)
                if record.blend_weight is not None else None
            ),
            "blend_weight_expected": float(blend.weight(mass)),
            "kernel": str(record.kernel),
            "record_pdg": (
                int(record.record_pdg)
                if record.record_pdg is not None else None
            ),
            "n_open_records": int(record.n_open_records),
            "family_id": record.family_id,
            "kinematics_draws": None,
        }
        # The expected route must agree with the builder's stamps.
        if route != record.route:
            row["route_stamp_mismatch"] = True
        if row["blend_weight_stamped"] != row["blend_weight_expected"]:
            row["weight_stamp_mismatch"] = True
        # Every routing call obeys its region budget (the retry
        # attempts obey the same per-invocation contract).
        if routing_uniforms != expected_uniforms:
            row["uniform_budget_mismatch"] = True
        if path_draws != 1:
            row["path_budget_mismatch"] = True
        if route != expected_route:
            row["route_derivation_mismatch"] = True
        source_rows.append(row)

    # -- the retry routing calls obey the same per-invocation contract --
    retry_rows_ok = True
    for entry in trace[:-2]:
        entry_mass, _route, e_before, e_after = (
            float(entry[0]), str(entry[1]), int(entry[2]), int(entry[3]),
        )
        expected = 1 if _region_of(entry_mass) == "transition" else 0
        if (e_after - e_before) != expected_uniforms_of(entry_mass):
            retry_rows_ok = False

    # -- the below-boundary bit-identity leg (the sealed R32 v0
    #    convention, re-verified at probe scale) ------------------------
    finals_rows = []
    comparable_v0 = True
    comparable_legacy = True
    for r23_record, v0_record in zip(
            r23_result.source_records, v0_result.source_records):
        mass = float(v0_record.mass)
        below_v0 = mass < blend.boundary
        row = {
            "source_id": int(v0_record.source_id),
            "mass": mass,
            "below_blend_boundary": bool(below_v0),
            "v0_kernel": str(v0_record.kernel),
            "r23_kernel": str(r23_record.kernel),
        }
        if below_v0 and v0_record.kernel == KERNEL_LABEL:
            if comparable_v0:
                row["v0_vs_r23"] = (
                    _finals_of_source(v0_result.event, v0_record.source_id)
                    == _finals_of_source(r23_result.event, r23_record.source_id)
                )
            else:
                row["v0_vs_r23"] = "skipped_perturbed"
        else:
            row["v0_vs_r23"] = "not_catalog_route"
        comparable_v0 = comparable_v0 and below_v0
        if r23_record.kernel == LEGACY_LABEL:
            if comparable_legacy:
                row["r23_vs_legacy"] = (
                    _finals_of_source(r23_result.event, r23_record.source_id)
                    == _finals_of_source(legacy_event, r23_record.source_id)
                )
            else:
                row["r23_vs_legacy"] = "skipped_perturbed"
        else:
            row["r23_vs_legacy"] = "skipped_new_kernel"
            comparable_legacy = False
        finals_rows.append(row)
    return {
        "event_id": int(event_id),
        "attempts": attempts,
        "layer_rows": layer_rows,
        "source_rows": source_rows,
        "finals_rows": finals_rows,
        "retry_route_calls": len(trace) - 2,
        "retry_route_calls_ok": bool(retry_rows_ok),
        "child_calls_total": len(child.calls),
        "kin_calls_total": len(kin.calls),
        "event_hash": v0_result.event.as_dict()["event_hash"],
    }


def _region_budget_table(source_rows: list) -> dict:
    """The per-region budget table of the probe (the card's Phase 2
    step 2): sources, routing uniforms, path draws, and routes per
    region, with every mismatched row named."""
    budgets = {}
    for region in ("catalog_region", "transition", "continuum_region"):
        rows = [r for r in source_rows if r["region"] == region]
        mismatches = [
            r for r in rows
            if r["routing_uniforms"] != r["routing_uniform_expected"]
            or r["path_draws"] != 1
            or r["route"] != r["route_expected"]
            or r.get("route_stamp_mismatch")
            or r.get("weight_stamp_mismatch")
            or r.get("uniform_budget_mismatch")
            or r.get("path_budget_mismatch")
            or r.get("route_derivation_mismatch")
        ]
        block = {
            "rule": REGION_RULES[region],
            "n_sources": len(rows),
            "routing_uniforms_total": int(
                sum(r["routing_uniforms"] for r in rows)),
            "expected_routing_uniforms_total": int(
                sum(r["routing_uniform_expected"] for r in rows)),
            "path_draws_total": int(sum(r["path_draws"] for r in rows)),
            "expected_path_draws_total": len(rows),
            "routes_catalog": int(
                sum(1 for r in rows if r["route"] == "catalog")),
            "routes_continuum": int(
                sum(1 for r in rows if r["route"] == "continuum")),
            "routes_expected_catalog": int(
                sum(1 for r in rows if r["route_expected"] == "catalog")),
            "routes_expected_continuum": int(
                sum(1 for r in rows
                    if r["route_expected"] == "continuum")),
            "weight_stamp_agreements": int(
                sum(1 for r in rows
                    if r["blend_weight_stamped"]
                    == r["blend_weight_expected"])),
            "mismatches": mismatches,
            "budget_exact": not mismatches,
        }
        if region == "transition":
            block["u_lt_w_agreements"] = int(
                sum(1 for r in rows if r["route"] == r["route_expected"]))
            block["band_uniforms_one_each"] = bool(
                all(r["routing_uniforms"] == 1 for r in rows))
        budgets[region] = block
    return budgets


def run_draw_budget(run_dir: Path) -> dict:
    """The 1,000-event fixed-seed probe across the three routing
    regions; the per-region budget table."""
    run_dir = Path(run_dir)
    assert_pre_registration(run_dir)
    blend = audit_blend()
    catalog = audit_catalog()
    config = audit_config()
    if float(blend.boundary) != KERNEL_BOUNDARY_GEV or float(
            blend.ramp_width) != RAMP_WIDTH_GEV:
        _refuse("the v0 blend constants drifted from the pre-registration")

    _install_route_tracer()
    events = []
    try:
        for event_id in range(PROBE_EVENTS):
            event = _probe_event(event_id, blend, catalog, config)
            if not event["retry_route_calls_ok"]:
                _refuse(
                    f"event {event_id}: a retry-attempt routing call "
                    "leaves the pre-registered per-invocation budget"
                )
            if not all(row["legacy_vs_r23"] and row["r23_vs_v0"]
                       for row in event["layer_rows"]):
                _refuse(
                    f"event {event_id}: the source-layer bit-identity leg "
                    "failed; the probe masses are not the sealed "
                    "stream's masses"
                )
            events.append(event)
    finally:
        _restore_route_tracer()

    source_rows = [row for event in events for row in event["source_rows"]]
    finals_rows = [row for event in events for row in event["finals_rows"]]
    budgets = _region_budget_table(source_rows)

    # -- the probe census (Leg B input) ---------------------------------
    n_catalog_routed = int(sum(
        1 for row in source_rows if row["route"] == "catalog"))
    n_sources = len(source_rows)
    probe_census = {
        "n_events": PROBE_EVENTS,
        "n_sources": n_sources,
        "n_below_boundary_sources": int(sum(
            1 for row in source_rows
            if row["region"] == "catalog_region")),
        "n_in_band_sources": int(sum(
            1 for row in source_rows if row["region"] == "transition")),
        "n_above_boundary_sources": int(sum(
            1 for row in source_rows
            if row["region"] == "continuum_region")),
        "catalog_routed": n_catalog_routed,
        "continuum_routed": n_sources - n_catalog_routed,
        "fraction_catalog": n_catalog_routed / n_sources,
    }

    # The sealed-envelope consistency of the probe span: the probe is a
    # prefix of the sealed 10,000-event stream, so each per-band probe
    # count is bounded by the sealed per-band total (informational).
    envelope = {
        "probe_below_le_sealed_below": bool(
            probe_census["n_below_boundary_sources"]
            <= BIT_IDENTITY_BAND_CENSUS["n_below_boundary_sources"]),
        "probe_in_band_le_sealed_in_band": bool(
            probe_census["n_in_band_sources"]
            <= BIT_IDENTITY_BAND_CENSUS["n_in_band_sources"]),
        "probe_above_le_sealed_above": bool(
            probe_census["n_above_boundary_sources"]
            <= BIT_IDENTITY_BAND_CENSUS["n_above_boundary_sources"]),
        "probe_catalog_routed_le_sealed": bool(
            probe_census["catalog_routed"]
            <= BIT_IDENTITY_BAND_CENSUS["v0_catalog_routed"]),
        "probe_continuum_routed_le_sealed": bool(
            probe_census["continuum_routed"]
            <= BIT_IDENTITY_BAND_CENSUS["v0_continuum_routed"]),
    }

    # -- the bit-identity leg aggregates --------------------------------
    # The per-comparison-instance accounting of the sealed R32 v0
    # bit-identity report: two comparisons (legacy vs R23 builder, R23
    # builder vs v0 blend builder) x three layers x per event.
    layer_compared = len(events) * 2 * 3
    layer_matched = int(sum(
        1 for event in events for row in event["layer_rows"]
        for hit in (row["legacy_vs_r23"], row["r23_vs_v0"]) if hit))
    finals_compared = int(sum(
        1 for row in finals_rows if isinstance(row.get("v0_vs_r23"), bool)))
    finals_matched = int(sum(
        1 for row in finals_rows if row.get("v0_vs_r23") is True))
    finals_skipped = int(sum(
        1 for row in finals_rows
        if row.get("v0_vs_r23") == "skipped_perturbed"))
    legacy_compared = int(sum(
        1 for row in finals_rows
        if isinstance(row.get("r23_vs_legacy"), bool)))
    legacy_matched = int(sum(
        1 for row in finals_rows if row.get("r23_vs_legacy") is True))
    bit_identity_reverified = bool(
        layer_matched == layer_compared
        and finals_matched == finals_compared
        and legacy_matched == legacy_compared)

    hash_list = [str(event["event_hash"]) for event in events]
    hash_digest = hashlib.sha256(
        json.dumps(hash_list, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    draw_budget_exact = bool(
        all(block["budget_exact"] for block in budgets.values())
        and all(event["retry_route_calls_ok"] for event in events))

    payload = {
        "schema": "na49_mixing.audit_r4_builder_blend_path.draw_budget/1",
        "run": RUN_ID,
        "card": CARD,
        "probe": (
            "the draw sequence of the routed sampling path across the "
            "three routing regions on the "
            f"{PROBE_EVENTS}-event fixed-seed probe"
        ),
        "probe_configuration": {
            "seed": BIT_IDENTITY_SEED,
            "kernel_master_seed": BIT_IDENTITY_MASTER_SEED,
            "kernel_replica_index": BIT_IDENTITY_REPLICA_INDEX,
            "n_events": PROBE_EVENTS,
            "event_ids": [0, PROBE_EVENTS - 1],
            "builders_per_event": [
                "the legacy staged PPEventBuilder (the R01 convention)",
                "the frozen R23 builder (blend = None)",
                "the traced v0 blend builder (the audit object)",
            ],
            "mass_fixture_note": (
                "the probe reuses the sealed stream's realized source-"
                "mass values (no new masses drawn; the masses are pinned "
                "fixtures re-verified by the source-layer bit-identity "
                "leg)"
            ),
        },
        "region_budgets": budgets,
        "probe_census": probe_census,
        "sealed_envelope": envelope,
        "bit_identity_leg": {
            "layer_definitions": (
                "the topology / exchange / spectral_state source layers, "
                "exact float equality, across the legacy staged builder, "
                "the frozen R23 builder (blend = None), and the traced v0 "
                "blend builder, event by event (the sealed R32 v0 bit-"
                "identity leg, re-verified at probe scale)"
            ),
            "layer_compared": int(layer_compared),
            "layer_matched": layer_matched,
            "below_boundary_finals_compared": finals_compared,
            "below_boundary_finals_matched": finals_matched,
            "below_boundary_finals_skipped_perturbed": finals_skipped,
            "legacy_finals_compared": legacy_compared,
            "legacy_finals_matched": legacy_matched,
            "reverified": bool(bit_identity_reverified),
            "note": (
                "recorded, not asserted: the above-boundary finals differ "
                "from R23 by design (the legacy H3 kernel is replaced by "
                "the continuum routing); the comparisons run under the "
                "R23 perturbation-skip convention"
            ),
        },
        "aggregate": {
            "events": len(events),
            "attempts_total": int(sum(e["attempts"] for e in events)),
            "retry_route_calls_total": int(sum(
                e["retry_route_calls"] for e in events)),
            "child_calls_total": int(sum(
                e["child_calls_total"] for e in events)),
            "kin_calls_total": int(sum(
                e["kin_calls_total"] for e in events)),
            "event_hash_digest": hash_digest,
        },
        "draw_budget_exact": draw_budget_exact,
        "per_event_rows": events,
        "no_fitting": True,
        "optimizer_invoked": False,
        "draws": 0,
        "draw_note": (
            "zero new random numbers: the probe is a fixed-seed "
            "re-evaluation of the sealed R32 v0 bit-identity stream "
            "configuration, so every consumed uniform is a uniform the "
            "sealed runs already consumed"
        ),
    }
    _write_json(run_dir / TRACE_NAME, payload)
    region_text = ", ".join(
        f"{region}={block['n_sources']} src, "
        f"{block['routing_uniforms_total']} routing uniforms"
        for region, block in sorted(budgets.items()))
    print(
        f"audit_r4_builder_blend_path: draw-budget trace written "
        f"({n_sources} sources; {region_text}; budget exact "
        f"{draw_budget_exact}; bit-identity re-verified "
        f"{bit_identity_reverified})",
        flush=True,
    )
    return payload


def events_of(payload: dict) -> list:
    """The per-event rows of a draw-budget trace payload."""
    return list(payload.get("per_event_rows", []))


def source_rows_of(payload: dict) -> list:
    """The flattened per-source rows of a draw-budget trace payload."""
    rows = []
    for event in events_of(payload):
        rows.extend(event.get("source_rows", []))
    return rows


def finals_rows_of(payload: dict) -> list:
    """The flattened per-event bit-identity comparison rows."""
    rows = []
    for event in events_of(payload):
        rows.extend(event.get("finals_rows", []))
    return rows


# ---------------------------------------------------------------------------
# Mode: --isolation (the boundary-matching-factor isolation)
# ---------------------------------------------------------------------------


def _spy_fixture() -> dict:
    """The pinned 200-event call fixture: the v0 blend builder runs the
    sealed stream's first 200 events with the blend-normalization
    surface under a call spy and the blend pieces cache watched."""
    counters = {name: 0 for name in FACTOR_SURFACE}
    originals = {
        name: getattr(spectral_density, name) for name in FACTOR_SURFACE
    }
    pieces_before = len(spectral_density._BLEND_PIECES_CACHE)
    blend = audit_blend()
    catalog = audit_catalog()
    config = audit_config()
    _install_route_tracer()

    def _make_spy(name):
        def _spy_call(*args, **kwargs):
            counters[name] += 1
            return originals[name](*args, **kwargs)
        return _spy_call

    source_rows: list[dict] = []
    try:
        for name in FACTOR_SURFACE:
            setattr(spectral_density, name, _make_spy(name))
        for event_id in range(SPY_EVENTS):
            builder = ResonancePPBuilder(
                config, catalog,
                seed=BIT_IDENTITY_SEED, event_id=event_id,
                master_seed=BIT_IDENTITY_MASTER_SEED,
                replica_index=BIT_IDENTITY_REPLICA_INDEX,
                blend=blend, families=audit_families(),
            )
            child = _TracingRNG(builder._rng_record)
            kin = _TracingRNG(builder._rng_kinematics)
            builder._rng_record = child
            builder._rng_kinematics = kin
            _ROUTE_TRACE.clear()
            _ROUTE_CHILD_HOLDER["rng"] = child
            result = builder.generate(stage=PROBE_STAGE)
            _ROUTE_CHILD_HOLDER["rng"] = None
            trace = list(_ROUTE_TRACE)
            if len(trace) < 2:
                _refuse(
                    f"spy fixture event {event_id}: {len(trace)} routing "
                    "calls; the final attempt decays both sources"
                )
            final_routes = trace[-2:]
            for index, record in enumerate(result.source_records):
                mass, route, n_before, n_after = (
                    float(final_routes[index][0]),
                    str(final_routes[index][1]),
                    int(final_routes[index][2]),
                    int(final_routes[index][3]),
                )
                region = _region_of(mass)
                routing_uniforms = n_after - n_before
                next_before = (
                    final_routes[index + 1][2]
                    if index + 1 < len(final_routes) else len(child.calls)
                )
                path_draws = next_before - n_after
                u = child.calls[n_before] if routing_uniforms == 1 else None
                source_rows.append({
                    "event_id": int(event_id),
                    "source_id": int(record.source_id),
                    "mass": mass,
                    "region": region,
                    "routing_uniforms": int(routing_uniforms),
                    "routing_uniform_expected": expected_uniforms_of(mass),
                    "path_draws": int(path_draws),
                    "route": route,
                    "route_expected": _expected_route(u, mass),
                    "u": u,
                    "w": float(blend.weight(mass)),
                    "blend_weight_stamped": (
                        float(record.blend_weight)
                        if record.blend_weight is not None else None
                    ),
                    "blend_weight_expected": float(blend.weight(mass)),
                    "kernel": str(record.kernel),
                })
    finally:
        for name in FACTOR_SURFACE:
            setattr(spectral_density, name, originals[name])
        _restore_route_tracer()
    pieces_after = len(spectral_density._BLEND_PIECES_CACHE)

    per_region = {
        region: int(sum(1 for row in source_rows
                        if row["region"] == region))
        for region in ("catalog_region", "transition", "continuum_region")
    }
    routes_exercised = {
        row["route"] for row in source_rows
    } == {"catalog", "continuum"}
    fixture_adequate = bool(
        per_region["transition"] > 0 and routes_exercised)
    if not fixture_adequate:
        _refuse(
            "the isolation spy fixture does not exercise all three "
            "routing regions and both routes (a driver defect; rework "
            "the driver, never the surface)"
        )
    budgets_ok = bool(all(
        row["routing_uniforms"] == expected_uniforms_of(row["mass"])
        and row["path_draws"] == 1 for row in source_rows))
    stamps_exact = bool(all(
        row["blend_weight_stamped"] == row["blend_weight_expected"]
        for row in source_rows))
    route_outcomes_exact = bool(all(
        row["route"] == _expected_route(row["u"], row["mass"])
        for row in source_rows))
    return {
        "spy_consultations": {name: int(count) for name, count in
                              counters.items()},
        "spy_consultations_total": int(sum(counters.values())),
        "blend_pieces_cache_before": int(pieces_before),
        "blend_pieces_cache_after": int(pieces_after),
        "n_sources": len(source_rows),
        "per_region_sources": per_region,
        "budgets_ok": bool(budgets_ok),
        "stamps_exact": bool(stamps_exact),
        "route_outcomes_exact": bool(route_outcomes_exact),
        "fixture_adequate": bool(fixture_adequate),
    }


def _ast_static_assertion() -> dict:
    """The static leg: no sampling-path function of either pinned
    surface references the boundary-matching factor or calls into the
    blend-normalization surface (no read, no argument, no side
    channel)."""
    builder_text = (MONO_ROOT / BUILDER_REL).read_text(encoding="utf-8")
    card6_text = (MONO_ROOT / CARD6_REL).read_text(encoding="utf-8")
    forbidden = set(FACTOR_SURFACE) | {FACTOR_NAME}
    problems: list[str] = []
    checked: list[str] = []
    for module_name, rel, want, text, functions in (
        ("na49_mixing.resonance_pp_builder", BUILDER_REL, BUILDER_SHA256,
         builder_text, BUILDER_SAMPLING_FUNCTIONS),
        ("na49_mixing.spectral_density", CARD6_REL, CARD6_SHA256,
         card6_text, CARD6_ROUTING_FUNCTIONS),
    ):
        if _sha256_file(MONO_ROOT / rel) != want:
            _refuse(f"the pinned surface {rel} is not byte-stable")
        found = {}
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name in functions:
                found[node.name] = node
        for fname in functions:
            fn = found.get(fname)
            if fn is None:
                problems.append(
                    f"{module_name}: sampling function {fname} not found")
                continue
            checked.append(f"{module_name}.{fname}")
            for sub in ast.walk(fn):
                if isinstance(sub, ast.Name) and sub.id in forbidden:
                    problems.append(
                        f"{module_name}.{fname}: references {sub.id!r}")
                if isinstance(sub, ast.Attribute) and sub.attr in forbidden:
                    problems.append(
                        f"{module_name}.{fname}: attribute reference "
                        f"{sub.attr!r}")
                if isinstance(sub, ast.Call):
                    target = sub.func
                    name = (
                        target.id if isinstance(target, ast.Name)
                        else (target.attr
                              if isinstance(target, ast.Attribute)
                              else None)
                    )
                    if name in forbidden:
                        problems.append(
                            f"{module_name}.{fname}: call into {name!r}")
    return {
        "checked_functions": checked,
        "forbidden_names": sorted(forbidden),
        "problems": problems,
        "clean": not problems,
        "note": (
            "the sampling call path is asserted to be free of the "
            "double-counting boundary-matching factor: no read, no "
            "argument, no side channel"
        ),
    }


def _direct_route_contract() -> dict:
    """The hand-fed stub-stream contract of the card-6 routing: the
    three-region rule and the ``u < w(M)`` outcome re-derived on pinned
    masses and hand-chosen uniforms (no generator, no new random
    number)."""
    catalog = audit_catalog()
    families = audit_families()
    blend = audit_blend()
    hand_uniforms = (0.0, 0.2499999999, 0.25, 0.5, 0.75, 0.9999999999)
    hand_masses = (
        KERNEL_BOUNDARY_GEV - 1e-9,
        KERNEL_BOUNDARY_GEV,
        KERNEL_BOUNDARY_GEV + 1e-9,
        KERNEL_BOUNDARY_GEV + 0.25,
        KERNEL_BOUNDARY_GEV + RAMP_WIDTH_GEV - 1e-9,
        KERNEL_BOUNDARY_GEV + RAMP_WIDTH_GEV,
        KERNEL_BOUNDARY_GEV + RAMP_WIDTH_GEV + 1e-9,
        5.0,
    )
    rows = []
    problems = []
    for mass in hand_masses:
        region = _region_of(mass)
        for u in hand_uniforms:
            stub = _StubRNG([u])
            route = spectral_density.route_source_decay(
                PROBE_SOURCE_QN, mass, blend, catalog, families, stub)
            consumed = len(stub.consumed)
            weight = float(blend.weight(mass))
            expected_route = _expected_route(u, mass)
            expected_consumed = expected_uniforms_of(mass)
            ok = bool(consumed == expected_consumed
                      and route == expected_route)
            if not ok:
                problems.append({
                    "mass": mass, "u": u, "route": route,
                    "route_expected": expected_route,
                    "consumed": consumed,
                    "expected_consumed": expected_consumed,
                })
            rows.append({
                "mass": mass, "u": u, "w": weight, "region": region,
                "route": route, "route_expected": expected_route,
                "uniforms_consumed": consumed,
                "uniforms_expected": expected_consumed,
                "pass": bool(ok),
            })
    return {
        "method": (
            "the audit's independent reimplementation of the card-6 "
            "routing contract (the three-region rule plus u < w(M)) on "
            "pinned masses and hand-chosen uniforms on a hand-fed stub "
            "stream; the hand case crosses the exact boundary values "
            "M = B (w = 0, one uniform consumed, the continuum never "
            "routed) and M = B + W (w = 1, zero uniforms, the continuum "
            "taken)"
        ),
        "hand_uniforms": list(hand_uniforms),
        "hand_masses": list(hand_masses),
        "rows": rows,
        "problems": problems,
        "clean": not problems,
    }


def run_isolation(run_dir: Path) -> dict:
    """The white-box assertion that the sampling path never reads the
    boundary-matching factor, plus the routing-weight consistency."""
    run_dir = Path(run_dir)
    assert_pre_registration(run_dir)
    spy = _spy_fixture()
    static = _ast_static_assertion()
    direct = _direct_route_contract()
    factor_isolated = bool(
        spy["spy_consultations_total"] == 0
        and spy["blend_pieces_cache_before"] == 0
        and spy["blend_pieces_cache_after"] == 0
        and spy["fixture_adequate"]
        and spy["budgets_ok"]
        and spy["stamps_exact"]
        and static["clean"]
        and direct["clean"])
    payload = {
        "schema": "na49_mixing.audit_r4_builder_blend_path.isolation/1",
        "run": RUN_ID,
        "card": CARD,
        "probe": (
            "the white-box boundary-matching-factor isolation of the "
            "ResonancePPBuilder blend path (the routed Monte Carlo "
            "consumes only the record draw and the routing uniform)"
        ),
        "spy_fixture": {
            "events": SPY_EVENTS,
            "stream": (
                "the sealed R32 v0 bit-identity stream configuration "
                f"(seed {BIT_IDENTITY_SEED}, kernel master "
                f"{BIT_IDENTITY_MASTER_SEED}, kernel replica index "
                f"{BIT_IDENTITY_REPLICA_INDEX}, event ids 0.."
                f"{SPY_EVENTS - 1})"
            ),
            "spy_surface": list(FACTOR_SURFACE),
            "spy_consultations": spy["spy_consultations"],
            "spy_consultations_total": spy["spy_consultations_total"],
            "blend_pieces_cache_before": spy["blend_pieces_cache_before"],
            "blend_pieces_cache_after": spy["blend_pieces_cache_after"],
            "fixture_adequate": spy["fixture_adequate"],
            "per_region_sources": spy["per_region_sources"],
            "budgets_ok": spy["budgets_ok"],
            "stamps_exact": spy["stamps_exact"],
        },
        "static_assertion": static,
        "direct_route_contract": direct,
        "routing_weight_consistency": {
            "statement": (
                "the routed region matches the w(M) weight the card-6 "
                "surface specifies for the same (M, B, W): every stamped "
                "blend_weight equals BlendSpec.weight(mass) exactly, the "
                "band outcome matches u < w(M), and the three-region draw "
                "budgets hold (0 / 1 / 0)"
            ),
            "stamps_exact": spy["stamps_exact"],
            "budgets_ok": spy["budgets_ok"],
            "route_outcomes_exact": spy["route_outcomes_exact"],
        },
        "factor_isolated": bool(factor_isolated),
        "no_fitting": True,
        "optimizer_invoked": False,
        "draws": 0,
    }
    _write_json(run_dir / ISOLATION_NAME, payload)
    print(
        f"audit_r4_builder_blend_path: isolation assertion written "
        f"(spy consultations {spy['spy_consultations_total']}, static "
        f"clean {static['clean']}, direct contract clean "
        f"{direct['clean']}, factor isolated {factor_isolated})",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# Mode: --census (the routing census re-derivation)
# ---------------------------------------------------------------------------


def pooled_census_from_payloads(records: list) -> dict:
    """One variant's routing census pooled from its sealed per-replica
    payload routing counts (pure integer arithmetic; zero draws)."""
    n_catalog = 0
    n_sources = 0
    replicas = 0
    for record in records:
        routing = record.get("routing", {})
        n_catalog += int(routing["n_catalog"])
        n_sources += int(routing["n_sources"])
        replicas += 1
    if n_sources <= 0:
        _refuse("the pooled routing census needs a positive source count")
    return {
        "replicas": replicas,
        "n_sources": n_sources,
        "n_catalog": n_catalog,
        "n_continuum": n_sources - n_catalog,
        "fraction_catalog": n_catalog / n_sources,
        "fraction_continuum": (n_sources - n_catalog) / n_sources,
    }


def census_disagreement(recomputed: dict, sealed: dict) -> dict:
    """The absolute disagreements of one variant's recomputed census
    against its sealed record (fractions and counts)."""
    fields = {}
    for key in ("fraction_catalog", "fraction_continuum"):
        fields[key] = abs(float(recomputed[key]) - float(sealed[key]))
    for key in ("n_catalog", "n_continuum", "n_sources"):
        fields[key] = abs(float(recomputed[key]) - float(sealed[key]))
    return fields


def load_payload_census(rel_dir: str, name_template: str, n_replicas: int,
                        ) -> dict:
    """The per-variant pooled census from one sealed archive's payload
    records (the template carries the variant and replica slots)."""
    per_variant: dict[str, dict] = {}
    for v in ("0", "1", "2"):
        records = []
        for r in range(int(n_replicas)):
            path = MONO_ROOT / rel_dir / name_template.format(v=v, r=r)
            if not path.is_file():
                _refuse(f"sealed payload record missing: {path}")
            records.append(_read_json(path))
        per_variant[v] = pooled_census_from_payloads(records)
    return per_variant


def _payload_identity_checks(records_by_variant: dict, role: str) -> list:
    """The sealed payload identity stamps: every payload the census
    reads asserts the builder digest it ran under (white-box
    provenance)."""
    rows = []
    for v in ("0", "1", "2"):
        for record in records_by_variant[v]:
            rows.append({
                "role": role,
                "variant": int(v),
                "builder_digest_ok": bool(
                    record.get("builder_module_sha256") == BUILDER_SHA256),
                "driver_digest_ok": bool(
                    record.get("driver_sha256") == R32_DRIVER_SHA256),
                "catalog_digest_ok": bool(
                    record.get("catalog_sha256") == CATALOG_SHA256),
                "blend_block_ok": bool(
                    int(v) != 0
                    or (float(record.get("blend", {}).get(
                        "boundary_gev", 0.0)) == KERNEL_BOUNDARY_GEV
                        and float(record.get("blend", {}).get(
                            "ramp_width_gev", 0.0)) == RAMP_WIDTH_GEV)),
            })
    bad = [row for row in rows if not (
        row["builder_digest_ok"] and row["driver_digest_ok"]
        and row["catalog_digest_ok"] and row["blend_block_ok"])]
    if bad:
        _refuse(
            f"a sealed {role} payload leaves the pre-registered identity "
            f"(first bad: {bad[0]!r}); the sealed archive must carry the "
            "pinned builder digest"
        )
    return rows


def run_census(run_dir: Path) -> dict:
    """The probe routing census against the sealed R32 v0 census (the
    1e-6 bound) and the probe census (internal exactness)."""
    run_dir = Path(run_dir)
    assert_pre_registration(run_dir)
    trace_path = run_dir / TRACE_NAME
    if not trace_path.is_file():
        _refuse(
            f"{trace_path} is missing; the census requires the draw-"
            "budget step (run --draw-budget first)"
        )
    trace = _read_json(trace_path)

    prod_records = {
        v: [_read_json(MONO_ROOT / R32_PROD_DIR
                       / f"variant_{v}_replica_{r}.json")
            for r in range(8)]
        for v in ("0", "1", "2")
    }
    conf_records = {
        v: [_read_json(MONO_ROOT / R32_CONF_DIR
                       / f"confirmation_variant_{v}_replica_{r}.json")
            for r in range(16)]
        for v in ("0", "1", "2")
    }
    identity_rows = (
        _payload_identity_checks(prod_records, "production")
        + _payload_identity_checks(conf_records, "confirmation")
    )

    # -- Leg A: the sealed census re-derivation (pure integer pooling) --
    prod_report = _read_json(MONO_ROOT / R32_PROD_DIR / SPREAD_REPORT_NAME)
    gate_record = _read_json(MONO_ROOT / R32_PROD_DIR / GATE_RECORD_NAME)
    conf_verification = _read_json(
        MONO_ROOT / R32_CONF_DIR / CONFIRMATION_VERIFICATION_NAME)
    prod_pooled = load_payload_census(
        R32_PROD_DIR, "variant_{v}_replica_{r}.json", 8)
    conf_pooled = load_payload_census(
        R32_CONF_DIR, "confirmation_variant_{v}_replica_{r}.json", 16)

    comparisons = []
    max_disagreement = 0.0
    for v in ("0", "1", "2"):
        sealed_prod = prod_report["routing_census"][v]
        fields = census_disagreement(prod_pooled[v], sealed_prod)
        gate_fraction = float(
            gate_record["clauses"]["no_boundary_dominance"]
            ["routing_census"][f"v{v}"])
        conf_sealed = conf_verification["routing_census"][v]
        fields_conf = census_disagreement(conf_pooled[v], conf_sealed)
        worst = max(
            max(fields.values()), max(fields_conf.values()),
            abs(prod_pooled[v]["fraction_catalog"] - gate_fraction),
        )
        max_disagreement = max(max_disagreement, worst)
        comparisons.append({
            "variant": int(v),
            "production": {
                "recomputed": prod_pooled[v],
                "sealed": sealed_prod,
                "disagreements": fields,
            },
            "gate_fraction_catalog_quoted": gate_fraction,
            "confirmation": {
                "recomputed": conf_pooled[v],
                "sealed": conf_sealed,
                "disagreements": fields_conf,
            },
            "worst_disagreement": worst,
        })
    leg_a_within = bool(max_disagreement <= CENSUS_DISAGREEMENT_BOUND)

    # -- Leg B: the probe census, re-derived from the traced rows -------
    source_rows = source_rows_of(trace)
    band_rows = [row for row in source_rows
                 if row["region"] == "transition"]
    rederived_catalog = int(sum(
        1 for row in source_rows if row["route"] == "catalog"))
    probe_census = trace.get("probe_census", {})
    matches_trace_census = bool(
        len(source_rows) == int(probe_census.get("n_sources", -1))
        and rederived_catalog == int(probe_census.get("catalog_routed", -2))
        and (len(source_rows) - rederived_catalog)
        == int(probe_census.get("continuum_routed", -2))
    )
    band_outcomes_ok = bool(all(
        row["route"] == _expected_route(row["u"], row["mass"])
        for row in band_rows))
    below_deterministic = bool(all(
        row["route"] == "catalog"
        for row in source_rows if row["region"] == "catalog_region"))
    above_deterministic = bool(all(
        row["route"] == "continuum"
        for row in source_rows if row["region"] == "continuum_region"))
    leg_b_exact = bool(matches_trace_census and band_outcomes_ok
                       and below_deterministic and above_deterministic)

    # -- Leg C (informational): the w(M)-implied bracket ----------------
    sealed_fraction = float(PRODUCTION_CENSUS_V0["fraction_catalog"])
    n_total_sealed = 2.0 * BIT_IDENTITY_EVENTS
    p_below = (BIT_IDENTITY_BAND_CENSUS["n_below_boundary_sources"]
               / n_total_sealed)
    p_band = (BIT_IDENTITY_BAND_CENSUS["n_in_band_sources"]
              / n_total_sealed)
    leg_c_bracket = {
        "p_below_boundary": p_below,
        "p_in_band": p_band,
        "bracket": [p_below, p_below + p_band],
        "sealed_fraction": sealed_fraction,
        "sealed_fraction_in_bracket": bool(
            p_below <= sealed_fraction <= p_below + p_band),
        "note": (
            "informational: the routing-weight consistency at the census "
            "level; the sealed production census fraction lies inside the "
            "w(M)-implied bracket [P(M < B), P(M < B) + P(B <= M < B + W)] "
            "of the sealed bit-identity stream's band census"
        ),
    }

    payload = {
        "schema": "na49_mixing.audit_r4_builder_blend_path.census/1",
        "run": RUN_ID,
        "card": CARD,
        "probe": (
            "the probe routing census against the sealed R32 v0 routing "
            "census (the probe re-derives, the sealed archive is the "
            "reference)"
        ),
        "leg_a": {
            "method": (
                "the sealed R32 v0 routing census re-derived from the "
                "sealed per-replica payload routing counts (production 8, "
                "confirmation 16; pure integer pooling, zero draws) and "
                "compared with the sealed census records (the spread "
                "report, the gate record, the confirmation verification); "
                "the sealed archive is the reference and the comparison "
                "span is the full realized source-mass span of the "
                "sealed production, identical on both sides by "
                "construction"
            ),
            "disagreement_bound": CENSUS_DISAGREEMENT_BOUND,
            "max_disagreement": float(max_disagreement),
            "within_bound": bool(leg_a_within),
            "variants": comparisons,
        },
        "leg_b": {
            "method": (
                "the probe's own routing census re-derived independently "
                "from the traced (mass, u, w, route) rows and asserted "
                "internally exact"
            ),
            "probe_census_matches": bool(matches_trace_census),
            "band_outcomes_rederived_ok": bool(band_outcomes_ok),
            "region_determinism_ok": bool(below_deterministic
                                          and above_deterministic),
            "n_band_rows": len(band_rows),
            "exact": bool(leg_b_exact),
        },
        "leg_c_informational": leg_c_bracket,
        "census_within_bound": bool(leg_a_within and leg_b_exact),
        "payload_identity_stamps": identity_rows,
        "no_fitting": True,
        "optimizer_invoked": False,
        "draws": 0,
    }
    _write_json(run_dir / CENSUS_NAME, payload)
    print(
        f"audit_r4_builder_blend_path: census written (max disagreement "
        f"{max_disagreement:.3e} against the bound "
        f"{CENSUS_DISAGREEMENT_BOUND:.0e}; leg B exact {leg_b_exact})",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# Mode: --verdict
# ---------------------------------------------------------------------------


def evaluate_verdict_criteria(trace: dict, isolation: dict,
                              census: dict) -> dict:
    """The pre-registered verdict criteria (a), (b), (c), (d) evaluated
    on the three measurement artifacts (a pure function; zero draws)."""
    criterion_a = {
        "criterion": "a",
        "statement": "the draw budget is exact in all three regions",
        "draw_budget_exact": bool(trace.get("draw_budget_exact")),
        "region_budgets_exact": {
            region: bool(block["budget_exact"])
            for region, block in trace.get("region_budgets", {}).items()
        },
        "fired": not bool(trace.get("draw_budget_exact")),
    }
    criterion_b = {
        "criterion": "b",
        "statement": "the below-boundary path re-verifies bit-identity",
        "reverified": bool(trace.get("bit_identity_leg", {}).get(
            "reverified")),
        "fired": not bool(trace.get("bit_identity_leg", {}).get(
            "reverified")),
    }
    criterion_c = {
        "criterion": "c",
        "statement": (
            "the boundary-matching factor is never consulted on the "
            "sampling path"
        ),
        "factor_isolated": bool(isolation.get("factor_isolated")),
        "fired": not bool(isolation.get("factor_isolated")),
    }
    criterion_d = {
        "criterion": "d",
        "statement": (
            "the census is consistent within 1e-6 (a driver defect, "
            "rework the driver, not the surface)"
        ),
        "census_within_bound": bool(census.get("census_within_bound")),
        "leg_a_within_bound": bool(
            census.get("leg_a", {}).get("within_bound")),
        "leg_b_exact": bool(census.get("leg_b", {}).get("exact")),
        "fired": not bool(census.get("census_within_bound")),
    }
    fired = [c for c in (criterion_a, criterion_b, criterion_c, criterion_d)
             if c["fired"]]
    return {
        "criteria": {"a": criterion_a, "b": criterion_b,
                     "c": criterion_c, "d": criterion_d},
        "fired": fired,
        "verdict": "RECOMMENDATION" if fired else "PASS",
    }


def _finding_narrative(mismatch_row: dict, kind: str) -> dict:
    """The named-finding narrative of one mismatched draw-budget row
    (region, event fixture, observed draw sequence, diagnosis, the
    proposed rework)."""
    observed = (
        f"routing uniforms {mismatch_row['routing_uniforms']} vs the "
        f"pre-registered {mismatch_row['routing_uniform_expected']}; "
        f"path draws {mismatch_row['path_draws']} vs the pre-registered "
        f"1; route {mismatch_row['route']!r} vs the hand-derived "
        f"{mismatch_row['route_expected']!r}; stamped blend_weight "
        f"{mismatch_row['blend_weight_stamped']!r} vs the pre-registered "
        f"w(M) {mismatch_row['blend_weight_expected']!r} at mass "
        f"{mismatch_row['mass']!r} GeV"
    )
    diagnosis = (
        "the routed sampling path leaves the pre-registered draw budget "
        "or the pre-registered w(M) routing weight at the frozen source "
        "mass; the card-6 blend routing contract (exactly the record "
        "draw and the w(M) routing uniform, never the boundary-matching "
        "factor) is not met on the pinned builder surface"
    )
    return {
        "kind": kind,
        "region": mismatch_row["region"],
        "event_id": mismatch_row["event_id"],
        "source_id": mismatch_row["source_id"],
        "observed": observed,
        "diagnosis": diagnosis,
        "responsible": (
            "the builder blend path "
            f"({BUILDER_REL}, sha256 {BUILDER_SHA256}, byte-unchanged by "
            "this audit)"
        ),
        "proposed_rework": (
            "a separate Round 4 rework card on the builder blend path "
            "surface to re-derive the routed sampling contract against "
            "the card-6 routing (the draw budget and the w(M) weight), "
            "with any new probe set pre-registered and the R32 bounds "
            "(2.0 / 3.0) carried; the rework card may alternatively "
            "confirm the surface as-is and re-classify the finding as "
            "standing model uncertainty"
        ),
        "evidence": dict(mismatch_row),
    }


def run_verdict(run_dir: Path) -> dict:
    """Assemble the verdict artifact (PASS or RECOMMENDATION with named
    findings)."""
    run_dir = Path(run_dir)
    pre = assert_pre_registration(run_dir)
    artifacts = {}
    for name, key in (
        (TRACE_NAME, "trace"),
        (ISOLATION_NAME, "isolation"),
        (CENSUS_NAME, "census"),
    ):
        path = run_dir / name
        if not path.is_file():
            _refuse(
                f"{path} is missing; the verdict requires the three "
                "measurement steps (run --draw-budget, --isolation, "
                "--census first)"
            )
        artifacts[key] = _read_json(path)
    evaluation = evaluate_verdict_criteria(
        artifacts["trace"], artifacts["isolation"], artifacts["census"])
    findings = []
    if evaluation["criteria"]["a"]["fired"]:
        rows = source_rows_of(artifacts["trace"])
        for region, block in artifacts["trace"]["region_budgets"].items():
            for row in block["mismatches"]:
                findings.append(_finding_narrative(row, "draw_budget"))
    if evaluation["criteria"]["b"]["fired"]:
        findings.append({
            "kind": "bit_identity",
            "region": None,
            "event_id": None,
            "source_id": None,
            "observed": (
                "the below-boundary bit-identity leg does not re-verify "
                "(layer matched "
                f"{artifacts['trace']['bit_identity_leg']['layer_matched']}"
                f"/{artifacts['trace']['bit_identity_leg']['layer_compared']}"
                ", below-boundary finals matched "
                f"{artifacts['trace']['bit_identity_leg']['below_boundary_finals_matched']}"
                "/"
                f"{artifacts['trace']['bit_identity_leg']['below_boundary_finals_compared']}"
                ", legacy finals matched "
                f"{artifacts['trace']['bit_identity_leg']['legacy_finals_matched']}"
                "/"
                f"{artifacts['trace']['bit_identity_leg']['legacy_finals_compared']}"
            ),
            "diagnosis": (
                "the sealed R32 v0 bit-identity leg fails at probe scale; "
                "a driver defect (rework the driver, never the surface) "
                "or a surface regression (the pinned builder bytes are "
                "unchanged, so a driver defect is the expected cause)"
            ),
            "responsible": (
                "the audit driver "
                "helpers/benchmarking/audit_r4_builder_blend_path.py"
            ),
            "proposed_rework": (
                "a separate rework card to fix and re-run the audit "
                "driver; the pinned builder surface is not implicated "
                "while its digest stays at the pinned value"
            ),
            "evidence": dict(artifacts["trace"]["bit_identity_leg"]),
        })
    if evaluation["criteria"]["c"]["fired"]:
        spy = artifacts["isolation"]["spy_fixture"]
        static = artifacts["isolation"]["static_assertion"]
        findings.append({
            "kind": "factor_isolation",
            "region": None,
            "event_id": None,
            "source_id": None,
            "observed": (
                f"spy consultations {spy['spy_consultations_total']} "
                f"({spy['spy_consultations']}); blend pieces cache "
                f"{spy['blend_pieces_cache_before']} -> "
                f"{spy['blend_pieces_cache_after']}; static problems "
                f"{static['problems']}"
            ),
            "diagnosis": (
                "the routed sampling path consults the double-counting "
                "boundary-matching factor of the card-6 blend (a read, "
                "an argument, or a side channel); the card's white-box "
                "contract is not met on the pinned builder surface"
            ),
            "responsible": (
                "the builder blend path "
                f"({BUILDER_REL}, sha256 {BUILDER_SHA256}, byte-unchanged "
                "by this audit)"
            ),
            "proposed_rework": (
                "a separate Round 4 rework card on the builder blend path "
                "surface to remove the boundary-matching-factor "
                "consultation from the sampling call path"
            ),
            "evidence": {
                "spy_fixture": dict(spy),
                "static_assertion": dict(static),
                "direct_route_contract": dict(
                    artifacts["isolation"]["direct_route_contract"]),
            },
        })
    if evaluation["criteria"]["d"]["fired"]:
        findings.append({
            "kind": "census_consistency",
            "region": None,
            "event_id": None,
            "source_id": None,
            "observed": (
                "max census disagreement "
                f"{artifacts['census']['leg_a']['max_disagreement']:.3e} "
                "against the 1e-6 bound (leg A) or the probe census is "
                "not internally exact (leg B)"
            ),
            "diagnosis": (
                "a driver defect; rework the driver, never the surface "
                "(the verdict contract criterion (d))"
            ),
            "responsible": (
                "the audit driver "
                "helpers/benchmarking/audit_r4_builder_blend_path.py"
            ),
            "proposed_rework": (
                "a separate rework card to fix and re-run the audit "
                "driver; the pinned builder surface is not implicated"
            ),
            "evidence": {
                "leg_a": dict(artifacts["census"]["leg_a"]),
                "leg_b": dict(artifacts["census"]["leg_b"]),
            },
        })
    verdict = evaluation["verdict"]
    recommendation_text = None
    if verdict == "RECOMMENDATION":
        parts = []
        if evaluation["criteria"]["a"]["fired"]:
            parts.append(
                "the routed sampling path leaves the pre-registered draw "
                "budget (0 / 1 / 0 routing uniforms across the catalog "
                "region, the transition band, and the continuum region) "
                "or the u < w(M) routing outcome"
            )
        if evaluation["criteria"]["b"]["fired"]:
            parts.append(
                "the below-boundary bit-identity leg does not re-verify "
                "at probe scale"
            )
        if evaluation["criteria"]["c"]["fired"]:
            parts.append(
                "the sampling path consults the double-counting "
                "boundary-matching factor (a read, an argument, or a "
                "side channel)"
            )
        if evaluation["criteria"]["d"]["fired"]:
            parts.append(
                "the census re-derivation disagrees with the sealed R32 "
                "v0 routing census beyond 1e-6, or the probe census is "
                "not internally exact (a driver defect)"
            )
        recommendation_text = (
            "Spawn a separate Round 4 rework card on the builder blend "
            f"path surface (na49_mixing/resonance_pp_builder.py, sha256 "
            f"{BUILDER_SHA256}, byte-unchanged by this audit): "
            + "; ".join(parts)
            + ". The rework card pre-registers its own probe set and "
            "carries the R32 bounds (bin spread p95 < 2.0 pooled-SE "
            "units, feed-down spread max < 3.0 sigma); it may "
            "alternatively confirm the surface as-is and re-classify "
            "the named findings as standing model uncertainty. The "
            "routing-formulation share is card 1's scope and the "
            "strange-sector family share is card 2's scope; the three "
            "audits are independent inputs to the Round 4 run cards."
        )
    payload = {
        "schema": "na49_mixing.audit_r4_builder_blend_path.verdict/1",
        "run": RUN_ID,
        "card": CARD,
        "verdict": verdict,
        "verdict_contract": pre["verdict_contract"],
        "probe_set": {
            "events": PROBE_EVENTS,
            "spy_fixture_events": SPY_EVENTS,
            "seed": BIT_IDENTITY_SEED,
            "kernel_master_seed": BIT_IDENTITY_MASTER_SEED,
            "kernel_replica_index": BIT_IDENTITY_REPLICA_INDEX,
            "source": pre["probe_set"]["source"],
        },
        "bounds": {
            "census_disagreement_bound": CENSUS_DISAGREEMENT_BOUND,
            "bin_spread_p95_bound": BIN_SPREAD_BOUND,
            "feeddown_spread_max_bound": FEEDDOWN_SPREAD_BOUND,
            "note": (
                "the pre-registered bounds are cited verbatim from the "
                "card and the R32 pre-registration; none is relaxed; "
                "this audit produces no new spread diagnostic"
            ),
        },
        "criteria": evaluation["criteria"],
        "findings": findings,
        "recommendation": recommendation_text,
        "draw_budget_summary": {
            "draw_budget_exact": bool(
                artifacts["trace"].get("draw_budget_exact")),
            "region_budgets": {
                region: {
                    "n_sources": block["n_sources"],
                    "routing_uniforms_total":
                        block["routing_uniforms_total"],
                    "path_draws_total": block["path_draws_total"],
                    "budget_exact": bool(block["budget_exact"]),
                }
                for region, block in
                artifacts["trace"].get("region_budgets", {}).items()
            },
            "bit_identity_reverified": bool(
                artifacts["trace"].get("bit_identity_leg", {}).get(
                    "reverified")),
        },
        "isolation_summary": {
            "factor_isolated": bool(
                artifacts["isolation"].get("factor_isolated")),
            "spy_consultations_total": int(
                artifacts["isolation"]["spy_fixture"][
                    "spy_consultations_total"]),
            "static_clean": bool(
                artifacts["isolation"]["static_assertion"]["clean"]),
            "direct_contract_clean": bool(
                artifacts["isolation"]["direct_route_contract"]["clean"]),
            "stamps_exact": bool(
                artifacts["isolation"]["routing_weight_consistency"][
                    "stamps_exact"]),
        },
        "census_summary": {
            "census_within_bound": bool(
                artifacts["census"].get("census_within_bound")),
            "max_disagreement": float(
                artifacts["census"]["leg_a"]["max_disagreement"]),
            "bound": float(CENSUS_DISAGREEMENT_BOUND),
        },
        "zero_draw": {
            "draws": 0,
            "no_fitting": True,
            "optimizer_invoked": False,
            "reserved_audit_block_fired": False,
        },
        "sealed_inputs": dict(SEALED_INPUT_DIGESTS),
        "surface": {
            "builder_path": BUILDER_REL,
            "builder_sha256": BUILDER_SHA256,
            "card6_path": CARD6_REL,
            "card6_sha256": CARD6_SHA256,
            "byte_unchanged_by_this_audit": True,
        },
    }
    _write_json(run_dir / VERDICT_NAME, payload)
    print(
        f"audit_r4_builder_blend_path: verdict {verdict} "
        f"({len(findings)} named findings; criteria fired: "
        f"{[c['criterion'] for c in evaluation['fired']] or 'none'})",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# Modes: --seal and --clean-validate
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
        "schema": "na49_mixing.audit_r4_builder_blend_path"
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
    """Seal the finished audit run directory (sums inventory plus seal
    manifest).  The pre-registration, the provenance, the three
    measurement artifacts, and the verdict must exist."""
    run_dir = Path(run_dir)
    for name in (
        PREREGISTRATION_NAME,
        PROVENANCE_NAME,
        TRACE_NAME,
        ISOLATION_NAME,
        CENSUS_NAME,
        VERDICT_NAME,
    ):
        if not (run_dir / name).is_file():
            _refuse(
                f"{run_dir / name} is missing before sealing (the verdict "
                "is part of the seal; run the measurement modes and "
                "--verdict first)"
            )
    if (run_dir / CLEAN_ENV_NAME).is_file():
        _refuse(
            f"{run_dir / CLEAN_ENV_NAME} exists; the clean-environment "
            "record is written after the seal and the seal is never "
            "re-run over it"
        )
    write_sha256sums(run_dir)
    seal = write_seal_manifest(run_dir)
    print(
        f"audit_r4_builder_blend_path: run dir sealed "
        f"({len(seal['files'])} files listed in {SEAL_NAME})",
        flush=True,
    )
    return seal


def run_clean_validate(run_dir: Path) -> dict:
    """Verify the seal, re-run the four deterministic measurement steps
    from a clean interpreter over the sealed artifacts, and confirm
    every artifact hash matches."""
    run_dir = Path(run_dir)
    if not (run_dir / PREREGISTRATION_NAME).is_file():
        _refuse(f"{run_dir / PREREGISTRATION_NAME} is missing")
    sums_path = run_dir / SHA256SUMS_NAME
    if not sums_path.is_file():
        _refuse(
            f"{sums_path} is missing; --clean-validate requires the "
            "sealed run dir (run --seal first)"
        )
    # -- seal verification (root-relative entries) ------------------------
    mismatches: list[str] = []
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
        _refuse(
            "seal verification failed; "
            f"{len(mismatches)} of {files_checked} SHA256SUMS.txt entries "
            f"do not re-hash correctly (first: {mismatches[0]!r}); the "
            "sealed run dir must be unmodified"
        )
    # -- clean-interpreter re-run of the four measurement steps -----------
    artifact_names = (TRACE_NAME, ISOLATION_NAME, CENSUS_NAME, VERDICT_NAME)
    before = {
        name: (run_dir / name).read_bytes() for name in artifact_names
    }
    driver_path = Path(__file__).resolve()
    commands = []
    probe_results = []
    for mode in ("draw-budget", "isolation", "census", "verdict"):
        argv = [
            sys.executable, "-I", str(driver_path),
            f"--{mode}", "--run-dir", str(run_dir),
        ]
        commands.append({
            "purpose": (
                f"clean-interpreter re-run of the {mode} step over the "
                "sealed artifacts (rewrites its artifact; the sealed "
                "bytes must be reproduced byte for byte)"
            ),
            "argv": argv,
            "cwd": str(MONO_ROOT),
        })
        probe = subprocess.run(argv, capture_output=True, text=True,
                               cwd=str(MONO_ROOT))
        probe_results.append((mode, probe.returncode, probe.stderr))
    recomputed = {
        name: (run_dir / name).read_bytes() for name in artifact_names
    }
    hashes_match = all(
        int(rc) == 0
        and hashlib.sha256(before[name]).hexdigest()
        == hashlib.sha256(recomputed[name]).hexdigest()
        and before[name] == recomputed[name]
        for name, (_mode, rc, _err) in zip(artifact_names, probe_results)
    )
    # Restore the sealed bytes: the probes rewrote the artifacts; the
    # sealed dir must keep its sealed content regardless of the probe
    # outcome.
    for name in artifact_names:
        if recomputed[name] != before[name]:
            _write_bytes(run_dir / name, before[name])
    failed = [
        {"mode": mode, "returncode": int(rc), "stderr": err[-2000:]}
        for mode, rc, err in probe_results if int(rc) != 0
    ]
    payload = {
        "schema": "na49_mixing.audit_r4_builder_blend_path"
                  ".clean_environment/1",
        "run": RUN_ID,
        "card": CARD,
        "verified_utc": _now_utc(),
        "seal_verification": {
            "files_checked": int(files_checked),
            "mismatches": mismatches,
        },
        "commands": commands,
        "environment": environment_fingerprint(),
        "probe_returncodes": {
            mode: int(rc) for mode, rc, _err in probe_results
        },
        "probe_failures": failed,
        "sealed_artifact_sha256": {
            name: hashlib.sha256(before[name]).hexdigest()
            for name in artifact_names
        },
        "recomputed_artifact_sha256": {
            name: hashlib.sha256(recomputed[name]).hexdigest()
            for name in artifact_names
        },
        "hashes_match": bool(hashes_match),
        "probe_tool": {
            "path": str(driver_path),
            "sha256": _sha256_file(driver_path),
            "note": (
                "the committed dual-tree driver is the probe instrument; "
                "the sealed artifacts and the sealed R32 archives are the "
                "inputs"
            ),
        },
    }
    _write_json(run_dir / CLEAN_ENV_NAME, payload)
    if not payload["hashes_match"]:
        print(
            "audit_r4_builder_blend_path: the clean-environment "
            "validation FAILED (the recomputed artifact bytes differ "
            "from the sealed artifacts)",
            flush=True,
        )
        raise SystemExit(1)
    print(
        "audit_r4_builder_blend_path: clean-environment validation "
        f"passed ({files_checked} seal files verified, the clean "
        "interpreter reproduced all four measurement artifacts byte "
        "for byte)",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audit_r4_builder_blend_path.py",
        description=(
            "Execute the Round 4 builder blend-path routing-weight audit "
            "(card TASK-20260920-r4-audit-builder-blend-path-routing-"
            "weight, Gate 3 follow-up obligation 3): confirm the "
            "ResonancePPBuilder blend path contributes no routing "
            "artifact beyond the pre-registered w(M) routing weight on "
            "the byte-stable builder surface.  Zero new random numbers; "
            "read-only on every pinned surface and sealed archive; PASS "
            "or RECOMMENDATION (a separate rework card), never an "
            "in-audit surface edit."
        ),
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--pre-registration", action="store_true",
        help="write pre_registration.json and provenance.json, then stop",
    )
    modes.add_argument(
        "--draw-budget", action="store_true",
        help="the 1,000-event fixed-seed traced probe and the per-region "
             "budget table, then stop",
    )
    modes.add_argument(
        "--isolation", action="store_true",
        help="the boundary-matching-factor isolation and the routing-"
             "weight consistency fixture, then stop",
    )
    modes.add_argument(
        "--census", action="store_true",
        help="the sealed-census re-derivation and the probe census, then "
             "stop",
    )
    modes.add_argument(
        "--verdict", action="store_true",
        help="write verdict.json (PASS or RECOMMENDATION with named "
             "findings), then stop",
    )
    modes.add_argument(
        "--seal", action="store_true",
        help="write seal_manifest.json and SHA256SUMS.txt, then stop",
    )
    modes.add_argument(
        "--clean-validate", action="store_true",
        help="verify the seal and re-run the four measurement steps from "
             "a clean interpreter, then stop",
    )
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help=(
            "the audit run dir (default: "
            f"dev/00runs/{RUN_LABEL}_<execution date> under the "
            "monorepo root); pass an absolute path"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_dir = Path(
        args.run_dir.resolve() if args.run_dir else default_run_dir()
    )
    if args.pre_registration:
        run_pre_registration(run_dir)
    elif args.draw_budget:
        run_draw_budget(run_dir)
    elif args.isolation:
        run_isolation(run_dir)
    elif args.census:
        run_census(run_dir)
    elif args.verdict:
        run_verdict(run_dir)
    elif args.seal:
        run_seal(run_dir)
    elif args.clean_validate:
        run_clean_validate(run_dir)
    else:  # pragma: no cover - argparse enforces exactly one mode
        _refuse("exactly one mode is required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
