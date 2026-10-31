#!/usr/bin/env python3
"""R4 audit driver: the transition routing formulation audit (Round 4, card 1).

Task ``TASK-20260920-r4-audit-transition-routing-formulation`` (Gate 3
decision card follow-up obligation 1; plan section 9.3, lines 625-632;
the Round 4 index card, cards 1-3).  The audit resolves the
K+-concentrated persisting spread of the sealed R32 evidence against the
blend ramp placement and width, on the byte-stable card-6 surface
(``na49_mixing/spectral_density.py``, sha256 ``6f889f6d...``).  The
audit is read-only on every pinned surface and on every sealed archive;
its outcome is a PASS verdict (the surface is adequate as-is) or a
RECOMMENDATION that spawns a separate rework card (the cards 8a/8b
precedent of Round 3).  The audit itself never edits the pinned surface.

Zero draws (hermetic, no Monte Carlo): the audit is an analytic
re-evaluation of the sealed R32 archives.  The probe set is the 32
persisting residual register entries of the sealed confirmation
(``dev/00runs/r32_spectral_systematic_pp_confirmation_2026-09-19/``,
18 K+ of 32).  No event is generated, no random number is drawn, no
optimizer runs (``no_fitting`` is true, ``optimizer_invoked`` is false,
``draws`` is 0; the pre-registered audit confirmation block 20301040 + k
under the Round 4 master seed 20260920 is reserved for the exceptional
case and must not fire on the first pass).

Probes (the card's pre-registered measurement contract, index item 6):

- **Census recompute** (``--census``): the per-variant routing census
  is recomputed from the sealed per-replica payload routing counts
  (production 3 x 8, confirmation 3 x 16) and compared with the sealed
  routing census of the spread report, the gate record, and the
  confirmation verification.  Disagreement bound 1e-6 (verdict
  criterion (c): a larger disagreement is a driver defect; rework the
  driver, never the surface).

- **Placement attribution** (``--attribution``): for each persisting
  bin, the per-variant (v0, v1, v2) delta in pooled-SE units is
  recorded against the median-variant reference (the sealed register
  convention; the reference cancels in the monotonicity test).  The
  attribution test is the monotonicity of the bin delta with the ramp
  start (the decision card diagnosis 4 mechanism: the more a variant
  routes sources to the continuum family, the more the bins move).
  Pre-registered placement-explained predicate, both parts evaluated at
  the sealed confirmation statistics:

  (P1) monotonicity: with the variants ordered by ramp start
  (v1 = B - 0.2 < v0 = B < v2 = B + 0.2), the delta sequence is
  monotone (non-decreasing or non-increasing);

  (P2) census-magnitude: the nominal variant's off-line residual from
  the census interpolation of the early and late variants,
  ``r = y_v0 - ((1 - lambda) y_v1 + lambda y_v2)`` with
  ``lambda = (c_v0 - c_v1) / (c_v2 - c_v1)`` from the sealed
  confirmation routing census, stays within 4.0 pooled-SE units (the
  finite-MC envelope of the sealed R32 named-residual classification).

  A bin is placement-explained iff (P1) and (P2) both hold.  The
  implied route contrast of each bin (its spread per unit of census
  shift) and the mass-blind route-contrast envelope are recorded
  informationally beside the verdict-gating predicate; the mass-blind
  envelope flags are evidence for the card-2 referral, never a verdict
  criterion of this card.

- **Width sensitivity** (``--reweight``): the sealed per-bin payloads
  are reweighted analytically for hypothetical ramp widths
  W' in {0.3, 0.7 GeV} at the same ramp starts (zero draws).  The
  census sensitivity is calibrated from the sealed ramp-start scan
  itself: the per-variant local catalog-census slope
  ``s_v = dc/dB`` (the adjacent-pair finite differences), the exact
  width identity ``dc/dW = <q>_band m_band / W^2`` linearized with the
  pre-registered neutral in-band position ``alpha = 0.5``
  (``<q>_band / W``; the alpha in (0, 1) bracket is carried as the
  envelope), giving the reweighted census
  ``c_v' = c_v + (W' - W) alpha s_v``.  Each bin's route contrast
  ``m_b = (y_v2 - y_v1) / (c_v2 - c_v1)`` reweights its per-variant
  values ``y_v' = y_v + m_b (c_v' - c_v)``; the reweighted bin spread
  and its pooled-SE sigma follow; the reweighted bin-spread p95 is the
  95th percentile over the finite reweighted sigmas (the sealed R32
  percentile convention).  Pre-registered contract: W = 0.5 GeV is
  adequate if the reweighted p95 stays within 20% of the production
  p95 2.9604990756942544 for both W'; a shift beyond 20% is a width
  finding (verdict criterion (b)).

- **Verdict** (``--verdict``): PASS if the persisting spread is
  explained by the ramp placement (the monotone attribution on the
  non-strange bins) and the strange-sector concentration (18 K+) is
  referred to card 2 (the originating families audit) as the
  responsible surface; RECOMMENDATION (separate rework card) with
  named findings if (a) any non-strange persisting bin is not
  placement-explained, (b) the width sensitivity exceeds the 20%
  contract, or (c) the census recomputation disagrees with the sealed
  routing census by more than 1e-6.

New spread diagnostics carry the R32 pre-registered bounds (bin spread
p95 < 2.0 pooled-SE units, feed-down spread max < 3.0 sigma): the
reweighted p95 rows record the bound beside the value; the bound is
never relaxed and its failure is a recorded finding, never a refit.

Modes (exactly one per invocation; each performs one step and stops):

- ``--pre-registration``: write ``pre_registration.json`` and
  ``provenance.json`` into the run dir, then stop.  Nothing is
  measured.  Verifies the sealed R32 production and confirmation seals,
  the sealed R30/R31/card-8c cross-reference seals, and the frozen
  substrate digests first.
- ``--census``: recompute the routing census from the sealed per-replica
  payloads, write ``routing_census_recomputed.json``, then stop.
  Requires the pre-registration.
- ``--attribution``: the per-variant delta pattern of the 32 persisting
  residuals, the monotone test, and the K+ referral, write
  ``persisting_bin_attribution.json``, then stop.  Requires the census
  step.
- ``--reweight``: the W' in {0.3, 0.7 GeV} analytic reweighting and the
  20% contract, write ``spread_reweighted_W.json``, then stop.
  Requires the attribution step.
- ``--verdict``: write ``verdict.json`` (PASS or RECOMMENDATION with
  named findings), then stop.  Requires the three measurement steps.
- ``--seal``: write ``seal_manifest.json`` and ``SHA256SUMS.txt`` over
  the run dir, then stop.  The run dir is immutable after the seal.
- ``--clean-validate``: verify the seal, re-run the four deterministic
  measurement steps from a clean interpreter over the sealed artifacts,
  confirm every artifact hash matches, write
  ``clean_environment_commands.json``, then stop.

The four measurement artifacts (census, attribution, reweight, verdict)
carry no wall-clock fields: their bytes are a pure function of the
sealed archives and the committed driver, which is what the
clean-environment validation re-hashes.

Dual-tree driver: byte-identical in the nested tree
(``dev/mixing2006/helpers/benchmarking/``) and the monorepo mirror
(``helpers/benchmarking/``); both locate the sealed archives with the
established ``_locate_roots`` pattern.  Hermetic in every mode: standard
library and numpy only; the sealed archives, the committed catalog, and
the committed continuum config only; no ``na49_mixing`` import, no
external SPS tables, no event generation, no random module.
"""

from __future__ import annotations

import argparse
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
# monorepo mirror (helpers/benchmarking/).  Both must find the sealed
# archives under the monorepo root.
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
        f"audit_r4_routing_formulation: {message}",
        file=sys.stderr, flush=True,
    )
    raise SystemExit(2)


NESTED_ROOT, MONO_ROOT = _locate_roots()

# ---------------------------------------------------------------------------
# Pre-registered audit identity (card TASK-20260920-r4-audit-transition-
# routing-formulation; the probe set, the bounds, and the verdict
# contract are verbatim from the card and the round 4 index item 6).
# ---------------------------------------------------------------------------

CARD = "TASK-20260920-r4-audit-transition-routing-formulation"
RUN_ID = "R4-audit-transition-routing-formulation"
RUN_LABEL = "r4_audit_transition_routing_formulation"

PREREGISTRATION_NAME = "pre_registration.json"
PROVENANCE_NAME = "provenance.json"
CENSUS_NAME = "routing_census_recomputed.json"
ATTRIBUTION_NAME = "persisting_bin_attribution.json"
REWEIGHT_NAME = "spread_reweighted_W.json"
VERDICT_NAME = "verdict.json"
CLEAN_ENV_NAME = "clean_environment_commands.json"
SHA256SUMS_NAME = "SHA256SUMS.txt"
SEAL_NAME = "seal_manifest.json"

# The byte-stable card-6 surface and the frozen substrate (read-only).
CARD6_REL = "dev/mixing2006/na49_mixing/spectral_density.py"
CARD6_SHA256 = "6f889f6d72968de53e7b27e32ba64d124d1b4e782a73606e6f6794eef9f156a9"
BUILDER_REL = "dev/mixing2006/na49_mixing/resonance_pp_builder.py"
BUILDER_SHA256 = "4ab612609e1ee0bc1e037dcbb2815b6484306ad32402b7289e6636f8e829cdb1"
ENGINE_REL = "dev/mixing2006/na49_mixing/threshold_metric.py"
ENGINE_SHA256 = "ee240f44645d2e592acfae462f7d1e9c3b3dd8cd78c96bd40b7b73509f7fcc63"
CATALOG_REL = "dev/mixing2006/config/v2_singlet/resonance_catalog.json"
CATALOG_SHA256 = "e42a5da7137d789ba90221478c9bb404898f4fe8a7ee4604cd85b5623b751752"
FAMILIES_REL = "dev/mixing2006/config/v2_singlet/continuum_resonances.json"
FAMILIES_SHA256 = "6da57d5709b94061bc178fe94781dfe5eddcc597c5dde46bfcd5db56985cde7d"
R32_DRIVER_NAME = "staged_r32_spectral_systematic.py"
R32_DRIVER_SHA256 = "99454facce856c93505e6c4b0d4e47b85eeb0ef54f72b1631e3d29fcc511cb90"

# The sealed R32 archives (the probe inputs) and the sealed
# cross-reference archives (read-only; their seals verify).
R32_PROD_DIR = "dev/00runs/r32_spectral_systematic_pp_2026-09-18"
R32_CONF_DIR = "dev/00runs/r32_spectral_systematic_pp_confirmation_2026-09-19"
R30_DIR = "dev/00runs/r30_spectral_normalization_2026-09-17"
R31_DIR = "dev/00runs/r31_catalog_continuum_matching_scan_2026-09-17"
R3_SE_AUDIT_DIR = "dev/00runs/r3_se_audit_2026-09-18"

SPREAD_REPORT_NAME = "spread_report.json"
GATE_RECORD_NAME = "gate3_r32.json"
CONFIRMATION_VERIFICATION_NAME = "confirmation_verification.json"

# Sealed R32 anchor digests (asserted by the pre-registration; the
# measurement modes re-hash every input they read against the
# pre-registered values).
SEALED_INPUT_DIGESTS = {
    f"{R32_PROD_DIR}/spread_report.json":
        "ed102b10e9c99d973a9c3e9dd262536eaca08802aeb8f313ea1894f7cefe022e",
    f"{R32_PROD_DIR}/gate3_r32.json":
        "64f104a8f0fdb71fdf506a3eb6a369091a2f3eed5047e499f1d344ad5668bab6",
    f"{R32_CONF_DIR}/confirmation_verification.json":
        "0553fed06a9b59e9f2020c0c53bbad7d75a11b1f2baf21f0469ee8b3e48d275f",
}

# Sealed R32 anchor numbers (asserted against the sealed artifacts).
PRODUCTION_P95 = 2.9604990756942544
CONFIRMATION_P95 = 4.028426092137674
PRODUCTION_CENSUS_CATALOG = {
    "0": 0.22329625,
    "1": 0.193558125,
    "2": 0.253725,
}
CONFIRMATION_CENSUS_CATALOG = {
    "0": 0.22357,
    "1": 0.1936659375,
    "2": 0.253850625,
}
PERSISTING_TOTAL = 32
PERSISTING_SPECIES_SPLIT = {
    "k_plus": 18,
    "proton": 9,
    "pi_plus": 4,
    "pi_minus": 1,
}
PERSISTING_SIGMA_MIN = 2.0002
PERSISTING_SIGMA_MAX = 7.5329
# The register quotes the sigma range rounded to four decimals; the
# range check carries the rounding tolerance.
PERSISTING_SIGMA_TOLERANCE = 5e-4

# The blend variants (the R32 pre-registration, verbatim): the ramp
# width W = 0.5 GeV fixed, the ramp start B / B - 0.2 / B + 0.2 GeV.
KERNEL_BOUNDARY_GEV = 2.680194110727528
RAMP_WIDTH_GEV = 0.5
RAMP_OFFSETS_GEV = {"0": 0.0, "1": -0.2, "2": 0.2}
# Ramp-start order: v1 (B - 0.2) < v0 (B) < v2 (B + 0.2).
RAMP_START_ORDER = ("1", "0", "2")

# Pre-registered bounds and contracts.
BIN_SPREAD_BOUND = 2.0
FEEDDOWN_SPREAD_BOUND = 3.0
CENSUS_DISAGREEMENT_BOUND = 1e-6
WIDTH_CONTRACT_REL = 0.20
WIDTH_PRIMES_GEV = (0.3, 0.7)
ALPHA_NEUTRAL = 0.5
ALPHA_BRACKET = (0.0, 1.0)
OFFLINE_ENVELOPE_SE = 4.0
# The envelope comparison carries a float-slop tolerance (the sealed
# register values are IEEE doubles; the envelope granularity is the
# 0.1-SE scale of the register quotes).
OFFLINE_ENVELOPE_TOLERANCE = 1e-9

# The reserved audit confirmation block (the exceptional case; must not
# fire on the first pass).  Master seed 20260920 (the Round 4 master),
# raw seeds 20301040 + k, k = 0..15, disjoint from every previously used
# raw block and from the Round 4 run blocks 20301020-20301039.
RESERVED_MASTER_SEED = 20260920
RESERVED_BLOCK_FIRST = 20301040
RESERVED_BLOCK_LAST = 20301055

PERSISTING_NON_STRANGE = ("proton", "pi_plus", "pi_minus")
STRANGE_REFERRAL_CARD = "TASK-20260920-r4-audit-strange-sector-continuum-families"


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    monorepo-root-relative paths (the R21/R31/R32 convention) or
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
        "card6_surface_sha256": _sha256_file(MONO_ROOT / CARD6_REL),
        "builder_module_sha256": _sha256_file(MONO_ROOT / BUILDER_REL),
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
    """Re-hash the sealed R32 anchor inputs against the pre-registered
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
# Mode: --pre-registration
# ---------------------------------------------------------------------------


def build_pre_registration(substrate: dict, seals: list) -> dict:
    """The audit pre-registration payload, written before any
    measurement: the probe set, the bounds, the verdict contract, and
    the zero-draw assertion."""
    return {
        "schema": "na49_mixing.audit_r4_routing_formulation.pre_registration/1",
        "run": RUN_ID,
        "card": CARD,
        "written_before_measurement": True,
        "surface": {
            "card6_blend_routing": {
                "path": CARD6_REL,
                "sha256": CARD6_SHA256,
                "note": (
                    "the card-6 surface is byte-unchanged by this audit; "
                    "a surface change is a separate rework card, never an "
                    "in-audit edit"
                ),
            },
            "builder_module_sha256": BUILDER_SHA256,
            "engine_module_sha256": ENGINE_SHA256,
            "catalog_sha256": CATALOG_SHA256,
            "families_config_sha256": FAMILIES_SHA256,
            "r32_driver_sha256": R32_DRIVER_SHA256,
        },
        "probe_set": {
            "source": (
                "the 32 persisting residual register entries of the sealed "
                f"R32 confirmation ({R32_CONF_DIR}/"
                "confirmation_verification.json, residual_reclassification "
                "classification == 'persisting')"
            ),
            "expected_entries": PERSISTING_TOTAL,
            "expected_species_split": PERSISTING_SPECIES_SPLIT,
            "expected_confirmation_sigma_range": [
                PERSISTING_SIGMA_MIN, PERSISTING_SIGMA_MAX,
            ],
            "payload_source": (
                "the per-variant bin payloads and the pooled SEs are read "
                "from the sealed archives (the register entries and the "
                "sealed bin_checks rows); no recomputation from events, no "
                "new draws"
            ),
        },
        "variants": {
            "ramp_width_gev": RAMP_WIDTH_GEV,
            "ramp_start_offsets_gev": RAMP_OFFSETS_GEV,
            "ramp_start_order": list(RAMP_START_ORDER),
            "kernel_boundary_gev": KERNEL_BOUNDARY_GEV,
            "note": (
                "v0 (nominal) ramp start B = 2.680194110727528 GeV, v1 "
                "(early) B - 0.2, v2 (late) B + 0.2; the ramp width W = 0.5 "
                "GeV fixed (the decision card scope)"
            ),
        },
        "census_recompute": {
            "method": (
                "the per-variant routing census pooled from the sealed "
                "per-replica payload routing counts (production 3 x 8, "
                "confirmation 3 x 16); the comparison set is the sealed "
                "routing census of the spread report, the gate record, and "
                "the confirmation verification"
            ),
            "disagreement_bound": CENSUS_DISAGREEMENT_BOUND,
            "bound_note": (
                "a disagreement beyond 1e-6 is a driver defect (verdict "
                "criterion (c)); rework the driver, never the surface"
            ),
        },
        "placement_attribution": {
            "delta_definition": (
                "the per-variant (v0, v1, v2) delta in pooled-SE units "
                "against the median-variant reference (the sealed register "
                "convention; the reference cancels in the monotonicity "
                "test)"
            ),
            "p1_monotonicity": (
                "with the variants ordered by ramp start (v1 < v0 < v2) "
                "the delta sequence is monotone (non-decreasing or "
                "non-increasing)"
            ),
            "p2_census_magnitude": (
                "the nominal variant's off-line residual from the census "
                "interpolation of the early and late variants, "
                "r = y_v0 - ((1 - lambda) y_v1 + lambda y_v2) with "
                "lambda = (c_v0 - c_v1)/(c_v2 - c_v1) from the sealed "
                "confirmation routing census, stays within 4.0 pooled-SE "
                "units (the finite-MC envelope of the sealed R32 "
                "named-residual classification)"
            ),
            "offline_envelope_se": OFFLINE_ENVELOPE_SE,
            "placement_explained": "P1 and P2 both hold",
            "informational_records": (
                "the implied route contrast (spread per unit of census "
                "shift) and the mass-blind route-contrast envelope are "
                "recorded per bin; the mass-blind envelope flags are "
                "evidence for the card-2 referral, never a verdict "
                "criterion of this card"
            ),
            "attribution_test_note": (
                "the attribution test is the monotonicity of the bin delta "
                "with the ramp start (the decision card diagnosis 4 "
                "mechanism: the more a variant routes sources to the "
                "continuum family, the more the bins move)"
            ),
        },
        "width_sensitivity": {
            "width_primes_gev": list(WIDTH_PRIMES_GEV),
            "method": (
                "analytic reweighting of the sealed per-bin payloads at "
                "the same ramp starts (zero draws): the per-variant local "
                "catalog-census slope s_v = dc/dB from the sealed "
                "ramp-start scan (adjacent-pair finite differences), the "
                "exact width identity dc/dW = <q>_band m_band / W^2 "
                "linearized with the pre-registered neutral in-band "
                "position alpha = 0.5, the reweighted census "
                "c_v' = c_v + (W' - W) alpha s_v, the per-bin route "
                "contrast m_b = (y_v2 - y_v1)/(c_v2 - c_v1), the "
                "reweighted values y_v' = y_v + m_b (c_v' - c_v), and the "
                "reweighted bin-spread p95 over the finite reweighted "
                "sigmas (the sealed R32 percentile convention)"
            ),
            "alpha_neutral": ALPHA_NEUTRAL,
            "alpha_bracket": list(ALPHA_BRACKET),
            "alpha_note": (
                "alpha = <q>_band / W is the in-band mean mass offset in "
                "ramp-width units; the neutral choice 0.5 is the band "
                "midpoint and the (0, 1) bracket is carried as the "
                "envelope"
            ),
            "contract": (
                "W = 0.5 GeV is adequate if the reweighted bin-spread p95 "
                "stays within 20% of the production p95 "
                "2.9604990756942544 for both W'; a shift beyond 20% is a "
                "width finding (verdict criterion (b))"
            ),
            "contract_relative": WIDTH_CONTRACT_REL,
            "production_p95_anchor": PRODUCTION_P95,
            "confirmation_p95_cross_check": CONFIRMATION_P95,
        },
        "new_spread_bounds": {
            "bin_spread_p95_bound": BIN_SPREAD_BOUND,
            "bin_spread_unit": (
                "the pooled MC standard error of the comparison "
                "sqrt(SE_0^2 + SE_1^2 + SE_2^2) (the sealed R32 unit)"
            ),
            "feeddown_spread_max_bound": FEEDDOWN_SPREAD_BOUND,
            "note": (
                "any new spread diagnostic produced by this audit carries "
                "the R32 pre-registered bounds (decision card lines "
                "106-109); the bounds are never relaxed and a bound "
                "failure is a recorded finding, never a refit"
            ),
        },
        "verdict_contract": {
            "pass": (
                "the persisting spread is explained by the ramp placement "
                "(the monotone attribution on the non-strange bins) and "
                "the strange-sector concentration (18 K+ of 32) is "
                "referred to card 2 (the originating families audit) as "
                "the responsible surface"
            ),
            "recommendation": (
                "RECOMMENDATION (separate rework card) with named findings "
                "if (a) any non-strange persisting bin is not "
                "placement-explained, (b) the width sensitivity exceeds "
                "the 20% contract, or (c) the census recomputation "
                "disagrees with the sealed routing census by more than "
                "1e-6 (a driver defect, rework the driver, not the "
                "surface)"
            ),
            "criteria": ["a", "b", "c"],
            "strange_referral_card": STRANGE_REFERRAL_CARD,
            "rework_precedent": (
                "the Round 3 cards 8a/8b precedent: a RECOMMENDATION "
                "spawns a separate rework card; this audit edits nothing "
                "pinned"
            ),
        },
        "zero_draw": {
            "draws": 0,
            "no_fitting": True,
            "optimizer_invoked": False,
            "method": (
                "the audit is an analytic re-evaluation of the sealed "
                "archives: no event generation, no sampling, no optimizer; "
                "the driver imports no random module and calls no numpy "
                "random function"
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
                    "the reserved confirmation block is pre-registered for "
                    "the exceptional case and must not fire on the first "
                    "pass"
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
    if substrate["card6_surface_sha256"] != CARD6_SHA256:
        _refuse("the card-6 surface is not byte-stable at the pinned digest")
    seals = [
        verify_sealed_dir(rel)
        for rel in (R32_PROD_DIR, R32_CONF_DIR, R30_DIR, R31_DIR,
                    R3_SE_AUDIT_DIR)
    ]
    verify_sealed_inputs()
    pre = build_pre_registration(substrate, seals)
    _write_json(run_dir / PREREGISTRATION_NAME, pre)
    driver_path = Path(__file__).resolve()
    provenance = {
        "schema": "na49_mixing.audit_r4_routing_formulation.provenance/1",
        "run": RUN_ID,
        "card": CARD,
        "card_chain": [
            "TASK-20260920-r3-gate3-decision",
            "TASK-20260920-round4-index",
            CARD,
        ],
        "written_utc": _now_utc(),
        "driver": "helpers/benchmarking/audit_r4_routing_formulation.py",
        "driver_sha256": {
            "nested": _sha256_file(
                NESTED_ROOT / "helpers" / "benchmarking"
                / "audit_r4_routing_formulation.py"
            ),
            "outer": _sha256_file(driver_path),
            "note": (
                "the dual-tree driver is byte-identical in both trees; the "
                "outer entry is the on-disk mirror's digest"
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
        f"audit_r4_routing_formulation: pre-registration written to "
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
    if pre.get("zero_draw", {}).get("reserved_audit_block", {}).get(
            "fired") is not False:
        _refuse(
            "the reserved audit block must not fire on the first pass"
        )
    return pre


# ---------------------------------------------------------------------------
# Mode: --census (the routing census recomputed from the sealed payloads)
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
        fields[key] = abs(
            float(recomputed[key]) - float(sealed[key])
        )
    for key in ("n_catalog", "n_continuum", "n_sources"):
        fields[key] = abs(
            float(recomputed[key]) - float(sealed[key])
        )
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


def run_census(run_dir: Path) -> dict:
    """Recompute the per-variant routing census from the sealed
    per-replica payloads and compare with the sealed routing census."""
    run_dir = Path(run_dir)
    assert_pre_registration(run_dir)
    census_path = run_dir / CENSUS_NAME

    prod_payloads = load_payload_census(
        R32_PROD_DIR, "variant_{v}_replica_{r}.json", 8,
    )
    conf_payloads = load_payload_census(
        R32_CONF_DIR, "confirmation_variant_{v}_replica_{r}.json", 16,
    )
    prod_report = _read_json(MONO_ROOT / R32_PROD_DIR / SPREAD_REPORT_NAME)
    gate_record = _read_json(MONO_ROOT / R32_PROD_DIR / GATE_RECORD_NAME)
    conf_verification = _read_json(
        MONO_ROOT / R32_CONF_DIR / CONFIRMATION_VERIFICATION_NAME
    )

    comparisons = []
    max_disagreement = 0.0
    for v in ("0", "1", "2"):
        sealed_prod = prod_report["routing_census"][v]
        fields = census_disagreement(prod_payloads[v], sealed_prod)
        gate_fraction = float(
            gate_record["clauses"]["no_boundary_dominance"]
            ["routing_census"][f"v{v}"]
        )
        fields["gate_fraction_catalog"] = abs(
            prod_payloads[v]["fraction_catalog"] - gate_fraction
        )
        sealed_conf = conf_verification["routing_census"][v]
        fields_conf = census_disagreement(conf_payloads[v], sealed_conf)
        worst = max(
            max(fields.values()), max(fields_conf.values())
        )
        max_disagreement = max(max_disagreement, worst)
        comparisons.append({
            "variant": int(v),
            "production": {
                "recomputed": prod_payloads[v],
                "sealed": sealed_prod,
                "disagreements": fields,
            },
            "confirmation": {
                "recomputed": conf_payloads[v],
                "sealed": sealed_conf,
                "disagreements": fields_conf,
            },
            "worst_disagreement": worst,
        })

    payload = {
        "schema": "na49_mixing.audit_r4_routing_formulation.census/1",
        "run": RUN_ID,
        "card": CARD,
        "probe": "the routing census recomputed from the sealed payloads",
        "sources": {
            "production_dir": R32_PROD_DIR,
            "production_replicas_per_variant": 8,
            "confirmation_dir": R32_CONF_DIR,
            "confirmation_replicas_per_variant": 16,
        },
        "comparison_set": [
            f"{R32_PROD_DIR}/{SPREAD_REPORT_NAME} routing_census",
            f"{R32_PROD_DIR}/{GATE_RECORD_NAME} "
            "clauses.no_boundary_dominance.routing_census",
            f"{R32_CONF_DIR}/{CONFIRMATION_VERIFICATION_NAME} "
            "routing_census",
        ],
        "disagreement_bound": CENSUS_DISAGREEMENT_BOUND,
        "max_disagreement": max_disagreement,
        "within_bound": bool(max_disagreement <= CENSUS_DISAGREEMENT_BOUND),
        "variants": comparisons,
        "no_fitting": True,
        "optimizer_invoked": False,
        "draws": 0,
    }
    _write_json(census_path, payload)
    print(
        f"audit_r4_routing_formulation: census recomputed "
        f"(max disagreement {max_disagreement:.3e} against the bound "
        f"{CENSUS_DISAGREEMENT_BOUND:.0e})",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# Mode: --attribution (the placement attribution of the persisting set)
# ---------------------------------------------------------------------------


def median_variant_index(values: list) -> int:
    """The index of the median variant value (ties broken by the lowest
    index; deterministic; the sealed register convention)."""
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    return int(order[(len(order) - 1) // 2])


def is_monotone_in_ramp_start(deltas_ramp_ordered: list) -> bool:
    """(P1): the delta sequence ordered by ramp start (v1, v0, v2) is
    monotone (non-decreasing or non-increasing)."""
    a, b, c = (float(x) for x in deltas_ramp_ordered)
    return (a <= b <= c) or (a >= b >= c)


def offline_residual_se(values: list, unit_se: float, lam: float) -> float:
    """(P2): the nominal variant's off-line residual from the census
    interpolation of the early and late variants, in pooled-SE units."""
    line = (1.0 - lam) * float(values[1]) + lam * float(values[2])
    return (float(values[0]) - line) / float(unit_se)


def attribution_rows(persisting: list, census_catalog: dict,
                     parent_rows: dict) -> list:
    """The per-bin attribution rows of the 32 persisting residuals
    (pure sealed-input arithmetic; zero draws)."""
    c0 = float(census_catalog["0"])
    c1 = float(census_catalog["1"])
    c2 = float(census_catalog["2"])
    if not (c1 < c0 < c2):
        _refuse(
            "the sealed routing census is not monotone in the ramp start; "
            "the placement attribution requires the census ordering"
        )
    lam = (c0 - c1) / (c2 - c1)
    delta_census = c2 - c1
    rows = []
    for entry in persisting:
        values = [float(x) for x in entry["confirmation_values"]]
        unit_se = float(entry["confirmation_unit_se"])
        if not (math.isfinite(unit_se) and unit_se > 0.0):
            _refuse(
                "a persisting register entry carries a non-finite pooled "
                f"SE ({entry.get('species')}, bin "
                f"{entry.get('bin_index')})"
            )
        median_i = median_variant_index(values)
        deltas = {
            v: (values[int(v)] - values[median_i]) / unit_se
            for v in ("0", "1", "2")
        }
        ramp_ordered = [deltas[v] for v in RAMP_START_ORDER]
        monotone = is_monotone_in_ramp_start(ramp_ordered)
        r_se = offline_residual_se(values, unit_se, lam)
        p2 = abs(r_se) <= (
            OFFLINE_ENVELOPE_SE + OFFLINE_ENVELOPE_TOLERANCE
        )
        spread = max(values) - min(values)
        sigma = spread / unit_se
        recorded_sigma = float(entry["confirmation_sigma"])
        if abs(sigma - recorded_sigma) > 1e-9 * max(1.0, recorded_sigma):
            _refuse(
                "the persisting register entry sigma disagrees with the "
                f"sealed values ({entry.get('species')}, bin "
                f"{entry.get('bin_index')})"
            )
        implied_contrast = spread / delta_census
        envelope_value = delta_census * max(
            values[1] / (1.0 - c1), values[2] / c2,
        )
        envelope_sigma = envelope_value / unit_se
        parent = parent_rows.get(
            (str(entry.get("species")), int(entry.get("bin_index")))
        )
        parent_block = None
        if parent is not None:
            p_values = [float(x) for x in parent["values"]]
            p_median = median_variant_index(p_values)
            parent_block = {
                "sigma": float(parent["sigma"]),
                "values": p_values,
                "unit_se": float(parent["unit_se"]),
                "deltas_vs_median_se_units": {
                    v: (p_values[int(v)] - p_values[p_median])
                    / float(parent["unit_se"])
                    for v in ("0", "1", "2")
                },
            }
        if monotone:
            if ramp_ordered[0] < ramp_ordered[2]:
                direction = "increasing_with_ramp_start"
                route_dominance = "catalog_dominant"
            else:
                direction = "decreasing_with_ramp_start"
                route_dominance = "continuum_dominant"
        else:
            direction = "non_monotone"
            route_dominance = None
        rows.append({
            "species": str(entry.get("species")),
            "bin_index": int(entry.get("bin_index")),
            "strange_species": str(entry.get("species")) == "k_plus",
            "parent_index": int(entry.get("parent_index")),
            "parent_sigma": float(entry.get("parent_sigma")),
            "confirmation_values": values,
            "confirmation_unit_se": unit_se,
            "confirmation_sigma": recorded_sigma,
            "sigma_recomputed": sigma,
            "median_variant": median_i,
            "deltas_vs_median_se_units": deltas,
            "deltas_ramp_start_order": ramp_ordered,
            "p1_monotone": bool(monotone),
            "delta_direction": direction,
            "route_dominance": route_dominance,
            "census_lambda": lam,
            "offline_residual_se": r_se,
            "p2_within_envelope": bool(p2),
            "offline_envelope_se": OFFLINE_ENVELOPE_SE,
            "placement_explained": bool(monotone and p2),
            "implied_route_contrast": implied_contrast,
            "mass_blind_envelope_sigma": envelope_sigma,
            "mass_blind_envelope_respected": bool(
                sigma <= envelope_sigma
            ),
            "parent_production": parent_block,
        })
    return rows


def species_summary(rows: list, species: str) -> dict:
    picked = [r for r in rows if r["species"] == species]
    return {
        "species": species,
        "bins": len(picked),
        "monotone": sum(1 for r in picked if r["p1_monotone"]),
        "within_census_envelope": sum(
            1 for r in picked if r["p2_within_envelope"]
        ),
        "placement_explained": sum(
            1 for r in picked if r["placement_explained"]
        ),
        "continuum_dominant": sum(
            1 for r in picked if r["route_dominance"] == "continuum_dominant"
        ),
        "catalog_dominant": sum(
            1 for r in picked if r["route_dominance"] == "catalog_dominant"
        ),
        "non_monotone": sum(1 for r in picked if not r["p1_monotone"]),
        "mass_blind_envelope_violations": sum(
            1 for r in picked if not r["mass_blind_envelope_respected"]
        ),
        "sigma_min": min((r["confirmation_sigma"] for r in picked),
                         default=0.0),
        "sigma_max": max((r["confirmation_sigma"] for r in picked),
                         default=0.0),
    }


def run_attribution(run_dir: Path) -> dict:
    """The per-variant delta pattern of the 32 persisting residuals,
    the monotone test, and the K+ referral."""
    run_dir = Path(run_dir)
    assert_pre_registration(run_dir)
    census_path = run_dir / CENSUS_NAME
    if not census_path.is_file():
        _refuse(
            f"{census_path} is missing; the attribution requires the "
            "census step (run --census first)"
        )
    census = _read_json(census_path)
    if not census.get("within_bound"):
        _refuse(
            "the census recomputation disagrees with the sealed routing "
            "census beyond the pre-registered bound; the attribution "
            "refuses to run on a driver defect (verdict criterion (c))"
        )
    conf_verification = _read_json(
        MONO_ROOT / R32_CONF_DIR / CONFIRMATION_VERIFICATION_NAME
    )
    prod_report = _read_json(MONO_ROOT / R32_PROD_DIR / SPREAD_REPORT_NAME)
    parent_rows = {
        (str(row["species"]), int(row["bin_index"])): row
        for row in prod_report["bin_checks"]
    }
    persisting = [
        entry for entry in conf_verification["residual_reclassification"]
        if entry.get("classification") == "persisting"
    ]
    if len(persisting) != PERSISTING_TOTAL:
        _refuse(
            f"the sealed confirmation register carries {len(persisting)} "
            f"persisting entries; the pre-registered probe set is "
            f"{PERSISTING_TOTAL}"
        )
    species_counts: dict[str, int] = {}
    for entry in persisting:
        species_counts[str(entry["species"])] = species_counts.get(
            str(entry["species"]), 0
        ) + 1
    if species_counts != PERSISTING_SPECIES_SPLIT:
        _refuse(
            "the persisting species split "
            f"{species_counts} != the pre-registered "
            f"{PERSISTING_SPECIES_SPLIT}"
        )
    sigmas = [float(e["confirmation_sigma"]) for e in persisting]
    if (min(sigmas) < PERSISTING_SIGMA_MIN - PERSISTING_SIGMA_TOLERANCE
            or max(sigmas) > PERSISTING_SIGMA_MAX + PERSISTING_SIGMA_TOLERANCE):
        _refuse(
            "the persisting confirmation sigmas leave the pre-registered "
            f"range [{PERSISTING_SIGMA_MIN}, {PERSISTING_SIGMA_MAX}] "
            f"(rounded quotes; tolerance {PERSISTING_SIGMA_TOLERANCE})"
        )
    census_catalog = {
        v: conf_verification["routing_census"][v]["fraction_catalog"]
        for v in ("0", "1", "2")
    }
    rows = attribution_rows(persisting, census_catalog, parent_rows)
    summaries = [
        species_summary(rows, species) for species in
        ("k_plus", "proton", "pi_plus", "pi_minus")
    ]
    non_strange_rows = [
        r for r in rows if r["species"] in PERSISTING_NON_STRANGE
    ]
    findings = [dict(r) for r in non_strange_rows if not r[
        "placement_explained"]]
    kplus_rows = [r for r in rows if r["species"] == "k_plus"]
    payload = {
        "schema": "na49_mixing.audit_r4_routing_formulation.attribution/1",
        "run": RUN_ID,
        "card": CARD,
        "probe": (
            "the placement attribution of the 32 persisting residuals "
            "(the per-variant delta pattern, the monotone test, the "
            "census-magnitude envelope)"
        ),
        "probe_set": {
            "source": (
                f"{R32_CONF_DIR}/{CONFIRMATION_VERIFICATION_NAME} "
                "residual_reclassification (classification == 'persisting')"
            ),
            "entries": len(rows),
            "species_split": species_counts,
            "sigma_range": [min(sigmas), max(sigmas)],
        },
        "census_fractions_used": census_catalog,
        "census_source": (
            "the sealed confirmation routing census (the same statistics "
            "as the confirmation values)"
        ),
        "predicate": {
            "p1": "monotone delta in the ramp start (v1 < v0 < v2)",
            "p2": (
                "the nominal variant's census-line residual within 4.0 "
                "pooled-SE units"
            ),
            "placement_explained": "P1 and P2",
            "note": (
                "the deltas are recorded against the median-variant "
                "reference; the reference cancels in the monotonicity test"
            ),
        },
        "rows": rows,
        "species_summaries": summaries,
        "non_strange": {
            "species": list(PERSISTING_NON_STRANGE),
            "bins": len(non_strange_rows),
            "placement_explained": sum(
                1 for r in non_strange_rows if r["placement_explained"]
            ),
            "findings": findings,
        },
        "k_plus_referral": {
            "referred_to_card": STRANGE_REFERRAL_CARD,
            "bins": len(kplus_rows),
            "monotone": sum(1 for r in kplus_rows if r["p1_monotone"]),
            "within_census_envelope": sum(
                1 for r in kplus_rows if r["p2_within_envelope"]
            ),
            "mass_blind_envelope_violations": sum(
                1 for r in kplus_rows
                if not r["mass_blind_envelope_respected"]
            ),
            "continuum_dominant": sum(
                1 for r in kplus_rows
                if r["route_dominance"] == "continuum_dominant"
            ),
            "note": (
                "the strange-sector concentration (18 K+ of 32) is "
                "referred to card 2 (the originating families audit) as "
                "the responsible surface; the referral is independent of "
                "this card's verdict and the two audits are independent "
                "inputs to the run cards"
            ),
        },
        "no_fitting": True,
        "optimizer_invoked": False,
        "draws": 0,
    }
    _write_json(run_dir / ATTRIBUTION_NAME, payload)
    print(
        f"audit_r4_routing_formulation: attribution written "
        f"({len(rows)} persisting bins; monotone "
        f"{sum(1 for r in rows if r['p1_monotone'])}/{len(rows)}; "
        f"non-strange placement-explained "
        f"{payload['non_strange']['placement_explained']}/"
        f"{len(non_strange_rows)}; findings {len(findings)})",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# Mode: --reweight (the analytic width reweighting, W' in {0.3, 0.7})
# ---------------------------------------------------------------------------


def census_slopes(census_catalog: dict, offset_step: float = 0.2) -> dict:
    """The per-variant local catalog-census slope s_v = dc/dB from the
    sealed ramp-start scan (adjacent-pair finite differences; the
    central variant takes the full-scan slope)."""
    c0 = float(census_catalog["0"])
    c1 = float(census_catalog["1"])
    c2 = float(census_catalog["2"])
    return {
        "1": (c0 - c1) / offset_step,
        "0": (c2 - c1) / (2.0 * offset_step),
        "2": (c2 - c0) / offset_step,
    }


def reweighted_census(census_catalog: dict, slopes: dict, width_prime: float,
                      alpha: float = ALPHA_NEUTRAL) -> dict:
    """The reweighted census ``c_v' = c_v + (W' - W) alpha s_v`` (the
    linearized exact width identity at the fixed ramp starts)."""
    shift = (float(width_prime) - RAMP_WIDTH_GEV) * float(alpha)
    return {
        v: float(census_catalog[v]) + shift * float(slopes[v])
        for v in ("0", "1", "2")
    }


def reweight_bin_row(values: list, unit_se: float, slopes: dict,
                     census_catalog: dict, census_prime: dict) -> dict:
    """One bin's analytic reweighting: the route contrast
    ``m_b = (y_v2 - y_v1)/(c_v2 - c_v1)``, the reweighted values
    ``y_v' = y_v + m_b (c_v' - c_v)``, the reweighted spread and its
    pooled-SE sigma.  The pooled SE is unchanged (the reweighting moves
    the mixture weights, not the Monte Carlo statistics)."""
    c1 = float(census_catalog["1"])
    c2 = float(census_catalog["2"])
    m_b = (float(values[2]) - float(values[1])) / (c2 - c1)
    reweighted = {
        v: float(values[int(v)]) + m_b * (
            float(census_prime[v]) - float(census_catalog[v])
        )
        for v in ("0", "1", "2")
    }
    spread_prime = max(reweighted.values()) - min(reweighted.values())
    sigma_prime = spread_prime / float(unit_se)
    return {
        "route_contrast": m_b,
        "reweighted_values": [reweighted["0"], reweighted["1"],
                              reweighted["2"]],
        "reweighted_spread": spread_prime,
        "reweighted_sigma": sigma_prime,
    }


def percentile95(sorted_or_not: list) -> float:
    """The sealed R32 p95 convention (numpy linear interpolation)."""
    if not sorted_or_not:
        return 0.0
    return float(np.percentile(np.asarray(sorted_or_not, dtype=float), 95.0))


def reweight_statistics(report: dict, slopes: dict, census_catalog: dict,
                        width_prime: float,
                        alpha: float = ALPHA_NEUTRAL) -> dict:
    """The full per-bin analytic reweighting of one sealed spread
    report at one hypothetical width and in-band position parameter
    (zero draws)."""
    census_prime = reweighted_census(census_catalog, slopes, width_prime,
                                     alpha)
    delta_census = float(census_catalog["2"]) - float(census_catalog["1"])
    delta_census_prime = census_prime["2"] - census_prime["1"]
    ratio = delta_census_prime / delta_census
    finite = []
    per_species: dict[str, list] = {}
    n_reweighted = 0
    n_vacuous = 0
    for row in report["bin_checks"]:
        status = row.get("status")
        if status not in ("pass", "fail"):
            n_vacuous += 1
            continue
        unit_se = float(row["unit_se"])
        if not (math.isfinite(unit_se) and unit_se > 0.0):
            n_vacuous += 1
            continue
        block = reweight_bin_row(
            [float(x) for x in row["values"]], unit_se, slopes,
            census_catalog, census_prime,
        )
        finite.append(block["reweighted_sigma"])
        per_species.setdefault(str(row["species"]), []).append(
            block["reweighted_sigma"]
        )
        n_reweighted += 1
    p95_prime = percentile95(finite)
    return {
        "width_prime_gev": float(width_prime),
        "census_prime": census_prime,
        "census_spread_prime": delta_census_prime,
        "census_spread": delta_census,
        "census_spread_ratio": ratio,
        "bins_reweighted": n_reweighted,
        "bins_vacuous_unchanged": n_vacuous,
        "reweighted_bin_spread_p95": p95_prime,
        "reweighted_bin_spread_p95_per_species": {
            species: percentile95(sig) for species, sig in
            sorted(per_species.items())
        },
        "uniform_ratio_cross_check_p95": (
            float(report["summary"]["bin_spread_p95"]) * ratio
        ),
    }


def run_reweight(run_dir: Path) -> dict:
    """The W' in {0.3, 0.7 GeV} analytic reweighting and the 20%
    contract."""
    run_dir = Path(run_dir)
    pre = assert_pre_registration(run_dir)
    for name in (CENSUS_NAME, ATTRIBUTION_NAME):
        if not (run_dir / name).is_file():
            _refuse(
                f"{run_dir / name} is missing; the reweighting requires "
                "the census and attribution steps (run --census and "
                "--attribution first)"
            )
    census = _read_json(run_dir / CENSUS_NAME)
    if not census.get("within_bound"):
        _refuse(
            "the census recomputation disagrees beyond the bound; the "
            "reweighting refuses to run on a driver defect"
        )
    prod_report = _read_json(MONO_ROOT / R32_PROD_DIR / SPREAD_REPORT_NAME)
    conf_verification = _read_json(
        MONO_ROOT / R32_CONF_DIR / CONFIRMATION_VERIFICATION_NAME
    )
    prod_p95 = float(prod_report["summary"]["bin_spread_p95"])
    if abs(prod_p95 - PRODUCTION_P95) > 1e-12:
        _refuse(
            f"the sealed production p95 {prod_p95!r} != the "
            f"pre-registered anchor {PRODUCTION_P95!r}"
        )
    conf_p95 = float(conf_verification["summary"]["bin_spread_p95"])
    if abs(conf_p95 - CONFIRMATION_P95) > 1e-12:
        _refuse(
            f"the sealed confirmation p95 {conf_p95!r} != the "
            f"pre-registered anchor {CONFIRMATION_P95!r}"
        )
    prod_census = {
        v: prod_report["routing_census"][v]["fraction_catalog"]
        for v in ("0", "1", "2")
    }
    conf_census = {
        v: conf_verification["routing_census"][v]["fraction_catalog"]
        for v in ("0", "1", "2")
    }
    pre_census = pre["width_sensitivity"]
    if abs(float(pre_census["production_p95_anchor"]) - PRODUCTION_P95) > 1e-12:
        _refuse("the pre-registration p95 anchor drifted")

    prod_slopes = census_slopes(prod_census)
    conf_slopes = census_slopes(conf_census)
    variants = []
    contract_results = {}
    for width_prime in WIDTH_PRIMES_GEV:
        block = reweight_statistics(
            prod_report, prod_slopes, prod_census, width_prime,
        )
        deviation = abs(
            block["reweighted_bin_spread_p95"] - PRODUCTION_P95
        )
        relative = deviation / PRODUCTION_P95
        within = bool(relative <= WIDTH_CONTRACT_REL)
        # The alpha envelope: the same per-bin statistic evaluated at
        # the bracket endpoints of the in-band position parameter.
        bracket = {}
        for alpha in ALPHA_BRACKET:
            alpha_block = reweight_statistics(
                prod_report, prod_slopes, prod_census, width_prime, alpha,
            )
            bracket[str(alpha)] = {
                "census_spread_ratio": alpha_block["census_spread_ratio"],
                "reweighted_bin_spread_p95":
                    alpha_block["reweighted_bin_spread_p95"],
                "relative_deviation": abs(
                    alpha_block["reweighted_bin_spread_p95"]
                    - PRODUCTION_P95
                ) / PRODUCTION_P95,
            }
        bracket_within = all(
            b["relative_deviation"] <= WIDTH_CONTRACT_REL
            for b in bracket.values()
        )
        block["contract"] = {
            "anchor_p95": PRODUCTION_P95,
            "absolute_deviation": deviation,
            "relative_deviation": relative,
            "contract_relative": WIDTH_CONTRACT_REL,
            "within_contract": within,
            "alpha_bracket": bracket,
            "alpha_bracket_within_contract": bool(bracket_within),
        }
        contract_results[str(width_prime)] = within and bracket_within
        variants.append(block)
    # The confirmation-statistics cross-check (the same algebra on the
    # sealed confirmation report; informational).
    conf_cross_check = []
    for width_prime in WIDTH_PRIMES_GEV:
        block = reweight_statistics(
            conf_verification, conf_slopes, conf_census, width_prime,
        )
        block["anchor_p95"] = CONFIRMATION_P95
        block["relative_deviation"] = abs(
            block["reweighted_bin_spread_p95"] - CONFIRMATION_P95
        ) / CONFIRMATION_P95
        conf_cross_check.append(block)
    width_adequate = all(contract_results.values())
    payload = {
        "schema": "na49_mixing.audit_r4_routing_formulation.reweight/1",
        "run": RUN_ID,
        "card": CARD,
        "probe": (
            "the analytic width reweighting of the sealed per-bin "
            "payloads at W' in {0.3, 0.7 GeV} (zero draws)"
        ),
        "algebra": {
            "census_slopes_production": prod_slopes,
            "census_slopes_confirmation": conf_slopes,
            "alpha_neutral": ALPHA_NEUTRAL,
            "alpha_bracket": list(ALPHA_BRACKET),
            "note": (
                "s_v = dc/dB from the sealed ramp-start scan; "
                "c_v' = c_v + (W' - W) alpha s_v; the per-bin route "
                "contrast m_b = (y_v2 - y_v1)/(c_v2 - c_v1); "
                "y_v' = y_v + m_b (c_v' - c_v); the pooled SE is "
                "unchanged"
            ),
        },
        "production": {
            "anchor_p95": PRODUCTION_P95,
            "census_fractions": prod_census,
            "width_variants": variants,
        },
        "confirmation_cross_check": {
            "anchor_p95": CONFIRMATION_P95,
            "census_fractions": conf_census,
            "width_variants": conf_cross_check,
            "note": (
                "the same algebra on the sealed confirmation statistics; "
                "informational, the contract anchor is the production p95"
            ),
        },
        "contract": {
            "relative_bound": WIDTH_CONTRACT_REL,
            "per_width_prime": contract_results,
            "width_adequate": bool(width_adequate),
            "note": (
                "W = 0.5 GeV is adequate if the reweighted bin-spread p95 "
                "stays within 20% of the production p95 for both W'; a "
                "shift beyond 20% is a width finding (verdict criterion "
                "(b))"
            ),
        },
        "new_spread_bounds": {
            "bin_spread_p95_bound": BIN_SPREAD_BOUND,
            "note": (
                "the reweighted p95 is a new spread diagnostic and carries "
                "the R32 pre-registered bound 2.0 pooled-SE units beside "
                "the value; the bound is never relaxed"
            ),
        },
        "no_fitting": True,
        "optimizer_invoked": False,
        "draws": 0,
    }
    _write_json(run_dir / REWEIGHT_NAME, payload)
    p95_texts = ", ".join(
        f"W'={v['width_prime_gev']}: p95 {v['reweighted_bin_spread_p95']:.6f}"
        for v in variants
    )
    print(
        f"audit_r4_routing_formulation: reweighting written "
        f"({p95_texts}; width adequate {width_adequate})",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# Mode: --verdict
# ---------------------------------------------------------------------------


def evaluate_verdict_criteria(attribution: dict, reweight: dict,
                              census: dict) -> dict:
    """The pre-registered verdict criteria (a), (b), (c) evaluated on
    the three measurement artifacts (a pure function; zero draws)."""
    criterion_a = {
        "criterion": "a",
        "statement": (
            "any non-strange persisting bin is not placement-explained"
        ),
        "non_strange_bins": attribution["non_strange"]["bins"],
        "placement_explained": attribution["non_strange"][
            "placement_explained"
        ],
        "findings": len(attribution["non_strange"]["findings"]),
        "fired": bool(attribution["non_strange"]["findings"]),
    }
    criterion_b = {
        "criterion": "b",
        "statement": "the width sensitivity exceeds the 20% contract",
        "width_adequate": reweight["contract"]["width_adequate"],
        "per_width_prime": reweight["contract"]["per_width_prime"],
        "fired": not bool(reweight["contract"]["width_adequate"]),
    }
    criterion_c = {
        "criterion": "c",
        "statement": (
            "the census recomputation disagrees with the sealed routing "
            "census by more than 1e-6 (a driver defect, rework the "
            "driver, not the surface)"
        ),
        "max_disagreement": census["max_disagreement"],
        "bound": census["disagreement_bound"],
        "fired": not bool(census["within_bound"]),
    }
    fired = [c for c in (criterion_a, criterion_b, criterion_c)
             if c["fired"]]
    return {
        "criteria": {"a": criterion_a, "b": criterion_b, "c": criterion_c},
        "fired": fired,
        "verdict": "RECOMMENDATION" if fired else "PASS",
    }


def _finding_narrative(row: dict) -> dict:
    """The named-finding narrative of one non-strange persisting bin
    (bin/species, observed, diagnosis, the responsible card, the
    proposed rework)."""
    deltas = row["deltas_ramp_start_order"]
    if deltas[0] <= deltas[1] and deltas[1] > deltas[2]:
        shape = "rises to the nominal variant and falls to the late variant"
    elif deltas[0] >= deltas[1] and deltas[1] < deltas[2]:
        shape = "falls to the nominal variant and rises to the late variant"
    else:
        shape = "is non-monotone across the three variants"
    parent_sigma = (
        row["parent_production"]["sigma"]
        if row.get("parent_production") else None
    )
    parent_text = (
        f" (parent sigma {parent_sigma:.3f} at production statistics)"
        if parent_sigma is not None else ""
    )
    observed = (
        f"confirmation deltas in ramp-start order (v1, v0, v2) = "
        f"[{deltas[0]:+.3f}, {deltas[1]:+.3f}, {deltas[2]:+.3f}] "
        f"pooled-SE units at sigma {row['confirmation_sigma']:.3f} "
        f"against the pre-registered bound 2.0{parent_text}; the "
        f"ramp-start sequence {shape}; the nominal variant's "
        f"census-line residual is {row['offline_residual_se']:+.3f} SE, "
        f"inside the 4.0-SE census-magnitude envelope"
    )
    diagnosis = (
        "one-sided routing response: the bin value responds to one ramp "
        "direction only, which the census-linked re-mixing (the placement "
        "mechanism) does not reproduce; the magnitude stays inside the "
        "census-shift envelope, so the finding is the monotonicity "
        "structure, not the size"
    )
    return {
        "kind": "bin",
        "species": row["species"],
        "bin_index": row["bin_index"],
        "observed": observed,
        "diagnosis": diagnosis,
        "responsible": (
            "the card-6 blend routing formulation "
            f"({CARD6_REL}, sha256 {CARD6_SHA256}, byte-unchanged by this "
            "audit)"
        ),
        "proposed_rework": (
            "a separate Round 4 rework card on the card-6 surface to "
            "re-derive the w(M) ramp formulation against the sealed "
            "three-variant payload structure (the ramp shape at partial "
            "weights and its interaction with the boundary-matching "
            "factor), with any new variant set pre-registered and the R32 "
            "bounds (2.0 / 3.0) carried; the rework card may "
            "alternatively confirm the formulation and re-classify the "
            "bin as standing model uncertainty"
        ),
        "evidence": {
            "confirmation_values": row["confirmation_values"],
            "confirmation_unit_se": row["confirmation_unit_se"],
            "confirmation_sigma": row["confirmation_sigma"],
            "deltas_vs_median_se_units": row["deltas_vs_median_se_units"],
            "p1_monotone": row["p1_monotone"],
            "p2_within_envelope": row["p2_within_envelope"],
            "offline_residual_se": row["offline_residual_se"],
            "parent_sigma_production": (
                row["parent_production"]["sigma"]
                if row.get("parent_production") else None
            ),
        },
    }


def run_verdict(run_dir: Path) -> dict:
    """Assemble the verdict artifact (PASS or RECOMMENDATION with named
    findings)."""
    run_dir = Path(run_dir)
    pre = assert_pre_registration(run_dir)
    artifacts = {}
    for name, key in (
        (CENSUS_NAME, "census"),
        (ATTRIBUTION_NAME, "attribution"),
        (REWEIGHT_NAME, "reweight"),
    ):
        path = run_dir / name
        if not path.is_file():
            _refuse(
                f"{path} is missing; the verdict requires the three "
                "measurement steps (run --census, --attribution, "
                "--reweight first)"
            )
        artifacts[key] = _read_json(path)
    evaluation = evaluate_verdict_criteria(
        artifacts["attribution"], artifacts["reweight"], artifacts["census"],
    )
    findings = []
    if evaluation["criteria"]["a"]["fired"]:
        findings.extend(
            _finding_narrative(row)
            for row in artifacts["attribution"]["non_strange"]["findings"]
        )
    if evaluation["criteria"]["b"]["fired"]:
        findings.append({
            "kind": "width",
            "species": None,
            "bin_index": None,
            "observed": (
                "the reweighted bin-spread p95 leaves the 20% contract "
                "band around the production p95 "
                f"{PRODUCTION_P95!r}"
            ),
            "diagnosis": (
                "the cross-variant spread is sensitive to the ramp width "
                "beyond the pre-registered contract; the width is a "
                "spread driver at the sealed placement"
            ),
            "responsible": (
                "the card-6 blend routing formulation "
                f"({CARD6_REL}, sha256 {CARD6_SHA256})"
            ),
            "proposed_rework": (
                "a separate Round 4 rework card on the card-6 surface to "
                "pre-register and evaluate the width variants as generated "
                "Monte Carlo (the reserved audit block), replacing the "
                "analytic reweighting"
            ),
            "evidence": {
                "per_width_prime":
                    artifacts["reweight"]["contract"]["per_width_prime"],
                "width_variants": [
                    {
                        "width_prime_gev": block["width_prime_gev"],
                        "reweighted_bin_spread_p95":
                            block["reweighted_bin_spread_p95"],
                        "relative_deviation":
                            block["contract"]["relative_deviation"],
                    }
                    for block in
                    artifacts["reweight"]["production"]["width_variants"]
                ],
            },
        })
    if evaluation["criteria"]["c"]["fired"]:
        findings.append({
            "kind": "census_driver_defect",
            "species": None,
            "bin_index": None,
            "observed": (
                "the recomputed routing census disagrees with the sealed "
                "census by "
                f"{artifacts['census']['max_disagreement']:.3e} beyond the "
                "1e-6 bound"
            ),
            "diagnosis": (
                "a driver defect; rework the driver, never the surface "
                "(the verdict contract criterion (c))"
            ),
            "responsible": (
                "the audit driver "
                "helpers/benchmarking/audit_r4_routing_formulation.py"
            ),
            "proposed_rework": (
                "a separate rework card to fix and re-run the audit "
                "driver; the card-6 surface is not implicated"
            ),
            "evidence": {
                "max_disagreement":
                    artifacts["census"]["max_disagreement"],
                "bound": artifacts["census"]["disagreement_bound"],
            },
        })
    verdict = evaluation["verdict"]
    attribution = artifacts["attribution"]
    reweight = artifacts["reweight"]
    recommendation_text = None
    if verdict == "RECOMMENDATION":
        parts = []
        if evaluation["criteria"]["a"]["fired"]:
            parts.append(
                "the ramp placement does not fully explain the persisting "
                "spread: "
                f"{len(attribution['non_strange']['findings'])} of "
                f"{attribution['non_strange']['bins']} non-strange "
                "persisting bins show non-monotone delta patterns in the "
                "ramp start (one-sided routing responses the census-linked "
                "re-mixing does not reproduce)"
            )
        if evaluation["criteria"]["b"]["fired"]:
            parts.append(
                "the width sensitivity exceeds the 20% contract"
            )
        if evaluation["criteria"]["c"]["fired"]:
            parts.append(
                "the census recomputation disagrees with the sealed "
                "routing census beyond 1e-6 (a driver defect)"
            )
        recommendation_text = (
            "Spawn a separate Round 4 rework card on the card-6 blend "
            "routing surface (na49_mixing/spectral_density.py, sha256 "
            f"{CARD6_SHA256}, byte-unchanged by this audit): "
            + "; ".join(parts)
            + ". The rework card pre-registers its own variant set and "
            "carries the R32 bounds (bin spread p95 < 2.0 pooled-SE "
            "units, feed-down spread max < 3.0 sigma); it may "
            "alternatively confirm the formulation as-is and re-classify "
            "the named bins as standing model uncertainty. The "
            "strange-sector concentration (18 K+ of 32 persisting bins) "
            "is referred to card 2 "
            f"({STRANGE_REFERRAL_CARD}) as the responsible surface; the "
            "two audits are independent inputs to the Round 4 run cards."
        )
    payload = {
        "schema": "na49_mixing.audit_r4_routing_formulation.verdict/1",
        "run": RUN_ID,
        "card": CARD,
        "verdict": verdict,
        "verdict_contract": pre["verdict_contract"],
        "probe_set": {
            "entries": attribution["probe_set"]["entries"],
            "species_split": attribution["probe_set"]["species_split"],
            "source": attribution["probe_set"]["source"],
        },
        "bounds": {
            "bin_spread_p95_bound": BIN_SPREAD_BOUND,
            "feeddown_spread_max_bound": FEEDDOWN_SPREAD_BOUND,
            "census_disagreement_bound": CENSUS_DISAGREEMENT_BOUND,
            "width_contract_relative": WIDTH_CONTRACT_REL,
            "offline_envelope_se": OFFLINE_ENVELOPE_SE,
            "note": (
                "the pre-registered bounds are cited verbatim from the "
                "card and the R32 pre-registration; none is relaxed"
            ),
        },
        "criteria": evaluation["criteria"],
        "findings": findings,
        "recommendation": recommendation_text,
        "attribution_summary": {
            "persisting_bins": attribution["probe_set"]["entries"],
            "monotone": sum(
                1 for r in attribution["rows"] if r["p1_monotone"]
            ),
            "within_census_envelope": sum(
                1 for r in attribution["rows"] if r["p2_within_envelope"]
            ),
            "placement_explained": sum(
                1 for r in attribution["rows"] if r["placement_explained"]
            ),
            "non_strange": attribution["non_strange"],
            "k_plus_referral": attribution["k_plus_referral"],
        },
        "width_summary": {
            "width_adequate": reweight["contract"]["width_adequate"],
            "per_width_prime": [
                {
                    "width_prime_gev": block["width_prime_gev"],
                    "reweighted_bin_spread_p95":
                        block["reweighted_bin_spread_p95"],
                    "census_spread_ratio": block["census_spread_ratio"],
                    "relative_deviation":
                        block["contract"]["relative_deviation"],
                    "within_contract":
                        block["contract"]["within_contract"],
                }
                for block in reweight["production"]["width_variants"]
            ],
            "carried_bound": BIN_SPREAD_BOUND,
            "note": (
                "the reweighted p95 rows carry the R32 pre-registered "
                "bound 2.0 pooled-SE units beside the value; the width "
                "criterion of this audit is the 20% contract"
            ),
        },
        "census_summary": {
            "max_disagreement": artifacts["census"]["max_disagreement"],
            "bound": artifacts["census"]["disagreement_bound"],
            "within_bound": artifacts["census"]["within_bound"],
        },
        "zero_draw": {
            "draws": 0,
            "no_fitting": True,
            "optimizer_invoked": False,
            "reserved_audit_block_fired": False,
        },
        "sealed_inputs": dict(SEALED_INPUT_DIGESTS),
        "surface": {
            "card6_path": CARD6_REL,
            "card6_sha256": CARD6_SHA256,
            "byte_unchanged_by_this_audit": True,
        },
    }
    _write_json(run_dir / VERDICT_NAME, payload)
    print(
        f"audit_r4_routing_formulation: verdict {verdict} "
        f"({len(findings)} named findings; criteria fired: "
        f"{[c['criterion'] for c in evaluation['fired']] or 'none'})",
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
        "schema": "na49_mixing.audit_r4_routing_formulation"
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
        CENSUS_NAME,
        ATTRIBUTION_NAME,
        REWEIGHT_NAME,
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
        f"audit_r4_routing_formulation: run dir sealed "
        f"({len(seal['files'])} files listed in {SEAL_NAME})",
        flush=True,
    )
    return seal


# ---------------------------------------------------------------------------
# Mode: --clean-validate
# ---------------------------------------------------------------------------


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
    artifact_names = (CENSUS_NAME, ATTRIBUTION_NAME, REWEIGHT_NAME,
                      VERDICT_NAME)
    before = {
        name: (run_dir / name).read_bytes() for name in artifact_names
    }
    driver_path = Path(__file__).resolve()
    commands = []
    probe_results = []
    for mode in ("census", "attribution", "reweight", "verdict"):
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
        "schema": "na49_mixing.audit_r4_routing_formulation"
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
            "audit_r4_routing_formulation: the clean-environment "
            "validation FAILED (the recomputed artifact bytes differ "
            "from the sealed artifacts)",
            flush=True,
        )
        raise SystemExit(1)
    print(
        "audit_r4_routing_formulation: clean-environment validation "
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
        prog="audit_r4_routing_formulation.py",
        description=(
            "Execute the Round 4 transition routing formulation audit "
            "(card TASK-20260920-r4-audit-transition-routing-formulation, "
            "Gate 3 follow-up obligation 1): resolve the K+-concentrated "
            "persisting spread of the sealed R32 evidence against the "
            "blend ramp placement and width on the byte-stable card-6 "
            "surface.  Zero draws; read-only on every pinned surface and "
            "sealed archive; PASS or RECOMMENDATION (a separate rework "
            "card), never an in-audit surface edit."
        ),
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--pre-registration", action="store_true",
        help="write pre_registration.json and provenance.json, then stop",
    )
    modes.add_argument(
        "--census", action="store_true",
        help="recompute the routing census from the sealed payloads, "
             "then stop",
    )
    modes.add_argument(
        "--attribution", action="store_true",
        help="the placement attribution of the 32 persisting residuals, "
             "then stop",
    )
    modes.add_argument(
        "--reweight", action="store_true",
        help="the analytic width reweighting (W' in {0.3, 0.7 GeV}) and "
             "the 20% contract, then stop",
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
    elif args.census:
        run_census(run_dir)
    elif args.attribution:
        run_attribution(run_dir)
    elif args.reweight:
        run_reweight(run_dir)
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
