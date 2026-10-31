#!/usr/bin/env python3
"""R30 driver: the spectral normalization check (Round 3, card 7).

Task ``TASK-20260917-r3-run-r30-spectral-normalization`` (plan section 9
run R30, line 638; the Round 3 card set, card 7; cards 4, 5, and 6 are
the prerequisites).  R30 is hermetic (round 3 index item 6): it draws no
random numbers, generates no events, and fits nothing.  It verifies the
cards 4-6 surfaces before any Monte Carlo run uses them:

- every spectral family of the committed
  ``config/v2_singlet/continuum_resonances.json`` -- the three activated
  families (baryonic ``(1, 1, 0)``, strange baryonic ``(1, 0, -1)``,
  neutral mesonic ``(0, 0, 0)``) and the four declared, inactive charged
  and strange mesonic families -- numerically integrates to 1 over its
  support on the fixed 10,001-node trapezoid (the Round 2 reference
  quadrature, the ``na49_mixing/resonance_widths.py`` convention),
  within the Round 2 spectral-normalization tolerance 1e-6;
- the threshold behavior of the family density: ``rho_a`` is exactly
  zero strictly below ``m_min``, and the one-sided derivatives agree to
  1e-12 at ``m_min`` and at ``m_min + ramp_width`` (C^1), verified on a
  fine grid with node spacing 1e-6 GeV;
- the card-6 blend: for each activated source QN set the integrals of
  ``rho_catalog``, ``rho_cont``, and ``rho_total`` on the union-support
  reference grid, with the double-counting identity
  ``int_{M<B}(rho_catalog) + int_{M>=B}(rho_total) = 1`` within 1e-6,
  and ``rho_total`` continuous at the blend boundary ``B`` and at
  ``B + W`` (the card-6 continuity definition).

No fitting of any kind (``no_fitting = true``): no optimizer is invoked
and no parameter of cards 4-6 is changed to make a check pass -- a
failing check stops the run and re-opens the originating card, never the
tolerance.

The driver implements the full run procedure and executes it in steps:
each invocation performs exactly one step and stops (the run card
dispatches the steps; pre-registration, integration, verification, gate
evidence, seal, and clean-environment validation land in this order, and
the run dir is immutable after the seal).

Modes (exactly one per invocation):

- ``--pre-registration``: write ``pre_registration.json`` and
  ``provenance.json`` into the run dir, then stop.  Nothing is
  evaluated: the pre-registration is written before any evaluation and
  asserted against the card (the families, the quadrature, the
  normalization tolerance, the threshold behavior spec, the
  double-counting identity, and the seeds block).
- ``--integrate``: the trapezoid integral of every family (activated
  and declared) over its support on the reference quadrature; for each
  activated QN set the integrals of ``rho_catalog``, ``rho_cont``, and
  ``rho_total`` and the double-counting identity.  Writes
  ``family_integrals.json``, then stop.  A failing check stops the run
  (exit 1) after the report is written.
- ``--verify``: the pre-registered threshold checks on the fine grid
  (node spacing 1e-6 GeV) around ``m_min``, ``B``, and ``B + W``.
  Writes ``threshold_checks.json``, then stop.
- ``--gate``: write ``gate3_r30.json`` (evidence only), then stop.
  Requires the integration and the threshold verification.
- ``--seal``: write ``seal_manifest.json`` and ``SHA256SUMS.txt`` over
  the run dir, then stop.  The run dir is immutable after the seal; the
  gate JSON is part of the seal.
- ``--clean-validate``: verify the seal, re-run the integration (step 2)
  from a clean interpreter over the committed config, and confirm the
  report hash matches; write ``clean_environment_commands.json``, then
  stop.
- ``--run-dir <path>``: override the run dir.

Pre-registration contract (``pre_registration.json``, written before any
evaluation, asserted against the card):

- **Families**: all seven records of the committed continuum config in
  config order, each with its quantum numbers, ``active`` flag, and
  density parameters; the three activated QN sets and the four declared
  inactive QN sets are recorded.
- **Quadrature**: the fixed 10,001-node trapezoid over each family's
  support ``[m_min, m_max]`` (the Round 2 reference quadrature;
  ``REFERENCE_GRID_POINTS`` of ``na49_mixing.spectral_density``, equal
  to ``SPECTRAL_QUADRATURE_NODES`` of ``na49_mixing.resonance_decay``,
  the ``na49_mixing/resonance_widths.py`` normalization convention).
- **Normalization tolerance**: 1e-6 (the Round 2 spectral-normalization
  tolerance) on the unit integral of every family and of ``rho_total``
  for the three activated source QN sets.
- **Threshold behavior**: ``rho_a == 0`` strictly below ``m_min``; C^1
  continuity (one-sided derivatives agree to 1e-12) at ``m_min`` and at
  ``m_min + ramp_width``; continuity of ``rho_total`` at the blend
  boundary ``B`` and at ``B + W`` (the card-6 definition: one-sided
  probes at +-1e-14 GeV agree to 1e-12).  The derivative estimator is
  the second-order three-point one-sided stencil on the fine grid (node
  spacing 1e-6 GeV); the observed two-sided ``rho_total`` jump at
  +-1e-6 GeV is recorded for the record (it is O(delta) and converges;
  the pass criterion is the card-6 probe pair).
- **Double-counting identity** (card 6):
  ``int_{M<B}(rho_catalog) + int_{M>=B}(rho_total) = 1`` within 1e-6
  for each activated QN set.
- **Seeds**: none (round 3 index item 6); the master seed ``20260917``
  is recorded with ``draws: 0`` for contract uniformity.  The driver
  consumes no uniform on any path: the family, blend, and report
  surfaces used here are pure compute (their caches are pure compute
  caches), and the mass-draw and decay-routing surfaces of the module
  are never entered.

``family_integrals.json``, ``threshold_checks.json``, and
``gate3_r30.json`` carry no wall-clock fields: their bytes are a pure
function of the committed inputs, which is what the clean-environment
validation re-hashes.

Dual-tree driver: byte-identical in the nested tree
(``dev/mixing2006/helpers/benchmarking/``) and the monorepo mirror
(``helpers/benchmarking/``); both locate the package with the
established ``_locate_roots`` pattern.

Hermetic: the committed continuum config, the committed resonance
catalog, and the ``na49_mixing`` package only; no external tables, no
event generation, no fitted values.
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
# monorepo mirror (helpers/benchmarking/).  Both must find the
# na49_mixing package and the committed configs.
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
        "staged_r30_spectral_normalization: " + message,
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(2)


NESTED_ROOT, MONO_ROOT = _locate_roots()
if str(NESTED_ROOT) not in sys.path:
    sys.path.insert(0, str(NESTED_ROOT))

from na49_mixing.hadron_catalog import HadronCatalog  # noqa: E402
from na49_mixing.resonance_decay import SPECTRAL_QUADRATURE_NODES  # noqa: E402
from na49_mixing.spectral_density import (  # noqa: E402
    BLEND_BOUNDARY,
    BLEND_RAMP_WIDTH,
    NORMALIZATION_TOLERANCE,
    REFERENCE_GRID_POINTS,
    SUPPORT_M_MAX,
    BlendSpec,
    blended_density,
    blended_normalization_report,
    family_density,
    family_normalization_report,
    families_from_json,
)

# ---------------------------------------------------------------------------
# Pre-registered run identity (card TASK-20260917-r3-run-r30-spectral-
# normalization; plan section 9 run R30 line 638, round 3 index items 2-4
# and 6; the tolerance and quadrature are the Round 2 conventions).
# ---------------------------------------------------------------------------

RUN_ID = "R30"
CARD = "TASK-20260917-r3-run-r30-spectral-normalization"
RUN_NAME = "r30_spectral_normalization_2026-09-17"
DEFAULT_RUN_DIR = "dev/00runs/r30_spectral_normalization_2026-09-17"
GATE_RECORD_NAME = "gate3_r30.json"
PREREGISTRATION_NAME = "pre_registration.json"
PROVENANCE_NAME = "provenance.json"
FAMILY_INTEGRALS_NAME = "family_integrals.json"
THRESHOLD_CHECKS_NAME = "threshold_checks.json"
CLEAN_ENV_NAME = "clean_environment_commands.json"
SHA256SUMS_NAME = "SHA256SUMS.txt"
SEAL_NAME = "seal_manifest.json"

CONTINUUM_CONFIG_REL = "config/v2_singlet/continuum_resonances.json"
CATALOG_REL = "config/v2_singlet/resonance_catalog.json"
LEGACY_MANIFEST_REL = "config/v2_singlet/legacy_v1_manifest.json"
SPECTRAL_MODULE_REL = "na49_mixing/spectral_density.py"
WIDTHS_MODULE_REL = "na49_mixing/resonance_widths.py"
DRIVER_REL = "helpers/benchmarking/staged_r30_spectral_normalization.py"
DRIVER_NESTED_REL = "dev/mixing2006/" + DRIVER_REL

#: The committed family records in config order (the card's family list:
#: the three activated families plus the four declared, inactive charged
#: and strange mesonic families).
EXPECTED_FAMILIES: tuple[tuple[str, tuple[int, int, int], bool], ...] = (
    ("baryonic_np", (1, 1, 0), True),
    ("baryonic_strange", (1, 0, -1), True),
    ("mesonic_neutral", (0, 0, 0), True),
    ("mesonic_charged", (0, 1, 0), False),
    ("mesonic_anticharged", (0, -1, 0), False),
    ("mesonic_strange", (0, 0, -1), False),
    ("mesonic_antistrange", (0, 0, 1), False),
)

#: The activated source QN sets (the card-6 blend report sets).
ACTIVATED_QN_SETS: tuple[tuple[int, int, int], ...] = (
    (1, 1, 0),
    (1, 0, -1),
    (0, 0, 0),
)

#: The declared, inactive QN sets (never sampled in Round 3; integrated
#: here for the schema-completeness normalization record).
DECLARED_INACTIVE_QN_SETS: tuple[tuple[int, int, int], ...] = (
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, -1),
    (0, 0, 1),
)

#: The hermetic seeds block (round 3 index item 6): no Monte Carlo, no
#: replica, no uniform draw; the master seed is recorded for contract
#: uniformity with a zero draw count.
MASTER_SEED = 20260917
DRAWS = 0

#: The fine threshold-verification grid (the card: node spacing 1e-6 GeV
#: around m_min, B, and B + W).  The window half-width 0.005 GeV gives
#: 5,000 nodes on each side of the junction plus the junction node.
FINE_GRID_SPACING = 1e-6
FINE_GRID_NODES_PER_SIDE = 5_000
FINE_GRID_HALF_WIDTH = FINE_GRID_SPACING * FINE_GRID_NODES_PER_SIDE

#: The C^1 derivative tolerance (the card's threshold spec) and the
#: second-order three-point one-sided stencil estimator on the fine grid.
DERIVATIVE_TOLERANCE = 1e-12

#: The rho_total continuity convention (the card-6 definition, the
#: established one-sided probe pair of tests/test_catalog_continuum_
#: matching.py): one-sided probes at +-1e-14 GeV around the junction
#: agree to 1e-12.
CONTINUITY_PROBE_H = 1e-14
CONTINUITY_TOLERANCE = 1e-12

#: The five named threshold checks of the card's gate block.
CHECK_ZERO_BELOW = "zero below m_min"
CHECK_C1_LOWER = "C1 at m_min"
CHECK_C1_UPPER = "C1 at m_min+W"
CHECK_CONT_B = "rho_total continuous at B"
CHECK_CONT_TOP = "rho_total continuous at B+W"
THRESHOLD_CHECK_NAMES = (
    CHECK_ZERO_BELOW,
    CHECK_C1_LOWER,
    CHECK_C1_UPPER,
    CHECK_CONT_B,
    CHECK_CONT_TOP,
)


def _now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats with JSON-safe string
    sentinels (the committed archive convention: ``"Infinity"`` /
    ``"-Infinity"`` / ``"NaN"``; RFC 8259 has no non-finite literals)."""
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0.0 else "-Infinity"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: dict) -> None:
    """UTF-8 JSON with the archive sentinel convention (identical inputs
    produce identical bytes; non-finite floats become string
    sentinels)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


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


# ---------------------------------------------------------------------------
# Committed inputs
# ---------------------------------------------------------------------------


def _load_families():
    return families_from_json(str(NESTED_ROOT / CONTINUUM_CONFIG_REL))


def _load_catalog() -> HadronCatalog:
    return HadronCatalog.from_json(NESTED_ROOT / CATALOG_REL)


def _load_blend_spec() -> BlendSpec:
    return BlendSpec()


def _load_legacy_match_mass() -> float:
    """The M8-frozen ``continuum.match_mass`` of the legacy_v1 manifest
    (the boundary the cards 4-6 constants are cross-linked to)."""
    manifest = json.loads(
        (NESTED_ROOT / LEGACY_MANIFEST_REL).read_text(encoding="utf-8")
    )
    return float(
        manifest["frozen_fit_outputs"]["m8_frozen_pp_parameters"][
            "continuum.match_mass"
        ]
    )


def _assert_dual_tree_driver() -> tuple[str, str]:
    """The two driver copies must be byte-identical (the dual-tree
    contract); returns ``(driver_sha256, driver_mirror_sha256)``."""
    nested_driver = NESTED_ROOT / DRIVER_REL
    mono_driver = MONO_ROOT / DRIVER_REL
    for path in (nested_driver, mono_driver):
        if not path.is_file():
            _refuse(f"dual-tree driver copy missing: {path}")
    nested_sha = _sha256_file(nested_driver)
    mono_sha = _sha256_file(mono_driver)
    if nested_sha != mono_sha:
        _refuse(
            "the dual-tree driver copies disagree (nested "
            f"{nested_sha!r} vs monorepo {mono_sha!r}); re-vendor the "
            "byte-identical mirror before the run"
        )
    return nested_sha, mono_sha


def _input_verification() -> dict:
    """The committed-input hash block (recorded at pre-registration and
    re-verified before every evaluation step)."""
    driver_sha, mirror_sha = _assert_dual_tree_driver()
    config_path = NESTED_ROOT / CONTINUUM_CONFIG_REL
    catalog_path = NESTED_ROOT / CATALOG_REL
    spectral_path = NESTED_ROOT / SPECTRAL_MODULE_REL
    widths_path = NESTED_ROOT / WIDTHS_MODULE_REL
    for path in (config_path, catalog_path, spectral_path, widths_path):
        if not path.is_file():
            _refuse(f"committed input missing: {path}")
    match_mass = _load_legacy_match_mass()
    if match_mass != float(BLEND_BOUNDARY):
        _refuse(
            "the M8-frozen continuum.match_mass of the legacy_v1 manifest "
            f"({match_mass!r}) does not equal the pre-registered blend "
            f"boundary ({float(BLEND_BOUNDARY)!r}); the cards 4-6 "
            "constants are out of alignment with the frozen set"
        )
    # The committed config's own catalog cross-reference must match the
    # live catalog bytes (the card-5 provenance discipline).
    config_raw = json.loads(config_path.read_text(encoding="utf-8"))
    recorded_catalog_sha = str(
        config_raw.get("catalog_reference", {}).get("sha256", "")
    )
    catalog_sha = _sha256_file(catalog_path)
    if recorded_catalog_sha and recorded_catalog_sha != catalog_sha:
        _refuse(
            "the continuum config's catalog_reference sha256 "
            f"({recorded_catalog_sha!r}) does not match the live catalog "
            f"({catalog_sha!r}); the committed pair is out of alignment"
        )
    return {
        "continuum_config_path": CONTINUUM_CONFIG_REL,
        "continuum_config_sha256": _sha256_file(config_path),
        "catalog_sha256": catalog_sha,
        "m8_continuum_match_mass": match_mass,
        "spectral_density_module_sha256": _sha256_file(spectral_path),
        "resonance_widths_module_sha256": _sha256_file(widths_path),
        "driver_sha256": driver_sha,
        "driver_mirror_sha256": mirror_sha,
    }


def _verify_inputs_against_pre(verification: dict, pre: dict) -> None:
    """Every recorded input hash must match the live bytes (a mismatch
    invalidates the run; re-register under a new date)."""
    live = _input_verification()
    for key in (
        "continuum_config_sha256",
        "catalog_sha256",
        "m8_continuum_match_mass",
        "spectral_density_module_sha256",
        "resonance_widths_module_sha256",
        "driver_sha256",
        "driver_mirror_sha256",
    ):
        recorded = pre.get("input_hashes", {}).get(key)
        if recorded is None or live[key] != recorded:
            _refuse(
                f"the live {key} ({live[key]!r}) differs from the "
                f"pre-registered value ({recorded!r}); the run is "
                "invalidated (re-register under a new date)"
            )


# ---------------------------------------------------------------------------
# Pre-registration
# ---------------------------------------------------------------------------


def _family_record_block(family) -> dict:
    return {
        "family_id": family.family_id,
        "qn": [int(v) for v in family.qn],
        "state_type": "continuum_resonance",
        "active": bool(family.active),
        "m_min": float(family.m_min),
        "t_h": float(family.t_h),
        "b_power": float(family.b_power),
        "n_norm": float(family.n_norm),
        "ramp_width": float(family.ramp_width),
        "nominal_width": float(family.nominal_width),
        "m_max": float(SUPPORT_M_MAX),
        "n_allowed_decay_classes": len(family.allowed_decay_classes),
    }


def build_pre_registration(verification: dict) -> dict:
    """The R30 pre-registration payload, written before any evaluation.

    The families are enumerated from the committed config (input
    verification, not evaluation); no integral or threshold check is
    computed here.
    """
    fams = _load_families()
    records = [_family_record_block(family) for family in fams]
    quadrature_nodes = int(REFERENCE_GRID_POINTS)
    if quadrature_nodes != int(SPECTRAL_QUADRATURE_NODES):
        _refuse(
            "the spectral-density reference grid "
            f"({REFERENCE_GRID_POINTS}) does not equal the Round 2 "
            f"reference quadrature ({SPECTRAL_QUADRATURE_NODES}); the "
            "quadrature convention is broken"
        )
    return {
        "schema": "na49_mixing.staged_r30_spectral_normalization"
                  ".pre_registration/1",
        "run": RUN_ID,
        "card": CARD,
        "written_before_evaluation": True,
        "tier": {
            "label": (
                "T0-equivalent hermetic (round 3 index item 6; plan "
                "section 5.1 T0 is the nearest tier): deterministic "
                "quadrature only, no Monte Carlo, no replicas, no event "
                "generation"
            ),
            "monte_carlo": False,
            "replicas": 0,
            "events": 0,
        },
        "families": {
            "source": CONTINUUM_CONFIG_REL,
            "sha256": verification["continuum_config_sha256"],
            "count": len(records),
            "activated_qn_sets": [list(qn) for qn in ACTIVATED_QN_SETS],
            "declared_inactive_qn_sets": [
                list(qn) for qn in DECLARED_INACTIVE_QN_SETS
            ],
            "records": records,
            "note": (
                "all seven records of the committed config in config "
                "order: the three activated families (baryonic (1, 1, 0), "
                "strange baryonic (1, 0, -1), neutral mesonic (0, 0, 0)) "
                "plus the four declared, inactive charged and strange "
                "mesonic families (round 3 index item 9); every record "
                "is unit-normalized over its support and the declared "
                "families are never sampled in Round 3"
            ),
        },
        "quadrature": {
            "rule": "trapezoid",
            "nodes": quadrature_nodes,
            "reference_grid_points": int(REFERENCE_GRID_POINTS),
            "spectral_quadrature_nodes": int(SPECTRAL_QUADRATURE_NODES),
            "convention": (
                "the Round 2 reference quadrature: the fixed 10,001-node "
                "trapezoid over each family's support [m_min, m_max] "
                "(the na49_mixing/resonance_widths.py normalization "
                "convention; SPECTRAL_QUADRATURE_NODES of "
                "na49_mixing.resonance_decay, equal to "
                "REFERENCE_GRID_POINTS of na49_mixing.spectral_density)"
            ),
            "support_per_family": [float(fams[0].m_min), float(SUPPORT_M_MAX)]
            if fams
            else None,
        },
        "normalization_tolerance": float(NORMALIZATION_TOLERANCE),
        "normalization_tolerance_note": (
            "the Round 2 spectral-normalization tolerance on the unit "
            "integral of every activated family and of rho_total for the "
            "three activated source QN sets; the declared inactive "
            "families carry the same recorded normalization for schema "
            "completeness and are integrated with the same tolerance"
        ),
        "threshold_behavior": {
            "zero_below_m_min": {
                "claim": (
                    "rho_a == 0.0 exactly (strictly) for every family "
                    "strictly below m_min, and at m_min itself; no "
                    "continuum strength sits at or below the "
                    "catalog-completeness boundary (plan section 9.3)"
                ),
                "fine_grid": {
                    "node_spacing_gev": FINE_GRID_SPACING,
                    "window_half_width_gev": FINE_GRID_HALF_WIDTH,
                    "nodes_per_side": FINE_GRID_NODES_PER_SIDE,
                },
            },
            "c1_junctions": {
                "claim": (
                    "the one-sided derivatives of rho_a agree to 1e-12 at "
                    "m_min and at m_min + ramp_width (C^1: the smoothstep "
                    "ramp s^2 (3 - 2 s) has exact zero derivative at "
                    "s = 0 and s = 1)"
                ),
                "tolerance": DERIVATIVE_TOLERANCE,
                "estimator": (
                    "second-order three-point one-sided stencils on the "
                    "fine grid (node spacing 1e-6 GeV); below m_min the "
                    "left side is the zero function (left derivative "
                    "exactly 0)"
                ),
            },
            "rho_total_continuity": {
                "claim": (
                    "rho_total is continuous at the blend boundary B and "
                    "at B + W (the card-6 definition)"
                ),
                "definition": (
                    "the card-6 one-sided probe pair: rho_total at "
                    "B - 1e-14 GeV and at B + 1e-14 GeV agree to 1e-12, "
                    "with rho_total(B-) == rho_catalog(B), "
                    "rho_total(B+W+) == rho_cont(B+W), and "
                    "rho_cont(B) == 0.0 exactly"
                ),
                "tolerance": CONTINUITY_TOLERANCE,
                "fine_grid_jump_record": (
                    "the observed two-sided rho_total jump at +-1e-6 GeV "
                    "is recorded for the record: it is O(delta) and "
                    "converges (the linear w ramp), and the pass "
                    "criterion is the card-6 probe pair above"
                ),
            },
        },
        "double_counting_identity": {
            "statement": (
                "int_{M<B}(rho_catalog) + int_{M>=B}(rho_total) = 1 "
                "within 1e-6 for each activated QN set (the card-6 "
                "identity on the fixed 10,001-node union-support grid; "
                "no strength is lost or doubled at the boundary)"
            ),
            "tolerance": float(NORMALIZATION_TOLERANCE),
            "qn_sets": [list(qn) for qn in ACTIVATED_QN_SETS],
        },
        "blend": {
            "boundary_B": float(BLEND_BOUNDARY),
            "ramp_width_W": float(BLEND_RAMP_WIDTH),
            "boundary_note": (
                "B is the M8-frozen continuum.match_mass of the "
                "legacy_v1 manifest (round 3 index item 3); the family "
                "m_min equals B for every family, so the family ramp "
                "junctions m_min and m_min + W coincide numerically with "
                "the blend junctions B and B + W"
            ),
        },
        "seeds": {
            "policy": (
                "none (round 3 index item 6: R30 draws no random "
                "numbers; no Monte Carlo, no replica, no event "
                "generation; the driver consumes no uniform on any path)"
            ),
            "master_seed": MASTER_SEED,
            "draws": DRAWS,
        },
        "no_fitting": True,
        "no_fitting_note": (
            "no optimizer is invoked and no parameter of cards 4-6 is "
            "changed to make a check pass: a failing check stops the run "
            "and re-opens the originating card, never the tolerance"
        ),
        "output_files": [
            PREREGISTRATION_NAME,
            PROVENANCE_NAME,
            FAMILY_INTEGRALS_NAME,
            THRESHOLD_CHECKS_NAME,
            GATE_RECORD_NAME,
            CLEAN_ENV_NAME,
            SEAL_NAME,
            SHA256SUMS_NAME,
        ],
        "input_hashes": dict(verification),
    }


def run_pre_registration(run_dir: Path) -> None:
    """Write ``pre_registration.json`` and ``provenance.json``, stop."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    pre_path = run_dir / PREREGISTRATION_NAME
    evaluation_present = any(
        (run_dir / name).is_file()
        for name in (
            FAMILY_INTEGRALS_NAME,
            THRESHOLD_CHECKS_NAME,
            GATE_RECORD_NAME,
        )
    )
    if pre_path.is_file() and evaluation_present:
        _refuse(
            f"{pre_path} exists and evaluation artifacts are present; "
            "the pre-registration is immutable after evaluation has "
            "started (a change invalidates the run; re-seal under a new "
            "date)"
        )
    verification = _input_verification()
    pre = build_pre_registration(verification)
    _assert_payload_against_card(pre)
    _write_json(pre_path, pre)
    provenance = {
        "schema": "na49_mixing.staged_r30_spectral_normalization"
                  ".provenance/1",
        "run": RUN_ID,
        "card": CARD,
        "driver": DRIVER_NESTED_REL,
        "driver_sha256": verification["driver_sha256"],
        "driver_mirror_sha256": verification["driver_mirror_sha256"],
        "git": _git_info(),
        "environment": environment_fingerprint(),
        "written_utc": _now_utc(),
        "written_before_evaluation": True,
        "inputs": dict(verification),
        "seeds": {"master_seed": MASTER_SEED, "draws": DRAWS,
                  "no_monte_carlo": True},
        "no_fitting": True,
    }
    _write_json(run_dir / PROVENANCE_NAME, provenance)
    print(
        "staged_r30_spectral_normalization: pre-registration written to "
        f"{pre_path} (7 families from the committed config, 10,001-node "
        f"reference quadrature, tolerance {NORMALIZATION_TOLERANCE!r}, "
        f"master seed {MASTER_SEED} with {DRAWS} draws); nothing evaluated"
    )


def _assert_payload_against_card(pre: dict) -> None:
    """Assert the pre-registration payload against the card (a mismatch
    invalidates the run)."""
    problems: list[str] = []
    if pre.get("run") != RUN_ID or pre.get("card") != CARD:
        problems.append("run/card identity mismatch")
    if pre.get("written_before_evaluation") is not True:
        problems.append("written_before_evaluation is not true")
    families = pre.get("families", {})
    if families.get("count") != len(EXPECTED_FAMILIES):
        problems.append("families.count mismatch (the card fixes 7)")
    records = families.get("records", [])
    if len(records) != len(EXPECTED_FAMILIES):
        problems.append("families.records length mismatch")
    else:
        for record, expected in zip(records, EXPECTED_FAMILIES):
            if (
                record.get("family_id") != expected[0]
                or tuple(record.get("qn", ())) != expected[1]
                or bool(record.get("active")) != expected[2]
            ):
                problems.append(
                    f"family record {expected[0]} does not match the card"
                )
    if [list(qn) for qn in families.get("activated_qn_sets", [])] != [
            list(qn) for qn in ACTIVATED_QN_SETS]:
        problems.append("activated_qn_sets mismatch")
    if [list(qn) for qn in families.get("declared_inactive_qn_sets", [])] != [
            list(qn) for qn in DECLARED_INACTIVE_QN_SETS]:
        problems.append("declared_inactive_qn_sets mismatch")
    quadrature = pre.get("quadrature", {})
    if quadrature.get("rule") != "trapezoid":
        problems.append("quadrature rule mismatch")
    if int(quadrature.get("nodes", -1)) != 10_001:
        problems.append("quadrature nodes mismatch (the card fixes 10,001)")
    if int(quadrature.get("spectral_quadrature_nodes", -1)) != 10_001:
        problems.append("spectral_quadrature_nodes mismatch")
    if float(pre.get("normalization_tolerance", -1.0)) != float(
            NORMALIZATION_TOLERANCE):
        problems.append("normalization tolerance mismatch (the card fixes "
                        "1e-6)")
    threshold = pre.get("threshold_behavior", {})
    c1 = threshold.get("c1_junctions", {})
    if float(c1.get("tolerance", -1.0)) != DERIVATIVE_TOLERANCE:
        problems.append("c1 tolerance mismatch (the card fixes 1e-12)")
    continuity = threshold.get("rho_total_continuity", {})
    if float(continuity.get("tolerance", -1.0)) != CONTINUITY_TOLERANCE:
        problems.append("rho_total continuity tolerance mismatch (the "
                        "card-6 definition fixes 1e-12)")
    identity = pre.get("double_counting_identity", {})
    if [list(qn) for qn in identity.get("qn_sets", [])] != [
            list(qn) for qn in ACTIVATED_QN_SETS]:
        problems.append("double-counting qn_sets mismatch")
    if float(identity.get("tolerance", -1.0)) != float(
            NORMALIZATION_TOLERANCE):
        problems.append("double-counting tolerance mismatch (the card "
                        "fixes 1e-6)")
    blend = pre.get("blend", {})
    if float(blend.get("boundary_B", -1.0)) != float(BLEND_BOUNDARY) or \
            float(blend.get("ramp_width_W", -1.0)) != float(BLEND_RAMP_WIDTH):
        problems.append("blend boundary/ramp mismatch")
    seeds = pre.get("seeds", {})
    if int(seeds.get("master_seed", -1)) != MASTER_SEED or \
            int(seeds.get("draws", -1)) != DRAWS:
        problems.append("seeds block mismatch (the card fixes master seed "
                        "20260917 with draws 0)")
    if pre.get("no_fitting") is not True:
        problems.append("no_fitting is not true")
    if problems:
        raise RuntimeError(
            "staged_r30_spectral_normalization: the written "
            "pre-registration does not match this card (the run is "
            "invalidated):\n  " + "\n  ".join(problems)
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
    """Assert the written pre-registration against this driver and the
    live committed inputs (a mismatch invalidates the run)."""
    pre = _load_pre_registration(run_dir)
    try:
        _assert_payload_against_card(pre)
    except RuntimeError as exc:
        _refuse(str(exc))
    _verify_inputs_against_pre(_input_verification(), pre)
    return pre


# ---------------------------------------------------------------------------
# Mode: --integrate
# ---------------------------------------------------------------------------


def run_integrate(run_dir: Path) -> dict:
    """The pre-registered integration (step 2): every family on the
    reference quadrature, and the per-QN-set blend integrals with the
    double-counting identity.

    Writes ``family_integrals.json`` and stops.  A failing check stops
    the run (exit 1) after the report is written; a failing check
    re-opens the originating card, never the tolerance.
    """
    run_dir = Path(run_dir)
    pre = _assert_pre_registration(run_dir)
    fams = _load_families()
    catalog = _load_catalog()
    spec = _load_blend_spec()

    family_rows = []
    for family in fams:
        report = family_normalization_report(family)
        report["active"] = bool(family.active)
        report["n_allowed_decay_classes"] = len(family.allowed_decay_classes)
        family_rows.append(report)
    max_family_dev = max(
        abs(float(row["integral"]) - 1.0) for row in family_rows
    )
    family_pass = all(bool(row["within_tolerance"]) for row in family_rows)

    qn_rows = []
    for qn in ACTIVATED_QN_SETS:
        qn_rows.append(blended_normalization_report(qn, spec, catalog, fams))
    identity_devs = []
    for row in qn_rows:
        identity_devs.append(
            float(row["double_counting_identity"]["split"]["deviation"])
        )
        identity_devs.append(
            float(row["double_counting_identity"]["total"]["deviation"])
        )
    max_identity_dev = max(identity_devs) if identity_devs else 0.0
    identity_pass = all(
        bool(row["double_counting_identity"]["split"]["within_tolerance"])
        and bool(row["double_counting_identity"]["total"]["within_tolerance"])
        for row in qn_rows
    )

    payload = {
        "schema": "na49_mixing.staged_r30_spectral_normalization"
                  ".integration/1",
        "run": RUN_ID,
        "card": CARD,
        "input_hashes": dict(pre["input_hashes"]),
        "quadrature": {
            "rule": "trapezoid",
            "nodes": int(REFERENCE_GRID_POINTS),
            "spectral_quadrature_nodes": int(SPECTRAL_QUADRATURE_NODES),
            "convention": (
                "the Round 2 reference quadrature (the fixed 10,001-node "
                "trapezoid over each family's support and over each "
                "blend's union support)"
            ),
        },
        "families": family_rows,
        "family_normalization": {
            "families_checked": len(family_rows),
            "max_abs_deviation_from_unity": float(max_family_dev),
            "tolerance": float(NORMALIZATION_TOLERANCE),
            "pass": bool(family_pass),
        },
        "activated_qn_sets": qn_rows,
        "double_counting_identity": {
            "qn_sets": [list(qn) for qn in ACTIVATED_QN_SETS],
            "max_abs_deviation_from_unity": float(max_identity_dev),
            "tolerance": float(NORMALIZATION_TOLERANCE),
            "pass": bool(identity_pass),
        },
        "no_fitting": True,
        "monte_carlo": {
            "draws": DRAWS,
            "no_monte_carlo": True,
            "master_seed_recorded": MASTER_SEED,
        },
    }
    _write_json(run_dir / FAMILY_INTEGRALS_NAME, payload)
    if not (family_pass and identity_pass):
        print(
            "staged_r30_spectral_normalization: the integration FAILED "
            f"(family normalization pass = {family_pass}, double-counting "
            f"pass = {identity_pass}); the run stops and the originating "
            "card is re-opened, never the tolerance",
            flush=True,
        )
        raise SystemExit(1)
    print(
        "staged_r30_spectral_normalization: integration written to "
        f"{run_dir / FAMILY_INTEGRALS_NAME} ({len(family_rows)} families, "
        f"max family deviation {max_family_dev:.3e}, max double-counting "
        f"deviation {max_identity_dev:.3e})",
        flush=True,
    )
    return payload


def _assert_integration(run_dir: Path) -> dict:
    path = Path(run_dir) / FAMILY_INTEGRALS_NAME
    if not path.is_file():
        _refuse(
            f"{path} is missing; the run requires the integration (run "
            "--integrate first)"
        )
    report = _read_json(path)
    if not report.get("family_normalization", {}).get("pass") or not \
            report.get("double_counting_identity", {}).get("pass"):
        _refuse(
            f"{path} records a failing block; the run stops (a failing "
            "check re-opens the originating card, never the tolerance)"
        )
    return report


# ---------------------------------------------------------------------------
# Mode: --verify
# ---------------------------------------------------------------------------


def _fine_window(center: float) -> list[float]:
    """The fine grid around *center*: nodes ``center + k * spacing`` for
    ``k = -nodes_per_side .. +nodes_per_side`` (node spacing 1e-6 GeV,
    the card's fine grid)."""
    return [
        center + k * FINE_GRID_SPACING
        for k in range(-FINE_GRID_NODES_PER_SIDE,
                       FINE_GRID_NODES_PER_SIDE + 1)
    ]


def _stencil_right(dens, M0: float, h: float) -> float:
    """Second-order three-point one-sided (forward) derivative stencil."""
    return (
        -3.0 * dens(M0) + 4.0 * dens(M0 + h) - dens(M0 + 2.0 * h)
    ) / (2.0 * h)


def _stencil_left(dens, M0: float, h: float) -> float:
    """Second-order three-point one-sided (backward) derivative
    stencil."""
    return (
        3.0 * dens(M0) - 4.0 * dens(M0 - h) + dens(M0 - 2.0 * h)
    ) / (2.0 * h)


def run_verify(run_dir: Path) -> dict:
    """The pre-registered threshold verification (step 3) on the fine
    grid (node spacing 1e-6 GeV) around ``m_min``, ``B``, and ``B + W``.

    Writes ``threshold_checks.json`` and stops.  A failing check stops
    the run (exit 1) after the report is written.
    """
    run_dir = Path(run_dir)
    pre = _assert_pre_registration(run_dir)
    fams = _load_families()
    catalog = _load_catalog()
    spec = _load_blend_spec()
    h = FINE_GRID_SPACING

    m_min = float(fams[0].m_min)
    for family in fams:
        if float(family.m_min) != m_min:
            _refuse(
                f"family {family.family_id} carries m_min "
                f"{float(family.m_min)!r} while the committed config "
                f"otherwise pins {m_min!r}; the shared-boundary contract "
                "of the cards 4-5 is broken"
            )
    B = float(spec.boundary)
    W = float(spec.ramp_width)
    top = B + W
    junctions = {
        "m_min": {"center": m_min,
                  "role": "the family threshold junction (the ramp start)"},
        "B": {"center": B, "role": "the blend boundary"},
        "B+W": {"center": top,
                "role": "the blend ramp end (the family ramp end)"},
    }
    windows = {name: _fine_window(block["center"])
               for name, block in junctions.items()}

    # -- zero below m_min (every family) ----------------------------------
    zero_rows = []
    below_nodes = [M for M in windows["m_min"] if M < m_min]
    at_node = m_min
    for family in fams:
        n_zero = 0
        for M in below_nodes:
            if family_density(family, M) == 0.0:
                n_zero += 1
        junction_zero = family_density(family, at_node) == 0.0
        zero_rows.append({
            "family_id": family.family_id,
            "qn": [int(v) for v in family.qn],
            "n_nodes_strictly_below": len(below_nodes),
            "n_zero_strictly_below": n_zero,
            "junction_value_zero": bool(junction_zero),
        })
    zero_pass = all(
        row["n_zero_strictly_below"] == row["n_nodes_strictly_below"]
        and row["junction_value_zero"]
        for row in zero_rows
    )

    # -- C^1 at m_min and at m_min + W (every family) ----------------------
    lower_rows = []
    upper_rows = []
    for family in fams:
        dens = lambda M, _fam=family: family_density(_fam, M)
        right_lower = _stencil_right(dens, m_min, h)
        left_lower = 0.0  # the density is the zero function below m_min
        lower_rows.append({
            "family_id": family.family_id,
            "left_derivative": float(left_lower),
            "right_derivative": float(right_lower),
            "mismatch": float(abs(right_lower - left_lower)),
        })
        left_upper = _stencil_left(dens, m_min + W, h)
        right_upper = _stencil_right(dens, m_min + W, h)
        upper_rows.append({
            "family_id": family.family_id,
            "left_derivative": float(left_upper),
            "right_derivative": float(right_upper),
            "mismatch": float(abs(right_upper - left_upper)),
        })
    lower_max = max(row["mismatch"] for row in lower_rows)
    upper_max = max(row["mismatch"] for row in upper_rows)
    lower_pass = lower_max <= DERIVATIVE_TOLERANCE
    upper_pass = upper_max <= DERIVATIVE_TOLERANCE

    # -- rho_total continuity at B and at B + W (card-6 definition) --------
    cont_b_rows = []
    cont_top_rows = []
    for qn in ACTIVATED_QN_SETS:
        rho_total_b, cat_at_b, cont_at_b = blended_density(
            qn, B, spec, catalog, fams
        )
        left_b = blended_density(qn, B - CONTINUITY_PROBE_H, spec,
                                 catalog, fams)[0]
        right_b = blended_density(qn, B + CONTINUITY_PROBE_H, spec,
                                  catalog, fams)[0]
        fine_jump_b = abs(
            blended_density(qn, B + h, spec, catalog, fams)[0]
            - blended_density(qn, B - h, spec, catalog, fams)[0]
        )
        cont_top_val = blended_density(qn, top, spec, catalog, fams)[2]
        below_top = blended_density(qn, top - CONTINUITY_PROBE_H, spec,
                                    catalog, fams)[0]
        above_top = blended_density(qn, top + CONTINUITY_PROBE_H, spec,
                                    catalog, fams)[0]
        fine_jump_top = abs(
            blended_density(qn, top + h, spec, catalog, fams)[0]
            - blended_density(qn, top - h, spec, catalog, fams)[0]
        )
        cont_b_rows.append({
            "qn": [int(v) for v in qn],
            "rho_total_at_boundary": float(rho_total_b),
            "rho_catalog_at_boundary": float(cat_at_b),
            "rho_cont_at_boundary": float(cont_at_b),
            "rho_cont_at_boundary_is_zero": bool(cont_at_b == 0.0),
            "probe_mismatch": float(abs(right_b - left_b)),
            "identity_left_equals_catalog": float(abs(left_b - cat_at_b)),
            "fine_grid_jump_1e-6": float(fine_jump_b),
        })
        cont_top_rows.append({
            "qn": [int(v) for v in qn],
            "rho_total_at_ramp_end": float(
                blended_density(qn, top, spec, catalog, fams)[0]
            ),
            "rho_cont_at_ramp_end": float(cont_top_val),
            "probe_mismatch": float(abs(above_top - below_top)),
            "identity_above_equals_continuum": float(
                abs(above_top - cont_top_val)
            ),
            "fine_grid_jump_1e-6": float(fine_jump_top),
        })
    cont_b_max = max(row["probe_mismatch"] for row in cont_b_rows)
    cont_b_pass = (
        cont_b_max <= CONTINUITY_TOLERANCE
        and all(row["rho_cont_at_boundary_is_zero"] for row in cont_b_rows)
        and all(
            row["identity_left_equals_catalog"] <= CONTINUITY_TOLERANCE
            for row in cont_b_rows
        )
    )
    cont_top_max = max(row["probe_mismatch"] for row in cont_top_rows)
    cont_top_pass = (
        cont_top_max <= CONTINUITY_TOLERANCE
        and all(
            row["identity_above_equals_continuum"] <= CONTINUITY_TOLERANCE
            for row in cont_top_rows
        )
    )

    checks = {
        CHECK_ZERO_BELOW: {
            "families": zero_rows,
            "n_families_checked": len(zero_rows),
            "n_nodes_strictly_below_per_family": len(below_nodes),
            "pass": bool(zero_pass),
        },
        CHECK_C1_LOWER: {
            "families": lower_rows,
            "max_derivative_mismatch": float(lower_max),
            "tolerance": DERIVATIVE_TOLERANCE,
            "pass": bool(lower_pass),
        },
        CHECK_C1_UPPER: {
            "families": upper_rows,
            "max_derivative_mismatch": float(upper_max),
            "tolerance": DERIVATIVE_TOLERANCE,
            "pass": bool(upper_pass),
        },
        CHECK_CONT_B: {
            "qn_sets": cont_b_rows,
            "max_probe_mismatch": float(cont_b_max),
            "tolerance": CONTINUITY_TOLERANCE,
            "pass": bool(cont_b_pass),
        },
        CHECK_CONT_TOP: {
            "qn_sets": cont_top_rows,
            "max_probe_mismatch": float(cont_top_max),
            "tolerance": CONTINUITY_TOLERANCE,
            "pass": bool(cont_top_pass),
        },
    }
    max_derivative_mismatch = max(lower_max, upper_max)
    all_pass = all(checks[name]["pass"] for name in THRESHOLD_CHECK_NAMES)

    payload = {
        "schema": "na49_mixing.staged_r30_spectral_normalization"
                  ".threshold/1",
        "run": RUN_ID,
        "card": CARD,
        "input_hashes": dict(pre["input_hashes"]),
        "fine_grid": {
            "node_spacing_gev": FINE_GRID_SPACING,
            "window_half_width_gev": FINE_GRID_HALF_WIDTH,
            "nodes_per_side": FINE_GRID_NODES_PER_SIDE,
            "nodes_per_window": 2 * FINE_GRID_NODES_PER_SIDE + 1,
            "junctions": {
                name: {
                    "center": block["center"],
                    "role": block["role"],
                    "first_node": windows[name][0],
                    "last_node": windows[name][-1],
                }
                for name, block in junctions.items()
            },
            "coincidence_note": (
                "m_min equals B and m_min + W equals B + W numerically "
                "(the cards 4-6 shared boundary values), so the m_min and "
                "B windows coincide node for node; the three named "
                "windows of the card are computed and recorded separately"
            ),
        },
        "checks": checks,
        "check_names": list(THRESHOLD_CHECK_NAMES),
        "max_derivative_mismatch": float(max_derivative_mismatch),
        "max_continuity_probe_mismatch": float(max(cont_b_max, cont_top_max)),
        "pass": bool(all_pass),
        "no_fitting": True,
        "monte_carlo": {
            "draws": DRAWS,
            "no_monte_carlo": True,
            "master_seed_recorded": MASTER_SEED,
        },
    }
    _write_json(run_dir / THRESHOLD_CHECKS_NAME, payload)
    if not all_pass:
        failed = [name for name in THRESHOLD_CHECK_NAMES
                  if not checks[name]["pass"]]
        print(
            "staged_r30_spectral_normalization: the threshold "
            f"verification FAILED for {failed}; the run stops and the "
            "originating card is re-opened, never the tolerance",
            flush=True,
        )
        raise SystemExit(1)
    print(
        "staged_r30_spectral_normalization: threshold verification "
        f"written to {run_dir / THRESHOLD_CHECKS_NAME} (all 5 checks "
        f"pass, max derivative mismatch {max_derivative_mismatch:.3e})",
        flush=True,
    )
    return payload


def _assert_threshold_checks(run_dir: Path) -> dict:
    path = Path(run_dir) / THRESHOLD_CHECKS_NAME
    if not path.is_file():
        _refuse(
            f"{path} is missing; the run requires the threshold "
            "verification (run --verify first)"
        )
    report = _read_json(path)
    if not report.get("pass"):
        _refuse(
            f"{path} records pass = false; the run stops (a failing "
            "check re-opens the originating card, never the tolerance)"
        )
    return report


# ---------------------------------------------------------------------------
# Mode: --gate
# ---------------------------------------------------------------------------


def run_gate(run_dir: Path) -> dict:
    """Write ``gate3_r30.json``: the Gate 3 family-side evidence,
    assembled from the integration and threshold reports (evidence
    only)."""
    run_dir = Path(run_dir)
    pre = _assert_pre_registration(run_dir)
    integration = _assert_integration(run_dir)
    threshold = _assert_threshold_checks(run_dir)

    family_block = dict(integration["family_normalization"])
    identity_block = dict(integration["double_counting_identity"])

    # Cross-check the stored summaries against their own rows (an
    # internally inconsistent report refuses the gate).
    rows_max = max(
        abs(float(row["integral"]) - 1.0)
        for row in integration["families"]
    )
    if abs(rows_max - float(family_block[
            "max_abs_deviation_from_unity"])) > 0.0:
        _refuse("the integration report's family summary does not match "
                "its own rows")
    qn_max = max(
        float(row["double_counting_identity"]["split"]["deviation"])
        for row in integration["activated_qn_sets"]
    )
    if abs(qn_max - float(identity_block[
            "max_abs_deviation_from_unity"])) > 0.0:
        _refuse("the integration report's identity summary does not match "
                "its own rows")
    threshold_max = max(
        float(threshold["checks"][name]["max_derivative_mismatch"])
        for name in (CHECK_C1_LOWER, CHECK_C1_UPPER)
    )
    if abs(threshold_max - float(threshold[
            "max_derivative_mismatch"])) > 0.0:
        _refuse("the threshold report's derivative summary does not match "
                "its own rows")

    threshold_names = list(THRESHOLD_CHECK_NAMES)
    blocks = {
        "family_normalization": {
            "families_checked": int(family_block["families_checked"]),
            "max_abs_deviation_from_unity": float(
                family_block["max_abs_deviation_from_unity"]
            ),
            "tolerance": float(family_block["tolerance"]),
            "pass": bool(family_block["pass"]),
        },
        "threshold_behavior": {
            "checks": threshold_names,
            "max_derivative_mismatch": float(
                threshold["max_derivative_mismatch"]
            ),
            "max_continuity_probe_mismatch": float(
                threshold["max_continuity_probe_mismatch"]
            ),
            "tolerances": {
                "derivative": DERIVATIVE_TOLERANCE,
                "continuity": CONTINUITY_TOLERANCE,
            },
            "pass": bool(threshold["pass"]),
        },
        "double_counting_identity": {
            "qn_sets": [list(qn) for qn in ACTIVATED_QN_SETS],
            "max_abs_deviation_from_unity": float(
                identity_block["max_abs_deviation_from_unity"]
            ),
            "tolerance": float(identity_block["tolerance"]),
            "pass": bool(identity_block["pass"]),
        },
    }
    payload = {
        "schema": "na49_mixing.staged_r30_spectral_normalization.gate/1",
        "gate": "gate 3 (plan section 9, line 650), family-side evidence",
        "run": RUN_ID,
        "card": CARD,
        "blocks": blocks,
        "input_hashes": {
            "family_integrals_sha256": _sha256_file(
                run_dir / FAMILY_INTEGRALS_NAME
            ),
            "threshold_checks_sha256": _sha256_file(
                run_dir / THRESHOLD_CHECKS_NAME
            ),
        },
        "no_fitting": True,
        "no_monte_carlo": {
            "draws": DRAWS,
            "no_monte_carlo": True,
            "master_seed_recorded": MASTER_SEED,
        },
        "sealed": True,
        "seal_note": (
            "sealed marks the gate evidence as the sealed form: the "
            "step-5 seal covers this file's bytes (the seal manifest "
            "records them after this file is written)"
        ),
        "decision_note": (
            "evidence only; the written proceed / revise / reject "
            "decision on the two Gate 3 clauses is the Gate 3 follow-up "
            "obligation (round 3 index card), recorded in _knowledge/"
        ),
    }
    _write_json(run_dir / GATE_RECORD_NAME, payload)
    print(
        "staged_r30_spectral_normalization: gate evidence written to "
        f"{run_dir / GATE_RECORD_NAME} (family_normalization pass = "
        f"{blocks['family_normalization']['pass']}, threshold_behavior "
        f"pass = {blocks['threshold_behavior']['pass']}, "
        "double_counting_identity pass = "
        f"{blocks['double_counting_identity']['pass']})",
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
    itself, ``seal_manifest.json``, ``provenance.json``, and the gate
    record, which the seal manifest records) and the post-seal
    clean-environment record.
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
    ``provenance.json``, and the gate record), each with its sha256 and
    byte size; it cannot contain its own digest.
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
            "seal_manifest.json, provenance.json, gate3_r30.json) and "
            "the post-seal clean_environment_commands.json; "
            "seal_manifest.json records the payload files plus "
            "SHA256SUMS.txt, provenance.json, and gate3_r30.json; "
            "seal_manifest.json cannot contain its own digest"
        ),
        "files": entries,
    }
    _write_json(run_dir / SEAL_NAME, seal)
    return seal


def run_seal(run_dir: Path) -> dict:
    """Seal the finished run directory (sums inventory plus seal
    manifest).  The pre-registration, the provenance, the integration,
    the threshold verification, and the gate record must exist."""
    run_dir = Path(run_dir)
    for name in (
        PREREGISTRATION_NAME,
        PROVENANCE_NAME,
        FAMILY_INTEGRALS_NAME,
        THRESHOLD_CHECKS_NAME,
        GATE_RECORD_NAME,
    ):
        if not (run_dir / name).is_file():
            _refuse(
                f"{run_dir / name} is missing before sealing (the gate "
                "JSON is part of the seal; run --gate first)"
            )
    write_sha256sums(run_dir)
    seal = write_seal_manifest(run_dir)
    print(
        "staged_r30_spectral_normalization: run dir sealed "
        f"({len(seal['files'])} files listed in {SEAL_NAME})",
        flush=True,
    )
    return seal


# ---------------------------------------------------------------------------
# Mode: --clean-validate
# ---------------------------------------------------------------------------


def run_clean_validate(run_dir: Path) -> dict:
    """Verify the seal, re-run the integration (step 2) from a clean
    interpreter over the committed config, and confirm the report hash
    matches."""
    run_dir = Path(run_dir)
    _load_pre_registration(run_dir)
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

    # -- clean-interpreter re-run of the integration ----------------------
    report_path = run_dir / FAMILY_INTEGRALS_NAME
    before = report_path.read_bytes()
    driver_path = Path(__file__).resolve()
    argv = [sys.executable, "-I", str(driver_path), "--integrate",
            "--run-dir", str(run_dir)]
    probe = subprocess.run(argv, capture_output=True, text=True,
                           cwd=str(MONO_ROOT))
    if probe.returncode != 0:
        _refuse(
            "the clean-interpreter integration probe failed "
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
        "schema": "na49_mixing.staged_r30_spectral_normalization"
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
                "clean-interpreter re-run of the card's integration step "
                "(step 2) over the committed config (rewrites "
                "family_integrals.json; the sealed bytes must be "
                "reproduced byte for byte)"
            ),
            "argv": argv,
            "cwd": str(MONO_ROOT),
        }],
        "environment": environment_fingerprint(),
        "probe_returncode": int(probe.returncode),
        "sealed_report_sha256": {
            FAMILY_INTEGRALS_NAME: hashlib.sha256(before).hexdigest(),
        },
        "recomputed_report_sha256": {
            FAMILY_INTEGRALS_NAME: hashlib.sha256(recomputed).hexdigest(),
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
                "the committed config, catalog, and package are the "
                "inputs"
            ),
        },
    }
    _write_json(run_dir / CLEAN_ENV_NAME, payload)
    if not payload["hashes_match"]:
        print(
            "staged_r30_spectral_normalization: the clean-environment "
            "validation FAILED (the recomputed integration bytes differ "
            "from the sealed report)",
            flush=True,
        )
        raise SystemExit(1)
    print(
        "staged_r30_spectral_normalization: clean-environment validation "
        f"passed ({files_checked} seal files verified, the clean "
        "interpreter reproduced family_integrals.json byte for byte)",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="staged_r30_spectral_normalization.py",
        description=(
            "Execute run R30, the spectral normalization check (Round 3, "
            "card 7): numerically integrate every spectral family and "
            "verify normalization and threshold behavior (plan section 9, "
            "line 638). Hermetic: no Monte Carlo, no seeds, no event "
            "generation, no fitting."
        ),
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--pre-registration", action="store_true",
        help="write pre_registration.json and provenance.json, then stop",
    )
    modes.add_argument(
        "--integrate", action="store_true",
        help="integrate every family and the per-QN-set blends, then stop",
    )
    modes.add_argument(
        "--verify", action="store_true",
        help="run the threshold checks on the fine grid, then stop",
    )
    modes.add_argument(
        "--gate", action="store_true",
        help="write gate3_r30.json (evidence only), then stop",
    )
    modes.add_argument(
        "--seal", action="store_true",
        help="write seal_manifest.json and SHA256SUMS.txt, then stop",
    )
    modes.add_argument(
        "--clean-validate", action="store_true",
        help="verify the seal and re-run the integration cleanly, then "
             "stop",
    )
    parser.add_argument(
        "--run-dir", default=DEFAULT_RUN_DIR,
        help=f"the run directory (default {DEFAULT_RUN_DIR})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_dir = Path(str(args.run_dir))
    if args.pre_registration:
        run_pre_registration(run_dir)
    elif args.integrate:
        run_integrate(run_dir)
    elif args.verify:
        run_verify(run_dir)
    elif args.gate:
        run_gate(run_dir)
    elif args.seal:
        run_seal(run_dir)
    elif args.clean_validate:
        run_clean_validate(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
