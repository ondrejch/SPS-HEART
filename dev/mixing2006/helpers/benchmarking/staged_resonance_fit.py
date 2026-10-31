#!/usr/bin/env python3
"""Staged resonance-decay fit driver (card TASK-2026-08-31, phase P3).

Implements the staged resonance-decay fit workflow of
``.collab/tasks/TASK-2026-08-31-resonance-decay-fit-workflow.md``:

- ``stage0``: the Stage-0 golden reference refit.  The committed golden
  fixture component caches (``tests/fixtures/components/<species>/``,
  captured on the pre-P1 generator at salt 2026-08-26.1) are refit with
  the established sigma-weighted NNLS pattern against the NA49
  paper-window tables.  No generation runs while the committed caches
  are present; on a checkout where a species' caches are absent or
  incomplete, they are regenerated in place from the seeded capture
  path and validated against the committed golden summaries (see
  ``tests/fixtures/README.md``).  This is the reference the Stage-A
  production refit must match within the pre-registered Monte Carlo
  band.
- ``stageA``: the Stage-A baryonic baseline refit at production
  statistics (120000 events per component by default).  Component caches
  are built through the established ``benchmark_sps_species``
  cache/fingerprint machinery under the pre-registered seed protocol:
  per-component base seed ``_component_seed(base_seed, raw_spec_key)``,
  band replicate ``k`` uses ``base + band_seed_stride * k`` (base seed
  20260831, stride 7919, 7 band seeds, per the golden fixture
  convention).  The train/test split is derived from the base seed
  alone and shared across all replicates and configs.
- ``ab``: the A/B harness.  Two soft-override configs run under
  identical seeds, event counts, component definitions, and
  observables; the comparison table carries the pre-registered
  Stage-B candidate boxes (``na49_mixing.parallel.
  stage_b_candidate_boxes``) and the B0 ablation mode (pair family
  forced to zero on the Stage-B fitted parameters).

The driver defines the openness guard floor as a named constant
(``OPENNESS_GUARD_FLOOR = 0.05``, card D1): an enabled mesonic channel
whose realized openness fraction falls below the floor is flagged
tail-only and treated as unidentifiable.  In Stage A the mesonic family
is off, so the guard is recorded as not applicable.

Cross sections are quoted as ``f`` (the invariant convention) under the
invariant weighting convention of the fit kernel.

Layout: the two copies of this driver (the top-level helper tree and the
vendored helper tree) differ only in the layout-path defaults for
``RUNS_ROOT`` and ``--mixing2006-dir``; the layout-sync check normalizes
that difference.  Run artifacts land under the tree-local run-artifact
root (see ``RUNS_ROOT``).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, cast

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers.benchmarking import sps_core
from helpers.benchmarking.benchmark_sps_species import (
    RESONANCE_DECAY_LIB_VERSION,
    _append_component_columns_from_tracks,
    _audit_tables_json,
    _build_resonance_component_cache_one,
    _cache_config_matches,
    _component_cache_path,
    _component_seed,
    _decay_lib_fingerprint,
    _materialize_regge_tower_gen_files,
    _normalize_fragmentation_mode,
    _resonance_spec_key,
    _resonance_specs_for_particle,
)
from helpers.benchmarking.particle_definitions import resolve_species
from helpers.benchmarking.sps_core import (
    PAPER_PT_MAX,
    PAPER_PT_MIN,
    PAPER_XF_MAX,
    PAPER_XF_MIN,
    fit_nnls_weighted,
    metrics_bundle,
    read_na49_points,
    sqrt_s_from_fixed_target_beam_momentum,
)

RUNS_ROOT = REPO_ROOT / "00runs"

# --- Pre-registered constants (card TASK-2026-08-31 rev 2) -------------
DEFAULT_BASE_SEED = 20260831
DEFAULT_BAND_SEEDS = 7
DEFAULT_BAND_SEED_STRIDE = 7919
DEFAULT_BAND_SIGMA_MULTIPLIER = 6.0
DEFAULT_EVENTS_PER_COMPONENT = 120000
OPENNESS_GUARD_FLOOR = 0.05
DEFAULT_MAX_UNDECAYED_FRACTION = 0.01
DEFAULT_BEAM_MOMENTUM_GEV_C = 158.0
GENERATION_BEAM_MOMENTUM_GEV_C = 158.0  # fixed by the component cache builder
DEFAULT_TOWER_MIN_MASS_GEV = 1.9
DEFAULT_TOWER_MAX_MASS_GEV = 2.3
DEFAULT_TOWER_LEVEL_COUNT = 3
DEFAULT_TOWER_SHAPE = "linear"
DEFAULT_TOWER_SHAPE_PARAM = 0.25
DEFAULT_TRAIN_FRAC = 0.7
DEFAULT_FIT_CONVENTION = "invariant"
# Effective M* (mesonic) proxy PDG ids; a channel is mesonic when the
# parent is an M* proxy or any daughter is an M* proxy.
MEFF_PDG_IDS = (9900113, 9900213, 9900313)
ARTIFACT_SCHEMA_VERSION = 1
STAGE_A_ARTIFACT_FILENAME = "stage_a_baseline.json"
STAGE0_REFERENCE_FILENAME = "stage0_golden_refit.json"
AB_COMPARISON_FILENAME = "ab_comparison.json"
TASK_CARD_RELATIVE_PATH = ".collab/tasks/TASK-2026-08-31-resonance-decay-fit-workflow.md"
STAGE_B_CANDIDATE_LABELS = (
    "B-450",
    "B-850",
    "B-1250",
    "B-850W",
    "B-1250W",
    "B-850W2",
    "B-1250W2",
    "B-850G",
    "B-850N",
    "B-850R",
    "B-850C",
    "B-850S",
    "B-850WR",
    "B-850P",
    "B-850L",
)
MESTAR_PPBAR_STRENGTH_FALLBACK_KEY = "mestar_eff_ppbar_strength"

# --- Pre-registered Stage-B grid (card phase P4) --------------------------
# Grid shapes selectable via ``--stage-b-grid``.
STAGE_B_GRID_SHAPES = ("coarse", "full")
# The Stage-B free set (card D4/D5) does not carry the pair-family strength
# c as a grid dimension: the point budgets are spent on the mass/width/
# strength axes.  The pair family therefore runs at this pre-registered
# constant on the pair-capable candidates (B-850, B-1250).  The Stage-C
# B0 ablation (force c = 0 on the fitted parameters) is only non-degenerate
# because the fitted candidate arm carries c > 0.
STAGE_B_PAIR_STRENGTH = 1.0
# Frozen per-level defaults for the non-candidate tower levels.  These
# mirror the effective-state defaults resolved by
# ``na49_mixing.parallel._resolve_effective_tower_overrides`` (card D4):
# the grid widens the tower window to [1.9, 5.0], which would otherwise
# re-space every level (e.g. the M* ladder to 0.45/1.475/2.5), so the
# non-candidate levels are pinned to their Stage-A values at every point.
STAGE_B_NSTAR_MASS_DEFAULTS = {1: 1.9, 2: 2.1, 3: 2.3}
STAGE_B_NSTAR_WIDTH_DEFAULTS = {1: 0.25, 2: 0.30, 3: 0.35}
STAGE_B_MSTAR_MASS_DEFAULTS = {1: 0.45, 2: 0.85, 3: 1.25}
STAGE_B_MSTAR_WIDTH_DEFAULTS = {1: 0.18, 2: 0.22, 3: 0.28}
# Pre-registered grid tables, one entry per candidate.  ``nstar_mass`` /
# ``nstar_width`` / ``strength`` are the free D4 coordinates (N* pole mass,
# pole width, and the D4 strength s with the mesonic branch share and the
# pion feed-down share split as (s, base - s)); pair-capable candidates add
# the two M* coordinates (pole mass and pole width of the candidate level).
# The first mass value concentrates the grid at the box lower edge
# (bound-chain lower limit).
STAGE_B_GRID_TABLES: Dict[str, Dict[str, object]] = {
    "B-450": {
        "level": 1,
        "nstar_mass": (2.07, 2.20, 2.40, 3.20, 5.00),
        "nstar_width": (0.125, 0.25, 0.50),
        "strength": (0.0, 0.075, 0.15),
    },
    "B-850": {
        "level": 2,
        "nstar_mass": (4.50, 4.60, 4.80, 5.00),
        "nstar_width": (0.15, 0.30, 0.60),
        "strength": (0.0, 0.15, 0.30),
        "mestar_mass": (1.88154462, 2.00, 2.25, 2.50),
        "mestar_width": (0.11, 0.22),
    },
    "B-1250": {
        "level": 3,
        "nstar_mass": (4.70, 4.80, 4.90, 5.00),
        "nstar_width": (0.175, 0.35, 0.70),
        "strength": (0.0, 0.225, 0.45),
        "mestar_mass": (1.88154462, 2.00, 2.25, 2.50),
        "mestar_width": (0.14, 0.28),
    },
    # Wide-budget round 2 (pre-registered 2026-09-02 in
    # notes/resonance_fit_stage_progress.md): strength axes extended past
    # the frozen D4 budget with br_nstarpi clamped at zero
    # (STAGE_B_STRENGTH_BUDGET_WIDE); every other axis identical to
    # round 1, so the round-1 points sit inside the W strength axes.
    "B-850W": {
        "level": 2,
        "nstar_mass": (4.50, 4.60, 4.80, 5.00),
        "nstar_width": (0.15, 0.30, 0.60),
        "strength": (0.0, 0.15, 0.30, 0.45, 0.60),
        "mestar_mass": (1.88154462, 2.00, 2.25, 2.50),
        "mestar_width": (0.11, 0.22),
        "strength_budget": 0.60,
    },
    "B-1250W": {
        "level": 3,
        "nstar_mass": (4.70, 4.80, 4.90, 5.00),
        "nstar_width": (0.175, 0.35, 0.70),
        "strength": (0.0, 0.225, 0.45, 0.675, 0.90),
        "mestar_mass": (1.88154462, 2.00, 2.25, 2.50),
        "mestar_width": (0.14, 0.28),
        "strength_budget": 0.90,
    },
    # Round 3 (pre-registered 2026-09-03 in
    # notes/resonance_fit_stage_progress.md): the strength scan completed
    # to the physical bound s = 1.0 with every other axis frozen; the
    # whole axis sits in the pi-branch-exhausted regime. An interior
    # minimum identifies the cascade branch fraction; a pin at 1.0
    # establishes a pure-cascade top level.
    "B-850W2": {
        "level": 2,
        "nstar_mass": (4.50, 4.60, 4.80, 5.00),
        "nstar_width": (0.15, 0.30, 0.60),
        "strength": (0.45, 0.60, 0.70, 0.80, 0.90, 1.00),
        "mestar_mass": (1.88154462, 2.00, 2.25, 2.50),
        "mestar_width": (0.11, 0.22),
        "strength_budget": 1.00,
    },
    "B-1250W2": {
        "level": 3,
        "nstar_mass": (4.70, 4.80, 4.90, 5.00),
        "nstar_width": (0.175, 0.35, 0.70),
        "strength": (0.45, 0.675, 0.7875, 0.90, 0.95, 1.00),
        "mestar_mass": (1.88154462, 2.00, 2.25, 2.50),
        "mestar_width": (0.14, 0.28),
        "strength_budget": 1.00,
    },
    # Round 4 (pre-registered 2026-09-03 in
    # notes/resonance_fit_stage_progress.md): three M*-side explorations
    # at the retained operating point (N*(2) = 4.5/0.15, M*(2) = 2.25,
    # s = 1.0, pair 1.0).
    "B-850G": {
        # E1: the M*/N* widths have pinned at their axis minima in every
        # round; scan both downward with masses and s fixed.
        "level": 2,
        "nstar_mass": (4.50,),
        "nstar_width": (0.08, 0.11, 0.13, 0.15),
        "strength": (1.00,),
        "mestar_mass": (2.25,),
            "strength_budget": 1.00,
    "mestar_width": (0.05, 0.07, 0.09, 0.11),
    },
    "B-850N": {
        # E2: the pair-weight shape exponent (frozen at 1.0 since the D5
        # registration) scanned jointly with the strength it trades
        # against in the near-threshold pbar shape.
        "level": 2,
        "nstar_mass": (4.50,),
        "nstar_width": (0.15,),
        "strength": (0.80, 0.90, 1.00),
        "mestar_mass": (2.25,),
        "mestar_width": (0.11,),
            "strength_budget": 1.00,
    "pair_exponent": (0.5, 1.0, 2.0, 4.0),
    },
    "B-850R": {
        # E3: the bound-chain ridge probe. The M* mass cannot exceed
        # 2.5 GeV at the N*(2) = 4.5 pole (m_N* >= 1.9 + m_M* + 0.1), so
        # the M*-mass extension moves up the ridge jointly. Low-N*/
        # high-M* corners are bound-chain-closed by design; the
        # openness machinery records them and the m_M* slice reads the
        # open points (m_N* >= 2.0 + m_M*).
        "level": 2,
        "box_variant": "wide_mestar",
        "mestar_window_max_mass": 2.75,
        "nstar_mass": (4.50, 4.75, 5.00),
        "nstar_width": (0.15,),
        "strength": (0.90, 1.00),
        "mestar_mass": (2.25, 2.50, 2.75),
        "mestar_width": (0.11,),
        "strength_budget": 1.00,
    },
    # Round 5 (pre-registered 2026-09-03 in
    # notes/resonance_fit_stage_progress.md): the combination grid. The
    # two independently measured round-4 gains (pair exponent 0.5, narrow
    # N* width 0.08) scanned jointly, with the exponent extended past its
    # round-2 axis edge (the surface fell monotonically to 0.5) and the
    # unidentified Gamma_M* flat region probed at its endpoints.
    "B-850C": {
        "level": 2,
        "nstar_mass": (4.50,),
        "nstar_width": (0.06, 0.08),
        "strength": (1.00,),
        "mestar_mass": (2.25,),
        "mestar_width": (0.05, 0.11),
        "pair_exponent": (0.25, 0.35, 0.50),
        "strength_budget": 1.00,
    },
    # Round 6 (pre-registered 2026-09-03 in
    # notes/resonance_fit_stage_progress.md): two parallel explorations
    # at the retained B-850C operating point (n = 0.25, Gamma_N* = 0.06,
    # Gamma_M* = 0.05, s = 1.0, pair 1.0).
    "B-850S": {
        # E-A: the step-plus-roll-off pair weight. The exponent family is
        # edge-pinned toward n = 0 (a flat share above threshold); test
        # the form directly with a logistic roll-on
        # P(M) = c / (1 + exp((M_0 - M)/w)) above the hard pair
        # threshold, with the roll-on midpoint and the roll-off width as
        # the new degrees of freedom. The exponent is carried at 0.25
        # for spec continuity and is inactive under the logistic law.
        "level": 2,
        "nstar_mass": (4.50,),
        "nstar_width": (0.06,),
        "strength": (1.00,),
        "mestar_mass": (2.25,),
        "mestar_width": (0.05,),
        "pair_exponent": (0.25,),
        "pair_law_flag": (1.00,),
        "pair_m0": (2.5, 3.0, 3.5),
        "pair_rolloff": (0.2, 0.4, 0.8),
        "strength_budget": 1.00,
    },
    "B-850WR": {
        # E-B: the width-resolution study. The Gamma_N* surfaces fell
        # monotonically to 0.06; scan down to find where the surface
        # flattens (the observable's resolution limit or a genuine
        # preference for delta-like poles).
        "level": 2,
        "nstar_mass": (4.50,),
        "nstar_width": (0.02, 0.03, 0.04, 0.06),
        "strength": (1.00,),
        "mestar_mass": (2.25,),
        "mestar_width": (0.05,),
        "pair_exponent": (0.25,),
        "strength_budget": 1.00,
    },
    # Round 7 (pre-registered 2026-09-03 in
    # notes/resonance_fit_stage_progress.md): three parallel items.
    "B-850P": {
        # E-A: the three-body p pbar pi0 admixture. The D5 family
        # registers the channel behind mestar_eff_ppbar_pi0_variant
        # (OFF, never scanned); round 7 scans the two/three-body split
        # through mestar_eff_ppbar_pi0_share at the retained point.
        # share = 0 reproduces the pure two-body family.
        "level": 2,
        "nstar_mass": (4.50,),
        "nstar_width": (0.06,),
        "strength": (1.00,),
        "mestar_mass": (2.25,),
        "mestar_width": (0.05,),
        "pair_exponent": (0.25,),
        "pair_pi0_share": (0.0, 0.25, 0.5, 0.75, 1.00),
        "strength_budget": 1.00,
    },
    "B-850L": {
        # E-B: multi-level activation. The level-1 cascade
        # (l1 -> N*(1520) + M*(1) = 0.45) under the retained l2 cascade,
        # geometrically bracketed to [2.07, 2.15] inside the 5.0 GeV
        # window (the l2 cascade needs l1 <= 4.5 - 2.25 - 0.1). pbar-
        # blind by construction; the signal is the pi system.
        "level": 1,
        "nstar_mass": (2.07, 2.15),
        "nstar_width": (0.25,),
        "strength": (0.0, 0.075, 0.15),
        "pin_overrides": {
            "nstar_eff_l2_mass": 4.50,
            "nstar_eff_l2_width": 0.06,
            "mestar_eff_l2_mass": 2.25,
            "mestar_eff_l2_width": 0.05,
            "nstar_eff_l2_br_nstarmeson": 1.00,
            "nstar_eff_l2_br_nstarpi": 0.00,
            "mestar_eff_ppbar_strength": 1.00,
            "mestar_eff_ppbar_exponent": 0.25,
        },
    },
}
# Pre-registered point-count ceilings (card phase P4; W/W2 rounds per the
# progress tracker).
STAGE_B_GRID_LIMITS = {
    "B-450": 60,
    "B-850": 300,
    "B-1250": 300,
    "B-850W": 600,
    "B-1250W": 600,
    "B-850W2": 600,
    "B-1250W2": 600,
    "B-850G": 60,
    "B-850N": 60,
    "B-850R": 60,
    "B-850C": 60,
    "B-850S": 60,
    "B-850WR": 60,
    "B-850P": 60,
    "B-850L": 60,
}
# Widened N* tower window used by every Stage-B grid (box window override):
# the per-level keys place the candidate pole anywhere inside, and the
# generator-card damping windows are stretched around realized poles.
STAGE_B_TOWER_WINDOW_MIN_MASS_GEV = 1.9
STAGE_B_TOWER_WINDOW_MAX_MASS_GEV = 5.0
# Primary PDG ids of the tower components.  A generated sample reads the
# Stage-B overrides only through its primary (the tower parents are the only
# generated states with tower mass/width overrides and D4/D5 branch keys),
# so this is the exact override-relevance partition: tower-primary
# components carry the full per-point override set, base components carry
# the empty set (their decay chains never touch tower states).
STAGE_B_TOWER_PDG_IDS = frozenset(
    {
        9911201, 9912201, 9921201, 9922201, 9931201, 9932201,
        -9911201, -9921201, -9931201,
        9900113, 9900213, 9900313,
    }
)
# Artifact base names; the per-candidate files are
# ``stage_b_grid_<tag>.json`` / ``stage_b_selection_<tag>.json``.
STAGE_B_GRID_FILENAME = "stage_b_grid"
STAGE_B_SELECTION_FILENAME = "stage_b_selection"
STAGE_B_MAX_WIDTH_GEV = 1.5  # sanity bound for grid width coordinates

EXIT_OK = 0
EXIT_RUNTIME_FAILURE = 1
EXIT_FINALIZE_INCOMPLETE = 2
EXIT_BAND_CHECK_FAILED = 3
EXIT_CONFIG_ERROR = 4


# --- Small utilities ----------------------------------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_head(path: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(path),
            capture_output=True,
            text=True,
            timeout=10,
        )
        val = out.stdout.strip()
        return val if out.returncode == 0 and val else None
    except Exception:
        return None


def _canonical_overrides_json(overrides: Optional[Dict[str, float]]) -> str:
    payload = {str(k): float(v) for k, v in dict(overrides or {}).items()}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _std_ddof1(values: Sequence[float]) -> float:
    arr = np.asarray([float(v) for v in values], dtype=float)
    if arr.size < 2:
        return 0.0
    return float(np.std(arr, ddof=1))


def _band_stats(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {"n_samples": 0, "mean": None, "sigma_ddof1": None}
    return {
        "n_samples": len(vals),
        "mean": float(np.mean(vals)),
        "sigma_ddof1": _std_ddof1(vals),
    }


def _to_jsonable(obj):
    """Recursively convert numpy scalars and non-finite floats for JSON."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_sha256_sidecar(path: Path) -> str:
    digest = _sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def _species_fixture_tag(species_key: str) -> str:
    spec = resolve_species(species_key)
    return spec.tag.replace("minus", "_minus").replace("plus", "_plus") if spec.tag in ("piminus", "piplus") else spec.tag


def _resolve_na49_table(species_key: str, explicit: str) -> Path:
    if explicit:
        return Path(explicit)
    return Path(resolve_species(species_key).na49_table)


def _parse_replicate_spec(raw: str, max_replicate: int) -> List[int]:
    """Parse "0,3,5" or "0-7" (or a single int) into replicate indices."""
    out: List[int] = []
    txt = str(raw).strip()
    if not txt:
        return list(range(max_replicate + 1))
    for part in txt.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo < 0 or hi < lo or hi > max_replicate:
                raise ValueError(
                    f"invalid replicate range '{part}'; expected 0..{max_replicate}"
                )
            out.extend(range(lo, hi + 1))
        else:
            val = int(part)
            if val < 0 or val > max_replicate:
                raise ValueError(
                    f"invalid replicate index {val}; expected 0..{max_replicate}"
                )
            out.append(val)
    seen = set()
    ordered: List[int] = []
    for val in out:
        if val not in seen:
            seen.add(val)
            ordered.append(val)
    return ordered


def _parse_arm_spec(raw: str, default_name: str) -> Tuple[str, Dict[str, float]]:
    """Parse "NAME" or "NAME:KEY=VALUE,KEY2=VALUE2" into (name, overrides)."""
    txt = str(raw).strip()
    if not txt:
        return default_name, {}
    if ":" in txt:
        name, body = txt.split(":", 1)
    else:
        name, body = txt, ""
    name = name.strip() or default_name
    overrides: Dict[str, float] = {}
    if body.strip():
        for chunk in body.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "=" not in chunk:
                raise ValueError(f"invalid override '{chunk}'; expected KEY=VALUE")
            key, val = chunk.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError(f"invalid override '{chunk}'; empty key")
            overrides[key] = float(val.strip())
    return name, overrides


def _apply_stage_b_candidate(
    overrides: Dict[str, float],
    candidate_label: str,
    boxes: Dict[int, Dict[str, object]],
) -> Tuple[Dict[str, float], int, Dict[str, object]]:
    """Merge a pre-registered Stage-B box into the soft-override config.

    Returns (overrides, level, box).  Only the N* tower window override is
    applied to the Stage-A generation machinery; the M* window override and
    the strength keys are recorded for the Stage-B fit (card phase P4).
    """
    label = str(candidate_label).strip()
    level = None
    for lvl, box in boxes.items():
        if str(box.get("label")) == label:
            level = int(lvl)
            break
    if level is None:
        raise ValueError(
            f"unknown Stage-B candidate '{label}'; expected one of "
            f"{sorted(str(b.get('label')) for b in boxes.values())}"
        )
    box = boxes[level]
    merged = dict(overrides)
    window = dict(box.get("window_overrides") or {})
    nstar_max = window.get("nstar_eff_tower_max_mass")
    if nstar_max is not None:
        merged["nstar_eff_tower_max_mass"] = float(nstar_max)
    merged["mestar_eff_tower_max_mass"] = float(window.get("mestar_eff_tower_max_mass", 0.0))
    return merged, level, box


def _apply_b0_ablation(overrides: Dict[str, float]) -> Dict[str, float]:
    """B0 ablation: force the M* p pbar pair family strength to zero."""
    merged = dict(overrides)
    merged[MESTAR_PPBAR_STRENGTH_FALLBACK_KEY] = 0.0
    return merged


# --- Stage-B grid (card phase P4) ------------------------------------------
def _stage_b_label_tag(candidate_label: str) -> str:
    """Filename-safe candidate tag ('B-850' -> 'B850')."""
    return str(candidate_label).replace("-", "")


def _stage_b_grid_axes(table: Dict[str, object]) -> Dict[str, List[float]]:
    """Ordered free axes of a candidate's pre-registered grid table."""
    level = int(table["level"])
    axes: Dict[str, List[float]] = {
        f"nstar_eff_l{level}_mass": [float(v) for v in cast(Sequence[float], table["nstar_mass"])],
        f"nstar_eff_l{level}_width": [float(v) for v in cast(Sequence[float], table["nstar_width"])],
        "strength": [float(v) for v in cast(Sequence[float], table["strength"])],
    }
    if "mestar_mass" in table:
        axes[f"mestar_eff_l{level}_mass"] = [float(v) for v in cast(Sequence[float], table["mestar_mass"])]
        axes[f"mestar_eff_l{level}_width"] = [float(v) for v in cast(Sequence[float], table["mestar_width"])]
    if "pair_exponent" in table:
        axes["pair_exponent"] = [float(v) for v in cast(Sequence[float], table["pair_exponent"])]
    if "pair_law_flag" in table:
        axes["pair_law_flag"] = [float(v) for v in cast(Sequence[float], table["pair_law_flag"])]
    if "pair_m0" in table:
        axes["pair_m0"] = [float(v) for v in cast(Sequence[float], table["pair_m0"])]
    if "pair_rolloff" in table:
        axes["pair_rolloff"] = [float(v) for v in cast(Sequence[float], table["pair_rolloff"])]
    if "pair_pi0_share" in table:
        axes["pair_pi0_share"] = [float(v) for v in cast(Sequence[float], table["pair_pi0_share"])]
    return axes


def _stage_b_coarse_subset(values: Sequence[float]) -> List[float]:
    """Coarse-grid subset of one axis: first, middle (index len//2), last.

    Axes with three or fewer values are kept unchanged, so the coarse grid
    is always a subset of the full grid per axis.
    """
    vals = [float(v) for v in values]
    if len(vals) <= 3:
        return vals
    return sorted({vals[0], vals[len(vals) // 2], vals[-1]})


def _stage_b_grid_points(
    candidate_label: str, grid_shape: str
) -> List[Dict[str, object]]:
    """Enumerate the pre-registered grid points of a candidate.

    Returns deterministic Cartesian-product points in fixed axis order
    (N* mass, N* width, D4 strength, then the M* mass/width axes on the
    pair-capable candidates).  Each point carries its free coordinates and
    the pre-registered tie-break displacement (sum of value minus the
    full-table axis minimum over the free coordinates).
    """
    shape = str(grid_shape).strip().lower()
    if shape not in STAGE_B_GRID_SHAPES:
        raise ValueError(
            f"unknown Stage-B grid shape '{shape}'; expected one of {STAGE_B_GRID_SHAPES}"
        )
    candidate = str(candidate_label).strip()
    table = STAGE_B_GRID_TABLES.get(candidate)
    if table is None:
        raise ValueError(
            f"no pre-registered Stage-B grid table for '{candidate}'; "
            f"expected one of {sorted(STAGE_B_GRID_TABLES)}"
        )
    axes_full = _stage_b_grid_axes(table)
    axes = {
        axis: (list(vals) if shape == "full" else _stage_b_coarse_subset(vals))
        for axis, vals in axes_full.items()
    }
    n_points = 1
    for vals in axes.values():
        n_points *= max(len(vals), 1)
    limit = int(STAGE_B_GRID_LIMITS[candidate])
    if n_points > limit:
        raise ValueError(
            f"Stage-B grid for {candidate} ({shape}) has {n_points} points, "
            f"exceeding the pre-registered limit of {limit}"
        )
    points: List[Dict[str, object]] = []
    idx = 0
    for combo in itertools.product(*(axes[axis] for axis in axes)):
        free: Dict[str, float] = dict(zip(axes.keys(), combo))
        points.append(
            {
                "point_id": f"{_stage_b_label_tag(candidate)}-p{idx:03d}",
                "free": free,
                "displacement": sum(
                    float(free[axis]) - min(axes_full[axis]) for axis in axes
                ),
            }
        )
        idx += 1
    return points


def _stage_b_strength_split(
    level: int, s: float, mixing2006_dir: Path, budget: Optional[float] = None
) -> Tuple[float, float]:
    """D4 strength split (s, base_level - s) from the frozen P1 API.

    Wide-budget rounds pass ``budget`` (the strength clamps above the
    frozen base and the pi complement clamps at zero, see
    STAGE_B_STRENGTH_BUDGET_WIDE); without it the frozen-budget split is
    returned unchanged.
    """
    mixdir = str(mixing2006_dir)
    if mixdir not in sys.path:
        sys.path.insert(0, mixdir)
    from na49_mixing import parallel as _parallel

    if budget is None:
        return _parallel.stage_b_strength_split(int(level), float(s))
    return _parallel.stage_b_strength_split(int(level), float(s), budget=float(budget))


def _stage_b_point_overrides(
    candidate_label: str, point: Dict[str, object], mixing2006_dir: Path
) -> Dict[str, float]:
    """Full soft-override set of one Stage-B grid point.

    Windows (the box override), the D4 free coordinates of the candidate
    level (N* mass/width, the strength split), the M* free coordinates plus
    the pre-registered pair strength on the pair-capable candidates, and
    the pinned defaults of every non-candidate level (required because the
    widened window would otherwise re-space the frozen levels).
    """
    candidate = str(candidate_label).strip()
    table = STAGE_B_GRID_TABLES[candidate]
    level = int(table["level"])
    free: Dict[str, float] = {str(k): float(v) for k, v in dict(point["free"]).items()}
    mestar_window = float(cast(float, table.get("mestar_window_max_mass", 2.5)))
    ovr: Dict[str, float] = {
        "nstar_eff_tower_max_mass": STAGE_B_TOWER_WINDOW_MAX_MASS_GEV,
        "mestar_eff_tower_max_mass": mestar_window,
    }
    for j in (1, 2, 3):
        ovr[f"nstar_eff_l{j}_mass"] = float(STAGE_B_NSTAR_MASS_DEFAULTS[j])
        ovr[f"nstar_eff_l{j}_width"] = float(STAGE_B_NSTAR_WIDTH_DEFAULTS[j])
        ovr[f"mestar_eff_l{j}_mass"] = float(STAGE_B_MSTAR_MASS_DEFAULTS[j])
        ovr[f"mestar_eff_l{j}_width"] = float(STAGE_B_MSTAR_WIDTH_DEFAULTS[j])
    ovr[f"nstar_eff_l{level}_mass"] = free[f"nstar_eff_l{level}_mass"]
    ovr[f"nstar_eff_l{level}_width"] = free[f"nstar_eff_l{level}_width"]
    budget_raw = table.get("strength_budget")
    budget = float(cast(float, budget_raw)) if budget_raw is not None else None
    br_nstarmeson, br_nstarpi = _stage_b_strength_split(
        level, free["strength"], mixing2006_dir, budget=budget
    )
    ovr[f"nstar_eff_l{level}_br_nstarmeson"] = float(br_nstarmeson)
    ovr[f"nstar_eff_l{level}_br_nstarpi"] = float(br_nstarpi)
    if "mestar_mass" in table:
        ovr[f"mestar_eff_l{level}_mass"] = free[f"mestar_eff_l{level}_mass"]
        ovr[f"mestar_eff_l{level}_width"] = free[f"mestar_eff_l{level}_width"]
        ovr[MESTAR_PPBAR_STRENGTH_FALLBACK_KEY] = STAGE_B_PAIR_STRENGTH
    if "pair_exponent" in table:
        # Round-4 E2: the pair-weight shape exponent becomes a free axis
        # (mestar_eff_ppbar_exponent; default path bit-identical at 1.0).
        ovr["mestar_eff_ppbar_exponent"] = free["pair_exponent"]
    if "pair_law_flag" in table:
        # Round-6 E-A: the step-plus-roll-off law selection and its two
        # shape parameters (inactive at the default flag 0).
        ovr["mestar_eff_ppbar_law_flag"] = free["pair_law_flag"]
        ovr["mestar_eff_ppbar_rollon_m0"] = free["pair_m0"]
        ovr["mestar_eff_ppbar_rolloff_w"] = free["pair_rolloff"]
    if "pair_pi0_share" in table:
        # Round-7 E-A: the three-body p pbar pi0 admixture (variant ON
        # with the scanned split; share 0 reproduces the pure two-body
        # family).
        ovr["mestar_eff_ppbar_pi0_variant"] = 1.0
        ovr["mestar_eff_ppbar_pi0_share"] = free["pair_pi0_share"]
    pins = table.get("pin_overrides")
    if pins:
        # Round-7 E-B: composition pins (applied last so a multi-level
        # table can hold the retained configuration of a non-candidate
        # level while its own level is the candidate).
        for k, v in dict(pins).items():
            ovr[str(k)] = float(v)
    return ovr


def _stage_b_selection_spec_string(
    name: str, overrides: Dict[str, float]
) -> str:
    """Serialize an override set as a 'NAME:KEY=VAL,...' arm spec.

    The exact-roundtrip float format keeps the string consumable by
    ``_parse_arm_spec`` (ab mode ``--config-b``) with bit-identical values.
    """
    body = ",".join(
        f"{key}={float(value):.17g}" for key, value in sorted(overrides.items())
    )
    return f"{str(name).strip() or 'stageB'}:{body}"


def _stage_b_component_plan(
    specs: Sequence[object],
    full_overrides: Dict[str, float],
    point_ns_suffix: str,
) -> Dict[str, Dict[str, object]]:
    """Per-label Stage-B cache plan (override relevance + namespace).

    Tower-primary components (primary PDG id in ``STAGE_B_TOWER_PDG_IDS``)
    carry the full per-point override set and a per-point cache namespace
    suffix (their caches are point-specific by construction).  Base
    components carry the empty override set and the stable candidate
    namespace, so their caches are built once per candidate and reused by
    every grid point and every species.
    """
    primary_pid_cache: Dict[str, int] = {}
    plan: Dict[str, Dict[str, object]] = {}
    for spec in specs:
        label = str(spec.label)
        gen_path = Path(str(spec.gen_file))
        key = str(gen_path)
        if key not in primary_pid_cache:
            primary_pid_cache[key] = int(gen_path.read_text(encoding="utf-8").split()[0])
        if primary_pid_cache[key] in STAGE_B_TOWER_PDG_IDS:
            plan[label] = {
                "overrides": dict(full_overrides),
                "ns_suffix": str(point_ns_suffix),
            }
        else:
            plan[label] = {"overrides": {}, "ns_suffix": ""}
    return plan


def _validate_stage_b_grid_table(
    candidate_label: str,
    level: int,
    boxes: Dict[int, Dict[str, object]],
    mixing2006_dir: Path,
) -> None:
    """Cross-check the pre-registered grid table against the frozen boxes.

    Validates the candidate level, the N* mass axis against the box N*
    range, the width/sanity bounds, the D4 strength range against the
    frozen base branch, and the M* axes against the box M* range on the
    pair-capable candidates (absent otherwise).
    """
    candidate = str(candidate_label).strip()
    table = STAGE_B_GRID_TABLES.get(candidate)
    if table is None:
        raise ValueError(f"no pre-registered Stage-B grid table for '{candidate}'")
    if int(table["level"]) != int(level):
        raise ValueError(
            f"Stage-B grid table for {candidate} names level "
            f"{int(table['level'])} but the frozen boxes name level {int(level)}"
        )
    box = boxes[int(level)]
    axes = _stage_b_grid_axes(table)
    nstar_range = [float(v) for v in cast(Sequence[float], box["nstar_mass_range"])]
    mass_axis = axes[f"nstar_eff_l{int(level)}_mass"]
    for value in mass_axis:
        if not (nstar_range[0] - 1e-9 <= value <= nstar_range[1] + 1e-9):
            raise ValueError(
                f"Stage-B grid value {value} for {candidate} lies outside the "
                f"frozen box N* range {nstar_range}"
            )
    if abs(min(mass_axis) - nstar_range[0]) > 1e-6:
        raise ValueError(
            f"Stage-B grid for {candidate} must concentrate at the box lower "
            f"edge {nstar_range[0]} (got first value {min(mass_axis)})"
        )
    width_axis = axes[f"nstar_eff_l{int(level)}_width"]
    if any(v <= 0.0 or v > STAGE_B_MAX_WIDTH_GEV for v in width_axis):
        raise ValueError(f"Stage-B grid width values out of bounds: {width_axis}")
    mixdir = str(mixing2006_dir)
    if mixdir not in sys.path:
        sys.path.insert(0, mixdir)
    from na49_mixing import parallel as _parallel

    base_k = float(_parallel.STAGE_B_BASE_NSTARPI[int(level)])
    budget_k = float(cast(float, table.get("strength_budget", base_k)))
    strength_axis = axes["strength"]
    if any(v < -1e-12 or v > budget_k + 1e-12 for v in strength_axis):
        raise ValueError(
            f"Stage-B grid strength values {strength_axis} for {candidate} "
            f"outside [0, budget {budget_k}]"
        )
    pair_capable = bool(box.get("pair_capable"))
    if pair_capable:
        mestar_range = [float(v) for v in cast(Sequence[float], box["mestar_mass_range"])]
        for axis_name in (f"mestar_eff_l{int(level)}_mass", f"mestar_eff_l{int(level)}_width"):
            if axis_name not in axes:
                raise ValueError(
                    f"Stage-B grid table for {candidate} is missing the "
                    f"pair-capable M* axis '{axis_name}'"
                )
        mestar_axis = axes[f"mestar_eff_l{int(level)}_mass"]
        for value in mestar_axis:
            if not (mestar_range[0] - 1e-9 <= value <= mestar_range[1] + 1e-9):
                raise ValueError(
                    f"Stage-B grid value {value} for {candidate} lies outside "
                    f"the frozen box M* range {mestar_range}"
                )
        mestar_width_axis = axes[f"mestar_eff_l{int(level)}_width"]
        if any(v <= 0.0 or v > STAGE_B_MAX_WIDTH_GEV for v in mestar_width_axis):
            raise ValueError(
                f"Stage-B grid M* width values out of bounds: {mestar_width_axis}"
            )
    else:
        for axis_name in (f"mestar_eff_l{int(level)}_mass", f"mestar_eff_l{int(level)}_width"):
            if axis_name in axes:
                raise ValueError(
                    f"Stage-B grid table for {candidate} must not carry the "
                    f"M* axis '{axis_name}' on a non-pair-capable candidate"
                )


def _import_stage_b_api(
    mixing2006_dir: Path, wide_mestar: bool = False
) -> Tuple[Dict[int, Dict[str, object]], str]:
    """Load the pre-registered Stage-B boxes and the pair strength key.

    ``wide_mestar=True`` selects the round-4 wide-M* box variant
    (M* axis to 2.75 GeV, pre-registered in
    notes/resonance_fit_stage_progress.md).
    """
    mixdir = str(mixing2006_dir)
    if mixdir not in sys.path:
        sys.path.insert(0, mixdir)
    from na49_mixing import parallel as _parallel
    from na49_mixing import decays as _decays

    if wide_mestar:
        boxes_raw = _parallel.stage_b_candidate_boxes_wide_mestar()
    else:
        boxes_raw = _parallel.stage_b_candidate_boxes()
    boxes: Dict[int, Dict[str, object]] = {}
    for level, box in boxes_raw.items():
        boxes[int(level)] = {
            str(k): (v if not isinstance(v, tuple) else list(v))
            for k, v in dict(box).items()
        }
    pair_key = str(
        getattr(_decays, "MESTAR_PPBAR_STRENGTH_KEY", MESTAR_PPBAR_STRENGTH_FALLBACK_KEY)
    )
    return boxes, pair_key


def _stage_b_boxes_payload(mixing2006_dir: Path) -> Optional[dict]:
    try:
        boxes, pair_key = _import_stage_b_api(mixing2006_dir)
    except Exception:
        return None
    return {
        "source": "na49_mixing.parallel.stage_b_candidate_boxes",
        "mestar_ppbar_strength_key": pair_key,
        "boxes": {str(k): v for k, v in sorted(boxes.items())},
    }


# --- Openness guard (card D1) -------------------------------------------
def _parse_flat_channel(key: str) -> Tuple[int, Tuple[int, ...]]:
    parent_s, ch_s = str(key).split("|", 1)
    return int(parent_s), tuple(int(p) for p in ch_s.split(","))


def _is_mesonic_channel(
    parent: int, daughters: Tuple[int, ...], meff_ids: Sequence[int]
) -> bool:
    meff = set(int(p) for p in meff_ids)
    if int(parent) in meff:
        return True
    return any(int(d) in meff for d in daughters)


def _evaluate_openness_guard(
    audit_payload: Optional[Dict[str, object]],
    *,
    meff_ids: Sequence[int] = MEFF_PDG_IDS,
    floor: float = OPENNESS_GUARD_FLOOR,
) -> Dict[str, object]:
    """Evaluate the D1 openness guard over one component's audit payload.

    A channel is mesonic when its parent is an M* proxy or any of its
    daughters is an M* proxy.  Enabled channels are those with a nonzero
    requested selection fraction.  An enabled channel whose realized
    (daughter-side) openness fraction is below the floor is flagged
    tail-only; a parent-side openness fraction below the floor is
    recorded as a violation.
    """
    if audit_payload is None:
        return {
            "status": "audit_unavailable",
            "guard_floor": float(floor),
            "enabled_channels": {},
            "tail_only_flags": [],
            "violations": [],
        }
    requested = dict(audit_payload.get("channel_requested") or {})
    openness = dict(audit_payload.get("openness_fraction") or {})
    parent_openness = dict(audit_payload.get("parent_openness_fraction") or {})
    enabled: Dict[str, Dict[str, Optional[float]]] = {}
    for key in sorted(requested):
        try:
            frac = float(requested[key])
        except (TypeError, ValueError):
            continue
        if frac <= 0.0:
            continue
        parent, daughters = _parse_flat_channel(key)
        if not _is_mesonic_channel(parent, daughters, meff_ids):
            continue
        enabled[key] = {
            "requested_fraction": frac,
            "openness_fraction": (
                float(openness[key]) if key in openness else None
            ),
            "parent_openness_fraction": (
                float(parent_openness[key]) if key in parent_openness else None
            ),
        }
    if not enabled:
        return {
            "status": "no_enabled_mesonic_channels",
            "guard_floor": float(floor),
            "enabled_channels": {},
            "tail_only_flags": [],
            "violations": [],
        }
    tail_only: List[Dict[str, object]] = []
    violations: List[Dict[str, object]] = []
    for key, val in sorted(enabled.items()):
        of = val["openness_fraction"]
        if of is not None and of < float(floor):
            tail_only.append({"channel": key, "openness_fraction": of})
        pof = val["parent_openness_fraction"]
        if pof is not None and pof < float(floor):
            violations.append(
                {"channel": key, "parent_openness_fraction": pof}
            )
    return {
        "status": "evaluated",
        "guard_floor": float(floor),
        "enabled_channels": enabled,
        "tail_only_flags": tail_only,
        "violations": violations,
    }


# --- Points and design matrix --------------------------------------------
def _load_species_points(
    table_path: Path,
    *,
    xf_min: float,
    xf_max: float,
    pt_min: float,
    pt_max: float,
) -> Dict[str, np.ndarray]:
    xf, pt, y, sigma = read_na49_points(table_path)
    mask = (
        (xf >= xf_min)
        & (xf <= xf_max)
        & (pt >= pt_min)
        & (pt <= pt_max)
        & (sigma > 0.0)
    )
    xf = xf[mask]
    pt = pt[mask]
    y = y[mask]
    sigma = sigma[mask]
    if xf.size == 0:
        raise ValueError(f"no NA49 points in the paper window of {table_path}")
    xedges = sps_core.edges_from_centers(xf)
    pedges = sps_core.edges_from_centers(pt)
    ix_data, ip_data = sps_core.grid_indices(xf, pt, xedges, pedges)
    return {
        "xf": xf,
        "pt": pt,
        "y": y,
        "sigma": sigma,
        "xedges": xedges,
        "pedges": pedges,
        "ix_data": ix_data,
        "ip_data": ip_data,
        "n_points": int(xf.size),
    }


def _component_arrays_from_cache(cache_path: Path) -> Dict[str, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as z:
        if "source_side" in z:
            side = np.asarray(z["source_side"], dtype=np.int8)
        else:
            side = None
        arrays = {
            "pdg": np.asarray(z["pdg"], dtype=np.int32),
            "xf": np.asarray(z["xf"], dtype=np.float64),
            "pt": np.asarray(z["pt"], dtype=np.float64),
        }
    if side is None:
        side = np.where(arrays["xf"] >= 0.0, 1, -1).astype(np.int8)
    arrays["source_side"] = side
    return arrays


def _build_design_matrix(
    *,
    species_pdg: int,
    components: Sequence[Tuple[str, Dict[str, np.ndarray]]],
    points: Dict[str, np.ndarray],
    sqrt_s_gev: float,
) -> Tuple[List[str], np.ndarray]:
    labels: List[str] = []
    mat_cols: List[np.ndarray] = []
    for label, arrays in components:
        _append_component_columns_from_tracks(
            labels=labels,
            mat_cols=mat_cols,
            component_label=label,
            pdg=arrays["pdg"],
            xf=arrays["xf"],
            pt=arrays["pt"],
            source_side=arrays["source_side"],
            particle_pdg=int(species_pdg),
            xedges=points["xedges"],
            pedges=points["pedges"],
            ix_data=points["ix_data"],
            ip_data=points["ip_data"],
            split_tower_by_source=False,
            target_feedover_xf_max=-1.0,
            projectile_feedover_xf_max=-1.0,
            convention=DEFAULT_FIT_CONVENTION,
            sqrt_s_gev=float(sqrt_s_gev),
        )
    if mat_cols:
        matrix = np.column_stack(mat_cols).astype(float)
    else:
        matrix = np.zeros((points["n_points"], 0), dtype=float)
    return labels, matrix


def _n_target_species(arrays: Dict[str, np.ndarray], species_pdg: int) -> int:
    return int(np.sum(np.asarray(arrays["pdg"], dtype=np.int32) == int(species_pdg)))


# --- Fit -------------------------------------------------------------------
def _train_test_split(
    n_points: int, seed: int, train_frac: float
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(int(n_points))
    n_train = int(float(train_frac) * int(n_points))
    train_idx = np.sort(perm[:n_train])
    test_idx = np.sort(perm[n_train:])
    return train_idx, test_idx


def _fit_and_score(
    matrix: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, object], Dict[str, float], Dict[str, float]]:
    if matrix.shape[1] == 0:
        raise ValueError("design matrix has no columns; fit is undefined")
    weights, info = fit_nnls_weighted(
        matrix[train_idx], y[train_idx], sigma[train_idx], allow_rank_deficient=True
    )
    pred = matrix @ weights
    n_params = max(int(info.get("n_active_weights", 1)), 1)
    train_metrics = metrics_bundle(
        pred[train_idx], y[train_idx], sigma[train_idx], n_params=n_params, prefix="train"
    )
    test_metrics = metrics_bundle(
        pred[test_idx], y[test_idx], sigma[test_idx], n_params=n_params, prefix="test"
    )
    return weights, info, dict(train_metrics), dict(test_metrics)


# --- Stage-0 golden reference ---------------------------------------------
def _load_golden_fixture(mixing2006_dir: Path, species_tag: str) -> dict:
    fixture_dir = mixing2006_dir / "tests" / "fixtures"
    summary_path = fixture_dir / f"golden_stage0_summary_{species_tag}.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing golden fixture summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary


def _golden_component_caches_complete(components_dir: Path, summary: dict) -> bool:
    """True when every component label of the golden summary has its npz cache."""
    labels = [str(label) for label in summary.get("component_labels", [])]
    if not labels:
        return True
    return all((components_dir / f"{label}.npz").is_file() for label in labels)


def _regenerate_golden_component_caches(
    *,
    mixing2006_dir: Path,
    species_tag: str,
    species_pdg: int,
    summary: dict,
) -> None:
    """Rebuild one species' golden fixture component caches in place.

    The component caches under ``tests/fixtures/components/<species>/`` are
    untracked by design (``tests/fixtures/README.md``): the committed records
    are the golden JSON summaries plus the deterministic, seeded capture
    script ``tests/fixtures/make_golden_fixtures.py``.  On a checkout where
    a species' caches are absent or incomplete (a source-only archive), the
    stage-0 reference fit would silently lose all design-matrix columns and
    the driver would exit 1; this rebuilds the missing species' caches
    through the same seeded capture path (``_build_resonance_component_cache_one``
    at the pre-registered per-component seeds recorded in the committed
    summary) and installs them only after validation.

    Validation against the committed golden summaries: the regenerated
    component label set must equal the summary's ``component_labels``, the
    per-component decay kinematic failures must match the committed values,
    and the regenerated species track count must sit inside the
    pre-registered seed-variation band recorded in the committed summary
    (``n_sigma_multiplier * n_target_sigma_used``); the tolerance is never
    loosened post hoc.  A disagreement raises RuntimeError with a message
    naming the drift (component, baseline value, regenerated value, band,
    and the fixture salt versus the salt in force): source or salt drift is
    a real signal, and a silently wrong cache is never accepted.  The
    committed summaries themselves are never rewritten here.
    """
    fixture_salt = str(summary.get("salt_in_force", ""))
    salt_note = (
        f"fixture salt {fixture_salt}, salt in force {RESONANCE_DECAY_LIB_VERSION}"
    )
    baseline_labels = [str(label) for label in summary.get("component_labels", [])]
    baseline_components = dict(summary.get("components", {}))
    n_events = int(summary.get("events_per_component", -1))
    if not baseline_labels:
        raise RuntimeError(
            f"golden fixture summary for {species_tag} records no component labels; "
            "cannot regenerate fixture caches"
        )
    if n_events <= 0:
        raise RuntimeError(
            f"golden fixture summary for {species_tag} does not record a usable "
            f"events_per_component ({n_events}); cannot regenerate fixture caches"
        )
    specs = _resonance_specs_for_particle(int(species_pdg), mixing2006_dir)
    regenerated_labels = [str(spec.label) for spec in specs]
    if regenerated_labels != baseline_labels:
        raise RuntimeError(
            f"golden fixture regeneration for {species_tag} would build a "
            f"different component set {regenerated_labels} than the committed "
            f"summary records {baseline_labels}; {salt_note}. Source drift is a "
            "real signal; regenerate the golden fixtures deliberately instead "
            "of accepting a silently wrong cache."
        )
    components_dir = (
        mixing2006_dir / "tests" / "fixtures" / "components" / species_tag
    )
    with tempfile.TemporaryDirectory(prefix="golden_fixture_regen_") as tmp_name:
        tmp_dir = Path(tmp_name)
        for spec in specs:
            spec_key = _resonance_spec_key(spec, mixing2006_dir)
            task = {
                "label": spec.label,
                "gen_file": str(spec.gen_file),
                "decay_files": [str(p) for p in spec.decay_files],
                "use_kaon_decay_factory": bool(spec.use_kaon_decay_factory),
                "use_soft_decay_factory": bool(spec.use_soft_decay_factory),
                "cache_file": str(tmp_dir / f"{spec.label}.npz"),
                "component_seed": _component_seed(
                    int(summary.get("base_seed", DEFAULT_BASE_SEED)), spec_key
                ),
                "spec_key": spec_key,
                "soft_decay_overrides": {},
                "weak_feed_policy": str(
                    summary.get("weak_feed_policy", "inclusive_weak")
                ),
                "beam_pdg": 2212,
                "target_pdg": 2212,
                "nucleus": None,
                "target_wound_range": None,
            }
            built = _build_resonance_component_cache_one(
                mixing2006_dir=str(mixing2006_dir),
                task=task,
                n_events_per_res=n_events,
                max_decay_passes=int(summary.get("max_decay_passes", 8)),
                fragmentation_mode=str(summary.get("fragmentation_mode", "both")),
                chunk_workers=1,
            )
            if not built.get("ok"):
                raise RuntimeError(
                    "golden fixture regeneration failed for "
                    f"{species_tag}/{spec.label} ({salt_note}): {built}"
                )
            with np.load(tmp_dir / f"{spec.label}.npz", allow_pickle=False) as z:
                pdg = np.asarray(z["pdg"], dtype=np.int32)
                kin_failures = int(z["meta_n_decay_kinematic_failures"])
            entry = dict(baseline_components.get(str(spec.label), {}))
            band = dict(entry.get("mc_band", {}))
            baseline_n = int(entry.get("n_target_species", -1))
            bound = float(band.get("n_sigma_multiplier", 0.0)) * float(
                band.get("n_target_sigma_used", 0.0)
            )
            n_target = int(np.count_nonzero(pdg == int(species_pdg)))
            if abs(n_target - baseline_n) > bound:
                raise RuntimeError(
                    "golden fixture regeneration for "
                    f"{species_tag}/{spec.label} disagrees with the committed "
                    "baseline: n_target_species baseline "
                    f"{baseline_n}, regenerated {n_target}, pre-registered band "
                    f"+/-{bound:.4f} ({float(band.get('n_sigma_multiplier', 0.0))} x "
                    f"{float(band.get('n_target_sigma_used', 0.0))}); {salt_note}. "
                    "Source or salt drift is a real signal; regenerate the golden "
                    "fixtures deliberately (tests/fixtures/make_golden_fixtures.py) "
                    "instead of accepting a silently wrong cache."
                )
            if kin_failures != int(entry.get("n_decay_kinematic_failures", -1)):
                raise RuntimeError(
                    "golden fixture regeneration for "
                    f"{species_tag}/{spec.label} disagrees with the committed "
                    "baseline: n_decay_kinematic_failures baseline "
                    f"{int(entry.get('n_decay_kinematic_failures', -1))}, "
                    f"regenerated {kin_failures}; {salt_note}. Source or salt "
                    "drift is a real signal; regenerate the golden fixtures "
                    "deliberately (tests/fixtures/make_golden_fixtures.py) "
                    "instead of accepting a silently wrong cache."
                )
            # Validated: only now install the cache in place.
            components_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                tmp_dir / f"{spec.label}.npz", components_dir / f"{spec.label}.npz"
            )
    print(
        f"staged-fit: regenerated {len(regenerated_labels)} golden fixture "
        f"component caches for {species_tag} ({salt_note}); validated against "
        "the committed summaries",
        flush=True,
    )


def _run_stage0_reference(
    *,
    mixing2006_dir: Path,
    species_key: str,
    table_path: Path,
    base_seed: int,
    train_frac: float,
    beam_momentum_gev_c: float,
    xf_min: float,
    xf_max: float,
    pt_min: float,
    pt_max: float,
) -> Optional[dict]:
    """Refit the committed golden fixture caches against one species table.

    When a species' component caches are absent or incomplete on this
    checkout, they are regenerated in place from the seeded capture path
    and validated against the committed golden summaries before the refit
    (``_regenerate_golden_component_caches``); with the caches present this
    is a pure refit of the committed caches.
    """
    sp = resolve_species(species_key)
    tag = _species_fixture_tag(species_key)
    try:
        summary = _load_golden_fixture(mixing2006_dir, tag)
    except FileNotFoundError:
        return None
    if not table_path.exists():
        return None
    points = _load_species_points(
        table_path, xf_min=xf_min, xf_max=xf_max, pt_min=pt_min, pt_max=pt_max
    )
    sqrt_s = sqrt_s_from_fixed_target_beam_momentum(beam_momentum_gev_c)
    fixture_components_dir = mixing2006_dir / "tests" / "fixtures" / "components" / tag
    if not _golden_component_caches_complete(fixture_components_dir, summary):
        _regenerate_golden_component_caches(
            mixing2006_dir=mixing2006_dir,
            species_tag=tag,
            species_pdg=int(sp.pdg),
            summary=summary,
        )
    components: List[Tuple[str, Dict[str, np.ndarray]]] = []
    component_meta: Dict[str, dict] = {}
    for label in list(summary.get("component_labels", [])):
        cache_path = fixture_components_dir / f"{label}.npz"
        if not cache_path.exists():
            continue
        arrays = _component_arrays_from_cache(cache_path)
        components.append((str(label), arrays))
        entry = dict(summary.get("components", {}).get(str(label), {}))
        component_meta[str(label)] = {
            "spec_key": str(entry.get("spec_key", "")),
            "component_seed": int(entry.get("component_seed", -1)),
            "n_events": int(entry.get("n_events", -1)),
            "n_tracks": int(entry.get("n_tracks", -1)),
            "n_target_species": _n_target_species(arrays, sp.pdg),
            "mc_band": dict(entry.get("mc_band", {})),
        }
    labels, matrix = _build_design_matrix(
        species_pdg=sp.pdg,
        components=components,
        points=points,
        sqrt_s_gev=sqrt_s,
    )
    y = points["y"]
    sigma = points["sigma"]
    train_idx, test_idx = _train_test_split(points["n_points"], base_seed, train_frac)
    weights, info, train_metrics, test_metrics = _fit_and_score(
        matrix, y, sigma, train_idx, test_idx
    )
    return {
        "species": sp.name,
        "species_pdg": int(sp.pdg),
        "fixture_provenance": {
            "salt_in_force": str(summary.get("salt_in_force", "")),
            "nested_head": str(summary.get("nested_head", "")),
            "outer_head": str(summary.get("outer_head", "")),
            "events_per_component": int(summary.get("events_per_component", -1)),
            "max_decay_passes": int(summary.get("max_decay_passes", -1)),
            "fragmentation_mode": str(summary.get("fragmentation_mode", "")),
            "weak_feed_policy": str(summary.get("weak_feed_policy", "")),
            "base_seed": int(summary.get("base_seed", -1)),
            "band_seeds": int(summary.get("band_seeds", -1)),
            "band_seed_stride": int(summary.get("band_seed_stride", -1)),
            "band_sigma_multiplier": float(summary.get("band_sigma_multiplier", 0.0)),
            "baseline_note": str(summary.get("baseline_note", "")),
        },
        "na49_table": str(table_path),
        "n_points": int(points["n_points"]),
        "train_points": int(train_idx.size),
        "test_points": int(test_idx.size),
        "sqrt_s_gev": float(sqrt_s),
        "fit_convention": DEFAULT_FIT_CONVENTION,
        "components": component_meta,
        "design_columns": labels,
        "weights": {label: float(weights[i]) for i, label in enumerate(labels)},
        "fit_info": dict(info),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }


# --- Stage-A component build loop ------------------------------------------
def _specs_for_species(
    mixing2006_dir: Path,
    species_pdg: int,
    *,
    tower_files: Optional[Dict[str, Path]],
) -> List[object]:
    return _resonance_specs_for_particle(
        int(species_pdg),
        mixing2006_dir,
        include_regge_tower=tower_files is not None,
        regge_tower_gen_files=tower_files,
    )


def _materialize_tower_files(
    *,
    mixing2006_dir: Path,
    out_dir: Path,
    config_ns: str,
    tower_min_mass: float,
    tower_max_mass: float,
    tower_level_count: int,
    tower_shape: str,
    tower_shape_param: float,
    disable_tower: bool,
    tower_mass_law: str = "bw",
    tower_pl_slope: float = 0.0,
) -> Optional[Dict[str, Path]]:
    if disable_tower:
        return None
    return _materialize_regge_tower_gen_files(
        mixing2006_dir=mixing2006_dir,
        tower_gen_root=out_dir / "tower_gen" / config_ns,
        min_mass=float(tower_min_mass),
        max_mass=float(tower_max_mass),
        level_count=int(tower_level_count),
        shape=str(tower_shape),
        shape_param=float(tower_shape_param),
        mass_law=str(tower_mass_law),
        pl_slope=float(tower_pl_slope),
    )


def _fingerprint_for(
    *,
    fingerprint_cache: Dict[Tuple[str, str], Optional[str]],
    gen_file: Path,
    overrides: Dict[str, float],
    mixing2006_dir: Path,
    beam_pdg: int = 2212,
    target_pdg: int = 2212,
    nucleus: Optional[Dict[str, object]] = None,
    target_wound_range: Optional[Tuple[int, int]] = None,
) -> Optional[str]:
    # The system identity (beam/target species, nucleus, wound range) is
    # hashed into the overrides payload so two collision systems sharing a
    # gen card never share a fingerprint (round-10 fix REV-d859919-01: the
    # reuse check alone could not distinguish systems because the fingerprint
    # was system-blind).
    ovr = dict(overrides)
    ovr["__system_beam_pdg__"] = float(beam_pdg)
    ovr["__system_target_pdg__"] = float(target_pdg)
    if nucleus:
        ovr["__system_nucleus_json__"] = float(
            int(hashlib.sha256(
                json.dumps(dict(nucleus), sort_keys=True).encode("utf-8")
            ).hexdigest()[:8], 16
        )
        / 1.0e6
    )
    if target_wound_range:
        ovr["__system_wound_range__"] = float(target_wound_range[0]) * 1000.0 + float(target_wound_range[1])
    key = (str(gen_file), _canonical_overrides_json(ovr))
    if key not in fingerprint_cache:
        fingerprint_cache[key] = _decay_lib_fingerprint(
            gen_file=Path(gen_file),
            soft_decay_overrides=dict(ovr),
            mixing2006_dir=mixing2006_dir,
            beam_pdg=int(beam_pdg),
            target_pdg=int(target_pdg),
        )
    return fingerprint_cache[key]


def _build_component_caches(
    *,
    mixing2006_dir: Path,
    specs: Sequence[object],
    cache_subdir: Path,
    config_ns: str,
    base_seed: int,
    band_seed_stride: int,
    replicate: int,
    events_per_component: int,
    max_decay_passes: int,
    fragmentation_mode: str,
    weak_feed_policy: str,
    overrides: Dict[str, float],
    force_rebuild: bool,
    res_workers: int,
    max_undecayed_fraction: float,
    fingerprint_cache: Dict[Tuple[str, str], Optional[str]],
    stage_b_component_plan: Optional[Dict[str, Dict[str, object]]] = None,
    beam_pdg: int = 2212,
    target_pdg: int = 2212,
    nucleus: Optional[Dict[str, object]] = None,
    target_wound_range: Optional[Tuple[int, int]] = None,
) -> Tuple[Dict[str, dict], Dict[str, Dict[str, np.ndarray]], int, int]:
    nucleus_wound = target_wound_range
    """Build or reuse one component cache per spec; return entries + arrays.

    ``stage_b_component_plan`` (Stage-B mode only) maps a component label
    to the override subset that defines its generated sample plus a cache
    namespace suffix.  Base components carry the empty subset (their decay
    chains never read the tower overrides) so their caches are shared
    across grid points; tower-primary components carry the full per-point
    set and a per-point namespace suffix.
    """
    fragmentation_mode = _normalize_fragmentation_mode(fragmentation_mode)
    plan = stage_b_component_plan or {}
    entries: Dict[str, dict] = {}
    arrays_by_label: Dict[str, Dict[str, np.ndarray]] = {}
    n_reused = 0
    n_built = 0
    cache_subdir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        label = str(spec.label)
        raw_key = _resonance_spec_key(spec, mixing2006_dir)
        spec_plan = plan.get(label)
        ovr = dict(cast(Dict[str, float], spec_plan["overrides"])) if spec_plan is not None else dict(overrides)
        ns_suffix = str(spec_plan.get("ns_suffix", "")) if spec_plan is not None else ""
        spec_key_ns = f"{config_ns}{ns_suffix}|{raw_key}"
        ovr_json = _canonical_overrides_json(ovr)
        cache_path = _component_cache_path(cache_subdir, spec_key_ns, label)
        component_seed = _component_seed(base_seed, raw_key) + int(band_seed_stride) * int(replicate)
        fingerprint = _fingerprint_for(
            fingerprint_cache=fingerprint_cache,
            gen_file=Path(spec.gen_file),
            overrides=ovr,
            mixing2006_dir=mixing2006_dir,
            beam_pdg=beam_pdg,
            target_pdg=target_pdg,
            nucleus=nucleus,
            target_wound_range=nucleus_wound,
        )
        reuse_ok = (not force_rebuild) and _cache_config_matches(
            cache_path,
            spec_key=spec_key_ns,
            component_seed=component_seed,
            n_events_per_res=int(events_per_component),
            max_decay_passes=int(max_decay_passes),
            fragmentation_mode=fragmentation_mode,
            soft_decay_overrides_json=ovr_json,
            weak_feed_policy=str(weak_feed_policy),
            decay_lib_fingerprint=fingerprint,
            beam_pdg=int(beam_pdg),
            target_pdg=int(target_pdg),
        )
        if reuse_ok:
            with np.load(cache_path, allow_pickle=False) as z:
                audit_json = None
                if "meta_channel_audit_json" in z:
                    try:
                        audit_json = json.loads(str(z["meta_channel_audit_json"]))
                    except Exception:
                        audit_json = None
                n_tracks = int(z["pdg"].shape[0])
            print(f"staged-fit: reused {label} rep{replicate} -> {cache_path}", flush=True)
            entries[label] = {
                "spec_key": raw_key,
                "gen_file": str(spec.gen_file),
                "component_seed_base": int(_component_seed(base_seed, raw_key)),
                "component_seed": int(component_seed),
                "cache_file": str(cache_path),
                "cache_reused": True,
                "decay_lib_fingerprint": str(fingerprint) if fingerprint else None,
                "n_tracks": n_tracks,
                "audit": audit_json,
                "audit_source": "cache_reused" if audit_json else "absent_from_cache",
            }
            arrays_by_label[label] = _component_arrays_from_cache(cache_path)
            n_reused += 1
            continue
        task = {
            "spec_key": spec_key_ns,
            "label": label,
            "gen_file": str(spec.gen_file),
            "decay_files": [str(p) for p in spec.decay_files],
            "use_kaon_decay_factory": bool(spec.use_kaon_decay_factory),
            "use_soft_decay_factory": bool(spec.use_soft_decay_factory),
            "cache_file": str(cache_path),
            "component_seed": int(component_seed),
            "fragmentation_mode": fragmentation_mode,
            "soft_decay_overrides": dict(ovr),
            "weak_feed_policy": str(weak_feed_policy),
            "nucleus": dict(nucleus) if nucleus else None,
            "target_wound_range": (
                (int(nucleus_wound[0]), int(nucleus_wound[1]))
                if nucleus_wound
                else None
            ),
            "beam_pdg": int(beam_pdg),
            "target_pdg": int(target_pdg),
            "max_undecayed_fraction": float(max_undecayed_fraction),
            "decay_lib_fingerprint": str(fingerprint) if fingerprint else "",
        }
        result = _build_resonance_component_cache_one(
            mixing2006_dir=str(mixing2006_dir),
            task=task,
            n_events_per_res=int(events_per_component),
            max_decay_passes=int(max_decay_passes),
            fragmentation_mode=fragmentation_mode,
            chunk_workers=max(int(res_workers), 1),
        )
        if not bool(result.get("ok", False)):
            raise RuntimeError(
                f"component build failed: {label} ({result.get('reason', 'unknown')})"
            )
        audit_payload = None
        try:
            audit_payload = json.loads(str(result.get("channel_audit_json") or "{}"))
        except Exception:
            audit_payload = None
        print(f"staged-fit: built {label} rep{replicate} -> {cache_path}", flush=True)
        entries[label] = {
            "spec_key": raw_key,
            "gen_file": str(spec.gen_file),
            "component_seed_base": int(_component_seed(base_seed, raw_key)),
            "component_seed": int(component_seed),
            "cache_file": str(cache_path),
            "cache_reused": False,
            "decay_lib_fingerprint": str(fingerprint) if fingerprint else None,
            "n_tracks": int(result.get("n_tracks", -1)) if "n_tracks" in result else None,
            "audit": audit_payload,
            "audit_source": "fresh_build" if audit_payload else "absent_from_build",
        }
        arrays_by_label[label] = _component_arrays_from_cache(cache_path)
        n_built += 1
    return entries, arrays_by_label, n_reused, n_built


# --- Per-(config, species, replicate) score ---------------------------------
def _score_replicate(
    *,
    mode_tag: str,
    config_ns: str,
    row_kind: str,
    candidate_level: Optional[int],
    point_id: Optional[str] = None,
    species_key: str,
    table_path: Path,
    points: Dict[str, np.ndarray],
    entries: Dict[str, dict],
    arrays_by_label: Dict[str, Dict[str, np.ndarray]],
    spec_order: Sequence[str],
    base_seed: int,
    band_seed_stride: int,
    replicate: int,
    train_frac: float,
    beam_momentum_gev_c: float,
    config_payload: dict,
    out_dir: Path,
) -> dict:
    sp = resolve_species(species_key)
    sqrt_s = sqrt_s_from_fixed_target_beam_momentum(beam_momentum_gev_c)
    components = [
        (label, arrays_by_label[label]) for label in spec_order if label in arrays_by_label
    ]
    labels, matrix = _build_design_matrix(
        species_pdg=sp.pdg,
        components=components,
        points=points,
        sqrt_s_gev=sqrt_s,
    )
    y = points["y"]
    sigma = points["sigma"]
    train_idx, test_idx = _train_test_split(points["n_points"], base_seed, train_frac)
    weights, info, train_metrics, test_metrics = _fit_and_score(
        matrix, y, sigma, train_idx, test_idx
    )
    # Openness guard over the components of this build.
    component_guards: Dict[str, dict] = {}
    any_audit = False
    for label in spec_order:
        entry = entries.get(label)
        if entry is None:
            continue
        guard_payload = _evaluate_openness_guard(entry.get("audit"))
        component_guards[label] = guard_payload
        if entry.get("audit") is not None:
            any_audit = True
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "mode": mode_tag,
        "config_name": config_ns,
        "row_kind": row_kind,
        "candidate_level": candidate_level,
        "point_id": point_id,
        "species": sp.name,
        "species_pdg": int(sp.pdg),
        "na49_table": str(table_path),
        "replicate": int(replicate),
        "base_seed": int(base_seed),
        "band_seed_stride": int(band_seed_stride),
        "component_seed_rule": "_component_seed(base_seed, raw_spec_key) + band_seed_stride*replicate",
        "events_per_component": int(config_payload.get("events_per_component", -1)),
        "max_decay_passes": int(config_payload.get("max_decay_passes", -1)),
        "fragmentation_mode": str(config_payload.get("fragmentation_mode", "")),
        "weak_feed_policy": str(config_payload.get("weak_feed_policy", "")),
        "soft_decay_overrides": dict(config_payload.get("soft_decay_overrides", {})),
        "paper_window": {
            "xf_min": config_payload.get("xf_min"),
            "xf_max": config_payload.get("xf_max"),
            "pt_min": config_payload.get("pt_min"),
            "pt_max": config_payload.get("pt_max"),
        },
        "train_frac": float(train_frac),
        "train_test_split_seed": int(base_seed),
        "fit_convention": DEFAULT_FIT_CONVENTION,
        "sqrt_s_gev": float(sqrt_s),
        "n_points": int(points["n_points"]),
        "train_points": int(train_idx.size),
        "test_points": int(test_idx.size),
        "components": {
            label: {
                "spec_key": entries[label]["spec_key"],
                "component_seed_base": entries[label]["component_seed_base"],
                "component_seed": entries[label]["component_seed"],
                "n_tracks": entries[label].get("n_tracks"),
                "n_target_species": _n_target_species(arrays_by_label[label], sp.pdg),
                "cache_file": entries[label]["cache_file"],
                "cache_reused": bool(entries[label]["cache_reused"]),
                "decay_lib_fingerprint": entries[label].get("decay_lib_fingerprint"),
                "audit_source": entries[label].get("audit_source"),
            }
            for label in spec_order
            if label in entries
        },
        "design_columns": labels,
        "weights": {label: float(weights[i]) for i, label in enumerate(labels)},
        "fit_info": dict(info),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "openness": {
            "guard_floor": OPENNESS_GUARD_FLOOR,
            "meff_pdg_ids": list(MEFF_PDG_IDS),
            "status": (
                "evaluated"
                if any(g.get("status") == "evaluated" for g in component_guards.values())
                else (
                    "no_enabled_mesonic_channels"
                    if any_audit
                    else "audit_unavailable"
                )
            ),
            "components": component_guards,
        },
        "resonance_decay_lib_version": RESONANCE_DECAY_LIB_VERSION,
        "created_at_utc": _utc_now_iso(),
    }
    tag = _species_fixture_tag(species_key)
    if mode_tag == "ab":
        fname = f"ab_results_{config_ns}_{tag}_rep{replicate}.json"
    elif mode_tag == "stageB":
        fname = f"stageB_results_{config_ns}_{point_id}_{tag}_rep{replicate}.json"
    else:
        fname = f"stageA_results_{tag}_rep{replicate}.json"
    _write_json(out_dir / fname, payload)
    print(
        f"staged-fit: scored {mode_tag} {config_ns} {sp.name} rep{replicate} "
        f"test_wrrmse={test_metrics.get('test_weighted_rel_rmse', float('nan')):.6f}",
        flush=True,
    )
    return payload


# --- Band check (acceptance 11) ---------------------------------------------
def _band_check_payload(
    *,
    base_wrrmse: Optional[float],
    band_wrrmseq: Sequence[float],
    stage0_wrrmse: Optional[float],
    band_seeds: int,
) -> dict:
    """Pre-registered Stage-A acceptance check, evaluated per species.

    Criterion (fixed before the production run): the held-out WRRMSE of the
    Stage-A baseline (base seed) must be within the pre-registered Monte
    Carlo band of the Stage-0 reference, i.e.

        stageA_test_wrrmse_base <= stage0_test_wrrmse + 2 * sigma(band seeds),

    where sigma is the ddof=1 standard deviation of the held-out WRRMSE over
    the band-seed replicates (k = 1..band_seeds).  An 8-sample mean +/- 2
    sigma range (base seed included, the golden fixture band convention) is
    recorded as a diagnostic.  The check is evaluated only when at least two
    band replicates and the Stage-0 reference are available; otherwise the
    result is deferred, not failed.
    """
    values = [float(v) for v in band_wrrmseq]
    base = float(base_wrrmse) if base_wrrmse is not None else None
    ref = float(stage0_wrrmse) if stage0_wrrmse is not None else None
    if base is None or ref is None or len(values) < 2:
        return {
            "evaluated": False,
            "reason": (
                "deferred: requires the Stage-0 reference and at least two "
                "band replicates"
            ),
            "base_test_wrrmse": base,
            "stage0_reference_test_wrrmse": ref,
            "band_test_wrrmse": values,
            "band_seeds": int(band_seeds),
            "passes": None,
        }
    sigma = _std_ddof1(values)
    bound = ref + 2.0 * sigma
    passes = bool(base <= bound)
    all_samples = [base] + values
    stats8 = _band_stats(all_samples)
    within_range = None
    if stats8["mean"] is not None and stats8["sigma_ddof1"] is not None:
        within_range = bool(
            base <= stats8["mean"] + 2.0 * stats8["sigma_ddof1"]
            and base >= stats8["mean"] - 2.0 * stats8["sigma_ddof1"]
        )
    return {
        "evaluated": True,
        "criterion": (
            "stageA_test_wrrmse_base <= "
            "stage0_test_wrrmse + 2*sigma(band_seed_test_wrrmse)"
        ),
        "base_test_wrrmse": base,
        "stage0_reference_test_wrrmse": ref,
        "band_test_wrrmse": values,
        "band_seeds": int(band_seeds),
        "band_sigma_ddof1": sigma,
        "bound": bound,
        "passes": passes,
        "diagnostic_8sample": {
            **stats8,
            "base_within_mean_plusminus_2sigma": within_range,
        },
    }


# --- Stage-A finalize ---------------------------------------------------------
def _finalize_stage_a(
    *,
    args,
    out_dir: Path,
    mixing2006_dir: Path,
    species_keys: Sequence[str],
    replicates: Sequence[int],
    config_payload: dict,
) -> Tuple[dict, List[str]]:
    """Assemble the frozen Stage-A baseline artifact from replicate results."""
    missing: List[str] = []
    results: Dict[str, Dict[int, dict]] = {sp: {} for sp in species_keys}
    for sp_key in species_keys:
        sp = resolve_species(sp_key)
        tag = _species_fixture_tag(sp_key)
        for k in replicates:
            path = out_dir / f"stageA_results_{tag}_rep{k}.json"
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                results[sp_key][int(k)] = payload
            else:
                missing.append(f"{path.name}")
    if missing:
        return {}, missing

    # Stage-0 reference (committed fixture caches refit against the same
    # tables); computed in-process so the artifact is self-contained.
    stage0: Dict[str, Optional[dict]] = {}
    for sp_key in species_keys:
        table_path = _resolve_na49_table(sp_key, args.na49_data)
        stage0[sp_key] = _run_stage0_reference(
            mixing2006_dir=mixing2006_dir,
            species_key=sp_key,
            table_path=table_path,
            base_seed=int(args.base_seed),
            train_frac=float(args.train_frac),
            beam_momentum_gev_c=float(args.beam_momentum),
            xf_min=float(args.xf_min),
            xf_max=float(args.xf_max),
            pt_min=float(args.pt_min),
            pt_max=float(args.pt_max),
        )

    species_payload: Dict[str, dict] = {}
    band_checks: Dict[str, dict] = {}
    for sp_key in species_keys:
        sp = resolve_species(sp_key)
        reps = results[sp_key]
        base_payload = reps.get(0)
        band_wrrmseq: List[float] = []
        for k in range(1, int(args.band_seeds) + 1):
            if k in reps:
                tm = dict(reps[k].get("test_metrics", {}))
                band_wrrmseq.append(float(tm.get("test_weighted_rel_rmse", 0.0)))
        base_wrrmse = None
        if base_payload is not None:
            base_wrrmse = float(
                dict(base_payload.get("test_metrics", {})).get("test_weighted_rel_rmse")
            )
        stage0_payload = stage0.get(sp_key)
        stage0_wrrmse = None
        if stage0_payload is not None:
            stage0_wrrmse = float(
                dict(stage0_payload.get("test_metrics", {})).get("test_weighted_rel_rmse")
            )
        band_checks[sp_key] = _band_check_payload(
            base_wrrmse=base_wrrmse,
            band_wrrmseq=band_wrrmseq,
            stage0_wrrmse=stage0_wrrmse,
            band_seeds=int(args.band_seeds),
        )
        species_payload[sp_key] = {
            "species": sp.name,
            "species_pdg": int(sp.pdg),
            "na49_table": str(_resolve_na49_table(sp_key, args.na49_data)),
            "base_replicate": base_payload,
            "band_replicates": {str(k): reps[k] for k in sorted(reps) if k > 0},
            "band": {
                "base_test_wrrmse": base_wrrmse,
                "replicate_test_wrrmse": band_wrrmseq,
                "band_sigma_ddof1": _std_ddof1(band_wrrmseq),
            },
        }

    all_passes: Optional[bool] = None
    evaluated = [c["passes"] for c in band_checks.values() if c.get("passes") is not None]
    if evaluated:
        all_passes = all(bool(p) for p in evaluated)

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact": "stage_a_baseline",
        "stage": "stageA",
        "task_card": TASK_CARD_RELATIVE_PATH,
        "created_at_utc": _utc_now_iso(),
        "producer": {
            "python": platform.python_version(),
            "numpy": str(np.__version__),
            "na49_mixing_head": _git_head(mixing2006_dir),
            "repo_head_outer": _git_head(REPO_ROOT),
            "repo_head_nested": _git_head(mixing2006_dir),
        },
        "environment": {
            "mixing2006_dir": str(mixing2006_dir),
            "cache_dir": str(config_payload.get("cache_dir", "")),
            "generation_beam_momentum_gev_c": GENERATION_BEAM_MOMENTUM_GEV_C,
        },
        "config": config_payload,
        "seed_protocol": {
            "base_seed": int(args.base_seed),
            "band_seeds": int(args.band_seeds),
            "band_seed_stride": int(args.band_seed_stride),
            "band_sigma_multiplier": float(args.band_sigma_multiplier),
            "component_seed_rule": (
                "_component_seed(base_seed, raw_spec_key) + band_seed_stride*replicate"
            ),
            "train_test_split_rule": (
                "np.random.default_rng(base_seed).permutation(n); "
                "n_train = int(train_frac * n); shared across replicates and configs"
            ),
        },
        "resonance_decay_lib_version": RESONANCE_DECAY_LIB_VERSION,
        "species": species_payload,
        "stage0_reference": {
            sp_key: stage0.get(sp_key) for sp_key in species_keys
        },
        "band_check": {
            "pre_registered": True,
            "per_species": {sp_key: band_checks[sp_key] for sp_key in species_keys},
            "all_passes": all_passes,
        },
        "stage_b_boxes": _stage_b_boxes_payload(mixing2006_dir),
    }
    return artifact, []


# --- A/B harness ---------------------------------------------------------------
def _finalize_ab_comparison(
    *,
    out_dir: Path,
    mixing2006_dir: Path,
    species_keys: Sequence[str],
    replicates: Sequence[int],
    arm_a: dict,
    arm_b: dict,
    shared_protocol: dict,
) -> dict:
    per_species: Dict[str, dict] = {}
    for sp_key in species_keys:
        sp = resolve_species(sp_key)
        tag = _species_fixture_tag(sp_key)

        def _arm_block(arm: dict) -> dict:
            base_payload = None
            band_values: List[float] = []
            for k in sorted(replicates):
                path = out_dir / f"ab_results_{arm['name']}_{tag}_rep{k}.json"
                if not path.exists():
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                if k == 0:
                    base_payload = payload
                else:
                    band_values.append(
                        float(dict(payload.get("test_metrics", {})).get("test_weighted_rel_rmse", 0.0))
                    )
            block = {
                "base": base_payload,
                "band": {
                    "replicate_test_wrrmse": band_values,
                    "band_sigma_ddof1": _std_ddof1(band_values),
                },
            }
            return block

        block_a = _arm_block(arm_a)
        block_b = _arm_block(arm_b)
        delta: Dict[str, object] = {}
        if block_a.get("base") is not None and block_b.get("base") is not None:
            tm_a = dict(block_a["base"].get("test_metrics", {}))
            tm_b = dict(block_b["base"].get("test_metrics", {}))
            for key in (
                "test_weighted_rel_rmse",
                "test_chi2_dof",
                "test_log_pull_rms",
                "test_smape_percent",
            ):
                if key in tm_a and key in tm_b:
                    delta[key] = float(tm_b[key]) - float(tm_a[key])
            weights_a = dict(block_a["base"].get("weights", {}))
            weights_b = dict(block_b["base"].get("weights", {}))
            delta["weights"] = {
                label: float(weights_b.get(label, 0.0)) - float(weights_a.get(label, 0.0))
                for label in sorted(set(weights_a) | set(weights_b))
            }
        per_species[sp_key] = {
            "a": block_a,
            "b": block_b,
            "delta_b_minus_a": delta,
        }
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact": "ab_comparison",
        "created_at_utc": _utc_now_iso(),
        "arms": {
            "a": {
                "config_name": arm_a["name"],
                "soft_decay_overrides": arm_a["overrides"],
                "candidate_level": arm_a.get("candidate_level"),
                "row_kind": arm_a.get("row_kind", "stageA_baseline"),
            },
            "b": {
                "config_name": arm_b["name"],
                "soft_decay_overrides": arm_b["overrides"],
                "candidate_level": arm_b.get("candidate_level"),
                "row_kind": arm_b.get("row_kind", "candidate"),
                "b0_ablation": bool(arm_b.get("b0_ablation", False)),
            },
        },
        "shared_protocol": shared_protocol,
        "per_species": per_species,
        "stage_b_boxes": _stage_b_boxes_payload(mixing2006_dir),
    }
    _write_json(out_dir / AB_COMPARISON_FILENAME, payload)
    _write_sha256_sidecar(out_dir / AB_COMPARISON_FILENAME)
    return payload


# --- Config payload -----------------------------------------------------------
def _config_payload(*, args, overrides: Dict[str, float], config_ns: str, extra: Optional[dict] = None) -> dict:
    payload = {
        "config_name": config_ns,
        "events_per_component": int(args.events_per_component),
        "max_decay_passes": int(args.res_max_decay_passes),
        "fragmentation_mode": str(args.res_fragmentation_mode),
        "weak_feed_policy": str(args.weak_feed_policy),
        "soft_decay_overrides": dict(overrides),
        "max_undecayed_fraction": float(args.max_undecayed_fraction),
        "res_workers": int(args.res_workers),
        "tower": {
            "enabled": not bool(args.disable_tower),
            "min_mass_gev": float(args.tower_min_mass),
            "max_mass_gev": float(args.tower_max_mass),
            "level_count": int(args.tower_level_count),
            "shape": str(args.tower_shape),
            "shape_param": float(args.tower_shape_param),
        },
        "xf_min": float(args.xf_min),
        "xf_max": float(args.xf_max),
        "pt_min": float(args.pt_min),
        "pt_max": float(args.pt_max),
        "beam_momentum_gev_c": float(args.beam_momentum),
        "train_frac": float(args.train_frac),
        "fit_convention": DEFAULT_FIT_CONVENTION,
        "cache_dir": str(args.cache_dir) if args.cache_dir else None,
    }
    if extra:
        payload.update(extra)
    return payload


# --- Modes -----------------------------------------------------------------------
def _run_stage0(args) -> int:
    mixing2006_dir = Path(args.mixing2006_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (RUNS_ROOT / args.run_description)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact": "stage0_golden_reference",
        "stage": "stage0_golden_baseline",
        "task_card": TASK_CARD_RELATIVE_PATH,
        "created_at_utc": _utc_now_iso(),
        "base_seed": int(args.base_seed),
        "train_frac": float(args.train_frac),
        "beam_momentum_gev_c": float(args.beam_momentum),
        "fit_convention": DEFAULT_FIT_CONVENTION,
        "paper_window": {
            "xf_min": float(args.xf_min),
            "xf_max": float(args.xf_max),
            "pt_min": float(args.pt_min),
            "pt_max": float(args.pt_max),
        },
        "species": {},
    }
    for sp_key in args.species:
        table_path = _resolve_na49_table(sp_key, args.na49_data)
        ref = _run_stage0_reference(
            mixing2006_dir=mixing2006_dir,
            species_key=sp_key,
            table_path=table_path,
            base_seed=int(args.base_seed),
            train_frac=float(args.train_frac),
            beam_momentum_gev_c=float(args.beam_momentum),
            xf_min=float(args.xf_min),
            xf_max=float(args.xf_max),
            pt_min=float(args.pt_min),
            pt_max=float(args.pt_max),
        )
        if ref is None:
            print(
                f"staged-fit: stage0 reference unavailable for {sp_key} "
                f"(missing fixture or table: {table_path})",
                flush=True,
            )
            continue
        payload["species"][sp_key] = ref
        tm = dict(ref.get("test_metrics", {}))
        print(
            f"staged-fit: stage0 {sp_key} test_wrrmse="
            f"{tm.get('test_weighted_rel_rmse', float('nan')):.6f} "
            f"test_log_pull_rms={tm.get('test_log_pull_rms', float('nan')):.6f}",
            flush=True,
        )
    if not payload["species"]:
        print("staged-fit: no stage0 references produced; check fixtures/tables", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    path = out_dir / STAGE0_REFERENCE_FILENAME
    _write_json(path, payload)
    digest = _write_sha256_sidecar(path)
    print(f"staged-fit: wrote {path} (sha256 {digest})", flush=True)
    return EXIT_OK


def _common_generation_kwargs(args) -> dict:
    return {
        "mixing2006_dir": Path(args.mixing2006_dir),
        "base_seed": int(args.base_seed),
        "band_seed_stride": int(args.band_seed_stride),
        "events_per_component": int(args.events_per_component),
        "max_decay_passes": int(args.res_max_decay_passes),
        "fragmentation_mode": str(args.res_fragmentation_mode),
        "weak_feed_policy": str(args.weak_feed_policy),
        "force_rebuild": bool(args.force_rebuild),
        "res_workers": int(args.res_workers),
        "max_undecayed_fraction": float(args.max_undecayed_fraction),
        "beam_pdg": int(getattr(args, "beam_pdg", 2212)),
        "target_pdg": int(getattr(args, "target_pdg", 2212)),
        "nucleus": getattr(args, "pa_nucleus_dict", None),
        "target_wound_range": getattr(args, "pa_wound_range", None),
    }


def _run_stage_a(args) -> int:
    mixing2006_dir = Path(args.mixing2006_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (RUNS_ROOT / args.run_description)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (out_dir / "res_cache")

    replicates = _parse_replicate_spec(args.replicates, int(args.band_seeds))
    kwargs = _common_generation_kwargs(args)
    overrides: Dict[str, float] = {}
    config_ns = "stageA"
    tower_files = _materialize_tower_files(
        mixing2006_dir=mixing2006_dir,
        out_dir=out_dir,
        config_ns=config_ns,
        tower_min_mass=float(args.tower_min_mass),
        tower_max_mass=float(args.tower_max_mass),
        tower_level_count=int(args.tower_level_count),
        tower_shape=str(args.tower_shape),
        tower_shape_param=float(args.tower_shape_param),
        disable_tower=bool(args.disable_tower),
    )
    config_payload = _config_payload(
        args=args,
        overrides=overrides,
        config_ns=config_ns,
        extra={
            "cache_dir": str(cache_dir),
            "replicates": replicates,
            "tower_files": {k: str(v) for k, v in (tower_files or {}).items()},
        },
    )
    if not args.finalize_only:
        fingerprint_cache: Dict[Tuple[str, str], Optional[str]] = {}
        for sp_key in args.species:
            sp = resolve_species(sp_key)
            specs = _specs_for_species(mixing2006_dir, sp.pdg, tower_files=tower_files)
            spec_order = [str(s.label) for s in specs]
            for k in replicates:
                print(
                    f"staged-fit: stageA {sp.name} rep{k} components={len(specs)} "
                    f"events={args.events_per_component}",
                    flush=True,
                )
                cache_subdir = cache_dir / config_ns / f"rep{k}"
                entries, arrays_by_label, n_reused, n_built = _build_component_caches(
                    specs=specs,
                    cache_subdir=cache_subdir,
                    config_ns=config_ns,
                    overrides=overrides,
                    replicate=k,
                    fingerprint_cache=fingerprint_cache,
                    **kwargs,
                )
                table_path = _resolve_na49_table(sp_key, args.na49_data)
                points = _load_species_points(
                    table_path,
                    xf_min=float(args.xf_min),
                    xf_max=float(args.xf_max),
                    pt_min=float(args.pt_min),
                    pt_max=float(args.pt_max),
                )
                _score_replicate(
                    mode_tag="stageA",
                    config_ns=config_ns,
                    row_kind="stageA_baseline",
                    candidate_level=None,
                    species_key=sp_key,
                    table_path=table_path,
                    points=points,
                    entries=entries,
                    arrays_by_label=arrays_by_label,
                    spec_order=spec_order,
                    base_seed=int(args.base_seed),
                    band_seed_stride=int(args.band_seed_stride),
                    replicate=k,
                    train_frac=float(args.train_frac),
                    beam_momentum_gev_c=float(args.beam_momentum),
                    config_payload=config_payload,
                    out_dir=out_dir,
                )
                print(
                    f"staged-fit: {sp.name} rep{k} cache built={n_built} reused={n_reused}",
                    flush=True,
                )
    if args.finalize or args.finalize_only:
        artifact, missing = _finalize_stage_a(
            args=args,
            out_dir=out_dir,
            mixing2006_dir=mixing2006_dir,
            species_keys=list(args.species),
            replicates=replicates,
            config_payload=config_payload,
        )
        if missing:
            print(
                "staged-fit: finalize incomplete; missing replicate results: "
                + ", ".join(sorted(set(missing))),
                file=sys.stderr,
            )
            return EXIT_FINALIZE_INCOMPLETE
        path = out_dir / STAGE_A_ARTIFACT_FILENAME
        _write_json(path, artifact)
        digest = _write_sha256_sidecar(path)
        all_passes = dict(artifact.get("band_check", {})).get("all_passes")
        print(f"staged-fit: wrote {path} (sha256 {digest})", flush=True)
        if all_passes is False:
            print(
                "staged-fit: BAND CHECK FAILED; Stage-A held-out WRRMSE outside the "
                "pre-registered band of the Stage-0 reference. See band_check in "
                + STAGE_A_ARTIFACT_FILENAME,
                file=sys.stderr,
            )
            return EXIT_BAND_CHECK_FAILED
        return EXIT_OK
    return EXIT_OK


def _run_stage_b(args) -> int:
    """Run the pre-registered Stage-B grid fit for one candidate.

    Grid points run on the base replicate (replicate 0, the pre-registered
    grid statistic; band replicates are a Stage-A Monte Carlo device and
    are not part of the grid).  Base components share one cache across the
    whole grid; tower-primary components carry the full per-point override
    set and are rebuilt per point.  The grid surface and the selection
    payload (with the exact arm-spec string for the Stage-C ``ab``
    handoff) are written with sha256 sidecars.
    """
    if not args.stage_b_candidate:
        print(
            "staged-fit: --mode stageB requires --stage-b-candidate", file=sys.stderr
        )
        return EXIT_CONFIG_ERROR
    candidate = str(args.stage_b_candidate)
    grid_shape = str(args.stage_b_grid)
    mixing2006_dir = Path(args.mixing2006_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (RUNS_ROOT / args.run_description)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (out_dir / "res_cache")

    table = STAGE_B_GRID_TABLES[candidate]
    boxes_raw, pair_key = _import_stage_b_api(
        mixing2006_dir, wide_mestar=(table.get("box_variant") == "wide_mestar")
    )
    boxes: Dict[int, Dict[str, object]] = {}
    for level, box in boxes_raw.items():
        boxes[int(level)] = {
            str(k): (list(v) if isinstance(v, (list, tuple)) else v)
            for k, v in box.items()
        }
    level = next(
        (int(l) for l, b in boxes.items() if b.get("label") == candidate), None
    )
    if level is None:
        # Round-2 W candidates share the frozen box of their level under a
        # W-suffixed label; resolve the level through the grid table. The
        # box cross-check itself happens in _validate_stage_b_grid_table.
        table_level = STAGE_B_GRID_TABLES.get(candidate, {}).get("level")
        if table_level is None or int(table_level) not in boxes:
            print(
                f"staged-fit: unknown Stage-B candidate '{candidate}'; expected one of "
                f"{sorted(STAGE_B_GRID_TABLES)}",
                file=sys.stderr,
            )
            return EXIT_CONFIG_ERROR
        level = int(table_level)
    _validate_stage_b_grid_table(candidate, level, boxes, mixing2006_dir)
    points = _stage_b_grid_points(candidate, grid_shape)

    tag = _stage_b_label_tag(candidate)
    config_ns = f"stageB_{tag}"
    kwargs = _common_generation_kwargs(args)
    tower_files = _materialize_tower_files(
        mixing2006_dir=mixing2006_dir,
        out_dir=out_dir,
        config_ns=config_ns,
        tower_min_mass=STAGE_B_TOWER_WINDOW_MIN_MASS_GEV,
        tower_max_mass=STAGE_B_TOWER_WINDOW_MAX_MASS_GEV,
        tower_level_count=int(args.tower_level_count),
        tower_shape=str(args.tower_shape),
        tower_shape_param=float(args.tower_shape_param),
        disable_tower=bool(args.disable_tower),
        tower_mass_law=str(table.get("tower_mass_law", "bw")),
    )
    axes_full = _stage_b_grid_axes(STAGE_B_GRID_TABLES[candidate])
    grid_path = out_dir / f"{STAGE_B_GRID_FILENAME}_{tag}.json"
    selection_path = out_dir / f"{STAGE_B_SELECTION_FILENAME}_{tag}.json"
    grid_payload: Dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact": "stage_b_grid",
        "stage": "stageB",
        "task_card": TASK_CARD_RELATIVE_PATH,
        "created_at_utc": _utc_now_iso(),
        "producer": {
            "python": platform.python_version(),
            "numpy": str(np.__version__),
            "na49_mixing_head": _git_head(mixing2006_dir),
            "repo_head_outer": _git_head(REPO_ROOT),
        },
        "candidate": candidate,
        "candidate_level": level,
        "box": boxes[level],
        "grid_shape": grid_shape,
        "grid_table": {
            "source": "staged_resonance_fit.STAGE_B_GRID_TABLES (pre-registered, card phase P4)",
            "axes_full": {k: list(v) for k, v in axes_full.items()},
            "coarse_subset_rule": (
                "per axis, keep {first, middle (index len//2), last} when the axis "
                "has more than 3 values; the coarse grid is a subset of the full grid"
            ),
            "point_count": len(points),
            "point_limit": int(STAGE_B_GRID_LIMITS[candidate]),
        },
        "pair_strength_key": pair_key,
        "pair_strength": (
            STAGE_B_PAIR_STRENGTH if bool(boxes[level].get("pair_capable")) else None
        ),
        "protocol": {
            "base_seed": int(args.base_seed),
            "replicate": 0,
            "replicate_note": (
                "the grid is pre-registered on the base replicate only; band "
                "replicates are a Stage-A Monte Carlo device"
            ),
            "component_seed_rule": "_component_seed(base_seed, raw_spec_key)",
            "tower_window": [
                STAGE_B_TOWER_WINDOW_MIN_MASS_GEV,
                STAGE_B_TOWER_WINDOW_MAX_MASS_GEV,
            ],
            "tower_files": {k: str(v) for k, v in (tower_files or {}).items()},
            "events_per_component": int(args.events_per_component),
            "max_decay_passes": int(args.res_max_decay_passes),
            "fragmentation_mode": str(args.res_fragmentation_mode),
            "weak_feed_policy": str(args.weak_feed_policy),
            "max_undecayed_fraction": float(args.max_undecayed_fraction),
            "res_workers": int(args.res_workers),
            "paper_window": {
                "xf_min": float(args.xf_min),
                "xf_max": float(args.xf_max),
                "pt_min": float(args.pt_min),
                "pt_max": float(args.pt_max),
            },
            "train_frac": float(args.train_frac),
            "train_test_split_seed": int(args.base_seed),
            "beam_momentum_gev_c": float(args.beam_momentum),
        },
        "species": list(args.species),
        "points": [],
        "selection": None,
    }

    fingerprint_cache: Dict[Tuple[str, str], Optional[str]] = {}
    table_cache: Dict[str, dict] = {}
    total_built = 0
    total_reused = 0
    for point in points:
        point_id = str(point["point_id"])
        ovr_full = _stage_b_point_overrides(candidate, point, mixing2006_dir)
        ns_suffix = f"_{point_id.split('-', 1)[1]}"
        point_entry: Dict[str, object] = {
            "point_id": point_id,
            "free": point["free"],
            "displacement": point["displacement"],
            "overrides": ovr_full,
            "selection_spec": _stage_b_selection_spec_string(
                f"{candidate}-sel", ovr_full
            ),
        }
        species_train: Dict[str, float] = {}
        species_payloads: Dict[str, dict] = {}
        point_built = 0
        point_reused = 0
        for sp_key in args.species:
            sp = resolve_species(sp_key)
            specs = _specs_for_species(mixing2006_dir, sp.pdg, tower_files=tower_files)
            spec_order = [str(s.label) for s in specs]
            plan = _stage_b_component_plan(specs, ovr_full, ns_suffix)
            cache_subdir = cache_dir / config_ns / "rep0"
            entries, arrays_by_label, n_reused, n_built = _build_component_caches(
                specs=specs,
                cache_subdir=cache_subdir,
                config_ns=config_ns,
                overrides=ovr_full,
                stage_b_component_plan=plan,
                replicate=0,
                fingerprint_cache=fingerprint_cache,
                **kwargs,
            )
            if sp_key not in table_cache:
                table_cache[sp_key] = _load_species_points(
                    _resolve_na49_table(sp_key, args.na49_data),
                    xf_min=float(args.xf_min),
                    xf_max=float(args.xf_max),
                    pt_min=float(args.pt_min),
                    pt_max=float(args.pt_max),
                )
            points_t = table_cache[sp_key]
            config_payload = _config_payload(
                args=args,
                overrides=ovr_full,
                config_ns=config_ns,
                extra={
                    "candidate": candidate,
                    "point_id": point_id,
                    "grid_shape": grid_shape,
                },
            )
            payload = _score_replicate(
                mode_tag="stageB",
                config_ns=config_ns,
                row_kind="grid_point",
                candidate_level=level,
                point_id=point_id,
                species_key=sp_key,
                table_path=_resolve_na49_table(sp_key, args.na49_data),
                points=points_t,
                entries=entries,
                arrays_by_label=arrays_by_label,
                spec_order=spec_order,
                base_seed=int(args.base_seed),
                band_seed_stride=int(args.band_seed_stride),
                replicate=0,
                train_frac=float(args.train_frac),
                beam_momentum_gev_c=float(args.beam_momentum),
                config_payload=config_payload,
                out_dir=out_dir,
            )
            train_wrrmse = float(
                dict(payload.get("train_metrics", {})).get(
                    "train_weighted_rel_rmse", float("nan")
                )
            )
            species_train[sp_key] = train_wrrmse
            openness = dict(payload.get("openness", {}))
            tail_only: List[dict] = []
            violations: List[dict] = []
            for comp_label, guard in cast(
                Dict[str, dict], openness.get("components", {})
            ).items():
                for flag in cast(Sequence[dict], guard.get("tail_only_flags", [])):
                    tail_only.append({"species": sp_key, "component": comp_label, **flag})
                for vio in cast(Sequence[dict], guard.get("violations", [])):
                    violations.append({"species": sp_key, "component": comp_label, **vio})
            species_payloads[sp_key] = {
                "train_metrics": payload.get("train_metrics"),
                "test_metrics": payload.get("test_metrics"),
                "openness_status": openness.get("status"),
                "results_file": str(
                    out_dir
                    / f"stageB_results_{config_ns}_{point_id}_{_species_fixture_tag(sp_key)}_rep0.json"
                ),
            }
            point_built += n_built
            point_reused += n_reused
        joint = sum(species_train.values()) if species_train else float("nan")
        point_entry["joint_train_wrrmse"] = joint
        point_entry["species_train_wrrmse"] = species_train
        point_entry["species"] = species_payloads
        point_entry["openness"] = {
            "guard_floor": OPENNESS_GUARD_FLOOR,
            "tail_only_flags": tail_only,
            "violations": violations,
            "flagged": bool(tail_only or violations),
        }
        point_entry["builds"] = {"built": point_built, "reused": point_reused}
        grid_payload["points"].append(point_entry)
        total_built += point_built
        total_reused += point_reused
        # Checkpoint the grid surface after every point (crash resilience:
        # completed points keep their caches and objectives).
        _write_json(grid_path, grid_payload)
        print(
            f"staged-fit: stageB {candidate} {point_id} "
            f"joint_train_wrrmse={joint:.6f} built={point_built} reused={point_reused}",
            flush=True,
        )

    valid_points = [
        p for p in grid_payload["points"]
        if isinstance(p, dict) and math.isfinite(float(p["joint_train_wrrmse"]))
    ]
    if not valid_points:
        raise RuntimeError(
            f"Stage-B grid for {candidate} produced no finite joint objective"
        )
    best = min(
        valid_points,
        key=lambda p: (
            float(p["joint_train_wrrmse"]),
            float(p["displacement"]),
            str(p["point_id"]),
        ),
    )
    grid_payload["totals"] = {"built": total_built, "reused": total_reused}
    grid_payload["selection"] = {
        "criterion": (
            "min joint train WRRMSE (sum over species); tie-break: smaller "
            "displacement (sum of free value minus the full-table axis minimum), "
            "then smaller point id"
        ),
        "selected_point_id": str(best["point_id"]),
        "joint_train_wrrmse": float(best["joint_train_wrrmse"]),
        "displacement": float(best["displacement"]),
        "selection_spec": str(best["selection_spec"]),
        "ab_handoff": {
            "config_b": str(best["selection_spec"]),
            "note": (
                "the selection spec carries the box windows, the pinned "
                "non-candidate levels, and the fitted free coordinates; "
                "--stage-b-candidate re-merges the same box window in ab mode"
            ),
        },
    }
    _write_json(grid_path, grid_payload)
    grid_digest = _write_sha256_sidecar(grid_path)
    selection_payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact": "stage_b_selection",
        "created_at_utc": _utc_now_iso(),
        "candidate": candidate,
        "candidate_level": level,
        "grid_shape": grid_shape,
        "selected_point_id": str(best["point_id"]),
        "free": best["free"],
        "joint_train_wrrmse": float(best["joint_train_wrrmse"]),
        "displacement": float(best["displacement"]),
        "overrides": best["overrides"],
        "selection_spec": str(best["selection_spec"]),
        "grid_surface": f"{grid_path.name} (sha256 {grid_digest})",
    }
    _write_json(selection_path, selection_payload)
    selection_digest = _write_sha256_sidecar(selection_path)
    print(
        f"staged-fit: stageB {candidate} selected {str(best['point_id'])} "
        f"(joint_train_wrrmse={float(best['joint_train_wrrmse']):.6f}); "
        f"wrote {grid_path.name} (sha256 {grid_digest}) and "
        f"{selection_path.name} (sha256 {selection_digest})",
        flush=True,
    )
    return EXIT_OK


def _run_ab(args) -> int:
    """Run the A/B harness: two arms, one shared generation protocol.

    Per-arm N* regge-tower window: arm A keeps the CLI tower window so the
    frozen Stage-A baseline is exact. Arm B materializes its tower from its
    own parsed override set: the Stage-B selection spec carries
    ``nstar_eff_tower_max_mass`` (the grid-point window), and a plain spec
    without the key falls back to the CLI value.
    """
    if not args.config_a or not args.config_b:
        print("staged-fit: --mode ab requires --config-a and --config-b", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    mixing2006_dir = Path(args.mixing2006_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (RUNS_ROOT / args.run_description)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (out_dir / "res_cache")
    replicates = _parse_replicate_spec(args.replicates, int(args.band_seeds))
    kwargs = _common_generation_kwargs(args)

    boxes_payload = _stage_b_boxes_payload(mixing2006_dir)
    boxes: Dict[int, Dict[str, object]] = {}
    if boxes_payload:
        boxes = {
            int(k): {str(kk): vv for kk, vv in v.items()}
            for k, v in dict(boxes_payload.get("boxes", {})).items()
        }

    arm_a_name, arm_a_overrides = _parse_arm_spec(args.config_a, "stageA")
    arm_b_name, arm_b_overrides = _parse_arm_spec(args.config_b, "candidate")
    arm_a = {"name": arm_a_name, "overrides": dict(arm_a_overrides)}
    arm_b = {"name": arm_b_name, "overrides": dict(arm_b_overrides)}

    if args.stage_b_candidate:
        if not boxes:
            print(
                "staged-fit: Stage-B boxes unavailable; cannot apply "
                f"--stage-b-candidate {args.stage_b_candidate}",
                file=sys.stderr,
            )
            return EXIT_CONFIG_ERROR
        arm_b_overrides, level, box = _apply_stage_b_candidate(
            arm_b_overrides, args.stage_b_candidate, boxes
        )
        arm_b["overrides"] = dict(arm_b_overrides)
        arm_b["candidate_level"] = int(level)
        arm_b["row_kind"] = "candidate"
        arm_b["box"] = box
        # The box window is already merged into arm_b_overrides by
        # _apply_stage_b_candidate (nstar_eff_tower_max_mass), so the per-arm
        # resolution below widens arm B's tower while arm A keeps the CLI
        # window. The shared args.tower_max_mass must stay unmutated here:
        # touching it would leak the box window into arm A's materialization
        # and break the frozen Stage-A baseline.
    if args.b0_ablation:
        arm_b["overrides"] = _apply_b0_ablation(arm_b_overrides)
        arm_b["row_kind"] = "b0_ablation"
        arm_b["b0_ablation"] = True
    arm_b.setdefault("row_kind", "candidate")
    arm_b.setdefault("candidate_level", None)

    shared_protocol = {
        "base_seed": int(args.base_seed),
        "band_seeds": int(args.band_seeds),
        "band_seed_stride": int(args.band_seed_stride),
        "events_per_component": int(args.events_per_component),
        "max_decay_passes": int(args.res_max_decay_passes),
        "fragmentation_mode": str(args.res_fragmentation_mode),
        "weak_feed_policy": str(args.weak_feed_policy),
        "paper_window": {
            "xf_min": float(args.xf_min),
            "xf_max": float(args.xf_max),
            "pt_min": float(args.pt_min),
            "pt_max": float(args.pt_max),
        },
        "train_frac": float(args.train_frac),
        "train_test_split_seed": int(args.base_seed),
        "fit_convention": DEFAULT_FIT_CONVENTION,
        "beam_momentum_gev_c": float(args.beam_momentum),
    }

    for arm, config_ns in ((arm_a, arm_a["name"]), (arm_b, arm_b["name"])):
        overrides = dict(arm["overrides"])
        # Per-arm N* regge-tower window. Arm A keeps the CLI window so the
        # frozen Stage-A baseline is exact. Arm B widens via its own spec:
        # the parsed arm-B override set carries the Stage-B grid-point
        # window (``nstar_eff_tower_max_mass``); a plain spec without the
        # key falls back to the CLI value.
        if config_ns == arm_b["name"]:
            tower_kwargs = {
                "tower_min_mass": float(args.tower_min_mass),
                "tower_max_mass": float(
                    arm_b_overrides.get("nstar_eff_tower_max_mass", args.tower_max_mass)
                ),
                "tower_mass_law": (
                    "regge"
                    if float(arm_b_overrides.get("tower_mass_law_flag", 0.0)) >= 0.5
                    else "bw"
                ),
                "tower_pl_slope": float(
                    arm_b_overrides.get("tower_pl_slope", 0.0)
                ),
            }
        else:
            tower_kwargs = {
                "tower_min_mass": float(args.tower_min_mass),
                "tower_max_mass": float(args.tower_max_mass),
                "tower_mass_law": "bw",
                "tower_pl_slope": 0.0,
            }
        tower_files = _materialize_tower_files(
            mixing2006_dir=mixing2006_dir,
            out_dir=out_dir,
            config_ns=config_ns,
            tower_level_count=int(args.tower_level_count),
            tower_shape=str(args.tower_shape),
            tower_shape_param=float(args.tower_shape_param),
            disable_tower=bool(args.disable_tower),
            **tower_kwargs,
        )
        config_payload = _config_payload(
            args=args,
            overrides=overrides,
            config_ns=config_ns,
            extra={
                "cache_dir": str(cache_dir),
                "replicates": replicates,
                "tower_files": {k: str(v) for k, v in (tower_files or {}).items()},
                "row_kind": arm.get("row_kind"),
                "candidate_level": arm.get("candidate_level"),
            },
        )
        # Record the per-arm tower window that was actually materialized,
        # so the audit matches the generation config for both arms.
        config_payload["tower"]["max_mass_gev"] = tower_kwargs["tower_max_mass"]
        fingerprint_cache: Dict[Tuple[str, str], Optional[str]] = {}
        for sp_key in args.species:
            sp = resolve_species(sp_key)
            specs = _specs_for_species(mixing2006_dir, sp.pdg, tower_files=tower_files)
            spec_order = [str(s.label) for s in specs]
            for k in replicates:
                cache_subdir = cache_dir / config_ns / f"rep{k}"
                entries, arrays_by_label, n_reused, n_built = _build_component_caches(
                    specs=specs,
                    cache_subdir=cache_subdir,
                    config_ns=config_ns,
                    overrides=overrides,
                    replicate=k,
                    fingerprint_cache=fingerprint_cache,
                    **kwargs,
                )
                table_path = _resolve_na49_table(sp_key, args.na49_data)
                points = _load_species_points(
                    table_path,
                    xf_min=float(args.xf_min),
                    xf_max=float(args.xf_max),
                    pt_min=float(args.pt_min),
                    pt_max=float(args.pt_max),
                )
                _score_replicate(
                    mode_tag="ab",
                    config_ns=config_ns,
                    row_kind=str(arm.get("row_kind", "candidate")),
                    candidate_level=arm.get("candidate_level"),
                    species_key=sp_key,
                    table_path=table_path,
                    points=points,
                    entries=entries,
                    arrays_by_label=arrays_by_label,
                    spec_order=spec_order,
                    base_seed=int(args.base_seed),
                    band_seed_stride=int(args.band_seed_stride),
                    replicate=k,
                    train_frac=float(args.train_frac),
                    beam_momentum_gev_c=float(args.beam_momentum),
                    config_payload=config_payload,
                    out_dir=out_dir,
                )
                print(
                    f"staged-fit: ab {config_ns} {sp.name} rep{k} "
                    f"built={n_built} reused={n_reused}",
                    flush=True,
                )
    _finalize_ab_comparison(
        out_dir=out_dir,
        mixing2006_dir=mixing2006_dir,
        species_keys=list(args.species),
        replicates=replicates,
        arm_a=arm_a,
        arm_b=arm_b,
        shared_protocol=shared_protocol,
    )
    print(
        f"staged-fit: wrote {out_dir / AB_COMPARISON_FILENAME}",
        flush=True,
    )
    return EXIT_OK


# --- CLI --------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Staged resonance-decay fit driver (card TASK-2026-08-31, phase P3)."
    )
    parser.add_argument(
        "--mode",
        choices=("stage0", "stageA", "ab", "stageB"),
        required=True,
        help=(
            "stage0: golden reference refit; stageA: production baseline; "
            "ab: A/B harness; stageB: pre-registered Stage-B grid fit "
            "(requires --stage-b-candidate)."
        ),
    )
    parser.add_argument(
        "--stage-b-grid",
        choices=STAGE_B_GRID_SHAPES,
        default="full",
        help="Stage-B grid shape (stageB mode): full (pre-registered table) or coarse.",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        default=["pi-"],
        help="Species keys (default: pi-).",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory. If empty, uses 00runs/<run-description>.",
    )
    parser.add_argument("--run-description", default="staged_fit")
    parser.add_argument("--mixing2006-dir", default=str(REPO_ROOT))
    parser.add_argument(
        "--na49-data",
        default="",
        help=(
            "Optional shared NA49 table path (xF pT value error). By default the "
            "per-species spshadrons.web.cern.ch table is used."
        ),
    )
    # Pre-registered seed protocol.
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--band-seeds", type=int, default=DEFAULT_BAND_SEEDS)
    parser.add_argument("--band-seed-stride", type=int, default=DEFAULT_BAND_SEED_STRIDE)
    parser.add_argument(
        "--band-sigma-multiplier",
        type=float,
        default=DEFAULT_BAND_SIGMA_MULTIPLIER,
        help="Recorded band multiplier of the golden fixture convention (diagnostic only).",
    )
    parser.add_argument(
        "--replicates",
        default="",
        help='Replicate indices to run, e.g. "0,3,5" or "0-7". Empty: all 0..band-seeds.',
    )
    # Generation.
    parser.add_argument(
        "--events-per-component",
        type=int,
        default=DEFAULT_EVENTS_PER_COMPONENT,
        help="Events per component (production statistics: 120000).",
    )
    parser.add_argument("--res-max-decay-passes", type=int, default=8)
    parser.add_argument(
        "--res-fragmentation-mode",
        choices=("projectile", "target", "both"),
        default="both",
    )
    parser.add_argument("--res-workers", type=int, default=4)
    parser.add_argument("--beam-pdg", type=int, default=2212,
                        help="Projectile PDG (2112 = neutron).")
    parser.add_argument("--target-pdg", type=int, default=2212,
                        help="Target nucleon PDG (p+A uses --pa-a instead).")
    parser.add_argument("--pa-a", type=int, default=0,
                        help="Target mass number > 0 selects p+A generation.")
    parser.add_argument("--pa-z", type=int, default=6)
    parser.add_argument("--pa-profile", default="woods_saxon")
    parser.add_argument("--pa-radius-fm", type=float, default=2.47)
    parser.add_argument("--pa-diffuseness-fm", type=float, default=0.44)
    parser.add_argument("--target-wound-range", default="",
                        help="Wounded-nucleon slice 'lo:hi' for p+A.")
    parser.add_argument(
        "--weak-feed-policy",
        choices=("inclusive_weak", "primary_only"),
        default="inclusive_weak",
        help="Weak feed-down policy (golden fixture and production caches use inclusive_weak).",
    )
    parser.add_argument(
        "--max-undecayed-fraction",
        type=float,
        default=DEFAULT_MAX_UNDECAYED_FRACTION,
        help="Fail a component when the undecayed-primary fraction exceeds this value.",
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Shared component cache directory. If empty, uses <out-dir>/res_cache.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild component caches even when the configuration matches.",
    )
    # Regge tower.
    parser.add_argument(
        "--disable-tower",
        action="store_true",
        help="Disable the effective high-mass N* tower components.",
    )
    parser.add_argument("--tower-min-mass", type=float, default=DEFAULT_TOWER_MIN_MASS_GEV)
    parser.add_argument("--tower-max-mass", type=float, default=DEFAULT_TOWER_MAX_MASS_GEV)
    parser.add_argument("--tower-level-count", type=int, default=DEFAULT_TOWER_LEVEL_COUNT)
    parser.add_argument("--tower-shape", choices=("linear", "regge_m2", "hagedorn"), default=DEFAULT_TOWER_SHAPE)
    parser.add_argument("--tower-shape-param", type=float, default=DEFAULT_TOWER_SHAPE_PARAM)
    # Fit.
    parser.add_argument("--beam-momentum", type=float, default=DEFAULT_BEAM_MOMENTUM_GEV_C)
    parser.add_argument("--xf-min", type=float, default=PAPER_XF_MIN)
    parser.add_argument("--xf-max", type=float, default=PAPER_XF_MAX)
    parser.add_argument("--pt-min", type=float, default=PAPER_PT_MIN)
    parser.add_argument("--pt-max", type=float, default=PAPER_PT_MAX)
    parser.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC)
    # Finalize.
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Assemble the Stage-A baseline artifact after building (stageA mode).",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Skip generation and assemble the artifact from existing results (stageA mode).",
    )
    # A/B harness.
    parser.add_argument(
        "--config-a",
        default="",
        help='Arm A spec: "NAME" or "NAME:KEY=VALUE,KEY2=VALUE2" (ab mode).',
    )
    parser.add_argument(
        "--config-b",
        default="",
        help='Arm B spec: "NAME" or "NAME:KEY=VALUE,KEY2=VALUE2" (ab mode).',
    )
    parser.add_argument(
        "--stage-b-candidate",
        choices=STAGE_B_CANDIDATE_LABELS,
        default="",
        help=(
            "Stage-B candidate box: required for stageB mode (the grid "
            "candidate); in ab mode it is applied to arm B."
        ),
    )
    parser.add_argument(
        "--b0-ablation",
        action="store_true",
        help="Force the M* p pbar pair family strength to zero on arm B (ab mode).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.pa_nucleus_dict = None
    args.pa_wound_range = None
    if int(getattr(args, "pa_a", 0)) > 0:
        args.pa_nucleus_dict = {
            "name": f"pa{int(args.pa_a)}",
            "A": int(args.pa_a),
            "Z": int(args.pa_z),
            "profile": str(args.pa_profile),
            "radius_fm": float(args.pa_radius_fm),
            "diffuseness_fm": float(args.pa_diffuseness_fm),
            "quad_alpha": 0.0,
            "sigma_nn_inel_mb": 30.0,
        }
        if str(args.target_wound_range).strip():
            lo_s, hi_s = str(args.target_wound_range).split(":")
            args.pa_wound_range = (int(lo_s), int(hi_s))
        if int(getattr(args, "target_pdg", 2212)) != 2212:
            print(
                "staged-fit: --pa-a selects the p+A system; drop --target-pdg",
                file=sys.stderr,
            )
            return EXIT_CONFIG_ERROR
    for sp_key in args.species:
        try:
            resolve_species(sp_key)
        except KeyError as exc:
            print(f"staged-fit: {exc}", file=sys.stderr)
            return EXIT_CONFIG_ERROR
    if args.base_seed <= 0:
        print("staged-fit: --base-seed must be positive", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    if args.band_seeds < 0:
        print("staged-fit: --band-seeds must be >= 0", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    try:
        if args.mode == "stage0":
            return _run_stage0(args)
        if args.mode == "stageA":
            return _run_stage_a(args)
        if args.mode == "stageB":
            return _run_stage_b(args)
        return _run_ab(args)
    except (RuntimeError, ValueError) as exc:
        print(f"staged-fit: failed: {exc}", file=sys.stderr)
        return EXIT_RUNTIME_FAILURE


if __name__ == "__main__":
    sys.exit(main())
