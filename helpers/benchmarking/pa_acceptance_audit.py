#!/usr/bin/env python3
"""p+A acceptance-bias audit (card TASK-20260905-physics-review, M5).

Runs p+Pb208 at the bundled staged configuration through the actual
staged construction path (the ``baryon_nucleus`` stage of
``na49_mixing.staged_evaluation``: ``PAEventBuilder`` with
``production_mode="regge"``, the nucleus and the Regge, continuum, and
simple parameter blocks taken verbatim from the configuration, and
``glauber_dr_fm`` / ``max_event_attempts`` from ``config.pA``) and
measures how the resampling of failed kinematic constructions moves the
accepted wound-count (nu) and impact-parameter (b) distributions
against the original Glauber draws.

Every accepted event records its generation attempt count and the last
failure reason (``event_builder_pA`` bookkeeping); the builder's
instance counters provide the per-exception-class failure counts, the
total attempt / acceptance / rejection tallies, and the full list of
Glauber draws ``(b, nu, accepted)``. The full draw list is the original
Glauber distribution at the same geometry, sampled without kinematic
rejection, with the same draw count as the generation itself. The audit
compares the accepted P(nu) against that Glauber P(nu), bin by bin with
a Wilson 95% score interval, and reports the generation acceptance
versus nu, versus b, and per joint (nu, b) bin.

Pre-registered decision rule (recorded in the run's
``pre_registration.json`` before the run): if the accepted P(nu) lies
inside the bin-by-bin band of the Glauber P(nu) for every nu bin with
at least 30 accepted events, the monotonic projectile-mass rule stands
as the named cumulative-excitation ansatz, with the review's caveat
(a collision could transfer energy into the newly created target
system and leave a lower-mass outgoing projectile source) recorded as
the known limitation. If the band is violated in any qualifying bin,
the milestone escalates within itself to the de-excitation option
(allow ``M_proj,new < M_proj,current`` subject to exact kinematics,
exact QN bookkeeping, and the source-mass minima) and reruns the audit.

CI convention. For a bin with x occurrences out of n draws the Wilson
score interval at level z is

    center = (p_hat + z^2 / (2 n)) / (1 + z^2 / n)
    half   = z * sqrt(p_hat (1 - p_hat) / n + z^2 / (4 n^2)) / (1 + z^2 / n)
    band   = (center - half, center + half),

with p_hat = x / n and z = 1.959964 (two-sided 95%). The band is
applied to the reference proportion (x = the bin's Glauber draw count,
n = the total Glauber draw count), and the accepted proportion
p_acc = accepted_nu / N_accepted is inside when it falls in the band.

Usage:

    python helpers/benchmarking/pa_acceptance_audit.py \
        [--config CONFIG] [--n-accepted N] [--seed-offset OFFSET] \
        [--out-dir DIR] [--probe N]

The audit seed is ``config.seed + seed-offset``; the default offset
20000 is distinct from the M4 fit offset 10000 (the fit substream is
``config.seed + 10000``). ``--probe N`` generates N accepted events and
prints the per-event wall time and the acceptance rate without writing
the report, so the full-run statistics can be chosen before the
pre-registered run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_THIS = Path(__file__).resolve()
RUN_NAME = "pa_acceptance_audit_2026-09-05"
DEFAULT_N_ACCEPTED = 50_000
DEFAULT_SEED_OFFSET = 20_000
DEFAULT_MIN_QUALIFYING = 30
WILSON_Z_95 = 1.959963984540054

DECISION_RULE = (
    "If the accepted P(nu) lies inside the bin-by-bin Wilson 95% band "
    "of the Glauber P(nu) for every nu bin with at least "
    f"{DEFAULT_MIN_QUALIFYING} accepted events, the monotonic "
    "projectile-mass rule stands as the named cumulative-excitation "
    "ansatz (known limitation: a collision could transfer energy into "
    "the newly created target system and leave a lower-mass outgoing "
    "projectile source). If the band is violated in any qualifying "
    "bin, the milestone escalates to the de-excitation option (allow "
    "M_proj,new < M_proj,current subject to exact kinematics, exact QN "
    "bookkeeping, and the source-mass minima) and reruns the audit."
)


# ---------------------------------------------------------------------------
# Layout bootstrap (identical file in the monorepo and the vendored tree)
# ---------------------------------------------------------------------------

def _mixing2006_root() -> Path:
    """Locate the ``dev/mixing2006`` package root from either tree.

    Vendored mirror: this file lives inside ``dev/mixing2006``, so the
    first parent directory carrying the ``na49_mixing`` package and the
    ``config`` directory is the root. Monorepo: the file lives under
    ``<root>/helpers/benchmarking``, so the root is
    ``<root>/dev/mixing2006``.
    """
    for parent in _THIS.parents:
        if (parent / "na49_mixing").is_dir() and (parent / "config").is_dir():
            return parent
    for parent in _THIS.parents:
        candidate = parent / "dev" / "mixing2006"
        if (candidate / "na49_mixing").is_dir() and (candidate / "config").is_dir():
            return candidate
    raise RuntimeError(
        "cannot locate the mixing2006 package root relative to "
        f"{_THIS}; run the helper from a monorepo or vendored checkout"
    )


def _bootstrap() -> None:
    mix = _mixing2006_root()
    if str(mix) not in sys.path:
        sys.path.insert(0, str(mix))


def _default_out_dir() -> Path:
    """Default report directory for this layout.

    Monorepo: ``<root>/dev/00runs/<RUN_NAME>``. Standalone (the file
    inside ``dev/mixing2006``): ``<mixing2006>/00runs/<RUN_NAME>``.
    """
    mix = _mixing2006_root()
    if mix in _THIS.parents:
        return mix / "00runs" / RUN_NAME
    return mix.parents[1] / "dev" / "00runs" / RUN_NAME


# ---------------------------------------------------------------------------
# Configuration and builder (mirrors the staged baryon_nucleus stage)
# ---------------------------------------------------------------------------

def load_bundled_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the staged-evaluation configuration (bundled by default)."""
    if path is None:
        path = _mixing2006_root() / "config" / "staged_excitation.json"
    return json.loads(Path(path).read_text())


def configuration_hash(config: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON form (staged_evaluation convention)."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _json_default(value: Any) -> Any:
    """JSON encoder fallback for the NumPy scalars the statistics carry."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {value!r} ({type(value).__name__})")


def build_pa_builder(config: dict[str, Any], seed: int):
    """Build the p+A event builder exactly as the staged pipeline does.

    ``production_mode`` is ``"regge"`` (the staged baryon_nucleus stage
    hard-codes it), the parameter blocks are taken verbatim from the
    configuration (the audit uses the bundled configuration unchanged:
    it measures acceptance bias, not production shape), and the nucleus,
    ``glauber_dr_fm``, and ``max_event_attempts`` follow ``config.pA``.
    """
    _bootstrap()
    from na49_mixing.continuum_channels import ContinuumParameters
    from na49_mixing.event_builder_pA import PAEventBuilder, PAExcitationConfig
    from na49_mixing.exchange_model import (
        ReggeExcitationParameters,
        SimpleExcitationParameters,
    )
    from na49_mixing.glauber import NucleusSpec

    pa_cfg = dict(config.get("pA", {}))
    return PAEventBuilder(
        PAExcitationConfig(
            beam_momentum=float(config.get("beam_momentum", 158.0)),
            production_mode="regge",
            simple=SimpleExcitationParameters(**(config.get("simple") or {})),
            regge=ReggeExcitationParameters(**(config.get("regge") or {})),
            continuum=ContinuumParameters(**(config.get("continuum") or {})),
            nucleus=NucleusSpec(**dict(config.get("nucleus") or {})),
            glauber_dr_fm=float(pa_cfg.get("glauber_dr_fm", 0.05)),
            max_event_attempts=int(pa_cfg.get("max_event_attempts", 100)),
        ),
        np.random.default_rng(int(seed)),
    )


# ---------------------------------------------------------------------------
# Acceptance sampling
# ---------------------------------------------------------------------------

def run_acceptance_sample(
    builder: Any,
    n_accepted: int,
    max_rejected: int | None = None,
) -> list:
    """Generate ``n_accepted`` accepted events and return them.

    Event ids run 0..n-1 over the accepted events (the staged runner's
    convention for the baryon_nucleus stage); a ``RuntimeError`` from an
    exhausted event is counted through the builder's
    ``generation_rejected`` counter and the sample continues until
    ``max_rejected`` (default: ``n_accepted``) exhausted events have
    accumulated, at which point the sample aborts.
    """
    n_accepted = int(n_accepted)
    if n_accepted < 1:
        raise ValueError("n_accepted must be positive")
    if max_rejected is None:
        max_rejected = n_accepted
    events = []
    event_id = 0
    while len(events) < n_accepted:
        try:
            events.append(builder.generate(event_id))
        except RuntimeError:
            if builder.generation_rejected > int(max_rejected):
                raise RuntimeError(
                    f"p+A generation exhausted max_event_attempts on "
                    f"{builder.generation_rejected} events; abandoning "
                    "the audit sample"
                ) from None
        event_id += 1
    return events


# ---------------------------------------------------------------------------
# Statistics (pure, hermetic: no event generation inside)
# ---------------------------------------------------------------------------

def wilson_interval(
    x: int, n: int, z: float = WILSON_Z_95
) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion.

    ``x`` successes out of ``n`` trials at significance level ``z``
    (default: two-sided 95%). For ``n = 0`` the vacuous interval
    ``(0.0, 1.0)`` is returned.
    """
    if n <= 0:
        return (0.0, 1.0)
    p_hat = float(x) / float(n)
    z2 = z * z
    denom = 1.0 + z2 / float(n)
    center = (p_hat + z2 / (2.0 * float(n))) / denom
    radicand = max(
        p_hat * (1.0 - p_hat) / float(n) + z2 / (4.0 * float(n) * float(n)),
        0.0,
    )
    half = z * math.sqrt(radicand) / denom
    return (max(center - half, 0.0), min(center + half, 1.0))


def b_grid_index(b_fm: float, dr_fm: float) -> int:
    """Glauber b-grid index of a draw: draws land exactly on ``i*dr_fm``."""
    if not (math.isfinite(float(b_fm)) and float(b_fm) >= 0.0):
        raise ValueError(f"b must be finite and non-negative, got {b_fm!r}")
    return int(round(float(b_fm) / float(dr_fm)))


def wound_comparison(
    accepted_counts: list[int],
    reference_counts: list[int],
    min_accepted: int = DEFAULT_MIN_QUALIFYING,
    z: float = WILSON_Z_95,
) -> dict[str, Any]:
    """Bin-by-bin accepted-versus-Glauber P(nu) comparison with the
    pre-registered decision rule.

    ``accepted_counts[i]`` and ``reference_counts[i]`` count nu = i + 1
    among the accepted events and among the full Glauber draw set (the
    same draws without kinematic rejection). For each nu bin the
    reference proportion ``p_ref = n_ref,nu / n_ref,total`` carries the
    Wilson interval with ``x = n_ref,nu`` and ``n = n_ref,total``; the
    accepted proportion ``p_acc = n_acc,nu / n_acc,total`` is inside
    when it falls in that band. The per-bin generation acceptance
    ``q = n_acc,nu / n_ref,nu`` carries its own Wilson interval
    (``x = n_acc,nu``, ``n = n_ref,nu``). A nu bin qualifies when it
    holds at least ``min_accepted`` accepted events;
    ``decision_passed`` is true exactly when every qualifying bin is
    inside its band (a bin with zero Glauber draws cannot qualify).
    """
    if len(accepted_counts) != len(reference_counts):
        raise ValueError("accepted and reference counts must have equal length")
    min_accepted = int(min_accepted)
    n_ref_total = int(sum(reference_counts))
    n_acc_total = int(sum(accepted_counts))
    entries: list[dict[str, Any]] = []
    for i, (n_acc, n_ref) in enumerate(zip(accepted_counts, reference_counts)):
        if n_acc <= 0 and n_ref <= 0:
            continue
        nu = i + 1
        p_ref = n_ref / n_ref_total if n_ref_total > 0 else 0.0
        p_acc = n_acc / n_acc_total if n_acc_total > 0 else 0.0
        band_low, band_high = wilson_interval(n_ref, n_ref_total, z)
        acc_low, acc_high = wilson_interval(n_acc, n_ref, z)
        entries.append({
            "nu": nu,
            "reference_draws": int(n_ref),
            "accepted": int(n_acc),
            "p_ref": float(p_ref),
            "band_low": float(band_low),
            "band_high": float(band_high),
            "p_acc": float(p_acc),
            "inside": bool(band_low <= p_acc <= band_high),
            "qualifies": bool(n_acc >= min_accepted),
            "acceptance": float(n_acc / n_ref) if n_ref > 0 else None,
            "acceptance_ci_low": float(acc_low) if n_ref > 0 else None,
            "acceptance_ci_high": float(acc_high) if n_ref > 0 else None,
        })
    qualifying = [entry for entry in entries if entry["qualifies"]]
    violated = [entry for entry in qualifying if not entry["inside"]]
    return {
        "n_reference_total": n_ref_total,
        "n_accepted_total": n_acc_total,
        "ci_convention": (
            "Wilson score interval, two-sided 95% (z = "
            f"{WILSON_Z_95:.6f}), band on the reference proportion "
            "p_ref = n_ref,nu / n_ref_total, accepted proportion "
            "p_acc = n_acc,nu / n_acc_total judged against that band"
        ),
        "min_accepted_for_qualification": min_accepted,
        "decision_rule": DECISION_RULE,
        "entries": entries,
        "n_qualifying_bins": len(qualifying),
        "violating_bins": [entry["nu"] for entry in violated],
        "decision_passed": len(violated) == 0,
        "verdict": (
            "monotonic projectile-mass rule retained: every qualifying "
            "nu bin is inside the Glauber P(nu) band"
            if not violated
            else "band violated; escalate to the de-excitation option "
                 "and rerun the audit"
        ),
    }


def summarize_acceptance(
    builder: Any,
    events: list,
    dr_fm: float | None = None,
    min_qualifying: int = DEFAULT_MIN_QUALIFYING,
    min_shift_draws: int = 50,
    coarse_dr_fm: float = 0.5,
) -> dict[str, Any]:
    """Aggregate the builder's bookkeeping and the accepted events.

    Reads the builder's instance counters and the per-event metadata
    only (no randomness is consumed), so the result is deterministic for
    a given generated sample. Returns the audit payload: aggregate
    acceptance and failure counts, the per-event attempt distribution,
    the per-nu table with the pre-registered band and decision, the
    per-b acceptance tables (native Glauber grid and the coarser
    ``coarse_dr_fm`` rebin), the centrality-shift metrics of the
    accepted events against the full Glauber draws, and the maximum
    conservation residual over the accepted events.
    """
    if dr_fm is None:
        dr_fm = float(builder.config.glauber_dr_fm)
    n_acc = len(events)
    if n_acc < 1:
        raise ValueError("summarize_acceptance requires at least one event")
    n_draws = int(builder.generation_attempts)
    draws = list(builder.glauber_draws)
    if len(draws) != n_draws:
        raise ValueError(
            f"builder draw list has {len(draws)} entries but "
            f"generation_attempts = {n_draws}"
        )
    n_acc_counter = int(builder.generation_accepted)
    n_accepted_marks = sum(1 for item in draws if item[2])
    if n_acc != n_acc_counter or n_acc != n_accepted_marks:
        raise ValueError(
            f"accepted-count mismatch: events={n_acc}, "
            f"counter={n_acc_counter}, draw marks={n_accepted_marks}"
        )

    nu_of_draw = [int(item[1]) for item in draws]
    nu_of_event = [int(event.metadata["target_wounds"]) for event in events]
    nu_max = max(max(nu_of_draw), max(nu_of_event))
    ref_nu = list(np.bincount(nu_of_draw, minlength=nu_max + 1)[1:])
    acc_nu = list(np.bincount(nu_of_event, minlength=nu_max + 1)[1:])

    b_index_of_draw = [b_grid_index(item[0], dr_fm) for item in draws]
    b_index_of_event = [
        b_grid_index(float(event.metadata["impact_parameter_fm"]), dr_fm)
        for event in events
    ]
    b_max = max(max(b_index_of_draw), max(b_index_of_event)) + 1
    draws_b = np.bincount(b_index_of_draw, minlength=b_max)
    acc_b = np.bincount(b_index_of_event, minlength=b_max)
    b_table = []
    for i in range(b_max):
        n_ref = int(draws_b[i])
        n_acc_bin = int(acc_b[i])
        if n_ref <= 0 and n_acc_bin <= 0:
            continue
        p_ref = n_ref / n_draws
        p_acc = n_acc_bin / n_acc
        rel_shift = (p_acc - p_ref) / p_ref if p_ref > 0.0 else None
        b_table.append({
            "b_lo_fm": float(i * dr_fm),
            "b_hi_fm": float((i + 1) * dr_fm),
            "reference_draws": n_ref,
            "accepted": n_acc_bin,
            "acceptance": float(n_acc_bin / n_ref) if n_ref > 0 else None,
            "p_ref": float(p_ref),
            "p_acc": float(p_acc),
            "relative_shift": float(rel_shift) if rel_shift is not None else None,
        })

    # Joint (nu, b) acceptance table: accepted/attempted per (nu, b-bin)
    # cell, restricted to the populated cells (the card's "acceptance
    # per (nu, b) bin"). The populated joint cells are the marginal
    # tables' refinement; the cells with at least one failed attempt are
    # the only ones that distinguish the two.
    joint: dict[tuple[int, int], list[int]] = {}
    for (b_i, w), a in zip(
        zip(b_index_of_draw, nu_of_draw), (item[2] for item in draws)
    ):
        cell = joint.setdefault((int(w), int(b_i)), [0, 0])
        cell[0] += 1
        if a:
            cell[1] += 1
    joint_table = [
        {
            "nu": int(w),
            "b_lo_fm": float(i * dr_fm),
            "b_hi_fm": float((i + 1) * dr_fm),
            "attempts": int(attempts),
            "accepted": int(accepted),
            "acceptance": float(accepted / attempts),
        }
        for (w, i), (attempts, accepted) in sorted(joint.items())
    ]
    failed_cells = [
        {
            "nu": int(w),
            "b_lo_fm": float(i * dr_fm),
            "b_hi_fm": float((i + 1) * dr_fm),
            "attempts": int(attempts),
            "accepted": int(accepted),
            "acceptance": float(accepted / attempts),
        }
        for (w, i), (attempts, accepted) in sorted(joint.items())
        if accepted < attempts
    ]

    # Coarse rebin of the native b table (0.5 fm by default).
    coarse_b = []
    if coarse_dr_fm > dr_fm:
        groups = int(math.ceil(coarse_dr_fm / dr_fm))
        for start in range(0, len(b_table), groups):
            chunk = b_table[start:start + groups]
            if not chunk:
                continue
            n_ref = sum(row["reference_draws"] for row in chunk)
            n_acc_bin = sum(row["accepted"] for row in chunk)
            p_ref = n_ref / n_draws
            p_acc = n_acc_bin / n_acc
            coarse_b.append({
                "b_lo_fm": chunk[0]["b_lo_fm"],
                "b_hi_fm": chunk[-1]["b_hi_fm"],
                "reference_draws": n_ref,
                "accepted": n_acc_bin,
                "acceptance": float(n_acc_bin / n_ref) if n_ref > 0 else None,
                "p_ref": float(p_ref),
                "p_acc": float(p_acc),
                "relative_shift": (
                    float((p_acc - p_ref) / p_ref) if p_ref > 0.0 else None
                ),
            })

    # Centrality-shift metrics (requirement 4 of the card): how much the
    # resampling of failed kinematic constructions moves the centrality
    # distribution. Two complementary numbers, both computed on the
    # accepted events against the full Glauber draws:
    #   * the per-b-bin relative shift (p_acc - p_ref) / p_ref, quoted
    #     as max |shift| and RMS over the b bins with at least
    #     min_shift_draws Glauber draws;
    #   * the two-sample Kolmogorov distance D = max |CDF_acc - CDF_ref|
    #     on the b distribution over the native grid.
    shifts = [
        row["relative_shift"]
        for row in b_table
        if row["relative_shift"] is not None
        and row["reference_draws"] >= int(min_shift_draws)
    ]
    cdf_ref = np.cumsum(draws_b) / float(n_draws)
    cdf_acc = np.cumsum(acc_b) / float(n_acc)
    kolmogorov_d = float(np.max(np.abs(cdf_acc - cdf_ref)))
    centrality_shift = {
        "metric": (
            "per-b-bin relative shift (p_acc - p_ref) / p_ref with "
            f"min_shift_draws = {int(min_shift_draws)} Glauber draws per "
            "bin, plus the two-sample Kolmogorov distance on the b "
            "distribution"
        ),
        "n_bins_in_shift_metric": len(shifts),
        "max_abs_relative_shift": float(max(abs(s) for s in shifts)) if shifts else None,
        "rms_relative_shift": (
            float(math.sqrt(sum(s * s for s in shifts) / len(shifts)))
            if shifts
            else None
        ),
        "kolmogorov_distance_b": kolmogorov_d,
        "mean_b_accepted_fm": float(np.mean([item[0] for item in draws if item[2]])),
        "mean_b_all_draws_fm": float(np.mean([item[0] for item in draws])),
        "mean_nu_accepted": float(np.mean(nu_of_event)),
        "mean_nu_all_draws": float(np.mean(nu_of_draw)),
    }

    attempts = list(builder.accepted_generation_attempts)
    attempt_hist = {
        str(attempt): int(count)
        for attempt, count in zip(
            *np.unique(attempts, return_counts=True)
        )
    }
    return {
        "n_accepted": n_acc,
        "n_glauber_draws": n_draws,
        "n_rejected_events": int(builder.generation_rejected),
        "overall_acceptance": float(n_acc / n_draws) if n_draws > 0 else None,
        "failure_reason_counts": dict(builder.failure_reason_counts),
        "attempt_distribution": attempt_hist,
        "mean_generation_attempt": float(np.mean(attempts)) if attempts else None,
        "max_generation_attempt": int(max(attempts)) if attempts else None,
        "wound": wound_comparison(acc_nu, ref_nu, min_accepted=min_qualifying),
        "b_table_native": b_table,
        "b_table_coarse": coarse_b,
        "nb_joint_table": joint_table,
        "nb_failed_cells": failed_cells,
        "centrality_shift": centrality_shift,
        "max_conservation_residual": float(
            max(
                event.conservation_residuals()["relative_max"]
                for event in events
            )
        ),
    }


# ---------------------------------------------------------------------------
# Run orchestration
# ---------------------------------------------------------------------------

def run_audit(
    config: dict[str, Any],
    seed: int,
    n_accepted: int = DEFAULT_N_ACCEPTED,
    min_qualifying: int = DEFAULT_MIN_QUALIFYING,
    min_shift_draws: int = 50,
    coarse_dr_fm: float = 0.5,
    out_dir: str | Path | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    """Run the acceptance audit and write the JSON report.

    ``seed`` is the full audit seed (``config.seed + offset``; the
    offset convention is documented in the run's pre-registration). The
    report is written to ``out_dir`` (default: the layout default,
    ``dev/00runs/<RUN_NAME>`` in the monorepo) as
    ``pa_acceptance_audit_report.json``.
    """
    builder = build_pa_builder(config, seed)
    t0 = time.monotonic()
    events = run_acceptance_sample(builder, n_accepted)
    wall = time.monotonic() - t0
    for event in events:
        event.metadata["random_seed"] = int(seed)
        event.metadata["random_substream"] = f"baryon_nucleus:{event.event_id}"
    payload = summarize_acceptance(
        builder,
        events,
        min_qualifying=min_qualifying,
        min_shift_draws=min_shift_draws,
        coarse_dr_fm=coarse_dr_fm,
    )
    nucleus = dict(config.get("nucleus") or {})
    report = {
        "card": "TASK-20260905-physics-review",
        "milestone": "M5",
        "run_name": RUN_NAME,
        "system": f"p+{nucleus.get('name', 'generic')}",
        "configuration": {
            "hash": configuration_hash(config),
            "nucleus": nucleus,
            "glauber_dr_fm": float(
                dict(config.get("pA", {})).get("glauber_dr_fm", 0.05)
            ),
            "max_event_attempts": int(
                dict(config.get("pA", {})).get("max_event_attempts", 100)
            ),
            "beam_momentum": float(config.get("beam_momentum", 158.0)),
            "production_mode": "regge",
            "regge_parameters": dict(config.get("regge") or {}),
            "note": (
                "the bundled configuration is used unchanged: the audit "
                "measures acceptance bias at the recorded production "
                "shape, not the fitted shape"
            ),
        },
        "seed": int(seed),
        "statistics": {
            "n_accepted": int(n_accepted),
            "n_glauber_draws": int(payload["n_glauber_draws"]),
            "wall_seconds": float(wall),
            "ms_per_accepted_event": float(1000.0 * wall / len(events)),
        },
        "decision": payload["wound"],
        "aggregate": {
            key: payload[key]
            for key in (
                "overall_acceptance",
                "n_rejected_events",
                "failure_reason_counts",
                "attempt_distribution",
                "mean_generation_attempt",
                "max_generation_attempt",
                "max_conservation_residual",
            )
        },
        "acceptance_vs_nu": payload["wound"]["entries"],
        "acceptance_vs_b_native": payload["b_table_native"],
        "acceptance_vs_b_coarse": payload["b_table_coarse"],
        "acceptance_vs_nu_b": {
            "joint_table": payload["nb_joint_table"],
            "failed_cells": payload["nb_failed_cells"],
            "note": (
                "accepted/attempted per joint (nu, b-bin) cell over the "
                "populated cells; failed_cells lists exactly the cells "
                "with at least one failed attempt (each failed attempt "
                "is resampled as a whole new Glauber draw, so a failed "
                "cell contributes attempts > accepted)"
            ),
        },
        "centrality_shift": payload["centrality_shift"],
    }
    if write_report:
        target = Path(out_dir) if out_dir is not None else _default_out_dir()
        target.mkdir(parents=True, exist_ok=True)
        out_path = target / "pa_acceptance_audit_report.json"
        out_path.write_text(
            json.dumps(
                report, indent=2, sort_keys=True, default=_json_default
            )
            + "\n"
        )
        report["report_path"] = str(out_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=None,
        help="staged-evaluation JSON (default: the bundled "
             "dev/mixing2006/config/staged_excitation.json)",
    )
    parser.add_argument(
        "--n-accepted",
        type=int,
        default=DEFAULT_N_ACCEPTED,
        help=f"accepted-event count (default {DEFAULT_N_ACCEPTED})",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=DEFAULT_SEED_OFFSET,
        help=(
            "audit seed offset added to config.seed (default "
            f"{DEFAULT_SEED_OFFSET}, distinct from the M4 fit offset 10000)"
        ),
    )
    parser.add_argument(
        "--min-accepted",
        type=int,
        default=DEFAULT_MIN_QUALIFYING,
        help=(
            "pre-registered qualifying-bin threshold: a nu bin qualifies "
            f"when it holds at least this many accepted events (default "
            f"{DEFAULT_MIN_QUALIFYING})"
        ),
    )
    parser.add_argument(
        "--min-shift-draws",
        type=int,
        default=50,
        help=(
            "minimum Glauber draws per b bin included in the "
            "centrality-shift metrics (default 50)"
        ),
    )
    parser.add_argument(
        "--coarse-dr-fm",
        type=float,
        default=0.5,
        help="coarse b rebin width for the report (default 0.5 fm)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="report directory (default: dev/00runs/" + RUN_NAME
             + " in the monorepo, 00runs/" + RUN_NAME + " standalone)",
    )
    parser.add_argument(
        "--probe",
        type=int,
        default=0,
        help=(
            "generate this many accepted events, print the per-event "
            "wall time and the acceptance rate, and write no report"
        ),
    )
    args = parser.parse_args(argv)

    _bootstrap()
    config = load_bundled_config(args.config)
    seed = int(config.get("seed", 20260831)) + int(args.seed_offset)

    if args.probe > 0:
        builder = build_pa_builder(config, seed)
        t0 = time.monotonic()
        events = run_acceptance_sample(builder, args.probe)
        wall = time.monotonic() - t0
        n_draws = builder.generation_attempts
        n_acc = len(events)
        estimate = wall / n_acc * float(args.n_accepted)
        print(
            f"probe: {n_acc} accepted events in {wall:.2f} s "
            f"({1000.0 * wall / n_acc:.2f} ms per accepted event), "
            f"{n_draws} Glauber draws, acceptance "
            f"{n_acc / n_draws:.4f}, rejected {builder.generation_rejected}, "
            f"failure counts {builder.failure_reason_counts}"
        )
        print(
            f"estimate for --n-accepted {args.n_accepted}: "
            f"{estimate / 60.0:.1f} min at the probed rate"
        )
        return 0

    report = run_audit(
        config=config,
        seed=seed,
        n_accepted=args.n_accepted,
        min_qualifying=args.min_accepted,
        min_shift_draws=args.min_shift_draws,
        coarse_dr_fm=args.coarse_dr_fm,
        out_dir=args.out_dir,
    )
    decision = report["decision"]
    print(json.dumps({
        "report": report.get("report_path"),
        "seed": report["seed"],
        "n_accepted": report["statistics"]["n_accepted"],
        "n_glauber_draws": report["statistics"]["n_glauber_draws"],
        "wall_seconds": report["statistics"]["wall_seconds"],
        "overall_acceptance": report["aggregate"]["overall_acceptance"],
        "failure_reason_counts": report["aggregate"]["failure_reason_counts"],
        "n_qualifying_bins": decision["n_qualifying_bins"],
        "violating_bins": decision["violating_bins"],
        "decision_passed": decision["decision_passed"],
        "verdict": decision["verdict"],
        "centrality_shift": report["centrality_shift"],
    }, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
