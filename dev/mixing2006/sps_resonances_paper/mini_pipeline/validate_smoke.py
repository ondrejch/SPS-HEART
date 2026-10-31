#!/usr/bin/env python3
"""Validate one mini-pipeline smoke run directory.

Called by ``run_smoke.sh`` after the fit. The checks cover artifact
presence, the provenance record, the summary structure, the fit-solver
identifiability payload, the existing details-NPZ schema validation, the
absolute-yield prior report, the committed-cache integrity, and the pinned
reference numbers from ``expected_smoke.py``. Every check prints one PASS
or FAIL line; the script exits 0 only when all checks pass.

Usage::

    python validate_smoke.py RUN_DIR [--fixture-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

MINI_PIPELINE_DIR = Path(__file__).resolve().parent
MIXING2006_ROOT = MINI_PIPELINE_DIR.parents[1]
if str(MIXING2006_ROOT) not in sys.path:
    sys.path.insert(0, str(MIXING2006_ROOT))

import numpy as np  # noqa: E402
from helpers.benchmarking.sps_core import validate_multispecies_details_npz  # noqa: E402

from expected_smoke import (  # noqa: E402
    EXPECTED_COMPONENT_WEIGHTS,
    EXPECTED_DROPPED_ZERO_COLUMN_COMPONENTS,
    EXPECTED_GLOBAL_METRICS,
    EXPECTED_N_ACTIVE,
    EXPECTED_N_COMPONENTS,
    EXPECTED_PRIOR_PULLS,
    EXPECTED_PRIOR_TARGETS,
    EXPECTED_SPECIES_N_POINTS,
    EXPECTED_SPECIES_WRRMSE,
    METRIC_RTOL,
    N_ACTIVE_BAND,
    PULL_RTOL,
    RMSE_RTOL,
    WEIGHT_ATOL,
    WEIGHT_RTOL,
)

SUMMARY_NAME = "multispecies_constrained_summary.json"
DETAILS_NAME = "multispecies_constrained_details.npz"
PROVENANCE_NAME = "run_provenance.json"

SUMMARY_TOP_LEVEL_KEYS = (
    "weak_feed_policy",
    "histogram_convention",
    "config",
    "species",
    "resonance",
    "fit_solver",
    "per_species",
    "particle_ratios",
    "seagull_metrics",
)

RUN_ARTIFACT_NAMES = (
    SUMMARY_NAME,
    DETAILS_NAME,
    PROVENANCE_NAME,
    "MULTISPECIES_CONSTRAINED_SUMMARY.md",
    "multispecies_constrained_wrrmse.csv",
    "multispecies_constrained_particle_ratios.csv",
    "multispecies_constrained_seagull_slopes.csv",
    "multispecies_constrained_seagull_profiles.csv",
)

_PER_SPECIES_METRIC_KEYS = (
    "full_chi2",
    "full_chi2_dof",
    "full_smape_percent",
    "full_weighted_rel_rmse",
    "full_log_pull_rms",
)

_CHECKS: list[str] = []
_FAILURES: list[str] = []


def _record(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        line = f"PASS {name}"
        if detail:
            line += f": {detail}"
        _CHECKS.append(line)
        print(line)
        return
    line = f"FAIL {name}"
    if detail:
        line += f": {detail}"
    _FAILURES.append(line)
    print(line)


def _close(actual: float, expected: float, rtol: float, atol: float = 0.0) -> bool:
    return math.isfinite(actual) and abs(actual - expected) <= atol + rtol * abs(expected)


def _metric_rtol(name: str) -> float:
    return RMSE_RTOL if "rel_rmse" in name else METRIC_RTOL


def check_artifacts(run_dir: Path) -> bool:
    missing = [name for name in RUN_ARTIFACT_NAMES if not (run_dir / name).is_file()]
    _record(
        "artifacts_exist",
        not missing,
        "" if not missing else f"missing {missing}",
    )
    return not missing


def check_provenance(run_dir: Path) -> None:
    path = run_dir / PROVENANCE_NAME
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _record("provenance_record", False, f"unreadable {path} ({exc})")
        return
    if not isinstance(record, dict):
        _record("provenance_record", False, "payload is not a JSON object")
        return

    producer = str(record.get("producer", ""))
    verification = record.get("input_verification")
    manifest_state = (
        str(verification.get("manifest", "")) if isinstance(verification, dict) else ""
    )
    mismatches = (
        list(verification.get("mismatches", [])) if isinstance(verification, dict) else ["absent"]
    )
    n_inputs = len(record.get("inputs", [])) if isinstance(record.get("inputs"), list) else 0
    ok = (
        "benchmark_multispecies_constrained" in producer
        and manifest_state in ("matched", "absent")
        and not mismatches
        and n_inputs >= 10
        and isinstance(record.get("argv"), list)
        and bool(record.get("argv"))
    )
    _record(
        "provenance_record",
        ok,
        (
            f"producer={producer!r}, manifest={manifest_state!r}, "
            f"inputs_with_manifest_entry="
            f"{verification.get('inputs_with_manifest_entry') if isinstance(verification, dict) else 'n/a'}, "
            f"recorded_inputs={n_inputs}"
        ),
    )


def check_summary_structure(run_dir: Path) -> dict | None:
    path = run_dir / SUMMARY_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _record("summary_structure", False, f"unreadable {path} ({exc})")
        return None
    if not isinstance(payload, dict):
        _record("summary_structure", False, "payload is not a JSON object")
        return None

    missing = [key for key in SUMMARY_TOP_LEVEL_KEYS if key not in payload]
    if missing:
        _record("summary_structure", False, f"missing keys {missing}")
        return payload

    problems: list[str] = []
    species = payload.get("species", {})
    for key, n_points in EXPECTED_SPECIES_N_POINTS.items():
        entry = species.get(key)
        if not isinstance(entry, dict) or entry.get("n_points") != n_points:
            problems.append(f"species[{key}].n_points != {n_points}")

    per_species = payload.get("per_species", {})
    for key in EXPECTED_SPECIES_N_POINTS:
        models = per_species.get(key, {})
        if set(models) != {"Resonance_NNLS_shared"}:
            problems.append(
                f"per_species[{key}] must carry only the shared resonance fit "
                f"(external models are skipped in the smoke run), found {sorted(models)}"
            )
            continue
        row = models["Resonance_NNLS_shared"]
        for metric in _PER_SPECIES_METRIC_KEYS:
            value = row.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                problems.append(f"per_species[{key}].{metric} is not finite")

    global_metrics = payload.get("resonance", {}).get("global_metrics", {})
    for metric in EXPECTED_GLOBAL_METRICS:
        value = global_metrics.get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            problems.append(f"resonance.global_metrics.{metric} is not finite")

    _record(
        "summary_structure",
        not problems,
        "" if not problems else "; ".join(problems),
    )
    return payload


def check_fit_solver(payload: dict) -> None:
    solver = payload.get("fit_solver", {})
    resonance = payload.get("resonance", {})
    n_components = resonance.get("n_components")
    identifiability = solver.get("identifiability")
    design_rank = solver.get("design_rank")
    design_n = solver.get("design_n_components")
    condition = solver.get("design_condition_number")

    ok = (
        solver.get("success") is True
        and identifiability == "full_rank_identifiable"
        and design_rank == EXPECTED_N_COMPONENTS
        and design_n == EXPECTED_N_COMPONENTS
        and n_components == EXPECTED_N_COMPONENTS
        and isinstance(condition, (int, float))
        and math.isfinite(float(condition))
        and float(condition) > 0.0
    )
    _record(
        "fit_solver_identifiability",
        ok,
        (
            f"success={solver.get('success')}, identifiability={identifiability!r}, "
            f"design_rank={design_rank}, n_components={n_components}, "
            f"condition_number={condition}"
        ),
    )

    n_active = resonance.get("n_active")
    _record(
        "n_active_weights",
        isinstance(n_active, int)
        and abs(n_active - EXPECTED_N_ACTIVE) <= N_ACTIVE_BAND,
        f"n_active={n_active}, expected {EXPECTED_N_ACTIVE} "
        f"within +/-{N_ACTIVE_BAND}",
    )

    dropped = resonance.get("dropped_zero_column_components")
    _record(
        "zero_column_components_dropped",
        dropped == EXPECTED_DROPPED_ZERO_COLUMN_COMPONENTS,
        f"dropped={dropped!r}",
    )


def check_details_npz(run_dir: Path, payload: dict) -> None:
    path = run_dir / DETAILS_NAME
    try:
        with np.load(path, allow_pickle=False) as zarr:
            validate_multispecies_details_npz(
                zarr, details_path=path, summary_payload=payload
            )
    except Exception as exc:  # noqa: BLE001 - one line per validation failure
        _record("details_npz_schema", False, f"{type(exc).__name__}: {exc}")
        return
    _record("details_npz_schema", True, "validate_multispecies_details_npz clean")


def check_prior_report(payload: dict) -> None:
    priors = payload.get("resonance", {}).get("priors", {})
    ay = priors.get("absolute_yield", {}) if isinstance(priors, dict) else {}
    fit_rows = ay.get("fit_rows", []) if isinstance(ay, dict) else []
    rows_by_key = {
        str(row.get("species_key")): row
        for row in fit_rows
        if isinstance(row, dict)
    }

    problems: list[str] = []
    if not ay.get("enabled", False):
        problems.append("absolute_yield prior is not enabled")
    if ay.get("n_rows") != len(EXPECTED_PRIOR_TARGETS):
        problems.append(f"n_rows={ay.get('n_rows')}")

    for key, target in EXPECTED_PRIOR_TARGETS.items():
        row = rows_by_key.get(key)
        if row is None:
            problems.append(f"no prior fit row for {key}")
            continue
        if not _close(float(row.get("target", float("nan"))), target, 1e-9):
            problems.append(f"{key} target {row.get('target')} != fixture {target}")
        pull = row.get("pull")
        expected_pull = EXPECTED_PRIOR_PULLS[key]
        if not isinstance(pull, (int, float)) or not _close(
            float(pull), expected_pull, PULL_RTOL
        ):
            problems.append(
                f"{key} pull {pull} outside {PULL_RTOL:.0%} of {expected_pull}"
            )

    _record("absyield_prior_report", not problems, "" if not problems else "; ".join(problems))


def check_pinned_numbers(payload: dict) -> None:
    resonance = payload.get("resonance", {})

    weights = resonance.get("weights_by_label_sum", {})
    problems: list[str] = []
    for label, expected in EXPECTED_COMPONENT_WEIGHTS.items():
        actual = weights.get(label)
        if not isinstance(actual, (int, float)) or not _close(
            float(actual), expected, WEIGHT_RTOL, WEIGHT_ATOL
        ):
            problems.append(f"weight[{label}]={actual!r}, expected {expected!r}")
    _record(
        "pinned_component_weights",
        not problems,
        "" if not problems else "; ".join(problems[:6]),
    )

    global_metrics = resonance.get("global_metrics", {})
    problems = []
    for metric, expected in EXPECTED_GLOBAL_METRICS.items():
        actual = global_metrics.get(metric)
        rtol = _metric_rtol(metric)
        if not isinstance(actual, (int, float)) or not _close(
            float(actual), expected, rtol
        ):
            problems.append(f"global[{metric}]={actual!r}, expected {expected!r}")
    _record(
        "pinned_global_metrics",
        not problems,
        "" if not problems else "; ".join(problems),
    )

    per_species = payload.get("per_species", {})
    problems = []
    for key, expected in EXPECTED_SPECIES_WRRMSE.items():
        row = per_species.get(key, {}).get("Resonance_NNLS_shared", {})
        actual = row.get("full_weighted_rel_rmse")
        if not isinstance(actual, (int, float)) or not _close(
            float(actual), expected, RMSE_RTOL
        ):
            problems.append(f"wrrmse[{key}]={actual!r}, expected {expected!r}")
    _record(
        "pinned_species_wrrmse",
        not problems,
        "" if not problems else "; ".join(problems),
    )


def check_cache_integrity(run_dir: Path, fixture_dir: Path) -> None:
    manifest_path = fixture_dir / "fixture_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _record("fixture_cache_integrity", False, f"unreadable {manifest_path} ({exc})")
        return
    expected_files = manifest.get("n_cache_files")
    run_cache = run_dir / "resonance_component_cache"
    actual = len(list(run_cache.glob("*.npz"))) if run_cache.is_dir() else -1
    _record(
        "fixture_cache_integrity",
        actual == expected_files,
        f"{actual} cache files in the run directory, fixture manifest records {expected_files}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate one mini-pipeline smoke run.")
    parser.add_argument("run_dir", help="Run directory written by run_smoke.sh")
    parser.add_argument(
        "--fixture-dir",
        default=str(MINI_PIPELINE_DIR / "fixture"),
        help="Committed fixture directory (default: mini_pipeline/fixture).",
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    fixture_dir = Path(args.fixture_dir).expanduser().resolve()

    if not check_artifacts(run_dir):
        print(f"FAIL mini-pipeline smoke validation: {run_dir}")
        return 1
    check_provenance(run_dir)
    payload = check_summary_structure(run_dir)
    if payload is None:
        print(f"FAIL mini-pipeline smoke validation: {run_dir}")
        return 1
    check_fit_solver(payload)
    check_details_npz(run_dir, payload)
    check_prior_report(payload)
    check_pinned_numbers(payload)
    check_cache_integrity(run_dir, fixture_dir)

    if _FAILURES:
        print(
            f"FAIL mini-pipeline smoke validation: {run_dir} "
            f"({len(_FAILURES)} of {len(_CHECKS) + len(_FAILURES)} checks failed)"
        )
        return 1
    print(f"PASS mini-pipeline smoke validation: {run_dir} ({len(_CHECKS)} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
