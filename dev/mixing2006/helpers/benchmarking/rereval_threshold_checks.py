#!/usr/bin/env python3
"""Sealed-archive re-evaluation driver for the re-registered
threshold-discontinuity metric (Round 3, card 2).

Task ``TASK-20260917-r3-gate2-followup-threshold-metric-reregistration``
(the Gate 2 closure obligation of the Round 2 decision, follow-up 2 of
``_knowledge/operations/round2-closure-2026-09-17.md``; plan section 8,
Gate 2 threshold-discontinuity clause, line 580).  The R21/R22
pre-registered metric applied ``delta_analytic = 0`` at open-set-
unchanged pairs, while the full analytic difference
``delta_analytic_full`` is nonzero (the daughter momentum moments move
with the source mass even at a constant open set).  R21: 169 of 351
applied checks fail; R22: 154 of 483.  This driver re-evaluates every
applied check of the two sealed archives against the sealed stored
``delta_analytic_full`` at the unchanged pre-registered 3.0-sigma
tolerance and assembles the revised Gate 2 threshold clause evidence.

No Monte Carlo and no new generation: the sealed R21/R22 samples
(2.0M and 2.8M decays) are reused read-only.  No random draws exist on
any code path; the master seed ``20260917`` is recorded in the
pre-registration with ``draws: 0`` for contract uniformity with the
Round 3 seed rule.

What the driver re-evaluates: the sealed ``gate2_r21.json`` /
``gate2_r22.json`` record the unapplied full analytic difference
``delta_analytic_full`` per check alongside the applied value, so the
re-evaluation recomputes each applied check's sigma from the sealed
stored fields through ``na49_mixing.threshold_metric.rereval_report``
(a deterministic, byte-stable recomputation; the sealed archives are
read-only and are not re-sealed, and their gate JSONs keep the original
metric's verdict).

What the driver validates: the new-run analytic engine
(``na49_mixing.threshold_metric.full_analytic_difference``) recomputed
over each sealed grid against the sealed stored values (the
``--crosscheck`` mode; the engine-validation evidence of the
re-registration, recorded per run with the maximum deviation per
observable).

Modes (exactly one per invocation; each performs exactly one step and
stops; read-only over the sealed archives; writes only into its own run
dir):

- ``--pre-registration``: write ``pre_registration.json`` and
  ``provenance.json`` into the run dir, then stop.  Nothing is
  computed.
- ``--rereval r21|r22|both``: re-evaluate the named sealed archive(s)
  and write ``rereval_<run>.json`` into the run dir, then stop.
  Requires the pre-registration.
- ``--crosscheck r21|r22|both``: recompute the new-run analytic engine
  over the named sealed grid(s) and write
  ``metric_engine_crosscheck_<run>.json`` into the run dir, then stop
  (the slow mode: the full-grid analytic quadrature; R22 takes tens of
  minutes).  Requires the pre-registration.
- ``--report``: assemble ``gate2_revised_threshold.json`` (the revised
  clause evidence: per-pair-class pass rates, the maximum sigma, the
  named residual pairs with their one-line diagnoses, and the
  re-registration acceptance verdict) from the two rereval payloads,
  then stop.
- ``--seal``: write ``seal_manifest.json`` and ``SHA256SUMS.txt`` over
  the run dir, then stop.  The run dir is immutable after the seal.
- ``--clean-validate``: verify the seal, re-run the re-evaluation and
  the report assembly from a clean interpreter over the sealed
  artifacts, and confirm the report hashes match; write
  ``clean_environment_commands.json`` into the run dir, then stop.

Card 8a phase-2 live modes (additive; the fixed-engine re-evaluation of
``TASK-20260918-r3-card2-engine-three-body-moment-fix``; each performs
exactly one step and stops; the live run dir is fixed to
``dev/00runs/r3_threshold_metric_engine_fix_2026-09-18/``; read-only
over the sealed R21/R22 archives, the stopped R31 artifacts, and the
sealed card-2 dir):

- ``--live-pre-registration``: write the engine-fix
  ``pre_registration.json`` and ``provenance.json`` into the live run
  dir, then stop.  Nothing is computed.
- ``--live-reeval r21|r22|r31|both``: live re-evaluation of the fixed
  engine's recomputed ``delta_analytic_full`` against the sealed
  ``delta_observed`` (r21|r22|both, the 834 applied checks) or the
  targeted R31 check (r31, the 422 applied checks of the stopped
  matching verification), at the unchanged 3.0-sigma tolerance and the
  pre-registered deterministic-SE floor convention; writes
  ``live_reeval_<run>.json`` into the live run dir, then stop.
  Requires the live pre-registration.
- ``--live-report``: assemble ``gate2_revised_threshold_v2.json`` (the
  improved Gate 2 threshold clause evidence: the fixed-engine values
  against the sealed card-2 baseline, the acceptance verdicts on the
  card's T1 and T2 criteria, and the named residual findings) from the
  three live payloads, then stop.
- ``--live-seal``: write ``seal_manifest.json`` and ``SHA256SUMS.txt``
  over the live run dir, then stop.
- ``--live-clean-validate``: verify the live seal, re-run the three
  live re-evaluations and the report assembly from a clean interpreter
  over the sealed artifacts, and confirm the hashes match; write
  ``clean_environment_commands.json`` into the live run dir, then stop.

Deterministic: the pre-registration, the rereval payloads, the
crosscheck payloads, and the revised-threshold report carry no
wall-clock fields (their bytes are a pure function of the sealed
archives and the committed code); only the provenance, the seal
manifest, and the clean-environment record carry wall-clock stamps.

Run dir: ``dev/00runs/r3_threshold_metric_reregistration_2026-09-17/``
(the Round 2 card-8 diagnostic-run pattern).

Dual-tree driver: byte-identical in the nested tree
(``dev/mixing2006/helpers/benchmarking/``) and the monorepo mirror
(``helpers/benchmarking/``); both locate the package with the
established ``_locate_roots`` pattern.

Hermetic: standard library, numpy, and the vendored ``na49_mixing``
modules only; no event generation, no external tables.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Tree location: this driver runs from the nested tree
# (dev/mixing2006/helpers/benchmarking/) and from its byte-identical
# monorepo mirror (helpers/benchmarking/).  Both must find the
# na49_mixing package and the sealed R21/R22 archives.
# ---------------------------------------------------------------------------


def _locate_roots() -> tuple["Path", "Path"]:
    here = Path(__file__).resolve()
    nested = None
    for base in (here.parents[2], here.parents[2] / "dev" / "mixing2006"):
        if (base / "na49_mixing").is_dir():
            nested = base.resolve()
            break
    if nested is None:
        print(
            "rereval_threshold_checks: cannot locate the na49_mixing "
            f"package next to {here}; the driver runs from the nested "
            "tree (dev/mixing2006/helpers/benchmarking/) or its "
            "monorepo mirror (helpers/benchmarking/)",
            file=sys.stderr,
        )
        raise SystemExit(2)
    mono = nested.parent.parent if nested.parent.name == "dev" else nested
    return nested, mono


NESTED_ROOT, MONO_ROOT = _locate_roots()
if str(NESTED_ROOT) not in sys.path:
    sys.path.insert(0, str(NESTED_ROOT))

import numpy as np  # noqa: E402

from na49_mixing.hadron_catalog import HadronCatalog  # noqa: E402
from na49_mixing.threshold_metric import (  # noqa: E402
    DETERMINISTIC_SE_FLOOR,
    OBSERVABLES,
    SIGMA_TOLERANCE,
    THRESHOLD_METRIC_ID,
    ZERO_RESIDUAL_TOL,
    RecordSet,
    _reevaluated_status,
    diagnose_residual,
    full_analytic_difference,
    rereval_report,
)

# ---------------------------------------------------------------------------
# Run identity and contract constants
# ---------------------------------------------------------------------------
RUN_ID = "R3-THRESHOLD-METRIC-REREGISTRATION"
CARD_ID = "TASK-20260917-r3-gate2-followup-threshold-metric-reregistration"
RUN_NAME = "r3_threshold_metric_reregistration_2026-09-17"
DEFAULT_RUN_DIR = MONO_ROOT / "dev" / "00runs" / RUN_NAME
RUN_DIR_REL = f"dev/00runs/{RUN_NAME}"

CATALOG_REL = "config/v2_singlet/resonance_catalog.json"

#: The sealed archives (read-only; never re-sealed).
ARCHIVES: dict[str, dict] = {
    "r21": {
        "run": "R21",
        "name": "r21_baryonic_source_grid_2026-09-13",
        "gate": "gate2_r21.json",
        "card": "TASK-20260913-r2-run-r21-baryonic-source-grid",
        "driver": "helpers/benchmarking/staged_source_grid.py",
        "source_type": "baryonic",
    },
    "r22": {
        "run": "R22",
        "name": "r22_mesonic_source_grid_2026-09-13",
        "gate": "gate2_r22.json",
        "card": "TASK-20260913-r2-run-r22-mesonic-source-grid",
        "driver": "helpers/benchmarking/staged_source_grid.py",
        "source_type": "mesonic",
    },
}

#: Master seed recorded for contract uniformity with the Round 3 seed
#: rule; the driver draws no random numbers (``draws: 0``).
MASTER_SEED = 20260917
DRAWS = 0

#: The pre-registered tolerance (the R21/R22 band; unchanged by this
#: card and never relaxed).
N_SIGMA = SIGMA_TOLERANCE

#: The re-registration acceptance criterion of the card: at least 95%
#: of the checks that failed under the original metric (169 R21 + 154
#: R22 = 323) pass at 3.0 sigma under ``delta_analytic_full``.
ACCEPTANCE_RATE = 0.95
N_ORIGINAL_FAILURES = 323

#: The applied-check statuses of the sealed metric (the re-evaluation
#: applies to these; ``absent_species`` checks carry no verdict).
APPLIED_STATUSES = ("pass", "fail", "vacuous_zero")

PRE_SCHEMA = "na49_mixing.rereval_threshold_checks.pre_registration/1"
PROV_SCHEMA = "na49_mixing.rereval_threshold_checks.provenance/1"
REREVAL_SCHEMA = "na49_mixing.threshold_metric.rereval/1"
CROSSCHECK_SCHEMA = "na49_mixing.rereval_threshold_checks.crosscheck/1"
REPORT_SCHEMA = "na49_mixing.rereval_threshold_checks.revised_clause/1"
SEAL_SCHEMA = "na49_mixing.rereval_threshold_checks.seal_manifest/1"
CLEAN_SCHEMA = "na49_mixing.rereval_threshold_checks.clean_environment/1"


# ---------------------------------------------------------------------------
# Small shared helpers (the Round 3 card 1 driver's conventions)
# ---------------------------------------------------------------------------
def _now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _write_json(path: Path, payload: dict) -> None:
    """Write *payload* as UTF-8 JSON (indent 2, fixed key order, a
    trailing newline, LF endings); identical inputs produce identical
    bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _canonical_json_bytes(payload: dict) -> bytes:
    """The exact byte sequence :func:`_write_json` writes for *payload*."""
    return (json.dumps(payload, indent=2) + "\n").encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_fingerprint() -> dict:
    """The environment fingerprint recorded in the provenance and the
    clean-environment validation."""
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy_version": np.__version__,
        "executable": sys.executable,
    }


def _git_info() -> dict:
    """The recorded git identities (defensive: a checkout without a
    working git records ``None``)."""
    info: dict = {}
    for name, cwd in (
        ("nested_head", NESTED_ROOT), ("outer_head", MONO_ROOT)
    ):
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


def _archive_dir(key: str) -> Path:
    return MONO_ROOT / "dev" / "00runs" / ARCHIVES[key]["name"]


def _verify_archive_seal(archive: Path) -> dict:
    """Read-only verification of a sealed archive; returns its seal
    identity (the file count and the SHA-256 of its SHA256SUMS.txt)."""
    sums_path = archive / "SHA256SUMS.txt"
    manifest_path = archive / "seal_manifest.json"
    if not sums_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(
            f"rereval_threshold_checks: {archive} is not a sealed "
            "archive (SHA256SUMS.txt / seal_manifest.json missing)"
        )
    checked = 0
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        # The sealed R21/R22 sums carry dev/00runs-relative entries;
        # run-dir-relative entries are resolved against the archive.
        path = archive / rel
        if not path.is_file():
            path = MONO_ROOT / rel
        if not path.is_file() or _sha256_file(path) != digest:
            raise RuntimeError(
                f"rereval_threshold_checks: the sealed archive "
                f"{archive} failed its own seal at {rel}; the archive "
                "is read-only evidence and must not be modified"
            )
        checked += 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sha256_sums_sha256") != _sha256_file(sums_path):
        raise RuntimeError(
            f"rereval_threshold_checks: the seal manifest of {archive} "
            "does not match its SHA256SUMS.txt"
        )
    return {"files_checked": checked,
            "sha256_sums_sha256": _sha256_file(sums_path)}


# ---------------------------------------------------------------------------
# Pre-registration and provenance
# ---------------------------------------------------------------------------
def _build_pre_registration() -> dict:
    return {
        "schema": PRE_SCHEMA,
        "run": RUN_ID,
        "card": CARD_ID,
        "mode": "diagnostic re-evaluation over the sealed archives "
                "(no Monte Carlo, no generation)",
        "master_seed": MASTER_SEED,
        "draws": DRAWS,
        "seed_note": (
            "no random draws exist on any code path; the master seed "
            "is recorded for contract uniformity with the Round 3 seed "
            "rule"
        ),
        "metric": {
            "id": THRESHOLD_METRIC_ID,
            "statement": (
                "for every adjacent pair (k, k+1) and every stable "
                "species, with SE_combined = sqrt(SE_k^2 + SE_{k+1}^2): "
                "|mu(k+1) - mu(k) - delta_analytic_full| <= 3 "
                "SE_combined on BOTH pair classes (open-set-change and "
                "open-set-unchanged), for the species multiplicity and "
                "the per-species momentum moments <p> and <p_T>_y"
            ),
            "tolerance_sigma": N_SIGMA,
            "tolerance_unchanged": True,
            "revises": (
                "the R21/R22 pre-registered metric applied "
                "delta_analytic = 0 at open-set-unchanged pairs; the "
                "re-registered metric compares both pair classes "
                "against the full analytic difference"
            ),
            "source": (
                "the sealed per-check delta_analytic_full values (the "
                "unapplied full analytic difference recorded by the "
                "sealed metric); the sealed archives are read-only and "
                "are not re-sealed, and their gate JSONs keep the "
                "original metric's verdict"
            ),
            "engine": "na49_mixing.threshold_metric.rereval_report",
        },
        "archives": {
            key: {
                "run": spec["run"],
                "path": f"dev/00runs/{spec['name']}",
                "gate_file": spec["gate"],
                "card": spec["card"],
                "read_only": True,
            }
            for key, spec in ARCHIVES.items()
        },
        "coverage": {
            "r21": {
                "adjacent_pairs": 19,
                "applied_checks_original": 351,
                "failed_original": 169,
                "failed_at_open_set_change": 4,
                "failed_at_unchanged": 165,
            },
            "r22": {
                "adjacent_pairs": 27,
                "applied_checks_original": 483,
                "failed_original": 154,
                "failed_at_open_set_change": 1,
                "failed_at_unchanged": 153,
            },
            "observables": list(OBSERVABLES),
            "pair_classes": ["open_set_change", "open_set_unchanged"],
            "total_original_failures": N_ORIGINAL_FAILURES,
        },
        "acceptance": {
            "criterion": (
                "at least 95% of the checks that failed under the "
                "original metric (169 R21 + 154 R22 = 323) pass at 3.0 "
                "sigma under delta_analytic_full; the pass rate is "
                "reported per pair class"
            ),
            "threshold": ACCEPTANCE_RATE,
            "n_original_failures": N_ORIGINAL_FAILURES,
        },
        "constraints": {
            "no_fitting": True,
            "sealed_archives_read_only": True,
            "tolerance_relaxed": False,
            "residual_policy": (
                "every residual failing pair is named with its sigma "
                "and a one-line diagnosis; open-set-change residuals "
                "are expected (Round 2 closure clause 3) and are "
                "recorded as findings, never absorbed into the metric"
            ),
        },
        "outputs": [
            "rereval_r21.json",
            "rereval_r22.json",
            "gate2_revised_threshold.json",
            "metric_engine_crosscheck_r21.json",
            "metric_engine_crosscheck_r22.json",
        ],
        "written_before_evaluation": True,
    }


def _run_pre_registration(run_dir: Path) -> None:
    pre_path = run_dir / "pre_registration.json"
    prov_path = run_dir / "provenance.json"
    if pre_path.is_file() and (
        (run_dir / "rereval_r21.json").is_file()
        or (run_dir / "rereval_r22.json").is_file()
        or (run_dir / "metric_engine_crosscheck_r21.json").is_file()
        or (run_dir / "metric_engine_crosscheck_r22.json").is_file()
    ):
        raise RuntimeError(
            f"rereval_threshold_checks: {pre_path} exists and "
            "evaluation artifacts are present; the pre-registration is "
            "immutable after evaluation has started (a change "
            "invalidates the run; re-run under a new date)"
        )
    _write_json(pre_path, _build_pre_registration())
    provenance = {
        "schema": PROV_SCHEMA,
        "run": RUN_ID,
        "card": CARD_ID,
        "written_utc": _now_utc(),
        "master_seed": MASTER_SEED,
        "draws": DRAWS,
        "catalog": {
            "path": CATALOG_REL,
            "sha256": _sha256_file(NESTED_ROOT / CATALOG_REL),
        },
        "sealed_archives": {
            key: {
                "path": f"dev/00runs/{spec['name']}",
                "gate_file": spec["gate"],
                "gate_sha256": _sha256_file(
                    _archive_dir(key) / spec["gate"]),
                "seal": _verify_archive_seal(_archive_dir(key)),
            }
            for key, spec in ARCHIVES.items()
        },
        "drivers": {
            "rereval_driver": "helpers/benchmarking/"
                              "rereval_threshold_checks.py",
            "rereval_driver_sha256": _sha256_file(Path(__file__).resolve()),
            "metric_module": "dev/mixing2006/na49_mixing/"
                             "threshold_metric.py",
            "metric_module_sha256": _sha256_file(
                NESTED_ROOT / "na49_mixing" / "threshold_metric.py"),
            "source_grid_driver": "helpers/benchmarking/"
                                  "staged_source_grid.py",
        },
        "git": _git_info(),
        "environment": environment_fingerprint(),
    }
    _write_json(prov_path, provenance)
    print(
        f"rereval_threshold_checks: wrote {pre_path} and {prov_path} "
        f"(master seed {MASTER_SEED}, draws {DRAWS}; nothing computed)"
    )


def _require_pre_registration(run_dir: Path) -> None:
    if not (run_dir / "pre_registration.json").is_file():
        raise RuntimeError(
            f"rereval_threshold_checks: {run_dir / 'pre_registration.json'} "
            "is missing; run --pre-registration before any evaluation"
        )


# ---------------------------------------------------------------------------
# Re-evaluation (the fast mode: the sealed stored values)
# ---------------------------------------------------------------------------
def _run_rereval(run_dir: Path, keys: list[str]) -> None:
    _require_pre_registration(run_dir)
    for key in keys:
        spec = ARCHIVES[key]
        archive = _archive_dir(key)
        _verify_archive_seal(archive)
        gate_path = archive / spec["gate"]
        gate_payload = json.loads(
            gate_path.read_text(encoding="utf-8"))
        payload = rereval_report(
            gate_payload,
            run=spec["run"],
            archive_rel=f"dev/00runs/{spec['name']}/{spec['gate']}",
            gate_sha256=_sha256_file(gate_path),
            card=CARD_ID,
        )
        out_path = run_dir / f"rereval_{key}.json"
        _write_json(out_path, payload)
        summary = payload["summary"]
        print(
            f"rereval_threshold_checks: wrote {out_path} "
            f"({spec['run']}: applied {summary['n_checks_applied']}, "
            f"pass {summary['n_pass']}, fail {summary['n_fail']}; "
            f"original failures {summary['original']['n_checks_failed']} "
            f"-> now passing "
            f"{summary['reeval']['original_failures_now_passing']}; "
            f"residuals {len(summary['residuals'])})"
        )


# ---------------------------------------------------------------------------
# Engine cross-check (the slow mode: the full-grid analytic quadrature)
# ---------------------------------------------------------------------------
def _crosscheck_payload(key: str) -> dict:
    """The new-run analytic engine against the sealed stored
    ``delta_analytic_full`` over one sealed grid (pure function of the
    sealed archive and the committed module; no draws)."""
    spec = ARCHIVES[key]
    archive = _archive_dir(key)
    _verify_archive_seal(archive)
    gate = json.loads(
        (archive / spec["gate"]).read_text(encoding="utf-8"))
    pre = json.loads(
        (archive / "pre_registration.json").read_text(encoding="utf-8"))
    clause = gate["clauses"]["threshold_discontinuity"]
    open_sets = {int(k): v for k, v in pre["open_sets"].items()}
    masses = [float(m) for m in pre["grid_masses_gev"]]
    catalog = HadronCatalog.from_json(NESTED_ROOT / CATALOG_REL)
    union_ids = sorted({pdg for ids in open_sets.values() for pdg in ids})
    records = [catalog.get(pdg) for pdg in union_ids]
    record_set = RecordSet(catalog, records)
    applied = [c for c in clause["checks"]
               if c["status"] in APPLIED_STATUSES]
    per_pair: dict[tuple[int, int], list[dict]] = {}
    for check in applied:
        per_pair.setdefault((int(check["pair"][0]),
                             int(check["pair"][1])), []).append(check)
    worst: dict = {obs: {"max": 0.0, "where": None} for obs in OBSERVABLES}
    n_compared = 0
    for (k, kp1), rows in sorted(per_pair.items()):
        for skey in sorted({c["species"] for c in rows}):
            delta = full_analytic_difference(
                record_set, masses[k], masses[kp1], skey)
            for check in rows:
                if check["species"] != skey:
                    continue
                engine = delta["delta_analytic_full"][check["observable"]]
                sealed = check["delta_analytic_full"]
                if engine is None or sealed is None:
                    continue
                deviation = abs(float(engine) - float(sealed))
                n_compared += 1
                if deviation > worst[check["observable"]]["max"]:
                    worst[check["observable"]] = {
                        "max": deviation,
                        "where": [k, kp1, skey, check["observable"],
                                  round(float(engine), 12),
                                  round(float(sealed), 12)],
                    }
    applied_n = len(applied)
    return {
        "schema": CROSSCHECK_SCHEMA,
        "card": CARD_ID,
        "run": spec["run"],
        "archive": {
            "path": f"dev/00runs/{spec['name']}",
            "gate_file": spec["gate"],
            "gate_sha256": _sha256_file(archive / spec["gate"]),
        },
        "engine": {
            "module": "na49_mixing.threshold_metric",
            "function": "full_analytic_difference",
            "record_set": "na49_mixing.threshold_metric.RecordSet",
            "quadrature": (
                "the truncated mixture window integrated on a fixed "
                "Gauss-Legendre grid with the card-3 density and the "
                "open-object weights kernel-exact at every node; "
                "daughter budget windows integrated on a fixed "
                "Gauss-Legendre grid at the integrated level; narrow "
                "line shapes at their card-3 draw-distribution point "
                "and deeper levels at the concentration point; no "
                "Monte Carlo"
            ),
            "purpose": (
                "the new-run discontinuity metric of the card-9 driver "
                "(DISCONTINUITY_METRIC); this cross-check is the "
                "engine-validation evidence of the re-registration, "
                "recorded as a diagnostic, never an assertion"
            ),
        },
        "reference": (
            "the sealed per-check delta_analytic_full values (the "
            "sealed engine's deterministic quadrature; its own "
            "documented approximation scale is the engine validation "
            "recorded in the sealed gate payload)"
        ),
        "n_checks_applied": applied_n,
        "n_compared": n_compared,
        "max_abs_deviation": {
            obs: {
                "max": worst[obs]["max"],
                "where": (
                    {
                        "pair": worst[obs]["where"][0:2],
                        "species": worst[obs]["where"][2],
                        "observable": worst[obs]["where"][3],
                        "engine": worst[obs]["where"][4],
                        "sealed": worst[obs]["where"][5],
                    }
                    if worst[obs]["where"] is not None else None
                ),
            }
            for obs in OBSERVABLES
        },
    }


def _run_crosscheck(run_dir: Path, keys: list[str]) -> None:
    _require_pre_registration(run_dir)
    for key in keys:
        payload = _crosscheck_payload(key)
        out_path = run_dir / f"metric_engine_crosscheck_{key}.json"
        _write_json(out_path, payload)
        deviations = ", ".join(
            f"{obs} {payload['max_abs_deviation'][obs]['max']:.3e}"
            for obs in OBSERVABLES
        )
        print(
            f"rereval_threshold_checks: wrote {out_path} "
            f"({payload['run']}: {payload['n_compared']} analytic "
            f"values compared; max |engine - sealed|: {deviations})"
        )


# ---------------------------------------------------------------------------
# The revised Gate 2 threshold clause evidence
# ---------------------------------------------------------------------------
def build_revised_threshold(r21: dict, r22: dict) -> dict:
    """Assemble the revised clause evidence from the two rereval
    payloads (pure function; byte-stable)."""
    runs = {}
    combined = {
        "applied": 0,
        "pass": 0,
        "fail": 0,
        "original_failures": 0,
        "original_failures_now_passing": 0,
        "residual_failures": 0,
        "by_pair_class": {
            "open_set_change": {"original_failures": 0,
                                "original_failures_now_passing": 0,
                                "residual_failures": 0},
            "open_set_unchanged": {"original_failures": 0,
                                   "original_failures_now_passing": 0,
                                   "residual_failures": 0},
        },
        "max_abs_sigma": 0.0,
        "max_abs_sigma_check": None,
    }
    findings: list[dict] = []
    for key, payload in (("r21", r21), ("r22", r22)):
        summary = payload["summary"]
        rates = summary["reeval"]["pass_rate_all_applied"]
        original_fail = summary["original"]["n_checks_failed"]
        now_pass = summary["reeval"]["original_failures_now_passing"]
        runs[key] = {
            "run": payload["run"],
            "archive": payload["archive"],
            "n_checks": summary["n_checks"],
            "n_checks_applied": summary["n_checks_applied"],
            "n_pass": summary["n_pass"],
            "n_fail": summary["n_fail"],
            "original_failures": original_fail,
            "original_failures_now_passing": now_pass,
            "pass_rate_among_original_failures": summary["reeval"][
                "pass_rate_among_original_failures"],
            "max_abs_sigma": summary["max_abs_sigma"],
            "max_abs_sigma_check": summary["max_abs_sigma_check"],
            "residual_failures": len(summary["residuals"]),
            "by_pair_class": {
                class_key: {
                    "applied": stats["applied"],
                    "pass": stats["pass"],
                    "fail": stats["fail"],
                    "pass_rate_all_applied": stats[
                        "pass_rate_all_applied"],
                    "original_failures": stats["original_failures"],
                    "original_failures_now_passing": stats[
                        "original_failures_now_passing"],
                    "pass_rate_among_original_failures": stats[
                        "pass_rate_among_original_failures"],
                }
                for class_key, stats in rates.items()
            },
        }
        combined["applied"] += summary["n_checks_applied"]
        combined["pass"] += summary["n_pass"]
        combined["fail"] += summary["n_fail"]
        combined["original_failures"] += original_fail
        combined["original_failures_now_passing"] += now_pass
        combined["residual_failures"] += len(summary["residuals"])
        if summary["max_abs_sigma"] > combined["max_abs_sigma"]:
            combined["max_abs_sigma"] = summary["max_abs_sigma"]
            combined["max_abs_sigma_check"] = (
                f"{payload['run']} {summary['max_abs_sigma_check']}")
        for class_key, stats in rates.items():
            bucket = combined["by_pair_class"][class_key]
            bucket["original_failures"] += stats["original_failures"]
            bucket["original_failures_now_passing"] += stats[
                "original_failures_now_passing"]
            bucket["residual_failures"] += stats["fail"]
        for residual in summary["residuals"]:
            findings.append({
                "run": payload["run"],
                "pair": residual["pair"],
                "species": residual["species"],
                "observable": residual["observable"],
                "pair_class": residual["pair_class"],
                "sigma": residual["sigma"],
                "residual": residual["residual"],
                "diagnosis": residual["diagnosis"],
            })
    combined["pass_rate_among_original_failures"] = (
        combined["original_failures_now_passing"]
        / combined["original_failures"]
        if combined["original_failures"] else None)
    combined["pass_rate_all_applied"] = (
        combined["pass"] / combined["applied"]
        if combined["applied"] else None)
    for class_key, bucket in combined["by_pair_class"].items():
        bucket["pass_rate_among_original_failures"] = (
            bucket["original_failures_now_passing"]
            / bucket["original_failures"]
            if bucket["original_failures"] else None)
    achieved = combined["pass_rate_among_original_failures"]
    return {
        "schema": REPORT_SCHEMA,
        "card": CARD_ID,
        "gate": "gate 2 (plan section 8, line 575)",
        "clause": "threshold_discontinuity",
        "revision": {
            "from": "the R21/R22 pre-registered metric applied "
                    "delta_analytic = 0 at open-set-unchanged pairs",
            "to": "the full analytic difference delta_analytic_full on "
                  "both pair classes (the open-set-change contribution "
                  "plus the analytic momentum drift)",
            "record": "_knowledge/operations/"
                      "round2-closure-2026-09-17.md, Gate 2 clause 3 "
                      "REVISE (follow-up 2)",
            "new_run_metric": (
                "the card-9 driver staged_source_grid.py carries "
                "DISCONTINUITY_METRIC (the re-registered metric) for "
                "every new source-grid run, including R31 (card 8)"
            ),
        },
        "tolerance": {
            "sigma": N_SIGMA,
            "unchanged": True,
            "se_combined": "sqrt(SE_k^2 + SE_{k+1}^2)",
            "relaxed": False,
        },
        "inputs": {
            "rereval_r21_sha256_note": (
                "the rereval payloads are pure functions of the sealed "
                "gate payloads; their digests are pinned by the run "
                "dir seal"
            ),
        },
        "runs": runs,
        "combined": combined,
        "acceptance": {
            "criterion": (
                "at least 95% of the checks that failed under the "
                "original metric (169 R21 + 154 R22 = 323) pass at "
                "3.0 sigma under delta_analytic_full"
            ),
            "threshold": ACCEPTANCE_RATE,
            "achieved": achieved,
            "pass": bool(achieved is not None
                         and achieved >= ACCEPTANCE_RATE),
        },
        "residual_findings": findings,
        "clause_status": (
            "the metric is re-registered on both pair classes; the "
            "residual failing pairs are named findings with one-line "
            "diagnoses (the open-set-change residuals are the genuine "
            "residuals expected by the Round 2 closure clause 3); no "
            "residual is absorbed into the metric and no tolerance is "
            "relaxed"
        ),
        "no_fitting": True,
    }


def _run_report(run_dir: Path) -> None:
    _require_pre_registration(run_dir)
    payloads = {}
    for key in ("r21", "r22"):
        path = run_dir / f"rereval_{key}.json"
        if not path.is_file():
            raise RuntimeError(
                f"rereval_threshold_checks: {path} is missing; run "
                "--rereval for both archives before --report"
            )
        payloads[key] = json.loads(path.read_text(encoding="utf-8"))
    report = build_revised_threshold(payloads["r21"], payloads["r22"])
    out_path = run_dir / "gate2_revised_threshold.json"
    _write_json(out_path, report)
    achieved = report["acceptance"]["achieved"]
    print(
        f"rereval_threshold_checks: wrote {out_path} "
        f"(combined: {report['combined']['original_failures']} "
        f"original failures -> "
        f"{report['combined']['original_failures_now_passing']} now "
        f"passing ({achieved:.4f}); residual failures "
        f"{report['combined']['residual_failures']}; max |sigma| "
        f"{report['combined']['max_abs_sigma']:.4f} at "
        f"{report['combined']['max_abs_sigma_check']}; acceptance "
        f"pass={report['acceptance']['pass']})"
    )


# ---------------------------------------------------------------------------
# Seal and clean-environment validation (the card 1 driver's pattern)
# ---------------------------------------------------------------------------
_SEAL_REQUIRED = (
    "pre_registration.json",
    "provenance.json",
    "rereval_r21.json",
    "rereval_r22.json",
    "gate2_revised_threshold.json",
)


def _run_seal(run_dir: Path) -> None:
    for required in _SEAL_REQUIRED:
        if not (run_dir / required).is_file():
            raise RuntimeError(
                f"rereval_threshold_checks: {run_dir / required} is "
                "missing; the seal requires the full evidence chain "
                "(pre-registration, re-evaluation, report)"
            )
    sums_path = run_dir / "SHA256SUMS.txt"
    manifest_path = run_dir / "seal_manifest.json"
    if sums_path.is_file() or manifest_path.is_file():
        raise RuntimeError(
            f"rereval_threshold_checks: {run_dir} is already sealed; "
            "the run dir is immutable after the seal"
        )
    entries: list[tuple[str, str]] = []
    total_bytes = 0
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(run_dir).as_posix()
        if rel in ("seal_manifest.json", "SHA256SUMS.txt"):
            continue
        entries.append((rel, _sha256_file(path)))
        total_bytes += path.stat().st_size
    with open(sums_path, "w", encoding="utf-8", newline="\n") as handle:
        for rel, digest in entries:
            handle.write(f"{digest}  {rel}\n")
    manifest = {
        "schema": SEAL_SCHEMA,
        "run": RUN_ID,
        "card": CARD_ID,
        "created_utc": _now_utc(),
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "sha256_sums_file": "SHA256SUMS.txt",
        "sha256_sums_sha256": _sha256_file(sums_path),
        "excluded_from_sums": ["seal_manifest.json", "SHA256SUMS.txt"],
        "note": "the run dir is immutable after the seal (the Round 1 "
                "sealing discipline); clean_environment_commands.json "
                "is the post-seal validation record, written after "
                "this manifest and never edited",
    }
    _write_json(manifest_path, manifest)
    print(
        f"rereval_threshold_checks: sealed {run_dir} "
        f"({len(entries)} files, {total_bytes} bytes, SHA256SUMS.txt "
        f"sha256 {manifest['sha256_sums_sha256']})"
    )


#: The clean-interpreter probe: imports this driver module fresh,
#: re-runs the re-evaluation and the report assembly over the sealed
#: run dir and the sealed archives, and prints the canonical payload
#: hashes (it writes nothing).
_CLEAN_PROBE_SOURCE = """\
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

driver_path, run_dir = sys.argv[1], sys.argv[2]
run_dir = Path(run_dir)
spec = importlib.util.spec_from_file_location(
    "rereval_threshold_checks_clean_probe", driver_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _canon(payload):
    return (json.dumps(payload, indent=2) + "\\n").encode("utf-8")


hashes = {}
payloads = {}
for key in ("r21", "r22"):
    spec_ = module.ARCHIVES[key]
    archive = module._archive_dir(key)
    gate_path = archive / spec_["gate"]
    gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
    payload = module.rereval_report(
        gate_payload,
        run=spec_["run"],
        archive_rel=f"dev/00runs/{spec_['name']}/{spec_['gate']}",
        gate_sha256=module._sha256_file(gate_path),
        card=module.CARD_ID,
    )
    payloads[key] = payload
    hashes[f"rereval_{key}.json"] = hashlib.sha256(
        _canon(payload)).hexdigest()
report = module.build_revised_threshold(payloads["r21"], payloads["r22"])
hashes["gate2_revised_threshold.json"] = hashlib.sha256(
    _canon(report)).hexdigest()
print(json.dumps({
    "hashes": hashes,
    "environment": module.environment_fingerprint(),
}))
"""


def _run_clean_validate(run_dir: Path) -> None:
    out_path = run_dir / "clean_environment_commands.json"
    if out_path.is_file():
        raise RuntimeError(
            f"rereval_threshold_checks: {out_path} exists; the "
            "clean-environment validation is recorded once per seal"
        )
    sums_path = run_dir / "SHA256SUMS.txt"
    manifest_path = run_dir / "seal_manifest.json"
    if not sums_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(
            f"rereval_threshold_checks: {run_dir} is not sealed; run "
            "--seal before the clean-environment validation"
        )
    checked = 0
    mismatches: list[str] = []
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        path = run_dir / rel
        if not path.is_file() or _sha256_file(path) != digest:
            mismatches.append(rel)
        checked += 1
    if mismatches:
        raise RuntimeError(
            "rereval_threshold_checks: the seal verification failed "
            "for: " + ", ".join(mismatches)
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sha256_sums_sha256") != _sha256_file(sums_path):
        raise RuntimeError(
            "rereval_threshold_checks: the seal manifest's "
            "SHA256SUMS.txt digest does not match the sums file"
        )
    sealed = {
        name: _sha256_file(run_dir / name)
        for name in (
            "rereval_r21.json",
            "rereval_r22.json",
            "gate2_revised_threshold.json",
        )
    }
    driver_path = Path(__file__).resolve()
    argv = [
        sys.executable, "-c", _CLEAN_PROBE_SOURCE,
        str(driver_path), str(run_dir),
    ]
    completed = subprocess.run(argv, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "rereval_threshold_checks: the clean-interpreter probe "
            "failed:\n" + completed.stderr
        )
    probe_line = completed.stdout.strip().splitlines()[-1]
    probe = json.loads(probe_line)
    recomputed = {
        str(name): str(digest)
        for name, digest in probe["hashes"].items()
    }
    hashes_match = sealed == recomputed
    payload = {
        "schema": CLEAN_SCHEMA,
        "run": RUN_ID,
        "card": CARD_ID,
        "verified_utc": _now_utc(),
        "seal_verification": {
            "files_checked": checked,
            "mismatches": [],
        },
        "commands": [
            {
                "purpose": "clean-interpreter re-run of the "
                           "re-evaluation and the report assembly over "
                           "the sealed archives (computes the rereval "
                           "and revised-threshold payloads and prints "
                           "their hashes; writes nothing; the "
                           "engine-crosscheck payloads are "
                           "deterministic functions of the sealed "
                           "archives and the committed module and are "
                           "covered by the seal digests)",
                "argv": argv,
                "cwd": str(Path.cwd()),
            }
        ],
        "environment": environment_fingerprint(),
        "probe_environment": probe.get("environment", {}),
        "sealed_report_sha256": sealed,
        "recomputed_report_sha256": recomputed,
        "hashes_match": bool(hashes_match),
    }
    _write_json(out_path, payload)
    if not hashes_match:
        raise RuntimeError(
            "rereval_threshold_checks: the clean-interpreter re-run "
            "does not reproduce the sealed report hashes (sealed "
            f"{sealed}, recomputed {recomputed}); the record is in "
            f"{out_path}"
        )
    print(
        "rereval_threshold_checks: clean-environment validation "
        f"passed ({checked} sealed files verified; the re-run from a "
        "clean interpreter reproduces the sealed rereval and "
        f"revised-threshold hashes); record at {out_path}"
    )


# ---------------------------------------------------------------------------
# Card 8a phase 2: the engine-fix live re-evaluation
# (TASK-20260918-r3-card2-engine-three-body-moment-fix).  Additive modes:
# the card-2 modes above are untouched; the live modes re-evaluate the
# FIXED metric engine's delta_analytic_full (recomputed by
# na49_mixing.threshold_metric.full_analytic_difference at sha256
# ee240f44..., the card-8a phase-1 engine) against the sealed stored
# delta_observed of the sealed R21/R22 archives and of the R31 stopped
# matching-verification artifacts, at the unchanged 3.0-sigma tolerance,
# with the pre-registered deterministic-SE floor convention.
# ---------------------------------------------------------------------------
LIVE_RUN_ID = "R3-THRESHOLD-METRIC-ENGINE-FIX"
LIVE_CARD_ID = "TASK-20260918-r3-card2-engine-three-body-moment-fix"
LIVE_RUN_NAME = "r3_threshold_metric_engine_fix_2026-09-18"
LIVE_RUN_DIR_REL = f"dev/00runs/{LIVE_RUN_NAME}"
LIVE_DEFAULT_RUN_DIR = MONO_ROOT / "dev" / "00runs" / LIVE_RUN_NAME
LIVE_MASTER_SEED = 20260918
LIVE_DRAWS = 0

#: The sealed-quality R31 stopped-run artifacts (read-only; the run
#: stopped at the matching verification per the card-8 stop rule, so the
#: dir carries no seal; the artifacts are verified by digest).
R31_RUN_NAME = "r31_catalog_continuum_matching_scan_2026-09-17"
R31_CARD_ID = "TASK-20260917-r3-run-r31-catalog-continuum-matching-scan"
R31_VERIFICATION_NAME = "matching_verification.json"
R31_PRE_REGISTRATION_NAME = "pre_registration.json"
R31_DIR = MONO_ROOT / "dev" / "00runs" / R31_RUN_NAME
N_R31_APPLIED = 422
N_R31_ORIGINAL_FAILURES = 14

#: The sealed card-2 re-evaluation dir (the baseline; read-only).
BASELINE_RUN_NAME = "r3_threshold_metric_reregistration_2026-09-17"
BASELINE_DIR = MONO_ROOT / "dev" / "00runs" / BASELINE_RUN_NAME
BASELINE_REPORT_NAME = "gate2_revised_threshold.json"

#: The sealed baseline acceptance numbers (the card-2 report's combined
#: block; asserted against the sealed report at live-report time).
N_APPLIED_R21 = 351
N_APPLIED_R22 = 483
N_APPLIED_R21R22 = 834
N_ORIGINAL_FAILURES_R21 = 169
N_ORIGINAL_FAILURES_R22 = 154
N_ORIGINAL_FAILURES_R21R22 = 323
SEALED_RESIDUAL_FAILURES = 11
SEALED_PASS_RATE = 0.9659442724458205
SEALED_MAX_ABS_SIGMA = 4.984844257802041

#: The four sealed moment-approximation-floor residual checks (the T2
#: clause; all four are sealed-fail rows on their archives).
FLOOR_RESIDUAL_KEYS = (
    ("r21", (1, 2), "pi_0", "mean_pt_y"),
    ("r21", (3, 4), "pi_0", "mean_pt_y"),
    ("r21", (15, 16), "pi_0", "mean_p"),
    ("r22", (23, 24), "pi_0", "mean_pt_y"),
)

#: The live evaluation's applied statuses (the 834/422 applied checks of
#: the sealed metric; the sealed ``vacuous_zero`` and ``absent_species``
#: rows are carried, never re-verdicted).
LIVE_APPLIED_STATUSES = ("pass", "fail")

LIVE_PRE_SCHEMA = (
    "na49_mixing.rereval_threshold_checks.live_pre_registration/1"
)
LIVE_PROV_SCHEMA = (
    "na49_mixing.rereval_threshold_checks.live_provenance/1"
)
LIVE_REREVAL_SCHEMA = "na49_mixing.rereval_threshold_checks.live_reeval/1"
LIVE_REPORT_SCHEMA = (
    "na49_mixing.rereval_threshold_checks.revised_clause_v2/1"
)

LIVE_SEAL_REQUIRED = (
    "pre_registration.json",
    "provenance.json",
    "live_reeval_r21.json",
    "live_reeval_r22.json",
    "live_reeval_r31_targeted.json",
    "gate2_revised_threshold_v2.json",
)

#: The engine mechanisms fixed by the card-8a phase-1 commit (recorded
#: by name and rule in the pre-registration, before any evaluation).
LIVE_ENGINE_MECHANISMS = {
    "a_exact_kallen_three_body_slots": (
        "_three_body_slot_nodes computes the exact on-shell two-body "
        "kinematics of the virtual leading pair: the Kallen decay "
        "momentum q_bc = sqrt((Mbc^2 - (mb + mc)^2)(Mbc^2 - (mb - "
        "mc)^2)) / (2 Mbc) and the exact pair-rest energies e_b = "
        "(Mbc^2 + mb^2 - mc^2) / (2 Mbc), e_c = (Mbc^2 + mc^2 - mb^2) "
        "/ (2 Mbc) in slots 0 and 1; on the mb == mc branch the "
        "committed equal-mass expressions q_bc = sqrt(Mbc^2/4 - mb^2) "
        "and e_b = e_c = 0.5 * Mbc are evaluated verbatim, so every "
        "equal-mass three-body channel evaluates byte-identically to "
        "the committed engine; the partner slot is unchanged"
    ),
    "b_deep_level_daughter_mass_integration": (
        "the depth-0 daughter-mass quadrature (the resonance daughter's "
        "budget window [mass_range[0], min(mass_range[1], M_parent - "
        "m_partner)] on the fixed DAUGHTER_GL_NODES grid, the card-3 "
        "density and the kernel-exact open-object weights at every "
        "node, the stable partner's lab motion joining the same "
        "quadrature) applies at every resonance level, replacing the "
        "concentration-point collapse of deep levels; the narrow "
        "line-shape treatment is retained (the draw-cell concentration "
        "for the eta and eta-prime needles, the two-resonance channels' "
        "collapsed needle evaluation); the committed catalog carries no "
        "resonance daughter below the first decay level, so no current "
        "channel changes and the defensive _MAX_NODES bound is "
        "untouched (the recorded rationale of the phase-1 commit)"
    ),
    "c_deterministic_se_floor": (
        "DETERMINISTIC_SE_FLOOR = 1e-12 (equal to the documented "
        "vacuous-zero threshold ZERO_RESIDUAL_TOL): a between-replica "
        "SE_combined collapsing below the floor on a deterministic "
        "observable is floored at the deterministic precision of the "
        "stored sums and the vacuous-zero rule applies at the floor -- "
        "a residual at or below ZERO_RESIDUAL_TOL is a vacuous pass "
        "(sigma 0.0), a larger residual evaluates at the floored SE "
        "under the unchanged 3.0-sigma band; SIGMA_TOLERANCE stays 3.0 "
        "and no pair is special-cased"
    ),
}

LIVE_METRIC_STATEMENT = (
    "for every adjacent pair (k, k+1) and every stable species, with "
    "SE_combined = sqrt(SE_k^2 + SE_{k+1}^2): "
    "|mu(k+1) - mu(k) - delta_analytic_full| <= 3 SE_combined on BOTH "
    "pair classes (open-set-change and open-set-unchanged), for the "
    "species multiplicity and the per-species momentum moments <p> and "
    "<p_T>_y; at the R31 transition-band pairs the routed flat-mixture "
    "expectation mu_routed(M) = (1 - w(M)) mu_catalog(M) + w(M) "
    "mu_continuum(M) replaces mu_catalog, exactly as the R31 driver "
    "applies it; the 3.0-sigma tolerance is never relaxed"
)


def _live_metric_module_sha256() -> str:
    return _sha256_file(NESTED_ROOT / "na49_mixing" / "threshold_metric.py")


def _build_live_pre_registration() -> dict:
    """The card-8a phase-2 pre-registration payload (written before any
    live re-evaluation computation; asserted against the card)."""
    return {
        "schema": LIVE_PRE_SCHEMA,
        "run": LIVE_RUN_ID,
        "card": LIVE_CARD_ID,
        "phase": 2,
        "mode": (
            "live re-evaluation of the fixed engine over the sealed "
            "archives (no Monte Carlo, no generation; the fixed "
            "engine's delta_analytic_full is recomputed and evaluated "
            "against the sealed stored delta_observed)"
        ),
        "master_seed": LIVE_MASTER_SEED,
        "draws": LIVE_DRAWS,
        "seed_note": (
            "no random draws exist on any code path; the master seed "
            "is recorded with zero draws for contract uniformity with "
            "the Round 3 seed rule"
        ),
        "engine": {
            "module": "na49_mixing.threshold_metric",
            "module_sha256_at_registration": _live_metric_module_sha256(),
            "sha256": (
                "ee240f44645d2e592acfae462f7d1e9c3b3dd8cd78c96bd40b7b"
                "73509f7fcc63"
            ),
            "phase_1_commit": "95096d2",
            "mechanisms": dict(LIVE_ENGINE_MECHANISMS),
        },
        "metric": {
            "id": THRESHOLD_METRIC_ID,
            "statement": LIVE_METRIC_STATEMENT,
            "tolerance_sigma": N_SIGMA,
            "tolerance_unchanged": True,
            "se_conventions": {
                "multiplicity": (
                    "the per-node pooled standard error of the mean "
                    "(the card-9 stored 'se')"
                ),
                "moments": (
                    "the between-replica standard error of the four "
                    "stored replica moments (std, ddof=1, over sqrt(4))"
                ),
                "se_combined": "sqrt(SE_k^2 + SE_{k+1}^2)",
            },
            "deterministic_se_floor": {
                "value": DETERMINISTIC_SE_FLOOR,
                "rule": (
                    "a between-replica SE_combined below the floor is "
                    "floored at the floor and the vacuous-zero rule "
                    "applies at the floor: a residual at or below "
                    "1e-12 is a vacuous pass (sigma 0.0), a larger "
                    "residual evaluates at the floored SE under the "
                    "unchanged 3.0-sigma band"
                ),
                "vacuous_zero_threshold": ZERO_RESIDUAL_TOL,
                "tolerance_unchanged": True,
                "r21_r22_note": (
                    "every applied R21/R22 check carries SE_combined at "
                    "1e-4 or above, so the floor is a no-op on the "
                    "834 applied card-2 checks"
                ),
            },
        },
        "archives": {
            key: {
                "run": spec["run"],
                "path": f"dev/00runs/{spec['name']}",
                "gate_file": spec["gate"],
                "card": spec["card"],
                "read_only": True,
            }
            for key, spec in ARCHIVES.items()
        },
        "r31_targeted": {
            "run": "R31",
            "path": f"dev/00runs/{R31_RUN_NAME}",
            "verification_file": R31_VERIFICATION_NAME,
            "card": R31_CARD_ID,
            "read_only": True,
            "statistics_note": (
                "the original 100,000-decay-per-node T1 statistics of "
                "the stopped card-8 matching verification (the 3 x 5 x "
                "4 x 25,000 sealed-quality artifacts), not the 8c "
                "confirmation statistics"
            ),
        },
        "baseline": {
            "run": RUN_ID,
            "path": f"dev/00runs/{BASELINE_RUN_NAME}",
            "report_file": BASELINE_REPORT_NAME,
            "read_only": True,
            "sealed_residual_failures": SEALED_RESIDUAL_FAILURES,
            "pass_rate_among_original_failures": SEALED_PASS_RATE,
            "max_abs_sigma": SEALED_MAX_ABS_SIGMA,
        },
        "coverage": {
            "r21": {
                "adjacent_pairs": 19,
                "applied_checks_original": N_APPLIED_R21,
                "failed_original": N_ORIGINAL_FAILURES_R21,
            },
            "r22": {
                "adjacent_pairs": 27,
                "applied_checks_original": N_APPLIED_R22,
                "failed_original": N_ORIGINAL_FAILURES_R22,
            },
            "r31": {
                "adjacent_pairs": 12,
                "applied_checks": N_R31_APPLIED,
                "failed_original": N_R31_ORIGINAL_FAILURES,
            },
            "observables": list(OBSERVABLES),
            "pair_classes": ["open_set_change", "open_set_unchanged"],
            "total_original_failures_r21_r22": N_ORIGINAL_FAILURES_R21R22,
            "applied_total_r21_r22": N_APPLIED_R21R22,
        },
        "acceptance": {
            "t1_targeted_r31": {
                "criterion": (
                    "all 14 R31 momentum-moment checks that failed at "
                    "the stopped verification pass at 3.0 sigma under "
                    "the fixed engine at the original statistics, "
                    "including the window-1 pi_plus mean_pt_y pair (2, "
                    "3) (global pair 6, boundary-straddling, +7.23 "
                    "sigma pre-fix) and pair (3, 4) (global pair 7, "
                    "interior, -5.61 sigma pre-fix), and the two "
                    "float-precision SE artifact checks (K0, Sigma+ "
                    "mean_p, window 0 pair (3, 4)) under the "
                    "pre-registered SE convention"
                ),
                "n_original_failures": N_R31_ORIGINAL_FAILURES,
                "required": "n_fail_live == 0",
            },
            "t2_card2_reeval": {
                "criterion": (
                    "under the fixed engine at 3.0 sigma over all 834 "
                    "applied R21/R22 checks, the failure count is "
                    "strictly below the sealed re-evaluation's 11 "
                    "residual failures; the sealed "
                    "moment-approximation-floor residuals (R21 pair "
                    "(1, 2) pi_0 mean_pt_y 3.36 sigma, R21 pair (3, 4) "
                    "pi_0 mean_pt_y 3.59 sigma, R21 pair (15, 16) pi_0 "
                    "mean_p 3.08 sigma, R22 pair (23, 24) pi_0 "
                    "mean_pt_y 4.98 sigma) all pass; the pass rate "
                    "among the original 323 failures is at least the "
                    "sealed 0.9659 (no regression); every remaining "
                    "residual is named in the report with its sigma "
                    "and a one-line diagnosis"
                ),
                "n_original_failures": N_ORIGINAL_FAILURES_R21R22,
                "floor_residual_checks": [
                    {
                        "run": run_key.upper(),
                        "pair": list(pair),
                        "species": species,
                        "observable": observable,
                    }
                    for run_key, pair, species, observable
                    in FLOOR_RESIDUAL_KEYS
                ],
                "required": (
                    "n_fail_live < 11 and all four floor residuals "
                    "pass and pass_rate >= 0.9659"
                ),
            },
            "sealed_dir_untouched": (
                "the sealed card-2 dir "
                "dev/00runs/r3_threshold_metric_reregistration_"
                "2026-09-17/ and its pin test stay byte-untouched "
                "(read-only; its seal must still verify)"
            ),
        },
        "constraints": {
            "no_fitting": True,
            "sealed_archives_read_only": True,
            "tolerance_relaxed": False,
            "no_pair_special_casing": True,
            "residual_policy": (
                "every residual failing pair is named with its sigma "
                "and a one-line diagnosis; a residual that cannot be "
                "diagnosed from the sealed data re-opens the "
                "originating kernel card; it is never absorbed into "
                "the metric"
            ),
        },
        "outputs": [
            "pre_registration.json",
            "provenance.json",
            "live_reeval_r21.json",
            "live_reeval_r22.json",
            "live_reeval_r31_targeted.json",
            "gate2_revised_threshold_v2.json",
            "clean_environment_commands.json",
            "seal_manifest.json",
            "SHA256SUMS.txt",
        ],
        "written_before_evaluation": True,
    }


def _assert_live_pre_registration(run_dir: Path) -> dict:
    """Assert the written live pre-registration against this driver's
    constants (a mismatch invalidates the run)."""
    pre = json.loads(
        (run_dir / "pre_registration.json").read_text(encoding="utf-8"))
    problems: list[str] = []
    if pre.get("card") != LIVE_CARD_ID:
        problems.append("card mismatch")
    if pre.get("master_seed") != LIVE_MASTER_SEED:
        problems.append("master seed mismatch")
    if pre.get("draws") != LIVE_DRAWS:
        problems.append("draws mismatch")
    metric = pre.get("metric", {})
    if float(metric.get("tolerance_sigma", -1.0)) != N_SIGMA:
        problems.append("metric tolerance mismatch")
    floor = metric.get("deterministic_se_floor", {})
    if float(floor.get("value", -1.0)) != DETERMINISTIC_SE_FLOOR:
        problems.append("deterministic SE floor value mismatch")
    coverage = pre.get("coverage", {})
    for key, applied, failed in (
        ("r21", N_APPLIED_R21, N_ORIGINAL_FAILURES_R21),
        ("r22", N_APPLIED_R22, N_ORIGINAL_FAILURES_R22),
    ):
        block = coverage.get(key, {})
        if int(block.get("applied_checks_original", -1)) != applied:
            problems.append(f"coverage {key} applied mismatch")
        if int(block.get("failed_original", -1)) != failed:
            problems.append(f"coverage {key} original failures mismatch")
    r31_block = coverage.get("r31", {})
    if int(r31_block.get("applied_checks", -1)) != N_R31_APPLIED:
        problems.append("coverage r31 applied mismatch")
    if int(r31_block.get("failed_original", -1)) != (
            N_R31_ORIGINAL_FAILURES):
        problems.append("coverage r31 original failures mismatch")
    baseline = pre.get("baseline", {})
    if int(baseline.get("sealed_residual_failures", -1)) != (
            SEALED_RESIDUAL_FAILURES):
        problems.append("sealed baseline residual count mismatch")
    if pre.get("written_before_evaluation") is not True:
        problems.append("written_before_evaluation flag mismatch")
    if problems:
        raise RuntimeError(
            "rereval_threshold_checks: the live pre-registration does "
            "not match this driver (the run is invalidated):\n  "
            + "\n  ".join(problems)
        )
    return pre


def _build_live_provenance(run_dir: Path) -> dict:
    """The card-8a phase-2 provenance (written with the pre-registration,
    before any live re-evaluation computation)."""
    gate_sha256s = {
        key: _sha256_file(_archive_dir(key) / spec["gate"])
        for key, spec in ARCHIVES.items()
    }
    return {
        "schema": LIVE_PROV_SCHEMA,
        "run": LIVE_RUN_ID,
        "card": LIVE_CARD_ID,
        "phase": 2,
        "written_utc": _now_utc(),
        "master_seed": LIVE_MASTER_SEED,
        "draws": LIVE_DRAWS,
        "engine": {
            "module": "dev/mixing2006/na49_mixing/threshold_metric.py",
            "module_sha256": _live_metric_module_sha256(),
            "phase_1_commit": "95096d2",
        },
        "catalog": {
            "path": CATALOG_REL,
            "sha256": _sha256_file(NESTED_ROOT / CATALOG_REL),
        },
        "sealed_archives": {
            key: {
                "path": f"dev/00runs/{spec['name']}",
                "gate_file": spec["gate"],
                "gate_sha256": gate_sha256s[key],
                "seal": _verify_archive_seal(_archive_dir(key)),
            }
            for key, spec in ARCHIVES.items()
        },
        "r31_artifacts": {
            "path": f"dev/00runs/{R31_RUN_NAME}",
            "verification_file": R31_VERIFICATION_NAME,
            "verification_sha256": _sha256_file(
                R31_DIR / R31_VERIFICATION_NAME),
            "pre_registration_sha256": _sha256_file(
                R31_DIR / R31_PRE_REGISTRATION_NAME),
            "card": R31_CARD_ID,
            "seal": None,
            "seal_note": (
                "the R31 dir is the card-8 stopped-run artifact (the "
                "run stopped at the matching verification per the "
                "card's stop rule); it carries no seal and is "
                "verified here by digest"
            ),
        },
        "baseline": {
            "path": f"dev/00runs/{BASELINE_RUN_NAME}",
            "report_file": BASELINE_REPORT_NAME,
            "report_sha256": _sha256_file(
                BASELINE_DIR / BASELINE_REPORT_NAME),
        },
        "drivers": {
            "rereval_driver": "helpers/benchmarking/"
                              "rereval_threshold_checks.py",
            "rereval_driver_sha256": _sha256_file(
                Path(__file__).resolve()),
        },
        "git": _git_info(),
        "environment": environment_fingerprint(),
    }


def _run_live_pre_registration(run_dir: Path) -> None:
    run_dir = Path(run_dir)
    pre_path = run_dir / "pre_registration.json"
    prov_path = run_dir / "provenance.json"
    if (run_dir / "seal_manifest.json").is_file():
        raise RuntimeError(
            "rereval_threshold_checks: "
            f"{run_dir} is already sealed; the live pre-registration "
            "refuses to write into a sealed dir"
        )
    if run_dir.resolve() == BASELINE_DIR.resolve():
        raise RuntimeError(
            "rereval_threshold_checks: the live pre-registration "
            f"refuses to write into the sealed baseline dir {BASELINE_DIR}"
        )
    if pre_path.is_file() and any(
            (run_dir / name).is_file() for name in LIVE_SEAL_REQUIRED[2:]):
        raise RuntimeError(
            f"rereval_threshold_checks: {pre_path} exists and live "
            "evaluation artifacts are present; the live "
            "pre-registration is immutable after evaluation has "
            "started (a change invalidates the run; re-run under a "
            "new date)"
        )
    pre = _build_live_pre_registration()
    module_sha = pre["engine"]["module_sha256_at_registration"]
    if module_sha != pre["engine"]["sha256"]:
        raise RuntimeError(
            "rereval_threshold_checks: the on-disk metric module "
            f"sha256 {module_sha} differs from the pre-registered "
            f"engine-fix digest {pre['engine']['sha256']}; the card-8a "
            "phase-1 engine is the contract"
        )
    _write_json(pre_path, pre)
    _write_json(prov_path, _build_live_provenance(run_dir))
    print(
        f"rereval_threshold_checks: wrote {pre_path} and {prov_path} "
        f"(master seed {LIVE_MASTER_SEED}, draws {LIVE_DRAWS}; the "
        f"fixed engine sha256 {module_sha[:8]}...; nothing computed)"
    )


def _live_residual_diagnosis(residual: dict) -> str:
    """The deterministic one-line diagnosis of a live residual failing
    check (the card-2 residual policy).  Open-set-change rows reuse the
    module's classifier verbatim; unchanged-pair rows at the 3.0-sigma
    edge reuse the marginal-excursion text; the larger unchanged-pair
    residuals carry the live-worded moment-approximation-floor text."""
    if residual["pair_class"] == "open_set_change":
        return diagnose_residual(residual)
    pair = residual["pair"]
    skey = residual["species"]
    observable = residual["observable"]
    value = float(residual["sigma"])
    amount = float(residual["residual"])
    where = f"pair({pair[0]},{pair[1]}) {skey} {observable}"
    if value <= 3.2:
        return (
            f"{where}: residual {amount:+.3e} = {value:.2f} sigma; a "
            "marginal statistical excursion at the 3.0-sigma edge (the "
            "per-check 3-sigma exceedance probability 0.27% over the "
            "applied-check count expects about two marginal "
            "exceedances); no kernel defect diagnosed, the metric is "
            "not relaxed"
        )
    return (
        f"{where}: residual {amount:+.3e} = {value:.2f} sigma; the "
        "fixed engine's moment-approximation floor against the "
        "kernel-exact Monte Carlo (the sealed card-2 re-evaluation's "
        "documented residual class; the analytic moments carry "
        "residuals at the 1e-3 scale against the stored moments) "
        "exceeds the 3.0-sigma statistical band at this pair; a "
        "reference-precision residual tracking the analytic drift in "
        "sign and order of magnitude, not a kinematic step; the metric "
        "is not relaxed"
    )


def _live_rereval_payload(key: str) -> dict:
    """The fixed-engine live re-evaluation payload of one sealed
    archive: every applied check of the sealed gate payload is
    re-evaluated against the recomputed delta_analytic_full of
    ``na49_mixing.threshold_metric.full_analytic_difference`` (the
    fixed engine) at the sealed stored ``se_combined`` and the
    unchanged 3.0-sigma tolerance, with the pre-registered
    deterministic-SE floor convention.  Pure function of the sealed
    archive and the committed code (byte-stable; no wall clock)."""
    spec = ARCHIVES[key]
    archive = _archive_dir(key)
    seal = _verify_archive_seal(archive)
    gate_path = archive / spec["gate"]
    gate_payload = json.loads(
        gate_path.read_text(encoding="utf-8"))
    pre = json.loads(
        (archive / "pre_registration.json").read_text(encoding="utf-8"))
    clause = gate_payload["clauses"]["threshold_discontinuity"]
    masses = [float(m) for m in pre["grid_masses_gev"]]
    open_sets = {int(k): v for k, v in pre["open_sets"].items()}
    catalog = HadronCatalog.from_json(NESTED_ROOT / CATALOG_REL)
    union_ids = sorted({pdg for ids in open_sets.values() for pdg in ids})
    record_set = RecordSet(catalog, [catalog.get(p) for p in union_ids])
    delta_cache: dict[tuple, dict] = {}
    max_dev: dict = {obs: 0.0 for obs in OBSERVABLES}
    rows: list[dict] = []
    summary = {
        "n_checks": len(clause["checks"]),
        "n_checks_applied": 0,
        "n_pass": 0,
        "n_fail": 0,
        "n_vacuous_zero_live": 0,
        "n_not_evaluable_live": 0,
        "original": {
            "n_checks_failed": int(
                clause.get("n_checks_failed", 0)),
            "failed_at_open_set_change": int(
                clause.get("n_failed_at_open_set_changes", 0)),
            "failed_at_unchanged": int(
                clause.get("n_failed_at_unchanged_pairs", 0)),
        },
        "reeval": {},
        "max_abs_sigma": 0.0,
        "max_abs_sigma_check": None,
        "engine_vs_sealed_stored": {},
        "floor_residuals": [],
        "residuals": [],
    }
    per_class = {
        "open_set_change": {"applied": 0, "pass": 0, "fail": 0,
                            "vacuous_zero": 0},
        "open_set_unchanged": {"applied": 0, "pass": 0, "fail": 0,
                               "vacuous_zero": 0},
    }
    original_fail_per_class = {
        "open_set_change": 0,
        "open_set_unchanged": 0,
    }
    original_fail_now_pass_per_class = {
        "open_set_change": 0,
        "open_set_unchanged": 0,
    }
    max_sigma = -1.0
    max_key = None
    for check in clause["checks"]:
        status_original = str(check["status"])
        pair = [int(k) for k in check["pair"]]
        skey = str(check["species"])
        observable = str(check["observable"])
        changed = bool(check["open_set_changes"])
        class_key = "open_set_change" if changed else "open_set_unchanged"
        row = {
            "pair": pair,
            "species": skey,
            "observable": observable,
            "open_set_changes": changed,
            "pair_class": class_key,
            "original_status": status_original,
            "scope": "applied",
            "observed": check.get("observed"),
            "delta_observed": check.get("delta_observed"),
            "delta_analytic_full_sealed": check.get("delta_analytic_full"),
            "delta_analytic_full_live": None,
            "se_combined": check.get("se_combined"),
            "sigma_original": check.get("sigma"),
            "sigma": None,
            "status": None,
            "pass": None,
            "se_floor_applied": False,
        }
        rows.append(row)
        if status_original not in LIVE_APPLIED_STATUSES:
            row["scope"] = "not_applied"
            continue
        summary["n_checks_applied"] += 1
        per_class[class_key]["applied"] += 1
        dkey = (pair[0], pair[1], skey)
        fixed_by_obs = delta_cache.get(dkey)
        if fixed_by_obs is None:
            full = full_analytic_difference(
                record_set, masses[pair[0]], masses[pair[1]], skey)
            fixed_by_obs = full["delta_analytic_full"]
            delta_cache[dkey] = fixed_by_obs
        fixed = fixed_by_obs.get(observable)
        sealed_stored = check.get("delta_analytic_full")
        if fixed is not None and sealed_stored is not None:
            deviation = abs(float(fixed) - float(sealed_stored))
            if deviation > max_dev[observable]:
                max_dev[observable] = deviation
        row["delta_analytic_full_live"] = fixed
        if fixed is None:
            row["status"] = "not_evaluable_live"
            row["pass"] = None
            summary["n_not_evaluable_live"] += 1
            continue
        live_check = dict(check)
        live_check["delta_analytic_full"] = fixed
        status, value, passed = _reevaluated_status(live_check)
        row["status"] = status
        row["sigma"] = value
        row["pass"] = passed
        row["se_floor_applied"] = bool(
            (check.get("se_combined") is None
             or float(check["se_combined"]) < DETERMINISTIC_SE_FLOOR)
            and status != "absent_species")
        if status_original == "fail":
            original_fail_per_class[class_key] += 1
            if passed:
                original_fail_now_pass_per_class[class_key] += 1
        if status == "vacuous_zero":
            summary["n_vacuous_zero_live"] += 1
            per_class[class_key]["vacuous_zero"] += 1
            continue
        if passed:
            summary["n_pass"] += 1
            per_class[class_key]["pass"] += 1
        else:
            summary["n_fail"] += 1
            per_class[class_key]["fail"] += 1
            if float(value) > max_sigma:
                max_sigma = float(value)
                max_key = f"pair({pair[0]},{pair[1]}) {skey} {observable}"
            summary["residuals"].append({
                "pair": pair,
                "species": skey,
                "observable": observable,
                "pair_class": class_key,
                "sigma": value,
                "residual": abs(float(check["delta_observed"])
                                - float(fixed)),
                "se_combined": float(check["se_combined"]),
                "sigma_original": check.get("sigma"),
                "delta_analytic_full_live": fixed,
            })
        fkey = (key, (pair[0], pair[1]), skey, observable)
        if fkey in FLOOR_RESIDUAL_KEYS and status_original == "fail":
            summary["floor_residuals"].append({
                "run": key.upper(),
                "pair": [pair[0], pair[1]],
                "species": skey,
                "observable": observable,
                "sealed_sigma": float(check["sigma"]),
                "live_sigma": value,
                "pass": bool(passed),
            })
    n_original_fail = int(clause.get("n_checks_failed", 0))
    now_pass_of_original = sum(
        original_fail_now_pass_per_class.values())
    rate_block = {}
    for class_key, stats in per_class.items():
        rate_block[class_key] = {
            "applied": stats["applied"],
            "pass": stats["pass"],
            "fail": stats["fail"],
            "vacuous_zero_live": stats["vacuous_zero"],
            "original_failures": original_fail_per_class[class_key],
            "original_failures_now_passing":
                original_fail_now_pass_per_class[class_key],
            "pass_rate_among_original_failures": (
                original_fail_now_pass_per_class[class_key]
                / original_fail_per_class[class_key]
                if original_fail_per_class[class_key] else None),
        }
    summary["reeval"] = {
        "original_failures_total": n_original_fail,
        "original_failures_now_passing": now_pass_of_original,
        "pass_rate_among_original_failures": (
            now_pass_of_original / n_original_fail
            if n_original_fail else None),
        "by_pair_class": rate_block,
    }
    summary["max_abs_sigma"] = (max_sigma if max_sigma >= 0.0 else 0.0)
    summary["max_abs_sigma_check"] = max_key
    summary["engine_vs_sealed_stored"] = dict(max_dev)
    for residual in summary["residuals"]:
        residual["diagnosis"] = _live_residual_diagnosis(residual)
    return {
        "schema": LIVE_REREVAL_SCHEMA,
        "card": LIVE_CARD_ID,
        "phase": 2,
        "run": spec["run"],
        "metric": {
            "id": THRESHOLD_METRIC_ID,
            "statement": LIVE_METRIC_STATEMENT,
            "tolerance_sigma": N_SIGMA,
            "tolerance_unchanged": True,
        },
        "engine": {
            "module": "na49_mixing.threshold_metric",
            "function": "full_analytic_difference",
            "module_sha256": _live_metric_module_sha256(),
            "mechanisms": dict(LIVE_ENGINE_MECHANISMS),
        },
        "archive": {
            "path": f"dev/00runs/{spec['name']}",
            "gate_file": spec["gate"],
            "gate_sha256": _sha256_file(gate_path),
            "seal": seal,
        },
        "evaluation_rule": (
            "the fixed engine's recomputed delta_analytic_full against "
            "the sealed stored delta_observed at the sealed stored "
            "se_combined and the unchanged 3.0-sigma tolerance, with "
            "the pre-registered deterministic-SE floor convention; the "
            "sealed vacuous_zero and absent_species rows are carried, "
            "never re-verdicted (the 834 applied checks are the scope)"
        ),
        "coverage": {
            "adjacent_pairs_checked": int(
                clause.get("adjacent_pairs_checked", 0)),
            "open_set_changes": [int(k) for k in
                                 clause.get("open_set_changes", [])],
            "n_checks": len(clause["checks"]),
            "observables": list(OBSERVABLES),
        },
        "summary": summary,
        "checks": rows,
    }


def _run_live_reeval(run_dir: Path, keys: list[str]) -> None:
    _require_live_pre_registration(run_dir)
    _assert_live_pre_registration(run_dir)
    for key in keys:
        payload = _live_rereval_payload(key)
        out_path = run_dir / f"live_reeval_{key}.json"
        _write_json(out_path, payload)
        summary = payload["summary"]
        print(
            f"rereval_threshold_checks: wrote {out_path} "
            f"({payload['run']}: applied {summary['n_checks_applied']}, "
            f"pass {summary['n_pass']}, fail {summary['n_fail']}; "
            f"original failures "
            f"{summary['original']['n_checks_failed']} -> now passing "
            f"{summary['reeval']['original_failures_now_passing']}; "
            f"residuals {len(summary['residuals'])}; max |sigma| "
            f"{summary['max_abs_sigma']:.4f})"
        )


def _r31_window_payloads(catalog, r31drv, families):
    """The per-window analytic payloads of the R31 verification (the
    exact ``_routed_payloads`` construction of the byte-unchanged R31
    driver: the union record set in sorted-pdg order, the catalog
    mixture payload at every node mass, the continuum payload at the
    nodes with a positive routing weight, and the blend weights)."""
    window_payloads = {}
    for spec_window in r31drv.WINDOWS:
        w = int(spec_window["window"])
        qn = tuple(int(v) for v in spec_window["qn"])
        union: dict = {}
        for mass in r31drv.NODE_MASSES:
            for record in r31drv._open_set(catalog, qn, mass):
                union.setdefault(int(record.pdg_id), record)
        record_set = RecordSet(
            catalog, [union[pdg] for pdg in sorted(union)])
        mu_catalog = {}
        for j, mass in enumerate(r31drv.NODE_MASSES):
            mu_catalog[j] = record_set.mixture_payload(
                record_set.open_records(float(mass)), float(mass))
        weights = {j: float(r31drv.SPEC.weight(M))
                   for j, M in enumerate(r31drv.NODE_MASSES)}
        family = r31drv.active_families(families, qn)[0]
        mu_continuum = {}
        for j, mass in enumerate(r31drv.NODE_MASSES):
            if weights[j] <= 0.0:
                continue
            mu_continuum[j] = r31drv.continuum_payload(
                record_set, family, float(mass))
        window_payloads[w] = (record_set, mu_catalog, mu_continuum,
                              weights)
    return window_payloads


def _r31_residual_diagnosis(row: dict) -> str:
    """The deterministic one-line diagnosis of a persisting R31 residual
    failing check (the card-2 residual policy applied to the targeted
    R31 scope)."""
    where = (f"w{row['window']} pair({row['pair'][0]},{row['pair'][1]}) "
             f"{row['species']} {row['observable']}")
    pair_label = (
        "routed blend pair" if row["blend_pair"] else "catalog-only pair")
    return (
        f"{where}: residual {row['residual']:+.3e} = "
        f"{row['sigma']:.2f} sigma (sealed "
        f"{row['sigma_original']:+.2f}, {pair_label}); the committed "
        "engine's moment-approximation floor against the kernel-exact "
        "Monte Carlo at the T1 statistics (the sealed card-2 "
        "re-evaluation's documented residual class), unchanged by the "
        "fixed engine on this record set; recorded as a finding, the "
        "metric is not relaxed"
    )


def _live_rereval_r31_payload() -> dict:
    """The targeted R31 re-evaluation payload: every applied check of
    the sealed matching_verification.json is re-evaluated against the
    fixed engine's recomputed routed delta_analytic_full at the stored
    ``se_combined`` and the unchanged 3.0-sigma tolerance, with the
    pre-registered deterministic-SE floor convention, at the original
    100,000-decay-per-node statistics (the stored between-replica
    standard errors).  Pure function of the sealed artifacts and the
    committed code (byte-stable; no wall clock)."""
    verification_path = R31_DIR / R31_VERIFICATION_NAME
    if not verification_path.is_file():
        raise RuntimeError(
            "rereval_threshold_checks: "
            f"{verification_path} is missing; the targeted R31 "
            "re-evaluation requires the stopped card-8 artifacts"
        )
    verification = json.loads(
        verification_path.read_text(encoding="utf-8"))
    r31_summary = verification.get("summary", {})
    if int(r31_summary.get("checks_applied", -1)) != N_R31_APPLIED:
        raise RuntimeError(
            "rereval_threshold_checks: the matching verification "
            "carries "
            f"{r31_summary.get('checks_applied')!r} applied checks; "
            f"the pre-registered scope is {N_R31_APPLIED}"
        )
    if int(r31_summary.get("checks_failed", -1)) != (
            N_R31_ORIGINAL_FAILURES):
        raise RuntimeError(
            "rereval_threshold_checks: the matching verification "
            "carries "
            f"{r31_summary.get('checks_failed')!r} failed checks; the "
            f"pre-registered scope is {N_R31_ORIGINAL_FAILURES}"
        )
    # The byte-unchanged card-8 R31 driver supplies the routed analytic
    # helpers (continuum_payload, _routed_observable, the blend spec);
    # the fixed engine underneath (threshold_metric) is the card-8a
    # phase-1 module.
    from helpers.benchmarking import (  # noqa: E402
        staged_r31_matching_scan as r31drv,
    )
    catalog = HadronCatalog.from_json(NESTED_ROOT / CATALOG_REL)
    families = r31drv.families_from_json(
        str(NESTED_ROOT / r31drv.FAMILIES_REL))
    window_payloads = _r31_window_payloads(catalog, r31drv, families)
    rows: list[dict] = []
    original_rows: list[dict] = []
    summary = {
        "n_checks_applied": 0,
        "n_pass": 0,
        "n_fail": 0,
        "n_vacuous_zero_live": 0,
        "n_not_evaluable_live": 0,
        "n_absent_carried": 0,
        "original": {
            "checks_failed": N_R31_ORIGINAL_FAILURES,
        },
        "max_abs_sigma": 0.0,
        "max_abs_sigma_check": None,
        "boundary_straddling_pairs": [int(v) for v in
                                      r31drv.BOUNDARY_PAIR_INDICES],
        "residuals": [],
    }
    boundary_applied = boundary_fail = 0
    interior_applied = interior_fail = 0
    max_sigma = -1.0
    max_key = None
    for w_key in sorted(verification["windows"], key=int):
        w = int(w_key)
        record_set, mu_catalog, mu_continuum, weights = (
            window_payloads[w])
        for j_key in sorted(
                verification["windows"][w_key]["pairs"], key=int):
            j = int(j_key)
            for check in verification["windows"][w_key]["pairs"][j_key][
                    "checks"]:
                if str(check.get("status")) not in LIVE_APPLIED_STATUSES:
                    summary["n_absent_carried"] += 1
                    continue
                skey = str(check["species"])
                observable = str(check["observable"])
                w_j, w_jp1 = weights[j], weights[j + 1]
                catalog_only = (w_j <= 0.0) and (w_jp1 <= 0.0)
                if catalog_only:
                    full = full_analytic_difference(
                        record_set, float(r31drv.NODE_MASSES[j]),
                        float(r31drv.NODE_MASSES[j + 1]), skey)
                    fixed = full["delta_analytic_full"].get(observable)
                else:
                    ana_j = r31drv._routed_observable(
                        mu_catalog, mu_continuum, weights, j, skey,
                        observable)
                    ana_jp1 = r31drv._routed_observable(
                        mu_catalog, mu_continuum, weights, j + 1, skey,
                        observable)
                    fixed = None if (ana_j is None or ana_jp1 is None) \
                        else float(ana_jp1) - float(ana_j)
                live_check = dict(check)
                live_check["delta_analytic_full"] = fixed
                status, value, passed = _reevaluated_status(live_check)
                is_boundary = (j == 2)
                row = {
                    "window": w,
                    "pair": [int(j), int(j + 1)],
                    "global_pair": int(check["global_pair"]),
                    "masses_gev": list(check["masses_gev"]),
                    "species": skey,
                    "observable": observable,
                    "blend_pair": bool(not catalog_only),
                    "original_status": str(check["status"]),
                    "observed": check.get("observed"),
                    "delta_observed": check.get("delta_observed"),
                    "delta_analytic_full_sealed": check.get(
                        "delta_analytic_full"),
                    "delta_analytic_full_live": fixed,
                    "se_combined": check.get("se_combined"),
                    "sigma_original": check.get("sigma"),
                    "sigma": value,
                    "status": status,
                    "pass": passed,
                    "se_floor_applied": bool(
                        (check.get("se_combined") is None
                         or float(check["se_combined"])
                         < DETERMINISTIC_SE_FLOOR)),
                }
                rows.append(row)
                summary["n_checks_applied"] += 1
                if is_boundary:
                    boundary_applied += 1
                else:
                    interior_applied += 1
                if str(check["status"]) == "fail":
                    original_rows.append({
                        "window": w,
                        "pair": [int(j), int(j + 1)],
                        "global_pair": int(check["global_pair"]),
                        "masses_gev": list(check["masses_gev"]),
                        "species": skey,
                        "observable": observable,
                        "delta_observed": float(
                            check["delta_observed"]),
                        "delta_analytic_full_sealed": float(
                            check["delta_analytic_full"]),
                        "se_combined": float(check["se_combined"]),
                        "sigma_original": float(check["sigma"]),
                        "delta_analytic_full_live": fixed,
                        "sigma": value,
                        "status": status,
                        "pass": bool(passed),
                    })
                if status == "vacuous_zero":
                    summary["n_vacuous_zero_live"] += 1
                    continue
                if fixed is None:
                    row["status"] = "not_evaluable_live"
                    row["pass"] = None
                    summary["n_not_evaluable_live"] += 1
                    continue
                if passed:
                    summary["n_pass"] += 1
                else:
                    summary["n_fail"] += 1
                    if is_boundary:
                        boundary_fail += 1
                    else:
                        interior_fail += 1
                    if float(value) > max_sigma:
                        max_sigma = float(value)
                        max_key = (f"w{w} pair({j},{j+1}) {skey} "
                                   f"{observable}")
                    summary["residuals"].append({
                        "window": w,
                        "pair": [int(j), int(j + 1)],
                        "global_pair": int(check["global_pair"]),
                        "species": skey,
                        "observable": observable,
                        "sigma": value,
                        "residual": abs(float(check["delta_observed"])
                                        - float(fixed)),
                        "se_combined": float(check["se_combined"]),
                        "sigma_original": float(check["sigma"]),
                        "blend_pair": bool(not catalog_only),
                    })
    summary["max_abs_sigma"] = (max_sigma if max_sigma >= 0.0 else 0.0)
    summary["max_abs_sigma_check"] = max_key
    summary["original_failures"] = original_rows
    summary["original_failures_passing"] = sum(
        1 for row in original_rows if row["pass"])
    summary["original_failures_persisting"] = sum(
        1 for row in original_rows if not row["pass"])
    summary["boundary_interior"] = {
        "boundary_pair_applied": boundary_applied,
        "boundary_pair_failed_live": boundary_fail,
        "interior_pairs_applied": interior_applied,
        "interior_pairs_failed_live": interior_fail,
    }
    for residual in summary["residuals"]:
        residual["diagnosis"] = _r31_residual_diagnosis(residual)
    return {
        "schema": LIVE_REREVAL_SCHEMA,
        "card": LIVE_CARD_ID,
        "phase": 2,
        "run": "R31",
        "scope": "targeted (the 14 original failures tracked "
                 "individually over all 422 applied checks)",
        "metric": {
            "id": THRESHOLD_METRIC_ID,
            "statement": LIVE_METRIC_STATEMENT,
            "tolerance_sigma": N_SIGMA,
            "tolerance_unchanged": True,
        },
        "engine": {
            "module": "na49_mixing.threshold_metric",
            "function": "full_analytic_difference",
            "module_sha256": _live_metric_module_sha256(),
            "mechanisms": dict(LIVE_ENGINE_MECHANISMS),
            "routed_helpers": (
                "the byte-unchanged card-8 R31 driver "
                "(helpers/benchmarking/staged_r31_matching_scan.py) "
                "supplies continuum_payload and _routed_observable; "
                "the routed blend mu_routed(M) = (1 - w(M)) "
                "mu_catalog(M) + w(M) mu_continuum(M) uses the fixed "
                "engine's catalog mixture payload"
            ),
        },
        "archive": {
            "path": f"dev/00runs/{R31_RUN_NAME}",
            "verification_file": R31_VERIFICATION_NAME,
            "verification_sha256": _sha256_file(verification_path),
            "pre_registration_sha256": _sha256_file(
                R31_DIR / R31_PRE_REGISTRATION_NAME),
            "seal": None,
            "seal_note": (
                "the R31 dir is the card-8 stopped-run artifact (no "
                "seal per the card's stop rule); verified by digest"
            ),
        },
        "evaluation_rule": (
            "the fixed engine's recomputed routed delta_analytic_full "
            "against the stored delta_observed at the stored "
            "se_combined and the unchanged 3.0-sigma tolerance, with "
            "the pre-registered deterministic-SE floor convention, at "
            "the original 100,000-decay-per-node statistics"
        ),
        "summary": summary,
        "checks": rows,
    }


def _require_live_pre_registration(run_dir: Path) -> None:
    if not (run_dir / "pre_registration.json").is_file():
        raise RuntimeError(
            "rereval_threshold_checks: "
            f"{run_dir / 'pre_registration.json'} is missing; run "
            "--live-pre-registration before any live evaluation"
        )


def _run_live_reeval_r31(run_dir: Path) -> None:
    _require_live_pre_registration(run_dir)
    _assert_live_pre_registration(run_dir)
    payload = _live_rereval_r31_payload()
    out_path = run_dir / "live_reeval_r31_targeted.json"
    _write_json(out_path, payload)
    summary = payload["summary"]
    print(
        f"rereval_threshold_checks: wrote {out_path} (R31 targeted: "
        f"applied {summary['n_checks_applied']}, pass "
        f"{summary['n_pass']}, vacuous zero "
        f"{summary['n_vacuous_zero_live']}, fail {summary['n_fail']}; "
        f"original failures {summary['original']['checks_failed']} -> "
        f"{summary['original_failures_passing']} passing, "
        f"{summary['original_failures_persisting']} persisting; max "
        f"|sigma| {summary['max_abs_sigma']:.4f})"
    )


def _build_live_report(r21: dict, r22: dict, r31: dict,
                       baseline_report: dict,
                       baseline_sha256: str) -> dict:
    """The improved Gate 2 threshold clause evidence (the card-8a
    phase-2 report): the fixed-engine live values against the sealed
    card-2 baseline, with the acceptance verdicts on the card's T1 and
    T2 criteria (pure function of the payloads and the sealed baseline
    report; byte-stable)."""
    runs = {}
    combined = {
        "applied": 0,
        "pass": 0,
        "fail": 0,
        "original_failures": 0,
        "original_failures_now_passing": 0,
        "residual_failures": 0,
        "max_abs_sigma": 0.0,
        "max_abs_sigma_check": None,
    }
    findings: list[dict] = []
    for key, payload in (("r21", r21), ("r22", r22)):
        summary = payload["summary"]
        original_fail = summary["original"]["n_checks_failed"]
        now_pass = summary["reeval"]["original_failures_now_passing"]
        runs[key] = {
            "run": payload["run"],
            "archive": payload["archive"],
            "n_checks": summary["n_checks"],
            "n_checks_applied": summary["n_checks_applied"],
            "n_pass": summary["n_pass"],
            "n_fail": summary["n_fail"],
            "n_vacuous_zero_live": summary["n_vacuous_zero_live"],
            "original_failures": original_fail,
            "original_failures_now_passing": now_pass,
            "pass_rate_among_original_failures": summary["reeval"][
                "pass_rate_among_original_failures"],
            "max_abs_sigma": summary["max_abs_sigma"],
            "max_abs_sigma_check": summary["max_abs_sigma_check"],
            "residual_failures": len(summary["residuals"]),
            "engine_vs_sealed_stored": summary["engine_vs_sealed_stored"],
            "by_pair_class": {
                class_key: {
                    "applied": stats["applied"],
                    "pass": stats["pass"],
                    "fail": stats["fail"],
                    "vacuous_zero_live": stats["vacuous_zero_live"],
                    "original_failures": stats["original_failures"],
                    "original_failures_now_passing": stats[
                        "original_failures_now_passing"],
                    "pass_rate_among_original_failures": stats[
                        "pass_rate_among_original_failures"],
                }
                for class_key, stats in summary["reeval"][
                    "by_pair_class"].items()
            },
        }
        combined["applied"] += summary["n_checks_applied"]
        combined["pass"] += summary["n_pass"]
        combined["fail"] += summary["n_fail"]
        combined["original_failures"] += original_fail
        combined["original_failures_now_passing"] += now_pass
        combined["residual_failures"] += len(summary["residuals"])
        if summary["max_abs_sigma"] > combined["max_abs_sigma"]:
            combined["max_abs_sigma"] = summary["max_abs_sigma"]
            combined["max_abs_sigma_check"] = (
                f"{payload['run']} {summary['max_abs_sigma_check']}")
        for residual in summary["residuals"]:
            findings.append({
                "run": payload["run"],
                "pair": residual["pair"],
                "species": residual["species"],
                "observable": residual["observable"],
                "pair_class": residual["pair_class"],
                "sigma": residual["sigma"],
                "residual": residual["residual"],
                "sigma_original": residual["sigma_original"],
                "diagnosis": residual["diagnosis"],
            })
    combined["pass_rate_among_original_failures"] = (
        combined["original_failures_now_passing"]
        / combined["original_failures"]
        if combined["original_failures"] else None)
    # The sealed card-2 rereval sigma of each named residual finding
    # (the T2 floor-residual clause quotes these values).
    baseline_findings = {
        (f["run"], tuple(f["pair"]), f["species"],
         f["observable"]): float(f["sigma"])
        for f in baseline_report["residual_findings"]
    }
    # The four sealed moment-approximation-floor residual checks (T2).
    floor_rows = []
    floor_pass = True
    for run_key, pair, species, observable in FLOOR_RESIDUAL_KEYS:
        payload = {"r21": r21, "r22": r22}[run_key]
        row = None
        for candidate in payload["summary"]["residuals"]:
            if (tuple(candidate["pair"]) == pair
                    and candidate["species"] == species
                    and candidate["observable"] == observable):
                row = candidate
                break
        if row is None:
            live_sigma = None
            live_pass = True   # a residual that no longer fails passes
        else:
            live_sigma = float(row["sigma"])
            live_pass = False
        floor_rows.append({
            "run": run_key.upper(),
            "pair": list(pair),
            "species": species,
            "observable": observable,
            "sealed_rereval_sigma": baseline_findings.get(
                (run_key.upper(), pair, species, observable)),
            "live_sigma": live_sigma,
            "pass": bool(live_pass),
        })
        floor_pass = floor_pass and bool(live_pass)
    achieved = combined["pass_rate_among_original_failures"]
    t2_failure_count_pass = (
        combined["fail"] < SEALED_RESIDUAL_FAILURES)
    t2_rate_pass = bool(
        achieved is not None and achieved >= SEALED_PASS_RATE)
    t2_pass = bool(t2_failure_count_pass and floor_pass and t2_rate_pass)
    # The targeted R31 acceptance (T1).
    r31_summary = r31["summary"]
    original_rows = r31_summary["original_failures"]
    t1_pass = bool(r31_summary["original_failures_persisting"] == 0)
    acceptance_pass = bool(t1_pass and t2_pass)
    return {
        "schema": LIVE_REPORT_SCHEMA,
        "card": LIVE_CARD_ID,
        "phase": 2,
        "gate": "gate 2 (plan section 8, line 575)",
        "clause": "threshold_discontinuity",
        "revision": {
            "from": (
                "the sealed card-2 re-evaluation (the re-registered "
                "delta_analytic_full metric evaluated against the "
                "sealed stored delta_analytic_full of the R21/R22 "
                "archives; 11 residual failures, pass rate 0.9659)"
            ),
            "to": (
                "the fixed engine's recomputed delta_analytic_full "
                "(the card-8a phase-1 engine sha256 ee240f44...) "
                "evaluated live against the sealed delta_observed at "
                "the unchanged 3.0-sigma tolerance, with the "
                "pre-registered deterministic-SE floor convention"
            ),
            "engine_mechanisms": dict(LIVE_ENGINE_MECHANISMS),
        },
        "tolerance": {
            "sigma": N_SIGMA,
            "unchanged": True,
            "se_combined": "sqrt(SE_k^2 + SE_{k+1}^2)",
            "deterministic_se_floor": DETERMINISTIC_SE_FLOOR,
            "relaxed": False,
        },
        "sealed_baseline": {
            "run": RUN_ID,
            "path": f"dev/00runs/{BASELINE_RUN_NAME}",
            "report_file": BASELINE_REPORT_NAME,
            "report_sha256": baseline_sha256,
            "combined": {
                "applied": baseline_report["combined"]["applied"],
                "pass": baseline_report["combined"]["pass"],
                "fail": baseline_report["combined"]["fail"],
                "residual_failures": baseline_report["combined"][
                    "residual_failures"],
                "pass_rate_among_original_failures":
                    baseline_report["combined"][
                        "pass_rate_among_original_failures"],
                "max_abs_sigma": baseline_report["combined"][
                    "max_abs_sigma"],
                "max_abs_sigma_check": baseline_report["combined"][
                    "max_abs_sigma_check"],
            },
        },
        "runs": runs,
        "combined": combined,
        "targeted_r31": {
            "run": "R31",
            "archive": r31["archive"],
            "scope": r31["scope"],
            "n_checks_applied": r31_summary["n_checks_applied"],
            "n_pass": r31_summary["n_pass"],
            "n_vacuous_zero_live": r31_summary["n_vacuous_zero_live"],
            "n_fail": r31_summary["n_fail"],
            "max_abs_sigma": r31_summary["max_abs_sigma"],
            "max_abs_sigma_check": r31_summary["max_abs_sigma_check"],
            "boundary_interior": r31_summary["boundary_interior"],
            "original_failures": original_rows,
            "original_failures_passing": r31_summary[
                "original_failures_passing"],
            "original_failures_persisting": r31_summary[
                "original_failures_persisting"],
        },
        "acceptance": {
            "t1_targeted_r31": {
                "criterion": (
                    "all 14 R31 momentum-moment checks that failed at "
                    "the stopped verification pass at 3.0 sigma under "
                    "the fixed engine at the original statistics"
                ),
                "n_original_failures": N_R31_ORIGINAL_FAILURES,
                "original_failures_passing": r31_summary[
                    "original_failures_passing"],
                "original_failures_persisting": r31_summary[
                    "original_failures_persisting"],
                "pass": t1_pass,
            },
            "t2_card2_reeval": {
                "criterion": (
                    "under the fixed engine at 3.0 sigma over all 834 "
                    "applied R21/R22 checks, the failure count is "
                    "strictly below the sealed re-evaluation's 11 "
                    "residual failures; the four sealed "
                    "moment-approximation-floor residuals all pass; "
                    "the pass rate among the original 323 failures is "
                    "at least the sealed 0.9659 (no regression)"
                ),
                "failure_count": {
                    "sealed": SEALED_RESIDUAL_FAILURES,
                    "achieved": combined["fail"],
                    "required": "strictly below the sealed 11",
                    "pass": t2_failure_count_pass,
                },
                "floor_residuals": {
                    "checks": floor_rows,
                    "required": "all four pass",
                    "pass": floor_pass,
                },
                "pass_rate_among_original_failures": {
                    "sealed": SEALED_PASS_RATE,
                    "threshold": SEALED_PASS_RATE,
                    "achieved": achieved,
                    "pass": t2_rate_pass,
                },
                "pass": t2_pass,
            },
            "pass": acceptance_pass,
        },
        "residual_findings": findings,
        "clause_status": (
            "the fixed engine's live re-evaluation over the sealed "
            "R21/R22 archives and the targeted R31 artifacts at the "
            "unchanged 3.0-sigma tolerance; the acceptance verdicts "
            "are recorded per the card's T1 and T2 criteria; no "
            "residual is absorbed into the metric and no tolerance is "
            "relaxed"
        ),
        "no_fitting": True,
    }


def _run_live_report(run_dir: Path) -> None:
    _require_live_pre_registration(run_dir)
    payloads = {}
    for name in ("live_reeval_r21.json", "live_reeval_r22.json",
                 "live_reeval_r31_targeted.json"):
        path = run_dir / name
        if not path.is_file():
            raise RuntimeError(
                f"rereval_threshold_checks: {path} is missing; run the "
                "live re-evaluations before --live-report"
            )
        payloads[name] = json.loads(path.read_text(encoding="utf-8"))
    baseline_path = BASELINE_DIR / BASELINE_REPORT_NAME
    if not baseline_path.is_file():
        raise RuntimeError(
            "rereval_threshold_checks: the sealed baseline report "
            f"{baseline_path} is missing"
        )
    baseline_report = json.loads(
        baseline_path.read_text(encoding="utf-8"))
    baseline_combined = baseline_report["combined"]
    if (int(baseline_combined["residual_failures"])
            != SEALED_RESIDUAL_FAILURES):
        raise RuntimeError(
            "rereval_threshold_checks: the sealed baseline report's "
            "residual count "
            f"{baseline_combined['residual_failures']} differs from "
            f"the pre-registered {SEALED_RESIDUAL_FAILURES}"
        )
    report = _build_live_report(
        payloads["live_reeval_r21.json"],
        payloads["live_reeval_r22.json"],
        payloads["live_reeval_r31_targeted.json"],
        baseline_report,
        _sha256_file(baseline_path),
    )
    out_path = run_dir / "gate2_revised_threshold_v2.json"
    _write_json(out_path, report)
    t1 = report["acceptance"]["t1_targeted_r31"]
    t2 = report["acceptance"]["t2_card2_reeval"]
    print(
        "rereval_threshold_checks: wrote "
        f"{out_path} (T1 targeted R31: "
        f"{t1['original_failures_passing']}/"
        f"{t1['n_original_failures']} original failures pass, "
        f"{t1['original_failures_persisting']} persisting, "
        f"pass={t1['pass']}; T2 card-2 re-eval: "
        f"{t2['failure_count']['achieved']} failures vs sealed "
        f"{t2['failure_count']['sealed']}, floor residuals pass="
        f"{t2['floor_residuals']['pass']}, rate "
        f"{t2['pass_rate_among_original_failures']['achieved']:.6f}, "
        f"pass={t2['pass']}; acceptance pass="
        f"{report['acceptance']['pass']})"
    )


def _run_live_seal(run_dir: Path) -> None:
    run_dir = Path(run_dir)
    for required in LIVE_SEAL_REQUIRED:
        if not (run_dir / required).is_file():
            raise RuntimeError(
                f"rereval_threshold_checks: {run_dir / required} is "
                "missing; the live seal requires the full evidence "
                "chain (pre-registration, the three live re-eval "
                "payloads, the revised clause report)"
            )
    sums_path = run_dir / "SHA256SUMS.txt"
    manifest_path = run_dir / "seal_manifest.json"
    if sums_path.is_file() or manifest_path.is_file():
        raise RuntimeError(
            f"rereval_threshold_checks: {run_dir} is already sealed; "
            "the run dir is immutable after the seal"
        )
    entries: list[tuple[str, str]] = []
    total_bytes = 0
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(run_dir).as_posix()
        if rel in ("seal_manifest.json", "SHA256SUMS.txt"):
            continue
        entries.append((rel, _sha256_file(path)))
        total_bytes += path.stat().st_size
    # The sums entries carry monorepo-root-relative paths (the R21/R22
    # production-archive convention), so the card's validation command
    # ``sha256sum -c dev/00runs/<run>/SHA256SUMS.txt`` verifies from the
    # repository root.  The manifest records the run-dir-relative form.
    with open(sums_path, "w", encoding="utf-8", newline="\n") as handle:
        for rel, digest in entries:
            handle.write(
                f"{digest}  {run_dir.relative_to(MONO_ROOT).as_posix()}"
                f"/{rel}\n")
    manifest = {
        "schema": SEAL_SCHEMA,
        "run": LIVE_RUN_ID,
        "card": LIVE_CARD_ID,
        "created_utc": _now_utc(),
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "sha256_sums_file": "SHA256SUMS.txt",
        "sha256_sums_sha256": _sha256_file(sums_path),
        "excluded_from_sums": ["seal_manifest.json", "SHA256SUMS.txt"],
        "path_convention": (
            "the SHA256SUMS.txt entries carry monorepo-root-relative "
            f"paths ({LIVE_RUN_DIR_REL}/<file>), so the card's "
            "validation command sha256sum -c "
            f"{LIVE_RUN_DIR_REL}/SHA256SUMS.txt verifies from the "
            "repository root (the R21/R22 production-archive "
            "convention); the manifest's entries below carry the "
            "run-dir-relative form"
        ),
        "note": "the run dir is immutable after the seal (the Round 1 "
                "sealing discipline); clean_environment_commands.json "
                "is the post-seal validation record, written after "
                "this manifest and never edited",
    }
    _write_json(manifest_path, manifest)
    print(
        f"rereval_threshold_checks: sealed {run_dir} "
        f"({len(entries)} files, {total_bytes} bytes, SHA256SUMS.txt "
        f"sha256 {manifest['sha256_sums_sha256']})"
    )


#: The clean-interpreter probe of the live modes: imports this driver
#: module fresh, re-runs the three live re-evaluations and the revised
#: clause report over the sealed artifacts, and prints the canonical
#: payload hashes (it writes nothing).
_LIVE_CLEAN_PROBE_SOURCE = """\
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

driver_path, run_dir, baseline_path = sys.argv[1], sys.argv[2], sys.argv[3]
run_dir = Path(run_dir)
spec = importlib.util.spec_from_file_location(
    "rereval_threshold_checks_live_clean_probe", driver_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _canon(payload):
    return (json.dumps(payload, indent=2) + "\\n").encode("utf-8")


hashes = {}
for key in ("r21", "r22"):
    payload = module._live_rereval_payload(key)
    hashes[f"live_reeval_{key}.json"] = hashlib.sha256(
        _canon(payload)).hexdigest()
payload = module._live_rereval_r31_payload()
hashes["live_reeval_r31_targeted.json"] = hashlib.sha256(
    _canon(payload)).hexdigest()
baseline_path = Path(baseline_path)
baseline_report = json.loads(
    baseline_path.read_text(encoding="utf-8"))
report = module._build_live_report(
    json.loads((run_dir / "live_reeval_r21.json").read_text(
        encoding="utf-8")),
    json.loads((run_dir / "live_reeval_r22.json").read_text(
        encoding="utf-8")),
    json.loads((run_dir / "live_reeval_r31_targeted.json").read_text(
        encoding="utf-8")),
    baseline_report,
    module._sha256_file(baseline_path),
)
hashes["gate2_revised_threshold_v2.json"] = hashlib.sha256(
    _canon(report)).hexdigest()
print(json.dumps({
    "hashes": hashes,
    "environment": module.environment_fingerprint(),
}))
"""


def _run_live_clean_validate(run_dir: Path) -> None:
    out_path = run_dir / "clean_environment_commands.json"
    if out_path.is_file():
        raise RuntimeError(
            f"rereval_threshold_checks: {out_path} exists; the "
            "clean-environment validation is recorded once per seal"
        )
    sums_path = run_dir / "SHA256SUMS.txt"
    manifest_path = run_dir / "seal_manifest.json"
    if not sums_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(
            f"rereval_threshold_checks: {run_dir} is not sealed; run "
            "--live-seal before the clean-environment validation"
        )
    checked = 0
    mismatches: list[str] = []
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        # The sums entries carry monorepo-root-relative paths (the
        # R21/R22 production-archive convention).
        path = MONO_ROOT / rel
        if not path.is_file() or _sha256_file(path) != digest:
            mismatches.append(rel)
        checked += 1
    if mismatches:
        raise RuntimeError(
            "rereval_threshold_checks: the seal verification failed "
            "for: " + ", ".join(mismatches)
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sha256_sums_sha256") != _sha256_file(sums_path):
        raise RuntimeError(
            "rereval_threshold_checks: the seal manifest's "
            "SHA256SUMS.txt digest does not match the sums file"
        )
    sealed = {
        name: _sha256_file(run_dir / name)
        for name in (
            "live_reeval_r21.json",
            "live_reeval_r22.json",
            "live_reeval_r31_targeted.json",
            "gate2_revised_threshold_v2.json",
        )
    }
    baseline_path = BASELINE_DIR / BASELINE_REPORT_NAME
    driver_path = Path(__file__).resolve()
    argv = [
        sys.executable, "-I", "-c", _LIVE_CLEAN_PROBE_SOURCE,
        str(driver_path), str(run_dir), str(baseline_path),
    ]
    completed = subprocess.run(argv, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "rereval_threshold_checks: the clean-interpreter probe "
            "failed:\n" + completed.stderr
        )
    probe_line = completed.stdout.strip().splitlines()[-1]
    probe = json.loads(probe_line)
    recomputed = {
        str(name): str(digest)
        for name, digest in probe["hashes"].items()
    }
    hashes_match = sealed == recomputed
    payload = {
        "schema": CLEAN_SCHEMA,
        "run": LIVE_RUN_ID,
        "card": LIVE_CARD_ID,
        "verified_utc": _now_utc(),
        "seal_verification": {
            "files_checked": checked,
            "mismatches": [],
        },
        "commands": [
            {
                "purpose": "clean-interpreter re-run of the live "
                           "re-evaluations (R21, R22, the targeted R31 "
                           "check) and the revised clause report over "
                           "the sealed artifacts (computes the payloads "
                           "and prints their canonical hashes; writes "
                           "nothing)",
                "argv": argv,
                "cwd": str(Path.cwd()),
            }
        ],
        "environment": environment_fingerprint(),
        "probe_environment": probe.get("environment", {}),
        "sealed_report_sha256": sealed,
        "recomputed_report_sha256": recomputed,
        "hashes_match": bool(hashes_match),
    }
    _write_json(out_path, payload)
    if not hashes_match:
        raise RuntimeError(
            "rereval_threshold_checks: the clean-interpreter re-run "
            "does not reproduce the sealed live report hashes (sealed "
            f"{sealed}, recomputed {recomputed}); the record is in "
            f"{out_path}"
        )
    print(
        "rereval_threshold_checks: clean-environment validation "
        f"passed ({checked} sealed files verified; the re-run from a "
        "clean interpreter reproduces the sealed live re-eval and "
        f"revised-clause hashes); record at {out_path}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rereval_threshold_checks.py",
        description=(
            "Round 3 card 2: the sealed-archive re-evaluation of the "
            "R21/R22 threshold-discontinuity metric under the "
            "re-registered delta_analytic_full metric, plus the card 8a "
            "phase-2 live re-evaluation modes of the engine-fix card "
            "(TASK-20260918-r3-card2-engine-three-body-moment-fix).  "
            "Each invocation performs exactly one step and stops."
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pre-registration", action="store_true",
        help="write pre_registration.json and provenance.json into the "
             "run dir, then stop (nothing is computed)",
    )
    group.add_argument(
        "--rereval", choices=["r21", "r22", "both"], default=None,
        help="re-evaluate the named sealed archive(s) against the "
             "sealed stored delta_analytic_full, write "
             "rereval_<run>.json, then stop",
    )
    group.add_argument(
        "--crosscheck", choices=["r21", "r22", "both"], default=None,
        help="recompute the new-run analytic engine over the named "
             "sealed grid(s) and write metric_engine_crosscheck_<run>"
             ".json, then stop (the slow mode: the full-grid analytic "
             "quadrature)",
    )
    group.add_argument(
        "--report", action="store_true",
        help="assemble gate2_revised_threshold.json from the two "
             "rereval payloads, then stop",
    )
    group.add_argument(
        "--seal", action="store_true",
        help="write seal_manifest.json and SHA256SUMS.txt over the "
             "run dir, then stop (the run dir is immutable after the "
             "seal)",
    )
    group.add_argument(
        "--clean-validate", action="store_true",
        help="verify the seal, re-run the re-evaluation and the report "
             "assembly from a clean interpreter, and confirm the "
             "report hashes match; write clean_environment_commands"
             ".json, then stop",
    )
    group.add_argument(
        "--live-pre-registration", action="store_true",
        help="card 8a phase 2: write the engine-fix live "
             "pre-registration and provenance into the live run dir "
             f"({LIVE_RUN_DIR_REL}), then stop (nothing is computed)",
    )
    group.add_argument(
        "--live-reeval", choices=["r21", "r22", "r31", "both"],
        default=None,
        help="card 8a phase 2: live re-evaluation of the fixed engine "
             "against the sealed delta_observed (r21|r22|both) or the "
             "targeted R31 check (r31), write live_reeval_<run>.json, "
             "then stop; requires --live-pre-registration",
    )
    group.add_argument(
        "--live-report", action="store_true",
        help="card 8a phase 2: assemble gate2_revised_threshold_v2.json "
             "from the three live payloads and the sealed baseline, "
             "then stop",
    )
    group.add_argument(
        "--live-seal", action="store_true",
        help="card 8a phase 2: write seal_manifest.json and "
             "SHA256SUMS.txt over the live run dir, then stop (the run "
             "dir is immutable after the seal)",
    )
    group.add_argument(
        "--live-clean-validate", action="store_true",
        help="card 8a phase 2: verify the live seal, re-run the live "
             "re-evaluations and the report assembly from a clean "
             "interpreter, and confirm the hashes match; write "
             "clean_environment_commands.json, then stop",
    )
    parser.add_argument(
        "--run-dir", type=Path, default=DEFAULT_RUN_DIR,
        help=f"the run directory (default: {RUN_DIR_REL})",
    )
    return parser


def _keys(choice: str) -> list[str]:
    return ["r21", "r22"] if choice == "both" else [choice]


def _live_run_dir() -> Path:
    """The live run dir of the card-8a phase-2 re-evaluation (fixed by
    the card; the live modes always target it and refuse unsafe
    targets)."""
    return LIVE_DEFAULT_RUN_DIR


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    if args.pre_registration:
        _run_pre_registration(run_dir)
    elif args.rereval:
        _run_rereval(run_dir, _keys(args.rereval))
    elif args.crosscheck:
        _run_crosscheck(run_dir, _keys(args.crosscheck))
    elif args.report:
        _run_report(run_dir)
    elif args.seal:
        _run_seal(run_dir)
    elif args.clean_validate:
        _run_clean_validate(run_dir)
    elif args.live_pre_registration:
        _run_live_pre_registration(_live_run_dir())
    elif args.live_reeval:
        if args.live_reeval == "r31":
            _run_live_reeval_r31(_live_run_dir())
        else:
            _run_live_reeval(_live_run_dir(), _keys(args.live_reeval))
    elif args.live_report:
        _run_live_report(_live_run_dir())
    elif args.live_seal:
        _run_live_seal(_live_run_dir())
    else:
        _run_live_clean_validate(_live_run_dir())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
