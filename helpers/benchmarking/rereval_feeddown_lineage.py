#!/usr/bin/env python3
"""Sealed-archive re-evaluation driver for the corrected R23 feed-down
lineage convention (Round 3, card 3).

Task ``TASK-20260917-r3-gate2-followup-feeddown-lineage-convention``
(the Gate 2 closure obligation, follow-up 3 of the Round 2 closure
record ``_knowledge/operations/round2-closure-2026-09-17.md``; plan
section 8 run R23, line 573, and plan section 9.3).  The R23 feed-down
lineage diagnostic reported an infinite deviation (``|z| = Infinity``
across all seven species) because the analytic calculation assumed all
final particles carry resonant parentage while the event tracker
records resonant parentage only for non-root cascade decays.  This
driver re-evaluates the seven-species feed-down diagnostic on the
sealed R23 archive under the corrected convention (the corrected
analytic chain of the committed R23 driver
``helpers/benchmarking/staged_r23_resonance_ablation.py``) and
evaluates the pre-registered transition-mismatch criterion that decides
whether the conditional single-nuisance fit sub-card (card 3a,
``TASK-20260917-r3-r23-feeddown-transition-nuisance-fit``) runs.

No Monte Carlo and no new generation: the sealed R23 samples (800,000
events, 1,600,000 realized sources) are reused read-only.  No random
draws exist on any code path; the master seed ``20260917`` is recorded
in the pre-registration with ``draws: 0`` for contract uniformity with
the Round 3 seed rule.

The corrected convention (the card's scope clause (a)): resonant
parentage is recorded only for non-root cascade decays -- a particle
produced directly by the source (including the stable-identity final of
a stable root record) is not resonantly produced.  The corrected
analytic expectation for a species is ``f_res = 1 - (direct-production
fraction)``, the catalog-mixture share of the species' finals produced
at the root vertex (or as the stable identity) at the source masses
realized in the sample, evaluated on the sealed below-boundary mass
histogram through the corrected ``_ChainExpectation`` of the R23
driver.  The tracker side is unchanged: the sealed tracker already
records the corrected convention.

The criterion (the round-3 index item 8 pre-registered criterion): a
transition mismatch persists if, for any of the seven species, the
deviation ``|observed - corrected_analytic|`` exceeds ``3.0 *
SE_combined``, the combined MC/analytic standard error
``sqrt(se_mc^2 + se_analytic^2)`` with ``se_mc = sqrt(f_obs (1 - f_obs)
/ n)`` and ``se_analytic = sqrt(f_ana (1 - f_ana) / n)`` (the codebase
"combined" convention, the R23 inclusive-metric quadrature; the 3.0-
sigma tolerance is the Round 2 pre-registration's, unchanged).  If the
trigger is false, card 3a stays closed and no nuisance fit is executed.

Modes (exactly one per invocation; each performs exactly one step and
stops; read-only over the sealed R23 archive; writes only into its own
run dir):

- ``--pre-registration``: write ``pre_registration.json`` and
  ``provenance.json`` into the run dir, then stop.  Nothing is
  computed.
- ``--rereval``: re-evaluate the sealed R23 feed-down diagnostic under
  the corrected convention and write ``feeddown_corrected.json`` and
  ``subcard_trigger.json`` into the run dir, then stop.  Requires the
  pre-registration.
- ``--seal``: write ``seal_manifest.json`` and ``SHA256SUMS.txt`` over
  the run dir, then stop.  The run dir is immutable after the seal.
- ``--clean-validate``: verify the seal, re-run the re-evaluation from
  a clean interpreter over the sealed archive, and confirm the payload
  hashes match; write ``clean_environment_commands.json`` into the run
  dir, then stop.

Deterministic: the pre-registration, ``feeddown_corrected.json``, and
``subcard_trigger.json`` carry no wall-clock fields (their bytes are a
pure function of the sealed archive and the committed code); only the
provenance, the seal manifest, and the clean-environment record carry
wall-clock stamps.

Run dir: ``dev/00runs/r3_feeddown_lineage_convention_2026-09-17/`` (the
Round 2 card-8 diagnostic-run pattern).

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
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Tree location: this driver runs from the nested tree
# (dev/mixing2006/helpers/benchmarking/) and from its byte-identical
# monorepo mirror (helpers/benchmarking/).  Both must find the
# na49_mixing package and the sealed R23 archive.
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
            "rereval_feeddown_lineage: cannot locate the na49_mixing "
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
_BENCHMARK_DIR = NESTED_ROOT / "helpers" / "benchmarking"
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

import numpy as np  # noqa: E402

import staged_r23_resonance_ablation as r23  # noqa: E402
from na49_mixing.hadron_catalog import HadronCatalog  # noqa: E402

# ---------------------------------------------------------------------------
# Run identity and contract constants
# ---------------------------------------------------------------------------
RUN_ID = "R3-FEEDDOWN-LINEAGE-CONVENTION"
CARD_ID = "TASK-20260917-r3-gate2-followup-feeddown-lineage-convention"
RUN_NAME = "r3_feeddown_lineage_convention_2026-09-17"
DEFAULT_RUN_DIR = MONO_ROOT / "dev" / "00runs" / RUN_NAME
RUN_DIR_REL = f"dev/00runs/{RUN_NAME}"

#: The sealed R23 archive (read-only; never re-sealed).
ARCHIVE = {
    "run": "R23",
    "name": "r23_resonance_only_pp_ablation_2026-09-13",
    "card": "TASK-20260913-r2-run-r23-resonance-only-pp-ablation",
    "pre_registration": "pre_registration.json",
}

#: Master seed recorded for contract uniformity with the Round 3 seed
#: rule; the driver draws no random numbers (``draws: 0``).
MASTER_SEED = 20260917
DRAWS = 0

#: The pre-registered criterion (the round-3 index item 8
#: interpretation contract; the 3.0-sigma tolerance of the Round 2
#: pre-registration, unchanged).
N_SIGMA = 3.0

#: The seven-species check floor (the sealed R23 clause-(iii) rule).
MIN_N_PER_SPECIES = r23.FEED_DOWN_MIN_N

#: The conditional sub-card this card's trigger decides.
SUBCARD_ID = "TASK-20260917-r3-r23-feeddown-transition-nuisance-fit"

PRE_SCHEMA = "na49_mixing.rereval_feeddown_lineage.pre_registration/1"
PROV_SCHEMA = "na49_mixing.rereval_feeddown_lineage.provenance/1"
FEEDDOWN_SCHEMA = "na49_mixing.rereval_feeddown_lineage.corrected_feeddown/1"
TRIGGER_SCHEMA = "na49_mixing.rereval_feeddown_lineage.subcard_trigger/1"
SEAL_SCHEMA = "na49_mixing.rereval_feeddown_lineage.seal_manifest/1"
CLEAN_SCHEMA = "na49_mixing.rereval_feeddown_lineage.clean_environment/1"

LINEAGE_CONVENTION = (
    "resonant parentage is recorded only for non-root cascade decays; "
    "a particle produced directly by the source (including the "
    "stable-identity final of a source whose drawn record is itself a "
    "stable state) is not resonantly produced"
)
CRITERION_TEXT = (
    "a transition mismatch persists if, for any of the seven species, "
    "|observed - corrected_analytic| > 3.0 * SE_combined (the round-3 "
    "index item 8 pre-registered criterion; the 3.0-sigma tolerance of "
    "the Round 2 pre-registration)"
)
SE_DEFINITION = (
    "SE_combined = sqrt(se_mc^2 + se_analytic^2), the combined "
    "MC/analytic standard error: se_mc = sqrt(f_obs (1 - f_obs) / n) "
    "is the pooled observed fraction's binomial standard error and "
    "se_analytic = sqrt(f_ana (1 - f_ana) / n) is the corrected "
    "analytic fraction's binomial standard error at the pooled "
    "statistics (the codebase combined-error quadrature of the R23 "
    "inclusive-stability metric; the analytic side carries no "
    "Monte Carlo error beyond the sealed mass histogram it is "
    "evaluated on, and its binomial term is the model's own spread at "
    "the same statistics)"
)


# ---------------------------------------------------------------------------
# Small shared helpers (the Round 3 card 2 driver's conventions)
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


def _archive_dir() -> Path:
    return MONO_ROOT / "dev" / "00runs" / ARCHIVE["name"]


def _verify_archive_seal(archive: Path) -> dict:
    """Read-only verification of the sealed R23 archive; returns its
    seal identity (the checked file count and the SHA-256 of its
    SHA256SUMS.txt).  The archive is never modified."""
    sums_path = archive / "SHA256SUMS.txt"
    manifest_path = archive / "seal_manifest.json"
    if not sums_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(
            f"rereval_feeddown_lineage: {archive} is not a sealed "
            "archive (SHA256SUMS.txt / seal_manifest.json missing)"
        )
    checked = 0
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        # The sealed R23 sums are run-dir-relative.
        path = archive / rel
        if not path.is_file() or _sha256_file(path) != digest:
            raise RuntimeError(
                f"rereval_feeddown_lineage: the sealed archive "
                f"{archive} failed its own seal at {rel}; the archive "
                "is read-only evidence and must not be modified"
            )
        checked += 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sums_digest = _sha256_file(sums_path)
    # The seal manifests in use record the sums-file digest either under
    # a dedicated key (the Round 3 re-evaluation convention) or as the
    # ``files`` entry of SHA256SUMS.txt itself (the sealed R23
    # convention); both bind the manifest to the sums file.
    recorded = manifest.get("sha256_sums_sha256")
    if recorded is None:
        for entry in manifest.get("files", []):
            if str(entry.get("path")) == "SHA256SUMS.txt":
                recorded = entry.get("sha256")
                break
    if recorded != sums_digest:
        raise RuntimeError(
            f"rereval_feeddown_lineage: the seal manifest of {archive} "
            "does not match its SHA256SUMS.txt"
        )
    return {"files_checked": checked,
            "sha256_sums_sha256": sums_digest}


def _load_replicas(archive: Path, pre: dict) -> dict:
    """The eight sealed production replicas, verified read-only against
    the sealed pre-registration (seed, event count, input digests,
    mode)."""
    config_sha = str(pre["config_hash"])
    tables_sha = str(pre["tables_sha256"])
    builder_sha = str(pre["builder_module_sha256"])
    driver_sha = str(pre["driver_sha256"])
    replicas: dict[int, dict] = {}
    for seed in r23.REPLICA_SEEDS:
        path = archive / r23.REPLICA_FMT.format(int(seed))
        if not path.is_file():
            raise RuntimeError(
                f"rereval_feeddown_lineage: sealed replica missing: {path}"
            )
        record = json.loads(path.read_text(encoding="utf-8"))
        identity_ok = (
            int(record.get("seed", -1)) == int(seed)
            and int(record.get("n_events", -1)) == r23.T2_EVENTS_PER_REPLICA
            and record.get("config_sha256") == config_sha
            and record.get("tables_sha256") == tables_sha
            and record.get("builder_module_sha256") == builder_sha
            and record.get("driver_sha256") == driver_sha
            and record.get("mode") == "production"
        )
        if not identity_ok:
            raise RuntimeError(
                f"rereval_feeddown_lineage: sealed replica {path} does "
                "not match the sealed pre-registration (seed, event "
                "count, input digests, or mode)"
            )
        replicas[int(seed)] = record
    return replicas


# ---------------------------------------------------------------------------
# The corrected re-evaluation payloads (pure functions of the sealed
# archive and the committed code; byte-stable, no wall clock)
# ---------------------------------------------------------------------------
def build_corrected_feeddown(catalog: HadronCatalog, replicas: dict) -> dict:
    """The corrected seven-species feed-down table over the sealed R23
    replicas (pure function; byte-stable)."""
    boundary_block = r23._pooled_boundary(replicas)
    feeddown_block = r23._pooled_feeddown(replicas)
    analytic, hist_total = r23._analytic_feeddown(catalog, boundary_block)

    rows: list[dict] = []
    max_abs_sigma = 0.0
    max_species = None
    n_checked = 0
    at_threshold: list[str] = []
    for key, slot in sorted(feeddown_block["per_species"].items()):
        n = int(slot["n"])
        if n < MIN_N_PER_SPECIES:
            continue
        n_res = int(slot["n_resonant"])
        f_obs = float(n_res) / float(n)
        expected = analytic.get(key)
        if expected is None:
            e_finals, e_res = 0.0, 0.0
        else:
            e_finals, e_res = float(expected[0]), float(expected[1])
        f_ana = e_res / e_finals if e_finals > 0.0 else 0.0
        se_mc = (
            (f_obs * (1.0 - f_obs) / float(n)) ** 0.5
            if 0.0 < f_obs < 1.0 else 0.0
        )
        se_ana = (
            (f_ana * (1.0 - f_ana) / float(n)) ** 0.5
            if 0.0 < f_ana < 1.0 else 0.0
        )
        se_combined = (se_mc * se_mc + se_ana * se_ana) ** 0.5
        if se_combined > 0.0:
            z = (f_obs - f_ana) / se_combined
        elif f_obs == f_ana:
            z = 0.0
        else:  # structurally unreachable under the corrected convention
            z = float("inf") if f_obs > f_ana else float("-inf")
        within = bool(abs(z) <= N_SIGMA)
        rows.append({
            "species": str(key),
            "n_finals": n,
            "n_resonant": n_res,
            "fraction_resonant_observed": float(f_obs),
            "expected_finals": float(e_finals),
            "expected_resonant": float(e_res),
            "fraction_resonant_analytic": float(f_ana),
            "se_mc": float(se_mc),
            "se_analytic": float(se_ana),
            "se_combined": float(se_combined),
            "z": float(z),
            "within_3_sigma": within,
        })
        n_checked += 1
        if abs(z) > N_SIGMA:
            at_threshold.append(str(key))
        if abs(z) > max_abs_sigma:
            max_abs_sigma = abs(z)
            max_species = str(key)

    n_new = int(boundary_block["n_new_kernel"])
    per_source_obs = (
        float(feeddown_block["n_finals_new_kernel"]) / float(n_new)
        if n_new else 0.0
    )
    expected_total = sum(
        float(v[0]) for v in analytic.values()
    )
    all_finite = all(
        math.isfinite(float(row["z"])) for row in rows
    )
    payload = {
        "schema": "na49_mixing.rereval_feeddown_lineage.corrected_feeddown/1",
        "card": CARD_ID,
        "run": ARCHIVE["run"],
        "archive": {
            "path": f"dev/00runs/{ARCHIVE['name']}",
            "card": ARCHIVE["card"],
            "read_only": True,
        },
        "master_seed": MASTER_SEED,
        "draws": DRAWS,
        "convention": {
            "lineage": LINEAGE_CONVENTION,
            "tracker": (
                "unchanged: the sealed R23 tracker _accumulate_feeddown "
                "already records the corrected convention (a final "
                "particle is resonant when its producing vertex's "
                "parent is a resonance other than the drawn root "
                "record); the degenerate convention was hosted on the "
                "analytic side only"
            ),
            "analytic": (
                "corrected: f_res = 1 - (direct-production fraction); "
                "the analytic chain walk marks only non-root cascade "
                "products resonant, keeps the stable records in the "
                "flat record mixture (the stable-identity finals), and "
                "weights every open channel object by its card-6 "
                "open-weight share; daughter resonances at their "
                "catalog pole masses over the sealed boundary mass "
                "histogram"
            ),
            "fix_location": (
                "the R23 driver's analytic chain _ChainExpectation in "
                "helpers/benchmarking/staged_r23_resonance_ablation.py "
                "(byte-identical nested + outer copies); the tracker, "
                "the builder, the kernel, and every draw sequence are "
                "untouched"
            ),
        },
        "se_combined_definition": SE_DEFINITION,
        "pooled": {
            "n_replicas": len(replicas),
            "n_events": int(sum(
                int(r["n_events"]) for r in replicas.values())),
            "n_sources": int(boundary_block["n_sources"]),
            "n_new_kernel": n_new,
            "n_finals_new_kernel": int(feeddown_block["n_finals_new_kernel"]),
            "expected_finals_per_new_kernel_source": float(
                sum(float(v[0]) for v in analytic.values())),
            "observed_finals_per_new_kernel_source": float(per_source_obs),
            "histogram_total": int(hist_total),
        },
        "rows": rows,
        "summary": {
            "n_checked": int(n_checked),
            "min_n_per_species": int(MIN_N_PER_SPECIES),
            "n_sigma": float(N_SIGMA),
            "max_abs_sigma": float(max_abs_sigma),
            "max_abs_sigma_species": max_species,
            "species_at_threshold": list(at_threshold),
            "all_finite": bool(all_finite),
            "pass": bool(n_checked > 0 and not at_threshold),
        },
        "degenerate_reference": {
            "sealed_report": (
                f"dev/00runs/{ARCHIVE['name']}/comparison_r01.json"
            ),
            "sealed_feed_down_max_abs_sigma": "Infinity",
            "note": (
                "the sealed R23 metric (iii) verdict under the "
                "degenerate convention (f_ana = 1 exactly, zero sigma); "
                "read-only reference, the sealed archive is not "
                "modified"
            ),
        },
        "no_fitting": True,
        "optimizer_invoked": False,
    }
    return payload


def build_subcard_trigger(corrected: dict) -> dict:
    """The trigger decision from the corrected payload (pure function;
    byte-stable)."""
    summary = corrected["summary"]
    trigger = bool(not summary["pass"])
    return {
        "schema": "na49_mixing.rereval_feeddown_lineage.subcard_trigger/1",
        "card": CARD_ID,
        "criterion": CRITERION_TEXT,
        "trigger": trigger,
        "max_abs_sigma": float(summary["max_abs_sigma"]),
        "max_abs_sigma_species": summary["max_abs_sigma_species"],
        "species_at_threshold": list(summary["species_at_threshold"]),
        "n_species_checked": int(summary["n_checked"]),
        "n_sigma": float(summary["n_sigma"]),
        "subcard": SUBCARD_ID,
        "subcard_unblocked": bool(trigger),
        "disposition": (
            "a transition mismatch persists at 3.0 sigma under the "
            "corrected convention; the conditional sub-card "
            f"{SUBCARD_ID} is unblocked and this run's report is its "
            "input"
            if trigger else
            "no transition mismatch persists at 3.0 sigma under the "
            "corrected convention: the trigger is false, the "
            f"conditional sub-card {SUBCARD_ID} stays closed, and no "
            "nuisance fit is executed (plan section 8 run R23 line 573: "
            "the fit runs only if required)"
        ),
        "no_fitting": True,
        "optimizer_invoked": False,
    }


# ---------------------------------------------------------------------------
# Pre-registration and provenance
# ---------------------------------------------------------------------------
def _build_pre_registration() -> dict:
    return {
        "schema": PRE_SCHEMA,
        "run": RUN_ID,
        "card": CARD_ID,
        "mode": "diagnostic re-evaluation over the sealed R23 archive "
                "(no Monte Carlo, no generation)",
        "master_seed": MASTER_SEED,
        "draws": DRAWS,
        "seed_note": (
            "no random draws exist on any code path; the master seed "
            "is recorded for contract uniformity with the Round 3 seed "
            "rule"
        ),
        "archive": {
            "run": ARCHIVE["run"],
            "path": f"dev/00runs/{ARCHIVE['name']}",
            "card": ARCHIVE["card"],
            "read_only": True,
        },
        "convention": {
            "lineage": LINEAGE_CONVENTION,
            "corrects": (
                "the sealed R23 analytic feed-down chain assumed all "
                "final particles carry resonant parentage (f_ana = 1 "
                "exactly, zero sigma, |z| = Infinity across all seven "
                "species) while the sealed tracker records resonant "
                "parentage only for non-root cascade decays"
            ),
            "corrected_expectation": (
                "f_res = 1 - (direct-production fraction), the "
                "catalog-mixture share of the species' finals produced "
                "at the root vertex (or as the stable identity) at the "
                "source masses realized in the sample, evaluated on "
                "the sealed below-boundary mass histogram"
            ),
            "tracker": "unchanged (the sealed tracker convention)",
        },
        "criterion": {
            "statement": CRITERION_TEXT,
            "se_combined": SE_DEFINITION,
            "n_sigma": float(N_SIGMA),
            "tolerance_unchanged": True,
            "min_n_per_species": int(MIN_N_PER_SPECIES),
            "n_species": 7,
        },
        "subcard": SUBCARD_ID,
        "constraints": {
            "no_fitting": True,
            "optimizer_invoked": False,
            "sealed_archive_read_only": True,
            "no_new_monte_carlo": True,
            "no_draw_sequence_change": (
                "the convention fix is diagnostic-only: the tracker, "
                "the builder, the kernel, and every draw sequence are "
                "untouched (the card-13 four-layer bit-identity pins "
                "are the guard)"
            ),
        },
        "outputs": ["feeddown_corrected.json", "subcard_trigger.json"],
        "written_before_evaluation": True,
    }


def _run_pre_registration(run_dir: Path) -> None:
    pre_path = run_dir / "pre_registration.json"
    prov_path = run_dir / "provenance.json"
    if pre_path.is_file() and (
        (run_dir / "feeddown_corrected.json").is_file()
        or (run_dir / "subcard_trigger.json").is_file()
    ):
        raise RuntimeError(
            f"rereval_feeddown_lineage: {pre_path} exists and "
            "evaluation artifacts are present; the pre-registration is "
            "immutable after evaluation has started (a change "
            "invalidates the run; re-run under a new date)"
        )
    archive = _archive_dir()
    seal = _verify_archive_seal(archive)
    _write_json(pre_path, _build_pre_registration())
    provenance = {
        "schema": PROV_SCHEMA,
        "run": RUN_ID,
        "card": CARD_ID,
        "written_utc": _now_utc(),
        "master_seed": MASTER_SEED,
        "draws": DRAWS,
        "catalog": {
            "path": r23.CATALOG_REL,
            "sha256": _sha256_file(NESTED_ROOT / r23.CATALOG_REL),
        },
        "sealed_archive": {
            "path": f"dev/00runs/{ARCHIVE['name']}",
            "pre_registration_sha256": _sha256_file(
                archive / ARCHIVE["pre_registration"]),
            "seal": seal,
        },
        "drivers": {
            "rereval_driver": "helpers/benchmarking/"
                              "rereval_feeddown_lineage.py",
            "rereval_driver_sha256": _sha256_file(Path(__file__).resolve()),
            "r23_driver": "helpers/benchmarking/"
                          "staged_r23_resonance_ablation.py",
            "r23_driver_sha256": _sha256_file(
                NESTED_ROOT / "helpers" / "benchmarking"
                / "staged_r23_resonance_ablation.py"),
        },
        "git": _git_info(),
        "environment": environment_fingerprint(),
    }
    _write_json(prov_path, provenance)
    print(
        f"rereval_feeddown_lineage: wrote {pre_path} and {prov_path} "
        f"(master seed {MASTER_SEED}, draws {DRAWS}; nothing computed)"
    )


def _require_pre_registration(run_dir: Path) -> None:
    if not (run_dir / "pre_registration.json").is_file():
        raise RuntimeError(
            f"rereval_feeddown_lineage: "
            f"{run_dir / 'pre_registration.json'} is missing; run "
            "--pre-registration before any evaluation"
        )


# ---------------------------------------------------------------------------
# Re-evaluation
# ---------------------------------------------------------------------------
def _run_rereval(run_dir: Path) -> None:
    _require_pre_registration(run_dir)
    archive = _archive_dir()
    _verify_archive_seal(archive)
    pre = json.loads(
        (archive / ARCHIVE["pre_registration"]).read_text(
            encoding="utf-8"))
    replicas = _load_replicas(archive, pre)
    catalog = HadronCatalog.from_json(NESTED_ROOT / r23.CATALOG_REL)
    corrected = build_corrected_feeddown(catalog, replicas)
    out_path = run_dir / "feeddown_corrected.json"
    _write_json(out_path, corrected)
    trigger_path = run_dir / "subcard_trigger.json"
    trigger = build_subcard_trigger(corrected)
    _write_json(trigger_path, trigger)
    summary = corrected["summary"]
    pooled = corrected["pooled"]
    print(
        "rereval_feeddown_lineage: wrote "
        f"{out_path} and {trigger_path} "
        f"({summary['n_checked']} species checked, all finite: "
        f"{summary['all_finite']}; max |sigma| "
        f"{summary['max_abs_sigma']:.4f} at "
        f"{summary['max_abs_sigma_species']}; expected "
        f"{pooled['expected_finals_per_new_kernel_source']:.4f} vs "
        f"observed "
        f"{pooled['observed_finals_per_new_kernel_source']:.4f} finals "
        f"per new-kernel source; trigger {trigger['trigger']})"
    )


# ---------------------------------------------------------------------------
# Seal and clean-environment validation (the card 2 driver's pattern)
# ---------------------------------------------------------------------------
_SEAL_REQUIRED = (
    "pre_registration.json",
    "provenance.json",
    "feeddown_corrected.json",
    "subcard_trigger.json",
)


def _run_seal(run_dir: Path) -> None:
    for required in _SEAL_REQUIRED:
        if not (run_dir / required).is_file():
            raise RuntimeError(
                f"rereval_feeddown_lineage: {run_dir / required} is "
                "missing; the seal requires the full evidence chain "
                "(pre-registration, re-evaluation, trigger)"
            )
    sums_path = run_dir / "SHA256SUMS.txt"
    manifest_path = run_dir / "seal_manifest.json"
    if sums_path.is_file() or manifest_path.is_file():
        raise RuntimeError(
            f"rereval_feeddown_lineage: {run_dir} is already sealed; "
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
        f"rereval_feeddown_lineage: sealed {run_dir} "
        f"({len(entries)} files, {total_bytes} bytes, SHA256SUMS.txt "
        f"sha256 {manifest['sha256_sums_sha256']})"
    )


#: The clean-interpreter probe: imports this driver module fresh,
#: re-runs the re-evaluation over the sealed run dir and the sealed
#: archive, and prints the canonical payload hashes (it writes nothing).
_CLEAN_PROBE_SOURCE = """\
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

driver_path, run_dir = sys.argv[1], sys.argv[2]
run_dir = Path(run_dir)
spec = importlib.util.spec_from_file_location(
    "rereval_feeddown_lineage_clean_probe", driver_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _canon(payload):
    return (json.dumps(payload, indent=2) + "\\n").encode("utf-8")


archive = module._archive_dir()
module._verify_archive_seal(archive)
pre = json.loads(
    (archive / module.ARCHIVE["pre_registration"]).read_text(
        encoding="utf-8"))
replicas = module._load_replicas(archive, pre)
catalog = module.HadronCatalog.from_json(
    module.NESTED_ROOT / module.r23.CATALOG_REL)
corrected = module.build_corrected_feeddown(catalog, replicas)
trigger = module.build_subcard_trigger(corrected)
hashes = {
    "feeddown_corrected.json": hashlib.sha256(_canon(corrected)).hexdigest(),
    "subcard_trigger.json": hashlib.sha256(_canon(trigger)).hexdigest(),
}
print(json.dumps({
    "hashes": hashes,
    "environment": module.environment_fingerprint(),
}))
"""


def _run_clean_validate(run_dir: Path) -> None:
    out_path = run_dir / "clean_environment_commands.json"
    if out_path.is_file():
        raise RuntimeError(
            f"rereval_feeddown_lineage: {out_path} exists; the "
            "clean-environment validation is recorded once per seal"
        )
    sums_path = run_dir / "SHA256SUMS.txt"
    manifest_path = run_dir / "seal_manifest.json"
    if not sums_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(
            f"rereval_feeddown_lineage: {run_dir} is not sealed; run "
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
            "rereval_feeddown_lineage: the seal verification failed "
            "for: " + ", ".join(mismatches)
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sha256_sums_sha256") != _sha256_file(sums_path):
        raise RuntimeError(
            "rereval_feeddown_lineage: the seal manifest's "
            "SHA256SUMS.txt digest does not match the sums file"
        )
    sealed = {
        name: _sha256_file(run_dir / name)
        for name in ("feeddown_corrected.json", "subcard_trigger.json")
    }
    driver_path = Path(__file__).resolve()
    argv = [
        sys.executable, "-c", _CLEAN_PROBE_SOURCE,
        str(driver_path), str(run_dir),
    ]
    completed = subprocess.run(argv, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "rereval_feeddown_lineage: the clean-interpreter probe "
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
                "purpose": "clean-interpreter re-run of the corrected "
                           "feed-down re-evaluation over the sealed R23 "
                           "archive (computes the corrected table and "
                           "the trigger payloads and prints their "
                           "hashes; writes nothing; both payloads are "
                           "deterministic functions of the sealed "
                           "archive and the committed code)",
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
            "rereval_feeddown_lineage: the clean-interpreter re-run "
            "does not reproduce the sealed payload hashes (sealed "
            f"{sealed}, recomputed {recomputed}); the record is in "
            f"{out_path}"
        )
    print(
        "rereval_feeddown_lineage: clean-environment validation "
        f"passed ({checked} sealed files verified; the re-run from a "
        "clean interpreter reproduces the sealed payload hashes); "
        f"record at {out_path}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rereval_feeddown_lineage.py",
        description=(
            "Round 3 card 3: the sealed-archive re-evaluation of the "
            "R23 feed-down lineage diagnostic under the corrected "
            "lineage convention, with the pre-registered "
            "transition-mismatch trigger decision.  Each invocation "
            "performs exactly one step and stops."
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pre-registration", action="store_true",
        help="write pre_registration.json and provenance.json into the "
             "run dir, then stop (nothing is computed)",
    )
    group.add_argument(
        "--rereval", action="store_true",
        help="re-evaluate the sealed R23 feed-down diagnostic under "
             "the corrected convention, write feeddown_corrected.json "
             "and subcard_trigger.json, then stop",
    )
    group.add_argument(
        "--seal", action="store_true",
        help="write seal_manifest.json and SHA256SUMS.txt over the "
             "run dir, then stop (the run dir is immutable after the "
             "seal)",
    )
    group.add_argument(
        "--clean-validate", action="store_true",
        help="verify the seal, re-run the re-evaluation from a clean "
             "interpreter, and confirm the payload hashes match; "
             "write clean_environment_commands.json, then stop",
    )
    parser.add_argument(
        "--run-dir", type=Path, default=DEFAULT_RUN_DIR,
        help=f"the run directory (default: {RUN_DIR_REL})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    if args.pre_registration:
        _run_pre_registration(run_dir)
    elif args.rereval:
        _run_rereval(run_dir)
    elif args.seal:
        _run_seal(run_dir)
    else:
        _run_clean_validate(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
