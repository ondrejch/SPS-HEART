"""Machine-readable per-run provenance records for the SPS benchmark drivers.

A production run directory carries ``run_provenance.json`` next to
``multispecies_constrained_summary.json``. The record names the source-tree
commits, or for a run from a release archive without git metadata the source
archive digest and/or the sha256 of the archive's verified file inventory
(``SHA256SUMS.txt``), the sha256 of the reproducibility manifest and of the
pinned requirements file,
the Python interpreter and key package versions, the platform, every input
file the run consumed with its sha256, the command line, the seed scheme,
and the relevant environment variables.

:func:`validate_run_provenance` re-reads a record and recomputes the recorded
hashes, so an auditor can check a finished run from its output directory and
a downstream consumer refuses an input whose recorded hash does not match the
file on disk. The module is side-effect-free apart from
:func:`write_run_provenance` writing one JSON file.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
from datetime import datetime, timedelta, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

RUN_PROVENANCE_FILENAME = "run_provenance.json"

# Version of the provenance record layout written by write_run_provenance.
RUN_PROVENANCE_SCHEMA_VERSION = 1

# Name of the release-archive file inventory (sha256sum -c format) that
# make_archive_checksums.py writes at the root of an extracted archive.
SOURCE_INVENTORY_FILENAME = "SHA256SUMS.txt"

# Key distributions recorded in every provenance record. Each lookup is
# guarded, so a missing optional package is listed under "not_installed"
# instead of failing the run.
DEFAULT_PROVENANCE_PACKAGES: Tuple[str, ...] = ("numpy", "scipy", "matplotlib")

REQUIRED_PROVENANCE_FIELDS: Tuple[str, ...] = (
    "$schema_version",
    "recorded_at_utc",
    "source",
    "lock_files",
    "python",
    "platform",
    "packages",
    "inputs",
    "argv",
    "env_overrides",
    "rng",
)

_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_GIT_COMMIT_RE = re.compile(r"\A[0-9a-f]{40}\Z")


def is_sha256(value: object) -> bool:
    """True when *value* is a lowercase 64-character hexadecimal sha256."""
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def is_git_commit(value: object) -> bool:
    """True when *value* is a lowercase 40-character hexadecimal commit id."""
    return isinstance(value, str) and bool(_GIT_COMMIT_RE.fullmatch(value))


def sha256_of_file(path: Path | str) -> str:
    """Return the sha256 hex digest of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(repo_dir: Path | str | None) -> Optional[str]:
    """Return the current commit of *repo_dir*, or None when git cannot tell.

    Any failure (git absent, no repository, timeout) yields None so a run
    from a release archive can still record a source archive digest instead.
    """
    if repo_dir is None:
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    head = completed.stdout.strip()
    return head if _GIT_COMMIT_RE.fullmatch(head) else None


# How many levels above the repository root the archive inventory probe
# climbs. The release-archive layout puts the top-level helpers at the
# archive root (depth 0) and the vendored tree one directory below it
# (``dev/mixing2006/``, depth 2 from its own REPO_ROOT); two levels of
# headroom covers both without wandering into unrelated directories.
_SOURCE_INVENTORY_MAX_DEPTH = 2


def find_source_inventory(repo_root: Optional[Path | str]) -> Optional[Path]:
    """Locate the release-archive file inventory at or above *repo_root*.

    The inventory (``SHA256SUMS.txt``) sits at the root of an extracted
    release archive. The top-level helper layout runs from that root
    directly; the vendored layout runs two levels below it
    (``<archive-root>/dev/mixing2006/``), so the probe climbs up to
    :data:`_SOURCE_INVENTORY_MAX_DEPTH` parents. Only regular files count;
    a missing inventory yields None.
    """
    if repo_root is None:
        return None
    root = Path(repo_root)
    candidate = root
    for _ in range(_SOURCE_INVENTORY_MAX_DEPTH + 1):
        probe = candidate / SOURCE_INVENTORY_FILENAME
        try:
            if probe.is_file():
                return probe
        except OSError:
            continue
        if candidate == candidate.parent:
            break
        candidate = candidate.parent
    return None


def collect_source_record(
    *,
    outer_repo: Optional[Path | str] = None,
    nested_repo: Optional[Path | str] = None,
    source_archive: str = "",
    source_inventory: Optional[Path | str] = None,
) -> Dict[str, object]:
    """Record both source-tree commits, with archive and inventory fallbacks.

    ``outer_repo`` is the monorepo root and ``nested_repo`` the
    dev/mixing2006 tree; each commit is read from git when the metadata is
    available. A run from a release archive carries no git metadata, so the
    caller injects the archive digest through ``source_archive`` (the
    ``--source-archive`` argument or the ``NA49_RUN_SOURCE_ARCHIVE``
    environment variable). When no git metadata is available at all, the
    verified file inventory of the release archive (``SHA256SUMS.txt``,
    found via :func:`find_source_inventory`) is digested into
    ``source_inventory_sha256`` as an automatic, immutable source-tree
    identity; ``source_inventory`` overrides the automatic search with an
    explicit path. When both paths resolve to the same repository (the
    standalone vendored layout), ``outer_head`` is null and a note says so
    instead of recording one commit twice.
    """
    outer_head = git_head(outer_repo) if outer_repo is not None else None
    nested_head = git_head(nested_repo) if nested_repo is not None else None
    archive = str(source_archive or "").strip()
    if archive.startswith("sha256:"):
        archive = archive[len("sha256:"):].strip()
    notes: List[str] = []
    if outer_repo is not None and nested_repo is not None:
        try:
            same_tree = Path(outer_repo).resolve() == Path(nested_repo).resolve()
        except OSError:
            same_tree = False
        if same_tree:
            outer_head = None
            notes.append(
                "outer_head omitted: outer and nested trees resolve to one git repository"
            )
    if source_inventory is not None:
        inventory_path = Path(source_inventory)
        if inventory_path.is_file():
            notes.append(
                f"source identity taken from the file inventory at {inventory_path}"
            )
        else:
            inventory_path = None
            notes.append(
                f"the source inventory path {source_inventory} is not a regular file; "
                "it was not recorded"
            )
    elif outer_head is None and nested_head is None:
        # No git identity at all: the extracted-archive fallback is the
        # verified file inventory at the repository root (or one level
        # above it for the vendored layout).
        inventory_path = find_source_inventory(outer_repo)
        if inventory_path is not None:
            notes.append(
                "no git metadata; source identity taken from the release-archive "
                f"file inventory {SOURCE_INVENTORY_FILENAME} at {inventory_path}"
            )
    else:
        inventory_path = None
    inventory_sha: Optional[str] = None
    if inventory_path is not None:
        inventory_sha = sha256_of_file(inventory_path)
    if (
        outer_head is None
        and nested_head is None
        and not archive
        and inventory_sha is None
    ):
        notes.append(
            "no git metadata, no source archive digest, and no source inventory "
            "recorded; the source trees could not be identified"
        )
    components: List[str] = []
    if outer_head is not None or nested_head is not None:
        components.append("git")
    if archive:
        components.append("source_archive")
    if inventory_sha is not None:
        components.append("source_inventory")
    recorded_from = "+".join(components) if components else "unresolved"
    return {
        "outer_head": outer_head,
        "nested_head": nested_head,
        "source_archive": archive if archive else None,
        "source_inventory_sha256": inventory_sha,
        "source_inventory_path": str(inventory_path) if inventory_path is not None else None,
        "recorded_from": recorded_from,
        "notes": notes,
    }


def collect_python_record() -> Dict[str, str]:
    """Record the running interpreter version and implementation."""
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
    }


def collect_platform_record() -> Dict[str, str]:
    """Record the platform: platform(), machine, and system."""
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "system": platform.system(),
    }


def collect_package_versions(
    package_names: Sequence[str] = ("numpy", "scipy", "matplotlib"),
) -> Dict[str, object]:
    """Record the installed version of each key package.

    Every package lookup is guarded: a package that is not installed (or
    whose metadata cannot be read) is listed under ``not_installed`` instead
    of failing the record.
    """
    versions: Dict[str, str] = {}
    not_installed: List[str] = []
    for name in package_names:
        try:
            versions[str(name)] = str(importlib_metadata.version(str(name)))
        except Exception:
            not_installed.append(str(name))
    return {"packages": versions, "not_installed": not_installed}


def lock_file_records(lock_specs: Sequence[tuple[str, Path | str]]) -> List[Dict[str, object]]:
    """Digest every lock file named by ``(role, path)`` pairs.

    A lock file that is not present is recorded with status "absent" and no
    digest instead of failing the run; the record states the expected path.
    """
    records: List[Dict[str, object]] = []
    for role, path in lock_specs:
        lock_path = Path(path)
        entry: Dict[str, object] = {"role": str(role), "path": str(lock_path)}
        if not lock_path.is_file():
            entry["status"] = "absent"
            entry["sha256"] = None
        else:
            try:
                entry["status"] = "present"
                entry["sha256"] = sha256_of_file(lock_path)
            except OSError:
                entry["status"] = "unreadable"
                entry["sha256"] = None
        records.append(entry)
    return records


def manifest_input_hashes(
    lock_path: Path | str,
    *,
    web_root: Optional[Path | str] = None,
    config_roots: Sequence[Path | str] = (),
) -> Dict[str, object]:
    """Read the input hashes recorded in the reproducibility manifest.

    The manifest covers the downloaded SPS HADRONS artifacts
    (``downloads.sps_web.artifacts``, resolved against ``web_root``) and the
    analysis configuration files (``analysis.configuration_files``, resolved
    against each directory in ``config_roots``). The returned dict carries
    ``manifest_path``, ``manifest`` (one of "absent", "unreadable",
    "verified"), and ``entries`` mapping file paths to the recorded sha256.
    A manifest that is absent yields status "absent" and no entries, so
    callers continue and record that fact in the run's provenance instead of
    failing. Manifest entries without a well-formed sha256 are skipped.
    """
    lock_file = Path(lock_path)
    if not lock_file.is_file():
        return {"manifest_path": str(lock_path), "manifest": "absent", "entries": {}}
    try:
        lock = json.loads(lock_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"manifest_path": str(lock_path), "manifest": "unreadable", "entries": {}}
    if not isinstance(lock, dict):
        return {"manifest_path": str(lock_path), "manifest": "unreadable", "entries": {}}
    entries: Dict[str, str] = {}

    downloads = lock.get("downloads")
    if isinstance(downloads, dict):
        sps_web = downloads.get("sps_web")
        artifacts = sps_web.get("artifacts") if isinstance(sps_web, dict) else None
        if isinstance(artifacts, list):
            for artifact in artifacts:
                if not isinstance(artifact, dict):
                    continue
                rel = str(artifact.get("path", "")).strip()
                sha = artifact.get("sha256")
                if not rel or not is_sha256(sha):
                    continue
                if web_root is not None:
                    _index_manifest_entry(entries, Path(web_root) / rel, str(sha))
                _index_manifest_entry(entries, Path(rel), str(sha))

    analysis = lock.get("analysis")
    configuration_files = (
        analysis.get("configuration_files") if isinstance(analysis, dict) else None
    )
    if isinstance(configuration_files, list):
        roots = [Path(root) for root in config_roots]
        for config_entry in configuration_files:
            if not isinstance(config_entry, dict):
                continue
            rel = str(config_entry.get("path", "")).strip()
            sha = config_entry.get("sha256")
            if not rel or not is_sha256(sha):
                continue
            for root in roots:
                _index_manifest_entry(entries, Path(root) / rel, str(sha))

    return {"manifest_path": str(lock_path), "manifest": "readable", "entries": entries}


def _index_manifest_entry(entries: Dict[str, str], candidate: Path, sha256: str) -> None:
    if not candidate.is_file():
        return
    entries[str(candidate)] = sha256
    try:
        entries[str(candidate.resolve())] = sha256
    except OSError:
        pass


def check_inputs_against_manifest(
    input_paths: Sequence[Path | str],
    manifest_info: Mapping[str, object],
) -> Dict[str, object]:
    """Verify every consumed input that the reproducibility manifest records.

    ``manifest_info`` comes from :func:`manifest_input_hashes`. For every
    consumed input that the manifest records, the file is rehashed and
    compared with the recorded value. Returns a summary dict with
    ``manifest`` (one of "absent", "unreadable", "matched", "mismatched"),
    the manifest path, the number of consumed inputs checked against the
    manifest, and the mismatch list; a consumer refuses to continue when the
    outcome is "mismatched".
    """
    entries_raw = manifest_info.get("entries")
    entries: Dict[str, str] = dict(entries_raw) if isinstance(entries_raw, dict) else {}
    manifest_state = str(manifest_info.get("manifest", "absent"))
    checked = 0
    mismatches: List[Dict[str, object]] = []
    if manifest_state == "readable":
        for raw in input_paths:
            path = Path(raw)
            expected = entries.get(str(path))
            if expected is None:
                try:
                    expected = entries.get(str(path.resolve()))
                except OSError:
                    expected = None
            if expected is None:
                continue
            checked += 1
            if not path.is_file():
                mismatches.append(
                    {"path": str(path), "expected": expected, "actual": None, "error": "file_missing"}
                )
                continue
            actual = sha256_of_file(path)
            if actual != expected:
                mismatches.append(
                    {"path": str(path), "expected": expected, "actual": actual}
                )
    state = manifest_state if manifest_state in ("absent", "unreadable") else "verified"
    if mismatches:
        state = "mismatched"
    elif manifest_state == "readable":
        state = "matched"
    return {
        "manifest": state,
        "manifest_path": str(manifest_info.get("manifest_path", "")),
        "inputs_with_manifest_entry": checked,
        "mismatches": mismatches,
    }


def manifest_crosscheck_problems(
    record: Mapping[str, object],
    manifest_entries: Mapping[str, str],
) -> List[str]:
    """Compare a run's recorded input hashes against the manifest entries.

    Every manifest entry whose file also appears in the record's ``inputs``
    must carry the same sha256 in both places; a disagreement means the
    recorded run and the manifest were written against different input
    states. Recorded inputs without a manifest entry are skipped: the
    manifest only covers the downloaded tables and the reference
    configuration files.
    """
    problems: List[str] = []
    inputs = record.get("inputs")
    if not isinstance(inputs, list):
        return problems
    recorded: Dict[str, str] = {}
    for entry in inputs:
        if isinstance(entry, dict) and is_sha256(entry.get("sha256")):
            recorded[str(entry.get("path"))] = str(entry["sha256"])
    for entry_path in sorted(manifest_entries):
        recorded_hash = recorded.get(entry_path)
        if recorded_hash is None:
            continue
        expected = str(manifest_entries[entry_path])
        if recorded_hash != expected:
            problems.append(
                f"recorded input hash for {entry_path} disagrees with the "
                f"reproducibility manifest: run recorded {recorded_hash}, "
                f"manifest records {expected}"
            )
    return problems


def write_run_provenance(
    run_dir: Path | str,
    *,
    input_paths: Sequence[Path | str],
    argv: Sequence[str],
    env_var_names: Sequence[str] = (),
    rng: Optional[Mapping[str, object]] = None,
    source_record: Optional[Mapping[str, object]] = None,
    lock_records: Optional[Sequence[Mapping[str, object]]] = None,
    input_verification: Optional[Mapping[str, object]] = None,
    producer: str = "",
    extra: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Write ``run_provenance.json`` into *run_dir* and return the record.

    The record names the source commits or the source archive digest, the
    lock-file digests, the interpreter and key package versions, the
    platform, every consumed input with its sha256, the command line, the
    relevant environment variables (an explicit allowlist; the whole
    environment is never dumped), and the random-number generator with its
    seeds. The only side effect is writing the JSON file; the returned
    mapping can be extended by the caller before other artifacts are
    written.
    """
    inputs: List[Dict[str, object]] = []
    seen: set[str] = set()
    for raw in input_paths or ():
        path = Path(raw)
        try:
            resolved = str(path.resolve())
        except OSError:
            resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        if not path.is_file():
            inputs.append({"path": resolved, "sha256": None, "error": "missing"})
            continue
        try:
            inputs.append({"path": resolved, "sha256": sha256_of_file(path)})
        except OSError as exc:
            inputs.append({"path": resolved, "sha256": None, "error": f"unreadable: {exc}"})
    record: Dict[str, object] = {
        "$schema_version": RUN_PROVENANCE_SCHEMA_VERSION,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "producer": str(producer),
        "source": dict(source_record or {}),
        "lock_files": [dict(entry) for entry in (lock_records or [])],
        "python": collect_python_record(),
        "platform": collect_platform_record(),
        "packages": collect_package_versions(),
        "inputs": inputs,
        "argv": [str(argument) for argument in argv],
        "env_overrides": {str(name): os.environ.get(name) for name in env_var_names},
        "rng": dict(rng or {}),
    }
    if input_verification is not None:
        record["input_verification"] = dict(input_verification)
    if extra:
        record.update(extra)
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / RUN_PROVENANCE_FILENAME).write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def validate_run_provenance(
    path: Path | str,
    *,
    expected_input_paths: Optional[Sequence[Path | str]] = None,
) -> List[str]:
    """Validate one ``run_provenance.json`` and return its problems.

    An empty list means the record is valid. Checks:

    - the file exists, parses as JSON, and holds a top-level object;
    - every required field is present, the schema version is supported, and
      the timestamp parses as ISO-8601 with a timezone whose UTC offset is
      zero (the writer promises a timezone-aware UTC value);
    - the source block names at least one git commit, a source archive
      digest, or a source inventory digest; every recorded commit is 40
      hexadecimal characters and every digest is 64 hexadecimal characters;
      ``recorded_from`` agrees with the populated source fields;
    - ``python.version`` and ``platform.platform`` are non-empty, and every
      package version is a non-empty string;
    - ``argv`` is a list, ``env_overrides`` is a mapping, and ``rng`` names
      the generator type and carries a seeds mapping;
    - every recorded input carries a non-empty path and a 64-hex sha256;
    - when ``expected_input_paths`` is given: every expected path is
      recorded, and each recorded hash still matches the file on disk after
      recomputation;
    - every lock-file record carries a known status, and a lock file
      recorded as present that still exists keeps the recorded digest.
    """
    provenance_path = Path(path)
    if not provenance_path.is_file():
        return [f"provenance file missing: {provenance_path}"]
    try:
        record = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"{provenance_path}: provenance file is not readable JSON ({exc})"]
    if not isinstance(record, dict):
        return [f"{provenance_path}: provenance payload must be a JSON object"]

    problems: List[str] = []
    for field in REQUIRED_PROVENANCE_FIELDS:
        if field not in record:
            problems.append(f"{provenance_path}: missing required provenance field {field!r}")

    schema_version = record.get("$schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        problems.append(
            f"{provenance_path}: $schema_version must be an integer, found {schema_version!r}"
        )
    elif schema_version != RUN_PROVENANCE_SCHEMA_VERSION:
        problems.append(
            f"{provenance_path}: unsupported $schema_version {schema_version}; "
            f"this validator reads version {RUN_PROVENANCE_SCHEMA_VERSION}"
        )

    timestamp = record.get("recorded_at_utc")
    if not isinstance(timestamp, str) or not timestamp.strip():
        problems.append(f"{provenance_path}: recorded_at_utc is missing or empty")
    else:
        try:
            parsed_ts = datetime.fromisoformat(timestamp)
        except ValueError:
            problems.append(
                f"{provenance_path}: recorded_at_utc {timestamp!r} is not an ISO-8601 timestamp"
            )
        else:
            if parsed_ts.tzinfo is None:
                problems.append(
                    f"{provenance_path}: recorded_at_utc {timestamp!r} is a naive "
                    f"timestamp; the field promises a timezone-aware UTC value"
                )
            elif parsed_ts.utcoffset() != timedelta(0):
                problems.append(
                    f"{provenance_path}: recorded_at_utc {timestamp!r} carries a "
                    f"non-zero UTC offset ({parsed_ts.utcoffset()}); the field "
                    f"is UTC"
                )

    source = record.get("source")
    if not isinstance(source, dict):
        problems.append(f"{provenance_path}: 'source' must be an object")
    else:
        for key in ("outer_head", "nested_head"):
            head = source.get(key)
            if head is not None and not _GIT_COMMIT_RE.fullmatch(str(head)):
                problems.append(
                    f"{provenance_path}: source.{key} must be null or a 40-hex "
                    f"commit id, found {head!r}"
                )
        archive = source.get("source_archive")
        if archive is not None and not is_sha256(archive):
            problems.append(
                f"{provenance_path}: source.source_archive must be null or a "
                f"64-hex sha256 digest, found {archive!r}"
            )
        inventory_digest = source.get("source_inventory_sha256")
        if inventory_digest is not None and not is_sha256(inventory_digest):
            problems.append(
                f"{provenance_path}: source.source_inventory_sha256 must be null "
                f"or a 64-hex sha256 digest, found {inventory_digest!r}"
            )
        has_commit = any(
            is_git_commit(source.get(key)) for key in ("outer_head", "nested_head")
        )
        has_archive = is_sha256(archive)
        has_inventory = is_sha256(inventory_digest)
        if not has_commit and not has_archive and not has_inventory:
            problems.append(
                f"{provenance_path}: 'source' records neither a git commit, a "
                f"source archive digest, nor a source inventory digest"
            )
        populated: List[str] = []
        if source.get("outer_head") is not None or source.get("nested_head") is not None:
            populated.append("git")
        if archive is not None:
            populated.append("source_archive")
        if inventory_digest is not None:
            populated.append("source_inventory")
        expected_from = "+".join(populated) if populated else "unresolved"
        if str(source.get("recorded_from", "")) != expected_from:
            problems.append(
                f"{provenance_path}: source.recorded_from "
                f"{source.get('recorded_from')!r} does not agree with the "
                f"populated source fields (expected {expected_from!r})"
            )

    python_info = record.get("python")
    if not isinstance(python_info, dict) or not str(python_info.get("version", "")).strip():
        problems.append(f"{provenance_path}: 'python.version' is missing or empty")
    platform_info = record.get("platform")
    if not isinstance(platform_info, dict) or not str(platform_info.get("platform", "")).strip():
        problems.append(f"{provenance_path}: 'platform.platform' is missing or empty")
    packages = record.get("packages")
    if not isinstance(packages, dict):
        problems.append(f"{provenance_path}: 'packages' must be an object of package versions")
    else:
        versions = packages.get("packages")
        if not isinstance(packages.get("packages"), dict):
            problems.append(f"{provenance_path}: 'packages.packages' must be an object of package name to version")
        else:
            for name, version in packages["packages"].items():
                if not isinstance(version, str) or not version.strip():
                    problems.append(f"{provenance_path}: packages.packages.{name} must record a non-empty version string")
        not_installed = packages.get("not_installed")
        if not_installed is not None and (
            not isinstance(not_installed, list) or not all(isinstance(name, str) for name in not_installed)
        ):
            problems.append(f"{provenance_path}: 'packages.not_installed' must be a list of package names")

    if not isinstance(record.get("argv"), list) or not all(
        isinstance(argument, str) for argument in record.get("argv", [])
    ):
        problems.append(f"{provenance_path}: 'argv' must be a list of strings")

    env_overrides = record.get("env_overrides")
    if not isinstance(env_overrides, dict):
        problems.append(f"{provenance_path}: 'env_overrides' must be an object")
    else:
        for name, value in env_overrides.items():
            if value is not None and not isinstance(value, str):
                problems.append(
                    f"{provenance_path}: env_overrides.{name} must be a string or null, found {value!r}"
                )

    rng = record.get("rng")
    if not isinstance(rng, dict):
        problems.append(f"{provenance_path}: 'rng' must be an object describing the random generator")
    else:
        if not str(rng.get("type", "")).strip():
            problems.append(f"{provenance_path}: 'rng.type' is missing or empty")
        seeds = rng.get("seeds")
        if not isinstance(seeds, dict):
            problems.append(f"{provenance_path}: 'rng.seeds' must be an object mapping seed names to values")

    inputs = record.get("inputs")
    if not isinstance(inputs, list):
        problems.append(f"{provenance_path}: 'inputs' must be a list of path/hash entries")
        inputs = []
    recorded_by_path: Dict[str, str] = {}
    for index, entry in enumerate(inputs):
        if not isinstance(entry, dict):
            problems.append(
                f"{provenance_path}: inputs[{index}] must be an object with 'path' and 'sha256'"
            )
            continue
        entry_path = entry.get("path")
        entry_hash = entry.get("sha256")
        if not isinstance(entry_path, str) or not entry_path.strip():
            problems.append(f"{provenance_path}: inputs[{index}].path is missing or empty")
            continue
        if not is_sha256(entry_hash):
            problems.append(
                f"{provenance_path}: inputs[{index}] ({entry_path}) sha256 has bad format "
                f"{entry_hash!r}; expected 64 lowercase hex characters"
            )
            continue
        recorded_by_path[str(entry_path)] = str(entry_hash)
        try:
            recorded_by_path[str(Path(entry_path).resolve())] = str(entry_hash)
        except OSError:
            pass

    if expected_input_paths is not None:
        for raw in expected_input_paths:
            path = Path(raw)
            recorded_hash = None
            for key in (str(path), str(path.resolve())):
                if key in recorded_by_path:
                    recorded_hash = recorded_by_path[key]
                    break
            if recorded_hash is None:
                problems.append(
                    f"{provenance_path}: no recorded input hash for expected input {path}"
                )
                continue
            if not path.is_file():
                problems.append(f"{provenance_path}: recorded input file is missing: {path}")
                continue
            actual = sha256_of_file(path)
            if actual != recorded_hash:
                problems.append(
                    f"{provenance_path}: input hash mismatch for {path}: recorded "
                    f"{recorded_hash}, file now hashes to {actual}"
                )

    lock_files = record.get("lock_files")
    if not isinstance(lock_files, list):
        problems.append(f"{provenance_path}: 'lock_files' must be a list")
    else:
        for index, entry in enumerate(lock_files):
            if not isinstance(entry, dict):
                problems.append(
                    f"{provenance_path}: lock_files[{index}] must be an object with role/path/status"
                )
                continue
            role = entry.get("role")
            lock_path = entry.get("path")
            status = entry.get("status")
            if not isinstance(role, str) or not role.strip():
                problems.append(f"{provenance_path}: lock_files[{index}].role is missing or empty")
            if not isinstance(lock_path, str) or not lock_path.strip():
                problems.append(f"{provenance_path}: lock_files[{index}].path is missing or empty")
            if status not in ("present", "absent", "unreadable"):
                problems.append(
                    f"{provenance_path}: lock_files[{index}].status must be one of "
                    f"'present', 'absent', 'unreadable'; found {status!r}"
                )
            if status == "present":
                recorded_digest = entry.get("sha256")
                if not is_sha256(recorded_digest):
                    problems.append(
                        f"{provenance_path}: lock_files[{index}] ({lock_path}) sha256 has bad "
                        f"format {recorded_digest!r}; expected 64 lowercase hex characters"
                    )
                    continue
                lock_on_disk = Path(str(lock_path))
                if lock_on_disk.is_file():
                    actual = sha256_of_file(lock_on_disk)
                    if actual != str(recorded_digest):
                        problems.append(
                            f"{provenance_path}: lock digest mismatch for {role}: recorded "
                            f"{recorded_digest}, file now hashes to {actual}"
                        )
    return problems
