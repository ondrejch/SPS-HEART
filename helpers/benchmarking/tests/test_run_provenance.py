"""Per-run provenance contract: writer, validator, and manifest refusal.

Covered here (committed behavior of helpers/benchmarking/run_provenance.py
and the provenance mode of benchmark_multispecies_constrained.py):

- the writer emits a complete schema-version-1 record with a git source
  pin (or a source archive digest, or the sha256 of the release-archive
  file inventory for a run from an extracted archive without git
  metadata), recomputable input hashes, and a closed env_override
  allowlist;
- collect_source_record records the file inventory automatically when no
  git metadata is available (probe at the repository root and one level
  above for the vendored layout), strips a leading "sha256:" prefix from
  the archive digest, and leaves a note for every identity it records or
  fails to record;
- validate_run_provenance accepts a consistent record and reports named
  problems for missing fields, bad hashes, unsupported schema versions,
  missing source pins, non-sha256 digests, naive or non-UTC timestamps,
  recorded_from fields that disagree with the populated source fields,
  edited inputs, unrecorded expected inputs, and stale lock-file digests;
- the run path records manifest verdicts (absent/matched/mismatched) and
  refuses a mismatched input with the expected and found digests;
- the --validate-provenance audit mode exits 0 on a consistent run
  directory and 1 with a problem list otherwise.

The git-dependent tests create throwaway repositories in tmp_path, so the
suite passes from an extracted archive that carries no git metadata.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, cast

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import helpers.benchmarking.benchmark_multispecies_constrained as bmc  # noqa: E402
from helpers.benchmarking import run_provenance as rp  # noqa: E402
from helpers.benchmarking.run_provenance import (  # noqa: E402
    REQUIRED_PROVENANCE_FIELDS,
    RUN_PROVENANCE_FILENAME,
    RUN_PROVENANCE_SCHEMA_VERSION,
    SOURCE_INVENTORY_FILENAME,
)

# Bare 64-hex digest: the validator requires exactly this format.
ARCHIVE_DIGEST = "b" * 64
ZERO_DIGEST = "0" * 64


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_input(
    tmp_path: Path, name: str = "xsect_pp_pi_minus.txt", content: bytes = b"table row v1\n"
) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _make_git_repo(path: Path, name: str, message: str) -> str:
    """Create a fresh single-commit git repository at *path*; return its head."""
    git_env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "provenance-test",
        "GIT_AUTHOR_EMAIL": "provenance-test@example.invalid",
        "GIT_COMMITTER_NAME": "provenance-test",
        "GIT_COMMITTER_EMAIL": "provenance-test@example.invalid",
    }

    def _git(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(path), *args],
            check=True,
            env=git_env,
            capture_output=True,
        )

    path.mkdir(parents=True, exist_ok=True)
    (path / "anchor.txt").write_text(f"{name} anchor\n", encoding="utf-8")
    _git("init", "-q")
    _git("add", "-A")
    _git("-c", "user.name=provenance-test", "-c", "user.email=provenance-test@example.invalid",
         "commit", "-q", "-m", message)
    commit = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        env=git_env,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit


def _inventory_content(rel_path: str = "dev/mixing2006/na49_mixing/__init__.py") -> str:
    """Plausible sha256sum -c inventory body for one entry."""
    return f"{ZERO_DIGEST}  {rel_path}\n"


def _write_inventory(repo_root: Path, rel_path: str = "dev/mixing2006/na49_mixing/__init__.py") -> Path:
    path = repo_root / SOURCE_INVENTORY_FILENAME
    path.write_text(_inventory_content(rel_path), encoding="utf-8")
    return path


def _inventory_sha256(inventory_path: Path) -> str:
    return hashlib.sha256(inventory_path.read_bytes()).hexdigest()


def _archive_source_record() -> Dict[str, object]:
    return rp.collect_source_record(outer_repo=None, nested_repo=None, source_archive=ARCHIVE_DIGEST)


def _write_valid_record(
    run_dir: Path,
    input_paths: Sequence[Path],
    *,
    source_record: Optional[Mapping[str, object]] = None,
    lock_specs: Sequence[Tuple[str, Path]] = (),
    input_verification: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    kwargs: Dict[str, object] = {}
    if input_verification is not None:
        kwargs["input_verification"] = dict(input_verification)
    return rp.write_run_provenance(
        run_dir,
        input_paths=input_paths,
        argv=["benchmark_multispecies_constrained", "--seed", "20260329"],
        env_var_names=bmc.PROVENANCE_ENV_ALLOWLIST,
        rng={
            "type": "numpy.random.Generator",
            "bit_generator": "PCG64",
            "seeds": {"base_seed": 20260329},
        },
        source_record=(
            dict(source_record) if source_record is not None else _archive_source_record()
        ),
        lock_records=rp.lock_file_records(list(lock_specs)),
        producer="test_run_provenance",
        **kwargs,  # type: ignore[arg-type]
    )


def _record_path(run_dir: Path) -> Path:
    return run_dir / RUN_PROVENANCE_FILENAME


def _notes(record: Dict[str, object]) -> List[str]:
    notes = record.get("notes")
    assert isinstance(notes, list)
    return [str(note) for note in notes]


def _rewrite_record(run_dir: Path, mutate: Callable[[Dict[str, object]], None]) -> None:
    path = _record_path(run_dir)
    record = json.loads(path.read_text(encoding="utf-8"))
    mutate(record)
    path.write_text(json.dumps(record), encoding="utf-8")


def _run_cli_audit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    run_dir: Path,
    lock: Path,
    tmp_path: Path,
) -> Tuple[object, str]:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_multispecies_constrained",
            "--validate-provenance",
            str(run_dir),
            "--repro-lock",
            str(lock),
            "--mixing2006-dir",
            str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        bmc.main()
    return excinfo.value.code, capsys.readouterr().out


# ---------------------------------------------------------------------------
# 1) writer: complete, structurally valid record
# ---------------------------------------------------------------------------


def test_writer_emits_complete_schema_v1_record_with_git_source_pin(tmp_path, monkeypatch):
    """Two throwaway repositories stand in for the monorepo and the vendored
    tree, so the test needs no git metadata from the checkout it runs in."""
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(var, raising=False)
    outer_repo = tmp_path / "outer"
    nested_repo = tmp_path / "outer" / "dev" / "mixing2006"
    outer_head = _make_git_repo(outer_repo, "outer", "outer commit")
    nested_head = _make_git_repo(nested_repo, "nested", "nested commit")

    inp = _write_input(tmp_path)
    run_dir = tmp_path / "run"
    record = _write_valid_record(
        run_dir,
        [inp],
        source_record=rp.collect_source_record(outer_repo=outer_repo, nested_repo=nested_repo),
    )

    for field in REQUIRED_PROVENANCE_FIELDS:
        assert field in record, f"missing top-level field {field}"
    assert record["$schema_version"] == RUN_PROVENANCE_SCHEMA_VERSION == 1
    assert record["producer"] == "test_run_provenance"

    parsed = rp.datetime.fromisoformat(str(record["recorded_at_utc"]))
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == rp.timedelta(0)

    # Two distinct repositories: the writer pins both 40-hex commit ids.
    source = record["source"]
    assert isinstance(source, dict)
    assert rp.is_git_commit(source["outer_head"])
    assert rp.is_git_commit(source["nested_head"])
    assert source["outer_head"] == outer_head
    assert source["nested_head"] == nested_head
    assert source["outer_head"] != source["nested_head"]
    assert source["recorded_from"] == "git"

    assert _record_path(run_dir).is_file()
    on_disk = json.loads(_record_path(run_dir).read_text(encoding="utf-8"))
    assert on_disk == record


def test_writer_hashes_inputs_with_recomputed_sha256(tmp_path):
    first = _write_input(tmp_path, "a.txt", b"alpha")
    second = _write_input(tmp_path, "b.txt", b"beta")
    record = _write_valid_record(tmp_path / "run", [first, second])

    inputs = record["inputs"]
    assert isinstance(inputs, list)
    assert len(inputs) == 2
    by_name = {Path(str(entry["path"])).name: entry["sha256"] for entry in inputs}
    assert by_name["a.txt"] == hashlib.sha256(b"alpha").hexdigest()
    assert by_name["b.txt"] == hashlib.sha256(b"beta").hexdigest()


def test_writer_deduplicates_resolved_inputs(tmp_path):
    first = _write_input(tmp_path, "a.txt", b"alpha")
    record = _write_valid_record(tmp_path / "run", [first, first, Path(first)])
    assert isinstance(record["inputs"], list)
    assert len(record["inputs"]) == 1


# ---------------------------------------------------------------------------
# 2) archive mode: no git metadata, source recorded from the override
# ---------------------------------------------------------------------------


def test_source_record_archive_mode_without_git(tmp_path):
    inp = _write_input(tmp_path)
    record = _write_valid_record(tmp_path / "run", [inp], source_record=_archive_source_record())

    source = record["source"]
    assert isinstance(source, dict)
    assert source["recorded_from"] == "source_archive"
    assert source["source_archive"] == ARCHIVE_DIGEST
    assert source["source_inventory_sha256"] is None
    assert source["outer_head"] is None
    assert source["nested_head"] is None
    assert source["notes"] == []
    # An archive-only source pin satisfies the validator.
    assert rp.validate_run_provenance(_record_path(tmp_path / "run")) == []


def test_source_record_normalizes_sha256_prefix(tmp_path):
    record = rp.collect_source_record(
        outer_repo=None, nested_repo=None, source_archive="sha256:" + ARCHIVE_DIGEST
    )
    assert record["source_archive"] == ARCHIVE_DIGEST
    assert record["recorded_from"] == "source_archive"


def test_source_record_unresolved_when_git_and_archive_missing(tmp_path, monkeypatch):
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(var, raising=False)
    record = rp.collect_source_record(outer_repo=tmp_path, nested_repo=None)
    assert record["outer_head"] is None
    assert record["nested_head"] is None
    assert record["source_archive"] is None
    assert record["source_inventory_sha256"] is None
    assert record["recorded_from"] == "unresolved"
    assert record["notes"], "unresolved source must leave a note"


def test_source_record_same_tree_deduplicates_outer_head(tmp_path, monkeypatch):
    """One throwaway repository used as both outer and nested tree records
    the commit once, with a note instead of a duplicated outer_head."""
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(var, raising=False)
    repo = tmp_path / "repo"
    head = _make_git_repo(repo, "same-tree", "single commit")
    record = rp.collect_source_record(outer_repo=repo, nested_repo=repo)
    assert record["outer_head"] is None
    assert record["nested_head"] == head
    assert isinstance(record["notes"], list)
    assert any("one git repository" in str(note) for note in record["notes"])
    assert record["recorded_from"] == "git"


# ---------------------------------------------------------------------------
# 2b) inventory mode: extracted archive without git metadata
# ---------------------------------------------------------------------------


def test_source_record_inventory_fallback_at_repository_root(tmp_path, monkeypatch):
    """Top-level layout: the code runs from the archive root, where the
    inventory sits; no git metadata anywhere."""
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(var, raising=False)
    inventory = _write_inventory(tmp_path)
    record = rp.collect_source_record(outer_repo=tmp_path, nested_repo=tmp_path / "dev" / "mixing2006")
    assert record["outer_head"] is None
    assert record["nested_head"] is None
    assert record["source_archive"] is None
    assert record["source_inventory_sha256"] == _inventory_sha256(inventory)
    assert record["source_inventory_path"] == str(inventory)
    assert record["recorded_from"] == "source_inventory"
    assert any(
        "file inventory" in note for note in _notes(record)
    ), "inventory identity must leave a note"


def test_source_record_inventory_fallback_one_level_above(tmp_path, monkeypatch):
    """Vendored layout: the code runs one level below the archive root, so
    the probe at the immediate parent must find the inventory."""
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(var, raising=False)
    inventory = _write_inventory(tmp_path)
    mix_root = tmp_path / "dev" / "mixing2006"
    mix_root.mkdir(parents=True)
    record = rp.collect_source_record(outer_repo=mix_root, nested_repo=mix_root)
    assert record["outer_head"] is None
    assert record["nested_head"] is None
    assert record["source_inventory_sha256"] == _inventory_sha256(inventory)
    assert record["source_inventory_path"] == str(inventory)
    assert record["recorded_from"] == "source_inventory"


def test_source_record_inventory_written_and_validated(tmp_path):
    """End to end: an inventory-only record written by the writer passes
    the validator, so an archive run without git or an archive digest is
    still auditable."""
    inventory = _write_inventory(tmp_path)
    source_record = rp.collect_source_record(
        outer_repo=tmp_path, nested_repo=tmp_path / "dev" / "mixing2006"
    )
    inp = _write_input(tmp_path)
    run_dir = tmp_path / "run"
    record = _write_valid_record(run_dir, [inp], source_record=source_record)
    source = record["source"]
    assert isinstance(source, dict)
    assert source["source_inventory_sha256"] == _inventory_sha256(inventory)
    assert rp.validate_run_provenance(_record_path(run_dir)) == []


def test_source_record_git_identity_takes_precedence_over_inventory(tmp_path, monkeypatch):
    """With git metadata present the inventory is not probed and the record
    stays a plain git pin."""
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(var, raising=False)
    _write_inventory(tmp_path)
    repo = tmp_path / "repo"
    head = _make_git_repo(repo, "outer", "outer commit")
    record = rp.collect_source_record(outer_repo=repo, nested_repo=repo)
    assert record["source_inventory_sha256"] is None
    assert record["source_inventory_path"] is None
    assert record["nested_head"] == head
    assert record["recorded_from"] == "git"


def test_source_record_explicit_inventory_path(tmp_path, monkeypatch):
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(var, raising=False)
    inventory = tmp_path / "elsewhere" / "inventory.txt"
    inventory.parent.mkdir(parents=True)
    inventory.write_text(_inventory_content(), encoding="utf-8")
    record = rp.collect_source_record(
        outer_repo=tmp_path, nested_repo=None, source_inventory=inventory
    )
    assert record["source_inventory_sha256"] == _inventory_sha256(inventory)
    assert record["source_inventory_path"] == str(inventory)
    assert record["recorded_from"] == "source_inventory"
    assert any(
        "source identity taken from the file inventory" in note for note in _notes(record)
    )


def test_source_record_explicit_inventory_path_missing_is_noted(tmp_path, monkeypatch):
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(var, raising=False)
    ghost = tmp_path / "no_such_inventory.txt"
    record = rp.collect_source_record(
        outer_repo=tmp_path, nested_repo=None, source_inventory=ghost
    )
    assert record["source_inventory_sha256"] is None
    assert record["source_inventory_path"] is None
    assert record["recorded_from"] == "unresolved"
    assert any("not a regular file" in note for note in _notes(record))


# ---------------------------------------------------------------------------
# 3) validator: accepts valid, reports named problems
# ---------------------------------------------------------------------------


def test_validator_accepts_valid_record(tmp_path):
    inp = _write_input(tmp_path)
    lock = tmp_path / "reproducibility.lock.json"
    lock.write_text(json.dumps({"downloads": {"sps_web": {"artifacts": []}}}), encoding="utf-8")
    run_dir = tmp_path / "run"
    _write_valid_record(
        run_dir,
        [inp],
        lock_specs=[
            ("reproducibility_manifest", lock),
            ("requirements_lock", tmp_path / "requirements-paper.lock.txt"),
        ],
    )
    path = _record_path(run_dir)
    assert rp.validate_run_provenance(path) == []
    assert rp.validate_run_provenance(path, expected_input_paths=[inp]) == []


def test_validator_reports_missing_required_field(tmp_path):
    inp = _write_input(tmp_path)
    run_dir = tmp_path / "run"
    _write_valid_record(run_dir, [inp])

    def _drop_python(record: Dict[str, object]) -> None:
        record.pop("python", None)

    _rewrite_record(run_dir, _drop_python)
    problems = rp.validate_run_provenance(_record_path(run_dir))
    assert any(
        "missing required provenance field" in problem and "'python'" in problem
        for problem in problems
    )


def test_validator_reports_non_hex_input_sha256(tmp_path):
    inp = _write_input(tmp_path)
    run_dir = tmp_path / "run"
    _write_valid_record(run_dir, [inp])

    def _corrupt_hash(record: Dict[str, object]) -> None:
        inputs = record["inputs"]
        assert isinstance(inputs, list)
        entry = inputs[0]
        assert isinstance(entry, dict)
        entry["sha256"] = "0" * 63

    _rewrite_record(run_dir, _corrupt_hash)
    problems = rp.validate_run_provenance(_record_path(run_dir))
    assert any("sha256 has bad format" in problem for problem in problems)


def test_validator_reports_unsupported_schema_version(tmp_path):
    inp = _write_input(tmp_path)
    run_dir = tmp_path / "run"
    _write_valid_record(run_dir, [inp])

    def _bump_schema(record: Dict[str, object]) -> None:
        record["$schema_version"] = 999

    _rewrite_record(run_dir, _bump_schema)
    problems = rp.validate_run_provenance(_record_path(run_dir))
    assert any("unsupported $schema_version 999" in problem for problem in problems)


def test_validator_reports_source_without_commit_or_digest(tmp_path):
    inp = _write_input(tmp_path)
    run_dir = tmp_path / "run"
    _write_valid_record(run_dir, [inp])

    def _drop_source(record: Dict[str, object]) -> None:
        record["source"] = {
            "outer_head": None,
            "nested_head": None,
            "source_archive": None,
            "source_inventory_sha256": None,
            "source_inventory_path": None,
            "recorded_from": "unresolved",
            "notes": [],
        }

    _rewrite_record(run_dir, _drop_source)
    problems = rp.validate_run_provenance(_record_path(run_dir))
    assert any(
        "records neither a git commit, a source archive digest, nor a source inventory digest"
        in problem
        for problem in problems
    )


def test_validator_reports_non_sha256_archive_digest(tmp_path):
    """An arbitrary digest string no longer passes as a source pin."""
    inp = _write_input(tmp_path)
    run_dir = tmp_path / "run"
    _write_valid_record(run_dir, [inp])

    def _tamper_archive(record: Dict[str, object]) -> None:
        cast(dict, record["source"])["source_archive"] = "unknown-digest"

    _rewrite_record(run_dir, _tamper_archive)
    problems = rp.validate_run_provenance(_record_path(run_dir))
    assert any(
        "source.source_archive must be null or a 64-hex sha256 digest" in problem
        for problem in problems
    )


def test_validator_reports_non_sha256_inventory_digest(tmp_path):
    inp = _write_input(tmp_path)
    run_dir = tmp_path / "run"
    _write_valid_record(run_dir, [inp])

    def _tamper_inventory(record: Dict[str, object]) -> None:
        cast(dict, record["source"])["source_inventory_sha256"] = "xyz"

    _rewrite_record(run_dir, _tamper_inventory)
    problems = rp.validate_run_provenance(_record_path(run_dir))
    assert any(
        "source.source_inventory_sha256 must be null or a 64-hex sha256 digest" in problem
        for problem in problems
    )


def test_validator_reports_recorded_from_mismatch(tmp_path):
    """recorded_from must name exactly the populated source fields."""
    inp = _write_input(tmp_path)
    run_dir = tmp_path / "run"
    _write_valid_record(run_dir, [inp])

    def _drop_archive_keep_from(record: Dict[str, object]) -> None:
        cast(dict, record["source"])["source_archive"] = None
        # recorded_from still claims the archive was populated.

    _rewrite_record(run_dir, _drop_archive_keep_from)
    problems = rp.validate_run_provenance(_record_path(run_dir))
    assert any(
        "source.recorded_from" in problem and "does not agree with the populated source fields"
        in problem
        for problem in problems
    )
    assert any(
        "records neither a git commit, a source archive digest, nor a source inventory digest"
        in problem
        for problem in problems
    )


def test_validator_reports_naive_timestamp(tmp_path):
    inp = _write_input(tmp_path)
    run_dir = tmp_path / "run"
    _write_valid_record(run_dir, [inp])

    def _naive(record: Dict[str, object]) -> None:
        record["recorded_at_utc"] = "2026-08-30T12:00:00"

    _rewrite_record(run_dir, _naive)
    problems = rp.validate_run_provenance(_record_path(run_dir))
    assert any("naive" in problem for problem in problems)


def test_validator_reports_non_utc_offset(tmp_path):
    inp = _write_input(tmp_path)
    run_dir = tmp_path / "run"
    _write_valid_record(run_dir, [inp])

    def _offset(record: Dict[str, object]) -> None:
        record["recorded_at_utc"] = "2026-08-30T14:00:00+02:00"

    _rewrite_record(run_dir, _offset)
    problems = rp.validate_run_provenance(_record_path(run_dir))
    assert any("non-zero UTC offset" in problem for problem in problems)


def test_validator_reports_input_hash_mismatch_for_edited_input(tmp_path):
    inp = _write_input(tmp_path, content=b"original bytes")
    run_dir = tmp_path / "run"
    _write_valid_record(run_dir, [inp])
    inp.write_bytes(b"tampered bytes")

    problems = rp.validate_run_provenance(_record_path(run_dir), expected_input_paths=[inp])
    assert any("input hash mismatch" in problem for problem in problems)
    assert any(
        hashlib.sha256(b"tampered bytes").hexdigest() in problem for problem in problems
    ), "problem must name the freshly computed digest"


def test_validator_reports_unrecorded_expected_input(tmp_path):
    inp = _write_input(tmp_path)
    run_dir = tmp_path / "run"
    _write_valid_record(run_dir, [inp])
    ghost = tmp_path / "ghost_input.txt"
    ghost.write_bytes(b"never recorded")

    problems = rp.validate_run_provenance(_record_path(run_dir), expected_input_paths=[ghost])
    assert any(
        "no recorded input hash for expected input" in problem for problem in problems
    )


# ---------------------------------------------------------------------------
# 4) stale lock digest
# ---------------------------------------------------------------------------


def test_validator_detects_stale_lock_digest(tmp_path):
    inp = _write_input(tmp_path)
    lock = tmp_path / "reproducibility.lock.json"
    lock.write_text('{"version": 1}', encoding="utf-8")
    run_dir = tmp_path / "run"
    _write_valid_record(run_dir, [inp], lock_specs=[("reproducibility_manifest", lock)])
    assert rp.validate_run_provenance(_record_path(run_dir)) == []

    lock.write_text('{"version": 2}', encoding="utf-8")
    problems = rp.validate_run_provenance(_record_path(run_dir))
    assert any(
        "lock digest mismatch" in problem and "reproducibility_manifest" in problem
        for problem in problems
    )


# ---------------------------------------------------------------------------
# 5) manifest verdicts on the run path (module functions, no subprocess)
# ---------------------------------------------------------------------------


def test_absent_manifest_records_absent_and_run_proceeds(tmp_path):
    inp = _write_input(tmp_path)
    info = rp.manifest_input_hashes(tmp_path / "no_such_reproducibility.lock.json")
    assert info["manifest"] == "absent"
    assert info["entries"] == {}

    check = rp.check_inputs_against_manifest([inp], info)
    assert check["manifest"] == "absent"
    assert check["inputs_with_manifest_entry"] == 0
    assert check["mismatches"] == []
    bmc._refuse_on_manifest_mismatch(check)  # run continues

    record = _write_valid_record(tmp_path / "run", [inp], input_verification=check)
    verification = record["input_verification"]
    assert isinstance(verification, dict)
    assert verification["manifest"] == "absent"


def test_matching_manifest_yields_matched(tmp_path):
    inp = _write_input(tmp_path, "downloads/xsect_pp_pi_minus.txt", b"table row v1")
    digest = hashlib.sha256(b"table row v1").hexdigest()
    lock = tmp_path / "reproducibility.lock.json"
    lock.write_text(
        json.dumps(
            {
                "downloads": {
                    "sps_web": {"artifacts": [{"path": str(inp), "sha256": digest}]}
                },
                "analysis": {"configuration_files": []},
            }
        ),
        encoding="utf-8",
    )
    info = rp.manifest_input_hashes(lock)
    assert info["manifest"] == "readable"
    entries = info["entries"]
    assert isinstance(entries, dict)
    assert digest in {str(value) for value in entries.values()}

    check = rp.check_inputs_against_manifest([inp], info)
    assert check["manifest"] == "matched"
    assert check["inputs_with_manifest_entry"] == 1
    bmc._refuse_on_manifest_mismatch(check)


def test_tampered_manifest_entry_refuses_with_expected_and_found(tmp_path):
    inp = _write_input(tmp_path, "downloads/xsect_pp_pi_minus.txt", b"table row v1")
    actual = hashlib.sha256(b"table row v1").hexdigest()
    lock = tmp_path / "reproducibility.lock.json"
    lock.write_text(
        json.dumps(
            {
                "downloads": {
                    "sps_web": {"artifacts": [{"path": str(inp), "sha256": ZERO_DIGEST}]}
                },
                "analysis": {"configuration_files": []},
            }
        ),
        encoding="utf-8",
    )
    check = rp.check_inputs_against_manifest([inp], rp.manifest_input_hashes(lock))
    assert check["manifest"] == "mismatched"
    mismatches = check["mismatches"]
    assert isinstance(mismatches, list) and mismatches
    assert mismatches[0]["expected"] == ZERO_DIGEST
    assert mismatches[0]["actual"] == actual

    with pytest.raises(SystemExit) as excinfo:
        bmc._refuse_on_manifest_mismatch(check)
    message = str(excinfo.value)
    assert ZERO_DIGEST in message, "refusal must name the manifest-expected digest"
    assert actual in message, "refusal must name the found digest"
    assert str(inp) in message


def test_unparsable_manifest_keeps_run_going_but_audit_refuses(
    tmp_path, monkeypatch, capsys
):
    inp = _write_input(tmp_path)
    lock = tmp_path / "reproducibility.lock.json"
    lock.write_text("{not json at all", encoding="utf-8")

    info = rp.manifest_input_hashes(lock)
    assert info["manifest"] == "unreadable"
    check = rp.check_inputs_against_manifest([inp], info)
    assert check["manifest"] == "unreadable"
    bmc._refuse_on_manifest_mismatch(check)  # run path continues on an unparsable manifest

    run_dir = tmp_path / "run"
    _write_valid_record(run_dir, [inp], lock_specs=[("reproducibility_manifest", lock)])
    code, out = _run_cli_audit(monkeypatch, capsys, run_dir, lock, tmp_path)
    assert code == 1
    assert "provenance INVALID" in out
    assert "not readable JSON" in out


# ---------------------------------------------------------------------------
# 6) --validate-provenance audit mode via the CLI main
# ---------------------------------------------------------------------------


def test_cli_validate_provenance_ok_on_synthetic_run_dir(tmp_path, monkeypatch, capsys):
    inp = _write_input(tmp_path, "tables/xsect_pp_pi_minus.txt", b"table row v1")
    lock = tmp_path / "reproducibility.lock.json"
    lock.write_text(
        json.dumps(
            {
                "downloads": {"sps_web": {"artifacts": []}},
                "analysis": {"configuration_files": []},
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    _write_valid_record(
        run_dir,
        [inp],
        lock_specs=[
            ("reproducibility_manifest", lock),
            ("requirements_lock", tmp_path / "requirements-paper.lock.txt"),
        ],
    )
    code, out = _run_cli_audit(monkeypatch, capsys, run_dir, lock, tmp_path)
    assert code == 0
    assert "provenance OK" in out
    assert "inputs=1" in out


def test_cli_validate_provenance_invalid_lists_input_hash_problem(tmp_path, monkeypatch, capsys):
    inp = _write_input(tmp_path, "tables/xsect_pp_pi_minus.txt", b"table row v1")
    lock = tmp_path / "reproducibility.lock.json"
    lock.write_text(json.dumps({"downloads": {"sps_web": {"artifacts": []}}}), encoding="utf-8")
    run_dir = tmp_path / "run"
    _write_valid_record(run_dir, [inp], lock_specs=[("reproducibility_manifest", lock)])
    inp.write_bytes(b"corrupted after the run")

    code, out = _run_cli_audit(monkeypatch, capsys, run_dir, lock, tmp_path)
    assert code == 1
    assert "provenance INVALID" in out
    assert "input hash mismatch" in out


def test_cli_validate_provenance_missing_run_dir_reports_problem(tmp_path, monkeypatch, capsys):
    lock = tmp_path / "reproducibility.lock.json"
    lock.write_text(json.dumps({"downloads": {"sps_web": {"artifacts": []}}}), encoding="utf-8")
    code, out = _run_cli_audit(monkeypatch, capsys, tmp_path / "no_such_run", lock, tmp_path)
    assert code == 1
    assert "provenance INVALID" in out
    assert "provenance file missing" in out


# ---------------------------------------------------------------------------
# 7) schema discipline: allowlist, argv typing, JSON round-trip
# ---------------------------------------------------------------------------


def test_env_overrides_allowlist_is_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("NA49_RUN_SOURCE_ARCHIVE", ARCHIVE_DIGEST)
    monkeypatch.setenv("NA49_BENCHMARK_NOT_ALLOWLISTED", "must-not-leak")
    inp = _write_input(tmp_path)
    record = _write_valid_record(tmp_path / "run", [inp])

    env = record["env_overrides"]
    assert isinstance(env, dict)
    assert env["NA49_RUN_SOURCE_ARCHIVE"] == ARCHIVE_DIGEST
    assert "NA49_BENCHMARK_NOT_ALLOWLISTED" not in env
    assert set(env) <= set(bmc.PROVENANCE_ENV_ALLOWLIST)


def test_record_is_json_round_trippable_with_string_argv(tmp_path):
    inp = _write_input(tmp_path)
    record = _write_valid_record(tmp_path / "run", [inp])

    argv = record["argv"]
    assert isinstance(argv, list)
    assert all(isinstance(argument, str) for argument in argv)
    rng = record["rng"]
    assert isinstance(rng, dict) and isinstance(rng["seeds"], dict)

    round_tripped = json.loads(json.dumps(record))
    assert round_tripped == record
