"""Integrity guard for the ``legacy_v1`` freeze (Round 1, card 1).

The manifest ``config/v2_singlet/legacy_v1_manifest.json`` pins the
staged-excitation two-source implementation at the verified 2026-09-11
baseline (outer tree HEAD ``56b10c737ee8314037d100505606e23214ab02eb``,
nested tree HEAD ``65b90dd40a0f4f598637927e903d3a1c6be5e4b9``, baseline
archive ``/home/o/NA49/2026/na49-res-source-2026-09-11.zip``).  The
``legacy_v1`` implementation is a negative-control model and must never
be silently modified into the new implementation.

Enforced here, against the manifest:

- every pinned regular file (``na49_mixing/``, ``config/``, ``tests/``,
  and both ``helpers/benchmarking/`` trees) must exist with the recorded
  sha256.  Modification or deletion fails and names the drifted paths.
- the binary NPZ fixtures under ``tests/fixtures/`` are pinned by name
  and size only.  They are untracked and regenerable via
  ``tests/fixtures/make_golden_fixtures.py``; a size mismatch fails,
  absence is not drift.
- the frozen M8 p+p parameter map must equal the module copy below.
  This keeps the "no parameter changes" clause of runs R00/R01
  checkable against a single source of truth.
- recorded run-directory names under ``dev/00runs/`` must still exist
  when the artifact store is present (fit outputs are preserved at the
  name level; the manifest does not re-hash the artifact store).
- the baseline archive identity (size and sha256) is verified whenever
  the archive file is present on the checkout.  When the archive path
  is absent the check is skipped with a recorded reason, and the
  inventory checks above have already run.

Files added under the pinned roots after the freeze raise a warning, not
a failure: the manifest pins the frozen set, and a sanctioned change
re-pins the inventory in its own commit (the documented exception is
TASK-20260911-r1-two-body-on-shell-repair).

The recorded git HEADs are the baseline identity.  They are deliberately
not re-checked by this test so that later commits do not fail the
freeze; the freeze-time verification is recorded in
``dev/00runs/r1_legacy_v1_freeze_2026-09-11/freeze_report.json``.

The manifest cannot contain its own digest; it is excluded from the
pinned inventory by path (``MANIFEST_SELF_REL``).

The ``m8_parameters_hash`` recorded in ``freeze_report.json`` is the
sha256 of ``json.dumps(m8_frozen_pp_parameters, sort_keys=True,
separators=(",", ":")).encode("utf-8")``.

Hermetic: standard library only, no external tables, no event
generation, safe to run on a clean archive extraction.
"""

from __future__ import annotations

import hashlib
import json
import re
import os
import warnings
from pathlib import Path

import pytest

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent  # dev/mixing2006
MONO_ROOT = NESTED_ROOT.parent.parent  # monorepo root: holds dev/ and helpers/

MANIFEST_REL = "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json"
MANIFEST_PATH = NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json"

# The manifest cannot pin its own bytes; it is excluded from the
# inventory by this relative path.
MANIFEST_SELF_REL = MANIFEST_REL

# Roots covered by source_tree_sha256, relative to the monorepo root.
PINNED_ROOTS = (
    "dev/mixing2006/na49_mixing",
    "dev/mixing2006/config",
    "dev/mixing2006/tests",
    "helpers/benchmarking",
    "dev/mixing2006/helpers/benchmarking",
)

# Tool-state directories that are never part of the source inventory.
EXCLUDED_DIR_NAMES = frozenset({"__pycache__", ".pytest_cache"})

# Binary NPZ fixtures are pinned by name and size only.
FIXTURES_NPZ_PREFIX = "dev/mixing2006/tests/fixtures/"
NPZ_SUFFIX = ".npz"

# Artifact store: run-directory names are recorded there.  The scratch
# area dev/00runs/tmp/ is excluded from the record.
RUNS_DIR_REL = "dev/00runs"
RUNS_SCRATCH_NAMES = frozenset({"tmp"})
# The freeze-time verification record; its presence marks the artifact store.
FREEZE_RUN_NAME = "r1_legacy_v1_freeze_2026-09-11"

# Verbatim copy of the frozen M8 p+p parameter set (combined-error
# optimum of dev/00runs/staged_binned_fit_mc_2026-09-06/, decay
# hypothesis H3).  The manifest must carry exactly these values.
FROZEN_M8_PARAMETERS = {
    "regge.alpha_exc0": 1.1418864566363858,
    "regge.effective_transfer_slope": 8.504140762083612,
    "regge.edge_power": 2.3769493784466396,
    "continuum.match_mass": 2.680194110727528,
    "continuum.pion_multiplicity_slope": 1.5226475471033067,
    "continuum.strange_lambda_strength": 0.30184606498191385,
    "continuum.strange_sigma_strength": 0.3777282734799746,
    "continuum.strange_pair_strength": 0.19187855210738686,
    "continuum.pair_strength": 0.08411858717003319,
    "continuum.pair_threshold_power": 1.7610429093564994,
}

_HEADING = "legacy_v1 manifest drift detected"
_GUIDANCE = (
    "Pinned legacy_v1 files may be changed only by a sanctioned, "
    "card-documented change that re-pins the inventory in its own "
    "commit (documented exception: TASK-20260911-r1-two-body-on-shell-repair)."
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_fixture_npz(rel_posix: str) -> bool:
    return rel_posix.startswith(FIXTURES_NPZ_PREFIX) and rel_posix.endswith(NPZ_SUFFIX)


def _compute_inventory():
    """Recompute the pinned inventory of the working tree.

    Returns ``(hashed, npz_sizes)``: relative posix paths mapped to
    sha256 digests for hash-pinned files, and to byte sizes for the
    binary NPZ fixtures.  Excludes ``__pycache__/`` and ``.pytest_cache/``
    directories, ``*.pyc`` files, the manifest itself, and (by moving
    them to ``npz_sizes``) the regenerable binary NPZ fixtures.
    """
    hashed: dict = {}
    npz_sizes: dict = {}
    for rel_root in PINNED_ROOTS:
        root = MONO_ROOT / rel_root
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDED_DIR_NAMES)
            for name in sorted(filenames):
                if name.endswith(".pyc"):
                    continue
                path = Path(dirpath) / name
                rel_posix = path.relative_to(MONO_ROOT).as_posix()
                if rel_posix == MANIFEST_SELF_REL:
                    continue
                if _is_fixture_npz(rel_posix):
                    npz_sizes[rel_posix] = path.stat().st_size
                    continue
                hashed[rel_posix] = _sha256_file(path)
    return hashed, npz_sizes


def test_legacy_v1_manifest_matches_working_tree():
    """Fail on any drift between the legacy_v1 manifest and the tree."""
    assert MANIFEST_PATH.is_file(), f"legacy_v1 manifest is missing: {MANIFEST_PATH}"
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == 1
    assert manifest["label"] == "legacy_v1"

    computed_hashed, computed_npz = _compute_inventory()
    recorded_hashed = manifest["source_tree_sha256"]
    recorded_npz = manifest["binary_npz_files"]["files"]

    changed = sorted(
        path
        for path in recorded_hashed.keys() & computed_hashed.keys()
        if recorded_hashed[path] != computed_hashed[path]
    )
    missing = sorted(recorded_hashed.keys() - computed_hashed.keys())
    added = sorted(computed_hashed.keys() - recorded_hashed.keys())
    npz_changed = sorted(
        path
        for path in recorded_npz.keys() & computed_npz.keys()
        if recorded_npz[path] != computed_npz[path]
    )

    problems = [_HEADING]
    for path in changed:
        problems.append(f"modified pinned file: {path}")
        problems.append(f"    recorded sha256: {recorded_hashed[path]}")
        problems.append(f"    actual    sha256: {computed_hashed[path]}")
    for path in missing:
        problems.append(f"deleted pinned file: {path}")
    for path in npz_changed:
        problems.append(
            f"binary npz size drift: {path} "
            f"(recorded {recorded_npz[path]} bytes, actual {computed_npz[path]} bytes)"
        )
    problems.append(_GUIDANCE)
    assert not changed and not missing and not npz_changed, "\n".join(problems)

    assert (
        manifest["frozen_fit_outputs"]["m8_frozen_pp_parameters"]
        == FROZEN_M8_PARAMETERS
    ), (
        "the frozen M8 p+p parameter map in the manifest no longer matches "
        "the legacy_v1 freeze record; runs R00/R01 pin these values"
    )

    # The artifact store counts as present when the freeze's own record is
    # there. A bare dev/00runs is no evidence: a test can create the
    # directory during a run on a checkout without the store.
    runs_dir = MONO_ROOT / RUNS_DIR_REL
    if (runs_dir / FREEZE_RUN_NAME / "freeze_report.json").is_file():
        absent_runs = [
            name
            for name in manifest["frozen_fit_outputs"]["run_descriptions"]
            if not (runs_dir / name).is_dir()
        ]
        assert not absent_runs, (
            "recorded run directories are missing under dev/00runs "
            "(fit outputs are preserved at the name level): "
            + ", ".join(absent_runs)
        )

    if added:
        warnings.warn(
            "files under the pinned legacy_v1 roots are not listed in the "
            "manifest (informational; additions do not fail the freeze): "
            + ", ".join(added),
            stacklevel=2,
        )

    baseline = manifest["baseline"]
    archive = Path(baseline["archive_path"])
    if not archive.is_file():
        pytest.skip(
            "baseline archive absent on this checkout "
            f"({baseline['archive_path']}); inventory drift checks passed "
            "and the archive identity check is skipped (recorded reason)"
        )
    actual_size = archive.stat().st_size
    assert actual_size == baseline["archive_size_bytes"], (
        f"baseline archive size drift: recorded "
        f"{baseline['archive_size_bytes']} bytes, actual {actual_size} bytes "
        f"({archive})"
    )
    actual_digest = _sha256_file(archive)
    assert actual_digest == baseline["archive_sha256"], (
        f"baseline archive sha256 drift: recorded "
        f"{baseline['archive_sha256']}, actual {actual_digest} ({archive})"
    )


# ----------------------------------------------------------------------
# The inventory count lives in the manifest (2026-09-28)
# ----------------------------------------------------------------------
def test_inventory_count_recorded_in_manifest():
    """The manifest records its own inventory size and the size matches
    the pinned set; the register tests read this value instead of
    carrying ``INVENTORY_COUNT = <n>`` literals, so a landing that adds
    files re-pins the manifest only."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest.get("inventory_count") == len(manifest["source_tree_sha256"]), (
        "the manifest's inventory_count field disagrees with its inventory")


def test_no_cross_file_inventory_count_literals():
    """No test module carries a hard-coded inventory count or scans another
    module's source for one (the count-pin chain of 2026-09 touched up
    to 49 files per landing and lagged the manifest once, review finding
    REV-6c778c6-01)."""
    offenders = []
    for path in sorted((NESTED_ROOT / "tests").glob("test_*.py")):
        if path.name == TEST_FILE.name:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            code = line.strip()
            if code.startswith("#"):
                continue  # the re-pin history comments may quote old counts
            if re.search(r'INVENTORY_COUNT\s*=\s*\d{3}|len\(keys\) == \d{3}|"INVENTORY_COUNT = \d', code):
                offenders.append(f"{path.name}: {code[:60]}")
    assert offenders == [], offenders

