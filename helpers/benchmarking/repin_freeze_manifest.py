#!/usr/bin/env python3
"""Re-pin the ``legacy_v1`` freeze manifest after a sanctioned change.

The freeze test (``dev/mixing2006/tests/test_legacy_v1_freeze.py``) pins
every regular file under five roots by sha256 and the binary NPZ
fixtures by size. A sanctioned, card-documented change re-pins that
inventory in its own commit. This tool performs the re-pin
mechanically, with the same walk as the test:

- roots: ``dev/mixing2006/na49_mixing``, ``dev/mixing2006/config``,
  ``dev/mixing2006/tests``, ``helpers/benchmarking``,
  ``dev/mixing2006/helpers/benchmarking``;
- excluded: ``__pycache__/`` and ``.pytest_cache/`` directories,
  ``*.pyc`` files, the manifest itself;
- ``tests/fixtures/*.npz`` are pinned by size under
  ``binary_npz_files``.

It rewrites ``source_tree_sha256`` (path-sorted), sets
``inventory_count``, refreshes the NPZ sizes, and appends one
``freeze_exceptions`` entry naming the card, the touched files
(added, modified, removed; path-sorted), and the reason. Nothing else
in the manifest changes. ``--dry-run`` prints the diff and writes
nothing.

Usage (from the monorepo root or the standalone tree)::

    python helpers/benchmarking/repin_freeze_manifest.py \
        --card TASK-YYYYMMDD-<slug> --reason "<one sentence>" [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

PINNED_ROOTS = (
    "dev/mixing2006/na49_mixing",
    "dev/mixing2006/config",
    "dev/mixing2006/tests",
    "helpers/benchmarking",
    "dev/mixing2006/helpers/benchmarking",
)
EXCLUDED_DIR_NAMES = frozenset({"__pycache__", ".pytest_cache"})
MANIFEST_REL = "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json"
FIXTURES_NPZ_PREFIX = "dev/mixing2006/tests/fixtures/"
NPZ_SUFFIX = ".npz"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_mono_root(script_path: Path) -> Path:
    """The monorepo root for the outer copy or the vendored copy."""
    here = script_path.resolve()
    for candidate in (here.parents[2], here.parents[4] if len(here.parents) > 4 else None):
        if candidate is not None and (candidate / "dev" / "mixing2006" / "na49_mixing").is_dir():
            return candidate
    raise SystemExit(f"error: cannot locate the monorepo root from {here}")


def compute_inventory(mono_root: Path) -> tuple[dict[str, str], dict[str, int]]:
    """The same walk as ``test_legacy_v1_freeze._compute_inventory``."""
    hashed: dict[str, str] = {}
    npz_sizes: dict[str, int] = {}
    for rel_root in PINNED_ROOTS:
        root = mono_root / rel_root
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDED_DIR_NAMES)
            for name in sorted(filenames):
                if name.endswith(".pyc"):
                    continue
                path = Path(dirpath) / name
                rel_posix = path.relative_to(mono_root).as_posix()
                if rel_posix == MANIFEST_REL:
                    continue
                if rel_posix.startswith(FIXTURES_NPZ_PREFIX) and rel_posix.endswith(NPZ_SUFFIX):
                    npz_sizes[rel_posix] = path.stat().st_size
                    continue
                hashed[rel_posix] = _sha256_file(path)
    return dict(sorted(hashed.items())), dict(sorted(npz_sizes.items()))


def _baseline_inventory(mono_root: Path, manifest: dict) -> dict[str, str]:
    """The inventory the change is measured against: the manifest at git
    HEAD of the nested tree when available (so repeated runs for one card
    report the same touched set and stay idempotent), else the current
    manifest."""
    nested = mono_root / "dev" / "mixing2006"
    rel = "config/v2_singlet/legacy_v1_manifest.json"
    try:
        import subprocess

        out = subprocess.run(
            ["git", "-C", str(nested), "show", f"HEAD:{rel}"],
            capture_output=True, check=False,
        )
        if out.returncode == 0 and out.stdout:
            return json.loads(out.stdout.decode("utf-8"))["source_tree_sha256"]
    except (OSError, ValueError, KeyError):
        pass
    return manifest["source_tree_sha256"]


def repin(mono_root: Path, card: str, reason: str, *, dry_run: bool) -> dict:
    manifest_path = mono_root / MANIFEST_REL
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = _baseline_inventory(mono_root, manifest)
    hashed, npz_sizes = compute_inventory(mono_root)
    added = sorted(set(hashed) - set(recorded))
    removed = sorted(set(recorded) - set(hashed))
    modified = sorted(p for p in set(hashed) & set(recorded) if hashed[p] != recorded[p])
    touched = sorted(set(added) | set(removed) | set(modified))
    summary = {
        "added": added,
        "removed": removed,
        "modified": modified,
        "inventory_count_before": len(recorded),
        "inventory_count_after": len(hashed),
    }
    if dry_run or not touched:
        return summary
    manifest["source_tree_sha256"] = hashed
    manifest["inventory_count"] = len(hashed)
    manifest["binary_npz_files"]["files"] = npz_sizes
    exceptions = [e for e in manifest.get("freeze_exceptions", []) if e.get("card") != card]
    exceptions.append({"card": card, "files": touched, "reason": reason})
    manifest["freeze_exceptions"] = exceptions
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--card", required=True, help="card or change identifier for the freeze exception")
    parser.add_argument("--reason", required=True, help="one-sentence reason recorded in the freeze exception")
    parser.add_argument("--dry-run", action="store_true", help="print the diff and write nothing")
    args = parser.parse_args(argv)
    mono_root = resolve_mono_root(Path(__file__))
    summary = repin(mono_root, args.card, args.reason, dry_run=args.dry_run)
    for key in ("added", "removed", "modified"):
        for rel in summary[key]:
            print(f"{key}: {rel}")
    print(
        f"inventory_count: {summary['inventory_count_before']} -> "
        f"{summary['inventory_count_after']}"
        + (" (dry run, nothing written)" if args.dry_run else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
