#!/usr/bin/env python3
"""Write the SHA-256 inventory for a release-archive staging directory.

The release-archive recipe stages the two source trees, writes
``SHA256SUMS.txt`` at the staging root, and zips the staging directory. This
script produces that inventory. It hashes every regular file under
``--staging`` (excluding the inventory itself), one
``<sha256>  <relative-path>`` line per file in sorted path order, in the
layout ``sha256sum -c`` expects: LF line endings, forward slashes, and no
absolute paths. Rerunning the script on an unchanged staging tree
reproduces the inventory byte for byte.

With ``--digest-zip`` the script also prints the SHA-256 of the finished
archive. An archive cannot contain its own digest, so the digest is
reported alongside the archive path, size, and file count instead of being
written into ``SHA256SUMS.txt``.

Symlinks are refused rather than followed: the recipe deletes them from the
staging tree before this script runs, and hashing a link target would
diverge from what the zip stores.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

INVENTORY_NAME = "SHA256SUMS.txt"
READ_CHUNK = 1 << 20
DESCRIPTION = "Write a sorted SHA-256 inventory (SHA256SUMS.txt) for a release-archive staging directory."


def sha256_of(path: Path) -> str:
    """Return the SHA-256 hex digest of ``path``, read in fixed-size chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(READ_CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def staged_files(staging: Path) -> list[Path]:
    """Every regular file under ``staging``, sorted by relative path."""
    files: list[Path] = []
    for path in sorted(staging.rglob("*")):
        if path.is_symlink():
            raise SystemExit(f"symlink in staging tree (delete it first): {path}")
        if path.is_file():
            files.append(path)
    return files


def write_inventory(staging: Path, output: Path) -> int:
    """Write ``<sha256>  <relative-path>`` lines for every staged file.

    The inventory file itself is excluded from its own listing, so the
    returned count is the number of files the archive will carry besides
    the inventory.
    """
    staging = staging.resolve()
    try:
        inventory_rel = output.resolve().relative_to(staging).as_posix()
    except ValueError:
        inventory_rel = ""
    lines = []
    for path in staged_files(staging):
        rel = path.relative_to(staging).as_posix()
        if rel == inventory_rel:
            continue
        lines.append(f"{sha256_of(path)}  {rel}\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(lines), encoding="utf-8", newline="\n")
    return len(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--staging",
        type=Path,
        required=True,
        help="staging directory whose contents will be zipped",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"inventory path (default: <staging>/{INVENTORY_NAME})",
    )
    parser.add_argument(
        "--digest-zip",
        type=Path,
        default=None,
        metavar="ZIP",
        help="also print the SHA-256 of the finished archive ZIP",
    )
    args = parser.parse_args(argv)

    staging = args.staging.resolve()
    if not staging.is_dir():
        parser.error(f"--staging is not a directory: {args.staging}")
    output = (args.output or staging / INVENTORY_NAME).resolve()
    count = write_inventory(staging, output)
    print(f"wrote {count} entries to {output}")
    if args.digest_zip is not None:
        zip_path = args.digest_zip
        if not zip_path.is_file():
            parser.error(f"--digest-zip is not a file: {args.digest_zip}")
        print(f"{sha256_of(zip_path)}  {zip_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
