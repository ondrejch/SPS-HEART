"""Copy compact M4-M13 run records into an external-review staging tree.

Large generated event samples stay excluded. The copied set is the
immutable compact record the 2026-09-11 source review asked for:
pre-registration JSON, report JSON, REPORT.md, checksums, and other
small text artifacts. Invoke from the monorepo root::

    python helpers/benchmarking/stage_review_evidence.py --staging "$STG"
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

COMPACT_RUNS = (
    "staged_binned_fit_2026-09-05",
    "staged_binned_fit_mc_2026-09-06",
    "pa_acceptance_audit_2026-09-05",
    "charge_complete_basis_2026-09-05",
    "staged_decay_compare_2026-09-05-r2",
    "staged_heldout_2026-09-07",
    "mass_law_compare_2026-09-07",
    "parameter_profiles_2026-09-07",
    "staged_heldout_ratios_pa_2026-09-10",
    "staged_centrality_sets_2026-09-11",
)

COMPACT_SUFFIXES = {".json", ".md", ".txt", ".sha256", ".csv"}
MAX_BYTES = 2 * 1024 * 1024


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_compact(path: Path) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    if path.suffix.lower() not in COMPACT_SUFFIXES:
        return False
    return path.stat().st_size <= MAX_BYTES


def stage_compact_evidence(
    staging: Path,
    *,
    runs_root: Path | None = None,
) -> list[Path]:
    """Copy compact files into ``staging/evidence/<run>/``. Returns copies."""
    src_root = runs_root if runs_root is not None else _repo_root() / "dev" / "00runs"
    dest_root = Path(staging) / "evidence"
    dest_root.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    missing: list[str] = []
    for name in COMPACT_RUNS:
        src = src_root / name
        if not src.is_dir():
            missing.append(name)
            continue
        dest = dest_root / name
        dest.mkdir(parents=True, exist_ok=True)
        for path in sorted(src.rglob("*")):
            if not _is_compact(path):
                continue
            relative = path.relative_to(src)
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied.append(target)
    index_lines = [
        "# Compact M4-M13 evidence",
        "",
        "Immutable compact run records staged for external review.",
        "Event samples, npz caches, and files larger than 2 MiB are omitted.",
        "",
    ]
    if missing:
        index_lines.append("Missing run directories (not staged):")
        index_lines.extend(f"- {name}" for name in missing)
        index_lines.append("")
    index_lines.append("Staged files:")
    for path in copied:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rel = path.relative_to(dest_root).as_posix()
        index_lines.append(f"- `{rel}` sha256={digest}")
    (dest_root / "README.md").write_text("\n".join(index_lines) + "\n")
    return copied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", required=True, help="archive staging root")
    parser.add_argument(
        "--runs-root",
        default=None,
        help="override dev/00runs (tests)",
    )
    args = parser.parse_args(argv)
    copied = stage_compact_evidence(
        Path(args.staging),
        runs_root=Path(args.runs_root) if args.runs_root else None,
    )
    print(f"staged {len(copied)} compact evidence files into {args.staging}/evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
