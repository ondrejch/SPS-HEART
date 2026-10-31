#!/usr/bin/env python3
"""Check drift between monorepo and mixing2006-vendored benchmarking helpers.

The two trees intentionally differ only in *layout path defaults*:

- monorepo: ``REPO_ROOT=01-NA49-res``, runs under ``dev/00runs``, package under
  ``dev/mixing2006``
- paper/standalone: ``REPO_ROOT=mixing2006``, runs under ``00runs``, package
  at repo root

Any residual logical drift after normalizing those path strings is a real
sync failure and should be fixed in both places (or by re-vendoring).

Exit codes:
  0 — trees match after layout normalization (or only layout-path diffs)
  1 — logical content drift detected
  2 — structural mismatch (file only on one side)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _normalize(text: str, *, side: str) -> str:
    """Map layout-specific path strings to a common monorepo dialect."""
    if side == "mix":
        # mixing2006 standalone dialect -> monorepo dialect
        text = text.replace('REPO_ROOT / "00runs"', 'REPO_ROOT / "dev" / "00runs"')
        text = text.replace("REPO_ROOT / '00runs'", "REPO_ROOT / 'dev' / '00runs'")
        text = text.replace('mixing2006_dir = REPO_ROOT', 'mixing2006_dir = REPO_ROOT / "dev" / "mixing2006"')
        text = text.replace(
            'mixing2006_dir = REPO_ROOT / "dev" / "mixing2006" / "dev" / "mixing2006"',
            'mixing2006_dir = REPO_ROOT / "dev" / "mixing2006"',
        )
        text = text.replace('default=str(REPO_ROOT)', 'default=str(REPO_ROOT / "dev" / "mixing2006")')
        text = text.replace(
            'default=str(REPO_ROOT / "dev" / "mixing2006" / "dev" / "mixing2006")',
            'default=str(REPO_ROOT / "dev" / "mixing2006")',
        )
        # remote / local external generator roots
        text = text.replace("{remote_root}/external/", "{remote_root}/dev/external/")
        text = text.replace("{remote_root}/dev/dev/external/", "{remote_root}/dev/external/")
        text = text.replace('REPO_ROOT / "external"', 'REPO_ROOT / "dev" / "external"')
        # sync path lists in gateway orchestrators (exact multiline form only)
        text = text.replace(
            '    "na49_mixing",\n    "pyproject.toml",\n',
            '    "dev/mixing2006/na49_mixing",\n    "dev/mixing2006/pyproject.toml",\n',
        )
        # paper figure defaults (standalone often writes into cwd)
        text = text.replace('default="."', 'default="latex/paper/fig"')
        text = text.replace(
            'default="spectra_sharedfit_bestmix_all6_species_grid.png"',
            'default="latex/paper/fig/spectra_sharedfit_bestmix_all6_species_grid.png"',
        )
        # JSON config: standalone package root is "."
        text = text.replace('"mixing2006_dir": "."', '"mixing2006_dir": "dev/mixing2006"')
        # Common default path strings used in argparse defaults / docs
        text = text.replace("uses 00runs/", "uses dev/00runs/")
        text = text.replace("defaults to 00runs/", "defaults to dev/00runs/")
        text = text.replace("default: 00runs/", "default: dev/00runs/")
        text = re.sub(r"(?<!dev/)00runs/", "dev/00runs/", text)
        text = text.replace("dev/dev/00runs/", "dev/00runs/")
    # Absolute monorepo machine paths in JSON provenance are layout-local.
    text = re.sub(
        r"/home/[^/\s\"]+/NA49/2026/01-NA49-res/",
        "",
        text,
    )
    # strip trailing whitespace noise
    text = re.sub(r"[ \t]+$", "", text, flags=re.M)
    return text


def _iter_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if "__pycache__" in p.parts or p.suffix in {".pyc", ".pyo"}:
            continue
        out.append(p)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--monorepo-helpers",
        type=Path,
        default=None,
        help="Path to monorepo helpers/benchmarking (default: this file's directory)",
    )
    parser.add_argument(
        "--mix-helpers",
        type=Path,
        default=None,
        help="Path to mixing2006 helpers/benchmarking",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print a one-line summary and non-zero exit on failure",
    )
    args = parser.parse_args(argv)

    mono = (args.monorepo_helpers or Path(__file__).resolve().parent).resolve()
    mix = (
        args.mix_helpers
        or (mono.parents[1] / "dev" / "mixing2006" / "helpers" / "benchmarking")
    ).resolve()

    if not mono.is_dir():
        print(f"ERROR: monorepo helpers not found: {mono}", file=sys.stderr)
        return 2
    if not mix.is_dir():
        print(f"ERROR: mixing2006 helpers not found: {mix}", file=sys.stderr)
        return 2

    mono_files = {p.relative_to(mono): p for p in _iter_files(mono)}
    mix_files = {p.relative_to(mix): p for p in _iter_files(mix)}

    # Tests + the sync checker itself live only on the monorepo side by design.
    skip_prefixes = ("tests/",)
    skip_names = {"check_helpers_layout_sync.py"}
    mono_keys = {
        k
        for k in mono_files
        if not str(k).startswith(skip_prefixes) and k.name not in skip_names
    }
    mix_keys = {k for k in mix_files if k.name not in skip_names}

    only_mono = sorted(mono_keys - mix_keys)
    only_mix = sorted(mix_keys - mono_keys)
    shared = sorted(mono_keys & mix_keys)

    logic_drift: list[str] = []
    layout_only = 0
    identical = 0

    for rel in shared:
        raw_a = mono_files[rel].read_text(encoding="utf-8", errors="replace")
        raw_b = mix_files[rel].read_text(encoding="utf-8", errors="replace")
        if raw_a == raw_b:
            identical += 1
            continue
        if _normalize(raw_a, side="mono") == _normalize(raw_b, side="mix"):
            layout_only += 1
            continue
        logic_drift.append(str(rel))

    status = 0
    if only_mono or only_mix or logic_drift:
        status = 1 if logic_drift else 2

    if not args.quiet or status != 0:
        print(f"monorepo: {mono}")
        print(f"mixing2006: {mix}")
        print(
            f"shared={len(shared)} identical={identical} "
            f"layout_only={layout_only} logic_drift={len(logic_drift)}"
        )
        if only_mono:
            print("only in monorepo (excluding tests/):")
            for r in only_mono:
                print(f"  + {r}")
        if only_mix:
            print("only in mixing2006:")
            for r in only_mix:
                print(f"  + {r}")
        if logic_drift:
            print("logical drift (after layout normalization):")
            for r in logic_drift:
                print(f"  ! {r}")
        if status == 0:
            print("OK: no logical drift (layout-only differences allowed).")
        else:
            print("FAIL: helper trees are out of sync.", file=sys.stderr)

    return status


if __name__ == "__main__":
    raise SystemExit(main())
