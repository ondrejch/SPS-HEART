#!/usr/bin/env python3
"""Select the tests relevant to a set of source changes.

Part of the solve-and-verify test protocol (see ``.opencode/protocol.md``):
during a task card, each phase commits its evidence against the *targeted*
set returned by this script; the full suite runs only at the land gate.

Selection rules
---------------
nested tree (``dev/mixing2006``):

- every changed ``tests/test_*.py`` is selected directly;
- every changed ``na49_mixing/`` Python module selects every test module
  that imports it, directly or transitively (static AST import scan over
  the package, no code execution; relative imports are resolved against
  the importing module's package, and the package ``__init__`` never
  propagates the closure, since it re-exports every submodule);
- every changed non-Python asset under ``na49_mixing/`` (``.gen`` cards,
  configs) selects every test module whose source references the asset's
  directory name;
- changed ``examples/*.py`` select test modules referencing the script
  basename;
- a pinned core of physics-invariant tests is always selected
  (``test_conservation_laws``, ``test_event_conservation``, ``test_decay``,
  ``test_two_body_on_shell``, ``test_isospin``).

helper tree (``helpers/benchmarking``):

- every changed ``tests/test_*.py`` is selected directly;
- every changed helper module selects every helper test module that
  imports it (the drivers are leaf modules of the tests);
- any other changed helper file selects the whole helper test tree
  (conservative; there is no finer module map for scripts/configs).

Usage
-----
    python helpers/benchmarking/select_tests_for_changes.py BASE_REF
    python helpers/benchmarking/select_tests_for_changes.py BASE_REF --tree nested
    python helpers/benchmarking/select_tests_for_changes.py BASE_REF --tree helpers

``BASE_REF`` must resolve in both git trees (outer monorepo and nested
``dev/mixing2006``); the changed set is ``git diff --name-only BASE_REF..HEAD``
in each tree. Output is one test path per line, relative to the root the
corresponding pytest invocation expects (nested: run from ``dev/mixing2006``;
helpers: run from the monorepo root with ``PYTHONPATH=.``).
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

#: Physics-invariant tests that must run for every targeted set.
NESTED_CORE_TESTS = (
    "tests/test_conservation_laws.py",
    "tests/test_event_conservation.py",
    "tests/test_decay.py",
    "tests/test_two_body_on_shell.py",
    "tests/test_isospin.py",
)

_PKG_PREFIXES = ("na49_mixing.", "helpers.benchmarking.")


def _module_name_for(rel_path: Path) -> str | None:
    """Dotted module name of a .py path, or None."""
    if rel_path.suffix != ".py":
        return None
    parts = list(rel_path.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]
    if not parts or parts[0] not in ("na49_mixing", "helpers"):
        return None
    return ".".join(parts)


def _module_set(changed: set[str], prefix: str, exclude_prefix: str | None = None) -> set[str]:
    mods: set[str] = set()
    for rel in changed:
        if not rel.startswith(prefix):
            continue
        if exclude_prefix and rel.startswith(exclude_prefix):
            continue
        mod = _module_name_for(Path(rel))
        if mod:
            mods.add(mod)
    return mods


def _resolve_relative(package: str | None, level: int, module: str | None) -> str | None:
    """Absolute dotted name of a relative import made from ``package``.

    ``package`` is the importing module's package (for
    ``na49_mixing/histogram.py`` it is ``na49_mixing``; for a package
    ``__init__`` it is the package itself). ``level`` dots climb
    ``level - 1`` packages above it. Returns None when the import
    cannot be resolved (no package context or climbing past the root).
    """
    if not package or level <= 0:
        return None
    parts = package.split(".")
    climb = level - 1
    if climb > len(parts) - 1:
        return None
    base = ".".join(parts[: len(parts) - climb])
    return f"{base}.{module}" if module else base


def module_imports(
    py_text: str, known_modules: set[str], package: str | None = None
) -> set[str]:
    """Absolute package module names imported by ``py_text``.

    Records the full dotted name for ``import a.b.c``; for
    ``from a.b import c`` records ``a.b`` and, when resolvable against
    ``known_modules``, the submodule ``a.b.c`` as well. Relative imports
    (``from .b import c``, ``from . import b``) are resolved against
    ``package``, the importing module's package, so the package-internal
    import graph (which is written with relative imports) is seen.
    """
    names: set[str] = set()
    try:
        tree = ast.parse(py_text)
    except SyntaxError:
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = _resolve_relative(package, node.level, node.module)
                if base is None:
                    continue
                names.add(base)
                for alias in node.names:
                    if alias.name != "*":
                        cand = f"{base}.{alias.name}"
                        if cand in known_modules:
                            names.add(cand)
                continue
            if node.module is None:
                continue
            names.add(node.module)
            for alias in node.names:
                if alias.name != "*":
                    cand = f"{node.module}.{alias.name}"
                    if cand in known_modules:
                        names.add(cand)
    return {n for n in names if any(n.startswith(p) for p in _PKG_PREFIXES)}


def _prefix_dependent(importer: set[str], module: str) -> bool:
    return any(name == module or name.startswith(module + ".") for name in importer)


def reverse_closure(changed: set[str], imports_by_module: dict[str, set[str]]) -> set[str]:
    """All modules that import ``changed`` directly or transitively.

    A package root (a name without a dot, the package ``__init__``)
    re-exports its submodules and would otherwise mark every importer
    of any submodule as affected. It therefore joins the closure only
    when it is itself in ``changed``; it never propagates.
    """
    affected = set(changed)
    frontier = list(changed)
    while frontier:
        mod = frontier.pop()
        for other, imps in imports_by_module.items():
            if other in affected or "." not in other:
                continue
            if _prefix_dependent(imps, mod):
                affected.add(other)
                frontier.append(other)
    return affected


def _git_changed(repo_root: Path, base_ref: str) -> set[str]:
    out = subprocess.run(
        ["git", "-C", str(repo_root), "diff", "--name-only", f"{base_ref}..HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        raise SystemExit(
            f"error: git diff {base_ref}..HEAD failed in {repo_root}: "
            f"{out.stderr.strip()}"
        )
    return {line.strip() for line in out.stdout.splitlines() if line.strip()}


def _resolve_roots(script_path: Path) -> tuple[Path, Path]:
    """Return (outer_root, nested_root) for monorepo or standalone layout."""
    here = script_path.resolve()
    root = here.parents[2]
    if (root / "dev" / "mixing2006").is_dir():
        return root, root / "dev" / "mixing2006"
    if (root / "na49_mixing").is_dir():
        return root, root
    raise SystemExit(f"error: cannot locate the mixing2006 tree from {here}")


def _package_module_map(root: Path, pkg: str) -> dict[str, set[str]]:
    """module name -> imported names, for every .py of ``pkg`` under root."""
    pkg_dir = root / pkg.split(".")[0]
    if not pkg_dir.is_dir():
        return {}
    files: dict[str, Path] = {}
    for p in pkg_dir.rglob("*.py"):
        mod = _module_name_for(p.relative_to(root))
        if mod:
            files[mod] = p
    known = set(files)
    return {
        mod: module_imports(
            p.read_text(encoding="utf-8", errors="replace"),
            known,
            package=(mod if p.name == "__init__.py" else mod.rsplit(".", 1)[0]),
        )
        for mod, p in files.items()
    }


def select_nested_tests(changed: set[str], nested_root: Path) -> set[str]:
    """Targeted nested-tree test file paths (relative to ``nested_root``)."""
    selected: set[str] = set(NESTED_CORE_TESTS)

    for rel in changed:
        if rel.startswith("tests/") and rel.endswith(".py"):
            selected.add(rel)

    package_map = _package_module_map(nested_root, "na49_mixing")
    helper_map = _package_module_map(nested_root, "helpers.benchmarking")
    known = set(package_map) | set(helper_map)

    affected = reverse_closure(_module_set(changed, "na49_mixing/"), package_map)
    affected |= reverse_closure(_module_set(changed, "helpers/benchmarking/"), helper_map)

    string_refs: list[str] = []
    for rel in changed:
        if rel.startswith("na49_mixing/") and not rel.endswith(".py"):
            parts = Path(rel).parts
            if len(parts) > 1:
                string_refs.append(parts[-2])
        if rel.startswith("examples/") and rel.endswith(".py"):
            string_refs.append(Path(rel).name)

    tests_dir = nested_root / "tests"
    for test_file in sorted(tests_dir.glob("test_*.py")):
        src = test_file.read_text(encoding="utf-8", errors="replace")
        imps = module_imports(src, known)
        hit = any(_prefix_dependent(imps, m) for m in affected)
        if string_refs:
            hit = hit or any(ref in src for ref in string_refs)
        if hit:
            selected.add(str(test_file.relative_to(nested_root)))
    return {s for s in selected if (nested_root / s).is_file()}


def select_helpers_tests(changed: set[str], outer_root: Path) -> set[str]:
    """Targeted helper-tree test file paths (relative to ``outer_root``)."""
    tests_dir = outer_root / "helpers" / "benchmarking" / "tests"
    if not tests_dir.is_dir():
        return set()

    selected = {
        rel
        for rel in changed
        if rel.startswith("helpers/benchmarking/tests/") and rel.endswith(".py")
    }

    helper_map = _package_module_map(outer_root, "helpers.benchmarking")
    py_changed = _module_set(
        changed, "helpers/benchmarking/", exclude_prefix="helpers/benchmarking/tests/"
    )
    if py_changed:
        for mod in py_changed:
            for test_mod, imps in helper_map.items():
                if test_mod.startswith("helpers.benchmarking.tests.") and _prefix_dependent(
                    imps, mod
                ):
                    selected.add(
                        f"helpers/benchmarking/tests/{test_mod.rsplit('.', 1)[1]}.py"
                    )
    elif any(
        rel.startswith("helpers/benchmarking/")
        and not rel.startswith("helpers/benchmarking/tests/")
        for rel in changed
    ):
        selected |= {str(p.relative_to(outer_root)) for p in tests_dir.glob("test_*.py")}

    return {s for s in selected if (outer_root / s).is_file()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("base_ref", help="git ref to diff against (valid in both trees)")
    parser.add_argument(
        "--tree",
        choices=("nested", "helpers", "both"),
        default="both",
        help="which test tree to select for (default: both)",
    )
    args = parser.parse_args(argv)

    outer_root, nested_root = _resolve_roots(Path(__file__))
    nested_changed = _git_changed(nested_root, args.base_ref)
    mono_changed = _git_changed(outer_root, args.base_ref)

    out: list[str] = []
    if args.tree in ("nested", "both"):
        out += sorted(select_nested_tests(nested_changed, nested_root))
    if args.tree in ("helpers", "both"):
        out += sorted(select_helpers_tests(mono_changed, outer_root))
    if not out:
        print("error: no relevant tests found", file=sys.stderr)
        return 1
    for line in out:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
