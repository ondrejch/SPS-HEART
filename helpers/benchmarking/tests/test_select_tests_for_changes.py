"""Tests for helpers/benchmarking/select_tests_for_changes.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from helpers.benchmarking.select_tests_for_changes import (
    NESTED_CORE_TESTS,
    module_imports,
    reverse_closure,
    select_helpers_tests,
    select_nested_tests,
)


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _nested_fixture(root: Path) -> None:
    _write(root, "na49_mixing/__init__.py", "from na49_mixing.core import core_fn\n")
    _write(root, "na49_mixing/core.py", "def core_fn():\n    return 1\n")
    _write(root, "na49_mixing/leaf.py", "from na49_mixing.core import core_fn\n")
    _write(root, "na49_mixing/unrelated.py", "def other():\n    return 2\n")
    _write(root, "tests/__init__.py", "")
    _write(root, "tests/test_core.py", "from na49_mixing.core import core_fn\n")
    _write(root, "tests/test_leaf.py", "from na49_mixing.leaf import core_fn\n")
    _write(root, "tests/test_unrelated.py", "from na49_mixing.unrelated import other\n")
    for name in (
        "test_conservation_laws.py",
        "test_event_conservation.py",
        "test_decay.py",
        "test_two_body_on_shell.py",
        "test_isospin.py",
    ):
        _write(root, f"tests/{name}", "def test_core():\n    assert True\n")


def test_select_nested_change_in_core_selects_direct_and_transitive(tmp_path: Path) -> None:
    _nested_fixture(tmp_path)
    selected = select_nested_tests({"na49_mixing/core.py"}, tmp_path)
    assert "tests/test_core.py" in selected
    assert "tests/test_leaf.py" in selected  # leaf imports core
    assert "tests/test_unrelated.py" not in selected
    assert set(NESTED_CORE_TESTS) <= selected


def _relative_import_fixture(root: Path) -> None:
    """The package-internal style of ``na49_mixing``: relative imports
    and an ``__init__`` that re-exports everything."""
    _write(root, "na49_mixing/__init__.py",
           "from .core import core_fn\nfrom .leaf import leaf_fn\nfrom .unrelated import other\n")
    _write(root, "na49_mixing/core.py", "def core_fn():\n    return 1\n")
    _write(root, "na49_mixing/leaf.py", "from .core import core_fn\n\ndef leaf_fn():\n    return core_fn()\n")
    _write(root, "na49_mixing/unrelated.py", "def other():\n    return 2\n")
    _write(root, "na49_mixing/sub/__init__.py", "")
    _write(root, "na49_mixing/sub/deep.py", "from ..leaf import leaf_fn\n")
    _write(root, "tests/__init__.py", "")
    _write(root, "tests/test_core.py", "from na49_mixing.core import core_fn\n")
    _write(root, "tests/test_leaf.py", "from na49_mixing.leaf import leaf_fn\n")
    _write(root, "tests/test_deep.py", "from na49_mixing.sub.deep import leaf_fn\n")
    _write(root, "tests/test_unrelated.py", "from na49_mixing.unrelated import other\n")
    for name in (
        "test_conservation_laws.py",
        "test_event_conservation.py",
        "test_decay.py",
        "test_two_body_on_shell.py",
        "test_isospin.py",
    ):
        _write(root, f"tests/{name}", "def test_core():\n    assert True\n")


def test_relative_imports_resolve_to_the_transitive_closure(tmp_path: Path) -> None:
    """``from .core import`` in ``leaf`` and ``from ..leaf import`` in
    ``sub.deep`` are seen: a change to ``core`` selects the leaf and the
    deep testers (FAB-01, 2026-09-28); the unrelated tester stays out
    although the package ``__init__`` re-exports it."""
    _relative_import_fixture(tmp_path)
    selected = select_nested_tests({"na49_mixing/core.py"}, tmp_path)
    assert {"tests/test_core.py", "tests/test_leaf.py", "tests/test_deep.py"} <= selected
    assert "tests/test_unrelated.py" not in selected


def test_package_init_changes_select_every_importer(tmp_path: Path) -> None:
    """``na49_mixing/__init__.py`` maps to the package name and a change
    to it reaches every test that imports the package."""
    from helpers.benchmarking.select_tests_for_changes import _module_name_for

    assert _module_name_for(Path("na49_mixing/__init__.py")) == "na49_mixing"
    assert _module_name_for(Path("na49_mixing/sub/__init__.py")) == "na49_mixing.sub"
    _relative_import_fixture(tmp_path)
    selected = select_nested_tests({"na49_mixing/__init__.py"}, tmp_path)
    assert {"tests/test_core.py", "tests/test_leaf.py", "tests/test_deep.py",
            "tests/test_unrelated.py"} <= selected


def test_relative_import_resolution_on_the_real_package() -> None:
    """On the committed package every module except pure leaves records
    at least one intra-package import, and a change to ``histogram``
    reaches its indirect importers."""
    from helpers.benchmarking.select_tests_for_changes import _package_module_map

    nested = Path(__file__).resolve().parents[3] / "dev" / "mixing2006"
    if not (nested / "na49_mixing").is_dir():
        return
    package_map = _package_module_map(nested, "na49_mixing")
    empty = sum(1 for imports in package_map.values() if not imports)
    assert empty < len(package_map) // 2, (empty, len(package_map))
    closure = reverse_closure({"na49_mixing.histogram"}, package_map)
    assert len(closure) >= 5, sorted(closure)


def test_select_nested_change_in_leaf_selects_only_leaf_tester(tmp_path: Path) -> None:
    _nested_fixture(tmp_path)
    selected = select_nested_tests({"na49_mixing/leaf.py"}, tmp_path)
    assert "tests/test_leaf.py" in selected
    assert "tests/test_core.py" not in selected
    assert "tests/test_unrelated.py" not in selected


def test_select_nested_direct_test_change_and_asset_reference(
    tmp_path: Path,
) -> None:
    _nested_fixture(tmp_path)
    _write(
        tmp_path,
        "tests/test_cards.py",
        'CARD = "na49_mixing/generators/michal/deltas-to-pion/delta++.gen"\n',
    )
    selected = select_nested_tests(
        {"na49_mixing/generators/michal/deltas-to-pion/delta++.gen"}, tmp_path
    )
    assert "tests/test_cards.py" in selected

    selected = select_nested_tests({"tests/test_unrelated.py"}, tmp_path)
    assert "tests/test_unrelated.py" in selected


def test_select_helpers_changed_module_selects_importing_tests(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "helpers/benchmarking/__init__.py", "")
    _write(tmp_path, "helpers/benchmarking/sps_core.py", "def f():\n    return 1\n")
    _write(tmp_path, "helpers/benchmarking/driver.py", "def d():\n    return 2\n")
    _write(tmp_path, "helpers/benchmarking/tests/__init__.py", "")
    _write(
        tmp_path,
        "helpers/benchmarking/tests/test_core.py",
        "from helpers.benchmarking.sps_core import f\n",
    )
    _write(
        tmp_path,
        "helpers/benchmarking/tests/test_driver.py",
        "from helpers.benchmarking.driver import d\n",
    )
    selected = select_helpers_tests(
        {"helpers/benchmarking/sps_core.py"}, tmp_path
    )
    assert selected == {"helpers/benchmarking/tests/test_core.py"}

    selected = select_helpers_tests({"helpers/benchmarking/tests/test_driver.py"}, tmp_path)
    assert selected == {"helpers/benchmarking/tests/test_driver.py"}


def test_select_helpers_other_file_change_selects_full_tree(tmp_path: Path) -> None:
    _write(tmp_path, "helpers/benchmarking/__init__.py", "")
    _write(tmp_path, "helpers/benchmarking/driver.py", "def d():\n    return 2\n")
    _write(tmp_path, "helpers/benchmarking/README.md", "docs\n")
    _write(tmp_path, "helpers/benchmarking/tests/__init__.py", "")
    _write(tmp_path, "helpers/benchmarking/tests/test_a.py", "def test_a():\n    pass\n")
    _write(tmp_path, "helpers/benchmarking/tests/test_b.py", "def test_b():\n    pass\n")
    selected = select_helpers_tests({"helpers/benchmarking/README.md"}, tmp_path)
    assert selected == {
        "helpers/benchmarking/tests/test_a.py",
        "helpers/benchmarking/tests/test_b.py",
    }


def test_module_imports_resolves_submodule_from_import() -> None:
    known = {"na49_mixing.core", "na49_mixing.core.sub"}
    imps = module_imports("from na49_mixing.core import sub\n", known)
    assert imps == {"na49_mixing.core", "na49_mixing.core.sub"}
    imps = module_imports("import na49_mixing.core.sub\n", known)
    assert imps == {"na49_mixing.core.sub"}


def test_reverse_closure_is_transitive() -> None:
    graph = {
        "na49_mixing.a": set(),
        "na49_mixing.b": {"na49_mixing.a"},
        "na49_mixing.c": {"na49_mixing.b"},
        "na49_mixing.d": {"na49_mixing.a"},
    }
    assert reverse_closure({"na49_mixing.a"}, graph) == {
        "na49_mixing.a",
        "na49_mixing.b",
        "na49_mixing.c",
        "na49_mixing.d",
    }


def _git_resolves_head(repo: Path) -> bool:
    probe = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", "HEAD"],
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


def test_cli_end_to_end_on_real_trees() -> None:
    # Against the real repo: diff HEAD..HEAD is empty, so the selection must
    # be exactly the pinned core (nested) plus no helper tests. A source-only
    # stage has no .git, and the selector exits when git diff cannot run.
    repo = Path(__file__).resolve().parents[3]
    nested = repo / "dev" / "mixing2006"
    if not (_git_resolves_head(repo) and _git_resolves_head(nested)):
        pytest.skip("git cannot resolve HEAD in the real source trees")
    script = (
        repo
        / "helpers"
        / "benchmarking"
        / "select_tests_for_changes.py"
    )
    out = subprocess.run(
        [sys.executable, str(script), "HEAD", "--tree", "nested"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.splitlines() == sorted(NESTED_CORE_TESTS)

    out = subprocess.run(
        [sys.executable, str(script), "HEAD", "--tree", "helpers"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 1  # no relevant tests -> error, not silent success
