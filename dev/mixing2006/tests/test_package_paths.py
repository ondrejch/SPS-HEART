"""Where the package finds ``config/``: the checkout, or the copy bundled into a wheel."""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

from na49_mixing import _paths

NESTED = Path(__file__).resolve().parents[1]


def _load_run_contract():
    path = NESTED / "helpers" / "benchmarking" / "run_contract.py"
    spec = importlib.util.spec_from_file_location("run_contract_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkout_root_is_the_nested_tree():
    assert _paths.nested_root() == NESTED
    assert (_paths.nested_root() / "config" / "v2_singlet" / "resonance_catalog.json").is_file()


def test_checkout_takes_precedence_over_a_bundled_copy(tmp_path, monkeypatch):
    pkg = tmp_path / "na49_mixing"
    (pkg / "_data" / "config").mkdir(parents=True)
    (tmp_path / "config").mkdir()
    monkeypatch.setattr(_paths, "PACKAGE_DIR", pkg)
    monkeypatch.setattr(_paths, "BUNDLED_ROOT", pkg / "_data")
    assert _paths.nested_root() == tmp_path


def test_installed_layout_uses_the_bundled_copy(tmp_path, monkeypatch):
    pkg = tmp_path / "na49_mixing"
    (pkg / "_data" / "config").mkdir(parents=True)
    monkeypatch.setattr(_paths, "PACKAGE_DIR", pkg)
    monkeypatch.setattr(_paths, "BUNDLED_ROOT", pkg / "_data")
    assert _paths.nested_root() == pkg / "_data"


def test_locate_roots_in_an_installed_layout(tmp_path):
    """In site-packages, na49_mixing and helpers are siblings and config is bundled:
    the nested root is the bundle and the mono root is the install directory."""
    site = tmp_path / "site-packages"
    (site / "na49_mixing" / "_data" / "config").mkdir(parents=True)
    driver = site / "helpers" / "benchmarking" / "run_engine.py"
    driver.parent.mkdir(parents=True)
    shutil.copy(NESTED / "helpers" / "benchmarking" / "run_contract.py", driver.parent / "run_contract.py")
    driver.write_text("")
    rc = _load_run_contract()
    nested, mono = rc.locate_roots(driver)
    assert nested == (site / "na49_mixing" / "_data").resolve()
    assert mono == site.resolve()


def test_locate_roots_in_the_checkout():
    rc = _load_run_contract()
    nested, mono = rc.locate_roots(NESTED / "helpers" / "benchmarking" / "run_engine.py")
    assert nested == NESTED
