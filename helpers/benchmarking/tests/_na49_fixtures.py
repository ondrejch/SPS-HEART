"""Shared availability guards for external NA49/SPS data fixtures.

The NA49 tables live under ``spshadrons.web.cern.ch/data`` outside the Python
packages and may be absent in a hermetic checkout or release package.
Data-dependent tests must skip with an explicit reason instead of failing on
missing input files. Reuse these callables in any new data-dependent test:

    from _na49_fixtures import PI_MINUS_TABLE, PT_INTEGRATED_PIMINUS_TABLE
    from _na49_fixtures import _require_na49_fixtures

    def test_something() -> None:
        _require_na49_fixtures(PI_MINUS_TABLE, PT_INTEGRATED_PIMINUS_TABLE)
        ...
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
NA49_DATA_DIR = REPO_ROOT / "spshadrons.web.cern.ch" / "data"

PI_MINUS_TABLE = NA49_DATA_DIR / "xsect_piminus.txt"
PT_INTEGRATED_PIMINUS_TABLE = NA49_DATA_DIR / "xsect_ptintegrated_piminus.txt"


def _missing_na49_fixtures(*paths: Path | str) -> list[Path]:
    """Return the subset of fixture paths that are absent on disk."""
    return [Path(p) for p in paths if not Path(p).is_file()]


def _na49_fixture_available(*paths: Path | str) -> bool:
    """True when every listed NA49 fixture file exists."""
    return not _missing_na49_fixtures(*paths)


def _require_na49_fixtures(*paths: Path | str) -> None:
    """Skip the current test with an explicit reason when fixtures are absent."""
    missing = _missing_na49_fixtures(*paths)
    if missing:
        listed = ", ".join(str(p) for p in missing)
        pytest.skip(f"external NA49 fixtures absent (hermetic run): {listed}")
