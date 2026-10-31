"""Hermetic smoke test for the staged source-grid driver (Round 2, card 9).

Task ``TASK-20260913-r2-source-grid-runner`` (plan section 22, task 17;
the runs R21/R22 of plan section 8, lines 559-565).  The tests exercise
the driver's ``--smoke`` path (2 points, k = 0 and 1, x 200 decays x 1
replica, hermetic) into ``tmp_path`` run dirs and check the card's
deterministic contract:

- ``test_smoke_determinism``: two smoke runs with the same arguments
  produce byte-identical ``point_00.json`` / ``point_01.json``.
- ``test_charge_balance``: every point's ``charge_balance`` block shows
  the exact source ``(B, Q)`` over all events.
- ``test_open_set``: the smoke points' ``open_records`` match the
  pre-registered open sets for those points, recomputed independently
  from the committed catalog with the card's truncated-spectral-window
  rule and cross-checked against the driver's ``pre_registration.json``.
- ``test_degenerate_window``: point 0 (the grid start) uses the
  degenerate-window rule (mass fixed at ``M_k``) and the point-0
  diagnostics are finite and non-degenerate in the multiplicity means.
- ``test_layout_sync_inputs``: the nested and outer copies of the
  driver satisfy the ``check_helpers_layout_sync.py`` contract (the
  copies are byte-identical, the strongest form of the layout-only
  allowance; the checker's own normalization keeps them equal too).

The smoke runs read the committed catalog
``config/v2_singlet/resonance_catalog.json`` and the vendored PDG
table only; no NA49/SPS external tables are touched.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

NESTED_ROOT = Path(__file__).resolve().parent.parent
DRIVER_REL = Path("helpers") / "benchmarking" / "staged_source_grid.py"
NESTED_DRIVER = NESTED_ROOT / DRIVER_REL
OUTER_DRIVER = NESTED_ROOT.parent.parent / DRIVER_REL
SYNC_CHECKER = NESTED_ROOT.parent.parent / "helpers" / "benchmarking" / (
    "check_helpers_layout_sync.py"
)
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"

#: The pre-registered source types and their grids (round 2 index
#: item 6): the source (B, Q, S) and the grid start M_0.
SOURCE_TYPES = ("baryonic", "mesonic")
SOURCE_QN = {"baryonic": (1, 1, 0), "mesonic": (0, 0, 0)}
GRID_START = {"baryonic": 1.07324891, "mesonic": 0.2699532}

#: The committed-catalog open sets at the smoke points (the stable
#: identity record at the grid start; the first resonances at point 1).
EXPECTED_OPEN_SETS = {
    "baryonic": {0: [2212], 1: [2212, 2214]},
    "mesonic": {0: [111], 1: [111, 113, 100113, 30113]},
}

#: The smoke statistics (the driver's --smoke mode).
SMOKE_POINTS = 2
SMOKE_DECAYS = 200
SMOKE_REPLICAS = 1


def _run_smoke(run_dir: Path, source_type: str) -> None:
    """Run the driver's ``--smoke`` mode into *run_dir* (a subprocess;
    the driver locates the package itself, so the environment carries
    no PYTHONPATH dependency)."""
    result = subprocess.run(
        [
            sys.executable, str(NESTED_DRIVER),
            "--smoke", "--source-type", str(source_type),
            "--run-dir", str(run_dir),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"the {source_type} smoke run failed "
        f"(exit {result.returncode}):\n{result.stdout}\n{result.stderr}"
    )


@pytest.fixture(scope="module")
def smoke_runs(tmp_path_factory):
    """Two smoke run dirs per source type; the second exists for the
    byte-determinism check."""
    runs: dict[str, tuple[Path, Path]] = {}
    for source_type in SOURCE_TYPES:
        first = tmp_path_factory.mktemp(f"smoke_first_{source_type}")
        second = tmp_path_factory.mktemp(f"smoke_second_{source_type}")
        _run_smoke(first, source_type)
        _run_smoke(second, source_type)
        runs[source_type] = (first, second)
    return runs


def _point(run_dir: Path, k: int) -> dict:
    return json.loads(
        (run_dir / f"point_{k:02d}.json").read_text(encoding="utf-8")
    )


def _pre_registration(run_dir: Path) -> dict:
    return json.loads(
        (run_dir / "pre_registration.json").read_text(encoding="utf-8")
    )


def _assert_all_finite(payload: dict) -> None:
    """Every float in the payload is finite (no NaN, no infinity)."""
    stack = [payload]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, float):
            assert math.isfinite(item), f"non-finite diagnostic: {item!r}"


def _expected_open_set(source_type: str, k: int) -> list[int]:
    """The pre-registered open set at point *k*, computed independently
    from the committed catalog with the card's truncated-spectral-window
    rule: the records with the source (B, Q, S) whose window
    ``[mass_range[0], min(mass_range[1], M_k)]`` is non-empty."""
    from na49_mixing.hadron_catalog import HadronCatalog

    catalog = HadronCatalog.from_json(CATALOG_PATH)
    qn = SOURCE_QN[source_type]
    mass = GRID_START[source_type] + 0.1 * k
    out: list[int] = []
    for record in catalog.records():
        if (int(record.B), int(record.Q), int(record.S)) != qn:
            continue
        lo = float(record.mass_range[0])
        hi = float(record.mass_range[1])
        if min(hi, mass) >= lo:
            out.append(int(record.pdg_id))
    return out


def _species_key(name: str) -> str:
    """The driver's species-key transliteration (the card's schema
    example: pi+ -> pi_plus, pi- -> pi_minus, pi0 -> pi_0)."""
    if name.endswith("+"):
        return name[:-1] + "_plus"
    if name.endswith("-"):
        return name[:-1] + "_minus"
    if name.endswith("0"):
        return name[:-1] + "_0"
    return name


def test_smoke_determinism(smoke_runs):
    """Two smoke runs with the same arguments produce byte-identical
    point_00.json / point_01.json."""
    for source_type in SOURCE_TYPES:
        first, second = smoke_runs[source_type]
        for k in range(SMOKE_POINTS):
            name = f"point_{k:02d}.json"
            a = (first / name).read_bytes()
            b = (second / name).read_bytes()
            assert a == b, (
                f"{source_type} {name} differs between two identical "
                "smoke runs"
            )


def test_charge_balance(smoke_runs):
    """Every point's charge_balance block shows the exact source (B, Q)
    over all events."""
    for source_type in SOURCE_TYPES:
        run_dir, _ = smoke_runs[source_type]
        baryon, charge = SOURCE_QN[source_type][0], SOURCE_QN[source_type][1]
        for k in range(SMOKE_POINTS):
            payload = _point(run_dir, k)
            assert payload["charge_balance"] == {
                "total_Q_exact": charge,
                "total_B_exact": baryon,
                "events_checked": payload["n_decays"],
            }
            assert payload["n_decays"] == SMOKE_DECAYS
            replica = json.loads(
                (run_dir / f"replica_0_point_{k:02d}.json").read_text(
                    encoding="utf-8"
                )
            )
            assert replica["charge_balance"]["total_Q_exact"] == charge
            assert replica["charge_balance"]["total_B_exact"] == baryon
            assert replica["n_decays"] == SMOKE_DECAYS


def test_open_set(smoke_runs):
    """The smoke points' open_records match the pre-registered open sets
    for those points (recomputed from the committed catalog) and the
    driver's own pre_registration.json entries."""
    for source_type in SOURCE_TYPES:
        run_dir, _ = smoke_runs[source_type]
        pre = _pre_registration(run_dir)
        assert pre["source_type"] == source_type
        for k in range(SMOKE_POINTS):
            expected = _expected_open_set(source_type, k)
            assert expected, "the recomputed open set must be non-empty"
            assert EXPECTED_OPEN_SETS[source_type][k] == expected
            payload = _point(run_dir, k)
            assert payload["open_records"] == expected
            assert pre["open_sets"][str(k)] == expected


def test_degenerate_window(smoke_runs):
    """Point 0 (the grid start) uses the degenerate-window rule (mass
    fixed at M_k): the open-set identity species carries the grid mass
    at rest, and the point-0 diagnostics are finite and non-degenerate
    in the multiplicity means."""
    from na49_mixing.hadron_catalog import HadronCatalog

    identity_pdg = {"baryonic": 2212, "mesonic": 111}
    catalog = HadronCatalog.from_json(CATALOG_PATH)
    for source_type in SOURCE_TYPES:
        run_dir, _ = smoke_runs[source_type]
        payload = _point(run_dir, 0)
        mass = float(payload["source_mass_gev"])
        assert mass == pytest.approx(GRID_START[source_type])

        # every diagnostic float is finite
        _assert_all_finite(payload)

        # non-degenerate multiplicity means: the species means are not
        # all identical at the grid start
        means = [float(v["mean"]) for v in payload["multiplicities"].values()]
        assert all(math.isfinite(value) for value in means)
        assert len(set(means)) > 1, (
            f"{source_type} point-0 multiplicity means are degenerate"
        )

        # the degenerate-window rule: the only open record at the grid
        # start is the stable identity; its events carry the grid mass
        # at rest (mean multiplicity 1, mean_e = M_0, mean_p = 0)
        assert payload["open_records"] == [identity_pdg[source_type]]
        record = catalog.get(identity_pdg[source_type])
        key = _species_key(str(record.name))
        block = payload["multiplicities"][key]
        assert block["mean"] == pytest.approx(1.0)
        assert block["se"] == pytest.approx(0.0)
        moments = payload["momentum_moments"][key]
        assert moments["mean_e_gev"] == pytest.approx(mass)
        assert moments["mean_p_gev"] == pytest.approx(0.0)
        # the identity events are direct source products, never a
        # resonance feed-down
        feed_down = payload["feed_down"][key]
        assert feed_down["fraction_resonant"] == pytest.approx(0.0)
        assert feed_down["by_parent"] == {}


def test_layout_sync_inputs():
    """The nested and outer copies of the driver satisfy the
    check_helpers_layout_sync.py contract: the copies are byte-identical
    (the strongest form of the layout-only-differences allowance), and
    the checker's own normalization keeps them equal."""
    assert NESTED_DRIVER.is_file(), NESTED_DRIVER
    assert OUTER_DRIVER.is_file(), OUTER_DRIVER
    nested = NESTED_DRIVER.read_text(encoding="utf-8")
    outer = OUTER_DRIVER.read_text(encoding="utf-8")
    assert nested == outer, (
        "the nested and outer staged_source_grid.py copies differ "
        "beyond the layout-only allowance"
    )
    spec = importlib.util.spec_from_file_location(
        "check_helpers_layout_sync_for_card9", str(SYNC_CHECKER)
    )
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    assert checker._normalize(nested, side="mix") == checker._normalize(
        outer, side="mono"
    )
    # structural presence: the file pair is not reported as drift
    report = checker.main([
        "--monorepo-helpers", str(OUTER_DRIVER.parent),
        "--mix-helpers", str(NESTED_DRIVER.parent),
    ])
    assert report == 0
