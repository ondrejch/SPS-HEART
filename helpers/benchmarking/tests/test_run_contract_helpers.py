"""The run-contract helpers added after the external reviews of 2026-10-02:
the atomic JSON write (q1 M7), the loaded-module freeze list (q1 H1), and
the box-bounded Hessian (q1 M3)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
import run_contract as RC  # noqa: E402


def test_write_json_is_atomic_and_byte_identical(tmp_path):
    payload = {"b": [1, 2.5, None], "a": {"x": "y"}}
    path = tmp_path / "record.json"
    RC.write_json(path, payload)
    assert path.read_text(encoding="utf-8") == json.dumps(payload, indent=1, sort_keys=True) + "\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["record.json"]  # no temporary left behind
    RC.write_json(path, {"a": 1})
    assert RC.read_json(path) == {"a": 1}


def test_loaded_module_rels_lists_project_modules():
    mono_root = HERE.parents[3] if (HERE.parents[3] / "dev" / "mixing2006").is_dir() else None
    if mono_root is None:
        pytest.skip("monorepo layout not present")
    rels = RC.loaded_module_rels(mono_root)
    assert "helpers/benchmarking/run_contract.py" in rels
    assert all(r.startswith(RC.PROJECT_SOURCE_ROOTS) and r.endswith(".py") for r in rels)
    assert rels == sorted(set(rels))


def _quadratic(H, a):
    H = np.asarray(H, dtype=float); a = np.asarray(a, dtype=float)
    return lambda p: float((np.asarray(p) - a) @ H @ (np.asarray(p) - a))


def test_bounded_hessian_interior_is_central():
    H = np.array([[4.0, 1.0], [1.0, 2.0]])
    rec = RC.bounded_hessian(_quadratic(H, [0.3, 0.4]), [0.3, 0.4], [0.01, 0.01], [-1, -1], [1, 1], ["x", "y"])
    assert rec["directions"] == ["central", "central"] and rec["bound_limited"] == []
    assert np.allclose(rec["matrix"], 2 * H, rtol=1e-6) and rec["positive_definite"]


@pytest.mark.parametrize("p0, expected", [([0.0, 0.4], ["forward", "central"]), ([0.3, 1.0], ["central", "backward"]),
                                          ([0.0, 1.0], ["forward", "backward"])])
def test_bounded_hessian_at_a_bound_is_one_sided(p0, expected):
    """A quadratic is recovered exactly by the one-sided differences, and
    every probe stays inside the box (the value function raises outside)."""
    H = np.array([[4.0, 1.0], [1.0, 2.0]])
    quad = _quadratic(H, [0.0, 1.0])
    lo, hi = np.array([0.0, -1.0]), np.array([1.0, 1.0])

    def boxed(p):
        if np.any(np.asarray(p) < lo - 1e-12) or np.any(np.asarray(p) > hi + 1e-12):
            raise AssertionError(f"probe {p} left the box")
        return quad(p)

    rec = RC.bounded_hessian(boxed, p0, [0.01, 0.01], lo, hi, ["x", "y"])
    assert rec["directions"] == expected
    assert set(rec["bound_limited"]) == {n for n, d in zip(["x", "y"], expected) if d != "central"}
    assert np.allclose(rec["matrix"], 2 * H, rtol=1e-6) and rec["positive_definite"]
    assert all(math.isfinite(v) for v in rec["errors"].values())


def test_bounded_hessian_reports_non_finite():
    rec = RC.bounded_hessian(lambda p: float("inf") if p[0] > 0.2 else float(p @ p), [0.2, 0.0], [0.01, 0.01],
                             [-1, -1], [1, 1], ["x", "y"])
    assert rec["non_finite"] and not rec["positive_definite"] and rec["errors"] is None
