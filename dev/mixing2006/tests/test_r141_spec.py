"""The R141 spec: the R140 contract with the aligned continuing baryonic steps (a fixed kappa and scope) and the
variant fits from the main optimum alone, generated at the R140 optimum (2026-11-05; blk140, bln140)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

NESTED_ROOT = Path(__file__).resolve().parents[1]
RUNS = NESTED_ROOT / "config" / "runs"


@pytest.fixture(scope="module")
def engine():
    spec = importlib.util.spec_from_file_location("run_engine_r141_spec_test", NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py")
    e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e)
    e.configure(RUNS / "r141.json")
    return e


def _specs():
    return tuple(json.loads((RUNS / f"{r}.json").read_text()) for r in ("r140", "r141"))


def test_the_r141_spec_is_r140_with_the_aligned_steps():
    r140, r141 = _specs()
    strip = lambda p: {k: v for k, v in p.items() if k != "gen0"}  # noqa: E731
    assert [strip(p) for p in r141["parameters"]] == [strip(p) for p in r140["parameters"]]
    added = {k: v for k, v in r141["kernel"].items() if k not in r140["kernel"]}
    assert set(added) <= {"baryon_line_kappa", "baryon_line_kappa_scope"} and 0.0 < added["baryon_line_kappa"] <= 50.0
    assert {k: v for k, v in r141["kernel"].items() if k in r140["kernel"]} == r140["kernel"]
    assert r141["fit"] == {**r140["fit"], "n_starts_variant": 1}
    for block in ("topology", "diffractive_split", "isovector", "fixed", "statistics", "central_component",
                  "single_diffraction_window", "single_diffraction_window_counts", "systematic_variants"):
        assert r141.get(block) == r140.get(block), block
    assert r141["generation_0"]["run"] == "r140_symmetric_remnants_2026-11-05"
    assert "r_bbar" in r141["generation_0"]["from_fit"] and "r_bbar" not in r141["generation_0"].get("judgment", {})
    assert r141["seeds"]["raw_first"] > r140["seeds"]["raw_first"] + 511
    path = NESTED_ROOT / ".." / "00runs" / "r140_symmetric_remnants_2026-11-05" / "branch_report_f0.json"
    if not path.is_file():
        return  # the gateway suite tree carries no run archive
    report = json.loads(path.read_text())
    for p in r141["parameters"]:
        if p["name"] in r141["generation_0"]["from_fit"]:
            assert p["gen0"] == report["fitted"][p["name"]], p["name"]


def test_the_engine_carries_the_options(engine):
    e = engine
    assert e.N_FREE_PARAMETERS == 14 and e.KERNEL.baryon_line_kappa is not None
    assert e.N_STARTS_VARIANT == 1
