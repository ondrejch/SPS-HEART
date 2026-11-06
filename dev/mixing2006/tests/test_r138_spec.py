"""The R138 spec: the R137 contract with the counted decays, the octet baryon pairs of the neutral mesonic family and
the Omega- record (2026-11-04), at the R137 optimum; the omega Dalitz fix is in the code."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

NESTED_ROOT = Path(__file__).resolve().parents[1]
RUNS = NESTED_ROOT / "config" / "runs"


@pytest.fixture(scope="module")
def engine():
    spec = importlib.util.spec_from_file_location("run_engine_r138_spec_test", NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py")
    e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e)
    e.configure(RUNS / "r138.json")
    return e


def test_the_r138_spec_is_r137_with_the_day_s_pieces():
    r137, r138 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r137", "r138"))
    strip = lambda p: {k: v for k, v in p.items() if k != "gen0"}  # noqa: E731
    assert [strip(p) for p in r138["parameters"]] == [strip(p) for p in r137["parameters"]]
    added = {"baryon_pair_completion": "strange_neutral", "baryon_pair_completion_members": "octet", "omega_baryon": True}
    assert {k: v for k, v in r138["kernel"].items() if k not in added} == r137["kernel"]
    assert {k: r138["kernel"][k] for k in added} == added
    assert r138["fit"] == {**r137["fit"], "counted_decays": True}
    for block in ("topology", "diffractive_split", "isovector", "fixed", "statistics", "central_component",
                  "single_diffraction_window", "single_diffraction_window_counts", "systematic_variants"):
        assert r138.get(block) == r137.get(block), block
    assert r138["generation_0"]["run"] == "r137_fourteen_parameters_2026-11-03"
    assert r138["seeds"]["raw_first"] > r137["seeds"]["raw_first"] + 511
    path = NESTED_ROOT / ".." / "00runs" / "r137_fourteen_parameters_2026-11-03" / "branch_report_f0.json"
    if not path.is_file():
        return  # the gateway suite tree carries no run archive
    report = json.loads(path.read_text())
    for p in r138["parameters"]:
        if p["name"] in r138["generation_0"]["from_fit"]:
            assert p["gen0"] == report["fitted"][p["name"]], p["name"]


def test_the_engine_carries_the_options(engine):
    e = engine
    assert e.N_FREE_PARAMETERS == 14 and e.COUNTED_DECAYS is True
    assert e.KERNEL.baryon_pair_completion == "strange_neutral" and e.KERNEL.omega_baryon is True
