"""The smooth upper edge of the leading-proton tail (2026-10-25, at the R116
optimum): the tail's law dsigma/dxi ~ xi^a peaks at its upper end, and the wall
at xi_hi = 0.5 put a step into the model's soft protons at x_F = 0.5 (0.287 ->
0.457 at p_T 0.1 GeV/c between x_F 0.45 and 0.50, against the data's 0.315 ->
0.367).  The topology's sd_tail [xi_lo, xi_hi, a, b, xi_c, w] multiplies the law
by 1 / (1 + exp((xi - xi_c) / w)) on a window reaching well above xi_c.

- the draw: exactly 2 uniforms, the density with its edge, and the wall as the
  limit of a narrow edge;
- generate_sd: the tail events follow the edge, the other events are the same
  bit for bit, malformed edges are refused;
- the engine reads a 6-value sd_tail and refuses malformed ones.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.rng_substreams import derive_substreams
from na49_mixing.spectral_density import families_from_json
from na49_mixing.topology_mass_laws import MassLawParams, draw_marginal, draw_marginal_edge
from na49_mixing.topology_single_diffraction import (NOMINAL_PARAMS as SD_PARAMS, NOMINAL_SD_REMNANT_WEIGHTS, SQRT_S,
                                                     generate_sd)

NESTED_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
CONTINUUM_PATH = NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json"
ENGINE = NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py"
RUNS = NESTED_ROOT / "config" / "runs"
S = SQRT_S * SQRT_S
TAIL_LAW = MassLawParams(epsilon=-2.0, b_sd=7.0)   # dsigma/dxi ~ xi (a = 1)


@pytest.fixture(scope="module")
def catalog():
    return HadronCatalog.from_json(CATALOG_PATH)


@pytest.fixture(scope="module")
def fams():
    return families_from_json(str(CONTINUUM_PATH))


def _rngs(i, labels=("spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")):
    subs = derive_substreams(20261025, 0, i)
    return tuple(np.random.default_rng(subs.stream_for(label)) for label in labels)


def _share_above(x, lo, hi, xi_c, w):
    xs = np.linspace(lo, hi, 200001)
    f = xs / (1.0 + np.exp((xs - xi_c) / w))
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(xs))])
    return 1.0 - np.interp(x, xs, cum) / cum[-1]


def test_the_edge_draw_takes_two_uniforms():
    a, b = np.random.default_rng(7), np.random.default_rng(7)
    draw_marginal_edge(0.1 * S, 0.89 * S, TAIL_LAW, S, 0.5, 0.05, a)
    b.random(), b.random()
    assert a.random() == b.random()


def test_the_edge_draw_follows_its_density():
    rng = np.random.default_rng(11)
    n = 6000
    xis = np.array([draw_marginal_edge(0.1 * S, 0.89 * S, TAIL_LAW, S, 0.5, 0.05, rng) / S for _ in range(n)])
    assert 0.1 <= xis.min() and xis.max() <= 0.89
    for x in (0.3, 0.45, 0.5, 0.55, 0.6):
        p = _share_above(x, 0.1, 0.89, 0.5, 0.05)
        assert abs(np.mean(xis > x) - p) < 4.0 * math.sqrt(p * (1.0 - p) / n) + 1e-3, x
    assert 0.12 < np.mean(xis > 0.5) < 0.19   # about 0.155 of the tail beyond the old wall


def test_a_narrow_edge_is_the_wall():
    rng_e, rng_w = np.random.default_rng(3), np.random.default_rng(4)
    n = 4000
    edge = np.array([draw_marginal_edge(0.1 * S, 0.89 * S, TAIL_LAW, S, 0.5, 1e-4, rng_e) / S for _ in range(n)])
    wall = np.array([draw_marginal(0.1 * S, 0.5 * S, TAIL_LAW, 0.0, rng_w) / S for _ in range(n)])
    assert edge.max() < 0.5 + 2e-3
    mean = (2.0 / 3.0) * (0.5 ** 3 - 0.1 ** 3) / (0.5 ** 2 - 0.1 ** 2)   # 0.344
    for xs in (edge, wall):
        assert abs(xs.mean() - mean) < 4.0 * xs.std() / math.sqrt(n)


def test_the_edge_draw_refuses_a_bad_edge():
    with pytest.raises(ValueError):
        draw_marginal_edge(0.1 * S, 0.89 * S, TAIL_LAW, S, 0.5, 0.0, np.random.default_rng(1))
    with pytest.raises(ValueError):
        draw_marginal_edge(0.1 * S, 0.89 * S, TAIL_LAW, 0.0, 0.5, 0.05, np.random.default_rng(1))


def test_generate_sd_draws_the_tail_with_its_edge(catalog, fams):
    wall = (0.3, 0.1, 0.89, 1.0, 7.0)
    edge = wall + (0.5, 0.05)
    tails_edge, tails_wall = [], []
    for i in range(120):
        e = generate_sd("target", SD_PARAMS, NOMINAL_SD_REMNANT_WEIGHTS, catalog, fams, *_rngs(i), exact_range=True,
                        xi_max=0.1, tail_component=edge)
        w = generate_sd("target", SD_PARAMS, NOMINAL_SD_REMNANT_WEIGHTS, catalog, fams, *_rngs(i), exact_range=True,
                        xi_max=0.1, tail_component=wall)
        assert bool(e.tail) == bool(w.tail)   # the share's uniform comes first, the same in both
        if e.tail:
            tails_edge.append(e.excitation_mass ** 2 / S)
            tails_wall.append(w.excitation_mass ** 2 / S)
        else:
            # every other event is the same bit for bit: the edge's draw takes the wall's 2 uniforms
            assert e.excitation_mass == w.excitation_mass and e.t == w.t
    assert len(tails_edge) >= 20
    assert all(0.1 <= x <= 0.89 for x in tails_edge)
    assert np.mean(tails_edge) < np.mean(tails_wall)   # the edge cuts the law's rise toward 0.89
    for bad in (wall + (0.05, 0.05), wall + (0.95, 0.05), wall + (0.5, 0.0), wall + (0.5,)):
        with pytest.raises(ValueError):
            for i in range(40):   # the check fires on the first tail event
                generate_sd("target", SD_PARAMS, NOMINAL_SD_REMNANT_WEIGHTS, catalog, fams, *_rngs(i), exact_range=True,
                            xi_max=0.1, tail_component=bad)


def _engine(name):
    spec = importlib.util.spec_from_file_location(name, ENGINE)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    return engine


def test_the_engine_reads_the_edge(tmp_path):
    E = _engine("run_engine_tail_edge_test")
    spec = json.loads((RUNS / "r116.json").read_text())
    assert spec["topology"]["sd_tail"] == [0.1, 0.5, 1.0, 7.0]
    path = tmp_path / "r116_edge.json"
    spec["topology"]["sd_tail"] = [0.1, 0.89, 1.0, 7.0, 0.5, 0.05]
    path.write_text(json.dumps(spec))
    E.configure(path)
    assert E.SD_TAIL_LAW == (0.1, 0.89, 1.0, 7.0, 0.5, 0.05)
    for bad in ([0.1, 0.89, 1.0, 7.0, 0.95, 0.05], [0.1, 0.89, 1.0, 7.0, 0.5, 0.0], [0.1, 0.89, 1.0, 7.0, 0.5]):
        spec["topology"]["sd_tail"] = bad
        path.write_text(json.dumps(spec))
        with pytest.raises(ValueError):
            E.configure(path)


def test_the_r118_spec_is_r117_with_the_tail_edge():
    r117, r118 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r117", "r118"))
    assert r117["topology"]["sd_tail"] == [0.1, 0.5, 1.0, 7.0]
    law = r118["topology"]["sd_tail"]
    assert law[:4] == [0.1, 0.89, 1.0, 7.0] and len(law) == 6 and 0.1 < law[4] < 0.89 and law[5] > 0.0
    assert {k: v for k, v in r118["topology"].items() if k != "sd_tail"} == \
        {k: v for k, v in r117["topology"].items() if k != "sd_tail"}
    for block in ("kernel", "fit", "diffractive_split", "isovector", "fixed", "statistics", "central_component",
                  "single_diffraction_window", "systematic_variants", "flavor_bbar_scope"):
        assert r118[block] == r117[block], block
    assert [p["name"] for p in r118["parameters"]] == [p["name"] for p in r117["parameters"]]
    assert r118["generation_0"]["run"] == "r117_bbar_scope_replica_2026-10-25"
    assert r118["generation_0"]["from_fit"] == r117["generation_0"]["from_fit"]
    assert r118["seeds"]["raw_first"] > r117["seeds"]["raw_first"] + 511
    branch = NESTED_ROOT.parent / "00runs" / "r117_bbar_scope_replica_2026-10-25" / "branch_report_f0.json"
    if branch.is_file():
        report = json.loads(branch.read_text())
        assert r118["generation_0"]["branch_digest"] == report["digest"]
        gen0 = {p["name"]: p["gen0"] for p in r118["parameters"]}
        for name in r118["generation_0"]["from_fit"]:
            assert float(gen0[name]) == float(report["fitted"][name]), name


def test_the_r119_spec_is_the_r118_replica():
    r118, r119 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r118", "r119"))
    assert r119["topology"]["sd_tail"] == [0.1, 0.89, 1.0, 7.0, 0.5, 0.1]
    for block in ("kernel", "topology", "fit", "diffractive_split", "isovector", "fixed", "statistics", "central_component",
                  "single_diffraction_window", "systematic_variants", "flavor_bbar_scope"):
        assert r119[block] == r118[block], block
    assert [p["name"] for p in r119["parameters"]] == [p["name"] for p in r118["parameters"]]
    assert r119["generation_0"]["run"] == "r118_tail_edge_2026-10-25"
    assert r119["generation_0"]["from_fit"] == r118["generation_0"]["from_fit"]
    assert "judgment" not in r119["generation_0"]
    assert r119["seeds"]["raw_first"] > r118["seeds"]["raw_first"] + 511
    branch = NESTED_ROOT.parent / "00runs" / "r118_tail_edge_2026-10-25" / "branch_report_f0.json"
    if branch.is_file():
        report = json.loads(branch.read_text())
        assert r119["generation_0"]["branch_digest"] == report["digest"]
        gen0 = {p["name"]: p["gen0"] for p in r119["parameters"]}
        for name in r119["generation_0"]["from_fit"]:
            assert float(gen0[name]) == float(report["fitted"][name]), name


def test_the_r120_spec_is_r119_with_epsilon_fixed():
    r119, r120 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r119", "r120"))
    for block in ("kernel", "topology", "fit", "diffractive_split", "isovector", "fixed", "statistics", "central_component",
                  "single_diffraction_window", "systematic_variants", "flavor_bbar_scope"):
        assert r120[block] == r119[block], block
    p119, p120 = ({p["name"]: p for p in r["parameters"]} for r in (r119, r120))
    assert list(p120) == list(p119)
    eps = p120["epsilon"]
    assert eps["free"] is False and eps["value"] == eps["gen0"] == 0.10
    assert {n: p["free"] for n, p in p120.items() if n != "epsilon"} == {n: p["free"] for n, p in p119.items() if n != "epsilon"}
    g0 = r120["generation_0"]
    assert g0["run"] == "r119_tail_edge_replica_2026-10-26"
    assert g0["from_fit"] == [n for n in r119["generation_0"]["from_fit"] if n != "epsilon"]
    assert "epsilon" in g0["judgment"]
    assert r120["seeds"]["raw_first"] > r119["seeds"]["raw_first"] + 511
    E = _engine("run_engine_r120_test")
    E.configure(RUNS / "r120.json")
    assert "epsilon" not in E.SHAPE_NAMES and E.FIXED_VALUES["epsilon"] == 0.10 and E.GEN0_EPSILON == 0.10
    branch = NESTED_ROOT.parent / "00runs" / "r119_tail_edge_replica_2026-10-26" / "branch_report_f0.json"
    if branch.is_file():
        report = json.loads(branch.read_text())
        assert g0["branch_digest"] == report["digest"]
        for name in g0["from_fit"]:
            assert float(p120[name]["gen0"]) == float(report["fitted"][name]), name
