"""The low-mass non-diffractive lines through the N* and Delta records
(2026-10-12): ``TopologyNominalParams.nd_duality_mass`` maps a vacuum
non-diffractive line below Md onto the low-mass records at its own mass,
and ``nd_duality_shape_mass`` redraws the mass below Ms from the records'
line shapes; off, the committed events."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.kernel_config import KernelConfig, load_kernel
from na49_mixing.nd_duality import DualityRecord, duality_pick, duality_records, duality_redraw_mass
from na49_mixing.topology_mass_laws import _excited_remnant_terms
from na49_mixing.topology_remnants import REMNANT_KIND_INDEX, KIND_HIGH_MASS_BARYONIC_CONTINUUM
from na49_mixing.topology_runner import TopologyNominalParams, run_topology

NESTED_ROOT = Path(__file__).resolve().parents[1]
R93 = json.loads((NESTED_ROOT / "config" / "runs" / "r93.json").read_text())
LABELS = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")
EPS_ND = 0.2334547747805388
BASE = TopologyNominalParams(
    epsilon=0.554, b_sd=6.767, epsilon_nd=EPS_ND, nd_central=False, exact_transfer_range=True, line_thresholds=True,
    transfer_slope_mass_exponent=1.25, low_mass_spin_isospin=True,
    remnant_weights={"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.049, "high_mass_baryonic_continuum": 0.951})
ON = dataclasses.replace(BASE, nd_duality_mass=2.0, nd_duality_shape_mass=1.4)
LO = 0.93827231 + 0.13957039  # the N pi threshold of a (1, 1, 0) line


@pytest.fixture(scope="module")
def kernel():
    return load_kernel(KernelConfig(**R93["kernel"]), NESTED_ROOT)


def _event(params, kernel, seed):
    catalog, fams = kernel
    rngs = {f"rng_{lab}": np.random.default_rng([seed, 11, k]) for k, lab in enumerate(LABELS)}
    return run_topology("nd", params, catalog, fams, **rngs)


def _law_cdf(eps, lo, m):
    return (lo ** (-2 * eps) - m ** (-2 * eps)) / eps


def test_off_is_the_committed_event(kernel):
    """Without the option, and with the shape mass alone at None, the
    event is the committed one; the record keeps no duality edges."""
    for seed in range(12):
        ev = _event(BASE, kernel, seed)
        assert ev.duality_edges is None
        mapped = _event(dataclasses.replace(BASE, nd_duality_mass=None, nd_duality_shape_mass=None), kernel, seed)
        assert mapped.final_state == ev.final_state and mapped.remnants == ev.remnants


def test_the_records(kernel):
    """The (1, 1, 0) records with a width, the Delta(1232) first among the
    I = 3/2 states, each with weight 2J + 1."""
    catalog, _fams = kernel
    recs = duality_records(catalog)
    pdgs = [r.pdg_id for r in recs]
    assert 2214 in pdgs and 2212 not in pdgs
    delta = next(r for r in recs if r.pdg_id == 2214)
    assert delta.weight == 4.0 and abs(delta.pole - 1.232) < 0.01
    assert all(r.width > 0.0 and r.lo < r.pole < r.hi for r in recs)


def test_the_redraw_is_measure_preserving(kernel):
    """Continuum masses at the quantiles of the generation law below Ms map
    onto the records' Breit-Wigner mixture: the redrawn masses' CDF is the
    mixture's at every probe point; masses at or above Ms keep theirs."""
    catalog, _fams = kernel
    recs = duality_records(catalog)
    ms, n = 1.4, 20000
    total = _law_cdf(EPS_ND, LO, ms)
    # the inverse CDF of (M^2)^(-1-eps) on [LO^2, ms^2] at the midpoints of n cells
    v = (np.arange(n) + 0.5) / n
    m = (LO ** (-2 * EPS_ND) - v * total * EPS_ND) ** (-0.5 / EPS_ND)
    redrawn = np.array([duality_redraw_mass(x, LO, ms, EPS_ND, recs) for x in m])
    assert np.all((redrawn > LO) & (redrawn < ms))

    def mixture_cdf(x):
        num = den = 0.0
        for r in recs:
            a, b = max(LO, r.lo), min(ms, r.hi)
            if not b > a:
                continue
            g = r.pole * r.width
            ta, tb = math.atan((a * a - r.pole ** 2) / g), math.atan((b * b - r.pole ** 2) / g)
            tx = math.atan((min(max(x, a), b) ** 2 - r.pole ** 2) / g)
            num += r.weight * (tx - ta) / r.pole
            den += r.weight * (tb - ta) / r.pole
        return num / den
    for x in (1.15, 1.2, 1.232, 1.26, 1.3, 1.35):
        assert abs(np.mean(redrawn < x) - mixture_cdf(x)) < 2e-3, x
    assert duality_redraw_mass(1.41, LO, ms, EPS_ND, recs) is None
    assert duality_redraw_mass(1.2, 1.25, ms, EPS_ND, recs) is None


def test_the_pick():
    """The pick weighs each record that holds the mass by (2J + 1) times
    its Breit-Wigner, and returns None outside every mass range."""
    recs = (DualityRecord(1, 1.232, 0.117, 4.0, 1.08, 1.6), DualityRecord(2, 1.52, 0.11, 4.0, 1.2, 1.9))
    u = np.linspace(0.0005, 0.9995, 2000)
    at_delta = np.mean([duality_pick(1.232, recs, x) == 1 for x in u])
    w1 = 4 * 0.117 / (1.232 * 0.117) ** 2
    w2 = 4 * 0.11 / ((1.232 ** 2 - 1.52 ** 2) ** 2 + (1.52 * 0.11) ** 2)
    assert abs(at_delta - w1 / (w1 + w2)) < 1e-3
    assert {duality_pick(1.1, recs, x) for x in u} == {1}
    assert duality_pick(1.95, recs, 0.5) is None


def test_the_reweighting_is_the_ratio_of_the_shares():
    """A redrawn line enters the exponent reweighting with the ratio of the
    continuum's shares below its shape edge: over a sample drawn at eps0
    the weights average to one at every eps, and the weighted share below
    the edge is the share at eps."""
    rng = np.random.default_rng(5)
    n, hi, ms = 400000, 12.0, 1.4
    u = rng.random(n)
    total = _law_cdf(EPS_ND, LO, hi)
    m = (LO ** (-2 * EPS_ND) - u * total * EPS_ND) ** (-0.5 / EPS_ND)
    below = m < ms
    excited = float(REMNANT_KIND_INDEX[KIND_HIGH_MASS_BARYONIC_CONTINUUM])
    arrays = {"k1": np.full(n, excited), "k2": np.zeros(n), "lo1": np.full(n, LO), "hi1": np.full(n, hi),
              "lo2": np.full(n, np.nan), "hi2": np.full(n, np.nan),
              "ms1": np.where(below, ms, np.nan), "ms2": np.full(n, np.nan)}
    # the redrawn masses (any value below ms: the term must not depend on it)
    m_rec = np.where(below, 1.232, m)
    for eps in (EPS_ND - 0.03, EPS_ND + 0.03):
        w = np.exp(_excited_remnant_terms(np.zeros(n), arrays, m_rec, np.ones(n), eps, EPS_ND))
        assert abs(np.mean(w) - 1.0) < 4e-3
        share = _law_cdf(eps, LO, ms) / _law_cdf(eps, LO, hi)
        assert abs(np.mean(w * below) - share) < 2e-3
        assert np.ptp(w[below]) < 1e-12  # one weight for every redrawn line


def test_events_under_the_option(kernel):
    """Under the option the events conserve charge and baryon number, a
    redrawn line sits below its shape edge and the edge is recorded, and a
    line at or above Md keeps the continuum cascade."""
    seen = 0
    for seed in range(60):
        ev = _event(ON, kernel, seed)
        assert sum(int(h.qn[1]) for h in ev.final_state) == 2 and sum(int(h.qn[0]) for h in ev.final_state) == 2
        assert ev.duality_edges is not None
        for remnant, edge in zip(ev.remnants, ev.duality_edges):
            if math.isfinite(edge):
                seen += 1
                assert remnant.pdg_id is None and remnant.mass < edge <= 1.4
    assert seen > 0


def test_the_refusals(kernel):
    catalog, fams = kernel
    rngs = {f"rng_{lab}": np.random.default_rng([0, 11, k]) for k, lab in enumerate(LABELS)}
    for bad in (dict(nd_duality_mass=1.5, nd_duality_shape_mass=1.6), dict(nd_duality_mass=None, nd_duality_shape_mass=1.3),
                dict(nd_duality_mass=1.0), dict(nd_duality_mass=float("nan"))):
        with pytest.raises(ValueError, match="nd_duality"):
            run_topology("nd", dataclasses.replace(BASE, **bad), catalog, fams, **rngs)


def test_the_run_spec_option(tmp_path):
    path = NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py"
    spec = importlib.util.spec_from_file_location("run_engine_nd_duality", path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    good = dict(R93, topology={**R93["topology"], "nd_duality_mass": 2.0, "nd_duality_shape_mass": 1.4})
    (tmp_path / "good.json").write_text(json.dumps(good))
    engine.configure(tmp_path / "good.json")
    assert engine.TOPOLOGY_OPTIONS["nd_duality_mass"] == 2.0 and engine.TOPOLOGY_OPTIONS["nd_duality_shape_mass"] == 1.4
    params = engine._params_at("nd0", (0.554, 6.767, EPS_ND))
    assert params.nd_duality_mass == 2.0 and params.nd_duality_shape_mass == 1.4


def test_the_r94_spec_is_r93_with_the_duality_at_the_r93_optimum():
    runs = NESTED_ROOT / "config" / "runs"
    r93, r94 = (json.loads((runs / f"{r}.json").read_text()) for r in ("r93", "r94"))
    assert r94["kernel"] == r93["kernel"]
    assert r94["topology"] == {**r93["topology"], "nd_duality_mass": 2.0, "nd_duality_shape_mass": 1.4}
    for block in ("fit", "diffractive_split", "isovector"):
        assert r94[block] == r93[block]
    assert r94["generation_0"]["run"] == "r93_nucleon_kstar_2026-10-12" and r94["generation_0"]["fit"] == "free"
    assert [p["name"] for p in r94["parameters"]] == [p["name"] for p in r93["parameters"]]
    assert r94["seeds"]["master"] == r93["seeds"]["master"]
    assert r94["seeds"]["raw_first"] > r93["seeds"]["raw_first"] + 511
    assert r94["seeds"]["partition_base"] != r93["seeds"]["partition_base"]
    # the generation-0 values are the R93 free optimum of its final branch report (with the sealed archive)
    branch = NESTED_ROOT.parent / "00runs" / "r93_nucleon_kstar_2026-10-12" / "branch_report_f0.json"
    if branch.is_file():
        fitted = json.loads(branch.read_text())["fitted"]
        gen0 = {p["name"]: p["gen0"] for p in r94["parameters"]}
        for name in r94["generation_0"]["from_fit"]:
            value = r94["generation_0"]["extra"][name] if name == "w_low_mass" else gen0[name]
            assert float(value) == float(fitted[name]), name
