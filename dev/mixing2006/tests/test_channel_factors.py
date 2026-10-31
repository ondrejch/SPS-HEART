"""The channel factors of the diagnostic fit R101 (2026-10-18; owner
2026-10-17: "ok approve 1 2 3", the model decision register): the omega,
rho, eta'(958) and Lambda(1520)/Sigma(1385) channel weights as effective
factors on top of the four flavor weights, their daughter counts as extra
flavor ledger key entries.  Without them every key and weight is the
plan's."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.cascade_decay import CascadeChannel
from na49_mixing.flavor_weights import (CHANNEL_FACTOR_DAUGHTERS, FlavorParams, apply_flavor_weights, exponent_key,
                                        flavor_exponents, flavor_weight, key_weight)
from na49_mixing.kernel_config import KernelConfig, load_kernel

NESTED_ROOT = Path(__file__).resolve().parents[1]
ENGINE = NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py"
RUNS = NESTED_ROOT / "config" / "runs"


def _engine(name):
    spec = importlib.util.spec_from_file_location(name, ENGINE)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    return engine


@pytest.fixture(scope="module")
def kernel():
    r100 = json.loads((RUNS / "r100.json").read_text())
    return load_kernel(KernelConfig(**r100["kernel"]), NESTED_ROOT)


def _channel(parent, *daughters):
    return CascadeChannel(parent_id=parent, daughters=tuple(f"catalog:{d}" for d in daughters), kind="established_hadron",
                          weight=1.0)


def test_without_factors_the_plan_keys_and_weights(kernel):
    cat, fams = kernel
    plain = FlavorParams(gamma_s_had=0.2, r_v=1.1, r_bbar=2.3, assoc_norm=2.5)
    ch = _channel("baryonic_np", 2212, 223)
    e = flavor_exponents((1, 1, 0), ch, cat, fams)
    assert e.n_channel == () and len(exponent_key(e)) == 4 and plain.channel_names == ()
    assert flavor_weight(e, plain) == key_weight(exponent_key(e), plain) == 1.1


def test_the_factors_weight_their_daughters(kernel):
    cat, fams = kernel
    params = FlavorParams(gamma_s_had=0.2, r_v=1.1, r_bbar=2.3, assoc_norm=2.5,
                          channel_factors={"w_rho": 0.9, "w_omega": 1.3, "w_lowy": 1.5, "w_etap": 1.2})
    assert params.channel_names == ("w_omega", "w_rho", "w_etap", "w_lowy")
    cases = {(2212, 223): (1, 0, 0, 0), (223, 223): (2, 0, 0, 0), (113, 223): (1, 1, 0, 0), (-213, 211): (0, 1, 0, 0),
             (331, 111): (0, 0, 1, 0), (102134, 321): (0, 0, 0, 1), (2212, 111): (0, 0, 0, 0)}
    for daughters, counts in cases.items():
        parent = (1, 1, 0) if any(abs(d) > 1000 for d in daughters) else (0, 0, 0)
        e = flavor_exponents(parent, _channel("f", *daughters), cat, fams, params.hidden_ssbar_content, params.channel_names)
        assert e.n_channel == counts, daughters
        key = exponent_key(e)
        assert key[4:] == counts and math.isclose(flavor_weight(e, params), key_weight(key, params), rel_tol=1e-14)
        base = flavor_weight(flavor_exponents(parent, _channel("f", *daughters), cat, fams), FlavorParams(
            gamma_s_had=0.2, r_v=1.1, r_bbar=2.3, assoc_norm=2.5))
        expect = base * 1.3 ** counts[0] * 0.9 ** counts[1] * 1.2 ** counts[2] * 1.5 ** counts[3]
        assert math.isclose(flavor_weight(e, params), expect, rel_tol=1e-14), daughters
    with pytest.raises(ValueError, match="unknown factors"):
        FlavorParams(channel_factors={"w_phi": 1.0})
    with pytest.raises(ValueError, match="channel-factor counts"):
        key_weight((0, 1, 0, 0), params)


def test_the_factors_reweight_a_vertex(kernel):
    """apply_flavor_weights renormalizes a vertex with the factors on; at
    every factor 1 the weights are the plan's."""
    cat, fams = kernel
    chans = [_channel("baryonic_np", 2212, 223), _channel("baryonic_np", 2212, 111), _channel("baryonic_np", 2112, 211)]
    plain = FlavorParams(r_v=1.1)
    unit = FlavorParams(r_v=1.1, channel_factors={n: 1.0 for n in CHANNEL_FACTOR_DAUGHTERS})
    w = [c.weight for c in apply_flavor_weights(chans, (1, 1, 0), cat, fams, plain)]
    assert [c.weight for c in apply_flavor_weights(chans, (1, 1, 0), cat, fams, unit)] == w
    on = [c.weight for c in apply_flavor_weights(chans, (1, 1, 0), cat, fams, FlavorParams(
        r_v=1.1, channel_factors={"w_omega": 2.0}))]
    assert math.isclose(on[0] / on[1], 2.0 * w[0] / w[1], rel_tol=1e-12) and math.isclose(sum(on), 1.0, rel_tol=1e-12)


def test_the_engine_carries_the_factors():
    e = _engine("run_engine_channel_factors")
    e.configure(RUNS / "r101.json")
    assert e.CHANNEL_FACTORS == ("w_omega", "w_rho", "w_etap", "w_lowy")
    assert e.FLAVOR_ORDER == e.BASE_FLAVOR_ORDER + e.CHANNEL_FACTORS
    assert set(e.CHANNEL_FACTORS) <= set(e.FLAVOR_FREE)
    point = (0.2, 1.1, 2.3, 2.5, 1.3, 0.9, 1.2, 1.5)
    params = e.flavor_params(point)
    assert params.channel_factors == {"w_omega": 1.3, "w_rho": 0.9, "w_etap": 1.2, "w_lowy": 1.5} and params.assoc_norm == 2.5
    assert e.GEN_FLAVOR[4:] == (1.0, 1.0, 1.0, 1.0)
    # the ledger on an eight-entry key table, against the vertex-by-vertex expression
    rng = np.random.default_rng(7)
    n_events, n_v = 40, 120
    v_event = np.sort(rng.integers(0, n_events, n_v))
    gkey, gv, gw, chosen = [], [], [], []
    for v in range(n_v):
        keys = {tuple(int(x) for x in rng.integers(0, 3, 8)) for _ in range(rng.integers(2, 5))}
        keys = sorted(keys)
        for k in keys:
            gkey.append(k); gv.append(v); gw.append(float(rng.uniform(0.1, 1.0)))
        chosen.append(keys[int(rng.integers(0, len(keys)))])
    arrays = {"led_event": np.asarray(v_event, dtype=np.int32), "led_gkey": np.asarray(gkey, dtype=np.int16),
              "led_chosen": np.asarray(chosen, dtype=np.int16), "led_gvertex": np.asarray(gv, dtype=np.int32),
              "led_gw": np.asarray(gw, dtype=np.float32)}
    led = e.Templates._ledger(arrays)
    to = np.asarray(point)
    gen = np.asarray(e.GEN_FLAVOR)
    delta = np.log(to) - np.log(gen)
    dk = led["keys"] @ delta
    z_to = np.bincount(led["gv"], weights=led["gw"] * np.exp(dk)[led["g_idx"]], minlength=n_v)
    got = np.bincount(led["v_event"], weights=dk[led["c_idx"]] + led["log_z_from"] - np.log(z_to), minlength=n_events)
    want = np.zeros(n_events)
    p_to, p_gen = e.flavor_params(point), e.flavor_params(e.GEN_FLAVOR)
    for v in range(n_v):
        rows = [i for i in range(len(gv)) if gv[i] == v]
        def kw(k, p):
            k = (k[0] / 2.0,) + tuple(k[1:])
            return key_weight(k, p)
        z_from = sum(float(np.float32(gw[i])) for i in rows)
        z_t = sum(float(np.float32(gw[i])) * kw(gkey[i], p_to) / kw(gkey[i], p_gen) for i in rows)
        want[v_event[v]] += math.log(kw(chosen[v], p_to) / kw(chosen[v], p_gen)) + math.log(z_from) - math.log(z_t)
    assert np.allclose(got, want, rtol=1e-9, atol=1e-9)


def test_a_model_spec_keeps_the_plan_ledger(tmp_path):
    """A spec without flavor_channel_factors keeps FLAVOR_ORDER and the
    four-entry keys, also after a diagnostic spec in the same process; the
    generalized key decoding reproduces the four-entry formula."""
    e = _engine("run_engine_plan_ledger")
    e.configure(RUNS / "r101.json")
    e.configure(RUNS / "r100.json")
    assert e.CHANNEL_FACTORS == () and e.FLAVOR_ORDER == ("gamma_s_had", "r_v", "r_bbar", "assoc_norm")
    rng = np.random.default_rng(3)
    keys = rng.integers(0, 40, (200, 4)).astype(np.int16)
    arrays = {"led_event": np.arange(100, dtype=np.int32), "led_gkey": keys, "led_chosen": keys[::2].copy(),
              "led_gvertex": np.repeat(np.arange(100, dtype=np.int32), 2), "led_gw": np.ones(200, dtype=np.float32)}
    led = e.Templates._ledger(arrays)
    both = np.concatenate([keys, keys[::2]]).astype(np.int64)
    codes = ((both[:, 0] * 100 + both[:, 1]) * 100 + both[:, 2]) * 100 + both[:, 3]
    table = np.unique(codes)
    k = np.column_stack([table // 1_000_000, (table // 10_000) % 100, (table // 100) % 100, table % 100]).astype(float)
    k[:, 0] /= 2.0
    assert np.array_equal(led["keys"], k)
    with pytest.raises(ValueError, match="flavor_channel_factors"):
        bad = json.loads((RUNS / "r100.json").read_text())
        bad["flavor_channel_factors"] = ["w_phi"]
        path = tmp_path / "bad_channel_factors.json"
        path.write_text(json.dumps(bad))
        e.configure(path)


def test_the_r101_spec_is_r100_with_the_factors_free():
    r100, r101 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r100", "r101"))
    assert r101["kernel"] == r100["kernel"] and r101["topology"] == r100["topology"]
    for block in ("fit", "diffractive_split", "isovector", "fixed", "statistics"):
        assert r101[block] == r100[block]
    assert r101["flavor_channel_factors"] == ["w_omega", "w_rho", "w_etap", "w_lowy"] and "flavor_channel_factors" not in r100
    names = [p["name"] for p in r101["parameters"]]
    assert [n for n in names if not n.startswith("w_") or n == "w_excited"] == [p["name"] for p in r100["parameters"]]
    rv = next(p for p in r101["parameters"] if p["name"] == "r_v")
    for n in r101["flavor_channel_factors"]:
        p = next(q for q in r101["parameters"] if q["name"] == n)
        assert p["free"] is True and p["gen0"] == 1.0 and p["transform"] == "log" and p["bounds"] == rv["bounds"]
    g0 = r101["generation_0"]
    assert g0["run"] == "r100_pdg_tables_2026-10-17"
    assert r101["seeds"]["raw_first"] > r100["seeds"]["raw_first"] + 511
    branch = NESTED_ROOT.parent / "00runs" / "r100_pdg_tables_2026-10-17" / "branch_report_f0.json"
    if branch.is_file():
        fitted = json.loads(branch.read_text())["fitted"]
        gen0 = {p["name"]: p["gen0"] for p in r101["parameters"]}
        for name in g0["from_fit"]:
            assert float(gen0[name]) == float(fitted[name]), name
