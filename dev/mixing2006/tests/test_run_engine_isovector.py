"""The run engine's isovector exchange and neutron segment
(TASK-20261004-r8-charge-sensitive-fit, R75): the spec configures the
reweighted isovector fraction, the vacuum-only refit and the neutron
dN/dx_F segment; the exchange reweighting is exact; the seed block is the
ledger reservation; a reduced run carries the isovector test.  The R74
register (test_run_engine.py) holds the engine bit for bit on the R74 spec."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

NESTED_ROOT = Path(__file__).resolve().parents[1]
ENGINE = NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py"
SPEC = NESTED_ROOT / "config" / "runs" / "r75.json"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ENGINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def engine():
    e = _load("run_engine_r75_under_test")
    e.configure(SPEC)
    return e


def test_the_r75_spec(engine):
    assert engine.ISOVECTOR_GEN == 0.55 and engine.NEUTRON_SEGMENT is True
    assert engine.SHAPE_NAMES[-1] == "f_isovector" and engine.N_FREE_PARAMETERS == len(engine.SHAPE_NAMES) + 1
    assert engine.SEGMENT_NAMES == engine.FIT_SPECIES + ("neutron",)
    assert engine.KERNEL.isospin_completion == ("baryonic", "strange_baryonic", "strange_mesonic") and engine.KERNEL.delta_lines
    assert engine.param_names(None, 0.0) == engine.SHAPE_NAMES[:-1] + ("f_central",)
    assert len(engine.bounds_for(None)) == len(engine.SHAPE_NAMES) + 1 and len(engine.bounds_for(None, 0.0)) == len(engine.SHAPE_NAMES)
    # the seed block is the ledger reservation, disjoint from every other run
    ledger = engine.RC.seed_ledger(NESTED_ROOT)
    out = engine.RC.check_seed_disjointness(ledger, engine.LEDGER_RUN, (engine.RAW_FIRST, engine.RAW_LAST), engine.PARTITION_BASE)
    assert out["disjoint"] and (engine.RAW_FIRST, engine.RAW_LAST) == (20320200, 20320711)
    # generation 0 is the sealed R74 null optimum
    spec = json.loads(SPEC.read_text())
    assert spec["generation_0"]["run"] == "r74_central_component_barrier_step_2026-10-03"
    assert all(p in spec["frozen_extra"] for p in ("dev/mixing2006/na49_mixing/isovector_exchange.py",
                                                  "dev/mixing2006/na49_mixing/continuum_isospin.py",
                                                  "dev/mixing2006/na49_mixing/kernel_config.py"))


@pytest.mark.parametrize("edit, message", [("no_block", "go together"), ("logit", "identity transform"),
                                           ("no_delta_lines", "Delta-like lines"), ("fraction", "must lie in")])
def test_the_spec_validates(tmp_path, edit, message):
    spec = json.loads(SPEC.read_text())
    bad = copy.deepcopy(spec)
    if edit == "no_block":
        del bad["isovector"]
    elif edit == "logit":
        next(p for p in bad["parameters"] if p["name"] == "f_isovector")["transform"] = "logit"
    elif edit == "no_delta_lines":
        bad["kernel"]["delta_lines"] = False
    else:
        bad["isovector"]["generation_fraction"] = 1.0
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    e = _load(f"run_engine_bad_{edit}")
    with pytest.raises(ValueError, match=message):
        e.configure(path)


def test_the_exchange_reweighting_is_exact(engine):
    n, n_iv = 1000, 550
    block = {"n": n, "shape": {}, "ledger": None, "iv": np.arange(n) < n_iv}
    t = engine.Templates.__new__(engine.Templates)
    t.point = (0.6, 2.9, 0.2)
    t.flavor_point = engine.GEN_FLAVOR

    def shares(f):
        log_w = t.log_weights("sd_projectile", block, *t.point, iso=f)
        w = engine.Templates.normalized(log_w)
        return float(np.sum(w[:n_iv])), log_w

    share, log_w = shares(0.55)
    assert np.all(log_w == 0.0) and share == pytest.approx(0.55, abs=1e-12)
    for f in (0.0, 0.2, 0.7, 1.0):
        assert shares(f)[0] == pytest.approx(f, abs=1e-12)
    assert engine.ess_fraction(shares(0.0)[1]) == pytest.approx(0.45, abs=1e-12)


def test_the_vacuum_only_coordinates(engine):
    a = engine.anchors()
    phys = engine.physical_from_x(engine.dispersed_starts()[0], None, a)
    assert phys["f_isovector"] == 0.55
    x0 = engine.x_from_physical(phys, None, 0.0)
    back = engine.physical_from_x(x0, None, a, 0.0)
    assert back["f_isovector"] == 0.0 and all(back[n] == pytest.approx(phys[n]) for n in engine.param_names(None, 0.0))
    starts = engine.dispersed_starts(None, 0.0)
    assert len(starts) == engine.N_STARTS_NULL and all(len(s) == len(engine.SHAPE_NAMES) for s in starts)


def test_the_neutron_table(engine):
    data, sigma = engine.neutron_table()
    assert data.size == engine.NEUTRON_XF_EDGES.size - 1 == 8
    assert data[0] == 0.481 and sigma[0] == pytest.approx(0.481 * 0.208)


def _reduced(module, run_dir, monkeypatch, n_events=60):
    monkeypatch.setattr(module, "REPLICAS", 2)
    monkeypatch.setattr(module, "N_SLOTS", len(module.CLASS_ORDER) * 2)
    monkeypatch.setattr(module, "SEEDED_SLOTS", (len(module.CLASS_ORDER) - 1) * 2)
    monkeypatch.setattr(module, "MESONIC_DECAYS", 20)
    monkeypatch.setattr(module, "N_STARTS", 2)
    monkeypatch.setattr(module, "N_STARTS_NULL", 2)
    monkeypatch.setattr(module, "N_STARTS_VARIANT", 2)
    monkeypatch.setattr(module, "FIT_MAXFEV", 300)
    monkeypatch.setattr(module, "FC_SCAN", (0.5,))
    monkeypatch.setattr(module.RC, "run_replicas", lambda worker, jobs, max_workers=None: [worker(*job) for job in jobs])
    contract = module._contract(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for slot in range(module.N_SLOTS):
        res = module.generate_slot(module.slot_class(slot), module.SMOKE_SEED + module.seed_slot(slot), n_events,
                                   module.SMOKE_PARTITION + 10_000 * module.seed_slot(slot), module.GEN0_SHAPE, module.FLOOR_GRID[0],
                                   module.GEN_FLAVOR)
        module.write_slot(contract, 0, 0, slot, res, "test")
    return contract


@pytest.mark.parametrize("spec_name", ["r75.json", "r76.json"])
def test_a_reduced_run_carries_the_isovector_test(tmp_path, monkeypatch, spec_name):
    import fit_fanout as FO
    engine = _load(f"run_engine_reduced_{spec_name[:3]}")
    engine.configure(NESTED_ROOT / "config" / "runs" / spec_name)
    run_dir = tmp_path / spec_name[:3]
    contract = _reduced(engine, run_dir, monkeypatch)
    meta = json.loads((run_dir / engine.SLOT_JSON_FMT.format(cls="nd", k=0, f=0, g=0)).read_text())
    assert meta["isovector_generation_fraction"] == engine.ISOVECTOR_GEN
    if engine.ISOVECTOR_SLOPE:
        assert meta["b_isovector"] == engine.GEN0_SHAPE[3]
    with np.load(run_dir / engine.SLOT_NPZ_FMT.format(cls="nd", k=0, f=0, g=0)) as arrays:
        iv = arrays["iv_config"]
    assert iv.size == 60 and set(np.unique(iv)) <= set(range(-1, 8)) and 0 < np.mean(iv >= 0) < 1
    monkeypatch.setattr(engine, "_FANOUT", FO.FanOut(run_dir, engine.FANOUT_REGISTRY, mode="local"))
    tables, a = engine.load_fit_tables(), engine.anchors()
    branch = engine.evaluate_generation(contract, 0, 0, tables, a)
    test = branch["isovector_test"]
    assert set(test) >= {"f_isovector_hat", "chi2_vacuum", "delta_chi2", "threshold", "required"}
    assert test["chi2_vacuum"] >= test["chi2_min"] - 1e-9
    assert branch["vacuum_only_fit"]["fitted"].keys() == set(engine.param_names(None, 0.0))
    if engine.ISOVECTOR_SLOPE:
        assert "b_isovector" in branch["fitted"] and "b_isovector" in branch["verdict"]["regeneration_point"]
        assert branch["generation_point"]["b_isovector"] == engine.GEN0_SHAPE[3]
    assert "neutron" in branch["objective_at_optimum"]["chi2_by_segment"]
    assert branch["observable_set"]["total"] == 1255 + 8
    variants = engine._variants(contract, branch, tables)
    assert all("isovector_test" in v for v in variants.values())
    summary = engine.test_summary(branch, variants)
    assert set(summary["isovector"]["outcomes"]) == {"primary", *variants}


# ----------------------------------------------------------------------
# The isovector transfer slope (R76)
# ----------------------------------------------------------------------
def _slope_spec(tmp_path, b0=8.0):
    spec = json.loads(SPEC.read_text())
    spec["isovector"]["own_slope"] = True
    params = spec["parameters"]
    params.insert(len(params) - 1, {"name": "b_isovector", "transform": "log", "bounds": [1.0, 40.0], "free": True, "gen0": b0,
                                    "hessian_step": 0.2, "starts": ["gen0", 4.0, 15.0, 6.0, 25.0, 3.0, 10.0, 30.0],
                                    "midpoint": "arithmetic", "bound_check": True})
    path = tmp_path / "slope.json"
    path.write_text(json.dumps(spec))
    return path


def test_the_slope_spec(tmp_path):
    e = _load("run_engine_slope_spec")
    e.configure(_slope_spec(tmp_path))
    assert e.ISOVECTOR_SLOPE and e.GEN0_SHAPE[3] == 8.0 and "b_isovector" in e.SHAPE_NAMES
    assert "b_isovector" not in e.free_shape_names(0.0) and "f_isovector" not in e.free_shape_names(0.0)
    phys = e.physical_from_x(e.dispersed_starts()[0], None, e.anchors())
    assert phys["b_isovector"] == pytest.approx(8.0, rel=1e-12)
    back = e.physical_from_x(e.x_from_physical(phys, None, 0.0), None, e.anchors(), 0.0)
    assert back["f_isovector"] == 0.0 and back["b_isovector"] == 8.0
    bad = json.loads(_slope_spec(tmp_path).read_text())
    del bad["isovector"]["own_slope"]
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="own_slope"):
        _load("run_engine_slope_bad").configure(tmp_path / "bad.json")


def test_the_slope_reweights_the_isovector_events_only(tmp_path):
    e = _load("run_engine_slope_weights")
    e.configure(_slope_spec(tmp_path))
    n = 400
    rng = np.random.default_rng(3)
    m2 = rng.uniform(1.2, 30.0, n)
    t = -rng.exponential(0.2, n) - 0.01
    block = {"n": n, "shape": {"m2": m2, "t": t, "m_rem": np.full(n, 0.93827231)}, "ledger": None, "iv": np.arange(n) % 3 == 0}
    tt = e.Templates.__new__(e.Templates)
    tt.point = (0.3, 3.0, 0.2, 8.0)
    tt.flavor_point = e.GEN_FLAVOR
    same = tt.log_weights("sd_projectile", block, 0.3, 3.0, 0.2, iso=0.55, b_iv=8.0)
    assert np.all(same == 0.0)
    steep = tt.log_weights("sd_projectile", block, 0.3, 3.0, 0.2, iso=0.55, b_iv=12.0)
    assert np.all(steep[~block["iv"]] == 0.0) and np.any(steep[block["iv"]] != 0.0)
    shared = tt.log_weights("sd_projectile", block, 0.3, 4.0, 0.2, iso=0.55, b_iv=8.0)
    assert np.all(shared[block["iv"]] == 0.0) and np.any(shared[~block["iv"]] != 0.0)


def test_the_r76_spec():
    e = _load("run_engine_r76_spec")
    e.configure(NESTED_ROOT / "config" / "runs" / "r76.json")
    assert e.ISOVECTOR_SLOPE and e.ISOVECTOR_GEN == 0.35 and e.GEN0_SHAPE[3] == 8.0
    assert e.SHAPE_NAMES[-2:] == ("f_isovector", "b_isovector") and e.N_FREE_PARAMETERS == 10
    spec = json.loads((NESTED_ROOT / "config" / "runs" / "r76.json").read_text())
    assert spec["generation_0"]["run"] == "r75_charge_sensitive_fit_2026-10-04"
    out = e.RC.check_seed_disjointness(e.RC.seed_ledger(NESTED_ROOT), e.LEDGER_RUN, (e.RAW_FIRST, e.RAW_LAST), e.PARTITION_BASE)
    assert out["disjoint"] and (e.RAW_FIRST, e.RAW_LAST) == (20320800, 20321311)
