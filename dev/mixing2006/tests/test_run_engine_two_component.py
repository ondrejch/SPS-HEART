"""The run engine's two-component mode and topology options (2026-10-06,
the single fit): a spec with "central_component": false fixes f_central = 0
by construction (the class with the central source is neither generated nor
loaded, no null fit, scan or central test), and its "topology" block passes
the exact transfer window and the per-line thresholds to the generators and
the exact window to the reweighting.  Without the two keys the engine is the
committed one (test_run_engine.py holds the R74 register bit for bit)."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.topology_mass_laws import log_density_ratio

NESTED_ROOT = Path(__file__).resolve().parents[1]
ENGINE = NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py"
R75 = NESTED_ROOT / "config" / "runs" / "r75.json"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ENGINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _two_component(tmp_path, **edits):
    spec = json.loads(R75.read_text())
    spec["central_component"] = False
    spec["topology"] = {"exact_transfer_range": True, "line_thresholds": True}
    spec["kernel"] = {**spec["kernel"], "three_body_phase_space": True}
    for p in spec["parameters"]:
        if p["name"] == "f_central":
            p.update({"free": False, "value": 0.0, "gen0": 0.0})
    for key, value in edits.items():
        spec[key] = value
    path = tmp_path / "two_component.json"
    path.write_text(json.dumps(spec))
    return path


def test_the_two_component_constants(tmp_path):
    e = _load("run_engine_two_component_constants")
    e.configure(_two_component(tmp_path))
    assert e.CENTRAL_COMPONENT is False and e.FC_DEFAULT == 0.0
    assert e.ACTIVE_CLASSES == ("sd_projectile", "sd_target", "dd", "nd0")
    assert all(e.slot_class(s) != "nd" for s in e.ACTIVE_SLOTS) and len(e.ACTIVE_SLOTS) == 4 * e.REPLICAS
    assert "f_central" not in e.param_names(None) and e.N_FREE_PARAMETERS == len(e.SHAPE_NAMES)
    assert len(e.x_from_physical({**{n: 1.0 for n in e.SHAPE_NAMES}, "w_excited": 0.8, "f_central": 0.3})) == len(e.SHAPE_NAMES)
    params = e._params_at("nd0", e.GEN0_SHAPE, e.GEN_FLAVOR)
    assert params.exact_transfer_range and params.line_thresholds and not params.nd_central
    assert e.EXACT_RANGE and e.KERNEL.three_body_phase_space
    assert e.class_fractions({"f_sd_total": 0.2, "f_dd": 0.1, "f_nd": 0.7, "f_central": 0.0})["nd"] == 0.0


def test_the_committed_mode_is_unchanged():
    e = _load("run_engine_committed_mode")
    e.configure(R75)
    assert e.CENTRAL_COMPONENT is True and e.FC_DEFAULT is None and e.ACTIVE_CLASSES == e.CLASS_ORDER
    assert e.TOPOLOGY_OPTIONS == {} and e.EXACT_RANGE is False
    assert e.param_names(None)[-1] == "f_central" and e.N_FREE_PARAMETERS == len(e.SHAPE_NAMES) + 1


@pytest.mark.parametrize("edit, message", [
    ({"central_component": "no"}, "central_component must be a bool"),
    ({"topology": {"exact_transfer_range": "yes"}}, "topology options are booleans"),
    ({"topology": {"clamp": True}}, "topology options are booleans"),
])
def test_the_spec_validates(tmp_path, edit, message):
    with pytest.raises(ValueError, match=message):
        _load("run_engine_two_component_bad").configure(_two_component(tmp_path, **edit))


def test_the_spec_refuses_a_free_central_fraction_and_keeps_the_window_off_the_central_class(tmp_path):
    spec = json.loads(_two_component(tmp_path).read_text())
    free = copy.deepcopy(spec)
    for p in free["parameters"]:
        if p["name"] == "f_central":
            p.update({"free": True, "value": None, "gen0": 0.1})
    (tmp_path / "free.json").write_text(json.dumps(free))
    with pytest.raises(ValueError, match="f_central is fixed at 0"):
        _load("run_engine_free_fc").configure(tmp_path / "free.json")
    # (2026-10-07) the exact window applies to the 2 -> 2 classes; the class
    # with the central source keeps its committed draw (at rest or recoiling)
    central = json.loads(R75.read_text())
    central["topology"] = {"exact_transfer_range": True, "central_recoil": True}
    (tmp_path / "central.json").write_text(json.dumps(central))
    e = _load("run_engine_window_central")
    e.configure(tmp_path / "central.json")
    nd = e._params_at("nd", e.GEN0_SHAPE, e.GEN_FLAVOR); nd0 = e._params_at("nd0", e.GEN0_SHAPE, e.GEN_FLAVOR)
    assert nd.nd_central and not nd.exact_transfer_range and nd.central_recoil
    assert not nd0.nd_central and nd0.exact_transfer_range
    recoil_alone = json.loads(_two_component(tmp_path).read_text())
    recoil_alone["topology"] = {**recoil_alone["topology"], "central_recoil": True}
    (tmp_path / "recoil.json").write_text(json.dumps(recoil_alone))
    with pytest.raises(ValueError, match="central_recoil needs central_component true"):
        _load("run_engine_recoil_alone").configure(tmp_path / "recoil.json")


def test_a_reduced_two_component_run(tmp_path, monkeypatch):
    import fit_fanout as FO
    e = _load("run_engine_two_component_reduced")
    e.configure(_two_component(tmp_path))
    replicas = 2
    monkeypatch.setattr(e, "REPLICAS", replicas)
    monkeypatch.setattr(e, "N_SLOTS", len(e.CLASS_ORDER) * replicas)
    monkeypatch.setattr(e, "SEEDED_SLOTS", (len(e.CLASS_ORDER) - 1) * replicas)
    monkeypatch.setattr(e, "ACTIVE_SLOTS", tuple(s for s in range(len(e.CLASS_ORDER) * replicas)
                                                 if e.CLASS_ORDER[s // replicas] in e.ACTIVE_CLASSES))
    for name, value in (("MESONIC_DECAYS", 20), ("N_STARTS", 2), ("N_STARTS_NULL", 2), ("N_STARTS_VARIANT", 2),
                        ("FIT_MAXFEV", 300)):
        monkeypatch.setattr(e, name, value)
    monkeypatch.setattr(e.RC, "run_replicas", lambda worker, jobs, max_workers=None: [worker(*job) for job in jobs])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    contract = e._contract(run_dir)
    for slot in e.ACTIVE_SLOTS:
        res = e.generate_slot(e.slot_class(slot), e.SMOKE_SEED + e.seed_slot(slot), 40,
                              e.SMOKE_PARTITION + 10_000 * e.seed_slot(slot), e.GEN0_SHAPE, e.FLOOR_GRID[0], e.GEN_FLAVOR)
        e.write_slot(contract, 0, 0, slot, res, "test")
    assert not (run_dir / e.SLOT_JSON_FMT.format(cls="nd", k=0, f=0, g=0)).exists()
    assert e._generation_complete(contract, 0, 0)

    # the reweighting runs on the exact window and the recorded line edges
    tables = e.load_fit_tables()
    templates = e.Templates(contract, 0, 0, tables)
    assert set(templates.by_class) == set(e.ACTIVE_CLASSES)
    block = templates.by_class["dd"][0]
    assert {"m_lo1", "m_lo2"} <= set(block["shape"])
    eps0, b0, _ = templates.point[:3]
    expected = log_density_ratio("dd", block["shape"], eps0 + 0.1, b0 * 0.9, eps0=eps0, b_sd0=b0, sqrt_s=e.SQRT_S,
                                 exact_range=True)
    got = templates.log_weights("dd", block, eps0 + 0.1, b0 * 0.9, templates.point[2])
    assert np.allclose(got, expected)

    monkeypatch.setattr(e, "_FANOUT", FO.FanOut(run_dir, e.FANOUT_REGISTRY, mode="local"))
    branch = e.evaluate_generation(contract, 0, 0, tables, e.anchors())
    test = branch["central_component_test"]
    assert test["absent_by_construction"] and test["f_central_hat"] == 0.0 and test["profile"] == []
    assert branch["fitted"]["f_central"] == 0.0 and branch["fitted"]["class_fractions"]["nd"] == 0.0
    assert branch["profile_scan"] == []
    assert branch["null_fit"]["objective"] == pytest.approx(branch["objective_at_optimum"]["value"])
    assert "isovector_test" in branch and branch["statistics"]["central_component"] is False
    assert branch["statistics"]["topology_options"] == {"exact_transfer_range": True, "line_thresholds": True}
    variants = e._variants(contract, branch, tables)
    assert all(v["central_component_test"]["absent_by_construction"] and "isovector_test" in v for v in variants.values())
    summary = e.test_summary(branch, variants)
    assert summary["central_component"].startswith("absent") and summary["compatible_in_primary"] is None
    assert set(summary["isovector"]["outcomes"]) == {"primary", *variants}
    report = e.compose_report(branch, variants)
    text = e.render_report_md(report)
    assert "absent by construction" in text


def test_the_r77_spec():
    """The single fit (2026-10-06, owner: "Confirm"): the two-component mode,
    the exact window and the per-line thresholds, the flat three-body phase
    space, the committed sibling weight and the isotropic cascade; the seeds
    are the ledger's r77 reservation; generation 0 is the R75 free optimum."""
    e = _load("run_engine_r77")
    e.configure(NESTED_ROOT / "config" / "runs" / "r77.json")
    assert e.CENTRAL_COMPONENT is False and e.ACTIVE_CLASSES == ("sd_projectile", "sd_target", "dd", "nd0")
    assert e.TOPOLOGY_OPTIONS == {"exact_transfer_range": True, "line_thresholds": True}
    assert e.KERNEL.three_body_phase_space and e.KERNEL.continuation_sibling_weight is None
    assert e.KERNEL.isospin_completion == ("baryonic", "strange_baryonic", "strange_mesonic") and e.KERNEL.delta_lines
    assert e.param_names(None) == ("epsilon", "b_sd", "epsilon_nd", "w_excited", "gamma_s_had", "assoc_norm", "r_bbar",
                                   "f_isovector")
    assert e.ISOVECTOR_GEN == 0.25 and e.NEUTRON_SEGMENT
    ledger = e.RC.seed_ledger(NESTED_ROOT)
    e.RC.check_seed_disjointness(ledger, e.LEDGER_RUN, (e.RAW_FIRST, e.RAW_LAST), e.PARTITION_BASE)
    assert (e.RAW_FIRST, e.RAW_LAST, e.PARTITION_BASE) == (20321400, 20321911, 4450000000)
    spec = json.loads((NESTED_ROOT / "config" / "runs" / "r77.json").read_text())
    assert spec["generation_0"]["fit"] == "free" and spec["generation_0"]["run"] == "r75_charge_sensitive_fit_2026-10-04"
