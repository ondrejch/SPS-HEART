"""The baryon table corrections of 2026-10-21 (``KernelConfig.
baryon_table_corrections``; the Gemini review of R104, `.collab/`, gap B3):
the PDG 2025 tables of the committed Delta(1600) and Delta(1700); and the
PDG 2025 widths of the committed N* and Delta records
(``baryon_pdg_widths``)."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.baryon_table_corrections import (M_UP_WIDTHS, baryon_pdg_widths_catalog, baryon_table_corrections_catalog,
                                                   load_baryon_table_corrections)
from na49_mixing.cascade_decay import _qn_of
from na49_mixing.kernel_config import KernelConfig, load_kernel
from na49_mixing.topology_runner import TopologyNominalParams, run_topology

NESTED_ROOT = Path(__file__).resolve().parents[1]
RUNS = NESTED_ROOT / "config" / "runs"
LABELS = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")
DELTAS = {"Delta(1600)": (202224, 202214, 202114, 201114), "Delta(1700)": (122224, 122214, 122114, 121114)}


@pytest.fixture(scope="module")
def kernels():
    r105 = json.loads((RUNS / "r105.json").read_text())
    base = load_kernel(KernelConfig(**r105["kernel"]), NESTED_ROOT)
    full = load_kernel(KernelConfig(**{**r105["kernel"], "baryon_table_corrections": True}), NESTED_ROOT)
    return r105, base, full


def _modes(cat, pdg):
    return {c.daughters: c.branching_fraction for c in cat.get(pdg).decay_channels}


def test_the_options_are_off_by_default_and_validated():
    cfg = KernelConfig()
    for name in ("baryon_table_corrections", "baryon_pdg_widths"):
        assert getattr(cfg, name) is False and cfg.as_dict()[name] is False
        with pytest.raises(ValueError, match=f"{name} must be a bool"):
            KernelConfig(**{name: 1})


def test_the_delta_tables(kernels):
    _r, (cat0, _f0), (cat, _f) = kernels
    for name, ids in DELTAS.items():
        for pdg in ids + tuple(-p for p in ids):
            rec, rec0 = cat.get(pdg), cat0.get(pdg)
            # the tables only: the committed mass, width and support
            assert (rec.pole_mass, rec.nominal_width, rec.mass_range) == (rec0.pole_mass, rec0.nominal_width, rec0.mass_range)
            assert math.isclose(math.fsum(c.branching_fraction for c in rec.decay_channels), 1.0, rel_tol=1e-12)
            for c in rec.decay_channels:
                assert tuple(sum(int(_qn_of(cat, d)[i]) for d in c.daughters) for i in range(3)) == (
                    int(rec.B), int(rec.Q), int(rec.S)), (rec.name, c.daughters)
    # Delta(1600)+: N pi 13.4%, Delta pi 66.2%, N(1440) pi 20.4% (the range rule), no N rho
    d1600 = _modes(cat, 202214)
    assert math.isclose(d1600[(2212, 111)] + d1600[(2112, 211)], 0.1344, abs_tol=5e-5)
    assert math.isclose(d1600[(12212, 111)] + d1600[(12112, 211)], 0.204, abs_tol=5e-5)
    assert not any(213 in abs_d or 113 in abs_d for abs_d in (tuple(map(abs, d)) for d in d1600))
    assert math.isclose(d1600[(2212, 111)] / d1600[(2112, 211)], 2.0, rel_tol=1e-9)
    # Delta(1700)++: Delta(1232) eta at 5.4%, N(1520) pi and N(1535) pi present
    d1700 = _modes(cat, 122224)
    assert math.isclose(d1700[(2224, 221)], 0.0542, abs_tol=5e-5)
    assert (2216, 211) in d1700 and (102212, 211) in d1700


def test_an_absent_record_is_refused(kernels):
    _r, (cat0, _f0), _full = kernels
    config = load_baryon_table_corrections(NESTED_ROOT)
    bad = {**config, "delta_tables": [{**config["delta_tables"][0], "pdg_ids": {"1": 9999999}}]}
    with pytest.raises(ValueError, match="absent"):
        baryon_table_corrections_catalog(cat0, bad)


def test_events_conserve_with_the_corrections(kernels):
    r105, _b, (cat, fams) = kernels
    params = TopologyNominalParams(epsilon=0.52, b_sd=6.43, epsilon_nd=0.247, nd_central=False,
                                   remnant_weights={"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.0,
                                                    "high_mass_baryonic_continuum": 1.0}, **r105["topology"])
    for cls in ("nd", "sd_projectile", "dd"):
        p = params if cls == "nd" else dataclasses.replace(params, remnant_weights={
            "intact_proton": 1.0, "low_mass_baryonic_resonance": 0.0, "high_mass_baryonic_continuum": 0.0})
        for seed in range(15):
            rngs = {f"rng_{lab}": np.random.default_rng([seed, 21022, k]) for k, lab in enumerate(LABELS)}
            ev = run_topology(cls, p, cat, fams, **rngs)
            for i in range(3):
                assert sum(int(h.qn[i]) for h in ev.final_state) == (2, 2, 0)[i]


def test_the_pdg_widths(kernels):
    _r, (cat0, _f0), _full = kernels
    config = load_baryon_table_corrections(NESTED_ROOT)
    cat = baryon_pdg_widths_catalog(cat0, config)
    expected = {12212: 0.35, 12112: 0.35, 2216: 0.11, 2116: 0.11, 22212: 0.12, 22112: 0.12, 212214: 0.25, 212114: 0.25,
                202224: 0.25, 202214: 0.25, 202114: 0.25, 201114: 0.25}
    assert {int(p) for spec in config["widths"] for p in spec["pdg_ids"]} == set(expected)
    for pdg, width in expected.items():
        rec, rec0 = cat.get(pdg), cat0.get(pdg)
        assert rec.nominal_width == width and rec.pole_mass == rec0.pole_mass
        assert rec.decay_channels == rec0.decay_channels
        assert rec.mass_range == (rec0.mass_range[0], rec0.pole_mass + M_UP_WIDTHS * width)
        # the committed records follow the five-width rule too
        assert math.isclose(rec0.mass_range[1], rec0.pole_mass + M_UP_WIDTHS * rec0.nominal_width, rel_tol=1e-12)
    assert not cat.validate().errors
    untouched = [r for r in cat.records() if int(r.pdg_id) not in expected]
    assert all(r is cat0.get(int(r.pdg_id)) for r in untouched)


def test_the_r106_spec_is_r105_with_the_corrections():
    r105, r106 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r105", "r106"))
    assert r106["kernel"] == {**r105["kernel"], "baryon_table_corrections": True, "baryon_pdg_widths": True}
    assert r106["topology"] == r105["topology"]
    for block in ("fit", "diffractive_split", "isovector", "fixed", "statistics"):
        assert r106[block] == r105[block]
    assert [p["name"] for p in r106["parameters"]] == [p["name"] for p in r105["parameters"]]
    assert r106["generation_0"]["run"] == "r105_completions_2026-10-21"
    assert r106["seeds"]["raw_first"] > r105["seeds"]["raw_first"] + 511
    branch = NESTED_ROOT.parent / "00runs" / "r105_completions_2026-10-21" / "branch_report_f0.json"
    if branch.is_file():
        report = json.loads(branch.read_text())
        assert r106["generation_0"]["branch_digest"] == report["digest"]
        gen0 = {p["name"]: p["gen0"] for p in r106["parameters"]}
        for name in r106["generation_0"]["from_fit"]:
            assert float(gen0[name]) == float(report["fitted"][name]), name
