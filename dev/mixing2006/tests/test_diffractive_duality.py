"""The diffractive lines through the I = 1/2 records (2026-10-15):
``TopologyNominalParams.diffractive_duality_mass`` maps a vacuum single- or
double-diffraction (1, 1, 0) line below Md onto the I = 1/2 low-mass records
at its own mass; off, the committed events."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import na49_mixing.topology_double_diffraction as DD
import na49_mixing.topology_single_diffraction as SD
from na49_mixing.kernel_config import KernelConfig, load_kernel
from na49_mixing.nd_duality import isoscalar_exchange_records
from na49_mixing.topology_runner import TopologyNominalParams, run_topology

NESTED_ROOT = Path(__file__).resolve().parents[1]
R97 = json.loads((NESTED_ROOT / "config" / "runs" / "r97.json").read_text())
LABELS = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")
BASE = TopologyNominalParams(
    epsilon=0.56, b_sd=6.16, epsilon_nd=0.253, nd_central=False, exact_transfer_range=True, line_thresholds=True,
    transfer_slope_mass_exponent=1.25, low_mass_spin_isospin=True,
    remnant_weights={"intact_proton": 1.0, "low_mass_baryonic_resonance": 0.0, "high_mass_baryonic_continuum": 0.0})
ON = dataclasses.replace(BASE, diffractive_duality_mass=1.7)


@pytest.fixture(scope="module")
def kernel():
    return load_kernel(KernelConfig(**R97["kernel"]), NESTED_ROOT)


def _event(cls, params, kernel, seed):
    catalog, fams = kernel
    rngs = {f"rng_{lab}": np.random.default_rng([seed, 17, k]) for k, lab in enumerate(LABELS)}
    return run_topology(cls, params, catalog, fams, **rngs)


def test_the_records(kernel):
    """The I = 1/2 records: the N* states with a width, no Delta."""
    catalog, _fams = kernel
    recs = isoscalar_exchange_records(catalog)
    assert recs and all(float(catalog.get(r.pdg_id).isospin) == 0.5 for r in recs)
    assert 2214 not in {r.pdg_id for r in recs} and 12212 in {r.pdg_id for r in recs}


def test_off_is_the_committed_event(kernel):
    for cls in ("sd_projectile", "dd"):
        for seed in range(8):
            a = _event(cls, BASE, kernel, seed)
            b = _event(cls, dataclasses.replace(BASE, diffractive_duality_mass=None), kernel, seed)
            assert a.final_state == b.final_state


def test_the_lines_below_md(kernel, monkeypatch):
    """A vacuum line below Md decays as an I = 1/2 record; a line at or
    above Md keeps the cascade; the events conserve B, Q and S."""
    picked = []
    orig = SD._record_decay

    def spy(momentum, mass, records, catalog, rc, rk, line):
        out = orig(momentum, mass, records, catalog, rc, rk, line)
        picked.append((float(mass), {r.pdg_id for r in records}))
        return out
    monkeypatch.setattr(SD, "_record_decay", spy)
    monkeypatch.setattr(DD, "_record_decay", spy)
    catalog, _fams = kernel
    i12 = {r.pdg_id for r in isoscalar_exchange_records(catalog)}
    for cls in ("sd_projectile", "sd_target", "dd"):
        for seed in range(60):
            ev = _event(cls, ON, kernel, seed)
            assert sum(int(h.qn[0]) for h in ev.final_state) == 2 and sum(int(h.qn[1]) for h in ev.final_state) == 2
            assert sum(int(h.qn[2]) for h in ev.final_state) == 0
    assert picked and all(m < 1.7 and recs == i12 for m, recs in picked)


def test_the_refusals(kernel):
    catalog, fams = kernel
    rngs = {f"rng_{lab}": np.random.default_rng([0, 17, k]) for k, lab in enumerate(LABELS)}
    for bad in (1.0, float("nan"), True):
        with pytest.raises(ValueError, match="diffractive_duality_mass"):
            run_topology("sd_projectile", dataclasses.replace(BASE, diffractive_duality_mass=bad), catalog, fams, **rngs)


def test_the_run_spec_option(tmp_path):
    path = NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py"
    spec = importlib.util.spec_from_file_location("run_engine_diffractive_duality", path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    (tmp_path / "s.json").write_text(json.dumps(dict(R97, topology={**R97["topology"], "diffractive_duality_mass": 1.7})))
    engine.configure(tmp_path / "s.json")
    point = (0.56, 6.16, 0.253, 1.37)
    assert engine._params_at("sd_projectile", point).diffractive_duality_mass == 1.7
    assert engine._params_at("dd", point).diffractive_duality_mass == 1.7
    assert engine._params_at("nd", point).diffractive_duality_mass is None
