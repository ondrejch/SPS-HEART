"""The light scalar mesons (2026-10-11): ``KernelConfig.light_scalar_mesons``
adds the f0(980) record, the three a0(980) charge states and their classes
in the neutral mesonic family and the non-strange baryonic family, from
``config/v2_singlet/light_scalar_mesons_pdg2025.json``; off, the committed
catalog and families."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.cascade_decay import ISOTROPIC_ALIGNMENT_LABEL, CascadeState, enumerate_channels
from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.kernel_config import CATALOG_REL, FAMILIES_REL, KernelConfig, load_kernel
from na49_mixing.light_scalar_mesons import (light_scalar_mesons_catalog, light_scalar_mesons_families,
                                             load_light_scalar_mesons)
from na49_mixing.spectral_density import families_from_json
from na49_mixing.topology_runner import TopologyNominalParams, run_topology

NESTED_ROOT = Path(__file__).resolve().parents[1]
R91 = KernelConfig(**json.loads((NESTED_ROOT / "config" / "runs" / "r91.json").read_text())["kernel"])
ON = dataclasses.replace(R91, light_scalar_mesons=True)
F0, A0_0, A0_P, A0_M = 9010221, 9000111, 9000211, -9000211
SCALARS = (F0, A0_0, A0_P, A0_M)
LABELS = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")


@pytest.fixture(scope="module")
def config():
    return load_light_scalar_mesons(NESTED_ROOT)


@pytest.fixture(scope="module")
def kernels():
    return {"off": load_kernel(R91, NESTED_ROOT), "on": load_kernel(ON, NESTED_ROOT)}


def _root(family_id, mass, qn):
    return CascadeState(momentum=(float(mass), 0.0, 0.0, 0.0), mass=float(mass), qn=tuple(qn), quantum={},
                        family_id=family_id, axis=(0.0, 0.0, 1.0), alignment_label=ISOTROPIC_ALIGNMENT_LABEL, parent=None,
                        exchange_ancestry=("scalar_probe",), depth=1)


def test_off_is_the_committed_kernel(kernels):
    """Off, no scalar record or class; on, the catalog adds the four
    records and every other record and family class stays."""
    cat_off, fams_off = kernels["off"]
    cat_on, fams_on = kernels["on"]
    assert not {int(r.pdg_id) for r in cat_off.records()} & set(SCALARS)
    assert not any(str(abs(p)) in c for p in SCALARS for f in fams_off for c in f.allowed_decay_classes)
    assert {int(r.pdg_id) for r in cat_on.records()} == {int(r.pdg_id) for r in cat_off.records()} | set(SCALARS)
    for record in cat_off.records():
        assert cat_on.get(record.pdg_id) == record
    for f_off, f_on in zip(fams_off, fams_on, strict=True):
        assert f_off.family_id == f_on.family_id
        kept = tuple(c for c in f_on.allowed_decay_classes if not any(str(abs(p)) in c for p in SCALARS))
        assert kept == tuple(f_off.allowed_decay_classes)
    assert R91.as_dict()["light_scalar_mesons"] is False and ON.as_dict()["light_scalar_mesons"] is True


def test_the_records(kernels):
    catalog, _fams = kernels["on"]
    f0 = catalog.get(F0)
    assert (f0.B, f0.Q, f0.S, f0.isospin, f0.isospin3, f0.spin, f0.parity) == (0, 0, 0, 0.0, 0, 0.0, "+")
    assert (f0.pole_mass, f0.nominal_width) == (0.990, 0.055)
    assert f0.mass_range == pytest.approx((2.0 * catalog._daughter_mass(111), 0.990 + 5 * 0.055))
    assert {c.daughters: c.branching_fraction for c in f0.decay_channels} == {
        (211, -211): pytest.approx(0.68 * 2 / 3), (111, 111): pytest.approx(0.68 / 3),
        (321, -321): pytest.approx(0.16), (311, -311): pytest.approx(0.16)}
    kk = 0.172 / 1.172
    expected = {A0_0: {(221, 111): 1 / 1.172, (321, -321): kk / 2, (311, -311): kk / 2},
                A0_P: {(221, 211): 1 / 1.172, (321, -311): kk}, A0_M: {(221, -211): 1 / 1.172, (-321, 311): kk}}
    for pid, (q, i3) in ((A0_0, (0, 0)), (A0_P, (1, 2)), (A0_M, (-1, -2))):
        a0 = catalog.get(pid)
        assert (a0.B, a0.Q, a0.S, a0.isospin, a0.isospin3) == (0, q, 0, 1.0, i3)
        assert (a0.pole_mass, a0.nominal_width, a0.mass_range[1]) == pytest.approx((0.980, 0.075, 0.980 + 5 * 0.075))
        assert {c.daughters: c.branching_fraction for c in a0.decay_channels} == pytest.approx(expected[pid])
    for pid in SCALARS:
        record = catalog.get(pid)
        assert math.fsum(c.branching_fraction for c in record.decay_channels) == pytest.approx(1.0, abs=1e-12)
        assert all(c.orbital_angular_momentum == 0 for c in record.decay_channels)
        assert not record.stable_for_decay
    assert tuple(catalog.validate().errors) == ()


def test_the_classes(kernels):
    """The neutral mesonic pool takes the four scalar classes of isospin 0; (N, f0(980))
    joins the I = 1/2 families beside (N, f0(500)) and opens there as a
    direct pair below the floor and as a continuation step above it, the
    I = 3/2 families take the pair at Clebsch-Gordan weight 0, and the
    conjugate towers carry its conjugate."""
    catalog, fams = kernels["on"]
    classes = {f.family_id: list(f.allowed_decay_classes) for f in fams}
    assert classes["mesonic_neutral"][-4:] == [f"(223,{F0})", f"({A0_0},111)", f"({A0_P},-211)", f"({A0_M},211)"]
    for family_id, nucleon in (("baryonic_np", 2212), ("baryonic_np_q0", 2112), ("anti_baryonic_np", -2212),
                               ("anti_baryonic_np_q0", -2112)):
        assert f"({nucleon},{F0})" in classes[family_id], family_id
    family = {f.family_id: f for f in fams}
    for family_id, nucleon, open_ in (("baryonic_np", 2212, True), ("baryonic_np_q0", 2112, True),
                                      ("baryonic_np_delta_q1", 2212, False), ("anti_baryonic_np", -2212, True)):
        chans = enumerate_channels(_root(family_id, 1.99, family[family_id].qn), catalog, fams)
        weight = math.fsum(c.weight for c in chans if set(c.daughters) == {f"catalog:{nucleon}", f"catalog:{F0}"})
        assert (weight > 0.0) is open_, family_id
    chans = enumerate_channels(_root("baryonic_np", 3.0, (1, 1, 0)), catalog, fams)
    assert any(f"catalog:{F0}" in c.daughters for c in chans)
    chans = enumerate_channels(_root("mesonic_neutral", 1.9, (0, 0, 0)), catalog, fams)
    opened = {d for c in chans for d in c.daughters}
    assert {f"catalog:{p}" for p in SCALARS} <= opened


def test_the_refusals(config):
    committed = HadronCatalog.from_json(NESTED_ROOT / CATALOG_REL)
    once = light_scalar_mesons_catalog(committed, config)
    with pytest.raises(ValueError, match="already holds"):
        light_scalar_mesons_catalog(once, config)
    fams = light_scalar_mesons_families(families_from_json(str(NESTED_ROOT / FAMILIES_REL)), config)
    with pytest.raises(ValueError, match="already holds"):
        light_scalar_mesons_families(fams, config)
    bad = json.loads(json.dumps(config))
    bad["records"][2]["modes"][1]["kind"] = "KKbar_neutral"
    with pytest.raises(ValueError, match="no 'KKbar_neutral' split"):
        light_scalar_mesons_catalog(committed, bad)
    for value in ("yes", 1, None):
        with pytest.raises(ValueError, match="light_scalar_mesons"):
            KernelConfig(light_scalar_mesons=value)


@pytest.mark.parametrize("cls", ["sd_projectile", "dd", "nd"])
def test_events_conserve_charge(kernels, cls):
    """Events under the option conserve the charge and baryon number of the
    p+p state, and the scalars never reach the final state."""
    catalog, fams = kernels["on"]
    weights = ({"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.05, "high_mass_baryonic_continuum": 0.95} if cls == "nd"
               else {"intact_proton": 0.5, "low_mass_baryonic_resonance": 0.5, "high_mass_baryonic_continuum": 0.0})
    params = TopologyNominalParams(epsilon=0.5, b_sd=6.9, epsilon_nd=0.24, remnant_weights=weights, nd_central=False,
                                   exact_transfer_range=True, line_thresholds=True)
    for seed in range(8):
        rngs = {f"rng_{lab}": np.random.default_rng([seed, 41, k]) for k, lab in enumerate(LABELS)}
        ev = run_topology(cls, params, catalog, fams, **rngs)
        qns = [tuple(int(v) for v in h.qn) if h.pdg_id is None else
               (catalog._daughter_quantum_numbers(h.pdg_id).B, catalog._daughter_quantum_numbers(h.pdg_id).Q)
               for h in ev.final_state]
        assert sum(q[1] for q in qns) == 2 and sum(q[0] for q in qns) == 2
        assert all(h.pdg_id not in SCALARS for h in ev.final_state)


def test_the_run_spec_option(tmp_path):
    path = NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py"
    spec = importlib.util.spec_from_file_location("run_engine_light_scalar_mesons", path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    base = json.loads((NESTED_ROOT / "config" / "runs" / "r91.json").read_text())
    good = dict(base, kernel={**base["kernel"], "light_scalar_mesons": True})
    (tmp_path / "good.json").write_text(json.dumps(good))
    engine.configure(tmp_path / "good.json")
    assert engine.KERNEL.light_scalar_mesons is True
    catalog, _fams = engine._load_physics(engine.FLOOR_GRID[0])
    assert catalog.get(F0).name == "f0(980)"


def test_the_r92_spec_is_r91_with_the_scalars_at_the_r91_optimum():
    runs = NESTED_ROOT / "config" / "runs"
    r91, r92 = (json.loads((runs / f"{r}.json").read_text()) for r in ("r91", "r92"))
    assert r92["kernel"] == {**r91["kernel"], "light_scalar_mesons": True}
    for block in ("topology", "fit", "diffractive_split", "isovector"):
        assert r92[block] == r91[block]
    assert r92["generation_0"]["run"] == "r91_meson_split_2026-10-11" and r92["generation_0"]["fit"] == "free"
    assert [p["name"] for p in r92["parameters"]] == [p["name"] for p in r91["parameters"]]
    assert r92["seeds"]["master"] == r91["seeds"]["master"]
    assert r92["seeds"]["raw_first"] > r91["seeds"]["raw_first"] + 511
    assert r92["seeds"]["partition_base"] != r91["seeds"]["partition_base"]
    assert KernelConfig(**r92["kernel"]).light_scalar_mesons is True
