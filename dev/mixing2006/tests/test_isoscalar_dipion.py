"""The isoscalar S-wave dipion of the nucleon resonances (2026-10-08):
``KernelConfig.isoscalar_dipion`` adds the f0(500) record, sets the six N*
multiplets to the PDG 2025 hadronic decay modes of
``config/v2_singlet/isoscalar_dipion_pdg2025.json`` and adds the
(N, f0(500)) class to the baryonic continuum's direct pool; off, the
committed catalog and families."""

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
from na49_mixing.isoscalar_dipion import (F0_500_PDG, isoscalar_dipion_catalog, isoscalar_dipion_families,
                                          load_isoscalar_dipion)
from na49_mixing.kernel_config import CATALOG_REL, FAMILIES_REL, R74, KernelConfig, load_kernel
from na49_mixing.spectral_density import families_from_json
from na49_mixing.topology_runner import TopologyNominalParams, run_topology

NESTED_ROOT = Path(__file__).resolve().parents[1]
BASE = dataclasses.replace(R74, isospin_completion=("baryonic", "strange_baryonic", "strange_mesonic"), delta_lines=True,
                           three_body_phase_space=True)
ON = dataclasses.replace(BASE, isoscalar_dipion=True)
LABELS = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")
CG = {"Npi": (1 / 3, 2 / 3), "Nrho": (1 / 3, 2 / 3), "N1440pi": (1 / 3, 2 / 3), "N1520pi": (1 / 3, 2 / 3),
      "Dpi": (1 / 2, 1 / 3, 1 / 6), "Nsigma": (1.0,), "Neta": (1.0,), "Nomega": (1.0,), "LK": (1.0,)}


@pytest.fixture(scope="module")
def config():
    return load_isoscalar_dipion(NESTED_ROOT)


@pytest.fixture(scope="module")
def kernel_on():
    return load_kernel(ON, NESTED_ROOT)


def _root(family_id, mass, qn):
    return CascadeState(momentum=(float(mass), 0.0, 0.0, 0.0), mass=float(mass), qn=tuple(qn), quantum={},
                        family_id=family_id, axis=(0.0, 0.0, 1.0), alignment_label=ISOTROPIC_ALIGNMENT_LABEL, parent=None,
                        exchange_ancestry=("dipion_probe",), depth=1)


def test_off_is_the_committed_kernel():
    cat_off, fams_off = load_kernel(BASE, NESTED_ROOT)
    committed = HadronCatalog.from_json(NESTED_ROOT / CATALOG_REL)
    assert cat_off.records() == committed.records()
    assert all(int(r.pdg_id) != F0_500_PDG for r in cat_off.records())
    assert not any(str(F0_500_PDG) in c for f in fams_off for c in f.allowed_decay_classes)
    assert BASE.as_dict()["isoscalar_dipion"] is False and ON.as_dict()["isoscalar_dipion"] is True


def test_the_f0_record(kernel_on):
    catalog, _fams = kernel_on
    f0 = catalog.get(F0_500_PDG)
    assert (f0.B, f0.Q, f0.S, f0.isospin, f0.isospin3, f0.spin, f0.parity) == (0, 0, 0, 0.0, 0, 0.0, "+")
    assert (f0.pole_mass, f0.nominal_width) == (0.475, 0.55)
    assert f0.mass_range[0] == pytest.approx(2.0 * catalog._daughter_mass(111)) and f0.mass_range[1] == 1.0
    shares = {tuple(sorted(c.daughters)): c.branching_fraction for c in f0.decay_channels}
    assert shares == {(-211, 211): pytest.approx(2 / 3), (111, 111): pytest.approx(1 / 3)}
    assert all(c.orbital_angular_momentum == 0 for c in f0.decay_channels)
    assert catalog.three_body_law == "phase_space"
    assert tuple(catalog.validate().errors) == ()


def test_the_nstar_tables_follow_the_config(config, kernel_on):
    """Each mode at its PDG midpoint over the multiplet's sum, split by the
    Clebsch-Gordan shares; every table sums to one; the other records keep
    their committed tables."""
    catalog, _fams = kernel_on
    committed = HadronCatalog.from_json(NESTED_ROOT / CATALOG_REL)
    replaced = set()
    for multiplet in config["nstar"]:
        total = math.fsum(m["fraction"] for m in multiplet["channels"])
        for pid in multiplet["pdg_ids"]:
            channels = catalog.get(pid).decay_channels
            replaced.add(int(pid))
            assert math.fsum(c.branching_fraction for c in channels) == pytest.approx(1.0, abs=1e-12)
            expected = sorted(m["fraction"] / total * share for m in multiplet["channels"] for share in CG[m["kind"]])
            assert sorted(c.branching_fraction for c in channels) == pytest.approx(expected, abs=1e-12)
    for record in committed.records():
        if int(record.pdg_id) not in replaced:
            assert catalog.get(record.pdg_id) == record


def _proton_fraction(catalog, pid):
    """The proton share of the final nucleon of a nonstrange baryon record,
    through its tables (channels without a nonstrange baryon are skipped)."""
    if pid in (2212, 2112):
        return 1.0 if pid == 2212 else 0.0
    total = weight = 0.0
    for c in catalog.get(pid).decay_channels:
        qns = {d: catalog._daughter_quantum_numbers(d) for d in c.daughters}
        nucleons = [d for d, q in qns.items() if q.B == 1 and q.S == 0]
        if nucleons:
            total += c.branching_fraction * _proton_fraction(catalog, nucleons[0])
            weight += c.branching_fraction
    return total / weight


def test_the_isoscalar_partner_keeps_the_proton(kernel_on):
    """The final nucleon of N(1440)+: 0.41 protons in the committed table
    (N sigma folded into N pi), 0.53 with N sigma represented."""
    catalog, _fams = kernel_on
    committed = HadronCatalog.from_json(NESTED_ROOT / CATALOG_REL)
    assert _proton_fraction(committed, 12212) == pytest.approx(0.41, abs=0.01)
    assert _proton_fraction(catalog, 12212) == pytest.approx(0.53, abs=0.01)
    assert _proton_fraction(catalog, 212214) > 0.75  # N(1720): N omega and Delta pi


def test_the_floor_class_reaches_the_charge_partners(kernel_on):
    catalog, fams = kernel_on
    classes = {f.family_id: set(f.allowed_decay_classes) for f in fams}
    assert f"(2212,{F0_500_PDG})" in classes["baryonic_np"]
    assert f"(2112,{F0_500_PDG})" in classes["baryonic_np_q0"]
    for family_id, qn, nucleon, open_ in (("baryonic_np", (1, 1, 0), 2212, True), ("baryonic_np_q0", (1, 0, 0), 2112, True),
                                          ("baryonic_np_delta_q1", (1, 1, 0), 2212, False)):
        chans = enumerate_channels(_root(family_id, 1.9, qn), catalog, fams)
        weight = math.fsum(c.weight for c in chans if set(c.daughters) == {f"catalog:{nucleon}", f"catalog:{F0_500_PDG}"})
        assert (weight > 0.0) is open_, family_id


def test_the_refusals(config):
    committed = HadronCatalog.from_json(NESTED_ROOT / CATALOG_REL)
    once = isoscalar_dipion_catalog(committed, config)
    with pytest.raises(ValueError, match="already holds the f0"):
        isoscalar_dipion_catalog(once, config)
    fams = isoscalar_dipion_families(families_from_json(str(NESTED_ROOT / FAMILIES_REL)), config)
    with pytest.raises(ValueError, match="already holds"):
        isoscalar_dipion_families(fams, config)
    for bad in ("yes", 1, None):
        with pytest.raises(ValueError, match="isoscalar_dipion"):
            KernelConfig(isoscalar_dipion=bad)


@pytest.mark.parametrize("cls", ["sd_projectile", "dd", "nd"])
def test_events_conserve_charge(kernel_on, cls):
    """Events under the option conserve the charge of the p+p state, and
    the f0(500) never reaches the final state."""
    catalog, fams = kernel_on
    weights = ({"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.25, "high_mass_baryonic_continuum": 0.75} if cls == "nd"
               else {"intact_proton": 0.5, "low_mass_baryonic_resonance": 0.5, "high_mass_baryonic_continuum": 0.0})
    params = TopologyNominalParams(epsilon=0.4, b_sd=3.0, epsilon_nd=0.16, remnant_weights=weights, nd_central=False,
                                   exact_transfer_range=True)
    for seed in range(8):
        rngs = {f"rng_{lab}": np.random.default_rng([seed, 31, k]) for k, lab in enumerate(LABELS)}
        ev = run_topology(cls, params, catalog, fams, **rngs)
        charge = sum(int(h.qn[1]) if h.pdg_id is None else catalog._daughter_quantum_numbers(h.pdg_id).Q
                     for h in ev.final_state)
        assert charge == 2
        assert all(h.pdg_id != F0_500_PDG for h in ev.final_state)


def test_the_run_spec_option(tmp_path):
    path = NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py"
    spec = importlib.util.spec_from_file_location("run_engine_isoscalar_dipion", path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    base = json.loads((NESTED_ROOT / "config" / "runs" / "r83.json").read_text())
    good = dict(base, kernel={**base["kernel"], "isoscalar_dipion": True})
    (tmp_path / "good.json").write_text(json.dumps(good))
    engine.configure(tmp_path / "good.json")
    assert engine.KERNEL.isoscalar_dipion is True
    catalog, _fams = engine._load_physics(engine.FLOOR_GRID[0])
    assert catalog.get(F0_500_PDG).name == "f0(500)"
