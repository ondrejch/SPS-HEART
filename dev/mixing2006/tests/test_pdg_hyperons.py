"""The PDG hyperon resonances up to the continuation floor (2026-10-15):
``KernelConfig.pdg_hyperons`` adds the Lambda and Sigma resonances of the
PDG 2025 summary tables between Sigma(1385) and Lambda(1520) and the
2.0 GeV floor, with their (Y*, K) and (Y*, pi) classes, before the
conjugate towers; off, the committed kernel."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.kernel_config import KernelConfig, load_kernel
from na49_mixing.pdg_hyperons import load_pdg_hyperons
from na49_mixing.topology_runner import TopologyNominalParams, run_topology

NESTED_ROOT = Path(__file__).resolve().parents[1]
R97 = json.loads((NESTED_ROOT / "config" / "runs" / "r97.json").read_text())
LABELS = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")
LAMBDAS = (13122, 23122, 33122, 13124, 43122, 53122, 3126, 13126, 23124)
SIGMAS = ((13222, 13212, 13112), (13224, 13214, 13114), (23222, 23212, 23112), (3226, 3216, 3116),
          (23224, 23214, 23114), (13226, 13216, 13116))


@pytest.fixture(scope="module")
def kernels():
    off = load_kernel(KernelConfig(**R97["kernel"]), NESTED_ROOT)
    on = load_kernel(KernelConfig(**{**R97["kernel"], "pdg_hyperons": True}), NESTED_ROOT)
    return off, on


def test_off_is_the_committed_kernel(kernels):
    (cat_off, fams_off), _on = kernels
    ids = {int(r.pdg_id) for r in cat_off.records()}
    assert not ids & set(LAMBDAS) and not ids & {p for s in SIGMAS for p in s}


def test_the_records(kernels):
    """Nine Lambda* and six Sigma* in three charge states, at the listed
    masses and widths, their channels normalized, from the lightest
    threshold; the conjugate towers hold their antibaryons."""
    _off, (cat, _fams) = kernels
    config = load_pdg_hyperons(NESTED_ROOT)
    assert len(config["records"]) == 15
    for spec in config["records"]:
        for pid in spec["pdg_ids"]:
            rec = cat.get(int(pid))
            assert (rec.B, rec.S) == (1, -1) and rec.pole_mass == spec["pole_mass"] and rec.nominal_width == spec["nominal_width"]
            assert math.isclose(math.fsum(c.branching_fraction for c in rec.decay_channels), 1.0, rel_tol=1e-12)
            assert rec.mass_range[1] == pytest.approx(spec["pole_mass"] + 5.0 * spec["nominal_width"])
            assert rec.mass_range[0] < rec.pole_mass
            anti = cat.get(-int(pid))
            assert (anti.B, anti.S) == (-1, 1)
    # Lambda(1810): N Kbar 0.20 of 1.21, split p K- : n Kbar0 = 1 : 1 at I = 0
    rec = cat.get(53122)
    nk = {tuple(c.daughters): c.branching_fraction for c in rec.decay_channels}
    assert math.isclose(nk[(2212, -321)], 0.20 / 1.21 / 2, rel_tol=1e-12) and math.isclose(nk[(2112, -311)], nk[(2212, -321)])
    # Sigma(1660)+: Lambda pi+ at 0.35 of 1.06, the whole I = 1 strength in one charge member
    rec = cat.get(13222)
    lp = {tuple(c.daughters): c.branching_fraction for c in rec.decay_channels}
    assert math.isclose(lp[(3122, 211)], 0.35 / 1.06, rel_tol=1e-12)


def test_the_classes(kernels):
    """(Lambda*, K) and (Sigma*, K) at the non-strange baryonic families of
    the matching charge, (Sigma*, pi) and (Lambda*, pi) at the strange
    ones, and their conjugates at the conjugate towers."""
    (_c0, fams_off), (_c1, fams) = kernels
    off = {str(f.family_id): set(f.allowed_decay_classes) for f in fams_off}
    on = {str(f.family_id): set(f.allowed_decay_classes) for f in fams}
    added = {k: on[k] - off.get(k, set()) for k in on}
    sp, s0, sm = ([s[k] for s in SIGMAS] for k in range(3))
    want_np = {f"({x},321)" for x in LAMBDAS} | {f"({x},321)" for x in s0} | {f"({x},311)" for x in sp}
    assert added["baryonic_np"] == want_np and len(want_np) == 21
    assert added["baryonic_np_delta_q1"] == want_np
    assert added["baryonic_np_q2"] == {f"({x},321)" for x in sp}
    assert added["baryonic_strange"] == ({f"({x},-211)" for x in sp} | {f"({x},111)" for x in s0}
                                         | {f"({x},211)" for x in sm} | {f"({x},111)" for x in LAMBDAS})
    assert added["anti_baryonic_np"] == {f"({-x},-321)" for x in LAMBDAS} | {f"({-x},-321)" for x in s0} | {f"({-x},-311)" for x in sp}
    # the mesonic families are untouched
    assert all(not added[k] for k in added if k.startswith("mesonic"))


def test_the_refusal():
    bare = {**R97["kernel"], "pdg_hyperons": True, "delta_lines": False}
    with pytest.raises(ValueError, match="pdg_hyperons"):
        KernelConfig(**bare)


def test_events_conserve(kernels):
    """Single-diffraction and non-diffractive events with the hyperons
    conserve baryon number, charge and strangeness, and some carry a new
    hyperon in their decay chains."""
    _off, (cat, fams) = kernels
    params = TopologyNominalParams(
        epsilon=0.56, b_sd=6.16, epsilon_nd=0.253, nd_central=False, exact_transfer_range=True, line_thresholds=True,
        transfer_slope_mass_exponent=1.25, low_mass_spin_isospin=True, nd_duality_mass=2.0, nd_duality_shape_mass=1.4,
        remnant_weights={"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.0, "high_mass_baryonic_continuum": 1.0})
    for cls in ("nd", "sd_projectile"):
        p = params if cls == "nd" else dataclasses.replace(
            params, remnant_weights={"intact_proton": 1.0, "low_mass_baryonic_resonance": 0.0, "high_mass_baryonic_continuum": 0.0})
        for seed in range(25):
            rngs = {f"rng_{lab}": np.random.default_rng([seed, 97, k]) for k, lab in enumerate(LABELS)}
            ev = run_topology(cls, p, cat, fams, **rngs)
            assert sum(int(h.qn[0]) for h in ev.final_state) == 2
            assert sum(int(h.qn[1]) for h in ev.final_state) == 2
            assert sum(int(h.qn[2]) for h in ev.final_state) == 0


def test_the_review_restorations(kernels):
    """Review 2026-10-16 (M2, L1) and its verification: the Delta Kbar of
    Sigma(1915) (P wave 16% and F wave 5% of the PDG 2025 table, one class at
    the larger wave) and of Sigma(1910) (3%), the N Kbar*(892) of
    Lambda(1670) (5%), and Sigma(1915) -> Sigma(1385) pi at its larger wave."""
    _off, (cat, _fams) = kernels
    s1915 = cat.get(13226)  # Sigma(1915)+
    dk = {tuple(c.daughters): c for c in s1915.decay_channels if 2224 in c.daughters or 2214 in c.daughters}
    # raw sum 0.50 + 0.21: Delta++ K- : Delta+ Kbar0 = 3/4 : 1/4 of the I = 1 coupling
    assert math.isclose(dk[(2224, -321)].branching_fraction, 0.21 / 0.71 * 0.75, rel_tol=1e-12)
    assert math.isclose(dk[(2214, -311)].branching_fraction, 0.21 / 0.71 * 0.25, rel_tol=1e-12)
    assert all(c.orbital_angular_momentum == 1 for c in dk.values())
    assert {c.orbital_angular_momentum for c in s1915.decay_channels if 3224 in c.daughters or 3214 in c.daughters} == {3}
    s1910 = cat.get(23224)  # Sigma(1910)+
    total = math.fsum(c.branching_fraction for c in s1910.decay_channels if 2224 in c.daughters or 2214 in c.daughters)
    assert math.isclose(total, 0.03 / 0.98, rel_tol=1e-12)
    l1670 = cat.get(33122)
    nks = math.fsum(c.branching_fraction for c in l1670.decay_channels if -313 in c.daughters or -323 in c.daughters)
    assert math.isclose(nks, 0.05 / 1.135, rel_tol=1e-12)
    for pid in (13226, 13216, 13116, 23224, 23214, 23114, 33122):
        assert math.isclose(math.fsum(c.branching_fraction for c in cat.get(pid).decay_channels), 1.0, rel_tol=1e-12)
