"""The PDG 2024 meson tables (``KernelConfig.pdg_meson_modes``) and the
N* range rule (``KernelConfig.nstar_range_rule``), 2026-10-16, the review
of R99 (H1, H2, M3, M6) and its verification."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.isoscalar_dipion import _pdg_range, load_isoscalar_dipion, range_position_fractions
from na49_mixing.kernel_config import KernelConfig, load_kernel
from na49_mixing.pdg_database import get_particle
from na49_mixing.resonance_decay import isospin_closure_report
from na49_mixing.topology_runner import TopologyNominalParams, run_topology

NESTED_ROOT = Path(__file__).resolve().parents[1]
R99 = json.loads((NESTED_ROOT / "config" / "runs" / "r99.json").read_text())
LABELS = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")
REVISED = (113, 223, 333, 225, 115, 215, -215, 10323, -10323, 10313, -10313, 325, -325, 315, -315)


@pytest.fixture(scope="module")
def kernels():
    off = load_kernel(KernelConfig(**R99["kernel"]), NESTED_ROOT)
    on = load_kernel(KernelConfig(**{**R99["kernel"], "pdg_meson_modes": True}), NESTED_ROOT)
    return off, on


def _bf(record, *daughters):
    return math.fsum(c.branching_fraction for c in record.decay_channels if tuple(c.daughters) == tuple(daughters))


def test_off_is_the_committed_catalog(kernels):
    (cat_off, _f0), (cat_on, _f1) = kernels
    for r in cat_off.records():
        if int(r.pdg_id) not in REVISED:
            assert cat_on.get(int(r.pdg_id)) == r


def test_the_tables(kernels):
    """The f2(1270) with its 4pi modes as rho pi pi, the a2(1320) with omega
    pi pi and K Kbar, the K1(1270) with K pi pi for K0*(1430) pi and
    K f0(1370), the K2*(1430) with K* pi pi, the PDG 2024 widths; every
    table sums to one and the catalog validates and passes the isospin
    audit."""
    _off, (cat, _fams) = kernels
    assert cat.validate().errors == ()
    assert isospin_closure_report(cat).errors == ()
    f2 = cat.get(225)
    assert math.isclose(_bf(f2, 113, 211, -211), 0.028 / 0.994, rel_tol=1e-12)
    assert math.isclose(_bf(f2, 213, -211, 111) + _bf(f2, -213, 211, 111), 0.077 / 0.994, rel_tol=1e-12)
    assert math.isclose(_bf(f2, 211, -211) + _bf(f2, 111, 111), 0.843 / 0.994, rel_tol=1e-12)
    a2 = cat.get(115)
    assert math.isclose(_bf(a2, 223, 211, -211), 0.106 / 1.001, rel_tol=1e-12)
    assert math.isclose(_bf(a2, 321, -321) + _bf(a2, 311, -311), 0.049 / 1.001, rel_tol=1e-12)
    k1 = cat.get(10323)
    assert math.isclose(_bf(k1, 321, 211, -211) + _bf(k1, 311, 211, 111) + _bf(k1, 321, 111, 111), 0.31 / 1.01, rel_tol=1e-12)
    k2 = cat.get(315)
    assert math.isclose(_bf(k2, 313, 211, -211) + _bf(k2, 313, 111, 111), 0.134 / 0.996, rel_tol=1e-12)
    assert (cat.get(113).nominal_width, cat.get(223).nominal_width, cat.get(325).nominal_width,
            cat.get(315).nominal_width) == (0.1474, 0.00868, 0.1, 0.109)
    for pid in REVISED:
        r = cat.get(pid)
        assert math.isclose(math.fsum(c.branching_fraction for c in r.decay_channels), 1.0, rel_tol=1e-12)
        entry = get_particle(abs(pid))
        assert entry is None or abs(r.pole_mass - entry.mass) < 1e-6
        assert all(len(c.daughters) == (3 if c.is_three_body else 2) for c in r.decay_channels)


def test_the_conjugates(kernels):
    """The strange mesons' antiparticles decay as the charge conjugates."""
    _off, (cat, _fams) = kernels
    for pid in (10323, 10313, 325, 315, 215):
        plus, minus = cat.get(pid), cat.get(-pid)
        assert len(plus.decay_channels) == len(minus.decay_channels)
        for a, b in zip(plus.decay_channels, minus.decay_channels):
            assert a.branching_fraction == b.branching_fraction and len(a.daughters) == len(b.daughters)


def test_events_conserve(kernels):
    _off, (cat, fams) = kernels
    params = TopologyNominalParams(
        epsilon=0.53, b_sd=6.13, epsilon_nd=0.249, nd_central=False, exact_transfer_range=True, line_thresholds=True,
        transfer_slope_mass_exponent=1.25, low_mass_spin_isospin=True, nd_duality_mass=2.0, nd_duality_shape_mass=1.4,
        remnant_weights={"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.0, "high_mass_baryonic_continuum": 1.0})
    for seed in range(25):
        rngs = {f"rng_{lab}": np.random.default_rng([seed, 1024, k]) for k, lab in enumerate(LABELS)}
        ev = run_topology("nd", params, cat, fams, **rngs)
        for i in range(3):
            assert sum(int(h.qn[i]) for h in ev.final_state) == (2, 2, 0)[i]


def test_the_range_rule():
    """Every N* mode at one common position inside its PDG range, the
    fractions summing to one; N(1650) -> N pi at 0.507 against the 0.40 of
    the normalized midpoints and its listed 50-70%."""
    config = load_isoscalar_dipion(NESTED_ROOT)
    for multiplet in config["nstar"]:
        fractions = range_position_fractions(multiplet["channels"])
        assert math.isclose(math.fsum(fractions), 1.0, rel_tol=1e-12)
        for mode, value in zip(multiplet["channels"], fractions):
            lo, hi = _pdg_range(mode["pdg_range"])
            assert lo - 1e-12 <= value <= hi + 1e-12, (multiplet["multiplet"], mode["kind"], value)
    n1650 = next(m for m in config["nstar"] if m["multiplet"].startswith("N(1650)"))
    npi = range_position_fractions(n1650["channels"])[0]
    assert n1650["channels"][0]["kind"] == "Npi" and math.isclose(npi, 0.5074, abs_tol=5e-4)
    on = load_kernel(KernelConfig(**{**R99["kernel"], "nstar_range_rule": True}), NESTED_ROOT)[0]
    total_npi = math.fsum(c.branching_fraction for c in on.get(122212).decay_channels
                          if sorted(abs(d) for d in c.daughters) in ([111, 2212], [211, 2112]))
    assert math.isclose(total_npi, npi, rel_tol=1e-12)
    with pytest.raises(ValueError, match="nstar_range_rule"):
        KernelConfig(**{**R99["kernel"], "nstar_range_rule": True, "isoscalar_dipion": False})


def test_the_r100_spec_is_r99_with_the_tables():
    """R100: the R99 contract with the PDG 2024 meson tables and the N*
    range rule, generated at the R99 optimum with every free coordinate
    from the fit."""
    runs = NESTED_ROOT / "config" / "runs"
    r100 = json.loads((runs / "r100.json").read_text())
    assert r100["kernel"] == {**R99["kernel"], "pdg_meson_modes": True, "nstar_range_rule": True}
    assert r100["topology"] == R99["topology"]
    for block in ("fit", "diffractive_split", "isovector", "fixed", "statistics"):
        assert r100[block] == R99[block]
    g0 = r100["generation_0"]
    assert g0["run"] == "r99_xi_review_2026-10-16" and "judgment" not in g0
    assert r100["seeds"]["raw_first"] > R99["seeds"]["raw_first"] + 511
    branch = NESTED_ROOT.parent / "00runs" / "r99_xi_review_2026-10-16" / "branch_report_f0.json"
    if branch.is_file():
        fitted = json.loads(branch.read_text())["fitted"]
        gen0 = {p["name"]: p["gen0"] for p in r100["parameters"]}
        for name in g0["from_fit"]:
            assert float(gen0[name]) == float(fitted[name]), name
