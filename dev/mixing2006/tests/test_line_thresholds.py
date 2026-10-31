"""The per-line lower mass edges of the excited lines (2026-10-06;
TASK-20261005-cascade-vertex-rules item 1, the code and physics review's
M-6): under ``TopologyNominalParams.line_thresholds`` each single-,
double- and non-diffractive excited line starts its mass law at its root
family's lowest direct threshold (p pi0 for Q = 1, n pi0 for Q = 0, p pi+
for Q = 2), so no root ends unidentified below its open channels; the
reweighting normalizes on the recorded edges (unit mean); the default draws
and arrays are unchanged."""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.cascade_decay import _daughter_pole_mass
from na49_mixing.isovector_exchange import LineCharge, line_mass_threshold
from na49_mixing.kernel_config import R74, load_kernel
from na49_mixing.rng_substreams import derive_substreams
from na49_mixing.topology_mass_laws import EXCITATION_M_MIN, log_density_ratio, log_dd_joint_mass
from na49_mixing.topology_runner import NOMINAL_PARAMS, run_topology

NESTED_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NESTED_ROOT / "helpers" / "benchmarking"))
import run_observables as OB  # noqa: E402

LABELS = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")
ND_WEIGHTS = {"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.25, "high_mass_baryonic_continuum": 0.75}
#: the R75 kernel (the isovector lines need the baryonic completion and the Delta-like roots)
R75_KERNEL = dataclasses.replace(R74, isospin_completion=("baryonic", "strange_baryonic", "strange_mesonic"),
                                 delta_lines=True)


@pytest.fixture(scope="module")
def kernel():
    return load_kernel(R75_KERNEL, NESTED_ROOT)


def _rngs(i: int):
    subs = derive_substreams(20261006, 37, 9_400_000_000 + i)
    return {f"rng_{lab}": np.random.default_rng(subs.stream_for(lab)) for lab in LABELS}


def _params(cls, on, f_iv=0.55, eps=0.3, b=3.0, eps_nd=0.2):
    p = dataclasses.replace(NOMINAL_PARAMS, epsilon=eps, b_sd=b, epsilon_nd=eps_nd, nd_central=False,
                            isovector_fraction=f_iv, exact_transfer_range=True, line_thresholds=on)
    if cls == "nd0":
        p = dataclasses.replace(p, remnant_weights=dict(ND_WEIGHTS))
    return p


def test_the_line_thresholds(kernel):
    catalog, fams = kernel
    m = lambda pdg: _daughter_pole_mass(catalog, pdg)  # noqa: E731
    assert line_mass_threshold(fams, catalog) == pytest.approx(m(2212) + m(111), abs=1e-12)
    assert line_mass_threshold(fams, catalog, LineCharge(1, 1.5)) == pytest.approx(m(2212) + m(111), abs=1e-12)
    for isospin in (0.5, 1.5):
        assert line_mass_threshold(fams, catalog, LineCharge(0, isospin)) == pytest.approx(m(2112) + m(111), abs=1e-12)
    assert line_mass_threshold(fams, catalog, LineCharge(2, 1.5)) == pytest.approx(m(2212) + m(211), abs=1e-12)
    assert line_mass_threshold(fams, catalog) == pytest.approx(EXCITATION_M_MIN, abs=1e-6)


@pytest.mark.parametrize("cls", ["sd_projectile", "sd_target", "dd", "nd0"])
def test_no_root_below_its_threshold(kernel, cls):
    catalog, fams = kernel
    edges_seen = set()
    for i in range(150):
        ev = run_topology("nd" if cls == "nd0" else cls, _params(cls, True), catalog, fams, **_rngs(i))
        assert all(h.pdg_id is not None for h in ev.final_state)  # nothing unidentified
        if cls.startswith("sd"):
            assert ev.excitation_mass >= ev.excitation_m_min
            edges_seen.add(round(ev.excitation_m_min, 6))
        elif cls == "dd":
            assert ev.m1 >= ev.line_m_min[0] and ev.m2 >= ev.line_m_min[1]
            edges_seen.update(round(e, 6) for e in ev.line_m_min)
        else:
            for remnant, (lo, _hi) in zip(ev.remnants, ev.remnant_windows):
                edges_seen.add(round(lo, 6))
                if remnant.kind == "high_mass_baryonic_continuum":
                    assert remnant.mass >= lo
    assert len(edges_seen) >= 2  # the charge-exchanged lines carry their own edges


@pytest.mark.parametrize("cls", ["sd_projectile", "dd"])
def test_the_default_is_unchanged(kernel, cls):
    catalog, fams = kernel
    for i in range(20):
        off = run_topology(cls, _params(cls, False), catalog, fams, **_rngs(500 + i))
        arrays = OB.shape_arrays(off)
        assert not {"m2_low", "m_lo1", "m_lo2"} & set(arrays)
        assert (off.excitation_m_min if cls.startswith("sd") else off.line_m_min) is None


def test_the_joint_mass_with_line_edges():
    sqrt_s = 17.270
    assert log_dd_joint_mass(0.3, sqrt_s, EXCITATION_M_MIN, EXCITATION_M_MIN) == log_dd_joint_mass(0.3, sqrt_s)
    for eps, lo1, lo2 in ((0.3, 1.0745, 1.0778), (0.0, 1.0778, 1.0732), (-0.2, 1.0778, 1.0778)):
        # the reduction against a direct two-dimensional quadrature in (M1, M2)
        m1 = np.linspace(lo1, sqrt_s - lo2, 4001)
        top = np.maximum(sqrt_s - m1, lo2)
        if eps == 0.0:
            inner = 2.0 * np.log(top / lo2)
        else:
            inner = (top ** (-2 * eps) - lo2 ** (-2 * eps)) / (-eps)
        direct = np.trapz(2.0 * m1 ** (-1.0 - 2.0 * eps) * inner, m1)
        assert log_dd_joint_mass(eps, sqrt_s, lo1, lo2) == pytest.approx(math.log(direct), abs=2e-6)


@pytest.mark.parametrize("cls", ["sd_projectile", "dd", "nd0"])
def test_the_reweighting_has_unit_mean(kernel, cls):
    catalog, fams = kernel
    arrays = {}
    for i in range(400):
        ev = run_topology("nd" if cls == "nd0" else cls, _params(cls, True), catalog, fams, **_rngs(10_000 + i))
        for k, v in OB.shape_arrays(ev).items():
            arrays.setdefault(k, []).append(v)
    arrays = {k: np.asarray(v, dtype=float) for k, v in arrays.items()}
    law = "nd" if cls == "nd0" else cls
    eps0 = 0.2 if cls == "nd0" else 0.3
    for eps, b in ((eps0, 2.0), (eps0 + 0.25, 3.0), (eps0 - 0.15, 4.0)):
        w = np.exp(log_density_ratio(law, arrays, eps, b, eps0=eps0, b_sd0=3.0, sqrt_s=17.270, exact_range=True))
        assert np.mean(w) == pytest.approx(1.0, abs=5.0 * np.std(w) / math.sqrt(w.size) + 0.01)
