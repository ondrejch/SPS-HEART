"""The exact transfer window of the 2 -> 2 topology classes (2026-10-05;
TASK-20261005-kernel-nd-dd-transfer-range, the code and physics review's
H-1): under ``TopologyNominalParams.exact_transfer_range`` the single- and
double-diffractive and the two-component non-diffractive events draw their
transfer on the reachable window of the realized pair, so no production
angle is clamped and the excited line carries the transfer's p_T; the
reweighting stays exact (unit mean); the class with the central source
refuses the option; the default draws are unchanged."""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.kernel_config import R74, load_kernel
from na49_mixing.rng_substreams import derive_substreams
from na49_mixing.topology_mass_laws import (
    PINNED_SQRT_S_EXACT,
    PROTON_MASS,
    MassLawParams,
    draw_t_window,
    log_density_ratio,
    log_t_window_norm_vec,
    pp_t_window,
)
from na49_mixing.topology_runner import NOMINAL_PARAMS, run_topology

NESTED_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NESTED_ROOT / "helpers" / "benchmarking"))
import run_observables as OB  # noqa: E402

LABELS = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")
ND_WEIGHTS = {"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.25, "high_mass_baryonic_continuum": 0.75}


@pytest.fixture(scope="module")
def kernel():
    return load_kernel(R74, NESTED_ROOT)


def _rngs(i: int):
    subs = derive_substreams(20261005, 31, 9_300_000_000 + i)
    return {f"rng_{lab}": np.random.default_rng(subs.stream_for(lab)) for lab in LABELS}


def _params(cls, exact, eps=0.3, b=3.0, eps_nd=0.2):
    p = dataclasses.replace(NOMINAL_PARAMS, epsilon=eps, b_sd=b, epsilon_nd=eps_nd, exact_transfer_range=exact,
                            nd_central=(cls != "nd0"))
    if cls in ("nd", "nd0"):
        p = dataclasses.replace(p, remnant_weights=dict(ND_WEIGHTS))
    return p


def test_the_window():
    mp2 = PROTON_MASS * PROTON_MASS
    lo, hi = pp_t_window(mp2, mp2)
    assert hi == pytest.approx(0.0, abs=1e-9) and lo == pytest.approx(4 * mp2 - PINNED_SQRT_S_EXACT ** 2, rel=1e-9)
    previous = 0.0
    for m in (1.2, 2.0, 4.0, 8.0, 12.0):
        lo, hi = pp_t_window(m * m, mp2)
        assert lo < hi < previous  # the minimum |t| grows with the excitation mass
        previous = hi
    rng = np.random.default_rng(3)
    params = MassLawParams(0.3, 3.0)
    lo, hi = pp_t_window(64.0, mp2)
    ts = np.array([draw_t_window(lo, hi, params, rng) for _ in range(40_000)])
    assert ts.min() >= lo and ts.max() <= hi
    mean = hi - 1.0 / 3.0 + (hi - lo) * math.exp(3.0 * (lo - hi)) / (1.0 - math.exp(3.0 * (lo - hi)))
    assert ts.mean() == pytest.approx(mean, abs=4 * 0.33 / math.sqrt(ts.size))
    assert log_t_window_norm_vec(3.0, [lo], [hi])[0] == pytest.approx(math.log((math.exp(3 * hi) - math.exp(3 * lo)) / 3.0))


@pytest.mark.parametrize("cls", ["sd_projectile", "sd_target", "dd", "nd0"])
def test_no_clamped_line(kernel, cls):
    catalog, fams = kernel
    clamped, n = 0, 120
    for i in range(n):
        ev = run_topology("nd" if cls == "nd0" else cls, _params(cls, True), catalog, fams, **_rngs(i))
        line = ev.lines[0]
        clamped += math.hypot(line.four_momentum[1], line.four_momentum[2]) == 0.0
        assert tuple(sum(int(h.qn[j]) for h in ev.final_state) for j in range(3)) == (2, 2, 0)
        if cls in ("dd", "nd0"):
            assert ev.t1 == ev.t2  # one transfer of the 2 -> 2 pair
    assert clamped == 0


def test_the_committed_draw_clamps(kernel):
    """The motivation, kept as a measured fact of the committed draw."""
    catalog, fams = kernel
    zero = sum(math.hypot(*run_topology("nd", _params("nd0", False), catalog, fams, **_rngs(i)).lines[0].four_momentum[1:3]) == 0.0
               for i in range(60))
    assert zero > 10


def test_the_central_class_refuses_the_option(kernel):
    catalog, fams = kernel
    with pytest.raises(ValueError):
        run_topology("nd", _params("nd", True), catalog, fams, **_rngs(0))


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
    for eps, b in ((eps0, 2.0), (eps0, 4.5), (eps0 + 0.15, 3.0)):
        w = np.exp(log_density_ratio(law, arrays, eps, b, eps0=eps0, b_sd0=3.0, sqrt_s=17.270, exact_range=True))
        assert np.mean(w) == pytest.approx(1.0, abs=5.0 * np.std(w) / math.sqrt(w.size) + 0.01)
