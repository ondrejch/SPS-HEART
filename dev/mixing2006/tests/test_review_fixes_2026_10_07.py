"""The fixes of the 2026-10-06 physics reviews (Muse, Grok, Fledge; triage
`_knowledge/operations/physics-review-triage-2026-10-06.md`), 2026-10-07:
the negative-slope transfer normalizations, the nd_multi refusal of the
topology options, the exact-window solve of ``sd_kinematics``, the exact
CM sqrt(s) of the observables, the bin-center factors, and the run-spec
checks of the isovector block and the bin-center option."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from na49_mixing.rng_substreams import derive_substreams
from na49_mixing.topology_mass_laws import PINNED_SQRT_S_EXACT, log_t_norm_vec, log_t_window_norm_vec, pp_t_window
from na49_mixing.topology_runner import NOMINAL_PARAMS, run_topology
from na49_mixing.topology_single_diffraction import sd_kinematics

NESTED_ROOT = Path(__file__).resolve().parents[1]
HELPERS = NESTED_ROOT / "helpers" / "benchmarking"
if str(HELPERS) not in sys.path:
    sys.path.insert(0, str(HELPERS))
import run_observables as OB  # noqa: E402
import sps_core  # noqa: E402

ENGINE = HELPERS / "run_engine.py"
R77 = NESTED_ROOT / "config" / "runs" / "r77.json"
LABELS = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")


@pytest.mark.parametrize("b", [-3.0, -1.0, -0.2, 0.5, 2.0, 7.5])
def test_the_transfer_normalizations_for_either_slope_sign(b):
    lo, hi = -4.0, -0.3
    window = float(log_t_window_norm_vec(b, np.array([lo]), np.array([hi]))[0])
    full = float(log_t_norm_vec(b, np.array([lo]))[0])
    assert window == pytest.approx(math.log(quad(lambda t: math.exp(b * t), lo, hi)[0]), abs=1e-12)
    assert full == pytest.approx(math.log(quad(lambda t: math.exp(b * t), lo, 0.0)[0]), abs=1e-12)


def test_nd_multi_refuses_the_topology_options():
    import dataclasses
    params = dataclasses.replace(NOMINAL_PARAMS, exact_transfer_range=True)
    subs = derive_substreams(20261007, 0, 0)
    rngs = {f"rng_{lab}": np.random.default_rng(subs.stream_for(lab)) for lab in LABELS}
    with pytest.raises(ValueError, match="nd_multi"):
        run_topology("nd_multi", params, None, None, **rngs)


def test_sd_kinematics_on_the_exact_window():
    from na49_mixing.topology_elastic import PROTON_MASS
    m, m_rem = 4.0, PROTON_MASS
    t_lo, t_hi = pp_t_window(m * m, m_rem * m_rem)
    for t in (t_lo, 0.5 * (t_lo + t_hi), t_hi):
        x, r = sd_kinematics(m, t, m_rem, 17.270, exact_range=True)
        total = [a + b for a, b in zip(x, r)]
        e_beam = math.sqrt(m_rem * m_rem + 158.0 ** 2)
        assert total[0] == pytest.approx(e_beam + m_rem, rel=1e-12) and total[3] == pytest.approx(158.0, rel=1e-12)
        assert x[0] ** 2 - x[1] ** 2 - x[2] ** 2 - x[3] ** 2 == pytest.approx(m * m, rel=1e-9)
    with pytest.raises(ValueError, match="reachable window"):
        sd_kinematics(m, t_lo - 0.5, m_rem, 17.270, exact_range=True)
    # the default keeps the committed incoming system
    assert sd_kinematics(m, -1.0, m_rem, 17.270) == sd_kinematics(m, -1.0, m_rem, 17.270, exact_range=False)


def test_the_observables_boost_is_one_lorentz_boost():
    beta, gamma = OB.cm_boost()
    assert OB.SQRT_S_CM == PINNED_SQRT_S_EXACT
    assert gamma == pytest.approx(1.0 / math.sqrt(1.0 - beta * beta), rel=1e-13)
    # the beam proton and the target proton are back to back with sqrt(s)/2 each
    m = OB.PROTON_MASS
    e_cm, _px, _py, _pz = OB.to_cm((math.sqrt(m * m + OB.BEAM_MOMENTUM ** 2), 0.0, 0.0, OB.BEAM_MOMENTUM), beta, gamma)
    assert e_cm == pytest.approx(0.5 * OB.SQRT_S_CM, rel=1e-13)


def test_the_one_dimensional_bin_center_factors():
    centers = np.array([0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8])
    edges = sps_core.edges_from_centers(centers)

    def exact(f):
        return np.array([f(c) / (quad(f, a, b)[0] / (b - a)) for c, a, b in zip(centers, edges[:-1], edges[1:])])
    # log-linear: exact everywhere (the end points use the tangent)
    lin = lambda x: math.exp(-6.0 * x)
    assert OB.bin_center_factors_1d(centers, [lin(c) for c in centers], edges) == pytest.approx(exact(lin), rel=1e-12)
    # log-quadratic: exact at the interior points; the end points drop the curvature
    gau = lambda x: math.exp(-3.0 * x - 4.0 * x * x)
    fac = OB.bin_center_factors_1d(centers, [gau(c) for c in centers], edges)
    assert fac[1:-1] == pytest.approx(exact(gau)[1:-1], rel=1e-12)
    assert np.all(np.abs(fac[[0, -1]] / exact(gau)[[0, -1]] - 1.0) < 0.02)


def test_the_two_dimensional_bin_center_factors(tmp_path):
    """A separable log-quadratic f(x_F, p_T) on an irregular grid: the
    interior factors are exact, and the asymmetric midpoint-edge bins carry
    the expected shift."""
    xs = np.array([0.0, 0.05, 0.1, 0.2, 0.35, 0.45, 0.6])
    ps = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    gx = lambda x: -2.0 * x - 3.0 * x * x
    gp = lambda p: -5.0 * p + 1.5 * p * p
    rows = [f"{x} {p} {math.exp(gx(x) + gp(p))} {0.01 * math.exp(gx(x) + gp(p))}" for x in xs for p in ps]
    path = tmp_path / "synthetic.dat"
    path.write_text("\n".join(rows) + "\n")
    from na49_mixing.staged_evaluation import BinnedFitTable
    table = BinnedFitTable.from_path(path, species="pi_minus", pdg_id=-211, mass_gev=0.1396, normalization="f_over_sigma_inel",
                                     sigma_inel_mb=31.78, window={"xf_min": 0.0, "xf_max": 1.0, "pt_min": 0.0, "pt_max": 2.0})
    fac = OB.bin_center_factors(table)
    xe, pe = sps_core.edges_from_centers(table.xf_centers), sps_core.edges_from_centers(table.pt_centers)
    for k in range(fac.size):
        ix, ip = int(table.bin_index_xf[k]), int(table.bin_index_pt[k])
        if ix in (0, xs.size - 1) or ip in (0, ps.size - 1):
            continue
        x0, p0 = table.xf_centers[ix], table.pt_centers[ip]
        mx = quad(lambda x: math.exp(gx(x) - gx(x0)), xe[ix], xe[ix + 1])[0] / (xe[ix + 1] - xe[ix])
        mp = quad(lambda p: math.exp(gp(p) - gp(p0)), pe[ip], pe[ip + 1])[0] / (pe[ip + 1] - pe[ip])
        assert fac[k] == pytest.approx(1.0 / (mx * mp), rel=1e-10)


def _r77_spec(tmp_path, **fit_or_iso):
    spec = json.loads(R77.read_text())
    for key, value in fit_or_iso.items():
        block, name = key.split("__")
        spec[block][name] = value
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return path


def _engine(name):
    spec = importlib.util.spec_from_file_location(name, ENGINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_run_spec_checks(tmp_path):
    e = _engine("run_engine_review_fixes")
    with pytest.raises(ValueError, match="isovector block"):
        e.configure(_r77_spec(tmp_path, isovector__delta_ratio=2.0))
    with pytest.raises(ValueError, match="bin_center_correction"):
        e.configure(_r77_spec(tmp_path, fit__bin_center_correction="yes"))
    e.configure(_r77_spec(tmp_path, fit__bin_center_correction=True))
    assert e.BIN_CENTER is True
    tables = e.load_fit_tables()
    name = "proton"
    plain = math.pi * e.SQRT_S_CM * OB.bin_areas(tables[name])
    assert e.species_divisor(tables[name]) == pytest.approx(plain / OB.bin_center_factors(tables[name]), rel=1e-14)
    e.configure(R77)
    assert e.BIN_CENTER is False and e.species_divisor(tables[name]) == pytest.approx(plain, rel=1e-14)


def test_the_recoiling_central_solve_closes_and_decouples_the_lines():
    """Stage B (2026-10-07): the recoiling three-system solve conserves the
    CM four-momentum exactly, puts the central system at rapidity 0 with the
    transverse recoil, and keeps each line's transverse momentum at its own
    transfer; beyond the transverse closure both k scale onto it."""
    from na49_mixing.topology_non_diffractive import _cm_triplet_recoil
    from na49_mixing.topology_mass_laws import PINNED_SQRT_S_EXACT as SQ
    for (m1, t1, m2, t2, mc, phi) in ((2.0, -0.3, 3.0, -0.6, 1.5, 1.0), (6.0, -1.2, 1.5, -0.1, 0.4, 2.5),
                                       (13.0, -4.0, 2.0, -3.0, 1.0, 0.3)):
        s1, s2, c = _cm_triplet_recoil(m1, t1, m2, t2, mc, phi)
        tot = [a + b + d for a, b, d in zip(s1, s2, c)]
        assert tot[0] == pytest.approx(SQ, rel=1e-12) and max(abs(x) for x in tot[1:]) < 1e-9
        assert c[3] == 0.0 and s1[3] >= -1e-12 and s2[3] <= 1e-12
        for (sys_, m) in ((s1, m1), (s2, m2), (c, mc)):
            assert sys_[0] ** 2 - sys_[1] ** 2 - sys_[2] ** 2 - sys_[3] ** 2 == pytest.approx(m * m, rel=1e-9)
    s1, s2, c = _cm_triplet_recoil(2.0, -0.3, 3.0, -0.6, 1.5, 1.0)
    assert math.hypot(s1[1], s1[2]) == pytest.approx(math.sqrt(0.3), rel=1e-12)
    assert math.hypot(s2[1], s2[2]) == pytest.approx(math.sqrt(0.6), rel=1e-12)
    # a transverse closure beyond reach scales the kicks onto it
    s1, s2, c = _cm_triplet_recoil(13.0, -40.0, 2.0, -30.0, 1.0, 0.3)
    assert math.hypot(s1[1], s1[2]) < math.sqrt(40.0)
    assert s1[0] + s2[0] + c[0] == pytest.approx(SQ, rel=1e-12)


def test_central_recoil_is_off_by_default_and_needs_the_central_source():
    import dataclasses
    from na49_mixing.topology_runner import TopologyNominalParams
    assert TopologyNominalParams().central_recoil is False
    params = dataclasses.replace(NOMINAL_PARAMS, central_recoil=True)
    subs = derive_substreams(20261007, 0, 7)
    rngs = {f"rng_{lab}": np.random.default_rng(subs.stream_for(lab)) for lab in LABELS}
    from na49_mixing.topology_runner import _committed_surfaces
    catalog, fams = _committed_surfaces()
    ev = run_topology("nd", params, catalog, fams, **rngs)
    tot = np.sum([h.four_momentum for h in ev.final_state], axis=0)
    assert abs(tot[1]) < 1e-7 and abs(tot[2]) < 1e-7
