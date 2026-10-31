"""The three kernel fixes of the feed-down and feed-over review (2026-10-02;
``_knowledge/evidence/feed-down-feed-over-review-2026-10-02.md``).

1. The target single-diffraction class is the center-of-mass mirror of the
   projectile construction (it had been a copy of the projectile class).
2. The single- and double-diffraction excitation mass laws start at the
   N pi threshold ``m_p + m_pi0`` (they started at ``m_p``, and up to 57%
   of the excitations stayed unidentified below the threshold); the
   reweighting normalizations share the new edge.
3. The radiative modes of eta, eta-prime, omega and phi are explicit
   channels with photons (they had been folded into hadronic channels).
"""

from __future__ import annotations

import collections
import math

import numpy as np
import pytest
from scipy.integrate import quad

from na49_mixing import topology_mass_laws as ML
from na49_mixing.cascade_ablation import _committed_catalog, _committed_fams
from na49_mixing.hadron_catalog import NON_HADRONIC_FINAL_STATES, PHOTON_PDG, STABLE_STATES
from na49_mixing.resonance_decay import ResonanceDecayer, isospin_derivation_plan, three_body_phase_space
from na49_mixing.rng_substreams import derive_substreams
from na49_mixing.topology_runner import TopologyNominalParams, run_topology
from na49_mixing.topology_single_diffraction import _beam_configuration

CATALOG = _committed_catalog()
FAMS = _committed_fams()
SQRT_S = 17.27
LABELS = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")


def _rngs(seed, event):
    subs = derive_substreams(seed, 0, event)
    return {f"rng_{label}": np.random.default_rng(subs.stream_for(label)) for label in LABELS}


def _params(epsilon):
    return TopologyNominalParams(epsilon=epsilon)


def _to_cm(p4):
    _e_beam, _e_total, _sq, _ps, beta, gamma = _beam_configuration()
    e, px, py, pz = p4
    return (gamma * (e - beta * pz), px, py, gamma * (pz - beta * e))


# ----------------------------------------------------------------------
# 1. the target single diffraction is the CM mirror
# ----------------------------------------------------------------------
@pytest.mark.parametrize("epsilon", [0.0, 1.0])
def test_target_single_diffraction_is_the_cm_mirror(epsilon):
    for i in range(40):
        proj = run_topology("sd_projectile", _params(epsilon), CATALOG, FAMS, **_rngs(31, i))
        targ = run_topology("sd_target", _params(epsilon), CATALOG, FAMS, **_rngs(31, i))
        assert targ.excitation_mass == proj.excitation_mass and targ.t == proj.t
        assert len(targ.final_state) == len(proj.final_state)
        for a, b in zip(proj.final_state, targ.final_state):
            assert (a.pdg_id, a.qn, a.line) == (b.pdg_id, b.qn, b.line)
            ca, cb = _to_cm(a.four_momentum), _to_cm(b.four_momentum)
            scale = max(ca[0], 1e-9)
            assert abs(ca[0] - cb[0]) / scale < 1e-12 and abs(ca[1] - cb[1]) / scale < 1e-12
            assert abs(ca[2] - cb[2]) / scale < 1e-12 and abs(ca[3] + cb[3]) / scale < 1e-12
        assert targ.gap_size == pytest.approx(proj.gap_size, rel=1e-9, abs=1e-12)
        # the excited system of the target class sits backward in the CM
        exc_pz = math.fsum(_to_cm(h.four_momentum)[3] for h in targ.final_state if h.line == "excitation")
        assert exc_pz < 0.0


# ----------------------------------------------------------------------
# 2. the diffractive N pi threshold
# ----------------------------------------------------------------------
def test_excitation_edge_is_the_n_pi_threshold():
    assert ML.EXCITATION_M_MIN == pytest.approx(0.93827231 + 0.1349766, abs=1e-15)
    lo, hi = ML.sd_limits(ML.PROTON_MASS, SQRT_S)
    assert lo == ML.EXCITATION_M_MIN ** 2 and hi == (SQRT_S - ML.PROTON_MASS) ** 2
    assert ML.EXCITATION_M_MIN == pytest.approx(CATALOG.get(2214).mass_range[0], abs=1e-12)


@pytest.mark.parametrize("cls", ["sd_projectile", "sd_target", "dd"])
def test_no_diffractive_excitation_stays_unidentified(cls):
    """At the steep exponent of R68 to R70 (3.0) more than half of the
    excitations used to stay below the threshold as unidentified finals."""
    for i in range(150):
        event = run_topology(cls, _params(3.0), CATALOG, FAMS, **_rngs(37, i))
        assert all(h.pdg_id is not None for h in event.final_state)
        masses = [event.excitation_mass] if cls != "dd" else [event.m1, event.m2]
        assert min(masses) >= ML.EXCITATION_M_MIN - 1e-12


def test_dd_joint_mass_matches_quadrature():
    for eps in (0.0, 0.5, 2.0):
        a = ML.EXCITATION_M_MIN

        def outer(u):
            inner = quad(lambda y: y ** (-1.0 - eps), a * a, (SQRT_S - u) ** 2, limit=200)[0]
            return 2.0 * u * (u * u) ** (-1.0 - eps) * inner

        ref = quad(outer, a, SQRT_S - a, limit=400)[0]
        assert ML.log_dd_joint_mass(eps, SQRT_S) == pytest.approx(math.log(ref), abs=1e-7)


@pytest.mark.parametrize("cls", ["sd_projectile", "dd"])
def test_reweighting_stays_exact(cls):
    """Draws at (eps0, b0) reweighted to (eps, b): the mean weight is 1."""
    rng = np.random.default_rng(20261002)
    p0 = ML.MassLawParams(epsilon=1.0, b_sd=2.57)
    lo, hi = ML.sd_limits(ML.PROTON_MASS, SQRT_S)
    n = 12000
    if cls == "dd":
        draws = np.array([ML.draw_joint_dd(lo, hi, lo, hi, SQRT_S, p0, rng) for _ in range(n)])
        arrays = {"m12": draws[:, 0], "m22": draws[:, 1], "t1": draws[:, 2], "t2": draws[:, 3]}
    else:
        m2 = np.array([ML.draw_marginal(lo, hi, p0, 0.0, rng) for _ in range(n)])
        t = np.array([ML.draw_t(ML.t_range(v, ML.PROTON_MASS, SQRT_S)[0], p0, rng) for v in m2])
        arrays = {"m2": m2, "t": t, "m_rem": np.full(n, ML.PROTON_MASS)}
    for eps, b in ((2.0, 4.0), (0.5, 2.0)):
        w = np.exp(ML.log_density_ratio(cls, arrays, eps, b, eps0=1.0, b_sd0=2.57, sqrt_s=SQRT_S))
        assert abs(w.mean() - 1.0) < 4.0 * w.std() / math.sqrt(n)


# ----------------------------------------------------------------------
# 3. the radiative channels
# ----------------------------------------------------------------------
def test_photon_is_a_stable_final_state_outside_the_records():
    assert PHOTON_PDG == 22 and NON_HADRONIC_FINAL_STATES == {22} and 22 not in STABLE_STATES
    assert 22 in CATALOG.stable_states()
    with pytest.raises(KeyError):
        CATALOG.get(22)
    assert isospin_derivation_plan(CATALOG) == ()


@pytest.mark.parametrize("name, expected", [
    ("eta", {(211, -211, 111): 0.2302, (111, 111, 111): 0.3256, (211, -211, 22): 0.0428, (22, 22): 0.3936}),
    ("omega", {(211, -211, 111): 0.892 / 0.9906, (111, 22): 0.0833 / 0.9906, (211, -211): 0.0153 / 0.9906}),
    ("phi(1020)", {(321, -321): 0.501, (310, 130): 0.336, (213, -211): 0.049, (113, 111): 0.049, (-213, 211): 0.049,
                   (221, 22): 0.01306, (111, 22): 0.00133}),
])
def test_radiative_channel_tables(name, expected):
    """The revised tables of the committed records; the revision's eta-prime(958) left the committed catalog with
    the orphan-record revision of 2026-10-29 (the kernel's eta' is the record 331 of KernelConfig.eta_prime)."""
    record = next(r for r in CATALOG if r.name == name)
    got = {tuple(int(d) for d in ch.daughters): float(ch.branching_fraction) for ch in record.decay_channels}
    assert set(got) == set(expected)
    for key, value in expected.items():
        assert got[key] == pytest.approx(value, abs=1e-12)
    radiative = {tuple(int(d) for d in ch.daughters) for ch in record.decay_channels if ch.is_radiative}
    assert radiative == {k for k in expected if 22 in k}
    assert record.mass_range[0] == pytest.approx(min(sum(CATALOG.get(d).pole_mass if d != 22 else 0.0 for d in k)
                                                     for k in expected), abs=1e-12)


@pytest.mark.parametrize("pdg, expect", [
    (221, {211: (0.2302 + 0.0428) / 0.9922, 22: (2 * 0.3936 + 0.0428) / 0.9922}),
    (223, {211: (0.892 + 0.0153) / 0.9906, 22: 0.0833 / 0.9906}),
])
def test_decay_yields_follow_the_pdg_table(pdg, expect):
    record = CATALOG.get(pdg)
    counts = collections.Counter()
    n = 3000
    for i in range(n):
        for vertex in ResonanceDecayer(CATALOG, 41, 0, i).decay_resonance(pdg, (record.pole_mass, 0.0, 0.0, 0.0)):
            for daughter in vertex.daughters:
                if not daughter.is_resonance:
                    counts[daughter.pdg_id] += 1
    for species, value in expect.items():
        sigma = math.sqrt(value * max(1.0 - value, 0.1) / n) + 1e-3
        assert abs(counts[species] / n - value) < 4.0 * sigma, (species, counts[species] / n, value)
    assert counts[211] == counts[-211]


def test_three_body_phase_space_takes_a_massless_daughter():
    rng = np.random.default_rng(5)
    for _ in range(200):
        points = three_body_phase_space(0.547862, 0.13957039, 0.13957039, 0.0, rng)
        total = np.sum(points, axis=0)
        assert np.allclose(total, [0.547862, 0.0, 0.0, 0.0], atol=1e-12)
        photon = points[2]
        assert abs(photon[0] ** 2 - photon[1] ** 2 - photon[2] ** 2 - photon[3] ** 2) < 1e-12
    with pytest.raises(ValueError):
        three_body_phase_space(0.5, 0.1, 0.1, -0.1, rng)


def test_events_carry_photons_and_close():
    """Photons reach the final state through the kernel with the vendored
    PDG quantum numbers; the event closes exactly."""
    seen = 0
    for i in range(80):
        event = run_topology("nd", _params(0.5), CATALOG, FAMS, **_rngs(43, i))
        for h in event.final_state:
            if h.pdg_id == 22:
                seen += 1
                assert h.mass == 0.0 and h.qn == (0, 0, 0, 0)
        assert sum(h.qn[1] for h in event.final_state) == 2
    assert seen > 0
