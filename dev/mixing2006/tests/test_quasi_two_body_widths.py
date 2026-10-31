"""The convolved width law of quasi-two-body channels with a broad daughter
and the joint daughter-mass draw (FAB-10, 2026-10-02; external review g1
B3).

A two-body channel whose daughter is a broad resonance integrates its
phase space over the daughter's line shape (the Manley-Saleski form),
``Phi_c(M) = int rho_d(m) q(M; m, m_o)**(2L+1) F_L(q)**2 dm`` and
``Gamma_c(M) = BR_c Gamma_0 (Phi_c(M)/M) / (Phi_c(M0)/M0)``.  The channel
opens at the daughter's lower support edge, so Delta(1600) and
Delta(1700) to N rho and K1(1270) to rho K and omega K carry their
branching fractions at the pole.  The kernel draws the broad daughter's
mass from the same integrand.  Channels with only stable or narrow
daughters keep the pole-mass law bit for bit, and without a catalog
every width function is the pole-mass law.
"""

from __future__ import annotations

import collections
import math

import numpy as np
import pytest
from scipy.integrate import quad

from na49_mixing import resonance_widths as W
from na49_mixing.cascade_ablation import _committed_catalog
from na49_mixing.cascade_decay import CATALOG_FAMILY_PREFIX, enumerate_channels
from na49_mixing.resonance_decay import ResonanceDecayer
from na49_mixing.resonance_widths import (
    BLATT_WEISSKOPF_BARRIER,
    broad_daughter_record,
    channel_width,
    decay_momentum,
    draw_broad_daughter_mass,
    spectral_function,
    total_width,
)

CATALOG = _committed_catalog()
KERNEL = ResonanceDecayer(CATALOG, 0, 0, 0)


def _record(name):
    return next(r for r in CATALOG if r.name == name)


def _decaying():
    stable = CATALOG.stable_states()
    return [r for r in CATALOG if r.decay_channels and int(r.pdg_id) not in stable]


def _channel(record, daughters):
    want = tuple(sorted(daughters))
    return next(ch for ch in record.decay_channels if tuple(sorted(int(d) for d in ch.daughters)) == want)


def _broad_and_other(channel):
    d0, d1 = (int(d) for d in channel.daughters)
    b0 = broad_daughter_record(CATALOG, d0)
    return (b0, d1) if b0 is not None else (broad_daughter_record(CATALOG, d1), d0)


def test_broad_daughters_are_the_wide_unstable_records():
    names = {r.name for r in CATALOG if broad_daughter_record(CATALOG, int(r.pdg_id)) is not None}
    assert {"rho0", "rho+", "omega", "Delta++", "Delta0", "K*(892)+", "K*(892)0"} <= names
    for pdg in (211, 111, 2212, 321, 221):  # stable states and the eta needle
        assert broad_daughter_record(CATALOG, pdg) is None
    assert broad_daughter_record(None, 113) is None


def test_pole_normalization_holds_for_every_record():
    """Gamma(M0) = nominal_width * sum(BR) for every decaying record,
    including the three whose channels were closed at the pole."""
    for record in _decaying():
        masses = KERNEL._daughter_mass_map(record)
        br = math.fsum(float(ch.branching_fraction) for ch in record.decay_channels)
        got = total_width(record.pole_mass, record, masses, catalog=CATALOG) / record.nominal_width
        assert abs(got - br) < 1e-12, record.name


@pytest.mark.parametrize("parent, daughters", [
    ("Delta(1700)++", (2212, 213)), ("Delta(1600)+", (2212, 113)),
    ("K1(1270)+", (311, 213)), ("K1(1270)+", (321, 223)),
])
def test_channels_closed_at_the_pole_open(parent, daughters):
    record = _record(parent)
    channel = _channel(record, daughters)
    masses = KERNEL._daughter_mass_map(record)
    assert channel_width(record.pole_mass, channel, record, masses) == 0.0
    got = channel_width(record.pole_mass, channel, record, masses, catalog=CATALOG)
    assert got == pytest.approx(float(channel.branching_fraction) * record.nominal_width, rel=1e-12)
    broad, other = _broad_and_other(channel)
    edge = float(broad.mass_range[0]) + masses[other]
    assert channel_width(edge - 1e-6, channel, record, masses, catalog=CATALOG) == 0.0
    assert channel_width(edge + 0.02, channel, record, masses, catalog=CATALOG) > 0.0


def _quadrature_phase(M, broad, m_other, L):
    dmap = KERNEL._daughter_mass_map(broad)
    lo = float(broad.mass_range[0])
    hi = min(float(broad.mass_range[1]), M - m_other)
    if not hi > lo:
        return 0.0

    def integrand(m):
        q = decay_momentum(M, m, m_other)
        f = BLATT_WEISSKOPF_BARRIER(L, q)
        return spectral_function(m, broad, dmap, catalog=CATALOG) * q ** (2 * L + 1) * f * f

    pole = float(broad.pole_mass)
    points = [p for p in (pole - broad.nominal_width, pole, pole + broad.nominal_width) if lo < p < hi]
    return quad(integrand, lo, hi, limit=500, points=points or None, epsabs=0.0, epsrel=1e-9)[0]


@pytest.mark.parametrize("parent, daughters", [
    ("Delta(1700)++", (2212, 213)), ("Delta(1700)++", (2224, 111)),
    ("N*(1520)+", (2224, -211)), ("K1(1270)+", (321, 223)), ("K1(1270)+", (313, 211)),
])
def test_convolution_matches_direct_quadrature(parent, daughters):
    """The tabulated width agrees with an adaptive quadrature over the
    daughter's line shape to 1% wherever the channel carries more than
    one percent of its pole width."""
    record = _record(parent)
    channel = _channel(record, daughters)
    masses = KERNEL._daughter_mass_map(record)
    broad, other = _broad_and_other(channel)
    L = int(channel.orbital_angular_momentum)
    M0 = float(record.pole_mass)
    phi0 = _quadrature_phase(M0, broad, masses[other], L)
    gamma0 = float(channel.branching_fraction) * record.nominal_width
    checked = 0
    for M in np.linspace(float(record.mass_range[0]) + 0.01, float(record.mass_range[1]) - 0.01, 9):
        ref = gamma0 * (_quadrature_phase(M, broad, masses[other], L) / M) / (phi0 / M0)
        got = channel_width(M, channel, record, masses, catalog=CATALOG)
        if ref > 0.01 * gamma0:
            assert got == pytest.approx(ref, rel=0.01), (M, got, ref)
            checked += 1
    assert checked >= 4


def test_channels_without_a_broad_daughter_keep_the_pole_law():
    """Bit for bit, at the pole and across each support."""
    n = 0
    for record in _decaying():
        masses = KERNEL._daughter_mass_map(record)
        for channel in record.decay_channels:
            if W._convolution(record, channel, masses, BLATT_WEISSKOPF_BARRIER, CATALOG) is not None:
                continue
            for M in np.linspace(float(record.mass_range[0]), float(record.mass_range[1]), 7):
                assert (channel_width(M, channel, record, masses, catalog=CATALOG)
                        == channel_width(M, channel, record, masses))
            n += 1
    assert n > 100


def test_records_with_only_stable_daughters_keep_their_line_shape():
    for name in ("Delta++", "rho0", "K*(892)+"):
        record = _record(name)
        masses = KERNEL._daughter_mass_map(record)
        for M in np.linspace(float(record.mass_range[0]), float(record.mass_range[1]), 11):
            assert spectral_function(M, record, masses, catalog=CATALOG) == spectral_function(M, record, masses)


def _integrand_cdf(M, broad, m_other, L, grid):
    dmap = KERNEL._daughter_mass_map(broad)
    q = np.array([decay_momentum(M, m, m_other) for m in grid])
    f = np.array([BLATT_WEISSKOPF_BARRIER(L, v) for v in q])
    y = np.array([spectral_function(m, broad, dmap, catalog=CATALOG) for m in grid]) * q ** (2 * L + 1) * f * f
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(grid))])
    return cdf / cdf[-1]


@pytest.mark.parametrize("parent, daughters, M", [
    ("Delta(1700)++", (2212, 213), 1.70), ("Delta(1700)++", (2224, 111), 1.55), ("K1(1270)+", (321, 223), 1.272),
])
def test_joint_draw_follows_the_width_integrand(parent, daughters, M):
    record = _record(parent)
    channel = _channel(record, daughters)
    masses = KERNEL._daughter_mass_map(record)
    broad, other = _broad_and_other(channel)
    L = int(channel.orbital_angular_momentum)
    rng = np.random.default_rng(20261002)
    n = 20000
    draws = np.array([draw_broad_daughter_mass(M, masses[other], L, broad, CATALOG, rng) for _ in range(n)])
    lo, hi = float(broad.mass_range[0]), min(float(broad.mass_range[1]), M - masses[other])
    assert draws.min() >= lo and draws.max() <= hi
    grid = np.linspace(lo, hi, 4001)
    ref = _integrand_cdf(M, broad, masses[other], L, grid)
    emp = np.searchsorted(np.sort(draws), grid, side="right") / n
    assert np.max(np.abs(emp - ref)) < 1.63 / math.sqrt(n) + 0.003  # KS at the 1% level plus the grid


def test_joint_draw_consumes_two_uniforms():
    record = _record("Delta(1700)++")
    channel = _channel(record, (2212, 213))
    masses = KERNEL._daughter_mass_map(record)
    broad, other = _broad_and_other(channel)
    a, b = np.random.default_rng(7), np.random.default_rng(7)
    for _ in range(25):
        assert draw_broad_daughter_mass(1.7, masses[other], 1, broad, CATALOG, a) is not None
        b.random(); b.random()
    assert a.random() == b.random()
    assert draw_broad_daughter_mass(float(broad.mass_range[0]) + masses[other] - 1e-4, masses[other], 1,
                                    broad, CATALOG, a) is None
    assert a.random() == b.random()


def test_kernel_channel_frequencies_and_daughter_masses_at_the_pole():
    """Delta(1700)++ at its pole decays to p rho+ with the catalog
    branching fraction, and the rho+ masses lie below M - m_p."""
    record = _record("Delta(1700)++")
    M = float(record.pole_mass)
    objects, total = KERNEL._open_objects(record, M)
    expected = collections.Counter()
    for _channel_obj, combination, weight in objects:
        expected[tuple(sorted(combination))] += weight / total
    counts = collections.Counter()
    rho_masses = []
    n = 1500
    for i in range(n):
        vertices = ResonanceDecayer(CATALOG, 31, 0, i).decay_resonance(int(record.pdg_id), (M, 0.0, 0.0, 0.0))
        key = tuple(sorted(vertices[0].charge_combination))
        counts[key] += 1
        for daughter in vertices[0].daughters:
            if daughter.pdg_id == 213:
                e, px, py, pz = daughter.four_momentum
                rho_masses.append(math.sqrt(e * e - px * px - py * py - pz * pz))
    assert expected[(213, 2212)] == pytest.approx(0.35, abs=1e-12)
    for key, share in expected.items():
        assert abs(counts[key] / n - share) < 4.0 * math.sqrt(share * (1 - share) / n)
    assert max(rho_masses) < M - 0.938272 and np.mean(rho_masses) < 0.775


def test_cascade_catalog_enumeration_keeps_pole_mass_splits():
    """The cascade places established daughters at their pole masses, so
    its catalog-parent enumeration leaves out a channel that only the
    convolved width opens; the remaining weights are normalized."""
    from na49_mixing.cascade_ablation import _committed_fams
    from na49_mixing.cascade_decay import CascadeState

    record = _record("Delta(1700)++")
    M = float(record.pole_mass)
    state = CascadeState(momentum=(M, 0.0, 0.0, 0.0), mass=M, qn=(int(record.B), int(record.Q), int(record.S)),
                         quantum={}, family_id=f"{CATALOG_FAMILY_PREFIX}{int(record.pdg_id)}", axis=(0.0, 0.0, 1.0),
                         alignment_label="", parent=None, exchange_ancestry=(), depth=0)
    channels = enumerate_channels(state, CATALOG, _committed_fams())
    pairs = {tuple(sorted(int(d[len(CATALOG_FAMILY_PREFIX):]) for d in ch.daughters)) for ch in channels}
    assert (213, 2212) not in pairs and (211, 2212) in pairs
    assert math.fsum(ch.weight for ch in channels) == pytest.approx(1.0, abs=1e-12)
