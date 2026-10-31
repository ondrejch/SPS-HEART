"""Tests for the density-weighted continuation mass law (2026-10-03; the
R71 residual diagnosis, ``_knowledge/evidence/fixed-kernel-residual-
mechanisms-2026-10-03.md``).

The card-4 rule realizes a continuation daughter at
max(floor, edge - nominal_width), so every step releases the fixed kinetic
energy nominal_width.  A family with ``continuation_mass_law = "density"``
draws the mass on [floor, edge] from m**(-b_power) * exp(m / t_h) *
q*(M; m, m_other), with one uniform on the decay_kinematics substream
before the two angle uniforms.  The absent key reproduces the card-4
cascade bit for bit; the source density, its ramp and n_norm are
unchanged; the channel weights keep the card-4 mass.  The optional
continuation_barrier_l = L adds the orbital barrier q*^(2L+1) F_L(q*)^2
(after R73); L = 0 or the absent key is the plain law.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from na49_mixing.cascade_decay import (
    CATALOG_FAMILY_PREFIX,
    ISOTROPIC_ALIGNMENT_LABEL,
    CascadeState,
    _LIFTED_CASCADE_DEPTH,
    _cascade_depth_limit,
    _continuation_ceiling,
    _continuation_floor,
    _continuum_daughter_mass,
    draw_decay,
    enumerate_channels,
)
from na49_mixing.cascade_termination import run_cascade
from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.resonance_widths import decay_momentum
from na49_mixing.spectral_density import CONTINUATION_MASS_LAWS, _family_from_record, families_from_json, family_density

NESTED_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
FAMILIES_PATH = NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json"
SEED = 20261003
PION = 0.13957039
PROTON = 0.93827231


@pytest.fixture(scope="module")
def catalog() -> HadronCatalog:
    return HadronCatalog.from_json(CATALOG_PATH)


@pytest.fixture(scope="module")
def fams():
    return families_from_json(str(FAMILIES_PATH))


@pytest.fixture(scope="module")
def records():
    return json.loads(FAMILIES_PATH.read_text(encoding="utf-8"))["families"]


def _with(fams, **kw):
    return [dataclasses.replace(f, **kw) for f in fams]


def _family(fams, family_id):
    return next(f for f in fams if f.family_id == family_id)


def _root(family_id: str, mass: float, qn) -> CascadeState:
    return CascadeState(
        momentum=(float(mass), 0.0, 0.0, 0.0), mass=float(mass), qn=tuple(qn),
        quantum={}, family_id=family_id, axis=(0.0, 0.0, 1.0),
        alignment_label=ISOTROPIC_ALIGNMENT_LABEL, parent=None,
        exchange_ancestry=("density_law_probe",), depth=0)


def _exact_cdf(family, M, m_other, floor, edge):
    def p(m):
        return (m ** -family.b_power) * math.exp((m - edge) / family.t_h) * decay_momentum(M, m, m_other)
    total = quad(p, floor, edge, limit=400, epsabs=0.0, epsrel=1e-12)[0]
    return lambda x: quad(p, floor, x, limit=400, epsabs=0.0, epsrel=1e-12)[0] / total


def test_committed_config_keeps_the_card4_rule(fams):
    assert CONTINUATION_MASS_LAWS == ("edge", "density")
    assert all(f.continuation_mass_law is None for f in fams)
    assert _cascade_depth_limit(fams) == 12


def test_loader_reads_and_validates_the_law(records):
    record = copy.deepcopy(records[0])
    base = _family_from_record(record)
    for law in CONTINUATION_MASS_LAWS:
        record["continuation_mass_law"] = law
        family = _family_from_record(record)
        assert family.continuation_mass_law == law and family.n_norm == base.n_norm
        for mass in (2.9, 3.5, 4.5):
            assert family_density(family, mass) == family_density(base, mass)
    for bad in ("Density", "", None, 1, "thermal"):
        record["continuation_mass_law"] = bad
        with pytest.raises(ValueError):
            _family_from_record(record)


def test_edge_law_is_the_card4_rule_bit_for_bit(catalog, fams):
    """The explicit "edge" value and the absent key give the same cascades,
    draw for draw."""

    explicit = _with(fams, continuation_mass_law="edge", continuation_floor=1.7, continuation_ceiling=17.3)
    absent = _with(fams, continuation_floor=1.7, continuation_ceiling=17.3)
    for family_id, qn, mass in (("baryonic_np", (1, 1, 0), 9.0), ("mesonic_neutral", (0, 0, 0), 6.0)):
        lines = []
        for families in (absent, explicit):
            rc = np.random.default_rng(SEED); rk = np.random.default_rng(SEED + 1)
            lines.append([(s.family_id, s.mass, s.momentum)
                          for _ in range(40) for s in run_cascade(_root(family_id, mass, qn), catalog, families, rc, rk)])
        assert lines[0] == lines[1]


@pytest.mark.parametrize("family_id, M, m_other", [("mesonic_neutral", 6.0, PION), ("baryonic_np", 3.0, PION),
                                                   ("mesonic_neutral", 4.0, PROTON), ("baryonic_np", 9.0, 0.493677)])
def test_density_draw_follows_its_law(catalog, fams, family_id, M, m_other):
    family = dataclasses.replace(_family(fams, family_id), continuation_floor=1.7, continuation_ceiling=17.3,
                                 continuation_mass_law="density")
    floor = _continuation_floor(family, catalog)
    edge = min(_continuation_ceiling(family), M - m_other)
    n = 4000
    draws = np.asarray([_continuum_daughter_mass(family, M, m_other, catalog, (i + 0.5) / n) for i in range(n)])
    assert np.all((draws >= floor) & (draws <= edge)) and np.all(np.diff(draws) >= 0.0)
    cdf = _exact_cdf(family, M, m_other, floor, edge)
    for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        assert abs(cdf(float(np.quantile(draws, q))) - q) < 2e-3
    # the release Q = edge - m stays near the tower temperature
    release = edge - draws
    assert 0.1 < release.mean() < 0.35


def test_density_law_needs_its_uniform(catalog, fams):
    family = dataclasses.replace(_family(fams, "mesonic_neutral"), continuation_mass_law="density")
    with pytest.raises(ValueError):
        _continuum_daughter_mass(family, 4.0, PION, catalog)


def test_the_shed_hadron_is_soft(catalog, fams):
    """From a 6 GeV mesonic state the card-4 rule sheds a pion with
    q* = 0.59 GeV/c; the density law gives about 0.37 GeV/c on average,
    and a proton about 0.61 against 0.96."""

    base = dataclasses.replace(_family(fams, "mesonic_neutral"), continuation_floor=1.7, continuation_ceiling=17.3)
    dens = dataclasses.replace(base, continuation_mass_law="density")
    for m_other, card4, low, high in ((PION, 0.591, 0.33, 0.41), (PROTON, 0.956, 0.55, 0.67)):
        q4 = decay_momentum(6.0, _continuum_daughter_mass(base, 6.0, m_other, catalog), m_other)
        assert q4 == pytest.approx(card4, abs=2e-3)
        n = 2000
        qd = np.mean([decay_momentum(6.0, _continuum_daughter_mass(dens, 6.0, m_other, catalog, (i + 0.5) / n), m_other)
                      for i in range(n)])
        assert low < qd < high


def test_draw_budget_and_conservation(catalog, fams):
    """A vertex with a density-law continuum daughter draws 1 channel
    uniform and 3 kinematics uniforms (mass first, then the two angles);
    the daughters close the parent four-momentum and sit on shell."""

    dens = _with(fams, continuation_floor=1.7, continuation_ceiling=17.3, continuation_mass_law="density")
    edge = _with(fams, continuation_floor=1.7, continuation_ceiling=17.3)
    state = CascadeState(momentum=(20.0, 0.3, -0.2, math.sqrt(20.0 ** 2 - 0.13 - 7.0 ** 2)), mass=7.0, qn=(1, 1, 0),
                         quantum={}, family_id="baryonic_np", axis=(0.0, 0.0, 1.0),
                         alignment_label=ISOTROPIC_ALIGNMENT_LABEL, parent=None, exchange_ancestry=("budget",), depth=0)
    state = dataclasses.replace(state, momentum=(math.sqrt(7.0 ** 2 + 0.13 + 9.0 ** 2), 0.3, -0.2, 9.0))
    channels = enumerate_channels(state, catalog, dens)
    assert channels == enumerate_channels(state, catalog, edge)  # the weights keep the card-4 mass
    n_continuum = 0
    for k in range(200):
        rc, rk = np.random.default_rng([SEED, k, 0]), np.random.default_rng([SEED, k, 1])
        chosen, daughters = draw_decay(state, channels, catalog, dens, rc, rk)
        continuum = any(not d.family_id.startswith(CATALOG_FAMILY_PREFIX) for d in daughters)
        n_continuum += continuum
        ref_c, ref_k = np.random.default_rng([SEED, k, 0]), np.random.default_rng([SEED, k, 1])
        ref_c.random()
        for _ in range(3 if continuum else 2):
            ref_k.random()
        assert rc.random() == ref_c.random() and rk.random() == ref_k.random()
        total = [sum(d.momentum[j] for d in daughters) for j in range(4)]
        assert max(abs(a - b) for a, b in zip(total, state.momentum)) < 1e-9
    assert n_continuum > 10


def test_density_cascades_terminate_below_the_guard(catalog, fams):
    """From 16 GeV the density law steps down mostly by shedding rho,
    omega, K* and baryon resonances (about 1 GeV per step): depth 13 on
    average and 16 at most in 60 cascades, well inside the lifted guard of
    48 (the card-4 guard of 12 would stop some).  The density law alone,
    without a lifted ceiling, also raises the guard to 48."""

    dens = _with(fams, continuation_floor=1.7, continuation_ceiling=17.3, continuation_mass_law="density")
    assert _cascade_depth_limit(dens) == _LIFTED_CASCADE_DEPTH == 48
    assert _cascade_depth_limit(_with(fams, continuation_mass_law="density")) == 48
    rc = np.random.default_rng(SEED); rk = np.random.default_rng(SEED + 1)
    depths = []
    for _ in range(60):
        line = run_cascade(_root("baryonic_np", 16.0, (1, 1, 0)), catalog, dens, rc, rk,
                           max_depth=_cascade_depth_limit(dens))
        depths.append(max(s.depth for s in line))
    assert max(depths) < _LIFTED_CASCADE_DEPTH - 20
    assert 10 < np.mean(depths) < 20


@pytest.mark.parametrize("cls", ["sd_projectile", "sd_target", "dd", "nd"])
def test_events_conserve_with_the_density_law(catalog, fams, cls):
    from na49_mixing.rng_substreams import derive_substreams
    from na49_mixing.topology_runner import NOMINAL_PARAMS, run_topology
    dens = _with(fams, continuation_floor=1.7, continuation_ceiling=17.3, continuation_mass_law="density")
    weights = ({"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.1, "high_mass_baryonic_continuum": 0.9}
               if cls == "nd" else dict(NOMINAL_PARAMS.remnant_weights))
    params = dataclasses.replace(NOMINAL_PARAMS, epsilon=0.5, epsilon_nd=0.3, remnant_weights=weights)
    for i in range(40):
        subs = derive_substreams(SEED, 5, 9_600_000_000 + i)
        rngs = {f"rng_{lab}": np.random.default_rng(subs.stream_for(lab)) for lab in
                ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")}
        event = run_topology(cls, params, catalog, dens, **rngs)
        assert tuple(sum(int(h.qn[j]) for h in event.final_state) for j in range(3)) == (2, 2, 0)
        assert all(h.pdg_id is not None or h.line == "central" for h in event.final_state)


# ----------------------------------------------------------------------
# the orbital barrier of the density law (continuation_barrier_l; after R73)
# ----------------------------------------------------------------------
def test_loader_reads_and_validates_the_barrier(records):
    record = copy.deepcopy(records[0])
    assert _family_from_record(record).continuation_barrier_l is None
    for L in (0, 1, 2):
        record["continuation_barrier_l"] = L
        assert _family_from_record(record).continuation_barrier_l == L
    for bad in (-1, 1.0, "1", None, True):
        record["continuation_barrier_l"] = bad
        with pytest.raises(ValueError):
            _family_from_record(record)


def test_barrier_zero_is_the_plain_law_bit_for_bit(catalog, fams):
    plain = dataclasses.replace(_family(fams, "mesonic_neutral"), continuation_floor=1.7, continuation_ceiling=17.3,
                                continuation_mass_law="density")
    zero = dataclasses.replace(plain, continuation_barrier_l=0)
    for M, m_other in ((6.0, PION), (4.0, PROTON), (11.0, 0.493677)):
        for i in range(200):
            u = (i + 0.5) / 200
            assert _continuum_daughter_mass(plain, M, m_other, catalog, u) == _continuum_daughter_mass(zero, M, m_other, catalog, u)


@pytest.mark.parametrize("L", [1, 2])
def test_barrier_draw_follows_its_law(catalog, fams, L):
    from na49_mixing.resonance_widths import BLATT_WEISSKOPF_BARRIER
    family = dataclasses.replace(_family(fams, "mesonic_neutral"), continuation_floor=1.7, continuation_ceiling=17.3,
                                 continuation_mass_law="density", continuation_barrier_l=L)
    M, m_other = 6.0, PION
    floor = _continuation_floor(family, catalog)
    edge = min(_continuation_ceiling(family), M - m_other)

    def p(m):
        q = decay_momentum(M, m, m_other)
        return math.exp((m - edge) / family.t_h) * q ** (2 * L + 1) * BLATT_WEISSKOPF_BARRIER(L, q) ** 2

    total = quad(p, floor, edge, limit=400, epsabs=0.0, epsrel=1e-12)[0]
    n = 4000
    draws = np.asarray([_continuum_daughter_mass(family, M, m_other, catalog, (i + 0.5) / n) for i in range(n)])
    for q_ in (0.1, 0.25, 0.5, 0.75, 0.9):
        cdf = quad(p, floor, float(np.quantile(draws, q_)), limit=400, epsabs=0.0, epsrel=1e-12)[0] / total
        assert abs(cdf - q_) < 2e-3


def test_the_p_wave_barrier_hardens_the_step(catalog, fams):
    """From a 6 GeV mesonic state the shed pion has <q*> 0.37 GeV/c under
    the plain density law, 0.56 with the P-wave barrier (the card-4 rule
    gives 0.59 for every step) and 0.79 with L = 2."""

    base = dataclasses.replace(_family(fams, "mesonic_neutral"), continuation_floor=1.7, continuation_ceiling=17.3,
                               continuation_mass_law="density")
    n = 2000
    means = {}
    for L in (0, 1, 2):
        fam = dataclasses.replace(base, continuation_barrier_l=L)
        means[L] = np.mean([decay_momentum(6.0, _continuum_daughter_mass(fam, 6.0, PION, catalog, (i + 0.5) / n), PION)
                            for i in range(n)])
    assert 0.34 < means[0] < 0.40 and 0.52 < means[1] < 0.60 and 0.75 < means[2] < 0.84


def test_events_conserve_with_the_barrier(catalog, fams):
    from na49_mixing.rng_substreams import derive_substreams
    from na49_mixing.topology_runner import NOMINAL_PARAMS, run_topology
    dens = _with(fams, continuation_floor=1.7, continuation_ceiling=17.3, continuation_mass_law="density", continuation_barrier_l=1)
    weights = {"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.1, "high_mass_baryonic_continuum": 0.9}
    params = dataclasses.replace(NOMINAL_PARAMS, epsilon=0.5, epsilon_nd=0.2, remnant_weights=weights)
    for cls in ("sd_projectile", "nd"):
        for i in range(30):
            subs = derive_substreams(SEED, 6, 9_700_000_000 + i)
            rngs = {f"rng_{lab}": np.random.default_rng(subs.stream_for(lab)) for lab in
                    ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")}
            event = run_topology(cls, params if cls == "nd" else NOMINAL_PARAMS, catalog, dens, **rngs)
            assert tuple(sum(int(h.qn[j]) for h in event.final_state) for j in range(3)) == (2, 2, 0)
