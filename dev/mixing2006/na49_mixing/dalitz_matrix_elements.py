"""The measured Dalitz distributions of the light-meson three-body decays
(``KernelConfig.dalitz_matrix_elements``, 2026-10-20).

The three-body vertex of the decayer draws its point from flat phase space
(the ``phase_space`` law).  The decays below have measured Dalitz
distributions, and under the ``phase_space_dalitz`` law the vertex draws
them in place of the flat one:

* omega -> pi+ pi- pi0 (90%): the P-wave of a 1-- isoscalar,
  ``|p+ x p-|^2`` in the omega frame, times ``1 + 2 alpha Z + 2 beta
  Z^(3/2) sin 3 phi`` with the BESIII values alpha = 0.1202, beta =
  0.0295 (Phys. Rev. D 98 (2018) 112007).  The P-wave vanishes on the
  boundary of the Dalitz plot, where one pion is at rest or the three are
  collinear; it empties the soft end of the charged-pion momentum spectrum.
* eta -> pi+ pi- pi0 (23%): ``1 + a y + b y^2 + d x^2 + f y^3 + g x^2 y``
  with the KLOE-2 values a = -1.095, b = 0.145, d = 0.081, f = 0.141, g =
  -0.044 (JHEP 05 (2016) 019).
* eta -> 3 pi0 (33%): ``1 + 2 alpha z`` with alpha = -0.0288 (PDG).
* eta'(958) -> eta pi+ pi- and eta pi0 pi0: ``1 + a y + b y^2 + d x^2``
  with the BESIII values (Phys. Rev. D 97 (2018) 012003): a = -0.056, b =
  -0.049, d = -0.063 for the charged pions and a = -0.087, b = -0.073, d =
  -0.074 for the neutral ones.

The Dalitz variables are the standard ones on the kinetic energies T in
the parent frame, Q = M - m_b - m_c - m_d: for three pions x = sqrt(3)
(T+ - T-) / Q, y = 3 T0 / Q - 1, Z = x^2 + y^2 (x = sqrt(Z) cos phi, y =
sqrt(Z) sin phi, so Z^(3/2) sin 3 phi = 3 x^2 y - y^3) and z = (2/3)
sum_i (3 T_i / Q - 1)^2; for eta' -> eta pi pi x = sqrt(3) (T_pi1 -
T_pi2) / Q and y = (m_eta + 2 m_pi) / m_pi * T_eta / Q - 1.

The draw keeps the four uniforms of the flat vertex in their order: the
(b, c) invariant mass by the inverse CDF of its marginal density
``q*(M_bc) q(M; M_bc) G(M_bc)`` (G the integral of |A|^2 over the
helicity cosine), the helicity cosine of b in the (b, c) frame about the
pair's flight direction by the inverse of its conditional CDF, the
azimuth about that direction, and the d direction as in the flat vertex.
A run with the law therefore keeps every channel draw and every other
kinematic draw of a run without it, and only the decays listed here move.
The helicity integrals use five-point Gauss-Legendre quadrature, exact for
the polynomials of degree at most nine in the cosine these elements are;
every element is positive inside the Dalitz plot (checked, never
repaired).
"""

from __future__ import annotations

import bisect
import math

from .resonance_widths import decay_momentum

__all__ = ["DALITZ_MATRIX_ELEMENTS", "dalitz_element_for", "dalitz_pair_mass", "dalitz_helicity_cosine",
           "dalitz_kinetic_energies"]

_SQRT3 = math.sqrt(3.0)

OMEGA_ALPHA, OMEGA_BETA = 0.1202, 0.0295
ETA_CHARGED = {"a": -1.095, "b": 0.145, "d": 0.081, "f": 0.141, "g": -0.044}
ETA_NEUTRAL_ALPHA = -0.0288
ETAP_CHARGED = {"a": -0.056, "b": -0.049, "d": -0.063}
ETAP_NEUTRAL = {"a": -0.087, "b": -0.073, "d": -0.074}


def _omega_3pi(t, m, q, cross2):
    # (pi+, pi-, pi0)
    x = _SQRT3 * (t[0] - t[1]) / q
    y = 3.0 * t[2] / q - 1.0
    z = x * x + y * y
    return cross2 * (1.0 + 2.0 * OMEGA_ALPHA * z + 2.0 * OMEGA_BETA * (3.0 * x * x * y - y * y * y))


def _eta_charged(t, m, q, cross2):
    # (pi+, pi-, pi0)
    p = ETA_CHARGED
    x = _SQRT3 * (t[0] - t[1]) / q
    y = 3.0 * t[2] / q - 1.0
    return 1.0 + p["a"] * y + p["b"] * y * y + p["d"] * x * x + p["f"] * y * y * y + p["g"] * x * x * y


def _eta_neutral(t, m, q, cross2):
    # (pi0, pi0, pi0)
    z = (2.0 / 3.0) * sum((3.0 * ti / q - 1.0) * (3.0 * ti / q - 1.0) for ti in t)
    return 1.0 + 2.0 * ETA_NEUTRAL_ALPHA * z


def _etap(params):
    def element(t, m, q, cross2):
        # (eta, pi, pi)
        x = _SQRT3 * (t[1] - t[2]) / q
        y = (m[0] + 2.0 * m[1]) / m[1] * t[0] / q - 1.0
        return 1.0 + params["a"] * y + params["b"] * y * y + params["d"] * x * x
    return element


#: (parent pdg, the daughters in the element's order) -> the element
#: ``|A|^2(T, m, Q, |p_0 x p_1|^2)`` on the kinetic energies T and masses m
#: in that order (unnormalized).
DALITZ_MATRIX_ELEMENTS: dict = {
    (223, (211, -211, 111)): _omega_3pi,
    (221, (211, -211, 111)): _eta_charged,
    (221, (111, 111, 111)): _eta_neutral,
    (331, (221, 211, -211)): _etap(ETAP_CHARGED),
    (331, (221, 111, 111)): _etap(ETAP_NEUTRAL),
}


_ELEMENTS_BY_CHANNEL: dict = {}


def dalitz_element_for(parent_pdg: int, daughters):
    """The element of a decay with *daughters* in channel order, as a
    function of the channel-order kinetic energies and masses, or None.
    The registry's order is matched up to a permutation; the cross product
    of two momenta of a three-body decay has one magnitude for every pair.
    One function object per channel (the tables are keyed on it)."""
    drawn = tuple(int(d) for d in daughters)
    key = (int(parent_pdg), drawn)
    if key not in _ELEMENTS_BY_CHANNEL:
        _ELEMENTS_BY_CHANNEL[key] = _element_for(int(parent_pdg), drawn)
    return _ELEMENTS_BY_CHANNEL[key]


def _element_for(parent_pdg, drawn):
    for (parent, order), element in DALITZ_MATRIX_ELEMENTS.items():
        if parent != parent_pdg or sorted(order) != sorted(drawn):
            continue
        left = list(range(3))
        perm = []
        for d in order:
            k = next(i for i in left if drawn[i] == d)
            left.remove(k)
            perm.append(k)
        if perm == [0, 1, 2]:
            return element

        def permuted(t, m, q, cross2, _e=element, _p=tuple(perm)):
            return _e(tuple(t[k] for k in _p), tuple(m[k] for k in _p), q, cross2)
        return permuted
    return None


def dalitz_kinetic_energies(M, mb, mc, md, m_bc, cos_star):
    """The parent-frame kinetic energies of (b, c, d) and ``|p_b x p_c|^2``
    at the (b, c) mass *m_bc* and the helicity cosine *cos_star* of b in the
    (b, c) frame about the pair's flight direction."""
    q_star = decay_momentum(m_bc, mb, mc)
    q = decay_momentum(M, m_bc, md)
    e_pair = (M * M + m_bc * m_bc - md * md) / (2.0 * M)
    gamma, gamma_beta = e_pair / m_bc, q / m_bc
    eb_star = (m_bc * m_bc + mb * mb - mc * mc) / (2.0 * m_bc)
    ec_star = (m_bc * m_bc - mb * mb + mc * mc) / (2.0 * m_bc)
    eb = gamma * eb_star + gamma_beta * q_star * cos_star
    ec = gamma * ec_star - gamma_beta * q_star * cos_star
    ed = (M * M - m_bc * m_bc + md * md) / (2.0 * M)
    cross2 = q_star * q_star * (1.0 - cos_star * cos_star) * q * q
    return (eb - mb, ec - mc, ed - md), cross2


# five-point Gauss-Legendre on [-1, 1]
_GL_X = (-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640)
_GL_W = (0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891)


def _g(element, M, masses, m_bc, c):
    t, cross2 = dalitz_kinetic_energies(M, masses[0], masses[1], masses[2], m_bc, c)
    q_value = M - masses[0] - masses[1] - masses[2]
    value = float(element(t, masses, q_value, cross2))
    if not value >= -1e-12:
        raise ValueError(f"dalitz matrix element negative inside the Dalitz plot ({value!r} at M = {M!r}, "
                         f"M_bc = {m_bc!r}, cos = {c!r}); checked, never repaired")
    return max(value, 0.0)


def _helicity_integral(element, M, masses, m_bc, lo=-1.0, hi=1.0):
    half, mid = 0.5 * (hi - lo), 0.5 * (hi + lo)
    return half * math.fsum(w * _g(element, M, masses, m_bc, mid + half * x) for x, w in zip(_GL_X, _GL_W))


_GRID_POINTS = 257
_TABLES: dict = {}
_TABLES_MAX = 4096


def dalitz_pair_mass(M, mb, mc, md, element, u):
    """The (b, c) invariant mass at the uniform *u*: the inverse CDF of
    ``q*(M_bc) q(M; M_bc) G(M_bc)`` in ``dM_bc`` on the grid of the flat
    vertex (``M_bc = lo + (hi - lo) sin^2 phi``, the trapezoid in phi,
    linear in phi between the nodes)."""
    lo, hi = mb + mc, M - md
    key = (M, mb, mc, md, element)
    table = _TABLES.get(key)
    if table is None:
        n = _GRID_POINTS
        half_pi = 0.5 * math.pi
        phis = [half_pi * j / (n - 1) for j in range(n)]
        g = []
        for phi in phis:
            s, c = math.sin(phi), math.cos(phi)
            m = lo + (hi - lo) * s * s
            base = decay_momentum(m, mb, mc) * decay_momentum(M, m, md) * s * c
            g.append(base * _helicity_integral(element, M, (mb, mc, md), m) if base > 0.0 else 0.0)
        cumulative = [0.0]
        for j in range(1, n):
            cumulative.append(cumulative[-1] + 0.5 * (g[j] + g[j - 1]) * (phis[j] - phis[j - 1]))
        if not cumulative[-1] > 0.0:
            raise ValueError(f"dalitz_pair_mass: the Dalitz density carries no weight at M = {M!r}")
        if len(_TABLES) >= _TABLES_MAX:
            _TABLES.clear()
        table = _TABLES[key] = (phis, cumulative)
    phis, cumulative = table
    target = float(u) * cumulative[-1]
    n = len(phis)
    k = min(max(bisect.bisect_left(cumulative, target), 1), n - 1)
    c_lo, c_hi = cumulative[k - 1], cumulative[k]
    frac = 0.0 if not c_hi > c_lo else (target - c_lo) / (c_hi - c_lo)
    phi = phis[k - 1] + frac * (phis[k] - phis[k - 1])
    s = math.sin(phi)
    return min(max(lo + (hi - lo) * s * s, lo), hi)


def dalitz_helicity_cosine(M, mb, mc, md, m_bc, element, u):
    """The helicity cosine of b at the uniform *u*: the inverse of the
    conditional CDF of |A|^2 at fixed M_bc (the integral by five-point
    Gauss-Legendre on [-1, c], solved by bisection to the float step)."""
    masses = (mb, mc, md)
    total = _helicity_integral(element, M, masses, m_bc)
    if not total > 0.0:
        return 2.0 * float(u) - 1.0
    target = float(u) * total
    a, b = -1.0, 1.0
    for _ in range(60):
        mid = 0.5 * (a + b)
        if _helicity_integral(element, M, masses, m_bc, -1.0, mid) < target:
            a = mid
        else:
            b = mid
    return 0.5 * (a + b)
