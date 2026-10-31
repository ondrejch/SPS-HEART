"""Two-sided hadronization of a heavy excited system (2026-10-26).

An excited system X of mass M_X formed by a proton and an exchanged object
(the Pomeron of diffraction, the positive pion of one-pion exchange) is a
collision at sqrt(s') = M_X.  The cascade decays it as one isotropic
continuum line, so its baryon and its hardest hadrons sit near the
system's own rapidity.  In a hadronic collision the baryon keeps its side:
the proton's fragments follow the proton and the exchanged object's
fragments the other way.  Here X hadronizes as two lines in its rest
frame: a baryonic line along the incoming proton, which in the X rest
frame runs along the X flight direction (the exchange taken at t = 0), and
the exchanged object's line opposite, neutral (0, 0, 0) for the Pomeron
and charged (0, +1, 0) for the pi+ of one-pion exchange.  Each line then
decays through the cascade.

The draw budget is fixed (no rejection):

- :func:`split_masses`: two uniforms, the baryonic line's mass from
  ``(M^2)^(-1-eps)`` on ``[lo1, M_X - lo2]``, then the other line's on
  ``[lo2, M_X - M_1]``;
- :func:`split_kinematics`: two uniforms, the transfer at the baryonic
  vertex from ``exp(b t)`` on its exact window and the azimuth.

The products of :func:`split_kinematics` are the two lines' four-momenta in
the frame of the excitation's own four-momentum (the caller's frame), and
they sum to it exactly up to rounding.
"""

from __future__ import annotations

import math

__all__ = [
    "draw_line_mass",
    "split_masses",
    "split_kinematics",
]

_PROTON_MASS: float = 0.93827231


def draw_line_mass(lo: float, hi: float, eps: float, u: float) -> float:
    """The inverse of the cumulative of ``(M^2)^(-1-eps)`` in ``M^2`` on
    ``[lo^2, hi^2]`` at the uniform *u*; ``eps = 0`` is flat in ``ln M^2``."""
    lo, hi, eps, u = float(lo), float(hi), float(eps), float(u)
    if not (0.0 < lo < hi and math.isfinite(hi)) or not 0.0 <= u <= 1.0:
        raise ValueError(f"draw_line_mass: needs 0 < lo < hi and u in [0, 1], got ({lo!r}, {hi!r}, {u!r})")
    a, b = lo * lo, hi * hi
    if abs(eps) < 1e-12:
        m2 = a * (b / a) ** u
    else:
        m2 = (a ** (-eps) - u * (a ** (-eps) - b ** (-eps))) ** (-1.0 / eps)
    return float(min(max(math.sqrt(m2), lo), hi))


def split_masses(mx: float, lo1: float, lo2: float, eps: float, rng) -> tuple[float, float]:
    """The two line masses on exactly two uniforms: the baryonic line on
    ``[lo1, M_X - lo2]``, then the other line on ``[lo2, M_X - M_1]``."""
    mx = float(mx)
    if not mx > float(lo1) + float(lo2):
        raise ValueError(f"split_masses: M_X = {mx!r} lies below the two thresholds {lo1!r} + {lo2!r}")
    m1 = draw_line_mass(lo1, mx - lo2, eps, float(rng.random()))
    hi2 = mx - m1
    u2 = float(rng.random())
    m2 = draw_line_mass(lo2, hi2, eps, u2) if hi2 > float(lo2) * (1.0 + 1e-12) else float(lo2)
    return m1, m2


def _boost(p4, beta):
    bx, by, bz = beta
    b2 = bx * bx + by * by + bz * bz
    if b2 <= 0.0:
        return tuple(float(v) for v in p4)
    g = 1.0 / math.sqrt(1.0 - b2)
    bp = bx * p4[1] + by * p4[2] + bz * p4[3]
    f = (g - 1.0) * bp / b2 + g * p4[0]
    return (g * (p4[0] + bp), p4[1] + f * bx, p4[2] + f * by, p4[3] + f * bz)


def split_kinematics(excitation, mx: float, m1: float, m2: float, b: float, rng):
    """The two lines' four-momenta in the frame of *excitation* (its
    four-momentum, mass *mx*) on exactly two uniforms: the baryonic line
    (mass *m1*) along the X flight direction at the transfer ``t`` from
    ``exp(b t)`` on its exact window (the incoming proton at energy
    ``(M_X^2 + m_p^2) / 2 M_X`` along the flight direction, the exchanged
    object massless), the azimuth flat; the other line (mass *m2*)
    opposite.  Returns ``(p4_1, p4_2)``."""
    mx, m1, m2, b = float(mx), float(m1), float(m2), float(b)
    if not (m1 > 0.0 and m2 > 0.0 and m1 + m2 < mx) or not b > 0.0:
        raise ValueError(f"split_kinematics: needs m1 + m2 < M_X and b > 0, got ({m1!r}, {m2!r}, {mx!r}, {b!r})")
    mp2 = _PROTON_MASS * _PROTON_MASS
    ep, pp = (mx * mx + mp2) / (2.0 * mx), (mx * mx - mp2) / (2.0 * mx)
    q = math.sqrt(max(0.0, (mx * mx - (m1 + m2) ** 2) * (mx * mx - (m1 - m2) ** 2))) / (2.0 * mx)
    e1 = (mx * mx + m1 * m1 - m2 * m2) / (2.0 * mx)
    t_hi = mp2 + m1 * m1 - 2.0 * (ep * e1 - pp * q)
    t_lo = mp2 + m1 * m1 - 2.0 * (ep * e1 + pp * q)
    u = float(rng.random())
    span = t_hi - t_lo
    t = t_hi + math.log(1.0 - u * (1.0 - math.exp(-b * span))) / b if span > 0.0 else t_hi
    c = max(-1.0, min(1.0, (t - mp2 - m1 * m1 + 2.0 * ep * e1) / (2.0 * pp * q))) if q > 0.0 else 1.0
    sn = math.sqrt(max(0.0, 1.0 - c * c))
    phi = 2.0 * math.pi * float(rng.random())
    loc1 = (e1, q * sn * math.cos(phi), q * sn * math.sin(phi), q * c)
    loc2 = (mx - e1, -loc1[1], -loc1[2], -loc1[3])
    px, py, pz = (float(excitation[1]), float(excitation[2]), float(excitation[3]))
    pn = math.sqrt(px * px + py * py + pz * pz)
    n = (px / pn, py / pn, pz / pn) if pn > 0.0 else (0.0, 0.0, 1.0)
    a = (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0)
    u1 = (a[1] * n[2] - a[2] * n[1], a[2] * n[0] - a[0] * n[2], a[0] * n[1] - a[1] * n[0])
    un = math.sqrt(sum(x * x for x in u1))
    u1 = tuple(x / un for x in u1)
    u2 = (n[1] * u1[2] - n[2] * u1[1], n[2] * u1[0] - n[0] * u1[2], n[0] * u1[1] - n[1] * u1[0])
    e_x = float(excitation[0])
    beta = (px / e_x, py / e_x, pz / e_x)
    out = []
    for loc in (loc1, loc2):
        vec = tuple(loc[1] * u1[k] + loc[2] * u2[k] + loc[3] * n[k] for k in range(3))
        out.append(_boost((loc[0],) + vec, beta))
    return out[0], out[1]
