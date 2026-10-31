"""The one-pion-exchange law of a leading nucleon (2026-10-26).

In p + p -> n + X the projectile proton turns into a neutron by emitting a
positive pion, and the pion and the other proton form the excited system
X++ of mass M (M^2 = xi s, the neutron at x_L ~ 1 - xi).  The double
differential cross section is the pion flux of the nucleon times the
pion-proton total cross section at sqrt(s') = M:

    d2sigma / dxi dt  ~  xi^(1 - 2 alpha_pi(t)) * (-t) / (t - m_pi^2)^2 * F(t)^2 * sigma_pi+p(M^2)

with the pion trajectory ``alpha_pi(t) = alpha' (t - m_pi^2)``.  The vertex
form factor is either the exponential ``F(t) = exp(r2 (t - m_pi^2))`` or
the proton's Dirac form factor (``form_factor="proton"``, the form of the
fixed pion term of Appleby et al., EPJC 76 (2016) 520, Eqs. 13 and 39):

    F_1(t) = (4 m_p^2 - 2.79 t) / (4 m_p^2 - t) / (1 - t / 0.71)^2 .

``sigma_pi+p`` is either constant over the excited masses (about 25 mb
above the resonance region) or the Regge fit of Donnachie and Landshoff,
Phys. Lett. B 296 (1992) 227 (``pion_proton="regge"``):

    sigma_pi+p(M^2) = 13.63 (M^2)^0.0808 + 27.56 (M^2)^-0.4525  mb .

The
factor ``(-t) / (t - m_pi^2)^2`` is the pseudoscalar coupling's spin flip
and the pion propagator: the flux vanishes at t = 0 and peaks near
``|t| ~ m_pi^2``.  At fixed xi the reachable window's upper end
``t_+ ~ -m_p^2 xi^2 / (1 - xi)`` closes the flux at large xi, so the
leading neutron peaks near x_L ~ 0.75 (the leading-neutron spectra of p+p
and e+p).  The law has no free parameter beyond ``alpha'`` and ``r2`` (none
beyond ``alpha'`` with the proton form factor); the share of the class is
the caller's.

The neutral partner (2026-10-27, ``pion="isospin"``): the pion couples to
the proton's isospin, so p -> n pi+ carries twice the squared coupling of
p -> p pi0 (``g^2_pi+ pn = 2 g^2_pi0 pp``).  The neutral exchange
p + p -> p + X+ has the same flux with the neutral pion's mass and the
``sigma_pi0 p`` of the pi+/pi- average (Regge coefficient 31.79 for the
27.56 of pi+ p).  Under ``"isospin"`` the share is the whole pion
exchange: :func:`ope_neutral_fraction` gives the neutral exchange's part,
``I_0 / (I_0 + 2 I_+)`` of the two integrated laws on the reachable
window (about one third), with no free parameter.  :func:`ope_parts`
gives the two single-charge laws that :func:`draw_ope` draws.

The absorptive correction (2026-10-27, ``absorption="eikonal"``): the
exchange takes place inside the p+p collision, whose elastic scattering
screens it.  In the absorption model (Sopkovich, Nuovo Cim. 26 (1962) 186;
Gottfried and Jackson, Nuovo Cim. 34 (1964) 735) the Born amplitude at
impact parameter ``b`` is multiplied by the elastic S-matrix of the initial
and the final two-body state, together ``1 - Gamma(b)`` with the p+p
profile ``Gamma(b) = sigma_tot / (4 pi B) exp(-b^2 / 2B)`` (``sigma_tot``
38.9 mb and the slope ``B`` 11.6 GeV^-2 at 158 GeV/c, :data:`PP_TOTAL_CROSS_SECTION_MB`,
:data:`PP_ELASTIC_SLOPE`; ``absorption="eikonal_pdg"``, 2026-10-30, takes the PDG compilation's 38.75 mb and the
slope 11.0 GeV^-2 at which the profile also gives the measured 6.97 mb elastic cross section, :func:`elastic_profile`).  The flux's two helicity amplitudes, the non-flip
one ``xi m_p F(t) / (q^2 + mu^2)`` and the flip one ``q F(t) / (q^2 + mu^2)``
with ``q`` the leading nucleon's transverse momentum and
``mu^2 = xi^2 m_p^2 + (1 - xi) m_pi^2``, are transformed to ``b`` (Bessel
``J_0`` and ``J_1``), screened and transformed back; their squares sum to
``(-t) F^2 / (t - m_pi^2)^2`` with ``t = -(q^2 + xi^2 m_p^2) / (1 - xi)``.
The survival factor ``S^2(xi, q) = |A_abs|^2 / |A|^2`` multiplies the flux
(:func:`ope_log_absorption`).  The pion's range ``1 / mu`` shrinks with
``xi``, so the heavy systems are absorbed more: averaged over ``t``, ``S^2``
is about 0.54 at ``x_L = 0.9``, 0.42 at 0.75 and 0.34 at 0.6, and it falls
from 0.9 at ``q -> 0`` to 0.17 at ``q = 0.6 GeV/c``.  The correction adds
no free parameter; the share stays the caller's.

:func:`draw_ope` draws one ``(M^2, t)`` pair on exactly three uniforms, the
budget of the single-diffraction mass and transfer draws: two for ``M^2``
on the marginal (the flux integrated over the reachable window, on the
geometric grid of :func:`~na49_mixing.topology_mass_laws.draw_marginal`),
one for ``t`` from the conditional on the window at that ``M^2``.  The
transfer is cut at ``-t_cut`` (20 GeV^2 by default), where the form
factor leaves less than ``exp(-2 r2 t_cut)`` of the flux (the square of
the proton form factor about 1e-5 there).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from .topology_mass_laws import PROTON_MASS, pp_t_window

__all__ = [
    "PION_MASS",
    "NEUTRAL_PION_MASS",
    "OPELaw",
    "ABSORPTIONS",
    "PP_TOTAL_CROSS_SECTION_MB",
    "PP_ELASTIC_SLOPE",
    "PP_TOTAL_CROSS_SECTION_MB_PDG",
    "PP_ELASTIC_CROSS_SECTION_MB_PDG",
    "PP_ELASTIC_SLOPE_PDG",
    "elastic_profile",
    "ope_law_of",
    "ope_component_of",
    "ope_parts",
    "ope_neutral_fraction",
    "ope_log_density",
    "ope_log_absorption",
    "pion_proton_cross_section",
    "ope_marginal_cells",
    "draw_ope",
]

#: The charged pion mass [GeV] (PDG 2024: 139.57039 MeV).
PION_MASS: float = 0.13957039
#: The neutral pion mass [GeV] (PDG 2024: 134.9768 MeV).
NEUTRAL_PION_MASS: float = 0.1349768

_GRID_POINTS: int = 4096
_T_POINTS: int = 512

#: The vertex form factors: ``exp(r2 (t - m_pi^2))`` and the proton's Dirac form factor.
FORM_FACTORS: tuple[str, ...] = ("exponential", "proton")
#: The pi+ p cross section over the excited masses: constant, or the Donnachie-Landshoff Regge fit.
PION_PROTON_LAWS: tuple[str, ...] = ("constant", "regge")
#: The exchanged pion: the charged one (p -> n pi+), the neutral one (p -> p pi0), or both at the coupling ratio of isospin.
PION_CHARGES: tuple[str, ...] = ("charged", "neutral", "isospin")
#: The absorptive correction: none, or the eikonal absorption model with the p+p elastic profile, at the R124 inputs
#: (``"eikonal"``) or at the PDG compilation's measurements (``"eikonal_pdg"``, 2026-10-30).
ABSORPTIONS: tuple[str, ...] = ("none", "eikonal", "eikonal_pdg")
#: The p+p total cross section [mb] at 158 GeV/c of ``"eikonal"`` (PDG compilation, 38.5-39.0 mb at p_lab 150-200 GeV/c).
PP_TOTAL_CROSS_SECTION_MB: float = 38.9
#: The p+p elastic slope [GeV^-2] at 158 GeV/c of ``"eikonal"`` (11.5-11.8 GeV^-2 at p_lab 150-200 GeV/c, |t| 0.05-0.3 GeV^2).
PP_ELASTIC_SLOPE: float = 11.6
#: The p+p total cross section [mb] of ``"eikonal_pdg"`` (2026-10-30, the manuscript review): the PDG compilation's
#: measurements at 150 and 170 GeV/c, 38.69 and 38.83 mb (Carroll et al., PL 61B (1976) 303), linear in p_lab at 158 GeV/c
#: (PDG, Workman et al., PTEP 2022, 083C01, the p+p total cross-section data file).
PP_TOTAL_CROSS_SECTION_MB_PDG: float = 38.69 + (38.83 - 38.69) * (158.0 - 150.0) / (170.0 - 150.0)
#: The p+p elastic cross section [mb] of ``"eikonal_pdg"``: the PDG compilation's measurements at 150 and 200 GeV/c,
#: 6.97 and 6.95 mb (Bartenev et al., Yad. Fiz. 22 (1974) 317), linear in p_lab at 158 GeV/c.
PP_ELASTIC_CROSS_SECTION_MB_PDG: float = 6.97 + (6.95 - 6.97) * (158.0 - 150.0) / (200.0 - 150.0)
_MB_TO_GEV2: float = 2.56819
#: The p+p elastic slope [GeV^-2] of ``"eikonal_pdg"``: the slope at which the purely absorptive Gaussian profile
#: ``Gamma(b) = sigma_tot / (4 pi B) exp(-b^2 / 2B)`` gives both measured cross sections, sigma_el = sigma_tot^2 / (16 pi B)
#: (11.0 GeV^-2).
PP_ELASTIC_SLOPE_PDG: float = (PP_TOTAL_CROSS_SECTION_MB_PDG ** 2 / (16.0 * math.pi * PP_ELASTIC_CROSS_SECTION_MB_PDG)
                               * _MB_TO_GEV2)
_DIPOLE_MASS2: float = 0.71
#: The grids of the survival-factor table: xi (log-spaced), the transverse momentum q [GeV/c] and impact parameter b [GeV^-1].
_ABS_XI: np.ndarray = np.exp(np.linspace(math.log(1e-3), math.log(0.95), 48))
_ABS_Q: np.ndarray = np.linspace(0.0, 6.0, 1200)
_ABS_B: np.ndarray = np.linspace(0.0, 80.0, 2000)


@dataclass(frozen=True)
class OPELaw:
    """The flux parameters: the pion trajectory slope ``alpha_prime``
    [GeV^-2] and the form factor slope ``r2`` [GeV^-2] of
    ``F(t) = exp(r2 (t - m_pi^2))``; ``t_cut`` [GeV^2] bounds ``|t|``.
    ``form_factor="proton"`` takes the proton's Dirac form factor instead
    (``r2`` then 0); ``pion_proton="regge"`` weights the masses with the
    Regge fit of ``sigma_pi+p(M^2)``.  ``pion`` names the exchanged pion
    (:data:`PION_CHARGES`): ``"neutral"`` takes the neutral pion's mass and
    ``sigma_pi0 p``, ``"isospin"`` both exchanges (:func:`ope_parts`).
    ``absorption="eikonal"`` multiplies the flux by the survival factor of
    the absorption model (:func:`ope_log_absorption`)."""

    alpha_prime: float
    r2: float
    t_cut: float = 20.0
    form_factor: str = "exponential"
    pion_proton: str = "constant"
    pion: str = "charged"
    absorption: str = "none"

    def __post_init__(self) -> None:
        for name in ("alpha_prime", "r2", "t_cut"):
            v = getattr(self, name)
            if isinstance(v, bool) or not (math.isfinite(float(v)) and float(v) >= 0.0):
                raise ValueError(f"OPELaw: {name} must be a finite non-negative number, got {v!r}")
        if not float(self.t_cut) > 0.0:
            raise ValueError(f"OPELaw: t_cut must be positive, got {self.t_cut!r}")
        if self.form_factor not in FORM_FACTORS or self.pion_proton not in PION_PROTON_LAWS:
            raise ValueError(f"OPELaw: form_factor is one of {FORM_FACTORS} and pion_proton one of {PION_PROTON_LAWS}, "
                             f"got {self.form_factor!r}, {self.pion_proton!r}")
        if self.form_factor == "proton" and float(self.r2) != 0.0:
            raise ValueError(f"OPELaw: the proton form factor has no slope r2, got {self.r2!r}")
        if self.pion not in PION_CHARGES:
            raise ValueError(f"OPELaw: pion is one of {PION_CHARGES}, got {self.pion!r}")
        if self.absorption not in ABSORPTIONS:
            raise ValueError(f"OPELaw: absorption is one of {ABSORPTIONS}, got {self.absorption!r}")

    @property
    def pion_mass(self) -> float:
        """The exchanged pion's mass [GeV] (the charged one for ``"isospin"``)."""
        return NEUTRAL_PION_MASS if self.pion == "neutral" else PION_MASS


def ope_law_of(alpha_prime, form, pion_proton: str = "constant", pion: str = "charged",
               absorption: str = "none") -> OPELaw:
    """The law of a component's ``(alpha', r2 | "proton"[, pion_proton[, pion[, absorption]]])``:
    a number is the exponential form factor's slope ``r2``, the string
    ``"proton"`` the proton's Dirac form factor."""
    if isinstance(form, str):
        if form != "proton":
            raise ValueError(f"ope_law_of: the form factor is a slope r2 or 'proton', got {form!r}")
        return OPELaw(float(alpha_prime), 0.0, form_factor="proton", pion_proton=pion_proton, pion=pion,
                      absorption=absorption)
    return OPELaw(float(alpha_prime), float(form), pion_proton=pion_proton, pion=pion, absorption=absorption)


def ope_component_of(component) -> tuple[float, float, float, OPELaw]:
    """A component ``(share, xi_lo, xi_hi, alpha', r2 | "proton"[, pion_proton[, pion[, absorption]]])``
    as its share, its window and its law; a malformed one (the share
    outside ``[0, 1)``, the window outside ``0 < xi_lo < xi_hi < 1``, a bad
    law) raises ``ValueError``."""
    comp = tuple(component)
    try:
        if len(comp) not in (5, 6, 7, 8):
            raise ValueError(f"5 to 8 entries, got {len(comp)}")
        share, xi_lo, xi_hi = (float(v) for v in comp[:3])
        law = ope_law_of(*comp[3:])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ope_component_of: the component is (share, xi_lo, xi_hi, alpha', r2 | 'proton'"
                         f"[, pion_proton[, pion[, absorption]]]), got {component!r} ({exc})") from exc
    if not (0.0 <= share < 1.0 and 0.0 < xi_lo < xi_hi < 1.0):
        raise ValueError(f"ope_component_of: needs a share in [0, 1) and 0 < xi_lo < xi_hi < 1, got {component!r}")
    return share, xi_lo, xi_hi, law


def ope_parts(law: OPELaw) -> tuple[OPELaw, OPELaw]:
    """The charged and the neutral single-charge laws of *law* (its form
    factor, trajectory and ``sigma_pi p`` law)."""
    return replace(law, pion="charged"), replace(law, pion="neutral")


def pion_proton_cross_section(m2: float, pion: str = "charged") -> float:
    """``sigma_pi+p`` [mb] at ``s' = M^2`` [GeV^2] from the Regge fit of
    Donnachie and Landshoff (1992): ``13.63 s'^0.0808 + 27.56 s'^-0.4525``;
    ``sigma_pi0 p`` for ``pion="neutral"``, the pi+/pi- average with the
    coefficient 31.79 (of 27.56 and 36.02, Appleby et al. 2016)."""
    m2 = float(m2)
    if not (m2 > 0.0 and math.isfinite(m2)):
        raise ValueError(f"pion_proton_cross_section: needs M^2 > 0, got {m2!r}")
    if pion not in ("charged", "neutral"):
        raise ValueError(f"pion_proton_cross_section: pion is 'charged' or 'neutral', got {pion!r}")
    return 13.63 * m2 ** 0.0808 + (31.79 if pion == "neutral" else 27.56) * m2 ** -0.4525


def _log_form_factor_squared(t, law: OPELaw):
    """``log F(t)^2`` of the law's vertex form factor (elementwise, ``t < 0``)."""
    if law.form_factor == "proton":
        mp4 = 4.0 * PROTON_MASS * PROTON_MASS
        d = 1.0 - t / _DIPOLE_MASS2
        return 2.0 * np.log((mp4 - 2.79 * t) / ((mp4 - t) * d * d))
    return 2.0 * float(law.r2) * (t - law.pion_mass * law.pion_mass)


def _form_factor(t, law: OPELaw):
    return np.exp(0.5 * _log_form_factor_squared(t, law))


def elastic_profile(law: OPELaw) -> tuple[float, float]:
    """The p+p total cross section [mb] and elastic slope [GeV^-2] of the
    law's absorption profile: the R124 inputs for ``"eikonal"``, the PDG
    compilation's for ``"eikonal_pdg"``."""
    if law.absorption == "eikonal_pdg":
        return PP_TOTAL_CROSS_SECTION_MB_PDG, PP_ELASTIC_SLOPE_PDG
    return PP_TOTAL_CROSS_SECTION_MB, PP_ELASTIC_SLOPE


@lru_cache(maxsize=8)
def _absorption_table(law: OPELaw) -> np.ndarray:
    """``log S^2`` on the grids ``(_ABS_XI, _ABS_Q)`` for the law's pion mass
    and form factor: the two helicity amplitudes of the flux at each ``xi``,
    taken to impact parameter (Hankel transforms of order 0 and 1), screened
    by ``1 - Gamma(b)`` and taken back; the common constants cancel in the
    ratio to the unscreened round trip."""
    from scipy.special import j0, j1
    q, b = _ABS_Q, _ABS_B
    dq, db = float(q[1] - q[0]), float(b[1] - b[0])
    j0bq, j1bq = j0(np.outer(b, q)), j1(np.outer(b, q))
    sigma_tot, slope = elastic_profile(law)
    gamma = sigma_tot * _MB_TO_GEV2 / (4.0 * math.pi * slope) * np.exp(-b * b / (2.0 * slope))
    mpi2 = law.pion_mass * law.pion_mass
    table = np.empty((_ABS_XI.size, q.size))
    for i, xi in enumerate(_ABS_XI):
        t = -(q * q + xi * xi * PROTON_MASS * PROTON_MASS) / (1.0 - xi)
        prop = _form_factor(t, law) / (q * q + xi * xi * PROTON_MASS * PROTON_MASS + (1.0 - xi) * mpi2)
        a0, a1 = xi * PROTON_MASS * prop, q * prop
        a0b = (j0bq * (q * a0)).sum(axis=1) * dq
        a1b = (j1bq * (q * a1)).sum(axis=1) * dq
        back = lambda jbq, ab, screen: (jbq.T * (b * ab * screen)).sum(axis=1) * db   # noqa: E731
        born = back(j0bq, a0b, 1.0) ** 2 + back(j1bq, a1b, 1.0) ** 2
        screened = back(j0bq, a0b, 1.0 - gamma) ** 2 + back(j1bq, a1b, 1.0 - gamma) ** 2
        table[i] = np.log(np.maximum(screened / np.maximum(born, 1e-300), 1e-12))
    return table


def ope_log_absorption(xi, t, law: OPELaw):
    """``log S^2(xi, q)`` of the law's absorptive correction (elementwise;
    zero for ``absorption="none"``), with ``q^2 = -t (1 - xi) - xi^2 m_p^2``
    the leading nucleon's transverse momentum (clipped at zero below the
    window's end); bilinear on the table in ``ln xi`` and ``q``, clamped at
    the grid's ends."""
    xi = np.asarray(xi, dtype=float)
    t = np.asarray(t, dtype=float)
    if law.absorption == "none":
        return np.zeros(np.broadcast(xi, t).shape)
    table = _absorption_table(replace(law, pion="charged" if law.pion == "isospin" else law.pion))
    xi_b, t_b = np.broadcast_arrays(xi, t)
    q = np.sqrt(np.maximum(-t_b * (1.0 - xi_b) - xi_b * xi_b * PROTON_MASS * PROTON_MASS, 0.0))
    lx = np.log(np.clip(xi_b, _ABS_XI[0], _ABS_XI[-1]))
    lgrid = np.log(_ABS_XI)
    i = np.clip(np.searchsorted(lgrid, lx, side="right") - 1, 0, lgrid.size - 2)
    wx = (lx - lgrid[i]) / (lgrid[i + 1] - lgrid[i])
    qc = np.clip(q, _ABS_Q[0], _ABS_Q[-1])
    k = np.clip(np.searchsorted(_ABS_Q, qc, side="right") - 1, 0, _ABS_Q.size - 2)
    wq = (qc - _ABS_Q[k]) / (_ABS_Q[k + 1] - _ABS_Q[k])
    row = lambda ii: table[ii, k] * (1.0 - wq) + table[ii, k + 1] * wq   # noqa: E731
    return row(i) * (1.0 - wx) + row(i + 1) * wx


def ope_log_density(xi, t, law: OPELaw):
    """``log`` of ``xi^(1 - 2 alpha_pi(t)) (-t) / (t - m_pi^2)^2 F(t)^2 S^2``
    (unnormalized; elementwise on arrays, ``t < 0``): the pion flux with the
    law's absorptive correction, without the ``sigma_pi+p(M^2)`` of the
    marginal."""
    _single(law, "ope_log_density")
    xi = np.asarray(xi, dtype=float)
    t = np.asarray(t, dtype=float)
    mpi2 = law.pion_mass * law.pion_mass
    alpha = float(law.alpha_prime) * (t - mpi2)
    return ((1.0 - 2.0 * alpha) * np.log(xi) + np.log(-t) - 2.0 * np.log(mpi2 - t)
            + _log_form_factor_squared(t, law) + ope_log_absorption(xi, t, law))


def _single(law: OPELaw, where: str) -> None:
    if law.pion == "isospin":
        raise ValueError(f"{where}: the law of both exchanges draws through its parts (ope_parts), got pion='isospin'")


def _t_grid(t_lo: float, t_hi: float, t_cut: float) -> np.ndarray:
    """The transfer grid of one mass: log-spaced in ``|t|`` from the
    window's upper end to its lower end or ``-t_cut``, as ``t`` values
    (increasing ``|t|``)."""
    a = max(-float(t_hi), 1e-12)
    b = min(-float(t_lo), float(t_cut))
    if not b > a:
        return np.array([-a])
    return -np.exp(np.linspace(math.log(a), math.log(b), _T_POINTS))


def _t_cdf(xi: float, t_lo: float, t_hi: float, law: OPELaw) -> tuple[np.ndarray, np.ndarray]:
    """The conditional transfer distribution at fixed ``xi``: the grid and
    its normalized cumulative (trapezoids in ``t``; ``cdf[0] = 0``)."""
    ts = _t_grid(t_lo, t_hi, law.t_cut)
    if ts.size < 2:
        return ts, np.ones(1)
    f = np.exp(ope_log_density(xi, ts, law))
    cell = 0.5 * (f[1:] + f[:-1]) * np.abs(np.diff(ts))
    cdf = np.concatenate([[0.0], np.cumsum(cell)])
    return ts, cdf


def _t_integral(xi: float, t_lo: float, t_hi: float, law: OPELaw) -> float:
    ts, cdf = _t_cdf(xi, t_lo, t_hi, law)
    return float(cdf[-1]) if ts.size > 1 else 0.0


@lru_cache(maxsize=32)
def ope_marginal_cells(m2_min: float, m2_max: float, m2_remnant: float, s: float, law: OPELaw,
                       sqrt_s_window: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The marginal of ``M^2`` on the geometric grid (uniform in
    ``y = ln M^2``): the grid ``ys`` and the per-cell masses of
    ``dsigma / dy = xi dsigma / dxi`` with ``dsigma/dxi`` the flux
    integrated over the reachable window of ``(M^2, m_remnant^2)``
    (:func:`~na49_mixing.topology_mass_laws.pp_t_window` at
    *sqrt_s_window*, its pinned default when ``None``), times
    ``sigma_pi+p(M^2)`` with ``pion_proton="regge"``."""
    _single(law, "ope_marginal_cells")
    lo, hi = float(m2_min), float(m2_max)
    if not (0.0 < lo < hi and math.isfinite(hi)) or not s > 0.0:
        raise ValueError(f"ope_marginal_cells: needs 0 < m2_min < m2_max and s > 0, got ({m2_min!r}, {m2_max!r}, {s!r})")
    ys = np.linspace(math.log(lo), math.log(hi), _GRID_POINTS)
    m2s = np.exp(ys)
    win = (lambda m2: pp_t_window(m2, float(m2_remnant))) if sqrt_s_window is None else (
        lambda m2: pp_t_window(m2, float(m2_remnant), float(sqrt_s_window)))
    g = np.empty(_GRID_POINTS)
    for i, m2 in enumerate(m2s):
        t_lo, t_hi = win(float(m2))
        xi = float(m2) / float(s)
        g[i] = xi * _t_integral(xi, t_lo, t_hi, law)
        if law.pion_proton == "regge":
            g[i] *= pion_proton_cross_section(float(m2), law.pion)
    cell = 0.5 * (g[1:] + g[:-1]) * np.diff(ys)
    if not np.sum(cell) > 0.0:
        raise ValueError("ope_marginal_cells: the flux vanishes on the mass window")
    return ys, cell


@lru_cache(maxsize=32)
def ope_neutral_fraction(m2_min: float, m2_max: float, m2_remnant_charged: float, m2_remnant_neutral: float, s: float,
                         law: OPELaw) -> float:
    """The neutral exchange's part of the whole pion exchange of *law* on
    the mass window: ``I_0 / (I_0 + 2 I_+)``, with ``I`` the integral of
    each single-charge law (:func:`ope_marginal_cells`, the neutron
    remnant for the charged exchange and the proton for the neutral one)
    and 2 the squared-coupling ratio ``g^2_pi+ pn / g^2_pi0 pp``."""
    charged, neutral = ope_parts(law)
    i_plus = float(np.sum(ope_marginal_cells(float(m2_min), float(m2_max), float(m2_remnant_charged), float(s), charged)[1]))
    i_zero = float(np.sum(ope_marginal_cells(float(m2_min), float(m2_max), float(m2_remnant_neutral), float(s), neutral)[1]))
    return i_zero / (i_zero + 2.0 * i_plus)


def draw_ope(m2_min: float, m2_max: float, m2_remnant: float, s: float, law: OPELaw, rng,
             sqrt_s_window: float | None = None) -> tuple[float, float]:
    """One ``(M^2, t)`` draw of the one-pion-exchange law on exactly three
    uniforms from *rng*: the grid cell and the position within it for
    ``M^2`` (the scheme of ``draw_marginal``), then ``t`` from the
    conditional on the reachable window at that ``M^2`` (the inverse of its
    cumulative on the transfer grid).  *s* sets ``xi = M^2 / s``."""
    ys, cell = ope_marginal_cells(float(m2_min), float(m2_max), float(m2_remnant), float(s), law,
                                  None if sqrt_s_window is None else float(sqrt_s_window))
    cdf = np.cumsum(cell) / float(np.sum(cell))
    u_segment = rng.random()
    idx = min(int(np.searchsorted(cdf, u_segment, side="right")), cell.size - 1)
    u_within = rng.random()
    y = ys[idx] + u_within * (ys[idx + 1] - ys[idx])
    m2 = float(min(max(math.exp(y), float(m2_min)), float(m2_max)))
    t_lo, t_hi = (pp_t_window(m2, float(m2_remnant)) if sqrt_s_window is None
                  else pp_t_window(m2, float(m2_remnant), float(sqrt_s_window)))
    ts, tcdf = _t_cdf(m2 / float(s), t_lo, t_hi, law)
    u_t = rng.random()
    if ts.size < 2:
        return m2, float(ts[0])
    t = float(np.interp(u_t * tcdf[-1], tcdf, ts))
    return m2, float(min(max(t, float(t_lo)), float(t_hi)))
