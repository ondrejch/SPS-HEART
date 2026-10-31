"""Mass-dependent channel widths and the normalized spectral function.

Round 2, card 3 (``TASK-20260913-r2-mass-dependent-widths``; plan
section 22 task 11, plan section 8.2 step 2; module name of plan
section 4.3), with the barrier factor of Round 2, card 4
(``TASK-20260913-r2-barrier-factors``; plan section 22 task 12).  The
module is a pure-function substrate: per-channel mass-dependent widths,
the channel-summed total width, the normalized spectral function, and
one deterministic mass draw.  Every physics input comes from the
card-2 catalog records (``na49_mixing.hadron_catalog``) or from the
caller's daughter-mass map; no value is fitted or transcribed here.

Width law (plan section 8.2, for a dominant two-body channel with
orbital momentum L)::

    Gamma_c(M) = Gamma_c,0
                 * (q(M) / q(M0))**(2L+1)
                 * (M0 / M)
                 * (F_L(q(M)) / F_L(q(M0)))**2,

with ``Gamma_c,0 = BR_c * nominal_width`` at the pole mass M0 and the
Kaellen decay momentum ``q(M) = sqrt(kallen(M**2, mb**2, mc**2)) / (2M)``
for the daughter pair (b, c).  The barrier factor F_L enters through
the squared ratio ``(F_L(q(M)) / F_L(q(M0)))**2``.  From card 4 the
default barrier is the documented Blatt-Weisskopf polynomial
:data:`BLATT_WEISSKOPF_BARRIER` with the hadronic radius
``BARRIER_R0 = 1.0 GeV^-1`` (plan section 8.2; round 2 index item 4)::

    F_0(q) = 1
    F_1(q) = 1 / sqrt(1 + (q r0)^2)
    F_2(q) = sqrt(9 / (z^2 + 3 z + 9)), z = (q r0)^2      (re-normalized,
             F_2(0) = 1)

with ``q`` in GeV.  The same re-normalized ratio convention continues
with the standard Blatt-Weisskopf polynomials (von Hippel and Quigg,
Phys. Rev. D 5, 624 (1972)) to every ``L >= 3``; the catalog's maximum
``orbital_angular_momentum`` is 4 (N*(1680) and rho3(1690) carry F-wave
``L = 3`` channels, f4(2050) carries a G-wave ``L = 4`` channel), and
the first two standard members above ``L = 2`` are::

    F_3(q) = 15 / sqrt((q r0)^6 + 6 (q r0)^4 + 45 (q r0)^2 + 225)
    F_4(q) = 105 / sqrt((q r0)^8 + 10 (q r0)^6 + 135 (q r0)^4
                        + 1575 (q r0)^2 + 11025)

i.e. the same squared-ratio form ``F_L(q)^2 = Z_L(0) / Z_L(z)`` with
``z = (q r0)^2`` and the standard polynomials ``Z_3(z) = z^3 + 6 z^2
+ 45 z + 225`` and ``Z_4(z) = z^4 + 10 z^3 + 135 z^2 + 1575 z
+ 11025`` (higher ``L`` through the same exact coefficient
construction, :func:`_blatt_weisskopf_z_coefficients`; the ``L = 1``
form above is the ``L = 1`` member of the same family, ``Z_1(z) = 1
+ z``).  A negative L is rejected with a ``ValueError`` naming the L
(a negative orbital angular momentum is not a partial wave).
``F_L(0) = 1`` for every supported L, so the barrier is
finite and continuous at the channel threshold and the threshold
behavior below is unchanged; at the pole momentum the ratio term is 1
by construction, so the pole normalization is unchanged.  ``r0`` is a
documented constant (plan section 8.2), not a fit parameter.  The
card-3 identity placeholder :data:`IDENTITY_BARRIER` (F_L == 1 for
every L and q) remains available as an explicit argument.

Zero width below threshold is exact, not a clamp: ``channel_width``
returns exactly ``0.0`` at and below the channel threshold, and the
``(q/q0)**(2L+1)`` factor makes the right limit at the threshold 0 for
every L >= 0, so the width is continuous at the threshold.  A channel
closed at the pole mass (q(M0) = 0) sits outside the pole
normalization; it carries zero width by convention.

Three-body channels (``is_three_body``, card 7) use the same law with
the two-body phase-space factor applied to the leading pair
(``daughters[0], daughters[1]``); the threshold remains the sum of all
daughter masses, matching ``HadronCatalog``.  Card 7 may refine this
convention; the refinement must not change two-body widths.

Line shape (round 2 index, item 5): the relativistic Breit-Wigner with
the channel-summed mass-dependent width::

    rho(M) proportional to  M**2 * Gamma(M)
              / ((M**2 - M0**2)**2 + M0**2 * Gamma(M)**2),  Gamma(M) = sum_c Gamma_c(M)

on the support ``[mass_range[0], mass_range[1]] = [threshold, m_up]``
(the record's ``mass_range``), zero outside the support, and
renormalized so the integral over the support equals 1 under the
documented normalization quadrature.  The normalization constant is
the trapezoid integral of the unnormalized shape on the fixed
10,001-node grid over the support (:data:`_NORMALIZATION_GRID_POINTS`),
the pre-registered reference quadrature of the round-2 check suite
(``SPECTRAL_QUADRATURE_NODES`` of ``na49_mixing.resonance_decay``), so
the discretely normalized density is grid-convergent by construction:
the same fixed quadrature defines the normalization and measures it,
and the renormalized density reproduces the unit integral on that grid
to float rounding for every record.  The original 1024-point
normalization grid predated that pre-registration, and its node
spacing exceeds the line width of the narrow-width records: for the
eta (hadronic width 1.31 keV) the spacing is about 100 times the line
width, for the eta-prime(958) the line width spans about 1.5 cells.
On such records the two discretizations disagree arbitrarily far (the
pre-registered check read 0.3334 for the eta and 1.0380 for the
eta-prime), and no normalization constant reconciles a line narrower
than the check grid's own node spacing with that measurement: the
check-grid trapezoid of the eta shape misses the unresolved peak, and
even a fully converged normalization constant would read about 0.06
there.  Normalizing on the pre-registered reference quadrature is the
resolution of the card-8 check round's re-open of this module (fix
round 2026-09-14); the width law, the support, and the pole
normalization are unchanged.

``draw_mass`` reuses the truncated, explicitly renormalized draw
convention of ``na49_mixing.breit_wigner.draw_truncated_bw``: the draw
inverts the CDF of the line shape restricted to the window, evaluated
on a fixed 1024-point grid (:data:`_DRAW_GRID_POINTS`, unchanged), so
the truncation is normalized by construction and no full draw followed
by rejection is involved.  The draw table re-normalizes over the
window internally, so the support normalization cancels and the draw
convention is independent of the normalization quadrature above.  It
consumes exactly two uniform draws per non-None draw (segment choice
plus within-cell position) and none for a ``None`` result, so the
per-decay draw count is stable across line shapes.  A window that
carries no line-shape mass returns ``None``.

Module-level caches keyed on the barrier identity, the immutable
record, and the mass map hold the normalization-grid and draw-table
evaluations for the two module-level barrier singletons (identity and
Blatt-Weisskopf), mirroring the ``breit_wigner`` caching discipline.
A stored value is the bit-identical result of the one grid evaluation;
the caches are pure compute caches and consume no random numbers on
any path.

Hermetic: standard library, numpy, and the card-1 catalog schema only;
no event generation, no external tables.
"""

from __future__ import annotations

import itertools
import math
import weakref
from collections.abc import Mapping
from typing import Protocol

import numpy as np

from .hadron_catalog import DecayChannel, HadronRecord

__all__ = [
    "IDENTITY_BARRIER",
    "BARRIER_R0",
    "BLATT_WEISSKOPF_BARRIER",
    "BROAD_DAUGHTER_MIN_WIDTH",
    "BlattWeisskopfBarrier",
    "BarrierFactor",
    "broad_daughter_record",
    "channel_width",
    "decay_momentum",
    "draw_broad_daughter_mass",
    "draw_mass",
    "kallen",
    "spectral_function",
    "total_width",
]


#: Fixed grid of the support-normalization convention: the
#: pre-registered reference quadrature of the round-2 check suite
#: (``SPECTRAL_QUADRATURE_NODES`` of ``na49_mixing.resonance_decay``;
#: the equality is pinned in ``tests/test_resonance_widths.py``).  The
#: normalization constant of ``rho`` is the trapezoid integral of the
#: unnormalized shape on this grid over the support, so the discretely
#: normalized density is grid-convergent on the reference quadrature
#: that measures it (the card-8 check round's re-open of the 1024-point
#: convention; fix round 2026-09-14).
_NORMALIZATION_GRID_POINTS = 10_001

#: Fixed grid of the truncated-draw convention (the
#: ``draw_truncated_bw`` grid of ``na49_mixing.breit_wigner``),
#: unchanged by the normalization-grid fix: the draw table
#: re-normalizes over the window internally, so the draw convention is
#: independent of the support-normalization quadrature.
_DRAW_GRID_POINTS = 1024


# ----------------------------------------------------------------------
# Kinematics
# ----------------------------------------------------------------------
def kallen(x: float, y: float, z: float) -> float:
    """The Kaellen triangle function ``lambda(x, y, z)``.

    ``lambda(x, y, z) = x^2 + y^2 + z^2 - 2xy - 2xz - 2yz``.
    """
    x = float(x)
    y = float(y)
    z = float(z)
    return x * x + y * y + z * z - 2.0 * x * y - 2.0 * x * z - 2.0 * y * z


def decay_momentum(M: float, mb: float, mc: float) -> float:
    """Decay momentum of a two-body channel (b, c) at parent mass M.

    ``q(M) = sqrt(kallen(M**2, mb**2, mc**2)) / (2 M)``; exactly ``0.0``
    at and below the threshold ``mb + mc``, where the two-body decay is
    kinematically forbidden.  The forbidden region is wider than the
    set where the Kaellen function is non-positive: ``kallen`` has its
    two roots at ``(mb - mc)**2`` and ``(mb + mc)**2`` and is positive
    also for ``0 < M < |mb - mc|``, so the threshold guard below is the
    physical condition and the ``kallen`` guard is a numerical safety
    net in the negative gap.
    """
    M = float(M)
    mb = float(mb)
    mc = float(mc)
    if not M > mb + mc:
        return 0.0
    lam = kallen(M * M, mb * mb, mc * mc)
    if lam <= 0.0:
        return 0.0
    return math.sqrt(lam) / (2.0 * M)


# ----------------------------------------------------------------------
# Barrier factors (the card-3 identity placeholder and the card-4
# documented Blatt-Weisskopf default; the signatures are unchanged)
# ----------------------------------------------------------------------
class BarrierFactor(Protocol):
    """The orbital-momentum barrier factor ``F_L(q)`` of the plan
    section 8.2 width law.

    A barrier implementation is any callable with this signature; the
    width law uses it through the ratio
    ``(barrier(L, q(M)) / barrier(L, q(M0)))**2``, so it must be
    positive at the pole momentum.
    """

    def __call__(self, L: int, q: float) -> float: ...


class _IdentityBarrier:
    """The card-3 placeholder ``F_L(q) == 1`` for every L and q."""

    def __call__(self, L: int, q: float) -> float:
        return 1.0


#: The card-3 identity placeholder, retained as an explicit argument
#: (the card-3 tests pin it): the ratio term of the width law is
#: exactly 1 under this barrier.
IDENTITY_BARRIER: BarrierFactor = _IdentityBarrier()


#: The documented hadronic radius of the Blatt-Weisskopf barrier (plan
#: section 8.2; round 2 index item 4), GeV^-1.  A documented constant,
#: not a fit parameter.
BARRIER_R0: float = 1.0


def _blatt_weisskopf_z_coefficients(L: int) -> tuple[float, ...]:
    """Low-to-high coefficients of the standard Blatt-Weisskopf
    (von Hippel-Quigg) polynomial ``Z_L`` for ``L >= 1``.

    The polynomial is defined by the exact spherical Hankel function of
    the first kind: with ``x = sqrt(z) >= 0``,
    ``|h_L(x)|^2 = Z_L(x^2) / x^(2L+2)``, where ``h_L`` carries the
    finite series ``h_L(x) = (phase) exp(i x)/x * S_L(1/(2x))`` with
    ``S_L(t) = sum_m i^m (L+m)!/(m!(L-m)!) t^m``.  Squaring the modulus
    (the odd-power terms cancel pairwise) gives the exact construction
    below: with ``b_m = (L+m)!/(m!(L-m)!)``,

        Z_L(z) = sum_j c_{2j} / 4^j * z^(L-j),
        c_n = sum_i b_i b_{n-i} (-1)^(n/2 - i)   (n even; 0 for odd n),

    so the constant term is ``Z_L(0) = ((2L-1)!!)^2`` (1, 9, 225,
    11025 for L = 1, 2, 3, 4) and every coefficient is a non-negative
    exact integer.  The documented ``L = 1`` member ``1 + z`` and the
    reference ``D``-wave polynomial ``z^2 + 3z + 9`` are the ``L = 1``
    and ``L = 2`` members of this family, and this helper serves every
    ``L >= 2`` (the pre-registered ``(4 + 5z)/(4 + z)`` special case for
    ``L = 2`` was retired on 2026-09-28, FAB-08).
    """
    b = [
        math.factorial(L + m)
        // (math.factorial(m) * math.factorial(L - m))
        for m in range(L + 1)
    ]
    coeffs = []
    for p in range(0, L + 1):
        j = L - p
        n = 2 * j
        m = n // 2
        total = 0
        for i in range(0, L + 1):
            if 0 <= n - i <= L:
                total += b[i] * b[n - i] * (-1) ** (m - i)
        coeffs.append(total / 4**j)
    return tuple(coeffs)


class BlattWeisskopfBarrier:
    """The standard Blatt-Weisskopf polynomial barrier ``F_L(q)``.

    Documented convention (plan section 8.2; round 2 index item 4)
    with the hadronic radius ``r0`` (default ``BARRIER_R0 = 1.0
    GeV^-1``) and ``q`` in GeV::

        F_0(q) = 1
        F_1(q) = 1 / sqrt(1 + (q r0)^2)
        F_2(q) = sqrt(9 / (z^2 + 3 z + 9)),  z = (q r0)^2  (re-normalized
                 form, F_2(0) = 1)

    extended by the standard Blatt-Weisskopf polynomials (von Hippel
    and Quigg, Phys. Rev. D 5, 624 (1972)) in the same re-normalized
    squared-ratio form ``F_L(q)^2 = Z_L(0) / Z_L(z)`` with
    ``z = (q r0)^2``::

        Z_3(z) = z^3 + 6 z^2 + 45 z + 225
        Z_4(z) = z^4 + 10 z^3 + 135 z^2 + 1575 z + 11025

    (the catalog maximum ``L = 4``; every higher ``L >= 5`` through the
    same exact coefficient construction, see
    :func:`_blatt_weisskopf_z_coefficients`).

    ``F_L(0) = 1`` for every supported L, so the factor is finite and
    continuous at the channel threshold: the width law keeps the card-3
    hard zero below the threshold and the vanishing ``(q/q0)**(2L+1)``
    right limit above it.  At the pole momentum the ratio term of the
    width law is 1 by construction, so the pole normalization is
    unchanged.  A negative L is rejected with a ``ValueError`` naming
    the L (a negative orbital angular momentum is not a partial wave);
    every non-negative integer L is supported.
    """

    def __init__(self, r0: float = BARRIER_R0) -> None:
        self._r0 = float(r0)

    def __call__(self, L: int, q: float) -> float:
        L = int(L)
        if L < 0:
            raise ValueError(
                "Blatt-Weisskopf barrier supports L >= 0 (a negative "
                f"orbital angular momentum is not a partial wave); "
                f"got L = {L}"
            )
        z = (float(q) * self._r0) ** 2
        if L == 0:
            return 1.0
        if L == 1:
            return 1.0 / math.sqrt(1.0 + z)
        # L >= 2: the standard von Hippel-Quigg polynomial in the same
        # re-normalized squared-ratio form F_L(q)^2 = Z_L(0)/Z_L(z);
        # for L = 2 this is 9 / (z**2 + 3 z + 9).  The special case
        # sqrt((4 + 5 z)/(4 + z)) committed by Round 2 card 4 grew with
        # q instead of falling (corrected 2026-09-28, FAB-08).
        coeffs = _blatt_weisskopf_z_coefficients(L)
        z_polynomial = 0.0
        for coefficient in reversed(coeffs):
            z_polynomial = z_polynomial * z + coefficient
        return math.sqrt(coeffs[0] / z_polynomial)


#: The default barrier of the width law from card 4 on: the documented
#: Blatt-Weisskopf factor with ``r0 = BARRIER_R0``.
BLATT_WEISSKOPF_BARRIER: BlattWeisskopfBarrier = BlattWeisskopfBarrier()


# ----------------------------------------------------------------------
# Channel bookkeeping helpers
# ----------------------------------------------------------------------
def _masses_key(daughter_masses: Mapping[int, float]):
    """Hashable key of the daughter-mass map (pure function of values)."""
    return tuple(sorted((int(k), float(v)) for k, v in daughter_masses.items()))


def _channel_threshold(
    channel: DecayChannel, daughter_masses: Mapping[int, float]
) -> float:
    """Sum of the daughter masses (the channel threshold), GeV/c^2.

    The sum runs over all daughters, matching ``HadronCatalog``; for a
    three-body channel the threshold is therefore the full three-body
    threshold while the phase-space factor uses the leading pair.
    """
    return math.fsum(
        float(daughter_masses[int(d)]) for d in channel.daughters
    )


def _leading_pair_masses(
    channel: DecayChannel, daughter_masses: Mapping[int, float]
) -> tuple[float, float]:
    """Masses of the leading pair ``(daughters[0], daughters[1])``.

    For a two-body channel this is the decay pair; for a three-body
    channel it is the leading pair of the card's documented convention
    (card 7 may refine it without changing two-body widths).
    """
    return (
        float(daughter_masses[int(channel.daughters[0])]),
        float(daughter_masses[int(channel.daughters[1])]),
    )


# ----------------------------------------------------------------------
# Widths
# ----------------------------------------------------------------------
def channel_width(
    M: float,
    channel: DecayChannel,
    record: HadronRecord,
    daughter_masses: Mapping[int, float],
    barrier: BarrierFactor = BLATT_WEISSKOPF_BARRIER,
    catalog=None,
) -> float:
    """The mass-dependent width of one channel, GeV (plan section 8.2).

    ``Gamma_c(M) = Gamma_c,0 * (q(M)/q(M0))**(2L+1) * (M0/M)
    * (F_L(q(M))/F_L(q(M0)))**2`` with ``Gamma_c,0 = BR_c *
    nominal_width`` and ``M0 = record.pole_mass``.  The default barrier
    is the documented Blatt-Weisskopf factor
    (:data:`BLATT_WEISSKOPF_BARRIER`); pass ``IDENTITY_BARRIER`` for
    the card-3 identity law.  Returns exactly ``0.0`` at and below the
    channel threshold (hard zero), so the width is continuous at the
    threshold: the ``(q/q0)**(2L+1)`` factor vanishes there for every
    L >= 0 while the barrier stays finite (``F_L(0) = 1``).  A channel
    closed at the pole mass (``q(M0) = 0``) carries zero width by
    convention: the law is normalized at the pole, which the closed
    channel never reaches.

    With *catalog* (FAB-10, 2026-10-02), a two-body channel with a broad
    daughter takes the convolved law of :func:`_convolution`: the phase
    space integrated over the daughter's line shape, normalized at the
    pole; the channel then opens at the daughter's lower support edge.
    Every other channel keeps this law bit for bit.
    """
    M = float(M)
    conv = _convolution(record, channel, daughter_masses, barrier, catalog)
    if conv is not None:
        phi0 = conv[2]
        if not phi0 > 0.0:
            return 0.0
        phi = _convolved_phase(M, conv, barrier)
        if not phi > 0.0:
            return 0.0
        M0 = float(record.pole_mass)
        return float(channel.branching_fraction) * float(record.nominal_width) * (phi / M) / (phi0 / M0)
    threshold = _channel_threshold(channel, daughter_masses)
    if not M > threshold:
        return 0.0
    M0 = float(record.pole_mass)
    mb, mc = _leading_pair_masses(channel, daughter_masses)
    L = int(channel.orbital_angular_momentum)
    q0 = decay_momentum(M0, mb, mc)
    if not q0 > 0.0:
        # Channel closed at the pole mass: outside the pole
        # normalization, zero partial width by convention.
        return 0.0
    qm = decay_momentum(M, mb, mc)
    gamma_c0 = float(channel.branching_fraction) * float(record.nominal_width)
    phase_space = (qm / q0) ** (2 * L + 1) * (M0 / M)
    barrier_ratio = (float(barrier(L, qm)) / float(barrier(L, q0))) ** 2
    return gamma_c0 * phase_space * barrier_ratio


# ----------------------------------------------------------------------
# Quasi-two-body channels with a broad daughter (FAB-10, 2026-10-02)
# ----------------------------------------------------------------------
#: A two-body channel whose daughter is a broad resonance (a catalog record
#: with decay channels, a nominal width of at least this value and a
#: non-degenerate support) takes the convolved width law: the phase space
#: is integrated over the daughter's own normalized line shape,
#:     Phi_c(M) = int rho_d(m) q(M; m, m_other)**(2L+1) F_L(q)**2 dm,
#: and Gamma_c(M) = BR_c * nominal_width * (Phi_c(M)/M) / (Phi_c(M0)/M0),
#: the Manley-Saleski form of a quasi-two-body partial width.  The channel
#: opens at the daughter's lower support edge plus the other mass (N rho at
#: m_N + 2 m_pi), so no catalog channel is closed at the pole (external
#: review g1 B3: Delta(1600) and Delta(1700) to N rho and K1(1270) to rho K
#: and omega K had zero width at every mass).  The width law takes the
#: convolution only when the caller passes the catalog; without it the
#: pole-mass law applies bit for bit.  Narrow unstable daughters (eta) keep
#: their pole masses in the width law.
BROAD_DAUGHTER_MIN_WIDTH = 1.0e-3
_CONVOLUTION_DAUGHTER_POINTS = 1201
_CONVOLUTION_PARENT_POINTS = 4001
_CONVOLUTION_TABLE_MARGIN = 0.25
_CATALOG_TOKENS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_TOKEN_COUNTER = itertools.count(1)
_STABLE_CACHE: dict[int, frozenset] = {}
_DAUGHTER_GRID_CACHE: dict[tuple, tuple] = {}
_CONVOLUTION_CACHE: dict[tuple, tuple | None] = {}


#: The tokens of the record objects in the cache keys (2026-10-20, the probe
#: profile: hashing a whole record, its channels and provenance with it, on
#: every lookup took a quarter of the generation time).  A token stands for
#: one record object; the weak reference detects an id reused after the
#: record's release, and tokens come from the catalog counter, never reused.
_RECORD_TOKENS: dict[int, tuple] = {}


def _record_token(record) -> int:
    """The cache token of one record object."""
    entry = _RECORD_TOKENS.get(id(record))
    if entry is None or entry[0]() is not record:
        entry = (weakref.ref(record), next(_TOKEN_COUNTER))
        _RECORD_TOKENS[id(record)] = entry
    return entry[1]


def _catalog_token(catalog) -> int | None:
    """A cache token of a catalog object, or ``None``.  Tokens come from
    a counter and are never reused, so a cache entry of a released
    catalog cannot be served to a new one."""
    if catalog is None:
        return None
    token = _CATALOG_TOKENS.get(catalog)
    if token is None:
        token = next(_TOKEN_COUNTER)
        _CATALOG_TOKENS[catalog] = token
    return token


def broad_daughter_record(catalog, pdg_id: int) -> HadronRecord | None:
    """The catalog record of *pdg_id* when it is a broad resonance under
    :data:`BROAD_DAUGHTER_MIN_WIDTH`, else ``None`` (stable states, narrow
    states, records outside the catalog, or no catalog)."""
    if catalog is None:
        return None
    token = _catalog_token(catalog)
    stable = _STABLE_CACHE.get(token)
    if stable is None:
        stable = frozenset(int(p) for p in catalog.stable_states())
        _STABLE_CACHE[token] = stable
    if int(pdg_id) in stable:
        return None
    try:
        record = catalog.get(int(pdg_id))
    except KeyError:
        return None
    if not record.decay_channels or not float(record.nominal_width) >= BROAD_DAUGHTER_MIN_WIDTH:
        return None
    if not float(record.mass_range[1]) > float(record.mass_range[0]):
        return None
    return record


def _pole_mass_map(catalog, record: HadronRecord) -> dict[int, float]:
    """The pole masses of a record's channel daughters: the catalog
    record, else the vendored PDG table (the decayer's daughter-mass
    map, ``ResonanceDecayer._daughter_mass_map``)."""
    from .pdg_database import get_particle

    masses: dict[int, float] = {}
    for channel in record.decay_channels:
        for d in channel.daughters:
            d = int(d)
            if d in masses:
                continue
            try:
                masses[d] = float(catalog.get(d).pole_mass)
            except KeyError:
                entry = get_particle(d)
                if entry is None:
                    raise KeyError(f"daughter pdg_id {d} is neither a catalog record nor in the PDG table") from None
                masses[d] = float(entry.mass)
    return masses


def _int_power_vec(x: np.ndarray, n: int) -> np.ndarray:
    """``x**n`` elementwise for a non-negative integer *n*, by repeated
    multiplication.  The array power of numpy dispatches to SIMD kernels
    that differ between CPUs in the last bit (the gateway workers and an
    AVX-512 host disagree on 26% of ``q**3``); products are exact IEEE
    operations, so the tables are the same on every machine."""
    out = np.ones_like(np.asarray(x, dtype=float))
    for _ in range(int(n)):
        out = out * x
    return out


def _barrier_squared_vec(barrier: BarrierFactor, L: int, q: np.ndarray) -> np.ndarray:
    """``F_L(q)**2`` elementwise (the closed forms of the two module
    barriers; any other barrier element by element)."""
    q = np.asarray(q, dtype=float)
    if barrier is IDENTITY_BARRIER or L == 0:
        return np.ones_like(q)
    if isinstance(barrier, BlattWeisskopfBarrier):
        qr = q * barrier._r0
        z = qr * qr
        if L == 1:
            return 1.0 / (1.0 + z)
        coeffs = _blatt_weisskopf_z_coefficients(int(L))
        poly = np.zeros_like(z)
        for coefficient in reversed(coeffs):
            poly = poly * z + coefficient
        return coeffs[0] / poly
    return np.asarray([float(barrier(int(L), float(v))) ** 2 for v in q])


def _decay_momentum_vec(M, m1, m2) -> np.ndarray:
    """``q(M; m1, m2)`` elementwise, 0 where the split is closed."""
    M = np.asarray(M, dtype=float)
    m1 = np.asarray(m1, dtype=float)
    m2 = np.asarray(m2, dtype=float)
    s12, d12 = m1 + m2, m1 - m2
    lam = (M * M - s12 * s12) * (M * M - d12 * d12)
    return np.where((M > m1 + m2) & (lam > 0.0), np.sqrt(np.maximum(lam, 0.0)) / (2.0 * M), 0.0)


def _daughter_grid(catalog, record: HadronRecord, barrier: BarrierFactor) -> tuple:
    """The broad daughter's support grid, its normalized line shape at the
    nodes, and the normalized trapezoid weights (cached per catalog)."""
    key = (barrier, _catalog_token(catalog), int(record.pdg_id))
    cacheable = barrier in _CACHED_BARRIERS
    if cacheable and key in _DAUGHTER_GRID_CACHE:
        return _DAUGHTER_GRID_CACHE[key]
    masses = _pole_mass_map(catalog, record)
    ms = np.linspace(float(record.mass_range[0]), float(record.mass_range[1]), _CONVOLUTION_DAUGHTER_POINTS)
    rho = np.asarray([spectral_function(float(m), record, masses, barrier, catalog=catalog) for m in ms])
    w = np.full(ms.size, ms[1] - ms[0])
    w[0] *= 0.5
    w[-1] *= 0.5
    w = w * rho
    total = float(w.sum())
    if not (math.isfinite(total) and total > 0.0):
        raise ValueError(f"the broad daughter {record.name} carries no line-shape mass on its support")
    entry = (ms, rho, w / total)
    if cacheable:
        _DAUGHTER_GRID_CACHE[key] = entry
    return entry


def _convolution(record: HadronRecord, channel: DecayChannel, daughter_masses: Mapping[int, float],
                 barrier: BarrierFactor, catalog):
    """The convolved phase-space record of a two-body channel with one
    broad daughter, or ``None`` for a channel without one (the pole-mass
    law applies): ``(Ms, phi, phi0, ms, w, m_other, L)``."""
    if catalog is None or getattr(channel, "is_three_body", False) or len(channel.daughters) != 2:
        return None
    key = (barrier, _catalog_token(catalog), _record_token(record), tuple(int(d) for d in channel.daughters),
           int(channel.orbital_angular_momentum), _masses_key(daughter_masses))
    cacheable = barrier in _CACHED_BARRIERS
    if cacheable and key in _CONVOLUTION_CACHE:
        return _CONVOLUTION_CACHE[key]
    d0, d1 = (int(d) for d in channel.daughters)
    b0, b1 = broad_daughter_record(catalog, d0), broad_daughter_record(catalog, d1)
    if b0 is None and b1 is None:
        value = None
    elif b0 is not None and b1 is not None:
        raise NotImplementedError(
            f"{record.name}: channel {(d0, d1)} has two broad daughters; the convolved width law "
            "covers one (the committed catalog carries none with two)")
    else:
        broad, other = (b0, d1) if b0 is not None else (b1, d0)
        m_other = float(daughter_masses[int(other)])
        ms, _rho, w = _daughter_grid(catalog, broad, barrier)
        L = int(channel.orbital_angular_momentum)
        M0 = float(record.pole_mass)
        lo = float(ms[0]) + m_other
        hi = max(float(record.mass_range[1]), M0) + _CONVOLUTION_TABLE_MARGIN
        Ms = np.linspace(lo, hi, _CONVOLUTION_PARENT_POINTS)
        q = _decay_momentum_vec(Ms[:, None], ms[None, :], m_other)
        phi = (w[None, :] * _int_power_vec(q, 2 * L + 1) * _barrier_squared_vec(barrier, L, q)).sum(axis=1)
        # The pole value is read from the same table, so Gamma_c(M0) is
        # BR_c * nominal_width to rounding.
        phi0 = float(np.interp(M0, Ms, phi)) if M0 > lo else 0.0
        value = (Ms, phi, phi0, ms, w, m_other, L)
    if cacheable:
        _CONVOLUTION_CACHE[key] = value
    return value


def _convolved_phase(M: float, conv, barrier: BarrierFactor) -> float:
    """``Phi_c(M)`` from the table (linear interpolation), or directly
    above the tabulated range."""
    Ms, phi, _phi0, ms, w, m_other, L = conv
    if not M > Ms[0]:
        return 0.0
    if M <= Ms[-1]:
        return float(np.interp(M, Ms, phi))
    q = _decay_momentum_vec(M, ms, m_other)
    return float((w * _int_power_vec(q, 2 * L + 1) * _barrier_squared_vec(barrier, L, q)).sum())


def draw_broad_daughter_mass(
    parent_mass: float,
    other_mass: float,
    L: int,
    daughter_record: HadronRecord,
    catalog,
    rng: np.random.Generator,
    barrier: BarrierFactor = BLATT_WEISSKOPF_BARRIER,
) -> float | None:
    """The joint daughter-mass draw of a quasi-two-body decay (FAB-10): the
    broad daughter's mass from ``rho_d(m) q(M; m, m_other)**(2L+1)
    F_L(q)**2`` on ``[lo_d, min(hi_d, M - m_other)]``, the integrand of the
    convolved width.  Two uniforms per non-None draw (the cell, then the
    position within it), as :func:`draw_mass`; ``None`` and no draw when
    the window carries no mass."""
    ms, rho, _w = _daughter_grid(catalog, daughter_record, barrier)
    hi_w = min(float(ms[-1]), float(parent_mass) - float(other_mass))
    if not hi_w > float(ms[0]):
        return None
    inside = ms < hi_w
    xs = np.concatenate([ms[inside], [hi_w]])
    ys_rho = np.concatenate([rho[inside], [float(np.interp(hi_w, ms, rho))]])
    q = _decay_momentum_vec(float(parent_mass), xs, float(other_mass))
    ys = np.maximum(ys_rho * _int_power_vec(q, 2 * int(L) + 1) * _barrier_squared_vec(barrier, int(L), q), 0.0)
    cell = 0.5 * (ys[1:] + ys[:-1]) * np.diff(xs)
    total = float(cell.sum())
    if not (math.isfinite(total) and total > 0.0):
        return None
    cdf = np.cumsum(cell) / total
    u_segment = rng.random()
    idx = int(np.searchsorted(cdf, u_segment, side="right"))
    if idx >= cdf.size:
        idx = cdf.size - 1
    u_within = rng.random()
    mass = xs[idx] + u_within * (xs[idx + 1] - xs[idx])
    return float(min(max(mass, xs[0]), hi_w))


def total_width(
    M: float,
    record: HadronRecord,
    daughter_masses: Mapping[int, float],
    barrier: BarrierFactor = BLATT_WEISSKOPF_BARRIER,
    catalog=None,
) -> float:
    """The channel-summed mass-dependent width ``Gamma(M)``, GeV.

    ``Gamma(M) = sum_c Gamma_c(M)`` over the record's channels; the
    record-level normalization ``Gamma(M0) == nominal_width`` holds
    when the channel branching fractions sum to 1 (the law normalizes
    at the pole by construction).
    """
    return math.fsum(
        channel_width(M, channel, record, daughter_masses, barrier, catalog)
        for channel in record.decay_channels
    )


# ----------------------------------------------------------------------
# Spectral function
# ----------------------------------------------------------------------
def _unnormalized_shape(
    M: float,
    record: HadronRecord,
    daughter_masses: Mapping[int, float],
    barrier: BarrierFactor,
    catalog=None,
) -> float:
    """The unnormalized relativistic Breit-Wigner shape at mass M.

    The production spectral function carries one power of the
    mass-dependent width in the numerator, ``M**2 Gamma(M)`` over the
    propagator denominator ``(M**2 - M0**2)**2 + M0**2 Gamma(M)**2``
    (the transport-model form and the ``legacy_v1`` ``bw_new``
    convention ``x * rmas * gamma(x)``).  The squared-width numerator
    committed by Round 2 card 3 is the formation cross-section shape
    ``Gamma_in Gamma_out / |D|**2`` of a resonance formed and decaying
    through the same channel; it multiplied the high-mass tail by
    ``Gamma(M)/Gamma_0`` and put the Delta(1232) mean at 1.62 GeV
    (corrected 2026-09-28, FAB-09).
    """
    M = float(M)
    gamma = total_width(M, record, daughter_masses, barrier, catalog)
    if not gamma > 0.0:
        return 0.0
    M0 = float(record.pole_mass)
    return (
        M * M * gamma
        / ((M * M - M0 * M0) ** 2 + M0 * M0 * gamma * gamma)
    )


def _trapezoid(xs: np.ndarray, ys: np.ndarray) -> float:
    """Trapezoid integral of *ys* over the abscissae *xs*."""
    if xs.size < 2:
        return 0.0
    return float((0.5 * (ys[1:] + ys[:-1]) * (xs[1:] - xs[:-1])).sum())


#: Pure compute cache of the support normalization, keyed on
#: ``(barrier, record, masses_key)`` and used for the module-level
#: barrier singletons (identity and Blatt-Weisskopf).  A stored value
#: is the trapezoid integral of the unnormalized shape on the fixed
#: 10,001-node normalization grid over the support.
_NORMALIZATION_CACHE: dict[tuple, float] = {}

#: The barrier singletons whose 1024-point evaluations are cached; the
#: cache key carries the barrier identity, so the two barriers never
#: share an entry.
_CACHED_BARRIERS = (IDENTITY_BARRIER, BLATT_WEISSKOPF_BARRIER)


def _normalization(
    record: HadronRecord,
    daughter_masses: Mapping[int, float],
    barrier: BarrierFactor,
    catalog=None,
) -> float:
    """Integral of the unnormalized shape over the support.

    The integral is the trapezoid of the shape on the fixed
    10,001-node normalization grid
    (:data:`_NORMALIZATION_GRID_POINTS`) over ``[mass_range[0],
    mass_range[1]]``; it is 0.0 for the degenerate support
    ``mass_range[1] <= mass_range[0]`` (no line-shape mass anywhere,
    e.g. a stable record).  The grid is the pre-registered reference
    quadrature of the round-2 check suite, so the renormalized density
    integrates to 1 on the grid that measures it (the card-8 check
    round's re-open of the 1024-point convention).
    """
    key = (barrier, _record_token(record), _masses_key(daughter_masses))
    if catalog is not None:
        key = key + (_catalog_token(catalog),)
    cacheable = barrier in _CACHED_BARRIERS
    if cacheable and key in _NORMALIZATION_CACHE:
        return _NORMALIZATION_CACHE[key]
    lo, hi = float(record.mass_range[0]), float(record.mass_range[1])
    if hi > lo:
        xs = np.linspace(lo, hi, _NORMALIZATION_GRID_POINTS)
        ys = np.array(
            [
                _unnormalized_shape(float(x), record, daughter_masses, barrier, catalog)
                for x in xs
            ]
        )
        value = _trapezoid(xs, ys)
    else:
        value = 0.0
    if cacheable:
        _NORMALIZATION_CACHE[key] = value
    return value


def spectral_function(
    M: float,
    record: HadronRecord,
    daughter_masses: Mapping[int, float],
    barrier: BarrierFactor = BLATT_WEISSKOPF_BARRIER,
    catalog=None,
) -> float:
    """The normalized relativistic Breit-Wigner line shape ``rho(M)``.

    ``rho(M) proportional to M**2 Gamma(M) / ((M**2 - M0**2)**2 +
    M0**2 Gamma(M)**2)`` with the channel-summed ``Gamma(M)``, on the
    support ``[threshold, m_up]`` (the record's ``mass_range``) and
    exactly ``0.0`` outside it.  The renormalization makes the
    integral over the support equal 1 under the documented
    10,001-node normalization quadrature
    (:data:`_NORMALIZATION_GRID_POINTS`, the pre-registered reference
    grid of the round-2 check suite): the normalization constant is
    that grid's trapezoid integral of the unnormalized shape, so the
    renormalized density reproduces the unit integral on the same grid
    to float rounding.  The density vanishes at the lower support
    endpoint, where every channel width is the hard zero at the
    threshold.
    """
    M = float(M)
    lo, hi = float(record.mass_range[0]), float(record.mass_range[1])
    if not lo <= M <= hi:
        return 0.0
    norm = _normalization(record, daughter_masses, barrier, catalog)
    if not norm > 0.0:
        return 0.0
    return _unnormalized_shape(M, record, daughter_masses, barrier, catalog) / norm


# ----------------------------------------------------------------------
# Truncated, explicitly renormalized mass draw
# ----------------------------------------------------------------------
#: Pure compute cache of the per-window draw tables (module-level
#: barrier singletons only), keyed on
#: ``(barrier, record, masses_key, lo, hi)``; an entry is the tuple
#: ``(lo_c, hi_c, cdf, cell_width)`` or ``None`` for a window
#: that carries no line-shape mass.
_DRAW_CACHE: dict[tuple, tuple | None] = {}


def _draw_entry(
    record: HadronRecord,
    daughter_masses: Mapping[int, float],
    lo: float,
    hi: float,
    barrier: BarrierFactor,
    catalog=None,
) -> tuple | None:
    """Build the inverse-CDF draw table for the clipped window.

    Returns ``None`` when the window carries no line-shape mass (it
    does not intersect the support, or the restricted integral
    vanishes).
    """
    support_lo, support_hi = (
        float(record.mass_range[0]),
        float(record.mass_range[1]),
    )
    lo_c = max(lo, support_lo)
    hi_c = min(hi, support_hi)
    if not hi_c > lo_c:
        return None
    xs = np.linspace(lo_c, hi_c, _DRAW_GRID_POINTS)
    ys = np.array(
        [
            spectral_function(float(x), record, daughter_masses, barrier, catalog)
            for x in xs
        ]
    )
    ys = np.maximum(ys, 0.0)
    total = _trapezoid(xs, ys)
    if not (math.isfinite(total) and total > 0.0):
        return None
    # Per-cell masses of the piecewise-linear density interpolant; the
    # normalization cancels in the CDF.
    cell = 0.5 * (ys[1:] + ys[:-1]) * (xs[1:] - xs[:-1])
    cdf = np.cumsum(cell) / total
    cell_width = (hi_c - lo_c) / cdf.size
    return (lo_c, hi_c, cdf, cell_width)


def draw_mass(
    M_window: tuple[float, float],
    record: HadronRecord,
    daughter_masses: Mapping[int, float],
    rng: np.random.Generator,
    barrier: BarrierFactor = BLATT_WEISSKOPF_BARRIER,
    catalog=None,
) -> float | None:
    """Draw one mass from ``rho`` restricted to ``M_window``.

    The truncated density is normalized by construction (the inverse
    CDF of the line shape integrated over the window, on the fixed
    1024-point grid), mirroring ``draw_truncated_bw`` of
    ``na49_mixing.breit_wigner``.  Two uniform draws are consumed per
    non-None draw (segment choice plus within-cell position) and none
    for a ``None`` result.  Returns ``None`` when the window carries
    no line-shape mass.
    """
    lo = float(M_window[0])
    hi = float(M_window[1])
    key = (barrier, _record_token(record), _masses_key(daughter_masses), lo, hi)
    if catalog is not None:
        key = key + (_catalog_token(catalog),)
    cacheable = barrier in _CACHED_BARRIERS
    if cacheable and key in _DRAW_CACHE:
        entry = _DRAW_CACHE[key]
    else:
        entry = _draw_entry(record, daughter_masses, lo, hi, barrier, catalog)
        if cacheable:
            _DRAW_CACHE[key] = entry
    if entry is None:
        return None
    lo_c, hi_c, cdf, cell_width = entry
    u_segment = rng.random()
    idx = int(np.searchsorted(cdf, u_segment, side="right"))
    if idx >= cdf.size:
        idx = cdf.size - 1
    u_within = rng.random()
    mass = lo_c + (idx + u_within) * cell_width
    return float(min(max(mass, lo_c), hi_c))
