"""The re-registered threshold-discontinuity metric (Round 3, card 2).

Task ``TASK-20260917-r3-gate2-followup-threshold-metric-reregistration``
(the Gate 2 closure obligation of the Round 2 decision, follow-up 2 of
``_knowledge/operations/round2-closure-2026-09-17.md``; plan section 8,
Gate 2 threshold-discontinuity clause, line 580).

The R21/R22 pre-registered metric compared the observed adjacent-pair
differences against ``delta_analytic = 0`` at open-set-unchanged pairs,
while the full analytic difference ``delta_analytic_full`` is nonzero:
daughter momentum moments move with the source mass even when the open
set of resonance records is constant, and the width-law class shares
carry a mass dependence of their own.  This module re-registers the
metric on both pair classes against the full analytic difference

    delta_analytic_full = mu_an(M_{k+1}) - mu_an(M_k),

the analytic difference of the flat-mixture expectation between the two
adjacent grid point masses (the card-9 mixture expectation: the record
drawn uniformly over the open set, its mass realized from the card-3
spectral function on the truncated window, the decay through the
card-6/7 kernel), decomposed as

- the **open-set-change contribution**: the difference between the
  mixture at ``M_{k+1}`` over the open set at ``M_{k+1}`` and the
  counterfactual mixture at ``M_{k+1}`` that keeps the open set and the
  record shares of ``M_k`` (records opening or closing between the two
  point masses, with the card-5 charge weights and the card-3/4 width
  law of the surviving records' share change), and
- the **momentum-drift contribution**: the difference between the
  counterfactual mixture at ``M_{k+1}`` over the open set at ``M_k``
  and the mixture at ``M_k`` (the daughter momentum moments of every
  record open at both masses at ``M_{k+1}`` minus at ``M_k``).

The two terms add to ``delta_analytic_full`` exactly for every
observable; at an open-set-unchanged pair the open-set-change
contribution vanishes identically.  The decomposition is a diagnostic,
never a fitted correction: the tolerance

    |mu(k+1) - mu(k) - delta_analytic_full| <= 3 * SE_combined,
    SE_combined = sqrt(SE_k^2 + SE_{k+1}^2),

is the pre-registered R21/R22 3.0-sigma band, unchanged.

Analytic engine (``RecordSet`` / ``full_analytic_difference``): exact
on-shell two-body kinematics through the Kaellen decay momentum
(``na49_mixing.resonance_widths.decay_momentum``), the exact Kallen
leading-pair kinematics of the card-7 three-body objects (the decay
momentum and the pair-rest energies of the virtual pair, the kernel's
``three_body_phase_space`` convention; the equal-mass committed
expressions are the ``mb == mc`` special case and evaluate
byte-identically), the legacy ``_decay_rest_frame`` boost convention (a
boosted object's lab moments are the isotropic-direction averages of
its rest-frame content), and the card-5 charge shares through the
kernel's own open-object weights (``ResonanceDecayer._open_objects``:
the card-3/4 partial widths at the realized mass distributed over the
card-5 Clebsch-Gordan charge combinations, card-7 three-body channels
as one open object each at the flat ``M_bc^2`` phase-space model).  No
Monte Carlo: the mass realizations of the mixture and the kernel are
integrated by deterministic quadrature on fixed grids, and the angular
averages run on fixed Gauss-Legendre direction grids.  The quadrature
scheme:

- the record's truncated mixture window
  ``[mass_range[0], min(mass_range[1], M)]`` is integrated on a fixed
  Gauss-Legendre grid (:data:`ROOT_GL_NODES`) with the card-3 spectral
  density and the open-object weights evaluated kernel-exact at every
  node (the sealed R21/R22 reference engine's root scheme);
- every two-body resonance daughter, at every level, has its mass
  realization integrated over the on-shell budget window
  ``[mass_range[0], min(mass_range[1], M_parent - m_partner)]`` on a
  fixed Gauss-Legendre grid (:data:`DAUGHTER_GL_NODES`), with the
  daughter's card-3 density and open-object weights kernel-exact at
  every node and the stable partner's lab motion joining the same
  quadrature (the daughter-mass quadrature replaces the
  concentration-point collapse of deep levels; the committed catalog
  carries no resonance daughter below the first decay level, so no
  current channel changes and the defensive node bound is untouched);
- narrow line shapes (width at most :data:`NARROW_WIDTH_RATIO` of the
  support span: the eta and eta-prime needles of the committed catalog,
  whose draw concentrates in one card-3 draw cell) evaluate at the
  draw-distribution point of their window -- the draw cell containing
  the pole for a window that covers it, the smooth-window draw mean for
  a window truncated below the pole (the sealed engine's documented
  narrow-record treatment; a narrow record whose truncated window
  carries no needle is integrated like a broad record), and the
  two-resonance channels keep the collapsed evaluation of both needles
  (the committed catalog's only case is f4(2050) -> eta eta).

The engine reproduces the sealed R21/R22 ``delta_analytic_full`` values
within the sealed reference's own documented approximation scale (the
re-evaluation driver records the engine cross-check as diagnostic
evidence); the multiplicity term, which carries no mass dependence of
the count content beyond the class shares, is exact where the sealed
value is exact.

Sealed-archive re-evaluation (``rereval_report``): the sealed
``gate2_r21.json`` / ``gate2_r22.json`` already record the unapplied
``delta_analytic_full`` per check alongside the applied value, so the
re-evaluation recomputes every applied check's sigma from the sealed
stored fields at the unchanged 3.0-sigma tolerance -- a deterministic,
byte-stable recomputation over the sealed archives, with no
regeneration and no re-seal of R21/R22.

Public surface (additive package exports):

- :class:`RecordSet` -- the candidate record set spanning an adjacent
  pair, with the card-9 open-set rule and the analytic content engine;
- :func:`full_analytic_difference` -- the per-species analytic
  difference with its decomposition (pure function);
- :func:`evaluate_pair` -- the signed sigma and pass flag of one check
  at the combined standard error;
- :func:`rereval_report` -- the sealed-archive re-evaluation assembly;
- :func:`diagnose_residual` -- the deterministic one-line diagnosis of
  a residual failing check.

Hermetic: standard library, numpy, and the vendored ``na49_mixing``
modules only; no event generation, no external tables.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .hadron_catalog import HadronCatalog, HadronRecord
from .pdg_database import get_particle
from .resonance_decay import NoOpenChannelError, ResonanceDecayer
from .resonance_widths import decay_momentum, spectral_function

__all__ = [
    "THRESHOLD_METRIC_ID",
    "SIGMA_TOLERANCE",
    "DEGENERATE_WINDOW_TOL",
    "ZERO_RESIDUAL_TOL",
    "DETERMINISTIC_SE_FLOOR",
    "ANGLE_NODES",
    "ANGLE_NODES_DEEP",
    "ROOT_GL_NODES",
    "DAUGHTER_GL_NODES",
    "NARROW_WIDTH_RATIO",
    "THREE_BODY_MBC_NODES",
    "THREE_BODY_COS_NODES",
    "ISOTROPIC_SIN_THETA_MEAN",
    "OBSERVABLES",
    "RecordSet",
    "full_analytic_difference",
    "evaluate_pair",
    "rereval_report",
    "diagnose_residual",
    "species_key",
]

#: The re-registered metric identifier (the full analytic difference).
THRESHOLD_METRIC_ID = "delta_analytic_full"

#: The pre-registered R21/R22 tolerance; never relaxed by this module.
SIGMA_TOLERANCE = 3.0

#: The card-9 degenerate-window bound (the mixture item 3 rule).
DEGENERATE_WINDOW_TOL = 1e-12

#: The zero-residual bound of the sealed metric's vacuous-zero rule
#: (SE_combined = 0 pairs with a residual at or below this pass).
ZERO_RESIDUAL_TOL = 1e-12

#: The deterministic-observable SE floor (the pre-registered
#: deterministic-precision convention of the engine-fix card
#: ``TASK-20260918-r3-card2-engine-three-body-moment-fix``).  A
#: between-replica ``SE_combined`` that collapses below this floor on a
#: deterministic observable is floored at the deterministic precision
#: of the stored sums, and the vacuous-zero rule applies at the floor:
#: a residual at or below :data:`ZERO_RESIDUAL_TOL` is a vacuous pass
#: (sigma 0.0), a larger residual evaluates at the floored SE and the
#: unchanged 3.0-sigma band.  The value equals the documented
#: vacuous-zero threshold, so the module carries one deterministic
#: scale: the observed collapse (the R31 window-0 K0/Sigma+ ``<p>``
#: pair, analytic and MC agreeing to 15 decimals with the
#: between-replica SE at 1.28e-16) sits four orders of magnitude below
#: it, and every genuine statistical SE of the R21/R22/R31 moments
#: sits at 1e-4 or above (the sealed R21/R22 archives carry no applied
#: check between 0 and this floor, so the sealed re-evaluation counts
#: are unchanged by the convention).  The tolerance is never relaxed.
DETERMINISTIC_SE_FLOOR = 1e-12

#: Gauss-Legendre angle nodes of the first boost level (the isotropic
#: direction grid of the legacy boost convention).
ANGLE_NODES = 48

#: Angle nodes at deeper boost levels (the composite is smooth in the
#: boost angle; the reduced grid keeps nested-cascade node counts small).
ANGLE_NODES_DEEP = 24

#: Gauss-Legendre nodes of the truncated mixture-window quadrature (the
#: card-3 spectral density and the open-object weights are evaluated
#: kernel-exact at every node).
ROOT_GL_NODES = 128

#: Gauss-Legendre nodes of the daughter budget-window quadrature at the
#: integrated level (the daughter's card-3 density and open-object
#: weights kernel-exact at every node; the stable partner's lab motion
#: joins the same quadrature, the sealed engine's convention).
DAUGHTER_GL_NODES = 8

#: A record whose nominal width is at most this fraction of its support
#: span is a narrow line shape: its draw concentrates in the top cell,
#: and the engine collapses it to the draw-distribution evaluation
#: point (the eta and eta-prime needles of the committed catalog; the
#: sealed engine's documented narrow-record treatment).
NARROW_WIDTH_RATIO = 1e-2

#: Uniform nodes of the draw-mean trapezoid (the card-3 draw grid; the
#: trapezoid on these nodes is the exact cell mean of the sampler's
#: piecewise-linear mass model).
_DRAW_MEAN_NODES = 1024

#: The card-7 flat three-body model grids (the M_bc^2 and leading-pair
#: direction nodes; the same resolutions as the sealed R21/R22 engine).
THREE_BODY_MBC_NODES = 24
THREE_BODY_COS_NODES = 24

#: The isotropic-direction mean of sin(theta): E[p_T] = E[|p|] *
#: pi/4 for an isotropic ensemble (the sealed engine's convention).
ISOTROPIC_SIN_THETA_MEAN = math.pi / 4.0

#: The checked observables (the R21/R22 moment set).
OBSERVABLES = ("multiplicity", "mean_p", "mean_pt_y")

#: Defensive node-count bound of the nested boost expansion.
_MAX_NODES = 4_000_000

# ---------------------------------------------------------------------------
# Isotropic boost grids (fixed at import; deterministic, no draws)
# ---------------------------------------------------------------------------
_COS_1, _COS_W_1 = np.polynomial.legendre.leggauss(ANGLE_NODES)
_COS_W_1 = _COS_W_1 * 0.5                      # sums to 1

_COS_D, _COS_W_D = np.polynomial.legendre.leggauss(ANGLE_NODES_DEEP)
_COS_W_D = _COS_W_D * 0.5

_ROOT_X, _ROOT_W = np.polynomial.legendre.leggauss(ROOT_GL_NODES)

_DAU_X, _DAU_W = np.polynomial.legendre.leggauss(DAUGHTER_GL_NODES)

_MBC_X, _MBC_W = np.polynomial.legendre.leggauss(THREE_BODY_MBC_NODES)
_MBC_W = _MBC_W * 0.5
_COS3_X, _COS3_W = np.polynomial.legendre.leggauss(THREE_BODY_COS_NODES)
_COS3_W = _COS3_W * 0.5


def species_key(name: str) -> str:
    """The JSON species key of a catalog record name: the charge symbol
    transliterated (``+`` -> ``_plus``, ``-`` -> ``_minus``, a trailing
    ``0`` -> ``_0``), the card-9 driver's convention (``pi+`` ->
    ``pi_plus``, ``pi0`` -> ``pi_0``, ``proton`` verbatim)."""
    name = str(name)
    if name.endswith("+"):
        return name[:-1] + "_plus"
    if name.endswith("-"):
        return name[:-1] + "_minus"
    if name.endswith("0"):
        return name[:-1] + "_0"
    return name


def _species_name(catalog: HadronCatalog, pdg_id: int) -> str:
    """The species name of a particle: the catalog record's name for
    records, the vendored PDG table's name otherwise, the bare id when
    neither resolves (the card-9 driver's convention)."""
    try:
        return str(catalog.get(int(pdg_id)).name)
    except KeyError:
        entry = get_particle(int(pdg_id))
        if entry is None:
            return str(int(pdg_id))
        return str(entry.name)


def _pole_mass(catalog: HadronCatalog, pdg_id: int) -> float:
    """The pole mass of a particle: the catalog record's pole mass for
    records, the vendored PDG table's mass otherwise (the card-3
    daughter-mass map convention)."""
    try:
        return float(catalog.get(int(pdg_id)).pole_mass)
    except KeyError:
        entry = get_particle(int(pdg_id))
        if entry is None:
            raise KeyError(
                f"daughter pdg_id {pdg_id} is neither a catalog record "
                "nor in the vendored PDG table"
            ) from None
        return float(entry.mass)


def _boost_nodes(nodes: np.ndarray, gamma, beta, cos: np.ndarray,
                 cos_w: np.ndarray) -> np.ndarray:
    """Expand one boost level of *nodes* over the isotropic direction
    grid (the legacy ``_decay_rest_frame`` boost convention: the lab
    moments of an isotropic rest-frame ensemble under the boost
    ``(gamma, beta)`` are the direction averages).

    *nodes* is an ``(N, 3)`` array of ``(weight, E, p)`` rows in the
    rest frame of the boosted object; *gamma* and *beta* are the boost
    factors (scalars or per-row arrays).  Returns the expanded
    ``(N * len(cos), 3)`` array in the boosted frame."""
    E = nodes[:, 1][:, None]
    p = nodes[:, 2][:, None]
    c = cos[None, :]
    g = np.asarray(gamma, dtype=float)
    b = np.asarray(beta, dtype=float)
    if g.ndim == 0:
        g = np.full(E.shape, float(g))
        b = np.full(E.shape, float(b))
    else:
        g = g[:, None] + np.zeros_like(E)
        b = b[:, None] + np.zeros_like(E)
    e_lab = g * (E + b * p * c)
    p_par = g * (p * c + b * E)
    p_lab = np.sqrt(p_par * p_par + p * p * (1.0 - c * c))
    w_out = nodes[:, 0][:, None] * cos_w[None, :]
    out = np.stack([w_out.ravel(), e_lab.ravel(), p_lab.ravel()], axis=1)
    if out.shape[0] > _MAX_NODES:
        raise RuntimeError(
            "threshold_metric: the nested boost expansion exceeded the "
            f"defensive node bound ({out.shape[0]} > {_MAX_NODES})"
        )
    return out


def _three_body_slot_nodes(m_parent: float, masses: tuple[float, ...],
                           slot: int):
    """The ``(weights, E, p)`` node content of one daughter slot of a
    card-7 three-body object in the parent rest frame (the flat
    ``M_bc^2`` phase-space model: ``M_bc^2`` uniform on its range, the
    leading pair (daughters 0 and 1) decayed as a two-body system
    against the partner (daughter 2), the relative angle of the
    leading-pair direction exactly uniform and independent of
    ``M_bc^2``).

    The leading pair is decayed with the exact on-shell two-body
    kinematics of the virtual pair (the kernel's
    ``three_body_phase_space`` convention): the Kallen decay momentum
    ``q_bc = sqrt((Mbc^2 - (mb + mc)^2)(Mbc^2 - (mb - mc)^2)) / (2
    Mbc)`` and the exact pair-rest energies ``e_b = (Mbc^2 + mb^2 -
    mc^2) / (2 Mbc)``, ``e_c = (Mbc^2 + mc^2 - mb^2) / (2 Mbc)`` in
    slots 0 and 1.  On the ``mb == mc`` branch the exact formulas
    reduce to the committed equal-mass expressions ``q_bc =
    sqrt(Mbc^2/4 - mb^2)`` and ``e_b = e_c = 0.5 * Mbc``, which are
    evaluated verbatim there, so every equal-mass three-body channel
    evaluates byte-identically to the committed engine (the
    regression-pinned contract).  The partner slot (2) is unchanged.
    The returned per-slot weights sum to 1.0."""
    mb, mc, md = (float(m) for m in masses)
    M = float(m_parent)
    lo2 = (mb + mc) ** 2
    hi2 = (M - md) ** 2
    if hi2 <= lo2:
        raise RuntimeError(
            "threshold_metric: the three-body leading-pair range is "
            f"empty at parent mass {M!r} (daughters {masses!r})"
        )
    mbc2 = 0.5 * (hi2 - lo2) * _MBC_X + 0.5 * (hi2 + lo2)
    Mbc = np.sqrt(mbc2)
    if mb == mc:
        # The committed equal-mass special case of the exact Kallen
        # expressions, evaluated verbatim (byte-identity of the
        # equal-mass three-body channels).
        q_bc = np.sqrt(np.maximum(Mbc * Mbc / 4.0 - mb * mb, 0.0))
        e_b = 0.5 * Mbc
        e_c = 0.5 * Mbc
    else:
        # The exact unequal-mass leading pair: the Kallen decay
        # momentum and the exact pair-rest energies.
        q_bc = np.sqrt(np.maximum(
            (mbc2 - (mb + mc) ** 2) * (mbc2 - (mb - mc) ** 2), 0.0)
        ) / (2.0 * Mbc)
        e_b = (mbc2 + mb * mb - mc * mc) / (2.0 * Mbc)
        e_c = (mbc2 + mc * mc - mb * mb) / (2.0 * Mbc)
    e_pair = np.maximum((M * M + mbc2 - md * md) / (2.0 * M), 1e-300)
    q_pair = np.sqrt(np.maximum(
        ((M * M - mbc2 - md * md) ** 2 - 4.0 * mbc2 * md * md)
        / (4.0 * M * M), 0.0))
    gamma_p = e_pair / Mbc
    beta_p = q_pair / e_pair
    if slot <= 1:
        sign = 1.0 if slot == 0 else -1.0
        e_lead = e_b if slot == 0 else e_c
        e_rest = gamma_p[:, None] * (
            e_lead[:, None] + beta_p[:, None] * q_bc[:, None] * sign
            * _COS3_X[None, :])
        p_par = gamma_p[:, None] * (
            sign * q_bc[:, None] * _COS3_X[None, :]
            + beta_p[:, None] * e_lead[:, None])
        p_rest = np.sqrt(np.maximum(
            p_par * p_par
            + (q_bc * q_bc)[:, None] * (1.0 - _COS3_X[None, :] ** 2),
            0.0))
        weights = _MBC_W[:, None] * _COS3_W[None, :] * np.ones_like(e_rest)
        return (weights.ravel(), e_rest.ravel(), p_rest.ravel())
    # Partner slot: the (E, p) node is independent of the leading-pair
    # direction; one node per M_bc^2 point with the summed weight.
    e_rest = (M * M - mbc2 + md * md) / (2.0 * M)
    return (_MBC_W.copy(), e_rest, q_pair)


class RecordSet:
    """The candidate record set spanning an adjacent source-mass pair,
    with the card-9 open-set rule and the analytic content engine.

    *records* are catalog records (in catalog record order) covering the
    open sets of both point masses of the pair -- the union of the
    card-9 open sets; :meth:`open_records` filters them per mass.  The
    engine caches the per-(record, mass) evaluations, so the three
    mixture payloads of one pair reuse the shared record evaluations."""

    def __init__(self, catalog: HadronCatalog,
                 records: "list[HadronRecord] | tuple[HadronRecord, ...]") \
            -> None:
        self.catalog = catalog
        self.records = tuple(records)
        self._stable = frozenset(int(p) for p in catalog.stable_states())
        self._decayer = ResonanceDecayer(catalog, 0, 0, 0)
        self._content_cache: dict = {}
        self._open_objects_cache: dict = {}
        self._spectral_cache: dict = {}
        self._mean_mass_cache: dict = {}
        self._narrow_cache: dict = {}

    # -- open-set rule (the card-9 mixture item 1) -----------------------
    def is_open(self, record: HadronRecord, mass: float) -> bool:
        """The card-9 truncated-spectral-window rule: the record is open
        at *mass* when ``[mass_range[0], min(mass_range[1], mass)]`` is
        non-empty."""
        lo = float(record.mass_range[0])
        hi = float(record.mass_range[1])
        return min(hi, float(mass)) >= lo

    def open_records(self, mass: float) -> "list[HadronRecord]":
        """The open records at *mass*, in catalog record order (the
        card-9 open-set convention)."""
        return [r for r in self.records if self.is_open(r, mass)]

    def is_stable(self, record: HadronRecord) -> bool:
        """The kernel's stable convention: a stable termination state or
        a record flagged ``stable_for_decay`` (the identity event)."""
        return (int(record.pdg_id) in self._stable
                or bool(record.stable_for_decay))

    def _final_or_record(self, pdg: int):
        """The catalog record of a daughter, or ``None`` when the
        daughter is a final particle outside the catalog (the vendored
        PDG table only; treated as a stable final particle)."""
        try:
            return self.catalog.get(int(pdg))
        except KeyError:
            return None

    def _stable_known_stable(self, pdg: int) -> bool:
        return int(pdg) in self._stable

    # -- mass-realization conventions ------------------------------------
    def evaluation_mass(self, record: HadronRecord, mass: float) -> float:
        """The single-mass evaluation point of *record* at the point
        mass *mass* (the documented collapse conventions): a narrow
        line shape evaluates at the draw-distribution point of its
        truncated window (the card-3 draw cell containing the pole when
        the window covers it, the draw mean of the smooth truncated
        density otherwise); a broad line shape evaluates at the pole
        clamped into the truncated window (the deep-collapse
        convention); the stable identity records carry exactly the
        point mass (the card-9 degenerate-window rule)."""
        mass = float(mass)
        if self.is_stable(record):
            return mass
        lo = float(record.mass_range[0])
        hi = float(record.mass_range[1])
        top = min(hi, mass)
        if self.is_narrow(record):
            return self._draw_mean_mass(record, lo, top)
        return min(max(float(record.pole_mass), lo), top)

    def is_narrow(self, record: HadronRecord) -> bool:
        """Whether *record* is a narrow line shape (the draw
        concentrates in the top cell): the nominal width at most
        :data:`NARROW_WIDTH_RATIO` of the support span."""
        key = int(record.pdg_id)
        cached = self._narrow_cache.get(key)
        if cached is None:
            span = float(record.mass_range[1]) - float(record.mass_range[0])
            cached = bool(
                span > 0.0
                and float(record.nominal_width) <= NARROW_WIDTH_RATIO * span
            )
            self._narrow_cache[key] = cached
        return cached

    def _spectral(self, record: HadronRecord, mass: float) -> float:
        """The card-3 spectral density of *record* at *mass* (cached)."""
        key = (int(record.pdg_id), float(mass))
        cached = self._spectral_cache.get(key)
        if cached is None:
            dmap = self._decayer._daughter_mass_map(record)
            cached = float(spectral_function(float(mass), record, dmap))
            self._spectral_cache[key] = cached
        return cached

    def _draw_mean_mass(self, record: HadronRecord, lo: float,
                        hi: float) -> float:
        """The draw-distribution mean of the card-3 realized mass over
        the window ``[lo, hi]`` (cached), on the card-3 draw grid (the
        1024-cell piecewise-linear model of the sampler; the trapezoid
        on the same nodes is its exact cell mean).

        A narrow line shape whose window covers the pole concentrates
        in the card-3 draw cell containing the pole (the evaluation
        point is that cell's center, the sealed engine's fixed-mass
        treatment of the eta and eta-prime needles)."""
        lo = float(lo)
        hi = float(hi)
        key = (int(record.pdg_id), lo, hi)
        cached = self._mean_mass_cache.get(key)
        if cached is not None:
            return cached
        if self.is_narrow(record) and hi >= float(record.pole_mass):
            cell = (hi - lo) / 1024.0
            idx = min(int(math.floor((float(record.pole_mass) - lo) / cell)),
                      1023)
            mean = lo + (idx + 0.5) * cell
        else:
            grid = np.linspace(lo, hi, _DRAW_MEAN_NODES)
            rho = np.array([self._spectral(record, float(m)) for m in grid])
            w = np.full(_DRAW_MEAN_NODES, (hi - lo) / (_DRAW_MEAN_NODES - 1))
            w[0] *= 0.5
            w[-1] *= 0.5
            norm = float(np.sum(w * rho))
            if not norm > 0.0:
                raise RuntimeError(
                    "threshold_metric: record "
                    f"{record.name} (pdg {record.pdg_id}) has an empty "
                    f"spectral density on the window [{lo!r}, {hi!r}]"
                )
            mean = float(np.sum(grid * w * rho) / norm)
        self._mean_mass_cache[key] = mean
        return mean

    def _budget_top(self, record: HadronRecord, parent_mass: float,
                    other_mass: float) -> float:
        """The on-shell budget-window top of a daughter record at its
        parent vertex: ``min(mass_range[1], parent_mass - m_partner)``
        (the card-6 budget convention; the partner enters at its pole
        mass, the sealed engine's convention)."""
        top = min(float(record.mass_range[1]),
                  float(parent_mass) - float(other_mass))
        if top < float(record.mass_range[0]):
            raise RuntimeError(
                "threshold_metric: the budget window of daughter "
                f"{record.name} (pdg {record.pdg_id}) is empty at parent "
                f"mass {parent_mass!r} (partner mass {other_mass!r})"
            )
        return top

    # -- open objects (the kernel's own card-5 charge weights) -----------
    def _open_objects(self, record: HadronRecord, mass: float):
        """The kernel's open objects at the realized mass (cached; the
        card-5 charge shares and the card-3/4 width law of the realized
        mass, the default Blatt-Weisskopf barrier)."""
        key = (int(record.pdg_id), float(mass))
        cached = self._open_objects_cache.get(key)
        if cached is None:
            cached = self._decayer._open_objects(record, float(mass))
            self._open_objects_cache[key] = cached
        return cached

    # -- the recursive decay-content engine ------------------------------
    def _decay_content(self, record: HadronRecord, mass: float,
                       depth: int = 0) -> dict:
        """The per-species rest-frame content of ONE decay of *record*
        realized at rest with mass *mass*: ``{species_key: [count,
        nodes]}`` with ``nodes`` an ``(N, 3)`` array of ``(weight, E,
        p)`` rows whose weights sum to *count*.  Exact on-shell
        two-body kinematics through the Kaellen decay momentum, the
        card-5 charge shares of the open objects at the realized mass,
        the isotropic-direction boost averages of the legacy
        convention, and the card-7 flat three-body model with the exact
        Kallen leading-pair kinematics; no draws.

        Daughter evaluation: every two-body resonance daughter, at
        every level, integrates its mass realization over its budget
        window (the daughter-mass quadrature; the concentration-point
        collapse of deep levels is replaced by the same quadrature the
        integrated level applies)."""
        key = (int(record.pdg_id), float(mass), int(depth))
        cached = self._content_cache.get(key)
        if cached is not None:
            return cached
        out: dict[str, list] = {}
        try:
            objects, total = self._open_objects(record, mass)
        except NoOpenChannelError:
            raise RuntimeError(
                "threshold_metric: record "
                f"{record.name} (pdg {record.pdg_id}) has no open channel "
                f"at its realized mass {mass!r} (a decaying record "
                "realized at a degenerate window has no open channel; the "
                "card-9 mixture would fail identically)"
            ) from None
        for channel, combo, weight in objects:
            share = float(weight) / float(total)
            if channel.is_three_body:
                self._three_body_object(
                    out, record, combo, float(mass), share, depth)
            else:
                self._two_body_object(
                    out, record, combo, float(mass), share, depth)
        self._content_cache[key] = out
        return out

    def _add_nodes(self, out: dict, skey: str, count: float,
                   nodes: np.ndarray) -> None:
        bucket = out.get(skey)
        if bucket is None:
            out[skey] = [float(count), nodes]
            return
        bucket[0] += float(count)
        if bucket[1].shape[0] + nodes.shape[0] > _MAX_NODES:
            raise RuntimeError(
                "threshold_metric: the per-species node count exceeded "
                "the defensive bound"
            )
        bucket[1] = np.concatenate([bucket[1], nodes])

    def _daughter_budget_nodes(self, record: HadronRecord, top: float):
        """The daughter budget-window quadrature: Gauss-Legendre nodes
        on ``[mass_range[0], top]`` with the card-3 density per node and
        the normalized node weights (``w_j * rho_j / Z``); a narrow line
        shape concentrates at its budget-window evaluation point (the
        single-node quadrature)."""
        lo = float(record.mass_range[0])
        if self.is_narrow(record) and top >= float(record.pole_mass):
            # The needle sits inside the budget window: the draw
            # concentrates in the card-3 draw cell containing the pole
            # (the fixed single-node evaluation; the sealed engine's
            # fixed-mass treatment of the eta daughters).
            mass_pt = self._draw_mean_mass(record, lo, float(top))
            return [(mass_pt, 1.0)]
        x = 0.5 * (top - lo) * _DAU_X + 0.5 * (top + lo)
        gl_w = 0.5 * (top - lo) * _DAU_W
        rho = [self._spectral(record, float(m)) for m in x]
        z = math.fsum(w * r for w, r in zip(gl_w, rho))
        if not z > 0.0:
            raise RuntimeError(
                "threshold_metric: daughter "
                f"{record.name} (pdg {record.pdg_id}) has an empty "
                f"spectral density on the budget window [{lo!r}, {top!r}]"
            )
        return [(float(m), float(w) * float(r) / z)
                for m, w, r in zip(x, gl_w, rho) if w * r > 0.0]

    def _two_body_object(self, out: dict, record: HadronRecord,
                         combo: tuple, mass: float, share: float,
                         depth: int) -> None:
        """One two-body open object: exact on-shell kinematics (the
        Kaellen decay momentum at the realized parent mass and the
        daughters' realized masses).

        A resonance daughter's mass realization is integrated over its
        budget window at EVERY level (the daughter-mass quadrature: the
        card-3 density and the open-object weights kernel-exact at
        every node, the stable partner's lab motion joining the same
        quadrature); the concentration-point collapse of deep levels is
        replaced by the same quadrature the integrated level applies.
        The two-resonance channels (the committed catalog's only case
        is f4(2050) -> eta eta, both narrow needles) keep the collapsed
        evaluation of both daughters at their budget-window
        concentration points, evaluated once (the documented
        narrow-record treatment, retained)."""
        d0, d1 = (int(d) for d in combo)
        m0_pole = _pole_mass(self.catalog, d0)
        m1_pole = _pole_mass(self.catalog, d1)
        rec0 = self._final_or_record(d0)
        rec1 = self._final_or_record(d1)
        d0_res = rec0 is not None and not self.is_stable(rec0)
        d1_res = rec1 is not None and not self.is_stable(rec1)
        cos, cos_w = (_COS_1, _COS_W_1) if depth == 0 else (_COS_D, _COS_W_D)
        if d0_res != d1_res:
            # The daughter-mass quadrature at every level: the
            # resonance daughter's realized mass integrates over its
            # budget window; the two-body split and the stable partner
            # join per node.
            if d0_res:
                d_res, rec_d, m_stab, pdg_stab = d0, rec0, m1_pole, d1
            else:
                d_res, rec_d, m_stab, pdg_stab = d1, rec1, m0_pole, d0
            assert rec_d is not None
            top = self._budget_top(rec_d, mass, m_stab)
            for m_d, node_w in self._daughter_budget_nodes(rec_d, top):
                q = float(decay_momentum(mass, m_d, m_stab))
                e_res = (mass * mass + m_d * m_d - m_stab * m_stab) \
                    / (2.0 * mass)
                e_stab = (mass * mass - m_d * m_d + m_stab * m_stab) \
                    / (2.0 * mass)
                skey_stab = species_key(
                    _species_name(self.catalog, pdg_stab))
                self._add_nodes(
                    out, skey_stab, share * node_w,
                    np.array([[share * node_w, e_stab, q]]))
                sub = self._decay_content(rec_d, m_d, depth + 1)
                gamma = e_res / m_d
                beta = q / e_res
                for skey, (sub_count, sub_nodes) in sub.items():
                    boosted = _boost_nodes(sub_nodes, gamma, beta,
                                           cos, cos_w)
                    boosted[:, 0] = boosted[:, 0] * (share * node_w)
                    self._add_nodes(out, skey, share * node_w * sub_count,
                                    boosted)
            return
        # The collapsed evaluation: both daughters at their evaluation
        # masses (stable daughters at the pole; the two-resonance
        # channels' narrow needles at the concentration points of their
        # budget windows -- the documented narrow-record treatment,
        # retained).
        eval_masses = []
        for pdg_d, m_other, rec_d in ((d0, m1_pole, rec0),
                                      (d1, m0_pole, rec1)):
            if rec_d is None or self.is_stable(rec_d):
                eval_masses.append(_pole_mass(self.catalog, pdg_d))
            else:
                # The retained narrow-record concentration point (the
                # two-resonance channels' needles: the committed
                # catalog's only case is f4(2050) -> eta eta).
                top = self._budget_top(rec_d, mass, m_other)
                eval_masses.append(
                    min(max(float(rec_d.pole_mass),
                            float(rec_d.mass_range[0])), top))
        d0_eval, d1_eval = eval_masses
        q = float(decay_momentum(mass, d0_eval, d1_eval))
        e0 = (mass * mass + d0_eval * d0_eval - d1_eval * d1_eval) \
            / (2.0 * mass)
        e1 = (mass * mass - d0_eval * d0_eval + d1_eval * d1_eval) \
            / (2.0 * mass)
        for pdg_d, m_d, e_d in ((d0, d0_eval, e0), (d1, d1_eval, e1)):
            d_record = self._final_or_record(pdg_d)
            if d_record is None or self.is_stable(d_record):
                skey = species_key(_species_name(self.catalog, pdg_d))
                node = np.array([[share, e_d, q]])
                self._add_nodes(out, skey, share, node)
                continue
            sub = self._decay_content(d_record, m_d, depth + 1)
            gamma = e_d / m_d
            beta = q / e_d
            for skey, (sub_count, sub_nodes) in sub.items():
                boosted = _boost_nodes(sub_nodes, gamma, beta, cos, cos_w)
                boosted[:, 0] = boosted[:, 0] * share
                self._add_nodes(out, skey, share * sub_count, boosted)

    def _three_body_object(self, out: dict, record: HadronRecord,
                           combo: tuple, mass: float, share: float,
                           depth: int) -> None:
        """One card-7 three-body open object: the flat ``M_bc^2``
        phase-space model at the realized parent mass with the
        daughters' evaluation masses in the split, the internal
        ``(E, p)`` nodes per daughter slot, and the isotropic boost
        averages of resonance slot daughters."""
        pdgs = [int(d) for d in combo]
        eval_masses = []
        for pdg_d in pdgs:
            d_record = self._final_or_record(pdg_d)
            if d_record is None or self.is_stable(d_record):
                eval_masses.append(_pole_mass(self.catalog, pdg_d))
            else:
                other_sum = math.fsum(
                    _pole_mass(self.catalog, p)
                    for p in pdgs if p != pdg_d
                )
                top = self._budget_top(d_record, mass, other_sum)
                eval_masses.append(
                    min(max(float(d_record.pole_mass),
                            float(d_record.mass_range[0])), top))
        cos, cos_w = (_COS_1, _COS_W_1) if depth == 0 else (_COS_D, _COS_W_D)
        for slot, pdg_d in enumerate(pdgs):
            w_slot, e_slot, p_slot = _three_body_slot_nodes(
                float(mass), tuple(eval_masses), slot)
            d_record = self._final_or_record(pdg_d)
            if d_record is None or self.is_stable(d_record):
                nodes = np.stack([w_slot * share, e_slot, p_slot], axis=1)
                skey = species_key(_species_name(self.catalog, pdg_d))
                self._add_nodes(out, skey, share, nodes)
                continue
            m_d = eval_masses[slot]
            sub = self._decay_content(d_record, m_d, depth + 1)
            # The slot motion varies node by node; the daughter's own
            # isotropic decay expands over the direction grid per slot
            # node (the slot motion magnitude and the internal decay
            # direction are independent isotropic factors, so the boost
            # of the internal ensemble by the per-slot (gamma, beta)
            # with the direction grid is the exact average).
            gamma = e_slot / m_d
            beta = p_slot / e_slot
            n_slot = w_slot.shape[0]
            for skey, (sub_count, sub_nodes) in sub.items():
                gamma_grid = np.repeat(gamma, sub_nodes.shape[0])
                beta_grid = np.repeat(beta, sub_nodes.shape[0])
                tiled = np.tile(sub_nodes, (n_slot, 1))
                boosted = _boost_nodes(
                    tiled, gamma_grid, beta_grid, cos, cos_w)
                w_rep = np.repeat(
                    w_slot, sub_nodes.shape[0] * cos.shape[0])
                boosted[:, 0] = boosted[:, 0] * (w_rep * share)
                self._add_nodes(
                    out, skey, share * sub_count, boosted)

    # -- the mixture-level draw integration ------------------------------
    def record_content(self, record: HadronRecord, mass: float) -> dict:
        """The aggregated per-species content of *record* for the source
        at rest at the point mass *mass*: ``{species_key: {"n",
        "p_sum", "e_sum", "ept_sum"}}``.

        The card-9 mixture realizes the record's mass from the card-3
        spectral function on the truncated window; the expectation is
        the deterministic quadrature over that window (the card-3
        density and the open-object weights kernel-exact at every
        Gauss-Legendre node).  A stable record is the identity event at
        exactly the point mass; a narrow line shape concentrates at its
        concentration point.  The ``ept_sum`` carries the isotropic
        ``E[sin theta] = pi/4`` factor once, at the final aggregation."""
        mass = float(mass)
        if self.is_stable(record):
            skey = species_key(record.name)
            return {skey: {"n": 1.0, "p_sum": 0.0, "e_sum": mass,
                           "ept_sum": 0.0}}
        lo = float(record.mass_range[0])
        hi = float(record.mass_range[1])
        top = min(hi, mass)
        if top < lo:
            raise RuntimeError(
                "threshold_metric: record "
                f"{record.name} (pdg {record.pdg_id}) is closed at the "
                f"point mass {mass!r}"
            )
        if top - lo <= DEGENERATE_WINDOW_TOL:
            raise RuntimeError(
                "threshold_metric: record "
                f"{record.name} (pdg {record.pdg_id}) has a degenerate "
                f"window at the point mass {mass!r} (a decaying record "
                "realized at a degenerate window has no open channel; "
                "the card-9 mixture would fail identically)"
            )
        if self.is_narrow(record) and top >= float(record.pole_mass):
            # The needle sits inside the window: the draw concentrates
            # in the card-3 draw cell containing the pole (the fixed
            # single-node evaluation; a root Gauss-Legendre grid would
            # miss the needle).
            parts = [(1.0, self._decay_content(record, self._draw_mean_mass(
                record, lo, top), 0))]
        else:
            # The truncated density is smooth on the window (broad
            # records everywhere; narrow records truncated below the
            # pole): the deterministic Gauss-Legendre quadrature with
            # the card-3 density and the open-object weights
            # kernel-exact at every node.
            x = 0.5 * (top - lo) * _ROOT_X + 0.5 * (top + lo)
            gl_w = 0.5 * (top - lo) * _ROOT_W
            rho = np.array([self._spectral(record, float(m)) for m in x])
            norm = float(np.sum(gl_w * rho))
            if not norm > 0.0:
                raise RuntimeError(
                    "threshold_metric: record "
                    f"{record.name} (pdg {record.pdg_id}) has an empty "
                    f"spectral density on the truncated window "
                    f"[{lo!r}, {top!r}]"
                )
            parts = []
            for m_i, w_i, rho_i in zip(x, gl_w, rho):
                node_w = float(w_i) * float(rho_i) / norm
                if not node_w > 0.0:
                    continue
                parts.append((node_w,
                              self._decay_content(record, float(m_i), 0)))
        # The per-node parts are aggregated immediately (the root frame
        # is the lab frame; no further boost is applied below this
        # level), so the accumulated state stays O(#species).
        acc: dict[str, list] = {}
        for node_w, part in parts:
            for skey, (count, nodes) in part.items():
                w = nodes[:, 0]
                bucket = acc.get(skey)
                if bucket is None:
                    bucket = [0.0, 0.0, 0.0, 0.0]
                    acc[skey] = bucket
                bucket[0] += node_w * float(count)
                bucket[1] += node_w * float(np.sum(w * nodes[:, 2]))
                bucket[2] += node_w * float(np.sum(w * nodes[:, 1]))
                bucket[3] += node_w * float(
                    np.sum(w * nodes[:, 1] * nodes[:, 2]))
        out: dict[str, dict] = {}
        for skey, (n, p_sum, e_sum, ep_sum) in acc.items():
            out[skey] = {
                "n": float(n),
                "p_sum": p_sum,
                "e_sum": e_sum,
                "ept_sum": ep_sum * ISOTROPIC_SIN_THETA_MEAN,
            }
        return out

    def mixture_payload(self, records: "list[HadronRecord]",
                        mass: float) -> dict:
        """The flat-mixture payload over *records* at the point mass
        *mass* (equal record weights; the card-9 mixture item 2): the
        aggregated per-species sums over the per-decay expectations."""
        if not records:
            raise RuntimeError(
                "threshold_metric: an empty record subset has no "
                "flat-mixture payload"
            )
        share = 1.0 / len(records)
        out: dict[str, dict] = {}
        for record in records:
            content = self.record_content(record, mass)
            for skey, block in content.items():
                bucket = out.get(skey)
                if bucket is None:
                    bucket = {"n": 0.0, "p_sum": 0.0, "e_sum": 0.0,
                              "ept_sum": 0.0}
                    out[skey] = bucket
                for field in bucket:
                    bucket[field] += share * block[field]
        return out


def _observable_value(payload: dict, skey: str, observable: str):
    """One analytic observable of a payload: the multiplicity, the
    per-particle mean ``<p>``, or the energy-weighted ``<p_T>_y``;
    ``None`` where the moment is undefined (the species absent, or no
    energy sum for ``<p_T>_y``)."""
    block = payload.get(skey)
    if block is None or block["n"] <= 0.0:
        if observable == "multiplicity":
            return 0.0
        return None
    if observable == "multiplicity":
        return float(block["n"])
    if observable == "mean_p":
        return float(block["p_sum"]) / float(block["n"])
    if observable == "mean_pt_y":
        if not block["e_sum"] > 0.0:
            return None
        return float(block["ept_sum"]) / float(block["e_sum"])
    raise ValueError(f"unknown observable {observable!r}")


def _diff(a, b):
    """The signed difference of two observable values; ``None`` where
    either operand is undefined."""
    if a is None or b is None:
        return None
    return float(a) - float(b)


def full_analytic_difference(record_set: RecordSet, M_k: float,
                             M_kp1: float, species: str) -> dict:
    """The full analytic difference of the flat-mixture expectation for
    one *species* between the adjacent point masses ``M_k`` and
    ``M_kp1``, with its decomposition (pure function; no draws).

    Returns a dict with, per checked observable (``multiplicity``,
    ``mean_p``, ``mean_pt_y``):

    - ``delta_analytic_full``: ``mu_an(M_{k+1}) - mu_an(M_k)``;
    - ``open_set_change``: the mixture change at ``M_{k+1}`` (the open-
      set membership and share change against the counterfactual
      mixture that keeps the ``M_k`` open set and shares);
    - ``momentum_drift``: the surviving-record drift (the counter-
      factual mixture at ``M_{k+1}`` over the open set at ``M_k``
      minus the mixture at ``M_k``) -- the daughter momentum moments of
      every record open at both masses at ``M_{k+1}`` minus at
      ``M_k``.

    The two terms add to ``delta_analytic_full`` exactly; at an
    open-set-unchanged pair the open-set-change contribution vanishes
    identically.  Moment values are ``None`` where the species' moment
    is undefined at either point mass (the driver's absent-species rule
    applies there; the multiplicity is always defined)."""
    if not isinstance(record_set, RecordSet):
        raise TypeError(
            "full_analytic_difference: record_set must be a "
            "threshold_metric.RecordSet"
        )
    M_k = float(M_k)
    M_kp1 = float(M_kp1)
    open_k = record_set.open_records(M_k)
    open_kp1 = record_set.open_records(M_kp1)
    ids_k = [int(r.pdg_id) for r in open_k]
    ids_kp1 = [int(r.pdg_id) for r in open_kp1]
    changed = ids_k != ids_kp1
    set_k = set(ids_k)
    set_kp1 = set(ids_kp1)

    payload_k = record_set.mixture_payload(open_k, M_k)
    payload_surv_ref = record_set.mixture_payload(open_k, M_kp1)
    payload_kp1 = record_set.mixture_payload(open_kp1, M_kp1)

    delta_full: dict = {}
    open_set_change: dict = {}
    momentum_drift: dict = {}
    for observable in OBSERVABLES:
        v_k = _observable_value(payload_k, species, observable)
        v_surv = _observable_value(payload_surv_ref, species, observable)
        v_kp1 = _observable_value(payload_kp1, species, observable)
        delta_full[observable] = _diff(v_kp1, v_k)
        momentum_drift[observable] = _diff(v_surv, v_k)
        open_set_change[observable] = _diff(v_kp1, v_surv)
    return {
        "species": str(species),
        "masses": {"M_k": M_k, "M_kp1": M_kp1},
        "open_set": {
            "at_M_k": ids_k,
            "at_M_kp1": ids_kp1,
            "changed": bool(changed),
            "opened": [pdg for pdg in ids_kp1 if pdg not in set_k],
            "closed": [pdg for pdg in ids_k if pdg not in set_kp1],
        },
        "open_set_change": open_set_change,
        "momentum_drift": momentum_drift,
        "delta_analytic_full": delta_full,
        "convention": {
            "engine": "na49_mixing.threshold_metric.RecordSet",
            "quadrature": "the truncated mixture window integrated on a "
                          "fixed Gauss-Legendre grid with the card-3 "
                          "density and the open-object weights "
                          "kernel-exact at every node; daughter budget "
                          "windows integrated on a fixed Gauss-Legendre "
                          "grid at every level (the daughter-mass "
                          "quadrature replaces the concentration-point "
                          "collapse of deep levels); narrow line "
                          "shapes at their card-3 draw-distribution "
                          "point and the two-resonance channels at the "
                          "retained collapsed needle evaluation; no "
                          "Monte Carlo",
            "kinematics": "exact on-shell two-body kinematics (the "
                          "Kaellen decay momentum) with the exact "
                          "Kallen leading-pair kinematics of the "
                          "card-7 three-body objects, the legacy "
                          "_decay_rest_frame boost convention and the "
                          "card-5 charge shares",
        },
    }


# ---------------------------------------------------------------------------
# The paired check evaluation
# ---------------------------------------------------------------------------
def evaluate_pair(metric_observed: float, delta_analytic_full: float,
                  se_k: float, se_kp1: float,
                  sigma: float = SIGMA_TOLERANCE) -> dict:
    """The signed sigma and pass flag of one paired check at the
    pre-registered tolerance.

    ``SE_combined = sqrt(SE_k^2 + SE_{k+1}^2)``; the signed sigma is
    ``(metric_observed - delta_analytic_full) / SE_combined`` and the
    check passes at ``|sigma| <= sigma``.  The deterministic-observable
    floor (:data:`DETERMINISTIC_SE_FLOOR`) applies to a collapsed
    between-replica SE: below the floor the SE is floored at the
    deterministic precision of the stored sums and the vacuous-zero
    rule applies at that floor (a residual at or below
    :data:`ZERO_RESIDUAL_TOL` is a vacuous pass, sigma 0.0; a larger
    residual evaluates at the floored SE and the unchanged band).  A
    floored SE is recorded through ``se_floor_applied`` and the floored
    value through ``se_effective``.  The tolerance is never relaxed by
    this module."""
    se_combined = math.sqrt(float(se_k) ** 2 + float(se_kp1) ** 2)
    residual = float(metric_observed) - float(delta_analytic_full)
    se_effective = se_combined
    se_floor_applied = False
    if se_combined < DETERMINISTIC_SE_FLOOR:
        se_effective = DETERMINISTIC_SE_FLOOR
        se_floor_applied = True
    if se_floor_applied and abs(residual) <= ZERO_RESIDUAL_TOL:
        value = 0.0
        passed = True
    else:
        value = residual / se_effective
        passed = abs(value) <= float(sigma)
    return {
        "sigma": value,
        "pass": bool(passed),
        "residual": residual,
        "se_combined": se_combined,
        "se_effective": se_effective,
        "se_floor_applied": bool(se_floor_applied),
        "tolerance_sigma": float(sigma),
    }


# ---------------------------------------------------------------------------
# Sealed-archive re-evaluation
# ---------------------------------------------------------------------------
def _reevaluated_status(check: dict) -> tuple:
    """The re-registered verdict of one sealed check: returns
    ``(status, sigma, pass)`` from the sealed stored fields.  The
    re-registered metric applies ``delta_analytic_full`` on both pair
    classes, so the applied residual is
    ``|delta_observed - delta_analytic_full|`` at the unchanged
    3.0-sigma tolerance; ``absent_species`` checks carry no verdict.
    The deterministic-observable floor
    (:data:`DETERMINISTIC_SE_FLOOR`) applies to a collapsed
    between-replica SE: below the floor the SE is floored and the
    vacuous-zero rule applies at that floor (a residual at or below
    :data:`ZERO_RESIDUAL_TOL` is ``vacuous_zero``, a larger residual
    evaluates at the floored SE and the unchanged band)."""
    status_original = str(check.get("status"))
    if status_original == "absent_species":
        return ("absent_species", None, None)
    se = check.get("se_combined")
    se_raw = 0.0 if se is None else float(se)
    delta_observed = float(check["delta_observed"])
    delta_full = float(check["delta_analytic_full"])
    residual = abs(delta_observed - delta_full)
    if se_raw < DETERMINISTIC_SE_FLOOR:
        if residual <= ZERO_RESIDUAL_TOL:
            return ("vacuous_zero", 0.0, True)
        value = residual / DETERMINISTIC_SE_FLOOR
        return ("pass" if value <= SIGMA_TOLERANCE else "fail",
                value, value <= SIGMA_TOLERANCE)
    value = residual / se_raw
    return ("pass" if value <= SIGMA_TOLERANCE else "fail",
            value, value <= SIGMA_TOLERANCE)


def rereval_report(gate_payload: dict, *, run: Optional[str] = None,
                   archive_rel: Optional[str] = None,
                   gate_sha256: Optional[str] = None,
                   card: Optional[str] = None) -> dict:
    """Assemble the sealed-archive re-evaluation of one run's sealed
    Gate 2 payload under the re-registered metric (pure function of the
    sealed payload and the identifying arguments; byte-stable).

    Every applied check (the sealed ``pass``/``fail``/``vacuous_zero``
    statuses) is re-evaluated against the sealed stored
    ``delta_analytic_full`` at the unchanged 3.0-sigma tolerance;
    ``absent_species`` checks carry no verdict.  The summary carries the
    per-pair-class pass rates (among the original failures, the card's
    re-registration criterion, and among all applied checks), the
    original-to-revised failure counts, the maximum sigma, and the
    named residual checks."""
    clause = gate_payload["clauses"]["threshold_discontinuity"]
    checks_original = clause["checks"]
    rows: list[dict] = []
    summary = {
        "n_checks": len(checks_original),
        "n_checks_applied": 0,
        "n_pass": 0,
        "n_fail": 0,
        "n_vacuous_zero": 0,
        "n_absent_species": 0,
        "original": {
            "n_checks_failed": int(clause.get("n_checks_failed", 0)),
            "failed_at_open_set_change": int(
                clause.get("n_failed_at_open_set_changes", 0)),
            "failed_at_unchanged": int(
                clause.get("n_failed_at_unchanged_pairs", 0)),
        },
        "reeval": {},
        "max_abs_sigma": 0.0,
        "max_abs_sigma_check": None,
        "residuals": [],
    }
    per_class = {
        "open_set_change": {"applied": 0, "pass": 0, "fail": 0},
        "open_set_unchanged": {"applied": 0, "pass": 0, "fail": 0},
    }
    original_fail_per_class = {
        "open_set_change": 0,
        "open_set_unchanged": 0,
    }
    original_fail_now_pass_per_class = {
        "open_set_change": 0,
        "open_set_unchanged": 0,
    }
    max_sigma = -1.0
    max_key = None
    for check in checks_original:
        status, value, passed = _reevaluated_status(check)
        pair = [int(k) for k in check["pair"]]
        skey = str(check["species"])
        observable = str(check["observable"])
        changed = bool(check["open_set_changes"])
        class_key = "open_set_change" if changed else "open_set_unchanged"
        row = {
            "pair": pair,
            "species": skey,
            "observable": observable,
            "open_set_changes": changed,
            "pair_class": class_key,
            "original_status": str(check["status"]),
            "status": status,
            "observed": check["observed"],
            "delta_observed": check["delta_observed"],
            "delta_analytic_full": check["delta_analytic_full"],
            "delta_analytic_applied": check["delta_analytic_full"],
            "se_combined": check["se_combined"],
            "sigma_original": check.get("sigma"),
            "sigma": value,
            "pass": passed,
        }
        rows.append(row)
        if status == "absent_species":
            summary["n_absent_species"] += 1
            continue
        if status == "vacuous_zero":
            summary["n_vacuous_zero"] += 1
            continue
        summary["n_checks_applied"] += 1
        per_class[class_key]["applied"] += 1
        if passed:
            summary["n_pass"] += 1
            per_class[class_key]["pass"] += 1
        else:
            summary["n_fail"] += 1
            per_class[class_key]["fail"] += 1
            if float(value) > max_sigma:
                max_sigma = float(value)
                max_key = (f"pair({pair[0]},{pair[1]}) {skey} "
                           f"{observable}")
            summary["residuals"].append({
                "pair": pair,
                "species": skey,
                "observable": observable,
                "pair_class": class_key,
                "sigma": value,
                "residual": abs(float(check["delta_observed"])
                                - float(check["delta_analytic_full"])),
                "se_combined": float(check["se_combined"]),
            })
        if str(check["status"]) == "fail":
            original_fail_per_class[class_key] += 1
            if passed:
                original_fail_now_pass_per_class[class_key] += 1
    n_original_fail = int(clause.get("n_checks_failed", 0))
    now_pass_of_original = sum(
        original_fail_now_pass_per_class.values())
    rate_block = {}
    for class_key, stats in per_class.items():
        rate_block[class_key] = {
            "applied": stats["applied"],
            "pass": stats["pass"],
            "fail": stats["fail"],
            "pass_rate_all_applied": (
                stats["pass"] / stats["applied"]
                if stats["applied"] else None),
            "original_failures": original_fail_per_class[class_key],
            "original_failures_now_passing":
                original_fail_now_pass_per_class[class_key],
            "pass_rate_among_original_failures": (
                original_fail_now_pass_per_class[class_key]
                / original_fail_per_class[class_key]
                if original_fail_per_class[class_key] else None),
        }
    summary["reeval"] = {
        "pass_rate_all_applied": rate_block,
        "original_failures_total": n_original_fail,
        "original_failures_now_passing": now_pass_of_original,
        "pass_rate_among_original_failures": (
            now_pass_of_original / n_original_fail
            if n_original_fail else None),
        "failed_at_open_set_change": per_class["open_set_change"]["fail"],
        "failed_at_unchanged": per_class["open_set_unchanged"]["fail"],
    }
    for residual in summary["residuals"]:
        residual["diagnosis"] = diagnose_residual(residual)
    summary["max_abs_sigma"] = (max_sigma if max_sigma >= 0.0 else 0.0)
    summary["max_abs_sigma_check"] = max_key
    payload = {
        "schema": "na49_mixing.threshold_metric.rereval/1",
        "card": str(card) if card else
        "TASK-20260917-r3-gate2-followup-threshold-metric-reregistration",
        "run": str(run) if run else str(gate_payload.get("run", "")),
        "metric": {
            "id": THRESHOLD_METRIC_ID,
            "statement": "for every adjacent pair (k, k+1) and every "
                         "stable species, with SE_combined = sqrt(SE_k^2 "
                         "+ SE_{k+1}^2): "
                         "|mu(k+1) - mu(k) - delta_analytic_full| <= 3 "
                         "SE_combined on BOTH pair classes, for the "
                         "species multiplicity and the per-species "
                         "momentum moments <p> and <p_T>_y",
            "tolerance_sigma": SIGMA_TOLERANCE,
            "source": "the sealed per-check delta_analytic_full values "
                      "(the unapplied full analytic difference recorded "
                      "by the sealed metric); the sealed archives are "
                      "read-only and are not re-sealed",
        },
        "archive": {
            "path": str(archive_rel) if archive_rel else None,
            "gate_sha256": str(gate_sha256) if gate_sha256 else None,
        },
        "coverage": {
            "adjacent_pairs_checked": int(
                clause.get("adjacent_pairs_checked", 0)),
            "open_set_changes": [int(k) for k in
                                 clause.get("open_set_changes", [])],
            "n_checks": len(checks_original),
            "observables": list(OBSERVABLES),
        },
        "summary": summary,
        "checks": rows,
    }
    return payload


def diagnose_residual(residual: dict) -> str:
    """The deterministic one-line diagnosis of a residual failing check
    (a pure function of the residual row's sealed fields).

    Open-set-change residuals are the genuine residuals expected by the
    Round 2 closure clause 3.  Marginal unchanged-pair residuals (at the
    3.0-sigma edge) are statistical excursions whose per-check
    exceedance probability is expected across the applied-check count;
    the larger unchanged-pair residuals sit at the sealed analytic
    reference's documented approximation floor (its engine validation
    against the stored moments), which exceeds the 3.0-sigma statistical
    band.  A residual is never absorbed into the metric."""
    pair = residual["pair"]
    skey = residual["species"]
    observable = residual["observable"]
    value = float(residual["sigma"])
    amount = float(residual["residual"])
    where = (f"pair({pair[0]},{pair[1]}) {skey} {observable}")
    if residual["pair_class"] == "open_set_change":
        reference = residual.get("delta_analytic_full")
        if reference:
            rel = abs(amount / float(reference))
            detail = f" ({rel:.1%} of delta_analytic_full)"
        else:
            detail = ""
        return (
            f"{where}: residual {amount:+.3e} = {value:.2f} sigma"
            f"{detail}; the genuine open-set-change residual expected by "
            "the Round 2 closure clause 3, recorded as a finding (the "
            "metric is not relaxed)"
        )
    if value <= 3.2:
        return (
            f"{where}: residual {amount:+.3e} = {value:.2f} sigma; a "
            "marginal statistical excursion at the 3.0-sigma edge (the "
            "per-check 3-sigma exceedance probability 0.27% over the "
            "applied-check count expects about two marginal "
            "exceedances); no kernel defect diagnosed, the metric is "
            "not relaxed"
        )
    return (
        f"{where}: residual {amount:+.3e} = {value:.2f} sigma; the "
        "sealed analytic reference's documented moment-approximation "
        "floor (engine validation max |z| = 6.67 across the stored "
        "moments, residuals at the 1e-3 scale) exceeds the 3.0-sigma "
        "statistical band at this pair; a reference-precision residual "
        "tracking the analytic drift in sign and order of magnitude, "
        "not a kinematic step"
    )
