"""Threshold-safe source decays for resolved proxies and continuum families.

The channel basis is charge-complete for the two baryonic source types
``(B, Q, S) = (1, 1, 0)`` (p-like) and ``(1, 0, 0)`` (n-like). Every family
conserves the source quantum numbers exactly, and a family opens only
above the sum of its daughter masses through the thresholded weight.

Charge-balanced extra pions. The extra-pion count beyond a base channel
is drawn from the multiplicity slope and a pi0 mass-budget clamp, and its
charge composition is sampled over every charge state consistent with the
source's ``(B, Q, S)`` after the base channel. Because every base channel
conserves the source exactly, the extra pions must sum to
``(0, 0, 0)``: for a p-like source the extra-pion set is built from pi0
singles and pi+ pi- pairs (both preserve ``Q = 1`` and ``S = 0``), and the
same set preserves ``Q = 0`` for an n-like source. The number of pi+ pi-
pairs is drawn uniformly over the admissible range ``0..n_extra // 2``.
The selected PDG masses are then clamped to the remaining source mass,
dropping charged pairs together and then pi0 singles, so a charged
composition near threshold cannot exceed the source. A baseline p+p
event can therefore produce pi-, and the strange-pair families produce
K- and anti-K0.

Strange-pair families. Each source type carries the strangeness-zero
baryon-plus-two-kaon families built from ground-state kaons. All of them
share the requested strength ``strange_pair_strength``, so
isospin-degenerate charge partners carry the same weight unless the
configuration overrides it.

Decay hypotheses. The ``decay_hypothesis`` selector of
``ContinuumParameters`` chooses the source-decay kinematics while the
channel basis and its requested weights stay unchanged:

- ``H3`` (default) is the retained recursive empirical kernel
  ``_decay_rest_frame``: an interpretable sequential cascade whose
  intermediate cluster mass is biased mildly toward low mass by a beta
  draw. The default configuration reproduces this kernel byte for byte.
- ``H1`` is numerically sampled Lorentz-invariant phase space with a
  fixed 1024-point grid approximation: the cluster mass at each
  recursion level is drawn from the density
  ``pstar(M, m, M') * 2 M' * Phi(M')`` evaluated on that grid (see the
  primitives in ``excitation_kinematics``).
- ``H2`` is the explicit sequential resonance picture: the permuted
  daughter list is paired in deterministic order into Delta(1232)
  (nucleon + pion) and rho(770) (pion + pion) states, each resonance
  mass is drawn from a truncated p-wave Breit-Wigner line shape built
  from the PDG mass and width, and unpaired particles are emitted free.
  A line-shape window carrying no mass makes the state ineligible, and
  the daughters are then emitted at threshold.

Every hypothesis conserves the source four-momentum at the 1e-10 level
and keeps the final particle set on shell at the PDG masses, so the
quantum-number and threshold checks of ``ExcitationEvent.validate``
apply to all three (card M7 phase A; see
``docs/color_neutral_regge_excitation_model_spec.md`` section 3.4.3).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .breit_wigner import BreitWigner, draw_truncated_bw
from .excitation_kinematics import (
    PHSP_GRID_SIZE,
    FourVector,
    boost,
    direction_about,
    draw_from_grid_density,
    kallen,
    two_body_momentum,
    two_body_split_safe,
)
from .pdg_database import get_mass, get_width
from .quantum_numbers import QuantumNumbers, sum_quantum_numbers
from .source_event import FinalParticle, SourceRecord


@dataclass(frozen=True)
class ContinuumParameters:
    """Continuum channel-basis parameters and the decay-hypothesis selector.

    ``decay_hypothesis`` chooses the source-decay kinematics: ``H3``
    (default) is the retained recursive empirical kernel, ``H1`` is exact
    Lorentz-invariant phase space, and ``H2`` is the explicit sequential
    resonance picture. See the module docstring.
    """

    match_mass: float = 2.4
    pion_multiplicity_slope: float = 1.15
    strange_lambda_strength: float = 0.10
    strange_sigma_strength: float = 0.05
    strange_pair_strength: float = 0.05
    strange_threshold_power: float = 1.5
    pair_strength: float = 0.015
    pair_threshold_power: float = 2.0
    max_pions: int = 12
    decay_hypothesis: str = "H3"

    def validate(self) -> None:
        numeric = (
            self.match_mass, self.pion_multiplicity_slope,
            self.strange_lambda_strength, self.strange_sigma_strength,
            self.strange_pair_strength, self.strange_threshold_power,
            self.pair_strength, self.pair_threshold_power,
        )
        if not all(math.isfinite(float(x)) for x in numeric):
            raise ValueError("all continuum parameters must be finite")
        if self.match_mass <= 0.0 or self.pion_multiplicity_slope < 0.0:
            raise ValueError("match_mass must be positive and pion slope non-negative")
        if min(
            self.strange_lambda_strength, self.strange_sigma_strength,
            self.strange_pair_strength, self.strange_threshold_power,
            self.pair_strength, self.pair_threshold_power,
        ) < 0.0:
            raise ValueError("continuum strengths and powers must be non-negative")
        if not (0 <= int(self.max_pions) <= 64):
            raise ValueError("max_pions must lie in [0, 64]")
        if str(self.decay_hypothesis).strip().upper() not in ("H1", "H2", "H3"):
            raise ValueError("decay_hypothesis must be one of H1, H2, or H3")


def minimum_excited_mass(qn: QuantumNumbers) -> float:
    """Lowest represented two-hadron threshold for a baryonic source."""
    mpi0 = get_mass(111)
    if qn == QuantumNumbers(1, 1, 0):
        return min(get_mass(2212) + mpi0, get_mass(2112) + get_mass(211))
    if qn == QuantumNumbers(1, 0, 0):
        return min(get_mass(2112) + mpi0, get_mass(2212) + get_mass(-211))
    raise ValueError(f"unsupported source quantum numbers {qn}")


def _isotropic_direction(rng: np.random.Generator) -> np.ndarray:
    c = 2.0 * rng.random() - 1.0
    return direction_about([0.0, 0.0, 1.0], c, 2.0 * math.pi * rng.random())


def _decay_rest_frame(
    parent_mass: float,
    masses: list[float],
    rng: np.random.Generator,
) -> list[FourVector]:
    """Recursive exact phase-space construction.

    The intermediate cluster mass is sampled inside its exact allowed range.
    This is deliberately an interpretable sequential cascade, not a claim of
    uniform Lorentz-invariant phase space.
    """
    if not masses:
        raise ValueError("a decay channel must contain at least one daughter")
    threshold = sum(masses)
    if threshold > parent_mass + 1e-12:
        raise ValueError(f"channel threshold {threshold} exceeds source mass {parent_mass}")
    if len(masses) == 1:
        if abs(parent_mass - masses[0]) > 1e-10:
            raise ValueError("one-body decay requires equal parent and daughter mass")
        return [FourVector(parent_mass, 0.0, 0.0, 0.0)]

    first_mass = masses[0]
    rest_threshold = sum(masses[1:])
    if len(masses) == 2:
        rest_mass = masses[1]
    else:
        upper = parent_mass - first_mass
        span = max(upper - rest_threshold, 0.0)
        # Bias mildly toward low intermediate mass, leaving kinetic energy for
        # both branches while retaining the complete physical support.
        rest_mass = rest_threshold + span * float(rng.beta(1.5, 2.0))

    pstar = two_body_momentum(parent_mass, first_mass, rest_mass)
    direction = _isotropic_direction(rng)
    first = FourVector(
        math.sqrt(first_mass * first_mass + pstar * pstar),
        *(pstar * direction),
    )
    remainder = FourVector(
        math.sqrt(rest_mass * rest_mass + pstar * pstar),
        *(-pstar * direction),
    )
    if len(masses) == 2:
        return [first, remainder]

    rest_daughters = _decay_rest_frame(rest_mass, masses[1:], rng)
    beta = remainder.spatial() / remainder.e
    return [first] + [boost(daughter, beta) for daughter in rest_daughters]


# ---------------------------------------------------------------------------
# Hypothesis H1: exact Lorentz-invariant phase space
# ---------------------------------------------------------------------------
#
# The n-body phase space factors through the intermediate cluster mass
# M_c (the invariant mass of the remaining daughters) as
#
#     dPhi_n/dM_c ∝ pstar(M, m_1, M_c) * 2 M_c * Phi_{n-1}(M_c),
#
# with an isotropic split at each level. The recursion is exact for every
# n, and the final distribution is independent of the daughter ordering
# (the uniform measure is permutation invariant), so the per-event
# permuted order of ``decay`` is an exact parameterization of the same
# phase space. The cluster-volume ladder V_s(M) (the s-body phase-space
# volume of the last s daughters, up to a positive constant) is evaluated
# once per event on the documented 1024-point grid:
#
#     V_2(M)  = pstar(M, m[k-2], m[k-1]) / M
#     V_s(M)  = (1/M) ∫ dM' pstar(M, m[k-s], M') 2 M' V_{s-1}(M').


def _phsp_pstar_pair(parent: np.ndarray, m1: float, m2: float) -> np.ndarray:
    """Two-body momentum over a parent-mass grid, zero at or below threshold.

    The Kallen argument is the shared vectorized form of the scalar
    ``two_body_momentum`` formula (``excitation_kinematics.kallen``), so the
    H1 ladders and the scalar two-body path cannot drift. The threshold
    mask mirrors ``two_body_momentum_safe``: it also zeroes the inverted
    region ``m1 + m2 >= M`` where the Kallen function is positive again
    although the decay is kinematically forbidden.
    """
    p2 = parent * parent
    b = m1 * m1
    c = m2 * m2
    lam = kallen(p2, b, c)
    pstar = np.sqrt(np.maximum(lam, 0.0)) / (2.0 * parent)
    return np.where(parent > m1 + m2, pstar, 0.0)


def _phsp_pstar_parent(
    parent_mass: float,
    m1: float,
    cluster: np.ndarray,
) -> np.ndarray:
    """pstar(parent_mass, m1, M_c) over a cluster-mass grid, zero below threshold.

    Shares the scalar ``kallen`` formula and threshold mask of
    ``two_body_momentum_safe``.
    """
    c2 = cluster * cluster
    pm2 = parent_mass * parent_mass
    b = m1 * m1
    lam = kallen(pm2, b, c2)
    pstar = np.sqrt(np.maximum(lam, 0.0)) / (2.0 * parent_mass)
    return np.where(parent_mass > m1 + cluster, pstar, 0.0)


def _phsp_pstar_grid(
    parent: np.ndarray,
    m1: float,
    cluster: np.ndarray,
) -> np.ndarray:
    """pstar(M_i, m1, M'_j) over parent-mass rows and cluster-mass columns.

    Shares the scalar ``kallen`` formula and threshold mask of
    ``two_body_momentum_safe``.
    """
    m = parent[:, None]
    c = cluster[None, :]
    b = m1 * m1
    lam = kallen(m * m, b, c * c)
    pstar = np.sqrt(np.maximum(lam, 0.0)) / (2.0 * m)
    return np.where(m > m1 + c, pstar, 0.0)


def _phsp_ladder_level(grid: np.ndarray, m_j: float, v_prev: np.ndarray) -> np.ndarray:
    """One recursion level of the volume ladder: V_s from V_{s-1}."""
    pstar = _phsp_pstar_grid(grid, m_j, grid)
    integrand = pstar * (2.0 * grid)[None, :] * v_prev[None, :]
    integrand = np.where(grid[None, :] <= grid[:, None] - m_j, integrand, 0.0)
    cells = 0.5 * (integrand[:, 1:] + integrand[:, :-1])
    integral = (cells * np.diff(grid)[None, :]).sum(axis=1)
    return integral / grid


def _phsp_volume_ladder(masses: list[float], grid: np.ndarray) -> list[np.ndarray]:
    """Volume ladder V_s for s = 2..len(masses)-1 on the mass grid."""
    m = np.asarray(masses, dtype=np.float64)
    ladder = [_phsp_pstar_pair(grid, m[-2], m[-1]) / grid]
    for s in range(3, len(masses)):
        ladder.append(_phsp_ladder_level(grid, m[-s], ladder[-1]))
    return ladder


def _decay_phsp(
    parent_mass: float,
    masses: list[float],
    rng: np.random.Generator,
) -> list[FourVector]:
    """Recursive phase-space construction (hypothesis H1).

    The cluster mass at each level is drawn from the density
    ``pstar(M, m, M') * 2 M' * V(M')`` on a per-level 1024-point grid
    spanning the physical window of that level. Cell masses are
    trapezoidal and the draw is uniform inside the chosen cell, so the
    sample is a controlled numerical approximation to Lorentz-invariant
    n-body phase space rather than an exact inversion of the density.
    The recursion mirrors ``_decay_rest_frame`` (each split is exact
    two-body kinematics and the remainder is boosted back), so the
    parent four-momentum is conserved to machine precision. A channel
    whose total kinetic margin falls below the ladder grid resolution
    carries no density on the grid and raises ``ValueError``; the event
    builder retries with a fresh draw, so the near-threshold corner
    below the documented grid bias is rejected, not distorted.
    """
    if not masses:
        raise ValueError("a decay channel must contain at least one daughter")
    threshold = sum(masses)
    if threshold > parent_mass + 1e-12:
        raise ValueError(f"channel threshold {threshold} exceeds source mass {parent_mass}")
    if len(masses) == 1:
        if abs(parent_mass - masses[0]) > 1e-10:
            raise ValueError("one-body decay requires equal parent and daughter mass")
        return [FourVector(parent_mass, 0.0, 0.0, 0.0)]
    if len(masses) == 2:
        return list(two_body_split_safe(parent_mass, masses[0], masses[1], rng))

    grid_lo = masses[-2] + masses[-1]
    if parent_mass - grid_lo <= 0.0:
        raise ValueError("degenerate phase-space ladder: daughters fill the parent mass")
    grid = np.linspace(grid_lo, parent_mass, PHSP_GRID_SIZE)
    ladder = _phsp_volume_ladder(masses, grid)

    def rec(idx: int, cluster_mass: float) -> list[FourVector]:
        """Daughters ``masses[idx:]`` from a cluster of invariant mass."""
        rest = masses[idx:]
        if len(rest) == 2:
            return list(two_body_split_safe(cluster_mass, rest[0], rest[1], rng))
        m_j = rest[0]
        cluster_size = len(rest) - 1
        window_lo = sum(rest[1:])
        window_hi = cluster_mass - m_j
        if window_hi <= window_lo:
            raise ValueError("degenerate phase-space ladder at a recursion level")
        xs = np.linspace(window_lo, window_hi, PHSP_GRID_SIZE)
        volume = np.interp(xs, grid, ladder[cluster_size - 2])
        density = _phsp_pstar_parent(cluster_mass, m_j, xs) * (2.0 * xs) * volume
        rest_mass = draw_from_grid_density(rng, xs, density)
        first, remainder = two_body_split_safe(cluster_mass, m_j, rest_mass, rng)
        rest_daughters = rec(idx + 1, rest_mass)
        beta = remainder.spatial() / remainder.e
        return [first] + [boost(daughter, beta) for daughter in rest_daughters]

    return rec(0, parent_mass)


# ---------------------------------------------------------------------------
# Hypothesis H2: explicit sequential resonances
# ---------------------------------------------------------------------------
#
# The permuted daughter list is paired in deterministic order: pass 1
# pairs every ground-state nucleon with the next unpaired pion into a
# Delta(1232) of the pair's charge; pass 2 pairs the remaining charged
# pions into rho(770) pairs ((pi+, pi-), (pi+, pi0), (pi-, pi0)).
# Unpaired particles stay free, and the final particle set is unchanged,
# so the quantum-number check of ``decay`` applies to the pairing as
# well. Each resonance mass is drawn from a truncated p-wave
# Breit-Wigner line shape built from the PDG mass and width with the
# threshold set to the exact daughter sum; a line-shape window carrying
# no mass makes the state ineligible, and the daughters are then emitted
# at threshold (no line). The line-shape shapes are fixed module
# constants; only the PDG mass and width enter, so the hypothesis adds
# no bounded fit parameters.


_H2_NUCLEONS = (2212, 2112, -2212, -2112)
_H2_PIONS = (211, -211, 111)
_H2_RHO_OPENERS = (211, -211)

# Nucleon + pion -> Delta(1232), by charge-conjugate pair.
_H2_DELTA_BY_PAIR = {
    (2212, 211): 2224,
    (2212, 111): 2214,
    (2212, -211): 2114,
    (2112, 211): 2214,
    (2112, 111): 2114,
    (2112, -211): 1114,
    (-2212, -211): -2224,
    (-2212, 111): -2214,
    (-2212, 211): -2114,
    (-2112, -211): -2214,
    (-2112, 111): -2114,
    (-2112, 211): -1114,
}

# Charged pion + pion -> rho(770); the pi0 cannot open a pair.
_H2_RHO_BY_PAIR = {
    (211, -211): 113,
    (211, 111): 213,
    (-211, 211): 113,
    (-211, 111): -213,
}

# Damping windows of the p-wave line shapes (GeV/c^2), carried over from
# the precedent decays-table shapes for the same resonances.
_H2_DELTA_DAMP = (1.6, 2.0)
_H2_RHO_DAMP = (1.2, 1.5)
_H2_DELTA_ABS_IDS = (2224, 2214, 2114, 1114)
_H2_RHO_ABS_IDS = (113, 213)

# Line-shape cache: (abs(res_id), d1 mass, d2 mass) -> BreitWigner.
# Matter and antimatter partners share a line shape (equal daughter
# masses), so the key is the charge magnitude plus the daughter masses.
_H2_LINE_SHAPES: dict[tuple[int, float, float], BreitWigner] = {}


@dataclass(frozen=True)
class _CascadeComponent:
    """A free final particle or a resonance pair in an H2 cascade."""

    kind: str  # "free" or "res"
    index: int  # position in the permuted id list (opener for a pair)
    min_mass: float  # minimum invariant mass of the component
    d1_mass: float = 0.0  # resonance daughter masses (resonance components)
    d2_mass: float = 0.0
    d1_id: int = 0
    d2_id: int = 0
    res_id: int = 0
    partner: int = -1


def _h2_resonant_pairs(ids: list[int]) -> list[tuple[int, int, int]]:
    """Deterministic resonance pairing over the permuted id list.

    Returns ``(opener_index, partner_index, resonance_pdg_id)`` triples in
    list order. Pass 1 pairs every ground-state nucleon with the next
    unpaired pion into a Delta(1232); pass 2 pairs the remaining charged
    pions into rho(770) pairs, skipping over pions that cannot partner
    the opener. The pairing is a function of the id ordering only.
    """
    hidden = [False] * len(ids)
    pairs: list[tuple[int, int, int]] = []
    for i, pid in enumerate(ids):
        if hidden[i] or pid not in _H2_NUCLEONS:
            continue
        for j in range(i + 1, len(ids)):
            if hidden[j] or ids[j] not in _H2_PIONS:
                continue
            hidden[i] = hidden[j] = True
            pairs.append((i, j, _H2_DELTA_BY_PAIR[pid, ids[j]]))
            break
    for i, pid in enumerate(ids):
        if hidden[i] or pid not in _H2_RHO_OPENERS:
            continue
        want = (-211, 111) if pid == 211 else (211, 111)
        for j in range(i + 1, len(ids)):
            if hidden[j] or ids[j] not in want:
                continue
            hidden[i] = hidden[j] = True
            pairs.append((i, j, _H2_RHO_BY_PAIR[pid, ids[j]]))
            break
    return pairs


def _h2_components(ids: list[int], masses: list[float]) -> list[_CascadeComponent]:
    """Cascade components (free particles and resonance pairs) in list order."""
    pairs = _h2_resonant_pairs(ids)
    consumed: set[int] = set()
    for i, j, _ in pairs:
        consumed.add(i)
        consumed.add(j)
    comps = [
        _CascadeComponent(kind="free", index=i, min_mass=masses[i])
        for i in range(len(ids))
        if i not in consumed
    ]
    for i, j, res_id in pairs:
        d1_id, d2_id = ids[i], ids[j]
        d1_mass, d2_mass = masses[i], masses[j]
        comps.append(
            _CascadeComponent(
                kind="res",
                index=i,
                min_mass=d1_mass + d2_mass,
                d1_mass=d1_mass,
                d2_mass=d2_mass,
                d1_id=d1_id,
                d2_id=d2_id,
                res_id=res_id,
                partner=j,
            )
        )
    comps.sort(key=lambda c: c.index)
    return comps


def _h2_line_shape(component: _CascadeComponent) -> BreitWigner:
    """p-wave line shape for an H2 resonance (cached per charge + daughters)."""
    abs_id = abs(component.res_id)
    key = (abs_id, component.d1_mass, component.d2_mass)
    cached = _H2_LINE_SHAPES.get(key)
    if cached is not None:
        return cached
    damp = _H2_DELTA_DAMP if abs_id in _H2_DELTA_ABS_IDS else _H2_RHO_DAMP
    bw = BreitWigner(
        component.d1_mass + component.d2_mass,  # threshold-safe lower edge
        get_mass(abs_id),
        get_width(abs_id),
        1.0,
        damp[0],
        damp[1],
    )
    _H2_LINE_SHAPES[key] = bw
    return bw


def _h2_resonance_mass(
    component: _CascadeComponent,
    cluster_mass: float,
    other_min: float,
    rng: np.random.Generator,
) -> float:
    """Resonance mass from the line shape truncated to the local window.

    The window is the daughter threshold up to the mass that leaves the
    rest of the cascade at its threshold. A window carrying no
    line-shape mass makes the state ineligible; the daughters are then
    emitted at threshold (no line), following the decay-engine
    truncation convention.
    """
    lo = component.min_mass
    hi = cluster_mass - other_min
    if hi <= lo:
        return lo
    mass = draw_truncated_bw(_h2_line_shape(component), rng, lo, hi)
    return lo if mass is None else mass


def _h2_emit(
    component: _CascadeComponent,
    mass: float,
    p4: FourVector,
    rng: np.random.Generator,
) -> list[tuple[int, FourVector]]:
    """Emit the final particles carried by one cascade component.

    Each four-vector carries its position in the permuted id list, so the
    caller can return the daughters in that order and ``decay``'s
    ``zip(ids, daughters)`` assigns each pdg id to its own four-vector.
    """
    if component.kind == "free":
        return [(component.index, p4)]
    d1, d2 = two_body_split_safe(mass, component.d1_mass, component.d2_mass, rng)
    beta = p4.spatial() / p4.e
    return [
        (component.index, boost(d1, beta)),
        (component.partner, boost(d2, beta)),
    ]


def _h2_cascade(
    cluster_mass: float,
    comps: list[_CascadeComponent],
    idx: int,
    rng: np.random.Generator,
) -> list[tuple[int, FourVector]]:
    """Recursive cascade split of a cluster of invariant mass.

    The component at ``idx`` is split off first: a resonance component
    draws its mass from the truncated line shape, a free component is
    on shell. With two components left, the split is the terminal
    two-body kinematics; with more, the remainder keeps its invariant
    mass and the cascade recurses, boosting the daughters back. Returns
    the daughters tagged with their positions in the permuted id list.
    """
    rest = comps[idx + 1:]
    comp = comps[idx]
    if not rest:
        if comp.kind == "res":
            # Single intermediate state: its invariant mass is fixed by
            # the cluster mass, and the daughters are on shell.
            return _h2_emit(
                comp, cluster_mass, FourVector(cluster_mass, 0.0, 0.0, 0.0), rng
            )
        if abs(cluster_mass - comp.min_mass) > 1e-10:
            raise ValueError("one-body decay requires equal parent and daughter mass")
        return [(comp.index, FourVector(cluster_mass, 0.0, 0.0, 0.0))]
    if len(rest) == 1:
        other = rest[0]
        if comp.kind == "res" and other.kind == "res":
            m1 = _h2_resonance_mass(comp, cluster_mass, other.min_mass, rng)
            m2 = _h2_resonance_mass(other, cluster_mass, m1, rng)
        elif comp.kind == "res":
            m1 = _h2_resonance_mass(comp, cluster_mass, other.min_mass, rng)
            m2 = other.min_mass
        elif other.kind == "res":
            m1 = comp.min_mass
            m2 = _h2_resonance_mass(other, cluster_mass, m1, rng)
        else:
            m1, m2 = comp.min_mass, other.min_mass
        p_a, p_b = two_body_split_safe(cluster_mass, m1, m2, rng)
        return _h2_emit(comp, m1, p_a, rng) + _h2_emit(other, m2, p_b, rng)
    rest_min = sum(c.min_mass for c in rest)
    if comp.kind == "res":
        m1 = _h2_resonance_mass(comp, cluster_mass, rest_min, rng)
    else:
        m1 = comp.min_mass
    p_a, p_b = two_body_split_safe(cluster_mass, m1, cluster_mass - m1, rng)
    rest_frame = _h2_cascade(cluster_mass - m1, comps, idx + 1, rng)
    beta = p_b.spatial() / p_b.e
    return _h2_emit(comp, m1, p_a, rng) + [
        (pos, boost(v, beta)) for pos, v in rest_frame
    ]


def _decay_resonant(
    parent_mass: float,
    ids: list[int],
    masses: list[float],
    rng: np.random.Generator,
) -> list[FourVector]:
    """Explicit sequential resonance decay (hypothesis H2).

    The permuted daughter list is paired into Delta(1232) and rho(770)
    states per :func:`_h2_resonant_pairs`; each resonance mass is drawn
    from the truncated p-wave line shape of the local cascade window, and
    the cascade recursion mirrors ``_decay_rest_frame``. Raises
    ``ValueError`` on an ineligible configuration; the event builder
    retries.
    """
    if not ids:
        raise ValueError("a decay channel must contain at least one daughter")
    threshold = sum(masses)
    if threshold > parent_mass + 1e-12:
        raise ValueError(f"channel threshold {threshold} exceeds source mass {parent_mass}")
    if len(ids) == 1:
        if abs(parent_mass - masses[0]) > 1e-10:
            raise ValueError("one-body decay requires equal parent and daughter mass")
        return [FourVector(parent_mass, 0.0, 0.0, 0.0)]
    comps = _h2_components(ids, masses)
    emitted = _h2_cascade(parent_mass, comps, 0, rng)
    # Every position in the permuted id list is filled exactly once (a
    # free position by itself, a pair position by its resonance).
    vecs: list[FourVector | None] = [None] * len(ids)
    for pos, vec in emitted:
        if vecs[pos] is not None:
            raise ValueError("resonant cascade assigned two daughters to one position")
        vecs[pos] = vec
    if any(v is None for v in vecs):
        raise ValueError("resonant cascade left an unassigned daughter")
    return vecs  # type: ignore[return-value]


def _threshold_factor(parent_mass: float, threshold: float, power: float) -> float:
    if parent_mass <= threshold:
        return 0.0
    return max(1.0 - (threshold * threshold) / (parent_mass * parent_mass), 0.0) ** power


def _family_spec(
    qn: QuantumNumbers,
    parameters: ContinuumParameters,
) -> tuple[tuple[list[int], ...], list[tuple[str, list[int], float, float]]]:
    """Channel families for a source quantum-number type.

    Returns ``(nonstrange_options, strange_and_pair)``: the two non-strange
    base charge partners (drawn uniformly) and the strangeness and pair
    families as ``(name, pdg_ids, strength, threshold_power)`` entries. The
    strange-pair entries are the strangeness-zero baryon-plus-two-kaon
    channels; all of them share the ``strange_pair_strength`` prior, so
    isospin-degenerate charge partners carry the same requested weight.
    """
    p = parameters
    if qn == QuantumNumbers(1, 1, 0):
        nonstrange_options = ([2212, 111], [2112, 211])
        strange_and_pair = (
            ("associated_KLambda", [321, 3122], p.strange_lambda_strength, p.strange_threshold_power),
            ("associated_KSigma", [321, 3212], p.strange_sigma_strength, p.strange_threshold_power),
            # Strangeness-zero strange-pair channels (p K+ K-, p K0 aK0,
            # n K+ aK0); each conserves (B, Q, S) = (1, 1, 0).
            ("strange_pair_pKpKm", [2212, 321, -321], p.strange_pair_strength, p.strange_threshold_power),
            ("strange_pair_pK0K0bar", [2212, 311, -311], p.strange_pair_strength, p.strange_threshold_power),
            ("strange_pair_nKpK0bar", [2112, 321, -311], p.strange_pair_strength, p.strange_threshold_power),
            ("baryon_pair", [2212, 2212, -2212], p.pair_strength, p.pair_threshold_power),
        )
    elif qn == QuantumNumbers(1, 0, 0):
        nonstrange_options = ([2112, 111], [2212, -211])
        strange_and_pair = (
            ("associated_KLambda", [311, 3122], p.strange_lambda_strength, p.strange_threshold_power),
            ("associated_KSigma", [321, 3112], p.strange_sigma_strength, p.strange_threshold_power),
            # Strangeness-zero strange-pair channels (n K+ K-, n K0 aK0,
            # p K0 K-); each conserves (B, Q, S) = (1, 0, 0).
            ("strange_pair_nKpKm", [2112, 321, -321], p.strange_pair_strength, p.strange_threshold_power),
            ("strange_pair_nK0K0bar", [2112, 311, -311], p.strange_pair_strength, p.strange_threshold_power),
            ("strange_pair_pK0Km", [2212, 311, -321], p.strange_pair_strength, p.strange_threshold_power),
            ("baryon_pair", [2112, 2212, -2212], p.pair_strength, p.pair_threshold_power),
        )
    else:
        raise ValueError(f"unsupported source quantum numbers {qn}")
    return nonstrange_options, list(strange_and_pair)


def family_requested_weights(
    mass: float,
    qn: QuantumNumbers,
    parameters: ContinuumParameters,
) -> dict[str, float]:
    """Requested family weights for a source of given mass and quantum numbers.

    Returns the explicit candidate weights over the complete family set,
    before renormalization. The non-strange family carries the unit weight;
    each strangeness and pair family carries its configured strength times
    the thresholded opening factor, and a family at or below its physical
    threshold carries zero weight and is absent from the returned dict. For
    a mass below ``match_mass`` the only family is the resolved non-strange
    proxy with unit weight.
    """
    if mass < parameters.match_mass:
        return {"resolved_nonstrange": 1.0}
    _, strange_and_pair = _family_spec(qn, parameters)
    weights: dict[str, float] = {"continuum_nonstrange": 1.0}
    for name, ids, strength, power in strange_and_pair:
        threshold = sum(get_mass(pid) for pid in ids)
        weight = strength * _threshold_factor(mass, threshold, power)
        if weight > 0.0:
            weights[name] = weight
    return weights


class SourceDecayer:
    """Decay fixed-quantum-number sources through a few legal channel families."""

    def __init__(
        self,
        parameters: ContinuumParameters,
        rng: np.random.Generator,
    ) -> None:
        parameters.validate()
        self.parameters = parameters
        self.rng = rng

    def _base_channels(self, source: SourceRecord) -> list[tuple[str, list[int], float]]:
        mass = source.mass
        qn = source.quantum_numbers
        p = self.parameters
        nonstrange_options, strange_and_pair = _family_spec(qn, p)

        legal_nonstrange = [
            list(ids)
            for ids in nonstrange_options
            if sum(get_mass(pid) for pid in ids) <= mass + 1e-12
        ]
        if not legal_nonstrange:
            raise ValueError(f"no represented hadronic channel below source mass {mass}")
        nonstrange = legal_nonstrange[int(self.rng.integers(len(legal_nonstrange)))]

        if mass < p.match_mass:
            return [("resolved_nonstrange", nonstrange, 1.0)]

        candidates = [("continuum_nonstrange", nonstrange, 1.0)]
        for name, ids, strength, power in strange_and_pair:
            threshold = sum(get_mass(pid) for pid in ids)
            weight = strength * _threshold_factor(mass, threshold, power)
            if weight > 0.0:
                candidates.append((name, ids, weight))
        return candidates

    def _extra_pions(self, n_extra: int) -> list[int]:
        """Charge-balanced extra pions for a fixed extra-pion count.

        The count is drawn by the caller with the multiplicity slope and the
        mass-budget clamp. The charge composition is sampled over every
        charge state consistent with the source after the base channel:
        ``k`` (pi+, pi-) pairs plus ``n_extra - 2k`` pi0, with ``k`` drawn
        uniformly over ``0..n_extra // 2``. Every composition carries
        ``(B, Q, S) = (0, 0, 0)``, so the source charge and strangeness are
        preserved.
        """
        if n_extra <= 0:
            return []
        max_pairs = n_extra // 2
        k = int(self.rng.integers(0, max_pairs + 1)) if max_pairs > 0 else 0
        return [211, -211] * k + [111] * (n_extra - 2 * k)

    def _clamp_extra_to_mass_budget(
        self, extra_ids: list[int], remaining: float
    ) -> list[int]:
        """Drop extra pions until the selected PDG masses fit ``remaining``.

        Charged pairs are removed together so the extra set stays at
        ``(B, Q, S) = (0, 0, 0)``. Neutral pions are dropped only after no
        charged pair remains.
        """
        ids = list(extra_ids)
        budget = float(remaining)
        while ids and sum(get_mass(pid) for pid in ids) > budget + 1e-12:
            try:
                i_plus = ids.index(211)
                i_minus = ids.index(-211)
                for index in sorted((i_plus, i_minus), reverse=True):
                    ids.pop(index)
            except ValueError:
                try:
                    ids.remove(111)
                except ValueError:
                    ids.pop()
        return ids

    def decay(self, source: SourceRecord, first_particle_id: int = 0) -> list[FinalParticle]:
        """Decay one source through a requested-weight legal channel family.

        The family is drawn from the renormalized requested weights of
        :func:`family_requested_weights` over the complete family set. The
        extra pions beyond the base channel are charge-balanced per
        :meth:`_extra_pions`. The integer ``(B, Q, S)`` sum of the final set
        is checked against the source before the kinematics run, and the
        thresholded weights keep every family closed below its physical
        threshold.
        """
        candidates = self._base_channels(source)
        weights = np.asarray([item[2] for item in candidates], dtype=float)
        weights /= weights.sum()
        family, base_ids, _ = candidates[
            int(self.rng.choice(len(candidates), p=weights))
        ]

        base_threshold = sum(get_mass(pid) for pid in base_ids)
        mpi0 = get_mass(111)
        max_extra = min(
            int(self.parameters.max_pions),
            max(int(math.floor((source.mass - base_threshold) / mpi0)), 0),
        )
        mean_extra = self.parameters.pion_multiplicity_slope * max(
            source.mass - base_threshold, 0.0
        )
        n_extra = min(int(self.rng.poisson(mean_extra)), max_extra)
        extra_ids = self._clamp_extra_to_mass_budget(
            self._extra_pions(n_extra),
            source.mass - base_threshold,
        )
        ids = list(base_ids) + extra_ids
        # Avoid a fixed recursive ordering becoming a hidden dynamics choice.
        order = self.rng.permutation(len(ids))
        ids = [ids[int(i)] for i in order]
        masses = [get_mass(pid) for pid in ids]
        if sum_quantum_numbers(ids) != source.quantum_numbers:
            raise ValueError(
                f"illegal {family} channel {ids} for source {source.quantum_numbers}"
            )

        # Hypothesis selector (card M7 phase A): H1 is exact phase space,
        # H2 the explicit sequential resonances, and H3 the retained
        # recursive empirical kernel (default, byte-identical stream).
        hypothesis = str(self.parameters.decay_hypothesis).strip().upper()
        if hypothesis == "H1":
            daughters_rest = _decay_phsp(source.mass, masses, self.rng)
        elif hypothesis == "H2":
            daughters_rest = _decay_resonant(source.mass, ids, masses, self.rng)
        else:
            daughters_rest = _decay_rest_frame(source.mass, masses, self.rng)
        beta = source.p4.spatial() / source.p4.e
        daughters = [boost(p4, beta) for p4 in daughters_rest]
        return [
            FinalParticle(
                particle_id=first_particle_id + i,
                pdg_id=pid,
                p4=p4,
                source_id=source.source_id,
                parent_id=source.source_id,
                channel_family=family,
            )
            for i, (pid, p4) in enumerate(zip(ids, daughters))
        ]
