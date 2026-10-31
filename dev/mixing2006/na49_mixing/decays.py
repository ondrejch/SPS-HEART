"""Decay table — port of ``PartDecays.h`` / ``PartDecays.cxx``.

Each :class:`DecayMode` holds the daughter PDG IDs and optional
:class:`~na49_mixing.breit_wigner.BreitWigner` objects for non-stable
intermediate masses.

:class:`ParticleDecays` groups all decay modes for one parent particle
type and provides random-mode selection.
"""

from __future__ import annotations
import math
import sys
from typing import Optional, List, Tuple
import numpy as np

from .breit_wigner import (
    BreitWigner,
    bw_draw_window,
    bw_fraction_above,
    bw_truncated_fraction,
)
from .pdg_database import get_mass


# Parents whose represented decays are WEAK (strangeness may change;
# charge and baryon number may not).  Used for ancestry tagging and for
# quantum-number audits; strong/electromagnetic parents must conserve S.
WEAK_DECAY_PARENTS = frozenset({3122, -3122, 3222, 3112, 310})


# Tolerance for threshold comparisons at the realized parent mass (GeV/c²).
# The fixed-mass threshold check inside the decay routines uses the same
# floats, so the selection-time and draw-time decisions stay consistent.
MASS_CONDITION_TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# Thresholded p pbar pair family (card TASK-2026-08-31 rev 2, decision D5)
# ---------------------------------------------------------------------------
#
# The pair-capable effective-meson levels M_eff(850) (l2) and M_eff(1250)
# (l3) register an exactly conserving two-body p pbar channel
# (B = Q = S = 0 on every M_eff table entry).  One aggregate family
# strength ``mestar_eff_ppbar_strength`` (default 0.0) carries the family;
# its share of the daughter-side decay budget at a realized parent mass M
# is the fixed thresholded shape
#
#     P_ppbar(M) = c * (1 - (M_th / M)^2) ** n     for M > M_th,  else 0
#
# with M_th = 2 m_p (GeV/c²) and exponent n = 1, both fixed and recorded
# here (n is a configuration constant, never a fit parameter).  The share
# is hard zero below M_th — below the threshold the channel is ineligible
# in C(M) by its daughter-mass sum, so it is excluded, never
# sampled-then-retried — and continuous at M_th with right limit zero.
# The surviving declared pionic weights (pipi / mestarpi / rhopi,
# including the residual-to-pipi slack) scale proportionally by
# (1 - P_ppbar(M)), so the level budget sums to exactly 1 at every
# realized mass.  l1 (M_eff(450)) registers no pair channel: raising it to
# pair-capable masses would compound the whole tower and is outside the
# card's scope.

MESTAR_PPBAR_STRENGTH_KEY = "mestar_eff_ppbar_strength"
MESTAR_PPBAR_STRENGTH_DEFAULT = 0.0

# Recorded three-body p pbar pi0 option behind the same strength key
# (threshold 2 m_p + m_pi0 = 2.0115 GeV).  Default OFF; when enabled the
# family share splits between the two variants by the fixed fraction
# below.  The rho0-driven pi+ pi- variant (threshold 2 m_p + m_rho0
# = 2.6518 GeV) is unreachable within the 2.5 GeV meson bound of the
# stage-B box and is not registered.
MESTAR_PPBAR_PI0_VARIANT_KEY = "mestar_eff_ppbar_pi0_variant"
MESTAR_PPBAR_PI0_VARIANT_DEFAULT = 0.0
PPBAR_PAIR_PI0_VARIANT_SHARE = 0.5
# Round 7 (pre-registered 2026-09-03): the two/three-body split becomes
# overridable; the default keeps the recorded 50/50 split bit-identical.
MESTAR_PPBAR_PI0_SHARE_KEY = "mestar_eff_ppbar_pi0_share"
MESTAR_PPBAR_PI0_SHARE_DEFAULT = PPBAR_PAIR_PI0_VARIANT_SHARE

# Shape exponent n of the thresholded pair weight. The recorded value
# is the default; round 4 (pre-registered 2026-09-03 in
# notes/resonance_fit_stage_progress.md) scans it through the override
# key below, whose default path is bit-identical to the constant.
PPBAR_PAIR_EXPONENT = 1.0
MESTAR_PPBAR_EXPONENT_KEY = "mestar_eff_ppbar_exponent"
MESTAR_PPBAR_EXPONENT_DEFAULT = PPBAR_PAIR_EXPONENT

# Round 6 (pre-registered 2026-09-03): an alternative step-plus-roll-off
# law, P(M) = c / (1 + exp((M_0 - M)/w)) for M > M_th (zero below; the
# hard pair threshold is unchanged), selected by the flag override. The
# default flag 0 keeps the thresholded power law bit-identical. M_0 is
# the roll-on midpoint and w the roll-off width; w -> 0 approaches a
# pure step, and the n -> 0 limit of the power family is reached
# smoothly for M_0 well below the realized parent masses.
MESTAR_PPBAR_LAW_FLAG_KEY = "mestar_eff_ppbar_law_flag"
MESTAR_PPBAR_LAW_FLAG_DEFAULT = 0.0
MESTAR_PPBAR_ROLLON_M0_KEY = "mestar_eff_ppbar_rollon_m0"
MESTAR_PPBAR_ROLLON_M0_DEFAULT = 3.0
MESTAR_PPBAR_ROLLOFF_W_KEY = "mestar_eff_ppbar_rolloff_w"
MESTAR_PPBAR_ROLLOFF_W_DEFAULT = 0.4

# Daughter PDG ids of the registered pair channels.  Proton, antiproton,
# and pi0 are fixed PDG-table entries consumed via ``bws=None``, so the
# pair family needs no new daughter-mass bookkeeping anywhere.
PPBAR_PAIR_TWO_BODY_DAUGHTERS = (2212, -2212)
PPBAR_PAIR_THREE_BODY_DAUGHTERS = (2212, -2212, 111)


def ppbar_pair_threshold() -> float:
    """Pair-family threshold ``2 m_p`` (GeV/c²), derived from the PDG table."""
    return 2.0 * float(get_mass(2212))


class PairFamilyLaw:
    """Mass-conditioned weight law of one level's p pbar pair family.

    Attached to a :class:`ParticleDecays` table as ``pair_family``; the
    mass-conditioned selection (see
    :meth:`ParticleDecays.renormalized_mode_weights`) then replaces the
    declared pair-channel weights by the thresholded share evaluated at
    the realized parent mass and rescales the surviving declared pionic
    weights proportionally.

    Parameters
    ----------
    strength : float
        Aggregate family strength ``c`` (shared by l2 and l3 by
        construction; the P3 driver fits one ``c`` per candidate).
    threshold : float
        Pair threshold ``M_th`` in GeV/c² (``2 m_p``).
    exponent : float
        Fixed shape exponent ``n`` (recorded configuration, not fitted).
    members : tuple of (int, float)
        ``(channel index, fraction of the share)`` pairs of the
        registered pair channels.  With the three-body pi0 variant off
        the two-body channel carries the full share.
    """

    __slots__ = ("strength", "threshold", "exponent", "members", "law", "rollon_m0", "rolloff_w")

    def __init__(
        self,
        strength: float,
        threshold: float,
        exponent: float = PPBAR_PAIR_EXPONENT,
        members: Tuple[Tuple[int, float], ...] = (),
        law: float = 0.0,
        rollon_m0: float = 3.0,
        rolloff_w: float = 0.4,
    ) -> None:
        c = float(strength)
        m_th = float(threshold)
        n = float(exponent)
        law_flag = float(law)
        m0 = float(rollon_m0)
        w = float(rolloff_w)
        if law_flag not in (0.0, 1.0):
            raise ValueError(
                f"PairFamilyLaw law flag must be 0 (thresholded power) or "
                f"1 (step-plus-roll-off), got {law!r}"
            )
        if not (math.isfinite(m0) and m0 > 0.0):
            raise ValueError(
                f"PairFamilyLaw roll-on midpoint must be finite and > 0, "
                f"got {rollon_m0!r}"
            )
        if not (math.isfinite(w) and w > 0.0):
            raise ValueError(
                f"PairFamilyLaw roll-off width must be finite and > 0, "
                f"got {rolloff_w!r}"
            )
        if not (math.isfinite(c) and c >= 0.0):
            raise ValueError(
                f"PairFamilyLaw strength must be finite and >= 0, got {strength!r}"
            )
        if not (math.isfinite(m_th) and m_th > 0.0):
            raise ValueError(
                f"PairFamilyLaw threshold must be finite and > 0, got {threshold!r}"
            )
        if not (math.isfinite(n) and n >= 0.0):
            raise ValueError(
                f"PairFamilyLaw exponent must be finite and >= 0, got {exponent!r}"
            )
        cleaned: List[Tuple[int, float]] = []
        frac_sum = 0.0
        for idx, frac in members:
            i = int(idx)
            f = float(frac)
            if i < 0 or not (math.isfinite(f) and 0.0 <= f <= 1.0):
                raise ValueError(
                    f"PairFamilyLaw member ({idx!r}, {frac!r}) is not a valid "
                    f"(channel index, share fraction) pair"
                )
            cleaned.append((i, f))
            frac_sum += f
        if cleaned and abs(frac_sum - 1.0) > 1e-9:
            raise ValueError(
                f"PairFamilyLaw member share fractions must sum to 1, "
                f"got {frac_sum!r}"
            )
        self.strength = c
        self.threshold = m_th
        self.exponent = n
        self.law = law_flag
        self.rollon_m0 = m0
        self.rolloff_w = w
        self.members: Tuple[Tuple[int, float], ...] = tuple(cleaned)

    def member_fraction(self, index: int) -> Optional[float]:
        """Share fraction of channel *index*, or ``None`` for non-members."""
        for i, f in self.members:
            if i == int(index):
                return f
        return None

    def share(self, parent_mass: float) -> float:
        """Thresholded pair share ``P_ppbar(M)`` at the realized parent mass.

        Exactly zero for ``M <= M_th`` (the channel is ineligible in C(M)
        there), continuous at ``M_th`` with right limit zero, growing
        toward the strength ``c`` above.
        """
        m = float(parent_mass)
        if not m > self.threshold:
            return 0.0
        if self.law >= 0.5:
            # Step-plus-roll-off (round 6): logistic roll-on around M_0
            # with width w; the hard pair threshold below M_th is
            # unchanged. w -> 0 approaches a pure step.
            return self.strength / (
                1.0 + math.exp((self.rollon_m0 - m) / self.rolloff_w)
            )
        return self.strength * (1.0 - (self.threshold / m) ** 2) ** self.exponent

    def __repr__(self) -> str:
        return (
            f"PairFamilyLaw(strength={self.strength}, "
            f"threshold={self.threshold}, exponent={self.exponent}, "
            f"law={self.law}, rollon_m0={self.rollon_m0}, "
            f"rolloff_w={self.rolloff_w}, members={self.members})"
        )


class NoLegalChannelError(ValueError):
    """A parent has no legal decay channel at its realized mass.

    Raised by the mass-conditioned selection when the legal-channel set
    C(M) is empty under the source constraint. Distinct from the plain
    ``ValueError`` raised for structurally broken decay tables (no modes,
    incomplete cumulative), so the decay engine can treat an empty C(M)
    as an undecayed parent while surfacing table corruption loudly.
    """


class DecayMode:
    """A single decay mode.

    Parameters
    ----------
    n_daughters : int
        Number of daughters (2 or 3).
    daughters : list of int
        PDG IDs of the daughters.
    bws : list of BreitWigner or None
        Optional Breit-Wigner for each daughter's mass.  ``None`` means
        use the PDG table mass.
    """

    def __init__(
        self,
        n_daughters: int,
        daughters: List[int],
        bws: Optional[List[Optional[BreitWigner]]] = None,
    ) -> None:
        n = int(n_daughters)
        if n not in (2, 3):
            raise ValueError(
                f"DecayMode n_daughters must be 2 or 3, got {n_daughters!r}"
            )
        dlist = list(daughters)
        if len(dlist) != n:
            raise ValueError(
                f"DecayMode expected {n} daughters, got {len(dlist)}: {dlist!r}"
            )
        if bws is None:
            bwlist: List[Optional[BreitWigner]] = [None] * n
        else:
            bwlist = list(bws)
            if len(bwlist) != n:
                raise ValueError(
                    f"DecayMode expected {n} Breit-Wigner slots, got {len(bwlist)}"
                )
        self.n_daughters = n
        self.daughters: List[int] = dlist
        self.bws: List[Optional[BreitWigner]] = bwlist

    def __repr__(self) -> str:
        return (
            f"DecayMode(n={self.n_daughters}, "
            f"daughters={self.daughters}, "
            f"has_bw={[b is not None for b in self.bws]})"
        )


class ParticleDecays:
    """Decay table for one parent particle type.

    Parameters
    ----------
    parent_id : int
        PDG ID of the parent.
    rng : numpy.random.Generator, optional
        Random-number generator to use.
    """

    def __init__(
        self,
        parent_id: int = 2224,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.parent_id = int(parent_id)
        self._modes: List[DecayMode] = []
        self._probabilities: List[float] = []  # cumulative
        self._rng = rng if rng is not None else np.random.default_rng()
        # Optional D5 pair-family law (thresholded p pbar share evaluated
        # at the realized parent mass).  ``None`` keeps the table plain:
        # the mass-conditioned selection then behaves exactly as in the
        # source-constrained machinery without the pair family.
        self.pair_family: Optional[PairFamilyLaw] = None

    def add_mode(self, probability: float, mode: DecayMode) -> None:
        """Add a decay mode with the given cumulative *probability* in [0, 1].

        Cumulative values must be finite and monotonically nondecreasing.
        """
        p = float(probability)
        if not math.isfinite(p) or p < 0.0 or p > 1.0:
            raise ValueError(
                f"ParticleDecays.add_mode: cumulative probability must be "
                f"finite and in [0, 1], got {probability!r} for parent "
                f"{self.parent_id}"
            )
        if self._probabilities and p + 1e-15 < self._probabilities[-1]:
            raise ValueError(
                f"ParticleDecays.add_mode: cumulative probabilities must be "
                f"nondecreasing, got {p} after {self._probabilities[-1]} "
                f"for parent {self.parent_id}"
            )
        if not isinstance(mode, DecayMode):
            raise TypeError(
                f"ParticleDecays.add_mode expected DecayMode, got {type(mode)!r}"
            )
        self._modes.append(mode)
        self._probabilities.append(p)

    def get_parent(self) -> int:
        return self.parent_id

    def n_modes(self) -> int:
        return len(self._modes)

    def get_mode(self) -> DecayMode:
        """Randomly select a decay mode according to the cumulative probabilities."""
        if not self._modes:
            raise ValueError(
                f"ParticleDecays.get_mode: parent {self.parent_id} has no modes"
            )
        last = float(self._probabilities[-1])
        if last < 1.0 - 1e-9:
            raise ValueError(
                f"ParticleDecays.get_mode: parent {self.parent_id} final "
                f"cumulative probability is {last}, expected 1 within 1e-9"
            )
        r = self._rng.random()
        for i, prob in enumerate(self._probabilities):
            if r <= prob:
                return self._modes[i]
        return self._modes[-1]

    def incremental_weights(self) -> List[float]:
        """Declared per-channel (non-cumulative) branching weights.

        Recovered from the loaded cumulative table by first differences;
        ``weights[i]`` sums to 1 over the declared channels within the
        cumulative table's 1e-9 completeness tolerance.
        """
        weights: List[float] = []
        prev = 0.0
        for p in self._probabilities:
            weights.append(float(p) - prev)
            prev = float(p)
        return weights

    def channel_keys(self) -> List[Tuple[int, ...]]:
        """Daughter-PDG tuples of the declared channels, in table order.

        Public accessor for the engine's per-channel audit tables, so
        consumers never need to reach into the private mode list.
        """
        return [tuple(int(x) for x in m.daughters) for m in self._modes]

    def _has_active_pair_family(self) -> bool:
        family = self.pair_family
        return family is not None and family.strength > 0.0

    def renormalized_mode_weights(
        self,
        parent_mass: float,
        resolved_cap: Optional[float] = None,
    ) -> List[float]:
        """Explicitly renormalized per-channel weights ``P(c|M)`` over C(M).

        The legal-channel set C(M) is evaluated at the realized parent
        mass: a channel carries its declared incremental weight when it is
        eligible (minimum daughter-mass sum under
        ``min(parent_mass, resolved_cap)`` and every Breit-Wigner slot
        keeps a positive truncated line-shape integral) and zero
        otherwise. When every declared channel is legal on a plain table
        (no active pair family) the returned vector IS the declared
        incremental weight vector, so the selection below reduces to
        :meth:`get_mode` bit for bit.

        A table carrying an active D5 pair family
        (:class:`PairFamilyLaw`) replaces the pair-channel weights by the
        thresholded share ``P_ppbar(M)`` at the realized parent mass and
        scales the surviving declared pionic weights proportionally by
        ``(1 - P_ppbar(M))``, so the level budget sums to exactly 1 before
        the C(M) renormalization; below the pair threshold the share is
        zero and the declared vector is recovered unchanged.

        Raises ``NoLegalChannelError`` when C(M) is empty (the parent
        cannot decay at this mass) and ``ValueError`` when the declared
        table is structurally incomplete.
        """
        if not self._modes:
            raise ValueError(
                f"ParticleDecays.renormalized_mode_weights: parent "
                f"{self.parent_id} has no modes"
            )
        m_parent = float(parent_mass)
        # Declared-table completeness check ahead of the eligibility
        # branch, so the restricted path surfaces table corruption
        # identically to the all-legal fast path (review REV-fc520a5-04).
        last = float(self._probabilities[-1])
        if last < 1.0 - 1e-9:
            raise ValueError(
                f"ParticleDecays.renormalized_mode_weights: parent "
                f"{self.parent_id} final cumulative probability is "
                f"{last}, expected 1 within 1e-9"
            )
        incs = self.incremental_weights()
        eligible = [
            conditional_mode_weight(mode, inc, m_parent, resolved_cap)
            for mode, inc in zip(self._modes, incs)
        ]
        if self._has_active_pair_family():
            family = self.pair_family
            assert family is not None  # narrows Optional for the type checker
            share = family.share(m_parent)
            complement = 1.0 - share
            adjusted: List[float] = []
            for i, (mode, w) in enumerate(zip(self._modes, eligible)):
                frac = family.member_fraction(i)
                if frac is None:
                    adjusted.append(w * complement)
                elif mode_mass_windows(mode, m_parent, resolved_cap) is not None:
                    adjusted.append(share * frac)
                else:
                    adjusted.append(0.0)
            eligible = adjusted
        elif all(w > 0.0 for w in eligible):
            # Fast path on a plain table: every declared channel is legal
            # at this mass, so the renormalized table IS the loaded table.
            return incs
        total = 0.0
        for w in eligible:
            if w > 0.0:
                total += w
        if not (math.isfinite(total) and total > 0.0):
            raise NoLegalChannelError(
                f"ParticleDecays.renormalized_mode_weights: parent "
                f"{self.parent_id} has no legal decay channel at mass "
                f"{m_parent}"
            )
        return [w / total if w > 0.0 else 0.0 for w in eligible]

    def get_mode_conditional(
        self,
        parent_mass: float,
        resolved_cap: Optional[float] = None,
    ):
        """Select a decay mode from the mass-conditioned channel set C(M).

        The legal-channel set is built BEFORE selection at the realized
        parent mass (see :meth:`renormalized_mode_weights`); branch
        weights are renormalized explicitly over C(M):
        ``P(c|M) = w_c(M) / sum_c' w_c'(M)``.

        Returns ``(mode, mode_index, p_cond, openness, weights)``: the
        selected mode, its declared-table index, its explicitly
        renormalized probability, the channel openness fraction at the
        realized parent mass, and the full renormalized weight vector
        over the declared channels (zeros outside C(M)) for audit
        accumulation. The index and :meth:`channel_keys` let audit
        consumers resolve channel keys without touching the private mode
        list (review REV-fc520a5-03). Consumes exactly one uniform random
        number; when every declared channel of a plain table is legal the
        walk runs over the loaded cumulative table unchanged, so the
        selection draw matches :meth:`get_mode` bit for bit in that case.

        Raises ``NoLegalChannelError`` when C(M) is empty (the parent
        cannot decay at this mass) and ``ValueError`` when the declared
        table is structurally broken.
        """
        if not self._modes:
            raise ValueError(
                f"ParticleDecays.get_mode_conditional: parent "
                f"{self.parent_id} has no modes"
            )
        m_parent = float(parent_mass)
        weights = self.renormalized_mode_weights(m_parent, resolved_cap)
        plain = not self._has_active_pair_family()

        if plain and all(w > 0.0 for w in weights):
            # Fast path: every declared channel is legal at this mass, so
            # the renormalized table IS the loaded table. Walk the loaded
            # cumulative floats unchanged (identical to get_mode()).
            r = self._rng.random()
            for i, prob in enumerate(self._probabilities):
                if r <= prob:
                    mode = self._modes[i]
                    return (
                        mode,
                        i,
                        weights[i],
                        mode_openness_fraction(mode, m_parent, resolved_cap),
                        weights,
                    )
            mode = self._modes[-1]
            return (
                mode,
                len(self._modes) - 1,
                weights[-1],
                mode_openness_fraction(mode, m_parent, resolved_cap),
                weights,
            )

        # Restricted path (and every selection on a table with an active
        # pair family, whose declared cumulative does not carry the
        # mass-conditioned weights): draw one mode from the renormalized
        # cumulative table (one uniform, same walk semantics; channels
        # outside C(M) carry zero weight and are skipped by the walk).
        cumulative: List[Optional[float]] = [None] * len(self._modes)
        acc = 0.0
        last_eligible_i = 0
        for i, w in enumerate(weights):
            if w <= 0.0:
                continue
            acc += w
            cumulative[i] = acc
            last_eligible_i = i
        cumulative[last_eligible_i] = 1.0
        r = self._rng.random()
        for i, cum_prob in enumerate(cumulative):
            if cum_prob is None:
                continue
            if r <= cum_prob:
                mode = self._modes[i]
                return (
                    mode,
                    i,
                    weights[i],
                    mode_openness_fraction(mode, m_parent, resolved_cap),
                    weights,
                )
        mode = self._modes[last_eligible_i]
        return (
            mode,
            last_eligible_i,
            weights[last_eligible_i],
            mode_openness_fraction(mode, m_parent, resolved_cap),
            weights,
        )

    def __repr__(self) -> str:
        return (
            f"ParticleDecays(parent={self.parent_id}, "
            f"n_modes={len(self._modes)})"
        )


# ---------------------------------------------------------------------------
# Mass-conditioned (source-constrained) channel selection
# ---------------------------------------------------------------------------
#
# Replaces the previous sample-then-retry channel handling (card
# TASK-2026-08-31, decision D1): line shapes are truncated to the
# kinematically allowed interval and channel probabilities are renormalized
# over the legal-channel set C(M) built at the realized parent mass, so an
# impossible mode is excluded by construction instead of being sampled and
# later replaced through an implicit retry.

def daughter_minimum_mass(pdg_id: int, bw: Optional[BreitWigner]) -> float:
    """Minimum drawable mass (GeV/c²) of one daughter slot.

    Fixed-mass daughters consume the PDG table mass. Breit-Wigner daughters
    take the lower edge of their line-shape support (``schwel``), which is
    the smallest mass the spectral draw can return.
    """
    if bw is None:
        return float(get_mass(pdg_id, default=0.0))
    return float(bw.get_parameter(0))


def mode_mass_windows(
    mode: DecayMode,
    parent_mass: float,
    resolved_cap: Optional[float] = None,
) -> Optional[List[tuple]]:
    """Per-daughter draw windows for *mode* at the realized parent mass.

    Returns ``None`` when the mode is ineligible: the minimum daughter-mass
    sum does not fit under ``min(parent_mass, resolved_cap)``, or any
    Breit-Wigner daughter slot keeps no line-shape mass inside its window
    (including the zero-width delta case with its pole outside the cap).
    Fixed-mass slots collapse to the point window ``(m, m)``; Breit-Wigner
    slots are clipped against the realized parent mass, the resolved cap
    and the line-shape upper cutoff.
    """
    m_parent = float(parent_mass)
    bound = m_parent if resolved_cap is None else min(m_parent, float(resolved_cap))
    min_masses = [
        daughter_minimum_mass(pid, bw)
        for pid, bw in zip(mode.daughters, mode.bws)
    ]
    if sum(min_masses) > bound + MASS_CONDITION_TOLERANCE:
        return None
    windows: List[tuple] = []
    for slot, (pid, bw) in enumerate(zip(mode.daughters, mode.bws)):
        if bw is None:
            windows.append((min_masses[slot], min_masses[slot]))
            continue
        others_min = sum(min_masses) - min_masses[slot]
        window = bw_draw_window(bw, bound - others_min)
        if window is None:
            return None
        windows.append(window)
    return windows


def conditional_mode_weight(
    mode: DecayMode,
    base_weight: float,
    parent_mass: float,
    resolved_cap: Optional[float] = None,
) -> float:
    """Per-channel weight ``w_c(M)`` evaluated at the realized parent mass.

    Zero when the channel is ineligible at *parent_mass* under the
    resolved cap; otherwise the declared incremental branching weight.
    The evaluation point is the realized parent mass by construction:
    this is the hook through which the thresholded ``p pbar`` pair weight
    of the D5 family enters the daughter-side budget (the family share is
    applied in :meth:`ParticleDecays.renormalized_mode_weights` on top of
    these per-channel eligibility evaluations).
    """
    inc = float(base_weight)
    if not (math.isfinite(inc) and inc > 0.0):
        return 0.0
    if mode_mass_windows(mode, parent_mass, resolved_cap) is None:
        return 0.0
    return inc


def mode_openness_fraction(
    mode: DecayMode,
    parent_mass: float,
    resolved_cap: Optional[float] = None,
) -> float:
    """Openness fraction of one channel at the realized parent mass.

    Product over the channel's Breit-Wigner daughter slots of the fraction
    of that line shape's total integral surviving inside the legal draw
    window; channels whose daughters all carry fixed masses contribute
    1.0. The stage-B machinery extends this diagnostic with the parent
    line-shape wiring; at the engine level it measures how much of the
    drawn spectral support survives the realized-mass truncation. A
    channel that is ineligible at this mass has zero openness.
    """
    windows = mode_mass_windows(mode, parent_mass, resolved_cap)
    if windows is None:
        return 0.0
    fraction = 1.0
    for (pid, bw), (lo, hi) in zip(zip(mode.daughters, mode.bws), windows):
        if bw is None:
            continue
        fraction *= bw_truncated_fraction(bw, lo, hi)
    return fraction


def channel_parent_openness(
    mode: DecayMode,
    parent_bw: Optional[BreitWigner],
    resolved_cap: Optional[float] = None,
) -> float:
    """Parent-side line-shape openness of one channel (card decision D1).

    Fraction of the parent's Breit-Wigner line-shape integral above the
    channel threshold — the minimum daughter-mass sum, capped by the
    resolved cap when active. This is the peak-openness diagnostic of the
    stage-B machinery: it measures how much of the parent mass
    distribution admits the channel, so a fitted candidate whose channel
    is open at the line-shape pole verifies its peak-openness through this
    fraction rather than through the box bounds alone. It complements
    :func:`mode_openness_fraction`, which measures the daughter-slot
    truncation at one realized parent mass; the two are recorded
    side by side in the engine's audit tables when a parent line shape is
    supplied to :class:`~na49_mixing.event.Event`.
    """
    if parent_bw is None:
        return 0.0
    threshold = sum(
        daughter_minimum_mass(pid, bw)
        for pid, bw in zip(mode.daughters, mode.bws)
    )
    cap = None if resolved_cap is None else float(resolved_cap)
    return bw_fraction_above(parent_bw, threshold, cap)


# ---------------------------------------------------------------------------
# File I/O — mirrors the C++ istream/ostream operators
# ---------------------------------------------------------------------------

def read_decays(filename: str, rng: Optional[np.random.Generator] = None) -> dict:
    """Parse a decay-table file.

    File format (same as the C++ code)::

        N_particle_types
        parent_pdg_id  n_modes
        cum_prob  n_daughters  daughter1_id  has_bw [6 bw params]  daughter2_id  has_bw ...
        ...

    Returns a dict mapping parent PDG ID → :class:`ParticleDecays`.
    """
    if rng is None:
        rng = np.random.default_rng()
    result: dict[int, ParticleDecays] = {}

    with open(filename) as fh:
        tokens = fh.read().split()

    idx = 0

    def next_token() -> str:
        nonlocal idx
        t = tokens[idx]
        idx += 1
        return t

    n_parts = int(next_token())
    for _ in range(n_parts):
        parent_id = int(next_token())
        n_modes = int(next_token())
        decays = ParticleDecays(parent_id, rng=rng)
        for _ in range(n_modes):
            cum_prob = float(next_token())
            n_daughters = int(next_token())
            daughters = []
            bws: List[Optional[BreitWigner]] = []
            for _ in range(n_daughters):
                d_id = int(next_token())
                has_bw = int(next_token())
                daughters.append(d_id)
                if has_bw:
                    p0 = float(next_token())
                    p1 = float(next_token())
                    p2 = float(next_token())
                    p3 = float(next_token())
                    p4 = float(next_token())
                    p5 = float(next_token())
                    bws.append(BreitWigner(p0, p1, p2, p3, p4, p5, rng=rng))
                else:
                    bws.append(None)
            mode = DecayMode(n_daughters, daughters, bws)
            try:
                decays.add_mode(cum_prob, mode)
            except ValueError as exc:
                raise ValueError(
                    f"{filename}: parent {parent_id}: {exc}"
                ) from exc
        if not decays._probabilities or float(decays._probabilities[-1]) < 1.0 - 1e-9:
            last = decays._probabilities[-1] if decays._probabilities else 0.0
            raise ValueError(
                f"{filename}: parent {parent_id} final cumulative "
                f"probability is {last}, expected 1"
            )
        result[parent_id] = decays

    return result


def write_decays(filename: str, decay_dict: dict) -> None:
    """Write a decay table to *filename* in the same format as the C++ code."""
    lines = [str(len(decay_dict))]
    for parent_id, decays in decay_dict.items():
        lines.append(f"{parent_id} {decays.n_modes()}")
        for i in range(decays.n_modes()):
            mode = decays._modes[i]
            prob = decays._probabilities[i]
            tokens = [str(prob), str(mode.n_daughters)]
            for j in range(mode.n_daughters):
                tokens.append(str(mode.daughters[j]))
                bw = mode.bws[j]
                if bw is not None:
                    tokens.append("1")
                    tokens.append(str(bw))
                else:
                    tokens.append("0")
            lines.append(" ".join(tokens))
    with open(filename, "w") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Programmatic decay-table factories for rho and omega mesons
# ---------------------------------------------------------------------------

def build_rho_decays(rng: Optional[np.random.Generator] = None) -> dict:
    """Build decay tables for ρ mesons (ground state and excited resonances).

    Only ρ⁰ charge states are included because the ρ±→π±π⁰ channel produces
    a single charged final-state pion and is not useful for invariant-mass
    analyses that require two charged tracks.

    Returns a dict mapping parent PDG ID → :class:`ParticleDecays`.

    Resonances included:

    * **ρ⁰(770)** (PDG 113): π⁺ + π⁻ (100 %)
    * **ρ⁰(1450)** (PDG 100113): π⁺ + π⁻ (100 %)
    * **ρ⁰(1700)** (PDG 30113): π⁺ + π⁻ (100 %)
    """
    if rng is None:
        rng = np.random.default_rng()

    result: dict[int, ParticleDecays] = {}

    # ρ⁰(770) → π⁺ + π⁻
    d = ParticleDecays(113, rng=rng)
    d.add_mode(1.0, DecayMode(2, [211, -211]))
    result[113] = d

    # ρ⁰(1450) → π⁺ + π⁻
    d = ParticleDecays(100113, rng=rng)
    d.add_mode(1.0, DecayMode(2, [211, -211]))
    result[100113] = d

    # ρ⁰(1700) → π⁺ + π⁻
    d = ParticleDecays(30113, rng=rng)
    d.add_mode(1.0, DecayMode(2, [211, -211]))
    result[30113] = d

    return result


def build_omega_decays(rng: Optional[np.random.Generator] = None) -> dict:
    """Build decay tables for ω mesons (ground state and excited resonances).

    Returns a dict mapping parent PDG ID → :class:`ParticleDecays`.

    Resonances included:

    * **ω(782)** (PDG 223):

      - Mode 1 (89.2 %): π⁺ + π⁻ + π⁰  (3-body)
      - Mode 2 ( 1.53 %): π⁺ + π⁻       (2-body, cumulative prob 0.9073)

    * **ω(1420)** (PDG 100223): ρ⁰(113) + π⁰  (ρ⁰ sampled from BW)
    * **ω(1650)** (PDG 30223):  ρ⁰(113) + π⁰  (ρ⁰ sampled from BW)
    """
    if rng is None:
        rng = np.random.default_rng()

    result: dict[int, ParticleDecays] = {}

    # ρ⁰ BW parameters used as the daughter mass distribution in ω(1420/1650)
    rho_bw_params = (0.2791, 0.77526, 0.1491, 1.0, 1.2, 1.5)

    # ω(782) — represented hadronic modes, renormalized to unity per the
    # library convention (unrepresented radiative/CKM-suppressed modes are
    # folded out; mirrors the soft-factory ω entry).
    d = ParticleDecays(223, rng=rng)
    d.add_mode(0.892 / 0.9073, DecayMode(3, [211, -211, 111]))
    d.add_mode(1.0, DecayMode(2, [211, -211]))
    result[223] = d

    # ω(1420) → ρ⁰ + π⁰  (ρ⁰ sampled from Breit-Wigner)
    rho_bw = BreitWigner(*rho_bw_params, rng=rng)
    d = ParticleDecays(100223, rng=rng)
    d.add_mode(1.0, DecayMode(2, [113, 111], [rho_bw, None]))
    result[100223] = d

    # ω(1650) → ρ⁰ + π⁰  (ρ⁰ sampled from Breit-Wigner)
    rho_bw = BreitWigner(*rho_bw_params, rng=rng)
    d = ParticleDecays(30223, rng=rng)
    d.add_mode(1.0, DecayMode(2, [113, 111], [rho_bw, None]))
    result[30223] = d

    return result


def build_kaon_decays(rng: Optional[np.random.Generator] = None) -> dict:
    """Build decay tables for kaon-carrying resonances.

    Returns a dict mapping parent PDG ID → :class:`ParticleDecays`.

    Resonances included:

    * **K*(892)** multiplet (PDG 323, -323, 313, -313)
    * **phi(1020)** (PDG 333)
    * **K1(1270)** multiplet (PDG 10323, -10323, 10313, -10313)
    * **K2*(1430)** multiplet (PDG 325, -325, 315, -315)

    Channel content is intentionally restricted to dominant hadronic 2-body
    modes that can be cascaded in this framework.
    """
    if rng is None:
        rng = np.random.default_rng()

    result: dict[int, ParticleDecays] = {}

    # Shared BW objects draw from the builder RNG so cascades stay reproducible.
    rho_bw = BreitWigner(0.2791, 0.77526, 0.1491, 1.0, 1.2, 1.5, rng=rng)
    delta_bw = BreitWigner(1.0765, 1.232, 0.117, 1.0, 1.6, 2.0, rng=rng)
    kstar_bw = BreitWigner(0.6337, 0.89166, 0.0508, 1.0, 1.15, 1.4, rng=rng)

    def _bw_for(pid):
        if pid in (113, 213, -213):
            return rho_bw
        if pid in (2224, 2214, 2114, 1114):
            return delta_bw
        if pid in (323, -323, 313, -313):
            return kstar_bw
        return None

    def _add(parent_id: int, modes: List[tuple[float, List[int]]]) -> None:
        d = ParticleDecays(parent_id, rng=rng)
        cum = 0.0
        for prob, daughters in modes:
            prob = float(max(prob, 0.0))
            if prob <= 0.0:
                continue
            cum += prob
            bws = [_bw_for(d) for d in daughters]
            d.add_mode(min(cum, 1.0), DecayMode(len(daughters), daughters, bws))
        if d._probabilities:
            d._probabilities[-1] = 1.0
        result[parent_id] = d

    # ------------------------------------------------------------------
    # K*(892) + phi(1020)
    # ------------------------------------------------------------------
    _add(323, [(0.667, [311, 211]), (0.333, [321, 111])])
    _add(-323, [(0.667, [-311, -211]), (0.333, [-321, 111])])
    _add(313, [(0.667, [321, -211]), (0.333, [311, 111])])
    _add(-313, [(0.667, [-321, 211]), (0.333, [-311, 111])])
    _add(333, [(0.492, [321, -321]), (0.508, [310, 130])])

    # ------------------------------------------------------------------
    # K1(1270): dominant channels represented as rho K, K* pi, omega K.
    # PDG central values are normalized over represented channels only.
    # ------------------------------------------------------------------
    k1_rhok = 0.42
    k1_kstpi = 0.16
    k1_omegak = 0.11
    k1_norm = k1_rhok + k1_kstpi + k1_omegak
    if k1_norm <= 0.0:
        k1_norm = 1.0
    k1_rhok /= k1_norm
    k1_kstpi /= k1_norm
    k1_omegak /= k1_norm

    _add(
        10323,  # K1(1270)+
        [
            ((2.0 / 3.0) * k1_rhok, [213, 311]),
            ((1.0 / 3.0) * k1_rhok, [113, 321]),
            ((2.0 / 3.0) * k1_kstpi, [313, 211]),
            ((1.0 / 3.0) * k1_kstpi, [323, 111]),
            (k1_omegak, [223, 321]),
        ],
    )
    _add(
        -10323,  # K1(1270)-
        [
            ((2.0 / 3.0) * k1_rhok, [-213, -311]),
            ((1.0 / 3.0) * k1_rhok, [113, -321]),
            ((2.0 / 3.0) * k1_kstpi, [-313, -211]),
            ((1.0 / 3.0) * k1_kstpi, [-323, 111]),
            (k1_omegak, [223, -321]),
        ],
    )
    _add(
        10313,  # K1(1270)0
        [
            ((2.0 / 3.0) * k1_rhok, [-213, 321]),
            ((1.0 / 3.0) * k1_rhok, [113, 311]),
            ((2.0 / 3.0) * k1_kstpi, [323, -211]),
            ((1.0 / 3.0) * k1_kstpi, [313, 111]),
            (k1_omegak, [223, 311]),
        ],
    )
    _add(
        -10313,  # anti-K1(1270)0
        [
            ((2.0 / 3.0) * k1_rhok, [213, -321]),
            ((1.0 / 3.0) * k1_rhok, [113, -311]),
            ((2.0 / 3.0) * k1_kstpi, [-323, 211]),
            ((1.0 / 3.0) * k1_kstpi, [-313, 111]),
            (k1_omegak, [223, -311]),
        ],
    )

    # ------------------------------------------------------------------
    # K2*(1430): dominant channels K pi, K* pi, rho K, omega K.
    # PDG central values are normalized over represented channels only.
    # ------------------------------------------------------------------
    k2_kpi = 0.499
    k2_kstpi = 0.248
    k2_rhok = 0.087
    k2_omegak = 0.028
    k2_norm = k2_kpi + k2_kstpi + k2_rhok + k2_omegak
    if k2_norm <= 0.0:
        k2_norm = 1.0
    k2_kpi /= k2_norm
    k2_kstpi /= k2_norm
    k2_rhok /= k2_norm
    k2_omegak /= k2_norm

    _add(
        325,  # K2*(1430)+
        [
            ((2.0 / 3.0) * k2_kpi, [311, 211]),
            ((1.0 / 3.0) * k2_kpi, [321, 111]),
            ((2.0 / 3.0) * k2_kstpi, [313, 211]),
            ((1.0 / 3.0) * k2_kstpi, [323, 111]),
            ((2.0 / 3.0) * k2_rhok, [213, 311]),
            ((1.0 / 3.0) * k2_rhok, [113, 321]),
            (k2_omegak, [223, 321]),
        ],
    )
    _add(
        -325,  # K2*(1430)-
        [
            ((2.0 / 3.0) * k2_kpi, [-311, -211]),
            ((1.0 / 3.0) * k2_kpi, [-321, 111]),
            ((2.0 / 3.0) * k2_kstpi, [-313, -211]),
            ((1.0 / 3.0) * k2_kstpi, [-323, 111]),
            ((2.0 / 3.0) * k2_rhok, [-213, -311]),
            ((1.0 / 3.0) * k2_rhok, [113, -321]),
            (k2_omegak, [223, -321]),
        ],
    )
    _add(
        315,  # K2*(1430)0
        [
            ((2.0 / 3.0) * k2_kpi, [321, -211]),
            ((1.0 / 3.0) * k2_kpi, [311, 111]),
            ((2.0 / 3.0) * k2_kstpi, [323, -211]),
            ((1.0 / 3.0) * k2_kstpi, [313, 111]),
            ((2.0 / 3.0) * k2_rhok, [-213, 321]),
            ((1.0 / 3.0) * k2_rhok, [113, 311]),
            (k2_omegak, [223, 311]),
        ],
    )
    _add(
        -315,  # anti-K2*(1430)0
        [
            ((2.0 / 3.0) * k2_kpi, [-321, 211]),
            ((1.0 / 3.0) * k2_kpi, [-311, 111]),
            ((2.0 / 3.0) * k2_kstpi, [-323, 211]),
            ((1.0 / 3.0) * k2_kstpi, [-313, 111]),
            ((2.0 / 3.0) * k2_rhok, [213, -321]),
            ((1.0 / 3.0) * k2_rhok, [113, -311]),
            (k2_omegak, [223, -311]),
        ],
    )

    return result


def build_soft_resonance_decays(
    rng: Optional[np.random.Generator] = None,
    branching_overrides: Optional[dict[str, float]] = None,
) -> dict:
    """Build an extended soft-hadron decay library with cascading channels.

    The branching structure is aligned with PDG summary/listing tables where
    quantitative branching fractions are available, and is normalized over
    the subset of channels represented in this 2-/3-body decay engine.

    Documented modelling approximations (see review notes):
    - ``pbrei`` is shape-neutral: distributions are normalized before
      sampling, so this factor never affects drawn masses.
    - Decay angles are isotropic in the parent rest frame (unpolarized
      parents; spin alignment and angular correlations neglected).
    - Branching fractions unreachable by the 2-/3-body engine are folded
      into the represented hadronic channels ("residual"), which changes
      inclusive daughter multiplicities by design rather than completing
      a PDG table literally.
    - Broad unstable daughters (rho, K*, Delta) are assigned Breit-Wigner
      spectral masses; narrow states (omega, phi, eta) keep pole masses.
    """
    if rng is None:
        rng = np.random.default_rng()
    if branching_overrides is None:
        branching_overrides = {}

    result: dict[int, ParticleDecays] = {}

    # Shared BW objects draw from the builder RNG so cascades stay reproducible.
    rho_bw = BreitWigner(0.2791, 0.77526, 0.1491, 1.0, 1.2, 1.5, rng=rng)
    delta_bw = BreitWigner(1.0765, 1.232, 0.117, 1.0, 1.6, 2.0, rng=rng)
    kstar_bw = BreitWigner(0.6337, 0.89166, 0.0508, 1.0, 1.15, 1.4, rng=rng)

    def _bw_for(pid):
        if pid in (113, 213, -213):
            return rho_bw
        if pid in (2224, 2214, 2114, 1114):
            return delta_bw
        if pid in (323, -323, 313, -313):
            return kstar_bw
        return None

    def _ovr(key: str, default: float) -> float:
        val = branching_overrides.get(key, default)
        return float(val)

    def _adder(parent_id: int):
        d = ParticleDecays(parent_id, rng=rng)
        cum = 0.0

        def add(prob: float, daughters: List[int]) -> None:
            nonlocal cum
            prob = float(max(prob, 0.0))
            if prob <= 0.0:
                return
            cum += prob
            bws = [_bw_for(d) for d in daughters]
            d.add_mode(min(cum, 1.0), DecayMode(len(daughters), daughters, bws))

        def finish() -> None:
            # Ensure the cumulative table always reaches 1.0.
            if d._probabilities:
                d._probabilities[-1] = 1.0

        return d, add, finish

    # ------------------------------------------------------------------
    # Delta resonances
    # ------------------------------------------------------------------
    d, add, fin = _adder(2224)       # Delta++
    add(1.0, [2212, 211])
    fin()
    result[2224] = d

    d, add, fin = _adder(2214)       # Delta+
    add(0.6667, [2212, 111])
    add(0.3333, [2112, 211])
    fin()
    result[2214] = d

    d, add, fin = _adder(2114)       # Delta0
    add(0.3333, [2212, -211])
    add(0.6667, [2112, 111])
    fin()
    result[2114] = d

    d, add, fin = _adder(1114)       # Delta-
    add(1.0, [2112, -211])
    fin()
    result[1114] = d

    # Physical anti-Delta- (charge +1): pure nbar pi+ charge conjugate of
    # Delta- -> n pi-. The historical matter-Delta+ alias on this id is
    # retired; generator cards use the true matter id 2214 directly, and
    # -1114 carries antibaryon quantum numbers only.
    d, add, fin = _adder(-1114)
    add(1.0, [-2112, 211])
    fin()
    result[-1114] = d

    # ------------------------------------------------------------------
    # N* resonances: include Nπ and Deltaπ branches.
    # PDG ranges are mapped to representative central values.
    # ------------------------------------------------------------------
    def _add_nstar(parent_id: int, is_positive: bool, br_npi: float, br_deltapi: float) -> None:
        d, add, fin = _adder(parent_id)

        # N* -> N pi (isospin CG charge split)
        if is_positive:
            add(br_npi * (1.0 / 3.0), [2212, 111])   # p pi0
            add(br_npi * (2.0 / 3.0), [2112, 211])   # n pi+
            # N*+ -> Delta pi
            add(br_deltapi * 0.50, [2224, -211])     # Delta++ pi-
            add(br_deltapi * 0.33, [2214, 111])      # Delta+  pi0
            add(br_deltapi * 0.17, [2114, 211])      # Delta0  pi+
        else:
            add(br_npi * (2.0 / 3.0), [2212, -211])  # p pi-
            add(br_npi * (1.0 / 3.0), [2112, 111])   # n pi0
            # N*0 -> Delta pi: exact I=1/2 CG for M=-1/2 (mirror of the
            # charged state): 1/6 Delta+pi-, 2/6 Delta0pi0, 3/6 Delta-pi+.
            add(br_deltapi * (1.0 / 6.0), [2214, -211])   # Delta+  pi-
            add(br_deltapi * (2.0 / 6.0), [2114, 111])    # Delta0  pi0
            add(br_deltapi * (3.0 / 6.0), [1114, 211])    # Delta-  pi+

        # Residual branching not explicitly represented: keep hadronic decay closed
        # by assigning it to N pi channels.
        residual = max(1.0 - br_npi - br_deltapi, 0.0)
        if is_positive:
            add(residual * (1.0 / 3.0), [2212, 111])
            add(residual * (2.0 / 3.0), [2112, 211])
        else:
            add(residual * (2.0 / 3.0), [2212, -211])
            add(residual * (1.0 / 3.0), [2112, 111])
        fin()
        result[parent_id] = d

    def _add_nstar_extended(
        parent_id: int,
        is_positive: bool,
        br_npi: float,
        br_deltapi: float = 0.0,
        br_neta: float = 0.0,
        br_nrho: float = 0.0,
        br_lambdak: float = 0.0,
        br_nstarpi: float = 0.0,
        br_nstarmeson: float = 0.0,
        lower_nstar_neutral_id: int | None = None,
        lower_nstar_positive_id: int | None = None,
        lower_nstar_meson_neutral_id: int | None = None,
        lower_nstar_meson_positive_id: int | None = None,
        lower_meson_tower_id: int | None = None,
    ) -> None:
        """N* states with extra channels (N eta, N rho, Lambda K)."""
        br_npi = max(float(br_npi), 0.0)
        br_deltapi = max(float(br_deltapi), 0.0)
        br_neta = max(float(br_neta), 0.0)
        br_nrho = max(float(br_nrho), 0.0)
        br_lambdak = max(float(br_lambdak), 0.0)
        br_nstarpi = max(float(br_nstarpi), 0.0)
        br_nstarmeson = max(float(br_nstarmeson), 0.0)

        # Keep branch sums physical even when effective-state inputs are tuned.
        total = br_npi + br_deltapi + br_neta + br_nrho + br_lambdak + br_nstarpi + br_nstarmeson
        if total > 1.0 and total > 0.0:
            scale = 1.0 / total
            br_npi *= scale
            br_deltapi *= scale
            br_neta *= scale
            br_nrho *= scale
            br_lambdak *= scale
            br_nstarpi *= scale
            br_nstarmeson *= scale

        d, add, fin = _adder(parent_id)

        # N* -> N pi
        if is_positive:
            add(br_npi * (1.0 / 3.0), [2212, 111])   # p pi0
            add(br_npi * (2.0 / 3.0), [2112, 211])   # n pi+
        else:
            add(br_npi * (2.0 / 3.0), [2212, -211])  # p pi-
            add(br_npi * (1.0 / 3.0), [2112, 111])   # n pi0

        # N* -> Delta pi
        if br_deltapi > 0.0:
            if is_positive:
                add(br_deltapi * 0.50, [2224, -211])     # Delta++ pi-
                add(br_deltapi * 0.33, [2214, 111])      # Delta+  pi0
                add(br_deltapi * 0.17, [2114, 211])      # Delta0  pi+
            else:
                # Neutral parent: exact mirrored CG (see _add_nstar note).
                add(br_deltapi * (1.0 / 6.0), [2214, -211])   # Delta+  pi-
                add(br_deltapi * (2.0 / 6.0), [2114, 111])    # Delta0  pi0
                add(br_deltapi * (3.0 / 6.0), [1114, 211])    # Delta-  pi+

        # N* -> N eta
        if br_neta > 0.0:
            add(br_neta, [2212, 221] if is_positive else [2112, 221])

        # N* -> N rho
        if br_nrho > 0.0:
            if is_positive:
                add(br_nrho * (1.0 / 3.0), [2212, 113])  # p rho0
                add(br_nrho * (2.0 / 3.0), [2112, 213])  # n rho+
            else:
                add(br_nrho * (2.0 / 3.0), [2212, -213]) # p rho-
                add(br_nrho * (1.0 / 3.0), [2112, 113])  # n rho0

        # N* -> Lambda K
        if br_lambdak > 0.0:
            add(br_lambdak, [3122, 321] if is_positive else [3122, 311])

        # Effective high-mass feed-down: R -> R' + pi to mimic unresolved tower cascading.
        if br_nstarpi > 0.0 and lower_nstar_neutral_id is not None:
            if is_positive:
                if lower_nstar_positive_id is not None:
                    add(br_nstarpi * (1.0 / 3.0), [lower_nstar_positive_id, 111])
                add(br_nstarpi * (2.0 / 3.0), [lower_nstar_neutral_id, 211])
            else:
                if lower_nstar_positive_id is not None:
                    add(br_nstarpi * (2.0 / 3.0), [lower_nstar_positive_id, -211])
                add(br_nstarpi * (1.0 / 3.0), [lower_nstar_neutral_id, 111])

        # Effective N* -> N* + M* links for unresolved mesonic tower feed-down.
        if br_nstarmeson > 0.0 and lower_meson_tower_id is not None:
            mes_neutral = lower_nstar_meson_neutral_id
            mes_positive = lower_nstar_meson_positive_id
            if mes_neutral is None:
                mes_neutral = lower_nstar_neutral_id
            if mes_positive is None:
                mes_positive = lower_nstar_positive_id
            if is_positive:
                if mes_positive is not None:
                    add(br_nstarmeson, [mes_positive, lower_meson_tower_id])
            else:
                if mes_neutral is not None:
                    add(br_nstarmeson, [mes_neutral, lower_meson_tower_id])

        residual = max(
            1.0 - br_npi - br_deltapi - br_neta - br_nrho - br_lambdak - br_nstarpi - br_nstarmeson,
            0.0,
        )
        if residual > 0.0:
            # Keep closure inside hadronic channels by assigning residual to N pi.
            if is_positive:
                add(residual * (1.0 / 3.0), [2212, 111])
                add(residual * (2.0 / 3.0), [2112, 211])
            else:
                add(residual * (2.0 / 3.0), [2212, -211])
                add(residual * (1.0 / 3.0), [2112, 111])
        fin()
        result[parent_id] = d

    def _add_delta_excited(
        parent_id: int,
        charge_state: str,
        br_npi: float,
        br_deltapi: float,
        br_nrho: float,
    ) -> None:
        """Delta* states with N pi, Delta pi, and N rho channels."""
        br_npi = max(float(br_npi), 0.0)
        br_deltapi = max(float(br_deltapi), 0.0)
        br_nrho = max(float(br_nrho), 0.0)
        # Keep branch sums physical even when effective-state inputs are tuned.
        total = br_npi + br_deltapi + br_nrho
        if total > 1.0 and total > 0.0:
            scale = 1.0 / total
            br_npi *= scale
            br_deltapi *= scale
            br_nrho *= scale

        d, add, fin = _adder(parent_id)

        # Delta* -> N pi
        if charge_state == "pp":
            add(br_npi, [2212, 211])                      # p pi+
        elif charge_state == "p":
            add(br_npi * (2.0 / 3.0), [2212, 111])        # p pi0
            add(br_npi * (1.0 / 3.0), [2112, 211])        # n pi+
        elif charge_state == "0":
            add(br_npi * (1.0 / 3.0), [2212, -211])       # p pi-
            add(br_npi * (2.0 / 3.0), [2112, 111])        # n pi0
        elif charge_state == "m":
            add(br_npi, [2112, -211])                     # n pi-

        # Delta* -> Delta pi. Exact I=3/2 Clebsch-Gordan weights for the
        # Delta(3/2) x pi -> I=3/2 partial wave:
        #   |3/2,+/-3/2> = one charge state;
        #   |3/2,+1/2>: 6/15 Delta++pi-, 1/15 Delta+pi0, 8/15 Delta0pi+;
        #   |3/2,-1/2>: 8/15 Delta+pi-, 1/15 Delta0pi0, 6/15 Delta-pi+.
        if charge_state == "pp":
            add(br_deltapi * 0.60, [2224, 111])           # Delta++ pi0
            add(br_deltapi * 0.40, [2214, 211])           # Delta+  pi+
        elif charge_state == "p":
            add(br_deltapi * (6.0 / 15.0), [2224, -211])  # Delta++ pi-
            add(br_deltapi * (1.0 / 15.0), [2214, 111])   # Delta+  pi0
            add(br_deltapi * (8.0 / 15.0), [2114, 211])   # Delta0  pi+
        elif charge_state == "0":
            add(br_deltapi * (8.0 / 15.0), [2214, -211])  # Delta+  pi-
            add(br_deltapi * (1.0 / 15.0), [2114, 111])   # Delta0  pi0
            add(br_deltapi * (6.0 / 15.0), [1114, 211])   # Delta-  pi+
        elif charge_state == "m":
            add(br_deltapi * 0.60, [1114, 111])           # Delta-  pi0
            add(br_deltapi * 0.40, [2114, -211])          # Delta0  pi-

        # Delta* -> N rho
        if charge_state == "pp":
            add(br_nrho, [2212, 213])                     # p rho+
        elif charge_state == "p":
            add(br_nrho * (2.0 / 3.0), [2212, 113])       # p rho0
            add(br_nrho * (1.0 / 3.0), [2112, 213])       # n rho+
        elif charge_state == "0":
            add(br_nrho * (1.0 / 3.0), [2212, -213])      # p rho-
            add(br_nrho * (2.0 / 3.0), [2112, 113])       # n rho0
        elif charge_state == "m":
            add(br_nrho, [2112, -213])                    # n rho-

        residual = max(1.0 - br_npi - br_deltapi - br_nrho, 0.0)
        if residual > 0.0:
            # Assign residual to N pi to keep full probability closure.
            if charge_state == "pp":
                add(residual, [2212, 211])
            elif charge_state == "p":
                add(residual * (2.0 / 3.0), [2212, 111])
                add(residual * (1.0 / 3.0), [2112, 211])
            elif charge_state == "0":
                add(residual * (1.0 / 3.0), [2212, -211])
                add(residual * (2.0 / 3.0), [2112, 111])
            elif charge_state == "m":
                add(residual, [2112, -211])
        fin()
        result[parent_id] = d

    # N(1440): Npi 55-75%, Delta pi 6-27%.
    _add_nstar(
        12112,
        is_positive=False,
        br_npi=_ovr("n1440_br_npi", 0.65),
        br_deltapi=_ovr("n1440_br_deltapi", 0.165),
    )
    _add_nstar(
        12212,
        is_positive=True,
        br_npi=_ovr("n1440_br_npi", 0.65),
        br_deltapi=_ovr("n1440_br_deltapi", 0.165),
    )

    # N(1520): Npi 55-65%, Delta pi 22-34%.
    _add_nstar(
        2116,
        is_positive=False,
        br_npi=_ovr("n1520_br_npi", 0.60),
        br_deltapi=_ovr("n1520_br_deltapi", 0.28),
    )
    _add_nstar(
        2216,
        is_positive=True,
        br_npi=_ovr("n1520_br_npi", 0.60),
        br_deltapi=_ovr("n1520_br_deltapi", 0.28),
    )

    # N(1680): Npi 60-70%, Delta pi 11-23%.
    _add_nstar(
        22112,
        is_positive=False,
        br_npi=_ovr("n1680_br_npi", 0.65),
        br_deltapi=_ovr("n1680_br_deltapi", 0.17),
    )
    _add_nstar(
        22212,
        is_positive=True,
        br_npi=_ovr("n1680_br_npi", 0.65),
        br_deltapi=_ovr("n1680_br_deltapi", 0.17),
    )

    # N(1535): Npi + Neta dominate.
    _add_nstar_extended(
        102112,
        is_positive=False,
        br_npi=_ovr("n1535_br_npi", 0.42),
        br_neta=_ovr("n1535_br_neta", 0.45),
    )
    _add_nstar_extended(
        102212,
        is_positive=True,
        br_npi=_ovr("n1535_br_npi", 0.42),
        br_neta=_ovr("n1535_br_neta", 0.45),
    )

    # N(1650): include Npi, Delta pi, Neta, Lambda K.
    _add_nstar_extended(
        122112,
        is_positive=False,
        br_npi=_ovr("n1650_br_npi", 0.65),
        br_deltapi=_ovr("n1650_br_deltapi", 0.15),
        br_neta=_ovr("n1650_br_neta", 0.08),
        br_lambdak=_ovr("n1650_br_lambdak", 0.12),
    )
    _add_nstar_extended(
        122212,
        is_positive=True,
        br_npi=_ovr("n1650_br_npi", 0.65),
        br_deltapi=_ovr("n1650_br_deltapi", 0.15),
        br_neta=_ovr("n1650_br_neta", 0.08),
        br_lambdak=_ovr("n1650_br_lambdak", 0.12),
    )

    # N(1720): include Npi, Delta pi, Nrho, Neta, Lambda K.
    _add_nstar_extended(
        212114,
        is_positive=False,
        br_npi=_ovr("n1720_br_npi", 0.15),
        br_deltapi=_ovr("n1720_br_deltapi", 0.35),
        br_nrho=_ovr("n1720_br_nrho", 0.35),
        br_neta=_ovr("n1720_br_neta", 0.05),
        br_lambdak=_ovr("n1720_br_lambdak", 0.10),
    )
    _add_nstar_extended(
        212214,
        is_positive=True,
        br_npi=_ovr("n1720_br_npi", 0.15),
        br_deltapi=_ovr("n1720_br_deltapi", 0.35),
        br_nrho=_ovr("n1720_br_nrho", 0.35),
        br_neta=_ovr("n1720_br_neta", 0.05),
        br_lambdak=_ovr("n1720_br_lambdak", 0.10),
    )

    # ------------------------------------------------------------------
    # Effective high-mass N* tower (Regge-inspired).
    #
    # Rationale:
    # - higher-spin/higher-mass baryon states are poorly resolved experimentally,
    # - but can contribute sizeable feed-down through cascading.
    # This adds a minimal effective tower with explicit R -> R' + pi links.
    # ------------------------------------------------------------------
    nstar_eff_enable = _ovr("nstar_eff_enable", 1.0)
    if nstar_eff_enable > 0.0:
        n1_0 = 9911201
        n1_p = 9912201
        n2_0 = 9921201
        n2_p = 9922201
        n3_0 = 9931201
        n3_p = 9932201
        m1_0 = 9900113
        m2_0 = 9900213
        m3_0 = 9900313

        # Effective neutral meson tower used in N* -> N* + M* feed-down.
        mestar_eff_enable = _ovr("mestar_eff_enable", 1.0)
        if mestar_eff_enable > 0.0:
            # D5 pair-family configuration: one aggregate strength shared
            # by the pair-capable levels l2/l3, plus the recorded
            # three-body pi0 option behind the same strength key.
            mestar_ppbar_strength = _ovr(
                MESTAR_PPBAR_STRENGTH_KEY, MESTAR_PPBAR_STRENGTH_DEFAULT
            )
            mestar_ppbar_pi0_variant = _ovr(
                MESTAR_PPBAR_PI0_VARIANT_KEY, MESTAR_PPBAR_PI0_VARIANT_DEFAULT
            )
            mestar_ppbar_pi0_share = _ovr(
                MESTAR_PPBAR_PI0_SHARE_KEY, MESTAR_PPBAR_PI0_SHARE_DEFAULT
            )
            mestar_ppbar_law_flag = _ovr(
                MESTAR_PPBAR_LAW_FLAG_KEY, MESTAR_PPBAR_LAW_FLAG_DEFAULT
            )
            mestar_ppbar_rollon_m0 = _ovr(
                MESTAR_PPBAR_ROLLON_M0_KEY, MESTAR_PPBAR_ROLLON_M0_DEFAULT
            )
            mestar_ppbar_rolloff_w = _ovr(
                MESTAR_PPBAR_ROLLOFF_W_KEY, MESTAR_PPBAR_ROLLOFF_W_DEFAULT
            )

            def _add_mestar(
                parent_id: int,
                br_pipi: float,
                br_mestarpi: float = 0.0,
                lower_mestar_id: int | None = None,
                br_rhopi: float = 0.0,
                pair_capable: bool = False,
            ) -> None:
                br_pipi = max(float(br_pipi), 0.0)
                br_mestarpi = max(float(br_mestarpi), 0.0)
                br_rhopi = max(float(br_rhopi), 0.0)
                total = br_pipi + br_mestarpi + br_rhopi
                if total > 1.0 and total > 0.0:
                    scale = 1.0 / total
                    br_pipi *= scale
                    br_mestarpi *= scale
                    br_rhopi *= scale

                d, add, fin = _adder(parent_id)
                if br_pipi > 0.0:
                    add((2.0 / 3.0) * br_pipi, [211, -211])
                    add((1.0 / 3.0) * br_pipi, [111, 111])
                if br_mestarpi > 0.0 and lower_mestar_id is not None:
                    add(br_mestarpi, [lower_mestar_id, 111])
                if br_rhopi > 0.0:
                    add(br_rhopi, [113, 111])
                residual = max(1.0 - br_pipi - br_mestarpi - br_rhopi, 0.0)
                if residual > 0.0:
                    add((2.0 / 3.0) * residual, [211, -211])
                    add((1.0 / 3.0) * residual, [111, 111])
                # D5 pair family on the pair-capable levels only (l2/l3;
                # l1 stays cascade-only). The pair channels enter the
                # declared table with a zero incremental share: their
                # budget is entirely mass-conditioned through the
                # PairFamilyLaw below, evaluated at the realized parent
                # mass, while the surviving pionic weights above keep
                # their declared normalization (including the
                # residual-to-pipi slack) and scale by (1 - P_ppbar(M)).
                # Daughters are fixed PDG-table entries (bws=None), so no
                # new daughter-mass bookkeeping is introduced.
                if pair_capable and mestar_ppbar_strength > 0.0:
                    members: List[Tuple[int, float]] = []
                    cum_now = (
                        float(d._probabilities[-1]) if d._probabilities else 0.0
                    )
                    d.add_mode(
                        cum_now,
                        DecayMode(
                            2, list(PPBAR_PAIR_TWO_BODY_DAUGHTERS)
                        ),
                    )
                    members.append((d.n_modes() - 1, 1.0))
                    if mestar_ppbar_pi0_variant > 0.0:
                        d.add_mode(
                            cum_now,
                            DecayMode(
                                3, list(PPBAR_PAIR_THREE_BODY_DAUGHTERS)
                            ),
                        )
                        pi0_share = mestar_ppbar_pi0_share
                        members[0] = (members[0][0], 1.0 - pi0_share)
                        members.append((d.n_modes() - 1, pi0_share))
                    d.pair_family = PairFamilyLaw(
                        strength=mestar_ppbar_strength,
                        threshold=ppbar_pair_threshold(),
                        exponent=_ovr(
                            MESTAR_PPBAR_EXPONENT_KEY,
                            MESTAR_PPBAR_EXPONENT_DEFAULT,
                        ),
                        members=tuple(members),
                        law=mestar_ppbar_law_flag,
                        rollon_m0=mestar_ppbar_rollon_m0,
                        rolloff_w=mestar_ppbar_rolloff_w,
                    )
                fin()
                result[parent_id] = d

            _add_mestar(
                m1_0,
                br_pipi=_ovr("mestar_eff_l1_br_pipi", 1.0),
            )
            _add_mestar(
                m2_0,
                br_pipi=_ovr("mestar_eff_l2_br_pipi", 0.35),
                br_mestarpi=_ovr("mestar_eff_l2_br_mestarpi", 0.65),
                lower_mestar_id=m1_0,
                pair_capable=True,
            )
            _add_mestar(
                m3_0,
                br_pipi=_ovr("mestar_eff_l3_br_pipi", 0.15),
                br_mestarpi=_ovr("mestar_eff_l3_br_mestarpi", 0.60),
                lower_mestar_id=m2_0,
                br_rhopi=_ovr("mestar_eff_l3_br_rhopi", 0.25),
                pair_capable=True,
            )

        # Level 1 ~1.9 GeV, feeding known N(1720).
        _add_nstar_extended(
            n1_0,
            is_positive=False,
            br_npi=_ovr("nstar_eff_l1_br_npi", 0.12),
            br_deltapi=_ovr("nstar_eff_l1_br_deltapi", 0.35),
            br_nrho=_ovr("nstar_eff_l1_br_nrho", 0.28),
            br_neta=_ovr("nstar_eff_l1_br_neta", 0.05),
            br_lambdak=_ovr("nstar_eff_l1_br_lambdak", 0.05),
            br_nstarpi=_ovr("nstar_eff_l1_br_nstarpi", 0.15),
            br_nstarmeson=_ovr("nstar_eff_l1_br_nstarmeson", 0.0),
            lower_nstar_neutral_id=212114,   # N(1720)0
            lower_nstar_positive_id=212214,  # N(1720)+
            lower_nstar_meson_neutral_id=2116,   # N*(1520)0
            lower_nstar_meson_positive_id=2216,  # N*(1520)+
            lower_meson_tower_id=m1_0,
        )
        _add_nstar_extended(
            n1_p,
            is_positive=True,
            br_npi=_ovr("nstar_eff_l1_br_npi", 0.12),
            br_deltapi=_ovr("nstar_eff_l1_br_deltapi", 0.35),
            br_nrho=_ovr("nstar_eff_l1_br_nrho", 0.28),
            br_neta=_ovr("nstar_eff_l1_br_neta", 0.05),
            br_lambdak=_ovr("nstar_eff_l1_br_lambdak", 0.05),
            br_nstarpi=_ovr("nstar_eff_l1_br_nstarpi", 0.15),
            br_nstarmeson=_ovr("nstar_eff_l1_br_nstarmeson", 0.0),
            lower_nstar_neutral_id=212114,
            lower_nstar_positive_id=212214,
            lower_nstar_meson_neutral_id=2116,
            lower_nstar_meson_positive_id=2216,
            lower_meson_tower_id=m1_0,
        )

        # Level 2 ~2.1 GeV, feeding level-1.
        _add_nstar_extended(
            n2_0,
            is_positive=False,
            br_npi=_ovr("nstar_eff_l2_br_npi", 0.08),
            br_deltapi=_ovr("nstar_eff_l2_br_deltapi", 0.30),
            br_nrho=_ovr("nstar_eff_l2_br_nrho", 0.22),
            br_neta=_ovr("nstar_eff_l2_br_neta", 0.05),
            br_lambdak=_ovr("nstar_eff_l2_br_lambdak", 0.05),
            br_nstarpi=_ovr("nstar_eff_l2_br_nstarpi", 0.30),
            br_nstarmeson=_ovr("nstar_eff_l2_br_nstarmeson", 0.0),
            lower_nstar_neutral_id=n1_0,
            lower_nstar_positive_id=n1_p,
            lower_nstar_meson_neutral_id=n1_0,
            lower_nstar_meson_positive_id=n1_p,
            lower_meson_tower_id=m2_0,
        )
        _add_nstar_extended(
            n2_p,
            is_positive=True,
            br_npi=_ovr("nstar_eff_l2_br_npi", 0.08),
            br_deltapi=_ovr("nstar_eff_l2_br_deltapi", 0.30),
            br_nrho=_ovr("nstar_eff_l2_br_nrho", 0.22),
            br_neta=_ovr("nstar_eff_l2_br_neta", 0.05),
            br_lambdak=_ovr("nstar_eff_l2_br_lambdak", 0.05),
            br_nstarpi=_ovr("nstar_eff_l2_br_nstarpi", 0.30),
            br_nstarmeson=_ovr("nstar_eff_l2_br_nstarmeson", 0.0),
            lower_nstar_neutral_id=n1_0,
            lower_nstar_positive_id=n1_p,
            lower_nstar_meson_neutral_id=n1_0,
            lower_nstar_meson_positive_id=n1_p,
            lower_meson_tower_id=m2_0,
        )

        # Level 3 ~2.3 GeV, feeding level-2.
        _add_nstar_extended(
            n3_0,
            is_positive=False,
            br_npi=_ovr("nstar_eff_l3_br_npi", 0.06),
            br_deltapi=_ovr("nstar_eff_l3_br_deltapi", 0.22),
            br_nrho=_ovr("nstar_eff_l3_br_nrho", 0.17),
            br_neta=_ovr("nstar_eff_l3_br_neta", 0.05),
            br_lambdak=_ovr("nstar_eff_l3_br_lambdak", 0.05),
            br_nstarpi=_ovr("nstar_eff_l3_br_nstarpi", 0.45),
            br_nstarmeson=_ovr("nstar_eff_l3_br_nstarmeson", 0.0),
            lower_nstar_neutral_id=n2_0,
            lower_nstar_positive_id=n2_p,
            lower_nstar_meson_neutral_id=n2_0,
            lower_nstar_meson_positive_id=n2_p,
            lower_meson_tower_id=m3_0,
        )
        _add_nstar_extended(
            n3_p,
            is_positive=True,
            br_npi=_ovr("nstar_eff_l3_br_npi", 0.06),
            br_deltapi=_ovr("nstar_eff_l3_br_deltapi", 0.22),
            br_nrho=_ovr("nstar_eff_l3_br_nrho", 0.17),
            br_neta=_ovr("nstar_eff_l3_br_neta", 0.05),
            br_lambdak=_ovr("nstar_eff_l3_br_lambdak", 0.05),
            br_nstarpi=_ovr("nstar_eff_l3_br_nstarpi", 0.45),
            br_nstarmeson=_ovr("nstar_eff_l3_br_nstarmeson", 0.0),
            lower_nstar_neutral_id=n2_0,
            lower_nstar_positive_id=n2_p,
            lower_nstar_meson_neutral_id=n2_0,
            lower_nstar_meson_positive_id=n2_p,
            lower_meson_tower_id=m3_0,
        )

    # Delta(1600)
    _add_delta_excited(
        202224,
        charge_state="pp",
        br_npi=_ovr("d1600_br_npi", 0.18),
        br_deltapi=_ovr("d1600_br_deltapi", 0.55),
        br_nrho=_ovr("d1600_br_nrho", 0.27),
    )
    _add_delta_excited(
        202214,
        charge_state="p",
        br_npi=_ovr("d1600_br_npi", 0.18),
        br_deltapi=_ovr("d1600_br_deltapi", 0.55),
        br_nrho=_ovr("d1600_br_nrho", 0.27),
    )
    _add_delta_excited(
        202114,
        charge_state="0",
        br_npi=_ovr("d1600_br_npi", 0.18),
        br_deltapi=_ovr("d1600_br_deltapi", 0.55),
        br_nrho=_ovr("d1600_br_nrho", 0.27),
    )
    _add_delta_excited(
        201114,
        charge_state="m",
        br_npi=_ovr("d1600_br_npi", 0.18),
        br_deltapi=_ovr("d1600_br_deltapi", 0.55),
        br_nrho=_ovr("d1600_br_nrho", 0.27),
    )

    # Delta(1700)
    _add_delta_excited(
        122224,
        charge_state="pp",
        br_npi=_ovr("d1700_br_npi", 0.15),
        br_deltapi=_ovr("d1700_br_deltapi", 0.50),
        br_nrho=_ovr("d1700_br_nrho", 0.35),
    )
    _add_delta_excited(
        122214,
        charge_state="p",
        br_npi=_ovr("d1700_br_npi", 0.15),
        br_deltapi=_ovr("d1700_br_deltapi", 0.50),
        br_nrho=_ovr("d1700_br_nrho", 0.35),
    )
    _add_delta_excited(
        122114,
        charge_state="0",
        br_npi=_ovr("d1700_br_npi", 0.15),
        br_deltapi=_ovr("d1700_br_deltapi", 0.50),
        br_nrho=_ovr("d1700_br_nrho", 0.35),
    )
    _add_delta_excited(
        121114,
        charge_state="m",
        br_npi=_ovr("d1700_br_npi", 0.15),
        br_deltapi=_ovr("d1700_br_deltapi", 0.50),
        br_nrho=_ovr("d1700_br_nrho", 0.35),
    )

    # ------------------------------------------------------------------
    # Light mesons
    # ------------------------------------------------------------------
    d, add, fin = _adder(113)        # rho0(770)
    add(1.0, [211, -211])
    fin()
    result[113] = d

    d, add, fin = _adder(-213)       # rho-(770)
    add(1.0, [-211, 111])
    fin()
    result[-213] = d

    d, add, fin = _adder(213)        # rho+(770)
    add(1.0, [211, 111])
    fin()
    result[213] = d

    # b1(1235): dominant omega pi channels.
    d, add, fin = _adder(10113)      # b1(1235)^0
    add(1.0, [223, 111])
    fin()
    result[10113] = d

    d, add, fin = _adder(10213)      # b1(1235)^+
    add(1.0, [223, 211])
    fin()
    result[10213] = d

    d, add, fin = _adder(-10213)     # b1(1235)^-
    add(1.0, [223, -211])
    fin()
    result[-10213] = d

    # a1(1260): represented via rho pi.
    d, add, fin = _adder(20113)      # a1(1260)^0
    add(0.5, [213, -211])            # rho+ pi-
    add(0.5, [-213, 211])            # rho- pi+
    fin()
    result[20113] = d

    d, add, fin = _adder(20213)      # a1(1260)^+
    add(0.5, [213, 111])             # rho+ pi0
    add(0.5, [113, 211])             # rho0 pi+
    fin()
    result[20213] = d

    d, add, fin = _adder(-20213)     # a1(1260)^-
    add(0.5, [-213, 111])            # rho- pi0
    add(0.5, [113, -211])            # rho0 pi-
    fin()
    result[-20213] = d

    # a2(1320): rho pi + eta pi channels, normalized over represented modes.
    a2_rhopi = _ovr("a2_br_rhopi", 0.70)
    a2_etapi = _ovr("a2_br_etapi", 0.145)
    a2_norm = a2_rhopi + a2_etapi
    if a2_norm <= 0.0:
        a2_norm = 1.0
    a2_rhopi /= a2_norm
    a2_etapi /= a2_norm

    d, add, fin = _adder(115)        # a2(1320)^0
    add(0.5 * a2_rhopi, [213, -211]) # rho+ pi-
    add(0.5 * a2_rhopi, [-213, 211]) # rho- pi+
    add(a2_etapi, [221, 111])        # eta pi0
    fin()
    result[115] = d

    d, add, fin = _adder(215)        # a2(1320)^+
    add(0.5 * a2_rhopi, [213, 111])  # rho+ pi0
    add(0.5 * a2_rhopi, [113, 211])  # rho0 pi+
    add(a2_etapi, [221, 211])        # eta pi+
    fin()
    result[215] = d

    d, add, fin = _adder(-215)       # a2(1320)^-
    add(0.5 * a2_rhopi, [-213, 111]) # rho- pi0
    add(0.5 * a2_rhopi, [113, -211]) # rho0 pi-
    add(a2_etapi, [221, -211])       # eta pi-
    fin()
    result[-215] = d

    # omega(782): normalized over explicit channels listed here.
    d, add, fin = _adder(223)
    omega_norm = 0.892 + 0.0833 + 0.0153
    add(0.892 / omega_norm, [211, -211, 111])  # pi+ pi- pi0
    add(0.0833 / omega_norm, [111, 22])        # pi0 gamma
    add(0.0153 / omega_norm, [211, -211])      # pi+ pi-
    fin()
    result[223] = d

    # eta(548): keep major channels including hadronic + pi+pi-gamma.
    d, add, fin = _adder(221)
    add(0.2302, [211, -211, 111])    # pi+ pi- pi0
    add(0.3256, [111, 111, 111])     # 3pi0
    add(0.0428, [211, -211, 22])     # pi+ pi- gamma
    add(0.3936, [22, 22])            # gamma gamma
    fin()
    result[221] = d

    # f2(1270): PDG fit fractions; normalized over channels represented here.
    d, add, fin = _adder(225)
    f2_pipi = 0.843
    f2_kk = 0.046
    f2_norm = f2_pipi + f2_kk
    add((2.0 / 3.0) * f2_pipi / f2_norm, [211, -211])  # pi+ pi-
    add((1.0 / 3.0) * f2_pipi / f2_norm, [111, 111])   # pi0 pi0
    add(0.5 * f2_kk / f2_norm, [321, -321])            # K+ K-
    add(0.5 * f2_kk / f2_norm, [310, 130])             # K0S K0L
    fin()
    result[225] = d

    # rho3(1690): PDG channels normalized over explicitly represented hadronic modes.
    rho3_pipi = _ovr("rho3_br_pipi", 0.236)
    rho3_omegapi = _ovr("rho3_br_omegapi", 0.16)
    rho3_kkpi = _ovr("rho3_br_kkpi", 0.038)
    rho3_kk = _ovr("rho3_br_kk", 0.0158)
    rho3_norm = rho3_pipi + rho3_omegapi + rho3_kkpi + rho3_kk
    if rho3_norm <= 0.0:
        rho3_norm = 1.0

    d, add, fin = _adder(117)        # rho3(1690)^0
    add(rho3_pipi / rho3_norm, [211, -211])
    add(rho3_omegapi / rho3_norm, [223, 111])
    add(rho3_kkpi / rho3_norm, [321, -321, 111])
    add(rho3_kk / rho3_norm, [321, -321])
    fin()
    result[117] = d

    d, add, fin = _adder(-217)       # rho3(1690)^-
    add(rho3_pipi / rho3_norm, [-211, 111])
    add(rho3_omegapi / rho3_norm, [223, -211])
    add(rho3_kkpi / rho3_norm, [311, -321, 111])
    add(rho3_kk / rho3_norm, [311, -321])
    fin()
    result[-217] = d

    d, add, fin = _adder(217)        # rho3(1690)^+
    add(rho3_pipi / rho3_norm, [211, 111])
    add(rho3_omegapi / rho3_norm, [223, 211])
    add(rho3_kkpi / rho3_norm, [-311, 321, 111])
    add(rho3_kk / rho3_norm, [-311, 321])
    fin()
    result[217] = d

    # f4(2050): PDG channels normalized over explicit represented channels.
    f4_pipi = _ovr("f4_br_pipi", 0.17)
    f4_kk = _ovr("f4_br_kk", 0.0068)
    f4_etaeta = _ovr("f4_br_etaeta", 0.0021)
    f4_norm = f4_pipi + f4_kk + f4_etaeta
    if f4_norm <= 0.0:
        f4_norm = 1.0

    d, add, fin = _adder(229)
    add((2.0 / 3.0) * f4_pipi / f4_norm, [211, -211])
    add((1.0 / 3.0) * f4_pipi / f4_norm, [111, 111])
    add(0.5 * f4_kk / f4_norm, [321, -321])
    add(0.5 * f4_kk / f4_norm, [310, 130])
    add(f4_etaeta / f4_norm, [221, 221])
    fin()
    result[229] = d

    # ------------------------------------------------------------------
    # Neutral-kaon feed-down convention.
    #
    # The (310, 130) daughter pairs used by phi, f2, f4 and rho3 channels
    # represent a neutral-kaon pair in which one member is observed as a
    # K0S.  Only K0S decays are represented here; its dominant modes are
    # the charged and neutral pion pairs.  K0L stays invisible because its
    # dominant decay modes are semileptonic and lie outside this hadronic
    # two-/three-body engine.  Direct 311/-311 daughters remain invisible
    # pending an explicit neutral-kaon mixing treatment.
    # ------------------------------------------------------------------
    k0s_br_pipi = _ovr("k0s_br_pipi", 0.692)
    k0s_br_pi0pi0 = _ovr("k0s_br_pi0pi0", 0.308)
    k0s_norm = k0s_br_pipi + k0s_br_pi0pi0
    if k0s_norm <= 0.0:
        k0s_norm = 1.0
    d, add, fin = _adder(310)        # K0S
    add(k0s_br_pipi / k0s_norm, [211, -211])   # pi+ pi-
    add(k0s_br_pi0pi0 / k0s_norm, [111, 111])  # pi0 pi0
    fin()
    result[310] = d

    # ------------------------------------------------------------------
    # Strange baryons and hyperon resonances
    # ------------------------------------------------------------------
    # Lambda weak decay channels (normalized over represented dominant modes).
    lambda_br_ppim = _ovr("lambda_br_ppim", 0.639)
    lambda_br_npi0 = _ovr("lambda_br_npi0", 0.358)
    lambda_norm = lambda_br_ppim + lambda_br_npi0
    if lambda_norm <= 0.0:
        lambda_norm = 1.0
    d, add, fin = _adder(3122)
    add(lambda_br_ppim / lambda_norm, [2212, -211])  # p pi-
    add(lambda_br_npi0 / lambda_norm, [2112, 111])   # n pi0
    fin()
    result[3122] = d

    d, add, fin = _adder(-3122)
    add(lambda_br_ppim / lambda_norm, [-2212, 211])  # pbar pi+
    add(lambda_br_npi0 / lambda_norm, [-2112, 111])  # nbar pi0
    fin()
    result[-3122] = d

    # Sigma ground-state weak/electromagnetic decays.
    sigma_p_br_ppi0 = _ovr("sigma_plus_br_ppi0", 0.516)
    sigma_p_br_npip = _ovr("sigma_plus_br_npip", 0.483)
    sigma_p_norm = sigma_p_br_ppi0 + sigma_p_br_npip
    if sigma_p_norm <= 0.0:
        sigma_p_norm = 1.0
    d, add, fin = _adder(3222)
    add(sigma_p_br_ppi0 / sigma_p_norm, [2212, 111])  # p pi0
    add(sigma_p_br_npip / sigma_p_norm, [2112, 211])  # n pi+
    fin()
    result[3222] = d

    d, add, fin = _adder(3212)
    add(1.0, [3122, 22])  # Lambda gamma
    fin()
    result[3212] = d

    d, add, fin = _adder(3112)
    add(1.0, [2112, -211])  # n pi-
    fin()
    result[3112] = d

    # Lambda(1520): NKbar + Sigma pi (+ small Lambda pi pi channel).
    l1520_nkbar = _ovr("lambda1520_br_nkbar", 0.45)
    l1520_sigmapi = _ovr("lambda1520_br_sigmapi", 0.42)
    l1520_lambdapipi = _ovr("lambda1520_br_lambdapipi", 0.10)
    l1520_norm = l1520_nkbar + l1520_sigmapi + l1520_lambdapipi
    if l1520_norm <= 0.0:
        l1520_norm = 1.0
    d, add, fin = _adder(102134)
    add(0.5 * l1520_nkbar / l1520_norm, [2212, -321])        # p K-
    add(0.5 * l1520_nkbar / l1520_norm, [2112, -311])        # n Kbar0
    add((1.0 / 3.0) * l1520_sigmapi / l1520_norm, [3222, -211])  # Sigma+ pi-
    add((1.0 / 3.0) * l1520_sigmapi / l1520_norm, [3212, 111])    # Sigma0 pi0
    add((1.0 / 3.0) * l1520_sigmapi / l1520_norm, [3112, 211])    # Sigma- pi+
    add(l1520_lambdapipi / l1520_norm, [3122, 211, -211])         # Lambda pi+ pi-
    fin()
    result[102134] = d

    # Sigma(1385): Lambda pi dominant with subleading Sigma pi.
    s1385_lpi = _ovr("sigma1385_br_lambdapi", 0.87)
    s1385_spi = _ovr("sigma1385_br_sigmapi", 0.117)
    s1385_norm = s1385_lpi + s1385_spi
    if s1385_norm <= 0.0:
        s1385_norm = 1.0

    d, add, fin = _adder(3224)  # Sigma(1385)+
    add(s1385_lpi / s1385_norm, [3122, 211])                   # Lambda pi+
    add(0.5 * s1385_spi / s1385_norm, [3222, 111])             # Sigma+ pi0
    add(0.5 * s1385_spi / s1385_norm, [3212, 211])             # Sigma0 pi+
    fin()
    result[3224] = d

    d, add, fin = _adder(3214)  # Sigma(1385)0
    add(s1385_lpi / s1385_norm, [3122, 111])                   # Lambda pi0
    # The Sigma-pi system couples to pure I=1, which forbids the
    # Sigma0 pi0 charge state; the two allowed states share the
    # branching equally.
    add(0.5 * s1385_spi / s1385_norm, [3222, -211])            # Sigma+ pi-
    add(0.5 * s1385_spi / s1385_norm, [3112, 211])             # Sigma- pi+
    fin()
    result[3214] = d

    d, add, fin = _adder(3114)  # Sigma(1385)-
    add(s1385_lpi / s1385_norm, [3122, -211])                  # Lambda pi-
    add(0.5 * s1385_spi / s1385_norm, [3112, 111])             # Sigma- pi0
    add(0.5 * s1385_spi / s1385_norm, [3212, -211])            # Sigma0 pi-
    fin()
    result[3114] = d

    # Include kaon-carrying resonances (K*, phi, K1, K2*) and the excited
    # omega states (omega(1420)/omega(1650) -> rho0 pi0) in the same table
    # so that ANY generated resonance id has a decay entry; a missing entry
    # would otherwise let a parent pass through the cascade undecayed.
    result.update(build_kaon_decays(rng=rng))
    result.update(build_omega_decays(rng=rng))

    return result
