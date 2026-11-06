# The legacy p+A path (marked 2026-11-04): the HEART p+p fits do not run this module; the p+A builder
# (event_builder_pA, parallel, glauber) and its imports do. Code reviews of the p+p model can skip it; review it
# with the p+A work. The cleanup and refactor after the p+A run decides its future.
"""Event container and resonance decay engine.

Ported from ``TIdEvent.h`` / ``TIdEvent.cxx``.

:class:`Event` holds a list of :class:`~na49_mixing.particle.Particle`
objects and performs two-body and three-body decays.
"""

from __future__ import annotations
import math
import sys
from typing import Optional, List, Dict, Tuple, TypedDict
import numpy as np

from .particle import Particle
from .decays import (
    ParticleDecays,
    DecayMode,
    NoLegalChannelError,
    read_decays,
)
from .decays import WEAK_DECAY_PARENTS, channel_parent_openness, mode_mass_windows
from .breit_wigner import BreitWigner, draw_truncated_bw
from .pdg_database import get_charge


class DecayStats(TypedDict):
    """Typed payload of :meth:`Event.decay_stats`.

    Counters are plain ints; ``primary_pdg_stats`` maps each primary PDG
    id to ``[registered, decayed]``.  The mass-conditioned channel
    machinery adds per-parent-PDG audit tables keyed by the daughter-PDG
    tuple of each channel: ``channel_requested`` (mean explicitly
    renormalized requested probability per channel), ``channel_realized``
    (successful fraction per channel), and ``openness_fraction`` (mean
    daughter-slot openness fraction per channel). Channels that are
    kinematically closed at the realized parent masses carry zero
    requested and zero realized fraction. When the Event is constructed
    with parent line shapes, ``parent_openness_fraction`` records the
    mean parent-side line-shape openness (fraction of the parent line
    shape above the channel threshold) of each selected channel; parents
    without a supplied line shape contribute no entries there.
    """

    n_decayed_parents: int
    n_decay_kinematic_failures: int
    n_decay_attempts: int
    n_primary_generated: int
    n_primary_decayed: int
    primary_pdg_stats: Dict[int, List[int]]
    channel_requested: Dict[int, Dict[Tuple[int, ...], float]]
    channel_realized: Dict[int, Dict[Tuple[int, ...], float]]
    openness_fraction: Dict[int, Dict[Tuple[int, ...], float]]
    parent_openness_fraction: Dict[int, Dict[Tuple[int, ...], float]]


class Event:
    """Container for identified particles in one collision event.

    Parameters
    ----------
    n_run : int
        Run number.
    n_event : int
        Event number within the run.
    rng : numpy.random.Generator, optional
        Shared random-number generator.
    """

    def __init__(
        self,
        n_run: int = 0,
        n_event: int = 0,
        rng: Optional[np.random.Generator] = None,
        resolved_match_mass: Optional[float] = None,
        parent_line_shapes: Optional[Dict[int, BreitWigner]] = None,
    ) -> None:
        self._particles: List[Particle] = []
        self._n_run = int(n_run)
        self._n_event = int(n_event)
        self._n_grey: int = 0
        self._e_veto: float = 0.0
        self._weight: float = 1.0
        self._has_fsb: int = 0
        self._fsb: Optional[Particle] = None
        self._decay_chain: Dict[int, ParticleDecays] = {}
        self._rng: np.random.Generator = rng if rng is not None else np.random.default_rng()
        self._iter_index: int = 0
        # Diagnostics: kinematic rejections that exhaust the retry budget leave
        # parents undecayed and can bias cascade feed-down if rates are high.
        self._n_decayed_parents: int = 0
        self._n_decay_kinematic_failures: int = 0
        self._n_decay_attempts: int = 0
        # Resolved-region cap (card decision D3): upper bound on drawn
        # resonance daughter masses.  Inactive (None) by default; when
        # inactive the cap reduces to the realized parent mass.
        if resolved_match_mass is not None:
            val = float(resolved_match_mass)
            if not (math.isfinite(val) and val > 0.0):
                raise ValueError(
                    f"Event resolved_match_mass must be None (inactive) or a "
                    f"finite positive float, got {resolved_match_mass!r}"
                )
        self._resolved_match_mass: Optional[float] = (
            None if resolved_match_mass is None else float(resolved_match_mass)
        )
        # Parent line shapes for the parent-side openness audit (stage-B
        # peak-openness diagnostic): PDG id -> Breit-Wigner of that
        # parent's mass distribution (e.g. the component primary's card
        # line shape). Parents absent from the map contribute no entries
        # to ``parent_openness_fraction``; the map never affects sampling.
        self._parent_line_shapes: Dict[int, BreitWigner] = (
            dict(parent_line_shapes) if parent_line_shapes else {}
        )
        # Primary-parent outcome bookkeeping: counts only generated primaries,
        # so secondary cascade decays cannot mask failed primaries.
        self._n_primary_generated: int = 0
        self._n_primary_decayed: int = 0
        self._primary_pdg_stats: Dict[int, List[int]] = {}
        # Mass-conditioned channel audit tables (per parent PDG):
        # attempts and requested-weight sums per daughter-PDG tuple, plus
        # the realized success count and openness fraction accumulator.
        self._channel_attempts: Dict[int, int] = {}
        self._channel_requested_sum: Dict[int, Dict[Tuple[int, ...], float]] = {}
        self._channel_realized: Dict[int, Dict[Tuple[int, ...], int]] = {}
        self._channel_openness_sum: Dict[int, Dict[Tuple[int, ...], float]] = {}
        self._channel_parent_openness_sum: Dict[int, Dict[Tuple[int, ...], float]] = {}
        self._channel_parent_openness_count: Dict[int, Dict[Tuple[int, ...], int]] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def n_run(self) -> int: return self._n_run
    @property
    def n_event(self) -> int: return self._n_event
    @property
    def n_particles(self) -> int: return len(self._particles)
    @property
    def weight(self) -> float: return self._weight

    def set_weight(self, w: float) -> None: self._weight = float(w)
    def set_n_run(self, n: int) -> None: self._n_run = int(n)
    def set_n_event(self, n: int) -> None: self._n_event = int(n)
    def set_n_grey(self, n: int) -> None: self._n_grey = int(n)
    def set_e_veto(self, e: float) -> None: self._e_veto = float(e)
    def get_n_grey(self) -> int: return self._n_grey
    def get_eveto(self) -> float: return self._e_veto
    def get_fsb(self) -> Optional[Particle]: return self._fsb
    def get_has_fsb(self) -> int: return self._has_fsb

    def particles(self) -> List[Particle]:
        """Return the internal particle list (direct reference)."""
        return self._particles

    def reset_decay_stats(self) -> None:
        """Clear cumulative decay diagnostic counters."""
        self._n_decayed_parents = 0
        self._n_decay_kinematic_failures = 0
        self._n_decay_attempts = 0
        self._n_primary_generated = 0
        self._n_primary_decayed = 0
        self._primary_pdg_stats = {}
        self._channel_attempts = {}
        self._channel_requested_sum = {}
        self._channel_realized = {}
        self._channel_openness_sum = {}
        self._channel_parent_openness_sum = {}
        self._channel_parent_openness_count = {}

    def mark_primary(self, particle: Particle) -> None:
        """Register *particle* as a generated primary parent for decay bookkeeping."""
        particle.set_is_primary(1)
        self._n_primary_generated += 1
        entry = self._primary_pdg_stats.setdefault(particle.get_id(), [0, 0])
        entry[0] += 1

    def _record_channel_outcome(
        self,
        parent_pdg: int,
        channel_key: Tuple[int, ...],
        openness: float,
        realized: bool,
        parent_openness: Optional[float] = None,
    ) -> None:
        """Accumulate the per-channel audit tables for one mode selection.

        ``channel_realized`` accumulates the success indicator and
        ``openness_fraction`` the channel's daughter-slot openness at the
        realized parent mass; the requested fractions are accumulated
        separately from the full renormalized weight vector (see
        ``_decay_once``), so ``channel_requested[c]`` estimates
        ``E[P(c|M)]``, which ``channel_realized[c]`` matches in
        expectation. When a parent line shape is supplied for the parent
        PDG, ``parent_openness`` carries the channel's parent-side
        line-shape openness; ``parent_openness_fraction`` reports its
        mean over the SELECTIONS of that channel (the openness is a
        property of the channel and the parent line shape, so this mean
        is the channel's peak-openness diagnostic rather than a
        selection-frequency-weighted quantity). Attempts are counted per
        parent PDG for the remaining tables, so their reported means are
        fractions of all channel-selected decay attempts of that parent
        type.
        """
        self._channel_attempts[parent_pdg] = (
            self._channel_attempts.get(parent_pdg, 0) + 1
        )
        rea = self._channel_realized.setdefault(parent_pdg, {})
        rea[channel_key] = rea.get(channel_key, 0) + (1 if realized else 0)
        opn = self._channel_openness_sum.setdefault(parent_pdg, {})
        opn[channel_key] = opn.get(channel_key, 0.0) + float(openness)
        if parent_openness is not None:
            popn = self._channel_parent_openness_sum.setdefault(parent_pdg, {})
            popn[channel_key] = popn.get(channel_key, 0.0) + float(parent_openness)
            poc = self._channel_parent_openness_count.setdefault(parent_pdg, {})
            poc[channel_key] = poc.get(channel_key, 0) + 1

    def decay_stats(self) -> DecayStats:
        """Return cumulative decay success/failure counters for this event.

        ``n_primary_generated`` / ``n_primary_decayed`` count only particles
        registered via :meth:`mark_primary`; ``primary_pdg_stats`` maps each
        primary PDG ID to ``[registered, decayed]``.  The channel audit
        tables report, per parent PDG, the mean explicitly renormalized
        requested probability (``channel_requested``), the successful
        fraction (``channel_realized``) and the mean daughter-slot openness
        fraction (``openness_fraction``) of each channel; channels that are
        kinematically closed at the realized parent masses carry zero
        requested and zero realized fraction there. When parent line
        shapes were supplied to the constructor,
        ``parent_openness_fraction`` reports the mean parent-side
        line-shape openness (fraction of the parent line shape above the
        channel threshold) over the selections of each channel.
        """
        channel_requested_out: Dict[int, Dict[Tuple[int, ...], float]] = {}
        channel_realized_out: Dict[int, Dict[Tuple[int, ...], float]] = {}
        openness_out: Dict[int, Dict[Tuple[int, ...], float]] = {}
        parent_openness_out: Dict[int, Dict[Tuple[int, ...], float]] = {}
        for pdg_id, attempts in self._channel_attempts.items():
            denom = float(attempts)
            if denom <= 0.0:
                continue
            req_sum = self._channel_requested_sum.get(pdg_id, {})
            rea = self._channel_realized.get(pdg_id, {})
            open_sum = self._channel_openness_sum.get(pdg_id, {})
            parent_open_sum = self._channel_parent_openness_sum.get(pdg_id, {})
            parent_open_cnt = self._channel_parent_openness_count.get(pdg_id, {})
            keys = set(req_sum) | set(rea) | set(open_sum) | set(parent_open_sum)
            req_means: Dict[Tuple[int, ...], float] = {}
            rea_means: Dict[Tuple[int, ...], float] = {}
            open_means: Dict[Tuple[int, ...], float] = {}
            parent_open_means: Dict[Tuple[int, ...], float] = {}
            for key in keys:
                req_val = req_sum.get(key)
                if req_val is not None:
                    req_means[key] = float(req_val) / denom
                rea_count = rea.get(key)
                if rea_count is not None:
                    rea_means[key] = float(rea_count) / denom
                open_val = open_sum.get(key)
                if open_val is not None:
                    open_means[key] = float(open_val) / denom
                if key in parent_open_sum:
                    n_sel = int(parent_open_cnt.get(key, 0))
                    if n_sel > 0:
                        parent_open_means[key] = (
                            float(parent_open_sum[key]) / float(n_sel)
                        )
            channel_requested_out[pdg_id] = req_means
            channel_realized_out[pdg_id] = rea_means
            openness_out[pdg_id] = open_means
            if parent_open_means:
                parent_openness_out[pdg_id] = parent_open_means
        return {
            "n_decayed_parents": int(self._n_decayed_parents),
            "n_decay_kinematic_failures": int(self._n_decay_kinematic_failures),
            "n_decay_attempts": int(self._n_decay_attempts),
            "n_primary_generated": int(self._n_primary_generated),
            "n_primary_decayed": int(self._n_primary_decayed),
            "primary_pdg_stats": {
                pdg: [int(reg), int(dec)]
                for pdg, (reg, dec) in self._primary_pdg_stats.items()
            },
            "channel_requested": channel_requested_out,
            "channel_realized": channel_realized_out,
            "openness_fraction": openness_out,
            "parent_openness_fraction": parent_openness_out,
        }

    # ------------------------------------------------------------------
    # Particle management
    # ------------------------------------------------------------------
    def add_particle(self, particle: Particle) -> None:
        self._particles.append(particle)

    def clear(self) -> None:
        """Remove all particles without deleting them."""
        self._particles.clear()
        self._has_fsb = 0
        self._fsb = None
        self._n_grey = 0
        self._e_veto = 0.0
        self._weight = 1.0

    def delete(self) -> None:
        """Delete all particles and clear the list."""
        self._particles.clear()

    def remove(self, particle: Particle) -> None:
        try:
            self._particles.remove(particle)
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Iterator interface (mirrors C++ SetToFirst / GetNext)
    # ------------------------------------------------------------------
    def set_to_first(self) -> None:
        self._iter_index = 0

    def get_next(self) -> Optional[Particle]:
        if self._iter_index < len(self._particles):
            p = self._particles[self._iter_index]
            self._iter_index += 1
            return p
        return None

    def __iter__(self):
        return iter(list(self._particles))

    # ------------------------------------------------------------------
    # Decay table loading
    # ------------------------------------------------------------------
    def read_decays(self, filename: str) -> None:
        """Load a decay table from *filename*.

        This populates the internal decay chain used by :meth:`decay`.
        """
        self._decay_chain = read_decays(filename, rng=self._rng)
        print("Decays:", file=sys.stderr)
        for parent_id, d in self._decay_chain.items():
            mode = d.get_mode()
            print(
                f"  {parent_id} -> {' '.join(str(x) for x in mode.daughters)}",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    # Two-body decay helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _lorentz_boost(
        p4: List[float],
        parent_p: float,
        parent_e: float,
        parent_px: float,
        parent_py: float,
        parent_pz: float,
    ) -> List[float]:
        """Boost a 4-vector p4=[E, px, py, pz] from the parent rest frame
        to the lab frame using the parent's 4-momentum.

        Implements the standard formula used in TIdEvent::decay().
        """
        betha = parent_p / parent_e
        gamma = 1.0 / math.sqrt(1.0 - betha * betha)
        e0, px0, py0, pz0 = p4

        skalsuc = px0 * parent_px + py0 * parent_py + pz0 * parent_pz
        factor = skalsuc * (gamma - 1.0) / (parent_p * parent_p) + gamma * betha * e0 / parent_p

        e_boosted = gamma * e0 + gamma * betha * skalsuc / parent_p
        return [
            e_boosted,
            px0 + parent_px * factor,
            py0 + parent_py * factor,
            pz0 + parent_pz * factor,
        ]

    _BW_MASS_DRAW_BUDGET = 200

    @staticmethod
    def _two_body_pstar(m_parent: float, m1: float, m2: float) -> float:
        """Daughter momentum |p*| in the parent rest frame (GeV/c)."""
        if m_parent <= 0.0:
            return 0.0
        arg = (m_parent * m_parent - (m1 + m2) ** 2) * (
            m_parent * m_parent - (m1 - m2) ** 2
        )
        if arg <= 0.0:
            return 0.0
        return math.sqrt(arg) / (2.0 * m_parent)


    @staticmethod
    def _stamp_ancestry(parent: "Particle", *daughters: "Particle") -> None:
        """Propagate cascade depth and weak-feed-down ancestry to daughters.

        Daughters also receive the direct ``parent`` link (stage-B
        ancestry traceability, card phase P2): an M_eff daughter of an
        enabled N* -> N*' + M* branch stays traceable to its N* parent,
        and cascade descendants inherit the chain through their own
        parent links. The link is bookkeeping only; it never affects
        sampling or kinematics.
        """
        parent_weak = parent.get_weak_feed()
        parent_depth = parent.get_gen_depth()
        self_weak = 1 if parent.get_id() in WEAK_DECAY_PARENTS else 0
        for d in daughters:
            d.parent = parent
            d.set_weak_feed(parent_weak | self_weak)
            d.set_gen_depth(parent_depth + 1)

    def _decay_2body(self, parent: Particle, mode: DecayMode) -> bool:
        """Perform a two-body decay of *parent* according to *mode*.

        Daughters are added to the particle list.  Returns ``True`` on
        success, ``False`` if kinematics are impossible.

        Daughter masses that come from a Breit-Wigner are drawn from the
        line shape restricted to the mass-conditioned legal window
        [schwel, min(M_source, M_match) - other daughters' minimum masses]
        (card decision D1): the truncation is normalized by construction,
        so the threshold rejection of the previous full-draw-then-reject
        loop is structurally absent.  Each draw is still accepted with
        probability q(m1, m2) / q_ref, where q = |p*| is the daughter
        momentum in the parent rest frame and q_ref evaluates q at the
        smallest drawable daughter masses (the lower edges of the legal
        windows).  This weights the truncated line shape by two-body
        phase space and softens the region near the kinematic limit, which
        a bare Breit-Wigner draw with pure threshold rejection leaves too
        hard.  A zero-width line shape is a delta at the pole: the pole is
        returned exactly when it lies inside the window, and the mode is
        ineligible when it does not.
        """
        id1 = mode.daughters[0]
        id2 = mode.daughters[1]

        d1 = Particle(reaction=parent.reaction)
        d2 = Particle(reaction=parent.reaction)
        d1.set_id(id1)
        d2.set_id(id2)

        m_parent = parent.get_mass()

        has_bw = mode.bws[0] is not None or mode.bws[1] is not None
        # Per-daughter legal draw windows at the realized parent mass.
        # ``None`` means the mode cannot decay at this mass under the
        # resolved cap (minimum daughter-mass sum over the bound, or a
        # zero-width pole outside the cap).
        windows = mode_mass_windows(mode, m_parent, self._resolved_match_mass)
        if windows is None:
            return False

        q_ref = 0.0
        if has_bw:
            # The smallest drawable masses maximise |p*|, which decreases
            # monotonically as either daughter mass grows; the window
            # lower edges are exactly those smallest drawable masses.
            q_ref = self._two_body_pstar(
                m_parent, float(windows[0][0]), float(windows[1][0])
            )

        n_draws = self._BW_MASS_DRAW_BUDGET if has_bw else 1
        m1 = m2 = 0.0
        accepted = False
        for _ in range(n_draws):
            # Optional BW mass for daughters, truncated to the legal window
            if mode.bws[0] is not None:
                mass1 = draw_truncated_bw(
                    mode.bws[0], self._rng, windows[0][0], windows[0][1]
                )
                if mass1 is None:
                    return False
                d1.set_mass(mass1)
            if mode.bws[1] is not None:
                mass2 = draw_truncated_bw(
                    mode.bws[1], self._rng, windows[1][0], windows[1][1]
                )
                if mass2 is None:
                    return False
                d2.set_mass(mass2)

            m1 = d1.get_mass()
            m2 = d2.get_mass()

            if m_parent <= 0.0 or (m1 + m2) > m_parent:
                continue

            if has_bw:
                q = self._two_body_pstar(m_parent, m1, m2)
                accept_prob = 1.0 if q_ref <= 0.0 else min(q / q_ref, 1.0)
                if self._rng.random() > accept_prob:
                    continue

            accepted = True
            break

        if not accepted:
            return False

        e1 = (m_parent * m_parent + m1 * m1 - m2 * m2) / (2.0 * m_parent)

        if e1 * e1 < m1 * m1:
            return False

        p1_len = math.sqrt(e1 * e1 - m1 * m1)

        # Isotropic in parent rest frame
        cos_theta = 2.0 * self._rng.random() - 1.0
        sin_theta = math.sqrt(max(1.0 - cos_theta * cos_theta, 0.0))
        phi = 2.0 * math.pi * self._rng.random()

        p1 = [
            e1,
            p1_len * sin_theta * math.cos(phi),
            p1_len * sin_theta * math.sin(phi),
            p1_len * cos_theta,
        ]

        # Boost to lab frame
        parent_p = parent.get_p()
        if parent_p != 0.0:
            p1_boosted = self._lorentz_boost(
                p1, parent_p, parent.get_e(),
                parent.get_px(), parent.get_py(), parent.get_pz(),
            )
        else:
            p1_boosted = p1

        # Daughter 2 from momentum conservation
        p2_px = parent.get_px() - p1_boosted[1]
        p2_py = parent.get_py() - p1_boosted[2]
        p2_pz = parent.get_pz() - p1_boosted[3]

        d1.set_p(p1_boosted[1], p1_boosted[2], p1_boosted[3])
        d2.set_p(p2_px, p2_py, p2_pz)

        self._stamp_ancestry(parent, d1, d2)
        self._particles.append(d1)
        self._particles.append(d2)
        return True

    # ------------------------------------------------------------------
    # Three-body decay helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compmass(x: float, params) -> float:
        """Constant-|M|² three-body weight as a function of s = m23².

        The flat Dalitz density is uniform in (m12², m23²); its projection
        on s = m23² is the width of the m12² band at fixed s,
        4 |p1| |p2*| = √λ(m0², m1², s) √λ(s, m2², m3²) / s, with |p1| the
        momentum of daughter 1 and |p2*| that of daughter 2 in the 23 rest
        frame (equivalently dΓ/dm23 ∝ |p1| |p2*|, PDG kinematics). This is
        the C++ ``compmass`` of ``TIdEvent.cxx``, (1/m0²)(1/s)√a√b.
        Overall constants drop out of the accept-reject envelope.
        """
        m0, m1, m2, m3 = params
        a = x * x - 2.0 * x * (m0 * m0 + m1 * m1) + (m0 * m0 - m1 * m1) ** 2
        b = x * x - 2.0 * x * (m2 * m2 + m3 * m3) + (m2 * m2 - m3 * m3) ** 2
        if a <= 0.0 or b <= 0.0 or x <= 0.0:
            return 0.0
        return (1.0 / (m0 * m0)) * (1.0 / x) * math.sqrt(a) * math.sqrt(b)

    def _decay_3body(self, parent: Particle, mode: DecayMode) -> bool:
        """Perform a three-body decay of *parent* according to *mode*.

        ``m23²`` is sampled with the constant-|M|² projection
        :meth:`_compmass`. No currently defined three-body mode assigns a
        Breit-Wigner to a daughter; if one is added, its draws run through
        the same mass-conditioned legal windows as the two-body path (see
        :meth:`_decay_2body`), and the residual sum-threshold retry below
        remains as a draw-level guard because independent per-slot windows
        cannot guarantee the three-mass sum.
        """
        id1, id2, id3 = mode.daughters[0], mode.daughters[1], mode.daughters[2]

        d1 = Particle(reaction=parent.reaction)
        d2 = Particle(reaction=parent.reaction)
        d3 = Particle(reaction=parent.reaction)
        d1.set_id(id1)
        d2.set_id(id2)
        d3.set_id(id3)

        m0 = parent.get_mass()
        # Mass-conditioned per-slot legal windows; ``None`` marks the mode
        # ineligible at the realized parent mass under the resolved cap.
        windows = mode_mass_windows(mode, m0, self._resolved_match_mass)
        if windows is None:
            return False

        # BW masses, truncated to the legal windows
        if mode.bws[0] is not None:
            mass1 = draw_truncated_bw(
                mode.bws[0], self._rng, windows[0][0], windows[0][1]
            )
            if mass1 is None:
                return False
            d1.set_mass(mass1)
        if mode.bws[1] is not None:
            mass2 = draw_truncated_bw(
                mode.bws[1], self._rng, windows[1][0], windows[1][1]
            )
            if mass2 is None:
                return False
            d2.set_mass(mass2)
        if mode.bws[2] is not None:
            mass3 = draw_truncated_bw(
                mode.bws[2], self._rng, windows[2][0], windows[2][1]
            )
            if mass3 is None:
                return False
            d3.set_mass(mass3)

        m1, m2, m3 = d1.get_mass(), d2.get_mass(), d3.get_mass()

        # Threshold check
        thresh = m1 + m2 + m3
        if thresh >= m0:
            return False

        # Retry loop for kinematically allowed daughter masses (draw-level
        # guard; per-slot windows already exclude the bulk of rejections)
        max_iter = 100
        for _ in range(max_iter):
            if mode.bws[0] is not None:
                mass1 = draw_truncated_bw(
                    mode.bws[0], self._rng, windows[0][0], windows[0][1]
                )
                if mass1 is None:
                    return False
                d1.set_mass(mass1)
            if mode.bws[1] is not None:
                mass2 = draw_truncated_bw(
                    mode.bws[1], self._rng, windows[1][0], windows[1][1]
                )
                if mass2 is None:
                    return False
                d2.set_mass(mass2)
            if mode.bws[2] is not None:
                mass3 = draw_truncated_bw(
                    mode.bws[2], self._rng, windows[2][0], windows[2][1]
                )
                if mass3 is None:
                    return False
                d3.set_mass(mass3)
            m1, m2, m3 = d1.get_mass(), d2.get_mass(), d3.get_mass()
            if m1 + m2 + m3 < m0:
                break
        else:
            return False

        params = (m0, m1, m2, m3)

        # Dalitz-plot limits for m_{23}^2 = (p_2 + p_3)^2:
        #   minimum = (m2 + m3)^2  (threshold)
        #   maximum = (m0 - m1)^2  (all phase space to the 23 subsystem)
        min_m23sq = (m2 + m3) ** 2
        max_m23sq = (m0 - m1) ** 2

        # Scan for maximum of phase-space function
        eps = (max_m23sq - min_m23sq) / 1000.0
        maximum = 0.0
        scan = min_m23sq
        while scan < max_m23sq:
            v = self._compmass(scan, params)
            if v > maximum:
                maximum = v
            scan += eps
        maximum *= 1.1

        if maximum <= 0.0:
            return False

        # Accept-reject for m23
        m23_sq = 0.0
        for _ in range(100_000):
            t1 = self._rng.random() * (max_m23sq - min_m23sq) + min_m23sq
            t2 = self._rng.random() * maximum
            if t2 < self._compmass(t1, params):
                m23_sq = t1
                break
        else:
            return False

        m23 = math.sqrt(m23_sq)

        # Two-body decay of parent → d1 + (d2d3)
        e1 = (m0 * m0 - m23 * m23 + m1 * m1) / (2.0 * m0)
        pl = math.sqrt(max(e1 * e1 - m1 * m1, 0.0))
        phi = 2.0 * math.pi * self._rng.random()
        ct = 2.0 * self._rng.random() - 1.0
        st = math.sqrt(max(1.0 - ct * ct, 0.0))

        p1 = [e1, pl * st * math.sin(phi), pl * st * math.cos(phi), pl * ct]
        p23 = [
            math.sqrt(pl * pl + m23 * m23),
            -p1[1], -p1[2], -p1[3],
        ]

        # Two-body decay of (d2d3) → d2 + d3
        e2 = (m23 * m23 - m3 * m3 + m2 * m2) / (2.0 * m23)
        pl2 = math.sqrt(max(e2 * e2 - m2 * m2, 0.0))
        phi2 = 2.0 * math.pi * self._rng.random()
        ct2 = 2.0 * self._rng.random() - 1.0
        st2 = math.sqrt(max(1.0 - ct2 * ct2, 0.0))

        p2 = [e2, pl2 * st2 * math.sin(phi2), pl2 * st2 * math.cos(phi2), pl2 * ct2]
        p3 = [math.sqrt(pl2 * pl2 + m3 * m3), -p2[1], -p2[2], -p2[3]]

        # Boost d2, d3 from (d2d3) rest frame to parent rest frame
        p23_len = math.sqrt(p23[1] ** 2 + p23[2] ** 2 + p23[3] ** 2)
        if p23_len > 0.0:
            betha_23 = p23_len / p23[0]
            gamma_23 = 1.0 / math.sqrt(1.0 - betha_23 * betha_23)
            # Boosts particles to parent rest frame; conserves momentum
            for pv in (p2, p3):
                skalp = pv[1] * p23[1] + pv[2] * p23[2] + pv[3] * p23[3]
                factor = skalp * (gamma_23 - 1.0) / (p23_len * p23_len) + gamma_23 * betha_23 * pv[0] / p23_len
                pv[1] += p23[1] * factor
                pv[2] += p23[2] * factor
                pv[3] += p23[3] * factor
                # Recalculates energy component for momentum conservation
                pv[0] = math.sqrt(pv[1] ** 2 + pv[2] ** 2 + pv[3] ** 2 + (m2 if pv is p2 else m3) ** 2)
            # Recalculates energy for third particle
            p3[0] = math.sqrt(p3[1] ** 2 + p3[2] ** 2 + p3[3] ** 2 + m3 * m3)

        # Boost all three daughters to lab frame
        parent_p = parent.get_p()
        if parent_p > 0.0:
            for pv in (p1, p2, p3):
                boosted = self._lorentz_boost(
                    pv, parent_p, parent.get_e(),
                    parent.get_px(), parent.get_py(), parent.get_pz(),
                )
                pv[1], pv[2], pv[3] = boosted[1], boosted[2], boosted[3]

        d1.set_p(p1[1], p1[2], p1[3])
        d2.set_p(p2[1], p2[2], p2[3])
        d3.set_p(p3[1], p3[2], p3[3])

        self._stamp_ancestry(parent, d1, d2, d3)
        self._particles.append(d1)
        self._particles.append(d2)
        self._particles.append(d3)
        return True

    # ------------------------------------------------------------------
    # High-level decay API
    # ------------------------------------------------------------------
    def decay_2b(self, which_id: int, first_id: int, second_id: int) -> None:
        """Decay all particles of type *which_id* → *first_id* + *second_id*."""
        to_remove = []
        mode = DecayMode(2, [first_id, second_id])
        for part in list(self._particles):
            if part.get_id() != which_id:
                continue
            if self._decay_2body(part, mode):
                to_remove.append(part)
        for p in to_remove:
            self.remove(p)

    def _decay_once(self) -> int:
        """Run one pass through the decay chain and return number of decayed parents.

        Channel selection runs through the mass-conditioned channel set
        C(M) built at the realized parent mass (card decision D1): the
        selected channel is legal by construction, its branch weight is
        renormalized explicitly over the legal set, and the previous
        sample-then-retry loop (up to 10,000 mode re-samples per parent)
        is removed from the decision path. A decay that still fails at the
        draw level (phase-space envelope budget, or C(M) empty at this
        mass) leaves the parent undecayed and counts once toward
        ``n_decay_kinematic_failures``.
        """
        particles = self._particles
        if not particles or not self._decay_chain:
            return 0

        # Build an index once per pass. This avoids scanning the full particle
        # list for every parent ID in the decay table.
        by_id: Dict[int, List[Particle]] = {}
        for p in particles:
            by_id.setdefault(p._pdg_id, []).append(p)

        removed: set[Particle] = set()
        n_decayed = 0

        for pdg_id, pdecays in self._decay_chain.items():
            parents = by_id.get(pdg_id)
            if not parents:
                continue

            # Public per-channel keys for the audit tables, and the
            # optional parent line shape for the parent-side openness
            # audit; both resolved once per parent PDG.
            channel_keys = pdecays.channel_keys()
            parent_bw = self._parent_line_shapes.get(pdg_id)

            # Snapshot the parent list for this PDG. This preserves previous
            # behavior where daughters created during this PDG step are only
            # eligible in subsequent PDG steps / cascade passes.
            for part in tuple(parents):
                if part in removed:
                    continue

                self._n_decay_attempts += 1
                m_parent = float(part.get_mass())
                try:
                    mode, mode_index, p_cond, openness, weights = (
                        pdecays.get_mode_conditional(
                            m_parent, self._resolved_match_mass
                        )
                    )
                except NoLegalChannelError:
                    # C(M) is empty at the realized parent mass: the parent
                    # cannot decay under the source constraint. This is the
                    # same undecayed-parent outcome the old retry loop
                    # produced after exhausting its budget, recorded once
                    # instead of after 10,000 illegal re-samples.
                    self._n_decay_kinematic_failures += 1
                    continue

                parent_openness: Optional[float] = None
                if parent_bw is not None:
                    parent_openness = channel_parent_openness(
                        mode, parent_bw, self._resolved_match_mass
                    )

                n_before = len(particles)
                if mode.n_daughters == 2:
                    ok = self._decay_2body(part, mode)
                elif mode.n_daughters == 3:
                    ok = self._decay_3body(part, mode)
                else:
                    ok = False

                channel_key = channel_keys[mode_index]
                self._record_channel_outcome(
                    pdg_id,
                    channel_key,
                    openness,
                    realized=ok,
                    parent_openness=parent_openness,
                )
                # Requested fractions accumulate the full renormalized
                # vector: channel_requested[c] estimates E[P(c|M)], which
                # channel_realized[c] (= realized success fraction)
                # matches in expectation.
                for i, w_c in enumerate(weights):
                    if w_c <= 0.0:
                        continue
                    key_c = channel_keys[i]
                    self._channel_requested_sum.setdefault(pdg_id, {})
                    tbl = self._channel_requested_sum[pdg_id]
                    tbl[key_c] = tbl.get(key_c, 0.0) + float(w_c)

                if ok:
                    removed.add(part)
                    n_decayed += 1
                    self._n_decayed_parents += 1
                    # Only generated primaries count here; daughters keep
                    # the flag at 0, so secondary decays stay excluded.
                    if part.get_is_primary():
                        self._n_primary_decayed += 1
                        entry = self._primary_pdg_stats.setdefault(
                            part.get_id(), [0, 0]
                        )
                        entry[1] += 1
                    # Register newly appended daughters so later PDG IDs in
                    # this pass can see them without a full rescan.
                    if len(particles) > n_before:
                        for d in particles[n_before:]:
                            by_id.setdefault(d._pdg_id, []).append(d)
                else:
                    self._n_decay_kinematic_failures += 1

        if removed:
            self._particles = [p for p in particles if p not in removed]
        return n_decayed

    def decay(self) -> None:
        """Decay all non-stable particles using one pass through the loaded chain."""
        self._decay_once()

    def decay_cascade(self, max_passes: int = 8) -> int:
        """Decay recursively by re-running decay passes until convergence.

        This is useful when a decay table contains intermediate resonances
        that only become available after the first pass due to parent ordering.

        Returns the total number of decayed parents over all passes.
        """
        total_decayed = 0
        for _ in range(max(1, int(max_passes))):
            n_decayed = self._decay_once()
            total_decayed += n_decayed
            # Fixed-point stop: once one full pass performs zero decays,
            # no additional parents in the current chain can be consumed.
            if n_decayed == 0:
                break
        return total_decayed

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"Event(run={self._n_run}, event={self._n_event}, "
            f"n_particles={len(self._particles)})"
        )
