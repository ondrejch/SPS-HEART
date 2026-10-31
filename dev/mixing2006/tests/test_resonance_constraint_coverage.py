"""Coverage extensions for the source-constrained decay machinery.

Companion to ``tests/test_resonance_constraint_workflow.py`` (card
TASK-2026-08-31-resonance-decay-fit-workflow, phase P1, committed as
fc520a5). The acceptance suite pins tests (i), (ii), (iii), (v), (viii)
and the golden-fixture regression; this module closes the review gaps
that suite leaves open:

- zero-width (``brei = 0``) Breit-Wigner daughters exercised through the
  full engine decision path (``_decay_once`` -> ``get_mode_conditional``
  -> ``_decay_2body``), pole inside and outside the realized-mass window
  and outside an active ``resolved_match_mass`` cap;
- the engine-level empty-C(M) outcome: ``NoLegalChannelError`` is caught
  once per parent, the parent stays undecayed, and the per-channel audit
  tables stay untouched for that parent type;
- audit-counter semantics: ``channel_requested`` sums to exactly 1 per
  attempt over mixed realized masses and estimates ``E[P(c|M)]``;
  ``openness_fraction`` coincides with ``channel_realized`` for
  fixed-mass channels; a draw-level failure records exactly one
  kinematic failure with zero realized fraction while the requested
  vector still sums to 1;
- fast-path bit parity of ``get_mode_conditional`` against ``get_mode``
  on the real factory tables (the multi-channel, BW-free parent class),
  not only on a synthetic table;
- ``resolved_match_mass`` plumbed through the p+A chunk worker (A=1
  point target, stream-identical to the pp path);
- ``Event`` constructor validation of ``resolved_match_mass``.

Phase P2 review gaps (commit 5b2f37d) closed here:

- the D5 pair family interacting with C(M) eligibility: when a pionic
  channel is kinematically ineligible at the realized mass while the
  pair share is active, the level budget still sums to 1 after the
  explicit renormalization walk and the surviving pionic weights keep
  their declared ratios; a resolved cap below 2 m_p evaporates the pair
  share and recovers the declared pionic vector exactly;
- the threshold boundary at ``M = M_th`` exactly: the pair weight is the
  float ``0.0`` (hard zero, not merely small) and the selection never
  returns the pair channel; the share approaches the strength ``c``
  from strictly below far above threshold;
- the three-body ``p pbar pi0`` variant end to end: at a fixed parent
  mass above 2.0115 GeV the realized two-body/three-body split is 50/50,
  below 2.0115 GeV the variant is absent from the realized channels, and
  its closure renormalizes the surviving pair member through the global
  C(M) walk;
- the charged anti-tower mirror links ``[-9912201, 9900213]`` and
  ``[-9922201, 9900313]`` for enabled levels 2/3 (branch-by-branch
  charge conjugation of the matter nstarmeson link) and the level-1
  terminal exception (no ``-2216`` daughter in the frozen table);
- ``Particle.parent`` ancestry bookkeeping surviving into parallel
  chunk workers (pair daughters carry cascade depth 2, weak-feed 0);
- the parent-openness audit degrading gracefully when ``Event`` is
  constructed without ``parent_line_shapes`` (absent table, not zeros).
"""
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from na49_mixing import pdg_database
from na49_mixing.breit_wigner import BreitWigner
from na49_mixing.decays import (
    DecayMode,
    MESTAR_PPBAR_PI0_VARIANT_KEY,
    MESTAR_PPBAR_STRENGTH_KEY,
    ParticleDecays,
    build_soft_resonance_decays,
    daughter_minimum_mass,
    ppbar_pair_threshold,
)
from na49_mixing.event import Event
from na49_mixing.glauber import NucleusSpec
from na49_mixing.parallel import (
    GenerationConfig,
    _inject_antibaryon_resonance_decays,
    generate_tracks_chunked,
    stage_b_strength_split,
    STAGE_B_STRENGTH_BUDGET_WIDE,
)
from na49_mixing.particle import Particle
from na49_mixing.reaction import Reaction


BEAM_MOMENTUM = 158.0

# Committed D5 pair threshold: 2 m_p derived from the PDG table.
_PAIR_THRESHOLD = ppbar_pair_threshold()

# Pole of the zero-width rho line shape used by the engine-path delta
# tests below. A zero nominal width is legal (BreitWigner._validate_params)
# and the class treats the resulting zero-total-density line shape as a
# delta at rmas; this is the documented brei = 0 route of the card.
_DELTA_POLE = 0.77526
_PI0_MASS = 0.1349766


class _RecordingEvent(Event):
    """Event that records every performed decay with its daughter objects."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.decay_records: list = []

    def _decay_2body(self, parent, mode):  # type: ignore[override]
        n_before = len(self._particles)
        ok = super()._decay_2body(parent, mode)
        if ok:
            self.decay_records.append(
                (float(parent.get_mass()), mode, tuple(self._particles[n_before:]))
            )
        return ok

    def _decay_3body(self, parent, mode):  # type: ignore[override]
        n_before = len(self._particles)
        ok = super()._decay_3body(parent, mode)
        if ok:
            self.decay_records.append(
                (float(parent.get_mass()), mode, tuple(self._particles[n_before:]))
            )
        return ok


class _Failing2BodyEvent(Event):
    """Event whose two-body decays always fail at the draw level.

    Simulates the documented draw-level failure branch of
    ``Event._decay_once`` (phase-space envelope budget exhausted after a
    channel was legally selected from C(M)) without touching production
    code, so the audit counters' response to ``ok == False`` is pinned.
    """

    def _decay_2body(self, parent, mode):  # type: ignore[override]
        return False


def _mode_min_mass_sum(mode: DecayMode) -> float:
    return sum(
        daughter_minimum_mass(pid, bw)
        for pid, bw in zip(mode.daughters, mode.bws)
    )


def _delta_pair_table(seed: int = 71) -> dict:
    """Synthetic M_eff(850) table with a zero-width BW daughter slot.

    Channel A (cumulative 0.5): rho0 + pi0 where the rho0 mass comes from
    a zero-width Breit-Wigner (delta at ``_DELTA_POLE``). Channel B
    (cumulative 1.0): pi+ pi-, legal whenever the pair threshold
    (0.2793 GeV) fits under the realized bound.
    """
    rng = np.random.default_rng(seed)
    d = ParticleDecays(9900213, rng=rng)
    delta_rho = BreitWigner(0.2791, _DELTA_POLE, 0.0, 1.0, 1.2, 1.5, rng=rng)
    assert delta_rho._delta_mass == pytest.approx(_DELTA_POLE)
    d.add_mode(0.5, DecayMode(2, [113, 111], bws=[delta_rho, None]))
    d.add_mode(1.0, DecayMode(2, [211, -211]))
    return {9900213: d}


def _delta_only_table(seed: int = 73) -> dict:
    """Synthetic table whose only channel carries the zero-width BW."""
    rng = np.random.default_rng(seed)
    d = ParticleDecays(9900213, rng=rng)
    delta_rho = BreitWigner(0.2791, _DELTA_POLE, 0.0, 1.0, 1.2, 1.5, rng=rng)
    d.add_mode(1.0, DecayMode(2, [113, 111], bws=[delta_rho, None]))
    return {9900213: d}


def _triple_channel_table(seed: int = 81) -> dict:
    """Synthetic M_eff(850) table: pionic + cascade + thresholded pair.

    Declared incremental weights 0.5 / 0.3 / 0.2 over three fixed-mass
    channels; the pair channel is closed below 2 m_p = 1.8765 GeV.
    """
    rng = np.random.default_rng(seed)
    d = ParticleDecays(9900213, rng=rng)
    d.add_mode(0.5, DecayMode(2, [211, -211]))      # pi+ pi-
    d.add_mode(0.8, DecayMode(2, [9900113, 111]))   # M_eff(450) + pi0
    d.add_mode(1.0, DecayMode(2, [2212, -2212]))    # p pbar pair channel
    return {9900213: d}


# ---------------------------------------------------------------------------
# Zero-width (brei = 0) Breit-Wigner through the engine decision path
# ---------------------------------------------------------------------------


class TestZeroWidthBreitWignerEnginePath:
    """brei = 0 through _decay_once -> get_mode_conditional -> _decay_2body.

    The helper-level delta coverage in
    tests/test_resonance_constraint_workflow.py pins the truncated draw
    primitives; here the same zero-width configuration must survive the
    full selection-plus-decay path: the pole is returned exactly when it
    fits the realized-mass window, and the channel drops out of C(M)
    when it does not (closed by the parent mass or by the resolved cap).
    """

    def test_delta_pole_inside_window_decays_at_pole(self):
        chain = _delta_pair_table()
        ev = _RecordingEvent(rng=np.random.default_rng(24))
        ev._decay_chain = chain
        reaction = Reaction(BEAM_MOMENTUM)
        n_parents = 4000
        for _ in range(n_parents):
            parent = Particle.from_id(0.0, 0.0, 0.5, 9900213, reaction)
            parent.set_mass(2.0)
            ev.add_particle(parent)
            ev.mark_primary(parent)
            ev.decay_cascade(max_passes=8)
            ev.delete()
        stats = ev.decay_stats()
        assert stats["n_decay_attempts"] == n_parents
        assert stats["n_decay_kinematic_failures"] == 0
        # Every performed delta-channel decay places the BW daughter
        # exactly on the pole: the truncated draw returns float(rmas)
        # with no sampling spread.
        pole_decays = 0
        for _m_parent, mode, daughters in ev.decay_records:
            if tuple(mode.daughters) == (113, 111):
                assert float(daughters[0].get_mass()) == _DELTA_POLE
                assert float(daughters[1].get_mass()) == pytest.approx(_PI0_MASS)
                pole_decays += 1
        # Realized branching matches the declared 0.5 share within 3 sigma.
        p = 0.5
        sigma = math.sqrt(p * (1.0 - p) / n_parents)
        assert abs(pole_decays / n_parents - p) <= 3.0 * sigma
        # Audit: the requested vector is the declared one, and for
        # fixed-acceptance channels the openness fraction coincides with
        # the realized fraction exactly (openness is 1.0 per selection).
        requested = stats["channel_requested"][9900213]
        realized = stats["channel_realized"][9900213]
        openness = stats["openness_fraction"][9900213]
        assert requested[(113, 111)] == pytest.approx(0.5, abs=1e-9)
        assert requested[(211, -211)] == pytest.approx(0.5, abs=1e-9)
        assert openness[(113, 111)] == realized[(113, 111)]
        assert openness[(211, -211)] == realized[(211, -211)]

    def test_delta_pole_above_parent_mass_channel_excluded(self):
        chain = _delta_pair_table()
        # Parent mass 0.7: the pi+pi- threshold (0.2793) fits, but the
        # zero-width pole (0.77526) sits above the allowed draw window
        # [schwel, 0.7 - 0.135 = 0.565], so bw_draw_window returns None
        # and the channel leaves C(M) by construction instead of being
        # sampled and rejected at the draw level.
        ev = _RecordingEvent(rng=np.random.default_rng(25))
        ev._decay_chain = chain
        reaction = Reaction(BEAM_MOMENTUM)
        n_parents = 500
        for _ in range(n_parents):
            parent = Particle.from_id(0.0, 0.0, 0.5, 9900213, reaction)
            parent.set_mass(0.7)
            ev.add_particle(parent)
            ev.mark_primary(parent)
            ev.decay_cascade(max_passes=8)
            ev.delete()
        stats = ev.decay_stats()
        assert stats["n_decay_attempts"] == n_parents
        assert stats["n_decay_kinematic_failures"] == 0
        assert stats["n_decayed_parents"] == n_parents
        realized = stats["channel_realized"][9900213]
        requested = stats["channel_requested"][9900213]
        assert set(realized) == {(211, -211)}
        assert set(requested) == {(211, -211)}
        assert realized[(211, -211)] == pytest.approx(1.0)
        for _m_parent, mode, _daughters in ev.decay_records:
            assert tuple(mode.daughters) == (211, -211)

    def test_delta_pole_above_resolved_cap_channel_excluded(self):
        chain = _delta_pair_table()
        # Same table at a parent mass far above the pole, but with the
        # resolved cap active at 0.7 GeV: min(M_source, M_match) closes
        # the delta channel through the cap branch of the window builder.
        ev = _RecordingEvent(
            rng=np.random.default_rng(26), resolved_match_mass=0.7
        )
        ev._decay_chain = chain
        reaction = Reaction(BEAM_MOMENTUM)
        n_parents = 500
        for _ in range(n_parents):
            parent = Particle.from_id(0.0, 0.0, 0.5, 9900213, reaction)
            parent.set_mass(2.0)
            ev.add_particle(parent)
            ev.mark_primary(parent)
            ev.decay_cascade(max_passes=8)
            ev.delete()
        stats = ev.decay_stats()
        assert stats["n_decay_attempts"] == n_parents
        assert stats["n_decay_kinematic_failures"] == 0
        realized = stats["channel_realized"][9900213]
        requested = stats["channel_requested"][9900213]
        assert set(realized) == {(211, -211)}
        assert set(requested) == {(211, -211)}
        assert realized[(211, -211)] == pytest.approx(1.0)

    @pytest.mark.parametrize("cap", [None, 0.5])
    def test_delta_only_channel_illegal_parent_undecays_once(self, cap):
        """Empty C(M) at the engine: loud, bounded, audit-silent.

        With the zero-width pole outside the realized bound (parent mass
        0.7, or the cap at 0.5 under a 2.0 GeV parent) the table's only
        channel is ineligible. The engine must catch the
        ``NoLegalChannelError`` once per parent, leave the parent
        undecayed, count exactly one kinematic failure (the retired
        retry loop burned up to 10,000 re-samples here), and record no
        per-channel audit entries because no channel was ever selected.
        """
        chain = _delta_only_table()
        ev = Event(
            rng=np.random.default_rng(27),
            resolved_match_mass=cap,
        )
        ev._decay_chain = chain
        reaction = Reaction(BEAM_MOMENTUM)
        n_parents = 5
        parent_mass = 0.7 if cap is None else 2.0
        for _ in range(n_parents):
            parent = Particle.from_id(0.0, 0.0, 0.5, 9900213, reaction)
            parent.set_mass(parent_mass)
            ev.add_particle(parent)
            ev.mark_primary(parent)
        ev.decay_cascade(max_passes=8)
        stats = ev.decay_stats()
        assert stats["n_decay_attempts"] == n_parents
        assert stats["n_decay_kinematic_failures"] == n_parents
        assert stats["n_decayed_parents"] == 0
        assert 9900213 not in stats["channel_requested"]
        assert 9900213 not in stats["channel_realized"]
        assert 9900213 not in stats["openness_fraction"]
        assert len(ev.particles()) == n_parents


# ---------------------------------------------------------------------------
# Audit-counter semantics
# ---------------------------------------------------------------------------


class TestAuditCounterSemantics:
    """channel_requested / channel_realized / openness_fraction contracts.

    ``channel_requested[c]`` accumulates the renormalized weight of
    channel c at EVERY attempt, so it estimates ``E[P(c|M)]`` and sums to
    exactly 1 per attempt even when the realized masses vary and close
    channels. ``channel_realized[c]`` counts successful selections.
    ``openness_fraction[c]`` accumulates the channel openness of the
    selected channel per attempt, which for fixed-mass channels (openness
    1.0) coincides with the realized fraction exactly.
    """

    def test_requested_vector_sums_to_one_over_mixed_realized_masses(self):
        chain = _triple_channel_table()
        ev = Event(rng=np.random.default_rng(91))
        ev._decay_chain = chain
        reaction = Reaction(BEAM_MOMENTUM)
        n_low, n_high = 2000, 2000
        # At 1.5 GeV the pair channel is closed: renormalized weights
        # (0.5, 0.3) / 0.8. At 2.0 GeV all three channels are legal with
        # declared weights (0.5, 0.3, 0.2).
        for mass, count in ((1.5, n_low), (2.0, n_high)):
            for _ in range(count):
                parent = Particle.from_id(0.0, 0.0, 0.5, 9900213, reaction)
                parent.set_mass(mass)
                ev.add_particle(parent)
                ev.mark_primary(parent)
                ev.decay_cascade(max_passes=8)
                ev.delete()
        stats = ev.decay_stats()
        assert stats["n_decay_attempts"] == n_low + n_high
        assert stats["n_decay_kinematic_failures"] == 0
        requested = stats["channel_requested"][9900213]
        realized = stats["channel_realized"][9900213]
        openness = stats["openness_fraction"][9900213]
        n = float(n_low + n_high)
        # E[P(c|M)] with the pair weight carried only by the high-mass
        # half: exact renormalized-arithmetic values, not statistics.
        assert requested[(211, -211)] == pytest.approx(0.5625, abs=1e-9)
        assert requested[(9900113, 111)] == pytest.approx(0.3375, abs=1e-9)
        assert requested[(2212, -2212)] == pytest.approx(0.1, abs=1e-9)
        assert sum(requested.values()) == pytest.approx(1.0, abs=1e-9)
        # Realized pair fraction matches its expectation within 3 sigma.
        p = 0.1
        sigma = math.sqrt(p * (1.0 - p) / n)
        assert abs(realized[(2212, -2212)] - p) <= 3.0 * sigma
        # Fixed-mass channels: openness per selection is exactly 1.0, so
        # the openness means equal the realized means key by key.
        for key in ((211, -211), (9900113, 111), (2212, -2212)):
            assert openness[key] == realized[key], key

    def test_draw_level_failure_records_one_failure_and_zero_realized(self):
        chain = _triple_channel_table()
        ev = _Failing2BodyEvent(rng=np.random.default_rng(92))
        ev._decay_chain = chain
        reaction = Reaction(BEAM_MOMENTUM)
        n_parents = 200
        for _ in range(n_parents):
            parent = Particle.from_id(0.0, 0.0, 0.5, 9900213, reaction)
            parent.set_mass(2.0)
            ev.add_particle(parent)
            ev.mark_primary(parent)
        ev.decay_cascade(max_passes=8)
        stats = ev.decay_stats()
        # One pass, one failure per parent, cascade fixed-point stop.
        assert stats["n_decay_attempts"] == n_parents
        assert stats["n_decay_kinematic_failures"] == n_parents
        assert stats["n_decayed_parents"] == 0
        assert len(ev.particles()) == n_parents
        requested = stats["channel_requested"][9900213]
        realized = stats["channel_realized"][9900213]
        openness = stats["openness_fraction"][9900213]
        # The requested vector still accumulates the full renormalized
        # weight vector per attempt, while every realized indicator is
        # zero: the divergence requested - realized is exactly the
        # draw-level failure signal the counters are built to expose.
        assert set(realized) <= {
            (211, -211), (9900113, 111), (2212, -2212)
        }
        assert sum(realized.values()) == 0.0
        assert all(v == 0.0 for v in realized.values())
        assert sum(requested.values()) == pytest.approx(1.0, abs=1e-9)
        # Openness is recorded for the selected channel even when the
        # draw fails, so the openness means still sum to one attempt
        # share while realized stays at zero.
        assert sum(openness.values()) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Fast-path bit parity on the real factory tables
# ---------------------------------------------------------------------------


class TestRealChainFastPathParity:
    """get_mode_conditional == get_mode bit for bit on factory tables.

    The card's RNG-parity claim ("when every declared channel is legal
    the walk runs over the loaded cumulative table unchanged") is pinned
    on a synthetic two-mode table by the acceptance suite and bitwise on
    real cascades only for the single-channel golden components. This
    test pins the remaining real class: every multi-channel, BW-free
    parent of the committed soft-decay factory (eta, omega, phi, the
    kaon pairs, the baryon pairs, the M_eff(450)/(850) towers, ...).
    """

    def test_all_legal_fast_path_matches_get_mode_on_factory_tables(self):
        chain_a = _build_parity_chain(seed=777)
        chain_b = _build_parity_chain(seed=777)
        targets = {
            pdg: pd_
            for pdg, pd_ in sorted(chain_a.items())
            if pd_.n_modes() > 1
            and not any(bw is not None for m in pd_._modes for bw in m.bws)
        }
        # The class must be non-empty in the committed factory, otherwise
        # the all-legal parity claim would hold vacuously on real tables.
        assert len(targets) >= 1
        n_draws = 1000
        for pdg, pd_a in targets.items():
            pd_b = chain_b[pdg]
            # Precondition: both builds produced the identical loaded
            # cumulative table, so any divergence below is a selection
            # divergence, not a construction one.
            assert pd_a._probabilities == pd_b._probabilities, pdg
            incs = pd_a.incremental_weights()
            mass = max(_mode_min_mass_sum(m) for m in pd_a._modes) + 1.0
            weights = pd_a.renormalized_mode_weights(mass, None)
            assert weights == incs, pdg
            for _ in range(n_draws):
                mode_a = pd_a.get_mode()
                mode_b, idx_b, p_cond, openness, w_b = (
                    pd_b.get_mode_conditional(mass, None)
                )
                assert tuple(mode_a.daughters) == tuple(mode_b.daughters), pdg
                assert w_b == incs, pdg
                assert idx_b == next(
                    i for i, m in enumerate(pd_b._modes) if m is mode_b
                ), pdg
                assert p_cond == incs[idx_b], pdg
                # BW-free tables: every window is a point window, so the
                # openness of a legal channel is exactly 1.0.
                assert openness == 1.0, pdg


def _build_parity_chain(seed: int) -> dict:
    """Deterministic soft-decay chain build (same recipe as the suite)."""
    rng = np.random.default_rng(seed)
    chain = build_soft_resonance_decays(rng=rng)
    _inject_antibaryon_resonance_decays(chain, rng, soft_decay_overrides={})
    return chain


# ---------------------------------------------------------------------------
# resolved_match_mass plumbing through the parallel generation paths
# ---------------------------------------------------------------------------


class TestResolvedMatchMassParallelPlumbing:
    def test_event_constructor_validates_resolved_match_mass(self):
        # Inactive default and a finite positive value are accepted.
        Event(rng=np.random.default_rng(1))
        Event(rng=np.random.default_rng(1), resolved_match_mass=None)
        Event(rng=np.random.default_rng(1), resolved_match_mass=2.0)
        for bad in (-1.0, 0.0, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                Event(rng=np.random.default_rng(1), resolved_match_mass=bad)

    def test_override_reaches_pa_generation_point_target(self):
        """The p+A chunk worker forwards the cap to its Event.

        At a point target (A = 1) the p+A path consumes the random
        stream in the same order as the pp path (pinned by
        test_pa_point_target_matches_pp_stream_exactly), so with the
        override active the p+A tracks must equal the pp tracks bit for
        bit, and both must differ from their inactive counterparts:
        otherwise the override either missed the p+A Event or changed
        nothing at all.
        """
        gen_file = str(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "na49_mixing",
                "generators",
                "michal",
                "nstar-to-delta",
                "n1520-zero.gen",
            )
        )
        base = dict(
            gen_file=gen_file,
            n_events=30,
            seed=97,
            beam_momentum=BEAM_MOMENTUM,
            max_decay_passes=8,
            fragmentation_mode="both",
            use_soft_decay_factory=True,
        )
        nucleus = NucleusSpec(name="p", A=1, Z=1)

        def _run(**extra):
            return generate_tracks_chunked(
                GenerationConfig(**base, **extra), n_workers=1, n_chunks=1
            )

        pp_inactive = _run()
        pp_active = _run(soft_decay_overrides=(("resolved_match_mass", 1.5),))
        pa_inactive = _run(nucleus=nucleus)
        pa_active = _run(
            nucleus=nucleus,
            soft_decay_overrides=(("resolved_match_mass", 1.5),),
        )
        for out in (pp_inactive, pp_active, pa_inactive, pa_active):
            assert out["n_decay_kinematic_failures"] == 0
        # Stream-equivalence precondition (no override): identical tracks.
        for key in ("pdg", "xf", "pt"):
            assert np.array_equal(pp_inactive[key], pa_inactive[key]), key
        # The override acts identically through the p+A worker's Event.
        for key in ("pdg", "xf", "pt"):
            assert np.array_equal(pp_active[key], pa_active[key]), key
        # The cap reaches the engine on the p+A path: same seed, the
        # active run realizes different tracks than the inactive one.
        same_tracks = all(
            np.array_equal(pa_inactive[key], pa_active[key])
            for key in ("pdg", "xf", "pt")
        )
        assert not same_tracks


# ---------------------------------------------------------------------------
# P2 review gaps: D5 pair family x C(M) interaction, threshold boundary,
# pi0 variant end to end, charged anti-tower mirror, chunk ancestry,
# parent-openness degradation (commit 5b2f37d).
# ---------------------------------------------------------------------------


def _build_production_chain(overrides: dict, seed: int = 2026) -> dict:
    """Full production-style chain (soft factory + anti-tower injection)."""
    rng = np.random.default_rng(seed)
    chain = build_soft_resonance_decays(
        rng=rng, branching_overrides=dict(overrides)
    )
    _inject_antibaryon_resonance_decays(
        chain, rng, soft_decay_overrides=dict(overrides)
    )
    return chain


def _nstarmeson_family_mass(dec: ParticleDecays, mstar_pdg: int) -> float:
    """Total incremental probability mass of modes carrying *mstar_pdg*."""
    total, prev = 0.0, 0.0
    for pr, m in zip(dec._probabilities, dec._modes):
        inc = pr - prev
        prev = pr
        if mstar_pdg in m.daughters:
            total += inc
    return total


class TestPairFamilyMixedEligibilityBudget:
    """D5 pair family applied on top of a partially closed pionic C(M).

    The shipped suite pins the all-pionic-eligible budget (test 17). The
    real risk is the mixed case: a pionic channel kinematically
    ineligible at the realized mass while the thresholded pair share is
    active. The explicit renormalization walk must still return a budget
    summing to exactly 1, with the surviving pionic weights scaled by
    ``(1 - P_ppbar(M))`` inside the global C(M) normalization and their
    declared ratios preserved. On the production tables all pionic
    thresholds sit below 2 m_p, so the mixed case is reached by raising
    the M_eff(450) daughter mass above the realized parent mass through
    the committed mass-override context.
    """

    @staticmethod
    def _family_table(pair_strength: float = 0.4, seed: int = 2029):
        chain = _build_production_chain(
            {MESTAR_PPBAR_STRENGTH_KEY: pair_strength}, seed=seed
        )
        return chain[9900213]

    def test_budget_sums_to_one_when_a_pionic_channel_is_ineligible(self):
        d2 = self._family_table()
        incs = d2.incremental_weights()
        keys = d2.channel_keys()
        i_pipi_charged = keys.index((211, -211))
        i_pipi_neutral = keys.index((111, 111))
        i_mestarpi = keys.index((9900113, 111))
        i_pair = keys.index((2212, -2212))
        share = d2.pair_family.share(2.2)
        with pdg_database.mass_width_overrides({9900113: (2.2, 0.18)}):
            # Threshold of the M_eff(450) + pi0 channel: 2.2 + 0.135
            # = 2.335 > 2.2, so the channel leaves C(M) while the pair
            # channel (2 m_p < 2.2) and both pipi channels stay legal.
            weights = d2.renormalized_mode_weights(2.2, None)
        assert weights[i_mestarpi] == 0.0
        pionic_declared = incs[i_pipi_charged] + incs[i_pipi_neutral]
        total = pionic_declared * (1.0 - share) + share
        expected = {
            i_pipi_charged: incs[i_pipi_charged] * (1.0 - share) / total,
            i_pipi_neutral: incs[i_pipi_neutral] * (1.0 - share) / total,
            i_mestarpi: 0.0,
            i_pair: share / total,
        }
        for i, expected_w in expected.items():
            assert weights[i] == pytest.approx(expected_w, abs=1e-12), keys[i]
        assert sum(weights) == pytest.approx(1.0, abs=1e-12)
        # The surviving pionic weights renormalize among themselves: their
        # declared ratio (2:1 for the (2/3, 1/3) pipi split) is preserved.
        assert weights[i_pipi_charged] / weights[i_pipi_neutral] == (
            pytest.approx(
                incs[i_pipi_charged] / incs[i_pipi_neutral], rel=1e-12
            )
        )

    def test_resolved_cap_below_pair_threshold_recovers_declared_vector(self):
        # A resolved cap below 2 m_p closes the pair channel through C(M)
        # even though the realized parent mass sits above the threshold
        # (reachable whenever resolved_match_mass < 2 m_p): the family
        # share evaporates in the renormalization and the declared
        # pionic vector is recovered exactly.
        d2 = self._family_table()
        incs = d2.incremental_weights()
        i_pair = d2.channel_keys().index((2212, -2212))
        weights = d2.renormalized_mode_weights(2.2, 1.0)
        assert weights[i_pair] == 0.0
        assert weights == pytest.approx(incs, abs=1e-12)
        assert sum(weights) == pytest.approx(1.0, abs=1e-12)

    def test_engine_path_mixed_case_records_only_surviving_channels(self):
        chain = _build_production_chain(
            {MESTAR_PPBAR_STRENGTH_KEY: 0.4}, seed=2029
        )
        d2 = chain[9900213]
        with pdg_database.mass_width_overrides({9900113: (2.2, 0.18)}):
            ev = Event(rng=np.random.default_rng(31))
            ev._decay_chain = {9900213: d2, 9900113: chain[9900113]}
            reaction = Reaction(BEAM_MOMENTUM)
            n_parents = 800
            for _ in range(n_parents):
                parent = Particle.from_id(0.0, 0.0, 0.5, 9900213, reaction)
                parent.set_mass(2.2)
                ev.add_particle(parent)
                ev.mark_primary(parent)
            ev.decay_cascade(max_passes=8)
            stats = ev.decay_stats()
        assert stats["n_decay_kinematic_failures"] == 0
        requested = stats["channel_requested"][9900213]
        realized = stats["channel_realized"][9900213]
        # The ineligible channel never enters the audit tables: the
        # requested accumulation skips its zero renormalized weight.
        assert set(requested) == {(211, -211), (111, 111), (2212, -2212)}
        assert (9900113, 111) not in realized
        # The requested means match the unit-level renormalized weights
        # (fixed parent mass, so no mass averaging).
        share = d2.pair_family.share(2.2)
        incs = d2.incremental_weights()
        keys = d2.channel_keys()
        pionic_declared = incs[0] + incs[1]
        total = pionic_declared * (1.0 - share) + share
        assert requested[(211, -211)] == pytest.approx(
            incs[0] * (1.0 - share) / total, abs=1e-9
        )
        assert requested[(111, 111)] == pytest.approx(
            incs[1] * (1.0 - share) / total, abs=1e-9
        )
        assert requested[(2212, -2212)] == pytest.approx(
            share / total, abs=1e-9
        )

    def test_family_table_skips_the_plain_fast_path(self):
        # Fast-path parity distinction (commit 5b2f37d): a table with an
        # active pair family must NOT reduce to the declared incremental
        # vector even when every declared channel is legal (the mass-
        # conditioned share is not in the declared cumulative), while a
        # plain table does exactly that (pinned bitwise on the real
        # factory tables by TestRealChainFastPathParity and on the M*
        # tables by TestMesonicOffDefaultRegression in the workflow
        # module).
        d2 = self._family_table()
        incs = d2.incremental_weights()
        i_pair = d2.channel_keys().index((2212, -2212))
        weights = d2.renormalized_mode_weights(2.2, None)
        assert weights != incs
        share = d2.pair_family.share(2.2)
        pionic_declared = sum(w for w in incs if w > 0.0)
        total = pionic_declared * (1.0 - share) + share
        for i, w in enumerate(incs):
            if w > 0.0:
                assert weights[i] == pytest.approx(
                    w * (1.0 - share) / total, abs=1e-12
                )
        assert weights[i_pair] == pytest.approx(share / total, abs=1e-12)


class TestPairThresholdBoundaryExact:
    """The pair threshold boundary at ``M = M_th`` exactly (card rev 2).

    The shipped suite pins the family law (share zero below ``M_th``,
    continuous with right limit zero). Here the boundary is pinned on
    the C(M) side: at ``M = M_th`` exactly the pair weight is the float
    ``0.0`` — a hard exclusion, not a merely small weight — and the
    selection never returns the pair channel.
    """

    def test_pair_weight_exactly_zero_and_never_selected_at_threshold(self):
        d2 = _build_production_chain(
            {MESTAR_PPBAR_STRENGTH_KEY: 0.6}, seed=2030
        )[9900213]
        keys = d2.channel_keys()
        i_pair = keys.index((2212, -2212))
        weights = d2.renormalized_mode_weights(_PAIR_THRESHOLD, None)
        assert weights[i_pair] == 0.0  # exact hard zero
        assert sum(weights) == pytest.approx(1.0, abs=1e-12)
        # Selection at the realized mass M_th: 500 seeded draws, none of
        # them the pair channel, every selected channel carrying a
        # positive renormalized probability.
        rng = np.random.default_rng(88)
        seen = set()
        for _ in range(500):
            mode, idx, p_cond, _openness, w = d2.get_mode_conditional(
                _PAIR_THRESHOLD, None
            )
            seen.add(keys[idx])
            assert p_cond > 0.0
            assert w[i_pair] == 0.0
        assert (2212, -2212) not in seen
        assert seen == {(211, -211), (111, 111), (9900113, 111)}

    def test_share_approaches_strength_from_below_far_above_threshold(self):
        # Asymptote of the recorded shape P(M) = c (1 - (M_th/M)^2)^n
        # with n = 1: strictly below c at any finite mass, within 2e-3
        # of c by M = 50 GeV.
        d2 = _build_production_chain(
            {MESTAR_PPBAR_STRENGTH_KEY: 0.6}, seed=2030
        )[9900213]
        family = d2.pair_family
        c = family.strength
        assert family.share(20.0) < c
        assert family.share(50.0) == pytest.approx(c, abs=2e-3)
        assert family.share(50.0) < c


class TestPi0VariantEndToEnd:
    """The recorded three-body ``p pbar pi0`` option, end to end.

    The shipped suite pins the registration, the default-off state and
    the unit-level share split. Here the variant is exercised through
    the engine at fixed realized parent masses: above its threshold
    (2 m_p + m_pi0 = 2.0115 GeV) the realized split between the two pair
    variants is 50/50; between the two-body and three-body thresholds
    the variant is absent from the realized channels while the two-body
    pair still fires; and its closure renormalizes through the global
    C(M) walk instead of distorting the budget.
    """

    @staticmethod
    def _variant_chain(pair_strength: float = 0.5, seed: int = 44):
        return _build_production_chain(
            {
                MESTAR_PPBAR_STRENGTH_KEY: pair_strength,
                MESTAR_PPBAR_PI0_VARIANT_KEY: 1.0,
            },
            seed=seed,
        )

    def test_closed_variant_renormalization_semantics(self):
        # Between 2 m_p and 2.0115 GeV the three-body member is
        # ineligible while the two-body member is legal: the closed
        # member's half of the family share drops out and the global
        # C(M) renormalization rescales the rest, so the two-body weight
        # is 0.5 * share / ((1 - share) + 0.5 * share) — the committed
        # reading of D1's renormalization-over-C(M) applied to the
        # adjusted family weights.
        d2 = self._variant_chain()[9900213]
        keys = d2.channel_keys()
        i_two = keys.index((2212, -2212))
        i_three = keys.index((2212, -2212, 111))
        weights = d2.renormalized_mode_weights(1.95, None)
        assert weights[i_three] == 0.0
        share = d2.pair_family.share(1.95)
        total = (1.0 - share) * 1.0 + 0.5 * share
        assert weights[i_two] == pytest.approx(0.5 * share / total, abs=1e-12)
        assert sum(weights) == pytest.approx(1.0, abs=1e-12)

    def test_variant_split_realized_end_to_end_above_variant_threshold(self):
        chain = self._variant_chain()
        ev = Event(rng=np.random.default_rng(77))
        ev._decay_chain = {9900213: chain[9900213], 9900113: chain[9900113]}
        reaction = Reaction(BEAM_MOMENTUM)
        n_parents = 3000
        for _ in range(n_parents):
            parent = Particle.from_id(0.0, 0.0, 0.5, 9900213, reaction)
            parent.set_mass(2.2)  # above 2.0115: both variants eligible
            ev.add_particle(parent)
            ev.mark_primary(parent)
        ev.decay_cascade(max_passes=8)
        stats = ev.decay_stats()
        assert stats["n_decay_kinematic_failures"] == 0
        realized = stats["channel_realized"][9900213]
        n_two = realized[(2212, -2212)] * n_parents
        n_three = realized[(2212, -2212, 111)] * n_parents
        assert n_two > 0
        assert n_three > 0
        # Fixed parent mass, both members legal: the realized split of
        # the family share between the variants is 50/50 within 3 sigma.
        p = 0.5
        n_pair = n_two + n_three
        sigma = math.sqrt(p * (1.0 - p) / n_pair)
        assert abs(n_three / n_pair - p) <= 3.0 * sigma

    def test_variant_ineligible_between_thresholds_end_to_end(self):
        chain = self._variant_chain()
        ev = Event(rng=np.random.default_rng(78))
        ev._decay_chain = {9900213: chain[9900213], 9900113: chain[9900113]}
        reaction = Reaction(BEAM_MOMENTUM)
        n_parents = 2000
        for _ in range(n_parents):
            parent = Particle.from_id(0.0, 0.0, 0.5, 9900213, reaction)
            parent.set_mass(1.95)  # 2 m_p < 1.95 < 2.0115
            ev.add_particle(parent)
            ev.mark_primary(parent)
        ev.decay_cascade(max_passes=8)
        stats = ev.decay_stats()
        assert stats["n_decay_kinematic_failures"] == 0
        realized = stats["channel_realized"][9900213]
        # The two-body pair still fires; the three-body variant is
        # excluded from C(M) by construction (never sampled, zero
        # kinematic failures) rather than retried.
        assert realized[(2212, -2212)] > 0.0
        assert (2212, -2212, 111) not in realized


class TestChargedAntitowerNstarmesonMirror:
    """Charged anti-tower mirror of the enabled N* -> N*' + M* link.

    The structural conjugation test
    (``test_charged_antitower_is_exact_charge_conjugate``) runs at the
    frozen defaults where ``br_nstarmeson = 0`` registers no mirror
    link, so the new ``[-9912201, 9900213]`` / ``[-9922201, 9900313]``
    branches need their own branch-by-branch conjugation check. The
    level-1 exception is pinned as intentional: the conjugate daughter
    of the matter channel [N*(1520)+, M_eff(450)] would be -2216, which
    the frozen PDG table does not define, so the charged level-1 partner
    stays terminal even with the level-1 link enabled.
    """

    def test_strength_split_wide_budget_clamps_pi_branch(self):
        """Round-2 wide budgets: cascade grows past the exhausted pi branch.

        Pre-registered in notes/resonance_fit_stage_progress.md: with
        ``budget`` given, s clamps into ``[0, budget]`` and the pi
        complement clamps at zero, while every s within ``[0, base]``
        reproduces the frozen-budget split exactly.
        """
        s, p = stage_b_strength_split(2, 0.20)
        assert s == pytest.approx(0.20)
        assert p == pytest.approx(0.10)
        assert stage_b_strength_split(3, 0.45)[1] == pytest.approx(0.00)
        assert stage_b_strength_split(1, 0.15)[1] == pytest.approx(0.00)
        wide2 = STAGE_B_STRENGTH_BUDGET_WIDE[2]
        assert wide2 == pytest.approx(0.60)
        s, p = stage_b_strength_split(2, 0.45, budget=wide2)
        assert s == pytest.approx(0.45)
        assert p == pytest.approx(0.00)
        s, p = stage_b_strength_split(2, 0.60, budget=wide2)
        assert s == pytest.approx(0.60)
        assert p == pytest.approx(0.00)
        wide3 = STAGE_B_STRENGTH_BUDGET_WIDE[3]
        assert wide3 == pytest.approx(0.90)
        s, p = stage_b_strength_split(3, 0.675, budget=wide3)
        assert s == pytest.approx(0.675)
        assert p == pytest.approx(0.00)
        s, p = stage_b_strength_split(3, 0.90, budget=wide3)
        assert s == pytest.approx(0.90)
        assert p == pytest.approx(0.00)
        s, p = stage_b_strength_split(3, 1.20, budget=wide3)
        assert s == pytest.approx(0.90)
        assert p == pytest.approx(0.00)

    def test_l2_l3_mirrors_conjugate_the_enabled_link(self):
        s2, nstarpi2 = stage_b_strength_split(2, 0.20)
        s3, nstarpi3 = stage_b_strength_split(3, 0.30)
        chain = _build_production_chain(
            {
                MESTAR_PPBAR_STRENGTH_KEY: 0.0,
                "nstar_eff_l2_br_nstarmeson": s2,
                "nstar_eff_l2_br_nstarpi": nstarpi2,
                "nstar_eff_l3_br_nstarmeson": s3,
                "nstar_eff_l3_br_nstarpi": nstarpi3,
            },
            seed=2027,
        )
        for level, br, matter_pid, charged_pid, mstar in (
            (2, s2, 9922201, -9922201, 9900213),
            (3, s3, 9932201, -9932201, 9900313),
        ):
            lower_matter = {2: 9912201, 3: 9922201}[level]
            lower_charged = {2: -9912201, 3: -9922201}[level]
            matter_keys = chain[matter_pid].channel_keys()
            charged_keys = chain[charged_pid].channel_keys()
            # Branch-by-branch conjugation of the new link: same
            # incremental weight on both towers, charge-conjugate lower
            # daughter, and the neutral M_eff state on both sides (no
            # charged M* variant exists in the table).
            assert (lower_matter, mstar) in matter_keys, matter_keys
            assert (lower_charged, mstar) in charged_keys, charged_keys
            assert _nstarmeson_family_mass(
                chain[matter_pid], mstar
            ) == pytest.approx(br, abs=1e-12)
            assert _nstarmeson_family_mass(
                chain[charged_pid], mstar
            ) == pytest.approx(br, abs=1e-12)
            # The D4 split keeps both branch sums at exactly 1, so the
            # towers stay unscaled and the link carries br directly.
            for mode in chain[charged_pid]._modes:
                if mstar in mode.daughters:
                    assert list(mode.daughters) == [lower_charged, mstar]
                    assert all(bw is None for bw in mode.bws)
            # Electric charge conjugation of the link.
            charge = pdg_database.get_charge
            assert (
                charge(lower_charged) + charge(mstar) == charge(charged_pid)
            )
            assert (
                charge(lower_matter) + charge(mstar) == charge(matter_pid)
            )
            assert charge(charged_pid) == -charge(matter_pid)

    def test_level1_charged_partner_stays_terminal_by_design(self):
        chain = _build_production_chain(
            {"nstar_eff_l1_br_nstarmeson": 0.10}, seed=2028
        )
        # Matter level 1 gains the link on both charge states.
        assert (2216, 9900113) in chain[9912201].channel_keys()
        assert (2116, 9900113) in chain[9911201].channel_keys()
        # The matter branch sum reaches 1.10 > 1, so the documented
        # proportional renormalization scales the link to 0.10 / 1.10.
        assert _nstarmeson_family_mass(
            chain[9912201], 9900113
        ) == pytest.approx(0.10 / 1.10, rel=1e-12)
        # The charged partner has no conjugate daughter (-2216 undefined
        # in the frozen table) and stays terminal: no M_eff daughter on
        # any charged anti-tower parent of this chain.
        for charged_pid in (-9912201, -9922201, -9932201):
            dec = chain[charged_pid]
            for mode in dec._modes:
                for daughter in mode.daughters:
                    assert daughter not in (9900113, 9900213, 9900313), (
                        charged_pid,
                        mode,
                    )
        assert _nstarmeson_family_mass(chain[-9912201], 9900113) == 0.0


class TestParallelChunkPairAncestry:
    """``Particle.parent`` ancestry bookkeeping in the chunk-worker path.

    The chunk track schema serializes no parent links; the ancestry
    stamping that runs inside every chunk worker's ``Event`` is
    observable through the serialized cascade depth and weak-feed
    flags. With the stage-B box open, the antiprotons of a matter
    projectile component can only come from the pair channel
    (``M_eff(850) -> p pbar`` at cascade depth 2), so the parallel chunk
    must carry them at exactly gen_depth 2 with weak_feed 0.
    """

    def test_pair_daughters_carry_cascade_depth_two_in_chunks(self):
        overrides = {
            MESTAR_PPBAR_STRENGTH_KEY: 0.5,
            "nstar_eff_l2_br_nstarmeson": 0.25,
            "nstar_eff_l2_br_nstarpi": 0.05,
            "nstar_eff_l2_mass": 4.6,
            "mestar_eff_l2_mass": 2.2,
            "nstar_eff_tower_max_mass": 5.0,
            "mestar_eff_tower_max_mass": 2.5,
        }
        cfg = GenerationConfig(
            gen_file=str(
                Path(__file__).resolve().parents[1]
                / "na49_mixing" / "generators" / "benchmark"
                / "nstar_eff2100_zero.gen"
            ),
            n_events=400,
            seed=314,
            beam_momentum=BEAM_MOMENTUM,
            max_decay_passes=8,
            fragmentation_mode="projectile",
            use_soft_decay_factory=True,
            soft_decay_overrides=tuple(overrides.items()),
        )
        result = generate_tracks_chunked(cfg, n_workers=2, n_chunks=2)
        assert result["n_decay_kinematic_failures"] == 0
        pdg = result["pdg"]
        depth = result["gen_depth"]
        weak = result["weak_feed"]
        n_pbar = int(np.count_nonzero(pdg == -2212))
        # Deterministic given the seed: 18 antiprotons at this
        # configuration (the pair channel is their only source on the
        # matter projectile side, per the two-knob smoke in the
        # workflow module).  15 before the three-body weight regained its
        # 1/s (2026-10-07), which changes the accept-reject draw count.
        assert n_pbar == 18
        assert np.all(depth[pdg == -2212] == 2)
        assert np.all(weak[pdg == -2212] == 0)


class TestParentOpennessAuditDegradation:
    """The parent-openness audit without the ``parent_line_shapes`` wiring.

    An ``Event`` constructed without ``parent_line_shapes`` (the default,
    and the state of every consumer that has not opted into the stage-B
    diagnostic) must degrade gracefully: the audit table stays absent
    for every parent type instead of recording zeros that would
    masquerade as openness data, while the remaining audit tables keep
    working.
    """

    def test_default_event_leaves_parent_openness_table_absent(self):
        chain = _build_production_chain({}, seed=55)
        ev = Event(rng=np.random.default_rng(79))
        ev._decay_chain = {9900213: chain[9900213], 9900113: chain[9900113]}
        reaction = Reaction(BEAM_MOMENTUM)
        n_parents = 50
        for _ in range(n_parents):
            parent = Particle.from_id(0.0, 0.0, 0.5, 9900213, reaction)
            parent.set_mass(1.2)
            ev.add_particle(parent)
            ev.mark_primary(parent)
        ev.decay_cascade(max_passes=8)
        stats = ev.decay_stats()
        # Absent, not zeros: decays happened, yet no parent-openness
        # entry appears anywhere. (n_decayed_parents also counts the
        # M_eff(450) cascade daughters of the mestarpi channel.)
        assert stats["n_decayed_parents"] >= n_parents
        assert stats["channel_realized"][9900213]
        assert stats["parent_openness_fraction"] == {}
        assert 9900213 not in stats["parent_openness_fraction"]
