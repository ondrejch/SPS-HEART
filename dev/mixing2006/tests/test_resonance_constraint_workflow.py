"""Engine-level acceptance tests for the source-constrained decay machinery.

Card TASK-2026-08-31-resonance-decay-fit-workflow. Phase P1 covers the
gap-analysis tests (i), (ii), (iii), (v), (viii) plus the inactive-path
regression against the Stage-0 golden fixtures (acceptance item 9), and
unit checks of the truncated, explicitly renormalized Breit-Wigner draw
helpers (zero-width delta case included). Phase P2 appends the mesonic
staging machinery: the rev-2 two-knob eligibility test (iv), the disabled
stage-zero ancestry check (vi), the enabled-stage traceable ancestry
(vii), the pair-channel peak-openness verification (15), the pair
threshold exclusion (16), the thresholded pair-weight arithmetic (17),
the Stage-B configuration constants, and the mesonic-off default
regression.
"""
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from na49_mixing import pdg_database
from na49_mixing.breit_wigner import (
    BreitWigner,
    _trapezoid_mass,
    bw_draw_window,
    bw_fraction_above,
    bw_new_array,
    bw_truncated_fraction,
    draw_truncated_bw,
)
from na49_mixing.decays import (
    DecayMode,
    MESTAR_PPBAR_PI0_VARIANT_KEY,
    MESTAR_PPBAR_STRENGTH_KEY,
    MASS_CONDITION_TOLERANCE,
    NoLegalChannelError,
    PPBAR_PAIR_EXPONENT,
    PairFamilyLaw,
    ParticleDecays,
    build_soft_resonance_decays,
    channel_parent_openness,
    conditional_mode_weight,
    daughter_minimum_mass,
    mode_mass_windows,
    ppbar_pair_threshold,
)
from na49_mixing.event import Event
from na49_mixing.generator import Generator
from na49_mixing.parallel import (
    GenerationConfig,
    STAGE_B_MESTAR_MASS_MAX_GEV,
    _inject_antibaryon_resonance_decays,
    _patch_generator_mass_width,
    _resolve_effective_tower_overrides,
    _resolved_match_mass_override,
    generate_tracks_chunked,
    stage_b_candidate_boxes,
    stage_b_mestar_mass_range,
    stage_b_strength_split,
)
from na49_mixing.particle import Particle
from na49_mixing.reaction import Reaction


BEAM_MOMENTUM = 158.0
_PAIR_THRESHOLD = ppbar_pair_threshold()


class _CountingRng:
    """RNG shim that counts every uniform draw (duck-typed Generator)."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self.n_calls = 0

    def random(self, size=None):
        self.n_calls += 1
        return self._rng.random(size)


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


def _build_soft_chain() -> dict:
    """The decay chain used by the benchmark resonance components."""
    rng = np.random.default_rng(12345)
    chain = build_soft_resonance_decays(rng=rng)
    _inject_antibaryon_resonance_decays(chain, rng, soft_decay_overrides={})
    return chain


_CHAIN_CACHE: dict = {}


def _soft_chain() -> dict:
    if "chain" not in _CHAIN_CACHE:
        _CHAIN_CACHE["chain"] = _build_soft_chain()
    return _CHAIN_CACHE["chain"]


def _mode_min_mass_sum(mode: DecayMode) -> float:
    return sum(
        daughter_minimum_mass(pid, bw)
        for pid, bw in zip(mode.daughters, mode.bws)
    )


def _add_parent(ev: Event, pdg_id: int, mass: float, pz: float = 3.0):
    reaction = Reaction(BEAM_MOMENTUM)
    parent = Particle.from_id(0.0, 0.0, pz, pdg_id, reaction)
    parent.set_mass(mass)
    ev.add_particle(parent)
    ev.mark_primary(parent)
    return parent


def _bw_slot_pdgs(chain: dict) -> set:
    """PDG ids whose decay table contains a Breit-Wigner daughter slot."""
    return {
        pdg
        for pdg, pdecays in chain.items()
        if any(bw is not None for m in pdecays._modes for bw in m.bws)
    }


def _component_bw_capable(spec_rel_path: str, chain: dict) -> bool:
    """True when the component's parent cascade can reach a BW-slot mode.

    The parent PDG is read from the gen card's first token; reachability
    follows the daughter links of the built decay chain, so the
    classification adapts to the factory instead of a hand-written map.
    """
    gen_file = Path(__file__).resolve().parents[1] / spec_rel_path
    parent_pdg = int(gen_file.read_text().split()[0])
    bw_pdgs = _bw_slot_pdgs(chain)
    seen: set = set()
    frontier = [parent_pdg]
    while frontier:
        pdg = frontier.pop()
        if pdg in seen or pdg not in chain:
            continue
        seen.add(pdg)
        if pdg in bw_pdgs:
            return True
        for mode in chain[pdg]._modes:
            frontier.extend(int(d) for d in mode.daughters)
    return False


def _component_bitwise_safe(spec_rel_path: str, chain: dict) -> bool:
    """True when the seeded run provably runs only untouched code paths.

    Every decay-table parent reachable from the component's primary must
    declare exactly one channel and no Breit-Wigner daughter slots. Then
    the mass-conditioned fast path always runs (bit-matched to get_mode),
    no BW daughter draw replaces a histogram draw, and the removed
    retry loop can never shift the random-number stream. Any multi-channel
    or BW-capable parent can in principle take the restricted selection
    path or draw a truncated mass, which keeps the realized channel
    distribution unchanged but moves the stream position.
    """
    gen_file = Path(__file__).resolve().parents[1] / spec_rel_path
    parent_pdg = int(gen_file.read_text().split()[0])
    bw_pdgs = _bw_slot_pdgs(chain)
    seen: set = set()
    frontier = [parent_pdg]
    while frontier:
        pdg = frontier.pop()
        if pdg in seen or pdg not in chain:
            continue
        seen.add(pdg)
        pdecays = chain[pdg]
        if pdecays.n_modes() != 1 or pdg in bw_pdgs:
            return False
        for mode in pdecays._modes:
            if any(bw is not None for bw in mode.bws):
                return False
            frontier.extend(int(d) for d in mode.daughters)
    return True


# ---------------------------------------------------------------------------
# Truncated line-shape helpers (breit_wigner.py additions)
# ---------------------------------------------------------------------------


class TestTruncatedBreitWignerHelpers:
    def test_bw_draw_window_delta_inside_and_outside(self):
        bw = BreitWigner(1.0765, 1.232, 0.0, 1.0, 1.6, 2.0)
        assert bw._delta_mass == pytest.approx(1.232)
        assert bw_draw_window(bw, 1.232) == (1.232, 1.232)
        assert bw_draw_window(bw, 2.0) == (1.232, 1.232)
        assert bw_draw_window(bw, 1.0) is None

    def test_bw_draw_window_full_support_and_empty(self):
        bw = BreitWigner(1.0765, 1.232, 0.117, 1.0, 1.6, 2.0)
        assert bw_draw_window(bw, 2.0) == (1.0765, 2.0)
        assert bw_draw_window(bw, 5.0) == (1.0765, 2.0)
        assert bw_draw_window(bw, 1.5) == (1.0765, 1.5)
        assert bw_draw_window(bw, 1.0765) is None
        assert bw_draw_window(bw, 0.5) is None

    def test_bw_truncated_fraction_delta_and_clipping(self):
        delta_bw = BreitWigner(1.0765, 1.232, 0.0, 1.0, 1.6, 2.0)
        assert bw_truncated_fraction(delta_bw, 1.0, 1.4) == 1.0
        assert bw_truncated_fraction(delta_bw, 1.3, 1.4) == 0.0
        bw = BreitWigner(1.0765, 1.232, 0.117, 1.0, 1.6, 2.0)
        assert bw_truncated_fraction(bw, 1.0765, 2.0) == pytest.approx(1.0)
        upper = bw_truncated_fraction(bw, 1.232, 2.0)
        assert 0.0 < upper < 1.0
        assert bw_truncated_fraction(bw, 1.232, 1.232) == 0.0

    def test_draw_truncated_bw_delta_consumes_no_randomness(self):
        bw = BreitWigner(1.0765, 1.232, 0.0, 1.0, 1.6, 2.0)
        counting = _CountingRng(3)
        assert draw_truncated_bw(bw, counting, 1.0, 1.4) == pytest.approx(1.232)
        assert draw_truncated_bw(bw, counting, 1.3, 1.4) is None
        assert counting.n_calls == 0

    def test_draw_truncated_bw_window_edges_and_support(self):
        bw = BreitWigner(1.0765, 1.232, 0.117, 1.0, 1.6, 2.0)
        counting = _CountingRng(7)
        for _ in range(200):
            m = draw_truncated_bw(bw, counting, 1.0765, 2.0)
            assert m is not None and 1.0765 <= m <= 2.0
        calls_before = counting.n_calls
        assert draw_truncated_bw(bw, counting, 1.232, 1.232) is None
        assert counting.n_calls == calls_before

    def test_draw_truncated_bw_consumes_two_uniforms(self):
        bw = BreitWigner(1.0765, 1.232, 0.117, 1.0, 1.6, 2.0)
        counting = _CountingRng(11)
        assert draw_truncated_bw(bw, counting, 1.0765, 2.0) is not None
        assert counting.n_calls == 2

    def test_draw_truncated_bw_follows_truncated_density(self):
        bw = BreitWigner(1.0765, 1.232, 0.117, 1.0, 1.6, 2.0)
        rng = np.random.default_rng(2026)
        lo, hi = 1.232, 1.9
        n = 4000
        draws = [
            m
            for m in (
                draw_truncated_bw(bw, rng, lo, hi) for _ in range(n)
            )
            if m is not None
        ]
        assert len(draws) == n
        xs = np.linspace(lo, hi, 2048)
        ys = np.maximum(bw_new_array(xs, bw.params), 0.0)
        cell = 0.5 * (ys[1:] + ys[:-1]) * (xs[1:] - xs[:-1])
        mid = 0.5 * (xs[1:] + xs[:-1])
        expected_mean = float((mid * cell).sum() / cell.sum())
        sigma_mean = float(np.std(draws)) / math.sqrt(n)
        assert abs(float(np.mean(draws)) - expected_mean) <= 4.0 * sigma_mean

    def test_full_support_integral_cache_resets_on_parameter_change(self):
        bw = BreitWigner(1.0765, 1.232, 0.117, 1.0, 1.6, 2.0)
        i1 = bw.full_support_integral()
        assert i1 > 0.0
        assert bw.full_support_integral() == i1
        # Narrowing the width sharpens the p-wave peak, which raises the
        # total integral over the fixed support; the cached value must
        # follow the new parameters rather than staying stale.
        bw.set_parameters(1.0765, 1.232, 0.001, 1.0, 1.6, 2.0)
        i2 = bw.full_support_integral()
        assert i2 > i1
        # Zero width: delta configuration, the ordinary density integral
        # is zero (the delta mass is handled separately).
        bw.set_parameters(1.0765, 1.232, 0.0, 1.0, 1.6, 2.0)
        assert bw.full_support_integral() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test (i) — resolved-state daughter mass within the realized parent mass
# ---------------------------------------------------------------------------


_ENGINE_PARENTS = (
    2224, 2214, 2114, 113, 213, -213, 313, 323, 221, 20113,
    102134, 9900113, 9900213, 9900313, 9911201,
)


class TestResolvedMassWithinSourceCap:
    def test_every_daughter_within_realized_parent_mass(self):
        chain = _soft_chain()
        for parent_pdg in _ENGINE_PARENTS:
            floor = min(
                _mode_min_mass_sum(m) for m in chain[parent_pdg]._modes
            )
            for mass in (floor + 0.05, floor + 0.3, floor + 0.8, floor + 1.6):
                for seed in (1, 2):
                    ev = _RecordingEvent(rng=np.random.default_rng(seed))
                    ev._decay_chain = chain
                    _add_parent(ev, parent_pdg, mass)
                    ev.decay_cascade(max_passes=8)
                    for m_parent, _mode, daughters in ev.decay_records:
                        for d in daughters:
                            assert d.get_mass() <= (
                                m_parent + MASS_CONDITION_TOLERANCE
                            ), (parent_pdg, mass, d.get_id(), d.get_mass())


# ---------------------------------------------------------------------------
# Test (ii) — resolved_match_mass cap active / inactive reduction
# ---------------------------------------------------------------------------


class TestResolvedMassWithinMatchCap:
    def test_cap_active_bounds_drawn_resolved_masses(self):
        chain = _soft_chain()
        cap = 1.20
        for parent_pdg in (9900113, 9900213, 9900313, 20113, 102134, 9911201):
            for mass in (1.6, 2.4, 3.2):
                ev = _RecordingEvent(
                    rng=np.random.default_rng(5), resolved_match_mass=cap
                )
                ev._decay_chain = chain
                _add_parent(ev, parent_pdg, mass)
                ev.decay_cascade(max_passes=8)
                for m_parent, mode, daughters in ev.decay_records:
                    bound = min(m_parent, cap)
                    for (pid, bw), d in zip(
                        zip(mode.daughters, mode.bws), daughters
                    ):
                        if bw is not None:
                            assert d.get_mass() <= (
                                bound + MASS_CONDITION_TOLERANCE
                            ), (parent_pdg, mass, pid, d.get_mass())

    def test_cap_inactive_reduces_to_kinematic_bound(self):
        chain = _soft_chain()
        for parent_pdg in (9900113, 9900213, 9900313, 20113, 102134):
            pdecays = chain[parent_pdg]
            for mode in pdecays._modes:
                for mass in (1.6, 2.4):
                    assert mode_mass_windows(mode, mass, None) == (
                        mode_mass_windows(mode, mass, mass)
                    )
                    for loose_cap in (10.0, 100.0):
                        assert mode_mass_windows(mode, mass, loose_cap) == (
                            mode_mass_windows(mode, mass, None)
                        )

    def test_override_resolution_unit(self):
        gen_file = (
            Path(__file__).resolve().parents[1]
            / "na49_mixing" / "generators" / "michal"
            / "deltas-to-pion" / "delta++.gen"
        )
        cfg_off = GenerationConfig(gen_file=str(gen_file), n_events=1, seed=1)
        assert _resolved_match_mass_override(cfg_off) is None
        cfg_on = GenerationConfig(
            gen_file=str(gen_file),
            n_events=1,
            seed=1,
            soft_decay_overrides=(("resolved_match_mass", 2.0),),
        )
        assert _resolved_match_mass_override(cfg_on) == pytest.approx(2.0)
        cfg_bad = GenerationConfig(
            gen_file=str(gen_file),
            n_events=1,
            seed=1,
            soft_decay_overrides=(("resolved_match_mass", -1.0),),
        )
        with pytest.raises(ValueError):
            _resolved_match_mass_override(cfg_bad)

    def test_override_reaches_event_level_generation(self):
        gen_file = (
            Path(__file__).resolve().parents[1]
            / "na49_mixing" / "generators" / "michal"
            / "nstar-to-delta" / "n1520-zero.gen"
        )
        base = dict(
            gen_file=str(gen_file),
            n_events=40,
            seed=97,
            beam_momentum=BEAM_MOMENTUM,
            max_decay_passes=8,
            fragmentation_mode="projectile",
            use_soft_decay_factory=True,
            inject_antibaryon_resonances=True,
        )
        inactive = generate_tracks_chunked(
            GenerationConfig(**base), n_workers=1, n_chunks=1
        )
        active = generate_tracks_chunked(
            GenerationConfig(
                **base, soft_decay_overrides=(("resolved_match_mass", 1.5),)
            ),
            n_workers=1,
            n_chunks=1,
        )
        assert inactive["n_decay_kinematic_failures"] == 0
        assert active["n_decay_kinematic_failures"] == 0
        # The cap reaches the engine: same seed, different realized tracks.
        same_tracks = (
            np.array_equal(inactive["pdg"], active["pdg"])
            and np.array_equal(inactive["xf"], active["xf"])
            and np.array_equal(inactive["pt"], active["pt"])
        )
        assert not same_tracks


# ---------------------------------------------------------------------------
# Test (iii) — enabled channels legal at the realized parent mass
# ---------------------------------------------------------------------------


class TestEnabledChannelsLegalAtRealizedParentMass:
    def test_mass_grid_decay_cascade_legal_and_failure_free(self):
        chain = _soft_chain()
        for parent_pdg in _ENGINE_PARENTS:
            pdecays = chain[parent_pdg]
            floor = min(_mode_min_mass_sum(m) for m in pdecays._modes)
            for mass in (
                floor + 0.05, floor + 0.3, floor + 0.8, floor + 1.6, floor + 2.4
            ):
                assert any(
                    mode_mass_windows(m, mass, None) is not None
                    for m in pdecays._modes
                ), (parent_pdg, mass)
                ev = _RecordingEvent(rng=np.random.default_rng(17))
                ev._decay_chain = chain
                _add_parent(ev, parent_pdg, mass)
                ev.decay_cascade(max_passes=8)
                stats = ev.decay_stats()
                assert stats["n_decay_kinematic_failures"] == 0, (
                    parent_pdg, mass, stats
                )
                for m_parent, mode, _daughters in ev.decay_records:
                    assert _mode_min_mass_sum(mode) <= (
                        m_parent + MASS_CONDITION_TOLERANCE
                    ), (parent_pdg, mass, mode.daughters)


# ---------------------------------------------------------------------------
# Test (v) — realized branching fractions match the renormalized request
# ---------------------------------------------------------------------------


def _synthetic_meff_chain() -> dict:
    """Synthetic M_eff(850) table: pionic + cascade + thresholded pair."""
    rng = np.random.default_rng(21)
    d = ParticleDecays(9900213, rng=rng)
    d.add_mode(0.5, DecayMode(2, [211, -211]))      # pi+ pi-
    d.add_mode(0.8, DecayMode(2, [9900113, 111]))   # M_eff(450) + pi0
    d.add_mode(1.0, DecayMode(2, [2212, -2212]))    # p pbar pair channel
    return {9900213: d}


class TestRealizedBranchingFractionsMatchRequest:
    def test_selector_realized_fractions_match_request_all_legal(self):
        rng = np.random.default_rng(21)
        pdecays = _synthetic_meff_chain()[9900213]
        weights = pdecays.renormalized_mode_weights(2.0, None)
        assert weights == pytest.approx([0.5, 0.3, 0.2])
        n = 20_000
        counts = np.zeros(3)
        for _ in range(n):
            mode, _idx, _p_cond, _openness, _w = pdecays.get_mode_conditional(2.0, None)
            idx = next(i for i, m in enumerate(pdecays._modes) if m is mode)
            counts[idx] += 1
        for k, p in enumerate(weights):
            sigma = math.sqrt(p * (1.0 - p) / n)
            assert abs(counts[k] / n - p) <= 3.0 * sigma, (k, counts[k] / n, p)

    def test_selector_realized_fractions_match_request_restricted(self):
        pdecays = _synthetic_meff_chain()[9900213]
        weights = pdecays.renormalized_mode_weights(1.5, None)
        assert weights[2] == 0.0
        assert weights[0] == pytest.approx(0.5 / 0.8)
        assert weights[1] == pytest.approx(0.3 / 0.8)
        n = 20_000
        counts = np.zeros(3)
        for _ in range(n):
            mode, _idx, _p_cond, _openness, _w = pdecays.get_mode_conditional(1.5, None)
            idx = next(i for i, m in enumerate(pdecays._modes) if m is mode)
            counts[idx] += 1
        assert counts[2] == 0
        for k in (0, 1):
            p = weights[k]
            sigma = math.sqrt(p * (1.0 - p) / n)
            assert abs(counts[k] / n - p) <= 3.0 * sigma, (k, counts[k] / n, p)

    def test_engine_audit_requested_matches_realized(self):
        chain = _synthetic_meff_chain()
        ev = Event(rng=np.random.default_rng(24))
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
        realized = stats["channel_realized"][9900213]
        requested = stats["channel_requested"][9900213]
        assert set(realized) == {(211, -211), (9900113, 111), (2212, -2212)}
        n = float(n_parents)
        for key, p in (
            ((211, -211), 0.5),
            ((9900113, 111), 0.3),
            ((2212, -2212), 0.2),
        ):
            sigma = math.sqrt(p * (1.0 - p) / n)
            assert abs(realized[key] - p) <= 3.0 * sigma, (key, realized[key])
            assert requested[key] == pytest.approx(p, abs=1e-9)
        assert sum(requested.values()) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Test (viii) — illegal channel excluded from C(M), retry not exercised
# ---------------------------------------------------------------------------


def _illegal_pair_table(seed: int) -> ParticleDecays:
    d = ParticleDecays(9900113, rng=np.random.default_rng(seed))
    d.add_mode(0.6, DecayMode(2, [211, -211]))      # legal at 1.0
    d.add_mode(1.0, DecayMode(2, [2212, -2212]))    # illegal at 1.0
    return d


class TestNoRetryRenormalizationOfIllegalChannel:
    def test_illegal_channel_excluded_and_weight_renormalized(self):
        pdecays = _illegal_pair_table(31)
        weights = pdecays.renormalized_mode_weights(1.0, None)
        assert weights[0] == pytest.approx(1.0)
        assert weights[1] == 0.0
        assert conditional_mode_weight(
            pdecays._modes[1], 0.4, 1.0, None
        ) == 0.0
        counting = _CountingRng(32)
        pdecays._rng = counting  # type: ignore[assignment]
        n = 5000
        for _ in range(n):
            mode, _idx, p_cond, _openness, _w = pdecays.get_mode_conditional(1.0, None)
            assert tuple(mode.daughters) == (211, -211)
            assert p_cond == pytest.approx(1.0)
        assert counting.n_calls == n

    def test_fast_path_matches_get_mode_bitwise_when_all_legal(self):
        d_a = ParticleDecays(2224, rng=np.random.default_rng(41))
        d_a.add_mode(0.5, DecayMode(2, [2212, 211]))
        d_a.add_mode(1.0, DecayMode(2, [2212, -211]))
        d_b = ParticleDecays(2224, rng=np.random.default_rng(41))
        d_b.add_mode(0.5, DecayMode(2, [2212, 211]))
        d_b.add_mode(1.0, DecayMode(2, [2212, -211]))
        picked_a, picked_b = [], []
        for _ in range(2000):
            mode_a = d_a.get_mode()
            mode_b, _i, _p, _o, _w = d_b.get_mode_conditional(2.0, None)
            picked_a.append(tuple(mode_a.daughters))
            picked_b.append(tuple(mode_b.daughters))
        assert picked_a == picked_b

    def test_event_level_retry_path_not_exercised(self):
        pdecays = _illegal_pair_table(51)
        ev = Event(rng=np.random.default_rng(52))
        ev._decay_chain = {9900113: pdecays}
        reaction = Reaction(BEAM_MOMENTUM)
        n_parents = 300
        for _ in range(n_parents):
            parent = Particle.from_id(0.0, 0.0, 0.5, 9900113, reaction)
            parent.set_mass(1.0)
            ev.add_particle(parent)
            ev.mark_primary(parent)
        ev.decay_cascade(max_passes=8)
        stats = ev.decay_stats()
        assert stats["n_decay_attempts"] == n_parents
        assert stats["n_decay_kinematic_failures"] == 0
        realized = stats["channel_realized"][9900113]
        requested = stats["channel_requested"][9900113]
        assert (2212, -2212) not in realized
        assert realized[(211, -211)] == pytest.approx(1.0)
        assert requested[(211, -211)] == pytest.approx(1.0)

    def test_empty_channel_set_raises_no_legal_channel(self):
        pdecays = _illegal_pair_table(61)
        # Mass 0.2 closes every channel (pi+pi- threshold 0.2793, pair
        # threshold 1.8765): C(M) is empty.
        with pytest.raises(NoLegalChannelError):
            pdecays.get_mode_conditional(0.2, None)
        broken = ParticleDecays(9900113, rng=np.random.default_rng(62))
        broken.add_mode(0.4, DecayMode(2, [211, -211]))
        with pytest.raises(ValueError) as excinfo:
            broken.get_mode_conditional(2.0, None)
        assert not isinstance(excinfo.value, NoLegalChannelError)


# ---------------------------------------------------------------------------
# Acceptance item 9 — inactive-path regression against the golden fixtures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("species_name,particle_pdg", [
    ("pi_minus", -211),
    ("pi_plus", 211),
])
def test_inactive_path_matches_golden_fixtures(species_name, particle_pdg):
    """Seeded generation with mesonic branches zero and no caps.

    Components whose reachable cascade provably runs only untouched code
    paths (single-channel, BW-free parents) reproduce the Stage-0 fixtures
    bitwise. All other components are bounded by the pre-registered
    3-sigma Poisson band on the fixture counts. Two documented deviation
    causes exist, both stream-level only (card risk note "Bitwise
    regression feasibility"; the realized channel distributions are
    unchanged because the old retry loop's accepted-channel distribution
    already equaled the C(M) renormalization):
    1. the truncated, explicitly renormalized BW daughter draw replaces
       the 500-bin histogram draw, and
    2. the removed sample-then-retry loop no longer consumes the extra
       uniforms of re-sampled illegal channels.
    The tolerance is never loosened post hoc.
    """
    fixture_path = (
        Path(__file__).resolve().parent
        / "fixtures" / f"golden_{species_name}.json"
    )
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    base_dir = Path(__file__).resolve().parents[1]
    chain = _soft_chain()
    for row in payload["components"]:
        label, rel_path = row["spec_key"].split("::", 1)
        cfg = GenerationConfig(
            gen_file=str(base_dir / rel_path),
            n_events=int(row["n_events"]),
            seed=int(row["component_seed"]),
            beam_momentum=BEAM_MOMENTUM,
            max_decay_passes=int(payload["max_decay_passes"]),
            fragmentation_mode=str(payload["fragmentation_mode"]),
            use_soft_decay_factory=True,
            inject_antibaryon_resonances=bool(
                payload["inject_antibaryon_resonances"]
            ),
        )
        tracks = generate_tracks_chunked(cfg, n_workers=1, n_chunks=1)
        n_now = int(np.count_nonzero(tracks["pdg"] == particle_pdg))
        n_fix = int(row["n_tracks_species"])
        assert tracks["n_decay_kinematic_failures"] == 0, label
        if _component_bitwise_safe(rel_path, chain):
            assert n_now == n_fix, (label, n_fix, n_now)
        else:
            band = 3.0 * math.sqrt(max(n_fix, 1.0))
            assert abs(n_now - n_fix) <= band, (label, n_fix, n_now)


# ---------------------------------------------------------------------------
# Phase P2 — mesonic staging machinery (Stage B, mesonic still off by
# default). Card TASK-2026-08-31 rev 2: D5 pair family, D4 enablement,
# pre-registered candidate boxes, ancestry traceability, parent-side
# openness.
# ---------------------------------------------------------------------------

_NSTAR_LEVEL_IDS = {1: 9911201, 2: 9921201, 3: 9931201}
_MSTAR_LEVEL_IDS = {1: 9900113, 2: 9900213, 3: 9900313}
_MEFF_PDGS = frozenset(_MSTAR_LEVEL_IDS.values())

# Box-consistent representative fitted masses per candidate (inside the
# pre-registered boxes) and the tight box corners, used by the
# peak-openness test (15).
_STAGE_B_FIT_MASSES = {
    2: {"nstar_eff_l2_mass": 4.6, "mestar_eff_l2_mass": 2.2},
    3: {"nstar_eff_l3_mass": 4.8, "mestar_eff_l3_mass": 2.2},
}
_STAGE_B_EDGE_MASSES = {
    2: {"nstar_eff_l2_mass": 4.5, "mestar_eff_l2_mass": 2.5},
    3: {"nstar_eff_l3_mass": 4.7, "mestar_eff_l3_mass": 2.5},
}


def _stage_b_overrides(
    level: int,
    *,
    strength: float = 0.25,
    pair_strength: float = 0.5,
    enable_parent: bool = True,
    pi0_variant: bool = False,
    mass_overrides: bool = True,
) -> dict:
    """Stage-B soft-override set for one candidate level.

    The D4 parent strength enters through the compensating split of
    :func:`stage_b_strength_split` (s in [0, base_k]); the D5 pair family
    through the aggregate strength key. Mass/width keys use the
    box-consistent representative values so every enabled channel is
    peak-open by construction.
    """
    ovr: dict = {MESTAR_PPBAR_STRENGTH_KEY: pair_strength}
    if pi0_variant:
        ovr[MESTAR_PPBAR_PI0_VARIANT_KEY] = 1.0
    if enable_parent:
        s, nstarpi = stage_b_strength_split(level, strength)
        ovr[f"nstar_eff_l{level}_br_nstarmeson"] = s
        ovr[f"nstar_eff_l{level}_br_nstarpi"] = nstarpi
    if mass_overrides:
        ovr.update(_STAGE_B_FIT_MASSES[level])
    return ovr


def _build_chain(overrides: dict, seed: int = 2026) -> dict:
    """Full production-style chain (soft factory + anti-tower injection)."""
    rng = np.random.default_rng(seed)
    chain = build_soft_resonance_decays(rng=rng, branching_overrides=dict(overrides))
    _inject_antibaryon_resonance_decays(
        chain, rng, soft_decay_overrides=dict(overrides)
    )
    return chain


def _ancestor_id_chain(particle: Particle) -> list:
    """PDG ids from the direct parent upward, following ``Particle.parent``."""
    ids = []
    cur = particle.parent
    while cur is not None:
        ids.append(int(cur.get_id()))
        cur = cur.parent
    return ids


def _component_primary_pdgs() -> list:
    """Primary PDG ids of every golden-fixture component (production roots)."""
    base = Path(__file__).resolve().parents[1]
    fixture_dir = Path(__file__).resolve().parent / "fixtures"
    roots = set()
    for species in ("pi_minus", "pi_plus"):
        payload = json.loads(
            (fixture_dir / f"golden_{species}.json").read_text(encoding="utf-8")
        )
        for row in payload["components"]:
            rel_path = row["spec_key"].split("::", 1)[1]
            roots.add(int((base / rel_path).read_text().split()[0]))
    return sorted(roots)


class TestGeneratorMassWidthSupportAtBoxCorners:
    """BW support verification for grown masses (card rev 2, phase P2).

    The stage-B box moves tower masses up to 2.5 GeV (mesons) and 4.7 GeV
    (nucleons). ``_patch_generator_mass_width`` must keep the pole inside
    the sampling support at those corners so the truncated draw helpers
    never clip the box.
    """

    @staticmethod
    def _patch(card_name: str, mass: float, width: float) -> tuple:
        card = (
            Path(__file__).resolve().parents[1]
            / "na49_mixing" / "generators" / "benchmark" / card_name
        )
        gen = Generator(reaction=Reaction(BEAM_MOMENTUM))
        gen.set_all(str(card))
        _patch_generator_mass_width(gen, {int(gen._id): (mass, width)})
        return tuple(gen._bw.params)

    def test_nucleon_level_at_4p7_keeps_pole_inside_support(self):
        # nstar_eff2300 card: schwel 1.22, rmas 2.30, brei 0.35,
        # dumplow 2.9, dumphigh 3.6. The B-1250 corner moves the peak to
        # 4.7 GeV: the damping window stretches above it.
        params = self._patch("nstar_eff2300_zero.gen", 4.7, 0.35)
        schwel, rmas, brei, _pbrei, dumplow, dumphigh = params
        assert rmas == pytest.approx(4.7)
        assert brei == pytest.approx(0.35)
        assert dumphigh > 4.7
        assert dumplow < dumphigh
        # The committed parameters stay valid and sampleable.
        bw = BreitWigner(*params)
        assert bw.full_support_integral() > 0.0
        drawn = bw.get_random_n(1000)
        assert float(drawn.min()) >= schwel
        assert float(drawn.max()) <= dumphigh

    def test_meson_level_at_2p5_keeps_pole_inside_support(self):
        # A meson-tower-style card body with the same mass law encoding:
        # the rho3 card (schwel 0.278, rmas 1.689, brei 0.161, dumplow
        # 2.0, dumphigh 2.5) patched to the box corner 2.5 GeV.
        params = self._patch("rho3_zero_1690.gen", 2.5, 0.28)
        schwel, rmas, _brei, _pbrei, dumplow, dumphigh = params
        assert rmas == pytest.approx(2.5)
        assert dumphigh > 2.5
        assert dumplow < dumphigh
        bw = BreitWigner(*params)
        assert bw.full_support_integral() > 0.0
        drawn = bw.get_random_n(1000)
        assert float(drawn.min()) >= schwel
        assert float(drawn.max()) <= dumphigh

    def test_b450_corner_below_the_old_damping_window(self):
        # B-450 moves the level-1 peak to 2.07 GeV, below the
        # nstar_eff1900 card's damping window start (2.5): the window is
        # kept and already covers the peak.
        params = self._patch("nstar_eff1900_zero.gen", 2.07, 0.25)
        schwel, rmas, _brei, _pbrei, dumplow, dumphigh = params
        assert rmas == pytest.approx(2.07)
        assert dumplow == pytest.approx(2.5)
        assert dumphigh == pytest.approx(3.1)
        bw = BreitWigner(*params)
        assert bw.full_support_integral() > 0.0


class TestMesonicOffDefaultRegression:
    """Default configuration behaves identically to the P1 frozen state."""

    def test_default_factory_registers_no_pair_channels_and_no_links(self):
        chain = build_soft_resonance_decays(rng=np.random.default_rng(11))
        for mstar in (9900113, 9900213, 9900313):
            d = chain[mstar]
            assert d.pair_family is None
            assert (2212, -2212) not in d.channel_keys()
            assert (2212, -2212, 111) not in d.channel_keys()
        # No N* tower parent declares an nstarmeson (N*' + M_eff) channel
        # at the frozen defaults, and no charged anti-tower partner does.
        for nstar_ids, mstar_id in zip(
            ((9911201, 9912201), (9921201, 9922201), (9931201, 9932201)),
            (9900113, 9900213, 9900313),
        ):
            for nstar in nstar_ids:
                for mode in chain[nstar]._modes:
                    assert mstar_id not in mode.daughters, (nstar, mode)

    def test_default_selection_parity_untouched_on_mstar_tables(self):
        # The M* tables stay plain: the all-legal fast path still walks
        # the loaded cumulative floats, bit for bit (P1 frozen state).
        chain_a = build_soft_resonance_decays(rng=np.random.default_rng(12))
        chain_b = build_soft_resonance_decays(rng=np.random.default_rng(12))
        for mstar in (9900213, 9900313):
            d_a, d_b = chain_a[mstar], chain_b[mstar]
            assert d_a._probabilities == d_b._probabilities
            mass = max(
                sum(daughter_minimum_mass(pid, bw)
                    for pid, bw in zip(m.daughters, m.bws))
                for m in d_a._modes
            ) + 0.5
            assert d_a.renormalized_mode_weights(mass, None) == (
                d_a.incremental_weights()
            )
            for _ in range(500):
                mode_a = d_a.get_mode()
                mode_b, _i, _p, _o, w_b = d_b.get_mode_conditional(mass, None)
                assert tuple(mode_a.daughters) == tuple(mode_b.daughters)
                assert w_b == d_b.incremental_weights()


class TestStageBConfiguration:
    """Pre-registered candidate boxes and window widening as configuration."""

    def test_pair_threshold_and_box_bounds(self):
        assert _PAIR_THRESHOLD == pytest.approx(2.0 * 0.93827231)
        lo, hi = stage_b_mestar_mass_range()
        assert lo == pytest.approx(_PAIR_THRESHOLD + 0.005)
        assert hi == pytest.approx(2.5)
        assert STAGE_B_MESTAR_MASS_MAX_GEV == pytest.approx(2.5)

    def test_candidate_boxes_follow_the_bound_chain(self):
        boxes = stage_b_candidate_boxes()
        assert set(boxes) == {1, 2, 3}
        # B-450: meson-cascade only, m_N*(1) >= 1.520 + 0.450 + 0.10.
        assert boxes[1]["pair_capable"] is False
        assert boxes[1]["mestar_mass_range"] == pytest.approx((0.450, 0.450))
        assert boxes[1]["nstar_mass_range"][0] == pytest.approx(2.07)
        # B-850 and B-1250: pair-capable with the [2 m_p + eps, 2.5] box.
        for level, lower_parent in ((2, 1.900), (3, 2.100)):
            assert boxes[level]["pair_capable"] is True
            lo, hi = boxes[level]["mestar_mass_range"]
            assert lo == pytest.approx(_PAIR_THRESHOLD + 0.005)
            assert hi == pytest.approx(2.5)
            assert boxes[level]["nstar_mass_range"][0] == pytest.approx(
                lower_parent + 2.5 + 0.10
            )
            assert boxes[level]["nstar_mass_range"][0] == pytest.approx(
                4.50 if level == 2 else 4.70
            )
        # Window widening recorded once per candidate.
        for level in (1, 2, 3):
            window = boxes[level]["window_overrides"]
            assert window["nstar_eff_tower_max_mass"] == pytest.approx(5.0)
            assert window["mestar_eff_tower_max_mass"] == pytest.approx(2.5)
            keys = boxes[level]["strength_keys"]
            assert keys["parent"] == f"nstar_eff_l{level}_br_nstarmeson"
            assert keys["pair"] == MESTAR_PPBAR_STRENGTH_KEY
            assert boxes[level]["pair_exponent"] == pytest.approx(
                PPBAR_PAIR_EXPONENT
            )

    def test_d4_strength_split_compensates_within_the_level_budget(self):
        # s inside [0, base_k]: compensating br_nstarpi = base_k - s.
        assert stage_b_strength_split(2, 0.20) == pytest.approx((0.20, 0.10))
        assert stage_b_strength_split(1, 0.0) == pytest.approx((0.0, 0.15))
        assert stage_b_strength_split(3, 0.45) == pytest.approx((0.45, 0.0))
        # Clamping: s below 0 and above base stays inside the budget.
        assert stage_b_strength_split(2, -0.5) == pytest.approx((0.0, 0.30))
        assert stage_b_strength_split(2, 1.0) == pytest.approx((0.30, 0.0))
        # Explicit Stage-A frozen base overrides the default.
        assert stage_b_strength_split(2, 0.10, base_nstarpi=0.25) == (
            pytest.approx((0.10, 0.15))
        )


class TestPairFamilyWeightArithmetic:
    """Test (17): thresholded pair weight, continuity, per-level budget."""

    def test_family_registered_on_pair_capable_levels_only(self):
        chain = _build_chain(_stage_b_overrides(2, pair_strength=0.4))
        assert isinstance(chain[9900213].pair_family, PairFamilyLaw)
        assert isinstance(chain[9900313].pair_family, PairFamilyLaw)
        assert chain[9900113].pair_family is None
        for family in (chain[9900213].pair_family, chain[9900313].pair_family):
            assert family.strength == pytest.approx(0.4)
            assert family.threshold == pytest.approx(_PAIR_THRESHOLD)
            assert family.exponent == pytest.approx(PPBAR_PAIR_EXPONENT)

    def test_share_exactly_zero_below_threshold_and_continuous_at_it(self):
        family = _build_chain(_stage_b_overrides(2))[9900213].pair_family
        assert family.share(0.5) == 0.0
        assert family.share(_PAIR_THRESHOLD - 0.01) == 0.0
        # At the threshold: hard zero; right limit zero (continuity).
        assert family.share(_PAIR_THRESHOLD) == 0.0
        assert family.share(_PAIR_THRESHOLD + 1e-9) == pytest.approx(
            0.0, abs=1e-8
        )
        # Fixed thresholded shape: P(M) = c * (1 - (M_th/M)^2)^n, growing.
        c = family.strength
        assert family.share(2.5) == pytest.approx(
            c * (1.0 - (_PAIR_THRESHOLD / 2.5) ** 2) ** PPBAR_PAIR_EXPONENT
        )
        assert 0.0 < family.share(2.0) < family.share(2.2) < family.share(2.5)

    def test_per_level_budget_sums_to_one_at_sampled_realized_masses(self):
        chain = _build_chain(_stage_b_overrides(2, pair_strength=0.4))
        d2 = chain[9900213]
        incs = d2.incremental_weights()
        pair_idx = d2.channel_keys().index((2212, -2212))
        rng = np.random.default_rng(77)
        masses = np.concatenate([
            rng.uniform(0.5, _PAIR_THRESHOLD, size=20),
            rng.uniform(_PAIR_THRESHOLD + 0.005, 2.6, size=40),
        ])
        n_checked_above = 0
        for mass in masses:
            weights = d2.renormalized_mode_weights(float(mass), None)
            assert sum(weights) == pytest.approx(1.0, abs=1e-12), mass
            if mass > 0.6:  # every pionic channel eligible above 0.585+eps
                share = d2.pair_family.share(float(mass))
                for i, w in enumerate(incs[:pair_idx]):
                    if w > 0.0:
                        assert weights[i] == pytest.approx(
                            w * (1.0 - share), abs=1e-12
                        ), (mass, i)
                assert weights[pair_idx] == pytest.approx(
                    share, abs=1e-12
                ), mass
                if mass > _PAIR_THRESHOLD:
                    n_checked_above += 1
        assert n_checked_above > 10

    def test_pionic_weights_untouched_below_threshold(self):
        chain = _build_chain(_stage_b_overrides(2, pair_strength=0.6))
        d2 = chain[9900213]
        incs = d2.incremental_weights()
        weights = d2.renormalized_mode_weights(_PAIR_THRESHOLD - 0.05, None)
        assert weights == pytest.approx(incs)

    def test_three_body_pi0_variant_is_a_recorded_option_default_off(self):
        # Default: only the two-body pair channel is registered.
        chain = _build_chain(_stage_b_overrides(2))
        d2 = chain[9900213]
        assert (2212, -2212, 111) not in d2.channel_keys()
        # Enabled: the variant joins the family behind the same strength
        # key, splitting the share; it stays ineligible below 2.0115 GeV.
        chain3 = _build_chain(_stage_b_overrides(2, pi0_variant=True))
        d3 = chain3[9900213]
        keys = d3.channel_keys()
        assert keys.count((2212, -2212, 111)) == 1
        i2, i3 = keys.index((2212, -2212)), keys.index((2212, -2212, 111))
        w_low = d3.renormalized_mode_weights(2.0, None)
        assert w_low[i3] == 0.0  # below 2 m_p + m_pi0 = 2.0115
        assert w_low[i2] > 0.0
        w_high = d3.renormalized_mode_weights(2.4, None)
        share = d3.pair_family.share(2.4)
        assert w_high[i2] == pytest.approx(0.5 * share, abs=1e-12)
        assert w_high[i3] == pytest.approx(0.5 * share, abs=1e-12)
        assert sum(w_high) == pytest.approx(1.0, abs=1e-12)


class TestMesonMassChangesChannelEligibility:
    """Test (iv), rev-2 two-knob form."""

    def test_daughter_side_knob_flips_pair_channel_membership_in_cm(self):
        chain = _build_chain(_stage_b_overrides(2, pair_strength=0.5))
        d2 = chain[9900213]
        i_pair = d2.channel_keys().index((2212, -2212))
        below = d2.renormalized_mode_weights(_PAIR_THRESHOLD - 0.01, None)
        above = d2.renormalized_mode_weights(_PAIR_THRESHOLD + 0.05, None)
        assert below[i_pair] == 0.0
        assert above[i_pair] > 0.0

    def test_parent_side_knob_flips_nstar_meson_membership_in_cm(self):
        # l2 enabled; the M*(2) daughter mass comes from the PDG table
        # under the production mass-override context. The channel
        # threshold is m_N*(k-1) + m_M*(k) = 1.900 + 2.2 = 4.10 GeV.
        overrides = _stage_b_overrides(2, strength=0.25, pair_strength=0.0)
        chain = _build_chain(overrides)
        updates = _resolve_effective_tower_overrides(overrides)
        n2 = chain[9921201]
        i_link = n2.channel_keys().index((9911201, 9900213))
        with pdg_database.mass_width_overrides(updates):
            below = n2.renormalized_mode_weights(4.05, None)
            above = n2.renormalized_mode_weights(4.15, None)
        assert below[i_link] == 0.0
        assert above[i_link] == pytest.approx(0.25)

    def test_two_knob_event_level_smoke(self):
        gen_file = str(
            Path(__file__).resolve().parents[1]
            / "na49_mixing" / "generators" / "benchmark"
            / "nstar_eff2100_zero.gen"
        )

        def _run(**extra):
            cfg = GenerationConfig(
                gen_file=gen_file,
                n_events=120,
                seed=314,
                beam_momentum=BEAM_MOMENTUM,
                max_decay_passes=8,
                fragmentation_mode="projectile",
                use_soft_decay_factory=True,
                **extra,
            )
            return generate_tracks_chunked(cfg, n_workers=1, n_chunks=1)

        window = {"nstar_eff_tower_max_mass": 5.0,
                  "mestar_eff_tower_max_mass": 2.5}
        # Open box: both knobs in their open position -> p pbar appears
        # (in a matter tower component the antiproton tracks can only
        # come from the pair channel).
        open_run = _run(
            soft_decay_overrides=tuple(
                _stage_b_overrides(2, strength=0.25, pair_strength=0.5).items()
            )
        )
        # Parent-side knob closed: the N* -> N*'M* link is unreachable
        # (the card line shape cannot reach 1.900 + m_M*(2)), so no pair
        # production even with the family strength active.
        parent_closed = _run(
            soft_decay_overrides=tuple({
                **_stage_b_overrides(
                    2, strength=0.25, pair_strength=0.5, mass_overrides=False
                ),
                "nstar_eff_l2_mass": 2.2,
                "mestar_eff_l2_mass": 2.2,
                **window,
            }.items())
        )
        # Daughter-side knob closed: the M* pole sits below 2 m_p, so the
        # pair channel is ineligible in C(M) while the meson link stays open.
        daughter_closed = _run(
            soft_decay_overrides=tuple({
                **_stage_b_overrides(
                    2, strength=0.25, pair_strength=0.5, mass_overrides=False
                ),
                "nstar_eff_l2_mass": 4.6,
                "mestar_eff_l2_mass": 1.80,
                **window,
            }.items())
        )

        def _count(result, pdg):
            return int(np.count_nonzero(result["pdg"] == pdg))

        assert open_run["n_decay_kinematic_failures"] == 0
        assert parent_closed["n_decay_kinematic_failures"] == 0
        assert daughter_closed["n_decay_kinematic_failures"] == 0
        assert _count(open_run, -2212) > 0
        assert _count(open_run, 2212) > 0
        assert _count(parent_closed, -2212) == 0
        assert _count(daughter_closed, -2212) == 0
        # The daughter-closed run still produces the meson link (pions
        # from the M* cascade), isolating the pair channel as the flip.
        assert _count(daughter_closed, -211) > 0


class TestDisabledMesonStageZeroMeffAncestry:
    """Test (vi): disabled stage zero M_eff ancestry."""

    @pytest.mark.parametrize("overrides", [
        {},
        {
            "nstar_eff_l1_br_nstarmeson": 0.0,
            "nstar_eff_l2_br_nstarmeson": 0.0,
            "nstar_eff_l3_br_nstarmeson": 0.0,
            MESTAR_PPBAR_STRENGTH_KEY: 0.0,
        },
    ])
    def test_meff_unreachable_in_built_chain_and_events(self, overrides):
        chain = _build_chain(overrides)
        # Reachability from every component primary of the golden
        # fixtures (the production roots): with the mesonic branches
        # disabled, no walk may reach an M_eff entry, and no pair family
        # may be active anywhere.
        for root in _component_primary_pdgs():
            seen = set()
            frontier = [root]
            while frontier:
                pdg = frontier.pop()
                if pdg in seen or pdg not in chain:
                    continue
                seen.add(pdg)
                assert pdg not in _MEFF_PDGS, (root, pdg)
                pdecays = chain[pdg]
                assert pdecays.pair_family is None
                for mode in pdecays._modes:
                    frontier.extend(int(d) for d in mode.daughters)
        # Event level: the highest tower component, seeded generation.
        gen_file = str(
            Path(__file__).resolve().parents[1]
            / "na49_mixing" / "generators" / "benchmark"
            / "nstar_eff2300_zero.gen"
        )
        cfg = GenerationConfig(
            gen_file=gen_file,
            n_events=100,
            seed=2718,
            beam_momentum=BEAM_MOMENTUM,
            max_decay_passes=8,
            fragmentation_mode="projectile",
            use_soft_decay_factory=True,
            soft_decay_overrides=tuple(overrides.items()),
        )
        tracks = generate_tracks_chunked(cfg, n_workers=1, n_chunks=1)
        assert tracks["n_decay_kinematic_failures"] == 0
        produced = set(int(x) for x in np.unique(tracks["pdg"]))
        assert not (produced & _MEFF_PDGS), produced


class TestEnabledMesonStageTraceableAncestry:
    """Test (vii): enabled stage traceable ancestry + cascade descendants."""

    def test_enabled_l2_chain_is_ancestor_traced(self):
        level = 2
        overrides = _stage_b_overrides(level, strength=0.25, pair_strength=0.5)
        chain = _build_chain(overrides, seed=99)
        updates = _resolve_effective_tower_overrides(overrides)
        card = (
            Path(__file__).resolve().parents[1]
            / "na49_mixing" / "generators" / "benchmark"
            / "nstar_eff2100_zero.gen"
        )
        nstar_pdg = _NSTAR_LEVEL_IDS[level]
        mstar_pdg = _MSTAR_LEVEL_IDS[level]
        lower_mstar_pdg = _MSTAR_LEVEL_IDS[level - 1]
        lower_nstar_pdg = _NSTAR_LEVEL_IDS[level - 1]

        meff_to_nstar = 0
        cascade_traced = 0
        link_realized = 0.0
        with pdg_database.mass_width_overrides(updates):
            rng = np.random.default_rng(4242)
            reaction = Reaction(BEAM_MOMENTUM)
            gen = Generator(reaction=reaction, rng=rng)
            gen.set_all(str(card))
            _patch_generator_mass_width(gen, updates)
            ev = Event(rng=rng)
            ev._decay_chain = chain
            for _ in range(120):
                part = gen.generate()
                ev.add_particle(part)
                ev.mark_primary(part)
                ev.decay_cascade(max_passes=8)
                for p in ev.particles():
                    ancestors = _ancestor_id_chain(p)
                    if ancestors[:2] == [mstar_pdg, nstar_pdg]:
                        meff_to_nstar += 1
                    if ancestors[:4] == [
                        lower_mstar_pdg, mstar_pdg, lower_nstar_pdg
                    ] or ancestors[:3] == [lower_mstar_pdg, mstar_pdg, nstar_pdg]:
                        cascade_traced += 1
                ev.delete()
            stats = ev.decay_stats()
            realized = stats["channel_realized"].get(nstar_pdg, {})
            link_realized = realized.get((lower_nstar_pdg, mstar_pdg), 0.0)

        # The N* -> N*'M* link fired, the M_eff daughter is traceable to
        # its N* parent through the parent link, and the cascade
        # descendants (M_eff(450) below M_eff(850), both below the N*)
        # remain traceable.
        assert link_realized > 0.0
        assert meff_to_nstar > 0
        assert cascade_traced > 0


class TestPairChannelPeakOpenAtFittedMass:
    """Test (15): pair channel peak-open at fitted masses, verified by the
    openness machinery (card rev 2 acceptance item 15)."""

    @staticmethod
    def _parent_bw(pole: float, width: float, schwel: float = 0.45) -> BreitWigner:
        """Card-style line shape at a fitted pole (damping window per the
        benchmark tower rewrite: dumplow = pole + 0.6, dumphigh = pole + 1.3)."""
        return BreitWigner(
            schwel, pole, width, 1.0, pole + 0.60, pole + 1.30,
            rng=np.random.default_rng(5),
        )

    def test_pair_channel_openness_at_mstar_pole(self):
        chain = _build_chain(_stage_b_overrides(2, pair_strength=0.5))
        d2 = chain[9900213]
        pair_mode = d2._modes[d2.channel_keys().index((2212, -2212))]
        # Pair-capable candidates l2 and l3 at representative fitted
        # masses and at the tight box corners: the openness fraction of
        # the parent line shape above 2 m_p exceeds 0.5 everywhere.
        for masses in (
            _STAGE_B_FIT_MASSES[2], _STAGE_B_FIT_MASSES[3],
            _STAGE_B_EDGE_MASSES[2], _STAGE_B_EDGE_MASSES[3],
        ):
            level = 2 if "mestar_eff_l2_mass" in masses else 3
            pole = masses[f"mestar_eff_l{level}_mass"]
            width = {2: 0.22, 3: 0.28}[level]
            bw = self._parent_bw(pole, width)
            openness = channel_parent_openness(pair_mode, bw)
            assert openness > 0.5, (level, pole, openness)

    def test_parent_channel_openness_at_nstar_pole(self):
        chain = _build_chain(_stage_b_overrides(2, pair_strength=0.0))
        n2 = chain[9921201]
        link_mode = n2._modes[n2.channel_keys().index((9911201, 9900213))]
        for masses in (
            _STAGE_B_FIT_MASSES[2], _STAGE_B_FIT_MASSES[3],
            _STAGE_B_EDGE_MASSES[2], _STAGE_B_EDGE_MASSES[3],
        ):
            level = 2 if "nstar_eff_l2_mass" in masses else 3
            pole = masses[f"nstar_eff_l{level}_mass"]
            width = {2: 0.30, 3: 0.35}[level]
            m_mstar = masses[f"mestar_eff_l{level}_mass"]
            bw = self._parent_bw(pole, width)
            with pdg_database.mass_width_overrides(
                {_MSTAR_LEVEL_IDS[level]: (m_mstar, 0.22)}
            ):
                openness = channel_parent_openness(link_mode, bw)
            assert openness > 0.5, (level, pole, openness)

    def test_engine_records_parent_openness_for_enabled_channels(self):
        overrides = _stage_b_overrides(2, strength=0.25, pair_strength=0.5)
        chain = _build_chain(overrides, seed=101)
        updates = _resolve_effective_tower_overrides(overrides)
        mstar_bw = self._parent_bw(2.2, 0.22)
        nstar_bw = self._parent_bw(4.6, 0.30)
        with pdg_database.mass_width_overrides(updates):
            rng = np.random.default_rng(606)
            reaction = Reaction(BEAM_MOMENTUM)
            ev = Event(rng=rng, parent_line_shapes={
                9900213: mstar_bw,
                9921201: nstar_bw,
            })
            ev._decay_chain = chain
            # M* parents sampled from their own line shape: masses below
            # the threshold occur, and there the pair channel is excluded
            # from C(M) instead of sampled.
            masses = mstar_bw.get_random_n(3000)
            for mass in masses:
                parent = Particle.from_id(0.0, 0.0, 0.5, 9900213, reaction)
                parent.set_mass(float(mass))
                ev.add_particle(parent)
                ev.mark_primary(parent)
            # N* parents from the tower line shape at the fitted pole.
            n_masses = nstar_bw.get_random_n(600)
            for mass in n_masses:
                parent = Particle.from_id(0.0, 0.0, 0.5, 9921201, reaction)
                parent.set_mass(float(mass))
                ev.add_particle(parent)
                ev.mark_primary(parent)
            ev.decay_cascade(max_passes=8)
            stats = ev.decay_stats()
        parent_openness = stats["parent_openness_fraction"]
        assert parent_openness[9900213][(2212, -2212)] > 0.5
        assert parent_openness[9921201][(9911201, 9900213)] > 0.5
        # Channels whose parent line shape was not supplied carry no
        # parent-openness entries.
        assert 9900113 not in parent_openness

    def test_bw_fraction_above_grid_consistent_extremes(self):
        bw = self._parent_bw(2.2, 0.22)
        assert bw_fraction_above(bw, -1.0) == pytest.approx(1.0)
        assert bw_fraction_above(bw, 10.0) == 0.0
        assert bw_fraction_above(bw, 2.2) > 0.5
        # Delta line shape: all mass above the threshold at the pole.
        delta = BreitWigner(0.45, 2.2, 0.0, 1.0, 2.8, 3.5)
        assert bw_fraction_above(delta, _PAIR_THRESHOLD) == 1.0
        assert bw_fraction_above(delta, 2.3) == 0.0


class TestNoPpbarBelowTwoProtonThreshold:
    """Test (16): no p pbar below 2 m_p; exclusion by construction."""

    def test_no_pair_decay_below_threshold_over_large_seeded_sample(self):
        overrides = _stage_b_overrides(2, strength=0.25, pair_strength=0.5)
        chain = _build_chain(overrides, seed=55)
        # Fixed-mass M* daughter at 2.2 GeV: the pair channel is open for
        # parents above the threshold.
        with pdg_database.mass_width_overrides({9900213: (2.2, 0.22)}):
            rng = np.random.default_rng(9090)
            reaction = Reaction(BEAM_MOMENTUM)
            ev = _RecordingEvent(rng=rng)
            ev._decay_chain = {9900213: chain[9900213], 9900113: chain[9900113]}
            n = 4000
            masses = rng.uniform(1.3, 2.6, size=n)
            for mass in masses:
                parent = Particle.from_id(0.0, 0.0, 0.5, 9900213, reaction)
                parent.set_mass(float(mass))
                ev.add_particle(parent)
                ev.mark_primary(parent)
            ev.decay_cascade(max_passes=8)
            stats = ev.decay_stats()
            n_below = 0
            n_pair_total = 0
            for m_parent, mode, daughters in ev.decay_records:
                if tuple(mode.daughters) != (2212, -2212):
                    continue
                n_pair_total += 1
                assert float(m_parent) >= _PAIR_THRESHOLD - MASS_CONDITION_TOLERANCE
                total = sum(float(d.get_mass()) for d in daughters)
                assert total >= 2.0 * 0.93827231 - 1e-12
                if float(m_parent) < _PAIR_THRESHOLD - MASS_CONDITION_TOLERANCE:
                    n_below += 1
        assert n_below == 0
        assert n_pair_total > 100
        # Exclusion by construction, never sampled-then-retried: zero
        # kinematic failures over the whole sample spanning the threshold.
        assert stats["n_decay_kinematic_failures"] == 0
        # Every M* parent was attempted once; the cascade daughters of
        # the surviving M_eff(450) add their own attempts on top.
        assert stats["n_decay_attempts"] >= n

    def test_pair_channel_absent_from_cm_below_threshold(self):
        chain = _build_chain(_stage_b_overrides(2, pair_strength=0.8))
        d2 = chain[9900213]
        i_pair = d2.channel_keys().index((2212, -2212))
        weights = d2.renormalized_mode_weights(_PAIR_THRESHOLD - 0.001, None)
        assert weights[i_pair] == 0.0
        assert sum(weights) == pytest.approx(1.0, abs=1e-12)

