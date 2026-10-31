"""Broadened coverage for the mixing2006 audit fixes.

Complements tests/test_audit_fixes_phase_a/b/c.py (which pin the six fix
behaviors themselves) with wider scenarios across commits 5765f8b
(zero-width BW delta sampling, same-event de-duplication), 83cf34d
(NPoint-free real-pair filling, normalization convention, merge chunk
schema), and aabe498 (save-collision refusal, eager domain validation):

- Spectra real vs mixed pair counting under the documented Option-A
  convention, warm-up depth gating and rolling-buffer bookkeeping,
  partially overlapping A/B selections (non-nested exclusives on both
  sides), PairCut parity between the real and mixed stages, event-weight
  propagation versus the unweighted ``get_entries()`` counter;
- events that pass the real-pair stage but fail the NPoint-enabled
  buffer filters (the converse direction is structurally unreachable,
  see the class docstrings);
- hand-counted unordered-pair expectations for 1-, 2-, and 3-track
  events and a many-event no-self-pair peak property;
- scalar ``get_random()`` vs vectorized ``get_random_n()`` consistency
  for Breit-Wigners (including a zero-total-density configuration
  reached through ``pbrei``) and the shared Histogram1D empty-policy
  across empty/filled/emptied/refilled state transitions.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from na49_mixing.breit_wigner import BreitWigner, bw_new_array
from na49_mixing.constants import EMPI
from na49_mixing.cuts import ParticleCut, PairCut
from na49_mixing.event import Event
from na49_mixing.histogram import Histogram1D
from na49_mixing.particle import Particle
from na49_mixing.spectra import Spectra


NBINS = 400
M_LOW, M_HIGH = 0.2, 0.6


def _make_pion(px, py=0.0, pz=1.0):
    return Particle(px, py, pz, 1.0, EMPI, 211)


def _make_spectra(name, depth=2):
    """Spectra sized so every pair mass used below sits well inside the
    histogram range; out-of-range fills land in under/overflow and would
    be invisible to the entry counts."""
    return Spectra(name, True, nbins=NBINS, m_low=M_LOW, m_high=M_HIGH,
                   depth=depth)


def _pair_mass(px1, px2):
    return _make_pion(px1).get_inv_mass(_make_pion(px2))


def _id_cut():
    cut = ParticleCut()
    cut.set_id_cut(211)
    return cut


def _npoint_cut(np_min=4):
    cut = ParticleCut()
    cut.set_id_cut(211)
    cut.set_npoint_cut(np_min)
    return cut


def _require_distinct_bins(hist, *values):
    """Guard so per-bin assertions cannot silently share one bin."""
    bins = [hist.find_bin(v) for v in values]
    assert len(set(bins)) == len(bins), (
        f"test design error: values {values} share bins {bins}"
    )
    return bins


class TestSpectraWarmupAndRolling:
    """Warm-up gating and steady-state rolling counts (depth 5, one pi+
    per event, identical A/B selections)."""

    DEPTH = 5

    def _events(self, n):
        events = []
        for k in range(n):
            ev = Event()
            ev.add_particle(_make_pion(0.05 * k))
            events.append(ev)
        return events

    def test_histogram_filling_starts_only_at_full_depth(self):
        s = _make_spectra("cov_warmup", depth=self.DEPTH)
        s.set_filters(_id_cut(), _id_cut())
        events = self._events(self.DEPTH)

        for ev in events[:self.DEPTH - 1]:
            assert s.fill(ev) is True
        # Every warm-up event is accepted into the real-pair stage, but
        # neither histogram moves before the buffer reaches full depth.
        assert s.get_entries() == self.DEPTH - 1
        assert s.get_mass_hist().get_entries() == pytest.approx(0.0)
        assert s.get_mix_hist().get_entries() == pytest.approx(0.0)

        # First ready event: still no real pair (single track), but the
        # current A particle pairs with the depth-1 buffered B particles.
        assert s.fill(events[self.DEPTH - 1]) is True
        assert s.get_entries() == self.DEPTH
        assert s.get_mass_hist().get_entries() == pytest.approx(0.0)
        assert s.get_mix_hist().get_entries() == pytest.approx(float(self.DEPTH - 1))

    def test_rolling_events_leave_no_self_pair_peak(self):
        """Property-style check over many single-particle events with
        distinct momenta: the real spectrum must stay exactly empty (no
        2*m_pi self-pair peak) while the mixed spectrum grows by
        (depth - 1) entries per rolling event."""
        n_events = 25
        s = _make_spectra("cov_rolling", depth=self.DEPTH)
        s.set_filters(_id_cut(), _id_cut())

        for ev in self._events(n_events):
            s.fill(ev)

        assert s.get_entries() == n_events
        mass_h = s.get_mass_hist()
        assert mass_h.get_entries() == pytest.approx(0.0)
        assert mass_h.get_bin_content(mass_h.find_bin(2.0 * EMPI)) == 0.0
        # Events 5..25 each contribute current-A x (depth-1) mixed pairs.
        n_filling = n_events - self.DEPTH + 1
        assert s.get_mix_hist().get_entries() == pytest.approx(
            float(n_filling * (self.DEPTH - 1)))


class TestSpectraRealVsMixedSelections:
    """Events whose tracks pass the real-pair stage but fail the
    NPoint-enabled buffer filters (Option-A convention). The converse
    direction cannot occur structurally: the real-stage filters are the
    buffer filters with the NPoint cut stripped, so any event entering
    the buffer automatically passes the real stage."""

    def test_npoint_failing_track_enters_real_but_not_mixed(self):
        s = _make_spectra("cov_split")
        s.set_filters(_npoint_cut(), _npoint_cut())

        for px in (0.00, 0.10):     # warm-up, both fully accepted
            ev = Event()
            p = _make_pion(px)
            p.set_n_point(9)
            ev.add_particle(p)
            s.fill(ev)
        assert s.get_entries() == 2

        # One accepted track plus one track failing only the NPoint cut.
        t_in = _make_pion(0.18)
        t_in.set_n_point(9)
        t_out = _make_pion(0.24)    # default n_point = 0
        ev = Event()
        ev.add_particle(t_in)
        ev.add_particle(t_out)

        m_pair = t_in.get_inv_mass(t_out)
        m_partner = _pair_mass(0.18, 0.10)
        m_warmup = _pair_mass(0.10, 0.00)
        mass_h = s.get_mass_hist()
        mix_h = s.get_mix_hist()
        _require_distinct_bins(mass_h, m_pair, 2.0 * EMPI)
        _require_distinct_bins(mix_h, m_partner, m_warmup, m_pair)

        assert s.fill(ev) is True   # t_in alone populates both buffer lists
        assert s.get_entries() == 3

        # Real stage: the NPoint-failing track participates exactly once.
        assert mass_h.get_entries() == pytest.approx(1.0)
        assert mass_h.get_bin_content(mass_h.find_bin(m_pair)) == pytest.approx(1.0)
        assert mass_h.get_bin_content(mass_h.find_bin(2.0 * EMPI)) == 0.0

        # Mixed stage: built exclusively from NPoint-passing copies; the
        # rejected track appears in no mixed pair.
        assert mix_h.get_entries() == pytest.approx(2.0)
        assert mix_h.get_bin_content(mix_h.find_bin(m_partner)) == pytest.approx(1.0)
        assert mix_h.get_bin_content(mix_h.find_bin(m_warmup)) == pytest.approx(1.0)
        assert mix_h.get_bin_content(mix_h.find_bin(t_out.get_inv_mass(t_in))) == 0.0

    def test_rejected_event_fills_real_pairs_without_replaying_mixed(self):
        s = _make_spectra("cov_interleave")
        s.set_filters(_npoint_cut(), _npoint_cut())

        puts = []
        for px in (0.00, 0.06):
            ev = Event()
            p = _make_pion(px)
            p.set_n_point(9)
            ev.add_particle(p)
            puts.append(s.fill(ev))
        assert puts == [True, True]
        assert s.get_entries() == 2
        assert s.get_mix_hist().get_entries() == pytest.approx(1.0)

        # A two-track event failing only NPoint: rejected by the buffer,
        # yet counted and paired in the real stage (buffer stays ready).
        ev_bad = Event()
        ev_bad.add_particle(_make_pion(0.18))
        ev_bad.add_particle(_make_pion(0.24))
        assert s.fill(ev_bad) is False
        assert s.get_entries() == 3
        mass_h = s.get_mass_hist()
        assert mass_h.get_entries() == pytest.approx(1.0)
        assert mass_h.get_bin_content(
            mass_h.find_bin(_pair_mass(0.18, 0.24))) == pytest.approx(1.0)

        # The next accepted event mixes with the last ACCEPTED event's
        # B copy only: no replay of the rejected event into the mixing
        # iteration.
        ev = Event()
        p = _make_pion(0.30)
        p.set_n_point(9)
        ev.add_particle(p)
        assert s.fill(ev) is True
        assert s.get_entries() == 4

        mix_h = s.get_mix_hist()
        m_first = _pair_mass(0.06, 0.00)
        m_second = _pair_mass(0.30, 0.06)
        _require_distinct_bins(mix_h, m_first, m_second,
                               _pair_mass(0.30, 0.18))
        assert mix_h.get_entries() == pytest.approx(2.0)
        assert mix_h.get_bin_content(mix_h.find_bin(m_first)) == pytest.approx(1.0)
        assert mix_h.get_bin_content(mix_h.find_bin(m_second)) == pytest.approx(1.0)
        assert mix_h.get_bin_content(
            mix_h.find_bin(_pair_mass(0.30, 0.18))) == 0.0

    def test_buffer_acceptance_implies_real_stage_acceptance(self):
        """Property-style sweep over NPoint-passing and NPoint-failing
        events: whenever the buffer accepts an event the normalization
        counter increments on the very same call; events failing only
        the buffer filters still increment it (documented independence),
        so the counter equals the number of real-stage acceptances."""
        s = _make_spectra("cov_implication", depth=3)
        s.set_filters(_npoint_cut(), _npoint_cut())

        plan = [
            (True, [9]),        # accepted by both stages
            (False, [0, 0]),    # real stage only
            (True, [9, 9]),     # accepted by both stages
            (False, [0]),       # real stage only
            (True, [9, 9, 9]),  # accepted by both stages
        ]
        for expected_put, npoints in plan:
            ev = Event()
            for i, npts in enumerate(npoints):
                p = _make_pion(0.05 * i + 0.02)
                p.set_n_point(npts)
                ev.add_particle(p)
            entries_before = s.get_entries()
            assert s.fill(ev) is expected_put
            # Real-stage acceptance happens regardless of the buffer.
            assert s.get_entries() == entries_before + 1
        assert s.get_entries() == len(plan)


class TestSpectraWeightsAndEntries:
    def test_weights_fill_histograms_while_entries_count_events(self):
        """Event weights propagate into both histogram fills, whereas
        ``get_entries()`` keeps counting accepted events (never
        weights)."""
        s = _make_spectra("cov_weights")
        s.set_filters(_id_cut(), _id_cut())

        ev_warm = Event()                       # warms up, never fills
        ev_warm.add_particle(_make_pion(0.40))
        ev_warm.set_weight(2.0)
        ev_m = Event()                          # first filling event
        ev_m.add_particle(_make_pion(0.10))
        ev_m.set_weight(3.0)
        ev_n = Event()                          # second filling event
        ev_n.add_particle(_make_pion(0.20))
        ev_n.set_weight(5.0)
        ev_o = Event()                          # third filling event
        ev_o.add_particle(_make_pion(0.00))
        ev_o.add_particle(_make_pion(0.06))
        ev_o.set_weight(7.0)

        for ev in (ev_warm, ev_m, ev_n, ev_o):
            s.fill(ev)

        assert s.get_entries() == 4             # unweighted event count

        mass_h = s.get_mass_hist()
        mix_h = s.get_mix_hist()
        m_real = _pair_mass(0.00, 0.06)
        _require_distinct_bins(
            mass_h, m_real,
            _pair_mass(0.10, 0.40), _pair_mass(0.20, 0.10),
            _pair_mass(0.00, 0.20), _pair_mass(0.06, 0.20))

        # Real spectrum: only ev_o owns an unordered pair -> weight 7.
        assert mass_h.get_entries() == pytest.approx(7.0)
        assert mass_h.get_bin_content(mass_h.find_bin(m_real)) == pytest.approx(7.0)

        # Mixed spectrum per filling event:
        #   ev_m: 1 x 1 pair against the warm-up event, weight 3;
        #   ev_n: 1 x 1 pair against ev_m, weight 5;
        #   ev_o: 2 A copies x 1 buffered B (ev_n), weight 7 each.
        assert mix_h.get_entries() == pytest.approx(3.0 + 5.0 + 2 * 7.0)
        assert mix_h.get_bin_content(
            mix_h.find_bin(_pair_mass(0.10, 0.40))) == pytest.approx(3.0)
        assert mix_h.get_bin_content(
            mix_h.find_bin(_pair_mass(0.20, 0.10))) == pytest.approx(5.0)
        assert mix_h.get_bin_content(
            mix_h.find_bin(_pair_mass(0.00, 0.20))) == pytest.approx(7.0)
        assert mix_h.get_bin_content(
            mix_h.find_bin(_pair_mass(0.06, 0.20))) == pytest.approx(7.0)


class TestSpectraHandCountedPairs:
    @pytest.mark.parametrize("n_tracks, expected_real", [
        (1, 0.0),
        (2, 1.0),
        (3, 3.0),
    ])
    def test_track_arities_match_hand_counted_pairs(self, n_tracks,
                                                    expected_real):
        """Identical A/B selections: an n-track event holds C(n,2)
        unordered pairs, each counted exactly once in the second
        (buffer-ready) event; the mixed stage contributes n x n pairs."""
        s = _make_spectra(f"cov_arity{n_tracks}")
        s.set_filters(_id_cut(), _id_cut())

        # Non-uniform momentum grid: parallel-track pair masses depend
        # only on the px difference, so equal spacing would collide.
        px_grid = (0.02, 0.11, 0.26)
        pair_masses = None
        for _ in range(2):
            ev = Event()
            for i in range(n_tracks):
                ev.add_particle(_make_pion(px_grid[i]))
            if pair_masses is None:
                pair_masses = [
                    ev.particles()[i].get_inv_mass(ev.particles()[j])
                    for i in range(n_tracks)
                    for j in range(i + 1, n_tracks)
                ]
            s.fill(ev)

        mass_h = s.get_mass_hist()
        assert s.get_entries() == 2
        assert mass_h.get_entries() == pytest.approx(expected_real)
        if pair_masses:
            _require_distinct_bins(mass_h, *pair_masses, 2.0 * EMPI)
        for m_expected in pair_masses:
            assert mass_h.get_bin_content(
                mass_h.find_bin(m_expected)) == pytest.approx(1.0)
        assert mass_h.get_bin_content(mass_h.find_bin(2.0 * EMPI)) == 0.0
        # Mixed: current n A copies against n buffered B copies.
        assert s.get_mix_hist().get_entries() == pytest.approx(
            float(n_tracks * n_tracks))


class TestSpectraOverlapConventions:
    def test_non_nested_overlap_counts_each_unordered_pair_once(self):
        """A and B overlap without nesting (hard-positive core, soft and
        hard-negative exclusives on either side). Documented convention:
        a pair with both members in the overlap has two orientations and
        is counted once; a pair whose exclusive member belongs to one
        selection only contributes that single orientation."""
        cut_a = ParticleCut()
        cut_a.set_id_cut(211)
        cut_a.set_pt_cut(0.20, 10.0)      # hard positives
        cut_b = ParticleCut()
        cut_b.set_id_cut(211)
        cut_b.set_pt_cut(0.05, 0.25)      # soft-to-medium positives

        s = _make_spectra("cov_nonnested")
        s.set_filters(cut_a, cut_b)
        for _ in range(2):
            ev = Event()
            ev.add_particle(_make_pion(0.30))   # A only (hard exclusive)
            ev.add_particle(_make_pion(0.22))   # both (overlap core)
            ev.add_particle(_make_pion(0.08))   # B only (soft exclusive)
            s.fill(ev)

        mass_h = s.get_mass_hist()
        m_hard_core = _pair_mass(0.30, 0.22)
        m_hard_soft = _pair_mass(0.30, 0.08)
        m_core_soft = _pair_mass(0.22, 0.08)
        _require_distinct_bins(mass_h, m_hard_core, m_hard_soft,
                               m_core_soft, 2.0 * EMPI)
        assert s.get_entries() == 2
        assert mass_h.get_entries() == pytest.approx(3.0)
        for m_expected in (m_hard_core, m_hard_soft, m_core_soft):
            assert mass_h.get_bin_content(
                mass_h.find_bin(m_expected)) == pytest.approx(1.0)
        assert mass_h.get_bin_content(mass_h.find_bin(2.0 * EMPI)) == 0.0


class TestSpectraPairCutParity:
    @pytest.mark.parametrize("label, m_lo, m_hi", [
        ("inside", 0.26, 0.30),   # contains the 2*m_pi pair mass
        ("outside", 0.35, 0.45),  # excludes it entirely
    ])
    def test_pair_cut_gates_real_and_mixed_identically(self, label, m_lo,
                                                       m_hi):
        pair_cut = PairCut()
        pair_cut.set_inv_mass_cut(m_lo, m_hi)
        s = _make_spectra(f"cov_paircut_{label}")
        s.set_filters(_id_cut(), _id_cut())
        s.set_pair_cut(pair_cut)

        # Two tracks sharing one momentum vector pair at exactly 2*m_pi.
        for _ in range(2):
            ev = Event()
            ev.add_particle(_make_pion(0.02))
            ev.add_particle(_make_pion(0.02))
            s.fill(ev)

        inside = label == "inside"
        # Normalization counts stage acceptance and ignores the pair veto.
        assert s.get_entries() == 2
        assert s.get_mass_hist().get_entries() == pytest.approx(
            1.0 if inside else 0.0)
        assert s.get_mix_hist().get_entries() == pytest.approx(
            4.0 if inside else 0.0)
        if inside:
            mass_h = s.get_mass_hist()
            mix_h = s.get_mix_hist()
            assert mass_h.get_bin_content(
                mass_h.find_bin(2.0 * EMPI)) == pytest.approx(1.0)
            assert mix_h.get_bin_content(
                mix_h.find_bin(2.0 * EMPI)) == pytest.approx(4.0)


class TestBreitWignerCrossPathConsistency:
    def test_scalar_and_vector_agree_on_nonzero_width_distribution(self):
        """Delta(1232) defaults: both sampling paths stay inside the
        support and agree on mean and spread within statistical
        tolerance."""
        n = 20_000
        schwel, dumphigh = 1.07, 2.5
        scalar_bw = BreitWigner(rng=np.random.default_rng(101))
        scalar = np.array([scalar_bw.get_random() for _ in range(n)])
        vector = BreitWigner(rng=np.random.default_rng(202)).get_random_n(n)

        for sample in (scalar, vector):
            assert sample.shape == (n,)
            assert sample.min() >= schwel
            assert sample.max() <= dumphigh
        assert abs(scalar.mean() - vector.mean()) < 0.01
        assert abs(scalar.std() - vector.std()) < 0.01

    def test_zero_total_density_via_pbrei_is_delta_at_rmas(self):
        """Zero total density selects delta-at-rmas mode regardless of
        which parameter voids the density; here pbrei = 0 with a finite
        nominal width. Scalar and vector paths must agree with the
        constructor route and after set_parameters()."""
        params = (1.0, 1.2, 0.15, 0.0, 1.6, 2.0)
        xs = np.linspace(params[0], params[5], 10_001)
        assert float(bw_new_array(xs, list(params)).sum()) == 0.0

        bw = BreitWigner(*params, rng=np.random.default_rng(3))
        assert bw.get_value(1.2) == 0.0
        for _ in range(20):
            assert bw.get_random() == pytest.approx(1.2, abs=1e-12)
        np.testing.assert_allclose(bw.get_random_n(32), 1.2, atol=1e-12)

        # The update path re-evaluates the density and re-enters delta
        # mode from an ordinary (sampleable) configuration.
        bw_live = BreitWigner(1.07, 1.232, 0.12, 1.0, 2.0, 2.5,
                              rng=np.random.default_rng(4))
        bw_live.set_parameters(*params)
        assert bw_live.get_random() == pytest.approx(1.2, abs=1e-12)
        np.testing.assert_allclose(bw_live.get_random_n(16), 1.2, atol=1e-12)


class TestHistogramEmptyPolicyTransitions:
    def test_empty_filled_emptied_refilled_follows_shared_policy(self):
        """The midpoint policy applies whenever the distribution carries
        no probability mass, and both paths resume inverse-CDF sampling
        as soon as mass returns; cache invalidation must not freeze an
        older policy."""
        h = Histogram1D(10, 0.0, 1.0)
        rng = np.random.default_rng(5)

        # Freshly constructed: empty -> midpoint on both paths.
        assert h.get_random(rng) == 0.5
        assert np.all(h.get_random_n(4, rng) == 0.5)

        # Single filled bin [0.2, 0.3): sampling resumes inside it.
        h.fill(0.25)
        draw = h.get_random(rng)
        assert 0.2 <= draw < 0.3
        assert np.all((h.get_random_n(16, rng) >= 0.2)
                      & (h.get_random_n(16, rng) < 0.3))

        # Scaling to zero empties the distribution again.
        h.scale(0.0)
        assert h.get_random(rng) == 0.5
        assert np.all(h.get_random_n(4, rng) == 0.5)

        # Refilling switches back, carrying the stored weight along.
        h.fill(0.75, weight=2.0)
        draw = h.get_random(rng)
        assert 0.7 <= draw < 0.8
        refilled = h.get_random_n(16, rng)
        assert np.all((refilled >= 0.7) & (refilled < 0.8))
        assert h.get_entries() == pytest.approx(2.0)
