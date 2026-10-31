"""Regression tests for Phase A of the mixing2006 physics-code audit fixes.

Covers findings 1 and 2 of notes/mixing2006_07rev_physics_code_audit.md:

1. Zero-width / empty-distribution sampling consistency:
   - a Breit-Wigner with zero total density samples its pole mass rmas,
     identically on the scalar and vectorized paths;
   - Histogram1D with an all-zero distribution returns the range midpoint
     on both sampling paths.
2. Same-event self-pairing and identical-filter double counting:
   - buffered copies carry per-event source-track identifiers;
   - real pairs never pair a track with itself;
   - unordered pairs with both members in both selections count once;
   - pairs with exactly one member exclusive to one selection follow the
     documented single-orientation convention;
   - cross-event mixed pairs are never rejected by identifiers.
"""

import copy
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from na49_mixing.breit_wigner import BreitWigner
from na49_mixing.constants import EMPI
from na49_mixing.cuts import ParticleCut
from na49_mixing.event import Event
from na49_mixing.histogram import Histogram1D
from na49_mixing.mixing import MixBuffer
from na49_mixing.particle import Particle
from na49_mixing.spectra import Spectra


# Audit reproduction parameters: zero-width BW whose support midpoint
# (1.5) differs from the pole mass (1.2), so a midpoint fallback cannot
# masquerade as a delta draw.
AUDIT_BW_PARAMS = dict(
    schwel=1.0, rmas=1.2, brei=0.0, pbrei=1.0, dumplow=1.6, dumphigh=2.0,
)


def _make_pion(px, py, pz, charge=1.0):
    return Particle(px, py, pz, charge, EMPI, 211 if charge > 0 else -211)


def _make_spectra(depth=2, nbins=400, m_low=0.2, m_high=0.6):
    """Spectra sized so that 2*m_pi (~0.279) and modest pair masses are
    inside the histogram range (out-of-range fills land in under/overflow
    and would be invisible to get_entries())."""
    return Spectra("audit", True, nbins=nbins, m_low=m_low, m_high=m_high,
                   depth=depth)


class TestZeroWidthBreitWigner:
    def test_scalar_and_vector_return_pole_mass_exactly(self):
        for seed in (1, 2, 3):
            bw = BreitWigner(rng=np.random.default_rng(seed), **AUDIT_BW_PARAMS)
            for _ in range(50):
                assert bw.get_random() == pytest.approx(1.2, abs=1e-12)
            draws = bw.get_random_n(64)
            assert draws.shape == (64,)
            assert float(np.max(np.abs(draws - 1.2))) <= 1e-12

    def test_delta_mode_survives_parameter_updates(self):
        bw = BreitWigner(1.07, 1.232, 0.12, 1.0, 2.0, 2.5,
                         rng=np.random.default_rng(11))
        # Shrink the width of an ordinary BW to zero via set_parameter.
        bw.set_parameter(2, 0.0)
        for _ in range(20):
            assert bw.get_random() == pytest.approx(1.232, abs=1e-12)
        assert float(np.max(np.abs(bw.get_random_n(16) - 1.232))) <= 1e-12
        # Full parameter update to another zero-width configuration.
        bw.set_parameters(1.0, 1.4, 0.0, 1.0, 1.7, 2.0)
        assert float(np.max(np.abs(bw.get_random_n(16) - 1.4))) <= 1e-12

    def test_nonzero_width_resumes_spread_sampling(self):
        params = dict(AUDIT_BW_PARAMS, brei=0.15)
        bw = BreitWigner(rng=np.random.default_rng(13), **params)
        draws = bw.get_random_n(1000)
        assert draws.min() >= 1.0
        assert draws.max() <= 2.0
        assert draws.std() > 0.0
        assert not np.allclose(draws, 1.2)


class TestHistogramEmptyDistributionPolicy:
    def test_all_zero_contents_return_midpoint_on_both_paths(self):
        h = Histogram1D(10, 1.0, 2.0)
        rng = np.random.default_rng(21)
        for _ in range(20):
            assert h.get_random(rng) == 1.5
        assert np.all(h.get_random_n(8, rng) == 1.5)

    def test_negative_only_contents_are_empty_too(self):
        h = Histogram1D(5, -1.0, 1.0, name="neg")
        h.fill_array(np.linspace(-0.9, 0.9, 7), weights=np.full(7, -2.0))
        rng = np.random.default_rng(22)
        assert h.get_random(rng) == 0.0
        assert np.all(h.get_random_n(5, rng) == 0.0)

    def test_empty_single_bin_returns_midpoint(self):
        h = Histogram1D(1, 3.0, 7.0)
        rng = np.random.default_rng(23)
        assert h.get_random(rng) == 5.0
        assert np.all(h.get_random_n(4, rng) == 5.0)

    def test_paths_agree_on_filled_single_bin_distribution(self):
        h = Histogram1D(1, 0.0, 2.0)
        h.fill(1.0)
        n = 20_000
        rng_scalar = np.random.default_rng(41)
        scalar = np.array([h.get_random(rng_scalar) for _ in range(n)])
        vector = h.get_random_n(n, np.random.default_rng(42))
        # Uniform on [0, 2]: mean 1.0, SEM ~ 0.004 at n = 20000.
        assert abs(scalar.mean() - 1.0) < 0.02
        assert abs(vector.mean() - 1.0) < 0.02
        for sample in (scalar, vector):
            assert sample.min() >= 0.0
            assert sample.max() <= 2.0

    def test_paths_agree_statistically_on_general_distribution(self):
        h = Histogram1D(10, 0.0, 1.0, name="ramp")
        for ibin in range(1, 11):
            h.set_bin_content(ibin, float(ibin))
        n = 20_000
        rng_scalar = np.random.default_rng(31)
        scalar = np.array([h.get_random(rng_scalar) for _ in range(n)])
        vector = h.get_random_n(n, np.random.default_rng(32))
        # Analytic mean of the piecewise-uniform distribution: 0.65.
        assert abs(scalar.mean() - 0.65) < 0.01
        assert abs(vector.mean() - 0.65) < 0.01
        assert abs(scalar.mean() - vector.mean()) < 0.02
        for sample in (scalar, vector):
            assert sample.min() >= 0.0
            assert sample.max() <= 1.0


class TestSourceTrackIdentity:
    def test_default_source_id_is_unassigned_sentinel(self):
        assert Particle().get_source_id() == -1

    def test_copy_propagates_source_id(self):
        p = _make_pion(0.1, 0.0, 1.0)
        c0 = Particle.copy(p)
        assert c0.get_source_id() == -1
        p.set_source_id(7)
        assert Particle.copy(p).get_source_id() == 7

    def test_process_event_stamps_shared_ids_on_both_copies(self):
        cut_a = ParticleCut()
        cut_a.set_id_cut(211)
        cut_b = ParticleCut()  # accepts everything -> overlap with A
        buf = MixBuffer(max_depth=2, a_filter=cut_a, b_filter=cut_b)

        ev = Event()
        ev.add_particle(_make_pion(0.10, 0.00, 1.0))
        ev.add_particle(_make_pion(-0.05, 0.02, 1.1, charge=-1.0))
        assert buf.process_event(ev) is True

        mev = buf._buffer[0]
        a_ids = [p.get_source_id() for p in mev.a_particles()]
        b_ids = [p.get_source_id() for p in mev.b_particles()]
        # The pi+ passes both filters; its A- and B-copies share one id.
        # Ids are assigned per event, compacted over accepted tracks, and
        # distinct across distinct tracks.
        assert a_ids[0] == b_ids[0]
        assert a_ids[0] >= 0
        assert b_ids[0] != b_ids[1]


class TestParticleCutEquivalence:
    def test_identical_settings_are_equivalent(self):
        c1 = ParticleCut()
        c1.set_pt_cut(0.1, 2.0)
        c1.set_id_cut(211)
        c1.set_npoint_cut(3)
        assert c1.is_equivalent(copy.deepcopy(c1))

    def test_empty_cuts_are_equivalent(self):
        assert ParticleCut().is_equivalent(ParticleCut())

    def test_different_bounds_are_not_equivalent(self):
        c1 = ParticleCut()
        c1.set_pt_cut(0.1, 2.0)
        c2 = ParticleCut()
        c2.set_pt_cut(0.1, 2.5)
        assert not c1.is_equivalent(c2)

    def test_npoint_asymmetry_breaks_equivalence(self):
        c1 = ParticleCut()
        c1.set_pt_cut(0.1, 2.0)
        c1.set_npoint_cut(3)
        plain = ParticleCut()
        plain.set_pt_cut(0.1, 2.0)
        assert not c1.is_equivalent(plain)
        # Mirrors the Spectra use case: a deep copy with the NPoint cut
        # switched off selects the same particles as the plain cut.
        stripped = copy.deepcopy(c1)
        stripped.off_npoint_cut()
        assert stripped.is_equivalent(plain)


class TestRealPairFilling:
    """Real-pair conventions with depth-2 buffers (event 1 warms up the
    buffer; the second fill reaches is_ready and fills event 2's pairs)."""

    def _identical_cut(self):
        cut = ParticleCut()
        cut.set_id_cut(211)
        return cut

    def test_one_track_per_event_identical_filters_has_no_self_pairs(self):
        s = _make_spectra()
        cut = self._identical_cut()
        s.set_filters(cut, cut)
        for _ in range(2):
            ev = Event()
            ev.add_particle(_make_pion(0.05, 0.0, 1.0))
            s.fill(ev)
        # No physical two-track pair existed; the artificial 2*m_pi
        # self-pair peak must be gone entirely.
        assert s.get_mass_hist().get_entries() == 0
        assert s.get_mass_hist().get_bin_content(
            s.get_mass_hist().find_bin(2.0 * EMPI)) == 0.0

    def test_two_tracks_identical_filters_count_each_unordered_pair_once(self):
        s = _make_spectra()
        cut = self._identical_cut()
        s.set_filters(cut, cut)
        for _ in range(2):
            ev = Event()
            ev.add_particle(_make_pion(0.05, 0.00, 1.0))
            ev.add_particle(_make_pion(-0.04, 0.03, 1.0))
            s.fill(ev)
        hist = s.get_mass_hist()
        # Two ordered orientations exist but only one unordered pair.
        assert hist.get_entries() == pytest.approx(1.0)
        t1 = _make_pion(0.05, 0.00, 1.0)
        t2 = _make_pion(-0.04, 0.03, 1.0)
        m12 = t1.get_inv_mass(t2)
        assert hist.get_bin_content(hist.find_bin(m12)) == pytest.approx(1.0)
        # No self-pair contribution at twice the pion mass.
        assert hist.get_bin_content(hist.find_bin(2.0 * EMPI)) == 0.0

    def test_partial_overlap_counts_overlap_pairs_once(self):
        """Documented convention: a pair whose members both pass both
        selections has two (A, B) orientations and is counted once. A pair
        with exactly one member exclusive to one selection has only one
        available orientation and contributes that single entry."""
        cut_a = ParticleCut()
        cut_a.set_id_cut(211)
        cut_a.set_pt_cut(0.2, 10.0)   # overlap selection: hard pions only
        cut_b = ParticleCut()
        cut_b.set_id_cut(211)         # superset of A

        s = _make_spectra()
        s.set_filters(cut_a, cut_b)
        for _ in range(2):
            ev = Event()
            ev.add_particle(_make_pion(0.30, 0.00, 1.0))  # in A and B
            ev.add_particle(_make_pion(0.00, 0.35, 1.0))  # in A and B
            ev.add_particle(_make_pion(0.05, 0.00, 1.0))  # soft: B only
            s.fill(ev)

        hist = s.get_mass_hist()
        hi1 = ev.particles()[0]
        hi2 = ev.particles()[1]
        lo = ev.particles()[2]
        expected = {
            hi1.get_inv_mass(hi2): 1.0,   # both in overlap: once, not twice
            hi1.get_inv_mass(lo): 1.0,    # exclusive orientation only
            hi2.get_inv_mass(lo): 1.0,    # exclusive orientation only
        }
        assert hist.get_entries() == pytest.approx(3.0)
        for mass, content in expected.items():
            assert hist.get_bin_content(hist.find_bin(mass)) == (
                pytest.approx(content))
        assert hist.get_bin_content(hist.find_bin(2.0 * EMPI)) == 0.0

    def test_partial_overlap_single_track_in_both_lists_has_no_self_pair(self):
        cut_a = ParticleCut()
        cut_a.set_id_cut(211)
        cut_a.set_pt_cut(0.2, 10.0)
        cut_b = ParticleCut()
        cut_b.set_id_cut(211)

        s = _make_spectra()
        s.set_filters(cut_a, cut_b)
        for _ in range(2):
            ev = Event()
            ev.add_particle(_make_pion(0.30, 0.0, 1.0))  # passes A and B
            s.fill(ev)
        assert s.get_mass_hist().get_entries() == 0

    def test_mixed_pairs_ignore_source_ids_across_events(self):
        """Both events assign source id 0 to their track, yet the
        cross-event mixed pair is counted: identifier-based rejection is
        strictly a same-event rule."""
        s = _make_spectra()
        cut = self._identical_cut()
        s.set_filters(cut, cut)
        for _ in range(2):
            ev = Event()
            ev.add_particle(_make_pion(0.05, 0.0, 1.0))
            s.fill(ev)
        assert s.get_mass_hist().get_entries() == 0
        assert s.get_mix_hist().get_entries() == pytest.approx(1.0)
        # Identical momenta put the mixed pair at exactly twice the pion
        # mass; it must be present there.
        mix_h = s.get_mix_hist()
        assert mix_h.get_bin_content(mix_h.find_bin(2.0 * EMPI)) == (
            pytest.approx(1.0))
