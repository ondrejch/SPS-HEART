"""Regression tests for Phase B of the mixing2006 physics-code audit fixes.

Covers finding 3 of notes/mixing2006_07rev_physics_code_audit.md
(Spectra real-pair / normalization convention, Option A):

- real pairs are formed from the original event through NPoint-free
  copies of the A/B filters, so tracks that pass every cut except
  NPoint still contribute real entries and +1 to ``get_entries()``;
- mixed pairs apply the original NPoint-enabled buffer filters, so such
  an event enters neither the buffer nor the mixed histogram;
- ``get_entries()`` counts events accepted into the real-pair stage;
  events without at least one A and one B candidate are not counted.

The finding-4 merge-schema regression tests live in
tests/test_parallel_generation.py next to the existing merge suite.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from na49_mixing.constants import EMPI
from na49_mixing.cuts import ParticleCut
from na49_mixing.event import Event
from na49_mixing.particle import Particle
from na49_mixing.spectra import Spectra


def _make_pion(px, py, pz, charge=1.0):
    return Particle(px, py, pz, charge, EMPI, 211 if charge > 0 else -211)


def _make_spectra(depth=2, nbins=400, m_low=0.2, m_high=0.6):
    """Spectra sized so that 2*m_pi (~0.279) and modest pair masses sit
    inside the histogram range (out-of-range fills land in under/overflow
    and would be invisible to the entry counts)."""
    return Spectra("audit_b", True, nbins=nbins, m_low=m_low, m_high=m_high,
                   depth=depth)


def _npoint_cut():
    """Identical A/B selection accepting pi+ with n_point >= 4."""
    cut = ParticleCut()
    cut.set_id_cut(211)
    cut.set_npoint_cut(4)
    return cut


class TestNPointOnlyFailureStillFillsRealPairs:
    """Documented Option-A behavior for events whose tracks fail only
    the NPoint cut."""

    def _fill_warmup(self, spectra):
        """Two single-pion events with n_point = 9 bring the depth-2
        buffer to readiness (event 2 already fills one mixed pair)."""
        for _ in range(2):
            ev = Event()
            p = _make_pion(0.05, 0.0, 1.0)
            p.set_n_point(9)
            ev.add_particle(p)
            spectra.fill(ev)

    def test_npoint_failing_event_counts_and_fills_real_pairs(self):
        s = _make_spectra()
        cut = _npoint_cut()
        s.set_filters(cut, cut)
        self._fill_warmup(s)
        assert s.get_entries() == 2
        assert s.get_mass_hist().get_entries() == pytest.approx(0.0)
        assert s.get_mix_hist().get_entries() == pytest.approx(1.0)

        # Both tracks pass every cut except NPoint (default n_point = 0).
        t1 = _make_pion(0.05, 0.00, 1.0)
        t2 = _make_pion(-0.04, 0.03, 1.0)
        ev = Event()
        ev.add_particle(t1)
        ev.add_particle(t2)

        put = s.fill(ev)

        # The NPoint-enabled buffer filters reject the event ...
        assert put is False
        # ... yet it is accepted into the real-pair stage: +1 entry and
        # exactly one unordered real pair filled.
        assert s.get_entries() == 3
        hist = s.get_mass_hist()
        assert hist.get_entries() == pytest.approx(1.0)
        assert hist.get_bin_content(hist.find_bin(t1.get_inv_mass(t2))) == (
            pytest.approx(1.0))
        # No artificial self-pair peak at twice the pion mass.
        assert hist.get_bin_content(hist.find_bin(2.0 * EMPI)) == 0.0

    def test_buffer_rejected_event_contributes_no_mixed_entries(self):
        s = _make_spectra()
        cut = _npoint_cut()
        s.set_filters(cut, cut)
        self._fill_warmup(s)
        mix_before = s.get_mix_hist().get_entries()

        ev = Event()
        ev.add_particle(_make_pion(0.05, 0.00, 1.0))
        ev.add_particle(_make_pion(-0.04, 0.03, 1.0))
        assert s.fill(ev) is False

        # The mixed histogram is untouched: the event never entered the
        # buffer, and the previous event's mixed pairs are not replayed.
        assert s.get_mix_hist().get_entries() == pytest.approx(mix_before)

    def test_rejected_event_during_warmup_counts_but_fills_nothing(self):
        """Histogram filling stays gated on buffer readiness (historical
        warm-up behavior); only the stage acceptance is counted."""
        s = _make_spectra()
        cut = _npoint_cut()
        s.set_filters(cut, cut)

        ev1 = Event()
        p = _make_pion(0.05, 0.0, 1.0)
        p.set_n_point(9)
        ev1.add_particle(p)
        assert s.fill(ev1) is True
        assert s.get_entries() == 1

        ev2 = Event()
        ev2.add_particle(_make_pion(0.05, 0.00, 1.0))
        ev2.add_particle(_make_pion(-0.04, 0.03, 1.0))
        assert s.fill(ev2) is False

        # Accepted into the real-pair stage (+1), but the buffer holds
        # only one event, so no histogram filling happened yet.
        assert s.get_entries() == 2
        assert s.get_mass_hist().get_entries() == 0
        assert s.get_mix_hist().get_entries() == 0


class TestNormalizationCountsRealPairStageOnly:
    """Events without at least one A and one B candidate under the
    NPoint-free filters are never counted by get_entries()."""

    def _filters(self):
        cut_a = ParticleCut()
        cut_a.set_id_cut(211)
        cut_b = ParticleCut()
        cut_b.set_id_cut(211)
        cut_b.set_pt_cut(5.0, 50.0)  # B: hard pions only
        return cut_a, cut_b

    def test_soft_only_event_is_not_counted(self):
        cut_a, cut_b = self._filters()
        s = _make_spectra()
        s.set_filters(cut_a, cut_b)

        for _ in range(2):
            ev = Event()
            p = _make_pion(5.5, 0.0, 1.0)  # passes both selections
            ev.add_particle(p)
            s.fill(ev)
        assert s.get_entries() == 2
        mix_before = s.get_mix_hist().get_entries()

        # Soft pion passes A but no B candidate: rejected before the
        # normalization increment and before the buffer.
        ev = Event()
        ev.add_particle(_make_pion(0.05, 0.0, 1.0))
        assert s.fill(ev) is False
        assert s.get_entries() == 2
        assert s.get_mix_hist().get_entries() == pytest.approx(mix_before)

    def test_event_outside_the_selection_is_not_counted(self):
        cut_a, cut_b = self._filters()
        s = _make_spectra()
        s.set_filters(cut_a, cut_b)

        for _ in range(2):
            ev = Event()
            p = _make_pion(5.5, 0.0, 1.0)
            ev.add_particle(p)
            s.fill(ev)

        # A proton matches neither selection.
        ev = Event()
        ev.add_particle(Particle.from_id(0.05, 0.0, 1.0, 2212))
        assert s.fill(ev) is False
        assert s.get_entries() == 2

        # An empty event likewise contributes nothing.
        assert s.fill(Event()) is False
        assert s.get_entries() == 2

    def test_event_cut_rejection_is_not_counted(self):
        cut = _npoint_cut()
        s = _make_spectra()
        s.set_filters(cut, cut)
        # Multiplicity window counted on pi+ tracks; the single-track
        # event falls below the minimum and is rejected outright.
        mult_cut = ParticleCut()
        mult_cut.set_id_cut(211)
        s.set_ev_cut(5, 100, mult_cut)

        ev = Event()
        p = _make_pion(0.05, 0.0, 1.0)
        p.set_n_point(9)
        ev.add_particle(p)
        assert s.fill(ev) is False
        assert s.get_entries() == 0
