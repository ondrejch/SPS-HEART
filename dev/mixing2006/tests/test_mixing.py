"""Tests for MixEvent and MixBuffer."""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from na49_mixing.mixing import MixEvent, MixBuffer
from na49_mixing.particle import Particle
from na49_mixing.constants import EMPI


def _make_pion(px, py, pz, charge=1.0):
    return Particle(px, py, pz, charge, EMPI, 211 if charge > 0 else -211)


class TestMixEvent:
    def test_empty_event(self):
        ev = MixEvent()
        assert not ev.has_a()
        assert not ev.has_b()

    def test_add_and_retrieve_particles(self):
        ev = MixEvent()
        a = _make_pion(0.1, 0.0, 1.0)
        b = _make_pion(-0.1, 0.0, 1.0, -1.0)
        ev.add_a(a)
        ev.add_b(b)
        assert ev.has_a()
        assert ev.has_b()
        assert ev.get_next_a() is a
        assert ev.get_next_b() is b

    def test_iteration_returns_none_at_end(self):
        ev = MixEvent()
        ev.add_a(_make_pion(0.1, 0.0, 1.0))
        ev.get_next_a()
        assert ev.get_next_a() is None

    def test_reset_iteration(self):
        ev = MixEvent()
        p = _make_pion(0.1, 0.0, 1.0)
        ev.add_a(p)
        ev.get_next_a()
        ev.set_to_first_a()
        assert ev.get_next_a() is p


class TestMixBufferGetNextBMix:
    """Verify MixBuffer.get_next_b_mix() iterates through all mixing events."""

    def _make_mix_event(self, n_a, n_b):
        ev = MixEvent()
        for i in range(n_a):
            ev.add_a(_make_pion(float(i) * 0.1, 0.0, 1.0))
        for i in range(n_b):
            ev.add_b(_make_pion(float(i) * 0.1, 0.0, 1.0, -1.0))
        return ev

    def test_returns_none_when_buffer_too_shallow(self):
        """Buffer with only 1 event (no mixing events) returns None."""
        buf = MixBuffer(max_depth=3)
        ev = self._make_mix_event(1, 2)
        buf.add_mix_event(ev)
        assert buf.get_next_b_mix() is None

    def test_iterates_all_b_particles_in_mixing_events(self):
        """With 2 mixing events of 2 B-particles each, get 4 particles then None."""
        buf = MixBuffer(max_depth=3)
        # Fill buffer to depth 3: events[0]=current, events[1..2]=mixing
        for _ in range(3):
            ev = self._make_mix_event(1, 2)
            buf.add_mix_event(ev)

        # Collect all B-particles from mixing events
        collected = []
        while True:
            p = buf.get_next_b_mix()
            if p is None:
                break
            collected.append(p)

        # 2 mixing events × 2 B-particles each = 4 particles
        assert len(collected) == 4

    def test_returns_none_after_exhaustion_then_restarts(self):
        """After iterating all mixing B-particles, returns None and resets.
        The subsequent call starts a new iteration cycle."""
        buf = MixBuffer(max_depth=3)
        for _ in range(3):
            buf.add_mix_event(self._make_mix_event(1, 1))

        # First cycle: drain all B-particles
        first_cycle = []
        while True:
            p = buf.get_next_b_mix()
            if p is None:
                break
            first_cycle.append(p)

        # After reset, a new cycle begins (non-None particle returned)
        assert len(first_cycle) == 2  # 2 mixing events × 1 B-particle each
        # The iterator has reset, so the next call starts a new cycle
        p = buf.get_next_b_mix()
        assert p is not None

    def test_can_iterate_again_after_reset(self):
        """After exhaustion and reset, the next full cycle returns the same count."""
        buf = MixBuffer(max_depth=3)
        for _ in range(3):
            buf.add_mix_event(self._make_mix_event(1, 3))

        def drain_cycle():
            particles = []
            while True:
                p = buf.get_next_b_mix()
                if p is None:
                    break
                particles.append(p)
            return particles

        first_run = drain_cycle()
        second_run = drain_cycle()
        # 2 mixing events × 3 B-particles each = 6 per cycle
        assert len(first_run) == 6
        assert len(second_run) == 6

    def test_single_mixing_event_all_particles(self):
        """Buffer at depth 2: 1 current + 1 mixing event."""
        buf = MixBuffer(max_depth=2)
        buf.add_mix_event(self._make_mix_event(1, 5))
        buf.add_mix_event(self._make_mix_event(1, 5))

        collected = []
        while True:
            p = buf.get_next_b_mix()
            if p is None:
                break
            collected.append(p)

        assert len(collected) == 5  # 1 mixing event × 5 B-particles
