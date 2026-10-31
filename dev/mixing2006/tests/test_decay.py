"""Tests for decay kinematics — energy-momentum conservation."""
import math
import os
import sys
import tempfile
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from na49_mixing.event import Event
from na49_mixing.particle import Particle
from na49_mixing.decays import DecayMode, ParticleDecays, read_decays, write_decays
from na49_mixing.reaction import Reaction
from na49_mixing.constants import EMPROT, EMPI


RNG = np.random.default_rng(42)


def make_event(rng=None) -> Event:
    return Event(rng=rng or RNG)


class TestTwoBodyDecay:
    """Delta(1232) → p + π⁺ should conserve 4-momentum."""

    def _decay_and_check(self, parent_px, parent_py, parent_pz, parent_mass):
        r = Reaction(158.0)
        ev = make_event()
        parent = Particle.from_id(parent_px, parent_py, parent_pz, 2224, r)
        parent.set_mass(parent_mass)
        ev.add_particle(parent)

        mode = DecayMode(2, [2212, 211])
        ok = ev._decay_2body(parent, mode)
        assert ok, "Decay kinematics failed"

        # After decay, parent is still in the list (not yet removed)
        # Children were appended at the end
        daughters = ev.particles()[-2:]
        d1, d2 = daughters

        px_sum = d1.get_px() + d2.get_px()
        py_sum = d1.get_py() + d2.get_py()
        pz_sum = d1.get_pz() + d2.get_pz()
        e_sum = d1.get_e() + d2.get_e()

        assert px_sum == pytest.approx(parent_px, abs=1e-5)
        assert py_sum == pytest.approx(parent_py, abs=1e-5)
        assert pz_sum == pytest.approx(parent_pz, abs=1e-5)
        assert e_sum == pytest.approx(parent.get_e(), rel=1e-5)

    def test_decay_at_rest(self):
        self._decay_and_check(0.0, 0.0, 0.0, 1.232)

    def test_decay_moving(self):
        self._decay_and_check(0.3, 0.2, 10.0, 1.232)

    def test_daughter_masses(self):
        r = Reaction(158.0)
        ev = make_event()
        parent = Particle.from_id(0.0, 0.0, 5.0, 2224, r)
        parent.set_mass(1.232)
        ev.add_particle(parent)
        mode = DecayMode(2, [2212, 211])
        ok = ev._decay_2body(parent, mode)
        assert ok
        d1, d2 = ev.particles()[-2:]
        assert d1.get_mass() == pytest.approx(EMPROT, rel=1e-4)
        assert d2.get_mass() == pytest.approx(EMPI, rel=1e-4)

    def test_daughter_pdg_ids(self):
        r = Reaction(158.0)
        ev = make_event()
        parent = Particle.from_id(0.0, 0.0, 5.0, 2224, r)
        parent.set_mass(1.232)
        ev.add_particle(parent)
        mode = DecayMode(2, [2212, 211])
        ev._decay_2body(parent, mode)
        d1, d2 = ev.particles()[-2:]
        assert d1.get_id() == 2212
        assert d2.get_id() == 211

    def test_impossible_decay_returns_false(self):
        """Decay fails if sum of daughter masses exceeds parent mass."""
        r = Reaction(158.0)
        ev = make_event()
        parent = Particle.from_id(0.0, 0.0, 0.0, 2224, r)
        parent.set_mass(0.5)  # too light for p + pi
        mode = DecayMode(2, [2212, 211])
        ok = ev._decay_2body(parent, mode)
        assert not ok


class TestEventDecay:
    def _write_decay_file(self, path):
        content = "1\n2224 1\n1 2 2212 0 211 0\n"
        with open(path, "w") as f:
            f.write(content)

    def test_full_event_decay(self):
        """Decay all Δ++ in an event and verify 4-momentum conservation."""
        r = Reaction(158.0)
        ev = Event(rng=RNG)

        # Add 3 Delta++ at various momenta
        initial_e = 0.0
        initial_px = initial_py = initial_pz = 0.0
        for i in range(3):
            pz = 5.0 * (i + 1)
            p = Particle.from_id(0.1 * i, 0.0, pz, 2224, r)
            p.set_mass(1.232)
            initial_e += p.get_e()
            initial_px += p.get_px()
            initial_py += p.get_py()
            initial_pz += p.get_pz()
            ev.add_particle(p)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".decays", delete=False) as f:
            f.write("1\n2224 1\n1 2 2212 0 211 0\n")
            fname = f.name

        try:
            ev.read_decays(fname)
            ev.decay()
        finally:
            os.unlink(fname)

        # After decay there should be 6 daughters (3 × 2)
        assert len(ev.particles()) == 6

        # Total 4-momentum must be conserved
        e_sum = sum(p.get_e() for p in ev.particles())
        px_sum = sum(p.get_px() for p in ev.particles())
        py_sum = sum(p.get_py() for p in ev.particles())
        pz_sum = sum(p.get_pz() for p in ev.particles())

        assert e_sum == pytest.approx(initial_e, rel=1e-4)
        assert px_sum == pytest.approx(initial_px, abs=1e-4)
        assert py_sum == pytest.approx(initial_py, abs=1e-4)
        assert pz_sum == pytest.approx(initial_pz, rel=1e-4)


class TestDecayFile:
    def test_read_write_roundtrip(self):
        """Write a decay file, read it back, and verify consistency."""
        from na49_mixing.breit_wigner import BreitWigner
        original = {
            2224: ParticleDecays(2224),
        }
        mode = DecayMode(2, [2212, 211])
        original[2224].add_mode(1.0, mode)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".dec", delete=False) as f:
            fname = f.name
        try:
            write_decays(fname, original)
            loaded = read_decays(fname)
        finally:
            os.unlink(fname)

        assert 2224 in loaded
        assert loaded[2224].n_modes() == 1
