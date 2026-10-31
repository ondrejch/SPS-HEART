"""Round-2 audit tests: seeded daughter spectra and primary decay bookkeeping."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from na49_mixing.decays import build_soft_resonance_decays
from na49_mixing.event import Event
from na49_mixing.particle import Particle
from na49_mixing.pdg_database import get_mass
from na49_mixing.reaction import Reaction


class TestSeededSpectralFunctions:
    def test_soft_cascade_reproducible_with_seeded_builder(self):
        def run_once():
            # Fresh builder seed per run: shared daughter Breit-Wigner objects
            # must draw from this stream, not from an internal unseeded one.
            decays = build_soft_resonance_decays(rng=np.random.default_rng(123))
            reaction = Reaction(158.0)
            ev = Event(rng=np.random.default_rng(777))
            ev._decay_chain = decays
            for pid in (212214, 9912201):
                mass = get_mass(pid)
                for _ in range(300):
                    parent = Particle.from_id(0.0, 0.0, 8.0, pid, reaction)
                    parent.set_mass(mass)
                    ev.add_particle(parent)
            ev.decay_cascade(max_passes=8)
            return sorted(
                (p.get_id(), round(p.get_pt(), 9), round(p.get_xf(reaction), 9))
                for p in ev.particles()
            )

        assert run_once() == run_once()


class TestPrimaryDecayBookkeeping:
    def test_all_primaries_decayed_counters(self):
        rng = np.random.default_rng(11)
        reaction = Reaction(158.0)
        ev = Event(rng=rng)
        ev._decay_chain = build_soft_resonance_decays(rng=rng)

        mass = get_mass(2224)
        for _ in range(40):
            parent = Particle.from_id(0.0, 0.0, 8.0, 2224, reaction)
            parent.set_mass(mass)
            ev.add_particle(parent)
            ev.mark_primary(parent)

        ev.decay_cascade(max_passes=8)

        stats = ev.decay_stats()
        assert stats["n_primary_generated"] == 40
        assert stats["n_primary_decayed"] == 40
        assert stats["primary_pdg_stats"][2224] == [40, 40]

    def test_mixed_primaries_and_all_cascade_counter(self):
        rng = np.random.default_rng(12)
        reaction = Reaction(158.0)
        ev = Event(rng=rng)
        ev._decay_chain = build_soft_resonance_decays(rng=rng)

        for pid in (3122, 2224):
            parent = Particle.from_id(0.0, 0.0, 8.0, pid, reaction)
            parent.set_mass(get_mass(pid))
            ev.add_particle(parent)
            ev.mark_primary(parent)

        ev.decay_cascade(max_passes=8)

        stats = ev.decay_stats()
        assert stats["n_primary_generated"] == 2
        assert stats["n_primary_decayed"] == 2
        # The all-cascade counter also includes secondary decays, so it can
        # only exceed or match the primary-only count.
        assert stats["n_decayed_parents"] >= stats["n_primary_decayed"]

    def test_primary_flag_survives_copy_and_daughters_unflagged(self):
        reaction = Reaction(158.0)
        parent = Particle.from_id(0.0, 0.0, 8.0, 2224, reaction)
        assert parent.get_is_primary() == 0

        child = Particle.copy(parent)
        assert child.get_is_primary() == 0

        parent.set_is_primary(1)
        assert parent.get_is_primary() == 1
        child = Particle.copy(parent)
        assert child.get_is_primary() == 1

        truthy = Particle()
        truthy.set_is_primary(True)
        assert truthy.get_is_primary() == 1
