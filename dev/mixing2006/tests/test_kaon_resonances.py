"""Tests for kaon resonances and multi-pass decay cascading."""
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from na49_mixing.decays import build_kaon_decays, ParticleDecays, DecayMode
from na49_mixing.event import Event
from na49_mixing.particle import Particle
from na49_mixing.pdg_database import get_particle
from na49_mixing.reaction import Reaction


class TestPDGKaonResonances:
    def test_kstar_plus_entry(self):
        p = get_particle(323)
        assert p is not None
        assert p.mass == pytest.approx(0.89166, abs=1e-6)
        assert p.charge == pytest.approx(1.0)

    def test_kstar_zero_entry(self):
        p = get_particle(313)
        assert p is not None
        assert p.mass == pytest.approx(0.89555, abs=1e-6)
        assert p.charge == pytest.approx(0.0)

    def test_phi_entry(self):
        p = get_particle(333)
        assert p is not None
        assert p.mass == pytest.approx(1.019461, abs=1e-6)
        assert p.charge == pytest.approx(0.0)


class TestKaonDecayFactory:
    def test_expected_parent_ids(self):
        d = build_kaon_decays(rng=np.random.default_rng(0))
        assert set(d.keys()) == {
            323, -323, 313, -313, 333,
            10323, -10323, 10313, -10313,
            325, -325, 315, -315,
        }

    def test_kstar0_mode_to_kplus_piminus(self):
        rng = np.random.default_rng(10)
        decays = build_kaon_decays(rng=rng)
        # Force mode 1 for deterministic daughter IDs.
        decays[313]._probabilities = [1.0, 1.0]

        ev = Event(rng=rng)
        r = Reaction(158.0)
        parent = Particle(0.0, 0.0, 5.0, 0.0, 0.89555, 313, r)
        ev.add_particle(parent)
        ev._decay_chain = decays
        ev.decay()

        ids = sorted(p.get_id() for p in ev.particles())
        assert ids == [-211, 321]

    def test_k1plus_contains_rho_k_and_kstar_pi_channels(self):
        decays = build_kaon_decays(rng=np.random.default_rng(11))
        daughter_sets = {tuple(m.daughters) for m in decays[10323]._modes}
        assert (213, 311) in daughter_sets   # rho+ K0
        assert (313, 211) in daughter_sets   # K*0 pi+

    def test_k2star_zero_contains_direct_kpi_channel(self):
        decays = build_kaon_decays(rng=np.random.default_rng(12))
        daughter_sets = {tuple(m.daughters) for m in decays[315]._modes}
        assert (321, -211) in daughter_sets


class TestDecayCascade:
    def _chain_with_intermediate_delta(self, rng):
        d_delta = ParticleDecays(2224, rng=rng)
        d_delta.add_mode(1.0, DecayMode(2, [2212, 211]))

        d_nstar = ParticleDecays(12112, rng=rng)
        d_nstar.add_mode(1.0, DecayMode(2, [2224, -211]))

        # Intentionally reverse order so single-pass decay leaves Delta undecayed.
        return {2224: d_delta, 12112: d_nstar}

    def test_decay_cascade_resolves_intermediate_resonance(self):
        rng = np.random.default_rng(123)
        r = Reaction(158.0)
        ev = Event(rng=rng)

        parent = Particle.from_id(0.0, 0.0, 6.0, 12112, r)
        parent.set_mass(1.68)
        ev.add_particle(parent)
        ev._decay_chain = self._chain_with_intermediate_delta(rng)

        ev.decay()
        ids_after_single_pass = sorted(p.get_id() for p in ev.particles())
        assert ids_after_single_pass == [-211, 2224]

        n_decayed = ev.decay_cascade(max_passes=4)
        ids_after_cascade = sorted(p.get_id() for p in ev.particles())

        assert n_decayed >= 1
        assert ids_after_cascade == [-211, 211, 2212]
