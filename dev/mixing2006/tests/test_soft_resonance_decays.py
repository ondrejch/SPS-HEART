"""Tests for the PDG-grounded soft-resonance decay library."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from na49_mixing.decays import build_soft_resonance_decays
from na49_mixing.event import Event
from na49_mixing.particle import Particle
from na49_mixing.pdg_database import get_particle
from na49_mixing.reaction import Reaction


class TestSoftResonanceFactory:
    def test_new_pdg_entries_exist(self):
        for pdg_id in (
            117, -217, 217, 229,
            10113, 20113, 115,
            102112, 122112, 212114,
            202224, 122224,
            102134, 3224, 3214, 3114,
            9911201, 9921201, 9931201,
            9912201, 9922201, 9932201,
        ):
            p = get_particle(pdg_id)
            assert p is not None

    def test_includes_table9_missing_mesons(self):
        decays = build_soft_resonance_decays(rng=np.random.default_rng(0))
        for pdg_id in (221, -213, 225, 117, -217, 229, 10113, 20113, 115):
            assert pdg_id in decays

    def test_f2_contains_kaon_channels(self):
        decays = build_soft_resonance_decays(rng=np.random.default_rng(1))
        daughter_sets = {tuple(mode.daughters) for mode in decays[225]._modes}
        assert (321, -321) in daughter_sets
        assert (310, 130) in daughter_sets

    def test_rho3_minus_contains_omega_pi_minus(self):
        decays = build_soft_resonance_decays(rng=np.random.default_rng(2))
        daughter_sets = {tuple(mode.daughters) for mode in decays[-217]._modes}
        assert (223, -211) in daughter_sets

    def test_eta_contains_hadronic_three_pion_mode(self):
        decays = build_soft_resonance_decays(rng=np.random.default_rng(3))
        daughter_sets = {tuple(mode.daughters) for mode in decays[221]._modes}
        assert (211, -211, 111) in daughter_sets

    def test_includes_new_baryon_resonance_families(self):
        decays = build_soft_resonance_decays(rng=np.random.default_rng(5))
        for pdg_id in (
            102112, 102212, 122112, 122212, 212114, 212214,
            202224, 202214, 202114, 201114,
            122224, 122214, 122114, 121114,
            9911201, 9921201, 9931201,
            9912201, 9922201, 9932201,
            102134, 3224, 3214, 3114,
            3122, -3122, 3222, 3212, 3112,
        ):
            assert pdg_id in decays

    def test_lambda1520_contains_nkbar_and_sigmapi(self):
        decays = build_soft_resonance_decays(rng=np.random.default_rng(6))
        daughter_sets = {tuple(mode.daughters) for mode in decays[102134]._modes}
        assert (2212, -321) in daughter_sets
        assert (3222, -211) in daughter_sets

    def test_a1_zero_contains_rho_pi_modes(self):
        decays = build_soft_resonance_decays(rng=np.random.default_rng(7))
        daughter_sets = {tuple(mode.daughters) for mode in decays[20113]._modes}
        assert (213, -211) in daughter_sets
        assert (-213, 211) in daughter_sets

    def test_branching_override_changes_nstar_pi_minus_expectation(self):
        baseline = build_soft_resonance_decays(rng=np.random.default_rng(4))
        tuned = build_soft_resonance_decays(
            rng=np.random.default_rng(4),
            branching_overrides={"n1440_br_npi": 0.55, "n1440_br_deltapi": 0.27},
        )
        baseline_modes = [tuple(m.daughters) for m in baseline[12112]._modes]
        tuned_modes = [tuple(m.daughters) for m in tuned[12112]._modes]
        assert baseline_modes == tuned_modes
        assert baseline[12112]._probabilities != tuned[12112]._probabilities


class TestSoftResonanceCascade:
    def test_f4_to_eta_eta_cascades_to_two_triplets(self):
        rng = np.random.default_rng(42)
        reaction = Reaction(158.0)
        ev = Event(rng=rng)

        parent = Particle.from_id(0.0, 0.0, 8.0, 229, reaction)
        parent.set_mass(2.05)
        ev.add_particle(parent)

        decays = build_soft_resonance_decays(rng=rng)
        # Force f4 -> eta eta.
        decays[229]._modes = [decays[229]._modes[-1]]
        decays[229]._probabilities = [1.0]
        # Force eta -> pi+ pi- pi0.
        decays[221]._modes = [decays[221]._modes[0]]
        decays[221]._probabilities = [1.0]

        ev._decay_chain = decays
        ev.decay_cascade(max_passes=6)

        ids = sorted(p.get_id() for p in ev.particles())
        assert ids == [-211, -211, 111, 111, 211, 211]

    def test_sigma1385_plus_cascades_via_sigma0_and_lambda(self):
        rng = np.random.default_rng(43)
        reaction = Reaction(158.0)
        ev = Event(rng=rng)

        parent = Particle.from_id(0.0, 0.0, 8.0, 3224, reaction)
        parent.set_mass(1.385)
        ev.add_particle(parent)

        decays = build_soft_resonance_decays(rng=rng)
        # Force Sigma(1385)+ -> Sigma0 pi+
        sigstar_modes = decays[3224]._modes
        idx_sigma0_pi = next(i for i, m in enumerate(sigstar_modes) if tuple(m.daughters) == (3212, 211))
        decays[3224]._modes = [sigstar_modes[idx_sigma0_pi]]
        decays[3224]._probabilities = [1.0]
        # Force Sigma0 -> Lambda gamma
        decays[3212]._probabilities = [1.0]
        # Force Lambda -> p pi-
        decays[3122]._modes = [decays[3122]._modes[0]]
        decays[3122]._probabilities = [1.0]

        ev._decay_chain = decays
        ev.decay_cascade(max_passes=8)

        ids = sorted(p.get_id() for p in ev.particles())
        assert ids == [-211, 22, 211, 2212]

    def test_effective_nstar_tower_cascades_down_levels(self):
        rng = np.random.default_rng(44)
        reaction = Reaction(158.0)
        ev = Event(rng=rng)

        parent = Particle.from_id(0.0, 0.0, 9.0, 9931201, reaction)
        parent.set_mass(2.30)
        ev.add_particle(parent)

        decays = build_soft_resonance_decays(rng=rng)

        # Force 2.3 -> 2.1 + pi0
        m3 = decays[9931201]._modes
        i3 = next(i for i, m in enumerate(m3) if tuple(m.daughters) == (9921201, 111))
        decays[9931201]._modes = [m3[i3]]
        decays[9931201]._probabilities = [1.0]

        # Force 2.1 -> 1.9 + pi0
        m2 = decays[9921201]._modes
        i2 = next(i for i, m in enumerate(m2) if tuple(m.daughters) == (9911201, 111))
        decays[9921201]._modes = [m2[i2]]
        decays[9921201]._probabilities = [1.0]

        # Force 1.9 -> p + pi-
        m1 = decays[9911201]._modes
        i1 = next(i for i, m in enumerate(m1) if tuple(m.daughters) == (2212, -211))
        decays[9911201]._modes = [m1[i1]]
        decays[9911201]._probabilities = [1.0]

        ev._decay_chain = decays
        ev.decay_cascade(max_passes=10)

        ids = sorted(p.get_id() for p in ev.particles())
        assert ids == [-211, 111, 111, 2212]
