"""Tests for rho/omega resonances: PDG database entries, decay factories,
and invariant-mass physics."""
import math
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from na49_mixing.pdg_database import get_particle, get_mass, get_charge, get_width
from na49_mixing.decays import build_rho_decays, build_omega_decays, DecayMode
from na49_mixing.event import Event
from na49_mixing.particle import Particle
from na49_mixing.reaction import Reaction
from na49_mixing import EMRHO, EMOMEGA, EMETA, EMPI

RNG = np.random.default_rng(12345)


# ---------------------------------------------------------------------------
# PDG database: new resonance entries
# ---------------------------------------------------------------------------

class TestPDGNewResonances:
    """All new rho/omega PDG entries must have correct mass, charge, width."""

    # rho(1450)
    def test_rho1450_neutral(self):
        p = get_particle(100113)
        assert p is not None
        assert p.mass == pytest.approx(1.465, abs=1e-6)
        assert p.charge == pytest.approx(0.0)
        assert p.width == pytest.approx(0.400, abs=1e-6)

    def test_rho1450_plus(self):
        p = get_particle(100213)
        assert p is not None
        assert p.mass == pytest.approx(1.465, abs=1e-6)
        assert p.charge == pytest.approx(1.0)
        assert p.width == pytest.approx(0.400, abs=1e-6)

    def test_rho1450_minus(self):
        p = get_particle(-100213)
        assert p is not None
        assert p.mass == pytest.approx(1.465, abs=1e-6)
        assert p.charge == pytest.approx(-1.0)
        assert p.width == pytest.approx(0.400, abs=1e-6)

    # rho(1700)
    def test_rho1700_neutral(self):
        p = get_particle(30113)
        assert p is not None
        assert p.mass == pytest.approx(1.720, abs=1e-6)
        assert p.charge == pytest.approx(0.0)
        assert p.width == pytest.approx(0.250, abs=1e-6)

    def test_rho1700_plus(self):
        p = get_particle(30213)
        assert p is not None
        assert p.mass == pytest.approx(1.720, abs=1e-6)
        assert p.charge == pytest.approx(1.0)
        assert p.width == pytest.approx(0.250, abs=1e-6)

    def test_rho1700_minus(self):
        p = get_particle(-30213)
        assert p is not None
        assert p.mass == pytest.approx(1.720, abs=1e-6)
        assert p.charge == pytest.approx(-1.0)
        assert p.width == pytest.approx(0.250, abs=1e-6)

    # omega(1420)
    def test_omega1420(self):
        p = get_particle(100223)
        assert p is not None
        assert p.mass == pytest.approx(1.410, abs=1e-6)
        assert p.charge == pytest.approx(0.0)
        assert p.width == pytest.approx(0.290, abs=1e-6)

    # omega(1650)
    def test_omega1650(self):
        p = get_particle(30223)
        assert p is not None
        assert p.mass == pytest.approx(1.670, abs=1e-6)
        assert p.charge == pytest.approx(0.0)
        assert p.width == pytest.approx(0.315, abs=1e-6)


class TestPDGWidthsExisting:
    """Existing resonances now carry non-zero widths."""

    def test_rho770_width(self):
        assert get_width(113) == pytest.approx(0.1491, abs=1e-6)

    def test_omega782_width(self):
        assert get_width(223) == pytest.approx(0.00849, abs=1e-6)

    def test_delta_width(self):
        assert get_width(2224) == pytest.approx(0.117, abs=1e-6)

    def test_unknown_particle_width_default(self):
        assert get_width(999999) == 0.0


# ---------------------------------------------------------------------------
# build_rho_decays
# ---------------------------------------------------------------------------

class TestBuildRhoDecays:
    """Factory function build_rho_decays() returns correct structure."""

    def test_returns_expected_parent_ids(self):
        d = build_rho_decays(rng=np.random.default_rng(0))
        assert set(d.keys()) == {113, 100113, 30113}

    def test_rho770_one_mode(self):
        d = build_rho_decays(rng=np.random.default_rng(0))
        assert d[113].n_modes() == 1

    def test_rho1450_one_mode(self):
        d = build_rho_decays(rng=np.random.default_rng(0))
        assert d[100113].n_modes() == 1

    def test_rho1700_one_mode(self):
        d = build_rho_decays(rng=np.random.default_rng(0))
        assert d[30113].n_modes() == 1

    def test_rho770_daughters_are_pions(self):
        d = build_rho_decays(rng=np.random.default_rng(0))
        mode = d[113]._modes[0]
        assert set(mode.daughters) == {211, -211}

    def test_rho770_no_bw_on_daughters(self):
        d = build_rho_decays(rng=np.random.default_rng(0))
        mode = d[113]._modes[0]
        assert all(bw is None for bw in mode.bws)


# ---------------------------------------------------------------------------
# build_omega_decays
# ---------------------------------------------------------------------------

class TestBuildOmegaDecays:
    """Factory function build_omega_decays() returns correct structure."""

    def test_returns_expected_parent_ids(self):
        d = build_omega_decays(rng=np.random.default_rng(0))
        assert set(d.keys()) == {223, 100223, 30223}

    def test_omega782_two_modes(self):
        d = build_omega_decays(rng=np.random.default_rng(0))
        assert d[223].n_modes() == 2

    def test_omega782_mode1_is_3body(self):
        d = build_omega_decays(rng=np.random.default_rng(0))
        mode = d[223]._modes[0]
        assert mode.n_daughters == 3

    def test_omega782_mode2_is_2body(self):
        d = build_omega_decays(rng=np.random.default_rng(0))
        mode = d[223]._modes[1]
        assert mode.n_daughters == 2

    def test_omega1420_one_mode(self):
        d = build_omega_decays(rng=np.random.default_rng(0))
        assert d[100223].n_modes() == 1

    def test_omega1420_daughter_is_rho(self):
        d = build_omega_decays(rng=np.random.default_rng(0))
        mode = d[100223]._modes[0]
        assert 113 in mode.daughters

    def test_omega1420_rho_daughter_has_bw(self):
        d = build_omega_decays(rng=np.random.default_rng(0))
        mode = d[100223]._modes[0]
        rho_idx = mode.daughters.index(113)
        assert mode.bws[rho_idx] is not None

    def test_omega1650_one_mode(self):
        d = build_omega_decays(rng=np.random.default_rng(0))
        assert d[30223].n_modes() == 1


# ---------------------------------------------------------------------------
# Decay kinematics
# ---------------------------------------------------------------------------

class TestRhoDecayKinematics:
    """ρ⁰(770) → π⁺π⁻ decay kinematics."""

    def _make_rho_event(self, mass, rng=None):
        rng = rng or np.random.default_rng(42)
        r = Reaction(158.0)
        ev = Event(rng=rng)
        rho = Particle(0.0, 0.0, 5.0, 0.0, mass, 113, r)
        ev.add_particle(rho)
        rho_decays = build_rho_decays(rng=rng)
        ev._decay_chain = rho_decays
        ev.decay()
        return ev

    def test_rho_decay_produces_two_daughters(self):
        ev = self._make_rho_event(0.77526)
        assert len(ev.particles()) == 2

    def test_rho_decay_daughter_ids(self):
        ev = self._make_rho_event(0.77526)
        ids = {p.get_id() for p in ev.particles()}
        assert ids == {211, -211}

    def test_rho_decay_4momentum_conservation(self):
        rng = np.random.default_rng(7)
        r = Reaction(158.0)
        ev = Event(rng=rng)
        mass = 0.77526
        px0, py0, pz0 = 0.1, 0.05, 8.0
        rho = Particle(px0, py0, pz0, 0.0, mass, 113, r)
        e0 = rho.get_e()
        ev.add_particle(rho)
        ev._decay_chain = build_rho_decays(rng=rng)
        ev.decay()
        daughters = ev.particles()
        assert len(daughters) == 2
        assert sum(p.get_px() for p in daughters) == pytest.approx(px0, abs=1e-5)
        assert sum(p.get_py() for p in daughters) == pytest.approx(py0, abs=1e-5)
        assert sum(p.get_pz() for p in daughters) == pytest.approx(pz0, abs=1e-5)
        assert sum(p.get_e() for p in daughters) == pytest.approx(e0, rel=1e-5)


class TestOmegaDecayKinematics:
    """ω(782) 3-body decay produces three daughters."""

    def test_omega_3body_produces_three_daughters(self):
        rng = np.random.default_rng(99)
        r = Reaction(158.0)
        ev = Event(rng=rng)
        # Use a mass safely above threshold (3 m_pi)
        omega = Particle(0.0, 0.0, 5.0, 0.0, 0.78265, 223, r)
        ev.add_particle(omega)
        omega_decays = build_omega_decays(rng=rng)
        # Force mode 1 (3-body) by setting cumulative prob so the first pick wins
        omega_decays[223]._probabilities = [1.0, 1.0]
        ev._decay_chain = omega_decays
        ev.decay()
        assert len(ev.particles()) == 3


# ---------------------------------------------------------------------------
# Invariant-mass physics check
# ---------------------------------------------------------------------------

class TestRhoInvariantMass:
    """Invariant mass of π⁺π⁻ from ρ⁰ decay must peak near the ρ mass."""

    def test_inv_mass_peak_near_rho_mass(self):
        """Invariant masses from ρ⁰→π⁺π⁻ must be in the rho mass region.

        Because inv_mass(π⁺π⁻) = parent mass exactly (4-momentum conservation),
        the distribution of invariant masses follows the sampled BW distribution.
        For the wide ρ resonance the BW mean lies above the pole; we therefore
        check that the median is within 200 MeV of 0.775 GeV and all masses
        are kinematically above 2·m_π.
        """
        rng = np.random.default_rng(2024)
        r = Reaction(158.0)
        masses_inv = []
        from na49_mixing.breit_wigner import BreitWigner
        bw_proto = BreitWigner(0.2791, 0.77526, 0.1491, 1.0, 1.5, 2.0, rng=rng)
        rho_decays = build_rho_decays(rng=rng)
        for _ in range(2000):
            rho_mass = bw_proto.get_random()
            ev = Event(rng=rng)
            rho = Particle(0.0, 0.0, 5.0, 0.0, rho_mass, 113, r)
            ev.add_particle(rho)
            ev._decay_chain = rho_decays
            ev.decay()
            daughters = ev.particles()
            if len(daughters) == 2:
                pi_plus = next((p for p in daughters if p.get_id() == 211), None)
                pi_minus = next((p for p in daughters if p.get_id() == -211), None)
                if pi_plus and pi_minus:
                    m_inv = pi_plus.get_inv_mass(pi_minus)
                    masses_inv.append(m_inv)

        assert len(masses_inv) > 100, "Too few successful decays"
        # All invariant masses must be above the kinematic threshold 2·m_π
        assert all(m > 2.0 * EMPI for m in masses_inv)
        # Median must be within 200 MeV of the ρ(770) pole mass
        median_m = float(np.median(masses_inv))
        assert abs(median_m - 0.77526) < 0.200


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestMesonConstants:
    """New meson mass constants must be importable and numerically correct."""

    def test_emrho_value(self):
        assert EMRHO == pytest.approx(0.77526, abs=1e-6)

    def test_emomega_value(self):
        assert EMOMEGA == pytest.approx(0.78265, abs=1e-6)

    def test_emeta_value(self):
        assert EMETA == pytest.approx(0.547862, abs=1e-6)

    def test_emrho_matches_pdg_mass(self):
        assert EMRHO == pytest.approx(get_mass(113), abs=1e-6)

    def test_emomega_matches_pdg_mass(self):
        assert EMOMEGA == pytest.approx(get_mass(223), abs=1e-6)
