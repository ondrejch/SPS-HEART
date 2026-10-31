"""Tests for particle kinematics (Particle class)."""
import math
import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from na49_mixing.particle import Particle
from na49_mixing.reaction import Reaction
from na49_mixing.constants import EMPROT, EMPI


class TestParticleBasics:
    def test_default_constructor(self):
        p = Particle()
        assert p.get_px() == 0.0
        assert p.get_py() == 0.0
        assert p.get_pz() == 0.0

    def test_energy_massless(self):
        p = Particle(1.0, 0.0, 0.0, 0.0, 0.0, 22)
        assert p.get_e() == pytest.approx(1.0, rel=1e-9)

    def test_energy_at_rest(self):
        p = Particle(0.0, 0.0, 0.0, 1.0, EMPROT, 2212)
        assert p.get_e() == pytest.approx(EMPROT, rel=1e-9)

    def test_pt(self):
        p = Particle(3.0, 4.0, 0.0, 1.0, 0.0)
        assert p.get_pt() == pytest.approx(5.0, rel=1e-9)

    def test_total_momentum(self):
        p = Particle(1.0, 2.0, 2.0, 0.0, 0.0)
        assert p.get_p() == pytest.approx(3.0, rel=1e-9)

    def test_from_id(self):
        r = Reaction(158.0)
        p = Particle.from_id(0.0, 0.0, 10.0, 2212, r)
        assert p.get_mass() == pytest.approx(EMPROT, rel=1e-5)
        assert p.get_charge() == pytest.approx(1.0)


class TestLorentzBoost:
    def setup_method(self):
        self.r = Reaction(158.0)

    def test_pz_cms_at_rest(self):
        """Proton at rest in the lab should have negative pz_cms."""
        p = Particle.from_id(0.0, 0.0, 0.0, 2212, self.r)
        pz_cms = p.get_pz_cms()
        # A particle at rest in the lab is backward-going in the CMS
        assert pz_cms < 0.0

    def test_lorentz_invariant_mass(self):
        """Invariant mass should be Lorentz-invariant."""
        p = Particle.from_id(1.0, 0.5, 50.0, 2212, self.r)
        m_lab = p.get_mass()
        pz_cms = p.get_pz_cms()
        e_cms = p.get_e_cms()
        m_cms = math.sqrt(e_cms ** 2 - p.get_px() ** 2 - p.get_py() ** 2 - pz_cms ** 2)
        assert m_cms == pytest.approx(m_lab, rel=1e-5)

    def test_xf_beam_proton(self):
        """The beam proton (plab=158, at rest for now) should have xF ≈ −1."""
        r = Reaction(158.0)
        # Stationary target proton
        p = Particle.from_id(0.0, 0.0, 0.0, 2212, r)
        xf = p.get_xf()
        # Target proton moves backward in CMS → xF should be negative
        assert xf < 0.0

    def test_rapidity_real_pion(self):
        r = Reaction(158.0)
        p = Particle(0.2, 0.1, 5.0, 1.0, EMPI, 211, r)
        y = p.get_y_cms()
        # Rapidity should be finite and reasonable
        assert math.isfinite(y)
        assert -5.0 < y < 5.0


class TestInvariantMass:
    def test_two_pion_inv_mass(self):
        """Invariant mass of two identical pions must be > 2*m_pi."""
        p1 = Particle(0.1, 0.0, 1.0, 1.0, EMPI, 211)
        p2 = Particle(-0.1, 0.0, 1.0, -1.0, EMPI, -211)
        m = p1.get_inv_mass(p2)
        assert m > 2.0 * EMPI

    def test_inv_mass_symmetry(self):
        p1 = Particle(0.2, 0.1, 3.0, 1.0, EMPI, 211)
        p2 = Particle(-0.1, 0.2, 2.0, -1.0, EMPI, -211)
        assert p1.get_inv_mass(p2) == pytest.approx(p2.get_inv_mass(p1), rel=1e-9)

    def test_inv_mass_at_rest_pair(self):
        """Two particles at rest: M = m1 + m2."""
        p1 = Particle(0.0, 0.0, 0.0, 1.0, EMPI, 211)
        p2 = Particle(0.0, 0.0, 0.0, -1.0, EMPI, -211)
        m = p1.get_inv_mass(p2)
        assert m == pytest.approx(2.0 * EMPI, rel=1e-9)


class TestParticleAdd:
    def test_add_conservation(self):
        """Adding two particles should conserve 4-momentum."""
        p1 = Particle(1.0, 0.0, 0.0, 0.0, EMPI, 211)
        p2 = Particle(-1.0, 0.0, 0.0, 0.0, EMPI, -211)
        e_before = p1.get_e() + p2.get_e()
        p1.add(p2)
        assert p1.get_e() == pytest.approx(e_before, rel=1e-9)
        assert p1.get_px() == pytest.approx(0.0, abs=1e-12)


class TestBetheBloch:
    def test_minimum_ionising(self):
        """dE/dx should have a minimum around βγ ≈ 3–4."""
        p = Particle()
        vals = [(bg, p.bethe_bloch(bg)) for bg in [1.0, 2.0, 3.0, 4.0, 10.0, 100.0]]
        dedx = [v for _, v in vals]
        min_idx = dedx.index(min(dedx))
        # minimum should be at βγ ≈ 3 (index 2 in our list)
        assert 1 <= min_idx <= 4

    def test_positive_dedx(self):
        p = Particle()
        for bg in [0.5, 1.0, 2.0, 5.0, 10.0]:
            assert p.bethe_bloch(bg) > 0.0


class TestGetPhi:
    def test_positive_charge_along_x(self):
        """Positive particle with px>0, py=0 → phi = 0°."""
        p = Particle(1.0, 0.0, 5.0, 1.0, EMPI, 211)
        assert p.get_phi() == pytest.approx(0.0, abs=1e-9)

    def test_positive_charge_along_y(self):
        """Positive particle with px=0, py>0 → phi = 90°."""
        p = Particle(0.0, 1.0, 5.0, 1.0, EMPI, 211)
        assert p.get_phi() == pytest.approx(90.0, rel=1e-9)

    def test_negative_charge_flips_sign(self):
        """Negative charge negates the x-component in atan2, changing the angle."""
        p_pos = Particle(1.0, 0.0, 5.0, 1.0, EMPI, 211)   # atan2(0, +1) = 0°
        p_neg = Particle(1.0, 0.0, 5.0, -1.0, EMPI, -211)  # atan2(0, -1) = 180°
        assert p_pos.get_phi() == pytest.approx(0.0, abs=1e-9)
        assert p_neg.get_phi() == pytest.approx(180.0, abs=1e-9)

    def test_returns_float_no_exception(self):
        """get_phi() must not raise any exception."""
        p = Particle(0.3, 0.4, 2.0, 1.0, EMPI, 211)
        result = p.get_phi()
        assert math.isfinite(result)

    def test_range(self):
        """Azimuthal angle must be in [-180, 180]."""
        for px, py in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]:
            p = Particle(float(px), float(py), 5.0, 1.0, EMPI, 211)
            phi = p.get_phi()
            assert -180.0 <= phi <= 180.0


class TestGetYCmsEdgeCases:
    def setup_method(self):
        self.r = Reaction(158.0)

    def test_rapidity_finite_for_normal_pion(self):
        p = Particle(0.2, 0.1, 5.0, 1.0, EMPI, 211, self.r)
        y = p.get_y_cms()
        assert math.isfinite(y)

    def test_rapidity_small_pt(self):
        """Particle with very small pt should still give finite rapidity."""
        p = Particle(1e-6, 1e-6, 5.0, 1.0, EMPI, 211, self.r)
        y = p.get_y_cms()
        assert math.isfinite(y)

    def test_rapidity_large_pz_forward(self):
        """Extremely forward particle should return a large positive sentinel."""
        p = Particle(0.0, 0.0, 1e9, 1.0, EMPI, 211, self.r)
        y = p.get_y_cms()
        assert math.isfinite(y)
        assert y > 0.0

    def test_rapidity_large_pz_backward(self):
        """Extremely backward particle should return a large negative sentinel."""
        p = Particle(0.0, 0.0, -1e9, 1.0, EMPI, 211, self.r)
        y = p.get_y_cms()
        assert math.isfinite(y)
        assert y < 0.0

    def test_rapidity_does_not_raise(self):
        """Domain guard prevents math errors for degenerate momenta."""
        p = Particle(0.0, 0.0, 0.0, 1.0, EMPI, 211, self.r)
        y = p.get_y_cms()
        assert math.isfinite(y)
