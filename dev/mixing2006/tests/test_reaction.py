"""Tests for Lorentz kinematics (Reaction class).

Validates against known values for NA49 Pb+Pb at 158 GeV/c.
"""
import math
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from na49_mixing.reaction import Reaction
from na49_mixing.constants import EMPROT, EMNEUT


def _reference_sqrt_s(plab: float, m_beam: float, m_target: float) -> float:
    """Independent fixed-target reference: s = m_b² + m_t² + 2 m_t E_b."""
    e_beam = math.sqrt(m_beam ** 2 + plab ** 2)
    return math.sqrt(m_beam ** 2 + m_target ** 2 + 2.0 * m_target * e_beam)


class TestReaction:
    def test_default_plab(self):
        r = Reaction()
        assert r.plab() == pytest.approx(158.0)

    def test_sqrt_s_at_158(self):
        r = Reaction(158.0)
        # From C++ formula: fS = sqrt(2*EMPROT*(EMPROT+sqrt(EMPROT^2+plab^2)))
        plab = 158.0
        expected = math.sqrt(
            2.0 * EMPROT * (EMPROT + math.sqrt(EMPROT ** 2 + plab ** 2))
        )
        assert r.sqrt_s() == pytest.approx(expected, rel=1e-9)

    def test_s_equals_sqrts_squared(self):
        r = Reaction(158.0)
        assert r.s() == pytest.approx(r.sqrt_s() ** 2, rel=1e-9)

    def test_beta_less_than_one(self):
        r = Reaction(158.0)
        assert 0.0 < r.betha() < 1.0

    def test_gamma_greater_than_one(self):
        r = Reaction(158.0)
        assert r.gamma() > 1.0

    def test_beta_gamma_relation(self):
        """β·γ must satisfy γ = 1/√(1−β²)."""
        r = Reaction(158.0)
        beta = r.betha()
        gamma_calc = 1.0 / math.sqrt(1.0 - beta ** 2)
        assert r.gamma() == pytest.approx(gamma_calc, rel=1e-9)

    def test_bet_gam(self):
        r = Reaction(158.0)
        assert r.bet_gam() == pytest.approx(r.betha() * r.gamma(), rel=1e-9)

    def test_plmax(self):
        r = Reaction(158.0)
        assert r.plmax() == pytest.approx(r.sqrt_s() / 2.0, rel=1e-9)

    def test_known_sqrt_s(self):
        """Known approximate value: √s ≈ 17.27 GeV at plab=158 GeV/c."""
        r = Reaction(158.0)
        assert r.sqrt_s() == pytest.approx(17.27, abs=0.1)

    def test_custom_plab(self):
        r = Reaction(80.0)
        assert r.plab() == pytest.approx(80.0)
        # Must be smaller than at 158 GeV/c
        r158 = Reaction(158.0)
        assert r.sqrt_s() < r158.sqrt_s()

    def test_pdg_lookup_proton(self):
        r = Reaction(158.0)
        p = r.get_base_particle(2212)
        assert p is not None
        assert p.mass == pytest.approx(EMPROT, rel=1e-5)

    def test_pdg_lookup_pion(self):
        from na49_mixing.constants import EMPI
        r = Reaction(158.0)
        p = r.get_base_particle(211)
        assert p is not None
        assert p.mass == pytest.approx(EMPI, rel=1e-5)

    def test_pdg_lookup_unknown(self):
        r = Reaction(158.0)
        assert r.get_base_particle(99999) is None

    def test_energy_momentum_conservation(self):
        """Total energy should equal beam + target at rest."""
        r = Reaction(158.0)
        plab = r.plab()
        e_beam = math.sqrt(EMPROT ** 2 + plab ** 2)
        e_target = EMPROT
        e_tot = e_beam + e_target
        beta_calc = plab / e_tot
        assert r.betha() == pytest.approx(beta_calc, rel=1e-9)


class TestSpeciesAwareReaction:
    """n+p / n+n reactions via beam_pdg / target_pdg (Phase 1, np plan)."""

    PROTON = 2212
    NEUTRON = 2112

    def test_default_is_proton_proton(self):
        r = Reaction(158.0)
        assert r.beam_pdg() == self.PROTON
        assert r.target_pdg() == self.PROTON
        assert r.beam_mass() == pytest.approx(EMPROT, rel=1e-12)
        assert r.target_mass() == pytest.approx(EMPROT, rel=1e-12)

    def test_default_sqrt_s_bitwise_legacy(self):
        """Default path must reproduce the legacy closed form exactly."""
        r = Reaction(158.0)
        plab = 158.0
        expected = math.sqrt(
            2.0 * EMPROT * (EMPROT + math.sqrt(EMPROT * EMPROT + plab * plab))
        )
        assert r.sqrt_s() == expected  # exact float equality, no tolerance

    @pytest.mark.parametrize("plab", [158.0, 20.0, 80.0])
    def test_np_sqrt_s_reference(self, plab):
        r = Reaction(plab, beam_pdg=self.NEUTRON, target_pdg=self.PROTON)
        expected = _reference_sqrt_s(plab, EMNEUT, EMPROT)
        assert r.sqrt_s() == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("plab", [158.0, 20.0, 80.0])
    def test_nn_sqrt_s_reference(self, plab):
        r = Reaction(plab, beam_pdg=self.NEUTRON, target_pdg=self.NEUTRON)
        expected = _reference_sqrt_s(plab, EMNEUT, EMNEUT)
        assert r.sqrt_s() == pytest.approx(expected, rel=1e-12)

    def test_np_nn_known_values_at_158(self):
        """Loose anchors: √s(np) ≈ 17.2703 GeV, √s(nn) ≈ 17.2821 GeV."""
        np_r = Reaction(158.0, beam_pdg=self.NEUTRON, target_pdg=self.PROTON)
        nn_r = Reaction(158.0, beam_pdg=self.NEUTRON, target_pdg=self.NEUTRON)
        pp_r = Reaction(158.0)
        assert np_r.sqrt_s() == pytest.approx(17.2703, abs=5e-4)
        assert nn_r.sqrt_s() == pytest.approx(17.2821, abs=5e-4)
        # Heavier nucleons raise √s monotonically.
        assert pp_r.sqrt_s() < np_r.sqrt_s() < nn_r.sqrt_s()

    def test_np_boost_consistency(self):
        """β = p_lab / (E_beam + m_target) holds for n+p as well."""
        r = Reaction(158.0, beam_pdg=self.NEUTRON, target_pdg=self.PROTON)
        e_beam = math.sqrt(EMNEUT ** 2 + 158.0 ** 2)
        beta_expected = 158.0 / (e_beam + EMPROT)
        assert r.betha() == pytest.approx(beta_expected, rel=1e-12)
        gamma_calc = 1.0 / math.sqrt(1.0 - r.betha() ** 2)
        assert r.gamma() == pytest.approx(gamma_calc, rel=1e-12)
        assert r.plmax() == pytest.approx(r.sqrt_s() / 2.0, rel=1e-12)

    def test_np_energy_tot(self):
        r = Reaction(158.0, beam_pdg=self.NEUTRON, target_pdg=self.PROTON)
        e_beam = math.sqrt(EMNEUT ** 2 + 158.0 ** 2)
        assert r._energy_tot == pytest.approx(e_beam + EMPROT, rel=1e-12)

    def test_swapped_species_asymmetric_sqrt_s_at_fixed_plab(self):
        """At fixed lab momentum, beam/target swap does NOT preserve √s.

        The target rest mass enters s linearly weighted by the full beam
        energy, while the beam-mass contribution is suppressed by ~p_lab,
        so p-beam-on-n-target has larger √s than n-beam-on-p-target. The
        NA49 inverse-kinematics configuration is n-beam on hydrogen, i.e.
        beam_pdg=2112 with target_pdg=2212.
        """
        n_beam = Reaction(158.0, beam_pdg=self.NEUTRON, target_pdg=self.PROTON)
        p_beam = Reaction(158.0, beam_pdg=self.PROTON, target_pdg=self.NEUTRON)
        assert p_beam.sqrt_s() > n_beam.sqrt_s()
        # Both stay within a percent of each other and of the pp value.
        assert p_beam.sqrt_s() == pytest.approx(n_beam.sqrt_s(), rel=1e-3)

    @pytest.mark.parametrize("plab", [20.0, 80.0, 158.0])
    def test_mixed_species_sqrt_s_ordering(self, plab):
        """Mixed-species results order strictly between the equal-mass cases."""
        pp_r = Reaction(plab)
        nn_r = Reaction(plab, beam_pdg=self.NEUTRON, target_pdg=self.NEUTRON)
        np_r = Reaction(plab, beam_pdg=self.NEUTRON, target_pdg=self.PROTON)
        pn_r = Reaction(plab, beam_pdg=self.PROTON, target_pdg=self.NEUTRON)
        assert pp_r.sqrt_s() < np_r.sqrt_s() < pn_r.sqrt_s() < nn_r.sqrt_s()

    def test_unknown_beam_pdg_raises(self):
        with pytest.raises(ValueError):
            Reaction(158.0, beam_pdg=99999)

    def test_unknown_target_pdg_raises(self):
        with pytest.raises(ValueError):
            Reaction(158.0, target_pdg=12345)
