"""Tests for ParticleCut, PairCut, and EventCut."""
import math
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from na49_mixing.particle import Particle
from na49_mixing.cuts import ParticleCut, PairCut
from na49_mixing.constants import EMPI
from na49_mixing.reaction import Reaction


class TestParticleCutEarlyExit:
    """Verify ParticleCut.check_particle() correctness after early-exit refactor."""

    def test_no_cuts_passes(self):
        cut = ParticleCut()
        p = Particle(1.0, 0.0, 5.0, 1.0, EMPI, 211)
        assert cut.check_particle(p) is True

    def test_id_cut_pass(self):
        cut = ParticleCut()
        cut.set_id_cut(211)
        p = Particle(1.0, 0.0, 5.0, 1.0, EMPI, 211)
        assert cut.check_particle(p) is True

    def test_id_cut_fail(self):
        cut = ParticleCut()
        cut.set_id_cut(211)
        p = Particle(1.0, 0.0, 5.0, -1.0, EMPI, -211)
        assert cut.check_particle(p) is False

    def test_pt_cut_pass(self):
        cut = ParticleCut()
        cut.set_pt_cut(0.0, 1.0)
        p = Particle(0.3, 0.4, 5.0, 1.0, EMPI, 211)  # pt = 0.5
        assert cut.check_particle(p) is True

    def test_pt_cut_fail(self):
        cut = ParticleCut()
        cut.set_pt_cut(0.0, 0.3)
        p = Particle(0.3, 0.4, 5.0, 1.0, EMPI, 211)  # pt = 0.5
        assert cut.check_particle(p) is False

    def test_multiple_cuts_all_pass(self):
        cut = ParticleCut()
        cut.set_id_cut(211)
        cut.set_pt_cut(0.0, 1.0)
        p = Particle(0.3, 0.4, 5.0, 1.0, EMPI, 211)
        assert cut.check_particle(p) is True

    def test_multiple_cuts_first_fails(self):
        """Particle fails the id cut; remaining cuts should not matter."""
        cut = ParticleCut()
        cut.set_id_cut(211)
        cut.set_pt_cut(0.0, 1.0)
        p = Particle(0.3, 0.4, 5.0, -1.0, EMPI, -211)
        assert cut.check_particle(p) is False

    def test_multiple_cuts_second_fails(self):
        """Particle passes id cut but fails pt cut."""
        cut = ParticleCut()
        cut.set_id_cut(211)
        cut.set_pt_cut(0.0, 0.3)
        p = Particle(0.3, 0.4, 5.0, 1.0, EMPI, 211)  # pt = 0.5
        assert cut.check_particle(p) is False

    def test_pz_cut_pass(self):
        cut = ParticleCut()
        cut.set_pz_cut(1.0, 10.0)
        p = Particle(0.1, 0.1, 5.0, 1.0, EMPI, 211)
        assert cut.check_particle(p) is True

    def test_pz_cut_fail(self):
        cut = ParticleCut()
        cut.set_pz_cut(1.0, 3.0)
        p = Particle(0.1, 0.1, 5.0, 1.0, EMPI, 211)
        assert cut.check_particle(p) is False

    def test_npoint_cut_pass(self):
        cut = ParticleCut()
        cut.set_npoint_cut(10)
        p = Particle(0.1, 0.1, 5.0, 1.0, EMPI, 211)
        p.set_n_point(15)
        assert cut.check_particle(p) is True

    def test_npoint_cut_fail(self):
        cut = ParticleCut()
        cut.set_npoint_cut(10)
        p = Particle(0.1, 0.1, 5.0, 1.0, EMPI, 211)
        p.set_n_point(5)
        assert cut.check_particle(p) is False


class TestPairCutCheckPair:
    """Verify PairCut.check_pair() correctness including the refactored get_p() reuse."""

    def _make_pion(self, px, py, pz, charge):
        return Particle(px, py, pz, charge, EMPI, 211 if charge > 0 else -211)

    def test_no_cuts_passes(self):
        cut = PairCut()
        a = self._make_pion(0.1, 0.0, 1.0, 1.0)
        b = self._make_pion(-0.1, 0.0, 1.0, -1.0)
        assert cut.check_pair(a, b) is True

    def test_opening_angle_cut_pass(self):
        cut = PairCut()
        cut.set_opening_angle_cut(0.0, math.pi)
        a = self._make_pion(1.0, 0.0, 0.0, 1.0)
        b = self._make_pion(0.0, 1.0, 0.0, -1.0)
        assert cut.check_pair(a, b) is True

    def test_opening_angle_cut_fail(self):
        """Antiparallel particles have angle = π; cut to < π/2 should fail."""
        cut = PairCut()
        cut.set_opening_angle_cut(0.0, math.pi / 2)
        a = self._make_pion(1.0, 0.0, 0.0, 1.0)
        b = self._make_pion(-1.0, 0.0, 0.0, -1.0)
        assert cut.check_pair(a, b) is False

    def test_cos_theta_cut_pass(self):
        """Parallel particles have cos_theta = 1; cut to [0.9, 1] should pass."""
        cut = PairCut()
        cut.set_cos_theta_cut(0.9, 1.0)
        a = self._make_pion(1.0, 0.0, 0.0, 1.0)
        b = self._make_pion(2.0, 0.0, 0.0, -1.0)
        assert cut.check_pair(a, b) is True

    def test_cos_theta_cut_fail(self):
        """Antiparallel particles cos_theta = -1; cut to [0, 1] fails."""
        cut = PairCut()
        cut.set_cos_theta_cut(0.0, 1.0)
        a = self._make_pion(1.0, 0.0, 0.0, 1.0)
        b = self._make_pion(-1.0, 0.0, 0.0, -1.0)
        assert cut.check_pair(a, b) is False

    def test_both_angle_cuts_consistent(self):
        """When both opening_angle and cos_theta cuts are active, they use the same angle."""
        cut = PairCut()
        cut.set_opening_angle_cut(0.0, math.pi)  # always passes
        cut.set_cos_theta_cut(-1.0, 1.0)         # always passes
        a = self._make_pion(0.3, 0.4, 2.0, 1.0)
        b = self._make_pion(0.1, 0.2, 2.0, -1.0)
        assert cut.check_pair(a, b) is True

    def test_inv_mass_cut_pass(self):
        cut = PairCut()
        cut.set_inv_mass_cut(0.0, 1.0)
        a = self._make_pion(0.1, 0.0, 1.0, 1.0)
        b = self._make_pion(-0.1, 0.0, 1.0, -1.0)
        m = a.get_inv_mass(b)
        assert cut.check_pair(a, b) is True
        assert m <= 1.0

    def test_inv_mass_cut_fail(self):
        cut = PairCut()
        cut.set_inv_mass_cut(0.0, 0.1)  # very tight — most pairs fail
        a = self._make_pion(1.0, 0.0, 5.0, 1.0)
        b = self._make_pion(-1.0, 0.0, 5.0, -1.0)
        assert cut.check_pair(a, b) is False
