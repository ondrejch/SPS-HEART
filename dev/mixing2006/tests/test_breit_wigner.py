"""Tests for the Breit-Wigner distribution (BreitWigner class)."""
import math
import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from na49_mixing.breit_wigner import BreitWigner, bw_new, bw_new_array, bw_fraction_above


class TestBWFunction:
    """Unit tests for the scalar bw_new function."""

    def test_zero_below_threshold(self):
        params = (1.07, 1.232, 0.12, 1.0, 2.0, 2.5)
        assert bw_new(1.07, params) == pytest.approx(0.0, abs=1e-12)
        assert bw_new(0.5, params) == 0.0

    def test_zero_above_dumphigh(self):
        params = (1.07, 1.232, 0.12, 1.0, 2.0, 2.5)
        assert bw_new(2.5, params) == 0.0
        assert bw_new(3.0, params) == 0.0

    def test_peak_near_nominal_mass(self):
        """BW should peak close to the nominal mass of 1.232 GeV."""
        params = (1.07, 1.232, 0.12, 1.0, 2.0, 2.5)
        masses = np.linspace(1.08, 1.99, 500)
        values = [bw_new(m, params) for m in masses]
        peak_mass = masses[np.argmax(values)]
        # Peak should be within 50 MeV of nominal
        assert abs(peak_mass - 1.232) < 0.05

    def test_non_negative(self):
        params = (1.07, 1.232, 0.12, 1.0, 2.0, 2.5)
        masses = np.linspace(0.5, 3.0, 1000)
        for m in masses:
            assert bw_new(m, params) >= 0.0

    def test_vectorised_matches_scalar(self):
        params = (1.07, 1.232, 0.12, 1.0, 2.0, 2.5)
        masses = np.linspace(0.9, 2.6, 200)
        scalar = np.array([bw_new(m, params) for m in masses])
        vector = bw_new_array(masses, params)
        np.testing.assert_allclose(scalar, vector, rtol=1e-10)

    def test_exact_formula_at_nominal_mass(self):
        """Check the formula value at x = rmas exactly."""
        schwel, rmas, brei, pbrei, dumplow, dumphigh = 1.07, 1.232, 0.12, 1.0, 2.0, 2.5
        x = rmas
        gamma = (
            brei
            * ((x - schwel) / (rmas - schwel)) ** 3
            * 2.0 * (rmas - schwel) ** 2
        )
        gamma /= (x - schwel) ** 2 + (rmas - schwel) ** 2
        expected = (x * rmas * gamma * pbrei) / (
            (x * x - rmas * rmas) ** 2 + rmas * rmas * gamma * gamma
        )
        params = (schwel, rmas, brei, pbrei, dumplow, dumphigh)
        assert bw_new(rmas, params) == pytest.approx(expected, rel=1e-9)


class TestBreitWigner:
    """Tests for the BreitWigner sampler class."""

    def setup_method(self):
        self.rng = np.random.default_rng(42)
        self.bw = BreitWigner(
            schwel=1.07, rmas=1.232, brei=0.12, pbrei=1.0,
            dumplow=2.0, dumphigh=2.5, rng=self.rng
        )

    def test_get_random_in_range(self):
        for _ in range(100):
            m = self.bw.get_random()
            assert 1.07 <= m <= 2.5

    def test_get_random_n(self):
        masses = self.bw.get_random_n(1000)
        assert len(masses) == 1000
        assert np.all(masses >= 1.07)
        assert np.all(masses <= 2.5)

    def test_get_value(self):
        v = self.bw.get_value(1.232)
        assert v > 0.0

    def test_sampling_mean_near_mass(self):
        """Mean of 10000 samples should be within 300 MeV of nominal mass.

        The Delta(1232) BW has a long tail toward higher masses (damping from
        2.0–2.5 GeV), so the distribution mean is shifted above the pole mass.
        """
        masses = self.bw.get_random_n(10000)
        assert abs(masses.mean() - 1.232) < 0.30

    def test_parameter_access(self):
        assert self.bw.get_parameter(0) == pytest.approx(1.07)
        assert self.bw.get_parameter(1) == pytest.approx(1.232)

    def test_set_parameters(self):
        self.bw.set_parameters(1.0, 1.5, 0.2, 1.0, 2.5, 3.0)
        assert self.bw.get_parameter(1) == pytest.approx(1.5)

    def test_from_string(self):
        bw2 = BreitWigner.from_string("1.07 1.232 0.12 1.0 2.0 2.5")
        assert bw2.get_parameter(1) == pytest.approx(1.232)


class TestBwFractionAboveMemoization:
    """REV-5b2f37d-01: bw_fraction_above memoizes its 1024-point grid
    result per line-shape instance (keyed on (lower, upper_cap)), reset
    by _build_hist() on any parameter change.

    The memo is a pure compute cache for the parent-side openness audit:
    a stored value is the bit-identical result of the one grid
    evaluation, and the primitive consumes no random numbers on either
    path, so no sampling stream can change.
    """

    @staticmethod
    def _bw() -> BreitWigner:
        return BreitWigner(
            0.45, 2.2, 0.22, 1.0, 2.8, 3.5, rng=np.random.default_rng(7)
        )

    def _count_grid_evals(self, monkeypatch) -> dict:
        """Wrap the module-level bw_new_array with a call counter."""
        import na49_mixing.breit_wigner as bwmod

        calls = {"n": 0}
        real = bwmod.bw_new_array

        def counting(xs, params):
            calls["n"] += 1
            return real(xs, params)

        monkeypatch.setattr(bwmod, "bw_new_array", counting)
        return calls

    def test_repeat_call_hits_the_cache_and_returns_the_identical_value(
        self, monkeypatch
    ):
        bw = self._bw()
        calls = self._count_grid_evals(monkeypatch)
        first = bw_fraction_above(bw, 1.9)
        # One uncached call evaluates the numerator and denominator grids.
        assert calls["n"] == 2
        second = bw_fraction_above(bw, 1.9)
        assert calls["n"] == 2, "repeat call re-evaluated the grid"
        assert second == first

    def test_cached_value_matches_an_uncached_fresh_instance_exactly(self):
        bw = self._bw()
        v1 = bw_fraction_above(bw, 1.9)
        v2 = bw_fraction_above(bw, 1.9)
        fresh = self._bw()
        assert v1 == v2 == bw_fraction_above(fresh, 1.9)

    def test_distinct_caps_occupy_distinct_cache_entries(self, monkeypatch):
        bw = self._bw()
        calls = self._count_grid_evals(monkeypatch)
        capped = bw_fraction_above(bw, 1.9, 2.6)
        uncapped = bw_fraction_above(bw, 1.9)
        assert calls["n"] == 4
        assert bw_fraction_above(bw, 1.9, 2.6) == capped
        assert bw_fraction_above(bw, 1.9) == uncapped
        assert calls["n"] == 4, "repeat call re-evaluated the grid"

    def test_parameter_update_invalidates_the_cached_fractions(self):
        bw = self._bw()
        before = bw_fraction_above(bw, 1.9)
        bw.set_parameters(0.45, 2.0, 0.22, 1.0, 2.6, 3.3)
        after = bw_fraction_above(bw, 1.9)
        fresh = BreitWigner(
            0.45, 2.0, 0.22, 1.0, 2.6, 3.3, rng=np.random.default_rng(7)
        )
        assert after == bw_fraction_above(fresh, 1.9)
        assert after != before, "sanity: the line shape actually moved"

    def test_delta_line_shape_stays_exact_outside_the_memo(self):
        delta = BreitWigner(0.45, 2.2, 0.0, 1.0, 2.8, 3.5)
        assert bw_fraction_above(delta, 1.9) == 1.0
        assert bw_fraction_above(delta, 2.3) == 0.0
