"""Card-parsing and sampling tests for the coordinate-preserving generator.

Covers review items 1.2 (tabulated-PDF parsing) and the code half of 1.5
(target reflection identity).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from na49_mixing.generator import Generator, TabulatedPDF
from na49_mixing.reaction import Reaction

BASE = os.path.join(os.path.dirname(__file__), "..", "na49_mixing", "generators")
ETA_CARD = os.path.join(BASE, "benchmark", "eta_548.gen")
N1440_CARD = os.path.join(BASE, "michal", "nstar-to-proton", "n1440-zero.gen")
OMEGA_CARD = os.path.join(BASE, "michal", "mesons-to-pion", "omega.gen")


class TestTabulatedPDF:
    def test_rejects_nonmonotonic_coordinates(self):
        with pytest.raises(ValueError):
            TabulatedPDF([0.0, 0.5, 0.4, 1.0], [1.0, 1.0, 1.0, 1.0])

    def test_nonuniform_spacing_preserved(self):
        # Clustered coordinates on the left carry the listed densities; the
        # probability left of x=0 must equal the trapezoid integral there.
        coords = np.array([-1.0, -0.9, -0.85, -0.8, 0.8, 0.9, 1.0])
        values = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 2.0, 1.0])
        pdf = TabulatedPDF(coords, values)
        rng = np.random.default_rng(0)
        s = pdf.get_random_n(400_000, rng)
        frac_left = float(np.mean(s < 0.0))
        # Independent numeric integral of P(x < 0) on a fine grid.
        grid = np.linspace(-1.0, 1.0, 400_001)
        dens = np.interp(grid, coords, values)
        w = np.where(grid < 0.0, dens, 0.0)
        trapz = getattr(np, "trapezoid", None) or np.trapz
        expect_left = float(trapz(w, grid) / trapz(dens, grid))
        assert frac_left == pytest.approx(expect_left, abs=0.01)

    def test_bin_constant_mode(self):
        coords = [0.0, 1.0, 3.0]
        dens = [1.0, 3.0]
        pdf = TabulatedPDF(coords, dens)
        assert pdf.domain == (0.0, 3.0)
        # masses 1 and 6 with centroids 0.5 and 2 -> mean 12.5/7
        assert pdf.mean() == pytest.approx(12.5 / 7.0, rel=1e-9)

    def test_mean_matches_numeric_integral(self):
        rng_coords = np.cumsum(np.abs(np.random.default_rng(1).normal(0.05, 0.02, 40)))
        coords = np.concatenate([[0.0], rng_coords]) - rng_coords[0]
        vals = np.abs(np.random.default_rng(2).normal(1.0, 0.6, coords.size)) + 0.05
        pdf = TabulatedPDF(coords, vals)
        s = pdf.get_random_n(300_000, np.random.default_rng(3))
        assert float(s.mean()) == pytest.approx(pdf.mean(), abs=0.01)


class TestCardParsing:
    def _pl_pdf(self, card):
        gen = Generator(reaction=Reaction(158.0))
        gen.set_all(card)
        return gen._y_ymax

    def test_eta_card_domain_is_full_card_range(self):
        pdf = self._pl_pdf(ETA_CARD)
        lo, hi = pdf.domain
        assert lo == pytest.approx(-1.0)
        assert hi == pytest.approx(1.0)

    def test_eta_symmetric_card_samples_centered(self):
        pdf = self._pl_pdf(ETA_CARD)
        s = pdf.get_random_n(200_000, np.random.default_rng(11))
        assert abs(float(s.mean())) < 0.02
        assert abs(pdf.mean()) < 1e-9

    def test_baryon_card_domain_ends_at_last_coordinate(self):
        pdf = self._pl_pdf(N1440_CARD)
        lo, hi = pdf.domain
        assert lo == pytest.approx(-1.0)
        assert hi == pytest.approx(1.0)

    def test_interior_coordinates_shape_the_sample(self):
        """The nonuniform meson grid must reproduce its own knot values."""
        pdf = self._pl_pdf(OMEGA_CARD)
        coords = pdf.coords
        # Nonuniform grid: fine interior knots with wide end intervals.
        d = np.diff(coords)
        assert d.min() == pytest.approx(0.05, abs=1e-9)
        assert d.max() == pytest.approx(0.40, abs=1e-9)
        seg = 0.5 * (pdf.values[1:] + pdf.values[:-1]) * np.diff(coords)
        # Probe a nonzero-density narrow segment (third interval).
        i = 2
        s = pdf.get_random_n(400_000, np.random.default_rng(12))
        frac = float(np.mean((s > coords[i]) & (s <= coords[i + 1])))
        expect = float(seg[i] / seg.sum())
        assert frac == pytest.approx(expect, abs=0.004)


class TestTargetReflection:
    def test_mirrored_distribution_is_reflected(self):
        react = Reaction(158.0)
        gen = Generator(reaction=react)
        gen.set_all(OMEGA_CARD)

        from na49_mixing.parallel import _mirror_projectile_to_target_in_place

        n = 60_000
        parts = gen.generate_n(n)
        xf_proj = np.array([p.get_xf(react) for p in parts])
        for p in parts:
            _mirror_projectile_to_target_in_place(p, react)
        xf_targ = np.array([p.get_xf(react) for p in parts])

        qs = np.linspace(0.05, 0.95, 19)
        q_proj = np.quantile(xf_proj, qs)
        q_targ = np.quantile(xf_targ, qs)
        # F_targ(xF) = F_proj(-xF): matched upper/lower quantiles cancel.
        assert np.max(np.abs(q_targ + q_proj[::-1])) < 0.01
