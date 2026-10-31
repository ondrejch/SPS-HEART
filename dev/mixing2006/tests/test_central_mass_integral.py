"""The central-production mass-law normalization at large exponents
(2026-10-02; external review g1 B1).

``log_central_mass_integral`` gave NaN at ``epsilon >= 1``: the upper
incomplete gamma took one ``s + 1`` recurrence step, which left a
non-positive order. The recurrence now applies recursively, so the
integral is finite and matches quadrature over the exponent range the
R67 to R69 fits open (epsilon_nd up to 1.5) and beyond; a NaN raises.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from na49_mixing.topology_mass_laws import _gamma_upper_vec, log_central_mass_integral, log_density_ratio

A = 0.1349766 ** 2


@pytest.mark.parametrize("eps", [0.3, 0.99, 1.0, 1.01, 1.2, 1.5, 2.0, 2.3])
@pytest.mark.parametrize("beta", [-0.3, -2.0, -6.0])
@pytest.mark.parametrize("c", [1.0, 30.0, 200.0])
def test_matches_quadrature(eps, beta, c):
    ref = quad(lambda x: x ** (-1 - eps) * math.exp(beta * x), A, c, limit=400, points=[0.05, 0.5, 2.0])[0]
    got = float(log_central_mass_integral(np.array([eps]), np.array([beta]), A, np.array([c]))[0])
    assert math.isfinite(got)
    assert abs(got - math.log(ref)) < 1e-8


def test_continuous_across_unit_exponent():
    eps = np.array([0.999, 0.9999, 1.0, 1.0001, 1.001])
    vals = log_central_mass_integral(eps, np.full(5, -2.0), A, np.full(5, 50.0))
    assert np.all(np.isfinite(vals)) and np.max(np.abs(np.diff(vals))) < 0.01


def test_orders_above_minus_one_are_unchanged():
    """The s > -1 branches are the committed expressions: the recursion
    only adds the s <= -1 orders."""
    import scipy.special as sp
    s = np.array([-0.5, -0.2, 0.4, 1.3]); z = np.array([0.3, 2.0, 0.7, 5.0])
    got = _gamma_upper_vec(s, z)
    neg = s < 0
    ref = np.empty(4)
    ref[neg] = (sp.gamma(s[neg] + 1) * sp.gammaincc(s[neg] + 1, z[neg]) - z[neg] ** s[neg] * np.exp(-z[neg])) / s[neg]
    ref[~neg] = sp.gamma(s[~neg]) * sp.gammaincc(s[~neg], z[~neg])
    assert np.array_equal(got, ref)


def test_nd_density_ratio_is_finite_at_large_epsilon_nd():
    rng = np.random.default_rng(3)
    n = 50
    arrays = {"t1": -rng.uniform(0.05, 1.0, n), "t2": -rng.uniform(0.05, 1.0, n), "mc2": rng.uniform(0.02, 2.0, n),
              "m1": rng.uniform(1.1, 3.0, n), "m2": rng.uniform(1.1, 3.0, n), "k1": np.full(n, 2.0), "k2": np.full(n, 2.0),
              "lo1": np.full(n, 1.073), "hi1": np.full(n, 15.0), "lo2": np.full(n, 1.073), "hi2": np.full(n, 15.0)}
    arrays["t_c"] = arrays["t1"] + arrays["t2"]
    for eps in (0.9, 1.0, 1.2, 1.5):
        out = log_density_ratio("nd", arrays, eps, 2.5, eps0=0.35, b_sd0=2.5, sqrt_s=17.27)
        assert np.all(np.isfinite(out))
