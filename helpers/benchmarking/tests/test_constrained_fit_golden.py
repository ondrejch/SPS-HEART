#!/usr/bin/env python3
"""Golden-path smoke tests for the constrained multispecies fit primitives.

These tests intentionally avoid full SPS table I/O and multi-million event MC.
They lock the shared-coefficient NNLS + smooth-prior assembly used by the paper
pipeline so accidental objective changes fail fast.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from helpers.benchmarking.benchmark_multispecies_constrained import (
    _build_nstar_ladder_smooth_prior,
)


def test_shared_nnls_recovers_nonneg_coefficients() -> None:
    """Synthetic full-rank design: NNLS recovers planted non-negative weights."""
    rng = np.random.default_rng(20260331)
    n_points, n_comp = 40, 5
    # Positive design columns with mild collinearity still full rank here.
    A = rng.random((n_points, n_comp)) + 0.1
    true_w = np.array([0.8, 0.0, 0.35, 0.15, 0.0], dtype=float)
    y = A @ true_w
    sigma = np.full(n_points, 0.05, dtype=float)
    Aw = A / sigma[:, None]
    yw = y / sigma
    w_hat, _ = nnls(Aw, yw)
    assert w_hat.shape == (n_comp,)
    assert np.all(w_hat >= -1e-12)
    # Recover active coefficients within a tight tolerance on noiseless data.
    assert np.allclose(w_hat, true_w, atol=1e-6, rtol=0.0)


def test_smooth_prior_rows_are_zero_sum_adjacent_levels() -> None:
    labels = [
        "nstar_effL01_zero",
        "nstar_effL02_zero",
        "nstar_effL03_zero",
        "anti_nstar_effL01_zero",
        "anti_nstar_effL02_zero",
        "eta",
    ]
    out = _build_nstar_ladder_smooth_prior(component_labels=labels)
    assert bool(out.get("ok", False))
    A = np.asarray(out["A"], dtype=float)
    b = np.asarray(out["b"], dtype=float)
    # baryon ladder: two adjacent pairs; anti ladder: one pair
    assert A.shape[0] == 3
    assert np.allclose(b, 0.0)
    # Each prior row is a +1/-1 difference of two coefficients.
    for row in A:
        assert np.isclose(row.sum(), 0.0)
        assert np.count_nonzero(np.abs(row) > 0) == 2


def test_augmented_nnls_with_smooth_prior_prefers_smooth_ladder() -> None:
    """With a strong smoothness prior, uneven weights are pulled flatter."""
    labels = [
        "nstar_effL01_zero",
        "nstar_effL02_zero",
        "nstar_effL03_zero",
    ]
    prior = _build_nstar_ladder_smooth_prior(component_labels=labels)
    A_prior = np.asarray(prior["A"], dtype=float)
    b_prior = np.asarray(prior["b"], dtype=float)

    # One data point that weakly favors unequal weights.
    A_data = np.array([[1.0, 0.2, 0.05]], dtype=float)
    y_data = np.array([1.0], dtype=float)
    sigma = np.array([0.2], dtype=float)

    lam = 25.0  # strong smoothness
    A_aug = np.vstack([A_data / sigma[:, None], np.sqrt(lam) * A_prior])
    y_aug = np.concatenate([y_data / sigma, np.sqrt(lam) * b_prior])
    w_hat, _ = nnls(A_aug, y_aug)

    assert np.all(w_hat >= -1e-12)
    # Adjacent levels should not remain extremely unbalanced under strong prior.
    assert abs(w_hat[0] - w_hat[1]) < abs(w_hat[0]) * 0.75 + 0.05
    assert abs(w_hat[1] - w_hat[2]) < max(abs(w_hat[1]), 0.05) + 0.15
