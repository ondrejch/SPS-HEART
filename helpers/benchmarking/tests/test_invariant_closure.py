"""Analytic closure test for the counts -> invariant-cross-section conversion.

Constructs a known f(xF, pT), samples MC tracks from the physical phase-space
density g ~ pT * f (uniform azimuth), runs the exact production gridding
helpers, and verifies that the bin-averaged invariant template is recovered.

The raw-counts path is asserted to FAIL the same recovery, so the test
detects the historical convention defect rather than passing vacuously.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from helpers.benchmarking.sps_core import (
    PDG_MASS_GEV,
    counts_on_grid,
    grid_point_areas,
    invariant_track_weights,
)

SQRT_S = 17.27
XEDGES = np.linspace(-0.5, 0.9, 15)
PEDGES = np.linspace(0.05, 2.05, 21)
SPECIES_PDG = -211


def _truth_f(x, p):
    return np.exp(-p / 0.45) * (1.0 + 0.6 * x * x)


def _bin_average_truth():
    """Dense numeric bin-averaged f over every grid cell."""
    xf_fine = np.linspace(XEDGES[0] + 1e-9, XEDGES[-1] - 1e-9, 240)
    pt_fine = np.linspace(PEDGES[0] + 1e-9, PEDGES[-1] - 1e-9, 240)
    P, X = np.meshgrid(pt_fine, xf_fine, indexing="ij")
    F = _truth_f(X, P)
    counts, _, _ = np.histogram2d(
        X.ravel(), P.ravel(), bins=[XEDGES, PEDGES]
    )
    sums, _, _ = np.histogram2d(
        X.ravel(), P.ravel(), bins=[XEDGES, PEDGES], weights=F.ravel()
    )
    return sums / counts


def _sample_tracks(n_target, rng):
    """Rejection-sample tracks from g(xF,pT) ~ f(xF,pT) * pT / E.

    With this density the per-track Jacobian weight E/pT multiplies back to
    exactly f, so the production estimator must reproduce the bin-averaged
    truth up to one global constant and MC noise.
    """
    m = PDG_MASS_GEV[SPECIES_PDG]
    def _g(x, p):
        pl = x * (0.5 * SQRT_S)
        e = np.sqrt(m * m + pl * pl + p * p)
        return _truth_f(x, p) * p / e
    got_x, got_p = [], []
    n_acc = 0
    pmax_env = None
    while n_acc < n_target:
        n_prop = 400_000
        x = rng.uniform(XEDGES[0], XEDGES[-1], n_prop)
        p = rng.uniform(PEDGES[0], PEDGES[-1], n_prop)
        g = _g(x, p)
        if pmax_env is None:
            xs = np.linspace(XEDGES[0], XEDGES[-1], 201)
            ps = np.linspace(PEDGES[0], PEDGES[-1], 401)
            PS, XS = np.meshgrid(ps, xs, indexing="ij")
            pmax_env = 1.05 * np.max(_g(XS, PS))
        u = rng.uniform(0.0, pmax_env, n_prop)
        keep = u < g
        got_x.append(x[keep])
        got_p.append(p[keep])
        n_acc += int(keep.sum())
    return (np.concatenate(got_x)[:n_target], np.concatenate(got_p)[:n_target])


class TestInvariantClosure:
    def test_weighted_columns_recover_bin_averaged_truth(self):
        rng = np.random.default_rng(20260823)
        xf, pt = _sample_tracks(500_000, rng)

        ix_data, ip_data = np.meshgrid(
            np.arange(len(XEDGES) - 1), np.arange(len(PEDGES) - 1), indexing="ij"
        )
        ix_data, ip_data = ix_data.ravel(), ip_data.ravel()

        w = invariant_track_weights(SPECIES_PDG, xf, pt, SQRT_S)
        # (xf, pt already in scope from _sample_tracks)
        f_hat = counts_on_grid(xf, pt, XEDGES, PEDGES, ix_data, ip_data, weights=w)
        areas = grid_point_areas(XEDGES, PEDGES, ix_data, ip_data)
        f_hat = f_hat / areas

        truth = _bin_average_truth().ravel()

        # Shape agreement up to one global normalisation constant.
        kappa = np.sum(f_hat * truth) / np.sum(f_hat * f_hat)
        rel = np.abs(kappa * f_hat - truth) / truth

        # Per-bin MC statistics of the weighted estimator: the weights are
        # heavy-tailed, so the relative sigma is sqrt(sum w^2)/sum w, NOT
        # Poisson 1/sqrt(N).
        sum_w2 = counts_on_grid(xf, pt, XEDGES, PEDGES, ix_data, ip_data,
                                weights=w * w)
        # f_hat = (sum w)/area -> sum w = f_hat * area.
        area_all = grid_point_areas(XEDGES, PEDGES, ix_data, ip_data)
        sum_w = f_hat * area_all
        sigma_rel = np.sqrt(sum_w2) / np.maximum(sum_w, 1e-300)
        tol = 0.02 + 3.0 * sigma_rel

        assert float(np.median(rel)) < 0.03, f"median rel bias {np.median(rel):.4f}"
        worst = int(np.argmax(rel - tol))
        assert bool(np.all(rel <= tol)), (
            f"bin {worst}: rel bias {rel[worst]:.3f} exceeds "
            f"statistical tolerance {tol[worst]:.3f} (sigma_rel={sigma_rel[worst]:.3f})"
        )

    @staticmethod
    def _sample_masses():
        """Cell integrals of the sampling density g (for per-bin statistics)."""
        xf_fine = np.linspace(XEDGES[0] + 1e-9, XEDGES[-1] - 1e-9, 240)
        pt_fine = np.linspace(PEDGES[0] + 1e-9, PEDGES[-1] - 1e-9, 240)
        P, X = np.meshgrid(pt_fine, xf_fine, indexing="ij")
        m = PDG_MASS_GEV[SPECIES_PDG]
        pl = X * (0.5 * SQRT_S)
        E = np.sqrt(m * m + pl * pl + P * P)
        G = _truth_f(X, P) * P / E
        sums, _, _ = np.histogram2d(
            X.ravel(), P.ravel(), bins=[XEDGES, PEDGES], weights=G.ravel()
        )
        return sums

    def test_raw_counts_path_fails_recovery(self):
        """The superseded raw-count convention must NOT pass this closure."""
        rng = np.random.default_rng(20260824)
        xf, pt = _sample_tracks(500_000, rng)
        ix_data, ip_data = np.meshgrid(
            np.arange(len(XEDGES) - 1), np.arange(len(PEDGES) - 1), indexing="ij"
        )
        ix_data, ip_data = ix_data.ravel(), ip_data.ravel()
        raw = counts_on_grid(xf, pt, XEDGES, PEDGES, ix_data, ip_data)
        areas = grid_point_areas(XEDGES, PEDGES, ix_data, ip_data)
        rho = raw / areas
        truth = _bin_average_truth().ravel()
        kappa = np.sum(rho * truth) / np.sum(rho * rho)
        rel = np.abs(kappa * rho - truth) / truth
        assert float(np.median(rel)) > 0.15, (
            "raw-counts path unexpectedly recovered f; sensitivity lost"
        )

    def test_weight_formula(self):
        pid = SPECIES_PDG
        m = PDG_MASS_GEV[pid]
        x = np.array([-0.3, 0.0, 0.4])
        p = np.array([0.5, 0.8, 1.5])
        w = invariant_track_weights(pid, x, p, SQRT_S)
        pl = x * (0.5 * SQRT_S)
        expect = np.sqrt(m * m + pl * pl + p * p) / p
        assert np.allclose(w, expect, rtol=1e-12)
