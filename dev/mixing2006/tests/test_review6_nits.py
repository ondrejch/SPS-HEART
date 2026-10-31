"""Nits from friend review 6: sampling overflow, PDF rank, histogram
geometry, ``from_dict`` contents, and defensive copies of fit scales.

Findings 2--6 of ``.collab/na49_code_review6.md``. Finding 1 (rank of
the MultiFitter design matrix) is covered by the design-matrix rank
tests in ``tests/test_fits_behavior.py``.
"""
from __future__ import annotations

import math
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from na49_mixing.data_io import load_histograms, load_histograms_hdf5
from na49_mixing.fits import MultiFitter
from na49_mixing.generator import TabulatedPDF
from na49_mixing.histogram import Histogram1D


OVERFLOW_EDGE = 1.0e308


def _mapped_npz(tmp_path, *, keys, names, datasets):
    path = tmp_path / "schema.npz"
    arrays = {
        "__hist_keys": np.asarray(keys, dtype="U"),
        "__hist_names": np.asarray(names, dtype="U"),
    }
    arrays.update(datasets)
    np.savez(str(path), **arrays)
    return path


# ---------------------------------------------------------------------------
# Finding 2 — histogram sampling with an overflowing raw sum
# ---------------------------------------------------------------------------


class TestHistogramSamplingOverflow:
    def _two_huge_bins(self) -> Histogram1D:
        h = Histogram1D(2, 0.0, 1.0, name="huge-sum")
        h.fill(0.25, weight=OVERFLOW_EDGE)
        h.fill(0.75, weight=OVERFLOW_EDGE)
        assert math.isfinite(h.get_bin_content(1))
        assert math.isfinite(h.get_bin_content(2))
        return h

    def test_scalar_and_vector_draws_stay_finite_without_overflow_warning(self):
        h = self._two_huge_bins()
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            scalar = h.get_random(rng=np.random.default_rng(7))
            vector = h.get_random_n(32, rng=np.random.default_rng(7))
        assert math.isfinite(scalar)
        assert 0.0 <= scalar < 1.0
        assert np.all(np.isfinite(vector))
        assert np.all((vector >= 0.0) & (vector < 1.0))
        np.testing.assert_allclose(h._probs, [0.5, 0.5])
        assert float(h._probs.sum()) == pytest.approx(1.0)

    def test_vector_choice_accepts_the_normalized_probabilities(self):
        h = self._two_huge_bins()
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            draws = h.get_random_n(2000, rng=np.random.default_rng(11))
        left = float(np.mean(draws < 0.5))
        assert 0.35 < left < 0.65


# ---------------------------------------------------------------------------
# Finding 3 — TabulatedPDF rank and overflowing derived masses
# ---------------------------------------------------------------------------


class TestTabulatedPDFRankAndOverflow:
    def test_multidimensional_coords_rejected(self):
        with pytest.raises(ValueError, match="one-dimensional"):
            TabulatedPDF(
                np.zeros((2, 2)),
                np.ones((2, 2)),
            )

    def test_multidimensional_values_rejected(self):
        with pytest.raises(ValueError, match="one-dimensional"):
            TabulatedPDF([0.0, 1.0, 2.0], np.ones((2, 2)))

    def test_overflowing_coordinate_span_rejected_without_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with pytest.raises(ValueError, match="coordinate spans must be finite"):
                TabulatedPDF([-OVERFLOW_EDGE, OVERFLOW_EDGE], [1.0, 1.0])

    def test_large_finite_densities_normalize_without_warning(self):
        # Raw trapezoid masses overflow (1e308 + 1e308); scale-first
        # normalization keeps a usable finite distribution.
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            pdf = TabulatedPDF([0.0, 1.0], [OVERFLOW_EDGE, OVERFLOW_EDGE])
            draws = pdf.get_random_n(16, rng=np.random.default_rng(3))
        assert pdf.domain == (0.0, 1.0)
        assert np.all(np.isfinite(draws))
        assert np.all((draws >= 0.0) & (draws <= 1.0))

    def test_large_bin_constant_mass_normalizes_without_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            pdf = TabulatedPDF([0.0, OVERFLOW_EDGE], [OVERFLOW_EDGE])
            draws = pdf.get_random_n(8, rng=np.random.default_rng(5))
        assert pdf.domain == (0.0, OVERFLOW_EDGE)
        assert np.all(np.isfinite(draws))


# ---------------------------------------------------------------------------
# Finding 4 — finite edges whose span overflows
# ---------------------------------------------------------------------------


class TestOverflowingHistogramSpan:
    def test_constructor_rejects_opposite_sign_near_max_edges(self):
        with pytest.raises(ValueError, match="span must be finite"):
            Histogram1D(10, -OVERFLOW_EDGE, OVERFLOW_EDGE)

    def test_npz_loader_rejects_overflowing_span(self, tmp_path):
        path = _mapped_npz(
            tmp_path,
            keys=["wide"],
            names=["Wide"],
            datasets={
                "wide__meta": np.array(
                    [4, -OVERFLOW_EDGE, OVERFLOW_EDGE], dtype=np.float64
                ),
                "wide__contents": np.zeros(6, dtype=np.float64),
            },
        )
        with pytest.raises(ValueError) as excinfo:
            load_histograms(str(path))
        msg = str(excinfo.value)
        assert "span must be finite" in msg
        assert str(path) in msg
        assert "Wide" in msg

    def test_hdf5_loader_rejects_overflowing_span(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        path = tmp_path / "wide.h5"
        with h5py.File(str(path), "w") as handle:
            grp = handle.create_group("g")
            grp.attrs.create("name", "Wide")
            grp.attrs["nbins"] = 4
            grp.attrs["xlow"] = -OVERFLOW_EDGE
            grp.attrs["xhigh"] = OVERFLOW_EDGE
            grp.create_dataset("contents", data=np.zeros(6))
        with pytest.raises(ValueError) as excinfo:
            load_histograms_hdf5(str(path))
        msg = str(excinfo.value)
        assert "span must be finite" in msg
        assert str(path) in msg
        assert "Wide" in msg


# ---------------------------------------------------------------------------
# Finding 5 — from_dict contents invariant
# ---------------------------------------------------------------------------


class TestHistogramFromDictContents:
    def test_round_trip_keeps_finite_storage(self):
        h = Histogram1D(3, 0.0, 1.5, name="rt")
        h.fill(0.2, 2.0)
        h.fill(1.1, -0.5)
        restored = Histogram1D.from_dict(h.to_dict())
        assert restored.nbins == 3
        np.testing.assert_array_equal(restored._contents, h._contents)
        assert restored._contents is not h._contents
        assert restored._contents.dtype == np.float64

    def test_wrong_rank_rejected(self):
        with pytest.raises(ValueError, match="one-dimensional"):
            Histogram1D.from_dict(
                {
                    "nbins": 2,
                    "xlow": 0.0,
                    "xhigh": 1.0,
                    "contents": np.zeros((4, 1)),
                }
            )

    def test_wrong_length_rejected(self):
        with pytest.raises(ValueError, match="contents length 1"):
            Histogram1D.from_dict(
                {
                    "nbins": 2,
                    "xlow": 0.0,
                    "xhigh": 1.0,
                    "contents": [1.0],
                }
            )

    def test_non_numeric_rejected(self):
        with pytest.raises(ValueError, match="real numeric dtype"):
            Histogram1D.from_dict(
                {
                    "nbins": 2,
                    "xlow": 0.0,
                    "xhigh": 1.0,
                    "contents": ["a", "b", "c", "d"],
                }
            )

    def test_nonfinite_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            Histogram1D.from_dict(
                {
                    "nbins": 2,
                    "xlow": 0.0,
                    "xhigh": 1.0,
                    "contents": [0.0, 1.0, float("nan"), 0.0],
                }
            )

    def test_missing_contents_rejected(self):
        with pytest.raises(ValueError, match="requires 'contents'"):
            Histogram1D.from_dict(
                {"nbins": 2, "xlow": 0.0, "xhigh": 1.0, "name": "empty"}
            )


# ---------------------------------------------------------------------------
# Finding 6 — returned MultiFitter scales are copies
# ---------------------------------------------------------------------------


class TestMultiFitterScaleCopies:
    def _fitted(self) -> MultiFitter:
        data = Histogram1D(4, 0.0, 1.0, name="data")
        mc = Histogram1D(4, 0.0, 1.0, name="mc")
        centers = np.array([0.125, 0.375, 0.625, 0.875])
        data.fill_array(centers, np.array([2.0, 4.0, 6.0, 8.0]))
        mc.fill_array(centers, np.array([1.0, 2.0, 3.0, 4.0]))
        mf = MultiFitter(data, [mc], fit_range=(0.05, 0.95))
        mf.fit()
        return mf

    def test_mutating_fit_return_does_not_change_stored_scales(self):
        mf = self._fitted()
        out = mf.fit()
        stored = mf.scales.copy()
        out[0] = float("nan")
        np.testing.assert_array_equal(mf.scales, stored)
        assert math.isfinite(float(mf.scales[0]))

    def test_mutating_scales_property_does_not_change_later_reads(self):
        mf = self._fitted()
        view = mf.scales
        stored = view.copy()
        view[0] = -100.0
        np.testing.assert_array_equal(mf.scales, stored)
        assert float(mf.scales[0]) != -100.0
