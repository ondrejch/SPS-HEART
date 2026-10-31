"""Behavioral depth for :mod:`na49_mixing.fits` (review3 Phase P7).

Task card TASK-2026-08-26-code-review3, Phase P7, fits.py items:

- fit recovery on synthetic data with known parameters;
- singular / ill-conditioned covariance inputs (rank-deficient MC stacks,
  flat objectives);
- zero-uncertainty inputs (the ``np.sqrt(np.maximum(data_vals, 1.0))``
  floor policy in :meth:`MultiFitter.fit`);
- optimizer boundary behavior (L-BFGS-B lower bounds ``(0, None)``, the
  ``mask = a > 0`` bin selection in :class:`Fitter`);
- fit-range selection flows through ``find_bin``, so the Phase P1
  non-finite policy applies: a fit range whose endpoints are NaN/±inf must
  raise ``ValueError`` naming ``find_bin``, never silently bin.

Every assertion checks numbers against independently computed formulas or
exact exception messages; nothing asserts mere execution. The P5/P1
interplay requested by the cross-phase list lives here too, exercised
through both consumers that route fit ranges through ``find_bin``
(``MultiFitter.fit`` and ``Spectra.fit``).

Review5 Phase P2 additions (task card TASK-2026-08-26-code-review5-fixes,
findings 2 and 3): fits whose selected template matrix carries no
information — all-zero templates, one identically-zero component among
informative ones, exactly duplicated components — are rejected before the
optimizer runs, and every failing refit clears the scales of an earlier
successful fit. These rejections supersede two earlier pins below: the
no-crash behavior of a flat all-zero objective and the sum-only recovery
of duplicated columns.

Review6 additions (task card TASK-2026-08-27-review6-stage-f-multifitter-rank):
the zero-row / duplicate refusals are only two named special cases of
linear dependence, so ``fit()`` now also forms the weighted design matrix
the chi-square actually uses, ``design = (mc_vals / errors).T`` of shape
``(n_selected, n_comp)``, and rejects any design whose numerical column
rank is below the component count — proportional templates, a template
equal to a combination of the others, and more components than selected
bins. A nearly-but-not-exactly dependent design with full numerical rank
keeps fitting, and the 2-norm condition number of the design matrix is
available as the ``design_condition_number`` diagnostic after a fully
successful fit. A design with non-finite entries is refused by its own
named diagnostic before the rank is computed.
"""

from __future__ import annotations

import numpy as np
import pytest

from na49_mixing.fits import Fitter, MultiFitter
from na49_mixing.histogram import Histogram1D
from na49_mixing.spectra import Spectra

NBINS = 10
XLOW, XHIGH = 0.0, 1.0

# Fixed positive interior shape, deliberately irregular so wrong fit
# windows cannot accidentally reproduce the right answer.
PATTERN = np.array([1.5, 0.8, 2.2, 1.1, 0.6, 1.9, 1.3, 0.4, 2.5, 0.9],
                   dtype=float)


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------

def _make_hist(name: str = "") -> Histogram1D:
    return Histogram1D(NBINS, XLOW, XHIGH, name=name)


def _bin_centers(hist: Histogram1D) -> np.ndarray:
    return hist.xlow + (np.arange(hist.nbins) + 0.5) * hist.bin_width


def _fill_interior(hist: Histogram1D, weights: np.ndarray) -> None:
    """Deposit one weight per interior bin (bin i+1 receives weights[i])."""
    hist.fill_array(_bin_centers(hist), np.asarray(weights, dtype=float))


def _closed_form_scale(data_vals: np.ndarray, mc_vals: np.ndarray) -> float:
    """Analytic minimizer of the MultiFitter chi² for one common scale.

    Mirrors the documented objective: errors are ``sqrt(max(d, 1))``, the
    zero-uncertainty floor, so ``e² = max(d, 1)`` and

        s* = Σ d·m/e² / Σ m²/e².
    """
    e2 = np.maximum(data_vals, 1.0)
    return float((data_vals * mc_vals / e2).sum()
                 / (mc_vals ** 2 / e2).sum())


def _two_component_fixture(
        ka: float, kb: float,
        xlow: float = XLOW, xhigh: float = XHIGH,
) -> tuple[Histogram1D, list[Histogram1D]]:
    """data = ka·A + kb·B with two independent positive shapes."""
    shape_b = PATTERN[::-1].copy() * 1.3 + 0.25
    data = Histogram1D(NBINS, xlow, xhigh, name="multi_data")
    _fill_interior(data, ka * PATTERN + kb * shape_b)
    mc_a = Histogram1D(NBINS, xlow, xhigh, name="mc_A")
    _fill_interior(mc_a, PATTERN)
    mc_b = Histogram1D(NBINS, xlow, xhigh, name="mc_B")
    _fill_interior(mc_b, shape_b)
    return data, [mc_a, mc_b]


class _PartialSpectra:
    """Duck-typed Spectra view with optional missing components.

    ``Fitter.add_spectra``/``subtract_spectra`` treat ``None`` as an absent
    component; the production class can produce these states through lazy
    initialization, so the None tolerances are part of the public contract
    exercised here with a minimal stand-in carrying only the five methods
    the fitter touches.
    """

    def __init__(self, is_data, dif=None, mass=None, mix=None, entries=0):
        self._is_data = bool(is_data)
        self._dif = dif
        self._mass = mass
        self._mix = mix
        self._entries = int(entries)

    def is_data(self):
        return self._is_data

    def get_dif_hist(self):
        return self._dif

    def get_mass_hist(self):
        return self._mass

    def get_mix_hist(self):
        return self._mix

    def get_entries(self):
        return self._entries


# ---------------------------------------------------------------------------
# Fitter population and routing
# ---------------------------------------------------------------------------

class TestFitterPopulationRouting:
    H_WEIGHTS_DIF = [1.0, 2.0, 3.0, 4.0]
    H_WEIGHTS_MASS = [10.0, 20.0, 30.0, 40.0]
    H_WEIGHTS_MIX = [100.0, 200.0, 300.0, 400.0]

    @staticmethod
    def _hist(kind: str) -> Histogram1D:
        table = {
            "dif": TestFitterPopulationRouting.H_WEIGHTS_DIF,
            "mass": TestFitterPopulationRouting.H_WEIGHTS_MASS,
            "mix": TestFitterPopulationRouting.H_WEIGHTS_MIX,
        }
        h = Histogram1D(4, 0.0, 2.0, name=kind)
        h.fill_array(np.array([0.25, 0.75, 1.25, 1.75]),
                     np.asarray(table[kind]))
        return h

    # Accumulator lookup: (side, kind) -> attribute holding the target.
    TARGET = {
        ("data", "dif"): "_his_a",
        ("data", "mass"): "_his_data_mass",
        ("data", "mix"): "_his_data_mix",
        ("mc", "dif"): "_his_b",
        ("mc", "mass"): "_his_mc_mass",
        ("mc", "mix"): "_his_mc_mix",
    }

    def test_add_spectra_routes_by_side_with_entry_counting(self):
        f = Fitter(nbins=4, x_low=0.0, x_high=2.0)
        data_spec = Spectra("routedata", True, nbins=4, m_low=0.0,
                            m_high=2.0, depth=2)
        mass_d = data_spec.get_mass_hist()
        assert mass_d is not None
        mass_d.fill_array(np.array([0.25]), np.array([7.0]))
        f.add_spectra(data_spec)
        expected = np.zeros(6)
        expected[1] = 7.0
        np.testing.assert_allclose(f._his_data_mass._contents, expected)
        np.testing.assert_allclose(f._his_data_mix._contents, np.zeros(6))
        np.testing.assert_allclose(f._his_a._contents, expected)  # dif
        # The MC accumulators stay untouched by a data spectrum.
        np.testing.assert_allclose(f.mc_diff._contents, np.zeros(6))
        assert f.n_mc_entries == 0

    def test_subtract_spectra_reverses_routing_and_counts(self):
        f = Fitter(nbins=4, x_low=0.0, x_high=2.0)
        mc_spec = Spectra("routemc", False, nbins=4, m_low=0.0,
                          m_high=2.0, depth=2)
        mass_m = mc_spec.get_mass_hist()
        assert mass_m is not None
        mass_m.fill_array(np.array([0.75]), np.array([3.0]))
        f.add_spectra(mc_spec)
        assert f.n_mc_entries == 0          # empty Spectra normalization
        f._n_mc_entries = 7                 # simulate counted normalizations
        # Subtracting the same spectrum cancels the earlier deposit ...
        f.subtract_spectra(mc_spec)
        np.testing.assert_allclose(f._his_mc_mass._contents, np.zeros(6))
        np.testing.assert_allclose(f._his_b._contents, np.zeros(6))
        # ... and the counter moves by get_entries(), which stays 0 for a
        # never-filled Spectra (counters move exactly by that accessor).
        assert f.n_mc_entries == 7

    @pytest.mark.parametrize("component", ["dif", "mass", "mix"])
    @pytest.mark.parametrize("side", ["data", "mc"])
    @pytest.mark.parametrize("op", ["add", "subtract"])
    def test_missing_component_is_skipped_others_processed(
            self, component, side, op):
        f = Fitter(nbins=4, x_low=0.0, x_high=2.0)
        present = {kind: self._hist(kind) for kind in
                   ("dif", "mass", "mix") if kind != component}
        spec = _PartialSpectra(is_data=(side == "data"),
                               dif=present.get("dif"),
                               mass=present.get("mass"),
                               mix=present.get("mix"), entries=3)
        sign = 1.0 if op == "add" else -1.0
        getattr(f, f"{op}_spectra")(spec)

        target_kind_all = {"dif": "mass"}   # placeholder, replaced below
        for kind in ("dif", "mass", "mix"):
            hist = getattr(f, self.TARGET[(side, kind)])
            want = np.zeros(6)
            if kind != component:
                src = self._hist(kind)._contents
                want = src * sign
            np.testing.assert_allclose(hist._contents, want,
                                       err_msg=f"{op} {side} {kind}")

        # The entry counter moves only for MC spectra, regardless of any
        # missing component.
        if side == "mc":
            expected = -3 if op == "subtract" else 3
            assert f.n_mc_entries == expected
        else:
            assert f.n_mc_entries == 0

    def test_fully_empty_mc_spectra_still_moves_the_counter(self):
        f = Fitter()
        f.add_spectra(_PartialSpectra(False, entries=11))  # type: ignore[arg-type]
        assert f.n_mc_entries == 11
        f.subtract_spectra(_PartialSpectra(False, entries=4))  # type: ignore[arg-type]
        assert f.n_mc_entries == 7

    def test_add_a_and_add_b_feed_the_fit_accumulators(self):
        f = Fitter(nbins=4, x_low=0.0, x_high=2.0)
        ha, hb = self._hist("mass"), self._hist("dif")
        f.add_a(ha)
        f.add_b(hb)
        np.testing.assert_allclose(f.data_diff._contents, ha._contents)
        np.testing.assert_allclose(f.mc_diff._contents, hb._contents)


# ---------------------------------------------------------------------------
# Fitter.get_fit / get_error: scale factor, masks, degenerate inputs
# ---------------------------------------------------------------------------

class TestFitterScaleRecovery:
    @staticmethod
    def _fitted(a_weights, b_weights) -> Fitter:
        f = Fitter(nbins=NBINS, x_low=XLOW, x_high=XHIGH)
        ha, hb = _make_hist("A"), _make_hist("B")
        _fill_interior(ha, a_weights)
        _fill_interior(hb, b_weights)
        f.add_a(ha)
        f.add_b(hb)
        return f

    def test_exact_proportional_input_recovers_known_factor(self):
        c = 2.5
        f = self._fitted(c * PATTERN, PATTERN)
        assert f.get_fit() == pytest.approx(c, rel=1e-12)
        # With a = c·b the statistical error is sqrt(c / Σb) over masked
        # bins; all interior weights are positive here.
        sigma = float(PATTERN.sum())
        assert f.get_error() == pytest.approx(np.sqrt(c / sigma),
                                              rel=1e-12)

    def test_formula_matches_public_accessor_computation(self):
        a_w = np.array([3.0, 0.5, 7.0, 2.0, 1.0, 4.0, 6.0, 0.2, 9.0, 1.5])
        b_w = np.array([2.0, 1.0, 3.0, 5.0, 0.1, 2.5, 4.0, 3.5, 1.0, 0.7])
        f = self._fitted(a_w, b_w)

        a = np.array([f.data_diff.get_bin_content(i)
                      for i in range(1, NBINS + 1)])
        b = np.array([f.mc_diff.get_bin_content(i)
                      for i in range(1, NBINS + 1)])
        mask = a > 0.0
        up = b[mask].sum()
        down = (b[mask] ** 2 / a[mask]).sum()
        assert f.get_fit() == pytest.approx(up / down, rel=1e-12)
        sig = (b[mask] ** 2 / a[mask]).sum()
        assert f.get_error() == pytest.approx(float(1.0 / np.sqrt(sig)),
                                              rel=1e-12)

    def test_single_positive_bin_drives_alpha_and_error_exactly(self):
        # Bin selection: only bins with a > 0 participate. Here exactly one
        # interior bin carries positive a (others carry negative content);
        # alpha = a_j / b_j and error = sqrt(a_j) / b_j for that bin.
        a_w = np.array([-3.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        b_w = np.array([9.0, 7.0, 5.0, 1.0, 8.0, 6.0, 2.0, 3.0, 4.0, 5.0])
        f = self._fitted(a_w, b_w)
        assert f.get_fit() == pytest.approx(4.0 / 1.0, rel=1e-12)
        assert f.get_error() == pytest.approx(2.0 / 1.0, rel=1e-12)

    def test_no_positive_bin_returns_zero_for_fit_and_error(self):
        f = self._fitted(np.zeros(NBINS), PATTERN)
        assert f.get_fit() == 0.0
        assert f.get_error() == 0.0

    def test_negative_only_content_also_returns_zero(self):
        a_w = -np.abs(PATTERN.copy())
        f = self._fitted(a_w, PATTERN)
        assert f.get_fit() == 0.0
        assert f.get_error() == 0.0

    def test_zero_template_where_data_positive_is_singular_down(self):
        # down = Σ b²/a vanishes when the template is identically zero on
        # every masked bin: get_fit and get_error must return 0.0 instead
        # of dividing by zero.
        a_w = PATTERN.copy()
        f = self._fitted(a_w, np.zeros(NBINS))
        assert f.get_fit() == 0.0
        assert f.get_error() == 0.0

    def test_clear_resets_every_accumulator_and_counter(self):
        f = self._fitted(2.0 * PATTERN, PATTERN)
        assert f.n_mc_entries == 0
        f._n_mc_entries = 42
        f.data_diff.get_random(rng=np.random.default_rng(0))
        f.clear()
        for hist in (f.data_diff, f.mc_diff):
            np.testing.assert_allclose(hist._contents, np.zeros(NBINS + 2))
            assert hist._cdf is None          # sampling cache dropped too
        assert f.n_mc_entries == 0
        assert f.get_fit() == 0.0

    def test_repr_reports_fit_and_error_values(self):
        f = self._fitted(2.0 * PATTERN, PATTERN)
        text = repr(f)
        assert text.startswith("Fitter(fit=2.0000")
        assert "error=" in text

    def test_multi_fitter_repr_names_component_count(self):
        data, mcs = _two_component_fixture(1.0, 1.0)
        mf = MultiFitter(data, mcs, fit_range=(0.03, 0.98))
        assert "n_components=2" in repr(mf)


# ---------------------------------------------------------------------------
# MultiFitter: recovery, zero-uncertainty floor, boundaries, conditioning
# ---------------------------------------------------------------------------

class TestMultiFitterRecovery:
    def test_two_component_synthetic_recovery(self):
        ka, kb = 2.0, 0.5
        data, mcs = _two_component_fixture(ka, kb)
        mf = MultiFitter(data, mcs, fit_range=(0.03, 0.98))
        scales = mf.fit()
        np.testing.assert_allclose(scales, [ka, kb], rtol=1e-4, atol=1e-7)
        # Reconstruction: predicted stack reproduces the data stack.
        window = range(data.find_bin(0.03), data.find_bin(0.98) + 1)
        pred = sum(s * np.array([h.get_bin_content(i) for i in window])
                   for s, h in zip(scales, mcs))
        obs = np.array([data.get_bin_content(i) for i in window])
        np.testing.assert_allclose(pred, obs, atol=1e-6)

    def test_scales_property_is_none_before_fit_and_result_after(self):
        data, mcs = _two_component_fixture(1.0, 1.0)
        # Explicit well-formed window: the constructor default (1.1, 2.1)
        # would select only flow bins against this [0, 1] fixture, which
        # is now rejected (review4 P2) instead of returning bogus scales.
        mf = MultiFitter(data, mcs, fit_range=(0.03, 0.98))
        assert mf.scales is None
        out = mf.fit()
        np.testing.assert_array_equal(mf.scales, out)
        assert mf.scales is not out
        assert out is not mf._scales

    def test_default_fit_range_matches_constructor_defaults(self):
        # Default MultiFitter ranges follow the Fitter defaults (1.1, 2.1);
        # exercising defaults documents the unchanged configuration surface.
        data, mcs = _two_component_fixture(1.7, 0.0,
                                           xlow=1.1, xhigh=2.1)
        mf = MultiFitter(data, mcs)     # default fit_range=(1.1, 2.1)
        scales = mf.fit()
        np.testing.assert_allclose(scales, [1.7, 0.0], rtol=1e-3,
                                   atol=1e-5)


class TestMultiFitterZeroUncertaintyFloor:
    def test_zero_uncertainty_bins_are_floored_not_divided(self):
        # Data content 0.0 would make sqrt(content) an uncertainty of zero;
        # the documented policy floors the variance at 1.0, so the analytic
        # optimum below (with e² = max(d, 1)) is reachable and finite.
        d = np.array([4.0, 0.0, 9.0, 0.0, 1.0])
        m = np.array([1.0, 2.0, 1.0, 3.0, 2.0])
        data = _make_hist()
        for i in range(NBINS):
            val = d[i % len(d)]
            data.set_bin_content(i + 1, val)
        mc = _make_hist()
        for i in range(NBINS):
            val = m[i % len(m)]
            mc.set_bin_content(i + 1, val)
        # Unused bins (index >= 5) get small-but-consistent copies so the
        # window stays representative of the fixture design above.
        for j in range(5, NBINS + 1):
            data.set_bin_content(j, 2.0)
            mc.set_bin_content(j, 1.0)

        d_full = np.array([data.get_bin_content(i)
                           for i in range(1, NBINS + 1)])
        m_full = np.array([mc.get_bin_content(i)
                           for i in range(1, NBINS + 1)])
        s_star = _closed_form_scale(d_full, m_full)

        mf = MultiFitter(data, [mc], fit_range=(0.0, 1.0))
        scales = mf.fit()
        assert scales[0] == pytest.approx(s_star, rel=1e-6)
        # Distinct from the unweighted least-squares solution, proving the
        # floor policy actually shaped the answer.
        unweighted = float((d_full * m_full).sum()
                           / (m_full ** 2).sum())
        assert abs(unweighted - s_star) > 1e-6


class TestMultiFitterBoundaryAndConditioning:
    def test_zero_data_window_pins_scale_at_lower_bound_zero(self):
        # Chi²(s) = Σ (s·m)²/e² is monotone increasing for s >= 0, so the
        # constrained minimum sits exactly at the bound 0.0.
        data, mcs = _two_component_fixture(0.0, 0.0)
        window = range(data.find_bin(0.03), data.find_bin(0.98) + 1)
        for i in window:
            data.set_bin_content(i, 0.0)
        mf = MultiFitter(data, mcs, fit_range=(0.03, 0.98))
        scales = mf.fit()
        # Terminated by the function-reduction criterion a few ulps above
        # zero; the pinned semantics is "at the bound", so 1e-6 suffices.
        assert float(scales[0]) < 1e-6
        assert float(scales[1]) < 1e-6


class TestMultiFitterFitRangeThroughFindBin:
    """Fit-range selection flows through Histogram1D.find_bin.

    For each window the expected scale is computed from the very bin list
    ``find_bin`` produces, including the underflow (0) and overflow
    (nbins+1) endpoints when the range exceeds the axis. Interior bins are
    filled with ``K_TRUE`` times the MC shape while the outside-garbage
    pair uses data = 0.5·mc: because the generating ratio (3.0) differs
    from the garbage ratio (0.5), any mis-selected bin provably moves the
    fitted scale away from the window value.
    """

    GARBAGE = 500.0
    K_TRUE = 3.0

    @staticmethod
    def _set_from_window(hist, contents):
        for i, val in enumerate(contents):
            hist.set_bin_content(i, float(val))

    @staticmethod
    def _slice_list(hist, lo, hi):
        i_min = hist.find_bin(lo)
        i_max = hist.find_bin(hi)
        return range(i_min, i_max + 1)

    def _run_window(self, lo, hi, interior_value_fn):
        """Build data/MC with signed garbage outside [lo, hi]; fit."""
        window = self._slice_list(_make_hist(), lo, hi)
        data_contents, mc_contents = [], []
        for i in range(NBINS + 2):           # underflow .. overflow
            if i in window:
                dv, mv = interior_value_fn(i)
                data_contents.append(dv)
                mc_contents.append(mv)
            else:
                junk = self.GARBAGE * (1 if (i % 2) else -1)
                data_contents.append(junk)
                mc_contents.append(junk * 0.5)
        data = _make_hist("win")
        mc = _make_hist("wmc")
        self._set_from_window(data, data_contents)
        self._set_from_window(mc, mc_contents)

        mf = MultiFitter(data, [mc], fit_range=(lo, hi))
        recovered = float(mf.fit()[0])

        d_win = np.array([data.get_bin_content(i) for i in window])
        m_win = np.array([mc.get_bin_content(i) for i in window])
        expected = _closed_form_scale(d_win, m_win)

        d_full = np.array(data._contents)     # noqa: SLF001 (reference only)
        m_full = np.array(mc._contents)       # noqa: SLF001
        full_form = _closed_form_scale(d_full, m_full)

        assert recovered == pytest.approx(expected, rel=1e-5)
        # Anti-vacuous guard: had the whole axis been used, the signed
        # garbage would move the estimate far away from the window value.
        assert abs(recovered - full_form) > 0.05 * abs(expected) \
            or abs(recovered - full_form) > 1.0

    def test_interior_window_selects_only_mapped_bins(self):
        k_true = self.K_TRUE
        self._run_window(
            0.23, 0.77,
            lambda i: (k_true * PATTERN[i - 1], PATTERN[i - 1]),
        )

    def test_range_below_axis_includes_underflow_bin_zero(self):
        w_under = 7.0
        k = self.K_TRUE
        self._run_window(
            -0.5, 0.45,
            lambda i: (w_under * k if i == 0 else k * PATTERN[i - 1],
                       w_under if i == 0 else PATTERN[i - 1]),
        )

    def test_range_above_axis_includes_overflow_bin(self):
        w_over = 4.0
        nb = NBINS
        k = self.K_TRUE
        self._run_window(
            0.55, 1.5,
            lambda i: (
                w_over * k if i == nb + 1 else k * PATTERN[i - 1],
                w_over if i == nb + 1 else PATTERN[i - 1]),
        )

    def test_nonfinite_endpoint_raises_valueerror_naming_find_bin(self):
        data, mcs = _two_component_fixture(1.0, 1.0)
        snapshot = (data._contents.tobytes(),
                    mcs[0]._contents.tobytes())  # noqa: SLF001
        mf = MultiFitter(data, mcs, fit_range=(0.1, 0.9))
        for bad in (float("nan"), float("inf"), float("-inf")):
            for slot in (0, 1):
                rng = [bad, 0.9] if slot == 0 else [0.1, bad]
                with pytest.raises(ValueError) as excinfo:
                    MultiFitter(data, mcs,
                                fit_range=(rng[0], rng[1])).fit()
                msg = str(excinfo.value)
                assert msg.startswith("find_bin: ")
                assert "non-finite value rejected" in msg
        # Nothing mutated; the scales cache still holds no result.
        assert (data._contents.tobytes(),
                mcs[0]._contents.tobytes()) == snapshot  # noqa: SLF001
        assert mf.scales is None

    def test_finite_edge_touching_endpoints_are_accepted(self):
        # Endpoints equal to or beyond the axis bounds map to bins 0 /
        # nbins+1 and stay legal for finite values: the window simply
        # includes under-/overflow content (covered numerically above);
        # this test pins the no-raise contract at the exact edges.
        data, mcs = _two_component_fixture(1.25, 0.75)
        mf = MultiFitter(data, mcs, fit_range=(XLOW - 1.0, XHIGH + 1.0))
        scales = mf.fit()
        assert np.all(np.isfinite(scales)) and np.all(scales >= 0.0)


class TestSpectraFitInterplayWithPolicy:
    """Spectra.fit routes its mass window through find_bin (spectra.py).

    This is the second public consumer of find_bin-driven fit ranges and
    therefore inherits the P1 rejection policy: non-finite endpoints raise
    ValueError, and a previously successful fit stays intact afterwards.
    """

    ALPHA_TRUE = 1.3

    def _spectra(self) -> Spectra:
        sp = Spectra("alpha_recovery", True, nbins=NBINS,
                     m_low=XLOW, m_high=XHIGH, depth=2)
        mix_h = sp.get_mix_hist()
        mass_h = sp.get_mass_hist()
        assert mix_h is not None and mass_h is not None
        _fill_interior(mix_h, PATTERN)
        _fill_interior(mass_h, self.ALPHA_TRUE * PATTERN)
        return sp

    def test_exact_alpha_recovery_through_public_api(self):
        sp = self._spectra()
        assert sp.fit(0.05, 0.95) == pytest.approx(self.ALPHA_TRUE,
                                                   rel=1e-12)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"),
                                     float("-inf")])
    @pytest.mark.parametrize("slot", [0, 1])
    def test_nonfinite_endpoint_raises_from_histogram_policy(self,
                                                             bad, slot):
        sp = self._spectra()
        args = ([bad, 0.95] if slot == 0 else [0.05, bad])
        with pytest.raises(ValueError) as excinfo:
            sp.fit(args[0], args[1])
        assert str(excinfo.value).startswith("find_bin: ")
        # The object remains usable for well-formed windows afterwards.
        assert sp.fit(0.05, 0.95) == pytest.approx(self.ALPHA_TRUE,
                                                   rel=1e-12)


# ---------------------------------------------------------------------------
# Review4 Phase P2: domain, compatibility, and optimizer-result validation
# ---------------------------------------------------------------------------

class _FakeOptimizeResult:
    """Minimal ``scipy.optimize.OptimizeResult`` stand-in."""

    def __init__(self, success, message, status, fun, x):
        self.success = bool(success)
        self.message = str(message)
        self.status = status
        self.fun = fun
        self.x = x


class TestMultiFitterDomainAndResultValidation:
    """Task card TASK-2026-08-26-code-review4, Phase P2 (finding 2).

    ``MultiFitter.fit()`` rejects invalid domains, empty component lists
    and incompatible MC geometries *before* any optimizer work, rejects
    unsuccessful / non-finite optimizer output afterwards, and leaves
    ``_scales`` at ``None`` on every raising path. Windows that include
    flow bins alongside interior bins stay legal (pinned by
    ``TestMultiFitterFitRangeThroughFindBin`` above).
    """

    @staticmethod
    def _boom_minimize(*_args, **_kwargs):
        """Sentinel proving optimization is never reached."""
        raise AssertionError("scipy.optimize.minimize must not be reached")

    @staticmethod
    def _patch_minimize(monkeypatch, fake):
        import scipy.optimize
        monkeypatch.setattr(scipy.optimize, "minimize", fake)

    # ------------------------------------------------------------------
    # Pre-optimization domain rejections
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("fit_range", [(0.8, 0.2), (0.4, 0.4)])
    def test_reversed_or_degenerate_range_raises(self, monkeypatch,
                                                 fit_range):
        data, mcs = _two_component_fixture(2.0, 0.5)
        self._patch_minimize(monkeypatch, self._boom_minimize)
        mf = MultiFitter(data, mcs, fit_range=fit_range)
        snapshot = (data._contents.tobytes(),
                    mcs[0]._contents.tobytes())  # noqa: SLF001
        with pytest.raises(ValueError) as excinfo:
            mf.fit()
        msg = str(excinfo.value)
        assert "fit_range[0] < fit_range[1]" in msg
        assert repr(fit_range) in msg
        assert mf.scales is None
        # No partial state: histograms bit-for-bit untouched, no caches.
        assert (data._contents.tobytes(),
                mcs[0]._contents.tobytes()) == snapshot  # noqa: SLF001

    @pytest.mark.parametrize("fit_range", [(2.0, 3.0), (-0.7, -0.2)])
    def test_window_touching_only_flow_bins_raises(self, monkeypatch,
                                                   fit_range):
        data, mcs = _two_component_fixture(1.0, 1.0)
        self._patch_minimize(monkeypatch, self._boom_minimize)
        mf = MultiFitter(data, mcs, fit_range=fit_range)
        with pytest.raises(ValueError) as excinfo:
            mf.fit()
        msg = str(excinfo.value)
        assert "at least one interior bin" in msg
        assert "under-/overflow bins" in msg
        assert repr(fit_range) in msg
        assert mf.scales is None

    def test_empty_mc_list_raises_before_any_optimizer_work(
            self, monkeypatch):
        data, _mcs = _two_component_fixture(1.0, 1.0)
        self._patch_minimize(monkeypatch, self._boom_minimize)
        mf = MultiFitter(data, [], fit_range=(0.03, 0.98))
        with pytest.raises(ValueError) as excinfo:
            mf.fit()
        assert "requires at least one MC component" in str(excinfo.value)
        assert mf.scales is None

    @pytest.mark.parametrize("mc_factory,label,nbins_str", [
        (lambda: Histogram1D(NBINS + 2, XLOW, XHIGH, name="wrong_nbins"),
         "wrong_nbins", "nbins=12"),
        (lambda: Histogram1D(NBINS, XLOW - 1.0, XHIGH, name="wrong_edges"),
         "wrong_edges", f"xlow={XLOW - 1.0!r}"),
        (lambda: Histogram1D(NBINS, XLOW, XHIGH + 2.0, name="wrong_xhigh"),
         "wrong_xhigh", f"xhigh={XHIGH + 2.0!r}"),
    ])
    def test_mismatched_mc_geometry_raises_naming_component(
            self, monkeypatch, mc_factory, label, nbins_str):
        data, good_mcs = _two_component_fixture(1.0, 1.0)
        self._patch_minimize(monkeypatch, self._boom_minimize)
        mf = MultiFitter(data, [good_mcs[0], mc_factory()],
                         fit_range=(0.03, 0.98))
        with pytest.raises(ValueError) as excinfo:
            mf.fit()
        msg = str(excinfo.value)
        assert "exact bin geometry" in msg
        assert "MC component 1" in msg       # offending index is named
        assert f"'{label}'" in msg           # offending component name too
        assert nbins_str in msg              # received MC geometry
        assert f"(nbins={NBINS}, " in msg    # data geometry side by side
        assert mf.scales is None

    # ------------------------------------------------------------------
    # Post-optimization result rejections (mocked optimizer seam)
    # ------------------------------------------------------------------

    def test_failed_optimizer_result_raises_descriptive_error(
            self, monkeypatch):
        data, mcs = _two_component_fixture(2.0, 0.5)
        self._patch_minimize(
            monkeypatch,
            lambda *a, **k: _FakeOptimizeResult(
                success=False, status=99,
                message="ABNORMAL_TERMINATION_IN_LNSRCH (fake)",
                fun=0.0, x=np.ones(2)),
        )
        mf = MultiFitter(data, mcs, fit_range=(0.03, 0.98))
        with pytest.raises(ValueError) as excinfo:
            mf.fit()
        msg = str(excinfo.value)
        assert "did not converge" in msg
        assert "ABNORMAL_TERMINATION_IN_LNSRCH (fake)" in msg
        assert "status=99" in msg
        assert "no scales were stored" in msg
        assert mf.scales is None

    def test_nonfinite_objective_raises_descriptive_error(self,
                                                          monkeypatch):
        data, mcs = _two_component_fixture(2.0, 0.5)
        self._patch_minimize(
            monkeypatch,
            lambda *a, **k: _FakeOptimizeResult(
                success=True, status=0, message="CONVERGENCE (fake)",
                fun=float("nan"), x=np.ones(2)),
        )
        mf = MultiFitter(data, mcs, fit_range=(0.03, 0.98))
        with pytest.raises(ValueError) as excinfo:
            mf.fit()
        msg = str(excinfo.value)
        assert "non-finite" in msg
        assert "objective" in msg
        assert "CONVERGENCE (fake)" in msg
        assert mf.scales is None

    def test_nonfinite_scale_raises_and_names_the_offender(
            self, monkeypatch):
        data, mcs = _two_component_fixture(2.0, 0.5)
        self._patch_minimize(
            monkeypatch,
            lambda *a, **k: _FakeOptimizeResult(
                success=True, status=0, message="CONVERGENCE (fake)",
                fun=3.5, x=np.array([float("nan"), 0.75])),
        )
        mf = MultiFitter(data, mcs, fit_range=(0.03, 0.98))
        with pytest.raises(ValueError) as excinfo:
            mf.fit()
        msg = str(excinfo.value)
        assert "returned scales are non-finite" in msg
        assert "scales[0]=nan" in msg
        assert "scales[1]=0.75" not in msg   # only offenders are listed
        assert mf.scales is None

    def test_successful_optimizer_result_path_is_unchanged(
            self, monkeypatch):
        # Control: a well-formed result is stored and returned as
        # equal arrays. The return value and the property are copies,
        # so mutating either cannot change the stored result.
        data, mcs = _two_component_fixture(2.0, 0.5)
        self._patch_minimize(
            monkeypatch,
            lambda *a, **k: _FakeOptimizeResult(
                success=True, status=0, message="CONVERGENCE (fake)",
                fun=3.5, x=np.array([1.25, 0.75])),
        )
        mf = MultiFitter(data, mcs, fit_range=(0.03, 0.98))
        out = mf.fit()
        np.testing.assert_allclose(out, [1.25, 0.75])
        np.testing.assert_array_equal(mf.scales, out)
        assert out is not mf._scales
        assert mf.scales is not out
        out[0] = float("nan")
        np.testing.assert_allclose(mf.scales, [1.25, 0.75])


# ---------------------------------------------------------------------------
# Review5 Phase P2: unconstrained-fit rejection (finding 2) and stale-scale
# clearing on failed refits (finding 3)
# ---------------------------------------------------------------------------

class TestMultiFitterUnconstrainedFitRejection:
    """Review5 finding 2: no fit without parameter sensitivity.

    After domain selection, ``fit()`` inspects the MC template matrix over
    exactly the selected bins ``[i_min..i_max]`` and rejects:

    - every template identically zero there (the objective is independent
      of all scales; historical behavior returned the arbitrary initial
      value ``array([1.])``);
    - one identically-zero component among otherwise-informative ones
      (that component's scale would stay arbitrary at the initial value);
    - exactly duplicated components (only their summed scale is pinned).

    The contract is exact-value comparison: near-distinct templates still
    fit normally (control below). Supersedes the review3-era pins
    ``test_flat_objective_on_all_zero_template_stays_safe`` and
    ``test_rank_deficient_stack_recovers_sum_not_split``.
    """

    @staticmethod
    def _boom_minimize(*_args, **_kwargs):
        """Sentinel proving optimization is never reached."""
        raise AssertionError("scipy.optimize.minimize must not be reached")

    @staticmethod
    def _patch_minimize(monkeypatch, fake):
        import scipy.optimize
        monkeypatch.setattr(scipy.optimize, "minimize", fake)

    def test_review_reproducer_all_zero_template_raises_valueerror(
            self, monkeypatch):
        # Exact v5-review reproducer: filled data, empty MC template,
        # window (0.1, 0.9). Historically this returned array([1.]).
        self._patch_minimize(monkeypatch, self._boom_minimize)
        data = Histogram1D(NBINS, XLOW, XHIGH, name="data")
        data.fill(0.5, 5.0)
        mc_zero = Histogram1D(NBINS, XLOW, XHIGH, name="mc_zero")
        mf = MultiFitter(data, [mc_zero], fit_range=(0.1, 0.9))
        with pytest.raises(ValueError) as excinfo:
            mf.fit()
        msg = str(excinfo.value)
        assert "no MC component contributes information in the fit " \
            "domain" in msg
        assert "(0.1, 0.9)" in msg          # the fit_range is named
        assert "1 MC component" in msg      # component count is named
        assert "zero" in msg and "unconstrained" in msg
        assert mf.scales is None

    def test_entirely_empty_data_and_templates_raise_without_optimizer(
            self, monkeypatch):
        # The both-empty case: no data, no templates, identical failure
        # mode. Also verifies the rejection precedes all optimizer work.
        self._patch_minimize(monkeypatch, self._boom_minimize)
        data = _make_hist("empty_data")
        mc_empty = _make_hist("empty_mc")
        mf = MultiFitter(data, [mc_empty], fit_range=(0.03, 0.98))
        with pytest.raises(ValueError) as excinfo:
            mf.fit()
        msg = str(excinfo.value)
        assert "no MC component contributes information in the fit " \
            "domain" in msg
        assert "all 1 MC component" in msg
        assert mf.scales is None

    def test_all_components_zeroed_over_selected_window_raise(
            self, monkeypatch):
        # Continuity with the superseded flat-objective pin: the same
        # two-component fixture with both templates zeroed inside the
        # window now raises instead of returning finite scales.
        self._patch_minimize(monkeypatch, self._boom_minimize)
        data, mcs = _two_component_fixture(1.0, 1.0)
        window = range(data.find_bin(0.03), data.find_bin(0.98) + 1)
        for i in window:
            for h in mcs:
                h.set_bin_content(i, 0.0)
        mf = MultiFitter(data, mcs, fit_range=(0.03, 0.98))
        with pytest.raises(ValueError) as excinfo:
            mf.fit()
        msg = str(excinfo.value)
        assert "no MC component contributes information in the fit " \
            "domain" in msg
        assert "2 MC component" in msg
        assert mf.scales is None

    @pytest.mark.parametrize("zero_index", [0, 1])
    def test_individually_zero_component_among_informative_ones_is_named(
            self, monkeypatch, zero_index):
        # One zero row among informative rows: the remaining scale stays
        # identifiable, yet the offending component's own scale would be
        # arbitrary, so the whole fit is rejected and names the offender.
        self._patch_minimize(monkeypatch, self._boom_minimize)
        data, mcs = _two_component_fixture(2.0, 0.5)
        names = [h.name for h in mcs]
        window = range(data.find_bin(0.03), data.find_bin(0.98) + 1)
        for i in window:
            mcs[zero_index].set_bin_content(i, 0.0)
        mf = MultiFitter(data, mcs, fit_range=(0.03, 0.98))
        with pytest.raises(ValueError) as excinfo:
            mf.fit()
        msg = str(excinfo.value)
        assert f"rejects MC component {zero_index}" in msg
        assert f"{names[zero_index]!r}" in msg
        assert "identically zero" in msg
        assert "unconstrained" in msg
        assert repr((0.03, 0.98)) in msg
        assert mf.scales is None

    def test_exactly_duplicated_components_raise_naming_both(
            self, monkeypatch):
        # Continuity with the superseded sum-only-recovery pin: two
        # bit-identical informative columns are rejected because only
        # their summed scale is determined by the chi-square.
        self._patch_minimize(monkeypatch, self._boom_minimize)
        shape = PATTERN.copy()
        total = 1.7
        data = _make_hist("dup_data")
        _fill_interior(data, total * shape)
        col_first = _make_hist("dup_first")
        _fill_interior(col_first, shape)
        col_second = _make_hist("dup_second")
        _fill_interior(col_second, shape)
        assert not np.any(shape == 0.0)     # genuinely informative rows
        mf = MultiFitter(data, [col_first, col_second],
                         fit_range=(0.03, 0.98))
        with pytest.raises(ValueError) as excinfo:
            mf.fit()
        msg = str(excinfo.value)
        assert "exactly duplicated MC components 0 ('dup_first') and " \
            "1 ('dup_second')" in msg
        assert "sum of their scales" in msg
        assert repr((0.03, 0.98)) in msg
        assert mf.scales is None

    def test_near_duplicate_but_distinct_components_still_fit(self):
        # Anti-overreach control: the degeneracy checks compare exact
        # values. Two templates differing in only a few bins remain fully
        # identifiable and recover the generating scales.
        ka, kb = 2.0, 0.5
        shape_a = PATTERN.copy()
        shape_b = PATTERN.copy()
        shape_b[0] += 0.37                  # two small deviations ...
        shape_b[6] -= 0.21                  # ... suffice to break equality
        assert not np.array_equal(shape_a, shape_b)
        data = _make_hist("near_dup_data")
        _fill_interior(data, ka * shape_a + kb * shape_b)
        mc_a = _make_hist("near_dup_A")
        _fill_interior(mc_a, shape_a)
        mc_b = _make_hist("near_dup_B")
        _fill_interior(mc_b, shape_b)
        mf = MultiFitter(data, [mc_a, mc_b], fit_range=(0.03, 0.98))
        scales = mf.fit()
        np.testing.assert_allclose(scales, [ka, kb], rtol=1e-4, atol=1e-7)


class TestStaleScalesClearedOnRefitFailure:
    """Review5 finding 3: every ``fit()`` entry clears ``self._scales``.

    A successful fit followed by any failing refit — validation failure,
    optimizer reporting failure, non-finite objective, non-finite scales —
    leaves ``scales`` at ``None``, so downstream code catching the
    exception cannot keep using results of the earlier fit. Optimizer-
    level injections patch ``scipy.optimize.minimize`` as imported inside
    ``fits.fit()``.
    """

    KA, KB = 2.0, 0.5

    def _fitted_after_success(self):
        data, mcs = _two_component_fixture(self.KA, self.KB)
        mf = MultiFitter(data, mcs, fit_range=(0.03, 0.98))
        first = mf.fit()
        np.testing.assert_allclose(first, [self.KA, self.KB],
                                   rtol=1e-4, atol=1e-7)
        np.testing.assert_array_equal(mf.scales, first)
        assert mf.scales is not first        # copies, not a shared array
        return mf

    @staticmethod
    def _patch_minimize(monkeypatch, fake):
        import scipy.optimize
        monkeypatch.setattr(scipy.optimize, "minimize", fake)

    def test_validation_failure_of_refit_clears_scales(self):
        mf = self._fitted_after_success()
        mf._mc.clear()                      # externally owned list emptied
        with pytest.raises(ValueError) as excinfo:
            mf.fit()
        assert "requires at least one MC component" in str(excinfo.value)
        assert mf.scales is None

    def test_failed_optimizer_report_clears_scales(self, monkeypatch):
        mf = self._fitted_after_success()
        self._patch_minimize(
            monkeypatch,
            lambda *a, **k: _FakeOptimizeResult(
                success=False, status=77,
                message="FAKE_REFIT_FAILURE",
                fun=0.0, x=np.ones(2)),
        )
        with pytest.raises(ValueError) as excinfo:
            mf.fit()
        assert "did not converge" in str(excinfo.value)
        assert "FAKE_REFIT_FAILURE" in str(excinfo.value)
        assert mf.scales is None

    def test_nonfinite_objective_of_refit_clears_scales(
            self, monkeypatch):
        mf = self._fitted_after_success()
        self._patch_minimize(
            monkeypatch,
            lambda *a, **k: _FakeOptimizeResult(
                success=True, status=0, message="CONVERGENCE (fake)",
                fun=float("nan"), x=np.array([self.KA, self.KB])),
        )
        with pytest.raises(ValueError) as excinfo:
            mf.fit()
        msg = str(excinfo.value)
        assert "non-finite" in msg and "objective" in msg
        assert mf.scales is None

    def test_nonfinite_scales_of_refit_clear_scales(self, monkeypatch):
        mf = self._fitted_after_success()
        self._patch_minimize(
            monkeypatch,
            lambda *a, **k: _FakeOptimizeResult(
                success=True, status=0, message="CONVERGENCE (fake)",
                fun=3.5, x=np.array([float("nan"), self.KB])),
        )
        with pytest.raises(ValueError) as excinfo:
            mf.fit()
        assert "returned scales are non-finite" in str(excinfo.value)
        assert mf.scales is None


# ---------------------------------------------------------------------------
# Review6 Stage F: weighted design-matrix column-rank identifiability rule
# ---------------------------------------------------------------------------

class TestMultiFitterDesignMatrixRank:
    """The design-matrix rank rule behind the zero-row / duplicate checks.

    The exact-value refusals catch only two named special cases of
    linear dependence. ``fit()`` now also forms the weighted design
    matrix the chi-square actually uses,
    ``design = (mc_vals / errors).T`` of shape ``(n_selected, n_comp)``,
    and rejects any design whose numerical column rank
    (``numpy.linalg.matrix_rank`` with its default SVD tolerance) is
    below the component count: proportional templates, a template equal
    to a combination of the others, and more components than selected
    bins. A nearly-but-not-exactly dependent design with full numerical
    rank keeps fitting. A design with non-finite entries is refused by
    its own diagnostic before any rank is computed. Optimizer-level
    injections patch ``scipy.optimize.minimize`` as imported inside
    ``fits.fit()``.
    """

    @staticmethod
    def _boom_minimize(*_args, **_kwargs):
        """Sentinel proving optimization is never reached."""
        raise AssertionError("scipy.optimize.minimize must not be reached")

    @staticmethod
    def _patch_minimize(monkeypatch, fake):
        import scipy.optimize
        monkeypatch.setattr(scipy.optimize, "minimize", fake)

    def test_proportional_templates_rejected_with_rank_count_and_names(
            self, monkeypatch):
        # The review's direct probe: data = 5*m1 with m2 = 2*m1. This
        # passes np.array_equal, and the optimizer historically returned
        # approximately [1.4, 1.8] — one of infinitely many equally good
        # non-negative splits of the identifiable combination
        # scale1 + 2*scale2 = 5. The rank rule must refuse it before
        # any optimizer work.
        self._patch_minimize(monkeypatch, self._boom_minimize)
        data = _make_hist("prop_data")
        _fill_interior(data, 5.0 * PATTERN)
        mc_a = _make_hist("prop_mc_a")
        _fill_interior(mc_a, PATTERN)
        mc_b = _make_hist("prop_mc_b")
        _fill_interior(mc_b, 2.0 * PATTERN)
        mf = MultiFitter(data, [mc_a, mc_b], fit_range=(0.03, 0.98))
        with pytest.raises(ValueError) as excinfo:
            mf.fit()
        msg = str(excinfo.value)
        assert "numerical rank 1" in msg
        assert "n_comp = 2" in msg
        assert "'prop_mc_a'" in msg and "'prop_mc_b'" in msg
        assert f"selected bins 1..{NBINS}" in msg
        assert repr((0.03, 0.98)) in msg
        assert mf.scales is None

    def test_three_component_linear_combination_rejected_with_all_names(
            self, monkeypatch):
        # m3 = m1 + m2 over the selected domain: every row is nonzero
        # and the three rows are pairwise unequal, so the zero-row and
        # duplicate special cases pass and only the rank rule fires,
        # naming all three components.
        self._patch_minimize(monkeypatch, self._boom_minimize)
        shape_a = PATTERN.copy()
        shape_b = PATTERN[::-1].copy() + 0.05
        shape_c = shape_a + shape_b
        assert not np.array_equal(shape_a, shape_b)
        assert not np.array_equal(shape_a, shape_c)
        assert not np.array_equal(shape_b, shape_c)
        assert bool((shape_a != 0.0).all())
        assert bool((shape_b != 0.0).all())
        assert bool((shape_c != 0.0).all())
        data = _make_hist("comb_data")
        _fill_interior(data, 0.9 * shape_a + 1.4 * shape_b + 0.2 * shape_c)
        mc_a = _make_hist("comb_mc_a")
        _fill_interior(mc_a, shape_a)
        mc_b = _make_hist("comb_mc_b")
        _fill_interior(mc_b, shape_b)
        mc_c = _make_hist("comb_mc_c")
        _fill_interior(mc_c, shape_c)
        mf = MultiFitter(data, [mc_a, mc_b, mc_c], fit_range=(0.03, 0.98))
        with pytest.raises(ValueError) as excinfo:
            mf.fit()
        msg = str(excinfo.value)
        assert "numerical rank 2" in msg
        assert "n_comp = 3" in msg
        for name in ("comb_mc_a", "comb_mc_b", "comb_mc_c"):
            assert f"'{name}'" in msg
        assert mf.scales is None

    def test_more_components_than_selected_bins_rejected(self, monkeypatch):
        # Four distinct nonzero templates on a one-bin window: the
        # design matrix has shape (1, 4), so its rank is at most
        # 1 < n_comp = 4 regardless of the template values.
        self._patch_minimize(monkeypatch, self._boom_minimize)
        lo, hi = 0.32, 0.38
        data = _make_hist("onebin_data")
        _fill_interior(data, 2.0 * PATTERN)
        mcs = []
        for k in range(4):
            h = _make_hist(f"onebin_mc_{k}")
            _fill_interior(h, (k + 1.0) * PATTERN + 0.25)
            mcs.append(h)
        i_min = data.find_bin(lo)
        i_max = data.find_bin(hi)
        assert i_min == i_max                # genuinely a one-bin window
        mf = MultiFitter(data, mcs, fit_range=(lo, hi))
        with pytest.raises(ValueError) as excinfo:
            mf.fit()
        msg = str(excinfo.value)
        assert "numerical rank 1" in msg
        assert "n_comp = 4" in msg
        assert f"selected bins {i_min}..{i_max}" in msg
        for k in range(4):
            assert f"'onebin_mc_{k}'" in msg
        assert mf.scales is None

    def test_near_dependent_full_rank_pair_still_fits_through_optimizer(
            self, monkeypatch):
        # Anti-overreach control: m2 = 2*m1 + eps with eps = 1e-3 on
        # several bins keeps the design matrix at full numerical rank,
        # so the fit must run the optimizer and return finite
        # non-negative scales. The near-dependence is visible through
        # the design_condition_number diagnostic, never through a
        # rejection.
        shape_a = PATTERN.copy()
        shape_b = 2.0 * PATTERN.copy()
        shape_b[[0, 3, 5, 8]] += 1e-3
        data = _make_hist("near_dep_data")
        _fill_interior(data, 1.3 * shape_a + 0.6 * shape_b)
        mc_a = _make_hist("near_dep_a")
        _fill_interior(mc_a, shape_a)
        mc_b = _make_hist("near_dep_b")
        _fill_interior(mc_b, shape_b)
        # Anti-vacuous: with this perturbation the design matrix the
        # chi-square consumes has numerical rank 2 over the selected
        # bins, i.e. the rank rule itself would not fire.
        window = range(data.find_bin(0.03), data.find_bin(0.98) + 1)
        data_sel = np.array([data.get_bin_content(i) for i in window])
        errors_sel = np.sqrt(np.maximum(data_sel, 1.0))
        a_sel = np.array([mc_a.get_bin_content(i) for i in window])
        b_sel = np.array([mc_b.get_bin_content(i) for i in window])
        design = np.stack([a_sel / errors_sel, b_sel / errors_sel], axis=1)
        assert int(np.linalg.matrix_rank(design)) == 2

        import scipy.optimize
        real_minimize = scipy.optimize.minimize
        calls = []

        def recording_minimize(*args, **kwargs):
            calls.append(args)
            return real_minimize(*args, **kwargs)

        monkeypatch.setattr(scipy.optimize, "minimize", recording_minimize)
        mf = MultiFitter(data, [mc_a, mc_b], fit_range=(0.03, 0.98))
        scales = mf.fit()
        assert len(calls) == 1               # the optimizer really ran
        assert bool(np.all(np.isfinite(scales)))
        assert bool(np.all(scales >= 0.0))
        # The near-dependence is visible only through the diagnostic:
        # the fit is accepted, and the 2-norm condition number of the
        # design matrix is orders of magnitude above the well-conditioned
        # fixture's (about 2.7; this design computes to about 1.6e4)
        # because the chi-square valley along scale_a + 2*scale_b is
        # nearly flat. The scales are deliberately not pinned to the
        # generating (1.3, 0.6): on that nearly flat valley the optimizer
        # may land anywhere with scale_a + 2*scale_b ~ 2.5.
        cond = mf.design_condition_number
        assert cond is not None
        assert np.isfinite(cond) and cond > 100.0

    def test_exact_duplicate_diagnostic_precedes_the_rank_check(
            self, monkeypatch):
        # Order of checks is part of the contract: exactly duplicated
        # templates are still diagnosed by the duplicate message (which
        # names both components), never by the rank message.
        self._patch_minimize(monkeypatch, self._boom_minimize)
        shape = PATTERN.copy()
        data = _make_hist("order_data")
        _fill_interior(data, 1.7 * shape)
        first = _make_hist("order_first")
        _fill_interior(first, shape)
        second = _make_hist("order_second")
        _fill_interior(second, shape)
        mf = MultiFitter(data, [first, second], fit_range=(0.03, 0.98))
        with pytest.raises(ValueError) as excinfo:
            mf.fit()
        msg = str(excinfo.value)
        assert "exactly duplicated MC components 0 ('order_first') and " \
            "1 ('order_second')" in msg
        assert "full column rank" not in msg
        assert "design matrix" not in msg
        assert mf.scales is None

    def test_condition_number_diagnostic_follows_the_scales_discipline(
            self):
        # The 2-norm condition number of the design matrix is a plain
        # float, reset to None on every fit() entry and written only
        # after a fully successful fit — the same stale-state discipline
        # as _scales, and never a second scale array.
        data, mcs = _two_component_fixture(2.0, 0.5)
        mf = MultiFitter(data, mcs, fit_range=(0.03, 0.98))
        assert mf.design_condition_number is None
        mf.fit()
        cond = mf.design_condition_number
        assert cond is not None
        assert not isinstance(cond, np.ndarray)
        assert isinstance(cond, float)
        assert np.isfinite(cond) and cond > 0.0
        # The stored number is the 2-norm condition number of the very
        # design matrix the chi-square consumes, recomputed here from the
        # histogram contents exactly the way fit() builds it.
        window = range(data.find_bin(0.03), data.find_bin(0.98) + 1)
        data_vals = np.array([data.get_bin_content(i) for i in window])
        mc_vals = np.array([[h.get_bin_content(i) for i in window]
                            for h in mcs])
        errors = np.sqrt(np.maximum(data_vals, 1.0))
        design = (mc_vals / errors).T
        assert cond == pytest.approx(float(np.linalg.cond(design, p=2)),
                                     rel=1e-12)
        # A failing refit (m2 proportional to m1 over the window) clears
        # the diagnostic together with the scales.
        window = range(data.find_bin(0.03), data.find_bin(0.98) + 1)
        for i in window:
            mcs[1].set_bin_content(i, 2.0 * mcs[0].get_bin_content(i))
        with pytest.raises(ValueError) as excinfo:
            mf.fit()
        assert "full column rank" in str(excinfo.value)
        assert mf.design_condition_number is None
        assert mf.scales is None

    @pytest.mark.parametrize("bad", [float("inf"), float("nan")],
                             ids=["inf", "nan"])
    def test_nonfinite_design_matrix_refused_by_name_before_optimizer(
            self, monkeypatch, bad):
        # A non-finite entry in the weighted design matrix is refused by
        # name instead of being fed to numpy.linalg.matrix_rank, whose
        # SVD would either fail outright or silently rank the matrix as
        # zero. The public fill policies reject non-finite deposits, so
        # such contents reach the fitter only through the documented
        # set_bin_content escape hatch. The refusal carries its own
        # diagnostic, never the rank message.
        self._patch_minimize(monkeypatch, self._boom_minimize)
        data = _make_hist("nonfinite_data")
        _fill_interior(data, 1.7 * PATTERN)
        first = _make_hist("nonfinite_mc_a")
        _fill_interior(first, PATTERN)
        second = _make_hist("nonfinite_mc_b")
        _fill_interior(second, PATTERN[::-1].copy())
        second.set_bin_content(3, bad)
        # Anti-vacuous: the corrupted bin is inside the selected window,
        # and the earlier named special cases (zero rows, duplicates)
        # deliberately do not fire for inf/nan contents, so the
        # non-finite design guard is the check under test.
        assert data.find_bin(0.03) <= 3 <= data.find_bin(0.98)
        stored = second.get_bin_content(3)
        if np.isnan(bad):
            assert np.isnan(stored)
        else:
            assert stored == bad
        first_vals = [first.get_bin_content(i) for i in range(1, NBINS + 1)]
        second_vals = [second.get_bin_content(i)
                       for i in range(1, NBINS + 1)]
        assert all(v != 0.0 for v in first_vals)
        assert all(v != 0.0 for v in second_vals)
        assert not np.array_equal(first_vals, second_vals)
        mf = MultiFitter(data, [first, second], fit_range=(0.03, 0.98))
        with pytest.raises(ValueError) as excinfo:
            mf.fit()
        msg = str(excinfo.value)
        assert "weighted design matrix" in msg
        assert "contains non-finite values" in msg
        assert f"selected bins 1..{NBINS}" in msg
        assert repr((0.03, 0.98)) in msg
        assert "full column rank" not in msg
        assert mf.scales is None
        assert mf.design_condition_number is None

    def test_optimizer_failure_refit_clears_condition_number(
            self, monkeypatch):
        # The diagnostic follows _scales on every failed refit, not just
        # rank refusals: a successful fit writes it, a refit whose
        # optimizer reports failure clears it together with the scales.
        data, mcs = _two_component_fixture(2.0, 0.5)
        mf = MultiFitter(data, mcs, fit_range=(0.03, 0.98))
        mf.fit()
        assert mf.design_condition_number is not None
        self._patch_minimize(
            monkeypatch,
            lambda *a, **k: _FakeOptimizeResult(
                success=False, status=77,
                message="FAKE_REFIT_FAILURE",
                fun=0.0, x=np.ones(2)),
        )
        with pytest.raises(ValueError) as excinfo:
            mf.fit()
        assert "did not converge" in str(excinfo.value)
        assert "FAKE_REFIT_FAILURE" in str(excinfo.value)
        assert mf.design_condition_number is None
        assert mf.scales is None
