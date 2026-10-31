"""Non-finite rejection policy for Histogram1D (review3 Phase P1).

Pinned contract (task card TASK-2026-08-26-code-review3, Phase P1):

- NaN, ``+inf`` and ``-inf`` raise ``ValueError`` from ``fill``,
  ``fill_array`` and ``find_bin``, scalar and weighted, before any bin
  content changes.
- Rejection is all-or-nothing: an array containing any non-finite
  element leaves bin contents bit-for-bit unchanged.
- The error message names the method, whether a value or a weight
  offended, the first offending position, and the total count.
- Finite-input behavior is frozen exactly: underflow to bin 0, overflow
  to nbins+1, in-range floor+1 clipped to [1, nbins]; CDF/probs cache
  invalidation on successful fills is untouched.
- Zero and negative weights stay legal (negative-weight fills are used
  by design elsewhere in the suite).
- No NumPy RuntimeWarning may be emitted for non-finite inputs.

Extended contract (task card TASK-2026-08-26-code-review5-fixes,
Phase P3, finding 4):

- ``scale``/``add`` validate the scalar factor as finite before any
  mutation (message shape: method, factor value, remediation);
- every mutating operation commits only an all-finite result computed
  on a private array: bins accumulated past the largest finite double
  by repeated finite weights, single ``fill_array`` calls whose bin
  sums overflow, and finite factors whose application yields infinity
  (including ``0 * inf`` products) reject atomically;
- a rejected arithmetic call keeps ``_contents`` bit-for-bit unchanged
  and preserves the sampling caches exactly (``None`` stays ``None``, a
  built array stays the same object).
"""

import math
import sys
import warnings

import numpy as np
import pytest

from na49_mixing.histogram import Histogram1D

NONFINITE = [float("nan"), float("inf"), float("-inf")]


def _make_hist():
    return Histogram1D(4, 0.0, 1.0, name="policy")


# ---------------------------------------------------------------------------
# Scalar fill — value rejection
# ---------------------------------------------------------------------------


class TestScalarFillValueRejection:
    @pytest.mark.parametrize("bad", NONFINITE)
    def test_nonfinite_value_raises_valueerror(self, bad):
        h = _make_hist()
        snapshot = h._contents.copy()
        with pytest.raises(ValueError) as excinfo:
            h.fill(bad)
        msg = str(excinfo.value)
        assert "fill" in msg
        assert "non-finite value" in msg
        assert math.isnan(h.get_bin_content(1)) is False
        np.testing.assert_array_equal(h._contents, snapshot)

    def test_message_names_position_and_count(self):
        h = _make_hist()
        with pytest.raises(ValueError) as excinfo:
            h.fill(float("nan"))
        assert "first at position 0" in str(excinfo.value)
        assert "1 non-finite value" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Scalar fill — weight rejection
# ---------------------------------------------------------------------------


class TestScalarFillWeightRejection:
    @pytest.mark.parametrize("bad", NONFINITE)
    def test_nonfinite_weight_raises_and_leaves_state_intact(self, bad):
        h = _make_hist()
        h.fill(0.25)          # seed one known entry first
        snapshot = h._contents.copy()
        with pytest.raises(ValueError) as excinfo:
            h.fill(0.75, weight=bad)
        msg = str(excinfo.value)
        assert "fill" in msg
        assert "non-finite weight" in msg
        assert "first at position 0" in msg
        assert "1 non-finite weight" in msg
        np.testing.assert_array_equal(h._contents, snapshot)
        # The histogram remains fully usable after a rejected call.
        h.fill(0.75, weight=2.0)
        assert h.get_bin_content(4) == pytest.approx(2.0)

    def test_nonfinite_weight_does_not_clobber_prior_bin(self):
        h = _make_hist()
        h.fill(0.25)
        with pytest.raises(ValueError):
            h.fill(0.25, weight=float("nan"))
        assert h.get_bin_content(2) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Vectorised fill — values and weights
# ---------------------------------------------------------------------------


class TestVectorizedValueRejection:
    @pytest.mark.parametrize("bad", NONFINITE)
    def test_single_nonfinite_element_raises(self, bad):
        h = _make_hist()
        snapshot = h._contents.copy()
        with pytest.raises(ValueError) as excinfo:
            h.fill_array(np.array([1.0, bad]))
        msg = str(excinfo.value)
        assert "fill_array" in msg
        assert "non-finite value" in msg
        np.testing.assert_array_equal(h._contents, snapshot)

    def test_mixed_array_names_index_one_and_count_one(self):
        # Acceptance criterion 2 of Phase P1 exactly.
        h = _make_hist()
        with pytest.raises(ValueError) as excinfo:
            h.fill_array(np.array([1.0, float("nan"), 0.5]))
        msg = str(excinfo.value)
        assert "first at position 1" in msg
        assert "1 non-finite value" in msg

    def test_multiple_offenders_report_first_index_and_total_count(self):
        h = _make_hist()
        with pytest.raises(ValueError) as excinfo:
            h.fill_array(np.array(
                [float("nan"), 1.0, float("nan"), float("nan")]))
        msg = str(excinfo.value)
        assert "first at position 0" in msg
        assert "3 non-finite values" in msg

    def test_later_offender_reports_its_own_position(self):
        h = _make_hist()
        with pytest.raises(ValueError) as excinfo:
            h.fill_array(np.array([1.0, 2.0, float("inf")]))
        msg = str(excinfo.value)
        assert "first at position 2" in msg
        assert "1 non-finite value" in msg


class TestVectorizedWeightRejection:
    @pytest.mark.parametrize("bad", NONFINITE)
    def test_nonfinite_weight_raises_with_named_kind(self, bad):
        h = _make_hist()
        h.fill(0.25)   # prior state that must survive untouched
        snapshot = h._contents.copy()
        with pytest.raises(ValueError) as excinfo:
            h.fill_array(
                np.array([-0.5, 0.3, 0.7]),
                weights=np.array([1.0, bad, 2.0]),
            )
        msg = str(excinfo.value)
        assert "fill_array" in msg
        assert "non-finite weight" in msg
        assert "first at position 1" in msg
        assert "1 non-finite weight" in msg
        np.testing.assert_array_equal(h._contents, snapshot)

    def test_multiple_bad_weights_counted(self):
        h = _make_hist()
        with pytest.raises(ValueError) as excinfo:
            h.fill_array(
                np.zeros(4),
                weights=np.array([float("-inf"), 1.0, 1.0, float("inf")]),
            )
        msg = str(excinfo.value)
        assert "first at position 0" in msg
        assert "2 non-finite weights" in msg

    def test_value_offense_takes_precedence_over_weight_offense(self):
        # Values are validated first; the raised error names the value.
        h = _make_hist()
        with pytest.raises(ValueError) as excinfo:
            h.fill_array(
                np.array([float("nan"), 1.0]),
                weights=np.array([float("nan"), 1.0]),
            )
        assert "non-finite value" in str(excinfo.value)


# ---------------------------------------------------------------------------
# All-or-nothing mutation guarantee
# ---------------------------------------------------------------------------


class TestAllOrNothingGuarantee:
    def _seeded(self):
        h = _make_hist()
        for v, w in ((-0.5, 1.5), (0.1, 1.0), (0.3, -2.0),
                     (0.55, 0.25), (0.9, 4.0), (1.5, 1.0)):
            h.fill(v, weight=w)
        return h

    def test_rejected_long_fill_keeps_contents_bit_for_bit(self):
        h = self._seeded()
        snapshot_bytes = h._contents.tobytes()
        long_values = [0.05, 0.15, 0.25, 0.35, 0.45, 0.95]
        bad_values = np.array(
            long_values + [float("nan")] + list(reversed(long_values)))
        with pytest.raises(ValueError):
            h.fill_array(bad_values)
        assert h._contents.tobytes() == snapshot_bytes

    def test_plain_python_sequence_with_nan_also_rejected(self):
        # fill_array converts its input through np.asarray, so a raw
        # Python sequence is accepted for finite content and rejected
        # under the policy when it contains a non-finite element.
        h = _make_hist()
        snapshot_bytes = h._contents.tobytes()
        with pytest.raises(ValueError) as excinfo:
            h.fill_array((0.1, float("nan"), 0.9))  # type: ignore[arg-type]
        assert "first at position 1" in str(excinfo.value)
        assert h._contents.tobytes() == snapshot_bytes

    def test_rejected_weighted_fill_keeps_contents_bit_for_bit(self):
        h = self._seeded()
        snapshot_bytes = h._contents.tobytes()
        xs = np.array([-0.9, 0.2, 0.4, 0.6])
        ws = np.array([1.0, 2.0, float("inf"), 3.0])
        with pytest.raises(ValueError):
            h.fill_array(xs, weights=ws)
        assert h._contents.tobytes() == snapshot_bytes

    def test_rejected_fill_keeps_sampling_cache_alive(self):
        # A rejected call must not invalidate the CDF/probs cache either;
        # sampling continues from the pre-failure distribution unchanged.
        h = self._seeded()
        rng = np.random.default_rng(1234)
        before = h.get_random_n(16, rng=np.random.default_rng(7))
        h.get_random(rng=rng)                    # force cache construction
        cdf_object = h._cdf
        probs_object = h._probs
        assert cdf_object is not None and probs_object is not None
        with pytest.raises(ValueError):
            h.fill_array(np.array([0.2, float("nan"), 0.8]))
        assert h._cdf is cdf_object
        assert h._probs is probs_object
        after = h.get_random_n(16, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(after, before)

    def test_contents_not_partially_updated_after_failure(self):
        h = self._seeded()
        per_bin_before = [
            h.get_bin_content(i) for i in range(h.nbins + 2)
        ]
        # Only the finite part would land in underflow/bin 1 if the call
        # were partial; verify nothing moved anywhere including under/
        # overflow bins.
        with pytest.raises(ValueError):
            h.fill_array(np.array([-1.0, 1.0, float("nan")]),
                         weights=np.array([10.0, 10.0, 1.0]))
        per_bin_after = [h.get_bin_content(i) for i in range(h.nbins + 2)]
        assert per_bin_after == per_bin_before


# ---------------------------------------------------------------------------
# find_bin parity with the fill paths
# ---------------------------------------------------------------------------


class TestFindBinParity:
    @pytest.mark.parametrize("method_args", [("value",), ("weight",)])
    def test_find_bin_rejects_nonfinite_like_fill_paths(self, method_args):
        kind, = method_args
        h = _make_hist()
        args_by_kind = {
            "value": [(float("nan"),), (float("inf"),), (float("-inf"),)],
            "weight": [(0.5, float("nan")), (0.5, float("inf")),
                       (0.5, float("-inf"))],
        }
        for args in args_by_kind[kind]:
            with pytest.raises(ValueError):
                getattr(h, "find_bin" if kind == "value" else "fill")(*args)

    def test_find_bin_agrees_with_where_fill_deposits(self):
        # For every probe point the incremented bin of a scalar fill must
        # be exactly find_bin(x); below-range -> 0, >= xhigh -> nbins+1.
        probes = [-5.0, 0.0, 0.05, 0.3, 0.6, 0.99, 1.0, 1.7]
        for x in probes:
            hi = Histogram1D(4, 0.0, 1.0)
            hi.fill(x)
            changed = [
                i for i in range(hi.nbins + 2)
                if hi.get_bin_content(i) != 0.0
            ]
            assert changed == [hi.find_bin(x)], f"x={x!r}"

    def test_scalar_and_vectorized_agree_for_shared_finite_values(self):
        values = np.array([-0.2, 0.0, 0.25, 0.5, 1.0, 1.2])
        h_s = Histogram1D(4, 0.0, 1.0)
        h_v = Histogram1D(4, 0.0, 1.0)
        for v in values:
            h_s.find_bin(v)          # never raises on finite input
            h_s.fill(v)
        h_v.fill_array(values)
        np.testing.assert_allclose(h_s._contents, h_v._contents)


# ---------------------------------------------------------------------------
# Zero and negative weights stay legal
# ---------------------------------------------------------------------------


class TestZeroAndNegativeWeightsLegal:
    def test_zero_weight_accepted_everywhere(self):
        h_s = Histogram1D(4, 0.0, 1.0)
        h_v = Histogram1D(4, 0.0, 1.0)
        h_s.fill(0.3, weight=0.0)               # no exception
        h_v.fill_array(np.array([0.3]), weights=np.array([0.0]))  # no exception
        np.testing.assert_allclose(h_s._contents, h_v._contents)
        assert h_v.get_entries() == 0.0

    def test_negative_weight_accepted_scalar_and_vectorized(self):
        h_s = Histogram1D(4, 0.0, 1.0)
        h_v = Histogram1D(4, 0.0, 1.0)
        h_s.fill(0.3, weight=-2.0)
        # One negative-weight element per region: underflow, three
        # in-range bins, overflow.
        h_v.fill_array(
            np.array([-0.5, 0.1, 0.5, 0.95, 1.5]),
            weights=np.array([-2.0, -2.0, -2.0, -2.0, -2.0]),
        )
        assert h_s.get_bin_content(h_s.find_bin(0.3)) == pytest.approx(-2.0)
        assert h_v.get_entries() == pytest.approx(-6.0)
        assert h_v.get_bin_content(0) == pytest.approx(-2.0)
        assert h_v.get_bin_content(h_v.nbins + 1) == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# Frozen finite-input routing
# ---------------------------------------------------------------------------


class TestFiniteRoutingFrozen:
    def test_underflow_overflow_and_clip_unaffected(self):
        h = Histogram1D(4, 0.0, 1.0)
        h.fill(-0.01)                 # underflow -> bin 0
        h.fill(1.0)                   # x == xhigh -> overflow bin nbins+1
        h.fill(0.999999)              # in-range last bin, no clip surprise
        h.fill_array(np.array([0.99999999999]))  # clipped into [1, nbins]
        assert h.get_bin_content(0) == pytest.approx(1.0)
        assert h.get_bin_content(h.nbins + 1) == pytest.approx(1.0)
        assert h.get_bin_content(h.nbins) == pytest.approx(2.0)

    def test_successful_fills_invalidate_cdf_cache_as_before(self):
        h = Histogram1D(4, 0.0, 1.0)
        h.get_random(rng=np.random.default_rng(0))   # empty distribution
        assert isinstance(h._cdf, np.ndarray)        # cache built
        h.fill(0.5)
        assert h._cdf is None and h._probs is None   # invalidated by fill
        h.get_random_n(4, rng=np.random.default_rng(0))
        assert isinstance(h._cdf, np.ndarray)
        h.fill_array(np.array([0.6]))
        assert h._cdf is None and h._probs is None


# ---------------------------------------------------------------------------
# No RuntimeWarning for non-finite inputs
# ---------------------------------------------------------------------------


class TestNoRuntimeWarningForNonfiniteInputs:
    @staticmethod
    def _assert_clean_valueerror(callable_, *args, **kwargs):
        with warnings.catch_warnings():
            # Any warning at all — RuntimeWarning in particular — fails
            # the check; only the policy ValueError may escape.
            warnings.simplefilter("error")
            with pytest.raises(ValueError) as excinfo:
                callable_(*args, **kwargs)
            assert not isinstance(excinfo.value, Warning)

    @pytest.mark.parametrize("bad", NONFINITE)
    def test_scalar_fill_emits_no_warning(self, bad):
        h = _make_hist()
        self._assert_clean_valueerror(h.fill, bad)
        self._assert_clean_valueerror(h.fill, 0.5, bad)

    @pytest.mark.parametrize("bad", NONFINITE)
    def test_vectorized_fill_emits_no_warning(self, bad):
        h = _make_hist()
        self._assert_clean_valueerror(
            h.fill_array, np.array([1.0, 2.0, bad]))
        self._assert_clean_valueerror(
            h.fill_array,
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, bad, 1.0]),
        )

    @pytest.mark.parametrize("bad", NONFINITE)
    def test_find_bin_emits_no_warning(self, bad):
        h = _make_hist()
        self._assert_clean_valueerror(h.find_bin, bad)

    def test_nan_deposits_nothing_into_first_in_range_bin(self):
        # The historical defect: NaN landed silently in bin 1 via an
        # invalid int cast. Verify bin 1 stays empty under warning-as-error.
        h = _make_hist()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError):
                h.fill_array(np.array([1.0, float("nan"), 0.5]))
        assert h.get_bin_content(1) == 0.0


# ---------------------------------------------------------------------------
# review5 finding 4: scalar-factor finiteness for scale() / add()
# ---------------------------------------------------------------------------

def _rejected_arithmetic_call(h: Histogram1D, kind: str) -> None:
    """Invoke the arithmetic variant named by *kind* so its commit MUST be refused.

    The shared prepared histogram carries 1e308 in the bin that 0.3 maps
    to (bin 2 of a Histogram1D(4, 0.0, 1.0)), so every branch here has a
    deterministic path to a non-finite result:

    - ``scale``: finite factor whose application overflows (2e308);
    - ``add``: ``0 * inf`` from zero-weighted tainted storage;
    - ``fill``: accumulation past the largest finite double;
    - ``fill_array``: two finite weights overflowing one bin sum.
    """
    if kind == "scale":
        h.scale(2.0)
    elif kind == "add":
        other = Histogram1D(h.nbins, h.xlow, h.xhigh, name="tainted")
        other.set_bin_content(3, float("inf"))
        h.add(other, scale=0.0)
    elif kind == "fill":
        h.fill(0.3, weight=1e308)
    elif kind == "fill_array":
        h.fill_array(
            np.array([0.3, 0.3]), weights=np.array([5e307, 6e307])
        )
    else:  # pragma: no cover - guards against typos in test tables
        raise AssertionError(f"unknown arithmetic kind {kind!r}")


ARITHMETIC_KINDS = ("scale", "add", "fill", "fill_array")


class TestScalarFactorFinitenessPolicy:
    """scale()/add() validate the scalar factor before any mutation."""

    @staticmethod
    def _pair():
        h = Histogram1D(4, 0.0, 1.0, name="left")
        h.fill_array(np.array([0.1, 0.3, 0.5, 0.7]),
                     weights=np.array([1.0, -2.0, 0.5, 3.0]))
        o = Histogram1D(4, 0.0, 1.0, name="right")
        o.fill_array(np.array([0.2, 0.6]), weights=np.array([1.5, -0.25]))
        return h, o

    @pytest.mark.parametrize("bad", NONFINITE)
    def test_scale_rejects_before_any_mutation(self, bad):
        h, _ = self._pair()
        snapshot = h._contents.tobytes()
        with pytest.raises(ValueError) as excinfo:
            h.scale(bad)
        msg = str(excinfo.value)
        assert msg.startswith("scale:")
        assert repr(bad) in msg
        assert "only finite inputs are accepted" in msg
        assert h._contents.tobytes() == snapshot

    @pytest.mark.parametrize("bad", NONFINITE)
    def test_add_rejects_before_any_mutation(self, bad):
        h, o = self._pair()
        snapshot = h._contents.tobytes()
        with pytest.raises(ValueError) as excinfo:
            h.add(o, scale=bad)
        msg = str(excinfo.value)
        assert msg.startswith("add:")
        assert repr(bad) in msg
        assert "only finite inputs are accepted" in msg
        assert h._contents.tobytes() == snapshot

    def test_scale_validation_precedes_axis_compatibility_check(self):
        # Pinned ordering choice: argument validity (finite scale) is
        # reported before cross-argument axis faults.
        h = _make_hist()
        wrong = Histogram1D(5, 0.0, 1.0)
        with pytest.raises(ValueError) as excinfo:
            h.add(wrong, scale=float("nan"))
        msg = str(excinfo.value)
        assert "non-finite scale rejected" in msg
        assert "different axes" not in msg

    def test_scale_nan_on_zero_bins_preserves_zero_storage_and_cache(self):
        # Acceptance criterion: even zero bins end up NaN in no surviving
        # state; contents and cache state are identical before/after.
        h = Histogram1D(4, 0.0, 1.0)
        snapshot = h._contents.tobytes()
        h.get_random(rng=np.random.default_rng(0))   # empty-policy cache
        cdf_obj = h._cdf
        probs_obj = h._probs                          # None when empty
        assert cdf_obj is not None and probs_obj is None
        with pytest.raises(ValueError):
            h.scale(float("nan"))
        assert np.isfinite(h._contents).all()
        assert np.all(h._contents == 0.0)
        assert h._contents.tobytes() == snapshot
        assert h._cdf is cdf_obj and h._probs is probs_obj


# ---------------------------------------------------------------------------
# review5 finding 4: atomic commit + cache preservation on overflow paths
# ---------------------------------------------------------------------------


class TestArithmeticAtomicityAndCaches:
    @staticmethod
    def _prepared_hist():
        h = Histogram1D(4, 0.0, 1.0, name="atomic")
        h.fill(0.3, weight=1e308)       # last successful mutation
        return h

    @pytest.mark.parametrize("kind", ARITHMETIC_KINDS)
    def test_never_built_caches_stay_none_after_rejection(self, kind):
        h = self._prepared_hist()
        assert h._cdf is None and h._probs is None
        with pytest.raises(ValueError):
            _rejected_arithmetic_call(h, kind)
        assert h._cdf is None and h._probs is None
        assert np.isfinite(h._contents).all()

    @pytest.mark.parametrize("kind", ARITHMETIC_KINDS)
    def test_built_caches_keep_same_object_and_distribution(self, kind):
        h = self._prepared_hist()
        rng_a = np.random.default_rng(5)
        before = [h.get_random(rng=rng_a) for _ in range(8)]
        cdf_obj = h._cdf
        probs_obj = h._probs
        assert cdf_obj is not None and probs_obj is not None
        snapshot = h._contents.tobytes()
        with pytest.raises(ValueError) as excinfo:
            _rejected_arithmetic_call(h, kind)
        msg = str(excinfo.value)
        assert kind in msg
        assert "would leave" in msg
        assert "non-finite" in msg
        assert "unchanged" in msg
        assert not isinstance(excinfo.value, Warning)
        assert h._contents.tobytes() == snapshot
        assert h._cdf is cdf_obj and h._probs is probs_obj
        rng_b = np.random.default_rng(5)
        after = [h.get_random(rng=rng_b) for _ in range(8)]
        assert after == before


class TestAccumulationOverflowRejection:
    """The concrete reproducers named by review finding 4 / Phase P3."""

    def test_repeated_finite_weight_fills_overflow_one_bin_atomically(self):
        h = Histogram1D(4, 0.0, 1.0)
        h.fill(0.5, weight=1e308)               # succeeds: 1e308 finite
        snapshot = h._contents.tobytes()
        with pytest.raises(ValueError) as excinfo:
            h.fill(0.5, weight=1e308)           # 2e308 -> inf
        msg = str(excinfo.value)
        assert msg.startswith("fill:")
        assert "a deposit of 1e+308 into bin 3" in msg
        assert "non-finite" in msg
        assert h._contents.tobytes() == snapshot
        assert h.get_bin_content(3) == 1e308
        # The histogram remains fully usable after the rejected call.
        h.fill(0.85)
        assert h.get_bin_content(4) == pytest.approx(1.0)

    def test_single_fill_array_bin_sum_overflow_is_atomic(self):
        h = Histogram1D(4, 0.0, 1.0)
        h.fill(0.25, weight=7.7e307)            # state before failed call
        snapshot = h._contents.tobytes()
        with pytest.raises(ValueError) as excinfo:
            h.fill_array(
                np.array([0.25, 0.25]),
                weights=np.array([6.6e307, 5.5e307]),
            )
        # Partial sums stay finite (14.3e307); the final sum 19.8e307
        # exceeds the largest finite double and must refuse the commit.
        assert "fill_array" in str(excinfo.value)
        assert h._contents.tobytes() == snapshot
        assert h.get_bin_content(2) == pytest.approx(7.7e307)

    def test_finite_factor_whose_application_overflows_is_rejected(self):
        # Card acceptance case: contents 1e300 scaled by 1e300 -> inf.
        h = Histogram1D(4, 0.0, 1.0)
        h.set_bin_content(2, 1e300)             # out-of-policy constructor
        snapshot = h._contents.tobytes()
        with pytest.raises(ValueError) as excinfo:
            h.scale(1e300)
        msg = str(excinfo.value)
        assert msg.startswith("scale:")
        assert "scaling by 1e+300" in msg
        assert "would leave 1 non-finite bin content" in msg
        assert repr(float("inf")) in msg
        assert h._contents.tobytes() == snapshot

    def test_add_zero_weight_times_infinite_storage_is_rejected(self):
        # Card acceptance case: 0 * inf must never survive an add().
        h = Histogram1D(4, 0.0, 1.0)
        h.fill(0.15, weight=2.0)
        snapshot = h._contents.tobytes()
        other = Histogram1D(4, 0.0, 1.0, name="tainted")
        other.set_bin_content(3, float("inf"))
        with pytest.raises(ValueError) as excinfo:
            h.add(other, scale=0.0)
        msg = str(excinfo.value)
        assert msg.startswith("add:")
        assert "would leave 1 non-finite bin content" in msg
        assert "got nan" in msg
        assert h._contents.tobytes() == snapshot

    @pytest.mark.parametrize("bad", NONFINITE)
    def test_scale_zero_factor_on_nonfinite_storage_also_rejected(self, bad):
        # A non-finite SOURCE element can never be laundered through
        # arithmetic: any result array still carrying it rejects.
        h = Histogram1D(4, 0.0, 1.0)
        h.set_bin_content(2, bad)
        snapshot = h._contents.tobytes()
        with pytest.raises(ValueError):
            h.scale(2.0)
        assert h._contents.tobytes() == snapshot


class TestLegalLargeArithmeticStillAccepted:
    """Sanity guard: valid work stays valid under the stricter policy."""

    def test_accumulation_to_largest_finite_double_accepted(self):
        h = Histogram1D(4, 0.0, 1.0)
        top = sys.float_info.max / 2.0
        h.fill(0.3, weight=top)
        h.fill(0.3, weight=top)          # exact doubling -> max double
        assert h.get_bin_content(2) == sys.float_info.max

    def test_zero_factor_clears_finite_contents_and_invalidates_cache(self):
        h = Histogram1D(4, 0.0, 1.0)
        h.fill_array(np.linspace(0.05, 0.95, 5))
        h.get_random(rng=np.random.default_rng(0))
        h.scale(0.0)
        assert np.all(h._contents == 0.0)
        assert h._cdf is None and h._probs is None

    def test_scaled_add_matches_manual_computation_exactly(self):
        h = Histogram1D(4, 0.0, 1.0)
        o = Histogram1D(4, 0.0, 1.0)
        h.fill(0.35, weight=10.0)
        o.fill(0.55, weight=-2.5)
        h.add(o, scale=8.0)
        expected = np.zeros(6)
        expected[2] = 10.0
        expected[3] += -20.0             # find_bin(0.55) == 3
        np.testing.assert_array_equal(h._contents, expected)


# ---------------------------------------------------------------------------
# No RuntimeWarning on overflow rejection paths
# ---------------------------------------------------------------------------


class TestNoRuntimeWarningOnOverflowPaths:
    @pytest.mark.parametrize("kind", ARITHMETIC_KINDS)
    def test_rejected_arithmetic_emits_no_warning(self, kind):
        h = Histogram1D(4, 0.0, 1.0)
        h.fill(0.3, weight=1e308)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError) as excinfo:
                _rejected_arithmetic_call(h, kind)
            assert not isinstance(excinfo.value, Warning)

    def test_overflowing_multiplication_on_success_path_emits_no_warning(self):
        h = Histogram1D(4, 0.0, 1.0)
        h.set_bin_content(2, 1e300)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError):
                h.scale(1e300)


# ---------------------------------------------------------------------------
# review5 widening: signed-overflow matrix and NumPy-scalar factor parity
# ---------------------------------------------------------------------------


class TestSignedOverflowAtomicity:
    """The storage invariant is sign-symmetric: -inf results reject too.

    Every kind below drives one finite-sign overflow — a negative bin
    scaled to -inf, repeated negative deposits accumulating past the
    smallest finite double, sums of large same-sign bins to +inf, a
    ``0 * (-inf)`` taint, and a multi-deposit negative fill_array sum.
    """

    @staticmethod
    def _histogram_and_trigger(kind: str):
        """Build (prepared finite histogram, zero-arg failing trigger).

        Preparation uses only in-policy public calls except where noted;
        each histogram carries some positive mass so forced cache builds
        produce real CDF/probs arrays for identity checks.
        """
        if kind == "scale-negative":
            h = Histogram1D(4, 0.0, 1.0)
            h.fill(0.25, weight=-1e300)     # seeded out-of-policy-free
            h.fill(0.85, weight=1.0)

            def trigger() -> None:
                # -1e300 * 1e300 would land as -inf.
                h.scale(1e300)
        elif kind == "fill-negative-accumulation":
            h = Histogram1D(4, 0.0, 1.0)
            h.fill(0.3, weight=-1e308)      # last successful mutation
            h.fill(0.85, weight=1.0)

            def trigger() -> None:
                # -1e308 + -1e308 overflows to -inf.
                h.fill(0.3, weight=-1e308)
        elif kind == "add-positive-overflow":
            h = Histogram1D(4, 0.0, 1.0)
            h.fill(0.3, weight=1e308)
            h.fill(0.85, weight=1.0)
            other = Histogram1D(4, 0.0, 1.0, name="heavy")
            other.set_bin_content(2, 1e308)   # documented escape hatch

            def trigger() -> None:
                # 1e308 + 1e308 overflows to +inf under scale 1.
                h.add(other, scale=1.0)
        elif kind == "add-negative-infinity":
            h = Histogram1D(4, 0.0, 1.0)
            h.fill(0.15, weight=2.0)
            h.fill(0.55, weight=1.0)
            other = Histogram1D(4, 0.0, 1.0, name="tainted-negative")
            other.set_bin_content(3, float("-inf"))

            def trigger() -> None:
                # 0 * -inf keeps the historic NaN/-inf poison alive.
                h.add(other, scale=0.0)
        elif kind == "fill-array-negative-overflow":
            h = Histogram1D(4, 0.0, 1.0)
            h.fill(0.25, weight=-7e307)
            h.fill(0.85, weight=1.0)

            def trigger() -> None:
                # Partial sums stay finite; the total reaches -inf.
                h.fill_array(
                    np.array([0.25, 0.25]),
                    weights=np.array([-5e307, -6e307]),
                )
        else:   # pragma: no cover - guards against typos in test tables
            raise AssertionError(f"unknown signed kind {kind!r}")
        return h, trigger

    @pytest.mark.parametrize(
        "kind",
        [
            "scale-negative",
            "fill-negative-accumulation",
            "add-positive-overflow",
            "add-negative-infinity",
            "fill-array-negative-overflow",
        ],
    )
    def test_signed_overflow_rejected_with_cache_identity_kept(self, kind):
        h, trigger = self._histogram_and_trigger(kind)
        snapshot = h._contents.tobytes()
        h.get_random(rng=np.random.default_rng(3))   # force cache build
        cdf_obj = h._cdf
        probs_obj = h._probs
        assert cdf_obj is not None and probs_obj is not None
        with pytest.raises(ValueError) as excinfo:
            trigger()
        msg = str(excinfo.value)
        assert "non-finite" in msg
        assert "unchanged" in msg
        assert not isinstance(excinfo.value, Warning)
        assert h._contents.tobytes() == snapshot
        # Cache identity: the very objects built before the failed call.
        assert h._cdf is cdf_obj
        assert h._probs is probs_obj

    @pytest.mark.parametrize(
        "kind",
        [
            "scale-negative",
            "fill-negative-accumulation",
            "add-positive-overflow",
            "add-negative-infinity",
            "fill-array-negative-overflow",
        ],
    )
    def test_signed_overflow_never_builds_caches_from_scratch(self, kind):
        h, trigger = self._histogram_and_trigger(kind)
        assert h._cdf is None and h._probs is None
        with pytest.raises(ValueError):
            trigger()
        assert h._cdf is None and h._probs is None


class TestNumpyScalarFactorParity:
    """np.float64 factors behave exactly like Python floats everywhere."""

    @staticmethod
    def _pair():
        h = Histogram1D(4, 0.0, 1.0, name="left")
        h.fill(0.35, weight=3.0)
        o = Histogram1D(4, 0.0, 1.0, name="right")
        o.fill(0.55, weight=-2.5)
        return h, o

    def test_numpy_float_factor_scales_exactly_and_invalidates_cache(self):
        h, _ = self._pair()
        h.get_random(rng=np.random.default_rng(0))   # build cache first
        h.scale(np.float64(2.0))                     # type: ignore[arg-type]
        assert type(h.get_bin_content(2)) is float
        assert h.get_bin_content(2) == pytest.approx(6.0)
        assert h._cdf is None and h._probs is None

    @pytest.mark.parametrize("method", ["scale", "add"])
    @pytest.mark.parametrize("bad", NONFINITE)
    def test_numpy_nonfinite_factor_rejected_before_mutation(
            self, method, bad):
        factor = np.float64(bad)
        if method == "scale":
            h, o = self._pair()

            def call():
                h.scale(factor)                        # type: ignore[arg-type]
            target = h
        else:
            h, o = self._pair()

            def call():
                h.add(o, scale=factor)                 # type: ignore[arg-type]
            target = h
        snapshot = target._contents.tobytes()
        with pytest.raises(ValueError) as excinfo:
            call()
        msg = str(excinfo.value)
        assert msg.startswith(f"{method}:")
        assert "non-finite" in msg
        assert "only finite inputs are accepted" in msg
        assert target._contents.tobytes() == snapshot
