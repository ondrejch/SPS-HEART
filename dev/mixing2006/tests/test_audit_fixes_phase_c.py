"""Phase-C audit-fix regression tests (findings 5 and 6 of the physics code review).

Covers:

- ``save_histograms()`` refuses normalized-key collisions (finding 5);
  non-colliding saves keep round-tripping through ``load_histograms()``
  (loaded names are the sanitized storage keys — restoring original
  names was an explicitly optional stretch and is not implemented).
- Eager domain validation for ``Histogram1D``, ``TabulatedPDF`` and
  ``BreitWigner`` construction/update (finding 6), each raising a clear
  ``ValueError`` naming the invalid argument.
- Zero-width Breit-Wigners stay legal (delta at ``rmas``), guarding the
  finding-6 validation against regressing the Phase-A delta policy.
- ``parallel._patch_generator_mass_width`` keeps working under eager
  validation because it commits its final parameter set atomically;
  transient states never reach the validated setters.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from na49_mixing.breit_wigner import BreitWigner
from na49_mixing.data_io import load_histograms, save_histograms
from na49_mixing.generator import Generator, TabulatedPDF
from na49_mixing.histogram import Histogram1D
from na49_mixing.parallel import _patch_generator_mass_width
from na49_mixing.reaction import Reaction


def _assert_message_mentions(excinfo, *fragments):
    msg = str(excinfo.value)
    for fragment in fragments:
        assert fragment in msg, (
            f"expected {fragment!r} in error message, got: {msg}"
        )


# ---------------------------------------------------------------------------
# Finding 5 — normalized-key collisions on save
# ---------------------------------------------------------------------------


class TestSaveHistogramCollisions:
    def test_collision_raises_and_names_both_offenders(self, tmp_path):
        h1 = Histogram1D(5, 0.0, 1.0, name="a b")
        h1.fill(0.5)
        h2 = Histogram1D(7, 0.0, 2.0, name="a_b")
        h2.fill(1.0)
        out = tmp_path / "collide"
        with pytest.raises(ValueError) as excinfo:
            save_histograms(str(out), {"a b": h1, "a_b": h2})
        # Both distinct source names must be named, plus the shared key.
        _assert_message_mentions(excinfo, "a b", "a_b")
        # Nothing may be written when the save is refused.
        assert not out.exists()
        assert not tmp_path.joinpath("collide.npz").exists()

    def test_non_colliding_names_round_trip(self, tmp_path):
        h_ab = Histogram1D(5, 0.0, 1.0, name="a b")
        h_cd = Histogram1D(4, -2.0, 2.0, name="c-d")
        h_ab.fill(0.3)
        h_ab.fill(0.7)
        h_cd.fill(-1.0, weight=2.0)
        out = tmp_path / "ok.npz"
        save_histograms(str(out), {"a b": h_ab, "c-d": h_cd})
        loaded = load_histograms(str(out))
        assert sorted(loaded.keys()) == ["a b", "c-d"]
        ha, hc = loaded["a b"], loaded["c-d"]
        assert (ha.nbins, ha.xlow, ha.xhigh) == (5, 0.0, 1.0)
        assert (hc.nbins, hc.xlow, hc.xhigh) == (4, -2.0, 2.0)
        np.testing.assert_allclose(ha._contents, h_ab._contents)
        np.testing.assert_allclose(hc._contents, h_cd._contents)


# ---------------------------------------------------------------------------
# Finding 6 — Histogram1D constructor validation
# ---------------------------------------------------------------------------


class TestHistogram1DValidation:
    @pytest.mark.parametrize("nbins", [0, -3])
    def test_non_positive_bins_rejected(self, nbins):
        with pytest.raises(ValueError) as excinfo:
            Histogram1D(nbins, 0.0, 1.0)
        _assert_message_mentions(excinfo, "nbins")

    def test_fractional_bins_rejected(self):
        with pytest.raises(ValueError) as excinfo:
            Histogram1D(10.5, 0.0, 1.0)
        _assert_message_mentions(excinfo, "nbins")

    @pytest.mark.parametrize("xlow, xhigh", [
        (float("nan"), 1.0),
        (0.0, float("nan")),
        (float("-inf"), 1.0),
        (0.0, float("inf")),
    ])
    def test_non_finite_edges_rejected(self, xlow, xhigh):
        with pytest.raises(ValueError) as excinfo:
            Histogram1D(10, xlow, xhigh)
        _assert_message_mentions(excinfo, "finite")

    @pytest.mark.parametrize("xlow, xhigh", [(1.0, 1.0), (2.0, 1.0)])
    def test_non_increasing_range_rejected(self, xlow, xhigh):
        with pytest.raises(ValueError) as excinfo:
            Histogram1D(10, xlow, xhigh)
        _assert_message_mentions(excinfo, "xlow", "xhigh")

    def test_single_bin_still_constructs(self):
        h = Histogram1D(1, -1.0, 1.0)
        assert h.nbins == 1
        assert h.bin_width == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Finding 6 — TabulatedPDF validation
# ---------------------------------------------------------------------------


class TestTabulatedPDFValidation:
    def test_nan_coordinate_rejected(self):
        # A NaN coordinate passes the strictly-increasing comparison
        # (comparisons with NaN are false), so it must fail the finiteness
        # check instead.
        with pytest.raises(ValueError) as excinfo:
            TabulatedPDF([0.0, float("nan"), 1.0], [1.0, 1.0, 1.0])
        _assert_message_mentions(excinfo, "coordinate", "finite")

    def test_inf_coordinate_rejected(self):
        with pytest.raises(ValueError) as excinfo:
            TabulatedPDF([0.0, float("inf"), 1.0], [1.0, 1.0, 1.0])
        _assert_message_mentions(excinfo, "coordinate", "finite")

    def test_wrong_values_length_rejected(self):
        with pytest.raises(ValueError) as excinfo:
            TabulatedPDF([0.0, 1.0, 2.0], [1.0, 1.0, 1.0, 1.0])
        _assert_message_mentions(excinfo, "values")

    @pytest.mark.parametrize("values", [[1.0, -0.5, 1.0], [1.0, float("nan"), 1.0]])
    def test_bad_density_values_rejected(self, values):
        with pytest.raises(ValueError):
            TabulatedPDF([0.0, 1.0, 2.0], values)

    def test_zero_integral_rejected(self):
        with pytest.raises(ValueError):
            TabulatedPDF([0.0, 1.0], [0.0, 0.0])


# ---------------------------------------------------------------------------
# Finding 6 — BreitWigner validation (zero width stays legal)
# ---------------------------------------------------------------------------

_VALID_BW = (1.07, 1.232, 0.12, 1.0, 2.0, 2.5)


def _bw_with(**overrides):
    params = dict(zip(
        ("schwel", "rmas", "brei", "pbrei", "dumplow", "dumphigh"), _VALID_BW
    ))
    params.update(overrides)
    return params


class TestBreitWignerValidation:
    @pytest.mark.parametrize("overrides, fragments", [
        ({"rmas": 1.07}, ("rmas", "schwel")),      # equal threshold/pole mass
        ({"rmas": 0.9}, ("rmas", "schwel")),       # pole below threshold
        ({"dumphigh": 1.07}, ("dumphigh", "schwel")),
        ({"dumplow": 2.5, "dumphigh": 2.5}, ("dumplow", "dumphigh")),
        ({"dumplow": 2.6}, ("dumplow", "dumphigh")),
        ({"brei": -0.1}, ("brei",)),
    ])
    def test_invalid_constructor_domains_rejected(self, overrides, fragments):
        with pytest.raises(ValueError) as excinfo:
            BreitWigner(**_bw_with(**overrides))
        _assert_message_mentions(excinfo, *fragments)

    @pytest.mark.parametrize("index, name", enumerate(
        ("schwel", "rmas", "brei", "pbrei", "dumplow", "dumphigh")
    ))
    def test_non_finite_parameters_rejected_by_name(self, index, name):
        kwargs = _bw_with()
        kwargs[name] = float("nan")
        with pytest.raises(ValueError) as excinfo:
            BreitWigner(**kwargs)
        _assert_message_mentions(excinfo, name)
        kwargs[name] = float("inf")
        with pytest.raises(ValueError) as excinfo:
            BreitWigner(**kwargs)
        _assert_message_mentions(excinfo, name)

    def test_set_parameters_revalidates_and_leaves_state_unchanged(self):
        bw = BreitWigner(*_VALID_BW)
        before = list(bw.params)
        with pytest.raises(ValueError) as excinfo:
            # rmas equal to schwel: rejected.
            bw.set_parameters(1.07, 1.07, 0.12, 1.0, 2.0, 2.5)
        _assert_message_mentions(excinfo, "rmas")
        assert bw.params == before

    def test_set_parameter_revalidates_atomically(self):
        bw = BreitWigner(*_VALID_BW)
        before = list(bw.params)
        # Moving the pole under the threshold must be refused as a whole.
        with pytest.raises(ValueError) as excinfo:
            bw.set_parameter(1, 1.07)
        _assert_message_mentions(excinfo, "rmas")
        assert bw.params == before
        # The sampler still works after the rejected update.
        m = bw.get_random()
        assert 1.07 <= m <= 2.5


class TestZeroWidthStillLegal:
    AUDIT_PARAMS = (1.0, 1.2, 0.0, 1.0, 1.7, 2.0)

    def test_constructor_accepts_zero_width_delta_at_rmas(self):
        for seed in range(3):
            bw = BreitWigner(*self.AUDIT_PARAMS,
                             rng=np.random.default_rng(seed))
            assert bw.get_random() == pytest.approx(1.2, abs=1e-12)
            draws = bw.get_random_n(64)
            np.testing.assert_allclose(draws, 1.2, atol=1e-12)

    def test_setter_can_shrink_width_to_zero(self):
        bw = BreitWigner(1.0, 1.4, 0.15, 1.0, 1.7, 2.0,
                         rng=np.random.default_rng(7))
        bw.set_parameter(2, 0.0)
        assert bw.get_random() == pytest.approx(1.4, abs=1e-12)
        np.testing.assert_allclose(bw.get_random_n(32), 1.4, atol=1e-12)


# ---------------------------------------------------------------------------
# Finding 6 — Generator setter domain checks
# ---------------------------------------------------------------------------


class TestGeneratorSetterDomain:
    def _generator(self):
        return Generator(reaction=Reaction(158.0))

    @pytest.mark.parametrize("pt_type, pt_param", [
        (2, 0.0),
        (2, -1.5),
        (1, 0.0),
        (1, -0.5),
        (2, float("nan")),
    ])
    def test_degenerate_pt_laws_rejected(self, pt_type, pt_param):
        gen = self._generator()
        with pytest.raises(ValueError) as excinfo:
            gen.set_pt(pt_type, pt_param)
        _assert_message_mentions(excinfo, "pt_param")
        # Rejected call leaves the configuration untouched.
        assert gen._pt_type == 0
        assert gen._pt_param == 0.0
        assert gen._pt_hist is None

    def test_valid_pt_law_still_configures(self):
        gen = self._generator()
        gen.set_pt(2, 0.8)
        assert gen._pt_type == 2
        assert gen._pt_param == pytest.approx(0.8)
        assert gen._pt_hist is not None


# ---------------------------------------------------------------------------
# Finding 6 interaction — mass/width overrides stay atomic under validation
# ---------------------------------------------------------------------------

_BASE_BW = (1.07, 1.9, 0.3, 1.0, 2.0, 2.5)


def _bw_generator(rng_seed=5):
    gen = Generator(reaction=Reaction(158.0))
    gen._id = 0
    gen._mass_type = 3
    gen._bw = BreitWigner(*_BASE_BW, rng=np.random.default_rng(rng_seed))
    return gen


class TestMassWidthOverrideUnderValidation:
    def test_override_above_damping_window_commits_atomically(self):
        gen = _bw_generator()
        _patch_generator_mass_width(gen, {0: (2.6, 0.15)})
        p = gen._bw.params
        # One-at-a-time updates would pass through dumplow > dumphigh here;
        # the atomic commit must land the exact final state instead.
        assert p[0] == pytest.approx(1.07)
        assert p[1] == pytest.approx(2.6)
        assert p[2] == pytest.approx(0.15)
        span = max(2.5 - 2.0, 0.3)
        assert p[4] == pytest.approx(2.6 + 0.2 * span)
        assert p[5] == pytest.approx(2.6 + span)
        masses = gen._bw.get_random_n(2000)
        assert np.all(masses >= p[0]) and np.all(masses <= p[5])

    def test_override_below_threshold_lowers_support_edge(self):
        gen = _bw_generator()
        _patch_generator_mass_width(gen, {0: (0.9, 0.05)})
        p = gen._bw.params
        assert p[0] == pytest.approx(0.45)
        assert p[1] == pytest.approx(0.9)
        assert p[2] == pytest.approx(0.05)
        assert p[4] == pytest.approx(2.0)
        assert p[5] == pytest.approx(2.5)
        assert len(gen._bw.get_random_n(500)) == 500

    def test_width_only_override_patches_width_alone(self):
        gen = _bw_generator()
        _patch_generator_mass_width(gen, {0: (float("nan"), 0.2)})
        p = gen._bw.params
        assert p[:2] == pytest.approx(list(_BASE_BW[:2]))
        assert p[2] == pytest.approx(0.2)
        assert p[3:] == pytest.approx(list(_BASE_BW[3:]))

    def test_zero_width_override_yields_delta_at_new_mass(self):
        gen = _bw_generator()
        _patch_generator_mass_width(gen, {0: (2.6, 0.0)})
        assert gen._bw.get_random() == pytest.approx(2.6, abs=1e-12)
        np.testing.assert_allclose(gen._bw.get_random_n(32), 2.6, atol=1e-12)
