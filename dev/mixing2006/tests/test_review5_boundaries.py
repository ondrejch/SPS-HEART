"""Consolidated v5-review regression contract (authored post-commit).

Every test here reproduces one cited defect from ``.collab/
na49_code_review5.md`` end-to-end, exactly as written in the review,
against the committed fix commits:

- 58ee355 fix(glauber): normalize NucleusSpec scalars, validate
  point-target sigma, keep grid-budget refusals on overflowing plans;
- c86ad99 fix(fits): reject unconstrained MultiFitter fits; clear stale
  scales on every fit() entry;
- 16d99b0 fix(histogram,data_io): enforce finite-storage invariant across
  histogram arithmetic and persistence; reject malformed HDF5 schemas and
  duplicate display names.

Purpose (review testing-gap item 5): a single module future release
archives carry as the explicit v5-fix contract. The wide parametrized
sweeps live in their phase homes (tests/test_glauber.py,
tests/test_fits_behavior.py, tests/test_histogram_nonfinite_policy.py,
tests/test_data_io_schema.py); this file pins ONLY the reviewed
reproducers verbatim. Assertions match stable message substrings so
wording refinements cannot break the archive contract unintentionally.

Historical behavior notes quote what each reproducer did before the fix.
"""

from __future__ import annotations

import numpy as np
import pytest

from na49_mixing.data_io import (
    load_histograms_hdf5,
    save_histograms,
    save_histograms_hdf5,
)
from na49_mixing.fits import MultiFitter
from na49_mixing.glauber import NucleusSpec, WoundedNucleonSampler
from na49_mixing.histogram import Histogram1D


@pytest.fixture
def require_h5py():
    """Skip only the calling HDF5 test when the optional backend is absent."""
    pytest.importorskip("h5py")


_SENTINEL = b"v5-contract-pre-existing-bytes\x00\x01must-survive"


def _pb_like_spec() -> NucleusSpec:
    return NucleusSpec(
        name="Pb208",
        A=208,
        Z=82,
        profile="woods_saxon",
        radius_fm=6.62,
        diffuseness_fm=0.546,
        sigma_nn_inel_mb=31.7,
    )


# ---------------------------------------------------------------------------
# Finding 1 — NucleusSpec accepts inconsistent scalars (glauber.py)
# ---------------------------------------------------------------------------

REVIEW_FINDING1_EXAMPLES = [
    pytest.param(
        {"A": 208.5, "Z": 82},
        "A must be an integer",
        id="fractional-A",
    ),
    pytest.param(
        {"A": 12, "Z": 6.8},
        "Z must be an integer",
        id="fractional-Z",
    ),
    pytest.param(
        {"A": True, "Z": True},
        "A must be an integer",
        id="boolean-A-Z",
    ),
    pytest.param(
        {"A": "208", "Z": "82"},
        "A must be an integer",
        id="string-A-Z",
    ),
    pytest.param(
        {"A": np.float64(208.5), "Z": 82},
        "A must be an integer",
        id="numpy-fractional-A",
    ),
    pytest.param(
        {"A": np.bool_(True), "Z": 82},
        "A must be an integer",
        id="numpy-boolean-A",
    ),
]


@pytest.mark.parametrize(("kwargs", "fragment"), REVIEW_FINDING1_EXAMPLES)
def test_finding1_spec_examples_raise_valueerror_naming_the_parameter(
        kwargs, fragment):
    """The four literal examples plus NumPy analogues now refuse."""
    with pytest.raises(ValueError) as excinfo:
        NucleusSpec(**kwargs)
    msg = str(excinfo.value)
    assert fragment in msg
    assert "NucleusSpec" in msg


def test_finding1_numeric_strings_normalize_instead_of_reaching_sampler():
    """Numeric strings used to survive raw into the sampler and die there
    with an unrelated TypeError; they now parse at construction."""
    spec = NucleusSpec(
        **{"name": "Pb208", "A": 208, "Z": 82, "profile": "woods_saxon",
           "radius_fm": "6.62", "diffuseness_fm": "0.546",
           "sigma_nn_inel_mb": "31.7"}
    )
    for field, expected in (
        ("radius_fm", 6.62),
        ("diffuseness_fm", 0.546),
        ("sigma_nn_inel_mb", 31.7),
    ):
        value = getattr(spec, field)
        assert type(value) is float, field
        assert value == expected, field
    sampler = WoundedNucleonSampler(spec, rng=np.random.default_rng(0))
    reference = WoundedNucleonSampler(
        _pb_like_spec(), rng=np.random.default_rng(0)
    )
    assert sampler._mu.tobytes() == reference._mu.tobytes()
    assert sampler.sigma_pA_mb() == reference.sigma_pA_mb()


@pytest.mark.parametrize(
    "sigma", [0.0, -5.0, float("nan"), float("inf")],
    ids=["zero", "negative", "nan", "inf"],
)
def test_finding1_point_target_cross_sections_rejected_before_sampler(sigma):
    """Point targets accepted invalid sigmas; sigma_pA_mb handed them back.

    The A == 1 early-out now sits behind the sigma check."""
    with pytest.raises(ValueError) as excinfo:
        NucleusSpec(name="p", A=1, Z=1, sigma_nn_inel_mb=sigma)
    msg = str(excinfo.value)
    assert "sigma_nn_inel_mb" in msg
    assert "finite and positive" in msg


def test_finding1_valid_point_target_still_returns_its_sigma():
    spec = NucleusSpec(name="p", A=1, Z=1, sigma_nn_inel_mb=31.7)
    sampler = WoundedNucleonSampler(spec, rng=np.random.default_rng(0))
    assert sampler.sigma_pA_mb() == pytest.approx(31.7)


# ---------------------------------------------------------------------------
# Finding 2 — MultiFitter unconstrained fits returned array([1.]) (fits.py)
# ---------------------------------------------------------------------------


def test_finding2_zero_template_reproducer_no_longer_returns_initial_values():
    """Exact review listing:

        data = Histogram1D(10, 0.0, 1.0)
        mc = Histogram1D(10, 0.0, 1.0)
        data.fill(0.5, 5.0)

        MultiFitter(data, [mc], fit_range=(0.1, 0.9)).fit()
        # array([1.]) although the scale is completely unconstrained

    The fit is rejected before any optimizer work instead; a regression
    would either return a non-None result or raise a different error.
    """
    data = Histogram1D(10, 0.0, 1.0)
    mc = Histogram1D(10, 0.0, 1.0)
    data.fill(0.5, 5.0)
    mf = MultiFitter(data, [mc], fit_range=(0.1, 0.9))
    with pytest.raises(ValueError) as excinfo:
        mf.fit()
    msg = str(excinfo.value)
    assert "no MC component contributes information" in msg
    assert "zero" in msg                    # templates identically zero
    assert "unconstrained" in msg           # the arbitrary-scale verdict
    # The historical outcome survives nowhere: no scale was ever stored
    # or returned through the property either.
    assert mf.scales is None


def test_finding2_both_empty_case_raises_without_optimizer():
    """Review wording: 'the same arbitrary result occurs when both data
    and templates are empty'."""
    data = Histogram1D(10, 0.0, 1.0)
    mc = Histogram1D(10, 0.0, 1.0)
    mf = MultiFitter(data, [mc], fit_range=(0.03, 0.98))
    with pytest.raises(ValueError) as excinfo:
        mf.fit()
    assert "no MC component contributes information" in \
        str(excinfo.value)
    assert mf.scales is None


# ---------------------------------------------------------------------------
# Finding 3 — failed second fit() left prior scales visible (fits.py)
# ---------------------------------------------------------------------------

_SHAPE_A = np.array(
    [1.5, 0.8, 2.2, 1.1, 0.6, 1.9, 1.3, 0.4, 2.5, 0.9]
)
_SHAPE_B = _SHAPE_A[::-1].copy() * 1.3 + 0.25


def _filled_hist(weights: np.ndarray, name: str) -> Histogram1D:
    h = Histogram1D(10, 0.0, 1.0, name=name)
    centers = np.arange(10) * 0.1 + 0.05
    h.fill_array(centers, np.asarray(weights, dtype=float))
    return h


def test_finding3_failed_refit_clears_the_first_fit_result():
    """Review reproducer: successful fit, then the externally owned MC
    list mutated to empty. The refit raised correctly even historically,
    but mf.scales kept serving the first fit's values afterwards."""
    data = _filled_hist(2.0 * _SHAPE_A + 0.5 * _SHAPE_B, "data")
    mcs = [_filled_hist(_SHAPE_A, "mc_A"),
           _filled_hist(_SHAPE_B, "mc_B")]
    mf = MultiFitter(data, mcs, fit_range=(0.03, 0.98))
    first = mf.fit()
    assert first is not None and mf.scales is not None
    np.testing.assert_allclose(first, [2.0, 0.5], rtol=1e-3, atol=1e-4)
    mf._mc.clear()                          # external mutation per review
    with pytest.raises(ValueError) as excinfo:
        mf.fit()
    assert "requires at least one MC component" in str(excinfo.value)
    assert mf.scales is None                # previously still [~2, ~0.5]


# ---------------------------------------------------------------------------
# Finding 4 — arithmetic violated the finite-storage invariant (histogram)
# ---------------------------------------------------------------------------


def test_finding4_scale_nan_on_zero_bins_can_no_longer_produce_nan_storage():
    """Review sentence: 'scale(float(\"nan\")) turns even zero bins into
    NaN.' The call now refuses and nothing exists to taint."""
    h = Histogram1D(4, 0.0, 1.0)
    snapshot = h._contents.tobytes()
    with pytest.raises(ValueError) as excinfo:
        h.scale(float("nan"))
    assert "scale:" in str(excinfo.value)
    assert np.all(h._contents == 0.0)
    assert h._contents.tobytes() == snapshot


def test_finding4_add_with_inf_scale_rejected_before_zero_times_inf():
    """Review sentence: 'add(other, scale=float(\"inf\")) produces NaN
    from 0 * inf and may produce infinity elsewhere.'"""
    h = Histogram1D(4, 0.0, 1.0, name="left")
    h.fill(0.35, weight=2.0)
    other = Histogram1D(4, 0.0, 1.0, name="right")
    other.fill(0.55, weight=-1.5)
    snapshot = h._contents.tobytes()
    with pytest.raises(ValueError) as excinfo:
        h.add(other, scale=float("inf"))
    assert "add:" in str(excinfo.value)
    assert h._contents.tobytes() == snapshot


def test_finding4_repeated_finite_weight_fills_overflow_atomically():
    """Repeated finite deposits may overflow a bin 'from large but
    individually finite weights' (review finding 4); the second deposit
    rejects with the histogram holding exactly the first's state."""
    h = Histogram1D(4, 0.0, 1.0)
    h.fill(0.5, weight=1e308)               # individually finite
    snapshot = h._contents.tobytes()
    with pytest.raises(ValueError):
        h.fill(0.5, weight=1e308)
    assert h.get_bin_content(3) == 1e308
    assert h._contents.tobytes() == snapshot


@pytest.mark.parametrize("backend", ["npz", "hdf5"])
def test_finding4_save_refuses_nonfinite_payload_and_spares_destination(
        tmp_path, require_h5py, backend):
    """Persistence leg: 'Both save functions currently write these invalid
    arrays successfully' — save-time validation now runs BEFORE the
    destination opens, so pre-existing bytes survive untouched."""
    ext = "npz" if backend == "npz" else "h5"
    path = tmp_path / f"f4_{backend}.{ext}"
    path.write_bytes(_SENTINEL)
    h = Histogram1D(4, 0.0, 1.0, name="victim")
    h.fill_array(np.linspace(0.05, 0.95, 4))
    h.set_bin_content(2, float("inf"))      # escape hatch drives the boundary
    saver = save_histograms if backend == "npz" else save_histograms_hdf5
    with pytest.raises(ValueError) as excinfo:
        saver(str(path), {"victim": h})
    msg = str(excinfo.value)
    assert "non-finite value(s)" in msg
    assert "'victim'" in msg
    assert str(path) in msg
    assert path.read_bytes() == _SENTINEL


# ---------------------------------------------------------------------------
# Finding 5 — HDF5 loader lost entries with duplicate display names (io)
# ---------------------------------------------------------------------------


def _craft_duplicate_named_groups(path, gname_a, gname_b):
    import h5py
    with h5py.File(str(path), "w") as handle:
        for gname in (gname_a, gname_b):
            grp = handle.create_group(gname)
            grp.attrs.create("name", "shared-display")
            grp.attrs["nbins"] = 2
            grp.attrs["xlow"] = 0.0
            grp.attrs["xhigh"] = 1.0
            grp.create_dataset("contents", data=np.array([1.0, 0.0, 0.0, 0.0]))


def test_finding5_duplicate_display_names_raise_instead_of_silently_winning(
        tmp_path, require_h5py):
    """Review repro: two groups declaring the same 'name' attribute made
    the later group silently replace the earlier histogram. Loading now
    raises naming path, duplicate display name, and both group names."""
    path = tmp_path / "dupnames.h5"
    _craft_duplicate_named_groups(path, "first_group", "second_group")
    with pytest.raises(ValueError) as excinfo:
        load_histograms_hdf5(str(path))
    msg = str(excinfo.value)
    assert "duplicate display name 'shared-display'" in msg
    assert "'first_group'" in msg and "'second_group'" in msg
    assert str(path) in msg


def test_finding5_distinct_display_names_load_normally(tmp_path, require_h5py):
    # Control: the hardening leaves ordinary files byte-compatible.
    import h5py
    path = tmp_path / "ok.h5"
    with h5py.File(str(path), "w") as handle:
        grp = handle.create_group("g")
        grp.attrs["name"] = "fine"
        grp.attrs["nbins"] = 2
        grp.attrs["xlow"] = 0.0
        grp.attrs["xhigh"] = 1.0
        grp.create_dataset("contents", data=np.zeros(4))
    loaded = load_histograms_hdf5(str(path))
    assert set(loaded) == {"fine"}
    assert loaded["fine"].name == "fine"


# ---------------------------------------------------------------------------
# Finding 6 — planner overflowed before its budget error (glauber.py)
# ---------------------------------------------------------------------------


def test_finding6_subnormal_dr_repro_raises_budget_error_not_overflowerror():
    """Exact review listing:

        WoundedNucleonSampler(spec, rng, dr_fm=5e-324)
        # OverflowError: cannot convert float infinity to integer

    Construction now refuses inside the documented budget ValueError."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError) as excinfo:
        WoundedNucleonSampler(_pb_like_spec(), rng, dr_fm=5e-324)
    # OverflowError is a distinct class from ValueError, so an exact type
    # match rules the historical failure out.
    assert excinfo.type is ValueError
    assert "budget exceeded" in str(excinfo.value)
    assert "coarsen dr_fm" in str(excinfo.value)


def test_finding6_overflowing_derived_extents_keep_budget_contract():
    """Review sentence: 'Very large but finite radius/diffuseness values
    can similarly overflow the derived endpoints.'"""
    spec_kwargs = {
        "name": "HugeOscillator",
        "A": 97,
        "Z": 48,
        "profile": "harmonic_oscillator",
        "radius_fm": 3e307,     # r_max = 8R overflows double precision
        "sigma_nn_inel_mb": 31.7,
    }
    spec = NucleusSpec(**spec_kwargs)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError) as excinfo:
        WoundedNucleonSampler(spec, rng)
    msg = str(excinfo.value)
    assert excinfo.type is ValueError
    assert "grid-cell budget exceeded" in msg
    assert "HugeOscillator" in msg
