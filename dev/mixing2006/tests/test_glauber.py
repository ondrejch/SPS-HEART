"""Tests for na49_mixing.glauber (p+A wounded-nucleon geometry)."""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from na49_mixing.glauber import (
    MB_PER_FM2,
    NucleusSpec,
    WoundedNucleonSampler,
    _arange_cell_count,
    _plan_grids,
    _GRID_SPAN_OVERFLOW_SENTINEL,
    _MAX_GRID_CELLS_PER_AXIS,
)


def _c12_spec() -> NucleusSpec:
    return NucleusSpec(
        name="C12",
        A=12,
        Z=6,
        profile="harmonic_oscillator",
        # Pure-oscillator length reproducing the measured point-nucleon
        # rms radius (~2.3 fm): a = rms / sqrt(3/2).
        radius_fm=1.88,
        sigma_nn_inel_mb=31.7,
    )


def _pb208_spec() -> NucleusSpec:
    return NucleusSpec(
        name="Pb208",
        A=208,
        Z=82,
        profile="woods_saxon",
        radius_fm=6.62,
        diffuseness_fm=0.546,
        sigma_nn_inel_mb=31.7,
    )


def _conditional_pmf(mu: float, k: int) -> float:
    return math.exp(-mu) * mu**k / math.factorial(k) / (-math.expm1(-mu))


def test_nucleus_spec_validation() -> None:
    with pytest.raises(ValueError, match="A >= 1"):
        NucleusSpec(A=0, Z=0)
    with pytest.raises(ValueError, match="1 <= Z <= A"):
        NucleusSpec(A=12, Z=13)
    with pytest.raises(ValueError, match="profile"):
        NucleusSpec(A=12, Z=6, profile="gaussian")
    with pytest.raises(ValueError, match="sigma_nn_inel_mb"):
        NucleusSpec(A=12, Z=6, profile="harmonic_oscillator", radius_fm=1.07,
                    sigma_nn_inel_mb=-1.0)
    with pytest.raises(ValueError, match="radius_fm"):
        NucleusSpec(A=208, Z=82, radius_fm=0.0)
    # A == 1 forces the point branch and skips density validation.
    point = NucleusSpec(A=1, Z=1)
    assert point.resolved_profile() == "point"


def test_point_target_constants_and_no_rng_consumption() -> None:
    spec = NucleusSpec(name="p1", A=1, Z=1, sigma_nn_inel_mb=31.7)
    rng = np.random.default_rng(7)
    state_before = rng.bit_generator.state
    sampler = WoundedNucleonSampler(spec, rng=rng)

    assert sampler.sigma_pA_mb() == pytest.approx(31.7)
    assert sampler.mean_target_wounds_minbias() == pytest.approx(1.0)
    assert sampler.fraction_for_wound_range(1, 10**6) == pytest.approx(1.0)
    assert sampler.fraction_for_wound_range(2, 5) == 0.0

    b, k = sampler.sample_event()
    assert (b, k) == (0.0, 1)
    assert rng.bit_generator.state == state_before


def test_thickness_scale_is_physical() -> None:
    """Central thickness of order 1 nucleons/fm^2 for C and Pb."""
    c = WoundedNucleonSampler(_c12_spec(), rng=np.random.default_rng(1))
    pb = WoundedNucleonSampler(_pb208_spec(), rng=np.random.default_rng(2))
    assert float(np.max(c._mu)) / (31.7 / MB_PER_FM2) == pytest.approx(
        0.9, abs=0.6
    )
    t_pb0 = float(pb._mu[0]) / (31.7 / MB_PER_FM2)
    assert 1.2 < t_pb0 < 3.0


def test_wound_pmf_matches_conditional_poisson() -> None:
    rng = np.random.default_rng(20260825)
    sampler = WoundedNucleonSampler(_pb208_spec(), rng=rng)

    mu_mid = float(np.max(sampler._mu))
    n_draw = 40_000
    counts: dict[int, int] = {}
    for _ in range(n_draw):
        k = sampler.sample_conditional_wounds(mu_mid)
        counts[k] = counts.get(k, 0) + 1
    for k in range(1, 8):
        empirical = counts.get(k, 0) / n_draw
        analytic = _conditional_pmf(mu_mid, k)
        assert empirical == pytest.approx(analytic, abs=0.01)


def test_minbias_moments_and_cross_sections_ordered() -> None:
    s_c = WoundedNucleonSampler(_c12_spec(), rng=np.random.default_rng(11))
    s_pb = WoundedNucleonSampler(_pb208_spec(), rng=np.random.default_rng(12))

    # Cross sections ordered like the physical systems.
    assert s_pb.sigma_pA_mb() > s_c.sigma_pA_mb() > 31.7
    # pPb inelastic is of order a barn.
    assert 1500.0 < s_pb.sigma_pA_mb() < 2500.0

    # Analytic min-bias mean vs empirical mean over many events.
    for sampler in (s_c, s_pb):
        analytic = sampler.mean_target_wounds_minbias()
        draws = [sampler.sample_event()[1] for _ in range(20_000)]
        empirical = float(np.mean(draws))
        assert empirical == pytest.approx(analytic, abs=0.05)
        assert all(k >= 1 for k in draws)


def test_centrality_fractions_sum_to_one_and_slice_means_monotone() -> None:
    sampler = WoundedNucleonSampler(_pb208_spec(), rng=np.random.default_rng(3))

    ranges = [(1, 1), (2, 2), (3, 4), (5, 10 ** 6)]
    fractions = [sampler.fraction_for_wound_range(lo, hi) for lo, hi in ranges]
    assert sum(fractions) == pytest.approx(1.0, abs=1e-6)
    assert all(f > 0.0 for f in fractions)

    # The most central slice carries wounds near its lower bound by
    # construction; verify against an empirical restricted draw.
    lo, hi = ranges[2]
    draws = []
    for _ in range(4_000):
        while True:
            _, k = sampler.sample_event()
            if lo <= k <= hi:
                break
        draws.append(k)
    assert float(np.mean(draws)) >= lo


def test_fraction_respects_truncated_support() -> None:
    """Slice fractions follow the truncated-at-A pmf of the sampler."""

    c12 = WoundedNucleonSampler(_c12_spec(), rng=np.random.default_rng(5))
    # lo > A is structurally impossible: exactly zero.
    assert c12.fraction_for_wound_range(13, 13) == 0.0
    assert c12.fraction_for_wound_range(13, 10**6) == 0.0

    # Small support where the K >= A pile-up carries real mass: a
    # compact A=8 oscillator reaches mu > A at center.
    small_a = NucleusSpec(
        name="A8compact",
        A=8,
        Z=4,
        profile="harmonic_oscillator",
        radius_fm=1.0,
        sigma_nn_inel_mb=31.7,
    )
    sampler = WoundedNucleonSampler(small_a, rng=np.random.default_rng(9))
    piled = sampler.fraction_for_wound_range(8, 10**6)
    assert piled > 0.05
    total = sum(
        sampler.fraction_for_wound_range(lo, hi)
        for lo, hi in [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 10**6)]
    )
    assert total == pytest.approx(1.0, abs=1e-6)

    # For Pb208 the K >= 208 pile-up bin underflows double precision;
    # disjoint slices over the measurable support still sum to one
    # within the same tolerance once that bin contributes ~0.
    pb = WoundedNucleonSampler(_pb208_spec(), rng=np.random.default_rng(6))
    assert pb.fraction_for_wound_range(1, 10**6) == pytest.approx(1.0)


def test_minbias_stream_unchanged_by_conditioned_sampling() -> None:
    """Min-bias (b, K) sequences for fixed seeds are frozen.

    These literals were captured from ``sample_event`` before the
    range-conditioned draw existed; ``sample_event_in_range`` must not
    perturb the minimum-bias path.
    """

    expected = {
        20260825: [
            (6.44, 2),
            (4.8, 4),
            (1.4000000000000001, 4),
            (6.54, 4),
            (7.22, 1),
            (7.9, 1),
            (5.1000000000000005, 4),
            (5.84, 2),
        ],
        7: [
            (5.82, 5),
            (6.5600000000000005, 1),
            (4.0200000000000005, 8),
            (0.54, 9),
            (6.66, 2),
            (4.04, 4),
            (3.7, 5),
            (5.22, 4),
        ],
    }
    for seed, pairs in expected.items():
        sampler = WoundedNucleonSampler(_pb208_spec(), rng=np.random.default_rng(seed))
        drawn = [sampler.sample_event() for _ in range(len(pairs))]
        assert drawn == pairs


def test_conditioned_draw_matches_interaction_weighted_slice() -> None:
    """Conditioned b and K marginals follow the restricted measure.

    The K shares inside a slice must reproduce ratios of
    ``fraction_for_wound_range`` (closed form vs histogram, now for a
    window too narrow for min-bias rejection), and the b marginal must
    track the interaction-weighted slice weight on the sampler grid.
    """

    sampler = WoundedNucleonSampler(_pb208_spec(), rng=np.random.default_rng(99))
    analytic = WoundedNucleonSampler(_pb208_spec(), rng=np.random.default_rng(0))

    n = 60_000
    ks = np.asarray([sampler.sample_event_in_range(18, 28)[1] for _ in range(n)])
    assert np.all((ks >= 18) & (ks <= 28))
    share21 = float(np.count_nonzero(ks == 21) / n)
    closed = analytic.fraction_for_wound_range(21, 21) / analytic.fraction_for_wound_range(
        18, 28
    )
    assert share21 == pytest.approx(closed, abs=0.005)

    marginal = analytic._slice_marginal_weights(24, 24)
    b_mean_closed = float(np.sum(marginal * analytic._b_grid) / np.sum(marginal))
    bs = np.asarray([sampler.sample_event_in_range(24, 24)[0] for _ in range(n)])
    assert float(np.mean(bs)) == pytest.approx(b_mean_closed, abs=0.05)

    # Rare single-bin slices populate directly and stay put.
    for lo in (24, 25):
        rare = np.asarray([sampler.sample_event_in_range(lo, lo)[1] for _ in range(20_000)])
        assert np.all(rare == lo)
        assert np.all((rare >= 1) & (rare <= analytic.spec.A))


def test_conditioned_draw_honors_truncated_tail_pileup() -> None:
    """A slice reaching K = A carries the pile-up share of the pmf tail."""

    small_a = NucleusSpec(
        name="A8compact",
        A=8,
        Z=4,
        profile="harmonic_oscillator",
        radius_fm=1.0,
        sigma_nn_inel_mb=31.7,
    )
    analytic = WoundedNucleonSampler(small_a, rng=np.random.default_rng(0))
    sampler = WoundedNucleonSampler(small_a, rng=np.random.default_rng(123))
    ks = np.asarray(
        [sampler.sample_event_in_range(6, 10**6)[1] for _ in range(40_000)]
    )
    assert np.all(ks >= 6)
    empirical_cap = float(np.count_nonzero(ks == 8) / ks.size)
    closed_cap = analytic.fraction_for_wound_range(8, 10**6) / analytic.fraction_for_wound_range(
        6, 10**6
    )
    assert empirical_cap == pytest.approx(closed_cap, abs=0.02)


def test_conditioned_draw_consumes_exactly_two_uniforms_per_event() -> None:
    rng = np.random.default_rng(11)
    sampler = WoundedNucleonSampler(_pb208_spec(), rng=rng)
    n = 50
    for _ in range(n):
        sampler.sample_event_in_range(2, 6)
    replay = np.random.default_rng(11)
    for _ in range(2 * n):
        replay.random()
    assert rng.bit_generator.state == replay.bit_generator.state


def test_sample_event_in_range_refuses_empty_slices(monkeypatch) -> None:
    """Structurally empty slices raise instead of drawing or looping."""

    c12 = WoundedNucleonSampler(_c12_spec(), rng=np.random.default_rng(1))
    with pytest.raises(ValueError, match="K <= A"):
        c12.sample_event_in_range(13, 13)
    # A slice inside the truncated support whose interaction-weighted
    # mass underflows to zero has nothing to condition on.
    monkeypatch.setattr(
        WoundedNucleonSampler,
        "_slice_marginal_weights",
        lambda self, lo, hi: np.zeros_like(self._mu),
    )
    with pytest.raises(ValueError, match="no interaction"):
        c12.sample_event_in_range(3, 5)


def test_point_target_conditioned_draw_costs_nothing() -> None:
    """A = 1: K = 1 inside its range, no RNG consumption, empty refused."""

    point_spec = NucleusSpec(name="p1", A=1, Z=1, sigma_nn_inel_mb=31.7)
    rng = np.random.default_rng(5)
    point = WoundedNucleonSampler(point_spec, rng=rng)
    state_before = rng.bit_generator.state
    assert point.sample_event_in_range(1, 1) == (0.0, 1)
    assert rng.bit_generator.state == state_before
    with pytest.raises(ValueError, match="K = 1"):
        point.sample_event_in_range(2, 3)


def test_sigma_units_conversion_constant() -> None:
    assert MB_PER_FM2 == 10.0


# ---------------------------------------------------------------------------
# NucleusSpec scalar normalization and point-target sigma (review5 #1)
# ---------------------------------------------------------------------------

def _loose_spec_kwargs(**overrides: object) -> dict[str, Any]:
    """Spec keyword bundle whose values skip declared-type narrowing.

    The normalization matrix deliberately feeds NumPy scalars and
    numeric strings into ``NucleusSpec``; returning ``Any``-valued
    kwargs lets the call sites express that intent to the type checker
    instead of fighting the dataclass annotations.
    """
    return dict(overrides)


NON_INTEGER_VALUES = [
    pytest.param(208.5, id="fractional-float"),
    pytest.param(np.float64(208.5), id="fractional-numpy"),
    pytest.param(True, id="bool"),
    pytest.param(np.bool_(True), id="numpy-bool"),
    pytest.param("208", id="string"),
]


@pytest.mark.parametrize("bad", NON_INTEGER_VALUES)
def test_nucleus_spec_rejects_non_integer_A(bad) -> None:
    with pytest.raises(ValueError) as excinfo:
        NucleusSpec(A=bad, Z=82)
    msg = str(excinfo.value)
    assert "A must be an integer" in msg
    assert repr(bad) in msg


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(6.8, id="fractional-float"),
        pytest.param(np.float64(6.8), id="fractional-numpy"),
        pytest.param(False, id="bool"),
        pytest.param(np.bool_(False), id="numpy-bool"),
        pytest.param("82.0", id="string"),
    ],
)
def test_nucleus_spec_rejects_non_integer_Z(bad) -> None:
    with pytest.raises(ValueError) as excinfo:
        NucleusSpec(A=12, Z=bad)
    msg = str(excinfo.value)
    assert "Z must be an integer" in msg
    assert repr(bad) in msg


@pytest.mark.parametrize(
    ("A", "Z"),
    [(np.int64(208), np.int64(82)), (208.0, 82.0)],
)
def test_nucleus_spec_normalizes_integral_scalars_to_builtin_int(
    A, Z
) -> None:
    spec = NucleusSpec(A=A, Z=Z)
    assert type(spec.A) is int and type(spec.Z) is int
    assert spec.A == 208 and spec.Z == 82


def test_nucleus_spec_float_fields_normalized_to_builtin_float() -> None:
    """Every profile field stores a built-in float, whatever arrived.

    NumPy scalars, Python ints, and numeric strings all land as the
    same stored value; no raw string or foreign object survives.
    """
    spec = NucleusSpec(
        **_loose_spec_kwargs(
            name="C12",
            A=np.int64(12),
            Z=6.0,
            profile="harmonic_oscillator",
            radius_fm=np.float64(1.88),
            diffuseness_fm=2,
            quad_alpha="0.25",
            sigma_nn_inel_mb=np.float64(31.7),
        )
    )
    assert type(spec.A) is int and spec.A == 12
    assert type(spec.Z) is int and spec.Z == 6
    for field, expected in (
        ("radius_fm", 1.88),
        ("diffuseness_fm", 2.0),
        ("quad_alpha", 0.25),
        ("sigma_nn_inel_mb", 31.7),
    ):
        value = getattr(spec, field)
        assert type(value) is float, field
        assert value == expected, field


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("radius_fm", "six-point-two", id="radius-word"),
        pytest.param("diffuseness_fm", "0p546", id="diffuseness-malformed"),
        pytest.param("quad_alpha", "flat", id="alpha-word"),
        pytest.param("sigma_nn_inel_mb", "big", id="sigma-word"),
        pytest.param("radius_fm", True, id="radius-bool"),
    ],
)
def test_nucleus_spec_unparseable_float_fields_raise_naming_parameter(
    field: str, value
) -> None:
    with pytest.raises(ValueError) as excinfo:
        NucleusSpec(A=208, Z=82, **{field: value})
    msg = str(excinfo.value)
    assert f"{field} must be a float" in msg
    assert repr(value) in msg


def test_string_valued_geometry_matches_numeric_counterpart_bitwise() -> None:
    """String normalization is lossless end to end, grids included.

    Together with the frozen min-bias stream literals pinned above,
    this pins the no-behavior-change side of the review5 fixes.
    """
    numeric = NucleusSpec(
        name="Pb208",
        A=208,
        Z=82,
        profile="woods_saxon",
        radius_fm=6.62,
        diffuseness_fm=0.546,
        sigma_nn_inel_mb=31.7,
    )
    textual = NucleusSpec(
        **_loose_spec_kwargs(
            name="Pb208",
            A=208,
            Z=82,
            profile="woods_saxon",
            radius_fm="6.62",
            diffuseness_fm="0.546",
            sigma_nn_inel_mb="31.7",
        )
    )
    sampler_num = WoundedNucleonSampler(numeric, rng=np.random.default_rng(3))
    sampler_txt = WoundedNucleonSampler(textual, rng=np.random.default_rng(3))
    assert sampler_num._b_grid.tobytes() == sampler_txt._b_grid.tobytes()
    assert sampler_num._mu.tobytes() == sampler_txt._mu.tobytes()
    assert sampler_num._b_cdf.tobytes() == sampler_txt._b_cdf.tobytes()
    assert sampler_num.sigma_pA_mb() == sampler_txt.sigma_pA_mb()


@pytest.mark.parametrize(
    "bad", [0.0, -5.0, float("nan"), float("inf"), float("-inf")]
)
def test_point_target_rejects_invalid_sigma_nn_inel_mb(bad) -> None:
    """The A == 1 early-out no longer skips the cross-section check."""
    with pytest.raises(ValueError) as excinfo:
        NucleusSpec(name="p", A=1, Z=1, sigma_nn_inel_mb=bad)
    msg = str(excinfo.value)
    assert "sigma_nn_inel_mb" in msg
    assert "finite and positive" in msg
    assert repr(bad) in msg


def test_point_target_positive_sigma_reaches_sampler_return_value() -> None:
    spec = NucleusSpec(name="p", A=1, Z=1, sigma_nn_inel_mb=42.5)
    sampler = WoundedNucleonSampler(spec, rng=np.random.default_rng(0))
    assert sampler.sigma_pA_mb() == pytest.approx(42.5)


def test_point_target_exempts_only_unused_shape_parameters() -> None:
    """Point targets skip shape-value checks; parsing still applies."""
    spec = NucleusSpec(
        name="p",
        A=1,
        Z=1,
        profile="point",
        radius_fm=0.0,
        diffuseness_fm=-1.0,
        quad_alpha=-5.0,
        sigma_nn_inel_mb=42.5,
    )
    assert spec.resolved_profile() == "point"
    # An unparseable shape value still refuses even on the point branch
    # because storage normalization never depends on the profile.
    with pytest.raises(ValueError, match="radius_fm"):
        NucleusSpec(
            **_loose_spec_kwargs(
                name="p", A=1, Z=1, profile="point", radius_fm="junk"
            )
        )


# ---------------------------------------------------------------------------
# Planner overflow contract: non-finite spans hit the budget refusal
# ---------------------------------------------------------------------------


def test_arange_cell_count_reports_sentinel_for_overflowing_span() -> None:
    """Subnormal steps and infinite stops return an above-cap count."""
    # 1.0 / 5e-324 overflows double precision.
    n_subnormal = _arange_cell_count(0.0, 1.0, 5e-324)
    assert isinstance(n_subnormal, int)
    assert n_subnormal > _MAX_GRID_CELLS_PER_AXIS
    assert n_subnormal == _GRID_SPAN_OVERFLOW_SENTINEL
    assert _arange_cell_count(0.0, math.inf, 1.0) > _MAX_GRID_CELLS_PER_AXIS
    assert _arange_cell_count(0.0, 1.0, float("nan")) > _MAX_GRID_CELLS_PER_AXIS


def test_arange_cell_count_unchanged_for_finite_spans() -> None:
    """The helper still replicates len(np.arange) exactly when finite."""
    dr = 0.02
    plan_args = (6.62 + 10.0 * 0.546 + dr, 6.62 + 10.0 * 0.546 + dr, dr)
    assert _arange_cell_count(0.0, plan_args[0], dr) == len(
        np.arange(0.0, plan_args[0], dr)
    )
    assert _arange_cell_count(5.0, 5.0, dr) == 0
    assert _arange_cell_count(3.0, 1.0, dr) == 0


def test_plan_grids_flags_overflowing_derived_extents() -> None:
    """Finite geometry inputs whose extents overflow report sentinels.

    r_max = 1e307 and surface = 4e307 are finite, but z_half =
    r_max + 10*surface overflows double precision.
    """
    plan = _plan_grids(1e307, 4e307, 0.02)
    for count in (plan.n_r, plan.n_b, plan.n_z):
        assert count == _GRID_SPAN_OVERFLOW_SENTINEL
        assert count > _MAX_GRID_CELLS_PER_AXIS


def test_sampler_refuses_subnormal_dr_fm_with_budget_wording() -> None:
    """dr_fm = 5e-324 lands in the budget ValueError, not OverflowError."""
    with pytest.raises(ValueError) as excinfo:
        WoundedNucleonSampler(
            _pb208_spec(), rng=np.random.default_rng(0), dr_fm=5e-324
        )
    assert excinfo.type is ValueError
    msg = str(excinfo.value)
    assert "grid-cell budget exceeded" in msg
    assert "dr_fm" in msg
    assert "Pb208" in msg
    assert "coarsen dr_fm" in msg


def test_sampler_refuses_overflowing_geometry_with_budget_wording() -> None:
    """Huge-but-finite diffuseness keeps its descriptive refusal."""
    spec = NucleusSpec(
        name="Huge",
        A=208,
        Z=82,
        profile="woods_saxon",
        radius_fm=6.62,
        diffuseness_fm=1e308,
        sigma_nn_inel_mb=31.7,
    )
    with pytest.raises(ValueError) as excinfo:
        WoundedNucleonSampler(spec, rng=np.random.default_rng(0))
    assert excinfo.type is ValueError
    msg = str(excinfo.value)
    assert "grid-cell budget exceeded" in msg
    assert "Huge" in msg
    assert "coarsen dr_fm" in msg


# ---------------------------------------------------------------------------
# Review5 testing-gap sweeps (items 1-3 and 9 of the review5 task card):
# parametrized type matrices widening the focused phase regressions above
# onto committed behavior (58ee355).
# ---------------------------------------------------------------------------


# Gap item 1: A/Z rejection AND normalization across the scalar-type matrix.

BAD_AZ_SCALAR_MATRIX = [
    pytest.param(208.5, id="fractional-float"),
    pytest.param(np.float64(208.5), id="fractional-numpy-float"),
    pytest.param(True, id="bool-true"),
    pytest.param(False, id="bool-false"),
    pytest.param(np.bool_(True), id="numpy-bool"),
    pytest.param("208", id="string-int-literal"),
    pytest.param(np.str_("82"), id="numpy-string"),
    pytest.param(float("nan"), id="nan-float"),
    pytest.param(float("-inf"), id="minus-inf-float"),
]


@pytest.mark.parametrize("slot", ["A", "Z"])
@pytest.mark.parametrize("bad", BAD_AZ_SCALAR_MATRIX)
def test_az_type_matrix_rejects_every_non_integer_kind(slot, bad) -> None:
    """Fractional, boolean, string, and non-finite numerics never enter A/Z."""
    with pytest.raises(ValueError) as excinfo:
        NucleusSpec(**_loose_spec_kwargs(**{slot: bad}))
    msg = str(excinfo.value)
    assert f"NucleusSpec {slot} must be an integer" in msg
    assert repr(bad) in msg


@pytest.mark.parametrize(
    ("A", "Z"),
    [
        pytest.param(np.int64(208), np.int64(82), id="numpy-integers"),
        pytest.param(
            np.float64(208.0), np.float64(82.0),
            id="numpy-integral-floats",
        ),
        pytest.param(208, 82.0, id="mixed-int-and-integral-float"),
    ],
)
def test_az_type_matrix_normalizes_accepted_kinds_to_builtin_int(
    A, Z
) -> None:
    spec = NucleusSpec(A=A, Z=Z)
    assert type(spec.A) is int and type(spec.Z) is int
    assert spec.A == 208 and spec.Z == 82


# Gap item 2: point-target sigma boundary matrix, including NumPy-scalar
# carriers of each offending value and the smallest representable positives.

BAD_POINT_SIGMA_MATRIX = [
    pytest.param(0.0, id="zero"),
    pytest.param(-0.0, id="negative-zero"),
    pytest.param(-5.0, id="negative"),
    pytest.param(-5e-324, id="negative-subnormal"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(np.float64(float("nan")), id="nan-numpy"),
    pytest.param(float("inf"), id="plus-inf"),
    pytest.param(np.float64(float("-inf")), id="minus-inf-numpy"),
]


@pytest.mark.parametrize("bad", BAD_POINT_SIGMA_MATRIX)
def test_point_target_sigma_boundary_matrix_rejected(bad) -> None:
    with pytest.raises(ValueError) as excinfo:
        NucleusSpec(name="p", A=1, Z=1, sigma_nn_inel_mb=bad)
    msg = str(excinfo.value)
    assert "sigma_nn_inel_mb must be finite and positive" in msg
    assert repr(float(bad)) in msg


VALID_POINT_SIGMA_MATRIX = [
    pytest.param(5e-324, id="smallest-positive-subnormal"),
    pytest.param(42.5, id="ordinary-positive"),
    pytest.param(np.float64(31.7), id="numpy-float64"),
]


@pytest.mark.parametrize("good", VALID_POINT_SIGMA_MATRIX)
def test_point_target_sigma_boundary_matrix_accepts_positive_finite(
    good,
) -> None:
    spec = NucleusSpec(name="p", A=1, Z=1, sigma_nn_inel_mb=good)
    assert type(spec.sigma_nn_inel_mb) is float
    sampler = WoundedNucleonSampler(spec, rng=np.random.default_rng(0))
    assert sampler.sigma_pA_mb() == good       # exact value preservation


# Gap item 3: every profile float field accepts Python floats, numeric
# strings, and NumPy floats, always storing a built-in float of the same
# exact value.


def _as_numeric_string(target: float) -> str:
    return f"{target}"


PROFILE_FLOAT_FIELDS = (
    ("radius_fm", 6.62),
    ("diffuseness_fm", 0.546),
    ("quad_alpha", 0.25),
    ("sigma_nn_inel_mb", 31.7),
)

FLOAT_FIELD_KIND_MATRIX = [
    pytest.param(field, value, value, id=f"{field}-builtin-float")
    for field, value in PROFILE_FLOAT_FIELDS
] + [
    pytest.param(
        field, _as_numeric_string(value), value,
        id=f"{field}-numeric-string",
    )
    for field, value in PROFILE_FLOAT_FIELDS
] + [
    pytest.param(field, np.float64(value), value, id=f"{field}-numpy-float64")
    for field, value in PROFILE_FLOAT_FIELDS
]


@pytest.mark.parametrize(("field", "given", "expected"), FLOAT_FIELD_KIND_MATRIX)
def test_profile_fields_store_builtin_float_across_scalar_kinds(
    field, given, expected
) -> None:
    spec = NucleusSpec(**_loose_spec_kwargs(**{field: given}))
    stored = getattr(spec, field)
    assert type(stored) is float, field
    assert stored == expected, field          # bit-stable conversion


BOOLEAN_AND_FOREIGN_FIELD_MATRIX = [
    pytest.param("radius_fm", True, id="radius-bool"),
    pytest.param("diffuseness_fm", np.bool_(False), id="diffuseness-numpy-bool"),
    pytest.param("quad_alpha", True, id="alpha-bool"),
    pytest.param("sigma_nn_inel_mb", np.bool_(True), id="sigma-numpy-bool"),
    pytest.param("radius_fm", None, id="radius-none"),
    pytest.param("diffuseness_fm", [], id="diffuseness-list"),
    pytest.param("quad_alpha", b"6.62", id="alpha-bytes"),
    pytest.param("sigma_nn_inel_mb", {}, id="sigma-dict"),
]


@pytest.mark.parametrize(("field", "value"), BOOLEAN_AND_FOREIGN_FIELD_MATRIX)
def test_profile_fields_reject_boolean_and_foreign_kinds(
    field, value
) -> None:
    """Booleans and anything float() would mis-coerce are refused up front."""
    with pytest.raises(ValueError) as excinfo:
        NucleusSpec(**_loose_spec_kwargs(**{field: value}))
    msg = str(excinfo.value)
    assert f"{field} must be a float" in msg
    assert repr(value) in msg


def test_harmonic_oscillator_string_spec_matches_numeric_grid_bitwise() -> None:
    """String normalization is lossless for the oscillator branch too.

    Complements the Woods-Saxon pin above: the sampled tables built from
    a fully string-valued C12 spec are bit-identical to the numeric-built
    counterpart, including mu, p_inel, and the min-bias CDF.
    """
    common = dict(
        name="C12",
        A=12,
        Z=6,
        profile="harmonic_oscillator",
    )
    numeric = NucleusSpec(
        **_loose_spec_kwargs(
            **common,
            radius_fm=1.88,
            quad_alpha=0.25,
            sigma_nn_inel_mb=31.7,
        )
    )
    textual = NucleusSpec(
        **_loose_spec_kwargs(
            **common,
            radius_fm="1.88",
            quad_alpha="0.25",
            sigma_nn_inel_mb="31.7",
        )
    )
    sampler_num = WoundedNucleonSampler(numeric, rng=np.random.default_rng(17))
    sampler_txt = WoundedNucleonSampler(textual, rng=np.random.default_rng(17))
    for attr in ("_b_grid", "_mu", "_p_inel", "_b_cdf"):
        assert getattr(sampler_num, attr).tobytes() \
            == getattr(sampler_txt, attr).tobytes(), attr
    assert sampler_num.sigma_pA_mb() == sampler_txt.sigma_pA_mb()
    assert (
        sampler_num.mean_target_wounds_minbias()
        == sampler_txt.mean_target_wounds_minbias()
    )


# Gap item 9 widening: the budget-refusal contract holds across the
# subnormal-step family, both extended-profile branches with overflowing
# geometry, and the ordinary (finite-span) over-budget path.

SUBNORMAL_DR_VARIANTS = [
    pytest.param(5e-324, id="smallest-subnormal"),
    pytest.param(4.9e-324, id="next-subnormal"),
    pytest.param(np.float64(5e-324), id="subnormal-numpy"),
]


@pytest.mark.parametrize("dr", SUBNORMAL_DR_VARIANTS)
def test_planner_budget_refusal_covers_subnormal_step_family(dr) -> None:
    """Every subnormal-step carrier lands in the documented budget refusal.

    The sampler normalizes the validated step to a built-in float at
    entry, so both carrier kinds run the planner span arithmetic in
    pure-Python doubles: no NumPy RuntimeWarning can be emitted on the
    rejection path, matching the no-warning-on-rejection policy of the
    histogram layer. Any leaked warning fails via the error filter.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError) as excinfo:
            WoundedNucleonSampler(
                _pb208_spec(), rng=np.random.default_rng(0), dr_fm=dr
            )
    assert excinfo.type is ValueError
    msg = str(excinfo.value)
    assert "grid-cell budget exceeded" in msg
    assert "coarsen dr_fm" in msg


def test_numpy_carrier_dr_fm_matches_builtin_float_grids_bitwise() -> None:
    """float() normalization of dr_fm is value-preserving.

    A valid np.float64 step produces byte-identical b/z-dependent tables
    to the built-in-float carrier, with exact cross-section equality,
    under strict warning filters on both constructions.
    """
    spec = _pb208_spec()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sampler_py = WoundedNucleonSampler(
            spec, rng=np.random.default_rng(11), dr_fm=0.02
        )
        sampler_np = WoundedNucleonSampler(
            spec, rng=np.random.default_rng(11),
            dr_fm=np.float64(0.02),    # type: ignore[arg-type]
        )
    assert type(sampler_np._dr) is float
    for attr in ("_b_grid", "_mu", "_p_inel", "_b_cdf"):
        assert getattr(sampler_py, attr).tobytes() \
            == getattr(sampler_np, attr).tobytes(), attr
    assert sampler_py.sigma_pA_mb() == sampler_np.sigma_pA_mb()
    assert (
        sampler_py.mean_target_wounds_minbias()
        == sampler_np.mean_target_wounds_minbias()
    )


OVERFLOWING_GEOMETRY_MATRIX = [
    pytest.param(
        dict(name="Overflower", A=97, Z=48, profile="woods_saxon",
             radius_fm=6.62, diffuseness_fm=1e307, sigma_nn_inel_mb=31.7),
        id="woods-saxon-huge-diffuseness",
    ),
    pytest.param(
        dict(name="Overflower", A=97, Z=48, profile="harmonic_oscillator",
             radius_fm=3e307, sigma_nn_inel_mb=31.7),
        id="oscillator-huge-radius",
    ),
]


@pytest.mark.parametrize("spec_kwargs", OVERFLOWING_GEOMETRY_MATRIX)
def test_planner_budget_refusal_covers_derived_extent_overflow(
    spec_kwargs,
) -> None:
    """Finite geometry whose derived endpoints overflow double precision."""
    spec = NucleusSpec(**_loose_spec_kwargs(**spec_kwargs))
    with pytest.raises(ValueError) as excinfo:
        WoundedNucleonSampler(spec, rng=np.random.default_rng(0))
    assert excinfo.type is ValueError
    msg = str(excinfo.value)
    assert "grid-cell budget exceeded" in msg
    assert "Overflower" in msg
    assert "coarsen dr_fm" in msg


def test_ordinary_over_budget_step_keeps_axis_counting_contract() -> None:
    """Finite spans still go through exact counting, then the budget error.

    dr = 1e-9 fm plans ~1e10 cells per axis without any arithmetic
    overflow: the counted-axis path reports the first capped axis (n_r)
    and the remediation, proving the sentinel route above did not displace
    the original per-axis accounting.
    """
    with pytest.raises(ValueError) as excinfo:
        WoundedNucleonSampler(
            _pb208_spec(), rng=np.random.default_rng(0), dr_fm=1e-9
        )
    msg = str(excinfo.value)
    assert "grid-cell budget exceeded" in msg
    assert "n_r" in msg                      # first capped axis is named
    assert "cells" in msg
    assert "coarsen dr_fm" in msg
