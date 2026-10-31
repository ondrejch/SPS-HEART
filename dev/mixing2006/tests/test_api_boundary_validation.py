"""Fail-fast validation at public numerical API boundaries (review3 P5).

Covers the three constructors named by the review:

- ``WoundedNucleonSampler.dr_fm`` must be finite and positive on every
  profile branch (point target and extended profiles), checked before
  the point-target early return, and subject to an allocation budget
  (per-axis cell limits plus a total b x z table-memory bound) that is
  resolved entirely ahead of any grid allocation.
- ``Reaction.plab`` must be finite and positive before mass resolution
  and kinematics; negative momenta are unsupported by declaration.
- ``GenerationConfig.beam_momentum`` must be finite and positive,
  matching the Reaction policy.

Every invalid case raises immediately with a ValueError that names the
parameter and echoes the received repr.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from na49_mixing.fits import Fitter
from na49_mixing.glauber import (
    _BYTES_PER_TABLE_CELL,
    _MAX_GRID_CELLS_PER_AXIS,
    _MAX_TABLE_MEMORY_BYTES,
    _plan_grids,
    NucleusSpec,
    WoundedNucleonSampler,
)
from na49_mixing.histogram import Histogram1D
from na49_mixing.parallel import GenerationConfig
from na49_mixing.reaction import Reaction
from na49_mixing.spectra import Spectra

INVALID_NUMERICS = [
    pytest.param(0.0, id="zero"),
    pytest.param(-1e-6, id="negative-small"),
    pytest.param(-158.0, id="negative-large"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="pos-inf"),
    pytest.param(float("-inf"), id="neg-inf"),
]

# A=1 point branch (p+p bookkeeping); NucleusSpec(A=1) resolves to the
# point profile whatever profile string is stored.
POINT_SPEC = NucleusSpec(name="p", A=1, Z=1, profile="point")

EXTENDED_SPECS = [
    pytest.param(POINT_SPEC, id="point-target"),
    pytest.param(
        NucleusSpec(
            name="Pb208",
            A=208,
            Z=82,
            profile="woods_saxon",
            radius_fm=6.62,
            diffuseness_fm=0.546,
            sigma_nn_inel_mb=31.7,
        ),
        id="extended-pb208",
    ),
]


def _valid_generation_config_kwargs() -> dict:
    """Keyword arguments for a config valid except for beam_momentum.

    No filesystem access happens during construction; the gen card is
    only read when neutron species validation or generation needs it.
    """
    return dict(gen_file="unused.gen", n_events=10, seed=0)


@pytest.mark.parametrize("spec", EXTENDED_SPECS)
@pytest.mark.parametrize("bad", INVALID_NUMERICS)
def test_dr_fm_invalid_values_raise_on_every_profile_branch(spec, bad) -> None:
    with pytest.raises(ValueError) as excinfo:
        WoundedNucleonSampler(spec, rng=np.random.default_rng(0), dr_fm=bad)
    msg = str(excinfo.value)
    assert "dr_fm" in msg
    assert "finite and positive" in msg
    assert repr(bad) in msg


def test_grid_budget_refuses_absurdly_fine_positive_step() -> None:
    """A positive but far too fine step trips the budget before allocation.

    dr_fm = 1e-12 fm would request ~1e13 cells per axis; the guard sits
    ahead of every ``np.arange`` in the constructor, so the refusal does
    not allocate any grid table.
    """
    spec = NucleusSpec(
        name="Pb208",
        A=208,
        Z=82,
        profile="woods_saxon",
        radius_fm=6.62,
        diffuseness_fm=0.546,
        sigma_nn_inel_mb=31.7,
    )
    with pytest.raises(ValueError) as excinfo:
        WoundedNucleonSampler(spec, rng=np.random.default_rng(0), dr_fm=1e-12)
    msg = str(excinfo.value)
    assert "budget" in msg
    assert "dr_fm" in msg
    assert f"{_MAX_GRID_CELLS_PER_AXIS} budget" in msg


def test_grid_budget_accepts_steps_above_default_resolution() -> None:
    """The budget leaves headroom above the default Pb208 configuration.

    The heaviest configured system builds its tables at the default
    dr_fm = 0.02 fm with an order of magnitude of headroom under the
    cap; construction stays successful and physical.
    """
    spec = NucleusSpec(
        name="Pb208",
        A=208,
        Z=82,
        profile="woods_saxon",
        radius_fm=6.62,
        diffuseness_fm=0.546,
        sigma_nn_inel_mb=31.7,
    )
    sampler = WoundedNucleonSampler(spec, rng=np.random.default_rng(0))
    assert sampler.sigma_pA_mb() > spec.sigma_nn_inel_mb


def _pb_geometry() -> tuple[float, float]:
    """(r_max, surface) of the Pb208 Woods-Saxon reference spec.

    Replicates ``_density_r_max``: r_max = R + 10a with a as surface
    scale for woods_saxon.
    """
    return 6.62 + 10.0 * 0.546, 0.546


@pytest.mark.parametrize(
    ("dr_fm", "offending_axis"),
    [
        pytest.param(6.04e-4, "n_r", id="review-case-6p04e-4"),
        pytest.param(1e-3, "n_z", id="mid-range"),
    ],
)
def test_grid_budget_refuses_fine_step_admitted_by_old_r_axis_only_guard(
    dr_fm: float, offending_axis: str
) -> None:
    """Steps inside the old ratio-only check no longer build tables.

    Review finding 1: dr_fm values around and above the old
    ``r_max/dr_fm <= N`` boundary pass that single-axis ratio yet plan
    far more cells along the derived b/z axes and a multi-gigabyte
    table. The guard refuses before any allocation, names the offending
    axis, and reports the planned table size.
    """
    spec = _pb_like_spec()
    plan = _plan_grids(*_pb_geometry(), dr_fm)
    offenders = {
        name
        for name, cells in (
            ("n_r", plan.n_r),
            ("n_b", plan.n_b),
            ("n_z", plan.n_z),
        )
        if cells > _MAX_GRID_CELLS_PER_AXIS
    }
    assert offending_axis in offenders
    with pytest.raises(ValueError) as excinfo:
        WoundedNucleonSampler(spec, rng=np.random.default_rng(0), dr_fm=dr_fm)
    msg = str(excinfo.value)
    assert "budget" in msg
    assert f"{_MAX_GRID_CELLS_PER_AXIS} budget" in msg
    assert f"{offending_axis} grid axis" in msg
    # The refusal reports the planned cell count and estimated GB so
    # the review's fatal-scale case is visible from the message alone.
    assert "GB estimated" in msg
    assert f"{plan.n_b * plan.n_z} cells" in msg
    assert "coarsen dr_fm" in msg


def test_grid_budget_reports_table_cells_and_gb_near_memory_cap() -> None:
    """With every axis under its cap the memory budget still binds.

    At dr_fm = 1.8e-3 fm each axis stays within the per-axis limit but
    the b x z table exceeds the declared byte budget; the message
    reports the planned n_b x n_z product and its estimated size.
    """
    dr_fm = 1.8e-3
    plan = _plan_grids(*_pb_geometry(), dr_fm)
    assert max(plan.n_r, plan.n_b, plan.n_z) <= _MAX_GRID_CELLS_PER_AXIS
    assert plan.n_b * plan.n_z * _BYTES_PER_TABLE_CELL > _MAX_TABLE_MEMORY_BYTES
    with pytest.raises(ValueError) as excinfo:
        WoundedNucleonSampler(
            _pb_like_spec(), rng=np.random.default_rng(0), dr_fm=dr_fm
        )
    msg = str(excinfo.value)
    assert "table-memory budget exceeded" in msg
    assert f"{plan.n_b} x {plan.n_z}" in msg
    assert "GB estimated" in msg
    assert "coarsen dr_fm" in msg


def test_grid_budget_refusal_precedes_any_numpy_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The verdict lands before np.arange / zeros_like can allocate."""
    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("numpy allocation ran before the budget refusal")

    monkeypatch.setattr(np, "arange", _boom)
    monkeypatch.setattr(np, "zeros_like", _boom)
    with pytest.raises(ValueError) as excinfo:
        WoundedNucleonSampler(
            _pb_like_spec(), rng=np.random.default_rng(0), dr_fm=1.8e-3
        )
    assert "table-memory budget exceeded" in str(excinfo.value)


def test_grid_budget_boundary_plan_pair_around_declared_limit() -> None:
    """Plan-level near-boundary pair at the declared byte budget.

    The inside plan (dr_fm = 2.0e-3 fm) fits both bounds; full
    construction would transiently hold roughly two gigabytes of real
    tables, so only the pure planner is asserted on that side. The
    outside neighbor (dr_fm = 1.98e-3 fm) refuses by the memory budget.
    """
    r_max, surface = _pb_geometry()
    inside = _plan_grids(r_max, surface, 2.0e-3)
    assert max(inside.n_r, inside.n_b, inside.n_z) <= _MAX_GRID_CELLS_PER_AXIS
    inside_bytes = inside.n_b * inside.n_z * _BYTES_PER_TABLE_CELL
    assert inside_bytes <= _MAX_TABLE_MEMORY_BYTES
    outside = _plan_grids(r_max, surface, 1.98e-3)
    outside_bytes = outside.n_b * outside.n_z * _BYTES_PER_TABLE_CELL
    assert outside_bytes > _MAX_TABLE_MEMORY_BYTES
    with pytest.raises(ValueError) as excinfo:
        WoundedNucleonSampler(
            _pb_like_spec(), rng=np.random.default_rng(0), dr_fm=1.98e-3
        )
    msg = str(excinfo.value)
    assert "table-memory budget exceeded" in msg
    assert f"{outside_bytes / 1e9:.2f} GB estimated" in msg


# ---------------------------------------------------------------------------
# Planner-seam coverage for the grid-budget constants (review4 finding 1)
# ---------------------------------------------------------------------------

def test_grid_budget_both_configured_defaults_clear_budgets_widely() -> None:
    """Both configured defaults fit both budgets with quantized headroom.

    The C12 harmonic-oscillator branch derives its extents differently
    (surface scale = radius_fm, r_max = 8R), so its axis counts are not
    implied by the Woods-Saxon geometry above; each default's plan is
    checked against the constants it must satisfy, with the explicit
    margin factors the task-card wording promises.
    """
    geometries = {
        "Pb208 woods_saxon": _pb_geometry(),
        "C12 harmonic_oscillator": (8.0 * 1.88, 1.88),
    }
    for label, (r_max, surface) in geometries.items():
        plan = _plan_grids(r_max, surface, 0.02)
        worst_axis = max(plan.n_r, plan.n_b, plan.n_z)
        estimated_bytes = plan.n_b * plan.n_z * _BYTES_PER_TABLE_CELL
        assert worst_axis <= _MAX_GRID_CELLS_PER_AXIS, label
        assert estimated_bytes <= _MAX_TABLE_MEMORY_BYTES, label
        axis_headroom = _MAX_GRID_CELLS_PER_AXIS / worst_axis
        byte_headroom = _MAX_TABLE_MEMORY_BYTES / estimated_bytes
        assert axis_headroom >= 5.0, f"{label}: axis headroom {axis_headroom:.1f}"
        assert byte_headroom >= 30.0, f"{label}: byte headroom {byte_headroom:.1f}"


def test_grid_budget_harmonic_oscillator_default_constructs_physically() -> None:
    """The second configured default really builds through the guard.

    The plan-level bounds above are cheap; this companion construction
    stays affordable for the C12 size (axes stay in the hundreds-to-
    thousands range) and confirms the oscillator branch produces a
    plausible pA cross section without tripping either budget.
    """
    spec = NucleusSpec(
        name="C12",
        A=12,
        Z=6,
        profile="harmonic_oscillator",
        radius_fm=1.88,
        sigma_nn_inel_mb=31.7,
    )
    sampler = WoundedNucleonSampler(spec, rng=np.random.default_rng(0))
    sigma_pA = sampler.sigma_pA_mb()
    assert spec.sigma_nn_inel_mb < sigma_pA < 12 * spec.sigma_nn_inel_mb


def test_grid_plan_counts_replicate_constructed_arange_lengths() -> None:
    """The planner formulas stand in for the exact arange lengths used.

    Pinned decision behind the guard: the ``n_*`` counts replicate
    ``len(np.arange(...))`` bit-exactly so no allocation happens before
    the verdict. The equality is asserted against real NumPy sizing here,
    and the constructed table confirms the b axis carries exactly the
    planned length downstream.
    """
    r_max, surface = _pb_geometry()
    dr = 0.02
    plan = _plan_grids(r_max, surface, dr)
    assert plan.n_r == len(np.arange(0.0, plan.r_stop, dr))
    assert plan.n_b == len(np.arange(0.0, plan.b_stop, dr))
    assert plan.n_z == len(np.arange(-plan.z_half, plan.z_stop, dr))
    sampler = WoundedNucleonSampler(
        _pb_like_spec(), rng=np.random.default_rng(0), dr_fm=dr
    )
    assert int(sampler._b_grid.shape[0]) == plan.n_b
    assert int(sampler._p_inel.shape[0]) == plan.n_b


@pytest.mark.parametrize("bad", INVALID_NUMERICS)
def test_reaction_plab_invalid_values_raise(bad) -> None:
    with pytest.raises(ValueError) as excinfo:
        Reaction(bad)
    msg = str(excinfo.value)
    assert "plab" in msg
    assert repr(bad) in msg


def test_reaction_plab_checked_before_species_resolution() -> None:
    """The plab check fires ahead of mass resolution / _compute()."""
    with pytest.raises(ValueError) as excinfo:
        Reaction(-158.0, beam_pdg=99999, target_pdg=12345)
    # The PDG ids are unknown, yet the momentum error wins: it is
    # evaluated first.
    assert "plab" in str(excinfo.value)


def test_reaction_negative_plab_unsupported_by_declaration() -> None:
    """Negative plab must not silently mirror the positive magnitude."""
    with pytest.raises(ValueError):
        Reaction(-158.0)
    # Sanity anchor for what would otherwise happen: e_beam depends on
    # p*p only, so without the check sqrt_s would match Reaction(158.0).
    pp = Reaction(158.0)
    expected_sqrt_s = math.sqrt(
        2.0 * pp.beam_mass()
        * (pp.beam_mass() + math.hypot(pp.beam_mass(), 158.0))
    )
    assert pp.sqrt_s() == pytest.approx(expected_sqrt_s)


@pytest.mark.parametrize("bad", INVALID_NUMERICS)
def test_generation_config_beam_momentum_invalid_values_raise(bad) -> None:
    kwargs = _valid_generation_config_kwargs()
    with pytest.raises(ValueError) as excinfo:
        GenerationConfig(**kwargs, beam_momentum=bad)
    # Message style pinned by the task card verbatim.
    expected = (
        "GenerationConfig beam_momentum must be a finite positive "
        f"float, got {bad!r}"
    )
    assert str(excinfo.value) == expected


def test_happy_path_point_target_sampler_constructs_with_typical_dr() -> None:
    point = WoundedNucleonSampler(POINT_SPEC, rng=np.random.default_rng(0))
    assert point.mean_target_wounds_minbias() == 1.0
    b, k = point.sample_event()
    assert (b, k) == (0.0, 1)


def test_happy_path_lead_like_sampler_constructs_with_typical_dr() -> None:
    pb = WoundedNucleonSampler(_pb_like_spec(), rng=np.random.default_rng(0))
    assert pb.sigma_pA_mb() == pytest.approx(1685.73, rel=1e-3)


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


def test_happy_path_reaction_at_158_gev_over_c_pp() -> None:
    r = Reaction()
    assert r.plab() == 158.0
    assert r.sqrt_s() == pytest.approx(17.2702, abs=1e-3)
    assert r.gamma() > 1.0
    pp_explicit = Reaction(158.0)
    assert pp_explicit.sqrt_s() == pytest.approx(r.sqrt_s())


def test_happy_path_generation_config_defaults_construct() -> None:
    cfg = GenerationConfig(**_valid_generation_config_kwargs())
    assert cfg.beam_momentum == 158.0
    assert cfg.n_events == 10
    assert cfg.seed == 0


# ---------------------------------------------------------------------------
# Histogram1D.add: persisted bin geometry must match exactly (review4 #3)
# ---------------------------------------------------------------------------

def test_histogram_add_equal_nbins_different_range_raises_naming_both() -> None:
    lhs = Histogram1D(10, 0.0, 2.0, name="low_axis")
    rhs = Histogram1D(10, 100.0, 200.0, name="high_axis")
    with pytest.raises(ValueError) as excinfo:
        lhs.add(rhs)
    msg = str(excinfo.value)
    # Both histograms (name attributes and/or reprs) and both axes named.
    assert "Histogram1D.add()" in msg
    assert "low_axis" in msg and "high_axis" in msg
    assert "100.0" in msg and "200.0" in msg


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(dict(nbins=6, xlow=0.0, xhigh=2.0), id="nbins"),
        pytest.param(dict(nbins=5, xlow=0.5, xhigh=2.0), id="xlow"),
        pytest.param(dict(nbins=5, xlow=0.0, xhigh=2.5), id="xhigh"),
    ],
)
def test_histogram_add_every_geometry_field_is_exact(kwargs) -> None:
    """Equal nbins alone never qualifies; each field is compared exactly."""
    base = Histogram1D(5, 0.0, 2.0, name="base")
    other = Histogram1D(name="other", **kwargs)
    snapshot = base._contents.tobytes()
    with pytest.raises(ValueError, match="differing fields"):
        base.add(other)
    assert base._contents.tobytes() == snapshot


def test_histogram_add_rejected_call_leaves_lhs_bitidentical() -> None:
    lhs = Histogram1D(5, 0.0, 2.0, name="lhs")
    for i in range(5):
        lhs.fill(lhs.get_bin_center(i + 1), weight=2.0 + i)
    rhs = Histogram1D(6, 0.0, 2.0, name="wrong_nbins")
    before = lhs._contents.tobytes()
    with pytest.raises(ValueError):
        lhs.add(rhs)
    assert lhs._contents.tobytes() == before


def test_histogram_add_rejected_call_keeps_built_caches() -> None:
    """The cached CDF/probabilities survive a rejected add untouched."""
    lhs = Histogram1D(4, 0.0, 1.0, name="lhs")
    lhs.fill(0.375)
    lhs.get_random(np.random.default_rng(42))
    assert lhs._cdf is not None and lhs._probs is not None
    cdf_before, probs_before = lhs._cdf, lhs._probs
    rhs = Histogram1D(4, 10.0, 11.0, name="rhs")
    with pytest.raises(ValueError):
        lhs.add(rhs)
    assert lhs._cdf is cdf_before and lhs._probs is probs_before


def test_histogram_add_same_axis_behavior_unchanged() -> None:
    acc = Histogram1D(3, 0.0, 3.0, name="acc")
    piece = Histogram1D(3, 0.0, 3.0, name="piece")
    acc.fill_array(np.array([0.25, 1.25, 2.99]))
    piece.fill_array(np.array([0.9, 2.5]), weights=np.array([2.0, 3.0]))
    snapshot = acc._contents.copy()
    acc.add(piece, scale=2.0)
    np.testing.assert_array_equal(acc._contents, snapshot + 2.0 * piece._contents)


def test_histogram_add_int_edges_coerce_to_same_geometry() -> None:
    """Exactness applies to stored geometry; int edges coerce like floats."""
    a = Histogram1D(5, 0.0, 2.0, name="a")
    b = Histogram1D(5, 0, 2, name="b")
    a.fill(1.0)
    b.fill(1.5)
    a.add(b)
    # 1.0 filled a's bin 3, 1.5 filled b's bin 4; each add carries unit
    # weight, so after combining both bins read 1.0.
    assert a.get_bin_content(3) == pytest.approx(1.0)
    assert a.get_bin_content(4) == pytest.approx(1.0)


def test_histogram_iadd_inherits_the_geometry_check() -> None:
    a = Histogram1D(4, 0.0, 1.0, name="a")
    b = Histogram1D(4, 0.0, 2.0, name="b")
    with pytest.raises(ValueError, match="different axes"):
        a += b


@pytest.mark.parametrize("method", ["add_a", "add_b"])
def test_fitter_direct_hist_methods_surface_the_axis_error(method) -> None:
    fitter = Fitter(nbins=4, x_low=0.0, x_high=2.0)
    wrong = Histogram1D(4, 100.0, 200.0, name="mc_hist")
    with pytest.raises(ValueError, match=r"xlow \(0\.0 vs 100\.0\)"):
        getattr(fitter, method)(wrong)


@pytest.mark.parametrize("method", ["add_spectra", "subtract_spectra"])
def test_fitter_spectra_methods_surface_the_axis_error(method) -> None:
    """No weaker pre-validation and no bypass: Fitter delegates to add()."""
    fitter = Fitter(nbins=4, x_low=0.0, x_high=2.0)
    shifted = Spectra(
        name="shifted", is_data=True, nbins=4, m_low=100.0, m_high=200.0
    )
    with pytest.raises(ValueError, match=r"xlow \(0\.0 vs 100\.0\)"):
        getattr(fitter, method)(shifted)
    assert float(fitter.data_diff.get_entries()) == 0.0


def test_fitter_subtract_spectra_refusal_is_atomic_across_accumulators() -> None:
    """A refused MC subtraction mutates nothing, entry counter included.

    The MC branch of ``Fitter.subtract_spectra`` touches three histogram
    accumulators before decrementing the entry counter; the geometry
    gate must reject on the first touch so every accumulator stays
    bit-for-bit identical and the counter keeps its prior value.
    """
    fitter = Fitter(nbins=4, x_low=0.0, x_high=2.0)
    good = Histogram1D(4, 0.0, 2.0, name="good")
    good.fill_array(np.array([0.5, 1.5]), np.array([2.0, 3.0]))
    for hist in (
        fitter.data_diff,
        fitter.mc_diff,
        fitter._his_data_mass,
        fitter._his_data_mix,
        fitter._his_mc_mass,
        fitter._his_mc_mix,
    ):
        hist.add(good)
    fitter._n_mc_entries = 5
    snapshot = {
        attr: getattr(fitter, attr)._contents.tobytes()
        for attr in (
            "_his_a",
            "_his_b",
            "_his_data_mass",
            "_his_data_mix",
            "_his_mc_mass",
            "_his_mc_mix",
        )
    }
    shifted = Spectra(
        name="mc_shifted", is_data=False, nbins=4, m_low=100.0, m_high=200.0
    )
    with pytest.raises(ValueError):
        fitter.subtract_spectra(shifted)
    for attr, before in snapshot.items():
        assert getattr(fitter, attr)._contents.tobytes() == before, attr
    assert fitter.n_mc_entries == 5


def test_fitter_add_b_refused_call_leaves_accumulator_and_counter() -> None:
    fitter = Fitter(nbins=4, x_low=0.0, x_high=2.0)
    fitter.add_a(Histogram1D(4, 0.0, 2.0, name="a"))
    before = fitter.mc_diff._contents.tobytes()
    fitter._n_mc_entries = 3
    wrong = Histogram1D(4, 100.0, 200.0, name="wrong_range")
    with pytest.raises(ValueError):
        fitter.add_b(wrong)
    assert fitter.mc_diff._contents.tobytes() == before
    assert fitter.n_mc_entries == 3
