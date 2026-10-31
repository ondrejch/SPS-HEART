#!/usr/bin/env python3
"""Regression tests for the production fit validation and identifiability seam.

Covers the shared validation seam in ``sps_core`` and the production solver
``_fit_global_nonnegative_weights`` in ``benchmark_multispecies_constrained``
(committed behavior of 41eef69 and 395fdfb):

1. an unsuccessful optimizer result is fatal and produces no usable weights;
2. a non-finite objective at a "successful" solution is fatal;
3. a non-finite fitted weight is fatal;
4. malformed matrix/vector shapes fail before SciPy is called;
5. non-finite data, sigma, component, and prior values fail before SciPy is
   called;
6. a rank-deficient production design is refused by default; the explicit
   opt-out succeeds and stamps the payload non-identifiable; the library
   helper ``fit_nnls_weighted`` carries the opposite default and stamps its
   payload;
7. a failed L-BFGS-B result followed by a failed SLSQP result is fatal and
   names the primary failure; a failed-but-lower-objective alternate is never
   selected;
8. a successful fallback is accepted only when its complete result passes
   validation.

SciPy is stubbed with monkeypatch at the names the modules under test look
up: both modules use module-level ``from scipy.optimize import minimize, nnls``
imports, so the patch targets are ``benchmark_multispecies_constrained.minimize``/``.nnls``
and ``sps_core.nnls``. All tests are hermetic: no external tables, no
``external_data`` marker.
"""

from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import helpers.benchmarking.benchmark_multispecies_constrained as bmc
import helpers.benchmarking.sps_core as sps_core
from helpers.benchmarking.sps_core import (
    ACTIVE_WEIGHT_THRESHOLD,
    FIT_BOUND_CLIP_TOLERANCE,
    FULL_RANK_STAMP,
    RANK_DEFICIENT_STAMP,
    fit_nnls_weighted,
)

N_COMP = 3
N_ROWS = 12
SEED = 20260830
W_TRUE = np.array([1.0, 0.5, 0.2])
PRIOR_A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
PRIOR_B = np.array([1.0, 0.5])


def _design():
    """Deterministic well-conditioned sigma-weighted fit problem."""
    rng = np.random.default_rng(SEED)
    a_raw = rng.uniform(0.4, 2.0, size=(N_ROWS, N_COMP))
    sigma = rng.uniform(0.05, 0.25, size=N_ROWS)
    y = a_raw @ W_TRUE
    a_weighted = a_raw / sigma[:, None]
    b_weighted = y / sigma
    return a_raw, y, sigma, a_weighted, b_weighted


def _fit_kwargs(
    a_raw, y, sigma, a_weighted, b_weighted, overrides=None, **more
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = dict(
        objective="gaussian_point_mle",
        A_train_raw=a_raw,
        y_train=y,
        sigma_train=sigma,
        A_train_weighted=a_weighted,
        b_train_weighted=b_weighted,
        A_prior=None,
        b_prior=None,
        table9_fraction_rows=None,
        table9_fraction_targets=None,
        table9_fraction_lambda=0.0,
        table9_fraction_eps=1e-12,
        maxiter=200,
        ftol=1e-10,
    )
    if overrides:
        kwargs.update(overrides)
    kwargs.update(more)
    return kwargs


def _forbid_scipy(monkeypatch):
    """Stub both SciPy entry points with sentinels that fail if called."""

    def _sentinel(name):
        def _fail(*args, **kwargs):
            raise AssertionError(
                f"{name} must not be called: input validation has to fail first"
            )

        return _fail

    monkeypatch.setattr(bmc, "minimize", _sentinel("scipy.optimize.minimize"))
    monkeypatch.setattr(bmc, "nnls", _sentinel("scipy.optimize.nnls"))


def _stub_warm_start_nnls(monkeypatch, w=W_TRUE):
    """Stub the warm-start NNLS call of the point-likelihood paths."""

    def _nnls(A, b):
        return np.asarray(w, dtype=float).copy(), 0.0

    monkeypatch.setattr(bmc, "nnls", _nnls)


def _opt_result(x, fun, success, status=0, message="ok", nfev=1, nit=1):
    return OptimizeResult(
        x=np.asarray(x, dtype=float),
        success=bool(success),
        status=int(status),
        message=str(message),
        fun=float(fun),
        nfev=int(nfev),
        nit=int(nit),
    )


def _stub_minimize(monkeypatch, results):
    """Queue optimizer results; record every call's method and start point."""
    calls = []
    queue = list(results)

    def _minimize(fun, x0=None, method=None, **kwargs):
        calls.append({"method": method, "x0": np.asarray(x0, dtype=float).copy()})
        return queue.pop(0)

    monkeypatch.setattr(bmc, "minimize", _minimize)
    return calls


# ---------------------------------------------------------------------------
# Case 1: success=False raises and produces no usable weights.
# ---------------------------------------------------------------------------


def test_case1_unsuccessful_minimize_result_is_fatal(monkeypatch):
    """An optimizer verdict of success=False must raise, never return weights."""
    a_raw, y, sigma, a_w, b_w = _design()
    _stub_warm_start_nnls(monkeypatch)
    _stub_minimize(
        monkeypatch,
        [_opt_result([0.5, 0.5, 0.5], 12.0, success=False, status=2,
                     message="LBFGSB-DIVERGED")],
    )
    with pytest.raises(RuntimeError) as excinfo:
        bmc._fit_global_nonnegative_weights(
            **_fit_kwargs(a_raw, y, sigma, a_w, b_w)
        )
    msg = str(excinfo.value)
    assert "did not converge" in msg
    assert "LBFGSB-DIVERGED" in msg
    assert "status=2" in msg


# ---------------------------------------------------------------------------
# Case 2: success=True with a non-finite objective raises.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_fun", [float("nan"), float("inf"), float("-inf")])
def test_case2_nonfinite_objective_is_fatal(monkeypatch, bad_fun):
    """A successful verdict with a non-finite objective must raise."""
    a_raw, y, sigma, a_w, b_w = _design()
    _stub_warm_start_nnls(monkeypatch)
    _stub_minimize(
        monkeypatch,
        [_opt_result(W_TRUE, bad_fun, success=True, message="CONVERGED")],
    )
    with pytest.raises(RuntimeError) as excinfo:
        bmc._fit_global_nonnegative_weights(
            **_fit_kwargs(a_raw, y, sigma, a_w, b_w)
        )
    msg = str(excinfo.value)
    assert "non-finite" in msg
    assert "objective" in msg


# ---------------------------------------------------------------------------
# Case 3: success=True with a NaN or Inf weight raises.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_weight", [float("nan"), float("inf"), float("-inf")])
def test_case3_nonfinite_weight_is_fatal(monkeypatch, bad_weight):
    """A successful verdict with a non-finite weight entry must raise."""
    a_raw, y, sigma, a_w, b_w = _design()
    _stub_warm_start_nnls(monkeypatch)
    _stub_minimize(
        monkeypatch,
        [_opt_result([1.0, bad_weight, 0.5], 1.0, success=True, message="CONVERGED")],
    )
    with pytest.raises(RuntimeError) as excinfo:
        bmc._fit_global_nonnegative_weights(
            **_fit_kwargs(a_raw, y, sigma, a_w, b_w)
        )
    msg = str(excinfo.value)
    assert "non-finite" in msg
    assert "w[1]" in msg


# ---------------------------------------------------------------------------
# Case 4: malformed matrix/vector shapes fail before SciPy is called.
# ---------------------------------------------------------------------------


def _shape_override(case, a_raw, y, sigma, a_w, b_w):
    if case == "design-column-count":
        return {"A_train_weighted": a_w[:, :-1]}
    if case == "target-row-count":
        return {"y_train": y[:-1]}
    if case == "sigma-row-count":
        return {"sigma_train": sigma[:-1]}
    if case == "weighted-target-row-count":
        return {"b_train_weighted": b_w[:-1]}
    if case == "prior-design-columns":
        return {"A_prior": np.ones((2, N_COMP - 1)), "b_prior": np.ones(2)}
    if case == "prior-target-rows":
        return {"A_prior": np.ones((2, N_COMP)), "b_prior": np.ones(3)}
    raise AssertionError(f"unknown shape case {case!r}")


_SHAPE_CASES = (
    ("design-column-count", "A_train_weighted"),
    ("target-row-count", "y_train"),
    ("sigma-row-count", "sigma_train"),
    ("weighted-target-row-count", "b_train_weighted"),
    ("prior-design-columns", "A_prior"),
    ("prior-target-rows", "b_prior"),
)


@pytest.mark.parametrize("case,expected_name", _SHAPE_CASES)
def test_case4_malformed_shapes_fail_before_scipy(monkeypatch, case, expected_name):
    """Wrong row/col counts and prior shapes raise ValueError before any solve."""
    a_raw, y, sigma, a_w, b_w = _design()
    _forbid_scipy(monkeypatch)
    overrides = _shape_override(case, a_raw, y, sigma, a_w, b_w)
    with pytest.raises(ValueError) as excinfo:
        bmc._fit_global_nonnegative_weights(
            **_fit_kwargs(
                a_raw, y, sigma, a_w, b_w,
                overrides=overrides, objective="nnls_weighted",
            )
        )
    assert expected_name in str(excinfo.value)


# ---------------------------------------------------------------------------
# Case 5: non-finite data, sigma, component, and prior values fail before
# SciPy is called.
# ---------------------------------------------------------------------------


def _nonfinite_override(case, a_raw, y, sigma, a_w, b_w):
    if case == "data-matrix":
        a_bad = a_raw.copy()
        a_bad[2, 1] = np.nan
        return {"A_train_raw": a_bad}, "row 2, column 1"
    if case == "target-vector":
        y_bad = y.copy()
        y_bad[3] = np.inf
        return {"y_train": y_bad}, None
    if case == "sigma-vector":
        s_bad = sigma.copy()
        s_bad[5] = np.nan
        return {"sigma_train": s_bad}, None
    if case == "weighted-design":
        a_bad = a_w.copy()
        a_bad[0, 2] = np.inf
        return {"A_train_weighted": a_bad}, None
    if case == "weighted-target":
        b_bad = b_w.copy()
        b_bad[7] = np.nan
        return {"b_train_weighted": b_bad}, None
    if case == "prior-design":
        ap_bad = np.array([[1.0, 0.0, np.nan], [0.0, 1.0, 0.0]])
        return {"A_prior": ap_bad, "b_prior": PRIOR_B}, None
    raise AssertionError(f"unknown non-finite case {case!r}")


_NONFINITE_CASES = (
    ("data-matrix", "A_train_raw"),
    ("target-vector", "y_train"),
    ("sigma-vector", "sigma_train"),
    ("weighted-design", "A_train_weighted"),
    ("weighted-target", "b_train_weighted"),
    ("prior-design", "A_prior"),
)


@pytest.mark.parametrize("case,expected_name", _NONFINITE_CASES)
def test_case5_nonfinite_inputs_fail_before_scipy(monkeypatch, case, expected_name):
    """NaN/Inf inputs raise ValueError before any solve, offender located."""
    a_raw, y, sigma, a_w, b_w = _design()
    _forbid_scipy(monkeypatch)
    overrides, locator = _nonfinite_override(case, a_raw, y, sigma, a_w, b_w)
    with pytest.raises(ValueError) as excinfo:
        bmc._fit_global_nonnegative_weights(
            **_fit_kwargs(
                a_raw, y, sigma, a_w, b_w,
                overrides=overrides, objective="nnls_weighted",
            )
        )
    msg = str(excinfo.value)
    assert "non-finite" in msg
    assert expected_name in msg
    if locator is not None:
        assert locator in msg


# ---------------------------------------------------------------------------
# Case 6: rank-deficient production design policy and library contract.
# ---------------------------------------------------------------------------


def _rank_deficient_design():
    """Well-conditioned problem with a deliberately zero component column."""
    a_raw, y, sigma, a_w, b_w = _design()
    a_raw = a_raw.copy()
    a_w = a_w.copy()
    a_raw[:, 2] = 0.0
    a_w[:, 2] = 0.0
    return a_raw, y, sigma, a_w, b_w


def test_case6_rank_deficient_design_refused_by_default(monkeypatch):
    """Production default refuses a rank-deficient design before the solve."""
    a_raw, y, sigma, a_w, b_w = _rank_deficient_design()
    _forbid_scipy(monkeypatch)
    with pytest.raises(ValueError) as excinfo:
        bmc._fit_global_nonnegative_weights(
            **_fit_kwargs(a_raw, y, sigma, a_w, b_w, objective="nnls_weighted")
        )
    msg = str(excinfo.value)
    assert "does not have full column rank" in msg
    assert "numerical rank 2 < n_components = 3" in msg
    assert "weighted_data_rows" in msg
    assert "allow_rank_deficient=True" in msg


def test_case6_rank_deficient_opt_out_stamps_non_identifiable(monkeypatch):
    """Explicit opt-out succeeds; payload pinned against numpy references."""
    a_raw, y, sigma, a_w, b_w = _rank_deficient_design()
    w, info = bmc._fit_global_nonnegative_weights(
        **_fit_kwargs(
            a_raw, y, sigma, a_w, b_w,
            objective="nnls_weighted", allow_rank_deficient=True,
        )
    )
    ref_rank = int(np.linalg.matrix_rank(a_w))
    ref_cond = float(np.linalg.cond(a_w, p=2))
    ref_svals = np.linalg.svd(a_w, compute_uv=False)
    assert info["identifiability"] == RANK_DEFICIENT_STAMP
    assert info["design_scope"] == "weighted_data_rows"
    assert info["design_n_components"] == N_COMP
    assert info["design_rank"] == ref_rank == 2
    assert info["design_condition_number"] == pytest.approx(ref_cond, rel=1e-13)
    np.testing.assert_allclose(
        np.asarray(info["design_singular_values"], dtype=float),
        ref_svals, rtol=1e-13, atol=0.0,
    )
    assert np.all(np.diff(np.asarray(info["design_singular_values"])) <= 0.0)
    assert w.shape == (N_COMP,)
    assert w[2] == 0.0
    assert info["n_active_weights"] == int(np.count_nonzero(w > ACTIVE_WEIGHT_THRESHOLD))
    assert info["success"] is True


def test_case6_fit_nnls_weighted_rank_contract():
    """Library helper defaults to permissive and stamps; strict mode refuses."""
    a_raw, y, sigma, a_w, b_w = _rank_deficient_design()
    sv = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    design = a_w / sv[:, None]
    ref_rank = int(np.linalg.matrix_rank(design))
    ref_cond = float(np.linalg.cond(design, p=2))
    ref_svals = np.linalg.svd(design, compute_uv=False)

    w, info = fit_nnls_weighted(a_w, y, sigma)
    assert info["method"] == "scipy.optimize.nnls"
    assert int(info["nnls_maxiter"]) >= 1000
    assert info["identifiability"] == RANK_DEFICIENT_STAMP
    assert info["design_scope"] == "weighted_data_rows"
    assert info["design_rank"] == ref_rank == 2
    assert info["design_condition_number"] == pytest.approx(ref_cond, rel=1e-13)
    np.testing.assert_allclose(
        np.asarray(info["design_singular_values"], dtype=float),
        ref_svals, rtol=1e-13, atol=0.0,
    )
    assert info["n_active_weights"] == int(np.count_nonzero(w > ACTIVE_WEIGHT_THRESHOLD))

    with pytest.raises(ValueError) as excinfo:
        fit_nnls_weighted(a_w, y, sigma, allow_rank_deficient=False)
    msg = str(excinfo.value)
    assert "does not have full column rank" in msg
    assert "numerical rank 2 < n_components = 3" in msg


def test_case6_rank_policy_defaults_are_opposite():
    """Production solver refuses by default; the library helper stamps by default."""
    prod_default = inspect.signature(
        bmc._fit_global_nonnegative_weights
    ).parameters["allow_rank_deficient"].default
    lib_default = inspect.signature(
        fit_nnls_weighted
    ).parameters["allow_rank_deficient"].default
    assert prod_default is False
    assert lib_default is True


# ---------------------------------------------------------------------------
# Case 7: failed L-BFGS-B followed by failed SLSQP is fatal; a failed
# lower-objective alternate is never selected.
# ---------------------------------------------------------------------------


def test_case7_failed_primary_and_failed_fallback_is_fatal(monkeypatch):
    """Neither optimizer succeeds: fatal, naming the primary failure."""
    a_raw, y, sigma, a_w, b_w = _design()
    _stub_warm_start_nnls(monkeypatch)
    primary = _opt_result([0.5, 0.5, 0.5], 5.0, success=False, status=1,
                          message="PRIMARY-LBFGSB-FAILED")
    # The alternate reports a LOWER objective (-1.0 < 5.0) but failed; the old
    # acceptance rule would have selected it and returned its weights.
    alternate = _opt_result([9.9, 9.9, 9.9], -1.0, success=False, status=4,
                            message="ALT-SLSQP-ALSO-FAILED")
    calls = _stub_minimize(monkeypatch, [primary, alternate])
    with pytest.raises(RuntimeError) as excinfo:
        bmc._fit_global_nonnegative_weights(
            **_fit_kwargs(a_raw, y, sigma, a_w, b_w,
                          objective="lognormal_point_mle")
        )
    msg = str(excinfo.value)
    assert "did not converge" in msg
    assert "PRIMARY-LBFGSB-FAILED" in msg
    assert "ALT-SLSQP-ALSO-FAILED" in msg
    assert [c["method"] for c in calls] == ["L-BFGS-B", "SLSQP"]


# ---------------------------------------------------------------------------
# Case 8: a successful fallback is accepted only when its complete result
# passes validation.
# ---------------------------------------------------------------------------


def test_case8a_valid_successful_fallback_is_accepted(monkeypatch):
    """Failed L-BFGS-B then fully valid SLSQP: the fallback result is used."""
    a_raw, y, sigma, a_w, b_w = _design()
    _stub_warm_start_nnls(monkeypatch)
    primary = _opt_result([0.5, 0.5, 0.5], 5.0, success=False, status=1,
                          message="PRIMARY-FAILED")
    alternate = _opt_result(W_TRUE, 1.25, success=True, status=0,
                            message="SLSQP-CONVERGED")
    calls = _stub_minimize(monkeypatch, [primary, alternate])
    w, info = bmc._fit_global_nonnegative_weights(
        **_fit_kwargs(a_raw, y, sigma, a_w, b_w,
                      objective="lognormal_point_mle")
    )
    assert [c["method"] for c in calls] == ["L-BFGS-B", "SLSQP"]
    # The fallback restarts from the clipped primary solution.
    np.testing.assert_allclose(calls[1]["x0"], [0.5, 0.5, 0.5], rtol=0, atol=0)
    np.testing.assert_allclose(w, W_TRUE, rtol=0, atol=0)
    assert info["success"] is True
    assert info["status"] == 0
    assert info["message"] == "SLSQP-CONVERGED"
    assert info["fun"] == 1.25
    assert info["identifiability"] == FULL_RANK_STAMP
    assert info["n_active_weights"] == int(np.count_nonzero(w > ACTIVE_WEIGHT_THRESHOLD))
    # The payload must name the optimizer whose result was actually accepted:
    # a fit adopted from the SLSQP fallback is never reported as L-BFGS-B.
    assert info["method"] == "scipy.optimize.minimize(SLSQP)"
    assert info["method_used"] == "SLSQP"
    assert info["primary_method"] == "L-BFGS-B"
    assert info["primary_success"] is False
    assert info["fallback_attempted"] is True
    assert info["fallback_method"] == "SLSQP"
    assert info["fallback_success"] is True


@pytest.mark.parametrize("bad_weight", [float("nan"), float("inf")])
def test_case8b_successful_fallback_with_invalid_result_is_fatal(monkeypatch, bad_weight):
    """SLSQP success with a non-finite weight is fatal, context names the fallback."""
    a_raw, y, sigma, a_w, b_w = _design()
    _stub_warm_start_nnls(monkeypatch)
    primary = _opt_result([0.5, 0.5, 0.5], 5.0, success=False, status=1,
                          message="PRIMARY-FAILED")
    alternate = _opt_result([1.0, bad_weight, 0.2], 1.25, success=True, status=0,
                            message="SLSQP-CONVERGED")
    _stub_minimize(monkeypatch, [primary, alternate])
    with pytest.raises(RuntimeError) as excinfo:
        bmc._fit_global_nonnegative_weights(
            **_fit_kwargs(a_raw, y, sigma, a_w, b_w,
                          objective="lognormal_point_mle")
        )
    msg = str(excinfo.value)
    assert "non-finite" in msg
    assert "w[1]" in msg
    assert "SLSQP fallback fit after failed primary" in msg


# ---------------------------------------------------------------------------
# Well-conditioned success: diagnostics present and self-consistent.
# ---------------------------------------------------------------------------


def _echo_minimize(monkeypatch):
    """Successful optimizer stub that returns the warm start unchanged."""

    def _minimize(fun, x0=None, method=None, **kwargs):
        x = np.asarray(x0, dtype=float).copy()
        fval, _ = fun(x)
        return _opt_result(x, float(fval), success=True, status=0,
                           message="CONVERGED")

    monkeypatch.setattr(bmc, "minimize", _minimize)


@pytest.mark.parametrize(
    "objective,use_priors,expected_scope",
    [
        ("nnls_weighted", False, "weighted_data_rows"),
        ("nnls_weighted", True, "augmented_weighted_design"),
        ("gaussian_point_mle", False, "weighted_linear_component_matrix"),
        ("gaussian_point_mle", True, "weighted_linear_component_matrix_with_prior_rows"),
        ("lognormal_point_mle", False, "weighted_linear_component_matrix"),
    ],
    ids=[
        "nnls",
        "nnls-with-prior-rows",
        "gaussian",
        "gaussian-with-prior-rows",
        "lognormal",
    ],
)
def test_well_conditioned_fit_diagnostics_are_self_consistent(
    monkeypatch, objective, use_priors, expected_scope
):
    """A full-rank fit records rank, singular values, condition, and stamp."""
    a_raw, y, sigma, a_w, b_w = _design()
    prior_kwargs = (
        {"A_prior": PRIOR_A, "b_prior": PRIOR_B} if use_priors else {}
    )
    if objective == "nnls_weighted":
        # Real SciPy NNLS solves the production design; no stubs needed.
        w, info = bmc._fit_global_nonnegative_weights(
            **_fit_kwargs(a_raw, y, sigma, a_w, b_w,
                          objective=objective, **prior_kwargs)
        )
        design = a_w if not use_priors else np.vstack([a_w, PRIOR_A])
    else:
        _stub_warm_start_nnls(monkeypatch)
        _echo_minimize(monkeypatch)
        w, info = bmc._fit_global_nonnegative_weights(
            **_fit_kwargs(a_raw, y, sigma, a_w, b_w,
                          objective=objective, **prior_kwargs)
        )
        design = a_raw / sigma[:, None]
        if use_priors:
            design = np.vstack([design, PRIOR_A])

    ref_rank = int(np.linalg.matrix_rank(design))
    ref_cond = float(np.linalg.cond(design, p=2))
    ref_svals = np.linalg.svd(design, compute_uv=False)

    assert info["success"] is True
    assert info["design_scope"] == expected_scope
    assert info["design_n_components"] == N_COMP
    assert info["design_rank"] == N_COMP == ref_rank
    cond = float(info["design_condition_number"])  # type: ignore[arg-type]
    assert math.isfinite(cond)
    assert 0.0 < cond < 1e3
    assert cond == pytest.approx(ref_cond, rel=1e-13)
    svals = np.asarray(info["design_singular_values"], dtype=float)
    assert svals.shape == (N_COMP,)
    np.testing.assert_allclose(svals, ref_svals, rtol=1e-13, atol=0.0)
    assert np.all(np.diff(svals) <= 0.0)
    assert info["identifiability"] == FULL_RANK_STAMP
    assert info["n_active_weights"] == int(np.count_nonzero(w > ACTIVE_WEIGHT_THRESHOLD))
    if objective != "nnls_weighted":
        jac_cond = float(info["jacobian_condition_number"])  # type: ignore[arg-type]
        assert math.isfinite(jac_cond)
        assert jac_cond > 0.0


# ---------------------------------------------------------------------------
# Bound-tolerance constant pin (documented clip tolerance of the seam).
# ---------------------------------------------------------------------------


def test_bound_clip_tolerance_constant_is_pinned():
    """The documented numerical slack on the non-negativity bound."""
    assert FIT_BOUND_CLIP_TOLERANCE == 1e-9


# ---------------------------------------------------------------------------
# Case 9: raw sigma is validated before the positive floor is applied.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_sigma", [float("-inf"), float("inf"), float("nan")])
def test_case9_nonfinite_sigma_fails_before_the_floor(monkeypatch, bad_sigma):
    """A non-finite uncertainty cannot be masked by the documented 1e-12
    floor: -inf used to be floored into a tiny positive sigma, so its
    validation had to run on the raw vector before the floor was applied."""
    a_raw, y, sigma, a_w, b_w = _design()
    s_bad = sigma.copy()
    s_bad[5] = bad_sigma
    _forbid_scipy(monkeypatch)
    with pytest.raises(ValueError) as excinfo:
        bmc._fit_global_nonnegative_weights(
            **_fit_kwargs(
                a_raw, y, sigma, a_w, b_w,
                overrides={"sigma_train": s_bad}, objective="nnls_weighted",
            )
        )
    msg = str(excinfo.value)
    assert "non-finite" in msg
    assert "sigma_train" in msg


# ---------------------------------------------------------------------------
# Case 10: paired presence of linear priors and Table-9 fraction inputs.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("half_specified", ["A_prior_only", "b_prior_only"])
def test_case10_half_specified_linear_prior_raises(monkeypatch, half_specified):
    """Exactly one of A_prior/b_prior present is a configuration error, not
    a silent no-op: it must fail before any solver call with the contract
    named."""
    a_raw, y, sigma, a_w, b_w = _design()
    _forbid_scipy(monkeypatch)
    overrides = (
        {"A_prior": PRIOR_A} if half_specified == "A_prior_only" else {"b_prior": PRIOR_B}
    )
    with pytest.raises(ValueError) as excinfo:
        bmc._fit_global_nonnegative_weights(
            **_fit_kwargs(
                a_raw, y, sigma, a_w, b_w,
                overrides=overrides, objective="nnls_weighted",
            )
        )
    msg = str(excinfo.value)
    assert "must be supplied as a pair" in msg
    assert "exactly one" in msg


def test_case10_empty_prior_pair_keeps_noop_behavior(monkeypatch):
    """Both prior arrays present and consistently empty stays the documented
    no-op: the design is the weighted data rows and no prior rows enter."""
    a_raw, y, sigma, a_w, b_w = _design()
    empty_ap = np.empty((0, N_COMP))
    empty_bp = np.empty(0)
    w, info = bmc._fit_global_nonnegative_weights(
        **_fit_kwargs(
            a_raw, y, sigma, a_w, b_w,
            overrides={"A_prior": empty_ap, "b_prior": empty_bp},
            objective="nnls_weighted",
        )
    )
    assert info["design_scope"] == "weighted_data_rows"

    # The shared library enforces the same contract (raw, unweighted inputs).
    lib_w, lib_info = fit_nnls_weighted(
        a_raw, y, sigma,
        A_prior=empty_ap, b_prior=empty_bp,
    )
    assert lib_info["n_prior_rows"] == 0
    assert lib_info["design_scope"] == "weighted_data_rows"
    np.testing.assert_allclose(lib_w, w, rtol=0, atol=0)


def test_case10_half_specified_prior_raises_in_library(monkeypatch):
    """fit_nnls_weighted rejects a half-specified prior pair before SciPy."""
    a_raw, y, sigma, a_w, b_w = _design()
    _forbid_scipy(monkeypatch)
    with pytest.raises(ValueError, match="must be supplied as a pair"):
        fit_nnls_weighted(a_raw, y, sigma, A_prior=PRIOR_A, b_prior=None)
    with pytest.raises(ValueError, match="must be supplied as a pair"):
        fit_nnls_weighted(a_raw, y, sigma, A_prior=None, b_prior=PRIOR_B)


@pytest.mark.parametrize("half_specified", ["rows_only", "targets_only"])
def test_case10_half_specified_table9_fraction_raises(half_specified):
    """The Table-9 fraction rows and targets follow the same pair rule."""
    a_raw, y, sigma, a_w, b_w = _design()
    t9r = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    t9f = np.array([0.3, 0.7])
    overrides = {"table9_fraction_rows": t9r} if half_specified == "rows_only" \
        else {"table9_fraction_targets": t9f}
    with pytest.raises(ValueError) as excinfo:
        bmc._fit_global_nonnegative_weights(
            **_fit_kwargs(
                a_raw, y, sigma, a_w, b_w,
                overrides=overrides, objective="nnls_weighted",
            )
        )
    assert "must be supplied as a pair" in str(excinfo.value)


def test_case10_positive_table9_lambda_requires_the_pair():
    """A positive fraction lambda with no fraction rows at all is a
    configuration error: the term the caller asked for cannot be formed."""
    a_raw, y, sigma, a_w, b_w = _design()
    with pytest.raises(ValueError) as excinfo:
        bmc._fit_global_nonnegative_weights(
            **_fit_kwargs(
                a_raw, y, sigma, a_w, b_w,
                overrides={"table9_fraction_lambda": 5.0},
                objective="nnls_weighted",
            )
        )
    assert "requires both table9_fraction_rows and table9_fraction_targets" in str(excinfo.value)


def test_case10_zero_lambda_table9_arrays_validated_and_reported_inactive():
    """Arrays supplied with a zero lambda are validated (a shape error must
    still fail) and reported as inactive rather than silently dropped."""
    a_raw, y, sigma, a_w, b_w = _design()
    good_rows = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    good_targets = np.array([0.3, 0.7])
    w, info = bmc._fit_global_nonnegative_weights(
        **_fit_kwargs(
            a_raw, y, sigma, a_w, b_w,
            overrides={
                "table9_fraction_rows": good_rows,
                "table9_fraction_targets": good_targets,
                "table9_fraction_lambda": 0.0,
            },
            objective="nnls_weighted",
        )
    )
    assert info["table9_fraction_active"] is False

    # A malformed pair with the same zero lambda still fails validation.
    bad_rows = np.ones((2, N_COMP - 1))
    with pytest.raises(ValueError) as excinfo:
        bmc._fit_global_nonnegative_weights(
            **_fit_kwargs(
                a_raw, y, sigma, a_w, b_w,
                overrides={
                    "table9_fraction_rows": bad_rows,
                    "table9_fraction_targets": good_targets,
                    "table9_fraction_lambda": 0.0,
                },
                objective="nnls_weighted",
            )
        )
    assert "fraction row matrix" in str(excinfo.value)


def test_case10_active_table9_fraction_refused_on_nnls_objective():
    """An active fraction prior (positive lambda, non-empty rows) remains
    refused on the nnls_weighted objective: point-likelihood only."""
    a_raw, y, sigma, a_w, b_w = _design()
    t9r = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    t9f = np.array([0.3, 0.7])
    with pytest.raises(ValueError) as excinfo:
        bmc._fit_global_nonnegative_weights(
            **_fit_kwargs(
                a_raw, y, sigma, a_w, b_w,
                overrides={
                    "table9_fraction_rows": t9r,
                    "table9_fraction_targets": t9f,
                    "table9_fraction_lambda": 5.0,
                },
                objective="nnls_weighted",
            )
        )
    assert "only supported for point-likelihood objectives" in str(excinfo.value)


def test_case10_active_table9_fraction_flag_on_point_objective(monkeypatch):
    """On the point-likelihood path the payload records the fraction term as
    active when the lambda is positive and the rows are non-empty."""
    a_raw, y, sigma, a_w, b_w = _design()
    t9r = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    t9f = np.array([0.3, 0.7])
    _stub_warm_start_nnls(monkeypatch)
    _echo_minimize(monkeypatch)
    _, info = bmc._fit_global_nonnegative_weights(
        **_fit_kwargs(
            a_raw, y, sigma, a_w, b_w,
            overrides={
                "table9_fraction_rows": t9r,
                "table9_fraction_targets": t9f,
                "table9_fraction_lambda": 5.0,
            },
            objective="gaussian_point_mle",
        )
    )
    assert info["table9_fraction_active"] is True


# ---------------------------------------------------------------------------
# Case 11: the primary-success path names L-BFGS-B and records that no
# fallback was attempted.
# ---------------------------------------------------------------------------


def test_case11_primary_success_records_method_metadata(monkeypatch):
    a_raw, y, sigma, a_w, b_w = _design()
    _stub_warm_start_nnls(monkeypatch)
    _stub_minimize(
        monkeypatch,
        [_opt_result(W_TRUE, 1.1, success=True, status=0, message="CONVERGED")],
    )
    _, info = bmc._fit_global_nonnegative_weights(
        **_fit_kwargs(a_raw, y, sigma, a_w, b_w, objective="gaussian_point_mle")
    )
    assert info["method"] == "scipy.optimize.minimize(L-BFGS-B)"
    assert info["method_used"] == "L-BFGS-B"
    assert info["primary_method"] == "L-BFGS-B"
    assert info["primary_success"] is True
    assert info["fallback_attempted"] is False
    assert info["fallback_method"] is None
    assert info["fallback_success"] is None
