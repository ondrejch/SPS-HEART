"""Deterministic tests for :mod:`na49_mixing.fit_likelihood` (Round 1,
task card TASK-20260911-r1-covariance-nuisances).

Hermetic: no event generation; the (d, m, V) inputs are hand-constructed.
The closure reference is computed inline from the same inputs with the
explicit M8 formula of
``na49_mixing.staged_evaluation._fit_regge_binned``::

    sigma_combined = sqrt(sigma_data**2 + sigma_mc**2)   # per bin
    residual = (m - d) / sigma_combined
    chi2 = sum(residual**2)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.fit_likelihood import (
    BinnedObservable,
    NormalizationNuisance,
    combined_covariance,
    diagonal_covariance,
    evaluate_likelihood,
    unflatten_covariance,
)

NESTED_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "likelihood_nuisances.json"

# The species keys of the six-species binned NA49 set, matching the
# fit.binned.tables entries of config/staged_excitation.json.
SIX_SPECIES_SCOPES = {"pi_minus", "pi_plus", "k_plus", "k_minus", "proton", "antiproton"}


def _m8_combined_chi2(
    values: tuple[float, ...],
    model: tuple[float, ...],
    stat_errors: tuple[float, ...],
    sigma_mc: tuple[float, ...],
) -> float:
    """Inline M8 reference: the diagonal combined-error chi-square.

    Mirrors staged_evaluation.py lines 824-828 (residual (m - d) against
    the per-bin combined error); the squared residual is symmetric under
    (m - d) vs (d - m).
    """
    sigma_data = np.asarray(stat_errors, dtype=float)
    sigma_mc_arr = np.asarray(sigma_mc, dtype=float)
    sigma_combined = np.sqrt(sigma_data**2 + sigma_mc_arr**2)
    residual = (np.asarray(model, dtype=float) - np.asarray(values, dtype=float)) / sigma_combined
    return float(np.dot(residual, residual))


def test_combined_covariance_diagonal():
    """``data_cov=None`` contributes no data term; the diagonal case
    sums element-wise."""
    mc_cov = diagonal_covariance((0.05, 0.08, 0.03))
    # None means no data contribution: the result is the MC covariance
    # itself (its flattened upper triangle, zeros off the diagonal).
    assert combined_covariance(None, mc_cov) == mc_cov
    mc_matrix = unflatten_covariance(mc_cov)
    assert mc_matrix[0, 1] == 0.0 and mc_matrix[1, 2] == 0.0
    assert mc_matrix[0, 0] == pytest.approx(0.05**2, rel=1e-15)
    assert mc_matrix[2, 2] == pytest.approx(0.03**2, rel=1e-15)
    # A diagonal data covariance sums element-wise with the MC diagonal.
    data_cov = diagonal_covariance((0.20, 0.30, 0.15))
    combined = combined_covariance(data_cov, mc_cov)
    assert combined == tuple(a + b for a, b in zip(data_cov, mc_cov))
    combined_matrix = unflatten_covariance(combined)
    assert combined_matrix[0, 0] == pytest.approx(0.20**2 + 0.05**2, rel=1e-15)
    assert combined_matrix[1, 1] == pytest.approx(0.30**2 + 0.08**2, rel=1e-15)


def test_combined_covariance_full():
    """A full 3x3 case sums element-wise, hand-verified."""
    data_cov = (0.04, 0.008, 0.002, 0.09, 0.004, 0.0225)
    mc_cov = (0.0025, 0.0004, 0.0, 0.0064, 0.0002, 0.0009)
    expected = (0.0425, 0.0084, 0.002, 0.0964, 0.0042, 0.0234)
    result = combined_covariance(data_cov, mc_cov)
    assert result == pytest.approx(expected, rel=1e-12)
    matrix = unflatten_covariance(result)
    assert matrix[0, 2] == pytest.approx(matrix[2, 0], rel=1e-15)
    assert matrix[1, 0] == pytest.approx(0.0084, rel=1e-12)
    assert matrix[2, 2] == pytest.approx(0.0234, rel=1e-12)


def test_evaluate_likelihood_diagonal_no_nuisance_closure():
    """The diagonal, no-nuisance limit reproduces the M8 combined chi2
    to < 1e-9 relative."""
    values = (1.10, 0.80, 0.50, 0.30)
    stat_errors = (0.20, 0.30, 0.15, 0.25)
    sigma_mc = (0.05, 0.08, 0.03, 0.06)
    model = (1.05, 0.90, 0.55, 0.22)
    d = BinnedObservable(
        species="pi_minus",
        observable="f_xsect",
        bin_edges=(0.0, 0.2, 0.4, 0.05, 0.775, 1.5),
        values=values,
        stat_errors=stat_errors,
        cov=None,
        normalization_nuisance=None,
    )
    V = combined_covariance(
        diagonal_covariance(stat_errors), diagonal_covariance(sigma_mc)
    )
    minus2lnL, components = evaluate_likelihood(d, model, V, None)
    chi2_m8 = _m8_combined_chi2(values, model, stat_errors, sigma_mc)
    assert chi2_m8 > 0.0
    assert abs(minus2lnL - chi2_m8) / chi2_m8 < 1e-9
    # The returned value is the component dict's total.
    assert components["chi2_total"] == minus2lnL
    # The split components reproduce the combined quadratic form here too.
    assert components["chi2_data"] + components["chi2_mc"] == pytest.approx(
        chi2_m8, rel=1e-9
    )


def test_evaluate_likelihood_nuisance_shift():
    """One nuisance with eta=0.01, sigma_eta=0.02: the shift is
    0.01 * nominal value per bin, chi2_nuisance is 0.25, and chi2_total
    is the sum of the three components."""
    values = (1.0, 0.5, 0.25)
    stat_errors = (0.20, 0.30, 0.25)
    sigma_mc = (0.05, 0.08, 0.06)
    model = (0.90, 0.55, 0.30)
    d = BinnedObservable(
        species="pi_minus",
        observable="f_xsect",
        bin_edges=(0.0, 0.2, 0.4, 0.05, 0.775, 1.5),
        values=values,
        stat_errors=stat_errors,
        cov=None,
        normalization_nuisance=None,
    )
    V = combined_covariance(
        diagonal_covariance(stat_errors), diagonal_covariance(sigma_mc)
    )
    nuisance = NormalizationNuisance(
        name="sps_test_norm", eta=0.01, sigma_eta=0.02, table_scope="all"
    )
    minus2lnL, components = evaluate_likelihood(d, model, V, [nuisance])

    # Nuisance penalty: (0.01 / 0.02)^2 = 0.25.
    assert components["chi2_nuisance"] == pytest.approx(0.25, rel=1e-12)

    # The shift is eta * nominal_value per bin, subtracted from d - m.
    values_arr = np.asarray(values, dtype=float)
    shifted_residual = values_arr - np.asarray(model, dtype=float) - 0.01 * values_arr
    v_matrix = unflatten_covariance(V)
    expected_quad = float(shifted_residual @ np.linalg.solve(v_matrix, shifted_residual))
    assert minus2lnL == pytest.approx(expected_quad + 0.25, rel=1e-12)

    # chi2_total is the sum of the three components.
    assert components["chi2_total"] == pytest.approx(
        components["chi2_data"] + components["chi2_mc"] + components["chi2_nuisance"],
        rel=1e-12,
    )

    # The shift is actually applied: it differs from the no-nuisance value.
    base_quad, _ = evaluate_likelihood(d, model, V, None)
    assert abs(minus2lnL - base_quad) > 1e-6

    # The same shift through the explicit name wiring and through a
    # species scope (the observable carries no explicit reference).
    wired = BinnedObservable(
        species="pi_minus",
        observable="f_xsect",
        bin_edges=(0.0, 0.2, 0.4, 0.05, 0.775, 1.5),
        values=values,
        stat_errors=stat_errors,
        cov=None,
        normalization_nuisance="sps_test_norm",
    )
    by_name = NormalizationNuisance(
        name="sps_test_norm", eta=0.01, sigma_eta=0.02, table_scope="pi_minus"
    )
    wired_value, _ = evaluate_likelihood(wired, model, V, [by_name])
    assert wired_value == pytest.approx(expected_quad + 0.25, rel=1e-12)

    # A nuisance out of scope shifts nothing but still pays its penalty.
    out_of_scope = NormalizationNuisance(
        name="other_norm", eta=0.01, sigma_eta=0.02, table_scope="k_minus"
    )
    oos_quad, oos_components = evaluate_likelihood(d, model, V, [out_of_scope])
    assert oos_quad == pytest.approx(base_quad + 0.25, rel=1e-12)


def test_component_separation():
    """Exactly the four component keys; chi2_total is the sum of the
    three components to < 1e-12 on a full (non-diagonal) covariance."""
    values = (1.2, 0.7, 0.4)
    model = (1.1, 0.8, 0.35)
    data_cov = (0.0225, 0.006, 0.002, 0.04, 0.003, 0.0324)
    mc_cov = (0.0025, 0.0004, 0.0001, 0.0064, 0.0002, 0.0036)
    d = BinnedObservable(
        species="k_plus",
        observable="f_xsect",
        bin_edges=(0.0, 0.2, 0.4, 0.05, 0.775, 1.5),
        values=values,
        stat_errors=(0.15, 0.20, 0.18),
        cov=data_cov,
        normalization_nuisance=None,
    )
    V = combined_covariance(data_cov, mc_cov)
    nuisances = [
        NormalizationNuisance(
            name="sps_test_norm", eta=0.02, sigma_eta=0.05, table_scope="all"
        )
    ]
    minus2lnL, components = evaluate_likelihood(d, model, V, nuisances)

    assert set(components) == {
        "chi2_data",
        "chi2_mc",
        "chi2_nuisance",
        "chi2_total",
    }
    assert components["chi2_total"] == pytest.approx(
        components["chi2_data"] + components["chi2_mc"] + components["chi2_nuisance"],
        rel=1e-12,
    )
    assert components["chi2_total"] == minus2lnL
    assert components["chi2_nuisance"] == pytest.approx(0.16, rel=1e-12)
    assert all(math.isfinite(v) and v >= 0.0 for v in components.values())


def test_config_nuisances_schema():
    """The bundled config parses with the expected schema; the
    use_data_covariance flag is false and the scopes cover the six
    species of the binned NA49 set."""
    assert CONFIG_PATH.is_file(), f"missing config: {CONFIG_PATH}"
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert set(config) == {"nuisances", "use_data_covariance"}
    assert config["use_data_covariance"] is False
    entries = config["nuisances"]
    assert isinstance(entries, list) and entries
    for entry in entries:
        assert set(entry) == {"name", "sigma_eta", "table_scope"}
        assert isinstance(entry["name"], str) and entry["name"]
        assert isinstance(entry["sigma_eta"], float)
        assert math.isfinite(entry["sigma_eta"]) and entry["sigma_eta"] > 0.0
        assert isinstance(entry["table_scope"], str) and entry["table_scope"]
    names = [entry["name"] for entry in entries]
    assert len(names) == len(set(names))
    assert {entry["table_scope"] for entry in entries} == SIX_SPECIES_SCOPES
