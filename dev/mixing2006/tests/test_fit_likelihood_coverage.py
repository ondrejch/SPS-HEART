"""Coverage extension for the binned likelihood with combined data+MC
covariance and per-table normalization nuisances (Round 1, card 7).

Task card ``TASK-20260911-r1-covariance-nuisances``.
``tests/test_fit_likelihood.py`` holds the six card-named deterministic
tests.  This file extends coverage of the same committed behavior
(``na49_mixing/fit_likelihood.py`` at the card commit) without touching
that file:

- closure beyond the committed single-table case: a three-table sum of
  ``evaluate_likelihood`` equals the sum of the inline M8 formula of
  ``staged_evaluation``'s binned objective (per-bin combined error
  ``sqrt(sigma_data**2 + sigma_mc**2)``) to < 1e-9 relative, per table
  and in the sum;
- degenerate small-sigma limits of the solve path: ``sigma_mc = 1e-8``,
  ``sigma_data = 1e-8``, and both at once.  The components stay finite
  and non-negative, the data/MC split identity holds to < 1e-12, the
  M8 closure holds to < 1e-9, and the documented component limits hold
  (``sigma_MC -> 0``: the data component takes the pure-data quadratic
  form and the MC component is negligible; symmetric for
  ``sigma_data -> 0``).  An exactly zero MC diagonal contributes
  ``chi2_mc = 0.0`` exactly (the "an exactly zero component matrix
  contributes 0" rule).  Committed underflow edge pinned:
  ``stat_errors = 1e-200`` underflows ``sigma**2`` to ``0.0`` so the
  data covariance is the all-zero matrix and the documented zero-matrix
  rule drops the data term silently (``chi2_data = 0.0``);
- off-diagonal data covariance: a 3-bin case with a hand-built
  non-diagonal ``d.cov``; ``chi2_total`` equals
  ``(d-m)^T (V_data + V_mc)^{-1} (d-m)`` against the independent
  ``numpy.linalg.inv``/``solve`` reference to < 1e-9 relative, and the
  component split uses the *full* data covariance (not the
  ``stat_errors`` diagonal) with ``chi2_data + chi2_mc = chi2_total``
  to < 1e-12;
- nuisance scope semantics beyond the committed case: an explicitly
  wired observable receives shifts only from the named nuisance (the
  ``table_scope`` rule is bypassed entirely, so an in-scope but
  differently named nuisance shifts nothing); an unnamed observable
  receives the ``"all"``- and species-scoped nuisances and skips a
  species-mismatch scope; multiple in-scope nuisances add their shifts;
  an out-of-scope nuisance shifts nothing but still pays its
  ``(eta/sigma_eta)**2`` penalty (the scope only controls the shift);
  ``eta = 0`` is nominal (zero shift, zero penalty); ``[]`` behaves as
  ``None``; the committed ``table_scope.strip()`` leniency (``" all "``
  matches, ``"ALL"`` does not: the comparison is case-sensitive) and
  the hand-computed single-bin sign convention
  (residual ``d - m - eta * d`` with the data value as nominal value);
- component dict: exactly the four keys in every configuration, plain
  ``float`` values, ``chi2_total`` the returned value and the sum of
  the three components to < 1e-12, ``chi2_nuisance == 0.0`` for
  ``None`` and for ``[]``; a ``sigma_eta`` of ``0``, negative, ``nan``,
  or ``inf`` raises ``ValueError`` before the likelihood math, even for
  an out-of-scope nuisance (the width guard runs over every nuisance
  passed to the call);
- flattening: ``flatten_covariance``/``unflatten_covariance`` roundtrip
  on a 4x4 symmetric matrix (exact, and the row-major upper-triangle
  order spelled out), the upper-triangle-only behavior on asymmetric
  input, ``n = 1``; ``diagonal_covariance`` produces variances
  ``sigma**2`` (not sigmas) with exact zero off-diagonal entries;
  ``combined_covariance`` on two full 4x4 upper-triangle tuples equals
  the element-wise sum and ``None`` returns the MC tuple unchanged;
- input-validation pins (committed messages): wrong ``m`` /
  ``stat_errors`` lengths, an empty observable, a non-triangular ``V``
  length, non-positive-definite ``V``, ``d.cov``/``V`` shape mismatch,
  non-positive-definite ``d.cov``, non-positive-definite
  ``V_MC = V - d.cov``, and non-positive ``stat_errors`` with
  ``d.cov = None``;
- config schema beyond the committed test: exactly six entries, unique
  names *and* unique scopes covering the six species, every
  ``sigma_eta`` a JSON number of Python type ``float`` (positive and
  finite), ``use_data_covariance`` the JSON boolean ``false`` (Python
  ``False`` of type ``bool``); the directory state pin that card 7
  adds exactly one file under ``config/v2_singlet/`` (every file there
  other than card 1's manifest is the card's config; the manifest's
  presence and content are not asserted here, the freeze test owns
  them);
- the card's "no existing code path reads the config in Round 1"
  constraint: no ``na49_mixing/`` module other than ``fit_likelihood.py``
  mentions the config name, and ``fit_likelihood.py`` mentions it only
  inside its module docstring;
- ``BinnedObservable`` / ``NormalizationNuisance`` frozen-ness:
  mutating any field raises ``dataclasses.FrozenInstanceError``,
  ``dataclasses.replace`` works, the instances compare and hash by
  value, and the field order and defaults match the card's public
  interface (``cov`` and ``normalization_nuisance`` default to
  ``None``; ``NormalizationNuisance`` has no defaults).

Committed behavior noted, not fixed (documented in the module
docstring; judged as resolutions of internal card tensions pinned by
the card's own tests, not defects):

- ``combined_covariance(None, mc_cov)`` returns ``mc_cov`` unchanged.
  The card's "if ``data_cov`` is None, ``V_data`` is the diagonal of
  ``d.stat_errors``" sentence cannot hold inside
  ``combined_covariance`` (it has no access to ``d``); the card's own
  ``test_combined_covariance_diagonal`` pins the return-``mc_cov``
  behavior, and the statistical diagonal enters at the call site
  through ``diagonal_covariance(d.stat_errors)``.
- ``chi2_data`` / ``chi2_mc`` are the additive split of the combined
  quadratic form through
  ``V^{-1} = V^{-1} (V_data + V_MC) V^{-1}``, not the literal
  per-matrix quadratic forms (e.g. ``r^T V_data^{-1} r``): only the
  additive split sums to the combined form, which the card's
  ``chi2_total`` (the returned ``-2 ln L``) requires.

Hermetic: hand-constructed inputs only, no event generation, no
network, no new config files.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from na49_mixing import fit_likelihood
from na49_mixing.fit_likelihood import (
    BinnedObservable,
    NormalizationNuisance,
    combined_covariance,
    diagonal_covariance,
    evaluate_likelihood,
    flatten_covariance,
    unflatten_covariance,
)

NESTED_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = NESTED_ROOT / "config" / "v2_singlet"
CONFIG_PATH = CONFIG_DIR / "likelihood_nuisances.json"
MANIFEST_NAME = "legacy_v1_manifest.json"

# The species keys of the six-species binned NA49 set, matching the
# fit.binned.tables entries of config/staged_excitation.json.
SIX_SPECIES_SCOPES = {
    "pi_minus", "pi_plus", "k_plus", "k_minus", "proton", "antiproton",
}


def _m8_combined_chi2(
    values: tuple[float, ...],
    model: tuple[float, ...],
    stat_errors: tuple[float, ...],
    sigma_mc: tuple[float, ...],
) -> float:
    """Inline M8 reference: the diagonal combined-error chi-square.

    Mirrors ``staged_evaluation``'s binned objective (per bin
    ``sigma_combined = sqrt(table.sigma**2 + sigma_mc**2)``, residual
    ``(m - d) / sigma_combined``, chi2 the dot product of the squared
    residuals).
    """
    sigma_combined = np.sqrt(
        np.asarray(stat_errors, dtype=float) ** 2
        + np.asarray(sigma_mc, dtype=float) ** 2
    )
    residual = (np.asarray(model, dtype=float) - np.asarray(values, dtype=float)) / sigma_combined
    return float(np.dot(residual, residual))


def _obs(
    species: str,
    values: tuple[float, ...],
    stat_errors: tuple[float, ...],
    cov: tuple[float, ...] | None = None,
    nuisance: str | None = None,
) -> BinnedObservable:
    """Hand-constructed observable with placeholder bin edges.

    The likelihood math uses bin counts, not edges, so the edges only
    need to be a well-formed tuple of floats.
    """
    n = len(values)
    return BinnedObservable(
        species=species,
        observable="f_xsect",
        bin_edges=tuple(i / 10.0 for i in range(2 * n)),
        values=tuple(float(v) for v in values),
        stat_errors=tuple(float(s) for s in stat_errors),
        cov=cov,
        normalization_nuisance=nuisance,
    )


# ---------------------------------------------------------------------
# Closure beyond the committed single-table case
# ---------------------------------------------------------------------


def test_multitable_closure_against_inline_m8():
    """Three tables with different bin counts and sigma scales: the sum
    of ``evaluate_likelihood`` equals the sum of the inline M8 formula
    to < 1e-9 relative, per table and in the sum."""
    tables = (
        (
            "pi_minus",
            (1.10, 0.80, 0.50, 0.30),
            (0.20, 0.30, 0.15, 0.25),
            (0.05, 0.08, 0.03, 0.06),
            (1.05, 0.90, 0.55, 0.22),
        ),
        (
            "k_plus",
            (0.62, 0.41, 0.28, 0.19, 0.11),
            (0.12, 0.15, 0.10, 0.08, 0.05),
            (0.02, 0.03, 0.025, 0.015, 0.01),
            (0.58, 0.45, 0.30, 0.16, 0.09),
        ),
        (
            "proton",
            (3.0, 2.0, 1.2),
            (0.5, 0.4, 0.3),
            (0.11, 0.13, 0.17),
            (2.8, 2.1, 1.35),
        ),
    )
    total_likelihood = 0.0
    total_m8 = 0.0
    for species, values, stat, sigma_mc, model in tables:
        d = _obs(species, values, stat)
        V = combined_covariance(
            diagonal_covariance(stat), diagonal_covariance(sigma_mc)
        )
        minus2lnL, components = evaluate_likelihood(d, model, V, None)
        chi2_m8 = _m8_combined_chi2(values, model, stat, sigma_mc)
        assert chi2_m8 > 0.0
        assert abs(minus2lnL - chi2_m8) / chi2_m8 < 1e-9
        assert components["chi2_nuisance"] == 0.0
        total_likelihood += minus2lnL
        total_m8 += chi2_m8
    assert total_m8 > 0.0
    assert abs(total_likelihood - total_m8) / total_m8 < 1e-9


def test_degenerate_small_sigmas_stability():
    """The solve path stays well behaved as one or both sigmas go to
    1e-8: finite, non-negative components, the split identity to
    < 1e-12, closure to < 1e-9, and the documented component limits."""
    values = (1.0, 0.5, 0.25)
    model = (0.9, 0.55, 0.30)
    stat = (0.2, 0.3, 0.25)
    sigma_mc = (0.05, 0.07, 0.04)
    r = np.asarray(values, dtype=float) - np.asarray(model, dtype=float)

    # sigma_mc -> 0: the data component takes the pure-data quadratic
    # form and the MC component is negligible.
    sigma_tiny = (1e-8, 1e-8, 1e-8)
    d = _obs("pi_minus", values, stat)
    V_mc_tiny = combined_covariance(
        diagonal_covariance(stat), diagonal_covariance(sigma_tiny)
    )
    minusA, compA = evaluate_likelihood(d, model, V_mc_tiny, None)
    chi2_data_pure = float(np.sum(r**2 / np.asarray(stat, dtype=float) ** 2))
    m8A = _m8_combined_chi2(values, model, stat, sigma_tiny)
    assert all(math.isfinite(v) and v >= 0.0 for v in compA.values())
    assert compA["chi2_data"] == pytest.approx(chi2_data_pure, rel=1e-9)
    assert 0.0 <= compA["chi2_mc"] < 1e-12
    assert abs(minusA - m8A) / m8A < 1e-9

    # sigma_data -> 0: symmetric statement for the MC component.
    d_tiny_stat = _obs("pi_minus", values, sigma_tiny)
    V_stat_tiny = combined_covariance(
        diagonal_covariance(sigma_tiny), diagonal_covariance(sigma_mc)
    )
    minusB, compB = evaluate_likelihood(d_tiny_stat, model, V_stat_tiny, None)
    chi2_mc_pure = float(np.sum(r**2 / np.asarray(sigma_mc, dtype=float) ** 2))
    m8B = _m8_combined_chi2(values, model, sigma_tiny, sigma_mc)
    assert all(math.isfinite(v) and v >= 0.0 for v in compB.values())
    assert compB["chi2_mc"] == pytest.approx(chi2_mc_pure, rel=1e-9)
    assert 0.0 <= compB["chi2_data"] < 1e-12
    assert abs(minusB - m8B) / m8B < 1e-9

    # Both at 1e-8: a large but finite chi2, the two components equal
    # by symmetry, and the split identity at the large magnitude.
    V_both = combined_covariance(
        diagonal_covariance(sigma_tiny), diagonal_covariance(sigma_tiny)
    )
    minusC, compC = evaluate_likelihood(d_tiny_stat, model, V_both, None)
    m8C = _m8_combined_chi2(values, model, sigma_tiny, sigma_tiny)
    assert m8C > 0.0
    assert math.isfinite(m8C)
    assert abs(minusC - m8C) / m8C < 1e-9
    assert compC["chi2_data"] == pytest.approx(compC["chi2_mc"], rel=1e-12)
    assert compC["chi2_data"] == pytest.approx(m8C / 2.0, rel=1e-9)
    assert compC["chi2_data"] + compC["chi2_mc"] == pytest.approx(
        compC["chi2_total"], rel=1e-12
    )

    # An exactly zero MC diagonal contributes 0.0 exactly (the
    # documented "exactly zero component matrix contributes 0" rule).
    V_zero_mc = combined_covariance(
        diagonal_covariance(stat), diagonal_covariance((0.0, 0.0, 0.0))
    )
    minusZ, compZ = evaluate_likelihood(_obs("k_plus", values, stat), model, V_zero_mc, None)
    m8Z = _m8_combined_chi2(values, model, stat, (0.0, 0.0, 0.0))
    assert compZ["chi2_mc"] == 0.0
    assert compZ["chi2_data"] == pytest.approx(minusZ, rel=1e-12)
    assert abs(minusZ - m8Z) / m8Z < 1e-9

    # Committed underflow edge: 1e-200**2 underflows to 0.0, so the
    # data covariance is the all-zero matrix and the zero-matrix rule
    # drops the data term silently (chi2_data = 0.0).
    sigma_underflow = (1e-200, 1e-200, 1e-200)
    assert all(s * s == 0.0 for s in sigma_underflow)  # underflows to 0
    V_underflow = combined_covariance(
        diagonal_covariance(sigma_underflow),
        diagonal_covariance(sigma_mc),
    )
    minusU, compU = evaluate_likelihood(
        _obs("pi_minus", values, sigma_underflow), model, V_underflow, None
    )
    assert compU["chi2_data"] == 0.0
    assert all(math.isfinite(v) and v >= 0.0 for v in compU.values())
    assert compU["chi2_total"] == pytest.approx(compU["chi2_mc"], rel=1e-12)


# ---------------------------------------------------------------------
# Off-diagonal data covariance
# ---------------------------------------------------------------------


def test_off_diagonal_data_covariance_reference():
    """A 3-bin case with a non-diagonal d.cov: chi2_total equals the
    independent (d-m)^T (V_data + V_mc)^{-1} (d-m) reference to
    < 1e-9 relative, and the split uses the full d.cov (not the
    stat_errors diagonal) with chi2_data + chi2_mc = chi2_total to
    < 1e-12."""
    values = (1.2, 0.7, 0.4)
    model = (1.1, 0.8, 0.35)
    # Hand-built positive-definite data covariance (leading minors
    # 0.04, 0.0035, 0.00020671, all positive).
    v_data_mat = np.array(
        [
            [0.04, 0.01, -0.005],
            [0.01, 0.09, 0.004],
            [-0.005, 0.004, 0.06],
        ]
    )
    assert np.all(np.linalg.eigvalsh(v_data_mat) > 0.0)
    data_cov = (0.04, 0.01, -0.005, 0.09, 0.004, 0.06)
    # Deliberately different from stat_errors**2: the split must use
    # the full d.cov, not the statistical diagonal.
    d = _obs("k_plus", values, (0.15, 0.2, 0.18), cov=data_cov)
    sigma_mc = (0.05, 0.07, 0.04)
    mc_cov = diagonal_covariance(sigma_mc)
    V = combined_covariance(data_cov, mc_cov)
    minus2lnL, components = evaluate_likelihood(d, model, V, None)

    r = np.asarray(values, dtype=float) - np.asarray(model, dtype=float)
    v_mc_mat = np.diag(np.asarray(sigma_mc, dtype=float) ** 2)
    v_full = v_data_mat + v_mc_mat
    chi2_ref = float(r @ np.linalg.inv(v_full) @ r)
    assert chi2_ref > 0.0
    assert minus2lnL == pytest.approx(chi2_ref, rel=1e-9)

    # The additive split against the independent V^{-1} reference.
    w_ref = np.linalg.solve(v_full, r)
    chi2_data_ref = float(w_ref @ (v_data_mat @ w_ref))
    chi2_mc_ref = float(w_ref @ (v_mc_mat @ w_ref))
    assert components["chi2_data"] == pytest.approx(chi2_data_ref, rel=1e-9)
    assert components["chi2_mc"] == pytest.approx(chi2_mc_ref, rel=1e-9)
    assert components["chi2_nuisance"] == 0.0
    assert components["chi2_data"] + components["chi2_mc"] == pytest.approx(
        components["chi2_total"], rel=1e-12
    )


# ---------------------------------------------------------------------
# Nuisance scope semantics
# ---------------------------------------------------------------------


def test_nuisance_explicit_wiring_ignores_scope():
    """An observable that names a nuisance receives shifts only from
    that name; in-scope but differently named nuisances shift nothing
    (the scope rule is bypassed by the explicit wiring), while their
    penalties still apply."""
    values = (1.0, 0.5, 0.25)
    model = (0.9, 0.55, 0.30)
    stat = (0.2, 0.3, 0.25)
    sigma_mc = (0.05, 0.08, 0.06)
    V = combined_covariance(
        diagonal_covariance(stat), diagonal_covariance(sigma_mc)
    )
    v_mat = unflatten_covariance(V)
    r = np.asarray(values, dtype=float) - np.asarray(model, dtype=float)

    A = NormalizationNuisance("sps_piminus_norm", 0.01, 0.02, "all")
    B = NormalizationNuisance("other_norm", 0.03, 0.03, "pi_minus")
    C = NormalizationNuisance("third_norm", 0.02, 0.05, "all")
    penalty = (0.01 / 0.02) ** 2 + (0.03 / 0.03) ** 2 + (0.02 / 0.05) ** 2

    # Wired: only A's shift applies, though B and C are in scope by
    # their table_scope.
    wired = _obs("pi_minus", values, stat, nuisance="sps_piminus_norm")
    minus, components = evaluate_likelihood(wired, model, V, [A, B, C])
    r_shift = r - 0.01 * np.asarray(values, dtype=float)
    chi2_ref = float(r_shift @ np.linalg.solve(v_mat, r_shift))
    assert components["chi2_nuisance"] == pytest.approx(penalty, rel=1e-12)
    assert minus == pytest.approx(chi2_ref + penalty, rel=1e-12)

    # Unnamed: B and C do shift, so the two evaluations genuinely
    # differ in the quadratic part, not just by chance.
    unnamed = _obs("pi_minus", values, stat)
    minus_un, comp_un = evaluate_likelihood(unnamed, model, V, [A, B, C])
    r_shift_un = r - (0.01 + 0.03 + 0.02) * np.asarray(values, dtype=float)
    chi2_ref_un = float(r_shift_un @ np.linalg.solve(v_mat, r_shift_un))
    assert comp_un["chi2_nuisance"] == pytest.approx(penalty, rel=1e-12)
    assert minus_un == pytest.approx(chi2_ref_un + penalty, rel=1e-12)
    assert abs(minus_un - minus) > 1e-6


def test_nuisance_scope_all_species_match_and_penalty_sum():
    """An unnamed observable receives the 'all'- and species-scoped
    nuisances and skips the species-mismatch scope; multiple in-scope
    nuisances add their shifts; an out-of-scope nuisance contributes
    only its penalty."""
    values = (1.0, 0.5, 0.25)
    model = (0.9, 0.55, 0.30)
    stat = (0.2, 0.3, 0.25)
    sigma_mc = (0.05, 0.08, 0.06)
    V = combined_covariance(
        diagonal_covariance(stat), diagonal_covariance(sigma_mc)
    )
    v_mat = unflatten_covariance(V)
    d = _obs("k_plus", values, stat)

    X = NormalizationNuisance("x_norm", 0.01, 0.02, "all")
    Y = NormalizationNuisance("y_norm", 0.02, 0.03, "k_plus")
    Z = NormalizationNuisance("z_norm", 0.04, 0.05, "k_minus")
    minus, components = evaluate_likelihood(d, model, V, [X, Y, Z])
    r = np.asarray(values, dtype=float) - np.asarray(model, dtype=float)
    # X and Y in scope; Z (k_minus) out of scope.  Shifts add.
    r_shift = r - (0.01 + 0.02) * np.asarray(values, dtype=float)
    chi2_ref = float(r_shift @ np.linalg.solve(v_mat, r_shift))
    penalty = (0.01 / 0.02) ** 2 + (0.02 / 0.03) ** 2 + (0.04 / 0.05) ** 2
    assert components["chi2_nuisance"] == pytest.approx(penalty, rel=1e-12)
    assert minus == pytest.approx(chi2_ref + penalty, rel=1e-12)

    # Z's out-of-scope presence is visible only through the penalty:
    # dropping Z changes the total by exactly its penalty term.
    minus_noz, _ = evaluate_likelihood(d, model, V, [X, Y])
    assert minus_noz == pytest.approx(minus - (0.04 / 0.05) ** 2, rel=1e-12)


def test_nuisance_zero_eta_and_empty_list_nominal():
    """eta = 0 gives shift 0 and penalty 0 (identical to the
    no-nuisance evaluation); an empty list behaves as None."""
    values = (1.0, 0.5, 0.25)
    model = (0.9, 0.55, 0.30)
    stat = (0.2, 0.3, 0.25)
    sigma_mc = (0.05, 0.08, 0.06)
    d = _obs("pi_minus", values, stat)
    V = combined_covariance(
        diagonal_covariance(stat), diagonal_covariance(sigma_mc)
    )
    base, base_comp = evaluate_likelihood(d, model, V, None)
    assert base_comp["chi2_nuisance"] == 0.0

    zero_eta = NormalizationNuisance("z_norm", 0.0, 0.02, "all")
    minus, comp = evaluate_likelihood(d, model, V, [zero_eta])
    assert minus == base
    assert comp["chi2_nuisance"] == 0.0

    minus_e, comp_e = evaluate_likelihood(d, model, V, [])
    assert minus_e == base
    assert comp_e["chi2_nuisance"] == 0.0


def test_nuisance_scope_strip_leniency_and_case_sensitivity():
    """The committed table_scope.strip() leniency: ' all ' matches
    after stripping; 'ALL' does not (the comparison is case-
    sensitive), so the shift appears only in the first case."""
    values = (1.0,)
    model = (0.9,)
    stat = (0.2,)
    sigma_mc = (0.05,)
    d = _obs("pi_minus", values, stat)
    V = combined_covariance(
        diagonal_covariance(stat), diagonal_covariance(sigma_mc)
    )
    v_mat = unflatten_covariance(V)
    base, _ = evaluate_likelihood(d, model, V, None)
    penalty = 0.25  # (0.01 / 0.02) ** 2

    ws = NormalizationNuisance("ws_norm", 0.01, 0.02, " all ")
    minus_ws, _ = evaluate_likelihood(d, model, V, [ws])
    r_shift = np.asarray([1.0 - 0.9 - 0.01], dtype=float)
    chi2_ref = float(r_shift @ np.linalg.solve(v_mat, r_shift))
    assert minus_ws == pytest.approx(chi2_ref + penalty, rel=1e-12)

    cs = NormalizationNuisance("cs_norm", 0.01, 0.02, "ALL")
    minus_cs, _ = evaluate_likelihood(d, model, V, [cs])
    assert minus_cs == pytest.approx(base + penalty, rel=1e-12)


def test_nuisance_sign_convention_hand_computed():
    """Single-bin hand arithmetic pins the sign convention: the
    residual is d - m - eta * d with the data value as nominal value
    (positive eta moves the data away from the model for positive
    residuals).  chi2_quad = 0.8**2 / 0.0125 = 51.2, penalty
    (0.1/0.02)**2 = 25."""
    d = _obs("pi_minus", (2.0,), (0.1,))
    V = combined_covariance(
        diagonal_covariance((0.1,)), diagonal_covariance((0.05,))
    )
    nu = NormalizationNuisance("sps_piminus_norm", 0.1, 0.02, "all")
    minus, comp = evaluate_likelihood(d, (1.0,), V, [nu])
    assert comp["chi2_nuisance"] == pytest.approx(25.0, rel=1e-12)
    assert minus == pytest.approx(51.2 + 25.0, rel=1e-12)
    assert comp["chi2_total"] == minus


# ---------------------------------------------------------------------
# Component dict and nuisance-width validation
# ---------------------------------------------------------------------


def test_component_dict_exact_keys_types_and_sum():
    """Exactly the four keys in every configuration, plain float
    values, chi2_total the returned value and the sum of the three
    components to < 1e-12."""
    d = _obs("k_plus", (1.2, 0.7, 0.4), (0.15, 0.2, 0.18))
    V = combined_covariance(
        diagonal_covariance((0.15, 0.2, 0.18)),
        diagonal_covariance((0.05, 0.07, 0.04)),
    )
    for nuis in (
        None,
        [],
        [NormalizationNuisance("n_norm", 0.02, 0.05, "all")],
        [
            NormalizationNuisance("n1_norm", 0.01, 0.02, "k_plus"),
            NormalizationNuisance("n2_norm", 0.03, 0.06, "other"),
        ],
    ):
        minus, comp = evaluate_likelihood(d, (1.1, 0.8, 0.35), V, nuis)
        assert set(comp) == {"chi2_data", "chi2_mc", "chi2_nuisance", "chi2_total"}
        assert all(type(v) is float for v in comp.values())
        assert comp["chi2_total"] == minus
        assert comp["chi2_total"] == pytest.approx(
            comp["chi2_data"] + comp["chi2_mc"] + comp["chi2_nuisance"], rel=1e-12
        )


@pytest.mark.parametrize("sigma_eta", [0.0, -0.01, float("inf"), float("nan")])
def test_sigma_eta_must_be_positive_finite(sigma_eta):
    """A non-positive, non-finite width raises before the likelihood
    math (no division by zero or nan propagation)."""
    d = _obs("pi_minus", (1.0,), (0.2,))
    V = combined_covariance(
        diagonal_covariance((0.2,)), diagonal_covariance((0.05,))
    )
    nu = NormalizationNuisance("bad_norm", 0.01, sigma_eta, "all")
    with pytest.raises(ValueError, match="positive sigma_eta"):
        evaluate_likelihood(d, (0.9,), V, [nu])


def test_sigma_eta_validation_precedes_scope():
    """The width guard runs over every nuisance passed to the call,
    even an out-of-scope one."""
    d = _obs("pi_minus", (1.0,), (0.2,))
    V = combined_covariance(
        diagonal_covariance((0.2,)), diagonal_covariance((0.05,))
    )
    nu = NormalizationNuisance("bad_norm", 0.0, 0.0, "k_minus")
    with pytest.raises(ValueError, match="positive sigma_eta"):
        evaluate_likelihood(d, (0.9,), V, [nu])


# ---------------------------------------------------------------------
# Input-validation pins (committed messages)
# ---------------------------------------------------------------------


def test_evaluate_input_validation():
    values = (1.0, 0.5, 0.25)
    model = (0.9, 0.55, 0.30)
    stat = (0.2, 0.3, 0.25)
    mc = (0.05, 0.08, 0.06)
    V = combined_covariance(diagonal_covariance(stat), diagonal_covariance(mc))

    with pytest.raises(ValueError, match="at least one bin"):
        evaluate_likelihood(_obs("pi_minus", (), ()), (), (0.01,))
    with pytest.raises(ValueError, match="model vector"):
        evaluate_likelihood(_obs("pi_minus", values, stat), (0.9, 0.55), V)
    with pytest.raises(ValueError, match="stat_errors"):
        evaluate_likelihood(_obs("pi_minus", values, (0.2, 0.3)), model, V)
    with pytest.raises(ValueError, match=r"n\(n\+1\)/2"):
        evaluate_likelihood(
            _obs("pi_minus", values, stat), model, (0.01, 0.0, 0.02, 0.0, 0.03)
        )
    with pytest.raises(ValueError, match="not positive definite"):
        evaluate_likelihood(
            _obs("pi_minus", values, stat),
            model,
            (-0.01, 0.0, 0.02, 0.04, 0.0, 0.0625),
        )
    # d.cov / V shape mismatch.
    with pytest.raises(ValueError, match="equal shape"):
        evaluate_likelihood(
            _obs("pi_minus", values, stat, cov=(0.04, 0.01, 0.0)), model, V
        )
    # Non-positive-definite d.cov (V itself stays positive definite:
    # the diagonal sum is element-wise positive).
    bad_data = (-0.001, 0.0, 0.0, 0.09, 0.0, 0.06)
    V_data_sum = combined_covariance(bad_data, diagonal_covariance(mc))
    with pytest.raises(ValueError, match="data covariance"):
        evaluate_likelihood(
            _obs("pi_minus", values, stat, cov=bad_data), model, V_data_sum
        )
    # V_MC = V - d.cov not positive definite (V and d.cov both are):
    # V is the diagonal d.cov with the last variance shrunk by 1e-4.
    cov = diagonal_covariance((0.2, 0.3, 0.25))
    v_small = cov[:5] + (cov[5] - 1e-4,)
    with pytest.raises(ValueError, match="MC covariance"):
        evaluate_likelihood(_obs("pi_minus", values, stat, cov=cov), model, v_small)
    # Non-positive stat_errors with d.cov = None.
    V_zero_stat = combined_covariance(
        diagonal_covariance((0.0, 0.2)), diagonal_covariance((0.05, 0.08))
    )
    with pytest.raises(ValueError, match="positive when the observable"):
        evaluate_likelihood(
            _obs("pi_minus", (1.0, 0.5), (0.0, 0.2)), (0.9, 0.55), V_zero_stat
        )


# ---------------------------------------------------------------------
# Flattening conventions
# ---------------------------------------------------------------------


def test_flatten_unflatten_roundtrip_4x4():
    """The row-major upper-triangle order spelled out, and an exact
    roundtrip on a symmetric 4x4 matrix."""
    m = np.array(
        [
            [1.0, 0.5, 0.25, 0.125],
            [0.5, 2.0, 0.375, 0.0625],
            [0.25, 0.375, 3.0, 0.4375],
            [0.125, 0.0625, 0.4375, 4.0],
        ]
    )
    flat = flatten_covariance(m)
    assert len(flat) == 10
    assert flat == (1.0, 0.5, 0.25, 0.125, 2.0, 0.375, 0.0625, 3.0, 0.4375, 4.0)
    back = unflatten_covariance(flat)
    assert np.array_equal(back, m)
    assert (back == back.T).all()


def test_flatten_unflatten_edge_cases():
    """n = 1, the upper-triangle-only behavior on asymmetric input,
    and the shape/length rejections."""
    assert flatten_covariance(np.array([[7.0]])) == (7.0,)
    assert unflatten_covariance((7.0,))[0, 0] == 7.0

    # The lower triangle of an asymmetric input is ignored on flatten;
    # unflatten symmetrizes the upper triangle.
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert flatten_covariance(a) == (1.0, 2.0, 4.0)
    back = unflatten_covariance(flatten_covariance(a))
    assert (back == np.array([[1.0, 2.0], [2.0, 4.0]])).all()

    with pytest.raises(ValueError, match="square"):
        flatten_covariance(np.ones((2, 3)))
    with pytest.raises(ValueError, match="square"):
        flatten_covariance(np.ones(3))
    with pytest.raises(ValueError, match=r"n\(n\+1\)/2"):
        unflatten_covariance((1.0, 2.0, 3.0, 4.0, 5.0))
    with pytest.raises(ValueError, match="at least 1 entry"):
        unflatten_covariance(())


def test_diagonal_covariance_is_variances_not_sigmas():
    """The diagonal entries are sigma**2 (exact float products), the
    off-diagonals are exactly 0.0; the sigmas themselves are not
    carried."""
    sigmas = (0.05, 0.2, 1.5)
    flat = diagonal_covariance(sigmas)
    assert flat == (0.05 * 0.05, 0.0, 0.0, 0.2 * 0.2, 0.0, 1.5 * 1.5)
    assert flat[0] != 0.05 and flat[3] != 0.2 and flat[5] != 1.5
    with pytest.raises(ValueError, match="at least 1 entry"):
        diagonal_covariance(())


def test_combined_covariance_full_triangle_and_none():
    """A full 4x4 upper-triangle sum, the None return-mc_cov behavior,
    and the shape rejections."""
    data = (
        0.04, 0.01, 0.02, 0.09, 0.03, 0.06, 0.05, 0.07, 0.002, 0.11,
    )
    mc = (
        0.0025, 0.001, 0.0005, 0.0049, 0.0007, 0.001, 0.0009, 0.0011, 0.0003, 0.0036,
    )
    result = combined_covariance(data, mc)
    assert result == tuple(a + b for a, b in zip(data, mc))
    assert combined_covariance(None, mc) == mc
    with pytest.raises(ValueError, match="equal shape"):
        combined_covariance(data[:6], mc)
    with pytest.raises(ValueError, match=r"n\(n\+1\)/2"):
        combined_covariance(None, data[:5])


# ---------------------------------------------------------------------
# Config schema beyond the committed test
# ---------------------------------------------------------------------


def test_config_nuisances_schema_extended():
    """Exactly six entries with unique names and unique scopes covering
    the six species; every sigma_eta a JSON number (Python type
    float), positive and finite; use_data_covariance the JSON boolean
    false."""
    assert CONFIG_PATH.is_file(), f"missing config: {CONFIG_PATH}"
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    entries = config["nuisances"]
    assert isinstance(entries, list) and len(entries) == 6
    names = [entry["name"] for entry in entries]
    scopes = [entry["table_scope"] for entry in entries]
    assert len(set(names)) == 6
    assert len(set(scopes)) == 6
    assert set(scopes) == SIX_SPECIES_SCOPES
    for entry in entries:
        assert set(entry) == {"name", "sigma_eta", "table_scope"}
        assert isinstance(entry["name"], str) and entry["name"].startswith("sps_")
        sigma = entry["sigma_eta"]
        assert type(sigma) is float  # a JSON number, not a string/int
        assert math.isfinite(sigma) and sigma > 0.0
        assert isinstance(entry["table_scope"], str) and entry["table_scope"]
    # The flag is the JSON literal false: Python False of type bool
    # (not the string "false", not 0).
    assert config["use_data_covariance"] is False
    assert type(config["use_data_covariance"]) is bool


def test_config_is_only_card_addition_under_v2_singlet():
    """Card 7 adds exactly one file under config/v2_singlet/.  Every
    file there other than card 1's manifest is either the card's config
    or a later sanctioned plan section 4.4 config file.  The manifest is
    excluded on purpose: this test pins the card's "new files only"
    clause and does not assert the manifest's presence or content (the
    freeze test owns those)."""
    # regular files only: an interpreter that writes bytecode leaves a
    # __pycache__ directory here, which is not a config addition
    files = {
        name for name in os.listdir(CONFIG_DIR)
        if os.path.isfile(os.path.join(str(CONFIG_DIR), name))
    }
    # Round 2 card 2 (TASK-20260913-r2-provenance-catalog-import) adds the
    # plan section 4.4 resonance catalog; later Round 2 cards add further
    # plan section 4.4 config files, each in its own card.  Round 3
    # card 4 (TASK-20260917-r3-baryonic-continuum-spectral-family) adds
    # the plan section 4.4 continuum_resonances.json (the baryonic
    # continuum spectral families of plan section 9.1).
    # The 2026-09-28 review follow-up adds three provenance files of its
    # own cards: the external diffractive cross sections (the R51b/R51c
    # anchors), the seed ledger (consumed and reserved seed blocks), and
    # the NA49 total yields (the total-yield readout's reference).
    # TASK-20261008-r84-isoscalar-dipion (2026-10-08) adds the PDG 2025
    # decay modes of the kernel option KernelConfig.isoscalar_dipion.
    # TASK-20261012-r92-light-scalars (2026-10-11) adds the PDG 2025
    # records of the kernel option KernelConfig.light_scalar_mesons.
    # TASK-20261015-r98-completion (2026-10-15) adds the PDG 2025 records
    # of KernelConfig.pdg_hyperons and KernelConfig.eta_prime, and the
    # S = -2 baryons of KernelConfig.xi_baryons (2026-10-16), and the PDG
    # 2024 meson tables of KernelConfig.pdg_meson_modes (2026-10-16).
    # TASK-20261021-completions (2026-10-21) adds the PDG 2024 axial
    # isoscalars of KernelConfig.axial_isoscalars and the PDG 2025 N and
    # Delta resonances of KernelConfig.pdg_nucleon_resonances, and the
    # baryon table corrections of KernelConfig.baryon_table_corrections.
    assert files - {MANIFEST_NAME} == {
        "likelihood_nuisances.json",
        "resonance_catalog.json",
        "continuum_resonances.json",
        "external_cross_sections.json",
        "seed_ledger.json",
        "na49_pp_total_yields.json",
        "isoscalar_dipion_pdg2025.json",
        "light_scalar_mesons_pdg2025.json",
        "pdg_hyperons_pdg2025.json",
        "eta_prime_pdg2025.json",
        "xi_baryons_pdg2025.json",
        "meson_modes_pdg2024.json",
        "axial_isoscalars_pdg2024.json",
        "pdg_nucleon_resonances_pdg2025.json",
        "baryon_table_corrections_pdg2025.json",
        # TASK-20261028-excited-kaons (2026-10-28) adds the PDG 2024 records of
        # the kernel option KernelConfig.excited_kaons.
        "excited_kaons_pdg2024.json",
        # 2026-11-04 adds the PDG 2025 Omega- record of KernelConfig.omega_baryon and the counted decays of the
        # observables (na49_mixing.counted_decays, the run spec's fit.counted_decays).
        "omega_baryon_pdg2025.json",
        "counted_decays_pdg2025.json",
    }


def test_no_existing_code_path_reads_the_config():
    """The card forbids any existing code path from reading the v2
    config in Round 1.  A docstring mention in the documenting module
    is not a read; anything else is: no na49_mixing module other than
    fit_likelihood.py mentions the config name, and fit_likelihood.py
    mentions it only inside its module docstring."""
    hits = []
    for path in sorted((NESTED_ROOT / "na49_mixing").rglob("*.py")):
        if "likelihood_nuisances" in path.read_text(encoding="utf-8"):
            hits.append(path)
    assert hits == [NESTED_ROOT / "na49_mixing" / "fit_likelihood.py"]
    src = (NESTED_ROOT / "na49_mixing" / "fit_likelihood.py").read_text(
        encoding="utf-8"
    )
    doc = fit_likelihood.__doc__ or ""
    assert src.count("likelihood_nuisances") == doc.count("likelihood_nuisances") >= 1


# ---------------------------------------------------------------------
# Dataclass frozen-ness, field order, defaults
# ---------------------------------------------------------------------


def test_binned_observable_frozen_fields_and_defaults():
    """Mutation of any field raises FrozenInstanceError; the field
    order and defaults match the card's public interface."""
    o = BinnedObservable("pi-", "f_xsect", (0.0, 0.2, 0.4), (1.0, 0.5), (0.1, 0.1))
    # cov and normalization_nuisance default to None.
    assert o.cov is None and o.normalization_nuisance is None
    fields = dataclasses.fields(BinnedObservable)
    assert tuple(f.name for f in fields) == (
        "species",
        "observable",
        "bin_edges",
        "values",
        "stat_errors",
        "cov",
        "normalization_nuisance",
    )
    assert all(f.default is dataclasses.MISSING for f in fields[:5])
    assert fields[5].default is None and fields[6].default is None
    for field in fields:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(o, field.name, field.name)
    # Frozen instances still compare and hash by value, and
    # dataclasses.replace builds the modified copy.
    o2 = dataclasses.replace(o, cov=(0.01, 0.0, 0.0, 0.02))
    assert o2.cov == (0.01, 0.0, 0.0, 0.02)
    assert o2 != o
    same = BinnedObservable("pi-", "f_xsect", (0.0, 0.2, 0.4), (1.0, 0.5), (0.1, 0.1))
    assert o == same and hash(o) == hash(same)


def test_normalization_nuisance_frozen_fields_and_defaults():
    """No defaults (the card's interface lists none); mutation of any
    field raises FrozenInstanceError; value comparison and hashing."""
    n = NormalizationNuisance("sps_piminus_norm", 0.01, 0.02, "pi_minus")
    fields = dataclasses.fields(NormalizationNuisance)
    assert tuple(f.name for f in fields) == (
        "name",
        "eta",
        "sigma_eta",
        "table_scope",
    )
    assert all(f.default is dataclasses.MISSING for f in fields)
    for field in fields:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(n, field.name, "nope")
    assert NormalizationNuisance("sps_piminus_norm", 0.01, 0.02, "pi_minus") == n
    assert hash(n) == hash(
        NormalizationNuisance("sps_piminus_norm", 0.01, 0.02, "pi_minus")
    )
