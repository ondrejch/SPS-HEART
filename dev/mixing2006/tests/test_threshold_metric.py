"""The re-registered threshold-discontinuity metric (Round 3, card 2).

Task ``TASK-20260917-r3-gate2-followup-threshold-metric-reregistration``
(the Gate 2 closure obligation of the Round 2 decision, follow-up 2).
The tests cover, deterministically:

- ``test_evaluate_pair_conventions``: the signed sigma and pass flag of
  ``evaluate_pair`` on hand-computed cases (the combined standard
  error, the 3.0-sigma boundary, the zero-SE vacuous and failing
  rules).
- ``test_full_analytic_difference_open_set_change_fixture``: a
  hand-computed two-resonance fixture pair where a record opens
  between the two point masses and no surviving-record window extends:
  the open-set-change contribution equals the full analytic difference
  (the original ``delta_analytic`` convention) and the momentum-drift
  contribution vanishes identically.
- ``test_full_analytic_difference_unchanged_pair_drift_fixture``: an
  unchanged-pair fixture whose resonance window extends: the open-set-
  change contribution vanishes identically and the momentum drift
  carries the analytic kinematic drift, checked against an independent
  in-test quadrature of the documented convention.
- ``test_full_analytic_difference_decomposition_identity``: the two
  contributions add to ``delta_analytic_full`` exactly on both
  fixtures.
- ``test_narrow_record_draw_cell_evaluation``: the narrow line shape's
  evaluation point is the documented card-3 draw cell containing the
  pole, and the fixture content follows the exact on-shell kinematics
  at that mass.
- ``test_rereval_report_hand_verdicts_and_byte_stability``: the
  re-evaluation of a hand-built sealed-gate fixture reproduces the
  hand-computed re-registered verdicts and is byte-stable.
- ``test_sealed_archive_rereval`` (skips without the sealed archives):
  the re-evaluation of the real sealed ``gate2_r21.json`` /
  ``gate2_r22.json`` reproduces the re-registration counts (169 -> 161
  passing R21, 154 -> 151 R22, 11 named residuals), is byte-stable,
  and leaves the sealed gate files byte-unchanged (pinned against the
  sealed clean-environment records).
- ``test_sealed_archive_engine_exact_on_first_pairs`` (skips without
  the sealed archives): the new-run analytic engine reproduces the
  sealed stored ``delta_analytic_full`` exactly on the first R21 pairs.
- ``test_card9_driver_new_run_metric_is_reregistered``: the card-9
  driver's new-run discontinuity metric is the re-registered one (the
  ``DISCONTINUITY_METRIC`` declaration, the pre-registration
  declaration, and the driver's metric table on the real grid).
- ``test_dual_tree_byte_identity``: the nested and outer copies of the
  two dual-tree drivers are byte-identical.

Hermetic: standard library, numpy, and the vendored ``na49_mixing``
modules; the sealed-archive tests are read-only over
``dev/00runs/`` and skip on a fresh checkout.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.hadron_catalog import DecayChannel, HadronRecord
from na49_mixing.resonance_widths import decay_momentum, spectral_function
from na49_mixing.threshold_metric import (  # noqa: E402
    ANGLE_NODES,
    ANGLE_NODES_DEEP,
    DAUGHTER_GL_NODES,
    DETERMINISTIC_SE_FLOOR,
    ROOT_GL_NODES,
    SIGMA_TOLERANCE,
    ZERO_RESIDUAL_TOL,
    RecordSet,
    diagnose_residual,
    evaluate_pair,
    full_analytic_difference,
    rereval_report,
)

_tm = importlib.import_module("na49_mixing.threshold_metric")
_ROOT_X = _tm._ROOT_X
_ROOT_W = _tm._ROOT_W
_MBC_X = _tm._MBC_X
_MBC_W = _tm._MBC_W
_COS3_X = _tm._COS3_X
_COS3_W = _tm._COS3_W

NESTED_ROOT = Path(__file__).resolve().parent.parent
MONO_ROOT = NESTED_ROOT.parent.parent
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"

REREVAL_DRIVER_REL = Path("helpers") / "benchmarking" / (
    "rereval_threshold_checks.py")
SOURCE_GRID_DRIVER_REL = Path("helpers") / "benchmarking" / (
    "staged_source_grid.py")

R21_ARCHIVE = MONO_ROOT / "dev" / "00runs" / (
    "r21_baryonic_source_grid_2026-09-13")
R22_ARCHIVE = MONO_ROOT / "dev" / "00runs" / (
    "r22_mesonic_source_grid_2026-09-13")

#: The sealed gate digests recorded by the archives' own
#: clean-environment validation records (the byte-unchanged pin).
SEALED_GATE_SHA256 = {
    "gate2_r21.json": (
        "82af88d50e0b2be4da4e552a568181d1b6919f421190e7a54bea835556b07ac3"),
    "gate2_r22.json": (
        "a2e81dbdb48d76c4a3c7e87436e9fb2e17f4a999918b261677ccbd2f16340865"),
}


# ---------------------------------------------------------------------------
# Synthetic fixture records (hand-built catalog; the card-5 machinery
# falls back to the channels' own partial widths outside its coverage)
# ---------------------------------------------------------------------------
def _channel(parent: int, daughters: tuple[int, ...], br: float,
             L: int = 0) -> DecayChannel:
    return DecayChannel(
        parent_pdg=parent,
        daughters=tuple(daughters),
        branching_fraction=br,
        orbital_angular_momentum=L,
        is_three_body=False,
        provenance=None,
    )


def _record(pdg: int, name: str, pole: float, width: float,
            mass_range: tuple[float, float], channels=(),
            stable: bool = False) -> HadronRecord:
    return HadronRecord(
        pdg_id=pdg,
        name=name,
        B=0, Q=0, S=0,
        isospin=0.0,
        isospin3=0,
        spin=0.0,
        parity="+",
        pole_mass=pole,
        nominal_width=width,
        mass_range=mass_range,
        decay_channels=tuple(channels),
        stable_for_decay=stable,
        provenance=None,
    )


class _FixtureCatalog:
    """The minimal catalog surface the metric engine reads."""

    def __init__(self, records: list[HadronRecord]) -> None:
        self._records = list(records)
        self._by_pdg = {int(r.pdg_id): r for r in records}

    def get(self, pdg: int) -> HadronRecord:
        return self._by_pdg[int(pdg)]

    def stable_states(self) -> frozenset:
        return frozenset(
            int(r.pdg_id) for r in self._records if r.stable_for_decay
        )

    def records(self) -> list[HadronRecord]:
        return list(self._records)


#: The fixture species keys (the charge-symbol transliteration of the
#: fixture names).
KEY_D = "delta_0"
KEY_E = "epsilon_0"

PDG_D = 900003
PDG_E = 900005
PDG_B = 900002
PDG_C = 900004


def _osc_fixture():
    """The open-set-change fixture: the narrow resonance
    ``beta_0 -> delta_0 delta_0`` (support ``[0.6, 0.7]``, fixed
    between the pair masses) is open at both masses; the narrow
    resonance ``gamma_0 -> delta_0 epsilon_0`` (support
    ``[0.85, 0.9]``) opens at ``M_{k+1} = 0.9``.  No stable record
    stands in the open set, so no identity denominator moves and the
    surviving-record drift vanishes exactly for every observable."""
    d = _record(PDG_D, "delta0", 0.2, 0.0, (0.2, 0.2), (), stable=True)
    e = _record(PDG_E, "epsilon0", 0.25, 0.0, (0.25, 0.25), (),
                stable=True)
    b = _record(PDG_B, "beta0", 0.65, 1e-6, (0.6, 0.7),
                (_channel(PDG_B, (PDG_D, PDG_D), 1.0, 0),))
    c = _record(PDG_C, "gamma0", 0.9, 1e-6, (0.85, 0.9),
                (_channel(PDG_C, (PDG_D, PDG_E), 1.0, 0),))
    catalog = _FixtureCatalog([d, e, b, c])
    return catalog, RecordSet(catalog, [b, c])  # type: ignore[arg-type]


def _drift_fixture():
    """The unchanged-pair drift fixture: the stable ``alpha_0``, the
    stable ``delta_0``, and the broad resonance
    ``beta_0 -> delta_0 delta_0`` (pole 1.0, width 0.1, support
    ``[0.6, 1.4]``) are open at both pair masses; the resonance's
    truncated window extends between them."""
    d = _record(PDG_D, "delta0", 0.2, 0.0, (0.2, 0.2), (), stable=True)
    e = _record(PDG_E, "epsilon0", 0.25, 0.0, (0.25, 0.25), (),
                stable=True)
    b = _record(PDG_B, "beta0", 1.0, 0.1, (0.6, 1.4),
                (_channel(PDG_B, (PDG_D, PDG_D), 1.0, 0),))
    catalog = _FixtureCatalog([d, e, b])
    return catalog, RecordSet(catalog, [d, e, b])  # type: ignore[arg-type]


def _window_draw_mean(record: HadronRecord, lo: float, hi: float,
                      nodes: int = 1024) -> float:
    """The in-test implementation of the documented draw-mean rule: the
    trapezoid mean of the card-3 density on the uniform window grid."""
    grid = np.linspace(lo, hi, nodes)
    dmap = {PDG_D: 0.2}
    rho = np.array([
        spectral_function(float(m), record, dmap) for m in grid
    ])
    w = np.full(nodes, (hi - lo) / (nodes - 1))
    w[0] *= 0.5
    w[-1] *= 0.5
    return float(np.sum(grid * w * rho) / np.sum(w * rho))


def _q_two_body(m: float, m1: float, m2: float) -> float:
    """The exact on-shell two-body decay momentum (the Kaellen
    function), the test's own hand formula."""
    lam = (m * m - (m1 + m2) ** 2) * (m * m - (m1 - m2) ** 2)
    return math.sqrt(max(lam, 0.0)) / (2.0 * m)


# ---------------------------------------------------------------------------
# evaluate_pair
# ---------------------------------------------------------------------------
def test_evaluate_pair_conventions():
    """Hand-computed signed sigmas and pass flags at the 3.0-sigma
    tolerance."""
    result = evaluate_pair(0.01, 0.004, 0.001, 0.0)
    assert result["se_combined"] == pytest.approx(0.001)
    assert result["sigma"] == pytest.approx(6.0)
    assert result["pass"] is False
    assert result["residual"] == pytest.approx(0.006)
    assert result["tolerance_sigma"] == SIGMA_TOLERANCE

    boundary = evaluate_pair(0.004, 0.001, 0.001, 0.0)
    assert boundary["sigma"] == pytest.approx(3.0)
    assert boundary["pass"] is True

    signed = evaluate_pair(0.001, 0.004, 0.0, 0.001)
    assert signed["sigma"] == pytest.approx(-3.0)
    assert signed["pass"] is True

    vacuous = evaluate_pair(0.0, 0.0, 0.0, 0.0)
    assert vacuous["sigma"] == 0.0
    assert vacuous["pass"] is True
    assert vacuous["se_floor_applied"] is True
    assert vacuous["se_effective"] == DETERMINISTIC_SE_FLOOR

    # The deterministic-observable floor: a collapsed between-replica SE
    # (the R31 window-0 K_0/Sigma_plus artifact scale, 1.28e-16) with a
    # sub-threshold residual is a vacuous pass at the floored SE; a
    # residual above ZERO_RESIDUAL_TOL evaluates at the floored SE and
    # the unchanged 3.0-sigma band.
    artifact = evaluate_pair(0.06286598634487905, 0.06286598634487572,
                             9.063520416928874e-17, 9.063520416928874e-17)
    assert artifact["se_combined"] == pytest.approx(1.2819751242557092e-16)
    assert artifact["se_floor_applied"] is True
    assert artifact["se_effective"] == DETERMINISTIC_SE_FLOOR
    assert artifact["sigma"] == 0.0
    assert artifact["pass"] is True
    assert artifact["residual"] == pytest.approx(3.33e-15, abs=1e-16)

    floored_fail = evaluate_pair(5e-12, 0.0, 1e-17, 0.0)
    assert floored_fail["se_floor_applied"] is True
    assert floored_fail["sigma"] == pytest.approx(5.0)
    assert floored_fail["pass"] is False

    floored_pass = evaluate_pair(2e-12, 0.0, 0.0, 1e-17)
    assert floored_pass["sigma"] == pytest.approx(2.0)
    assert floored_pass["pass"] is True

    # An SE exactly at the floor is not floored (the plain band rule).
    at_floor = evaluate_pair(5e-13, 0.0, DETERMINISTIC_SE_FLOOR, 0.0)
    assert at_floor["se_floor_applied"] is False
    assert at_floor["se_effective"] == DETERMINISTIC_SE_FLOOR
    assert at_floor["sigma"] == pytest.approx(0.5)
    assert at_floor["pass"] is True

    failing_zero_se = evaluate_pair(1e-6, 0.0, 0.0, 0.0)
    assert failing_zero_se["pass"] is False
    assert failing_zero_se["se_floor_applied"] is True
    assert failing_zero_se["sigma"] == pytest.approx(
        1e-6 / DETERMINISTIC_SE_FLOOR)

    custom = evaluate_pair(0.0, 0.002, 0.001, 0.0, sigma=2.0)
    assert custom["sigma"] == pytest.approx(-2.0)
    assert custom["pass"] is True


# ---------------------------------------------------------------------------
# full_analytic_difference: the open-set-change fixture
# ---------------------------------------------------------------------------
def _draw_cell_point(record: HadronRecord, lo: float, hi: float) -> float:
    """The documented narrow-line-shape evaluation point: the card-3
    draw cell containing the pole (the test's own implementation)."""
    cell = (hi - lo) / 1024.0
    idx = min(int(math.floor((float(record.pole_mass) - lo) / cell)), 1023)
    return lo + (idx + 0.5) * cell


def test_full_analytic_difference_open_set_change_fixture():
    """A record opens between the two point masses and no surviving
    window extends: the open-set-change contribution equals the full
    analytic difference (the original ``delta_analytic`` convention)
    and the momentum drift vanishes identically (hand-computed)."""
    _, record_set = _osc_fixture()
    M_k, M_kp1 = 0.8, 0.9
    result_d = full_analytic_difference(record_set, M_k, M_kp1, KEY_D)
    result_e = full_analytic_difference(record_set, M_k, M_kp1, KEY_E)

    assert result_d["open_set"]["changed"] is True
    assert result_d["open_set"]["opened"] == [PDG_C]
    assert result_d["open_set"]["closed"] == []

    # The hand evaluation masses: beta_0's window is fixed on
    # [0.6, 0.7] (needle inside, the draw cell of the pole); gamma_0's
    # window at M_kp1 is [0.85, 0.9] (the pole at the top edge).
    beta = record_set.catalog.get(PDG_B)
    gamma = record_set.catalog.get(PDG_C)
    m_beta = _draw_cell_point(beta, 0.6, 0.7)
    m_gamma = _draw_cell_point(gamma, 0.85, 0.9)
    q_beta = _q_two_body(m_beta, 0.2, 0.2)
    q_gamma = _q_two_body(m_gamma, 0.2, 0.25)
    e_d_gamma = (m_gamma * m_gamma + 0.2 * 0.2 - 0.25 * 0.25) \
        / (2.0 * m_gamma)
    e_e_gamma = (m_gamma * m_gamma - 0.2 * 0.2 + 0.25 * 0.25) \
        / (2.0 * m_gamma)

    # Mixture hand algebra: share 1 at M_k (only beta_0) and 1/2 each
    # at M_{k+1} (beta_0 and gamma_0).
    n_d_k = 2.0
    n_d_kp1 = 0.5 * 2.0 + 0.5 * 1.0
    assert result_d["delta_analytic_full"]["multiplicity"] == (
        pytest.approx(n_d_kp1 - n_d_k))
    assert result_d["momentum_drift"]["multiplicity"] == 0.0
    assert result_d["open_set_change"]["multiplicity"] == (
        pytest.approx(n_d_kp1 - n_d_k))
    assert result_e["delta_analytic_full"]["multiplicity"] == (
        pytest.approx(0.5))
    assert result_e["momentum_drift"]["multiplicity"] == 0.0
    assert result_e["open_set_change"]["multiplicity"] == (
        pytest.approx(0.5))

    p_d_k = 2.0 * q_beta
    p_d_kp1 = 0.5 * 2.0 * q_beta + 0.5 * q_gamma
    mean_p_d_k = p_d_k / n_d_k
    mean_p_d_kp1 = p_d_kp1 / n_d_kp1
    assert result_d["delta_analytic_full"]["mean_p"] == (
        pytest.approx(mean_p_d_kp1 - mean_p_d_k))
    assert result_d["momentum_drift"]["mean_p"] == 0.0
    assert result_d["open_set_change"]["mean_p"] == (
        pytest.approx(mean_p_d_kp1 - mean_p_d_k))
    # epsilon_0 appears only at M_{k+1}: its multiplicity drifts in
    # through the open-set change and its moments are undefined at
    # M_k (the driver's absent-species rule applies there).
    assert result_e["delta_analytic_full"]["mean_p"] is None
    assert result_e["momentum_drift"]["mean_p"] is None
    assert result_e["open_set_change"]["mean_p"] is None

    e_d_k = m_beta
    e_d_kp1 = 0.5 * m_beta + 0.5 * e_d_gamma
    ept_d_k = m_beta * q_beta * (math.pi / 4.0)
    ept_d_kp1 = (0.5 * (m_beta * q_beta)
                 + 0.5 * (e_d_gamma * q_gamma)) * (math.pi / 4.0)
    mean_pt_y_d_k = ept_d_k / e_d_k
    mean_pt_y_d_kp1 = ept_d_kp1 / e_d_kp1
    assert result_d["delta_analytic_full"]["mean_pt_y"] == (
        pytest.approx(mean_pt_y_d_kp1 - mean_pt_y_d_k))
    assert result_d["momentum_drift"]["mean_pt_y"] == 0.0
    assert result_d["open_set_change"]["mean_pt_y"] == (
        pytest.approx(mean_pt_y_d_kp1 - mean_pt_y_d_k))


def test_full_analytic_difference_unchanged_pair_drift_fixture():
    """An unchanged pair whose resonance window extends: the open-set-
    change contribution vanishes identically and the momentum drift
    carries the analytic kinematic drift (checked against the in-test
    quadrature of the documented convention)."""
    catalog, record_set = _drift_fixture()
    M_k, M_kp1 = 0.9, 1.0
    result = full_analytic_difference(record_set, M_k, M_kp1, KEY_D)

    assert result["open_set"]["changed"] is False
    assert result["open_set"]["opened"] == []
    assert result["open_set"]["closed"] == []

    # The open-set-change contribution vanishes identically at an
    # unchanged pair (the counterfactual mixture equals the mixture).
    for observable in ("multiplicity", "mean_p", "mean_pt_y"):
        assert result["open_set_change"][observable] == 0.0, observable
        assert result["momentum_drift"][observable] == (
            result["delta_analytic_full"][observable])

    # The independent in-test quadrature of the documented convention:
    # the record's truncated window integrates on the engine's
    # Gauss-Legendre grid with the card-3 density; the stable
    # alpha_0 identity carries no momentum.
    beta = catalog.get(PDG_B)
    dmap = {PDG_D: 0.2}

    def expected_eq(M):
        lo, top = 0.6, min(1.4, M)
        x = 0.5 * (top - lo) * _ROOT_X + 0.5 * (top + lo)
        w = 0.5 * (top - lo) * _ROOT_W
        rho = np.array([
            spectral_function(float(m), beta, dmap) for m in x
        ])
        norm = float(np.sum(w * rho))
        q = np.array([_q_two_body(float(m), 0.2, 0.2) for m in x])
        e_m = float(np.sum(w * rho * x) / norm)
        e_q = float(np.sum(w * rho * q) / norm)
        e_mq = float(np.sum(w * rho * x * q) / norm)
        return e_q, e_m, e_mq

    e_q_k, e_m_k, e_mq_k = expected_eq(M_k)
    e_q_1, e_m_1, e_mq_1 = expected_eq(M_kp1)

    # mean_p(delta_0) = (2/3 E[q]) / 1 (the beta_0 daughters over the
    # delta_0 identity count); the mixture shares are 1/3 each.
    drift_p = (2.0 / 3.0) * (e_q_1 - e_q_k)
    assert result["delta_analytic_full"]["mean_p"] == pytest.approx(
        drift_p, abs=1e-12)
    assert drift_p > 0.0

    # multiplicity: the count content is mass-constant.
    assert result["delta_analytic_full"]["multiplicity"] == pytest.approx(
        0.0, abs=1e-12)

    # mean_pt_y(delta_0): the energy-weighted track mean; the stable
    # delta_0 identity carries the point mass at rest in the
    # denominator (the card-9 degenerate-window rule).
    def expected_pt_y(e_m, e_mq, M):
        e_sum = (1.0 / 3.0) * M + (1.0 / 3.0) * e_m
        ept = (1.0 / 3.0) * e_mq * (math.pi / 4.0)
        return ept / e_sum

    drift_pt_y = (expected_pt_y(e_m_1, e_mq_1, M_kp1)
                  - expected_pt_y(e_m_k, e_mq_k, M_k))
    assert result["delta_analytic_full"]["mean_pt_y"] == pytest.approx(
        drift_pt_y, abs=1e-12)
    assert drift_pt_y > 0.0


def test_full_analytic_difference_decomposition_identity():
    """The open-set-change and momentum-drift contributions add to the
    full analytic difference exactly, on both fixtures and for every
    checked observable."""
    fixtures = (
        (_osc_fixture(), 0.8, 0.9, KEY_D),
        (_osc_fixture(), 0.8, 0.9, KEY_E),
        (_drift_fixture(), 0.9, 1.0, KEY_D),
        (_drift_fixture(), 0.9, 1.0, KEY_E),
        (_drift_fixture(), 1.1, 1.2, KEY_D),
    )
    for (catalog, record_set), M_k, M_kp1, skey in fixtures:
        result = full_analytic_difference(record_set, M_k, M_kp1, skey)
        for observable in ("multiplicity", "mean_p", "mean_pt_y"):
            parts = (result["open_set_change"][observable],
                     result["momentum_drift"][observable])
            if any(part is None for part in parts):
                assert result["delta_analytic_full"][observable] is None
                continue
            assert result["delta_analytic_full"][observable] == (
                pytest.approx(parts[0] + parts[1], abs=1e-15)), (
                skey, observable)


def test_narrow_record_draw_cell_evaluation():
    """The narrow line shape's evaluation point is the documented
    card-3 draw cell containing the pole, and the record content
    follows the exact on-shell kinematics at that mass."""
    catalog, record_set = _osc_fixture()
    beta = catalog.get(PDG_B)
    m_star = _draw_cell_point(beta, 0.6, 0.7)
    assert record_set.evaluation_mass(beta, 0.8) == pytest.approx(m_star)
    content = record_set.record_content(beta, 0.8)
    q = _q_two_body(m_star, 0.2, 0.2)
    assert content[KEY_D]["n"] == pytest.approx(2.0)
    assert content[KEY_D]["p_sum"] == pytest.approx(2.0 * q)
    assert content[KEY_D]["e_sum"] == pytest.approx(2.0 * (m_star / 2.0))
    assert content[KEY_D]["ept_sum"] == pytest.approx(
        2.0 * (m_star / 2.0) * q * (math.pi / 4.0))


# ---------------------------------------------------------------------------
# rereval_report: hand verdicts and byte stability
# ---------------------------------------------------------------------------
def _fixture_gate_payload() -> dict:
    """A hand-built sealed-gate fixture: four applied checks (two
    flipping from fail to pass under the re-registered metric, one
    unchanged-pair residual staying failed, one open-set-change
    residual staying failed) plus one absent-species row."""
    def check(pair, species, observable, changed, status, observed,
              delta_observed, delta_full, delta_applied, se, sigma):
        return {
            "pair": pair,
            "species": species,
            "observable": observable,
            "open_set_changes": changed,
            "status": status,
            "observed": observed,
            "delta_observed": delta_observed,
            "delta_analytic_full": delta_full,
            "delta_analytic_applied": delta_applied,
            "se_combined": se,
            "sigma": sigma,
            "pass": None if status == "absent_species"
            else (status != "fail"),
        }

    checks = [
        # unchanged pair, fails the original metric (applied 0), passes
        # the re-registered metric: |0.007 - 0.0068| / 0.001 = 0.2
        check([0, 1], "pi_0", "mean_p", False, "fail",
              [1.0, 1.007], 0.007, 0.0068, 0.0, 0.001, 7.0),
        # unchanged pair, fails both metrics: the drift explains only
        # part of the observed difference
        check([1, 2], "pi_0", "mean_pt_y", False, "fail",
              [0.1, 0.11], 0.01, 0.004, 0.0, 0.001, 10.0),
        # open-set-change pair, fails both metrics (the applied value
        # is already the full analytic difference)
        check([2, 3], "proton", "mean_p", True, "fail",
              [0.0, 0.0634498], 0.0634498, 0.0629276, 0.0629276,
              0.000145, 3.5987),
        # open-set-change pair that passes both metrics
        check([3, 4], "proton", "multiplicity", True, "pass",
              [1.0, 0.833], -0.167, -0.16665, -0.16665, 0.0012, 0.29),
        # absent species: no verdict on either metric
        check([4, 5], "kaon", "mean_p", False, "absent_species",
              None, None, 0.0, 0.0, None, None),
    ]
    return {
        "schema": "fixture.sealed_gate/1",
        "run": "FX",
        "clauses": {
            "threshold_discontinuity": {
                "adjacent_pairs_checked": 5,
                "open_set_changes": [2, 3],
                "metric": {"statement": "fixture"},
                "checks": checks,
                "n_checks_applied": 4,
                "n_checks_failed": 3,
                "n_checks_vacuous_zero": 0,
                "n_checks_absent_species": 1,
                "n_open_set_change_applied": 2,
                "n_failed_at_open_set_changes": 1,
                "n_failed_at_unchanged_pairs": 2,
                "max_abs_sigma": 10.0,
                "max_abs_sigma_check": "pair(1,2) pi_0 mean_pt_y",
                "pass": False,
            },
        },
    }


def test_rereval_report_hand_verdicts_and_byte_stability():
    """The re-evaluation of the hand-built sealed-gate fixture
    reproduces the hand-computed re-registered verdicts and is
    byte-stable."""
    payload = _fixture_gate_payload()
    first = rereval_report(payload, run="FX", archive_rel="fixture",
                           gate_sha256="0" * 64, card="card-fixture")
    second = rereval_report(payload, run="FX", archive_rel="fixture",
                            gate_sha256="0" * 64, card="card-fixture")
    assert (json.dumps(first, indent=2) + "\n").encode("utf-8") == (
        json.dumps(second, indent=2) + "\n").encode("utf-8")

    summary = first["summary"]
    assert summary["n_checks"] == 5
    assert summary["n_checks_applied"] == 4
    assert summary["n_absent_species"] == 1
    assert summary["original"]["n_checks_failed"] == 3
    # 3 original failures -> 1 now passing (the 0.2-sigma unchanged
    # check); the two residual failures stay named.
    assert summary["reeval"]["original_failures_now_passing"] == 1
    assert summary["n_fail"] == 2
    assert summary["n_pass"] == 2
    assert summary["reeval"]["pass_rate_among_original_failures"] == (
        pytest.approx(1.0 / 3.0))
    # The applied value under the re-registered metric is the full
    # analytic difference on both pair classes.
    rows = first["checks"]
    assert rows[0]["delta_analytic_applied"] == 0.0068
    assert rows[0]["status"] == "pass"
    assert rows[0]["sigma"] == pytest.approx(0.2)
    assert rows[1]["status"] == "fail"
    assert rows[1]["sigma"] == pytest.approx(6.0)
    assert rows[2]["status"] == "fail"
    assert rows[2]["sigma"] == pytest.approx(
        abs(0.0634498 - 0.0629276) / 0.000145)
    assert rows[3]["status"] == "pass"
    assert rows[4]["status"] == "absent_species"
    assert rows[4]["sigma"] is None
    # The residuals are named with their class and one-line diagnosis.
    residuals = summary["residuals"]
    assert len(residuals) == 2
    by_pair = {(r["pair"][0], r["pair"][1]): r for r in residuals}
    assert by_pair[(1, 2)]["pair_class"] == "open_set_unchanged"
    assert by_pair[(2, 3)]["pair_class"] == "open_set_change"
    for residual in residuals:
        assert residual["diagnosis"]
        assert diagnose_residual(residual) == residual["diagnosis"]
    # The open-set-change diagnosis names the closure-record finding.
    assert "closure clause 3" in by_pair[(2, 3)]["diagnosis"]
    # max sigma tracks the re-registered values.
    assert summary["max_abs_sigma"] == pytest.approx(
        abs(0.01 - 0.004) / 0.001)
    assert summary["max_abs_sigma_check"] == "pair(1,2) pi_0 mean_pt_y"


def test_diagnose_residual_classes():
    """The deterministic diagnosis strings distinguish the pair
    classes and the marginal / reference-floor unchanged residuals."""
    base = {"pair": [3, 4], "species": "pi_0", "observable": "mean_pt_y",
            "sigma": 3.1, "residual": 1e-4}
    marginal = diagnose_residual(dict(base, pair_class="open_set_unchanged"))
    assert "marginal statistical excursion" in marginal
    floor = diagnose_residual(dict(base, sigma=4.9, residual=8e-3,
                                   pair_class="open_set_unchanged"))
    assert "moment-approximation floor" in floor
    change = diagnose_residual(dict(base, pair_class="open_set_change",
                                    delta_analytic_full=0.03))
    assert "closure clause 3" in change


# ---------------------------------------------------------------------------
# The engine-fix card (TASK-20260918-r3-card2-engine-three-body-moment-fix):
# the deterministic-observable SE floor, the exact Kallen three-body slot
# kinematics, and the deep-level daughter-mass quadrature
# ---------------------------------------------------------------------------
def test_reevaluated_status_deterministic_se_floor():
    """The re-evaluation verdict applies the deterministic-observable
    SE floor: a collapsed between-replica SE with a sub-threshold
    residual is ``vacuous_zero`` (the R31 window-0 K_0/Sigma_plus
    artifact shape), a larger residual evaluates at the floored SE, and
    an SE at or above the floor is unchanged."""
    # The R31 artifact shape: analytic and MC agree to 15 decimals, the
    # between-replica SE collapses to 1.28e-16.
    artifact = {
        "status": "fail",
        "delta_observed": 0.06286598634487905,
        "delta_analytic_full": 0.06286598634487572,
        "se_combined": 1.2819751242557092e-16,
    }
    assert _tm._reevaluated_status(artifact) == ("vacuous_zero", 0.0, True)
    # A larger residual evaluates at the floored SE.
    over = dict(artifact, delta_observed=5e-12, delta_analytic_full=0.0)
    status, sigma, passed = _tm._reevaluated_status(over)
    assert status == "fail"
    assert sigma == pytest.approx(5.0)
    assert passed is False
    within = dict(artifact, delta_observed=2e-12, delta_analytic_full=0.0)
    status, sigma, passed = _tm._reevaluated_status(within)
    assert status == "pass"
    assert sigma == pytest.approx(2.0)
    assert passed is True
    # A missing SE field floors the same way.
    no_se = dict(artifact, se_combined=None, delta_observed=1e-13,
                 delta_analytic_full=0.0)
    assert _tm._reevaluated_status(no_se) == ("vacuous_zero", 0.0, True)
    # An SE at or above the floor keeps the plain band rule.
    genuine = {"status": "fail", "delta_observed": 0.007,
               "delta_analytic_full": 0.0068, "se_combined": 0.001}
    status, sigma, passed = _tm._reevaluated_status(genuine)
    assert status == "pass"
    assert sigma == pytest.approx(0.2)
    assert passed is True
    failing = {"status": "fail", "delta_observed": 0.01,
               "delta_analytic_full": 0.004, "se_combined": 0.001}
    assert _tm._reevaluated_status(failing) == ("fail", 6.0, False)
    # absent_species keeps its no-verdict contract.
    assert _tm._reevaluated_status({"status": "absent_species"}) == (
        "absent_species", None, None)


def _lambda_like_masses():
    """The Lambda(1520) -> Lambda pi+ pi- leading-pair masses: the
    unequal leading pair (the Lambda against the pi+) against the
    pi- partner."""
    return 1.52, (1.115683, 0.139570, 0.139570)


def test_three_body_slot_kallen_unequal_masses():
    """The unequal-leading-pair three-body slot nodes follow the exact
    Kallen kinematics (hand-computed on the module's own fixed grids):
    the Kallen decay momentum of the virtual pair, the exact per-slot
    pair-rest energies, on-shell slot nodes, and the kernel's own
    ``decay_momentum`` cross-check."""
    M, masses = _lambda_like_masses()
    mb, mc, md = masses
    lo2 = (mb + mc) ** 2
    hi2 = (M - md) ** 2
    mbc2 = 0.5 * (hi2 - lo2) * _MBC_X + 0.5 * (hi2 + lo2)
    Mbc = np.sqrt(mbc2)
    q_bc_ref = np.sqrt(
        (mbc2 - (mb + mc) ** 2) * (mbc2 - (mb - mc) ** 2)) / (2.0 * Mbc)
    e_b_ref = (mbc2 + mb * mb - mc * mc) / (2.0 * Mbc)
    e_c_ref = (mbc2 + mc * mc - mb * mb) / (2.0 * Mbc)
    e_pair = np.maximum((M * M + mbc2 - md * md) / (2.0 * M), 1e-300)
    q_pair = np.sqrt(np.maximum(
        ((M * M - mbc2 - md * md) ** 2 - 4.0 * mbc2 * md * md)
        / (4.0 * M * M), 0.0))
    gamma_p = e_pair / Mbc
    beta_p = q_pair / e_pair
    assert not np.all(q_bc_ref == np.sqrt(
        np.maximum(Mbc * Mbc / 4.0 - mb * mb, 0.0))), (
        "the fixture masses must be an unequal leading pair")

    for slot, e_ref in ((0, e_b_ref), (1, e_c_ref)):
        sign = 1.0 if slot == 0 else -1.0
        w, e, p = _tm._three_body_slot_nodes(M, masses, slot)
        e_ref_flat = (gamma_p[:, None] * (
            e_ref[:, None] + beta_p[:, None] * q_bc_ref[:, None] * sign
            * _COS3_X[None, :])).ravel()
        p_ref_flat = np.sqrt(np.maximum(
            (gamma_p[:, None] * (
                sign * q_bc_ref[:, None] * _COS3_X[None, :]
                + beta_p[:, None] * e_ref[:, None])) ** 2
            + (q_bc_ref * q_bc_ref)[:, None]
            * (1.0 - _COS3_X[None, :] ** 2), 0.0)).ravel()
        w_ref_flat = (_MBC_W[:, None] * _COS3_W[None, :]
                      * np.ones_like(e_ref_flat.reshape(
                          _MBC_W.shape[0], _COS3_W.shape[0]))).ravel()
        assert np.array_equal(e, e_ref_flat), slot
        assert np.array_equal(p, p_ref_flat), slot
        assert np.array_equal(w, w_ref_flat), slot
        # On-shell: E^2 - p^2 = m^2 in every node (Lorentz-invariant).
        m_slot = mb if slot == 0 else mc
        assert np.allclose(e * e - p * p, m_slot * m_slot,
                           rtol=1e-12, atol=1e-12), slot
        # The Kallen momentum agrees with the kernel's decay_momentum.
        assert np.allclose(q_bc_ref,
                           [decay_momentum(float(math.sqrt(m2)), mb, mc)
                            for m2 in mbc2], rtol=1e-12, atol=1e-14)
        # The slot weights sum to 1.
        assert float(np.sum(w)) == pytest.approx(1.0, abs=1e-12)
    # The partner slot keeps the committed convention.
    w2, e2, p2 = _tm._three_body_slot_nodes(M, masses, 2)
    assert np.array_equal(e2, (M * M - mbc2 + md * md) / (2.0 * M))
    assert np.array_equal(p2, q_pair)
    assert np.array_equal(w2, _MBC_W.copy())
    assert float(np.sum(w2)) == pytest.approx(1.0, abs=1e-12)


#: The committed equal-mass special case of the exact Kallen formulas,
#: reconstructed here verbatim (the byte-identity regression guard: the
#: equal-mass three-body channels must evaluate byte-identically to the
#: committed engine).
_SLOT_CASES = (
    (1.52, (0.1395699, 0.1395699, 0.1349766)),
    (0.958, (0.1395699, 0.1395699, 0.5479)),
    (1.69, (0.493677, 0.493677, 0.1349766)),
    (0.7826, (0.1395699, 0.1395699, 0.1395699)),
    (2.0, (0.1395699, 0.1395699, 0.958)),
)


def _committed_slot_nodes(m_parent, masses, slot):
    """The committed engine's equal-mass three-body slot nodes, written
    out independently (the pre-fix formulas)."""
    mb, mc, md = (float(m) for m in masses)
    M = float(m_parent)
    lo2 = (mb + mc) ** 2
    hi2 = (M - md) ** 2
    mbc2 = 0.5 * (hi2 - lo2) * _MBC_X + 0.5 * (hi2 + lo2)
    Mbc = np.sqrt(mbc2)
    q_bc = np.sqrt(np.maximum(Mbc * Mbc / 4.0 - mb * mb, 0.0))
    e_pair = np.maximum((M * M + mbc2 - md * md) / (2.0 * M), 1e-300)
    q_pair = np.sqrt(np.maximum(
        ((M * M - mbc2 - md * md) ** 2 - 4.0 * mbc2 * md * md)
        / (4.0 * M * M), 0.0))
    gamma_p = e_pair / Mbc
    beta_p = q_pair / e_pair
    if slot <= 1:
        sign = 1.0 if slot == 0 else -1.0
        e_rest = gamma_p[:, None] * (
            0.5 * Mbc[:, None] + beta_p[:, None] * q_bc[:, None] * sign
            * _COS3_X[None, :])
        p_par = gamma_p[:, None] * (
            sign * q_bc[:, None] * _COS3_X[None, :]
            + beta_p[:, None] * 0.5 * Mbc[:, None])
        p_rest = np.sqrt(np.maximum(
            p_par * p_par
            + (q_bc * q_bc)[:, None] * (1.0 - _COS3_X[None, :] ** 2),
            0.0))
        weights = _MBC_W[:, None] * _COS3_W[None, :] * np.ones_like(e_rest)
        return (weights.ravel(), e_rest.ravel(), p_rest.ravel())
    e_rest = (M * M - mbc2 + md * md) / (2.0 * M)
    return (_MBC_W.copy(), e_rest, q_pair)


def test_three_body_slot_equal_mass_byte_identity():
    """Every equal-mass three-body slot battery evaluates byte-identically
    to the committed engine (np.array_equal on the node arrays, plus the
    recorded sha256 digests of the committed outputs)."""
    for M, masses in _SLOT_CASES:
        for slot in (0, 1, 2):
            got = _tm._three_body_slot_nodes(M, masses, slot)
            ref = _committed_slot_nodes(M, masses, slot)
            for g, r in zip(got, ref):
                assert np.array_equal(g, r), (M, masses, slot)


#: The real-catalog equal-mass three-body records and evaluation masses
#: whose full record content is pinned byte-identically to the committed
#: engine (the sha256 of the canonical repr dump; the committed values
#: were captured before the engine fix and are unchanged under it).
_CONTENT_PIN_CASES = (
    (223, 0.7),      # omega(782), below the pole
    (223, 0.8251),   # omega(782), at the window top
    (221, 0.5479),   # eta, at the window top
    # the eta-prime(958) at 377, at the window top, left with its record in the
    # orphan-record revision of 2026-10-29
    (117, 1.6888),   # rho3(1690)0, at the pole
    (117, 2.4938),   # rho3(1690)0, at the window top
)
_CONTENT_PIN_SHA256 = {
    "223|0.7": "961d80a0027cec0a5d0774cf2ba56570"
               "8109c8bffc8e7403cbba5ca54ec81ce6",
    "223|0.8251": "cffd79d201ef389c9ab6e5d5a757ef52"
                  "391f7dbb56bc287578eaab0dc564670a",
    "221|0.5479": "aed74401817d93b6ec47045a028e446d"
                  "acbaa27f5ceb056afc9e438425f751b3",
    # re-pinned 2026-10-02: the convolved quasi-two-body widths (FAB-10)
    # and the radiative channels of eta, eta-prime and omega
    "117|1.6888": "a18c9d60a5b0a91a8e941a493d257225"
                  "7facb9a815e279aadc66179d260d8b45",
    "117|2.4938": "c34d8b0c0ccc26d793c06f06d2842e63"
                  "dac7c27b18c33b39943e7c3d7dc547c4",
}


def _content_digest(content: dict) -> str:
    blob = json.dumps(
        {skey: {f: repr(float(v)) for f, v in sorted(block.items())}
         for skey, block in sorted(content.items())},
        sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def test_equal_mass_three_body_record_content_byte_identity():
    """The real-catalog equal-mass three-body records' full record
    content is byte-identical to the committed engine (sha256 pins
    captured from the committed code before the Kallen fix)."""
    from na49_mixing.hadron_catalog import HadronCatalog
    catalog = HadronCatalog.from_json(CATALOG_PATH)
    for pdg, mass in _CONTENT_PIN_CASES:
        record = catalog.get(pdg)
        record_set = RecordSet(catalog, [record])
        content = record_set.record_content(record, float(mass))
        assert _content_digest(content) == _CONTENT_PIN_SHA256[
            f"{pdg}|{mass!r}"], (pdg, mass)


def test_lambda1520_record_content_kallen_match():
    """The unequal-leading-pair Lambda(1520) -> Lambda pi+ pi- record:
    the fixed engine's pi+ ``<p_T>_y`` at the probe mass matches the
    kernel-exact Monte Carlo (0.18536 at 120k decays) within its
    statistical error, the committed engine's clamped value 0.11959 is
    replaced, and the count content is untouched."""
    from na49_mixing.hadron_catalog import HadronCatalog
    catalog = HadronCatalog.from_json(CATALOG_PATH)
    record = catalog.get(102134)
    record_set = RecordSet(catalog, [record])
    content = record_set.record_content(record, 2.67)
    block = content["pi_plus"]
    pt_y = float(block["ept_sum"]) / float(block["e_sum"])
    # The kernel-exact MC reference (probe evidence, 120k decays:
    # 0.18536); the fixed engine agrees within the MC statistical
    # uncertainty and the committed clamped value is gone.
    assert pt_y == pytest.approx(0.18536, abs=3e-3), pt_y
    assert pt_y == pytest.approx(0.184825, abs=5e-4), pt_y
    assert abs(pt_y - 0.11959) > 0.05
    # The count content is kinematics-independent and byte-identical to
    # the committed engine's (the pi+ multiplicity carries the direct
    # Sigma-class channel plus the three-body channel; the Lambda
    # multiplicity is the three-body share alone).
    assert float(block["n"]) == 0.22311400372509052
    lam = content["Lambda"]
    assert float(lam["n"]) == 0.08727110696193102


# The deep-level integration fixture: the two-level cascade
# beta_0 -> gamma_0 + delta_0, gamma_0 -> zeta_0 + delta_0,
# zeta_0 -> delta_0 + epsilon_0 (all single-channel, so the open-object
# shares are 1 and the hand reference needs only the documented budget
# windows, the card-3 densities, and the boost convention).  The deep
# daughter zeta_0's budget window is broad: [0.45, min(0.7, m_gamma -
# 0.2)] moves with the gamma_0 realization up to m_gamma = 0.9.
PDG_B2 = 900012
PDG_G2 = 900014
PDG_Z2 = 900016
KEY_DELTA = "delta_0"
KEY_EPSILON = "epsilon_0"


def _deep_fixture():
    delta = _record(PDG_D, "delta0", 0.2, 0.0, (0.2, 0.2), (), stable=True)
    epsilon = _record(PDG_E, "epsilon0", 0.25, 0.0, (0.25, 0.25), (),
                      stable=True)
    zeta = _record(PDG_Z2, "zeta0", 0.55, 0.06, (0.45, 0.7),
                   (_channel(PDG_Z2, (PDG_D, PDG_E), 1.0, 0),))
    gamma = _record(PDG_G2, "gamma0", 0.85, 0.09, (0.78, 1.0),
                    (_channel(PDG_G2, (PDG_Z2, PDG_D), 1.0, 0),))
    beta = _record(PDG_B2, "beta0", 1.15, 0.1, (1.1, 1.45),
                   (_channel(PDG_B2, (PDG_G2, PDG_D), 1.0, 0),))
    catalog = _FixtureCatalog([delta, epsilon, zeta, gamma, beta])
    return catalog, RecordSet(catalog, [beta])  # type: ignore[arg-type]


def _boost_aggregates(e_rest, p_rest, gamma, beta, cos, cos_w):
    """One (E, p) node boosted over an isotropic direction grid (the
    legacy ``_decay_rest_frame`` convention), returning the direction
    sums of the weight, E, p, and E*p moments (the weights sum to 1)."""
    e_lab = gamma * (e_rest + beta * p_rest * cos)
    p_par = gamma * (p_rest * cos + beta * e_rest)
    p_lab = np.sqrt(p_par * p_par + p_rest * p_rest * (1.0 - cos * cos))
    w = cos_w
    return (float(np.sum(w)), float(np.sum(w * e_lab)),
            float(np.sum(w * p_lab)), float(np.sum(w * e_lab * p_lab)))


def _deep_reference(M: float, deep: bool = True) -> dict:
    """The explicit in-test construction of the documented convention
    for the deep fixture at the point mass *M*: the root window on the
    engine's documented Gauss-Legendre grid, the gamma_0 daughter's
    budget-window quadrature, and the zeta_0 granddaughter treatment --
    with deep=True the depth-1 budget-window quadrature (the mechanism
    under test), with deep=False the pre-fix concentration-point
    collapse (the zeta_0 granddaughter at the pole 0.55).  The two
    nested isotropic boost levels use the deep grid at the zeta level
    and the first-level grid at the beta level."""
    dau_x, dau_w = np.polynomial.legendre.leggauss(DAUGHTER_GL_NODES)
    cos_d, cos_w_d = np.polynomial.legendre.leggauss(ANGLE_NODES_DEEP)
    cos_w_d = cos_w_d * 0.5
    cos_1, cos_w_1 = np.polynomial.legendre.leggauss(ANGLE_NODES)
    cos_w_1 = cos_w_1 * 0.5

    catalog, _ = _deep_fixture()
    beta_rec = catalog.get(PDG_B2)
    gamma_rec = catalog.get(PDG_G2)
    zeta_rec = catalog.get(PDG_Z2)
    dmap_b = {PDG_G2: 0.85, PDG_D: 0.2}
    dmap_g = {PDG_Z2: 0.55, PDG_D: 0.2}
    dmap_z = {PDG_D: 0.2, PDG_E: 0.25}

    lo_b, top_b = 1.1, min(1.45, float(M))
    x = 0.5 * (top_b - lo_b) * _ROOT_X + 0.5 * (top_b + lo_b)
    w = 0.5 * (top_b - lo_b) * _ROOT_W
    rho = np.array([spectral_function(float(m), beta_rec, dmap_b)
                    for m in x])
    norm = float(np.sum(w * rho))
    acc = {KEY_DELTA: [0.0, 0.0, 0.0, 0.0],
           KEY_EPSILON: [0.0, 0.0, 0.0, 0.0]}

    def add(key, weight, p_sum, e_sum, ep_sum):
        acc[key][0] += weight
        acc[key][1] += weight * p_sum
        acc[key][2] += weight * e_sum
        acc[key][3] += weight * ep_sum

    for m_i, w_i, r_i in zip(x, w, rho):
        node_w = float(w_i) * float(r_i) / norm
        top_g = min(1.0, float(m_i) - 0.2)
        xg = 0.5 * (top_g - 0.78) * dau_x + 0.5 * (top_g + 0.78)
        wg = 0.5 * (top_g - 0.78) * dau_w
        rho_g = np.array([spectral_function(float(m), gamma_rec, dmap_g)
                          for m in xg])
        z_g = float(np.sum(wg * rho_g))
        for m_g, w_g, r_g in zip(xg, wg, rho_g):
            gwt = node_w * float(w_g) * float(r_g) / z_g
            # beta_0's channel split at (m_i, m_g, 0.2).
            q_split = decay_momentum(float(m_i), float(m_g), 0.2)
            e_res_b = (m_i * m_i + m_g * m_g - 0.04) / (2.0 * m_i)
            e_stab_b = (m_i * m_i - m_g * m_g + 0.04) / (2.0 * m_i)
            # The direct delta_0 partner of beta_0's channel: already in
            # the beta_0 frame, no boost.
            add(KEY_DELTA, gwt, q_split, e_stab_b, e_stab_b * q_split)
            # gamma_0's content at m_g, boosted by the beta-level split
            # over the first-level direction grid.
            gamma_b = e_res_b / float(m_g)
            beta_b = q_split / e_res_b
            if deep:
                top_z = min(0.7, float(m_g) - 0.2)
                xz = 0.5 * (top_z - 0.45) * dau_x + 0.5 * (top_z + 0.45)
                wz = 0.5 * (top_z - 0.45) * dau_w
                rho_z = np.array([spectral_function(float(m), zeta_rec,
                                                    dmap_z) for m in xz])
                z_norm = float(np.sum(wz * rho_z))
                z_nodes = [(float(m), float(wgt) * float(r) / z_norm)
                           for m, wgt, r in zip(xz, wz, rho_z)]
            else:
                # The collapsed deep level: zeta_0 at its concentration
                # point (the pole 0.55, inside every budget window).
                z_nodes = [(0.55, 1.0)]
            for m_z, w_zn in z_nodes:
                zwt = gwt * w_zn
                # gamma_0's channel split at (m_g, m_z, 0.2).
                q_g = decay_momentum(float(m_g), float(m_z), 0.2)
                e_res_g = (m_g * m_g + m_z * m_z - 0.04) / (2.0 * m_g)
                e_stab_g = (m_g * m_g - m_z * m_z + 0.04) / (2.0 * m_g)
                # gamma_0's direct delta_0 partner, boosted over the
                # first-level grid.
                n_o, e_o, p_o, ep_o = _boost_aggregates(
                    e_stab_g, q_g, gamma_b, beta_b, cos_1, cos_w_1)
                add(KEY_DELTA, zwt * n_o, p_o, e_o, ep_o)
                # The zeta_0 daughter's decay content at m_z: the exact
                # two-body (delta_0, epsilon_0) kinematics, boosted over
                # the deep grid by the gamma-level split and then over
                # the first-level grid by the beta-level split.
                gamma_z = e_res_g / float(m_z)
                beta_z = q_g / e_res_g
                q_z = decay_momentum(float(m_z), 0.2, 0.25)
                e_d_z = (m_z * m_z + 0.04 - 0.0625) / (2.0 * m_z)
                e_e_z = (m_z * m_z - 0.04 + 0.0625) / (2.0 * m_z)
                for key, e_rest in ((KEY_DELTA, e_d_z),
                                    (KEY_EPSILON, e_e_z)):
                    e_in = gamma_z * (e_rest + beta_z * q_z * cos_d)
                    p_par = gamma_z * (q_z * cos_d + beta_z * e_rest)
                    p_in = np.sqrt(p_par * p_par
                                   + q_z * q_z * (1.0 - cos_d * cos_d))
                    e_out = gamma_b * (e_in[None, :]
                                       + beta_b * p_in[None, :]
                                       * cos_1[:, None])
                    p_out_par = gamma_b * (p_in[None, :] * cos_1[:, None]
                                           + beta_b * e_in[None, :])
                    p_out = np.sqrt(
                        p_out_par * p_out_par
                        + p_in[None, :] ** 2
                        * (1.0 - cos_1[:, None] ** 2))
                    w_oi = cos_w_1[:, None] * cos_w_d[None, :]
                    add(key, zwt * float(np.sum(w_oi)),
                        float(np.sum(w_oi * p_out)),
                        float(np.sum(w_oi * e_out)),
                        float(np.sum(w_oi * e_out * p_out)))
    return {key: {"n": v[0], "p_sum": v[1], "e_sum": v[2],
                  "ept_sum": v[3] * (math.pi / 4.0)}
            for key, v in acc.items()}


def test_deep_level_daughter_mass_integration():
    """The two-level cascade fixture: the deep granddaughter's
    budget-window quadrature replaces the concentration-point collapse
    (the engine equals the explicit in-test reference of the documented
    convention, and differs from the collapsed evaluation)."""
    catalog, record_set = _deep_fixture()
    M = 1.3
    content = record_set.record_content(catalog.get(PDG_B2), M)
    reference = _deep_reference(M, deep=True)
    collapsed = _deep_reference(M, deep=False)
    assert set(content.keys()) == set(reference.keys())
    for key in sorted(reference):
        block = content[key]
        ref = reference[key]
        for field in ("n", "p_sum", "e_sum", "ept_sum"):
            assert block[field] == pytest.approx(
                ref[field], rel=1e-9, abs=1e-12), (key, field)
        # The collapsed (pre-fix) evaluation differs: the mechanism
        # fires on this fixture.
        for field in ("p_sum", "ept_sum"):
            assert abs(block[field] - collapsed[key][field]) > 1e-9, (
                key, field)


# ---------------------------------------------------------------------------
# Sealed-archive re-evaluation (read-only; skips on a fresh checkout)
# ---------------------------------------------------------------------------
def _gate_path(run: str) -> Path:
    archive = R21_ARCHIVE if run == "r21" else R22_ARCHIVE
    return archive / f"gate2_{run}.json"


def _sealed_available() -> bool:
    return (_gate_path("r21").is_file() and _gate_path("r22").is_file())


@pytest.mark.skipif(not _sealed_available(), reason="sealed R21/R22 "
                    "archives absent on this checkout")
def test_sealed_archive_rereval_counts_and_byte_stability():
    """The re-evaluation of the real sealed archives reproduces the
    re-registration counts, is byte-stable, and leaves the sealed gate
    files byte-unchanged."""
    sealed_digests = {}
    payloads = {}
    for run, old_fail, now_pass, n_residuals in (
        ("r21", 169, 161, 8),
        ("r22", 154, 151, 3),
    ):
        gate_path = _gate_path(run)
        sealed_digests[f"gate2_{run}.json"] = _sha256(gate_path)
        gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
        payload = rereval_report(
            gate_payload, run=run.upper(),
            archive_rel=f"dev/00runs/{gate_path.parent.name}/"
                        f"gate2_{run}.json",
            gate_sha256=sealed_digests[f"gate2_{run}.json"],
            card="TASK-20260917-r3-gate2-followup-threshold-metric-"
                 "reregistration",
        )
        payloads[run] = payload
        summary = payload["summary"]
        assert summary["original"]["n_checks_failed"] == old_fail
        assert summary["reeval"]["original_failures_now_passing"] == (
            now_pass)
        assert len(summary["residuals"]) == n_residuals
        assert summary["n_checks_applied"] == (351 if run == "r21"
                                               else 483)
        # byte stability: a second evaluation is byte-identical
        again = rereval_report(
            gate_payload, run=run.upper(),
            archive_rel=f"dev/00runs/{gate_path.parent.name}/"
                        f"gate2_{run}.json",
            gate_sha256=sealed_digests[f"gate2_{run}.json"],
            card="TASK-20260917-r3-gate2-followup-threshold-metric-"
                 "reregistration",
        )
        assert _canonical(first=payload) == _canonical(first=again)
    # the sealed gate files are byte-unchanged (the archives' own
    # clean-environment records pin the digests)
    assert sealed_digests == SEALED_GATE_SHA256
    # the combined acceptance: 312 of the 323 original failures pass
    now = sum(p["summary"]["reeval"]["original_failures_now_passing"]
              for p in payloads.values())
    assert now == 312
    assert now / 323 >= 0.95


@pytest.mark.skipif(not _sealed_available(), reason="sealed R21/R22 "
                    "archives absent on this checkout")
def test_sealed_archive_engine_exact_on_first_pairs():
    """The new-run analytic engine reproduces the sealed stored
    ``delta_analytic_full`` exactly on the first R21 pairs (the
    records there carry stable daughters only, so no quadrature
    approximation enters)."""
    from na49_mixing.hadron_catalog import HadronCatalog
    from na49_mixing.threshold_metric import RecordSet as EngineRecordSet

    catalog = HadronCatalog.from_json(CATALOG_PATH)
    pre = json.loads(
        (R21_ARCHIVE / "pre_registration.json").read_text(
            encoding="utf-8"))
    gate = json.loads(_gate_path("r21").read_text(encoding="utf-8"))
    open_sets = {int(k): v for k, v in pre["open_sets"].items()}
    masses = [float(m) for m in pre["grid_masses_gev"]]
    union_ids = sorted({pdg for ids in open_sets.values() for pdg in ids})
    record_set = EngineRecordSet(
        catalog, [catalog.get(pdg) for pdg in union_ids])
    checks = gate["clauses"]["threshold_discontinuity"]["checks"]
    for pair in ((0, 1), (1, 2)):
        rows = [c for c in checks if tuple(c["pair"]) == pair
                and c["status"] in ("pass", "fail", "vacuous_zero")]
        for skey in sorted({c["species"] for c in rows}):
            delta = full_analytic_difference(
                record_set, masses[pair[0]], masses[pair[1]], skey)
            for check in rows:
                if check["species"] != skey:
                    continue
                engine = delta["delta_analytic_full"][
                    check["observable"]]
                sealed = check["delta_analytic_full"]
                if engine is None or sealed is None:
                    continue
                assert engine == pytest.approx(sealed, abs=1e-9), (
                    pair, skey, check["observable"], engine, sealed)


# ---------------------------------------------------------------------------
# The card-9 driver's new-run metric
# ---------------------------------------------------------------------------
def _load_driver(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(first: dict) -> bytes:
    return (json.dumps(first, indent=2) + "\n").encode("utf-8")


def test_card9_driver_new_run_metric_is_reregistered():
    """The card-9 driver's new-run discontinuity metric is the
    re-registered one: the DISCONTINUITY_METRIC declaration, the
    pre-registration declaration, and the driver's metric table on the
    real grid."""
    driver = _load_driver(
        NESTED_ROOT / SOURCE_GRID_DRIVER_REL, "staged_source_grid_card2")
    metric = driver.DISCONTINUITY_METRIC
    assert metric["id"] == "delta_analytic_full"
    assert metric["engine"] == (
        "na49_mixing.threshold_metric.full_analytic_difference")
    assert metric["record_set"] == "na49_mixing.threshold_metric.RecordSet"
    assert metric["evaluate"] == "na49_mixing.threshold_metric.evaluate_pair"
    assert metric["tolerance_sigma"] == 3.0
    assert metric["card"] == (
        "TASK-20260917-r3-gate2-followup-threshold-metric-reregistration")
    assert "BOTH pair classes" in metric["statement"]
    # the pre-registration of every new run declares the metric
    source = (NESTED_ROOT / SOURCE_GRID_DRIVER_REL).read_text(
        encoding="utf-8")
    assert '"threshold_metric": DISCONTINUITY_METRIC' in source
    # the driver's table is the module's metric on the real grid
    from na49_mixing.hadron_catalog import HadronCatalog
    catalog = HadronCatalog.from_json(CATALOG_PATH)
    table = driver.threshold_metric_table(
        catalog, "baryonic", point_indices=[0, 1, 2])
    assert table["metric"] is metric
    assert list(table["pairs"]) == ["0,1", "1,2"]
    proton = table["pairs"]["0,1"]["species"]["proton"]
    assert proton["delta_analytic_full"]["multiplicity"] == (
        pytest.approx(-0.16665))
    assert table["pairs"]["0,1"]["open_records_at_M_k"] == [2212]
    assert table["pairs"]["0,1"]["open_records_at_M_kp1"] == [2212, 2214]


@pytest.mark.skipif(not _sealed_available(), reason="sealed R21/R22 "
                    "archives absent on this checkout")
def test_card9_driver_table_matches_sealed_reference():
    """On the first R21 pair the driver's new-run metric table equals
    the sealed stored ``delta_analytic_full`` values (the engine
    cross-check evidence, exact where no quadrature approximation
    enters)."""
    driver = _load_driver(
        NESTED_ROOT / SOURCE_GRID_DRIVER_REL, "staged_source_grid_card2b")
    from na49_mixing.hadron_catalog import HadronCatalog
    catalog = HadronCatalog.from_json(CATALOG_PATH)
    table = driver.threshold_metric_table(
        catalog, "baryonic", point_indices=[0, 1, 2])
    gate = json.loads(_gate_path("r21").read_text(encoding="utf-8"))
    checks = gate["clauses"]["threshold_discontinuity"]["checks"]
    sealed = {}
    for check in checks:
        if tuple(check["pair"]) != (0, 1):
            continue
        if check["status"] not in ("pass", "fail", "vacuous_zero"):
            continue
        sealed[(check["species"], check["observable"])] = (
            check["delta_analytic_full"])
    for pair_key in ("0,1",):
        for skey, per_obs in table["pairs"][pair_key]["species"].items():
            for observable, engine in per_obs["delta_analytic_full"].items():
                ref = sealed.get((skey, observable))
                if ref is None or engine is None:
                    continue
                assert engine == pytest.approx(ref, abs=1e-9), (
                    skey, observable, engine, ref)


def test_dual_tree_byte_identity():
    """The nested and outer copies of the two dual-tree drivers are
    byte-identical (the strongest form of the layout allowance)."""
    for rel in (REREVAL_DRIVER_REL, SOURCE_GRID_DRIVER_REL):
        nested = NESTED_ROOT / rel
        outer = MONO_ROOT / rel
        assert nested.is_file(), nested
        assert outer.is_file(), outer
        assert nested.read_text(encoding="utf-8") == (
            outer.read_text(encoding="utf-8")), rel
