"""Deterministic tests for the sector-wise deviance decomposition
(Round 1, task card TASK-20260911-r1-sector-wise-fit-reports).

The decomposition is a statistical partition of the existing binned
chi-square (plan section 5.4); it fits no parameter.  The hand-
constructed (d, m, V) cases use no event generation.  The two CLI runs
execute the 20-event smoke configuration through the identical
``main()`` entry of ``python -m na49_mixing.staged_evaluation`` (the
``na49-staged-excitation`` console script is the same entry point and
is not required here).

Call-path choices (recorded per the card): the physics-convention pins
call ``sector_decomposition`` and ``write_sector_report`` directly on
hand-built observables, and the region-boundary pin reads the private
membership helper ``_sector_region_of`` (the pure per-axis rule), the
narrowest entry point that consumes plain numbers.  The flag-on CLI run
reads the NA49 SPS tables through ``fit.binned`` and carries the
``external_data`` marker (the hermetic archive checkouts deselect it);
the flag-off bit-identity run needs no tables (the smoke fit is
disabled) and stays unmarked.

Documented behaviors pinned here (the card is silent on each; the
choices are recorded in the ``fit_likelihood`` module docstring):

- sector key ``"{species}|{observable}|{xF_region}|{pT_region}"`` — the
  (xF region, pT region) cross product behind the card's
  ``n_sectors == 9 x 6`` acceptance number;
- region membership half-open ``[lo, hi)`` on every region of an axis,
  with the last region of the axis closed ``[lo, hi]``;
- the nuisance penalty booked once per observable, into the sector of
  the lowest-index assigned bin, so the sector partition stays exact;
- ``residual_mean`` / ``worst_bin.pull`` use the card's unshifted
  difference ``(d - m) / sigma`` with the nominal model value and the
  combined per-bin error ``sqrt(V_ii)``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.fit_likelihood import (
    SECTOR_PT_REGION_ORDER,
    SECTOR_REGIONS,
    SECTOR_XF_REGION_ORDER,
    BinnedObservable,
    NormalizationNuisance,
    _sector_region_of,
    combined_covariance,
    diagonal_covariance,
    evaluate_likelihood,
    sector_decomposition,
    write_sector_report,
)
from na49_mixing.staged_evaluation import SECTOR_REPORT_FILENAME, main

NESTED_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = NESTED_ROOT / "config" / "staged_excitation_smoke.json"


# ---------------------------------------------------------------------------
# Hand-constructed observables (no event generation)
# ---------------------------------------------------------------------------


def _nine_bin_observable(
    values: tuple[float, ...],
    model: tuple[float, ...],
    stat_errors: tuple[float, ...],
) -> BinnedObservable:
    """The card's 9-bin case: 3 xF bins x 3 pT bins, one bin per sector.

    xF edges (0.0, 0.1, 0.2, 0.3) give midpoints 0.05, 0.15, 0.25 ->
    central, transition, forward; pT edges (0.05, 0.5, 1.0, 1.5) give
    midpoints 0.275, 0.75, 1.25 -> low, mid, high.  The flat layout is
    ``i = ip * 3 + ix`` (pT-major, xF-minor).
    """
    return BinnedObservable(
        species="pi-",
        observable="f_xsect",
        bin_edges=((0.0, 0.1, 0.2, 0.3), (0.05, 0.5, 1.0, 1.5)),
        values=values,
        stat_errors=stat_errors,
        cov=None,
        normalization_nuisance=None,
    )


def _eighteen_bin_observable(
    pulls: tuple[float, ...],
) -> tuple[BinnedObservable, tuple[float, ...]]:
    """18 bins = 6 xF bins x 3 pT bins, two xF bins per xF region.

    xF midpoints 0.00, 0.05 (central), 0.10, 0.15 (transition), 0.20,
    0.25 (forward); pT midpoints 0.1, 0.7, 1.4 (low, mid, high).  Each
    sector holds exactly two bins.  The model is 0.5 per bin, the
    combined error is 0.1 per bin (data-only diagonal), and the data
    values realize the hand-chosen pulls ``(d - m) / sigma``.
    """
    assert len(pulls) == 18
    stat_error = 0.1
    model = tuple(0.5 for _ in pulls)
    values = tuple(0.5 + pull * stat_error for pull in pulls)
    observable = BinnedObservable(
        species="K+",
        observable="f_xsect",
        bin_edges=(
            (-0.025, 0.025, 0.075, 0.125, 0.175, 0.225, 0.275),
            (0.0, 0.2, 1.2, 1.6),
        ),
        values=values,
        stat_errors=tuple(stat_error for _ in pulls),
        cov=None,
        normalization_nuisance=None,
    )
    return observable, model


# ---------------------------------------------------------------------------
# test_sector_decomposition_partition
# ---------------------------------------------------------------------------


def test_sector_decomposition_partition():
    """The 9-sector chi2_total sum equals the global chi2 to < 1e-9.

    The partition is exact on the card's 9-bin hand case, stays exact
    with a nuisance in scope (the penalty booked once, into the sector
    of the lowest-index assigned bin), and stays exact for an
    off-diagonal combined covariance (the row-restricted quadratic
    form).  The half-open boundary rule is pinned on the abutting
    midpoints of the pre-registered grid.
    """
    values = (1.0, 0.8, 0.5, 1.2, 0.9, 0.4, 1.1, 0.7, 0.3)
    model = (0.9, 0.85, 0.55, 1.1, 0.8, 0.45, 1.2, 0.6, 0.25)
    stat_errors = (0.1, 0.2, 0.15, 0.1, 0.2, 0.15, 0.1, 0.2, 0.15)
    sigma_mc = (0.05, 0.1, 0.05, 0.05, 0.1, 0.05, 0.05, 0.1, 0.05)
    d = _nine_bin_observable(values, model, stat_errors)
    v_combined = combined_covariance(
        diagonal_covariance(stat_errors), diagonal_covariance(sigma_mc)
    )

    # Diagonal case, no nuisance: 9 sectors, exact partition.
    decomposition = sector_decomposition(d, model, v_combined, None)
    assert decomposition["unassigned_bins"] == 0
    sectors = decomposition["sectors"]
    assert len(sectors) == 9
    expected_keys = {
        f"pi-|f_xsect|{xf}|{pt}"
        for xf in SECTOR_XF_REGION_ORDER
        for pt in SECTOR_PT_REGION_ORDER
    }
    assert set(sectors) == expected_keys
    chi2_global, components = evaluate_likelihood(d, model, v_combined, None)
    sector_sum = sum(s["chi2_total"] for s in sectors.values())
    assert abs(sector_sum - chi2_global) / chi2_global < 1e-9
    data_sum = sum(s["chi2_data"] for s in sectors.values())
    mc_sum = sum(s["chi2_mc"] for s in sectors.values())
    assert abs(data_sum - components["chi2_data"]) / components["chi2_data"] < 1e-9
    assert abs(mc_sum - components["chi2_mc"]) / components["chi2_mc"] < 1e-9
    for sector in sectors.values():
        assert sector["chi2_data"] + sector["chi2_mc"] == pytest.approx(
            sector["chi2_total"], rel=1e-12
        )
        assert sector["chi2_nuisance"] == 0.0
        assert sector["n_bins"] == 1
    assert sum(s["fraction_of_total"] for s in sectors.values()) == pytest.approx(
        1.0, rel=1e-12
    )

    # One in-scope nuisance: the penalty (0.01 / 0.02)**2 = 0.25 is booked
    # exactly once, into the sector containing bin 0 (xF_central x pT_low);
    # the partition against evaluate_likelihood stays exact.
    nuisance = NormalizationNuisance(
        name="sps_test_norm", eta=0.01, sigma_eta=0.02, table_scope="pi-"
    )
    chi2_global_n, components_n = evaluate_likelihood(
        d, model, v_combined, [nuisance]
    )
    decomposition_n = sector_decomposition(d, model, v_combined, [nuisance])
    sectors_n = decomposition_n["sectors"]
    sector_sum_n = sum(s["chi2_total"] for s in sectors_n.values())
    assert abs(sector_sum_n - chi2_global_n) / chi2_global_n < 1e-9
    penalty_sectors = [
        key for key, s in sectors_n.items() if s["chi2_nuisance"] != 0.0
    ]
    assert penalty_sectors == ["pi-|f_xsect|xF_central|pT_low"]
    assert sectors_n[penalty_sectors[0]]["chi2_nuisance"] == pytest.approx(
        0.25, rel=1e-12
    )
    booked = sectors_n[penalty_sectors[0]]
    assert booked["chi2_total"] == pytest.approx(
        booked["chi2_data"] + booked["chi2_mc"] + booked["chi2_nuisance"],
        rel=1e-12,
    )
    assert components_n["chi2_nuisance"] == pytest.approx(0.25, rel=1e-12)

    # Off-diagonal combined covariance: the partition stays exact and the
    # per-component sums reproduce evaluate_likelihood's component split.
    stat_arr = np.asarray(stat_errors, dtype=float)
    mc_arr = np.asarray(sigma_mc, dtype=float)

    def _offdiag_squared(diag_values, coupling: float):
        matrix = np.diag(np.asarray(diag_values, dtype=float) ** 2)
        off = coupling * np.outer(
            np.asarray(diag_values, dtype=float),
            np.asarray(diag_values, dtype=float),
        )
        return matrix + np.triu(off, 1) + np.tril(off, -1)

    v_data_off = _offdiag_squared(stat_arr, 0.2)
    v_mc_off = _offdiag_squared(mc_arr, 0.1)
    observable_off = BinnedObservable(
        species="pi-",
        observable="f_xsect",
        bin_edges=((0.0, 0.1, 0.2, 0.3), (0.05, 0.5, 1.0, 1.5)),
        values=values,
        stat_errors=stat_errors,
        cov=tuple(float(v) for v in v_data_off[np.triu_indices(9)]),
        normalization_nuisance=None,
    )
    v_off = tuple(float(v) for v in (v_data_off + v_mc_off)[np.triu_indices(9)])
    chi2_off, components_off = evaluate_likelihood(
        observable_off, model, v_off, None
    )
    sectors_off = sector_decomposition(observable_off, model, v_off, None)[
        "sectors"
    ]
    sector_sum_off = sum(s["chi2_total"] for s in sectors_off.values())
    assert abs(sector_sum_off - chi2_off) / chi2_off < 1e-9
    data_sum_off = sum(s["chi2_data"] for s in sectors_off.values())
    mc_sum_off = sum(s["chi2_mc"] for s in sectors_off.values())
    assert (
        abs(data_sum_off - components_off["chi2_data"])
        / components_off["chi2_data"]
        < 1e-9
    )
    assert (
        abs(mc_sum_off - components_off["chi2_mc"]) / components_off["chi2_mc"]
        < 1e-9
    )

    # Boundary pin (documented choice): half-open [lo, hi) on every region
    # of an axis, with the last region closed [lo, hi].
    assert _sector_region_of(0.0, SECTOR_XF_REGION_ORDER) == "xF_central"
    assert _sector_region_of(0.10, SECTOR_XF_REGION_ORDER) == "xF_transition"
    assert _sector_region_of(0.2, SECTOR_XF_REGION_ORDER) == "xF_forward"
    assert _sector_region_of(0.4, SECTOR_XF_REGION_ORDER) == "xF_forward"
    assert _sector_region_of(0.3999999999, SECTOR_XF_REGION_ORDER) == "xF_forward"
    assert _sector_region_of(-0.05, SECTOR_XF_REGION_ORDER) is None
    assert _sector_region_of(0.5, SECTOR_XF_REGION_ORDER) is None
    assert _sector_region_of(0.05, SECTOR_PT_REGION_ORDER) == "pT_low"
    assert _sector_region_of(0.5, SECTOR_PT_REGION_ORDER) == "pT_mid"
    assert _sector_region_of(1.0, SECTOR_PT_REGION_ORDER) == "pT_high"
    assert _sector_region_of(1.5, SECTOR_PT_REGION_ORDER) == "pT_high"
    assert _sector_region_of(1.4999999999, SECTOR_PT_REGION_ORDER) == "pT_high"
    assert _sector_region_of(0.0499999999, SECTOR_PT_REGION_ORDER) is None
    assert _sector_region_of(2.0, SECTOR_PT_REGION_ORDER) is None


# ---------------------------------------------------------------------------
# test_sector_decomposition_worst_bin
# ---------------------------------------------------------------------------


def test_sector_decomposition_worst_bin():
    """A hand-placed large pull is the sector's worst_bin with the
    correct bin_index and pull (data-only diagonal, so the pulls are
    exact rationals)."""
    values = (1.0, 0.8, 0.5, 1.2, 0.9, 0.4, 1.1, 1.2, 0.3)
    model = (0.9, 0.85, 0.55, 1.1, 0.8, 0.45, 1.2, 0.6, 0.25)
    stat_errors = (0.1, 0.2, 0.15, 0.1, 0.2, 0.15, 0.1, 0.2, 0.15)
    d = _nine_bin_observable(values, model, stat_errors)
    v = diagonal_covariance(stat_errors)
    decomposition = sector_decomposition(d, model, v, None)
    sectors = decomposition["sectors"]
    # Bin 7 (ix=1 -> xF_transition, ip=2 -> pT_high): pull (1.2 - 0.6)/0.2 = 3.0,
    # the largest |pull| of the whole observable.
    worst_sector = sectors["pi-|f_xsect|xF_transition|pT_high"]
    assert worst_sector["worst_bin"]["bin_index"] == 7
    assert worst_sector["worst_bin"]["d"] == 1.2
    assert worst_sector["worst_bin"]["m"] == 0.6
    assert worst_sector["worst_bin"]["sigma"] == 0.2
    # (1.2 - 0.6) / 0.2 up to binary floating-point rounding.
    assert worst_sector["worst_bin"]["pull"] == pytest.approx(3.0, rel=1e-12)
    # No other sector reports bin 7; each remaining sector's worst bin is
    # its own single bin (one bin per sector in the 9-bin grid).
    for key, sector in sectors.items():
        if key == "pi-|f_xsect|xF_transition|pT_high":
            continue
        assert sector["worst_bin"]["bin_index"] != 7
        index = sector["worst_bin"]["bin_index"]
        expected_pull = (values[index] - model[index]) / stat_errors[index]
        assert sector["worst_bin"]["pull"] == pytest.approx(expected_pull, rel=1e-12)
        assert sector["worst_bin"]["sigma"] == stat_errors[index]


# ---------------------------------------------------------------------------
# test_sector_decomposition_residual_mean
# ---------------------------------------------------------------------------


def test_sector_decomposition_residual_mean():
    """Hand-chosen pulls (d - m) / sigma; each sector's residual_mean is
    the hand-computed mean of its two bins, to < 1e-12."""
    pulls = (
        2.0, 0.0,  # (xF_central, pT_low)
        -1.0, 1.0,  # (xF_transition, pT_low)
        3.0, -1.0,  # (xF_forward, pT_low)
        0.5, -0.5,  # (xF_central, pT_mid)
        1.0, 2.0,  # (xF_transition, pT_mid)
        -2.0, 4.0,  # (xF_forward, pT_mid)
        -3.0, 1.0,  # (xF_central, pT_high)
        0.0, 0.0,  # (xF_transition, pT_high)
        -0.5, 0.5,  # (xF_forward, pT_high)
    )
    hand_means = {
        "K+|f_xsect|xF_central|pT_low": 1.0,
        "K+|f_xsect|xF_transition|pT_low": 0.0,
        "K+|f_xsect|xF_forward|pT_low": 1.0,
        "K+|f_xsect|xF_central|pT_mid": 0.0,
        "K+|f_xsect|xF_transition|pT_mid": 1.5,
        "K+|f_xsect|xF_forward|pT_mid": 1.0,
        "K+|f_xsect|xF_central|pT_high": -1.0,
        "K+|f_xsect|xF_transition|pT_high": 0.0,
        "K+|f_xsect|xF_forward|pT_high": 0.0,
    }
    d, model = _eighteen_bin_observable(pulls)
    decomposition = sector_decomposition(
        d, model, diagonal_covariance(d.stat_errors), None
    )
    assert decomposition["unassigned_bins"] == 0
    sectors = decomposition["sectors"]
    assert set(sectors) == set(hand_means)
    for key, expected in hand_means.items():
        assert abs(sectors[key]["residual_mean"] - expected) < 1e-12
        assert sectors[key]["n_bins"] == 2
    # The worst bin of a two-bin sector is the larger-|pull| one (ties to
    # the lowest index): the (xF_forward, pT_mid) sector holds pulls -2 and 4.
    worst = sectors["K+|f_xsect|xF_forward|pT_mid"]["worst_bin"]
    assert worst["bin_index"] == 11
    assert worst["pull"] == pytest.approx(4.0, rel=1e-12)


# ---------------------------------------------------------------------------
# test_write_sector_report_schema
# ---------------------------------------------------------------------------


def test_write_sector_report_schema(tmp_path):
    """The report carries the card's top-level block plus the sectors,
    with the per-species / per-observable / per-region totals consistent
    with the sector payload."""
    values = (1.0, 0.8, 0.5, 1.2, 0.9, 0.4, 1.1, 0.7, 0.3)
    model = (0.9, 0.85, 0.55, 1.1, 0.8, 0.45, 1.2, 0.6, 0.25)
    stat_errors = (0.1, 0.2, 0.15, 0.1, 0.2, 0.15, 0.1, 0.2, 0.15)
    sectors: dict[str, dict] = {}
    unassigned = 0
    total_chi2 = 0.0
    for species in ("pi-", "K+"):
        d = BinnedObservable(
            species=species,
            observable="f_xsect",
            bin_edges=((0.0, 0.1, 0.2, 0.3), (0.05, 0.5, 1.0, 1.5)),
            values=values,
            stat_errors=stat_errors,
        )
        decomp = sector_decomposition(d, model, diagonal_covariance(stat_errors), None)
        sectors.update(decomp["sectors"])
        unassigned += decomp["unassigned_bins"]
        chi2, _ = evaluate_likelihood(d, model, diagonal_covariance(stat_errors), None)
        total_chi2 += chi2
    decomposition = {"sectors": sectors, "unassigned_bins": unassigned}
    path = tmp_path / "sector_report.json"
    write_sector_report(str(path), decomposition, total_chi2, dof=18)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {
        "schema",
        "total_chi2",
        "dof",
        "n_sectors",
        "unassigned_bins",
        "per_species_totals",
        "per_observable_totals",
        "per_region_totals",
        "sectors",
    }
    assert payload["total_chi2"] == pytest.approx(total_chi2, rel=1e-12)
    assert payload["dof"] == 18
    assert payload["n_sectors"] == 18
    assert payload["unassigned_bins"] == 0
    assert set(payload["per_species_totals"]) == {"pi-", "K+"}
    assert set(payload["per_observable_totals"]) == {"f_xsect"}
    assert set(payload["per_region_totals"]) == set(SECTOR_REGIONS)
    for region in SECTOR_REGIONS:
        lo, hi = SECTOR_REGIONS[region]
        assert math.isfinite(payload["per_region_totals"][region])
    per_sector_keys = {
        "chi2_data",
        "chi2_mc",
        "chi2_nuisance",
        "chi2_total",
        "fraction_of_total",
        "residual_mean",
        "worst_bin",
        "n_bins",
    }
    assert payload["sectors"] == sectors
    for sector in payload["sectors"].values():
        assert set(sector) == per_sector_keys
    assert sum(payload["per_species_totals"].values()) == pytest.approx(
        sum(s["chi2_total"] for s in payload["sectors"].values()), rel=1e-12
    )
    # The region marginals: the xF regions partition each observable's
    # sectors by their xF axis, so the three xF totals sum to the grand
    # total; the same holds for the pT axis.
    xf_total = sum(
        payload["per_region_totals"][name] for name in SECTOR_XF_REGION_ORDER
    )
    pt_total = sum(
        payload["per_region_totals"][name] for name in SECTOR_PT_REGION_ORDER
    )
    grand = sum(s["chi2_total"] for s in payload["sectors"].values())
    assert xf_total == pytest.approx(grand, rel=1e-12)
    assert pt_total == pytest.approx(grand, rel=1e-12)


# ---------------------------------------------------------------------------
# test_sector_flag_off_legacy_unchanged
# ---------------------------------------------------------------------------


def test_sector_flag_off_legacy_unchanged(tmp_path):
    """Flag off: two smoke runs write bit-identical staged evaluation
    reports and no sector_report.json is written."""
    out_a = tmp_path / "staged_a.json"
    out_b = tmp_path / "staged_b.json"
    argv_common = ["--config", str(SMOKE_CONFIG)]
    assert main(argv_common + ["--output", str(out_a)]) == 0
    assert main(argv_common + ["--output", str(out_b)]) == 0
    assert out_a.read_bytes() == out_b.read_bytes()
    assert not (tmp_path / SECTOR_REPORT_FILENAME).exists()
    assert not (tmp_path / "sector_report.json").exists()


# ---------------------------------------------------------------------------
# test_sector_flag_on_writes_report
# ---------------------------------------------------------------------------


@pytest.mark.external_data
def test_sector_flag_on_writes_report(tmp_path):
    """Flag on: sector_report.json is written beside the staged report
    with n_sectors == 9 x 6 for the six-species set and
    unassigned_bins == 0; the staged report bytes are unchanged."""
    out_off = tmp_path / "staged_off.json"
    out_on = tmp_path / "staged_on.json"
    argv_common = ["--config", str(SMOKE_CONFIG)]
    assert main(argv_common + ["--output", str(out_off)]) == 0
    assert main(argv_common + ["--output", str(out_on), "--sector-report"]) == 0
    # The flag adds only the sector report file; the staged report is
    # bit-identical in both flag states.
    assert out_on.read_bytes() == out_off.read_bytes()
    sector_path = tmp_path / SECTOR_REPORT_FILENAME
    assert sector_path.is_file()
    payload = json.loads(sector_path.read_text(encoding="utf-8"))
    assert payload["n_sectors"] == 9 * 6
    assert payload["unassigned_bins"] == 0
    species_keys = {
        "pi_minus",
        "pi_plus",
        "k_plus",
        "k_minus",
        "proton",
        "antiproton",
    }
    assert set(payload["per_species_totals"]) == species_keys
    assert set(payload["per_region_totals"]) == set(SECTOR_REGIONS)
    # 9 sectors per species-observable, every sector populated.
    for species in species_keys:
        keys = [
            key
            for key in payload["sectors"]
            if key.split("|", 4)[0] == species
        ]
        assert len(keys) == 9
        for key in keys:
            assert payload["sectors"][key]["n_bins"] >= 1
    # The partition is exact on the smoke path too: the sector chi2_total
    # sum reproduces the single-sample global chi2.
    sector_sum = sum(s["chi2_total"] for s in payload["sectors"].values())
    assert abs(sector_sum - payload["total_chi2"]) / payload["total_chi2"] < 1e-9
    # No parameter is fitted on the diagnostic path: dof is the windowed
    # bin count of the six-species set (229 + 230 + 153 + 148 + 135 + 122).
    assert payload["dof"] == 1017
