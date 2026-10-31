"""Coverage extension for the sector-wise deviance decomposition
(Round 1, card 8).

Task card ``TASK-20260911-r1-sector-wise-fit-reports``.
``tests/test_sector_report.py`` holds the six card-named deterministic
tests.  This file extends coverage of the same committed behavior
(``na49_mixing/fit_likelihood.py`` and the ``--sector-report`` /
``report.sector_wise`` switch of ``na49_mixing/staged_evaluation.py``
at the card commit) without touching either file:

- the pre-registered grid: ``SECTOR_REGIONS`` is exactly the card's
  six entries with the card's boundary values, and the three regions
  of each axis abut without a gap into the paper window;
- boundary handling beyond the committed helper pins: a fine grid over
  the paper window (4001 xF points, 14501 pT points) is assigned to
  exactly one region per axis at every point (no gap), the region
  sequence is monotone in the pre-registered order (no double
  assignment), and the abutting midpoints (0.10, 0.2 / 0.5, 1.0) and
  the window corners (0.0, 0.4 / 0.05, 1.5) land as committed; the
  same convention is pinned at the sector level on an observable
  whose per-bin midpoints sit exactly on the abutting boundaries and
  the window corners;
- partition with nuisances beyond the committed single-observable case:
  two bins per sector (18-bin grid) with an in-scope nuisance stays
  exact against ``evaluate_likelihood`` to < 1e-9 and books the
  penalty into exactly one sector per observable; an off-diagonal
  combined covariance with the nuisance in scope at the same time;
  two observables with per-species nuisances (one booking per
  observable, the combined partition exact); an out-of-scope nuisance
  shifts nothing but its penalty is still booked (the scope only
  controls the shift); with an unassigned bin present the penalty is
  booked into the lowest-index *assigned* bin's sector and the sector
  sum reproduces the global chi2 minus the unassigned bin's shifted
  quadratic contribution;
- an observable with no assigned bin at all: the nine sectors are
  still emitted (empty, all-zero, ``worst_bin: null``), the penalty
  stays unbooked in every sector, and the sector sum is 0.0 while the
  global chi2 remains the value of record;
- per-region marginals of ``write_sector_report``: each of the six
  named region totals equals the hand sum of the ``chi2_total`` of the
  sectors sharing that region to < 1e-9 (the committed test pins only
  the per-axis sums against the grand total);
- worst-bin tie handling: two bins with equal ``|pull|`` report the
  lowest bin index (first-max semantics), and a sector with all-zero
  residuals reports ``worst_bin`` at its lowest bin with ``pull ==
  0.0`` (the empty-sector ``worst_bin: null`` representation is
  pinned here through the report bytes);
- ``fraction_of_total`` beyond the committed relative pin: absolute
  sum-to-1 to < 1e-9 for one observable, empty sectors carry exactly
  0.0, the denominator is the observable's own -2 ln L (the quadratic
  plus the booked penalty, so the fractions still sum to 1 with a
  nuisance in scope), and the committed ``observable_total > 0.0``
  guard makes every fraction 0.0 on a zero-chi2 observable;
- ``write_sector_report`` bytes: the top-level keys include the
  committed ``"schema": "na49_mixing.sector_report/v1"``, the
  per-sector payload carries exactly the eight committed fields, a
  second write over the same path is byte-identical (deterministic
  serialization), and two writes to two paths produce identical bytes;
- the switch: the private ``_sector_wise_requested`` helper (the flag
  beats an explicit ``report.sector_wise: false``, the config key
  alone triggers, and a non-dict ``report`` block never triggers);
- the config-key trigger end to end (``external_data``): with
  ``report.sector_wise: true`` and the flag absent the sector report
  is written with the builder's pinned output, the flag beats an
  explicit ``sector_wise: false``, the two trigger paths produce
  byte-identical sector reports, and the legacy staged report differs
  from the flag-off baseline only in ``configuration_hash`` (the
  sha256 of the modified config: the config key adds no field of its
  own to the legacy report).

Committed behavior noted, not fixed:

- ``sector_decomposition`` emits all nine sector keys even when the
  observable has no assigned bin (the module docstring's "emits no
  sector" sentence reads as "no populated sector"); the penalty stays
  unbooked and the sector sum (0.0) does not reproduce the global
  chi2, which the driver keeps as the value of record.
- With unassigned bins the ``fraction_of_total`` sum is below 1 (the
  denominator stays the observable's full -2 ln L), and a zero-chi2
  observable yields all-zero fractions; both hold on the paper window
  by construction (``unassigned_bins == 0``, chi2 > 0).
- The per-region marginal sums agree with the hand sums only to
  floating-point addition order (~1e-15 absolute on O(1) chi2 values);
  the pins use < 1e-9.

Hermetic except the one ``external_data``-marked end-to-end test (it
drives the staged evaluation, which reads the SPS tables through
``fit.binned``); no new config files (the modified smoke configs are
written to the test's ``tmp_path``).
"""

from __future__ import annotations

import hashlib
import json
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
    diagonal_covariance,
    evaluate_likelihood,
    sector_decomposition,
    write_sector_report,
)
from na49_mixing.staged_evaluation import (
    SECTOR_REPORT_FILENAME,
    _sector_wise_requested,
    main,
)

NESTED_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = NESTED_ROOT / "config" / "staged_excitation_smoke.json"

XF_REGION_ORDER = SECTOR_XF_REGION_ORDER
PT_REGION_ORDER = SECTOR_PT_REGION_ORDER

# The card's pre-registered grid, verbatim.
CARD_SECTOR_REGIONS = {
    "xF_forward": (0.2, 0.4),
    "xF_central": (0.0, 0.10),
    "xF_transition": (0.10, 0.2),
    "pT_low": (0.05, 0.5),
    "pT_mid": (0.5, 1.0),
    "pT_high": (1.0, 1.5),
}


# ---------------------------------------------------------------------------
# Hand-constructed observables (no event generation)
# ---------------------------------------------------------------------------


def _eighteen_bin_observable(
    pulls: tuple[float, ...],
    species: str = "K+",
) -> tuple[BinnedObservable, tuple[float, ...]]:
    """18 bins = 6 xF bins x 3 pT bins, two xF bins per xF region.

    xF midpoints 0.00, 0.05 (central), 0.10, 0.15 (transition), 0.20,
    0.25 (forward); pT midpoints 0.1, 0.7, 1.4 (low, mid, high).  Each
    sector holds exactly two bins (bin ``i`` at ``ix = i % 6``,
    ``ip = i // 6``), so the xF_central x pT_low sector holds bins 0, 1
    and the lowest-index assigned bin is bin 0.  The model is 0.5 per
    bin, the combined error is 0.1 per bin (data-only diagonal), and
    the data values realize the hand-chosen pulls ``(d - m) / sigma``.
    """
    assert len(pulls) == 18
    stat_error = 0.1
    model = tuple(0.5 for _ in pulls)
    values = tuple(0.5 + pull * stat_error for pull in pulls)
    return (
        BinnedObservable(
            species=species,
            observable="f_xsect",
            bin_edges=(
                (-0.025, 0.025, 0.075, 0.125, 0.175, 0.225, 0.275),
                (0.0, 0.2, 1.2, 1.6),
            ),
            values=values,
            stat_errors=tuple(stat_error for _ in pulls),
            cov=None,
            normalization_nuisance=None,
        ),
        model,
    )


def _offdiag_squared(diag_values, coupling: float) -> np.ndarray:
    """Symmetric covariance: the diagonal of ``diag_values**2`` plus a
    constant ``coupling`` off-diagonal structure (positive definite for
    the couplings used here)."""
    arr = np.asarray(diag_values, dtype=float)
    matrix = np.diag(arr ** 2)
    off = coupling * np.outer(arr, arr)
    return matrix + np.triu(off, 1) + np.tril(off, -1)


# ---------------------------------------------------------------------------
# The pre-registered grid
# ---------------------------------------------------------------------------


def test_sector_regions_is_the_pre_registered_grid():
    """SECTOR_REGIONS is exactly the card's six entries with the card's
    boundary values, and the three regions of each axis abut without a
    gap into the paper window."""
    assert SECTOR_REGIONS == CARD_SECTOR_REGIONS
    for order, window in (
        (XF_REGION_ORDER, (0.0, 0.4)),
        (PT_REGION_ORDER, (0.05, 1.5)),
    ):
        assert SECTOR_REGIONS[order[0]][0] == window[0]
        assert SECTOR_REGIONS[order[-1]][1] == window[1]
        for name_a, name_b in zip(order, order[1:]):
            assert SECTOR_REGIONS[name_a][1] == SECTOR_REGIONS[name_b][0]
        for name in order:
            lo, hi = SECTOR_REGIONS[name]
            assert lo < hi


# ---------------------------------------------------------------------------
# Boundary handling: helper level, fine grid, sector level
# ---------------------------------------------------------------------------


def test_sector_boundary_membership_fine_grid():
    """Half-open ``[lo, hi)`` on every region of an axis, with the last
    region of the axis closed ``[lo, hi]``, pinned on a focused table of
    midpoint values and on a fine grid over the whole paper window."""
    # The exact abutting boundaries land in the higher region.
    assert _sector_region_of(0.10, XF_REGION_ORDER) == "xF_transition"
    assert _sector_region_of(0.2, XF_REGION_ORDER) == "xF_forward"
    assert _sector_region_of(0.5, PT_REGION_ORDER) == "pT_mid"
    assert _sector_region_of(1.0, PT_REGION_ORDER) == "pT_high"
    # The window corners are assigned: the low corner of the first
    # region, the closed high corner of the last.
    assert _sector_region_of(0.0, XF_REGION_ORDER) == "xF_central"
    assert _sector_region_of(0.4, XF_REGION_ORDER) == "xF_forward"
    assert _sector_region_of(0.05, PT_REGION_ORDER) == "pT_low"
    assert _sector_region_of(1.5, PT_REGION_ORDER) == "pT_high"
    # Just inside the window is assigned; just outside is not.
    assert _sector_region_of(-1e-9, XF_REGION_ORDER) is None
    assert _sector_region_of(0.05 - 1e-9, PT_REGION_ORDER) is None
    assert _sector_region_of(0.4 + 1e-9, XF_REGION_ORDER) is None
    assert _sector_region_of(1.5 + 1e-9, PT_REGION_ORDER) is None
    assert _sector_region_of(0.1 - 1e-12, XF_REGION_ORDER) == "xF_central"
    assert _sector_region_of(0.2 + 1e-12, XF_REGION_ORDER) == "xF_forward"
    assert _sector_region_of(0.5 - 1e-12, PT_REGION_ORDER) == "pT_low"
    assert _sector_region_of(1.0 - 1e-12, PT_REGION_ORDER) == "pT_mid"

    def expected_xf(v: float) -> str:
        if v < 0.10:
            return "xF_central"
        if v < 0.2:
            return "xF_transition"
        return "xF_forward"

    def expected_pt(v: float) -> str:
        if v < 0.5:
            return "pT_low"
        if v < 1.0:
            return "pT_mid"
        return "pT_high"

    # Fine grid over the paper window: every midpoint is assigned to
    # exactly one region (no gap), the region sequence is monotone in
    # the pre-registered order (no double assignment), and every region
    # actually occurs on the grid.
    xf_grid = np.linspace(0.0, 0.4, 4001)
    xf_regions: list[str] = []
    for v in xf_grid:
        region = _sector_region_of(float(v), XF_REGION_ORDER)
        assert region is not None
        xf_regions.append(region)
    assert all(
        region == expected_xf(float(v)) for v, region in zip(xf_grid, xf_regions)
    )
    xf_rank = {name: i for i, name in enumerate(XF_REGION_ORDER)}
    assert all(
        xf_rank[a] <= xf_rank[b] for a, b in zip(xf_regions, xf_regions[1:])
    )
    for name in XF_REGION_ORDER:
        assert xf_regions.count(name) > 0

    pt_grid = np.linspace(0.05, 1.5, 14501)
    pt_regions: list[str] = []
    for v in pt_grid:
        region = _sector_region_of(float(v), PT_REGION_ORDER)
        assert region is not None
        pt_regions.append(region)
    assert all(
        region == expected_pt(float(v)) for v, region in zip(pt_grid, pt_regions)
    )
    pt_rank = {name: i for i, name in enumerate(PT_REGION_ORDER)}
    assert all(
        pt_rank[a] <= pt_rank[b] for a, b in zip(pt_regions, pt_regions[1:])
    )
    for name in PT_REGION_ORDER:
        assert pt_regions.count(name) > 0


def test_sector_boundary_midpoints_at_the_sector_level():
    """The half-open convention pinned at the sector-assignment level:
    an observable whose per-bin midpoints sit exactly on the abutting
    boundaries (0.10, 0.2 / 0.5, 1.0) and the window corners (0.0, 0.4
    / 0.05, 1.5) is fully assigned with each bin in the expected
    sector.  This also pins the per-bin (sparse) edge form: every bin
    carries the edges of its own cell."""
    # Bin midpoints (xF, pT): (0.10, 0.50), (0.20, 1.00), (0.00, 0.05),
    # (0.40, 1.50), (0.10, 1.00), (0.20, 0.50).
    xf_cells = (
        (0.05, 0.15), (0.15, 0.25), (-0.025, 0.025),
        (0.375, 0.425), (0.05, 0.15), (0.15, 0.25),
    )
    pt_cells = (
        (0.4, 0.6), (0.9, 1.1), (0.0, 0.1),
        (1.45, 1.55), (0.9, 1.1), (0.4, 0.6),
    )
    d = BinnedObservable(
        species="K-",
        observable="f_xsect",
        bin_edges=(
            tuple(v for cell in xf_cells for v in cell),
            tuple(v for cell in pt_cells for v in cell),
        ),
        values=(1.0, 1.0, 0.8, 0.9, 1.1, 1.0),
        stat_errors=(0.1, 0.1, 0.2, 0.2, 0.2, 0.1),
        cov=None,
        normalization_nuisance=None,
    )
    model = (0.9, 0.9, 0.8, 0.8, 0.9, 0.9)
    v = diagonal_covariance(d.stat_errors)
    decomposition = sector_decomposition(d, model, v, None)
    assert decomposition["unassigned_bins"] == 0
    sectors = decomposition["sectors"]
    assert sum(s["n_bins"] for s in sectors.values()) == 6
    # Bins 1 and 3 share one sector (both midpoints at (0.20 / 0.40,
    # 1.00 / 1.50)); bin 1 has the larger |pull| (1.0 vs 0.5) and is the
    # sector's worst bin.
    forward_high = sectors["K-|f_xsect|xF_forward|pT_high"]
    assert forward_high["n_bins"] == 2
    assert forward_high["worst_bin"]["bin_index"] == 1
    for key, n_bins, worst in (
        ("K-|f_xsect|xF_transition|pT_mid", 1, 0),
        ("K-|f_xsect|xF_central|pT_low", 1, 2),
        ("K-|f_xsect|xF_transition|pT_high", 1, 4),
        ("K-|f_xsect|xF_forward|pT_mid", 1, 5),
    ):
        assert sectors[key]["n_bins"] == n_bins
        assert sectors[key]["worst_bin"]["bin_index"] == worst
    # Every other sector is empty.
    assert sum(1 for s in sectors.values() if s["n_bins"] == 0) == 9 - 5

    # The flat provenance form and malformed edge counts are rejected.
    flat = BinnedObservable(
        species="K-",
        observable="f_xsect",
        bin_edges=(0.05, 0.15, 0.15, 0.25, -0.025, 0.025),
        values=(1.0, 1.0, 0.8, 0.9, 1.1, 1.0),
        stat_errors=(0.1, 0.1, 0.2, 0.2, 0.2, 0.1),
    )
    with pytest.raises(ValueError):
        sector_decomposition(flat, model, v, None)
    # Neither form fits: the grid form needs n_x * n_p == 6 (3 x 3 = 9
    # does not) and the per-bin form needs 2 * 6 edges per axis.
    malformed = BinnedObservable(
        species="K-",
        observable="f_xsect",
        bin_edges=((0.0, 0.1, 0.2, 0.3), (0.0, 0.2, 1.2, 1.6)),
        values=(1.0, 1.0, 0.8, 0.9, 1.1, 1.0),
        stat_errors=(0.1, 0.1, 0.2, 0.2, 0.2, 0.1),
    )
    with pytest.raises(ValueError):
        sector_decomposition(malformed, model, v, None)


# ---------------------------------------------------------------------------
# Partition with nuisances
# ---------------------------------------------------------------------------


def test_sector_partition_with_nuisances():
    """The sector partition stays exact against ``evaluate_likelihood``
    with nuisances in scope: the penalty is booked once per observable,
    into the sector of the lowest-index assigned bin, in every
    combination pinned here."""
    pulls = (
        2.0, -0.5, 1.0, -1.0, 3.0, 0.5,
        -0.25, 1.5, -2.0, 0.75, 4.0, -1.5,
        0.25, -3.5, 1.25, -0.75, 2.5, -2.25,
    )
    d, model = _eighteen_bin_observable(pulls)
    v = diagonal_covariance(d.stat_errors)
    in_scope = NormalizationNuisance(
        name="sps_kplus_norm", eta=0.01, sigma_eta=0.02, table_scope="K+"
    )
    chi2_global, _ = evaluate_likelihood(d, model, v, [in_scope])
    sectors = sector_decomposition(d, model, v, [in_scope])["sectors"]
    assert abs(sum(s["chi2_total"] for s in sectors.values()) - chi2_global) / chi2_global < 1e-9
    penalty_sectors = [k for k, s in sectors.items() if s["chi2_nuisance"] != 0.0]
    assert penalty_sectors == ["K+|f_xsect|xF_central|pT_low"]
    assert sectors[penalty_sectors[0]]["chi2_nuisance"] == pytest.approx(0.25, rel=1e-12)
    for sector in sectors.values():
        assert (
            sector["chi2_data"] + sector["chi2_mc"] + sector["chi2_nuisance"]
        ) == pytest.approx(sector["chi2_total"], rel=1e-12)

    # Off-diagonal combined covariance with the nuisance in scope at the
    # same time: the row-restricted partition stays exact.
    values = (1.0, 0.8, 0.5, 1.2, 0.9, 0.4, 1.1, 0.7, 0.3)
    model9 = (0.9, 0.85, 0.55, 1.1, 0.8, 0.45, 1.2, 0.6, 0.25)
    stat_errors = (0.1, 0.2, 0.15, 0.1, 0.2, 0.15, 0.1, 0.2, 0.15)
    sigma_mc = (0.05, 0.1, 0.05, 0.05, 0.1, 0.05, 0.05, 0.1, 0.05)
    d9 = BinnedObservable(
        species="pi-",
        observable="f_xsect",
        bin_edges=((0.0, 0.1, 0.2, 0.3), (0.05, 0.5, 1.0, 1.5)),
        values=values,
        stat_errors=stat_errors,
        cov=tuple(float(v) for v in _offdiag_squared(stat_errors, 0.2)[np.triu_indices(9)]),
        normalization_nuisance=None,
    )
    v9 = tuple(
        float(v)
        for v in (_offdiag_squared(stat_errors, 0.2) + _offdiag_squared(sigma_mc, 0.1))[
            np.triu_indices(9)
        ]
    )
    nuisance9 = NormalizationNuisance(
        name="sps_piminus_norm", eta=0.01, sigma_eta=0.02, table_scope="pi-"
    )
    chi2_9, _ = evaluate_likelihood(d9, model9, v9, [nuisance9])
    sectors9 = sector_decomposition(d9, model9, v9, [nuisance9])["sectors"]
    assert (
        abs(sum(s["chi2_total"] for s in sectors9.values()) - chi2_9) / chi2_9 < 1e-9
    )
    penalty_9 = [k for k, s in sectors9.items() if s["chi2_nuisance"] != 0.0]
    assert penalty_9 == ["pi-|f_xsect|xF_central|pT_low"]

    # Two observables with per-species nuisances: one booking per
    # observable, the combined partition across both exact.
    pulls_p = (
        0.75, -1.25, 1.5, -0.5, 2.25, -3.0,
        1.0, -2.0, 0.5, -1.5, 3.5, -0.25,
        -1.0, 0.5, 2.0, -2.5, 1.75, -0.75,
    )
    d_p, model_p = _eighteen_bin_observable(pulls_p, species="proton")
    nuisances = [
        NormalizationNuisance(
            name="sps_kplus_norm", eta=0.01, sigma_eta=0.02, table_scope="K+"
        ),
        NormalizationNuisance(
            name="sps_proton_norm", eta=0.03, sigma_eta=0.01, table_scope="proton"
        ),
    ]
    combined_global = 0.0
    combined_sectors = 0.0
    for obs, model_o, species in ((d, model, "K+"), (d_p, model_p, "proton")):
        v_o = diagonal_covariance(obs.stat_errors)
        chi2_o, _ = evaluate_likelihood(obs, model_o, v_o, nuisances)
        sectors_o = sector_decomposition(obs, model_o, v_o, nuisances)["sectors"]
        assert abs(sum(s["chi2_total"] for s in sectors_o.values()) - chi2_o) / chi2_o < 1e-9
        booked = [k for k, s in sectors_o.items() if s["chi2_nuisance"] != 0.0]
        assert booked == [f"{species}|f_xsect|xF_central|pT_low"]
        combined_global += chi2_o
        combined_sectors += sum(s["chi2_total"] for s in sectors_o.values())
    assert abs(combined_sectors - combined_global) / combined_global < 1e-9

    # An out-of-scope nuisance shifts nothing but its penalty is still
    # booked into the lowest-index assigned bin's sector (the scope only
    # controls the shift), and the partition against evaluate_likelihood
    # (which pays the same penalty) stays exact.
    out_of_scope = NormalizationNuisance(
        name="sps_piminus_norm", eta=0.02, sigma_eta=0.01, table_scope="pi-"
    )
    chi2_oos, _ = evaluate_likelihood(d, model, v, [out_of_scope])
    sectors_oos = sector_decomposition(d, model, v, [out_of_scope])["sectors"]
    assert (
        abs(sum(s["chi2_total"] for s in sectors_oos.values()) - chi2_oos) / chi2_oos < 1e-9
    )
    booked_oos = [k for k, s in sectors_oos.items() if s["chi2_nuisance"] != 0.0]
    assert booked_oos == ["K+|f_xsect|xF_central|pT_low"]
    assert sectors_oos[booked_oos[0]]["chi2_nuisance"] == pytest.approx(4.0, rel=1e-12)


def test_sector_partition_with_unassigned_bins():
    """With a bin outside the paper window: the penalty is booked into
    the sector of the lowest-index *assigned* bin (not the unassigned
    one), and the sector sum reproduces the global chi2 minus the
    unassigned bin's shifted quadratic contribution."""
    midpoints = (
        (-0.05, 0.2), (0.05, 0.2), (0.15, 0.2),
        (0.25, 0.7), (0.05, 0.7), (0.15, 0.7),
        (0.25, 1.2), (0.05, 1.2), (0.15, 1.2),
    )
    d = BinnedObservable(
        species="p",
        observable="f_xsect",
        bin_edges=(
            tuple(v for xm, _ in midpoints for v in (xm - 0.025, xm + 0.025)),
            tuple(v for _, pm in midpoints for v in (pm - 0.05, pm + 0.05)),
        ),
        values=(1.0, 0.8, 0.5, 1.2, 0.9, 0.4, 1.1, 0.7, 0.3),
        stat_errors=(0.1, 0.2, 0.15, 0.1, 0.2, 0.15, 0.1, 0.2, 0.15),
        cov=None,
        normalization_nuisance=None,
    )
    model = (0.9, 0.85, 0.55, 1.1, 0.8, 0.45, 1.2, 0.6, 0.25)
    v = diagonal_covariance(d.stat_errors)
    nuisance = NormalizationNuisance(
        name="sps_proton_norm", eta=0.02, sigma_eta=0.01, table_scope="p"
    )
    chi2_global, _ = evaluate_likelihood(d, model, v, [nuisance])
    decomposition = sector_decomposition(d, model, v, [nuisance])
    assert decomposition["unassigned_bins"] == 1
    sectors = decomposition["sectors"]
    # Bin 0 is unassigned; the penalty lands in bin 1's sector.
    booked = [k for k, s in sectors.items() if s["chi2_nuisance"] != 0.0]
    assert booked == ["p|f_xsect|xF_central|pT_low"]
    # Data-only diagonal: the unassigned bin's shifted contribution is
    # its hand-computed squared residual.
    r0 = d.values[0] - model[0] - nuisance.eta * d.values[0]
    expected = chi2_global - r0 * r0 / (d.stat_errors[0] ** 2)
    sector_sum = sum(s["chi2_total"] for s in sectors.values())
    assert abs(sector_sum - expected) < 1e-9


def test_all_unassigned_observable_leaves_penalty_unbooked():
    """An observable with no assigned bin at all: the nine sectors are
    still emitted (empty, ``worst_bin: null``), the sector sum is 0.0
    while the global chi2 remains the value of record, and the in-scope
    penalty stays unbooked in every sector."""
    d = BinnedObservable(
        species="p",
        observable="f_xsect",
        bin_edges=(
            (-0.15, -0.05, -0.25, -0.15, -0.35, -0.25),
            (2.0, 2.2, 2.2, 2.4, 2.4, 2.6),
        ),
        values=(1.0, 0.8, 0.5),
        stat_errors=(0.1, 0.2, 0.15),
        cov=None,
        normalization_nuisance=None,
    )
    model = (0.9, 0.85, 0.55)
    v = diagonal_covariance((0.1, 0.2, 0.15))
    chi2_global, _ = evaluate_likelihood(d, model, v, None)
    decomposition = sector_decomposition(d, model, v, None)
    assert decomposition["unassigned_bins"] == 3
    sectors = decomposition["sectors"]
    assert len(sectors) == 9
    assert all(
        s["n_bins"] == 0 and s["worst_bin"] is None and s["chi2_total"] == 0.0
        for s in sectors.values()
    )
    assert sum(s["chi2_total"] for s in sectors.values()) == 0.0
    assert chi2_global > 0.0

    # With an in-scope nuisance the caller's -2 ln L carries the penalty
    # (and the shifted quadratic); no sector may book the penalty.
    nuisance = NormalizationNuisance(
        name="sps_proton_norm", eta=0.02, sigma_eta=0.01, table_scope="p"
    )
    _, components_n = evaluate_likelihood(d, model, v, [nuisance])
    assert components_n["chi2_nuisance"] == pytest.approx(4.0, rel=1e-12)
    decomposition_n = sector_decomposition(d, model, v, [nuisance])
    assert not any(
        s["chi2_nuisance"] != 0.0 for s in decomposition_n["sectors"].values()
    )
    assert sum(s["chi2_total"] for s in decomposition_n["sectors"].values()) == 0.0


# ---------------------------------------------------------------------------
# Per-region marginals
# ---------------------------------------------------------------------------


def test_per_region_marginals_match_hand_sums(tmp_path):
    """Each of the six named region totals equals the hand sum of the
    chi2_total of the sectors sharing that region to < 1e-9, and the
    three xF (resp. pT) marginals sum to the grand total."""
    pulls = (
        2.0, -0.5, 1.0, -1.0, 3.0, 0.5,
        -0.25, 1.5, -2.0, 0.75, 4.0, -1.5,
        0.25, -3.5, 1.25, -0.75, 2.5, -2.25,
    )
    d, model = _eighteen_bin_observable(pulls)
    v = diagonal_covariance(d.stat_errors)
    decomposition = sector_decomposition(d, model, v, None)
    chi2_global, _ = evaluate_likelihood(d, model, v, None)
    path = tmp_path / "sector_report.json"
    write_sector_report(str(path), decomposition, chi2_global, dof=18)
    payload = json.loads(path.read_text(encoding="utf-8"))
    sectors = payload["sectors"]
    for region in SECTOR_REGIONS:
        index = 2 if region.startswith("xF") else 3
        hand = sum(
            s["chi2_total"]
            for key, s in sectors.items()
            if key.split("|")[index] == region
        )
        assert abs(payload["per_region_totals"][region] - hand) < 1e-9
    grand = sum(s["chi2_total"] for s in sectors.values())
    xf_sum = sum(payload["per_region_totals"][r] for r in XF_REGION_ORDER)
    pt_sum = sum(payload["per_region_totals"][r] for r in PT_REGION_ORDER)
    assert abs(xf_sum - grand) < 1e-9
    assert abs(pt_sum - grand) < 1e-9


# ---------------------------------------------------------------------------
# Worst-bin tie handling
# ---------------------------------------------------------------------------


def test_worst_bin_tie_break_and_zero_residual_sector():
    """Two bins with equal |pull| report the lowest bin index (first-max
    semantics, with the sign of that bin's pull); a sector with all-zero
    residuals reports its worst bin at the sector's lowest bin with
    pull 0.0."""
    pulls = (
        3.0, -3.0, 0.0, 0.0, 1.0, -0.9,
        0.25, -0.5, 0.5, -0.75, 0.75, -0.4,
        0.1, -0.2, 0.2, -0.1, 0.3, -0.3,
    )
    d, model = _eighteen_bin_observable(pulls)
    sectors = sector_decomposition(
        d, model, diagonal_covariance(d.stat_errors), None
    )["sectors"]
    tied = sectors["K+|f_xsect|xF_central|pT_low"]
    assert tied["n_bins"] == 2
    assert tied["worst_bin"]["bin_index"] == 0
    assert tied["worst_bin"]["pull"] == pytest.approx(3.0, rel=1e-12)
    assert tied["worst_bin"]["d"] == d.values[0]
    assert tied["worst_bin"]["m"] == model[0]
    assert tied["worst_bin"]["sigma"] == d.stat_errors[0]
    zero = sectors["K+|f_xsect|xF_transition|pT_low"]
    assert zero["n_bins"] == 2
    assert zero["worst_bin"]["bin_index"] == 2
    assert zero["worst_bin"]["pull"] == 0.0
    assert zero["worst_bin"]["d"] == zero["worst_bin"]["m"] == 0.5
    assert zero["worst_bin"]["sigma"] == 0.1
    assert zero["residual_mean"] == 0.0


# ---------------------------------------------------------------------------
# fraction_of_total
# ---------------------------------------------------------------------------


def test_fraction_of_total_and_empty_sectors():
    """The sector fractions of one observable sum to 1 in absolute value
    to < 1e-9 (denominating on the observable's own -2 ln L, including
    a booked penalty); an empty sector has fraction 0.0; a zero-chi2
    observable yields all-zero fractions (the observable_total > 0.0
    guard)."""
    pulls = (
        2.0, -0.5, 1.0, -1.0, 3.0, 0.5,
        -0.25, 1.5, -2.0, 0.75, 4.0, -1.5,
        0.25, -3.5, 1.25, -0.75, 2.5, -2.25,
    )
    d, model = _eighteen_bin_observable(pulls)
    v = diagonal_covariance(d.stat_errors)
    chi2_global, _ = evaluate_likelihood(d, model, v, None)
    sectors = sector_decomposition(d, model, v, None)["sectors"]
    assert abs(sum(s["fraction_of_total"] for s in sectors.values()) - 1.0) < 1e-9
    for sector in sectors.values():
        assert sector["fraction_of_total"] == pytest.approx(
            sector["chi2_total"] / chi2_global, rel=1e-12
        )

    # 6-bin grid (3 xF x 2 pT): the three pT_high sectors are empty.
    d6 = BinnedObservable(
        species="pi-",
        observable="f_xsect",
        bin_edges=((0.0, 0.1, 0.2, 0.3), (0.0, 0.3, 0.9)),
        values=(1.0, 0.8, 0.5, 1.2, 0.9, 0.4),
        stat_errors=(0.1, 0.2, 0.15, 0.1, 0.2, 0.15),
        cov=None,
        normalization_nuisance=None,
    )
    model6 = (0.9, 0.85, 0.55, 1.1, 0.8, 0.45)
    v6 = diagonal_covariance(d6.stat_errors)
    decomposition6 = sector_decomposition(d6, model6, v6, None)
    sectors6 = decomposition6["sectors"]
    assert len(sectors6) == 9
    empty = sorted(k for k, s in sectors6.items() if s["n_bins"] == 0)
    assert empty == sorted(f"pi-|f_xsect|{xf}|pT_high" for xf in XF_REGION_ORDER)
    for key in empty:
        sector = sectors6[key]
        assert sector["chi2_data"] == 0.0
        assert sector["chi2_mc"] == 0.0
        assert sector["chi2_nuisance"] == 0.0
        assert sector["chi2_total"] == 0.0
        assert sector["fraction_of_total"] == 0.0
        assert sector["residual_mean"] == 0.0
        assert sector["worst_bin"] is None
    populated = [s for s in sectors6.values() if s["n_bins"] > 0]
    assert len(populated) == 6
    assert abs(sum(s["fraction_of_total"] for s in populated) - 1.0) < 1e-9

    # With a booked penalty the denominator is the observable's own
    # -2 ln L (quadratic + penalty); the fractions still sum to 1.
    nuisance = NormalizationNuisance(
        name="sps_pi_minus_norm", eta=0.01, sigma_eta=0.02, table_scope="pi-"
    )
    sectors6n = sector_decomposition(d6, model6, v6, [nuisance])["sectors"]
    assert abs(sum(s["fraction_of_total"] for s in sectors6n.values()) - 1.0) < 1e-9

    # Zero-chi2 observable (d == m exactly, no nuisance): the
    # observable_total > 0.0 guard makes every fraction 0.0.
    d0 = BinnedObservable(
        species="pi+",
        observable="f_xsect",
        bin_edges=((0.0, 0.1, 0.2, 0.3), (0.05, 0.5, 1.0, 1.5)),
        values=(1.0, 0.8, 0.5, 1.2, 0.9, 0.4, 1.1, 0.7, 0.3),
        stat_errors=(0.1, 0.2, 0.15, 0.1, 0.2, 0.15, 0.1, 0.2, 0.15),
        cov=None,
        normalization_nuisance=None,
    )
    chi2_0, _ = evaluate_likelihood(
        d0, d0.values, diagonal_covariance(d0.stat_errors), None
    )
    assert chi2_0 == 0.0
    sectors0 = sector_decomposition(
        d0, d0.values, diagonal_covariance(d0.stat_errors), None
    )["sectors"]
    assert all(s["fraction_of_total"] == 0.0 for s in sectors0.values())


# ---------------------------------------------------------------------------
# write_sector_report: schema and deterministic bytes
# ---------------------------------------------------------------------------


def test_write_sector_report_schema_and_deterministic_bytes(tmp_path):
    """The report carries the committed top-level keys (with the
    pinned schema) and per-sector fields; a second write over the same
    path and a second write to another path are byte-identical
    (deterministic serialization); empty sectors serialize with
    ``worst_bin: null``."""
    pi_values = (1.0, 0.8, 0.5, 1.2, 0.9, 0.4, 1.1, 0.7, 0.3)
    pi_model = (0.9, 0.85, 0.55, 1.1, 0.8, 0.45, 1.2, 0.6, 0.25)
    pi_errors = (0.1, 0.2, 0.15, 0.1, 0.2, 0.15, 0.1, 0.2, 0.15)
    k_values = (1.0, 0.8, 0.5, 1.2, 0.9, 0.4)
    k_model = (0.9, 0.85, 0.55, 1.1, 0.8, 0.45)
    k_errors = (0.1, 0.2, 0.15, 0.1, 0.2, 0.15)
    sectors: dict[str, dict] = {}
    unassigned = 0
    total_chi2 = 0.0
    for species, values, model, stat_errors, edges in (
        ("pi-", pi_values, pi_model, pi_errors, ((0.0, 0.1, 0.2, 0.3), (0.05, 0.5, 1.0, 1.5))),
        ("K+", k_values, k_model, k_errors, ((0.0, 0.1, 0.2, 0.3), (0.0, 0.3, 0.9))),
    ):
        obs = BinnedObservable(
            species=species,
            observable="f_xsect",
            bin_edges=edges,
            values=values,
            stat_errors=stat_errors,
            cov=None,
            normalization_nuisance=None,
        )
        v = diagonal_covariance(stat_errors)
        decomposition = sector_decomposition(obs, model, v, None)
        sectors.update(decomposition["sectors"])
        unassigned += decomposition["unassigned_bins"]
        chi2, _ = evaluate_likelihood(obs, model, v, None)
        total_chi2 += chi2
    decomposition = {"sectors": sectors, "unassigned_bins": unassigned}

    path = tmp_path / "sector_report.json"
    write_sector_report(str(path), decomposition, total_chi2, dof=15)
    first = path.read_bytes()
    write_sector_report(str(path), decomposition, total_chi2, dof=15)
    assert path.read_bytes() == first
    write_sector_report(str(tmp_path / "other.json"), decomposition, total_chi2, dof=15)
    assert (tmp_path / "other.json").read_bytes() == first

    payload = json.loads(first.decode("utf-8"))
    assert payload["schema"] == "na49_mixing.sector_report/v1"
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
    for sector in payload["sectors"].values():
        assert set(sector) == per_sector_keys
    assert payload["n_sectors"] == 18
    assert payload["unassigned_bins"] == 0
    assert payload["dof"] == 15
    assert payload["total_chi2"] == pytest.approx(total_chi2, rel=1e-12)
    # The three empty K+ pT_high sectors serialize with worst_bin null.
    nulls = [k for k, s in payload["sectors"].items() if s["worst_bin"] is None]
    assert nulls == sorted(f"K+|f_xsect|{xf}|pT_high" for xf in XF_REGION_ORDER)
    for key in nulls:
        assert payload["sectors"][key]["n_bins"] == 0


# ---------------------------------------------------------------------------
# The sector-report switch
# ---------------------------------------------------------------------------


def test_sector_wise_switch_combinations():
    """The private switch: the config key alone triggers, the flag beats
    an explicit config false, and a non-dict ``report`` block never
    triggers (both default off)."""
    assert _sector_wise_requested({}, False) is False
    assert _sector_wise_requested({"report": {"sector_wise": True}}, False) is True
    assert _sector_wise_requested({"report": {"sector_wise": False}}, False) is False
    assert _sector_wise_requested({"report": {}}, False) is False
    assert _sector_wise_requested({"report": "not-a-dict"}, False) is False
    assert _sector_wise_requested({}, True) is True
    assert _sector_wise_requested({"report": {"sector_wise": False}}, True) is True
    assert _sector_wise_requested({"report": {"sector_wise": True}}, True) is True


@pytest.mark.external_data
def test_sector_config_key_triggers_report(tmp_path):
    """End to end (reads the SPS tables through ``fit.binned``): with
    ``report.sector_wise: true`` and the flag absent the sector report
    is written with the builder's pinned output; the flag beats an
    explicit ``sector_wise: false``; the two trigger paths produce
    byte-identical sector reports; and the legacy staged report
    differs from the flag-off baseline only in ``configuration_hash``
    (the sha256 of the modified config: the config key adds no field of
    its own to the legacy report)."""
    base_cfg = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))

    def run(config: dict, extra: list[str], tag: str) -> Path:
        cfg_path = tmp_path / f"smoke_{tag}.json"
        cfg_path.write_text(json.dumps(config), encoding="utf-8")
        out = tmp_path / f"staged_{tag}.json"
        assert main(["--config", str(cfg_path), "--output", str(out)] + extra) == 0
        return out

    sector_path = tmp_path / SECTOR_REPORT_FILENAME

    # Baseline: flag off, no report key (the committed flag-off test
    # pins this run's properties; it is the reference for the legacy
    # report here).
    out_a = run(base_cfg, [], "a")
    assert not sector_path.exists()
    report_a = json.loads(out_a.read_text(encoding="utf-8"))

    # Config key on, flag absent.
    cfg_b = dict(base_cfg)
    cfg_b["report"] = {"sector_wise": True}
    out_b = run(cfg_b, [], "b")
    assert sector_path.is_file()
    sector_b_bytes = sector_path.read_bytes()
    payload_b = json.loads(sector_b_bytes.decode("utf-8"))
    assert payload_b["n_sectors"] == 9 * 6
    assert payload_b["unassigned_bins"] == 0
    assert payload_b["dof"] == 1017
    report_b = json.loads(out_b.read_text(encoding="utf-8"))
    assert set(report_a) == set(report_b)
    diff = [k for k in report_a if report_a[k] != report_b[k]]
    assert diff == ["configuration_hash"]
    canonical_b = json.dumps(cfg_b, sort_keys=True, separators=(",", ":"))
    assert report_b["configuration_hash"] == hashlib.sha256(
        canonical_b.encode("utf-8")
    ).hexdigest()

    # Explicit config false + the flag: the flag wins, and the sector
    # report is byte-identical to the config-triggered one (same builder
    # inputs: the seeded sample and the windowed tables).
    cfg_c = dict(base_cfg)
    cfg_c["report"] = {"sector_wise": False}
    out_c = run(cfg_c, ["--sector-report"], "c")
    assert sector_path.is_file()
    assert sector_path.read_bytes() == sector_b_bytes
    report_c = json.loads(out_c.read_text(encoding="utf-8"))
    diff_c = [k for k in report_a if report_a[k] != report_c[k]]
    assert diff_c == ["configuration_hash"]
    canonical_c = json.dumps(cfg_c, sort_keys=True, separators=(",", ":"))
    assert report_c["configuration_hash"] == hashlib.sha256(
        canonical_c.encode("utf-8")
    ).hexdigest()
