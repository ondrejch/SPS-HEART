"""Hermetic self-check suite for the physics-separated observable
combinations module (Round 1, card 13,
``TASK-20260911-r1-run-r11-legacy-observable-decomposition``).

The card's six named tests, all hand-computed (no event generation, no
tables, no archive), plus the ratio zero-division guard the card's
semantics define:

- ``test_evaluate_combination_sum``: ``pi_sigma`` is the element-wise sum.
- ``test_evaluate_combination_difference``: ``pi_delta`` is the
  element-wise difference (pi- minus pi+, the card's species order).
- ``test_evaluate_combination_k_pair``: ``k_pair`` is ``2 * f(K-)``.
- ``test_forward_central_backward_integral``: forward / central /
  backward match hand-computed midpoint-rectangle sums; the backward
  integral is 0 on the p+p paper window ``xF in [0, 0.4]``.
- ``test_seagull_moment_subscripts``: the output carries the mandatory
  subscripts ``pT_x_per_xF`` / ``pT_y_per_xF`` and matches the
  hand-computed ``<pT>_x`` / ``<pT>_y`` per xF bin.
- ``test_ranked_residual_map_ordering``: one cell at ``|pull| = 5.0``
  ranks first with ``n_flagged == 1``.
"""

from __future__ import annotations

import math

import pytest

from na49_mixing.fit_observables import (
    COMBINATION_NAMES,
    ObservableCombination,
    combination_by_name,
    evaluate_combination,
    forward_central_backward_integral,
    ranked_residual_map,
    residual_cell_key,
    seagull_moment,
    undefined_combination_bins,
)

TOL = 1e-12


def test_evaluate_combination_sum():
    """pi_sigma = f(pi-) + f(pi+) element-wise to < 1e-12."""
    binned = {"pi-": (1.0, 2.5, -3.25), "pi+": (0.5, 0.125, 4.0)}
    values = evaluate_combination(combination_by_name("pi_sigma"), binned)
    assert len(values) == 3
    for got, (a, b) in zip(values, zip(binned["pi-"], binned["pi+"])):
        assert got is not None and abs(got - (float(a) + float(b))) < TOL


def test_evaluate_combination_difference():
    """pi_delta = f(pi-) - f(pi+) element-wise (the card's species order)."""
    binned = {"pi-": (1.0, 2.5, -3.25), "pi+": (0.5, 0.25, 4.0)}
    values = evaluate_combination(combination_by_name("pi_delta"), binned)
    for got, (a, b) in zip(values, zip(binned["pi-"], binned["pi+"])):
        assert got is not None and abs(got - (float(a) - float(b))) < TOL


def test_evaluate_combination_k_pair():
    """k_pair = 2 * f(K-) per bin to < 1e-12."""
    binned = {"K-": (1.0, 2.5, 0.75)}
    values = evaluate_combination(combination_by_name("k_pair"), binned)
    for got, want in zip(values, (2.0, 5.0, 1.5)):
        assert got is not None and abs(got - want) < TOL
    # The single-species b_pair combination behaves identically.
    values_b = evaluate_combination(
        combination_by_name("b_pair"), {"pbar": (0.5, 1.5)}
    )
    for got, want in zip(values_b, (1.0, 3.0)):
        assert got is not None and abs(got - want) < TOL


def test_forward_central_backward_integral():
    """Hand-computed midpoint-rectangle sums; backward is 0 on p+p."""
    # A dense 4 x 2 grid: xF edges (-0.1, 0.0, 0.1, 0.2, 0.3) give xF
    # midpoints (-0.05, 0.05, 0.15, 0.25) -> backward / central / central /
    # forward under the card's split (forward > 0.2, central [0, 0.2],
    # backward < 0).  Cell value v(ix, ip) = 10*ix + ip; cell area
    # 0.1 * 0.5 = 0.05.  Dense-grid layout: bin i at (i mod 4, i // 4).
    x_edges = (-0.1, 0.0, 0.1, 0.2, 0.3)
    p_edges = (0.0, 0.5, 1.0)
    values = tuple(10.0 * ix + ip for ip in range(2) for ix in range(4))
    result = forward_central_backward_integral(
        {"s": values}, (x_edges, p_edges), "s"
    )
    # forward: ix = 3 cells (30, 31): 0.05 * (30 + 31) = 3.05
    # central: ix = 1, 2 cells (10, 11, 20, 21): 0.05 * 62 = 3.10
    # backward: ix = 0 cells (0, 1): 0.05 * (0 + 1) = 0.05
    assert abs(result["forward"] - 3.05) < TOL
    assert abs(result["central"] - 3.10) < TOL
    assert abs(result["backward"] - 0.05) < TOL
    # The same bins addressed through explicit (ix, ip) indices (the
    # sparse-table form the R11 driver uses) give identical sums.
    indices = [(i % 4, i // 4) for i in range(8)]
    by_index = forward_central_backward_integral(
        {"s": values}, (x_edges, p_edges), "s", bin_indices=indices
    )
    assert by_index == result
    # The p+p paper window xF in [0, 0.4]: every bin midpoint lies in
    # [0, 0.4], so the backward integral is exactly 0 (documented).
    window_edges = ((0.0, 0.2, 0.4), (0.05, 0.775, 1.5))
    window_values = (1.0, 2.0, 3.0, 4.0)
    window = forward_central_backward_integral(
        {"p": window_values}, window_edges, "p"
    )
    assert window["backward"] == 0.0
    # forward: ix = 1 cells (2, 4) with width 0.2 x 0.725 each;
    # central: ix = 0 cells (1, 3).
    assert abs(window["forward"] - 0.2 * 0.725 * (2.0 + 4.0)) < TOL
    assert abs(window["central"] - 0.2 * 0.725 * (1.0 + 3.0)) < TOL


def test_seagull_moment_subscripts():
    """The two conventions, with the mandatory subscripts, hand-computed."""
    # A dense 2 x 2 grid: xF midpoints 0.1 / 0.3, pT midpoints 0.25 / 0.75;
    # mass 0.14 GeV, sqrt(s) = 20 GeV -> pL = 1.0 / 3.0 GeV.
    edges = ((0.0, 0.2, 0.4), (0.0, 0.5, 1.0))
    values = (1.0, 2.0, 3.0, 4.0)
    mass = 0.14
    sqrt_s = 20.0
    result = seagull_moment(
        {"s": values}, edges, "s", sqrt_s_gev=sqrt_s, mass_gev=mass
    )
    assert set(result) == {"pT_x_per_xF", "pT_y_per_xF"}
    assert len(result["pT_x_per_xF"]) == 2
    assert len(result["pT_y_per_xF"]) == 2
    # Hand computation per xF bin: <pT> = sum(pT * density * dp) /
    # sum(density * dp) with density_x = (pT/E) f, density_y = pT f,
    # E = sqrt(m^2 + pL^2 + pT^2) at the cell midpoints, dp = 0.5.
    hand_x = []
    hand_y = []
    slices = (
        (0.1, ((0.25, 1.0), (0.75, 3.0))),
        (0.3, ((0.25, 2.0), (0.75, 4.0))),
    )
    for x_mid, cells in slices:
        p_l = x_mid * sqrt_s / 2.0
        num_x = den_x = num_y = den_y = 0.0
        for p_t, v in cells:
            energy = math.sqrt(mass * mass + p_l * p_l + p_t * p_t)
            num_x += p_t * (p_t / energy) * v * 0.5
            den_x += (p_t / energy) * v * 0.5
            num_y += p_t * p_t * v * 0.5
            den_y += p_t * v * 0.5
        hand_x.append(num_x / den_x)
        hand_y.append(num_y / den_y)
    for got, want in zip(result["pT_x_per_xF"], hand_x):
        assert got is not None and abs(got - want) < TOL
    for got, want in zip(result["pT_y_per_xF"], hand_y):
        assert got is not None and abs(got - want) < TOL
    # The two conventions are distinct densities; on this grid they give
    # distinct means (on real pion data they differ by up to 0.1 GeV/c).
    assert all(
        x is not None and y is not None and abs(x - y) > 1e-9
        for x, y in zip(result["pT_x_per_xF"], result["pT_y_per_xF"])
    )
    # The function refuses to run without the physics it needs.
    with pytest.raises(ValueError):
        seagull_moment({"s": values}, edges, "s")


def test_ranked_residual_map_ordering():
    """One cell at |pull| = 5.0 ranks first; n_flagged == 1."""
    d = {}
    m = {}
    v = {}
    keys = [
        ("pi_minus", "f_xsect", "xF_central", 0),
        ("pi_minus", "f_xsect", "xF_central", 1),
        ("pi_minus", "f_xsect", "xF_forward", 0),
        ("pi_minus", "f_xsect", "xF_forward", 1),
    ]
    for i, (sector, observable, region, bin_index) in enumerate(keys):
        key = residual_cell_key(sector, observable, region, bin_index)
        if i == 2:
            # pull = (d - m) / sigma = (1.0 - 0.8) / 0.04 = 5.0
            d[key] = (1.0,)
            m[key] = (0.8,)
            v[key] = (0.0016,)
        else:
            d[key] = (1.0,)
            m[key] = (1.0,)
            v[key] = (0.25,)
    result = ranked_residual_map(d, m, v, top_n=100)
    assert result["n_total_cells"] == 4
    assert result["n_flagged"] == 1
    assert result["threshold"] == 3.0
    rows = result["rows"]
    assert [row["rank"] for row in rows] == [1, 2, 3, 4]
    assert rows[0]["sector"] == "pi_minus"
    assert rows[0]["observable"] == "f_xsect"
    assert rows[0]["region"] == "xF_forward"
    assert rows[0]["bin_index"] == 0
    assert abs(rows[0]["pull"] - 5.0) < TOL
    assert abs(rows[0]["delta_over_sigma"] - rows[0]["pull"]) < TOL
    pulls = [abs(row["pull"]) for row in rows]
    assert pulls == sorted(pulls, reverse=True)
    # A negative pull of the same magnitude also ranks first.
    flagged_key = residual_cell_key(*keys[2])
    m[flagged_key] = (1.2,)
    negative = ranked_residual_map(d, m, v, top_n=10)
    assert negative["n_flagged"] == 1
    assert abs(negative["rows"][0]["pull"] + 5.0) < TOL
    assert negative["rows"][0]["rank"] == 1


def test_evaluate_combination_ratio_zero_division_guard():
    """A ratio bin whose denominator is 0 yields None and is flagged."""
    ratio = ObservableCombination("r", ("K+", "K-"), "ratio", "K+ / K-")
    guarded = evaluate_combination(
        ratio, {"K+": (2.0, 4.0), "K-": (1.0, 0.0)}
    )
    assert guarded[0] == pytest.approx(2.0)
    assert guarded[1] is None
    assert undefined_combination_bins(guarded) == (1,)
    # The pre-registered combinations are sums and differences: total on
    # aligned cells, never None.
    values = evaluate_combination(
        combination_by_name("k_associated"),
        {"K+": (2.0, 4.0), "K-": (1.0, 0.0)},
    )
    assert undefined_combination_bins(values) == ()
    assert COMBINATION_NAMES == (
        "pi_sigma",
        "pi_delta",
        "k_pair",
        "k_associated",
        "b_pair",
        "b_transport",
    )
