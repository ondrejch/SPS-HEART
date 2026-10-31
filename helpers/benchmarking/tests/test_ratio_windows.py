from __future__ import annotations

import numpy as np

import helpers.benchmarking.benchmark_multispecies_constrained as bmc
from helpers.benchmarking.benchmark_multispecies_constrained import (
    SpeciesData,
    _build_particle_ratio_summary,
    _parse_xf_windows,
    _point_bin_window_areas,
)


def _dummy_species() -> SpeciesData:
    # Two xF bins centered at 0.10 and 0.20 with width 0.10 each.
    xf = np.asarray([0.10, 0.20], dtype=float)
    pt = np.asarray([0.30, 0.30], dtype=float)
    xedges = np.asarray([0.05, 0.15, 0.25], dtype=float)
    pedges = np.asarray([0.20, 0.40], dtype=float)
    ix_data = np.asarray([0, 1], dtype=np.int64)
    ip_data = np.asarray([0, 0], dtype=np.int64)
    return SpeciesData(
        key="pi-",
        name="pi-",
        tag="piminus",
        pdg=-211,
        mass_gev=0.13957,
        na49_path=None,  # type: ignore[arg-type]
        xf=xf,
        pt=pt,
        y=np.asarray([1.0, 1.0], dtype=float),
        sigma=np.asarray([1.0, 1.0], dtype=float),
        xedges=xedges,
        pedges=pedges,
        ix_data=ix_data,
        ip_data=ip_data,
        train_idx=np.asarray([0, 1], dtype=np.int64),
        test_idx=np.asarray([0, 1], dtype=np.int64),
        matrix=np.zeros((2, 1), dtype=float),
        pred_res=np.zeros(2, dtype=float),
        pred_res_sigma=np.zeros(2, dtype=float),
    )


def test_parse_xf_windows_deduplicates_and_orders_bounds() -> None:
    wins = _parse_xf_windows("0.2:0.1,0.1:0.2,0.0:0.4")
    assert len(wins) == 2
    assert wins[0][0] == "xf_0.100_0.200"
    assert wins[0][1] == 0.1
    assert wins[0][2] == 0.2
    assert wins[1][0] == "xf_0.000_0.400"


def test_point_window_area_uses_boundary_overlap() -> None:
    sd = _dummy_species()
    # Window [0.10, 0.20] overlaps each xF bin by 0.05. pT width is 0.20.
    areas = _point_bin_window_areas(sd, 0.10, 0.20)
    assert np.allclose(areas, np.asarray([0.01, 0.01], dtype=float), rtol=0.0, atol=1e-12)


def _charged_pion_pair() -> dict:
    """pi+/pi- pair with distinct data and resonance prediction levels.

    Levels are constant per species so the point-wise yield kernel cancels
    exactly in each ratio; Data and Resonance_shared use different levels so
    the test can tell which array fed which model.
    """
    out = {}
    for key, tag, pdg, y_vals, res_vals in (
        ("pi+", "piplus", 211, [3.0, 3.0], [4.0, 4.0]),
        ("pi-", "piminus", -211, [1.0, 1.0], [2.0, 2.0]),
    ):
        sd = _dummy_species()
        sd.key = key
        sd.name = key
        sd.tag = tag
        sd.pdg = pdg
        sd.y = np.asarray(y_vals, dtype=float)
        sd.sigma = np.asarray([0.1, 0.1], dtype=float)
        sd.pred_res = np.asarray(res_vals, dtype=float)
        out[key] = sd
    return out


def test_particle_ratio_summary_omits_external_generators(monkeypatch) -> None:
    """Policy: external per-species shape fits stay out of ratio windows."""
    monkeypatch.setattr(bmc, "SQRT_S_GEV", 17.26)
    summary = _build_particle_ratio_summary(
        species_data_by_key=_charged_pion_pair(),
        xf_windows=[("xf_0.000_0.400", 0.0, 0.4)],
    )
    expected_models = {"Data", "Resonance_shared"}
    for ratio_vals in summary["values_by_window"]["xf_0.000_0.400"].values():
        assert set(ratio_vals.keys()) == expected_models
    for ratio_vals in summary["values"].values():
        assert set(ratio_vals.keys()) == expected_models

    note = str(summary["external_normalization"]["note"])
    assert "omitted conservatively" in note
    assert "shape-only" in note
    # Same yield kernel as the main summary (invariant f integrated as a yield).
    assert "(pT/E)" in str(summary["integration"])

    # Exact ratios: constant per-species levels cancel the kernel pointwise.
    vals = summary["values"]["pi_plus_over_pi_minus"]
    assert vals["Data"] == 3.0
    assert vals["Resonance_shared"] == 2.0
