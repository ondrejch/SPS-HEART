"""Unit coverage for the postprocess window-yield kernel.

postprocess_multispecies_windows.py integrates windowed particle ratios with
the yield kernel w = (pT/E) * overlap(delta_xF) * delta_pT, matching
benchmark_multispecies_constrained._invariant_integration_weights.  The D3
fix (commit bdf62a7) was validated by a scratch harness only
(dev/00runs/tmp/p4_postproc_smoke/); these tests pin the committed behavior:

1. kernel values equal (pT/E)*overlap_xF*dPT on hand-computable grid points;
2. a window cutting through bin interiors weights each bin by its partial
   overlap, which differs from full-bin area-only integration;
3. the kernel is mass aware: identical kinematics with different hadron
   masses give different weights;
4. _write_ratios applies the kernel end to end on a hermetic NPZ, keeps
   external-generator entries out of the ratio windows, and emits None for
   windows without grid coverage;
5. even when a production-style NPZ DOES contain {tag}_pred_pythia/_epos/
   _urqmd arrays, the external ratios never reach the ratio CSV or the
   returned payload: the columns are fenced out entirely, matching the
   omission policy of _build_particle_ratio_summary.
"""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest

from helpers.benchmarking.postprocess_multispecies_windows import (
    _parse_windows,
    _window_yield_kernel,
    _write_ratios,
)

# Benchmark grid (same convention as test_ratio_windows.py):
# xF centers [0.10, 0.20] give edges [0.05, 0.15, 0.25];
# pT centers [0.30, 0.40] give edges [0.25, 0.35, 0.45], so dPT = 0.10.
XF = np.asarray([0.10, 0.20], dtype=float)
PT = np.asarray([0.30, 0.40], dtype=float)
DPT = 0.10
SQRT_S_GEV = 17.26
# Rest masses as resolved by _pdg_mass_gev for the benchmarked species.
M_PION = 0.13957
M_KAON = 0.493677
M_PROTON = 0.9382720813


def _yield_weight(*, xf_val: float, pt_val: float, mass_gev: float,
                  overlap_xf: float) -> float:
    """Independent scalar restatement of w = (pT/E)*overlap(delta_xF)*dPT."""
    p_l = xf_val * 0.5 * SQRT_S_GEV
    energy = math.sqrt(mass_gev**2 + p_l**2 + pt_val**2)
    return (pt_val / energy) * overlap_xf * DPT


def _kernel(mass_gev: float, lo: float, hi: float) -> np.ndarray:
    return _window_yield_kernel(
        xf=XF,
        pt=PT,
        mass_gev=mass_gev,
        sqrt_s_gev=SQRT_S_GEV,
        xf_min=lo,
        xf_max=hi,
    )


def test_kernel_equals_pt_over_e_times_overlap_times_dpt() -> None:
    # Window [0, 0.4] covers both xF bins fully, so overlap = width 0.10.
    kernel = _kernel(M_PION, 0.0, 0.4)
    expected = np.asarray([
        _yield_weight(xf_val=0.10, pt_val=0.30, mass_gev=M_PION, overlap_xf=0.10),
        _yield_weight(xf_val=0.20, pt_val=0.40, mass_gev=M_PION, overlap_xf=0.10),
    ])
    assert np.allclose(kernel, expected, rtol=1e-12, atol=0.0)
    # Literal anchor for point A: 0.30/sqrt(0.13957^2 + 0.863^2 + 0.30^2)*0.01.
    assert abs(kernel[0] - 3.24585466288396371e-03) <= 1e-12 * kernel[0]


def test_boundary_window_weights_partial_overlap_not_area_only() -> None:
    # Window [0.12, 0.18] cuts through the interior of both xF bins, keeping
    # overlap 0.03 from bin [0.05, 0.15] and 0.03 from bin [0.15, 0.25].
    kernel = _kernel(M_PION, 0.12, 0.18)
    expected = np.asarray([
        _yield_weight(xf_val=0.10, pt_val=0.30, mass_gev=M_PION, overlap_xf=0.03),
        _yield_weight(xf_val=0.20, pt_val=0.40, mass_gev=M_PION, overlap_xf=0.03),
    ])
    assert np.allclose(kernel, expected, rtol=1e-12, atol=0.0)
    assert abs(kernel[0] - 9.73756398865189784e-04) <= 1e-12 * kernel[0]

    # Full-bin area-only integration would keep the whole 0.10-wide bins; the
    # partial-overlap weights sit exactly a factor 10/3 below that.
    full_bin = np.asarray([
        _yield_weight(xf_val=0.10, pt_val=0.30, mass_gev=M_PION, overlap_xf=0.10),
        _yield_weight(xf_val=0.20, pt_val=0.40, mass_gev=M_PION, overlap_xf=0.10),
    ])
    assert not np.allclose(kernel, full_bin, rtol=1e-12, atol=0.0)
    assert np.allclose(full_bin / kernel, np.full(2, 10.0 / 3.0), rtol=1e-12, atol=0.0)

    # Dropping the (pT/E) Jacobian (the pre-D3 area kernel) differs too.
    area_only = np.asarray([0.03 * DPT, 0.03 * DPT])
    assert not np.allclose(kernel, area_only, rtol=1e-12, atol=0.0)


def test_window_without_grid_coverage_gives_zero_weights() -> None:
    kernel = _kernel(M_PION, 0.5, 0.6)
    assert np.all(kernel == 0.0)


def test_kernel_is_mass_aware_at_identical_kinematics() -> None:
    pion = _kernel(M_PION, 0.0, 0.4)
    kaon = _kernel(M_KAON, 0.0, 0.4)
    proton = _kernel(M_PROTON, 0.0, 0.4)
    # Heavier species carry smaller yield weights at fixed kinematics because
    # E = sqrt(m^2 + pL^2 + pT^2) grows with the rest mass.
    assert np.all(kaon < pion)
    assert np.all(proton < kaon)
    for mass, kernel in ((M_KAON, kaon), (M_PROTON, proton)):
        expected = np.asarray([
            _yield_weight(xf_val=0.10, pt_val=0.30, mass_gev=mass, overlap_xf=0.10),
            _yield_weight(xf_val=0.20, pt_val=0.40, mass_gev=mass, overlap_xf=0.10),
        ])
        assert np.allclose(kernel, expected, rtol=1e-12, atol=0.0)


def test_parse_windows_normalizes_bounds_and_deduplicates() -> None:
    wins = _parse_windows("0.2:0.1, 0.10:0.20 ,0.05:0.35")
    assert len(wins) == 2
    assert wins[0] == ("xf_0.100_0.200", 0.1, 0.2)
    assert wins[1] == ("xf_0.050_0.350", 0.05, 0.35)
    with pytest.raises(ValueError):
        _parse_windows("0.2")
    with pytest.raises(ValueError):
        _parse_windows("   ")


# ---------------------------------------------------------------------------
# End-to-end coverage of _write_ratios on a fabricated run directory.
# ---------------------------------------------------------------------------

# Production-format species rows as emitted into the summary JSON.  Tags use
# the resolve_species spellings; no {tag}_pred_pythia/_epos/_urqmd keys exist
# in the fabricated NPZ, mirroring the external-free policy.
SPECIES_MAP = {
    "pi+": {"name": "pi+", "tag": "piplus", "pdg": 211},
    "pi-": {"name": "pi-", "tag": "piminus", "pdg": -211},
    "k+": {"name": "k+", "tag": "kaplus", "pdg": 321},
    "k-": {"name": "k-", "tag": "kaminus", "pdg": -321},
}

LEVELS = {
    # tag: {"data": [...], "res": [...]}, one entry per grid point.
    "piplus": {"data": [6.0, 4.0], "res": [8.0, 4.0]},
    "piminus": {"data": [2.0, 1.0], "res": [2.0, 2.0]},
    "kaplus": {"data": [1.0, 0.5], "res": [1.5, 0.5]},
    "kaminus": {"data": [0.5, 0.5], "res": [1.0, 0.5]},
}


def _fabricated_npz(tmp_path):
    payload = {}
    for row in SPECIES_MAP.values():
        tag = row["tag"]
        levels = LEVELS[tag]
        payload[f"{tag}_xf"] = XF
        payload[f"{tag}_pt"] = PT
        payload[f"{tag}_y"] = np.asarray(levels["data"], dtype=float)
        payload[f"{tag}_pred_res"] = np.asarray(levels["res"], dtype=float)
    npz_path = tmp_path / "details.npz"
    np.savez(npz_path, **payload)
    return npz_path


def test_write_ratios_end_to_end_on_hermetic_npz(tmp_path) -> None:
    npz_path = _fabricated_npz(tmp_path)
    out_csv = tmp_path / "ratios.csv"
    windows = [("xf_0.000_0.400", 0.0, 0.4), ("xf_0.500_0.600", 0.5, 0.6)]

    with np.load(npz_path, allow_pickle=False) as z:
        payload = _write_ratios(
            z=z,
            species_map=SPECIES_MAP,
            windows=windows,
            out_csv=out_csv,
            sqrt_s_gev=SQRT_S_GEV,
        )

    integration = str(payload["integration"])
    assert "(pT/E)" in integration
    assert "yield kernel" in integration

    covered = payload["values_by_window"]["xf_0.000_0.400"]
    assert set(covered) == {"pi+/pi-", "k+/k-", "(k++k-)/(pi++pi-)"}

    vals = covered["pi+/pi-"]
    assert set(vals) == {"Data", "Resonance_shared"}

    def _levels(tag, key):
        return LEVELS[tag][key]

    def _weighted(tag, key, mass):
        return sum(
            val * _yield_weight(xf_val=float(x), pt_val=float(p),
                                mass_gev=mass, overlap_xf=0.10)
            for val, x, p in zip(_levels(tag, key), XF, PT)
        )

    exp_data_pion = (_weighted("piplus", "data", M_PION)
                     / _weighted("piminus", "data", M_PION))
    exp_res_pion = (_weighted("piplus", "res", M_PION)
                    / _weighted("piminus", "res", M_PION))
    assert np.isclose(vals["Data"], exp_data_pion, rtol=1e-12, atol=0.0)
    assert np.isclose(vals["Resonance_shared"], exp_res_pion, rtol=1e-12, atol=0.0)

    # Mixed-mass combined ratio: kaon numerator, pion denominator.  The value
    # must differ from the kernel-free level quotient, proving the mass-aware
    # weights actually enter the sums.
    exp_combined = ((_weighted("kaplus", "data", M_KAON)
                     + _weighted("kaminus", "data", M_KAON))
                    / (_weighted("piplus", "data", M_PION)
                       + _weighted("piminus", "data", M_PION)))
    # Kernel-free level quotient of the same kaon-over-pion sums.
    naive_level_quotient = ((1.0 + 0.5 + 0.5 + 0.5) / (6.0 + 4.0 + 2.0 + 1.0))
    comb = covered["(k++k-)/(pi++pi-)"]["Data"]
    assert not np.isclose(comb, naive_level_quotient, rtol=1e-6, atol=0.0)
    assert np.isclose(comb, exp_combined, rtol=1e-12, atol=0.0)

    # Window outside the grid coverage emits None for every model.
    empty = payload["values_by_window"]["xf_0.500_0.600"]
    for ratio_vals in empty.values():
        assert all(v is None for v in ratio_vals.values())

    with out_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == [
        "window", "xf_min", "xf_max", "ratio", "data", "resonance_shared",
    ]
    assert all(len(r) == 6 for r in rows)
    assert len(rows) == 1 + len(windows) * 3
    covered_rows = [r for r in rows[1:] if r[0] == "xf_0.000_0.400"]
    uncovered_rows = [r for r in rows[1:] if r[0] == "xf_0.500_0.600"]
    assert all(r[4] != "" and r[5] != "" for r in covered_rows)
    data_row = next(r for r in covered_rows if r[3] == "pi+/pi-")
    assert np.isclose(float(data_row[4]), exp_data_pion, rtol=1e-9, atol=0.0)
    assert np.isclose(float(data_row[5]), exp_res_pion, rtol=1e-9, atol=0.0)
    assert all(cell == "" for r in uncovered_rows for cell in r[4:])


def _fabricated_npz_with_external_preds(tmp_path):
    """Hermetic NPZ plus the {tag}_pred_* arrays production runs supply.

    Production writes these keys whenever external-generator CSVs exist
    (benchmark_multispecies_constrained stores per-species external
    predictions), so this fixture reproduces the exact input that used to
    leak cross-species external ratios into the ratio CSV.
    """
    with np.load(_fabricated_npz(tmp_path), allow_pickle=False) as z:
        payload = {k: z[k] for k in z.files}
    for row in SPECIES_MAP.values():
        tag = row["tag"]
        # Offsets off the data levels keep the fenced columns numerically
        # computable, so a policy leak would produce visible wrong cells.
        payload[f"{tag}_pred_pythia"] = 2.0 * payload[f"{tag}_y"]
        payload[f"{tag}_pred_epos"] = 0.5 * payload[f"{tag}_y"]
        payload[f"{tag}_pred_urqmd"] = payload[f"{tag}_y"]
    npz_path = tmp_path / "details_with_externals.npz"
    np.savez(npz_path, **payload)
    return npz_path


def test_write_ratios_fences_external_pred_keys_out_of_ratio_artifacts(tmp_path) -> None:
    npz_path = _fabricated_npz_with_external_preds(tmp_path)
    out_csv = tmp_path / "ratios_with_externals.csv"
    windows = [("xf_0.000_0.400", 0.0, 0.4)]

    with np.load(npz_path, allow_pickle=False) as z:
        # The fixture really does carry the external prediction arrays.
        assert "piplus_pred_pythia" in z.files
        assert "kaminus_pred_urqmd" in z.files
        payload = _write_ratios(
            z=z,
            species_map=SPECIES_MAP,
            windows=windows,
            out_csv=out_csv,
            sqrt_s_gev=SQRT_S_GEV,
        )

    # No external model appears anywhere in the returned payload, and the
    # Data / Resonance_shared values are untouched by the extra arrays.
    for ratio_vals in payload["values_by_window"]["xf_0.000_0.400"].values():
        assert set(ratio_vals) == {"Data", "Resonance_shared"}
    pion_vals = payload["values_by_window"]["xf_0.000_0.400"]["pi+/pi-"]

    def _weighted(tag, key, mass):
        return sum(
            val * _yield_weight(xf_val=float(x), pt_val=float(p),
                                mass_gev=mass, overlap_xf=0.10)
            for val, x, p in zip(LEVELS[tag][key], XF, PT)
        )
    exp_data_pion = (_weighted("piplus", "data", M_PION)
                     / _weighted("piminus", "data", M_PION))
    exp_res_pion = (_weighted("piplus", "res", M_PION)
                    / _weighted("piminus", "res", M_PION))
    assert np.isclose(pion_vals["Data"], exp_data_pion, rtol=1e-12, atol=0.0)
    assert np.isclose(pion_vals["Resonance_shared"], exp_res_pion,
                      rtol=1e-12, atol=0.0)

    with out_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == [
        "window", "xf_min", "xf_max", "ratio", "data", "resonance_shared",
    ]
    assert all(len(r) == 6 for r in rows)
    joined = "\n".join(",".join(r) for r in rows).lower()
    for token in ("pythia", "epos", "urqmd"):
        assert token not in joined
