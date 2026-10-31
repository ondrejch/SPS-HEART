"""Round-3 tests for the multispecies constrained fit policy plumbing.

Covers:
- `_species_column_from_tracks` math contract: raw vs invariant column
  ratios equal per-track (E/pT)/bin-area weights;
- CLI defaults for `--weak-feed-policy` and `--convention`.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import helpers.benchmarking.benchmark_multispecies_constrained as bmc
from helpers.benchmarking.sps_core import PDG_MASS_GEV


SQRT_S_GEV = 17.26


def _make_species_namespace() -> types.SimpleNamespace:
    """Minimal SpeciesData-like grid: two data points in distinct bins."""
    return types.SimpleNamespace(
        pdg=211,
        y=np.zeros(2, dtype=float),
        xedges=np.array([0.0, 0.1, 0.2]),
        pedges=np.array([0.05, 0.10, 0.15]),
        ix_data=np.array([0, 1], dtype=np.int64),
        ip_data=np.array([0, 1], dtype=np.int64),
    )


def _tracks():
    xf = np.array([0.05, 0.15])
    pt = np.array([0.075, 0.125])
    pdg = np.full(xf.shape, 211, dtype=np.int32)
    return pdg, xf, pt


def _expected_track_weights(xf: np.ndarray, pt: np.ndarray) -> np.ndarray:
    mass = PDG_MASS_GEV[211]
    pl = xf * (0.5 * SQRT_S_GEV)
    energy = np.sqrt(mass ** 2 + pl ** 2 + pt ** 2)
    return energy / pt


def test_invariant_column_ratio_equals_e_over_pt_over_area():
    sd = _make_species_namespace()
    pdg, xf, pt = _tracks()

    raw = bmc._species_column_from_tracks(
        pdg_arr=pdg, xf_arr=xf, pt_arr=pt, sd=sd,
        convention="raw", sqrt_s_gev=SQRT_S_GEV,
    )
    inv = bmc._species_column_from_tracks(
        pdg_arr=pdg, xf_arr=xf, pt_arr=pt, sd=sd,
        convention="invariant", sqrt_s_gev=SQRT_S_GEV,
    )

    areas = bmc.bss._grid_point_areas(sd.xedges, sd.pedges, sd.ix_data, sd.ip_data)
    w = _expected_track_weights(xf, pt)

    # One track per data bin under 'raw'.
    np.testing.assert_allclose(raw, [1.0, 1.0], rtol=0, atol=0)
    np.testing.assert_allclose(inv / raw, w / areas, rtol=1e-12, atol=0)


def test_invariant_column_sums_multiple_tracks_per_bin():
    sd = _make_species_namespace()
    # Two tracks in bin (0, 0), none in bin (1, 1).
    pdg = np.array([211, 211], dtype=np.int32)
    xf = np.array([0.05, 0.02])
    pt = np.array([0.075, 0.06])

    raw = bmc._species_column_from_tracks(
        pdg_arr=pdg, xf_arr=xf, pt_arr=pt, sd=sd,
        convention="raw", sqrt_s_gev=SQRT_S_GEV,
    )
    inv = bmc._species_column_from_tracks(
        pdg_arr=pdg, xf_arr=xf, pt_arr=pt, sd=sd,
        convention="invariant", sqrt_s_gev=SQRT_S_GEV,
    )

    assert raw[0] == 2.0 and raw[1] == 0.0
    area = 0.1 * 0.05
    w_sum = float(np.sum(_expected_track_weights(xf, pt)))
    np.testing.assert_allclose(inv, [w_sum / area, 0.0], rtol=1e-12, atol=0)


def test_zero_column_for_nonmatching_pdg():
    sd = _make_species_namespace()
    pdg = np.full(3, -321, dtype=np.int32)
    counts = bmc._species_column_from_tracks(
        pdg_arr=pdg, xf_arr=np.array([0.05, 0.15, 0.12]),
        pt_arr=np.array([0.075, 0.125, 0.09]), sd=sd,
        convention="invariant", sqrt_s_gev=SQRT_S_GEV,
    )
    np.testing.assert_array_equal(counts, np.zeros(len(sd.y)))


def _option(parser, flag: str):
    return parser._option_string_actions[flag]


def test_arg_parser_weak_feed_policy_default():
    parser = bmc._build_arg_parser()
    act = _option(parser, "--weak-feed-policy")
    assert act.choices == ["inclusive_weak", "primary_only"]
    assert act.default == "primary_only"
    parsed = parser.parse_args([])
    assert parsed.weak_feed_policy == "primary_only"


def test_arg_parser_convention_default():
    parser = bmc._build_arg_parser()
    act = _option(parser, "--convention")
    assert act.choices == ["invariant", "raw"]
    assert act.default == "invariant"
    parsed = parser.parse_args([])
    assert parsed.convention == "invariant"
