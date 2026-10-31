"""Hermetic guard + regression test for the seagull mean-pT kernel.

The regression check reads external NA49 fixture tables; it skips with an
explicit reason when those fixtures are absent so the default test command
stays green in hermetic checkouts (see _na49_fixtures.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import helpers.benchmarking.benchmark_sps_species as bss
from helpers.benchmarking.benchmark_multispecies_constrained import (
    _load_species_points,
    _seagull_profile_for_prediction,
)
from helpers.benchmarking.particle_definitions import resolve_species

from _na49_fixtures import PT_INTEGRATED_PIMINUS_TABLE
from _na49_fixtures import _require_na49_fixtures


def _read_sps_meanpt_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, comments="#", dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise RuntimeError(f"Unexpected SPS pt-integrated table format: {path}")
    xf = np.asarray(arr[:, 0], dtype=float)
    mean_pt = np.asarray(arr[:, 5], dtype=float)
    return xf, mean_pt


@pytest.mark.external_data
def test_seagull_kernel_matches_sps_meanpt_for_pi_minus() -> None:
    # pi- species table (read via _load_species_points) + pt-integrated reference.
    _require_na49_fixtures(resolve_species("pi-").na49_table, PT_INTEGRATED_PIMINUS_TABLE)
    sd = _load_species_points(
        species_key="pi-",
        xf_min=0.0,
        xf_max=0.4,
        pt_min=0.05,
        pt_max=1.5,
        seed=20260330,
    )
    sqrt_s = float(bss._sqrt_s_from_fixed_target_beam_momentum(158.0))
    idx = np.arange(sd.y.shape[0], dtype=np.int64)
    xf_prof, mean_pt_prof, _ = _seagull_profile_for_prediction(
        sd=sd,
        prediction=np.asarray(sd.y, dtype=float),
        prediction_sigma=np.asarray(sd.sigma, dtype=float),
        idx=idx,
        sqrt_s_from_beam_momentum=sqrt_s,
    )
    assert not np.any(np.isclose(xf_prof, 0.075, atol=1e-12))

    ref_xf, ref_mean_pt = _read_sps_meanpt_table(PT_INTEGRATED_PIMINUS_TABLE)
    ref_map = {round(float(x), 4): float(y) for x, y in zip(ref_xf, ref_mean_pt)}
    common_pred: list[float] = []
    common_ref: list[float] = []
    for x, mu in zip(xf_prof, mean_pt_prof):
        key = round(float(x), 4)
        if key in ref_map:
            common_pred.append(float(mu))
            common_ref.append(float(ref_map[key]))
    assert len(common_pred) >= 10

    diff = np.abs(np.asarray(common_pred, dtype=float) - np.asarray(common_ref, dtype=float))
    mae = float(np.mean(diff))
    # Regression guard: with invariant kernel + pT-coverage filtering this should
    # stay close to the SPS pt-integrated reference profile.
    assert mae < 0.01
