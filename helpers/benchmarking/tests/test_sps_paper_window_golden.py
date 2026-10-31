#!/usr/bin/env python3
"""Golden tests on the real SPS paper kinematic window.

Locks table loading + metrics + a tiny multi-component NNLS fit against the
published pi+ table in the project protocol window
(0 <= xF <= 0.4, 0.05 <= pT <= 1.5 GeV/c).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from helpers.benchmarking.particle_definitions import resolve_species
from helpers.benchmarking.sps_core import (
    PAPER_PT_MAX,
    PAPER_PT_MIN,
    PAPER_XF_MAX,
    PAPER_XF_MIN,
    counts_on_grid,
    fit_nnls_weighted,
    load_species_paper_window,
    metrics_bundle,
    relative_rmse_weighted,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MIXING2006 = REPO_ROOT / "dev" / "mixing2006"

# Every test in this module loads the published pi+ table through the
# shared fixture; without the external tables they skip. The helpers
# gate deselects them entirely for hermetic archive runs.
pytestmark = pytest.mark.external_data


@pytest.fixture(scope="module")
def piplus_window():
    table = Path(resolve_species("pi+").na49_table)
    if not table.is_file():
        pytest.skip(f"SPS table missing: {table}")
    return load_species_paper_window(table)


def test_paper_window_piplus_point_count_and_anchors(piplus_window) -> None:
    """Published pi+ table: paper window size and edge anchors."""
    y = piplus_window["y"]
    xf = piplus_window["xf"]
    pt = piplus_window["pt"]
    sigma = piplus_window["sigma"]

    assert y.shape == xf.shape == pt.shape == sigma.shape
    # Stable count for the SPS HADRONS pi+ table in the protocol window.
    assert int(y.shape[0]) == 230
    assert float(np.min(xf)) >= PAPER_XF_MIN
    assert float(np.max(xf)) <= PAPER_XF_MAX
    assert float(np.min(pt)) >= PAPER_PT_MIN
    assert float(np.max(pt)) <= PAPER_PT_MAX
    assert np.all(sigma > 0.0)

    # First/last points in file order after window filter (not re-sorted).
    assert xf[0] == pytest.approx(0.0)
    assert pt[0] == pytest.approx(0.05)
    assert y[0] == pytest.approx(62.87363, rel=0, abs=1e-5)
    assert sigma[0] == pytest.approx(0.38353, rel=0, abs=1e-5)
    assert xf[-1] == pytest.approx(0.35)
    assert pt[-1] == pytest.approx(1.5)
    assert y[-1] == pytest.approx(0.011, rel=0, abs=1e-6)


def test_metrics_bundle_identity_prediction(piplus_window) -> None:
    y = piplus_window["y"]
    sigma = piplus_window["sigma"]
    pred = y.copy()
    m = metrics_bundle(pred, y, sigma, n_params=0, prefix="full")
    assert m["full_chi2"] == pytest.approx(0.0, abs=1e-12)
    assert m["full_weighted_rel_rmse"] == pytest.approx(0.0, abs=1e-12)
    assert m["full_smape_percent"] == pytest.approx(0.0, abs=1e-12)
    assert m["full_log_pull_rms"] == pytest.approx(0.0, abs=1e-12)


def test_tiny_resonance_nnls_on_real_piplus_window(piplus_window) -> None:
    """End-to-end: generate a few resonance components and fit the real table."""
    if str(MIXING2006) not in sys.path:
        sys.path.insert(0, str(MIXING2006))
    from na49_mixing.parallel import GenerationConfig, generate_tracks_chunked

    gen_dir = MIXING2006 / "na49_mixing" / "generators"
    components = [
        (
            "delta_pp",
            gen_dir / "michal" / "deltas-to-pion" / "delta++.gen",
            True,
        ),
        (
            "rho_minus",
            gen_dir / "benchmark" / "rho_minus_770.gen",
            True,
        ),
        (
            "nstar_eff",
            gen_dir / "benchmark" / "nstar_eff1900_zero.gen",
            True,
        ),
    ]
    for _, path, _ in components:
        if not path.is_file():
            pytest.skip(f"missing gen card: {path}")

    y = piplus_window["y"]
    sigma = piplus_window["sigma"]
    xedges = piplus_window["xedges"]
    pedges = piplus_window["pedges"]
    ix = piplus_window["ix_data"]
    ip = piplus_window["ip_data"]
    pdg_piplus = 211

    cols = []
    for seed_offset, (label, gen_path, use_soft) in enumerate(components):
        cfg = GenerationConfig(
            gen_file=str(gen_path),
            n_events=800,
            seed=20260331 + seed_offset,
            beam_momentum=158.0,
            max_decay_passes=4,
            use_soft_decay_factory=bool(use_soft),
            fragmentation_mode="both",
        )
        tracks = generate_tracks_chunked(cfg, n_workers=1, n_chunks=2)
        pdg = np.asarray(tracks["pdg"])
        xf = np.asarray(tracks["xf"], dtype=float)
        pt = np.asarray(tracks["pt"], dtype=float)
        keep = pdg == pdg_piplus
        if not np.any(keep):
            # Still include a zero column so shape is stable; NNLS can zero it.
            cols.append(np.zeros_like(y))
            continue
        col = counts_on_grid(xf[keep], pt[keep], xedges, pedges, ix, ip)
        cols.append(np.asarray(col, dtype=float))

    A = np.column_stack(cols)
    assert A.shape == (y.shape[0], len(components))
    # At least one component should fire in the window.
    assert float(A.sum()) > 0.0

    w, info = fit_nnls_weighted(A, y, sigma)
    assert info["method"] == "scipy.optimize.nnls"
    assert w.shape == (len(components),)
    assert np.all(w >= -1e-12)
    assert float(np.sum(w)) > 0.0

    pred = A @ w
    wrrmse = relative_rmse_weighted(pred, y, sigma)
    m = metrics_bundle(pred, y, sigma, n_params=int(np.count_nonzero(w > 1e-12)), prefix="full")
    assert np.isfinite(wrrmse)
    assert wrrmse < 5.0  # loose ceiling: tiny MC, not paper-quality stats
    assert np.isfinite(float(m["full_chi2"]))
    assert float(m["full_chi2"]) < float(metrics_bundle(np.zeros_like(y), y, sigma, 0, "z")["z_chi2"])
