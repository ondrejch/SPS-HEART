"""Tests for pA data-system resolution and cache namespaces."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from helpers.benchmarking.particle_definitions import na49_table_for
from helpers.benchmarking.sps_core import (
    disambiguate_duplicate_pt_labels,
    paper_window_mask,
    read_na49_points,
)


def test_pp_system_keeps_historical_mapping() -> None:
    assert na49_table_for("pi-").endswith("xsect_piminus.txt")
    assert na49_table_for("p", "pp").endswith("xsect_proton_pp.txt")


def test_pc_system_maps_mirror_tables() -> None:
    for key, tail in [
        ("pi-", "xsect_piminus_pC.txt"),
        ("pi+", "xsect_piplus_pC.txt"),
        ("p", "xsect_proton_pC.txt"),
        ("pbar", "xsect_antiproton_pC.txt"),
    ]:
        path = Path(na49_table_for(key, "pC"))
        assert path.name == tail
        if not path.exists():
            pytest.skip(f"missing table {path}")


def test_pc_kaons_raise_with_available_list() -> None:
    with pytest.raises(ValueError, match="pC data not available"):
        na49_table_for("k+", "pC")


@pytest.mark.external_data
def test_ppb_system_maps_thesis_tables() -> None:
    for key, tail in [
        ("pi-", "xsect_piminus_pPb.txt"),
        ("pi+", "xsect_piplus_pPb.txt"),
    ]:
        path = Path(na49_table_for(key, "pPb"))
        assert path.name == tail
        if not path.exists():
            pytest.skip(f"missing table {path}")
        # Non-empty beyond its provenance header.
        rows = [
            ln for ln in path.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        assert len(rows) > 100, f"suspiciously few rows in {path}"


def test_ppb_baryons_raise_and_unknown_system_rejected() -> None:
    with pytest.raises(ValueError, match="pPb data not available"):
        na49_table_for("proton", "pPb")
    with pytest.raises(ValueError, match="Unknown data system"):
        na49_table_for("pi-", "CaCa")


def test_duplicate_printed_pt_recovers_fine_grid() -> None:
    xf = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pt = np.array([0.00, 0.10, 0.10, 0.20, 0.20, 0.30])
    out = disambiguate_duplicate_pt_labels(xf, pt)
    np.testing.assert_allclose(out, [0.00, 0.05, 0.10, 0.15, 0.20, 0.30])
    unique = np.unique(np.stack([xf, pt], axis=1), axis=0)
    assert unique.shape[0] < xf.size
    unique_out = np.unique(np.stack([xf, out], axis=1), axis=0)
    assert unique_out.shape[0] == xf.size


@pytest.mark.external_data
def test_ppb_loader_has_unique_window_keys() -> None:
    path = Path(na49_table_for("pi-", "pPb"))
    if not path.exists():
        pytest.skip(f"missing table {path}")
    xf, pt, y, e = read_na49_points(path)
    mask = paper_window_mask(xf, pt, e)
    xf_w, pt_w = xf[mask], pt[mask]
    keys = np.round(np.stack([xf_w, pt_w], axis=1), decimals=6)
    n_unique = np.unique(keys, axis=0).shape[0]
    assert n_unique == xf_w.size


@pytest.mark.external_data
def test_pp_loader_unchanged_on_unique_grid() -> None:
    path = Path(na49_table_for("pi-"))
    if not path.exists():
        pytest.skip(f"missing table {path}")
    xf, pt, y, e = read_na49_points(path)
    # Re-read raw printed pT without disambiguation.
    raw_pt = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        raw_pt.append(float(line.split()[1]))
    np.testing.assert_allclose(pt, np.asarray(raw_pt, dtype=float))
