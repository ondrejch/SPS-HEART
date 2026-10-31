"""CLI nits for digitize_vector_figure (default out-dir, fig1b keys)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

pymupdf = pytest.importorskip("pymupdf")

from helpers.benchmarking import digitize_vector_figure as dvf  # noqa: E402


def _dummy_fig2a(_pdf):
    return {
        "source": "dummy.pdf",
        "page_index": 1,
        "figure": "Figure 2(a)",
        "xf_convention": "x_F = p_L/p_L^max",
        "calibration": {"x_labels_used": 2, "y_labels_used": 2},
        "series_pp_piplus_over_piminus": [
            {"xf": 0.1, "value": 1.2, "err": 0.05}
        ],
        "series_np_piminus_over_piplus": [
            {"xf": 0.1, "value": 1.1, "err": 0.04}
        ],
    }


def _dummy_fig1b(_pdf):
    pt = [{"pt": 0.2, "value": 3.0, "err": 0.1}]
    return {
        "source": "dummy.pdf",
        "page_index": 1,
        "figure": "Figure 1(b)",
        "observable": "F(pT)",
        "calibration": {"x_labels_used": 2, "y_labels_used": 2},
        "series_pp_kplus": pt,
        "series_np_kplus": pt,
        "series_pp_kminus": pt,
        "series_np_kminus": pt,
    }


def test_fig2a_default_outdir_without_out_dir(tmp_path, monkeypatch):
    notes = tmp_path / "notes" / "benchmarks"
    notes.mkdir(parents=True)
    monkeypatch.setattr(dvf, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(dvf, "extract_fig2a", _dummy_fig2a)
    dummy_pdf = tmp_path / "dummy.pdf"
    dummy_pdf.write_bytes(b"%PDF")
    rc = dvf.main(["--pdf", str(dummy_pdf), "--figure", "fig2a"])
    assert rc == 0
    out = notes / "digitized" / "na49"
    assert (out / "na49_pp_pioverpi_xf.csv").is_file()
    assert (out / "na49_np_pioverpi_xf.csv").is_file()


def test_fig1b_writes_system_charge_series(tmp_path, monkeypatch):
    monkeypatch.setattr(dvf, "extract_fig1b", _dummy_fig1b)
    dummy_pdf = tmp_path / "dummy.pdf"
    dummy_pdf.write_bytes(b"%PDF")
    out = tmp_path / "csv"
    rc = dvf.main(
        ["--pdf", str(dummy_pdf), "--figure", "fig1b", "--out-dir", str(out)]
    )
    assert rc == 0
    for system in ("pp", "np"):
        for charge in ("kplus", "kminus"):
            assert (out / f"na49_{system}_{charge}_pt.csv").is_file()
