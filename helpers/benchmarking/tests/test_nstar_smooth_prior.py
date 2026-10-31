#!/usr/bin/env python3

from __future__ import annotations

import numpy as np

from helpers.benchmarking.benchmark_multispecies_constrained import _build_nstar_ladder_smooth_prior


def test_nstar_smooth_prior_builds_adjacent_rows() -> None:
    labels = [
        "eta",
        "nstar_effL01_zero",
        "nstar_effL02_zero",
        "nstar_effL04_zero",
        "anti_nstar_effL01_zero",
        "anti_nstar_effL02_zero",
        "phi_1020",
    ]
    out = _build_nstar_ladder_smooth_prior(component_labels=labels)
    assert bool(out.get("ok", False))
    A = np.asarray(out["A"], dtype=float)
    b = np.asarray(out["b"], dtype=float)

    assert A.shape == (3, len(labels))
    assert b.shape == (3,)
    assert np.allclose(b, 0.0)
    assert int(out.get("n_ladders", 0)) == 2

    rows = out.get("row_meta", [])
    assert isinstance(rows, list)
    assert len(rows) == 3
    diffs = {(int(r["level_lo"]), int(r["level_hi"])) for r in rows if isinstance(r, dict)}
    assert (1, 2) in diffs
    assert (2, 4) in diffs


def test_nstar_smooth_prior_handles_missing_ladder() -> None:
    out = _build_nstar_ladder_smooth_prior(component_labels=["eta", "phi_1020"])
    assert not bool(out.get("ok", False))
    assert str(out.get("reason", "")) == "no_nstar_ladders"

