"""Pipeline regressions for the 08rev bugs."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from helpers.benchmarking.benchmark_multispecies_constrained import (
    validate_data_system_generation,
)
from helpers.benchmarking.validate_np_scattering import (
    _HERE,
    _DEFAULT_MIXING_DIR,
    _layout_roots,
    _normalized_counts,
    _normalized_shape,
    check_data_overlays,
)


def test_data_system_requires_matching_generation() -> None:
    with pytest.raises(ValueError, match="requires --pa-a"):
        validate_data_system_generation("pPb", 0)
    with pytest.raises(ValueError, match="requires --pa-a"):
        validate_data_system_generation("pC", 0)
    with pytest.raises(ValueError, match="pair it with --data-system"):
        validate_data_system_generation("pp", 208)
    validate_data_system_generation("pPb", 208)
    validate_data_system_generation("pp", 0)
    validate_data_system_generation("pPb", 0, allow_mismatch=True)


def test_kaon_helper_normalizes_counts_not_as_xf() -> None:
    counts = np.array([10.0, 30.0, 60.0])
    frac, _ = _normalized_counts(counts)
    np.testing.assert_allclose(frac, [0.1, 0.3, 0.6])
    # The old bug treated those counts as xF samples on [-1, 1].
    wrong, _ = _normalized_shape(counts, np.linspace(-1.0, 1.0, 15))
    assert not np.allclose(frac, wrong[: frac.size])


def test_np_validator_defaults_to_package_dir() -> None:
    mix, runs = _layout_roots(_HERE)
    assert (mix / "na49_mixing").is_dir()
    assert mix == _DEFAULT_MIXING_DIR
    assert runs.name == "00runs" or "00runs" in str(runs)


def test_overlay_hook_loads_csv_and_scores(tmp_path) -> None:
    csv_path = tmp_path / "na49_pp_piplus_xf.csv"
    csv_path.write_text(
        "xf,value,err\n"
        "-0.25,1.0,0.1\n"
        "0.25,2.0,0.1\n"
    )
    tracks = {
        "pdg": np.array([211, 211, 211, -211], dtype=np.int32),
        "xf": np.array([-0.2, -0.3, 0.3, 0.1], dtype=float),
    }
    out = check_data_overlays(tmp_path, summed_pp=tracks, summed_np=tracks)
    assert out["status"] == "present"
    rec = out["overlays"]["pp_piplus"]
    assert rec["status"] == "loaded"
    assert rec["n_points"] == 2
    assert rec["n_bins_compared"] == 2
    assert "chi2" in rec
    missing = check_data_overlays(None)
    assert missing["status"] == "absent"
