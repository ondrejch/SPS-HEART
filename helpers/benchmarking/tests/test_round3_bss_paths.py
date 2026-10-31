"""Round-3 path checks for benchmark_sps_species (bss).

Covers two audit paths added on 2026-08-23:

1. External-generator CSV loading under the invariant-cross-section
   convention: each CSV track carries the Jacobian weight E/pT and the
   accumulated cell sum is divided by the point's bin area.
2. Interface completeness of the direct PYTHIA path (convention keyword)
   and the NA49-compatible weak-feed default of the CLI parser.
"""

import inspect
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from helpers.benchmarking import benchmark_sps_species as bss
from helpers.benchmarking.sps_core import PDG_MASS_GEV

SQRT_S = 17.27


def test_external_loader_invariant_shapes(tmp_path):
    xedges = np.array([0.0, 0.2, 0.4])
    pedges = np.array([0.0, 0.5, 1.0])
    # One track per occupied cell so raw * (E/pT) / area is exact per point.
    rows = ((0.05, 0.10), (0.25, 0.70))
    csv_path = tmp_path / "pythia_pi_minus.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write("xf,pt,value\n")
        for xf, pt in rows:
            fh.write(f"{xf},{pt},1\n")

    ix_data = np.array([0, 1])
    ip_data = np.array([0, 1])

    raw = bss._load_external_grid_counts(csv_path, xedges, pedges, ix_data, ip_data)
    inv = bss._load_external_grid_counts(
        csv_path,
        xedges,
        pedges,
        ix_data,
        ip_data,
        convention="invariant",
        sqrt_s_gev=SQRT_S,
        mass_gev=PDG_MASS_GEV[-211],
    )

    assert np.allclose(raw, [1.0, 1.0])

    m = PDG_MASS_GEV[-211]
    dx = float(np.diff(xedges)[0])
    dp = float(np.diff(pedges)[0])
    expected = []
    for xf, pt in rows:
        pl = xf * (0.5 * SQRT_S)
        e_over_pt = np.sqrt(m * m + pl * pl + pt * pt) / pt
        expected.append(e_over_pt / (dx * dp))
    assert np.allclose(inv, expected)


def test_pythia_signature_accepts_convention():
    params = inspect.signature(bss._run_pythia).parameters
    assert "convention" in params
    assert params["convention"].default == "raw"

    parser = bss._build_arg_parser()
    args = parser.parse_args([])
    assert args.weak_feed_policy == "primary_only"
    assert (
        parser.parse_args(["--weak-feed-policy", "inclusive_weak"]).weak_feed_policy
        == "inclusive_weak"
    )
