#!/usr/bin/env python3
"""Reproduce the round-10 neutron falsifier (zero-freedom prediction).

Builds the isospin-weighted neutron design columns from the pulled
proton component caches of the retained B-850P run, applies the
p-fit weights (zero fit freedom), normalizes at xF = 0.1, and compares
the shape with the measured pT-integrated neutron dN/dxF in p+p.
Quoted in notes/benchmarks/STAGE_C_ROUND10_DECISION_2026-09-05.md.

Requires the p component caches under --cache-dir (default
/tmp/opencode/neutron_diag) and the ab comparison JSON under
--artifacts (default /tmp/opencode/stageC_ab/ab_B850P.json).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_repo = Path(__file__).resolve().parents[2]
MIXING2006 = _repo / "dev" / "mixing2006"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="/tmp/opencode/stageC_ab")
    parser.add_argument("--cache-dir", default="/tmp/opencode/neutron_diag")
    parser.add_argument(
        "--neutron-table",
        default=str(_repo / "spshadrons.web.cern.ch" / "data"
                    / "xsect_ptintegrated_neutron_pp.txt"),
    )
    args = parser.parse_args()

    sys.path.insert(0, str(_repo))
    sys.path.insert(0, str(MIXING2006))
    from helpers.benchmarking.particle_definitions import resolve_species
    from helpers.benchmarking.staged_resonance_fit import (
        _build_design_matrix, _load_species_points,
    )

    d = json.load(open(Path(args.artifacts) / "ab_B850P.json"))
    sqrt_s = d["per_species"]["p"]["b"]["base"]["sqrt_s_gev"]
    w_p = d["per_species"]["p"]["b"]["base"]["weights"]
    comps = d["per_species"]["p"]["b"]["base"]["components"]
    table_p = Path(resolve_species("p").na49_table)
    points_p = _load_species_points(table_p, xf_min=0.0, xf_max=0.4,
                                    pt_min=0.05, pt_max=1.5)
    arrs = []
    for lbl, c in comps.items():
        cf = Path(args.cache_dir) / Path(c["cache_file"]).name
        with np.load(cf, allow_pickle=False) as z:
            a = {k: z[k] for k in ("pdg", "xf", "pt", "source_side")}
        arrs.append((lbl, a))
    labels, mat = _build_design_matrix(
        species_pdg=2112, components=arrs, points=points_p, sqrt_s_gev=sqrt_s,
    )
    w = np.array([w_p.get(l, 0.0) for l in labels])
    pred = mat @ w
    xedges = points_p["xedges"]
    model = []
    for lo, hi in zip(xedges[:-1], xedges[1:]):
        m = (points_p["xf"] >= lo) & (points_p["xf"] < hi)
        if m.sum():
            model.append(((lo + hi) / 2, pred[m].sum()))

    data = []
    for ln in open(args.neutron_table):
        if ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) != 3:
            continue
        xf, val, errp = float(parts[0]), float(parts[1]), float(parts[2])
        rows_d = (points_p["xf"] >= xf - 0.055) & (points_p["xf"] < xf + 0.055)
        mv = None
        for lo, hi in zip(xedges[:-1], xedges[1:]):
            if lo <= xf < hi:
                mm = (points_p["xf"] >= lo) & (points_p["xf"] < hi)
                mv = pred[mm].sum()
                break
        else:
            mv = None
        if mv is not None and mv != 0:
            model.append((xf, mv))
    model_xf = [m[0] for m in model]
    model_val = [m[1] for m in model]
    anchor = None
    for xf_d, y_d in ((0.1, None),):
        pass
    # anchor at the first data point inside the paper window
    data_rows = []
    for ln in open(args.neutron_table):
        if ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) != 3:
            continue
        xf, val, errp = float(parts[0]), float(parts[1]), float(parts[2])
        if 0.05 <= xf <= 0.45:
            data_rows.append((xf, val, val * errp / 100.0))
    a_xf, a_val, _ = data_rows[0]
    anchor_val = None
    for xf_d, mv in model:
        if abs(xf_d - a_xf) <= 0.055:
            anchor_val = mv
            break
    print("Neutron falsifier (zero fit freedom), normalized at the first data point")
    print(f"{'xF':>5s} {'data dN/dxF':>12s} {'model norm':>12s} {'data/model':>10s}")
    chi2 = 0.0
    first = True
    for xf_d, y_d, e_d in data_rows:
        mv = None
        for lo, hi in zip(xedges[:-1], xedges[1:]):
            if lo <= xf_d < hi:
                mm = (points_p["xf"] >= lo) & (points_p["xf"] < hi)
                mv = pred[mm].sum()
                break
        if mv is None:
            continue
        mn = mv * (data_rows[0][1] / anchor_val0) if False else mv
        print(f"{xf_d:5.2f} {y_d:12.4f} {mv:12.4e}")
    # normalized comparison
    anchor_row = data_rows[0]
    for lo, hi in zip(xedges[:-1], xedges[1:]):
        if lo <= anchor_row[0] < hi:
            mm = (points_p["xf"] >= lo) & (points_p["xf"] < hi)
            anchor_model = pred[mm].sum()
    norm = anchor_row[1] / anchor_model
    print("normalized:")
    for xf_d, y_d, e_d in data_rows:
        mv = None
        for lo, hi in zip(xedges[:-1], xedges[1:]):
            if lo <= xf_d < hi:
                mm = (points_p["xf"] >= lo) & (points_p["xf"] < hi)
                mv = pred[mm].sum()
                break
        if mv is None:
            continue
        m_norm = mv * norm
        print(f"{xf_d:5.2f} {y_d:12.4f} {m_norm:12.4f}  ratio {y_d/m_norm:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
