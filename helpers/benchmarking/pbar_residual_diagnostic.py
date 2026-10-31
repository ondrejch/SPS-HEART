#!/usr/bin/env python3
"""Reproduce the campaign-closing residual diagnostic (round 8/9 report).

Re-runs the exact fit machinery on the pulled pbar component caches of
the retained B-850P A/B run and its B0 ablation, and prints the x_F and
p_T residual decomposition quoted in
notes/benchmarks/STAGE_C_ROUND89_DECISION_2026-09-04.md.

Requires the pbar component caches under --cache-dir (default
/tmp/opencode/diag_pbar) and the ab comparison JSONs under
--artifacts (default /tmp/opencode/stageC_ab).
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
    parser.add_argument("--cache-dir", default="/tmp/opencode/diag_pbar")
    args = parser.parse_args()

    sys.path.insert(0, str(_repo))
    sys.path.insert(0, str(MIXING2006))
    from helpers.benchmarking.particle_definitions import resolve_species
    from helpers.benchmarking.staged_resonance_fit import (
        _build_design_matrix,
        _fit_and_score,
        _load_species_points,
        _train_test_split,
    )

    art = Path(args.artifacts)
    cdir = Path(args.cache_dir)
    table = Path(resolve_species("pbar").na49_table)
    points = _load_species_points(table, xf_min=0.0, xf_max=0.4, pt_min=0.05, pt_max=1.5)
    d = json.load(open(art / "ab_B850P.json"))
    sqrt_s = d["per_species"]["pbar"]["b"]["base"]["sqrt_s_gev"]

    def build(tag: str):
        dd = json.load(open(art / f"{tag}.json"))
        comps = dd["per_species"]["pbar"]["b"]["base"]["components"]
        arrs = []
        for lbl, c in comps.items():
            cf = cdir / Path(c["cache_file"]).name
            with np.load(cf, allow_pickle=False) as z:
                arrs.append((lbl, {k: z[k] for k in ("pdg", "xf", "pt", "source_side")}))
        labels, matrix = _build_design_matrix(
            species_pdg=-2212, components=arrs, points=points, sqrt_s_gev=sqrt_s,
        )
        return labels, matrix

    labels, mat = build("ab_B850P")
    _, mat0 = build("ab0_B850P")
    y, sigma = points["y"], points["sigma"]
    train_idx, test_idx = _train_test_split(len(y), 20260831, 0.7)
    w, _info, _tr, te = _fit_and_score(mat, y, sigma, train_idx, test_idx)
    print(f"recomputed test WRRMSE: {te['test_weighted_rel_rmse']:.6f} (reported 0.340648)")
    pred = mat @ w
    rel = (pred - y) / y
    print(f"mean relative residual {rel.mean():+.3f}, mean |relative| {np.abs(rel).mean():.3f}")
    for edges, vals, name in (
        ([0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4], points["xf"], "xF"),
        ([0.05, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5], points["pt"], "pT"),
    ):
        print(f"--- by {name}")
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (vals >= lo) & (vals < hi)
            if m.sum():
                print(f"  {name} [{lo:.2f},{hi:.2f}): n={m.sum():3d}  mean rel {rel[m].mean():+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
