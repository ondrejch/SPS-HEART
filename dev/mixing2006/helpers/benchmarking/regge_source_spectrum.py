#!/usr/bin/env python3
"""Reproduce the round-7 E-C production-side source-mass spectrum.

The round-7 confrontation (notes/benchmarks/STAGE_C_ROUND7_DECISION_2026-09-04.md)
compared the production-side Regge source-mass spectrum with the fitted
effective tower. This script regenerates every quoted number: the
spectrum statistics (mean, median, tail fractions, the coarse
histogram), the simple-mode cross-check, and the coverage of the
fitted l1 and l2 windows.

Usage: python helpers/benchmarking/regge_source_spectrum.py [--n-samples N]
Seed-fixed (20260831) so the output is reproducible run to run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

MIXING2006 = Path(__file__).resolve().parents[2] / "dev" / "mixing2006"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=400_000)
    args = parser.parse_args()

    mixdir = str(MIXING2006)
    if mixdir not in sys.path:
        sys.path.insert(0, mixdir)
    from na49_mixing.event_builder_pp import PPEventBuilder, PPExcitationConfig
    from na49_mixing.exchange_model import ReggeExcitationParameters

    rng = np.random.default_rng(20260831)
    cfg = PPExcitationConfig(beam_momentum=158.0, production_mode="regge")
    builder = PPEventBuilder(cfg, rng=rng)
    masses = np.array([builder.mass_sampler.sample()[0] for _ in range(args.n_samples)])

    print(f"Regge source-mass spectrum, {args.n_samples} samples, seed 20260831")
    print(f"  mean {masses.mean():.2f} GeV, median {float(np.median(masses)):.2f} GeV")
    bins = [1.0, 1.5, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.5, 5.0, 6.0, 8.0, 12.0, 17.4]
    hist, edges = np.histogram(masses, bins=bins)
    for h, lo, hi in zip(hist, edges[:-1], edges[1:]):
        print(f"   [{lo:5.1f},{hi:5.1f}): {h:7d}  ({100 * h / len(masses):5.2f} %)")
    print(f"  fraction above 4.5: {100 * float((masses > 4.5).mean()):.2f} %")
    print(f"  fraction above 6.0: {100 * float((masses > 6.0).mean()):.2f} %")
    print(f"  l1 window [1.65, 2.15]: {100 * float(((masses >= 1.65) & (masses <= 2.15)).mean()):.2f} %")
    print(f"  l2 window [4.35, 4.65]: {100 * float(((masses >= 4.35) & (masses <= 4.65)).mean()):.2f} %")
    print(f"  l2 narrow core [4.47, 4.53]: {100 * float(((masses >= 4.47) & (masses <= 4.53)).mean()):.3f} %")

    cfg_simple = PPExcitationConfig(beam_momentum=158.0, production_mode="simple")
    builder_simple = PPEventBuilder(cfg_simple, rng=np.random.default_rng(20260831))
    simple = np.array(
        [builder_simple.mass_sampler.sample()[0] for _ in range(args.n_samples)]
    )
    print(f"simple mode: mean {simple.mean():.2f} GeV, "
          f"fraction above 4.5: {100 * float((simple > 4.5).mean()):.2f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
