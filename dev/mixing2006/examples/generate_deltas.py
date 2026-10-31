#!/usr/bin/env python3
"""Generate Delta resonance events — Python port of ``generate.cxx``.

Usage::

    python generate_deltas.py <resonance_name> [options]

where ``<resonance_name>`` is the base name of a ``.gen`` file located in the
same directory (or the current working directory).  The script reads
``<resonance_name>.gen``, builds decay tables either from
``build_soft_resonance_decays()`` (bundled, no external files) or from a
decay table file given with ``--decays``, and writes an ``.npz`` output.

Example (no files needed beyond this repository)::

    python examples/generate_deltas.py delta++ \\
        --genfile na49_mixing/generators/michal/deltas-to-pion/delta++.gen \\
        --use-soft-decay-factory --nevents 500

Example using a historic decay table from a full development checkout::

    python examples/generate_deltas.py delta++ \\
        --decays .legacy/resonance-picture/michal/deltas-to-pion/decays
"""

from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np

# Allow running directly from the examples directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from na49_mixing.reaction import Reaction
from na49_mixing.generator import Generator
from na49_mixing.event import Event
from na49_mixing.histogram import Histogram1D
from na49_mixing.decays import build_soft_resonance_decays
from na49_mixing.data_io import save_histograms


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate Delta resonance events")
    parser.add_argument("resname", help="Base name of the .gen file (e.g. delta++)")
    parser.add_argument("--nevents", type=int, default=1_000_000,
                        help="Number of events to generate (default: 1 000 000)")
    parser.add_argument("--decays", default="decays",
                        help="Path to the decay table file (default: 'decays'; "
                             "ignored when --use-soft-decay-factory is set)")
    parser.add_argument("--use-soft-decay-factory", action="store_true",
                        help="Build decay tables with build_soft_resonance_decays() "
                             "instead of reading --decays; needs no files outside "
                             "this repository")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional RNG seed for reproducible runs "
                             "(default: system entropy)")
    parser.add_argument("--genfile", default=None,
                        help="Path to .gen file (default: <resname>.gen)")
    parser.add_argument("--outfile", default=None,
                        help="Output .npz file (default: <resname>.npz)")
    args = parser.parse_args(argv)

    gen_file = args.genfile or f"{args.resname}.gen"
    out_file = args.outfile or f"{args.resname}.npz"
    decays_file = args.decays

    if not os.path.exists(gen_file):
        sys.exit(f"ERROR: generator file '{gen_file}' not found")
    if not args.use_soft_decay_factory and not os.path.exists(decays_file):
        sys.exit(f"ERROR: decay file '{decays_file}' not found")

    rng = np.random.default_rng(args.seed)

    # --- Set up reaction and generator ---
    r = Reaction(158.0)
    gen = Generator(reaction=r, rng=rng)
    gen.set_all(gen_file)
    print(f"Loaded generator from '{gen_file}'", flush=True)

    # --- Set up event with decay chain ---
    ev = Event(rng=rng)
    if args.use_soft_decay_factory:
        ev._decay_chain = build_soft_resonance_decays(rng=rng)
        print("Built soft-resonance decay tables (no external files needed)",
              flush=True)
    else:
        ev.read_decays(decays_file)
        print(f"Loaded decay table from '{decays_file}'", flush=True)

    # --- Histograms ---
    mp   = Histogram1D(100, 1.0, 2.5,  name="mp")    # parent mass
    xfp  = Histogram1D(40, -1.0, 1.0,  name="xfp")   # parent xF
    xfdp = Histogram1D(40, -1.0, 1.0,  name="xfdp")  # pi+ xF after decay
    xfdz = Histogram1D(40, -1.0, 1.0,  name="xfdz")  # pi- xF after decay

    n_events = args.nevents
    t0 = time.time()

    for i in range(n_events):
        particle = gen.generate()
        if particle is None:
            continue

        ev.add_particle(particle)
        ev.set_to_first()
        p = ev.get_next()
        if p is not None:
            mp.fill(p.get_mass())
            xfp.fill(p.get_xf())

        ev.decay()

        for p in ev:
            if p.get_id() == 211:
                xfdp.fill(p.get_xf())
            elif p.get_id() == -211:
                xfdz.fill(p.get_xf())

        ev.delete()

        if (i + 1) % 10_000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(
                f"\r{i + 1:8d} / {n_events}  ({rate:.0f} ev/s)  ",
                end="",
                flush=True,
            )

    print(f"\nGenerated {n_events} events in {time.time() - t0:.1f} s")

    histograms = {"mp": mp, "xfp": xfp, "xfdp": xfdp, "xfdz": xfdz}
    save_histograms(out_file, histograms)
    print(f"Saved histograms to '{out_file}'")


if __name__ == "__main__":
    main()
