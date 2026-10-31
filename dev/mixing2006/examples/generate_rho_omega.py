#!/usr/bin/env python3
"""Generate ρ and ω resonance events — example script.

Creates ρ⁰(770) and ω(782) resonances, decays them to their dominant
charged-pion final states, and histograms the resulting π⁺π⁻ invariant mass.

Usage::

    python generate_rho_omega.py [--nevents N]

Example::

    cd python/examples
    python generate_rho_omega.py --nevents 100000
"""

from __future__ import annotations
import argparse
import math
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
from na49_mixing.breit_wigner import BreitWigner
from na49_mixing.decays import build_rho_decays, build_omega_decays
from na49_mixing.data_io import save_histograms
from na49_mixing.constants import EMRHO, EMOMEGA, EMPI


def _make_flat_y_ymax(n_bins: int = 100) -> Histogram1D:
    """Return a uniform y/ymax histogram (flat rapidity distribution)."""
    h = Histogram1D(n_bins, 0.0, 1.0)
    for i in range(1, n_bins + 1):
        h.set_bin_content(i, 1.0)
    return h


def _build_generator(pdg_id: int, reaction: Reaction, bw: BreitWigner,
                     rng: np.random.Generator) -> Generator:
    """Build a simple Generator for a meson with Breit-Wigner mass sampling."""
    gen = Generator(pdg_id=pdg_id, reaction=reaction, rng=rng)
    # pT: Gaussian with inverse-width parameter (≈ exponential slope ~ 4 GeV⁻¹)
    gen.set_pt(1, 4.0)
    # y/ymax: flat distribution
    gen.set_y_ymax(1, _make_flat_y_ymax())
    # Mass: BW distribution
    gen.set_mass_bw(bw)
    return gen


def generate_rho(n_events: int, rng: np.random.Generator,
                 reaction: Reaction) -> dict:
    """Generate ρ⁰(770)→π⁺π⁻ events and return histogram dict."""
    bw_rho = BreitWigner(0.2791, EMRHO, 0.1491, 1.0, 1.5, 2.0, rng=rng)
    gen = _build_generator(113, reaction, bw_rho, rng)
    rho_decays = build_rho_decays(rng=rng)

    mp    = Histogram1D(100, 0.4, 1.2, name="rho_mp")    # parent BW mass
    minv  = Histogram1D(100, 0.4, 1.2, name="rho_minv")  # π⁺π⁻ inv. mass
    xfp   = Histogram1D(40, -1.0, 1.0, name="rho_xfp")   # parent xF
    xfdp  = Histogram1D(40, -1.0, 1.0, name="rho_xfdp")  # π⁺ xF
    xfdm  = Histogram1D(40, -1.0, 1.0, name="rho_xfdm")  # π⁻ xF

    t0 = time.time()
    for i in range(n_events):
        particle = gen.generate()
        if particle is None:
            continue

        mp.fill(particle.get_mass())
        xfp.fill(particle.get_xf())

        ev = Event(rng=rng)
        ev.add_particle(particle)
        ev._decay_chain = rho_decays
        ev.decay()

        pi_plus = pi_minus = None
        for p in ev:
            if p.get_id() == 211:
                pi_plus = p
                xfdp.fill(p.get_xf())
            elif p.get_id() == -211:
                pi_minus = p
                xfdm.fill(p.get_xf())

        if pi_plus is not None and pi_minus is not None:
            minv.fill(pi_plus.get_inv_mass(pi_minus))

        if (i + 1) % 10_000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"\r  rho: {i + 1:8d}/{n_events}  ({rate:.0f} ev/s)  ",
                  end="", flush=True)

    print()
    return {"rho_mp": mp, "rho_minv": minv,
            "rho_xfp": xfp, "rho_xfdp": xfdp, "rho_xfdm": xfdm}


def generate_omega(n_events: int, rng: np.random.Generator,
                   reaction: Reaction) -> dict:
    """Generate ω(782)→π⁺π⁻π⁰ events and return histogram dict."""
    bw_omega = BreitWigner(0.4187, EMOMEGA, 0.00849, 1.0, 1.2, 1.5, rng=rng)
    gen = _build_generator(223, reaction, bw_omega, rng)
    omega_decays = build_omega_decays(rng=rng)

    mp    = Histogram1D(100, 0.6, 1.0, name="omega_mp")    # parent BW mass
    minv  = Histogram1D(100, 0.2, 0.9, name="omega_minv")  # π⁺π⁻ inv. mass
    xfp   = Histogram1D(40, -1.0, 1.0, name="omega_xfp")   # parent xF
    xfdp  = Histogram1D(40, -1.0, 1.0, name="omega_xfdp")  # π⁺ xF
    xfdm  = Histogram1D(40, -1.0, 1.0, name="omega_xfdm")  # π⁻ xF

    t0 = time.time()
    for i in range(n_events):
        particle = gen.generate()
        if particle is None:
            continue

        mp.fill(particle.get_mass())
        xfp.fill(particle.get_xf())

        ev = Event(rng=rng)
        ev.add_particle(particle)
        ev._decay_chain = omega_decays
        ev.decay()

        pi_plus = pi_minus = None
        for p in ev:
            if p.get_id() == 211:
                pi_plus = p
                xfdp.fill(p.get_xf())
            elif p.get_id() == -211:
                pi_minus = p
                xfdm.fill(p.get_xf())

        if pi_plus is not None and pi_minus is not None:
            minv.fill(pi_plus.get_inv_mass(pi_minus))

        if (i + 1) % 10_000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"\r  omega: {i + 1:8d}/{n_events}  ({rate:.0f} ev/s)  ",
                  end="", flush=True)

    print()
    return {"omega_mp": mp, "omega_minv": minv,
            "omega_xfp": xfp, "omega_xfdp": xfdp, "omega_xfdm": xfdm}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate rho and omega resonance events")
    parser.add_argument("--nevents", type=int, default=50_000,
                        help="Number of events per resonance (default: 50 000)")
    parser.add_argument("--outfile-rho", default="rho770.npz",
                        help="Output .npz file for rho results (default: rho770.npz)")
    parser.add_argument("--outfile-omega", default="omega782.npz",
                        help="Output .npz file for omega results (default: omega782.npz)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional RNG seed for reproducible runs "
                             "(default: system entropy)")
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    reaction = Reaction(158.0)

    print(f"Generating {args.nevents} rho(770) events ...")
    rho_hists = generate_rho(args.nevents, rng, reaction)
    save_histograms(args.outfile_rho, rho_hists)
    print(f"Saved rho histograms to '{args.outfile_rho}'")

    print(f"Generating {args.nevents} omega(782) events ...")
    omega_hists = generate_omega(args.nevents, rng, reaction)
    save_histograms(args.outfile_omega, omega_hists)
    print(f"Saved omega histograms to '{args.outfile_omega}'")


if __name__ == "__main__":
    main()
