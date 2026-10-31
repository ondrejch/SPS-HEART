#!/usr/bin/env python3
"""Plot invariant-mass spectra — Python port of ``plots.cxx``.

Reads ``.npz`` files produced by ``generate_deltas.py`` and produces
a matplotlib figure with x_F distributions of the resonances and
their decay products.

Usage::

    python plot_spectra.py [--files delta++.npz delta+.npz ...]
                          [--labels "Delta++" "Delta+" ...]
                          [--out plots.pdf]
"""

from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for batch use
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from na49_mixing.data_io import load_histograms


def plot_spectra(npz_files, labels=None, out="plots.pdf", scale=8.97):
    """Main plotting routine.

    Parameters
    ----------
    npz_files : list of str
        Path to ``.npz`` files.
    labels : list of str, optional
        Legend labels for each file.
    out : str
        Output PDF/PNG file.
    scale : float
        Cross-section scale (mb) for normalisation.
    """
    if labels is None:
        labels = [os.path.splitext(os.path.basename(f))[0] for f in npz_files]

    n_files = len(npz_files)
    fig, axes = plt.subplots(1, n_files, figsize=(5 * n_files, 5), sharey=False)
    if n_files == 1:
        axes = [axes]

    for ax, path, label in zip(axes, npz_files, labels):
        hists = load_histograms(path)

        # Parent x_F
        if "xfp" in hists:
            h = hists["xfp"]
            bw = h.bin_width
            xs = np.array([h.get_bin_center(i + 1) for i in range(h.nbins)])
            ys = np.array([h.get_bin_content(i + 1) for i in range(h.nbins)])
            n_events = h.get_entries()
            norm = scale / n_events / bw if n_events > 0 else 1.0
            # forward hemisphere only (xF > 0)
            mask = xs > 0.0
            ax.plot(xs[mask], ys[mask] * norm, "o-", markersize=4, label=label + " (parent)")

        # pi+ x_F
        if "xfdp" in hists:
            h = hists["xfdp"]
            bw = h.bin_width
            xs = np.array([h.get_bin_center(i + 1) for i in range(h.nbins)])
            ys = np.array([h.get_bin_content(i + 1) for i in range(h.nbins)])
            n_events_ref = (
                hists["xfp"].get_entries() if "xfp" in hists else h.get_entries()
            )
            norm = scale / max(n_events_ref, 1) / bw
            mask = xs > 0.0
            ax.plot(xs[mask], ys[mask] * norm, "s--", markersize=4, label=r"$\pi^+$")

        # pi- x_F
        if "xfdz" in hists:
            h = hists["xfdz"]
            bw = h.bin_width
            xs = np.array([h.get_bin_center(i + 1) for i in range(h.nbins)])
            ys = np.array([h.get_bin_content(i + 1) for i in range(h.nbins)])
            n_events_ref = (
                hists["xfp"].get_entries() if "xfp" in hists else h.get_entries()
            )
            norm = scale / max(n_events_ref, 1) / bw
            mask = xs > 0.0
            ax.plot(xs[mask], ys[mask] * norm, "^:", markersize=4, label=r"$\pi^-$")

        ax.set_xlabel(r"$X_F$")
        ax.set_ylabel(r"$\mathrm{d}\sigma/\mathrm{d}X_F$ (mb)")
        ax.set_yscale("log")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved plot to '{out}'")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot Delta decay spectra")
    parser.add_argument(
        "--files", nargs="+", default=None,
        help="Input .npz files (default: all *.npz in current directory)",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Legend labels (one per file)",
    )
    parser.add_argument("--out", default="plots.pdf", help="Output file")
    parser.add_argument("--scale", type=float, default=8.97,
                        help="Cross-section scale in mb (default: 8.97)")
    args = parser.parse_args(argv)

    files = args.files
    if files is None:
        import glob
        files = sorted(glob.glob("*.npz"))
        if not files:
            sys.exit("No .npz files found. Run generate_deltas.py first.")

    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        sys.exit(f"Files not found: {missing}")

    plot_spectra(files, labels=args.labels, out=args.out, scale=args.scale)


if __name__ == "__main__":
    main()
