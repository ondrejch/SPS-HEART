#!/usr/bin/env python3
"""Compose six shared-fit species plots into a 2x3 grid figure."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig-dir", default=".")
    parser.add_argument("--out-png", default="spectra_sharedfit_bestmix_all6_species_grid.png")
    args = parser.parse_args()

    fig_dir = Path(args.fig_dir)
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    panels: List[Tuple[str, str]] = [
        ("pi-", "piminus_spectra_sharedfit_bestmix.png"),
        ("pi+", "piplus_spectra_sharedfit_bestmix.png"),
        ("K+", "kplus_spectra_sharedfit_bestmix.png"),
        ("K-", "kminus_spectra_sharedfit_bestmix.png"),
        ("p", "p_spectra_sharedfit_bestmix.png"),
        ("pbar", "antiproton_spectra_sharedfit_bestmix.png"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (title, fname) in zip(axes.ravel(), panels):
        path = fig_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing panel image: {path}")
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.suptitle("Shared-fit spectra comparison across six species", fontsize=14)
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.96))
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"Saved figure: {out_png}")


if __name__ == "__main__":
    main()
