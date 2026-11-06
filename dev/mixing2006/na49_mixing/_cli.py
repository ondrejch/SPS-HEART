# Not used by the HEART p+p fits (marked 2026-11-04): no production generation, fit, probe tool or paper script
# runs this module (an execution trace of the R137 generation and an import scan of the engine, the probe tools
# and the paper scripts). Code reviews of the p+p model can skip it. It stays until the cleanup and refactor after
# the p+A run.
"""Console entry points for the ``na49-mixing`` package.

The ``pyproject.toml`` script ``na49-generate`` maps here. Keep this module
lightweight: it wraps the same generation path used by examples and tests.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .data_io import save_histograms
from .glauber import NucleusSpec
from .histogram import Histogram1D
from .parallel import GenerationConfig, generate_tracks_chunked


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="na49-generate",
        description="Generate resonance cascade tracks with na49_mixing.",
    )
    p.add_argument(
        "--genfile",
        required=True,
        help="Path to a .gen card (e.g. na49_mixing/generators/.../*.gen).",
    )
    p.add_argument(
        "--outfile",
        required=True,
        help="Output .npz path for tracks (pdg, xf, pt, source_side) and meta.",
    )
    p.add_argument("--nevents", type=int, default=10_000, help="Number of parent events.")
    p.add_argument("--seed", type=int, default=1, help="Base RNG seed.")
    p.add_argument("--beam-momentum", type=float, default=158.0, help="Lab beam p (GeV/c).")
    p.add_argument(
        "--beam-pdg",
        type=int,
        default=2212,
        choices=(2212, 2112),
        help="Beam nucleon PDG id (2212=proton, 2112=neutron).",
    )
    p.add_argument(
        "--target-pdg",
        type=int,
        default=2212,
        choices=(2212, 2112),
        help="Target nucleon PDG id (2212=proton, 2112=neutron).",
    )
    p.add_argument(
        "--fragmentation-mode",
        choices=("projectile", "target", "both"),
        default="both",
        help="Which fragmentation side(s) to populate.",
    )
    p.add_argument(
        "--max-decay-passes",
        type=int,
        default=8,
        help="Cascade decay passes (1 = single pass).",
    )
    p.add_argument(
        "--decay-file",
        action="append",
        default=[],
        help="Decay table file (repeatable). Optional if using factories.",
    )
    p.add_argument(
        "--use-soft-decay-factory",
        action="store_true",
        help="Load build_soft_resonance_decays() tables.",
    )
    p.add_argument(
        "--use-kaon-decay-factory",
        action="store_true",
        help="Load build_kaon_decays() tables.",
    )
    p.add_argument(
        "--inject-antibaryon-resonances",
        action="store_true",
        help="Inject anti-baryon decay tables missing from the soft factory.",
    )
    p.add_argument(
        "--weak-feed-policy",
        choices=("inclusive_weak", "primary_only"),
        default="inclusive_weak",
        help="inclusive_weak keeps weak-decay daughters; primary_only drops them.",
    )
    p.add_argument(
        "--pa-a",
        type=int,
        default=0,
        help="Target mass number A; 0 keeps elementary nucleon-nucleon generation.",
    )
    p.add_argument("--pa-z", type=int, default=1)
    p.add_argument("--pa-name", default="")
    p.add_argument(
        "--pa-profile",
        choices=("point", "woods_saxon", "harmonic_oscillator"),
        default="woods_saxon",
    )
    p.add_argument("--pa-radius-fm", type=float, default=6.62)
    p.add_argument("--pa-diffuseness-fm", type=float, default=0.546)
    p.add_argument("--pa-quad-alpha", type=float, default=0.0)
    p.add_argument("--sigma-nn-inel-mb", type=float, default=31.7)
    p.add_argument(
        "--target-wound-range",
        default="",
        help="Centrality slice on wounded target nucleons, 'lo:hi'; empty = min-bias.",
    )
    p.add_argument(
        "--soft-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Soft-decay / tower override, e.g. nstar_eff_l1_mass=2.1",
    )
    p.add_argument("--n-workers", type=int, default=1, help="Process-pool workers.")
    p.add_argument("--n-chunks", type=int, default=0, help="Chunks (0 => n_workers).")
    p.add_argument(
        "--histograms",
        action="store_true",
        help="Also write simple parent-style xf/pt histograms next to --outfile.",
    )
    return p


def _parse_soft_overrides(items: Sequence[str]) -> tuple[tuple[str, float], ...]:
    out: List[tuple[str, float]] = []
    for raw in items:
        txt = str(raw).strip()
        if not txt:
            continue
        if "=" not in txt:
            raise SystemExit(f"Invalid --soft-override '{txt}' (expected KEY=VALUE)")
        key, val = txt.split("=", 1)
        out.append((key.strip(), float(val)))
    return tuple(sorted(out))


def generate_main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``na49-generate`` console script."""
    args = _build_parser().parse_args(argv)

    gen_file = Path(args.genfile)
    if not gen_file.is_file():
        print(f"ERROR: genfile not found: {gen_file}", file=sys.stderr)
        return 2

    soft_overrides = _parse_soft_overrides(args.soft_override)
    n_chunks = int(args.n_chunks) if int(args.n_chunks) > 0 else None

    nucleus = None
    wound_range = None
    if int(args.pa_a) > 0:
        nucleus = NucleusSpec(
            name=str(args.pa_name) or f"{args.pa_profile}{args.pa_a}",
            A=int(args.pa_a),
            Z=int(args.pa_z),
            profile=str(args.pa_profile),
            radius_fm=float(args.pa_radius_fm),
            diffuseness_fm=float(args.pa_diffuseness_fm),
            quad_alpha=float(args.pa_quad_alpha),
            sigma_nn_inel_mb=float(args.sigma_nn_inel_mb),
        )
        if str(args.target_wound_range).strip():
            lo_s, hi_s = str(args.target_wound_range).split(":")
            wound_range = (int(lo_s), int(hi_s))
    elif str(args.target_wound_range).strip():
        raise SystemExit("--target-wound-range requires --pa-a")

    cfg = GenerationConfig(
        gen_file=str(gen_file),
        n_events=int(args.nevents),
        seed=int(args.seed),
        beam_momentum=float(args.beam_momentum),
        beam_pdg=int(args.beam_pdg),
        target_pdg=int(args.target_pdg),
        max_decay_passes=int(args.max_decay_passes),
        decay_files=tuple(str(p) for p in args.decay_file),
        use_kaon_decay_factory=bool(args.use_kaon_decay_factory),
        use_soft_decay_factory=bool(args.use_soft_decay_factory),
        inject_antibaryon_resonances=bool(args.inject_antibaryon_resonances),
        soft_decay_overrides=soft_overrides,
        fragmentation_mode=str(args.fragmentation_mode),
        weak_feed_policy=str(args.weak_feed_policy),
        nucleus=nucleus,
        target_wound_range=wound_range,
    )

    t0 = time.time()
    result = generate_tracks_chunked(
        cfg,
        n_workers=int(args.n_workers),
        n_chunks=n_chunks,
    )
    elapsed = time.time() - t0

    out = Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "gen_file": str(gen_file),
        "n_events": int(args.nevents),
        "seed": int(args.seed),
        "beam_momentum": float(args.beam_momentum),
        "beam_pdg": int(args.beam_pdg),
        "target_pdg": int(args.target_pdg),
        "fragmentation_mode": str(args.fragmentation_mode),
        "weak_feed_policy": str(args.weak_feed_policy),
        "max_decay_passes": int(args.max_decay_passes),
        "n_workers": int(args.n_workers),
        "n_tracks": int(np.asarray(result["pdg"]).shape[0]),
        "n_decayed_parents": int(result.get("n_decayed_parents", 0)),
        "n_decay_kinematic_failures": int(result.get("n_decay_kinematic_failures", 0)),
        "n_decay_attempts": int(result.get("n_decay_attempts", 0)),
        "elapsed_s": float(elapsed),
        "soft_decay_overrides": dict(soft_overrides),
        "pa_nucleus": result.get("pa_nucleus"),
    }
    np.savez_compressed(
        out,
        pdg=np.asarray(result["pdg"], dtype=np.int32),
        xf=np.asarray(result["xf"], dtype=np.float32),
        pt=np.asarray(result["pt"], dtype=np.float32),
        source_side=np.asarray(result["source_side"], dtype=np.int8),
        weak_feed=np.asarray(result["weak_feed"], dtype=np.int8),
        gen_depth=np.asarray(result["gen_depth"], dtype=np.int16),
        n_wounds=np.asarray(result["n_wounds"], dtype=np.int16),
        meta_json=np.asarray(json.dumps(meta)),
    )

    if args.histograms:
        xf_h = Histogram1D(40, -1.0, 1.0, name="xf")
        pt_h = Histogram1D(50, 0.0, 2.5, name="pt")
        xf_h.fill_array(np.asarray(result["xf"], dtype=float))
        pt_h.fill_array(np.asarray(result["pt"], dtype=float))
        hist_path = out.with_name(out.stem + "_hist.npz")
        save_histograms(str(hist_path), {"xf": xf_h, "pt": pt_h})

    fail = int(result.get("n_decay_kinematic_failures", 0))
    att = int(result.get("n_decay_attempts", 0))
    print(
        f"Wrote {meta['n_tracks']} tracks from {args.nevents} events to {out} "
        f"in {elapsed:.2f}s "
        f"(decay attempts={att}, kinematic_failures={fail})",
        flush=True,
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return generate_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
