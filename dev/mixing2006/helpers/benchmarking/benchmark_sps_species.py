#!/usr/bin/env python3
"""Benchmark SPS hadron spectra against MC and resonance hypotheses.

Inputs are NA49 SPS tabulated spectra in `xF pT value error` format.
Outputs are summary/details files per selected species.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import nnls

from helpers.benchmarking.particle_definitions import resolve_species
from helpers.benchmarking import sps_core

MPROTON_GEV = sps_core.MPROTON_GEV
PDG_MASS_GEV = sps_core.PDG_MASS_GEV
REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "00runs"
FRAGMENTATION_MODES: tuple[str, ...] = ("projectile", "target", "both")
FIT_WEIGHT_MODES: tuple[str, ...] = sps_core.FIT_WEIGHT_MODES
NUMPY_VERSION = str(np.__version__)
NUMPY_BIT_GENERATOR = type(np.random.default_rng().bit_generator).__name__

# Backward-compatible private aliases (call sites throughout this module).
_sqrt_s_from_fixed_target_beam_momentum = sps_core.sqrt_s_from_fixed_target_beam_momentum
_read_na49_points = sps_core.read_na49_points
_edges_from_centers = sps_core.edges_from_centers
_grid_indices = sps_core.grid_indices
_counts_on_grid = sps_core.counts_on_grid
_invariant_track_weights = sps_core.invariant_track_weights
_grid_point_areas = sps_core.grid_point_areas
_fit_single_scale = sps_core.fit_single_scale
_rank_bin_indices = sps_core.rank_bin_indices
_compute_fit_row_scales = sps_core.compute_fit_row_scales
_chi2 = sps_core.chi2
_log_chi2 = sps_core.log_chi2
_smape_percent = sps_core.smape_percent
_relative_rmse_weighted = sps_core.relative_rmse_weighted
_metrics_bundle = sps_core.metrics_bundle


@dataclass(frozen=True)
class ResonanceSpec:
    label: str
    gen_file: Path
    decay_files: Tuple[Path, ...] = ()
    use_kaon_decay_factory: bool = False
    use_soft_decay_factory: bool = False


def _run_pythia(
    n_events: int,
    seed: int,
    sqrt_s: float,
    particle_pdg: int,
    xedges: np.ndarray,
    pedges: np.ndarray,
    ix_data: np.ndarray,
    ip_data: np.ndarray,
    convention: str = "raw",
) -> np.ndarray:
    import pythia8mc as p8

    p = p8.Pythia()
    p.readString("Print:quiet = on")
    p.readString("Random:setSeed = on")
    p.readString(f"Random:seed = {seed}")
    p.readString("Beams:idA = 2212")
    p.readString("Beams:idB = 2212")
    p.readString(f"Beams:eCM = {sqrt_s:.8f}")
    p.readString("SoftQCD:all = on")
    ok = p.init()
    if not ok:
        raise RuntimeError("PYTHIA failed to initialize.")

    xf_vals: List[float] = []
    pt_vals: List[float] = []
    pmax = sqrt_s / 2.0

    accepted = 0
    for _ in range(n_events):
        if not p.next():
            continue
        accepted += 1
        ev = p.event
        for i in range(ev.size()):
            par = ev[i]
            if not par.isFinal() or par.id() != particle_pdg:
                continue
            xf_vals.append(par.pz() / pmax)
            pt_vals.append(par.pT())

    if accepted == 0:
        raise RuntimeError("PYTHIA generated zero accepted events.")

    xf_arr = np.asarray(xf_vals)
    pt_arr = np.asarray(pt_vals)
    if convention == "invariant":
        w = _invariant_track_weights(particle_pdg, xf_arr, pt_arr, sqrt_s)
        return _counts_on_grid(
            xf_arr, pt_arr, xedges, pedges, ix_data, ip_data, weights=w
        ) / _grid_point_areas(xedges, pedges, ix_data, ip_data)
    return _counts_on_grid(
        xf_arr,
        pt_arr,
        xedges,
        pedges,
        ix_data,
        ip_data,
    )


def _import_mixing2006(mixing2006_dir: Path):
    if str(mixing2006_dir) not in sys.path:
        sys.path.insert(0, str(mixing2006_dir))
    from na49_mixing.event import Event
    from na49_mixing.generator import Generator
    from na49_mixing.reaction import Reaction
    from na49_mixing.decays import (
        DecayMode,
        ParticleDecays,
        build_kaon_decays,
        build_soft_resonance_decays,
        read_decays,
    )

    return Event, Generator, Reaction, DecayMode, ParticleDecays, build_kaon_decays, build_soft_resonance_decays, read_decays


def _resonance_specs_for_particle(
    particle_pdg: int,
    base_dir: Path,
    include_regge_tower: bool = True,
    regge_tower_gen_files: Optional[Dict[str, Path]] = None,
) -> List[ResonanceSpec]:
    michal = base_dir / "na49_mixing" / "generators" / "michal"
    gen_dir = base_dir / "na49_mixing" / "generators" / "benchmark"

    def _soft(label: str, gen_path: Path) -> ResonanceSpec:
        return ResonanceSpec(label=label, gen_file=gen_path, use_soft_decay_factory=True)

    def _tower_path(label: str, default_path: Path) -> Path:
        if regge_tower_gen_files is None:
            return default_path
        return regge_tower_gen_files.get(label, default_path)

    def _tower_labels(anti: bool) -> List[str]:
        if regge_tower_gen_files:
            if anti:
                return [
                    lbl
                    for lbl in regge_tower_gen_files.keys()
                    if lbl.startswith("anti_nstar_eff") and lbl.endswith("_zero")
                ]
            return [
                lbl
                for lbl in regge_tower_gen_files.keys()
                if (not lbl.startswith("anti_")) and lbl.startswith("nstar_eff") and lbl.endswith("_zero")
            ]
        if anti:
            return ["anti_nstar_eff1900_zero", "anti_nstar_eff2100_zero", "anti_nstar_eff2300_zero"]
        return ["nstar_eff1900_zero", "nstar_eff2100_zero", "nstar_eff2300_zero"]

    if particle_pdg == -211:
        specs = [
            _soft("delta++", michal / "deltas-to-pion" / "delta++.gen"),
            _soft("delta+", michal / "deltas-to-pion" / "delta+.gen"),
            _soft("delta0", michal / "deltas-to-pion" / "delta0.gen"),
            _soft("delta-", michal / "deltas-to-pion" / "delta-.gen"),
            _soft(
                "n1440_inclusive",
                michal / "nstar-to-delta" / "n1440-zero.gen",
            ),
            _soft(
                "n1520_inclusive",
                michal / "nstar-to-delta" / "n1520-zero.gen",
            ),
            _soft(
                "n1680_inclusive",
                michal / "nstar-to-delta" / "n1680-zero.gen",
            ),
            _soft("eta", gen_dir / "eta_548.gen"),
            _soft("rho0", michal / "mesons-to-pion" / "rho0.gen"),
            _soft("rho_minus", gen_dir / "rho_minus_770.gen"),
            _soft("omega", michal / "mesons-to-pion" / "omega.gen"),
            _soft("f2", gen_dir / "f2_1270.gen"),
            _soft("rho3_zero", gen_dir / "rho3_zero_1690.gen"),
            _soft("rho3_minus", gen_dir / "rho3_minus_1690.gen"),
            _soft("f4", gen_dir / "f4_2050.gen"),
        ]
        if include_regge_tower:
            specs.extend(
                [
                    _soft(lbl, _tower_path(lbl, gen_dir / f"{lbl}.gen"))
                    for lbl in _tower_labels(anti=False)
                ]
            )
        return specs

    if particle_pdg == 211:
        specs = [
            _soft("delta++", michal / "deltas-to-pion" / "delta++.gen"),
            _soft("delta+", michal / "deltas-to-pion" / "delta+.gen"),
            _soft("delta0", michal / "deltas-to-pion" / "delta0.gen"),
            _soft("delta-", michal / "deltas-to-pion" / "delta-.gen"),
            _soft("n1440_inclusive", michal / "nstar-to-delta" / "n1440-zero.gen"),
            _soft("n1520_inclusive", michal / "nstar-to-delta" / "n1520-zero.gen"),
            _soft("n1680_inclusive", michal / "nstar-to-delta" / "n1680-zero.gen"),
            _soft("eta", gen_dir / "eta_548.gen"),
            _soft("rho0", michal / "mesons-to-pion" / "rho0.gen"),
            _soft("rho_plus", gen_dir / "rho_plus_770.gen"),
            _soft("omega", michal / "mesons-to-pion" / "omega.gen"),
            _soft("f2", gen_dir / "f2_1270.gen"),
            _soft("rho3_zero", gen_dir / "rho3_zero_1690.gen"),
            _soft("rho3_plus", gen_dir / "rho3_plus_1690.gen"),
            _soft("f4", gen_dir / "f4_2050.gen"),
        ]
        if include_regge_tower:
            specs.extend(
                [
                    _soft(lbl, _tower_path(lbl, gen_dir / f"{lbl}.gen"))
                    for lbl in _tower_labels(anti=False)
                ]
            )
        return specs

    if particle_pdg == 2212:
        specs = [
            _soft("delta++", michal / "deltas-to-proton" / "delta++.gen"),
            _soft("delta+", michal / "deltas-to-proton" / "delta+.gen"),
            _soft("delta0", michal / "deltas-to-proton" / "delta0.gen"),
            _soft("delta-", michal / "deltas-to-proton" / "delta-.gen"),
            _soft("n1440", michal / "nstar-to-proton" / "n1440-zero.gen"),
            _soft("n1520", michal / "nstar-to-proton" / "n1520-zero.gen"),
            _soft("n1680", michal / "nstar-to-proton" / "n1680-zero.gen"),
        ]
        if include_regge_tower:
            specs.extend(
                [
                    _soft(lbl, _tower_path(lbl, gen_dir / f"{lbl}.gen"))
                    for lbl in _tower_labels(anti=False)
                ]
            )
        return specs

    if particle_pdg == -2212:
        specs = [
            _soft("anti_delta_minusminus", gen_dir / "anti_delta_minusminus.gen"),
            _soft("anti_delta_minus", gen_dir / "anti_delta_minus.gen"),
            _soft("anti_delta_zero", gen_dir / "anti_delta_zero.gen"),
            _soft("anti_n1440_inclusive", gen_dir / "anti_n1440_zero.gen"),
            _soft("anti_n1520_inclusive", gen_dir / "anti_n1520_zero.gen"),
            _soft("anti_n1680_inclusive", gen_dir / "anti_n1680_zero.gen"),
            _soft("anti_n1535_to_pbar", gen_dir / "anti_n1535_zero.gen"),
            _soft("anti_n1650_to_pbar", gen_dir / "anti_n1650_zero.gen"),
            _soft("anti_n1720_to_pbar", gen_dir / "anti_n1720_zero.gen"),
        ]
        if include_regge_tower:
            specs.extend(
                [
                    _soft(lbl, _tower_path(lbl, gen_dir / f"{lbl}.gen"))
                    for lbl in _tower_labels(anti=True)
                ]
            )
        return specs

    if abs(particle_pdg) == 321:
        return [
            _soft("kstar_plus_892", gen_dir / "kstar_plus_892.gen"),
            _soft("kstar_minus_892", gen_dir / "kstar_minus_892.gen"),
            _soft("kstar_zero_892", gen_dir / "kstar_zero_892.gen"),
            _soft("kstar_zero_bar_892", gen_dir / "kstar_zero_bar_892.gen"),
            _soft("phi_1020", gen_dir / "phi_1020.gen"),
            _soft("f2_1270", gen_dir / "f2_1270.gen"),
            _soft("rho3_zero_1690", gen_dir / "rho3_zero_1690.gen"),
            _soft("rho3_plus_1690", gen_dir / "rho3_plus_1690.gen"),
            _soft("rho3_minus_1690", gen_dir / "rho3_minus_1690.gen"),
            _soft("f4_2050", gen_dir / "f4_2050.gen"),
        ]

    return []


def _resonance_spec_key(spec: ResonanceSpec, base_dir: Path) -> str:
    try:
        rel = spec.gen_file.relative_to(base_dir).as_posix()
    except ValueError:
        rel = str(spec.gen_file)
    return f"{spec.label}::{rel}"


def _canonical_nucleus_namespace(
    nucleus: Optional[Dict[str, object]],
    target_wound_range: Optional[Tuple[int, int]],
    data_system: str = "pp",
) -> Optional[str]:
    """Canonical namespace string separating pA cache/seed domains.

    ``None`` (p+p, no nucleus) keeps historical keys untouched. When a
    nucleus is set the namespace encodes system + full nucleus spec +
    wound range so component caches and component seeds can never mix
    across systems or centrality slices.
    """
    if not nucleus:
        return None
    payload = {
        "data_system": str(data_system),
        "nucleus": {str(k): nucleus[k] for k in sorted(nucleus)},
        "target_wound_range": (
            [int(target_wound_range[0]), int(target_wound_range[1])]
            if target_wound_range
            else None
        ),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def species_namespace_tag(beam_pdg: int, target_pdg: int) -> Optional[str]:
    """Namespace tag separating neutron-system caches from proton ones.

    ``None`` for the historical p+p default so existing cache files and
    component seeds stay untouched. Neutron systems return 'np', 'pn',
    or 'nn' so an n+p run can never read or overwrite p+p cache files.
    """
    beam, targ = int(beam_pdg), int(target_pdg)
    if beam == 2212 and targ == 2212:
        return None
    name = {2212: "p", 2112: "n"}.get(beam, "?") + {2212: "p", 2112: "n"}.get(targ, "?")
    return name


def wounded_pair_factor(nucleus: Dict[str, object]) -> float:
    """Wounded-scaling factor <N_w>/2 = (1 + <K>)/2 for a nucleus spec.

    Lazy-imports na49_mixing.glauber; callers must have the mixing2006
    package importable (the benchmark drivers insert its directory).
    """
    if not nucleus:
        return 0.5
    from na49_mixing.glauber import NucleusSpec, WoundedNucleonSampler

    spec = NucleusSpec(**{str(k): v for k, v in dict(nucleus).items()})
    sampler = WoundedNucleonSampler(spec, rng=np.random.default_rng(0))
    return 0.5 * (1.0 + float(sampler.mean_target_wounds_minbias()))


def _safe_cache_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in label)


REGGE_TOWER_SHAPES: tuple[str, ...] = ("linear", "regge_m2", "hagedorn")


def _compute_regge_tower_level_masses(min_mass: float, max_mass: float) -> Tuple[float, float, float]:
    lo = float(min_mass)
    hi = float(max_mass)
    if hi < lo:
        hi = lo
    mid = 0.5 * (lo + hi)
    return lo, mid, hi


def _compute_regge_tower_mass_ladder(
    min_mass: float,
    max_mass: float,
    *,
    level_count: int,
    shape: str,
    shape_param: float,
) -> np.ndarray:
    n = max(int(level_count), 1)
    lo = float(min_mass)
    hi = float(max_mass)
    if hi < lo:
        hi = lo

    if n == 1:
        return np.asarray([hi], dtype=float)

    shp = str(shape).strip().lower()
    if shp not in REGGE_TOWER_SHAPES:
        raise ValueError(f"Unsupported regge tower shape '{shape}'. Expected one of {REGGE_TOWER_SHAPES}.")

    u = np.linspace(0.0, 1.0, n, dtype=float)
    if shp == "linear":
        masses = lo + (hi - lo) * u
    elif shp == "regge_m2":
        lo2 = lo * lo
        hi2 = hi * hi
        masses = np.sqrt(lo2 + (hi2 - lo2) * u)
    else:
        # Equal-state bins for rho(m) ~ exp((m-m0)/T_H): masses bunch toward high m.
        th = max(float(shape_param), 1e-6)
        span = max((hi - lo) / th, 0.0)
        denom = np.expm1(span)
        if denom <= 0.0:
            masses = lo + (hi - lo) * u
        else:
            masses = lo + th * np.log1p(u * denom)

    masses[0] = lo
    masses[-1] = hi
    return np.asarray(masses, dtype=float)


def _tower_proxy_ids_for_mass(
    mass: float,
    *,
    min_mass: float,
    max_mass: float,
    anti: bool,
) -> int:
    m1, m2, m3 = _compute_regge_tower_level_masses(min_mass, max_mass)
    cut1 = 0.5 * (m1 + m2)
    cut2 = 0.5 * (m2 + m3)
    if mass <= cut1:
        return -9911201 if anti else 9911201
    if mass <= cut2:
        return -9921201 if anti else 9921201
    return -9931201 if anti else 9931201


def _regge_tower_width_for_mass(mass: float) -> float:
    # Preserve the legacy trend: 1.9->0.25, 2.1->0.30, 2.3->0.35 and extend smoothly.
    return float(np.clip(0.25 + 0.25 * (mass - 1.9), 0.20, 1.10))


def _offshell_broadened_tower_width(
    mass: float,
    *,
    min_mass: float,
    max_mass: float,
    offshell_kappa: float,
    offshell_beta: float,
) -> float:
    base = _regge_tower_width_for_mass(mass)
    kappa = max(float(offshell_kappa), 0.0)
    beta = max(float(offshell_beta), 0.0)
    span = max(float(max_mass) - float(min_mass), 1e-12)
    x = np.clip((float(mass) - float(min_mass)) / span, 0.0, 1.0)
    scale = 1.0 + kappa * (x**beta if beta > 0.0 else 1.0)
    return float(np.clip(base * scale, 0.05, 2.50))


_REGGE_MARGINAL_CACHE: Dict[Tuple[str, float, float, int, int, int], Tuple[List[float], List[float]]] = {}


def _regge_marginal_table(
    mixing2006_dir: Path,
    min_mass: float,
    max_mass: float,
    n_bins: int = 64,
    seed: int = 20260831,
    n_samples: int = 1_000_000,
) -> Tuple[List[float], List[float]]:
    """Tabulated Regge source-mass marginal on ``[min_mass, max_mass]``.

    Round 8 (pre-registered 2026-09-04 in
    notes/resonance_fit_stage_progress.md): the tower baryon components
    sample their masses from the production-side Regge source spectrum
    (the marginal of the two-source triangular boundary) instead of a
    fitted Breit-Wigner. The marginal is sampled once per process
    (seed-fixed) and cached; the card writer emits it as a mass_type-2
    tabulated block.
    """
    key = (str(mixing2006_dir), float(min_mass), float(max_mass), int(n_bins), int(seed), int(n_samples))
    cached = _REGGE_MARGINAL_CACHE.get(key)
    if cached is not None:
        return cached
    mixdir = str(mixing2006_dir)
    if mixdir not in sys.path:
        sys.path.insert(0, mixdir)
    from na49_mixing.event_builder_pp import PPEventBuilder, PPExcitationConfig
    from na49_mixing.exchange_model import ReggeExcitationParameters

    cfg = PPExcitationConfig(beam_momentum=158.0, production_mode="regge")
    builder = PPEventBuilder(cfg, rng=np.random.default_rng(seed))
    lo, hi = float(min_mass), float(max_mass)
    draws = np.empty(int(n_samples), dtype=np.float64)
    filled = 0
    while filled < int(n_samples):
        m1, _m2 = builder.mass_sampler.sample()
        if lo <= m1 <= hi:
            draws[filled] = m1
            filled += 1
    counts, edges = np.histogram(draws[:filled], bins=int(n_bins), range=(lo, hi))
    widths = np.diff(edges)
    dens = counts / (float(filled) * widths)
    table = ([float(e) for e in edges], [float(d) for d in dens])
    _REGGE_MARGINAL_CACHE[key] = table
    return table


def _rewrite_tower_gen_file(
    src: Path,
    dst: Path,
    pdg_id: int,
    mass: float,
    width: float,
    *,
    offshell_window_scale: float = 1.0,
    mass_table: Optional[Tuple[List[float], List[float]]] = None,
    pl_slope: Optional[float] = None,
) -> None:
    tokens = src.read_text(encoding="utf-8").split()
    if not tokens:
        raise RuntimeError(f"Empty generator file: {src}")

    idx = 0
    tokens[idx] = str(int(pdg_id))
    idx += 1  # particle id

    pt_type = int(tokens[idx]); idx += 1
    if pt_type == 3:
        nbins_pt = int(tokens[idx]); idx += 1
        idx += nbins_pt * 2 + 1
    else:
        idx += 1  # pt param

    pl_idx = idx  # pl_type token position
    idx += 1  # pl_type
    nbins_pl = int(tokens[idx]); idx += 1
    idx += (nbins_pl + 1) * 2
    if pl_slope is not None and int(tokens[pl_idx]) == 2:
        # Round 9 (pre-registered 2026-09-04): reweight the tabulated xF
        # production profile by exp(-k x) and renormalize. k > 0 shifts
        # weight from forward to central xF; k = 0 (default) is
        # physics-identical (the tokens are reformatted, which shifts the
        # card fingerprint and rebuilds the affected caches). The token
        # count is unchanged.
        k = float(pl_slope)
        base = pl_idx + 2
        coords = [float(tokens[base + 2 * j]) for j in range(nbins_pl + 1)]
        conts = [float(tokens[base + 2 * j + 1]) for j in range(nbins_pl + 1)]
        reweighted = [c * math.exp(-k * x) for x, c in zip(coords, conts)]
        total = sum(
            0.5 * (reweighted[j] + reweighted[j + 1]) * (coords[j + 1] - coords[j])
            for j in range(nbins_pl)
        )
        if total > 0.0:
            for j in range(nbins_pl + 1):
                tokens[base + 2 * j + 1] = f"{reweighted[j] / total:.8e}"

    mass_type = int(tokens[idx]); idx += 1
    if mass_table is not None:
        # Round-8 Regge-sourced tower: replace the whole (trailing) mass
        # block with a mass_type-2 tabulated histogram. The mass block is
        # the final block of the card, so truncation is safe.
        edges, dens = mass_table
        head = tokens[: idx - 1]
        block = ["2", str(len(dens))]
        for e, d in zip(edges, dens):
            block.append(f"{float(e):.6f}")
            block.append(f"{float(d):.8e}")
        block.append(f"{float(edges[-1]):.6f}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(" ".join(head + block) + "\n", encoding="utf-8")
        return
    wscale = max(float(offshell_window_scale), 0.25)
    dump_low = max(float(mass) + 0.60 * wscale, float(mass) + 2.0 * wscale * float(width))
    dump_high = max(float(mass) + 1.30 * wscale, float(mass) + 3.0 * wscale * float(width))
    if dump_high <= dump_low + 0.05:
        dump_high = dump_low + 0.05

    if mass_type == 3:
        # [schwel, rmas, brei, pbrei, dumplow, dumphigh]
        tokens[idx + 1] = f"{float(mass):.6f}"
        tokens[idx + 2] = f"{float(width):.6f}"
        tokens[idx + 4] = f"{dump_low:.6f}"
        tokens[idx + 5] = f"{dump_high:.6f}"
    elif mass_type == 1:
        # Short encoding [schwel, rmas, brei, dumplow, dumphigh]
        tokens[idx + 1] = f"{float(mass):.6f}"
        tokens[idx + 2] = f"{float(width):.6f}"
        tokens[idx + 3] = f"{dump_low:.6f}"
        tokens[idx + 4] = f"{dump_high:.6f}"
    else:
        raise RuntimeError(f"Unsupported mass_type={mass_type} in {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(" ".join(tokens) + "\n", encoding="utf-8")


def _materialize_regge_tower_gen_files(
    *,
    mixing2006_dir: Path,
    tower_gen_root: Path,
    min_mass: float,
    max_mass: float,
    level_count: int = 3,
    shape: str = "linear",
    shape_param: float = 0.25,
    offshell_kappa: float = 0.0,
    offshell_beta: float = 1.0,
    offshell_window_scale: float = 1.0,
    mass_law: str = "bw",
    pl_slope: float = 0.0,
) -> Dict[str, Path]:
    gen_dir = mixing2006_dir / "na49_mixing" / "generators" / "benchmark"
    masses_ladder = _compute_regge_tower_mass_ladder(
        min_mass,
        max_mass,
        level_count=int(level_count),
        shape=str(shape),
        shape_param=float(shape_param),
    )

    # Backward-compatible labels when using legacy 3-level linear tower.
    if int(level_count) == 3 and str(shape).strip().lower() == "linear":
        masses = {
            "nstar_eff1900_zero": float(masses_ladder[0]),
            "nstar_eff2100_zero": float(masses_ladder[1]),
            "nstar_eff2300_zero": float(masses_ladder[2]),
            "anti_nstar_eff1900_zero": float(masses_ladder[0]),
            "anti_nstar_eff2100_zero": float(masses_ladder[1]),
            "anti_nstar_eff2300_zero": float(masses_ladder[2]),
        }
        pdg_ids = {
            "nstar_eff1900_zero": 9911201,
            "nstar_eff2100_zero": 9921201,
            "nstar_eff2300_zero": 9931201,
            "anti_nstar_eff1900_zero": -9911201,
            "anti_nstar_eff2100_zero": -9921201,
            "anti_nstar_eff2300_zero": -9931201,
        }
    else:
        masses: Dict[str, float] = {}
        pdg_ids: Dict[str, int] = {}
        for idx, mass in enumerate(masses_ladder, start=1):
            lbl = f"nstar_effL{idx:02d}_zero"
            masses[lbl] = float(mass)
            pdg_ids[lbl] = _tower_proxy_ids_for_mass(
                float(mass),
                min_mass=float(min_mass),
                max_mass=float(max_mass),
                anti=False,
            )
            albl = f"anti_nstar_effL{idx:02d}_zero"
            masses[albl] = float(mass)
            pdg_ids[albl] = _tower_proxy_ids_for_mass(
                float(mass),
                min_mass=float(min_mass),
                max_mass=float(max_mass),
                anti=True,
            )

    if int(level_count) == 3 and str(shape).strip().lower() == "linear":
        scenario = (
            f"mmin{int(round(float(masses_ladder[0]) * 1000.0)):04d}_"
            f"mmax{int(round(float(masses_ladder[-1]) * 1000.0)):04d}"
        )
    else:
        shape_tag = str(shape).strip().lower().replace("_", "")
        scenario = (
            f"mmin{int(round(float(masses_ladder[0]) * 1000.0)):04d}_"
            f"mmax{int(round(float(masses_ladder[-1]) * 1000.0)):04d}_"
            f"n{int(level_count):02d}_{shape_tag}"
        )
        if str(shape).strip().lower() == "hagedorn":
            scenario += f"_t{int(round(float(shape_param) * 1000.0)):04d}"
    if str(mass_law).strip().lower() != "bw":
        scenario += f"_{str(mass_law).strip().lower()}"
    if abs(float(pl_slope)) > 1e-12:
        scenario += f"_pl{int(round(float(pl_slope) * 1000.0)):04d}"
    if float(offshell_kappa) > 0.0 or abs(float(offshell_window_scale) - 1.0) > 1e-12:
        scenario += (
            f"_offk{int(round(float(offshell_kappa) * 1000.0)):04d}"
            f"_offb{int(round(float(offshell_beta) * 1000.0)):04d}"
            f"_offw{int(round(float(offshell_window_scale) * 1000.0)):04d}"
        )
    scenario_dir = tower_gen_root / scenario
    out: Dict[str, Path] = {}
    for label, mass in masses.items():
        if label.startswith("anti_"):
            src = gen_dir / "anti_nstar_eff2300_zero.gen"
        else:
            src = gen_dir / "nstar_eff2300_zero.gen"
        dst = scenario_dir / f"{label}.gen"
        _rewrite_tower_gen_file(
            src=src,
            dst=dst,
            pdg_id=pdg_ids[label],
            mass=mass,
            width=_offshell_broadened_tower_width(
                mass,
                min_mass=float(min_mass),
                max_mass=float(max_mass),
                offshell_kappa=float(offshell_kappa),
                offshell_beta=float(offshell_beta),
            ),
            offshell_window_scale=float(offshell_window_scale),
            mass_table=(
                _regge_marginal_table(mixing2006_dir, float(min_mass), float(max_mass))
                if str(mass_law).strip().lower() == "regge"
                else None
            ),
            pl_slope=float(pl_slope),
        )
        out[label] = dst
    return out


# Bump when na49_mixing physics changes (decay library, mass/pT sampling)
# so cached component tracks are regenerated rather than silently reused.
# The version salts the cache filename digest; component seeds stay derived
# from the unsalted spec key. Source edits to the fingerprinted modules
# invalidate caches even without a bump (see _decay_lib_fingerprint).
RESONANCE_DECAY_LIB_VERSION = "2026-09-03.3"

# Post-build gate tolerance for terminal-unstable residuals: fraction of
# generated primaries allowed to remain as terminal-unstable tracks (floor
# 0; with no generated primaries, any residual fails). The P1/P2 off-shell
# broadened Delta line shape samples realized Delta masses just below the
# p+pi threshold (PDG-table nucleon 0.93956541 + pi- 0.1395699 =
# 1.07913531 GeV); the D1 mass-conditioned channel gate (mode_mass_windows
# returns None, Q < 0) correctly leaves such sub-threshold Deltas
# undecayed. Measured at production statistics (2026-09-02, seed 20260831,
# per 120k events): N*(1440) 8/120k, N*(1680) 1/120k, N*(1520) 0/120k in
# the p-species components; pi species unaffected. These residuals are a
# known physical edge of the line shape, not a contamination of the fit
# basis: the surviving resonance tracks carry unstable PDG ids that never
# match any observed NA49 species. The 2e-4 tolerance carries a 3x margin
# over the observed worst case (8/120k = 6.7e-5). The residual count is
# always recorded in meta_n_terminal_unstable.
TERMINAL_UNSTABLE_TOLERANCE = 2e-4


def _canonical_json_bytes(payload) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _audit_tables_json(tracks: Dict[str, object]) -> str:
    """Serialize per-channel audit tables from a chunked generation.

    The tables map parent PDG id to a channel (daughter PDG tuple) to a
    fraction or count.  Parent ids and daughter tuples are flattened into
    ``"parent|d1,d2,..."`` string keys so the payload round-trips through
    JSON; it is stored in the component cache archive as
    ``meta_channel_audit_json`` (card TASK-2026-08-31 rev 2, D1 openness
    guard) so openness fractions stay recorded when a cache is reused.
    Missing tables (older chunked paths) yield empty sections.
    """

    def _flatten(table: object) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for parent_key, channels in dict(table or {}).items():
            for channel_key, value in dict(channels).items():
                ch_str = ",".join(str(int(p)) for p in tuple(channel_key))
                out[f"{int(parent_key)}|{ch_str}"] = float(value)
        return out

    payload = {
        "channel_requested": _flatten(tracks.get("channel_requested")),
        "channel_realized": _flatten(tracks.get("channel_realized")),
        "openness_fraction": _flatten(tracks.get("openness_fraction")),
        "parent_openness_fraction": _flatten(
            tracks.get("parent_openness_fraction")
        ),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _decay_lib_fingerprint(
    *,
    gen_file,
    soft_decay_overrides=None,
    mixing2006_dir=None,
    beam_pdg: int = 2212,
    target_pdg: int = 2212,
) -> Optional[str]:
    """sha256 over everything that defines the installed generation behavior.

    Hash inputs, in order: canonical bytes of the imported
    ``na49_mixing.decays``, ``na49_mixing.parallel``,
    ``na49_mixing.generator``, ``na49_mixing.breit_wigner``,
    ``na49_mixing.histogram``, ``na49_mixing.glauber``,
    ``na49_mixing.event``, ``na49_mixing.isospin``,
    ``na49_mixing.reaction``, and ``na49_mixing.pdg_database`` source
    files, the spec's generator-card bytes, the sort-keys JSON of the
    soft-decay overrides, the resolved effective-tower mass/width
    update set, and the beam/target nucleon PDG ids.
    Module files are resolved through the imported package itself
    (``importlib``), never through REPO_ROOT string arithmetic, so the
    monorepo tree and the vendored tree hash identical inputs.

    Returns ``None`` when the decay library cannot be inspected; callers
    treat ``None`` as a cache mismatch, so the following rebuild surfaces
    the underlying problem loudly instead of silently reusing stale tracks.
    """
    try:
        if mixing2006_dir is not None:
            mixdir = str(mixing2006_dir)
            if mixdir not in sys.path:
                sys.path.insert(0, mixdir)
        import importlib

        chunks: List[Tuple[str, bytes]] = []
        for modname in (
            "na49_mixing.decays",
            "na49_mixing.parallel",
            "na49_mixing.generator",
            "na49_mixing.breit_wigner",
            "na49_mixing.histogram",
            "na49_mixing.glauber",
            "na49_mixing.event",
            "na49_mixing.isospin",
            "na49_mixing.reaction",
            "na49_mixing.pdg_database",
        ):
            mod = importlib.import_module(modname)
            origin = getattr(mod, "__file__", None)
            if not origin:
                raise RuntimeError(f"cannot resolve source file of {modname}")
            with open(origin, "rb") as fh:
                chunks.append((f"source:{modname}", fh.read()))
        chunks.append(("gen_card", Path(gen_file).read_bytes()))
        overrides = {
            str(k): float(v) for k, v in dict(soft_decay_overrides or {}).items()
        }
        chunks.append(("soft_overrides", _canonical_json_bytes(overrides)))
        from na49_mixing.parallel import _resolve_effective_tower_overrides

        tower_updates = {
            str(int(pid)): [float(pair[0]), float(pair[1])]
            for pid, pair in _resolve_effective_tower_overrides(overrides).items()
        }
        chunks.append(("tower_updates", _canonical_json_bytes(tower_updates)))
        chunks.append(("beam_pdg", str(int(beam_pdg)).encode("utf-8")))
        chunks.append(("target_pdg", str(int(target_pdg)).encode("utf-8")))

        h = hashlib.sha256()
        for tag, payload in chunks:
            tag_b = tag.encode("utf-8")
            h.update(len(tag_b).to_bytes(8, "big"))
            h.update(tag_b)
            h.update(len(payload).to_bytes(8, "big"))
            h.update(payload)
        return h.hexdigest()
    except Exception:
        return None


def _component_cache_path(cache_dir: Path, spec_key: str, label: str) -> Path:
    salted = f"{spec_key}|decay_lib={RESONANCE_DECAY_LIB_VERSION}"
    digest = hashlib.sha1(salted.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{_safe_cache_label(label)}__{digest}.npz"


def _component_seed(base_seed: int, spec_key: str) -> int:
    digest = hashlib.sha1(spec_key.encode("utf-8")).hexdigest()[:8]
    seed = (int(base_seed) + int(digest, 16)) % (2**31 - 1)
    return seed if seed > 0 else 1


def _normalize_fragmentation_mode(mode: str) -> str:
    val = str(mode).strip().lower()
    if val not in FRAGMENTATION_MODES:
        raise ValueError(
            f"Unsupported fragmentation mode '{mode}'. Expected one of {FRAGMENTATION_MODES}."
        )
    return val


def _parse_soft_decay_overrides(raw_items: List[str]) -> Dict[str, float]:
    overrides: Dict[str, float] = {}
    for raw in raw_items:
        txt = str(raw).strip()
        if not txt:
            continue
        if "=" not in txt:
            raise ValueError(f"Invalid --res-soft-override '{raw}'; expected KEY=VALUE.")
        key, val = txt.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --res-soft-override '{raw}'; empty key.")
        overrides[key] = float(val.strip())
    return overrides


def _is_tower_component_label(label: str) -> bool:
    txt = str(label).strip().lower()
    return "nstar_eff" in txt


def _infer_source_side_from_xf(xf: np.ndarray) -> np.ndarray:
    x = np.asarray(xf, dtype=float)
    # Backward-compatible fallback for caches produced before source-side tagging.
    return np.where(x >= 0.0, 1, -1).astype(np.int8)


def _feedover_keep_mask(
    xf: np.ndarray,
    source_side: np.ndarray,
    *,
    target_feedover_xf_max: float,
    projectile_feedover_xf_max: float,
) -> np.ndarray:
    x = np.asarray(xf, dtype=float)
    s = np.asarray(source_side, dtype=np.int8)
    keep = np.ones(x.shape[0], dtype=bool)
    tmax = float(target_feedover_xf_max)
    pmax = float(projectile_feedover_xf_max)
    if np.isfinite(tmax) and tmax >= 0.0:
        keep &= ~((s < 0) & (x > tmax))
    if np.isfinite(pmax) and pmax >= 0.0:
        keep &= ~((s > 0) & (x < -pmax))
    return keep


def _append_component_columns_from_tracks(
    *,
    labels: List[str],
    mat_cols: List[np.ndarray],
    component_label: str,
    pdg: np.ndarray,
    xf: np.ndarray,
    pt: np.ndarray,
    source_side: np.ndarray,
    particle_pdg: int,
    xedges: np.ndarray,
    pedges: np.ndarray,
    ix_data: np.ndarray,
    ip_data: np.ndarray,
    split_tower_by_source: bool,
    target_feedover_xf_max: float,
    projectile_feedover_xf_max: float,
    convention: str = "raw",
    sqrt_s_gev: Optional[float] = None,
) -> None:
    """Append design-matrix columns for one resonance component.

    Under ``convention == "raw"`` columns are raw bin counts (superseded
    pre-audit behaviour).  Under ``"invariant"`` each selected track carries
    its per-track Jacobian weight E/pT and the accumulated sum is divided by
    the point's bin area, giving the bin-averaged invariant-cross-section
    template f(xF, pT) up to a constant absorbed by fitted coefficients.
    """
    pdg_arr = np.asarray(pdg, dtype=np.int32)
    xf_arr = np.asarray(xf, dtype=float)
    pt_arr = np.asarray(pt, dtype=float)
    side_arr = np.asarray(source_side, dtype=np.int8)
    if side_arr.shape[0] != xf_arr.shape[0]:
        side_arr = _infer_source_side_from_xf(xf_arr)

    base = (pdg_arr == int(particle_pdg))
    if not np.any(base):
        return

    keep_base = _feedover_keep_mask(
        xf_arr[base],
        side_arr[base],
        target_feedover_xf_max=float(target_feedover_xf_max),
        projectile_feedover_xf_max=float(projectile_feedover_xf_max),
    )
    if not np.any(keep_base):
        return

    base_idx = np.where(base)[0]
    kept_idx = base_idx[keep_base]

    def _column(sel_idx: np.ndarray) -> np.ndarray:
        if convention == "invariant":
            if sqrt_s_gev is None:
                raise ValueError("sqrt_s_gev required for invariant columns")
            w = _invariant_track_weights(
                int(particle_pdg), xf_arr[sel_idx], pt_arr[sel_idx], sqrt_s_gev
            )
            return _counts_on_grid(
                xf_arr[sel_idx], pt_arr[sel_idx], xedges, pedges,
                ix_data, ip_data, weights=w,
            ) / _grid_point_areas(xedges, pedges, ix_data, ip_data)
        return _counts_on_grid(
            xf_arr[sel_idx], pt_arr[sel_idx], xedges, pedges, ix_data, ip_data
        )

    if bool(split_tower_by_source) and _is_tower_component_label(component_label):
        for side_val, suffix in ((1, "proj"), (-1, "targ")):
            sel_idx = kept_idx[side_arr[kept_idx] == side_val]
            if sel_idx.size == 0:
                continue
            labels.append(f"{component_label}_{suffix}")
            mat_cols.append(_column(sel_idx))
        return

    labels.append(str(component_label))
    mat_cols.append(_column(kept_idx))


def _all_resonance_specs(
    base_dir: Path,
    include_regge_tower: bool = True,
    regge_tower_gen_files: Optional[Dict[str, Path]] = None,
) -> Dict[str, ResonanceSpec]:
    by_key: Dict[str, ResonanceSpec] = {}
    # Include all currently benchmarked hadron targets.
    for pdg in (-211, 211, 2212, -2212, 321):
        for spec in _resonance_specs_for_particle(
            pdg,
            base_dir,
            include_regge_tower=include_regge_tower,
            regge_tower_gen_files=regge_tower_gen_files,
        ):
            by_key[_resonance_spec_key(spec, base_dir)] = spec
    return by_key


def _cache_config_matches(
    path: Path,
    *,
    spec_key: str,
    component_seed: int,
    n_events_per_res: int,
    max_decay_passes: int,
    fragmentation_mode: str,
    soft_decay_overrides_json: str,
    weak_feed_policy: str = "inclusive_weak",
    decay_lib_fingerprint: Optional[str] = None,
    beam_pdg: int = 2212,
    target_pdg: int = 2212,
) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            if "source_side" not in z:
                return False
            if "meta_soft_decay_overrides_json" in z:
                cache_ovr = str(z["meta_soft_decay_overrides_json"])
            else:
                cache_ovr = "{}"
            if "meta_fragmentation_mode" in z:
                cache_frag = str(z["meta_fragmentation_mode"])
            else:
                cache_frag = "projectile"
            if "meta_weak_feed_policy" in z:
                cache_policy = str(z["meta_weak_feed_policy"])
            else:
                cache_policy = ""
            if "meta_decay_lib_version" in z:
                cache_version = str(z["meta_decay_lib_version"])
            else:
                cache_version = ""
            if "meta_decay_lib_fingerprint" in z:
                cache_fingerprint = str(z["meta_decay_lib_fingerprint"])
            else:
                # Caches written before source-fingerprint tagging carry no
                # fingerprint; they mismatch and are rebuilt.
                cache_fingerprint = ""
            if "meta_beam_pdg" in z and "meta_target_pdg" in z:
                cache_beam = int(z["meta_beam_pdg"])
                cache_targ = int(z["meta_target_pdg"])
            else:
                # Caches written before species tagging are p+p by definition.
                cache_beam, cache_targ = 2212, 2212
            return (
                cache_version == RESONANCE_DECAY_LIB_VERSION
                and bool(decay_lib_fingerprint)
                and cache_fingerprint == str(decay_lib_fingerprint)
                and
                str(z["meta_spec_key"]) == spec_key
                and int(z["meta_component_seed"]) == int(component_seed)
                and int(z["meta_n_events_per_res"]) == int(n_events_per_res)
                and int(z["meta_max_decay_passes"]) == int(max_decay_passes)
                and cache_frag == str(fragmentation_mode)
                and cache_ovr == str(soft_decay_overrides_json)
                and cache_policy == str(weak_feed_policy)
                and cache_beam == int(beam_pdg)
                and cache_targ == int(target_pdg)
            )
    except Exception:
        return False


def _gateway_reserved_tasks() -> Optional[int]:
    raw = str(os.environ.get("NA49_CLUSTER_GW_TASKS", "")).strip()
    if not raw:
        return None
    try:
        val = int(raw)
    except ValueError:
        return None
    if val <= 0:
        return None
    return int(val)


def _resolve_resonance_worker_count(requested_workers: int) -> int:
    requested = max(int(requested_workers), 1)
    gw_tasks = _gateway_reserved_tasks()
    if gw_tasks is None:
        return requested
    if requested <= 1:
        # Gateway jobs default to reserved task slots when caller did not request
        # explicit local parallelism.
        return int(gw_tasks)
    return int(min(requested, gw_tasks))


def _auto_resonance_batch_size(n_tasks: int, n_workers: int) -> int:
    tasks = max(int(n_tasks), 0)
    workers = max(int(n_workers), 1)
    if tasks <= 1:
        return 1
    if workers <= 1:
        return tasks
    # Keep batches small enough for load balancing while reducing dispatch overhead.
    target_batches = max(workers * 3, 1)
    return max(int(math.ceil(tasks / target_batches)), 1)


def _split_resonance_batches(tasks: List[Dict[str, object]], *, n_workers: int) -> List[List[Dict[str, object]]]:
    if not tasks:
        return []
    batch_size = _auto_resonance_batch_size(len(tasks), n_workers)
    return [tasks[i : i + batch_size] for i in range(0, len(tasks), batch_size)]


def _build_resonance_component_cache_batch(
    *,
    mixing2006_dir: str,
    tasks: List[Dict[str, object]],
    n_events_per_res: int,
    max_decay_passes: int,
    fragmentation_mode: str,
    chunk_workers: int = 1,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for task in tasks:
        out.append(
            _build_resonance_component_cache_one(
                mixing2006_dir=str(mixing2006_dir),
                task=task,
                n_events_per_res=int(n_events_per_res),
                max_decay_passes=int(max_decay_passes),
                fragmentation_mode=str(fragmentation_mode),
                chunk_workers=int(chunk_workers),
            )
        )
    return out


def _build_resonance_component_cache(
    *,
    mixing2006_dir: Path,
    cache_dir: Path,
    specs_by_key: Dict[str, ResonanceSpec],
    n_events_per_res: int,
    seed: int,
    max_decay_passes: int,
    fragmentation_mode: str,
    force_rebuild: bool,
    res_workers: int,
    soft_decay_overrides: Optional[Dict[str, float]] = None,
    weak_feed_policy: str = "inclusive_weak",
    nucleus: Optional[Dict[str, object]] = None,
    target_wound_range: Optional[Tuple[int, int]] = None,
    cache_namespace: Optional[str] = None,
    beam_pdg: int = 2212,
    target_pdg: int = 2212,
) -> Dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fragmentation_mode = _normalize_fragmentation_mode(fragmentation_mode)
    # Neutron-system runs get their own namespace so they can neither read
    # nor overwrite proton-system cache files.
    species_tag = species_namespace_tag(beam_pdg, target_pdg)
    if species_tag:
        cache_namespace = (
            f"{species_tag}|{cache_namespace}" if cache_namespace else species_tag
        )
    requested_workers = max(int(res_workers), 1)
    effective_workers = _resolve_resonance_worker_count(requested_workers)
    soft_decay_overrides_json = json.dumps(soft_decay_overrides or {}, sort_keys=True, separators=(",", ":"))
    weak_feed_policy = str(weak_feed_policy)

    key_to_path: Dict[str, Path] = {}
    # Namespace-qualified cache key -> original specs_by_key key.
    key_origin: Dict[str, str] = {}
    manifest_entries: List[Dict[str, object]] = []
    to_build: List[Dict[str, object]] = []
    schedule_mode = "serial"
    schedule_batches = 0
    schedule_batch_size = 0

    for spec_key_raw in sorted(specs_by_key.keys()):
        spec = specs_by_key[spec_key_raw]
        spec_key = (
            f"{cache_namespace}|{spec_key_raw}" if cache_namespace else spec_key_raw
        )
        key_origin[spec_key] = spec_key_raw
        cache_path = _component_cache_path(cache_dir, spec_key, spec.label)
        component_seed = _component_seed(seed, spec_key)
        decay_lib_fingerprint = _decay_lib_fingerprint(
            gen_file=Path(spec.gen_file),
            soft_decay_overrides=dict(soft_decay_overrides or {}),
            mixing2006_dir=mixing2006_dir,
            beam_pdg=int(beam_pdg),
            target_pdg=int(target_pdg),
        )

        reuse_ok = (not force_rebuild) and _cache_config_matches(
            cache_path,
            spec_key=spec_key,
            component_seed=component_seed,
            n_events_per_res=n_events_per_res,
            max_decay_passes=max_decay_passes,
            fragmentation_mode=fragmentation_mode,
            soft_decay_overrides_json=soft_decay_overrides_json,
            weak_feed_policy=weak_feed_policy,
            decay_lib_fingerprint=decay_lib_fingerprint,
            beam_pdg=int(beam_pdg),
            target_pdg=int(target_pdg),
        )

        if reuse_ok:
            print(f"res-cache: reused {spec.label} -> {cache_path}", flush=True)
            key_to_path[spec_key] = cache_path
            continue

        to_build.append(
            {
                "spec_key": spec_key,
                "label": spec.label,
                "gen_file": str(spec.gen_file),
                "decay_files": [str(p) for p in spec.decay_files],
                "use_kaon_decay_factory": bool(spec.use_kaon_decay_factory),
                "use_soft_decay_factory": bool(spec.use_soft_decay_factory),
                "cache_file": str(cache_path),
                "component_seed": int(component_seed),
                "fragmentation_mode": str(fragmentation_mode),
                "soft_decay_overrides": dict(soft_decay_overrides or {}),
                "weak_feed_policy": str(weak_feed_policy),
                "nucleus": (dict(nucleus) if nucleus else None),
                "target_wound_range": (
                    [int(target_wound_range[0]), int(target_wound_range[1])]
                    if target_wound_range
                    else None
                ),
                "beam_pdg": int(beam_pdg),
                "target_pdg": int(target_pdg),
                "decay_lib_fingerprint": (
                    str(decay_lib_fingerprint) if decay_lib_fingerprint else ""
                ),
            }
        )

    if to_build:
        if effective_workers > 1 and len(to_build) > 1:
            worker_count = min(int(effective_workers), len(to_build))
            batches = _split_resonance_batches(to_build, n_workers=worker_count)
            schedule_mode = "batched_pool"
            schedule_batches = len(batches)
            schedule_batch_size = max(len(batch) for batch in batches) if batches else 0
            print(
                "res-cache: "
                + f"workers(requested={requested_workers}, effective={effective_workers}) "
                + f"batch_count={schedule_batches} batch_size<={schedule_batch_size}",
                flush=True,
            )
            with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as ex:
                futures = [
                    ex.submit(
                        _build_resonance_component_cache_batch,
                        mixing2006_dir=str(mixing2006_dir),
                        tasks=batch,
                        n_events_per_res=int(n_events_per_res),
                        max_decay_passes=int(max_decay_passes),
                        fragmentation_mode=str(fragmentation_mode),
                        chunk_workers=1,
                    )
                    for batch in batches
                ]
                for fut in concurrent.futures.as_completed(futures):
                    for result in fut.result():
                        if not bool(result.get("ok", False)):
                            print(
                                f"res-cache: skipped {result.get('label', '<unknown>')} ({result.get('reason', 'unknown')})",
                                flush=True,
                            )
                            continue
                        cache_path = Path(str(result["cache_file"]))
                        print(f"res-cache: built {result['label']} -> {cache_path}", flush=True)
                        key_to_path[str(result["spec_key"])] = cache_path
        else:
            schedule_mode = "single_component_chunked" if (effective_workers > 1 and len(to_build) == 1) else "serial"
            for task in to_build:
                result = _build_resonance_component_cache_one(
                    mixing2006_dir=str(mixing2006_dir),
                    task=task,
                    n_events_per_res=int(n_events_per_res),
                    max_decay_passes=int(max_decay_passes),
                    fragmentation_mode=str(fragmentation_mode),
                    chunk_workers=effective_workers,
                )
                if not bool(result.get("ok", False)):
                    print(
                        f"res-cache: skipped {result.get('label', '<unknown>')} ({result.get('reason', 'unknown')})",
                        flush=True,
                    )
                    continue
                cache_path = Path(str(result["cache_file"]))
                print(f"res-cache: built {result['label']} -> {cache_path}", flush=True)
                key_to_path[str(result["spec_key"])] = cache_path

    for full_key in sorted(key_to_path.keys()):
        spec = specs_by_key[key_origin[full_key]]
        spec_key = full_key
        cache_path = key_to_path[full_key]
        with np.load(cache_path, mmap_mode="r", allow_pickle=False) as z:
            n_tracks = int(z["pdg"].shape[0])
            n_decayed = int(z.get("meta_n_decayed_parents", np.asarray(0)))
            n_weak = int(z.get("meta_n_weak_feed_tracks", np.asarray(-1)))
            weak_feed_policy = str(z.get("meta_weak_feed_policy", np.asarray("inclusive_weak")))
            n_weak_dropped = int(z.get("meta_n_weak_dropped", np.asarray(0)))
            n_terminal_unstable = int(z.get("meta_n_terminal_unstable", np.asarray(0)))
            n_primary_generated = int(z.get("meta_n_primary_generated", np.asarray(0)))
            n_primary_decayed = int(z.get("meta_n_primary_decayed", np.asarray(0)))
            cache_fingerprint = str(
                z.get("meta_decay_lib_fingerprint", np.asarray(""))
            )
            undecayed_primary_fraction = float(
                z.get("meta_undecayed_primary_fraction", np.asarray(float("nan")))
            )
            try:
                parent_meta = json.loads(str(z["meta_primary_parent_json"]))
            except Exception:
                parent_meta = {}
            # Read pA provenance inside the open archive; the manifest
            # dict below is built after the context closes.
            pa_nucleus_meta: Optional[Dict[str, object]] = None
            pa_wound_meta: Optional[List[int]] = None
            if "meta_pa_nucleus_json" in z:
                try:
                    pa_nucleus_meta = json.loads(str(z["meta_pa_nucleus_json"]))
                except Exception:
                    pa_nucleus_meta = None
            if "meta_pa_target_wound_range_json" in z:
                try:
                    pa_wound_meta = json.loads(
                        str(z["meta_pa_target_wound_range_json"])
                    )
                except Exception:
                    pa_wound_meta = None
        manifest_entries.append(
            {
                "spec_key": spec_key,
                "label": spec.label,
                "gen_file": str(spec.gen_file),
                "cache_file": str(cache_path),
                "component_seed": _component_seed(seed, spec_key),
                "decay_lib_version": RESONANCE_DECAY_LIB_VERSION,
                "decay_lib_fingerprint": cache_fingerprint,
                "tracks": n_tracks,
                "n_decayed_parents": n_decayed,
                "undecayed_primary_fraction": undecayed_primary_fraction,
                "n_primary_generated": n_primary_generated,
                "n_primary_decayed": n_primary_decayed,
                "n_weak_feed_tracks": n_weak,
                "weak_feed_policy": weak_feed_policy,
                "n_weak_dropped": n_weak_dropped,
                "pa_nucleus": (
                    pa_nucleus_meta
                    if pa_nucleus_meta is not None
                    else (dict(nucleus) if nucleus else None)
                ),
                "pa_target_wound_range": (
                    pa_wound_meta
                    if pa_wound_meta is not None
                    else (
                        [int(target_wound_range[0]), int(target_wound_range[1])]
                        if target_wound_range
                        else None
                    )
                ),
                "n_terminal_unstable": n_terminal_unstable,
                "numpy_version": NUMPY_VERSION,
                "numpy_bit_generator": NUMPY_BIT_GENERATOR,
                "parent_pdg": parent_meta.get("parent_pdg", None),
            }
        )

    manifest = {
        "mixing2006_dir": str(mixing2006_dir),
        "cache_dir": str(cache_dir),
        "n_events_per_res": int(n_events_per_res),
        "seed": int(seed),
        "max_decay_passes": int(max_decay_passes),
        "fragmentation_mode": str(fragmentation_mode),
        "soft_decay_overrides": dict(soft_decay_overrides or {}),
        "res_workers_requested": int(requested_workers),
        "res_workers_effective": int(effective_workers),
        "schedule_mode": str(schedule_mode),
        "schedule_batch_count": int(schedule_batches),
        "schedule_batch_size": int(schedule_batch_size),
        "gateway_reserved_tasks": _gateway_reserved_tasks(),
        "components": manifest_entries,
    }
    (cache_dir / "cache_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return key_to_path


def _terminal_unstable_gate(n_terminal_unstable: int, n_gen: int) -> Optional[str]:
    """Return a failure reason when terminal-unstable residuals exceed the
    pre-registered tolerance, else None. Tolerance: TERMINAL_UNSTABLE_TOLERANCE
    of n_gen (floor 0). Zero generated primaries: any residual fails."""
    allowed = int(max(0.0, TERMINAL_UNSTABLE_TOLERANCE * float(n_gen)))
    if n_terminal_unstable <= allowed:
        return None
    return (
        f"terminal_unstable_tracks:{n_terminal_unstable} > {allowed}"
        f" (tol {TERMINAL_UNSTABLE_TOLERANCE:g} x n_gen {n_gen})"
    )


def _build_resonance_component_cache_one(
    *,
    mixing2006_dir: str,
    task: Dict[str, object],
    n_events_per_res: int,
    max_decay_passes: int,
    fragmentation_mode: str,
    chunk_workers: int = 1,
) -> Dict[str, object]:
    spec = ResonanceSpec(
        label=str(task["label"]),
        gen_file=Path(str(task["gen_file"])),
        decay_files=tuple(Path(str(p)) for p in list(task.get("decay_files", []))),
        use_kaon_decay_factory=bool(task.get("use_kaon_decay_factory", False)),
        use_soft_decay_factory=bool(task.get("use_soft_decay_factory", False)),
    )
    cache_path = Path(str(task["cache_file"]))
    component_seed = int(task["component_seed"])

    if not spec.gen_file.exists():
        return {
            "ok": False,
            "spec_key": str(task["spec_key"]),
            "label": spec.label,
            "reason": f"missing_gen:{spec.gen_file}",
        }

    mixdir = Path(mixing2006_dir)
    if str(mixdir) not in sys.path:
        sys.path.insert(0, str(mixdir))
    try:
        from na49_mixing.parallel import GenerationConfig, generate_tracks_chunked
        from na49_mixing.glauber import NucleusSpec
    except Exception as exc:
        return {
            "ok": False,
            "spec_key": str(task["spec_key"]),
            "label": spec.label,
            "reason": f"import_parallel_failed:{exc}",
        }

    cfg = GenerationConfig(
        gen_file=str(spec.gen_file),
        n_events=int(n_events_per_res),
        seed=int(component_seed),
        beam_momentum=158.0,
        max_decay_passes=int(max_decay_passes),
        fragmentation_mode=str(_normalize_fragmentation_mode(fragmentation_mode)),
        decay_files=tuple(str(p) for p in spec.decay_files),
        use_kaon_decay_factory=bool(spec.use_kaon_decay_factory),
        use_soft_decay_factory=bool(spec.use_soft_decay_factory),
        soft_decay_overrides=tuple(
            sorted((str(k), float(v)) for k, v in dict(task.get("soft_decay_overrides", {})).items())
        ),
        # Keep compatibility with benchmark antiproton component handling.
        inject_antibaryon_resonances=True,
        weak_feed_policy=str(task.get("weak_feed_policy", "inclusive_weak")),
        beam_pdg=int(task.get("beam_pdg", 2212)),
        target_pdg=int(task.get("target_pdg", 2212)),
        nucleus=(
            NucleusSpec(**{str(k): v for k, v in dict(task["nucleus"]).items()})
            if task.get("nucleus")
            else None
        ),
        target_wound_range=(
            (int(task["target_wound_range"][0]), int(task["target_wound_range"][1]))
            if task.get("target_wound_range")
            else None
        ),
    )
    max_undecayed_fraction = float(task.get("max_undecayed_fraction", 0.01))
    try:
        tracks = generate_tracks_chunked(
            cfg,
            n_workers=max(int(chunk_workers), 1),
            n_chunks=max(int(chunk_workers), 1),
        )
    except Exception as exc:
        return {
            "ok": False,
            "spec_key": str(task["spec_key"]),
            "label": spec.label,
            "reason": f"chunk_generation_failed:{exc}",
        }

    # Generation diagnostics gate: a component with a non-negligible
    # undecayed-primary fraction must fail loudly rather than silently
    # contaminate the fit basis. Depth-zero counters compare generated
    # primaries with primaries that entered the decay cascade directly, so
    # weak feed-down and deeper cascade decays do not dilute the gate.
    n_gen = int(tracks.get("n_primary_generated", 0))
    n_dec = int(tracks.get("n_primary_decayed", 0))
    undecayed_primary_fraction = (1.0 - n_dec / n_gen) if n_gen > 0 else None
    if (
        undecayed_primary_fraction is not None
        and undecayed_primary_fraction > max_undecayed_fraction
    ):
        return {
            "ok": False,
            "spec_key": str(task["spec_key"]),
            "label": spec.label,
            "reason": (
                f"undecayed_primary_fraction:{undecayed_primary_fraction:.4f}"
                f">:{max_undecayed_fraction}"
            ),
        }

    n_terminal_unstable = int(tracks.get("n_terminal_unstable", 0))
    terminal_unstable_reason = _terminal_unstable_gate(n_terminal_unstable, n_gen)
    if terminal_unstable_reason is not None:
        return {
            "ok": False,
            "spec_key": str(task["spec_key"]),
            "label": spec.label,
            "reason": terminal_unstable_reason,
        }

    # Atomic write: concurrent gateway jobs can share this cache directory,
    # and a reader must never observe a partially written npz (a direct
    # write let a concurrent reader die on EOFError mid-file, killing the
    # Stage-C W runs of 2026-09-02). Write to a unique temporary name in
    # the same directory and rename atomically; readers either see the
    # old file or the complete new one.
    tmp_cache_path = cache_path.with_name(
        cache_path.stem + f".tmp{os.getpid()}.npz"
    )
    np.savez_compressed(
        tmp_cache_path,
        pdg=np.asarray(tracks["pdg"], dtype=np.int32),
        xf=np.asarray(tracks["xf"], dtype=np.float32),
        pt=np.asarray(tracks["pt"], dtype=np.float32),
        source_side=np.asarray(tracks.get("source_side", _infer_source_side_from_xf(np.asarray(tracks["xf"], dtype=float))), dtype=np.int8),
        weak_feed=np.asarray(tracks.get("weak_feed", np.empty(0, dtype=np.int8)), dtype=np.int8),
        gen_depth=np.asarray(tracks.get("gen_depth", np.empty(0, dtype=np.int16)), dtype=np.int16),
        meta_label=np.asarray(spec.label),
        meta_spec_key=np.asarray(str(task["spec_key"])),
        meta_decay_lib_version=np.asarray(RESONANCE_DECAY_LIB_VERSION),
        meta_decay_lib_fingerprint=np.asarray(str(task.get("decay_lib_fingerprint", ""))),
        meta_n_decayed_parents=np.asarray(int(tracks.get("n_decayed_parents", 0)), dtype=np.int64),
        meta_n_decay_kinematic_failures=np.asarray(int(tracks.get("n_decay_kinematic_failures", 0)), dtype=np.int64),
        meta_n_weak_feed_tracks=np.asarray(int(tracks.get("n_weak_feed_tracks", 0)), dtype=np.int64),
        meta_n_primary_generated=np.asarray(int(tracks.get("n_primary_generated", 0)), dtype=np.int64),
        meta_n_primary_decayed=np.asarray(int(tracks.get("n_primary_decayed", 0)), dtype=np.int64),
        meta_n_weak_dropped=np.asarray(int(tracks.get("n_weak_dropped", 0)), dtype=np.int64),
        meta_n_terminal_unstable=np.asarray(int(tracks.get("n_terminal_unstable", 0)), dtype=np.int64),
        meta_weak_feed_policy=np.asarray(str(task.get("weak_feed_policy", "inclusive_weak"))),
        meta_pa_nucleus_json=np.asarray(
            json.dumps(task.get("nucleus"), sort_keys=True)
        ),
        meta_pa_target_wound_range_json=np.asarray(
            json.dumps(task.get("target_wound_range"), sort_keys=True)
        ),
        meta_undecayed_primary_fraction=np.asarray(
            float(undecayed_primary_fraction)
            if undecayed_primary_fraction is not None
            else float("nan")
        ),
        meta_primary_parent_json=np.asarray(json.dumps(tracks.get("primary_parent"), sort_keys=True)),
        meta_mass_law_json=np.asarray(json.dumps(tracks.get("mass_law"), sort_keys=True)),
        meta_component_seed=np.asarray(component_seed, dtype=np.int64),
        meta_n_events_per_res=np.asarray(int(n_events_per_res), dtype=np.int64),
        meta_max_decay_passes=np.asarray(int(max_decay_passes), dtype=np.int64),
        meta_beam_pdg=np.asarray(int(task.get("beam_pdg", 2212)), dtype=np.int64),
        meta_target_pdg=np.asarray(int(task.get("target_pdg", 2212)), dtype=np.int64),
        meta_fragmentation_mode=np.asarray(str(_normalize_fragmentation_mode(fragmentation_mode))),
        meta_soft_decay_overrides_json=np.asarray(
            json.dumps(dict(task.get("soft_decay_overrides", {})), sort_keys=True, separators=(",", ":"))
        ),
        # Per-channel audit tables from the chunked generation path (card
        # TASK-2026-08-31 rev 2, D1 openness guard; review REV-CUM-01).
        # Stored in the archive so openness fractions survive cache reuse.
        meta_channel_audit_json=np.asarray(_audit_tables_json(tracks)),
    )
    os.replace(tmp_cache_path, cache_path)
    # Per-channel audit tables from the chunked generation path (card
    # TASK-2026-08-31 rev 2, D1 openness guard; review REV-CUM-01).  The
    # tables are computed by generate_tracks_chunked and returned here so
    # cache-building consumers (the staged resonance-decay fit driver) can
    # record openness fractions for the build; older chunked paths without
    # the tables yield empty dicts.
    audit_json = _audit_tables_json(tracks)
    return {
        "ok": True,
        "spec_key": str(task["spec_key"]),
        "label": spec.label,
        "cache_file": str(cache_path),
        "channel_audit_json": audit_json,
        "channel_requested": dict(tracks.get("channel_requested") or {}),
        "channel_realized": dict(tracks.get("channel_realized") or {}),
        "openness_fraction": dict(tracks.get("openness_fraction") or {}),
        "parent_openness_fraction": dict(tracks.get("parent_openness_fraction") or {}),
    }


def _build_decay_chain(
    decay_spec: ResonanceSpec,
    rng: np.random.Generator,
    DecayMode,
    ParticleDecays,
    build_kaon_decays,
    build_soft_resonance_decays,
    read_decays,
    soft_decay_overrides: Optional[Dict[str, float]] = None,
) -> Optional[dict]:
    """Single canonical decay-chain construction.

    Delegates to ``na49_mixing.parallel._build_decay_chain`` so cached and
    uncached generation share ONE antibaryon/decay implementation.  The
    class arguments are accepted for backward compatibility and ignored.
    """
    try:
        from na49_mixing.parallel import (
            GenerationConfig as _GenCfg,
            _build_decay_chain as _pkg_build,
        )
    except Exception:
        return None
    cfg = _GenCfg(
        gen_file=str(decay_spec.gen_file),
        n_events=0,
        seed=0,
        decay_files=tuple(str(p) for p in decay_spec.decay_files),
        use_kaon_decay_factory=bool(decay_spec.use_kaon_decay_factory),
        use_soft_decay_factory=bool(decay_spec.use_soft_decay_factory),
        soft_decay_overrides=tuple(
            sorted((str(k), float(v)) for k, v in dict(soft_decay_overrides or {}).items())
        ),
        inject_antibaryon_resonances=True,
    )
    return _pkg_build(cfg, rng)


def _run_resonance_components(
    mixing2006_dir: Path,
    n_events_per_res: int,
    seed: int,
    particle_pdg: int,
    max_decay_passes: int,
    fragmentation_mode: str,
    res_workers: int,
    xedges: np.ndarray,
    pedges: np.ndarray,
    ix_data: np.ndarray,
    ip_data: np.ndarray,
    cache_dir: Optional[Path] = None,
    cache_scope: str = "species",
    cache_force_rebuild: bool = False,
    include_regge_tower: bool = True,
    regge_tower_gen_files: Optional[Dict[str, Path]] = None,
    soft_decay_overrides: Optional[Dict[str, float]] = None,
    split_tower_by_source: bool = False,
    target_feedover_xf_max: float = -1.0,
    projectile_feedover_xf_max: float = -1.0,
    convention: str = "raw",
    sqrt_s_gev: Optional[float] = None,
    weak_feed_policy: str = "inclusive_weak",
) -> Tuple[List[str], np.ndarray]:
    fragmentation_mode = _normalize_fragmentation_mode(fragmentation_mode)
    base_dir = mixing2006_dir
    specs = _resonance_specs_for_particle(
        particle_pdg,
        base_dir,
        include_regge_tower=include_regge_tower,
        regge_tower_gen_files=regge_tower_gen_files,
    )

    labels: List[str] = []
    mat_cols: List[np.ndarray] = []

    if cache_dir is not None:
        all_specs = _all_resonance_specs(
            base_dir,
            include_regge_tower=include_regge_tower,
            regge_tower_gen_files=regge_tower_gen_files,
        )
        if cache_scope == "all":
            build_specs = all_specs
        else:
            build_specs = {_resonance_spec_key(s, base_dir): s for s in specs}

        cache_index = _build_resonance_component_cache(
            mixing2006_dir=mixing2006_dir,
            cache_dir=cache_dir,
            specs_by_key=build_specs,
            n_events_per_res=n_events_per_res,
            seed=seed,
            max_decay_passes=max_decay_passes,
            fragmentation_mode=fragmentation_mode,
            force_rebuild=cache_force_rebuild,
            res_workers=res_workers,
            soft_decay_overrides=soft_decay_overrides,
            weak_feed_policy=str(weak_feed_policy),
        )

        for spec in specs:
            spec_key = _resonance_spec_key(spec, base_dir)
            cache_path = cache_index.get(spec_key)
            if cache_path is None or (not cache_path.exists()):
                continue
            with np.load(cache_path, mmap_mode="r", allow_pickle=False) as z:
                pdg = np.asarray(z["pdg"], dtype=np.int32)
                xf = np.asarray(z["xf"], dtype=float)
                pt = np.asarray(z["pt"], dtype=float)
                if "source_side" in z:
                    source_side = np.asarray(z["source_side"], dtype=np.int8)
                else:
                    source_side = _infer_source_side_from_xf(xf)
            _append_component_columns_from_tracks(
                labels=labels,
                mat_cols=mat_cols,
                component_label=spec.label,
                pdg=pdg,
                xf=xf,
                pt=pt,
                source_side=source_side,
                particle_pdg=particle_pdg,
                xedges=xedges,
                pedges=pedges,
                ix_data=ix_data,
                ip_data=ip_data,
                split_tower_by_source=bool(split_tower_by_source and fragmentation_mode == "both"),
                target_feedover_xf_max=float(target_feedover_xf_max),
                projectile_feedover_xf_max=float(projectile_feedover_xf_max),
                convention=convention,
                sqrt_s_gev=sqrt_s_gev,
            )
    else:
        Event, Generator, Reaction, DecayMode, ParticleDecays, build_kaon_decays, build_soft_resonance_decays, read_decays = _import_mixing2006(mixing2006_dir)

        reaction = Reaction(158.0)
        rng = np.random.default_rng(seed)

        for spec in specs:
            if not spec.gen_file.exists():
                continue

            gen = Generator(reaction=reaction, rng=rng)
            gen.set_all(str(spec.gen_file))

            ev = Event(rng=rng)
            decay_chain = _build_decay_chain(
                spec,
                rng,
                DecayMode,
                ParticleDecays,
                build_kaon_decays,
                build_soft_resonance_decays,
                read_decays,
                soft_decay_overrides=soft_decay_overrides,
            )
            if decay_chain is None or not decay_chain:
                continue
            ev._decay_chain = decay_chain

            pdg_vals: List[int] = []
            xf_vals: List[float] = []
            pt_vals: List[float] = []
            source_side_vals: List[int] = []
            weak_feed_vals: List[int] = []

            def _decay_and_collect(parent, side_tag: int) -> None:
                if parent is None:
                    return
                ev.add_particle(parent)
                if max_decay_passes <= 1:
                    ev.decay()
                else:
                    ev.decay_cascade(max_passes=max_decay_passes)
                for pp in ev:
                    xv = float(pp.get_xf())
                    pv = float(pp.get_pt())
                    if not np.isfinite(xv) or not np.isfinite(pv):
                        continue
                    pdg_vals.append(int(pp.get_id()))
                    xf_vals.append(xv)
                    pt_vals.append(pv)
                    source_side_vals.append(int(side_tag))
                    weak_feed_vals.append(int(bool(pp.get_weak_feed())))
                ev.delete()

            for _ in range(n_events_per_res):
                if fragmentation_mode in ("projectile", "both"):
                    part = gen.generate()
                    _decay_and_collect(part, +1)

                if fragmentation_mode in ("target", "both"):
                    part_t = gen.generate()
                    if part_t is not None:
                        pz_cms = part_t.get_pz_cms(reaction)
                        e_cms = part_t.get_e_cms(reaction)
                        pz_lab_target = reaction.gamma() * (-pz_cms + reaction.betha() * e_cms)
                        part_t.set_p(part_t.get_px(), part_t.get_py(), pz_lab_target)
                    _decay_and_collect(part_t, -1)

            if not xf_vals:
                continue

            pdg_arr = np.asarray(pdg_vals, dtype=np.int32)
            xf_arr = np.asarray(xf_vals, dtype=float)
            pt_arr = np.asarray(pt_vals, dtype=float)
            side_arr = np.asarray(source_side_vals, dtype=np.int8)

            if str(weak_feed_policy) == "primary_only":
                keep_weak = np.asarray(weak_feed_vals, dtype=np.int8) == 0
                n_weak_dropped = int(keep_weak.size - np.count_nonzero(keep_weak))
                print(
                    f"weak-feed[{spec.label}]: primary_only kept "
                    f"{int(np.count_nonzero(keep_weak))}/{int(keep_weak.size)} tracks "
                    f"(dropped {n_weak_dropped})",
                    flush=True,
                )
                pdg_arr = pdg_arr[keep_weak]
                xf_arr = xf_arr[keep_weak]
                pt_arr = pt_arr[keep_weak]
                side_arr = side_arr[keep_weak]
                if pdg_arr.size == 0:
                    continue

            _append_component_columns_from_tracks(
                labels=labels,
                mat_cols=mat_cols,
                component_label=spec.label,
                pdg=pdg_arr,
                xf=xf_arr,
                pt=pt_arr,
                source_side=side_arr,
                particle_pdg=particle_pdg,
                xedges=xedges,
                pedges=pedges,
                ix_data=ix_data,
                ip_data=ip_data,
                split_tower_by_source=bool(split_tower_by_source and fragmentation_mode == "both"),
                target_feedover_xf_max=float(target_feedover_xf_max),
                projectile_feedover_xf_max=float(projectile_feedover_xf_max),
                convention=convention,
                sqrt_s_gev=sqrt_s_gev,
            )

    if not mat_cols:
        raise RuntimeError(f"No resonance components produced pdg={particle_pdg} samples.")
    return labels, np.column_stack(mat_cols)


def _load_external_grid_counts(
    csv_path: Path,
    xedges: np.ndarray,
    pedges: np.ndarray,
    ix_data: np.ndarray,
    ip_data: np.ndarray,
    convention: str = "raw",
    sqrt_s_gev: Optional[float] = None,
    mass_gev: Optional[float] = None,
) -> np.ndarray:
    """Load a per-particle external-generator CSV onto the analysis grid.

    CSV rows are (xf, pt, value=1) for every final particle of one species.
    Under ``convention == "invariant"`` each row carries the Jacobian weight
    E/pT and the accumulated sum is divided by the point's bin area, giving
    bin-averaged invariant-cross-section templates on the same footing as
    the resonance components.
    """
    xs: List[float] = []
    ps: List[float] = []
    vs: List[float] = []
    with csv_path.open(encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            xs.append(float(row["xf"]))
            ps.append(float(row["pt"]))
            vs.append(float(row["value"]))
    xs = np.asarray(xs)
    ps = np.asarray(ps)
    vs = np.asarray(vs)
    if convention == "invariant":
        if sqrt_s_gev is None:
            raise ValueError("sqrt_s_gev required for invariant external grids")
        vs = vs * _invariant_track_weights(None, xs, ps, sqrt_s_gev, mass_gev=mass_gev)
    hsum = np.zeros((len(xedges) - 1, len(pedges) - 1), dtype=float)
    ix = np.digitize(xs, xedges) - 1
    ip = np.digitize(ps, pedges) - 1
    for i, j, v in zip(ix, ip, vs):
        if 0 <= i < hsum.shape[0] and 0 <= j < hsum.shape[1]:
            hsum[i, j] += v
    out = hsum[ix_data, ip_data]
    if convention == "invariant":
        out = out / _grid_point_areas(xedges, pedges, ix_data, ip_data)
    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark SPS hadron spectra vs model hypotheses.")
    parser.add_argument("--species", default="pi-", help="One of: pi-, pi+, k+, k-, p, pbar")
    parser.add_argument("--na49-data", default="", help="Path to NA49 SPS table (xF pT value error)")
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory. If empty, uses 00runs/<run_description>.",
    )
    parser.add_argument("--run-description", default="local_default")
    parser.add_argument("--out-prefix", default="", help="Output prefix; defaults to na49_benchmark_<species_tag>")
    parser.add_argument("--seed", type=int, default=20260329)
    parser.add_argument("--beam-momentum", type=float, default=158.0, help="Fixed-target proton beam momentum [GeV/c]")
    parser.add_argument("--pythia-events", type=int, default=120000)
    parser.add_argument("--disable-pythia", action="store_true")
    parser.add_argument("--res-events-per-res", type=int, default=30000)
    parser.add_argument(
        "--res-max-decay-passes",
        type=int,
        default=8,
        help="Maximum decay-chain passes for resonance cascading (1 disables cascading).",
    )
    parser.add_argument(
        "--res-disable-regge-tower",
        action="store_true",
        help="Disable effective high-mass N* tower components in resonance generation.",
    )
    parser.add_argument(
        "--res-regge-tower-min-mass",
        type=float,
        default=1.9,
        help="Lower anchor mass [GeV] for effective N* tower generation (level-1).",
    )
    parser.add_argument(
        "--res-regge-tower-max-mass",
        type=float,
        default=2.3,
        help="Upper cutoff mass [GeV] for effective N* tower generation (level-3).",
    )
    parser.add_argument(
        "--res-regge-tower-level-count",
        type=int,
        default=3,
        help="Number of effective N* tower levels to materialize in production templates.",
    )
    parser.add_argument(
        "--res-regge-tower-shape",
        choices=REGGE_TOWER_SHAPES,
        default="linear",
        help=(
            "Mass-ladder shape for effective N* tower: "
            "linear in m, linear in m^2 (regge_m2), or Hagedorn-like exponential density (hagedorn)."
        ),
    )
    parser.add_argument(
        "--res-regge-tower-shape-param",
        type=float,
        default=0.25,
        help=(
            "Shape parameter used for --res-regge-tower-shape=hagedorn "
            "(interpreted as T_H in GeV for rho(m)~exp(m/T_H))."
        ),
    )
    parser.add_argument(
        "--res-regge-tower-offshell-kappa",
        type=float,
        default=0.0,
        help=(
            "Stage-1 off-shell broadening strength for effective N* ladder widths: "
            "Gamma_eff = Gamma_base * (1 + kappa * x^beta), x in [0,1] along the ladder."
        ),
    )
    parser.add_argument(
        "--res-regge-tower-offshell-beta",
        type=float,
        default=1.0,
        help="Exponent beta in the off-shell width broadening law for effective N* ladder levels.",
    )
    parser.add_argument(
        "--res-regge-tower-offshell-window-scale",
        type=float,
        default=1.0,
        help=(
            "Scale factor for truncated mass-window bounds in tower generator files. "
            "Values >1 broaden the sampled off-shell window."
        ),
    )
    parser.add_argument(
        "--res-workers",
        type=int,
        default=1,
        help="Number of worker processes for resonance component generation within this job.",
    )
    parser.add_argument(
        "--res-fragmentation-mode",
        choices=FRAGMENTATION_MODES,
        default="both",
        help="Event-level resonance source mode: projectile, target, or both (default).",
    )
    parser.add_argument(
        "--res-split-tower-by-source",
        action="store_true",
        help=(
            "Split effective high-mass N* tower templates into projectile-side and "
            "target-side subcomponents (only meaningful with --res-fragmentation-mode both)."
        ),
    )
    parser.add_argument(
        "--res-target-feedover-xf-max",
        type=float,
        default=-1.0,
        help=(
            "If >=0, keep target-source tracks in projectile hemisphere only up to xF<=value. "
            "Negative disables this cut."
        ),
    )
    parser.add_argument(
        "--res-projectile-feedover-xf-max",
        type=float,
        default=-1.0,
        help=(
            "If >=0, keep projectile-source tracks in target hemisphere only up to |xF|<=value. "
            "Negative disables this cut."
        ),
    )
    parser.add_argument(
        "--res-cache-dir",
        default="",
        help=(
            "Optional directory for shared resonance component track cache. "
            "Each component stores final-state tracks (pdg, xF, pT) and can be reused across species."
        ),
    )
    parser.add_argument(
        "--res-cache-scope",
        choices=("species", "all"),
        default="species",
        help=(
            "When using --res-cache-dir: build only components for this species "
            "('species') or build the global union once ('all')."
        ),
    )
    parser.add_argument(
        "--res-cache-force-rebuild",
        action="store_true",
        help="Force regeneration of cached resonance components even if cache files already match configuration.",
    )
    parser.add_argument(
        "--res-soft-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override build_soft_resonance_decays branching parameters (repeatable), "
            "e.g. --res-soft-override nstar_eff_l3_br_nstarpi=0.60"
        ),
    )
    parser.add_argument(
        "--weak-feed-policy",
        choices=("inclusive_weak", "primary_only"),
        default="primary_only",
        help=(
            "Weak-decay feed-down policy for resonance generation: "
            "'inclusive_weak' keeps weak decay products, 'primary_only' drops them."
        ),
    )
    parser.add_argument("--disable-resonance", action="store_true")
    parser.add_argument("--mixing2006-dir", default=str(REPO_ROOT))
    parser.add_argument("--xf-min", type=float, default=0.0)
    parser.add_argument("--xf-max", type=float, default=0.4)
    parser.add_argument("--pt-min", type=float, default=0.05)
    parser.add_argument("--pt-max", type=float, default=1.5)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument(
        "--fit-weight-mode",
        choices=FIT_WEIGHT_MODES,
        default="sigma",
        help=(
            "Objective row-weighting mode for model normalization/fits: "
            "'sigma' (1/sigma), 'relative' (y/sigma), "
            "'relative_xfpt_balanced' (relative with balanced xF/pT macro-bins)."
        ),
    )
    parser.add_argument(
        "--fit-balance-xf-bins",
        type=int,
        default=4,
        help="Number of rank bins in xF used when --fit-weight-mode=relative_xfpt_balanced.",
    )
    parser.add_argument(
        "--fit-balance-pt-bins",
        type=int,
        default=4,
        help="Number of rank bins in pT used when --fit-weight-mode=relative_xfpt_balanced.",
    )
    parser.add_argument(
        "--pythia-csv",
        default="",
        help=(
            "Optional external PYTHIA CSV (xf,pt,value). "
            "If provided, this is used for PYTHIA scoring instead of in-process generation."
        ),
    )
    parser.add_argument("--epos-csv", default="", help="Optional external EPOS CSV (xf,pt,value)")
    parser.add_argument("--urqmd-csv", default="", help="Optional external UrQMD CSV (xf,pt,value)")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    soft_decay_overrides = _parse_soft_decay_overrides(list(args.res_soft_override))

    spec = resolve_species(args.species)
    na49_path = Path(args.na49_data or spec.na49_table)
    out_dir = Path(args.out_dir) if args.out_dir else (RUNS_ROOT / args.run_description)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_prefix = args.out_prefix or f"na49_benchmark_{spec.tag}"
    summary_path = out_dir / f"{out_prefix}_summary.json"
    details_path = out_dir / f"{out_prefix}_details.npz"

    xf, pt, y, sigma = _read_na49_points(na49_path)
    mask = (
        (xf >= args.xf_min)
        & (xf <= args.xf_max)
        & (pt >= args.pt_min)
        & (pt <= args.pt_max)
        & (sigma > 0.0)
    )
    xf = xf[mask]
    pt = pt[mask]
    y = y[mask]
    sigma = sigma[mask]

    xedges = _edges_from_centers(xf)
    pedges = _edges_from_centers(pt)
    ix_data, ip_data = _grid_indices(xf, pt, xedges, pedges)

    n = len(y)
    sqrt_s_eff = _sqrt_s_from_fixed_target_beam_momentum(args.beam_momentum)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_train = int(args.train_frac * n)
    train_idx = np.sort(perm[:n_train])
    test_idx = np.sort(perm[n_train:])
    fit_row_scale_train, fit_weight_meta = _compute_fit_row_scales(
        y=y[train_idx],
        sigma=sigma[train_idx],
        xf=xf[train_idx],
        pt=pt[train_idx],
        mode=str(args.fit_weight_mode),
        balance_xf_bins=int(args.fit_balance_xf_bins),
        balance_pt_bins=int(args.fit_balance_pt_bins),
    )

    summary: List[Dict[str, float | int | str]] = []
    pythia_pred = None

    if args.disable_pythia:
        summary.append({"model": "PYTHIA8_SoftQCD_all", "status": "disabled"})
        pythia_status = "disabled"
    else:
        pythia_status = "not_run"
        try:
            if args.pythia_csv:
                pythia_path = Path(args.pythia_csv)
                if not pythia_path.exists():
                    raise FileNotFoundError(f"PYTHIA CSV not found: {pythia_path}")
                pythia_counts = _load_external_grid_counts(
                    pythia_path,
                    xedges,
                    pedges,
                    ix_data,
                    ip_data,
                    convention="invariant",
                    sqrt_s_gev=sqrt_s_eff,
                    mass_gev=PDG_MASS_GEV[spec.pdg],
                )
                pythia_source = "external_csv"
            else:
                pythia_counts = _run_pythia(
                    args.pythia_events,
                    args.seed + 7,
                    sqrt_s_eff,
                    spec.pdg,
                    xedges,
                    pedges,
                    ix_data,
                    ip_data,
                    convention="invariant",
                )
                pythia_source = "internal"
            alpha = _fit_single_scale(
                pythia_counts[train_idx],
                y[train_idx],
                sigma[train_idx],
                row_scale=fit_row_scale_train,
            )
            pythia_pred = alpha * pythia_counts
            row: Dict[str, float | int | str] = {
                "model": "PYTHIA8_SoftQCD_all",
                "n_params": 1,
                "source": pythia_source,
            }
            row.update(_metrics_bundle(pythia_pred[train_idx], y[train_idx], sigma[train_idx], n_params=1, prefix="train"))
            row.update(_metrics_bundle(pythia_pred[test_idx], y[test_idx], sigma[test_idx], n_params=1, prefix="test"))
            summary.append(row)
            pythia_status = pythia_source
        except Exception as exc:
            summary.append({"model": "PYTHIA8_SoftQCD_all", "status": f"failed: {exc}"})
            pythia_status = f"failed: {exc}"

    resonance_labels: List[str] = []
    resonance_weights: Dict[str, float] = {}
    pred_res = np.zeros_like(y)
    M = np.zeros((len(y), 0), dtype=float)
    w = np.array([], dtype=float)
    res_cache_dir = Path(args.res_cache_dir).resolve() if args.res_cache_dir else None
    regge_tower_gen_files: Optional[Dict[str, Path]] = None
    if not bool(args.res_disable_regge_tower):
        regge_tower_gen_files = _materialize_regge_tower_gen_files(
            mixing2006_dir=Path(args.mixing2006_dir),
            tower_gen_root=out_dir / "tower_gen",
            min_mass=float(args.res_regge_tower_min_mass),
            max_mass=float(args.res_regge_tower_max_mass),
            level_count=int(args.res_regge_tower_level_count),
            shape=str(args.res_regge_tower_shape),
            shape_param=float(args.res_regge_tower_shape_param),
            offshell_kappa=float(args.res_regge_tower_offshell_kappa),
            offshell_beta=float(args.res_regge_tower_offshell_beta),
            offshell_window_scale=float(args.res_regge_tower_offshell_window_scale),
        )
        masses = _compute_regge_tower_mass_ladder(
            float(args.res_regge_tower_min_mass),
            float(args.res_regge_tower_max_mass),
            level_count=int(args.res_regge_tower_level_count),
            shape=str(args.res_regge_tower_shape),
            shape_param=float(args.res_regge_tower_shape_param),
        )
        low = float(np.min(masses))
        high = float(np.max(masses))
        print(
            "resonance-tower: "
            + f"levels={int(args.res_regge_tower_level_count)} "
            + f"shape={str(args.res_regge_tower_shape)} "
            + f"mass-range=[{low:.3f}, {high:.3f}] GeV "
            + f"offshell(kappa={float(args.res_regge_tower_offshell_kappa):.3f}, "
            + f"beta={float(args.res_regge_tower_offshell_beta):.3f}, "
            + f"window_scale={float(args.res_regge_tower_offshell_window_scale):.3f})",
            flush=True,
        )

    if args.disable_resonance:
        summary.append({"model": "Resonance_NNLS", "status": "disabled"})
    else:
        try:
            resonance_labels, M = _run_resonance_components(
                mixing2006_dir=Path(args.mixing2006_dir),
                n_events_per_res=args.res_events_per_res,
                seed=args.seed + 19,
                particle_pdg=spec.pdg,
                max_decay_passes=args.res_max_decay_passes,
                fragmentation_mode=args.res_fragmentation_mode,
                res_workers=max(int(args.res_workers), 1),
                xedges=xedges,
                pedges=pedges,
                ix_data=ix_data,
                ip_data=ip_data,
                cache_dir=res_cache_dir,
                cache_scope=args.res_cache_scope,
                cache_force_rebuild=args.res_cache_force_rebuild,
                include_regge_tower=(not bool(args.res_disable_regge_tower)),
                regge_tower_gen_files=regge_tower_gen_files,
                soft_decay_overrides=soft_decay_overrides,
                split_tower_by_source=bool(args.res_split_tower_by_source),
                convention="invariant",
                sqrt_s_gev=float(sqrt_s_eff),
                target_feedover_xf_max=float(args.res_target_feedover_xf_max),
                projectile_feedover_xf_max=float(args.res_projectile_feedover_xf_max),
                weak_feed_policy=str(args.weak_feed_policy),
            )
            A = M[train_idx] * fit_row_scale_train[:, None]
            b = y[train_idx] * fit_row_scale_train
            w, _ = nnls(A, b)
            pred_res = M @ w
            n_eff = int(np.sum(w > 1e-12))
            row = {"model": "Resonance_NNLS", "n_params": n_eff}
            eff_params = max(n_eff, 1)
            row.update(_metrics_bundle(pred_res[train_idx], y[train_idx], sigma[train_idx], n_params=eff_params, prefix="train"))
            row.update(_metrics_bundle(pred_res[test_idx], y[test_idx], sigma[test_idx], n_params=eff_params, prefix="test"))
            summary.append(row)
            resonance_weights = {lbl: float(val) for lbl, val in zip(resonance_labels, w)}
        except Exception as exc:
            summary.append({"model": "Resonance_NNLS", "status": f"failed: {exc}"})

    for tag, csv_path in (("EPOS_external", args.epos_csv), ("UrQMD_external", args.urqmd_csv)):
        if not csv_path:
            summary.append({"model": tag, "status": "not_provided"})
            continue
        p = Path(csv_path)
        if not p.exists():
            summary.append({"model": tag, "status": f"missing_file:{csv_path}"})
            continue
        raw = _load_external_grid_counts(
            p,
            xedges,
            pedges,
            ix_data,
            ip_data,
            convention="invariant",
            sqrt_s_gev=sqrt_s_eff,
            mass_gev=PDG_MASS_GEV[spec.pdg],
        )
        alpha = _fit_single_scale(raw[train_idx], y[train_idx], sigma[train_idx], row_scale=fit_row_scale_train)
        pred = alpha * raw
        row = {"model": tag, "n_params": 1}
        row.update(_metrics_bundle(pred[train_idx], y[train_idx], sigma[train_idx], n_params=1, prefix="train"))
        row.update(_metrics_bundle(pred[test_idx], y[test_idx], sigma[test_idx], n_params=1, prefix="test"))
        summary.append(row)

    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "config": vars(args),
                "species": {
                    "name": spec.name,
                    "pdg": spec.pdg,
                    "tag": spec.tag,
                    "na49_table": str(na49_path),
                },
                "derived": {
                    "sqrt_s_from_beam_momentum": sqrt_s_eff,
                    "weak_feed_policy": str(args.weak_feed_policy),
                    "histogram_convention": "invariant",
                    "numpy_version": NUMPY_VERSION,
                    "numpy_bit_generator": NUMPY_BIT_GENERATOR,
                    "beam_momentum": float(args.beam_momentum),
                },
                "n_points": int(n),
                "train_points": int(len(train_idx)),
                "test_points": int(len(test_idx)),
                "resonance_components": resonance_labels,
                "resonance_weights": resonance_weights,
                "fit_weighting": {
                    "mode": str(args.fit_weight_mode),
                    "balance_xf_bins": int(args.fit_balance_xf_bins),
                    "balance_pt_bins": int(args.fit_balance_pt_bins),
                    "train_meta": fit_weight_meta,
                },
                "pythia_status": pythia_status,
                "soft_decay_overrides": soft_decay_overrides,
                "summary": summary,
            },
            fh,
            indent=2,
        )

    np.savez_compressed(
        details_path,
        xf=xf,
        pt=pt,
        y=y,
        sigma=sigma,
        train_idx=train_idx,
        test_idx=test_idx,
        resonance_pred=pred_res,
        resonance_matrix=M,
        resonance_weights=w,
        fit_row_scale_train=fit_row_scale_train,
        pythia_pred=np.array([] if pythia_pred is None else pythia_pred),
    )

    print(
        "Fit weighting:",
        {
            "mode": str(args.fit_weight_mode),
            "balance_xf_bins": int(args.fit_balance_xf_bins),
            "balance_pt_bins": int(args.fit_balance_pt_bins),
            "train_meta": fit_weight_meta,
        },
    )
    print(f"Saved summary: {summary_path}")
    print(f"Saved details: {details_path}")
    print("Model summary:")
    for item in summary:
        print(item)


if __name__ == "__main__":
    main()
