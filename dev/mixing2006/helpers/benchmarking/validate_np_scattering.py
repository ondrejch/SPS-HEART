#!/usr/bin/env python3
"""Validation driver for neutron-projectile reactions (n+p support).

Runs a fixed set of resonance components on p+p and n+p with shared
seeds and quantifies the model-internal isospin relations that anchor
the NA49 inverse-kinematics program:

1. Mirror identity: pi+(x_F) from the neutron-projectile hemisphere
   versus pi-(x_F) from the proton-projectile hemisphere (and the
   conjugate direction). Exact charge conjugation is required;
   kinematics are compared with bin-wise pulls.
2. Strangeness redistribution: summed charged-kaon spectra and
   per-component yield tables. Single conjugating components move
   charged-K content between hemispheres by construction, so shape
   agreement is asserted only on the summed mixture.
3. Antiproton yield ratio n+p versus p+p (optional; the current
   framework leaves the injected antibaryon sector unrotated, so the
   expected ratio is unity within errors; the known NA49 enhancement
   off neutrons is a Phase-4 calibration item, not a validation
   failure).
4. Per-event charge conservation through a dedicated event loop.
5. Composition cross-check: the f_p/f_n-weighted pi ratio built from
   direct n+p measurements versus the pp-only rotation prediction.

NA49 data overlays are supported through optional CSVs in --data-dir
with columns xf,value,err (bin centers) named
na49_{pp|np}_{piplus|piminus|kplus|kminus|pbar}_xf.csv. Absent files
are reported as pending digitization, never silently skipped.

Outputs land in --out-dir (default under RUNS_ROOT): summary.json and
summary.txt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

def _layout_roots(here: Path) -> Tuple[Path, Path]:
    """Return ``(mixing2006_dir, default_runs_root)`` for either tree.

    Vendored layout: ``mixing2006/helpers/benchmarking`` → package at
    ``mixing2006/na49_mixing``. Monorepo layout:
    ``01-NA49-res/helpers/benchmarking`` → ``dev/mixing2006``.
    """
    root = here.parent.parent
    if (root / "na49_mixing" / "parallel.py").is_file():
        return root, root / "00runs"
    mix = root / "dev" / "mixing2006"
    if (mix / "na49_mixing" / "parallel.py").is_file():
        return mix, root / "dev" / "00runs"
    return root, root / "00runs"


_HERE = Path(__file__).resolve().parent
_DEFAULT_MIXING_DIR, _DEFAULT_RUNS_ROOT = _layout_roots(_HERE)
if str(_DEFAULT_MIXING_DIR) not in sys.path:
    sys.path.insert(0, str(_DEFAULT_MIXING_DIR))
if str(_DEFAULT_MIXING_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_DEFAULT_MIXING_DIR.parent))

PI_PLUS = 211
PI_MINUS = -211
K_PLUS = 321
K_MINUS = -321
PBAR = -2212

DEFAULT_COMPONENTS: Tuple[str, ...] = (
    "generators/michal/deltas-to-pion/delta-.gen",
    "generators/michal/deltas-to-pion/delta0.gen",
    "generators/benchmark/kstar_zero_892.gen",
    "generators/benchmark/nstar_eff1900_zero.gen",
    "generators/benchmark/anti_nstar_eff1900_zero.gen",
)

X_F_EDGES = np.linspace(-1.0, 1.0, 15)


# ----------------------------------------------------------------------
# Generation plumbing
# ----------------------------------------------------------------------

def _stable_seed(base_seed: int, rel_path: str) -> int:
    digest = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:8]
    seed = (int(base_seed) + int(digest, 16)) % (2**31 - 1)
    return seed if seed > 0 else 1


def _run_component(
    mixing_dir: Path,
    gen_file: Path,
    *,
    beam_pdg: int,
    target_pdg: int,
    n_events: int,
    seed: int,
    max_decay_passes: int,
    include_pbar: bool,
) -> Dict[str, np.ndarray]:
    """Generate one component on one reaction system; return track arrays."""
    sys.path.insert(0, str(mixing_dir))
    from na49_mixing.parallel import GenerationConfig, generate_tracks_chunked

    cfg = GenerationConfig(
        gen_file=str(gen_file),
        n_events=int(n_events),
        seed=int(seed),
        beam_momentum=158.0,
        max_decay_passes=int(max_decay_passes),
        use_soft_decay_factory=True,
        fragmentation_mode="both",
        inject_antibaryon_resonances=bool(include_pbar),
        beam_pdg=int(beam_pdg),
        target_pdg=int(target_pdg),
    )
    tracks = generate_tracks_chunked(cfg, n_workers=1, n_chunks=1)
    return {
        "pdg": np.asarray(tracks["pdg"]),
        "xf": np.asarray(tracks["xf"], dtype=np.float64),
        "pt": np.asarray(tracks["pt"], dtype=np.float64),
        "side": np.asarray(tracks["source_side"]),
    }


# ----------------------------------------------------------------------
# Spectrum helpers
# ----------------------------------------------------------------------

def _binned_counts(
    run: Dict[str, np.ndarray], species: Tuple[int, ...],
    edges: np.ndarray, side_tag: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bin counts for one species group selected by construction-side tag.

    Selection uses the generator's source-side tag, deliberately NOT the
    sign of lab x_F: parent x_F distributions are symmetric for many
    cards, so 'projectile side' is a construction tag rather than a
    kinematic hemisphere, and soft backward-emitted daughters legitimately
    land at negative lab x_F inside a tagged hemisphere. Tag-versus-sign
    agreement is reported separately as a purity diagnostic.
    """
    sel = np.isin(run["pdg"], species)
    if side_tag is not None:
        sel &= run["side"] == side_tag
    counts, _ = np.histogram(run["xf"][sel], bins=edges)
    errs = np.sqrt(counts.astype(float))
    return counts.astype(float), errs


def hemisphere_purity(
    run: Dict[str, np.ndarray], species: Tuple[int, ...],
) -> Dict[str, object]:
    """Fraction of tag-selected tracks whose lab x_F sign matches the tag."""
    out: Dict[str, object] = {}
    for tag, name in ((+1, "projectile"), (-1, "target")):
        sel = np.isin(run["pdg"], species) & (run["side"] == tag)
        n = int(np.count_nonzero(sel))
        if n == 0:
            out[name] = None
            continue
        matching = np.count_nonzero(np.sign(run["xf"][sel]) == tag)
        out[name] = {
            "n_tracks": n,
            "purity": round(float(matching) / n, 4),
        }
    return out


def _mirror_pull_table(
    counts_a: np.ndarray, err_a: np.ndarray,
    counts_b: np.ndarray, err_b: np.ndarray,
    centers: np.ndarray,
    min_stats: float = 25.0,
) -> Dict[str, object]:
    """Bin-wise pulls between two spectra; low-stat bins are excluded."""
    ok = (err_a**2 + err_b**2) > 0
    ok &= (counts_a >= min_stats) | (counts_b >= min_stats)
    pulls = np.full(counts_a.shape, np.nan)
    pulls[ok] = (counts_a[ok] - counts_b[ok]) / np.sqrt(
        err_a[ok] ** 2 + err_b[ok] ** 2
    )
    finite = pulls[np.isfinite(pulls)]
    n_used = int(finite.size)
    chi2 = float(np.sum(finite**2)) if n_used else float("nan")
    dof = max(n_used - 0, 1)
    return {
        "bins_xf": [float(c) for c in centers],
        "pulls": [None if not math.isfinite(p) else round(float(p), 3)
                  for p in pulls],
        "n_bins_used": n_used,
        "chi2": round(chi2, 2),
        "chi2_per_dof": round(chi2 / dof, 3) if n_used else None,
        "max_abs_pull": round(float(np.max(np.abs(finite))), 2) if n_used else None,
    }


def _normalized_shape(
    xf: np.ndarray, edges: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    counts, _ = np.histogram(xf, bins=edges)
    return _normalized_counts(counts)


def _normalized_counts(
    counts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize an already-binned count histogram to fractions."""
    c = np.asarray(counts, dtype=float)
    total = float(c.sum())
    if total <= 0.0:
        z = np.zeros_like(c)
        return z, z
    frac = c / total
    return frac, np.sqrt(np.maximum(frac, 0.0) / total)


def _shape_smape(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.abs(a) + np.abs(b)
    ok = denom > 0
    if not np.any(ok):
        return float("nan")
    return float(100.0 * np.mean(
        2.0 * np.abs(a[ok] - b[ok]) / denom[ok]
    ))


def _normalized_pull_table(
    frac_a: np.ndarray, frac_b: np.ndarray,
    edges: np.ndarray, centers: np.ndarray,
) -> Dict[str, object]:
    """Bin-wise pulls between two normalized spectra.

    Normalization uncertainties follow the multinomial approximation
    sigma_i = sqrt(f_i (1 - f_i) / N) with equal effective N per spectrum.
    """
    n_eff = 1.0 / max(float(np.sum(frac_a**2)), 1e-12)
    sig_a = np.sqrt(np.maximum(frac_a * (1.0 - frac_a), 1e-12) / n_eff)
    sig_b = np.sqrt(np.maximum(frac_b * (1.0 - frac_b), 1e-12) / n_eff)
    pulls = np.full(frac_a.shape, np.nan)
    ok = (sig_a + sig_b) > 0
    pulls[ok] = (frac_a[ok] - frac_b[ok]) / np.sqrt(sig_a[ok]**2 + sig_b[ok]**2)
    finite = pulls[np.isfinite(pulls)]
    n_used = int(finite.size)
    chi2 = float(np.sum(finite**2)) if n_used else float("nan")
    return {
        "bins_xf": [float(c) for c in centers],
        "pulls": [None if not math.isfinite(p) else round(float(p), 3)
                  for p in pulls],
        "n_bins_used": n_used,
        "chi2": round(chi2, 2),
        "max_abs_pull": round(float(np.max(np.abs(finite))), 2) if n_used else None,
    }


# ----------------------------------------------------------------------
# Check 1: mirror identity (projectile hemispheres)
# ----------------------------------------------------------------------

def check_mirror_identity(
    pp: Dict[str, np.ndarray], np_: Dict[str, np.ndarray], centers: np.ndarray,
) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for name, sp_neg, sp_pos in (
        ("pi+", PI_MINUS, PI_PLUS),   # np pi+  vs  pp pi-
        ("pi-", PI_PLUS, PI_MINUS),   # np pi-  vs  pp pi+
    ):
        c_pp, e_pp = _binned_counts(pp, (sp_neg,), X_F_EDGES, side_tag=+1)
        c_np, e_np = _binned_counts(np_, (sp_pos,), X_F_EDGES, side_tag=+1)
        out[name] = _mirror_pull_table(c_np, e_np, c_pp, e_pp, centers)
        # Conjugation must hold on integrated yields exactly at the
        # channel level; report them alongside the binned pulls.
        out[f"{name}_integrated"] = {
            "n_pp_reference": float(c_pp.sum()),
            "n_np_conjugate": float(c_np.sum()),
        }
    return out


# ----------------------------------------------------------------------
# Check 2: strangeness redistribution
# ----------------------------------------------------------------------

def check_kaons(
    runs_pp: List[Dict[str, np.ndarray]],
    runs_np: List[Dict[str, np.ndarray]],
    centers: np.ndarray,
) -> Dict[str, object]:
    def _summed(runs: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        return {
            "xf": np.concatenate([r["xf"] for r in runs]),
            "pdg": np.concatenate([r["pdg"] for r in runs]),
        }

    s_pp, s_np = _summed(runs_pp), _summed(runs_np)
    out: Dict[str, object] = {"per_component_yields": []}
    # Charged-K content redistributes between conjugate channels, so the
    # assertion targets the SUMMED mixture shape; per-component numbers
    # are reported as context.
    for r, tag in ((s_pp, "pp"), (s_np, "np")):
        n_kp = int(np.count_nonzero(r["pdg"] == K_PLUS))
        n_km = int(np.count_nonzero(r["pdg"] == K_MINUS))
        out[f"summed_{tag}"] = {"n_kplus": n_kp, "n_kminus": n_km}

    kp_pp, ekp_pp = _binned_counts(s_pp, (K_PLUS,), X_F_EDGES)
    kp_np, ekp_np = _binned_counts(s_np, (K_PLUS,), X_F_EDGES)
    shape_pp, _ = _normalized_counts(kp_pp)
    shape_np, _ = _normalized_counts(kp_np)
    # smape is meaningful only on well-populated bins; tails with a few
    # tracks dominate it otherwise.
    well_populated = (kp_pp >= 50) & (kp_np >= 50)
    smape_val = (
        _shape_smape(shape_pp[well_populated], shape_np[well_populated])
        if np.any(well_populated) else float("nan")
    )
    out["kplus_summed_shape"] = {
        "smape_percent_well_populated": round(smape_val, 3),
        "n_bins_smape": int(np.count_nonzero(well_populated)),
        # Pulls on normalized fractions: yield differences between
        # conjugate channels are structural redistribution, not failure.
        "pulls_normalized": _normalized_pull_table(
            shape_np, shape_pp, X_F_EDGES, centers
        ),
    }
    return out


# ----------------------------------------------------------------------
# Check 3: antiproton ratio (optional)
# ----------------------------------------------------------------------

def check_pbar(
    runs_pp: List[Dict[str, np.ndarray]],
    runs_np: List[Dict[str, np.ndarray]],
) -> Dict[str, object]:
    def _count(runs: List[Dict[str, np.ndarray]]) -> int:
        pdg = np.concatenate([r["pdg"] for r in runs])
        return int(np.count_nonzero(pdg == PBAR))

    n_pp, n_np = _count(runs_pp), _count(runs_np)
    ratio = (float(n_np) / n_pp) if n_pp else float("nan")
    return {
        "n_pbar_pp": n_pp,
        "n_pbar_np": n_np,
        "ratio_np_over_pp": round(ratio, 4) if math.isfinite(ratio) else None,
        "note": (
            "Injected antibaryon sector is intentionally unrotated in the "
            "current framework; ratio near 1 within MC errors is the "
            "expected outcome. Reproducing the NA49 pbar enhancement off "
            "neutrons is a Phase-4 calibration decision."
        ),
    }


# ----------------------------------------------------------------------
# Check 4: per-event charge conservation
# ----------------------------------------------------------------------

def check_charge_conservation(
    mixing_dir: Path,
    gen_file: Path,
    *,
    beam_pdg: int,
    target_pdg: int,
    n_events: int,
    seed: int,
    max_decay_passes: int,
) -> Dict[str, object]:
    """Decay-level charge flow per hemisphere sub-process.

    Each generated hemisphere is an independent sub-process seeded by one
    resonance parent; the framework does not generate beam remnants, so
    the conservation statement is Q(final state) == Q(primary parent),
    not a full NN event charge balance.
    """
    sys.path.insert(0, str(mixing_dir))
    import numpy as _np
    from na49_mixing.pdg_database import get_charge
    from na49_mixing.event import Event
    from na49_mixing.reaction import Reaction
    from na49_mixing.decays import build_soft_resonance_decays
    from na49_mixing.parallel import (
        _patch_generator_mass_width,
        _mirror_projectile_to_target_in_place,
        _build_side_generator,
    )

    rng = _np.random.default_rng(seed)
    reaction = Reaction(158.0, beam_pdg=beam_pdg, target_pdg=target_pdg)
    gen = _build_side_generator(
        reaction, str(gen_file), {}, rng, side_pdg=int(reaction.beam_pdg())
    )
    gen_target = None
    if beam_pdg != target_pdg:
        gen_target = _build_side_generator(
            reaction, str(gen_file), {}, rng, side_pdg=int(reaction.target_pdg())
        )

    ev = Event(rng=rng)
    ev._decay_chain = dict(build_soft_resonance_decays())
    ev.reset_decay_stats()

    violations = 0
    hist: Dict[str, int] = {}

    def _run_side(generator) -> None:
        nonlocal violations
        part = generator.generate()
        if part is None:
            return
        q_parent = float(get_charge(int(generator._id)))
        ev.add_particle(part)
        ev.mark_primary(part)
        ev.decay_cascade(max_passes=max_decay_passes)
        q_final = sum(
            float(get_charge(int(p.get_id()))) for p in ev.particles()
        )
        dq = round(q_final - q_parent, 6)
        key = str(dq)
        hist[key] = hist.get(key, 0) + 1
        if abs(dq) > 1e-9:
            violations += 1
        ev.delete()

    for _ in range(int(n_events)):
        _run_side(gen)
        _run_side(gen_target or gen)

    return {
        "gen_file": gen_file.name,
        "system": ("np" if beam_pdg != target_pdg else "pp"),
        "n_side_events": int(n_events),
        "violations": violations,
        "delta_q_histogram": hist,
    }


# ----------------------------------------------------------------------
# Check 5b: final-state neutrons versus NA49 pp measurement
# ----------------------------------------------------------------------

def _load_na49_neutron_table(path: Path) -> Tuple[np.ndarray, ...]:
    xs, ys, es = [], [], []
    for line in path.read_text(errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 3:
            continue
        xf, v, e = float(parts[0]), float(parts[1]), float(parts[2])
        xs.append(xf)
        ys.append(v)
        es.append(abs(v) * abs(e) / 100.0)
    return np.asarray(xs), np.asarray(ys), np.asarray(es)


def check_final_state_neutrons(
    runs_pp: List[Dict[str, np.ndarray]],
    runs_np: List[Dict[str, np.ndarray]],
    n_events_per_run: int,
    centers: np.ndarray,
) -> Dict[str, object]:
    """Shape benchmark: generated final-state neutrons versus NA49 pp data.

    Uses the repository's tabulated NA49 measurement of dN/dxF for
    neutrons in p+p at 158 GeV/c. Single-component superpositions have no
    absolute per-inelastic-event normalization, so a single scale factor
    is fitted per system and only shape agreement is asserted.
    """
    candidates = []
    base = _DEFAULT_MIXING_DIR
    for _ in range(6):
        candidates.append(
            base / "spshadrons.web.cern.ch" / "data"
            / "xsect_ptintegrated_neutron_pp.txt"
        )
        if base.parent == base:
            break
        base = base.parent
    table_path = next((c for c in candidates if c.exists()), None)
    if table_path is None:
        return {
            "status": "absent",
            "note": "NA49 neutron pp table not found in repository.",
        }

    xd, yd, ed = _load_na49_neutron_table(table_path)
    bin_width = float(X_F_EDGES[1] - X_F_EDGES[0])
    out: Dict[str, object] = {
        "table": str(table_path),
        "n_data_points": int(len(xd)),
        "n_events_per_run": int(n_events_per_run),
        "note": (
            "Equal-weight component superposition is NOT the Fischer-fit "
            "mixture; a large shape chi2 against real NA49 data is the "
            "expected outcome here and quantifies the gap Phase-4 fitted "
            "weighting must close. Only the fitted_scale and its "
            "stability across systems carry interpretive weight."
        ),
        "systems": {},
    }
    for tag, runs in (("pp", runs_pp), ("np", runs_np)):
        pdg = np.concatenate([r["pdg"] for r in runs])
        xf = np.concatenate([r["xf"] for r in runs])
        sel = pdg == 2112
        counts, _ = np.histogram(xf[sel], bins=X_F_EDGES)
        n_events_total = len(runs) * int(n_events_per_run)
        density = counts.astype(float) / (n_events_total * bin_width)
        dens_err = np.sqrt(counts.astype(float)) / (n_events_total * bin_width)

        pairs = []
        for xi, yi, ei in zip(xd, yd, ed):
            j = int(np.argmin(np.abs(centers - xi)))
            if abs(centers[j] - xi) <= bin_width / 2 and dens_err[j] > 0:
                pairs.append((j, yi, ei))
        if not pairs:
            out["systems"][tag] = {"status": "no_overlap"}
            continue
        idx = np.array([p[0] for p in pairs])
        d = np.array([p[1] for p in pairs])
        s = np.array([p[2] for p in pairs])
        m = density[idx]
        # single scale fit: a = sum(d m / s^2) / sum(m^2 / s^2)
        w = 1.0 / np.maximum(s, 1e-12) ** 2
        a = float(np.sum(d * m * w) / np.sum(m * m * w))
        chi2 = float(np.sum(((a * m - d) ** 2) * w))
        dof = max(len(idx) - 1, 1)
        out["systems"][tag] = {
            "status": "ok",
            "n_bins": int(len(idx)),
            "fitted_scale": round(a, 5),
            "chi2": round(chi2, 2),
            "chi2_per_dof": round(chi2 / dof, 3),
            "model_neutrons_total": int(sel.sum()),
        }
    return out


# ----------------------------------------------------------------------
# Check 6: composition cross-check
# ----------------------------------------------------------------------

def check_composition(
    pp: Dict[str, np.ndarray],
    np_: Dict[str, np.ndarray],
    centers: np.ndarray,
    f_n_values: Tuple[float, ...] = (0.4, 0.5),
) -> Dict[str, object]:
    """f_n-weighted pi ratio: direct-np construction vs pp-only rotation.

    Both constructions must agree because the np generator is, by
    design, the reflected pp machinery; the check pins that identity
    numerically end-to-end through the full pipeline.
    """
    out: Dict[str, object] = {}
    c_pp_m, e_pp_m = _binned_counts(pp, (PI_MINUS,), X_F_EDGES, side_tag=+1)
    c_pp_p, e_pp_p = _binned_counts(pp, (PI_PLUS,), X_F_EDGES, side_tag=+1)
    c_np_p, e_np_p = _binned_counts(np_, (PI_PLUS,), X_F_EDGES, side_tag=+1)
    c_np_m, e_np_m = _binned_counts(np_, (PI_MINUS,), X_F_EDGES, side_tag=+1)

    for f_n in f_n_values:
        f_p = 1.0 - f_n
        # Direct construction: proton frag from pp, neutron frag from np.
        num_direct = f_p * c_pp_m + f_n * c_np_p
        den_direct = f_p * c_pp_p + f_n * c_np_m
        # Rotation-only construction from pp alone.
        num_rot = (f_p + f_n) * c_pp_m
        den_rot = (f_p + f_n) * c_pp_p
        with np.errstate(divide="ignore", invalid="ignore"):
            r_direct = np.where(den_direct > 0, num_direct / den_direct, np.nan)
            r_rot = np.where(den_rot > 0, num_rot / den_rot, np.nan)
        ok = np.isfinite(r_direct) & np.isfinite(r_rot) & (den_direct > 25)
        if np.any(ok):
            rel = np.abs(r_direct[ok] - r_rot[ok]) / r_rot[ok]
            out[f"f_n={f_n:.2f}"] = {
                "n_bins_used": int(np.count_nonzero(ok)),
                "max_rel_diff": round(float(np.max(rel)), 5),
                "mean_rel_diff": round(float(np.mean(rel)), 5),
            }
        else:
            out[f"f_n={f_n:.2f}"] = {"n_bins_used": 0}
    return out


# ----------------------------------------------------------------------
# Data overlay hook
# ----------------------------------------------------------------------

DATA_SPECS = ("piplus", "piminus", "kplus", "kminus", "pbar")
_OVERLAY_PDG = {
    "piplus": PI_PLUS,
    "piminus": PI_MINUS,
    "kplus": K_PLUS,
    "kminus": K_MINUS,
    "pbar": PBAR,
}


def _load_overlay_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read an overlay CSV with columns xf,value,err (bin centers)."""
    xs: List[float] = []
    ys: List[float] = []
    es: List[float] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.replace(",", " ").split()
            if len(cols) < 3:
                continue
            try:
                xf, val, err = float(cols[0]), float(cols[1]), float(cols[2])
            except ValueError:
                continue
            xs.append(xf)
            ys.append(val)
            es.append(err)
    return (
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=float),
        np.asarray(es, dtype=float),
    )


def check_data_overlays(
    data_dir: Optional[Path],
    summed_pp: Optional[Dict[str, np.ndarray]] = None,
    summed_np: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, object]:
    out: Dict[str, object] = {}
    expected = [
        f"na49_{sys}_{sp}_xf.csv" for sys in ("pp", "np") for sp in DATA_SPECS
    ]
    if data_dir is None or not Path(data_dir).is_dir():
        out["status"] = "absent"
        out["expected_files"] = expected
        out["note"] = (
            "Digitization of NA49 n+p proceedings/thesis figures is a "
            "pending acquisition task (see notes/benchmarks/); overlays "
            "score generated xF spectra against xf,value,err CSVs when "
            "those files are present."
        )
        return out
    found = sorted(p.name for p in Path(data_dir).glob("na49_*_xf.csv"))
    out["files_found"] = found
    overlays: Dict[str, object] = {}
    any_loaded = False
    for sys_name, tracks in (("pp", summed_pp), ("np", summed_np)):
        for sp in DATA_SPECS:
            name = f"na49_{sys_name}_{sp}_xf.csv"
            path = Path(data_dir) / name
            key = f"{sys_name}_{sp}"
            if not path.is_file():
                overlays[key] = {"status": "missing", "file": name}
                continue
            xf, val, err = _load_overlay_csv(path)
            rec: Dict[str, object] = {
                "status": "loaded",
                "file": name,
                "n_points": int(xf.size),
            }
            any_loaded = True
            if tracks is not None and xf.size >= 2:
                pdg = _OVERLAY_PDG[sp]
                mask = np.asarray(tracks["pdg"]) == pdg
                model_xf = np.asarray(tracks["xf"])[mask]
                edges = np.empty(xf.size + 1, dtype=float)
                edges[1:-1] = 0.5 * (xf[:-1] + xf[1:])
                edges[0] = xf[0] - 0.5 * (xf[1] - xf[0])
                edges[-1] = xf[-1] + 0.5 * (xf[-1] - xf[-2])
                counts, _ = np.histogram(model_xf, bins=edges)
                total = float(counts.sum())
                model = counts.astype(float) / total if total > 0.0 else counts.astype(float)
                data = val / float(val.sum()) if float(val.sum()) > 0.0 else val
                ok = err > 0
                if np.any(ok) and total > 0.0:
                    pulls = (model[ok] - data[ok]) / np.maximum(err[ok] / max(float(val.sum()), 1e-12), 1e-12)
                    rec["n_bins_compared"] = int(np.count_nonzero(ok))
                    rec["max_abs_pull"] = round(float(np.max(np.abs(pulls))), 3)
                    rec["chi2"] = round(float(np.sum(pulls**2)), 2)
                else:
                    rec["n_bins_compared"] = 0
            overlays[key] = rec
    out["status"] = "present" if any_loaded else "absent"
    out["overlays"] = overlays
    return out


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="validate_np_scattering",
        description=__doc__.splitlines()[0],
    )
    ap.add_argument("--mixing2006-dir", default=str(_DEFAULT_MIXING_DIR))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-events", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--max-decay-passes", type=int, default=8)
    ap.add_argument("--include-pbar", action="store_true")
    ap.add_argument("--charge-events", type=int, default=4000)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args(argv)

    mixing_dir = Path(args.mixing2006_dir).resolve()
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else _DEFAULT_RUNS_ROOT / "np_validation"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    centers = 0.5 * (X_F_EDGES[:-1] + X_F_EDGES[1:])

    results: Dict[str, object] = {
        "n_events": int(args.n_events),
        "seed": int(args.seed),
        "components": [],
    }

    runs_pp: List[Dict[str, np.ndarray]] = []
    runs_np: List[Dict[str, np.ndarray]] = []

    for idx, rel in enumerate(DEFAULT_COMPONENTS):
        gen_file = mixing_dir / rel
        if not gen_file.exists():
            print(f"SKIP missing component card: {rel}")
            continue
        comp_seed = _stable_seed(args.seed, rel)
        entry: Dict[str, object] = {"card": rel, "seed": comp_seed}
        print(f"[{idx+1}/{len(DEFAULT_COMPONENTS)}] {rel}", flush=True)
        pp = _run_component(
            mixing_dir, gen_file, beam_pdg=2212, target_pdg=2212,
            n_events=args.n_events, seed=comp_seed,
            max_decay_passes=args.max_decay_passes,
            include_pbar=args.include_pbar,
        )
        np_ = _run_component(
            mixing_dir, gen_file, beam_pdg=2112, target_pdg=2212,
            n_events=args.n_events, seed=comp_seed,
            max_decay_passes=args.max_decay_passes,
            include_pbar=args.include_pbar,
        )
        runs_pp.append(pp)
        runs_np.append(np_)
        entry["mirror"] = check_mirror_identity(pp, np_, centers)
        entry["hemisphere_purity"] = {
            "pp": hemisphere_purity(pp, (PI_PLUS, PI_MINUS)),
            "np": hemisphere_purity(np_, (PI_PLUS, PI_MINUS)),
        }
        results["components"].append(entry)

    def _concat(runs: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        return {k: np.concatenate([r[k] for r in runs]) for k in runs[0]}

    summed_pp, summed_np = _concat(runs_pp), _concat(runs_np)
    results["kaons"] = check_kaons(runs_pp, runs_np, centers)
    results["final_state_neutrons"] = check_final_state_neutrons(
        runs_pp, runs_np, int(args.n_events), centers
    )
    if args.include_pbar:
        results["pbar"] = check_pbar(runs_pp, runs_np)
    results["composition"] = check_composition(summed_pp, summed_np, centers)

    cons = []
    for rel in DEFAULT_COMPONENTS[:2]:
        gen_file = mixing_dir / rel
        if not gen_file.exists():
            continue
        cons.append(check_charge_conservation(
            mixing_dir, gen_file,
            beam_pdg=2112, target_pdg=2212,
            n_events=args.charge_events,
            seed=_stable_seed(args.seed, rel + "|charge"),
            max_decay_passes=args.max_decay_passes,
        ))
    results["charge_conservation"] = cons

    results["data_overlay"] = check_data_overlays(
        Path(args.data_dir) if args.data_dir else None,
        summed_pp=summed_pp,
        summed_np=summed_np,
    )

    # Verdicts ----------------------------------------------------------
    verdict_rows = []
    for entry in results["components"]:
        mirror = entry["mirror"]
        for sp in ("pi+", "pi-"):
            tab = mirror[sp]
            chi2dof = tab.get("chi2_per_dof")
            verdict_rows.append((entry["card"].split("/")[-1], sp, chi2dof))
    results["verdicts"] = [
        {"card": c, "species": s, "chi2_per_dof": v} for c, s, v in verdict_rows
    ]

    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    lines = [f"n+p validation — {args.n_events} events/component, "
             f"seed {args.seed}", ""]
    for row in verdict_rows:
        lines.append(f"  mirror {row[0]:24s} {row[1]:5s} chi2/dof = {row[2]}")
    lines.append("")
    lines.append(json.dumps(results["kaons"]["kplus_summed_shape"], indent=2))
    lines.append(json.dumps(results["composition"], indent=2))
    for c in cons:
        lines.append(f"  charge {c['system']} {c['gen_file']}: "
                     f"{c['violations']} violations / {2*c['n_side_events']} hemispheres")
    (out_dir / "summary.txt").write_text("\n".join(lines))
    print(f"\nwrote {out_dir/'summary.json'}")
    print("\n".join(lines[:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
