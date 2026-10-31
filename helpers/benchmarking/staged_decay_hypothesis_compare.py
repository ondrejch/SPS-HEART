#!/usr/bin/env python3
"""Three-hypothesis decay-model comparison driver (card TASK-20260905-physics-review, M7 phase B).

Runs the staged p+p excitation event builder under the three decay
hypotheses of the M7 phase-A selector (``continuum.decay_hypothesis``):

- ``H1``: exact Lorentz-invariant phase space;
- ``H2``: explicit sequential resonances (Delta(1232) and rho(770) with
  PDG masses and widths);
- ``H3``: the retained recursive empirical kernel (default).

At each declared parameter point the driver reports, for each hypothesis:

- the M4 calibration observables, the binned per-species chi-square,
  chi-square per degree of freedom, and WRRMSE over the windowed NA49
  tables in cross-section-per-inelastic-event units (``f / sigma_inel``
  with ``sigma_inel = 31.78`` mb, the paper window ``x_F`` in [0.0, 0.4]
  and ``p_T`` in [0.05, 1.5] GeV/c);
- the mean final multiplicity over the full event;
- the longitudinal (``x_F``) spectra, the per-inelastic-event
  ``dn/dx_F`` obtained by integrating the binned invariant density over
  ``p_T^2`` (``pi * int M dp_T^2`` with ``M = f / sigma_inel``);
- the mean transverse momentum profiles under both convention subscripts
  of the AGENTS.md terminology rule: ``mean_pt_x`` is the
  xF-density convention ``<p_T>_x`` (density ``p_T/E * f``; with the
  invariant per-track weight ``E/p_T`` the pT weight of every track is
  constant, so the profile is the unweighted track mean) and
  ``mean_pt_y`` is the y-density convention ``<p_T>_y`` (density
  ``p_T * f``; the pT weight of a track is proportional to its energy,
  so the profile is the energy-weighted track mean).

Conservation is re-asserted on every run event: ``event.validate()`` at
``1e-10`` (four-momentum closure, integer ``B, Q, S`` globally and per
source, and the channel-threshold rule that the sum of the daughter
masses does not exceed the source mass). For the ``H2`` arms the driver
additionally audits every resonance pair against its daughter threshold.

The model uncertainty of the decay kinematics is the spread of the
fitted source-mass and channel-strength parameters across hypotheses.
Each hypothesis is refitted with the M4 binned objective (same
parameters, bounds, and initial values) at the pre-registered reduced
refit settings, because a full M4-style fit at 2000 events per
evaluation is infeasible under ``H1`` (measured per-event cost about
138 ms against about 1.4 ms for ``H3``; see
``dev/00runs/tmp/m7_phase_b/timing_probe.py``). The reduced settings and
their noise-floor caveat are pre-registered in
``dev/00runs/staged_decay_compare_2026-09-05/pre_registration.json``.

The shared benchmark conventions (table I/O, paper window, metric
bundles) come from ``sps_core``; the binned objective, the event
generation, and the refit machinery are reused from
``na49_mixing.staged_evaluation`` (no private copies of that math).

Layout: the two copies of this driver (the top-level helper tree and the
vendored helper tree) differ only in the layout-path defaults for
``RUNS_ROOT`` and ``--mixing2006-dir``; the layout-sync check normalizes
that difference.  Run artifacts land under the tree-local run-artifact
root (see ``RUNS_ROOT``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from helpers.benchmarking import sps_core

RUNS_ROOT = REPO_ROOT / "dev" / "00runs"

# --- Pre-registered design (dev/00runs/staged_decay_compare_2026-09-05) ---
HYPOTHESES = ("H1", "H2", "H3")
PARAMETER_POINTS = ("post_fit", "initial")
# M4 fit result (dev/00runs/staged_binned_fit_2026-09-05/): the two moved
# production-shape parameters; the other eight fit parameters stay at
# their bundled initial values in both parameter points.
POST_FIT = {
    "regge.alpha_exc0": 0.2327532335505365,
    "regge.effective_transfer_slope": 8.262950772648646,
}
# One shared event sample per arm (hypothesis x parameter point); the
# count is uniform across hypotheses so the arms differ only in the decay
# kernel and the production-shape parameters.
OBSERVABLES_EVENTS = 500
# Seed protocol: config seed 20260831 + 30000 (M4 fit +10000, M5 audit
# +20000). One shared seed across all arms and all refits.
SEED_OFFSET = 30000
# Reduced refit settings per hypothesis (declared reduced alternative for
# the parameter spread; a full 2000-events-per-evaluation M4-style fit is
# infeasible under H1 at the measured ~138 ms/event).
FIT_EVENTS_PER_EVAL = {"H1": 100, "H2": 200, "H3": 200}
FIT_MAX_ITERATIONS = {"H1": 20, "H2": 50, "H3": 50}
# dof convention for the binned chi2/dof: the M4 fit carried 10 declared
# parameters, and the comparison arms reuse that convention so the
# chi2/dof values are comparable to the M4 report.
N_PARAMS_DOF = 10
# Fixed grids for the longitudinal and transverse-momentum observables.
XF_BINS = 20
XF_MIN, XF_MAX = 0.0, 0.4
# 0.05 GeV/c grid on [0.05, 1.55): the paper window pT <= 1.5 with the
# outermost bin extended by half a bin width, matching the M4 binned
# objective edge convention (edges_from_centers on the table pT centers).
PT_MIN, PT_MAX_EXCL = 0.05, 1.55
PT_BIN_WIDTH = 0.05
H2_THRESHOLD_MARGIN_GEV = 1e-9
ARTIFACT_SCHEMA_VERSION = 1


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_head(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(path),
            capture_output=True,
            text=True,
            timeout=10,
        )
        val = out.stdout.strip()
        return val if out.returncode == 0 and val else None
    except Exception:
        return None


def _to_jsonable(obj):
    """Recursively convert numpy scalars and non-finite floats for JSON."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _invariant_mass(a, b) -> float:
    e = a.e + b.e
    px = a.px + b.px
    py = a.py + b.py
    pz = a.pz + b.pz
    return math.sqrt(max(e * e - (px * px + py * py + pz * pz), 0.0))


def _arm_parameters(
    hypothesis: str,
    point: str,
    regge0,
    continuum0,
):
    """Parameter-set arm: the point's regge values plus the hypothesis."""
    if point not in PARAMETER_POINTS:
        raise ValueError(f"unknown parameter point {point!r}; expected {PARAMETER_POINTS}")
    regge = regge0
    if point == "post_fit":
        for name, value in POST_FIT.items():
            _group, field = name.split(".", 1)
            regge = replace(regge, **{field: float(value)})
    continuum = replace(continuum0, decay_hypothesis=hypothesis)
    return regge, continuum


def _h2_resonance_audit(events, get_mass) -> tuple[int, float]:
    """Re-derive the H2 deterministic pairing per source; assert thresholds.

    Returns (n_pairs, min_margin_gev) where the margin is the smallest
    (resonance invariant mass - daughter-threshold) over all audited
    pairs. A negative margin below -1e-9 GeV is a hard failure.
    """
    from na49_mixing.continuum_channels import _h2_resonant_pairs

    n_pairs = 0
    min_margin = math.inf
    for event in events:
        for source in event.sources:
            daughters = [
                p for p in event.final_particles if p.source_id == source.source_id
            ]
            ids = [p.pdg_id for p in daughters]
            for i_idx, j_idx, _res_id in _h2_resonant_pairs(ids):
                m_inv = _invariant_mass(daughters[i_idx].p4, daughters[j_idx].p4)
                threshold = get_mass(ids[i_idx]) + get_mass(ids[j_idx])
                margin = m_inv - threshold
                min_margin = min(min_margin, margin)
                if margin < -H2_THRESHOLD_MARGIN_GEV:
                    raise AssertionError(
                        f"event {event.event_id} source {source.source_id}: "
                        f"H2 pair at {m_inv:.6f} GeV below daughter threshold "
                        f"{threshold:.6f} GeV"
                    )
                n_pairs += 1
    if n_pairs == 0:
        return 0, 0.0
    return n_pairs, min_margin


def _xf_pt_observables(
    events,
    tables,
    sqrt_s_gev: float,
) -> dict[str, dict]:
    """Longitudinal xF spectra and mean-pT profiles on the fixed grid.

    dn/dxF (per inelastic event) = pi * int M dpT^2 with M = f/sigma_inel
    (the M4 binned invariant density in 1/GeV^2). With the invariant
    per-track weight E/pT the pT integration cancels the bin width and the
    per-track contribution is proportional to the track energy, so
    dn/dxF(bin) = 2/(sqrt(s) * N_ev * dxF) * sum(E).

    mean_pt_x: xF-density convention <p_T>_x (density p_T/E * f). The
    pT weight of a track is (p_T/E) * (E/p_T)/(dxF dpT), which is
    constant, so the profile is the unweighted track mean over the xF bin.
    mean_pt_y: y-density convention <p_T>_y (density p_T * f). The pT
    weight of a track is (p_T) * (E/p_T)/(dxF dpT), proportional to the
    track energy, so the profile is the energy-weighted track mean.
    """
    n_ev = float(len(events))
    if n_ev <= 0.0:
        raise ValueError("cannot build xF/pT observables from an empty sample")
    xedges = np.linspace(XF_MIN, XF_MAX, XF_BINS + 1)
    n_pt_bins = int(round((PT_MAX_EXCL - PT_MIN) / PT_BIN_WIDTH))
    pedges = np.linspace(PT_MIN, PT_MAX_EXCL, n_pt_bins + 1)
    pdgs: list[int] = []
    pzs: list[float] = []
    pts: list[float] = []
    es: list[float] = []
    for event in events:
        for particle in event.final_particles:
            pdgs.append(particle.pdg_id)
            p4 = particle.p4
            px = p4.px
            py = p4.py
            pz = p4.pz
            pt = math.hypot(px, py)
            pzs.append(pz)
            pts.append(pt)
            es.append(p4.e)
    pdg = np.asarray(pdgs, dtype=np.int64)
    pz = np.asarray(pzs, dtype=float)
    pt = np.asarray(pts, dtype=float)
    e = np.asarray(es, dtype=float)
    xf_all = 2.0 * pz / float(sqrt_s_gev)
    xf_width = float(xedges[1] - xedges[0])
    out: dict[str, dict] = {}
    for table in tables:
        sel = pdg == int(table.pdg_id)
        xf = xf_all[sel]
        pt_s = pt[sel]
        e_s = e[sel]
        ix, ip = sps_core.grid_indices(xf, pt_s, xedges, pedges)
        ok = (ix >= 0) & (ix < XF_BINS) & (ip >= 0) & (ip < n_pt_bins)
        ix_o, pt_o, e_o = ix[ok], pt_s[ok], e_s[ok]
        dn_dxf = (
            2.0
            / (float(sqrt_s_gev) * n_ev * xf_width)
            * np.bincount(ix_o, weights=e_o, minlength=XF_BINS)[:XF_BINS]
        )
        n_in = np.bincount(ix_o, minlength=XF_BINS)[:XF_BINS]
        s_pt = np.bincount(ix_o, weights=pt_o, minlength=XF_BINS)[:XF_BINS]
        s_pte = np.bincount(ix_o, weights=pt_o * e_o, minlength=XF_BINS)[:XF_BINS]
        s_e = np.bincount(ix_o, weights=e_o, minlength=XF_BINS)[:XF_BINS]
        mean_pt_x = np.where(
            n_in > 0, s_pt / np.maximum(n_in, 1), np.nan
        )
        mean_pt_y = np.where(
            s_e > 0.0, s_pte / np.maximum(s_e, 1e-30), np.nan
        )
        out[table.species] = {
            "dn_dx_f": dn_dxf,
            "total_yield_window": float(dn_dxf.sum()),
            "n_tracks_window": int(n_in.sum()),
            "mean_pt_x": mean_pt_x,
            "mean_pt_y": mean_pt_y,
            "window_mean_pt_x": float(np.sum(s_pt) / np.sum(n_in)) if n_in.sum() > 0 else None,
            "window_mean_pt_y": float(np.sum(s_pte) / np.sum(s_e)) if s_e.sum() > 0 else None,
        }
    return out


def _conservation_reassert(events) -> tuple[int, float]:
    """Re-assert event.validate() at 1e-10 on every event of one arm."""
    max_res = 0.0
    for event in events:
        event.validate()  # default relative_tolerance = 1e-10
        max_res = max(max_res, event.conservation_residuals()["relative_max"])
    return len(events), max_res


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mixing2006-dir",
        default=str(REPO_ROOT / "dev" / "mixing2006"),
        help="path to the mixing2006 package root (config, na49_mixing, tables)",
    )
    parser.add_argument(
        "--config",
        default="",
        help="staged excitation JSON config (default: <mixing2006-dir>/config/staged_excitation.json)",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory. If empty, uses dev/00runs/staged_decay_compare_2026-09-05.",
    )
    parser.add_argument(
        "--hypotheses",
        default=",".join(HYPOTHESES),
        help="comma-separated subset of H1,H2,H3",
    )
    parser.add_argument(
        "--points",
        default=",".join(PARAMETER_POINTS),
        help="comma-separated subset of post_fit,initial",
    )
    parser.add_argument(
        "--events",
        type=int,
        default=OBSERVABLES_EVENTS,
        help="events per arm for the observables run (pre-registered: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="shared seed for all arms and refits (default: config seed + 30000)",
    )
    parser.add_argument("--skip-refits", action="store_true", help="skip the per-hypothesis refits")
    parser.add_argument("--skip-observables", action="store_true", help="skip the observables arms")
    args = parser.parse_args(argv)

    mixdir = Path(args.mixing2006_dir)
    if not (mixdir / "na49_mixing").is_dir():
        raise SystemExit(f"na49_mixing package not found under {mixdir}")
    if str(mixdir) not in sys.path:
        sys.path.insert(0, str(mixdir))
    from na49_mixing.continuum_channels import ContinuumParameters
    from na49_mixing.exchange_model import (
        ReggeExcitationParameters,
        SimpleExcitationParameters,
    )
    from na49_mixing.pdg_database import get_mass
    from na49_mixing.reaction import Reaction
    from na49_mixing.staged_evaluation import (
        BinnedFitTable,
        _binned_mc_spectra,
        _dataclass_from_dict,
        _fit_regge_binned,
        _generate_pp,
        _resolve_binned_path,
        _validate_fit_parameters,
    )

    config_path = Path(args.config) if args.config else mixdir / "config" / "staged_excitation.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir) if args.out_dir else RUNS_ROOT / "staged_decay_compare_2026-09-05"
    out_dir.mkdir(parents=True, exist_ok=True)

    simple = _dataclass_from_dict(SimpleExcitationParameters, config.get("simple"))
    regge0 = _dataclass_from_dict(ReggeExcitationParameters, config.get("regge"))
    continuum0 = _dataclass_from_dict(ContinuumParameters, config.get("continuum"))
    beam_momentum = float(config.get("beam_momentum", 158.0))
    base_seed = int(config.get("seed", 20260831))
    seed = args.seed if args.seed is not None else base_seed + SEED_OFFSET
    hypotheses = [h.strip() for h in args.hypotheses.split(",") if h.strip()]
    for hyp in hypotheses:
        if hyp not in HYPOTHESES:
            raise SystemExit(f"unknown hypothesis {hyp!r}; expected one of {HYPOTHESES}")
    points = [p.strip() for p in args.points.split(",") if p.strip()]
    for point in points:
        if point not in PARAMETER_POINTS:
            raise SystemExit(f"unknown parameter point {point!r}; expected one of {PARAMETER_POINTS}")

    sqrt_s_gev = math.sqrt(Reaction(beam_momentum, 2212, 2212).s())
    canonical_config = json.dumps(config, sort_keys=True, separators=(",", ":"))
    configuration_hash = hashlib.sha256(canonical_config.encode("utf-8")).hexdigest()

    fit_cfg = dict(config.get("fit", {}))
    binned_cfg = dict(fit_cfg.get("binned", {}))
    if not binned_cfg:
        raise SystemExit("config carries no fit.binned block; the M4 binned tables are required")
    sigma_inel_mb = float(binned_cfg.get("sigma_inel_mb", 31.78))
    window_cfg = dict(binned_cfg.get("window", {}))
    window = {
        key: float(window_cfg[key])
        for key in ("xf_min", "xf_max", "pt_min", "pt_max")
        if key in window_cfg
    }
    path_root = Path(binned_cfg.get("path_root", mixdir))
    tables = [
        BinnedFitTable.from_path(
            _resolve_binned_path(entry["path"], path_root),
            species=str(entry["species"]),
            pdg_id=int(entry["pdg"]),
            mass_gev=float(entry["mass_gev"]),
            normalization=str(entry.get("normalization", "f")),
            sigma_inel_mb=sigma_inel_mb,
            window=window or None,
        )
        for entry in binned_cfg.get("tables", [])
    ]
    if not tables:
        raise SystemExit("fit.binned.tables names no tables")

    started = _utc_now_iso()
    t_start = time.perf_counter()
    report: dict = {
        "schema": f"staged_decay_hypothesis_compare/v{ARTIFACT_SCHEMA_VERSION}",
        "card": "TASK-20260905-physics-review",
        "milestone": "M7",
        "phase": "B",
        "run_description": "staged_decay_compare_2026-09-05",
        "date_utc": started,
        "provenance": {
            "outer_head": _git_head(REPO_ROOT),
            "nested_head": _git_head(mixdir),
            "config_path": str(config_path),
            "configuration_hash": configuration_hash,
            "configuration_hash_note": (
                "sha256 of the canonical JSON of the loaded config dict; the M4 "
                "report records 85861af2e4629b50239508360a7a046cfa988f0aaf13befc711cd9f671364042"
            ),
        },
        "seed_protocol": {
            "config_seed": base_seed,
            "seed_offset": SEED_OFFSET,
            "seed": int(seed),
            "note": (
                "one shared seed across all arms and all refits; offset 30000 "
                "is distinct from the M4 fit substream offset 10000 (20270831) "
                "and the M5 audit offset 20000 (20280831)"
            ),
        },
        "design": {
            "hypotheses": {
                "H1": "exact Lorentz-invariant phase space (M7 phase A)",
                "H2": "explicit sequential resonances, PDG masses/widths (M7 phase A)",
                "H3": "retained recursive empirical kernel (default)",
            },
            "parameter_points": {
                "post_fit": {
                    "description": (
                        "M4 calibration parameter set: the two moved production-shape "
                        "parameters at their fitted values, the other eight fit "
                        "parameters at their bundled initial values"
                    ),
                    "parameters": {k: float(v) for k, v in POST_FIT.items()},
                },
                "initial": {
                    "description": (
                        "pre-fit point: all ten fit parameters at their bundled "
                        "initial values"
                    ),
                },
            },
            "observables_events": int(args.events),
            "refit_settings": {
                hyp: {
                    "events_per_evaluation": FIT_EVENTS_PER_EVAL[hyp],
                    "max_iterations": FIT_MAX_ITERATIONS[hyp],
                }
                for hyp in hypotheses
            },
            "conventions": {
                "sigma_inel_mb": sigma_inel_mb,
                "sqrt_s_gev": sqrt_s_gev,
                "binned_window": window,
                "binned_n_params_dof": N_PARAMS_DOF,
                "xf_grid": {"bins": XF_BINS, "edges": [float(v) for v in np.linspace(XF_MIN, XF_MAX, XF_BINS + 1)]},
                "pt_grid": {"min": PT_MIN, "max_exclusive": PT_MAX_EXCL, "width": PT_BIN_WIDTH,
                            "note": "paper window pT <= 1.5 with the outermost bin extended by half a bin width"},
                "dn_dxf_definition": (
                    "per-inelastic-event dn/dxF = pi * int M dpT^2 with M = f/sigma_inel; "
                    "with the invariant per-track weight E/pT the per-track contribution "
                    "is proportional to the track energy: dn/dxF(bin) = 2/(sqrt(s)*N_ev*dxF)*sum(E)"
                ),
                "mean_pt_x_definition": (
                    "xF-density convention <p_T>_x (density p_T/E * f): the pT weight of "
                    "each track is constant, so the profile is the unweighted track mean"
                ),
                "mean_pt_y_definition": (
                    "y-density convention <p_T>_y (density p_T * f): the pT weight of a "
                    "track is proportional to its energy, so the profile is the "
                    "energy-weighted track mean"
                ),
            },
            "tables": [
                {
                    "species": t.species,
                    "pdg": int(t.pdg_id),
                    "path": t.path,
                    "normalization": t.normalization,
                    "n_points": int(t.data.size),
                    "n_sigma_fallback": int(t.n_sigma_fallback),
                }
                for t in tables
            ],
        },
    }

    # --- Observables arms --------------------------------------------------
    arms: dict[str, dict] = {}
    if not args.skip_observables:
        for hyp in hypotheses:
            for point in points:
                arm_key = f"{hyp}_{point}"
                regge_arm, continuum_arm = _arm_parameters(hyp, point, regge0, continuum0)
                t0 = time.perf_counter()
                events, summary = _generate_pp(
                    args.events, seed, simple, regge_arm, continuum_arm,
                    "regge", beam_momentum, "decay_compare",
                )
                wall = time.perf_counter() - t0
                n_validated, max_res = _conservation_reassert(events)
                arm = {
                    "hypothesis": hyp,
                    "parameter_point": point,
                    "seed": int(seed),
                    "n_events": len(events),
                    "wall_s": wall,
                    "parameters": {
                        "regge": {k: v for k, v in regge_arm.__dict__.items()},
                        "continuum": {k: v for k, v in continuum_arm.__dict__.items()},
                    },
                    "conservation": {
                        "n_events_validated": n_validated,
                        "tolerance": 1e-10,
                        "max_relative_residual": max_res,
                        "checks": "event.validate() per event: four-momentum closure at 1e-10 (global and per source), integer B,Q,S residuals (0,0,0) globally and per source, channel threshold (sum of daughter masses <= source mass + 1e-10)",
                    },
                    "mean_final_multiplicity": float(
                        sum(len(ev.final_particles) for ev in events) / len(events)
                    ),
                }
                if hyp == "H2":
                    n_pairs, min_margin = _h2_resonance_audit(events, get_mass)
                    arm["h2_resonance_audit"] = {
                        "n_pairs": n_pairs,
                        "min_margin_gev": min_margin,
                        "note": "deterministic Delta(1232)/rho(770) pairing re-derived from the final-particle ids; every resonance invariant mass at or above its daughter threshold",
                    }
                spectra = _binned_mc_spectra(events, tables, sqrt_s_gev, N_PARAMS_DOF)
                binned_payload = {}
                total_chi2 = 0.0
                for table in tables:
                    mc, chi2_table, metrics = spectra[table.species]
                    total_chi2 += chi2_table
                    binned_payload[table.species] = {
                        "n_bins": int(table.data.size),
                        "chi2": float(chi2_table),
                        "chi2_dof": float(metrics[f"{table.species}_chi2_dof"]),
                        "weighted_rel_rmse": float(metrics[f"{table.species}_weighted_rel_rmse"]),
                        "smape_percent": float(metrics[f"{table.species}_smape_percent"]),
                        "mc": mc,
                        "table_point_xf": table.xf_centers[table.bin_index_xf],
                        "table_point_pt": table.pt_centers[table.bin_index_pt],
                    }
                arm["binned_observables"] = {
                    "total_chi2": float(total_chi2),
                    "n_windowed_bins": int(sum(int(t.data.size) for t in tables)),
                    "species": binned_payload,
                }
                arm["xf_spectra"] = _xf_pt_observables(events, tables, sqrt_s_gev)
                arms[arm_key] = arm
                print(
                    f"[arm] {arm_key}: n={len(events)} wall={wall:.2f}s "
                    f"max_residual={max_res:.2e} "
                    f"<n>={arm['mean_final_multiplicity']:.3f} "
                    f"chi2_total={total_chi2:.1e}",
                    flush=True,
                )

    # --- Incremental persistence (the H1 refit can outlive a single shell) ---
    artifact = out_dir / "comparison_report.json"
    existing_refits: dict[str, dict] = {}
    if artifact.exists():
        existing = json.loads(artifact.read_text(encoding="utf-8"))
        existing_refits = dict(existing.get("refits", {}))
        if args.skip_observables:
            arms = dict(existing.get("arms", {}))
            if not arms:
                raise SystemExit(
                    "resuming without observables but the existing artifact carries no arms"
                )
    report["arms"] = arms
    report["refits"] = existing_refits
    report["wall_time_total_s"] = time.perf_counter() - t_start
    report["finished_utc"] = _utc_now_iso()
    _write_json(artifact, report)
    print(
        f"[persist] {artifact}: {len(arms)} arms, "
        f"{len(existing_refits)} refits",
        flush=True,
    )

    if args.skip_observables and not arms:
        raise SystemExit(
            "refits-only run requires the existing artifact to carry the arms"
        )

    # --- Per-hypothesis refits (parameter spread) ---------------------------
    names, initial, bounds = _validate_fit_parameters(fit_cfg, regge0, continuum0)
    refits: dict[str, dict] = dict(existing_refits)
    if not args.skip_refits:
        for hyp in hypotheses:
            fit_cfg_hyp = dict(fit_cfg)
            fit_cfg_hyp["events_per_evaluation"] = FIT_EVENTS_PER_EVAL[hyp]
            fit_cfg_hyp["max_iterations"] = FIT_MAX_ITERATIONS[hyp]
            continuum_hyp = replace(continuum0, decay_hypothesis=hyp)
            t0 = time.perf_counter()
            _fitted_regge, _fitted_continuum, fit_report = _fit_regge_binned(
                fit_cfg_hyp, simple, regge0, continuum_hyp,
                beam_momentum, seed, names, initial, bounds,
            )
            wall = time.perf_counter() - t0
            refits[hyp] = {
                "events_per_evaluation": FIT_EVENTS_PER_EVAL[hyp],
                "max_iterations": FIT_MAX_ITERATIONS[hyp],
                "seed": int(seed),
                "n_objective_evaluations": int(fit_report["n_objective_evaluations"]),
                "wall_s": wall,
                "success": bool(fit_report["success"]),
                "message": str(fit_report["message"]),
                "chi2": float(fit_report["chi2"]),
                "parameters": {k: float(v) for k, v in fit_report["parameters"].items()},
                "species": fit_report["species"],
            }
            print(
                f"[refit] {hyp}: nfev={fit_report['n_objective_evaluations']} "
                f"wall={wall / 60.0:.1f}min chi2={fit_report['chi2']:.3e}",
                flush=True,
            )
            report["refits"] = refits
            report["wall_time_total_s"] = time.perf_counter() - t_start
            report["finished_utc"] = _utc_now_iso()
            _write_json(artifact, report)
            print(
                f"[persist] {artifact}: {len(arms)} arms, "
                f"{len(refits)} refits",
                flush=True,
            )

    # --- Parameter spread (requires all three hypotheses) -------------------
    missing = [h for h in HYPOTHESES if h not in refits]
    if not missing:
        spread_params: dict[str, dict] = {}
        for name in names:
            values = {h: float(refits[h]["parameters"][name]) for h in HYPOTHESES}
            vals = np.asarray(list(values.values()), dtype=float)
            spread_params[name] = {
                "initial": float(initial[names.index(name)]),
                **values,
                "min": float(vals.min()),
                "max": float(vals.max()),
                "spread_max_minus_min": float(vals.max() - vals.min()),
                "std_ddof1": float(np.std(vals, ddof=1)) if vals.size >= 2 else 0.0,
            }
        report["parameter_spread"] = {
            "status": "complete",
            "basis": (
                "per-parameter range (max - min) and sample standard deviation "
                "(ddof = 1) across the per-hypothesis refits at the pre-registered "
                "reduced settings; this spread is the reported model uncertainty of "
                "the decay kinematics. At these reduced statistics the range "
                "includes the Monte-Carlo noise floor of the refit itself; the "
                "interpretation states which parameters move beyond it"
            ),
            "parameters": spread_params,
        }
    else:
        report["parameter_spread"] = {
            "status": "incomplete",
            "missing_hypotheses": missing,
            "note": (
                "the spread is computed once refits for all of H1, H2, H3 are "
                "present in the refits block"
            ),
        }
        print(
            f"[spread] incomplete: missing {missing}",
            flush=True,
        )

    report["refits"] = refits
    report["wall_time_total_s"] = time.perf_counter() - t_start
    report["finished_utc"] = _utc_now_iso()
    _write_json(artifact, report)
    print(f"[done] artifact: {artifact}")
    print(
        f"[done] total wall {report['wall_time_total_s'] / 60.0:.1f} min, "
        f"seed {seed}, {len(arms)} arms, {len(refits)} refits"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
