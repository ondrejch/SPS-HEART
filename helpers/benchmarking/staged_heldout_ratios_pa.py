#!/usr/bin/env python3
"""Held-out ratios and p+A trends helper (card TASK-20260910-heldout-ratios-pa-trends, M12).

Pre-registration-time machinery for the two M12 workstreams at the frozen
M8 calibration parameter set:

- Workstream A (phase P2): p+p particle ratios and seagull <pT>(xF)
  profiles in matched phase-space windows against the NA49 p+p 158 GeV/c
  tables of spshadrons.web.cern.ch/data/.
- Workstream B (phase P3): minimum-bias p+A trends (p+Pb208, p+C12 at
  158 GeV/c): per-event pion densities, invariant cross sections with the
  measured sigma_inel anchors, the wounded-scaling diagnostic R(xF), and
  the A-scaling trend.

Phase P1 fixes the protocol before any production event generation. This
module carries the pure table arithmetic the pre-registration records
(common-support grids, per-ratio window supports, the data-side R(xF)
recomputation from the pT-integrated tables, the data-data convention
cross-checks), the seed-block derivation with the no-collision statement
verified against the M11 pre-registration, the pre-registered probes
(determinism, per-event cost cells, gateway capacity, read-only), and the
harness cross-check command that verifies the frozen M8 parameter set
against the M8 record by sha256 and the seed block against the M11
pre-registration. The harness refuses to run on any mismatch: the bands,
the seeds, the windows, the kernels, and the statistics menu are fixed
before the events exist and are never tuned after a run (spec section 7.3
falsifier policy). A recorded fail is a legitimate result.

Phase P2 (workstream A production) adds the production path on top of the
P1 machinery, unchanged in protocol: the cells manifest (the pre-registered
per-window and seagull cell sets as pure table arithmetic, written before
any generation and consumed byte-identically by the per-replica runner),
the per-replica production runner (10000 events per replica at the
workstream-A seeds, event.validate() re-asserted per event at 1e-10, the
spec 7.1 source-quantum-number and antibaryon-traceability re-assertions,
first-match binning onto the manifest cells, per-event kernel
contributions for the pooled bootstrap, and the symmetry-grid counts for
the spec 7.1 test-6 re-assertion), the same-environment double-generation
determinism re-assertion at production scale, and the collection step that
recomputes the data-side statistics from the tables (refusing any mismatch
against the pre-registered values), evaluates the pre-registered verdict
bands, and writes ratios_report.json.

Data conventions (pinned here and in the pre-registration):

- The ratio yield kernel is Y_h proportional to
  sum_i (pT_i/E_i) f_h(cell_i) dxF_i dpT_i over the common-support cells
  of a ratio (the kernel of the paper detailed review section 2). The
  per-cell dxF dpT areas are shared by numerator and denominator of a
  ratio and cancel; the species-dependent E does not cancel.
- The seagull kernel is identical on the model and the data sides: the
  <pT>(xF) of the pT-integrated tables is the pT/E-weighted
  invariant-density mean with E = sqrt(m^2 + pL^2 + pT^2) and
  pL = xF sqrt(s)/2; the helper-suite test reproduces this closed form on
  a synthetic sample.
- p+Pb (thesis definitions, Chvala 2006): d(xF,pT) =
  E/(pi sqrt(s) pT) d^2n/dxF dpT per event and f(xF,pT) = sigma_inel * d
  with the measured sigma_inel(p+Pb) = 1806 +/- 25 mb; p+C uses the
  measured sigma_inel(p+C) = 226 +/- 7 mb.
- Frame (card TASK-20261005-pa-frame-nn-cm-xf, 2026-10-06): x_F is
  2 p_z / sqrt(s_NN) in the nucleon-nucleon CM frame on both systems.
  The p+A builder emits laboratory-frame events and records the frame,
  so particle_xf boosts the p+A tracks to the NN CM (beta = p_beam /
  (E_beam + m_p), gamma = (E_beam + m_p) / sqrt(s_NN)); the p+p events
  are CM events and keep their x_F bit for bit. The sealed P3 record of
  2026-09-10 divided the lab-frame p_z by sqrt(s_NN) and is withdrawn.

Layout: this file is byte-identical in the monorepo helper tree
(helpers/benchmarking/) and the vendored tree
(dev/mixing2006/helpers/benchmarking/); the tree is detected from
__file__, so the mixing2006 package and the tree-local run-artifact root
resolve correctly on both sides (monorepo: dev/00runs; vendored: 00runs).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# --- Layout (byte-identical in both helper trees) ---------------------------

_BENCH_DIR = Path(__file__).resolve().parent
if (_BENCH_DIR.parents[1] / "na49_mixing").is_dir():
    # vendored tree: dev/mixing2006/helpers/benchmarking/this_file.py
    MIXING2006_DIR = _BENCH_DIR.parents[1]
    RUNS_ROOT = MIXING2006_DIR / "00runs"
    REPO_ROOT = MIXING2006_DIR.parent
else:
    # monorepo tree: helpers/benchmarking/this_file.py
    REPO_ROOT = _BENCH_DIR.parents[1]
    MIXING2006_DIR = REPO_ROOT / "dev" / "mixing2006"
    RUNS_ROOT = REPO_ROOT / "dev" / "00runs"

# --- Pre-registered protocol constants (mirrored in the pre-registration;
#     the harness cross-checks the file against these and refuses a mismatch) ---

ARTIFACT_SCHEMA_VERSION = 1
CARD = "TASK-20260910-heldout-ratios-pa-trends"
MILESTONE = "M12"
PHASE = "P1"
RUN_DESCRIPTION = "staged_heldout_ratios_pa_2026-09-10"
STAGE_LABEL_RATIOS = "heldout_ratios"
STAGE_LABEL_PA = "heldout_pa_trends"
STAGE_LABEL_PPREF = "heldout_pa_trends_pp_reference"
PRODUCTION_MODE = "regge"
BEAM_MOMENTUM_GEV_C = 158.0
SQRT_S_GEV = 17.270205201372406  # sqrt(s) of Reaction(158.0, 2212, 2212); the NN sqrt(s)

# Frozen M8 calibration set (source: the baryon_nucleus.frozen_pp_parameters
# block of the M8 evaluation report; decay hypothesis H3 as the dataclass default).
M8_RECORD_RELPATH = "staged_binned_fit_mc_2026-09-06/staged_evaluation_report.json"
M8_RECORD_SHA256 = "ecd931d58ce79b9aecdd273c703b4421fe0ca9119cdde90d2ad3c7dfb9d49b65"
M8_FROZEN_BLOCK_SHA256 = "5a46a9ad8b4845791236d1bde33ea1fcb611f9060b3126374dbb1d59010f47f3"
M8_PARAMETER_SET = {
    "regge.alpha_exc0": 1.1418864566363858,
    "regge.effective_transfer_slope": 8.504140762083612,
    "regge.edge_power": 2.3769493784466396,
    "continuum.match_mass": 2.680194110727528,
    "continuum.pion_multiplicity_slope": 1.5226475471033067,
    "continuum.strange_lambda_strength": 0.30184606498191385,
    "continuum.strange_sigma_strength": 0.3777282734799746,
    "continuum.strange_pair_strength": 0.19187855210738686,
    "continuum.pair_strength": 0.08411858717003319,
    "continuum.pair_threshold_power": 1.7610429093564994,
}
M8_FROZEN_PP_PARAMETERS = {
    "regge": {
        "alpha_exc0": 1.1418864566363858,
        "alpha_p0": 1.0808,
        "alpha_p_prime": 0.25,
        "edge_power": 2.3769493784466396,
        "effective_transfer_slope": 8.504140762083612,
        "grid_size": 96,
        "s0": 1.0,
        "transfer_slope": 4.5,
    },
    "continuum": {
        "decay_hypothesis": "H3",
        "match_mass": 2.680194110727528,
        "max_pions": 12,
        "pair_strength": 0.08411858717003319,
        "pair_threshold_power": 1.7610429093564994,
        "pion_multiplicity_slope": 1.5226475471033067,
        "strange_lambda_strength": 0.30184606498191385,
        "strange_pair_strength": 0.19187855210738686,
        "strange_sigma_strength": 0.3777282734799746,
        "strange_threshold_power": 1.5,
    },
}

# Geometry (fixed, no tuning). C12 is the reference construction of the
# glauber test suite: harmonic oscillator with length a = rms/sqrt(3/2)
# ~ 1.88 fm (point-nucleon rms ~ 2.3 fm; T(0) ~ 1.1 nucleons/fm^2,
# sigma_pA ~ 0.20 b), quad_alpha 0.0, sigma_NN 31.7 mb.
GEOMETRY_PB208 = {
    "name": "Pb208",
    "A": 208,
    "Z": 82,
    "profile": "woods_saxon",
    "radius_fm": 6.62,
    "diffuseness_fm": 0.546,
    "quad_alpha": 0.0,
    "sigma_nn_inel_mb": 31.7,
}
GEOMETRY_C12 = {
    "name": "C12",
    "A": 12,
    "Z": 6,
    "profile": "harmonic_oscillator",
    "radius_fm": 1.88,
    "diffuseness_fm": 0.546,
    "quad_alpha": 0.0,
    "sigma_nn_inel_mb": 31.7,
}
PA_RUNNER_SETTINGS = {"glauber_dr_fm": 0.05, "max_event_attempts": 100}

# Normalization anchors (measured).
SIGMA_INEL_PP_MB = 31.78
SIGMA_INEL_PPb_MB = 1806.0
SIGMA_INEL_PPb_ERR_MB = 25.0
SIGMA_INEL_PC_MB = 226.0
SIGMA_INEL_PC_ERR_MB = 7.0
NU_PPB_MEASURED = 3.65  # measured <nu> (p+Pb); the anchor-deviation systematic
NU_PC_MEASURED = 1.69  # measured <nu> (p+C)

# --- Seed protocol (shared block; M12 base = config seed + 80000) ------------

CONFIG_SEED = 20260831
M12_SEED_BASE = 20340831  # config seed + 80000 (M12)
NEXT_MILESTONE_OFFSET_SEED = 20350831  # config seed + 90000 (untouched)
M12_BLOCK_TOP = 20350830  # the highest seed any M12 sub-stream may use
MIN_SUBBLOCK_STRIDE = 1000  # the M10 stride rule (anchor + 8 replicas + spare)
M11_PREREG_RELPATH = "parameter_profiles_2026-09-07/pre_registration.json"
M11_PREREG_SHA256 = "d2e206c9bc5b7cf66ca64fbb6358dd739005585968faeb1deb149dc16788b57a"
M11_SEED_BASE = 20330831
M11_CONSUMED_TOP = 20330839
PILOT_SEED_ANCHOR = 20339831  # offset -1000; yield-counting only
PROBE_SEED_PP_DETERMINISM = 20339831  # pilot anchor
PROBE_SEED_PA_DETERMINISM = 20339832  # pilot anchor + 1
PROBE_SEED_PP_COST = 20339833  # pilot anchor + 2
PROBE_SEED_PPB_COST = 20339834  # pilot anchor + 3
PROBE_SEED_PC_COST = 20339835  # pilot anchor + 5
WS_A_PRODUCTION_SEED = 20340831
WS_A_REPLICA_SEEDS = tuple(WS_A_PRODUCTION_SEED + 1 + k for k in range(8))
WS_A_SHUFFLE_SEED_BASE = 20340841  # realizations r = 0..15 use base + r
WS_A_SHUFFLE_REALIZATIONS = 16
WS_A_BOOTSTRAP_SEED = 20342831  # base + 2000; the M9 bootstrap convention
WS_B_PRODUCTION_SEED = 20343831  # base + 3000
WS_B_REPLICA_SEEDS = tuple(WS_B_PRODUCTION_SEED + 1 + k for k in range(8))
WS_B_BOOTSTRAP_SEED = 20345831  # base + 5000
WS_B_PPREF_PRODUCTION_SEED = 20346831  # derived: WS-B bootstrap seed + 1000
WS_B_PPREF_REPLICA_SEEDS = tuple(WS_B_PPREF_PRODUCTION_SEED + 1 + k for k in range(8))
BOOTSTRAP_REPLICATES = 200  # the M9 convention

# --- Statistics menu, budget, verdict bands -----------------------------------

K_REPLICAS = 8
STATISTICS_MENU_EVENTS_PER_REPLICA = (10000, 5000, 2500, 1000)
PRIMARY_EVENTS_PER_REPLICA = 10000
MAX_PROJECTED_WALL_S = 18000.0  # per system, serial-equivalent (the M10 budget)
DETERMINISM_REPEATS = 5
COST_CELL_EVENTS = 300
DETERMINISM_PA_EVENTS = 50

# Verdict bands: fixed here, before any run (the M9/M10 pattern: bands fixed
# before the number is looked at). The combined 1-sigma uncertainty is the
# quadrature of the data error from the table and the MC error from the
# replica spread.
K_PASS = 2.0
K_TENSION = 5.0

# --- Observable windows (workstream A) -----------------------------------------

IN_WINDOW_XF_MAX = 0.4
IN_WINDOW_PT_MAX = 1.5
WINDOW_CLASSES = ("in_window", "forward_xF_gt_0.4", "high_pT_gt_1.5",
                  "negative_xF_out_of_windows")
PREDICTIVE_CONTENT = {
    "in_window": "in_window_reduced_content",
    "forward_xF_gt_0.4": "out_of_window_extrapolation",
    "high_pT_gt_1.5": "out_of_window_extrapolation",
    "negative_xF_out_of_windows": "outside_registered_windows",
}
RATIO_NUMERATORS_DENOMINATORS = (
    ("pi-", "pi+"),
    ("p", "pi-"),
    ("p", "pi+"),
    ("k+", "k-"),
    ("p", "pbar"),
)
WINDOW_DEFINITIONS_LITERAL = {
    "in_window_xf_max": IN_WINDOW_XF_MAX,
    "in_window_pT_max": IN_WINDOW_PT_MAX,
    "ratio_numerator_denominator_pairs": [
        list(pair) for pair in RATIO_NUMERATORS_DENOMINATORS],
    "predictive_content": {
        "in_window": "in_window_reduced_content",
        "forward_xF_gt_0.4": "out_of_window_extrapolation",
        "high_pT_gt_1.5": "out_of_window_extrapolation",
        "negative_xF_out_of_windows": "outside_registered_windows",
    },
}
PA_ANCHORS_LITERAL = {
    "sigma_inel_pPb_mb": SIGMA_INEL_PPb_MB,
    "sigma_inel_pPb_err_mb": SIGMA_INEL_PPb_ERR_MB,
    "sigma_inel_pC_mb": SIGMA_INEL_PC_MB,
    "sigma_inel_pC_err_mb": SIGMA_INEL_PC_ERR_MB,
    "nu_pPb_measured": NU_PPB_MEASURED,
    "nu_pC_measured": NU_PC_MEASURED,
    "R_xF_definition": (
        "R(xF) = (dn/dxF)_pA / ((1+<K>_sampler)/2 (dn/dxF)_pp); the "
        "measured-anchor deviation is stated as the systematic"),
    "R_xF_anchor_deviation": "uniform rescaling (1+<nu>)/(1+<K>_sampler)",
}
SPECIES_MASS_GEV = {
    "pi-": 0.13957039,
    "pi+": 0.13957039,
    "k+": 0.493677,
    "k-": 0.493677,
    "p": 0.93827231,
    "pbar": 0.93827231,
}

# Data file layout.
PP_TABLE_NAMES = {
    "pi-": "xsect_piminus.txt",
    "pi+": "xsect_piplus.txt",
    "k+": "xsect_kaplus_pp.txt",
    "k-": "xsect_kaminus_pp.txt",
    "p": "xsect_proton_pp.txt",
    "pbar": "xsect_antiproton_pp.txt",
}
PP_PTINTEGRATED_NAMES = {
    "pi-": "xsect_ptintegrated_piminus.txt",
    "pi+": "xsect_ptintegrated_piplus.txt",
    "k+": "xsect_ptintegrated_kaplus_pp.txt",
    "k-": "xsect_ptintegrated_kaminus_pp.txt",
    "p": "xsect_ptintegrated_proton_pp.txt",
    "pbar": "xsect_ptintegrated_antiproton_pp.txt",
}
PC_XSECT_NAMES = {
    "pi-": "xsect_piminus_pC.txt",
    "pi+": "xsect_piplus_pC.txt",
    "p": "xsect_proton_pC.txt",
    "pbar": "xsect_antiproton_pC.txt",
}
PC_PTINTEGRATED_NAMES = {
    "pi-": "xsect_ptintegrated_piminus_pC.txt",
    "pi+": "xsect_ptintegrated_piplus_pC.txt",
    "p": "xsect_ptintegrated_proton_pC.txt",
    "pbar": "xsect_ptintegrated_antiproton_pC.txt",
}
PA_DATA_DIRNAME = "pa_data"
PPB_DENSITY_NAMES = {
    "pi-": "pPb_dens_minbias_piminus.txt",
    "pi+": "pPb_dens_minbias_piplus.txt",
}
PPB_XSECT_NAMES = {
    "pi-": "xsect_piminus_pPb.txt",
    "pi+": "xsect_piplus_pPb.txt",
}
PPB_PTINTEGRATED_NAMES = {
    "pi-": "ptintegrated_piminus_pPb.txt",
    "pi+": "ptintegrated_piplus_pPb.txt",
}


# --- Small utilities ---------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_head(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(path),
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_json_sha256(obj) -> str:
    canon = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_jsonable(payload), indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def m8_record_path() -> Path:
    return RUNS_ROOT / M8_RECORD_RELPATH


def m11_preregistration_path() -> Path:
    return RUNS_ROOT / M11_PREREG_RELPATH


def spshadrons_data_dir() -> Path:
    """The NA49 mirror tables (external data; present in full checkouts)."""
    return REPO_ROOT / "spshadrons.web.cern.ch" / "data"


def pa_data_dir() -> Path:
    return _BENCH_DIR / PA_DATA_DIRNAME


# --- Table readers (pure reads; no generation) --------------------------------


def read_points(path: Path) -> np.ndarray:
    """Four-column (xF, pT, value, abs_error) spectra table."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        rows.append([float(x) for x in s.split()[:4]])
    return np.asarray(rows, dtype=float)


def read_ptintegrated(path: Path) -> np.ndarray:
    """Nine-column pT-integrated table (xF F err dN/dxF err <pT> err <pT^2> err).

    The table header pins 'Errors are always given in percent', so the error
    columns are percent values; the seagull comparison converts the <pT>
    error to an absolute uncertainty by multiplying the published value.
    The ninth column (the <pT^2> error) is carried additively; the P1
    readers consumed the first eight columns and every existing column
    index is unchanged.
    """
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        rows.append([float(x) for x in s.split()[:9]])
    return np.asarray(rows, dtype=float)


def read_pa_ptintegrated(path: Path) -> np.ndarray:
    """Seven-column p+A pT-integrated table (xF dn/dxF err <pT> err <pT^2> err)."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        rows.append([float(x) for x in s.split()[:7]])
    return np.asarray(rows, dtype=float)


def fine_pt_labels(xf: np.ndarray, pt: np.ndarray) -> np.ndarray:
    """Map consecutive duplicated pT labels onto the fine grid (pa_data
    anomaly 1): a run of equal labels at one xF steps 0.05 downward from the
    first occurrence (0.10, 0.10 -> 0.10, 0.05)."""
    xf = np.asarray(xf, dtype=float)
    pt_out = np.asarray(pt, dtype=float).copy()
    i = 1
    while i < len(xf):
        same_x = math.isclose(xf[i], xf[i - 1], rel_tol=0.0, abs_tol=1e-9)
        same_pt = math.isclose(pt_out[i], pt_out[i - 1], rel_tol=0.0, abs_tol=1e-9)
        if same_x and same_pt:
            j = i
            while (j + 1 < len(xf)
                   and math.isclose(xf[j + 1], xf[i - 1], rel_tol=0.0, abs_tol=1e-9)
                   and math.isclose(pt_out[j + 1], pt_out[i - 1], rel_tol=0.0, abs_tol=1e-9)):
                j += 1
            base = float(pt_out[i - 1])
            for q in range(i, j + 1):
                pt_out[q] = base - 0.05 * (j - q + 1)
            i = j + 1
        else:
            i += 1
    return pt_out


def common_support(table_a: np.ndarray, table_b: np.ndarray) -> list[tuple[float, float]]:
    """The exact (xF, pT) grid points both tables carry (pure table arithmetic)."""
    keys_a = {(round(float(r[0]), 4), round(float(r[1]), 4)) for r in table_a}
    keys_b = {(round(float(r[0]), 4), round(float(r[1]), 4)) for r in table_b}
    return sorted(keys_a & keys_b)


def window_class(xf: float, pt: float) -> str:
    """The per-cell predictive-content class (card scoping decision 3): the
    in-window class is the M8 calibration window xF in [0, 0.4] and
    pT <= 1.5 GeV/c with reduced predictive content; xF > 0.4 and pT > 1.5
    are the out-of-window extrapolation classes; xF < 0 lies outside every
    registered window."""
    if xf < -1e-9:
        return "negative_xF_out_of_windows"
    if xf > IN_WINDOW_XF_MAX + 1e-9:
        return "forward_xF_gt_0.4"
    if pt > IN_WINDOW_PT_MAX + 1e-9:
        return "high_pT_gt_1.5"
    return "in_window"


def window_support(tables: dict, numerator: str, denominator: str) -> dict:
    """Per-window common support of one ratio: counts, extents, and the cells
    themselves (pure table arithmetic; no generation)."""
    cells = common_support(tables[numerator], tables[denominator])
    per_window: dict[str, list[tuple[float, float]]] = {name: [] for name in WINDOW_CLASSES}
    for xf, pt in cells:
        per_window[window_class(xf, pt)].append((xf, pt))
    out: dict = {"total_common_cells": len(cells)}
    for name, win_cells in per_window.items():
        entry: dict = {"n_cells": len(win_cells)}
        if win_cells:
            xs = [x for x, _ in win_cells]
            pts = [p for _, p in win_cells]
            entry.update({
                "xF_min": min(xs),
                "xF_max": max(xs),
                "pT_min": min(pts),
                "pT_max": max(pts),
                "cells": [[x, p] for x, p in win_cells],
            })
        out[name] = entry
    return out


def kernel_weight(pt: np.ndarray, mass_gev: float, xf: float) -> np.ndarray:
    """The yield-kernel factor (pT/E) with E = sqrt(m^2 + pL^2 + pT^2) and
    pL = xF sqrt(s)/2 (the paper detailed review section 2 kernel). The
    per-cell dxF dpT areas cancel in a ratio and are not carried here."""
    pt = np.asarray(pt, dtype=float)
    p_l = float(xf) * SQRT_S_GEV / 2.0
    energy = np.sqrt(mass_gev**2 + p_l**2 + pt**2)
    return pt / energy


def window_yield(table: np.ndarray, cells: list, mass_gev: float) -> float:
    """Y_h proportional to the sum over the common cells of
    (pT/E) f dxF dpT - the pre-registered yield-kernel definition with the
    per-cell areas (the standard bin widths of the pinned binning rule).
    DEV-3 (P2): the P1 form omitted the dxF dpT areas; the pre-registered
    definition carries them and the species-dependent kernel factor does
    the physics, so both sides consume the with-areas form."""
    index = {}
    for row in table:
        index[(round(float(row[0]), 4), round(float(row[1]), 4))] = float(row[2])
    total = 0.0
    for xf, pt in cells:
        f = index[(round(float(xf), 4), round(float(pt), 4))]
        area = 2.0 * cell_xf_half_width(xf) * 2.0 * cell_pt_half_width(pt)
        total += (area
                  * float(kernel_weight(np.asarray([pt]), mass_gev, xf)[0])
                  * f)
    return total


def ratio_window_statistics(table_num: np.ndarray, table_den: np.ndarray,
                            numerator: str, denominator: str) -> dict:
    """Data-side window statistics for one ratio: per-window support,
    predictive-content class, kernel yields, and the data ratio."""
    mass_num = SPECIES_MASS_GEV[numerator]
    mass_den = SPECIES_MASS_GEV[denominator]
    tables = {numerator: table_num, denominator: table_den}
    sup = window_support(tables, numerator, denominator)
    out = {
        "numerator": numerator,
        "denominator": denominator,
        "total_common_cells": sup["total_common_cells"],
        "windows": {},
    }
    for name in WINDOW_CLASSES:
        win = sup[name]
        if not win["n_cells"]:
            out["windows"][name] = {
                "n_cells": 0,
                "predictive_content": PREDICTIVE_CONTENT[name],
                "status": "dropped_no_common_support",
            }
            continue
        cells = [(c[0], c[1]) for c in win["cells"]]
        y_num = window_yield(table_num, cells, mass_num)
        y_den = window_yield(table_den, cells, mass_den)
        out["windows"][name] = {
            "n_cells": win["n_cells"],
            "predictive_content": PREDICTIVE_CONTENT[name],
            "xF_min": win["xF_min"], "xF_max": win["xF_max"],
            "pT_min": win["pT_min"], "pT_max": win["pT_max"],
            "Y_numerator": y_num,
            "Y_denominator": y_den,
            "ratio_data": (y_num / y_den) if y_den else None,
            "cells": [[float(x), float(p)] for x, p in cells],
            "status": "registered",
        }
    return out


def standard_dp(pt_node: float) -> float:
    """The standard pT bin width of the NA49 grids: 0.05 GeV/c below pT 0.3,
    0.10 GeV/c above (the fine and coarse grids share this spacing rule)."""
    return 0.05 if pt_node <= 0.30 + 1e-9 else 0.10


def seagull_mean_pt(pt_values, density_values, mass_gev: float, xf: float,
                    dp_values=None) -> float:
    """The pinned seagull arithmetic on one xF column: the pT/E-weighted
    invariant-density mean, E = sqrt(m^2 + pL^2 + pT^2), pL = xF sqrt(s)/2,
    with the pT bin widths (standard_dp when not given). Identical on the
    model and the data sides; the helper-suite synthetic-sample test
    reproduces this closed form exactly."""
    pt = np.asarray(pt_values, dtype=float)
    f = np.asarray(density_values, dtype=float)
    dp = (np.asarray(dp_values, dtype=float) if dp_values is not None
          else np.array([standard_dp(p) for p in pt]))
    w = f * kernel_weight(pt, mass_gev, xf) * dp
    den = float(np.sum(w))
    if den <= 0.0 or not math.isfinite(den):
        return float("nan")
    return float(np.sum(pt * w) / den)


def seagull_count_mean_pt(cells, counts) -> float:
    """The identical-kernel seagull value in count space: the plain
    count-weighted pT mean over one column's cells.

    The pinned kernel form multiplies the per-cell invariant density (the
    thesis density with its E/(pi sqrt(s) pT) Jacobian) by (pT/E) and the
    pT bin width; the Jacobian cancels both factors, so the identical
    estimator applied to raw cell counts is sum(n pT) / sum(n). Pinned
    algebraically against the f-space form (which reproduces the published
    ptintegrated <pT> at the percent level) on the real tables.
    """
    counts = np.asarray(counts, dtype=float)
    pts = np.asarray([float(c[1]) for c in cells], dtype=float)
    total = float(np.sum(counts))
    if total <= 0.0 or not math.isfinite(total):
        return float("nan")
    return float(np.sum(pts * counts) / total)


def data_side_seagull_crosscheck(tables_2d: dict, ptintegrated: dict,
                                 species: str, xf: float) -> dict:
    """Data-data convention cross-check: the 2D-table kernel mean against the
    published ptintegrated <pT> at one xF (pions reproduce at the percent
    level; the kaon, proton, and antiproton tables carry coarser pT grids and
    their deviation is the recorded reconstruction limitation)."""
    t = tables_2d[species]
    ti = ptintegrated[species]
    sel = np.isclose(t[:, 0], xf, atol=1e-6) & (t[:, 1] <= IN_WINDOW_PT_MAX + 1e-9)
    pts, fs = t[sel, 1], t[sel, 2]
    calc = (seagull_mean_pt(pts, fs, SPECIES_MASS_GEV[species], xf)
            if pts.size else float("nan"))
    ref_row = ti[np.isclose(ti[:, 0], xf, atol=1e-6)]
    ref = float(ref_row[0, 5]) if len(ref_row) else float("nan")
    rel = (calc - ref) / ref if (ref == ref and ref != 0.0) else None
    return {
        "species": species,
        "xF": float(xf),
        "n_cells": int(sel.sum()),
        "kernel_mean_pt": calc,
        "table_mean_pt": ref,
        "rel_deviation": rel,
    }


def load_pp_tables() -> tuple[dict, dict]:
    """The six p+p spectra tables and the six pT-integrated tables."""
    base = spshadrons_data_dir()
    spectra = {sp: read_points(base / name) for sp, name in PP_TABLE_NAMES.items()}
    integrated = {sp: read_ptintegrated(base / name)
                  for sp, name in PP_PTINTEGRATED_NAMES.items()}
    return spectra, integrated


def all_ratio_window_statistics() -> dict:
    """The pre-registered per-ratio, per-window common supports and data-side
    ratios (pure table arithmetic over all five ratio pairs)."""
    spectra, _integrated = load_pp_tables()
    out = {}
    for num, den in RATIO_NUMERATORS_DENOMINATORS:
        key = f"{num}_over_{den}"
        out[key] = ratio_window_statistics(spectra[num], spectra[den], num, den)
    return out


def dropped_windows_record() -> list:
    """Windows with no common data support on both sides, with the reason."""
    stats = all_ratio_window_statistics()
    dropped = []
    for key, ratio in stats.items():
        for win_name, win in ratio["windows"].items():
            if win.get("status") == "dropped_no_common_support":
                dropped.append({
                    "ratio": key,
                    "window": win_name,
                    "n_cells": 0,
                    "reason": "no common (xF, pT) data support on both sides "
                              "(pure table arithmetic)",
                })
    return dropped


# --- Data-side R(xF) recomputation (workstream B) -------------------------------


def data_side_r_xf(system: str, mean_target_wounds: float) -> dict:
    """R(xF) = (dn/dxF)_pA / ((1+<K>)/2 (dn/dxF)_pp) recomputed from the
    pT-integrated tables, with the sampler's own <K>; the measured-anchor
    deviation is recorded as the stated systematic."""
    if system == "pPb":
        names, reader, base, nu_measured = (
            PPB_PTINTEGRATED_NAMES, read_pa_ptintegrated, pa_data_dir(),
            NU_PPB_MEASURED)
    elif system == "pC":
        names, reader, base, nu_measured = (
            PC_PTINTEGRATED_NAMES, read_ptintegrated, spshadrons_data_dir(),
            NU_PC_MEASURED)
    else:
        raise ValueError(f"unsupported system {system!r}")
    # The dN/dxF column of the pA table: column 1 in the 7-column p+Pb
    # format, column 3 in the 8-column p+C format.
    dndx_f_col = 1 if system == "pPb" else 3
    err_col = 2 if system == "pPb" else 4
    n_w = (1.0 + float(mean_target_wounds)) / 2.0
    n_w_anchor = (1.0 + nu_measured) / 2.0
    out = {}
    for sp, name in names.items():
        pa_t = reader(base / name)
        pp_t = read_ptintegrated(spshadrons_data_dir() / PP_PTINTEGRATED_NAMES[sp])
        rows = []
        for r_pa in pa_t:
            xf = r_pa[0]
            match = np.isclose(pp_t[:, 0], xf, atol=1e-6)
            if not match.any():
                continue
            r_pp = pp_t[match][0]
            r_val = float(r_pa[dndx_f_col]) / (n_w * float(r_pp[3]))
            anchor = float(r_pa[dndx_f_col]) / (n_w_anchor * float(r_pp[3]))
            rows.append({
                "xF": float(xf),
                "dn_dxF_pa": float(r_pa[dndx_f_col]),
                "dn_dxF_err": float(r_pa[err_col]),
                "dn_dxF_pp": float(r_pp[3]),
                "R_samplerK": r_val,
                "R_measuredAnchor": anchor,
                "anchor_deviation": (r_val - anchor) / anchor if anchor else None,
            })
        out[sp] = rows
    return out


# --- Data-data convention cross-checks -----------------------------------------


def data_data_unit_checks() -> dict:
    """The pre-registration unit cross-checks (pure table reads).

    1. f = sigma_inel * d on the common physical p+Pb bins: the f table's
       duplicated pT labels are mapped to the fine grid and paired with the
       density table (which carries the fine grid natively); the per-cell
       scatter reflects the two different thesis source tables (tabicall
       versus tabicmb), not a normalization failure.
    2. dn/dxF from the density table through the thesis kernel
       d^2n/dxF dpT = (pi sqrt(s)/E) pT d, against the ptintegrated column.
    3. The <pT>(xF=0) cross-system check: p+p versus p+Pb pi-, re-derived.
    """
    dens = read_points(pa_data_dir() / PPB_DENSITY_NAMES["pi-"])
    ftab = read_points(pa_data_dir() / PPB_XSECT_NAMES["pi-"])
    ptint = read_pa_ptintegrated(pa_data_dir() / PPB_PTINTEGRATED_NAMES["pi-"])

    pt_dis = fine_pt_labels(ftab[:, 0].copy(), ftab[:, 1].copy())
    density_index = {
        (round(float(row[0]), 4), round(float(row[1]), 4)): float(row[2])
        for row in dens
    }
    ratios = []
    for i in range(len(ftab)):
        key = (round(float(ftab[i, 0]), 4), round(float(pt_dis[i]), 4))
        if key in density_index and ftab[i, 2] > 0:
            ratios.append(float(ftab[i, 2]) / (SIGMA_INEL_PPb_MB * density_index[key]))
    ratios_arr = np.asarray(ratios, dtype=float)

    per_xf = []
    for xf_t in sorted({round(float(x), 4) for x in dens[:, 0]}):
        col = dens[np.isclose(dens[:, 0], xf_t, atol=1e-6)]
        if not col.size:
            continue
        pts, ds = col[:, 1], col[:, 2]
        dp = np.full_like(pts, 0.05)
        dp[pts > 0.30] = 0.10
        energy = np.sqrt(
            SPECIES_MASS_GEV["pi-"] ** 2
            + (xf_t * SQRT_S_GEV / 2.0) ** 2
            + pts ** 2
        )
        val = float(np.sum((math.pi * SQRT_S_GEV / energy) * pts * ds * dp))
        ref = ptint[np.isclose(ptint[:, 0], xf_t, atol=1e-6)]
        per_xf.append({
            "xF": xf_t,
            "dn_dxF_from_density": val,
            "dn_dxF_ptintegrated": float(ref[0][1]) if len(ref) else None,
        })

    pp_ti = read_ptintegrated(spshadrons_data_dir() / PP_PTINTEGRATED_NAMES["pi-"])
    return {
        "f_over_sigma_times_d_pi-": {
            "n_cells": int(ratios_arr.size),
            "mean": float(np.mean(ratios_arr)),
            "std": float(np.std(ratios_arr)),
            "median": float(np.median(ratios_arr)),
        },
        "dn_dxF_cross_check_pi-": per_xf,
        "mean_pt_xF0_pp_pi-": float(pp_ti[np.isclose(pp_ti[:, 0], 0.0)][0][5]),
        "mean_pt_xF0_pPb_pi-": float(ptint[np.isclose(ptint[:, 0], 0.0)][0][3]),
        "convention_statement": (
            "thesis definitions (Chvala 2006, section 'Cross section "
            "evaluation and corrections'): d(xF,pT) = E/(pi sqrt(s) pT) "
            "d^2n/dxF dpT per event and f(xF,pT) = sigma_inel d(xF,pT); the "
            "density table carries the fine pT grid while the f table "
            "carries the coarse doubled-label grid"
        ),
    }


# --- Geometry anchors -----------------------------------------------------------


def geometry_anchors() -> dict:
    """sigma_pA and <K> from the sampler at the pinned geometry (the P3
    sanity anchors; the measured anchors carry the systematic statement)."""
    if str(MIXING2006_DIR) not in sys.path:
        sys.path.insert(0, str(MIXING2006_DIR))
    from na49_mixing.glauber import NucleusSpec, WoundedNucleonSampler

    out = {}
    for label, fields in (("Pb208", GEOMETRY_PB208), ("C12", GEOMETRY_C12)):
        spec = NucleusSpec(**fields)
        sampler = WoundedNucleonSampler(spec, np.random.default_rng(1), dr_fm=0.05)
        out[label] = {
            "geometry": dict(fields),
            "sigma_pA_mb": float(sampler.sigma_pA_mb()),
            "mean_target_wounds": float(sampler.mean_target_wounds_minbias()),
        }
    return out


# --- Seed-block derivation and the no-collision statement -----------------------


def derive_seed_block() -> dict:
    """The full M12 seed-block layout (pairwise disjoint, stride >= 1000)."""
    return {
        "config_seed": CONFIG_SEED,
        "m12_seed_base": M12_SEED_BASE,
        "m12_seed_base_derivation": (
            "config seed 20260831 + 80000 (M12, the next milestone offset of "
            "the established pattern M4 +10000, M5 +20000, M7 +30000, M8 "
            "+40000, M9 +50000, M10 +60000, M11 +70000)"),
        "pattern": (
            "established pattern base 20260831 + 10000 x milestone index; "
            "M4 +10000 (20270831), M5 +20000 (20280831), M7 +30000 "
            "(20290831), M8 replicas +40000 (20300831..20300838), M9 +50000 "
            "(top 20312831), M10 +60000 (top 20320889), M11 +70000 (top "
            "20330839)"),
        "block_span": [M12_SEED_BASE, M12_BLOCK_TOP],
        "next_milestone_offset_seed": NEXT_MILESTONE_OFFSET_SEED,
        "no_collision_statement": (
            "the block 20340831..20350830 sits strictly above the M11 "
            "consumed top 20330839 (verified against the M11 "
            "pre-registration seed_protocol block, file sha256 "
            + M11_PREREG_SHA256 + ") and strictly below the next milestone "
            "offset +90000 = 20350831; no other recorded stream occupies it"),
        "pilot_sub_block": {
            "pilot_seed_anchor": PILOT_SEED_ANCHOR,
            "pilot_top": PROBE_SEED_PC_COST,
            "role": (
                "yield-counting and probe cells only; excluded from "
                "production statistics (the M9 pilot convention)"),
            "sub_streams": {
                "determinism_probe_pp": PROBE_SEED_PP_DETERMINISM,
                "determinism_probe_pa": PROBE_SEED_PA_DETERMINISM,
                "cost_cell_pp": PROBE_SEED_PP_COST,
                "cost_cell_pa_pPb208": PROBE_SEED_PPB_COST,
                "cost_cell_pa_C12": PROBE_SEED_PC_COST,
            },
            "derivation_note": (
                "all probe sub-streams derive from the pilot anchor "
                "20339831 by +1 steps (pilot anchor .. +5); the pilot block "
                "sits 1000 below the M12 production base"),
        },
        "workstream_A": {
            "production_seed": WS_A_PRODUCTION_SEED,
            "replica_rule": (
                "replica k (k = 0..7) uses seed = production seed + 1 + k "
                "(the M8 replica convention)"),
            "replica_seeds": list(WS_A_REPLICA_SEEDS),
            "shuffle_seed_base": WS_A_SHUFFLE_SEED_BASE,
            "shuffle_rule": (
                "shuffle realization r (r = 0..15) uses seed = "
                "shuffle_seed_base + r, if the shuffle baseline is used"),
            "bootstrap_seed": WS_A_BOOTSTRAP_SEED,
            "bootstrap_rule": (
                "one generator consumed sequentially; 200 resamples (the M9 "
                "convention at the +2000 seed offset)"),
        },
        "workstream_B": {
            "production_seed": WS_B_PRODUCTION_SEED,
            "replica_rule": (
                "replica k (k = 0..7) uses seed = production seed + 1 + k"),
            "replica_seeds": list(WS_B_REPLICA_SEEDS),
            "bootstrap_seed": WS_B_BOOTSTRAP_SEED,
            "bootstrap_rule": "one generator consumed sequentially; 200 resamples",
            "pp_reference_production_seed": WS_B_PPREF_PRODUCTION_SEED,
            "pp_reference_derivation": (
                "WS-B bootstrap seed + 1000 (the stride rule); the p+p "
                "reference sample for R(xF) at the same replica convention, "
                "with common random numbers at the level the seed layout "
                "allows (replica k of every system carries the same "
                "replica-index offset +1+k from its anchor)"),
            "pp_reference_replica_seeds": list(WS_B_PPREF_REPLICA_SEEDS),
        },
        "stride_statement": (
            "all sub-blocks are pairwise disjoint with stride at least 1000 "
            "between block starts (the M10 stride rule: 1 anchor + 8 "
            "replicas + spare << 1000); every sub-stream stays below "
            + str(NEXT_MILESTONE_OFFSET_SEED)),
        "stage_labels": {
            "workstream_A": STAGE_LABEL_RATIOS,
            "workstream_B": STAGE_LABEL_PA,
            "workstream_B_pp_reference": STAGE_LABEL_PPREF,
        },
        "determinism": (
            "identical configuration and seeds reproduce identical events "
            "(spec 7.1 reproducibility test 5); the pre-registered probe "
            "verifies five bit-identical repeated evaluations at a fixed "
            "point"),
    }


def verify_no_collision() -> dict:
    """Verify the no-collision statement against the M11 pre-registration.

    Refuses (SystemExit) when the M11 file is missing, its sha256 does not
    match the recorded value, or its seed_protocol block does not place the
    M11 consumed top strictly below the M12 block.
    """
    path = m11_preregistration_path()
    if not path.exists():
        raise SystemExit(
            f"M11 pre-registration not found at {path}; the no-collision "
            "statement cannot be verified")
    actual_sha = _sha256_file(path)
    if actual_sha != M11_PREREG_SHA256:
        raise SystemExit(
            f"M11 pre-registration sha256 {actual_sha!r} != recorded "
            f"{M11_PREREG_SHA256!r}; refusing to derive the seed block")
    prereg = json.loads(path.read_text(encoding="utf-8"))
    seeds = prereg.get("seed_protocol") or {}
    problems = []
    if seeds.get("m11_seed_base") != M11_SEED_BASE:
        problems.append(
            f"m11_seed_base: M11 file {seeds.get('m11_seed_base')!r} != "
            f"harness {M11_SEED_BASE!r}")
    block = seeds.get("shared_replica_block")
    if not (isinstance(block, (list, tuple)) and len(block) == 2
            and int(block[1]) == M11_CONSUMED_TOP):
        problems.append(
            f"M11 replica block top: M11 file {block!r} != harness top "
            f"{M11_CONSUMED_TOP}")
    if problems:
        raise SystemExit(
            "seed-block verification failed; refusing to run:\n  "
            + "\n  ".join(problems))
    if not (M11_CONSUMED_TOP < M12_SEED_BASE):
        raise SystemExit("the M12 block does not sit strictly above the M11 top")
    if not (M12_BLOCK_TOP < NEXT_MILESTONE_OFFSET_SEED):
        raise SystemExit("the M12 block top reaches the next milestone offset")
    starts = sorted([
        PILOT_SEED_ANCHOR, WS_A_PRODUCTION_SEED, WS_A_BOOTSTRAP_SEED,
        WS_B_PRODUCTION_SEED, WS_B_BOOTSTRAP_SEED,
        WS_B_PPREF_PRODUCTION_SEED,
    ])
    strides = [b - a for a, b in zip(starts, starts[1:])]
    if min(strides) < MIN_SUBBLOCK_STRIDE:
        raise SystemExit(
            f"sub-block stride violation: {strides} (the M10 stride rule "
            f"requires >= {MIN_SUBBLOCK_STRIDE})")
    tops = [PILOT_SEED_ANCHOR + 4, WS_A_SHUFFLE_SEED_BASE + 15,
            WS_A_BOOTSTRAP_SEED, WS_B_PRODUCTION_SEED + 8,
            WS_B_BOOTSTRAP_SEED, WS_B_PPREF_PRODUCTION_SEED + 8]
    if max(tops) >= NEXT_MILESTONE_OFFSET_SEED:
        raise SystemExit("an M12 sub-stream reaches the next milestone offset")
    return {
        "verified": True,
        "m11_preregistration_path": str(path),
        "m11_preregistration_sha256": actual_sha,
        "m11_seed_base": M11_SEED_BASE,
        "m11_consumed_top": M11_CONSUMED_TOP,
        "m12_block": [M12_SEED_BASE, M12_BLOCK_TOP],
        "next_milestone_offset": NEXT_MILESTONE_OFFSET_SEED,
        "sub_block_starts": starts,
        "sub_block_strides": strides,
        "sub_block_tops": sorted(tops),
        "highest_consumed_m12_seed": max(tops),
    }


# --- Harness cross-checks ---------------------------------------------------------


def cross_check_m8_record(path: Path | None = None) -> dict:
    """Verify the frozen M8 parameter set against the M8 record by sha256.

    Recomputes (a) the sha256 of the M8 record file and (b) the sha256 of the
    canonical JSON of its baryon_nucleus.frozen_pp_parameters block, and
    compares both against the recorded values. The path defaults to the
    tree-local run root (overridable for the gateway execution environment).
    """
    path = Path(path) if path is not None else m8_record_path()
    if not path.exists():
        raise SystemExit(f"M8 record not found at {path}")
    problems: list[str] = []
    actual_file_sha = _sha256_file(path)
    if actual_file_sha != M8_RECORD_SHA256:
        problems.append(
            f"m8_record_file: sha256 {actual_file_sha!r} != harness "
            f"{M8_RECORD_SHA256!r}")
    block = json.loads(path.read_text(encoding="utf-8"))[
        "baryon_nucleus"]["frozen_pp_parameters"]
    actual_block_sha = canonical_json_sha256(block)
    if actual_block_sha != M8_FROZEN_BLOCK_SHA256:
        problems.append(
            f"frozen_pp_parameters block: canonical sha256 {actual_block_sha!r} "
            f"!= recorded {M8_FROZEN_BLOCK_SHA256!r}")
    if block != M8_FROZEN_PP_PARAMETERS:
        problems.append(
            "frozen_pp_parameters block values differ from the harness constants")
    if problems:
        raise SystemExit(
            "M8 frozen-set cross-check failed; refusing to run:\n  "
            + "\n  ".join(problems))
    return {
        "verified": True,
        "m8_record_path": str(path),
        "m8_record_sha256": actual_file_sha,
        "m8_frozen_block_sha256": actual_block_sha,
    }


def cross_check_preregistration(prereg: dict) -> None:
    """Refuse to run when the pre-registration and the constants disagree."""
    problems: list[str] = []

    def check(name: str, expected, actual) -> None:
        if expected != actual:
            problems.append(
                f"{name}: pre-registration {expected!r} != harness {actual!r}")

    check("card", prereg.get("card"), CARD)
    check("milestone", prereg.get("milestone"), MILESTONE)
    check("phase", prereg.get("phase"), PHASE)
    check("run_description", prereg.get("run_description"), RUN_DESCRIPTION)
    machine = prereg.get("machine_constants", {})
    check("production_mode", machine.get("production_mode"), PRODUCTION_MODE)
    check("beam_momentum_gev_c", machine.get("beam_momentum_gev_c"),
          BEAM_MOMENTUM_GEV_C)
    check("sqrt_s_gev", machine.get("sqrt_s_gev"), SQRT_S_GEV)
    decision = prereg.get("calibration_set_decision_of_record", {})
    check("m8_parameter_set", decision.get("m8_parameter_set"), M8_PARAMETER_SET)
    check("m8_frozen_pp_parameters", decision.get("m8_frozen_pp_parameters"),
          M8_FROZEN_PP_PARAMETERS)
    geometry = prereg.get("geometry", {})
    check("geometry_pb208", geometry.get("Pb208"), GEOMETRY_PB208)
    check("geometry_c12", geometry.get("C12"), GEOMETRY_C12)
    check("pa_runner_settings", geometry.get("pa_runner_settings"),
          PA_RUNNER_SETTINGS)
    anchors = prereg.get("normalization_anchors", {})
    check("sigma_inel_pp_mb", anchors.get("sigma_inel_pp_mb"), SIGMA_INEL_PP_MB)
    check("sigma_inel_pPb_mb", anchors.get("sigma_inel_pPb_mb"), SIGMA_INEL_PPb_MB)
    check("sigma_inel_pC_mb", anchors.get("sigma_inel_pC_mb"), SIGMA_INEL_PC_MB)
    check("nu_pPb_measured", anchors.get("nu_pPb_measured"), NU_PPB_MEASURED)
    check("nu_pC_measured", anchors.get("nu_pC_measured"), NU_PC_MEASURED)
    check("seed_protocol", prereg.get("seed_protocol"), derive_seed_block())
    stats = prereg.get("statistics", {})
    check("k_replicas", stats.get("k_replicas"), K_REPLICAS)
    check("menu_events_per_replica",
          stats.get("menu_events_per_replica"),
          list(STATISTICS_MENU_EVENTS_PER_REPLICA))
    check("primary_events_per_replica", stats.get("primary_events_per_replica"),
          PRIMARY_EVENTS_PER_REPLICA)
    check("max_projected_wall_s", stats.get("max_projected_wall_s"),
          MAX_PROJECTED_WALL_S)
    check("bootstrap_replicates", stats.get("bootstrap_replicates"),
          BOOTSTRAP_REPLICATES)
    check("determinism_repeats", stats.get("determinism_repeats"),
          DETERMINISM_REPEATS)
    check("cost_cell_events", stats.get("cost_cell_events"), COST_CELL_EVENTS)
    check("determinism_pa_events", stats.get("determinism_pa_events"),
          DETERMINISM_PA_EVENTS)
    bands = prereg.get("verdict_bands", {})
    check("k_pass", bands.get("k_pass"), K_PASS)
    check("k_tension", bands.get("k_tension"), K_TENSION)
    protocol_a = prereg.get("protocol_A") or {}
    windows = protocol_a.get("window_definitions") or {}
    check("protocol_A.window_definitions", windows, WINDOW_DEFINITIONS_LITERAL)
    check("protocol_A.ratio_pairs",
          windows.get("ratio_numerator_denominator_pairs"),
          WINDOW_DEFINITIONS_LITERAL["ratio_numerator_denominator_pairs"])
    check("protocol_A.predictive_content",
          windows.get("predictive_content"),
          WINDOW_DEFINITIONS_LITERAL["predictive_content"])
    protocol_b = prereg.get("protocol_B") or {}
    check("protocol_B.normalization_anchors",
          protocol_b.get("normalization_anchors"), PA_ANCHORS_LITERAL)
    if problems:
        raise SystemExit(
            "pre-registration mismatch; refusing to run (the pre-registered "
            "protocol is authoritative):\n  " + "\n  ".join(problems))


# --- Generation (the M7/M8 path; probe-scale use in P1) --------------------------


def build_parameter_set(config: dict):
    """Bundled-config parameter dataclasses with the ten M8 values applied."""
    from dataclasses import replace

    from na49_mixing.continuum_channels import ContinuumParameters
    from na49_mixing.exchange_model import (
        ReggeExcitationParameters,
        SimpleExcitationParameters,
    )

    def from_dict(cls, values):
        return cls(**(values or {}))

    simple = from_dict(SimpleExcitationParameters, config.get("simple"))
    regge = from_dict(ReggeExcitationParameters, config.get("regge"))
    continuum = from_dict(ContinuumParameters, config.get("continuum"))
    for name, value in M8_PARAMETER_SET.items():
        group, field_name = name.split(".", 1)
        if group == "regge":
            regge = replace(regge, **{field_name: float(value)})
        else:
            continuum = replace(continuum, **{field_name: float(value)})
    applied = {
        "regge": {k: float(v) for k, v in regge.__dict__.items()},
        "continuum": {
            k: (v if isinstance(v, str) else float(v))
            for k, v in continuum.__dict__.items()
        },
    }
    for name, value in M8_PARAMETER_SET.items():
        group, field_name = name.split(".", 1)
        if applied[group][field_name] != float(value):
            raise SystemExit(f"parameter application failed for {name}")
    if str(applied["continuum"]["decay_hypothesis"]).strip().upper() != "H3":
        raise SystemExit("the H3 decay kernel must apply as the dataclass default")
    return simple, regge, continuum, applied


def generate_pp_events(n_events: int, seed: int, config_path: Path,
                       stage_label: str = STAGE_LABEL_RATIOS):
    """The p+p sample via _generate_pp (regge mode, M8 parameters)."""
    if str(MIXING2006_DIR) not in sys.path:
        sys.path.insert(0, str(MIXING2006_DIR))
    from na49_mixing.staged_evaluation import _generate_pp

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    simple, regge, continuum, applied = build_parameter_set(config)
    beam = float(config.get("beam_momentum", BEAM_MOMENTUM_GEV_C))
    t0 = time.perf_counter()
    events, summary = _generate_pp(
        int(n_events), int(seed), simple, regge, continuum,
        PRODUCTION_MODE, beam, stage_label,
    )
    wall = time.perf_counter() - t0
    max_res = 0.0
    for event in events:
        event.validate()  # re-assert conservation at 1e-10 (the M7 convention)
        max_res = max(max_res, float(event.conservation_residuals()["relative_max"]))
        event.metadata["random_seed"] = int(seed)
        event.metadata["random_substream"] = f"{stage_label}:{event.event_id}"
    return events, summary, wall, applied, beam


def generate_pa_events(n_events: int, seed: int, config_path: Path,
                       nucleus_fields: dict, stage_label: str = STAGE_LABEL_PA):
    """The p+A sample via PAEventBuilder (the staged runner's baryon_nucleus
    path) with the frozen M8 parameters and the pinned nucleus fields —
    the identical GenerationConfig nucleus fields of the staged runner,
    consumed directly at the frozen set (no refit)."""
    if str(MIXING2006_DIR) not in sys.path:
        sys.path.insert(0, str(MIXING2006_DIR))
    from na49_mixing.event_builder_pA import PAEventBuilder, PAExcitationConfig
    from na49_mixing.glauber import NucleusSpec

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    simple, regge, continuum, applied = build_parameter_set(config)
    beam = float(config.get("beam_momentum", BEAM_MOMENTUM_GEV_C))
    pa_cfg = dict(config.get("pA") or PA_RUNNER_SETTINGS)
    nucleus = NucleusSpec(**nucleus_fields)
    builder = PAEventBuilder(
        PAExcitationConfig(
            beam_momentum=beam,
            production_mode=PRODUCTION_MODE,
            simple=simple,
            regge=regge,
            continuum=continuum,
            nucleus=nucleus,
            glauber_dr_fm=float(pa_cfg.get("glauber_dr_fm", 0.05)),
            max_event_attempts=int(pa_cfg.get("max_event_attempts", 100)),
        ),
        np.random.default_rng(int(seed)),
    )
    t0 = time.perf_counter()
    events = [builder.generate(index) for index in range(int(n_events))]
    wall = time.perf_counter() - t0
    max_res = 0.0
    for event in events:
        event.validate()
        max_res = max(max_res, float(event.conservation_residuals()["relative_max"]))
        event.metadata["random_seed"] = int(seed)
        event.metadata["random_substream"] = f"{stage_label}:{event.event_id}"
    summary = {
        "n_events": float(len(events)),
        "mean_target_wounds": float(np.mean([
            float(event.metadata.get("target_wounds", 1.0)) for event in events])),
        "mean_final_multiplicity": float(
            sum(len(event.final_particles) for event in events)
            / max(len(events), 1)),
        "max_conservation_residual": float(max_res),
        "wall_s": float(wall),
    }
    return events, summary, wall, applied, beam


# --- Model-side binning (pinned rule; probe-scale use in P1) ---------------------

SPECIES_PDG = {"pi-": -211, "pi+": 211, "k+": 321, "k-": -321,
               "p": 2212, "pbar": -2212}


_NN_CM_XF = None


def _nn_cm_xf_function():
    """The frame-aware x_F helper of the p+A builder (imported once)."""
    global _NN_CM_XF
    if _NN_CM_XF is None:
        if str(MIXING2006_DIR) not in sys.path:
            sys.path.insert(0, str(MIXING2006_DIR))
        from na49_mixing.event_builder_pA import nn_cm_xf
        _NN_CM_XF = nn_cm_xf
    return _NN_CM_XF


def particle_xf(particle, sqrt_s: float = SQRT_S_GEV, *, event) -> float:
    """Feynman x_F = 2 p_z^CM / sqrt(s_NN) of a final particle of ``event``.

    The p+A builder emits laboratory-frame events (target nucleons at
    rest) and records the frame and the nucleon-nucleon CM boost in the
    event metadata; their tracks are boosted to the NN CM before the
    division (na49_mixing.event_builder_pA.nn_cm_xf). The p+p builder's
    events are CM events and give 2 p_z / sqrt(s) bit for bit. The event
    is a required keyword so that no call site can drop the frame.
    """
    return _nn_cm_xf_function()(event, particle.p4, sqrt_s)


def particle_pt(particle) -> float:
    return math.hypot(float(particle.p4.px), float(particle.p4.py))


def cell_xf_half_width(xf_node: float) -> float:
    """xF 0.05-wide bins below xF 0.4, 0.10-wide above (the coarser forward
    data columns)."""
    return (0.05 if xf_node <= IN_WINDOW_XF_MAX + 1e-9 else 0.10) / 2.0


def cell_pt_half_width(pt_node: float) -> float:
    """pT bins of 0.05 below pT 0.3 and 0.10 above."""
    return (0.05 if pt_node <= 0.30 + 1e-9 else 0.10) / 2.0


def bin_species_counts(events, cells, pdg_id: int,
                       sqrt_s: float = SQRT_S_GEV) -> tuple:
    """Count final particles of one species on the given (xF, pT) cells with
    the pinned binning rule: a particle enters the first cell (in the sorted
    cell order) whose node lies within the cell half-widths in both
    variables; particles matching no cell are counted in the overflow.

    Returns (counts_array, overflow_count).
    """
    counts = np.zeros(len(cells), dtype=np.int64)
    overflow = 0
    for event in events:
        for particle in event.final_particles:
            if int(particle.pdg_id) != int(pdg_id):
                continue
            xf = particle_xf(particle, sqrt_s, event=event)
            pt = particle_pt(particle)
            placed = False
            for i, (xf_i, pt_j) in enumerate(cells):
                if (abs(xf - xf_i) <= cell_xf_half_width(xf_i) + 1e-9
                        and abs(pt - pt_j) <= cell_pt_half_width(pt_j) + 1e-9):
                    counts[i] += 1
                    placed = True
                    break
            if not placed:
                overflow += 1
    return counts, overflow


def model_window_yield(counts, cells, mass: float) -> float:
    """Y_h from the binned model counts: sum over the cells of n (pT/E),
    the same kernel the data side uses (the per-cell areas cancel in a ratio)."""
    total = 0.0
    for (xf, pt), n in zip(cells, counts):
        if n:
            total += n * float(kernel_weight(np.asarray([pt]), mass, xf)[0])
    return total


def pp_window_model_ratios(events) -> list:
    """Model-side values of the pre-registered ratio windows, from the event
    sample binned onto the data common-support cells (probe scale).

    The window set is the pre-registered verdict set derived from pure table
    arithmetic: windows with no common cells are skipped, the negative-xF
    windows are never registered (the pre-registration records that the
    negative-xF statement belongs to workstream B), and windows whose cells
    span a single pT label carry no pT lever arm for a verdict (the
    recorded k+/k- high-pT drop) - eleven values over the five ratios.
    """
    spectra, _integrated = load_pp_tables()
    values = []
    for num, den in RATIO_NUMERATORS_DENOMINATORS:
        stats = ratio_window_statistics(spectra[num], spectra[den], num, den)
        for win_name in WINDOW_CLASSES:
            if win_name == "negative_xF_out_of_windows":
                continue
            win = stats["windows"][win_name]
            if win.get("status") != "registered" or not win.get("cells"):
                continue
            cells = [(c[0], c[1]) for c in win["cells"]]
            if len({round(p, 4) for _x, p in cells}) < 2:
                continue  # no pT lever arm (the recorded k+/k- high-pT drop)
            counts_num, _ov = bin_species_counts(events, cells, SPECIES_PDG[num])
            counts_den, _ov2 = bin_species_counts(events, cells,
                                                  SPECIES_PDG[den])
            y_num = model_window_yield(counts_num, cells,
                                       SPECIES_MASS_GEV[num])
            y_den = model_window_yield(counts_den, cells,
                                       SPECIES_MASS_GEV[den])
            values.append(y_num / y_den if y_den else float("nan"))
    return values


def pp_seagull_model_values(events) -> list:
    """Model-side <pT>(xF=0) per species from the binned counts at the xF=0
    column of each species' common p+p support.

    The pinned seagull kernel is the pT/E-weighted invariant-density mean,
    identical on the model and the data sides. The pinned kernel form
    multiplies the per-cell invariant density (the table's f value) by
    (pT/E) and the pT bin width; because the invariant density carries the
    thesis Jacobian E/(pi sqrt(s) pT), the (pT/E) and dp factors cancel and
    the identical-kernel value in count space is the plain count-weighted
    mean sum(n pT) / sum(n) over the column's cells. Deviation DEV-1
    (recorded at P2 production, before any production verdict was computed):
    the P1 form passed raw cell counts into the closed form, which weights
    the counts as if they were densities and double-counts the pT bin width
    (up-weighting the coarse cells); the production statistic uses the
    plain count-weighted mean. The P1 probe record (bit-identity) is
    unaffected; no seed or generation parameter changes.
    """
    spectra, _integrated = load_pp_tables()
    values = []
    for sp in ("pi-", "pi+", "k+", "k-", "p", "pbar"):
        cells = sorted({(round(float(r[0]), 4), round(float(r[1]), 4))
                        for r in spectra[sp]
                        if abs(float(r[0])) < 1e-9
                        and float(r[1]) <= IN_WINDOW_PT_MAX + 1e-9})
        counts, _ov = bin_species_counts(events, cells, SPECIES_PDG[sp])
        values.append(seagull_count_mean_pt(cells, counts))
    return values


def pp_statistic_vector(events) -> list:
    """The flat, deterministic p+p statistic vector of the determinism probe."""
    values = pp_window_model_ratios(events)
    values.extend(pp_seagull_model_values(events))
    multiplicity = (sum(len(event.final_particles) for event in events)
                    / max(len(events), 1))
    values.append(float(multiplicity))
    return values


# --- p+A statistic vector (probe scale; P3 extends it) --------------------------


def pa_density_grid_columns(path: Path, xf_nodes=None) -> list:
    """The (xF, pT) cells of the p+A density grid (optionally one xF column)."""
    table = read_points(path)
    cells = sorted({(round(float(r[0]), 4), round(float(r[1]), 4))
                    for r in table
                    if xf_nodes is None
                    or any(abs(float(r[0]) - x) < 1e-6 for x in xf_nodes)})
    return cells


def pa_per_event_density(counts, cells, n_events: int) -> list:
    """The thesis per-event invariant density per cell:
    d(xF,pT) = E/(pi sqrt(s) pT) * n/(N_ev dxF dpT), with the pinned cell
    half-width convention doubled into the full bin widths."""
    values = []
    if n_events <= 0:
        return [float("nan")] * len(cells)
    for (xf, pt), n in zip(cells, counts):
        d_xf = 2.0 * cell_xf_half_width(xf)
        d_pt = 2.0 * cell_pt_half_width(pt)
        mass = SPECIES_MASS_GEV["pi-"]
        energy = math.sqrt(mass**2 + (xf * SQRT_S_GEV / 2.0) ** 2 + pt**2)
        density = (energy / (math.pi * SQRT_S_GEV * pt)) * (n / (n_events * d_xf * d_pt))
        values.append(float(density))
    return values


def pa_statistic_vector(pb_events, pc_events) -> list:
    """The flat, deterministic p+A statistic vector of the determinism probe."""
    values: list = []
    # p+Pb: the per-event density at the pi- and pi+ xF=0 columns of the
    # minimum-bias grid (the primary observable, binned on the data's own grid).
    for sp in ("pi-", "pi+"):
        cells = pa_density_grid_columns(
            pa_data_dir() / PPB_DENSITY_NAMES[sp], xf_nodes=[0.0])
        counts, overflow = bin_species_counts(
            pb_events, cells, SPECIES_PDG[sp])
        values.extend(pa_per_event_density(counts, cells, len(pb_events)))
        values.append(float(overflow))
    wounds_pb = [float(event.metadata.get("target_wounds", 1.0))
                 for event in pb_events]
    values.append(float(np.mean(wounds_pb)) if wounds_pb else float("nan"))
    # p+C: the f-side kernel-integrated dn/dxF at three xF nodes and <K>.
    for xf_t in (0.0, 0.1, 0.2):
        values.append(pa_kernel_dn_dxf(pc_events, xf_t, "pi-"))
    wounds_pc = [float(event.metadata.get("target_wounds", 1.0))
                 for event in pc_events]
    values.append(float(np.mean(wounds_pc)) if wounds_pc else float("nan"))
    # The model's backward-xF support (the p+C xF < 0 limitation pin).
    if pc_events:
        min_xf = min(particle_xf(p, event=event)
                     for event in pc_events
                     for p in event.final_particles
                     if int(particle_pdg(p)) in (-211, 211))
        values.append(float(min_xf))
    else:
        values.append(float("nan"))
    return values


def particle_pdg(particle):
    return particle.pdg_id


def pa_kernel_dn_dxf(events, xf_t: float, species: str) -> float:
    """dn/dxF(xF_t) from the event sample through the thesis kernel:
    dn/dxF = sum over the pT cells of (pi sqrt(s)/E) pT d(xF,pT) dpT."""
    cells = pa_density_grid_columns(
        pa_data_dir() / PPB_DENSITY_NAMES["pi-"], xf_nodes=[xf_t])
    counts, _overflow = bin_species_counts(events, cells, SPECIES_PDG[species])
    n_events = max(len(events), 1)
    total = 0.0
    mass = SPECIES_MASS_GEV[species]
    for (xf, pt), n in zip(cells, counts):
        d_xf = 2.0 * cell_xf_half_width(xf)
        d_pt = 2.0 * cell_pt_half_width(pt)
        density = (n / (n_events * d_xf * d_pt))
        energy = math.sqrt(mass**2 + (xf * SQRT_S_GEV / 2.0) ** 2 + pt**2)
        total += (math.pi * SQRT_S_GEV / energy) * pt * density * d_pt
    return float(total)


# --- Phase P2 production path (workstream A) -------------------------------------

PHASE_P2 = "P2"
WILSON_Z = 1.959963984540054  # the M9 Wilson 95% score interval z
REPLICA_SCHEMA = "staged_heldout_ratios_pa_replica_pp/v1"
MANIFEST_SCHEMA = "staged_heldout_ratios_pa_cells_manifest/v1"
RATIOS_REPORT_SCHEMA = "staged_heldout_ratios_pa_ratios_report/v1"
DETERMINISM_PRODUCTION_SCHEMA = "staged_heldout_ratios_pa_determinism_production/v1"
SEAGULL_NODE_MIN_CELLS = 2  # a node needs at least this many 2D-table cells
SYMMETRY_GRID_XF_NODES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
# spec 7.1 test 7 (no-primary-antibaryon-source): every final antibaryon's
# parent source mass must sit above the baryon-pair threshold (the M9
# criterion, 2 x 0.93827231 GeV for p pbar; per species the antibaryon mass
# plus the proton mass is the minimal pair content).
ANTIBARYON_PAIR_MARGIN_TOL = 1e-9
# spec 7.1 test 6 acceptance, pinned before production: the projectile-source
# (source_id 0) and the reflected target-source (source_id 1) distributions
# agree within 5 sigma per species, in the pooled totals and in every
# mirrored grid cell.
SYMMETRY_TEST_MAX_ABS_Z = 5.0

# Recorded deviations of the P2 production path (fixed before any production
# verdict was published; the pre-registration file is unchanged).
P2_DEVIATIONS = [
    {
        "id": "DEV-1",
        "statement": (
            "the P1 helper passed raw binned cell counts into the pinned "
            "seagull closed form, which weights the counts as if they were "
            "densities and double-counts the pT bin width (up-weighting the "
            "coarse cells). The pinned kernel form multiplies the per-cell "
            "invariant density by (pT/E) and the pT bin width; because the "
            "invariant density carries the thesis Jacobian "
            "E/(pi sqrt(s) pT), both factors cancel and the "
            "identical-kernel value in count space is the plain "
            "count-weighted pT mean sum(n pT)/sum(n) over the column's "
            "cells (pinned algebraically against the f-space form, which "
            "reproduces the published ptintegrated <pT> at the percent "
            "level, on the real tables). Recorded before any production "
            "verdict was computed; the artifacts store the raw per-cell "
            "counts, so the statistic is recomputed at collection and no "
            "event is regenerated; the P1 probe record (bit-identity) is "
            "unaffected; no seed, generation, or pre-registration change "
            "(the pre-registration sha256 is unchanged, no seed "
            "re-derivation is required)"),
        "test": (
            "test_model_side_seagull_uses_the_data_side_closed_form updated "
            "to pin the plain count-weighted mean and its algebraic "
            "identity with the f-space kernel form applied to the "
            "Jacobian-corrected densities"),
    },
    {
        "id": "DEV-2",
        "statement": (
            "the P1 ratio_window_statistics output carried the per-window "
            "yields and the data ratio but not the cell list, so the P1 "
            "determinism probe's ratio path (pp_window_model_ratios, which "
            "guards on win.get('cells')) silently skipped every window and "
            "the P1 probe statistic vector degenerated to the seagull and "
            "multiplicity entries. The fix carries the cell list additively "
            "in the registered-window output (existing consumers unaffected "
            "through .get); the manifest builder and the model ratio path "
            "then receive the pre-registered cell sets. Recorded before any "
            "production statistic was computed; the P1 probe record "
            "(bit-identity) is unaffected; no seed, generation, or "
            "pre-registration change"),
        "test": (
            "the P2 helper-suite pin that pp_window_model_ratios returns "
            "one value per registered ratio window (11 values) on a "
            "synthetic event sample"),
    },
    {
        "id": "DEV-3",
        "statement": (
            "the pre-registration's embedded data-side window values "
            "(ratio_data, Y_numerator, Y_denominator) were computed by the "
            "P1 helper without the per-cell dxF dpT areas, while the "
            "pre-registered yield-kernel definition specifies the "
            "with-areas sum, Y_h proportional to sum over the cells of "
            "(pT/E) f dxF dpT. The data side is recomputed with the areas "
            "per the pre-registered definition (the correction is largest "
            "for the proton-over-pion in-window ratios, +22 percent, where "
            "the proton spectrum concentrates in the coarse pT cells); the "
            "pre-registration file's registered_and_dropped_windows numbers "
            "are updated accordingly (the file changes, its sha256 is "
            "re-registered in-directory, and the harness cross-check is "
            "re-run and recorded). The model side already computes the "
            "with-areas form (the per-particle kernel contributions), so "
            "no event is regenerated and no seed changes; the per-cell "
            "counts of the production artifacts are unchanged"),
        "test": (
            "the helper-suite external-data pins updated to the with-areas "
            "data-side values; the model side pins unchanged"),
    },
]


def symmetry_grid_pt_nodes() -> tuple:
    """The standard pT grid of the symmetry check (0.05 below pT 0.3, 0.10
    above), nodes 0.05..1.40 GeV/c."""
    nodes = []
    pt = 0.05
    while pt <= 1.40 + 1e-9:
        nodes.append(round(pt, 4))
        pt = round(pt + (0.10 if pt >= 0.30 - 1e-9 else 0.05), 4)
    return tuple(nodes)


SYMMETRY_GRID_PT_NODES = symmetry_grid_pt_nodes()


def binning_rule_text() -> str:
    return (
        "a final particle enters the first cell (in the sorted cell order) "
        "whose node lies within the cell half-widths in both variables (xF "
        "bins 0.05 wide below xF 0.4, 0.10 above; pT bins 0.05 below pT "
        "0.3, 0.10 above); particles matching no cell of a cell set are "
        "unplaced for that set and the per-sample overflow counts the "
        "species particles that match no cell of any cell set binned for "
        "the species")


def seagull_node_support(spectra: dict, integrated: dict, species: str) -> dict:
    """Pure-table-arithmetic seagull support record for one species.

    For every ptintegrated xF node with 2D-table cells at
    pT <= IN_WINDOW_PT_MAX: the published <pT> (column 5) with its percent
    error (column 6; the table header pins 'Errors are always given in
    percent'), the identical-kernel reconstruction mean over the column's
    cells, its per-cell error propagation, the reconstruction deviation
    (kernel minus published, relative), and the cell count. The record is
    computed before production and the collection step re-derives it from
    the tables and refuses a mismatch.
    """
    t = spectra[species]
    ti = integrated[species]
    mass = SPECIES_MASS_GEV[species]
    nodes = {}
    for row in ti:
        xf_t = round(float(row[0]), 4)
        sel = (np.isclose(t[:, 0], xf_t, atol=1e-6)
               & (t[:, 1] <= IN_WINDOW_PT_MAX + 1e-9))
        col = t[sel]
        if not len(col):
            nodes[str(xf_t)] = {
                "xF": xf_t, "n_cells": 0, "supported": False,
                "reason": "no 2D-table cells at this ptintegrated node",
            }
            continue
        order = np.argsort(col[:, 1])
        pts = col[order, 1]
        fs = col[order, 2]
        errs = col[order, 3]
        dp = np.asarray([standard_dp(p) for p in pts])
        k = kernel_weight(pts, mass, xf_t)
        w = fs * k * dp
        w_sum = float(np.sum(w))
        mean = float(np.sum(pts * w) / w_sum)
        grad = k * dp * (pts - mean) / w_sum
        err = float(np.sqrt(np.sum((grad * errs) ** 2)))
        published = float(row[5])
        published_err = (float(row[6]) / 100.0) * published
        dev = ((mean - published) / published) if published else None
        nodes[str(xf_t)] = {
            "xF": xf_t,
            "n_cells": int(len(pts)),
            "pT_min": float(pts.min()), "pT_max": float(pts.max()),
            "cells": [[xf_t, float(p)] for p in pts],
            "published_mean_pt": published,
            "published_err_abs": published_err,
            "kernel_mean_pt": mean,
            "kernel_err_abs": err,
            "reconstruction_deviation": dev,
            "supported": bool(len(pts) >= SEAGULL_NODE_MIN_CELLS),
            "reason": ("" if len(pts) >= SEAGULL_NODE_MIN_CELLS
                       else "fewer than " + str(SEAGULL_NODE_MIN_CELLS)
                       + " 2D-table cells at this node"),
        }
    return {"species": species, "nodes": nodes}


def build_cells_manifest(config_path: Path, prereg_path: Path,
                         out_dir: Path) -> dict:
    """The P2 cells manifest: the pre-registered per-window and seagull cell
    sets as pure table arithmetic, written before any production generation
    and consumed byte-identically by the per-replica runner (local or
    gateway). Refuses on any harness mismatch (the frozen M8 set, the
    pre-registration constants, the bundled-config hash)."""
    out_dir = Path(out_dir)
    prereg = json.loads(Path(prereg_path).read_text(encoding="utf-8"))
    cross_check_preregistration(prereg)
    m8 = cross_check_m8_record()
    config_hash = canonical_configuration_hash(config_path)
    if config_hash != prereg.get("provenance", {}).get(
            "configuration_bundled_canonical_sha256"):
        raise SystemExit(
            "bundled config hash mismatch against the pre-registration "
            "provenance; refusing to build the cells manifest")
    spectra, integrated = load_pp_tables()
    stats = all_ratio_window_statistics()
    # The pre-registration's own window registry is authoritative for the
    # verdict set: windows dropped there (no common support, the single-pT
    # sliver, negative xF outside every registered window) stay out of the
    # model path with their recorded reason.
    prereg_win_registry = (prereg["protocol_A"]
                           ["registered_and_dropped_windows"])
    ratio_windows = {}
    for num, den in RATIO_NUMERATORS_DENOMINATORS:
        key = f"{num}_over_{den}"
        for win_name in WINDOW_CLASSES:
            win = stats[key]["windows"][win_name]
            registered_rec = prereg_win_registry[key][win_name]
            entry = {
                "numerator": num,
                "denominator": den,
                "predictive_content": registered_rec["predictive_content"],
                "status": registered_rec["status"],
            }
            if registered_rec["status"] == "registered":
                if not (win.get("cells")
                        and math.isclose(
                            float(win["ratio_data"]),
                            float(registered_rec["ratio_data"]),
                            rel_tol=1e-12, abs_tol=1e-12)):
                    raise SystemExit(
                        f"registered window {key}|{win_name}: the "
                        f"recomputed table arithmetic does not reproduce "
                        f"the pre-registered support")
                entry.update({
                    "n_cells": win["n_cells"],
                    "xF_min": win["xF_min"], "xF_max": win["xF_max"],
                    "pT_min": win["pT_min"], "pT_max": win["pT_max"],
                    "cells": [list(c) for c in win["cells"]],
                    "Y_numerator": win["Y_numerator"],
                    "Y_denominator": win["Y_denominator"],
                    "ratio_data": win["ratio_data"],
                })
            else:
                entry["drop_reason"] = registered_rec.get(
                    "drop_reason",
                    registered_rec.get("reason", "dropped at pre-registration"))
            ratio_windows[f"{key}|{win_name}"] = entry
    seagull = {}
    for sp in PP_TABLE_NAMES:
        seagull[sp] = seagull_node_support(spectra, integrated, sp)
    grid_cells = sorted(
        [(round(float(xf), 4), round(float(pt), 4))
         for xf in SYMMETRY_GRID_XF_NODES for pt in SYMMETRY_GRID_PT_NODES]
        + [(round(-float(xf), 4), round(float(pt), 4))
           for xf in SYMMETRY_GRID_XF_NODES for pt in SYMMETRY_GRID_PT_NODES])
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "card": CARD, "milestone": MILESTONE, "phase": PHASE_P2,
        "run_description": RUN_DESCRIPTION,
        "provenance": probe_provenance(config_hash),
        "m8_crosscheck": m8,
        "binning_rule": binning_rule_text(),
        "yield_kernel": (
            "Y_h proportional to the sum over the common-support cells of "
            "(pT/E) f dxF dpT with E = sqrt(m^2 + pL^2 + pT^2) and "
            "pL = xF sqrt(s)/2; the per-cell dxF dpT areas are shared by "
            "numerator and denominator of a ratio and the pre-registered "
            "cell-sum form applies them identically on both sides"),
        "seagull_kernel": (
            "<pT>(xF) is the pT/E-weighted invariant-density mean with "
            "E = sqrt(m^2 + pL^2 + pT^2), pL = xF sqrt(s)/2, and the pT bin "
            "widths (0.05 GeV/c below pT 0.3, 0.10 above); the kernel is "
            "identical on the model and the data sides and the model side "
            "converts its binned counts to per-cell densities before the "
            "identical closed form (DEV-1)"),
        "ratio_windows": ratio_windows,
        "seagull": seagull,
        "symmetry_grid": {
            "xF_nodes": list(SYMMETRY_GRID_XF_NODES),
            "pT_nodes": list(SYMMETRY_GRID_PT_NODES),
            "cells": [[x, p] for x, p in grid_cells],
            "statement": (
                "the spec 7.1 test-6 re-assertion grid: per species the "
                "projectile-source (source_id 0) counts at (xF, pT) are "
                "compared to the target-source (source_id 1) counts at "
                "(-xF, pT)"),
        },
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    _write_json(out_dir / "cells_manifest.json", manifest)
    n_nodes = sum(len(s["nodes"]) for s in seagull.values())
    print(f"[manifest] {len(ratio_windows)} ratio-window entries, "
          f"{n_nodes} seagull nodes, {len(grid_cells)} symmetry cells",
          flush=True)
    return manifest


def _bin_first_match(xf_arr: np.ndarray, pt_arr: np.ndarray,
                     cells: list) -> tuple:
    """The pinned first-match binning, vectorized: each particle enters the
    first cell (in the given order) whose node lies within the cell
    half-widths in both variables; unplaced particles carry -1. Identical to
    the scalar rule of bin_species_counts on a sorted cell list."""
    xf_arr = np.asarray(xf_arr, dtype=float)
    pt_arr = np.asarray(pt_arr, dtype=float)
    counts = np.zeros(len(cells), dtype=np.int64)
    assign = np.full(len(xf_arr), -1, dtype=np.int64)
    for i, (xf_i, pt_i) in enumerate(cells):
        hxi = cell_xf_half_width(xf_i)
        hpj = cell_pt_half_width(pt_i)
        mask = ((assign < 0)
                & (np.abs(xf_arr - xf_i) <= hxi + 1e-9)
                & (np.abs(pt_arr - pt_i) <= hpj + 1e-9))
        counts[i] = int(np.count_nonzero(mask))
        assign[mask] = i
    return counts, assign


def _cell_kernel_factors(cells: list, mass: float) -> np.ndarray:
    """The yield-kernel factor (pT/E) per manifest cell (the cell nodes)."""
    pts = np.asarray([c[1] for c in cells], dtype=float)
    xfs = np.asarray([c[0] for c in cells], dtype=float)
    out = np.empty(len(cells), dtype=float)
    for i in range(len(cells)):
        out[i] = float(kernel_weight(pts[i:i + 1], mass, float(xfs[i]))[0])
    return out


def _antibaryon_pair_margin(pdg_id: int) -> float:
    """The minimal baryon-pair source mass for a final antibaryon of this
    species: the antibaryon mass plus the proton mass (the baryon partner
    the local conservation law requires in the same source)."""
    if str(MIXING2006_DIR) not in sys.path:
        sys.path.insert(0, str(MIXING2006_DIR))
    from na49_mixing.pdg_database import get_mass

    return float(get_mass(int(pdg_id))) + 0.93827231


def build_replica_payload(seed: int, config_path: Path, manifest_path: Path,
                          n_events: int = PRIMARY_EVENTS_PER_REPLICA,
                          m8_record: Path | None = None,
                          preregistration: Path | None = None,
                          events: list | None = None) -> dict:
    """One workstream-A production replica payload.

    Generation at the frozen M8 set (or a supplied event list for the
    determinism re-assertion and the hermetic tests), event.validate()
    re-asserted per event at 1e-10 (inside generate_pp_events), the spec 7.1
    source-quantum-number and antibaryon-traceability re-assertions per
    event, first-match binning onto the manifest cells (ratio windows,
    seagull columns, symmetry grid), per-event kernel contributions for the
    pooled bootstrap, and the overflow bookkeeping. Wall fields are added by
    produce_replica and are excluded from the determinism comparison.
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    embedded_sha = manifest.pop("manifest_sha256", None)
    if embedded_sha is None or canonical_json_sha256(manifest) != embedded_sha:
        raise SystemExit(
            f"cells manifest embedded sha mismatch: {manifest_path}")
    m8_record_check = (cross_check_m8_record(Path(m8_record))
                       if m8_record is not None else cross_check_m8_record())
    if preregistration is not None:
        prereg = json.loads(Path(preregistration).read_text(encoding="utf-8"))
        cross_check_preregistration(prereg)
        config_hash = canonical_configuration_hash(config_path)
        if config_hash != prereg.get("provenance", {}).get(
                "configuration_bundled_canonical_sha256"):
            raise SystemExit(
                "bundled config hash mismatch against the pre-registration "
                "provenance; refusing generation")
    else:
        config_hash = canonical_configuration_hash(config_path)

    if events is None:
        events, summary, wall, _applied, _beam = generate_pp_events(
            int(n_events), int(seed), config_path)
    if not events:
        raise SystemExit("no events generated")

    max_res = 0.0
    source_qn_violations = 0
    n_antibaryons = 0
    antibaryon_violations = 0
    min_pair_margin = None
    for event in events:
        max_res = max(max_res,
                      float(event.conservation_residuals()["relative_max"]))
        for source in event.sources:
            if source.quantum_numbers.as_tuple() != (1, 1, 0):
                source_qn_violations += 1
        for particle in event.final_particles:
            if particle.quantum_numbers.B < 0:
                n_antibaryons += 1
                source = event.sources[int(particle.source_id)]
                partner = _antibaryon_pair_margin(int(particle.pdg_id))
                margin = float(source.mass) - partner
                if margin < -ANTIBARYON_PAIR_MARGIN_TOL:
                    antibaryon_violations += 1
                min_pair_margin = (margin if min_pair_margin is None
                                   else min(min_pair_margin, margin))

    arrays = {}
    for sp, pdg in SPECIES_PDG.items():
        xs, ps, evs, srcs = [], [], [], []
        for ev_index, event in enumerate(events):
            for particle in event.final_particles:
                if int(particle.pdg_id) != int(pdg):
                    continue
                xs.append(particle_xf(particle, event=event))
                ps.append(particle_pt(particle))
                evs.append(ev_index)
                srcs.append(int(particle.source_id))
        arrays[sp] = {
            "xf": np.asarray(xs, dtype=float),
            "pt": np.asarray(ps, dtype=float),
            "ev": np.asarray(evs, dtype=np.int64),
            "src": np.asarray(srcs, dtype=np.int64),
        }

    window_records = {}
    per_event: dict = {}
    placed_any = {sp: np.zeros(len(arrays[sp]["xf"]), dtype=bool)
                  for sp in arrays}
    for win_key in sorted(manifest["ratio_windows"]):
        entry = manifest["ratio_windows"][win_key]
        if entry.get("status") != "registered" or not entry.get("cells"):
            window_records[win_key] = {
                "status": entry.get("status", "dropped_no_common_support"),
                "n_cells": 0,
            }
            continue
        cells = [(float(c[0]), float(c[1])) for c in entry["cells"]]
        rec = {"n_cells": entry["n_cells"], "cells": entry["cells"],
               "predictive_content": entry["predictive_content"]}
        per_event[win_key] = {}
        for role, sp in (("num", entry["numerator"]),
                         ("den", entry["denominator"])):
            arr = arrays[sp]
            counts, assign = _bin_first_match(arr["xf"], arr["pt"], cells)
            k_cell = _cell_kernel_factors(cells, SPECIES_MASS_GEV[sp])
            placed = assign >= 0
            weights = np.zeros(len(assign), dtype=float)
            weights[placed] = k_cell[assign[placed]]
            contrib = np.bincount(arr["ev"][placed], weights=weights[placed],
                                  minlength=len(events))
            per_event[win_key][role] = [float(v) for v in contrib]
            rec[role] = {
                "counts": [int(v) for v in counts],
                "yield": float(contrib.sum()),
                "n_placed": int(placed.sum()),
            }
            placed_any[sp] |= placed
        window_records[win_key] = rec

    seagull_counts = {}
    for sp in sorted(arrays):
        arr = arrays[sp]
        rec = {}
        for node_key in sorted(manifest["seagull"][sp]["nodes"]):
            node = manifest["seagull"][sp]["nodes"][node_key]
            if not node.get("cells"):
                continue
            cells = [(float(c[0]), float(c[1])) for c in node["cells"]]
            counts, assign = _bin_first_match(arr["xf"], arr["pt"], cells)
            placed = assign >= 0
            placed_any[sp] |= placed
            rec[node_key] = [int(v) for v in counts]
        seagull_counts[sp] = rec

    grid_cells = [(float(c[0]), float(c[1]))
                  for c in manifest["symmetry_grid"]["cells"]]
    symmetry_counts = {}
    for sp in sorted(arrays):
        arr = arrays[sp]
        counts, assign = _bin_first_match(arr["xf"], arr["pt"], grid_cells)
        placed = assign >= 0
        placed_any[sp] |= placed
        per_src = np.zeros((2, len(grid_cells)), dtype=np.int64)
        np.add.at(per_src, (arr["src"][placed], assign[placed]), 1)
        symmetry_counts[sp] = per_src.tolist()

    species_binning = {}
    for sp in sorted(arrays):
        n_total = int(len(arrays[sp]["xf"]))
        species_binning[sp] = {
            "n_particles": n_total,
            "overflow": int(n_total - int(placed_any[sp].sum())),
            "seagull_counts": seagull_counts[sp],
            "symmetry_counts": symmetry_counts[sp],
        }
    payload = {
        "schema": REPLICA_SCHEMA,
        "card": CARD, "milestone": MILESTONE, "phase": PHASE_P2,
        "run_description": RUN_DESCRIPTION,
        # The volatile fields (the write timestamp and the wall clocks) are
        # added by produce_replica, so build_replica_payload is the pure,
        # bit-identical function of (seed, config, manifest) that the
        # determinism re-assertion compares.
        "provenance": {
            "configuration_hash": config_hash,
            "config_seed": CONFIG_SEED,
            "m12_seed_base": M12_SEED_BASE,
            "milestone": MILESTONE,
            "phase": PHASE_P2,
            "run_description": RUN_DESCRIPTION,
            "seed": int(seed),
            "stage_label": STAGE_LABEL_RATIOS,
            "helper_sha256": _sha256_file(Path(__file__)),
        },
        "m8_crosscheck": m8_record_check,
        "cells_manifest_path": str(manifest_path),
        "cells_manifest_sha256": _sha256_file(manifest_path),
        "n_events": int(len(events)),
        "mean_final_multiplicity": float(
            sum(len(event.final_particles) for event in events)
            / max(len(events), 1)),
        "conservation": {
            "n_events_validated": int(len(events)),
            "max_residual": float(max_res),
            "source_quantum_number_violations": int(source_qn_violations),
            "n_antibaryons": int(n_antibaryons),
            "antibaryon_pair_violations": int(antibaryon_violations),
            "min_antibaryon_pair_margin_gev": (
                float(min_pair_margin) if min_pair_margin is not None else None),
        },
        "species_binning": species_binning,
        "window_records": window_records,
        "per_event": per_event,
    }
    return payload


def produce_replica(seed: int, config_path: Path, manifest_path: Path,
                    out_path: Path, n_events: int = PRIMARY_EVENTS_PER_REPLICA,
                    m8_record: Path | None = None,
                    preregistration: Path | None = None) -> dict:
    """Build one production replica payload and write the artifact."""
    t0 = time.perf_counter()
    payload = build_replica_payload(
        seed, config_path, manifest_path, n_events=n_events,
        m8_record=m8_record, preregistration=preregistration)
    wall = time.perf_counter() - t0
    payload["wall_s"] = float(wall)
    payload["ms_per_event"] = float(
        1000.0 * wall / max(int(payload["n_events"]), 1))
    payload["provenance"]["written_utc"] = _utc_now_iso()
    _write_json(Path(out_path), payload)
    print(f"[replica seed {seed}] {payload['n_events']} events, "
          f"wall {wall:.1f}s, max residual "
          f"{payload['conservation']['max_residual']:.3e}", flush=True)
    return payload


def run_determinism_production(out_dir: Path, config_path: Path,
                               manifest_path: Path,
                               preregistration: Path | None = None,
                               m8_record: Path | None = None) -> dict:
    """Same-environment double generation at production scale: the full
    replica-0 payload is built twice at the workstream-A replica seed and
    the two payloads must be bit-identical (the spec 7.1 reproducibility
    test 5 at production scale; the M10 probe_A pattern)."""
    out_dir = Path(out_dir)
    seed = int(WS_A_REPLICA_SEEDS[0])
    t0 = time.perf_counter()
    payload_a = build_replica_payload(
        seed, config_path, manifest_path, m8_record=m8_record,
        preregistration=preregistration)
    payload_b = build_replica_payload(
        seed, config_path, manifest_path, m8_record=m8_record,
        preregistration=preregistration)
    wall = time.perf_counter() - t0
    canon_a = json.dumps(_to_jsonable(payload_a), sort_keys=True,
                         separators=(",", ":"))
    canon_b = json.dumps(_to_jsonable(payload_b), sort_keys=True,
                         separators=(",", ":"))
    record = {
        "schema": DETERMINISM_PRODUCTION_SCHEMA,
        "probe": "determinism_production",
        "pattern": "M10 probe_A at production scale, same environment",
        "provenance": {**probe_provenance(
            canonical_configuration_hash(config_path)), "seed": int(seed)},
        "n_events": int(payload_a["n_events"]),
        "n_repeats": 2,
        "bit_identical": bool(canon_a == canon_b),
        "payload_sha256": [hashlib.sha256(canon_a.encode()).hexdigest(),
                           hashlib.sha256(canon_b.encode()).hexdigest()],
        "wall_s": float(wall),
        "written_utc": _utc_now_iso(),
    }
    _write_json(out_dir / "probe_determinism_production.json", record)
    print(f"[determinism production] bit_identical="
          f"{record['bit_identical']}", flush=True)
    return record


def verdict_from_z(z: float) -> str:
    """The pre-registered band rule (k_pass = 2, k_tension = 5). A
    non-finite z carries no verdict (recorded as no_model_support)."""
    z = float(z)
    if not math.isfinite(z):
        return "no_model_support"
    az = abs(z)
    if az <= K_PASS:
        return "pass"
    if az <= K_TENSION:
        return "tension"
    return "beyond_tension"


def wilson_ratio_interval(n_num: int, n_den: int,
                          z: float = WILSON_Z) -> dict | None:
    """Wilson 95% score interval on the window count fraction, mapped to the
    ratio scale through the odds transform; recorded alongside the bootstrap
    interval (the pre-registered interval convention)."""
    n_total = int(n_num) + int(n_den)
    if n_total <= 0:
        return None
    p = float(n_num) / float(n_total)
    denom = 1.0 + z * z / n_total
    center = (p + z * z / (2.0 * n_total)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / n_total
                          + z * z / (4.0 * n_total * n_total))) / denom
    lo = max(center - half, 0.0)
    hi = min(center + half, 1.0)
    ratio_lo = lo / (1.0 - lo) if lo < 1.0 else float("inf")
    ratio_hi = hi / (1.0 - hi) if hi < 1.0 else float("inf")
    return {
        "z": float(z),
        "fraction": p,
        "fraction_interval": [lo, hi],
        "ratio_interval": [ratio_lo, ratio_hi],
    }


def bootstrap_ratio_interval(num_arr: np.ndarray, den_arr: np.ndarray,
                             replicates: int = BOOTSTRAP_REPLICATES,
                             seed: int = WS_A_BOOTSTRAP_SEED) -> dict:
    """Event-level bootstrap of the window ratio over the pooled replicas
    (the M9 convention: one generator consumed sequentially, 200 resamples;
    percentile interval)."""
    num = np.asarray(num_arr, dtype=float)
    den = np.asarray(den_arr, dtype=float)
    n = num.size
    rng = np.random.default_rng(int(seed))
    vals = np.empty(int(replicates), dtype=float)
    invalid = 0
    for b in range(int(replicates)):
        idx = rng.integers(0, n, n)
        d = float(den[idx].sum())
        if d > 0.0:
            vals[b] = float(num[idx].sum()) / d
        else:
            vals[b] = float("nan")
            invalid += 1
    finite = vals[np.isfinite(vals)]
    lo, hi = (float(np.percentile(finite, 2.5)),
              float(np.percentile(finite, 97.5))) if finite.size else (
        float("nan"), float("nan"))
    return {
        "replicates": int(replicates),
        "seed": int(seed),
        "invalid_replicates": int(invalid),
        "percentile_interval": [lo, hi],
    }


def _ratio_data_errors(table_num: np.ndarray, table_den: np.ndarray,
                       cells: list, mass_num: float,
                       mass_den: float) -> dict:
    """The pre-registered data-side error: the tables' per-cell statistical
    errors propagated through the with-areas yield kernel, combined for the
    ratio (independent numerator and denominator tables)."""
    def sigma_yield(table: np.ndarray, mass: float) -> float:
        index = {(round(float(r[0]), 4), round(float(r[1]), 4)):
                 (float(r[2]), float(r[3])) for r in table}
        s2 = 0.0
        for xf, pt in cells:
            _f, err = index[(round(float(xf), 4), round(float(pt), 4))]
            k = float(kernel_weight(np.asarray([pt]), mass, xf)[0])
            area = 2.0 * cell_xf_half_width(xf) * 2.0 * cell_pt_half_width(pt)
            s2 += (k * area * err) ** 2
        return math.sqrt(s2)

    y_num = window_yield(table_num, cells, mass_num)
    y_den = window_yield(table_den, cells, mass_den)
    s_num = sigma_yield(table_num, mass_num)
    s_den = sigma_yield(table_den, mass_den)
    ratio = (window_yield(table_num, cells, mass_num)
             / window_yield(table_den, cells, mass_den))
    sigma = abs(ratio) * math.sqrt((s_num / y_num) ** 2 + (s_den / y_den) ** 2)
    return {
        "ratio_data": float(ratio),
        "sigma_yield_numerator": float(s_num),
        "sigma_yield_denominator": float(s_den),
        "ratio_data_err": float(sigma),
    }


def _replica_seagull_mean(art: dict, species: str, node_key: str,
                          cells: list) -> float:
    """One replica's <pT> at a seagull node from the artifact's stored
    counts: the identical-kernel value in count space (the plain
    count-weighted pT mean, seagull_count_mean_pt)."""
    cells = [(float(c[0]), float(c[1])) for c in cells]
    counts = art["species_binning"][species]["seagull_counts"][node_key]
    return seagull_count_mean_pt(cells, counts)


def falsifier_cross_reference(ratio_records: dict) -> dict:
    """The pre-registered falsifier cross-reference (spec 7.3 falsifiers 1
    and 5) evaluated on the per-window verdicts.

    Falsifier 1 (ratios under local conservation): the pre-registered
    direction is the joint violation pattern - pi-/pi+ deviating toward
    charge asymmetry beyond k_tension in all three pi-/pi+ windows together
    with K+/K- deviating in the same charge-transfer direction beyond
    k_tension. Operationalized: all three registered pi-/pi+ windows carry
    |z| > K_TENSION with the same delta sign and |R_model - 1| > |R_data - 1|
    in every window, and the K+/K- in-window |z| > K_TENSION with the
    opposite delta sign (a charge transfer from K+ toward pi- when the
    pi-/pi+ delta is positive, and the mirrored direction when negative).

    Falsifier 5 (parameter values varying by species or xF window): one
    ratio's per-window deltas flipping sign between adjacent registered
    windows beyond k_tension on both sides of the flip.
    """
    pi_key = "pi-_over_pi+"
    pi_windows = [
        (win, ratio_records[f"{pi_key}|{win}"])
        for win in WINDOW_CLASSES
        if ratio_records.get(f"{pi_key}|{win}", {}).get("status")
        == "registered"]
    kk = ratio_records.get("k+_over_k-|in_window", {})

    falsifier_1 = {
        "spec": "7.3 falsifier 1 (ratios under local conservation)",
        "direction": (
            "the joint charge-asymmetry pattern: all three registered "
            "pi-/pi+ windows beyond k_tension in the same direction away "
            "from R = 1, together with K+/K- in-window beyond k_tension in "
            "the opposite delta sign (the same charge-transfer direction)"),
        "pi-_over_pi+_windows": {
            win: {"delta": rec.get("delta"),
                  "delta_over_sigma_combined":
                      rec.get("delta_over_sigma_combined"),
                  "verdict": rec.get("verdict")}
            for win, rec in pi_windows},
        "k+_over_k-_in_window": {
            "delta": kk.get("delta"),
            "delta_over_sigma_combined":
                kk.get("delta_over_sigma_combined"),
            "verdict": kk.get("verdict")},
    }
    z_pi = [rec.get("delta_over_sigma_combined") for _w, rec in pi_windows]
    deltas_pi = [rec.get("delta") for _w, rec in pi_windows]
    cond_joint = (
        len(z_pi) == len(pi_windows) and all(z is not None for z in z_pi)
        and all(abs(z) > K_TENSION for z in z_pi)
        and all((d > 0) == (deltas_pi[0] > 0) for d in deltas_pi))
    cond_asym = all(
        abs(rec["model_ratio"] - 1.0) > abs(rec["data_ratio"] - 1.0)
        for _w, rec in pi_windows)
    kk_z = kk.get("delta_over_sigma_combined")
    cond_kk = (
        kk_z is not None and abs(kk_z) > K_TENSION
        and kk.get("delta") is not None
        and ((kk["delta"] > 0) != (deltas_pi[0] > 0)))
    falsifier_1["fired"] = bool(cond_joint and cond_asym and cond_kk)

    flips = []
    for num, den in RATIO_NUMERATORS_DENOMINATORS:
        key = f"{num}_over_{den}"
        zs = []
        for win in WINDOW_CLASSES:
            rec = ratio_records.get(f"{key}|{win}", {})
            if rec.get("status") == "registered":
                z = rec.get("delta_over_sigma_combined")
                if z is not None:
                    zs.append((win, z))
        for (w1, z1), (w2, z2) in zip(zs, zs[1:]):
            if ((z1 > K_TENSION and z2 < -K_TENSION)
                    or (z1 < -K_TENSION and z2 > K_TENSION)):
                flips.append({
                    "ratio": key,
                    "window_pair": [w1, w2],
                    "z_pair": [z1, z2],
                })
    return {
        "falsifier_1_ratios": falsifier_1,
        "falsifier_5_parameter_variation": {
            "spec": ("7.3 falsifier 5 (required parameters vary "
                     "substantially by species or xF window)"),
            "direction": (
                "one ratio's per-window deltas flipping sign between "
                "adjacent windows beyond k_tension on both sides"),
            "flips": flips,
            "fired": bool(flips),
        },
        "policy": ("a recorded fail is a legitimate result (the M9/M10 "
                   "both-outcomes rule); a fired falsifier drives the P4 "
                   "wording and does not block the land gate"),
    }


def collect_ratios_report(run_dir: Path, config_path: Path,
                          prereg_path: Path,
                          gateway_manifest_path: Path | None = None) -> dict:
    """The P2 collection: re-verify the frozen set and the pre-registration,
    load the eight replica artifacts, recompute every data-side statistic
    from the tables (refusing a mismatch against the pre-registered values),
    evaluate the pre-registered verdict bands per ratio window and seagull
    node, run the pooled bootstrap and the Wilson intervals, evaluate the
    falsifier cross-reference and the spec 7.1 test-6 symmetry re-assertion,
    and write ratios_report.json."""
    run_dir = Path(run_dir)
    prereg = json.loads(Path(prereg_path).read_text(encoding="utf-8"))
    cross_check_preregistration(prereg)
    m8 = cross_check_m8_record()
    config_hash = canonical_configuration_hash(config_path)
    if config_hash != prereg.get("provenance", {}).get(
            "configuration_bundled_canonical_sha256"):
        raise SystemExit(
            "bundled config hash mismatch against the pre-registration "
            "provenance; refusing collection")

    manifest_path = run_dir / "cells_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = _sha256_file(manifest_path)
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if canonical_json_sha256(body) != manifest.get("manifest_sha256"):
        raise SystemExit("cells manifest embedded sha mismatch at collection")

    arts = []
    for seed in WS_A_REPLICA_SEEDS:
        path = run_dir / f"replica_{seed}.json"
        if not path.exists():
            raise SystemExit(f"missing replica artifact {path}")
        art = json.loads(path.read_text(encoding="utf-8"))
        if int(art["provenance"]["seed"]) != int(seed):
            raise SystemExit(f"replica seed mismatch in {path}")
        if art.get("cells_manifest_sha256") != manifest_sha:
            raise SystemExit(
                f"cells manifest sha mismatch in {path}")
        helper_sha = art["provenance"].get("helper_sha256")
        if helper_sha and helper_sha != _sha256_file(Path(__file__)):
            raise SystemExit(
                f"helper sha mismatch in {path}: the artifact was produced "
                "by a different helper version")
        arts.append(art)

    # Conservation is inviolable: refuse the collection on any violation.
    total_events = sum(int(a["n_events"]) for a in arts)
    max_res = max(float(a["conservation"]["max_residual"]) for a in arts)
    qn_violations = sum(int(a["conservation"]["source_quantum_number_violations"])
                        for a in arts)
    ab_violations = sum(int(a["conservation"]["antibaryon_pair_violations"])
                        for a in arts)
    if (max_res > 1e-10 or qn_violations or ab_violations):
        raise SystemExit(
            "conservation violation in the production samples; refusing "
            "collection")

    spectra, integrated = load_pp_tables()
    stats = all_ratio_window_statistics()
    prereg_windows = prereg["protocol_A"]["registered_and_dropped_windows"]

    # --- ratio windows ------------------------------------------------------
    ratio_windows_out: dict = {}
    seagull_out: dict = {}
    registered_order: list = []
    for num, den in RATIO_NUMERATORS_DENOMINATORS:
        key = f"{num}_over_{den}"
        data_stats = stats[key]
        for win_name in WINDOW_CLASSES:
            win = data_stats["windows"][win_name]
            win_key = f"{key}|{win_name}"
            # The pre-registration's window registry (mirrored in the
            # manifest) is authoritative for the verdict set.
            registered_rec = prereg_windows[key][win_name]
            record = {
                "numerator": num,
                "denominator": den,
                "predictive_content": registered_rec["predictive_content"],
                "status": registered_rec["status"],
            }
            if registered_rec["status"] != "registered" or not win.get("cells"):
                record["reason"] = registered_rec.get(
                    "drop_reason",
                    registered_rec.get(
                        "reason", "dropped at pre-registration"))
                ratio_windows_out[win_key] = record
                continue
            registered_order.append(win_key)
            cells = [(float(c[0]), float(c[1])) for c in win["cells"]]
            # Data side recomputed from the tables and cross-checked against
            # the pre-registered value (refusal on any mismatch).
            data_ratio = window_yield(spectra[num], cells,
                                      SPECIES_MASS_GEV[num]) / window_yield(
                spectra[den], cells, SPECIES_MASS_GEV[den])
            registered = prereg_windows[key][win_name]
            if not math.isclose(data_ratio, float(registered["ratio_data"]),
                                rel_tol=1e-12, abs_tol=1e-12):
                raise SystemExit(
                    f"data ratio mismatch against the pre-registration at "
                    f"{win_key}: {data_ratio!r} != "
                    f"{registered['ratio_data']!r}")
            errs = _ratio_data_errors(spectra[num], spectra[den], cells,
                                      SPECIES_MASS_GEV[num],
                                      SPECIES_MASS_GEV[den])
            # Per-replica model ratios from the per-event contributions, and
            # the per-event/binned-counts consistency check.
            per_replica = []
            consistency = []
            for art in arts:
                pe = art["per_event"][win_key]
                s_num = float(np.sum(pe["num"]))
                s_den = float(np.sum(pe["den"]))
                per_replica.append(s_num / s_den if s_den else float("nan"))
                mass_num = SPECIES_MASS_GEV[num]
                mass_den = SPECIES_MASS_GEV[den]
                for role, sp, mass in (("num", num, mass_num),
                                       ("den", den, mass_den)):
                    counts = art["window_records"][win_key][role]["counts"]
                    cells_list = [(float(c[0]), float(c[1]))
                                  for c in art["window_records"][win_key]["cells"]]
                    k_cell = _cell_kernel_factors(cells_list, mass)
                    expected = float(np.sum(
                        np.asarray(counts, dtype=float) * k_cell))
                    recorded = float(art["window_records"][win_key][role]["yield"])
                    consistency.append(abs(expected - recorded))
            model = (float(np.mean([v for v in per_replica
                                    if math.isfinite(v)]))
                     if any(math.isfinite(v) for v in per_replica)
                     else float("nan"))
            finite = [v for v in per_replica if math.isfinite(v)]
            mc = (float(np.std(finite, ddof=1)) if len(finite) > 1
                  else float("nan"))
            sigma_comb = math.sqrt(
                errs["ratio_data_err"] ** 2
                + (mc if math.isfinite(mc) else 0.0) ** 2)
            delta = model - data_ratio
            z = (delta / sigma_comb) if sigma_comb else float("nan")
            boot = bootstrap_ratio_interval(
                np.concatenate([np.asarray(a["per_event"][win_key]["num"],
                                           dtype=float) for a in arts]),
                np.concatenate([np.asarray(a["per_event"][win_key]["den"],
                                           dtype=float) for a in arts]))
            pooled_num = sum(
                sum(a["window_records"][win_key]["num"]["counts"])
                for a in arts)
            pooled_den = sum(
                sum(a["window_records"][win_key]["den"]["counts"])
                for a in arts)
            record.update({
                "n_cells": win["n_cells"],
                "xF_min": win["xF_min"], "xF_max": win["xF_max"],
                "pT_min": win["pT_min"], "pT_max": win["pT_max"],
                "n_replicas": len(finite),
                "per_replica_ratios": per_replica,
                "model_ratio": model,
                "mc_err_replica_spread": mc,
                "data_ratio": data_ratio,
                "data_ratio_err": errs["ratio_data_err"],
                "sigma_combined": sigma_comb,
                "delta": delta,
                "delta_over_sigma_combined": (z if sigma_comb else None),
                "verdict": (verdict_from_z(z) if sigma_comb
                            else "no_model_support"),
                "pooled_counts": {"numerator": int(pooled_num),
                                  "denominator": int(pooled_den)},
                "wilson_ratio_interval": wilson_ratio_interval(
                    pooled_num, pooled_den),
                "bootstrap": boot,
                "per_event_vs_counts_max_abs_diff": float(max(consistency)),
            })
            ratio_windows_out[win_key] = record

    # --- seagull -------------------------------------------------------------
    for sp in PP_TABLE_NAMES:
        support = seagull_node_support(spectra, integrated, sp)["nodes"]
        if support != manifest["seagull"][sp]["nodes"]:
            raise SystemExit(
                f"seagull support record mismatch against the tables for "
                f"{sp}; refusing collection")
        node_results = {}
        for node_key in sorted(support, key=lambda s: float(s)):
            node_rec = support[node_key]
            entry = {
                "supported": bool(node_rec["supported"]),
                "n_cells": node_rec["n_cells"],
                "pT_min": node_rec.get("pT_min"),
                "pT_max": node_rec.get("pT_max"),
                "published_mean_pt": node_rec.get("published_mean_pt"),
                "published_err_abs": node_rec.get("published_err_abs"),
                "kernel_mean_pt": node_rec.get("kernel_mean_pt"),
                "kernel_err_abs": node_rec.get("kernel_err_abs"),
                "reconstruction_deviation": node_rec.get(
                    "reconstruction_deviation"),
            }
            if not node_rec["supported"]:
                entry["verdict"] = "no_model_support"
                entry["reason"] = node_rec["reason"]
                node_results[node_key] = entry
                continue
            cells = [(float(c[0]), float(c[1])) for c in node_rec["cells"]]
            mass = SPECIES_MASS_GEV[sp]
            per_replica_means = []
            for art in arts:
                per_replica_means.append(_replica_seagull_mean(
                    art, sp, node_key, node_rec["cells"]))
            finite = [v for v in per_replica_means if math.isfinite(v)]
            model = float(np.mean(finite)) if finite else float("nan")
            mc = (float(np.std(finite, ddof=1)) if len(finite) > 1
                  else float("nan"))
            published = float(node_rec["published_mean_pt"])
            pub_err = float(node_rec["published_err_abs"])
            sigma_comb = math.sqrt(
                pub_err ** 2 + (mc if math.isfinite(mc) else 0.0) ** 2)
            delta = model - published
            z = (delta / sigma_comb) if sigma_comb else float("nan")
            recon = node_rec.get("reconstruction_deviation")
            attribution = bool(
                recon is not None
                and abs(recon) * published > K_PASS * sigma_comb)
            # kernel-consistent diagnostic (the identical estimator on both
            # sides; zero reconstruction systematic)
            k_delta = model - float(node_rec["kernel_mean_pt"])
            k_sigma = math.sqrt(
                float(node_rec["kernel_err_abs"]) ** 2
                + (mc if math.isfinite(mc) else 0.0) ** 2)
            entry.update({
                "n_replicas": len(finite),
                "model_mean_pt": model,
                "mc_err_replica_spread": mc,
                "delta": delta,
                "sigma_combined": sigma_comb,
                "delta_over_sigma_combined": (z if sigma_comb else None),
                "verdict": (verdict_from_z(z) if sigma_comb
                            else "no_model_support"),
                "reconstruction_dominates_band": attribution,
                "kernel_consistent_delta": k_delta,
                "kernel_consistent_sigma": k_sigma,
                "kernel_consistent_z": (k_delta / k_sigma
                                        if k_sigma else None),
            })
            node_results[node_key] = entry
        # slope over the supported nodes (at least two cells with model support)
        supported = sorted(
            (float(k) for k, v in node_results.items() if v.get("supported")
             and v.get("model_mean_pt") is not None
             and math.isfinite(float(v["model_mean_pt"]))))
        slope_record: dict = {"supported_nodes": supported}
        if len(supported) >= 2:
            def _slope(xs: list, ys: list) -> float:
                xa = np.asarray(xs, dtype=float)
                ya = np.asarray(ys, dtype=float)
                xm = float(np.mean(xa))
                sxx = float(np.sum((xa - xm) ** 2))
                return float(np.sum((xa - xm) * (ya - np.mean(ya)))) / sxx

            def _slope_err(xs: list, sigmas: list) -> float:
                xa = np.asarray(xs, dtype=float)
                xm = float(np.mean(xa))
                sxx = float(np.sum((xa - xm) ** 2))
                num = math.sqrt(float(np.sum(
                    (xa - xm) ** 2 * np.asarray(sigmas, dtype=float) ** 2)))
                return num / (sxx * sxx) ** 0.5 if sxx else float("nan")

            pub_ys = [float(support[str(x)]["published_mean_pt"])
                      for x in supported]
            pub_sig = [float(support[str(x)]["published_err_abs"])
                       for x in supported]
            kern_ys = [float(support[str(x)]["kernel_mean_pt"])
                       for x in supported]
            kern_sig = [float(support[str(x)]["kernel_err_abs"])
                        for x in supported]
            replica_slopes = []
            for art in arts:
                ys = []
                ok = True
                for x in supported:
                    v = _replica_seagull_mean(
                        art, sp, str(x), support[str(x)]["cells"])
                    if not math.isfinite(v):
                        ok = False
                        break
                    ys.append(v)
                if ok and len(ys) == len(supported):
                    replica_slopes.append(_slope(supported, ys))
                else:
                    replica_slopes.append(float("nan"))
            finite_slopes = [v for v in replica_slopes if math.isfinite(v)]
            model_slope = (float(np.mean(finite_slopes))
                           if finite_slopes else float("nan"))
            mc_slope = (float(np.std(finite_slopes, ddof=1))
                        if len(finite_slopes) > 1 else float("nan"))
            data_slope = _slope(supported, pub_ys)
            data_slope_err = _slope_err(supported, pub_sig)
            sigma_comb = math.sqrt(
                data_slope_err ** 2
                + (mc_slope if math.isfinite(mc_slope) else 0.0) ** 2)
            delta = model_slope - data_slope
            z = (delta / sigma_comb) if sigma_comb else float("nan")
            kernel_slope = _slope(supported, kern_ys)
            slope_record.update({
                "estimator": (
                    "unweighted least-squares slope of the <pT>(xF) profile "
                    "over the supported ptintegrated nodes; the data-side "
                    "slope error propagates the per-node published errors "
                    "through the least-squares formula, the MC error is the "
                    "replica spread of the per-replica slopes"),
                "n_replicas": len(finite_slopes),
                "model_slope": model_slope,
                "mc_err_replica_spread": mc_slope,
                "data_slope_published": data_slope,
                "data_slope_err_published": data_slope_err,
                "sigma_combined": sigma_comb,
                "delta": delta,
                "delta_over_sigma_combined": (z if sigma_comb else None),
                "verdict": (verdict_from_z(z) if sigma_comb
                            else "no_model_support"),
                "kernel_consistent": {
                    "data_slope_kernel": kernel_slope,
                    "delta_model_minus_kernel": (
                        model_slope - kernel_slope
                        if math.isfinite(model_slope) else None),
                },
            })
        seagull_out[sp] = {"nodes": node_results, "slope": slope_record}

    # --- spec 7.1 test 6: p+p projectile-target symmetry ----------------------
    grid_cells = [(float(c[0]), float(c[1]))
                  for c in manifest["symmetry_grid"]["cells"]]
    grid_index = {(round(x, 4), round(p, 4)): i
                  for i, (x, p) in enumerate(grid_cells)}
    symmetry_out = {}
    for sp in PP_TABLE_NAMES:
        pooled = np.zeros((2, len(grid_cells)), dtype=np.int64)
        for art in arts:
            pooled += np.asarray(
                art["species_binning"][sp]["symmetry_counts"], dtype=np.int64)
        n0 = int(pooled[0].sum())
        n1 = int(pooled[1].sum())
        z_total = ((n0 - n1) / math.sqrt(n0 + n1)) if (n0 + n1) else 0.0
        cell_zs = []
        for xf in SYMMETRY_GRID_XF_NODES:
            for pt in SYMMETRY_GRID_PT_NODES:
                i_pos = grid_index[(round(float(xf), 4), round(float(pt), 4))]
                i_neg = grid_index[(round(-float(xf), 4), round(float(pt), 4))]
                a = int(pooled[0, i_pos])
                b = int(pooled[1, i_neg])
                if a + b == 0:
                    continue
                cell_zs.append(((round(float(xf), 4), round(float(pt), 4)),
                                (a - b) / math.sqrt(a + b)))
        max_cell_z = max((abs(z) for _c, z in cell_zs), default=0.0)
        chi2 = (sum(z * z for _c, z in cell_zs) / len(cell_zs)
                if cell_zs else 0.0)
        symmetry_out[sp] = {
            "n_source0": n0,
            "n_source1": n1,
            "z_total": z_total,
            "n_cells_compared": len(cell_zs),
            "max_abs_cell_z": max_cell_z,
            "chi2_per_pair": chi2,
            "max_abs_z_allowed": SYMMETRY_TEST_MAX_ABS_Z,
            "verdict": ("pass" if abs(z_total) <= SYMMETRY_TEST_MAX_ABS_Z
                        and max_cell_z <= SYMMETRY_TEST_MAX_ABS_Z
                        else "fail"),
        }
    symmetry_pass = all(v["verdict"] == "pass" for v in symmetry_out.values())

    # --- falsifier cross-reference (spec 7.3 falsifiers 1 and 5) ---------------
    falsifiers = falsifier_cross_reference(ratio_windows_out)

    # --- wall and budget accounting ---------------------------------------------
    walls = [float(a.get("wall_s", 0.0)) for a in arts]
    ms = [float(a.get("ms_per_event", 0.0)) for a in arts]
    wall_accounting = {
        "n_replicas": len(arts),
        "events_per_replica": int(arts[0]["n_events"]),
        "total_events": int(total_events),
        "wall_s_per_replica": walls,
        "total_wall_s": float(sum(walls)),
        "ms_per_event_per_replica": ms,
        "mean_ms_per_event": float(np.mean(ms)),
        "pre_registered_projection": {
            "entry": "8 x 10000 (the primary statistics menu entry)",
            "projected_wall_s": 171.3055981323123,
            "source": "the P1 cost probe menu decision (probe_cost_cells.json)",
        },
        "per_point_budget_s": MAX_PROJECTED_WALL_S,
        "declared_reduction": None,
        "gateway_manifest": (json.loads(
            Path(gateway_manifest_path).read_text(encoding="utf-8"))
            if gateway_manifest_path and Path(gateway_manifest_path).exists()
            else None),
    }

    determinism = {}
    for name in ("probe_determinism.json", "probe_determinism_p2.json",
                 "probe_determinism_production.json"):
        p = run_dir / name
        if p.exists():
            determinism[name] = json.loads(p.read_text(encoding="utf-8"))

    report = {
        "schema": RATIOS_REPORT_SCHEMA,
        "card": CARD, "milestone": MILESTONE, "phase": PHASE_P2,
        "run_description": RUN_DESCRIPTION,
        "written_utc": _utc_now_iso(),
        "provenance": {
            "configuration_hash": config_hash,
            "config_seed": CONFIG_SEED,
            "m12_seed_base": M12_SEED_BASE,
            "run_description": RUN_DESCRIPTION,
            "m8_record_sha256": m8["m8_record_sha256"],
            "m8_frozen_block_sha256": m8["m8_frozen_block_sha256"],
            "pre_registration_sha256": _sha256_file(Path(prereg_path)),
            "pre_registration_changed": False,
            "cells_manifest_sha256": manifest_sha,
            "helper_sha256": _sha256_file(Path(__file__)),
            "replica_seeds": list(WS_A_REPLICA_SEEDS),
            "bootstrap_seed": WS_A_BOOTSTRAP_SEED,
            "outer_head": _git_head(REPO_ROOT),
            "nested_head": _git_head(MIXING2006_DIR),
        },
        "calibration_set": m8,
        "statistics": {
            "replicas": len(arts),
            "events_per_replica": int(arts[0]["n_events"]),
            "mc_error_convention": (
                "the replica spread (std, ddof=1) of the per-replica "
                "statistic; the replica mean is the model prediction"),
            "data_error_convention": (
                "the tables' per-cell statistical errors propagated through "
                "the pre-registered kernel; combined with the MC error in "
                "quadrature"),
            "interval_methods": (
                "bootstrap intervals (200 event-level resamples at the "
                "workstream bootstrap seed 20342831, consumed sequentially "
                "in the registered window order) with the Wilson 95% score "
                "interval recorded alongside"),
            "registered_bootstrap_order": registered_order,
        },
        "conservation": {
            "n_events": int(total_events),
            "max_residual": float(max_res),
            "tolerance": 1e-10,
            "source_quantum_number_violations": int(qn_violations),
            "antibaryon_pair_violations": int(ab_violations),
            "min_antibaryon_pair_margin_gev": min(
                float(a["conservation"]["min_antibaryon_pair_margin_gev"])
                for a in arts
                if a["conservation"]["min_antibaryon_pair_margin_gev"]
                is not None),
            "per_replica": [
                {"seed": int(a["provenance"]["seed"]),
                 "max_residual": float(a["conservation"]["max_residual"]),
                 "n_events": int(a["n_events"])}
                for a in arts],
            "tests": "spec 7.1 tests 1-5 and 7 re-asserted per event; "
                     "the p+p symmetry test 6 evaluated on the pooled sample",
        },
        "symmetry_test_6": {
            "statement": ("spec 7.1 test 6 (p+p projectile-target symmetry "
                          "before analysis cuts): the projectile-source "
                          "(source_id 0) and the reflected target-source "
                          "(source_id 1) per-species distributions on the "
                          "symmetry grid; acceptance pinned before "
                          "production at |z| <= "
                          + str(SYMMETRY_TEST_MAX_ABS_Z) + " per species "
                          "(totals and per-cell maximum)"),
            "species": symmetry_out,
            "verdict": ("pass" if symmetry_pass else "fail"),
        },
        "ratio_windows": ratio_windows_out,
        "seagull": seagull_out,
        "falsifiers": falsifiers,
        "wall_budget_accounting": wall_accounting,
        "determinism": determinism,
        "deviations": P2_DEVIATIONS,
        "overflow": {
            sp: {
                "n_particles": int(a["species_binning"][sp]["n_particles"]),
                "overflow": int(a["species_binning"][sp]["overflow"]),
                "convention": binning_rule_text(),
            }
            for sp in PP_TABLE_NAMES
            for a in [arts[0]]
        },
        "replicas": [
            {"seed": int(a["provenance"]["seed"]),
             "n_events": int(a["n_events"]),
             "wall_s": float(a.get("wall_s", 0.0)),
             "mean_final_multiplicity": float(a["mean_final_multiplicity"]),
             "max_conservation_residual": float(
                 a["conservation"]["max_residual"]),
             "helper_sha256": a["provenance"].get("helper_sha256"),
             "artifact": f"replica_{a['provenance']['seed']}.json"}
            for a in arts],
    }
    _write_json(run_dir / "ratios_report.json", report)
    print(f"[collect] {len(ratio_windows_out)} ratio-window entries, "
          f"{sum(len(s['nodes']) for s in seagull_out.values())} seagull "
          f"nodes; max residual {max_res:.3e}", flush=True)
    return report


# --- Phase P3 production path (workstream B) -------------------------------------

PHASE_P3 = "P3"
PA_REPLICA_SCHEMA = "staged_heldout_ratios_pa_replica_pa/v1"
PPREF_REPLICA_SCHEMA = "staged_heldout_ratios_pa_replica_ppref/v1"
PA_MANIFEST_SCHEMA = "staged_heldout_ratios_pa_pa_cells_manifest/v1"
PA_REPORT_SCHEMA = "staged_heldout_ratios_pa_pa_trends_report/v1"
PA_DETERMINISM_SCHEMA = "staged_heldout_ratios_pa_determinism_production_pa/v1"
PILOT_SCHEMA = "staged_heldout_ratios_pa_pilot_pins/v1"
PA_MANIFEST_NAME = "pa_cells_manifest.json"
PILOT_RECORD_NAME = "pilot_pa_pins.json"
PA_DETERMINISM_NAME = "probe_determinism_production_pa.json"
GATEWAY_MANIFEST_P3_NAME = "gateway_manifest_p3.json"
PA_SYSTEM_GEOMETRY = {"pPb208": GEOMETRY_PB208, "C12": GEOMETRY_C12}
PA_SYSTEMS = ("pPb208", "C12")
PA_SYSTEM_DATA_KEY = {"pPb208": "pPb", "C12": "pC"}
PA_F_SPECIES = {"pPb208": ("pi-", "pi+"), "C12": ("pi-", "pi+", "p", "pbar")}
PPREF_SPECIES = ("pi-", "pi+", "p", "pbar")
# The foundations section 5 rare-slice regression pins: Pb208 single-bin
# slices with the recorded expected min-bias fractions, one event each
# through the chunked worker (the recorded seeds of the regression test).
RARE_SLICE_KS = (21, 24, 25)
RARE_SLICE_FRACTIONS = {21: 5.6e-7, 24: 1.2e-8, 25: 3.1e-9}
RARE_SLICE_SEED_BASE = 20260826  # the recorded regression-test convention
POINT_TARGET_SEED = 123456789    # the recorded regression-test seed
POINT_TARGET_EVENTS = 50
# The pion A-scaling common-xF nodes: the intersection of the three
# systems' ptintegrated node sets restricted to the pre-registered R(xF)
# rows (seven nodes per pion charge).
A_SCALING_XF_NODES = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
# The p+Pb f table's extra lowest point sits at the pT label 0.00 (pa_data
# README anomaly 1); the pinned model-side Jacobian estimator is undefined
# there and the pre-registered kernel weight (pT/E) vanishes on the data
# side, so the point is excluded from the model comparison with the
# recorded reason (its data-side kernel contribution is exactly zero).
PPB_F_PT_NODE_MIN = 1e-9
# Falsifier-4 operational definition (recorded before production): the
# forward half of the p+Pb R(xF) support.
FALSIFIER4_FORWARD_XF_MIN = 0.2
# Recorded deviations of the P3 production path (fixed before any production
# verdict was published; the pre-registration file is unchanged).
P3_DEVIATIONS: list = []


def grid_edges(values) -> list[tuple[float, float]]:
    """Midpoint bin edges around sorted unique node values, with the
    boundary half-spacing extension.

    On the standard grids (0.05 spacing below the 0.3/0.4 knot, 0.10 above)
    this reproduces the pinned half-width convention exactly (pinned by a
    helper-suite test). The p+C tables carry non-standard columns (xF
    spacings down to 0.005, per-column pT grids), where the pinned
    fixed-width first-match rule is structurally undefined (overlapping
    windows would assign forward particles to backward cells under the
    sorted first-match order); there the table's own midpoint bins are the
    data's own grid.
    """
    vals = sorted({round(float(v), 6) for v in np.asarray(values, dtype=float)})
    n = len(vals)
    edges: list[tuple[float, float]] = []
    for i, v in enumerate(vals):
        lo_step = ((vals[i] - vals[i - 1]) / 2.0 if i > 0
                   else (vals[1] - vals[0]) / 2.0 if n > 1 else 0.025)
        hi_step = ((vals[i + 1] - vals[i]) / 2.0 if i < n - 1
                   else (vals[i] - vals[i - 1]) / 2.0 if n > 1 else 0.025)
        edges.append((round(v - lo_step, 9), round(v + hi_step, 9)))
    return edges


def _grid_matches_pinned(nodes, half_width_fn) -> bool:
    """The standard-grid coincidence pin: on a 0.05/0.10-spaced grid the
    midpoint edges equal the pinned half-width windows."""
    vals = sorted({round(float(v), 6) for v in np.asarray(nodes, dtype=float)})
    edges = grid_edges(vals)
    return all(
        math.isclose(lo, v - half_width_fn(v), abs_tol=1e-9)
        and math.isclose(hi, v + half_width_fn(v), abs_tol=1e-9)
        for v, (lo, hi) in zip(vals, edges))


def column_pT_range(pt_values, width_rule: str) -> tuple:
    """The pT acceptance range of one grid column: the outermost bin edges
    under the column's width rule (the pinned half-widths, or the table's
    own midpoint edges for the p+C grids)."""
    vals = sorted({round(float(v), 6)
                   for v in np.asarray(pt_values, dtype=float)})
    if not vals:
        raise ValueError("empty pT column")
    if width_rule == PA_WIDTH_RULE_OWN:
        edges = dict(zip(vals, grid_edges(vals)))
        lo = min(e[0] for e in edges.values())
        hi = max(e[1] for e in edges.values())
    else:
        lo = min(v - cell_pt_half_width(v) for v in vals)
        hi = max(v + cell_pt_half_width(v) for v in vals)
    return (lo, hi)


PA_WIDTH_RULE_PINNED = "pinned_half_widths"
PA_WIDTH_RULE_OWN = "table_own_midpoint_edges"

PA_BINNING_STATEMENT = (
    "the p+A cell comparisons bin the model with the pinned first-match "
    "rule and the pinned half-widths on the p+Pb density and f grids (the "
    "recorded workstream-A rule; the grids are standard so the table's own "
    "midpoint edges coincide, pinned by a helper-suite test). The p+C "
    "tables carry non-standard columns (xF spacings down to 0.005, "
    "per-column pT grids) where the pinned fixed-width windows overlap and "
    "the sorted first-match order would assign forward particles to "
    "backward cells; there the model bins into the table's own midpoint "
    "bins (the data's own grid), one exact cell per particle. The R(xF) "
    "node estimator is a box count, not a tiling binning: at a registered "
    "node each system counts the species particles within the pinned xF "
    "window and its own data-grid pT acceptance at that node (recorded per "
    "node), mirroring how each system's published ptintegrated dn/dxF "
    "carries its own acceptance; the xF width cancels in the R ratio. The "
    "overflow bookkeeping covers the tiling cell sets only; the R node box "
    "counts carry no overflow semantics. The p+A binned observables count "
    "the produced final particles of the sources; the target spectators "
    "(at rest, source_id -1) enter the accounting test only.")

PA_R_NODE_STATEMENT = (
    "R(xF) = (dn/dxF)_pA / ((1+<K>_sampler)/2 (dn/dxF)_pp). Data side: the "
    "pre-registered recomputation from the ptintegrated tables (refused on "
    "any mismatch against the pre-registration). Model side: at a "
    "registered node, (dn/dxF) is the species count within the pinned xF "
    "window (half-width 0.025 below xF 0.4, 0.05 above) inside the "
    "recorded common pT window (the intersection of the system's own "
    "data-grid column range and the pp reference column range, so the pT "
    "acceptance cancels inside the model ratio), divided by N_events "
    "times the xF bin width; the width cancels in R and the per-replica R "
    "values use the pooled production <K> of the system. The model R is "
    "the like-for-like model-only diagnostic; the data-side R carries "
    "each system's published acceptance (the pre-registered record), and "
    "the residual acceptance systematic is auditable per node through the "
    "recorded column ranges.")


def _density_cell_set(species: str) -> dict:
    """The p+Pb per-event density cell set of one charge: the data's own
    fine grid with the pinned half-width widths and the data values."""
    table = read_points(pa_data_dir() / PPB_DENSITY_NAMES[species])
    rows = sorted((round(float(r[0]), 4), round(float(r[1]), 4),
                   float(r[2]), float(r[3])) for r in table)
    xf_vals = sorted({r[0] for r in rows})
    pt_vals = sorted({r[1] for r in rows})
    cells = []
    for xf, pt, value, err in rows:
        cells.append({
            "xF": xf, "pT": pt,
            "dxf": 2.0 * cell_xf_half_width(xf),
            "dpt": 2.0 * cell_pt_half_width(pt),
            "data_value": value, "data_err": err,
        })
    return {
        "species": species,
        "quantity": "per-event invariant density d(xF,pT)",
        "width_rule": PA_WIDTH_RULE_PINNED,
        "grid_coincides_with_pinned_xF": _grid_matches_pinned(
            xf_vals, cell_xf_half_width),
        "grid_coincides_with_pinned_pT": _grid_matches_pinned(
            pt_vals, cell_pt_half_width),
        "n_cells": len(cells),
        "cells": cells,
    }


def _pPb_f_cell_set(species: str) -> dict:
    """The p+Pb invariant-cross-section cell set of one charge: the
    duplicated pT labels resolved through the fine-grid disambiguation,
    the data used as-is as separate points, the pT = 0.00 point excluded
    with the recorded reason (no pinned model-side Jacobian estimator;
    zero data-side kernel weight)."""
    table = read_points(pa_data_dir() / PPB_XSECT_NAMES[species])
    pt_dis = fine_pt_labels(table[:, 0].copy(), table[:, 1].copy())
    rows = []
    n_excluded = 0
    for i in range(len(table)):
        pt = round(float(pt_dis[i]), 4)
        if pt <= PPB_F_PT_NODE_MIN:
            n_excluded += 1
            continue
        rows.append((round(float(table[i, 0]), 4), pt,
                     float(table[i, 2]), float(table[i, 3])))
    rows.sort()
    cells = []
    for xf, pt, value, err in rows:
        cells.append({
            "xF": xf, "pT": pt,
            "dxf": 2.0 * cell_xf_half_width(xf),
            "dpt": 2.0 * cell_pt_half_width(pt),
            "data_value": value, "data_err": err,
        })
    return {
        "species": species,
        "quantity": "invariant cross section f(xF,pT) [mb/(GeV/c)^2]",
        "width_rule": PA_WIDTH_RULE_PINNED,
        "duplicated_pT_rule": (
            "the duplicated pT labels 0.10/0.20 are preserved as separate "
            "points and paired through the fine-grid disambiguation "
            "(pa_data anomaly 1); the pT = 0.00 point is excluded from the "
            "model comparison (the pinned Jacobian estimator is undefined "
            "at pT = 0 and the pre-registered kernel weight (pT/E) is zero "
            "there on the data side)"),
        "n_cells_table": int(len(table)),
        "n_cells_excluded_pT0": int(n_excluded),
        "n_cells": len(cells),
        "cells": cells,
    }


def _pC_f_cell_set(species: str, min_xf_supported=None) -> dict:
    """The p+C invariant-cross-section cell set of one species: the
    table's own variable bins (midpoint edges per xF column and
    per-column pT edges). If the model carries no support at xF below
    min_xf_supported, the cells below that bound are excluded with the
    recorded reason (the pre-registered target/projectile-separation
    restriction)."""
    table = read_points(spshadrons_data_dir() / PC_XSECT_NAMES[species])
    columns: dict[float, list[tuple[float, float, float]]] = {}
    for row in table:
        xf = round(float(row[0]), 4)
        columns.setdefault(xf, []).append(
            (round(float(row[1]), 4), float(row[2]), float(row[3])))
    xf_vals = sorted(columns)
    xf_edges = dict(zip(xf_vals, grid_edges(xf_vals)))
    cells = []
    n_excluded = 0
    for xf in xf_vals:
        col = sorted(columns[xf])
        pt_vals = [c[0] for c in col]
        pt_edges = dict(zip(pt_vals, grid_edges(pt_vals)))
        for pt, value, err in col:
            if (min_xf_supported is not None
                    and xf < float(min_xf_supported) - 1e-9):
                n_excluded += 1
                continue
            lo_xf, hi_xf = xf_edges[xf]
            lo_pt, hi_pt = pt_edges[pt]
            cells.append({
                "xF": xf, "pT": pt,
                "lo_xF": lo_xf, "hi_xF": hi_xf,
                "lo_pT": pt_edges[pt][0], "hi_pT": pt_edges[pt][1],
                "dxf": round(hi_xf - lo_xf, 9), "dpt": round(
                    hi_pt := pt_edges[pt][1] - pt_edges[pt][0], 9),
                "data_value": value, "data_err": err,
            })
    return {
        "species": species,
        "quantity": "invariant cross section f(xF,pT) [mb/(GeV/c)^2]",
        "width_rule": PA_WIDTH_RULE_OWN,
        "xF_restriction_min": (None if min_xf_supported is None
                               else float(min_xf_supported)),
        "n_cells_excluded_backward_support": int(n_excluded),
        "n_cells": len(cells),
        "cells": cells,
    }


def _r_node_entries(system: str, species: str, analytic_K: float) -> list:
    """The pre-registered R(xF) node plan of one system and species: the
    data-side rows (recomputed and pinned against the pre-registration at
    manifest build) with the system's own data-grid pT acceptance per
    node."""
    data_key = PA_SYSTEM_DATA_KEY[system]
    rows = data_side_r_xf(data_key, analytic_K)[species]
    if system == "pPb208":
        grid = read_points(pa_data_dir() / PPB_DENSITY_NAMES[species])
        support_source = "pPb_density_grid"
    else:
        grid = read_points(spshadrons_data_dir() / PC_XSECT_NAMES[species])
        support_source = "pC_f_table"
    entries = []
    for row in rows:
        xf = float(row["xF"])
        col = grid[np.isclose(grid[:, 0], xf, atol=1e-6)]
        if not len(col):
            entries.append({
                "xF": xf, "species": species, "supported": False,
                "reason": "no data-grid column at this node",
                "data_R_samplerK": row["R_samplerK"],
                "data_R_measuredAnchor": row["R_measuredAnchor"],
            })
            continue
        lo, hi = column_pT_range(
            col[:, 1],
            PA_WIDTH_RULE_PINNED if support_source == "pPb_density_grid"
            else PA_WIDTH_RULE_OWN)
        entries.append({
            "xF": xf,
            "species": species,
            "xF_half_width": cell_xf_half_width(xf),
            "supported": True,
            "support_source": support_source,
            "pT_lo": float(lo), "pT_hi": float(hi),
            "n_pT_cells": int(len(col)),
            "data_R_samplerK": float(row["R_samplerK"]),
            "data_R_measuredAnchor": float(row["R_measuredAnchor"]),
            "anchor_deviation": float(row["anchor_deviation"]),
            "data_dn_dxF_pa": float(row["dn_dxF_pa"]),
            "data_dn_dxF_err_raw": float(row["dn_dxF_err"]),
            "data_dn_dxF_pp": float(row["dn_dxF_pp"]),
        })
    return entries


def _ppref_r_node_entries(species: str, nodes: list) -> list:
    """The p+p reference side of the R(xF) plan: the pp xsect-table pT
    acceptance at each needed node."""
    grid = read_points(spshadrons_data_dir() / PP_TABLE_NAMES[species])
    entries = []
    for xf in sorted(nodes):
        col = grid[np.isclose(grid[:, 0], float(xf), atol=1e-6)]
        if not len(col):
            entries.append({
                "xF": float(xf), "supported": False,
                "reason": "no pp xsect-table column at this node",
            })
            continue
        lo, hi = column_pT_range(col[:, 1], PA_WIDTH_RULE_PINNED)
        entries.append({
            "xF": float(xf),
            "xF_half_width": cell_xf_half_width(float(xf)),
            "supported": True,
            "support_source": "pp_xsect_table",
            "pT_lo": float(lo), "pT_hi": float(hi),
            "n_pT_cells": int(len(col)),
        })
    return entries


def build_pa_cells_manifest(config_path: Path, prereg_path: Path,
                            out_dir: Path,
                            pilot_record_path: Path | None = None) -> dict:
    """The P3 cells manifest: the pre-registered p+A cell sets and R(xF)
    node plans as pure table arithmetic, written before any production
    generation and consumed byte-identically by the per-replica runner
    (local or gateway). Refuses on any harness mismatch (the frozen M8
    set, the pre-registration constants, the bundled-config hash) and on
    a missing pilot record (the model backward-xF support pin and the
    regression pins precede production).

    The p+C cell restriction follows the recorded model backward-xF
    support of the pilot record: if the model carries no support at
    xF < 0, the p+C comparison restricts to xF >= 0 with the recorded
    limitation (the pre-registration model_xF_support_pin).
    """
    out_dir = Path(out_dir)
    prereg = json.loads(Path(prereg_path).read_text(encoding="utf-8"))
    cross_check_preregistration(prereg)
    m8 = cross_check_m8_record()
    config_hash = canonical_configuration_hash(config_path)
    if config_hash != prereg.get("provenance", {}).get(
            "configuration_bundled_canonical_sha256"):
        raise SystemExit(
            "bundled config hash mismatch against the pre-registration "
            "provenance; refusing to build the p+A cells manifest")

    pilot_path = (Path(pilot_record_path) if pilot_record_path is not None
                  else out_dir / PILOT_RECORD_NAME)
    if not pilot_path.exists():
        raise SystemExit(
            f"pilot pin record not found at {pilot_path}; the pilot "
            "precedes production (the card refusal rule)")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if not pilot.get("all_pins_hold"):
        raise SystemExit(
            "the pilot pin record does not hold; refusing to build the "
            "p+A cells manifest (the card refusal rule)")
    support = pilot.get("model_backward_xF_support", {})

    geo = geometry_anchors()
    prereg_anchors = prereg["geometry"]["sampler_anchors"]
    for label in ("Pb208", "C12"):
        for key in ("sigma_pA_mb", "mean_target_wounds"):
            expected = float(prereg_anchors[label][key])
            if not math.isclose(float(geo[label][key]), expected,
                                rel_tol=1e-12, abs_tol=1e-12):
                raise SystemExit(
                    f"sampler anchor drift for {label}.{key}: "
                    f"{geo[label][key]!r} != pre-registered {expected!r}")

    # The pre-registered p+C restriction rule: if the model carries no
    # support at xF < 0 (the pilot-recorded backward-xF minimum of the
    # pions at the P1 probe seeds is >= 0), the p+C comparison restricts
    # to xF >= 0 with the recorded limitation.
    c12_support = support.get("C12") or {}
    min_pion_xf_c12 = c12_support.get("min_pion_xF")
    pc_restriction_min_xf = (0.0 if (min_pion_xf_c12 is None
                                     or float(min_pion_xf_c12) >= 0.0)
                             else None)

    # The p+p reference column ranges first: the model-side R window at a
    # node is the intersection of the two systems' data-grid column
    # ranges (the like-for-like rule recorded in the manifest).
    pp_column_ranges: dict = {}
    for sp in PPREF_SPECIES:
        grid = read_points(spshadrons_data_dir() / PP_TABLE_NAMES[sp])
        pp_column_ranges[sp] = {}
        for xf in sorted({round(float(r[0]), 4) for r in grid}):
            col = grid[np.isclose(grid[:, 0], float(xf), atol=1e-6)]
            if len(col):
                pp_column_ranges[sp][xf] = column_pT_range(
                    col[:, 1], PA_WIDTH_RULE_PINNED)

    systems = {}
    r_plan_registry: list = []
    pc_restriction_record = {
        "restricted": pc_restriction_min_xf is not None,
        "min_xF": pc_restriction_min_xf,
        "model_backward_min_pion_xF": min_pion_xf_c12,
        "statement": (
            "the pre-registered rule applied to the recorded pilot "
            "backward-xF support: if the model carries no support at "
            "xF < 0, the p+C comparison restricts to xF >= 0 with the "
            "recorded limitation, and the restriction is reported as a "
            "statement on target/projectile separation (the spec "
            "section 9 observable)"),
    }

    for system in PA_SYSTEMS:
        analytic_K = float(geo[("Pb208" if system == "pPb208" else "C12")
                               ]["mean_target_wounds"])
        cell_sets = {}
        if system == "pPb208":
            for sp in ("pi-", "pi+"):
                cell_sets[f"density_{sp}"] = _density_cell_set(sp)
                cell_sets[f"f_{sp}"] = _pPb_f_cell_set(sp)
        else:
            for sp in PA_F_SPECIES[system]:
                cell_sets[f"f_{sp}"] = _pC_f_cell_set(
                    sp, pc_restriction_min_xf)
        r_nodes = {}
        for sp in PA_F_SPECIES[system]:
            entries = _r_node_entries(system, sp, analytic_K)
            r_nodes[sp] = entries
            for entry in entries:
                if entry.get("supported"):
                    r_plan_registry.append([system, sp, entry["xF"]])
        systems[system] = {
            "nucleus": dict(PA_SYSTEM_GEOMETRY[system]),
            "stage_label": STAGE_LABEL_PA,
            "runner_settings": dict(PA_RUNNER_SETTINGS),
            "analytic_mean_target_wounds": float(analytic_K),
            "cell_sets": cell_sets,
            "r_nodes": r_nodes,
        }

    # The p+p reference plan: the union of the p+A node sets per species.
    ppref_nodes: dict = {}
    for sp in PPREF_SPECIES:
        needed = set()
        for system in PA_SYSTEMS:
            if sp in PA_F_SPECIES[system]:
                for entry in systems[system]["r_nodes"][sp]:
                    if entry.get("supported"):
                        needed.add(round(float(entry["xF"]), 4))
        # The pp side reuses the node list; its supports come from the pp
        # xsect table columns.
        entries = _ppref_node_entries(sp, sorted(needed))
        ppref_nodes[str(sp)] = entries
        for entry in entries:
            if entry.get("supported"):
                r_plan_registry.append(["ppref", sp, entry["xF"]])
    systems["ppref"] = {
        "nucleus": None,
        "stage_label": STAGE_LABEL_PPREF,
        "r_nodes": ppref_nodes,
    }

    # The model-side R window at each supported node: the intersection of
    # the system's own data-grid column range and the pp reference column
    # range (the same window on both sides of the model ratio, so the pT
    # acceptance cancels inside R; each side's own published range is
    # recorded for the acceptance statement).
    for system in PA_SYSTEMS:
        for sp, entries in systems[system]["r_nodes"].items():
            for entry in entries:
                if not entry.get("supported"):
                    continue
                pp_range = pp_column_ranges.get(sp, {}).get(
                    round(float(entry["xF"]), 4))
                if pp_range is None:
                    entry["model_window"] = None
                    entry["model_window_reason"] = (
                        "no pp xsect-table column at this node")
                    continue
                lo = max(float(entry["pT_lo"]), float(pp_range[0]))
                hi = min(float(entry["pT_hi"]), float(pp_range[1]))
                if lo >= hi:
                    entry["model_window"] = None
                    entry["model_window_reason"] = (
                        "empty common pT window with the pp reference "
                        "column")
                    continue
                entry["model_window"] = [lo, hi]
                entry["pT_range_pp_reference"] = [float(pp_range[0]),
                                                  float(pp_range[1])]

    system_rank = {"pPb208": 0, "C12": 1, "ppref": 2}
    species_rank = {"pi-": 0, "pi+": 1, "p": 2, "pbar": 3}
    registered_order = [
        [s, sp, xF] for s, sp, xF in sorted(
            r_plan_registry,
            key=lambda t: (system_rank[t[0]], species_rank[t[1]],
                           float(t[2])))]

    manifest = {
        "schema": PA_MANIFEST_SCHEMA,
        "card": CARD, "milestone": MILESTONE, "phase": PHASE_P3,
        "run_description": RUN_DESCRIPTION,
        "provenance": probe_provenance(config_hash),
        "m8_crosscheck": cross_check_m8_record(),
        "binning_statement": PA_BINNING_STATEMENT,
        "model_backward_xF_support": {
            "pPb208": support.get("pPb208"),
            "C12": support.get("C12"),
            "statement": (
                "the model's backward-xF support minimum per system, "
                "re-derived at the P1 probe seeds in the pilot record; the "
                "pre-registered restriction rule applies to the p+C "
                "comparison only through this recorded value"),
        },
        "pC_xF_restriction": pc_restriction_record,
        "a_scaling": {
            "xF_nodes": list(A_SCALING_XF_NODES),
            "definition": (
                "the pion A-scaling comparison at common xF across A = 1 "
                "(p+p), A = 12 (p+C), A = 208 (p+Pb): the per-event "
                "dn/dxF of each system against (1+<K>)/2 superposition, "
                "reported as the directional trend statement with the "
                "named mechanism, pions-only, minimum-bias scope"),
            "mechanism": (
                "one continuing projectile source plus K backward-shifted "
                "target sources (the asymmetric wounded-source structure), "
                "under the cumulative-excitation ansatz whose recorded "
                "limitation is the excluded de-excitation channel"),
            "directional_rule": (
                "per common node: the model and data orderings of "
                "R(A=12) versus R(A=208) must agree in direction, and the "
                "per-node superposition ratios carry the pre-registered "
                "band verdicts (the R(xF) records at the common nodes); "
                "the falsifier-4 inverted-ordering flag evaluates the "
                "model ordering against the data ordering beyond "
                "k_tension"),
        },
        "r_node_plan": {
            "definition": (
                "R(xF) = (dn/dxF)_pA / ((1+<K>_sampler)/2 (dn/dxF)_pp); "
                "the data side is the pre-registered recomputation from "
                "the ptintegrated tables (refused on mismatch); the model "
                "side counts the species particles in the pinned xF window "
                "inside the system's own data-grid pT acceptance at the "
                "node, divided by N_events times the xF bin width (the "
                "width cancels in R); the per-replica R values use the "
                "pooled production <K>"),
            "xF_window_rule": (
                "the pinned half-width convention (0.025 below xF 0.4, "
                "0.05 above), the same window on both systems of a ratio"),
            "pT_acceptance_rule": (
                "the model-side R window at a node is the intersection of "
                "the system's own data-grid column range and the pp "
                "reference column range (the same window on both sides of "
                "the model ratio, so the pT acceptance cancels inside R); "
                "each side's own published column range is recorded per "
                "node (pT_acceptance_system_column, "
                "pT_acceptance_pp_column) so the residual acceptance "
                "systematic against the published ptintegrated integrals "
                "is auditable"),
            "data_error_rule": (
                "the ptintegrated error columns: p+Pb absolute (the "
                "converted table), p+C percent (the mirror header pins "
                "'Errors are always given in percent'), p+p percent; "
                "converted to absolute and propagated through R"),
            "registered_node_order": registered_order,
            "bootstrap_seed": WS_B_BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "pooled_yield_kernel": (
            "the per-observable verdict of a cell-set observable is the "
            "pooled pre-registered kernel yield Y = sum over the cells of "
            "(pT/E) f dxF dpT on the data side and the identical form on "
            "the model side (which collapses algebraically to "
            "anchor * sum(placed counts) / (N_events * pi * sqrt(s)) "
            "because the model density carries the thesis Jacobian); the "
            "data error propagates the per-cell statistical errors through "
            "the kernel and the MC error is the replica spread"),
        "systems": systems,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    _write_json(out_dir / PA_MANIFEST_NAME, manifest)
    n_cells = sum(len(cs["cells"]) for s in systems.values()
                  for cs in s.get("cell_sets", {}).values())
    n_nodes = len(registered_order)
    print(f"[pa manifest] {n_cells} comparison cells, "
          f"{n_nodes} registered R nodes", flush=True)
    return manifest


def _ppref_node_entries(species: str, nodes: list) -> list:
    """The p+p reference node entries (the pp xsect-table columns)."""
    grid = read_points(spshadrons_data_dir() / PP_TABLE_NAMES[species])
    entries = []
    for xf in nodes:
        col = grid[np.isclose(grid[:, 0], float(xf), atol=1e-6)]
        if not len(col):
            entries.append({
                "xF": float(xf), "supported": False,
                "reason": "no pp xsect-table column at this node",
            })
            continue
        lo, hi = column_pT_range(col[:, 1], PA_WIDTH_RULE_PINNED)
        entries.append({
            "xF": float(xf),
            "xF_half_width": cell_xf_half_width(float(xf)),
            "supported": True,
            "support_source": "pp_xsect_table",
            "pT_lo": float(lo), "pT_hi": float(hi),
            "n_pT_cells": int(len(col)),
        })
    return entries


def _bin_variable_grid(xf_arr: np.ndarray, pt_arr: np.ndarray,
                       cells: list) -> tuple:
    """Exact histogram binning into the table's own variable bins (the p+C
    grids): each particle enters the first cell (in the sorted cell order)
    whose xF and pT edges bracket it; unplaced particles carry -1."""
    xf_arr = np.asarray(xf_arr, dtype=float)
    pt_arr = np.asarray(pt_arr, dtype=float)
    counts = np.zeros(len(cells), dtype=np.int64)
    assign = np.full(len(xf_arr), -1, dtype=np.int64)
    for i, cell in enumerate(cells):
        mask = ((assign < 0)
                & (xf_arr >= float(cell["lo_xF"]) - 1e-12)
                & (xf_arr <= float(cell["hi_xF"]) + 1e-12)
                & (pt_arr >= float(cell["lo_pT"]) - 1e-12)
                & (pt_arr <= float(cell["hi_pT"]) + 1e-12))
        counts[i] = int(np.count_nonzero(mask))
        assign[mask] = i
    return counts, assign


def _species_arrays(events, pdg_id: int, sqrt_s: float = SQRT_S_GEV) -> dict:
    xs, ps, evs = [], [], []
    for ev_index, event in enumerate(events):
        for particle in event.final_particles:
            if int(particle.pdg_id) != int(pdg_id):
                continue
            xs.append(particle_xf(particle, sqrt_s, event=event))
            ps.append(particle_pt(particle))
            evs.append(ev_index)
    return {
        "xf": np.asarray(xs, dtype=float),
        "pt": np.asarray(ps, dtype=float),
        "ev": np.asarray(evs, dtype=np.int64),
    }


def probe_provenance(config_hash: str) -> dict:
    return {
        "configuration_hash": config_hash,
        "config_seed": CONFIG_SEED,
        "m12_seed_base": M12_SEED_BASE,
        "milestone": MILESTONE,
        "phase": PHASE,
        "run_description": RUN_DESCRIPTION,
        "written_utc": _utc_now_iso(),
    }


def build_pa_replica_payload(seed: int, config_path: Path,
                             manifest_path: Path, system: str,
                             n_events: int = PRIMARY_EVENTS_PER_REPLICA,
                             m8_record: Path | None = None,
                             preregistration: Path | None = None,
                             events: list | None = None) -> dict:
    """One workstream-B production replica payload.

    ``system`` is one of ``pPb208``, ``C12`` (the p+A path of the staged
    runner, the baryon_nucleus stage, at the pinned nucleus fields) or
    ``ppref`` (the p+p reference sample for R(xF) at the same replica
    convention). Generation at the frozen M8 set, event.validate()
    re-asserted per event at 1e-10 (inside the generators), the spec 7.1
    source-quantum-number and antibaryon-traceability re-assertions per
    event, the p+A accounting test (spec 7.1 test 8: total event baryon
    number A+1 and charge Z+1, over the final particles and the target
    spectators) per event, the wound-count bookkeeping, first-match
    binning onto the manifest cell sets, and the per-event R(xF) node box
    counts for the pooled bootstrap. Wall fields are added by
    produce_pa_replica and are excluded from the determinism comparison.
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    embedded_sha = manifest.pop("manifest_sha256", None)
    if embedded_sha is None or canonical_json_sha256(manifest) != embedded_sha:
        raise SystemExit(
            f"p+A cells manifest embedded sha mismatch: {manifest_path}")
    if system not in manifest["systems"]:
        raise SystemExit(f"unknown system {system!r} for the p+A payload")
    m8_record_check = (cross_check_m8_record(Path(m8_record))
                       if m8_record is not None else cross_check_m8_record())
    if preregistration is not None:
        prereg = json.loads(Path(preregistration).read_text(
            encoding="utf-8"))
        cross_check_preregistration(prereg)
        config_hash = canonical_configuration_hash(config_path)
        if config_hash != prereg.get("provenance", {}).get(
                "configuration_bundled_canonical_sha256"):
            raise SystemExit(
                "bundled config hash mismatch against the pre-registration "
                "provenance; refusing generation")
    else:
        config_hash = canonical_configuration_hash(config_path)

    sys_spec = manifest["systems"][system]
    stage_label = sys_spec["stage_label"]
    if events is None:
        if system == "ppref":
            events, _summary, _wall, _applied, _beam = generate_pp_events(
                int(n_events), int(seed), config_path,
                stage_label=STAGE_LABEL_PPREF)
        else:
            events, _summary, _wall, _applied, _beam = generate_pa_events(
                int(n_events), int(seed), config_path,
                sys_spec["nucleus"], stage_label=stage_label)
    if not events:
        raise SystemExit("no events generated")

    if system == "ppref":
        A_target = Z_target = 1  # the elementary p+p framing of test 8
    else:
        nucleus = sys_spec["nucleus"]
        A_target = int(nucleus["A"])
        Z_target = int(nucleus["Z"])

    max_res = 0.0
    source_qn_violations = 0
    accounting_violations = 0
    n_antibaryons = 0
    antibaryon_violations = 0
    min_pair_margin = None
    n_spectators = 0
    wound_values: list = []
    for event in events:
        max_res = max(max_res,
                      float(event.conservation_residuals()["relative_max"]))
        for source in event.sources:
            qn = source.quantum_numbers
            ok = int(qn.B) == 1 and int(qn.S) == 0
            if source.role == "target" and system != "ppref":
                # The p+A target sources carry their target identity; the
                # elementary p+p reference sources carry (1,1,0) exactly.
                target_pdg = int(source.metadata.get("target_pdg", 0))
                ok = ok and int(qn.Q) == (1 if target_pdg == 2212 else 0)
            elif source.role == "projectile":
                ok = ok and qn.as_tuple() == (1, 1, 0)
            if not ok:
                source_qn_violations += 1
        b_sum = sum(int(p.quantum_numbers.B)
                    for p in event.final_particles)
        q_sum = sum(int(p.quantum_numbers.Q)
                    for p in event.final_particles)
        b_sum += sum(int(s.quantum_numbers.B) for s in event.spectators)
        q_sum += sum(int(s.quantum_numbers.Q) for s in event.spectators)
        n_spectators += len(event.spectators)
        if b_sum != A_target + 1 or q_sum != Z_target + 1:
            accounting_violations += 1
        for particle in event.final_particles:
            if particle.quantum_numbers.B < 0:
                n_antibaryons += 1
                source = event.sources[int(particle.source_id)]
                partner = _antibaryon_pair_margin(int(particle.pdg_id))
                margin = float(source.mass) - partner
                if margin < -ANTIBARYON_PAIR_MARGIN_TOL:
                    antibaryon_violations += 1
                min_pair_margin = (margin if min_pair_margin is None
                                   else min(min_pair_margin, margin))
        if system != "ppref":
            wound_values.append(
                float(event.metadata.get("target_wounds", 1.0)))

    species_list = (PPREF_SPECIES if system == "ppref"
                    else PA_F_SPECIES[system])
    arrays = {sp: _species_arrays(events, SPECIES_PDG[sp])
              for sp in species_list}

    cell_counts: dict = {}
    placed_any = {sp: np.zeros(len(arrays[sp]["xf"]), dtype=bool)
                  for sp in arrays} if system != "ppref" else {}
    if system != "ppref":
        for set_name in sorted(sys_spec["cell_sets"]):
            entry = sys_spec["cell_sets"][set_name]
            sp = entry["species"]
            arr = arrays[sp]
            if entry["width_rule"] == PA_WIDTH_RULE_OWN:
                counts, assign = _bin_variable_grid(arr["xf"], arr["pt"],
                                                    entry["cells"])
            else:
                cells = [(float(c["xF"]), float(c["pT"]))
                         for c in entry["cells"]]
                counts, assign = _bin_first_match(arr["xf"], arr["pt"],
                                                  cells)
            placed = assign >= 0
            placed_any[sp] |= placed
            cell_counts[set_name] = [int(v) for v in counts]

    r_node_counts: dict = {}
    r_node_per_event: dict = {}
    for sp in species_list:
        arr = arrays[sp]
        counts_sp: dict = {}
        per_event_sp: dict = {}
        for node in sys_spec["r_nodes"].get(sp, []):
            if not node.get("supported"):
                continue
            if system == "ppref":
                # The pp reference side counts in its own data-grid column
                # range; the p+A systems count in the recorded common
                # model window (the manifest's like-for-like rule).
                lo, hi = float(node["pT_lo"]), float(node["pT_hi"])
            else:
                window = node.get("model_window")
                if window is None:
                    continue
                lo, hi = float(window[0]), float(window[1])
            node_key = f"{node['xF']:.4g}"
            half = float(node.get("xF_half_width",
                                  cell_xf_half_width(float(node["xF"]))))
            mask = ((np.abs(arr["xf"] - float(node["xF"])) <= half + 1e-9)
                    & (arr["pt"] >= lo - 1e-9)
                    & (arr["pt"] <= hi + 1e-9))
            per_event = np.bincount(arr["ev"][mask], minlength=len(events))
            counts_sp[node_key] = int(per_event.sum())
            per_event_sp[node_key] = [int(v) for v in per_event]
        r_node_counts[sp] = counts_sp
        r_node_per_event[sp] = per_event_sp

    species_binning = {}
    for sp in species_list:
        n_total = int(len(arrays[sp]["xf"]))
        rec = {
            "n_particles": n_total,
            "r_node_totals": r_node_counts[sp],
            "r_node_per_event": r_node_per_event[sp],
        }
        if system != "ppref":
            rec["overflow"] = int(n_total - int(placed_any[sp].sum()))
        species_binning[sp] = rec

    wounds_summary = None
    if wound_values:
        arr_w = np.asarray(wound_values, dtype=float)
        wounds_summary = {
            "mean": float(np.mean(arr_w)),
            "std": float(np.std(arr_w, ddof=1)) if len(arr_w) > 1 else 0.0,
            "min": float(np.min(arr_w)),
            "max": float(np.max(arr_w)),
            "n_events": int(len(wound_values)),
        }
    payload = {
        "schema": (PPREF_REPLICA_SCHEMA if system == "ppref"
                   else PA_REPLICA_SCHEMA),
        "card": CARD, "milestone": MILESTONE, "phase": PHASE_P3,
        "run_description": RUN_DESCRIPTION,
        # The volatile fields (the write timestamp and the wall clocks) are
        # added by produce_pa_replica, so build_pa_replica_payload is the
        # pure, bit-identical function of (seed, config, manifest, system)
        # that the determinism re-assertion compares.
        "provenance": {
            "configuration_hash": config_hash,
            "config_seed": CONFIG_SEED,
            "m12_seed_base": M12_SEED_BASE,
            "milestone": MILESTONE,
            "phase": PHASE_P3,
            "run_description": RUN_DESCRIPTION,
            "seed": int(seed),
            "system": system,
            "stage_label": (STAGE_LABEL_PPREF if system == "ppref"
                            else STAGE_LABEL_PA),
            "helper_sha256": _sha256_file(Path(__file__)),
        },
        "m8_crosscheck": m8_record_check,
        "pa_cells_manifest_path": str(manifest_path),
        "pa_cells_manifest_sha256": _sha256_file(manifest_path),
        "nucleus": sys_spec.get("nucleus"),
        "n_events": int(len(events)),
        "mean_final_multiplicity": float(
            sum(len(event.final_particles) for event in events)
            / max(len(events), 1)),
        "conservation": {
            "n_events_validated": int(len(events)),
            "max_residual": float(max_res),
            "source_quantum_number_violations": int(source_qn_violations),
            "accounting_violations": int(accounting_violations),
            "accounting_rule": (
                f"spec 7.1 test 8: total event baryon number "
                f"{A_target + 1} and charge {Z_target + 1} per event, "
                "summed over the final particles and the target "
                "spectators"),
            "n_spectators_pooled": int(n_spectators),
            "n_antibaryons": int(n_antibaryons),
            "antibaryon_pair_violations": int(antibaryon_violations),
            "min_antibaryon_pair_margin_gev": (
                float(min_pair_margin) if min_pair_margin is not None
                else None),
        },
        "wounds": wounds_summary,
        "species_binning": species_binning,
        "cell_counts": cell_counts,
    }
    return payload


def produce_pa_replica(seed: int, config_path: Path, manifest_path: Path,
                       system: str, out_path: Path,
                       n_events: int = PRIMARY_EVENTS_PER_REPLICA,
                       m8_record: Path | None = None,
                       preregistration: Path | None = None) -> dict:
    """Build one workstream-B production replica payload and write it."""
    t0 = time.perf_counter()
    payload = build_pa_replica_payload(
        seed, config_path, manifest_path, system, n_events=n_events,
        m8_record=m8_record, preregistration=preregistration)
    wall = time.perf_counter() - t0
    payload["wall_s"] = float(wall)
    payload["ms_per_event"] = float(
        1000.0 * wall / max(int(payload["n_events"]), 1))
    payload["provenance"]["written_utc"] = _utc_now_iso()
    _write_json(Path(out_path), payload)
    print(f"[pa replica {system} seed {seed}] {payload['n_events']} events, "
          f"wall {wall:.1f}s, max residual "
          f"{payload['conservation']['max_residual']:.3e}", flush=True)
    return payload


def run_pa_determinism_production(out_dir: Path, config_path: Path,
                                  manifest_path: Path,
                                  preregistration: Path | None = None,
                                  m8_record: Path | None = None) -> dict:
    """Same-environment double generation at production scale on the p+A
    path: the replica-0 payload of each system (p+Pb208, C12, and the p+p
    reference) is built twice at its workstream-B replica seed and the two
    payloads must be bit-identical (the spec 7.1 reproducibility test 5 at
    production scale; the M10 probe_A pattern as executed in P2)."""
    out_dir = Path(out_dir)
    config_hash = canonical_configuration_hash(config_path)
    units = {}
    for system, seed in (("pPb208", int(WS_B_REPLICA_SEEDS[0])),
                         ("C12", int(WS_B_REPLICA_SEEDS[0])),
                         ("ppref", int(WS_B_PPREF_REPLICA_SEEDS[0]))):
        t0 = time.perf_counter()
        payload_a = build_pa_replica_payload(
            seed, config_path, manifest_path, system,
            m8_record=m8_record, preregistration=preregistration)
        payload_b = build_pa_replica_payload(
            seed, config_path, manifest_path, system,
            m8_record=m8_record, preregistration=preregistration)
        wall = time.perf_counter() - t0
        canon_a = json.dumps(_to_jsonable(payload_a), sort_keys=True,
                             separators=(",", ":"))
        canon_b = json.dumps(_to_jsonable(payload_b), sort_keys=True,
                             separators=(",", ":"))
        units[system] = {
            "seed": int(seed),
            "n_events": int(payload_a["n_events"]),
            "bit_identical": bool(canon_a == canon_b),
            "payload_sha256": [hashlib.sha256(canon_a.encode()).hexdigest(),
                               hashlib.sha256(canon_b.encode()).hexdigest()],
            "wall_s": float(wall),
        }
        print(f"[determinism production pa {system}] "
              f"bit_identical={units[system]['bit_identical']}", flush=True)
    record = {
        "schema": PA_DETERMINISM_SCHEMA,
        "probe": "determinism_production_pa",
        "pattern": "M10 probe_A at production scale, same environment",
        "provenance": {**probe_provenance(config_hash)},
        "n_events": PRIMARY_EVENTS_PER_REPLICA,
        "n_repeats": 2,
        "units": units,
        "deterministic": all(u["bit_identical"] for u in units.values()),
        "written_utc": _utc_now_iso(),
    }
    _write_json(out_dir / PA_DETERMINISM_NAME, record)
    print(f"[determinism production pa] deterministic={record['deterministic']}",
          flush=True)
    return record


def run_pilot_pins(out_dir: Path, config_path: Path, prereg: dict) -> dict:
    """The pre-production pilot: the regression pins of the p+A machinery
    (foundations section 5) plus the C12 NucleusSpec construction pin, the
    model backward-xF support re-derivation at the P1 probe seeds, the
    pilot-scale determinism re-assertion, and the P1 probe-expectation
    re-assertions. Production is refused (SystemExit) on any missed pin
    without a recorded deviation."""
    out_dir = Path(out_dir)
    cross_check_preregistration(prereg)
    config_hash = canonical_configuration_hash(config_path)
    if str(MIXING2006_DIR) not in sys.path:
        sys.path.insert(0, str(MIXING2006_DIR))
    from na49_mixing.glauber import NucleusSpec, WoundedNucleonSampler

    # Pin 1: the C12 NucleusSpec construction against the pre-registration
    # pin (the P1 field values from glauber.py).
    spec_c12 = NucleusSpec(**GEOMETRY_C12)
    constructed = {k: getattr(spec_c12, k) for k in GEOMETRY_C12}
    pin_c12 = {
        "constructed_fields": _to_jsonable(constructed),
        "matches_harness_constants": constructed == GEOMETRY_C12,
        "matches_preregistration": constructed == prereg["geometry"]["C12"],
        "resolved_profile": spec_c12.resolved_profile(),
        "holds": bool(constructed == GEOMETRY_C12
                      == prereg["geometry"]["C12"]
                      and spec_c12.resolved_profile() == "harmonic_oscillator"),
    }

    # Pin 2: the rare-slice regression pins (foundations section 5):
    # Pb208 single-bin slices K = 21, 24, 25 with the recorded expected
    # min-bias fractions, one event each through the chunked worker at the
    # recorded regression-test seeds.
    from na49_mixing.parallel import GenerationConfig, generate_tracks_chunked

    pb_spec = NucleusSpec(**GEOMETRY_PB208)
    analytic = WoundedNucleonSampler(pb_spec, np.random.default_rng(0),
                                     dr_fm=0.05)
    delta_gen = str(MIXING2006_DIR / "na49_mixing" / "generators" / "michal"
                    / "deltas-to-pion" / "delta++.gen")
    rare_records = []
    for k in RARE_SLICE_KS:
        fraction = float(analytic.fraction_for_wound_range(k, k))
        expected = RARE_SLICE_FRACTIONS[k]
        out = generate_tracks_chunked(
            GenerationConfig(
                gen_file=delta_gen,
                n_events=1,
                seed=RARE_SLICE_SEED_BASE + k,
                beam_momentum=158.0,
                max_decay_passes=1,
                fragmentation_mode="both",
                nucleus=pb_spec,
                target_wound_range=(k, k),
            ),
            n_workers=1, n_chunks=1,
        )
        wounds = np.asarray(out["n_wounds"], dtype=np.int64)
        rare_records.append({
            "k": int(k),
            "expected_min_bias_fraction": float(expected),
            "sampler_fraction": fraction,
            "fraction_matches": bool(math.isclose(
                fraction, expected, rel_tol=0.05)),
            "one_event_generated": True,
            "wound_track_count": int(wounds.size),
            "wounds_all_k": bool(np.all(wounds == k)),
            "holds": bool(math.isclose(fraction, expected, rel_tol=0.05)
                          and np.all(wounds == k)),
        })

    # Pin 3: the point-target regression pin (A = 1 consumes no sampler
    # randomness; the same-seed p+A stream is bit-identical to the p+p
    # chunked-worker stream).
    base = dict(
        gen_file=delta_gen,
        n_events=POINT_TARGET_EVENTS,
        seed=POINT_TARGET_SEED,
        beam_momentum=158.0,
        max_decay_passes=2,
        use_soft_decay_factory=True,
        fragmentation_mode="both",
    )
    pp_chunk = generate_tracks_chunked(GenerationConfig(**base),
                                       n_workers=1, n_chunks=2)
    pa_chunk = generate_tracks_chunked(
        GenerationConfig(**base, nucleus=NucleusSpec(name="p", A=1, Z=1)),
        n_workers=1, n_chunks=2)
    point_target = {
        "n_events": int(POINT_TARGET_EVENTS),
        "seed": int(POINT_TARGET_SEED),
        "streams_bit_identical": bool(all(
            np.array_equal(pp_chunk[key], pa_chunk[key])
            for key in ("pdg", "xf", "pt", "source_side"))),
        "pa_wounds_all_one": bool(np.all(
            np.asarray(pa_chunk["n_wounds"], dtype=np.int64) >= 1)),
        "holds": bool(all(
            np.array_equal(pp_chunk[key], pa_chunk[key])
            for key in ("pdg", "xf", "pt", "source_side"))),
    }

    # Pin 4: the model backward-xF support re-derived at the P1 probe seeds
    # (the Risk-2 pin of the pre-registration's model_xF_support_pin).
    pb_events, _s1, _w1, _a1, _b1 = generate_pa_events(
        DETERMINISM_PA_EVENTS, PROBE_SEED_PA_DETERMINISM, config_path,
        GEOMETRY_PB208)
    pc_events, _s2, _w2, _a2, _b2 = generate_pa_events(
        DETERMINISM_PA_EVENTS, PROBE_SEED_PA_DETERMINISM + 10, config_path,
        GEOMETRY_C12)
    support = {}
    for label, evs in (("pPb208", pb_events), ("C12", pc_events)):
        xs = [particle_xf(p, event=event)
              for event in evs for p in event.final_particles
              if int(particle_pdg(p)) in (-211, 211)]
        support[label] = {
            "min_pion_xF": float(min(xs)) if xs else None,
            "n_pions": int(len(xs)),
            "n_events": int(len(evs)),
            "seed": int(PROBE_SEED_PA_DETERMINISM if label == "pPb208"
                        else PROBE_SEED_PA_DETERMINISM + 10),
        }
    support["statement"] = (
        "the model's backward-xF support minimum per system at the P1 "
        "probe seeds (the pilot samples are bit-identical to the P1 probe "
        "statistic vectors by the verified determinism); the pre-registered "
        "p+C restriction applies only if the model carries no support at "
        "xF < 0")

    # Pin 5: the pilot-scale determinism re-assertion (five bit-identical
    # repeated evaluations, both statistic pipelines).
    determinism = run_probe_determinism(
        out_dir, config_path, prereg, out_name="probe_determinism_p3.json")

    # Pin 6: the P1 probe expectations re-asserted from the recorded probes
    # (the pre-registered expectations: every cost cell completed with a
    # finite ms/event, a conservation residual <= 1e-10, and a menu
    # entry fitting the budget; the P1 determinism probe was
    # bit-identical).
    p1_expectations = {}
    cost_path = out_dir / "probe_cost_cells.json"
    if cost_path.exists():
        cost = json.loads(cost_path.read_text(encoding="utf-8"))
        cost_cells = cost.get("cells", {})
        residuals = [float(c.get("max_conservation_residual", 1.0))
                     for c in cost_cells.values()]
        costs = [float(c.get("ms_per_event", float("nan")))
                 for c in cost_cells.values()]
        p1_expectations["cost_cells"] = {
            "expectation_met": bool(
                all(math.isfinite(v) for v in costs)
                and all(v <= 1e-10 for v in residuals)
                and cost.get("menu_decision", {}).get(
                    "pa_pPb208", {}).get("chosen_events_per_replica")
                == PRIMARY_EVENTS_PER_REPLICA
                and cost.get("menu_decision", {}).get(
                    "pa_C12", {}).get("chosen_events_per_replica")
                == PRIMARY_EVENTS_PER_REPLICA),
            "menu_pPb208": cost.get("menu_decision", {}).get(
                "pa_pPb208", {}).get("chosen_events_per_replica"),
            "menu_C12": cost.get("menu_decision", {}).get(
                "pa_C12", {}).get("chosen_events_per_replica"),
            "max_conservation_residual": (max(residuals) if residuals
                                          else None),
            "ms_per_event_pPb208": cost_cells.get(
                "pa_pPb208", {}).get("ms_per_event"),
            "ms_per_event_C12": cost_cells.get(
                "pa_C12", {}).get("ms_per_event"),
        }
    det1_path = out_dir / "probe_determinism.json"
    if det1_path.exists():
        det1 = json.loads(det1_path.read_text(encoding="utf-8"))
        p1_expectations["determinism_p1"] = {
            "deterministic": det1.get("deterministic"),
            "pa_max_residual": det1.get("units", {}).get(
                "pa", {}).get("max_conservation_residual"),
        }
    cost_rec = p1_expectations.get("cost_cells", {})
    p1_holds = bool(
        cost_rec.get("expectation_met") is True
        and cost_rec.get("menu_pPb208") == PRIMARY_EVENTS_PER_REPLICA
        and cost_rec.get("menu_C12") == PRIMARY_EVENTS_PER_REPLICA
        and p1_expectations.get("determinism_p1", {}).get("deterministic")
        is True)
    p1_expectations["holds"] = p1_holds

    all_holds = bool(
        pin_c12["holds"] and all(r["holds"] for r in rare_records)
        and point_target["holds"] and determinism["deterministic"]
        and p1_holds)
    record = {
        "schema": PILOT_SCHEMA,
        "probe": "pilot_pins",
        "provenance": probe_provenance(config_hash),
        "c12_spec_pin": pin_c12,
        "rare_slice_pins": rare_records,
        "point_target_pin": point_target,
        "model_backward_xF_support": support,
        "determinism_re_assertion": {
            "artifact": "probe_determinism_p3.json",
            "deterministic": determinism["deterministic"],
        },
        "p1_expectation_re_assertions": p1_expectations,
        "all_pins_hold": all_holds,
        "refusal_rule": (
            "production is refused on a missed expectation without a "
            "recorded deviation (the card rule); the pilot precedes the "
            "cells manifest and the production launch"),
        "written_utc": _utc_now_iso(),
    }
    _write_json(out_dir / PILOT_RECORD_NAME, record)
    print(f"[pilot pins] all_pins_hold={all_holds}", flush=True)
    if not all_holds:
        raise SystemExit(
            "pilot pin expectation missed; production refused without a "
            "recorded deviation")
    return record


def _refuse_r_rows_vs_prereg(system: str, species: str, entries: list,
                             prereg: dict) -> None:
    """Refuse when the recomputed R(xF) data rows differ from the
    pre-registered rows (pure table arithmetic; the pre-registration is
    authoritative)."""
    data_key = PA_SYSTEM_DATA_KEY[system]
    registered = prereg["protocol_B"]["R_xF_data_side"][data_key][species]
    if len(entries) != len(registered):
        raise SystemExit(
            f"R(xF) row count mismatch for {system}/{species}: "
            f"{len(entries)} recomputed vs {len(registered)} registered")
    for entry, reg in zip(entries, registered):
        for key in ("xF", "R_samplerK", "R_measuredAnchor",
                    "anchor_deviation", "dn_dxF_pa", "dn_dxF_err",
                    "dn_dxF_pp"):
            actual = entry.get(
                {"xF": "xF", "R_samplerK": "data_R_samplerK",
                 "R_measuredAnchor": "data_R_measuredAnchor",
                 "anchor_deviation": "anchor_deviation",
                 "dn_dxF_pa": "data_dn_dxF_pa",
                 "dn_dxF_err": "data_dn_dxF_err_raw",
                 "dn_dxF_pp": "data_dn_dxF_pp"}[key])
            if not math.isclose(float(actual), float(reg[key]),
                                rel_tol=1e-12, abs_tol=1e-12):
                raise SystemExit(
                    f"R(xF) data row mismatch for {system}/{species} at "
                    f"xF {entry.get('xF')}: field {key} recomputed "
                    f"{actual!r} != pre-registered {reg[key]!r}")


def _pooled_K(arts: list) -> dict:
    """The pooled wound-count bookkeeping of one p+A system: the
    event-weighted mean and the combined standard deviation of the
    per-replica wound records."""
    total_events = sum(int(a["n_events"]) for a in arts)
    mean = (sum(float(a["wounds"]["mean"]) * int(a["n_events"])
                for a in arts) / total_events)
    var = (sum((float(a["wounds"]["std"]) ** 2
                + (float(a["wounds"]["mean"]) - mean) ** 2)
               * int(a["n_events"]) for a in arts) / total_events)
    return {
        "pooled_mean": float(mean),
        "pooled_std": float(math.sqrt(var)),
        "n_events": int(total_events),
        "per_replica_means": [float(a["wounds"]["mean"]) for a in arts],
    }


def collect_pa_trends_report(run_dir: Path, config_path: Path,
                             prereg_path: Path,
                             gateway_manifest_path: Path | None = None) -> dict:
    """The P3 collection: re-verify the frozen set and the pre-registration,
    load the eight p+Pb, eight p+C, and eight p+p-reference replica
    artifacts, re-derive every data-side value from the tables (refusing
    any mismatch against the pre-registered values), evaluate the
    pre-registered verdict bands per cell, per R(xF) node, and per pooled
    kernel yield, run the pooled bootstrap over the registered node order,
    evaluate the geometry-anchor and wound-count cross-checks and the
    falsifier-4 directions, and write pa_trends_report.json."""
    run_dir = Path(run_dir)
    prereg = json.loads(Path(prereg_path).read_text(encoding="utf-8"))
    cross_check_preregistration(prereg)
    m8 = cross_check_m8_record()
    config_hash = canonical_configuration_hash(config_path)
    if config_hash != prereg.get("provenance", {}).get(
            "configuration_bundled_canonical_sha256"):
        raise SystemExit(
            "bundled config hash mismatch against the pre-registration "
            "provenance; refusing the p+A collection")

    manifest_path = run_dir / PA_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = _sha256_file(manifest_path)
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if canonical_json_sha256(body) != manifest.get("manifest_sha256"):
        raise SystemExit("p+A cells manifest embedded sha mismatch")

    replica_specs = {
        "pPb208": (WS_B_REPLICA_SEEDS, PA_REPLICA_SCHEMA),
        "C12": (WS_B_REPLICA_SEEDS, PA_REPLICA_SCHEMA),
        "ppref": (WS_B_PPREF_REPLICA_SEEDS, PPREF_REPLICA_SCHEMA),
    }
    arts: dict = {}
    for system, (seeds, schema) in replica_specs.items():
        arts[system] = []
        for seed in seeds:
            name = (f"ppref_replica_{seed}.json" if system == "ppref"
                    else f"pa_replica_{system}_{seed}.json")
            path = run_dir / name
            if not path.exists():
                raise SystemExit(f"missing p+A replica artifact {path}")
            art = json.loads(path.read_text(encoding="utf-8"))
            if int(art["provenance"]["seed"]) != int(seed):
                raise SystemExit(f"replica seed mismatch in {path}")
            if art.get("provenance", {}).get("system") != system:
                raise SystemExit(f"replica system mismatch in {path}")
            if art.get("schema") != schema:
                raise SystemExit(f"replica schema mismatch in {path}")
            if art.get("pa_cells_manifest_sha256") != manifest_sha:
                raise SystemExit(f"p+A cells manifest sha mismatch in {path}")
            helper_sha = art["provenance"].get("helper_sha256")
            if helper_sha and helper_sha != _sha256_file(Path(__file__)):
                raise SystemExit(
                    f"helper sha mismatch in {path}: the artifact was "
                    "produced by a different helper version")
            arts[system].append(art)

    # Conservation is inviolable: refuse the collection on any violation.
    conservation_by_system = {}
    for system, system_arts in arts.items():
        total = sum(int(a["n_events"]) for a in system_arts)
        max_res = max(float(a["conservation"]["max_residual"])
                      for a in system_arts)
        qn_v = sum(int(a["conservation"]["source_quantum_number_violations"])
                   for a in system_arts)
        acc_v = sum(int(a["conservation"]["accounting_violations"])
                    for a in system_arts)
        ab_v = sum(int(a["conservation"]["antibaryon_pair_violations"])
                   for a in system_arts)
        if max_res > 1e-10 or qn_v or acc_v or ab_v:
            raise SystemExit(
                f"conservation violation in the {system} production "
                "samples; refusing the p+A collection")
        conservation_by_system[system] = {
            "n_events": int(total),
            "max_residual": float(max_res),
            "source_quantum_number_violations": int(qn_v),
            "accounting_violations": int(acc_v),
            "antibaryon_pair_violations": int(ab_v),
            "min_antibaryon_pair_margin_gev": min(
                float(a["conservation"]["min_antibaryon_pair_margin_gev"])
                for a in system_arts
                if a["conservation"]["min_antibaryon_pair_margin_gev"]
                is not None),
            "accounting_rule": system_arts[0]["conservation"]["accounting_rule"],
            "n_spectators_pooled": sum(
                int(a["conservation"]["n_spectators_pooled"])
                for a in system_arts),
        }

    # Geometry anchors (recomputed) and the wound-count bookkeeping.
    geo = geometry_anchors()
    for system in PA_SYSTEMS:
        label = "Pb208" if system == "pPb208" else "C12"
        analytic = float(geo[label]["mean_target_wounds"])
        recorded = float(manifest["systems"][system][
            "analytic_mean_target_wounds"])
        if not math.isclose(analytic, recorded, rel_tol=1e-12, abs_tol=1e-12):
            raise SystemExit(
                f"analytic <K> drift for {system}: {analytic!r} != "
                f"{recorded!r} (the manifest record)")
    wound_bookkeeping = {}
    for system in PA_SYSTEMS:
        label = "Pb208" if system == "pPb208" else "C12"
        pooled = _pooled_K(arts[system])
        analytic = float(geo[label]["mean_target_wounds"])
        se = pooled["pooled_std"] / math.sqrt(pooled["n_events"])
        z_consistency = ((pooled["pooled_mean"] - analytic) / se
                         if se else None)
        wound_bookkeeping[system] = {
            "pooled": pooled,
            "analytic_mean_target_wounds": analytic,
            "consistency_z": z_consistency,
        }

    # Data-side refusal: every manifest cell value and R row re-derived
    # from the tables (the pre-registration is authoritative).
    for system in PA_SYSTEMS:
        sys_entry = manifest["systems"][system]
        for set_name, entry in sys_entry["cell_sets"].items():
            sp = entry["species"]
            if set_name.startswith("density_"):
                table = read_points(pa_data_dir() / PPB_DENSITY_NAMES[sp])
                index = {(round(float(r[0]), 4), round(float(r[1]), 4)):
                         (float(r[2]), float(r[3])) for r in table}
                for cell in entry["cells"]:
                    key = (round(float(cell["xF"]), 4),
                           round(float(cell["pT"]), 4))
                    value, err = index[key]
                pairs = [(round(float(c["xF"]), 4), round(float(c["pT"]), 4))
                         for c in entry["cells"]]
            elif system == "pPb208":
                table = read_points(pa_data_dir() / PPB_XSECT_NAMES[sp])
                pt_dis = fine_pt_labels(table[:, 0].copy(),
                                        table[:, 1].copy())
                index = {}
                for i in range(len(table)):
                    pt = round(float(pt_dis[i]), 4)
                    if pt <= PPB_F_PT_NODE_MIN:
                        continue
                    index[(round(float(table[i, 0]), 4), pt)] = (
                        float(table[i, 2]), float(table[i, 3]))
                pairs = [(round(float(c["xF"]), 4), round(float(c["pT"]), 4))
                         for c in entry["cells"]]
            else:
                table = read_points(
                    spshadrons_data_dir() / PC_XSECT_NAMES[sp])
                index = {(round(float(r[0]), 4), round(float(r[1]), 4)):
                         (float(r[2]), float(r[3])) for r in table}
                pairs = [(round(float(c["xF"]), 4),
                          round(float(c["pT"]), 4))
                         for c in entry["cells"]]
            if len(pairs) != len(entry["cells"]):
                raise SystemExit(
                    f"cell count mismatch for {system}/{set_name}")
            for cell, key in zip(entry["cells"], pairs):
                if key not in index:
                    raise SystemExit(
                        f"cell {key} of {system}/{set_name} missing from "
                        "the table at collection")
                value, err = index[key]
                if (not math.isclose(float(cell["data_value"]), value,
                                     rel_tol=1e-12, abs_tol=1e-12)
                        or not math.isclose(float(cell["data_err"]), err,
                                            rel_tol=1e-12, abs_tol=1e-12)):
                    raise SystemExit(
                        f"cell data value mismatch for {system}/{set_name} "
                        f"at {key}: manifest {cell['data_value']!r} vs "
                        f"table {value!r}")
        for sp, entries in sys_entry["r_nodes"].items():
            analytic = float(sys_entry["analytic_mean_target_wounds"])
            fresh = _r_node_entries(system, sp, analytic)
            _refuse_r_rows_vs_prereg(system, sp, fresh, prereg)
            # The manifest entries carry the derived model-window keys of
            # the post-pass; the fresh plan is compared against the
            # pre-post-pass form.
            stored = [{k: v for k, v in entry.items()
                       if k not in ("model_window", "model_window_reason",
                                    "pT_range_pp_reference")}
                      for entry in entries]
            if fresh != stored:
                raise SystemExit(
                    f"R(xF) node plan drift for {system}/{sp} against the "
                    "tables")

    geometry_anchors_record = {
        "definition": (
            "sigma_pA and <K> from the sampler at the pinned geometry "
            "(a sanity cross-check, not a gate), against the measured "
            "anchors sigma_inel(p+Pb) = 1806 +/- 25 mb with <nu> 3.65 and "
            "sigma_inel(p+C) = 226 +/- 7 mb with <nu> 1.69"),
        "systems": {},
        "systematic_statement": (
            "the sigma_NN/Woods-Saxon choice is the systematic on the "
            "wounded-scaling interpretation: using the sampler <K> instead "
            "of the measured <nu> rescales R uniformly by "
            "(1+<nu>)/(1+<K>)"),
        "tower_model_era_context": (
            "the tower-model-era reference values (sigma_pA 1686 mb, "
            "-6.7%; <K> 3.93, +7.7%) are quoted as context only and are "
            "never staged-model results"),
    }
    for system in PA_SYSTEMS:
        label = "Pb208" if system == "pPb208" else "C12"
        measured_sigma = (SIGMA_INEL_PPb_MB if system == "pPb208"
                          else SIGMA_INEL_PC_MB)
        measured_nu = (NU_PPB_MEASURED if system == "pPb208"
                       else NU_PC_MEASURED)
        sampler_sigma = float(geo[label]["sigma_pA_mb"])
        sampler_K = float(geo[label]["mean_target_wounds"])
        geometry_anchors_record["systems"][system] = {
            "sampler_sigma_pA_mb": sampler_sigma,
            "sampler_mean_target_wounds": sampler_K,
            "measured_sigma_inel_mb": measured_sigma,
            "measured_nu": measured_nu,
            "sigma_pA_percent_deviation": (
                100.0 * (sampler_sigma - measured_sigma) / measured_sigma),
            "K_percent_deviation": (
                100.0 * (sampler_K - measured_nu) / measured_nu),
            "anchor_rescale_percent": (
                100.0 * ((1.0 + measured_nu) / (1.0 + sampler_K) - 1.0)),
        }

    # --- cell-set observables ------------------------------------------------
    anchors = {"pPb208": SIGMA_INEL_PPb_MB, "C12": SIGMA_INEL_PC_MB}
    cell_observables: dict = {}
    for system in PA_SYSTEMS:
        for set_name in sorted(manifest["systems"][system]["cell_sets"]):
            entry = dict(
                manifest["systems"][system]["cell_sets"][set_name])
            entry["set_name"] = set_name
            entry["anchor_mb"] = (1.0 if set_name.startswith("density_")
                                  else anchors[system])
            role = ("primary_per_event_density"
                    if set_name.startswith("density_")
                    else ("secondary_f_with_anchor"
                          if system == "pPb208" else "primary_f_with_anchor"))
            rec = _pa_cell_set_records(entry, arts[system], set_name)
            rec["system"] = system
            rec["role"] = role
            rec["n_replicas"] = len(arts[system])
            cell_observables[f"{system}.{set_name}"] = rec

    # --- R(xF) ----------------------------------------------------------------
    n_pp_events = int(arts["ppref"][0]["n_events"])
    pooled_K = {system: _pooled_K(arts[system]) for system in PA_SYSTEMS}
    rng = np.random.default_rng(WS_B_BOOTSTRAP_SEED)
    r_xf_records: dict = {system: {} for system in PA_SYSTEMS}
    registered_bootstrap_order: list = []
    for system in PA_SYSTEMS:
        for sp, entries in manifest["systems"][system]["r_nodes"].items():
            node_records = []
            for entry in entries:
                xF = float(entry["xF"])
                node_key = f"{xF:.4g}"
                if not entry.get("supported"):
                    node_records.append({
                        "xF": xF, "verdict": "no_model_support",
                        "reason": entry.get(
                            "reason", "node unsupported in the data grid"),
                        "data_R": entry.get("data_R_samplerK"),
                    })
                    continue
                pp_entries = manifest["systems"]["ppref"]["r_nodes"][sp]
                pp_entry = next((e for e in pp_entries
                                 if math.isclose(float(e["xF"]), xF,
                                                 abs_tol=1e-9)), None)
                if pp_entry is None or not pp_entry.get("supported"):
                    node_records.append({
                        "xF": xF, "verdict": "no_model_support",
                        "reason": "no pp reference support at this node",
                        "data_R": entry.get("data_R_samplerK"),
                    })
                    continue
                if entry.get("model_window") is None:
                    node_records.append({
                        "xF": xF, "verdict": "no_model_support",
                        "reason": entry.get(
                            "model_window_reason",
                            "no common model pT window at this node"),
                        "data_R": entry.get("data_R_samplerK"),
                    })
                    continue
                registered_bootstrap_order.append([system, sp, xF])
                K_pool = pooled_K[system]["pooled_mean"]
                n_w = (1.0 + K_pool) / 2.0
                per_replica_R = []
                for r_idx, art in enumerate(arts[system]):
                    n_node_pa = int(art["species_binning"][sp][
                        "r_node_totals"][node_key])
                    n_node_pp = int(
                        arts["ppref"][r_idx]["species_binning"][sp][
                            "r_node_totals"][node_key])
                    per_replica_R.append(
                        (n_node_pa / float(art["n_events"]))
                        / (n_w * (n_node_pp / float(n_pp_events)))
                        if n_node_pp else float("nan"))
                finite = [v for v in per_replica_R if math.isfinite(v)]
                model_R = float(np.mean(finite)) if finite else float("nan")
                mc = (float(np.std(finite, ddof=1)) if len(finite) > 1
                      else float("nan"))
                data_R = float(entry["data_R_samplerK"])
                data_err = float(_pa_r_data_error(
                    {**entry, "species": sp}, PA_SYSTEM_DATA_KEY[system]))
                sigma_comb = math.sqrt(
                    data_err**2 + (mc if math.isfinite(mc) else 0.0)**2)
                delta = model_R - data_R
                z = (delta / sigma_comb) if sigma_comb else float("nan")
                node_records.append({
                    "xF": xF,
                    "xF_window_half_width": float(
                        pp_entry["xF_half_width"]),
                    "pT_acceptance_model_window": list(
                        entry["model_window"]),
                    "pT_acceptance_system_column": [
                        entry["pT_lo"], entry["pT_hi"]],
                    "pT_acceptance_pp_column": [
                        pp_entry["pT_lo"], pp_entry["pT_hi"]],
                    "pooled_K_used": K_pool,
                    "n_replicas": len(finite),
                    "per_replica_R": per_replica_R,
                    "model_R": model_R,
                    "mc_err_replica_spread": mc,
                    "data_R": data_R,
                    "data_R_measuredAnchor": float(
                        entry["data_R_measuredAnchor"]),
                    "anchor_deviation": float(entry["anchor_deviation"]),
                    "data_err_propagated": data_err,
                    "sigma_combined": sigma_comb,
                    "delta": delta,
                    "delta_over_sigma_combined": (z if sigma_comb
                                                  else None),
                    "verdict": (verdict_from_z(z) if sigma_comb
                                else "no_model_support"),
                })
            r_xf_records[system][sp] = node_records

    # The pooled bootstrap over the registered node order (one generator
    # consumed sequentially; the M9 convention, 200 resamples per node).
    for system, sp, xF in registered_bootstrap_order:
        node_key = f"{float(xF):.4g}"
        rec = next(r for r in r_xf_records[system][sp]
                   if math.isclose(float(r["xF"]), float(xF), abs_tol=1e-9)
                   and "verdict" in r and r.get("reason") is None)
        num_parts, den_parts = [], []
        for art in arts[system]:
            arr = art["species_binning"][sp]["r_node_per_event"][node_key]
            num_parts.append(np.asarray(arr, dtype=float))
        for art in arts["ppref"]:
            arr = art["species_binning"][sp]["r_node_per_event"][node_key]
            den_parts.append(np.asarray(arr, dtype=float))
        num = np.concatenate(num_parts)
        den = np.concatenate(den_parts)
        K_pool = pooled_K[system]["pooled_mean"]
        n_w = (1.0 + K_pool) / 2.0
        vals = np.empty(int(BOOTSTRAP_REPLICATES), dtype=float)
        invalid = 0
        for b in range(int(BOOTSTRAP_REPLICATES)):
            idx = rng.integers(0, num.size, num.size)
            n_num = float(num[idx].sum())
            n_den = float(den[idx].sum())
            if n_den > 0:
                vals[b] = ((n_num / num.size)
                           / (n_w * (n_den / den.size)))
            else:
                vals[b] = float("nan")
                invalid += 1
        finite_b = vals[np.isfinite(vals)]
        lo, hi = ((float(np.percentile(finite_b, 2.5)),
                   float(np.percentile(finite_b, 97.5)))
                  if finite_b.size else (float("nan"), float("nan")))
        rec["bootstrap"] = {
            "replicates": int(BOOTSTRAP_REPLICATES),
            "seed": int(WS_B_BOOTSTRAP_SEED),
            "invalid_replicates": int(invalid),
            "percentile_interval": [lo, hi],
        }

    # --- A-scaling (pions, A = 1, 12, 208, at common xF) ----------------------
    a_scaling_records: dict = {}
    for sp in ("pi-", "pi+"):
        pPb_by_xF = {round(float(r["xF"]), 4): r
                     for r in r_xf_records["pPb208"].get(sp, [])}
        pC_by_xF = {round(float(r["xF"]), 4): r
                    for r in r_xf_records["C12"].get(sp, [])}
        nodes = []
        for xF in A_SCALING_XF_NODES:
            key = round(float(xF), 4)
            r_pPb = pPb_by_xF.get(key)
            r_pC = pC_by_xF.get(key)
            if (r_pPb is None or r_pC is None
                    or r_pPb.get("verdict") == "no_model_support"
                    or r_pC.get("verdict") == "no_model_support"):
                nodes.append({
                    "xF": float(xF), "verdict": "no_model_support",
                    "reason": "no model R support at this common node",
                })
                continue
            model_diff = r_pC["model_R"] - r_pPb["model_R"]
            data_diff = (float(r_pC["data_R"])
                         - float(r_pPb["data_R"]))
            sigma_diff = math.sqrt(
                (float(r_pC["mc_err_replica_spread"])
                 if math.isfinite(float(r_pC["mc_err_replica_spread"]))
                 else 0.0) ** 2
                + (float(r_pPb["mc_err_replica_spread"])
                   if math.isfinite(float(r_pPb["mc_err_replica_spread"]))
                   else 0.0) ** 2
                + float(r_pC["data_err_propagated"]) ** 2
                + float(r_pPb["data_err_propagated"]) ** 2)
            ordering_z = ((model_diff - data_diff) / sigma_diff
                          if sigma_diff else None)
            direction_agrees = ((model_diff > 0) == (data_diff > 0))
            nodes.append({
                "xF": float(xF),
                "data_R_C12": r_pC["data_R"],
                "data_R_pPb208": r_pPb["data_R"],
                "model_R_C12": r_pC["model_R"],
                "model_R_pPb208": r_pPb["model_R"],
                "model_ordering": float(model_diff),
                "data_ordering": float(data_diff),
                "ordering_z": ordering_z,
                "direction_agrees": bool(direction_agrees),
                "verdicts": {"C12": r_pC["verdict"],
                             "pPb208": r_pPb["verdict"]},
            })
        n_agree = sum(1 for n in nodes if n.get("direction_agrees"))
        n_total_dir = sum(1 for n in nodes if "direction_agrees" in n)
        a_scaling_records[sp] = {
            "nodes": nodes,
            "direction_agreements": int(n_agree),
            "direction_nodes": int(n_total_dir),
            "directional_verdict": (
                "direction_agrees" if n_total_dir
                and n_agree == n_total_dir else "direction_disagrees"),
            "mechanism": (
                "one continuing projectile source plus K backward-shifted "
                "target sources (the asymmetric wounded-source structure), "
                "under the cumulative-excitation ansatz whose recorded "
                "limitation is the excluded de-excitation channel"),
            "scope": "pions only, minimum bias, A = 1, 12, 208",
        }

    # --- falsifier cross-reference (spec 7.3 falsifier 4) ----------------------
    falsifiers = pa_falsifier4_cross_reference(r_xf_records,
                                               a_scaling_records)

    # --- wall and budget accounting ---------------------------------------------
    wall_accounting = {"systems": {}}
    for system, system_arts in arts.items():
        walls = [float(a.get("wall_s", 0.0)) for a in system_arts]
        ms = [float(a.get("ms_per_event", 0.0)) for a in system_arts]
        projection = {"pPb208": 837.4377056024969, "C12": 401.18365678936243,
                      "ppref": 171.3055981323123}[system]
        wall_accounting["systems"][system] = {
            "n_replicas": len(system_arts),
            "events_per_replica": int(system_arts[0]["n_events"]),
            "total_events": sum(int(a["n_events"]) for a in system_arts),
            "wall_s_per_replica": walls,
            "total_wall_s": float(sum(walls)),
            "ms_per_event_per_replica": ms,
            "mean_ms_per_event": float(np.mean(ms)),
            "pre_registered_projection_serial_equivalent_s": projection,
            "source": "the P1 cost probe menu decision "
                      "(probe_cost_cells.json)",
        }
    wall_accounting["per_point_budget_s"] = MAX_PROJECTED_WALL_S
    wall_accounting["declared_reduction"] = None
    wall_accounting["gateway_manifest"] = (json.loads(
        Path(gateway_manifest_path).read_text(encoding="utf-8"))
        if gateway_manifest_path and Path(gateway_manifest_path).exists()
        else None)

    # --- determinism, pilot, overflow, deviations -------------------------------
    determinism = {}
    for name in ("probe_determinism.json", "probe_determinism_p3.json",
                 PA_DETERMINISM_NAME):
        p = run_dir / name
        if p.exists():
            determinism[name] = json.loads(p.read_text(encoding="utf-8"))
    pilot_path = run_dir / PILOT_RECORD_NAME
    pilot_summary = (json.loads(pilot_path.read_text(encoding="utf-8"))
                     if pilot_path.exists() else None)
    overflow = {}
    for system in PA_SYSTEMS:
        species_sorted = sorted({entry["species"] for entry in
                                 manifest["systems"][system][
                                     "cell_sets"].values()})
        overflow[system] = {
            sp: {"n_particles": int(arts[system][0]["species_binning"][sp][
                "n_particles"]),
                 "overflow": int(arts[system][0]["species_binning"][sp].get(
                     "overflow", 0)),
                 "convention": binning_rule_text()}
            for sp in species_sorted
        }
    overflow["ppref"] = {
        "convention": "the p+p reference carries no tiling cell sets; "
                      "the R node box counts carry no overflow semantics "
                      "(the manifest binning statement)",
        "n_particles": {sp: int(arts["ppref"][0]["species_binning"][sp][
            "n_particles"]) for sp in PPREF_SPECIES},
    }

    # --- claim statement --------------------------------------------------------
    cell_verdicts = {
        key: rec["pooled_kernel_yield"]["verdict"]
        for key, rec in cell_observables.items()}
    n_cell_pass = sum(1 for v in cell_verdicts.values() if v == "pass")
    n_cell_beyond = sum(1 for v in cell_verdicts.values()
                        if v == "beyond_tension")
    n_r_nodes = sum(len(nodes) for system in PA_SYSTEMS
                    for nodes in r_xf_records[system].values())
    n_r_pass = sum(1 for system in PA_SYSTEMS
                   for nodes in r_xf_records[system].values()
                   for r in nodes if r.get("verdict") == "pass")
    n_r_beyond = sum(1 for system in PA_SYSTEMS
                     for nodes in r_xf_records[system].values()
                     for r in nodes if r.get("verdict") == "beyond_tension")
    n_r_beyond_pPb = sum(
        1 for nodes in r_xf_records["pPb208"].values()
        for r in nodes if r.get("verdict") == "beyond_tension")
    dir_agrees = all(rec["directional_verdict"] == "direction_agrees"
                     for rec in a_scaling_records.values())
    claim_statement = (
        "This phase executes the declared held-out predictive validation "
        "of the staged excitation model against minimum-bias p+A trends "
        "at the frozen M8 calibration set, at the recorded level: "
        f"{n_cell_pass} of {len(cell_verdicts)} pre-registered cell-set "
        "observables pass the pooled kernel-yield band "
        f"({n_cell_beyond} land beyond tension), and {n_r_pass} of "
        f"{n_r_nodes} registered R(xF) nodes pass ({n_r_beyond} beyond "
        f"tension, {n_r_beyond_pPb} of them on the p+Pb primary). "
        "The pion A-scaling trend "
        + ("reproduces the data's direction within the pre-registered "
           "directional bands" if dir_agrees else
           "disagrees with the data's direction at the common nodes")
        + ". The scope is minimum bias only; the centrality test "
        "(sets 1..5, the E_BE-to-K cross-calibration) remains the "
        "declared gap of the third validation status, and no claim "
        "extends beyond the minimum-bias p+A observables registered "
        "here. A fired falsifier would be recorded under the "
        "pre-registered policy; it is "
        + ("fired" if falsifiers["fired"] else "not fired") + ".")
    report = {
        "schema": PA_REPORT_SCHEMA,
        "card": CARD, "milestone": MILESTONE, "phase": PHASE_P3,
        "run_description": RUN_DESCRIPTION,
        "written_utc": _utc_now_iso(),
        "provenance": {
            "configuration_hash": config_hash,
            "config_seed": CONFIG_SEED,
            "m12_seed_base": M12_SEED_BASE,
            "run_description": RUN_DESCRIPTION,
            "m8_record_sha256": m8["m8_record_sha256"],
            "m8_frozen_block_sha256": m8["m8_frozen_block_sha256"],
            "pre_registration_sha256": _sha256_file(Path(prereg_path)),
            "cells_manifest_sha256": manifest_sha,
            "helper_sha256": _sha256_file(Path(__file__)),
            "replica_seeds": {"pPb208": list(WS_B_REPLICA_SEEDS),
                              "C12": list(WS_B_REPLICA_SEEDS),
                              "ppref": list(WS_B_PPREF_REPLICA_SEEDS)},
            "bootstrap_seed": WS_B_BOOTSTRAP_SEED,
            "outer_head": _git_head(REPO_ROOT),
            "nested_head": _git_head(MIXING2006_DIR),
        },
        "calibration_set": m8,
        "statistics": {
            "replicas": {system: len(arts[system]) for system in arts},
            "events_per_replica": int(arts["pPb208"][0]["n_events"]),
            "mc_error_convention": (
                "the replica spread (std, ddof=1) of the per-replica "
                "statistic; the replica mean is the model prediction"),
            "data_error_convention": (
                "the tables' statistical errors propagated through the "
                "pre-registered kernel (the pT-integrated error columns "
                "converted per their headers), combined with the MC error "
                "in quadrature"),
            "interval_methods": (
                "bootstrap intervals for R(xF) (200 event-level resamples "
                "at the workstream-B bootstrap seed, consumed sequentially "
                "in the registered node order); the cell-set observables "
                "carry the replica-spread MC error"),
            "registered_bootstrap_order": registered_bootstrap_order,
            "pooled_K_used": {system: pooled_K[system]["pooled_mean"]
                              for system in PA_SYSTEMS},
        },
        "conservation": {
            "tolerance": 1e-10,
            "tests": ("spec 7.1 tests 1-5 re-asserted per event "
                      "(event.validate at 1e-10); test 7 re-asserted per "
                      "final antibaryon; test 8 (the p+A accounting: "
                      "total event baryon number A+1 and charge Z+1) "
                      "re-asserted per event over the final particles "
                      "and the target spectators"),
            "by_system": conservation_by_system,
            "max_residual": max(
                rec["max_residual"]
                for rec in conservation_by_system.values()),
            "total_events": sum(rec["n_events"]
                                for rec in conservation_by_system.values()),
        },
        "geometry_anchor_crosscheck": geometry_anchors_record,
        "wound_bookkeeping": wound_bookkeeping,
        "cell_observables": cell_observables,
        "r_xf": r_xf_records,
        "a_scaling": a_scaling_records,
        "falsifiers": falsifiers,
        "wall_budget_accounting": wall_accounting,
        "determinism": determinism,
        "pilot_pins": {
            "artifact": PILOT_RECORD_NAME,
            "all_pins_hold": (pilot_summary or {}).get("all_pins_hold"),
            "c12_spec_pin": (pilot_summary or {}).get("c12_spec_pin"),
            "model_backward_xF_support": (pilot_summary or {}).get(
                "model_backward_xF_support"),
            "statement": (
                "the pilot preceded production and all regression pins of "
                "the p+A machinery (the C12 NucleusSpec construction, the "
                "rare-slice and point-target pins of the foundations "
                "section 5, the pilot-scale determinism, and the P1 "
                "probe-expectation re-assertions) held; production was "
                "not refused"),
        },
        "deviations": P3_DEVIATIONS,
        "overflow": overflow,
        "replicas": [
            {"system": system, "seed": int(a["provenance"]["seed"]),
             "n_events": int(a["n_events"]),
             "wall_s": float(a.get("wall_s", 0.0)),
             "mean_target_wounds": (float(a["wounds"]["mean"])
                                    if a.get("wounds") else None),
             "mean_final_multiplicity": float(a["mean_final_multiplicity"]),
             "max_conservation_residual": float(
                 a["conservation"]["max_residual"]),
             "artifact": (("ppref_replica_" if system == "ppref"
                           else f"pa_replica_{system}_")
                          + str(a["provenance"]["seed"]) + ".json")}
            for system in arts for a in arts[system]],
        "claim_statement": claim_statement,
        "open_questions": [
            "the orchestrator prompt describes the p+C R(xF) comparison "
            "as '7 nodes per charge per the P1 common-support finding'; "
            "the pre-registered record carries 11 data-side nodes per "
            "pion charge (22 proton, 7 antiproton), and every "
            "pre-registered node received a verdict; the 7-node subset "
            "(xF 0.0-0.3) is the A-scaling common-xF support",
            "the pT acceptance of the model-side R(xF) estimator mirrors "
            "each system's own data-grid column (the recorded rule); "
            "nodes whose column is truncated relative to the published "
            "ptintegrated acceptance carry the recorded estimator "
            "artifact",
        ],
    }
    _write_json(run_dir / "pa_trends_report.json", report)
    print(f"[pa collect] {len(cell_observables)} cell-set observables, "
          f"{len(registered_bootstrap_order)} R nodes; max residual "
          f"{report['conservation']['max_residual']:.3e}", flush=True)
    return report


def _pa_r_data_error(entry: dict, data_key: str) -> float:
    """The combined data-side relative error of one pre-registered R(xF)
    row: the p+A ptintegrated error (absolute for the converted p+Pb
    table, percent for the p+C mirror) and the p+p ptintegrated error
    (percent), each converted to absolute and propagated through R."""
    dn_pa = float(entry["data_dn_dxF_pa"])
    dn_pp = float(entry["data_dn_dxF_pp"])
    err_raw_pa = float(entry["data_dn_dxF_err_raw"])
    err_pa = err_raw_pa if data_key == "pPb" else (err_raw_pa / 100.0) * dn_pa
    pp_table = read_ptintegrated(
        spshadrons_data_dir() / PP_PTINTEGRATED_NAMES[entry["species"]])
    match = np.isclose(pp_table[:, 0], float(entry["xF"]), atol=1e-6)
    if not match.any():
        raise SystemExit(
            f"no pp ptintegrated row at xF {entry['xF']} for "
            f"{entry['species']}; the R data error cannot be propagated")
    row = pp_table[match][0]
    err_pp = (float(row[4]) / 100.0) * dn_pp
    rel = math.sqrt((err_pa / dn_pa) ** 2 + (err_pp / dn_pp) ** 2)
    return rel * float(entry["data_R_samplerK"])


def _pa_pooled_kernel_yield(entry: dict, values) -> tuple:
    """The pooled pre-registered kernel yield of one cell-set observable:
    Y = sum over the cells of (pT/E) v dxF dpT, with v the per-cell
    density-scale value (the per-event density, or f scaled by the
    anchor). Returns (yield, data_error) for the data side, or the model
    yield when values are the model's per-cell values."""
    cells = entry["cells"]
    sp = entry["species"]
    mass = SPECIES_MASS_GEV[sp]
    total = 0.0
    err2 = 0.0
    for i, cell in enumerate(cells):
        xf, pt = float(cell["xF"]), float(cell["pT"])
        energy = math.sqrt(mass**2 + (xf * SQRT_S_GEV / 2.0) ** 2 + pt**2)
        k = pt / energy
        area = float(cell["dxf"]) * float(cell["dpt"])
        total += k * float(values[i]) * area
        err2 += (k * float(cell["data_err"]) * area) ** 2
    return total, math.sqrt(err2)


def _pa_cell_set_records(manifest_entry: dict, arts: list,
                         set_name: str) -> dict:
    """The per-cell comparison records and the pooled kernel-yield verdict
    of one cell-set observable (the p+Pb per-event density, the p+Pb f, or
    the p+C f of one species).

    Model per cell: the thesis Jacobian density from the replica's binned
    counts, scaled by the measured sigma_inel anchor for the f-side
    observables (anchor 1 for the per-event density); the replica mean is
    the model prediction and the replica spread (std ddof=1) is the MC
    error. The pooled kernel yield is the pre-registered
    Y = sum over the cells of (pT/E) f dxF dpT, evaluated identically on
    both sides; the per-observable verdict is its band verdict.
    """
    cells = manifest_entry["cells"]
    sp = manifest_entry["species"]
    anchor = float(manifest_entry.get("anchor_mb", 1.0))
    mass = SPECIES_MASS_GEV[sp]
    n_events = int(arts[0]["n_events"])
    pooled_counts = np.zeros(len(cells), dtype=np.int64)
    per_replica_values = np.zeros((len(arts), len(cells)), dtype=float)
    for r, art in enumerate(arts):
        counts = np.asarray(art["cell_counts"][set_name], dtype=np.int64)
        pooled_counts += counts
        for i, cell in enumerate(cells):
            if counts[i] == 0:
                continue
            xf, pt = float(cell["xF"]), float(cell["pT"])
            energy = math.sqrt(mass**2 + (xf * SQRT_S_GEV / 2.0) ** 2
                               + pt**2)
            jacobian = energy / (math.pi * SQRT_S_GEV * pt)
            density = (float(counts[i]) / float(art["n_events"])
                       / (float(cell["dxf"]) * float(cell["dpt"])))
            per_replica_values[r, i] = jacobian * density
    model_values = anchor * per_replica_values.mean(axis=0)
    mc_per_cell = (anchor * per_replica_values.std(axis=0, ddof=1)
                   if len(arts) > 1 else np.zeros(len(cells)))
    cell_records = []
    verdict_counts = {"pass": 0, "tension": 0, "beyond_tension": 0,
                      "no_model_support": 0}
    for i, cell in enumerate(cells):
        data_value = float(cell["data_value"])
        data_err = float(cell["data_err"])
        mc = float(mc_per_cell[i])
        sigma_comb = math.sqrt(data_err**2 + mc**2)
        delta = float(model_values[i] - data_value)
        pooled = int(pooled_counts[i])
        if pooled == 0:
            verdict, z = "no_model_support", None
        elif sigma_comb > 0:
            z = delta / sigma_comb
            verdict = verdict_from_z(z)
        else:
            verdict, z = "no_model_support", None
        verdict_counts[verdict] += 1
        cell_records.append({
            "xF": float(cell["xF"]), "pT": float(cell["pT"]),
            "model": float(model_values[i]),
            "data": data_value, "data_err": data_err,
            "delta": delta,
            "sigma_combined": float(sigma_comb),
            "delta_over_sigma_combined": z,
            "verdict": verdict,
            "pooled_model_counts": pooled,
        })
    # The pooled kernel yield: the data side with its propagated error, the
    # model side per replica (the identical kernel form).
    data_yield, data_yield_err = _pa_pooled_kernel_yield(
        manifest_entry, [float(c["data_value"]) for c in cells])
    model_yields = []
    for r in range(len(arts)):
        y, _e = _pa_pooled_kernel_yield(
            manifest_entry, [float(v) for v in per_replica_values[r]])
        model_yields.append(anchor * y)
    model_yield = float(np.mean(model_yields))
    mc_yield = (float(np.std(model_yields, ddof=1))
                if len(model_yields) > 1 else 0.0)
    sigma_comb_yield = math.sqrt(data_yield_err**2 + mc_yield**2)
    delta_yield = model_yield - data_yield
    z_yield = (delta_yield / sigma_comb_yield) if sigma_comb_yield else None
    return {
        "species": sp,
        "anchor_mb": anchor,
        "n_cells": len(cells),
        "n_cells_no_model_support": int(sum(
            1 for c in cell_records
            if c["verdict"] == "no_model_support")),
        "cells": cell_records,
        "verdict_counts": verdict_counts,
        "pooled_kernel_yield": {
            "model": float(model_yield),
            "mc_err_replica_spread": mc_yield,
            "data": float(data_yield),
            "data_err_propagated": float(data_yield_err),
            "sigma_combined": float(sigma_comb_yield),
            "delta": float(delta_yield),
            "delta_over_sigma_combined": z_yield,
            "verdict": (verdict_from_z(z_yield) if z_yield is not None
                        else "no_model_support"),
        },
    }


def pa_falsifier4_cross_reference(r_xf_records: dict,
                                  a_scaling_records: dict) -> dict:
    """The pre-registered falsifier-4 cross-reference (spec 7.3 falsifier
    4: p+A target-side scaling or projectile stopping), evaluated on the
    per-node R(xF) verdicts.

    - Projectile-stopping direction: R(xF) of p+Pb (primary) above the
      data beyond k_tension at forward xF (xF >= FALSIFIER4_FORWARD_XF_MIN
      within the p+Pb support).
    - Target-side-scaling direction: an inverted A-ordering, i.e. the
      model's R(A=12) minus R(A=208) ordering at a common node deviating
      from the data's ordering beyond k_tension (the model introducing an
      A-ordering the data does not carry).
    """
    stopping = []
    for system, species_records in r_xf_records.items():
        if system != "pPb208":
            continue
        for sp, nodes in species_records.items():
            for rec in nodes:
                z = rec.get("delta_over_sigma_combined")
                if (rec.get("xF", 0.0) >= FALSIFIER4_FORWARD_XF_MIN
                        and z is not None and z > K_TENSION):
                    stopping.append({
                        "species": sp, "xF": rec["xF"], "z": z,
                        "model_R": rec.get("model_R"),
                        "data_R": rec.get("data_R"),
                    })
    ordering = []
    for sp, rec in a_scaling_records.items():
        for node in rec["nodes"]:
            z_diff = node.get("ordering_z")
            if z_diff is not None and z_diff > K_TENSION:
                ordering.append({"species": sp, "xF": node["xF"],
                                 "ordering_z": z_diff})
    return {
        "spec": "7.3 falsifier 4 (p+A target-side scaling or projectile "
                "stopping)",
        "directions": {
            "projectile_stopping": (
                "R(xF) of p+Pb (primary) above the data beyond k_tension "
                "at forward xF (xF >= "
                + str(FALSIFIER4_FORWARD_XF_MIN) + ")"),
            "target_side_scaling": (
                "an inverted A-ordering: the model's R(A=12) - R(A=208) "
                "ordering at a common node deviating from the data's "
                "ordering beyond k_tension"),
        },
        "projectile_stopping_nodes": stopping,
        "inverted_ordering_nodes": ordering,
        "fired": bool(stopping or ordering),
        "policy": ("a fired falsifier is a recorded result that drives "
                   "the P4 wording and does not block the land gate (the "
                   "M9/M10 both-outcomes rule)"),
    }




def canonical_configuration_hash(config_path: Path) -> str:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return canonical_json_sha256(config)


def run_probe_determinism(out_dir: Path, config_path: Path,
                          prereg: dict,
                          out_name: str = "probe_determinism.json") -> dict:
    """Pre-registered determinism probe (the M10 probe_A pattern): five
    repeated evaluations at a fixed point must be bit-identical.

    Units: (i) the p+p statistic pipeline (300 events at the pilot seed,
    the full workstream-A statistic vector); (ii) the p+A statistic
    pipeline (50 p+Pb events plus 50 p+C events at the pilot seed, the
    workstream-B statistic vector).
    """
    out_dir = Path(out_dir)
    cross_check_preregistration(prereg)
    config_hash = canonical_configuration_hash(config_path)

    # Unit (i): the p+p pipeline.
    walls_pp: list = []
    vectors_pp: list = []
    residuals_pp: list = []
    for _rep in range(DETERMINISM_REPEATS):
        events, _summary, wall, _applied, _beam = generate_pp_events(
            COST_CELL_EVENTS, PROBE_SEED_PP_DETERMINISM, config_path)
        vec = json.dumps(_to_jsonable(pp_statistic_vector(events)))
        vectors_pp.append(vec)
        walls_pp.append(float(wall))
        for event in events:
            residuals_pp.append(
                float(event.conservation_residuals()["relative_max"]))
    unit_pp = {
        "n_events": int(COST_CELL_EVENTS),
        "seed": int(PROBE_SEED_PP_DETERMINISM),
        "wall_s": walls_pp,
        "bit_identical": len(set(vectors_pp)) == 1,
        "max_conservation_residual": float(max(residuals_pp)),
    }

    # Unit (ii): the p+A pipeline (p+Pb and p+C at the pilot seed).
    walls_pa: list = []
    vectors_pa: list = []
    residuals_pa: list = []
    for _rep in range(DETERMINISM_REPEATS):
        pb_events, _s1, wall1, _a1, _b1 = generate_pa_events(
            DETERMINISM_PA_EVENTS, PROBE_SEED_PA_DETERMINISM, config_path,
            GEOMETRY_PB208)
        pc_events, _s2, wall2, _a2, _b2 = generate_pa_events(
            DETERMINISM_PA_EVENTS, PROBE_SEED_PA_DETERMINISM + 10,
            config_path, GEOMETRY_C12)
        vec = json.dumps(_to_jsonable(
            pa_statistic_vector(pb_events, pc_events)))
        walls_pa.append(float(wall1))
        vectors_pa.append(vec)
        for event in list(pb_events) + list(pc_events):
            residuals_pa.append(
                float(event.conservation_residuals()["relative_max"]))
    unit_pa = {
        "n_events_pPb208": int(DETERMINISM_PA_EVENTS),
        "n_events_C12": int(DETERMINISM_PA_EVENTS),
        "seed_pPb208": int(PROBE_SEED_PA_DETERMINISM),
        "seed_C12": int(PROBE_SEED_PA_DETERMINISM + 10),
        "wall_s": walls_pa,
        "bit_identical": len(set(vectors_pa)) == 1,
        "max_conservation_residual": float(max(residuals_pa)),
    }

    payload = {
        "schema": "staged_heldout_ratios_pa_probe_determinism/v1",
        "probe": "determinism",
        "provenance": probe_provenance(config_hash),
        "n_repeats": int(DETERMINISM_REPEATS),
        "units": {"pp": unit_pp, "pa": unit_pa},
        "deterministic": bool(unit_pp["bit_identical"] and unit_pa["bit_identical"]),
    }
    _write_json(out_dir / out_name, payload)
    print(
        f"[probe determinism] pp bit_identical={unit_pp['bit_identical']}, "
        f"pa bit_identical={unit_pa['bit_identical']}",
        flush=True,
    )
    return payload


def run_probe_cost_cells(out_dir: Path, config_path: Path,
                         prereg: dict) -> dict:
    """Pre-registered per-event cost cells (the M10 probe_C pattern): 300
    events per cell, one single sample per cell at the fixed M8 parameters,
    for the p+p path and the p+A p+Pb208/p+C12 paths; plus the pre-registered
    statistics-menu decision per system."""
    out_dir = Path(out_dir)
    config_hash = canonical_configuration_hash(config_path)
    cells: dict = {}

    def measure(name: str, n_events: int, seed: int, generator) -> dict:
        t0 = time.perf_counter()
        events, summary, wall, _applied, _beam = generator(
            int(n_events), int(seed), config_path)
        wall = time.perf_counter() - t0
        return {
            "n_events": int(len(events)),
            "wall_s": float(wall),
            "ms_per_event": float(1000.0 * wall / max(len(events), 1)),
            "mean_target_wounds": float(summary.get("mean_target_wounds", 1.0)),
            "mean_final_multiplicity": float(summary.get("mean_final_multiplicity", 0.0)),
            "max_conservation_residual": float(summary.get("max_conservation_residual", 0.0)),
        }

    cells["pp"] = measure("pp", COST_CELL_EVENTS, PROBE_SEED_PP_COST,
                          generate_pp_events)
    cells["pa_pPb208"] = measure(
        "pa_pPb208", COST_CELL_EVENTS, PROBE_SEED_PPB_COST,
        lambda n, s, c: generate_pa_events(n, s, c, GEOMETRY_PB208))
    cells["pa_C12"] = measure(
        "pa_C12", COST_CELL_EVENTS, PROBE_SEED_PC_COST,
        lambda n, s, c: generate_pa_events(n, s, c, GEOMETRY_C12))

    menu_decision = {
        system: menu_entry_for(system, cell["ms_per_event"])
        for system, cell in cells.items()
    }

    payload = {
        "schema": "staged_heldout_ratios_pa_probe_cost/v1",
        "probe": "cost_cells",
        "provenance": probe_provenance(config_hash),
        "n_events_per_cell": int(COST_CELL_EVENTS),
        "cells": cells,
        "menu_decision": menu_decision,
        "menu_rule": (
            "the largest menu entry {10000, 5000, 2500, 1000} whose "
            "projected production wall (measured ms/event x K=8 replicas x "
            "E events) stays under the pre-registered 18000 s per-system "
            "budget; the declared reduction, if used, carries the MC-sigma "
            "caveat"),
    }
    _write_json(out_dir / "probe_cost_cells.json", payload)
    for name, cell in cells.items():
        print(f"[cost {name}] {cell['ms_per_event']:.2f} ms/event", flush=True)
    return payload


def menu_entry_for(system: str, ms_per_event: float) -> dict:
    """The largest pre-registered menu entry whose projected production wall
    (measured ms/event x K=8 x E) stays under the per-system budget."""
    chosen = None
    for e1 in STATISTICS_MENU_EVENTS_PER_REPLICA:
        projected = ms_per_event * K_REPLICAS * e1 / 1000.0
        if projected < MAX_PROJECTED_WALL_S:
            chosen = {
                "system": system,
                "chosen_events_per_replica": int(e1),
                "projected_wall_s": float(projected),
            }
            break  # the menu is ascending; the first fitting entry is the largest
    if chosen is None:
        chosen = {
            "system": system,
            "chosen_events_per_replica": None,
            "projected_wall_s": None,
            "status": "no_menu_entry_fits_budget",
        }
    return chosen


def run_probe_gateway_capacity(out_dir: Path, config_path: Path) -> dict:
    """Pre-registered gateway capacity probe, read-only: the per-worker task
    capacity of the grid, queried through the gateway client CLI (the same
    command path the M11 grid used); on any failure the recorded serial
    fallback statement is written instead."""
    out_dir = Path(out_dir)
    config_hash = canonical_configuration_hash(config_path)
    payload: dict = {
        "schema": "staged_heldout_ratios_pa_probe_gateway/v1",
        "probe": "gateway_capacity",
        "mode": "read_only",
        "provenance": probe_provenance(config_hash),
    }
    cmd = [sys.executable, "-m", "helpers.na49_gateway.cluster_client",
           "capacity", "--max-tasks-per-worker", "64"]
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=180, cwd=str(REPO_ROOT), env=env)
        payload["command"] = "python -m helpers.na49_gateway.cluster_client capacity"
        if proc.returncode != 0:
            raise RuntimeError(
                f"capacity query exited {proc.returncode}: {proc.stderr.strip()[:200]}")
        workers = json.loads(proc.stdout).get("workers", [])
        payload["status"] = "ok"
        payload["workers"] = workers
        payload["free_tasks_total"] = sum(int(w.get("free_tasks", 0))
                                          for w in workers)
    except Exception as exc:  # recorded serial fallback (the M11 pattern)
        payload["status"] = "unavailable"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["serial_fallback_statement"] = (
            "the gateway grid is unavailable from this session; the P2/P3 "
            "production points run serially with the recorded per-point "
            "wall budgets (the recorded serial fallback of the M11 pattern)")
    _write_json(out_dir / "probe_gateway_capacity.json", payload)
    print(
        f"[probe gateway] status={payload.get('status')} "
        f"workers={len(payload.get('workers', []))}",
        flush=True,
    )
    return payload


def run_cross_check(out_dir: Path, prereg_path: Path) -> int:
    """The harness cross-check command: verify the pre-registration constants
    verbatim, the frozen M8 parameter set against the M8 record by sha256,
    and the seed block against the M11 pre-registration; record the command,
    its output, and its exit status in the run directory."""
    out_dir = Path(out_dir)
    record: dict = {
        "schema": "staged_heldout_ratios_pa_harness_cross_check/v1",
        "command": (
            "PYTHONPATH=. python helpers/benchmarking/"
            "staged_heldout_ratios_pa.py --cross-check"),
        "argv": sys.argv,
        "written_utc": _utc_now_iso(),
        "checks": {},
    }
    exit_code = 0
    try:
        record["checks"]["m8_record"] = cross_check_m8_record()
        record["checks"]["no_collision"] = verify_no_collision()
        prereg = json.loads(Path(prereg_path).read_text(encoding="utf-8"))
        cross_check_preregistration(prereg)
        record["checks"]["pre_registration_constants"] = "verified"
        record["pre_registration_sha256"] = _sha256_file(prereg_path)
        record["status"] = "pass"
    except SystemExit as exc:
        exit_code = 1
        record["status"] = "fail"
        record["message"] = str(exc)
    record["exit_code"] = int(exit_code)
    _write_json(out_dir / "harness_cross_check.json", record)
    print(f"[cross-check] status={record['status']} "
          f"exit_code={record['exit_code']}", flush=True)
    return exit_code


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-check", action="store_true",
                        help="verify the pre-registration constants, the M8 "
                             "frozen set (by sha256), and the seed-block "
                             "no-collision statement; record the exit status")
    parser.add_argument("--probe-determinism", action="store_true",
                        help="five bit-identical repeated evaluations at a "
                             "fixed point (p+p and p+A units)")
    parser.add_argument("--determinism-out-name", default="",
                        help="artifact name for --probe-determinism (default "
                             "probe_determinism.json; the P2 re-assertion "
                             "uses probe_determinism_p2.json)")
    parser.add_argument("--probe-cost", action="store_true",
                        help="per-event cost cells (300 events, one sample "
                             "per cell: p+p, p+Pb208, p+C12) and the menu "
                             "decision")
    parser.add_argument("--probe-gateway", action="store_true",
                        help="gateway capacity probe (read-only)")
    parser.add_argument("--probe-all", action="store_true",
                        help="run the full pre-registered probe schedule")
    parser.add_argument("--build-manifest", action="store_true",
                        help="P2: build the cells manifest (pre-registered "
                             "per-window and seagull cell sets, pure table "
                             "arithmetic) in the run directory")
    parser.add_argument("--produce-replica", type=int, default=None,
                        metavar="SEED",
                        help="P2: run one production replica at SEED and "
                             "write replica_<SEED>.json (gateway per-replica "
                             "entry point)")
    parser.add_argument("--cells-manifest", default="",
                        help="cells manifest JSON (default: "
                             "<out-dir>/cells_manifest.json)")
    parser.add_argument("--n-events", type=int,
                        default=PRIMARY_EVENTS_PER_REPLICA,
                        help="events per replica (default: the pre-registered "
                             "primary statistics)")
    parser.add_argument("--m8-record", default="",
                        help="M8 record path override (the gateway "
                             "environment resolves the tree-local run root "
                             "differently)")
    parser.add_argument("--determinism-production", action="store_true",
                        help="P2: same-environment double generation at "
                             "production scale (bit-identical payloads)")
    parser.add_argument("--collect", action="store_true",
                        help="P2: collect the eight replica artifacts into "
                             "ratios_report.json (recomputes every data-side "
                             "statistic and refuses mismatches)")
    parser.add_argument("--gateway-manifest", default="",
                        help="gateway manifest JSON to record in the report "
                             "(default: <out-dir>/gateway_manifest_p2.json "
                             "when present)")
    parser.add_argument("--pilot-pins", action="store_true",
                        help="P3: run the pilot (the C12 NucleusSpec "
                             "construction pin, the rare-slice and "
                             "point-target regression pins, the model "
                             "backward-xF support re-derivation, the "
                             "pilot-scale determinism, and the P1 "
                             "expectation re-assertions); refuses on a "
                             "missed expectation")
    parser.add_argument("--build-pa-manifest", action="store_true",
                        help="P3: build the p+A cells manifest "
                             "(pre-registered cell sets and R(xF) node "
                             "plans, pure table arithmetic) in the run "
                             "directory; requires the pilot record")
    parser.add_argument("--produce-pa-replica", nargs=2, default=None,
                        metavar=("SYSTEM", "SEED"),
                        help="P3: run one production replica of SYSTEM "
                             "(pPb208, C12, or ppref) at SEED and write "
                             "the replica artifact (gateway per-replica "
                             "entry point)")
    parser.add_argument("--determinism-production-pa", action="store_true",
                        help="P3: same-environment double generation at "
                             "production scale on the p+A path (bit-"
                             "identical payloads for pPb208, C12, ppref)")
    parser.add_argument("--collect-pa", action="store_true",
                        help="P3: collect the p+A replica artifacts into "
                             "pa_trends_report.json (recomputes every "
                             "data-side value and refuses mismatches)")
    parser.add_argument("--pa-gateway-manifest", default="",
                        help="gateway manifest JSON to record in the p+A "
                             "report (default: <out-dir>/"
                             "gateway_manifest_p3.json when present)")
    parser.add_argument("--out-dir", default=str(RUNS_ROOT / RUN_DESCRIPTION))
    parser.add_argument("--config", default="",
                        help="staged excitation JSON config (default: "
                             "<mixing2006>/config/staged_excitation.json)")
    parser.add_argument("--preregistration", default="",
                        help="pre-registration JSON (default: "
                             "<out-dir>/pre_registration.json)")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    config_path = (Path(args.config) if args.config
                   else MIXING2006_DIR / "config" / "staged_excitation.json")
    prereg_path = (Path(args.preregistration) if args.preregistration
                   else out_dir / "pre_registration.json")
    manifest_path = (Path(args.cells_manifest) if args.cells_manifest
                     else out_dir / "cells_manifest.json")
    m8_override = Path(args.m8_record) if args.m8_record else None

    if args.build_manifest:
        out_dir.mkdir(parents=True, exist_ok=True)
        build_cells_manifest(config_path, prereg_path, out_dir)
        return 0
    if args.produce_replica is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        produce_replica(
            int(args.produce_replica), config_path, manifest_path,
            out_dir / f"replica_{int(args.produce_replica)}.json",
            n_events=int(args.n_events), m8_record=m8_override,
            preregistration=prereg_path)
        return 0
    if args.determinism_production:
        out_dir.mkdir(parents=True, exist_ok=True)
        run_determinism_production(out_dir, config_path, manifest_path,
                                   preregistration=prereg_path,
                                   m8_record=m8_override)
        return 0
    if args.collect:
        out_dir.mkdir(parents=True, exist_ok=True)
        gw = (Path(args.gateway_manifest) if args.gateway_manifest
              else out_dir / "gateway_manifest_p2.json")
        collect_ratios_report(out_dir, config_path, prereg_path,
                              gateway_manifest_path=gw)
        return 0
    if args.pilot_pins:
        out_dir.mkdir(parents=True, exist_ok=True)
        prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
        run_pilot_pins(out_dir, config_path, prereg)
        return 0
    if args.build_pa_manifest:
        out_dir.mkdir(parents=True, exist_ok=True)
        build_pa_cells_manifest(config_path, prereg_path, out_dir)
        return 0
    if args.produce_pa_replica is not None:
        system, seed = args.produce_pa_replica
        if system not in ("pPb208", "C12", "ppref"):
            raise SystemExit(
                f"unknown system {system!r}; use pPb208, C12, or ppref")
        out_dir.mkdir(parents=True, exist_ok=True)
        name = (f"ppref_replica_{int(seed)}.json" if system == "ppref"
                else f"pa_replica_{system}_{int(seed)}.json")
        produce_pa_replica(
            int(seed), config_path, manifest_path, system,
            out_dir / name, n_events=int(args.n_events),
            m8_record=m8_override, preregistration=prereg_path)
        return 0
    if args.determinism_production_pa:
        out_dir.mkdir(parents=True, exist_ok=True)
        run_pa_determinism_production(out_dir, config_path, manifest_path,
                                      preregistration=prereg_path,
                                      m8_record=m8_override)
        return 0
    if args.collect_pa:
        out_dir.mkdir(parents=True, exist_ok=True)
        gw = (Path(args.pa_gateway_manifest) if args.pa_gateway_manifest
              else out_dir / GATEWAY_MANIFEST_P3_NAME)
        collect_pa_trends_report(out_dir, config_path, prereg_path,
                                 gateway_manifest_path=gw)
        return 0
    if args.probe_all:
        out_dir.mkdir(parents=True, exist_ok=True)
        prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
        cross_check_preregistration(prereg)
        run_probe_determinism(out_dir, config_path, prereg)
        run_probe_cost_cells(out_dir, config_path, prereg)
        run_probe_gateway_capacity(out_dir, config_path)
        return 0
    if args.probe_determinism:
        out_dir.mkdir(parents=True, exist_ok=True)
        prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
        run_probe_determinism(out_dir, config_path, prereg,
                              out_name=(args.determinism_out_name
                                        or "probe_determinism.json"))
        return 0
    if args.probe_cost:
        out_dir.mkdir(parents=True, exist_ok=True)
        prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
        run_probe_cost_cells(out_dir, config_path, prereg)
        return 0
    if args.probe_gateway:
        out_dir.mkdir(parents=True, exist_ok=True)
        run_probe_gateway_capacity(out_dir, config_path)
        return 0
    if args.cross_check:
        out_dir.mkdir(parents=True, exist_ok=True)
        return run_cross_check(out_dir, prereg_path)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
