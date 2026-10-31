#!/usr/bin/env python3
"""Centrality-set cross-calibration helper (card TASK-20260911-centrality-sets-1-5, M13).

Pre-registration-time machinery for the p+A centrality test at the frozen
M8 calibration parameter set: the five centrality-selected sets of the
Chvala thesis for p+Pb208 at 158 GeV/c (sets 1..5, per-event invariant
density d(xF,pT)), compared through the pre-registered E_BE-to-K
cross-calibration that maps each thesis set (a forward-energy E_BE
selection via grey-proton counting in the NA49 centrality detector) to a
wounded-nucleon number K slice of the model's Glauber geometry.

Phase P1 fixes the protocol before any production event generation. This
module carries the pure table arithmetic the pre-registration records
(per-set/charge grid inventory, unique-cell grids, the duplicated-label
merge rule, the common support against the pinned model support, the
closed-form E_BE-to-K mapping table from the sampler's
``fraction_for_wound_range`` arithmetic, and the data-side kernel-weighted
dn/dxF and wounded-scaling R_i recomputation), the seed-block derivation
with the no-collision statement verified against the M12 pre-registration,
the pre-registration builder, and the harness cross-check command that
verifies the frozen M8 parameter set against the M8 record, the geometry
against the bundled configuration, the data files (existence, row counts,
grid inventory), the seed block against the M12 record, and the mapping
table by deterministic recomputation of the closed-form arithmetic. The
harness refuses to run on any mismatch: the bands, the seeds, the mapping,
and the statistics menu are fixed before the events exist and are never
tuned after a run (spec section 7.3 falsifier policy). A recorded fail is
a legitimate result.

Data conventions (pinned here and in the pre-registration):

- The thesis per-event invariant density is d(xF,pT) = E/(pi sqrt(s) pT)
  d^2n/dxF dpT per event (Chvala 2006), feed-down corrected, statistical
  errors only; the flat-file conversion is helpers/benchmarking/pa_data/
  (2026-08-25).
- The coarse-grid set files carry the duplicated pT labels 0.10 and 0.20
  (pa_data/README.md anomaly 1): two data rows per affected cell,
  preserved as-is. The unique-cell grid merges all rows of one (xF, pT)
  cell by inverse-variance weighting, with the merged absolute error; the
  per-cell records keep every row.
- The pT label 0.00 of the coarse-grid files is dropped from the
  registered support at pre-registration (recorded reason): the thesis
  Jacobian and the pinned model density convention are both singular at a
  pT node of zero.
- The model support is pinned by a small fixed-seed probe sample through
  the sliced builder path (pilot sub-stream, yield-counting only,
  excluded from production statistics); common-support windows with no
  data support on either side are dropped at pre-registration with the
  recorded reason.
- Frame (card TASK-20261005-pa-frame-nn-cm-xf, 2026-10-06): x_F is
  2 p_z / sqrt(s_NN) in the nucleon-nucleon CM frame. The p+A builder
  emits laboratory-frame events and records the frame, so track_xf boosts
  the p+A tracks to the NN CM (beta = p_beam / (E_beam + m_p), gamma =
  (E_beam + m_p) / sqrt(s_NN)); the p+p reference events are CM events
  and keep their x_F bit for bit. The sealed P2 record of 2026-09-11
  divided the lab-frame p_z by sqrt(s_NN) and is withdrawn.

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
CARD = "TASK-20260911-centrality-sets-1-5"
MILESTONE = "M13"
PHASE = "P1"
RUN_DESCRIPTION = "staged_centrality_sets_2026-09-11"
STAGE_LABEL_SLICES = "centrality_sets"
STAGE_LABEL_PPREF = "centrality_sets_pp_reference"
PRODUCTION_MODE = "regge"
BEAM_MOMENTUM_GEV_C = 158.0
SQRT_S_GEV = 17.270205201372406  # sqrt(s) of Reaction(158.0, 2212, 2212); NN sqrt(s)

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

# Geometry (fixed, no tuning): the bundled Pb208 Woods-Saxon values.
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
PA_RUNNER_SETTINGS = {"glauber_dr_fm": 0.05, "max_event_attempts": 100}
BUNDLED_CONFIG_RELPATH = "dev/mixing2006/config/staged_excitation.json"
BUNDLED_CONFIG_CANONICAL_SHA256 = (
    "3a83d7c8817ef934656c5b45d27d7b9da53af7b51152fab29b4aafa497c7bfee")

# Normalization and thesis anchors (measured).
SIGMA_INEL_PPb_MB = 1806.0
SIGMA_INEL_PPb_ERR_MB = 25.0
NU_PPB_MEASURED = 3.65  # thesis Tab. tabinelcs2; the geometry-anchor cross-check
NU_MINBIAS_CONTEXT = 3.7  # thesis Tab. 5.4 (tabcdbins) context row
NU_ALLTRIG_CONTEXT = 5.1  # thesis Tab. 5.4 context row (n_grey > 0)
NU_SYSTEMATIC_FRACTION = 0.10  # the +-10% systematic on <nu> (thesis sec:nungreyrel)
# Thesis Tab. 5.4 (tabcdbins): the grey-proton selections, event counts, and
# measured mean intranuclear collision numbers per set, p+Pb at 158 GeV/c.
THESIS_CENTRALITY_SETS = {
    "1": {"n_grey_selection": "1, 2", "events": 249509, "nu_measured": 2.9},
    "2": {"n_grey_selection": "3, 4, 5", "events": 467234, "nu_measured": 4.1},
    "3": {"n_grey_selection": "6, 7, 8", "events": 490228, "nu_measured": 5.2},
    "4": {"n_grey_selection": "9 -- 13", "events": 400683, "nu_measured": 6.0},
    "5": {"n_grey_selection": "14 -- 22", "events": 62039, "nu_measured": 6.9},
}
THESIS_EVENTS_ELIMINATED_ABOVE_SET5 = 260  # events with n_grey > 22 (thesis L2385)

# --- The closed-form E_BE-to-K mapping constants (fixed before any run) ---------

K_MAX_CUMULATIVE_MASS = 0.999
WOUND_BAND_WIDTHS = (0, 1, 2, 3)  # band = [K, K+w]; band size 1..4
MAPPING_TIE_BREAK_RULE = (
    "ties in |<K>_band - <nu>| resolve to the narrower band, then to the "
    "lower band start")
CONTEXT_ROW_NU = {"min_bias_all": 3.7, "all_triggered_gt0": 5.1}

# --- Seed protocol (shared block; M13 base = config seed + 90000) ---------------

CONFIG_SEED = 20260831
M13_SEED_BASE = 20350831  # config seed + 90000 (M13)
NEXT_MILESTONE_OFFSET_SEED = 20360831  # config seed + 100000 (untouched)
M13_BLOCK_TOP = 20360830  # the highest seed any M13 sub-stream may use
MIN_SUBBLOCK_STRIDE = 1000  # the M10 stride rule (anchor + 8 replicas + spare)
M12_PREREG_RELPATH = "staged_heldout_ratios_pa_2026-09-10/pre_registration.json"
M12_PREREG_SHA256 = (
    "e3dbe1778aefb93df8017736824982583b6278ae92f640bc75a281b6b084442e")
M12_SEED_BASE = 20340831
M12_CONSUMED_TOP = 20346839  # the highest M12 sub-stream seed (ppref replica 8)
PILOT_SEED_ANCHOR = 20349831  # offset -1000 from the base; the M9 pilot convention
# Pilot sub-streams (yield-counting and probe cells only; excluded from
# production statistics). The P2-start pilot probes (determinism, per-slice
# cost cells, acceptance sanity) run at the scheduled sub-streams.
PROBE_SEED_PP_DETERMINISM = 20349831  # pilot anchor
PROBE_SEED_SLICED_DETERMINISM = 20349832  # pilot anchor + 1; mid set 3
PROBE_SEED_COST_SET1 = 20349833  # pilot anchor + 2
PROBE_SEED_COST_SET2 = 20349834
PROBE_SEED_COST_SET3 = 20349835
PROBE_SEED_COST_SET4 = 20349836
PROBE_SEED_COST_SET5 = 20349837
PROBE_SEED_ACCEPT_SET1 = 20349838
PROBE_SEED_ACCEPT_SET2 = 20349839
PROBE_SEED_ACCEPT_SET3 = 20349840
PROBE_SEED_ACCEPT_SET4 = 20349841
PROBE_SEED_ACCEPT_SET5 = 20349842
PROBE_SEED_MODEL_SUPPORT = 20349843  # pilot anchor + 12 (this module's probe)
SLICE_N_REPLICAS = 8
SLICE_BOOTSTRAP_OFFSET = 900  # anchor + 900 within each 2000-stride slice
PPREF_ANCHOR = 20359831  # slice-5 anchor + 1000 (the stride rule)
PPREF_BOOTSTRAP_SEED = PPREF_ANCHOR + SLICE_BOOTSTRAP_OFFSET  # 20360731

# --- Statistics menu, budget, verdict bands -----------------------------------

K_REPLICAS = 8
STATISTICS_MENU_EVENTS_PER_REPLICA = (10000, 5000, 2500, 1000)
PRIMARY_EVENTS_PER_REPLICA = 10000
MAX_PROJECTED_WALL_S = 18000.0  # per slice, serial-equivalent (the M10 budget)
BOOTSTRAP_REPLICATES = 200  # the M9 convention, one generator sequentially
DETERMINISM_REPEATS = 5
COST_CELL_EVENTS = 300
ACCEPTANCE_SANITY_EVENTS = 10000
PP_REF_EVENTS_PER_REPLICA = 10000
WOUND_BOOKKEEPING_N_SIGMA = 4.0

# Verdict bands: fixed here, before any run (the M9/M10/M12 pattern). The
# combined 1-sigma uncertainty is the quadrature of the data error from the
# tables and the MC error from the replica spread, plus the stated
# systematics (the mapping residual and the anchor deviation).
K_PASS = 2.0
K_TENSION = 5.0

FALSIFIER_DIRECTIONS = {
    "F-M13-1_centrality_trend": (
        "the sign of the linear trend of the pooled kernel yield versus set "
        "order (the data side and the model side each, over the mappable "
        "sets, at least three) is opposite between data and model, with the "
        "endpoint-set pooled-yield verdicts (set 1 and set 5) beyond the "
        "tension band in the direction consistent with the opposite trend. "
        "Fires as the centrality-dependence direction of the wounded "
        "scaling"),
    "F-M13-2_wounded_scaling": (
        "the model R_i(xF) exceeds 1 by more than k_tension combined 1-sigma "
        "at all common nodes xF <= 0.10 while the data R_i(xF) lies below 1 "
        "(or symmetrically the model below while the data above) beyond "
        "k_tension at the same nodes; cross-reference spec section 7.3 "
        "falsifier 4 (projectile-side stopping), the M12 P3 direction"),
    "band_ambiguity_flags": (
        "the band-ambiguity flags of the E_BE-to-K mapping are recorded "
        "systematics, not falsifiers"),
}

# --- Model-support probe (this module's P1 fixed-seed sample) -------------------

MODEL_SUPPORT_PROBE_SEED = 20349843  # pilot anchor + 12
MODEL_SUPPORT_PROBE_EVENTS = 300
MODEL_SUPPORT_PROBE_BAND = (4, 7)  # the mapped band of the representative mid set 3

# --- Data file layout -----------------------------------------------------------

PION_MASS_GEV = 0.13957039
PION_PDGS = (-211, 211)
PA_DATA_DIRNAME = "pa_data"
SET_DENSITY_NAMES = {
    (set_no, charge): f"pPb_dens_set{set_no}_{filename}.txt"
    for set_no in (1, 2, 3, 4, 5)
    for charge, filename in (("pi+", "piplus"), ("pi-", "piminus"))
}
MINBIAS_CONTEXT_DENSITY_NAMES = {
    "pi-": "pPb_dens_minbias_piminus.txt",
    "pi+": "pPb_dens_minbias_piplus.txt",
}
PP_REF_PTINTEGRATED_NAMES = {
    "pi-": "xsect_ptintegrated_piminus.txt",
    "pi+": "xsect_ptintegrated_piplus.txt",
}
PT_ZERO_DROP_REASON = (
    "the pT label 0.00 of the coarse-grid set tables carries a singular "
    "thesis Jacobian (d = E/(pi sqrt(s) pT) d^2n/dxF dpT at pT = 0), so "
    "neither the data-side kernel integration nor the pinned model density "
    "convention defines a finite cell value there; the cell is dropped from "
    "the registered support on both sides at pre-registration (the raw rows "
    "stay in the per-file inventory)")
ZERO_ERROR_DROP_REASON = (
    "the thesis table row carries a zero statistical error, so the "
    "inverse-variance merge is degenerate and the cell cannot carry a "
    "combined 1-sigma; the cell is dropped from the registered support at "
    "pre-registration (the raw rows stay in the per-file inventory)")
XF_NODES_SETS_1_4 = tuple(round(0.05 * k, 4) for k in range(9))  # 0.00..0.40
XF_NODES_SET5 = tuple(round(0.05 * k, 4) for k in range(8))  # 0.00..0.35

CROSS_CHECK_COMMAND = (
    "PYTHONPATH=. python helpers/benchmarking/staged_centrality_sets.py "
    "--cross-check")


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


def m8_record_path() -> Path:
    return RUNS_ROOT / M8_RECORD_RELPATH


def m12_preregistration_path() -> Path:
    return RUNS_ROOT / M12_PREREG_RELPATH


def bundled_config_path() -> Path:
    return REPO_ROOT / BUNDLED_CONFIG_RELPATH


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
    columns are percent values; absolute errors are pct/100 * value.
    """
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        rows.append([float(x) for x in s.split()[:9]])
    return np.asarray(rows, dtype=float)


# --- Grid inventory, duplicated-label merge rule, common support ---------------


def grid_inventory(table: np.ndarray) -> dict:
    """Per-file grid inventory (pure table arithmetic): row count, unique
    (xF, pT) cells, the duplicated-label cells and extra rows, the xF nodes,
    the pT labels, and the pT = 0.00 label cells."""
    cells: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for row in table:
        key = (round(float(row[0]), 4), round(float(row[1]), 4))
        cells.setdefault(key, []).append((float(row[2]), float(row[3])))
    dup = {k: v for k, v in cells.items() if len(v) > 1}
    return {
        "rows": int(len(table)),
        "unique_cells": len(cells),
        "duplicated_label_cells": len(dup),
        "duplicated_extra_rows": int(sum(len(v) - 1 for v in dup.values())),
        "xF_nodes": sorted({k[0] for k in cells}),
        "pT_labels": sorted({k[1] for k in cells}),
        "duplicated_cells": [[x, p] for x, p in sorted(dup)],
        "pT_zero_cells": int(sum(1 for (x, p) in cells if abs(p) < 1e-9)),
    }


def merged_value(rows) -> float:
    """Inverse-variance merged value of one cell's rows."""
    num = sum(float(v) / float(s) ** 2 for v, s in rows)
    den = sum(1.0 / float(s) ** 2 for v, s in rows)
    return num / den


def merged_error(rows) -> float:
    """Merged absolute error of one cell's rows."""
    den = sum(1.0 / float(s) ** 2 for v, s in rows)
    return math.sqrt(1.0 / den)


def merge_unique_cells(table: np.ndarray) -> dict:
    """The unique-cell grid with inverse-variance-merged values and errors
    (the pre-registered merge rule; the duplicated-label rows merge with
    every other row of the same (xF, pT) cell).

    Rows with a non-positive absolute error (the thesis tables carry one
    such row) cannot enter an inverse-variance weight; a cell containing
    any such row, or whose merged error is not a finite positive number,
    is flagged ``ill_defined`` and is dropped from the registered support
    with the recorded reason (ZERO_ERROR_DROP_REASON), keeping the raw
    rows in the inventory.
    """
    cells: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for row in table:
        key = (round(float(row[0]), 4), round(float(row[1]), 4))
        cells.setdefault(key, []).append((float(row[2]), float(row[3])))
    out = {"cells": [], "values": [], "errors": [], "rows_per_cell": [],
           "ill_defined": []}
    for key in sorted(cells):
        rows = cells[key]
        rows_pos = [(v, s) for v, s in rows if s > 0.0]
        out["cells"].append([key[0], key[1]])
        out["rows_per_cell"].append(len(rows))
        if not rows_pos:
            out["values"].append(None)
            out["errors"].append(None)
            out["ill_defined"].append(True)
            continue
        out["values"].append(merged_value(rows_pos))
        err = merged_error(rows_pos)
        out["errors"].append(err)
        out["ill_defined"].append(bool(
            len(rows_pos) < len(rows)
            or not (math.isfinite(err) and err > 0.0)))
    return out


def common_support(merged: dict, support: dict) -> tuple[dict, list]:
    """The data cells inside the pinned model support (pure table arithmetic).

    Three recorded drop rules apply, in order: the pT = 0.00 label cells
    (PT_ZERO_DROP_REASON), the ill-defined cells whose inverse-variance
    merge cannot carry a combined sigma (ZERO_ERROR_DROP_REASON), and any
    cell outside the probe-measured model xF/pT envelope. Windows with no
    data support on either side are dropped here, with the recorded reason,
    before any run.
    """
    xf_min = float(support["xF_min"]) - 1e-9
    xf_max = float(support["xF_max"]) + 1e-9
    pt_min = float(support["pT_min"]) - 1e-9
    pt_max = float(support["pT_max"]) + 1e-9
    keep = {"cells": [], "values": [], "errors": [], "rows_per_cell": []}
    dropped = []
    ill_flags = merged.get("ill_defined") or [False] * len(merged["cells"])
    for i, (xf, pt) in enumerate(merged["cells"]):
        if abs(pt) < 1e-9:
            dropped.append({
                "cell": [float(xf), float(pt)],
                "n_rows": int(merged["rows_per_cell"][i]),
                "reason": PT_ZERO_DROP_REASON,
            })
            continue
        if ill_flags[i]:
            dropped.append({
                "cell": [float(xf), float(pt)],
                "n_rows": int(merged["rows_per_cell"][i]),
                "reason": ZERO_ERROR_DROP_REASON,
            })
            continue
        if not (xf_min <= xf <= xf_max and pt_min <= pt <= pt_max):
            dropped.append({
                "cell": [float(xf), float(pt)],
                "n_rows": int(merged["rows_per_cell"][i]),
                "reason": "outside the pinned model support (probe envelope)",
            })
            continue
        for key in keep:
            keep[key].append(merged[key][i])
    return keep, dropped


# --- Kernels and data-side profile recomputation -------------------------------


def standard_dp(pt_node: float) -> float:
    """The standard pT bin width of the NA49 grids: 0.05 GeV/c below pT 0.3,
    0.10 GeV/c above."""
    return 0.05 if pt_node <= 0.30 + 1e-9 else 0.10


def kernel_dn_dxf_column(table: np.ndarray, xf: float) -> tuple:
    """The kernel-weighted pT integration of one xF column of a set table:
    dn/dxF = sum over the pT > 0 rows of (pi sqrt(s)/E) pT d(xF,pT) dpT, with
    E = sqrt(m_pi^2 + pL^2 + pT^2), pL = xF sqrt(s)/2, and quadrature error
    propagation over the per-row statistical errors. The pT = 0.00 label
    contributes zero kernel weight and is excluded."""
    sel = (np.isclose(table[:, 0], xf, atol=1e-6)) & (table[:, 1] > 1e-9)
    total = 0.0
    err2 = 0.0
    for row in table[sel]:
        pt, d, e = float(row[1]), float(row[2]), float(row[3])
        dp = standard_dp(pt)
        p_l = float(xf) * SQRT_S_GEV / 2.0
        energy = math.sqrt(PION_MASS_GEV**2 + p_l**2 + pt * pt)
        w = (math.pi * SQRT_S_GEV / energy) * pt * dp
        total += w * d
        err2 += (w * e) ** 2
    return float(total), float(math.sqrt(err2))


def data_side_dn_dxf(table: np.ndarray) -> list:
    """Per-xF-node data-side dn/dxF with quadrature error propagation."""
    nodes = sorted({round(float(r[0]), 4) for r in table})
    rows = []
    for xf in nodes:
        value, err = kernel_dn_dxf_column(table, xf)
        rows.append({"xF": float(xf), "dn_dxF": value, "dn_dxF_err": err})
    return rows


def data_side_r_xf(set_no: int, charge: str, nu_measured: float) -> list:
    """R_i(xF) on the data side: (dn/dxF)_set,i(xF) / ((1+<nu>_i)/2
    (dn/dxF)_pp(xF)) at the common nodes of the set grid and the per-charge
    spshadrons p+p reference (the M12 seagull tables; the reference error is
    the percent error of the dN/dxF column converted to absolute)."""
    table = read_points(pa_data_dir() / SET_DENSITY_NAMES[(set_no, charge)])
    pp_t = read_ptintegrated(
        spshadrons_data_dir() / PP_REF_PTINTEGRATED_NAMES[charge])
    n_w = (1.0 + float(nu_measured)) / 2.0
    rows = []
    for rec in data_side_dn_dxf(table):
        xf = rec["xF"]
        match = np.isclose(pp_t[:, 0], xf, atol=1e-6)
        if not match.any():
            continue
        pp_row = pp_t[match][0]
        pp_val = float(pp_row[3])
        pp_err_abs = float(pp_row[4]) / 100.0 * pp_val  # percent -> absolute
        r_val = rec["dn_dxF"] / (n_w * pp_val)
        r_err = r_val * math.sqrt(
            (rec["dn_dxF_err"] / rec["dn_dxF"]) ** 2
            + (pp_err_abs / pp_val) ** 2)
        rows.append({
            "xF": xf,
            "dn_dxF_set": rec["dn_dxF"],
            "dn_dxF_set_err": rec["dn_dxF_err"],
            "dn_dxF_pp": pp_val,
            "R_data": r_val,
            "R_data_err": r_err,
        })
    return rows


# --- The closed-form E_BE-to-K mapping table (no generation) --------------------


def _load_sampler():
    if str(MIXING2006_DIR) not in sys.path:
        sys.path.insert(0, str(MIXING2006_DIR))
    from na49_mixing.glauber import NucleusSpec, WoundedNucleonSampler

    spec = NucleusSpec(**GEOMETRY_PB208)
    return WoundedNucleonSampler(spec, np.random.default_rng(1),
                                 dr_fm=PA_RUNNER_SETTINGS["glauber_dr_fm"])


def glauber_pk(sampler) -> dict:
    """P(K) per integer bin from the sampler's closed-form arithmetic."""
    out = {}
    for k in range(1, int(sampler.spec.A) + 1):
        f = float(sampler.fraction_for_wound_range(k, k))
        if f <= 0.0:
            break
        out[k] = f
    return out


def k_max_from_cumulative(pk: dict,
                          threshold: float = K_MAX_CUMULATIVE_MASS) -> int:
    """The smallest K_max with cumulative mass P(1..K_max) >= threshold."""
    cum = 0.0
    for k in sorted(pk):
        cum += pk[k]
        if cum >= threshold:
            return int(k)
    raise SystemExit(
        "the P(K) cumulative mass never reaches the truncation threshold "
        f"{threshold}")


def band_candidates(pk: dict, k_max: int,
                    widths=WOUND_BAND_WIDTHS) -> list:
    """All candidate bands [K, K+w] within the truncated support with
    positive mass: {"lo", "hi", "k_band_mean", "slice_mass"}."""
    out = []
    for lo in range(1, k_max + 1):
        for w in widths:
            hi = lo + w
            if hi > k_max:
                continue
            ks = range(lo, hi + 1)
            mass = sum(pk[k] for k in ks)
            if mass <= 0.0:
                continue
            mean = sum(k * pk[k] for k in ks) / mass
            out.append({"lo": int(lo), "hi": int(hi), "k_band_mean": mean,
                        "slice_mass": mass})
    return out


def choose_band(candidates: list, nu: float) -> dict:
    """The band minimizing |<K>_band - <nu>|; ties resolve to the narrower
    band, then to the lower band start (MAPPING_TIE_BREAK_RULE)."""
    return min(candidates,
               key=lambda c: (abs(c["k_band_mean"] - nu),
                              c["hi"] - c["lo"], c["lo"], c["hi"]))


def band_mean_and_variance(pk: dict, lo: int, hi: int) -> tuple:
    """Slice mean, variance, and mass of K inside [lo, hi] (closed form)."""
    ks = range(lo, hi + 1)
    mass = sum(pk[k] for k in ks)
    mean = sum(k * pk[k] for k in ks) / mass
    var = sum(k * k * pk[k] for k in ks) / mass - mean * mean
    return mean, var, mass


def build_mapping_table(sampler=None) -> dict:
    """The full pre-registered E_BE-to-K mapping table (closed form, no
    generation): the P(K) bins to K_max, the K_max pin, per set the chosen
    band, <K>_band, the residual (absolute and percent), the slice fraction
    versus the thesis event-count fraction, and the band-ambiguity flags
    under the +-10% systematic, plus the context rows, the pairwise-disjoint
    statement, and the unmappable-band record."""
    if sampler is None:
        sampler = _load_sampler()
    pk = glauber_pk(sampler)
    k_max = k_max_from_cumulative(pk)
    pk_pin = {str(k): pk[k] for k in range(1, k_max + 1)}
    candidates = band_candidates(pk, k_max)
    events_total = sum(r["events"] for r in THESIS_CENTRALITY_SETS.values())

    def assign(nu: float) -> dict:
        chosen = choose_band(candidates, nu)
        low = choose_band(candidates, (1.0 - NU_SYSTEMATIC_FRACTION) * nu)
        high = choose_band(candidates, (1.0 + NU_SYSTEMATIC_FRACTION) * nu)
        lo, hi = chosen["lo"], chosen["hi"]
        mean, var, mass = band_mean_and_variance(pk, lo, hi)
        frac = float(sampler.fraction_for_wound_range(lo, hi))
        ambiguous = ([low["lo"], low["hi"]] != [lo, hi]
                     or [high["lo"], high["hi"]] != [lo, hi])
        return {
            "band": [lo, hi],
            "k_band_mean": mean,
            "k_band_variance": var,
            "slice_mass": mass,
            "residual_absolute": mean - nu,
            "residual_percent": 100.0 * (mean - nu) / nu,
            "slice_fraction_model": frac,
            "band_ambiguity": bool(ambiguous),
            "alternative_band_low_nu": [low["lo"], low["hi"]],
            "alternative_band_high_nu": [high["lo"], high["hi"]],
            "nu_low": (1.0 - NU_SYSTEMATIC_FRACTION) * nu,
            "nu_high": (1.0 + NU_SYSTEMATIC_FRACTION) * nu,
        }

    sets_out = {}
    for set_no, row in THESIS_CENTRALITY_SETS.items():
        rec = assign(row["nu_measured"])
        rec.update({
            "nu_measured": row["nu_measured"],
            "n_grey_selection": row["n_grey_selection"],
            "thesis_events": row["events"],
            "thesis_event_count_fraction": row["events"] / events_total,
            "role": "set",
        })
        sets_out[str(set_no)] = rec
    context = {}
    for label, nu in CONTEXT_ROW_NU.items():
        chosen = choose_band(candidates, nu)
        context[label] = {
            "nu_measured": nu,
            "band": [chosen["lo"], chosen["hi"]],
            "k_band_mean": chosen["k_band_mean"],
            "residual_absolute": chosen["k_band_mean"] - nu,
            "role": "context_only",
        }
    overlap_pairs = []
    labels = sorted(sets_out)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            lo_i, hi_i = sets_out[labels[i]]["band"]
            lo_j, hi_j = sets_out[labels[j]]["band"]
            if not (hi_i < lo_j or hi_j < lo_i):
                overlap_pairs.append([labels[i], labels[j]])
    return {
        "k_max_pin": int(k_max),
        "k_max_cumulative_mass": K_MAX_CUMULATIVE_MASS,
        "k_max_cumulative_mass_value": float(
            sum(pk[k] for k in range(1, k_max + 1))),
        "pk_bins_1_to_kmax": pk_pin,
        "candidate_band_space": (
            "contiguous integer intervals [K, K+w] with K in 1..K_max and "
            "w in {0, 1, 2, 3} (band size 1..4), K+w <= K_max, and a slice "
            "mass strictly greater than zero"),
        "n_candidate_bands": len(candidates),
        "candidate_bands": candidates,
        "band_selection_rule": (
            "each set is assigned the band minimizing "
            "|<K>_band - <nu>_i|; " + MAPPING_TIE_BREAK_RULE),
        "nu_systematic_fraction": NU_SYSTEMATIC_FRACTION,
        "ambiguity_rule": (
            "if <nu>_i +- 0.10 <nu>_i changes the minimizing band, the set "
            "carries a band-ambiguity flag with the alternative band "
            "recorded; the comparison uses the chosen band and the "
            "ambiguity is stated as the systematic on that set's verdict"),
        "thesis_event_count_fraction_rule": (
            "the thesis cross-check fraction is events_i / sum(events of "
            "sets 1..5); the model slice fraction is a minimum-bias "
            "interaction-weighted fraction, so the comparison is a recorded "
            "cross-check with the normalization difference stated, not a "
            "gate"),
        "thesis_events_total_sets": events_total,
        "sets": sets_out,
        "context_rows": context,
        "pairwise_disjoint": not overlap_pairs,
        "overlap_pairs": overlap_pairs,
        "disjointness_statement": (
            "the five chosen bands are the sets' own unique minimizers; the "
            "overlap is recorded (the model slice predictions share wound "
            "counts on the overlapping integers, and the F-M13-1 trend "
            "carries that overlap as a stated systematic)"),
        "unmappable_bands": [],
        "unmappable_rule": (
            "a candidate band with no P(K) mass is unmappable and its "
            "comparison is dropped at pre-registration with the recorded "
            "reason"),
    }


# --- Model-support probe (small fixed-seed sample; yield-counting only) --------


_NN_CM_XF = None


def track_xf(event, particle, sqrt_s: float = SQRT_S_GEV) -> float:
    """Feynman x_F = 2 p_z^CM / sqrt(s_NN) of a final particle of ``event``.

    The p+A builder emits laboratory-frame events (target nucleons at
    rest) and records the frame and the nucleon-nucleon CM boost in the
    event metadata; their tracks are boosted to the NN CM before the
    division (na49_mixing.event_builder_pA.nn_cm_xf). The p+p reference
    events are CM events and give 2 p_z / sqrt(s) bit for bit.
    """
    global _NN_CM_XF
    if _NN_CM_XF is None:
        if str(MIXING2006_DIR) not in sys.path:
            sys.path.insert(0, str(MIXING2006_DIR))
        from na49_mixing.event_builder_pA import nn_cm_xf
        _NN_CM_XF = nn_cm_xf
    return _NN_CM_XF(event, particle.p4, sqrt_s)


def run_model_support_probe(config_path: Path, sampler=None) -> dict:
    """The pinned model-support probe: a small fixed-seed test-scale sample
    of sliced p+Pb208 events through the staged builder path at the frozen
    M8 set, measuring the final-pion xF/pT envelope that the common-support
    grids are cut against. Pilot sub-stream (PROBE_SEED_MODEL_SUPPORT),
    yield-counting and support measurement only, excluded from production
    statistics."""
    if str(MIXING2006_DIR) not in sys.path:
        sys.path.insert(0, str(MIXING2006_DIR))
    if sampler is None:
        sampler = _load_sampler()
    mapping = build_mapping_table(sampler)
    band = mapping["sets"]["3"]["band"]  # the representative mid set 3
    if tuple(band) != tuple(MODEL_SUPPORT_PROBE_BAND):
        raise SystemExit(
            f"the representative mid set 3 band {band} no longer matches "
            f"the pre-registered probe band "
            f"{list(MODEL_SUPPORT_PROBE_BAND)}")
    from dataclasses import replace

    from na49_mixing.continuum_channels import ContinuumParameters
    from na49_mixing.event_builder_pA import PAEventBuilder, PAExcitationConfig
    from na49_mixing.exchange_model import (
        ReggeExcitationParameters,
        SimpleExcitationParameters,
    )
    from na49_mixing.glauber import NucleusSpec

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    simple = SimpleExcitationParameters(**(config.get("simple") or {}))
    regge = ReggeExcitationParameters(**(config.get("regge") or {}))
    continuum = ContinuumParameters(**(config.get("continuum") or {}))
    for name, value in M8_PARAMETER_SET.items():
        group, field_name = name.split(".", 1)
        if group == "regge":
            regge = replace(regge, **{field_name: float(value)})
        else:
            continuum = replace(continuum, **{field_name: float(value)})
    pa_cfg = dict(config.get("pA") or PA_RUNNER_SETTINGS)
    builder = PAEventBuilder(
        PAExcitationConfig(
            beam_momentum=float(config.get("beam_momentum",
                                           BEAM_MOMENTUM_GEV_C)),
            production_mode=PRODUCTION_MODE,
            simple=simple,
            regge=regge,
            continuum=continuum,
            nucleus=NucleusSpec(**GEOMETRY_PB208),
            glauber_dr_fm=float(pa_cfg.get("glauber_dr_fm", 0.05)),
            max_event_attempts=int(pa_cfg.get("max_event_attempts", 100)),
            target_wound_range=tuple(band),
        ),
        np.random.default_rng(int(MODEL_SUPPORT_PROBE_SEED)),
    )
    t0 = time.perf_counter()
    xs, pts, ks, max_res = [], [], [], 0.0
    for eid in range(int(MODEL_SUPPORT_PROBE_EVENTS)):
        event = builder.generate(eid)
        event.validate()
        max_res = max(max_res,
                      float(event.conservation_residuals()["relative_max"]))
        ks.append(int(event.metadata["target_wounds"]))
        for particle in event.final_particles:
            if int(particle.pdg_id) not in PION_PDGS:
                continue
            px, py = float(particle.p4.px), float(particle.p4.py)
            xs.append(track_xf(event, particle))
            pts.append(math.hypot(px, py))
    wall = time.perf_counter() - t0
    xs_arr = np.asarray(xs)
    pts_arr = np.asarray(pts)
    nodes = {}
    for node in XF_NODES_SETS_1_4:
        nodes[f"{node:.2f}"] = int(
            np.count_nonzero(np.abs(xs_arr - node) <= 0.025 + 1e-9))
    return {
        "seed": int(MODEL_SUPPORT_PROBE_SEED),
        "n_events": int(MODEL_SUPPORT_PROBE_EVENTS),
        "band": [int(band[0]), int(band[1])],
        "wall_s": float(wall),
        "ms_per_event": 1000.0 * wall / max(len(ks), 1),
        "max_conservation_residual": float(max_res),
        "mean_target_wounds": float(np.mean(ks)) if ks else float("nan"),
        "closed_form_k_band_mean": float(mapping["sets"]["3"]["k_band_mean"]),
        "n_final_pions": int(xs_arr.size),
        "xF_min": float(xs_arr.min()),
        "xF_max": float(xs_arr.max()),
        "pT_min": float(pts_arr.min()),
        "pT_max": float(pts_arr.max()),
        "per_data_xF_node_populated": nodes,
        "role": (
            "pilot sub-stream, yield-counting and support measurement only; "
            "excluded from production statistics"),
    }


# --- Seed-block derivation and the no-collision statement -----------------------


def slice_anchor(set_no: int) -> int:
    """The production anchor of slice i (sets 1..5, i = 0..4): base + 2000*i."""
    return M13_SEED_BASE + 2000 * (int(set_no) - 1)


def slice_replica_seeds(set_no: int) -> list:
    return [slice_anchor(set_no) + 1 + k for k in range(K_REPLICAS)]


def slice_bootstrap_seed(set_no: int) -> int:
    return slice_anchor(set_no) + SLICE_BOOTSTRAP_OFFSET


def derive_seed_block() -> dict:
    """The full M13 seed-block layout (pairwise disjoint, stride >= 1000)."""
    return {
        "config_seed": CONFIG_SEED,
        "m13_seed_base": M13_SEED_BASE,
        "m13_seed_base_derivation": (
            "config seed 20260831 + 90000 (M13, the next milestone offset "
            "of the established pattern M4 +10000, M5 +20000, M7 +30000, "
            "M8 +40000, M9 +50000, M10 +60000, M11 +70000, M12 +80000)"),
        "pattern": (
            "established pattern base 20260831 + 10000 x milestone index; "
            "M4 +10000 (20270831), M5 +20000 (20280831), M7 +30000 "
            "(20290831), M8 replicas +40000 (20300831..20300838), M9 +50000 "
            "(top 20312831), M10 +60000 (top 20320889), M11 +70000 (top "
            "20330839), M12 +80000 (top 20346839)"),
        "block_span": [M13_SEED_BASE, M13_BLOCK_TOP],
        "next_milestone_offset_seed": NEXT_MILESTONE_OFFSET_SEED,
        "no_collision_statement": (
            "the block 20350831..20360830 sits strictly above the M12 "
            "consumed top 20346839 (verified against the M12 "
            "pre-registration seed_protocol block, file sha256 "
            + M12_PREREG_SHA256 + ") and strictly below the next milestone "
            "offset +100000 = 20360831; no other recorded stream occupies it"),
        "pilot_sub_block": {
            "pilot_seed_anchor": PILOT_SEED_ANCHOR,
            "pilot_top": PROBE_SEED_MODEL_SUPPORT,
            "role": (
                "probe cells and the per-slice acceptance sanity at pilot "
                "scale only, a conservation-level statement, excluded from "
                "production statistics; the P2-start pilot probes "
                "(determinism, per-slice cost cells, acceptance sanity) run "
                "at the scheduled sub-streams"),
            "sub_streams": {
                "determinism_probe_pp": PROBE_SEED_PP_DETERMINISM,
                "determinism_probe_sliced_mid_set3": PROBE_SEED_SLICED_DETERMINISM,
                "cost_cell_set1": PROBE_SEED_COST_SET1,
                "cost_cell_set2": PROBE_SEED_COST_SET2,
                "cost_cell_set3": PROBE_SEED_COST_SET3,
                "cost_cell_set4": PROBE_SEED_COST_SET4,
                "cost_cell_set5": PROBE_SEED_COST_SET5,
                "acceptance_sanity_set1": PROBE_SEED_ACCEPT_SET1,
                "acceptance_sanity_set2": PROBE_SEED_ACCEPT_SET2,
                "acceptance_sanity_set3": PROBE_SEED_ACCEPT_SET3,
                "acceptance_sanity_set4": PROBE_SEED_ACCEPT_SET4,
                "acceptance_sanity_set5": PROBE_SEED_ACCEPT_SET5,
                "model_support_probe": PROBE_SEED_MODEL_SUPPORT,
            },
            "derivation_note": (
                "sub-streams derive by +1 steps from the pilot anchor "
                "20349831 (offset -1000 from the base, the M9 pilot "
                "convention): determinism probe p+p 20349831, determinism "
                "probe sliced p+Pb at the representative mid set 3 "
                "20349832, cost cells 20349833..20349837 (300 events per "
                "slice, sets 1..5 in order), acceptance sanity cells "
                "20349838..20349842 (10000 accepted events per slice, sets "
                "1..5 in order); the model-support probe of the P1 "
                "pre-registration extends the pilot territory by one step "
                "(20349843, the acceptance-sanity sub-block top + 1), "
                "still strictly below the slice-1 anchor"),
        },
        "slices": {
            str(set_no): {
                "production_seed": slice_anchor(set_no),
                "replica_rule": (
                    "replica k (k = 0..7) uses seed = slice anchor + 1 + k "
                    "(the M8 replica convention)"),
                "replica_seeds": slice_replica_seeds(set_no),
                "bootstrap_seed": slice_bootstrap_seed(set_no),
                "bootstrap_rule": (
                    "one generator consumed sequentially; 200 resamples "
                    "(the M9 convention at the anchor +900 offset within "
                    "each 2000-stride slice sub-block)"),
            }
            for set_no in (1, 2, 3, 4, 5)
        },
        "pp_reference": {
            "production_seed": PPREF_ANCHOR,
            "derivation": (
                "slice-5 anchor + 1000 (the stride rule); the p+p "
                "reference pool for the R_i(xF) diagnostic at the same "
                "replica convention (8 x 10000), with common random "
                "numbers at the level the seed layout allows (replica k of "
                "every stream carries the same replica-index offset +1+k "
                "from its anchor)"),
            "replica_seeds": [PPREF_ANCHOR + 1 + k for k in range(K_REPLICAS)],
            "bootstrap_seed": PPREF_BOOTSTRAP_SEED,
            "bootstrap_rule": "one generator consumed sequentially; 200 resamples",
        },
        "bootstrap_offset_derivation": (
            "the five-slice density within one milestone block is the "
            "recorded reason the bootstrap offset moves from the M9 +2000 "
            "to +900 within each 2000-stride slice sub-block: anchor + 1..8 "
            "(replicas) and anchor + 900 (bootstrap) stay disjoint inside "
            "the slice sub-block, and every sub-stream stays strictly below "
            "the next block start"),
        "stride_statement": (
            "all sub-blocks are pairwise disjoint with stride at least "
            "1000 between block starts (the M10 stride rule: 1 anchor + 8 "
            "replicas + spare << 1000); every sub-stream stays below "
            + str(NEXT_MILESTONE_OFFSET_SEED)),
        "stage_labels": {
            "slices": STAGE_LABEL_SLICES,
            "pp_reference": STAGE_LABEL_PPREF,
        },
        "determinism": (
            "identical configuration and seeds reproduce identical events "
            "(spec 7.1 reproducibility test 5); the pre-registered probe "
            "verifies five bit-identical repeated evaluations at a fixed "
            "point on the sliced builder path and the minimum-bias path"),
    }


def verify_no_collision() -> dict:
    """Verify the no-collision statement against the M12 pre-registration.

    Refuses (SystemExit) when the M12 file is missing, its sha256 does not
    match the recorded value, or its seed_protocol block does not place the
    M12 consumed top strictly below the M13 block.
    """
    path = m12_preregistration_path()
    if not path.exists():
        raise SystemExit(
            "M12 pre-registration not found at "
            f"{path}; the no-collision statement cannot be verified")
    actual_sha = _sha256_file(path)
    if actual_sha != M12_PREREG_SHA256:
        raise SystemExit(
            f"M12 pre-registration sha256 {actual_sha!r} != recorded "
            f"{M12_PREREG_SHA256!r}; refusing to derive the seed block")
    prereg = json.loads(path.read_text(encoding="utf-8"))
    seeds = prereg.get("seed_protocol") or {}
    problems = []
    if seeds.get("m12_seed_base") != M12_SEED_BASE:
        problems.append(
            f"m12_seed_base: M12 file {seeds.get('m12_seed_base')!r} != "
            f"harness {M12_SEED_BASE!r}")
    ws_b = seeds.get("workstream_B") or {}
    ppref = ws_b.get("pp_reference_replica_seeds") or []
    if not ppref or int(ppref[-1]) != M12_CONSUMED_TOP:
        problems.append(
            "M12 consumed top: M12 file pp_reference replica top "
            f"{ppref[-1] if ppref else None!r} != harness top "
            f"{M12_CONSUMED_TOP}")
    if problems:
        raise SystemExit(
            "seed-block verification failed; refusing to run:\n  "
            + "\n  ".join(problems))
    if not (M12_CONSUMED_TOP < M13_SEED_BASE):
        raise SystemExit(
            "the M13 block does not sit strictly above the M12 consumed top")
    if not (M13_BLOCK_TOP < NEXT_MILESTONE_OFFSET_SEED):
        raise SystemExit("the M13 block top reaches the next milestone offset")
    starts = sorted([PILOT_SEED_ANCHOR]
                    + [slice_anchor(s) for s in (1, 2, 3, 4, 5)]
                    + [PPREF_ANCHOR])
    strides = [b - a for a, b in zip(starts, starts[1:])]
    if min(strides) < MIN_SUBBLOCK_STRIDE:
        raise SystemExit(
            f"sub-block stride violation: {strides} (the M10 stride rule "
            f"requires >= {MIN_SUBBLOCK_STRIDE})")
    tops = ([PROBE_SEED_MODEL_SUPPORT]
            + [slice_bootstrap_seed(s) for s in (1, 2, 3, 4, 5)]
            + [PPREF_BOOTSTRAP_SEED])
    if max(tops) >= NEXT_MILESTONE_OFFSET_SEED:
        raise SystemExit("an M13 sub-stream reaches the next milestone offset")
    return {
        "verified": True,
        "m12_preregistration_path": str(path),
        "m12_preregistration_sha256": actual_sha,
        "m12_seed_base": M12_SEED_BASE,
        "m12_consumed_top": M12_CONSUMED_TOP,
        "m13_block": [M13_SEED_BASE, M13_BLOCK_TOP],
        "next_milestone_offset": NEXT_MILESTONE_OFFSET_SEED,
        "sub_block_starts": starts,
        "sub_block_strides": strides,
        "sub_block_tops": sorted(tops),
        "highest_consumed_m13_seed": max(tops),
    }


# --- Harness cross-checks ---------------------------------------------------------


def cross_check_m8_record(path: Path | None = None) -> dict:
    """Verify the frozen M8 parameter set against the M8 record by sha256.

    Recomputes (a) the sha256 of the M8 record file and (b) the sha256 of the
    canonical JSON of its baryon_nucleus.frozen_pp_parameters block, and
    compares both against the recorded values.
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


def cross_check_geometry(config_path: Path | None = None) -> dict:
    """Verify the geometry against the bundled configuration: the canonical
    configuration sha256 and the nucleus/pA block values."""
    path = Path(config_path) if config_path is not None else bundled_config_path()
    if not path.exists():
        raise SystemExit(f"bundled configuration not found at {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    problems: list[str] = []
    actual_hash = canonical_json_sha256(config)
    if actual_hash != BUNDLED_CONFIG_CANONICAL_SHA256:
        problems.append(
            f"bundled configuration canonical sha256 {actual_hash!r} != "
            f"recorded {BUNDLED_CONFIG_CANONICAL_SHA256!r}")
    nucleus = dict(config.get("nucleus") or {})
    normalized = {k: (float(v) if isinstance(v, (int, float)) else v)
                  for k, v in nucleus.items()}
    if normalized != GEOMETRY_PB208:
        problems.append(
            f"bundled nucleus block {nucleus!r} != pinned geometry "
            f"{GEOMETRY_PB208!r}")
    pA = config.get("pA") or {}
    if {k: float(v) for k, v in pA.items()} != {
            k: float(v) for k, v in PA_RUNNER_SETTINGS.items()}:
        problems.append(
            f"bundled pA block {pA!r} != pinned {PA_RUNNER_SETTINGS!r}")
    if problems:
        raise SystemExit(
            "geometry cross-check failed; refusing to run:\n  "
            + "\n  ".join(problems))
    return {
        "verified": True,
        "bundled_config_path": str(path),
        "configuration_bundled_canonical_sha256": actual_hash,
        "nucleus_block": nucleus,
        "pA_block": pA,
        "sampler_anchors": geometry_anchors(),
    }


def geometry_anchors() -> dict:
    """sigma_pA and <K> from the sampler at the pinned geometry (the
    geometry-anchor sanity cross-check values; the measured anchors carry
    the systematic statement)."""
    sampler = _load_sampler()
    return {
        "Pb208": {
            "geometry": dict(GEOMETRY_PB208),
            "mean_target_wounds": float(sampler.mean_target_wounds_minbias()),
            "sigma_pA_mb": float(sampler.sigma_pA_mb()),
        },
    }


def cross_check_data_files(prereg: dict) -> dict:
    """Verify the data files (existence and the row-count and grid
    inventory) by recomputing the full inventory from the tables and
    comparing it against the pre-registration's embedded record."""
    support = prereg.get("common_support", {}).get("model_support_pin")
    if not support:
        raise SystemExit("pre-registration carries no model-support pin")
    problems: list[str] = []
    fresh = build_data_inventory(support=support)
    stored = prereg.get("data_inventory") or {}
    if set(fresh) != set(stored):
        problems.append(
            f"data_inventory keys drift: fresh {sorted(fresh)} != stored "
            f"{sorted(stored)}")
    for key in sorted(set(fresh) & set(stored)):
        if fresh[key] != stored[key]:
            differing = [k for k in sorted(set(fresh[key]) | set(stored[key]))
                         if fresh[key].get(k) != stored[key].get(k)]
            problems.append(
                f"data_inventory[{key}] drift in fields {differing[:6]}")
    return {
        "verified": not problems,
        "files": {key: fresh[key]["file"] for key in sorted(fresh)},
        "problems": problems,
    }


def cross_check_mapping_table(prereg: dict, sampler=None) -> dict:
    """Deterministic recomputation of the closed-form mapping arithmetic
    against the pre-registration's embedded mapping table (exact equality;
    the same code path produces the same doubles)."""
    fresh = build_mapping_table(sampler)
    stored = prereg.get("mapping_table") or {}
    if fresh != stored:
        differing = [key for key in sorted(set(fresh) | set(stored))
                     if fresh.get(key) != stored.get(key)]
        raise SystemExit(
            "mapping-table recomputation drift; refusing to run (the "
            "closed-form arithmetic is deterministic):\n  "
            + "\n  ".join(differing[:10]))
    return {"verified": True, "k_max_pin": fresh["k_max_pin"],
            "n_sets": len(fresh["sets"]),
            "pairwise_disjoint": fresh["pairwise_disjoint"]}


# The recorded platform tolerance of the mapping-table recomputation on the
# gateway workers (DEV-2): every FLOAT leaf of the recomputed table within
# this relative tolerance of the pre-registered value, every non-numeric
# leaf (the chosen bands, the flags, the rules) exact. The local harness
# keeps the exact-equality check and is the recorded authority.
MAPPING_PLATFORM_REL_TOL = 1e-12
_PREREG_CHECK_PLATFORM_NOTES: list[str] = []


def prereg_check_platform_notes() -> list:
    """The platform-tolerance notes of the last preregistration check
    (empty when every comparison matched exactly; the DEV-2 note appears
    when the gateway-side recomputation matched under the recorded
    tolerance)."""
    return list(_PREREG_CHECK_PLATFORM_NOTES)


def mapping_table_matches_platform(fresh: dict, stored: dict,
                                   rel_tol: float = MAPPING_PLATFORM_REL_TOL
                                   ) -> bool:
    """The recorded platform tolerance for the gateway-side recomputation
    of the closed-form mapping table (DEV-2): every FLOAT leaf within
    ``rel_tol`` relative of the pre-registered value and every non-numeric
    leaf exact."""
    def walk(a, b) -> bool:
        if isinstance(a, dict) and isinstance(b, dict):
            return (set(a) == set(b)
                    and all(walk(a[k], b[k]) for k in a))
        if isinstance(a, list) and isinstance(b, list):
            return (len(a) == len(b)
                    and all(walk(x, y) for x, y in zip(a, b)))
        if (isinstance(a, float) and not isinstance(a, bool)
                and isinstance(b, float) and not isinstance(b, bool)):
            denom = max(abs(a), abs(b), 1e-300)
            return abs(a - b) / denom <= rel_tol
        return a == b

    return walk(fresh, stored)


def cross_check_preregistration(prereg: dict) -> None:
    """Refuse to run when the pre-registration and the constants disagree.

    The platform-notes list is reset per call: the notes describe the
    outcome of THIS check (the DEV-2 tolerant match when it fires), so a
    repeated check in one process (the determinism double run) yields
    identical provenance on both builds."""
    del _PREREG_CHECK_PLATFORM_NOTES[:]
    problems: list[str] = []

    def check(name: str, expected, actual) -> None:
        if expected != actual:
            problems.append(
                f"{name}: pre-registration {expected!r} != harness {actual!r}")

    check("card", prereg.get("card"), CARD)
    check("milestone", prereg.get("milestone"), MILESTONE)
    check("phase", prereg.get("phase"), PHASE)
    check("run_description", prereg.get("run_description"), RUN_DESCRIPTION)
    check("date", prereg.get("date"), "2026-09-11")
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
    check("pa_runner_settings", geometry.get("pa_runner_settings"),
          PA_RUNNER_SETTINGS)
    anchors = prereg.get("normalization_anchors", {})
    check("sigma_inel_pPb_mb", anchors.get("sigma_inel_pPb_mb"), SIGMA_INEL_PPb_MB)
    check("sigma_inel_pPb_err_mb", anchors.get("sigma_inel_pPb_err_mb"),
          SIGMA_INEL_PPb_ERR_MB)
    check("nu_pPb_measured", anchors.get("nu_pPb_measured"), NU_PPB_MEASURED)
    check("nu_systematic_fraction", anchors.get("nu_systematic_fraction"),
          NU_SYSTEMATIC_FRACTION)
    thesis = prereg.get("thesis_centrality_table", {})
    check("thesis_centrality_sets", thesis.get("sets"), THESIS_CENTRALITY_SETS)
    check("thesis_context_nu", thesis.get("context_nu"), CONTEXT_ROW_NU)
    check("thesis_events_eliminated_above_set5",
          thesis.get("events_eliminated_above_set5"),
          THESIS_EVENTS_ELIMINATED_ABOVE_SET5)
    mapping = prereg.get("mapping_table", {})
    fresh_mapping = build_mapping_table()
    if mapping != fresh_mapping:
        # The recorded platform tolerance (DEV-2): the closed-form P(K)
        # integrals shift at the 1e-16 relative level between numpy
        # versions; a gateway-side recomputation matches when every FLOAT
        # leaf sits within MAPPING_PLATFORM_REL_TOL relative and every
        # non-numeric leaf (the chosen bands, the flags, the rules) is
        # exact. The local harness keeps the exact-equality check.
        if mapping_table_matches_platform(fresh_mapping, mapping):
            _PREREG_CHECK_PLATFORM_NOTES.append(
                "mapping_table: matched under the recorded platform "
                "tolerance (DEV-2, every FLOAT leaf within "
                f"{MAPPING_PLATFORM_REL_TOL:g} relative, every non-numeric "
                "leaf exact); the exact-equality check passes locally and "
                "is the recorded authority (harness_cross_check.json)")
        else:
            problems.append(
                f"mapping_table: pre-registration {mapping!r} != harness "
                f"{fresh_mapping!r}")
    check("seed_protocol", prereg.get("seed_protocol"), derive_seed_block())
    stats = prereg.get("statistics", {})
    check("k_replicas", stats.get("k_replicas"), K_REPLICAS)
    check("menu_events_per_replica", stats.get("menu_events_per_replica"),
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
    check("acceptance_sanity_events", stats.get("acceptance_sanity_events"),
          ACCEPTANCE_SANITY_EVENTS)
    check("pp_reference_events_per_replica",
          stats.get("pp_reference_events_per_replica"),
          PP_REF_EVENTS_PER_REPLICA)
    bands = prereg.get("verdict_bands", {})
    check("k_pass", bands.get("k_pass"), K_PASS)
    check("k_tension", bands.get("k_tension"), K_TENSION)
    check("falsifier_directions", bands.get("falsifier_directions"),
          FALSIFIER_DIRECTIONS)
    probe = prereg.get("model_support_probe", {})
    check("model_support_probe.seed", probe.get("seed"),
          MODEL_SUPPORT_PROBE_SEED)
    check("model_support_probe.n_events", probe.get("n_events"),
          MODEL_SUPPORT_PROBE_EVENTS)
    check("model_support_probe.band", probe.get("band"),
          list(MODEL_SUPPORT_PROBE_BAND))
    if problems:
        raise SystemExit(
            "pre-registration mismatch; refusing to run (the pre-registered "
            "protocol is authoritative):\n  " + "\n  ".join(problems))


def run_harness_cross_check(prereg_path: Path, out_path: Path) -> dict:
    """Run every pre-generation cross-check and record the outcome.

    The pre-registration's constants are verified verbatim, the M8 record
    and the bundled geometry by sha256, the data files by recomputed
    inventory, the seed block against the M12 pre-registration, and the
    mapping table by deterministic recomputation.
    """
    prereg_path = Path(prereg_path)
    record: dict = {
        "cross_check_command": CROSS_CHECK_COMMAND,
        "pre_registration_path": str(prereg_path),
        "written_utc": _utc_now_iso(),
    }
    try:
        prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
        m8 = cross_check_m8_record()
        geo = cross_check_geometry()
        cross_check_preregistration(prereg)
        data = cross_check_data_files(prereg)
        seed = verify_no_collision()
        mapping = cross_check_mapping_table(prereg)
        if data["problems"]:
            raise SystemExit(
                "data-file cross-check failed; refusing to run:\n  "
                + "\n  ".join(data["problems"]))
        seed_block = verify_no_collision()
        record.update({
            "status": "pass",
            "exit_code": 0,
            "pre_registration_sha256": _sha256_file(prereg_path),
            "m8_crosscheck": m8,
            "geometry_crosscheck": geo,
            "data_files_crosscheck": data,
            "seed_block_crosscheck": seed_block,
            "mapping_table_crosscheck": mapping,
        })
    except SystemExit as exc:
        record.update({
            "status": "fail",
            "exit_code": 1,
            "pre_registration_sha256": (_sha256_file(prereg_path)
                                        if Path(prereg_path).exists() else None),
            "error": str(exc),
        })
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(_to_jsonable(record), indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    if record["status"] != "pass":
        print(json.dumps(record, indent=1, sort_keys=True))
        raise SystemExit(1)
    return record


# --- Data inventory assembly (pure table reads) ---------------------------------


def build_data_inventory(sampler=None, support=None) -> dict:
    """Per set/charge: the file name, the grid inventory, the merged
    unique-cell grid, the registered common support against the pinned
    model support, and the dropped-cell record (pure table reads)."""
    if support is None:
        raise ValueError(
            "build_data_inventory requires the pinned model-support record")
    out = {}
    for (set_no, charge), filename in sorted(SET_DENSITY_NAMES.items()):
        table = read_points(pa_data_dir() / filename)
        merged = merge_unique_cells(table)
        registered, dropped = common_support(merged, support)
        out[f"set{set_no}_{charge}"] = {
            "file": filename,
            "set": int(set_no),
            "charge": charge,
            "inventory": grid_inventory(table),
            "merged_unique_cells": merged,
            "registered": registered,
            "dropped": dropped,
            "n_registered_cells": len(registered["cells"]),
        }
    return out


def merge_fixture_records() -> dict:
    """The pinned merge-rule fixture: a few real duplicated-label cells with
    their rows, merged values, and merged errors (asserted by a test)."""
    table = read_points(pa_data_dir() / SET_DENSITY_NAMES[(1, "pi+")])
    cells: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for row in table:
        key = (round(float(row[0]), 4), round(float(row[1]), 4))
        cells.setdefault(key, []).append((float(row[2]), float(row[3])))
    fixture = {}
    for key in ((0.0, 0.1), (0.0, 0.2), (0.05, 0.1)):
        rows = cells[key]
        fixture[f"{key[0]:.2f}|{key[1]:.2f}"] = {
            "rows": [[float(v), float(s)] for v, s in rows],
            "merged_value": merged_value(rows),
            "merged_error": merged_error(rows),
        }
    return fixture


# --- Phase P2: production path (card requirement sequence 1-6) -------------------
#
# Sequence per the card: (1) probes first (determinism, per-slice cost cells,
# per-slice acceptance sanity, gateway capacity; production refused on a
# missed expectation without a recorded deviation), (2) production generation
# at the menu outcome on the gateway grid with the recorded serial fallback,
# (3) conservation and wound bookkeeping, (4) the geometry anchor cross-check,
# (5) the comparison exactly as pre-registered, (6) the report. The
# pre-registered constants above are never re-derived here; the P2 code only
# consumes them.

PHASE_P2 = "P2"
WILSON_Z = 1.959963984540054  # two-sided 95% (the M5/M9 convention)

# Derived pilot sub-stream (recorded derivation; the seed_protocol block of
# the pre-registration allows additional sub-streams derived from the
# anchors with the derivation recorded): the minimum-bias builder
# determinism unit extends the pilot territory by one step beyond the
# model-support probe top 20349843 (pilot anchor + 13), still strictly
# below the slice-1 anchor 20350831. The pre-registered sub-stream
# PROBE_SEED_PP_DETERMINISM (20349831, "determinism probe p+p") runs the
# p+p reference generator unit; the min-bias builder unit (named by the
# card's P1 requirement 7 probe list) consumes this derived seed.
PROBE_SEED_MINBIAS_DETERMINISM = 20349844

SLICE_MANIFEST_SCHEMA = "staged_centrality_sets_cells_manifest/v1"
SLICE_REPLICA_SCHEMA = "staged_centrality_sets_slice_replica/v1"
PPREF_REPLICA_SCHEMA = "staged_centrality_sets_ppref_replica/v1"
PROBE_DETERMINISM_SCHEMA = "staged_centrality_sets_probe_determinism/v1"
PROBE_COST_SCHEMA = "staged_centrality_sets_probe_cost/v1"
PROBE_ACCEPT_SCHEMA = "staged_centrality_sets_probe_acceptance/v1"
PROBE_GATEWAY_SCHEMA = "staged_centrality_sets_probe_gateway/v1"
PILOT_SCHEMA = "staged_centrality_sets_pilot/v1"
DETERMINISM_PRODUCTION_SCHEMA = "staged_centrality_sets_determinism_production/v1"
REPORT_SCHEMA = "staged_centrality_sets_centrality_report/v1"

SLICE_MANIFEST_NAME = "centrality_cells_manifest.json"
PILOT_RECORD_NAME = "pilot_p2.json"
GATEWAY_MANIFEST_NAME = "gateway_manifest_p2.json"
REPORT_JSON_NAME = "centrality_report.json"
REPORT_MD_NAME = "REPORT.md"

# The accepted-versus-analytic acceptance sanity (the pre-registered rule):
# bin-by-bin 95% binomial band of the sliced Glauber analytic distribution
# (the count-scale normal-approximation band N p_k +- z sqrt(N p_k (1-p_k))),
# qualifying bins with expected count N p_k at least 5.
ACCEPTANCE_BAND_MIN_EXPECTED = 5.0

# The band-ambiguity systematic statement (recorded per ambiguous set).
BAND_AMBIGUITY_SYSTEMATIC_STATEMENT = (
    "the +-10% VENUS systematic on <nu> moves the minimizing band of every "
    "mapped set (the pre-registration records band_ambiguity true with the "
    "alternative bands); the comparison uses the chosen band and the "
    "ambiguity is stated as the systematic on that set's verdicts: the "
    "alternative bands imply different generated slices, so the set's "
    "verdict is conditional on the chosen band and the mapping residual "
    "delta_i plus the band ambiguity together bound the mapping "
    "systematic. The band-ambiguity flags are recorded systematics, not "
    "falsifiers (the pre-registered falsifier list)")

# Recorded deviations of the P2 production path (fixed before any production
# verdict was published; the pre-registration file is unchanged unless a
# deviation's statement says otherwise).
P2_DEVIATIONS: list[dict] = [
    {
        "id": "DEV-2",
        "statement": (
            "the harness cross-check's exact-equality recomputation of the "
            "closed-form mapping table fails on the gateway workers: the "
            "workers run a different platform (numpy 2.0.2 / python 3.9.25 "
            "on the cluster; the pre-registration was written locally), "
            "and the closed-form P(K) integrals shift at the 1e-16 "
            "relative level between the numpy versions (a remote "
            "diagnostic diff counted 64 differing FLOAT leaves with a "
            "maximum relative deviation of 3.7e-14 and ZERO non-float "
            "diffs; the chosen bands [2,4], [4,4], [4,7], [6,6], [7,7], "
            "the ambiguity flags, the K_max pin, and every rule string "
            "are identical). The harness comparison is executed with the "
            "recorded platform tolerance: every FLOAT leaf of the "
            "recomputed mapping table within 1e-12 relative of the "
            "pre-registered value and every non-numeric leaf exact. The "
            "local harness (the run record's authority, recorded in "
            "harness_cross_check.json) keeps the exact-equality check, "
            "which passes locally. Recorded before any production verdict "
            "was published; no pre-registration file, seed, or protocol "
            "constant change (the pre-registration sha256 is unchanged); "
            "the generated events are unaffected (the sampling path never "
            "consumes the mapping floats)"),
        "coverage": {"probe": "gateway_preregistration_check",
                     "field": "mapping_table"},
        "test": (
            "the helper-suite pin that the platform-tolerance comparator "
            "accepts the recorded 1e-16-class float drift and refuses any "
            "structural (non-float) difference, and that the P2 deviation "
            "registry records DEV-2"),
    },
    {
        "id": "DEV-1",
        "statement": (
            "the pilot-scale per-slice acceptance sanity of set 1 (10000 "
            "accepted events, seed 20349838, band [2, 4]) missed its "
            "pre-registered bin-by-bin expectation: the k = 2 bin holds "
            "3769 accepted events against the 95% binomial band "
            "[3575, 3764] of the sliced Glauber analytic distribution, a "
            "+2.06 sigma excursion (the slice-mean wound count 2.9209 "
            "against the closed form 2.9370 sits -2.0 standard errors "
            "inside its 4-standard-error tolerance). The other fourteen "
            "qualifying bins of the five pilot slices sit inside their "
            "bands, and the excursion is of the size expected once among "
            "fifteen 95% bands under the correct-model null (the per-bin "
            "false-excision rate is 5%, so the expected number of "
            "excursions among the fifteen pilot bins is 0.75 and observing "
            "one is unremarkable). No generation or binning bug is "
            "implicated: the pilot's sliced-stream draw pin (the builder "
            "draws equal sample_event_in_range exactly) passed, the "
            "conservation record is clean at the 1e-12 level, and the "
            "single-bin slices (sets 2, 4, 5) hold trivially. Recorded "
            "before any production verdict was published; no seed, "
            "generation parameter, protocol constant, or pre-registration "
            "change (the pre-registration sha256 is unchanged, no seed "
            "re-derivation is required). The binding acceptance record is "
            "the production-scale sanity (80000 accepted events per "
            "slice), where the same binomial band tightens by a factor of "
            "sqrt(8); a real K-dependent acceptance bias would appear "
            "there and would be recorded as the acceptance-sanity result "
            "on that set's interpretation"),
        "coverage": {"probe": "acceptance_sanity_pilot", "set": "set1"},
        "test": (
            "the helper-suite pin that the P2 deviation registry records "
            "the set1 pilot acceptance excursion (id DEV-1, coverage set1) "
            "and that the pilot gate treats exactly the covered failures "
            "as recorded; the external-data pin that the committed pilot "
            "artifact carries the recorded k = 2 excursion values"),
    },
    {
        "id": "DEV-3",
        "statement": (
            "the first production determinism double-run on the gateway "
            "failed spuriously (bit_identical false for both the set-1 "
            "slice and the ppref units): the DEV-2 platform-note list of "
            "the pre-registration check accumulated across the two "
            "payload builds in one process, so the second build's "
            "provenance carried two notes where the first carried one and "
            "the payloads differed in that provenance field alone. The "
            "fix resets the notes at each cross_check_preregistration "
            "call (the notes describe the outcome of the current check); "
            "no seed, generation, manifest, or physics change, and the "
            "generated events are identical by construction. The re-run "
            "of the determinism arm (same seeds, same manifest) completed "
            "bit-identical on both units; the 48 replica payloads are "
            "unaffected (one check per worker process, one note each). "
            "Recorded before any production verdict was published; the "
            "pre-registration sha256 is unchanged"),
        "coverage": {"probe": "determinism_production",
                     "field": "payload_provenance"},
        "test": (
            "the helper-suite pin that cross_check_preregistration resets "
            "the platform-note list per call (two consecutive checks leave "
            "the notes non-accumulating), plus the recorded re-run of the "
            "determinism arm"),
    },
    {
        "id": "DEV-4",
        "statement": (
            "the per-event sparse cell triples of the replica payloads "
            "(the event-level bootstrap input) were encoded with the "
            "particle position index standing in for the event index, so "
            "the triples exceeded the per-replica event count and the "
            "collection's dense rebuild raised an index error. The "
            "per-cell totals, the pooled yields, the per-node dn/dxF, "
            "and the R_i(xF) records derive from the per-cell counts and "
            "are unaffected; the bug is confined to the payload's "
            "per-event encoding field. The fix passes the event-index "
            "array explicitly (sparse_cell_triples(ev, assign)); the "
            "affected payloads are re-produced at the same pre-registered "
            "seeds (the event generation is unchanged, so no seed "
            "re-derivation is required; the regenerated events are "
            "identical by the verified determinism). Recorded before any "
            "production verdict was published; the pre-registration "
            "sha256 is unchanged"),
        "coverage": {"probe": "slice_replica_payload",
                     "field": "per_event_cell_triples"},
        "test": (
            "the helper-suite round-trip pin updated to the (event, cell) "
            "pairing with multi-particle events, plus the re-produced "
            "payloads collected with the corrected triples"),
    },
    {
        "id": "DEV-5",
        "statement": (
            "the pooled-yield bootstrap values of the collection's "
            "bootstrap unit multiplied the resampled per-cell counts by "
            "the per-cell (pT/E) dxF dpT kernel factor while omitting the "
            "density Jacobian, so the recorded pooled-yield percentile "
            "intervals sat at the kernel-factor scale (about 3e-5) "
            "instead of the yield scale (about 0.14 for set 1). The point "
            "estimates, the replica-spread MC errors, the verdicts, the "
            "per-node dn/dxF values, and the per-cell bootstrap sigmas "
            "are unaffected (they use the explicit kernel form or the "
            "correct collapse). The fix uses the pinned pooled-kernel "
            "collapse directly (the resample yield is sum(n)/(N pi "
            "sqrt(s))), the same algebraic identity the collection's "
            "explicit form satisfies and the helper suite pins. The "
            "collection is re-run (no event regeneration; the artifacts "
            "store the per-cell counts and per-event triples, so the "
            "bootstrap recomputes from them). Recorded before any "
            "production verdict was published; the pre-registration "
            "sha256 is unchanged"),
        "coverage": {"probe": "collection_bootstrap",
                     "field": "yield_percentile_interval"},
        "test": (
            "the helper-suite pin that the bootstrap unit's pooled-yield "
            "values equal sum(n)/(N pi sqrt(s)) on a synthetic dense "
            "matrix, alongside the existing explicit-form collapse pin"),
    },
]

# The pilot-gate coverage map: a missed pilot expectation whose slice is
# covered by a recorded deviation in P2_DEVIATIONS is treated as recorded
# (the card's refusal rule: production is refused on a missed expectation
# WITHOUT a recorded deviation).
PILOT_DEVIATION_COVERAGE = {
    "acceptance_sanity": {"set1": "DEV-1"},
}


def _write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_jsonable(payload), indent=1, sort_keys=True) + "\n",
        encoding="utf-8")


def canonical_configuration_hash(config_path: Path) -> str:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return canonical_json_sha256(config)


def build_parameter_set(config: dict):
    """Bundled-config parameter dataclasses with the ten frozen M8 values
    applied (the M12 build_parameter_set convention; refuses on any
    application failure and asserts the H3 decay kernel)."""
    from dataclasses import replace

    from na49_mixing.continuum_channels import ContinuumParameters
    from na49_mixing.exchange_model import (
        ReggeExcitationParameters,
        SimpleExcitationParameters,
    )

    simple = SimpleExcitationParameters(**(config.get("simple") or {}))
    regge = ReggeExcitationParameters(**(config.get("regge") or {}))
    continuum = ContinuumParameters(**(config.get("continuum") or {}))
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


def build_pa_builder(seed: int, config_path: Path, band=None,
                     config: dict | None = None):
    """The staged p+A builder at the frozen M8 set and the pinned geometry,
    with the optional pre-registered centrality slice (the P1 wiring:
    ``PAExcitationConfig.target_wound_range``)."""
    if str(MIXING2006_DIR) not in sys.path:
        sys.path.insert(0, str(MIXING2006_DIR))
    from na49_mixing.event_builder_pA import PAEventBuilder, PAExcitationConfig
    from na49_mixing.glauber import NucleusSpec

    raw_config = (json.loads(Path(config_path).read_text(encoding="utf-8"))
                  if config is None else config)
    simple, regge, continuum, applied = build_parameter_set(raw_config)
    beam = float(raw_config.get("beam_momentum", BEAM_MOMENTUM_GEV_C))
    pa_cfg = dict(raw_config.get("pA") or PA_RUNNER_SETTINGS)
    builder = PAEventBuilder(
        PAExcitationConfig(
            beam_momentum=beam,
            production_mode=PRODUCTION_MODE,
            simple=simple,
            regge=regge,
            continuum=continuum,
            nucleus=NucleusSpec(**GEOMETRY_PB208),
            glauber_dr_fm=float(pa_cfg.get("glauber_dr_fm", 0.05)),
            max_event_attempts=int(pa_cfg.get("max_event_attempts", 100)),
            target_wound_range=(None if band is None
                                else (int(band[0]), int(band[1]))),
        ),
        np.random.default_rng(int(seed)),
    )
    return builder, applied, beam


_last_builder_summary: dict = {}


def generate_slice_events(n_events: int, seed: int, config_path: Path,
                          band) -> tuple:
    """One sliced p+Pb208 sample through the builder path at the frozen M8
    set (conservation re-asserted per event at 1e-10). The deterministic
    part of the builder bookkeeping is stored in the module-level
    ``_last_builder_summary`` (the generation counters; the wall fields
    stay out of the payloads and live in the caller's return value)."""
    builder, applied, beam = build_pa_builder(seed, config_path, band)
    t0 = time.perf_counter()
    events = [builder.generate(index) for index in range(int(n_events))]
    wall = time.perf_counter() - t0
    max_res = 0.0
    for event in events:
        event.validate()
        max_res = max(max_res,
                      float(event.conservation_residuals()["relative_max"]))
        event.metadata["random_seed"] = int(seed)
        event.metadata["random_substream"] = (
            f"{STAGE_LABEL_SLICES}:set_band_{band[0]}_{band[1]}:{event.event_id}")
    summary = {
        "n_events": float(len(events)),
        "mean_target_wounds": float(np.mean([
            float(event.metadata["target_wounds"]) for event in events]))
        if events else float("nan"),
        "mean_final_multiplicity": float(
            sum(len(event.final_particles) for event in events)
            / max(len(events), 1)),
        "max_conservation_residual": float(max_res),
        "generation_attempts": int(builder.generation_attempts),
        "generation_accepted": int(builder.generation_accepted),
        "generation_rejected": int(builder.generation_rejected),
        "failure_reason_counts": dict(builder.failure_reason_counts),
        "wall_s": float(wall),
    }
    global _last_builder_summary
    _last_builder_summary = {
        key: summary[key] for key in ("generation_attempts",
                                      "generation_accepted",
                                      "generation_rejected",
                                      "failure_reason_counts")}
    return events, summary, wall, applied, beam


def generate_minbias_events(n_events: int, seed: int, config_path: Path):
    """The minimum-bias builder sample (no target_wound_range); probe-scale
    use only in this card (the min-bias p+A record stands as M12's)."""
    builder, applied, beam = build_pa_builder(seed, config_path, None)
    t0 = time.perf_counter()
    events = [builder.generate(index) for index in range(int(n_events))]
    wall = time.perf_counter() - t0
    max_res = 0.0
    for event in events:
        event.validate()
        max_res = max(max_res,
                      float(event.conservation_residuals()["relative_max"]))
        event.metadata["random_seed"] = int(seed)
        event.metadata["random_substream"] = (
            f"centrality_sets_minbias_probe:{event.event_id}")
    summary = {
        "n_events": float(len(events)),
        "mean_target_wounds": float(np.mean([
            float(event.metadata["target_wounds"]) for event in events]))
        if events else float("nan"),
        "mean_final_multiplicity": float(
            sum(len(event.final_particles) for event in events)
            / max(len(events), 1)),
        "max_conservation_residual": float(max_res),
        "wall_s": float(wall),
    }
    return events, summary, wall, applied, beam


def generate_pp_events(n_events: int, seed: int, config_path: Path,
                       stage_label: str = STAGE_LABEL_PPREF):
    """The p+p reference sample via _generate_pp (regge mode, the frozen M8
    parameters; the ppref pool generator of the R_i(xF) diagnostic)."""
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
        event.validate()
        max_res = max(max_res,
                      float(event.conservation_residuals()["relative_max"]))
        event.metadata["random_seed"] = int(seed)
        event.metadata["random_substream"] = f"{stage_label}:{event.event_id}"
    summary = dict(summary)
    summary["max_conservation_residual"] = float(max_res)
    summary["wall_s"] = float(wall)
    return events, summary, wall, applied, beam


# --- Model-side binning (the pre-registered conventions) --------------------------


def cell_xf_half_width(xf_node: float) -> float:
    """The pinned xF cell half-width: 0.05-wide bins below xF 0.4, 0.10-wide
    above (every registered set node sits at xF <= 0.40)."""
    return (0.05 if xf_node <= 0.40 + 1e-9 else 0.10) / 2.0


def cell_pt_half_width(pt_node: float) -> float:
    """The pinned pT cell half-width convention (the M12 rule): 0.05-wide
    bins below pT 0.3 and 0.10-wide above, halved."""
    return (0.05 if pt_node <= 0.30 + 1e-9 else 0.10) / 2.0


def cell_full_widths(xf: float, pt: float) -> tuple:
    return (2.0 * cell_xf_half_width(xf), 2.0 * cell_pt_half_width(pt))


def pion_energy(xf: float, pt: float,
                mass: float = PION_MASS_GEV) -> float:
    return math.sqrt(mass * mass + (float(xf) * SQRT_S_GEV / 2.0) ** 2
                     + float(pt) * float(pt))


def bin_first_match(xf_arr, pt_arr, cells: list) -> tuple:
    """The pinned first-match binning: each particle enters the first cell
    (in the given cell order) whose node lies within the cell half-widths in
    both variables; unplaced particles carry -1 (the recorded overflow)."""
    xf_arr = np.asarray(xf_arr, dtype=float)
    pt_arr = np.asarray(pt_arr, dtype=float)
    counts = np.zeros(len(cells), dtype=np.int64)
    assign = np.full(len(xf_arr), -1, dtype=np.int64)
    for i, (xf_i, pt_i) in enumerate(cells):
        mask = ((assign < 0)
                & (np.abs(xf_arr - xf_i) <= cell_xf_half_width(xf_i) + 1e-9)
                & (np.abs(pt_arr - pt_i)
                   <= cell_pt_half_width(pt_i) + 1e-9))
        counts[i] = int(np.count_nonzero(mask))
        assign[mask] = i
    return counts, assign


def species_arrays(events, pdg_id: int, sqrt_s: float = SQRT_S_GEV) -> dict:
    """Per-particle arrays of one species: xF, pT, event index, and the
    source role (projectile/target; spectators never decay into pions)."""
    xs, ps, evs, roles = [], [], [], []
    for ev_index, event in enumerate(events):
        for particle in event.final_particles:
            if int(particle.pdg_id) != int(pdg_id):
                continue
            xs.append(track_xf(event, particle, sqrt_s))
            ps.append(math.hypot(float(particle.p4.px),
                                 float(particle.p4.py)))
            evs.append(ev_index)
            source_id = int(particle.source_id)
            role = ("unknown" if source_id < 0
                    else str(event.sources[source_id].role))
            roles.append(role)
    return {
        "xf": np.asarray(xs, dtype=float),
        "pt": np.asarray(ps, dtype=float),
        "ev": np.asarray(evs, dtype=np.int64),
        "role": roles,
    }


def sparse_cell_triples(ev_arr, assign: np.ndarray) -> list:
    """Per-event cell counts as a flat sparse triple list
    [event, cell, count, ...] (the event-level bootstrap input; aligned
    with the manifest cell order). ``ev_arr`` carries the event index of
    each placed particle and ``assign`` its cell assignment; unplaced
    particles (assign -1) are excluded, the overflow is recorded
    separately."""
    ev_arr = np.asarray(ev_arr, dtype=np.int64)
    assign = np.asarray(assign, dtype=np.int64)
    if ev_arr.shape != assign.shape:
        raise ValueError(
            "sparse_cell_triples requires matched event-index and "
            f"assignment arrays, got {ev_arr.shape} vs {assign.shape}")
    placed = assign >= 0
    if not placed.any():
        return []
    pairs = np.stack([ev_arr[placed], assign[placed]], axis=1)
    uniq, counts = np.unique(pairs, axis=0, return_counts=True)
    triples: list[int] = []
    for (ev, cell), n in zip(uniq, counts):
        triples.extend([int(ev), int(cell), int(n)])
    return triples


def expand_triples(triples: list, n_events: int, n_cells: int) -> np.ndarray:
    """The inverse of sparse_cell_triples: the dense per-event count
    matrix (n_events x n_cells)."""
    dense = np.zeros((int(n_events), int(n_cells)), dtype=np.int64)
    arr = np.asarray(triples, dtype=np.int64).reshape(-1, 3)
    dense[arr[:, 0], arr[:, 1]] = arr[:, 2]
    return dense


def pooled_kernel_yield_model(counts, cells, n_events: int) -> float:
    """The pre-registered pooled kernel yield on the model side:
    Y = sum over the cells of (pT/E) d(xF,pT) dxF dpT with the model
    per-event density d = E/(pi sqrt(s) pT) * n/(N dxF dpT); the Jacobian
    factors cancel algebraically, Y = sum(n) / (N pi sqrt(s)) (the M12
    pooled-kernel collapse; the explicit form is pinned by a test)."""
    total = 0.0
    for (xf, pt), n in zip(cells, counts):
        if not n:
            continue
        dxf, dpt = cell_full_widths(xf, pt)
        energy = pion_energy(xf, pt)
        density = (energy / (math.pi * SQRT_S_GEV * pt)) * (
            float(n) / (float(n_events) * dxf * dpt))
        total += (pt / energy) * density * dxf * dpt
    return float(total)


def pooled_kernel_yield_data(values, errors, cells) -> tuple:
    """The pre-registered pooled kernel yield on the data side (the merged
    unique-cell values) with the quadrature error propagation."""
    total = 0.0
    err2 = 0.0
    for (xf, pt), v, e in zip(cells, values, errors):
        dxf, dpt = cell_full_widths(xf, pt)
        energy = pion_energy(xf, pt)
        w = (pt / energy) * dxf * dpt
        total += w * float(v)
        err2 += (w * float(e)) ** 2
    return float(total), float(math.sqrt(err2))


def model_dn_dxf_nodes(counts, cells, n_events: int, nodes) -> dict:
    """The model-side kernel-weighted pT integration per xF node over the
    registered cells at that node: dn/dxF = sum of (pi sqrt(s)/E) pT
    d(xF,pT) dpT with the model binned densities (the pre-registered
    definition; collapses algebraically to sum(n)/(N dxF))."""
    out = {}
    for node in nodes:
        total = 0.0
        for (xf, pt), n in zip(cells, counts):
            if abs(xf - float(node)) > 1e-6 or not n:
                continue
            dxf, dpt = cell_full_widths(xf, pt)
            energy = pion_energy(xf, pt)
            density = (energy / (math.pi * SQRT_S_GEV * pt)) * (
                float(n) / (float(n_events) * dxf * dpt))
            total += (math.pi * SQRT_S_GEV / energy) * pt * density * dpt
        out[f"{float(node):.4g}"] = float(total)
    return out


def verdict_from_z(z) -> str:
    """The pre-registered band rule: pass |z| <= k_pass, tension
    k_pass < |z| <= k_tension, beyond_tension |z| > k_tension."""
    if z is None:
        return "undefined"
    z = float(z)
    if not math.isfinite(z):
        return "undefined"
    az = abs(z)
    if az <= K_PASS:
        return "pass"
    if az <= K_TENSION:
        return "tension"
    return "beyond_tension"


def acceptance_sanity_band(pk: dict, lo: int, hi: int, accepted_counts: dict,
                           n_accepted: int) -> dict:
    """The pre-registered per-slice acceptance sanity: the accepted
    wound-count distribution inside the slice against the bin-by-bin 95%
    binomial band of the sliced Glauber analytic distribution
    (p_k = P(K=k)/mass on [lo, hi]); qualifying bins with expected count
    N p_k >= 5. The decision passes exactly when every qualifying bin is
    inside its band."""
    mass = sum(pk[k] for k in range(lo, hi + 1))
    entries = []
    all_inside = True
    n_qualifying = 0
    for k in range(lo, hi + 1):
        p_k = pk[k] / mass
        expected = n_accepted * p_k
        observed = int(accepted_counts.get(k, 0))
        half = WILSON_Z * math.sqrt(max(n_accepted * p_k * (1.0 - p_k), 0.0))
        low = expected - half
        high = expected + half
        qualifies = bool(expected >= ACCEPTANCE_BAND_MIN_EXPECTED)
        inside = bool(low <= observed <= high)
        if qualifies:
            n_qualifying += 1
            all_inside = all_inside and inside
        entries.append({
            "k": int(k),
            "p_analytic": float(p_k),
            "expected": float(expected),
            "observed": int(observed),
            "band_low": float(low),
            "band_high": float(high),
            "qualifies": qualifies,
            "inside": inside,
        })
    return {
        "lo": int(lo), "hi": int(hi),
        "slice_mass": float(mass),
        "n_accepted": int(n_accepted),
        "band_convention": (
            "bin-by-bin 95% binomial band of the sliced Glauber analytic "
            "distribution: expected N p_k with the band N p_k +- "
            "1.959964 sqrt(N p_k (1 - p_k)) on the count scale; qualifying "
            "bins with expected count at least 5 (the M5 audit statement "
            "adapted to slices, the pre-registered rule)"),
        "bins": entries,
        "n_qualifying_bins": int(n_qualifying),
        "n_violating_bins": int(sum(
            1 for e in entries if e["qualifies"] and not e["inside"])),
        "holds": bool(all_inside),
    }


# --- Pre-registered probe schedule (card requirement 1) ---------------------------


def _probe_provenance(config_hash: str, probe: str) -> dict:
    return {
        "probe": probe,
        "configuration_hash": config_hash,
        "config_seed": CONFIG_SEED,
        "m13_seed_base": M13_SEED_BASE,
        "milestone": MILESTONE,
        "phase": PHASE_P2,
        "run_description": RUN_DESCRIPTION,
        "helper_sha256": _sha256_file(Path(__file__)),
        "written_utc": _utc_now_iso(),
    }


# The P1 min-bias (b, K) draw-stream literals (dev/mixing2006/tests/
# test_pa_builder_wound_slices.py, the Pb208 entries): the pilot re-asserts
# the pin on the builder path. The draws depend only on the Glauber
# sampler's RNG stream, so the pins run against the light audit builder
# exactly as the P1 tests do.
MINBIAS_DRAW_STREAM_PIN = {
    ("Pb208", 20260911): [
        (6.0, 2), (1.8, 3), (6.6, 3), (2.75, 11), (7.7, 1), (3.05, 6),
        (6.25, 1), (5.4, 3), (9.05, 1), (6.05, 2), (4.95, 4), (6.0, 5),
    ],
    ("Pb208", 7): [
        (5.85, 5), (2.7, 3), (4.1, 6), (5.3, 2), (6.3, 5), (5.8, 3),
        (2.4, 8), (5.25, 5), (6.8, 1), (6.05, 2), (3.4, 6), (1.85, 5),
    ],
}


def _light_pb_builder(seed: int, wound_range=None):
    """The light audit builder (the P1 test configuration: the pinned audit
    Regge parameters and the default ContinuumParameters), used only by the
    pilot draw-stream pins whose literals were captured on this exact
    configuration."""
    from na49_mixing.continuum_channels import ContinuumParameters
    from na49_mixing.event_builder_pA import PAEventBuilder, PAExcitationConfig
    from na49_mixing.exchange_model import ReggeExcitationParameters

    light_regge = ReggeExcitationParameters(
        alpha_p0=1.0808, alpha_p_prime=0.25, alpha_exc0=0.5,
        transfer_slope=4.5, edge_power=2.0, s0=1.0, grid_size=96,
        effective_transfer_slope=7.3489827740435505)
    return PAEventBuilder(
        PAExcitationConfig(
            production_mode="regge", regge=light_regge,
            continuum=ContinuumParameters(),
            nucleus=_pb208_spec_lite(),
            glauber_dr_fm=0.05, max_event_attempts=100,
            target_wound_range=wound_range),
        np.random.default_rng(seed))


def _pb208_spec_lite():
    from na49_mixing.glauber import NucleusSpec

    return NucleusSpec(**GEOMETRY_PB208)


def _stub_attempt_fn(event_id, impact_parameter, wounds):
    """A stub kinematic attempt for the pilot draw-stream pins (the P1
    test pattern): generate() then only consumes the Glauber draw, so the
    draw stream is isolated from the shared-RNG consumption of real
    attempts."""
    import types

    return types.SimpleNamespace(metadata={})


def _sliced_statistic_vector(events, prereg: dict) -> list:
    """Deterministic statistic vector of the sliced determinism probe: the
    per-charge pooled kernel yields over the representative mid set 3
    registered cells (read from the pre-registration inventory), the mean
    target wounds, the final multiplicity, and the final-pion count."""
    cells_by_charge = {}
    for charge in ("pi-", "pi+"):
        rec = prereg["data_inventory"][f"set3_{charge}"]["registered"]
        cells_by_charge[charge] = [(float(c[0]), float(c[1]))
                                   for c in rec["cells"]]
    pdgs = {"pi-": -211, "pi+": 211}
    values: list = []
    for charge in ("pi-", "pi+"):
        arrays = species_arrays(events, pdgs[charge])
        counts, _assign = bin_first_match(arrays["xf"], arrays["pt"],
                                          cells_by_charge[charge])
        values.append(pooled_kernel_yield_model(
            counts, cells_by_charge[charge], len(events)))
    values.append(float(np.mean([
        float(event.metadata["target_wounds"]) for event in events])))
    values.append(float(sum(len(event.final_particles)
                            for event in events) / max(len(events), 1)))
    values.append(float(sum(
        1 for event in events for particle in event.final_particles
        if int(particle.pdg_id) in PION_PDGS)))
    return values


def _minbias_statistic_vector(events) -> list:
    """Deterministic statistic vector of the min-bias determinism probe:
    the per-event invariant density at the min-bias pi- xF = 0 column (the
    M12 pa_per_event_density convention on the min-bias grid), the mean
    target wounds, and the multiplicity."""
    table = read_points(pa_data_dir() / MINBIAS_CONTEXT_DENSITY_NAMES["pi-"])
    cells = sorted({(round(float(r[0]), 4), round(float(r[1]), 4))
                    for r in table if abs(float(r[0])) < 1e-6})
    arrays = species_arrays(events, -211)
    counts, _overflow = bin_first_match(arrays["xf"], arrays["pt"], cells)
    values = []
    n_events = max(len(events), 1)
    for (xf, pt), n in zip(cells, counts):
        dxf, dpt = cell_full_widths(xf, pt)
        energy = pion_energy(xf, pt)
        values.append(float(
            (energy / (math.pi * SQRT_S_GEV * pt))
            * (float(n) / (n_events * dxf * dpt))))
    values.append(float(np.mean([
        float(event.metadata["target_wounds"]) for event in events]))
        if events else float("nan"))
    values.append(float(sum(len(event.final_particles)
                            for event in events) / n_events))
    return values


def _pp_statistic_vector(events) -> list:
    """Deterministic statistic vector of the p+p reference determinism
    probe: the multiplicity, the per-charge pion counts, and the mean pion
    pT (fixed cells of the set-1 pi- registered grid)."""
    values = [
        float(sum(len(event.final_particles) for event in events)
              / max(len(events), 1)),
        float(sum(1 for event in events for particle in event.final_particles
                  if int(particle.pdg_id) == -211)),
        float(sum(1 for event in events for particle in event.final_particles
                  if int(particle.pdg_id) == 211)),
    ]
    pts = [math.hypot(float(particle.p4.px), float(particle.p4.py))
           for event in events for particle in event.final_particles
           if int(particle.pdg_id) in PION_PDGS]
    values.append(float(np.mean(pts)) if pts else float("nan"))
    return values


def run_probe_determinism(out_dir: Path, config_path: Path, prereg: dict,
                          out_name: str = "probe_determinism_p2.json") -> dict:
    """Pre-registered determinism probe (the M10 probe_A pattern): five
    bit-identical repeated evaluations at a fixed point.

    Units: (i) the sliced builder path (300 sliced p+Pb events at the mid
    set 3 band [4, 7], seed PROBE_SEED_SLICED_DETERMINISM); (ii) the
    minimum-bias builder path (300 events, seed
    PROBE_SEED_MINBIAS_DETERMINISM; the derived pilot sub-stream, derivation
    recorded); (iii) the p+p reference generator (300 events, seed
    PROBE_SEED_PP_DETERMINISM, the pre-registered "determinism probe p+p"
    sub-stream).
    """
    out_dir = Path(out_dir)
    cross_check_preregistration(prereg)
    config_hash = canonical_configuration_hash(config_path)
    band3 = prereg["mapping_table"]["sets"]["3"]["band"]
    if list(band3) != list(MODEL_SUPPORT_PROBE_BAND):
        raise SystemExit(
            f"the mid set 3 band {band3} does not match the pinned probe "
            f"band {list(MODEL_SUPPORT_PROBE_BAND)}")

    def five_repeats(generator, seed, vector_fn, n_events):
        vectors, walls, residuals = [], [], []
        for _rep in range(DETERMINISM_REPEATS):
            events, _summary, wall, _applied, _beam = generator(
                int(n_events), int(seed), config_path)
            vectors.append(json.dumps(_to_jsonable(vector_fn(events))))
            walls.append(float(wall))
            residuals.extend(float(event.conservation_residuals()[
                "relative_max"]) for event in events)
        return {
            "n_events": int(n_events),
            "seed": int(seed),
            "wall_s": walls,
            "bit_identical": len(set(vectors)) == 1,
            "max_conservation_residual": float(max(residuals)),
        }

    units = {
        "sliced_builder": five_repeats(
            lambda n, s, c: generate_slice_events(n, s, c, band3),
            PROBE_SEED_SLICED_DETERMINISM,
            lambda evs: _sliced_statistic_vector(evs, prereg),
            COST_CELL_EVENTS),
        "min_bias_builder": five_repeats(
            generate_minbias_events, PROBE_SEED_MINBIAS_DETERMINISM,
            _minbias_statistic_vector, COST_CELL_EVENTS),
        "pp_reference_generator": five_repeats(
            generate_pp_events, PROBE_SEED_PP_DETERMINISM,
            _pp_statistic_vector, COST_CELL_EVENTS),
    }

    payload = {
        "schema": PROBE_DETERMINISM_SCHEMA,
        "probe": "determinism",
        "pattern": "M10 probe_A, five bit-identical repeated evaluations",
        "provenance": _probe_provenance(config_hash, "determinism"),
        "units": units,
        "seed_derivation_note": (
            "PROBE_SEED_SLICED_DETERMINISM 20349832 and "
            "PROBE_SEED_PP_DETERMINISM 20349831 are the pre-registered "
            "pilot sub-streams; PROBE_SEED_MINBIAS_DETERMINISM 20349844 "
            "derives from the pilot anchor by +1 steps beyond the "
            "model-support probe top 20349843 (pilot anchor + 13), still "
            "strictly below the slice-1 anchor 20350831 (the seed protocol "
            "rule: additional sub-streams derive from the anchors with the "
            "derivation recorded)"),
        "deterministic": all(u["bit_identical"] for u in units.values()),
        "written_utc": _utc_now_iso(),
    }
    _write_json(out_dir / out_name, payload)
    print(f"[probe determinism] deterministic={payload['deterministic']}",
          flush=True)
    return payload


def run_probe_cost_cells(out_dir: Path, config_path: Path,
                         prereg: dict) -> dict:
    """Pre-registered per-slice cost cells (the M10 probe_C pattern): 300
    events per slice, sets 1..5 in order, one single sample per cell at the
    fixed frozen M8 parameters, plus the pre-registered statistics-menu
    decision per slice."""
    out_dir = Path(out_dir)
    cross_check_preregistration(prereg)
    config_hash = canonical_configuration_hash(config_path)
    mapping = prereg["mapping_table"]["sets"]
    cells = {}
    for set_no in (1, 2, 3, 4, 5):
        seed = [PROBE_SEED_COST_SET1, PROBE_SEED_COST_SET2,
                PROBE_SEED_COST_SET3, PROBE_SEED_COST_SET4,
                PROBE_SEED_COST_SET5][set_no - 1]
        band = mapping[str(set_no)]["band"]
        events, summary, wall, _applied, _beam = generate_slice_events(
            COST_CELL_EVENTS, seed, config_path, band)
        cells[f"set{set_no}"] = {
            "n_events": int(len(events)),
            "seed": int(seed),
            "band": [int(band[0]), int(band[1])],
            "wall_s": float(wall),
            "ms_per_event": float(1000.0 * wall / max(len(events), 1)),
            "mean_target_wounds": float(summary["mean_target_wounds"]),
            "closed_form_k_band_mean": float(
                mapping[str(set_no)]["k_band_mean"]),
            "mean_final_multiplicity": float(summary["mean_final_multiplicity"]),
            "max_conservation_residual": float(
                summary["max_conservation_residual"]),
        }
    menu_decision = {
        name: menu_entry_for(cell["ms_per_event"])
        for name, cell in cells.items()
    }
    payload = {
        "schema": PROBE_COST_SCHEMA,
        "probe": "cost_cells",
        "provenance": _probe_provenance(config_hash, "cost_cells"),
        "n_events_per_cell": int(COST_CELL_EVENTS),
        "cells": cells,
        "menu_decision": menu_decision,
        "menu_rule": (
            "the largest menu entry {10000, 5000, 2500, 1000} whose "
            "projected production wall (measured ms/event x K=8 replicas x "
            "E events) stays under the pre-registered 18000 s per-slice "
            "budget; the declared reduction, if used, carries the "
            "MC-sigma caveat (the replica-mean MC sigma grows as "
            "sqrt(10000/E) relative to the primary entry)"),
        "written_utc": _utc_now_iso(),
    }
    _write_json(out_dir / "probe_cost_cells_p2.json", payload)
    for name, cell in cells.items():
        print(f"[cost {name}] {cell['ms_per_event']:.2f} ms/event "
              f"(band {cell['band']})", flush=True)
    return payload


def menu_entry_for(ms_per_event: float) -> dict:
    """The largest pre-registered menu entry whose projected production wall
    (measured ms/event x K = 8 x E) stays under the per-slice budget."""
    chosen = None
    for e1 in STATISTICS_MENU_EVENTS_PER_REPLICA:
        projected = ms_per_event * K_REPLICAS * e1 / 1000.0
        if projected < MAX_PROJECTED_WALL_S:
            chosen = {
                "chosen_events_per_replica": int(e1),
                "projected_wall_s": float(projected),
                "declared_reduction": bool(e1 != PRIMARY_EVENTS_PER_REPLICA),
            }
            break  # the menu is ascending; the first fitting entry is the largest
    if chosen is None:
        chosen = {
            "chosen_events_per_replica": None,
            "projected_wall_s": None,
            "status": "no_menu_entry_fits_budget",
        }
    return chosen


def run_probe_acceptance_sanity(out_dir: Path, config_path: Path,
                                prereg: dict) -> dict:
    """Pre-registered per-slice acceptance sanity at pilot scale: 10000
    accepted events per slice (sets 1..5 in order), the accepted
    wound-count distribution inside the slice against the bin-by-bin 95%
    binomial band of the sliced Glauber analytic distribution, plus the
    conservation-level record of the pilot sample (excluded from
    production statistics)."""
    out_dir = Path(out_dir)
    cross_check_preregistration(prereg)
    config_hash = canonical_configuration_hash(config_path)
    mapping = prereg["mapping_table"]["sets"]
    pk = {int(k): float(v)
          for k, v in prereg["mapping_table"]["pk_bins_1_to_kmax"].items()}
    if not pk:
        pk = {int(k): float(v) for k, v in
              build_mapping_table()["pk_bins_1_to_kmax"].items()}
    slices = {}
    all_ok = True
    for set_no, seed in ((1, PROBE_SEED_ACCEPT_SET1), (2, PROBE_SEED_ACCEPT_SET2),
                         (3, PROBE_SEED_ACCEPT_SET3), (4, PROBE_SEED_ACCEPT_SET4),
                         (5, PROBE_SEED_ACCEPT_SET5)):
        band = mapping[str(set_no)]["band"]
        events, summary, wall, _applied, _beam = generate_slice_events(
            ACCEPTANCE_SANITY_EVENTS, seed, config_path, band)
        counts: dict[int, int] = {}
        for event in events:
            k = int(event.metadata["target_wounds"])
            counts[k] = counts.get(k, 0) + 1
        sanity = acceptance_sanity_band(pk, int(band[0]), int(band[1]),
                                        counts, len(events))
        slices[f"set{set_no}"] = {
            "seed": int(seed),
            "band": [int(band[0]), int(band[1])],
            "n_events": int(len(events)),
            "wall_s": float(wall),
            "ms_per_event": float(1000.0 * wall / max(len(events), 1)),
            "mean_target_wounds": float(summary["mean_target_wounds"]),
            "closed_form_k_band_mean": float(
                mapping[str(set_no)]["k_band_mean"]),
            "closed_form_k_band_variance": float(
                mapping[str(set_no)]["k_band_variance"]),
            "max_conservation_residual": float(
                summary["max_conservation_residual"]),
            "acceptance_sanity": sanity,
        }
        all_ok = all_ok and sanity["holds"]
        print(f"[acceptance set{set_no}] n={len(events)} "
              f"holds={sanity['holds']} "
              f"violating={sanity['n_violating_bins']}/"
              f"{sanity['n_qualifying_bins']}", flush=True)
    payload = {
        "schema": PROBE_ACCEPT_SCHEMA,
        "probe": "acceptance_sanity_pilot",
        "provenance": _probe_provenance(config_hash, "acceptance_sanity_pilot"),
        "n_events_per_cell": int(ACCEPTANCE_SANITY_EVENTS),
        "slices": slices,
        "all_slices_hold": bool(all_ok),
        "written_utc": _utc_now_iso(),
    }
    _write_json(out_dir / "probe_acceptance_sanity_p2.json", payload)
    print(f"[probe acceptance] all_slices_hold={all_ok}", flush=True)
    return payload


def run_probe_gateway_capacity(out_dir: Path, config_path: Path) -> dict:
    """Pre-registered gateway capacity probe, read-only: the per-worker task
    capacity of the grid, queried through the gateway client CLI (the M11/M12
    command path); on any failure the recorded serial fallback statement is
    written instead."""
    out_dir = Path(out_dir)
    config_hash = canonical_configuration_hash(config_path)
    payload: dict = {
        "schema": PROBE_GATEWAY_SCHEMA,
        "probe": "gateway_capacity",
        "mode": "read_only",
        "provenance": _probe_provenance(config_hash, "gateway_capacity"),
    }
    cmd = [sys.executable, "-m", "helpers.na49_gateway.cluster_client",
           "capacity", "--max-tasks-per-worker", "64"]
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=180, cwd=str(REPO_ROOT), env=env)
        payload["command"] = ("python -m helpers.na49_gateway.cluster_client "
                              "capacity")
        if proc.returncode != 0:
            raise RuntimeError(
                f"capacity query exited {proc.returncode}: "
                f"{proc.stderr.strip()[:200]}")
        workers = json.loads(proc.stdout).get("workers", [])
        payload["status"] = "ok"
        payload["workers"] = workers
        payload["free_tasks_total"] = sum(int(w.get("free_tasks", 0))
                                          for w in workers)
    except Exception as exc:  # recorded serial fallback (the M11 pattern)
        payload["status"] = "unavailable"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["serial_fallback_statement"] = (
            "the gateway grid is unavailable from this session; the P2 "
            "production points run serially with the recorded per-point "
            "wall budgets (the recorded serial fallback of the M11 pattern)")
    _write_json(out_dir / "probe_gateway_capacity_p2.json", payload)
    print(f"[probe gateway] status={payload.get('status')} "
          f"workers={len(payload.get('workers', []))}", flush=True)
    return payload


def run_pilot(out_dir: Path, config_path: Path, prereg: dict) -> dict:
    """The P2-start pilot: the P1 regression pins re-asserted at pilot
    scale (the min-bias draw-stream literals, the sliced draw stream against
    ``sample_event_in_range``, the rare-slice and point-target builder
    events, the builder/worker slice-mean equivalence), the pilot-scale
    determinism, the per-slice cost cells, the per-slice acceptance sanity,
    the model-support probe reproduction, and the gateway capacity probe.
    Production is refused (SystemExit) on any missed expectation without a
    recorded deviation."""
    out_dir = Path(out_dir)
    cross_check_preregistration(prereg)
    config_hash = canonical_configuration_hash(config_path)
    if str(MIXING2006_DIR) not in sys.path:
        sys.path.insert(0, str(MIXING2006_DIR))
    from na49_mixing.glauber import NucleusSpec, WoundedNucleonSampler
    from na49_mixing.parallel import GenerationConfig, generate_tracks_chunked

    # Pin 1: the min-bias (b, K) builder stream against the P1 frozen
    # literals (the foundations section 5 pin on the builder path). The
    # literal streams were captured with the light audit Regge parameters
    # and ContinuumParameters(); the (b, K) draw stream depends only on the
    # Glauber sampler's RNG stream, which the builder seeds identically, so
    # the pins run against the light builder exactly as the P1 tests do.
    minbias_literal_pins = []
    for seed in (20260911, 7):
        light_builder = _light_pb_builder(seed)
        pairs = MINBIAS_DRAW_STREAM_PIN[("Pb208", seed)]
        for event_id in range(len(pairs)):
            light_builder.generate(event_id)
        draws = [(round(float(b), 10), int(w))
                 for b, w, _ok in light_builder.glauber_draws]
        expected = [(float(b), int(w)) for b, w in pairs]
        minbias_literal_pins.append({
            "seed": int(seed),
            "n_draws": len(expected),
            "matches": bool(draws == expected),
        })
    minbias_pin = {
        "pins": minbias_literal_pins,
        "holds": all(p["matches"] for p in minbias_literal_pins),
    }

    # Pin 2: the sliced builder draw stream equals sample_event_in_range.
    # The draw-stream pin stubs the kinematic attempt (the P1 test
    # pattern): a real attempt consumes the shared builder RNG, so only a
    # stubbed attempt isolates the two-uniform-per-draw Glauber stream.
    spec = _pb208_spec_lite()
    wound_range = (4, 7)
    sliced_builder = _light_pb_builder(20260911, wound_range)
    sliced_builder._attempt = _stub_attempt_fn
    n_draws = 25
    for event_id in range(n_draws):
        sliced_builder.generate(event_id)
    draws = [(float(b), int(w)) for b, w, _ok in sliced_builder.glauber_draws]
    reference = WoundedNucleonSampler(spec, np.random.default_rng(20260911),
                                      dr_fm=0.05)
    expected = [reference.sample_event_in_range(*wound_range)
                for _ in range(n_draws)]
    sliced_pin = {
        "band": list(wound_range),
        "n_draws": n_draws,
        "matches_sample_event_in_range": bool(draws == expected),
        "holds": bool(draws == expected),
    }

    # Pin 3: the rare-slice (K = 21) builder event.
    rare_builder = _light_pb_builder(424242, (21, 21))
    rare_event = rare_builder.generate(0)
    rare_event_pin = {
        "target_wounds": int(rare_event.metadata["target_wounds"]),
        "max_residual": float(
            rare_event.conservation_residuals()["relative_max"]),
        "holds": bool(int(rare_event.metadata["target_wounds"]) == 21
                      and float(rare_event.conservation_residuals()[
                          "relative_max"]) <= 1e-10),
    }

    # Pin 4: the builder/worker slice-mean equivalence (the M12 P3
    # requirement at the pinned test-scale sample; the chunked-worker
    # reference needs the bundled .gen card).
    delta_gen = str(MIXING2006_DIR / "na49_mixing" / "generators" / "michal"
                    / "deltas-to-pion" / "delta++.gen")
    eq_range = (2, 4)
    eq_n = 100
    eq_builder = _light_pb_builder(20771, eq_range)
    ks_builder = [int(eq_builder.generate(i).metadata["target_wounds"])
                  for i in range(eq_n)]
    out = generate_tracks_chunked(
        GenerationConfig(
            gen_file=delta_gen, n_events=eq_n, seed=20771,
            beam_momentum=158.0, max_decay_passes=1,
            fragmentation_mode="both", nucleus=spec,
            target_wound_range=eq_range),
        n_workers=1, n_chunks=2)
    wounds = np.asarray(out["n_wounds"], dtype=np.int64)
    sides = np.asarray(out["source_side"], dtype=np.int64)
    per_event_k = wounds[sides == 1]
    analytic = WoundedNucleonSampler(spec, np.random.default_rng(0),
                                     dr_fm=0.05)
    pk_eq = {k: float(analytic.fraction_for_wound_range(k, k))
             for k in range(1, 15)}
    lo, hi = eq_range
    mass_eq = sum(pk_eq[k] for k in range(lo, hi + 1))
    mean_closed = sum(k * pk_eq[k] for k in range(lo, hi + 1)) / mass_eq
    var_eq = (sum(k * k * pk_eq[k] for k in range(lo, hi + 1)) / mass_eq
              - mean_closed**2)
    se_eq = math.sqrt(var_eq / eq_n)
    mean_builder = float(np.mean(ks_builder))
    mean_worker = float(np.mean(per_event_k))
    equivalence_pin = {
        "band": list(eq_range),
        "n_events": int(eq_n),
        "mean_builder": mean_builder,
        "mean_worker": mean_worker,
        "closed_form_mean": float(mean_closed),
        "holds": bool(
            abs(mean_builder - mean_closed) <= 4.0 * se_eq
            and abs(mean_worker - mean_closed) <= 4.0 * se_eq
            and abs(mean_builder - mean_worker)
            <= 4.0 * se_eq * math.sqrt(2)),
    }

    # Pin 5: the model-support probe record reproduction (the P1 pinned
    # probe sample through the sliced builder path at the frozen M8 set).
    support = run_model_support_probe(config_path)
    support_pin = {
        "band": support["band"],
        "n_events": support["n_events"],
        "n_final_pions": support["n_final_pions"],
        "matches_pinned_record": bool(
            support["band"] == list(MODEL_SUPPORT_PROBE_BAND)
            and support["n_final_pions"]
            == prereg["model_support_probe"]["n_final_pions"]
            and support["mean_target_wounds"]
            == prereg["model_support_probe"]["mean_target_wounds"]),
    }

    # Probes: determinism, cost cells, acceptance sanity, gateway capacity.
    determinism = run_probe_determinism(out_dir, config_path, prereg)
    cost = run_probe_cost_cells(out_dir, config_path, prereg)
    acceptance = run_probe_acceptance_sanity(out_dir, config_path, prereg)
    gateway = run_probe_gateway_capacity(out_dir, config_path)

    menu_ok = all(
        decision.get("chosen_events_per_replica") is not None
        for decision in cost["menu_decision"].values())
    # The recorded-deviation coverage: a missed pilot expectation whose
    # slice carries a recorded deviation (P2_DEVIATIONS, fixed before the
    # re-run) is treated as recorded; any uncovered failure refuses
    # production (the card's refusal rule).
    acceptance_failing = [name for name, rec in acceptance["slices"].items()
                          if not rec["acceptance_sanity"]["holds"]]
    covered = set(PILOT_DEVIATION_COVERAGE.get("acceptance_sanity", {}))
    uncovered_failures = [name for name in acceptance_failing
                          if name not in covered]
    recorded_deviations_applied = [
        dev["id"] for dev in P2_DEVIATIONS
        if dev.get("coverage", {}).get("probe") == "acceptance_sanity_pilot"
        and dev.get("coverage", {}).get("set") in acceptance_failing]
    all_holds = bool(
        minbias_pin["holds"] and sliced_pin["holds"]
        and rare_event_pin["holds"] and equivalence_pin["holds"]
        and support_pin["matches_pinned_record"]
        and determinism["deterministic"]
        and not uncovered_failures
        and menu_ok)
    record = {
        "schema": PILOT_SCHEMA,
        "probe": "pilot_pins",
        "provenance": _probe_provenance(config_hash, "pilot_pins"),
        "minbias_stream_pin": minbias_pin,
        "sliced_stream_pin": sliced_pin,
        "rare_slice_event_pin": rare_event_pin,
        "builder_worker_equivalence_pin": equivalence_pin,
        "model_support_probe_reproduction": support_pin,
        "determinism_re_assertion": {
            "artifact": "probe_determinism_p2.json",
            "deterministic": determinism["deterministic"],
        },
        "cost_cells": {
            name: {"ms_per_event": cell["ms_per_event"],
                   "menu": cost["menu_decision"][name]}
            for name, cell in cost["cells"].items()},
        "acceptance_sanity_pilot": {
            name: rec["acceptance_sanity"]["holds"]
            for name, rec in acceptance["slices"].items()},
        "acceptance_sanity_failures": acceptance_failing,
        "uncovered_failures": uncovered_failures,
        "recorded_deviations_applied": recorded_deviations_applied,
        "gateway_capacity": {
            "status": gateway.get("status"),
            "free_tasks_total": gateway.get("free_tasks_total"),
            "serial_fallback_statement":
                gateway.get("serial_fallback_statement"),
        },
        "all_pins_hold": all_holds,
        "refusal_rule": (
            "production is refused on a missed expectation without a "
            "recorded deviation (the card rule); the pilot precedes the "
            "cells manifest and the production launch"),
        "written_utc": _utc_now_iso(),
    }
    _write_json(out_dir / PILOT_RECORD_NAME, record)
    print(f"[pilot] all_pins_hold={all_holds}", flush=True)
    if not all_holds:
        raise SystemExit(
            "pilot pin expectation missed; production refused without a "
            "recorded deviation")
    return record


# --- The cells manifest (pure table arithmetic; built before production) ----------


def _grid_matches_pinned(nodes, half_width_fn) -> bool:
    """The standard-grid coincidence check (the M12 convention): on a
    0.05/0.10-spaced grid the midpoint edges equal the pinned half-width
    windows; a grid that does not tile under the pinned half-widths is
    recorded per set/charge in the manifest."""
    vals = sorted({round(float(v), 6) for v in np.asarray(nodes, dtype=float)})
    n = len(vals)
    for i, v in enumerate(vals):
        lo_step = ((vals[i] - vals[i - 1]) / 2.0 if i > 0
                   else (vals[1] - vals[0]) / 2.0 if n > 1 else 0.025)
        hi_step = ((vals[i + 1] - vals[i]) / 2.0 if i < n - 1
                   else (vals[i] - vals[i - 1]) / 2.0 if n > 1 else 0.025)
        if not (math.isclose(v - lo_step, v - half_width_fn(v), abs_tol=1e-9)
                and math.isclose(v + hi_step, v + half_width_fn(v),
                                 abs_tol=1e-9)):
            return False
    return True


def build_cells_manifest(config_path: Path, prereg_path: Path,
                         out_dir: Path,
                         menu_decision: dict | None = None) -> dict:
    """The P2 cells manifest: the pre-registered per-set/charge registered
    cell sets (from the pre-registration's inventory, recomputed and
    refused on drift), the node plans for the dn/dxF and R_i(xF) model
    integrations, the closed-form mapping table, the seed manifest, and the
    statistics-menu outcome. Consumed byte-identically by the per-replica
    runner (local or gateway)."""
    out_dir = Path(out_dir)
    prereg = json.loads(Path(prereg_path).read_text(encoding="utf-8"))
    cross_check_preregistration(prereg)
    m8 = cross_check_m8_record()
    data = cross_check_data_files(prereg)
    if data["problems"]:
        raise SystemExit(
            "data-file cross-check failed; refusing to build the manifest:\n  "
            + "\n  ".join(data["problems"]))
    seed_block = verify_no_collision()
    config_hash = canonical_configuration_hash(config_path)
    if config_hash != prereg.get("provenance", {}).get(
            "configuration_bundled_canonical_sha256"):
        raise SystemExit(
            "bundled config hash mismatch against the pre-registration "
            "provenance; refusing to build the cells manifest")

    cell_sets = {}
    for key in sorted(prereg["data_inventory"]):
        entry = prereg["data_inventory"][key]
        table = read_points(pa_data_dir() / entry["file"])
        registered = entry["registered"]
        cells = []
        node_cells: dict[str, list[int]] = {}
        for idx, (xf, pt) in enumerate(registered["cells"]):
            dxf, dpt = cell_full_widths(float(xf), float(pt))
            cells.append({
                "xF": float(xf), "pT": float(pt),
                "dxf": float(dxf), "dpt": float(dpt),
                "data_value": float(registered["values"][idx]),
                "data_err": float(registered["errors"][idx]),
                "n_rows": int(registered["rows_per_cell"][idx]),
            })
            node_cells.setdefault(f"{float(xf):.4g}", []).append(idx)
        grid = entry["inventory"]
        cell_sets[key] = {
            "set": int(entry["set"]),
            "charge": entry["charge"],
            "quantity": "per-event invariant density d(xF,pT)",
            "width_rule": "pinned_half_widths",
            "grid_coincides_with_pinned_xF": _grid_matches_pinned(
                grid["xF_nodes"], cell_xf_half_width),
            "grid_coincides_with_pinned_pT": _grid_matches_pinned(
                grid["pT_labels"], cell_pt_half_width),
            "n_cells": len(cells),
            "cells": cells,
            "xf_nodes": sorted(float(k) for k in node_cells),
            "node_cells": node_cells,
            "raw_rows": [[float(r[0]), float(r[1]), float(r[2]), float(r[3])]
                         for r in read_points(pa_data_dir() / entry["file"])],
        }

    # The p+p reference node plan: per (set, charge) the model pp reference
    # dn/dxF integrates over the set's own registered cells at the node
    # (the like-for-like acceptance-cancellation rule), with the published
    # p+p column range recorded for the acceptance audit.
    ppref_nodes = {}
    for key in sorted(cell_sets):
        cs = cell_sets[key]
        pp_t = read_ptintegrated(
            spshadrons_data_dir() / PP_REF_PTINTEGRATED_NAMES[cs["charge"]])
        nodes = []
        for xf in cs["xf_nodes"]:
            match = pp_t[np.isclose(pp_t[:, 0], float(xf), atol=1e-6)]
            nodes.append({
                "xF": float(xf),
                "node_key": f"{float(xf):.4g}",
                "cell_indices": cs["node_cells"][f"{float(xf):.4g}"],
                "pp_reference_present": bool(len(match) > 0),
                "pp_published_pT_range": (
                    [float(match[:, 1].min()), float(match[:, 1].max())]
                    if len(match) else None),
            })
        ppref_nodes[key] = nodes

    mapping = prereg["mapping_table"]
    manifest = {
        "schema": SLICE_MANIFEST_SCHEMA,
        "card": CARD, "milestone": MILESTONE, "phase": PHASE_P2,
        "run_description": RUN_DESCRIPTION,
        "provenance": _probe_provenance(config_hash, "cells_manifest"),
        "m8_crosscheck": m8,
        "seed_block_crosscheck": seed_block,
        "binning_statement": (
            "a final pion enters the first registered cell (in the "
            "manifest cell order) whose node lies within the pinned cell "
            "half-widths in both variables (xF bins 0.05 wide at xF <= "
            "0.40; pT bins 0.05 wide below pT 0.3 and 0.10 wide above); "
            "particles matching no registered cell of the cell set are "
            "unplaced and the per-replica overflow counts them (the coarse "
            "pT grids leave inter-cell gaps under the pinned half-widths; "
            "the overflow bookkeeping records the unplaced fraction and "
            "the grid-coincidence flags record where the data's own grid "
            "does not tile under the pinned convention)"),
        "kernel_statement": (
            "the pooled kernel yield Y = sum over the registered cells of "
            "(pT/E) d(xF,pT) dxF dpT with E = sqrt(m_pi^2 + (xF sqrt(s)/2)"
            "^2 + pT^2) on both sides; the model side uses the M12 "
            "pa_per_event_density convention binned on the data's own "
            "grid, which collapses algebraically to "
            "sum(n)/(N pi sqrt(s)) (pinned by a helper-suite test); the "
            "data side uses the inverse-variance merged unique-cell "
            "values with the quadrature error propagation"),
        "r_node_statement": (
            "R_i(xF) = (dn/dxF)_i(xF) / ((1+<N_w>)/2 (dn/dxF)_pp(xF)) with "
            "the foundations convention <N_w> = 1 + <K>; the data side "
            "uses the measured <nu>_i and the per-charge spshadrons p+p "
            "ptintegrated reference; the model side uses the sampler's "
            "closed-form <K>_slice and the model p+p reference pool "
            "integrated over the set's own registered cells at the node "
            "(the same cell set as the numerator, so the pT acceptance "
            "cancels inside the model ratio); per-node z = (R_model - "
            "R_data)/sigma_combined with the mapping residual delta_i "
            "carried as a stated systematic on the superposition factor"),
        "mapping_table": mapping,
        "cell_sets": cell_sets,
        "ppref_node_plan": ppref_nodes,
        "seed_manifest": {
            "slices": {
                str(set_no): {
                    "production_seed": slice_anchor(set_no),
                    "replica_seeds": slice_replica_seeds(set_no),
                    "bootstrap_seed": slice_bootstrap_seed(set_no),
                }
                for set_no in (1, 2, 3, 4, 5)
            },
            "pp_reference": {
                "production_seed": PPREF_ANCHOR,
                "replica_seeds": [PPREF_ANCHOR + 1 + k
                                  for k in range(K_REPLICAS)],
                "bootstrap_seed": PPREF_BOOTSTRAP_SEED,
            },
            "bootstrap_order": (
                "one generator consumed sequentially (default_rng of the "
                "bootstrap seed); per (set asc, charge pi- then pi+) the "
                "200 event-level resamples draw in that fixed order"),
        },
        "menu_decision": menu_decision or {},
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    _write_json(out_dir / SLICE_MANIFEST_NAME, manifest)
    n_cells = sum(cs["n_cells"] for cs in cell_sets.values())
    print(f"[manifest] {n_cells} registered cells, "
          f"{len(cell_sets)} cell sets", flush=True)
    return manifest


# --- Production replica payloads (the per-point gateway entry points) -------------


def _pion_source_role_record(events, pdg_id: int) -> dict:
    """The model-internal target/projectile separation of one charge: the
    source-role decomposition of the final pion yields (the named
    mechanism: one continuing projectile source plus K backward-shifted
    target sources; no data counterpart, a model-internal statement)."""
    n_projectile = 0
    n_target = 0
    xf_projectile = []
    xf_target = []
    for event in events:
        for particle in event.final_particles:
            if int(particle.pdg_id) != int(pdg_id):
                continue
            source_id = int(particle.source_id)
            role = (str(event.sources[source_id].role)
                    if 0 <= source_id < len(event.sources) else "unknown")
            xf = track_xf(event, particle)
            if role == "projectile":
                n_projectile += 1
                xf_projectile.append(xf)
            elif role == "target":
                n_target += 1
                xf_target.append(xf)
    n_total = n_projectile + n_target
    return {
        "n_projectile": int(n_projectile),
        "n_target": int(n_target),
        "fraction_target": (float(n_target / n_total) if n_total else None),
        "mean_xF_projectile": (float(np.mean(xf_projectile))
                               if xf_projectile else None),
        "mean_xF_target": (float(np.mean(xf_target)) if xf_target else None),
    }


def _conservation_record(events, system: str, A_target: int,
                         Z_target: int) -> dict:
    """Spec 7.1 conservation re-assertions per event: the four-momentum and
    quantum-number closures through event.conservation_residuals (tests
    1-4), the source quantum numbers, the no-primary-antibaryon-source rule
    (test 7, the M9 pair-margin criterion), and the p+A accounting test
    (test 8: total event baryon number A+1 and charge Z+1 over the final
    particles and the target spectators)."""
    max_res = 0.0
    source_qn_violations = 0
    accounting_violations = 0
    n_antibaryons = 0
    antibaryon_violations = 0
    min_pair_margin = None
    n_spectators = 0
    for event in events:
        max_res = max(max_res,
                      float(event.conservation_residuals()["relative_max"]))
        for source in event.sources:
            qn = source.quantum_numbers
            ok = int(qn.B) == 1 and int(qn.S) == 0
            if source.role == "target" and system != "ppref":
                target_pdg = int(source.metadata.get("target_pdg", 0))
                ok = ok and int(qn.Q) == (1 if target_pdg == 2212 else 0)
            elif source.role == "projectile":
                ok = ok and qn.as_tuple() == (1, 1, 0)
            if not ok:
                source_qn_violations += 1
        b_sum = sum(int(p.quantum_numbers.B) for p in event.final_particles)
        q_sum = sum(int(p.quantum_numbers.Q) for p in event.final_particles)
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
    return {
        "n_events_validated": int(len(events)),
        "max_residual": float(max_res),
        "source_quantum_number_violations": int(source_qn_violations),
        "accounting_violations": int(accounting_violations),
        "accounting_rule": (
            f"spec 7.1 test 8: total event baryon number {A_target + 1} and "
            f"charge {Z_target + 1} per event, summed over the final "
            "particles and the target spectators"),
        "n_spectators_pooled": int(n_spectators),
        "n_antibaryons": int(n_antibaryons),
        "antibaryon_pair_violations": int(antibaryon_violations),
        "antibaryon_rule": (
            "spec 7.1 test 7: every final antibaryon's parent source mass "
            "sits above the minimal baryon-pair threshold (the M9 pair "
            "margin criterion)"),
        "min_antibaryon_pair_margin_gev": (
            float(min_pair_margin) if min_pair_margin is not None else None),
        "tests_1_to_4_statement": (
            "spec 7.1 tests 1-4 (four-momentum closure, integer B/Q/S, "
            "threshold legality, decay-graph ancestry) are re-asserted per "
            "event through event.validate() at the 1e-10 tolerance inside "
            "the generators; the recorded max_residual is the maximum "
            "relative four-momentum closure residual"),
        "test_5_statement": (
            "spec 7.1 test 5 (reproducibility) is the determinism evidence: "
            "the pilot determinism probe and the production-scale "
            "bit-identity double runs"),
        "test_6_statement": (
            "spec 7.1 test 6 (the p+p projectile-target symmetry) is "
            "established at the M12 P2 production record on the identical "
            "_generate_pp path (unchanged source bytes); the M13 p+p "
            "reference pool consumes the same path and is not re-tested "
            "here"),
    }


def _antibaryon_pair_margin(pdg_id: int) -> float:
    """The minimal baryon-pair content of an antibaryon (the M9 criterion:
    the antibaryon mass plus the proton mass, the minimal pair content)."""
    masses = {int(pdg_id): get_mass_cached(int(pdg_id))}
    partner = masses[int(pdg_id)] + get_mass_cached(2212)
    return float(partner)


def get_mass_cached(pdg_id: int) -> float:
    """The PDG mass lookup (cached per process; the builder's own source)."""
    global _PDG_MASS_CACHE
    if "_PDG_MASS_CACHE" not in globals():
        _PDG_MASS_CACHE = {}
    if pdg_id not in _PDG_MASS_CACHE:
        if str(MIXING2006_DIR) not in sys.path:
            sys.path.insert(0, str(MIXING2006_DIR))
        from na49_mixing.pdg_database import get_mass

        _PDG_MASS_CACHE[pdg_id] = float(get_mass(pdg_id))
    return _PDG_MASS_CACHE[pdg_id]


ANTIBARYON_PAIR_MARGIN_TOL = 1e-9


def build_slice_replica_payload(set_no: int, seed: int, config_path: Path,
                                manifest_path: Path,
                                n_events: int = PRIMARY_EVENTS_PER_REPLICA,
                                m8_record: Path | None = None,
                                preregistration: Path | None = None,
                                events=None) -> dict:
    """One production slice-replica payload (the pure, bit-identical
    function of (set, seed, config, manifest, n_events) the determinism
    double run compares; the volatile wall/timestamp fields are added by
    produce_slice_replica).

    Generation at the frozen M8 set through the builder path with the
    pre-registered ``target_wound_range`` (the mapped band of the set from
    the manifest), event.validate() re-asserted per event at 1e-10, the
    spec 7.1 source-quantum-number, antibaryon-traceability, and p+A
    accounting re-assertions per event, the wound-count bookkeeping, and
    the first-match binning onto the manifest's registered cell sets for
    the set's two charges with the per-event sparse cell triples (the
    event-level bootstrap input) and the target/projectile separation."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    embedded_sha = manifest.pop("manifest_sha256", None)
    if embedded_sha is None or canonical_json_sha256(manifest) != embedded_sha:
        raise SystemExit(
            f"centrality cells manifest embedded sha mismatch: {manifest_path}")
    set_key = f"set{int(set_no)}"
    if not any(k.startswith(set_key + "_") for k in manifest["cell_sets"]):
        raise SystemExit(f"unknown set {set_key!r} for the slice payload")
    set_entry = manifest["mapping_table"]["sets"][str(int(set_no))]
    band = set_entry["band"]
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
        events, _summary, _wall, _applied, _beam = generate_slice_events(
            int(n_events), int(seed), config_path, band)
    if not events:
        raise SystemExit("no events generated")

    nucleus = dict(GEOMETRY_PB208)
    conservation = _conservation_record(events, "pPb208",
                                        int(nucleus["A"]), int(nucleus["Z"]))
    wound_values = [int(event.metadata["target_wounds"]) for event in events]
    wound_counts: dict[str, int] = {}
    for k in wound_values:
        wound_counts[str(k)] = wound_counts.get(str(k), 0) + 1
    wounds_summary = {
        "mean": float(np.mean(wound_values)),
        "std": float(np.std(wound_values, ddof=1)) if len(wound_values) > 1
        else 0.0,
        "min": int(min(wound_values)),
        "max": int(max(wound_values)),
        "n_events": int(len(wound_values)),
        "counts_per_k": wound_counts,
        "band": [int(band[0]), int(band[1])],
    }

    charges = ("pi-", "pi+")
    pdgs = {"pi-": -211, "pi+": 211}
    species_binning: dict = {}
    tp_separation: dict = {}
    cell_sets_of_set = {
        key: cs for key, cs in manifest["cell_sets"].items()
        if cs["set"] == int(set_no)}
    for charge in charges:
        key = f"{set_key}_{charge}"
        cs = cell_sets_of_set[key]
        cells = [(float(c["xF"]), float(c["pT"])) for c in cs["cells"]]
        arrays = species_arrays(events, pdgs[charge])
        counts, assign = bin_first_match(arrays["xf"], arrays["pt"], cells)
        species_binning[charge] = {
            "cell_set_key": key,
            "n_particles": int(len(arrays["xf"])),
            "placed": int(np.count_nonzero(assign >= 0)),
            "overflow": int(np.count_nonzero(assign < 0)),
            "cell_counts": [int(v) for v in counts],
            "per_event_cell_triples": sparse_cell_triples(arrays["ev"],
                                                          assign),
        }
        tp_separation[charge] = _pion_source_role_record(events, pdgs[charge])

    payload = {
        "schema": SLICE_REPLICA_SCHEMA,
        "card": CARD, "milestone": MILESTONE, "phase": PHASE_P2,
        "run_description": RUN_DESCRIPTION,
        "provenance": {
            "configuration_hash": config_hash,
            "config_seed": CONFIG_SEED,
            "m13_seed_base": M13_SEED_BASE,
            "milestone": MILESTONE,
            "phase": PHASE_P2,
            "run_description": RUN_DESCRIPTION,
            "seed": int(seed),
            "set": int(set_no),
            "band": [int(band[0]), int(band[1])],
            "stage_label": STAGE_LABEL_SLICES,
            "helper_sha256": _sha256_file(Path(__file__)),
            "prereg_check_platform_notes": prereg_check_platform_notes(),
        },
        "m8_crosscheck": m8_record_check,
        "cells_manifest_path": str(manifest_path),
        "cells_manifest_sha256": _sha256_file(manifest_path),
        "nucleus": nucleus,
        "n_events": int(len(events)),
        "mean_final_multiplicity": float(
            sum(len(event.final_particles) for event in events)
            / max(len(events), 1)),
        "conservation": conservation,
        "wounds": wounds_summary,
        "generation_bookkeeping": {
            "generation_attempts": int(_last_builder_summary.get(
                "generation_attempts", len(events))),
            "generation_accepted": int(len(events)),
            "generation_rejected": 0,
            "failure_reason_counts": {},
        },
        "species_binning": species_binning,
        "target_projectile_separation": tp_separation,
    }
    return payload


_last_builder_summary: dict = {}


def produce_slice_replica(set_no: int, seed: int, config_path: Path,
                          manifest_path: Path, out_path: Path,
                          n_events: int = PRIMARY_EVENTS_PER_REPLICA,
                          m8_record: Path | None = None,
                          preregistration: Path | None = None) -> dict:
    """Build one production slice-replica payload and write it (the volatile
    fields are added here, so build_slice_replica_payload stays the pure
    determinism surface)."""
    t0 = time.perf_counter()
    payload = build_slice_replica_payload(
        set_no, seed, config_path, manifest_path, n_events=n_events,
        m8_record=m8_record, preregistration=preregistration)
    wall = time.perf_counter() - t0
    payload["wall_s"] = float(wall)
    payload["ms_per_event"] = float(
        1000.0 * wall / max(int(payload["n_events"]), 1))
    payload["provenance"]["written_utc"] = _utc_now_iso()
    _write_json(Path(out_path), payload)
    print(f"[slice replica set{set_no} seed {seed}] {payload['n_events']} "
          f"events, wall {wall:.1f}s, max residual "
          f"{payload['conservation']['max_residual']:.3e}", flush=True)
    return payload


def build_ppref_replica_payload(seed: int, config_path: Path,
                                manifest_path: Path,
                                n_events: int = PP_REF_EVENTS_PER_REPLICA,
                                m8_record: Path | None = None,
                                preregistration: Path | None = None,
                                events=None) -> dict:
    """One p+p reference replica payload (the model pp pool of the R_i(xF)
    diagnostic): the same frozen M8 parameters via _generate_pp, the spec
    7.1 conservation re-assertions (the elementary p+p framing of the
    accounting test), and the first-match binning onto every set's
    registered cell sets (the model pp dn/dxF integrates over the same
    registered cells as the slice numerator, so the pT acceptance cancels
    inside the model R_i)."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    embedded_sha = manifest.pop("manifest_sha256", None)
    if embedded_sha is None or canonical_json_sha256(manifest) != embedded_sha:
        raise SystemExit(
            f"centrality cells manifest embedded sha mismatch: {manifest_path}")
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
        events, _summary, _wall, _applied, _beam = generate_pp_events(
            int(n_events), int(seed), config_path,
            stage_label=STAGE_LABEL_PPREF)
    if not events:
        raise SystemExit("no events generated")

    conservation = _conservation_record(events, "ppref", 1, 1)
    charges = ("pi-", "pi+")
    pdgs = {"pi-": -211, "pi+": 211}
    species_binning: dict = {}
    cell_sets = manifest["cell_sets"]
    placed_any: dict = {}
    for key in sorted(cell_sets):
        cs = cell_sets[key]
        charge = cs["charge"]
        cells = [(float(c["xF"]), float(c["pT"])) for c in cs["cells"]]
        arrays = species_arrays(events, pdgs[charge])
        counts, assign = bin_first_match(arrays["xf"], arrays["pt"], cells)
        placed_any[key] = assign
        species_binning[key] = {
            "set": int(cs["set"]),
            "charge": charge,
            "n_cells": int(len(cells)),
            "cell_counts": [int(v) for v in counts],
        }

    payload = {
        "schema": PPREF_REPLICA_SCHEMA,
        "card": CARD, "milestone": MILESTONE, "phase": PHASE_P2,
        "run_description": RUN_DESCRIPTION,
        "provenance": {
            "configuration_hash": config_hash,
            "config_seed": CONFIG_SEED,
            "m13_seed_base": M13_SEED_BASE,
            "milestone": MILESTONE,
            "phase": PHASE_P2,
            "run_description": RUN_DESCRIPTION,
            "seed": int(seed),
            "system": "ppref",
            "stage_label": STAGE_LABEL_PPREF,
            "helper_sha256": _sha256_file(Path(__file__)),
            "prereg_check_platform_notes": prereg_check_platform_notes(),
        },
        "m8_crosscheck": m8_record_check,
        "cells_manifest_path": str(manifest_path),
        "cells_manifest_sha256": _sha256_file(manifest_path),
        "n_events": int(len(events)),
        "mean_final_multiplicity": float(
            sum(len(event.final_particles) for event in events)
            / max(len(events), 1)),
        "conservation": conservation,
        "species_binning": species_binning,
    }
    return payload


def produce_ppref_replica(seed: int, config_path: Path, manifest_path: Path,
                          out_path: Path,
                          n_events: int = PP_REF_EVENTS_PER_REPLICA,
                          m8_record: Path | None = None,
                          preregistration: Path | None = None) -> dict:
    """Build one p+p reference replica payload and write it."""
    t0 = time.perf_counter()
    payload = build_ppref_replica_payload(
        seed, config_path, manifest_path, n_events=n_events,
        m8_record=m8_record, preregistration=preregistration)
    wall = time.perf_counter() - t0
    payload["wall_s"] = float(wall)
    payload["ms_per_event"] = float(
        1000.0 * wall / max(int(payload["n_events"]), 1))
    payload["provenance"]["written_utc"] = _utc_now_iso()
    _write_json(Path(out_path), payload)
    print(f"[ppref replica seed {seed}] {payload['n_events']} events, "
          f"wall {wall:.1f}s, max residual "
          f"{payload['conservation']['max_residual']:.3e}", flush=True)
    return payload


def run_determinism_production(out_dir: Path, config_path: Path,
                               manifest_path: Path,
                               preregistration: Path | None = None,
                               m8_record: Path | None = None) -> dict:
    """Same-environment double generation at production scale on the M13
    paths: the set-1 replica-0 payload and the ppref replica-0 payload are
    each built twice at their production seeds and the payloads must be
    bit-identical (the spec 7.1 reproducibility test 5 at production scale;
    the M10 probe_A pattern as executed in the M12 P2/P3 production)."""
    out_dir = Path(out_dir)
    config_hash = canonical_configuration_hash(config_path)
    units = {}
    for label, kind, seed in (
            ("slice_set1", "slice", int(slice_replica_seeds(1)[0])),
            ("ppref", "ppref", int(PPREF_ANCHOR + 1))):
        t0 = time.perf_counter()
        if kind == "slice":
            payload_a = build_slice_replica_payload(
                1, seed, config_path, manifest_path,
                m8_record=m8_record, preregistration=preregistration)
            payload_b = build_slice_replica_payload(
                1, seed, config_path, manifest_path,
                m8_record=m8_record, preregistration=preregistration)
        else:
            payload_a = build_ppref_replica_payload(
                seed, config_path, manifest_path,
                m8_record=m8_record, preregistration=preregistration)
            payload_b = build_ppref_replica_payload(
                seed, config_path, manifest_path,
                m8_record=m8_record, preregistration=preregistration)
        wall = time.perf_counter() - t0
        canon_a = json.dumps(_to_jsonable(payload_a), sort_keys=True,
                             separators=(",", ":"))
        canon_b = json.dumps(_to_jsonable(payload_b), sort_keys=True,
                             separators=(",", ":"))
        units[label] = {
            "kind": kind,
            "seed": int(seed),
            "n_events": int(payload_a["n_events"]),
            "bit_identical": bool(canon_a == canon_b),
            "payload_sha256": [hashlib.sha256(canon_a.encode()).hexdigest(),
                               hashlib.sha256(canon_b.encode()).hexdigest()],
            "wall_s": float(wall),
        }
        print(f"[determinism production {label}] "
              f"bit_identical={units[label]['bit_identical']}", flush=True)
    record = {
        "schema": DETERMINISM_PRODUCTION_SCHEMA,
        "probe": "determinism_production",
        "pattern": "M10 probe_A at production scale, same environment",
        "provenance": _probe_provenance(config_hash, "determinism_production"),
        "n_events_per_unit": PRIMARY_EVENTS_PER_REPLICA,
        "units": units,
        "deterministic": all(u["bit_identical"] for u in units.values()),
        "written_utc": _utc_now_iso(),
    }
    _write_json(out_dir / "determinism_production_p2.json", record)
    print(f"[determinism production] "
          f"deterministic={record['deterministic']}", flush=True)
    return record


# --- Collection: the comparison exactly as pre-registered (card sequence 5) -------

REPORT_CLAIM_STATEMENT = (
    "the held-out predictive validation against the p+A centrality classes "
    "is executed at the recorded level through the executed E_BE-to-K "
    "mapping, with the per-set verdicts recorded against the "
    "pre-registered bands; the minimum-bias p+A part stands as M12 "
    "recorded (this card does not re-test it); and the claim-boundary "
    "consequence: the M12 recorded deviations stand, so the spec section "
    "10 excluded-claim text remains in force regardless of the M13 "
    "verdicts")

FALSIFIER4_CROSS_REFERENCE = (
    "spec section 7.3 falsifier 4 (projectile-side stopping) is the M12 P3 "
    "direction; the M13 F-M13-2 wounded-scaling direction is its "
    "centrality-resolved counterpart: an R_i(xF) that drifts away from the "
    "wounded-scaling superposition (1+<N_w>)/2 in a definite direction at "
    "the forward nodes is the same projectile-stopping signal resolved by "
    "impact-parameter class")

GEOMETRY_ANCHOR_SYSTEMATIC_STATEMENT = (
    "the sigma_NN/Woods-Saxon choice stays at the bundled pinned values; "
    "the recorded sigma_pA and <K> deviations against the measured anchors "
    "(sigma_inel(p+Pb) = 1806 +/- 25 mb and <nu> = 3.65) are the stated "
    "systematics on the wounded-scaling interpretation (the superposition "
    "factor (1+<N_w>)/2 and the per-set mapping); a geometry-tuning "
    "campaign is a recorded follow-up, not this card")


def geometry_anchor_crosscheck() -> dict:
    """The geometry-anchor sanity cross-check (sanity, not a gate): sigma_pA
    and <K> from the sampler at the pinned geometry versus the measured
    anchors, with the percent deviations recorded and the anchor-choice
    systematic stated."""
    anchors = geometry_anchors()["Pb208"]
    sigma = float(anchors["sigma_pA_mb"])
    mean_k = float(anchors["mean_target_wounds"])
    sigma_dev = 100.0 * (sigma - SIGMA_INEL_PPb_MB) / SIGMA_INEL_PPb_MB
    k_dev = 100.0 * (mean_k - NU_PPB_MEASURED) / NU_PPB_MEASURED
    return {
        "sampler": anchors,
        "measured_anchors": {
            "sigma_inel_pPb_mb": SIGMA_INEL_PPb_MB,
            "sigma_inel_pPb_err_mb": SIGMA_INEL_PPb_ERR_MB,
            "nu_pPb_measured": NU_PPB_MEASURED,
        },
        "sigma_pA_deviation_percent": float(sigma_dev),
        "sigma_pA_pull": float((sigma - SIGMA_INEL_PPb_MB)
                               / SIGMA_INEL_PPb_ERR_MB),
        "mean_K_deviation_percent": float(k_dev),
        "status": "sanity_not_a_gate",
        "systematic_statement": GEOMETRY_ANCHOR_SYSTEMATIC_STATEMENT,
        "tower_model_context": (
            "the 2026-08-25 cross-check at these pinned values (sigma_pA "
            "1686 mb, -6.7%; <K> 3.93, +7.7%) ran the tower-model lineage "
            "and is cited as geometry context only, never as staged-model "
            "results"),
    }


def _load_manifest_verified(run_dir: Path) -> dict:
    path = Path(run_dir) / SLICE_MANIFEST_NAME
    if not path.exists():
        raise SystemExit(f"cells manifest not found at {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    embedded = manifest.pop("manifest_sha256", None)
    if embedded is None or canonical_json_sha256(manifest) != embedded:
        raise SystemExit(
            f"cells manifest embedded sha mismatch: {path}")
    return manifest


def _load_replicas(run_dir: Path, manifest: dict) -> tuple:
    """Load and validate the 40 slice replicas and the 8 ppref replicas.
    Refuses on a missing artifact, a schema mismatch, a wrong set/band or
    seed, a wrong manifest hash, or a wrong n_events (the menu outcome)."""
    menu = manifest.get("menu_decision") or {}
    slice_arts: dict[tuple[int, int], dict] = {}
    for set_no in (1, 2, 3, 4, 5):
        seeds = slice_replica_seeds(set_no)
        expected_events = menu.get(f"set{set_no}", {}).get(
            "chosen_events_per_replica", PRIMARY_EVENTS_PER_REPLICA)
        for seed in seeds:
            path = (Path(run_dir)
                    / f"slice_replica_set{set_no}_{int(seed)}.json")
            if not path.exists():
                raise SystemExit(f"slice replica missing: {path}")
            art = json.loads(path.read_text(encoding="utf-8"))
            if art.get("schema") != SLICE_REPLICA_SCHEMA:
                raise SystemExit(f"schema mismatch in {path}")
            if int(art["provenance"]["seed"]) != int(seed):
                raise SystemExit(f"seed mismatch in {path}")
            if int(art["provenance"]["set"]) != int(set_no):
                raise SystemExit(f"set mismatch in {path}")
            band = art["provenance"]["band"]
            if list(band) != manifest["mapping_table"]["sets"][
                    str(set_no)]["band"]:
                raise SystemExit(f"band mismatch in {path}")
            if art.get("cells_manifest_sha256") != _sha256_file(
                    Path(run_dir) / SLICE_MANIFEST_NAME):
                raise SystemExit(f"manifest hash mismatch in {path}")
            if int(art["n_events"]) != int(expected_events):
                raise SystemExit(
                    f"n_events mismatch in {path}: {art['n_events']} != "
                    f"the menu outcome {expected_events}")
            slice_arts[(int(set_no), int(seed))] = art
    ppref_arts: dict[int, dict] = {}
    for k in range(K_REPLICAS):
        seed = PPREF_ANCHOR + 1 + k
        path = Path(run_dir) / f"ppref_replica_{int(seed)}.json"
        if not path.exists():
            raise SystemExit(f"ppref replica missing: {path}")
        art = json.loads(path.read_text(encoding="utf-8"))
        if art.get("schema") != PPREF_REPLICA_SCHEMA:
            raise SystemExit(f"schema mismatch in {path}")
        if int(art["provenance"]["seed"]) != int(seed):
            raise SystemExit(f"seed mismatch in {path}")
        if art.get("cells_manifest_sha256") != _sha256_file(
                Path(run_dir) / SLICE_MANIFEST_NAME):
            raise SystemExit(f"manifest hash mismatch in {path}")
        ppref_arts[int(seed)] = art
    return slice_arts, ppref_arts


def _pooled_cells(set_no: int, charge: str, slice_arts: dict,
                  n_cells: int) -> tuple:
    """The pooled and per-replica per-cell counts of one set/charge from
    the loaded replicas."""
    arts = [art for (s, _seed), art in sorted(slice_arts.items())
            if s == set_no]
    per_replica = np.asarray(
        [art["species_binning"][charge]["cell_counts"] for art in arts],
        dtype=np.int64)
    pooled = per_replica.sum(axis=0)
    n_per_replica = [int(art["n_events"]) for art in arts]
    n_pooled = int(sum(n_per_replica))
    return pooled, per_replica, n_pooled, n_per_replica


def _pooled_wounds(set_no: int, slice_arts: dict) -> tuple:
    """The pooled accepted wound counts per k over the slice's replicas."""
    arts = [art for (s, _seed), art in sorted(slice_arts.items())
            if s == set_no]
    counts: dict[int, int] = {}
    total = 0
    k_sum = 0
    for art in arts:
        for k_str, n in art["wounds"]["counts_per_k"].items():
            k = int(k_str)
            counts[k] = counts.get(k, 0) + int(n)
            total += int(n)
            k_sum += int(k) * int(n)
    return counts, total, (k_sum / total if total else float("nan"))


def _bootstrap_slice_cells(dense: np.ndarray, cells: list, nodes: list,
                           rng, n_resamples: int) -> dict:
    """One bootstrap unit: 200 event-level resamples of the pooled
    per-event cell-count matrix; the per-cell MC sigma (the pre-registered
    bootstrap product), the pooled-kernel-yield percentile interval, and
    the per-node dn/dxF percentile intervals. One generator consumed
    sequentially in the recorded order (set asc, charge pi- then pi+)."""
    n_total = int(dense.shape[0])
    xf_arr = np.asarray([float(c[0]) for c in cells])
    pt_arr = np.asarray([float(c[1]) for c in cells])
    dxf_arr = np.asarray([2.0 * cell_xf_half_width(x) for x in xf_arr])
    dpt_arr = np.asarray([2.0 * cell_pt_half_width(p) for p in pt_arr])
    energy_arr = np.asarray([pion_energy(x, p)
                             for x, p in zip(xf_arr, pt_arr)])
    jacobian = energy_arr / (math.pi * SQRT_S_GEV * pt_arr)
    sum_dens = np.zeros(dense.shape[1])
    sum_dens2 = np.zeros(dense.shape[1])
    yield_vals = np.zeros(n_resamples)
    node_keys = [f"{float(n):.4g}" for n in nodes]
    node_masks = [np.isclose(xf_arr, float(n), atol=1e-6) for n in nodes]
    node_sums = {k: np.zeros(n_resamples) for k in node_keys}
    invalid = 0
    for b in range(n_resamples):
        idx = rng.integers(0, n_total, n_total)
        counts_b = dense[idx].sum(axis=0)
        densities_b = jacobian * (counts_b
                                  / (float(n_total) * dxf_arr * dpt_arr))
        sum_dens += densities_b
        sum_dens2 += densities_b * densities_b
        # the pooled-kernel collapse (pinned algebraically): the model
        # yield of the resample is sum(n)/(N pi sqrt(s)) exactly, the same
        # identity the explicit kernel form satisfies
        yield_vals[b] = float(counts_b.sum()
                              / (float(n_total) * math.pi * SQRT_S_GEV))
        for key, mask in zip(node_keys, node_masks):
            # the model node integration collapses to sum(n)/(N dxF)
            node_sums[key][b] = float(
                (counts_b[mask] / (float(n_total) * dxf_arr[mask])).sum()
                if mask.any() else 0.0)
    mean_dens = sum_dens / n_resamples
    var_dens = (sum_dens2 / n_resamples - mean_dens * mean_dens)
    var_dens = np.maximum(var_dens, 0.0) * (n_resamples / (n_resamples - 1))
    cell_sigmas = np.sqrt(var_dens)
    return {
        "cell_bootstrap_sigma": [float(v) for v in cell_sigmas],
        "yield_percentile": [float(np.percentile(yield_vals, 2.5)),
                             float(np.percentile(yield_vals, 97.5))],
        "node_percentile": {
            k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in node_sums.items()},
        "resamples": int(n_resamples),
        "invalid_replicates": int(invalid),
    }


def _trend_slope(values: list, indices: list) -> float:
    """The least-squares linear trend of the pooled kernel yield versus set
    order (the F-M13-1 trend statistic)."""
    x = np.asarray(indices, dtype=float)
    y = np.asarray(values, dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    return slope


def _fm13_1_evaluation(pooled: dict) -> dict:
    """F-M13-1 (centrality-trend direction): the sign of the linear trend
    of the pooled kernel yield versus set order (the data side and the
    model side each, over the mappable sets, at least three) is opposite
    between data and model, with the endpoint-set pooled-yield verdicts
    (set 1 and set 5) beyond the tension band in the direction consistent
    with the opposite trend."""
    charges = {}
    for charge in ("pi-", "pi+"):
        sets = sorted({int(k.split("_")[0][3:]) for k in pooled
                       if k.startswith("set") and k.endswith(f"_{charge}")})
        if len(sets) < 3:
            continue
        data_trend = _trend_slope(
            [pooled[f"set{s}_{charge}"]["yield_data"] for s in sets], sets)
        model_trend = _trend_slope(
            [pooled[f"set{s}_{charge}"]["yield_model"] for s in sets], sets)
        endpoint_records = {}
        direction_consistent = True
        endpoint_beyond = True
        for s in (sets[0], sets[-1]):
            rec = pooled[f"set{s}_{charge}"]
            endpoint_records[str(s)] = rec["yield_verdict"]
            if rec["yield_verdict"] != "beyond_tension":
                endpoint_beyond = False
        # the direction consistency: the endpoint delta signs follow the
        # opposite trend (the model rises where the data falls, or the
        # reverse), evaluated on the endpoint deltas' signs relative to the
        # trend difference
        first, last = sets[0], sets[-1]
        data_delta = (pooled[f"set{last}_{charge}"]["yield_data"]
                      - pooled[f"set{first}_{charge}"]["yield_data"])
        model_delta = (pooled[f"set{last}_{charge}"]["yield_model"]
                       - pooled[f"set{first}_{charge}"]["yield_model"])
        opposite_trend = bool(data_delta * model_delta < 0)
        endpoint_deltas_consistent = bool(
            (data_delta * model_delta < 0) and endpoint_beyond)
        fired = bool(opposite_trend and endpoint_beyond
                     and endpoint_deltas_consistent)
        charges[charge] = {
            "sets_evaluated": sets,
            "data_trend_slope": data_trend,
            "model_trend_slope": model_trend,
            "trend_signs_opposite": opposite_trend,
            "endpoint_verdicts": endpoint_records,
            "endpoints_beyond_tension": bool(endpoint_beyond),
            "endpoint_deltas_consistent_with_opposite_trend":
                endpoint_deltas_consistent,
            "fired": fired,
        }
    fired_any = any(c["fired"] for c in charges.values())
    return {
        "direction": FALSIFIER_DIRECTIONS["F-M13-1_centrality_trend"],
        "per_charge": charges,
        "fired": bool(fired_any),
        "statement": (
            "the falsification-flag requires the opposite trend between "
            "data and model AND both endpoint-set pooled-yield verdicts "
            "beyond the tension band in the direction consistent with the "
            "opposite trend (the pre-registered rule)"),
    }


def _fm13_2_evaluation(r_records: dict, sigma_rule: str) -> dict:
    """F-M13-2 (wounded-scaling direction): the model R_i(xF) exceeds 1 by
    more than k_tension combined 1-sigma at all common nodes xF <= 0.10
    while the data R_i(xF) lies below 1 (or symmetrically the model below
    while the data above) beyond k_tension at the same nodes; the spec
    section 7.3 falsifier 4 cross-reference."""
    units = {}
    fired_any = False
    for key in sorted(r_records):
        nodes = [n for n in r_records[key]
                 if float(n["xF"]) <= 0.10 + 1e-9 and n.get("verdict")
                 not in (None, "no_model_support")]
        if not nodes:
            continue
        model_above = all(
            n["model_R"] - 1.0 > K_TENSION * n["sigma_combined"]
            for n in nodes)
        data_below = all(float(n["data_R"]) < 1.0 for n in nodes)
        model_below = all(
            1.0 - n["model_R"] > K_TENSION * n["sigma_combined"]
            for n in nodes)
        data_above = all(float(n["data_R"]) > 1.0 for n in nodes)
        fired = bool((model_above and data_below)
                     or (model_below and data_above))
        fired_any = fired_any or fired
        units[key] = {
            "nodes_xF_le_010": [
                {"xF": n["xF"], "model_R": n["model_R"],
                 "data_R": n["data_R"],
                 "model_minus_1_over_sigma": (
                     (n["model_R"] - 1.0) / n["sigma_combined"]
                     if n["sigma_combined"] else None)}
                for n in nodes],
            "fired": fired,
        }
    return {
        "direction": FALSIFIER_DIRECTIONS["F-M13-2_wounded_scaling"],
        "per_set_charge": units,
        "fired": bool(fired_any),
        "falsifier_4_cross_reference": FALSIFIER4_CROSS_REFERENCE,
    }


def write_report_md(report: dict, path: Path) -> None:
    """The human-readable REPORT.md: the verdict per observable against the
    pre-registered bands, the mapping table, the falsifier cross-reference,
    the geometry anchor statement, the acceptance sanity results, the wall
    and budget accounting, the determinism result, and the claim
    statement."""
    mapping = report["mapping_table"]["sets"]
    pooled = report["pooled_kernel_yields"]
    lines: list[str] = []
    lines.append("# p+A centrality sets 1-5 at the frozen M8 calibration set — "
                 "2026-09-11")
    lines.append("")
    lines.append(
        "Card TASK-20260911-centrality-sets-1-5, milestone M13, phase P2 "
        "(production and record). The production ran at the pre-registered "
        "statistics through the executed E_BE-to-K mapping: the five "
        "thesis centrality sets (the forward-energy E_BE selection via "
        "grey-proton counting in the NA49 centrality detector) mapped to "
        "wounded-nucleon number K slices of the model's Glauber geometry, "
        "generated at the frozen M8 ten-parameter set with the fixed "
        "Pb208 Woods-Saxon geometry, compared against the Chvala thesis "
        "per-event pion densities. The bands, the seeds, the mapping, the "
        "grids, and the statistics menu were fixed before any event "
        "existed (the pre-registration and its harness cross-check) and "
        "were not tuned after any number was looked at.")
    lines.append("")
    lines.append("## Claim statement")
    lines.append("")
    lines.append(report["claim_statement"])
    lines.append("")
    lines.append("## E_BE-to-K mapping table (closed-form, pre-registered)")
    lines.append("")
    lines.append("| set | n_grey selection | <nu> | chosen band | <K>_band | "
                 "residual abs | residual % | slice fraction (model) | "
                 "thesis event fraction | ambiguous |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for set_no in ("1", "2", "3", "4", "5"):
        rec = mapping[set_no]
        lines.append(
            f"| {set_no} | {rec['n_grey_selection']} | {rec['nu_measured']:.1f} "
            f"| [{rec['band'][0]}, {rec['band'][1]}] | {rec['k_band_mean']:.4f} "
            f"| {rec['residual_absolute']:+.4f} | "
            f"{rec['residual_percent']:+.2f}% | "
            f"{rec['slice_fraction_model']:.4f} | "
            f"{rec['thesis_event_count_fraction']:.4f} | "
            f"{'yes' if rec['band_ambiguity'] else 'no'} |")
    lines.append("")
    lines.append("All five sets carry the band-ambiguity flag under the "
                 "+-10% VENUS systematic on <nu> (the alternative bands "
                 "are recorded in the mapping table). "
                 + report["band_ambiguity_systematic"])
    lines.append("")
    lines.append("## Pooled kernel yield verdicts (primary, banded)")
    lines.append("")
    lines.append("Combined 1-sigma = quadrature of the data error (the "
                 "merged unique-cell errors propagated through the "
                 "kernel-weighted yield) and the MC error (std ddof=1 of "
                 "the 8 per-replica yields). Band rule: pass |z| <= 2, "
                 "tension 2 < |z| <= 5, beyond-tension |z| > 5.")
    lines.append("")
    lines.append("| set/charge | model Y | data Y | MC err | data err | "
                 "z | verdict |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for key in sorted(pooled):
        rec = pooled[key]
        lines.append(
            f"| {key} | {rec['yield_model']:.5f} | {rec['yield_data']:.5f} "
            f"| {rec['yield_mc_err']:.5f} | {rec['yield_data_err']:.5f} | "
            f"{rec['z']:+.2f} | {rec['yield_verdict']} |")
    lines.append("")
    lines.append("Every verdict also carries the 200-resample event-level "
                 "bootstrap percentile interval (centrality_report.json).")
    lines.append("")
    lines.append("## pT-integrated dn/dxF profiles (per-node z-records)")
    lines.append("")
    lines.append("| set/charge | nodes | pass | tension | beyond |")
    lines.append("| --- | --- | --- | --- | --- |")
    for key in sorted(pooled):
        counts = pooled[key]["dn_dxF_count_summary"]
        lines.append(
            f"| {key} | {counts['n_nodes']} | {counts['n_pass']} | "
            f"{counts['n_tension']} | {counts['n_beyond']} |")
    lines.append("")
    lines.append("Per-node z-values (delta over the combined 1-sigma, the "
                 "same node order as the data grid):")
    for key in sorted(pooled):
        zs = ", ".join(
            f"{n['xF']:.2f}:{n['delta_over_sigma_combined']:+.1f}"
            for n in pooled[key]["dn_dxF_nodes"])
        lines.append(f"- {key}: {zs}")
    lines.append("")
    lines.append("## Wounded-scaling R_i(xF) node verdicts")
    lines.append("")
    lines.append("Model side: (dn/dxF)_model,i / ((1+<K>_slice)/2 "
                 "(dn/dxF)_model,pp) with the sampler's closed-form slice "
                 "mean; data side: (dn/dxF)_data,i / ((1+<nu>_i)/2 "
                 "(dn/dxF)_data,pp) with the measured <nu>_i and the "
                 "per-charge spshadrons p+p reference. The mapping "
                 "residual delta_i is the stated systematic on the "
                 "superposition factor.")
    lines.append("")
    lines.append("| set/charge | nodes | pass | tension | beyond | "
                 "no model support |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for key in sorted(pooled):
        counts = pooled[key]["r_count_summary"]
        lines.append(
            f"| {key} | {counts['n_nodes']} | {counts['n_pass']} | "
            f"{counts['n_tension']} | {counts['n_beyond']} | "
            f"{counts['n_no_model_support']} |")
    lines.append("")
    lines.append("Per-node model R (data R) values, same node order:")
    for key in sorted(pooled):
        parts = []
        for n in pooled[key]["r_nodes"]:
            if n.get("verdict") == "no_model_support":
                parts.append(f"{n['xF']:.2f}:n/s")
            else:
                parts.append(
                    f"{n['xF']:.2f}:{n['model_R']:.2f}"
                    f"({n['data_R']:.2f})")
        lines.append(f"- {key}: " + ", ".join(parts))
    lines.append("")
    lines.append("## Falsifier evaluations")
    lines.append("")
    f1 = report["falsifier_F_M13_1"]
    f2 = report["falsifier_F_M13_2"]
    lines.append(f"- **F-M13-1 (centrality-trend direction):** "
                 f"{'FIRED' if f1['fired'] else 'not fired'}.")
    for charge, rec in sorted(f1["per_charge"].items()):
        lines.append(
            f"  - {charge}: data trend {rec['data_trend_slope']:+.3e}, "
            f"model trend {rec['model_trend_slope']:+.3e}, "
            f"trend signs opposite: "
            f"{rec['trend_signs_opposite']}, endpoint verdicts "
            f"{rec['endpoint_verdicts']}, endpoints beyond tension: "
            f"{rec['endpoints_beyond_tension']}.")
    lines.append(f"- **F-M13-2 (wounded-scaling direction):** "
                 f"{'FIRED' if f2['fired'] else 'not fired'}; "
                 + f2["falsifier_4_cross_reference"])
    lines.append("")
    lines.append("## Conservation, wound bookkeeping, and acceptance sanity")
    lines.append("")
    cons = report["conservation"]
    lines.append(
        f"- Conservation (spec section 7.1): max relative residual "
        f"{cons['max_residual']:.3e} over all 48 production artifacts; "
        f"accounting violations {cons['accounting_violations']}, source "
        f"quantum-number violations "
        f"{cons['source_quantum_number_violations']}, antibaryon pair "
        f"violations {cons['antibaryon_pair_violations']}. "
        + cons["tests_statement"])
    lines.append("- Wound bookkeeping (generated <K>_slice versus the "
                 "closed-form slice mean, tolerance 4 standard errors):")
    for set_key in sorted(report["wound_bookkeeping"]):
        rec = report["wound_bookkeeping"][set_key]
        lines.append(
            f"  - {set_key}: generated {rec['generated_mean_K']:.5f} vs "
            f"closed form {rec['closed_form_mean_K']:.5f} "
            f"(tolerance {rec['tolerance']:.5f}) — "
            f"{'holds' if rec['holds'] else 'VIOLATED'}")
    lines.append("- Per-slice acceptance sanity at production scale (the "
                 "bin-by-bin 95% binomial band of the sliced Glauber "
                 "analytic distribution, qualifying bins expected >= 5):")
    for set_key in sorted(report["acceptance_sanity_production"]):
        rec = report["acceptance_sanity_production"][set_key]
        lines.append(
            f"  - {set_key}: {'holds' if rec['holds'] else 'VIOLATED'} "
            f"({rec['n_violating_bins']} violating of "
            f"{rec['n_qualifying_bins']} qualifying bins)")
    pilot = report.get("pilot") or {}
    pilot_acc = pilot.get("acceptance_sanity_pilot") or {}
    if pilot_acc:
        failing = [name for name, ok in sorted(pilot_acc.items()) if not ok]
        lines.append(
            "- Pilot-scale acceptance sanity (10000 accepted events per "
            "slice): "
            + ", ".join(f"{k} {'holds' if v else 'excursion'}"
                        for k, v in sorted(pilot_acc.items()))
            + (f"; the failing {failing} is the recorded deviation DEV-1 "
               "of this card, and the production-scale record above is "
               "the binding acceptance record (the excursion did not "
               "reproduce at 8x statistics)" if failing else
               "; all slices hold"))
    lines.append("")
    lines.append("## Geometry anchor cross-check (sanity, not a gate)")
    lines.append("")
    geo = report["geometry_anchor_crosscheck"]
    lines.append(
        f"sigma_pA = {geo['sampler']['sigma_pA_mb']:.1f} mb versus the "
        f"measured 1806 +/- 25 mb ({geo['sigma_pA_deviation_percent']:+.2f}%, "
        f"pull {geo['sigma_pA_pull']:+.2f} sigma); <K> = "
        f"{geo['sampler']['mean_target_wounds']:.4f} versus the measured "
        f"<nu> = 3.65 ({geo['mean_K_deviation_percent']:+.2f}%). "
        + geo["systematic_statement"])
    lines.append("")
    lines.append("## Target/projectile separation (model-internal, per slice)")
    lines.append("")
    lines.append("| set/charge | projectile pions | target pions | target "
                 "fraction |")
    lines.append("| --- | --- | --- | --- |")
    for set_key in sorted(report["target_projectile_separation"]):
        for charge, rec in sorted(report[
                "target_projectile_separation"][set_key].items()):
            frac = (f"{rec['fraction_target']:.4f}"
                    if rec["fraction_target"] is not None else "n/a")
            lines.append(
                f"| {set_key}_{charge} | {rec['n_projectile']} | "
                f"{rec['n_target']} | {frac} |")
    lines.append("")
    lines.append("The named mechanism: one continuing projectile source "
                 "plus K backward-shifted target sources (the asymmetric "
                 "wounded-source structure). The data are inclusive "
                 "forward densities and carry no decomposition, so the "
                 "statement is a model-internal one.")
    lines.append("")
    lines.append("## Statistics, wall, and budget accounting")
    lines.append("")
    wall = report["wall_and_budget_accounting"]
    lines.append(f"- Total events generated: "
                 f"{wall['total_events_generated']} (slices plus the p+p "
                 f"reference pool).")
    for set_key in sorted(wall["slice_wall_records"]):
        rec = wall["slice_wall_records"][set_key]
        mean_ms = (sum(rec["replica_ms_per_event"])
                   / max(len(rec["replica_ms_per_event"]), 1))
        proj = wall["menu_decision"].get(set_key, {})
        proj_wall = proj.get("projected_wall_s")
        proj_text = (f"{proj_wall:.0f}" if isinstance(proj_wall, float)
                     else str(proj_wall))
        lines.append(
            f"  - {set_key}: {rec['n_events_total']} events, mean "
            f"{mean_ms:.2f} ms/event, menu entry "
            f"{proj.get('chosen_events_per_replica', 'n/a')} "
            f"(projected serial wall {proj_text} s, budget "
            f"{wall['per_point_budget_s']:.0f} s per slice).")
    gw = wall.get("gateway") or {}
    if gw.get("record"):
        arms = (gw["record"].get("arms") or {})
        n_arms = len(arms)
        completed = sum(1 for a in arms.values()
                        if a.get("final_state") == "COMPLETED")
        lines.append(
            f"- Gateway grid: {n_arms} arms recorded, {completed} "
            f"COMPLETED (gateway_manifest_p2.json).")
    else:
        lines.append(f"- Gateway: {gw.get('status')}. "
                     + str(gw.get("statement", "")))
    lines.append("")
    lines.append("## Determinism")
    lines.append("")
    det = report["determinism"]
    lines.append(
        f"- Pilot determinism probe (five repeated evaluations, three "
        f"units): bit-identical = {det['pilot_probe']['deterministic']}.")
    lines.append(
        f"- Production-scale double generation (set-1 replica-0 and the "
        f"ppref replica-0 payloads built twice): bit-identical = "
        f"{det['production_double_run']['deterministic']}.")
    lines.append("")
    lines.append("## Deviations")
    lines.append("")
    if report["deviations"]:
        for dev in sorted(report["deviations"], key=lambda d: d["id"]):
            lines.append(f"- **{dev['id']}:** {dev['statement']}")
            lines.append(f"  - test: {dev['test']}")
    else:
        lines.append("None recorded (the production ran on the "
                     "pre-registered protocol without a deviation).")
    lines.append("")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] {path}", flush=True)


def collect_centrality_report(run_dir: Path, config_path: Path,
                              prereg_path: Path,
                              gateway_manifest_path: Path | None = None,
                              m8_record: Path | None = None) -> dict:
    """The P2 collection: every pre-registered observable computed against
    the pre-registered bands from the 48 production artifacts, the
    conservation and wound records, the geometry anchor cross-check, the
    falsifier evaluations, and the report writer. Refuses (SystemExit) on
    any missing artifact, mismatched constant, failed conservation check,
    or wound-bookkeeping violation."""
    run_dir = Path(run_dir)
    prereg = json.loads(Path(prereg_path).read_text(encoding="utf-8"))
    cross_check_preregistration(prereg)
    data_cc = cross_check_data_files(prereg)
    if data_cc["problems"]:
        raise SystemExit(
            "data-file cross-check failed at collection:\n  "
            + "\n  ".join(data_cc["problems"]))
    config_hash = canonical_configuration_hash(config_path)
    if config_hash != prereg.get("provenance", {}).get(
            "configuration_bundled_canonical_sha256"):
        raise SystemExit(
            "bundled config hash mismatch against the pre-registration "
            "provenance; refusing to collect")
    manifest = _load_manifest_verified(run_dir)
    slice_arts, ppref_arts = _load_replicas(run_dir, manifest)
    mapping = manifest["mapping_table"]

    # --- conservation gate over all 48 artifacts -----------------------------
    all_res = [art["conservation"]["max_residual"]
               for art in list(slice_arts.values()) + list(ppref_arts.values())]
    max_residual = float(max(all_res))
    accounting_violations = int(sum(
        art["conservation"]["accounting_violations"]
        for art in slice_arts.values()))
    qn_violations = int(sum(art["conservation"][
        "source_quantum_number_violations"]
        for art in list(slice_arts.values()) + list(ppref_arts.values())))
    antibaryon_violations = int(sum(art["conservation"][
        "antibaryon_pair_violations"]
        for art in list(slice_arts.values()) + list(ppref_arts.values())))
    conservation_ok = bool(
        max_residual <= 1e-10 and accounting_violations == 0
        and qn_violations == 0 and antibaryon_violations == 0)
    if not conservation_ok:
        raise SystemExit(
            "conservation gate failed at collection; refusing to publish "
            f"(max residual {max_residual:.3e}, accounting violations "
            f"{accounting_violations}, QN violations {qn_violations}, "
            f"antibaryon violations {antibaryon_violations})")

    # --- per-set/charge comparison --------------------------------------------
    pooled: dict = {}
    wound_records: dict = {}
    acceptance_records: dict = {}
    tp_records: dict = {}
    bootstrap_records: dict = {}
    bootstrap_yield_intervals: dict = {}
    bootstrap_node_intervals: dict = {}
    per_cell_records: dict = {}
    dn_records: dict = {}
    r_records: dict = {}
    wall_records: dict = {}
    pk = {int(k): float(v)
          for k, v in mapping["pk_bins_1_to_kmax"].items()}
    for set_no in (1, 2, 3, 4, 5):
        set_key = f"set{set_no}"
        band = mapping["sets"][str(set_no)]["band"]
        nu = float(mapping["sets"][str(set_no)]["nu_measured"])
        k_band_mean = float(mapping["sets"][str(set_no)]["k_band_mean"])
        k_band_var = float(mapping["sets"][str(set_no)]["k_band_variance"])
        residual_abs = float(mapping["sets"][str(set_no)]["residual_absolute"])
        residual_pct = float(mapping["sets"][str(set_no)]["residual_percent"])
        band_ambiguity = bool(mapping["sets"][str(set_no)]["band_ambiguity"])
        arts = [art for (s, _seed), art in sorted(slice_arts.items())
                if s == set_no]
        wall_records[set_key] = {
            "replica_walls_s": [float(art["wall_s"]) for art in arts],
            "replica_ms_per_event": [float(art["ms_per_event"])
                                     for art in arts],
            "n_events_total": int(sum(art["n_events"] for art in arts)),
        }
        # wound bookkeeping (the internal-consistency record)
        counts_k, n_pooled_w, mean_k_gen = _pooled_wounds(set_no, slice_arts)
        tol = (WOUND_BOOKKEEPING_N_SIGMA
               * math.sqrt(k_band_var / max(n_pooled_w, 1)) + 1e-9)
        wound_ok = bool(abs(mean_k_gen - k_band_mean) <= tol)
        wound_records[set_key] = {
            "band": [int(band[0]), int(band[1])],
            "generated_mean_K": float(mean_k_gen),
            "closed_form_mean_K": float(k_band_mean),
            "closed_form_variance": float(k_band_var),
            "n_events": int(n_pooled_w),
            "counts_per_k": {str(k): int(v) for k, v in sorted(counts_k.items())},
            "tolerance": float(tol),
            "tolerance_rule": INTERNAL_CONSISTENCY_TEXT["wound_bookkeeping"],
            "holds": wound_ok,
        }
        if not wound_ok:
            raise SystemExit(
                f"wound bookkeeping violated for {set_key}: generated "
                f"{mean_k_gen:.6f} vs closed form {k_band_mean:.6f} "
                f"exceeds the tolerance {tol:.6g}")
        # acceptance sanity at production scale
        sanity = acceptance_sanity_band(pk, int(band[0]), int(band[1]),
                                        counts_k, n_pooled_w)
        acceptance_records[set_key] = sanity
        # target/projectile separation (pooled)
        tp_records[set_key] = {}
        for charge in ("pi-", "pi+"):
            n_proj = sum(art["target_projectile_separation"][charge][
                "n_projectile"] for art in arts)
            n_targ = sum(art["target_projectile_separation"][charge][
                "n_target"] for art in arts)
            tp_records[set_key][charge] = {
                "n_projectile": int(n_proj),
                "n_target": int(n_targ),
                "fraction_target": (float(n_targ / (n_proj + n_targ))
                                    if (n_proj + n_targ) else None),
                "mechanism": (
                    "one continuing projectile source plus "
                    f"{int(band[0])}..{int(band[1])} backward-shifted "
                    "target sources (the asymmetric wounded-source "
                    "structure); the data are inclusive forward densities "
                    "and carry no decomposition, so the statement is a "
                    "model-internal one"),
            }

        # the bootstrap (one generator consumed sequentially, fixed order):
        # it runs before the banded records so the percentile intervals are
        # available inside the charge loop
        bootstrap_seed = slice_bootstrap_seed(set_no)
        rng = np.random.default_rng(int(bootstrap_seed))
        boot: dict = {}
        for charge in ("pi-", "pi+"):
            key = f"{set_key}_{charge}"
            cs = manifest["cell_sets"][key]
            cells = [(float(c["xF"]), float(c["pT"])) for c in cs["cells"]]
            dense_parts = []
            for (_s, _seed) in sorted(k for k in slice_arts if k[0] == set_no):
                art = slice_arts[(_s, _seed)]
                triples = art["species_binning"][charge][
                    "per_event_cell_triples"]
                dense_parts.append(expand_triples(
                    triples, int(art["n_events"]), len(cells)))
            dense = np.vstack(dense_parts) if dense_parts else np.zeros(
                (1, len(cells)), dtype=np.int64)
            unit = _bootstrap_slice_cells(dense, cells, cs["xf_nodes"], rng,
                                          BOOTSTRAP_REPLICATES)
            boot[charge] = {
                "cell_bootstrap_sigma": unit["cell_bootstrap_sigma"],
                "yield_percentile_interval": unit["yield_percentile"],
                "node_percentile_intervals": unit["node_percentile"],
                "resamples": unit["resamples"],
                "invalid_replicates": unit["invalid_replicates"],
            }
        bootstrap_records[set_key] = {
            "bootstrap_seed": int(bootstrap_seed),
            "order": ["pi-", "pi+"],
            "rule": ("200 event-level resamples per charge from one "
                     "generator consumed sequentially (the M9 convention "
                     "at the anchor +900 offset); the per-cell MC sigma "
                     "is the pre-registered product, the pooled-yield and "
                     "per-node dn/dxF percentile intervals ride the same "
                     "resamples"),
            "charges": boot,
        }
        for charge in ("pi-", "pi+"):
            key = f"{set_key}_{charge}"
            bootstrap_yield_intervals[key] = \
                bootstrap_records[set_key]["charges"][charge][
                    "yield_percentile_interval"]
            bootstrap_node_intervals.update(
                bootstrap_records[set_key]["charges"][charge][
                    "node_percentile_intervals"])

        for charge in ("pi-", "pi+"):
            key = f"{set_key}_{charge}"
            cs = manifest["cell_sets"][key]
            cells = [(float(c["xF"]), float(c["pT"])) for c in cs["cells"]]
            values = [float(c["data_value"]) for c in cs["cells"]]
            errors = [float(c["data_err"]) for c in cs["cells"]]
            pooled_counts, per_replica_counts, n_pooled, n_per = (
                _pooled_cells(set_no, charge, slice_arts, len(cells)))
            # data-side pooled kernel yield (merged unique-cell values)
            y_data, y_data_err = pooled_kernel_yield_data(values, errors,
                                                          cells)
            # model side: pooled and per-replica
            y_model = pooled_kernel_yield_model(pooled_counts, cells,
                                                n_pooled)
            y_per_replica = [
                pooled_kernel_yield_model(per_replica_counts[r], cells,
                                          n_per[r])
                for r in range(len(n_per))]
            y_mc = (float(np.std(y_per_replica, ddof=1))
                    if len(y_per_replica) > 1 else 0.0)
            sigma_comb = math.sqrt(y_data_err**2 + y_mc**2)
            z_yield = ((y_model - y_data) / sigma_comb
                       if sigma_comb > 0 else None)
            # the model dn/dxF per node (pooled + per-replica); the data
            # side is the pre-registered kernel-weighted integration of the
            # raw table rows
            nodes = cs["xf_nodes"]
            table = read_points(pa_data_dir()
                                / prereg["data_inventory"][key]["file"])
            data_dn = {f"{float(rec['xF']):.4g}": rec
                       for rec in data_side_dn_dxf(table)}
            dn_model_pooled = model_dn_dxf_nodes(pooled_counts, cells,
                                                 n_pooled, nodes)
            dn_per_replica = [model_dn_dxf_nodes(per_replica_counts[r],
                                                 cells, n_per[r], nodes)
                              for r in range(len(n_per))]
            # the model pp reference per node (pooled + per-replica, over
            # the set's own registered cells: the acceptance-cancellation
            # rule pinned in the manifest)
            ppref_arts_ordered = [ppref_arts[PPREF_ANCHOR + 1 + k]
                                  for k in range(K_REPLICAS)]
            pp_cell_counts = np.asarray(
                [art["species_binning"][key]["cell_counts"]
                 for art in ppref_arts_ordered], dtype=np.int64)
            pp_pooled_counts = pp_cell_counts.sum(axis=0)
            pp_n_per = [int(art["n_events"]) for art in ppref_arts_ordered]
            pp_n_pooled = int(sum(pp_n_per))
            dn_pp_pooled = model_dn_dxf_nodes(pp_pooled_counts, cells,
                                              pp_n_pooled, nodes)
            dn_pp_per_replica = [model_dn_dxf_nodes(pp_cell_counts[r],
                                                    cells, pp_n_per[r],
                                                    nodes)
                                 for r in range(K_REPLICAS)]
            # the R_i(xF) records (common nodes with the pp reference)
            r_data_rows = data_side_r_xf(set_no, charge, nu)
            r_data_by_node = {f"{float(rec['xF']):.4g}": rec
                              for rec in r_data_rows}
            superposition_model = (1.0 + k_band_mean) / 2.0
            node_entries = []
            for node in nodes:
                node_key = f"{float(node):.4g}"
                if node_key not in r_data_by_node:
                    continue
                d_data = r_data_by_node[node_key]
                if dn_model_pooled.get(node_key, 0.0) <= 0.0:
                    node_entries.append({
                        "xF": float(node),
                        "verdict": "no_model_support",
                        "reason": "no model support at this node",
                    })
                    continue
                dn_model_i = float(dn_model_pooled[node_key])
                dn_model_i_reps = [float(rec[node_key])
                                   for rec in dn_per_replica]
                dn_pp_p = float(dn_pp_pooled[node_key])
                dn_pp_reps = [float(rec[node_key]) for rec in dn_pp_per_replica]
                r_per_replica = [
                    (num / (superposition_model * den)
                     if den > 0 else float("nan"))
                    for num, den in zip(dn_model_i_reps, dn_pp_reps)]
                finite_r = [r for r in r_per_replica if math.isfinite(r)]
                r_mc = (float(np.std(finite_r, ddof=1))
                        if len(finite_r) > 1 else 0.0)
                r_model = (dn_model_i
                           / (superposition_model * dn_pp_p)
                           if dn_pp_p > 0 else float("nan"))
                r_data = float(d_data["R_data"])
                r_data_err = float(d_data["R_data_err"])
                sigma_r = math.sqrt(r_mc**2 + r_data_err**2)
                z_r = ((r_model - r_data) / sigma_r
                       if sigma_r > 0 and math.isfinite(r_model) else None)
                no_support_reason = None
                if not math.isfinite(r_model):
                    no_support_reason = (
                        "the model p+p reference integration is zero at "
                        "this node, so the model R_i is not defined")
                node_entries.append({
                    "xF": float(node),
                    "dn_dxF_model_i": dn_model_i,
                    "dn_dxF_model_pp": dn_pp_p,
                    "superposition_factor_model": float(superposition_model),
                    "superposition_factor_data": (1.0 + nu) / 2.0,
                    "model_R": (float(r_model)
                                if math.isfinite(r_model) else None),
                    "mc_err_replica_spread": r_mc,
                    "data_R": r_data,
                    "data_R_err": r_data_err,
                    "sigma_combined": sigma_r,
                    "delta": (r_model - r_data
                              if math.isfinite(r_model) else None),
                    "delta_over_sigma_combined": z_r,
                    "verdict": (verdict_from_z(z_r)
                                if z_r is not None else "no_model_support"),
                    "reason": no_support_reason,
                    "bootstrap_percentile": bootstrap_node_intervals.get(
                        node_key),
                    "stated_systematic_mapping_residual": {
                        "absolute": residual_abs,
                        "percent": residual_pct,
                        "statement": (
                            "the mapping residual delta_i on <nu>_i versus "
                            "<K>_slice carries as the stated systematic on "
                            "the superposition factor; the band-ambiguity "
                            "flag of the set is recorded alongside"),
                    },
                })
            # the dn/dxF per-node z-records and the per-set count summary
            dn_node_entries = []
            for node in nodes:
                node_key = f"{float(node):.4g}"
                d = data_dn.get(node_key)
                m = dn_model_pooled.get(node_key)
                if d is None or m is None:
                    continue
                reps = [float(rec[node_key]) for rec in dn_per_replica]
                mc = (float(np.std(reps, ddof=1)) if len(reps) > 1 else 0.0)
                sigma_dn = math.sqrt(float(d["dn_dxF_err"])**2 + mc**2)
                z_dn = ((m - float(d["dn_dxF"])) / sigma_dn
                        if sigma_dn > 0 else None)
                dn_node_entries.append({
                    "xF": float(node),
                    "dn_dxF_data": float(d["dn_dxF"]),
                    "dn_dxF_data_err": float(d["dn_dxF_err"]),
                    "dn_dxF_model": float(m),
                    "mc_err_replica_spread": mc,
                    "sigma_combined": sigma_dn,
                    "delta": (m - float(d["dn_dxF"])),
                    "delta_over_sigma_combined": z_dn,
                    "verdict": verdict_from_z(z_dn),
                    "bootstrap_percentile": bootstrap_node_intervals.get(
                        node_key),
                })
            dn_counts = {
                "n_nodes": len(dn_node_entries),
                "n_pass": sum(1 for n in dn_node_entries
                              if n["verdict"] == "pass"),
                "n_tension": sum(1 for n in dn_node_entries
                                 if n["verdict"] == "tension"),
                "n_beyond": sum(1 for n in dn_node_entries
                                if n["verdict"] == "beyond_tension"),
            }
            r_counts = {
                "n_nodes": sum(1 for n in node_entries
                               if "verdict" in n),
                "n_pass": sum(1 for n in node_entries
                              if n.get("verdict") == "pass"),
                "n_tension": sum(1 for n in node_entries
                                 if n.get("verdict") == "tension"),
                "n_beyond": sum(1 for n in node_entries
                                if n.get("verdict") == "beyond_tension"),
                "n_no_model_support": sum(
                    1 for n in node_entries
                    if n.get("verdict") == "no_model_support"),
            }
            pooled[key] = {
                "yield_data": y_data,
                "yield_data_err": y_data_err,
                "yield_model": y_model,
                "yield_mc_err": y_mc,
                "sigma_combined": sigma_comb,
                "delta": (y_model - y_data),
                "z": z_yield,
                "yield_verdict": verdict_from_z(z_yield),
                "yield_bootstrap_percentile": bootstrap_yield_intervals.get(
                    key),
                "mapping_residual_absolute": residual_abs,
                "mapping_residual_percent": residual_pct,
                "band_ambiguity": band_ambiguity,
                "n_cells": len(cells),
                "n_events_pooled": n_pooled,
                "dn_dxF_nodes": dn_node_entries,
                "dn_dxF_count_summary": dn_counts,
                "r_nodes": node_entries,
                "r_count_summary": r_counts,
            }
            dn_records[key] = dn_node_entries
            r_records[key] = node_entries
            # per-cell records: every data row versus the pooled model cell
            # value with its own error (recorded, not banded)
            cell_density_pooled = []
            for (xf, pt), n_pooled_cell in zip(cells, pooled_counts):
                dxf, dpt = cell_full_widths(xf, pt)
                energy = pion_energy(xf, pt)
                cell_density_pooled.append(float(
                    (energy / (math.pi * SQRT_S_GEV * pt))
                    * (int(n_pooled_cell) / (float(n_pooled) * dxf * dpt))))
            per_repl_densities = []
            for r in range(len(n_per)):
                per_repl_densities.append([
                    float((pion_energy(xf, pt) / (math.pi * SQRT_S_GEV * pt))
                          * (int(n) / (float(n_per[r])
                                       * cell_full_widths(xf, pt)[0]
                                       * cell_full_widths(xf, pt)[1])))
                    for (xf, pt), n in zip(cells, per_replica_counts[r])])
            per_cell_recs = []
            for row_idx, (xf, pt, v, e) in enumerate(cs["raw_rows"]):
                cell_key = (round(float(xf), 4), round(float(pt), 4))
                in_registered = cell_key in {tuple(c) for c in cells}
                cell_idx = ({tuple(c): i for i, c in enumerate(cells)}
                            .get(cell_key))
                model_value = (cell_density_pooled[cell_idx]
                               if cell_idx is not None else None)
                mc_value = (float(np.std(
                    [per_repl_densities[r][cell_idx]
                     for r in range(len(n_per))], ddof=1))
                    if cell_idx is not None and len(n_per) > 1 else None)
                per_cell_recs.append({
                    "xF": float(xf), "pT": float(pt),
                    "data_value": float(v), "data_err": float(e),
                    "model_cell_value": model_value,
                    "model_mc_err_replica_spread": mc_value,
                    "in_registered_support": bool(in_registered),
                    "row_index": int(row_idx),
                })
            per_cell_records[key] = {
                "n_rows": len(per_cell_recs),
                "n_registered_rows": sum(
                    1 for r in per_cell_recs if r["in_registered_support"]),
                "records": per_cell_recs,
            }

    # --- falsifier evaluations (card sequence 5) ------------------------------
    falsifier_1 = _fm13_1_evaluation(pooled)
    falsifier_2 = _fm13_2_evaluation(r_records, verdict_bands_rule := (
        BAND_RULE_TEXT))

    # --- geometry anchor cross-check (sanity, not a gate) ----------------------
    geometry = geometry_anchor_crosscheck()

    # --- wall and budget accounting --------------------------------------------
    menu = manifest.get("menu_decision") or {}
    total_events = sum(rec["n_events_total"] for rec in wall_records.values())
    total_wall = (sum(sum(rec["replica_walls_s"]) for rec in
                      wall_records.values())
                  + sum(float(art["wall_s"]) for art in ppref_arts.values()))
    ppref_walls = {int(seed): float(art["wall_s"])
                   for seed, art in sorted(ppref_arts.items())}
    wall_accounting = {
        "menu_decision": menu,
        "menu_rule": (
            "the largest menu entry whose projected wall (measured "
            "per-slice per-event cost x K=8 replicas x E events) stays "
            "under the pre-registered 18000 s per-slice budget; a declared "
            "reduction carries the MC-sigma caveat"),
        "slice_wall_records": wall_records,
        "ppref_replica_walls_s": ppref_walls,
        "ppref_events_total": int(sum(art["n_events"]
                                      for art in ppref_arts.values())),
        "total_events_generated": int(total_events
                                      + sum(art["n_events"]
                                            for art in ppref_arts.values())),
        "total_wall_s": float(total_wall),
        "per_point_budget_s": float(MAX_PROJECTED_WALL_S),
        "per_point_budget_rule": (
            "each production point is one replica at the pre-registered "
            "statistics; the pre-registered per-slice budget bounds the "
            "projected serial wall of the whole slice (8 replicas x E "
            "events)"),
        "gateway": None,
    }
    gateway_path = (Path(gateway_manifest_path)
                    if gateway_manifest_path is not None
                    else run_dir / GATEWAY_MANIFEST_NAME)
    if gateway_path.exists():
        wall_accounting["gateway"] = {
            "manifest_path": str(gateway_path),
            "record": json.loads(gateway_path.read_text(encoding="utf-8")),
        }
    else:
        probe_gw = run_dir / "probe_gateway_capacity_p2.json"
        fallback = None
        if probe_gw.exists():
            payload = json.loads(probe_gw.read_text(encoding="utf-8"))
            fallback = payload.get("serial_fallback_statement")
        wall_accounting["gateway"] = {
            "status": "serial_fallback",
            "statement": (fallback
                          or "the production points ran serially with the "
                             "recorded per-point wall budgets (the "
                             "recorded serial fallback of the M11 "
                             "pattern)"),
        }

    # --- determinism summary -----------------------------------------------------
    det_probe_path = run_dir / "probe_determinism_p2.json"
    det_prod_path = run_dir / "determinism_production_p2.json"
    if not det_probe_path.exists() or not det_prod_path.exists():
        raise SystemExit(
            "the determinism artifacts are missing at collection ("
            f"{det_probe_path}, {det_prod_path}); refusing to publish")
    det_probe = json.loads(det_probe_path.read_text(encoding="utf-8"))
    det_prod = json.loads(det_prod_path.read_text(encoding="utf-8"))
    determinism_summary = {
        "pilot_probe": {
            "path": str(det_probe_path),
            "deterministic": det_probe.get("deterministic"),
            "units": {name: {"seed": unit["seed"],
                             "bit_identical": unit["bit_identical"]}
                      for name, unit in det_probe.get("units", {}).items()},
        },
        "production_double_run": {
            "path": str(det_prod_path),
            "deterministic": det_prod.get("deterministic"),
            "units": {name: {"seed": unit["seed"],
                             "bit_identical": unit["bit_identical"],
                             "payload_sha256": unit["payload_sha256"]}
                      for name, unit in det_prod.get("units", {}).items()},
        },
        "bit_identical": bool(det_probe.get("deterministic")
                              and det_prod.get("deterministic")),
    }
    if not determinism_summary["bit_identical"]:
        raise SystemExit(
            "the determinism evidence is not bit-identical at collection; "
            "refusing to publish")

    # --- pilot record ------------------------------------------------------------
    pilot_path = run_dir / PILOT_RECORD_NAME
    pilot_summary = None
    if pilot_path.exists():
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        pilot_summary = {
            "path": str(pilot_path),
            "all_pins_hold": pilot.get("all_pins_hold"),
            "acceptance_sanity_pilot": pilot.get(
                "acceptance_sanity_pilot"),
            "cost_cells": pilot.get("cost_cells"),
        }
    else:
        pilot_summary = {
            "path": str(pilot_path),
            "status": "missing (recorded; the pilot record is expected "
                      "before the production launch)",
        }

    # --- the report ---------------------------------------------------------------
    mapping_sets = mapping["sets"]
    report = {
        "schema": REPORT_SCHEMA,
        "card": CARD, "milestone": MILESTONE, "phase": PHASE_P2,
        "run_description": RUN_DESCRIPTION,
        "provenance": {
            "configuration_hash": config_hash,
            "config_seed": CONFIG_SEED,
            "m13_seed_base": M13_SEED_BASE,
            "pre_registration_sha256": _sha256_file(Path(prereg_path)),
            "cells_manifest_path": str(run_dir / SLICE_MANIFEST_NAME),
            "cells_manifest_sha256": _sha256_file(
                run_dir / SLICE_MANIFEST_NAME),
            "helper_sha256": _sha256_file(Path(__file__)),
            "written_utc": _utc_now_iso(),
        },
        "claim_statement": REPORT_CLAIM_STATEMENT,
        "band_rule": {
            "k_pass": K_PASS, "k_tension": K_TENSION,
            "rule": BAND_RULE_TEXT,
            "combined_sigma_convention": (
                "the band denominator is the quadrature of the data error "
                "from the tables and the MC error from the replica spread "
                "(std ddof=1 of the 8 per-replica values); the mapping "
                "residual and the anchor deviation are recorded as stated "
                "systematics alongside every verdict (the M12 execution of "
                "the pre-registered combined-error convention)"),
        },
        "mapping_table": mapping,
        "band_ambiguity_systematic": BAND_AMBIGUITY_SYSTEMATIC_STATEMENT,
        "pooled_kernel_yields": pooled,
        "per_cell_records": per_cell_records,
        "wound_bookkeeping": wound_records,
        "acceptance_sanity_production": acceptance_records,
        "target_projectile_separation": tp_records,
        "bootstrap": bootstrap_records,
        "falsifier_F_M13_1": falsifier_1,
        "falsifier_F_M13_2": falsifier_2,
        "geometry_anchor_crosscheck": geometry,
        "wall_and_budget_accounting": wall_accounting,
        "determinism": determinism_summary,
        "pilot": pilot_summary,
        "conservation": {
            "max_residual": max_residual,
            "accounting_violations": accounting_violations,
            "source_quantum_number_violations": qn_violations,
            "antibaryon_pair_violations": antibaryon_violations,
            "holds": conservation_ok,
            "tests_statement": (
                "spec section 7.1 tests 1-4 per event through "
                "event.validate() at 1e-10 (the max residual recorded), "
                "test 5 the determinism evidence, test 6 established at "
                "the M12 P2 record on the identical _generate_pp path, "
                "test 7 the antibaryon pair margins per event, test 8 the "
                "p+A accounting A+1/Z+1 per event; the p+A accounting is "
                "the card's required test and its maximum residual is "
                "recorded per replica and pooled"),
            "per_artifact_max_residuals": {
                f"set{set_no}_{seed}": float(art["conservation"][
                    "max_residual"])
                for (set_no, seed), art in sorted(slice_arts.items())},
        },
        "deviations": P2_DEVIATIONS,
    }
    _write_json(run_dir / REPORT_JSON_NAME, report)
    write_report_md(report, run_dir / REPORT_MD_NAME)
    print(f"[collect] report written: {run_dir / REPORT_JSON_NAME}",
          flush=True)
    return report


# --- Pre-registered protocol texts ------------------------------------------------

OBJECTIVE_TEXT = (
    "Held-out predictive validation of the staged excitation model at the "
    "frozen M8 ten-parameter calibration set (no refit): the p+A "
    "centrality test across impact-parameter classes, the five "
    "centrality-selected sets of the Chvala thesis for p+Pb208 at 158 "
    "GeV/c (per-event invariant density d(xF,pT)), through the "
    "pre-registered E_BE-to-K cross-calibration that maps each thesis "
    "centrality set (a forward-energy E_BE selection via grey-proton "
    "counting) to a wounded-nucleon number K slice of the model's Glauber "
    "geometry; verdicts recorded against the pre-registered bands with the "
    "mapping residual carried as a stated systematic. This single "
    "machine-readable protocol fixes the observables, the mapping table, "
    "the grids and merge rule, the seeds, the statistics menu, and the "
    "verdict bands before any production event generation; the harness "
    "cross-checks its constants verbatim and refuses a mismatch.")

DECISION_RULE_TEXT = (
    "the production runs of P2 at these pre-registered settings are the "
    "record; the bands, the seeds, the mapping, the grids, and the "
    "statistics menu are not tuned after seeing a run. Any protocol change "
    "during a run is recorded as a deviation with its reason in the run "
    "report.")

ANCHOR_CONTEXT_STATEMENT = (
    "the sampler values at the pinned geometry (sigma_pA 1692.0 mb and <K> "
    "3.9371 for Pb208) are the geometry-anchor sanity cross-checks against "
    "the measured anchors sigma_inel(p+Pb) = 1806 +/- 25 mb and <nu> = 3.65 "
    "(sanity, not a gate); the 2026-08-25 cross-check at these values gave "
    "sigma_pA 1686 mb versus 1806 +/- 25 mb measured, -6.7%, and <K> 3.93 "
    "versus <nu> 3.65, +7.7%, and ran the tower-model lineage, cited as "
    "geometry context only")

NORMALIZATION_STATEMENT = (
    "the per-event invariant density d(xF,pT) of the thesis set samples is "
    "the primary observable (it needs no cross-section anchor, mirroring "
    "the M12 P3 convention); the model density uses the M12 "
    "pa_per_event_density convention (E/(pi sqrt(s) pT) * n/(N_ev dxF "
    "dpT), binned on the data's own grid, the pinned cell half-width "
    "convention doubled into the full bin widths). The R_i(xF) diagnostic "
    "uses per-event dn/dxF on both sides, with the measured <nu>_i in the "
    "data-side superposition factor and the sampler's <K>_slice in the "
    "model-side factor (the foundations convention <N_w> = 1 + <K>). The "
    "measured anchors sigma_inel(p+Pb) = 1806 +/- 25 mb and <nu> = 3.65 "
    "enter as the geometry anchor cross-check (sanity, not a gate)")

MAPPING_RULE_TEXT = (
    "E_BE-to-K mapping rule (pre-registered, closed-form, no generation): "
    "for the pinned geometry the Glauber wounded-count distribution P(K) "
    "and the per-band slice means <K>_band are computed from the sampler's "
    "closed-form arithmetic (fraction_for_wound_range and the per-bin "
    "conditional means, glauber.py), with no generation. Candidate bands "
    "are contiguous integer intervals [K, K+w] with width w in {0, 1, 2, "
    "3} and K in 1..K_max, where K_max is the 99.9% cumulative-mass "
    "truncation of P(K) (pinned in the pre-registration). Each set i is "
    "assigned the band minimizing |<K>_band - <nu>_i|; the residual delta_i "
    "(absolute and percent), the slice fraction f_i (model) versus the "
    "thesis event count fraction (cross-check, recorded), and the +-10% "
    "systematic are recorded per set. If <nu>_i +- 0.10 <nu>_i changes the "
    "minimizing band, the set carries a band-ambiguity flag (the "
    "alternative band is recorded; the comparison uses the chosen band; "
    "the ambiguity is stated as the systematic on that set's verdict). "
    "The five bands must be pairwise disjoint; a conflict resolves by each "
    "set keeping its own unique minimizer, with the overlap recorded. A "
    "band with no P(K) mass is unmappable, and its comparison is dropped "
    "at pre-registration with the recorded reason.")

MERGE_RULE_TEXT = (
    "each unique (xF, pT) cell's data value is the inverse-variance "
    "weighted merge of all rows of that cell (including the "
    "duplicated-label rows), value = sum(v_i / s_i^2) / sum(1 / s_i^2), "
    "with the merged absolute error 1 / sqrt(sum(1 / s_i^2)); the "
    "per-cell records keep every row (duplicated rows each) against the "
    "one model cell value; the merge fixture pins a few real cells' merged "
    "values and errors")

COMMON_SUPPORT_RULE_TEXT = (
    "the common support is the data's own unique-cell grid after two "
    "recorded drops (pure table arithmetic, no generation): the pT = 0.00 "
    "label cells of the coarse-grid files (the singular thesis Jacobian, "
    "PT_ZERO_DROP_REASON), and any cell outside the pinned model support "
    "(the probe-measured final-pion xF/pT envelope, model_support_probe "
    "block). Windows with no data support on either side are dropped "
    "here, with the recorded reason, before any run")

OBSERVABLE_DEFINITIONS = {
    "pooled_kernel_yield": (
        "(a) the pooled kernel yield Y_{i,c} = sum over the registered "
        "common-support cells of (pT/E) d(xF,pT) dxF dpT, with E = "
        "sqrt(m_pi^2 + pL^2 + pT^2), pL = xF sqrt(s)/2 (the kernel of the "
        "paper detailed review section 2, the M12 convention); the model "
        "side from the M12 pa_per_event_density convention (E/(pi sqrt(s) "
        "pT) * n/(N_ev dxF dpT), binned on the data's own grid with the "
        "first-match rule and the pinned cell half-width convention "
        "doubled into the full bin widths); the data side on the "
        "unique-cell grid, each unique cell's data value the "
        "inverse-variance merged value with the merged absolute error; "
        "the combined 1-sigma the quadrature of the data error and the MC "
        "error from the replica spread, plus the stated systematics (the "
        "mapping residual delta_i and the anchor deviation)"),
    "per_cell_records": (
        "(b) every data row (duplicated rows each) versus the model cell "
        "value at that (xF, pT) node with its own absolute error; "
        "recorded, not banded"),
    "dn_dxF_profile": (
        "(c) the pT-integrated dn/dxF(xF) per xF node: the data side the "
        "kernel-weighted pT integration of the 2D rows (pi sqrt(s)/E) pT "
        "d dpT with quadrature error propagation (the pT = 0.00 label "
        "carries zero kernel weight); the model side the kernel-weighted "
        "integration of the model's binned densities; per-node z-records "
        "with the combined 1-sigma, and the per-set count of nodes in "
        "band / at tension / beyond (the M12 P3 R(xF) summary style)"),
    "wounded_scaling_R_i": (
        "(d) the wounded-scaling diagnostic R_i(xF): data side "
        "(dn/dxF)_data,i(xF) / ((1+<nu>_i)/2 * (dn/dxF)_data,pp(xF)) with "
        "the measured <nu>_i and the per-charge spshadrons p+p reference "
        "(the M12 seagull tables, percent errors converted to absolute); "
        "model side (dn/dxF)_model,i(xF) / ((1+<K>_slice,i)/2 * "
        "(dn/dxF)_model,pp(xF)) with the sampler's slice mean and the "
        "model p+p reference pool; per-node z = (R_model - R_data)/sigma_"
        "combined with the mapping residual delta_i carried as a stated "
        "systematic on the superposition factor"),
    "wound_bookkeeping": (
        "(e) the wound bookkeeping: the generated <K>_slice of the "
        "accepted events versus the closed-form <K>_slice,i, within the "
        "pre-registered statistical consistency tolerance (the "
        "internal_consistency block)"),
    "target_projectile_separation": (
        "(f) the model-internal target/projectile separation per slice "
        "(the source-role decomposition of the pion yields; the named "
        "mechanism: one continuing projectile source plus K "
        "backward-shifted target sources, the asymmetric structure of "
        "_knowledge/foundations/pa-wounded-nucleon.md section 2); the "
        "data are inclusive forward densities and carry no decomposition, "
        "so there is no data-side counterpart and the statement is a "
        "model-internal one"),
    "acceptance_sanity": (
        "(g) the per-slice acceptance sanity: the accepted wound-count "
        "distribution inside the slice, bin-by-bin 95% binomial band of "
        "the sliced Glauber analytic distribution, qualifying bins with "
        "expected count at least 5 (the M5 audit statement adapted to "
        "slices)"),
    "combined_sigma_rule": (
        "the combined 1-sigma of a banded observable is the quadrature of "
        "the data error from the tables and the MC error from the replica "
        "spread, plus the stated systematics (the mapping residual "
        "delta_i on the mapped band and the anchor deviation); the R_i "
        "denominator carries the p+p reference error in the same "
        "quadrature"),
}

STATISTICS_BLOCK = {
    "k_replicas": K_REPLICAS,
    "menu_events_per_replica": list(STATISTICS_MENU_EVENTS_PER_REPLICA),
    "primary_events_per_replica": PRIMARY_EVENTS_PER_REPLICA,
    "max_projected_wall_s": MAX_PROJECTED_WALL_S,
    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    "determinism_repeats": DETERMINISM_REPEATS,
    "cost_cell_events": COST_CELL_EVENTS,
    "acceptance_sanity_events": ACCEPTANCE_SANITY_EVENTS,
    "pp_reference_events_per_replica": PP_REF_EVENTS_PER_REPLICA,
    "per_slice_statement": (
        "per slice: 8 fixed-seed replicas of 10000 events (400000 p+Pb "
        "events over the five slices) at the M8 set, menu [8x10000, "
        "8x5000, 8x2500, 8x1000], the largest menu entry whose projected "
        "wall (measured per-slice per-event cost from the P2 cost probe, "
        "300-event cells, one single sample per cell at fixed parameters, "
        "M10 probe_C pattern) stays under the pre-registered budget; the "
        "declared reduction, if used, carries the MC-sigma caveat (M10 "
        "pattern: the replica-mean MC sigma grows as sqrt(10000/E) "
        "relative to the primary entry)"),
    "pp_reference_statement": (
        "the p+p reference pool for the R_i(xF) diagnostic at the same "
        "replica convention (8 x 10000; the p+p staged kernel ran at "
        "about 2 ms/event under H3 in the M10 probe)"),
    "bootstrap_statement": (
        "bootstrap per slice: 200 event-level resamples (the M9 "
        "convention) from one bootstrap seed, one generator consumed "
        "sequentially, for the per-cell MC sigma"),
}

BAND_RULE_TEXT = (
    "per banded observable: pass if |delta| at most k_pass = 2 times the "
    "combined 1-sigma uncertainty (data error from the tables, MC error "
    "from the replica spread, combined in quadrature, plus the stated "
    "systematics), tension if between k_pass and k_tension = 5 times the "
    "combined uncertainty, and falsification-flag if beyond k_tension in "
    "a pre-registered physical direction. k_pass and k_tension are fixed "
    "here, before any run (the M9/M10 pattern: bands fixed before the "
    "number is looked at)")

COST_PLAN_TEXT = {
    "probe_schedule": (
        "the probes run at the start of P2 at the scheduled pilot "
        "sub-streams: determinism (five bit-identical repeated evaluations "
        "at a fixed point, M10 probe_A pattern, at the sliced builder path "
        "and the minimum-bias path), per-slice cost cells (300 events per "
        "slice, sets 1..5 in order, one single sample per cell at fixed "
        "parameters, M10 probe_C pattern), per-slice acceptance sanity "
        "(10000 accepted events per slice), and a gateway capacity probe "
        "(read-only)"),
    "budget": (
        "the production launch is refused if a probe misses its "
        "pre-registered expectation without a recorded deviation; the "
        "statistics menu entry is the largest entry whose projected wall "
        "(measured per-slice per-event cost x K = 8 replicas x E events) "
        "stays under the pre-registered per-slice budget"),
    "gateway_statement": (
        "P1 uses no gateway jobs; the cluster (workers c0801-c0809) is "
        "available for the P2 production runs"),
    "pilot_scale_statement": (
        "no production-statistics generation happens in P1; the "
        "regression-pin tests run small fixed-seed samples (the standard "
        "test-suite scale) and the model-support probe of the "
        "pre-registration is a small fixed-seed yield-counting sample, "
        "excluded from production statistics; the pilot-scale probes run "
        "at the start of P2"),
}

REGRESSION_PINS_TEXT = {
    "file": "dev/mixing2006/tests/test_pa_builder_wound_slices.py",
    "pins": [
        {"id": "min_bias_stream_unchanged",
         "statement": (
             "at a fixed seed with no target_wound_range, the new code "
             "path reproduces the pre-existing sample_event stream on the "
             "builder path (the (b, K) sequence frozen by a test against "
             "the pre-change behavior, the foundations section 5 "
             "'min-bias (b, K) stream frozen against pre-conditioning "
             "literals' pin extended to the builder path)")},
        {"id": "sliced_stream_matches_sample_event_in_range",
         "statement": (
             "at a fixed seed and a fixed slice, the builder's (b, K) "
             "draws match the pinned sample_event_in_range behavior (the "
             "existing chunked-worker pins apply unchanged; the new pin "
             "covers the builder path)")},
        {"id": "rare_slice_and_point_target_pins_unchanged",
         "statement": (
             "the rare-slice pins (Pb208 K = 21, 24, 25 fractions 5.6e-7, "
             "1.2e-8, 3.1e-9) and the point-target pin (foundations "
             "section 5) remain green unchanged, extended by a "
             "builder-path rare-slice and point-target event")},
        {"id": "builder_worker_equivalence",
         "statement": (
             "the builder's sliced stream and the chunked-worker's "
             "GenerationConfig.target_wound_range stream, at the same "
             "nucleus fields and matched seed convention, agree on the "
             "slice-mean wound count statistics (the M12 P3 'equivalent "
             "generation entry' requirement)")},
        {"id": "config_validation",
         "statement": (
             "the PAExcitationConfig target_wound_range construction "
             "rules mirror the GenerationConfig rules: integer lo >= 1, "
             "hi >= lo, lo <= A, the point-target K = 1 rule, the "
             "non-zero restricted probability at builder construction, "
             "and the unchanged min-bias default")},
    ],
}

INTERNAL_CONSISTENCY_TEXT = {
    "wound_bookkeeping": (
        "the generated <K>_slice versus the closed-form <K>_slice,i "
        f"(internal consistency record): the consistency tolerance is "
        f"{WOUND_BOOKKEEPING_N_SIGMA:g} standard errors of the generated "
        "mean, sqrt(var_slice / N_gen) with the closed-form per-set slice "
        "variance, plus 1e-9"),
    "acceptance_sanity": (
        "the per-slice acceptance sanity: the accepted wound-count "
        "distribution inside the slice, bin-by-bin 95% binomial band of "
        "the sliced Glauber analytic distribution, qualifying bins with "
        "expected count at least 5 (the M5 audit statement adapted to "
        "slices); a conservation-level statement at pilot scale, excluded "
        "from production statistics"),
}

HARNESS_TEXT = {
    "cross_check_verifies": [
        "the frozen M8 parameter set against the baryon_nucleus."
        "frozen_pp_parameters block of "
        "dev/00runs/staged_binned_fit_mc_2026-09-06/"
        "staged_evaluation_report.json by file sha256 and by canonical "
        "block sha256",
        "the geometry against the bundled configuration (the canonical "
        "configuration sha256 and the nucleus/pA block values)",
        "the data files (existence and the row-count and grid inventory, "
        "recomputed from the tables against the pre-registration)",
        "the seed block against the M12 record (the M12 pre-registration "
        "sha256) for the no-collision statement",
        "the mapping table (deterministic recomputation of the closed-form "
        "arithmetic)",
        "every protocol constant of this file verbatim (seeds, menu, "
        "bands, mapping table, grids, model-support pin, and the protocol "
        "constant blocks)",
    ],
    "entry_point": CROSS_CHECK_COMMAND + " (from the monorepo root; RUNS_ROOT dev/00runs)",
    "mirror_rule": (
        "the two helper copies are byte-identical; the tree is detected "
        "from __file__ so the mixing2006 package and the tree-local run "
        "root resolve on both sides; the layout-sync check must pass with "
        "zero drift"),
    "planned_paths": [
        "helpers/benchmarking/staged_centrality_sets.py (new)",
        "dev/mixing2006/helpers/benchmarking/staged_centrality_sets.py "
        "(byte-identical vendored mirror)",
        "dev/00runs/staged_centrality_sets_2026-09-11/pre_registration.json "
        "(this file)",
        "dev/00runs/staged_centrality_sets_2026-09-11/pre_registration.sha256 "
        "(the recorded sha256)",
        "dev/00runs/staged_centrality_sets_2026-09-11/"
        "harness_cross_check.json (the cross-check record with its exit "
        "status)",
        "dev/00runs/tmp/m13_phase_1/ (scratch)",
    ],
    "test_plan": (
        "helpers/benchmarking/tests/test_staged_centrality_sets.py: the "
        "table-reading tests carry @pytest.mark.external_data (the nested "
        "helpers gate runs -m \"not external_data\"); the seed-block, "
        "merge-rule, band-selection, and cross-check machinery tests are "
        "hermetic and unmarked; every test reading spshadrons.web.cern.ch/ "
        "or helpers/benchmarking/pa_data/ carries the external_data "
        "marker"),
}


# --- Pre-registration builder -----------------------------------------------------


def build_pre_registration(config_path: Path, sampler=None) -> dict:
    """Compose the single machine-readable M13 P1 protocol from the helper's
    own computed values (no manual transcription): the grid and merge-rule
    inventory, the closed-form mapping table, the data-side profiles, the
    seed block, the statistics menu, the verdict bands and falsifier
    directions, the cost plan, and the pinned model support."""
    if sampler is None:
        sampler = _load_sampler()
    mapping = build_mapping_table(sampler)
    support_probe = run_model_support_probe(config_path, sampler)
    inventory = build_data_inventory(sampler=sampler, support=support_probe)
    prereg = {
        "card": CARD,
        "milestone": MILESTONE,
        "phase": PHASE,
        "date": "2026-09-11",
        "run_description": RUN_DESCRIPTION,
        "objective": OBJECTIVE_TEXT,
        "decision_rule": DECISION_RULE_TEXT,
        "calibration_set_decision_of_record": {
            "bundled_config_canonical_sha256": BUNDLED_CONFIG_CANONICAL_SHA256,
            "decision": (
                "M13 evaluates at the frozen M8 ten-parameter calibration "
                "set (the current calibration under the uncertainty-aware "
                "convention), decay hypothesis H3. No refit, no tuning: no "
                "optimizer runs on the set data, and no per-target, "
                "per-centrality, or per-species retuning of any p+p decay "
                "parameter (spec section 11). A deviation from the frozen "
                "set is a milestone failure unless the orchestrator "
                "authorizes it on this card."),
            "error_convention": (
                "the M13 comparisons use the pre-registered combined-error "
                "convention (data error from the tables, MC error from the "
                "replica spread, combined in quadrature, plus the stated "
                "systematics: the mapping residual and the anchor "
                "deviation) and are not comparable to the M8 chi-2 numbers "
                "or to the M11 replica-block values"),
            "m8_frozen_block_sha256": M8_FROZEN_BLOCK_SHA256,
            "m8_frozen_pp_parameters": M8_FROZEN_PP_PARAMETERS,
            "m8_parameter_set": M8_PARAMETER_SET,
            "m8_record_path": M8_RECORD_RELPATH,
            "m8_record_sha256": M8_RECORD_SHA256,
            "other_fields": (
                "all other dataclass fields keep their bundled-config "
                "values; the config declares no continuum.decay_hypothesis, "
                "so the H3 kernel applies as the dataclass default"),
            "verification": (
                "the harness recomputes the sha256 of the M8 record file "
                "and the sha256 of the canonical JSON of its "
                "baryon_nucleus.frozen_pp_parameters block and refuses a "
                "mismatch before any generation"),
        },
        "machine_constants": {
            "beam_momentum_gev_c": BEAM_MOMENTUM_GEV_C,
            "production_mode": PRODUCTION_MODE,
            "sqrt_s_gev": SQRT_S_GEV,
            "sqrt_s_note": (
                "the sqrt(s) of Reaction(158.0, 2212, 2212), the M4 "
                "convention; the NN sqrt(s) of the p+A observables"),
        },
        "geometry": {
            "Pb208": GEOMETRY_PB208,
            "pa_runner_settings": PA_RUNNER_SETTINGS,
            "anchor_context_statement": ANCHOR_CONTEXT_STATEMENT,
            "pinned_statement": (
                "the geometry is fixed at the bundled configuration values, "
                "no density-parameter tuning in this card; the recorded "
                "anchor deviation is the stated systematic, and a "
                "geometry-tuning campaign is a recorded follow-up, not "
                "this card"),
            "sampler_anchors": {
                "Pb208": {
                    "geometry": dict(GEOMETRY_PB208),
                    "mean_target_wounds": float(
                        sampler.mean_target_wounds_minbias()),
                    "sigma_pA_mb": float(sampler.sigma_pA_mb()),
                },
            },
        },
        "normalization_anchors": {
            "sigma_inel_pPb_mb": SIGMA_INEL_PPb_MB,
            "sigma_inel_pPb_err_mb": SIGMA_INEL_PPb_ERR_MB,
            "nu_pPb_measured": NU_PPB_MEASURED,
            "nu_systematic_fraction": NU_SYSTEMATIC_FRACTION,
            "normalization_statement": NORMALIZATION_STATEMENT,
        },
        "thesis_centrality_table": {
            "source": (
                "Chvala 2006 thesis, Table 5.4 (tabcdbins, thesis lines "
                "2388-2405): the grey-proton selections and measured mean "
                "intranuclear collision numbers per set, p+Pb at 158 GeV/c"),
            "sets": THESIS_CENTRALITY_SETS,
            "context_nu": CONTEXT_ROW_NU,
            "events_eliminated_above_set5": THESIS_EVENTS_ELIMINATED_ABOVE_SET5,
            "nu_to_nu_grey_relation_statement": (
                "the n_grey -> nu relation is estimated by VENUS 4.12 with "
                "intranuclear cascading in the Centrality Detector "
                "simulation, with a systematic uncertainty of +-10% on nu "
                "(thesis section sec:nungreyrel, lines 1358-1377, Fig. "
                "fig-ncoll-nu); the most central bin ends at n_grey = 22, "
                "eliminating 260 events with a higher grey-proton count"),
        },
        "mapping_rule": MAPPING_RULE_TEXT,
        "mapping_table": mapping,
        "data_inventory": inventory,
        "merge_rule": MERGE_RULE_TEXT,
        "merge_fixture": merge_fixture_records(),
        "common_support": {
            "model_support_pin": {
                "seed": int(support_probe["seed"]),
                "n_events": int(support_probe["n_events"]),
                "band": list(support_probe["band"]),
                "xF_min": float(support_probe["xF_min"]),
                "xF_max": float(support_probe["xF_max"]),
                "pT_min": float(support_probe["pT_min"]),
                "pT_max": float(support_probe["pT_max"]),
            },
            "rule": COMMON_SUPPORT_RULE_TEXT,
            "per_set_registered_cells": {
                key: entry["n_registered_cells"]
                for key, entry in inventory.items()},
            "per_set_dropped_cells": {
                key: len(entry["dropped"]) for key, entry in inventory.items()},
        },
        "observable_definitions": OBSERVABLE_DEFINITIONS,
        "statistics": STATISTICS_BLOCK,
        "seed_protocol": derive_seed_block(),
        "verdict_bands": {
            "k_pass": K_PASS,
            "k_tension": K_TENSION,
            "rule": BAND_RULE_TEXT,
            "falsifier_directions": FALSIFIER_DIRECTIONS,
            "outcomes_policy": (
                "both outcomes and recorded failures are legitimate "
                "results (the M10 both-outcomes-acceptable rule); a "
                "recorded failure drives the P3 wording and does not block "
                "the land gate"),
        },
        "cost_plan_and_probe_schedule": COST_PLAN_TEXT,
        "regression_pins": REGRESSION_PINS_TEXT,
        "internal_consistency": INTERNAL_CONSISTENCY_TEXT,
        "model_support_probe": support_probe,
        "harness": HARNESS_TEXT,
        "provenance": {
            "configuration_bundled_canonical_sha256":
                BUNDLED_CONFIG_CANONICAL_SHA256,
            "configuration_bundled_path": BUNDLED_CONFIG_RELPATH,
            "m12_preregistration_path": M12_PREREG_RELPATH,
            "m12_preregistration_sha256": M12_PREREG_SHA256,
            "m8_frozen_block_sha256": M8_FROZEN_BLOCK_SHA256,
            "m8_record_path": M8_RECORD_RELPATH,
            "m8_record_sha256": M8_RECORD_SHA256,
            "nested_head_at_preregistration": _git_head(MIXING2006_DIR),
            "outer_head_at_preregistration": _git_head(REPO_ROOT),
        },
        "written": _utc_now_iso(),
    }
    return prereg


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-preregistration", action="store_true",
                        help="write the pre-registration JSON into the run "
                             "directory (refuses to overwrite without --force)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing pre-registration file")
    parser.add_argument("--cross-check", action="store_true",
                        help="run the pre-generation harness cross-check and "
                             "record harness_cross_check.json in the run "
                             "directory")
    parser.add_argument("--probe-determinism", action="store_true",
                        help="P2: five bit-identical repeated evaluations at "
                             "a fixed point (the sliced builder path, the "
                             "min-bias builder path, the p+p reference "
                             "generator)")
    parser.add_argument("--probe-cost", action="store_true",
                        help="P2: per-slice cost cells (300 events, sets 1..5 "
                             "in order) and the statistics-menu decision")
    parser.add_argument("--probe-acceptance", action="store_true",
                        help="P2: per-slice acceptance sanity at pilot scale "
                             "(10000 accepted events per slice)")
    parser.add_argument("--probe-gateway", action="store_true",
                        help="P2: gateway capacity probe (read-only)")
    parser.add_argument("--pilot", action="store_true",
                        help="P2: the full pilot (the P1 regression pins "
                             "re-asserted, the model-support probe "
                             "reproduction, and the four probes); refuses on "
                             "a missed expectation")
    parser.add_argument("--build-manifest", action="store_true",
                        help="P2: build the cells manifest (the registered "
                             "cell sets, the node plans, the seeds, and the "
                             "menu outcome) in the run directory")
    parser.add_argument("--menu-decision", default="",
                        help="P2: the cost-probe artifact to embed in the "
                             "manifest (default: probe_cost_cells_p2.json in "
                             "the run directory)")
    parser.add_argument("--produce-slice-replica", nargs=2, default=None,
                        metavar=("SET", "SEED"),
                        help="P2: run one production slice replica of SET "
                             "(1..5) at SEED and write slice_replica_setN_"
                             "SEED.json (the gateway per-replica entry "
                             "point)")
    parser.add_argument("--produce-ppref-replica", type=int, default=None,
                        metavar="SEED",
                        help="P2: run one p+p reference replica at SEED and "
                             "write ppref_replica_SEED.json (the gateway "
                             "per-replica entry point)")
    parser.add_argument("--determinism-production", action="store_true",
                        help="P2: same-environment double generation at "
                             "production scale (bit-identical payloads)")
    parser.add_argument("--collect", action="store_true",
                        help="P2: collect the production artifacts into "
                             "centrality_report.json and REPORT.md (refuses "
                             "on any mismatch or failed check)")
    parser.add_argument("--gateway-manifest", default="",
                        help="gateway manifest JSON to record in the report "
                             "(default: <out-dir>/gateway_manifest_p2.json "
                             "when present)")
    parser.add_argument("--cells-manifest", default="",
                        help="cells manifest JSON (default: "
                             "<out-dir>/centrality_cells_manifest.json)")
    parser.add_argument("--n-events", type=int,
                        default=PRIMARY_EVENTS_PER_REPLICA,
                        help="events per replica (default: the pre-registered "
                             "primary statistics)")
    parser.add_argument("--m8-record", default="",
                        help="M8 record path override (the gateway "
                             "environment resolves the tree-local run root "
                             "differently)")
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
                     else out_dir / SLICE_MANIFEST_NAME)
    m8_override = Path(args.m8_record) if args.m8_record else None

    if args.write_preregistration:
        if prereg_path.exists() and not args.force:
            raise SystemExit(
                f"pre-registration already exists at {prereg_path}; refusing "
                "to overwrite without --force")
        prereg = build_pre_registration(bundled_config_path())
        out_dir.mkdir(parents=True, exist_ok=True)
        prereg_path.write_text(
            json.dumps(_to_jsonable(prereg), indent=1, sort_keys=True) + "\n",
            encoding="utf-8")
        print(f"pre-registration written: {prereg_path}")
        print(f"sha256: {_sha256_file(prereg_path)}")
        return 0
    if args.cross_check:
        if not prereg_path.exists():
            raise SystemExit(f"pre-registration not found at {prereg_path}")
        record = run_harness_cross_check(
            prereg_path, out_dir / "harness_cross_check.json")
        print(json.dumps(record, indent=1, sort_keys=True))
        return 0
    if args.pilot or args.probe_determinism or args.probe_cost \
            or args.probe_acceptance or args.probe_gateway:
        out_dir.mkdir(parents=True, exist_ok=True)
        prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
        if args.pilot:
            run_pilot(out_dir, config_path, prereg)
            return 0
        if args.probe_determinism:
            run_probe_determinism(out_dir, config_path, prereg)
        if args.probe_cost:
            run_probe_cost_cells(out_dir, config_path, prereg)
        if args.probe_acceptance:
            run_probe_acceptance_sanity(out_dir, config_path, prereg)
        if args.probe_gateway:
            run_probe_gateway_capacity(out_dir, config_path)
        return 0
    if args.build_manifest:
        out_dir.mkdir(parents=True, exist_ok=True)
        menu = None
        menu_path = (Path(args.menu_decision) if args.menu_decision
                     else out_dir / "probe_cost_cells_p2.json")
        if menu_path.exists():
            cost = json.loads(menu_path.read_text(encoding="utf-8"))
            menu = cost.get("menu_decision") or None
        build_cells_manifest(config_path, prereg_path, out_dir,
                             menu_decision=menu)
        return 0
    if args.produce_slice_replica is not None:
        set_no, seed = args.produce_slice_replica
        if int(set_no) not in (1, 2, 3, 4, 5):
            raise SystemExit(f"unknown set {set_no!r}; use 1..5")
        out_dir.mkdir(parents=True, exist_ok=True)
        produce_slice_replica(
            int(set_no), int(seed), config_path, manifest_path,
            out_dir / f"slice_replica_set{int(set_no)}_{int(seed)}.json",
            n_events=int(args.n_events), m8_record=m8_override,
            preregistration=prereg_path)
        return 0
    if args.produce_ppref_replica is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        produce_ppref_replica(
            int(args.produce_ppref_replica), config_path, manifest_path,
            out_dir / f"ppref_replica_{int(args.produce_ppref_replica)}.json",
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
              else out_dir / GATEWAY_MANIFEST_NAME)
        collect_centrality_report(out_dir, config_path, prereg_path,
                                  gateway_manifest_path=gw,
                                  m8_record=m8_override)
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
