#!/usr/bin/env python3
"""Mass-law comparison helper (card TASK-20260905-physics-review, M10 phase A).

Pre-registered comparison of the two source-production mass laws of
``na49_mixing/exchange_model.py``:

- ``regge`` mode (lines 100-110): the Regge-informed law, a density in the
  squared-mass measure ``dM1^2 dM2^2`` with the sampled Jacobian
  ``4 M1 M2`` (M1 option A; ``tests/test_source_mass_measure.py``). Free
  parameters: ``regge.alpha_exc0`` (mass exponent, defined with respect to
  the squared-mass measure), ``regge.effective_transfer_slope`` (the single
  fittable transfer degree of freedom at one beam energy), ``regge.edge_power``.
- ``simple`` mode (lines 93-99): the flexible empirical law, a density in
  ``dM1 dM2`` by construction. Free parameters: ``simple.mass_scale``,
  ``simple.edge_power``, ``simple.transfer_slope`` (the simple law's
  transfer-slope degree of freedom through the shared transfer kernel
  ``exchange_model.transfer_slope``).

FORM 2 of the card (predictive validation): a pre-registered train/test
split on xF; each law's three free parameters are fitted under the identical
protocol (the M8 uncertainty-aware binned objective restricted to the 676
training bins, K = 8 fixed-seed replicas of 2000 events, bounded Powell
xtol/ftol 1e-3, the seven fitted continuum parameters frozen at the M8
calibration values per the M9 calibration-set decision of record), and the
held-out 341 bins are scored once at the fitted point. The primary metric
is the pooled held-out WRRMSE via the ``sps_core`` metrics. The M7
hypothesis spread is carried as the model uncertainty on the comparison:
the six arms are {regge, simple} x {H1, H2, H3} and the verdict is stated
at H3 with the cross-hypothesis range quoted.

The pre-registration
(``dev/00runs/mass_law_compare_2026-09-07/pre_registration.json``) is
authoritative: this harness reads it in every mode, cross-checks every
pre-registered constant against the module constants below, and refuses to
run on any mismatch. The verdict rule is implemented exactly as
pre-registered and is never tuned after a fit result exists.

No new sampler: the two existing mass laws are used as-is and no source
file is edited. The objective machinery is imported from
``na49_mixing.staged_evaluation`` (``_generate_pp``, ``_bin_tracks``,
``_replica_binned_spectra``, ``BinnedFitTable``, ``_resolve_binned_path``)
and the metrics from ``helpers.benchmarking.sps_core``. The only fitting
logic implemented here is the per-law arm configuration (including the
simple-mode parameter group, which the staged-evaluation fitter does not
support), the train-window restriction of the objective, and the
collection/verdict logic.

H1 declared reduction (pre-registered): the two H1 arms run at K = 8
replicas x E1 events per replica with a 2-iteration / 150-evaluation cap,
E1 chosen from the menu {200, 100, 50} by the pre-registered kernel-cost
probe (the largest E1 whose measured projected per-arm wall stays under
5 hours), and recorded in the pre-registration's ``h1_declared_reduction``
``chosen_E1`` field before any H1 arm fit. The H1 arms enter only the
cross-hypothesis spread, never the primary verdict at H3.

Layout: this file is byte-identical in the monorepo helper tree
(``helpers/benchmarking/``) and the vendored tree
(``dev/mixing2006/helpers/benchmarking/``); the tree is detected from
``__file__``, so the mixing2006 package and the tree-local run-artifact
root resolve correctly on both sides (monorepo: ``dev/00runs``; vendored:
``00runs``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# --- Layout (byte-identical in both helper trees) ---------------------------

_BENCH_DIR = Path(__file__).resolve().parent
if (_BENCH_DIR.parents[1] / "na49_mixing").is_dir():
    # vendored tree: dev/mixing2006/helpers/benchmarking/this_file.py
    MIXING2006_DIR = _BENCH_DIR.parents[1]
    RUNS_ROOT = MIXING2006_DIR / "00runs"
else:
    # monorepo tree: helpers/benchmarking/this_file.py
    REPO_ROOT = _BENCH_DIR.parents[1]
    MIXING2006_DIR = REPO_ROOT / "dev" / "mixing2006"
    RUNS_ROOT = REPO_ROOT / "dev" / "00runs"

if str(MIXING2006_DIR) not in sys.path:
    sys.path.insert(0, str(MIXING2006_DIR))

from helpers.benchmarking import sps_core  # noqa: E402

# --- Pre-registered protocol constants (mirrored in the pre-registration;
#     the run cross-checks the file against these and refuses a mismatch) ---

ARTIFACT_SCHEMA_VERSION = 1
CARD = "TASK-20260905-physics-review"
MILESTONE = "M10"
PHASE = "A"
RUN_DESCRIPTION = "mass_law_compare_2026-09-07"
ARTIFACT_NAME = "mass_law_compare_report.json"
STAGE_LABEL = "mass_law_compare_replica"
BEAM_MOMENTUM_GEV_C = 158.0
SQRT_S_GEV = 17.270205201372406  # sqrt(s) of Reaction(158.0, 2212, 2212)

CONFIG_SEED = 20260831
KERNEL_COST_PROBE_SEED = 20260831  # timing only (no fit result depends on it)

FORM_CHOICE = "form2_predictive_validation"
SPLIT_XF = 0.2
TRAIN_WINDOW = {"xf_min": 0.0, "xf_max": 0.2, "pt_min": 0.05, "pt_max": 1.5}
HELDOUT_WINDOW = {"xf_min": 0.2, "xf_max": 0.4, "pt_min": 0.05, "pt_max": 1.5}
BIN_COUNTS_TOTAL = {"train": 676, "heldout": 341, "sum": 1017}

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

SPECIES_ORDER = ("pi_minus", "pi_plus", "k_plus", "k_minus", "proton", "antiproton")
BIN_COUNTS = {
    "pi_minus": {"total": 229, "train": 158, "heldout": 71},
    "pi_plus": {"total": 230, "train": 158, "heldout": 72},
    "k_plus": {"total": 153, "train": 105, "heldout": 48},
    "k_minus": {"total": 148, "train": 105, "heldout": 43},
    "proton": {"total": 135, "train": 75, "heldout": 60},
    "antiproton": {"total": 122, "train": 75, "heldout": 47},
}

FREE_PARAMETERS = {
    "regge": ["regge.alpha_exc0", "regge.effective_transfer_slope", "regge.edge_power"],
    "simple": ["simple.mass_scale", "simple.edge_power", "simple.transfer_slope"],
}
BOUNDS = {
    "regge": {
        "regge.alpha_exc0": [0.0, 1.2],
        "regge.effective_transfer_slope": [5.07, 10.2],
        "regge.edge_power": [0.25, 6.0],
    },
    "simple": {
        "simple.mass_scale": [0.1, 5.0],
        "simple.edge_power": [0.25, 6.0],
        "simple.transfer_slope": [5.07, 10.2],
    },
}
STARTING_POINTS = {
    "regge": {
        "regge.alpha_exc0": 0.5,
        "regge.effective_transfer_slope": 7.3489827740435505,
        "regge.edge_power": 2.0,
    },
    "simple": {
        "simple.mass_scale": 0.65,
        "simple.edge_power": 1.5,
        "simple.transfer_slope": 5.5,
    },
}

EVENTS_PER_REPLICA_MAIN = 2000
K_REPLICAS = 8
XTOL = 1e-3
FTOL = 1e-3
CAP_MAX_ITERATIONS_MAIN = 10
N_PARAMS_FIT = 3

H1_MENU_EVENTS_PER_REPLICA = [200, 100, 50]
H1_MAX_ITERATIONS = 2
H1_MAX_EVALUATIONS = 150
H1_MAX_PROJECTED_ARM_WALL_S = 18000.0

M10_SEED_BASE = 20320831
ARM_SEED_STRIDE = 10
REPLICA_SEED_OFFSET = 1

ARM_ORDERING = [
    {"a": 0, "law": "regge", "hypothesis": "H1"},
    {"a": 1, "law": "regge", "hypothesis": "H2"},
    {"a": 2, "law": "regge", "hypothesis": "H3"},
    {"a": 3, "law": "simple", "hypothesis": "H1"},
    {"a": 4, "law": "simple", "hypothesis": "H2"},
    {"a": 5, "law": "simple", "hypothesis": "H3"},
]
ARM_SEEDS = {
    "regge_H1": 20320831,
    "regge_H2": 20320841,
    "regge_H3": 20320851,
    "simple_H1": 20320861,
    "simple_H2": 20320871,
    "simple_H3": 20320881,
}
REPLICA_SEED_BLOCKS = {
    "regge_H1": [20320832, 20320839],
    "regge_H2": [20320842, 20320849],
    "regge_H3": [20320852, 20320859],
    "simple_H1": [20320862, 20320869],
    "simple_H2": [20320872, 20320879],
    "simple_H3": [20320882, 20320889],
}

M8_RECORD_RELATIVE = ("staged_binned_fit_mc_2026-09-06", "staged_evaluation_report.json")
M8_RECORD_SHA256 = "ecd931d58ce79b9aecdd273c703b4421fe0ca9119cdde90d2ad3c7dfb9d49b65"

VERDICT_RULE = {
    "written_before_fitting": True,
    "primary_metric": "the pooled held-out WRRMSE per arm (see identical_protocol.heldout_metrics), evaluated at the arm's fitted point",
    "law_values_at_h3": "M_regge = the regge-H3 arm's pooled held-out WRRMSE; M_simple = the simple-H3 arm's pooled held-out WRRMSE; delta = M_simple - M_regge (delta > 0 means the regge law has the lower held-out WRRMSE)",
    "model_uncertainty": "band_L = the cross-hypothesis range of law L's pooled held-out WRRMSE, max minus min over {H1, H2, H3} (the M7 spread convention applied to the metric); combined band = sqrt(band_regge^2 + band_simple^2)",
    "prefer_regge": "delta > combined band: the data prefer the Regge-informed mass law",
    "prefer_simple": "delta < -combined band: the data prefer the flexible empirical mass law",
    "not_distinguished": "|delta| <= combined band: the data do not distinguish the two mass laws within the bands",
    "statement_convention": "the verdict is stated at H3 with the per-hypothesis values and the spread quoted; the H2/H3-only range (the identical-statistics subset) is quoted as a diagnostic alongside the full H1/H2/H3 range, with the H1 caveats of h1_declared_reduction",
    "both_outcomes_acceptable": "either a preference or a non-distinction is a legitimate recorded result; a preference for the simple law is recorded with the spec's Regge-informed framing stated and the reason for retaining (or switching) the Regge form",
}

H1_CAVEAT_TEXT = (
    "the H1 arms enter only the cross-hypothesis spread, never the primary "
    "verdict at H3. The H1 replica-mean MC sigma is sqrt(2000/E1) times the "
    "M8-statistics value (sigma_MC scales as 1/sqrt(events x K)), and the H1 "
    "fitted values are 2-iteration descents from the starting point, not "
    "converged fits (the M7 H1-cap honesty convention). Both caveats are "
    "quoted with the spread."
)


# --- Small utilities ---------------------------------------------------------


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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def load_config(config_path: Path) -> dict:
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def canonical_configuration_hash(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def m8_record_path() -> Path:
    return RUNS_ROOT / M8_RECORD_RELATIVE[0] / M8_RECORD_RELATIVE[1]


def arm_json_name(law: str, hypothesis: str) -> str:
    return f"arm_{law}_{hypothesis}.json"


def sqrt_s(beam_momentum: float = BEAM_MOMENTUM_GEV_C) -> float:
    from na49_mixing.reaction import Reaction

    value = math.sqrt(Reaction(beam_momentum, 2212, 2212).s())
    if abs(value - SQRT_S_GEV) > 1e-9:
        raise SystemExit(
            f"sqrt(s) mismatch: computed {value!r} against the pinned {SQRT_S_GEV!r}"
        )
    return value


# --- Parameter sets ----------------------------------------------------------


def _dataclass_from_dict(cls, values: dict | None):
    return cls(**(values or {}))


def build_base_parameter_set(config: dict):
    """Bundled-config parameter dataclasses with the ten M8 values applied.

    The M9 build-parameter-set convention: all other dataclass fields keep
    their bundled-config values, and the config declares no
    ``continuum.decay_hypothesis``, so the H3 kernel applies as the
    dataclass default.
    """
    from na49_mixing.continuum_channels import ContinuumParameters
    from na49_mixing.exchange_model import (
        ReggeExcitationParameters,
        SimpleExcitationParameters,
    )

    simple = _dataclass_from_dict(SimpleExcitationParameters, config.get("simple"))
    regge = _dataclass_from_dict(ReggeExcitationParameters, config.get("regge"))
    continuum = _dataclass_from_dict(ContinuumParameters, config.get("continuum"))
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
        "simple": {k: float(v) for k, v in simple.__dict__.items()},
    }
    for name, value in M8_PARAMETER_SET.items():
        group, field_name = name.split(".", 1)
        if applied[group][field_name] != float(value):
            raise SystemExit(f"parameter application failed for {name}")
    if str(applied["continuum"]["decay_hypothesis"]).strip().upper() != "H3":
        raise SystemExit("the H3 decay kernel must apply as the dataclass default")
    simple.validate()
    regge.validate()
    continuum.validate()
    return simple, regge, continuum, applied


def arm_parameters(law: str, hypothesis: str, config: dict) -> dict:
    """Build one arm's configuration: mode switch + frozen blocks + free block.

    Both arms start from the base parameter set (bundled config with the ten
    M8 values applied). Per arm, the fitted law's three free fields are then
    set to their pre-registered starting values and freed; the other law's
    block stays at the base parameter set (inert in that mode). The arm's
    decay hypothesis is applied to the continuum before the frozen record is
    snapshotted, so the frozen record reports the kernel the arm actually
    runs.
    """
    if law not in ("regge", "simple"):
        raise SystemExit(f"unknown law {law!r}")
    if hypothesis not in ("H1", "H2", "H3"):
        raise SystemExit(f"unknown hypothesis {hypothesis!r}")
    simple, regge, continuum, applied = build_base_parameter_set(config)
    # The M7 decay-hypothesis selector applies before the frozen-record
    # snapshot below, so the frozen continuum reports the kernel the arm
    # actually runs (not the dataclass default). The set_decay_hypothesis
    # call in the arm and probe runners re-applies the same value.
    continuum = replace(continuum, decay_hypothesis=hypothesis)
    continuum.validate()
    start = dict(STARTING_POINTS[law])
    fields = {name.split(".", 1)[1]: float(value) for name, value in start.items()}
    if law == "regge":
        regge = replace(regge, **fields)
        mode = "regge"
    else:
        simple = replace(simple, **fields)
        mode = "simple"
    simple.validate()
    regge.validate()
    frozen_record = {
        "simple": {k: float(v) for k, v in simple.__dict__.items()},
        "regge": {k: (None if v is None else float(v)) for k, v in regge.__dict__.items()},
        "continuum": {
            k: (v if isinstance(v, str) else float(v))
            for k, v in continuum.__dict__.items()
        },
    }
    return {
        "law": law,
        "hypothesis": hypothesis,
        "mode": mode,
        "simple": simple,
        "regge": regge,
        "continuum": continuum,
        "free_parameters": list(FREE_PARAMETERS[law]),
        "bounds": [tuple(BOUNDS[law][name]) for name in FREE_PARAMETERS[law]],
        "start": [float(STARTING_POINTS[law][name]) for name in FREE_PARAMETERS[law]],
        "frozen_record": frozen_record,
    }


def apply_parameter_vector(arm: dict, x):
    """Set the arm's free parameters to ``x`` (validate; frozen blocks stay)."""
    if len(x) != N_PARAMS_FIT:
        raise ValueError(f"expected {N_PARAMS_FIT} free parameters, got {len(x)}")
    fields = {
        name.split(".", 1)[1]: float(value)
        for name, value in zip(arm["free_parameters"], x)
    }
    if arm["law"] == "regge":
        regge = replace(arm["regge"], **fields)
        regge.validate()
        return arm["simple"], regge, arm["continuum"]
    simple = replace(arm["simple"], **fields)
    simple.validate()
    return simple, arm["regge"], arm["continuum"]


def set_decay_hypothesis(arm: dict, hypothesis: str) -> dict:
    """Set ``continuum.decay_hypothesis`` in memory (the M7 selector path)."""
    continuum = replace(arm["continuum"], decay_hypothesis=hypothesis)
    continuum.validate()
    arm = dict(arm)
    arm["continuum"] = continuum
    return arm


# --- Tables and the pre-registered split -------------------------------------


def build_tables(config: dict):
    """The M4 windowed observable set through the production table machinery."""
    from na49_mixing.staged_evaluation import BinnedFitTable, _resolve_binned_path

    binned_cfg = dict(config["fit"]["binned"])
    window_cfg = dict(binned_cfg.get("window", {}))
    window = {
        key: float(window_cfg[key])
        for key in ("xf_min", "xf_max", "pt_min", "pt_max")
        if key in window_cfg
    }
    sigma_inel_mb = float(binned_cfg.get("sigma_inel_mb", 31.78))
    path_root = Path(binned_cfg.get("path_root", MIXING2006_DIR))
    entries = list(binned_cfg.get("tables", []))
    if not entries:
        raise SystemExit("fit.binned.tables must name at least one table")
    return [
        BinnedFitTable.from_path(
            _resolve_binned_path(entry["path"], path_root),
            species=str(entry["species"]),
            pdg_id=int(entry["pdg"]),
            mass_gev=float(entry["mass_gev"]),
            normalization=str(entry.get("normalization", "f")),
            sigma_inel_mb=sigma_inel_mb,
            window=window or None,
        )
        for entry in entries
    ]


def split_masks(tables, split_xf: float = SPLIT_XF):
    """Per-table boolean masks: train (xF center < split) and held-out."""
    train_masks = []
    heldout_masks = []
    for table in tables:
        xf_point = table.xf_centers[table.bin_index_xf]
        train = xf_point < float(split_xf)
        train_masks.append(np.asarray(train, dtype=bool))
        heldout_masks.append(np.asarray(~train, dtype=bool))
    return train_masks, heldout_masks


def _assert_bin_counts(tables, train_masks, heldout_masks) -> None:
    """The split bin counts are pinned by the pre-registration."""
    per_species = {}
    total = {"train": 0, "heldout": 0, "sum": 0}
    for table, train_mask, heldout_mask in zip(tables, train_masks, heldout_masks):
        n_train = int(train_mask.sum())
        n_held = int(heldout_mask.sum())
        n_total = int(table.data.size)
        per_species[table.species] = {
            "total": n_total,
            "train": n_train,
            "heldout": n_held,
        }
        total["train"] += n_train
        total["heldout"] += n_held
        total["sum"] += n_total
    if per_species != BIN_COUNTS:
        raise SystemExit(
            f"train/held-out bin counts {per_species!r} do not match the "
            f"pre-registered pin {BIN_COUNTS!r}"
        )
    if total != BIN_COUNTS_TOTAL:
        raise SystemExit(
            f"total bin counts {total!r} do not match the pre-registered pin "
            f"{BIN_COUNTS_TOTAL!r}"
        )


# --- The uncertainty-aware objective on the training bins ---------------------


def evaluate_point(
    x,
    arm: dict,
    tables,
    train_masks,
    n_events: int,
    replica_seeds,
    beam: float,
):
    """One uncertainty-aware evaluation restricted to the training bins.

    K = len(replica_seeds) fixed-seed replicas of ``n_events`` events are
    generated at the trial parameters through the production
    ``_generate_pp`` path (common-random-number property preserved: the
    replica seeds are identical across evaluations), binned onto the fixed
    table binning through ``_bin_tracks``, and combined through
    ``_replica_binned_spectra``. The objective chi-square is the diagonal
    per-bin chi-square under the combined error
    ``sigma_i^2 = sigma_data,i^2 + sigma_MC,i^2`` summed over the training
    bins of the six species and nothing else. The held-out bins never enter.
    """
    from na49_mixing.staged_evaluation import (
        _bin_tracks,
        _generate_pp,
        _replica_binned_spectra,
    )

    simple, regge, continuum = apply_parameter_vector(arm, x)
    replica_spectra = []
    max_residual = 0.0
    for seed in replica_seeds:
        replica_events, summary = _generate_pp(
            int(n_events),
            int(seed),
            simple,
            regge,
            continuum,
            arm["mode"],
            float(beam),
            STAGE_LABEL,
        )
        max_residual = max(
            max_residual, float(summary["max_conservation_residual"])
        )
        replica_spectra.append(_bin_tracks(replica_events, tables, SQRT_S_GEV))
    m_mean, sigma_mc, sigma_replica = _replica_binned_spectra(replica_spectra, tables)
    chi2 = 0.0
    per_species: dict[str, float] = {}
    for table, train_mask in zip(tables, train_masks):
        sigma_combined = np.sqrt(table.sigma**2 + sigma_mc[table.species] ** 2)
        residual = (m_mean[table.species] - table.data) / sigma_combined
        chi2_table = float(np.dot(residual[train_mask], residual[train_mask]))
        chi2 += chi2_table
        per_species[table.species] = chi2_table
    return {
        "chi2": float(chi2),
        "per_species_train_chi2": per_species,
        "max_conservation_residual": max_residual,
        "_m_mean": m_mean,
        "_sigma_mc": sigma_mc,
        "_sigma_replica": sigma_replica,
    }


def heldout_metrics(m_mean, tables, heldout_masks) -> dict:
    """Pre-registered held-out scoring (replica-mean prediction, data sigma)."""
    per_species_wrrmse: dict[str, float] = {}
    per_species_chi2: dict[str, float] = {}
    per_species_bins: dict[str, int] = {}
    pooled_pred: list[np.ndarray] = []
    pooled_data: list[np.ndarray] = []
    pooled_sigma: list[np.ndarray] = []
    for table, heldout_mask in zip(tables, heldout_masks):
        pred = np.asarray(m_mean[table.species], dtype=float)[heldout_mask]
        data = table.data[heldout_mask]
        sigma = table.sigma[heldout_mask]
        per_species_wrrmse[table.species] = sps_core.relative_rmse_weighted(
            pred, data, sigma
        )
        residual = (pred - data) / np.maximum(sigma, 1e-30)
        per_species_chi2[table.species] = float(np.dot(residual, residual))
        per_species_bins[table.species] = int(heldout_mask.sum())
        pooled_pred.append(pred)
        pooled_data.append(data)
        pooled_sigma.append(sigma)
    pooled_wrrmse = sps_core.relative_rmse_weighted(
        np.concatenate(pooled_pred),
        np.concatenate(pooled_data),
        np.concatenate(pooled_sigma),
    )
    return {
        "per_species_wrrmse": per_species_wrrmse,
        "per_species_chi2_data_only": per_species_chi2,
        "per_species_bins": per_species_bins,
        "pooled": {
            "wrrmse": float(pooled_wrrmse),
            "n_bins": int(sum(per_species_bins.values())),
            "chi2_data_only": float(sum(per_species_chi2.values())),
        },
    }


def mc_sigma_summary(sigma_mc, sigma_replica, tables) -> dict:
    out = {}
    for table in tables:
        out[table.species] = {
            "mc_sigma_mean_over_bins": float(np.mean(sigma_mc[table.species])),
            "mc_sigma_max_over_bins": float(np.max(sigma_mc[table.species])),
            "per_replica_std_mean_over_bins": float(
                np.mean(sigma_replica[table.species])
            ),
        }
    return out


# --- Arm runner ---------------------------------------------------------------


def run_arm(
    law: str,
    hypothesis: str,
    out_dir: Path,
    config: dict,
    prereg: dict,
    resume: bool = False,
) -> dict:
    """Run one arm: a 3-parameter bounded-Powell fit plus the held-out scoring."""
    from scipy.optimize import minimize

    arm_name = f"{law}_{hypothesis}"
    if arm_name not in ARM_SEEDS:
        raise SystemExit(f"unknown arm {arm_name!r} (pre-registered grid)")
    arm_seed = ARM_SEEDS[arm_name]
    replica_seeds = [arm_seed + REPLICA_SEED_OFFSET + k for k in range(K_REPLICAS)]

    declared_reduction = None
    if hypothesis == "H1":
        chosen = prereg["h1_declared_reduction"].get("chosen_E1")
        if not isinstance(chosen, int) or chosen not in H1_MENU_EVENTS_PER_REPLICA:
            raise SystemExit(
                "the H1 declared reduction must be decided (chosen_E1 from the "
                "pre-registered menu) before any H1 arm fit; refusing to run"
            )
        n_events = int(chosen)
        maxiter = H1_MAX_ITERATIONS
        maxfev: int | None = H1_MAX_EVALUATIONS
        declared_reduction = {
            "events_per_replica": n_events,
            "max_iterations": maxiter,
            "max_evaluations": maxfev,
            "caveat": H1_CAVEAT_TEXT,
        }
    else:
        n_events = EVENTS_PER_REPLICA_MAIN
        maxiter = CAP_MAX_ITERATIONS_MAIN
        maxfev = None

    arm = arm_parameters(law, hypothesis, config)
    arm = set_decay_hypothesis(arm, hypothesis)
    tables = build_tables(config)
    train_masks, heldout_masks = split_masks(tables)
    _assert_bin_counts(tables, train_masks, heldout_masks)
    beam = float(config.get("beam_momentum", BEAM_MOMENTUM_GEV_C))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = out_dir / "scratch_traces"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    trace_path = scratch_dir / f"trace_{arm_name}.jsonl"

    trace: list[dict] = []
    restarts = 0
    prior_evaluations = 0
    if resume and trace_path.exists():
        loaded = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if loaded:
            trace = loaded
            prior_evaluations = len(loaded)
            restarts = 1

    options: dict = {"maxiter": maxiter, "xtol": XTOL, "ftol": FTOL}
    if maxfev is not None:
        remaining = maxfev - prior_evaluations
        if remaining <= 0:
            options = None
        else:
            options["maxfev"] = remaining

    def objective(x) -> float:
        t0 = time.perf_counter()
        res = evaluate_point(
            x, arm, tables, train_masks, n_events, replica_seeds, beam
        )
        wall = time.perf_counter() - t0
        entry = {
            "parameters": {
                name: float(value)
                for name, value in zip(arm["free_parameters"], x)
            },
            "chi2": res["chi2"],
            "per_species": res["per_species_train_chi2"],
            "wall_s": wall,
        }
        trace.append(entry)
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_to_jsonable(entry), sort_keys=True) + "\n")
        return res["chi2"]

    t_fit = time.perf_counter()
    result = None
    if options is not None:
        start_x = np.asarray(arm["start"], dtype=float)
        if restarts:
            best_prev = min(trace, key=lambda item: item["chi2"])
            start_x = np.asarray(
                [best_prev["parameters"][n] for n in arm["free_parameters"]],
                dtype=float,
            )
        result = minimize(
            objective,
            start_x,
            method="Powell",
            bounds=[tuple(b) for b in arm["bounds"]],
            options=options,
        )
    fit_wall = time.perf_counter() - t_fit

    if not trace:
        raise SystemExit(f"arm {arm_name}: no objective evaluations recorded")
    best = min(trace, key=lambda item: item["chi2"])
    best_x = np.asarray(
        [best["parameters"][n] for n in arm["free_parameters"]], dtype=float
    )
    t_final = time.perf_counter()
    final = evaluate_point(
        best_x, arm, tables, train_masks, n_events, replica_seeds, beam
    )
    final_wall = time.perf_counter() - t_final
    determinism_consistent = abs(final["chi2"] - best["chi2"]) <= 1e-9 * max(
        1.0, abs(best["chi2"])
    )
    heldout = heldout_metrics(final["_m_mean"], tables, heldout_masks)
    fitted = {
        name: float(value)
        for name, value in zip(arm["free_parameters"], best_x)
    }
    start_record = {
        name: float(value)
        for name, value in zip(arm["free_parameters"], arm["start"])
    }
    n_train = BIN_COUNTS_TOTAL["train"]

    payload = {
        "schema": f"staged_mass_law_compare_arm/v{ARTIFACT_SCHEMA_VERSION}",
        "card": CARD,
        "milestone": MILESTONE,
        "phase": PHASE,
        "run_description": RUN_DESCRIPTION,
        "arm": f"{law}:{hypothesis}",
        "law": law,
        "hypothesis": hypothesis,
        "status": "completed",
        "date_utc": _utc_now_iso(),
        "finished_utc": _utc_now_iso(),
        "seed_protocol": {
            "arm_seed": int(arm_seed),
            "replica_seeds": [int(s) for s in replica_seeds],
            "stage_label": STAGE_LABEL,
            "config_seed": CONFIG_SEED,
        },
        "statistics": {
            "k_replicas": int(K_REPLICAS),
            "events_per_replica": int(n_events),
            "events_per_evaluation": int(K_REPLICAS * n_events),
            "declared_reduction": declared_reduction,
        },
        "protocol": {
            "form": FORM_CHOICE,
            "split": {
                "train_window": dict(TRAIN_WINDOW),
                "heldout_window": dict(HELDOUT_WINDOW),
                "split_xf": float(SPLIT_XF),
                "n_train": int(n_train),
                "n_heldout": int(BIN_COUNTS_TOTAL["heldout"]),
                "per_species": BIN_COUNTS,
            },
            "objective": "M8 uncertainty-aware binned objective on the training bins (combined error sigma_data + standard error of the K-replica mean); the held-out bins never enter",
            "n_params_fit": int(N_PARAMS_FIT),
            "xtol": XTOL,
            "ftol": FTOL,
            "max_iterations": int(maxiter),
            "max_evaluations": maxfev,
        },
        "free_parameters": list(arm["free_parameters"]),
        "bounds": {
            name: list(BOUNDS[law][name]) for name in arm["free_parameters"]
        },
        "start_parameters": start_record,
        "fitted_parameters": fitted,
        "frozen_record": arm["frozen_record"],
        "fit": {
            "success": bool(result.success) if result is not None else False,
            "message": str(result.message) if result is not None else "resumed with exhausted evaluation budget; best of prior trace",
            "n_objective_evaluations": int(len(trace)),
            "prior_evaluations_from_restarts": int(prior_evaluations),
            "restarts": int(restarts),
            "train_chi2_combined_best": float(best["chi2"]),
            "train_chi2_dof": float(best["chi2"] / (n_train - N_PARAMS_FIT)),
            "final_check_chi2": float(final["chi2"]),
            "determinism_consistent": bool(determinism_consistent),
            "wall_s": float(fit_wall),
        },
        "final_evaluation": {
            "wall_s": float(final_wall),
            "train_chi2_combined": float(final["chi2"]),
            "per_species_train_chi2": final["per_species_train_chi2"],
            "heldout": heldout,
            "mc_sigma_summary": mc_sigma_summary(
                final["_sigma_mc"], final["_sigma_replica"], tables
            ),
            "max_conservation_residual": float(final["max_conservation_residual"]),
        },
        "config": {
            "beam_momentum_gev_c": float(beam),
            "sqrt_s_gev": SQRT_S_GEV,
            "configuration_hash": canonical_configuration_hash(config),
        },
        "provenance": {
            "outer_head": _git_head(MIXING2006_DIR.parents[1]),
            "nested_head": _git_head(MIXING2006_DIR),
            "m8_record_sha256": M8_RECORD_SHA256,
        },
    }
    _write_json(out_dir / arm_json_name(law, hypothesis), payload)
    held = heldout["pooled"]
    print(
        f"[arm {law}:{hypothesis}] pooled held-out WRRMSE {held['wrrmse']:.6f} "
        f"({held['n_bins']} bins); train chi2 {best['chi2']:.1f} at "
        f"{len(trace)} evaluations; {fit_wall + final_wall:.1f} s",
        flush=True,
    )
    return payload


# --- Probes -------------------------------------------------------------------


def run_probe(
    law: str,
    start_kind: str,
    n_evaluations: int,
    out_dir: Path,
    config: dict,
) -> dict:
    """Pre-registered cost probe A/B at a fixed point (H3 kernel, K = 8 x 2000)."""
    if law == "simple" and start_kind == "anchor":
        raise SystemExit(
            "the simple law has no M8 anchor; --probe-start bundled only"
        )
    arm = arm_parameters(law, "H3", config)
    arm = set_decay_hypothesis(arm, "H3")
    if law == "regge" and start_kind == "anchor":
        m8_fields = {
            name.split(".", 1)[1]: float(M8_PARAMETER_SET[name])
            for name in FREE_PARAMETERS["regge"]
        }
        arm["regge"] = replace(arm["regge"], **m8_fields)
        arm["regge"].validate()
        point = {name: float(M8_PARAMETER_SET[name]) for name in FREE_PARAMETERS["regge"]}
        label = "regge_anchor_m8"
    else:
        point = dict(STARTING_POINTS[law])
        label = f"{law}_bundled_start"
    tables = build_tables(config)
    train_masks, heldout_masks = split_masks(tables)
    _assert_bin_counts(tables, train_masks, heldout_masks)
    beam = float(config.get("beam_momentum", BEAM_MOMENTUM_GEV_C))
    x = np.asarray([point[n] for n in arm["free_parameters"]], dtype=float)
    chi2s = []
    walls = []
    residuals = []
    for _ in range(int(n_evaluations)):
        t0 = time.perf_counter()
        res = evaluate_point(
            x, arm, tables, train_masks, EVENTS_PER_REPLICA_MAIN,
            [CONFIG_SEED + 1 + k for k in range(K_REPLICAS)], beam,
        )
        walls.append(time.perf_counter() - t0)
        chi2s.append(res["chi2"])
        residuals.append(res["max_conservation_residual"])
    payload = {
        "schema": "staged_mass_law_compare_probe/v1",
        "probe": label,
        "law": law,
        "start_kind": start_kind,
        "n_evaluations": int(n_evaluations),
        "k_replicas": int(K_REPLICAS),
        "events_per_replica": int(EVENTS_PER_REPLICA_MAIN),
        "events_per_evaluation": int(K_REPLICAS * EVENTS_PER_REPLICA_MAIN),
        "point_parameters": point,
        "chi2_train_window": [float(v) for v in chi2s],
        "wall_s": [float(v) for v in walls],
        "mean_wall_s": float(np.mean(walls)),
        "deterministic": bool(len(set(chi2s)) == 1),
        "max_conservation_residual": float(max(residuals)),
    }
    _write_json(Path(out_dir) / f"probe_{label}.json", payload)
    print(
        f"[probe {label}] mean wall {payload['mean_wall_s']:.2f} s/evaluation; "
        f"chi2 {chi2s[0]:.1f}; deterministic={payload['deterministic']}",
        flush=True,
    )
    return payload


def run_kernel_cost_probe(n_events: int, out_dir: Path, config: dict) -> dict:
    """Pre-registered kernel-cost probe C: ms/event for the six mode x kernel cells.

    Times ``_generate_pp`` plus ``_bin_tracks`` (the per-replica work of the
    objective) at the regge-anchor parameter set, and applies the
    pre-registered H1 menu rule: the largest E1 in
    ``H1_MENU_EVENTS_PER_REPLICA`` whose projected per-arm wall
    (worst measured H1 seconds-per-evaluation x H1_MAX_EVALUATIONS) stays
    under ``H1_MAX_PROJECTED_ARM_WALL_S``.
    """
    from na49_mixing.staged_evaluation import _bin_tracks, _generate_pp

    simple, regge, continuum, _applied = build_base_parameter_set(config)
    tables = build_tables(config)
    beam = float(config.get("beam_momentum", BEAM_MOMENTUM_GEV_C))
    cells = {}
    for mode in ("regge", "simple"):
        for hypothesis in ("H1", "H2", "H3"):
            cont = replace(continuum, decay_hypothesis=hypothesis)
            t0 = time.perf_counter()
            events, _summary = _generate_pp(
                int(n_events), KERNEL_COST_PROBE_SEED, simple, regge, cont,
                mode, beam, STAGE_LABEL,
            )
            _bin_tracks(events, tables, SQRT_S_GEV)
            wall = time.perf_counter() - t0
            cells[f"{mode}_{hypothesis}"] = {
                "n_events": int(len(events)),
                "wall_s": float(wall),
                "ms_per_event": float(1000.0 * wall / max(len(events), 1)),
            }
    h1_worst = max(
        cells["regge_H1"]["ms_per_event"], cells["simple_H1"]["ms_per_event"]
    )
    menu_decision = None
    for e1 in H1_MENU_EVENTS_PER_REPLICA:  # ascending menu; keep the largest that fits
        projected = h1_worst * K_REPLICAS * e1 / 1000.0 * H1_MAX_EVALUATIONS
        if projected < H1_MAX_PROJECTED_ARM_WALL_S:
            menu_decision = {
                "chosen_E1": int(e1),
                "h1_worst_ms_per_event": float(h1_worst),
                "projected_arm_wall_s": float(projected),
                "rule": "largest E1 in the pre-registered menu whose projected per-arm wall (worst measured H1 s/evaluation x 150) stays under 18000 s",
            }
    payload = {
        "schema": "staged_mass_law_compare_kernel_cost/v1",
        "n_events_per_cell": int(n_events),
        "seed": int(KERNEL_COST_PROBE_SEED),
        "cells": cells,
        "h1_menu_decision": menu_decision,
    }
    _write_json(Path(out_dir) / "kernel_cost_probe.json", payload)
    for name, cell in cells.items():
        print(f"[kernel-cost {name}] {cell['ms_per_event']:.2f} ms/event", flush=True)
    if menu_decision is None:
        print("[kernel-cost] no E1 in the menu fits the 5-hour budget", flush=True)
    else:
        print(
            f"[kernel-cost] chosen_E1 = {menu_decision['chosen_E1']} "
            f"(projected per-arm wall {menu_decision['projected_arm_wall_s']:.0f} s)",
            flush=True,
        )
    return payload


# --- Verdict machinery (implements the pre-registered rule) -------------------


def compare_laws(values: dict) -> dict:
    """The pre-registered comparison: values[law][hypothesis] -> verdict.

    ``values`` maps each law to its per-hypothesis pooled held-out WRRMSE.
    delta = M_simple(H3) - M_regge(H3); band_L = max - min over {H1, H2, H3};
    combined band = sqrt(band_regge^2 + band_simple^2); the verdict follows
    ``VERDICT_RULE`` exactly.
    """
    m_regge = float(values["regge"]["H3"])
    m_simple = float(values["simple"]["H3"])
    delta = m_simple - m_regge
    band_regge = max(float(v) for v in values["regge"].values()) - min(
        float(v) for v in values["regge"].values()
    )
    band_simple = max(float(v) for v in values["simple"].values()) - min(
        float(v) for v in values["simple"].values()
    )
    combined = math.sqrt(band_regge**2 + band_simple**2)
    if delta > combined:
        verdict_key = "prefer_regge"
        verdict = "prefer_regge"
    elif delta < -combined:
        verdict_key = "prefer_simple"
        verdict = "prefer_simple"
    else:
        verdict_key = "not_distinguished"
        verdict = "not_distinguished"
    band_h2h3 = {
        law: max(float(values[law][h]) for h in ("H2", "H3"))
        - min(float(values[law][h]) for h in ("H2", "H3"))
        for law in ("regge", "simple")
    }
    return {
        "metric": VERDICT_RULE["primary_metric"],
        "law_values": {law: {h: float(v) for h, v in hyps.items()} for law, hyps in values.items()},
        "m_regge_h3": m_regge,
        "m_simple_h3": m_simple,
        "delta": float(delta),
        "band_regge": float(band_regge),
        "band_simple": float(band_simple),
        "combined_band": float(combined),
        "verdict_key": verdict_key,
        "verdict": verdict,
        "verdict_statement": VERDICT_RULE[verdict_key],
        "diagnostic_h2h3_only": {
            "band_regge": float(band_h2h3["regge"]),
            "band_simple": float(band_h2h3["simple"]),
            "note": "the identical-statistics subset (H2 and H3 run at the pre-registered main statistics); the full-range bands carry the H1 declared-reduction caveats",
        },
        "h1_caveat": H1_CAVEAT_TEXT,
    }


# --- Collection ----------------------------------------------------------------


def collect(out_dir: Path, config: dict, prereg: dict, execution_note: str = "") -> dict:
    """Merge the six arm artifacts; write the merged artifact only if all completed."""
    out_dir = Path(out_dir)
    arms: dict[str, dict] = {}
    missing: list[str] = []
    for item in ARM_ORDERING:
        name = f"{item['law']}_{item['hypothesis']}"
        path = out_dir / arm_json_name(item["law"], item["hypothesis"])
        if not path.exists():
            missing.append(str(path))
            continue
        arms[name] = json.loads(path.read_text(encoding="utf-8"))
    if missing:
        raise SystemExit(
            "refusing to write the artifact before all six arms are present "
            "and completed; missing: " + ", ".join(missing)
        )
    for name, payload in arms.items():
        if payload.get("status") != "completed":
            raise SystemExit(f"arm {name} is not completed; refusing to collect")
        expected_seed = ARM_SEEDS[name]
        if int(payload["seed_protocol"]["arm_seed"]) != expected_seed:
            raise SystemExit(f"arm {name} seed mismatch against the pre-registered grid")
        expected_replicas = [
            expected_seed + REPLICA_SEED_OFFSET + k for k in range(K_REPLICAS)
        ]
        if [int(s) for s in payload["seed_protocol"]["replica_seeds"]] != expected_replicas:
            raise SystemExit(f"arm {name} replica seeds mismatch against the pre-registered grid")
        heldout_bins = int(payload["final_evaluation"]["heldout"]["pooled"]["n_bins"])
        if heldout_bins != BIN_COUNTS_TOTAL["heldout"]:
            raise SystemExit(f"arm {name} held-out bin count {heldout_bins} != pre-registered {BIN_COUNTS_TOTAL['heldout']}")
        if not payload["fit"]["determinism_consistent"]:
            raise SystemExit(f"arm {name} final-evaluation determinism check failed")

    values = {
        law: {
            hyp: float(
                arms[f"{law}_{hyp}"]["final_evaluation"]["heldout"]["pooled"]["wrrmse"]
            )
            for hyp in ("H1", "H2", "H3")
        }
        for law in ("regge", "simple")
    }
    comparison = compare_laws(values)

    arm_summaries = {}
    for item in ARM_ORDERING:
        name = f"{item['law']}_{item['hypothesis']}"
        payload = arms[name]
        path = out_dir / arm_json_name(item["law"], item["hypothesis"])
        arm_summaries[name] = {
            "arm": payload["arm"],
            "status": payload["status"],
            "artifact": str(path),
            "artifact_sha256": _sha256_file(path),
            "arm_seed": payload["seed_protocol"]["arm_seed"],
            "replica_seeds": payload["seed_protocol"]["replica_seeds"],
            "statistics": payload["statistics"],
            "free_parameters": payload["free_parameters"],
            "start_parameters": payload["start_parameters"],
            "fitted_parameters": payload["fitted_parameters"],
            "fit": payload["fit"],
            "heldout_pooled": payload["final_evaluation"]["heldout"]["pooled"],
            "heldout_per_species_wrrmse": payload["final_evaluation"]["heldout"]["per_species_wrrmse"],
        }

    gateway_manifest_path = out_dir / "gateway_manifest.json"
    if gateway_manifest_path.exists():
        execution = {
            "mode": "gateway_grid",
            "manifest": json.loads(gateway_manifest_path.read_text(encoding="utf-8")),
        }
    else:
        execution = {"mode": "serial_local", "manifest": None}
    execution["note"] = execution_note
    execution["arm_wall_s"] = {
        name: float(arms[name]["fit"]["wall_s"])
        + float(arms[name]["final_evaluation"]["wall_s"])
        for name in arm_summaries
    }

    deviations = list(prereg.get("deviations", []))
    for name, payload in arms.items():
        if int(payload["fit"]["restarts"]):
            deviations.append(
                f"{name}: {payload['fit']['restarts']} restart(s) from the "
                f"pre-registered scratch-trace restart rule "
                f"({payload['fit']['prior_evaluations_from_restarts']} prior "
                "evaluations carried over)"
            )

    prereg_path = out_dir / "pre_registration.json"
    artifact = {
        "schema": f"staged_mass_law_compare/v{ARTIFACT_SCHEMA_VERSION}",
        "card": CARD,
        "milestone": MILESTONE,
        "phase": PHASE,
        "run_description": RUN_DESCRIPTION,
        "date_utc": _utc_now_iso(),
        "pre_registration": {
            "path": str(prereg_path),
            "sha256": _sha256_file(prereg_path),
            "cross_check": "all pre-registered constants verified against the harness before every mode",
        },
        "form_of_record": {
            "choice": FORM_CHOICE,
            "rationale": prereg["form_of_record"]["rationale"],
        },
        "split_of_record": prereg["split_of_record"],
        "calibration_set_decision_of_record": prereg["identical_protocol"]["frozen_continuum_m8_values"],
        "identical_protocol": {
            "m8_fitted_parameters": dict(M8_PARAMETER_SET),
            "frozen_continuum_m8_values": prereg["identical_protocol"]["frozen_continuum_m8_values"]["values"],
            "free_parameters": FREE_PARAMETERS,
            "bounds": BOUNDS,
            "starting_points": STARTING_POINTS,
            "objective": prereg["identical_protocol"]["objective"],
            "k_replicas": K_REPLICAS,
            "events_per_replica_main": EVENTS_PER_REPLICA_MAIN,
            "sigma_inel_mb": 31.78,
            "optimizer": {
                "method": "bounded Powell",
                "xtol": XTOL,
                "ftol": FTOL,
                "cap_max_iterations_main": CAP_MAX_ITERATIONS_MAIN,
            },
            "heldout_metrics": prereg["identical_protocol"]["heldout_metrics"],
        },
        "measure_of_each_law": prereg["measure_of_each_law"],
        "seed_protocol": prereg["seed_protocol"],
        "arm_grid": prereg["arm_grid"],
        "h1_declared_reduction": prereg["h1_declared_reduction"],
        "verdict_rule": VERDICT_RULE,
        "arms": arm_summaries,
        "law_metrics": {
            "primary": "pooled held-out WRRMSE per arm",
            "values": values,
        },
        "comparison": comparison,
        "execution": execution,
        "provenance": {
            "outer_head": _git_head(MIXING2006_DIR.parents[1]),
            "nested_head": _git_head(MIXING2006_DIR),
            "config_path": str(Path(out_dir) / ".." / ".." / "mixing2006" / "config" / "staged_excitation.json"),
            "configuration_hash": canonical_configuration_hash(config),
            "m8_record": str(m8_record_path()),
            "m8_record_sha256": M8_RECORD_SHA256,
        },
        "deviations": deviations,
        "decision_rule": prereg["decision_rule"],
        "finished_utc": _utc_now_iso(),
    }
    _write_json(out_dir / ARTIFACT_NAME, artifact)
    c = comparison
    print(
        f"[compare] regge-H3 {c['m_regge_h3']:.6f} simple-H3 {c['m_simple_h3']:.6f} "
        f"delta {c['delta']:+.6f} combined band {c['combined_band']:.6f} "
        f"-> {c['verdict']}",
        flush=True,
    )
    print(f"[done] artifact: {out_dir / ARTIFACT_NAME}", flush=True)
    return artifact


def self_check(out_dir: Path, config: dict, prereg: dict) -> int:
    """Re-read the pre-registration against the recorded run (artifact self-check)."""
    out_dir = Path(out_dir)
    artifact_path = out_dir / ARTIFACT_NAME
    if not artifact_path.exists():
        raise SystemExit(f"artifact not found at {artifact_path}")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    problems: list[str] = []

    # Re-derive the comparison from the arm artifacts and compare.
    values = {
        law: {
            hyp: float(
                json.loads(
                    (out_dir / arm_json_name(law, hyp)).read_text(encoding="utf-8")
                )["final_evaluation"]["heldout"]["pooled"]["wrrmse"]
            )
            for hyp in ("H1", "H2", "H3")
        }
        for law in ("regge", "simple")
    }
    recomputed = compare_laws(values)
    if recomputed["verdict"] != artifact["comparison"]["verdict"]:
        problems.append(
            f"verdict drift: recomputed {recomputed['verdict']!r} against recorded "
            f"{artifact['comparison']['verdict']!r}"
        )
    if abs(recomputed["delta"] - float(artifact["comparison"]["delta"])) > 0.0:
        problems.append("delta drift between the arm artifacts and the merged artifact")
    if artifact["comparison"]["verdict_statement"] != VERDICT_RULE[recomputed["verdict_key"]]:
        problems.append("verdict statement does not match the pre-registered rule")
    # The artifact's protocol constants must match the harness constants.
    if artifact["form_of_record"]["choice"] != FORM_CHOICE:
        problems.append("form drift in the artifact")
    if artifact["identical_protocol"]["m8_fitted_parameters"] != M8_PARAMETER_SET:
        problems.append("M8 parameter set drift in the artifact")
    if artifact["verdict_rule"] != VERDICT_RULE:
        problems.append("verdict rule drift in the artifact")
    if artifact["seed_protocol"]["arm_seeds"] != ARM_SEEDS:
        problems.append("arm seeds drift in the artifact")
    # Every arm status and bin count.
    for item in ARM_ORDERING:
        name = f"{item['law']}_{item['hypothesis']}"
        summary = artifact["arms"][name]
        if summary["status"] != "completed":
            problems.append(f"{name} not completed")
        if summary["heldout_pooled"]["n_bins"] != BIN_COUNTS_TOTAL["heldout"]:
            problems.append(f"{name} held-out bin count drift")
    if problems:
        raise SystemExit(
            "artifact self-check FAILED:\n  " + "\n  ".join(problems)
        )
    print(
        f"[self-check] OK: artifact {artifact_path} matches the pre-registered "
        f"protocol and the six arm artifacts; verdict {recomputed['verdict']} "
        f"re-derived from the arms",
        flush=True,
    )
    return 0


# --- Pre-registration cross-check ----------------------------------------------


def cross_check_preregistration(prereg: dict, config: dict, m8_record=None) -> None:
    """Refuse to run when the pre-registration and the harness constants disagree."""
    problems: list[str] = []

    def check(name: str, expected, actual) -> None:
        if expected != actual:
            problems.append(f"{name}: pre-registration {expected!r} != harness {actual!r}")

    check("form_of_record", prereg["form_of_record"]["choice"], FORM_CHOICE)
    split = prereg["split_of_record"]
    check("train_window", split["train_window"], TRAIN_WINDOW)
    check("heldout_window", split["heldout_window"], HELDOUT_WINDOW)
    if float(split["train_window"]["xf_max"]) != SPLIT_XF:
        problems.append(
            f"split_xf: train window edge {split['train_window']['xf_max']!r} != harness {SPLIT_XF!r}"
        )
    check("bin_counts_per_species", split["bin_counts_measured"]["per_species"], BIN_COUNTS)
    check("bin_counts_total", split["bin_counts_measured"]["total"], BIN_COUNTS_TOTAL)
    proto = prereg["identical_protocol"]
    check("m8_fitted_parameters", proto["m8_fitted_parameters"]["values"], M8_PARAMETER_SET)
    check("frozen_continuum_m8_values", proto["frozen_continuum_m8_values"]["values"],
          {k: v for k, v in M8_PARAMETER_SET.items() if k.startswith("continuum.")})
    check("free_parameters", proto["free_parameters"], FREE_PARAMETERS)
    check("bounds_regge", proto["bounds"]["regge"], BOUNDS["regge"])
    check("bounds_simple", proto["bounds"]["simple"], BOUNDS["simple"])
    check("starting_points_regge", proto["starting_points"]["regge"], STARTING_POINTS["regge"])
    check("starting_points_simple", proto["starting_points"]["simple"], STARTING_POINTS["simple"])
    check("events_per_replica_main", proto["events_per_replica_main"], EVENTS_PER_REPLICA_MAIN)
    check("k_replicas", proto["k_replicas"], K_REPLICAS)
    optimizer = proto["optimizer"]
    check("xtol", optimizer["xtol"], XTOL)
    check("ftol", optimizer["ftol"], FTOL)
    check("cap_max_iterations_main", optimizer["cap_max_iterations_main"], CAP_MAX_ITERATIONS_MAIN)
    seeds = prereg["seed_protocol"]
    check("m10_seed_base", seeds["m10_seed_base"], M10_SEED_BASE)
    check("stage_label", seeds["stage_label"], STAGE_LABEL)
    check("arm_seeds", seeds["arm_seeds"], ARM_SEEDS)
    check("replica_seed_blocks", seeds["replica_seed_blocks"], REPLICA_SEED_BLOCKS)
    check("arm_ordering", prereg["arm_grid"]["ordering"], ARM_ORDERING)
    h1 = prereg["h1_declared_reduction"]
    check("h1_menu_events_per_replica", h1["menu_events_per_replica"], H1_MENU_EVENTS_PER_REPLICA)
    check("h1_max_iterations", h1["max_iterations"], H1_MAX_ITERATIONS)
    check("h1_max_evaluations", h1["max_evaluations"], H1_MAX_EVALUATIONS)
    check("h1_max_projected_arm_wall_s", h1["max_projected_arm_wall_s"], H1_MAX_PROJECTED_ARM_WALL_S)
    check("verdict_rule", prereg["verdict_rule"], VERDICT_RULE)
    provenance = prereg["provenance"]
    check("config_canonical_sha256", provenance["config_canonical_sha256"],
          canonical_configuration_hash(config))
    check("m8_record_sha256", provenance["m8_record_sha256"], M8_RECORD_SHA256)
    if m8_record is not None and Path(m8_record).exists():
        actual = _sha256_file(Path(m8_record))
        if actual != M8_RECORD_SHA256:
            problems.append(f"m8_record_file: sha256 {actual!r} != harness {M8_RECORD_SHA256!r}")
    if problems:
        raise SystemExit(
            "pre-registration mismatch; refusing to run (the pre-registered "
            "protocol is authoritative):\n  " + "\n  ".join(problems)
        )


# --- Main -----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--arm", metavar="LAW:HYPOTHESIS",
                      help="run one arm, e.g. regge:H3 (law in {regge, simple})")
    mode.add_argument("--probe", action="store_true",
                      help="cost probe A/B: fixed-point objective evaluations")
    mode.add_argument("--kernel-cost-probe", action="store_true",
                      help="kernel-cost probe C: ms/event for the six mode x kernel cells")
    mode.add_argument("--collect", action="store_true",
                      help="merge the six arm artifacts and write the comparison")
    mode.add_argument("--self-check", action="store_true",
                      help="re-read the pre-registration against the recorded run")
    parser.add_argument("--probe-law", choices=("regge", "simple"), default="regge")
    parser.add_argument("--probe-start", choices=("anchor", "bundled"), default="bundled")
    parser.add_argument("--probe-evaluations", type=int, default=5)
    parser.add_argument("--kernel-cost-events", type=int, default=300)
    parser.add_argument("--resume", action="store_true",
                        help="resume an interrupted arm from its scratch trace (the pre-registered restart rule)")
    parser.add_argument("--out-dir", default=str(RUNS_ROOT / RUN_DESCRIPTION))
    parser.add_argument("--config", default="",
                        help="staged excitation JSON config "
                             "(default: <mixing2006>/config/staged_excitation.json)")
    parser.add_argument("--preregistration", default="",
                        help="pre-registration JSON (default: <out-dir>/pre_registration.json)")
    parser.add_argument("--execution-note", default="",
                        help="free-text execution note recorded by --collect")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    config_path = Path(args.config) if args.config else MIXING2006_DIR / "config" / "staged_excitation.json"
    config = load_config(config_path)
    prereg_path = Path(args.preregistration) if args.preregistration else out_dir / "pre_registration.json"
    if not prereg_path.exists():
        raise SystemExit(
            f"pre-registration not found at {prereg_path}; the protocol must be "
            "pre-registered before any fit"
        )
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    cross_check_preregistration(prereg, config, m8_record_path())
    prereg_sha = _sha256_file(prereg_path)
    print(
        f"[prereg] {prereg_path} (sha256 {prereg_sha[:12]}...): constants verified",
        flush=True,
    )

    if args.arm:
        if ":" not in args.arm:
            raise SystemExit("--arm expects LAW:HYPOTHESIS, e.g. regge:H3")
        law, hypothesis = args.arm.split(":", 1)
        run_arm(law, hypothesis, out_dir, config, prereg, resume=args.resume)
        return 0
    if args.probe:
        run_probe(args.probe_law, args.probe_start, args.probe_evaluations,
                  out_dir, config)
        return 0
    if args.kernel_cost_probe:
        run_kernel_cost_probe(args.kernel_cost_events, out_dir, config)
        return 0
    if args.collect:
        collect(out_dir, config, prereg, execution_note=args.execution_note)
        return 0
    return self_check(out_dir, config, prereg)


if __name__ == "__main__":
    raise SystemExit(main())
