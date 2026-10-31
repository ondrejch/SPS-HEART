#!/usr/bin/env python3
"""R02 legacy p+A reference run driver (Round 1, card 11).

Runs the pre-registered legacy p+A reference benchmark R02 of plan section
6 and writes its acceptance record ``gate0_r02.json``.  The run evaluates
the frozen ``legacy_v1`` two-source model in the p+A configuration
(``PAEventBuilder`` with the ``Pb208`` ``NucleusSpec`` and the wounded-K
centrality slices) at the M8 ten-parameter set of card 1's manifest
against the six-species SPS p+Pb tables, with no parameter changes and no
fitting:

- centrality classes: ``minimum_bias`` (no slice) plus ``set_1`` ..
  ``set_5``, the five wounded-K slices of the M13 pre-registered
  E_BE-to-K mapping table (``dev/00runs/staged_centrality_sets_2026-09-11/
  pre_registration.json``, the ``staged_centrality_sets.py`` convention).
  The generation slice per set is the mapped band applied as
  ``PAExcitationConfig.target_wound_range``; the pre-registered
  ``e_be_per_class`` value of a set is the mapping's calibrated mean
  target-wound number ``<K>`` of that band.  The five bands and means are
  embedded below as canonical constants (the M13 table lives under the
  gitignored ``dev/00runs/`` artifact store) and are cross-checked
  against that file at runtime when it exists;
- per (class, replica) seed: ``replica_{i,k} = seed_base + 50000 + 100 * i
  + k`` with ``i`` the class index (0 = minimum_bias .. 5 = set_5) and
  ``k = 0..7`` (the declared T2 tier of 8 replicas x 100,000 events per
  class).  The offset base 50000 is distinct from R01's 40000, so the two
  runs draw on disjoint seed streams;
- per replica, one ``PAEventBuilder`` seeded
  ``numpy.random.default_rng(replica_seed)`` generates the events
  sequentially (the M13 ``generate_slice_events`` builder convention; the
  p+A builder carries the Glauber sampler and decay state, so the replica
  stream is one sequential generator).  Replica units are independent
  seeded streams, so the worker pool spreads whole replicas and the
  results are independent of the worker count;
- all completed replicas of a class are pooled into one binned prediction
  (the estimator's ``N_events`` is the total over the pooled replicas):
  ``M_bin = (1 / (pi * sqrt(s) * N_events)) * sum_events (E / pT) / (dxF *
  dpT)``, the invariant cross section per inelastic event, i.e. ``f /
  sigma_inel`` with ``sigma_inel = 31.78 mb`` (the M4/M8 convention), on
  the table binning of ``na49_mixing.staged_evaluation`` (the paper
  window ``xF in [0, 0.4]``, ``pT in [0.05, 1.5]``; species bin counts
  pi- 229, pi+ 230, K+ 153, K- 148, p 135, pbar 122, total 1017), with
  ``sqrt(s) = 17.270205201372406 GeV`` the NN sqrt(s) of
  ``Reaction(158.0, 2212, 2212)`` (the M13 p+A convention); the track
  xF and E are nucleon-nucleon CM values: the p+A builder emits
  laboratory-frame events, and each event is boosted to the NN CM
  (``event_in_nn_cm``) before binning (card
  TASK-20261005-pa-frame-nn-cm-xf, 2026-10-06; an R02 record binned
  before that date carries lab-frame xF and E/pT and is withdrawn);
- hemisphere yields (new in R02): the integrated ``f / sigma_inel`` per
  species over the three pre-registered xF regions, classified per final
  particle at the event level (the ``source_side`` track-field sign
  convention): ``projectile_side`` xF > 0.2, ``central`` 0.0 <= xF <= 0.2,
  ``target_side`` xF < 0.0; the yield is ``sum_tracks (E / pT) / (pi *
  sqrt(s) * N_events)`` over all pT > 0 tracks of the species (the
  paper-window pT cut does not apply; the hemisphere rule is the
  pre-registered xF split);
- per-bin statistics: the model prediction's statistical error follows
  the standard relative convention ``stat_error = value * sqrt(1/N_bin)``
  (N_bin the number of that species' tracks binned in the cell); a bin
  with ``N_bin == 0`` carries an infinite MC error, is serialized as
  the string sentinel ``"Infinity"`` (the R01 convention, RFC 8259
  safe; ``float()`` restores ``inf``), and is listed in
  ``failure_map_<class>.json`` under ``empty_bins`` instead of being
  dropped;
- residuals ``delta_over_sigma = (d - m) / sigma`` with ``d`` the SPS
  p+Pb table value (the ``helpers/benchmarking/pa_data/`` per-event
  invariant densities, the same ``f / sigma_inel`` units as the model
  bins), ``m`` the pooled model prediction, and ``sigma = sqrt(V_data_bin
  + V_mc_bin)`` the combined per-bin error of card 7 (diagonal in Round
  1).  A model bin matches a data cell when the class's published table
  covers the bin's (xF_mid, pT_mid); duplicated thesis pT labels are
  inverse-variance merged (the M13 merged-unique-cells convention) and
  the singular pT = 0.00 labels are excluded.  Bins without a matched
  data cell carry ``null`` residuals; a species with no published p+Pb
  table (K+, K-, p, pbar) is model-only; a class whose species all lack
  tables is recorded ``model_only`` with ``n_flagged 0``.  The
  normalization nuisance is recorded with ``eta = 0.0`` and the card-7
  width ``sigma_eta`` (read from
  ``config/v2_singlet/likelihood_nuisances.json``), scope
  ``all_six_tables``;
- the failure map flags cells with ``|delta_over_sigma| > 3.0``, ranked
  by ``|pull|`` descending (rank 1 = largest |pull|);
- per-species chi-square ``sum(delta_over_sigma**2)`` over the species'
  data-matched bins, degrees of freedom ``data-matched bins - 1``
  (recorded convention; R02 fits no parameter), written to the run
  manifest per class.

Verification before any generation: the config file sha256 must match
the pin recorded in card 1's manifest
(``config/v2_singlet/legacy_v1_manifest.json``) and, for the production
config, the pre-registration's ``config_hash``; the canonical sha256 of
the manifest's ``m8_frozen_pp_parameters`` block must match the
pre-registered ``m8_parameter_hash``.  The card's
``config/staged_excitation_pA.json`` CLI path resolves onto the bundled
staged-excitation config (``config/staged_excitation.json``, the same
bytes card 1 freezes and the M13 p+A convention consumes; the file
carries the pinned ``Pb208`` ``nucleus`` block and the ``pA`` runner
settings).  A mismatch refuses the run with a non-zero exit code before
a single event is generated.

Output files under ``--out``: ``pre_registration.json`` (written before
generation), one ``replica_<class>_<seed>.json`` per completed replica
(written as the replica completes, so incremental progress survives an
interruption and a re-run reuses the completed replicas),
``run_manifest.json`` (rewritten after each class completes; per-class
seed lists, replica records with ``event_hash_set_sha256``, model-only
flags, per-species chi2, hemisphere yields, and the tier record), and
per class ``bin_predictions_<class>.json``,
``hemisphere_yields_<class>.json``, ``covariance_<class>.json``,
``residuals_<class>.json``, and ``failure_map_<class>.json``.  With
``--seal``: the immutable-archive triple ``provenance.json`` (via the
``run_provenance.py`` convention), ``SHA256SUMS.txt`` (sha256sum -c
format, sorted, every regular file under the run directory except the
seal/provenance/acceptance files, which cannot carry their own digests),
and ``seal_manifest.json`` (records the payload files plus
``SHA256SUMS.txt``, ``provenance.json``, and ``gate0_r02.json``; it
cannot contain its own digest).  ``gate0_r02.json`` is assembled only
for the canonical production run (the production config, all six
classes, the declared T2 tier, the pre-registered seed base); its
``clean_environment_commands`` block is read from
``clean_environment_commands.json`` in the run directory when present.

Exit codes: 0 when every executed assertion holds, 2 when a
verification refusal stops the run before any generation (the refusal
text goes to stderr), 1 when a post-generation assertion or the gate
verdict fails.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import importlib
import json
import math
import multiprocessing
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Mapping, NoReturn

import numpy as np

# ---------------------------------------------------------------------------
# Tree location: this driver runs from the nested tree
# (dev/mixing2006/helpers/benchmarking/) and from its byte-identical
# monorepo mirror (helpers/benchmarking/).  Both must find the
# na49_mixing package and card 1's manifest.
# ---------------------------------------------------------------------------


def _locate_roots() -> tuple[Path, Path]:
    here = Path(__file__).resolve()
    nested = None
    for base in (here.parents[2], here.parents[2] / "dev" / "mixing2006"):
        if (base / "na49_mixing").is_dir():
            nested = base.resolve()
            break
    if nested is None:
        _refuse(
            "cannot locate the na49_mixing package next to "
            f"{here}; the driver runs from the nested tree "
            "(dev/mixing2006/helpers/benchmarking/) or its monorepo mirror "
            "(helpers/benchmarking/)"
        )
    mono = nested.parent.parent if nested.parent.name == "dev" else nested
    return nested, mono


def _refuse(message: str) -> NoReturn:
    """Print a refusal to stderr and exit with code 2 before generation."""
    print(f"staged_r02_reference: {message}", file=sys.stderr, flush=True)
    raise SystemExit(2)


NESTED_ROOT, MONO_ROOT = _locate_roots()
if str(NESTED_ROOT) not in sys.path:
    sys.path.insert(0, str(NESTED_ROOT))

from helpers.benchmarking import sps_core  # noqa: E402
from helpers.benchmarking.benchmark_sps_species import (  # noqa: E402
    RESONANCE_DECAY_LIB_VERSION,
)
from helpers.benchmarking.run_provenance import (  # noqa: E402
    RUN_PROVENANCE_FILENAME,
    collect_source_record,
    validate_run_provenance,
    write_run_provenance,
)
from helpers.benchmarking import staged_r01_reference as r01  # noqa: E402
from na49_mixing.continuum_channels import ContinuumParameters  # noqa: E402
from na49_mixing.event_builder_pA import (  # noqa: E402
    PAEventBuilder,
    PAExcitationConfig,
    event_in_nn_cm,
    nn_cm_p4,
)
from na49_mixing.exchange_model import (  # noqa: E402
    ReggeExcitationParameters,
    SimpleExcitationParameters,
)
from na49_mixing.glauber import NucleusSpec  # noqa: E402
from na49_mixing.reaction import Reaction  # noqa: E402
from na49_mixing.staged_evaluation import (  # noqa: E402
    BinnedFitTable,
    preflight_binned_tables,
)

# ---------------------------------------------------------------------------
# Pre-registered run identity (card TASK-20260911-r1-run-r02-legacy-pa-
# reference; the seed rule, tier, normalization, observable set, and
# hemisphere rule are verbatim from the card, the hashes are read off the
# frozen tree).
# ---------------------------------------------------------------------------

CARD = "TASK-20260911-r1-run-r02-legacy-pa-reference"
RUN_DESCRIPTION = "r02_legacy_pa_reference_2026-09-11"
GATE_RECORD_NAME = "gate0_r02.json"
GATE_CLAUSE = "archive_commands_tests_and_immutable_pa_reference"

PREREGISTRATION_NAME = "pre_registration.json"
RUN_MANIFEST_NAME = "run_manifest.json"
CLEAN_ENV_NAME = "clean_environment_commands.json"
SHA256SUMS_NAME = "SHA256SUMS.txt"
SEAL_NAME = "seal_manifest.json"
PROVENANCE_NAME = "provenance.json"

BIN_PREDICTIONS_TEMPLATE = "bin_predictions_{class}.json"
HEMISPHERE_TEMPLATE = "hemisphere_yields_{class}.json"
COVARIANCE_TEMPLATE = "covariance_{class}.json"
RESIDUALS_TEMPLATE = "residuals_{class}.json"
FAILURE_MAP_TEMPLATE = "failure_map_{class}.json"
# (The templates document the card's per-class output names; the name
# helpers below build the actual file names with f-strings because
# `class` is a reserved Python keyword and cannot be a keyword argument.)

STAGE_LABEL = "r02_reference"

SEED_BASE = 20260911
SEED_OFFSET_BASE = 50000
SEED_OFFSET_PER_CLASS = 100
T2_REPLICAS = 8
T2_EVENTS_PER_REPLICA = 100000
SIGMA_INEL_MB = 31.78

# The 1017-bin six-species observable set (paper window, M4/M8
# convention) — the same set as R01.
SPECIES_BIN_COUNTS = {
    "pi-": 229,
    "pi+": 230,
    "K+": 153,
    "K-": 148,
    "p": 135,
    "pbar": 122,
}
TOTAL_BINS = 1017
SPECIES_DISPLAY = {
    "pi_minus": "pi-",
    "pi_plus": "pi+",
    "k_plus": "K+",
    "k_minus": "K-",
    "proton": "p",
    "antiproton": "pbar",
}
PAPER_WINDOW = {"xF": [0.0, 0.4], "pT": [0.05, 1.5]}

PULL_THRESHOLD = 3.0
DEFAULT_WORKERS = 4

# Centrality classes and the M13 pre-registered E_BE-to-K mapping.
#
# Provenance: the mapped wound bands and the calibrated mean target-wound
# number <K> per set are the M13 pre-registered E_BE-to-K mapping table
# (card TASK-20260911-centrality-sets-1-5, pre-registration
# dev/00runs/staged_centrality_sets_2026-09-11/pre_registration.json,
# mapping_table.sets; the closed-form table is reproduced by
# helpers/benchmarking/staged_centrality_sets.py build_mapping_table()).
# The M13 table lives under the gitignored dev/00runs/ artifact store, so
# the five values are embedded here as canonical constants; the driver
# cross-checks them against that file at runtime when it exists.
CENTRALITY_CLASSES = (
    "minimum_bias",
    "set_1",
    "set_2",
    "set_3",
    "set_4",
    "set_5",
)
# The generation slice per set: PAExcitationConfig.target_wound_range
# (minimum_bias keeps the unconditioned draw, target_wound_range None).
WOUND_BANDS = {
    "minimum_bias": None,
    "set_1": (2, 4),
    "set_2": (4, 4),
    "set_3": (4, 7),
    "set_4": (6, 6),
    "set_5": (7, 7),
}
# The pre-registered e_be_per_class values: the M13 mapping's calibrated
# mean target-wound number <K>_band of each set's band (the E_BE-selected
# thesis set's mapped K; minimum_bias carries no E_BE selection -> null).
E_BE_K_BAND_MEAN = {
    "minimum_bias": None,
    "set_1": 2.9370128803339877,
    "set_2": 4.0,
    "set_3": 5.247809226460652,
    "set_4": 6.0,
    "set_5": 7.0,
}
# The measured thesis <nu> per set (Chvala 2006 Tab. 5.4), recorded next
# to the mapped band for provenance.
NU_MEASURED = {
    "minimum_bias": None,
    "set_1": 2.9,
    "set_2": 4.1,
    "set_3": 5.2,
    "set_4": 6.0,
    "set_5": 6.9,
}
M13_PREREG_RELPATH = (
    "dev/00runs/staged_centrality_sets_2026-09-11/pre_registration.json"
)

# The pinned Pb208 geometry (the M13 GEOMETRY_PB208 constants; the
# bundled config carries the identical nucleus block, and both are
# verified at runtime).
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

# The NN sqrt(s) of Reaction(158.0, 2212, 2212) — the M13 p+A convention
# (staged_centrality_sets.SQRT_S_GEV); re-derived from Reaction and
# cross-checked at runtime.
SQRT_S_NN_GEV_PIN = 17.270205201372406

# Hemisphere rule (pre-registered; the source_side track-field sign
# convention extended by the forward threshold 0.2).
HEMISPHERE_RULE = {
    "projectile_side": "xF > 0.2",
    "central": "0.0 <= xF <= 0.2",
    "target_side": "xF < 0.0",
}
HEMISPHERE_XF_FORWARD = 0.2
HEMISPHERE_CONVENTION = (
    "hemisphere yields are the integrated f / sigma_inel per species over "
    "the pre-registered xF regions, classified per final particle at the "
    "event level (xF = 2 pL / sqrt(s)): projectile_side xF > 0.2, central "
    "0.0 <= xF <= 0.2, target_side xF < 0.0; the yield is sum_tracks "
    "(E / pT) / (pi * sqrt(s) * N_events) over all pT > 0 tracks of the "
    "species (no paper-window pT cut; the hemisphere rule is the "
    "pre-registered xF split, not the 1017-bin window)"
)

# Frozen M8 p+p calibration set: verbatim copy of the manifest's
# frozen_fit_outputs.m8_frozen_pp_parameters block (combined-error optimum
# of dev/00runs/staged_binned_fit_mc_2026-09-06/, decay hypothesis H3).
# M8_PARAMETER_HASH is the sha256 of the canonical JSON
# (json.dumps(sort_keys=True, separators=(",", ":"))); the freeze report
# dev/00runs/r1_legacy_v1_freeze_2026-09-11/freeze_report.json records the
# same digest as m8_parameters_hash.
M8_FROZEN_PP_PARAMETERS = {
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
M8_PARAMETER_HASH = "358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1"

# Config pins from the legacy_v1 manifest source_tree_sha256 inventory.
# The card's CLI path config/staged_excitation_pA.json names the staged
# excitation p+A model; the bundled file carrying that model (the pinned
# Pb208 nucleus block plus the pA runner settings) is
# config/staged_excitation.json, and both CLI names resolve onto the same
# pinned bytes.  No separate p+A config file exists in the frozen tree;
# the M13 p+A convention consumes exactly these bytes.
PRODUCTION_CONFIG_NAME = "staged_excitation.json"
PA_CONFIG_NAME = "staged_excitation_pA.json"
PRODUCTION_CONFIG_RELPATH = "dev/mixing2006/config/staged_excitation.json"
PRODUCTION_CONFIG_SHA256 = (
    "154e41028fc68d63daf74af73ab812789ae11915fcdabb3b1a6d32deabf6693f"
)
SMOKE_CONFIG_NAME = "staged_excitation_smoke.json"
SMOKE_CONFIG_RELPATH = "dev/mixing2006/config/staged_excitation_smoke.json"
SMOKE_CONFIG_SHA256 = (
    "7d36656a128a42da9f0c2342680befe6c81b500c432df97be689d61131881035"
)
MANIFEST_RELPATH = "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json"
NUISANCES_RELPATH = "dev/mixing2006/config/v2_singlet/likelihood_nuisances.json"

TARGET = "Pb208"
UNAVAILABLE_TARGETS = ["C", "W"]

PA_EVENT_SEED_RULE = (
    "one PAEventBuilder per replica seeded "
    "numpy.random.default_rng(replica_seed); events drawn sequentially "
    "(event ids 0..n-1) from the builder's Glauber sampler and decay "
    "state (the M13 staged_centrality_sets.generate_slice_events "
    "convention); the stream is a pure function of the replica seed, and "
    "replica units are independent, so the worker count does not affect "
    "any generated event"
)

# Clean-environment command keys recorded verbatim in gate0_r02.json.
CLEAN_ENV_PYTEST_KEY = "cd dev/mixing2006 && python -m pytest"
CLEAN_ENV_SMOKE_KEY = (
    "na49-staged-excitation --config config/staged_excitation_smoke.json"
)

# Modules whose installed source defines the staged p+A generation
# behavior fingerprinted into provenance.json (salted with the
# RESONANCE_DECAY_LIB_VERSION cache salt).
FINGERPRINT_MODULES = (
    "na49_mixing.event_builder_pA",
    "na49_mixing.glauber",
    "na49_mixing.parallel",
    "na49_mixing.continuum_channels",
    "na49_mixing.exchange_model",
    "na49_mixing.excitation_kinematics",
    "na49_mixing.source_event",
    "na49_mixing.decays",
    "na49_mixing.particle",
    "na49_mixing.quantum_numbers",
    "na49_mixing.pdg_database",
    "na49_mixing.reaction",
    "na49_mixing.state_validation",
    "na49_mixing.event_conservation",
    "na49_mixing.staged_evaluation",
)

# SPS p+Pb data tables (helpers/benchmarking/pa_data/, the Chvala 2006
# thesis flat files): per-event invariant densities d(xF, pT) = f /
# sigma_inel, feed-down subtracted.  The model bins carry the same
# f / sigma_inel units, so data and model are directly comparable.  The
# minimum-bias class consumes the "All centrality triggered" density
# tables (thesis ictable.000), the five sets the centrality-selected
# density tables (thesis ictable.001..005).  The xsect_* f-tables and the
# ptintegrated_* files are not consumed by R02 (the density tables carry
# the estimator's convention; a different data convention files its own
# card).
PA_DATA_DIRNAME = "pa_data"
PA_DATA_FILES = {
    "minimum_bias": {
        "pi_minus": "pPb_dens_minbias_piminus.txt",
        "pi_plus": "pPb_dens_minbias_piplus.txt",
    },
    "set_1": {
        "pi_minus": "pPb_dens_set1_piminus.txt",
        "pi_plus": "pPb_dens_set1_piplus.txt",
    },
    "set_2": {
        "pi_minus": "pPb_dens_set2_piminus.txt",
        "pi_plus": "pPb_dens_set2_piplus.txt",
    },
    "set_3": {
        "pi_minus": "pPb_dens_set3_piminus.txt",
        "pi_plus": "pPb_dens_set3_piplus.txt",
    },
    "set_4": {
        "pi_minus": "pPb_dens_set4_piminus.txt",
        "pi_plus": "pPb_dens_set4_piplus.txt",
    },
    "set_5": {
        "pi_minus": "pPb_dens_set5_piminus.txt",
        "pi_plus": "pPb_dens_set5_piplus.txt",
    },
}
PA_DATA_CONVENTION = (
    "the data side is the SPS p+Pb per-event invariant density tables of "
    "Chvala 2006 (helpers/benchmarking/pa_data/), feed-down subtracted; "
    "d(xF, pT) = f / sigma_inel is the estimator's own convention, so "
    "data and model bins compare directly; species without a published "
    "p+Pb table (K+, K-, p, pbar) are model-only"
)
PA_DATA_MATCH_CONVENTION = (
    "a model bin matches a data cell when the bin's tabulated (xF_mid, "
    "pT_mid) equals the cell center (exact to 1e-9 after rounding); "
    "duplicated thesis pT labels are inverse-variance merged (the M13 "
    "merged-unique-cells convention) and the singular pT = 0.00 labels "
    "are excluded; model bins without a matched data cell carry null "
    "residuals"
)


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds"
    )


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(obj: Any) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats with JSON-safe string sentinels.

    RFC 8259 has no infinity or NaN literals.  Empty bins carry an
    infinite MC error by the committed convention (recorded, never
    dropped); the committed JSON encoding is the string sentinel
    ``"Infinity"`` (``"-Infinity"`` / ``"NaN"`` for the other
    non-finite values), which ``json.load`` returns as a string and
    ``float()`` restores to the float value.  Archives sealed before
    this change carry Python's bare ``Infinity`` token; both encodings
    round-trip through ``float()``.
    """
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0.0 else "-Infinity"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    # ``allow_nan=False`` makes the sentinel the only escape hatch: a
    # non-finite value that slips past ``_json_safe`` raises instead of
    # emitting a non-RFC 8259 token.
    Path(path).write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def bin_predictions_name(class_name: str) -> str:
    # The card names the per-class outputs bin_predictions_<class>.json
    # (str.format cannot take the reserved keyword `class`).
    return f"bin_predictions_{class_name}.json"


def hemisphere_yields_name(class_name: str) -> str:
    return f"hemisphere_yields_{class_name}.json"


def covariance_name(class_name: str) -> str:
    return f"covariance_{class_name}.json"


def residuals_name(class_name: str) -> str:
    return f"residuals_{class_name}.json"


def failure_map_name(class_name: str) -> str:
    return f"failure_map_{class_name}.json"


def replica_record_name(class_name: str, seed: int) -> str:
    return f"replica_{class_name}_{int(seed)}.json"


# ---------------------------------------------------------------------------
# Pre-registration and verification
# ---------------------------------------------------------------------------


def class_replica_seeds(seed_base: int, class_index: int, replicas: int) -> list[int]:
    """The per-class replica seeds: ``seed_base + 50000 + 100*i + k``."""
    base = (
        int(seed_base)
        + SEED_OFFSET_BASE
        + SEED_OFFSET_PER_CLASS * int(class_index)
    )
    return [base + k for k in range(int(replicas))]


def default_pre_registration() -> dict:
    """The R02 pre-registration payload, exactly as the card defines it."""
    return {
        "target": TARGET,
        "unavailable_targets": list(UNAVAILABLE_TARGETS),
        "centrality_classes": list(CENTRALITY_CLASSES),
        "e_be_per_class": {
            name: E_BE_K_BAND_MEAN[name] for name in CENTRALITY_CLASSES
        },
        "seed_base": SEED_BASE,
        "seed_offset_base": SEED_OFFSET_BASE,
        "seed_offset_per_class": SEED_OFFSET_PER_CLASS,
        "replicas_t2": T2_REPLICAS,
        "events_per_replica_t2": T2_EVENTS_PER_REPLICA,
        "config_hash": PRODUCTION_CONFIG_SHA256,
        "m8_parameter_hash": M8_PARAMETER_HASH,
        "sigma_inel_mb": SIGMA_INEL_MB,
        "hemisphere_rule": dict(HEMISPHERE_RULE),
        "tier": {
            "declared": "T2",
            "declared_replicas": T2_REPLICAS,
            "declared_events_per_replica": T2_EVENTS_PER_REPLICA,
        },
    }


def _config_pin(name: str) -> tuple[str, str]:
    """Map a bundled config file name to (pinned sha256, manifest key)."""
    if name in (PRODUCTION_CONFIG_NAME, PA_CONFIG_NAME):
        return PRODUCTION_CONFIG_SHA256, PRODUCTION_CONFIG_RELPATH
    if name == SMOKE_CONFIG_NAME:
        return SMOKE_CONFIG_SHA256, SMOKE_CONFIG_RELPATH
    _refuse(
        f"refusing config {name!r}; R02 consumes only the bundled "
        f"legacy_v1 staged-excitation configs ({PA_CONFIG_NAME!r} -> "
        f"{PRODUCTION_CONFIG_NAME!r}, {SMOKE_CONFIG_NAME!r})"
    )


def verify_config_and_m8(config_path: Path, pre_registration: dict) -> dict:
    """Verify the config sha256 and the M8 parameter hash against card 1.

    The config file bytes must hash to the manifest pin (and, for the
    production config, to the pre-registration's ``config_hash``).  The
    manifest's ``m8_frozen_pp_parameters`` block must hash canonically to
    the pre-registered ``m8_parameter_hash`` and equal the driver's
    verbatim copy.  Any mismatch raises ``SystemExit`` (non-zero exit)
    before generation starts.
    """
    path = Path(config_path)
    if not path.is_file():
        _refuse(f"config not found: {path}")
    name = path.name
    pinned_sha, manifest_key = _config_pin(name)
    actual_sha = sha256_file(path)

    problems: list[str] = []
    if pinned_sha == PRODUCTION_CONFIG_SHA256:
        registered = pre_registration.get("config_hash")
        if registered != actual_sha:
            problems.append(
                f"pre-registration config_hash {registered!r} != config file "
                f"sha256 {actual_sha!r} ({path})"
            )
    if actual_sha != pinned_sha:
        problems.append(
            f"config sha256 {actual_sha!r} != pinned {pinned_sha!r} "
            f"({manifest_key})"
        )

    manifest_path = NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json"
    if not manifest_path.is_file():
        _refuse(f"card 1 manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    m8_block = manifest.get("frozen_fit_outputs", {}).get(
        "m8_frozen_pp_parameters"
    )
    if m8_block is None:
        problems.append("manifest carries no frozen_fit_outputs."
                        "m8_frozen_pp_parameters block")
    else:
        block_sha = canonical_json_sha256(m8_block)
        if block_sha != M8_PARAMETER_HASH:
            problems.append(
                f"manifest m8_frozen_pp_parameters canonical sha256 "
                f"{block_sha!r} != pre-registered {M8_PARAMETER_HASH!r}"
            )
        if m8_block != r01.M8_FROZEN_PP_PARAMETERS:
            problems.append(
                "manifest m8_frozen_pp_parameters values differ from the "
                "driver's verbatim frozen copy"
            )
        registered_m8 = pre_registration.get("m8_parameter_hash")
        if registered_m8 != M8_PARAMETER_HASH:
            problems.append(
                f"pre-registration m8_parameter_hash {registered_m8!r} != "
                f"{M8_PARAMETER_HASH!r}"
            )
    inventory = manifest.get("source_tree_sha256", {})
    manifest_pin = inventory.get(manifest_key)
    if manifest_pin is None:
        problems.append(f"manifest does not pin {manifest_key}")
    elif manifest_pin != actual_sha:
        problems.append(
            f"manifest pin for {manifest_key} is {manifest_pin!r} but the "
            f"config file hashes to {actual_sha!r}"
        )

    if problems:
        _refuse(
            "config/M8 verification failed; refusing to run:\n  "
            + "\n  ".join(problems)
        )
    return {
        "config_path": str(path),
        "config_sha256": actual_sha,
        "manifest_path": str(manifest_path),
        "manifest_config_pin": manifest_pin,
        "m8_parameter_hash": M8_PARAMETER_HASH,
        "production_config": pinned_sha == PRODUCTION_CONFIG_SHA256,
        "config_name": name,
    }


def verify_centrality_constants() -> dict:
    """Cross-check the embedded M13 mapping constants when the table exists.

    The five wound bands and the calibrated mean target-wound numbers are
    embedded canonical constants (the M13 pre-registration lives under the
    gitignored dev/00runs/ artifact store).  When that file exists in the
    checkout, every band and every mapped <K> must match it exactly; a
    mismatch refuses the run.
    """
    path = MONO_ROOT / M13_PREREG_RELPATH
    if not path.is_file():
        return {
            "cross_checked": False,
            "path": str(path),
            "note": (
                "the M13 pre-registration file is absent; the embedded "
                "canonical constants stand alone"
            ),
        }
    prereg = json.loads(path.read_text(encoding="utf-8"))
    sets = (prereg.get("mapping_table") or {}).get("sets") or {}
    problems: list[str] = []
    for name in CENTRALITY_CLASSES:
        if name == "minimum_bias":
            continue
        set_no = name.split("_", 1)[1]
        record = sets.get(set_no)
        if record is None:
            problems.append(f"M13 mapping table carries no set {set_no}")
            continue
        band = WOUND_BANDS[name]
        if list(record.get("band", [])) != list(band):
            problems.append(
                f"{name}: embedded band {band!r} != M13 pre-registered "
                f"{record.get('band')!r}"
            )
        mean = E_BE_K_BAND_MEAN[name]
        if float(record.get("k_band_mean", float("nan"))) != float(mean):
            problems.append(
                f"{name}: embedded e_be (mapped <K>) {mean!r} != M13 "
                f"pre-registered k_band_mean {record.get('k_band_mean')!r}"
            )
    if problems:
        _refuse(
            "M13 centrality-mapping cross-check failed; refusing to run:\n  "
            + "\n  ".join(problems)
        )
    return {
        "cross_checked": True,
        "path": str(path),
        "sha256": sha256_file(path),
        "bands": {name: WOUND_BANDS[name] for name in CENTRALITY_CLASSES},
        "e_be_k_band_mean": {
            name: E_BE_K_BAND_MEAN[name] for name in CENTRALITY_CLASSES
        },
    }


def build_pa_config(raw_config: dict) -> tuple[PAExcitationConfig, dict]:
    """Parameter dataclasses, geometry, and the p+A config at the frozen M8 set.

    Mirrors the M13 ``build_pa_builder`` convention: the bundled config
    blocks build the dataclasses, each of the ten M8 values overrides its
    field, the application is asserted, the H3 decay kernel must hold as
    the dataclass default, and the ``nucleus`` block must be the pinned
    M13 Pb208 geometry (a different target or geometry files its own
    card).  ``target_wound_range`` is applied by the caller per class.
    """
    simple = SimpleExcitationParameters(**(raw_config.get("simple") or {}))
    regge = ReggeExcitationParameters(**(raw_config.get("regge") or {}))
    continuum = ContinuumParameters(**(raw_config.get("continuum") or {}))
    for name, value in M8_FROZEN_PP_PARAMETERS.items():
        group, field_name = name.split(".", 1)
        if group == "regge":
            regge = dataclasses.replace(regge, **{field_name: float(value)})
        else:
            continuum = dataclasses.replace(continuum, **{field_name: float(value)})
    applied = {
        "regge": {k: float(v) for k, v in regge.__dict__.items()},
        "continuum": {
            k: (v if isinstance(v, str) else float(v))
            for k, v in continuum.__dict__.items()
        },
    }
    for name, value in M8_FROZEN_PP_PARAMETERS.items():
        group, field_name = name.split(".", 1)
        if applied[group][field_name] != float(value):
            _refuse(f"parameter application failed for {name}")
    if str(applied["continuum"]["decay_hypothesis"]).strip().upper() != "H3":
        _refuse("the H3 decay kernel must apply as the dataclass default")
    spec = NucleusSpec(**(raw_config.get("nucleus") or {}))
    geometry = {
        "name": spec.name,
        "A": int(spec.A),
        "Z": int(spec.Z),
        "profile": str(spec.resolved_profile()),
        "radius_fm": float(spec.radius_fm),
        "diffuseness_fm": float(spec.diffuseness_fm),
        "quad_alpha": float(spec.quad_alpha),
        "sigma_nn_inel_mb": float(spec.sigma_nn_inel_mb),
    }
    for key, value in GEOMETRY_PB208.items():
        if geometry[key] != value:
            _refuse(
                f"config nucleus.{key} {geometry[key]!r} != the pinned M13 "
                f"geometry value {value!r}; a different target or geometry "
                "files its own card"
            )
    pa_settings = dict(raw_config.get("pA") or PA_RUNNER_SETTINGS)
    config = PAExcitationConfig(
        beam_momentum=float(raw_config.get("beam_momentum", 158.0)),
        production_mode="regge",
        simple=simple,
        regge=regge,
        continuum=continuum,
        nucleus=spec,
        glauber_dr_fm=float(pa_settings.get("glauber_dr_fm", 0.05)),
        max_event_attempts=int(pa_settings.get("max_event_attempts", 100)),
        target_wound_range=None,
    )
    return config, applied


def config_payload(raw_config: dict) -> dict:
    """Plain-dict projection of the config for worker processes."""
    return {
        "beam_momentum": float(raw_config.get("beam_momentum", 158.0)),
        "simple": dict(raw_config.get("simple") or {}),
        "regge": dict(raw_config.get("regge") or {}),
        "continuum": dict(raw_config.get("continuum") or {}),
        "nucleus": dict(raw_config.get("nucleus") or {}),
        "pA": dict(raw_config.get("pA") or {}),
    }


# ---------------------------------------------------------------------------
# Observable set: the 1017-bin six-species binned tables
# ---------------------------------------------------------------------------


def load_binned_tables(
    raw_config: dict, path_root: Path | None = None
) -> list[BinnedFitTable]:
    """Load the windowed six-species tables and verify the 1017-bin set.

    The tables come from the config's ``fit.binned`` block (paths resolved
    against the nested root, as in the M4/M8 machinery), the normalization
    must be the M4/M8 convention ``sigma_inel = 31.78 mb`` over the paper
    window, and the loaded per-species bin counts must equal the
    pre-registered observable set (pi- 229, pi+ 230, K+ 153, K- 148,
    p 135, pbar 122; total 1017) in the config's table order.  A deviation
    refuses the run: a different observable set files its own card.
    """
    binned = (raw_config.get("fit") or {}).get("binned") or {}
    if not binned:
        _refuse("config carries no fit.binned block; R02 bins the six-species set")
    sigma_inel = float(binned.get("sigma_inel_mb", 31.78))
    if abs(sigma_inel - SIGMA_INEL_MB) > 0.0:
        _refuse(
            f"fit.binned.sigma_inel_mb {sigma_inel!r} != the M4/M8 convention "
            f"{SIGMA_INEL_MB!r}; a different sigma_inel convention files its "
            "own card"
        )
    window = dict(binned.get("window") or {})
    expected_window = {
        "xf_min": sps_core.PAPER_XF_MIN,
        "xf_max": sps_core.PAPER_XF_MAX,
        "pt_min": sps_core.PAPER_PT_MIN,
        "pt_max": sps_core.PAPER_PT_MAX,
    }
    for key, value in expected_window.items():
        if float(window.get(key, value)) != float(value):
            _refuse(
                f"fit.binned.window.{key} {window.get(key)!r} != the paper "
                f"window value {value!r}; a different window files its own card"
            )
    root = Path(path_root) if path_root is not None else NESTED_ROOT
    preflight_binned_tables(binned, root)
    tables: list[BinnedFitTable] = []
    for entry in binned.get("tables", []):
        tables.append(
            BinnedFitTable.from_path(
                root / entry["path"],
                species=str(entry["species"]),
                pdg_id=int(entry["pdg"]),
                mass_gev=float(entry["mass_gev"]),
                normalization=str(entry["normalization"]),
                sigma_inel_mb=sigma_inel,
                window=window,
            )
        )
    registered_counts = list(SPECIES_BIN_COUNTS.items())
    if len(tables) != len(registered_counts):
        _refuse(
            f"config declares {len(tables)} binned tables; the pre-registered "
            f"observable set has {len(registered_counts)}"
        )
    for table, (display, count) in zip(tables, registered_counts):
        if int(table.data.size) != int(count):
            _refuse(
                f"species {table.species!r} ({display}) loads "
                f"{int(table.data.size)} windowed bins; the pre-registered "
                f"observable set declares {int(count)} (a different binning "
                "files its own card)"
            )
    return tables


def tables_digest(tables: list[BinnedFitTable]) -> str:
    """Digest binding the pooled binning to the exact loaded table state."""
    return r01.tables_digest(tables)


# ---------------------------------------------------------------------------
# Hemisphere yields (event-level, the pre-registered xF split)
# ---------------------------------------------------------------------------


def _hemisphere_slot(xf: float) -> int:
    """Region index of a track: 0 projectile (xF > 0.2), 1 central
    (0.0 <= xF <= 0.2), 2 target (xF < 0.0)."""
    if xf > HEMISPHERE_XF_FORWARD:
        return 0
    if xf >= 0.0:
        return 1
    return 2


def _hemisphere_track_sums(
    events: list,
    species_pdgs: Mapping[str, tuple[int, ...]],
    sqrt_s_gev: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per-species pooled hemisphere weight sums and track counts.

    For every final particle of a listed species with pT > 0, the Jacobian
    weight ``E / pT`` is accumulated into the particle's hemisphere
    (projectile xF > 0.2, central 0.0 <= xF <= 0.2, target xF < 0.0, with
    xF = 2 pL / sqrt(s)).  The pooled hemisphere yield is
    ``sum / (pi * sqrt(s) * N_events)`` — the integrated ``f / sigma_inel``
    of the region.  ``E`` and ``pL`` are the nucleon-nucleon CM values: a
    laboratory-frame p+A event (the builder's frame record) is read
    through ``nn_cm_p4``, and a CM-frame event is read as is.
    """
    sums = {species: np.zeros(3, dtype=float) for species in species_pdgs}
    counts = {
        species: np.zeros(3, dtype=np.int64) for species in species_pdgs
    }
    pdg_to_species: dict[int, str] = {}
    for species, pdgs in species_pdgs.items():
        for pdg in pdgs:
            pdg_to_species[int(pdg)] = species
    sqrt_s = float(sqrt_s_gev)
    for event in events:
        for particle in event.final_particles:
            species = pdg_to_species.get(int(particle.pdg_id))
            if species is None:
                continue
            p4 = nn_cm_p4(event, particle.p4)
            pt = math.hypot(p4.px, p4.py)
            if not (pt > 0.0):
                continue
            slot = _hemisphere_slot(2.0 * p4.pz / sqrt_s)
            sums[species][slot] += p4.e / pt
            counts[species][slot] += 1
    return sums, counts


def hemisphere_yields_payload(
    hemisphere_sums: Mapping[str, list | np.ndarray],
    n_events_total: int,
    sqrt_s_gev: float,
) -> list[dict]:
    """The per-species hemisphere yields in integrated ``f / sigma_inel``.

    Each entry carries exactly ``{"species", "projectile_side", "central",
    "target_side"}``; the yield of a region is the pooled Jacobian weight
    sum of the region divided by ``pi * sqrt(s) * N_events`` (the pooled
    event count over all replicas).
    """
    if int(n_events_total) <= 0:
        raise ValueError("the hemisphere yields need a positive event count")
    norm = math.pi * float(sqrt_s_gev) * float(n_events_total)
    entries: list[dict] = []
    for species in SPECIES_DISPLAY:
        if species not in hemisphere_sums:
            raise ValueError(f"hemisphere sums carry no species {species!r}")
        values = np.asarray(hemisphere_sums[species], dtype=float)
        if values.shape != (3,):
            raise ValueError(
                f"hemisphere sums for {species!r} must hold the three "
                f"regions, got shape {values.shape}"
            )
        entries.append(
            {
                "species": species,
                "projectile_side": float(values[0] / norm),
                "central": float(values[1] / norm),
                "target_side": float(values[2] / norm),
            }
        )
    return entries


# ---------------------------------------------------------------------------
# Generation (one seeded PAEventBuilder per replica, the M13 convention)
# ---------------------------------------------------------------------------


def _generate_replica(task: tuple) -> dict:
    """Generate one replica in a worker process and bin it.

    The p+A builder carries the Glauber sampler and decay state, so the
    replica's events are drawn sequentially from one seeded builder (the
    M13 ``generate_slice_events`` convention).  Returns the replica
    record: the pooled per-bin weight sums and track counts, the
    per-hemisphere weight sums and track counts, and the ordered
    ``event_hash`` digest list hashed into ``event_hash_set_sha256``.  No
    event payload crosses the process boundary.
    """
    (
        class_name,
        replica_index,
        seed,
        n_events,
        payload,
        tables,
        sqrt_s_gev,
        workers,
    ) = task
    pa_config, _applied = build_pa_config(payload)
    band = WOUND_BANDS[str(class_name)]
    if band is not None:
        pa_config = dataclasses.replace(
            pa_config, target_wound_range=(int(band[0]), int(band[1]))
        )
    builder = PAEventBuilder(pa_config, np.random.default_rng(int(seed)))
    species_pdgs = {t.species: (int(t.pdg_id),) for t in tables}
    hashes: list[str] = []
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    hemi_sums = {species: np.zeros(3, dtype=float) for species in species_pdgs}
    hemi_counts = {
        species: np.zeros(3, dtype=np.int64) for species in species_pdgs
    }
    wound_total = 0
    multiplicity_total = 0
    started = time.perf_counter()
    started_iso = _utc_now_iso()
    for event_id in range(int(n_events)):
        event = builder.generate(int(event_id))
        hashes.append(event.as_dict()["event_hash"])
        wound_total += int(event.metadata["target_wounds"])
        multiplicity_total += len(event.final_particles)
        # The builder emits laboratory-frame events; the binned x_F and the
        # E/pT weights are nucleon-nucleon CM quantities.
        event = event_in_nn_cm(event)
        chunk_sums, chunk_counts = r01._bin_track_sums([event], tables, sqrt_s_gev)
        for species, values in chunk_sums.items():
            sums[species] = (
                values.astype(float, copy=True)
                if species not in sums
                else sums[species] + values
            )
        for species, values in chunk_counts.items():
            counts[species] = (
                values.astype(np.int64, copy=True)
                if species not in counts
                else counts[species] + values
            )
        chunk_h_sums, chunk_h_counts = _hemisphere_track_sums(
            [event], species_pdgs, sqrt_s_gev
        )
        for species in species_pdgs:
            hemi_sums[species] += chunk_h_sums[species]
            hemi_counts[species] += chunk_h_counts[species]
    wall = time.perf_counter() - started
    finished_iso = _utc_now_iso()
    payload_digest = json.dumps(hashes, separators=(",", ":")).encode("utf-8")
    n_ev = int(n_events)
    return {
        "class": str(class_name),
        "replica_index": int(replica_index),
        "seed": int(seed),
        "band": [int(band[0]), int(band[1])] if band is not None else None,
        "n_events": n_ev,
        "worker_count": int(workers),
        "event_seed_rule": PA_EVENT_SEED_RULE,
        "event_hash_set_sha256": hashlib.sha256(payload_digest).hexdigest(),
        "config_sha256": None,  # stamped by the parent
        "tables_sha256": None,  # stamped by the parent
        "wall_seconds": round(wall, 3),
        "started_utc": started_iso,
        "finished_utc": finished_iso,
        "generation_attempts": int(builder.generation_attempts),
        "generation_accepted": int(builder.generation_accepted),
        "generation_rejected": int(builder.generation_rejected),
        "failure_reason_counts": dict(builder.failure_reason_counts),
        "mean_target_wounds": float(wound_total / n_ev) if n_ev else 0.0,
        "mean_final_multiplicity": (
            float(multiplicity_total / n_ev) if n_ev else 0.0
        ),
        "sums": {s: [float(v) for v in values] for s, values in sums.items()},
        "counts": {s: [int(v) for v in values] for s, values in counts.items()},
        "hemisphere_sums": {
            s: [float(v) for v in values] for s, values in hemi_sums.items()
        },
        "hemisphere_counts": {
            s: [int(v) for v in values] for s, values in hemi_counts.items()
        },
        "reused": False,
    }


def run_class_replicas(
    class_name: str,
    class_index: int,
    seeds: list[int],
    n_events: int,
    payload: dict,
    tables: list[BinnedFitTable],
    sqrt_s_gev: float,
    workers: int,
    out_dir: Path,
    config_sha256: str,
    table_digest: str,
) -> Iterator[dict]:
    """Yield one class's replica records, writing each record as it lands.

    Replica units are independent seeded streams, so the worker pool
    spreads whole replicas (the p+A builder is sequential inside a
    replica).  An existing ``replica_<class>_<seed>.json`` that matches
    the class, band, config hash, table digest, and requested event count
    is reused unchanged so the declared tier can be completed
    incrementally; anything else is regenerated.
    """
    out_dir = Path(out_dir)
    expected_band = (
        list(WOUND_BANDS[class_name]) if WOUND_BANDS[class_name] is not None else None
    )
    tasks: list[tuple] = []
    for replica_index, seed in enumerate(seeds):
        record_path = out_dir / replica_record_name(class_name, int(seed))
        stored = None
        if record_path.is_file():
            try:
                stored = json.loads(record_path.read_text(encoding="utf-8"))
            except ValueError:
                stored = None
        if (
            isinstance(stored, dict)
            and stored.get("class") == class_name
            and int(stored.get("seed", -1)) == int(seed)
            and int(stored.get("n_events", -1)) == int(n_events)
            and stored.get("config_sha256") == config_sha256
            and stored.get("tables_sha256") == table_digest
            and stored.get("band") == expected_band
            and isinstance(stored.get("sums"), dict)
            and isinstance(stored.get("counts"), dict)
            and isinstance(stored.get("hemisphere_sums"), dict)
        ):
            stored["reused"] = True
            print(
                f"[r02] {class_name} replica {seed}: reusing completed record "
                f"({int(stored['n_events'])} events)",
                flush=True,
            )
            yield stored
            continue
        tasks.append(
            (
                class_name,
                replica_index,
                int(seed),
                int(n_events),
                payload,
                tables,
                sqrt_s_gev,
                int(workers),
            )
        )
    if not tasks:
        return
    context = multiprocessing.get_context("fork")
    started = time.perf_counter()
    done = 0
    with context.Pool(processes=int(workers)) as pool:
        for record in pool.imap_unordered(_generate_replica, tasks, chunksize=1):
            done += 1
            record["config_sha256"] = config_sha256
            record["tables_sha256"] = table_digest
            _write_json(
                out_dir / replica_record_name(class_name, int(record["seed"])),
                record,
            )
            elapsed = time.perf_counter() - started
            print(
                f"[r02] {class_name} replica {record['seed']} completed "
                f"({done}/{len(tasks)} dispatched this invocation; "
                f"{elapsed:.1f}s; replica wall {record['wall_seconds']}s, "
                f"mean K {record['mean_target_wounds']:.3f})",
                flush=True,
            )
            yield record


# ---------------------------------------------------------------------------
# SPS p+Pb data tables (helpers/benchmarking/pa_data/)
# ---------------------------------------------------------------------------


def pa_data_dir() -> Path:
    return NESTED_ROOT / "helpers" / "benchmarking" / PA_DATA_DIRNAME


def _read_points(path: Path) -> np.ndarray:
    """Four-column (xF, pT, value, abs_error) spectra table (the
    ``staged_centrality_sets.read_points`` loader convention)."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rows.append([float(x) for x in stripped.split()[:4]])
    return np.asarray(rows, dtype=float)


def load_pa_data(class_name: str) -> dict:
    """The class's published p+Pb tables, per species.

    Returns ``{species: {"file", "sha256", "cells", "n_rows",
    "n_unique_cells", "n_pt_zero_cells"}}`` where ``cells`` maps the
    (xF, pT) cell center (rounded to 1e-9) to the inverse-variance-merged
    ``[value, error, n_rows]`` (the M13 merged-unique-cells convention for
    the duplicated thesis pT labels; the singular pT = 0.00 labels are
    excluded).  Species without a published table are absent from the
    mapping (model-only).
    """
    out: dict[str, dict] = {}
    base = pa_data_dir()
    for species, filename in PA_DATA_FILES.get(class_name, {}).items():
        path = base / filename
        if not path.is_file():
            continue
        points = _read_points(path)
        acc: dict[tuple[float, float], list[float]] = {}
        rows_per_cell: dict[tuple[float, float], int] = {}
        n_pt_zero = 0
        for xf, pt, value, err in points:
            if not (pt > 0.0):
                n_pt_zero += 1
                continue
            if not (err > 0.0):
                continue
            key = (round(float(xf), 9), round(float(pt), 9))
            weight = 1.0 / (float(err) * float(err))
            cell = acc.get(key)
            if cell is None:
                acc[key] = [weight * float(value), weight]
                rows_per_cell[key] = 1
            else:
                cell[0] += weight * float(value)
                cell[1] += weight
                rows_per_cell[key] += 1
        cells = {
            key: [float(wsum / wtot), float(1.0 / math.sqrt(wtot)), rows_per_cell[key]]
            for key, (wsum, wtot) in acc.items()
        }
        out[species] = {
            "file": filename,
            "sha256": sha256_file(path),
            "cells": cells,
            "n_rows": int(points.shape[0]),
            "n_unique_cells": len(cells),
            "n_pt_zero_cells": int(n_pt_zero),
        }
    return out


def _match_pa_cell(
    pa_species_data: dict | None, xf_mid: float, pt_mid: float
) -> list[float] | None:
    """The merged data cell at a model bin center, or None.

    The match key is the (xF_mid, pT_mid) pair rounded to 1e-9 — the
    thesis xF/pT grids coincide with the model bin centers on the
    overlapping nodes.
    """
    if not pa_species_data:
        return None
    key = (round(float(xf_mid), 9), round(float(pt_mid), 9))
    cell = pa_species_data["cells"].get(key)
    if cell is None:
        return None
    return [float(cell[0]), float(cell[1]), int(cell[2])]


# ---------------------------------------------------------------------------
# Per-class artifact builders (pure functions over plain arrays; unit-tested)
# ---------------------------------------------------------------------------


def residuals_payload(
    tables: list[BinnedFitTable],
    values_by_species: dict[str, np.ndarray],
    mc_error_by_species: dict[str, np.ndarray],
    pa_data: Mapping[str, dict],
) -> list[dict]:
    """Per-bin residuals ``(d - m) / sigma`` with the combined sigma.

    ``d`` is the class's published SPS p+Pb table value at the bin's
    (xF_mid, pT_mid) when the table covers it, else ``None``; ``sigma`` is
    ``sqrt(V_data_bin + V_mc_bin)`` with ``V_data_bin`` the matched cell's
    squared absolute error.  An empty MC bin under an existing data cell
    carries an infinite sigma and pull 0.0 (the R01 infinite-MC-error
    convention, recorded, never dropped); bins without a matched data
    cell carry all-null data fields.
    """
    entries: list[dict] = []
    for table in tables:
        mc_err = np.asarray(mc_error_by_species[table.species], dtype=float)
        m = np.asarray(values_by_species[table.species], dtype=float)
        species_data = pa_data.get(table.species)
        for i in range(table.data.size):
            matched = _match_pa_cell(
                species_data,
                float(table.xf_centers[table.bin_index_xf[i]]),
                float(table.pt_centers[table.bin_index_pt[i]]),
            )
            if matched is None:
                entries.append(
                    {
                        "species": table.species,
                        "bin_index": int(i),
                        "d": None,
                        "m": float(m[i]),
                        "sigma": None,
                        "delta_over_sigma": None,
                    }
                )
                continue
            d_value, d_err, _n_rows = matched
            sigma = math.sqrt(float(d_err) ** 2 + float(mc_err[i]) ** 2)
            if math.isinf(sigma):
                pull = 0.0
            else:
                pull = (float(d_value) - float(m[i])) / sigma
            entries.append(
                {
                    "species": table.species,
                    "bin_index": int(i),
                    "d": float(d_value),
                    "m": float(m[i]),
                    "sigma": float(sigma),
                    "delta_over_sigma": float(pull),
                }
            )
    return entries


def covariance_payload(
    tables: list[BinnedFitTable],
    mc_error_by_species: dict[str, np.ndarray],
    sigma_eta: float,
    pa_data: Mapping[str, dict],
) -> dict:
    """The card-7 combined covariance, diagonal in Round 1 (per class).

    ``V_data`` is the diagonal of the matched p+Pb cell's squared absolute
    error (0.0 where the class's table does not cover the bin), ``V_mc``
    the per-bin MC error squared, ``V_combined = V_data + V_mc``.  The
    normalization nuisance is recorded at its nominal ``eta = 0.0`` with
    the card-7 width ``sigma_eta`` and scope ``all_six_tables``.
    """
    data_entries: list[dict] = []
    mc_entries: list[dict] = []
    combined_entries: list[dict] = []
    for table in tables:
        mc_err = np.asarray(mc_error_by_species[table.species], dtype=float)
        species_data = pa_data.get(table.species)
        for i in range(table.data.size):
            matched = _match_pa_cell(
                species_data,
                float(table.xf_centers[table.bin_index_xf[i]]),
                float(table.pt_centers[table.bin_index_pt[i]]),
            )
            v_data = float(matched[1]) ** 2 if matched is not None else 0.0
            v_mc = float(mc_err[i]) * float(mc_err[i])
            data_entries.append(
                {
                    "species": table.species,
                    "bin_index": int(i),
                    "variance": float(v_data),
                }
            )
            mc_entries.append(
                {
                    "species": table.species,
                    "bin_index": int(i),
                    "variance": float(v_mc),
                }
            )
            combined_entries.append(
                {
                    "species": table.species,
                    "bin_index": int(i),
                    "variance": float(v_data + v_mc),
                }
            )
    return {
        "convention": {
            "V_data": (
                "diagonal; per-bin squared absolute error of the class's "
                "matched SPS p+Pb table cell (f / sigma_inel units); 0.0 "
                "where no p+Pb cell covers the bin (model-only)"
            ),
            "V_mc": (
                "diagonal; per-bin squared MC error (value * sqrt(1/N_bin), "
                "infinite for empty bins)"
            ),
            "V_combined": (
                "V_data + V_mc, per card 7's combined_covariance (diagonal "
                "in Round 1)"
            ),
            "empty_bins": (
                'carried as the string sentinel "Infinity" (RFC 8259 safe; '
                "float() restores inf), never dropped"
            ),
        },
        "V_data": data_entries,
        "normalization_nuisance": {
            "eta": 0.0,
            "sigma_eta": float(sigma_eta),
            "table_scope": "all_six_tables",
        },
        "V_mc": mc_entries,
        "V_combined": combined_entries,
    }


def build_failure_map(
    residuals: list[dict],
    counts_by_species: Mapping[str, list | np.ndarray],
    threshold: float = PULL_THRESHOLD,
) -> dict:
    """Flag cells with ``|delta_over_sigma| > threshold``, ranked by |pull|.

    Rank 1 is the largest |pull|; ties break to the config species order
    and then the lower bin index.  Bins with ``N_bin == 0`` are listed
    under ``empty_bins`` (species and bin index) and are never silently
    dropped; their pull is 0.0 by the infinite-MC-error convention, so
    they enter the map only through ``empty_bins``.  Model-only bins (no
    matched p+Pb cell) carry null pulls and are never flagged; a class
    whose species all lack published tables records ``model_only`` true
    with ``n_flagged 0`` — no data-side pulls exist.
    """
    counts_by_species = {
        species: np.asarray(values, dtype=np.int64)
        for species, values in counts_by_species.items()
    }
    flagged = [
        {
            "species": entry["species"],
            "bin_index": int(entry["bin_index"]),
            "pull": float(entry["delta_over_sigma"]),
        }
        for entry in residuals
        if entry["delta_over_sigma"] is not None
        and abs(float(entry["delta_over_sigma"])) > float(threshold)
    ]
    species_order = {name: i for i, name in enumerate(SPECIES_DISPLAY)}
    flagged.sort(
        key=lambda entry: (
            -abs(entry["pull"]),
            species_order.get(entry["species"], len(species_order)),
            entry["bin_index"],
        )
    )
    for rank, entry in enumerate(flagged, start=1):
        entry["rank"] = rank
    empty_bins = [
        {"species": entry["species"], "bin_index": int(entry["bin_index"])}
        for entry in residuals
        if int(counts_by_species[entry["species"]][int(entry["bin_index"])]) == 0
    ]
    data_species = sorted(
        {entry["species"] for entry in residuals if entry["d"] is not None}
    )
    model_only_species = [
        species for species in SPECIES_DISPLAY if species not in data_species
    ]
    return {
        "flagged_cells": flagged,
        "threshold": float(threshold),
        "n_flagged": len(flagged),
        "n_total_bins": TOTAL_BINS,
        "empty_bins": empty_bins,
        "n_empty_bins": len(empty_bins),
        "data_species": data_species,
        "model_only": len(data_species) == 0,
        "model_only_species": model_only_species,
        "convention": (
            "pull = (d - m) / sigma with sigma = sqrt(V_data_bin + V_mc_bin), "
            "V_data the matched SPS p+Pb cell error and V_mc the pooled MC "
            "error; a bin with N_bin == 0 carries an infinite MC error, its "
            "pull is 0.0, and it is recorded under empty_bins instead of "
            "being silently dropped; a species without a published p+Pb "
            "table is model-only (null residuals, never flagged); a class "
            "with no published table at all records model_only true and "
            "n_flagged 0 because no data-side pulls exist; bin_index is the "
            "0-based position of the bin in its species' windowed table order"
        ),
    }


CHI2_DOF_CONVENTION = "data-matched bins - 1 (card R02; R02 fits no parameter)"


def chi2_payload(residuals: list[dict]) -> dict:
    """Per-species chi-square over the species' data-matched bins.

    A species with no matched data (model-only) records a null chi2 with
    the reason instead of a number.
    """
    per_species: dict[str, dict] = {}
    total_chi2 = 0.0
    total_bins = 0
    for species in SPECIES_DISPLAY:
        entries = [e for e in residuals if e["species"] == species]
        if not entries:
            continue
        matched = [e for e in entries if e["d"] is not None]
        if not matched:
            per_species[species] = {
                "chi2": None,
                "dof": None,
                "chi2_dof": None,
                "n_bins": 0,
                "n_empty_bins": 0,
                "model_only": True,
                "reason": "no published SPS p+Pb table (model-only species)",
            }
            continue
        n_bins = len(matched)
        n_empty = sum(
            1 for e in matched if not math.isfinite(float(e["sigma"]))
        )
        chi2_value = float(
            sum(float(e["delta_over_sigma"]) ** 2 for e in matched)
        )
        dof = n_bins - 1
        per_species[species] = {
            "chi2": chi2_value,
            "dof": int(dof),
            "chi2_dof": float(chi2_value / dof) if dof > 0 else float("nan"),
            "n_bins": n_bins,
            "n_empty_bins": n_empty,
            "model_only": False,
        }
        total_chi2 += chi2_value
        total_bins += n_bins
    return {
        "dof_convention": CHI2_DOF_CONVENTION,
        "per_species": per_species,
        "total_chi2": float(total_chi2),
        "total_data_matched_bins": int(total_bins),
        "total_dof": int(total_bins - 1) if total_bins > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Provenance, seal, acceptance record
# ---------------------------------------------------------------------------


def staged_generation_code_fingerprint() -> str:
    """sha256 over the installed staged-generation sources plus the salt.

    Hash inputs, in order: canonical tagged source bytes of the modules
    in :data:`FINGERPRINT_MODULES`, the ``RESONANCE_DECAY_LIB_VERSION``
    salt, and this driver's own source bytes.
    """
    chunks: list[tuple[str, bytes]] = []
    for modname in FINGERPRINT_MODULES:
        mod = importlib.import_module(modname)
        origin = getattr(mod, "__file__", None)
        if not origin:
            raise RuntimeError(f"cannot resolve source file of {modname}")
        with open(origin, "rb") as handle:
            chunks.append((f"source:{modname}", handle.read()))
    chunks.append(
        (
            "salt:RESONANCE_DECAY_LIB_VERSION",
            RESONANCE_DECAY_LIB_VERSION.encode("utf-8"),
        )
    )
    chunks.append(("driver", Path(__file__).resolve().read_bytes()))
    digest = hashlib.sha256()
    for tag, payload in chunks:
        tag_b = tag.encode("utf-8")
        digest.update(len(tag_b).to_bytes(8, "big"))
        digest.update(tag_b)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def write_provenance(
    out_dir: Path,
    *,
    verification: dict,
    pre_registration: dict,
    seeds_per_class: Mapping[str, list[int]],
    replicas_per_class: Mapping[str, int],
    events_per_replica: int,
    events_completed: int,
    argv: list[str],
    wall_seconds: float,
    tier_status: str,
    tier_deferral_reason: str | None,
    input_paths: list[Path],
) -> dict:
    """Write ``provenance.json`` via the ``run_provenance.py`` convention.

    The record carries the config hash, the M8 parameter hash, the seed
    list per class, the E_BE (mapped K) values per class, the replica and
    event counts, and the code fingerprint (``RESONANCE_DECAY_LIB_VERSION``
    salt).  ``write_run_provenance`` writes ``run_provenance.json``; the
    file is renamed to the card's ``provenance.json`` name.
    """
    fingerprint = staged_generation_code_fingerprint()
    extra = {
        "card": CARD,
        "run": RUN_DESCRIPTION,
        "gate_clause": GATE_CLAUSE,
        "config_sha256": verification["config_sha256"],
        "config_name": verification["config_name"],
        "config_note": (
            "the card's config/staged_excitation_pA.json CLI path resolves "
            "onto the bundled legacy_v1 staged-excitation config "
            "(dev/mixing2006/config/staged_excitation.json), the same bytes "
            "card 1 freezes and the M13 p+A convention consumes"
        ),
        "m8_parameter_hash": verification["m8_parameter_hash"],
        "sigma_inel_mb": SIGMA_INEL_MB,
        "target": TARGET,
        "unavailable_targets": list(UNAVAILABLE_TARGETS),
        "centrality_classes": list(CENTRALITY_CLASSES),
        "wound_bands": {name: WOUND_BANDS[name] for name in CENTRALITY_CLASSES},
        "e_be_per_class": {
            name: E_BE_K_BAND_MEAN[name] for name in CENTRALITY_CLASSES
        },
        "e_be_provenance": (
            "the pre-registered e_be_per_class value of a set is the M13 "
            "E_BE-to-K mapping's calibrated mean target-wound number <K> of "
            "the set's band (mapping_table.sets[k].k_band_mean of "
            f"{M13_PREREG_RELPATH}); the generation slice is the band itself"
        ),
        "nu_measured_per_class": {
            name: NU_MEASURED[name] for name in CENTRALITY_CLASSES
        },
        "hemisphere_rule": dict(HEMISPHERE_RULE),
        "hemisphere_convention": HEMISPHERE_CONVENTION,
        "seed_base": int(pre_registration["seed_base"]),
        "seed_offset_base": int(pre_registration["seed_offset_base"]),
        "seed_offset_per_class": int(pre_registration["seed_offset_per_class"]),
        "replica_seeds_per_class": {
            name: [int(s) for s in seeds]
            for name, seeds in seeds_per_class.items()
        },
        "replicas_per_class_requested": {
            name: int(n) for name, n in replicas_per_class.items()
        },
        "events_per_replica": int(events_per_replica),
        "events_completed": int(events_completed),
        "tier": dict(pre_registration["tier"]),
        "tier_status": tier_status,
        "tier_deferral_reason": tier_deferral_reason,
        "observable_set": {
            "window": dict(PAPER_WINDOW),
            "species": dict(SPECIES_BIN_COUNTS),
            "total_bins": TOTAL_BINS,
        },
        "code_fingerprint": {
            "resonance_decay_lib_version": RESONANCE_DECAY_LIB_VERSION,
            "staged_generation_code_sha256": fingerprint,
            "fingerprint_modules": list(FINGERPRINT_MODULES),
            "fingerprint_convention": (
                "sha256 over tagged source bytes of the fingerprint "
                "modules, the RESONANCE_DECAY_LIB_VERSION salt, and this "
                "driver's own source"
            ),
        },
        "wall_seconds": round(float(wall_seconds), 3),
    }
    record = write_run_provenance(
        Path(out_dir),
        input_paths=[Path(p) for p in input_paths],
        argv=[str(a) for a in argv],
        env_var_names=["NA49_RUN_SOURCE_ARCHIVE"],
        rng={
            "type": PA_EVENT_SEED_RULE,
            "seeds": {
                "seed_base": int(pre_registration["seed_base"]),
                "seed_offset_base": int(pre_registration["seed_offset_base"]),
                "seed_offset_per_class": int(
                    pre_registration["seed_offset_per_class"]
                ),
                "replica_seeds_per_class": {
                    name: [int(s) for s in seeds]
                    for name, seeds in seeds_per_class.items()
                },
            },
        },
        source_record=collect_source_record(
            outer_repo=MONO_ROOT, nested_repo=NESTED_ROOT
        ),
        producer="staged_r02_reference.py",
        extra=extra,
    )
    problems = validate_run_provenance(Path(out_dir) / RUN_PROVENANCE_FILENAME)
    if problems:
        _refuse(
            "written provenance record failed validation:\n  "
            + "\n  ".join(problems)
        )
    written = Path(out_dir) / RUN_PROVENANCE_FILENAME
    target = Path(out_dir) / PROVENANCE_NAME
    if target.exists():
        target.unlink()
    shutil.move(str(written), str(target))
    return record


def seal_payload_files(out_dir: Path) -> list[Path]:
    """Every regular file under the run directory, sorted by relative path."""
    out_dir = Path(out_dir)
    files = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and not path.is_symlink():
            files.append(path)
    return files


def write_sha256sums(out_dir: Path) -> Path:
    """Write ``SHA256SUMS.txt`` over every payload file (sha256sum -c).

    The inventory covers every regular file under the run directory
    except the files that cannot carry their own digests: the inventory
    itself, ``seal_manifest.json``, ``provenance.json`` (recorded in the
    seal manifest instead), and ``gate0_r02.json`` (it carries the
    inventory's own digest).
    """
    out_dir = Path(out_dir)
    skip = {SHA256SUMS_NAME, SEAL_NAME, PROVENANCE_NAME, GATE_RECORD_NAME}
    lines = []
    for path in seal_payload_files(out_dir):
        if path.name in skip:
            continue
        rel = path.relative_to(out_dir).as_posix()
        lines.append(f"{sha256_file(path)}  {rel}\n")
    sums_path = out_dir / SHA256SUMS_NAME
    sums_path.write_text("".join(lines), encoding="utf-8", newline="\n")
    return sums_path


def write_seal_manifest(out_dir: Path, card: str = CARD) -> dict:
    """Write ``seal_manifest.json`` over every other file plus the sums.

    The seal manifest records every regular file under the run directory
    except itself (the payload files, ``SHA256SUMS.txt``,
    ``provenance.json``, and ``gate0_r02.json`` when present), each with
    its sha256 and byte size; it cannot contain its own digest.
    """
    out_dir = Path(out_dir)
    entries = []
    for path in seal_payload_files(out_dir):
        if path.name == SEAL_NAME:
            continue
        entries.append(
            {
                "path": path.relative_to(out_dir).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    seal = {
        "sealed_at_utc": _utc_now_iso(),
        "run_directory": out_dir.name,
        "card": card,
        "format": (
            "SHA256SUMS.txt covers every payload file in sha256sum -c "
            "format (sorted relative paths), excluding the files that "
            "cannot carry their own digests (SHA256SUMS.txt itself, "
            "seal_manifest.json, provenance.json, gate0_r02.json); "
            "seal_manifest.json records the payload files plus "
            "SHA256SUMS.txt, provenance.json, and gate0_r02.json; "
            "seal_manifest.json cannot contain its own digest"
        ),
        "files": entries,
    }
    _write_json(out_dir / SEAL_NAME, seal)
    return seal


def seal_run_dir(out_dir: Path) -> dict:
    """Seal the finished run directory (sums inventory plus seal manifest).

    ``provenance.json`` must already exist (write_provenance); the seal
    pair is written last and never cleaned.
    """
    out_dir = Path(out_dir)
    if not (out_dir / PROVENANCE_NAME).is_file():
        _refuse(f"provenance missing before sealing: {out_dir / PROVENANCE_NAME}")
    write_sha256sums(out_dir)
    return write_seal_manifest(out_dir)


def tier_status_decision(
    classes_completed: int,
    replicas_per_class_completed: int,
    declared_classes: int,
    declared_replicas: int,
) -> str:
    """Classify the tier counts against the card's strict acceptance rule.

    The card's acceptance output fixes the tier semantics: ``tier_status``
    is ``complete`` iff ``classes_completed == declared_classes`` AND
    ``replicas_per_class_completed == declared_replicas`` (the declared T2
    tier of 6 classes x 8 replicas); otherwise it is ``deferred`` with a
    non-null deferral reason.  Strict equality on both counts — an
    over-complete run is not the declared tier either.  The run manifest
    and the gate0 acceptance record both classify through this single
    predicate, so the two records cannot disagree.
    """
    complete = (
        int(classes_completed) == int(declared_classes)
        and int(replicas_per_class_completed) == int(declared_replicas)
    )
    return "complete" if complete else "deferred"


def read_clean_environment(out_dir: Path) -> dict:
    """Read the recorded clean-environment command results, when present."""
    path = Path(out_dir) / CLEAN_ENV_NAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def assemble_gate0(
    out_dir: Path, verification: dict, pre_registration: dict
) -> dict:
    """Assemble ``gate0_r02.json`` from the sealed run's artifacts.

    The record matches the card's acceptance schema exactly.  The verdict
    is ``pass`` only when the hashes verified, the archive is sealed and
    self-consistent, both clean-environment commands recorded ``pass``,
    and every class carries exactly 1017 bins.  ``tier_status`` is
    ``complete`` iff all 6 classes completed all 8 declared T2 replicas;
    otherwise ``deferred`` with the recorded reason (a deferral does not
    block Gate 0, but the archive must exist).
    """
    out_dir = Path(out_dir)
    prereg_path = out_dir / PREREGISTRATION_NAME
    if not prereg_path.is_file():
        _refuse(f"pre-registration missing: {prereg_path}")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))

    manifest_path = out_dir / RUN_MANIFEST_NAME
    if not manifest_path.is_file():
        _refuse(f"run manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    classes = manifest.get("classes") or {}
    class_reports = [classes.get(name) or {} for name in CENTRALITY_CLASSES]
    classes_completed = sum(
        1
        for report in class_reports
        if int(report.get("replicas_requested", 0)) > 0
        and int(report.get("replicas_completed", 0))
        == int(report.get("replicas_requested", 0))
    )
    replicas_per_class_completed = min(
        (int(report.get("replicas_completed", 0)) for report in class_reports),
        default=0,
    )
    events_completed = sum(
        int(report.get("events_completed", 0)) for report in class_reports
    )

    total_bins_ok = True
    failure_maps: dict[str, dict] = {}
    for name in CENTRALITY_CLASSES:
        failure_path = out_dir / failure_map_name(name)
        if not failure_path.is_file():
            total_bins_ok = False
            break
        failure_map = json.loads(failure_path.read_text(encoding="utf-8"))
        failure_maps[name] = failure_map
        if int(failure_map.get("n_total_bins", -1)) != TOTAL_BINS:
            total_bins_ok = False
            break

    sums_path = out_dir / SHA256SUMS_NAME
    if not sums_path.is_file():
        _refuse(f"seal inventory missing: {sums_path}")
    archive_seal_sha256 = sha256_file(sums_path)

    clean_raw = read_clean_environment(out_dir)
    clean_environment_commands = {
        CLEAN_ENV_PYTEST_KEY: clean_raw.get(CLEAN_ENV_PYTEST_KEY, "not_recorded"),
        CLEAN_ENV_SMOKE_KEY: clean_raw.get(CLEAN_ENV_SMOKE_KEY, "not_recorded"),
    }

    config_hash_verified = bool(
        verification.get("config_sha256") == prereg.get("config_hash")
        and verification.get("config_sha256") == PRODUCTION_CONFIG_SHA256
    )
    m8_verified = bool(
        verification.get("m8_parameter_hash") == prereg.get("m8_parameter_hash")
        and verification.get("m8_parameter_hash") == M8_PARAMETER_HASH
    )
    # The card's archive_sealed assertion: the sealed directory exists
    # with a SHA256SUMS.txt and a provenance.json.  The seal manifest
    # itself is written after the acceptance record (gate0_r02.json
    # carries the sums digest, so it cannot precede the record it
    # seals); main() re-verifies every seal entry after writing it and
    # fails the run on any mismatch.
    archive_sealed = bool(
        sums_path.is_file() and (out_dir / PROVENANCE_NAME).is_file()
    )
    immutable = archive_sealed
    if immutable:
        # Every sums entry must still verify at acceptance time; the
        # seal manifest itself is written after this record (see the
        # comment above) and is re-verified by main() after writing.
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, name = line.split("  ", 1)
            payload = out_dir / name
            if not payload.is_file() or sha256_file(payload) != digest:
                immutable = False
                break

    declared_replicas = int(prereg.get("replicas_t2", 0))
    declared_events = int(prereg.get("events_per_replica_t2", 0))
    tier_status = tier_status_decision(
        classes_completed,
        replicas_per_class_completed,
        len(CENTRALITY_CLASSES),
        declared_replicas,
    )
    if tier_status == "complete":
        tier_deferral_reason = None
    else:
        tier_deferral_reason = manifest.get("tier_deferral_reason") or (
            f"{classes_completed} of {len(CENTRALITY_CLASSES)} classes with "
            f"{replicas_per_class_completed} of {declared_replicas} declared "
            f"T2 replicas per class completed ({events_completed} of "
            f"{len(CENTRALITY_CLASSES) * declared_replicas * declared_events} "
            "events); the remaining replicas are a follow-up obligation"
        )

    n_flagged = sum(int(fm.get("n_flagged", 0)) for fm in failure_maps.values())
    n_empty = sum(int(fm.get("n_empty_bins", 0)) for fm in failure_maps.values())

    verdict = (
        "pass"
        if (
            config_hash_verified
            and m8_verified
            and archive_sealed
            and immutable
            and clean_environment_commands[CLEAN_ENV_PYTEST_KEY] == "pass"
            and clean_environment_commands[CLEAN_ENV_SMOKE_KEY] == "pass"
            and total_bins_ok
        )
        else "fail"
    )
    gate = {
        "gate": "gate0",
        "clause": GATE_CLAUSE,
        "config_hash_verified": config_hash_verified,
        "m8_parameter_hash_verified": m8_verified,
        "archive_sealed": archive_sealed,
        "archive_seal_sha256": archive_seal_sha256,
        "clean_environment_commands": clean_environment_commands,
        "target": prereg.get("target", TARGET),
        "unavailable_targets": list(
            prereg.get("unavailable_targets", UNAVAILABLE_TARGETS)
        ),
        "centrality_classes": list(
            prereg.get("centrality_classes", CENTRALITY_CLASSES)
        ),
        "tier": {
            "declared": prereg.get("tier", {}).get("declared", "T2"),
            "declared_replicas": declared_replicas,
            "declared_events_per_replica": declared_events,
            "classes_completed": int(classes_completed),
            "replicas_per_class_completed": int(replicas_per_class_completed),
            "events_completed": int(events_completed),
        },
        "tier_status": tier_status,
        "tier_deferral_reason": tier_deferral_reason,
        "n_total_bins_per_class": TOTAL_BINS,
        "n_classes": len(CENTRALITY_CLASSES),
        "n_flagged_cells": int(n_flagged),
        "empty_bins": int(n_empty),
        "immutable": bool(immutable),
        "verdict": verdict,
    }
    _write_json(out_dir / GATE_RECORD_NAME, gate)
    return gate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def resolve_out_dir(raw: str) -> Path:
    """Resolve ``--out``; ``dev/00runs/...`` paths are monorepo-relative.

    The card's commands invoke this driver from the nested tree with
    ``--out dev/00runs/r02_legacy_pa_reference_2026-09-11``, so an
    artifact-store-shaped path resolves against the monorepo root even
    though the process working directory is the nested tree.  Absolute
    paths and every other relative path resolve normally.
    """
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    if path.parts[:2] == ("dev", "00runs"):
        return MONO_ROOT / path
    return Path.cwd() / path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "R02 legacy p+A reference run driver: seeded PAEventBuilder "
            "replicas of the staged two-source builder at the frozen M8 set "
            "per centrality class, the 1017-bin six-species binned "
            "prediction with per-hemisphere yields, the combined "
            "covariance, residuals against the SPS p+Pb tables, failure "
            "maps, and the sealed immutable negative-control archive"
        )
    )
    parser.add_argument(
        "--config", required=True, help="bundled legacy_v1 config JSON"
    )
    parser.add_argument(
        "--target",
        default=TARGET,
        help=f"target nucleus (only {TARGET} is available; C and W have no data)",
    )
    parser.add_argument(
        "--centrality-classes",
        default=",".join(CENTRALITY_CLASSES),
        help=(
            "comma-separated centrality classes "
            "(minimum_bias,set_1,set_2,set_3,set_4,set_5); default: all six"
        ),
    )
    parser.add_argument(
        "--replicas",
        type=int,
        required=True,
        help="number of replicas per class (the declared T2 tier is 8)",
    )
    parser.add_argument(
        "--events-per-replica",
        type=int,
        required=True,
        help="events per replica (the declared T2 tier is 100,000)",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=SEED_BASE,
        help=(
            f"replica seed base (default {SEED_BASE}); per-class seeds are "
            "seed_base + 50000 + 100*i + k"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            "worker processes spreading independent replica units "
            "(default 4); the worker count does not affect any generated "
            "event"
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
        help="run directory (dev/00runs/... resolves against the monorepo root)",
    )
    parser.add_argument(
        "--seal",
        action="store_true",
        help=(
            "write provenance.json, SHA256SUMS.txt, and seal_manifest.json "
            "(and, for the canonical production run, gate0_r02.json) after "
            "the per-class artifacts"
        ),
    )
    return parser


def _parse_classes(raw: str) -> list[str]:
    classes = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not classes:
        _refuse("--centrality-classes must list at least one class")
    unknown = [name for name in classes if name not in CENTRALITY_CLASSES]
    if unknown:
        _refuse(
            f"unknown centrality classes {unknown!r}; the pre-registered "
            f"classes are {list(CENTRALITY_CLASSES)} (a different centrality "
            "set files its own card)"
        )
    if len(set(classes)) != len(classes):
        _refuse(f"duplicated centrality classes in {raw!r}")
    return classes


def _tier_fields(
    class_reports: Mapping[str, dict],
    declared_replicas: int,
    declared_events_per_replica: int,
) -> tuple[int, int, int, str, str | None]:
    """Classes/replicas/events completed plus the strict tier decision."""
    classes_completed = sum(
        1
        for report in class_reports.values()
        if int(report["replicas_requested"]) > 0
        and int(report["replicas_completed"]) == int(report["replicas_requested"])
    )
    replicas_per_class_completed = (
        min(int(report["replicas_completed"]) for report in class_reports.values())
        if class_reports
        else 0
    )
    events_completed = sum(
        int(report["events_completed"]) for report in class_reports.values()
    )
    tier_status = tier_status_decision(
        classes_completed,
        replicas_per_class_completed,
        len(CENTRALITY_CLASSES),
        declared_replicas,
    )
    tier_deferral_reason = None
    if tier_status == "deferred":
        tier_deferral_reason = (
            f"{classes_completed} of {len(CENTRALITY_CLASSES)} classes with "
            f"{replicas_per_class_completed} of {declared_replicas} declared "
            f"T2 replicas per class completed ({events_completed} of "
            f"{len(CENTRALITY_CLASSES) * declared_replicas * declared_events_per_replica} "
            "events); the remaining replicas are a follow-up obligation"
        )
    return (
        classes_completed,
        replicas_per_class_completed,
        events_completed,
        tier_status,
        tier_deferral_reason,
    )


def _write_run_manifest(
    out_dir: Path,
    *,
    args,
    classes: list[str],
    seeds_per_class: Mapping[str, list[int]],
    class_reports: Mapping[str, dict],
    verification: dict,
    mapping_crosscheck: dict,
    table_digest: str,
    sigma_eta: float,
    sqrt_s_gev: float,
    run_wall_seconds: float,
    tier_status: str,
    tier_deferral_reason: str | None,
) -> Path:
    """(Re)write ``run_manifest.json`` (after each class and at the end)."""
    classes_completed, replicas_per_class_completed, events_completed, _, _ = (
        _tier_fields(class_reports, T2_REPLICAS, T2_EVENTS_PER_REPLICA)
    )
    payload = {
        "card": CARD,
        "run": RUN_DESCRIPTION,
        "gate_clause": GATE_CLAUSE,
        "target": TARGET,
        "unavailable_targets": list(UNAVAILABLE_TARGETS),
        "centrality_classes": list(classes),
        "wound_bands": {name: WOUND_BANDS[name] for name in classes},
        "e_be_per_class": {name: E_BE_K_BAND_MEAN[name] for name in classes},
        "e_be_provenance": (
            "the pre-registered e_be_per_class value of a set is the M13 "
            "E_BE-to-K mapping's calibrated mean target-wound number <K> of "
            "the set's band (mapping_table.sets[k].k_band_mean of "
            f"{M13_PREREG_RELPATH}); the generation slice is the band itself"
        ),
        "nu_measured_per_class": {name: NU_MEASURED[name] for name in classes},
        "centrality_mapping_crosscheck": mapping_crosscheck,
        "hemisphere_rule": dict(HEMISPHERE_RULE),
        "hemisphere_convention": HEMISPHERE_CONVENTION,
        "data_convention": PA_DATA_CONVENTION,
        "data_match_convention": PA_DATA_MATCH_CONVENTION,
        "seed_base": int(args.seed_base),
        "seed_offset_base": SEED_OFFSET_BASE,
        "seed_offset_per_class": SEED_OFFSET_PER_CLASS,
        "replica_seeds_per_class": {
            name: [int(s) for s in seeds_per_class[name]] for name in classes
        },
        "replicas_requested": int(args.replicas),
        "events_per_replica": int(args.events_per_replica),
        "declared_t2": {
            "replicas": T2_REPLICAS,
            "events_per_replica": T2_EVENTS_PER_REPLICA,
            "seeds_per_class": {
                name: class_replica_seeds(SEED_BASE, i, T2_REPLICAS)
                for i, name in enumerate(CENTRALITY_CLASSES)
            },
        },
        "sigma_inel_mb": SIGMA_INEL_MB,
        "sqrt_s_gev": float(sqrt_s_gev),
        "sigma_eta": float(sigma_eta),
        "observable_set": {
            "window": dict(PAPER_WINDOW),
            "species": dict(SPECIES_BIN_COUNTS),
            "total_bins": TOTAL_BINS,
        },
        "tables_sha256": table_digest,
        "config_sha256": verification["config_sha256"],
        "m8_parameter_hash": verification["m8_parameter_hash"],
        "classes": {
            name: class_reports[name] for name in classes if name in class_reports
        },
        "tier": {
            "declared": "T2",
            "declared_replicas": T2_REPLICAS,
            "declared_events_per_replica": T2_EVENTS_PER_REPLICA,
            "classes_completed": int(classes_completed),
            "replicas_per_class_completed": int(replicas_per_class_completed),
            "events_completed": int(events_completed),
        },
        "tier_status": tier_status,
        "tier_deferral_reason": tier_deferral_reason,
        "worker_count": int(args.workers),
        "wall_seconds": round(float(run_wall_seconds), 3),
        "finished_utc": _utc_now_iso(),
        "manifest_convention": (
            "rewritten after each class completes; the per-replica records "
            "live in replica_<class>_<seed>.json with the ordered event_hash "
            "digest set per replica"
        ),
    }
    manifest_path = Path(out_dir) / RUN_MANIFEST_NAME
    _write_json(manifest_path, payload)
    return manifest_path


def main(argv: list | None = None) -> int:
    args = _build_parser().parse_args(argv)
    # Positivity checks fire before the --out directory is created: 0 and
    # negative counts are refused before any generation.
    if int(args.replicas) <= 0:
        _refuse(
            f"--replicas must be a positive integer, got {args.replicas}; "
            "0 and negative counts are refused before any generation"
        )
    if int(args.events_per_replica) <= 0:
        _refuse(
            f"--events-per-replica must be a positive integer, got "
            f"{args.events_per_replica}; 0 and negative counts are refused "
            "before any generation"
        )
    if str(args.target) != TARGET:
        _refuse(
            f"--target {args.target!r} is unavailable: {TARGET} is the only "
            f"p+A target with data in the repository "
            f"({', '.join(UNAVAILABLE_TARGETS)} are recorded as unavailable); "
            "a new target files its own card"
        )
    classes = _parse_classes(args.centrality_classes)
    out_dir = resolve_out_dir(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload_expected = default_pre_registration()
    prereg_path = out_dir / PREREGISTRATION_NAME
    if prereg_path.is_file():
        loaded = json.loads(prereg_path.read_text(encoding="utf-8"))
        if loaded != payload_expected:
            _refuse(
                f"pre-registration mismatch: {prereg_path} does not equal the "
                "R02 pre-registration (target, centrality classes, E_BE "
                "values, seed rule, tier, hashes, normalization, hemisphere "
                "rule, and the 1017-bin observable set are pre-registered; a "
                "different registration files its own card)"
            )
    else:
        _write_json(prereg_path, payload_expected)
    pre_registration = payload_expected

    config_path = Path(args.config)
    if not config_path.is_file() and config_path.name == PA_CONFIG_NAME:
        # The card's config/staged_excitation_pA.json names the staged
        # excitation p+A model; the bundled staged-excitation config
        # carries that model (the pinned Pb208 nucleus block plus the pA
        # runner settings), no separate p+A config file exists in the
        # frozen tree, and the name resolves onto the same bytes card 1
        # pins.  The hash verification below holds the resolution to the
        # frozen bytes either way.
        resolved = NESTED_ROOT / "config" / PRODUCTION_CONFIG_NAME
        print(
            f"[r02] {config_path} resolves onto the bundled "
            f"{PRODUCTION_CONFIG_NAME} (the p+A model config; same pinned "
            "bytes)",
            flush=True,
        )
        config_path = resolved
    verification = verify_config_and_m8(config_path, pre_registration)
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    build_pa_config(raw_config)  # refuse before any generation on a bad application
    mapping_crosscheck = verify_centrality_constants()

    print(
        f"[r02] config {config_path} sha256 {verification['config_sha256']}",
        flush=True,
    )
    print(
        f"[r02] M8 parameter hash {verification['m8_parameter_hash']} verified "
        f"against {verification['manifest_path']}",
        flush=True,
    )
    print(
        f"[r02] centrality mapping: bands "
        f"{ {name: WOUND_BANDS[name] for name in classes} }, e_be (mapped <K>) "
        f"{ {name: E_BE_K_BAND_MEAN[name] for name in classes} }; M13 "
        f"cross-check {'verified' if mapping_crosscheck.get('cross_checked') else 'file absent'}",
        flush=True,
    )

    tables = load_binned_tables(raw_config)
    table_digest = tables_digest(tables)
    sigma_eta = r01.read_sigma_eta()
    sqrt_s_gev = math.sqrt(
        Reaction(float(raw_config.get("beam_momentum", 158.0)), 2212, 2212).s()
    )
    if abs(sqrt_s_gev - SQRT_S_NN_GEV_PIN) > 1e-9:
        _refuse(
            f"sqrt(s) {sqrt_s_gev!r} != the M13 p+A NN convention "
            f"{SQRT_S_NN_GEV_PIN!r}; a different energy convention files "
            "its own card"
        )
    print(
        f"[r02] observable set: {sum(int(t.data.size) for t in tables)} bins "
        f"over {len(tables)} species; sigma_inel {SIGMA_INEL_MB} mb; "
        f"sqrt(s)_NN {sqrt_s_gev:.6f} GeV; sigma_eta {sigma_eta}",
        flush=True,
    )

    production_run = bool(
        verification["production_config"]
        and tuple(classes) == tuple(CENTRALITY_CLASSES)
        and int(args.replicas) == T2_REPLICAS
        and int(args.events_per_replica) == T2_EVENTS_PER_REPLICA
        and int(args.seed_base) == SEED_BASE
    )

    seeds_per_class = {
        name: class_replica_seeds(args.seed_base, index, args.replicas)
        for index, name in enumerate(classes)
    }
    payload = config_payload(raw_config)
    run_started = time.perf_counter()
    class_reports: dict[str, dict] = {}
    for class_index, class_name in enumerate(classes):
        band = WOUND_BANDS[class_name]
        seeds = seeds_per_class[class_name]
        print(
            f"[r02] class {class_name} (band {band}), {args.replicas} x "
            f"{args.events_per_replica} events, seeds {seeds[0]}..{seeds[-1]}",
            flush=True,
        )
        records = list(
            run_class_replicas(
                class_name,
                class_index,
                seeds,
                int(args.events_per_replica),
                payload,
                tables,
                sqrt_s_gev,
                int(args.workers),
                out_dir,
                verification["config_sha256"],
                table_digest,
            )
        )
        records_by_seed = {int(record["seed"]): record for record in records}
        ordered = [
            records_by_seed[int(seed)] for seed in seeds if int(seed) in records_by_seed
        ]
        if len(ordered) != len(seeds):
            _refuse(
                f"class {class_name}: {len(ordered)} of {len(seeds)} replica "
                "records completed; the worker pool failed mid-class"
            )

        totals: dict[str, np.ndarray] = {}
        count_totals: dict[str, np.ndarray] = {}
        hemi_totals: dict[str, np.ndarray] = {}
        for record in ordered:
            for species, values in record["sums"].items():
                values = np.asarray(values, dtype=float)
                totals[species] = (
                    values if species not in totals else totals[species] + values
                )
            for species, values in record["counts"].items():
                values = np.asarray(values, dtype=np.int64)
                count_totals[species] = (
                    values
                    if species not in count_totals
                    else count_totals[species] + values
                )
            for species, values in record["hemisphere_sums"].items():
                values = np.asarray(values, dtype=float)
                hemi_totals[species] = (
                    values
                    if species not in hemi_totals
                    else hemi_totals[species] + values
                )
        events_completed = sum(int(record["n_events"]) for record in ordered)
        entries, values_by_species, errors_by_species = (
            r01.bin_predictions_payload(
                tables, totals, count_totals, events_completed, sqrt_s_gev
            )
        )
        _write_json(out_dir / bin_predictions_name(class_name), entries)

        hemisphere = hemisphere_yields_payload(
            hemi_totals, events_completed, sqrt_s_gev
        )
        _write_json(out_dir / hemisphere_yields_name(class_name), hemisphere)

        pa_data = load_pa_data(class_name)
        covariance = covariance_payload(tables, errors_by_species, sigma_eta, pa_data)
        _write_json(out_dir / covariance_name(class_name), covariance)

        residuals = residuals_payload(
            tables, values_by_species, errors_by_species, pa_data
        )
        _write_json(out_dir / residuals_name(class_name), residuals)

        failure_map = build_failure_map(residuals, count_totals)
        _write_json(out_dir / failure_map_name(class_name), failure_map)

        chi2 = chi2_payload(residuals)
        class_wall = round(
            sum(float(record["wall_seconds"]) for record in ordered), 3
        )
        data_species = sorted(pa_data.keys())
        class_reports[class_name] = {
            "class_index": int(class_index),
            "band": list(band) if band is not None else None,
            "e_be_k_band_mean": E_BE_K_BAND_MEAN[class_name],
            "nu_measured": NU_MEASURED[class_name],
            "seeds": [int(s) for s in seeds],
            "replicas_requested": int(args.replicas),
            "replicas_completed": len(ordered),
            "events_per_replica": int(args.events_per_replica),
            "events_completed": int(events_completed),
            "wall_seconds": class_wall,
            "replicas": [
                {
                    "seed": int(record["seed"]),
                    "replica_index": int(record["replica_index"]),
                    "status": "completed",
                    "n_events": int(record["n_events"]),
                    "wall_seconds": record["wall_seconds"],
                    "event_hash_set_sha256": record["event_hash_set_sha256"],
                    "mean_target_wounds": record["mean_target_wounds"],
                    "record": replica_record_name(class_name, int(record["seed"])),
                }
                for record in ordered
            ],
            "data_species": {
                species: {
                    "file": entry["file"],
                    "sha256": entry["sha256"],
                    "n_rows": entry["n_rows"],
                    "n_unique_cells": entry["n_unique_cells"],
                    "n_pt_zero_cells": entry["n_pt_zero_cells"],
                }
                for species, entry in pa_data.items()
            },
            "model_only": len(data_species) == 0,
            "model_only_species": [
                species for species in SPECIES_DISPLAY if species not in data_species
            ],
            "n_data_matched_bins": {
                species: sum(
                    1
                    for entry in residuals
                    if entry["species"] == species and entry["d"] is not None
                )
                for species in SPECIES_DISPLAY
            },
            "n_flagged": int(failure_map["n_flagged"]),
            "n_empty_bins": int(failure_map["n_empty_bins"]),
            "chi2_per_species": chi2["per_species"],
            "hemisphere_yields": hemisphere,
            "artifacts": [
                bin_predictions_name(class_name),
                hemisphere_yields_name(class_name),
                covariance_name(class_name),
                residuals_name(class_name),
                failure_map_name(class_name),
            ],
        }
        chi2_summary = ", ".join(
            (
                f"{species} {per['chi2']:.2f}"
                if per.get("chi2") is not None
                else f"{species} model-only"
            )
            for species, per in chi2["per_species"].items()
        )
        print(
            f"[r02] class {class_name}: {events_completed} events over "
            f"{len(ordered)} replica(s); bins {len(entries)}; n_flagged "
            f"{failure_map['n_flagged']}; empty_bins "
            f"{failure_map['n_empty_bins']}; chi2 {chi2_summary}",
            flush=True,
        )
        (
            classes_completed,
            replicas_per_class_completed,
            events_total,
            tier_status,
            tier_deferral_reason,
        ) = _tier_fields(class_reports, T2_REPLICAS, T2_EVENTS_PER_REPLICA)
        _write_run_manifest(
            out_dir,
            args=args,
            classes=classes,
            seeds_per_class=seeds_per_class,
            class_reports=class_reports,
            verification=verification,
            mapping_crosscheck=mapping_crosscheck,
            table_digest=table_digest,
            sigma_eta=sigma_eta,
            sqrt_s_gev=sqrt_s_gev,
            run_wall_seconds=time.perf_counter() - run_started,
            tier_status=tier_status,
            tier_deferral_reason=tier_deferral_reason,
        )

    wall = time.perf_counter() - run_started
    (
        classes_completed,
        replicas_per_class_completed,
        events_completed,
        tier_status,
        tier_deferral_reason,
    ) = _tier_fields(class_reports, T2_REPLICAS, T2_EVENTS_PER_REPLICA)
    manifest_path = _write_run_manifest(
        out_dir,
        args=args,
        classes=classes,
        seeds_per_class=seeds_per_class,
        class_reports=class_reports,
        verification=verification,
        mapping_crosscheck=mapping_crosscheck,
        table_digest=table_digest,
        sigma_eta=sigma_eta,
        sqrt_s_gev=sqrt_s_gev,
        run_wall_seconds=wall,
        tier_status=tier_status,
        tier_deferral_reason=tier_deferral_reason,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    print(
        f"[r02] pooled predictions: {events_completed} events over "
        f"{classes_completed} class(es) at {replicas_per_class_completed} "
        f"replica(s) per class; tier_status {manifest['tier_status']}",
        flush=True,
    )

    exit_code = 0
    if args.seal:
        input_paths = [
            Path(verification["config_path"]),
            Path(verification["manifest_path"]),
            NESTED_ROOT / "config" / "v2_singlet" / "likelihood_nuisances.json",
        ] + [Path(t.path) for t in tables]
        for class_name in classes:
            for entry in (class_reports[class_name].get("data_species") or {}).values():
                input_paths.append(pa_data_dir() / entry["file"])
        write_provenance(
            out_dir,
            verification=verification,
            pre_registration=pre_registration,
            seeds_per_class=seeds_per_class,
            replicas_per_class={name: int(args.replicas) for name in classes},
            events_per_replica=int(args.events_per_replica),
            events_completed=events_completed,
            argv=[str(a) for a in sys.argv[1:]],
            wall_seconds=wall,
            tier_status=tier_status,
            tier_deferral_reason=tier_deferral_reason,
            input_paths=input_paths,
        )
        write_sha256sums(out_dir)
        gate = None
        if production_run:
            gate = assemble_gate0(out_dir, verification, pre_registration)
        # The seal manifest is written after the acceptance record so it
        # covers gate0_r02.json too (gate0 carries the sums digest, so
        # the sums inventory excludes gate0; the seal manifest cannot
        # contain its own digest).  Every seal entry is re-verified
        # afterwards; any mismatch fails the run.
        seal = write_seal_manifest(out_dir)
        seal_ok = True
        for entry in seal["files"]:
            path = out_dir / entry["path"]
            if not path.is_file() or sha256_file(path) != entry["sha256"]:
                seal_ok = False
                break
        sums_ok = True
        sums_path = out_dir / SHA256SUMS_NAME
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, name = line.split("  ", 1)
            payload_file = out_dir / name
            if not payload_file.is_file() or sha256_file(payload_file) != digest:
                sums_ok = False
                break
        if not seal_ok or not sums_ok:
            print(
                "[r02] seal verification FAILED; the run directory is not "
                "immutable as written",
                flush=True,
            )
            return 1
        print(
            f"[r02] run directory sealed: {out_dir} "
            f"({len(seal['files'])} sealed files);"
            + (
                f" gate0_r02.json verdict {gate['verdict']}"
                if gate is not None
                else " no gate record (not the canonical production run)"
            ),
            flush=True,
        )
        if gate is not None and gate["verdict"] != "pass":
            exit_code = 1
    else:
        print(
            "[r02] per-class artifacts written; rerun with --seal to write "
            "provenance.json, SHA256SUMS.txt, seal_manifest.json, and the "
            "gate0_r02.json acceptance record",
            flush=True,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
