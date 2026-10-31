#!/usr/bin/env python3
"""R01 legacy p+p reference run driver (Round 1, card 10).

Runs the pre-registered legacy p+p reference benchmark R01 of plan section
6 and writes its acceptance record ``gate0_r01.json``.  The run evaluates
the frozen ``legacy_v1`` two-source model (``PPEventBuilder`` at the M8
ten-parameter set of card 1's manifest) against the 1017-bin six-species
NA49 binned observable set, with no parameter changes and no fitting:

- per replica, ``replica_k = seed_base + 40000 + k`` (the M8 substream
  offset convention; ``config_m8.json`` uses ``seed = config_seed + 40000
  + k``), a separate seeded staged p+p generation of
  ``--events-per-replica`` events;
- all completed replicas are pooled into one binned prediction (the
  estimator's ``N_events`` is the total over the pooled replicas):
  ``M_bin = (1 / (pi * sqrt(s) * N_events)) * sum_events (E / pT) / (dxF *
  dpT)``, the invariant cross section per inelastic event, i.e. ``f /
  sigma_inel`` with ``sigma_inel = 31.78 mb`` (the M4/M8 convention), on
  the table binning of ``na49_mixing.staged_evaluation`` (the paper
  window ``xF in [0, 0.4]``, ``pT in [0.05, 1.5]``; species bin counts
  pi- 229, pi+ 230, K+ 153, K- 148, p 135, pbar 122, total 1017);
- per-bin statistics: the model prediction's statistical error follows
  the standard relative convention ``stat_error = value * sqrt(1/N_bin)``
  (N_bin the number of that species' tracks binned in the cell); a bin
  with ``N_bin == 0`` carries an infinite MC error and is listed in
  ``failure_map.json`` under ``empty_bins`` instead of being dropped;
- residuals ``delta_over_sigma = (d - m) / sigma`` with ``d`` the SPS
  table value (converted to ``f / sigma_inel``), ``m`` the pooled model
  prediction, and ``sigma = sqrt(V_data_bin + V_mc_bin)`` the combined
  per-bin error of card 7 (diagonal in Round 1); the normalization
  nuisance is recorded with ``eta = 0.0`` and the card-7 width
  ``sigma_eta = 0.01`` (read from
  ``config/v2_singlet/likelihood_nuisances.json``), scope
  ``all_six_tables``;- the failure map flags cells with ``|delta_over_sigma| > 3.0``, ranked
  by ``|pull|`` descending (rank 1 = largest |pull|);
- per-species chi-square ``sum(delta_over_sigma**2)`` over that species'
  bins, degrees of freedom ``bins - 1`` (recorded convention; R01 fits
  no parameter), written to ``chi2.json``.

Verification before any generation: the config file sha256 must match
the pin recorded in card 1's manifest
(``config/v2_singlet/legacy_v1_manifest.json``) and, for the production
config, the pre-registration's ``config_hash``; the canonical sha256 of
the manifest's ``m8_frozen_pp_parameters`` block must match the
pre-registered ``m8_parameter_hash``.  A mismatch refuses the run with
a non-zero exit code before a single event is generated.

Per-event seeding contract: event ``e`` of seed ``s`` draws from
``numpy.random.default_rng(numpy.random.SeedSequence(s,
spawn_key=(e,)))``.  The rule is a pure function of ``(s, e)`` and
consumes no shared generator state, so any worker count produces the
same stream; the worker pool exists to spread the wall clock.  Each
replica records a completion manifest (seed, n_events, wall_seconds,
sha256 of the ordered event-hash digest set, and the pooled per-bin
weight sums and track counts) so the declared T3 tier
(20 x 250,000 events) can be completed incrementally by re-running the
driver over the same output directory.

Output files under ``--out``: ``pre_registration.json`` (written before
generation), one ``replica_<seed>.json`` per completed replica,
``replica_seeds.json``, ``bin_predictions.json``, ``covariance.json``,
``residuals.json``, ``failure_map.json``, ``chi2.json``, and with
``--seal`` the immutable-archive triple ``provenance.json`` (via the
``run_provenance.py`` convention), ``SHA256SUMS.txt`` (sha256sum -c
format, sorted, every regular file under the run directory except the
seal/provenance/acceptance files, which cannot carry their own
digests), and ``seal_manifest.json`` (records the payload files plus
``SHA256SUMS.txt``, ``provenance.json``, and ``gate0_r01.json``; it
cannot contain its own digest).  ``gate0_r01.json`` is assembled only
for the production config run; its ``clean_environment_commands`` block
is read from ``clean_environment_commands.json`` in the run directory
when present.

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
import json
import math
import multiprocessing
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, NoReturn

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
    print(f"staged_r01_reference: {message}", file=sys.stderr, flush=True)
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
from na49_mixing.continuum_channels import ContinuumParameters  # noqa: E402
from na49_mixing.event_builder_pp import PPEventBuilder, PPExcitationConfig  # noqa: E402
from na49_mixing.exchange_model import (  # noqa: E402
    ReggeExcitationParameters,
    SimpleExcitationParameters,
)
from na49_mixing.reaction import Reaction  # noqa: E402
from na49_mixing.staged_evaluation import (  # noqa: E402
    BinnedFitTable,
    preflight_binned_tables,
)

# ---------------------------------------------------------------------------
# Pre-registered run identity (card TASK-20260911-r1-run-r01-legacy-pp-
# reference; the seed rule, tier, normalization, and observable set are
# verbatim from the card, the hashes are read off the frozen tree).
# ---------------------------------------------------------------------------

CARD = "TASK-20260911-r1-run-r01-legacy-pp-reference"
RUN_DESCRIPTION = "r01_legacy_pp_reference_2026-09-11"
GATE_RECORD_NAME = "gate0_r01.json"
GATE_CLAUSE = "archive_commands_tests_and_immutable_pp_reference"

PREREGISTRATION_NAME = "pre_registration.json"
REPLICA_MANIFEST_NAME = "replica_seeds.json"
BIN_PREDICTIONS_NAME = "bin_predictions.json"
COVARIANCE_NAME = "covariance.json"
RESIDUALS_NAME = "residuals.json"
FAILURE_MAP_NAME = "failure_map.json"
CHI2_NAME = "chi2.json"
CLEAN_ENV_NAME = "clean_environment_commands.json"
SHA256SUMS_NAME = "SHA256SUMS.txt"
SEAL_NAME = "seal_manifest.json"
PROVENANCE_NAME = "provenance.json"

STAGE_LABEL = "r01_reference"

SEED_BASE = 20260911
SEED_OFFSET = 40000
T3_REPLICAS = 20
T3_EVENTS_PER_REPLICA = 250000
SIGMA_INEL_MB = 31.78

# The 1017-bin six-species observable set (paper window, M4/M8
# convention).  Keys are the pre-registered display names in the config's
# fit.binned.tables order; SPECIES_DISPLAY maps the config species keys.
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
CHUNK_SIZE = 2500
DEFAULT_WORKERS = 4

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
PRODUCTION_CONFIG_NAME = "staged_excitation.json"
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

EVENT_SEED_RULE = (
    "numpy.random.default_rng(numpy.random.SeedSequence(seed, "
    "spawn_key=(event_id,))); a pure function of (seed, event_id), no "
    "shared generator state, so any worker count is bit-identical"
)

# Clean-environment command keys recorded verbatim in gate0_r01.json.
CLEAN_ENV_PYTEST_KEY = "cd dev/mixing2006 && python -m pytest"
CLEAN_ENV_SMOKE_KEY = (
    "na49-staged-excitation --config config/staged_excitation_smoke.json"
)

# Modules whose installed source defines the staged p+p generation
# behavior fingerprinted into provenance.json (salted with the
# RESONANCE_DECAY_LIB_VERSION cache salt).
FINGERPRINT_MODULES = (
    "na49_mixing.event_builder_pp",
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


# ---------------------------------------------------------------------------
# Pre-registration and verification
# ---------------------------------------------------------------------------


def replica_seeds(seed_base: int, replicas: int) -> list[int]:
    """The per-replica seeds: ``seed_base + 40000 + k``, k = 0..replicas-1."""
    return [int(seed_base) + SEED_OFFSET + k for k in range(int(replicas))]


def default_pre_registration() -> dict:
    """The R01 pre-registration payload, exactly as the card defines it."""
    return {
        "seed_base": SEED_BASE,
        "seed_offset": SEED_OFFSET,
        "replicas_t3": T3_REPLICAS,
        "events_per_replica_t3": T3_EVENTS_PER_REPLICA,
        "config_hash": PRODUCTION_CONFIG_SHA256,
        "m8_parameter_hash": M8_PARAMETER_HASH,
        "sigma_inel_mb": SIGMA_INEL_MB,
        "observable_set": {
            "window": {"xF": [0.0, 0.4], "pT": [0.05, 1.5]},
            "species": dict(SPECIES_BIN_COUNTS),
            "total_bins": TOTAL_BINS,
        },
        "tier": {
            "declared": "T3",
            "declared_replicas": T3_REPLICAS,
            "declared_events_per_replica": T3_EVENTS_PER_REPLICA,
        },
        # Derived, pre-registered seed list: the M8 substream rule applied
        # to the declared tier (the full 20-seed archive the card carries).
        "replica_seeds": replica_seeds(SEED_BASE, T3_REPLICAS),
    }


def _config_pin(name: str) -> tuple[str, str]:
    """Map a bundled config file name to (pinned sha256, manifest key)."""
    if name == PRODUCTION_CONFIG_NAME:
        return PRODUCTION_CONFIG_SHA256, PRODUCTION_CONFIG_RELPATH
    if name == SMOKE_CONFIG_NAME:
        return SMOKE_CONFIG_SHA256, SMOKE_CONFIG_RELPATH
    _refuse(
        f"refusing config {name!r}; R01 consumes only the two bundled "
        f"legacy_v1 configs ({PRODUCTION_CONFIG_NAME!r}, {SMOKE_CONFIG_NAME!r})"
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
    if name == PRODUCTION_CONFIG_NAME:
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
        if m8_block != M8_FROZEN_PP_PARAMETERS:
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
        "production_config": name == PRODUCTION_CONFIG_NAME,
    }


def build_pp_config(raw_config: dict) -> tuple[PPExcitationConfig, dict]:
    """Parameter dataclasses from the config with the frozen M8 set applied.

    Mirrors the M13 ``build_parameter_set`` convention: the bundled
    config blocks build the dataclasses, then each of the ten M8 values
    overrides its field, the application is asserted, and the H3 decay
    kernel must hold as the dataclass default.
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
    config = PPExcitationConfig(
        beam_momentum=float(raw_config.get("beam_momentum", 158.0)),
        production_mode="regge",
        simple=simple,
        regge=regge,
        continuum=continuum,
    )
    return config, applied


def config_payload(raw_config: dict) -> dict:
    """Plain-dict projection of the config for worker processes."""
    return {
        "beam_momentum": float(raw_config.get("beam_momentum", 158.0)),
        "simple": dict(raw_config.get("simple") or {}),
        "regge": dict(raw_config.get("regge") or {}),
        "continuum": dict(raw_config.get("continuum") or {}),
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
        _refuse("config carries no fit.binned block; R01 bins the six-species set")
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
    registered_counts = list(
        default_pre_registration()["observable_set"]["species"].items()
    )
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
    payload = [
        {
            "species": t.species,
            "pdg_id": int(t.pdg_id),
            "path": t.path,
            "sha256": sha256_file(t.path),
            "n_bins": int(t.data.size),
            "xf_centers": [float(v) for v in t.xf_centers],
            "pt_centers": [float(v) for v in t.pt_centers],
        }
        for t in tables
    ]
    return canonical_json_sha256(payload)


# ---------------------------------------------------------------------------
# Binning (the staged_evaluation._bin_tracks convention, pooled)
# ---------------------------------------------------------------------------


def _bin_track_sums(
    events: list,
    tables: list[BinnedFitTable],
    sqrt_s_gev: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Pooled per-bin weight sums and track counts at the table bins.

    Mirrors ``na49_mixing.staged_evaluation._bin_tracks`` exactly (same
    grid edges, same ``sps_core.grid_indices`` binning, same pT-major
    flat layout, same in-bounds drop), but returns the raw per-bin sums
    of the Jacobian weights ``E/pT`` and the per-bin track counts instead
    of the normalized densities, so replicas pool as
    ``M_bin = sum_sums / (pi * sqrt(s) * N_total * dxF * dpT)`` with the
    total event count across replicas.  On a single batch the normalized
    result is bit-identical to ``_bin_tracks`` (pinned by the harness
    test).
    """
    n_ev = float(len(events))
    if n_ev <= 0.0:
        raise ValueError("cannot build binned spectra from an empty sample")
    pdgs: list[int] = []
    pxs: list[float] = []
    pys: list[float] = []
    pzs: list[float] = []
    es: list[float] = []
    for event in events:
        for particle in event.final_particles:
            pdgs.append(particle.pdg_id)
            p4 = particle.p4
            pxs.append(p4.px)
            pys.append(p4.py)
            pzs.append(p4.pz)
            es.append(p4.e)
    pdg = np.asarray(pdgs, dtype=np.int64)
    px = np.asarray(pxs, dtype=float)
    py = np.asarray(pys, dtype=float)
    pz = np.asarray(pzs, dtype=float)
    e = np.asarray(es, dtype=float)
    pt = np.hypot(px, py)
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    for table in tables:
        xedges = sps_core.edges_from_centers(table.xf_centers)
        pedges = sps_core.edges_from_centers(table.pt_centers)
        n_x = xedges.size - 1
        n_p = pedges.size - 1
        selected = (pdg == table.pdg_id) & (pt > 0.0)
        if selected.any():
            x_sel = 2.0 * pz[selected] / sqrt_s_gev
            p_sel = pt[selected]
            w_sel = e[selected] / p_sel
            ix_sel, ip_sel = sps_core.grid_indices(x_sel, p_sel, xedges, pedges)
            in_bounds = (
                (ix_sel >= 0) & (ix_sel < n_x) & (ip_sel >= 0) & (ip_sel < n_p)
            )
            flat = ip_sel[in_bounds] * n_x + ix_sel[in_bounds]
            weights_flat = np.bincount(
                flat, weights=w_sel[in_bounds], minlength=n_p * n_x
            )[: n_p * n_x]
            counts_flat = np.bincount(flat, minlength=n_p * n_x)[: n_p * n_x]
        else:
            weights_flat = np.zeros(n_p * n_x, dtype=float)
            counts_flat = np.zeros(n_p * n_x, dtype=np.int64)
        pick = table.bin_index_pt * n_x + table.bin_index_xf
        sums[table.species] = np.asarray(weights_flat[pick], dtype=float)
        counts[table.species] = np.asarray(counts_flat[pick], dtype=np.int64)
    return sums, counts


def _event_rng(seed: int, event_id: int):
    return np.random.default_rng(
        np.random.SeedSequence(int(seed), spawn_key=(int(event_id),))
    )


def _generate_chunk(task: tuple):
    """Generate one chunk of events in a worker process and bin it.

    Returns ``(start_event_id, event_hashes, sums, counts, n_events)``.
    ``event_hashes`` holds the chunk's ``ExcitationEvent.event_hash``
    values in event order; ``sums``/``counts`` hold the per-species binned
    weight sums and track counts at the table bins.  No event payload
    crosses the process boundary.
    """
    seed, event_ids, payload, tables, sqrt_s_gev = task
    pp_config, _ = build_pp_config(payload)
    hashes: list[str] = []
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    for event_id in event_ids:
        builder = PPEventBuilder(pp_config, _event_rng(seed, event_id))
        event = builder.generate(int(event_id), stage=STAGE_LABEL)
        hashes.append(event.as_dict()["event_hash"])
        chunk_sums, chunk_counts = _bin_track_sums([event], tables, sqrt_s_gev)
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
    return int(event_ids[0]), hashes, sums, counts, len(event_ids)


def _map_chunks(tasks: list, workers: int, progress_label: str):
    """Yield worker results as they complete over a fork worker pool."""
    context = multiprocessing.get_context("fork")
    completed = 0
    started = time.perf_counter()
    with context.Pool(processes=workers) as pool:
        for result in pool.imap_unordered(_generate_chunk, tasks, chunksize=1):
            completed += 1
            yield result
            if completed % 10 == 0 or completed == len(tasks):
                elapsed = time.perf_counter() - started
                print(
                    f"[r01] {progress_label}: chunk {completed}/{len(tasks)} "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )


def run_replica(
    seed: int,
    replica_index: int,
    n_events: int,
    payload: dict,
    tables: list[BinnedFitTable],
    sqrt_s_gev: float,
    workers: int,
    out_dir: Path,
    config_sha256: str,
    table_digest: str,
) -> dict:
    """One seeded replica: chunked generation, pooled per-bin sums, manifest.

    An existing ``replica_<seed>.json`` that matches the config hash, the
    table digest, and the requested event count is reused unchanged so
    the declared tier can be completed incrementally; anything else is
    regenerated.
    """
    out_dir = Path(out_dir)
    record_path = out_dir / f"replica_{int(seed)}.json"
    if record_path.is_file():
        try:
            stored = json.loads(record_path.read_text(encoding="utf-8"))
        except ValueError:
            stored = None
        if (
            isinstance(stored, dict)
            and int(stored.get("seed", -1)) == int(seed)
            and int(stored.get("n_events", -1)) == int(n_events)
            and stored.get("config_sha256") == config_sha256
            and stored.get("tables_sha256") == table_digest
            and isinstance(stored.get("sums"), dict)
            and isinstance(stored.get("counts"), dict)
        ):
            stored["reused"] = True
            print(
                f"[r01] replica {seed}: reusing completed record "
                f"({int(stored['n_events'])} events)",
                flush=True,
            )
            return stored

    event_ids = list(range(int(n_events)))
    chunks = [
        event_ids[start : start + CHUNK_SIZE]
        for start in range(0, len(event_ids), CHUNK_SIZE)
    ]
    tasks = [(int(seed), chunk, payload, tables, sqrt_s_gev) for chunk in chunks]
    started = time.perf_counter()
    started_iso = _utc_now_iso()
    parts = sorted(
        _map_chunks(tasks, int(workers), f"replica {seed}"),
        key=lambda part: part[0],
    )
    hashes = [h for _, chunk_hashes, _, _, _ in parts for h in chunk_hashes]
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    for _, _, part_sums, part_counts, _ in parts:
        for species, values in part_sums.items():
            values = np.asarray(values, dtype=float)
            sums[species] = values if species not in sums else sums[species] + values
        for species, values in part_counts.items():
            values = np.asarray(values, dtype=np.int64)
            counts[species] = (
                values if species not in counts else counts[species] + values
            )
    wall = time.perf_counter() - started
    finished_iso = _utc_now_iso()
    payload_digest = json.dumps(hashes, separators=(",", ":")).encode("utf-8")
    record = {
        "seed": int(seed),
        "n_events": int(n_events),
        "chunk_size": CHUNK_SIZE,
        "chunks": len(chunks),
        "worker_count": int(workers),
        "event_seed_rule": EVENT_SEED_RULE,
        "event_hash_set_sha256": hashlib.sha256(payload_digest).hexdigest(),
        "config_sha256": config_sha256,
        "tables_sha256": table_digest,
        "wall_seconds": round(wall, 3),
        "started_utc": started_iso,
        "finished_utc": finished_iso,
        "sums": {s: [float(v) for v in values] for s, values in sums.items()},
        "counts": {s: [int(v) for v in values] for s, values in counts.items()},
        "reused": False,
    }
    _write_json(record_path, record)
    return record


# ---------------------------------------------------------------------------
# Artifact builders (pure functions over plain arrays; unit-tested)
# ---------------------------------------------------------------------------


def mc_error_array(values: list | np.ndarray, counts: list | np.ndarray) -> np.ndarray:
    """Per-bin MC statistical error: ``value * sqrt(1 / N_bin)``.

    The standard relative convention for a binned prediction; a bin with
    ``N_bin == 0`` carries an infinite MC error (recorded, never dropped).
    """
    values = np.asarray(values, dtype=float)
    counts = np.asarray(counts, dtype=float)
    populated = counts > 0
    errors = np.full(values.shape, np.inf, dtype=float)
    errors[populated] = values[populated] * np.sqrt(1.0 / counts[populated])
    return errors


def combined_sigma(
    sigma_data: list | np.ndarray, mc_error: list | np.ndarray
) -> np.ndarray:
    """Combined per-bin error ``sqrt(V_data_bin + V_mc_bin)`` (card 7)."""
    sd = np.asarray(sigma_data, dtype=float)
    sm = np.asarray(mc_error, dtype=float)
    return np.sqrt(sd * sd + sm * sm)


def bin_predictions_payload(
    tables: list[BinnedFitTable],
    sums: dict[str, np.ndarray],
    counts: dict[str, np.ndarray],
    n_events_total: int,
    sqrt_s_gev: float,
) -> tuple[list[dict], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """The 1017 per-species binned predictions plus per-species arrays.

    Each entry carries ``{"species", "bin_index", "xF_mid", "pT_mid",
    "value", "stat_error"}``; ``bin_index`` is the 0-based position of the
    bin in its species' windowed table order, and the midpoints are the
    tabulated (xF, pT) centers, which are the bin midpoints of the
    midpoint-edge grid convention.  Returns the entry list plus the
    per-species model values and MC errors for the downstream artifacts.
    """
    if int(n_events_total) <= 0:
        raise ValueError("the pooled prediction needs a positive event count")
    entries: list[dict] = []
    values_by_species: dict[str, np.ndarray] = {}
    errors_by_species: dict[str, np.ndarray] = {}
    for table in tables:
        species = table.species
        xedges = sps_core.edges_from_centers(table.xf_centers)
        pedges = sps_core.edges_from_centers(table.pt_centers)
        n_x = xedges.size - 1
        dx = np.diff(xedges)
        dp = np.diff(pedges)
        area = (dp[:, None] * dx[None, :]).ravel()
        pick = table.bin_index_pt * n_x + table.bin_index_xf
        area_sel = area[pick]
        m = np.asarray(sums[species], dtype=float) / (
            math.pi * float(sqrt_s_gev) * float(n_events_total) * area_sel
        )
        n_bin = np.asarray(counts[species], dtype=np.int64)
        mc_err = mc_error_array(m, n_bin)
        values_by_species[species] = m
        errors_by_species[species] = mc_err
        for i in range(table.data.size):
            entries.append(
                {
                    "species": species,
                    "bin_index": int(i),
                    "xF_mid": float(table.xf_centers[table.bin_index_xf[i]]),
                    "pT_mid": float(table.pt_centers[table.bin_index_pt[i]]),
                    "value": float(m[i]),
                    "stat_error": float(mc_err[i]),
                }
            )
    meta_counts = {t.species: int(t.data.size) for t in tables}
    if sum(meta_counts.values()) != TOTAL_BINS:
        raise ValueError(
            f"expected {TOTAL_BINS} binned points, got {sum(meta_counts.values())}"
        )
    return entries, values_by_species, errors_by_species


def covariance_payload(
    tables: list[BinnedFitTable],
    mc_error_by_species: dict[str, np.ndarray],
    sigma_eta: float,
) -> dict:
    """The card-7 combined covariance, diagonal in Round 1.

    ``V_data`` is the diagonal of the tables' per-bin statistical errors
    (converted to ``f / sigma_inel`` units by ``BinnedFitTable``),
    ``V_mc`` the per-bin MC error squared, ``V_combined = V_data + V_mc``.
    The normalization nuisance is recorded at its nominal ``eta = 0.0``
    with the card-7 width ``sigma_eta`` and scope ``all_six_tables``.
    """
    data_entries: list[dict] = []
    mc_entries: list[dict] = []
    combined_entries: list[dict] = []
    for table in tables:
        mc_err = np.asarray(mc_error_by_species[table.species], dtype=float)
        combined = combined_sigma(table.sigma, mc_err)
        for i in range(table.data.size):
            data_entries.append(
                {
                    "species": table.species,
                    "bin_index": int(i),
                    "variance": float(table.sigma[i] * table.sigma[i]),
                }
            )
            mc_entries.append(
                {
                    "species": table.species,
                    "bin_index": int(i),
                    "variance": float(mc_err[i] * mc_err[i]),
                }
            )
            combined_entries.append(
                {
                    "species": table.species,
                    "bin_index": int(i),
                    "variance": float(combined[i] * combined[i]),
                }
            )
    return {
        "convention": {
            "V_data": (
                "diagonal; per-bin squared statistical error of the SPS "
                "table (converted to f / sigma_inel units)"
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


def residuals_payload(
    tables: list[BinnedFitTable],
    values_by_species: dict[str, np.ndarray],
    mc_error_by_species: dict[str, np.ndarray],
) -> list[dict]:
    """Per-bin residuals ``(d - m) / sigma`` with the combined sigma."""
    entries: list[dict] = []
    for table in tables:
        mc_err = np.asarray(mc_error_by_species[table.species], dtype=float)
        m = np.asarray(values_by_species[table.species], dtype=float)
        sigma = combined_sigma(table.sigma, mc_err)
        d = np.asarray(table.data, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            pulls = (d - m) / sigma
        # An empty MC bin carries an infinite sigma; its pull is 0.0 by
        # the infinite-MC-error convention (recorded, never dropped).
        pulls = np.where(np.isfinite(pulls), pulls, 0.0)
        for i in range(table.data.size):
            entries.append(
                {
                    "species": table.species,
                    "bin_index": int(i),
                    "d": float(d[i]),
                    "m": float(m[i]),
                    "sigma": float(sigma[i]),
                    "delta_over_sigma": float(pulls[i]),
                }
            )
    return entries


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
    they enter the map only through ``empty_bins``.
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
        if abs(float(entry["delta_over_sigma"])) > float(threshold)
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
    return {
        "flagged_cells": flagged,
        "threshold": float(threshold),
        "n_flagged": len(flagged),
        "n_total_bins": TOTAL_BINS,
        "empty_bins": empty_bins,
        "n_empty_bins": len(empty_bins),
        "convention": (
            "pull = (d - m) / sigma with sigma = sqrt(V_data_bin + V_mc_bin); "
            "a bin with N_bin == 0 carries an infinite MC error, its pull is "
            "0.0, and it is recorded under empty_bins instead of being "
            "silently dropped; bin_index is the 0-based position of the bin "
            "in its species' windowed table order"
        ),
    }


CHI2_DOF_CONVENTION = "bins - 1 (card R01; R01 fits no parameter)"


def chi2_payload(
    residuals: list[dict],
    counts_by_species: Mapping[str, list | np.ndarray],
) -> dict:
    """Per-species chi-square over that species' bins with DOF bins - 1."""
    per_species: dict[str, dict] = {}
    total_chi2 = 0.0
    total_bins = 0
    for species in SPECIES_DISPLAY:
        entries = [e for e in residuals if e["species"] == species]
        if not entries:
            continue
        n_bins = len(entries)
        counts = np.asarray(counts_by_species[species], dtype=np.int64)
        n_empty = sum(1 for e in entries if int(counts[int(e["bin_index"])]) == 0)
        chi2_value = float(sum(float(e["delta_over_sigma"]) ** 2 for e in entries))
        dof = n_bins - 1
        per_species[species] = {
            "chi2": chi2_value,
            "dof": int(dof),
            "chi2_dof": float(chi2_value / dof) if dof > 0 else float("nan"),
            "n_bins": n_bins,
            "n_empty_bins": n_empty,
        }
        total_chi2 += chi2_value
        total_bins += n_bins
    return {
        "dof_convention": CHI2_DOF_CONVENTION,
        "per_species": per_species,
        "total_chi2": float(total_chi2),
        "total_bins": int(total_bins),
        "total_dof": int(total_bins - 1) if total_bins > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Nuisance width (card 7 config), provenance, seal
# ---------------------------------------------------------------------------


def read_sigma_eta() -> float:
    """The card-7 normalization-nuisance width from the committed config.

    ``config/v2_singlet/likelihood_nuisances.json`` declares one nuisance
    per SPS table, each with the same ``sigma_eta``; the shared width is
    used for the driver's single ``all_six_tables`` nuisance.  A mixed
    set of widths refuses the run (use the existing value; a different
    convention files its own card).
    """
    path = NESTED_ROOT / "config" / "v2_singlet" / "likelihood_nuisances.json"
    if not path.is_file():
        _refuse(f"card 7 nuisance config missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    nuisances = payload.get("nuisances") or []
    widths = {float(entry["sigma_eta"]) for entry in nuisances}
    if len(widths) != 1:
        _refuse(
            f"card 7 nuisance config carries mixed sigma_eta widths {widths!r}; "
            "the R01 normalization nuisance records the shared card-7 width"
        )
    width = widths.pop()
    if not (math.isfinite(width) and width > 0.0):
        _refuse(f"card 7 sigma_eta must be positive and finite, got {width!r}")
    return width


def staged_generation_code_fingerprint() -> str:
    """sha256 over the installed staged-generation sources plus the salt.

    Hash inputs, in order: canonical tagged source bytes of the modules
    in :data:`FINGERPRINT_MODULES`, the ``RESONANCE_DECAY_LIB_VERSION``
    salt, and this driver's own source bytes.
    """
    import importlib

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
    seeds: list[int],
    replicas_requested: int,
    replicas_completed: int,
    events_per_replica: int,
    events_completed: int,
    argv: list[str],
    wall_seconds: float,
    tier_status: str,
    tier_deferral_reason: str | None,
    input_paths: list[Path],
) -> dict:
    """Write ``provenance.json`` via the ``run_provenance.py`` convention.

    The record carries the config hash, the M8 parameter hash, the full
    seed list, the replica/event counts, and the code fingerprint
    (``RESONANCE_DECAY_LIB_VERSION`` salt).  ``write_run_provenance``
    writes ``run_provenance.json``; the file is renamed to the card's
    ``provenance.json`` name.
    """
    fingerprint = staged_generation_code_fingerprint()
    extra = {
        "card": CARD,
        "run": RUN_DESCRIPTION,
        "gate_clause": GATE_CLAUSE,
        "config_sha256": verification["config_sha256"],
        "m8_parameter_hash": verification["m8_parameter_hash"],
        "sigma_inel_mb": SIGMA_INEL_MB,
        "seed_base": int(pre_registration["seed_base"]),
        "seed_offset": int(pre_registration["seed_offset"]),
        "replica_seeds": [int(s) for s in seeds],
        "replicas_requested": int(replicas_requested),
        "replicas_completed": int(replicas_completed),
        "events_per_replica": int(events_per_replica),
        "events_completed": int(events_completed),
        "tier": dict(pre_registration["tier"]),
        "tier_status": tier_status,
        "tier_deferral_reason": tier_deferral_reason,
        "observable_set": dict(pre_registration["observable_set"]),
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
            "type": EVENT_SEED_RULE,
            "seeds": {
                "seed_base": int(pre_registration["seed_base"]),
                "seed_offset": int(pre_registration["seed_offset"]),
                "replica_seeds": [int(s) for s in seeds],
            },
        },
        source_record=collect_source_record(
            outer_repo=MONO_ROOT, nested_repo=NESTED_ROOT
        ),
        producer="staged_r01_reference.py",
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
    seal manifest instead), and ``gate0_r01.json`` (it carries the
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
    ``provenance.json``, and ``gate0_r01.json`` when present), each with
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
            "seal_manifest.json, provenance.json, gate0_r01.json); "
            "seal_manifest.json records the payload files plus "
            "SHA256SUMS.txt, provenance.json, and gate0_r01.json; "
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


# ---------------------------------------------------------------------------
# Acceptance record
# ---------------------------------------------------------------------------


def tier_status_decision(replicas_completed: int, declared_replicas: int) -> str:
    """Classify the replica count against the card's acceptance rule.

    The card's acceptance output fixes the tier semantics: ``tier_status``
    is ``complete`` iff ``replicas_completed == declared_replicas`` (the
    declared T3 count of 20); otherwise it is ``deferred`` with a
    non-null deferral reason.  An over-declared run with more replicas
    than the declared tier is therefore also ``deferred``, since it is
    not the declared tier.  The per-replica manifest and provenance
    record and the gate0 acceptance record both classify through this
    single predicate, so the two records cannot disagree.
    """
    return "complete" if replicas_completed == declared_replicas else "deferred"


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
    """Assemble ``gate0_r01.json`` from the sealed run's artifacts.

    The record matches the card's acceptance schema exactly.  The verdict
    is ``pass`` only when the hashes verified, the archive is sealed and
    self-consistent, both clean-environment commands recorded ``pass``,
    and the observable set carries exactly 1017 bins.  ``tier_status`` is
    ``complete`` iff the declared 20 replicas completed; otherwise
    ``deferred`` with the recorded reason (a deferral does not block
    Gate 0, but the archive must exist).
    """
    out_dir = Path(out_dir)
    prereg_path = out_dir / PREREGISTRATION_NAME
    if not prereg_path.is_file():
        _refuse(f"pre-registration missing: {prereg_path}")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))

    manifest_path = out_dir / REPLICA_MANIFEST_NAME
    if not manifest_path.is_file():
        _refuse(f"replica manifest missing: {manifest_path}")
    replica_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completed = [
        entry
        for entry in replica_manifest.get("seeds", [])
        if entry.get("status") == "completed"
    ]
    replicas_completed = len(completed)
    events_completed = int(sum(int(e.get("n_events", 0)) for e in completed))

    predictions_path = out_dir / BIN_PREDICTIONS_NAME
    if not predictions_path.is_file():
        _refuse(f"bin predictions missing: {predictions_path}")
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    n_total_bins = len(predictions)

    failure_path = out_dir / FAILURE_MAP_NAME
    if not failure_path.is_file():
        _refuse(f"failure map missing: {failure_path}")
    failure_map = json.loads(failure_path.read_text(encoding="utf-8"))

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
    # itself is written after the acceptance record (gate0_r01.json
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

    tier = dict(prereg.get("tier") or {})
    tier["replicas_completed"] = int(replicas_completed)
    tier["events_completed"] = int(events_completed)
    declared_replicas = int(prereg.get("replicas_t3", 0))
    tier_status = tier_status_decision(replicas_completed, declared_replicas)
    if tier_status == "complete":
        tier_deferral_reason = None
    else:
        tier_deferral_reason = replica_manifest.get("tier_deferral_reason") or (
            f"{replicas_completed} of {declared_replicas} declared T3 "
            f"replicas completed ({events_completed} of "
            f"{declared_replicas * int(prereg.get('events_per_replica_t3', 0))} "
            "events); the remaining replicas are a follow-up obligation"
        )

    verdict = (
        "pass"
        if (
            config_hash_verified
            and m8_verified
            and archive_sealed
            and immutable
            and clean_environment_commands[CLEAN_ENV_PYTEST_KEY] == "pass"
            and clean_environment_commands[CLEAN_ENV_SMOKE_KEY] == "pass"
            and n_total_bins == TOTAL_BINS
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
        "tier": tier,
        "tier_status": tier_status,
        "tier_deferral_reason": tier_deferral_reason,
        "n_total_bins": int(n_total_bins),
        "n_flagged_cells": int(failure_map.get("n_flagged", 0)),
        "empty_bins": int(failure_map.get("n_empty_bins", 0)),
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
    ``--out dev/00runs/r01_legacy_pp_reference_2026-09-11``, so an
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
            "R01 legacy p+p reference run driver: pooled seeded replicas of "
            "the staged p+p two-source builder at the frozen M8 set, the "
            "1017-bin six-species binned prediction with the combined "
            "covariance, residuals, failure map, and the sealed immutable "
            "negative-control archive"
        )
    )
    parser.add_argument(
        "--config", required=True, help="bundled legacy_v1 config JSON"
    )
    parser.add_argument(
        "--replicas",
        type=int,
        required=True,
        help="number of replicas to generate (the declared T3 tier is 20)",
    )
    parser.add_argument(
        "--events-per-replica",
        type=int,
        required=True,
        help="events per replica (the declared T3 tier is 250,000)",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=SEED_BASE,
        help=(
            f"replica seed base (default {SEED_BASE}); seeds are "
            "seed_base + 40000 + k"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="worker processes for chunked generation (default 4)",
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
            "(and, for the production config run, gate0_r01.json) after the "
            "binned artifacts"
        ),
    )
    return parser


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
    out_dir = resolve_out_dir(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload_expected = default_pre_registration()
    prereg_path = out_dir / PREREGISTRATION_NAME
    if prereg_path.is_file():
        loaded = json.loads(prereg_path.read_text(encoding="utf-8"))
        if loaded != payload_expected:
            _refuse(
                f"pre-registration mismatch: {prereg_path} does not equal the "
                "R01 pre-registration (seed rule, tier, hashes, normalization, "
                "and the 1017-bin observable set are pre-registered; a "
                "different registration files its own card)"
            )
    else:
        _write_json(prereg_path, payload_expected)
    pre_registration = payload_expected

    config_path = Path(args.config)
    verification = verify_config_and_m8(config_path, pre_registration)
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    build_pp_config(raw_config)  # refuse before any generation on a bad application

    print(
        f"[r01] config {config_path} sha256 {verification['config_sha256']}",
        flush=True,
    )
    print(
        f"[r01] M8 parameter hash {verification['m8_parameter_hash']} verified "
        f"against {verification['manifest_path']}",
        flush=True,
    )

    tables = load_binned_tables(raw_config)
    table_digest = tables_digest(tables)
    sigma_eta = read_sigma_eta()
    sqrt_s_gev = math.sqrt(
        Reaction(float(raw_config.get("beam_momentum", 158.0)), 2212, 2212).s()
    )
    print(
        f"[r01] observable set: {sum(int(t.data.size) for t in tables)} bins "
        f"over {len(tables)} species; sigma_inel {SIGMA_INEL_MB} mb; "
        f"sqrt(s) {sqrt_s_gev:.6f} GeV; sigma_eta {sigma_eta}",
        flush=True,
    )

    seeds = replica_seeds(args.seed_base, args.replicas)
    payload = config_payload(raw_config)
    started = time.perf_counter()
    replica_records = []
    for index, seed in enumerate(seeds, start=1):
        print(
            f"[r01] replica {seed} ({index}/{len(seeds)}), "
            f"{args.events_per_replica} events",
            flush=True,
        )
        replica_records.append(
            run_replica(
                seed,
                index - 1,
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
    wall = time.perf_counter() - started

    replicas_completed = len(replica_records)
    events_completed = sum(int(r["n_events"]) for r in replica_records)
    declared_replicas = int(payload_expected["replicas_t3"])
    tier_status = tier_status_decision(replicas_completed, declared_replicas)
    tier_deferral_reason = (
        None
        if tier_status == "complete"
        else (
            f"{replicas_completed} of {declared_replicas} declared T3 replicas "
            f"completed ({events_completed} of "
            f"{declared_replicas * int(payload_expected['events_per_replica_t3'])} "
            "events); the remaining replicas are a follow-up obligation"
        )
    )

    # Pooled per-replica weight sums and track counts.
    totals: dict[str, np.ndarray] = {}
    count_totals: dict[str, np.ndarray] = {}
    for record in replica_records:
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
    entries, values_by_species, errors_by_species = bin_predictions_payload(
        tables, totals, count_totals, events_completed, sqrt_s_gev
    )
    _write_json(out_dir / BIN_PREDICTIONS_NAME, entries)

    covariance = covariance_payload(tables, errors_by_species, sigma_eta)
    _write_json(out_dir / COVARIANCE_NAME, covariance)

    residuals = residuals_payload(tables, values_by_species, errors_by_species)
    _write_json(out_dir / RESIDUALS_NAME, residuals)

    failure_map = build_failure_map(residuals, count_totals)
    _write_json(out_dir / FAILURE_MAP_NAME, failure_map)

    chi2 = chi2_payload(residuals, count_totals)
    _write_json(out_dir / CHI2_NAME, chi2)

    # Per-replica seed archive with completed/pending status over the
    # declared T3 seed list.
    completed_by_seed = {int(r["seed"]): r for r in replica_records}
    declared_seeds = list(payload_expected["replica_seeds"])
    archive_entries = []
    for k, seed in enumerate(declared_seeds):
        record = completed_by_seed.get(int(seed))
        archive_entries.append(
            {
                "seed": int(seed),
                "replica_index": int(k),
                "status": "completed" if record is not None else "pending",
                "n_events": int(record["n_events"]) if record else None,
                "wall_seconds": record["wall_seconds"] if record else None,
                "event_hash_set_sha256": (
                    record["event_hash_set_sha256"] if record else None
                ),
                "record": f"replica_{int(seed)}.json" if record else None,
            }
        )
    requested_entries = []
    for k, seed in enumerate(seeds):
        record = completed_by_seed[int(seed)]
        requested_entries.append(
            {
                "seed": int(seed),
                "replica_index": int(k),
                "status": "completed",
                "n_events": int(record["n_events"]),
                "wall_seconds": record["wall_seconds"],
                "event_hash_set_sha256": record["event_hash_set_sha256"],
                "record": f"replica_{int(seed)}.json",
            }
        )
    _write_json(
        out_dir / REPLICA_MANIFEST_NAME,
        {
            "seed_base": int(args.seed_base),
            "seed_offset": SEED_OFFSET,
            "requested_replicas": int(args.replicas),
            "requested_events_per_replica": int(args.events_per_replica),
            "declared_t3": {
                "replicas": declared_replicas,
                "events_per_replica": int(
                    payload_expected["events_per_replica_t3"]
                ),
                "seeds": declared_seeds,
            },
            "tier_status": tier_status,
            "tier_deferral_reason": tier_deferral_reason,
            "seeds": requested_entries,
            "t3_seed_archive": archive_entries,
            "config_sha256": verification["config_sha256"],
            "tables_sha256": table_digest,
            "wall_seconds": round(wall, 3),
            "started_utc": (
                replica_records[0]["started_utc"] if replica_records else _utc_now_iso()
            ),
            "finished_utc": _utc_now_iso(),
        },
    )

    print(
        f"[r01] pooled prediction: {events_completed} events over "
        f"{replicas_completed} replica(s); total_bins {len(entries)}; "
        f"n_flagged {failure_map['n_flagged']}; empty_bins "
        f"{failure_map['n_empty_bins']}; chi2 "
        + ", ".join(
            f"{species} {per['chi2']:.2f}"
            for species, per in chi2["per_species"].items()
        ),
        flush=True,
    )

    exit_code = 0
    if args.seal:
        input_paths = [
            Path(verification["config_path"]),
            Path(verification["manifest_path"]),
            NESTED_ROOT / "config" / "v2_singlet" / "likelihood_nuisances.json",
        ] + [Path(t.path) for t in tables]
        write_provenance(
            out_dir,
            verification=verification,
            pre_registration=payload_expected,
            seeds=seeds,
            replicas_requested=int(args.replicas),
            replicas_completed=replicas_completed,
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
        if verification["production_config"]:
            gate = assemble_gate0(out_dir, verification, payload_expected)
        # The seal manifest is written after the acceptance record so it
        # covers gate0_r01.json too (gate0 carries the sums digest, so
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
            payload = out_dir / name
            if not payload.is_file() or sha256_file(payload) != digest:
                sums_ok = False
                break
        if not seal_ok or not sums_ok:
            print(
                "[r01] seal verification FAILED; the run directory is not "
                "immutable as written",
                flush=True,
            )
            return 1
        print(
            f"[r01] run directory sealed: {out_dir} "
            f"({len(seal['files'])} sealed files);"
            + (
                f" gate0_r01.json verdict {gate['verdict']}"
                if gate is not None
                else " no gate record (non-production config)"
            ),
            flush=True,
        )
        if gate is not None and gate["verdict"] != "pass":
            exit_code = 1
    else:
        print(
            "[r01] binned artifacts written; rerun with --seal to write "
            "provenance.json, SHA256SUMS.txt, seal_manifest.json, and the "
            "gate0_r01.json acceptance record",
            flush=True,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
