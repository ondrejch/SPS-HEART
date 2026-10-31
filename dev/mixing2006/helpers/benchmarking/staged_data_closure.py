#!/usr/bin/env python3
"""R10 data pipeline closure driver (Round 1, card 12).

Runs the pre-registered data inventory + pipeline closure benchmark R10 of
plan section 7 and writes its acceptance record ``gate1_r10.json`` (Gate 1
clause ``data_provenance_and_pipeline_closure``: proceed only when every
datum to be fit has provenance, normalization convention, feeddown
convention, and uncertainty treatment).  Two stages, single phase:

- ``--mode inventory``: the §7.1 data inventory ``data_inventory.json``.
  One record per data table with the card's §7.1 field list verbatim
  (``table``, ``experiment``, ``paper``, ``beam_momentum_gev_c``,
  ``sqrt_s_gev``, ``event_class``, ``trigger_definition``,
  ``normalization_convention``, ``species``, ``feeddown_convention``,
  ``bin_edges``, ``units``, ``uncertainty_treatment``, ``covariance``,
  ``acceptance_correction``, ``split``).  The inventory covers the six SPS
  158 GeV/c p+p tables of the bundled config's ``fit.binned`` block (pi-,
  pi+, K+, K-, p, pbar; the 1017-bin training set) and the twelve SPS p+Pb
  per-event invariant density tables of Chvala 2006 under
  ``helpers/benchmarking/pa_data/`` (minimum-bias plus centrality sets
  1..5, pi- and pi+; the R02 data side, held out per the M13 convention).
  Every ``bin_edges`` value is read from the actual table: the midpoint
  edges (``sps_core.edges_from_centers``) of the paper-window tabulated
  centers, i.e. exactly the binning the pipeline consumes.  The covariance
  and data-inventory conventions are exercised through
  ``na49_mixing.fit_likelihood`` (``BinnedObservable``,
  ``NormalizationNuisance``, ``diagonal_covariance``): the card-7 config
  ``config/v2_singlet/likelihood_nuisances.json`` declares one correlated
  normalization nuisance per SPS table with the shared width ``sigma_eta``
  and ``use_data_covariance = false`` (Round 1), so every table's data
  covariance is the per-bin statistical diagonal and the normalization
  uncertainty is the nuisance.

- ``--mode closure``: the pipeline closure against three analytic toy
  spectra with closed-form bin integrals (the toy spectra are mathematical
  functions, not a physics model; no ``PPEventBuilder`` /
  ``PAEventBuilder`` generation runs here):

  1. separable power law ``f = A (1 - xF)^a exp(-b pT)``, A=1, a=2, b=1;
     bin integral ``A [(1-xF_lo)^(a+1) - (1-xF_hi)^(a+1)]/(a+1) *
     [exp(-b pT_lo) - exp(-b pT_hi)]/b`` (the xF and pT factors);
  2. 2-D Gaussian ``f = A exp(-0.5 ((xF-mu_xF)/sig_xF)^2)
     exp(-0.5 ((pT-mu_pT)/sig_pT)^2)``, A=1, mu_xF=0.2, sig_xF=0.1,
     mu_pT=0.5, sig_pT=0.3; bin integral ``A sig_xF sig_pT (pi/2) DE_xF
     DE_pT`` with ``DE = erf((hi-mu)/(sig sqrt(2))) - erf((lo-mu)/(sig
     sqrt(2)))`` (the product of two 1-D Gaussian CDF differences via the
     error function; exact, no numeric quadrature);
  3. uniform ``f = A``, A=1; bin integral ``A (xF_hi-xF_lo)(pT_hi-pT_lo)``.

  For each toy the *model* side is the direct per-bin closed-form value
  ``I_bin / (dxF dpT sigma_inel)`` with ``sigma_inel = 31.78 mb`` (the
  M4/M8 conversion).  Two paths are compared against it, both on the
  1017-bin six-species observable set (the six SPS tables' windowed grids,
  sparse picks 229/230/153/148/135/122):

  - the **analytic-integration path** (the Gate 1 criterion): the exact
    bin integrals are fed through the pipeline's binning / normalization /
    conversion code path — the full-grid integral array in the pT-major
    flat layout, picked with ``bin_index_pt * n_x + bin_index_xf``, the
    exact expected weight sums ``S_bin = pi sqrt(s) N I_bin / sigma_inel``
    divided through the estimator formula ``M_bin = S_bin / (pi sqrt(s) N
    dxF dpT)`` — and must reproduce the model side to
    ``max_rel_deviation < 1e-6`` per toy (the arithmetic identity; the
    tolerance guards against float-rounding drift).
  - the **MC-sampling path** (a sanity check, not the Gate 1 criterion):
    a deterministic toy generator draws ``n_events`` toy events, each a
    Poisson(Lambda) draw of tracks whose (xF, pT) follow the track density
    implied by the declared invariant density, ``rho = (pi sqrt(s) /
    sigma_inel) f(xF, pT) pT / E`` with ``E = sqrt(m^2 + pL^2 + pT^2)`` and
    ``pL = xF sqrt(s)/2`` (the standard Jacobian relation between a
    per-event track density and the invariant cross section).  The tracks
    carry the pi- rest mass, which cancels exactly in the closure identity
    because the same mass enters the sampler density and the pipeline's
    E/pT weight.  ``Lambda`` (the mean tracks per event) is the
    generator's multiplicity bookkeeping, computed once per toy on a fixed
    midpoint grid; its discretization error enters the MC expectation only
    as the global factor ``Lambda_hat / Lambda``, orders of magnitude
    below the tolerance floor.  The tracks are binned and normalized
    through the exact production arithmetic (the ``_bin_tracks``
    convention: grid edges from the table centers, ``sps_core.grid_indices``,
    the in-bounds drop, the pT-major flat layout, ``M_bin = sum(E/pT) /
    (pi sqrt(s) N_events dxF dpT)``), and the per-bin relative deviation
    against the model side must satisfy the pre-registered tolerance
    ``max(3 / sqrt(N_bin), 1e-3)`` for at least 95% of the non-empty bins.
    Bins with ``N_bin == 0`` are excluded from the tolerance check and
    counted under ``empty_bins``.  The sampler draws on the union of the
    six grids' edge spans (xF [-0.0125, 0.45], pT [0.025, 1.6] for the
    bundled tables), so every grid bin's full rectangle is covered and the
    per-bin identity holds without boundary clipping; the paper window and
    the 1017-bin observable set themselves are unchanged (a different
    binning files its own card).

- ``--mode all``: both, in order; with ``--seal`` the run directory is
  sealed and ``gate1_r10.json`` is assembled.

Pre-registration: ``pre_registration.json`` is written into ``--out``
before any work; on re-run it must equal the registered payload exactly
(the three toy spectra, the MC seed 20303000 — the card's pre-registration
schema and both test sections agree on 20303000; the card's prose
arithmetic "20260911 + 30000" is an internal slip and is not used —,
1,000,000 registered events, the tolerance rule, the 1e-6 analytic
tolerance, and the 1017-bin observable set).  ``--mc-events`` scales the
executed MC sample (the smoke command uses 5000); the registered and
executed counts are both recorded in ``closure_report.json``.

Output files under ``--out``: ``pre_registration.json``,
``data_inventory.json``, ``closure_report.json``, and with ``--seal`` the
immutable-archive triple ``provenance.json`` (via the ``run_provenance.py``
convention), ``SHA256SUMS.txt`` (sha256sum -c format, sorted, every
regular file under the run directory except the seal/provenance/acceptance
files, which cannot carry their own digests), ``gate1_r10.json``
(assembled before the seal manifest so the seal covers it), and
``seal_manifest.json`` (records the payload files plus ``SHA256SUMS.txt``,
``provenance.json``, and ``gate1_r10.json``; it cannot contain its own
digest).  The gate verdict is ``pass`` iff the inventory is complete (all
§7.1 fields non-null for every table), every toy's
``max_rel_deviation_analytic < 1e-6``, and every toy's
``mc_sanity_pass_fraction >= 0.95``; a single analytic deviation >= 1e-6
or a missing inventory field flips the verdict to ``fail`` and blocks
Gate 1.

Exit codes: 0 when every executed assertion holds, 2 when a verification
refusal stops the run before any work (the refusal text goes to stderr),
1 when a post-work assertion or the gate verdict fails.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
from scipy.special import erf  # the closed-form Gaussian bin integrals

# ---------------------------------------------------------------------------
# Tree location: this driver runs from the nested tree
# (dev/mixing2006/helpers/benchmarking/) and from its byte-identical
# monorepo mirror (helpers/benchmarking/).  Both must find the
# na49_mixing package, the bundled config, and the data tables.
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
    """Print a refusal to stderr and exit with code 2 before any work."""
    print(f"staged_data_closure: {message}", file=sys.stderr, flush=True)
    raise SystemExit(2)


NESTED_ROOT, MONO_ROOT = _locate_roots()
if str(NESTED_ROOT) not in sys.path:
    sys.path.insert(0, str(NESTED_ROOT))

from helpers.benchmarking import sps_core  # noqa: E402
from helpers.benchmarking.run_provenance import (  # noqa: E402
    RUN_PROVENANCE_FILENAME,
    collect_source_record,
    validate_run_provenance,
    write_run_provenance,
)
from na49_mixing.fit_likelihood import (  # noqa: E402
    BinnedObservable,
    NormalizationNuisance,
    diagonal_covariance,
)
from na49_mixing.reaction import Reaction  # noqa: E402
from na49_mixing.staged_evaluation import (  # noqa: E402
    BinnedFitTable,
    preflight_binned_tables,
)

# ---------------------------------------------------------------------------
# Run identity (card TASK-20260911-r1-run-r10-data-pipeline-closure)
# ---------------------------------------------------------------------------

CARD = "TASK-20260911-r1-run-r10-data-pipeline-closure"
RUN_DESCRIPTION = "r10_data_pipeline_closure_2026-09-11"
GATE_RECORD_NAME = "gate1_r10.json"
GATE_CLAUSE = "data_provenance_and_pipeline_closure"

PREREGISTRATION_NAME = "pre_registration.json"
INVENTORY_NAME = "data_inventory.json"
CLOSURE_REPORT_NAME = "closure_report.json"
SHA256SUMS_NAME = "SHA256SUMS.txt"
SEAL_NAME = "seal_manifest.json"
PROVENANCE_NAME = "provenance.json"

SIGMA_INEL_MB = 31.78
# The protocol-quoted c.m. energy (benchmark-protocol.md; the card's §7.1
# value for the SPS tables).  The closure arithmetic itself uses the full
# pipeline convention sqrt(s) = Reaction(158.0, 2212, 2212).s()**0.5.
SQRT_S_QUOTED_GEV = 17.270203
BEAM_MOMENTUM_GEV_C = 158.0

# The MC seed of the pre-registration and of both card test sections (the
# durable record).  The card's Public-interface prose computes
# "20260911 + 30000" = 20290911, contradicting its own pre-registration
# schema and test sections; the pre-registration governs.
MC_SEED = 20303000
DEFAULT_MC_EVENTS = 1_000_000
ANALYTIC_TOLERANCE = 1e-6
MC_TOLERANCE_RULE = "max(3 / sqrt(N_bin), 1e-3) per bin"
MC_PASS_FRACTION_MIN = 0.95
TOTAL_BINS = 1017
PAPER_WINDOW = {"xF": [0.0, 0.4], "pT": [0.05, 1.5]}
SPECIES_BIN_COUNTS = {
    "pi-": 229,
    "pi+": 230,
    "K+": 153,
    "K-": 148,
    "p": 135,
    "pbar": 122,
}
SPECIES_DISPLAY = {
    "pi_minus": "pi-",
    "pi_plus": "pi+",
    "k_plus": "K+",
    "k_minus": "K-",
    "proton": "p",
    "antiproton": "pbar",
}
SPECIES_CONFIG_KEY = {
    display: key for key, display in SPECIES_DISPLAY.items()
}

# Toy tracks carry the pi- rest mass; the mass cancels exactly in the
# closure identity (the same mass enters the sampler density and the
# pipeline's E/pT weight).  Documented bookkeeping, not a physics choice.
TOY_MASS_GEV = 0.1395699
# Midpoint-grid resolution of the sampler's Lambda bookkeeping integral.
LAMBDA_GRID = 2000

# The three pre-registered toy spectra (card schema verbatim).
TOY_SPECTRA: tuple[dict, ...] = (
    {"name": "separable_power_law", "A": 1.0, "a": 2.0, "b": 1.0},
    {
        "name": "gaussian_2d",
        "A": 1.0,
        "mu_xF": 0.2,
        "sig_xF": 0.1,
        "mu_pT": 0.5,
        "sig_pT": 0.3,
    },
    {"name": "uniform", "A": 1.0},
)

# ---------------------------------------------------------------------------
# §7.1 inventory metadata (values with in-repository provenance)
# ---------------------------------------------------------------------------

INVENTORY_FIELDS = (
    "table",
    "experiment",
    "paper",
    "beam_momentum_gev_c",
    "sqrt_s_gev",
    "event_class",
    "trigger_definition",
    "normalization_convention",
    "species",
    "feeddown_convention",
    "bin_edges",
    "units",
    "uncertainty_treatment",
    "covariance",
    "acceptance_correction",
    "split",
)

SPS_EXPERIMENT = "SPS"
SPS_PAPER = (
    "NA49 Collaboration, SPS HADRONS: Detailed Analysis of Soft Hadronic "
    "Interactions (data tables and documentation), "
    "https://spshadrons.web.cern.ch/ (accessed 2026-03-29; the "
    "latex/paper/references.bib entry spshadrons2026)"
)
SPS_EVENT_CLASS = "inelastic"
SPS_TRIGGER = "double-arm"
SPS_NORMALIZATION = "f / 31.78 mb"
SPS_FEEDDOWN = "as_published_inclusive"
SPS_ACCEPTANCE = "trigger_bias_included"
SPLIT_SPS = "training"

PAPB_PAPER = (
    "O. Chvala, PhD thesis, Charles University Prague (2006), thesis tables "
    "tabicall / tabiccs1..tabiccs5 via datables/ictable.NNN.*.dat, "
    "converted 2026-08-25 (helpers/benchmarking/pa_data/README.md)"
)
PAPB_EVENT_CLASS = "inelastic"
PAPB_TRIGGER_MINBIAS = (
    "all_centrality_triggered (minimum-bias-like triggered sample; thesis "
    "ictable.000 convention)"
)
PAPB_TRIGGER_SET = (
    "forward-energy centrality slice set_{n} (ring-calorimeter E_BE bands; "
    "thesis ictable.00{n} convention)"
)
PAPB_NORMALIZATION = (
    "per-event invariant density d(xF, pT) = f / sigma_inel (thesis p+Pb "
    "anchor sigma_inel = 1806 +/- 25 mb; consumed as the estimator's "
    "f / sigma_inel convention in R02)"
)
PAPB_FEEDDOWN = "as_published_feeddown_subtracted"
PAPB_ACCEPTANCE_MINBIAS = (
    "trigger_bias_included (the minimum-bias trigger acceptance is carried "
    "by the published per-event densities)"
)
PAPB_ACCEPTANCE_SET = (
    "trigger_bias_included; centrality slice selection included "
    "(forward-energy E_BE bands)"
)
SPLIT_PAPB = "held_out"

UNITS = "1/GeV^2"
UNCERTAINTY_TREATMENT = "per_bin_statistical + correlated_norm_nuisance"
COVARIANCE_TREATMENT = "none_published; diagonal + norm_nuisance"

FEEDDOWN_CONVENTION_QUOTE = (
    "benchmark-protocol.md: fit the published inclusive cross sections, do "
    "not add additional post-load feedover corrections; keep feed-down "
    "conventions aligned with the published data product"
)

CONFIG_RELPATH = "config/staged_excitation.json"
NUISANCES_RELPATH = "config/v2_singlet/likelihood_nuisances.json"


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


def _write_json(path: Path, payload: Any) -> None:
    Path(path).write_text(
        json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8"
    )


def _jsonable(value: Any) -> Any:
    """Plain-Python projection of numpy scalars/arrays for JSON output."""
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Pre-registration (card schema verbatim)
# ---------------------------------------------------------------------------


def default_pre_registration() -> dict:
    """The R10 pre-registration payload, exactly as the card defines it."""
    return {
        "toy_spectra": [dict(toy) for toy in TOY_SPECTRA],
        "mc_seed": MC_SEED,
        "mc_events": DEFAULT_MC_EVENTS,
        "mc_tolerance_rule": MC_TOLERANCE_RULE,
        "analytic_tolerance": ANALYTIC_TOLERANCE,
        "observable_set": {
            "window": {"xF": [0.0, 0.4], "pT": [0.05, 1.5]},
            "total_bins": TOTAL_BINS,
        },
    }


def pipeline_sqrt_s() -> float:
    """The pipeline's sqrt(s): Reaction(158.0, 2212, 2212).s()**0.5."""
    return math.sqrt(Reaction(BEAM_MOMENTUM_GEV_C, 2212, 2212).s())


# ---------------------------------------------------------------------------
# Observable set: the six SPS tables and their 1017 windowed bins
# ---------------------------------------------------------------------------


def sps_root() -> Path:
    """Root for the SPS HADRONS table paths (monorepo copy when present)."""
    if (MONO_ROOT / "spshadrons.web.cern.ch" / "data").is_dir():
        return MONO_ROOT
    return NESTED_ROOT


def sps_config_binned() -> dict:
    """The bundled config's fit.binned block (read-only; the M4 source)."""
    path = NESTED_ROOT / CONFIG_RELPATH
    if not path.is_file():
        _refuse(f"bundled config missing: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    binned = (raw.get("fit") or {}).get("binned") or {}
    if not binned or not binned.get("tables"):
        _refuse(f"{path} carries no fit.binned.tables block")
    sigma_inel = float(binned.get("sigma_inel_mb", SIGMA_INEL_MB))
    if sigma_inel != SIGMA_INEL_MB:
        _refuse(
            f"fit.binned.sigma_inel_mb {sigma_inel!r} != the M4/M8 "
            f"convention {SIGMA_INEL_MB!r}; a different sigma_inel "
            "convention files its own card"
        )
    window = dict(binned.get("window") or {})
    expected = {
        "xf_min": sps_core.PAPER_XF_MIN,
        "xf_max": sps_core.PAPER_XF_MAX,
        "pt_min": sps_core.PAPER_PT_MIN,
        "pt_max": sps_core.PAPER_PT_MAX,
    }
    for key, value in expected.items():
        if float(window.get(key, value)) != float(value):
            _refuse(
                f"fit.binned.window.{key} {window.get(key)!r} != the paper "
                f"window value {value!r}; a different window files its own card"
            )
    return binned


def load_sps_tables() -> list[BinnedFitTable]:
    """Load the windowed six-species tables and verify the 1017-bin set.

    The paths resolve against the SPS root (the monorepo copy when
    present), the normalization is the M4/M8 convention
    ``sigma_inel = 31.78 mb`` over the paper window, and the loaded
    per-species bin counts must equal the pre-registered observable set
    (pi- 229, pi+ 230, K+ 153, K- 148, p 135, pbar 122; total 1017).  A
    deviation refuses the run: a different observable set files its own
    card.
    """
    binned = sps_config_binned()
    root = sps_root()
    preflight_binned_tables(binned, root)
    window = dict(binned.get("window") or {})
    tables: list[BinnedFitTable] = []
    for entry in binned.get("tables", []):
        tables.append(
            BinnedFitTable.from_path(
                Path(root) / entry["path"],
                species=str(entry["species"]),
                pdg_id=int(entry["pdg"]),
                mass_gev=float(entry["mass_gev"]),
                normalization=str(entry["normalization"]),
                sigma_inel_mb=float(binned["sigma_inel_mb"]),
                window=window,
            )
        )
    registered = {SPECIES_CONFIG_KEY[k]: v for k, v in SPECIES_BIN_COUNTS.items()}
    counts = {t.species: int(t.data.size) for t in tables}
    if counts != registered or sum(counts.values()) != TOTAL_BINS:
        _refuse(
            f"loaded bin counts {counts!r} != the pre-registered 1017-bin "
            f"observable set {registered!r}; a different binning files its "
            "own card"
        )
    return tables


# ---------------------------------------------------------------------------
# The p+Pb tables (the R02 data side, helpers/benchmarking/pa_data/)
# ---------------------------------------------------------------------------

PA_DATA_DIRNAME = "pa_data"
PA_DATA_FILES = {
    "minimum_bias": {
        "pi-": "pPb_dens_minbias_piminus.txt",
        "pi+": "pPb_dens_minbias_piplus.txt",
    },
    "set_1": {
        "pi-": "pPb_dens_set1_piminus.txt",
        "pi+": "pPb_dens_set1_piplus.txt",
    },
    "set_2": {
        "pi-": "pPb_dens_set2_piminus.txt",
        "pi+": "pPb_dens_set2_piplus.txt",
    },
    "set_3": {
        "pi-": "pPb_dens_set3_piminus.txt",
        "pi+": "pPb_dens_set3_piplus.txt",
    },
    "set_4": {
        "pi-": "pPb_dens_set4_piminus.txt",
        "pi+": "pPb_dens_set4_piplus.txt",
    },
    "set_5": {
        "pi-": "pPb_dens_set5_piminus.txt",
        "pi+": "pPb_dens_set5_piplus.txt",
    },
}


def pa_data_dir() -> Path:
    return NESTED_ROOT / "helpers" / "benchmarking" / PA_DATA_DIRNAME


def _read_points(path: Path) -> np.ndarray:
    """Four-column (xF, pT, value, abs_error) spectra table (the R02
    ``staged_r02_reference._read_points`` loader convention)."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rows.append([float(x) for x in stripped.split()[:4]])
    return np.asarray(rows, dtype=float)


def _papb_windowed_edges(path: Path) -> dict:
    """The paper-window cell centers of one p+Pb table and their edges.

    R02's cell rules: the singular pT = 0.00 labels are excluded, cells
    with a non-positive error are excluded, and duplicated thesis pT
    labels merge onto one cell keyed by the (xF, pT) center rounded to
    1e-9.  The window is the paper window.  The recorded edges are the
    midpoint edges (``sps_core.edges_from_centers``) of the windowed
    unique cell centers.
    """
    points = _read_points(path)
    keys = {
        (round(float(xf), 9), round(float(pt), 9))
        for xf, pt, _value, err in points
        if pt > 0.0 and err > 0.0
    }
    if not keys:
        _refuse(f"no usable cells in {path}")
    xs = np.asarray(sorted({k[0] for k in keys}), dtype=float)
    ps = np.asarray(sorted({k[1] for k in keys}), dtype=float)
    xs_w = xs[
        (xs >= sps_core.PAPER_XF_MIN) & (xs <= sps_core.PAPER_XF_MAX)
    ]
    ps_w = ps[
        (ps >= sps_core.PAPER_PT_MIN) & (ps <= sps_core.PAPER_PT_MAX)
    ]
    if xs_w.size == 0 or ps_w.size == 0:
        _refuse(f"no {path.name} cells inside the paper window")
    n_window = sum(
        1
        for kx, kp in keys
        if sps_core.PAPER_XF_MIN <= kx <= sps_core.PAPER_XF_MAX
        and sps_core.PAPER_PT_MIN <= kp <= sps_core.PAPER_PT_MAX
    )
    return {
        "xF": [float(v) for v in sps_core.edges_from_centers(xs_w)],
        "pT": [float(v) for v in sps_core.edges_from_centers(ps_w)],
        "n_cells": len(keys),
        "n_window_cells": int(n_window),
    }


# ---------------------------------------------------------------------------
# Toy spectra: declared densities and closed-form bin integrals
# ---------------------------------------------------------------------------


def toy_density(toy: dict, xF, pT):
    """The declared invariant density f(xF, pT) of one toy spectrum."""
    name = toy["name"]
    amp = float(toy.get("A", 1.0))
    xv = np.asarray(xF, dtype=float)
    pv = np.asarray(pT, dtype=float)
    if name == "separable_power_law":
        return amp * np.power(1.0 - xv, float(toy["a"])) * np.exp(
            -float(toy["b"]) * pv
        )
    if name == "gaussian_2d":
        return amp * np.exp(
            -0.5 * np.square((xv - float(toy["mu_xF"])) / float(toy["sig_xF"]))
        ) * np.exp(
            -0.5 * np.square((pv - float(toy["mu_pT"])) / float(toy["sig_pT"]))
        )
    if name == "uniform":
        shape = np.broadcast(xv, pv).shape
        return np.full(shape, amp, dtype=float)
    _refuse(f"unknown toy spectrum {name!r}")


_SQRT2 = math.sqrt(2.0)


def toy_bin_integral(toy: dict, xlo, xhi, plo, phi):
    """The closed-form bin integral of the toy density over one bin.

    Exact elementary antiderivatives (no numeric quadrature),
    broadcast-capable over numpy arrays:

    - separable power law: the xF and pT integrals factorize,
      ``A [(1-xF_lo)^(a+1) - (1-xF_hi)^(a+1)]/(a+1)
      [exp(-b pT_lo) - exp(-b pT_hi)]/b``;
    - 2-D Gaussian: the product of two 1-D Gaussian CDF differences via
      the error function, ``A sig_xF sig_pT (pi/2) DE_xF DE_pT``;
    - uniform: ``A (xF_hi - xF_lo) (pT_hi - pT_lo)``.
    """
    name = toy["name"]
    amp = float(toy.get("A", 1.0))
    xlv = np.asarray(xlo, dtype=float)
    xhv = np.asarray(xhi, dtype=float)
    plv = np.asarray(plo, dtype=float)
    phv = np.asarray(phi, dtype=float)
    if name == "separable_power_law":
        a = float(toy["a"])
        b = float(toy["b"])
        x_part = (
            np.power(1.0 - xlv, a + 1.0) - np.power(1.0 - xhv, a + 1.0)
        ) / (a + 1.0)
        p_part = (np.exp(-b * plv) - np.exp(-b * phv)) / b
        return amp * x_part * p_part
    if name == "gaussian_2d":
        mx = float(toy["mu_xF"])
        sx = float(toy["sig_xF"])
        mp = float(toy["mu_pT"])
        sp = float(toy["sig_pT"])
        de_x = erf((xhv - mx) / (sx * _SQRT2)) - erf((xlv - mx) / (sx * _SQRT2))
        de_p = erf((phv - mp) / (sp * _SQRT2)) - erf((plv - mp) / (sp * _SQRT2))
        return amp * sx * sp * (math.pi / 2.0) * de_x * de_p
    if name == "uniform":
        return amp * (xhv - xlv) * (phv - plv)
    _refuse(f"unknown toy spectrum {name!r}")


def toy_envelope(toy: dict, domain: tuple[float, float, float, float]) -> float:
    """A provable upper bound of f(xF, pT) * pT / E over the sampler domain.

    The pT/E factor is maximized at pL = 0 (xF = 0, inside the domain) and
    pT at the domain's upper edge.  The density maximum is analytic per
    toy: the power law decreases in xF for xF < 1 (the domain's upper edge
    is below 1), the Gaussian mean is an interior domain point, and the
    uniform density is flat.
    """
    name = toy["name"]
    amp = float(toy.get("A", 1.0))
    xlo, _xhi, _plo, phi = domain
    if name == "separable_power_law":
        f_max = amp * (1.0 - xlo) ** float(toy["a"])
    elif name == "gaussian_2d":
        f_max = amp
    elif name == "uniform":
        f_max = amp
    else:
        _refuse(f"unknown toy spectrum {name!r}")
    w_max = phi / math.sqrt(TOY_MASS_GEV * TOY_MASS_GEV + phi * phi)
    envelope = f_max * w_max * (1.0 + 1e-12)
    if not (math.isfinite(envelope) and envelope > 0.0):
        _refuse(f"toy {name!r} envelope must be positive, got {envelope!r}")
    return envelope


# ---------------------------------------------------------------------------
# Closure grids (the six SPS windowed grids; the pipeline binning)
# ---------------------------------------------------------------------------


class ClosureGrid:
    """One species' windowed bin grid plus its sparse picked bins.

    The pick is the production pick ``bin_index_pt * n_x + bin_index_xf``
    over the pT-major flat grid of ``_bin_tracks``; the areas are the
    picked ``np.outer(np.diff(pedges), np.diff(xedges))`` cells.
    """

    def __init__(self, table: BinnedFitTable):
        self.species_display = SPECIES_DISPLAY[table.species]
        self.xedges = sps_core.edges_from_centers(table.xf_centers)
        self.pedges = sps_core.edges_from_centers(table.pt_centers)
        self.n_x = int(self.xedges.size) - 1
        self.n_p = int(self.pedges.size) - 1
        self.pick = np.asarray(
            table.bin_index_pt * self.n_x + table.bin_index_xf, dtype=np.int64
        )
        self.xf_mid = np.asarray(table.xf_centers[table.bin_index_xf], dtype=float)
        self.pt_mid = np.asarray(table.pt_centers[table.bin_index_pt], dtype=float)
        areas = np.outer(np.diff(self.pedges), np.diff(self.xedges)).ravel()
        self.areas_pick = np.asarray(areas[self.pick], dtype=float)

    @property
    def n_bins(self) -> int:
        return int(self.pick.size)

    def domain_span(self) -> tuple[float, float, float, float]:
        return (
            float(self.xedges[0]),
            float(self.xedges[-1]),
            float(self.pedges[0]),
            float(self.pedges[-1]),
        )


def closure_grids() -> list[ClosureGrid]:
    """The six loaded SPS grids with the 1017-bin sparse picks."""
    grids = [ClosureGrid(t) for t in load_sps_tables()]
    total = sum(g.n_bins for g in grids)
    if total != TOTAL_BINS:
        _refuse(
            f"closure grids pick {total} bins != the pre-registered "
            f"{TOTAL_BINS}"
        )
    return grids


def sampler_domain(
    grids: list[ClosureGrid],
) -> tuple[float, float, float, float]:
    """The union of the grids' edge spans (the sampler's support).

    Every grid bin's full rectangle lies inside this domain, so the
    per-bin identity holds without boundary clipping.
    """
    spans = np.asarray([g.domain_span() for g in grids], dtype=float)
    return (
        float(spans[:, 0].min()),
        float(spans[:, 1].max()),
        float(spans[:, 2].min()),
        float(spans[:, 3].max()),
    )


# ---------------------------------------------------------------------------
# Analytic path: exact integrals through the pipeline conversion chain
# ---------------------------------------------------------------------------


def analytic_closure(
    toy: dict,
    grids: list[ClosureGrid],
    sqrt_s: float,
    n_ref: int,
) -> dict:
    """The analytic-integration path for one toy (the Gate 1 criterion).

    Model side (the reference): the direct per-bin closed-form value
    ``I_bin / (dxF dpT sigma_inel)`` computed from the picked bin's own
    edges.  Pipeline side: the exact bin integrals are fed through the
    pipeline's binning / normalization / conversion code path — the
    full-grid integral array in the pT-major flat layout, picked with
    ``bin_index_pt * n_x + bin_index_xf``, converted to the exact expected
    weight sums ``S_bin = pi sqrt(s) N I_bin / sigma_inel`` and divided
    through the estimator formula ``M_bin = S_bin / (pi sqrt(s) N dxF
    dpT)``.  The deviation ``|pipeline - model| / model`` per bin is the
    arithmetic identity of the card's physics rule; the 1e-6 tolerance
    guards against float-rounding drift.
    """
    name = toy["name"]
    pi_sqrt_s_n = math.pi * float(sqrt_s) * float(n_ref)
    records: list[dict] = []
    for grid in grids:
        # Broadcast the edge pairs to the (n_p, n_x) cell grid; the flat
        # order is the pT-major layout ip * n_x + ix of _bin_tracks.
        xlo = grid.xedges[:-1][None, :]
        xhi = grid.xedges[1:][None, :]
        plo = grid.pedges[:-1][:, None]
        phi = grid.pedges[1:][:, None]
        integral_grid = np.asarray(
            toy_bin_integral(toy, xlo, xhi, plo, phi), dtype=float
        ).reshape(grid.n_p, grid.n_x).ravel()
        expected_sums = pi_sqrt_s_n * integral_grid / SIGMA_INEL_MB
        areas = np.outer(np.diff(grid.pedges), np.diff(grid.xedges)).ravel()
        pipeline_vals = expected_sums[grid.pick] / (
            pi_sqrt_s_n * areas[grid.pick]
        )
        for i in range(grid.n_bins):
            flat = int(grid.pick[i])
            ix = flat % grid.n_x
            ip = flat // grid.n_x
            xlo = float(grid.xedges[ix])
            xhi = float(grid.xedges[ix + 1])
            plo = float(grid.pedges[ip])
            phi = float(grid.pedges[ip + 1])
            i_direct = float(toy_bin_integral(toy, xlo, xhi, plo, phi))
            model = i_direct / ((xhi - xlo) * (phi - plo) * SIGMA_INEL_MB)
            pipeline = float(pipeline_vals[i])
            records.append(
                {
                    "species": grid.species_display,
                    "bin_index": int(i),
                    "xF_mid": float(grid.xf_mid[i]),
                    "pT_mid": float(grid.pt_mid[i]),
                    "model": model,
                    "analytic_pipeline": pipeline,
                    "rel_deviation_analytic": abs(pipeline - model)
                    / abs(model),
                }
            )
    max_dev = max(r["rel_deviation_analytic"] for r in records)
    return {
        "name": name,
        "params": dict(toy),
        "n_bins": len(records),
        "max_rel_deviation_analytic": float(max_dev),
        "n_ref": int(n_ref),
        "bins": records,
    }


# ---------------------------------------------------------------------------
# MC path: the deterministic toy generator through the pipeline
# ---------------------------------------------------------------------------


def _track_shape(
    toy: dict, xF: np.ndarray, pT: np.ndarray, sqrt_s: float
) -> np.ndarray:
    """The sampler shape f(xF, pT) * pT / E (the track-density shape)."""
    xv = np.asarray(xF, dtype=float)
    pv = np.asarray(pT, dtype=float)
    energy = np.sqrt(
        TOY_MASS_GEV * TOY_MASS_GEV + np.square(xv * (0.5 * float(sqrt_s)))
        + np.square(pv)
    )
    return toy_density(toy, xv, pv) * pv / np.maximum(energy, 1e-300)


def lambda_tracks_per_event(
    toy: dict, domain: tuple[float, float, float, float], sqrt_s: float
) -> float:
    """Mean tracks per event: ``(pi sqrt(s)/sigma_inel) * int f pT/E dxF dpT``.

    Deterministic midpoint Riemann sum on a fixed grid (generator
    bookkeeping only; its discretization error enters the MC expectation
    solely as the global factor Lambda_hat / Lambda).
    """
    xlo, xhi, plo, phi = domain
    gx = np.linspace(xlo, xhi, LAMBDA_GRID + 1)
    gp = np.linspace(plo, phi, LAMBDA_GRID + 1)
    xc = 0.5 * (gx[:-1] + gx[1:])
    pc = 0.5 * (gp[:-1] + gp[1:])
    total = 0.0
    block = 250
    for start in range(0, LAMBDA_GRID, block):
        X, P = np.meshgrid(xc, pc[start : start + block], indexing="ij")
        total += float(_track_shape(toy, X, P, float(sqrt_s)).sum())
    area = ((xhi - xlo) / LAMBDA_GRID) * ((phi - plo) / LAMBDA_GRID)
    return math.pi * float(sqrt_s) / SIGMA_INEL_MB * total * area


def _reject_sample(
    rng: np.random.Generator,
    toy: dict,
    domain: tuple[float, float, float, float],
    envelope: float,
    sqrt_s: float,
    n_target: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Deterministic rejection sampling of (xF, pT) from f * pT / E.

    Uniform proposals over the domain, acceptance ``shape / envelope``
    with the provable envelope of :func:`toy_envelope`; the accepted
    sample is i.i.d. from the normalized track density regardless of the
    multiplicity bookkeeping.  Deterministic under the passed generator.
    """
    xlo, xhi, plo, phi = domain
    out_x: list[np.ndarray] = []
    out_p: list[np.ndarray] = []
    collected = 0
    proposals = 0
    rate = 0.25
    while collected < n_target:
        remaining = n_target - collected
        n_prop = int(min(max(remaining / max(rate, 1e-3) * 1.25, 100_000), 4_000_000))
        u1 = rng.uniform(xlo, xhi, n_prop)
        u2 = rng.uniform(plo, phi, n_prop)
        u3 = rng.uniform(0.0, 1.0, n_prop)
        accept = u3 * envelope <= _track_shape(toy, u1, u2, float(sqrt_s))
        n_acc = int(np.count_nonzero(accept))
        if n_acc:
            take = min(n_acc, remaining)
            out_x.append(u1[accept][:take])
            out_p.append(u2[accept][:take])
            collected += take
        proposals += n_prop
        rate = collected / float(proposals)
    return np.concatenate(out_x), np.concatenate(out_p), proposals


def _bin_toy_tracks(
    xF: np.ndarray,
    pT: np.ndarray,
    sqrt_s: float,
    grid: ClosureGrid,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin the toy tracks on one grid (the ``_bin_tracks`` convention).

    Mirrors the production arithmetic exactly: grid edges from the table
    centers, ``sps_core.grid_indices``, the in-bounds drop, the pT-major
    flat layout, and the per-bin sums of the Jacobian weights ``E/pT`` and
    track counts picked with ``bin_index_pt * n_x + bin_index_xf``.  The
    toy tracks are built as four-vectors (pL = xF sqrt(s)/2, the pi- rest
    mass) and recovered as xF = 2 pz / sqrt(s), so the production xF
    reconstruction runs verbatim.
    """
    pz = np.asarray(xF, dtype=float) * (0.5 * float(sqrt_s))
    pt = np.asarray(pT, dtype=float)
    energy = np.sqrt(TOY_MASS_GEV * TOY_MASS_GEV + pz * pz + pt * pt)
    x_sel = 2.0 * pz / float(sqrt_s)
    w_sel = energy / pt
    ix, ip = sps_core.grid_indices(x_sel, pt, grid.xedges, grid.pedges)
    in_bounds = (ix >= 0) & (ix < grid.n_x) & (ip >= 0) & (ip < grid.n_p)
    flat = ip[in_bounds] * grid.n_x + ix[in_bounds]
    weights_flat = np.bincount(
        flat, weights=w_sel[in_bounds], minlength=grid.n_p * grid.n_x
    )[: grid.n_p * grid.n_x]
    counts_flat = np.bincount(flat, minlength=grid.n_p * grid.n_x)[
        : grid.n_p * grid.n_x
    ]
    return (
        np.asarray(weights_flat[grid.pick], dtype=float),
        np.asarray(counts_flat[grid.pick], dtype=np.int64),
    )


def mc_closure(
    toy: dict,
    toy_index: int,
    grids: list[ClosureGrid],
    model_by_species: dict[str, np.ndarray],
    sqrt_s: float,
    n_events: int,
    mc_seed: int,
) -> dict:
    """The MC-sampling path for one toy (the pre-registered sanity check).

    One seeded generator draws the total track count ``T ~
    Poisson(n_events * Lambda)`` and then the tracks i.i.d. from the
    normalized track density (rejection sampling from ``f pT/E``).  The
    binned, normalized prediction ``M_bin = sum(E/pT) / (pi sqrt(s)
    N_events dxF dpT)`` is compared against the model side with the
    pre-registered per-bin tolerance ``max(3 / sqrt(N_bin), 1e-3)``; bins
    with ``N_bin == 0`` are excluded and counted under ``empty_bins``.
    """
    domain = sampler_domain(grids)
    lam = lambda_tracks_per_event(toy, domain, float(sqrt_s))
    rng = np.random.default_rng(
        np.random.SeedSequence(int(mc_seed), spawn_key=(int(toy_index),))
    )
    n_tracks = int(rng.poisson(float(n_events) * lam))
    xF, pT, proposals = _reject_sample(
        rng, toy, domain, toy_envelope(toy, domain), float(sqrt_s), n_tracks
    )

    entries: list[dict] = []
    n_non_empty = 0
    n_empty = 0
    n_pass = 0
    max_dev_mc = 0.0
    for grid in grids:
        sums, counts = _bin_toy_tracks(xF, pT, float(sqrt_s), grid)
        model = model_by_species[grid.species_display]
        mc_vals = sums / (
            math.pi * float(sqrt_s) * float(n_events) * grid.areas_pick
        )
        for i in range(grid.n_bins):
            n_bin = int(counts[i])
            if n_bin == 0:
                n_empty += 1
                entries.append(
                    {
                        "species": grid.species_display,
                        "bin_index": int(i),
                        "mc": 0.0,
                        "n_bin": 0,
                        "rel_deviation_mc": None,
                        "tolerance": None,
                        "mc_pass": None,
                    }
                )
                continue
            rel = abs(float(mc_vals[i]) - float(model[i])) / abs(float(model[i]))
            tol = max(3.0 / math.sqrt(n_bin), 1e-3)
            passed = bool(rel <= tol)
            n_non_empty += 1
            if passed:
                n_pass += 1
            max_dev_mc = max(max_dev_mc, float(rel))
            entries.append(
                {
                    "species": grid.species_display,
                    "bin_index": int(i),
                    "mc": float(mc_vals[i]),
                    "n_bin": n_bin,
                    "rel_deviation_mc": float(rel),
                    "tolerance": float(tol),
                    "mc_pass": passed,
                }
            )
    pass_fraction = (
        float(n_pass) / float(n_non_empty) if n_non_empty > 0 else 0.0
    )
    empty_records = [
        {"species": e["species"], "bin_index": e["bin_index"]}
        for e in entries
        if e["n_bin"] == 0
    ]
    return {
        "name": toy["name"],
        "mc_events_used": int(n_events),
        "mc_seed": int(mc_seed),
        "lambda_tracks_per_event": float(lam),
        "mc_tracks": int(n_tracks),
        "sampler_proposals": int(proposals),
        "n_non_empty_bins": int(n_non_empty),
        "empty_bins": int(n_empty),
        "empty_bin_records": empty_records,
        "max_rel_deviation_mc": float(max_dev_mc),
        "mc_passes": int(n_pass),
        "mc_sanity_pass_fraction": pass_fraction,
        "bins": entries,
    }


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------


def read_card7_nuisances() -> dict:
    """The card-7 nuisance config (sigma_eta width, use_data_covariance).

    ``config/v2_singlet/likelihood_nuisances.json`` declares one nuisance
    per SPS table with the shared width ``sigma_eta`` and
    ``use_data_covariance = false`` in Round 1.  A mixed set of widths or
    an enabled data covariance refuses the run.
    """
    path = NESTED_ROOT / NUISANCES_RELPATH
    if not path.is_file():
        _refuse(f"card 7 nuisance config missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    nuisances = payload.get("nuisances") or []
    widths = {float(entry["sigma_eta"]) for entry in nuisances}
    if len(widths) != 1:
        _refuse(
            f"card 7 nuisance config carries mixed sigma_eta widths "
            f"{widths!r}; the inventory records the shared card-7 width"
        )
    width = widths.pop()
    if not (math.isfinite(width) and width > 0.0):
        _refuse(f"card 7 sigma_eta must be positive and finite, got {width!r}")
    use_data_covariance = bool(payload.get("use_data_covariance", False))
    if use_data_covariance:
        _refuse(
            "card 7 use_data_covariance must be false in Round 1 "
            "(diagonal + normalization nuisance)"
        )
    return {
        "sigma_eta": float(width),
        "use_data_covariance": use_data_covariance,
        "nuisances": [
            {
                "name": str(entry.get("name")),
                "sigma_eta": float(entry["sigma_eta"]),
                "table_scope": str(entry.get("table_scope")),
            }
            for entry in nuisances
        ],
    }


def build_inventory() -> dict:
    """The §7.1 data inventory (six SPS tables + twelve p+Pb tables)."""
    card7 = read_card7_nuisances()
    binned = sps_config_binned()
    tables = load_sps_tables()
    sqrt_s_pipeline = pipeline_sqrt_s()

    conventions = {
        "sigma_inel_mb": SIGMA_INEL_MB,
        "normalization_convention": (
            "f / sigma_inel with sigma_inel = 31.78 mb (the M4/M8 "
            "convention); the tabulated f values are divided by 31.78 mb "
            "on load (BinnedFitTable, normalization 'f')"
        ),
        "sigma_eta": card7["sigma_eta"],
        "use_data_covariance": card7["use_data_covariance"],
        "nuisances": card7["nuisances"],
        "nuisance_scope": (
            "one correlated normalization nuisance per SPS table "
            "(config/v2_singlet/likelihood_nuisances.json, card 7); the "
            "p+Pb tables are held out in Round 1 and carry no nuisance"
        ),
        "sqrt_s_quoted_gev": SQRT_S_QUOTED_GEV,
        "sqrt_s_pipeline_gev": sqrt_s_pipeline,
        "beam_momentum_gev_c": BEAM_MOMENTUM_GEV_C,
        "feeddown_convention": FEEDDOWN_CONVENTION_QUOTE,
        "bin_edges_convention": (
            "midpoint edges (sps_core.edges_from_centers) of the "
            "paper-window tabulated centers; for the SPS tables the "
            "windowed grid of the 1017-bin observable set, for the p+Pb "
            "tables the merged windowed cell centers (the R02 cell rules)"
        ),
        "split_convention": (
            "training = the six SPS p+p tables; held_out = the p+Pb "
            "tables (the M13 convention)"
        ),
        "covariance_module": (
            "na49_mixing.fit_likelihood (BinnedObservable / "
            "NormalizationNuisance / diagonal_covariance)"
        ),
        "observable_set": {
            "window": dict(PAPER_WINDOW),
            "species_bin_counts": dict(SPECIES_BIN_COUNTS),
            "total_bins": TOTAL_BINS,
        },
    }

    fit_observables: list[dict] = []
    records: list[dict] = []

    # The six SPS 158 GeV/c tables (the 1017-bin training set).
    for table in tables:
        display = SPECIES_DISPLAY[table.species]
        xedges = sps_core.edges_from_centers(table.xf_centers)
        pedges = sps_core.edges_from_centers(table.pt_centers)
        nuisance = NormalizationNuisance(
            name=f"sps_{table.species}_norm",
            eta=0.0,
            sigma_eta=card7["sigma_eta"],
            table_scope=table.species,
        )
        observable = BinnedObservable(
            species=display,
            observable="f_xsect",
            bin_edges=(
                tuple(float(v) for v in xedges),
                tuple(float(v) for v in pedges),
            ),
            values=tuple(float(v) for v in table.data),
            stat_errors=tuple(float(v) for v in table.sigma),
            cov=None,
            normalization_nuisance=nuisance.name,
        )
        v_data = diagonal_covariance(observable.stat_errors)
        fit_observables.append(
            {
                "species": display,
                "observable": observable.observable,
                "n_bins": int(table.data.size),
                "cov": None,
                "v_data_flattened_length": len(v_data),
                "normalization_nuisance": nuisance.name,
                "sigma_eta": nuisance.sigma_eta,
            }
        )
        records.append(
            {
                "table": f"spshadrons.web.cern.ch/data/{Path(table.path).name}",
                "experiment": SPS_EXPERIMENT,
                "paper": SPS_PAPER,
                "beam_momentum_gev_c": BEAM_MOMENTUM_GEV_C,
                "sqrt_s_gev": SQRT_S_QUOTED_GEV,
                "event_class": SPS_EVENT_CLASS,
                "trigger_definition": SPS_TRIGGER,
                "normalization_convention": SPS_NORMALIZATION,
                "species": display,
                "feeddown_convention": SPS_FEEDDOWN,
                "bin_edges": {
                    "xF": [float(v) for v in xedges],
                    "pT": [float(v) for v in pedges],
                },
                "units": UNITS,
                "uncertainty_treatment": UNCERTAINTY_TREATMENT,
                "covariance": COVARIANCE_TREATMENT,
                "acceptance_correction": SPS_ACCEPTANCE,
                "split": SPLIT_SPS,
            }
        )

    # The twelve SPS p+Pb per-event invariant density tables (the R02
    # data side, held out per the M13 convention).
    base = pa_data_dir()
    for class_name, species_files in PA_DATA_FILES.items():
        for display, filename in species_files.items():
            path = base / filename
            if not path.is_file():
                _refuse(f"p+Pb table missing: {path}")
            edges = _papb_windowed_edges(path)
            set_number = class_name.removeprefix("set_")
            is_minbias = class_name == "minimum_bias"
            trigger = (
                PAPB_TRIGGER_MINBIAS
                if is_minbias
                else PAPB_TRIGGER_SET.format(n=set_number)
            )
            acceptance = (
                PAPB_ACCEPTANCE_MINBIAS if is_minbias else PAPB_ACCEPTANCE_SET
            )
            fit_observables.append(
                {
                    "species": display,
                    "observable": "f_xsect",
                    "n_bins": int(edges["n_window_cells"]),
                    "cov": None,
                    "v_data_flattened_length": None,
                    "normalization_nuisance": None,
                    "sigma_eta": None,
                    "held_out": True,
                }
            )
            records.append(
                {
                    "table": (
                        "dev/mixing2006/helpers/benchmarking/pa_data/"
                        + filename
                    ),
                    "experiment": SPS_EXPERIMENT,
                    "paper": PAPB_PAPER,
                    "beam_momentum_gev_c": BEAM_MOMENTUM_GEV_C,
                    "sqrt_s_gev": SQRT_S_QUOTED_GEV,
                    "event_class": PAPB_EVENT_CLASS,
                    "trigger_definition": trigger,
                    "normalization_convention": PAPB_NORMALIZATION,
                    "species": display,
                    "feeddown_convention": PAPB_FEEDDOWN,
                    "bin_edges": {
                        "xF": list(edges["xF"]),
                        "pT": list(edges["pT"]),
                    },
                    "units": UNITS,
                    "uncertainty_treatment": UNCERTAINTY_TREATMENT,
                    "covariance": COVARIANCE_TREATMENT,
                    "acceptance_correction": acceptance,
                    "split": SPLIT_PAPB,
                }
            )

    return {
        "card": CARD,
        "run": RUN_DESCRIPTION,
        "generated_utc": _utc_now_iso(),
        "field_list": list(INVENTORY_FIELDS),
        "conventions": conventions,
        "fit_likelihood_observables": fit_observables,
        "n_tables": len(records),
        "tables": records,
    }


def record_complete(record: dict) -> bool:
    """All §7.1 fields present and non-null for one table record."""
    for field in INVENTORY_FIELDS:
        value = record.get(field)
        if value is None:
            return False
        if field == "bin_edges":
            if not isinstance(value, dict):
                return False
            if not value.get("xF") or not value.get("pT"):
                return False
    return True


def inventory_completeness(payload: dict) -> tuple[bool, int, int]:
    records = payload.get("tables", [])
    flags = [record_complete(r) for r in records]
    return (bool(flags) and all(flags), len(records), int(sum(flags)))


# ---------------------------------------------------------------------------
# Closure report
# ---------------------------------------------------------------------------


def build_closure_report(n_events: int) -> dict:
    """The pipeline closure report for the three pre-registered toys."""
    grids = closure_grids()
    sqrt_s = pipeline_sqrt_s()
    domain = sampler_domain(grids)

    toys: list[dict] = []
    for index, toy in enumerate(TOY_SPECTRA):
        analytic = analytic_closure(toy, grids, sqrt_s, n_ref=int(n_events))
        model_by_species: dict[str, np.ndarray] = {}
        offset = 0
        for grid in grids:
            model_by_species[grid.species_display] = np.asarray(
                [analytic["bins"][offset + i]["model"] for i in range(grid.n_bins)],
                dtype=float,
            )
            offset += grid.n_bins
        mc = mc_closure(
            toy,
            index,
            grids,
            model_by_species,
            sqrt_s,
            n_events=int(n_events),
            mc_seed=MC_SEED,
        )
        toys.append(
            {
                "name": toy["name"],
                "params": dict(toy),
                "n_bins": analytic["n_bins"],
                "max_rel_deviation_analytic": analytic[
                    "max_rel_deviation_analytic"
                ],
                "analytic_n_ref": analytic["n_ref"],
                "mc_events_used": mc["mc_events_used"],
                "mc_seed": mc["mc_seed"],
                "mc_tracks": mc["mc_tracks"],
                "lambda_tracks_per_event": mc["lambda_tracks_per_event"],
                "sampler_proposals": mc["sampler_proposals"],
                "n_non_empty_bins": mc["n_non_empty_bins"],
                "empty_bins": mc["empty_bins"],
                "empty_bin_records": mc["empty_bin_records"],
                "max_rel_deviation_mc": mc["max_rel_deviation_mc"],
                "mc_passes": mc["mc_passes"],
                "mc_sanity_pass_fraction": mc["mc_sanity_pass_fraction"],
                "mc_tolerance_rule": MC_TOLERANCE_RULE,
                "bins": {"analytic": analytic["bins"], "mc": mc["bins"]},
            }
        )

    return {
        "card": CARD,
        "run": RUN_DESCRIPTION,
        "generated_utc": _utc_now_iso(),
        "pre_registered_mc_events": DEFAULT_MC_EVENTS,
        "mc_events_used": int(n_events),
        "mc_seed": MC_SEED,
        "mc_seed_rule": (
            "numpy.random.default_rng(SeedSequence(20303000, "
            "spawn_key=(toy_index,)))"
        ),
        "mc_tolerance_rule": MC_TOLERANCE_RULE,
        "analytic_tolerance": ANALYTIC_TOLERANCE,
        "mc_pass_fraction_min": MC_PASS_FRACTION_MIN,
        "sigma_inel_mb": SIGMA_INEL_MB,
        "toy_mass_gev": TOY_MASS_GEV,
        "toy_mass_note": (
            "the toy tracks carry the pi- rest mass; the mass cancels "
            "exactly in the closure identity (the same mass enters the "
            "sampler density and the pipeline E/pT weight)"
        ),
        "sqrt_s_pipeline_gev": sqrt_s,
        "sampler_domain": {
            "xF": [domain[0], domain[1]],
            "pT": [domain[2], domain[3]],
            "convention": (
                "the union of the six windowed grids' edge spans, so every "
                "grid bin's full rectangle is covered and the per-bin "
                "identity holds without boundary clipping; the paper "
                "window and the 1017-bin observable set are unchanged"
            ),
        },
        "observable_set": {
            "window": dict(PAPER_WINDOW),
            "species_bin_counts": {g.species_display: g.n_bins for g in grids},
            "total_bins": TOTAL_BINS,
        },
        "grids": [
            {
                "species": g.species_display,
                "n_bins": g.n_bins,
                "n_x": g.n_x,
                "n_p": g.n_p,
                "xF_edges": [float(v) for v in g.xedges],
                "pT_edges": [float(v) for v in g.pedges],
            }
            for g in grids
        ],
        "toys": toys,
        "max_rel_deviation_analytic_all": max(
            t["max_rel_deviation_analytic"] for t in toys
        ),
        "mc_sanity_pass_fraction_min": min(
            t["mc_sanity_pass_fraction"] for t in toys
        ),
    }


# ---------------------------------------------------------------------------
# Provenance, seal, gate
# ---------------------------------------------------------------------------


def inventory_input_paths() -> list[Path]:
    """Every consumed input: the config, the card-7 config, the pa_data
    README, the six SPS tables, and the twelve p+Pb tables."""
    binned = sps_config_binned()
    root = sps_root()
    paths = [
        NESTED_ROOT / CONFIG_RELPATH,
        NESTED_ROOT / NUISANCES_RELPATH,
        NESTED_ROOT / "helpers" / "benchmarking" / "pa_data" / "README.md",
    ]
    for entry in binned.get("tables", []):
        paths.append(Path(root) / entry["path"])
    for species_files in PA_DATA_FILES.values():
        for filename in species_files.values():
            paths.append(pa_data_dir() / filename)
    missing = [p for p in paths if not p.is_file()]
    if missing:
        _refuse(
            "inventory input(s) missing:\n  "
            + "\n  ".join(str(p) for p in missing)
        )
    return paths


def write_provenance(
    out_dir: Path,
    *,
    pre_registration: dict,
    n_events: int,
    argv: list[str],
    wall_seconds: float,
) -> dict:
    """Write ``provenance.json`` via the ``run_provenance.py`` convention."""
    card7 = read_card7_nuisances()
    extra = {
        "card": CARD,
        "run": RUN_DESCRIPTION,
        "gate_clause": GATE_CLAUSE,
        "sigma_inel_mb": SIGMA_INEL_MB,
        "sigma_eta": card7["sigma_eta"],
        "use_data_covariance": card7["use_data_covariance"],
        "mc_seed": int(pre_registration["mc_seed"]),
        "mc_events_registered": int(pre_registration["mc_events"]),
        "mc_events_executed": int(n_events),
        "mc_tolerance_rule": MC_TOLERANCE_RULE,
        "analytic_tolerance": ANALYTIC_TOLERANCE,
        "observable_set": dict(pre_registration["observable_set"]),
        "sqrt_s_pipeline_gev": pipeline_sqrt_s(),
        "wall_seconds": round(float(wall_seconds), 3),
    }
    record = write_run_provenance(
        Path(out_dir),
        input_paths=inventory_input_paths(),
        argv=[str(a) for a in argv],
        env_var_names=["NA49_RUN_SOURCE_ARCHIVE"],
        rng={
            "type": (
                "toy sampler: numpy.random.default_rng(SeedSequence(mc_seed, "
                "spawn_key=(toy_index,))); Poisson(Lambda) tracks per event "
                "from the track density (pi sqrt(s)/sigma_inel) f pT/E"
            ),
            "seeds": {
                "mc_seed": int(pre_registration["mc_seed"]),
                "toy_indices": list(range(len(TOY_SPECTRA))),
            },
        },
        source_record=collect_source_record(
            outer_repo=MONO_ROOT, nested_repo=NESTED_ROOT
        ),
        producer="staged_data_closure.py",
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
    seal manifest instead), and ``gate1_r10.json`` (it carries the
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
    ``provenance.json``, and ``gate1_r10.json``), each with its sha256 and
    byte size; it cannot contain its own digest.
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
            "seal_manifest.json, provenance.json, gate1_r10.json); "
            "seal_manifest.json records the payload files plus "
            "SHA256SUMS.txt, provenance.json, and gate1_r10.json; "
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
        _refuse(
            f"provenance missing before sealing: {out_dir / PROVENANCE_NAME}"
        )
    write_sha256sums(out_dir)
    return write_seal_manifest(out_dir)


def assemble_gate1(out_dir: Path) -> dict:
    """Assemble ``gate1_r10.json`` from the run's artifacts.

    The record matches the card's acceptance schema exactly.  The verdict
    is ``pass`` only when the inventory is complete (all §7.1 fields
    non-null for every table and
    ``n_tables_with_all_fields_non_null == n_tables``), every toy's
    analytic-path deviation is below the pre-registered 1e-6, and every
    toy's MC sanity pass fraction is at least 0.95.
    """
    out_dir = Path(out_dir)
    prereg_path = out_dir / PREREGISTRATION_NAME
    if not prereg_path.is_file():
        _refuse(f"pre-registration missing: {prereg_path}")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))

    inventory_path = out_dir / INVENTORY_NAME
    if not inventory_path.is_file():
        _refuse(f"data inventory missing: {inventory_path}")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))

    closure_path = out_dir / CLOSURE_REPORT_NAME
    if not closure_path.is_file():
        _refuse(f"closure report missing: {closure_path}")
    closure = json.loads(closure_path.read_text(encoding="utf-8"))

    records = inventory.get("tables", [])
    flags = [record_complete(r) for r in records]
    n_tables = len(records)
    n_complete = int(sum(flags))
    inventory_complete = bool(flags) and all(flags)

    toy_records = []
    for toy in closure.get("toys", []):
        toy_records.append(
            {
                "name": toy["name"],
                "max_rel_deviation_analytic": float(
                    toy["max_rel_deviation_analytic"]
                ),
                "mc_sanity_pass_fraction": float(
                    toy["mc_sanity_pass_fraction"]
                ),
            }
        )
    if len(toy_records) != len(TOY_SPECTRA):
        _refuse(
            f"closure report carries {len(toy_records)} toys; the "
            f"pre-registration declares {len(TOY_SPECTRA)}"
        )

    analytic_ok = all(
        t["max_rel_deviation_analytic"] < ANALYTIC_TOLERANCE
        for t in toy_records
    )
    mc_ok = all(
        t["mc_sanity_pass_fraction"] >= MC_PASS_FRACTION_MIN
        for t in toy_records
    )
    verdict = (
        "pass"
        if (
            inventory_complete
            and n_complete == n_tables
            and analytic_ok
            and mc_ok
        )
        else "fail"
    )
    gate = {
        "gate": "gate1",
        "clause": GATE_CLAUSE,
        "data_inventory_complete": inventory_complete,
        "n_tables": int(n_tables),
        "n_tables_with_all_fields_non_null": int(n_complete),
        "tables": records,
        "closure": {
            "toy_spectra": toy_records,
            "analytic_tolerance": ANALYTIC_TOLERANCE,
            "mc_events": int(prereg.get("mc_events", DEFAULT_MC_EVENTS)),
            "mc_seed": int(prereg.get("mc_seed", MC_SEED)),
        },
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
    ``--out dev/00runs/r10_data_pipeline_closure_2026-09-11``, so an
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
            "R10 data pipeline closure driver: the §7.1 data inventory for "
            "the six SPS tables and the p+Pb tables, and the pipeline "
            "closure of the binning / normalization / estimator code path "
            "against three analytic toy spectra with closed-form bin "
            "integrals (analytic path < 1e-6; MC sanity at "
            "max(3/sqrt(N_bin), 1e-3))"
        )
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("inventory", "closure", "all"),
        help="inventory, closure, or both in order",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="run directory (dev/00runs/... resolves against the monorepo root)",
    )
    parser.add_argument(
        "--mc-events",
        type=int,
        default=DEFAULT_MC_EVENTS,
        help=(
            "toy MC events per closure toy (default "
            f"{DEFAULT_MC_EVENTS}; the registered sample size)"
        ),
    )
    parser.add_argument(
        "--seal",
        action="store_true",
        help=(
            "write provenance.json, SHA256SUMS.txt, seal_manifest.json, "
            "and the gate1_r10.json acceptance record (requires --mode all)"
        ),
    )
    return parser


def main(argv: list | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if int(args.mc_events) <= 0:
        _refuse(
            f"--mc-events must be a positive integer, got {args.mc_events}; "
            "0 and negative counts are refused before any work"
        )
    if args.seal and args.mode != "all":
        _refuse(
            "--seal requires --mode all: the gate needs the data inventory "
            "and the closure report"
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
                "R10 pre-registration (the three toy spectra, the MC seed "
                "20303000, the registered 1,000,000 events, the tolerance "
                "rule, the 1e-6 analytic tolerance, and the 1017-bin "
                "observable set are pre-registered; a different registration "
                "files its own card)"
            )
    else:
        _write_json(prereg_path, payload_expected)
    pre_registration = payload_expected

    started = time.perf_counter()

    if args.mode in ("inventory", "all"):
        inventory = build_inventory()
        complete, n_tables, n_complete = inventory_completeness(inventory)
        _write_json(out_dir / INVENTORY_NAME, inventory)
        print(
            f"[r10] inventory: {n_tables} tables ({n_complete} with all "
            "§7.1 fields non-null); sigma_eta "
            f"{inventory['conventions']['sigma_eta']}; use_data_covariance "
            f"{inventory['conventions']['use_data_covariance']}",
            flush=True,
        )
        if args.mode == "inventory" and n_complete != n_tables:
            print(
                "[r10] data inventory INCOMPLETE; the Gate 1 clause blocks",
                flush=True,
            )
            return 1

    if args.mode in ("closure", "all"):
        closure = build_closure_report(int(args.mc_events))
        _write_json(out_dir / CLOSURE_REPORT_NAME, closure)
        print(
            "[r10] closure: analytic max rel deviation per toy "
            + ", ".join(
                f"{t['name']} {t['max_rel_deviation_analytic']:.3e}"
                for t in closure["toys"]
            )
            + "; MC pass fraction "
            + ", ".join(
                f"{t['name']} {t['mc_sanity_pass_fraction']:.4f}"
                for t in closure["toys"]
            )
            + " (empty bins "
            + ", ".join(str(t["empty_bins"]) for t in closure["toys"])
            + f"); mc_events {args.mc_events}, seed {MC_SEED}",
            flush=True,
        )
        if closure["max_rel_deviation_analytic_all"] >= ANALYTIC_TOLERANCE:
            print(
                "[r10] analytic-path closure FAILED (>= 1e-6); the Gate 1 "
                "< 1e-6 criterion blocks",
                flush=True,
            )
            return 1

    wall = time.perf_counter() - started

    exit_code = 0
    if args.seal:
        write_provenance(
            out_dir,
            pre_registration=payload_expected,
            n_events=int(args.mc_events),
            argv=[str(a) for a in sys.argv[1:]],
            wall_seconds=wall,
        )
        write_sha256sums(out_dir)
        gate = assemble_gate1(out_dir)
        # The seal manifest is written after the acceptance record so it
        # covers gate1_r10.json too (the gate carries the sums digest, so
        # the sums inventory excludes the gate; the seal manifest cannot
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
                "[r10] seal verification FAILED; the run directory is not "
                "immutable as written",
                flush=True,
            )
            return 1
        print(
            f"[r10] run directory sealed: {out_dir} "
            f"({len(seal['files'])} sealed files); gate1_r10.json verdict "
            f"{gate['verdict']}",
            flush=True,
        )
        if gate["verdict"] != "pass":
            exit_code = 1
    else:
        print(
            "[r10] artifacts written; rerun with --mode all --seal to write "
            "provenance.json, SHA256SUMS.txt, seal_manifest.json, and the "
            "gate1_r10.json acceptance record",
            flush=True,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
