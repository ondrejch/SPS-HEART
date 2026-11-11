# Partly legacy (marked 2026-11-04): the HEART p+p fits use only BinnedFitTable here (through
# helpers/benchmarking/run_observables.py); the staged objectives and the p+A drivers are legacy, and code reviews
# of the p+p model can skip them. The cleanup and refactor after the p+A run splits the module.
"""Staged evaluation runner: simple p+p -> Regge fit -> p+A transfer.

The stage boundary is deliberate:

1. ``simple_excitation`` validates exact source/event closure with an empirical
   mass law.
2. ``regge_fit`` replaces only the production law and may fit its declared
   parameters using common-random-number simulations. Two objectives are
   supported (``fit.objective``): the diagnostic aggregate objective against
   summary targets (``"aggregate"``, the historical default) and the binned
   NA49 objective (``"binned"``), a diagonal chi-square over the windowed
   per-bin invariant spectra of the identified species. Since M8 the binned
   objective is uncertainty-aware: each evaluation generates K fixed-seed
   MC replicas (``fit.mc_replicas``), the per-bin model prediction is the
   replica mean, and every bin of every binned observable carries the
   combined error ``sigma_i^2 = sigma_data,i^2 + sigma_MC,i^2`` with
   ``sigma_MC,i`` the standard error of the replica mean. The aggregate
   objective is unchanged. Both objectives regenerate their event
   sample(s) per objective evaluation.
3. ``baryon_nucleus`` freezes the p+p physics parameters and transfers them to
   a Glauber p+A calculation with one sequential projectile source.

Run with::

    python -m na49_mixing.staged_evaluation --config config/staged_excitation.json

M11 parameter profiles (additive machinery): the one-dimensional profile
chi-square with the nuisance parameters refit at every scan point, the
finite-difference Hessian and correlation estimator, and the profile
artifacts, all on the uncertainty-aware binned objective above. The
pre-registration ``dev/00runs/parameter_profiles_2026-09-07/
pre_registration.json`` is authoritative and is cross-checked in every
profile mode (``--profile-run``). The staged-evaluation paths below are
unchanged.

Sector-wise deviance decomposition (Round 1, card
TASK-20260911-r1-sector-wise-fit-reports, additive): the optional
``--sector-report`` flag (or the ``report.sector_wise: true`` config
key) writes ``sector_report.json`` alongside the staged evaluation
report — the per-sector deviance partition over the pre-registered
``SECTOR_REGIONS`` grid (3 xF x 3 pT per species-observable, the
``na49_mixing.fit_likelihood`` machinery), with the worst-bin pull and
the residual mean per sector. The flag is off by default; the staged
evaluation report is bit-identical in both flag states (the sector
report is an additional output file, never a modification of the
staged run).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from enum import IntEnum
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from helpers.benchmarking import sps_core

from .continuum_channels import ContinuumParameters
from .event_builder_pA import PAEventBuilder, PAExcitationConfig
from .event_builder_pp import PPEventBuilder, PPExcitationConfig
from .exchange_model import ReggeExcitationParameters, SimpleExcitationParameters
from .fit_likelihood import (
    BinnedObservable,
    diagonal_covariance,
    sector_decomposition,
    write_sector_report,
)
from .glauber import NucleusSpec
from .reaction import Reaction
from . import rng_substreams
from .source_event import ExcitationEvent

# RNG substream block (Round 1, card 5, additive): recorded into the run
# metadata report only when the config key ``substreams.enabled`` is
# true.  The block documents the derivation contract; it never alters
# the generated events (the legacy default without the key, and the
# ``enabled: false`` explicit path, produce bit-identical streams).
_SUBSTREAM_REPLICA_SEED_RULE = (
    "derive_substreams(master_seed, replica_index, event_index) with "
    "master_seed = config.seed: numpy.random.SeedSequence(config.seed)"
    ".spawn(replica_index + 1)[replica_index].spawn(event_index + 1)"
    "[event_index].spawn(len(substream_labels)) yields one 64-bit stream "
    "id per label, a pure function of (master_seed, replica_index, "
    "event_index); a draw in a labeled layer of event e of replica k is "
    "seeded with (stream_id_of_layer, e), the <config.seed>:<event_id> "
    "convention with the physical layer labeled"
)

# Nested source-tree root (config-relative table paths). The helpers.benchmarking
# import above is an installed package sibling of na49_mixing, not a
# repository-relative sys.path injection.
from na49_mixing._paths import nested_root as _nested_root  # noqa: E402

_NESTED_ROOT = _nested_root()

_BINNED_DATA_PREREQUISITE = (
    "the binned six-species objective reads the NA49 SPS tables under "
    "spshadrons.web.cern.ch/ (excluded from source-only archives). "
    "Provide the tables, set fit.binned.path_root to a directory that "
    "contains them, or run the smoke configuration "
    "config/staged_excitation_smoke.json (fit disabled)."
)


class EvaluationStage(IntEnum):
    SIMPLE_EXCITATION = 1
    REGGE_FIT = 2
    BARYON_NUCLEUS = 3

    @classmethod
    def parse(cls, value: str | int) -> "EvaluationStage":
        if isinstance(value, int):
            return cls(value)
        key = str(value).strip().lower().replace("-", "_")
        aliases = {
            "simple": cls.SIMPLE_EXCITATION,
            "simple_excitation": cls.SIMPLE_EXCITATION,
            "regge": cls.REGGE_FIT,
            "regge_fit": cls.REGGE_FIT,
            "pa": cls.BARYON_NUCLEUS,
            "p+a": cls.BARYON_NUCLEUS,
            "baryon_nucleus": cls.BARYON_NUCLEUS,
        }
        if key not in aliases:
            raise ValueError(f"unknown evaluation stage {value!r}")
        return aliases[key]


_FITTABLE_STAGE2 = {
    "regge.effective_transfer_slope",
    "regge.alpha_exc0",
    "regge.edge_power",
    "continuum.match_mass",
    "continuum.pion_multiplicity_slope",
    "continuum.strange_lambda_strength",
    "continuum.strange_sigma_strength",
    "continuum.strange_pair_strength",
    "continuum.strange_threshold_power",
    "continuum.pair_strength",
    "continuum.pair_threshold_power",
}


def _dataclass_from_dict(cls, values: dict[str, Any] | None):
    return cls(**(values or {}))


def _summarize(events: list[ExcitationEvent]) -> dict[str, float]:
    if not events:
        raise ValueError("cannot summarize an empty event sample")
    source_masses = [source.mass for event in events for source in event.sources]
    transfers = [
        abs(float(source.metadata["t"]))
        for event in events
        for source in event.sources
        if "t" in source.metadata
    ]
    final = [particle for event in events for particle in event.final_particles]
    continuum_sources = {
        particle.source_id + event.event_id * 1_000_000
        for event in events
        for particle in event.final_particles
        if particle.channel_family.startswith("continuum")
        or particle.channel_family.startswith("associated")
        or particle.channel_family == "baryon_pair"
    }
    all_sources = sum(len(event.sources) for event in events)
    strange_sources = {
        particle.source_id + event.event_id * 1_000_000
        for event in events
        for particle in event.final_particles
        if particle.channel_family.startswith("associated")
    }
    pair_sources = {
        particle.source_id + event.event_id * 1_000_000
        for event in events
        for particle in event.final_particles
        if particle.channel_family == "baryon_pair"
    }
    residuals = [event.conservation_residuals()["relative_max"] for event in events]
    wounds = [float(event.metadata.get("target_wounds", 1.0)) for event in events]
    return {
        "n_events": float(len(events)),
        "mean_source_mass": float(np.mean(source_masses)),
        "std_source_mass": float(np.std(source_masses)),
        "mean_abs_t": float(np.mean(transfers)) if transfers else 0.0,
        "mean_final_multiplicity": float(len(final) / len(events)),
        "continuum_source_fraction": float(len(continuum_sources) / all_sources),
        "strange_source_fraction": float(len(strange_sources) / all_sources),
        "pair_source_fraction": float(len(pair_sources) / all_sources),
        "mean_target_wounds": float(np.mean(wounds)),
        "max_conservation_residual": float(max(residuals)),
    }


def _generate_pp(
    n_events: int,
    seed: int,
    simple: SimpleExcitationParameters,
    regge: ReggeExcitationParameters,
    continuum: ContinuumParameters,
    mode: str,
    beam_momentum: float,
    stage: str,
) -> tuple[list[ExcitationEvent], dict[str, float]]:
    builder = PPEventBuilder(
        PPExcitationConfig(
            beam_momentum=beam_momentum,
            production_mode=mode,
            simple=simple,
            regge=regge,
            continuum=continuum,
        ),
        np.random.default_rng(seed),
    )
    events = [builder.generate(i, stage=stage) for i in range(n_events)]
    for event in events:
        event.metadata["random_seed"] = int(seed)
        event.metadata["random_substream"] = f"{stage}:{event.event_id}"
    return events, _summarize(events)


def _set_parameter(
    regge: ReggeExcitationParameters,
    continuum: ContinuumParameters,
    name: str,
    value: float,
) -> tuple[ReggeExcitationParameters, ContinuumParameters]:
    group, field = name.split(".", 1)
    if group == "regge":
        return replace(regge, **{field: float(value)}), continuum
    if group == "continuum":
        return regge, replace(continuum, **{field: float(value)})
    raise ValueError(f"unsupported parameter group {group!r}")


def _validate_fit_parameters(
    fit_cfg: dict[str, Any],
    regge: ReggeExcitationParameters,
    continuum: ContinuumParameters,
) -> tuple[list[str], list[float], list[tuple[float, float]]]:
    """Validate the declared parameter list and build initial values and bounds.

    The refusals apply to every stage-2 objective: a parameter that is not
    identifiable at one beam energy is refused before any event generation or
    RNG consumption.
    """
    names = list(fit_cfg.get("parameters", []))
    for name in names:
        if name == "regge.alpha_p0":
            raise ValueError(
                "regge.alpha_p0 is not fittable at one fixed energy because it "
                "only changes the overall normalized event weight"
            )
        if name in ("regge.alpha_p_prime", "regge.transfer_slope"):
            raise ValueError(
                f"{name} is not fittable at one fixed energy because at one "
                "beam momentum the normalized t distribution depends only on "
                "the effective transfer slope B_eff = B + 2 alpha'_P ln(s/s0); "
                "regge.effective_transfer_slope is the single fittable transfer "
                "degree of freedom. Separating alpha'_P from B requires an "
                "energy lever arm or independent transfer data"
            )
        if name not in _FITTABLE_STAGE2:
            raise ValueError(
                f"{name!r} is not a declared Regge-stage parameter; "
                "p+A-only parameters cannot be back-fitted here"
            )

    bounds_cfg = dict(fit_cfg.get("bounds", {}))
    initial: list[float] = []
    bounds: list[tuple[float, float]] = []
    for name in names:
        group, field = name.split(".", 1)
        obj = regge if group == "regge" else continuum
        value = getattr(obj, field)
        if value is None:
            raise ValueError(
                f"{name} is not set in the configuration and cannot be fitted"
            )
        initial.append(float(value))
        if name not in bounds_cfg or len(bounds_cfg[name]) != 2:
            raise ValueError(f"fit bound [low, high] required for {name}")
        low, high = map(float, bounds_cfg[name])
        if not (math.isfinite(low) and math.isfinite(high) and low < high):
            raise ValueError(f"invalid bounds for {name}: {bounds_cfg[name]!r}")
        bounds.append((low, high))
    return names, initial, bounds


def _fit_regge(
    fit_cfg: dict[str, Any],
    simple: SimpleExcitationParameters,
    regge: ReggeExcitationParameters,
    continuum: ContinuumParameters,
    beam_momentum: float,
    seed: int,
) -> tuple[ReggeExcitationParameters, ContinuumParameters, dict[str, Any]]:
    objective_kind = str(fit_cfg.get("objective", "aggregate")).strip().lower()
    if objective_kind not in {"aggregate", "binned"}:
        raise ValueError(
            f"unknown fit objective {objective_kind!r}; "
            "expected 'aggregate' or 'binned'"
        )
    names, initial, bounds = _validate_fit_parameters(fit_cfg, regge, continuum)
    if not bool(fit_cfg.get("enabled", False)) or not names:
        return regge, continuum, {
            "performed": False,
            "reason": "fit disabled or no parameters/targets supplied",
            "objective": objective_kind,
        }
    if objective_kind == "binned":
        return _fit_regge_binned(
            fit_cfg, simple, regge, continuum, beam_momentum, seed,
            names, initial, bounds,
        )
    targets = dict(fit_cfg.get("targets", {}))
    if not targets:
        return regge, continuum, {
            "performed": False,
            "reason": "fit disabled or no parameters/targets supplied",
            "objective": objective_kind,
        }

    n_eval = int(fit_cfg.get("events_per_evaluation", 300))
    maxiter = int(fit_cfg.get("max_iterations", 30))
    evaluations: list[dict[str, Any]] = []

    def unpack(x):
        r, c = regge, continuum
        for name, value in zip(names, x):
            r, c = _set_parameter(r, c, name, float(value))
        r.validate()
        c.validate()
        return r, c

    def objective(x) -> float:
        r, c = unpack(x)
        _, summary = _generate_pp(
            n_eval, seed, simple, r, c, "regge", beam_momentum, "regge_fit_trial"
        )
        chi2 = 0.0
        for observable, target in targets.items():
            if observable not in summary:
                raise ValueError(f"unknown fit observable {observable!r}")
            value = float(target["value"])
            sigma = float(target["sigma"])
            if not (math.isfinite(sigma) and sigma > 0.0):
                raise ValueError(f"target sigma must be positive for {observable}")
            chi2 += ((summary[observable] - value) / sigma) ** 2
        evaluations.append({
            "parameters": {name: float(v) for name, v in zip(names, x)},
            "chi2": float(chi2),
        })
        return float(chi2)

    from scipy.optimize import minimize

    result = minimize(
        objective,
        np.asarray(initial),
        method="Powell",
        bounds=bounds,
        options={"maxiter": maxiter, "xtol": 1e-3, "ftol": 1e-3},
    )
    best = min(evaluations, key=lambda item: item["chi2"])
    best_x = np.asarray([best["parameters"][name] for name in names], dtype=float)
    fitted_regge, fitted_continuum = unpack(best_x)
    return fitted_regge, fitted_continuum, {
        "performed": True,
        "objective": "aggregate",
        "success": bool(result.success),
        "message": str(result.message),
        "chi2": float(best["chi2"]),
        "parameters": dict(best["parameters"]),
        "n_objective_evaluations": int(result.nfev),
        "trace": evaluations,
    }


# ---------------------------------------------------------------------------
# Binned NA49 objective (card TASK-20260905-physics-review, milestone M4)
# ---------------------------------------------------------------------------

_BINNED_NORMALIZATIONS = ("f", "f_over_sigma_inel")


def _resolve_binned_path(path: str | Path, path_root: str | Path) -> Path:
    """Resolve a table path: absolute as-is, else relative to ``path_root``.

    Relative paths are the bundled convention (nested-root relative, e.g.
    ``spshadrons.web.cern.ch/data/xsect_piminus.txt``); ``path_root`` is the
    optional ``fit.binned.path_root`` override used by hermetic synthetic
    table tests.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (Path(path_root) / candidate)


def preflight_binned_tables(
    binned_cfg: dict[str, Any],
    path_root: str | Path | None = None,
) -> list[Path]:
    """Resolve and require every binned-table path before any table read.

    Raises ``FileNotFoundError`` naming every missing file and the data
    prerequisite, so a source-only checkout fails with a deliberate
    preflight rather than an incidental open() error on the first table.
    """
    root = Path(path_root) if path_root is not None else Path(
        binned_cfg.get("path_root", _NESTED_ROOT)
    )
    entries = list(binned_cfg.get("tables", []))
    if not entries:
        raise ValueError("fit.binned.tables must name at least one table")
    resolved: list[Path] = []
    missing: list[Path] = []
    for entry in entries:
        path = _resolve_binned_path(entry["path"], root)
        resolved.append(path)
        if not path.is_file():
            missing.append(path)
    if missing:
        listed = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(
            "binned-fit data preflight failed; missing table file(s):\n"
            f"{listed}\n{_BINNED_DATA_PREREQUISITE}"
        )
    return resolved


@dataclass(frozen=True)
class BinnedFitTable:
    """One windowed NA49 binned spectrum converted to fit units.

    ``data`` (D) and ``sigma`` hold the per-bin values of the invariant cross
    section divided by the inelastic cross section (1/GeV^2) after the
    per-table normalization declaration is applied: a table declared
    ``"f"`` is divided by ``sigma_inel_mb``; a table declared
    ``"f_over_sigma_inel"`` is used as-is. ``bin_index_xf`` and
    ``bin_index_pt`` index the unique windowed (xF, pT) centers, whose
    midpoint edges define the bin grid. Bins whose tabulated uncertainty is
    missing or non-positive use the documented fallback uncertainty (Poisson
    data content, unit-count floor on empty bins) and are counted in
    ``n_sigma_fallback``.
    """

    species: str
    pdg_id: int
    mass_gev: float
    normalization: str
    sigma_inel_mb: float
    path: str
    xf_centers: np.ndarray
    pt_centers: np.ndarray
    bin_index_xf: np.ndarray
    bin_index_pt: np.ndarray
    data: np.ndarray
    sigma: np.ndarray
    n_sigma_fallback: int

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        species: str,
        pdg_id: int,
        mass_gev: float,
        normalization: str,
        sigma_inel_mb: float,
        window: dict[str, float] | None = None,
    ) -> "BinnedFitTable":
        xf, pt, value, error = sps_core.read_na49_points(path)
        return cls.from_arrays(
            xf=xf,
            pt=pt,
            value=value,
            error=error,
            species=species,
            pdg_id=pdg_id,
            mass_gev=mass_gev,
            normalization=normalization,
            sigma_inel_mb=sigma_inel_mb,
            path=str(path),
            window=window,
        )

    @classmethod
    def from_arrays(
        cls,
        *,
        species: str,
        pdg_id: int,
        mass_gev: float,
        normalization: str,
        sigma_inel_mb: float,
        path: str,
        xf: np.ndarray,
        pt: np.ndarray,
        value: np.ndarray,
        error: np.ndarray,
        window: dict[str, float] | None = None,
    ) -> "BinnedFitTable":
        if normalization not in _BINNED_NORMALIZATIONS:
            raise ValueError(
                f"unknown table normalization {normalization!r}; expected one of "
                f"{_BINNED_NORMALIZATIONS}"
            )
        if not (math.isfinite(float(sigma_inel_mb)) and float(sigma_inel_mb) > 0.0):
            raise ValueError(f"sigma_inel_mb must be positive, got {sigma_inel_mb!r}")
        window = window or {}
        xf_arr = np.asarray(xf, dtype=float)
        pt_arr = np.asarray(pt, dtype=float)
        value_arr = np.asarray(value, dtype=float)
        error_arr = np.asarray(error, dtype=float)
        keep = sps_core.paper_window_mask(
            xf_arr,
            pt_arr,
            require_positive_sigma=False,
            xf_min=float(window.get("xf_min", sps_core.PAPER_XF_MIN)),
            xf_max=float(window.get("xf_max", sps_core.PAPER_XF_MAX)),
            pt_min=float(window.get("pt_min", sps_core.PAPER_PT_MIN)),
            pt_max=float(window.get("pt_max", sps_core.PAPER_PT_MAX)),
        )
        xf_w = xf_arr[keep]
        pt_w = pt_arr[keep]
        data = value_arr[keep] / (
            float(sigma_inel_mb) if normalization == "f" else 1.0
        )
        sigma = error_arr[keep] / (
            float(sigma_inel_mb) if normalization == "f" else 1.0
        )
        if xf_w.size == 0:
            raise ValueError(f"no {species} table points inside the fit window")
        bad = ~np.isfinite(sigma) | (sigma <= 0.0)
        if bad.any():
            fallback = np.where(data > 0.0, np.sqrt(np.maximum(data, 0.0)), 1.0)
            sigma = np.where(bad, fallback, sigma)
        xf_centers = np.unique(xf_w)
        pt_centers = np.unique(pt_w)
        xedges = sps_core.edges_from_centers(xf_centers)
        pedges = sps_core.edges_from_centers(pt_centers)
        ix, ip = sps_core.grid_indices(xf_w, pt_w, xedges, pedges)
        return cls(
            species=species,
            pdg_id=int(pdg_id),
            mass_gev=float(mass_gev),
            normalization=normalization,
            sigma_inel_mb=float(sigma_inel_mb),
            path=str(path),
            xf_centers=xf_centers,
            pt_centers=pt_centers,
            bin_index_xf=np.asarray(ix, dtype=np.int64),
            bin_index_pt=np.asarray(ip, dtype=np.int64),
            data=data,
            sigma=sigma,
            n_sigma_fallback=int(bad.sum()),
        )


def _bin_tracks(
    events: list[ExcitationEvent],
    tables: list[BinnedFitTable],
    sqrt_s_gev: float,
) -> dict[str, np.ndarray]:
    """Per-species binned invariant densities from one event sample.

    Every final track of a table species contributes its Jacobian weight
    E/pT to the bin of its (xF, pT) point, with
    xF = 2 p_z / sqrt(s) taken from the track four-vector. The binned
    density is

        M_bin = (1 / (pi * sqrt(s) * N_ev)) * sum(E/pT) / (dxF * dpT),

    which is the invariant cross section per inelastic event (one generated
    event per inelastic event), i.e. f / sigma_inel in 1/GeV^2. That matches
    the converted table data, so the per-bin chi-square is dimensionless.
    The binning is fixed by the table bin centers, so all replicas of the
    M8 uncertainty-aware objective bin onto identical bins.
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
    results: dict[str, np.ndarray] = {}
    for table in tables:
        xedges = sps_core.edges_from_centers(table.xf_centers)
        pedges = sps_core.edges_from_centers(table.pt_centers)
        n_x = xedges.size - 1
        n_p = pedges.size - 1
        selected = (pdg == table.pdg_id) & (pt > 0.0)
        weights_flat = np.zeros(n_p * n_x, dtype=float)
        if selected.any():
            x_sel = 2.0 * pz[selected] / sqrt_s_gev
            p_sel = pt[selected]
            w_sel = e[selected] / p_sel
            ix, ip = sps_core.grid_indices(x_sel, p_sel, xedges, pedges)
            in_bounds = (ix >= 0) & (ix < n_x) & (ip >= 0) & (ip < n_p)
            flat = ip[in_bounds] * n_x + ix[in_bounds]
            weights_flat = np.bincount(
                flat, weights=w_sel[in_bounds], minlength=n_p * n_x
            )[: n_p * n_x]
        areas = np.outer(np.diff(pedges), np.diff(xedges)).ravel()
        m_flat = weights_flat / (math.pi * sqrt_s_gev * n_ev * areas)
        results[table.species] = m_flat[table.bin_index_pt * n_x + table.bin_index_xf]
    return results


def _binned_mc_spectra(
    events: list[ExcitationEvent],
    tables: list[BinnedFitTable],
    sqrt_s_gev: float,
    n_params: int,
) -> dict[str, tuple[np.ndarray, float, dict[str, float]]]:
    """Single-sample binned densities with the data-only chi-square and metrics.

    The single-sample, data-only convention of the M4/M7 diagnostics: one
    event sample per call, per-bin residual against the data-only sigma.
    Retained unchanged for the M7 decay-comparison harness arms; the M8
    uncertainty-aware fit objective uses the replica machinery below.
    """
    densities = _bin_tracks(events, tables, sqrt_s_gev)
    results: dict[str, tuple[np.ndarray, float, dict[str, float]]] = {}
    for table in tables:
        m_table = densities[table.species]
        residual = (m_table - table.data) / table.sigma
        chi2_table = float(np.dot(residual, residual))
        metrics = sps_core.metrics_bundle(
            m_table, table.data, table.sigma, n_params, table.species
        )
        results[table.species] = (m_table, chi2_table, metrics)
    return results


def _mc_replica_std(replica_values: np.ndarray) -> np.ndarray:
    """Per-bin sample standard deviation (ddof=1) across the replica axis.

    ``replica_values`` has shape (K, n_bins): one binned-density row per
    replica, identical bins across replicas by construction.
    """
    values = np.asarray(replica_values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError(
            "replica sigma requires a (K, n_bins) array with K >= 2, "
            f"got shape {values.shape}"
        )
    return np.std(values, axis=0, ddof=1)


def _mc_replica_sigma(replica_values: np.ndarray) -> np.ndarray:
    """Per-bin MC sigma: the sample standard deviation across the K replicas
    divided by sqrt(K).

    The binned objective's prediction is the mean of the K replica
    densities, so its per-bin MC statistical uncertainty is the standard
    error of that mean: ``sigma_MC,i = std(replicas, ddof=1)_i / sqrt(K)``.
    This is the term that enters the combined error
    ``sigma_i^2 = sigma_data,i^2 + sigma_MC,i^2`` and the quantity whose
    1/sqrt(K) scaling in the replica count the M8 acceptance test pins.
    """
    values = np.asarray(replica_values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError(
            "replica sigma requires a (K, n_bins) array with K >= 2, "
            f"got shape {values.shape}"
        )
    return _mc_replica_std(values) / math.sqrt(values.shape[0])


def _replica_binned_spectra(
    replica_spectra: list[dict[str, np.ndarray]],
    tables: list[BinnedFitTable],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Replica-mean densities and per-bin MC sigmas from K per-replica spectra.

    Returns per species the replica-mean prediction, the MC sigma
    (standard error of the replica mean, ``_mc_replica_sigma``), and the
    per-replica sample standard deviation (the single-replica fluctuation
    scale, reported for provenance).
    """
    if not replica_spectra:
        raise ValueError("cannot build replica spectra from an empty replica list")
    m_mean: dict[str, np.ndarray] = {}
    sigma_mc: dict[str, np.ndarray] = {}
    sigma_replica: dict[str, np.ndarray] = {}
    for table in tables:
        stack = np.stack(
            [spectra[table.species] for spectra in replica_spectra], axis=0
        )
        m_mean[table.species] = stack.mean(axis=0)
        sigma_replica[table.species] = _mc_replica_std(stack)
        sigma_mc[table.species] = _mc_replica_sigma(stack)
    return m_mean, sigma_mc, sigma_replica


def _mc_replica_seed_base(fit_seed: int, seed_offset: int) -> int:
    """The M8 replica-seed base: ``config.seed + seed_offset``.

    The runner passes the fit substream seed ``config.seed + 10_000`` into
    the stage-2 fit, so the base is ``fit_seed - 10_000 + seed_offset``.
    For the bundled protocol (seed 20260831, M8 offset 40000) the base is
    20300831, distinct from the M4 fit substream (20270831), the M5 audit
    stream (20280831) and the M7 comparison stream (20290831).
    """
    return int(fit_seed) - 10_000 + int(seed_offset)


def _fit_regge_binned(
    fit_cfg: dict[str, Any],
    simple: SimpleExcitationParameters,
    regge: ReggeExcitationParameters,
    continuum: ContinuumParameters,
    beam_momentum: float,
    seed: int,
    names: list[str],
    initial: list[float],
    bounds: list[tuple[float, float]],
) -> tuple[ReggeExcitationParameters, ContinuumParameters, dict[str, Any]]:
    """Fit the declared parameters against the windowed binned NA49 spectra.

    M8 uncertainty-aware objective: the per-bin model prediction is the
    mean of ``K = fit.mc_replicas.count`` fixed-seed replica samples, each
    of ``events_per_evaluation`` events generated in regge mode at the
    trial parameters (replica k from seed ``config.seed + seed_offset + k``,
    identical across evaluations, so the streams stay common-random-number
    by construction). Every bin of every binned observable carries the
    combined error ``sigma_i^2 = sigma_data,i^2 + sigma_MC,i^2`` with
    ``sigma_MC,i`` the standard error of the replica mean
    (``_mc_replica_sigma``); the objective is the diagonal per-bin
    chi-square under that combined error, summed over the table species,
    each converted to f / sigma_inel units under its per-table
    normalization declaration. The per-species log-space diagnostics are
    reference quantities computed on the replica-mean prediction with the
    data-only sigma; they are not objective terms. There is no
    non-binned component in this objective.
    """
    binned_cfg = dict(fit_cfg.get("binned", {}))
    if not binned_cfg:
        raise ValueError("fit objective 'binned' requires a fit.binned block")
    sigma_inel_mb = float(binned_cfg.get("sigma_inel_mb", 31.78))
    if not (math.isfinite(sigma_inel_mb) and sigma_inel_mb > 0.0):
        raise ValueError(
            f"fit.binned.sigma_inel_mb must be positive, got {sigma_inel_mb!r}"
        )
    window_cfg = dict(binned_cfg.get("window", {}))
    window = {
        key: float(window_cfg[key])
        for key in ("xf_min", "xf_max", "pt_min", "pt_max")
        if key in window_cfg
    }
    path_root = Path(binned_cfg.get("path_root", _NESTED_ROOT))
    entries = list(binned_cfg.get("tables", []))
    if not entries:
        raise ValueError("fit.binned.tables must name at least one table")
    mc_cfg = fit_cfg.get("mc_replicas")
    if not isinstance(mc_cfg, dict) or not mc_cfg:
        raise ValueError(
            "the uncertainty-aware binned objective requires fit.mc_replicas "
            "(replica count and seed offset); the bundled config records the protocol"
        )
    if "count" not in mc_cfg:
        raise ValueError("fit.mc_replicas.count is required")
    n_replicas = int(mc_cfg["count"])
    if n_replicas < 5:
        raise ValueError(
            f"fit.mc_replicas.count must be >= 5, got {n_replicas}"
        )
    if "seed_offset" not in mc_cfg:
        raise ValueError("fit.mc_replicas.seed_offset is required")
    seed_offset = int(mc_cfg["seed_offset"])
    if seed_offset < 0:
        raise ValueError(
            f"fit.mc_replicas.seed_offset must be non-negative, got {seed_offset}"
        )
    preflight_binned_tables(binned_cfg, path_root)
    mc_seed_base = _mc_replica_seed_base(seed, seed_offset)
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
        for entry in entries
    ]
    n_eval = int(fit_cfg.get("events_per_evaluation", 300))
    maxiter = int(fit_cfg.get("max_iterations", 30))
    sqrt_s_gev = math.sqrt(Reaction(beam_momentum, 2212, 2212).s())
    evaluations: list[dict[str, Any]] = []

    def unpack(x):
        r, c = regge, continuum
        for name, value in zip(names, x):
            r, c = _set_parameter(r, c, name, float(value))
        r.validate()
        c.validate()
        return r, c

    def binned_objective(x) -> float:
        r, c = unpack(x)
        replica_spectra: list[dict[str, np.ndarray]] = []
        for k in range(n_replicas):
            replica_events, _ = _generate_pp(
                n_eval, mc_seed_base + k, simple, r, c, "regge",
                beam_momentum, "regge_fit_replica",
            )
            replica_spectra.append(_bin_tracks(replica_events, tables, sqrt_s_gev))
        m_mean, sigma_mc, sigma_replica = _replica_binned_spectra(
            replica_spectra, tables
        )
        chi2 = 0.0
        species_payload: dict[str, Any] = {}
        for table in tables:
            # Combined per-bin error: data sigma plus the MC standard error
            # of the replica-mean prediction.
            sigma_combined = np.sqrt(table.sigma**2 + sigma_mc[table.species] ** 2)
            residual = (m_mean[table.species] - table.data) / sigma_combined
            chi2_table = float(np.dot(residual, residual))
            metrics = sps_core.metrics_bundle(
                m_mean[table.species],
                table.data,
                table.sigma,
                len(names),
                table.species,
            )
            chi2 += chi2_table
            species_payload[table.species] = {
                "chi2": chi2_table,
                "mc_sigma_mean_over_bins": float(np.mean(sigma_mc[table.species])),
                "mc_sigma_max_over_bins": float(np.max(sigma_mc[table.species])),
                "mc_sigma_per_replica_mean_over_bins": float(
                    np.mean(sigma_replica[table.species])
                ),
                **metrics,
            }
        evaluations.append({
            "parameters": {name: float(v) for name, v in zip(names, x)},
            "chi2": float(chi2),
            "species": species_payload,
        })
        return float(chi2)

    from scipy.optimize import minimize

    result = minimize(
        binned_objective,
        np.asarray(initial),
        method="Powell",
        bounds=bounds,
        options={"maxiter": maxiter, "xtol": 1e-3, "ftol": 1e-3},
    )
    best = min(evaluations, key=lambda item: item["chi2"])
    best_x = np.asarray([best["parameters"][name] for name in names], dtype=float)
    fitted_regge, fitted_continuum = unpack(best_x)
    return fitted_regge, fitted_continuum, {
        "performed": True,
        "objective": "binned",
        "success": bool(result.success),
        "message": str(result.message),
        "chi2": float(best["chi2"]),
        "parameters": dict(best["parameters"]),
        "n_objective_evaluations": int(result.nfev),
        "fit_seed": int(seed),
        "events_per_evaluation": int(n_eval),
        "mc_replicas": {
            "count": int(n_replicas),
            "seed_offset": int(seed_offset),
            "replica_seed_base": int(mc_seed_base),
            "seed_rule": (
                "replica k (k = 0..K-1) is generated from seed = config.seed + "
                "seed_offset + k; the runner passes the fit substream seed = "
                "config.seed + 10000, so the replica base is fit_seed + "
                "(seed_offset - 10000)"
            ),
            "events_per_replica": int(n_eval),
            "stage_label": "regge_fit_replica",
            "prediction": (
                "per-bin model prediction is the mean of the K replica binned "
                "f/sigma_inel densities on the fixed table binning"
            ),
            "sigma_convention": (
                "per-bin MC sigma is the sample standard deviation (ddof=1) of the "
                "per-replica binned densities across the K replicas divided by "
                "sqrt(K), the standard error of the replica mean; the objective "
                "chi-square uses sigma_i^2 = sigma_data,i^2 + sigma_MC,i^2 on every "
                "bin of every binned observable"
            ),
            "diagnostics_convention": (
                "the reported sps_core metrics are computed on the replica-mean "
                "prediction with the data-only sigma (reference quantities, "
                "M4-comparable convention)"
            ),
        },
        "max_iterations": int(maxiter),
        "sqrt_s_gev": float(sqrt_s_gev),
        "sigma_inel_mb": sigma_inel_mb,
        "window": window,
        "sigma_fallback": "poisson_data_content",
        "tables": [
            {
                "species": t.species,
                "pdg": t.pdg_id,
                "mass_gev": t.mass_gev,
                "normalization": t.normalization,
                "path": t.path,
                "n_points": int(t.data.size),
                "n_sigma_fallback": int(t.n_sigma_fallback),
            }
            for t in tables
        ],
        "species": best["species"],
        "trace": evaluations,
    }


# ---------------------------------------------------------------------------
# M11 parameter profiles (card TASK-20260905-physics-review, milestone M11)
#
# One-dimensional profile chi-square machinery on the M8 uncertainty-aware
# binned objective: the six-parameter profile model (the profile set free,
# the five out-of-set parameters frozen at their M8 fitted values), the
# anchor fit, the profile scans with the nuisance parameters refit at every
# scan point, the finite-difference Hessian and correlation estimator, the
# 2-D cross sections (full grids for the in-set pairs, the targeted
# mixed-pair curvature study for the card's mixed pairs), and the artifact
# writers. The pre-registration
# (dev/00runs/parameter_profiles_2026-09-07/pre_registration.json) is
# authoritative: every profile mode cross-checks its constants against the
# file and refuses to run on a mismatch. The existing fit/evaluation
# behavior is unchanged; everything below is additive.
# ---------------------------------------------------------------------------

M11_CARD = "TASK-20260905-physics-review"
M11_MILESTONE = "M11"
M11_PHASE = "A"
M11_RUN_DESCRIPTION = "parameter_profiles_2026-09-07"
M11_STAGE_LABEL = "parameter_profiles_replica"

M11_K_REPLICAS = 8
M11_EVENTS_PER_REPLICA_FULL = 2000
M11_XTOL = 1e-3
M11_FTOL = 1e-3
M11_MAX_ITERATIONS = 10
M11_DOF = 1011
M11_N_BINS_TOTAL = 1017

M11_PROFILE_SET = [
    "regge.alpha_exc0",
    "regge.edge_power",
    "continuum.strange_threshold_power",
    "continuum.pair_threshold_power",
    "continuum.match_mass",
    "continuum.pion_multiplicity_slope",
]

M11_M8_PARAMETER_SET = {
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

# The five out-of-set parameters stay frozen at their M8 fitted values in
# every M11 evaluation (the M10/M9 freeze convention; the M8 combined-error
# set is the calibration of record). continuum.strange_threshold_power is
# NOT in this dict: it was frozen at its bundled value 1.5 in M4/M8 and
# becomes free in the M11 profile model.
M11_FREEZE_VALUES = {
    "regge.effective_transfer_slope": 8.504140762083612,
    "continuum.strange_lambda_strength": 0.30184606498191385,
    "continuum.strange_sigma_strength": 0.3777282734799746,
    "continuum.strange_pair_strength": 0.19187855210738686,
    "continuum.pair_strength": 0.08411858717003319,
}

# continuum.strange_threshold_power was frozen at its bundled value in
# M4/M8 (not an M8-fitted parameter) and becomes free in M11; it enters the
# anchor at this value.
M11_STRANGE_THRESHOLD_M8_FROZEN = 1.5


def m11_m8_start_theta() -> list:
    """The M8 ten-parameter point projected onto the six free parameters."""
    return [
        float(
            M11_M8_PARAMETER_SET.get(n, M11_STRANGE_THRESHOLD_M8_FROZEN)
        )
        for n in M11_PROFILE_SET
    ]

# Governing scan and refit bounds: the M8 pre-registration windows (the
# fit.bounds of the M8 record config; identical to the bundled config for
# all ten M8-fitted parameters). continuum.strange_threshold_power has no
# M8 window (not an M8-fitted parameter), so the bundled bound governs.
M11_BOUNDS = {
    "regge.alpha_exc0": (0.0, 1.2),
    "regge.edge_power": (0.25, 6.0),
    "continuum.strange_threshold_power": (0.0, 6.0),
    "continuum.pair_threshold_power": (0.0, 6.0),
    "continuum.match_mass": (1.5, 4.0),
    "continuum.pion_multiplicity_slope": (0.0, 4.0),
}

M11_FREEZE_BOUNDS = {
    "regge.effective_transfer_slope": (5.07, 10.2),
    "continuum.strange_lambda_strength": (0.0, 0.5),
    "continuum.strange_sigma_strength": (0.0, 0.5),
    "continuum.strange_pair_strength": (0.0, 0.5),
    "continuum.pair_strength": (0.0, 0.15),
}

M11_HESSIAN_LADDERS = {
    "regge.alpha_exc0": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "regge.edge_power": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
    "continuum.strange_threshold_power": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
    "continuum.pair_threshold_power": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
    "continuum.match_mass": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
    "continuum.pion_multiplicity_slope": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
}

M11_MIXED_PAIR_LADDERS = {
    "continuum.strange_lambda_strength": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
    "continuum.pair_strength": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
}

M11_MIXED_PAIRS = {
    "pair1": ("continuum.strange_threshold_power", "continuum.strange_lambda_strength"),
    "pair2": ("continuum.pair_threshold_power", "continuum.pair_strength"),
}

M11_FULL_GRID_PAIRS = {
    "pair0": ("regge.alpha_exc0", "regge.edge_power"),
    "pair3": ("continuum.pion_multiplicity_slope", "continuum.match_mass"),
}

M11_GRID_N = 11
M11_GRID_HALF_WIDTH_SIGMAS = 4.0
M11_GRID_FALLBACK_WINDOW_FRACTION = 0.1
M11_ANCHOR_SCAN_TOLERANCE = 1e-9
M11_DELTA_CHI2 = {"sigma1": 1.0, "sigma2": 4.0}
M11_HESSIAN_ACCEPT = (1.0, 400.0)
M11_HESSIAN_MAX_ADAPTATIONS = 3
M11_HESSIAN_ADAPT_FACTOR = 4.0
M11_HESSIAN_FALLBACK_WINDOW_FRACTION = 0.05
M11_DEGENERACY_RHO = 0.9
M11_FLAT_PROFILE_COVERAGE = 0.9
M11_NEAR_BOUND_WINDOW_FRACTION = 0.1
M11_ELONGATION_THRESHOLD = 9.0
M11_FULL_GRID_N = 7

M11_STATISTICS_MENU = [2000, 1000, 500, 250]
M11_PER_POINT_WALL_BUDGET_S = 5400.0
M11_CAP_FLOOR = 30
M11_CAP_CEILING = 150
M11_CAP_MULTIPLE = 3
M11_ANCHOR_CAP_FLOOR = 60
M11_ANCHOR_CAP_CEILING = 320
M11_ANCHOR_CAP_MULTIPLE = 2
M11_M8_EVALUATIONS = 309

M11_CONFIG_SEED = 20260831
M11_SEED_BASE = 20330831
M11_REPLICA_SEEDS = [20330832 + k for k in range(8)]

M11_SQRT_S_GEV = 17.270205201372406
M11_BEAM_MOMENTUM = 158.0
M11_SIGMA_INEL_MB = 31.78
M11_WINDOW = {"xf_min": 0.0, "xf_max": 0.4, "pt_min": 0.05, "pt_max": 1.5}

M11_CONFIG_BUNDLED_RELATIVE = "dev/mixing2006/config/staged_excitation.json"
M11_CONFIG_BUNDLED_CANONICAL_SHA256 = (
    "3a83d7c8817ef934656c5b45d27d7b9da53af7b51152fab29b4aafa497c7bfee"
)
M11_CONFIG_M8_RELATIVE = (
    "dev/00runs/staged_binned_fit_mc_2026-09-06/config_m4_final.json"
)
M11_CONFIG_M8_CANONICAL_SHA256 = (
    "9b184770cd13a4deae70dd6af9208cac0cd28760905f906b46400a916aa81440"
)
M11_M8_RECORD_RELATIVE = (
    "dev/00runs/staged_binned_fit_mc_2026-09-06/staged_evaluation_report.json"
)
M11_M8_RECORD_SHA256 = (
    "ecd931d58ce79b9aecdd273c703b4421fe0ca9119cdde90d2ad3c7dfb9d49b65"
)
M11_OUTER_HEAD = "a5659350dbe3da93da840e0075c9efc4f32ab7fa"
M11_NESTED_HEAD = "5151cee08e55115f5a53d3d5df7deb372cd5078a"


def _m11_utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _m11_sha256_file(path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _m11_git_head(path: Path):
    import subprocess

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


def _m11_jsonable(obj):
    """Recursively convert numpy scalars, arrays and non-finite floats."""
    if isinstance(obj, dict):
        return {str(k): _m11_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_m11_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _m11_jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _m11_write_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_m11_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _m11_append_trace(trace_path, entry: dict) -> None:
    if trace_path is None:
        return
    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_m11_jsonable(entry), sort_keys=True) + "\n")


def m11_run_dir(default_root=None) -> Path:
    """The M11 run directory (nested-tree relative dev/00runs layout)."""
    if default_root is not None:
        return Path(default_root)
    return Path(_NESTED_ROOT).parent / "00runs" / M11_RUN_DESCRIPTION


def _canonical_config_hash(config: dict) -> str:
    """The canonical configuration hash (sort_keys compact JSON, sha256)."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    import hashlib

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def m11_load_preregistration(path) -> dict:
    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"pre-registration not found at {path}; the protocol must be "
            "pre-registered before any profile run"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def m11_cross_check_preregistration(
    prereg: dict,
    *,
    config_bundled: dict,
    config_m8: dict,
    m8_record_path=None,
    run_dir=None,
) -> None:
    """Refuse to run when the pre-registration and the constants disagree."""
    problems: list[str] = []

    def check(name, expected, actual) -> None:
        if expected != actual:
            problems.append(
                f"{name}: pre-registration {expected!r} != harness {actual!r}"
            )

    check("card", prereg.get("card"), M11_CARD)
    check("milestone", prereg.get("milestone"), M11_MILESTONE)
    check("phase", prereg.get("phase"), M11_PHASE)
    check("run_description", prereg.get("run_description"), M11_RUN_DESCRIPTION)
    mc = prereg.get("machine_constants")
    if not isinstance(mc, dict):
        problems.append("machine_constants block missing")
        raise SystemExit(
            "pre-registration mismatch; refusing to run:\n  " + "\n  ".join(problems)
        )
    check("k_replicas", mc.get("k_replicas"), M11_K_REPLICAS)
    check("events_per_replica_full", mc.get("events_per_replica_full"),
          M11_EVENTS_PER_REPLICA_FULL)
    check("xtol", mc.get("xtol"), M11_XTOL)
    check("ftol", mc.get("ftol"), M11_FTOL)
    check("max_iterations", mc.get("max_iterations"), M11_MAX_ITERATIONS)
    check("degrees_of_freedom", mc.get("degrees_of_freedom"), M11_DOF)
    check("n_bins_total", mc.get("n_bins_total"), M11_N_BINS_TOTAL)
    check("grid_points_per_scan", mc.get("grid_points_per_scan"), M11_GRID_N)
    check("grid_half_width_sigmas", mc.get("grid_half_width_sigmas"),
          M11_GRID_HALF_WIDTH_SIGMAS)
    check("grid_fallback_window_fraction", mc.get("grid_fallback_window_fraction"),
          M11_GRID_FALLBACK_WINDOW_FRACTION)
    check("anchor_scan_point_tolerance", mc.get("anchor_scan_point_tolerance"),
          M11_ANCHOR_SCAN_TOLERANCE)
    check("delta_chi2", mc.get("delta_chi2"), M11_DELTA_CHI2)
    check("hessian_accept_window", mc.get("hessian_accept_window"),
          list(M11_HESSIAN_ACCEPT))
    check("hessian_max_adaptations", mc.get("hessian_max_adaptations"),
          M11_HESSIAN_MAX_ADAPTATIONS)
    check("hessian_adapt_factor", mc.get("hessian_adapt_factor"),
          M11_HESSIAN_ADAPT_FACTOR)
    check("hessian_fallback_window_fraction",
          mc.get("hessian_fallback_window_fraction"),
          M11_HESSIAN_FALLBACK_WINDOW_FRACTION)
    check("hessian_symmetrize", mc.get("hessian_symmetrize"), True)
    check("degeneracy_rho_threshold", mc.get("degeneracy_rho_threshold"),
          M11_DEGENERACY_RHO)
    check("flat_profile_coverage_fraction", mc.get("flat_profile_coverage_fraction"),
          M11_FLAT_PROFILE_COVERAGE)
    check("near_bound_window_fraction", mc.get("near_bound_window_fraction"),
          M11_NEAR_BOUND_WINDOW_FRACTION)
    check("elongation_threshold", mc.get("elongation_threshold"),
          M11_ELONGATION_THRESHOLD)
    check("full_grid_points_per_axis", mc.get("full_grid_points_per_axis"),
          M11_FULL_GRID_N)
    check("statistics_menu", mc.get("statistics_menu_events_per_replica"),
          M11_STATISTICS_MENU)
    check("per_point_wall_budget_s", mc.get("per_point_wall_budget_s"),
          M11_PER_POINT_WALL_BUDGET_S)
    check("cap_rule", mc.get("cap_rule"),
          {"floor": M11_CAP_FLOOR, "ceiling": M11_CAP_CEILING,
           "multiple_of_probe": M11_CAP_MULTIPLE})
    check("anchor_cap_rule", mc.get("anchor_cap_rule"),
          {"floor": M11_ANCHOR_CAP_FLOOR, "ceiling": M11_ANCHOR_CAP_CEILING,
           "multiple_of_m8": M11_ANCHOR_CAP_MULTIPLE,
           "m8_evaluations": M11_M8_EVALUATIONS})
    seed = mc.get("seed", {})
    check("config_seed", seed.get("config_seed"), M11_CONFIG_SEED)
    check("m11_seed_base", seed.get("m11_seed_base"), M11_SEED_BASE)
    check("replica_seeds", seed.get("replica_seeds"), M11_REPLICA_SEEDS)
    check("stage_label", seed.get("stage_label"), M11_STAGE_LABEL)
    check("window", mc.get("window"), M11_WINDOW)
    check("sigma_inel_mb", mc.get("sigma_inel_mb"), M11_SIGMA_INEL_MB)
    check("sqrt_s_gev", mc.get("sqrt_s_gev"), M11_SQRT_S_GEV)
    check("beam_momentum_gev_c", mc.get("beam_momentum_gev_c"), M11_BEAM_MOMENTUM)
    check("m8_parameter_set", mc.get("m8_parameter_set"), M11_M8_PARAMETER_SET)
    check("profile_bounds", mc.get("profile_bounds"),
          {name: list(b) for name, b in M11_BOUNDS.items()})
    check("hessian_step_ladders", mc.get("hessian_step_ladders"), M11_HESSIAN_LADDERS)
    check("mixed_pair_ladders",
          {k: v for k, v in mc.get("mixed_pair_ladders", {}).items() if k != "note"},
          M11_MIXED_PAIR_LADDERS)
    check("freeze_bounds",
          {k: v for k, v in mc.get("freeze_bounds", {}).items() if k != "note"},
          {name: list(b) for name, b in M11_FREEZE_BOUNDS.items()})
    check("freeze_values", mc.get("freeze_values"), M11_FREEZE_VALUES)
    check("profile_set_order", mc.get("profile_set_order"), M11_PROFILE_SET)
    check("twod_pairs", mc.get("twod_pairs"), [
        {"id": "pair0", "members": list(M11_FULL_GRID_PAIRS["pair0"]),
         "mode": "full_grid"},
        {"id": "pair1", "members": list(M11_MIXED_PAIRS["pair1"]),
         "mode": "equivalent_sensitivity_study"},
        {"id": "pair2", "members": list(M11_MIXED_PAIRS["pair2"]),
         "mode": "equivalent_sensitivity_study"},
        {"id": "pair3", "members": list(M11_FULL_GRID_PAIRS["pair3"]),
         "mode": "full_grid"},
    ])
    ps = prereg.get("profile_set", {}).get("parameters", {})
    for name in M11_PROFILE_SET:
        entry = ps.get(name, {})
        check(f"profile_set.{name}.bounds", entry.get("bounds"),
              list(M11_BOUNDS[name]))
        check(f"profile_set.{name}.ladder", entry.get("hessian_step_ladder"),
              M11_HESSIAN_LADDERS[name])
    provenance = prereg.get("provenance", {})
    check("config_bundled_canonical_sha256",
          provenance.get("config_bundled_canonical_sha256"),
          _canonical_config_hash(config_bundled))
    check("config_m8_window_canonical_sha256",
          provenance.get("config_m8_window_canonical_sha256"),
          _canonical_config_hash(config_m8))
    check("outer_head", provenance.get("outer_head_at_preregistration"), M11_OUTER_HEAD)
    check("nested_head", provenance.get("nested_head_at_preregistration"),
          M11_NESTED_HEAD)
    check("m8_record_sha256", provenance.get("m8_record_sha256"),
          M11_M8_RECORD_SHA256)
    if m8_record_path is not None and Path(m8_record_path).exists():
        actual = _m11_sha256_file(m8_record_path)
        if actual != M11_M8_RECORD_SHA256:
            problems.append(
                f"m8_record_file: sha256 {actual!r} != harness "
                f"{M11_M8_RECORD_SHA256!r}"
            )
    if problems:
        raise SystemExit(
            "pre-registration mismatch; refusing to run (the pre-registered "
            "protocol is authoritative):\n  " + "\n  ".join(problems)
        )


def m11_build_model(config: dict):
    """The frozen M11 base model: the ten M8 fitted values applied.

    The M11 profile model starts from the M8 calibration set of record; the
    six profile parameters are varied by :func:`m11_unpack_theta`, the five
    out-of-set parameters stay at these values in every evaluation.
    """
    simple = SimpleExcitationParameters(**(config.get("simple") or {}))
    regge = ReggeExcitationParameters(**(config.get("regge") or {}))
    continuum = ContinuumParameters(**(config.get("continuum") or {}))
    for name, value in M11_M8_PARAMETER_SET.items():
        regge, continuum = _set_parameter(regge, continuum, name, float(value))
    applied = {
        "regge": {k: float(v) for k, v in regge.__dict__.items()},
        "continuum": {
            k: (v if isinstance(v, str) else float(v))
            for k, v in continuum.__dict__.items()
        },
    }
    for name, value in M11_M8_PARAMETER_SET.items():
        group, field = name.split(".", 1)
        if applied[group][field] != float(value):
            raise ValueError(f"M8 value application failed for {name}")
    for name, value in M11_FREEZE_VALUES.items():
        group, field = name.split(".", 1)
        if applied[group][field] != float(value):
            raise ValueError(f"freeze value application failed for {name}")
    if str(applied["continuum"]["decay_hypothesis"]).upper() != "H3":
        raise ValueError("the H3 decay kernel must apply as the dataclass default")
    if abs(float(applied["continuum"]["strange_threshold_power"]) - 1.5) > 0.0:
        raise ValueError("continuum.strange_threshold_power must enter at 1.5")
    simple.validate()
    regge.validate()
    continuum.validate()
    return simple, regge, continuum


def m11_unpack_theta(theta, base_regge, base_continuum, extra_free=None):
    """Set the profile parameters (and optional extra frees) on the base."""
    values = dict(zip(M11_PROFILE_SET, [float(v) for v in theta]))
    if extra_free:
        values.update({name: float(v) for name, v in extra_free})
    regge, continuum = base_regge, base_continuum
    for name, value in values.items():
        regge, continuum = _set_parameter(regge, continuum, name, value)
    regge.validate()
    continuum.validate()
    return regge, continuum


def m11_build_tables(config: dict):
    """The M4 windowed observable set through the production table machinery."""
    binned_cfg = dict(config["fit"]["binned"])
    window_cfg = dict(binned_cfg.get("window", {}))
    window = {
        key: float(window_cfg[key])
        for key in ("xf_min", "xf_max", "pt_min", "pt_max")
        if key in window_cfg
    }
    sigma_inel_mb = float(binned_cfg.get("sigma_inel_mb", M11_SIGMA_INEL_MB))
    path_root = Path(binned_cfg.get("path_root", _NESTED_ROOT))
    entries = list(binned_cfg.get("tables", []))
    if not entries:
        raise ValueError("fit.binned.tables must name at least one table")
    preflight_binned_tables(binned_cfg, path_root)
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


def m11_production_evaluator(
    config: dict,
    *,
    replica_seeds,
    n_events: int,
    trace_path=None,
    extra_free=None,
    extra_free_bounds=None,
    stage_label: str = M11_STAGE_LABEL,
):
    """Build the M11 profile-model objective on the M8 binned convention.

    Returns ``(evaluate, meta)``. ``evaluate(theta)`` takes the six-vector
    of the profile set (plus the extra-free values appended when
    ``extra_free`` is given) and returns a dict with the combined-error
    chi-square over all 1017 bins, the per-species breakdown, the
    conservation-residual maximum and the evaluation wall. The K replicas
    are generated from the fixed ``replica_seeds`` (the shared
    common-random-number block), so the objective is deterministic in the
    configuration and the seeds and identical across all points.
    """
    simple, regge_base, continuum_base = m11_build_model(config)
    tables = m11_build_tables(config)
    n_bins = int(sum(t.data.size for t in tables))
    if n_bins != M11_N_BINS_TOTAL:
        raise ValueError(
            f"the observable set must carry {M11_N_BINS_TOTAL} windowed bins, "
            f"got {n_bins}"
        )
    beam = float(config.get("beam_momentum", M11_BEAM_MOMENTUM))
    sqrt_s = math.sqrt(Reaction(beam, 2212, 2212).s())
    if abs(sqrt_s - M11_SQRT_S_GEV) > 1e-9:
        raise ValueError(
            f"sqrt(s) mismatch: computed {sqrt_s!r} against the pinned "
            f"{M11_SQRT_S_GEV!r}"
        )
    replica_seeds = [int(s) for s in replica_seeds]
    if len(replica_seeds) != M11_K_REPLICAS:
        raise ValueError(
            f"the M11 objective requires K = {M11_K_REPLICAS} replicas, "
            f"got {len(replica_seeds)}"
        )

    def evaluate(theta):
        t0 = time.perf_counter()
        extra = None
        if extra_free is not None:
            tail = [float(v) for v in theta[len(M11_PROFILE_SET):]]
            if len(tail) != len(extra_free):
                raise ValueError(
                    f"expected {len(M11_PROFILE_SET) + len(extra_free)} values, "
                    f"got {len(theta)}"
                )
            extra = list(zip(extra_free, tail))
            theta_core = [float(v) for v in theta[: len(M11_PROFILE_SET)]]
        else:
            theta_core = [float(v) for v in theta]
        if len(theta_core) != len(M11_PROFILE_SET):
            raise ValueError(
                f"expected {len(M11_PROFILE_SET)} profile values, got {len(theta_core)}"
            )
        regge, continuum = m11_unpack_theta(
            theta_core, regge_base, continuum_base, extra_free=extra
        )
        replica_spectra = []
        max_residual = 0.0
        for seed in replica_seeds:
            replica_events, summary = _generate_pp(
                int(n_events), seed, simple, regge, continuum, "regge",
                beam, stage_label,
            )
            max_residual = max(
                max_residual, float(summary["max_conservation_residual"])
            )
            replica_spectra.append(_bin_tracks(replica_events, tables, sqrt_s))
        m_mean, sigma_mc, sigma_replica = _replica_binned_spectra(
            replica_spectra, tables
        )
        chi2 = 0.0
        per_species: dict[str, float] = {}
        mc_summary: dict[str, dict[str, float]] = {}
        for table in tables:
            sigma_combined = np.sqrt(table.sigma**2 + sigma_mc[table.species] ** 2)
            residual = (m_mean[table.species] - table.data) / sigma_combined
            chi2_table = float(np.dot(residual, residual))
            chi2 += chi2_table
            per_species[table.species] = chi2_table
            mc_summary[table.species] = {
                "mc_sigma_mean_over_bins": float(np.mean(sigma_mc[table.species])),
                "mc_sigma_max_over_bins": float(np.max(sigma_mc[table.species])),
            }
        wall = time.perf_counter() - t0
        entry = {
            "theta": {
                name: float(v) for name, v in zip(M11_PROFILE_SET, theta_core)
            },
            "extra_free": (
                {name: float(v) for name, v in extra} if extra else {}
            ),
            "chi2": float(chi2),
            "per_species": per_species,
            "mc_sigma_summary": mc_summary,
            "max_conservation_residual": max_residual,
            "wall_s": wall,
            "utc": _m11_utc_now(),
        }
        _m11_append_trace(trace_path, entry)
        return entry

    meta = {
        "replica_seeds": replica_seeds,
        "n_events_per_replica": int(n_events),
        "events_per_evaluation": int(M11_K_REPLICAS * n_events),
        "stage_label": stage_label,
        "n_bins": n_bins,
        "sqrt_s_gev": sqrt_s,
        "extra_free": list(extra_free or []),
        "extra_free_bounds": (
            {name: list(b) for name, b in (extra_free_bounds or {}).items()}
        ),
        "frozen_record": {
            "regge": {
                k: (None if v is None else float(v))
                for k, v in regge_base.__dict__.items()
            },
            "continuum": {
                k: (v if isinstance(v, str) else float(v))
                for k, v in continuum_base.__dict__.items()
            },
            "simple": {k: float(v) for k, v in simple.__dict__.items()},
        },
    }
    return evaluate, meta


def m11_theta_within_tolerance(value: float, anchor_value: float) -> bool:
    """The pre-registered anchor-scan-point tolerance."""
    return abs(float(value) - float(anchor_value)) <= M11_ANCHOR_SCAN_TOLERANCE * max(
        1.0, abs(float(anchor_value))
    )


def m11_refit_nuisance(
    evaluate,
    *,
    fixed_values: dict,
    theta_start: dict,
    bounds: dict,
    xtol: float = M11_XTOL,
    ftol: float = M11_FTOL,
    max_iterations: int = M11_MAX_ITERATIONS,
    cap_evaluations: int = M11_CAP_CEILING,
    trace_path=None,
):
    """Bounded-Powell refit of the free profile coordinates.

    The coordinates named in ``fixed_values`` are held; the remaining
    profile-set coordinates refit from ``theta_start`` within ``bounds``.
    Returns the best-of-trace point (the runner convention), the refit
    minimum chi2, the evaluation count, and a final re-evaluation whose
    chi2 must reproduce the best bit-identically (the determinism check).
    """
    from scipy.optimize import minimize

    free_names = [n for n in M11_PROFILE_SET if n not in fixed_values]
    if not free_names:
        raise ValueError("a nuisance refit must free at least one coordinate")
    free_pos = [M11_PROFILE_SET.index(n) for n in free_names]
    # The fixed coordinates are HELD at their fixed values (never at the
    # start values): the assembled base vector applies fixed_values first.
    base = [
        float(fixed_values[n]) if n in fixed_values else float(theta_start[n])
        for n in M11_PROFILE_SET
    ]
    x0 = [float(theta_start[n]) for n in free_names]
    bnds = [(float(bounds[n][0]), float(bounds[n][1])) for n in free_names]
    x0 = [min(max(v, lo), hi) for v, (lo, hi) in zip(x0, bnds)]
    history: list[dict] = []

    def objective(x_free) -> float:
        theta = list(base)
        for pos, value in zip(free_pos, x_free):
            theta[pos] = float(value)
        entry = evaluate(theta)
        record = {
            "theta": entry["theta"],
            "chi2": entry["chi2"],
            "wall_s": entry["wall_s"],
        }
        history.append(record)
        _m11_append_trace(trace_path, record)
        return entry["chi2"]

    result = minimize(
        objective,
        np.asarray(x0, dtype=float),
        method="Powell",
        bounds=bnds,
        options={
            "maxiter": int(max_iterations),
            "xtol": float(xtol),
            "ftol": float(ftol),
            "maxfev": int(cap_evaluations),
        },
    )
    if not history:
        raise RuntimeError("the nuisance refit produced no evaluations")
    best = min(history, key=lambda item: item["chi2"])
    theta_best = [float(base[i]) for i in range(len(M11_PROFILE_SET))]
    for pos, name in zip(free_pos, free_names):
        theta_best[pos] = float(best["theta"][name])
    final = evaluate(theta_best)
    determinism_consistent = final["chi2"] == best["chi2"]
    return {
        "best_theta": {
            name: float(v) for name, v in zip(M11_PROFILE_SET, theta_best)
        },
        "fixed_values": {name: float(v) for name, v in fixed_values.items()},
        "free_names": free_names,
        "chi2": float(best["chi2"]),
        "final_check_chi2": float(final["chi2"]),
        "determinism_consistent": bool(determinism_consistent),
        "n_evaluations": int(len(history)),
        "success": bool(result.success),
        "message": str(result.message),
        "max_conservation_residual": float(
            final.get("max_conservation_residual", 0.0)
        ),
        "per_species": final.get("per_species", {}),
        "mc_sigma_summary": final.get("mc_sigma_summary", {}),
        "cap_hit": bool(len(history) >= int(cap_evaluations)),
    }


def m11_single_evaluation(evaluate, theta) -> dict:
    """One logged evaluation at a fixed point (no optimization)."""
    entry = evaluate(theta)
    return {
        "theta": entry["theta"],
        "extra_free": entry.get("extra_free", {}),
        "chi2": float(entry["chi2"]),
        "wall_s": float(entry["wall_s"]),
        "per_species": entry["per_species"],
        "mc_sigma_summary": entry["mc_sigma_summary"],
        "max_conservation_residual": float(entry["max_conservation_residual"]),
    }


def m11_scan_grid(param: str, anchor_value: float, sigma, bounds=None):
    """The pre-registered 1-D scan grid for one profile parameter.

    Half-width W = 4 x sigma_Hessian, clipped to the governing bounds; the
    0.1-window fallback applies when the Hessian sigma is unavailable.
    Returns (grid_points, fallback_flag).
    """
    if bounds is None:
        bounds = M11_BOUNDS[param]
    lo, hi = (float(bounds[0]), float(bounds[1]))
    v = float(anchor_value)
    fallback = False
    if sigma is None or not (float(sigma) > 0.0):
        W = M11_GRID_FALLBACK_WINDOW_FRACTION * (hi - lo)
        fallback = True
    else:
        W = M11_GRID_HALF_WIDTH_SIGMAS * float(sigma)
    g_lo = max(lo, v - W)
    g_hi = min(hi, v + W)
    n = M11_GRID_N
    grid = [g_lo + (g_hi - g_lo) * k / (n - 1) for k in range(n)]
    return grid, fallback


def m11_grid2d_axes(pair_id: str, anchor_values: dict, sigmas: dict,
                    hessian_record: dict | None = None):
    """The pre-registered 7-point axes for one full-grid 2-D pair."""
    names = M11_FULL_GRID_PAIRS[pair_id]
    axes = []
    fallbacks = {}
    for name in names:
        sigma = sigmas.get(name)
        if hessian_record is not None and _m11_sigma_unavailable(name, hessian_record):
            sigma = None
        grid, fallback = m11_scan_grid(
            name,
            anchor_values[name],
            sigma,
            bounds=M11_BOUNDS[name],
        )
        # The 2-D axes use the pre-registered 7-point spacing on the same
        # half-width window.
        lo, hi = grid[0], grid[-1]
        n = M11_FULL_GRID_N
        axes.append([lo + (hi - lo) * k / (n - 1) for k in range(n)])
        fallbacks[name] = fallback
    return axes, fallbacks


def _m11_sigma_unavailable(name: str, hessian_record: dict) -> bool:
    """The pre-registered fallback condition for the scan half-width.

    The Hessian sigma is unavailable when it is None OR when the parameter's
    only accepted rung is a flagged fallback (no rung entered the acceptance
    window, the M8 discreteness regime) or found no positive curvature.
    """
    if hessian_record.get("sigma_per_parameter", {}).get(name) is None:
        return True
    rung = hessian_record.get("rungs", {}).get(name, {})
    flags = rung.get("flags", [])
    return bool(
        "fallback_step_no_window" in flags or "no_positive_curvature" in flags
    )


def m11_expected_grids(anchor_values: dict, hessian_record: dict) -> dict:
    """Recompute every pre-registered grid from the anchor and Hessian.

    Used both for registry materialization and for the artifact
    self-check: the recorded grids must reproduce this function exactly.
    A parameter whose Hessian sigma is unavailable (no acceptance-window
    rung) takes the pre-registered 0.1-window fallback half-width.
    """
    sigmas = hessian_record.get("sigma_per_parameter", {})
    grids_1d = {}
    fallbacks = {}
    for name in M11_PROFILE_SET:
        sigma = sigmas.get(name)
        if _m11_sigma_unavailable(name, hessian_record):
            sigma = None  # the 0.1-window fallback fires
        grid, fallback = m11_scan_grid(name, anchor_values[name], sigma)
        grids_1d[name] = grid
        fallbacks[name] = fallback
    grids_2d = {}
    fallbacks_2d = {}
    for pair_id in M11_FULL_GRID_PAIRS:
        axes, fb = m11_grid2d_axes(pair_id, anchor_values, sigmas,
                                   hessian_record=hessian_record)
        grids_2d[pair_id] = axes
        fallbacks_2d[pair_id] = fb
    return {
        "grids_1d": grids_1d,
        "fallbacks_1d": fallbacks,
        "grids_2d": grids_2d,
        "fallbacks_2d": fallbacks_2d,
    }


def m11_materialize_point_registry(
    anchor_values: dict,
    hessian_record: dict,
    *,
    cap_evaluations: int,
    events_per_replica: int,
) -> list[dict]:
    """Build the production point registry from the pre-registered rules.

    Point ordering (pre-registered): the 1-D scans in profile-set order and
    scan-index order, then the 2-D grids in pair id order row-major. The
    anchor-scan-point rule records a grid point equal to the anchor value
    (or a 2-D point equal to the anchor in both coordinates) as chi2_0
    without a refit.
    """
    expected = m11_expected_grids(anchor_values, hessian_record)
    points: list[dict] = []
    # 1-D profile scans.
    for name in M11_PROFILE_SET:
        for k, value in enumerate(expected["grids_1d"][name]):
            if m11_theta_within_tolerance(value, anchor_values[name]):
                continue  # the anchor-scan-point rule: chi2_0 by construction
            fixed = {name: float(value)}
            points.append({
                "id": f"p1d_{name.replace('.', '__')}_{k:02d}",
                "kind": "nuisance_refit",
                "group": "profile_1d",
                "parameter": name,
                "scan_index": k,
                "scan_value": float(value),
                "fixed": fixed,
                "nuisance": [n for n in M11_PROFILE_SET if n not in fixed],
                "start": dict(anchor_values),
                "bounds": {n: list(M11_BOUNDS[n]) for n in M11_PROFILE_SET},
                "cap_evaluations": int(cap_evaluations),
                "events_per_replica": int(events_per_replica),
            })
    # 2-D full grids.
    for pair_id in sorted(M11_FULL_GRID_PAIRS):
        axis0, axis1 = expected["grids_2d"][pair_id]
        names = M11_FULL_GRID_PAIRS[pair_id]
        for i, v0 in enumerate(axis0):
            for j, v1 in enumerate(axis1):
                if m11_theta_within_tolerance(v0, anchor_values[names[0]]) and \
                        m11_theta_within_tolerance(v1, anchor_values[names[1]]):
                    continue  # both coordinates at the anchor: chi2_0
                fixed = {names[0]: float(v0), names[1]: float(v1)}
                points.append({
                    "id": f"p2d_{pair_id}_i{i:02d}_j{j:02d}",
                    "kind": "nuisance_refit",
                    "group": "profile_2d",
                    "pair_id": pair_id,
                    "grid_i": i,
                    "grid_j": j,
                    "fixed": fixed,
                    "nuisance": [n for n in M11_PROFILE_SET if n not in fixed],
                    "start": dict(anchor_values),
                    "bounds": {n: list(M11_BOUNDS[n]) for n in M11_PROFILE_SET},
                    "cap_evaluations": int(cap_evaluations),
                    "events_per_replica": int(events_per_replica),
                })
    return points


def m11_profile_intervals(
    grid_values,
    grid_chi2,
    *,
    theta0: float,
    chi2_0: float,
    thresholds=None,
):
    """Read the 1-sigma and 2-sigma intervals from a profile curve.

    The profile curve is the union of the anchor point (theta0, chi2_0)
    and the scan grid, sorted. A level crossing between two adjacent curve
    points that bracket the level is located by parabolic interpolation
    through the three consecutive curve points containing the bracket
    (exact on a locally quadratic profile), with a linear-interpolation
    fallback when the parabola has no root inside the bracket; the method
    used is recorded per edge. A level not bracketed on a side because the
    extreme curve point is at or below the level is reported as 'interval
    at scan edge, quoted one-sided' (at_edge flag).
    """
    if thresholds is None:
        thresholds = M11_DELTA_CHI2
    curve = sorted(
        [(float(theta0), float(chi2_0))]
        + [(float(v), float(c)) for v, c in zip(grid_values, grid_chi2)]
    )
    deduped = [curve[0]]
    for point in curve[1:]:
        if not m11_theta_within_tolerance(point[0], deduped[-1][0]):
            deduped.append(point)
    curve = deduped

    def _crossing(k: int, level: float):
        """Crossing inside the bracket (curve[k], curve[k+1]); (value, method)."""
        x0, c0 = curve[k]
        x1, c1 = curve[k + 1]
        if c1 == c0:
            return None, "degenerate"
        if k >= 1:
            tx = [curve[k - 1][0], curve[k][0], curve[k + 1][0]]
            tc = [curve[k - 1][1], curve[k][1], curve[k + 1][1]]
        else:
            tx = [curve[0][0], curve[1][0], curve[2][0]]
            tc = [curve[0][1], curve[1][1], curve[2][1]]
        roots = []
        method = "linear"
        denom = (tx[0] - tx[1]) * (tx[0] - tx[2]) * (tx[1] - tx[2])
        if denom != 0.0:
            a = (
                tx[2] * (tc[1] - tc[0])
                + tx[1] * (tc[0] - tc[2])
                + tx[0] * (tc[2] - tc[1])
            ) / denom
            if tx[1] != tx[0]:
                b = (tc[1] - tc[0]) / (tx[1] - tx[0]) - a * (tx[1] + tx[0])
                c_coef = tc[1] - a * tx[1] * tx[1] - b * tx[1]
                aa, bb, cc = a, b, c_coef - level
                if abs(aa) < 1e-30:
                    if bb != 0.0:
                        roots.append(-cc / bb)
                else:
                    disc = bb * bb - 4.0 * aa * cc
                    if disc >= 0.0:
                        sq = math.sqrt(disc)
                        roots.append((-bb + sq) / (2.0 * aa))
                        roots.append((-bb - sq) / (2.0 * aa))
        lo_b = min(x0, x1)
        hi_b = max(x0, x1)
        tol_lo = 1e-12 * max(1.0, abs(lo_b))
        tol_hi = 1e-12 * max(1.0, abs(hi_b))
        for root in roots:
            if lo_b - tol_lo <= root <= hi_b + tol_hi:
                return float(root), "parabolic"
        frac = (level - c0) / (c1 - c0)
        return float(x0 + frac * (x1 - x0)), method

    results = {}
    for key, level_delta in thresholds.items():
        level = float(chi2_0) + float(level_delta)
        lower = None
        upper = None
        at_edge_lower = False
        at_edge_upper = False
        method_lower = None
        method_upper = None
        for k in range(len(curve) - 1):
            c0 = curve[k][1]
            c1 = curve[k + 1][1]
            if (c0 - level) * (c1 - level) <= 0.0 and not (
                c0 == level and c1 == level
            ):
                crossing, method = _crossing(k, level)
                if crossing is None:
                    continue
                if crossing < theta0 and lower is None:
                    lower = crossing
                    method_lower = method
                elif crossing >= theta0 and upper is None:
                    upper = crossing
                    method_upper = method
        if lower is None and curve[0][1] <= level:
            at_edge_lower = True
            lower = curve[0][0]
        if upper is None and curve[-1][1] <= level:
            at_edge_upper = True
            upper = curve[-1][0]
        results[key] = {
            "level_chi2": level,
            "lower": lower,
            "upper": upper,
            "at_edge_lower": bool(at_edge_lower),
            "at_edge_upper": bool(at_edge_upper),
            "one_sided": bool(at_edge_lower or at_edge_upper),
            "method_lower": method_lower,
            "method_upper": method_upper,
        }
    return results


def m11_correlation_from_hessian(h_matrix):
    """Correlation matrix from the profile-model Hessian (the primary
    estimator): symmetrize, invert (pseudo-inverse fallback with a
    degeneracy flag), rho_ij = C_ij / sqrt(C_ii C_jj)."""
    H = np.asarray(h_matrix, dtype=float)
    H_sym = 0.5 * (H + H.T)
    eigenvalues = np.linalg.eigvalsh(H_sym)
    positive_definite = bool(np.all(eigenvalues > 0.0))
    if positive_definite:
        C = np.linalg.inv(H_sym)
        used_pinv = False
    else:
        C = np.linalg.pinv(H_sym)
        used_pinv = True
    diag = np.sqrt(np.maximum(np.diag(C), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = C / np.outer(diag, diag)
    rho = np.where(np.isfinite(rho), rho, 0.0)
    np.fill_diagonal(rho, 1.0)
    return {
        "rho": rho,
        "covariance": C,
        "eigenvalues": eigenvalues,
        "positive_definite": positive_definite,
        "used_pinv": used_pinv,
    }


def m11_hessian_rung(
    evaluate,
    theta: list,
    index: int,
    ladder,
    chi2_0: float,
    bounds,
    trace_path=None,
):
    """One diagonal rung search: the first ladder step with 1 <= D <= 400.

    Steps are clipped to the governing bounds (a clipped step is recorded);
    up to the pre-registered number of adaptations by factors of four
    beyond the ladder; the fallback step is a fraction of the window width.
    Returns (accepted_h, D, H_ii, trials, flags).
    """
    name = M11_PROFILE_SET[index]
    lo, hi = (float(bounds[name][0]), float(bounds[name][1]))
    v = float(theta[index])
    trials = []
    flags: list[str] = []
    steps = list(ladder)
    for adaptation in range(M11_HESSIAN_MAX_ADAPTATIONS):
        steps.append(float(ladder[-1]) * (M11_HESSIAN_ADAPT_FACTOR ** (adaptation + 1)))
    accepted_h = None
    accepted_D = None
    fallback_h = M11_HESSIAN_FALLBACK_WINDOW_FRACTION * (hi - lo)
    for h in steps:
        hp = min(v + h, hi)
        hm = max(v - h, lo)
        h_eff_p = hp - v
        h_eff_m = v - hm
        if h_eff_p <= 0.0 and h_eff_m <= 0.0:
            trials.append({"h": float(h), "clipped": True, "skipped": True})
            continue
        theta_p = list(theta)
        theta_p[index] = hp
        theta_m = list(theta)
        theta_m[index] = hm
        cp = evaluate(theta_p)["chi2"]
        cm = evaluate(theta_m)["chi2"]
        D = float(cp) + float(cm) - 2.0 * float(chi2_0)
        h_eff = 0.5 * (h_eff_p + h_eff_m)
        trials.append({
            "h": float(h),
            "h_eff": float(h_eff),
            "clipped": bool(h_eff_p != h or h_eff_m != h),
            "D": float(D),
        })
        if M11_HESSIAN_ACCEPT[0] <= D <= M11_HESSIAN_ACCEPT[1]:
            accepted_h = float(h_eff)
            accepted_D = float(D)
            break
    if accepted_h is None:
        # The pre-registered fallback step (flagged; used when D > 0).
        hp = min(v + fallback_h, hi)
        hm = max(v - fallback_h, lo)
        theta_p = list(theta)
        theta_p[index] = hp
        theta_m = list(theta)
        theta_m[index] = hm
        cp = evaluate(theta_p)["chi2"]
        cm = evaluate(theta_m)["chi2"]
        D = float(cp) + float(cm) - 2.0 * float(chi2_0)
        h_eff = 0.5 * ((hp - v) + (v - hm))
        trials.append({
            "h": float(fallback_h),
            "h_eff": float(h_eff),
            "clipped": bool((hp - v) != fallback_h or (v - hm) != fallback_h),
            "D": float(D),
            "fallback": True,
        })
        if D > 0.0 and h_eff > 0.0:
            accepted_h = float(h_eff)
            accepted_D = float(D)
            flags.append("fallback_step_no_window")
            flags.append(f"D_at_fallback={D:.3e}")
        else:
            flags.append("no_positive_curvature")
    if accepted_h is None or accepted_D is None:
        return None, None, None, trials, flags
    H_ii = float(accepted_D) / (float(accepted_h) ** 2)
    return accepted_h, accepted_D, H_ii, trials, flags


def m11_run_hessian(
    evaluate,
    *,
    anchor_values: dict,
    chi2_0: float,
    evaluator_factory,
    out_dir,
    trace_path=None,
) -> dict:
    """The profile-model Hessian at the anchor: diagonal rungs, the 15
    mixed pairs, the correlation matrix, and the targeted mixed-pair
    curvature studies (the pre-registered equivalent sensitivity study)."""
    theta_anchor = [float(anchor_values[n]) for n in M11_PROFILE_SET]
    H = np.zeros((6, 6), dtype=float)
    rungs: dict[str, dict] = {}
    for i, name in enumerate(M11_PROFILE_SET):
        h, D, H_ii, trials, flags = m11_hessian_rung(
            evaluate, theta_anchor, i, M11_HESSIAN_LADDERS[name], chi2_0, M11_BOUNDS,
            trace_path=trace_path,
        )
        rungs[name] = {
            "accepted_h": h,
            "D": D,
            "H_ii": H_ii,
            "trials": trials,
            "flags": flags,
        }
        if H_ii is not None:
            H[i, i] = H_ii
    for i in range(6):
        for j in range(i + 1, 6):
            hi_ = rungs[M11_PROFILE_SET[i]]["accepted_h"]
            hj_ = rungs[M11_PROFILE_SET[j]]["accepted_h"]
            key = f"{M11_PROFILE_SET[i]}|{M11_PROFILE_SET[j]}"
            if hi_ is None or hj_ is None:
                rungs[key] = {"skipped": "diagonal rung unavailable"}
                continue
            base = list(theta_anchor)
            combos = {}
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    theta = list(base)
                    pi_ = min(max(theta[i] + si * hi_, M11_BOUNDS[M11_PROFILE_SET[i]][0]),
                              M11_BOUNDS[M11_PROFILE_SET[i]][1])
                    pj_ = min(max(theta[j] + sj * hj_, M11_BOUNDS[M11_PROFILE_SET[j]][0]),
                              M11_BOUNDS[M11_PROFILE_SET[j]][1])
                    theta[i] = pi_
                    theta[j] = pj_
                    combos[(si, sj)] = evaluate(theta)["chi2"]
            H_ij = (
                combos[(1.0, 1.0)] - combos[(1.0, -1.0)]
                - combos[(-1.0, 1.0)] + combos[(-1.0, -1.0)]
            ) / (4.0 * hi_ * hj_)
            H[i, j] = H_ij
            H[j, i] = H_ij
            rungs[key] = {"h_i": float(hi_), "h_j": float(hj_), "H_ij": float(H_ij)}
    corr = m11_correlation_from_hessian(H)
    sigma_per_parameter = {}
    C = corr["covariance"]
    for i, name in enumerate(M11_PROFILE_SET):
        var = float(C[i, i])
        # Profile-likelihood convention: chi2 = (x-mu)^T V^{-1} (x-mu) has the
        # second-derivative matrix H = 2 V^{-1}, so the covariance is
        # V = 2 H^{-1} and the 1-sigma width is sqrt(2 C_ii). The correlation
        # matrix is unaffected by the factor.
        sigma_per_parameter[name] = (
            float(math.sqrt(2.0 * var))
            if var > 0.0 and rungs[name]["H_ii"] is not None
            else None
        )
    # Targeted mixed-pair curvature studies (partners temporarily freed).
    mixed_pair_results = {}
    for pair_id, (inset_name, partner_name) in M11_MIXED_PAIRS.items():
        inset_index = M11_PROFILE_SET.index(inset_name)
        evaluate_pair = evaluator_factory(
            extra_free=[partner_name],
            extra_free_bounds={partner_name: M11_FREEZE_BOUNDS[partner_name]},
            trace_path=trace_path,
        )
        theta7 = list(theta_anchor) + [float(M11_FREEZE_VALUES[partner_name])]
        chi2_pair0 = evaluate_pair(theta7)["chi2"]
        ladder_inset = M11_HESSIAN_LADDERS[inset_name]
        ladder_partner = M11_MIXED_PAIR_LADDERS[partner_name]

        # Diagonal rung for the in-set member (index within the 6-vector).
        h_in, D_in, H_in, trials_in, flags_in = m11_hessian_rung(
            evaluate_pair, theta7, inset_index, ladder_inset, chi2_pair0,
            {**M11_BOUNDS, partner_name: M11_FREEZE_BOUNDS[partner_name]},
            trace_path=trace_path,
        )
        # Diagonal rung for the partner (index 6 of the 7-vector).
        partner_index = 6
        lo_p, hi_p = M11_FREEZE_BOUNDS[partner_name]
        vp = float(theta7[partner_index])
        H_partner = None
        h_partner = None
        trials_partner = []
        for h in list(ladder_partner) + [
            float(ladder_partner[-1]) * (M11_HESSIAN_ADAPT_FACTOR ** (a + 1))
            for a in range(M11_HESSIAN_MAX_ADAPTATIONS)
        ]:
            pp = min(vp + h, hi_p)
            pm = max(vp - h, lo_p)
            if (pp - vp) <= 0.0 and (vp - pm) <= 0.0:
                continue
            t_up = list(theta7)
            t_up[partner_index] = pp
            t_dn = list(theta7)
            t_dn[partner_index] = pm
            cp = evaluate_pair(t_up)["chi2"]
            cm = evaluate_pair(t_dn)["chi2"]
            D = float(cp) + float(cm) - 2.0 * float(chi2_pair0)
            h_eff = 0.5 * ((pp - vp) + (vp - pm))
            trials_partner.append({"h": float(h), "h_eff": float(h_eff), "D": float(D)})
            if M11_HESSIAN_ACCEPT[0] <= D <= M11_HESSIAN_ACCEPT[1]:
                h_partner = float(h_eff)
                H_partner = float(D) / (h_eff ** 2)
                break
        rho_pair = None
        elongation = None
        if H_in is not None and H_partner is not None and h_in is not None \
                and h_partner is not None:
            h_in_f = float(h_in)
            h_partner_f = float(h_partner)
            base7 = list(theta7)
            combos = {}
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    theta = list(base7)
                    pi_ = min(max(theta[inset_index] + si * h_in_f,
                                  M11_BOUNDS[inset_name][0]), M11_BOUNDS[inset_name][1])
                    pj_ = min(max(theta[partner_index] + sj * h_partner_f, lo_p), hi_p)
                    theta[inset_index] = pi_
                    theta[partner_index] = pj_
                    combos[(si, sj)] = evaluate_pair(theta)["chi2"]
            H_ip = (
                combos[(1.0, 1.0)] - combos[(1.0, -1.0)]
                - combos[(-1.0, 1.0)] + combos[(-1.0, -1.0)]
            ) / (4.0 * h_in_f * h_partner_f)
            H2 = np.array([[H_in, H_ip], [H_ip, H_partner]], dtype=float)
            H2s = 0.5 * (H2 + H2.T)
            evals2 = np.linalg.eigvalsh(H2s)
            if np.all(evals2 > 0.0):
                C2 = np.linalg.inv(H2s)
                rho_pair = float(
                    C2[0, 1] / math.sqrt(C2[0, 0] * C2[1, 1])
                )
                elongation = float(math.sqrt(evals2.max() / evals2.min()))
            else:
                C2 = np.linalg.pinv(H2s)
                denom = math.sqrt(max(C2[0, 0], 0.0) * max(C2[1, 1], 0.0))
                rho_pair = float(C2[0, 1] / denom) if denom > 0 else None
                elongation = None
            mixed_pair_results[pair_id] = {
                "members": [inset_name, partner_name],
                "chi2_at_anchor_with_partner_freed": float(chi2_pair0),
                "h_inset": float(h_in_f),
                "D_inset": (float(D_in) if D_in is not None else None),
                "H_inset": float(H_in),
                "h_partner": float(h_partner_f),
                "H_partner": float(H_partner),
                "H_mixed": float(H_ip),
                "rho_pair": rho_pair,
                "elongation_ratio": elongation,
                "eigenvalues": [float(e) for e in evals2],
                "trials_inset": trials_in,
                "trials_partner": trials_partner,
                "flags": flags_in,
            }
        else:
            mixed_pair_results[pair_id] = {
                "members": [inset_name, partner_name],
                "skipped": "a diagonal rung was unavailable",
                "trials_inset": trials_in,
                "trials_partner": trials_partner,
            }
    payload = {
        "H": H,
        "eigenvalues": [float(e) for e in corr["eigenvalues"]],
        "positive_definite": corr["positive_definite"],
        "used_pinv": corr["used_pinv"],
        "rho": corr["rho"],
        "sigma_per_parameter": sigma_per_parameter,
        "rungs": rungs,
        "mixed_pair_results": mixed_pair_results,
    }
    _m11_write_json(Path(out_dir) / "hessian.json", payload)
    return payload


def m11_run_anchor(
    evaluate,
    *,
    out_dir,
    trace_path=None,
    cap_evaluations: int = M11_ANCHOR_CAP_CEILING,
):
    """The anchor: the six-parameter bounded-Powell fit from the M8 point."""
    from scipy.optimize import minimize

    theta_start = m11_m8_start_theta()
    bnds = [tuple(M11_BOUNDS[n]) for n in M11_PROFILE_SET]
    history: list[dict] = []

    def objective(x) -> float:
        entry = evaluate(list(x))
        record = {
            "theta": entry["theta"],
            "chi2": entry["chi2"],
            "wall_s": entry["wall_s"],
        }
        history.append(record)
        _m11_append_trace(trace_path, record)
        return entry["chi2"]

    t0 = time.perf_counter()
    result = minimize(
        objective,
        np.asarray(theta_start, dtype=float),
        method="Powell",
        bounds=bnds,
        options={
            "maxiter": M11_MAX_ITERATIONS,
            "xtol": M11_XTOL,
            "ftol": M11_FTOL,
            "maxfev": int(cap_evaluations),
        },
    )
    wall = time.perf_counter() - t0
    if not history:
        raise RuntimeError("the anchor fit produced no evaluations")
    best = min(history, key=lambda item: item["chi2"])
    theta_best = [float(best["theta"][n]) for n in M11_PROFILE_SET]
    final = evaluate(theta_best)
    determinism_consistent = final["chi2"] == best["chi2"]
    deltas = {
        name: float(best["theta"][name]) - float(M11_M8_PARAMETER_SET[name])
        for name in M11_PROFILE_SET
        if name in M11_M8_PARAMETER_SET
    }
    payload = {
        "anchor_values": {name: float(best["theta"][name]) for name in M11_PROFILE_SET},
        "chi2_0": float(best["chi2"]),
        "chi2_at_m8_point_first_evaluation": float(history[0]["chi2"]),
        "distance_from_m8_shared_parameters": deltas,
        "n_evaluations": int(len(history)),
        "cap_evaluations": int(cap_evaluations),
        "cap_hit": bool(len(history) >= int(cap_evaluations)),
        "success": bool(result.success),
        "message": str(result.message),
        "wall_s": float(wall),
        "final_check_chi2": float(final["chi2"]),
        "determinism_consistent": bool(determinism_consistent),
        "per_species": final["per_species"],
        "mc_sigma_summary": final["mc_sigma_summary"],
        "max_conservation_residual": float(final["max_conservation_residual"]),
        "start_point": {
            name: (
                float(M11_M8_PARAMETER_SET[name])
                if name in M11_M8_PARAMETER_SET
                else float(M11_STRANGE_THRESHOLD_M8_FROZEN)
            )
            for name in M11_PROFILE_SET
        },
    }
    _m11_write_json(Path(out_dir) / "anchor.json", payload)
    return payload


def run_staged_evaluation(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[ExcitationEvent]]]:
    through = EvaluationStage.parse(config.get("through", "baryon_nucleus"))
    seed = int(config.get("seed", 20260831))
    n_events = int(config.get("events_per_stage", 500))
    if n_events < 1:
        raise ValueError("events_per_stage must be positive")
    beam_momentum = float(config.get("beam_momentum", 158.0))
    simple = _dataclass_from_dict(SimpleExcitationParameters, config.get("simple"))
    regge = _dataclass_from_dict(ReggeExcitationParameters, config.get("regge"))
    continuum = _dataclass_from_dict(ContinuumParameters, config.get("continuum"))
    canonical_config = json.dumps(config, sort_keys=True, separators=(",", ":"))
    import hashlib
    report: dict[str, Any] = {
        "seed": seed,
        "configuration_hash": hashlib.sha256(canonical_config.encode("utf-8")).hexdigest(),
        "through": through.name.lower(),
        "stage_order": [],
        "configuration": {
            "simple": asdict(simple),
            "regge_initial": asdict(regge),
            "continuum_initial": asdict(continuum),
        },
    }
    # RNG substream record (Round 1, card 5, additive): gated by the
    # config key substreams.enabled (default false).  The block is run
    # metadata only; the generated events are identical with and
    # without it.
    if bool(dict(config.get("substreams", {})).get("enabled", False)):
        report["substreams"] = {
            "master_seed": int(seed),
            "replica_seed_rule": _SUBSTREAM_REPLICA_SEED_RULE,
            "substream_labels": list(rng_substreams.SUBSTREAM_LABELS),
        }
    samples: dict[str, list[ExcitationEvent]] = {}

    simple_events, simple_summary = _generate_pp(
        n_events, seed, simple, regge, continuum, "simple",
        beam_momentum, "simple_excitation",
    )
    samples["simple_excitation"] = simple_events
    report["stage_order"].append("simple_excitation")
    report["simple_excitation"] = {
        "status": "passed",
        "summary": simple_summary,
        "parameters": asdict(simple),
    }
    if through == EvaluationStage.SIMPLE_EXCITATION:
        return report, samples

    fitted_regge, fitted_continuum, fit_report = _fit_regge(
        dict(config.get("fit", {})),
        simple, regge, continuum, beam_momentum, seed + 10_000,
    )
    regge_events, regge_summary = _generate_pp(
        n_events, seed + 1, simple, fitted_regge, fitted_continuum,
        "regge", beam_momentum, "regge_fit",
    )
    samples["regge_fit"] = regge_events
    report["stage_order"].append("regge_fit")
    report["regge_fit"] = {
        "status": "passed",
        "fit": fit_report,
        "summary": regge_summary,
        "parameters": {
            "regge": asdict(fitted_regge),
            "continuum": asdict(fitted_continuum),
        },
    }
    if through == EvaluationStage.REGGE_FIT:
        return report, samples

    nucleus = _dataclass_from_dict(NucleusSpec, config.get("nucleus"))
    pa_cfg = dict(config.get("pA", {}))
    pa_builder = PAEventBuilder(
        PAExcitationConfig(
            beam_momentum=beam_momentum,
            production_mode="regge",
            simple=simple,
            regge=fitted_regge,
            continuum=fitted_continuum,
            nucleus=nucleus,
            glauber_dr_fm=float(pa_cfg.get("glauber_dr_fm", 0.05)),
            max_event_attempts=int(pa_cfg.get("max_event_attempts", 100)),
        ),
        np.random.default_rng(seed + 2),
    )
    pa_events = [pa_builder.generate(i) for i in range(n_events)]
    for event in pa_events:
        event.metadata["random_seed"] = int(seed + 2)
        event.metadata["random_substream"] = f"baryon_nucleus:{event.event_id}"
    pa_summary = _summarize(pa_events)
    samples["baryon_nucleus"] = pa_events
    report["stage_order"].append("baryon_nucleus")
    report["baryon_nucleus"] = {
        "status": "passed",
        "summary": pa_summary,
        "nucleus": asdict(nucleus),
        "frozen_pp_parameters": {
            "regge": asdict(fitted_regge),
            "continuum": asdict(fitted_continuum),
        },
    }
    return report, samples


def m11_make_evaluator_factory(config: dict, events_per_replica: int):
    """Wire one production evaluator factory (one RNG environment)."""

    def factory(extra_free=None, extra_free_bounds=None, trace_path=None):
        evaluate, _meta = m11_production_evaluator(
            config,
            replica_seeds=M11_REPLICA_SEEDS,
            n_events=int(events_per_replica),
            trace_path=trace_path,
            extra_free=extra_free,
            extra_free_bounds=extra_free_bounds,
        )
        return evaluate

    return factory


def m11_default_paths(run_dir=None, preregistration=None):
    run_dir = m11_run_dir(run_dir)
    prereg_path = (
        Path(preregistration) if preregistration else run_dir / "pre_registration.json"
    )
    return run_dir, prereg_path


def _m11_load_and_check(prereg_path, config):
    """Shared entry: load the pre-registration, cross-check, return it."""
    prereg = m11_load_preregistration(prereg_path)
    bundled = json.loads(
        (Path(_NESTED_ROOT) / "config" / "staged_excitation.json").read_text()
    )
    m8_record = Path(_NESTED_ROOT).parent / M11_M8_RECORD_RELATIVE
    m11_cross_check_preregistration(
        prereg,
        config_bundled=bundled,
        config_m8=config,
        m8_record_path=m8_record,
    )
    prereg_sha = _m11_sha256_file(prereg_path)
    print(
        f"[m11] prereg {prereg_path} (sha256 {prereg_sha[:12]}...): "
        "constants verified",
        flush=True,
    )
    return prereg, prereg_sha


def m11_run_determinism_probe(config, run_dir, preregistration=None, n_repeats=3):
    """Probe 1: repeated evaluations at the M8 point must be bit-identical."""
    run_dir, prereg_path = m11_default_paths(run_dir, preregistration)
    prereg, _sha = _m11_load_and_check(prereg_path, config)
    evaluate, _meta = m11_production_evaluator(
        config,
        replica_seeds=M11_REPLICA_SEEDS,
        n_events=M11_EVENTS_PER_REPLICA_FULL,
        trace_path=run_dir / "traces" / "trace_probe_determinism.jsonl",
    )
    theta_m8 = m11_m8_start_theta()
    chi2s = []
    walls = []
    residuals = []
    for _ in range(int(n_repeats)):
        entry = evaluate(theta_m8)
        chi2s.append(float(entry["chi2"]))
        walls.append(float(entry["wall_s"]))
        residuals.append(float(entry["max_conservation_residual"]))
    payload = {
        "probe": "determinism",
        "n_repeats": int(n_repeats),
        "theta": {name: float(v) for name, v in zip(M11_PROFILE_SET, theta_m8)},
        "chi2_values": chi2s,
        "chi2_reference_m8_point": chi2s[0],
        "deterministic": bool(len(set(chi2s)) == 1),
        "wall_s": walls,
        "mean_wall_s": float(np.mean(walls)),
        "max_conservation_residual": float(max(residuals)),
        "note": (
            "the chi2 at the M8 ten-parameter point under the M11 replica "
            "block is a fresh convention reference and is not comparable to "
            "the M8 record's 44088.090230523405 (different replica seeds)"
        ),
    }
    _m11_write_json(run_dir / "probe_determinism.json", payload)
    print(
        f"[m11 probe] determinism: chi2 {chi2s[0]!r} x{n_repeats}, "
        f"deterministic={payload['deterministic']}, "
        f"mean wall {payload['mean_wall_s']:.2f} s",
        flush=True,
    )
    return payload


def m11_run_nuisance_probe(config, run_dir, preregistration=None):
    """Probe 4: one representative 5-parameter nuisance refit.

    Runs after the anchor and Hessian: the displacement is
    4 x sigma_Hessian(alpha_exc0) from its anchor value (the 0.1-window
    fallback if the sigma is unavailable), the other five profile
    parameters refit from the anchor values at the ceiling cap.
    """
    run_dir, prereg_path = m11_default_paths(run_dir, preregistration)
    prereg, _sha = _m11_load_and_check(prereg_path, config)
    anchor = json.loads((run_dir / "anchor.json").read_text())
    hessian = json.loads((run_dir / "hessian.json").read_text())
    name = M11_PROFILE_SET[0]
    anchor_values = anchor["anchor_values"]
    sigma = hessian["sigma_per_parameter"].get(name)
    grid, fallback = m11_scan_grid(name, anchor_values[name], sigma)
    # The probe displaces to the grid edge nearest the +4-sigma side.
    target = max(grid)
    if fallback:
        print("[m11 probe] nuisance probe uses the fallback half-width", flush=True)
    evaluate = m11_make_evaluator_factory(config, M11_EVENTS_PER_REPLICA_FULL)(
        trace_path=run_dir / "traces" / "trace_probe_nuisance.jsonl"
    )
    fixed = {name: float(target)}
    res = m11_refit_nuisance(
        evaluate,
        fixed_values=fixed,
        theta_start=anchor_values,
        bounds=M11_BOUNDS,
        cap_evaluations=M11_CAP_CEILING,
    )
    payload = {
        "probe": "nuisance_refit",
        "parameter_fixed": name,
        "fixed_value": float(target),
        "fallback_half_width": bool(fallback),
        "chi2": res["chi2"],
        "chi2_0": float(anchor["chi2_0"]),
        "n_evaluations": res["n_evaluations"],
        "success": res["success"],
        "message": res["message"],
        "cap_hit": res["cap_hit"],
        "determinism_consistent": res["determinism_consistent"],
        "cap_rule": (
            f"cap_evals = min({M11_CAP_CEILING}, max({M11_CAP_FLOOR}, "
            f"{M11_CAP_MULTIPLE} x measured))"
        ),
    }
    _m11_write_json(run_dir / "probe_nuisance.json", payload)
    print(
        f"[m11 probe] nuisance refit at {name}={target:.6g}: "
        f"chi2 {res['chi2']:.1f} in {res['n_evaluations']} evaluations "
        f"(cap hit: {res['cap_hit']})",
        flush=True,
    )
    return payload


def m11_run_anchor_cli(config, run_dir, preregistration=None):
    run_dir, prereg_path = m11_default_paths(run_dir, preregistration)
    prereg, _sha = _m11_load_and_check(prereg_path, config)
    cap = min(
        M11_ANCHOR_CAP_CEILING,
        max(M11_ANCHOR_CAP_FLOOR, M11_ANCHOR_CAP_MULTIPLE * M11_M8_EVALUATIONS),
    )
    evaluate = m11_make_evaluator_factory(config, M11_EVENTS_PER_REPLICA_FULL)(
        trace_path=run_dir / "traces" / "trace_anchor.jsonl"
    )
    return m11_run_anchor(
        evaluate,
        out_dir=run_dir,
        trace_path=run_dir / "traces" / "trace_anchor.jsonl",
        cap_evaluations=cap,
    )


def m11_run_hessian_cli(config, run_dir, preregistration=None, events_per_replica=None):
    run_dir, prereg_path = m11_default_paths(run_dir, preregistration)
    prereg, _sha = _m11_load_and_check(prereg_path, config)
    anchor = json.loads((run_dir / "anchor.json").read_text())
    events = int(events_per_replica or M11_EVENTS_PER_REPLICA_FULL)
    factory = m11_make_evaluator_factory(config, events)
    evaluate = factory(trace_path=run_dir / "traces" / "trace_hessian.jsonl")
    return m11_run_hessian(
        evaluate,
        anchor_values=anchor["anchor_values"],
        chi2_0=float(anchor["chi2_0"]),
        evaluator_factory=factory,
        out_dir=run_dir,
        trace_path=run_dir / "traces" / "trace_hessian.jsonl",
    )


def m11_run_point_cli(config, run_dir, preregistration=None, point_id=None):
    """Run one registered production point (the gateway job entry)."""
    if point_id is None:
        raise SystemExit("--profile-run point requires --point-id")
    run_dir, prereg_path = m11_default_paths(run_dir, preregistration)
    prereg, _sha = _m11_load_and_check(prereg_path, config)
    registry = prereg.get("point_registry", {}).get("points", [])
    point = next((p for p in registry if p["id"] == point_id), None)
    if point is None:
        raise SystemExit(
            f"point id {point_id!r} is not in the materialized point_registry"
        )
    anchor = json.loads((run_dir / "anchor.json").read_text())
    events = int(point.get("events_per_replica", M11_EVENTS_PER_REPLICA_FULL))
    evaluate = m11_make_evaluator_factory(config, events)(
        trace_path=run_dir / "traces" / f"trace_{point_id}.jsonl"
    )
    if point["kind"] == "fixed":
        result = m11_single_evaluation(
            evaluate, [float(point["theta_full"][n]) for n in M11_PROFILE_SET]
        )
        result["point_id"] = point_id
        result["status"] = "completed"
        _m11_write_json(run_dir / f"point_{point_id}.json", result)
        return result
    res = m11_refit_nuisance(
        evaluate,
        fixed_values={k: float(v) for k, v in point["fixed"].items()},
        theta_start={n: float(v) for n, v in point["start"].items()},
        bounds={n: tuple(point["bounds"][n]) for n in M11_PROFILE_SET},
        cap_evaluations=int(point["cap_evaluations"]),
    )
    payload = {
        "point_id": point_id,
        "kind": point["kind"],
        "group": point.get("group"),
        "parameter": point.get("parameter"),
        "scan_index": point.get("scan_index"),
        "scan_value": point.get("scan_value"),
        "pair_id": point.get("pair_id"),
        "grid_i": point.get("grid_i"),
        "grid_j": point.get("grid_j"),
        "fixed": point["fixed"],
        "chi2": res["chi2"],
        "best_theta": res["best_theta"],
        "n_evaluations": res["n_evaluations"],
        "cap_evaluations": int(point["cap_evaluations"]),
        "cap_hit": res["cap_hit"],
        "success": res["success"],
        "message": res["message"],
        "determinism_consistent": res["determinism_consistent"],
        "final_check_chi2": res["final_check_chi2"],
        "per_species": res["per_species"],
        "mc_sigma_summary": res["mc_sigma_summary"],
        "max_conservation_residual": res["max_conservation_residual"],
        "statistics": {
            "k_replicas": M11_K_REPLICAS,
            "events_per_replica": events,
        },
        "replica_seeds": M11_REPLICA_SEEDS,
        "status": "completed",
        "finished_utc": _m11_utc_now(),
    }
    _m11_write_json(run_dir / f"point_{point_id}.json", payload)
    print(
        f"[m11 point {point_id}] chi2 {res['chi2']:.1f} "
        f"({res['n_evaluations']} evaluations, cap hit {res['cap_hit']})",
        flush=True,
    )
    return payload


def m11_run_profile_1d_cli(config, run_dir, preregistration=None, parameter=None):
    """Serial fallback: run one parameter's full 1-D scan from the registry."""
    if parameter is None:
        raise SystemExit("--profile-run profile-1d requires --parameter")
    run_dir, prereg_path = m11_default_paths(run_dir, preregistration)
    prereg, _sha = _m11_load_and_check(prereg_path, config)
    registry = prereg.get("point_registry", {}).get("points", [])
    points = [p for p in registry if p.get("group") == "profile_1d"
              and p.get("parameter") == parameter]
    if not points:
        raise SystemExit(f"no profile_1d registry points for {parameter!r}")
    results = []
    for point in points:
        path = run_dir / f"point_{point['id']}.json"
        if path.exists():
            payload = json.loads(path.read_text())
            if payload.get("status") == "completed":
                results.append(payload)
                continue
        results.append(
            m11_run_point_cli(config, run_dir, preregistration, point_id=point["id"])
        )
    return results


def m11_run_profile_2d_cli(config, run_dir, preregistration=None, pair_id=None):
    """Serial fallback: run one pair's full 2-D grid from the registry."""
    if pair_id is None:
        raise SystemExit("--profile-run profile-2d requires --pair-id")
    run_dir, prereg_path = m11_default_paths(run_dir, preregistration)
    prereg, _sha = _m11_load_and_check(prereg_path, config)
    registry = prereg.get("point_registry", {}).get("points", [])
    points = [p for p in registry if p.get("group") == "profile_2d"
              and p.get("pair_id") == pair_id]
    if not points:
        raise SystemExit(f"no profile_2d registry points for {pair_id!r}")
    results = []
    for point in points:
        path = run_dir / f"point_{point['id']}.json"
        if path.exists():
            payload = json.loads(path.read_text())
            if payload.get("status") == "completed":
                results.append(payload)
                continue
        results.append(
            m11_run_point_cli(config, run_dir, preregistration, point_id=point["id"])
        )
    return results


def m11_degeneracy_decisions(
    anchor_values: dict,
    hessian: dict,
    profile_tables: dict,
) -> dict:
    """The pre-registered mechanical decision mapping (spec section 6.3)."""
    rho = np.asarray(hessian["rho"], dtype=float)
    mixed = hessian.get("mixed_pair_results", {})
    mixed_rho = {}
    for pair_id, result in mixed.items():
        if result.get("rho_pair") is not None:
            inset_name = result["members"][0]
            partner_name = result["members"][1]
            mixed_rho[inset_name] = {
                "rho_pair": float(result["rho_pair"]),
                "partner": partner_name,
                "elongation_ratio": result.get("elongation_ratio"),
            }
    decisions = {}
    for i, name in enumerate(M11_PROFILE_SET):
        # rho_max over the 6x6 matrix (off-diagonal) plus the mixed pairs.
        candidates = []
        for j in range(6):
            if j == i:
                continue
            candidates.append((abs(float(rho[i, j])), M11_PROFILE_SET[j], "hessian"))
        if name in mixed_rho:
            candidates.append(
                (abs(mixed_rho[name]["rho_pair"]), mixed_rho[name]["partner"],
                 "mixed_pair_targeted")
            )
        rho_max, partner, source = max(candidates, key=lambda item: item[0])
        table = profile_tables.get(name, {})
        interval2 = (table.get("intervals") or {}).get("sigma2", {})
        lower = interval2.get("lower")
        upper = interval2.get("upper")
        lo, hi = M11_BOUNDS[name]
        if lower is not None and upper is not None:
            coverage = (float(upper) - float(lower)) / (float(hi) - float(lo))
        else:
            coverage = 1.0  # a one-sided interval reaching both edges is full
        one_sided = interval2.get("one_sided", False)
        anchor_v = float(anchor_values[name])
        near_bound = min(
            abs(anchor_v - float(lo)), abs(float(hi) - anchor_v)
        ) <= M11_NEAR_BOUND_WINDOW_FRACTION * (float(hi) - float(lo))
        if rho_max < M11_DEGENERACY_RHO:
            decision = "retained as-is"
            reason = (
                f"max |rho| = {rho_max:.4f} < {M11_DEGENERACY_RHO} (vs "
                f"{partner}, {source} estimator); the 2-sigma profile "
                f"interval covers {100.0 * coverage:.1f} percent of the "
                "governing window; spec section 6.3 requires no action"
            )
        elif coverage >= M11_FLAT_PROFILE_COVERAGE:
            decision = "removed"
            reason = (
                f"max |rho| = {rho_max:.4f} >= {M11_DEGENERACY_RHO} (vs "
                f"{partner}, {source} estimator) and the 2-sigma profile "
                f"interval covers {100.0 * coverage:.1f} percent of the "
                "governing window: the data identify nothing about this "
                "parameter inside its window; spec section 6.3 'removed'"
            )
        elif near_bound:
            decision = "constrained externally"
            reason = (
                f"max |rho| = {rho_max:.4f} >= {M11_DEGENERACY_RHO} (vs "
                f"{partner}, {source} estimator) and the anchor value sits "
                "within 10 percent of the governing window width from a "
                "bound: the governing bound is the external constraint; "
                "spec section 6.3 'constrained externally'"
            )
        else:
            decision = "grouped"
            reason = (
                f"max |rho| = {rho_max:.4f} >= {M11_DEGENERACY_RHO} (vs "
                f"{partner}, {source} estimator) with an informative 2-sigma "
                f"profile interval ({100.0 * coverage:.1f} percent of the "
                "governing window): the model keeps both parameters and "
                "reports them jointly; spec section 6.3 'grouped'"
            )
        decisions[name] = {
            "decision": decision,
            "reason": reason,
            "rho_max": float(rho_max),
            "degenerate_partner": partner,
            "rho_source": source,
            "coverage_fraction_sigma2": float(coverage),
            "interval_one_sided": bool(one_sided),
            "near_bound": bool(near_bound),
        }
    freeze_status = {
        name: {
            "status": "constrained externally (frozen at the M8 calibration "
                      "value in every M11 evaluation)",
            "m8_value": float(M11_FREEZE_VALUES[name]),
        }
        for name in M11_FREEZE_VALUES
    }
    for pair_id in ("pair1", "pair2"):
        result = mixed.get(pair_id, {})
        if result.get("rho_pair") is not None:
            partner = result["members"][1]
            freeze_status[partner]["mixed_pair_correlation"] = {
                "with": result["members"][0],
                "rho_pair": float(result["rho_pair"]),
                "elongation_ratio": result.get("elongation_ratio"),
            }
    return {"profile_set": decisions, "freeze_set": freeze_status}


def m11_collect(config, run_dir, preregistration=None, execution_note=""):
    """Merge every artifact into the report; refuse before all points exist."""
    run_dir, prereg_path = m11_default_paths(run_dir, preregistration)
    prereg, prereg_sha = _m11_load_and_check(prereg_path, config)
    anchor = json.loads((run_dir / "anchor.json").read_text())
    hessian = json.loads((run_dir / "hessian.json").read_text())
    registry = prereg.get("point_registry", {}).get("points", [])
    if not registry:
        raise SystemExit("the point_registry is not materialized; refusing to collect")
    missing = []
    artifacts = {}
    for point in registry:
        path = run_dir / f"point_{point['id']}.json"
        if not path.exists():
            missing.append(point["id"])
            continue
        payload = json.loads(path.read_text())
        if payload.get("status") != "completed":
            missing.append(point["id"])
            continue
        artifacts[point["id"]] = payload
    if missing:
        raise SystemExit(
            "refusing to write the report before every production point is "
            "present and completed; missing: " + ", ".join(missing)
        )
    # The grids of record must reproduce the pre-registered rule exactly.
    expected = m11_expected_grids(anchor["anchor_values"], hessian)
    for name in M11_PROFILE_SET:
        grid_expected = expected["grids_1d"][name]
        for point in [p for p in registry if p.get("parameter") == name]:
            k = point["scan_index"]
            if abs(float(point["scan_value"]) - grid_expected[k]) > 0.0:
                raise SystemExit(
                    f"registry grid drift for {name} index {k}: "
                    f"{point['scan_value']!r} != {grid_expected[k]!r}"
                )
    # Per-parameter profile tables and intervals.
    chi2_0 = float(anchor["chi2_0"])
    profile_tables = {}
    for name in M11_PROFILE_SET:
        grid = expected["grids_1d"][name]
        entries = []
        for k, value in enumerate(grid):
            if m11_theta_within_tolerance(value, anchor["anchor_values"][name]):
                entries.append({"scan_value": float(value), "chi2": chi2_0,
                                "source": "anchor"})
                continue
            point = next(
                p for p in registry
                if p.get("parameter") == name and p["scan_index"] == k
            )
            artifact = artifacts[point["id"]]
            entries.append({
                "scan_value": float(value),
                "chi2": float(artifact["chi2"]),
                "source": point["id"],
                "n_evaluations": int(artifact["n_evaluations"]),
                "cap_hit": bool(artifact["cap_hit"]),
            })
        values = [e["scan_value"] for e in entries]
        chi2s = [e["chi2"] for e in entries]
        intervals = m11_profile_intervals(
            values, chi2s,
            theta0=float(anchor["anchor_values"][name]),
            chi2_0=chi2_0,
        )
        off_anchor = [e["chi2"] for e in entries if e["source"] != "anchor"]
        lo, hi = M11_BOUNDS[name]
        interval2 = intervals["sigma2"]
        if interval2["lower"] is not None and interval2["upper"] is not None:
            # The interval span relative to the governing window (the same
            # definition the decision mapping uses; the at-edge bounds are
            # the scan-grid edges).
            coverage = (interval2["upper"] - interval2["lower"]) / (hi - lo)
        else:
            coverage = None
        profile_tables[name] = {
            "bounds": [lo, hi],
            "grid_half_width_rule": (
                f"4 x sigma_Hessian ({'recorded' if not expected['fallbacks_1d'][name] else 'fallback: 0.1 x window'})"
            ),
            "sigma_hessian": hessian["sigma_per_parameter"].get(name),
            "grid_fallback": bool(expected["fallbacks_1d"][name]),
            "table": entries,
            "profile_chi2_min_off_anchor": (
                float(min(off_anchor)) if off_anchor else None
            ),
            "chi2_0": chi2_0,
            "intervals": intervals,
            "coverage_fraction_sigma2": coverage,
            "at_edge_any": bool(
                interval2["at_edge_lower"] or interval2["at_edge_upper"]
                or intervals["sigma1"]["at_edge_lower"]
                or intervals["sigma1"]["at_edge_upper"]
            ),
            "anchor_value": float(anchor["anchor_values"][name]),
        }
    # 2-D full grids.
    twod_tables = {}
    for pair_id in sorted(M11_FULL_GRID_PAIRS):
        names = M11_FULL_GRID_PAIRS[pair_id]
        axis0, axis1 = expected["grids_2d"][pair_id]
        cells = []
        for i, v0 in enumerate(axis0):
            for j, v1 in enumerate(axis1):
                if m11_theta_within_tolerance(v0, anchor["anchor_values"][names[0]]) \
                        and m11_theta_within_tolerance(
                            v1, anchor["anchor_values"][names[1]]):
                    cells.append({"i": i, "j": j, "values": [float(v0), float(v1)],
                                  "chi2": chi2_0, "source": "anchor"})
                    continue
                point = next(
                    p for p in registry
                    if p.get("pair_id") == pair_id and p["grid_i"] == i
                    and p["grid_j"] == j
                )
                artifact = artifacts[point["id"]]
                cells.append({
                    "i": i, "j": j, "values": [float(v0), float(v1)],
                    "chi2": float(artifact["chi2"]), "source": point["id"],
                    "n_evaluations": int(artifact["n_evaluations"]),
                    "cap_hit": bool(artifact["cap_hit"]),
                })
        inset_indices = [M11_PROFILE_SET.index(n) for n in names]
        H = np.asarray(hessian["H"], dtype=float)
        H2 = H[np.ix_(inset_indices, inset_indices)]
        evals2 = np.linalg.eigvalsh(0.5 * (H2 + H2.T))
        elongation = (
            float(math.sqrt(evals2.max() / evals2.min()))
            if np.all(evals2 > 0.0) else None
        )
        off = [c["chi2"] for c in cells if c["source"] != "anchor"]
        twod_tables[pair_id] = {
            "members": list(names),
            "axis0": [float(v) for v in axis0],
            "axis1": [float(v) for v in axis1],
            "cells": cells,
            "chi2_min_off_anchor": float(min(off)) if off else None,
            "hessian_2x2_eigenvalues": [float(e) for e in evals2],
            "elongation_ratio": elongation,
            "elongation_threshold": M11_ELONGATION_THRESHOLD,
            "elongation_corroborates_degeneracy": bool(
                elongation is not None and elongation >= M11_ELONGATION_THRESHOLD
            ),
        }
    decisions = m11_degeneracy_decisions(
        anchor["anchor_values"], hessian, profile_tables
    )
    gateway_manifest = run_dir / "gateway_manifest.json"
    if gateway_manifest.exists():
        execution = {
            "mode": "gateway_grid",
            "manifest": json.loads(gateway_manifest.read_text()),
        }
    else:
        execution = {"mode": "serial_local", "manifest": None}
    execution["note"] = execution_note
    execution["point_wall_s"] = {
        pid: float(a.get("wall_s_total", 0.0)) for pid, a in artifacts.items()
    }
    deviations = list(prereg.get("deviations", []))
    for pid, artifact in artifacts.items():
        if artifact.get("cap_hit"):
            deviations.append(
                f"{pid}: hit the per-point evaluation cap "
                f"({artifact['n_evaluations']} evaluations); the fitted "
                "value is a capped descent (the M7 H1-cap honesty convention)"
            )
    report = {
        "schema": "parameter_profiles/v1",
        "card": M11_CARD,
        "milestone": M11_MILESTONE,
        "phase": M11_PHASE,
        "run_description": M11_RUN_DESCRIPTION,
        "date_utc": _m11_utc_now(),
        "pre_registration": {
            "path": str(prereg_path),
            "sha256": prereg_sha,
        },
        "profile_model": prereg["profile_model"],
        "anchor": anchor,
        "hessian": {
            "rho": hessian["rho"],
            "H": hessian["H"],
            "eigenvalues": hessian["eigenvalues"],
            "positive_definite": hessian["positive_definite"],
            "used_pinv": hessian["used_pinv"],
            "sigma_per_parameter": hessian["sigma_per_parameter"],
            "rungs": hessian["rungs"],
        },
        "mixed_pair_results": hessian.get("mixed_pair_results", {}),
        "profile_set": M11_PROFILE_SET,
        "freeze_values": M11_FREEZE_VALUES,
        "thresholds": M11_DELTA_CHI2,
        "profile_tables": profile_tables,
        "twod_tables": twod_tables,
        "degeneracy_decisions": decisions,
        "execution": execution,
        "deviations": deviations,
        "provenance": {
            "outer_head": _m11_git_head(Path(_NESTED_ROOT).parents[1]),
            "nested_head": _m11_git_head(Path(_NESTED_ROOT)),
            "config_path": prereg["provenance"]["config_m8_window_path"],
            "configuration_hash": _canonical_config_hash(config),
            "m8_record": prereg["provenance"]["m8_record"],
            "m8_record_sha256": M11_M8_RECORD_SHA256,
            "preregistration_sha256": prereg_sha,
        },
        "decision_rule_of_record": prereg["decision_rule_of_record"],
        "finished_utc": _m11_utc_now(),
    }
    _m11_write_json(run_dir / "parameter_profiles_report.json", report)
    _m11_write_report_md(run_dir, report, prereg)
    print(f"[m11 collect] wrote {run_dir / 'parameter_profiles_report.json'}", flush=True)
    return report


def _m11_write_report_md(run_dir, report, prereg) -> None:
    """The run report (protocol, decision of record, tables, provenance)."""
    anchor = report["anchor"]
    hessian = report["hessian"]
    lines: list[str] = []
    lines.append("# Parameter profiles and degeneracies — 2026-09-08")
    lines.append("")
    lines.append(
        "Card TASK-20260905-physics-review, milestone M11, phase A. The "
        "pre-registered production profile run of the six-parameter profile "
        "model under the M8 uncertainty-aware binned objective (1017 bins, "
        "K = 8 replicas x 2000 events, H3 kernel), executed exactly as "
        "pre-registered in `pre_registration.json`."
    )
    lines.append("")
    lines.append("## Anchor")
    lines.append("")
    lines.append(
        f"chi2_0 = {anchor['chi2_0']:.6f} on {M11_DOF} degrees of freedom "
        f"(1017 fitted bins minus 6 free parameters), from "
        f"{anchor['n_evaluations']} Powell evaluations in "
        f"{anchor['wall_s']:.1f} s. The fit starts at the M8 ten-parameter "
        "point; chi2 at that point under the M11 replica block is "
        f"{anchor['chi2_at_m8_point_first_evaluation']:.6f} (a fresh "
        "convention reference)."
    )
    lines.append("")
    lines.append("| parameter | anchor value | M8 value | delta |")
    lines.append("| --- | --- | --- | --- |")
    for name in M11_PROFILE_SET:
        m8v = M11_M8_PARAMETER_SET.get(name)
        delta = anchor["distance_from_m8_shared_parameters"].get(name)
        lines.append(
            f"| {name} | {anchor['anchor_values'][name]:.6f} | "
            f"{m8v if m8v is not None else '1.5 (frozen in M4/M8)'} | "
            f"{delta if delta is not None else 'n/a (newly free)'} |"
        )
    lines.append("")
    lines.append("## Profile tables")
    lines.append("")
    for name in M11_PROFILE_SET:
        table = report["profile_tables"][name]
        lines.append(f"### {name}")
        lines.append("")
        lines.append(
            f"Grid: {len(table['table'])} points on "
            f"[{table['table'][0]['scan_value']:.6g}, "
            f"{table['table'][-1]['scan_value']:.6g}] "
            f"({table['grid_half_width_rule']}); profile chi2 minimum "
            f"off-anchor {table['profile_chi2_min_off_anchor']:.4f} against "
            f"chi2_0 {table['chi2_0']:.4f}."
        )
        lines.append("")
        for key, label in (("sigma1", "1-sigma"), ("sigma2", "2-sigma")):
            iv = table["intervals"][key]
            edge = ""
            if iv["one_sided"]:
                edge = " (interval at scan edge, quoted one-sided)"
            lines.append(
                f"- {label} (delta-chi2 = {M11_DELTA_CHI2[key]}): "
                f"[{iv['lower']:.6g}, {iv['upper']:.6g}]{edge}"
            )
        lines.append("")
    lines.append("## Correlation matrix (the primary estimator)")
    lines.append("")
    rho = report["hessian"]["rho"]
    lines.append("| | " + " | ".join(
        n.split(".")[-1] for n in M11_PROFILE_SET) + " |")
    lines.append("|" + " --- |" * (len(M11_PROFILE_SET) + 1))
    for i, name in enumerate(M11_PROFILE_SET):
        row = " | ".join(f"{rho[i][j]:+.3f}" for j in range(len(M11_PROFILE_SET)))
        lines.append(f"| {name.split('.')[-1]} | {row} |")
    lines.append("")
    lines.append("## 2-D cross sections")
    lines.append("")
    for pair_id, table in report["twod_tables"].items():
        lines.append(
            f"- {pair_id} ({table['members'][0]}, {table['members'][1]}): "
            f"7 x 7 grid, chi2 minimum off anchor "
            f"{table['chi2_min_off_anchor']:.4f}, elongation ratio "
            f"{table['elongation_ratio'] if table['elongation_ratio'] is not None else 'n/a'}"
        )
    lines.append("")
    lines.append("## Degeneracy decisions (spec section 6.3)")
    lines.append("")
    for name, entry in report["degeneracy_decisions"]["profile_set"].items():
        lines.append(f"- **{name}: {entry['decision']}** — {entry['reason']}")
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    prov = report["provenance"]
    lines.append(
        f"- Outer HEAD {prov['outer_head']}, nested HEAD {prov['nested_head']}; "
        f"pre-registration sha256 {prov['preregistration_sha256']}."
    )
    lines.append("")
    lines.append(
        "The decisions are applied to the spec by the doc-writer from this "
        "artifact; this record carries the evidence and the rule outcomes."
    )
    lines.append("")
    path = run_dir / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def m11_self_check(config, run_dir, preregistration=None) -> int:
    """Re-read the pre-registration against the recorded run (self-check)."""
    run_dir, prereg_path = m11_default_paths(run_dir, preregistration)
    prereg, prereg_sha = _m11_load_and_check(prereg_path, config)
    report_path = run_dir / "parameter_profiles_report.json"
    if not report_path.exists():
        raise SystemExit(f"report not found at {report_path}")
    report = json.loads(report_path.read_text())
    anchor = json.loads((run_dir / "anchor.json").read_text())
    hessian = json.loads((run_dir / "hessian.json").read_text())
    problems: list[str] = []
    # The grids of record.
    expected = m11_expected_grids(anchor["anchor_values"], hessian)
    for name in M11_PROFILE_SET:
        recorded = [e["scan_value"] for e in report["profile_tables"][name]["table"]]
        if recorded != [float(v) for v in expected["grids_1d"][name]]:
            problems.append(f"grid drift for {name}")
    # The intervals re-derived from the recorded tables.
    for name in M11_PROFILE_SET:
        table = report["profile_tables"][name]
        values = [e["scan_value"] for e in table["table"]]
        chi2s = [e["chi2"] for e in table["table"]]
        recomputed = m11_profile_intervals(
            values, chi2s,
            theta0=float(anchor["anchor_values"][name]),
            chi2_0=float(anchor["chi2_0"]),
        )
        if recomputed != table["intervals"]:
            problems.append(f"interval drift for {name}")
    # The correlation matrix re-derived from the recorded Hessian.
    corr = m11_correlation_from_hessian(hessian["H"])
    if not np.allclose(
        np.asarray(hessian["rho"], dtype=float), corr["rho"], rtol=0.0, atol=0.0
    ):
        if not np.array_equal(np.asarray(hessian["rho"], dtype=float), corr["rho"]):
            problems.append("correlation matrix drift against the recorded H")
    # Every registry point present and completed.
    registry = prereg.get("point_registry", {}).get("points", [])
    for point in registry:
        path = run_dir / f"point_{point['id']}.json"
        if not path.exists():
            problems.append(f"missing point artifact {point['id']}")
            continue
        payload = json.loads(path.read_text())
        if payload.get("status") != "completed":
            problems.append(f"point {point['id']} not completed")
        if not payload.get("determinism_consistent"):
            problems.append(f"point {point['id']} determinism check failed")
    # The decisions re-derived from the rule.
    decisions = m11_degeneracy_decisions(
        anchor["anchor_values"], hessian, report["profile_tables"]
    )
    if decisions != report["degeneracy_decisions"]:
        problems.append("degeneracy decisions drift against the pre-registered rule")
    if report["pre_registration"]["sha256"] != prereg_sha:
        problems.append("pre-registration sha drift in the report")
    if problems:
        raise SystemExit("artifact self-check FAILED:\n  " + "\n  ".join(problems))
    print(
        "[m11 self-check] OK: the report matches the pre-registered protocol, "
        "the anchor and Hessian artifacts, and every point artifact; the "
        "decisions re-derive from the rule",
        flush=True,
    )
    return 0


M11_PROFILE_RUN_MODES = (
    "determinism-probe",
    "anchor",
    "hessian",
    "nuisance-probe",
    "point",
    "profile-1d",
    "profile-2d",
    "collect",
    "self-check",
)


def _m11_main(args) -> int:
    """The M11 profile-mode dispatcher (additive; the existing paths are
    unchanged when --profile-run is absent)."""
    run_dir, prereg_path = m11_default_paths(args.run_dir, args.preregistration)
    config = json.loads(Path(args.config).read_text())
    mode = args.profile_run
    if mode == "determinism-probe":
        m11_run_determinism_probe(config, run_dir, args.preregistration)
    elif mode == "anchor":
        m11_run_anchor_cli(config, run_dir, args.preregistration)
    elif mode == "hessian":
        m11_run_hessian_cli(config, run_dir, args.preregistration)
    elif mode == "nuisance-probe":
        m11_run_nuisance_probe(config, run_dir, args.preregistration)
    elif mode == "point":
        m11_run_point_cli(config, run_dir, args.preregistration, point_id=args.point_id)
    elif mode == "profile-1d":
        m11_run_profile_1d_cli(
            config, run_dir, args.preregistration, parameter=args.parameter
        )
    elif mode == "profile-2d":
        m11_run_profile_2d_cli(
            config, run_dir, args.preregistration, pair_id=args.pair_id
        )
    elif mode == "collect":
        m11_collect(config, run_dir, args.preregistration,
                    execution_note=args.execution_note)
    elif mode == "self-check":
        return m11_self_check(config, run_dir, args.preregistration)
    else:
        raise SystemExit(f"unknown --profile-run mode {mode!r}")
    return 0


# ---------------------------------------------------------------------------
# Sector-wise deviance decomposition (Round 1, card 8, additive)
# ---------------------------------------------------------------------------

SECTOR_REPORT_FILENAME = "sector_report.json"
SECTOR_REPORT_OBSERVABLE = "f_xsect"


def _sector_wise_requested(config: dict[str, Any], flag: bool = False) -> bool:
    """The sector-report switch: the ``--sector-report`` flag or the
    ``report.sector_wise`` config key (both default off)."""
    if flag:
        return True
    report_cfg = config.get("report")
    if isinstance(report_cfg, dict):
        return bool(report_cfg.get("sector_wise", False))
    return False


def _build_sector_report(
    config: dict[str, Any],
    samples: dict[str, list[ExcitationEvent]],
) -> tuple[dict, float, int]:
    """Build the staged evaluation's sector decomposition (diagnostic).

    The decomposition consumes the same sources as the M8 binned
    objective where they exist: the windowed NA49 tables through the
    production table machinery (``m11_build_tables`` — the per-table
    normalization declaration, the paper window, and the Poisson
    fallback uncertainty give the data values ``d = table.data`` and
    the per-bin sigma), and the per-species binned invariant densities
    of one generated event sample (``_bin_tracks``) as the model
    prediction ``m``.  The M8 objective's replica-mean prediction and
    its MC sigma term exist only inside a performed fit; the smoke run
    executes with the fit disabled (20 events per stage), so the sector
    report consumes the staged run's own ``regge_fit`` sample (regge
    mode at the staged parameters — the fitted parameters when the fit
    ran, the initial ones otherwise) under the retained single-sample,
    data-only convention of ``_binned_mc_spectra``: the combined
    covariance is the data-only diagonal ``diag(sigma**2)`` with no MC
    term.  The global ``total_chi2`` is that convention's chi-square
    over all tables; ``dof`` is the windowed bin count (no parameter is
    fitted on the diagnostic path).  Purely additive: nothing here
    changes the staged run, the fit, or the M8 objective.
    """
    binned_cfg = dict(dict(config.get("fit", {})).get("binned", {}))
    if not binned_cfg:
        raise SystemExit(
            "--sector-report requires a fit.binned block naming the "
            "windowed NA49 tables (the bundled smoke and production "
            "configs both carry one); there is nothing to decompose"
        )
    sample = samples.get("regge_fit") or []
    if not sample:
        raise SystemExit(
            "--sector-report needs the regge_fit stage sample; set "
            "config 'through' to regge_fit or baryon_nucleus"
        )
    tables = m11_build_tables(config)
    beam = float(config.get("beam_momentum", 158.0))
    sqrt_s_gev = math.sqrt(Reaction(beam, 2212, 2212).s())
    densities = _bin_tracks(sample, tables, sqrt_s_gev)
    sectors: dict[str, dict] = {}
    unassigned = 0
    total_chi2 = 0.0
    dof = 0
    for table in tables:
        xedges = sps_core.edges_from_centers(table.xf_centers)
        pedges = sps_core.edges_from_centers(table.pt_centers)
        # Per-bin edges in the table's gather order (the sparse-tabulation
        # form of the sector machinery: every tabulated point carries the
        # edges of its own cell in the fit binning).
        x_edges_flat: list[float] = []
        p_edges_flat: list[float] = []
        for ix, ip in zip(
            table.bin_index_xf.tolist(), table.bin_index_pt.tolist()
        ):
            x_edges_flat.append(float(xedges[ix]))
            x_edges_flat.append(float(xedges[ix + 1]))
            p_edges_flat.append(float(pedges[ip]))
            p_edges_flat.append(float(pedges[ip + 1]))
        observable = BinnedObservable(
            species=table.species,
            observable=SECTOR_REPORT_OBSERVABLE,
            bin_edges=(tuple(x_edges_flat), tuple(p_edges_flat)),
            values=tuple(float(v) for v in table.data),
            stat_errors=tuple(float(v) for v in table.sigma),
        )
        model = tuple(float(v) for v in densities[table.species])
        decomposition = sector_decomposition(
            observable, model, diagonal_covariance(table.sigma), None
        )
        sectors.update(decomposition["sectors"])
        unassigned += int(decomposition["unassigned_bins"])
        residual = (np.asarray(model) - table.data) / table.sigma
        total_chi2 += float(np.dot(residual, residual))
        dof += int(table.data.size)
    merged = {"sectors": sectors, "unassigned_bins": int(unassigned)}
    return merged, float(total_chi2), int(dof)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON staged-evaluation configuration")
    parser.add_argument("--output", default="staged_evaluation_report.json")
    parser.add_argument(
        "--events-output",
        default=None,
        help="optional JSONL file containing complete event records",
    )
    # --- M11 profile machinery (additive; defaults leave the paths above
    # unchanged) ---
    parser.add_argument(
        "--profile-run",
        default=None,
        choices=M11_PROFILE_RUN_MODES,
        help="run one M11 parameter-profile mode instead of the staged evaluation",
    )
    parser.add_argument("--point-id", default=None, help="point id from the materialized point_registry")
    parser.add_argument("--parameter", default=None, help="parameter name for --profile-run profile-1d")
    parser.add_argument("--pair-id", default=None, help="pair id for --profile-run profile-2d")
    parser.add_argument(
        "--preregistration",
        default=None,
        help="pre-registration JSON (default: <run-dir>/pre_registration.json)",
    )
    parser.add_argument("--run-dir", default=None, help="M11 run directory")
    parser.add_argument("--execution-note", default="", help="free-text note recorded by --profile-run collect")
    # --- Sector-wise deviance report (Round 1, card 8, additive; the
    # default off leaves every path above unchanged) ---
    parser.add_argument(
        "--sector-report",
        action="store_true",
        help=(
            "write sector_report.json (the sector-wise deviance "
            "decomposition over the pre-registered SECTOR_REGIONS grid, "
            "with the worst-bin pull and the residual mean per sector) "
            "alongside the staged evaluation report; off by default, "
            "also selectable through the report.sector_wise config key"
        ),
    )
    args = parser.parse_args(argv)
    if args.profile_run is not None:
        return _m11_main(args)
    config = json.loads(Path(args.config).read_text())
    report, samples = run_staged_evaluation(config)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.events_output:
        with Path(args.events_output).open("w") as handle:
            for stage in report["stage_order"]:
                for event in samples[stage]:
                    handle.write(json.dumps(event.as_dict(), sort_keys=True) + "\n")
    if _sector_wise_requested(config, args.sector_report):
        decomposition, total_chi2, dof = _build_sector_report(config, samples)
        sector_path = Path(args.output).with_name(SECTOR_REPORT_FILENAME)
        write_sector_report(str(sector_path), decomposition, total_chi2, dof)
        print(
            f"[sector-report] wrote {sector_path} "
            f"(n_sectors={len(decomposition['sectors'])}, "
            f"unassigned_bins={decomposition['unassigned_bins']}, "
            f"total_chi2={total_chi2:.6f}, dof={dof})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
