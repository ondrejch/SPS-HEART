"""Measure-pinning tests for the SourceMassSampler cell weights.

Task TASK-20260905-physics-review, milestone M1 (option A, Jacobian),
pinning the committed behavior of the nested tree at c927e3a:

- regge mode samples the documented ansatz, which is a density in the
  squared-mass measure dM1^2 dM2^2 (spec section 3.2.1, "Sampled
  measure"). The grid is uniform in (M1, M2), so the cell weight carries
  the Jacobian dM1^2 dM2^2 = 4 M1 M2 dM1 dM2. The sampled mass-space
  density is the squared-mass law
  (M1^2/s0)^(alpha_exc0-1) (M2^2/s0)^(alpha_exc0-1) times 4 M1 M2 times
  the edge suppression, zero where sqrt_s < M1 + M2.
- simple mode is a dM1 dM2 density by construction: the cell weight is
  the exponential mass fall-off exp(-(M1-m1min + M2-m2min)/mass_scale)
  times the edge suppression. The simple branch is byte-identical before
  and after the M1 commit, so this test pins the unchanged formula.

Independence: every density below is re-derived from the spec formula in
plain numpy inside this file. It is not shared with the sampler code, and
the grid geometry (uniform in M1 and M2 over [m_min, sqrt_s - m_min],
exact triangular boundary) is re-derived independently as well. The
sampler side of each comparison is its own normalized cell probabilities
(the cumulative distribution built by ``_prepare``, recovered with
``np.diff``); the documented density is evaluated here.

Grid and tolerances (documented per the card's acceptance criterion 2):
the primary comparisons run on the bundled 96x96 production grid, and the
independent evaluation integrates each cell with a 16x16 midpoint
subgrid, so the effective independent resolution is 1536x1536. The
flatness test runs on an explicit 128x128 grid. Documented tolerances:

- ``TOL_NORMALIZED_WEIGHTS``: normalized cell weights against the
  independent density at cell centers. float64 normalization noise;
  realized ~1e-16.
- ``TOL_CELL_INTEGRAL``: cell weight against the theta-cut integral of
  the independent density over the cell, interior cells with center
  edge >= 0.05. Midpoint-rule discretization of a density that varies
  by at most a few percent per cell there; realized ~1.5e-2 worst case.
- ``TOL_FLATNESS``: constancy of the cell weight divided by
  edge^edge_power at alpha_exc0 = 0.5 (the mass-space density is flat
  before edge suppression). Realized ~5e-10.
- ``TOL_CORNER_MASS``: combined independent-density mass of the
  triangle-corner cells whose weight falls below the float64 cumsum
  resolution and therefore carries exactly zero cell probability.
  Realized ~1e-33.
- Monte-Carlo marginal chi-square: upper bound ``CHI2_OVER_DF`` times df
  over bins with expected count >= 5. The bound exceeds the 0.999
  chi-square quantile for df >= 23 (realized chi2/df <= 1.11).

Hermetic: reads only the bundled ``config/staged_excitation.json`` and
``na49_mixing`` sources; no network, no spshadrons tables, deterministic
seeds.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from helpers.benchmarking import sps_core
from helpers.benchmarking import staged_mass_law_compare as mlc

from na49_mixing.continuum_channels import minimum_excited_mass
from na49_mixing.exchange_model import (
    ReggeExcitationParameters,
    SimpleExcitationParameters,
    SourceMassSampler,
)
from na49_mixing.quantum_numbers import QuantumNumbers
from na49_mixing.reaction import Reaction

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_CONFIG = REPO_ROOT / "config" / "staged_excitation.json"

# Documented grid tolerances (see the module docstring).
TOL_NORMALIZED_WEIGHTS = 1e-9
TOL_CELL_INTEGRAL = 0.05
TOL_FLATNESS = 1e-6
TOL_CORNER_MASS = 1e-30
EDGE_INTERIOR_CUTOFF = 0.05
INTEGRAL_SUBGRID = 16
FLATNESS_GRID_SIZE = 128

# Seeded Monte-Carlo marginal consistency.
MC_SEED = 20260905
MC_N = 10_000
CHI2_OVER_DF = 2.0
MIN_EXPECTED_COUNT = 5.0
MIN_CHI2_DOF = 10


def _load_bundled_config() -> dict:
    return json.loads(BUNDLED_CONFIG.read_text())


def _pp_kinematics(config: dict) -> tuple[float, float]:
    """sqrt_s and the p+p source-mass minimum, as the event builder uses them."""
    sqrt_s = Reaction(float(config["beam_momentum"]), 2212, 2212).sqrt_s()
    minimum = minimum_excited_mass(QuantumNumbers(1, 1, 0))
    return sqrt_s, minimum


def _grid(m_min: float, upper: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform mass grid: edges and cell centers, re-derived independently."""
    edges = np.linspace(m_min, upper, n + 1)
    return edges, 0.5 * (edges[:-1] + edges[1:])


def _sampler_cell_probabilities(sampler: SourceMassSampler) -> np.ndarray:
    """The sampler's normalized cell probabilities, (n_m1, n_m2).

    ``_prepare`` stores the cumulative sum of the normalized, row-major
    cell weights; differencing recovers the per-cell probability masses.
    """
    n = int(sampler.parameters.grid_size)
    flat = np.diff(np.concatenate(([0.0], sampler._cdf)))
    return flat.reshape(n, n)


def _documented_density(
    mode: str,
    m1: np.ndarray,
    m2: np.ndarray,
    params,
    m_min: float,
    sqrt_s: float,
) -> np.ndarray:
    """Documented mass-space density dP/dM1 dM2 (unnormalized), spec 3.2.1.

    Regge mode: the documented squared-mass law
    (M1^2/s0)^(alpha_exc0-1) (M2^2/s0)^(alpha_exc0-1) times the Jacobian
    4 M1 M2 and the edge suppression edge^edge_power. Simple mode: the
    exponential mass fall-off times the edge suppression. Zero where
    sqrt_s < M1 + M2.
    """
    m1, m2 = np.broadcast_arrays(m1, m2)
    edge = 1.0 - (m1 + m2) / sqrt_s
    valid = edge > 0.0
    w = np.zeros_like(m1)
    if mode == "regge":
        law = (m1[valid] ** 2 / params.s0) ** (params.alpha_exc0 - 1.0) * (
            m2[valid] ** 2 / params.s0
        ) ** (params.alpha_exc0 - 1.0)
        w[valid] = (
            4.0 * m1[valid] * m2[valid] * law * edge[valid] ** params.edge_power
        )
    elif mode == "simple":
        w[valid] = (
            np.exp(-(m1[valid] - m_min + m2[valid] - m_min) / params.mass_scale)
            * edge[valid] ** params.edge_power
        )
    else:
        raise ValueError(f"unknown source-mass mode {mode!r}")
    return w


def _theta_cut_cell_integrals(
    mode: str,
    params,
    m_min: float,
    sqrt_s: float,
    n: int,
    subgrid: int = INTEGRAL_SUBGRID,
) -> np.ndarray:
    """Integrate the documented density over each cell, theta cut applied.

    A (subgrid x subgrid) midpoint mesh per cell, independent of the
    sampler's grid arithmetic; the mass boundary is enforced pointwise by
    the density's own support.
    """
    e1, _ = _grid(m_min, sqrt_s - m_min, n)
    e2, _ = _grid(m_min, sqrt_s - m_min, n)
    t = np.linspace(0.0, 1.0, subgrid + 2)[1:-1]
    s1 = e1[:-1][:, None] + (e1[1:] - e1[:-1])[:, None] * t[None, :]
    s2 = e2[:-1][:, None] + (e2[1:] - e2[:-1])[:, None] * t[None, :]
    w = _documented_density(mode, s1[:, :, None, None], s2[None, None, :, :],
                            params, m_min, sqrt_s)
    dA = (e1[1] - e1[0]) * (e2[1] - e2[0])
    return w.sum(axis=(1, 3)) * dA / subgrid**2


def _assert_cell_weights_match_documented_density(
    mode: str,
    params,
    sqrt_s: float,
    m_min: float,
) -> None:
    """Shared comparison machinery for the regge and simple modes."""
    sampler = SourceMassSampler(
        sqrt_s, m_min, m_min, mode, params, np.random.default_rng(0)
    )
    p = _sampler_cell_probabilities(sampler)
    n = int(params.grid_size)
    e1, c1 = _grid(m_min, sqrt_s - m_min, n)
    e2, c2 = _grid(m_min, sqrt_s - m_min, n)
    m1, m2 = np.meshgrid(c1, c2, indexing="ij")
    edge = 1.0 - (m1 + m2) / sqrt_s
    valid = edge > 0.0

    # No cell weight outside the triangular support (the valid mass
    # window, evaluated at the cell centers).
    assert not np.any((p > 0.0) & ~valid)

    # Independent density at the cell centers, normalized the same way.
    w = _documented_density(mode, m1, m2, params, m_min, sqrt_s)
    wn = w / w.sum()
    max_abs = float(np.abs(p - wn).max())
    assert max_abs < TOL_NORMALIZED_WEIGHTS, (
        f"{mode}: normalized cell weights deviate from the documented "
        f"density by {max_abs:.3e} (tolerance {TOL_NORMALIZED_WEIGHTS:.1e})"
    )

    # Triangle-corner cells below the float64 cumsum resolution carry
    # exactly zero cell probability; their true mass must be negligible.
    corner = valid & (p == 0.0)
    corner_mass = float(wn[corner].sum())
    assert corner_mass < TOL_CORNER_MASS, (
        f"{mode}: corner cells with zero cell probability carry mass "
        f"{corner_mass:.3e} (tolerance {TOL_CORNER_MASS:.1e})"
    )

    # Grid tolerance: the cell weight (piecewise-constant approximation)
    # must approximate the theta-cut integral of the documented density
    # over the cell on interior cells away from the boundary layer.
    interior = valid & ~(e1[1:][:, None] + e2[1:][None, :] >= sqrt_s)
    near = interior & (edge >= EDGE_INTERIOR_CUTOFF)
    integrals = _theta_cut_cell_integrals(mode, params, m_min, sqrt_s, n)
    dA = (e1[1] - e1[0]) * (e2[1] - e2[0])
    relative = np.abs(w[near] * dA - integrals[near]) / integrals[near]
    max_rel = float(relative.max())
    assert max_rel < TOL_CELL_INTEGRAL, (
        f"{mode}: cell weight deviates from the cell integral of the "
        f"documented density by {max_rel:.3e} relative (tolerance "
        f"{TOL_CELL_INTEGRAL:.3f}) on {int(near.sum())} interior cells"
    )


def test_regge_cell_weights_match_documented_density_with_jacobian():
    """Regge cell weights implement the documented dM1^2 dM2^2 law.

    The independent density is the squared-mass law times the Jacobian
    4 M1 M2 times the edge suppression, evaluated on the bundled 96x96
    production grid (parameters from the bundled configuration).
    """
    config = _load_bundled_config()
    sqrt_s, m_min = _pp_kinematics(config)
    params = ReggeExcitationParameters(**config["regge"])
    _assert_cell_weights_match_documented_density("regge", params, sqrt_s, m_min)


def test_regge_default_exponent_gives_flat_mass_space_density():
    """At alpha_exc0 = 0.5 the mass-space density is flat before the edge.

    The Jacobian 4 M1 M2 cancels the (M1 M2)^-1 left by
    alpha_exc0 - 1 = -1/2, so on a 128x128 grid (documented fine grid)
    the cell weight must be proportional to edge^edge_power alone. The
    same ratio varies by tens of percent for any other exponent, so this
    pins the Jacobian by its consequence, not by re-deriving the full
    formula.
    """
    config = _load_bundled_config()
    sqrt_s, m_min = _pp_kinematics(config)
    base = config["regge"]
    params = ReggeExcitationParameters(
        alpha_exc0=0.5, s0=1.0, grid_size=FLATNESS_GRID_SIZE,
        **{k: v for k, v in base.items() if k not in {"alpha_exc0", "s0", "grid_size"}},
    )
    assert params.alpha_exc0 == 0.5 and params.s0 == 1.0
    sampler = SourceMassSampler(
        sqrt_s, m_min, m_min, "regge", params, np.random.default_rng(0)
    )
    p = _sampler_cell_probabilities(sampler)
    _, c1 = _grid(m_min, sqrt_s - m_min, params.grid_size)
    _, c2 = _grid(m_min, sqrt_s - m_min, params.grid_size)
    m1, m2 = np.meshgrid(c1, c2, indexing="ij")
    edge = 1.0 - (m1 + m2) / sqrt_s
    support = (edge > 0.0) & (p > 0.0)
    ratio = p[support] / edge[support] ** params.edge_power
    max_rel = float(np.abs(ratio / np.median(ratio) - 1.0).max())
    assert max_rel < TOL_FLATNESS, (
        f"cell weight / edge^edge_power varies by {max_rel:.3e} relative "
        f"(tolerance {TOL_FLATNESS:.1e}); the mass-space density is not "
        "flat at alpha_exc0 = 0.5"
    )


def test_simple_cell_weights_match_documented_density_unchanged():
    """Simple mode pins the documented dM1 dM2 density, unchanged by M1.

    The exponential mass fall-off with the config's mass scale times the
    edge suppression, on the uniform mass grid, with no Jacobian factor.
    The simple branch is byte-identical before and after the M1 commit.
    """
    config = _load_bundled_config()
    sqrt_s, m_min = _pp_kinematics(config)
    params = SimpleExcitationParameters(**config["simple"])
    _assert_cell_weights_match_documented_density("simple", params, sqrt_s, m_min)


def test_seeded_sampled_marginals_match_cell_weights():
    """Seeded Monte-Carlo draws follow the normalized cell-weight marginals.

    For each mode, MC_N seeded draws are binned onto the (independently
    re-derived) mass grid; the binned marginals of M1 and M2 must match
    the normalized cell-weight marginals within the pre-registered
    chi-square-like bound CHI2_OVER_DF * df over bins with expected count
    >= MIN_EXPECTED_COUNT. Every draw must also lie in the physical
    region m1 + m2 < sqrt_s with both masses at or above the source
    minimum.
    """
    config = _load_bundled_config()
    sqrt_s, m_min = _pp_kinematics(config)
    for mode, params in (
        ("regge", ReggeExcitationParameters(**config["regge"])),
        ("simple", SimpleExcitationParameters(**config["simple"])),
    ):
        sampler = SourceMassSampler(
            sqrt_s, m_min, m_min, mode, params, np.random.default_rng(MC_SEED)
        )
        p = _sampler_cell_probabilities(sampler)
        n = int(params.grid_size)
        e1, _ = _grid(m_min, sqrt_s - m_min, n)
        e2, _ = _grid(m_min, sqrt_s - m_min, n)
        samples = np.array([sampler.sample() for _ in range(MC_N)])
        assert np.all(samples[:, 0] + samples[:, 1] < sqrt_s)
        assert np.all(samples >= m_min)
        for axis, edges in ((0, e1), (1, e2)):
            index = np.clip(
                np.searchsorted(edges, samples[:, axis], side="right") - 1,
                0, n - 1,
            )
            counts = np.bincount(index, minlength=n).astype(float)
            expected = MC_N * p.sum(axis=1 - axis)
            usable = expected >= MIN_EXPECTED_COUNT
            chi2 = float(((counts[usable] - expected[usable]) ** 2
                          / expected[usable]).sum())
            dof = int(usable.sum()) - 1
            assert dof >= MIN_CHI2_DOF
            assert chi2 <= CHI2_OVER_DF * dof, (
                f"{mode}: M{axis + 1} marginal chi-square {chi2:.1f} "
                f"exceeds {CHI2_OVER_DF:.0f} * {dof} degrees of freedom"
            )


# ===========================================================================
# M10: mass-law comparison machinery (staged_mass_law_compare)
# ===========================================================================
#
# Pins the committed behavior of the vendored mirror
# ``helpers/benchmarking/staged_mass_law_compare.py`` (byte-identical to the
# monorepo copy; sha256 below), card TASK-20260905-physics-review, milestone
# M10 phase A.
#
# Provenance: the M10 pre-registration
# dev/00runs/mass_law_compare_2026-09-07/pre_registration.json and the merged
# artifact mass_law_compare_report.json are the source of the recorded
# constants pinned as literals below (split bin counts, law values, the
# comparison numbers, the config canonical hash, the seeds). The tests are
# hermetic: they read only the bundled config, generate synthetic
# NA49-format tables under a tmp path_root override (the
# test_staged_fit_binned.py pattern), and never load the production tables,
# the run records, the gateway, or the network.
#
# Documented tolerances (one rationale each):
# - ``TOL_M10_CLOSED_FORM``: closed-form metric agreement (the pooled
#   held-out WRRMSE against the direct sps_core call on the same arrays;
#   the band/combined-band arithmetic against the hand computation; the
#   production comparison re-derived from the pinned law values): absolute
#   1e-12. The two sides are algebraically identical expressions on the
#   same inputs; the difference is pure double rounding.
# - ``TOL_M10_CONSERVATION_RESIDUAL``: the per-event conservation-law
#   residual of a generated event batch. The production probe measured it
#   at ~1e-14; 1e-10 sits six orders of magnitude above the measured
#   machine-epsilon-scale value, so a real regression fails while platform
#   noise cannot.
# - Exact pins (no tolerance): split assignment, bin counts, seeds, the
#   verdict boundary, bit-identity reproducibility of the seeded
#   objective, the sigma factor sqrt(2000/50), the arm frozen-record
#   decay-hypothesis pin, and the stubbed run-arm bookkeeping. The math
#   is exact in double precision and the stubs are deterministic
#   quadratic objectives.
#
# Recorded readings (judgment calls documented by this section):
# - Empty held-out: sps_core.relative_rmse_weighted returns 0.0 for empty
#   input (its documented degenerate convention). The helper's
#   heldout_metrics inherits that convention and does not refuse an empty
#   held-out at the metric level; the production path is guarded upstream
#   by the _assert_bin_counts refusal, which pins the split bin counts.
# - Best-of-trace: run_arm reports at min(trace, key=chi2), the first
#   occurrence of the minimum, i.e. the best objective evaluation of the
#   whole trace, not the last one. The cap-hit record is pinned through the
#   resume budget-exhaustion path (remaining budget <= 0 -> the optimizer
#   options are dropped -> the fit is skipped -> success = False with the
#   recorded message M10_CAP_HIT_MESSAGE), which is distinct from the
#   converged success = True record.
#
# Test-seed block: M10_TEST_SEED_BASE = 20331001 (replicas +1..+4 for
# K = 4). Clear of the production M10 block 20320831..20320889, the M9
# production block 20310831..20310847 + 20312831, and the M9 test block
# 20330001..20330006; the test stream is never a registered run.

M10_HELPER_SHA256 = (
    "6477bfe585f4db46abc114eb2ad371bf669d088653271eefa955e9b91e06d921"
)
M10_TEST_SEED_BASE = 20331001
M10_TEST_K = 4
M10_TEST_EVENTS = 150
TOL_M10_CLOSED_FORM = 1e-12
TOL_M10_CONSERVATION_RESIDUAL = 1e-10

PI_MASS_GEV = 0.1395699
M10_SYNTH_XF = (0.10, 0.199, 0.200, 0.201, 0.30)
M10_SYNTH_PT = (0.30, 0.50, 0.70, 0.90, 1.10)

M10_CONFIG_CANONICAL_SHA256 = (
    "3a83d7c8817ef934656c5b45d27d7b9da53af7b51152fab29b4aafa497c7bfee"
)
M10_M8_RECORD_SHA256 = (
    "ecd931d58ce79b9aecdd273c703b4421fe0ca9119cdde90d2ad3c7dfb9d49b65"
)

# The M9 seed block of record (source: the M9 pre-registration seed
# protocol, pinned by helpers/benchmarking/tests/
# test_staged_heldout_correlations.py): the production arm seed, the 16
# shuffle seeds, and the bootstrap seed. The highest consumed M9 seed is
# the bootstrap seed.
M9_PRODUCTION_SEEDS = ((20310831,) + tuple(range(20310832, 20310848))
                       + (20312831,))
M9_HIGHEST_CONSUMED_SEED = 20312831

# The production split of record (pre-registration split_of_record).
M10_PRODUCTION_SPLIT_TOTAL = {"train": 676, "heldout": 341, "sum": 1017}
M10_PRODUCTION_SPLIT_PER_SPECIES = {
    "pi_minus": {"total": 229, "train": 158, "heldout": 71},
    "pi_plus": {"total": 230, "train": 158, "heldout": 72},
    "k_plus": {"total": 153, "train": 105, "heldout": 48},
    "k_minus": {"total": 148, "train": 105, "heldout": 43},
    "proton": {"total": 135, "train": 75, "heldout": 60},
    "antiproton": {"total": 122, "train": 75, "heldout": 47},
}

# The per-hypothesis pooled held-out WRRMSE of record (the merged
# artifact law_metrics.values).
M10_PRODUCTION_LAW_VALUES = {
    "regge": {"H1": 1.9942530538358547, "H2": 1.2604935086628049,
              "H3": 1.047704796409968},
    "simple": {"H1": 1.848541320852332, "H2": 0.8301533535849469,
               "H3": 0.8380219363648985},
}
M10_PRODUCTION_COMPARISON = {
    "verdict": "not_distinguished",
    "delta": -0.20968286004506942,
    "band_regge": 0.9465482574258868,
    "band_simple": 1.018387967267385,
    "combined_band": 1.3903480339508447,
}
M10_PRODUCTION_DIAGNOSTIC_BANDS = {
    "band_regge": 0.2127887122528369,
    "band_simple": 0.007868582779951616,
}

M10_VERDICT_STATEMENTS = {
    "prefer_regge": (
        "delta > combined band: the data prefer the Regge-informed mass law"
    ),
    "prefer_simple": (
        "delta < -combined band: the data prefer the flexible empirical "
        "mass law"
    ),
    "not_distinguished": (
        "|delta| <= combined band: the data do not distinguish the two "
        "mass laws within the bands"
    ),
}

M10_H1_CAVEAT_TEXT = (
    "the H1 arms enter only the cross-hypothesis spread, never the primary "
    "verdict at H3. The H1 replica-mean MC sigma is sqrt(2000/E1) times the "
    "M8-statistics value (sigma_MC scales as 1/sqrt(events x K)), and the H1 "
    "fitted values are 2-iteration descents from the starting point, not "
    "converged fits (the M7 H1-cap honesty convention). Both caveats are "
    "quoted with the spread."
)
M10_CAP_HIT_MESSAGE = (
    "resumed with exhausted evaluation budget; best of prior trace"
)


def _m10_load_config() -> dict:
    return json.loads(BUNDLED_CONFIG.read_text())


def _m10_write_table(path: Path, rows) -> None:
    """A synthetic NA49-format table (4 columns: xF pT value error).

    xF is written at 4 decimals so the 0.199/0.200/0.201 split-boundary
    centers survive the write/read round trip.
    """
    lines = ["# synthetic M10 table (4 columns: xF pT value error)"]
    for xf, pt, value, error in rows:
        lines.append(f"{xf:.4f} {pt:.2f} {value!r} {error!r}")
    path.write_text("\n".join(lines) + "\n")


def _m10_scratch_config(tmp_path: Path, tables: list[dict]) -> dict:
    config = _m10_load_config()
    config["fit"]["binned"]["path_root"] = str(tmp_path)
    config["fit"]["binned"]["tables"] = tables
    return config


def _m10_two_species_config(tmp_path: Path) -> dict:
    """Two synthetic species on the five split-boundary xF centers."""
    rows = [
        (xf, pt, 1.0e-2 + 1.0e-3 * i, 1.0e-3)
        for i, (xf, pt) in enumerate(zip(M10_SYNTH_XF, M10_SYNTH_PT))
    ]
    _m10_write_table(tmp_path / "synth_pi_minus.txt", rows)
    _m10_write_table(tmp_path / "synth_pi_plus.txt", rows)
    tables = [
        {"species": "pi_minus", "pdg": -211, "mass_gev": PI_MASS_GEV,
         "path": "synth_pi_minus.txt", "normalization": "f"},
        {"species": "pi_plus", "pdg": 211, "mass_gev": PI_MASS_GEV,
         "path": "synth_pi_plus.txt", "normalization": "f"},
    ]
    return _m10_scratch_config(tmp_path, tables)


def _m10_prereg_record() -> dict:
    """The pre-registration record rebuilt from the harness constants.

    This is what the cross-check is asked to verify: every field mirrors a
    module constant, and the provenance fields carry the recorded hashes.
    """
    return {
        "form_of_record": {"choice": mlc.FORM_CHOICE},
        "split_of_record": {
            "train_window": dict(mlc.TRAIN_WINDOW),
            "heldout_window": dict(mlc.HELDOUT_WINDOW),
            "bin_counts_measured": {
                "per_species": {k: dict(v) for k, v in mlc.BIN_COUNTS.items()},
                "total": dict(mlc.BIN_COUNTS_TOTAL),
            },
        },
        "identical_protocol": {
            "m8_fitted_parameters": {"values": dict(mlc.M8_PARAMETER_SET)},
            "frozen_continuum_m8_values": {
                "values": {k: v for k, v in mlc.M8_PARAMETER_SET.items()
                           if k.startswith("continuum.")}
            },
            "free_parameters": {k: list(v)
                                for k, v in mlc.FREE_PARAMETERS.items()},
            "bounds": {k: {kk: list(vv) for kk, vv in v.items()}
                       for k, v in mlc.BOUNDS.items()},
            "starting_points": {k: dict(v)
                                for k, v in mlc.STARTING_POINTS.items()},
            "events_per_replica_main": mlc.EVENTS_PER_REPLICA_MAIN,
            "k_replicas": mlc.K_REPLICAS,
            "optimizer": {"xtol": mlc.XTOL, "ftol": mlc.FTOL,
                          "cap_max_iterations_main": mlc.CAP_MAX_ITERATIONS_MAIN},
        },
        "seed_protocol": {
            "m10_seed_base": mlc.M10_SEED_BASE,
            "stage_label": mlc.STAGE_LABEL,
            "arm_seeds": dict(mlc.ARM_SEEDS),
            "replica_seed_blocks": {k: list(v)
                                    for k, v in mlc.REPLICA_SEED_BLOCKS.items()},
        },
        "arm_grid": {"ordering": [dict(a) for a in mlc.ARM_ORDERING]},
        "h1_declared_reduction": {
            "chosen_E1": 50,
            "menu_events_per_replica": list(mlc.H1_MENU_EVENTS_PER_REPLICA),
            "max_iterations": mlc.H1_MAX_ITERATIONS,
            "max_evaluations": mlc.H1_MAX_EVALUATIONS,
            "max_projected_arm_wall_s": mlc.H1_MAX_PROJECTED_ARM_WALL_S,
        },
        "verdict_rule": dict(mlc.VERDICT_RULE),
        "provenance": {
            "config_canonical_sha256": M10_CONFIG_CANONICAL_SHA256,
            "m8_record_sha256": M10_M8_RECORD_SHA256,
        },
    }


def _m10_write_trace(trace_dir: Path, arm_name: str, entries: list[dict]) -> Path:
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"trace_{arm_name}.jsonl"
    path.write_text(
        "\n".join(json.dumps(e, sort_keys=True) for e in entries) + "\n"
    )
    return path


def _m10_stub_objective(target: list[float]):
    """A deterministic quadratic objective standing in for evaluate_point."""

    def stub(x, arm, tables, train_masks, n_events, replica_seeds, beam):
        xv = [float(v) for v in np.asarray(x, dtype=float)]
        chi2 = float(sum((a - b) ** 2 for a, b in zip(xv, target)) + 1.0)
        return {
            "chi2": chi2,
            "per_species_train_chi2": {t.species: chi2 / 2.0 for t in tables},
            "max_conservation_residual": 0.0,
            "_m_mean": {t.species: np.full(t.data.size, 0.5) for t in tables},
            "_sigma_mc": {t.species: np.zeros(t.data.size) for t in tables},
            "_sigma_replica": {t.species: np.zeros(t.data.size)
                               for t in tables},
        }

    return stub


def test_m10_helper_layout_and_byte_identity():
    """The vendored mirror resolves the nested tree and pins byte identity.

    The monorepo copy and the vendored mirror must be the same file (the
    layout-sync rule): same sha256, same bytes, and the mirror resolves
    its own tree (MIXING2006_DIR = the nested root, RUNS_ROOT = 00runs).
    """
    expected = REPO_ROOT / "helpers" / "benchmarking" / "staged_mass_law_compare.py"
    assert Path(mlc.__file__).resolve() == expected
    assert mlc.MIXING2006_DIR == REPO_ROOT
    assert mlc.RUNS_ROOT == REPO_ROOT / "00runs"
    # REPO_ROOT here is the nested root (dev/mixing2006); the monorepo
    # copy sits two levels up.
    monorepo_copy = (REPO_ROOT.parent.parent / "helpers" / "benchmarking"
                     / "staged_mass_law_compare.py")
    assert mlc._sha256_file(expected) == M10_HELPER_SHA256
    assert mlc._sha256_file(monorepo_copy) == M10_HELPER_SHA256
    assert expected.read_bytes() == monorepo_copy.read_bytes()


def test_m10_arm_frozen_record_carries_arm_decay_hypothesis():
    """The frozen record reports the kernel each arm actually runs.

    Regression pin for the round-1 fix (REV-0092af6-01): the arm's decay
    hypothesis is applied to the frozen continuum before the frozen_record
    snapshot, so for every (law, hypothesis) arm the frozen record's
    continuum carries the arm's hypothesis; for the H3 arms it is "H3",
    which is also the ContinuumParameters dataclass default. The frozen
    record is therefore self-sufficient per arm. This needs a full
    bundled-config load (a hermetic JSON read, no event generation):
    arm_parameters builds the base parameter set from the config.
    """
    config = _m10_load_config()
    for law in ("regge", "simple"):
        for hypothesis in ("H1", "H2", "H3"):
            payload = mlc.arm_parameters(law, hypothesis, config)
            assert payload["hypothesis"] == hypothesis
            assert payload["continuum"].decay_hypothesis == hypothesis
            assert payload["frozen_record"]["continuum"]["decay_hypothesis"] \
                == hypothesis


def test_m10_protocol_constants_pinned():
    """The pre-registered protocol constants, as literals of record."""
    assert mlc.CARD == "TASK-20260905-physics-review"
    assert mlc.MILESTONE == "M10"
    assert mlc.PHASE == "A"
    assert mlc.FORM_CHOICE == "form2_predictive_validation"
    assert mlc.SPLIT_XF == 0.2
    assert mlc.TRAIN_WINDOW == {"xf_min": 0.0, "xf_max": 0.2,
                                "pt_min": 0.05, "pt_max": 1.5}
    assert mlc.HELDOUT_WINDOW == {"xf_min": 0.2, "xf_max": 0.4,
                                  "pt_min": 0.05, "pt_max": 1.5}
    assert mlc.BIN_COUNTS_TOTAL == M10_PRODUCTION_SPLIT_TOTAL
    assert mlc.BIN_COUNTS == M10_PRODUCTION_SPLIT_PER_SPECIES
    assert mlc.SPECIES_ORDER == ("pi_minus", "pi_plus", "k_plus", "k_minus",
                                 "proton", "antiproton")
    assert mlc.M8_PARAMETER_SET == {
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
    assert mlc.FREE_PARAMETERS == {
        "regge": ["regge.alpha_exc0", "regge.effective_transfer_slope",
                  "regge.edge_power"],
        "simple": ["simple.mass_scale", "simple.edge_power",
                   "simple.transfer_slope"],
    }
    assert mlc.BOUNDS == {
        "regge": {"regge.alpha_exc0": [0.0, 1.2],
                  "regge.effective_transfer_slope": [5.07, 10.2],
                  "regge.edge_power": [0.25, 6.0]},
        "simple": {"simple.mass_scale": [0.1, 5.0],
                   "simple.edge_power": [0.25, 6.0],
                   "simple.transfer_slope": [5.07, 10.2]},
    }
    assert mlc.STARTING_POINTS == {
        "regge": {"regge.alpha_exc0": 0.5,
                  "regge.effective_transfer_slope": 7.3489827740435505,
                  "regge.edge_power": 2.0},
        "simple": {"simple.mass_scale": 0.65,
                   "simple.edge_power": 1.5,
                   "simple.transfer_slope": 5.5},
    }
    assert mlc.EVENTS_PER_REPLICA_MAIN == 2000
    assert mlc.K_REPLICAS == 8
    assert mlc.XTOL == 1e-3 and mlc.FTOL == 1e-3
    assert mlc.CAP_MAX_ITERATIONS_MAIN == 10
    assert mlc.N_PARAMS_FIT == 3
    assert mlc.H1_MENU_EVENTS_PER_REPLICA == [200, 100, 50]
    assert mlc.H1_MAX_ITERATIONS == 2
    assert mlc.H1_MAX_EVALUATIONS == 150
    assert mlc.H1_MAX_PROJECTED_ARM_WALL_S == 18000.0
    assert mlc.M10_SEED_BASE == 20320831
    assert mlc.ARM_SEED_STRIDE == 10
    assert mlc.REPLICA_SEED_OFFSET == 1
    assert mlc.ARM_ORDERING == [
        {"a": 0, "law": "regge", "hypothesis": "H1"},
        {"a": 1, "law": "regge", "hypothesis": "H2"},
        {"a": 2, "law": "regge", "hypothesis": "H3"},
        {"a": 3, "law": "simple", "hypothesis": "H1"},
        {"a": 4, "law": "simple", "hypothesis": "H2"},
        {"a": 5, "law": "simple", "hypothesis": "H3"},
    ]
    assert mlc.ARM_SEEDS == {
        "regge_H1": 20320831,
        "regge_H2": 20320841,
        "regge_H3": 20320851,
        "simple_H1": 20320861,
        "simple_H2": 20320871,
        "simple_H3": 20320881,
    }
    assert mlc.REPLICA_SEED_BLOCKS == {
        "regge_H1": [20320832, 20320839],
        "regge_H2": [20320842, 20320849],
        "regge_H3": [20320852, 20320859],
        "simple_H1": [20320862, 20320869],
        "simple_H2": [20320872, 20320879],
        "simple_H3": [20320882, 20320889],
    }
    assert mlc.STAGE_LABEL == "mass_law_compare_replica"
    assert mlc.CONFIG_SEED == 20260831
    assert mlc.BEAM_MOMENTUM_GEV_C == 158.0
    assert mlc.SQRT_S_GEV == 17.270205201372406
    assert mlc.M8_RECORD_SHA256 == M10_M8_RECORD_SHA256
    # The verdict rule is written before fitting and pinned as a literal.
    assert mlc.VERDICT_RULE["written_before_fitting"] is True
    for key, text in M10_VERDICT_STATEMENTS.items():
        assert mlc.VERDICT_RULE[key] == text
    assert mlc.H1_CAVEAT_TEXT == M10_H1_CAVEAT_TEXT


def test_m10_split_mask_assigns_train_heldout_by_xf_center(tmp_path):
    """The split is the strict xF-center < 0.2 rule, per table.

    On the synthetic two-species set (xF centers 0.10, 0.199, 0.200,
    0.201, 0.30) each species carries 2 train and 3 held-out bins, and the
    per-row mask matches the pre-registered rule row by row.
    """
    config = _m10_two_species_config(tmp_path)
    tables = mlc.build_tables(config)
    assert [t.species for t in tables] == ["pi_minus", "pi_plus"]
    train_masks, heldout_masks = mlc.split_masks(tables)
    for table, train, heldout in zip(tables, train_masks, heldout_masks):
        assert table.data.size == 5
        assert table.xf_centers.tolist() == [0.10, 0.199, 0.2, 0.201, 0.3]
        assert train.tolist() == [True, True, False, False, False]
        assert heldout.tolist() == [False, False, True, True, True]
        assert int(train.sum()) == 2 and int(heldout.sum()) == 3
        for i in range(table.data.size):
            assert bool(train[i]) == (table.xf_centers[i] < 0.2)
    assert sum(int(m.sum()) for m in train_masks) == 4
    assert sum(int(m.sum()) for m in heldout_masks) == 6


def test_m10_split_boundary_bin_at_0_2_is_heldout(tmp_path):
    """A bin centered exactly at xF = 0.2 is held out (the train side is strict).

    On the xF centers 0.199, 0.200, 0.201: 0.199 is train, and the bin
    centered exactly at 0.2 (and everything at or above it) is held out.
    """
    rows = [(xf, 0.50, 1.0e-2, 1.0e-3) for xf in (0.199, 0.200, 0.201)]
    _m10_write_table(tmp_path / "synth_boundary.txt", rows)
    config = _m10_scratch_config(tmp_path, [
        {"species": "pi_minus", "pdg": -211, "mass_gev": PI_MASS_GEV,
         "path": "synth_boundary.txt", "normalization": "f"},
    ])
    tables = mlc.build_tables(config)
    train_masks, heldout_masks = mlc.split_masks(tables)
    assert train_masks[0].tolist() == [True, False, False]
    assert heldout_masks[0].tolist() == [False, True, True]


def test_m10_bin_count_pin_refuses_non_production_split(tmp_path):
    """The pre-registered bin-count pin refuses any non-production split."""
    config = _m10_two_species_config(tmp_path)
    tables = mlc.build_tables(config)
    train_masks, heldout_masks = mlc.split_masks(tables)
    with pytest.raises(SystemExit,
                       match="do not match the pre-registered pin"):
        mlc._assert_bin_counts(tables, train_masks, heldout_masks)


def test_m10_per_law_scoring_reproduces_bit_identically(tmp_path):
    """Both laws' seeded objective reproduces bit-identically at fixed seed.

    Reduced test statistics (K = 4, E = 150 against the production K = 8 x
    2000) keep the suite fast; the bit-identity property is independent of
    the statistics. The objective also depends on the free parameters (an
    in-bounds shifted point scores differently), and the held-out scoring
    at the same point reproduces.
    """
    config = _m10_two_species_config(tmp_path)
    tables = mlc.build_tables(config)
    train_masks, heldout_masks = mlc.split_masks(tables)
    replica_seeds = [M10_TEST_SEED_BASE + 1 + k for k in range(M10_TEST_K)]
    for law, shift in (("regge", (0.3, 1.0, 1.0)),
                       ("simple", (2.0, 1.0, 1.5))):
        arm = mlc.set_decay_hypothesis(
            mlc.arm_parameters(law, "H3", config), "H3")
        x = list(arm["start"])
        res_a = mlc.evaluate_point(x, arm, tables, train_masks,
                                   M10_TEST_EVENTS, replica_seeds, 158.0)
        res_b = mlc.evaluate_point(x, arm, tables, train_masks,
                                   M10_TEST_EVENTS, replica_seeds, 158.0)
        assert res_a["chi2"] == res_b["chi2"]
        assert res_a["per_species_train_chi2"] == res_b["per_species_train_chi2"]
        assert res_a["max_conservation_residual"] < TOL_M10_CONSERVATION_RESIDUAL
        # An in-bounds shifted point changes the objective (the fit's job).
        x_alt = [x[i] + shift[i] for i in range(3)]
        res_alt = mlc.evaluate_point(x_alt, arm, tables, train_masks,
                                     M10_TEST_EVENTS, replica_seeds, 158.0)
        assert res_alt["chi2"] != res_a["chi2"]
        held_a = mlc.heldout_metrics(res_a["_m_mean"], tables, heldout_masks)
        held_b = mlc.heldout_metrics(res_b["_m_mean"], tables, heldout_masks)
        assert held_a["pooled"]["wrrmse"] == held_b["pooled"]["wrrmse"]
        assert held_a["pooled"]["n_bins"] == 6


def test_m10_heldout_wrrmse_matches_direct_sps_core(tmp_path):
    """The pooled held-out WRRMSE equals the direct sps_core closed form."""
    config = _m10_two_species_config(tmp_path)
    tables = mlc.build_tables(config)
    train_masks, heldout_masks = mlc.split_masks(tables)
    arm = mlc.set_decay_hypothesis(
        mlc.arm_parameters("regge", "H3", config), "H3")
    res = mlc.evaluate_point(
        list(arm["start"]), arm, tables, train_masks, M10_TEST_EVENTS,
        [M10_TEST_SEED_BASE + 1 + k for k in range(M10_TEST_K)], 158.0)
    held = mlc.heldout_metrics(res["_m_mean"], tables, heldout_masks)
    # The same arrays scored by sps_core directly (the metric's definition).
    pred_c, data_c, sig_c = [], [], []
    for table, mask in zip(tables, heldout_masks):
        pred_c.append(np.asarray(res["_m_mean"][table.species],
                                 dtype=float)[mask])
        data_c.append(table.data[mask])
        sig_c.append(table.sigma[mask])
    direct_pooled = sps_core.relative_rmse_weighted(
        np.concatenate(pred_c), np.concatenate(data_c),
        np.concatenate(sig_c))
    assert abs(held["pooled"]["wrrmse"] - direct_pooled) < TOL_M10_CLOSED_FORM
    assert held["pooled"]["n_bins"] == 6
    for table, mask in zip(tables, heldout_masks):
        direct = sps_core.relative_rmse_weighted(
            np.asarray(res["_m_mean"][table.species], dtype=float)[mask],
            table.data[mask], table.sigma[mask])
        assert abs(held["per_species_wrrmse"][table.species] - direct) \
            < TOL_M10_CLOSED_FORM


def test_m10_verdict_rule_branches_and_boundaries():
    """The verdict branches exactly at and just beyond the combined band.

    At the boundary one law is flat (band 0) and the other spans the band
    with H3 interior to its range, so a 1e-9 perturbation of H3 moves the
    delta without moving the band.
    """
    def verdict_for(regge, simple):
        return mlc.compare_laws({"regge": regge, "simple": simple})

    at_plus = verdict_for(
        {"H1": 1.0, "H2": 1.0, "H3": 1.0},
        {"H1": 1.5, "H2": 3.5, "H3": 3.0})
    assert at_plus["delta"] == 2.0 and at_plus["combined_band"] == 2.0
    assert at_plus["verdict"] == "not_distinguished"
    assert at_plus["verdict_statement"] == M10_VERDICT_STATEMENTS[
        "not_distinguished"]
    above = verdict_for(
        {"H1": 1.0, "H2": 1.0, "H3": 1.0},
        {"H1": 1.5, "H2": 3.5, "H3": 3.0 + 1e-9})
    assert abs(above["delta"] - (2.0 + 1e-9)) < TOL_M10_CLOSED_FORM
    assert above["combined_band"] == 2.0
    assert above["verdict"] == "prefer_regge"
    assert above["verdict_statement"] == M10_VERDICT_STATEMENTS[
        "prefer_regge"]
    at_minus = verdict_for(
        {"H1": 1.5, "H2": 3.5, "H3": 3.0},
        {"H1": 1.0, "H2": 1.0, "H3": 1.0})
    assert at_minus["delta"] == -2.0 and at_minus["combined_band"] == 2.0
    assert at_minus["verdict"] == "not_distinguished"
    below = verdict_for(
        {"H1": 1.5, "H2": 3.5, "H3": 3.0 + 1e-9},
        {"H1": 1.0, "H2": 1.0, "H3": 1.0})
    assert below["verdict"] == "prefer_simple"
    assert below["verdict_statement"] == M10_VERDICT_STATEMENTS[
        "prefer_simple"]


def test_m10_band_arithmetic_hand_computed():
    """The band/combined-band arithmetic re-derived by hand in the test."""
    v = mlc.compare_laws({
        "regge": {"H1": 0.5, "H2": 2.5, "H3": 1.5},
        "simple": {"H1": 1.25, "H2": 0.75, "H3": 2.75},
    })
    assert abs(v["band_regge"] - (2.5 - 0.5)) < TOL_M10_CLOSED_FORM
    assert abs(v["band_simple"] - (2.75 - 0.75)) < TOL_M10_CLOSED_FORM
    assert abs(v["combined_band"] - math.sqrt(2.0 ** 2 + 2.0 ** 2)) \
        < TOL_M10_CLOSED_FORM
    # delta = M_simple(H3) - M_regge(H3) = 2.75 - 1.5.
    assert abs(v["delta"] - (2.75 - 1.5)) < TOL_M10_CLOSED_FORM
    assert v["verdict"] == "not_distinguished"
    # The identical-statistics diagnostic uses the H2/H3 range only.
    assert abs(v["diagnostic_h2h3_only"]["band_regge"] - (2.5 - 1.5)) \
        < TOL_M10_CLOSED_FORM
    assert abs(v["diagnostic_h2h3_only"]["band_simple"] - (2.75 - 0.75)) \
        < TOL_M10_CLOSED_FORM
    assert v["h1_caveat"] == M10_H1_CAVEAT_TEXT
    assert v["metric"] == mlc.VERDICT_RULE["primary_metric"]


def test_m10_production_comparison_rederived_from_law_values():
    """The recorded production comparison re-derived from the law values."""
    lv = M10_PRODUCTION_LAW_VALUES
    v = mlc.compare_laws(lv)
    rec = M10_PRODUCTION_COMPARISON
    assert v["verdict"] == rec["verdict"]
    assert abs(v["delta"] - rec["delta"]) < TOL_M10_CLOSED_FORM
    assert abs(v["band_regge"] - rec["band_regge"]) < TOL_M10_CLOSED_FORM
    assert abs(v["band_simple"] - rec["band_simple"]) < TOL_M10_CLOSED_FORM
    assert abs(v["combined_band"] - rec["combined_band"]) \
        < TOL_M10_CLOSED_FORM
    assert v["m_regge_h3"] == lv["regge"]["H3"]
    assert v["m_simple_h3"] == lv["simple"]["H3"]
    assert v["verdict_statement"] == M10_VERDICT_STATEMENTS[
        "not_distinguished"]
    # Hand re-derivation from the pinned law values (no helper call).
    assert abs(lv["simple"]["H3"] - lv["regge"]["H3"] - rec["delta"]) \
        < TOL_M10_CLOSED_FORM
    assert abs(max(lv["regge"].values()) - min(lv["regge"].values())
               - rec["band_regge"]) < TOL_M10_CLOSED_FORM
    assert abs(max(lv["simple"].values()) - min(lv["simple"].values())
               - rec["band_simple"]) < TOL_M10_CLOSED_FORM
    hand_combined = math.sqrt(rec["band_regge"] ** 2 + rec["band_simple"] ** 2)
    assert abs(hand_combined - rec["combined_band"]) < TOL_M10_CLOSED_FORM
    assert abs(v["diagnostic_h2h3_only"]["band_regge"]
               - M10_PRODUCTION_DIAGNOSTIC_BANDS["band_regge"]) \
        < TOL_M10_CLOSED_FORM
    assert abs(v["diagnostic_h2h3_only"]["band_simple"]
               - M10_PRODUCTION_DIAGNOSTIC_BANDS["band_simple"]) \
        < TOL_M10_CLOSED_FORM


def test_m10_cross_check_preregistration_passes_on_recorded_values():
    """The cross-check passes on the recorded pre-registration values."""
    config = _m10_load_config()
    assert mlc.canonical_configuration_hash(config) == M10_CONFIG_CANONICAL_SHA256
    mlc.cross_check_preregistration(_m10_prereg_record(), config,
                                    m8_record=None)


M10_TAMPER_CASES = (
    ("form_of_record",
     lambda p: p["form_of_record"].__setitem__(
         "choice", "form1_penalized_likelihood")),
    ("split boundary",
     lambda p: p["split_of_record"]["train_window"].__setitem__(
         "xf_max", 0.3)),
    ("seed base",
     lambda p: p["seed_protocol"].__setitem__("m10_seed_base", 20320832)),
    ("main cap",
     lambda p: p["identical_protocol"]["optimizer"].__setitem__(
         "cap_max_iterations_main", 8)),
    ("h1 menu",
     lambda p: p["h1_declared_reduction"].__setitem__(
         "menu_events_per_replica", [200, 100, 40])),
    ("k_replicas",
     lambda p: p["identical_protocol"].__setitem__("k_replicas", 7)),
    ("arm seed",
     lambda p: p["seed_protocol"]["arm_seeds"].__setitem__("regge_H3",
                                                          20320852)),
)


@pytest.mark.parametrize("label,mutate", M10_TAMPER_CASES,
                         ids=[c[0] for c in M10_TAMPER_CASES])
def test_m10_cross_check_preregistration_refuses_tamperings(label, mutate):
    """A tampered pre-registration is refused (the negative controls)."""
    config = _m10_load_config()
    prereg = _m10_prereg_record()
    mutate(prereg)
    with pytest.raises(SystemExit, match="pre-registration mismatch"):
        mlc.cross_check_preregistration(prereg, config, m8_record=None)


def test_m10_h1_sigma_factor_and_caveat():
    """The H1 declared-reduction sigma factor at the chosen E1 = 50."""
    assert 50 in mlc.H1_MENU_EVENTS_PER_REPLICA
    # The replica-mean MC sigma scales as 1/sqrt(events x K): at E1 = 50
    # against the M8 statistics (2000 events, the same K) the factor is
    # sqrt(2000 / 50), exact in double precision.
    assert math.sqrt(2000 / 50) == 6.324555320336759
    assert mlc.H1_CAVEAT_TEXT == M10_H1_CAVEAT_TEXT
    assert "sqrt(2000/E1)" in mlc.H1_CAVEAT_TEXT


@pytest.mark.parametrize(
    "mutate",
    (
        lambda p: p["h1_declared_reduction"].__setitem__("chosen_E1", None),
        lambda p: p["h1_declared_reduction"].__setitem__("chosen_E1", 250),
    ),
    ids=("missing", "off-menu"),
)
def test_m10_h1_arm_refuses_undecided_reduction(tmp_path, mutate):
    """An H1 arm refuses to run before the declared reduction is decided."""
    config = _m10_two_species_config(tmp_path)
    prereg = _m10_prereg_record()
    mutate(prereg)
    with pytest.raises(SystemExit,
                       match="the H1 declared reduction must be decided"):
        mlc.run_arm("regge", "H1", tmp_path / "arm_out", config, prereg)


def test_m10_arm_seed_derivation_and_m9_non_collision():
    """The seed grid derivation, and non-collision with the M9 block."""
    base = mlc.M10_SEED_BASE
    for a, item in enumerate(mlc.ARM_ORDERING):
        name = f"{item['law']}_{item['hypothesis']}"
        assert item["a"] == a
        assert mlc.ARM_SEEDS[name] == base + mlc.ARM_SEED_STRIDE * a
        seed = mlc.ARM_SEEDS[name]
        # The replica seed block records the first and last of the
        # K = 8 replica seeds seed + 1 .. seed + 8.
        assert mlc.REPLICA_SEED_BLOCKS[name] == [seed + 1, seed + 8]
        replicas = [seed + mlc.REPLICA_SEED_OFFSET + k
                    for k in range(mlc.K_REPLICAS)]
        assert len(set(replicas)) == mlc.K_REPLICAS
    # The per-arm replica streams are pairwise disjoint (stride 10 > 9).
    all_replicas = []
    for item in mlc.ARM_ORDERING:
        name = f"{item['law']}_{item['hypothesis']}"
        first = mlc.REPLICA_SEED_BLOCKS[name][0]
        all_replicas.extend(first + k for k in range(mlc.K_REPLICAS))
    assert len(set(all_replicas)) == 6 * mlc.K_REPLICAS
    # Non-collision with the M9 seed block of record: every M10 seed sits
    # strictly above the highest consumed M9 seed (the bootstrap seed).
    m10_seeds = set(mlc.ARM_SEEDS.values()) | set(all_replicas)
    assert M9_HIGHEST_CONSUMED_SEED < min(m10_seeds)
    assert all(seed < min(m10_seeds) for seed in M9_PRODUCTION_SEEDS)


def test_m10_best_of_trace_reports_at_best_evaluation(tmp_path, monkeypatch):
    """The arm reports at the best evaluation of the whole trace.

    A pre-written 3-entry scratch trace whose minimum is the middle entry
    (not the last) is resumed; the stubbed objective is a deterministic
    quadratic whose minimum is exactly the pre-written best entry. The arm
    must report that entry (best-of-trace convention), with the converged
    success record, the determinism check consistent, and the final
    re-evaluation reproducing the best chi2.
    """
    config = _m10_two_species_config(tmp_path)
    prereg = _m10_prereg_record()
    names = list(mlc.FREE_PARAMETERS["regge"])
    target = {"regge.alpha_exc0": 1.0,
              "regge.effective_transfer_slope": 8.5,
              "regge.edge_power": 2.4}
    prior = (
        {"regge.alpha_exc0": 0.5,
         "regge.effective_transfer_slope": 7.3489827740435505,
         "regge.edge_power": 2.0},
        target,
        {"regge.alpha_exc0": 0.7,
         "regge.effective_transfer_slope": 8.0,
         "regge.edge_power": 2.2},
    )
    entries = [
        {"parameters": {n: p[n] for n in names},
         "chi2": float(sum((p[n] - target[n]) ** 2 for n in names) + 1.0),
         "per_species": {"pi_minus": 1.0, "pi_plus": 1.0},
         "wall_s": 0.0}
        for p in prior
    ]
    out_dir = tmp_path / "run_arm"
    trace_path = _m10_write_trace(out_dir / "scratch_traces", "regge_H2",
                                  entries)
    monkeypatch.setattr(mlc, "evaluate_point",
                        _m10_stub_objective([target[n] for n in names]))
    monkeypatch.setattr(mlc, "_assert_bin_counts", lambda *a, **k: None)
    payload = mlc.run_arm("regge", "H2", out_dir, config, prereg, resume=True)
    fit = payload["fit"]
    assert fit["success"] is True
    assert fit["message"] == "Optimization terminated successfully."
    assert fit["prior_evaluations_from_restarts"] == 3
    assert fit["restarts"] == 1
    assert payload["fitted_parameters"] == target
    assert fit["train_chi2_combined_best"] == 1.0
    assert payload["final_evaluation"]["train_chi2_combined"] == 1.0
    assert fit["determinism_consistent"] is True
    final_trace = [json.loads(line) for line in trace_path.read_text().splitlines()
                   if line.strip()]
    assert len(final_trace) == fit["n_objective_evaluations"] > 3
    # The reported best is the first occurrence of the trace minimum, which
    # is the pre-written entry, not the last evaluation.
    best_idx = min(range(len(final_trace)),
                   key=lambda i: final_trace[i]["chi2"])
    assert best_idx < len(final_trace) - 1
    assert final_trace[best_idx]["parameters"] == target


def test_m10_cap_hit_arm_recorded_distinct_from_converged(tmp_path, monkeypatch):
    """A budget-exhausted H1 arm records the cap-hit record.

    A pre-written 150-entry scratch trace fills the pre-registered H1
    budget (max_evaluations = 150); on resume the remaining budget is
    zero, the optimizer is skipped, and the arm records success = False
    with the cap-hit message, the best-of-trace fitted point, and the
    declared reduction with its caveat. This is distinct from the
    converged success = True record pinned above.
    """
    config = _m10_two_species_config(tmp_path)
    prereg = _m10_prereg_record()
    names = list(mlc.FREE_PARAMETERS["simple"])
    target = {"simple.mass_scale": 1.0,
              "simple.edge_power": 2.4,
              "simple.transfer_slope": 8.5}

    def chi2_at(p):
        return float(sum((p[n] - target[n]) ** 2 for n in names) + 1.0)

    entries = []
    for k in range(mlc.H1_MAX_EVALUATIONS):
        if k == 7:
            p = dict(target)
        else:
            p = {"simple.mass_scale": 0.5 + 0.001 * k,
                 "simple.edge_power": 1.5,
                 "simple.transfer_slope": 7.0 + 0.001 * k}
        entries.append({"parameters": {n: p[n] for n in names},
                        "chi2": chi2_at(p),
                        "per_species": {"pi_minus": 1.0, "pi_plus": 1.0},
                        "wall_s": 0.0})
    out_dir = tmp_path / "run_arm_h1"
    _m10_write_trace(out_dir / "scratch_traces", "simple_H1", entries)
    monkeypatch.setattr(mlc, "evaluate_point",
                        _m10_stub_objective([target[n] for n in names]))
    monkeypatch.setattr(mlc, "_assert_bin_counts", lambda *a, **k: None)
    payload = mlc.run_arm("simple", "H1", out_dir, config, prereg, resume=True)
    fit = payload["fit"]
    assert fit["success"] is False
    assert fit["message"] == M10_CAP_HIT_MESSAGE
    assert fit["n_objective_evaluations"] == mlc.H1_MAX_EVALUATIONS
    assert fit["prior_evaluations_from_restarts"] == mlc.H1_MAX_EVALUATIONS
    assert payload["fitted_parameters"] == target
    assert fit["train_chi2_combined_best"] == 1.0
    assert payload["final_evaluation"]["train_chi2_combined"] == 1.0
    assert fit["determinism_consistent"] is True
    # The declared reduction is recorded with the caveat of record.
    assert payload["statistics"]["declared_reduction"] == {
        "events_per_replica": 50,
        "max_iterations": mlc.H1_MAX_ITERATIONS,
        "max_evaluations": mlc.H1_MAX_EVALUATIONS,
        "caveat": M10_H1_CAVEAT_TEXT,
    }
    assert payload["statistics"]["events_per_replica"] == 50
    assert payload["protocol"]["max_iterations"] == mlc.H1_MAX_ITERATIONS
    assert payload["protocol"]["max_evaluations"] == mlc.H1_MAX_EVALUATIONS
    # The simple_H1 seed block of the recorded run.
    assert payload["seed_protocol"]["arm_seed"] == mlc.ARM_SEEDS["simple_H1"]
    assert payload["seed_protocol"]["replica_seeds"] == [
        mlc.ARM_SEEDS["simple_H1"] + 1 + k for k in range(mlc.K_REPLICAS)]


def test_m10_empty_heldout_is_defined_zero_metric(tmp_path):
    """An empty held-out set scores the defined zero metric (no refusal).

    sps_core.relative_rmse_weighted returns 0.0 for empty input (its
    documented degenerate convention); the helper's heldout_metrics
    inherits it. The production path is guarded upstream by the
    _assert_bin_counts refusal pinned in
    test_m10_bin_count_pin_refuses_non_production_split.
    """
    rows = [(xf, pt, 1.0e-2, 1.0e-3)
            for xf, pt in zip((0.05, 0.15), M10_SYNTH_PT[:2])]
    _m10_write_table(tmp_path / "synth_alltrain.txt", rows)
    config = _m10_scratch_config(tmp_path, [
        {"species": "pi_minus", "pdg": -211, "mass_gev": PI_MASS_GEV,
         "path": "synth_alltrain.txt", "normalization": "f"},
    ])
    tables = mlc.build_tables(config)
    train_masks, heldout_masks = mlc.split_masks(tables)
    assert int(train_masks[0].sum()) == 2
    assert int(heldout_masks[0].sum()) == 0
    m_mean = {t.species: np.ones(t.data.size) for t in tables}
    held = mlc.heldout_metrics(m_mean, tables, heldout_masks)
    assert held["pooled"] == {"wrrmse": 0.0, "n_bins": 0,
                              "chi2_data_only": 0.0}
    assert held["per_species_wrrmse"] == {"pi_minus": 0.0}
    assert held["per_species_chi2_data_only"] == {"pi_minus": 0.0}
    assert held["per_species_bins"] == {"pi_minus": 0}
    assert all(math.isfinite(v) for v in held["pooled"].values())


def test_m10_single_species_single_heldout_bin_scoring(tmp_path):
    """The smallest non-degenerate shape stays finite (one species, one
    held-out bin): the scoring path must not produce NaNs.
    """
    rows = [(0.10, 0.50, 1.0e-2, 1.0e-3), (0.30, 0.50, 2.0e-2, 2.0e-3)]
    _m10_write_table(tmp_path / "synth_single.txt", rows)
    config = _m10_scratch_config(tmp_path, [
        {"species": "pi_minus", "pdg": -211, "mass_gev": PI_MASS_GEV,
         "path": "synth_single.txt", "normalization": "f"},
    ])
    tables = mlc.build_tables(config)
    train_masks, heldout_masks = mlc.split_masks(tables)
    assert int(train_masks[0].sum()) == 1
    assert int(heldout_masks[0].sum()) == 1
    arm = mlc.set_decay_hypothesis(
        mlc.arm_parameters("regge", "H3", config), "H3")
    res = mlc.evaluate_point(list(arm["start"]), arm, tables, train_masks,
                             100, [M10_TEST_SEED_BASE + 1,
                                   M10_TEST_SEED_BASE + 2], 158.0)
    assert math.isfinite(res["chi2"]) and res["chi2"] > 0.0
    held = mlc.heldout_metrics(res["_m_mean"], tables, heldout_masks)
    assert held["pooled"]["n_bins"] == 1
    assert all(math.isfinite(v) for v in held["pooled"].values())
    assert all(math.isfinite(v)
               for v in held["per_species_wrrmse"].values())
