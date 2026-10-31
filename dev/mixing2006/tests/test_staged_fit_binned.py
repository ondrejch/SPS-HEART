"""Binned six-species NA49 fit objective tests.

Task TASK-20260905-physics-review, milestone M4, pinning the committed
behavior of the nested tree at b5c44d8:

- The stage-2 fit objective ``"binned"`` is a diagonal per-bin
  chi-square over the windowed per-bin invariant spectra. The model
  estimator is the exact invariant-convention identity: a single
  synthetic event carrying one track of a table species in a single
  windowed bin must yield
  M_bin = (1 / (pi * sqrt(s) * N_ev)) * (E/pT) / (dxF * dpT),
  taken from the track four-vector and the bin edges, and a table
  whose value equals that density must give per-species chi2 0.0.
- The per-table normalization declaration is consumed: a table
  declared ``"f"`` divides data and sigma by ``sigma_inel_mb``; a
  table declared ``"f_over_sigma_inel`` is used as-is; unknown
  normalization strings and non-positive ``sigma_inel_mb`` are
  refused.
- Points outside the configured window are dropped (the window
  bounds are inclusive); a table with no points inside the window
  is refused.
- Non-positive or non-finite tabulated errors take the documented
  fallback uncertainty, the Poisson data content
  sqrt(max(data, 0)), with a unit-count floor of 1.0 on zero-data
  bins; the replaced bins are counted in ``n_sigma_fallback``.
- The M3 refusals (``regge.alpha_p0``, ``regge.transfer_slope``,
  ``regge.alpha_p_prime``) apply to the binned objective as well,
  before any event generation or table read; unknown objective
  strings, a missing ``fit.binned`` block, an empty table list, and
  a non-positive ``fit.binned.sigma_inel_mb`` are refused.
- A minimal hermetic runner run through the public
  ``run_staged_evaluation`` pins the report payload: objective,
  performed, finite chi2, fit seed protocol, statistics, sqrt(s),
  sigma_inel, window, per-table n_points/n_sigma_fallback, the
  per-species chi2 and ``sps_core`` metrics, and the freeze of the
  fitted p+p parameters into the p+A stage.
- The real pi- table pin (n_points, fallback count, data content)
  reads ``spshadrons.web.cern.ch/`` and is marked ``external_data``
  (the hermetic archive checkouts deselect it).
- M8 (uncertainty-aware objective): the binned fit prediction per bin
  is the mean of K fixed-seed MC replicas and every bin carries the
  combined error ``sigma_i^2 = sigma_data,i^2 + sigma_MC,i^2`` with
  ``sigma_MC,i`` the standard error of the replica mean (the sample
  standard deviation across the K replicas divided by sqrt(K));
  ``fit.mc_replicas`` (count >= 5 and a seed offset) is required, the
  replica seeds derive from the configuration seed plus the M8 offset
  40000, and the reported ``sps_core`` metrics stay on the data-only
  sigma (reference quantities). New tests pin the MC term's positivity
  and its 1/sqrt(K) scaling in the replica count (K = 8 against 4K = 32
  with the first 8 replicas shared: the bin-averaged ratio within 5
  percent of 2 and the per-bin median ratio within 10 percent of 2, the
  pre-registered tolerances), the zero-variance bins, the replica-seed
  rule, and the ``mc_replicas`` refusals. The objective-level pins drive
  the real ``binned_objective`` on hand-built replica samples: the
  replica-mean prediction and the combined-error chi2 pinned by hand
  (chi2 = 1/31 at K = 5 against the data-only 1/25 and the linear
  4e-6/(0.01 + sigma_MC)^2), the 1/sqrt(K) scaling at the objective
  (K = 5 against K = 10, the first 5 replicas shared: the exact
  finite-sample MC-sigma ratio 3/2 pinned to 1e-12 and the naive
  sqrt(10/5) within 10 percent, the pre-registered tolerance), the
  single-replica seed sensitivity, the bit-identical determinism of
  repeated identical runs (common random numbers), the bundled K = 8 /
  offset 40000 / base 20300831 protocol from the committed config, the
  negative seed offset refusal, and the retained data-only single-sample
  convention of ``_binned_mc_spectra``.
- M11 (one-dimensional profiles and degeneracies, acceptance criterion
  3): the profile machinery is pinned on the synthetic quadratic-form
  objective the hermetic smoke established,
  ``chi2(theta) = (theta - a)^T A (theta - a) + c0 + noise(theta;
  seeds)``: A a known positive-definite block matrix (a non-trivial
  correlation structure), a the known anchor, and the noise term seeded
  by the M11 replica-seed convention (20330832 + k, one fixed draw per
  replica). The objective's Hessian is exactly 2A, so every analytic
  reference is closed form (V = A^{-1}, the 1-sigma profile width
  sqrt(diag(V)), the inv(A) correlation, the profile crossings at
  d = sqrt(delta * V_ii), and the diagonal-dominant form
  d = sqrt(delta / A_ii)). The new pins: the pre-registration
  cross-check on a synthetic pre-registration (the untampered constant
  set accepted, each tampered constant refused with "refusing to run",
  the missing file refused) plus the read-only pin of the production
  run record (the committed pre-registration sha256, its constants
  against the harness, chi2_0, the flagged Hessian); the 1-D profile
  chi-square reproducibility at the fixed replica seeds (the seeded
  scan bit-identical across runs, the noiseless scan on the analytic
  quadratic fidelity); the 1-sigma/2-sigma interval recovery at the
  analytic crossings on the diagonal-dominant coordinate (parabolic
  reading, exact on the quadratic); the Hessian correlation matrix
  against the analytic inv(A) correlation, the symmetrize step, and the
  1/sqrt convention sigma = sqrt(2 * C_ii) with C = (2A)^{-1}; the
  at-edge flag on a deliberately flat and an edge-bounded profile (a
  level exactly at the grid edge stays bracketed); the
  non-positive-definite path (a negative-eigenvalue A: the rung reports
  no_positive_curvature, the row is zeroed, the pseudo-inverse fallback
  is flagged, and no crash); the best-of-trace + cap convention on the
  nuisance refit (a capped descent reports the cap, the trace minimum,
  and the bit-identical final re-evaluation); the 2-D point registry
  and the elongation computation against the analytic 2x2 eigenvalue
  ratio (the production-record 60 + 96 = 156 point count, the anchor
  scan-point rule, the 0.1-window fallback, and the elongation
  threshold). Call-path choice (recorded per the card): the M11
  production modes hard-require the production pre-registration and
  production-grid re-runs, so the pins drive the documented entry
  points directly (``m11_refit_nuisance``, ``m11_run_hessian``,
  ``m11_scan_grid``, ``m11_profile_intervals``, ``m11_expected_grids``,
  ``m11_materialize_point_registry``, ``m11_single_evaluation``,
  ``m11_cross_check_preregistration``) on the synthetic objective,
  with no production data and no event generation.

Call-path choices (recorded per the card): the physics-convention
tests call ``BinnedFitTable.from_arrays`` and ``_binned_mc_spectra``
directly, the private entry points that consume pure arrays; the
estimator identity test builds the event record by hand (one
``FinalParticle`` four-vector, no event generation), so the identity
is exact rather than statistical. The M8 tests call ``_mc_replica_sigma``
and ``_replica_binned_spectra`` directly on synthetic replica arrays and
hand-built replica event samples, so positivity and scaling are exact
and hermetic. The refusal tests call
``_fit_regge`` directly with ``objective: "binned"``; the refusals
fire during parameter validation and configuration checks before any
event generation, RNG consumption, or table read, so they stay
immediate and hermetic. The binned blocks carried by those
refusal tests name a table path that does not exist, which pins the
ordering: a regression that loaded the tables before validating
the parameters would fail with a file error, not the pinned
ValueError. The objective-level M8 tests replace
``na49_mixing.staged_evaluation._generate_pp`` with a test-side stub
(pytest's monkeypatch; the production module is not modified) that
returns a single-track pi- sample per replica seed, designed to realize
a hand-computed bin density through the estimator identity. The stub
keys the density on the exact replica seed ``M8_OBJECTIVE_BASE + k``
(a wrong seed raises KeyError), so the seed rule is pinned
end-to-end; the real ``_bin_tracks`` binning and the real objective
remain in the loop, and the determinism test runs the unstubbed
objective on generated events at minimal statistics. The report-payload
test runs the public
``run_staged_evaluation`` on a synthetic two-table configuration
at minimal statistics (4 events per objective evaluation, 1
Powell iteration, 4 events per stage) with ``fit.binned.path_root``
pointed at a temporary directory holding two tiny NA49-format
table files (4 columns: xF pT value error; ``#`` comment lines
allowed, per the ``sps_core.read_na49_points`` contract).
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from helpers.benchmarking import sps_core
from na49_mixing.continuum_channels import ContinuumParameters
from na49_mixing.exchange_model import (
    ReggeExcitationParameters,
    SimpleExcitationParameters,
)
from na49_mixing.excitation_kinematics import FourVector
from na49_mixing.quantum_numbers import QuantumNumbers
from na49_mixing.reaction import Reaction
from na49_mixing.source_event import ExcitationEvent, FinalParticle
from na49_mixing.staged_evaluation import (
    BinnedFitTable,
    _binned_mc_spectra,
    _bin_tracks,
    _fit_regge,
    _mc_replica_seed_base,
    _mc_replica_sigma,
    _replica_binned_spectra,
    run_staged_evaluation,
)
from na49_mixing import staged_evaluation as se

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_PI_MINUS_TABLE = (
    REPO_ROOT / "spshadrons.web.cern.ch" / "data" / "xsect_piminus.txt"
)

BEAM_MOMENTUM = 158.0
SEED = 20260905
PI_MASS_GEV = 0.1395699
SIGMA_INEL_MB = 31.78
PAPER_WINDOW = {"xf_min": 0.0, "xf_max": 0.4, "pt_min": 0.05, "pt_max": 1.5}
BUNDLED_CONFIG_PATH = REPO_ROOT / "config" / "staged_excitation.json"


def _write_na49_table(path: Path, rows: list[tuple[float, float, float, float]]) -> None:
    """Write a minimal NA49 text table (4 columns: xF pT value error).

    The ``#`` comment line exercises the parse contract of
    ``sps_core.read_na49_points``: blank and ``#`` lines are skipped,
    the first four whitespace-separated columns are read.
    """
    lines = ["# synthetic table for the hermetic binned fit"]
    for xf, pt, value, error in rows:
        lines.append(f"{xf:.2f} {pt:.2f} {value!r} {error!r}")
    path.write_text("\n".join(lines) + "\n")


def _single_bin_table(
    *,
    value: float,
    error: float,
    normalization: str = "f_over_sigma_inel",
) -> BinnedFitTable:
    """A single (xF, pT) bin table at the paper-window center (0.2, 1.0)."""
    return BinnedFitTable.from_arrays(
        species="pi_minus",
        pdg_id=-211,
        mass_gev=PI_MASS_GEV,
        normalization=normalization,
        sigma_inel_mb=SIGMA_INEL_MB,
        path="synthetic_single_bin",
        xf=np.array([0.2]),
        pt=np.array([1.0]),
        value=np.array([value]),
        error=np.array([error]),
    )


# ---------------------------------------------------------------------------
# 2a: the exact model-estimator identity
# ---------------------------------------------------------------------------


def test_binned_estimator_identity_single_track_single_bin():
    """M_bin = (1/(pi*sqrt(s)*N_ev)) * (E/pT)/(dxF*dpT), exactly.

    One synthetic event carries one pi- track (the four-vector
    (E, 0.6, 0.8, 2.0) GeV) in a single-bin table at the paper
    window. The expected density is computed by hand from the track
    four-vector and the bin edges: a single-center table bin spans
    center +/- 0.5, so dxF = dpT = 1.0. With pT = 1.0 and
    sqrt(s) = 20 the identity reduces to M = E/(pi*sqrt(s)), and a
    table whose value equals that density must give per-species
    chi2 0.0.
    """
    sqrt_s = 20.0
    n_ev = 1
    px, py, pz = 0.6, 0.8, 2.0
    pt = float(np.hypot(px, py))
    assert pt == 1.0  # the identity below relies on the exact pT
    e = math.sqrt(PI_MASS_GEV * PI_MASS_GEV + pt * pt + pz * pz)
    assert 2.0 * pz / sqrt_s == 0.2  # xF lands inside the single xF bin
    assert pt < 1.5 and 0.05 < pt  # inside the single pT bin [0.5, 1.5)
    track = FourVector(e, px, py, pz)
    event = ExcitationEvent(
        event_id=0,
        stage="regge_fit_trial",
        incoming=[FourVector(100.0, 0.0, 0.0, 0.0)],
        incoming_quantum_numbers=QuantumNumbers(1, 1, 0),
        sources=[],
        final_particles=[
            FinalParticle(
                particle_id=1,
                pdg_id=-211,
                p4=track,
                source_id=0,
                parent_id=None,
                channel_family="continuum",
            )
        ],
    )
    xedges = sps_core.edges_from_centers(np.array([0.2]))
    pedges = sps_core.edges_from_centers(np.array([1.0]))
    dxf, dpt = float(xedges[1] - xedges[0]), float(pedges[1] - pedges[0])
    assert (dxf, dpt) == (1.0, 1.0)  # single-center bins span center +/- 0.5
    m_expected = (e / pt) / (math.pi * sqrt_s * float(n_ev) * (dxf * dpt))
    table = _single_bin_table(value=m_expected, error=0.01)
    assert table.bin_index_xf[0] == 0 and table.bin_index_pt[0] == 0

    results = _binned_mc_spectra([event], [table], sqrt_s, n_params=1)
    m_table, chi2_table, metrics = results["pi_minus"]
    assert m_table.size == 1
    assert m_table[0] == m_expected  # exact estimator identity
    assert chi2_table == 0.0  # the data equal the model density
    assert metrics["pi_minus_chi2"] == 0.0


def test_binned_mc_spectra_rejects_empty_sample():
    """The estimator cannot be built from an empty event sample."""
    table = _single_bin_table(value=1.0, error=0.1)
    with pytest.raises(ValueError, match="empty sample"):
        _binned_mc_spectra([], [table], 20.0, n_params=1)


# ---------------------------------------------------------------------------
# 2b: the per-table normalization declaration
# ---------------------------------------------------------------------------


def test_normalization_declaration_conversions():
    """'f' divides data and sigma by sigma_inel_mb; the other as-is."""
    # Values that are exact binary multiples of sigma_inel_mb make the
    # division exact in floating point, so the converted arrays are
    # pinned bit-for-bit.
    table_f = BinnedFitTable.from_arrays(
        species="pi_minus",
        pdg_id=-211,
        mass_gev=PI_MASS_GEV,
        normalization="f",
        sigma_inel_mb=SIGMA_INEL_MB,
        path="synthetic",
        xf=np.array([0.1, 0.2, 0.3]),
        pt=np.array([0.5, 0.5, 0.5]),
        value=np.array([SIGMA_INEL_MB, 2.0 * SIGMA_INEL_MB, 4.0 * SIGMA_INEL_MB]),
        error=np.array([SIGMA_INEL_MB, 2.0 * SIGMA_INEL_MB, 4.0 * SIGMA_INEL_MB]),
    )
    assert table_f.normalization == "f"
    assert table_f.sigma_inel_mb == SIGMA_INEL_MB
    assert np.array_equal(table_f.data, [1.0, 2.0, 4.0])
    assert np.array_equal(table_f.sigma, [1.0, 2.0, 4.0])

    table_inel = BinnedFitTable.from_arrays(
        species="pi_minus",
        pdg_id=-211,
        mass_gev=PI_MASS_GEV,
        normalization="f_over_sigma_inel",
        sigma_inel_mb=SIGMA_INEL_MB,
        path="synthetic",
        xf=np.array([0.1, 0.2, 0.3]),
        pt=np.array([0.5, 0.5, 0.5]),
        value=np.array([1.5, 2.25, 3.0]),
        error=np.array([0.1, 0.2, 0.3]),
    )
    assert table_inel.normalization == "f_over_sigma_inel"
    assert np.array_equal(table_inel.data, [1.5, 2.25, 3.0])
    assert np.array_equal(table_inel.sigma, [0.1, 0.2, 0.3])


def test_unknown_normalization_refused():
    """A normalization string outside the declared pair is refused."""
    with pytest.raises(ValueError, match="unknown table normalization"):
        BinnedFitTable.from_arrays(
            species="pi_minus",
            pdg_id=-211,
            mass_gev=PI_MASS_GEV,
            normalization="f/sigma_inel",
            sigma_inel_mb=SIGMA_INEL_MB,
            path="synthetic",
            xf=np.array([0.2]),
            pt=np.array([0.5]),
            value=np.array([1.0]),
            error=np.array([0.1]),
        )


@pytest.mark.parametrize("sigma_inel_mb", [0.0, -5.0, float("nan")])
def test_nonpositive_sigma_inel_mb_refused(sigma_inel_mb):
    """sigma_inel_mb must be finite and positive for every table."""
    with pytest.raises(ValueError, match="sigma_inel_mb must be positive"):
        BinnedFitTable.from_arrays(
            species="pi_minus",
            pdg_id=-211,
            mass_gev=PI_MASS_GEV,
            normalization="f",
            sigma_inel_mb=sigma_inel_mb,
            path="synthetic",
            xf=np.array([0.2]),
            pt=np.array([0.5]),
            value=np.array([1.0]),
            error=np.array([0.1]),
        )


# ---------------------------------------------------------------------------
# 2c: the paper-window masking
# ---------------------------------------------------------------------------


def test_window_masking_drops_points_outside_the_window():
    """Windowed masking: outside points are dropped, bounds included.

    Four of seven synthetic points lie inside the default paper
    window (xF in [0.0, 0.4], pT in [0.05, 1.5]; the bounds are
    inclusive, so xF = 0.4 and pT = 1.5 are kept); a narrowed
    window drops the points above its xf_max.
    """
    xf = np.array([0.1, 0.3, 0.4, 0.1, 0.45, 0.1, -0.1])
    pt = np.array([0.5, 0.5, 0.5, 1.5, 0.5, 0.04, 0.5])
    value = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    error = np.full(7, 0.1)
    table = BinnedFitTable.from_arrays(
        species="pi_minus",
        pdg_id=-211,
        mass_gev=PI_MASS_GEV,
        normalization="f_over_sigma_inel",
        sigma_inel_mb=SIGMA_INEL_MB,
        path="synthetic",
        xf=xf,
        pt=pt,
        value=value,
        error=error,
    )
    assert table.data.size == 4
    assert np.array_equal(table.data, [1.0, 2.0, 3.0, 4.0])
    assert np.array_equal(table.xf_centers, [0.1, 0.3, 0.4])
    assert np.array_equal(table.pt_centers, [0.5, 1.5])

    narrow = BinnedFitTable.from_arrays(
        species="pi_minus",
        pdg_id=-211,
        mass_gev=PI_MASS_GEV,
        normalization="f_over_sigma_inel",
        sigma_inel_mb=SIGMA_INEL_MB,
        path="synthetic",
        xf=xf,
        pt=pt,
        value=value,
        error=error,
        window={"xf_min": 0.0, "xf_max": 0.2, "pt_min": 0.05, "pt_max": 1.5},
    )
    assert narrow.data.size == 2
    # Kept: (0.1, 0.5) -> 1.0 and (0.1, 1.5) -> 4.0; the xf > 0.2 points
    # (values 2.0, 3.0, 5.0) and the pT < 0.05 point (value 6.0) are dropped.
    assert np.array_equal(narrow.data, [1.0, 4.0])


def test_all_points_outside_window_refused():
    """A table with no points inside the window cannot be fitted."""
    with pytest.raises(ValueError, match="no pi_minus table points inside the fit window"):
        BinnedFitTable.from_arrays(
            species="pi_minus",
            pdg_id=-211,
            mass_gev=PI_MASS_GEV,
            normalization="f",
            sigma_inel_mb=SIGMA_INEL_MB,
            path="synthetic",
            xf=np.array([0.9, 0.95]),
            pt=np.array([0.5, 0.6]),
            value=np.array([1.0, 1.0]),
            error=np.array([0.1, 0.1]),
        )


# ---------------------------------------------------------------------------
# 2d: the sigma fallback
# ---------------------------------------------------------------------------


def test_sigma_fallback_is_poisson_data_content_with_unit_floor():
    """Bad tabulated errors take sqrt(data), floored at 1.0 on empty bins.

    Four bins: a good error, a zero error on a zero-data bin (unit
    floor), a NaN error on a positive-data bin (Poisson content),
    and another good error. Exactly the two bad bins are replaced
    and counted.
    """
    table = BinnedFitTable.from_arrays(
        species="pi_minus",
        pdg_id=-211,
        mass_gev=PI_MASS_GEV,
        normalization="f_over_sigma_inel",
        sigma_inel_mb=SIGMA_INEL_MB,
        path="synthetic",
        xf=np.array([0.1, 0.2, 0.3, 0.4]),
        pt=np.array([0.5, 0.6, 0.7, 0.8]),
        value=np.array([2.0, 0.0, 5.0, 3.0]),
        error=np.array([1.0, 0.0, float("nan"), 4.0]),
    )
    assert table.n_sigma_fallback == 2
    assert np.array_equal(table.data, [2.0, 0.0, 5.0, 3.0])
    assert np.array_equal(table.sigma, [1.0, 1.0, math.sqrt(5.0), 4.0])


# ---------------------------------------------------------------------------
# 2e: the refusals under the binned objective
# ---------------------------------------------------------------------------


def _binned_refusal_error(parameters: list[str]) -> ValueError:
    """Run ``_fit_regge`` with the binned objective and return the ValueError.

    The ``fit.binned`` block names a table path that does not exist
    (no ``path_root`` override: it would resolve against the nested
    root). The refusals fire before any table read or event
    generation, so a correct implementation never touches the
    missing file; a regression that loaded the tables first would
    fail with a file error instead of the pinned ValueError.
    """
    fit_cfg = {
        "enabled": True,
        "objective": "binned",
        "parameters": list(parameters),
        "binned": {
            "sigma_inel_mb": SIGMA_INEL_MB,
            "tables": [
                {
                    "species": "pi_minus",
                    "pdg": -211,
                    "mass_gev": PI_MASS_GEV,
                    "path": "definitely_absent_table.txt",
                    "normalization": "f",
                }
            ],
        },
    }
    with pytest.raises(ValueError) as excinfo:
        _fit_regge(
            fit_cfg,
            SimpleExcitationParameters(),
            ReggeExcitationParameters(),
            ContinuumParameters(),
            BEAM_MOMENTUM,
            seed=SEED,
        )
    return excinfo.value


@pytest.mark.parametrize(
    ("parameters", "refused"),
    [
        (["regge.alpha_p0"], "regge.alpha_p0"),
        (["regge.transfer_slope"], "regge.transfer_slope"),
        (["regge.alpha_p_prime"], "regge.alpha_p_prime"),
    ],
)
def test_binned_objective_refuses_non_identifiable_parameters(parameters, refused):
    """The M3 refusals apply to the binned objective as well."""
    message = str(_binned_refusal_error(parameters))
    assert f"{refused} is not fittable at one fixed energy" in message
    if refused == "regge.alpha_p0":
        assert "only changes the overall normalized event weight" in message
    else:
        assert "B_eff = B + 2 alpha'_P ln(s/s0)" in message
        assert (
            "regge.effective_transfer_slope is the single fittable transfer "
            "degree of freedom" in message
        )
        assert "energy lever arm or independent transfer data" in message


def test_unknown_fit_objective_refused():
    """Objective strings outside {'aggregate', 'binned'} are refused."""
    fit_cfg = {
        "enabled": True,
        "objective": "bogus",
        "parameters": ["regge.edge_power"],
        "bounds": {"regge.edge_power": [0.25, 6.0]},
    }
    with pytest.raises(ValueError, match="unknown fit objective 'bogus'"):
        _fit_regge(
            fit_cfg,
            SimpleExcitationParameters(),
            ReggeExcitationParameters(),
            ContinuumParameters(),
            BEAM_MOMENTUM,
            seed=SEED,
        )


def test_binned_objective_requires_binned_block():
    """The binned objective without a fit.binned block is refused."""
    fit_cfg = {
        "enabled": True,
        "objective": "binned",
        "parameters": ["regge.edge_power"],
        "bounds": {"regge.edge_power": [0.25, 6.0]},
    }
    with pytest.raises(ValueError, match="requires a fit.binned block"):
        _fit_regge(
            fit_cfg,
            SimpleExcitationParameters(),
            ReggeExcitationParameters(),
            ContinuumParameters(),
            BEAM_MOMENTUM,
            seed=SEED,
        )


def test_binned_objective_requires_at_least_one_table():
    """An empty fit.binned.tables list is refused."""
    fit_cfg = {
        "enabled": True,
        "objective": "binned",
        "parameters": ["regge.edge_power"],
        "bounds": {"regge.edge_power": [0.25, 6.0]},
        "binned": {"sigma_inel_mb": SIGMA_INEL_MB, "tables": []},
    }
    with pytest.raises(ValueError, match="must name at least one table"):
        _fit_regge(
            fit_cfg,
            SimpleExcitationParameters(),
            ReggeExcitationParameters(),
            ContinuumParameters(),
            BEAM_MOMENTUM,
            seed=SEED,
        )


@pytest.mark.parametrize("sigma_inel_mb", [0.0, -1.0])
def test_binned_objective_requires_positive_sigma_inel(sigma_inel_mb):
    """A non-positive fit.binned.sigma_inel_mb is refused before any read."""
    fit_cfg = {
        "enabled": True,
        "objective": "binned",
        "parameters": ["regge.edge_power"],
        "bounds": {"regge.edge_power": [0.25, 6.0]},
        "binned": {
            "sigma_inel_mb": sigma_inel_mb,
            "tables": [
                {
                    "species": "pi_minus",
                    "pdg": -211,
                    "mass_gev": PI_MASS_GEV,
                    "path": "definitely_absent_table.txt",
                    "normalization": "f",
                }
            ],
        },
    }
    with pytest.raises(ValueError, match="fit.binned.sigma_inel_mb must be positive"):
        _fit_regge(
            fit_cfg,
            SimpleExcitationParameters(),
            ReggeExcitationParameters(),
            ContinuumParameters(),
            BEAM_MOMENTUM,
            seed=SEED,
        )


# ---------------------------------------------------------------------------
# 2e': M8 MC-replica machinery (acceptance criterion 1)
# ---------------------------------------------------------------------------

MC_TEST_SEED = 20260906


def test_mc_replica_sigma_positive_and_scales_as_1_over_sqrt_k():
    """The MC term is positive and scales as 1/sqrt(K) in the replica count.

    M8 acceptance criterion 1, hermetic synthetic replica arrays (no
    event generation, no production data). Pre-registered K pair and
    tolerances: K = 8 against 4K = 32, the 8 replica rows being the
    first 8 of the 32 (the fixed-seed subset), with the bin-averaged
    ratio sigma_MC(K = 8) / sigma_MC(K = 32) within 5 percent of the
    expected 2 and the per-bin median ratio within 10 percent of 2.
    sigma_MC is ``_mc_replica_sigma``: the sample standard deviation
    (ddof = 1) across the K replicas divided by sqrt(K), the standard
    error of the replica-mean prediction, so the 1/sqrt(K) scaling is
    the estimator definition and the ratio test pins it against the
    sampling noise of the two std estimates. Bins whose replica values
    are identical (zero replica variance) give zero to within the last-ulp
    summation rounding of the identical values.
    """
    rng = np.random.default_rng(MC_TEST_SEED)
    n_bins = 256
    mu = np.linspace(0.5, 4.0, n_bins)
    sigma_true = np.full(n_bins, 0.8)
    zero_variance_bins = np.arange(0, n_bins, 40)  # 6 identical-replica bins
    sigma_true[zero_variance_bins] = 0.0
    rows = mu[None, :] + rng.normal(0.0, 1.0, size=(32, n_bins)) * sigma_true[None, :]

    sigma_mc_8 = _mc_replica_sigma(rows[:8])
    sigma_mc_32 = _mc_replica_sigma(rows)

    # Estimator identity: sigma_MC * sqrt(K) is the ddof=1 sample std
    # (equal up to the last-ulp rounding of the sqrt division/multiplication).
    np.testing.assert_allclose(
        sigma_mc_8 * math.sqrt(8.0),
        np.std(rows[:8], axis=0, ddof=1),
        rtol=1e-12,
        atol=0.0,
    )
    # Positivity: strictly positive on bins with non-zero replica variance.
    nonzero = sigma_true > 0.0
    assert np.all(sigma_mc_8[nonzero] > 0.0)
    assert np.all(sigma_mc_32[nonzero] > 0.0)
    # Zero replica variance gives zero up to the last-ulp summation
    # rounding of the identical values (the std of a constant column sits
    # at the 1e-16 level, not exactly 0.0).
    assert np.all(sigma_mc_8[~nonzero] >= 0.0)
    assert np.all(sigma_mc_8[~nonzero] < 1e-15)
    assert np.all(sigma_mc_32[~nonzero] >= 0.0)
    assert np.all(sigma_mc_32[~nonzero] < 1e-15)
    # 1/sqrt(K) scaling within the pre-registered tolerances.
    ratio = sigma_mc_8[nonzero] / sigma_mc_32[nonzero]
    assert float(np.mean(ratio)) == pytest.approx(2.0, rel=0.05)
    assert float(np.median(ratio)) == pytest.approx(2.0, rel=0.10)


def test_mc_replica_sigma_requires_two_replicas():
    """A (K, n_bins) array with K >= 2 is required."""
    with pytest.raises(ValueError, match="K >= 2"):
        _mc_replica_sigma(np.ones((1, 3)))
    with pytest.raises(ValueError, match="K >= 2"):
        _mc_replica_sigma(np.ones(3))


def test_replica_binned_spectra_exact_match_gives_zero_combined_chi2():
    """Identical replicas make sigma_MC zero, so an exact data match gives chi2 0.

    M8 counterpart of the M4 single-sample identity: five identical
    hand-built replica samples (the single pi- track of the estimator
    identity test) have zero replica variance, so sigma_MC = 0 and the
    combined error reduces to the data sigma; a table whose value equals
    the density must give per-species combined chi2 exactly 0.0.
    """
    sqrt_s = 20.0
    px, py, pz = 0.6, 0.8, 2.0
    pt = float(np.hypot(px, py))
    e = math.sqrt(PI_MASS_GEV * PI_MASS_GEV + pt * pt + pz * pz)
    track = FourVector(e, px, py, pz)

    def one_event() -> ExcitationEvent:
        return ExcitationEvent(
            event_id=0,
            stage="regge_fit_replica",
            incoming=[FourVector(100.0, 0.0, 0.0, 0.0)],
            incoming_quantum_numbers=QuantumNumbers(1, 1, 0),
            sources=[],
            final_particles=[
                FinalParticle(
                    particle_id=1,
                    pdg_id=-211,
                    p4=track,
                    source_id=0,
                    parent_id=None,
                    channel_family="continuum",
                )
            ],
        )

    xedges = sps_core.edges_from_centers(np.array([0.2]))
    pedges = sps_core.edges_from_centers(np.array([1.0]))
    dxf, dpt = float(xedges[1] - xedges[0]), float(pedges[1] - pedges[0])
    m_expected = (e / pt) / (math.pi * sqrt_s * 1.0 * (dxf * dpt))
    table = _single_bin_table(value=m_expected, error=0.01)
    replica_spectra = [
        _bin_tracks([one_event()], [table], sqrt_s) for _ in range(5)
    ]
    m_mean, sigma_mc, sigma_replica = _replica_binned_spectra(
        replica_spectra, [table]
    )
    assert m_mean["pi_minus"][0] == m_expected  # exact estimator identity
    assert np.all(sigma_replica["pi_minus"] == 0.0)
    assert np.all(sigma_mc["pi_minus"] == 0.0)  # identical replicas
    # The objective's combined-error formula with sigma_MC = 0.
    sigma_combined = np.sqrt(table.sigma**2 + sigma_mc["pi_minus"] ** 2)
    residual = (m_mean["pi_minus"] - table.data) / sigma_combined
    assert float(np.dot(residual, residual)) == 0.0


def test_mc_replica_seed_rule_derives_from_the_config_seed():
    """The replica seeds are config.seed + seed_offset + k, M8 offset 40000.

    The runner passes the fit substream seed config.seed + 10000 into the
    stage-2 fit, so the base is fit_seed - 10000 + seed_offset. For the
    bundled protocol that is 20260831 + 40000 = 20300831, and the bundled
    K = 8 replica seeds collide with neither the fit event-generation
    stream nor the recorded M4/M5/M7 milestone streams.
    """
    base = _mc_replica_seed_base(fit_seed=20270831, seed_offset=40000)
    assert base == 20300831
    assert base == 20260831 + 40_000  # the runner identity
    replica_seeds = [base + k for k in range(8)]
    recorded_streams = {
        20260831,  # stage-1 simple stream
        20260832,  # post-fit regge stage (config.seed + 1)
        20260833,  # p+A stage (config.seed + 2)
        20270831,  # M4 fit substream (config.seed + 10000)
        20280831,  # M5 audit stream (offset 20000)
        20290831,  # M7 comparison stream (offset 30000)
    }
    assert not (set(replica_seeds) & recorded_streams)


def _binned_mc_refusal_error(mc_cfg: dict | None) -> ValueError:
    """Run ``_fit_regge`` with the binned objective and the given mc block.

    Same nonexistent-table ordering pin as the other binned refusals:
    the ``mc_replicas`` checks sit with the other configuration checks,
    before any table read or event generation.
    """
    fit_cfg = {
        "enabled": True,
        "objective": "binned",
        "parameters": ["regge.edge_power"],
        "bounds": {"regge.edge_power": [0.25, 6.0]},
        "binned": {
            "sigma_inel_mb": SIGMA_INEL_MB,
            "tables": [
                {
                    "species": "pi_minus",
                    "pdg": -211,
                    "mass_gev": PI_MASS_GEV,
                    "path": "definitely_absent_table.txt",
                    "normalization": "f",
                }
            ],
        },
    }
    if mc_cfg is not None:
        fit_cfg["mc_replicas"] = mc_cfg
    with pytest.raises(ValueError) as excinfo:
        _fit_regge(
            fit_cfg,
            SimpleExcitationParameters(),
            ReggeExcitationParameters(),
            ContinuumParameters(),
            BEAM_MOMENTUM,
            seed=SEED,
        )
    return excinfo.value


def test_binned_objective_requires_mc_replicas_block():
    """The uncertainty-aware binned objective has no data-only fallback."""
    message = str(_binned_mc_refusal_error(None))
    assert "requires fit.mc_replicas" in message
    assert "the bundled config records the protocol" in message


def test_binned_objective_requires_mc_replica_count_at_least_5():
    """The pre-registered mechanism requires K >= 5 replicas."""
    message = str(_binned_mc_refusal_error({"count": 4, "seed_offset": 40000}))
    assert "fit.mc_replicas.count must be >= 5, got 4" in message


def test_binned_objective_requires_mc_replica_seed_offset():
    """The seed offset is part of the recorded protocol, not a default."""
    message = str(_binned_mc_refusal_error({"count": 5}))
    assert "fit.mc_replicas.seed_offset is required" in message


# ---------------------------------------------------------------------------
# 2e'': M8 objective-level pins (acceptance criteria 1 and 3, end-to-end)
# ---------------------------------------------------------------------------

# Objective-level design: one pi- track per replica event against a single
# (0.2, 1.0) bin table. The track four-vector (E, 0.6, 0.8, p_z) realizes
# the bin density D = E / (pi * sqrt(s)) exactly (pT = 1.0 and single-bin
# areas dxF = dpT = 1.0), and the xF values of the densities below all fall
# inside the bin, so the section 2a estimator identity holds replica by
# replica. The objective arithmetic is then closed form: with table value
# 0.05 and data sigma 0.01 the replica-mean residual and the combined
# variance are hand-computable, and the single-bin chi-square reduces to
# the residual squared over that combined variance.
M8_OBJECTIVE_SEED = SEED + 10_000  # the runner's fit-seed protocol
M8_OBJECTIVE_OFFSET = 40_000
M8_OBJECTIVE_BASE = _mc_replica_seed_base(
    M8_OBJECTIVE_SEED, M8_OBJECTIVE_OFFSET
)
M8_SQRT_S = math.sqrt(Reaction(BEAM_MOMENTUM, 2212, 2212).s())
M8_TABLE_VALUE = 0.05
M8_TABLE_SIGMA = 0.01


def _m8_single_track_event(density: float) -> ExcitationEvent:
    """A single pi- track whose (0.2, 1.0) bin density is ``density``.

    The estimator identity (section 2a) with N_ev = 1, pT = 1.0, and
    single-bin areas 1.0 gives M_bin = E / (pi * sqrt(s)), so the energy
    E = density * pi * sqrt(s) realizes the density exactly and p_z
    follows from the on-shell constraint.
    """
    e = density * math.pi * M8_SQRT_S
    pz = math.sqrt(e * e - PI_MASS_GEV * PI_MASS_GEV - 1.0)
    track = FourVector(e, 0.6, 0.8, pz)
    return ExcitationEvent(
        event_id=0,
        stage="regge_fit_replica",
        incoming=[FourVector(100.0, 0.0, 0.0, 0.0)],
        incoming_quantum_numbers=QuantumNumbers(1, 1, 0),
        sources=[],
        final_particles=[
            FinalParticle(
                particle_id=1,
                pdg_id=-211,
                p4=track,
                source_id=0,
                parent_id=None,
                channel_family="continuum",
            )
        ],
    )


def _run_m8_stubbed_objective(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    densities: list[float],
) -> dict:
    """Drive the real binned objective on hand-built replica densities.

    Event generation is replaced by a test-side stub keyed on the exact
    replica seed ``M8_OBJECTIVE_BASE + k`` (a wrong seed raises
    KeyError), so the M8 seed rule is pinned end-to-end through
    ``_fit_regge``. The real ``_bin_tracks`` binning, the real replica
    machinery, and the real objective remain in the loop. The stub
    ignores the trial parameters, so the objective is flat and the fit
    reports the initial parameters at the constant hand-computed chi2.
    """
    by_seed = {
        M8_OBJECTIVE_BASE + k: _m8_single_track_event(d)
        for k, d in enumerate(densities)
    }

    def stub(n_events, seed, simple, regge, continuum, mode, beam_momentum, stage):
        assert stage == "regge_fit_replica"
        assert mode == "regge"
        assert n_events == 1
        return [by_seed[int(seed)]], {}

    monkeypatch.setattr("na49_mixing.staged_evaluation._generate_pp", stub)

    _write_na49_table(
        tmp_path / "pi_minus_m8.txt",
        [(0.20, 1.00, M8_TABLE_VALUE, M8_TABLE_SIGMA)],
    )
    fit_cfg = {
        "enabled": True,
        "objective": "binned",
        "events_per_evaluation": 1,
        "max_iterations": 1,
        "mc_replicas": {
            "count": len(densities),
            "seed_offset": M8_OBJECTIVE_OFFSET,
        },
        "parameters": ["regge.edge_power"],
        "bounds": {"regge.edge_power": [0.25, 6.0]},
        "binned": {
            "sigma_inel_mb": SIGMA_INEL_MB,
            "path_root": str(tmp_path),
            "tables": [
                {
                    "species": "pi_minus",
                    "pdg": -211,
                    "mass_gev": PI_MASS_GEV,
                    "path": "pi_minus_m8.txt",
                    "normalization": "f_over_sigma_inel",
                }
            ],
        },
    }
    _fitted_regge, _fitted_continuum, report = _fit_regge(
        fit_cfg,
        SimpleExcitationParameters(),
        ReggeExcitationParameters(),
        ContinuumParameters(),
        BEAM_MOMENTUM,
        seed=M8_OBJECTIVE_SEED,
    )
    return report


def test_binned_objective_replica_mean_and_combined_error_hand_computed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """The objective chi2 is the hand-computed combined-error chi-square.

    K = 5 replicas of the designed densities 0.05 +/- 0.01 (three at +s,
    two at -s): the replica mean is D + s/5 = 0.052, the sample
    standard deviation across the replicas is s * sqrt(1.2), and the MC
    sigma is s * sqrt(1.2 / 5). The single-bin objective chi2 is
    (s/5)^2 / (s^2 * (1 + 1.2/5)) = 1/31, against the data-only metrics
    chi2 (s/5)^2 / s^2 = 1/25 on the same replica-mean prediction: the
    combined error is the larger sigma, so the objective chi2 is the
    smaller number. A linear sigma addition (s + sigma_MC in the
    denominator) is a different number again, separated by more than
    the pin tolerance.
    """
    s = M8_TABLE_SIGMA
    densities = [0.06, 0.04, 0.06, 0.04, 0.06]  # three +s, two -s
    report = _run_m8_stubbed_objective(monkeypatch, tmp_path, densities)

    sigma_mc = s * math.sqrt(1.2 / 5.0)
    assert report["chi2"] == pytest.approx(1.0 / 31.0, rel=1e-12)
    payload = report["species"]["pi_minus"]
    assert payload["pi_minus_chi2"] == pytest.approx(1.0 / 25.0, rel=1e-12)
    assert payload["pi_minus_chi2_dof"] == payload["pi_minus_chi2"]  # dof = 1
    assert report["chi2"] < payload["pi_minus_chi2"]
    assert payload["mc_sigma_mean_over_bins"] == pytest.approx(sigma_mc, rel=1e-12)
    assert payload["mc_sigma_max_over_bins"] == payload["mc_sigma_mean_over_bins"]
    # Linear sigma addition is a distinct (smaller) chi-square.
    linear_chi2 = (s / 5.0) ** 2 / (s + sigma_mc) ** 2
    assert report["chi2"] - linear_chi2 > 1e-3
    # Common random numbers: the same seed reproduces the run exactly.
    report2 = _run_m8_stubbed_objective(monkeypatch, tmp_path, densities)
    assert report2["chi2"] == report["chi2"]
    assert report2["parameters"] == report["parameters"]


def test_binned_objective_mc_term_scales_as_1_over_sqrt_k(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """The objective's MC term scales as 1/sqrt(K) in the replica count.

    Pre-registered objective-level K pair: K = 5 against K = 10, the 5
    replica rows being the first 5 of the 10 (the fixed-seed subset) on
    the designed +/- 0.01 pattern (six +s and four -s at K = 10). The
    exact finite-sample MC-sigma ratio is 3/2 (the designed variances
    are 1.2 s^2 at K = 5 and 9.6 s^2 / 9 at K = 10), pinned to 1e-12;
    the naive asymptotic sqrt(10/5) = sqrt(2) is within the
    pre-registered 10 percent tolerance, the same tolerance the
    helper-level test applies. The objective chi2s are 1/31 (K = 5)
    and 3/83 (K = 10): the replica mean stays 0.052 in both runs.
    """
    b5 = [0.06, 0.04, 0.06, 0.04, 0.06]
    b10 = [0.06, 0.04, 0.06, 0.04, 0.06, 0.06, 0.04, 0.06, 0.04, 0.06]
    report5 = _run_m8_stubbed_objective(monkeypatch, tmp_path, b5)
    report10 = _run_m8_stubbed_objective(monkeypatch, tmp_path, b10)

    sigma5 = report5["species"]["pi_minus"]["mc_sigma_mean_over_bins"]
    sigma10 = report10["species"]["pi_minus"]["mc_sigma_mean_over_bins"]
    ratio = sigma5 / sigma10
    assert ratio == pytest.approx(1.5, rel=1e-12)  # exact finite-sample ratio
    assert ratio == pytest.approx(math.sqrt(10.0 / 5.0), rel=0.10)
    assert report5["chi2"] == pytest.approx(1.0 / 31.0, rel=1e-12)
    assert report10["chi2"] == pytest.approx(3.0 / 83.0, rel=1e-12)


def test_binned_objective_single_replica_change_moves_mc_sigma(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Changing one replica's seed moves the MC sigma and the chi2.

    The third replica realizes 0.055 instead of 0.06: the replica mean
    drops to 0.051, the sample variance drops from 1.2e-4 to
    1.05e-4, the MC sigma to 0.004582575694955840, and the objective
    chi2 to (0.001)^2 / (1e-4 + 2.1e-5) = 1/121. The objective is a
    function of all K replica streams: no single replica may be dropped
    or altered without moving the reported chi2.
    """
    report = _run_m8_stubbed_objective(
        monkeypatch, tmp_path, [0.06, 0.04, 0.055, 0.04, 0.06]
    )
    payload = report["species"]["pi_minus"]
    assert report["chi2"] == pytest.approx(1.0 / 121.0, rel=1e-12)
    assert payload["mc_sigma_mean_over_bins"] == pytest.approx(
        0.004582575694955840, rel=1e-12
    )
    assert (
        abs(payload["mc_sigma_mean_over_bins"] - M8_TABLE_SIGMA * math.sqrt(1.2 / 5.0))
        > 1e-4
    )


def test_binned_objective_common_random_numbers_deterministic(tmp_path):
    """Repeated identical runs are bit-equal (common random numbers).

    Minimal statistics over one synthetic table (1 event per replica,
    K = 5, 1 Powell iteration, one parameter): the unstubbed objective
    on generated events is deterministic in the configuration and the
    seed, so two runs at the same fit seed report bit-equal chi2,
    parameters, and evaluation count. The reported mc_replicas block
    records the test protocol (count, seed offset, replica seed base),
    and the per-species objective chi2 stays at or below the data-only
    metrics chi2, strictly here because the generated replicas carry
    non-zero variance.
    """
    _write_na49_table(
        tmp_path / "pi_minus.txt",
        [
            (0.10, 0.30, 1.0, 0.1),
            (0.20, 0.30, 2.0, 0.2),
            (0.30, 0.60, 3.0, 0.3),
            (0.10, 0.60, 4.0, 0.4),
        ],
    )
    fit_cfg = {
        "enabled": True,
        "objective": "binned",
        "events_per_evaluation": 1,
        "max_iterations": 1,
        "mc_replicas": {
            "count": 5,
            "seed_offset": M8_OBJECTIVE_OFFSET,
        },
        "parameters": ["regge.edge_power"],
        "bounds": {"regge.edge_power": [0.25, 6.0]},
        "binned": {
            "sigma_inel_mb": SIGMA_INEL_MB,
            "path_root": str(tmp_path),
            "tables": [
                {
                    "species": "pi_minus",
                    "pdg": -211,
                    "mass_gev": PI_MASS_GEV,
                    "path": "pi_minus.txt",
                    "normalization": "f",
                }
            ],
        },
    }
    _r1, _c1, report1 = _fit_regge(
        fit_cfg,
        SimpleExcitationParameters(),
        ReggeExcitationParameters(),
        ContinuumParameters(),
        BEAM_MOMENTUM,
        seed=SEED + 10_000,
    )
    _r2, _c2, report2 = _fit_regge(
        fit_cfg,
        SimpleExcitationParameters(),
        ReggeExcitationParameters(),
        ContinuumParameters(),
        BEAM_MOMENTUM,
        seed=SEED + 10_000,
    )
    assert report1["chi2"] == report2["chi2"]
    assert report1["parameters"] == report2["parameters"]
    assert report1["n_objective_evaluations"] == report2["n_objective_evaluations"]

    mc = report1["mc_replicas"]
    assert mc["count"] == 5
    assert mc["seed_offset"] == M8_OBJECTIVE_OFFSET
    assert mc["replica_seed_base"] == SEED + M8_OBJECTIVE_OFFSET
    assert mc["events_per_replica"] == 1

    payload = report1["species"]["pi_minus"]
    assert payload["mc_sigma_max_over_bins"] > 0.0
    assert payload["chi2"] <= payload["pi_minus_chi2"]
    assert payload["chi2"] < payload["pi_minus_chi2"]


def test_replica_binned_spectra_two_replica_hand_computed_combined_error():
    """Two hand-built replicas pin the replica machinery and its chi2.

    The objective requires K >= 5, so this two-replica case exercises
    ``_replica_binned_spectra`` directly: replicas [1.0, 3.0] and
    [2.0, 4.0] on two bins with data [1.4, 3.6] and data sigma
    [0.25, 0.5]. The replica mean is [1.5, 3.5] exactly, the MC sigma
    is |d1 - d2| / 2 = 0.5 per bin (the K = 2 standard error of the
    mean), and the per-replica sigma is 1/sqrt(2). Under the objective
    convention the combined-error chi2 is 0.032 + 0.02 = 0.052, the
    data-only chi2 is 0.2, and a linear sigma addition gives 0.0278.
    """
    table = BinnedFitTable.from_arrays(
        species="pi_minus",
        pdg_id=-211,
        mass_gev=PI_MASS_GEV,
        normalization="f_over_sigma_inel",
        sigma_inel_mb=SIGMA_INEL_MB,
        path="synthetic_two_bin",
        xf=np.array([0.1, 0.3]),
        pt=np.array([0.5, 0.5]),
        value=np.array([1.4, 3.6]),
        error=np.array([0.25, 0.5]),
    )
    replica_spectra = [
        {"pi_minus": np.array([1.0, 3.0])},
        {"pi_minus": np.array([2.0, 4.0])},
    ]
    m_mean, sigma_mc, sigma_replica = _replica_binned_spectra(
        replica_spectra, [table]
    )
    np.testing.assert_array_equal(m_mean["pi_minus"], [1.5, 3.5])
    np.testing.assert_allclose(sigma_mc["pi_minus"], [0.5, 0.5], atol=1e-15)
    np.testing.assert_allclose(
        sigma_replica["pi_minus"],
        [1.0 / math.sqrt(2.0)] * 2,
        rtol=1e-15,
    )
    residual = m_mean["pi_minus"] - table.data
    sigma_combined = np.sqrt(table.sigma**2 + sigma_mc["pi_minus"] ** 2)
    chi2_combined = float(np.dot(residual / sigma_combined, residual / sigma_combined))
    assert chi2_combined == pytest.approx(0.052, rel=1e-12)
    chi2_data = float(np.dot(residual / table.sigma, residual / table.sigma))
    assert chi2_data == pytest.approx(0.2, rel=1e-12)
    chi2_linear = float(
        np.dot(
            residual / (table.sigma + sigma_mc["pi_minus"]),
            residual / (table.sigma + sigma_mc["pi_minus"]),
        )
    )
    assert chi2_linear == pytest.approx(0.027777777777777776, rel=1e-12)
    assert chi2_combined - chi2_linear > 1e-3
    assert chi2_combined < chi2_data


def test_bundled_config_records_the_m8_replica_protocol():
    """The committed config records K, the seed offset, and the seed rule.

    M8 acceptance criterion 3: the configuration records the replica
    count and seed rule, not just the report. The committed bundled
    protocol is K = 8 at offset 40000 from seed 20260831, replica
    seeds 20300831 + k, distinct from the M4 fit substream
    (20270831), the M5 audit stream (20280831), and the M7 comparison
    stream (20290831).
    """
    config = json.loads(BUNDLED_CONFIG_PATH.read_text())
    seed = config["seed"]
    assert seed == 20260831
    fit = config["fit"]
    assert fit["objective"] == "binned"
    mc = fit["mc_replicas"]
    assert mc["count"] == 8
    assert mc["seed_offset"] == 40000
    # The runner passes config.seed + 10000 to the stage-2 fit.
    assert _mc_replica_seed_base(seed + 10_000, mc["seed_offset"]) == 20300831
    assert "20300831 + k" in mc["seed_rule"]
    replica_seeds = {20300831 + k for k in range(mc["count"])}
    recorded_streams = {
        20260831,  # stage-1 simple stream
        20260832,  # post-fit regge stage (config.seed + 1)
        20260833,  # p+A stage (config.seed + 2)
        20270831,  # M4 fit substream (config.seed + 10000)
        20280831,  # M5 audit stream (offset 20000)
        20290831,  # M7 comparison stream (offset 30000)
    }
    assert not (replica_seeds & recorded_streams)


def test_binned_objective_refuses_negative_mc_replica_seed_offset():
    """A negative seed offset is refused, like the other protocol checks."""
    message = str(_binned_mc_refusal_error({"count": 5, "seed_offset": -40000}))
    assert "fit.mc_replicas.seed_offset must be non-negative, got -40000" in message


def test_binned_mc_spectra_retains_the_data_only_single_sample_convention():
    """The M7 harness entry point keeps the single-sample data-only chi2.

    One single-track event (density 0.06 through the estimator
    identity) against a single-bin table (value 0.05, sigma 0.01):
    ``_binned_mc_spectra`` reports the data-only chi2 (0.06 - 0.05)^2 /
    0.01^2 = 1.0 on the single-sample density, with the metrics on the
    same density and sigma (dof = max(1 - 1, 1) = 1, so chi2_dof
    equals chi2), and the reported density is the shared ``_bin_tracks``
    core, bit-equal.
    """
    event = _m8_single_track_event(0.06)
    table = _single_bin_table(value=0.05, error=0.01)
    densities, chi2_table, metrics = _binned_mc_spectra(
        [event], [table], M8_SQRT_S, n_params=1
    )["pi_minus"]
    assert densities[0] == pytest.approx(0.06, rel=1e-12)
    assert chi2_table == pytest.approx(1.0, rel=1e-12)
    assert metrics["pi_minus_chi2"] == chi2_table
    assert metrics["pi_minus_chi2_dof"] == chi2_table
    shared = _bin_tracks([event], [table], M8_SQRT_S)["pi_minus"]
    np.testing.assert_array_equal(densities, shared)


# ---------------------------------------------------------------------------
# 2f: the report payload of a minimal hermetic binned run
# ---------------------------------------------------------------------------


def _hermetic_binned_config(tmp_path: Path) -> dict:
    """A two-table binned configuration over synthetic NA49-format files.

    ``fit.binned.path_root`` points at ``tmp_path`` so the relative
    table paths resolve against the temporary directory instead of
    the nested root (the hermetic override the M4 machinery exists
    for). The pi+ table carries one zero error so the reported
    ``n_sigma_fallback`` is nonzero and pinned.
    """
    _write_na49_table(
        tmp_path / "pi_minus.txt",
        [
            (0.10, 0.30, 1.0, 0.1),
            (0.20, 0.30, 2.0, 0.2),
            (0.30, 0.60, 3.0, 0.3),
            (0.10, 0.60, 4.0, 0.4),
        ],
    )
    _write_na49_table(
        tmp_path / "pi_plus.txt",
        [
            (0.10, 0.30, 1.0, 0.1),
            (0.20, 0.30, 2.0, 0.0),
            (0.30, 0.60, 3.0, 0.3),
            (0.10, 0.60, 4.0, 0.4),
        ],
    )
    return {
        "seed": SEED,
        "through": "baryon_nucleus",
        "events_per_stage": 4,
        "beam_momentum": BEAM_MOMENTUM,
        "regge": {
            "alpha_p0": 1.0808,
            "alpha_p_prime": 0.25,
            "alpha_exc0": 0.5,
            "transfer_slope": 4.5,
            "edge_power": 2.0,
            "s0": 1.0,
            "grid_size": 96,
            "effective_transfer_slope": 7.3489827740435505,
        },
        "fit": {
            "enabled": True,
            "objective": "binned",
            "events_per_evaluation": 4,
            "max_iterations": 1,
            "mc_replicas": {
                "count": 5,
                "events_per_replica": 4,
                "seed_offset": 40000,
                "seed_rule": "replica k uses seed = SEED - 10000 + 40000 + k (test protocol)",
            },
            "parameters": [
                "regge.alpha_exc0",
                "regge.effective_transfer_slope",
                "regge.edge_power",
            ],
            "bounds": {
                "regge.alpha_exc0": [0.0, 1.2],
                "regge.effective_transfer_slope": [5.07, 10.2],
                "regge.edge_power": [0.25, 6.0],
            },
            "binned": {
                "sigma_inel_mb": SIGMA_INEL_MB,
                "window": dict(PAPER_WINDOW),
                "path_root": str(tmp_path),
                "tables": [
                    {
                        "species": "pi_minus",
                        "pdg": -211,
                        "mass_gev": PI_MASS_GEV,
                        "path": "pi_minus.txt",
                        "normalization": "f",
                    },
                    {
                        "species": "pi_plus",
                        "pdg": 211,
                        "mass_gev": PI_MASS_GEV,
                        "path": "pi_plus.txt",
                        "normalization": "f",
                    },
                ],
            },
        },
        "nucleus": {
            "name": "C12",
            "A": 12,
            "Z": 6,
            "profile": "harmonic_oscillator",
            "radius_fm": 1.7,
            "diffuseness_fm": 0.5,
            "quad_alpha": 0.0,
            "sigma_nn_inel_mb": 31.7,
        },
        "pA": {"glauber_dr_fm": 0.1, "max_event_attempts": 100},
    }


def test_hermetic_binned_run_report_payload(tmp_path):
    """The binned runner report carries the M4/M8 payload, deterministically.

    Minimal statistics (4 events per objective evaluation and per
    replica, K = 5 replicas, 1 Powell iteration, 4 events per stage)
    over the two synthetic tables. The run is repeated at the same
    seed: the uncertainty-aware binned objective is deterministic in
    the configuration and the seed, so the reported chi2 and
    parameters must be bit-equal across runs. The M8 pins: the
    mc_replicas protocol block (count, seed offset, replica seed
    base), the per-species MC-sigma summaries, and the convention
    split between the combined-error objective chi2 and the data-only
    metrics chi2.
    """
    config = _hermetic_binned_config(tmp_path)
    report, samples = run_staged_evaluation(config)
    assert report["stage_order"] == [
        "simple_excitation",
        "regge_fit",
        "baryon_nucleus",
    ]
    assert list(samples) == report["stage_order"]

    fit = report["regge_fit"]["fit"]
    assert fit["performed"] is True
    assert fit["objective"] == "binned"
    assert math.isfinite(fit["chi2"])
    assert fit["fit_seed"] == SEED + 10_000  # the runner's fit-seed protocol
    assert fit["events_per_evaluation"] == 4
    mc = fit["mc_replicas"]
    assert mc["count"] == 5
    assert mc["seed_offset"] == 40000
    assert mc["replica_seed_base"] == SEED + 40_000  # config.seed + 40000
    assert mc["events_per_replica"] == 4
    assert fit["max_iterations"] == 1
    assert fit["n_objective_evaluations"] >= 1
    assert len(fit["trace"]) == fit["n_objective_evaluations"]
    assert fit["sqrt_s_gev"] == math.sqrt(Reaction(BEAM_MOMENTUM, 2212, 2212).s())
    assert fit["sigma_inel_mb"] == SIGMA_INEL_MB
    assert fit["window"] == PAPER_WINDOW
    assert fit["sigma_fallback"] == "poisson_data_content"
    assert set(fit["parameters"]) == set(config["fit"]["parameters"])

    tables = fit["tables"]
    assert [t["species"] for t in tables] == ["pi_minus", "pi_plus"]
    assert all(t["n_points"] == 4 for t in tables)
    # The pi+ table carries one zero error: the fallback count is pinned.
    assert [t["n_sigma_fallback"] for t in tables] == [0, 1]
    assert all(t["normalization"] == "f" for t in tables)

    assert set(fit["species"]) == {"pi_minus", "pi_plus"}
    some_mc_sigma_positive = False
    for species, payload in fit["species"].items():
        assert math.isfinite(payload["chi2"])
        assert payload["chi2"] >= 0.0
        for key in (
            f"{species}_chi2",
            f"{species}_chi2_dof",
            f"{species}_smape_percent",
            f"{species}_weighted_rel_rmse",
            "mc_sigma_mean_over_bins",
            "mc_sigma_max_over_bins",
            "mc_sigma_per_replica_mean_over_bins",
        ):
            assert key in payload
            assert math.isfinite(payload[key])
        # M8 MC-sigma summaries: non-negative, max >= mean over bins; at
        # 4-event replicas at least one bin carries non-zero variance.
        assert payload["mc_sigma_mean_over_bins"] >= 0.0
        assert payload["mc_sigma_max_over_bins"] >= payload["mc_sigma_mean_over_bins"]
        assert payload["mc_sigma_per_replica_mean_over_bins"] >= 0.0
        some_mc_sigma_positive |= payload["mc_sigma_max_over_bins"] > 0.0
        # 4 table points against 3 free parameters: dof = max(4 - 3, 1) = 1,
        # so the data-only metrics chi2_dof equals the data-only metrics chi2.
        assert payload[f"{species}_chi2_dof"] == payload[f"{species}_chi2"]
        # Convention split: the objective chi2 uses the combined error
        # (data + MC), the metrics keep the data-only sigma on the same
        # replica-mean prediction, so combined <= data-only per species.
        assert payload["chi2"] <= payload[f"{species}_chi2"]
    assert some_mc_sigma_positive
    # The total chi2 is the sum over the table species, in table order.
    assert fit["chi2"] == (
        fit["species"]["pi_minus"]["chi2"] + fit["species"]["pi_plus"]["chi2"]
    )

    # The fitted p+p parameters are the ones frozen into the p+A stage.
    frozen = report["baryon_nucleus"]["frozen_pp_parameters"]
    for name in config["fit"]["parameters"]:
        group, field = name.split(".", 1)
        assert frozen[group][field] == fit["parameters"][name]
    # The generated samples stay exactly closed in every stage.
    for stage in report["stage_order"]:
        assert report[stage]["status"] == "passed"
        assert report[stage]["summary"]["max_conservation_residual"] < 1e-10

    # Determinism: the same configuration and seed reproduce the fit.
    report2, _ = run_staged_evaluation(config)
    fit2 = report2["regge_fit"]["fit"]
    assert fit2["chi2"] == fit["chi2"]
    assert fit2["parameters"] == fit["parameters"]


# ---------------------------------------------------------------------------
# external_data pin of the real pi- table
# ---------------------------------------------------------------------------


@pytest.mark.external_data
def test_pi_minus_real_table_pins_paper_window_content():
    """The real pi- table pins its paper-window fit content.

    Reads ``spshadrons.web.cern.ch/data/xsect_piminus.txt`` through
    ``BinnedFitTable.from_path`` at the paper window (the default
    window: xF in [0.0, 0.4], pT in [0.05, 1.5]) with the bundled
    normalization declaration ("f", sigma_inel = 31.78 mb). The
    hermetic archive checkouts deselect this test via
    ``-m "not external_data"``.
    """
    table = BinnedFitTable.from_path(
        REAL_PI_MINUS_TABLE,
        species="pi_minus",
        pdg_id=-211,
        mass_gev=PI_MASS_GEV,
        normalization="f",
        sigma_inel_mb=SIGMA_INEL_MB,
    )
    assert table.data.size == 229
    assert table.n_sigma_fallback == 0
    assert np.all(np.isfinite(table.data))
    assert np.all(table.data > 0.0)


# ---------------------------------------------------------------------------
# M11: the one-dimensional profile and degeneracy machinery on the synthetic
# quadratic objective (acceptance criterion 3)
# ---------------------------------------------------------------------------
#
# The M11 production modes hard-require the production pre-registration and
# the production-grid re-runs, so the machinery is pinned on the synthetic
# quadratic objective the hermetic smoke established:
#     chi2(theta) = (theta - a)^T A (theta - a) + c0 + noise(theta; seeds)
# A is a known positive-definite block matrix (a non-trivial correlation
# structure inside each block), a the known anchor, and the noise term is
# seeded by the M11 replica-seed convention (20330832 + k, one fixed draw
# per replica). The objective's Hessian is exactly 2A, so every analytic
# reference is closed form: V = A^{-1}, the 1-sigma profile width
# sqrt(diag(V)), the inv(A) correlation (the factor of two in H = 2A
# cancels in the correlation), the profile crossings at
# d = sqrt(delta * V_ii), and the diagonal-dominant form
# d = sqrt(delta / A_ii). The entry points tested are the documented pure
# functions the smoke drove (m11_refit_nuisance, m11_run_hessian,
# m11_scan_grid, m11_profile_intervals, m11_expected_grids,
# m11_materialize_point_registry, m11_single_evaluation,
# m11_cross_check_preregistration, m11_degeneracy_decisions); none of them
# generate events or read production data.
#
M11_SYNTH_M = np.array([
    [1.0, 0.45, 0.0, 0.0, 0.0, 0.0],
    [0.45, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, -0.3, 0.0, 0.0],
    [0.0, 0.0, -0.3, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.05],
    [0.0, 0.0, 0.0, 0.0, 0.05, 1.0],
])
M11_SYNTH_SCALES = np.array([3.0e3, 1.5e3, 2.0e3, 1.2e3, 2.5e3, 1.8e3])
M11_SYNTH_A = 0.5 * (
    M11_SYNTH_M @ np.diag(M11_SYNTH_SCALES) @ M11_SYNTH_M.T
)
M11_SYNTH_ANCHOR = np.array([0.5, 2.0, 1.5, 2.0, 2.5, 1.2])
M11_SYNTH_C0 = 137.0
M11_SYNTH_PARTNERS = (
    "continuum.strange_lambda_strength",
    "continuum.pair_strength",
)
M11_SYNTH_SCAN_PARAM = "continuum.match_mass"
M11_SYNTH_SCAN_INDEX = se.M11_PROFILE_SET.index(M11_SYNTH_SCAN_PARAM)
M11_SYNTH_V = np.linalg.inv(M11_SYNTH_A)
M11_SYNTH_SIGMA_TRUE = np.sqrt(np.diag(M11_SYNTH_V))
# The analytic correlation: C = (2A)^{-1}, rho_ij = C_ij / sqrt(C_ii C_jj)
# (identical to the inv(A) correlation up to the cancelling factor of two).
M11_SYNTH_C_TRUE = np.linalg.inv(2.0 * M11_SYNTH_A)
M11_SYNTH_RHO_TRUE = M11_SYNTH_C_TRUE / np.sqrt(
    np.outer(np.diag(M11_SYNTH_C_TRUE), np.diag(M11_SYNTH_C_TRUE))
)
np.fill_diagonal(M11_SYNTH_RHO_TRUE, 1.0)
M11_SYNTH_ANCHOR_VALUES = {
    name: float(value)
    for name, value in zip(se.M11_PROFILE_SET, M11_SYNTH_ANCHOR)
}

# The production run record (read-only pins; absent from hermetic archives).
# dev/00runs is the artifact-store symlink visible from the nested tree.
M11_PROD_RUN_DIR = REPO_ROOT.parent / "00runs" / se.M11_RUN_DESCRIPTION
M11_PROD_PREREG_PATH = M11_PROD_RUN_DIR / "pre_registration.json"
M11_PROD_M8_CONFIG = (
    REPO_ROOT.parent
    / "00runs"
    / "staged_binned_fit_mc_2026-09-06"
    / "config_m4_final.json"
)
M11_PROD_M8_RECORD = (
    REPO_ROOT.parent
    / "00runs"
    / "staged_binned_fit_mc_2026-09-06"
    / "staged_evaluation_report.json"
)
M11_PROD_PREREG_SHA256 = (
    "d2e206c9bc5b7cf66ca64fbb6358dd739005585968faeb1deb149dc16788b57a"
)


def _m11_synthetic_objective(
    amplitude: float = 0.0,
    partner: str | None = None,
    partner_scale: float = 2.0e3,
    a55_override: float | None = None,
):
    """The synthetic chi-square (the 6-vector, or the 7-vector when the
    named partner is the extra free coordinate at index 6, anchored at its
    freeze value). ``amplitude`` scales the seeded noise term; a negative
    ``a55_override`` makes the form indefinite (the non-PD path)."""
    A6 = M11_SYNTH_A.copy()
    if a55_override is not None:
        A6[5, 5] = float(a55_override)
    A7 = np.zeros((7, 7), dtype=float)
    A7[:6, :6] = A6
    anchor = np.zeros(7)
    anchor[:6] = M11_SYNTH_ANCHOR
    if partner is not None:
        A7[6, 6] = float(partner_scale)
        anchor[6] = float(se.M11_FREEZE_VALUES[partner])

    def objective(theta):
        t = np.asarray(theta, dtype=float)
        n = t.size
        d = t - anchor[:n]
        base = float(d @ A7[:n, :n] @ d) + M11_SYNTH_C0
        noise = 0.0
        if amplitude > 0.0:
            for k in range(se.M11_K_REPLICAS):
                u = np.random.default_rng(se.M11_REPLICA_SEEDS[k]).random(4)
                basis = np.array([
                    math.sin(3.0 * (t[0] - M11_SYNTH_ANCHOR[0]) + u[0] * 6.283),
                    math.cos(2.0 * (t[4] - M11_SYNTH_ANCHOR[4]) + u[1] * 6.283),
                    math.sin((t[2] - M11_SYNTH_ANCHOR[2]) * (0.5 + u[2])),
                    math.cos((t[5] - M11_SYNTH_ANCHOR[5]) * (0.5 + u[3])),
                ])
                noise += amplitude * float(np.dot(basis, basis))
        return base + noise

    return objective


def _m11_synthetic_evaluator(objective, partner: str | None = None):
    """A production-shaped entry consumer for the synthetic objective:
    the same dict contract ``m11_single_evaluation`` and
    ``m11_refit_nuisance`` consume from the production evaluator."""
    def evaluate(theta):
        core = list(theta)[: len(se.M11_PROFILE_SET)]
        entry = {
            "theta": {
                name: float(value)
                for name, value in zip(se.M11_PROFILE_SET, core)
            },
            "extra_free": {},
            "chi2": float(objective(theta)),
            "per_species": {},
            "mc_sigma_summary": {},
            "max_conservation_residual": 0.0,
            "wall_s": 0.0,
        }
        if partner is not None:
            entry["extra_free"] = {partner: float(theta[6])}
        return entry

    return evaluate


def _m11_synthetic_factory(scales: dict):
    """The 7-vector evaluator factory of the mixed-pair studies (one
    partner scale per partner name)."""
    def factory(extra_free, extra_free_bounds=None, trace_path=None):
        partner = extra_free[0]
        return _m11_synthetic_evaluator(
            _m11_synthetic_objective(partner=partner, partner_scale=scales[partner]),
            partner=partner,
        )

    return factory


def _m11_synthetic_profile_scan(objective, grid):
    """The one-parameter 1-D profile at the pre-registered anchor rule.

    Every non-anchor grid point is a bounded-Powell nuisance refit (the
    production point convention, ``m11_refit_nuisance``); the anchor scan
    point records chi2_0 without a refit. Returns
    (chi2_values, chi2_0, n_refits).
    """
    param = M11_SYNTH_SCAN_PARAM
    anchor_v = M11_SYNTH_ANCHOR_VALUES[param]
    chi2_0 = float(objective(list(M11_SYNTH_ANCHOR)))
    evaluate = _m11_synthetic_evaluator(objective)
    chi2s = []
    n_refits = 0
    for value in grid:
        if se.m11_theta_within_tolerance(value, anchor_v):
            chi2s.append(chi2_0)
            continue
        res = se.m11_refit_nuisance(
            evaluate,
            fixed_values={param: float(value)},
            theta_start=dict(M11_SYNTH_ANCHOR_VALUES),
            bounds={n: tuple(se.M11_BOUNDS[n]) for n in se.M11_PROFILE_SET},
        )
        assert res["determinism_consistent"]
        chi2s.append(res["chi2"])
        n_refits += 1
    return chi2s, chi2_0, n_refits


def _m11_synthetic_prereg(config_bundled: dict, config_m8: dict) -> dict:
    """A synthetic pre-registration built from the harness constants.

    Mirrors the production pre-registration structure so the
    cross-check runs on the documented protocol without the production
    file: every constant is taken from the module at test time, never
    re-typed (a builder regression is therefore caught by the untampered
    acceptance test together with the tamper tests).
    """
    return {
        "card": se.M11_CARD,
        "milestone": se.M11_MILESTONE,
        "phase": se.M11_PHASE,
        "run_description": se.M11_RUN_DESCRIPTION,
        "machine_constants": {
            "k_replicas": se.M11_K_REPLICAS,
            "events_per_replica_full": se.M11_EVENTS_PER_REPLICA_FULL,
            "xtol": se.M11_XTOL,
            "ftol": se.M11_FTOL,
            "max_iterations": se.M11_MAX_ITERATIONS,
            "degrees_of_freedom": se.M11_DOF,
            "n_bins_total": se.M11_N_BINS_TOTAL,
            "grid_points_per_scan": se.M11_GRID_N,
            "grid_half_width_sigmas": se.M11_GRID_HALF_WIDTH_SIGMAS,
            "grid_fallback_window_fraction": se.M11_GRID_FALLBACK_WINDOW_FRACTION,
            "anchor_scan_point_tolerance": se.M11_ANCHOR_SCAN_TOLERANCE,
            "delta_chi2": dict(se.M11_DELTA_CHI2),
            "hessian_accept_window": list(se.M11_HESSIAN_ACCEPT),
            "hessian_max_adaptations": se.M11_HESSIAN_MAX_ADAPTATIONS,
            "hessian_adapt_factor": se.M11_HESSIAN_ADAPT_FACTOR,
            "hessian_fallback_window_fraction": se.M11_HESSIAN_FALLBACK_WINDOW_FRACTION,
            "hessian_symmetrize": True,
            "degeneracy_rho_threshold": se.M11_DEGENERACY_RHO,
            "flat_profile_coverage_fraction": se.M11_FLAT_PROFILE_COVERAGE,
            "near_bound_window_fraction": se.M11_NEAR_BOUND_WINDOW_FRACTION,
            "elongation_threshold": se.M11_ELONGATION_THRESHOLD,
            "full_grid_points_per_axis": se.M11_FULL_GRID_N,
            "statistics_menu_events_per_replica": list(se.M11_STATISTICS_MENU),
            "per_point_wall_budget_s": se.M11_PER_POINT_WALL_BUDGET_S,
            "cap_rule": {
                "floor": se.M11_CAP_FLOOR,
                "ceiling": se.M11_CAP_CEILING,
                "multiple_of_probe": se.M11_CAP_MULTIPLE,
            },
            "anchor_cap_rule": {
                "floor": se.M11_ANCHOR_CAP_FLOOR,
                "ceiling": se.M11_ANCHOR_CAP_CEILING,
                "multiple_of_m8": se.M11_ANCHOR_CAP_MULTIPLE,
                "m8_evaluations": se.M11_M8_EVALUATIONS,
            },
            "seed": {
                "config_seed": se.M11_CONFIG_SEED,
                "m11_seed_base": se.M11_SEED_BASE,
                "replica_seeds": list(se.M11_REPLICA_SEEDS),
                "stage_label": se.M11_STAGE_LABEL,
            },
            "window": dict(se.M11_WINDOW),
            "sigma_inel_mb": se.M11_SIGMA_INEL_MB,
            "sqrt_s_gev": se.M11_SQRT_S_GEV,
            "beam_momentum_gev_c": se.M11_BEAM_MOMENTUM,
            "m8_parameter_set": dict(se.M11_M8_PARAMETER_SET),
            "profile_bounds": {
                name: list(b) for name, b in se.M11_BOUNDS.items()
            },
            "hessian_step_ladders": {
                k: list(v) for k, v in se.M11_HESSIAN_LADDERS.items()
            },
            "mixed_pair_ladders": {
                k: list(v) for k, v in se.M11_MIXED_PAIR_LADDERS.items()
            },
            "freeze_bounds": {
                name: list(b) for name, b in se.M11_FREEZE_BOUNDS.items()
            },
            "freeze_values": dict(se.M11_FREEZE_VALUES),
            "profile_set_order": list(se.M11_PROFILE_SET),
            "twod_pairs": [
                {"id": "pair0", "members": list(se.M11_FULL_GRID_PAIRS["pair0"]),
                 "mode": "full_grid"},
                {"id": "pair1", "members": list(se.M11_MIXED_PAIRS["pair1"]),
                 "mode": "equivalent_sensitivity_study"},
                {"id": "pair2", "members": list(se.M11_MIXED_PAIRS["pair2"]),
                 "mode": "equivalent_sensitivity_study"},
                {"id": "pair3", "members": list(se.M11_FULL_GRID_PAIRS["pair3"]),
                 "mode": "full_grid"},
            ],
        },
        "profile_set": {
            "parameters": {
                name: {
                    "bounds": list(se.M11_BOUNDS[name]),
                    "hessian_step_ladder": list(se.M11_HESSIAN_LADDERS[name]),
                }
                for name in se.M11_PROFILE_SET
            }
        },
        "provenance": {
            "config_bundled_canonical_sha256": se._canonical_config_hash(
                config_bundled
            ),
            "config_m8_window_canonical_sha256": se._canonical_config_hash(
                config_m8
            ),
            "outer_head_at_preregistration": se.M11_OUTER_HEAD,
            "nested_head_at_preregistration": se.M11_NESTED_HEAD,
            "m8_record_sha256": se.M11_M8_RECORD_SHA256,
        },
    }


def test_m11_preregistration_cross_check_accepts_the_harness_constants():
    """The untampered synthetic pre-registration passes the cross-check.

    Guards the builder against silent drift: every constant is read from
    the module, so an acceptance pass is a non-vacuous pin of the
    documented protocol shape.
    """
    prereg = _m11_synthetic_prereg({"bundled": True}, {"m8": True})
    # Sanity: the builder reflects the harness constants it claims.
    assert prereg["machine_constants"]["seed"]["replica_seeds"] == list(
        se.M11_REPLICA_SEEDS
    )
    assert prereg["machine_constants"]["k_replicas"] == se.M11_K_REPLICAS
    # The cross-check runs clean (no SystemExit).
    se.m11_cross_check_preregistration(
        prereg, config_bundled={"bundled": True}, config_m8={"m8": True},
    )


@pytest.mark.parametrize(
    "field_path, tampered",
    [
        (("machine_constants", "k_replicas"), 7),
        (("machine_constants", "delta_chi2"), {"sigma1": 2.0, "sigma2": 4.0}),
        (("machine_constants", "seed", "m11_seed_base"), 20330832),
        (
            ("machine_constants", "profile_bounds", "continuum.match_mass"),
            [1.0, 4.0],
        ),
        (
            ("machine_constants", "hessian_step_ladders", "regge.alpha_exc0"),
            [1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
        ),
        (
            ("profile_set", "parameters", "continuum.match_mass", "bounds"),
            [1.0, 4.0],
        ),
        (("provenance", "outer_head_at_preregistration"), "0" * 40),
    ],
)
def test_m11_preregistration_tampered_constant_refused(field_path, tampered):
    """Any tampered pre-registered constant is refused (the negative
    control): the pre-registered protocol is authoritative."""
    prereg = _m11_synthetic_prereg({"bundled": True}, {"m8": True})
    node = prereg
    for key in field_path[:-1]:
        node = node[key]
    node[field_path[-1]] = tampered
    with pytest.raises(SystemExit, match="refusing to run"):
        se.m11_cross_check_preregistration(
            prereg, config_bundled={"bundled": True}, config_m8={"m8": True},
        )


def test_m11_preregistration_missing_machine_constants_block_refused():
    prereg = _m11_synthetic_prereg({"bundled": True}, {"m8": True})
    del prereg["machine_constants"]
    with pytest.raises(SystemExit, match="machine_constants block missing"):
        se.m11_cross_check_preregistration(
            prereg, config_bundled={"bundled": True}, config_m8={"m8": True},
        )


def test_m11_preregistration_missing_file_refused(tmp_path):
    """The protocol must be pre-registered before any profile run."""
    with pytest.raises(SystemExit, match="pre-registration not found"):
        se.m11_load_preregistration(tmp_path / "pre_registration.json")


@pytest.mark.skipif(
    not M11_PROD_PREREG_PATH.exists(),
    reason="the production run record is not present in this checkout",
)
def test_m11_production_preregistration_is_the_harness_protocol_of_record():
    """The committed production pre-registration pins the harness
    protocol (read-only record pins; the hermetic synthetic tests above
    pin the same code paths without the production file)."""
    assert se._m11_sha256_file(M11_PROD_PREREG_PATH) == M11_PROD_PREREG_SHA256
    prereg = se.m11_load_preregistration(M11_PROD_PREREG_PATH)
    # The cross-check runs clean against the committed configuration files.
    se.m11_cross_check_preregistration(
        prereg,
        config_bundled=json.loads(
            BUNDLED_CONFIG_PATH.read_text(encoding="utf-8")
        ),
        config_m8=json.loads(
            M11_PROD_M8_CONFIG.read_text(encoding="utf-8")
        ),
        m8_record_path=M11_PROD_M8_RECORD,
    )
    mc = prereg["machine_constants"]
    assert mc["k_replicas"] == 8
    assert mc["events_per_replica_full"] == 2000
    assert mc["seed"]["m11_seed_base"] == 20330831
    assert mc["seed"]["replica_seeds"] == [20330832 + k for k in range(8)]
    anchor = json.loads(
        (M11_PROD_RUN_DIR / "anchor.json").read_text(encoding="utf-8")
    )
    assert anchor["chi2_0"] == 47298.6113896293
    assert anchor["determinism_consistent"] is True
    assert anchor["cap_hit"] is False
    assert anchor["n_evaluations"] == 202
    hess = json.loads(
        (M11_PROD_RUN_DIR / "hessian.json").read_text(encoding="utf-8")
    )
    assert hess["positive_definite"] is False
    assert hess["used_pinv"] is True
    assert min(hess["eigenvalues"]) == -162758.89186516113
    # The record holds the pre-registered 156-point grid (60 1-D + 96 2-D).
    assert len(list(M11_PROD_RUN_DIR.glob("point_*.json"))) == 156


def test_m11_single_evaluation_reports_the_anchor_chi2_exactly():
    """The logged single evaluation at the anchor is exact: at the anchor
    the quadratic form contributes zero and the entry carries the
    production-shaped payload."""
    obj = _m11_synthetic_objective()
    entry = se.m11_single_evaluation(
        _m11_synthetic_evaluator(obj), list(M11_SYNTH_ANCHOR)
    )
    assert entry["chi2"] == M11_SYNTH_C0
    assert entry["theta"] == dict(
        zip(se.M11_PROFILE_SET, [float(v) for v in M11_SYNTH_ANCHOR])
    )
    assert entry["extra_free"] == {}
    assert entry["max_conservation_residual"] == 0.0
    assert entry["per_species"] == {}
    assert entry["mc_sigma_summary"] == {}
    assert entry["wall_s"] == 0.0


def test_m11_profile_scan_is_reproducible_at_the_fixed_replica_seeds():
    """The 1-D profile chi-square is bit-reproducible at the fixed M11
    replica seeds, and the seeded scan stays within the bounded noise
    band of the noiseless one."""
    sigma4 = float(M11_SYNTH_SIGMA_TRUE[M11_SYNTH_SCAN_INDEX])
    a4 = M11_SYNTH_ANCHOR_VALUES[M11_SYNTH_SCAN_PARAM]
    grid, fallback = se.m11_scan_grid(
        M11_SYNTH_SCAN_PARAM, a4, sigma4
    )
    assert fallback is False
    assert len(grid) == se.M11_GRID_N
    assert grid[0] == pytest.approx(a4 - 4.0 * sigma4, abs=1e-12)
    assert grid[-1] == pytest.approx(a4 + 4.0 * sigma4, abs=1e-12)

    chi2s_nl, chi2_0, n_refits = _m11_synthetic_profile_scan(
        _m11_synthetic_objective(), grid
    )
    # The anchor rule: exactly one grid point (the center) is recorded as
    # chi2_0 without a refit.
    assert n_refits == se.M11_GRID_N - 1
    anchor_idx = next(
        k for k, v in enumerate(grid)
        if se.m11_theta_within_tolerance(v, a4)
    )
    assert chi2s_nl[anchor_idx] == chi2_0

    chi2s_s1, _, _ = _m11_synthetic_profile_scan(
        _m11_synthetic_objective(amplitude=1.0e-6), grid
    )
    chi2s_s2, _, _ = _m11_synthetic_profile_scan(
        _m11_synthetic_objective(amplitude=1.0e-6), grid
    )
    # Bit-identical across repeated runs at the fixed replica seeds.
    assert chi2s_s1 == chi2s_s2
    # The noise is a sum of squares seeded once per replica: non-negative
    # and bounded by amplitude * 4 basis terms * 8 replicas.
    band = 1.0e-6 * 4.0 * se.M11_K_REPLICAS
    for seeded, clean in zip(chi2s_s1, chi2s_nl):
        assert -1e-9 <= seeded - clean <= band


def test_m11_profile_intervals_recover_the_analytic_sigma_edges():
    """The 1-sigma/2-sigma intervals recover the analytic profile
    crossings d = sqrt(delta * V_ii) on the near-diagonal-dominant
    coordinate, read parabolically (exact on the quadratic)."""
    sigma4 = float(M11_SYNTH_SIGMA_TRUE[M11_SYNTH_SCAN_INDEX])
    a4 = M11_SYNTH_ANCHOR_VALUES[M11_SYNTH_SCAN_PARAM]
    grid, _ = se.m11_scan_grid(M11_SYNTH_SCAN_PARAM, a4, sigma4)
    chi2s, chi2_0, _ = _m11_synthetic_profile_scan(_m11_synthetic_objective(), grid)

    # Profile fidelity: the refit profile equals the Gaussian profile
    # c0 + (v - a4)^2 / V44.
    V44 = float(M11_SYNTH_V[M11_SYNTH_SCAN_INDEX, M11_SYNTH_SCAN_INDEX])
    analytic = [M11_SYNTH_C0 + (v - a4) ** 2 / V44 for v in grid]
    max_dev = max(abs(c - a) for c, a in zip(chi2s, analytic))
    assert max_dev < 1e-6
    # The diagonal-dominant form of the task: the analytic width matches
    # sqrt(delta / A_ii) up to the residual cross-coupling (here < 1%).
    A44 = float(M11_SYNTH_A[M11_SYNTH_SCAN_INDEX, M11_SYNTH_SCAN_INDEX])
    assert sigma4 == pytest.approx(1.0 / math.sqrt(A44), rel=0.01)

    iv = se.m11_profile_intervals(grid, chi2s, theta0=a4, chi2_0=chi2_0)
    for key, delta in se.M11_DELTA_CHI2.items():
        half = math.sqrt(float(delta) * V44)
        block = iv[key]
        assert block["level_chi2"] == pytest.approx(
            M11_SYNTH_C0 + float(delta), abs=1e-9
        )
        assert block["lower"] == pytest.approx(a4 - half, rel=1e-9, abs=1e-12)
        assert block["upper"] == pytest.approx(a4 + half, rel=1e-9, abs=1e-12)
        assert block["at_edge_lower"] is False
        assert block["at_edge_upper"] is False
        assert block["one_sided"] is False
        assert block["method_lower"] == "parabolic"
        assert block["method_upper"] == "parabolic"


def test_m11_hessian_correlation_matches_the_analytic_inverse(tmp_path):
    """The Hessian machinery reproduces the analytic 2A, the inv(A)
    correlation, and the 1/sqrt-convention sigma = sqrt(diag(A^{-1}))
    on the positive-definite synthetic form."""
    hess = se.m11_run_hessian(
        _m11_synthetic_evaluator(_m11_synthetic_objective()),
        anchor_values=dict(M11_SYNTH_ANCHOR_VALUES),
        chi2_0=M11_SYNTH_C0,
        evaluator_factory=_m11_synthetic_factory(
            {name: 2.0e3 for name in M11_SYNTH_PARTNERS}
        ),
        out_dir=tmp_path,
    )
    assert hess["positive_definite"] is True
    assert hess["used_pinv"] is False
    assert min(hess["eigenvalues"]) > 0.0

    # The diagonal rungs and the 15 mixed-pair curvatures reconstruct the
    # analytic Hessian 2A.
    H = np.asarray(hess["H"], dtype=float)
    assert np.allclose(H, 2.0 * M11_SYNTH_A, rtol=1e-9, atol=1e-6)
    for i, name in enumerate(se.M11_PROFILE_SET):
        rung = hess["rungs"][name]
        assert rung["flags"] == []
        assert rung["accepted_h"] is not None
        assert rung["H_ii"] == pytest.approx(
            2.0 * float(M11_SYNTH_A[i, i]), rel=1e-6
        )
    assert hess["rungs"]["regge.alpha_exc0|regge.edge_power"]["H_ij"] == pytest.approx(
        2.0 * float(M11_SYNTH_A[0, 1]), rel=1e-6
    )
    assert hess["rungs"]["continuum.strange_threshold_power|"
                       "continuum.pair_threshold_power"]["H_ij"] == pytest.approx(
        2.0 * float(M11_SYNTH_A[2, 3]), rel=1e-6
    )
    assert hess["rungs"]["regge.alpha_exc0|continuum.match_mass"]["H_ij"] == pytest.approx(
        0.0, abs=1e-6
    )

    # The correlation matrix is the analytic inv(A) correlation (the factor
    # of two in H = 2A cancels in rho), and sigma = sqrt(diag(A^{-1})).
    rho = np.asarray(hess["rho"], dtype=float)
    assert float(np.max(np.abs(rho - M11_SYNTH_RHO_TRUE))) < 1e-9
    assert np.allclose(np.diag(rho), 1.0, atol=0.0)
    for i, name in enumerate(se.M11_PROFILE_SET):
        assert hess["sigma_per_parameter"][name] == pytest.approx(
            float(M11_SYNTH_SIGMA_TRUE[i]), rel=1e-9
        )
    assert hess["eigenvalues"] == pytest.approx(
        [float(e) for e in np.linalg.eigvalsh(2.0 * M11_SYNTH_A)],
        rel=1e-9,
    )

    # The targeted mixed-pair studies: on the decoupled partner the 2x2
    # block is diagonal, so rho_pair = 0 and the elongation is the analytic
    # eigenvalue ratio sqrt(2*s_partner / (2*A_ii)) (below the threshold).
    for pair_id, inset_name in (
        ("pair1", "continuum.strange_threshold_power"),
        ("pair2", "continuum.pair_threshold_power"),
    ):
        res = hess["mixed_pair_results"][pair_id]
        i_in = se.M11_PROFILE_SET.index(inset_name)
        H2 = np.array([
            [2.0 * float(M11_SYNTH_A[i_in, i_in]), 0.0],
            [0.0, 2.0 * 2.0e3],
        ])
        eigs = np.linalg.eigvalsh(H2)
        expected = math.sqrt(float(eigs.max()) / float(eigs.min()))
        assert res["H_mixed"] == pytest.approx(0.0, abs=1e-6)
        assert res["rho_pair"] == pytest.approx(0.0, abs=1e-8)
        assert res["elongation_ratio"] == pytest.approx(expected, rel=1e-9)
        assert res["elongation_ratio"] < se.M11_ELONGATION_THRESHOLD
        assert len(res["eigenvalues"]) == 2
        assert res["eigenvalues"][0] == pytest.approx(float(eigs[0]), rel=1e-9)
        assert res["eigenvalues"][1] == pytest.approx(float(eigs[1]), rel=1e-9)

    # The hessian.json artifact is written to the run directory.
    artifact = json.loads((tmp_path / "hessian.json").read_text(encoding="utf-8"))
    assert artifact["sigma_per_parameter"] == hess["sigma_per_parameter"]


def test_m11_correlation_symmetrizes_a_nonsymmetric_input():
    """The correlation step symmetrizes before inverting: a nonsymmetric
    Hessian input gives the same rho as its own symmetrization, and the
    symmetric case reproduces the analytic 2x2 correlation."""
    H_asym = np.array([[2.0, 0.7], [1.3, 3.0]])
    r_asym = se.m11_correlation_from_hessian(H_asym)
    r_sym = se.m11_correlation_from_hessian(0.5 * (H_asym + H_asym.T))
    assert r_asym["positive_definite"] is True
    assert r_asym["used_pinv"] is False
    assert np.array_equal(r_asym["rho"], r_sym["rho"])
    # Analytic: C = inv([[2, 1], [1, 3]]) = (1/5) [[3, -1], [-1, 2]].
    expected_off = -0.2 / math.sqrt(0.6 * 0.4)
    assert r_sym["rho"][0, 1] == pytest.approx(expected_off, abs=1e-9)
    assert r_sym["rho"][1, 0] == pytest.approx(expected_off, abs=1e-9)


def test_m11_negative_curvature_hessian_takes_the_pinv_fallback(tmp_path):
    """A negative-eigenvalue Hessian takes the documented fallback path:
    the rung reports no_positive_curvature, the row is zeroed, the
    pseudo-inverse is flagged, the sigma is unavailable, and the grid
    falls back to the pre-registered 0.1 window. No crash."""
    pms = "continuum.pion_multiplicity_slope"
    i5 = se.M11_PROFILE_SET.index(pms)
    hess = se.m11_run_hessian(
        _m11_synthetic_evaluator(_m11_synthetic_objective(a55_override=-1.8e3)),
        anchor_values=dict(M11_SYNTH_ANCHOR_VALUES),
        chi2_0=M11_SYNTH_C0,
        evaluator_factory=_m11_synthetic_factory(
            {name: 2.0e3 for name in M11_SYNTH_PARTNERS}
        ),
        out_dir=tmp_path,
    )
    assert hess["positive_definite"] is False
    assert hess["used_pinv"] is True
    assert min(hess["eigenvalues"]) <= 0.0

    # The negative-curvature coordinate: rung refused, sigma unavailable.
    rung = hess["rungs"][pms]
    assert rung["accepted_h"] is None
    assert rung["H_ii"] is None
    assert "no_positive_curvature" in rung["flags"]
    assert hess["sigma_per_parameter"][pms] is None
    assert se._m11_sigma_unavailable(pms, hess) is True

    # The zeroed row: all correlations with the coordinate are 0, the
    # diagonal stays 1 (the degenerate-coordinate convention).
    rho = np.asarray(hess["rho"], dtype=float)
    assert np.allclose(rho[i5, :], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], atol=1e-9)
    # The untouched blocks 0-1 and 2-3 keep the analytic correlations.
    for i in range(4):
        for j in range(4):
            if i != j:
                assert rho[i, j] == pytest.approx(
                    float(M11_SYNTH_RHO_TRUE[i, j]), abs=1e-9
                )
    # The sigmas of the untouched block coordinates match the analytic.
    for i in range(4):
        name = se.M11_PROFILE_SET[i]
        assert hess["sigma_per_parameter"][name] == pytest.approx(
            float(M11_SYNTH_SIGMA_TRUE[i]), rel=1e-9
        )
    # The decoupled 4-5 block: the pinv gives the coordinate-degenerate
    # width 1/sqrt(A_44) (the coupling to the zeroed row drops out).
    assert hess["sigma_per_parameter"]["continuum.match_mass"] == pytest.approx(
        1.0 / math.sqrt(float(M11_SYNTH_A[4, 4])), rel=1e-9
    )

    # Every mixed-pair combination involving the coordinate is recorded as
    # skipped (the diagonal rung is unavailable).
    skipped = sorted(
        key for key, value in hess["rungs"].items()
        if "skipped" in value
    )
    # Every combination whose diagonal rung is unavailable is skipped;
    # pms is the last profile-set member, so all combinations pair it
    # with an earlier member in profile-set order.
    assert skipped == sorted(
        f"{se.M11_PROFILE_SET[i]}|{pms}"
        for i in range(len(se.M11_PROFILE_SET) - 1)
    )
    # The pre-registered fallback grid: the 0.1 window around the anchor.
    grids = se.m11_expected_grids(dict(M11_SYNTH_ANCHOR_VALUES), hess)
    assert grids["fallbacks_1d"][pms] is True
    lo, hi = se.M11_BOUNDS[pms]
    anchor_v = M11_SYNTH_ANCHOR_VALUES[pms]
    width = se.M11_GRID_FALLBACK_WINDOW_FRACTION * (hi - lo)
    g = grids["grids_1d"][pms]
    assert len(g) == se.M11_GRID_N
    assert g[0] == pytest.approx(anchor_v - width, abs=1e-12)
    assert g[-1] == pytest.approx(anchor_v + width, abs=1e-12)
    # The mixed-pair studies still run on the partner-freed objective.
    for pair_id in ("pair1", "pair2"):
        assert "skipped" not in hess["mixed_pair_results"][pair_id]


def test_m11_degeneracy_decisions_apply_the_spec_mapping():
    """The mechanical degeneracy decision mapping (spec section 6.3) on a
    hand-built hessian payload: retained / grouped / constrained
    externally / removed, with the hessian and the mixed-pair rho
    estimators."""
    rho = np.zeros((6, 6))
    rho[0, 1] = rho[1, 0] = -0.95
    hess = {"rho": rho, "mixed_pair_results": {}}

    # Near-bound anchor + informative interval: constrained externally.
    anchor = dict(M11_SYNTH_ANCHOR_VALUES)
    anchor["regge.alpha_exc0"] = 0.05  # within 10% of the lower bound
    tables = {
        "regge.alpha_exc0": {
            "intervals": {
                "sigma2": {"lower": 0.1, "upper": 0.5, "one_sided": False}
            }
        },
        "regge.edge_power": {
            "intervals": {
                "sigma2": {"lower": 1.5, "upper": 2.5, "one_sided": False}
            }
        },
    }
    res = se.m11_degeneracy_decisions(anchor, hess, tables)
    alpha = res["profile_set"]["regge.alpha_exc0"]
    assert alpha["decision"] == "constrained externally"
    assert alpha["near_bound"] is True
    assert alpha["rho_max"] == pytest.approx(0.95, abs=1e-12)
    assert alpha["degenerate_partner"] == "regge.edge_power"
    assert alpha["rho_source"] == "hessian"
    edge = res["profile_set"]["regge.edge_power"]
    assert edge["decision"] == "grouped"
    assert edge["near_bound"] is False
    assert edge["rho_max"] == pytest.approx(0.95, abs=1e-12)
    match = res["profile_set"]["continuum.match_mass"]
    assert match["decision"] == "retained as-is"
    assert match["rho_max"] == pytest.approx(0.0, abs=1e-12)
    for name in se.M11_FREEZE_VALUES:
        assert name in res["freeze_set"]
        assert res["freeze_set"][name]["m8_value"] == se.M11_FREEZE_VALUES[name]

    # A flat (wide) 2-sigma interval at or above the rho threshold:
    # removed (the data identify nothing inside the window).
    tables2 = {
        "regge.alpha_exc0": {
            "intervals": {
                "sigma2": {"lower": 0.05, "upper": 1.15, "one_sided": False}
            }
        },
        "regge.edge_power": {
            "intervals": {
                "sigma2": {"lower": 1.5, "upper": 2.5, "one_sided": False}
            }
        },
    }
    res2 = se.m11_degeneracy_decisions(
        dict(M11_SYNTH_ANCHOR_VALUES), hess, tables2
    )
    removed = res2["profile_set"]["regge.alpha_exc0"]
    assert removed["decision"] == "removed"
    assert removed["coverage_fraction_sigma2"] == pytest.approx(
        1.10 / 1.2, abs=1e-9
    )

    # The mixed-pair rho is the estimator of record for the inset.
    hess3 = {
        "rho": np.zeros((6, 6)),
        "mixed_pair_results": {
            "pair1": {
                "members": list(se.M11_MIXED_PAIRS["pair1"]),
                "rho_pair": 0.97,
                "elongation_ratio": 12.0,
            }
        },
    }
    tables3 = {
        "continuum.strange_threshold_power": {
            "intervals": {
                "sigma2": {"lower": 0.5, "upper": 2.5, "one_sided": False}
            }
        },
    }
    res3 = se.m11_degeneracy_decisions(
        dict(M11_SYNTH_ANCHOR_VALUES), hess3, tables3
    )
    inset = res3["profile_set"]["continuum.strange_threshold_power"]
    assert inset["decision"] == "grouped"
    assert inset["rho_max"] == pytest.approx(0.97, abs=1e-12)
    assert inset["rho_source"] == "mixed_pair_targeted"
    assert inset["degenerate_partner"] == "continuum.strange_lambda_strength"


def test_m11_capped_refit_reports_the_cap_and_best_of_trace(tmp_path):
    """The capped nuisance refit reports the cap honestly: the capped
    evaluation count, the best-of-trace chi2 (the trace minimum, not the
    last evaluation), the bit-identical final re-evaluation, and the held
    fixed coordinate. The uncapped descent reaches the analytic
    conditional minimum, and the capped descent never claims a better
    value."""
    param = M11_SYNTH_SCAN_PARAM
    fixed_val = M11_SYNTH_ANCHOR_VALUES[param] + 0.5
    bounds = {n: tuple(se.M11_BOUNDS[n]) for n in se.M11_PROFILE_SET}
    trace = tmp_path / "trace.jsonl"
    res = se.m11_refit_nuisance(
        _m11_synthetic_evaluator(_m11_synthetic_objective()),
        fixed_values={param: fixed_val},
        theta_start=dict(M11_SYNTH_ANCHOR_VALUES),
        bounds=bounds,
        cap_evaluations=25,
        trace_path=trace,
    )
    assert res["n_evaluations"] == 25
    assert res["cap_hit"] is True
    assert res["determinism_consistent"] is True
    assert res["chi2"] == res["final_check_chi2"]
    lines = [json.loads(line) for line in trace.read_text().splitlines()]
    assert len(lines) == 25
    assert res["chi2"] == min(r["chi2"] for r in lines)
    assert res["best_theta"][param] == fixed_val
    assert res["free_names"] == [
        n for n in se.M11_PROFILE_SET if n != param
    ]

    # The uncapped descent reaches the analytic conditional minimum
    # c0 + d^2 / V44; the capped descent (fewer evaluations, the same
    # trajectory prefix) cannot beat it.
    V44 = float(M11_SYNTH_V[M11_SYNTH_SCAN_INDEX, M11_SYNTH_SCAN_INDEX])
    expected = M11_SYNTH_C0 + (0.5 ** 2) / V44
    res_free = se.m11_refit_nuisance(
        _m11_synthetic_evaluator(_m11_synthetic_objective()),
        fixed_values={param: fixed_val},
        theta_start=dict(M11_SYNTH_ANCHOR_VALUES),
        bounds=bounds,
        cap_evaluations=400,
    )
    assert res_free["cap_hit"] is False
    assert res_free["n_evaluations"] > 25
    assert res_free["chi2"] == pytest.approx(expected, abs=1e-6)
    assert res["chi2"] >= res_free["chi2"]

    # A repeat capped run is bit-identical.
    res2 = se.m11_refit_nuisance(
        _m11_synthetic_evaluator(_m11_synthetic_objective()),
        fixed_values={param: fixed_val},
        theta_start=dict(M11_SYNTH_ANCHOR_VALUES),
        bounds=bounds,
        cap_evaluations=25,
    )
    assert res2["chi2"] == res["chi2"]
    assert res2["n_evaluations"] == res["n_evaluations"]


def test_m11_elongation_crosses_the_pre_registered_threshold(tmp_path):
    """The elongation computation on an anisotropic 2x2 block: the
    partner-heavy block crosses the pre-registered elongation threshold
    at the analytic eigenvalue ratio; the partner-balanced block stays
    below it."""
    big_partner = M11_SYNTH_PARTNERS[0]
    scales = {
        big_partner: 2.5e5,
        M11_SYNTH_PARTNERS[1]: 2.0e3,
    }
    hess = se.m11_run_hessian(
        _m11_synthetic_evaluator(_m11_synthetic_objective()),
        anchor_values=dict(M11_SYNTH_ANCHOR_VALUES),
        chi2_0=M11_SYNTH_C0,
        evaluator_factory=_m11_synthetic_factory(scales),
        out_dir=tmp_path,
    )
    res1 = hess["mixed_pair_results"]["pair1"]
    res2 = hess["mixed_pair_results"]["pair2"]
    i1 = se.M11_PROFILE_SET.index("continuum.strange_threshold_power")
    i2 = se.M11_PROFILE_SET.index("continuum.pair_threshold_power")
    eigs1 = np.linalg.eigvalsh(np.array([
        [2.0 * float(M11_SYNTH_A[i1, i1]), 0.0],
        [0.0, 2.0 * scales[big_partner]],
    ]))
    expected1 = math.sqrt(float(eigs1.max()) / float(eigs1.min()))
    assert res1["elongation_ratio"] == pytest.approx(expected1, rel=1e-9)
    assert res1["elongation_ratio"] >= se.M11_ELONGATION_THRESHOLD
    eigs2 = np.linalg.eigvalsh(np.array([
        [2.0 * float(M11_SYNTH_A[i2, i2]), 0.0],
        [0.0, 2.0 * scales[M11_SYNTH_PARTNERS[1]]],
    ]))
    expected2 = math.sqrt(float(eigs2.max()) / float(eigs2.min()))
    assert res2["elongation_ratio"] == pytest.approx(expected2, rel=1e-9)
    assert res2["elongation_ratio"] < se.M11_ELONGATION_THRESHOLD
    for res in (res1, res2):
        assert res["H_mixed"] == pytest.approx(0.0, abs=1e-6)
        assert res["rho_pair"] == pytest.approx(0.0, abs=1e-8)


def test_m11_point_registry_materializes_the_pre_registered_point_rule():
    """The point registry materializes the pre-registered rules: the
    production-record 60 + 96 = 156 point count, the anchor scan-point
    rule, the 0.1-window fallback, the exact id scheme, and the grid
    reproducibility (the recorded grid values equal
    ``m11_expected_grids`` exactly)."""
    sigmas = {
        name: float(M11_SYNTH_SIGMA_TRUE[i])
        for i, name in enumerate(se.M11_PROFILE_SET)
    }
    record = {
        "sigma_per_parameter": {
            **sigmas,
            "regge.alpha_exc0": None,  # the sigma is unavailable
        },
        "rungs": {},
    }
    expected = se.m11_expected_grids(dict(M11_SYNTH_ANCHOR_VALUES), record)
    # The unavailable sigma takes the 0.1-window fallback.
    assert expected["fallbacks_1d"]["regge.alpha_exc0"] is True
    g_alpha = expected["grids_1d"]["regge.alpha_exc0"]
    lo, hi = se.M11_BOUNDS["regge.alpha_exc0"]
    width = se.M11_GRID_FALLBACK_WINDOW_FRACTION * (hi - lo)
    a0 = M11_SYNTH_ANCHOR_VALUES["regge.alpha_exc0"]
    assert g_alpha[0] == pytest.approx(a0 - width, abs=1e-12)
    assert g_alpha[-1] == pytest.approx(a0 + width, abs=1e-12)
    # The available sigmas take the 4-sigma half-width.
    a4 = M11_SYNTH_ANCHOR_VALUES[M11_SYNTH_SCAN_PARAM]
    s4 = float(M11_SYNTH_SIGMA_TRUE[M11_SYNTH_SCAN_INDEX])
    g4 = expected["grids_1d"][M11_SYNTH_SCAN_PARAM]
    assert not expected["fallbacks_1d"][M11_SYNTH_SCAN_PARAM]
    assert g4[0] == pytest.approx(a4 - 4.0 * s4, abs=1e-12)
    assert g4[-1] == pytest.approx(a4 + 4.0 * s4, abs=1e-12)

    points = se.m11_materialize_point_registry(
        dict(M11_SYNTH_ANCHOR_VALUES),
        record,
        cap_evaluations=se.M11_CAP_CEILING,
        events_per_replica=se.M11_EVENTS_PER_REPLICA_FULL,
    )
    p1d = [p for p in points if p["group"] == "profile_1d"]
    p2d = [p for p in points if p["group"] == "profile_2d"]
    # The production-record point count: 6 x 10 1-D + 2 x 48 2-D.
    assert len(points) == 156
    assert len(p1d) == 60
    assert len(p2d) == 96

    # 1-D: the anchor scan point (index 5) is absent per parameter; the
    # id scheme and the payload fields are exact.
    for name in se.M11_PROFILE_SET:
        got = sorted(
            p["scan_index"] for p in p1d if p["parameter"] == name
        )
        assert got == [k for k in range(se.M11_GRID_N) if k != 5]
        ids = {
            p["id"] for p in p1d if p["parameter"] == name
        }
        assert ids == {
            f"p1d_{name.replace('.', '__')}_{k:02d}"
            for k in range(se.M11_GRID_N)
            if k != 5
        }
    sample = p1d[0]
    assert sample["kind"] == "nuisance_refit"
    assert sample["parameter"] == se.M11_PROFILE_SET[0]
    assert sample["fixed"] == {
        sample["parameter"]: expected["grids_1d"][sample["parameter"]][
            sample["scan_index"]
        ]
    }
    assert sample["nuisance"] == [
        n for n in se.M11_PROFILE_SET if n != sample["parameter"]
    ]
    assert sample["start"] == dict(M11_SYNTH_ANCHOR_VALUES)
    assert sample["bounds"] == {
        n: list(se.M11_BOUNDS[n]) for n in se.M11_PROFILE_SET
    }
    assert sample["cap_evaluations"] == se.M11_CAP_CEILING
    assert sample["events_per_replica"] == se.M11_EVENTS_PER_REPLICA_FULL
    # Grid reproducibility: the recorded scan values equal the expected
    # grids exactly (the collect drift-check rule).
    for p in p1d:
        assert p["scan_value"] == expected["grids_1d"][p["parameter"]][
            p["scan_index"]
        ]

    # 2-D: 48 cells per pair (the anchor cell (3, 3) absent), row-major
    # ordering, exact fixed values.
    for pair_id in ("pair0", "pair3"):
        cells = sorted(
            (p["grid_i"], p["grid_j"])
            for p in p2d
            if p["pair_id"] == pair_id
        )
        assert len(cells) == 48
        assert (3, 3) not in cells
        axis0, axis1 = expected["grids_2d"][pair_id]
        names = se.M11_FULL_GRID_PAIRS[pair_id]
        for p in [q for q in p2d if q["pair_id"] == pair_id]:
            i, j = p["grid_i"], p["grid_j"]
            assert p["fixed"] == {names[0]: axis0[i], names[1]: axis1[j]}
            assert p["nuisance"] == [
                n for n in se.M11_PROFILE_SET
                if n not in (names[0], names[1])
            ]
        first = next(
            p for p in points if p["group"] == "profile_2d"
            and p["pair_id"] == pair_id
        )
        assert first["id"] == f"p2d_{pair_id}_i00_j00"
    last = points[-1]
    assert last["id"] == "p2d_pair3_i06_j06"

    # A 2-D axis whose sigma is unavailable takes the 0.1-window fallback.
    record2 = {
        "sigma_per_parameter": {**sigmas, "regge.edge_power": None},
        "rungs": {},
    }
    axes, fallbacks = se.m11_grid2d_axes(
        "pair0", dict(M11_SYNTH_ANCHOR_VALUES), sigmas,
        hessian_record=record2,
    )
    assert fallbacks["regge.alpha_exc0"] is False
    assert fallbacks["regge.edge_power"] is True
    lo_e, hi_e = se.M11_BOUNDS["regge.edge_power"]
    width_e = se.M11_GRID_FALLBACK_WINDOW_FRACTION * (hi_e - lo_e)
    a_edge = M11_SYNTH_ANCHOR_VALUES["regge.edge_power"]
    assert len(axes[1]) == se.M11_FULL_GRID_N
    assert axes[1][0] == pytest.approx(a_edge - width_e, abs=1e-12)
    assert axes[1][-1] == pytest.approx(a_edge + width_e, abs=1e-12)


def test_m11_profile_intervals_flags_levels_at_the_scan_edge():
    """The at-edge flags: a flat profile is quoted one-sided at both scan
    edges; a level beyond the grid edge is flagged one-sided; a level
    exactly at the grid edge stays bracketed (the crossing is at the
    edge, read parabolically)."""
    a4 = M11_SYNTH_ANCHOR_VALUES[M11_SYNTH_SCAN_PARAM]
    s1 = math.sqrt(
        float(M11_SYNTH_V[M11_SYNTH_SCAN_INDEX, M11_SYNTH_SCAN_INDEX])
    )

    # (a) Flat: every chi2 equals chi2_0; both levels are unbracketed.
    flat_grid = [2.4 + 0.025 * k for k in range(11)]
    flat_iv = se.m11_profile_intervals(
        flat_grid, [100.0] * 11, theta0=2.5, chi2_0=100.0
    )
    for key in ("sigma1", "sigma2"):
        block = flat_iv[key]
        assert block["at_edge_lower"] is True
        assert block["at_edge_upper"] is True
        assert block["one_sided"] is True
        assert block["lower"] == pytest.approx(2.4, abs=1e-12)
        assert block["upper"] == pytest.approx(2.65, abs=1e-12)
        assert block["method_lower"] is None
        assert block["method_upper"] is None

    # (b) Coarse quadratic: the 1-sigma level is interior (parabolic
    # reading, exact on the quadratic); the 2-sigma level is beyond the
    # grid edge (flagged one-sided at the edges).
    coarse = [a4 + d * s1 for d in (-1.8, -0.9, 0.0, 0.9, 1.8)]
    coarse_chi2 = [M11_SYNTH_C0 + d * d for d in (-1.8, -0.9, 0.0, 0.9, 1.8)]
    coarse_iv = se.m11_profile_intervals(
        coarse, coarse_chi2, theta0=a4, chi2_0=M11_SYNTH_C0
    )
    sig1 = coarse_iv["sigma1"]
    assert sig1["lower"] == pytest.approx(a4 - s1, rel=1e-9, abs=1e-12)
    assert sig1["upper"] == pytest.approx(a4 + s1, rel=1e-9, abs=1e-12)
    assert sig1["one_sided"] is False
    assert sig1["method_lower"] == "parabolic"
    assert sig1["method_upper"] == "parabolic"
    sig2 = coarse_iv["sigma2"]
    assert sig2["at_edge_lower"] is True
    assert sig2["at_edge_upper"] is True
    assert sig2["one_sided"] is True
    assert sig2["lower"] == pytest.approx(a4 - 1.8 * s1, abs=1e-12)
    assert sig2["upper"] == pytest.approx(a4 + 1.8 * s1, abs=1e-12)

    # (c) Level exactly at the grid edge: bracketed (the crossing is at
    # the edge), not flagged.
    edge = [a4 + d * s1 for d in (-2.0, -1.0, 0.0, 1.0, 2.0)]
    edge_chi2 = [M11_SYNTH_C0 + d * d for d in (-2.0, -1.0, 0.0, 1.0, 2.0)]
    edge_iv = se.m11_profile_intervals(
        edge, edge_chi2, theta0=a4, chi2_0=M11_SYNTH_C0
    )
    sig2e = edge_iv["sigma2"]
    assert sig2e["at_edge_lower"] is False
    assert sig2e["at_edge_upper"] is False
    assert sig2e["one_sided"] is False
    assert sig2e["lower"] == pytest.approx(a4 - 2.0 * s1, abs=1e-12)
    assert sig2e["upper"] == pytest.approx(a4 + 2.0 * s1, abs=1e-12)
    assert sig2e["method_lower"] == "parabolic"
    assert sig2e["method_upper"] == "parabolic"


def test_m11_scan_grid_half_width_clipping_and_fallback():
    """The 1-D scan grid: the 4-sigma half-width clipped to the
    governing bounds, the 0.1-window fallback when the sigma is
    unavailable, and the 11-point spacing in both cases."""
    # Fallback: sigma unavailable.
    g, fb = se.m11_scan_grid("regge.alpha_exc0", 0.5, None)
    assert fb is True
    assert len(g) == se.M11_GRID_N
    lo, hi = se.M11_BOUNDS["regge.alpha_exc0"]
    width = se.M11_GRID_FALLBACK_WINDOW_FRACTION * (hi - lo)
    assert g[0] == pytest.approx(0.5 - width, abs=1e-12)
    assert g[-1] == pytest.approx(0.5 + width, abs=1e-12)

    # 4-sigma half-width, unclipped.
    s4 = float(M11_SYNTH_SIGMA_TRUE[M11_SYNTH_SCAN_INDEX])
    a4 = M11_SYNTH_ANCHOR_VALUES[M11_SYNTH_SCAN_PARAM]
    g4, fb4 = se.m11_scan_grid(M11_SYNTH_SCAN_PARAM, a4, s4)
    assert fb4 is False
    assert len(g4) == se.M11_GRID_N
    assert g4[0] == pytest.approx(a4 - 4.0 * s4, abs=1e-12)
    assert g4[-1] == pytest.approx(a4 + 4.0 * s4, abs=1e-12)

    # Lower-bound clipping: the half-width is clipped at the bound.
    gc, fbc = se.m11_scan_grid("regge.alpha_exc0", 0.05, 0.1)
    assert fbc is False
    assert gc[0] == 0.0
    assert gc[-1] == pytest.approx(0.45, abs=1e-12)
    # Upper-bound clipping.
    gu, _ = se.m11_scan_grid("continuum.match_mass", 3.9, 0.1)
    assert gu[0] == pytest.approx(3.5, abs=1e-12)
    assert gu[-1] == 4.0


def test_m11_tolerance_and_m8_start_theta_conventions():
    """The anchor-scan-point tolerance is relative
    (1e-9 * max(1, |anchor|)), and the M11 start point is the M8
    ten-parameter point projected onto the profile set."""
    assert se.m11_theta_within_tolerance(2.5 + 1e-10, 2.5) is True
    assert se.m11_theta_within_tolerance(2.5 + 1e-8, 2.5) is False
    # At an anchor of zero the tolerance floor is 1e-9.
    assert se.m11_theta_within_tolerance(1e-9, 0.0) is True
    assert se.m11_theta_within_tolerance(2e-9, 0.0) is False

    expected = [
        se.M11_M8_PARAMETER_SET["regge.alpha_exc0"],
        se.M11_M8_PARAMETER_SET["regge.edge_power"],
        se.M11_STRANGE_THRESHOLD_M8_FROZEN,
        se.M11_M8_PARAMETER_SET["continuum.pair_threshold_power"],
        se.M11_M8_PARAMETER_SET["continuum.match_mass"],
        se.M11_M8_PARAMETER_SET["continuum.pion_multiplicity_slope"],
    ]
    assert se.m11_m8_start_theta() == expected
    assert se.m11_m8_start_theta()[2] == 1.5
