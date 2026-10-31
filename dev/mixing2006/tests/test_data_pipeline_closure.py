"""Hermetic self-check suite for the R10 data pipeline closure driver.

Five checks (card ``TASK-20260911-r1-run-r10-data-pipeline-closure``):

- ``test_data_inventory_schema``: the §7.1 data inventory carries all
  fields non-null for every table, the six SPS 158 GeV/c tables and the
  p+Pb tables are present, and ``split`` is ``"training"`` for the SPS
  tables and ``"held_out"`` for the p+Pb tables.
- ``test_toy_power_law_analytic``: the closed-form bin integral for the
  separable power law matches hand-computed values on a 3x3 bin grid to
  < 1e-12 (an independent antiderivative expression plus one hard-coded
  literal).
- ``test_toy_uniform_analytic``: the closed-form bin integral for the
  uniform density is ``A * dxF * dpT`` per bin, exact to < 1e-15.
- ``test_pipeline_closure_analytic``: the pipeline code path applied to
  the exact bin integrals of the three toy spectra achieves
  ``max_rel_deviation < 1e-6`` on the 1017-bin six-species observable set
  (the Gate 1 criterion).
- ``test_pipeline_closure_mc_sanity``: a small-N MC sample (100,000
  events, NOT the run's 1,000,000-event sample; seed 20303000) through
  the pipeline satisfies the pre-registered per-bin tolerance
  ``max(3 / sqrt(N_bin), 1e-3)`` for at least 95% of the non-empty bins,
  with ``N_bin == 0`` bins excluded from the check and counted under
  ``empty_bins``.

Hermetic: no physics-model event generation; the toy generator is the
driver's deterministic sampler from the declared densities and the
small-N samples below are the only sampling in the test (the
1,000,000-event sample is the run itself, not a test).  The inventory and
closure checks read the bundled SPS tables and the pa_data p+Pb tables
and skip with a recorded reason when the reference mirror is absent from
the checkout (the same convention as ``test_preflight_sps_check`` and the
R01/R02 harness tests).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from helpers.benchmarking import staged_data_closure as r10

SPS_DATA = r10.NESTED_ROOT / "spshadrons.web.cern.ch" / "data"
_SPS_TABLE_NAMES = (
    "xsect_piminus.txt",
    "xsect_piplus.txt",
    "xsect_kaplus_pp.txt",
    "xsect_kaminus_pp.txt",
    "xsect_proton_pp.txt",
    "xsect_antiproton_pp.txt",
)
_SPS_PRESENT = all((SPS_DATA / name).is_file() for name in _SPS_TABLE_NAMES)
_PAPB_PRESENT = (r10.pa_data_dir() / "pPb_dens_minbias_piminus.txt").is_file()

pytestmark = pytest.mark.skipif(
    not (_SPS_PRESENT and _PAPB_PRESENT),
    reason=(
        "bundled SPS/pa_data tables absent from the checkout (source-only "
        "archive); the R10 data-inventory and closure checks need them"
    ),
)


@pytest.fixture(scope="module")
def grids() -> list:
    return r10.closure_grids()


@pytest.fixture(scope="module")
def inventory() -> dict:
    return r10.build_inventory()


# ---------------------------------------------------------------------------
# Inventory schema
# ---------------------------------------------------------------------------


def test_data_inventory_schema(inventory):
    """All §7.1 fields non-null for every table; the expected table set."""
    complete, n_tables, n_complete = r10.inventory_completeness(inventory)
    assert n_tables == 18
    assert n_complete == n_tables
    assert complete is True

    tables = {record["table"]: record for record in inventory["tables"]}
    # The six SPS 158 GeV/c tables, training split.
    sps_paths = {
        f"spshadrons.web.cern.ch/data/{name}" for name in _SPS_TABLE_NAMES
    }
    for path in sps_paths:
        record = tables[path]
        assert record["split"] == "training"
        assert record["beam_momentum_gev_c"] == 158.0
        assert record["sqrt_s_gev"] == 17.270203
        assert record["normalization_convention"] == "f / 31.78 mb"
        assert record["units"] == "1/GeV^2"
        assert record["feeddown_convention"] == "as_published_inclusive"
        assert record["acceptance_correction"] == "trigger_bias_included"
        assert record["bin_edges"]["xF"] and record["bin_edges"]["pT"]
    assert {tables[p]["species"] for p in sps_paths} == {
        "pi-",
        "pi+",
        "K+",
        "K-",
        "p",
        "pbar",
    }

    # The twelve p+Pb tables (minimum bias + centrality sets 1..5), held out.
    papb_paths = [
        path
        for path in tables
        if path.startswith("dev/mixing2006/helpers/benchmarking/pa_data/")
    ]
    assert len(papb_paths) == 12
    for path in papb_paths:
        record = tables[path]
        assert record["split"] == "held_out"
        assert record["species"] in ("pi-", "pi+")
        assert record["feeddown_convention"] == "as_published_feeddown_subtracted"
        assert record["bin_edges"]["xF"] and record["bin_edges"]["pT"]

    # The 1017-bin observable set is recorded in the conventions block.
    observable = inventory["conventions"]["observable_set"]
    assert observable["total_bins"] == 1017
    assert observable["window"] == {"xF": [0.0, 0.4], "pT": [0.05, 1.5]}
    assert observable["species_bin_counts"] == r10.SPECIES_BIN_COUNTS
    # Card 7: the shared normalization-nuisance width, covariance off.
    assert inventory["conventions"]["use_data_covariance"] is False
    assert inventory["conventions"]["sigma_eta"] > 0.0


# ---------------------------------------------------------------------------
# Closed-form toy integrals (hand-computed anchors)
# ---------------------------------------------------------------------------

_POWER_LAW = {"name": "separable_power_law", "A": 1.0, "a": 2.0, "b": 1.0}
# A 3x3 bin grid: xF edges [0.0, 0.1, 0.2, 0.3], pT edges [0.05, 0.3, 0.6, 1.0].
_X_EDGES = (0.0, 0.1, 0.2, 0.3)
_P_EDGES = (0.05, 0.3, 0.6, 1.0)


def _hand_power_law(xlo, xhi, plo, phi):
    """Independent antiderivative form of the separable power law."""
    fx = lambda x: -((1.0 - x) ** 3.0) / 3.0  # d/dx F = (1-x)^2
    gp = lambda p: -math.exp(-p)  # d/dp G = exp(-p)
    return (fx(xhi) - fx(xlo)) * (gp(phi) - gp(plo))


def test_toy_power_law_analytic():
    """Closed form vs hand-computed values on a 3x3 grid, < 1e-12."""
    toy = dict(_POWER_LAW)
    worst = 0.0
    for ix in range(3):
        for ip in range(3):
            xlo, xhi = _X_EDGES[ix], _X_EDGES[ix + 1]
            plo, phi = _P_EDGES[ip], _P_EDGES[ip + 1]
            got = float(r10.toy_bin_integral(toy, xlo, xhi, plo, phi))
            want = _hand_power_law(xlo, xhi, plo, phi)
            worst = max(worst, abs(got - want))
    assert worst < 1e-12
    # One hard-coded literal anchors the first bin against regression.
    literal = float(
        r10.toy_bin_integral(_POWER_LAW, 0.0, 0.1, 0.05, 0.3)
    )
    assert literal == pytest.approx(0.01900714541164931, rel=0.0, abs=1e-12)


def test_toy_uniform_analytic():
    """The uniform bin integral is A * dxF * dpT per bin, exact < 1e-15."""
    toy = {"name": "uniform", "A": 1.0}
    for ix in range(3):
        for ip in range(3):
            xlo, xhi = _X_EDGES[ix], _X_EDGES[ix + 1]
            plo, phi = _P_EDGES[ip], _P_EDGES[ip + 1]
            got = float(r10.toy_bin_integral(toy, xlo, xhi, plo, phi))
            want = 1.0 * (xhi - xlo) * (phi - plo)
            assert abs(got - want) < 1e-15
    # The 2-D Gaussian closed form factorizes into two math.erf CDF
    # differences (scalar stdlib erf vs the driver's vectorized erf).
    toy_g = {
        "name": "gaussian_2d",
        "A": 1.0,
        "mu_xF": 0.2,
        "sig_xF": 0.1,
        "mu_pT": 0.5,
        "sig_pT": 0.3,
    }
    xlo, xhi, plo, phi = 0.1, 0.3, 0.2, 0.8
    de_x = math.erf((xhi - 0.2) / (0.1 * math.sqrt(2))) - math.erf(
        (xlo - 0.2) / (0.1 * math.sqrt(2))
    )
    de_p = math.erf((phi - 0.5) / (0.3 * math.sqrt(2))) - math.erf(
        (plo - 0.5) / (0.3 * math.sqrt(2))
    )
    want = 0.1 * 0.3 * (math.pi / 2.0) * de_x * de_p
    got = float(r10.toy_bin_integral(toy_g, xlo, xhi, plo, phi))
    assert abs(got - want) < 1e-15


# ---------------------------------------------------------------------------
# Pipeline closure (analytic path and MC sanity)
# ---------------------------------------------------------------------------


def test_pipeline_closure_analytic(grids):
    """The pipeline chain on the exact integrals: < 1e-6 for all toys."""
    sqrt_s = r10.pipeline_sqrt_s()
    assert sum(g.n_bins for g in grids) == 1017
    for toy in r10.TOY_SPECTRA:
        analytic = r10.analytic_closure(
            toy, grids, sqrt_s, n_ref=r10.DEFAULT_MC_EVENTS
        )
        assert analytic["n_bins"] == 1017
        assert analytic["max_rel_deviation_analytic"] < r10.ANALYTIC_TOLERANCE
        # Every per-bin deviation is a finite float far below the tolerance.
        devs = [b["rel_deviation_analytic"] for b in analytic["bins"]]
        assert len(devs) == 1017
        assert max(devs) < 1e-6
        assert all(math.isfinite(d) for d in devs)


def test_pipeline_closure_mc_sanity(grids):
    """Small-N MC sample at the pre-registered tolerance rule, >= 95%."""
    sqrt_s = r10.pipeline_sqrt_s()
    small_n = 100_000
    assert small_n != r10.DEFAULT_MC_EVENTS
    for index, toy in enumerate(r10.TOY_SPECTRA):
        analytic = r10.analytic_closure(toy, grids, sqrt_s, n_ref=small_n)
        model_by_species = {}
        offset = 0
        for grid in grids:
            model_by_species[grid.species_display] = np.asarray(
                [
                    analytic["bins"][offset + i]["model"]
                    for i in range(grid.n_bins)
                ],
                dtype=float,
            )
            offset += grid.n_bins
        mc = r10.mc_closure(
            toy,
            index,
            grids,
            model_by_species,
            sqrt_s,
            n_events=small_n,
            mc_seed=r10.MC_SEED,
        )
        assert mc["mc_seed"] == 20303000
        assert mc["mc_events_used"] == small_n
        assert mc["empty_bins"] == len(mc["empty_bin_records"])
        # Bins with N_bin == 0 are excluded from the tolerance check.
        assert mc["n_non_empty_bins"] + mc["empty_bins"] == 1017
        assert mc["mc_sanity_pass_fraction"] >= r10.MC_PASS_FRACTION_MIN
        for entry in mc["bins"]:
            if entry["n_bin"] == 0:
                assert entry["mc_pass"] is None
                assert entry["tolerance"] is None
            else:
                tol = max(3.0 / math.sqrt(entry["n_bin"]), 1e-3)
                assert entry["tolerance"] == pytest.approx(tol)
                assert entry["mc_pass"] == (entry["rel_deviation_mc"] <= tol)
