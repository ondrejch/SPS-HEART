"""Hermetic tests for the R4 transition routing formulation audit (Round 4, card 1).

Task ``TASK-20260920-r4-audit-transition-routing-formulation`` (the Gate 3
decision card follow-up obligation 1; plan section 9.3, lines 625-632).
The tests exercise the card's pre-registered measurement contract against
the committed audit driver, without any Monte Carlo:

- ``test_module_contract_pins``: the pre-registered probe-set identity
  (the 32 persisting residuals, the 18 K+ split), the blend variants
  (the ramp start B / B - 0.2 / B + 0.2 with W = 0.5 GeV fixed), the
  bounds (bin spread 2.0 / feed-down 3.0 / census 1e-6), the 20% width
  contract with W' in {0.3, 0.7}, the neutral alpha and its bracket, the
  4.0-SE census-magnitude envelope, the frozen substrate digests, and
  the reserved audit block.
- ``test_driver_copies_byte_identical``: the nested and outer driver
  copies are byte-identical (the dual-tree register).
- ``test_zero_draw_source_contract``: the driver source carries no
  random-number machinery (no ``random`` import, no numpy random call,
  no seed sequence), and the pre-registration and verdict builders
  record ``draws: 0``, ``no_fitting: true``, ``optimizer_invoked:
  false``, and the unfired reserved block.
- ``test_pooled_census_and_disagreement``: the routing-census pooling
  from sealed per-replica payloads against a hand-computed fixture, and
  the 1e-6 disagreement check flagging a deliberately wrong sealed
  census.
- ``test_monotone_predicate_and_offline_residual``: the (P1)
  monotonicity predicate and the (P2) census-line residual on synthetic
  three-variant payloads, including the exact 4.0-SE envelope boundary
  and a non-symmetric census interpolation.
- ``test_width_reweighting_hand_case``: the W' reweighting algebra
  against hand-computed two-variant cases (the flat-slope identity, the
  differential-slope value-space reweighting, and the p95 scaling on a
  synthetic bin set).
- ``test_evaluate_verdict_criteria_paths``: the criteria (a), (b), (c)
  firing paths and the all-clean PASS path on synthetic artifacts.
- ``test_finding_narrative``: the named-finding narrative fields of a
  non-monotone synthetic row.
- ``test_sealed_r32_readonly_pins``: the sealed R32 archives verify
  their seals, the persisting register carries exactly 32 entries with
  the pre-registered species split and sigma range, the sealed p95 and
  census anchors match the pre-registration, and the sealed input
  digests re-hash.
- ``test_card6_surface_byte_stability``: the card-6 blend routing
  surface stays at its pinned digest.
- ``test_freeze_manifest_repin``: the manifest carries the three new
  inventory entries with the on-disk digests and the ``freeze_exceptions``
  entry naming this card.
- ``test_audit_run_artifact_pins``: the sealed audit run dir's
  artifacts (activated once the run dir exists; skipped with a recorded
  reason before the audit run): the pre-registration contract, the
  provenance, the census recomputation, the attribution rows and the
  K+ referral, the width reweighting, the verdict, and the seal +
  clean-validation records.

The tests read the sealed R32 archives read-only (their seals verify)
and never write outside ``tmp_path``.  No event generation, no random
numbers, no optimizer, no external SPS tables.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from tests._sealed_archive import skip_unless_sealed

NESTED_ROOT = Path(__file__).resolve().parent.parent
DRIVER_REL = Path("helpers") / "benchmarking" / "audit_r4_routing_formulation.py"
NESTED_DRIVER = NESTED_ROOT / DRIVER_REL
OUTER_DRIVER = NESTED_ROOT.parent.parent / DRIVER_REL
SYNC_CHECKER = NESTED_ROOT.parent.parent / "helpers" / "benchmarking" / (
    "check_helpers_layout_sync.py"
)
MANIFEST_PATH = NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json"
CARD6_REL = "dev/mixing2006/na49_mixing/spectral_density.py"
CARD6_SHA256 = "561c65ee9a77a1b7f7eb07a7619dafb584779a51ccb23d7fe5651c0605e12563"
R32_PROD_DIR = "dev/00runs/r32_spectral_systematic_pp_2026-09-18"
R32_CONF_DIR = "dev/00runs/r32_spectral_systematic_pp_confirmation_2026-09-19"

# The sealed audit run dir (the audit's own Phase 2 output).  The pins
# activate once the dir exists; before the audit run they skip with a
# recorded reason (the R32 pins convention, activated at phase 2).
RUN_DIR_CANDIDATES = tuple(
    (NESTED_ROOT.parent.parent / "dev" / "00runs").glob(
        "r4_audit_transition_routing_formulation_*"
    )
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "audit_r4_routing_formulation_under_test", NESTED_DRIVER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    return _load_driver()


# ----------------------------------------------------------------------
# Pre-registered contract pins
# ----------------------------------------------------------------------


def test_module_contract_pins(driver):
    assert driver.CARD == (
        "TASK-20260920-r4-audit-transition-routing-formulation"
    )
    assert driver.RUN_LABEL == "r4_audit_transition_routing_formulation"
    # The byte-stable card-6 surface and the frozen substrate.
    assert driver.CARD6_SHA256 == CARD6_SHA256
    assert driver.BUILDER_SHA256 == (
        "4ab612609e1ee0bc1e037dcbb2815b6484306ad32402b7289e6636f8e829cdb1"
    )
    assert driver.ENGINE_SHA256 == (
        "ee240f44645d2e592acfae462f7d1e9c3b3dd8cd78c96bd40b7b73509f7fcc63"
    )
    assert driver.CATALOG_SHA256 == (
        "6e4de7372a5bbf99affd6fa3ccc295c4cef09404dab3b17acd8816cda762bdcd"
    )
    assert driver.FAMILIES_SHA256 == (
        "242f906cf040eadbe3796dfcfb61d0d51535277399858c901a162e6653d635e1"
    )
    assert driver.R32_DRIVER_SHA256 == (
        "99454facce856c93505e6c4b0d4e47b85eeb0ef54f72b1631e3d29fcc511cb90"
    )
    # The blend variants: the ramp width fixed, the ramp start B / B +/- 0.2.
    assert driver.RAMP_WIDTH_GEV == 0.5
    assert driver.RAMP_OFFSETS_GEV == {"0": 0.0, "1": -0.2, "2": 0.2}
    assert driver.RAMP_START_ORDER == ("1", "0", "2")
    assert driver.KERNEL_BOUNDARY_GEV == 2.680194110727528
    # The pre-registered bounds and contracts.
    assert driver.BIN_SPREAD_BOUND == 2.0
    assert driver.FEEDDOWN_SPREAD_BOUND == 3.0
    assert driver.CENSUS_DISAGREEMENT_BOUND == 1e-6
    assert driver.WIDTH_CONTRACT_REL == 0.20
    assert driver.WIDTH_PRIMES_GEV == (0.3, 0.7)
    assert driver.ALPHA_NEUTRAL == 0.5
    assert driver.ALPHA_BRACKET == (0.0, 1.0)
    assert driver.OFFLINE_ENVELOPE_SE == 4.0
    # The probe-set identity (the sealed R32 confirmation register).
    assert driver.PERSISTING_TOTAL == 32
    assert driver.PERSISTING_SPECIES_SPLIT == {
        "k_plus": 18, "proton": 9, "pi_plus": 4, "pi_minus": 1,
    }
    assert driver.PERSISTING_SIGMA_MIN == 2.0002
    assert driver.PERSISTING_SIGMA_MAX == 7.5329
    assert driver.PERSISTING_SIGMA_TOLERANCE == 5e-4
    assert driver.PERSISTING_NON_STRANGE == ("proton", "pi_plus", "pi_minus")
    # The sealed anchors.
    assert driver.PRODUCTION_P95 == 2.9604990756942544
    assert driver.CONFIRMATION_P95 == 4.028426092137674
    assert driver.PRODUCTION_CENSUS_CATALOG == {
        "0": 0.22329625, "1": 0.193558125, "2": 0.253725,
    }
    assert driver.CONFIRMATION_CENSUS_CATALOG == {
        "0": 0.22357, "1": 0.1936659375, "2": 0.253850625,
    }
    # The reserved audit block (must not fire on the first pass).
    assert driver.RESERVED_MASTER_SEED == 20260920
    assert driver.RESERVED_BLOCK_FIRST == 20301040
    assert driver.RESERVED_BLOCK_LAST == 20301055
    # The strange-sector referral (card 2) and the non-strange set.
    assert driver.STRANGE_REFERRAL_CARD == (
        "TASK-20260920-r4-audit-strange-sector-continuum-families"
    )


def test_driver_copies_byte_identical():
    assert NESTED_DRIVER.is_file(), NESTED_DRIVER
    assert OUTER_DRIVER.is_file(), OUTER_DRIVER
    nested_bytes = NESTED_DRIVER.read_bytes()
    outer_bytes = OUTER_DRIVER.read_bytes()
    assert nested_bytes == outer_bytes
    assert hashlib.sha256(nested_bytes).hexdigest() == (
        hashlib.sha256(outer_bytes).hexdigest()
    )


def test_zero_draw_source_contract(driver):
    source = NESTED_DRIVER.read_text(encoding="utf-8")
    forbidden = (
        "np.random", "default_rng", "SeedSequence", "import random",
        "random.", "randint", "shuffle(", "gauss(", "weibull",
        "getstate", "setstate",
    )
    hits = [token for token in forbidden if token in source]
    assert not hits, f"the audit driver carries random machinery: {hits}"
    # The recorded zero-draw contract of the pre-registration and the
    # verdict builders.
    pre_fields = driver.build_pre_registration(
        driver.substrate_digests(), []
    )
    assert pre_fields["zero_draw"]["draws"] == 0
    assert pre_fields["zero_draw"]["no_fitting"] is True
    assert pre_fields["zero_draw"]["optimizer_invoked"] is False
    block = pre_fields["zero_draw"]["reserved_audit_block"]
    assert block["fired"] is False
    assert block["raw_seeds"] == [20301040, 20301055]
    assert block["master_seed"] == 20260920


# ----------------------------------------------------------------------
# Census pooling and disagreement (hand-computed fixture)
# ----------------------------------------------------------------------


def test_pooled_census_and_disagreement(driver, tmp_path):
    records_v0 = [
        {"routing": {"n_catalog": 10, "n_sources": 100}},
        {"routing": {"n_catalog": 30, "n_sources": 100}},
    ]
    records_v1 = [
        {"routing": {"n_catalog": 50, "n_sources": 100}},
        {"routing": {"n_catalog": 50, "n_sources": 100}},
    ]
    pooled_0 = driver.pooled_census_from_payloads(records_v0)
    pooled_1 = driver.pooled_census_from_payloads(records_v1)
    # Hand-computed pooling: 40/200 and 100/200.
    assert pooled_0["n_catalog"] == 40
    assert pooled_0["n_sources"] == 200
    assert pooled_0["n_continuum"] == 160
    assert pooled_0["fraction_catalog"] == 0.2
    assert pooled_0["fraction_continuum"] == 0.8
    assert pooled_1["fraction_catalog"] == 0.5
    assert pooled_0["replicas"] == 2
    # Exact agreement with a matching sealed census.
    sealed = {
        "fraction_catalog": 0.2, "fraction_continuum": 0.8,
        "n_catalog": 40, "n_continuum": 160, "n_sources": 200,
    }
    fields = driver.census_disagreement(pooled_0, sealed)
    assert max(fields.values()) == 0.0
    # A deliberately wrong sealed census is flagged beyond 1e-6.
    wrong = dict(sealed, fraction_catalog=0.3)
    fields = driver.census_disagreement(pooled_0, wrong)
    assert max(fields.values()) == pytest.approx(0.1)
    assert max(fields.values()) > driver.CENSUS_DISAGREEMENT_BOUND
    # A sealed census rounded at the 1e-6 scale stays inside the bound.
    rounded = dict(sealed, fraction_catalog=0.2 + 5e-7)
    fields = driver.census_disagreement(pooled_0, rounded)
    assert max(fields.values()) <= driver.CENSUS_DISAGREEMENT_BOUND


def test_census_disagreement_bound_semantics(driver):
    """A 9e-7 disagreement passes the 1e-6 bound; 1.1e-6 fails it."""
    base = {
        "fraction_catalog": 0.25, "fraction_continuum": 0.75,
        "n_catalog": 50, "n_continuum": 150, "n_sources": 200,
    }
    near = dict(base, fraction_catalog=0.25 + 9e-7)
    assert max(driver.census_disagreement(base, near).values()) <= (
        driver.CENSUS_DISAGREEMENT_BOUND
    )
    far = dict(base, fraction_catalog=0.25 + 1.1e-6)
    assert max(driver.census_disagreement(base, far).values()) > (
        driver.CENSUS_DISAGREEMENT_BOUND
    )


# ----------------------------------------------------------------------
# The attribution predicate (P1 monotonicity, P2 census magnitude)
# ----------------------------------------------------------------------


def _row_from_values(drv, values, unit_se, census, parent_values=None):
    """A synthetic attribution row in the driver's row shape."""
    median_i = drv.median_variant_index(values)
    deltas = {
        v: (values[int(v)] - values[median_i]) / unit_se
        for v in ("0", "1", "2")
    }
    ramp_ordered = [deltas[v] for v in drv.RAMP_START_ORDER]
    monotone = drv.is_monotone_in_ramp_start(ramp_ordered)
    lam = (census["0"] - census["1"]) / (census["2"] - census["1"])
    r_se = drv.offline_residual_se(values, unit_se, lam)
    spread = max(values) - min(values)
    return {
        "values": values,
        "unit_se": unit_se,
        "median_variant": median_i,
        "deltas": deltas,
        "ramp_ordered": ramp_ordered,
        "p1_monotone": monotone,
        "lambda": lam,
        "offline_residual_se": r_se,
        "p2_within_envelope": abs(r_se) <= (
            drv.OFFLINE_ENVELOPE_SE + drv.OFFLINE_ENVELOPE_TOLERANCE
        ),
        "sigma": spread / unit_se,
        "mass_blind_envelope_sigma": (
            (census["2"] - census["1"])
            * max(values[1] / (1.0 - census["1"]), values[2] / census["2"])
            / unit_se
        ),
    }


def test_median_variant_index_deterministic(driver):
    # The middle element of the value-sorted (value, index) order: the
    # returned index is the variant whose value is the median.
    assert driver.median_variant_index([1.0, 2.0, 3.0]) == 1
    assert driver.median_variant_index([3.0, 2.0, 1.0]) == 1
    assert driver.median_variant_index([3.0, 1.0, 2.0]) == 2
    # Ties break to the lowest index (the sealed register convention).
    assert driver.median_variant_index([2.0, 2.0, 1.0]) == 0
    assert driver.median_variant_index([1.0, 2.0, 2.0]) == 1


def test_monotone_predicate_synthetic(driver):
    # Monotone increasing in the ramp start: y(v1) < y(v0) < y(v2).
    census = {"0": 0.25, "1": 0.22, "2": 0.28}
    row = _row_from_values(driver, [2.0, 1.0, 3.0], 0.1, census)
    assert row["p1_monotone"] is True
    assert row["ramp_ordered"] == [-10.0, 0.0, 10.0]
    # Monotone decreasing in the ramp start (the continuum-dominant
    # direction the K+ bins follow).
    row = _row_from_values(driver, [2.0, 3.0, 1.0], 0.1, census)
    assert row["p1_monotone"] is True
    assert row["ramp_ordered"] == [10.0, 0.0, -10.0]
    # Non-monotone: the nominal variant off the early/late line.
    row = _row_from_values(driver, [3.0, 1.0, 2.0], 0.1, census)
    assert row["p1_monotone"] is False
    assert row["ramp_ordered"] == [-10.0, 10.0, 0.0]
    # A flat middle step is still monotone (non-strict).
    row = _row_from_values(driver, [1.0, 1.0, 3.0], 0.1, census)
    assert row["p1_monotone"] is True


def test_offline_residual_census_magnitude(driver):
    # Symmetric census (lambda = 0.5): the line is the early/late mean.
    census = {"0": 0.25, "1": 0.225, "2": 0.275}
    row = _row_from_values(driver, [1.5, 1.0, 2.0], 0.1, census)
    assert row["lambda"] == pytest.approx(0.5)
    assert row["offline_residual_se"] == pytest.approx(0.0)
    assert row["p2_within_envelope"] is True
    # The exact 4.0-SE envelope boundary is inclusive: with dyadic
    # values, y0 = 5.0, line = (2.0 + 6.0)/2 = 4.0, r = 1.0, se = 0.25
    # gives r_se = 4.0 exactly and passes; y0 = 5.1 gives 4.4 and fails.
    row = _row_from_values(driver, [5.0, 2.0, 6.0], 0.25, census)
    assert row["offline_residual_se"] == pytest.approx(4.0)
    assert row["p2_within_envelope"] is True
    row = _row_from_values(driver, [5.1, 2.0, 6.0], 0.25, census)
    assert row["offline_residual_se"] == pytest.approx(4.4)
    assert row["p2_within_envelope"] is False
    # Just inside the envelope: r_se = 3.9.
    row = _row_from_values(driver, [4.975, 2.0, 6.0], 0.25, census)
    assert row["offline_residual_se"] == pytest.approx(3.9)
    assert row["p2_within_envelope"] is True
    # A P1-pass / P2-fail case: monotone but off the census line.
    census_asym = {"0": 0.25, "1": 0.19, "2": 0.30}
    row = _row_from_values(driver, [2.05, 1.0, 2.0], 0.1, census_asym)
    lam = (0.25 - 0.19) / (0.30 - 0.19)
    # lambda = 6/11: line at the nominal = (5/11)*1.0 + (6/11)*2.0.
    line = (1.0 - lam) * 1.0 + lam * 2.0
    assert row["lambda"] == pytest.approx(lam)
    assert row["offline_residual_se"] == pytest.approx((2.05 - line) / 0.1)
    assert row["offline_residual_se"] > driver.OFFLINE_ENVELOPE_SE
    assert row["p2_within_envelope"] is False


def test_attribution_rows_end_to_end_synthetic(driver):
    """The full row builder on a synthetic three-entry register: the
    deltas, the predicate, the route-contrast records, and the parent
    block."""
    census = {"0": 0.22357, "1": 0.1936659375, "2": 0.253850625}
    entries = [
        {
            "species": "proton", "bin_index": 7, "parent_index": 1,
            "parent_sigma": 2.5,
            "confirmation_values": [0.010, 0.008, 0.012],
            "confirmation_unit_se": 0.001,
            "confirmation_sigma": 4.0,
        },
        {
            "species": "k_plus", "bin_index": 2, "parent_index": 2,
            "parent_sigma": 3.1,
            "confirmation_values": [0.003, 0.004, 0.002],
            "confirmation_unit_se": 0.00025,
            "confirmation_sigma": 8.0,
        },
    ]
    parent_rows = {
        ("proton", 7): {
            "values": [0.011, 0.009, 0.013], "unit_se": 0.0012,
            "sigma": 3.33,
        },
    }
    rows = driver.attribution_rows(entries, census, parent_rows)
    assert len(rows) == 2
    proton = rows[0]
    # Deltas against the median variant (the sealed register convention):
    # the median of [0.010, 0.008, 0.012] is index 0.
    assert proton["median_variant"] == 0
    assert proton["deltas_vs_median_se_units"]["1"] == pytest.approx(-2.0)
    assert proton["deltas_vs_median_se_units"]["0"] == pytest.approx(0.0)
    assert proton["deltas_vs_median_se_units"]["2"] == pytest.approx(2.0)
    assert proton["p1_monotone"] is True
    assert proton["delta_direction"] == "increasing_with_ramp_start"
    assert proton["route_dominance"] == "catalog_dominant"
    assert proton["placement_explained"] is True
    assert proton["implied_route_contrast"] == pytest.approx(
        0.004 / (census["2"] - census["1"])
    )
    assert proton["parent_production"]["sigma"] == 3.33
    kplus = rows[1]
    assert kplus["p1_monotone"] is True
    assert kplus["route_dominance"] == "continuum_dominant"
    assert kplus["strange_species"] is True
    assert proton["strange_species"] is False


# ----------------------------------------------------------------------
# The width reweighting algebra (hand-computed cases)
# ----------------------------------------------------------------------


def test_census_slopes_hand_case(driver):
    # Flat census scan: c = 0.20 / 0.25 / 0.30 at B - 0.2 / B / B + 0.2.
    census = {"0": 0.25, "1": 0.20, "2": 0.30}
    slopes = driver.census_slopes(census)
    assert slopes["1"] == pytest.approx(0.25)
    assert slopes["0"] == pytest.approx(0.25)
    assert slopes["2"] == pytest.approx(0.25)
    # Differential scan: the later band carries the steeper slope.
    census_d = {"0": 0.25, "1": 0.19, "2": 0.30}
    slopes_d = driver.census_slopes(census_d)
    assert slopes_d["1"] == pytest.approx(0.30)
    assert slopes_d["0"] == pytest.approx(0.275)
    assert slopes_d["2"] == pytest.approx(0.25)


def test_reweighted_census_hand_case(driver):
    census = {"0": 0.25, "1": 0.19, "2": 0.30}
    slopes = {"1": 0.30, "0": 0.275, "2": 0.25}
    # W' = 0.7 GeV: the shift is +0.2 * 0.5 * s_v.
    prime = driver.reweighted_census(census, slopes, 0.7)
    assert prime["1"] == pytest.approx(0.19 + 0.2 * 0.5 * 0.30)
    assert prime["0"] == pytest.approx(0.25 + 0.2 * 0.5 * 0.275)
    assert prime["2"] == pytest.approx(0.30 + 0.2 * 0.5 * 0.25)
    # W' = 0.3 GeV mirrors the sign.
    prime_narrow = driver.reweighted_census(census, slopes, 0.3)
    assert prime_narrow["1"] == pytest.approx(0.19 - 0.2 * 0.5 * 0.30)
    assert prime_narrow["0"] == pytest.approx(0.25 + (-0.2) * 0.5 * 0.275)
    # The alpha bracket endpoints: alpha = 0 leaves the census unchanged.
    flat = driver.reweighted_census(census, slopes, 0.7, 0.0)
    assert flat == census
    # alpha = 1 doubles the neutral shift.
    full = driver.reweighted_census(census, slopes, 0.7, 1.0)
    assert full["1"] == pytest.approx(0.19 + 0.2 * 0.30)


def test_reweight_bin_row_hand_case(driver):
    """The value-space reweighting against a hand computation.

    Differential case: census c = (c0, c1, c2) = (0.26, 0.20, 0.30)
    gives the slopes s = (s0, s1, s2) = (0.25, 0.30, 0.20); W' = 0.7 GeV
    gives the census shifts (dc_v) = 0.1 * s_v = (+0.025, +0.030,
    +0.020).  The bin values (y_v0, y_v1, y_v2) = (2.0, 1.0, 3.0) give
    the route contrast m_b = (3.0 - 1.0) / (0.30 - 0.20) = 20.  The
    reweighted values are y'_v = y_v + m_b * dc_v = (2.50, 1.60, 3.40);
    the reweighted spread is 1.80 = m_b * (dc + dc_v2 - dc_v1) =
    20 * 0.09, scaling the sealed spread 2.00 = m_b * 0.10 by the
    census-spread ratio 0.90."""
    census = {"0": 0.26, "1": 0.20, "2": 0.30}
    slopes = driver.census_slopes(census)
    assert slopes == {"0": pytest.approx(0.25), "1": pytest.approx(0.30),
                      "2": pytest.approx(0.20)}
    census_prime = driver.reweighted_census(census, slopes, 0.7)
    block = driver.reweight_bin_row(
        [2.0, 1.0, 3.0], 0.1, slopes, census, census_prime,
    )
    assert block["route_contrast"] == pytest.approx(20.0)
    assert block["reweighted_values"] == pytest.approx([2.50, 1.60, 3.40])
    assert block["reweighted_spread"] == pytest.approx(1.80)
    assert block["reweighted_sigma"] == pytest.approx(18.0)
    assert block["reweighted_spread"] == pytest.approx(
        block["route_contrast"] * (census_prime["2"] - census_prime["1"])
    )
    # The flat-slope case: W' = 0.3 shifts every variant equally, the
    # census spread is unchanged, and the spread scales by exactly 1.
    census_flat = {"0": 0.25, "1": 0.20, "2": 0.30}
    slopes_flat = driver.census_slopes(census_flat)
    prime_flat = driver.reweighted_census(census_flat, slopes_flat, 0.3)
    block = driver.reweight_bin_row(
        [2.0, 1.0, 3.0], 0.1, slopes_flat, census_flat, prime_flat,
    )
    assert block["reweighted_values"] == pytest.approx([1.5, 0.5, 2.5])
    assert block["reweighted_spread"] == pytest.approx(2.0)
    assert block["reweighted_sigma"] == pytest.approx(20.0)


def test_reweight_statistics_p95_hand_case(driver):
    """A synthetic 20-bin report of monotone bins: the reweighted p95
    equals the sealed p95 scaled by the census-spread ratio, and the
    per-bin value-space reweighting reproduces the ratio exactly."""
    census = {"0": 0.25, "1": 0.20, "2": 0.30}
    slopes = driver.census_slopes(census)
    rows = []
    for i in range(20):
        base = 0.010 + 0.001 * i
        spread = 0.002 + 0.0001 * i
        se = spread / (2.0 + 0.1 * i)
        rows.append({
            "species": "pi_minus",
            "bin_index": i,
            "status": "pass",
            # Monotone in the ramp start: y(v1) < y(v0) < y(v2).
            "values": [base, base - spread / 2.0, base + spread / 2.0],
            "unit_se": se,
            "sigma": spread / se,
        })
    report = {
        "bin_checks": rows,
        "summary": {"bin_spread_p95": driver.percentile95(
            [r["sigma"] for r in rows])},
    }
    sealed_p95 = report["summary"]["bin_spread_p95"]
    # Flat slopes: the reweighted census spread is unchanged, so the
    # reweighted p95 equals the sealed p95 exactly (the value-space
    # reweighting shifts every variant by the same amount).
    block = driver.reweight_statistics(report, slopes, census, 0.7)
    assert block["census_spread_ratio"] == pytest.approx(1.0)
    assert block["reweighted_bin_spread_p95"] == pytest.approx(sealed_p95)
    assert block["bins_reweighted"] == 20
    assert block["bins_vacuous_unchanged"] == 0
    # A differential census scan scales every monotone bin's sigma by
    # the reweighted census-spread ratio, so the p95 follows exactly.
    census_d = {"0": 0.25, "1": 0.19, "2": 0.30}
    slopes_d = driver.census_slopes(census_d)
    prime = driver.reweighted_census(census_d, slopes_d, 0.7)
    ratio = (prime["2"] - prime["1"]) / (
        census_d["2"] - census_d["1"]
    )
    # Hand-computed: dc = (+0.0275, +0.030, +0.025); the reweighted
    # census spread is 0.11 + 0.025 - 0.030 = 0.105 over the sealed
    # 0.11, so R = 0.105/0.11.
    assert ratio == pytest.approx(0.105 / 0.11)
    block = driver.reweight_statistics(report, slopes_d, census_d, 0.7)
    assert block["census_spread_ratio"] == pytest.approx(ratio)
    assert block["reweighted_bin_spread_p95"] == pytest.approx(
        sealed_p95 * ratio
    )
    assert block["uniform_ratio_cross_check_p95"] == pytest.approx(
        sealed_p95 * ratio
    )
    assert block["bins_reweighted"] == 20
    assert block["bins_vacuous_unchanged"] == 0
    # Vacuous rows are carried unchanged and excluded from the p95.
    rows_vacuous = [dict(r, status="vacuous_empty_bin") for r in rows]
    report_v = {"bin_checks": rows_vacuous, "summary": {"bin_spread_p95": 0.0}}
    block = driver.reweight_statistics(
        report_v, slopes, census, 0.7,
    )
    assert block["bins_reweighted"] == 0
    assert block["bins_vacuous_unchanged"] == 20
    assert block["reweighted_bin_spread_p95"] == 0.0


# ----------------------------------------------------------------------
# The verdict criteria (the firing paths)
# ----------------------------------------------------------------------


def _synthetic_artifacts(driver, findings=(), width_adequate=True,
                         census_within=True):
    attribution = {
        "non_strange": {
            "bins": 14,
            "placement_explained": 14 - len(list(findings)),
            "findings": list(findings),
        },
        "rows": [],
        "probe_set": {"entries": 32, "species_split": {}, "source": "x"},
    }
    reweight = {
        "contract": {
            "width_adequate": width_adequate,
            "per_width_prime": {"0.3": True, "0.7": True},
        },
        "production": {"width_variants": []},
    }
    census = {
        "max_disagreement": 0.0 if census_within else 1e-3,
        "disagreement_bound": 1e-6,
        "within_bound": census_within,
    }
    return attribution, reweight, census


def test_evaluate_verdict_criteria_paths(driver):
    finding = {
        "species": "pi_plus", "bin_index": 173,
        "deltas_ramp_start_order": [-2.2, 0.45, 0.0],
        "confirmation_sigma": 2.665,
        "offline_residual_se": 1.56,
        "parent_production": {"sigma": 2.1},
        "confirmation_values": [0.007, 0.0055, 0.0065],
        "confirmation_unit_se": 0.0006,
        "deltas_vs_median_se_units": {"0": 0.45, "1": -2.2, "2": 0.0},
        "p1_monotone": False,
        "p2_within_envelope": True,
    }
    # Criterion (a): a non-strange finding fires; the verdict is a
    # RECOMMENDATION.
    attribution, reweight, census = _synthetic_artifacts(
        driver, findings=[finding],
    )
    evaluation = driver.evaluate_verdict_criteria(
        attribution, reweight, census,
    )
    assert evaluation["verdict"] == "RECOMMENDATION"
    assert [c["criterion"] for c in evaluation["fired"]] == ["a"]
    # Criterion (b): the width contract fails.
    attribution, reweight, census = _synthetic_artifacts(
        driver, width_adequate=False,
    )
    evaluation = driver.evaluate_verdict_criteria(
        attribution, reweight, census,
    )
    assert evaluation["verdict"] == "RECOMMENDATION"
    assert [c["criterion"] for c in evaluation["fired"]] == ["b"]
    # Criterion (c): the census recomputation disagrees.
    attribution, reweight, census = _synthetic_artifacts(
        driver, census_within=False,
    )
    evaluation = driver.evaluate_verdict_criteria(
        attribution, reweight, census,
    )
    assert evaluation["verdict"] == "RECOMMENDATION"
    assert [c["criterion"] for c in evaluation["fired"]] == ["c"]
    # All clean: the PASS path (the K+ referral stands independently).
    attribution, reweight, census = _synthetic_artifacts(driver)
    evaluation = driver.evaluate_verdict_criteria(
        attribution, reweight, census,
    )
    assert evaluation["verdict"] == "PASS"
    assert evaluation["fired"] == []


def test_finding_narrative(driver):
    row = {
        "species": "pi_minus",
        "bin_index": 67,
        "deltas_ramp_start_order": [0.0, -0.4, 1.97],
        "deltas_vs_median_se_units": {"0": -0.4, "1": 0.0, "2": 1.97},
        "confirmation_values": [0.0133, 0.0134, 0.0158],
        "confirmation_unit_se": 0.00102,
        "confirmation_sigma": 2.367,
        "offline_residual_se": -1.377,
        "p1_monotone": False,
        "p2_within_envelope": True,
        "parent_production": {"sigma": 2.367},
    }
    finding = driver._finding_narrative(row)
    assert finding["kind"] == "bin"
    assert finding["species"] == "pi_minus"
    assert finding["bin_index"] == 67
    assert "falls to the nominal variant and rises to the late variant" in (
        finding["observed"]
    )
    assert "+0.000" in finding["observed"]
    assert "-1.377" in finding["observed"]
    assert finding["responsible"].startswith(
        "the card-6 blend routing formulation"
    )
    assert CARD6_SHA256 in finding["responsible"]
    assert "rework card" in finding["proposed_rework"]
    assert finding["evidence"]["p1_monotone"] is False
    # The mirrored shape for a rises-then-falls pattern.
    row["deltas_ramp_start_order"] = [-2.22, 0.45, 0.0]
    finding = driver._finding_narrative(row)
    assert "rises to the nominal variant and falls to the late variant" in (
        finding["observed"]
    )


# ----------------------------------------------------------------------
# Sealed-archive read-only pins
# ----------------------------------------------------------------------


def test_sealed_r32_readonly_pins(driver):
    # The two probe archives verify their seals (read-only).
    skip_unless_sealed(driver.MONO_ROOT / driver.R32_PROD_DIR)
    prod = driver.verify_sealed_dir(driver.R32_PROD_DIR)
    conf = driver.verify_sealed_dir(driver.R32_CONF_DIR)
    assert prod["verified"] is True and prod["entries"] == 29
    assert conf["verified"] is True and conf["entries"] == 55
    # The sealed anchor digests re-hash.
    verified = driver.verify_sealed_inputs()
    assert verified == driver.SEALED_INPUT_DIGESTS
    # The persisting register: exactly 32 entries with the
    # pre-registered species split and sigma range.
    cv = driver._read_json(
        driver.MONO_ROOT / driver.R32_CONF_DIR
        / driver.CONFIRMATION_VERIFICATION_NAME
    )
    persisting = [
        e for e in cv["residual_reclassification"]
        if e.get("classification") == "persisting"
    ]
    assert len(persisting) == driver.PERSISTING_TOTAL
    counts = {}
    for entry in persisting:
        counts[entry["species"]] = counts.get(entry["species"], 0) + 1
    assert counts == driver.PERSISTING_SPECIES_SPLIT
    sigmas = [float(e["confirmation_sigma"]) for e in persisting]
    assert min(sigmas) >= (driver.PERSISTING_SIGMA_MIN
                           - driver.PERSISTING_SIGMA_TOLERANCE)
    assert max(sigmas) <= (driver.PERSISTING_SIGMA_MAX
                           + driver.PERSISTING_SIGMA_TOLERANCE)
    # The sealed anchors: the p95 values and the routing census.
    report = driver._read_json(
        driver.MONO_ROOT / driver.R32_PROD_DIR / driver.SPREAD_REPORT_NAME
    )
    gate = driver._read_json(
        driver.MONO_ROOT / driver.R32_PROD_DIR / driver.GATE_RECORD_NAME
    )
    assert float(report["summary"]["bin_spread_p95"]) == (
        driver.PRODUCTION_P95
    )
    assert float(cv["summary"]["bin_spread_p95"]) == (
        driver.CONFIRMATION_P95
    )
    for v in ("0", "1", "2"):
        assert float(
            report["routing_census"][v]["fraction_catalog"]
        ) == driver.PRODUCTION_CENSUS_CATALOG[v]
        assert float(
            cv["routing_census"][v]["fraction_catalog"]
        ) == driver.CONFIRMATION_CENSUS_CATALOG[v]
        assert float(
            gate["clauses"]["no_boundary_dominance"]
            ["routing_census"][f"v{v}"]
        ) == driver.PRODUCTION_CENSUS_CATALOG[v]
    # The gate record carries the failing spread (the audit's input
    # state): production p95 2.960499 against the bound 2.0.
    clause = gate["clauses"]["no_boundary_dominance"]
    assert clause["bin_spread_p95_abs_se"] == driver.PRODUCTION_P95
    assert clause["bin_spread_bound"] == driver.BIN_SPREAD_BOUND
    assert clause["feeddown_spread_max_sigma"] == pytest.approx(
        23.667769190391247
    )
    assert clause["pass"] is False


def test_card6_surface_byte_stability(driver):
    assert driver.substrate_digests()["card6_surface_sha256"] == CARD6_SHA256
    digests = driver.substrate_digests()
    assert digests["r32_driver_nested_sha256"] == (
        digests["r32_driver_outer_sha256"]
    )


# ----------------------------------------------------------------------
# The freeze-manifest re-pin
# ----------------------------------------------------------------------


def test_freeze_manifest_repin():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    inventory = manifest["source_tree_sha256"]
    # The strict path-sorted inventory rule: the recorded keys are
    # sorted by relative path.
    keys = list(manifest["source_tree_sha256"].keys())
    assert keys == sorted(keys)
    # The three new inventory entries carry the on-disk digests.
    new_entries = (
        "dev/mixing2006/helpers/benchmarking/audit_r4_routing_formulation.py",
        "dev/mixing2006/tests/test_audit_r4_routing_formulation.py",
        "helpers/benchmarking/audit_r4_routing_formulation.py",
    )
    for rel in new_entries:
        assert rel in keys, f"missing re-pin entry: {rel}"
        path = NESTED_ROOT.parent.parent / rel
        assert manifest["source_tree_sha256"][rel] == _sha256_file(path)
    # The nested and outer driver entries share the digest.
    assert manifest["source_tree_sha256"][
        "dev/mixing2006/helpers/benchmarking/audit_r4_routing_formulation.py"
    ] == manifest["source_tree_sha256"][
        "helpers/benchmarking/audit_r4_routing_formulation.py"
    ]
    # The freeze exception names this card with the audit's files.
    exceptions = manifest["freeze_exceptions"]
    entries = [e for e in exceptions if e["card"] == (
        "TASK-20260920-r4-audit-transition-routing-formulation"
    )]
    assert len(entries) == 1
    files = entries[0]["files"]
    for rel in (
        "dev/mixing2006/helpers/benchmarking/audit_r4_routing_formulation.py",
        "helpers/benchmarking/audit_r4_routing_formulation.py",
        "dev/mixing2006/tests/test_audit_r4_routing_formulation.py",
        "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json",
    ):
        assert rel in files
    # Every pinned frozen file stays byte-unchanged: the pinned M8 map
    # is intact (the freeze test itself enforces the full inventory).
    assert manifest["frozen_fit_outputs"]["m8_frozen_pp_parameters"][
        "continuum.match_mass"
    ] == 2.680194110727528


# ----------------------------------------------------------------------
# The sealed audit run dir's artifact pins (activated at phase 2)
# ----------------------------------------------------------------------


def _sealed_run_dir():
    if not RUN_DIR_CANDIDATES:
        return None
    return RUN_DIR_CANDIDATES[0]


def test_audit_run_artifact_pins(driver):
    run_dir = _sealed_run_dir()
    if run_dir is None:
        pytest.skip(
            "the audit run dir (dev/00runs/r4_audit_transition_routing_"
            "formulation_*) is absent; the artifact pins activate at "
            "phase 2 (the audit run)"
        )
    run_dir = Path(run_dir)
    # -- the seal and the sums (read-only re-hash from the root) --------
    seal = json.loads((run_dir / "seal_manifest.json").read_text(
        encoding="utf-8"))
    assert seal["card"] == driver.CARD
    listed = {entry["path"] for entry in seal["files"]}
    verified = 0
    sums_names = set()
    for line in (run_dir / "SHA256SUMS.txt").read_text(
            encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        payload = run_dir.parent.parent.parent / name
        assert payload.is_file(), name
        assert _sha256_file(payload) == digest, name
        verified += 1
        sums_names.add(Path(name).name)
    # The sums cover the sealed payloads (6 at this run's shape); the
    # seal covers them plus the sums file itself at seal time.
    assert verified == len(sums_names) == len(listed) - 1
    # The seal covers every regular file except itself at seal time; the
    # clean-environment record is the documented post-seal file (the R31
    # convention) and is never seal-covered.
    regular = {
        p.name for p in run_dir.rglob("*")
        if p.is_file() and not p.is_symlink()
    }
    assert regular - listed == {
        "seal_manifest.json", "clean_environment_commands.json",
    }
    # -- the pre-registration -------------------------------------------
    pre = json.loads((run_dir / "pre_registration.json").read_text(
        encoding="utf-8"))
    assert pre["card"] == driver.CARD
    assert pre["written_before_measurement"] is True
    assert pre["probe_set"]["expected_entries"] == 32
    assert pre["zero_draw"]["draws"] == 0
    assert pre["zero_draw"]["no_fitting"] is True
    assert pre["zero_draw"]["optimizer_invoked"] is False
    assert pre["zero_draw"]["reserved_audit_block"]["fired"] is False
    assert pre["surface"]["card6_blend_routing"]["sha256"] == CARD6_SHA256
    assert pre["width_sensitivity"]["width_primes_gev"] == [0.3, 0.7]
    assert pre["width_sensitivity"]["contract_relative"] == 0.20
    assert pre["census_recompute"]["disagreement_bound"] == 1e-6
    # -- the provenance --------------------------------------------------
    prov = json.loads((run_dir / "provenance.json").read_text(
        encoding="utf-8"))
    assert prov["zero_draw"]["draws"] == 0
    assert prov["zero_draw"]["no_fitting"] is True
    assert prov["zero_draw"]["optimizer_invoked"] is False
    assert prov["zero_draw"]["reserved_audit_block_fired"] is False
    assert prov["driver_sha256"]["nested"] == prov["driver_sha256"]["outer"]
    # -- the census recomputation ---------------------------------------
    census = json.loads((run_dir / "routing_census_recomputed.json")
                        .read_text(encoding="utf-8"))
    assert census["within_bound"] is True
    assert census["max_disagreement"] <= driver.CENSUS_DISAGREEMENT_BOUND
    assert census["draws"] == 0 and census["no_fitting"] is True
    # -- the attribution -------------------------------------------------
    attribution = json.loads((run_dir / "persisting_bin_attribution.json")
                             .read_text(encoding="utf-8"))
    assert attribution["probe_set"]["entries"] == 32
    assert attribution["probe_set"]["species_split"] == (
        driver.PERSISTING_SPECIES_SPLIT
    )
    rows = attribution["rows"]
    assert len(rows) == 32
    assert sum(1 for r in rows if r["p1_monotone"]) == 27
    assert sum(1 for r in rows if r["p2_within_envelope"]) == 32
    assert sum(1 for r in rows if r["placement_explained"]) == 27
    findings = attribution["non_strange"]["findings"]
    assert {(f["species"], f["bin_index"]) for f in findings} == {
        ("pi_minus", 67), ("pi_plus", 173),
    }
    kplus = attribution["k_plus_referral"]
    assert kplus["bins"] == 18
    assert kplus["monotone"] == 15
    assert kplus["within_census_envelope"] == 18
    assert kplus["referred_to_card"] == (
        "TASK-20260920-r4-audit-strange-sector-continuum-families"
    )
    # -- the width reweighting -------------------------------------------
    reweight = json.loads((run_dir / "spread_reweighted_W.json")
                          .read_text(encoding="utf-8"))
    assert reweight["contract"]["width_adequate"] is True
    for block in reweight["production"]["width_variants"]:
        assert block["contract"]["within_contract"] is True
        assert block["contract"]["alpha_bracket_within_contract"] is True
        assert block["contract"]["relative_deviation"] <= 0.20
    # -- the verdict ------------------------------------------------------
    verdict = json.loads((run_dir / "verdict.json").read_text(
        encoding="utf-8"))
    assert verdict["verdict"] == "RECOMMENDATION"
    assert verdict["criteria"]["a"]["fired"] is True
    assert verdict["criteria"]["b"]["fired"] is False
    assert verdict["criteria"]["c"]["fired"] is False
    named = {(f["kind"], f["species"], f["bin_index"])
             for f in verdict["findings"]}
    assert named == {("bin", "pi_minus", 67), ("bin", "pi_plus", 173)}
    assert verdict["recommendation"] and "rework card" in verdict[
        "recommendation"]
    assert verdict["bounds"]["bin_spread_p95_bound"] == 2.0
    assert verdict["bounds"]["feeddown_spread_max_bound"] == 3.0
    assert verdict["bounds"]["census_disagreement_bound"] == 1e-6
    assert verdict["zero_draw"]["draws"] == 0
    assert verdict["surface"]["byte_unchanged_by_this_audit"] is True
    assert verdict["surface"]["card6_sha256"] == CARD6_SHA256
    # -- the clean-environment record -------------------------------------
    clean = json.loads((run_dir / "clean_environment_commands.json")
                       .read_text(encoding="utf-8"))
    assert clean["hashes_match"] is True
    assert clean["seal_verification"]["mismatches"] == []
    assert clean["probe_returncodes"] == {
        "census": 0, "attribution": 0, "reweight": 0, "verdict": 0,
    }
    assert clean["commands"] and len(clean["commands"]) == 4