"""Round 4 card 1: transition routing formulation audit contract pins
(test-author register).

Task ``TASK-20260920-r4-audit-transition-routing-formulation`` (Gate 3
decision card follow-up obligation 1; plan section 9.3, lines 625-632).
This is the test-author's independent pin register for the zero-draw,
hermetic dual-tree audit driver
``helpers/benchmarking/audit_r4_routing_formulation.py`` (nested commit
``5bd4800``, outer ``584193b``; byte-identical in both trees at sha256
``ef6c162e9e4ac82b9701af8311d262eda41fd72d0a9c19b5d4dac23f9320b838``)
and for the sealed audit run dir
``dev/00runs/r4_audit_transition_routing_formulation_2026-09-20/``
(verdict RECOMMENDATION, two named findings: pi- bin 67 and pi+ bin
173).  It is an independent cross-check of the solver's register in
``tests/test_audit_r4_routing_formulation.py``: the same pre-registered
measurement contract, pinned from an independently authored test with
its own hand-built delta fixtures, slope/census fixtures, and synthetic
artifacts (no solver fixture values reused).

Part 1 (the probe contract, pure, no sealed run dir needed):

- ``test_placement_predicate_p1_monotone_boundaries``: the (P1)
  monotone-delta predicate on hand-built delta sequences in the ramp
  start order (v1 < v0 < v2), including the exact zero-delta boundary
  cases (a flat step inside a monotone sequence, and the all-zero
  sequence).
- ``test_placement_predicate_p2_envelope_boundary``: the (P2)
  census-line residual against the 4.0-SE finite-MC envelope (the R32
  register precedent) on hand-built three-variant payloads, including
  the exact 4.0-SE boundary on both signs (inclusive) and the
  just-outside cases.
- ``test_width_reweighting_census_algebra``: the per-variant
  census-slope reweighting ``c_v' = c_v + (W' - W) alpha s_v`` for
  ``W' in {0.3, 0.7}`` and the alpha bracket ``{0, 1}`` on the
  test-author's own slope/census fixtures, including the flat-slope
  identity case (the census spread is invariant).
- ``test_width_reweighting_value_space_and_p95``: the per-bin value
  space reweighting ``y_v' = y_v + m_b (c_v' - c_v)`` against a
  hand-computed differential case, the flat-slope identity on a
  synthetic bin set, and the p95 convention (the sealed R32 numpy
  linear-interpolation percentile over the finite reweighted sigmas).
- ``test_width_contract_20_percent_pass_fail``: the 20% width contract
  ``|p95' - anchor| <= 20% of the anchor`` on hand fixtures, the pass
  side (zero and ~7% relative deviations through the driver's own
  algebra) and the fail side (a 50% reweighted shift).
- ``test_census_disagreement_exact_boundary``: the 1e-6 census
  disagreement bound on hand-computed fixtures, with the exact 1e-6
  boundary (inclusive, verdict criterion (c) does not fire) and the
  1.1e-6 fail side.
- ``test_verdict_criteria_logic``: the verdict-criteria logic (PASS iff
  the non-strange bins are placement-explained and nothing else fires;
  RECOMMENDATION on (a) a non-strange finding, (b) the width contract
  failing, (c) the census bound failing) on synthetic artifacts, with
  the K+ referral recorded independently of the verdict.
- ``test_zero_draw_rng_surface_never_touched``: the zero-draw
  invariant, test-author technique (an independent probe, not the
  solver's source scan alone): every numpy RNG surface entry point is
  replaced by a sentinel that raises on attribute access, and the
  driver's full pure surface is exercised under it; a source scan with
  the test-author's own token list over both driver trees confirms no
  random-number machinery.

Part 2 (the sealed artifacts, run-dir pins; the run dir is gitignored,
the pins activate once the dir exists and skip with a recorded reason
on a fresh checkout):

- ``test_driver_dual_tree_byte_identity``: the nested and outer audit
  driver copies are byte-identical at the pinned digest.
- ``test_sealed_run_dir_seal_pins``: the seal manifest (7 files,
  95,594 bytes, per the on-disk ``seal_manifest.json``), the full
  per-file digest listing self-consistent against disk, and the
  ``SHA256SUMS.txt`` re-hash integrity (6 monorepo-root-relative
  entries).
- ``test_sealed_run_dir_verdict_pins``: the ``verdict.json`` content
  pins (verdict RECOMMENDATION, criteria fired [a], the two named
  findings pi- bin 67 and pi+ bin 173 with their delta signatures, the
  K+ referral to card 2, draws 0, no_fitting, optimizer_invoked, the
  census max_disagreement 0.0, and the reweighted p95 values with
  their relative deviations pinned with tolerance).
- ``test_sealed_run_dir_reweight_pins``: the
  ``spread_reweighted_W.json`` pins (the W' p95 values and relative
  deviations, the alpha-bracket endpoints, the contract record, the
  carried 2.0 bound).
- ``test_sealed_run_dir_measurement_artifact_pins``: the census and
  attribution artifact digest pins and structure pins (32 entries,
  the species split, the sigma range, the non-strange findings, the
  K+ referral counts, the zero-draw stamps).
- ``test_sealed_run_dir_prereg_provenance_pins``: the
  ``pre_registration.json`` sha256 pin (immutability), the seed
  block-not-fired statement, and the ``provenance.json`` driver digest
  pin (nested == outer == the on-disk dual-tree digest).
- ``test_card6_surface_and_builder_byte_stability``: the card-6 blend
  routing surface and the builder stay at their pinned on-disk
  digests.
- ``test_freeze_repin_register``: the legacy_v1 manifest re-pin for
  this register (the inventory entry at the on-disk digest, the strict
  path-sorted inventory, and the single ``freeze_exceptions`` entry
  naming this card carrying the audit's files plus this register's
  file, path-sorted).

Hermetic: standard library, numpy, and the committed audit driver; the
sealed archives are read read-only; no writes outside the process, no
event generation, no random numbers, no optimizer.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

NESTED_ROOT = Path(__file__).resolve().parent.parent
MONO_ROOT = NESTED_ROOT.parent.parent
DRIVER_REL = Path("helpers") / "benchmarking" / "audit_r4_routing_formulation.py"
NESTED_DRIVER = NESTED_ROOT / DRIVER_REL
OUTER_DRIVER = MONO_ROOT / DRIVER_REL
MANIFEST_PATH = NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json"

# The pinned dual-tree driver digest (nested 5bd4800, outer 584193b).
AUDIT_DRIVER_SHA256 = (
    "ef6c162e9e4ac82b9701af8311d262eda41fd72d0a9c19b5d4dac23f9320b838"
)
# The byte-stable card-6 surface and the builder blend path.
CARD6_SHA256 = (
    "561c65ee9a77a1b7f7eb07a7619dafb584779a51ccb23d7fe5651c0605e12563"
)
BUILDER_SHA256 = (
    "4ab612609e1ee0bc1e037dcbb2815b6484306ad32402b7289e6636f8e829cdb1"
)
# The strange-sector referral (Round 4 card 2).
CARD2_ID = "TASK-20260920-r4-audit-strange-sector-continuum-families"
CARD_ID = "TASK-20260920-r4-audit-transition-routing-formulation"

# The sealed audit run dir (gitignored; the pins activate once it
# exists, the R32 pins convention).
RUN_DIR_CANDIDATES = tuple(
    (MONO_ROOT / "dev" / "00runs").glob(
        "r4_audit_transition_routing_formulation_*"
    )
)

# The sealed run dir's file digests (on-disk, 2026-09-20).
SEAL_FILES = {
    "SHA256SUMS.txt": (
        "c5e03e9266e68bc4f4f4ed8a41bdd57dc4c29242cadcd424a79a1b292c60345d",
        907,
    ),
    "persisting_bin_attribution.json": (
        "9485af85c8e848b349038bd8c66f516037388b597cc892929cb6e5b9f97a5607",
        56708,
    ),
    "pre_registration.json": (
        "26c8f44975ff8f8fa37290e75ac16e01a67b9bfe728700840015f6c5fc8ff4a1",
        9221,
    ),
    "provenance.json": (
        "e1a45c8f142324a1532a8a08e05cf6cf8bcde882d8e5cfbd3686a78322c959f4",
        3089,
    ),
    "routing_census_recomputed.json": (
        "c8caa73dc47115bb890e1ab751f39d11b1bf388c7f229e182fbcb41a30167c9b",
        5440,
    ),
    "spread_reweighted_W.json": (
        "7a0787a8aea89e0e65a392c3609b71fbd7c0e6b1a092da9e47d4381702896294",
        6623,
    ),
    "verdict.json": (
        "b0c6c89d0737246143e92efb377a4fbc2bf650932677ea4cc599bbe029f6d139",
        13606,
    ),
}
SEAL_FILE_COUNT = 7
SEAL_TOTAL_BYTES = 95594
SEAL_MANIFEST_SHA256 = (
    "e0398b5aeb18781f7efc3b8bb976de494505635acb572883d46487a0492d25e1"
)
CLEAN_ENV_SHA256 = (
    "be96fc7cac64d4a1f514bd3d6da4adde8f8244f87b19a8540088da03c4216ebf"
)
# The reweighted bin-spread p95 values and their relative deviations
# against the production anchor 2.9604990756942544 (from the sealed
# verdict.json / spread_reweighted_W.json).
REWEIGHT_P95_W03 = 2.9453672609687778
REWEIGHT_P95_W07 = 2.969823676538042
REWEIGHT_RELDEV_W03 = 0.005111237780720517
REWEIGHT_RELDEV_W07 = 0.003149671932122062
# The two named findings' delta signatures (pooled-SE units, against
# the median-variant reference; ramp-start order (v1, v0, v2)).
FINDING_PI_MINUS_67 = {
    "0": -0.39924483621412904,
    "1": 0.0,
    "2": 1.9674813836398342,
}
FINDING_PI_MINUS_67_RESIDUAL = -1.3768304764810506
FINDING_PI_MINUS_67_SIGMA = 2.3667262198539634
FINDING_PI_PLUS_173 = {
    "0": 0.44715420497575314,
    "1": -2.217545314674526,
    "2": 0.0,
}
FINDING_PI_PLUS_173_RESIDUAL = 1.5628642116808036
FINDING_PI_PLUS_173_SIGMA = 2.664699519650279
PRODUCTION_P95 = 2.9604990756942544
PRE_REGISTRATION_SHA256 = SEAL_FILES["pre_registration.json"][0]
MY_TEST_REL = "dev/mixing2006/tests/test_r4_audit_routing_contract_pins.py"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_driver():
    """The committed dual-tree driver, loaded once under a unique
    module name (the driver's ``_locate_roots`` wiring resolves the
    nested roots at import time)."""
    name = "audit_r4_routing_formulation_test_author"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, NESTED_DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    return _load_driver()


def _sealed_run_dir():
    if not RUN_DIR_CANDIDATES:
        return None
    return Path(RUN_DIR_CANDIDATES[0])


def _skip_if_no_run_dir():
    run_dir = _sealed_run_dir()
    if run_dir is None:
        pytest.skip(
            "the sealed audit run dir "
            "(dev/00runs/r4_audit_transition_routing_formulation_*) is "
            "absent on this checkout; the sealed-artifact pins activate "
            "once the run dir exists"
        )
    return run_dir


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# Part 1: the probe contract (pure, hand-built fixtures)
# ----------------------------------------------------------------------
def test_placement_predicate_p1_monotone_boundaries(driver):
    """(P1) on the test-author's own delta sequences in the ramp start
    order (v1, v0, v2): strict and flat monotone sequences pass, the
    all-zero sequence and a zero-delta step inside a monotone sequence
    pass (non-strict), and the non-monotone shapes fail."""
    # Monotone decreasing with a zero delta (a flat late step).
    assert driver.is_monotone_in_ramp_start([2.0, 1.0, 1.0]) is True
    # Monotone increasing with a zero delta (a flat early step).
    assert driver.is_monotone_in_ramp_start([0.0, 0.0, 2.0]) is True
    # The all-zero delta sequence is monotone (both directions).
    assert driver.is_monotone_in_ramp_start([0.0, 0.0, 0.0]) is True
    # Strict shapes, both directions.
    assert driver.is_monotone_in_ramp_start([1.0, 0.0, -1.0]) is True
    assert driver.is_monotone_in_ramp_start([-1.0, 0.0, 1.0]) is True
    # The two non-monotone shapes (the nominal variant off the
    # early/late trend in either direction).
    assert driver.is_monotone_in_ramp_start([0.5, -0.5, 0.0]) is False
    assert driver.is_monotone_in_ramp_start([2.0, 1.0, 3.0]) is False
    assert driver.is_monotone_in_ramp_start([-1.0, 1.0, 0.0]) is False


def test_placement_predicate_p2_envelope_boundary(driver):
    """(P2) on the test-author's own three-variant payloads: the
    nominal variant's off-line residual from the census interpolation
    of the early and late variants, in pooled-SE units, against the
    4.0-SE finite-MC envelope (the R32 register precedent, inclusive).

    The census (c0, c1, c2) = (0.25, 0.22, 0.28) gives lambda = 0.5;
    with the early/late values (2.0, 7.0) the census line at the
    nominal variant is 4.5.  The payloads (5.5, 2.0, 7.0) at unit SE
    0.25 put the residual at exactly +4.0 SE (the boundary, inclusive)
    and (3.5, 2.0, 7.0) at -4.0 SE (the negative boundary); a further
    0.00025 nudge in either unit leaves the envelope."""
    census = {"0": 0.25, "1": 0.22, "2": 0.28}
    lam = (census["0"] - census["1"]) / (census["2"] - census["1"])
    assert lam == pytest.approx(0.5, abs=1e-15)
    # The exact +4.0-SE boundary (inclusive, with the driver's 1e-9
    # float-slop tolerance).
    r_se = driver.offline_residual_se([5.5, 2.0, 7.0], 0.25, lam)
    assert r_se == pytest.approx(4.0, abs=1e-9)
    assert abs(r_se) <= driver.OFFLINE_ENVELOPE_SE + (
        driver.OFFLINE_ENVELOPE_TOLERANCE
    )
    # The exact -4.0-SE boundary (the envelope is on the magnitude).
    r_se_neg = driver.offline_residual_se([3.5, 2.0, 7.0], 0.25, lam)
    assert r_se_neg == pytest.approx(-4.0, abs=1e-9)
    assert abs(r_se_neg) <= driver.OFFLINE_ENVELOPE_SE + (
        driver.OFFLINE_ENVELOPE_TOLERANCE
    )
    # Just outside the envelope on both signs.
    r_out = driver.offline_residual_se([5.50025, 2.0, 7.0], 0.25, lam)
    assert r_out == pytest.approx(4.001, abs=1e-12)
    assert abs(r_out) > driver.OFFLINE_ENVELOPE_SE + (
        driver.OFFLINE_ENVELOPE_TOLERANCE
    )
    r_out_neg = driver.offline_residual_se([3.49975, 2.0, 7.0], 0.25, lam)
    assert abs(r_out_neg) > driver.OFFLINE_ENVELOPE_SE + (
        driver.OFFLINE_ENVELOPE_TOLERANCE
    )
    # On the census line: the residual vanishes.
    assert driver.offline_residual_se([4.5, 2.0, 7.0], 0.25, lam) == (
        pytest.approx(0.0, abs=1e-12)
    )


# ----------------------------------------------------------------------
# The width reweighting algebra (hand-computed fixtures)
# ----------------------------------------------------------------------
def test_width_reweighting_census_algebra(driver):
    """The per-variant census-slope reweighting ``c_v' = c_v + (W' - W)
    alpha s_v`` on the test-author's own census fixture (c0, c1, c2) =
    (0.30, 0.22, 0.36) at W = 0.5 GeV, for W' in {0.3, 0.7} and the
    alpha bracket {0, 1}: the adjacent-pair finite-difference slopes
    (s1, s0, s2) = (0.4, 0.35, 0.3), the reweighted census values, the
    census-spread ratios, and the flat-slope identity case."""
    census = {"0": 0.30, "1": 0.22, "2": 0.36}
    slopes = driver.census_slopes(census)
    assert slopes["1"] == pytest.approx(0.4, rel=1e-15)
    assert slopes["0"] == pytest.approx(0.35, rel=1e-15)
    assert slopes["2"] == pytest.approx(0.3, rel=1e-15)
    spread = census["2"] - census["1"]
    # W' = 0.7 GeV, alpha = 0.5: the shifts are (W' - W) alpha s_v =
    # +0.1 * s_v -> (0.335, 0.26, 0.39); the spread ratio is 0.13/0.14.
    prime = driver.reweighted_census(census, slopes, 0.7, 0.5)
    assert prime["0"] == pytest.approx(0.335, rel=1e-15)
    assert prime["1"] == pytest.approx(0.26, rel=1e-15)
    assert prime["2"] == pytest.approx(0.39, rel=1e-15)
    assert (prime["2"] - prime["1"]) / spread == pytest.approx(13 / 14)
    # W' = 0.3 GeV mirrors the sign: (0.265, 0.18, 0.33), ratio 15/14.
    prime_narrow = driver.reweighted_census(census, slopes, 0.3, 0.5)
    assert prime_narrow["0"] == pytest.approx(0.265, rel=1e-15)
    assert prime_narrow["1"] == pytest.approx(0.18, rel=1e-15)
    assert prime_narrow["2"] == pytest.approx(0.33, rel=1e-15)
    assert (prime_narrow["2"] - prime_narrow["1"]) / spread == (
        pytest.approx(15 / 14)
    )
    # The alpha bracket endpoint alpha = 0 leaves the census
    # unchanged (the identity case) at both widths.
    for width_prime in (0.3, 0.7):
        identity = driver.reweighted_census(census, slopes, width_prime,
                                             0.0)
        assert identity == census
    # The alpha = 1 endpoint doubles the neutral shift: W' = 0.7 gives
    # (0.37, 0.30, 0.42), ratio 12/14 = 6/7.
    full = driver.reweighted_census(census, slopes, 0.7, 1.0)
    assert full["0"] == pytest.approx(0.37, rel=1e-15)
    assert full["1"] == pytest.approx(0.30, rel=1e-15)
    assert full["2"] == pytest.approx(0.42, rel=1e-15)
    assert (full["2"] - full["1"]) / spread == pytest.approx(6 / 7)
    # The flat-slope identity: a linear census scan (all slopes 0.25)
    # shifts every variant by the same amount, so the census spread is
    # invariant at every W' and alpha (the ratio is exactly 1 up to
    # float rounding).
    census_flat = {"0": 0.25, "1": 0.20, "2": 0.30}
    slopes_flat = driver.census_slopes(census_flat)
    assert slopes_flat["0"] == pytest.approx(0.25, rel=1e-15)
    assert slopes_flat["1"] == pytest.approx(0.25, rel=1e-15)
    assert slopes_flat["2"] == pytest.approx(0.25, rel=1e-15)
    for width_prime in (0.3, 0.7):
        for alpha in (0.5, 1.0):
            prime_flat = driver.reweighted_census(
                census_flat, slopes_flat, width_prime, alpha,
            )
            ratio_flat = (prime_flat["2"] - prime_flat["1"]) / (
                census_flat["2"] - census_flat["1"]
            )
            assert ratio_flat == pytest.approx(1.0, rel=1e-12)


def _ta_bin_report():
    """The test-author's synthetic 4-bin report: monotone bins (v1 < v0
    < v2) with the sealed sigmas (1.0, 2.0, 3.0, 4.0), so the sealed
    p95 is the hand value 3.85 and every reweighted sigma scales by
    the census-spread ratio exactly."""
    rows = []
    for i, (base, spread, sigma) in enumerate(
        ((1.0, 0.5, 1.0), (2.0, 1.0, 2.0), (3.0, 1.5, 3.0),
         (4.0, 2.0, 4.0))
    ):
        rows.append({
            "species": "pi_minus",
            "bin_index": 100 + i,
            "status": "pass",
            "values": [base, base - spread / 2.0, base + spread / 2.0],
            "unit_se": spread / sigma,
            "sigma": sigma,
        })
    sigmas = [row["sigma"] for row in rows]
    return {
        "bin_checks": rows,
        "summary": {"bin_spread_p95": float(np.percentile(sigmas, 95.0))},
    }


def test_width_reweighting_value_space_and_p95(driver):
    """The per-bin value-space reweighting against a hand computation,
    and the p95 convention (the sealed R32 numpy linear-interpolation
    percentile over the finite reweighted sigmas).

    Differential case on the test-author's census (0.30, 0.22, 0.36):
    W' = 0.7 GeV at alpha = 0.5 gives the census shifts
    (dc0, dc1, dc2) = (+0.035, +0.04, +0.03).  The bin values
    (y0, y1, y2) = (1.0, 0.5, 1.5) give the route contrast
    m_b = (1.5 - 0.5) / (0.36 - 0.22) = 10/1.4, and the reweighted
    values y_v' = y_v + m_b dc_v = (1.25, 0.785714...,
    1.714285...); the reweighted spread is 13/14 of the sealed spread
    1.0 and the reweighted sigma is 13/14 of the sealed sigma 4.0.
    The 4-bin report with sealed sigmas (1, 2, 3, 4) reweights to the
    hand p95 values 3.85 * 13/14 = 3.575 (W' = 0.7) and 3.85 * 15/14
    = 4.125 (W' = 0.3)."""
    census = {"0": 0.30, "1": 0.22, "2": 0.36}
    slopes = driver.census_slopes(census)
    prime = driver.reweighted_census(census, slopes, 0.7, 0.5)
    block = driver.reweight_bin_row([1.0, 0.5, 1.5], 0.25, slopes, census,
                                    prime)
    assert block["route_contrast"] == pytest.approx(10.0 / 1.4, rel=1e-15)
    assert block["reweighted_values"] == pytest.approx(
        [1.25, 0.7857142857142857, 1.7142857142857143], rel=1e-12,
    )
    assert block["reweighted_spread"] == pytest.approx(13 / 14, rel=1e-12)
    assert block["reweighted_sigma"] == pytest.approx(4.0 * 13 / 14,
                                                      rel=1e-12)
    # The synthetic 4-bin report: the sealed p95 is the hand value
    # 3.85 (numpy linear interpolation at position 0.95 * 3 = 2.85 of
    # the sorted sigmas).
    report = _ta_bin_report()
    sealed_p95 = report["summary"]["bin_spread_p95"]
    assert sealed_p95 == pytest.approx(3.85, rel=1e-12)
    # The differential reweighting scales the p95 by the census-spread
    # ratio exactly (each monotone bin's sigma follows the ratio).
    block_wide = driver.reweight_statistics(report, slopes, census, 0.7)
    assert block_wide["census_spread_ratio"] == pytest.approx(13 / 14)
    assert block_wide["reweighted_bin_spread_p95"] == (
        pytest.approx(3.575, rel=1e-12)
    )
    assert block_wide["uniform_ratio_cross_check_p95"] == (
        pytest.approx(3.85 * 13 / 14, rel=1e-12)
    )
    assert block_wide["bins_reweighted"] == 4
    assert block_wide["bins_vacuous_unchanged"] == 0
    block_narrow = driver.reweight_statistics(report, slopes, census,
                                              0.3)
    assert block_narrow["census_spread_ratio"] == pytest.approx(15 / 14)
    assert block_narrow["reweighted_bin_spread_p95"] == (
        pytest.approx(4.125, rel=1e-12)
    )
    # The flat-slope identity case on the same report: the reweighted
    # p95 equals the sealed p95 (the census spread is invariant) at
    # both widths.
    census_flat = {"0": 0.25, "1": 0.20, "2": 0.30}
    slopes_flat = driver.census_slopes(census_flat)
    for width_prime in (0.3, 0.7):
        block_flat = driver.reweight_statistics(
            report, slopes_flat, census_flat, width_prime,
        )
        assert block_flat["census_spread_ratio"] == (
            pytest.approx(1.0, rel=1e-12)
        )
        assert block_flat["reweighted_bin_spread_p95"] == (
            pytest.approx(sealed_p95, rel=1e-12)
        )
    # The p95 convention pins (the sealed R32 percentile convention:
    # numpy linear interpolation over the finite values, 0.0 for the
    # empty set).
    assert driver.percentile95([]) == 0.0
    assert driver.percentile95([3.25]) == 3.25
    assert driver.percentile95([1.0, 2.0, 3.0, 4.0]) == (
        pytest.approx(3.85, rel=1e-15)
    )
    assert driver.percentile95(list(range(1, 21))) == (
        pytest.approx(19.05, rel=1e-15)
    )
    # Vacuous rows are excluded from the reweighted p95.
    vacuous = {
        "bin_checks": [dict(row, status="vacuous_empty_bin")
                       for row in report["bin_checks"]],
        "summary": {"bin_spread_p95": 0.0},
    }
    block_vac = driver.reweight_statistics(vacuous, slopes, census, 0.7)
    assert block_vac["bins_reweighted"] == 0
    assert block_vac["bins_vacuous_unchanged"] == 4
    assert block_vac["reweighted_bin_spread_p95"] == 0.0


def test_width_contract_20_percent_pass_fail(driver):
    """The 20% width contract ``|p95' - anchor| <= 20% of the anchor``
    on the test-author's own hand fixtures, evaluated with the
    driver's own algebra at the driver's own contract constant: the
    pass side (the flat-slope zero deviation and the differential
    ~7.14% deviation) and the fail side (the 50% deviation)."""
    assert driver.WIDTH_CONTRACT_REL == pytest.approx(0.20)
    report = _ta_bin_report()
    anchor = report["summary"]["bin_spread_p95"]  # 3.85, hand value
    # Pass side: the flat-slope census leaves the p95 unchanged (zero
    # deviation, inside the contract).
    census_flat = {"0": 0.25, "1": 0.20, "2": 0.30}
    slopes_flat = driver.census_slopes(census_flat)
    block_flat = driver.reweight_statistics(
        report, slopes_flat, census_flat, 0.7,
    )
    deviation_flat = abs(block_flat["reweighted_bin_spread_p95"] - anchor)
    assert deviation_flat <= driver.WIDTH_CONTRACT_REL * anchor
    assert deviation_flat == pytest.approx(0.0, abs=1e-12)
    # Pass side: the mild differential census (ratio 13/14) shifts the
    # p95 by ~7.14%, inside the 20% contract.
    census = {"0": 0.30, "1": 0.22, "2": 0.36}
    slopes = driver.census_slopes(census)
    block = driver.reweight_statistics(report, slopes, census, 0.7)
    deviation = abs(block["reweighted_bin_spread_p95"] - anchor)
    relative = deviation / anchor
    assert relative == pytest.approx(1.0 - 13 / 14, rel=1e-9)
    assert relative < driver.WIDTH_CONTRACT_REL
    assert deviation <= driver.WIDTH_CONTRACT_REL * anchor
    # Fail side: the strongly curved census (0.30, 0.10, 0.30) at
    # W' = 0.7 halves the census spread (ratio 0.5), so the p95 leaves
    # the contract band by 50%.
    census_fail = {"0": 0.30, "1": 0.10, "2": 0.30}
    slopes_fail = driver.census_slopes(census_fail)
    block_fail = driver.reweight_statistics(
        report, slopes_fail, census_fail, 0.7,
    )
    assert block_fail["census_spread_ratio"] == pytest.approx(0.5,
                                                             rel=1e-12)
    deviation_fail = abs(block_fail["reweighted_bin_spread_p95"] - anchor)
    assert deviation_fail > driver.WIDTH_CONTRACT_REL * anchor
    assert deviation_fail / anchor == pytest.approx(0.5, rel=1e-12)


def test_census_disagreement_exact_boundary(driver):
    """The 1e-6 census disagreement bound (verdict criterion (c)) on
    hand-computed fixtures: a disagreement of exactly 1e-6 stays
    inside the bound (inclusive, the driver's ``<=`` comparison, so
    criterion (c) does not fire) while 1.1e-6 leaves it."""
    base = {
        "fraction_catalog": 0.25, "fraction_continuum": 0.75,
        "n_catalog": 50, "n_continuum": 150, "n_sources": 200,
    }
    at_bound = dict(base, fraction_catalog=base["fraction_catalog"]
                    + 1e-6)
    max_at_bound = max(driver.census_disagreement(base, at_bound).values())
    # The float realization of 1e-6 on the 0.25 base lands within
    # 1e-12 of the bound and does not exceed it.
    assert max_at_bound <= driver.CENSUS_DISAGREEMENT_BOUND
    assert max_at_bound == pytest.approx(1e-6, rel=1e-9)
    beyond = dict(base, fraction_catalog=base["fraction_catalog"]
                  + 1.1e-6)
    max_beyond = max(driver.census_disagreement(base, beyond).values())
    assert max_beyond > driver.CENSUS_DISAGREEMENT_BOUND
    assert max_beyond == pytest.approx(1.1e-6, rel=1e-9)


def test_verdict_criteria_logic(driver):
    """The verdict-criteria logic on the test-author's own synthetic
    artifacts: PASS iff the non-strange bins are placement-explained
    and nothing else fires; RECOMMENDATION on (a) a non-strange
    finding, (b) the width contract failing, or (c) the census bound
    failing.  The K+ referral is recorded independently of the verdict
    (it stands on the PASS path and on the RECOMMENDATION path)."""
    kplus_referral = {
        "referred_to_card": CARD2_ID,
        "bins": 18,
        "monotone": 15,
        "within_census_envelope": 18,
    }

    def _artifacts(findings=(), width_adequate=True, census_within=True,
                   max_disagreement=None):
        attribution = {
            "non_strange": {
                "bins": 14,
                "placement_explained": 14 - len(findings),
                "findings": list(findings),
            },
            "k_plus_referral": dict(kplus_referral),
        }
        reweight = {
            "contract": {
                "width_adequate": width_adequate,
                "per_width_prime": {"0.3": width_adequate,
                                    "0.7": width_adequate},
            },
        }
        census = {
            "max_disagreement": (
                0.0 if max_disagreement is None else max_disagreement
            ),
            "disagreement_bound": driver.CENSUS_DISAGREEMENT_BOUND,
            "within_bound": census_within,
        }
        return attribution, reweight, census

    # The PASS path: all clean, the K+ referral stands independently.
    attribution, reweight, census = _artifacts()
    evaluation = driver.evaluate_verdict_criteria(
        attribution, reweight, census,
    )
    assert evaluation["verdict"] == "PASS"
    assert evaluation["fired"] == []
    assert all(
        c["fired"] is False for c in evaluation["criteria"].values()
    )
    # Criterion (a): any non-strange bin not placement-explained.
    finding = {
        "species": "pi_plus", "bin_index": 41,
        "deltas_ramp_start_order": [1.2, -0.4, 0.0],
        "p1_monotone": False, "p2_within_envelope": True,
    }
    attribution, reweight, census = _artifacts(findings=[finding])
    evaluation = driver.evaluate_verdict_criteria(
        attribution, reweight, census,
    )
    assert evaluation["verdict"] == "RECOMMENDATION"
    assert [c["criterion"] for c in evaluation["fired"]] == ["a"]
    # Criterion (b): the width sensitivity exceeds the 20% contract.
    attribution, reweight, census = _artifacts(width_adequate=False)
    evaluation = driver.evaluate_verdict_criteria(
        attribution, reweight, census,
    )
    assert evaluation["verdict"] == "RECOMMENDATION"
    assert [c["criterion"] for c in evaluation["fired"]] == ["b"]
    # Criterion (c): the census recomputation disagrees beyond 1e-6
    # (a driver defect).
    attribution, reweight, census = _artifacts(
        census_within=False, max_disagreement=1e-5,
    )
    evaluation = driver.evaluate_verdict_criteria(
        attribution, reweight, census,
    )
    assert evaluation["verdict"] == "RECOMMENDATION"
    assert [c["criterion"] for c in evaluation["fired"]] == ["c"]
    assert evaluation["criteria"]["c"]["max_disagreement"] == (
        pytest.approx(1e-5)
    )
    # The compound case: every criterion fires, in the (a), (b), (c)
    # order of the driver's criteria tuple.
    attribution, reweight, census = _artifacts(
        findings=[finding], width_adequate=False, census_within=False,
        max_disagreement=1e-5,
    )
    evaluation = driver.evaluate_verdict_criteria(
        attribution, reweight, census,
    )
    assert evaluation["verdict"] == "RECOMMENDATION"
    assert [c["criterion"] for c in evaluation["fired"]] == ["a", "b",
                                                             "c"]


# ----------------------------------------------------------------------
# The zero-draw invariant (test-author technique)
# ----------------------------------------------------------------------
class _DrawSentinel:
    """Any attribute access on the numpy RNG surface during the audit's
    pure surface is a contract violation (the zero-draw invariant is
    probed by replacing ``np.random`` with a raising sentinel, not
    only by a source scan)."""

    def __getattr__(self, name):
        raise AssertionError(
            "the audit driver touched the RNG surface: np.random."
            f"{name} (zero-draw contract violation)"
        )


def test_zero_draw_rng_surface_never_touched(driver, monkeypatch):
    """The driver performs no RNG draws: the full pure surface
    (attribution rows, reweighting statistics, census pooling, verdict
    criteria, pre-registration builder, percentile) is exercised
    under a sentinel on ``np.random`` that raises on any attribute
    access, and the source of both driver trees carries no
    random-number machinery (the test-author's own token list)."""
    sentinel = _DrawSentinel()
    monkeypatch.setattr(np, "random", sentinel)
    # The placement attribution surface (hand fixture, ramp-order
    # census).
    census = {"0": 0.25, "1": 0.22, "2": 0.28}
    entries = [
        {
            "species": "proton", "bin_index": 3, "parent_index": 0,
            "parent_sigma": 2.5,
            "confirmation_values": [0.020, 0.015, 0.025],
            "confirmation_unit_se": 0.005,
            "confirmation_sigma": 2.0,
        },
        {
            "species": "k_plus", "bin_index": 8, "parent_index": 1,
            "parent_sigma": 3.0,
            "confirmation_values": [0.004, 0.006, 0.002],
            "confirmation_unit_se": 0.001,
            "confirmation_sigma": 4.0,
        },
    ]
    parent_rows = {
        ("proton", 3): {
            "values": [0.021, 0.016, 0.026], "unit_se": 0.0055,
            "sigma": 2.0,
        },
    }
    rows = driver.attribution_rows(entries, census, parent_rows)
    assert len(rows) == 2
    assert rows[0]["p1_monotone"] is True
    assert rows[1]["strange_species"] is True
    assert rows[1]["route_dominance"] == "continuum_dominant"
    # The width reweighting surface (hand fixtures).
    slopes = driver.census_slopes(census)
    prime = driver.reweighted_census(census, slopes, 0.7, 0.5)
    block = driver.reweight_bin_row([1.0, 0.5, 1.5], 0.25, slopes, census,
                                    prime)
    assert block["reweighted_spread"] == pytest.approx(
        block["route_contrast"] * (prime["2"] - prime["1"]), rel=1e-12,
    )
    report = _ta_bin_report()
    stats = driver.reweight_statistics(report, slopes, census, 0.7)
    assert stats["bins_reweighted"] == 4
    assert stats["reweighted_bin_spread_p95"] > 0.0
    # The census pooling and disagreement surface.
    pooled = driver.pooled_census_from_payloads([
        {"routing": {"n_catalog": 7, "n_sources": 20}},
        {"routing": {"n_catalog": 13, "n_sources": 20}},
    ])
    assert pooled["n_catalog"] == 20
    assert pooled["fraction_catalog"] == pytest.approx(0.5)
    fields = driver.census_disagreement(
        pooled, {"fraction_catalog": 0.5, "fraction_continuum": 0.5,
                 "n_catalog": 20, "n_continuum": 20, "n_sources": 40},
    )
    assert max(fields.values()) == 0.0
    # The verdict criteria surface (the PASS path).
    evaluation = driver.evaluate_verdict_criteria(
        {"non_strange": {"bins": 14, "placement_explained": 14,
                         "findings": []}},
        {"contract": {"width_adequate": True,
                      "per_width_prime": {"0.3": True, "0.7": True}}},
        {"max_disagreement": 0.0,
         "disagreement_bound": 1e-6, "within_bound": True},
    )
    assert evaluation["verdict"] == "PASS"
    # The pre-registration builder (pure payload construction).
    pre = driver.build_pre_registration(driver.substrate_digests(), [])
    assert pre["zero_draw"]["draws"] == 0
    assert pre["zero_draw"]["no_fitting"] is True
    assert pre["zero_draw"]["optimizer_invoked"] is False
    assert pre["zero_draw"]["reserved_audit_block"]["fired"] is False
    assert pre["placement_attribution"]["offline_envelope_se"] == 4.0
    assert pre["census_recompute"]["disagreement_bound"] == 1e-6
    assert pre["width_sensitivity"]["contract_relative"] == 0.20
    # The percentile convention (numpy linear interpolation, not an
    # RNG surface call).
    assert driver.percentile95([1.0, 3.0]) == pytest.approx(2.9)
    # Source scan (the test-author's token list) over both trees.
    forbidden = (
        "np.random", "numpy.random", "default_rng", "SeedSequence",
        "RandomState", "import random", "randint", "randn(",
        "random.", "getstate", "setstate",
    )
    for source_path in (NESTED_DRIVER, OUTER_DRIVER):
        source = source_path.read_text(encoding="utf-8")
        hits = [token for token in forbidden if token in source]
        assert not hits, f"{source_path}: RNG machinery {hits}"


# ----------------------------------------------------------------------
# Part 2: the sealed artifacts (run-dir pins)
# ----------------------------------------------------------------------
def test_driver_dual_tree_byte_identity():
    """The nested and outer audit driver copies are byte-identical at
    the pinned digest (the dual-tree register; nested 5bd4800, outer
    584193b)."""
    assert NESTED_DRIVER.is_file(), NESTED_DRIVER
    assert OUTER_DRIVER.is_file(), OUTER_DRIVER
    nested_bytes = NESTED_DRIVER.read_bytes()
    outer_bytes = OUTER_DRIVER.read_bytes()
    assert nested_bytes == outer_bytes
    assert _sha256_file(NESTED_DRIVER) == AUDIT_DRIVER_SHA256
    assert _sha256_file(OUTER_DRIVER) == AUDIT_DRIVER_SHA256


def test_sealed_run_dir_seal_pins():
    """The seal manifest and the SHA256SUMS.txt integrity of the sealed
    audit run dir: the on-disk seal manifest carries 7 files totalling
    95,594 bytes (the clean-environment record is the documented
    post-seal file and is never seal-covered); every listed file
    re-hashes to its recorded digest; the SHA256SUMS.txt inventory
    covers the 6 pre-seal payload files with monorepo-root-relative
    paths, and every entry re-hashes from the monorepo root."""
    run_dir = _skip_if_no_run_dir()
    seal = _read_json(run_dir / "seal_manifest.json")
    assert seal["card"] == CARD_ID
    assert seal["file_count"] == SEAL_FILE_COUNT == 7
    assert seal["total_bytes"] == SEAL_TOTAL_BYTES == 95594
    assert seal["sha256_sums_sha256"] == SEAL_FILES["SHA256SUMS.txt"][0]
    assert seal["excluded_from_sums"] == ["seal_manifest.json",
                                          "SHA256SUMS.txt"]
    listed = {entry["path"]: (entry["sha256"], entry["bytes"])
              for entry in seal["files"]}
    assert set(listed) == set(SEAL_FILES)
    assert sum(size for _digest, size in listed.values()) == (
        SEAL_TOTAL_BYTES
    )
    for name, (digest, size) in SEAL_FILES.items():
        path = run_dir / name
        assert path.is_file(), name
        assert path.stat().st_size == size, name
        assert _sha256_file(path) == digest, name
    assert _sha256_file(run_dir / "seal_manifest.json") == (
        SEAL_MANIFEST_SHA256
    )
    # The SHA256SUMS.txt inventory: 6 monorepo-root-relative entries,
    # each re-hashing from the monorepo root.
    sums_text = (run_dir / "SHA256SUMS.txt").read_text(encoding="utf-8")
    lines = [line for line in sums_text.splitlines() if line.strip()]
    assert len(lines) == 6
    covered = set()
    for line in lines:
        digest, name = line.split("  ", 1)
        prefix = "dev/00runs/r4_audit_transition_routing_formulation_"
        assert name.startswith(prefix), name
        payload = MONO_ROOT / name
        assert payload.is_file(), name
        assert _sha256_file(payload) == digest, name
        covered.add(Path(name).name)
    assert covered == set(SEAL_FILES) - {
        "seal_manifest.json", "SHA256SUMS.txt",
    }
    # The clean-environment record (the post-seal validation record).
    assert _sha256_file(run_dir / "clean_environment_commands.json") == (
        CLEAN_ENV_SHA256
    )
    clean = _read_json(run_dir / "clean_environment_commands.json")
    assert clean["hashes_match"] is True
    assert clean["seal_verification"]["mismatches"] == []
    assert clean["probe_returncodes"] == {
        "census": 0, "attribution": 0, "reweight": 0, "verdict": 0,
    }


def test_sealed_run_dir_verdict_pins():
    """The verdict.json content pins: the RECOMMENDATION verdict with
    criteria fired [a]; the two named findings pi- bin 67 and pi+ bin
    173 with their delta signatures (pooled-SE units against the
    median-variant reference, in ramp-start order); the K+ referral to
    card 2; draws 0 / no_fitting / optimizer_invoked; the census
    max_disagreement 0.0; and the reweighted p95 values with their
    relative deviations (tolerance pins against the sealed doubles)."""
    run_dir = _skip_if_no_run_dir()
    verdict = _read_json(run_dir / "verdict.json")
    assert verdict["card"] == CARD_ID
    assert verdict["verdict"] == "RECOMMENDATION"
    # The criteria: (a) fired, (b) and (c) not.
    assert verdict["criteria"]["a"]["fired"] is True
    assert verdict["criteria"]["b"]["fired"] is False
    assert verdict["criteria"]["c"]["fired"] is False
    criterion_a = verdict["criteria"]["a"]
    assert criterion_a["non_strange_bins"] == 14
    assert criterion_a["placement_explained"] == 12
    assert criterion_a["findings"] == 2
    assert verdict["criteria"]["b"]["width_adequate"] is True
    assert verdict["criteria"]["c"]["max_disagreement"] == 0.0
    assert verdict["criteria"]["c"]["bound"] == 1e-6
    # The bounds carried verbatim (never relaxed).
    assert verdict["bounds"]["bin_spread_p95_bound"] == 2.0
    assert verdict["bounds"]["feeddown_spread_max_bound"] == 3.0
    assert verdict["bounds"]["census_disagreement_bound"] == 1e-6
    assert verdict["bounds"]["width_contract_relative"] == 0.20
    assert verdict["bounds"]["offline_envelope_se"] == 4.0
    # The named findings: exactly pi- bin 67 and pi+ bin 173, each a
    # bin finding with the placement-mechanism diagnosis.
    named = {
        (f["kind"], f["species"], f["bin_index"]): f
        for f in verdict["findings"]
    }
    assert set(named) == {("bin", "pi_minus", 67), ("bin", "pi_plus",
                                                    173)}
    pi_minus = named[("bin", "pi_minus", 67)]
    assert pi_minus["evidence"]["p1_monotone"] is False
    assert pi_minus["evidence"]["p2_within_envelope"] is True
    deltas = pi_minus["evidence"]["deltas_vs_median_se_units"]
    for variant, value in FINDING_PI_MINUS_67.items():
        assert deltas[variant] == pytest.approx(value, rel=1e-12,
                                                abs=1e-12)
    assert pi_minus["evidence"]["offline_residual_se"] == (
        pytest.approx(FINDING_PI_MINUS_67_RESIDUAL, rel=1e-12)
    )
    assert pi_minus["evidence"]["confirmation_sigma"] == (
        pytest.approx(FINDING_PI_MINUS_67_SIGMA, rel=1e-12)
    )
    assert "falls to the nominal variant and rises to the late " in (
        pi_minus["observed"]
    )
    assert "the pre-registered bound 2.0" in pi_minus["observed"]
    pi_plus = named[("bin", "pi_plus", 173)]
    assert pi_plus["evidence"]["p1_monotone"] is False
    assert pi_plus["evidence"]["p2_within_envelope"] is True
    deltas = pi_plus["evidence"]["deltas_vs_median_se_units"]
    for variant, value in FINDING_PI_PLUS_173.items():
        assert deltas[variant] == pytest.approx(value, rel=1e-12,
                                                abs=1e-12)
    assert pi_plus["evidence"]["offline_residual_se"] == (
        pytest.approx(FINDING_PI_PLUS_173_RESIDUAL, rel=1e-12)
    )
    assert pi_plus["evidence"]["confirmation_sigma"] == (
        pytest.approx(FINDING_PI_PLUS_173_SIGMA, rel=1e-12)
    )
    assert "rises to the nominal variant and falls to the late " in (
        pi_plus["observed"]
    )
    for finding in (pi_minus, pi_plus):
        assert finding["responsible"].startswith(
            "the card-6 blend routing formulation"
        )
        assert CARD6_SHA256 in finding["responsible"]
        assert "rework card" in finding["proposed_rework"]
    # The K+ referral to card 2 (independent of the verdict).
    kplus = verdict["attribution_summary"]["k_plus_referral"]
    assert kplus["referred_to_card"] == CARD2_ID
    assert kplus["bins"] == 18
    assert kplus["monotone"] == 15
    assert kplus["within_census_envelope"] == 18
    assert kplus["continuum_dominant"] == 15
    # The zero-draw record and the surface byte-stability statement.
    assert verdict["zero_draw"]["draws"] == 0
    assert verdict["zero_draw"]["no_fitting"] is True
    assert verdict["zero_draw"]["optimizer_invoked"] is False
    assert verdict["zero_draw"]["reserved_audit_block_fired"] is False
    assert verdict["surface"]["card6_sha256"] == CARD6_SHA256
    assert verdict["surface"]["byte_unchanged_by_this_audit"] is True
    # The census summary: the recomputation agrees exactly.
    assert verdict["census_summary"]["max_disagreement"] == 0.0
    assert verdict["census_summary"]["within_bound"] is True
    # The reweighted p95 values and their relative deviations (the
    # 20% contract pins, tolerance against the sealed doubles).
    assert verdict["width_summary"]["width_adequate"] is True
    per_w = {
        block["width_prime_gev"]: block
        for block in verdict["width_summary"]["per_width_prime"]
    }
    assert per_w[0.3]["reweighted_bin_spread_p95"] == (
        pytest.approx(REWEIGHT_P95_W03, rel=1e-12)
    )
    assert per_w[0.3]["relative_deviation"] == (
        pytest.approx(REWEIGHT_RELDEV_W03, rel=1e-12)
    )
    assert per_w[0.3]["within_contract"] is True
    assert per_w[0.7]["reweighted_bin_spread_p95"] == (
        pytest.approx(REWEIGHT_P95_W07, rel=1e-12)
    )
    assert per_w[0.7]["relative_deviation"] == (
        pytest.approx(REWEIGHT_RELDEV_W07, rel=1e-12)
    )
    assert per_w[0.7]["within_contract"] is True
    # The recommendation text: the rework-card obligation and the card
    # 2 referral.
    assert "rework card" in verdict["recommendation"]
    assert CARD2_ID in verdict["recommendation"]


def test_sealed_run_dir_reweight_pins():
    """The spread_reweighted_W.json pins: the W' p95 values and
    relative deviations (cross-checked against the production anchor),
    the alpha-bracket endpoints, the contract record, the carried 2.0
    bound, and the zero-draw stamps."""
    run_dir = _skip_if_no_run_dir()
    reweight = _read_json(run_dir / "spread_reweighted_W.json")
    assert reweight["card"] == CARD_ID
    assert reweight["production"]["anchor_p95"] == PRODUCTION_P95
    variants = {
        block["width_prime_gev"]: block
        for block in reweight["production"]["width_variants"]
    }
    assert set(variants) == {0.3, 0.7}
    for width, p95, reldev in (
        (0.3, REWEIGHT_P95_W03, REWEIGHT_RELDEV_W03),
        (0.7, REWEIGHT_P95_W07, REWEIGHT_RELDEV_W07),
    ):
        block = variants[width]
        assert block["reweighted_bin_spread_p95"] == (
            pytest.approx(p95, rel=1e-12)
        )
        assert block["contract"]["anchor_p95"] == PRODUCTION_P95
        assert block["contract"]["relative_deviation"] == (
            pytest.approx(reldev, rel=1e-12)
        )
        assert block["contract"]["contract_relative"] == 0.20
        assert block["contract"]["within_contract"] is True
        assert block["contract"]["alpha_bracket_within_contract"] is True
        # The alpha bracket endpoints: alpha = 0 leaves the p95 at the
        # anchor (zero deviation); alpha = 1 doubles the neutral shift.
        bracket = block["contract"]["alpha_bracket"]
        assert set(bracket) == {"0.0", "1.0"}
        assert bracket["0.0"]["reweighted_bin_spread_p95"] == (
            pytest.approx(PRODUCTION_P95, rel=1e-12)
        )
        assert bracket["0.0"]["relative_deviation"] == 0.0
        assert bracket["1.0"]["relative_deviation"] > (
            block["contract"]["relative_deviation"]
        )
    # The contract record and the carried bound.
    assert reweight["contract"]["width_adequate"] is True
    assert reweight["contract"]["per_width_prime"] == {
        "0.3": True, "0.7": True,
    }
    assert reweight["new_spread_bounds"]["bin_spread_p95_bound"] == 2.0
    assert reweight["draws"] == 0
    assert reweight["no_fitting"] is True
    assert reweight["optimizer_invoked"] is False
    # The algebra record: the neutral alpha and its bracket.
    assert reweight["algebra"]["alpha_neutral"] == 0.5
    assert reweight["algebra"]["alpha_bracket"] == [0.0, 1.0]


def test_sealed_run_dir_measurement_artifact_pins():
    """The census and attribution artifact digest pins and structure
    pins: the census recomputation agrees with the sealed routing
    census exactly (max_disagreement 0.0); the attribution carries the
    32-entry probe set with the pre-registered species split and sigma
    range, the 27/32 monotone census, the two non-strange findings,
    and the K+ referral counts."""
    run_dir = _skip_if_no_run_dir()
    census = _read_json(run_dir / "routing_census_recomputed.json")
    assert census["card"] == CARD_ID
    assert census["max_disagreement"] == 0.0
    assert census["disagreement_bound"] == 1e-6
    assert census["within_bound"] is True
    assert census["draws"] == 0
    assert census["no_fitting"] is True
    assert census["optimizer_invoked"] is False
    assert census["sources"]["production_replicas_per_variant"] == 8
    assert census["sources"]["confirmation_replicas_per_variant"] == 16
    attribution = _read_json(run_dir / "persisting_bin_attribution.json")
    assert attribution["card"] == CARD_ID
    probe = attribution["probe_set"]
    assert probe["entries"] == 32
    assert probe["species_split"] == {
        "k_plus": 18, "proton": 9, "pi_plus": 4, "pi_minus": 1,
    }
    assert probe["sigma_range"] == pytest.approx(
        [2.0001998039779263, 7.532929574244998], rel=1e-12,
    )
    rows = attribution["rows"]
    assert len(rows) == 32
    assert sum(1 for row in rows if row["p1_monotone"]) == 27
    assert sum(1 for row in rows if row["p2_within_envelope"]) == 32
    assert sum(1 for row in rows if row["placement_explained"]) == 27
    non_strange = attribution["non_strange"]
    assert non_strange["bins"] == 14
    assert non_strange["placement_explained"] == 12
    assert {(f["species"], f["bin_index"])
            for f in non_strange["findings"]} == {
        ("pi_minus", 67), ("pi_plus", 173),
    }
    kplus = attribution["k_plus_referral"]
    assert kplus["referred_to_card"] == CARD2_ID
    assert kplus["bins"] == 18
    assert kplus["monotone"] == 15
    assert kplus["within_census_envelope"] == 18
    assert kplus["mass_blind_envelope_violations"] == 16
    assert kplus["continuum_dominant"] == 15
    assert attribution["draws"] == 0
    assert attribution["no_fitting"] is True


def test_sealed_run_dir_prereg_provenance_pins():
    """The pre-registration immutability pin (the sha256 as on disk)
    and the seed-block-not-fired statement, and the provenance driver
    digest pin (nested == outer == the on-disk dual-tree digest)."""
    run_dir = _skip_if_no_run_dir()
    assert _sha256_file(run_dir / "pre_registration.json") == (
        PRE_REGISTRATION_SHA256
    )
    pre = _read_json(run_dir / "pre_registration.json")
    assert pre["card"] == CARD_ID
    assert pre["written_before_measurement"] is True
    assert pre["probe_set"]["expected_entries"] == 32
    assert pre["probe_set"]["expected_species_split"] == {
        "k_plus": 18, "proton": 9, "pi_plus": 4, "pi_minus": 1,
    }
    # The zero-draw contract and the seed-block-not-fired statement.
    zero = pre["zero_draw"]
    assert zero["draws"] == 0
    assert zero["no_fitting"] is True
    assert zero["optimizer_invoked"] is False
    block = zero["reserved_audit_block"]
    assert block["fired"] is False
    assert block["master_seed"] == 20260920
    assert block["raw_seeds"] == [20301040, 20301055]
    # The pre-registered bounds and contracts.
    assert pre["census_recompute"]["disagreement_bound"] == 1e-6
    assert pre["placement_attribution"]["offline_envelope_se"] == 4.0
    assert pre["width_sensitivity"]["width_primes_gev"] == [0.3, 0.7]
    assert pre["width_sensitivity"]["contract_relative"] == 0.20
    assert pre["width_sensitivity"]["production_p95_anchor"] == (
        PRODUCTION_P95
    )
    assert pre["new_spread_bounds"]["bin_spread_p95_bound"] == 2.0
    assert pre["new_spread_bounds"]["feeddown_spread_max_bound"] == 3.0
    assert pre["verdict_contract"]["criteria"] == ["a", "b", "c"]
    assert pre["verdict_contract"]["strange_referral_card"] == CARD2_ID
    # The provenance: the dual-tree driver digest recorded at run
    # time equals the on-disk dual-tree digest (the byte-identity
    # cross-check).
    provenance = _read_json(run_dir / "provenance.json")
    assert provenance["driver_sha256"]["nested"] == (
        AUDIT_DRIVER_SHA256
    )
    assert provenance["driver_sha256"]["outer"] == (
        AUDIT_DRIVER_SHA256
    )
    assert provenance["zero_draw"]["draws"] == 0
    assert provenance["zero_draw"]["no_fitting"] is True
    assert provenance["zero_draw"]["optimizer_invoked"] is False
    assert provenance["zero_draw"]["reserved_audit_block_fired"] is False
    assert _sha256_file(NESTED_DRIVER) == (
        provenance["driver_sha256"]["nested"]
    )
    assert _sha256_file(OUTER_DRIVER) == (
        provenance["driver_sha256"]["outer"]
    )


def test_card6_surface_and_builder_byte_stability():
    """The card-6 blend routing surface and the builder blend path
    stay at their pinned on-disk digests (the audit is byte-unchanged
    on both; a surface change is a separate rework card)."""
    assert _sha256_file(
        MONO_ROOT / "dev" / "mixing2006" / "na49_mixing"
        / "spectral_density.py"
    ) == CARD6_SHA256
    assert _sha256_file(
        MONO_ROOT / "dev" / "mixing2006" / "na49_mixing"
        / "resonance_pp_builder.py"
    ) == BUILDER_SHA256


def test_freeze_repin_register():
    """The legacy_v1 manifest re-pin for this register: the strict
    path-sorted inventory carries this test file at its on-disk
    digest; the freeze_exceptions carry exactly one entry naming this
    card, with the audit's files plus this register's file,
    path-sorted; the pinned M8 parameter map stays intact."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    inventory = manifest["source_tree_sha256"]
    # The strict path-sorted inventory rule.
    keys = list(inventory.keys())
    assert keys == sorted(keys)
    # This register's inventory entry at its on-disk digest.
    assert inventory.get(MY_TEST_REL) == _sha256_file(
        NESTED_ROOT / "tests" / "test_r4_audit_routing_contract_pins.py"
    ), "the manifest does not pin this register at its on-disk digest"
    # The audit's own entries (nested + outer driver, the solver's
    # test) stay at their on-disk digests.
    for rel in (
        "dev/mixing2006/helpers/benchmarking/"
        "audit_r4_routing_formulation.py",
        "helpers/benchmarking/audit_r4_routing_formulation.py",
        "dev/mixing2006/tests/test_audit_r4_routing_formulation.py",
    ):
        assert inventory.get(rel) == _sha256_file(MONO_ROOT / rel), rel
    assert inventory[
        "dev/mixing2006/helpers/benchmarking/"
        "audit_r4_routing_formulation.py"
    ] == inventory[
        "helpers/benchmarking/audit_r4_routing_formulation.py"
    ] == AUDIT_DRIVER_SHA256
    # The freeze exception: exactly one entry naming this card (the
    # solver's pin counts it exactly), path-sorted files, carrying the
    # audit's files plus this register's file.
    exceptions = manifest["freeze_exceptions"]
    entries = [e for e in exceptions if e.get("card") == CARD_ID]
    assert len(entries) == 1, (
        "the freeze_exceptions carry exactly one entry naming this "
        "card"
    )
    files = entries[0]["files"]
    assert files == sorted(files), (
        "the card's freeze-exception files are path-sorted"
    )
    for rel in (
        "dev/mixing2006/helpers/benchmarking/"
        "audit_r4_routing_formulation.py",
        "helpers/benchmarking/audit_r4_routing_formulation.py",
        "dev/mixing2006/tests/test_audit_r4_routing_formulation.py",
        "dev/mixing2006/tests/test_r4_audit_routing_contract_pins.py",
        "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json",
    ):
        assert rel in files, f"the freeze exception names {rel}"
    # The pinned M8 map stays intact (the freeze test enforces the
    # full inventory).
    assert manifest["frozen_fit_outputs"]["m8_frozen_pp_parameters"][
        "continuum.match_mass"
    ] == 2.680194110727528
