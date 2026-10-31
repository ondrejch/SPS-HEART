"""Round 4 card 3: builder blend-path routing-weight audit contract pins
(test-author register).

Task ``TASK-20260920-r4-audit-builder-blend-path-routing-weight`` (Gate 3
decision card follow-up obligation 3; plan section 9.3).  This is the
test-author's independent pin register for the zero-draw, hermetic
dual-tree audit driver
``helpers/benchmarking/audit_r4_builder_blend_path.py`` (nested commit
``37ab205``, outer ``0f76132``; byte-identical in both trees at sha256
``ab8ea4d36fef574318aae6b65d3458156d03e84a6cc5536de5ad501536ac2003``)
and for the sealed audit run dir
``dev/00runs/r4_audit_builder_blend_path_routing_weight_2026-09-20/``
(verdict PASS, 0 named findings).  It is an independent cross-check of
the solver's register in
``tests/test_audit_r4_builder_blend_path.py``: the same pre-registered
measurement contract, pinned from an independently authored test with
its own hand-built mass/uniform fixtures, synthetic budget rows,
synthetic census payloads, and synthetic verdict inputs (no solver
fixture values reused).

Part 1 (the probe contract, pure, no sealed run dir needed):

- ``test_three_region_routing_rule_ta_fixture``: the three-region
  routing rule (M < B / B <= M < B + W / M >= B) with the draw budgets
  0 / 1 / 0 and the pre-registered ramp ``w(M)`` on the test-author's
  own mass fixtures, including the exact boundary values M = B and
  M = B + W, cross-checked against the card-6 surface's
  ``BlendSpec.weight`` at every fixture mass.
- ``test_direct_route_contract_on_stub_stream_ta_fixture``: the card-6
  ``route_source_decay`` routing contract on hand-fed stub streams
  (no generator): the consumed-uniform budget per region (0 / 1 / 0),
  the strict ``u < w(M)`` band outcome, and the band boundary at
  ``u == w(M)`` (the continuum is never routed on equality).
- ``test_draw_budget_expectation_builder_ta_fixture``: the
  per-region budget-table builder on the test-author's own synthetic
  source rows (region assignments chosen independently of the
  solver's probe): the clean table, and the named-mismatch rows for
  every violation class (uniform budget, path draw, route
  derivation, weight stamp).
- ``test_census_pooling_and_disagreement_ta_fixture``: the pooled
  routing-census arithmetic (pure integer pooling over the per-replica
  payload routing counts, the fraction identities, and the refusal of
  the empty pool) and the census-disagreement arithmetic on the
  test-author's own synthetic payloads.
- ``test_verdict_criteria_logic_ta_fixture``: the verdict-criteria
  logic (PASS iff (a) the draw budget is exact in all three regions,
  (b) the below-boundary path re-verifies bit-identity, (c) the
  boundary-matching factor is never consulted on the sampling path,
  (d) the census is consistent within 1e-6) on the test-author's own
  synthetic measurement artifacts, including the inclusive 1e-6
  bound boundary and the compound all-criteria case.
- ``test_zero_draw_measurement_path_ta_fixture``: the zero-draw
  invariants on the test-author's own technique (independent of the
  solver's token scan): an AST analysis pinning every ``np.random``
  construction site in the driver to the pinned bit-identity seed
  20300913 (no other seed literal anywhere in the driver), the
  reserved audit block 20301040..20301055 appearing nowhere except
  its pre-registration constants, and the pre-registration builder's
  zero-draw record on the on-disk substrate.

Part 2 (the sealed artifacts, run-dir pins; the run dir is gitignored,
the pins activate once the dir exists and skip with a recorded reason
on a fresh checkout):

- ``test_driver_dual_tree_byte_identity``: the nested and outer audit
  driver copies are byte-identical at the pinned digest.
- ``test_surface_byte_stability``: the builder blend path, the card-6
  blend routing surface, the engine module, the catalog, the
  continuum families config, and the R32 driver (both trees) stay at
  their pinned on-disk digests (re-hashed independently of the
  driver).
- ``test_sealed_run_dir_seal_pins``: the seal manifest (7 files,
  2,649,524 bytes, per the on-disk ``seal_manifest.json``), the full
  per-file digest listing self-consistent against disk, the
  ``SHA256SUMS.txt`` re-hash integrity (6 monorepo-root-relative
  entries, self-digest pinned), and the post-seal
  clean-environment record.
- ``test_sealed_run_dir_trace_pins``: the ``draw_budget_trace.json``
  pins (the probe configuration at the sealed seed, the region
  budgets 366 / 151 / 1483 sources with 0 / 1 / 0 routing uniforms,
  the band outcome ``u < w(M)`` re-derived on all 151 band rows, the
  probe census, the sealed-envelope comparisons, the bit-identity leg
  6000 / 6000 layers, 209 / 209 below-boundary finals, 1461 / 1461
  legacy finals, the call aggregate, and the zero-draw stamps).
- ``test_sealed_run_dir_isolation_pins``: the
  ``isolation_assertion.json`` pins (the 200-event spy fixture with 0
  boundary-factor consultations, the 7-function static assertion,
  the 48-row direct route contract with the boundary crossings at
  M = B and M = B + W, the routing-weight consistency, and the
  factor-isolated verdict).
- ``test_sealed_run_dir_census_pins``: the
  ``census_consistency.json`` pins (leg A within the 1e-6 bound at
  max disagreement 0.0, the v0 fractions 0.22329625 production and
  0.22357 confirmation exact, leg B exact on the 151 band rows, the
  informational leg-C bracket [0.188, 0.2624], the 72 payload
  identity stamps, and the zero-draw stamps).
- ``test_sealed_run_dir_verdict_pins``: the ``verdict.json`` content
  pins (verdict PASS, 0 named findings, the four criteria all
  unfired, the draw-budget summary with the region budget table, the
  bit-identity aggregates, the isolation summary, the census summary,
  the zero-draw record, the pinned surface digests, and the sealed
  input digests).
- ``test_sealed_run_dir_prereg_provenance_pins``: the
  ``pre_registration.json`` sha256 pin (immutability), the probe-set
  contract, the region rules against the on-disk driver constants,
  the bounds, the zero-draw contract with the reserved audit block
  20301040 + k not fired, the sealed-input and sealed-run-dir
  re-verification records (29 + 55 + 12 entries, 0 mismatches), the
  substrate digests, and the ``provenance.json`` driver digest pin
  (nested == outer == the on-disk dual-tree digest).
- ``test_freeze_repin_register``: the legacy_v1 manifest re-pin for
  this register (the inventory entry at the on-disk digest, the
  strict path-sorted inventory, and the single ``freeze_exceptions``
  entry naming this card carrying the audit's files plus this
  register's file, path-sorted).

Hermetic: the committed audit driver and the sealed archives are read
read-only; no writes outside the process, no event generation, no new
random numbers, no optimizer.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

NESTED_ROOT = Path(__file__).resolve().parent.parent
MONO_ROOT = NESTED_ROOT.parent.parent
DRIVER_REL = Path("helpers") / "benchmarking" / (
    "audit_r4_builder_blend_path.py"
)
NESTED_DRIVER = NESTED_ROOT / DRIVER_REL
OUTER_DRIVER = MONO_ROOT / DRIVER_REL
MANIFEST_PATH = NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json"

# The pinned dual-tree driver digest (nested 37ab205, outer 0f76132).
AUDIT_DRIVER_SHA256 = (
    "ab8ea4d36fef574318aae6b65d3458156d03e84a6cc5536de5ad501536ac2003"
)
# The byte-stable surfaces and the frozen substrate.
BUILDER_SHA256 = (
    "4ab612609e1ee0bc1e037dcbb2815b6484306ad32402b7289e6636f8e829cdb1"
)
CARD6_SHA256 = (
    "561c65ee9a77a1b7f7eb07a7619dafb584779a51ccb23d7fe5651c0605e12563"
)
ENGINE_SHA256 = (
    "ee240f44645d2e592acfae462f7d1e9c3b3dd8cd78c96bd40b7b73509f7fcc63"
)
CATALOG_SHA256 = (
    "6e4de7372a5bbf99affd6fa3ccc295c4cef09404dab3b17acd8816cda762bdcd"
)
FAMILIES_SHA256 = (
    "242f906cf040eadbe3796dfcfb61d0d51535277399858c901a162e6653d635e1"
)
R32_DRIVER_SHA256 = (
    "99454facce856c93505e6c4b0d4e47b85eeb0ef54f72b1631e3d29fcc511cb90"
)

CARD_ID = "TASK-20260920-r4-audit-builder-blend-path-routing-weight"
SOLVER_TEST_REL = (
    "dev/mixing2006/tests/test_audit_r4_builder_blend_path.py"
)
MY_TEST_REL = (
    "dev/mixing2006/tests/test_r4_audit_blend_path_contract_pins.py"
)
NESTED_DRIVER_REL = (
    "dev/mixing2006/helpers/benchmarking/audit_r4_builder_blend_path.py"
)
OUTER_DRIVER_REL = (
    "helpers/benchmarking/audit_r4_builder_blend_path.py"
)

# The pinned blend ramp (the M8-frozen card-6 decimals).
BLEND_BOUNDARY_GEV = 2.680194110727528
RAMP_WIDTH_GEV = 0.5

# The pinned probe / reservation seed constants (the zero-draw
# contract: fixed-seed re-evaluation of the sealed stream).
RAW_SEED = 20300913
KERNEL_MASTER_SEED = 20260913
KERNEL_REPLICA_INDEX = 0
RESERVED_MASTER_SEED = 20260920
RESERVED_BLOCK_FIRST = 20301040
RESERVED_BLOCK_LAST = 20301055

# The sealed input digests (the R32 / R23 archives the audit
# re-verifies).
SEALED_INPUT_DIGESTS = {
    "dev/00runs/r32_spectral_systematic_pp_2026-09-18/"
    "spread_report.json": (
        "ed102b10e9c99d973a9c3e9dd262536eaca08802aeb8f313ea1894f7cefe022e"
    ),
    "dev/00runs/r32_spectral_systematic_pp_2026-09-18/"
    "gate3_r32.json": (
        "64f104a8f0fdb71fdf506a3eb6a369091a2f3eed5047e499f1d344ad5668bab6"
    ),
    "dev/00runs/r32_spectral_systematic_pp_2026-09-18/"
    "bit_identity_v0.json": (
        "9ec4c6077642422283ae68fd3b3ae6bd00fa8c9b30172610ab259013bb8e5c5c"
    ),
    "dev/00runs/r32_spectral_systematic_pp_confirmation_2026-09-19/"
    "confirmation_verification.json": (
        "0553fed06a9b59e9f2020c0c53bbad7d75a11b1f2baf21f0469ee8b3e48d275f"
    ),
    "dev/00runs/r23_resonance_only_pp_ablation_2026-09-13/"
    "bit_identity_report.json": (
        "5c168540d3269c0f315cd47fba652ad4f1721f023fb310944d40f0f9b548e857"
    ),
}

# The sealed audit run dir (gitignored; the pins activate once it
# exists, the R32 pins convention).
RUN_DIR_CANDIDATES = tuple(
    (MONO_ROOT / "dev" / "00runs").glob(
        "r4_audit_builder_blend_path_routing_weight_*"
    )
)

# The sealed run dir's file digests (on-disk, 2026-09-20).
SEAL_FILES = {
    "SHA256SUMS.txt": (
        "b64fbe6d925999c60f534164154e4b75992823cc722fac3cf7c951f423237695",
        909,
    ),
    "census_consistency.json": (
        "d9bd9d2106efc8a7005bff74f6acd2a0c84cd5489734e7da190f05694fc52623",
        20211,
    ),
    "draw_budget_trace.json": (
        "44bb7635c3587e613d38082c851b08eb2c3311942c5258137561749a5472d580",
        2592792,
    ),
    "isolation_assertion.json": (
        "219a237186c9c436a537cc26aafa2c59e8ee7662b05b4e9fb4bcf7a92492e78f",
        16751,
    ),
    "pre_registration.json": (
        "ee654a98d0caf8095bb3738897f94b7535c1eaeb9267774fbc20d0349bc0f6bb",
        10294,
    ),
    "provenance.json": (
        "f80e1cbe4d97664c9560164609c476451646e109664a7849988af6afcb400977",
        3157,
    ),
    "verdict.json": (
        "502176a8d5084d2663171710396f3665883091c67e1b9e8ac72ae3e4118fc76c",
        5410,
    ),
}
SEAL_FILE_COUNT = 7
SEAL_TOTAL_BYTES = 2649524
SEAL_MANIFEST_SHA256 = (
    "06e845420f3c98bb297fdb25e5d9814a7da14d4de217ef3f1098880385303c7d"
)
CLEAN_ENV_SHA256 = (
    "51269de559276004c7059510d6bb75590d80530856ae6503bf50afbd236867df"
)
PRE_REGISTRATION_SHA256 = SEAL_FILES["pre_registration.json"][0]
PROVENANCE_SHA256 = SEAL_FILES["provenance.json"][0]
CENSUS_SHA256 = SEAL_FILES["census_consistency.json"][0]
ISOLATION_SHA256 = SEAL_FILES["isolation_assertion.json"][0]
TRACE_SHA256 = SEAL_FILES["draw_budget_trace.json"][0]
VERDICT_SHA256 = SEAL_FILES["verdict.json"][0]

# The pinned probe census (1,000 events, 2,000 sources).
PROBE_CENSUS = {
    "n_events": 1000,
    "n_sources": 2000,
    "n_below_boundary_sources": 366,
    "n_in_band_sources": 151,
    "n_above_boundary_sources": 1483,
    "catalog_routed": 445,
    "continuum_routed": 1555,
    "fraction_catalog": 0.2225,
}
# The pinned bit-identity leg (the sealed R32 v0 leg, re-verified at
# probe scale).
BIT_IDENTITY_LAYER = (6000, 6000)
BIT_IDENTITY_BELOW_BOUNDARY_FINALS = (209, 209, 157)
BIT_IDENTITY_LEGACY_FINALS = (1461, 1461)
# The pinned call aggregate.
EVENT_HASH_DIGEST = (
    "0d9accdefd88c9af3cdd25aa85cd9650379e587584d13c214edb74285d4fe781"
)

# The pinned census (leg A reference records, v0).
PRODUCTION_V0 = {
    "replicas": 8,
    "n_sources": 1600000,
    "n_catalog": 357274,
    "n_continuum": 1242726,
    "fraction_catalog": 0.22329625,
    "fraction_continuum": 0.77670375,
}
CONFIRMATION_V0 = {
    "replicas": 16,
    "n_sources": 3200000,
    "n_catalog": 715424,
    "n_continuum": 2484576,
    "fraction_catalog": 0.22357,
    "fraction_continuum": 0.77643,
}
GATE_CENSUS_V0 = 0.22329625
# The informational leg-C bracket (the w(M)-implied band census of the
# sealed bit-identity stream).
LEG_C_P_BELOW_BOUNDARY = 0.188
LEG_C_P_IN_BAND = 0.0744

# The pinned spy fixture (200 events, the boundary-factor
# consultations).
SPY_EVENTS = 200
SPY_PER_REGION = {
    "catalog_region": 70,
    "transition": 37,
    "continuum_region": 293,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_driver():
    """The committed dual-tree driver, loaded once under a unique
    module name (the driver's ``_locate_roots`` wiring resolves the
    nested roots at import time; the driver imports numpy at module
    level, which the nested suite provides)."""
    name = "audit_r4_builder_blend_path_test_author"
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
            "(dev/00runs/r4_audit_builder_blend_path_routing_weight_*) "
            "is absent on this checkout; the sealed-artifact pins "
            "activate once the run dir exists"
        )
    return run_dir


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# The test-author's own fixtures (independent of the solver's)
# ----------------------------------------------------------------------

#: The test-author's mass fixtures: the interior of each region plus
#: the two exact boundary values (M = B, M = B + W) and one far
#: outside the band.
_B = BLEND_BOUNDARY_GEV
_W = RAMP_WIDTH_GEV
TA_MASSES = [
    _B - 0.375,       # catalog interior
    _B - 1e-9,        # catalog, epsilon below B
    _B,               # transition, exact lower boundary (w = 0)
    _B + 0.125,       # transition, w = 0.25
    _B + 0.375,       # transition, w = 0.75
    _B + _W - 1e-9,   # transition, epsilon below B + W (w = 1 - 2e-9)
    _B + _W,          # continuum, exact upper boundary (w = 1)
    _B + _W + 0.3125,  # continuum interior
    4.0,              # continuum, far above the band
]
TA_REGIONS = [
    "catalog_region",
    "catalog_region",
    "transition",
    "transition",
    "transition",
    "transition",
    "continuum_region",
    "continuum_region",
    "continuum_region",
]
TA_UNIFORMS = [0, 0, 1, 1, 1, 1, 0, 0, 0]
#: The test-author's uniform fixtures (including the band-boundary
#: values).
TA_UNIFORMS_VALUES = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.9999999999, None]


def _row(region: str, routing_uniforms: int, path_draws: int, route: str,
         route_expected: str, stamped: float = 1.0,
         expected_w: float = 1.0, **flags) -> dict:
    """One synthetic source row for the budget-table builder (the
    test-author's own region assignments)."""
    row = {
        "region": region,
        "routing_uniforms": routing_uniforms,
        "routing_uniform_expected": 1 if region == "transition" else 0,
        "path_draws": path_draws,
        "route": route,
        "route_expected": route_expected,
        "blend_weight_stamped": stamped,
        "blend_weight_expected": expected_w,
    }
    row.update(flags)
    return row


# ----------------------------------------------------------------------
# Part 1: the probe contract (pure, no sealed run dir needed)
# ----------------------------------------------------------------------


def test_three_region_routing_rule_ta_fixture(driver):
    """The three-region routing rule with the draw budgets 0 / 1 / 0
    and the pre-registered ramp ``w(M)`` on the test-author's own mass
    fixtures: the region assignment at every fixture mass (including
    the exact boundaries M = B, which is the transition, and M = B +
    W, which is the continuum region), the routing-uniform budget,
    the band outcome ``u < w(M)`` with the strict inequality at
    ``u == w(M)``, the refusal of the missing transition uniform, and
    the ramp cross-checked against the card-6 surface's
    ``BlendSpec.weight`` at every fixture mass."""
    import na49_mixing.spectral_density as card6

    assert driver.KERNEL_BOUNDARY_GEV == BLEND_BOUNDARY_GEV
    assert driver.RAMP_WIDTH_GEV == RAMP_WIDTH_GEV
    surface_weight = card6.BlendSpec(BLEND_BOUNDARY_GEV, RAMP_WIDTH_GEV)

    for mass, region, uniforms in zip(TA_MASSES, TA_REGIONS, TA_UNIFORMS):
        assert driver._region_of(mass) == region, mass
        assert driver.expected_uniforms_of(mass) == uniforms, mass
        # The ramp: 0 at and below B, linear in the band, 1 at and
        # above B + W, and identical to the card-6 surface weight.
        expected = (
            0.0
            if mass <= BLEND_BOUNDARY_GEV
            else 1.0
            if mass >= BLEND_BOUNDARY_GEV + RAMP_WIDTH_GEV
            else (mass - BLEND_BOUNDARY_GEV) / RAMP_WIDTH_GEV
        )
        assert driver._expected_weight(mass) == expected, mass
        assert driver._expected_weight(mass) == surface_weight.weight(mass)

    # The band outcome: catalog and continuum regions never consume a
    # uniform and route to their own side; the transition routes to
    # the continuum exactly when u < w(M) (strict).
    for mass in (TA_MASSES[0], TA_MASSES[1]):
        for u in (0.0, 0.5, None):
            assert driver._expected_route(u, mass) == "catalog", (u, mass)
    for mass in (TA_MASSES[6], TA_MASSES[7], TA_MASSES[8]):
        for u in (0.0, 0.5, None):
            assert driver._expected_route(u, mass) == "continuum", (u, mass)

    # M = B: w = 0, the continuum is never routed (any u).
    for u in (0.0, 0.9999999999):
        assert driver._expected_route(u, _B) == "catalog", u
    # M = B + 0.125: w = 0.25.
    mass = _B + 0.125
    w = driver._expected_weight(mass)
    assert abs(w - 0.25) < 1e-12
    assert driver._expected_route(0.1, mass) == "continuum"
    assert driver._expected_route(0.4, mass) == "catalog"
    # The band boundary: u == w(M) is not < w(M).
    assert driver._expected_route(w, mass) == "catalog"
    # M = B + 0.375: w = 0.75.
    mass = _B + 0.375
    w = driver._expected_weight(mass)
    assert abs(w - 0.75) < 1e-12
    assert driver._expected_route(0.5, mass) == "continuum"
    assert driver._expected_route(0.9, mass) == "catalog"
    assert driver._expected_route(w, mass) == "catalog"
    # M = B + W - 1e-9: w = 1 - 2e-9 (epsilon below full weight).
    mass = _B + _W - 1e-9
    w = driver._expected_weight(mass)
    assert abs(w - (1.0 - 2e-9)) < 1e-15
    assert driver._expected_route(0.5, mass) == "continuum"
    assert driver._expected_route(0.9999999999, mass) == "catalog"
    assert driver._expected_route(w, mass) == "catalog"
    # The missing transition uniform is refused.
    with pytest.raises(SystemExit):
        driver._expected_route(None, _B + 0.125)


def test_direct_route_contract_on_stub_stream_ta_fixture(driver):
    """The card-6 ``route_source_decay`` routing contract on hand-fed
    stub streams (no generator): the consumed-uniform budget per
    region (0 / 1 / 0), the band outcome, and the uniform the
    transition consumes (the pre-chosen value, taken before any
    record draw)."""
    import na49_mixing.spectral_density as card6

    catalog = driver.audit_catalog()
    families = driver.audit_families()
    spec = driver.audit_blend()
    qn = driver.PROBE_SOURCE_QN
    cases = [
        # (mass, u, expected route, expected uniforms consumed)
        (_B - 0.375, 0.0, "catalog", 0),
        (_B, 0.0, "catalog", 1),
        (_B + 0.125, 0.1, "continuum", 1),
        (_B + 0.375, 0.9, "catalog", 1),
        (_B + _W, 0.0, "continuum", 0),
        (_B + _W + 0.3125, 0.9999999999, "continuum", 0),
    ]
    for mass, u, expected_route, expected_uniforms in cases:
        stub = driver._StubRNG([u])
        route = card6.route_source_decay(
            qn, mass, spec, catalog, families, stub
        )
        assert route == expected_route, (mass, u)
        assert len(stub.consumed) == expected_uniforms, (mass, u)
        if expected_uniforms == 1:
            assert stub.consumed == [u], (mass, u)
        else:
            assert stub.consumed == [], (mass, u)


def test_draw_budget_expectation_builder_ta_fixture(driver):
    """The per-region budget-table builder on the test-author's own
    synthetic source rows: the clean table (the totals, the route
    counts, the stamp agreements, and the transition-only fields) and
    the named-mismatch rows for every violation class (the uniform
    budget, the path draw, the route derivation, and the weight
    stamp)."""
    clean = [
        _row("catalog_region", 0, 1, "catalog", "catalog"),
        _row("catalog_region", 0, 1, "catalog", "catalog"),
        _row("transition", 1, 1, "continuum", "continuum"),
        _row("transition", 1, 1, "continuum", "continuum"),
        _row("transition", 1, 1, "catalog", "catalog"),
        _row("continuum_region", 0, 1, "continuum", "continuum"),
        _row("continuum_region", 0, 1, "continuum", "continuum"),
    ]
    table = driver._region_budget_table(clean)
    assert set(table) == {
        "catalog_region", "transition", "continuum_region",
    }
    catalog = table["catalog_region"]
    assert catalog["n_sources"] == 2
    assert catalog["routing_uniforms_total"] == 0
    assert catalog["expected_routing_uniforms_total"] == 0
    assert catalog["path_draws_total"] == 2
    assert catalog["expected_path_draws_total"] == 2
    assert catalog["routes_catalog"] == 2
    assert catalog["routes_continuum"] == 0
    assert catalog["weight_stamp_agreements"] == 2
    assert catalog["mismatches"] == []
    assert catalog["budget_exact"] is True
    transition = table["transition"]
    assert transition["n_sources"] == 3
    assert transition["routing_uniforms_total"] == 3
    assert transition["expected_routing_uniforms_total"] == 3
    assert transition["path_draws_total"] == 3
    assert transition["expected_path_draws_total"] == 3
    assert transition["routes_catalog"] == 1
    assert transition["routes_continuum"] == 2
    assert transition["routes_expected_catalog"] == 1
    assert transition["routes_expected_continuum"] == 2
    assert transition["u_lt_w_agreements"] == 3
    assert transition["band_uniforms_one_each"] is True
    assert transition["budget_exact"] is True
    continuum = table["continuum_region"]
    assert continuum["n_sources"] == 2
    assert continuum["routing_uniforms_total"] == 0
    assert continuum["path_draws_total"] == 2
    assert continuum["routes_catalog"] == 0
    assert continuum["routes_continuum"] == 2
    assert continuum["budget_exact"] is True

    # The violation rows: one per mismatch class, each named in the
    # table's mismatch list, each flipping its region's budget.
    uniform_bad = _row(
        "transition", 2, 1, "continuum", "continuum",
        uniform_budget_mismatch=True,
    )
    path_bad = _row(
        "catalog_region", 0, 2, "catalog", "catalog",
        path_budget_mismatch=True,
    )
    derivation_bad = _row(
        "transition", 1, 1, "continuum", "catalog",
        route_derivation_mismatch=True,
    )
    stamp_bad = _row(
        "continuum_region", 0, 1, "continuum", "continuum",
        stamped=0.9, expected_w=1.0, weight_stamp_mismatch=True,
    )
    broken = [uniform_bad, path_bad, derivation_bad, stamp_bad]
    table = driver._region_budget_table(broken)
    assert table["catalog_region"]["budget_exact"] is False
    assert table["catalog_region"]["mismatches"] == [path_bad]
    assert table["transition"]["budget_exact"] is False
    assert sorted(
        table["transition"]["mismatches"],
        key=lambda r: id(r),
    ) == sorted([uniform_bad, derivation_bad], key=lambda r: id(r))
    assert set(map(id, table["transition"]["mismatches"])) == {
        id(uniform_bad), id(derivation_bad),
    }
    assert table["transition"]["routing_uniforms_total"] == 3
    assert table["transition"]["u_lt_w_agreements"] == 1
    assert table["transition"]["band_uniforms_one_each"] is False
    assert table["continuum_region"]["budget_exact"] is False
    assert table["continuum_region"]["mismatches"] == [stamp_bad]
    assert table["continuum_region"]["weight_stamp_agreements"] == 0

    # The empty probe: no rows, no mismatches, the budgets hold
    # vacuously.
    table = driver._region_budget_table([])
    for region, block in table.items():
        assert block["n_sources"] == 0
        assert block["budget_exact"] is True, region


def test_census_pooling_and_disagreement_ta_fixture(driver):
    """The pooled routing-census arithmetic (pure integer pooling over
    the per-replica payload routing counts: the counts, the
    complement, and the fraction identities) and the
    census-disagreement arithmetic on the test-author's own
    synthetic payloads, including the refusal of the empty pool."""
    three = [
        {"routing": {"n_catalog": 7, "n_sources": 50}},
        {"routing": {"n_catalog": 13, "n_sources": 50}},
        {"routing": {"n_catalog": 20, "n_sources": 50}},
    ]
    pooled = driver.pooled_census_from_payloads(three)
    assert pooled["replicas"] == 3
    assert pooled["n_sources"] == 150
    assert pooled["n_catalog"] == 40
    assert pooled["n_continuum"] == 110
    assert pooled["fraction_catalog"] == 40 / 150
    assert pooled["fraction_continuum"] == 110 / 150
    assert pooled["fraction_catalog"] + pooled["fraction_continuum"] == 1.0

    four = [
        {"routing": {"n_catalog": 10, "n_sources": 100}},
        {"routing": {"n_catalog": 40, "n_sources": 200}},
        {"routing": {"n_catalog": 15, "n_sources": 100}},
        {"routing": {"n_catalog": 55, "n_sources": 200}},
    ]
    pooled = driver.pooled_census_from_payloads(four)
    assert pooled["replicas"] == 4
    assert pooled["n_sources"] == 600
    assert pooled["n_catalog"] == 120
    assert pooled["n_continuum"] == 480
    assert pooled["fraction_catalog"] == 0.2
    assert pooled["fraction_continuum"] == 0.8

    # The refusal: the empty pool and the zero-source pool.
    with pytest.raises(SystemExit):
        driver.pooled_census_from_payloads([])
    with pytest.raises(SystemExit):
        driver.pooled_census_from_payloads(
            [{"routing": {"n_catalog": 0, "n_sources": 0}}]
        )

    # The disagreement arithmetic: the absolute per-field
    # disagreements (fractions and counts) on hand fixtures with
    # exactly representable values.
    base = {
        "fraction_catalog": 0.25,
        "fraction_continuum": 0.75,
        "n_catalog": 40,
        "n_continuum": 120,
        "n_sources": 160,
    }
    assert driver.census_disagreement(base, dict(base)) == {
        "fraction_catalog": 0.0,
        "fraction_continuum": 0.0,
        "n_catalog": 0.0,
        "n_continuum": 0.0,
        "n_sources": 0.0,
    }
    shifted = dict(base)
    shifted["fraction_catalog"] = 0.5
    shifted["fraction_continuum"] = 0.5
    shifted["n_catalog"] = 37
    shifted["n_continuum"] = 119
    fields = driver.census_disagreement(base, shifted)
    assert fields["fraction_catalog"] == 0.25
    assert fields["fraction_continuum"] == 0.25
    assert fields["n_catalog"] == 3.0
    assert fields["n_continuum"] == 1.0
    assert fields["n_sources"] == 0.0


def test_verdict_criteria_logic_ta_fixture(driver):
    """The verdict-criteria logic on the test-author's own synthetic
    measurement artifacts: PASS iff (a) the draw budget is exact in
    all three regions, (b) the below-boundary path re-verifies
    bit-identity, (c) the boundary-matching factor is never
    consulted on the sampling path, and (d) the census is consistent
    within 1e-6 (inclusive bound); RECOMMENDATION with the named
    fired criteria otherwise, including the compound case."""
    clean_trace = {
        "draw_budget_exact": True,
        "region_budgets": {
            region: {"budget_exact": True}
            for region in ("catalog_region", "transition", "continuum_region")
        },
        "bit_identity_leg": {"reverified": True},
    }
    clean_isolation = {"factor_isolated": True}
    clean_census = {
        "census_within_bound": True,
        "leg_a": {"within_bound": True},
        "leg_b": {"exact": True},
    }
    outcome = driver.evaluate_verdict_criteria(
        clean_trace, clean_isolation, clean_census
    )
    assert outcome["verdict"] == "PASS"
    assert outcome["fired"] == []
    for criterion in ("a", "b", "c", "d"):
        assert outcome["criteria"][criterion]["fired"] is False

    # (a): the draw budget fails.
    broken_trace = {
        "draw_budget_exact": False,
        "region_budgets": {
            region: {"budget_exact": False}
            for region in ("catalog_region", "transition", "continuum_region")
        },
        "bit_identity_leg": {"reverified": True},
    }
    outcome = driver.evaluate_verdict_criteria(
        broken_trace, clean_isolation, clean_census
    )
    assert outcome["verdict"] == "RECOMMENDATION"
    assert [c["criterion"] for c in outcome["fired"]] == ["a"]

    # (b): the below-boundary leg fails.
    outcome = driver.evaluate_verdict_criteria(
        {**clean_trace, "bit_identity_leg": {"reverified": False}},
        clean_isolation,
        clean_census,
    )
    assert [c["criterion"] for c in outcome["fired"]] == ["b"]

    # (c): the isolation assertion fails.
    outcome = driver.evaluate_verdict_criteria(
        clean_trace, {"factor_isolated": False}, clean_census
    )
    assert [c["criterion"] for c in outcome["fired"]] == ["c"]

    # (d): the census leaves the bound.
    outcome = driver.evaluate_verdict_criteria(
        clean_trace,
        clean_isolation,
        {
            "census_within_bound": False,
            "leg_a": {"within_bound": False},
            "leg_b": {"exact": True},
        },
    )
    assert [c["criterion"] for c in outcome["fired"]] == ["d"]

    # The inclusive 1e-6 bound: a census that sits exactly on the
    # bound is within it (criterion d does not fire), one beyond it
    # fires.
    at_bound = {
        "census_within_bound": True,
        "leg_a": {"within_bound": True, "max_disagreement": 1e-6},
        "leg_b": {"exact": True},
    }
    beyond_bound = {
        "census_within_bound": False,
        "leg_a": {"within_bound": False, "max_disagreement": 2e-6},
        "leg_b": {"exact": True},
    }
    assert (
        driver.evaluate_verdict_criteria(
            clean_trace, clean_isolation, at_bound
        )["criteria"]["d"]["fired"]
        is False
    )
    assert (
        driver.evaluate_verdict_criteria(
            clean_trace, clean_isolation, beyond_bound
        )["criteria"]["d"]["fired"]
        is True
    )

    # The compound case: every criterion fires, in order.
    outcome = driver.evaluate_verdict_criteria(
        {**broken_trace, "bit_identity_leg": {"reverified": False}},
        {"factor_isolated": False},
        beyond_bound,
    )
    assert outcome["verdict"] == "RECOMMENDATION"
    assert [c["criterion"] for c in outcome["fired"]] == ["a", "b", "c", "d"]
    for criterion in ("a", "b", "c", "d"):
        assert outcome["criteria"][criterion]["fired"] is True


def test_zero_draw_measurement_path_ta_fixture(driver):
    """The zero-draw invariants on the test-author's own technique
    (independent of the solver's token scan): an AST analysis of the
    committed driver pinning every ``np.random`` construction site to
    the pinned bit-identity seed 20300913 (no other seed literal
    anywhere in the driver), the reserved audit block
    20301040..20301055 appearing nowhere except its
    pre-registration constants, the pinned seed constants, and the
    pre-registration builder's zero-draw record on the on-disk
    substrate."""
    source = NESTED_DRIVER.read_text(encoding="utf-8")
    tree = ast.parse(source)

    # The seed constants, pinned.
    assert driver.BIT_IDENTITY_SEED == RAW_SEED == 20300913
    assert driver.BIT_IDENTITY_MASTER_SEED == KERNEL_MASTER_SEED == 20260913
    assert driver.BIT_IDENTITY_REPLICA_INDEX == KERNEL_REPLICA_INDEX == 0
    assert driver.RESERVED_MASTER_SEED == RESERVED_MASTER_SEED == 20260920
    assert driver.RESERVED_BLOCK_FIRST == RESERVED_BLOCK_FIRST == 20301040
    assert driver.RESERVED_BLOCK_LAST == RESERVED_BLOCK_LAST == 20301055
    # The reserved block is 16 seeds, disjoint from the probe.
    assert driver.RESERVED_BLOCK_LAST == driver.RESERVED_BLOCK_FIRST + 15
    assert driver.BIT_IDENTITY_SEED < driver.RESERVED_BLOCK_FIRST

    # Every ``np.random.<attr>`` construction site is a
    # default_rng / SeedSequence construction, and every such
    # construction is seeded (through the nested SeedSequence) from
    # the pinned bit-identity seed constant: no other seed literal
    # anywhere in the driver.
    rng_sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr in ("default_rng", "SeedSequence")
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "random"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "np"
        ):
            rng_sites.append((func.attr, node))
    assert {attr for attr, _site in rng_sites} == {
        "default_rng", "SeedSequence",
    }, "the driver constructs no RNG machinery other than the " \
      "pinned-seed re-evaluation stream"
    assert rng_sites, "the driver carries its pinned-seed stream"

    def _seed_root(arg: ast.AST):
        node = arg
        while isinstance(node, ast.Call):
            if not node.args:
                return None
            node = node.args[0]
        if isinstance(node, ast.Name):
            return node.id
        return None

    for _attr, node in rng_sites:
        assert node.args, "the RNG construction carries its seed"
        assert _seed_root(node.args[0]) == "BIT_IDENTITY_SEED", (
            "every np.random construction in the driver is seeded "
            "from the pinned bit-identity seed (20300913); no other "
            "seed literal exists in the driver"
        )

    # The reserved audit block never appears as a call argument:
    # the only integer literals in the reserved range are the two
    # pre-registration constants themselves.
    reserved_values = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
            and RESERVED_BLOCK_FIRST <= node.value <= RESERVED_BLOCK_LAST
        ):
            reserved_values.add(node.value)
    assert reserved_values == {RESERVED_BLOCK_FIRST, RESERVED_BLOCK_LAST}

    # The pre-registration builder's zero-draw record on the on-disk
    # substrate (computed, not asserted).
    substrate = driver.substrate_digests()
    assert substrate == {
        "builder_module_sha256": BUILDER_SHA256,
        "card6_surface_sha256": CARD6_SHA256,
        "engine_module_sha256": ENGINE_SHA256,
        "catalog_sha256": CATALOG_SHA256,
        "families_config_sha256": FAMILIES_SHA256,
        "r32_driver_nested_sha256": R32_DRIVER_SHA256,
        "r32_driver_outer_sha256": R32_DRIVER_SHA256,
    }
    prereg = driver.build_pre_registration(substrate, [])
    assert prereg["written_before_measurement"] is True
    assert prereg["zero_draw"]["draws"] == 0
    assert prereg["zero_draw"]["no_fitting"] is True
    assert prereg["zero_draw"]["optimizer_invoked"] is False
    reserved = prereg["zero_draw"]["reserved_audit_block"]
    assert reserved["master_seed"] == RESERVED_MASTER_SEED
    assert reserved["raw_seeds"] == [RESERVED_BLOCK_FIRST,
                                     RESERVED_BLOCK_LAST]
    assert reserved["fired"] is False
    assert prereg["sealed_inputs"] == SEALED_INPUT_DIGESTS
    assert prereg["substrate"] == substrate


# ----------------------------------------------------------------------
# Part 2: the sealed artifacts (run-dir pins)
# ----------------------------------------------------------------------


def test_driver_dual_tree_byte_identity():
    """The nested and outer audit driver copies are byte-identical at
    the pinned digest."""
    assert _sha256_file(NESTED_DRIVER) == AUDIT_DRIVER_SHA256
    assert _sha256_file(OUTER_DRIVER) == AUDIT_DRIVER_SHA256
    assert NESTED_DRIVER.read_bytes() == OUTER_DRIVER.read_bytes()


def test_surface_byte_stability():
    """The builder blend path, the card-6 blend routing surface, the
    engine module, the catalog, the continuum families config, and
    the R32 driver (both trees) stay at their pinned on-disk digests
    (re-hashed independently of the driver)."""
    pinned = {
        "dev/mixing2006/na49_mixing/resonance_pp_builder.py": BUILDER_SHA256,
        "dev/mixing2006/na49_mixing/spectral_density.py": CARD6_SHA256,
        "dev/mixing2006/na49_mixing/threshold_metric.py": ENGINE_SHA256,
        "dev/mixing2006/config/v2_singlet/resonance_catalog.json": (
            CATALOG_SHA256
        ),
        "dev/mixing2006/config/v2_singlet/continuum_resonances.json": (
            FAMILIES_SHA256
        ),
        "dev/mixing2006/helpers/benchmarking/"
        "staged_r32_spectral_systematic.py": R32_DRIVER_SHA256,
        "helpers/benchmarking/staged_r32_spectral_systematic.py": (
            R32_DRIVER_SHA256
        ),
    }
    for rel, digest in pinned.items():
        assert _sha256_file(MONO_ROOT / rel) == digest, rel


def test_sealed_run_dir_seal_pins():
    """The seal manifest and the SHA256SUMS.txt integrity of the
    sealed audit run dir: the on-disk seal manifest carries 7 files
    totalling 2,649,524 bytes (the clean-environment record is the
    documented post-seal file and is never seal-covered); every
    listed file re-hashes to its recorded digest; the
    SHA256SUMS.txt inventory covers the 6 pre-seal payload files with
    monorepo-root-relative paths, and every entry re-hashes from the
    monorepo root; the clean-environment record verifies."""
    run_dir = _skip_if_no_run_dir()
    seal = _read_json(run_dir / "seal_manifest.json")
    assert seal["card"] == CARD_ID
    assert seal["file_count"] == SEAL_FILE_COUNT == 7
    assert seal["total_bytes"] == SEAL_TOTAL_BYTES == 2649524
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
        prefix = "dev/00runs/r4_audit_builder_blend_path_routing_weight_"
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
    assert clean["seal_verification"]["files_checked"] == 6
    assert clean["probe_returncodes"] == {
        "draw-budget": 0, "isolation": 0, "census": 0, "verdict": 0,
    }
    assert clean["probe_failures"] == []
    assert clean["sealed_artifact_sha256"] == clean[
        "recomputed_artifact_sha256"
    ]
    assert clean["probe_tool"]["sha256"] == AUDIT_DRIVER_SHA256


def test_sealed_run_dir_trace_pins():
    """The draw_budget_trace.json pins: the probe configuration at
    the sealed seed (the first 1,000 event ids of the sealed R32 v0
    bit-identity stream), the region budgets 366 / 151 / 1483 sources
    with 0 / 1 / 0 routing uniforms and the exact budget in every
    region, the band outcome u < w(M) re-derived on all 151 band
    rows, the probe census, the sealed-envelope comparisons, the
    bit-identity leg (6000 / 6000 layers, 209 / 209 below-boundary
    finals, 1461 / 1461 legacy finals), the call aggregate, and the
    zero-draw stamps."""
    run_dir = _skip_if_no_run_dir()
    assert _sha256_file(run_dir / "draw_budget_trace.json") == TRACE_SHA256
    trace = _read_json(run_dir / "draw_budget_trace.json")
    assert trace["card"] == CARD_ID
    assert trace["draw_budget_exact"] is True
    config = trace["probe_configuration"]
    assert config["seed"] == RAW_SEED
    assert config["kernel_master_seed"] == KERNEL_MASTER_SEED
    assert config["kernel_replica_index"] == KERNEL_REPLICA_INDEX
    assert config["n_events"] == 1000
    assert config["event_ids"] == [0, 999]

    budgets = trace["region_budgets"]
    pinned_sources = {
        "catalog_region": 366, "transition": 151, "continuum_region": 1483,
    }
    pinned_uniforms = {
        "catalog_region": 0, "transition": 151, "continuum_region": 0,
    }
    for region, n_sources in pinned_sources.items():
        block = budgets[region]
        assert block["n_sources"] == n_sources, region
        assert block["routing_uniforms_total"] == pinned_uniforms[region]
        assert block["expected_routing_uniforms_total"] == (
            pinned_uniforms[region]
        )
        assert block["path_draws_total"] == n_sources, region
        assert block["expected_path_draws_total"] == n_sources, region
        assert block["routes_expected_catalog"] == block["routes_catalog"]
        assert block["routes_expected_continuum"] == block["routes_continuum"]
        assert block["weight_stamp_agreements"] == n_sources, region
        assert block["budget_exact"] is True, region
    assert budgets["transition"]["u_lt_w_agreements"] == 151
    assert budgets["transition"]["band_uniforms_one_each"] is True
    assert budgets["transition"]["routes_catalog"] == 79
    assert budgets["transition"]["routes_continuum"] == 72

    assert trace["probe_census"] == PROBE_CENSUS
    for key in ("probe_below_le_sealed_below",
                "probe_in_band_le_sealed_in_band",
                "probe_above_le_sealed_above",
                "probe_catalog_routed_le_sealed",
                "probe_continuum_routed_le_sealed"):
        assert trace["sealed_envelope"][key] is True, key

    leg = trace["bit_identity_leg"]
    assert leg["layer_compared"] == BIT_IDENTITY_LAYER[0]
    assert leg["layer_matched"] == BIT_IDENTITY_LAYER[1]
    assert leg["below_boundary_finals_compared"] == (
        BIT_IDENTITY_BELOW_BOUNDARY_FINALS[0]
    )
    assert leg["below_boundary_finals_matched"] == (
        BIT_IDENTITY_BELOW_BOUNDARY_FINALS[1]
    )
    assert leg["below_boundary_finals_skipped_perturbed"] == (
        BIT_IDENTITY_BELOW_BOUNDARY_FINALS[2]
    )
    assert leg["legacy_finals_compared"] == BIT_IDENTITY_LEGACY_FINALS[0]
    assert leg["legacy_finals_matched"] == BIT_IDENTITY_LEGACY_FINALS[1]
    assert leg["reverified"] is True

    assert trace["aggregate"]["events"] == 1000
    assert trace["aggregate"]["attempts_total"] == 1000
    assert trace["aggregate"]["retry_route_calls_total"] == 0
    assert trace["aggregate"]["child_calls_total"] == 2151
    assert trace["aggregate"]["kin_calls_total"] == 4206
    assert trace["aggregate"]["event_hash_digest"] == EVENT_HASH_DIGEST
    assert len(trace["per_event_rows"]) == 1000
    assert trace["no_fitting"] is True
    assert trace["optimizer_invoked"] is False
    assert trace["draws"] == 0


def test_sealed_run_dir_isolation_pins():
    """The isolation_assertion.json pins: the 200-event spy fixture
    with 0 boundary-factor consultations (all four spy surfaces,
    the cache side channel empty before and after), the 7-function
    static assertion (the sampling call path free of the
    double-counting factor), the 48-row direct route contract with
    the boundary crossings at M = B (w = 0, one uniform consumed,
    the catalog route) and M = B + W (w = 1, zero uniforms, the
    continuum route), the routing-weight consistency, and the
    factor-isolated verdict."""
    run_dir = _skip_if_no_run_dir()
    assert _sha256_file(run_dir / "isolation_assertion.json") == (
        ISOLATION_SHA256
    )
    isolation = _read_json(run_dir / "isolation_assertion.json")
    assert isolation["card"] == CARD_ID
    spy = isolation["spy_fixture"]
    assert spy["events"] == SPY_EVENTS == 200
    assert spy["per_region_sources"] == SPY_PER_REGION
    for surface in ("blended_density",
                    "blended_normalization_report",
                    "_blended_pieces",
                    "_blended_support"):
        assert spy["spy_consultations"][surface] == 0, surface
    assert spy["spy_consultations_total"] == 0
    assert spy["blend_pieces_cache_before"] == 0
    assert spy["blend_pieces_cache_after"] == 0
    assert spy["fixture_adequate"] is True
    assert spy["budgets_ok"] is True
    assert spy["stamps_exact"] is True

    static = isolation["static_assertion"]
    assert len(static["checked_functions"]) == 7
    for name in ("decay_source", "_decay_blended", "_decay_resonance",
                 "_decay_continuum"):
        assert (
            f"na49_mixing.resonance_pp_builder.{name}"
            in static["checked_functions"]
        )
    for name in ("route_source_decay", "_require_open_catalog",
                 "_require_active_family"):
        assert (
            f"na49_mixing.spectral_density.{name}"
            in static["checked_functions"]
        )
    assert static["forbidden_names"] == [
        "_blended_pieces", "_blended_support", "blended_density",
        "blended_normalization_report", "matching_factor",
    ]
    assert static["problems"] == []
    assert static["clean"] is True

    direct = isolation["direct_route_contract"]
    assert direct["problems"] == []
    assert direct["clean"] is True
    rows = direct["rows"]
    assert len(rows) == 48
    assert all(row["pass"] for row in rows)
    # The boundary crossings: M = B (w = 0, one uniform, catalog) and
    # M = B + W (w = 1, zero uniforms, continuum).
    at_b = [row for row in rows if row["mass"] == BLEND_BOUNDARY_GEV]
    assert at_b, "the direct contract crosses the exact lower boundary"
    assert all(
        row["region"] == "transition" and row["w"] == 0.0
        and row["route"] == "catalog"
        and row["uniforms_consumed"] == 1
        and row["uniforms_expected"] == 1
        for row in at_b
    )
    at_bw = [row for row in rows
             if row["mass"] == BLEND_BOUNDARY_GEV + RAMP_WIDTH_GEV]
    assert at_bw, "the direct contract crosses the exact upper boundary"
    assert all(
        row["region"] == "continuum_region" and row["w"] == 1.0
        and row["route"] == "continuum"
        and row["uniforms_consumed"] == 0
        and row["uniforms_expected"] == 0
        for row in at_bw
    )

    consistency = isolation["routing_weight_consistency"]
    assert consistency["stamps_exact"] is True
    assert consistency["budgets_ok"] is True
    assert consistency["route_outcomes_exact"] is True
    assert isolation["factor_isolated"] is True
    assert isolation["no_fitting"] is True
    assert isolation["optimizer_invoked"] is False
    assert isolation["draws"] == 0


def test_sealed_run_dir_census_pins():
    """The census_consistency.json pins: leg A within the 1e-6 bound
    at max disagreement 0.0, the v0 fractions 0.22329625 (production)
    and 0.22357 (confirmation) exact on both the recomputed and the
    sealed side, leg B exact on the 151 band rows, the
    informational leg-C bracket [0.188, 0.2624] carrying the sealed
    production fraction, the 72 payload identity stamps, and the
    zero-draw stamps."""
    run_dir = _skip_if_no_run_dir()
    assert _sha256_file(run_dir / "census_consistency.json") == CENSUS_SHA256
    census = _read_json(run_dir / "census_consistency.json")
    assert census["card"] == CARD_ID
    leg_a = census["leg_a"]
    assert leg_a["disagreement_bound"] == 1e-06
    assert leg_a["max_disagreement"] == 0.0
    assert leg_a["within_bound"] is True

    variant0 = leg_a["variants"][0]
    production = variant0["production"]
    assert production["recomputed"] == PRODUCTION_V0
    assert production["sealed"]["fraction_catalog"] == (
        PRODUCTION_V0["fraction_catalog"]
    )
    assert production["sealed"]["n_sources"] == (
        PRODUCTION_V0["n_sources"]
    )
    assert production["sealed"]["n_catalog"] == PRODUCTION_V0["n_catalog"]
    assert production["sealed"]["n_continuum"] == (
        PRODUCTION_V0["n_continuum"]
    )
    assert variant0["gate_fraction_catalog_quoted"] == GATE_CENSUS_V0
    confirmation = variant0["confirmation"]
    assert confirmation["recomputed"] == CONFIRMATION_V0
    assert confirmation["sealed"]["fraction_catalog"] == (
        CONFIRMATION_V0["fraction_catalog"]
    )
    assert variant0["worst_disagreement"] == 0.0
    assert census["leg_b"]["n_band_rows"] == 151
    assert census["leg_b"]["probe_census_matches"] is True
    assert census["leg_b"]["band_outcomes_rederived_ok"] is True
    assert census["leg_b"]["region_determinism_ok"] is True
    assert census["leg_b"]["exact"] is True

    leg_c = census["leg_c_informational"]
    assert leg_c["p_below_boundary"] == LEG_C_P_BELOW_BOUNDARY
    assert leg_c["p_in_band"] == LEG_C_P_IN_BAND
    assert leg_c["bracket"] == [
        LEG_C_P_BELOW_BOUNDARY,
        LEG_C_P_BELOW_BOUNDARY + LEG_C_P_IN_BAND,
    ]
    assert leg_c["sealed_fraction"] == GATE_CENSUS_V0
    assert leg_c["sealed_fraction_in_bracket"] is True

    stamps = census["payload_identity_stamps"]
    assert len(stamps) == 72
    assert all(stamp["builder_digest_ok"] for stamp in stamps)
    assert all(stamp["driver_digest_ok"] for stamp in stamps)
    assert all(stamp["catalog_digest_ok"] for stamp in stamps)
    assert all(stamp["blend_block_ok"] for stamp in stamps)
    assert census["census_within_bound"] is True
    assert census["no_fitting"] is True
    assert census["optimizer_invoked"] is False
    assert census["draws"] == 0


def test_sealed_run_dir_verdict_pins():
    """The verdict.json content pins: the PASS verdict with 0 named
    findings, the four pre-registered criteria all unfired, the
    draw-budget summary (the region budget table 366 / 151 / 1483
    sources with 0 / 1 / 0 routing uniforms, the bit-identity leg
    re-verified), the isolation summary, the census summary (max
    disagreement 0.0 against the 1e-6 bound), the zero-draw record,
    the pinned surface digests, the sealed input digests, and the
    probe-set contract."""
    run_dir = _skip_if_no_run_dir()
    assert _sha256_file(run_dir / "verdict.json") == VERDICT_SHA256
    verdict = _read_json(run_dir / "verdict.json")
    assert verdict["card"] == CARD_ID
    assert verdict["verdict"] == "PASS"
    assert verdict["findings"] == []
    assert verdict["recommendation"] is None
    contract = verdict["verdict_contract"]
    assert contract["criteria"] == ["a", "b", "c", "d"]
    for criterion in ("a", "b", "c", "d"):
        block = verdict["criteria"][criterion]
        assert block["fired"] is False, criterion
    assert verdict["criteria"]["a"]["region_budgets_exact"] == {
        "catalog_region": True, "transition": True,
        "continuum_region": True,
    }

    probe = verdict["probe_set"]
    assert probe["events"] == 1000
    assert probe["spy_fixture_events"] == SPY_EVENTS
    assert probe["seed"] == RAW_SEED
    assert probe["kernel_master_seed"] == KERNEL_MASTER_SEED
    assert probe["kernel_replica_index"] == KERNEL_REPLICA_INDEX
    assert verdict["bounds"]["census_disagreement_bound"] == 1e-06
    assert verdict["bounds"]["bin_spread_p95_bound"] == 2.0
    assert verdict["bounds"]["feeddown_spread_max_bound"] == 3.0

    summary = verdict["draw_budget_summary"]
    assert summary["draw_budget_exact"] is True
    regions = summary["region_budgets"]
    assert regions["catalog_region"]["n_sources"] == 366
    assert regions["catalog_region"]["routing_uniforms_total"] == 0
    assert regions["catalog_region"]["path_draws_total"] == 366
    assert regions["transition"]["n_sources"] == 151
    assert regions["transition"]["routing_uniforms_total"] == 151
    assert regions["transition"]["path_draws_total"] == 151
    assert regions["continuum_region"]["n_sources"] == 1483
    assert regions["continuum_region"]["routing_uniforms_total"] == 0
    assert regions["continuum_region"]["path_draws_total"] == 1483
    for region, block in regions.items():
        assert block["budget_exact"] is True, region
    assert summary["bit_identity_reverified"] is True

    isolation_summary = verdict["isolation_summary"]
    assert isolation_summary["factor_isolated"] is True
    assert isolation_summary["spy_consultations_total"] == 0
    assert isolation_summary["static_clean"] is True
    assert isolation_summary["direct_contract_clean"] is True
    assert isolation_summary["stamps_exact"] is True

    census_summary = verdict["census_summary"]
    assert census_summary["census_within_bound"] is True
    assert census_summary["max_disagreement"] == 0.0
    assert census_summary["bound"] == 1e-06

    zero_draw = verdict["zero_draw"]
    assert zero_draw["draws"] == 0
    assert zero_draw["no_fitting"] is True
    assert zero_draw["optimizer_invoked"] is False
    assert zero_draw["reserved_audit_block_fired"] is False

    surface = verdict["surface"]
    assert surface["builder_sha256"] == BUILDER_SHA256
    assert surface["card6_sha256"] == CARD6_SHA256
    assert surface["byte_unchanged_by_this_audit"] is True
    assert verdict["sealed_inputs"] == SEALED_INPUT_DIGESTS


def test_sealed_run_dir_prereg_provenance_pins():
    """The pre_registration.json sha256 pin (immutability), the
    probe-set contract, the region rules against the on-disk driver
    constants, the bounds, the zero-draw contract with the reserved
    audit block 20301040 + k not fired, the sealed-input and
    sealed-run-dir re-verification records (29 + 55 + 12 entries,
    0 mismatches), the substrate digests, and the provenance.json
    driver digest pin (nested == outer == the on-disk dual-tree
    digest)."""
    run_dir = _skip_if_no_run_dir()
    driver = _load_driver()

    assert _sha256_file(run_dir / "pre_registration.json") == (
        PRE_REGISTRATION_SHA256
    )
    prereg = _read_json(run_dir / "pre_registration.json")
    assert prereg["card"] == CARD_ID
    assert prereg["written_before_measurement"] is True
    assert prereg["probe_set"]["events"] == 1000
    assert prereg["probe_set"]["event_ids"] == [0, 999]
    assert prereg["probe_set"]["seed"] == RAW_SEED
    assert prereg["probe_set"]["kernel_master_seed"] == KERNEL_MASTER_SEED
    assert prereg["probe_set"]["kernel_replica_index"] == KERNEL_REPLICA_INDEX
    assert prereg["probe_set"]["spy_fixture_events"] == SPY_EVENTS
    assert prereg["blend"]["boundary_gev"] == BLEND_BOUNDARY_GEV
    assert prereg["blend"]["ramp_width_gev"] == RAMP_WIDTH_GEV
    # The region rules match the on-disk driver constants.
    assert prereg["regions"]["catalog_region"] == (
        driver.REGION_RULES["catalog_region"]
    )
    assert prereg["regions"]["transition"] == driver.REGION_RULES["transition"]
    assert prereg["regions"]["continuum_region"] == (
        driver.REGION_RULES["continuum_region"]
    )
    assert prereg["census_consistency"]["leg_a"]["disagreement_bound"] == 1e-06
    assert prereg["new_spread_bounds"]["bin_spread_p95_bound"] == 2.0
    assert prereg["new_spread_bounds"]["feeddown_spread_max_bound"] == 3.0
    assert prereg["verdict_contract"]["criteria"] == ["a", "b", "c", "d"]
    zero_draw = prereg["zero_draw"]
    assert zero_draw["draws"] == 0
    assert zero_draw["no_fitting"] is True
    assert zero_draw["optimizer_invoked"] is False
    reserved = zero_draw["reserved_audit_block"]
    assert reserved["master_seed"] == RESERVED_MASTER_SEED
    assert reserved["raw_seeds"] == [RESERVED_BLOCK_FIRST,
                                     RESERVED_BLOCK_LAST]
    assert reserved["fired"] is False
    assert prereg["sealed_inputs"] == SEALED_INPUT_DIGESTS
    verified = prereg["sealed_run_dirs_verified"]
    assert [entry["entries"] for entry in verified] == [29, 55, 12]
    assert all(entry["verified"] is True for entry in verified)
    assert prereg["substrate"]["builder_module_sha256"] == BUILDER_SHA256
    assert prereg["substrate"]["card6_surface_sha256"] == CARD6_SHA256
    assert prereg["substrate"]["engine_module_sha256"] == ENGINE_SHA256
    assert prereg["substrate"]["catalog_sha256"] == CATALOG_SHA256
    assert prereg["substrate"]["families_config_sha256"] == FAMILIES_SHA256
    assert prereg["substrate"]["r32_driver_nested_sha256"] == (
        R32_DRIVER_SHA256
    )
    assert prereg["substrate"]["r32_driver_outer_sha256"] == (
        R32_DRIVER_SHA256
    )

    assert _sha256_file(run_dir / "provenance.json") == PROVENANCE_SHA256
    provenance = _read_json(run_dir / "provenance.json")
    assert provenance["card"] == CARD_ID
    assert provenance["driver_sha256"]["nested"] == AUDIT_DRIVER_SHA256
    assert provenance["driver_sha256"]["outer"] == AUDIT_DRIVER_SHA256
    assert _sha256_file(NESTED_DRIVER) == (
        provenance["driver_sha256"]["nested"]
    )
    assert _sha256_file(OUTER_DRIVER) == provenance["driver_sha256"]["outer"]
    frozen = provenance["frozen_substrate"]
    assert frozen["builder_module_sha256"] == BUILDER_SHA256
    assert frozen["card6_surface_sha256"] == CARD6_SHA256
    assert frozen["engine_module_sha256"] == ENGINE_SHA256
    assert frozen["catalog_sha256"] == CATALOG_SHA256
    assert frozen["families_config_sha256"] == FAMILIES_SHA256
    assert frozen["r32_driver_nested_sha256"] == R32_DRIVER_SHA256
    assert frozen["r32_driver_outer_sha256"] == R32_DRIVER_SHA256
    assert [entry["entries"] for entry in
            provenance["sealed_run_dirs_verified"]] == [29, 55, 12]
    assert all(entry["verified"] is True
               for entry in provenance["sealed_run_dirs_verified"])
    assert provenance["zero_draw"]["draws"] == 0
    assert provenance["zero_draw"]["no_fitting"] is True
    assert provenance["zero_draw"]["optimizer_invoked"] is False
    assert provenance["zero_draw"]["reserved_audit_block_fired"] is False


