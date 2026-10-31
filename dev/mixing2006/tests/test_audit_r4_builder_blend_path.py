"""Hermetic tests for the R4 builder blend-path audit (Round 4, card 3).

Task ``TASK-20260920-r4-audit-builder-blend-path-routing-weight`` (the
Gate 3 decision card follow-up obligation 3; plan section 9.3, lines
625-632).  The tests exercise the card's pre-registered measurement
contract against the committed audit driver:

- ``test_module_contract_pins``: the pre-registered audit identity (the
  1,000-event probe at the sealed bit-identity seed, the 200-event spy
  fixture, the blend constants, the three region rules, the bounds,
  the factor surface, and the reserved audit block).
- ``test_driver_copies_byte_identical``: the nested and outer driver
  copies are byte-identical (the dual-tree register).
- ``test_zero_draw_source_contract``: the driver carries no random
  module and its only numpy random machinery is the pinned probe-stream
  construction (the legacy per-event generator of the sealed R32 v0
  bit-identity stream configuration); the pre-registration builder
  records ``draws: 0``, ``no_fitting: true``, ``optimizer_invoked:
  false``, and the unfired reserved block.
- ``test_expected_route_hand_case``: the audit's independent
  reimplementation of the card-6 routing contract (the three-region
  rule plus ``u < w(M)``) against hand-computed cases, including the
  exact boundary values M = B and M = B + W.
- ``test_direct_route_contract``: the hand-fed stub-stream contract
  against the pinned card-6 surface passes on every pinned (mass,
  uniform) pair.
- ``test_pooled_census_and_disagreement``: the routing-census pooling
  from sealed per-replica payloads against a hand-computed fixture, and
  the 1e-6 disagreement bound semantics.
- ``test_probe_event_white_box``: a small pinned slice of the
  1,000-event probe machinery: the per-region draw budgets (0 / 1 / 0),
  the exact routing stamps, the source-layer bit-identity leg, and the
  below-boundary catalog-route finals against the R23 builder
  convention (the sealed R32 v0 leg, re-verified at probe scale).
- ``test_isolation_spy_and_static``: the white-box boundary-matching-
  factor isolation on the pinned 200-event call fixture (zero spy
  consultations, an empty blend pieces cache before and after, exact
  routing-weight stamps) plus the AST static assertion over the
  sampling-path functions of both pinned surfaces.
- ``test_sealed_r32_readonly_pins``: the sealed R32 production,
  confirmation, and R23 archives verify their seals, the sealed census
  anchors match the pre-registration, and the sealed input digests
  re-hash.
- ``test_pinned_surface_byte_stability``: the builder surface, the
  card-6 surface, the engine, the catalog, and the continuum config
  stay at their pinned digests, and the frozen R32 driver is
  byte-identical across both helper trees.
- ``test_layout_sync``: the helper-tree layout sync reports no logical
  drift.
- ``test_freeze_manifest_repin``: the manifest carries the new
  inventory entries with the on-disk digests (strict path-sorted) and
  the ``freeze_exceptions`` entry naming this card.
- ``test_audit_run_artifact_pins``: the sealed audit run dir's
  artifacts (activated once the run dir exists; skipped with a
  recorded reason before the audit run): the pre-registration
  contract, the provenance, the draw-budget trace, the isolation
  assertion, the census consistency, the verdict, and the seal +
  clean-validation records.

The tests read the sealed R32/R23 archives read-only (their seals
verify) and never write outside ``tmp_path``.  No event generation
beyond the pinned probe fixtures (a fixed-seed re-evaluation of the
sealed R32 v0 bit-identity stream configuration, so no new random
number is consumed), no optimizer, no external SPS tables.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from tests._sealed_archive import skip_unless_sealed

NESTED_ROOT = Path(__file__).resolve().parent.parent
MONO_ROOT = NESTED_ROOT.parent.parent
DRIVER_REL = Path("helpers") / "benchmarking" / (
    "audit_r4_builder_blend_path.py")
NESTED_DRIVER = NESTED_ROOT / DRIVER_REL
OUTER_DRIVER = MONO_ROOT / DRIVER_REL
SYNC_CHECKER = MONO_ROOT / "helpers" / "benchmarking" / (
    "check_helpers_layout_sync.py"
)
MANIFEST_PATH = NESTED_ROOT / "config" / "v2_singlet" / (
    "legacy_v1_manifest.json")
CARD = "TASK-20260920-r4-audit-builder-blend-path-routing-weight"
BUILDER_REL = "dev/mixing2006/na49_mixing/resonance_pp_builder.py"
BUILDER_SHA256 = "4ab612609e1ee0bc1e037dcbb2815b6484306ad32402b7289e6636f8e829cdb1"
CARD6_REL = "dev/mixing2006/na49_mixing/spectral_density.py"
CARD6_SHA256 = "561c65ee9a77a1b7f7eb07a7619dafb584779a51ccb23d7fe5651c0605e12563"
ENGINE_REL = "dev/mixing2006/na49_mixing/threshold_metric.py"
ENGINE_SHA256 = "ee240f44645d2e592acfae462f7d1e9c3b3dd8cd78c96bd40b7b73509f7fcc63"
CATALOG_REL = "dev/mixing2006/config/v2_singlet/resonance_catalog.json"
CATALOG_SHA256 = "6e4de7372a5bbf99affd6fa3ccc295c4cef09404dab3b17acd8816cda762bdcd"
FAMILIES_REL = "dev/mixing2006/config/v2_singlet/continuum_resonances.json"
FAMILIES_SHA256 = "242f906cf040eadbe3796dfcfb61d0d51535277399858c901a162e6653d635e1"
R32_DRIVER_REL = "dev/mixing2006/helpers/benchmarking/staged_r32_spectral_systematic.py"
R32_DRIVER_SHA256 = "99454facce856c93505e6c4b0d4e47b85eeb0ef54f72b1631e3d29fcc511cb90"
R32_PROD_DIR = "dev/00runs/r32_spectral_systematic_pp_2026-09-18"
R32_CONF_DIR = "dev/00runs/r32_spectral_systematic_pp_confirmation_2026-09-19"
R23_DIR = "dev/00runs/r23_resonance_only_pp_ablation_2026-09-13"

# The sealed audit run dir (the audit's own Phase 2 output).  The pins
# activate once the dir exists; before the audit run they skip with a
# recorded reason (the R32 pins convention, activated at phase 2).
RUN_DIR_CANDIDATES = tuple(
    (MONO_ROOT / "dev" / "00runs").glob(
        "r4_audit_builder_blend_path_routing_weight_*")
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_driver():
    """The committed dual-tree driver, loaded once under a unique
    module name (the driver's own ``_locate_roots`` wiring resolves
    the nested roots at import time)."""
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


# ----------------------------------------------------------------------
# Module contract pins
# ----------------------------------------------------------------------


def test_module_contract_pins(driver):
    assert driver.CARD == CARD
    assert driver.RUN_LABEL == "r4_audit_builder_blend_path_routing_weight"
    # The probe (the sealed R32 v0 bit-identity stream configuration).
    assert driver.PROBE_EVENTS == 1000
    assert driver.SPY_EVENTS == 200
    assert driver.BIT_IDENTITY_SEED == 20300913
    assert driver.BIT_IDENTITY_MASTER_SEED == 20260913
    assert driver.BIT_IDENTITY_REPLICA_INDEX == 0
    assert driver.BIT_IDENTITY_EVENTS == 10000
    # The blend constants (the R32 v0 nominal variant, verbatim).
    assert driver.KERNEL_BOUNDARY_GEV == 2.680194110727528
    assert driver.RAMP_WIDTH_GEV == 0.5
    # The sealed anchor numbers.
    assert driver.PRODUCTION_CENSUS_V0 == {
        "n_sources": 1600000,
        "n_catalog": 357274,
        "fraction_catalog": 0.22329625,
        "fraction_continuum": 0.77670375,
    }
    assert driver.GATE_CENSUS_V0 == 0.22329625
    assert driver.CONFIRMATION_CENSUS_V0 == {
        "n_sources": 3200000,
        "n_catalog": 715424,
        "fraction_catalog": 0.22357,
        "fraction_continuum": 0.77643,
    }
    assert driver.BIT_IDENTITY_BAND_CENSUS == {
        "n_below_boundary_sources": 3760,
        "n_in_band_sources": 1488,
        "n_above_boundary_sources": 14752,
        "v0_catalog_routed": 4520,
        "v0_continuum_routed": 15480,
    }
    # The bounds (never relaxed).
    assert driver.CENSUS_DISAGREEMENT_BOUND == 1e-6
    assert driver.BIN_SPREAD_BOUND == 2.0
    assert driver.FEEDDOWN_SPREAD_BOUND == 3.0
    # The reserved audit block (must not fire on the first pass).
    assert driver.RESERVED_MASTER_SEED == 20260920
    assert driver.RESERVED_BLOCK_FIRST == 20301040
    assert driver.RESERVED_BLOCK_LAST == 20301055
    # The three region rules and the factor surface.
    assert set(driver.REGION_RULES) == {
        "catalog_region", "transition", "continuum_region"}
    assert driver.FACTOR_SURFACE == (
        "blended_density", "blended_normalization_report",
        "_blended_pieces", "_blended_support")
    assert driver.FACTOR_NAME == "matching_factor"
    assert driver.BUILDER_SAMPLING_FUNCTIONS == (
        "decay_source", "_decay_blended", "_decay_resonance",
        "_decay_continuum")
    assert driver.CARD6_ROUTING_FUNCTIONS == (
        "route_source_decay", "_require_open_catalog",
        "_require_active_family")


def test_driver_copies_byte_identical():
    assert NESTED_DRIVER.is_file() and OUTER_DRIVER.is_file()
    nested = _sha256_file(NESTED_DRIVER)
    outer = _sha256_file(OUTER_DRIVER)
    assert nested == outer, "the dual-tree driver copies drift"


def test_zero_draw_source_contract(driver):
    source = NESTED_DRIVER.read_text(encoding="utf-8")
    # No random module, no other numpy random machinery beyond the
    # pinned probe-stream construction.
    assert "import random" not in source
    assert "np.random.seed" not in source
    assert "randint" not in source
    assert "shuffle(" not in source
    import ast
    tree = ast.parse(source)
    # The only numpy random machinery is default_rng / SeedSequence.
    rng_attrs = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "random"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "np"
    }
    assert rng_attrs == {"default_rng", "SeedSequence"}, rng_attrs
    # The only construction site is the legacy per-event generator of
    # the sealed bit-identity stream configuration (inside
    # ``_probe_event``).
    probe_fn = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_probe_event"
    ]
    assert len(probe_fn) == 1
    fn_span = (probe_fn[0].lineno, probe_fn[0].end_lineno)
    rng_sites = [
        (node.lineno, node.attr)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "random"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "np"
    ]
    assert rng_sites, "the probe-stream construction is missing"
    for lineno, _attr in rng_sites:
        assert fn_span[0] <= lineno <= fn_span[1], (
            f"numpy random machinery outside the pinned probe stream at "
            f"line {lineno}"
        )
    fn_src = ast.get_source_segment(source, probe_fn[0])
    assert fn_src is not None
    assert "np.random.SeedSequence(BIT_IDENTITY_SEED" in fn_src
    assert "np.random.default_rng" in fn_src
    # The recorded zero-draw contract of the pre-registration builder.
    pre_fields = driver.build_pre_registration(
        driver.substrate_digests(), [])
    assert pre_fields["zero_draw"]["draws"] == 0
    assert pre_fields["zero_draw"]["no_fitting"] is True
    assert pre_fields["zero_draw"]["optimizer_invoked"] is False
    block = pre_fields["zero_draw"]["reserved_audit_block"]
    assert block["fired"] is False
    assert block["raw_seeds"] == [20301040, 20301055]
    assert block["master_seed"] == 20260920


# ----------------------------------------------------------------------
# The hand-computed routing contract (the card-6 algebra)
# ----------------------------------------------------------------------


def test_expected_route_hand_case(driver):
    B = driver.KERNEL_BOUNDARY_GEV
    W = driver.RAMP_WIDTH_GEV
    # The region rule and the uniform budget (0 / 1 / 0).
    assert driver._region_of(B - 1e-9) == "catalog_region"
    assert driver._region_of(B) == "transition"
    assert driver._region_of(B + 0.25) == "transition"
    assert driver._region_of(B + W - 1e-9) == "transition"
    assert driver._region_of(B + W) == "continuum_region"
    assert driver._region_of(5.0) == "continuum_region"
    assert driver.expected_uniforms_of(B - 1e-9) == 0
    assert driver.expected_uniforms_of(B) == 1
    assert driver.expected_uniforms_of(B + 0.25) == 1
    assert driver.expected_uniforms_of(B + W) == 0
    # The hand-derived routes: catalog below, u < w(M) in the band,
    # continuum from B + W upward.  At the exact boundary M = B the
    # weight is 0, so the continuum is never routed (u < 0 impossible).
    assert driver._expected_route(0.0, B - 1e-9) == "catalog"
    assert driver._expected_route(0.0, B) == "catalog"
    assert driver._expected_route(0.999, B) == "catalog"
    assert driver._expected_route(0.0, B + 1e-9) == "continuum"
    assert driver._expected_route(0.4999999999, B + 0.25) == "continuum"
    assert driver._expected_route(0.5, B + 0.25) == "catalog"
    assert driver._expected_route(0.6, B + 0.25) == "catalog"
    assert driver._expected_route(0.9999999999, B + W - 1e-9) == "catalog"
    assert driver._expected_route(None, B + W) == "continuum"
    assert driver._expected_route(None, 5.0) == "continuum"
    # The hand-derived w(M) ramp.
    assert driver._expected_weight(B) == 0.0
    assert driver._expected_weight(B + 0.25) == 0.5
    assert driver._expected_weight(B + W) == 1.0
    assert driver._expected_weight(B + W + 0.1) == 1.0


def test_direct_route_contract(driver):
    """The hand-fed stub-stream contract against the pinned card-6
    surface passes on every pinned (mass, uniform) pair."""
    payload = driver._direct_route_contract()
    assert payload["clean"], payload["problems"]
    rows = payload["rows"]
    assert len(rows) == len(payload["hand_masses"]) * len(
        payload["hand_uniforms"])
    assert all(row["pass"] for row in rows)
    # The exact boundary cases (the card-6 algebra).
    by_mass_u = {(round(row["mass"], 12), row["u"]): row for row in rows}
    B = driver.KERNEL_BOUNDARY_GEV
    W = driver.RAMP_WIDTH_GEV
    at_b = by_mass_u[(round(B, 12), 0.5)]
    assert at_b["route"] == "catalog" and at_b["uniforms_consumed"] == 1
    at_bw = by_mass_u[(round(B + W, 12), 0.5)]
    assert at_bw["route"] == "continuum" and at_bw["uniforms_consumed"] == 0
    just_below = by_mass_u[(round(B - 1e-9, 12), 0.5)]
    assert just_below["route"] == "catalog"
    assert just_below["uniforms_consumed"] == 0
    just_above = by_mass_u[(round(B + W + 1e-9, 12), 0.5)]
    assert just_above["route"] == "continuum"
    assert just_above["uniforms_consumed"] == 0
    # The band outcome u < w(M): with u = 0.0 every open-band mass
    # routes continuum; with u = 0.9999999999 only masses with
    # w(M) > 0.9999999999 route continuum.
    open_band = by_mass_u[(round(B + 1e-9, 12), 0.0)]
    assert open_band["route"] == "continuum"
    late_band = by_mass_u[(round(B + W - 1e-9, 12), 0.9999999999)]
    assert late_band["route"] == "catalog"


# ----------------------------------------------------------------------
# The census pooling algebra (hand-computed fixture)
# ----------------------------------------------------------------------


def test_pooled_census_and_disagreement(driver):
    records_v0 = [
        {"routing": {"n_catalog": 10, "n_sources": 100}},
        {"routing": {"n_catalog": 30, "n_sources": 100}},
    ]
    pooled = driver.pooled_census_from_payloads(records_v0)
    assert pooled["n_catalog"] == 40
    assert pooled["n_sources"] == 200
    assert pooled["n_continuum"] == 160
    assert pooled["fraction_catalog"] == 0.2
    assert pooled["fraction_continuum"] == 0.8
    assert pooled["replicas"] == 2
    # Exact agreement with a matching sealed census.
    sealed = {
        "fraction_catalog": 0.2, "fraction_continuum": 0.8,
        "n_catalog": 40, "n_continuum": 160, "n_sources": 200,
    }
    fields = driver.census_disagreement(pooled, sealed)
    assert max(fields.values()) == 0.0
    # A deliberately wrong sealed census is flagged beyond 1e-6.
    wrong = dict(sealed, fraction_catalog=0.3)
    fields = driver.census_disagreement(pooled, wrong)
    assert max(fields.values()) == pytest.approx(0.1)
    assert max(fields.values()) > driver.CENSUS_DISAGREEMENT_BOUND
    # A sealed census rounded at the 1e-6 scale stays inside the bound.
    rounded = dict(sealed, fraction_catalog=0.2 + 5e-7)
    fields = driver.census_disagreement(pooled, rounded)
    assert max(fields.values()) <= driver.CENSUS_DISAGREEMENT_BOUND


# ----------------------------------------------------------------------
# The white-box probe machinery (a small pinned slice)
# ----------------------------------------------------------------------


def test_probe_event_white_box(driver):
    blend = driver.audit_blend()
    catalog = driver.audit_catalog()
    config = driver.audit_config()
    driver._install_route_tracer()
    try:
        events = [
            driver._probe_event(event_id, blend, catalog, config)
            for event_id in range(12)
        ]
    finally:
        driver._restore_route_tracer()
    for event in events:
        # The per-region draw budgets (0 / 1 / 0) and the exact routing
        # stamps (the card's draw-budget contract).
        for row in event["source_rows"]:
            assert row["routing_uniforms"] == row[
                "routing_uniform_expected"], row
            assert row["path_draws"] == 1, row
            assert row["route"] == row["route_expected"], row
            assert row["blend_weight_stamped"] == row[
                "blend_weight_expected"], row
            assert not row.get("route_stamp_mismatch")
            assert not row.get("weight_stamp_mismatch")
            assert not row.get("uniform_budget_mismatch")
            assert not row.get("path_budget_mismatch")
            assert not row.get("route_derivation_mismatch")
        # The retry attempts obey the same per-invocation budget.
        assert event["retry_route_calls_ok"]
        # The source-layer bit-identity leg (exact float equality).
        for row in event["layer_rows"]:
            assert row["legacy_vs_r23"] and row["r23_vs_v0"], row
        # The below-boundary catalog-route finals match the R23 builder
        # convention wherever the comparison is not perturbation-skipped.
        for row in event["finals_rows"]:
            if row["v0_vs_r23"] is True:
                assert row["below_blend_boundary"]
            if row["r23_vs_legacy"] is True:
                assert row["r23_kernel"] == "legacy"
    # The pinned slice exercises at least two regions.
    regions = {
        row["region"] for event in events for row in event["source_rows"]}
    assert len(regions) >= 2, regions


# ----------------------------------------------------------------------
# The white-box isolation (spy fixture + static assertion)
# ----------------------------------------------------------------------


def test_isolation_spy_and_static(driver):
    spy = driver._spy_fixture()
    assert spy["spy_consultations_total"] == 0, spy["spy_consultations"]
    assert spy["blend_pieces_cache_before"] == 0
    assert spy["blend_pieces_cache_after"] == 0
    assert spy["fixture_adequate"]
    assert spy["budgets_ok"]
    assert spy["stamps_exact"]
    assert spy["route_outcomes_exact"]
    static = driver._ast_static_assertion()
    assert static["clean"], static["problems"]
    assert set(static["checked_functions"]) == {
        "na49_mixing.resonance_pp_builder.decay_source",
        "na49_mixing.resonance_pp_builder._decay_blended",
        "na49_mixing.resonance_pp_builder._decay_resonance",
        "na49_mixing.resonance_pp_builder._decay_continuum",
        "na49_mixing.spectral_density.route_source_decay",
        "na49_mixing.spectral_density._require_open_catalog",
        "na49_mixing.spectral_density._require_active_family",
    }


# ----------------------------------------------------------------------
# Sealed archive pins (read-only)
# ----------------------------------------------------------------------


def test_sealed_r32_readonly_pins(driver):
    skip_unless_sealed(driver.MONO_ROOT / driver.R32_PROD_DIR)
    seals = [
        driver.verify_sealed_dir(rel)
        for rel in (driver.R32_PROD_DIR, driver.R32_CONF_DIR,
                    driver.R23_DIR)
    ]
    assert all(seal["verified"] for seal in seals)
    assert driver.verify_sealed_inputs() == dict(driver.SEALED_INPUT_DIGESTS)
    import math
    prod_report = driver._read_json(
        driver.MONO_ROOT / driver.R32_PROD_DIR / "spread_report.json")
    sealed = prod_report["routing_census"]["0"]
    assert sealed["n_catalog"] == 357274
    assert sealed["n_sources"] == 1600000
    assert sealed["fraction_catalog"] == 0.22329625
    gate = driver._read_json(
        driver.MONO_ROOT / driver.R32_PROD_DIR / "gate3_r32.json")
    assert float(
        gate["clauses"]["no_boundary_dominance"]["routing_census"]["v0"]
    ) == 0.22329625
    conf = driver._read_json(
        driver.MONO_ROOT / driver.R32_CONF_DIR
        / "confirmation_verification.json")
    assert conf["routing_census"]["0"]["fraction_catalog"] == 0.22357
    bit = driver._read_json(
        driver.MONO_ROOT / driver.R32_PROD_DIR / "bit_identity_v0.json")
    assert bit["seed"] == 20300913
    assert bit["kernel_master_seed"] == 20260913
    assert bit["kernel_replica_index"] == 0
    assert bit["pass"] is True
    finals = bit["finals_census"]
    assert finals["n_below_boundary_sources"] == 3760
    assert finals["n_in_band_sources"] == 1488
    assert finals["n_above_boundary_sources"] == 14752
    assert finals["v0_catalog_routed"] == 4520
    assert finals["v0_continuum_routed"] == 15480
    r23 = driver._read_json(
        driver.MONO_ROOT / driver.R23_DIR / "bit_identity_report.json")
    assert r23["seed"] == 20300913 and r23["n_events"] == 10000
    assert r23["pass"] is True
    assert r23["layers"]["topology"]["n_matched"] == 10000
    assert r23["layers"]["spectral_state"]["n_matched"] == 10000


def test_pinned_surface_byte_stability():
    assert _sha256_file(MONO_ROOT / BUILDER_REL) == BUILDER_SHA256
    assert _sha256_file(MONO_ROOT / CARD6_REL) == CARD6_SHA256
    assert _sha256_file(MONO_ROOT / ENGINE_REL) == ENGINE_SHA256
    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)
    assert _sha256_file(MONO_ROOT / FAMILIES_REL) == FAMILIES_SHA256
    assert _sha256_file(MONO_ROOT / R32_DRIVER_REL) == R32_DRIVER_SHA256
    assert _sha256_file(
        MONO_ROOT / "helpers" / "benchmarking"
        / "staged_r32_spectral_systematic.py") == R32_DRIVER_SHA256


def test_layout_sync():
    probe = subprocess.run(
        [sys.executable, str(SYNC_CHECKER)],
        capture_output=True, text=True, cwd=str(MONO_ROOT))
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert "logic_drift=0" in probe.stdout


# ----------------------------------------------------------------------
# The freeze-manifest re-pin
# ----------------------------------------------------------------------


def test_freeze_manifest_repin():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    inventory = manifest["source_tree_sha256"]
    # The strict path-sorted inventory rule: the recorded keys are
    # sorted by relative path.
    keys = list(inventory.keys())
    assert keys == sorted(keys)
    # The new inventory entries carry the on-disk digests.
    new_entries = (
        "dev/mixing2006/helpers/benchmarking/"
        "audit_r4_builder_blend_path.py",
        "dev/mixing2006/tests/test_audit_r4_builder_blend_path.py",
        "helpers/benchmarking/audit_r4_builder_blend_path.py",
    )
    for rel in new_entries:
        assert rel in keys, f"missing re-pin entry: {rel}"
        path = MONO_ROOT / rel
        assert inventory[rel] == _sha256_file(path)
    # The nested and outer driver entries share the digest.
    assert inventory[
        "dev/mixing2006/helpers/benchmarking/"
        "audit_r4_builder_blend_path.py"
    ] == inventory["helpers/benchmarking/audit_r4_builder_blend_path.py"]
    # The freeze exception names this card with the audit's files.
    exceptions = manifest["freeze_exceptions"]
    entries = [e for e in exceptions if e["card"] == CARD]
    assert len(entries) == 1
    files = entries[0]["files"]
    for rel in (
        "dev/mixing2006/helpers/benchmarking/"
        "audit_r4_builder_blend_path.py",
        "helpers/benchmarking/audit_r4_builder_blend_path.py",
        "dev/mixing2006/tests/test_audit_r4_builder_blend_path.py",
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
            "the sealed audit run dir is absent on this checkout; the "
            "artifact pins activate at phase 2 (recorded reason)"
        )
    pre = driver._read_json(run_dir / "pre_registration.json")
    assert pre["card"] == CARD
    assert pre["written_before_measurement"] is True
    assert pre["zero_draw"]["draws"] == 0
    assert pre["zero_draw"]["reserved_audit_block"]["fired"] is False
    assert pre["probe_set"]["events"] == 1000
    assert pre["probe_set"]["seed"] == 20300913
    assert pre["census_consistency"]["leg_a"][
        "disagreement_bound"] == 1e-6
    provenance = driver._read_json(run_dir / "provenance.json")
    assert provenance["zero_draw"]["draws"] == 0
    assert provenance["driver_sha256"]["nested"] == provenance[
        "driver_sha256"]["outer"]
    trace = driver._read_json(run_dir / "draw_budget_trace.json")
    assert trace["draw_budget_exact"] is True
    assert trace["bit_identity_leg"]["reverified"] is True
    for region, block in trace["region_budgets"].items():
        assert block["budget_exact"] is True, (region, block)
    isolation = driver._read_json(run_dir / "isolation_assertion.json")
    assert isolation["factor_isolated"] is True
    assert isolation["spy_fixture"]["spy_consultations_total"] == 0
    assert isolation["static_assertion"]["clean"] is True
    assert isolation["direct_route_contract"]["clean"] is True
    census = driver._read_json(run_dir / "census_consistency.json")
    assert census["census_within_bound"] is True
    assert census["leg_a"]["within_bound"] is True
    assert census["leg_a"]["max_disagreement"] <= 1e-6
    assert census["leg_b"]["exact"] is True
    verdict = driver._read_json(run_dir / "verdict.json")
    assert verdict["verdict"] == "PASS"
    assert verdict["findings"] == []
    assert verdict["zero_draw"]["reserved_audit_block_fired"] is False
    seal = driver._read_json(run_dir / "seal_manifest.json")
    assert seal["file_count"] == len(seal["files"])
    sums = (run_dir / "SHA256SUMS.txt").read_text(encoding="utf-8")
    for line in sums.splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        path = driver.MONO_ROOT / name
        assert path.is_file() and _sha256_file(path) == digest, name
    clean = driver._read_json(run_dir / "clean_environment_commands.json")
    assert clean["hashes_match"] is True
    assert clean["probe_failures"] == []
    assert set(clean["probe_returncodes"].values()) == {0}