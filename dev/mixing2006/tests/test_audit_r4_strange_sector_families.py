"""Hermetic tests for the R4 strange-sector continuum families audit (Round 4, card 2).

Task ``TASK-20260920-r4-audit-strange-sector-continuum-families`` (the
Gate 3 decision card follow-up obligation 2; plan section 9.1, lines
598-605).  The tests exercise the card's pre-registered measurement
contract against the committed audit driver, without any Monte Carlo:

- ``test_module_contract_pins``: the pre-registered probe-set identity
  (the 18 K+ persisting bins and their bin indices, the K+ sigma
  range), the family-surface constants (the baryonic_strange record and
  the declared inactive strange mesonic families), the K+-bearing class
  pairs, the sealed count identities, the bounds (2.0 / 3.0), the
  frozen substrate digests, and the reserved audit block.
- ``test_driver_copies_byte_identical``: the nested and outer driver
  copies are byte-identical (the dual-tree register).
- ``test_zero_draw_source_contract``: the driver source carries no
  random-number machinery and imports no numpy (standard library only),
  and the pre-registration builder records ``draws: 0``,
  ``no_fitting: true``, ``optimizer_invoked: false``, and the unfired
  reserved block.
- ``test_vertex_conservation_predicate``: the conservation predicate on
  synthetic catalog records, including the catalog-miss and the
  non-conserving paths.
- ``test_class_pair_enumeration_pinned_catalog_fixture``: the
  completeness enumeration against a pinned synthetic catalog fixture
  (a conserving-but-undeclared class is detected; a non-conserving
  declared class is detected; the committed (1, 0, -1) set is complete
  and conservative on the real catalog, read-only).
- ``test_kplus_decomposition_algebra``: the class-decomposed values and
  the pooled-SE contributions against a hand-computed two-class payload
  (the flat-share identity and a differential-share case whose class
  contributions sum to the bin sigma exactly).
- ``test_replica_identities_fixture``: the sealed count identities on a
  synthetic clean replica and on deliberately broken payloads (each
  identity violation is named).
- ``test_evaluate_verdict_criteria_paths``: the criteria (a), (b), (c)
  firing paths and the all-clean PASS path on synthetic artifacts.
- ``test_sealed_r32_readonly_pins``: the sealed archives verify their
  seals, the persisting register carries exactly 32 entries with the 18
  K+ probe set at the pre-registered bin indices and sigma range, the
  sealed K+ p95 and census anchors match the pre-registration, the
  sealed count identities hold over all 72 payload replicas, and the
  card-1 referral block carries the sealed K+ numbers.
- ``test_config_surface_byte_stability``: the family configuration and
  the catalog stay at their pinned digests.
- ``test_freeze_manifest_repin``: the manifest carries the new
  inventory entries with the on-disk digests and the
  ``freeze_exceptions`` entry naming this card.
- ``test_layout_sync``: the layout checker reports ``logic_drift = 0``
  (the new driver is byte-identical on both sides).
- ``test_audit_run_artifact_pins``: the sealed audit run dir's
  artifacts (activated once the run dir exists; skipped with a recorded
  reason before the audit run): the pre-registration contract, the
  provenance, the class-pair re-derivation, the K+ decomposition rows,
  the config sanity, the verdict, and the seal + clean-validation
  records.

The tests read the sealed R32 archives read-only (their seals verify)
and never write outside ``tmp_path``.  No event generation, no random
numbers, no optimizer, no external SPS tables.
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
DRIVER_REL = Path("helpers") / "benchmarking" / (
    "audit_r4_strange_sector_families.py"
)
NESTED_DRIVER = NESTED_ROOT / DRIVER_REL
OUTER_DRIVER = NESTED_ROOT.parent.parent / DRIVER_REL
SYNC_CHECKER = NESTED_ROOT.parent.parent / "helpers" / "benchmarking" / (
    "check_helpers_layout_sync.py"
)
MANIFEST_PATH = NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json"
FAMILIES_REL = "dev/mixing2006/config/v2_singlet/continuum_resonances.json"
FAMILIES_SHA256 = "242f906cf040eadbe3796dfcfb61d0d51535277399858c901a162e6653d635e1"
CATALOG_SHA256 = "6e4de7372a5bbf99affd6fa3ccc295c4cef09404dab3b17acd8816cda762bdcd"
R32_PROD_DIR = "dev/00runs/r32_spectral_systematic_pp_2026-09-18"
R32_CONF_DIR = "dev/00runs/r32_spectral_systematic_pp_confirmation_2026-09-19"

# The sealed audit run dir (the audit's own Phase 2 output).  The pins
# activate once the dir exists; before the audit run they skip with a
# recorded reason (the R32 pins convention, activated at phase 2).
RUN_DIR_CANDIDATES = tuple(
    (NESTED_ROOT.parent.parent / "dev" / "00runs").glob(
        "r4_audit_strange_sector_continuum_families_*"
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
        "audit_r4_strange_sector_families_under_test", NESTED_DRIVER
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
        "TASK-20260920-r4-audit-strange-sector-continuum-families"
    )
    assert driver.RUN_LABEL == "r4_audit_strange_sector_continuum_families"
    # The byte-stable cards 4/5 surface and the frozen substrate.
    assert driver.FAMILIES_SHA256 == FAMILIES_SHA256
    assert driver.CATALOG_SHA256 == CATALOG_SHA256
    assert driver.CARD6_SHA256 == (
        "561c65ee9a77a1b7f7eb07a7619dafb584779a51ccb23d7fe5651c0605e12563"
    )
    assert driver.BUILDER_SHA256 == (
        "4ab612609e1ee0bc1e037dcbb2815b6484306ad32402b7289e6636f8e829cdb1"
    )
    assert driver.ENGINE_SHA256 == (
        "ee240f44645d2e592acfae462f7d1e9c3b3dd8cd78c96bd40b7b73509f7fcc63"
    )
    assert driver.R32_DRIVER_SHA256 == (
        "99454facce856c93505e6c4b0d4e47b85eeb0ef54f72b1631e3d29fcc511cb90"
    )
    # The probe set: the 18 K+ persisting register entries.
    assert driver.PERSISTING_TOTAL == 32
    assert driver.PERSISTING_KPLUS == 18
    assert driver.KPLUS_BIN_INDICES == (
        2, 4, 5, 7, 19, 20, 21, 27, 28, 31, 45, 68, 71, 74, 112, 136, 148,
        152,
    )
    assert driver.KPLUS_SIGMA_MIN == 2.109195731906308
    assert driver.KPLUS_SIGMA_MAX == 6.420463940482292
    # The strange-sector surface under audit (the cards 4/5 records).
    assert driver.STRANGE_FAMILY_ID == "baryonic_strange"
    assert driver.STRANGE_QN == (1, 0, -1)
    assert driver.STRANGE_DECLARED_CLASSES == (
        "(3122,111)", "(3212,111)", "(3112,211)", "(3222,-211)",
        "(2212,-321)", "(2112,-311)",
    )
    assert driver.STRANGE_M_MIN == 2.680194110727528
    assert driver.STRANGE_T_H == 0.165
    assert driver.STRANGE_B_POWER == 0.0
    assert driver.STRANGE_RAMP_WIDTH == 0.5
    assert driver.STRANGE_NOMINAL_WIDTH == 0.5
    assert driver.STRANGE_M_MAX == 5.0
    assert driver.STRANGE_STATE_TYPE == "continuum_resonance"
    assert driver.INACTIVE_STRANGE_MESONIC == {
        "mesonic_strange": {"qn": (0, 0, -1), "classes": 7},
        "mesonic_antistrange": {"qn": (0, 0, 1), "classes": 7},
    }
    assert driver.FAMILY_CENSUS_ACTIVE == {
        "baryonic_np": {"qn": (1, 1, 0), "classes": 8},
        "baryonic_strange": {"qn": (1, 0, -1), "classes": 6},
        "mesonic_neutral": {"qn": (0, 0, 0), "classes": 10},
    }
    assert driver.FAMILY_CENSUS_INACTIVE == {
        "mesonic_charged": {"qn": (0, 1, 0), "classes": 6},
        "mesonic_anticharged": {"qn": (0, -1, 0), "classes": 6},
        "mesonic_strange": {"qn": (0, 0, -1), "classes": 7},
        "mesonic_antistrange": {"qn": (0, 0, 1), "classes": 7},
    }
    # The K+-bearing class pairs of the activated surface and the
    # catalog-audit records of the K+ Lambda class.
    assert driver.KPLUS_BEARING_CLASSES == {
        "baryonic_np": ("(3122,321)", "(3212,321)"),
        "mesonic_neutral": ("(321,-321)"),
    }
    assert driver.KPLUS_LAMBDA_CATALOG_RECORDS == (122212, 212214)
    # The sealed count identities (the decomposition basis).
    assert driver.IDENTITY_KPLUS_HYPERON == (
        "K_plus.n == Lambda.n + Sigma_0.n"
    )
    assert driver.IDENTITY_ROOT_VERTEX == (
        "K_plus.n_resonant == 0 and by_parent == {}"
    )
    assert driver.IDENTITY_K0_SIGMA == "K_0.n == Sigma_plus.n"
    # The R32 p+p source topology (the builder's pin).
    assert driver.SOURCE_QN_R32 == (1, 1, 0)
    assert driver.BUILDER_SOURCE_QN_SNIPPET == "self._source_qn = (1, 1, 0)"
    # The pre-registered bounds and the sealed anchors.
    assert driver.BIN_SPREAD_BOUND == 2.0
    assert driver.FEEDDOWN_SPREAD_BOUND == 3.0
    assert driver.PERSISTING_TOTAL == 32
    assert driver.PERSISTING_KPLUS == 18
    assert driver.KPLUS_SIGMA_MIN == 2.109195731906308
    assert driver.KPLUS_SIGMA_MAX == 6.420463940482292
    assert driver.PRODUCTION_P95 == 2.9604990756942544
    assert driver.CONFIRMATION_P95 == 4.028426092137674
    assert driver.KPLUS_PRODUCTION_P95 == 9.439290719944319
    assert driver.PRODUCTION_CENSUS_CATALOG == {
        "0": 0.22329625, "1": 0.193558125, "2": 0.253725,
    }
    assert driver.CONFIRMATION_CENSUS_CATALOG == {
        "0": 0.22357, "1": 0.1936659375, "2": 0.253850625,
    }
    # The card-1 referral (the density-shape referral target).
    assert driver.CARD1_CARD == (
        "TASK-20260920-r4-audit-transition-routing-formulation"
    )
    assert driver.CARD1_REWORK_CARD == (
        "TASK-20260920-r4-rework-transition-routing-formulation"
    )
    # The reserved audit block (must not fire on the first pass).
    assert driver.RESERVED_MASTER_SEED == 20260920
    assert driver.RESERVED_BLOCK_FIRST == 20301040
    assert driver.RESERVED_BLOCK_LAST == 20301055


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
        "import random", "default_rng", "SeedSequence", "randint",
        "shuffle(", "gauss(", "weibull", "getstate", "setstate",
        "import numpy", "np.random",
    )
    hits = [token for token in forbidden if token in source]
    assert not hits, f"the audit driver carries random machinery: {hits}"
    # The recorded zero-draw contract of the pre-registration builder.
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
    assert pre_fields["probe_set"]["expected_kplus_entries"] == 18
    assert pre_fields["probe_set"]["expected_bin_indices"] == list(
        driver.KPLUS_BIN_INDICES
    )


# ----------------------------------------------------------------------
# The conservation predicate and the completeness enumeration
# ----------------------------------------------------------------------


def _mini_catalog():
    """A pinned synthetic catalog fixture: proton, pi0, K+, K-, Lambda,
    Xi-, and a Lambda(1520)-like record with two conserving channels."""
    return [
        {"pdg_id": 2212, "name": "proton", "B": 1, "Q": 1, "S": 0,
         "pole_mass": 0.938272, "decay_channels": []},
        {"pdg_id": 111, "name": "pi0", "B": 0, "Q": 0, "S": 0,
         "pole_mass": 0.134977, "decay_channels": []},
        {"pdg_id": 321, "name": "K+", "B": 0, "Q": 1, "S": 1,
         "pole_mass": 0.493677, "decay_channels": []},
        {"pdg_id": -321, "name": "K-", "B": 0, "Q": -1, "S": -1,
         "pole_mass": 0.493677, "decay_channels": []},
        {"pdg_id": 3122, "name": "Lambda", "B": 1, "Q": 0, "S": -1,
         "pole_mass": 1.115683, "decay_channels": []},
        {"pdg_id": 3312, "name": "Xi-", "B": 1, "Q": -1, "S": -2,
         "pole_mass": 1.32171, "decay_channels": []},
        {"pdg_id": 102134, "name": "Lambda(1520)", "B": 1, "Q": 0, "S": -1,
         "pole_mass": 1.5195,
         "decay_channels": [
             {"daughters": [2212, -321], "is_three_body": False},
             {"daughters": [3312, 321], "is_three_body": False},
         ]},
    ]


def test_vertex_conservation_predicate(driver):
    by_pid = {int(r["pdg_id"]): r for r in _mini_catalog()}
    # A conserving (1, 0, -1) pair: p + K-.
    row = driver.vertex_conserves("(2212,-321)", by_pid, (1, 0, -1))
    assert row["conserves"] is True
    assert row["daughter_qn_sums"] == [1, 0, -1]
    assert row["daughter_names"] == ["proton", "K-"]
    # A non-conserving pair: p + pi0 sums to (1, 1, 0).
    row = driver.vertex_conserves("(2212,111)", by_pid, (1, 0, -1))
    assert row["conserves"] is False
    assert row["daughter_qn_sums"] == [1, 1, 0]
    # A K+ daughter needs an S = -2 partner: Xi- + K+ conserves.
    row = driver.vertex_conserves("(3312,321)", by_pid, (1, 0, -1))
    assert row["conserves"] is True
    assert row["daughter_names"] == ["Xi-", "K+"]
    # A daughter missing from the catalog is flagged, never guessed.
    row = driver.vertex_conserves("(9999,111)", by_pid, (1, 0, -1))
    assert row["conserves"] is False
    assert "not in the committed catalog" in row["reason"]
    # A malformed class key raises through the predicate, not silently.
    row = driver.vertex_conserves("(2212)", by_pid, (1, 0, -1))
    assert row["conserves"] is False
    assert "malformed" in row["reason"]


def test_class_pair_enumeration_pinned_catalog_fixture(driver):
    fixture = _mini_catalog()
    found = driver.enumerate_catalog_class_pairs(fixture, (1, 0, -1))
    # The fixture records exactly two conserving class keys: the
    # declared (2212, -321) and the conserving-but-undeclared
    # Xi- K+ pair (the S = -2 partner case).
    assert found["conserving_class_keys"] == ["(2212,-321)", "(3312,321)"]
    assert found["classes"]["(3312,321)"]["carrying_records"] == [102134]
    assert found["classes"]["(2212,-321)"]["carrying_records"] == [102134]
    # A class set missing the Xi- K+ pair is flagged incomplete.
    families_config = {
        "families": [{
            "family_id": "baryonic_strange", "qn": [1, 0, -1],
            "allowed_decay_classes": ["(2212,-321)"],
        }],
    }
    audit = driver.class_pair_audit_rows(families_config, fixture)
    assert audit["missing_class_keys"] == ["(3312,321)"]
    assert audit["complete"] is False
    assert audit["conservative"] is True
    # A declared non-conserving pair is flagged.
    families_config["families"][0]["allowed_decay_classes"] = [
        "(2212,-321)", "(2212,111)",
    ]
    audit = driver.class_pair_audit_rows(families_config, fixture)
    assert audit["nonconserving_declared"] == ["(2212,111)"]
    assert audit["conservative"] is False
    # The fixture records exactly two conserving channels, so the
    # missing set beyond the declared (2212, -321) is the single
    # Xi- K+ pair (the S = -2 partner case).
    assert audit["missing_class_keys"] == ["(3312,321)"]
    assert audit["complete"] is False
    # The (H_b, H_c) reason: two baryon daughters carry B = 2, so no
    # conserving pair exists for the B = 1 family.
    found = driver.enumerate_catalog_class_pairs(fixture, (2, 0, -1))
    assert found["conserving_class_keys"] == []


def test_committed_strange_family_complete_and_conservative(driver):
    """The committed-config check (read-only): the declared (1, 0, -1)
    set is exactly the conserving catalog class set."""
    families_config = driver._read_json(driver.MONO_ROOT / driver.FAMILIES_REL)
    catalog = driver._read_json(driver.MONO_ROOT / driver.CATALOG_REL)
    audit = driver.class_pair_audit_rows(families_config, catalog["records"])
    assert audit["declared_classes"] == list(driver.STRANGE_DECLARED_CLASSES)
    assert audit["enumeration"]["conserving_class_keys"] == sorted(
        driver.STRANGE_DECLARED_CLASSES
    )
    assert audit["missing_class_keys"] == []
    assert audit["nonconserving_declared"] == []
    assert audit["declared_not_enumerated"] == []
    assert audit["complete"] is True and audit["conservative"] is True
    # Every class is kinematically open across the family support (the
    # max daughter pole-mass sum 1.437176 GeV < m_min).
    assert max(k["daughter_pole_mass_sum"] for k in audit["kinematics"]) == (
        pytest.approx(1.437176)
    )
    assert all(k["open_across_family_support"] for k in audit["kinematics"])
    assert all(
        k["at_least_one_daughter_lower_in_mass"]
        for k in audit["kinematics"]
    )
    # The informational re-derivation on the declared inactive strange
    # mesonic families passes as well (outside the verdict contract).
    for row in (
        driver.informational_family_audit(
            families_config, catalog["records"], "mesonic_strange"),
        driver.informational_family_audit(
            families_config, catalog["records"], "mesonic_antistrange"),
    ):
        assert row["complete"] is True and row["conservative"] is True
        assert row["active"] is False


# ----------------------------------------------------------------------
# The K+ decomposition algebra (hand-computed two-class fixture)
# ----------------------------------------------------------------------


def _fake_persisting(bin_index, values, unit_se, parent_index=1):
    return {
        "species": "k_plus", "bin_index": bin_index,
        "parent_index": parent_index, "parent_sigma": 2.5,
        "confirmation_values": list(values),
        "confirmation_unit_se": unit_se,
        "confirmation_sigma": (max(values) - min(values)) / unit_se,
    }


def test_kplus_decomposition_algebra(driver):
    # Flat shares: the class-decomposed values are exactly half the
    # payload, each class carries exactly half the pooled-SE sigma, the
    # contributions sum to the bin sigma, and the dominant class is the
    # tie-broken (3122, 321).
    shares = {"0": {"share_Lambda": 0.5, "share_Sigma0": 0.5},
              "1": {"share_Lambda": 0.5, "share_Sigma0": 0.5},
              "2": {"share_Lambda": 0.5, "share_Sigma0": 0.5}}
    entry = _fake_persisting(2, [0.003, 0.004, 0.002], 0.0005)
    rows = driver.kplus_bin_rows([entry], shares, {}, {})
    row = rows[0]
    assert row["confirmation_sigma"] == pytest.approx(4.0)
    assert row["class_values"]["(3122,321)_K_Lambda"] == pytest.approx(
        [0.0015, 0.002, 0.001]
    )
    assert row["contributions_pooled_se_units"][
        "(3122,321)_K_Lambda"] == pytest.approx(2.0)
    assert row["contributions_pooled_se_units"][
        "(3212,321)_K_Sigma0"] == pytest.approx(2.0)
    assert row["contribution_sum"] == pytest.approx(4.0)
    assert row["dominant_class_pair"] == "(3122,321)"
    assert row["class_attributable"] is True
    # Differential shares: s_Lambda = (0.6, 0.5, 0.4) and
    # s_Sigma0 = (0.4, 0.5, 0.6) on the values (3.0, 2.0, 1.0) with
    # unit SE 0.5 (bin sigma 4.0): the class spreads are 1.4 and 0.6,
    # so the pooled-SE contributions are 2.8 and 1.2, summing to the
    # bin sigma exactly, and the dominant class is (3122, 321).
    shares = {"0": {"share_Lambda": 0.6, "share_Sigma0": 0.4},
              "1": {"share_Lambda": 0.5, "share_Sigma0": 0.5},
              "2": {"share_Lambda": 0.4, "share_Sigma0": 0.6}}
    entry = _fake_persisting(7, [3.0, 2.0, 1.0], 0.5)
    rows = driver.kplus_bin_rows([entry], shares, {}, {})
    row = rows[0]
    assert row["class_values"]["(3122,321)_K_Lambda"] == pytest.approx(
        [1.8, 1.0, 0.4]
    )
    assert row["class_values"]["(3212,321)_K_Sigma0"] == pytest.approx(
        [1.2, 1.0, 0.6]
    )
    assert row["contributions_pooled_se_units"][
        "(3122,321)_K_Lambda"] == pytest.approx(2.8)
    assert row["contributions_pooled_se_units"][
        "(3212,321)_K_Sigma0"] == pytest.approx(1.2)
    assert row["contribution_sum"] == pytest.approx(4.0)
    assert row["dominant_class_pair"] == "(3122,321)"
    # The mirrored case: the Sigma0 share outweighs Lambda on the value
    # pattern and the dominant class flips.
    shares = {"0": {"share_Lambda": 0.4, "share_Sigma0": 0.6},
              "1": {"share_Lambda": 0.5, "share_Sigma0": 0.5},
              "2": {"share_Lambda": 0.6, "share_Sigma0": 0.4}}
    rows = driver.kplus_bin_rows([entry], shares, {}, {})
    row = rows[0]
    assert row["contributions_pooled_se_units"][
        "(3122,321)_K_Lambda"] == pytest.approx(1.2)
    assert row["contributions_pooled_se_units"][
        "(3212,321)_K_Sigma0"] == pytest.approx(2.8)
    assert row["dominant_class_pair"] == "(3212,321)"


def _fake_payload(k_plus, lam, s0, k0=None, sigma_plus=None, k_minus=0,
                  sigma_minus=0, k0s=0, k0l=0, ak0=0, n_resonant=0,
                  by_parent=None, conserved=True):
    species = {
        "K_plus": {"n": k_plus, "n_resonant": n_resonant,
                   "by_parent": {} if by_parent is None else by_parent},
        "Lambda": {"n": lam},
        "Sigma_0": {"n": s0},
        "K_0": {"n": k_plus - lam if k0 is None else k0},
        "Sigma_plus": {"n": k_plus - lam if sigma_plus is None
                       else sigma_plus},
        "K_minus": {"n": k_minus},
        "Sigma_minus": {"n": sigma_minus},
        "K0S": {"n": k0s}, "K0L": {"n": k0l}, "aK_0": {"n": ak0},
    }
    return {"conservation": {"event_b_q_conserved": conserved},
            "feed_down": {"per_species": species}}


def test_replica_identities_fixture(driver):
    # A clean replica: all four identities hold.
    clean = _fake_payload(100, 50, 50, k0=30, sigma_plus=30)
    assert driver.replica_identities(clean, "fixture") == []
    # Identity (i): K+ != Lambda + Sigma0.
    broken = _fake_payload(101, 50, 50)
    violations = driver.replica_identities(broken, "fixture")
    assert any("K_plus.n == Lambda.n + Sigma_0.n" in v for v in violations)
    # Identity (ii): a resonant K+ final violates the root-vertex pin.
    broken = _fake_payload(100, 50, 50, n_resonant=1, by_parent={"323": 1})
    violations = driver.replica_identities(broken, "fixture")
    assert any(IDENTITY in v for v in violations
               for IDENTITY in ("K_plus.n_resonant == 0",))
    # Identity (iii): a K- or Sigma- final violates the zero-strange pin.
    broken = _fake_payload(100, 50, 50, k_minus=2)
    assert any("K_minus.n == 2" in v
               for v in driver.replica_identities(broken, "fixture"))
    broken = _fake_payload(100, 50, 50, sigma_minus=3)
    assert any("Sigma_minus.n == 3" in v
               for v in driver.replica_identities(broken, "fixture"))
    # Identity (iv): K0 != Sigma+.
    broken = _fake_payload(100, 50, 50, k0=31)
    assert any("K_0.n == Sigma_plus.n" in v
               for v in driver.replica_identities(broken, "fixture"))
    # A false conservation flag is flagged.
    broken = _fake_payload(100, 50, 50, conserved=False)
    assert any("conservation flag" in v
               for v in driver.replica_identities(broken, "fixture"))


def test_pooled_class_counts(driver):
    records = [
        _fake_payload(100, 50, 50, k0=30, sigma_plus=30),
        _fake_payload(120, 60, 60, k0=36, sigma_plus=36),
    ]
    pooled = driver.pooled_class_counts(records)
    assert pooled["replicas"] == 2
    assert pooled["counts"]["K_plus"] == 220
    assert pooled["counts"]["Lambda"] == 110
    assert pooled["counts"]["Sigma_0"] == 110
    assert pooled["share_Lambda"] == pytest.approx(0.5)
    assert pooled["share_Sigma0"] == pytest.approx(0.5)
    assert pooled["identity_kplus_hyperon"] is True
    assert pooled["identity_k0_sigma"] is True


# ----------------------------------------------------------------------
# The verdict criteria (the firing paths)
# ----------------------------------------------------------------------


def _synthetic_artifacts(driver, missing=(), nonconserving=(),
                         unattributable=(), byte_stable=True):
    class_pairs = {
        "audit": {
            "family_id": "baryonic_strange", "qn": [1, 0, -1],
            "declared_classes": list(driver.STRANGE_DECLARED_CLASSES),
            "conservation_rows": [],
            "enumeration": {
                "conserving_class_keys": sorted(
                    driver.STRANGE_DECLARED_CLASSES),
            },
        },
        "contract": {
            "complete": not list(nonconserving) and not list(missing),
            "conservative": not list(nonconserving),
            "missing_class_keys": list(missing),
            "nonconserving_declared": list(nonconserving),
        },
        "informational_inactive_strange_mesonic": [],
    }
    kplus = {
        "probe_set": {"entries": 18, "bin_indices": [], "source": "x"},
        "identity_sweep": {"clean": True, "violations": []},
        "rows": [],
        "summary": {
            "persisting_kplus_bins": 18,
            "class_attributable": 18 - len(list(unattributable)),
            "unattributable_bins": list(unattributable),
            "all_class_attributable": not list(unattributable),
            "dominant_class_pair_counts": {"(3122,321)": 18},
            "bound_note": "x",
        },
    }
    config_sanity = {
        "surface_digests": {
            "families_config_sha256": driver.FAMILIES_SHA256
            if byte_stable else "drifted",
            "catalog_sha256": driver.CATALOG_SHA256
            if byte_stable else "drifted",
            "byte_stable": byte_stable,
        },
        "strange_record_problems": [],
        "inactive_problems": [],
        "r30_cross_reference": {},
        "r31_cross_reference": {"interior_pairs_failed": 0},
    }
    return class_pairs, kplus, config_sanity


def test_evaluate_verdict_criteria_paths(driver):
    # The all-clean PASS path.
    cp, kp, cs = _synthetic_artifacts(driver)
    evaluation = driver.evaluate_verdict_criteria(cp, kp, cs)
    assert evaluation["verdict"] == "PASS"
    assert evaluation["fired"] == []
    # Criterion (a): a missing or non-conserving class pair.
    cp, kp, cs = _synthetic_artifacts(
        driver, missing=["(3312,321)"], nonconserving=["(2212,111)"],
    )
    evaluation = driver.evaluate_verdict_criteria(cp, kp, cs)
    assert evaluation["verdict"] == "RECOMMENDATION"
    assert [c["criterion"] for c in evaluation["fired"]] == ["a"]
    # Criterion (b): a persisting K+ bin without a class-pair path.
    cp, kp, cs = _synthetic_artifacts(driver, unattributable=[152])
    evaluation = driver.evaluate_verdict_criteria(cp, kp, cs)
    assert evaluation["verdict"] == "RECOMMENDATION"
    assert [c["criterion"] for c in evaluation["fired"]] == ["b"]
    # Criterion (c): the config bytes drifted at verdict time.
    cp, kp, cs = _synthetic_artifacts(driver, byte_stable=False)
    evaluation = driver.evaluate_verdict_criteria(cp, kp, cs)
    assert evaluation["verdict"] == "RECOMMENDATION"
    assert [c["criterion"] for c in evaluation["fired"]] == ["c"]


# ----------------------------------------------------------------------
# Sealed-archive read-only pins
# ----------------------------------------------------------------------


def test_sealed_r32_readonly_pins(driver):
    # The probe archives and the cross-reference archives verify their
    # seals (read-only).
    skip_unless_sealed(driver.MONO_ROOT / driver.R32_PROD_DIR)
    prod = driver.verify_sealed_dir(driver.R32_PROD_DIR)
    conf = driver.verify_sealed_dir(driver.R32_CONF_DIR)
    assert prod["verified"] is True and prod["entries"] == 29
    assert conf["verified"] is True and conf["entries"] == 55
    assert driver.verify_sealed_dir(driver.R30_DIR)["entries"] == 3
    assert driver.verify_sealed_dir(driver.R31_DIR)["entries"] == 79
    assert driver.verify_sealed_dir(driver.CARD1_DIR)["entries"] == 6
    # The sealed anchor digests re-hash.
    verified = driver.verify_sealed_inputs()
    assert verified == driver.SEALED_INPUT_DIGESTS
    # The persisting register: exactly 32 entries, 18 K+ with the
    # pre-registered bin indices and sigma range.
    cv = driver._read_json(
        driver.MONO_ROOT / driver.R32_CONF_DIR
        / driver.CONFIRMATION_VERIFICATION_NAME
    )
    persisting = [
        e for e in cv["residual_reclassification"]
        if e.get("classification") == "persisting"
    ]
    assert len(persisting) == driver.PERSISTING_TOTAL
    kplus = [e for e in persisting if e["species"] == "k_plus"]
    assert len(kplus) == driver.PERSISTING_KPLUS
    assert sorted(int(e["bin_index"]) for e in kplus) == list(
        driver.KPLUS_BIN_INDICES
    )
    sigmas = [float(e["confirmation_sigma"]) for e in kplus]
    assert min(sigmas) == pytest.approx(driver.KPLUS_SIGMA_MIN,
                                        abs=1e-9)
    assert max(sigmas) == pytest.approx(driver.KPLUS_SIGMA_MAX,
                                        abs=1e-9)
    # The sealed anchors: the p95 values, the K+ p95, and the routing
    # census.
    report = driver._read_json(
        driver.MONO_ROOT / driver.R32_PROD_DIR / driver.SPREAD_REPORT_NAME
    )
    assert float(report["summary"]["bin_spread_p95"]) == (
        driver.PRODUCTION_P95
    )
    assert float(cv["summary"]["bin_spread_p95"]) == (
        driver.CONFIRMATION_P95
    )
    assert float(
        report["summary"]["bin_spread_p95_per_species"]["k_plus"]
    ) == driver.KPLUS_PRODUCTION_P95
    for v in ("0", "1", "2"):
        assert float(
            report["routing_census"][v]["fraction_catalog"]
        ) == driver.PRODUCTION_CENSUS_CATALOG[v]
        assert float(
            cv["routing_census"][v]["fraction_catalog"]
        ) == driver.CONFIRMATION_CENSUS_CATALOG[v]
    # The sealed count identities hold over every payload replica
    # (production 3 x 8, confirmation 3 x 16; the decomposition basis).
    prod = driver.load_payload_identities(
        driver.R32_PROD_DIR, "variant_{v}_replica_{r}.json", 8,
    )
    conf = driver.load_payload_identities(
        driver.R32_CONF_DIR,
        "confirmation_variant_{v}_replica_{r}.json", 16,
    )
    assert prod["identity_violations"] == []
    assert conf["identity_violations"] == []
    for v in ("0", "1", "2"):
        block = conf["per_variant"][v]
        assert block["identity_kplus_hyperon"] is True
        assert block["identity_k0_sigma"] is True
        assert 0.49 < block["share_Lambda"] < 0.51
    # The card-1 referral block carries the sealed K+ numbers.
    card1_verdict = driver._read_json(
        driver.MONO_ROOT / driver.CARD1_DIR / driver.CARD1_VERDICT_NAME
    )
    assert card1_verdict["verdict"] == "RECOMMENDATION"
    referral = (card1_verdict["attribution_summary"]["k_plus_referral"])
    assert referral["referred_to_card"] == driver.CARD
    assert referral["bins"] == 18
    assert referral["monotone"] == 15
    assert referral["within_census_envelope"] == 18
    assert referral["mass_blind_envelope_violations"] == 16
    assert referral["continuum_dominant"] == 15


def test_config_surface_byte_stability(driver):
    digests = driver.substrate_digests()
    assert digests["families_config_sha256"] == FAMILIES_SHA256
    assert digests["catalog_sha256"] == CATALOG_SHA256
    assert digests["card6_surface_sha256"] == (
        "561c65ee9a77a1b7f7eb07a7619dafb584779a51ccb23d7fe5651c0605e12563"
    )
    assert digests["builder_module_sha256"] == driver.BUILDER_SHA256
    assert digests["r32_driver_nested_sha256"] == (
        digests["r32_driver_outer_sha256"]
    )


# ----------------------------------------------------------------------
# The freeze-manifest re-pin and the dual-tree layout sync
# ----------------------------------------------------------------------


def test_freeze_manifest_repin():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    # The strict path-sorted inventory rule: the recorded keys are
    # sorted by relative path.
    keys = list(manifest["source_tree_sha256"].keys())
    assert keys == sorted(keys)
    # The new inventory entries carry the on-disk digests.
    new_entries = (
        "dev/mixing2006/helpers/benchmarking/"
        "audit_r4_strange_sector_families.py",
        "dev/mixing2006/tests/test_audit_r4_strange_sector_families.py",
        "helpers/benchmarking/audit_r4_strange_sector_families.py",
    )
    for rel in new_entries:
        assert rel in keys, f"missing re-pin entry: {rel}"
        path = NESTED_ROOT.parent.parent / rel
        assert manifest["source_tree_sha256"][rel] == _sha256_file(path)
    # The nested and outer driver entries share the digest.
    assert manifest["source_tree_sha256"][
        "dev/mixing2006/helpers/benchmarking/"
        "audit_r4_strange_sector_families.py"
    ] == manifest["source_tree_sha256"][
        "helpers/benchmarking/audit_r4_strange_sector_families.py"
    ]
    # The freeze exception names this card with the audit's files.
    exceptions = manifest["freeze_exceptions"]
    entries = [e for e in exceptions if e["card"] == (
        "TASK-20260920-r4-audit-strange-sector-continuum-families"
    )]
    assert len(entries) == 1
    files = entries[0]["files"]
    for rel in (
        "dev/mixing2006/helpers/benchmarking/"
        "audit_r4_strange_sector_families.py",
        "helpers/benchmarking/audit_r4_strange_sector_families.py",
        "dev/mixing2006/tests/test_audit_r4_strange_sector_families.py",
        "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json",
    ):
        assert rel in files
    # Every pinned frozen file stays byte-unchanged: the pinned M8 map
    # is intact (the freeze test itself enforces the full inventory).
    assert manifest["frozen_fit_outputs"]["m8_frozen_pp_parameters"][
        "continuum.match_mass"
    ] == 2.680194110727528


def test_layout_sync(driver):
    """The nested and outer driver copies are byte-identical and the
    layout checker reports no logic drift."""
    if NESTED_DRIVER.is_file() and OUTER_DRIVER.is_file():
        assert NESTED_DRIVER.read_bytes() == OUTER_DRIVER.read_bytes()
    if SYNC_CHECKER.is_file():
        probe = subprocess.run(
            [sys.executable, str(SYNC_CHECKER)],
            capture_output=True, text=True,
        )
        assert probe.returncode == 0, (
            f"check_helpers_layout_sync.py failed (exit "
            f"{probe.returncode}): {probe.stdout[-2000:]}"
        )


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
            "the audit run dir (dev/00runs/r4_audit_strange_sector_"
            "continuum_families_*) is absent; the artifact pins activate "
            "at phase 2 (the audit run)"
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
    assert pre["probe_set"]["expected_kplus_entries"] == 18
    assert pre["probe_set"]["expected_bin_indices"] == list(
        driver.KPLUS_BIN_INDICES
    )
    assert pre["zero_draw"]["draws"] == 0
    assert pre["zero_draw"]["no_fitting"] is True
    assert pre["zero_draw"]["optimizer_invoked"] is False
    assert pre["zero_draw"]["reserved_audit_block"]["fired"] is False
    assert pre["surface"]["continuum_families_config"][
        "sha256"] == FAMILIES_SHA256
    assert pre["surface"]["resonance_catalog"]["sha256"] == CATALOG_SHA256
    # -- the provenance --------------------------------------------------
    prov = json.loads((run_dir / "provenance.json").read_text(
        encoding="utf-8"))
    assert prov["zero_draw"]["draws"] == 0
    assert prov["zero_draw"]["no_fitting"] is True
    assert prov["zero_draw"]["optimizer_invoked"] is False
    assert prov["zero_draw"]["reserved_audit_block_fired"] is False
    assert prov["driver_sha256"]["nested"] == prov["driver_sha256"]["outer"]
    assert prov["card_chain"] == [
        "TASK-20260920-r3-gate3-decision",
        "TASK-20260920-round4-index",
        driver.CARD,
    ]
    # -- the class-pair re-derivation ------------------------------------
    class_pairs = json.loads((run_dir / "class_pair_audit.json").read_text(
        encoding="utf-8"))
    assert class_pairs["audit"]["family_id"] == "baryonic_strange"
    assert class_pairs["audit"]["qn"] == [1, 0, -1]
    assert class_pairs["audit"]["declared_classes"] == list(
        driver.STRANGE_DECLARED_CLASSES
    )
    assert class_pairs["contract"]["complete"] is True
    assert class_pairs["contract"]["conservative"] is True
    assert class_pairs["contract"]["missing_class_keys"] == []
    assert class_pairs["contract"]["nonconserving_declared"] == []
    assert len(class_pairs["audit"]["kinematics"]) == 6
    assert all(k["open_across_family_support"]
               for k in class_pairs["audit"]["kinematics"])
    info = class_pairs["informational_inactive_strange_mesonic"]
    assert {row["family_id"] for row in info} == {
        "mesonic_strange", "mesonic_antistrange"}
    assert all(row["complete"] and row["conservative"] for row in info)
    # -- the K+ decomposition --------------------------------------------
    kplus = json.loads((run_dir / "kplus_class_contribution.json")
                       .read_text(encoding="utf-8"))
    assert kplus["probe_set"]["entries"] == 18
    assert kplus["probe_set"]["bin_indices"] == list(
        driver.KPLUS_BIN_INDICES
    )
    assert kplus["identity_sweep"]["clean"] is True
    assert kplus["identity_sweep"]["production_payloads"] == 24
    assert kplus["identity_sweep"]["confirmation_payloads"] == 48
    rows = kplus["rows"]
    assert len(rows) == 18
    assert sum(1 for r in rows if r["class_attributable"]) == 18
    assert kplus["summary"]["unattributable_bins"] == []
    assert kplus["summary"]["dominant_class_pair_counts"] == {
        "(3122,321)": 16, "(3212,321)": 2,
    }
    for row in rows:
        assert row["bound"] == 2.0
        assert row["density_shape_referral"]["referred_to_card"] == (
            driver.CARD1_CARD
        )
    # -- the config sanity ------------------------------------------------
    sanity = json.loads((run_dir / "config_sanity.json").read_text(
        encoding="utf-8"))
    assert sanity["surface_digests"]["byte_stable"] is True
    assert sanity["surface_digests"]["families_config_sha256"] == (
        FAMILIES_SHA256
    )
    assert sanity["surface_digests"]["catalog_sha256"] == CATALOG_SHA256
    assert sanity["strange_record_problems"] == []
    assert sanity["inactive_problems"] == []
    assert all(block["pass"] for block in
               sanity["r30_cross_reference"].values())
    assert sanity["r30_cross_reference"]["double_counting_identity"][
        "strange_qn_covered"] is True
    assert sanity["r31_cross_reference"]["interior_pairs_failed"] == 0
    assert sanity["r31_cross_reference"]["boundary_pair_failed"] == 0
    assert sanity["sampling_scope"]["builder_source_qn_pin_found"] is True
    assert sanity["kinematic_recomputation"]["all_open_across_support"] is True
    assert sanity["kinematic_recomputation"]["max_pole_mass_sum"] == (
        pytest.approx(1.437176)
    )
    # -- the verdict ------------------------------------------------------
    verdict = json.loads((run_dir / "verdict.json").read_text(
        encoding="utf-8"))
    assert verdict["verdict"] == "PASS"
    assert verdict["criteria"]["a"]["fired"] is False
    assert verdict["criteria"]["b"]["fired"] is False
    assert verdict["criteria"]["c"]["fired"] is False
    assert verdict["findings"] == []
    assert verdict["recommendation"] is None
    assert verdict["bounds"]["bin_spread_p95_bound"] == 2.0
    assert verdict["bounds"]["feeddown_spread_max_bound"] == 3.0
    assert verdict["bounds"]["kplus_p95_anchor_production"] == (
        driver.KPLUS_PRODUCTION_P95
    )
    assert verdict["class_pair_summary"]["complete"] is True
    assert verdict["class_pair_summary"]["conservative"] is True
    assert verdict["kplus_decomposition_summary"][
        "all_class_attributable"] is True
    assert verdict["config_sanity_summary"]["byte_stable"] is True
    assert verdict["density_shape_referral"]["referred_to_card"] == (
        driver.CARD1_CARD
    )
    assert verdict["zero_draw"]["draws"] == 0
    assert verdict["surface"]["byte_unchanged_by_this_audit"] is True
    assert verdict["surface"]["families_sha256"] == FAMILIES_SHA256
    # -- the clean-environment record -------------------------------------
    clean = json.loads((run_dir / "clean_environment_commands.json")
                       .read_text(encoding="utf-8"))
    assert clean["hashes_match"] is True
    assert clean["seal_verification"]["mismatches"] == []
    assert clean["probe_returncodes"] == {
        "class-pairs": 0, "k-contribution": 0, "config-sanity": 0,
        "verdict": 0,
    }
    assert clean["commands"] and len(clean["commands"]) == 4