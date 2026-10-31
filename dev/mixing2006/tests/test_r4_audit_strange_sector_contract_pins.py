"""Round 4 card 2: strange-sector continuum families audit contract pins
(test-author register).

Task ``TASK-20260920-r4-audit-strange-sector-continuum-families`` (Gate 3
decision card follow-up obligation 2; plan section 9.1, lines 598-605).
This is the test-author's independent pin register for the zero-draw,
hermetic dual-tree audit driver
``helpers/benchmarking/audit_r4_strange_sector_families.py`` (nested
commit ``95acd0b``, outer ``0bf531c``; byte-identical in both trees at
sha256 ``3bd6c034fb0d198371df0911ac6f6c0b0bf36d426681ecccb0aca96c8a9ed2ce``)
and for the sealed audit run dir
``dev/00runs/r4_audit_strange_sector_continuum_families_2026-09-20/``
(verdict PASS, 0 named findings).  It is an independent cross-check of
the solver's register in
``tests/test_audit_r4_strange_sector_families.py``: the same
pre-registered measurement contract, pinned from an independently
authored test with its own hand-built catalog fixtures, share fixtures,
and synthetic artifacts (no solver fixture values reused).

Part 1 (the probe contract, pure, no sealed run dir needed):

- ``test_vertex_conservation_predicate_ta_fixture``: the
  vertex-conservation predicate (daughter (B, Q, S) sum vs the family
  quantum numbers) on the test-author's own catalog fixture for
  (1, 0, -1): a conserving pair with an S = -2 daughter (the partner a
  K+ daughter would need), a non-conserving S = -2 pair, a
  catalog-miss path, and the malformed class-key path.
- ``test_class_pair_enumeration_completeness_and_conservatism``: the
  class-pair enumeration semantics (completeness: every conserving
  catalog two-body channel with the right daughter sum is declared;
  conservativity: no declared pair non-conserving) on a pinned
  synthetic fixture with an intentionally missing pair and an
  intentionally non-conserving pair (the test-author's own fixture),
  including the carrying-record identifiers, the three-body and
  catalog-miss channel exclusions, and the empty B = 2 universe.
- ``test_kinematic_openness_boundary``: the kinematic-openness
  predicate (daughter pole-mass sum below m_min) on hand fixtures,
  including the exact m_min boundary (the sum equal to m_min is not
  open; the strict ``<`` comparison).
- ``test_kplus_decomposition_algebra_ta_fixture``: the K+
  decomposition algebra (the proportional census attribution
  ``y_v,c = s_c(v) * y_v``) on the test-author's own two-class fixtures:
  flat shares, a differential share case, and the dominant-class flip,
  each with the per-class pooled-SE contributions hand-computed and
  summing to the bin sigma exactly on the aligned cases.
- ``test_sealed_count_identities_ta_fixture``: the sealed per-replica
  count identities (K+ = Lambda + Sigma0; K- = Sigma- = K0S = K0L = aK0
  = 0; K0 = Sigma+) on the test-author's own two-class hand fixtures
  (a clean replica, one broken payload per identity, and the pooled
  two-replica share arithmetic).
- ``test_verdict_criteria_logic_ta_fixture``: the verdict-criteria
  logic (PASS iff the class set is complete and conservative, all K+
  bins class-attributable, and the config bytes intact;
  RECOMMENDATION on (a) a class pair missing/non-conserving, (b) an
  unattributable K+ bin, (c) a config disagreement) on the
  test-author's own criteria fixtures, including the compound
  all-criteria case.
- ``test_zero_draw_no_rng_no_numpy``: the zero-draw invariants on the
  test-author's own technique (independent of the solver's token scan):
  an AST import analysis pinning the driver's full import surface to
  the standard library with no numpy and no random, a source-level
  token scan with the test-author's own list over both driver trees, a
  fresh isolated-interpreter import probe (the driver's import pulls
  no numpy/random into ``sys.modules``), and the full pure surface
  exercised under raising sentinels on the ``numpy`` and ``random``
  module slots.

Part 2 (the sealed artifacts, run-dir pins; the run dir is gitignored,
the pins activate once the dir exists and skip with a recorded reason
on a fresh checkout):

- ``test_driver_dual_tree_byte_identity``: the nested and outer audit
  driver copies are byte-identical at the pinned digest.
- ``test_surface_byte_stability``: the continuum families config, the
  catalog, the card-6 blend routing surface, and the builder stay at
  their pinned on-disk digests (re-hashed independently of the
  driver).
- ``test_sealed_run_dir_seal_pins``: the seal manifest (7 files,
  100,966 bytes, per the on-disk ``seal_manifest.json``), the full
  per-file digest listing self-consistent against disk, and the
  ``SHA256SUMS.txt`` re-hash integrity (6 monorepo-root-relative
  entries), plus the post-seal clean-environment record.
- ``test_sealed_run_dir_verdict_pins``: the ``verdict.json`` content
  pins (verdict PASS, 0 named findings, the pass text with the card-1
  referral, the 18-bin probe summary, the card-1 referral numbers
  15/18 monotone / 15/18 continuum-dominant / 16/18 mass-blind
  envelope violations, the class-pair summary 6/6 conserve / 6/6
  declared, the config-sanity summary, the bounds, and the zero-draw
  record).
- ``test_sealed_run_dir_class_pair_pins``: the
  ``class_pair_audit.json`` pins (6/6 declared classes conserve, 6/6
  conserving catalog class pairs declared, complete + conservative
  true, the carrying catalog record ids 3214 / 102134 with their
  channel counts, the max daughter pole-mass sum 1.437176, the empty
  (H_b, H_c) enumeration, the informational inactive rows, and the
  zero-draw stamps).
- ``test_sealed_run_dir_config_sanity_pins``: the
  ``config_sanity.json`` pins (0 strange-record problems, 0 inactive
  problems, the family census, all committed-parameter fields
  matching, the R30 blocks pass, the R31 window with 0 failures, and
  the 1.432-vs-1.437176 informational note).
- ``test_sealed_run_dir_kplus_pins``: the
  ``kplus_class_contribution.json`` pins (the 18-bin probe set, the
  identity sweep over the 72 sealed replicas, the sealed per-variant
  class shares with the identity counts, the per-bin contributions
  summing to the bin sigma, the dominant-class counts, and the
  card-1 referral block).
- ``test_sealed_run_dir_prereg_provenance_pins``: the
  ``pre_registration.json`` sha256 pin (immutability), the probe-set
  contract, the zero-draw contract with the reserved audit block
  20301040 + k not fired, the bounds, the sealed-verification record,
  and the ``provenance.json`` driver digest pin (nested == outer ==
  the on-disk dual-tree digest).
- ``test_freeze_repin_register``: the legacy_v1 manifest re-pin for
  this register (the inventory entry at the on-disk digest, the strict
  path-sorted inventory, and the single ``freeze_exceptions`` entry
  naming this card carrying the audit's files plus this register's
  file, path-sorted).

Hermetic: standard library only; the committed audit driver and the
sealed archives are read read-only; no writes outside the process, no
event generation, no random numbers, no numpy, no optimizer.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

NESTED_ROOT = Path(__file__).resolve().parent.parent
MONO_ROOT = NESTED_ROOT.parent.parent
DRIVER_REL = Path("helpers") / "benchmarking" / (
    "audit_r4_strange_sector_families.py"
)
NESTED_DRIVER = NESTED_ROOT / DRIVER_REL
OUTER_DRIVER = MONO_ROOT / DRIVER_REL
MANIFEST_PATH = NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json"

# The pinned dual-tree driver digest (nested 95acd0b, outer 0bf531c).
AUDIT_DRIVER_SHA256 = (
    "3bd6c034fb0d198371df0911ac6f6c0b0bf36d426681ecccb0aca96c8a9ed2ce"
)
# The byte-stable cards 4/5 surface and the frozen substrate.
FAMILIES_SHA256 = (
    "242f906cf040eadbe3796dfcfb61d0d51535277399858c901a162e6653d635e1"
)
CATALOG_SHA256 = (
    "6e4de7372a5bbf99affd6fa3ccc295c4cef09404dab3b17acd8816cda762bdcd"
)
CARD6_SHA256 = (
    "561c65ee9a77a1b7f7eb07a7619dafb584779a51ccb23d7fe5651c0605e12563"
)
BUILDER_SHA256 = (
    "4ab612609e1ee0bc1e037dcbb2815b6484306ad32402b7289e6636f8e829cdb1"
)
CARD_ID = "TASK-20260920-r4-audit-strange-sector-continuum-families"
CARD1_ID = "TASK-20260920-r4-audit-transition-routing-formulation"
CARD1_REWORK_ID = (
    "TASK-20260920-r4-rework-transition-routing-formulation"
)
# The declared (1, 0, -1) class set (the cards 4/5 records).
STRANGE_DECLARED_CLASSES = [
    "(3122,111)", "(3212,111)", "(3112,211)",
    "(3222,-211)", "(2212,-321)", "(2112,-311)",
]
# The recomputed kinematic extreme of the (1, 0, -1) classes (the
# (2112, -311) n-Kbar0 pair; the sealed artifact carries the IEEE
# double).
KIN_MAX_POLE_MASS_SUM = 1.4371764100000002
# The K+ bin p95 anchor (the sealed production per-species value).
KPLUS_P95_ANCHOR = 9.439290719944319
KPLUS_SIGMA_RANGE = [2.109195731906308, 6.420463940482292]
KPLUS_BIN_INDICES = [
    2, 4, 5, 7, 19, 20, 21, 27, 28, 31, 45, 68, 71, 74, 112, 136, 148,
    152,
]

# The sealed audit run dir (gitignored; the pins activate once it
# exists, the R32 pins convention).
RUN_DIR_CANDIDATES = tuple(
    (MONO_ROOT / "dev" / "00runs").glob(
        "r4_audit_strange_sector_continuum_families_*"
    )
)

# The sealed run dir's file digests (on-disk, 2026-09-20).
SEAL_FILES = {
    "SHA256SUMS.txt": (
        "54e9813c61b1552ba781a15abd1611948512f42ff07f99875c29c43a0bf2e6d7",
        908,
    ),
    "class_pair_audit.json": (
        "9fc4efe0d5afcc4d5f96dc20fecc699f77c683b22a523ae58372870033509a11",
        8412,
    ),
    "config_sanity.json": (
        "b30b0ee095ac828293fdeb833104a3ba197f8bd32d1f90c20b360e875bd5a478",
        7405,
    ),
    "kplus_class_contribution.json": (
        "26a193309571a3fc07c13da47cb4672a56da2bad0e690a501acd0e59ecfbdbd0",
        47865,
    ),
    "pre_registration.json": (
        "b03d7204d4191446f45f0fe69b363a001057af56c5cbef02da42dfdf9a13768f",
        14028,
    ),
    "provenance.json": (
        "c7064f3cc6df5690d5efd1271f746425e85d370e7d75c5182d70eaf8d394480f",
        3820,
    ),
    "verdict.json": (
        "9884765a869798beedabd6d97a131bf711639f292450b0ae04611a3ec287f0a1",
        18528,
    ),
}
SEAL_FILE_COUNT = 7
SEAL_TOTAL_BYTES = 100966
SEAL_MANIFEST_SHA256 = (
    "290e475a9893c6f52416c1ae49448b1563155a2aeff6a79f4fc288c8d08a3376"
)
CLEAN_ENV_SHA256 = (
    "2e81c40e119f84cd33f44f07778e24c0260aa74b605adb3d4530bce3ef8444d4"
)
PRE_REGISTRATION_SHA256 = SEAL_FILES["pre_registration.json"][0]
# The sealed confirmation per-variant v0 pooled class counts (the
# identity sweep; 16 replicas per variant).
CONF_V0_COUNTS = {
    "K_plus": 624971, "K_minus": 0, "K0S": 0, "K0L": 0, "K_0": 311297,
    "aK_0": 0, "Lambda": 313984, "Sigma_0": 310987, "Sigma_plus": 311297,
    "Sigma_minus": 0,
}
CONF_V0_SHARE_LAMBDA = 0.5023977112538022
MY_TEST_REL = "dev/mixing2006/tests/test_r4_audit_strange_sector_contract_pins.py"


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
    name = "audit_r4_strange_sector_families_test_author"
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
            "(dev/00runs/r4_audit_strange_sector_continuum_families_*) "
            "is absent on this checkout; the sealed-artifact pins "
            "activate once the run dir exists"
        )
    return run_dir


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# The test-author's own fixtures (independent of the solver's)
# ----------------------------------------------------------------------


def _ta_catalog():
    """The test-author's synthetic catalog fixture (B, Q, S, pole mass):
    901 hyp_a (1, 0, -1, 1.0), 902 mes_a (0, 0, 0, the exact m_min
    complement of 1.0), 903 hyp_b (1, 1, -2, 1.4), 904 mes_b (0, -1,
    +1, 0.28), 905 mes_c (0, 0, 0, 0.3), 906 hyp_c (1, 0, -2, 1.32),
    and the parents 9901 / 9902 / 9903 with the conserving channels."""
    return [
        {"pdg_id": 901, "name": "hyp_a", "B": 1, "Q": 0, "S": -1,
         "pole_mass": 1.0, "decay_channels": []},
        {"pdg_id": 902, "name": "mes_a", "B": 0, "Q": 0, "S": 0,
         "pole_mass": 1.6801941107275278, "decay_channels": []},
        {"pdg_id": 903, "name": "hyp_b", "B": 1, "Q": 1, "S": -2,
         "pole_mass": 1.4, "decay_channels": []},
        {"pdg_id": 904, "name": "mes_b", "B": 0, "Q": -1, "S": 1,
         "pole_mass": 0.28, "decay_channels": []},
        {"pdg_id": 905, "name": "mes_c", "B": 0, "Q": 0, "S": 0,
         "pole_mass": 0.3, "decay_channels": []},
        {"pdg_id": 906, "name": "hyp_c", "B": 1, "Q": 0, "S": -2,
         "pole_mass": 1.32, "decay_channels": []},
        {"pdg_id": 9901, "name": "parent_a", "B": 1, "Q": 0, "S": -1,
         "pole_mass": 2.9,
         "decay_channels": [
             {"daughters": [901, 902], "is_three_body": False},
         ]},
        {"pdg_id": 9902, "name": "parent_b", "B": 1, "Q": 0, "S": -1,
         "pole_mass": 2.95,
         "decay_channels": [
             {"daughters": [903, 904], "is_three_body": False},
         ]},
        {"pdg_id": 9903, "name": "parent_c", "B": 1, "Q": 0, "S": -1,
         "pole_mass": 3.0,
         "decay_channels": [
             {"daughters": [901, 905], "is_three_body": False},
         ]},
    ]


def _ta_kinematics_config():
    """The families config driving the kinematic-openness fixture: all
    three conserving classes declared, one daughter pole-mass sum
    exactly at m_min (the boundary), two below it."""
    return {
        "families": [{
            "family_id": "baryonic_strange", "qn": [1, 0, -1],
            "allowed_decay_classes": ["(901,902)", "(903,904)",
                                      "(901,905)"],
        }],
    }


def _ta_enumeration_catalog():
    """The test-author's completeness fixture: three conserving
    channels (7002, 7003) / (7004, 7005) / (7006, 7007) carried by
    parents 7001 / 7010 / 7007, plus a three-body channel and a
    catalog-miss channel on 7010 (both excluded from the
    enumeration)."""
    return [
        {"pdg_id": 7001, "name": "par_1", "B": 1, "Q": 0, "S": -1,
         "pole_mass": 1.5,
         "decay_channels": [
             {"daughters": [7002, 7003], "is_three_body": False},
         ]},
        {"pdg_id": 7002, "name": "hyp_y", "B": 1, "Q": 1, "S": -1,
         "pole_mass": 1.2, "decay_channels": []},
        {"pdg_id": 7003, "name": "mes_y", "B": 0, "Q": -1, "S": 0,
         "pole_mass": 0.14, "decay_channels": []},
        {"pdg_id": 7004, "name": "hyp_z", "B": 1, "Q": 0, "S": -1,
         "pole_mass": 1.3, "decay_channels": []},
        {"pdg_id": 7005, "name": "mes_z", "B": 0, "Q": 0, "S": 0,
         "pole_mass": 0.1, "decay_channels": []},
        {"pdg_id": 7006, "name": "mes_w", "B": 0, "Q": 0, "S": 0,
         "pole_mass": 0.2, "decay_channels": []},
        {"pdg_id": 7007, "name": "par_w", "B": 1, "Q": 0, "S": -1,
         "pole_mass": 1.4,
         "decay_channels": [
             {"daughters": [7006, 7007], "is_three_body": False},
         ]},
        {"pdg_id": 7009, "name": "mes_v", "B": 0, "Q": 1, "S": 0,
         "pole_mass": 0.3, "decay_channels": []},
        {"pdg_id": 7010, "name": "par_2", "B": 1, "Q": 0, "S": -1,
         "pole_mass": 1.6,
         "decay_channels": [
             {"daughters": [7004, 7005], "is_three_body": False},
             {"daughters": [7002, 7003, 7004], "is_three_body": True},
             {"daughters": [7004, 99999], "is_three_body": False},
         ]},
    ]


def _ta_families_config(classes):
    return {
        "families": [{
            "family_id": "baryonic_strange", "qn": [1, 0, -1],
            "allowed_decay_classes": list(classes),
        }],
    }


def _ta_persisting(bin_index, values, unit_se):
    """One test-author persisting K+ register entry (the sigma is the
    spread over the unit SE, as the sealed register carries it)."""
    return {
        "species": "k_plus",
        "bin_index": bin_index,
        "parent_index": 5,
        "confirmation_values": list(values),
        "confirmation_unit_se": unit_se,
        "confirmation_sigma": (max(values) - min(values)) / unit_se,
    }


def _ta_payload(k_plus, lam, s0, k0, sigma_plus, k_minus=0, sigma_minus=0,
                k0s=0, k0l=0, ak0=0, n_resonant=0, by_parent=None,
                conserved=True):
    species = {
        "K_plus": {"n": k_plus, "n_resonant": n_resonant,
                   "by_parent": {} if by_parent is None else by_parent},
        "Lambda": {"n": lam},
        "Sigma_0": {"n": s0},
        "K_0": {"n": k0},
        "Sigma_plus": {"n": sigma_plus},
        "K_minus": {"n": k_minus},
        "Sigma_minus": {"n": sigma_minus},
        "K0S": {"n": k0s}, "K0L": {"n": k0l}, "aK_0": {"n": ak0},
    }
    return {"conservation": {"event_b_q_conserved": conserved},
            "feed_down": {"per_species": species}}


def _ta_verdict_artifacts(driver, missing=(), nonconserving=(),
                          unattributable=(), byte_stable=True):
    """The test-author's synthetic criteria artifacts (3-bin K+
    summary, 3-class family surface)."""
    class_pairs = {
        "audit": {
            "family_id": "baryonic_strange", "qn": [1, 0, -1],
            "declared_classes": ["(7002,7003)", "(7004,7005)",
                                 "(7006,7007)"],
            "conservation_rows": [],
            "enumeration": {
                "conserving_class_keys": ["(7002,7003)", "(7004,7005)",
                                          "(7006,7007)"],
            },
        },
        "contract": {
            "complete": not list(missing) and not list(nonconserving),
            "conservative": not list(nonconserving),
            "missing_class_keys": list(missing),
            "nonconserving_declared": list(nonconserving),
        },
    }
    kplus = {
        "summary": {
            "persisting_kplus_bins": 3,
            "class_attributable": 3 - len(list(unattributable)),
            "unattributable_bins": list(unattributable),
        },
    }
    config_sanity = {
        "surface_digests": {
            "families_config_sha256": (
                FAMILIES_SHA256 if byte_stable else "0" * 64
            ),
            "catalog_sha256": (
                CATALOG_SHA256 if byte_stable else "1" * 64
            ),
            "byte_stable": byte_stable,
        },
    }
    return class_pairs, kplus, config_sanity


# ----------------------------------------------------------------------
# Part 1: the probe contract (pure, hand-built fixtures)
# ----------------------------------------------------------------------
def test_vertex_conservation_predicate_ta_fixture(driver):
    """The vertex-conservation predicate on the test-author's own
    (1, 0, -1) fixture: the conserving S = -2 partner pair (a K+
    daughter at (1, 0, -1) would need it), a non-conserving S = -2
    pair, the catalog-miss path, and the malformed key path."""
    by_pid = {int(r["pdg_id"]): r for r in _ta_catalog()}
    # hyp_b (1, 1, -2) + mes_b (0, -1, +1) sums to (1, 0, -1): the
    # conserving S = -2 partner case.
    row = driver.vertex_conserves("(903,904)", by_pid, (1, 0, -1))
    assert row["conserves"] is True
    assert row["daughter_qn_sums"] == [1, 0, -1]
    assert row["family_qn"] == [1, 0, -1]
    assert row["daughter_names"] == ["hyp_b", "mes_b"]
    assert row["daughter_pole_masses"] == [1.4, 0.28]
    # hyp_c (1, 0, -2) + mes_c (0, 0, 0) sums to (1, 0, -2): an S = -2
    # daughter without the strangeness partner does not conserve.
    row = driver.vertex_conserves("(906,905)", by_pid, (1, 0, -1))
    assert row["conserves"] is False
    assert row["daughter_qn_sums"] == [1, 0, -2]
    # A plain non-conserving pair (hyp_a + mes_a sums to (1, 0, -1)
    # here, so use the charge-breaking hyp_b + mes_c instead).
    row = driver.vertex_conserves("(903,905)", by_pid, (1, 0, -1))
    assert row["conserves"] is False
    assert row["daughter_qn_sums"] == [1, 1, -2]
    # A daughter missing from the catalog is flagged, never guessed.
    row = driver.vertex_conserves("(999,905)", by_pid, (1, 0, -1))
    assert row["conserves"] is False
    assert "not in the committed catalog" in row["reason"]
    # A malformed class key is flagged with its reason.
    row = driver.vertex_conserves("(901)", by_pid, (1, 0, -1))
    assert row["conserves"] is False
    assert "malformed" in row["reason"]
    # The committed declared set conserves on the on-disk catalog
    # (read-only cross-check of the predicate against the surface).
    catalog = _read_json(MONO_ROOT / "dev" / "mixing2006" / "config"
                         / "v2_singlet" / "resonance_catalog.json")
    by_pid = {int(r["pdg_id"]): r for r in catalog["records"]}
    for cls in STRANGE_DECLARED_CLASSES:
        row = driver.vertex_conserves(cls, by_pid, (1, 0, -1))
        assert row["conserves"] is True, cls
        assert row["daughter_qn_sums"] == [1, 0, -1], cls


def test_class_pair_enumeration_completeness_and_conservatism(driver):
    """The class-pair enumeration semantics on the test-author's own
    fixture: the universe carries exactly the three conserving class
    keys (the three-body and catalog-miss channels excluded), and a
    declared set with an intentionally missing pair plus an
    intentionally non-conserving pair is flagged on both counts."""
    fixture = _ta_enumeration_catalog()
    found = driver.enumerate_catalog_class_pairs(fixture, (1, 0, -1))
    assert found["qn"] == [1, 0, -1]
    assert found["conserving_class_keys"] == [
        "(7002,7003)", "(7004,7005)", "(7006,7007)",
    ]
    # The carrying record identifiers and the channel counts.
    assert found["classes"]["(7002,7003)"]["carrying_records"] == [7001]
    assert found["classes"]["(7002,7003)"]["channel_count"] == 1
    assert found["classes"]["(7004,7005)"]["carrying_records"] == [7010]
    assert found["classes"]["(7006,7007)"]["carrying_records"] == [7007]
    # Completeness: the intentionally missing (7004, 7005) pair is
    # named; conservativity: the non-conserving declared (7004, 7009)
    # pair (daughter sum (1, 1, -1)) is named.
    audit = driver.class_pair_audit_rows(
        _ta_families_config(
            ["(7002,7003)", "(7006,7007)", "(7004,7009)"],
        ),
        fixture,
    )
    assert audit["missing_class_keys"] == ["(7004,7005)"]
    assert audit["nonconserving_declared"] == ["(7004,7009)"]
    assert audit["declared_not_enumerated"] == ["(7004,7009)"]
    assert audit["complete"] is False
    assert audit["conservative"] is False
    # The fully declared conserving set is complete and conservative
    # (the contract of the committed surface, on this fixture).
    audit = driver.class_pair_audit_rows(
        _ta_families_config(
            ["(7002,7003)", "(7004,7005)", "(7006,7007)"],
        ),
        fixture,
    )
    assert audit["missing_class_keys"] == []
    assert audit["nonconserving_declared"] == []
    assert audit["declared_not_enumerated"] == []
    assert audit["complete"] is True
    assert audit["conservative"] is True
    # The B = 2 universe is empty on this fixture (no two-baryon
    # channel sums B = 2 with the recorded daughters: the two-baryon
    # pair (7002, 7004) is not recorded and sums (2, 1, -2) anyway).
    assert driver.enumerate_catalog_class_pairs(
        fixture, (2, 1, -2),
    )["conserving_class_keys"] == []


def test_kinematic_openness_boundary(driver):
    """The kinematic-openness predicate (daughter pole-mass sum below
    m_min) on the test-author's hand fixtures: the exact boundary
    (the sum equal to m_min is not open; the predicate is the strict
    ``<``), and the two open classes below it."""
    catalog = _ta_catalog()
    # The exact-boundary float pair (1.0 + 1.6801941107275278 == the
    # m_min double, verified on this platform's IEEE arithmetic).
    assert 1.0 + 1.6801941107275278 == driver.STRANGE_M_MIN
    audit = driver.class_pair_audit_rows(
        _ta_kinematics_config(), catalog,
    )
    assert audit["complete"] is True
    assert audit["conservative"] is True
    kin = {row["class_key"]: row for row in audit["kinematics"]}
    # The boundary class: the daughter pole-mass sum equals m_min
    # exactly, so it is not open across the family support.
    boundary = kin["(901,902)"]
    assert boundary["daughter_pole_mass_sum"] == driver.STRANGE_M_MIN
    assert boundary["open_across_family_support"] is False
    # Both lower classes are open, and every daughter sits below m_min
    # (the plan section 10.2 line 690 predicate).
    for key in ("(903,904)", "(901,905)"):
        assert kin[key]["open_across_family_support"] is True
        assert kin[key]["daughter_pole_mass_sum"] < driver.STRANGE_M_MIN
        assert kin[key]["at_least_one_daughter_lower_in_mass"] is True
    assert boundary["at_least_one_daughter_lower_in_mass"] is True


def test_kplus_decomposition_algebra_ta_fixture(driver):
    """The K+ decomposition algebra (the proportional census
    attribution ``y_v,c = s_c(v) * y_v``) on the test-author's own
    two-class fixtures: flat shares, a differential share case with
    hand-computed pooled-SE contributions summing to the bin sigma
    exactly, and the dominant-class flip."""
    # Flat shares: each class carries exactly half the payload and
    # exactly half the pooled-SE sigma; the tie breaks to the K+
    # Lambda class (the driver's ``>=`` on the contribution).
    shares = {"0": {"share_Lambda": 0.5, "share_Sigma0": 0.5},
              "1": {"share_Lambda": 0.5, "share_Sigma0": 0.5},
              "2": {"share_Lambda": 0.5, "share_Sigma0": 0.5}}
    entry = _ta_persisting(11, [0.012, 0.008, 0.010], 0.002)
    rows = driver.kplus_bin_rows([entry], shares, {}, {})
    row = rows[0]
    assert row["confirmation_sigma"] == pytest.approx(2.0)
    assert row["class_values"]["(3122,321)_K_Lambda"] == pytest.approx(
        [0.006, 0.004, 0.005]
    )
    assert row["contributions_pooled_se_units"][
        "(3122,321)_K_Lambda"] == pytest.approx(1.0)
    assert row["contributions_pooled_se_units"][
        "(3212,321)_K_Sigma0"] == pytest.approx(1.0)
    assert row["contribution_sum"] == pytest.approx(2.0)
    assert row["dominant_class_pair"] == "(3122,321)"
    assert row["class_attributable"] is True
    # Differential shares, hand-computed: values (4.0, 2.0, 1.0) at
    # unit SE 0.4 (bin sigma 7.5); s_Lambda = (0.7, 0.5, 0.3) gives
    # y_Lambda = (2.8, 1.0, 0.3), spread 2.5, contribution 6.25;
    # s_Sigma0 = (0.3, 0.5, 0.7) gives y_Sigma0 = (1.2, 1.0, 0.7),
    # spread 0.5, contribution 1.25; the contributions sum to the
    # bin sigma exactly (the aligned monotone case).
    shares = {"0": {"share_Lambda": 0.7, "share_Sigma0": 0.3},
              "1": {"share_Lambda": 0.5, "share_Sigma0": 0.5},
              "2": {"share_Lambda": 0.3, "share_Sigma0": 0.7}}
    entry = _ta_persisting(12, [4.0, 2.0, 1.0], 0.4)
    rows = driver.kplus_bin_rows([entry], shares, {}, {})
    row = rows[0]
    assert row["confirmation_sigma"] == pytest.approx(7.5)
    assert row["class_values"]["(3122,321)_K_Lambda"] == pytest.approx(
        [2.8, 1.0, 0.3]
    )
    assert row["class_values"]["(3212,321)_K_Sigma0"] == pytest.approx(
        [1.2, 1.0, 0.7]
    )
    assert row["contributions_pooled_se_units"][
        "(3122,321)_K_Lambda"] == pytest.approx(6.25)
    assert row["contributions_pooled_se_units"][
        "(3212,321)_K_Sigma0"] == pytest.approx(1.25)
    assert row["contribution_sum"] == pytest.approx(7.5)
    assert row["dominant_class_pair"] == "(3122,321)"
    # The dominant-class flip: the mirrored shares reverse the
    # contributions on the same payload.
    shares = {"0": {"share_Lambda": 0.3, "share_Sigma0": 0.7},
              "1": {"share_Lambda": 0.5, "share_Sigma0": 0.5},
              "2": {"share_Lambda": 0.7, "share_Sigma0": 0.3}}
    rows = driver.kplus_bin_rows([entry], shares, {}, {})
    row = rows[0]
    assert row["contributions_pooled_se_units"][
        "(3122,321)_K_Lambda"] == pytest.approx(1.25)
    assert row["contributions_pooled_se_units"][
        "(3212,321)_K_Sigma0"] == pytest.approx(6.25)
    assert row["contribution_sum"] == pytest.approx(7.5)
    assert row["dominant_class_pair"] == "(3212,321)"
    # The class-decomposed share is the proportional census
    # attribution (the pure kernel).
    assert driver.class_decomposed_share([4.0, 2.0, 1.0],
                                         [0.7, 0.5, 0.3]) == (
        pytest.approx([2.8, 1.0, 0.3])
    )


def test_sealed_count_identities_ta_fixture(driver):
    """The sealed per-replica count identities (K+ = Lambda + Sigma0;
    K- = Sigma- = K0S = K0L = aK0 = 0; K0 = Sigma+) on the
    test-author's own two-class hand fixtures: a clean replica, one
    broken payload per identity (each violation named), and the
    pooled two-replica share arithmetic."""
    # A clean replica: all four identities hold.
    clean = _ta_payload(84, 36, 48, k0=27, sigma_plus=27)
    assert driver.replica_identities(clean, "ta_fixture") == []
    # Identity (i): K+ != Lambda + Sigma0.
    violations = driver.replica_identities(
        _ta_payload(85, 36, 48, k0=27, sigma_plus=27), "ta_fixture",
    )
    assert any("K_plus.n == Lambda.n + Sigma_0.n" in v
               for v in violations)
    # Identity (ii): a resonant K+ final with a non-empty by-parent
    # census violates the root-vertex pin.
    violations = driver.replica_identities(
        _ta_payload(84, 36, 48, k0=27, sigma_plus=27, n_resonant=1,
                    by_parent={"3212": 1}),
        "ta_fixture",
    )
    assert any("K_plus.n_resonant == 0" in v for v in violations)
    # Identity (iii): each of the zero-strange species is named.
    for name in ("K_minus", "Sigma_minus", "K0S", "K0L", "aK_0"):
        payload = _ta_payload(84, 36, 48, k0=27, sigma_plus=27)
        payload["feed_down"]["per_species"][name]["n"] = 1
        violations = driver.replica_identities(payload, "ta_fixture")
        assert any(f"{name}.n == 1" in v for v in violations), name
    # Identity (iv): K0 != Sigma+.
    violations = driver.replica_identities(
        _ta_payload(84, 36, 48, k0=28, sigma_plus=27), "ta_fixture",
    )
    assert any("K_0.n == Sigma_plus.n" in v for v in violations)
    # A false conservation flag is flagged.
    violations = driver.replica_identities(
        _ta_payload(84, 36, 48, k0=27, sigma_plus=27, conserved=False),
        "ta_fixture",
    )
    assert any("conservation flag" in v for v in violations)
    # The pooled two-replica arithmetic (the shares of the sealed
    # per-variant census): K+ 184 = Lambda 76 + Sigma0 108, K0 57 =
    # Sigma+ 57.
    pooled = driver.pooled_class_counts([
        _ta_payload(84, 36, 48, k0=27, sigma_plus=27),
        _ta_payload(100, 40, 60, k0=30, sigma_plus=30),
    ])
    assert pooled["replicas"] == 2
    assert pooled["counts"]["K_plus"] == 184
    assert pooled["counts"]["Lambda"] == 76
    assert pooled["counts"]["Sigma_0"] == 108
    assert pooled["share_Lambda"] == pytest.approx(76 / 184)
    assert pooled["share_Sigma0"] == pytest.approx(108 / 184)
    assert pooled["share_Lambda"] + pooled["share_Sigma0"] == (
        pytest.approx(1.0, rel=1e-15)
    )
    assert pooled["identity_kplus_hyperon"] is True
    assert pooled["identity_k0_sigma"] is True


def test_verdict_criteria_logic_ta_fixture(driver):
    """The verdict-criteria logic on the test-author's own synthetic
    artifacts: PASS iff the class set is complete and conservative,
    all K+ bins class-attributable, and the config bytes intact;
    RECOMMENDATION on (a), (b), (c), and on the compound case in the
    (a), (b), (c) order of the driver's criteria tuple."""
    # The all-clean PASS path.
    cp, kp, cs = _ta_verdict_artifacts(driver)
    evaluation = driver.evaluate_verdict_criteria(cp, kp, cs)
    assert evaluation["verdict"] == "PASS"
    assert evaluation["fired"] == []
    assert all(
        c["fired"] is False for c in evaluation["criteria"].values()
    )
    # Criterion (a): a class pair missing or non-conserving (each
    # side separately).
    cp, kp, cs = _ta_verdict_artifacts(
        driver, missing=["(7004,7005)"],
    )
    evaluation = driver.evaluate_verdict_criteria(cp, kp, cs)
    assert evaluation["verdict"] == "RECOMMENDATION"
    assert [c["criterion"] for c in evaluation["fired"]] == ["a"]
    cp, kp, cs = _ta_verdict_artifacts(
        driver, nonconserving=["(7004,7009)"],
    )
    evaluation = driver.evaluate_verdict_criteria(cp, kp, cs)
    assert [c["criterion"] for c in evaluation["fired"]] == ["a"]
    # Criterion (b): an unattributable persisting K+ bin.
    cp, kp, cs = _ta_verdict_artifacts(driver, unattributable=[41])
    evaluation = driver.evaluate_verdict_criteria(cp, kp, cs)
    assert evaluation["verdict"] == "RECOMMENDATION"
    assert [c["criterion"] for c in evaluation["fired"]] == ["b"]
    assert evaluation["criteria"]["b"]["unattributable_bins"] == [41]
    # Criterion (c): the config disagrees with its committed bytes.
    cp, kp, cs = _ta_verdict_artifacts(driver, byte_stable=False)
    evaluation = driver.evaluate_verdict_criteria(cp, kp, cs)
    assert evaluation["verdict"] == "RECOMMENDATION"
    assert [c["criterion"] for c in evaluation["fired"]] == ["c"]
    assert evaluation["criteria"]["c"]["byte_stable"] is False
    # The compound case: every criterion fires, in order.
    cp, kp, cs = _ta_verdict_artifacts(
        driver, missing=["(7004,7005)"], nonconserving=["(7004,7009)"],
        unattributable=[41], byte_stable=False,
    )
    evaluation = driver.evaluate_verdict_criteria(cp, kp, cs)
    assert evaluation["verdict"] == "RECOMMENDATION"
    assert [c["criterion"] for c in evaluation["fired"]] == (
        ["a", "b", "c"]
    )


# ----------------------------------------------------------------------
# The zero-draw invariants (test-author technique)
# ----------------------------------------------------------------------
class _ModuleSentinel:
    """Any import-time or attribute-time touch of the numpy/random
    module slots during the audit's pure surface is a zero-draw
    contract violation."""

    def __getattr__(self, name):
        raise AssertionError(
            f"the audit driver touched the banned module slot {name!r} "
            "(zero-draw contract: standard library only)"
        )


def test_zero_draw_no_rng_no_numpy(driver, monkeypatch):
    """The driver performs no RNG draws and imports no numpy: the AST
    import surface is pinned to the standard library, the source of
    both driver trees carries no random-number machinery (the
    test-author's own token list), a fresh isolated interpreter
    confirms the driver's import pulls no numpy/random into
    ``sys.modules``, and the full pure surface is exercised under
    raising sentinels on the ``numpy`` and ``random`` module slots."""
    # The AST import analysis (the test-author's technique, both
    # trees): the full import surface is the standard library.
    expected_imports = {
        "__future__", "argparse", "datetime", "hashlib", "json",
        "math", "pathlib", "subprocess", "sys", "typing",
    }
    for source_path in (NESTED_DRIVER, OUTER_DRIVER):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    modules.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    modules.add(node.module.split(".")[0])
        assert modules == expected_imports, (
            f"{source_path}: the driver import surface {sorted(modules)} "
            f"!= the pinned standard library {sorted(expected_imports)}"
        )
        assert "numpy" not in modules and "random" not in modules
    # The source-level token scan (the test-author's list).
    forbidden = (
        "import numpy", "import random", "numpy.random", "np.random",
        "default_rng", "SeedSequence", "RandomState", "randint(",
        "randn(", "shuffle(", "permutation(", "choice(", "gauss(",
        "getstate", "setstate", "bit_generator", "tofile",
    )
    for source_path in (NESTED_DRIVER, OUTER_DRIVER):
        source = source_path.read_text(encoding="utf-8")
        hits = [token for token in forbidden if token in source]
        assert not hits, f"{source_path}: RNG machinery {hits}"
    # The fresh isolated-interpreter probe: importing the driver
    # pulls no numpy/random into sys.modules.
    probe_code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location("
        f"'r4_ta_probe', {str(NESTED_DRIVER)!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "print([n for n in ('numpy', 'random') if n in sys.modules])\n"
    )
    probe = subprocess.run(
        [sys.executable, "-I", "-c", probe_code],
        capture_output=True, text=True,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "[]", (
        f"the driver import pulled banned modules into sys.modules: "
        f"{probe.stdout.strip()}"
    )
    # The full pure surface under the module sentinels.
    sentinel = _ModuleSentinel()
    monkeypatch.setitem(sys.modules, "numpy", sentinel)
    monkeypatch.setitem(sys.modules, "random", sentinel)
    by_pid = {int(r["pdg_id"]): r for r in _ta_catalog()}
    assert driver.vertex_conserves(
        "(903,904)", by_pid, (1, 0, -1),
    )["conserves"] is True
    assert driver.enumerate_catalog_class_pairs(
        _ta_enumeration_catalog(), (1, 0, -1),
    )["conserving_class_keys"] == [
        "(7002,7003)", "(7004,7005)", "(7006,7007)",
    ]
    audit = driver.class_pair_audit_rows(
        _ta_kinematics_config(), _ta_catalog(),
    )
    assert audit["complete"] is True and audit["conservative"] is True
    shares = {"0": {"share_Lambda": 0.7, "share_Sigma0": 0.3},
              "1": {"share_Lambda": 0.5, "share_Sigma0": 0.5},
              "2": {"share_Lambda": 0.3, "share_Sigma0": 0.7}}
    rows = driver.kplus_bin_rows([_ta_persisting(11, [4.0, 2.0, 1.0],
                                                 0.4)], shares, {}, {})
    # (a bare tolerance, not ``pytest.approx``: the approx machinery
    # itself probes the numpy module slot and would trip the sentinel)
    assert abs(rows[0]["contribution_sum"] - 7.5) < 1e-12
    pooled = driver.pooled_class_counts([
        _ta_payload(84, 36, 48, k0=27, sigma_plus=27),
    ])
    assert pooled["identity_kplus_hyperon"] is True
    assert driver.replica_identities(
        _ta_payload(84, 36, 48, k0=27, sigma_plus=27), "ta_fixture",
    ) == []
    cp, kp, cs = _ta_verdict_artifacts(driver)
    assert driver.evaluate_verdict_criteria(cp, kp, cs)["verdict"] == (
        "PASS"
    )
    # The recorded zero-draw contract of the pre-registration builder.
    pre = driver.build_pre_registration(driver.substrate_digests(), [])
    assert pre["zero_draw"]["draws"] == 0
    assert pre["zero_draw"]["no_fitting"] is True
    assert pre["zero_draw"]["optimizer_invoked"] is False
    block = pre["zero_draw"]["reserved_audit_block"]
    assert block["fired"] is False
    assert block["master_seed"] == 20260920
    assert block["raw_seeds"] == [20301040, 20301055]
    assert pre["probe_set"]["expected_kplus_entries"] == 18


# ----------------------------------------------------------------------
# Part 2: the sealed artifacts (run-dir pins)
# ----------------------------------------------------------------------
def test_driver_dual_tree_byte_identity():
    """The nested and outer audit driver copies are byte-identical at
    the pinned digest (the dual-tree register; nested 95acd0b, outer
    0bf531c)."""
    assert NESTED_DRIVER.is_file(), NESTED_DRIVER
    assert OUTER_DRIVER.is_file(), OUTER_DRIVER
    nested_bytes = NESTED_DRIVER.read_bytes()
    outer_bytes = OUTER_DRIVER.read_bytes()
    assert nested_bytes == outer_bytes
    assert _sha256_file(NESTED_DRIVER) == AUDIT_DRIVER_SHA256
    assert _sha256_file(OUTER_DRIVER) == AUDIT_DRIVER_SHA256


def test_surface_byte_stability():
    """The cards 4/5 family surface and the frozen substrate stay at
    their pinned on-disk digests (re-hashed independently of the
    driver; a surface change is a separate rework card)."""
    assert _sha256_file(
        MONO_ROOT / "dev" / "mixing2006" / "config" / "v2_singlet"
        / "continuum_resonances.json"
    ) == FAMILIES_SHA256
    pass  # source digest pin retired: the freeze manifest pins every file (TASK-20261004-refactor-run-engine-kernel-config P4)
    assert _sha256_file(
        MONO_ROOT / "dev" / "mixing2006" / "na49_mixing"
        / "spectral_density.py"
    ) == CARD6_SHA256
    assert _sha256_file(
        MONO_ROOT / "dev" / "mixing2006" / "na49_mixing"
        / "resonance_pp_builder.py"
    ) == BUILDER_SHA256


def test_sealed_run_dir_seal_pins():
    """The seal manifest and the SHA256SUMS.txt integrity of the sealed
    audit run dir: the on-disk seal manifest carries 7 files
    totalling 100,966 bytes (the clean-environment record is the
    documented post-seal file and is never seal-covered); every
    listed file re-hashes to its recorded digest; the
    SHA256SUMS.txt inventory covers the 6 pre-seal payload files with
    monorepo-root-relative paths, and every entry re-hashes from the
    monorepo root; the clean-environment record verifies."""
    run_dir = _skip_if_no_run_dir()
    seal = _read_json(run_dir / "seal_manifest.json")
    assert seal["card"] == CARD_ID
    assert seal["file_count"] == SEAL_FILE_COUNT == 7
    assert seal["total_bytes"] == SEAL_TOTAL_BYTES == 100966
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
        prefix = "dev/00runs/r4_audit_strange_sector_continuum_families_"
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
        "class-pairs": 0, "k-contribution": 0, "config-sanity": 0,
        "verdict": 0,
    }
    assert clean["probe_failures"] == []
    assert clean["sealed_artifact_sha256"] == clean[
        "recomputed_artifact_sha256"
    ]
    assert clean["probe_tool"]["sha256"] == AUDIT_DRIVER_SHA256


def test_sealed_run_dir_verdict_pins():
    """The verdict.json content pins: the PASS verdict with 0 named
    findings and the pass text (the card-1 referral recorded); the
    18-bin probe summary; the class-pair summary (6/6 conserve, 6/6
    declared, complete + conservative); the K+ decomposition summary
    (all 18 bins class-attributable, dominant classes 16/2, the
    identity sweep clean); the card-1 referral numbers (15/18
    monotone, 15/18 continuum-dominant, 16/18 mass-blind envelope
    violations); the config-sanity summary (byte-stable, 0 problems,
    the kinematic extreme); the bounds; and the zero-draw record."""
    run_dir = _skip_if_no_run_dir()
    assert _sha256_file(run_dir / "verdict.json") == (
        SEAL_FILES["verdict.json"][0]
    )
    verdict = _read_json(run_dir / "verdict.json")
    assert verdict["card"] == CARD_ID
    assert verdict["verdict"] == "PASS"
    assert verdict["findings"] == []
    assert verdict["recommendation"] is None
    assert verdict["pass_text"] is not None
    assert "adequate as-is" in verdict["pass_text"]
    assert "6 of 6 declared classes" in verdict["pass_text"]
    assert "not a missing-decay-class artifact" in verdict["pass_text"]
    # The card-1 referral names "card 1" and carries the rework card id
    # verbatim (the sealed pass text does not spell out the full card-1
    # id).
    assert "card 1" in verdict["pass_text"]
    assert CARD1_REWORK_ID in verdict["pass_text"]
    # The criteria: none fired.
    for key in ("a", "b", "c"):
        assert verdict["criteria"][key]["fired"] is False
    criterion_a = verdict["criteria"]["a"]
    assert criterion_a["family_id"] == "baryonic_strange"
    assert criterion_a["family_qn"] == [1, 0, -1]
    assert criterion_a["declared_classes"] == 6
    assert criterion_a["conserving_catalog_class_keys"] == 6
    assert criterion_a["missing_class_keys"] == []
    assert criterion_a["nonconserving_declared"] == []
    assert criterion_a["complete"] is True
    assert criterion_a["conservative"] is True
    assert verdict["criteria"]["b"]["persisting_kplus_bins"] == 18
    assert verdict["criteria"]["b"]["class_attributable"] == 18
    assert verdict["criteria"]["b"]["unattributable_bins"] == []
    assert verdict["criteria"]["c"]["byte_stable"] is True
    # The probe set (the 18 K+ persisting bins).
    assert verdict["probe_set"]["entries"] == 18
    assert verdict["probe_set"]["bin_indices"] == KPLUS_BIN_INDICES
    assert verdict["probe_set"]["sigma_range"] == KPLUS_SIGMA_RANGE
    # The class-pair summary.
    summary = verdict["class_pair_summary"]
    assert summary["complete"] is True
    assert summary["conservative"] is True
    assert summary["missing_class_keys"] == []
    assert summary["nonconserving_declared"] == []
    assert summary["declared_classes"] == STRANGE_DECLARED_CLASSES
    assert summary["conserving_catalog_class_keys"] == sorted(
        STRANGE_DECLARED_CLASSES
    )
    # The K+ decomposition summary.
    decomposition = verdict["kplus_decomposition_summary"]
    assert decomposition["persisting_kplus_bins"] == 18
    assert decomposition["class_attributable"] == 18
    assert decomposition["all_class_attributable"] is True
    assert decomposition["identity_sweep_clean"] is True
    assert decomposition["dominant_class_pair_counts"] == {
        "(3122,321)": 16, "(3212,321)": 2,
    }
    per_bin = decomposition["per_bin"]
    assert len(per_bin) == 18
    assert [row["bin_index"] for row in per_bin] == KPLUS_BIN_INDICES
    assert min(row["density_shape_share"] for row in per_bin) == (
        pytest.approx(KPLUS_SIGMA_RANGE[0], rel=1e-12)
    )
    # The card-1 referral numbers (independently recomputed from the
    # sealed per-bin rows).
    assert sum(1 for row in per_bin if row["card1_monotone"]) == 15
    assert sum(
        1 for row in per_bin
        if row["card1_route_dominance"] == "continuum_dominant"
    ) == 15
    assert sum(
        1 for row in per_bin
        if not row["card1_mass_blind_envelope_respected"]
    ) == 16
    # The config-sanity summary.
    sanity = verdict["config_sanity_summary"]
    assert sanity["byte_stable"] is True
    assert sanity["families_config_sha256"] == FAMILIES_SHA256
    assert sanity["catalog_sha256"] == CATALOG_SHA256
    assert sanity["strange_record_problems"] == []
    assert sanity["inactive_problems"] == []
    assert sanity["r30_blocks_pass"] is True
    assert sanity["r31_strange_window_interior_failed"] == 0
    assert sanity["kinematic_max_pole_mass_sum"] == (
        pytest.approx(KIN_MAX_POLE_MASS_SUM, rel=1e-12)
    )
    # The bounds (carried verbatim, never relaxed) and the referral.
    assert verdict["bounds"]["bin_spread_p95_bound"] == 2.0
    assert verdict["bounds"]["feeddown_spread_max_bound"] == 3.0
    assert verdict["bounds"]["kplus_p95_anchor_production"] == (
        KPLUS_P95_ANCHOR
    )
    assert verdict["density_shape_referral"]["referred_to_card"] == (
        CARD1_ID
    )
    assert verdict["density_shape_referral"]["rework_card"] == (
        CARD1_REWORK_ID
    )
    # The zero-draw record and the surface byte-stability statement.
    assert verdict["zero_draw"]["draws"] == 0
    assert verdict["zero_draw"]["no_fitting"] is True
    assert verdict["zero_draw"]["optimizer_invoked"] is False
    assert verdict["zero_draw"]["reserved_audit_block_fired"] is False
    assert verdict["surface"]["families_sha256"] == FAMILIES_SHA256
    assert verdict["surface"]["catalog_sha256"] == CATALOG_SHA256
    assert verdict["surface"]["byte_unchanged_by_this_audit"] is True
    # The pre-registered verdict contract carried verbatim.
    assert verdict["verdict_contract"]["criteria"] == ["a", "b", "c"]


def test_sealed_run_dir_class_pair_pins():
    """The class_pair_audit.json pins: the 6/6 conservation re-
    derivation (every declared class conserves (1, 0, -1)), the 6/6
    completeness (every conserving catalog class pair is declared),
    the carrying catalog record ids 3214 / 102134 with their channel
    counts, the kinematic openness of every class (max daughter
    pole-mass sum 1.437176 < m_min), the empty (H_b, H_c)
    enumeration, the informational inactive rows, and the zero-draw
    stamps."""
    run_dir = _skip_if_no_run_dir()
    assert _sha256_file(run_dir / "class_pair_audit.json") == (
        SEAL_FILES["class_pair_audit.json"][0]
    )
    audit_payload = _read_json(run_dir / "class_pair_audit.json")
    assert audit_payload["card"] == CARD_ID
    audit = audit_payload["audit"]
    assert audit["family_id"] == "baryonic_strange"
    assert audit["qn"] == [1, 0, -1]
    assert audit["declared_classes"] == STRANGE_DECLARED_CLASSES
    # The conservation rows: 6/6 conserve with the family sum.
    assert len(audit["conservation_rows"]) == 6
    for row in audit["conservation_rows"]:
        assert row["conserves"] is True
        assert row["daughter_qn_sums"] == [1, 0, -1]
        assert row["family_qn"] == [1, 0, -1]
    assert audit["nonconserving_declared"] == []
    # The completeness: the enumeration carries exactly the declared
    # 6 conserving class keys, nothing missing, nothing undeclared.
    enumeration = audit["enumeration"]
    assert enumeration["conserving_class_keys"] == sorted(
        STRANGE_DECLARED_CLASSES
    )
    assert audit["missing_class_keys"] == []
    assert audit["declared_not_enumerated"] == []
    assert audit["complete"] is True
    assert audit["conservative"] is True
    # The carrying catalog record ids (3214 Sigma(1385)0, 102134
    # Lambda(1520)) and the channel counts per class key.
    carrying = {
        key: entry["carrying_records"]
        for key, entry in enumeration["classes"].items()
    }
    assert carrying["(3122,111)"] == [3214]
    assert carrying["(2212,-321)"] == [102134]
    assert carrying["(2112,-311)"] == [102134]
    assert carrying["(3212,111)"] == [102134]
    # (the two-channel classes carry the record ids in descending pid
    # order on disk)
    assert carrying["(3112,211)"] == [102134, 3214]
    assert carrying["(3222,-211)"] == [102134, 3214]
    counts = {
        key: entry["channel_count"]
        for key, entry in enumeration["classes"].items()
    }
    assert counts["(3112,211)"] == 2
    assert counts["(3222,-211)"] == 2
    for key in ("(3122,111)", "(2212,-321)", "(2112,-311)",
                "(3212,111)"):
        assert counts[key] == 1
    # The kinematics: every class open across the family support, the
    # max daughter pole-mass sum at the (2112, -311) n-Kbar0 pair.
    kin = {row["class_key"]: row for row in audit["kinematics"]}
    assert len(kin) == 6
    assert all(row["open_across_family_support"] for row in kin.values())
    assert all(
        row["at_least_one_daughter_lower_in_mass"] for row in kin.values()
    )
    assert kin["(2112,-311)"]["daughter_pole_mass_sum"] == (
        pytest.approx(KIN_MAX_POLE_MASS_SUM, rel=1e-15)
    )
    assert max(row["daughter_pole_mass_sum"] for row in kin.values()) == (
        pytest.approx(KIN_MAX_POLE_MASS_SUM, rel=1e-15)
    )
    # The (H_b, H_c) enumeration: no two-baryon channel recorded.
    assert audit_payload["h_b_h_c_enumeration"]["recorded_count"] == 0
    assert (audit_payload["h_b_h_c_enumeration"]
            ["recorded_two_baryon_channels"]) == []
    # The informational inactive strange mesonic families.
    informational = {
        row["family_id"]: row
        for row in audit_payload["informational_inactive_strange_mesonic"]
    }
    assert set(informational) == {
        "mesonic_strange", "mesonic_antistrange",
    }
    for row in informational.values():
        assert row["active"] is False
        assert row["complete"] is True
        assert row["conservative"] is True
        assert row["scope"] == (
            "informational (outside the verdict contract)"
        )
    # The contract block, the sources, and the zero-draw stamps.
    assert audit_payload["contract"] == {
        "complete": True, "conservative": True, "missing_class_keys":
        [], "nonconserving_declared": [], "verdict_relevant": True,
    }
    assert audit_payload["sources"]["catalog_records"] == 98
    assert audit_payload["sources"]["families_config_sha256"] == (
        FAMILIES_SHA256
    )
    assert audit_payload["sources"]["catalog_sha256"] == CATALOG_SHA256
    assert audit_payload["no_fitting"] is True
    assert audit_payload["optimizer_invoked"] is False
    assert audit_payload["draws"] == 0


def test_sealed_run_dir_config_sanity_pins():
    """The config_sanity.json pins: the byte-stable surface digests,
    0 strange-record problems and 0 inactive problems, the family
    census (3 active / 4 declared inactive), every committed-
    parameter field matching, the R30 blocks all passing, the R31
    window with 0 boundary and 0 interior failures, the kinematic
    recomputation (max 1.437176, all open), and the 1.432-vs-1.437176
    informational note (not a parameter drift)."""
    run_dir = _skip_if_no_run_dir()
    assert _sha256_file(run_dir / "config_sanity.json") == (
        SEAL_FILES["config_sanity.json"][0]
    )
    sanity = _read_json(run_dir / "config_sanity.json")
    assert sanity["card"] == CARD_ID
    assert sanity["surface_digests"]["families_config_sha256"] == (
        FAMILIES_SHA256
    )
    assert sanity["surface_digests"]["catalog_sha256"] == CATALOG_SHA256
    assert sanity["surface_digests"]["byte_stable"] is True
    assert sanity["strange_record_problems"] == []
    assert sanity["inactive_problems"] == []
    # The family census of the committed configuration.
    census = sanity["family_census"]
    assert census["active"] == {
        "baryonic_np": {"qn": [1, 1, 0], "classes": 8,
                        "state_type": "continuum_resonance"},
        "baryonic_strange": {"qn": [1, 0, -1], "classes": 6,
                             "state_type": "continuum_resonance"},
        "mesonic_neutral": {"qn": [0, 0, 0], "classes": 10,
                            "state_type": "continuum_resonance"},
    }
    assert set(census["inactive"]) == {
        "mesonic_charged", "mesonic_anticharged", "mesonic_strange",
        "mesonic_antistrange",
    }
    assert census["inactive"]["mesonic_strange"]["qn"] == [0, 0, -1]
    assert census["inactive"]["mesonic_strange"]["classes"] == 7
    assert census["inactive"]["mesonic_antistrange"]["qn"] == [0, 0, 1]
    assert census["inactive"]["mesonic_antistrange"]["classes"] == 7
    # The committed-parameter check: every field matches.
    fields = sanity["strange_record_check"]["fields"]
    assert set(fields) == {
        "m_min", "t_h", "b_power", "n_norm", "ramp_width",
        "nominal_width", "m_max", "qn", "state_type", "active",
        "allowed_decay_classes",
    }
    assert all(field["matches"] is True for field in fields.values())
    assert fields["m_min"]["recorded"] == 2.680194110727528
    assert fields["t_h"]["recorded"] == 0.165
    assert fields["allowed_decay_classes"]["recorded"] == (
        STRANGE_DECLARED_CLASSES
    )
    # The R30 normalization cross-reference (all three blocks pass).
    r30 = sanity["r30_cross_reference"]
    assert set(r30) == {"family_normalization", "threshold_behavior",
                        "double_counting_identity"}
    assert all(block["pass"] is True for block in r30.values())
    assert r30["double_counting_identity"]["strange_qn_covered"] is True
    # The R31 smoothness cross-reference (the (1, 0, -1) window).
    r31 = sanity["r31_cross_reference"]
    assert r31["qn"] == [1, 0, -1]
    assert r31["boundary_pair_failed"] == 0
    assert r31["interior_pairs_failed"] == 0
    assert r31["no_boundary_dominance"] is True
    # The kinematic recomputation and the informational note.
    kin = sanity["kinematic_recomputation"]
    assert kin["max_pole_mass_sum"] == (
        pytest.approx(KIN_MAX_POLE_MASS_SUM, rel=1e-15)
    )
    assert kin["all_open_across_support"] is True
    note = kin["provenance_quote_note"]
    assert "1.432" in note
    assert "1.437176" in note
    assert "0.005176" in note
    assert "(2112, -311)" in note
    # The sampling scope (the R32 p+p source topology) and the
    # zero-draw stamps.
    assert sanity["sampling_scope"]["source_qn_r32"] == [1, 1, 0]
    assert sanity["sampling_scope"]["builder_source_qn_pin_found"] is True
    assert sanity["no_fitting"] is True
    assert sanity["optimizer_invoked"] is False
    assert sanity["draws"] == 0


def test_sealed_run_dir_kplus_pins():
    """The kplus_class_contribution.json pins: the 18-bin probe set,
    the identity sweep over the 72 sealed replicas (24 production +
    48 confirmation) clean, the sealed confirmation per-variant class
    shares with the identity counts (K+ 624971 = Lambda 313984 +
    Sigma0 310987; K0 311297 = Sigma+ 311297; the zero-strange
    species identically 0), the per-bin contributions summing to the
    bin sigma, the dominant-class counts (16/2), and the card-1
    referral block."""
    run_dir = _skip_if_no_run_dir()
    assert _sha256_file(run_dir / "kplus_class_contribution.json") == (
        SEAL_FILES["kplus_class_contribution.json"][0]
    )
    kplus = _read_json(run_dir / "kplus_class_contribution.json")
    assert kplus["card"] == CARD_ID
    assert kplus["probe_set"]["entries"] == 18
    assert kplus["probe_set"]["bin_indices"] == KPLUS_BIN_INDICES
    assert kplus["probe_set"]["sigma_range"] == KPLUS_SIGMA_RANGE
    # The identity sweep: 72 sealed replicas, clean.
    assert kplus["identity_sweep"] == {
        "production_payloads": 24, "confirmation_payloads": 48,
        "violations": [], "clean": True,
    }
    # The sealed confirmation per-variant v0 pooled class counts and
    # shares (the identity counts pinned on disk).
    v0 = kplus["class_shares"]["confirmation"]["0"]
    assert v0["replicas"] == 16
    assert v0["counts"] == CONF_V0_COUNTS
    assert v0["share_Lambda"] == (
        pytest.approx(CONF_V0_SHARE_LAMBDA, rel=1e-15)
    )
    assert v0["share_Lambda"] + v0["share_Sigma0"] == (
        pytest.approx(1.0, rel=1e-15)
    )
    assert CONF_V0_COUNTS["K_plus"] == (
        CONF_V0_COUNTS["Lambda"] + CONF_V0_COUNTS["Sigma_0"]
    )
    assert CONF_V0_COUNTS["K_0"] == CONF_V0_COUNTS["Sigma_plus"]
    # The per-bin rows: all class-attributable, the contributions
    # summing to the bin sigma, the dominant-class counts 16/2.
    rows = kplus["rows"]
    assert len(rows) == 18
    assert all(row["class_attributable"] is True for row in rows)
    for row in rows:
        assert row["contribution_sum"] == pytest.approx(
            row["confirmation_sigma"], rel=1e-12,
        )
        assert row["bound"] == 2.0
    assert kplus["summary"]["persisting_kplus_bins"] == 18
    assert kplus["summary"]["class_attributable"] == 18
    assert kplus["summary"]["unattributable_bins"] == []
    assert kplus["summary"]["all_class_attributable"] is True
    assert kplus["summary"]["dominant_class_pair_counts"] == {
        "(3122,321)": 16, "(3212,321)": 2,
    }
    # The card-1 cross-reference (the density-shape referral block).
    reference = kplus["card1_cross_reference"]
    assert reference["card"] == CARD1_ID
    assert reference["verdict"] == "RECOMMENDATION"
    referral = reference["k_plus_referral"]
    assert referral["referred_to_card"] == CARD_ID
    assert referral["bins"] == 18
    assert referral["monotone"] == 15
    assert referral["within_census_envelope"] == 18
    assert referral["mass_blind_envelope_violations"] == 16
    assert referral["continuum_dominant"] == 15
    # The zero-draw stamps.
    assert kplus["no_fitting"] is True
    assert kplus["optimizer_invoked"] is False
    assert kplus["draws"] == 0


def test_sealed_run_dir_prereg_provenance_pins():
    """The pre-registration immutability pin (the sha256 as on disk),
    the probe-set contract, the zero-draw contract with the reserved
    audit block 20301040 + k not fired, the bounds, the sealed
    verification record, and the provenance driver digest pin
    (nested == outer == the on-disk dual-tree digest)."""
    run_dir = _skip_if_no_run_dir()
    assert _sha256_file(run_dir / "pre_registration.json") == (
        PRE_REGISTRATION_SHA256
    )
    pre = _read_json(run_dir / "pre_registration.json")
    assert pre["card"] == CARD_ID
    assert pre["written_before_measurement"] is True
    assert pre["probe_set"]["expected_kplus_entries"] == 18
    assert pre["probe_set"]["expected_bin_indices"] == KPLUS_BIN_INDICES
    assert pre["probe_set"]["expected_sigma_range"] == KPLUS_SIGMA_RANGE
    # The zero-draw contract and the seed-block-not-fired statement.
    zero = pre["zero_draw"]
    assert zero["draws"] == 0
    assert zero["no_fitting"] is True
    assert zero["optimizer_invoked"] is False
    block = zero["reserved_audit_block"]
    assert block["fired"] is False
    assert block["master_seed"] == 20260920
    assert block["raw_seeds"] == [20301040, 20301055]
    # The pre-registered bounds and the verdict contract.
    bounds = pre["new_spread_bounds"]
    assert bounds["bin_spread_p95_bound"] == 2.0
    assert bounds["feeddown_spread_max_bound"] == 3.0
    assert bounds["kplus_p95_anchor"]["production"] == KPLUS_P95_ANCHOR
    assert pre["verdict_contract"]["criteria"] == ["a", "b", "c"]
    # The substrate digests (the frozen substrate, as on disk).
    substrate = pre["substrate"]
    assert substrate["families_config_sha256"] == FAMILIES_SHA256
    assert substrate["catalog_sha256"] == CATALOG_SHA256
    assert substrate["card6_surface_sha256"] == CARD6_SHA256
    assert substrate["builder_module_sha256"] == BUILDER_SHA256
    assert substrate["r32_driver_nested_sha256"] == (
        substrate["r32_driver_outer_sha256"]
    )
    # The sealed archives verified before any measurement.
    verified = {
        entry["path"]: entry["entries"]
        for entry in pre["sealed_run_dirs_verified"]
    }
    assert verified == {
        "dev/00runs/r32_spectral_systematic_pp_2026-09-18": 29,
        "dev/00runs/r32_spectral_systematic_pp_confirmation_2026-09-19":
            55,
        "dev/00runs/r30_spectral_normalization_2026-09-17": 3,
        "dev/00runs/r31_catalog_continuum_matching_scan_2026-09-17": 79,
        "dev/00runs/r4_audit_transition_routing_formulation_2026-09-20":
            6,
    }
    assert all(
        entry["verified"] is True
        for entry in pre["sealed_run_dirs_verified"]
    )
    # The provenance: the dual-tree driver digest recorded at run
    # time equals the on-disk dual-tree digest (the byte-identity
    # cross-check).
    assert _sha256_file(run_dir / "provenance.json") == (
        SEAL_FILES["provenance.json"][0]
    )
    provenance = _read_json(run_dir / "provenance.json")
    assert provenance["card"] == CARD_ID
    assert provenance["card_chain"] == [
        "TASK-20260920-r3-gate3-decision",
        "TASK-20260920-round4-index",
        CARD_ID,
    ]
    assert provenance["driver_sha256"]["nested"] == AUDIT_DRIVER_SHA256
    assert provenance["driver_sha256"]["outer"] == AUDIT_DRIVER_SHA256
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
        NESTED_ROOT / "tests" / "test_r4_audit_strange_sector_contract_pins.py"
    ), "the manifest does not pin this register at its on-disk digest"
    # The audit's own entries (nested + outer driver, the solver's
    # test) stay at their on-disk digests.
    for rel in (
        "dev/mixing2006/helpers/benchmarking/"
        "audit_r4_strange_sector_families.py",
        "helpers/benchmarking/audit_r4_strange_sector_families.py",
        "dev/mixing2006/tests/test_audit_r4_strange_sector_families.py",
    ):
        assert inventory.get(rel) == _sha256_file(MONO_ROOT / rel), rel
    assert inventory[
        "dev/mixing2006/helpers/benchmarking/"
        "audit_r4_strange_sector_families.py"
    ] == inventory[
        "helpers/benchmarking/audit_r4_strange_sector_families.py"
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
        "audit_r4_strange_sector_families.py",
        "helpers/benchmarking/audit_r4_strange_sector_families.py",
        "dev/mixing2006/tests/test_audit_r4_strange_sector_families.py",
        MY_TEST_REL,
        "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json",
    ):
        assert rel in files, f"the freeze exception names {rel}"
    # The pinned M8 map stays intact (the freeze test enforces the
    # full inventory).
    assert manifest["frozen_fit_outputs"]["m8_frozen_pp_parameters"][
        "continuum.match_mass"
    ] == 2.680194110727528
