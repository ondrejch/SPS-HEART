"""Independent complementary coverage for the mesonic continuum
spectral family (Round 3, card 5).

Round 3, card 5 (``TASK-20260917-r3-mesonic-continuum-spectral-family``;
plan section 22 task 19, plan sections 9.1-9.2) extended
``na49_mixing/spectral_density.py`` with the ``active_families``
selector, the ``family_draw`` active guard, and the mesonic sector of
``config/v2_singlet/continuum_resonances.json`` (the activated neutral
family ``(0, 0, 0)`` and the four declared-inactive charged and strange
families), with the solver's card tests in
``tests/test_spectral_density_mesonic.py``.  This file is the
test-author's independent complement: it pins the same committed
behavior from a different angle, without touching the solver's file and
without duplicating its assertions.  The coverage added here:

- ``test_active_families_selector_contract``: the ``active_families``
  contract over the qn values of all seven loaded families (the
  activated set is exactly baryonic_np, baryonic_strange, and
  mesonic_neutral; every declared mesonic qn selects nothing), an
  empty result for qn values with no loaded family at all and for the
  empty family list, and the input-list contract (no mutation of the
  input, the result a fresh list whose elements are the same family
  objects, file order preserved, a list-valued qn accepted by the
  ``int`` conversion of the signature).
- ``test_active_guard_named_value_error_per_declared_family``: for
  each of the four declared families ``family_draw`` raises
  ``ValueError`` whose message is pinned exactly (naming the family
  id and its qn, the "is not active" clause, the declared-not-sampled
  rationale), the guard fires before the guard's call consumes any
  uniform of the handed RNG (two-uniform budget untouched), the guard
  message is distinct from the loader and validation message families,
  and the active families draw without raising.
- ``test_config_integrity_seven_families_pinned_parameters``: the
  committed config carries exactly seven family records in stable
  order with the full 13-key record form, every record stamped
  ``continuum_resonance``; the three activated records and the four
  declared records each pin the pre-registered parameter values
  (``b_power = 0``, ``t_h = 0.165``, ``m_min = 2.680194110727528``,
  ``ramp_width = 0.5``, ``nominal_width = 0.5``, ``m_max = 5.0``, and
  the exact ``n_norm = 4.188698227099897e-13``), the ``active`` flag as
  a strict boolean, and the loaded records cross-checked field by
  field against the raw JSON.
- ``test_mesonic_class_pairs_conserve_family_qn_raw_catalog_audit``:
  all 48 class pairs of the five mesonic families (18 neutral, 8
  charged, 8 anticharged, 7 strange, 7 antistrange) conserve the
  family ``(B, Q, S)`` at the vertex, audited against
  ``config/v2_singlet/resonance_catalog.json`` read as raw JSON
  (independent of the ``HadronCatalog`` layer the solver's test
  uses); the per-family class counts and the total are pinned; the
  charge-conjugate symmetry holds (the anticharged and antistrange
  class sets are the charge-conjugate images of the charged and
  strange sets under the catalog's conjugation map, the conjugate of a
  recorded daughter being its antiparticle record where the catalog
  records it and the daughter itself where it does not; the neutral
  set is closed under the same map up to daughter order).
- ``test_baryonic_section_byte_stable_against_card4_commit``: the raw
  file-byte span of each of the two baryonic family records in the
  committed config hashes to the digest recomputed from card 4's
  committed bytes (``git show e08ed99:config/v2_singlet/
  continuum_resonances.json``), with the baryonic records remaining
  the first two entries of the families array.
- ``test_neutral_mesonic_edge_values_and_draw_budget``: the neutral
  mesonic density is exactly 0 at and below ``m_min`` (a probe set
  extending below zero mass), equals the independently recomputed
  ramp-top value at ``m_min + ramp_width`` and the constant-branch
  value at ``m_max`` (recomputed from the raw config values, not the
  loaded record), and 10,000 draws at a fixed probe seed (disjoint from
  the card's master seed 20260917 and the card-4 probe seed 20260918)
  all lie in ``[m_min, m_max]`` with the two-uniform budget verified
  at cumulative checkpoints (1, 10, 100, 1,000, 10,000) and bit-
  identical reproducibility.
- ``test_loaded_family_records_are_frozen``: every loaded family record
  is a frozen value object (attribute mutation raises
  ``dataclasses.FrozenInstanceError``), the record contract of the
  module's docstring.

The baryonic span digests were recomputed in this work from card 4's
committed config through the nested repository (``git show e08ed99:
config/v2_singlet/continuum_resonances.json``) and cross-checked
against the values pinned by the solver's test file
``tests/test_spectral_density_mesonic.py``; the test itself is
hermetic (no git access at run time).

Hermetic: the committed config and catalog files, numpy, and the
standard library only; no event generation, no external tables.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.spectral_density import (
    CONTINUUM_STATE_TYPE,
    REFERENCE_GRID_POINTS,
    SUPPORT_M_MAX,
    ContinuumFamily,
    active_families,
    family_density,
    family_draw,
    families_from_json,
)

TEST_FILE = Path(__file__).resolve()
NESTED_ROOT = TEST_FILE.parent.parent  # dev/mixing2006
CONFIG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "continuum_resonances.json"
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"

#: The committed family records in file order: the card-4 baryonic
#: section (byte-stable) followed by the card-5 mesonic section (the
#: activated neutral family, then the four declared-inactive families).
EXPECTED_FAMILY_IDS = (
    "baryonic_np",
    "baryonic_strange",
    "mesonic_neutral",
    "mesonic_charged",
    "mesonic_anticharged",
    "mesonic_strange",
    "mesonic_antistrange",
)

#: The family quantum numbers ``(B, Q, S)`` of each committed record.
EXPECTED_QNS = {
    "baryonic_np": (1, 1, 0),
    "baryonic_strange": (1, 0, -1),
    "mesonic_neutral": (0, 0, 0),
    "mesonic_charged": (0, 1, 0),
    "mesonic_anticharged": (0, -1, 0),
    "mesonic_strange": (0, 0, -1),
    "mesonic_antistrange": (0, 0, 1),
}

#: The activated families of Round 3 (index items 1 and 9): the two
#: card-4 baryonic families and the neutral mesonic family.
ACTIVATED_FAMILY_IDS = (
    "baryonic_np",
    "baryonic_strange",
    "mesonic_neutral",
)

#: The declared, inactive mesonic families (round 3 index item 9).
DECLARED_FAMILY_IDS = (
    "mesonic_charged",
    "mesonic_anticharged",
    "mesonic_strange",
    "mesonic_antistrange",
)

#: The per-family allowed-class counts of the five mesonic families in
#: config order (18 neutral + 8 charged + 8 anticharged + 7 strange +
#: 7 antistrange = 48 class pairs audited below; the Round 6 classes
#: of 2026-09-28 added 8 neutral and 2 + 2 charged pairs).
MESONIC_CLASS_COUNTS = (41, 8, 8, 7, 7)  # Round 6 classes added 2026-09-28; Block B stage 1 classes 2026-09-29

#: The exact record form of every family in the committed config
#: (the module's required-key set, pinned independently).
REQUIRED_RECORD_KEYS = frozenset({
    "family_id", "qn", "state_type", "m_min", "t_h", "b_power",
    "ramp_width", "nominal_width", "m_max", "n_norm",
    "allowed_decay_classes", "active", "field_provenance",
})

#: The pre-registered parameter values shared by every family (index
#: items 2-4, minimal choice): the exponential shape, the M8-frozen
#: minimum mass, the 0.5 GeV smoothstep ramp and nominal width, the
#: 5.0 GeV support upper end, and the exact unit normalization
#: ``N_a`` derived on the fixed 10,001-node reference quadrature.
PRE_REGISTERED_PARAMETERS = {
    "b_power": 0.0,
    "t_h": 0.165,
    "m_min": 2.680194110727528,
    "ramp_width": 0.5,
    "nominal_width": 0.5,
    "m_max": 5.0,
    "n_norm": 4.188698227099897e-13,
}

#: The probe seed of this file: disjoint from the card's master seed
#: 20260917 (tests/test_spectral_density_mesonic.py) and the card-4
#: probe seed 20260918 (tests/test_spectral_density_independent.py),
#: so the three test files draw from disjoint bit streams.
PROBE_SEED = 20260919

#: Cumulative draw-budget checkpoints of the 10,000-draw audit: after
#: the n-th draw the proxy RNG has consumed exactly 2n uniforms, so a
#: hidden rejection loop cannot hide in the running total.
DRAW_CHECKPOINTS = {1, 10, 100, 1_000, 10_000}

#: The byte-span digests of the two baryonic family records, recomputed
#: from card 4's committed config (nested commit e08ed99) via
#: ``git show e08ed99:config/v2_singlet/continuum_resonances.json``
#: and cross-checked against the values pinned by the solver's test
#: file tests/test_spectral_density_mesonic.py.  The card-5 re-pin
#: absorbs the config extension; this pin holds the baryonic section
#: to card 4's exact committed bytes.
BARYONIC_SPAN_SHA256 = {
    "baryonic_np": "ef02a19490ab14ecac388cba39898c80a687eda6bec322b6357f03e9d5a0eb71",
    "baryonic_strange": (
        "c805833b43127705ffb571f9e2433d290990361db4bae4daa60787fde81a4bff"
    ),
}

#: The exact message the ``family_draw`` active guard raises for an
#: inactive family (the committed guard text of the card-5 extension).
GUARD_MESSAGE_TEMPLATE = (
    "family {family_id} (qn {qn}) is not active: inactive families are "
    "declared, not sampled (the card-5 active flag controls continuum "
    "sampling)"
)

#: Substrings of the loader and validation message families (the
#: distinct named failures pinned by the card-4 test files): the guard
#: message must not overlap any of them.
LOADER_MESSAGE_MARKERS = (
    "missing required keys",
    "must be a 3-entry list",
    "state_type must be",
    "does not equal the pre-registered support upper end",
    "must be positive",
    "must be non-negative",
    "lies beyond the support upper end",
    "allowed_decay_classes must be",
    "carries duplicate",
    "field_provenance must be",
    "does not reproduce the unit normalization",
    "carries no non-empty",
)

_CLASS_ID_PATTERN = re.compile(r"\((-?\d+),(-?\d+)\)")


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def config_raw() -> dict:
    """The raw committed config payload (independent of the loader)."""
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def catalog_raw() -> dict:
    """The raw committed catalog payload (independent of
    ``HadronCatalog``)."""
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def families() -> list[ContinuumFamily]:
    """The committed families, loaded through the public loader (the
    card-4 baryonic pair followed by the card-5 mesonic families, the
    activated and the declared-inactive records alike)."""
    return families_from_json(str(CONFIG_PATH))


def _neutral_family(families: list[ContinuumFamily]) -> ContinuumFamily:
    (family,) = [f for f in families if f.family_id == "mesonic_neutral"]
    return family


# ----------------------------------------------------------------------
# active_families contract
# ----------------------------------------------------------------------
def test_active_families_selector_contract(families, config_raw):
    """The ``active_families`` selector returns exactly the activated
    family for each loaded qn, nothing for the declared mesonic qns and
    for qn values with no loaded family, and leaves its input list
    untouched (a fresh result list carrying the same family objects in
    file order)."""
    assert tuple(f.family_id for f in families) == EXPECTED_FAMILY_IDS

    # The selector over the qn of every loaded family: exactly the
    # expected active set (at most one active family per qn, so the
    # result is exactly [the activated family] or []).
    for record in config_raw["families"]:
        qn = tuple(record["qn"])
        expected = [
            f.family_id
            for f in families
            if f.qn == qn and f.active
        ]
        result = active_families(families, qn)
        assert [f.family_id for f in result] == expected, (
            f"qn {qn}: active_families returned "
            f"{[f.family_id for f in result]!r}, expected {expected!r}"
        )

    # The activated set is exactly the three Round 3 families.
    for family_id in ACTIVATED_FAMILY_IDS:
        qn = EXPECTED_QNS[family_id]
        (selected,) = active_families(families, qn)
        assert selected.family_id == family_id
        # The result carries the same family object from the input.
        assert any(f is selected for f in families)
    for family_id in DECLARED_FAMILY_IDS:
        qn = EXPECTED_QNS[family_id]
        assert active_families(families, qn) == [], (
            f"{family_id}: the declared mesonic qn {qn} must select "
            "no active family (declared, not activated)"
        )

    # qn values with no loaded family at all: empty result.
    for qn in ((2, 1, 0), (0, 2, 0), (1, 1, 1), (0, 0, 2), (0, 1, 1)):
        assert qn not in EXPECTED_QNS.values()
        assert active_families(families, qn) == []
    # The empty family list: empty result.
    assert active_families([], (0, 0, 0)) == []

    # Input-list contract: no mutation, file order preserved, the
    # result a fresh list (not aliased to the input) whose elements
    # are the same objects.
    snapshot = list(families)
    result = active_families(families, (0, 0, 0))
    assert list(families) == snapshot, (
        "active_families must not mutate the input list"
    )
    for original, current in zip(snapshot, families):
        assert original is current, (
            "active_families must not replace objects of the input list"
        )
    assert result is not families
    before = len(families)
    result.append(families[0])
    assert len(families) == before, (
        "the result list must be a fresh list, not an alias of the input"
    )
    assert [f for f in active_families(families, (1, 1, 0))][0] is families[0], (
        "the selector must preserve the input's file order"
    )

    # A list-valued qn is accepted (the signature converts every entry
    # with int).
    assert [
        f.family_id
        for f in active_families(families, [0, 0, 0])  # type: ignore[arg-type]
    ] == ["mesonic_neutral"]


# ----------------------------------------------------------------------
# Active guard
# ----------------------------------------------------------------------
class _CountingGenerator:
    """A proxy RNG wrapper counting uniform draws, so the guard's
    zero-consumption contract is auditable."""

    def __init__(self, seed: int) -> None:
        self._generator = np.random.default_rng(seed)
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        return float(self._generator.random())


def test_active_guard_named_value_error_per_declared_family(families):
    """``family_draw`` raises ``ValueError`` for each of the four
    declared families, with the exact committed guard message (naming
    the family id and its qn), the guard fires before any uniform of
    the handed RNG is consumed, and the guard message is distinct from
    the loader and validation message families."""
    for family_id in DECLARED_FAMILY_IDS:
        (family,) = [f for f in families if f.family_id == family_id]
        assert family.active is False
        proxy = _CountingGenerator(PROBE_SEED)
        with pytest.raises(ValueError) as excinfo:
            family_draw(family, proxy)  # type: ignore[arg-type]
        message = str(excinfo.value)
        assert family.family_id in message, (
            f"the guard message must name the family, got: {message!r}"
        )
        assert message == GUARD_MESSAGE_TEMPLATE.format(
            family_id=family.family_id, qn=tuple(family.qn)
        ), (
            f"family {family.family_id}: the guard message is not the "
            f"committed text, got: {message!r}"
        )
        # The guard fires before the two-uniform budget is touched.
        assert proxy.calls == 0, (
            f"family {family.family_id}: the guard consumed "
            f"{proxy.calls} uniforms before raising"
        )
        # The guard message is distinct from the loader/validation
        # message families.
        for marker in LOADER_MESSAGE_MARKERS:
            assert marker not in message, (
                f"family {family.family_id}: the guard message overlaps "
                f"the loader/validation marker {marker!r}"
            )
    # The activated families draw without raising (the guard does not
    # over-trigger).
    for family_id in ACTIVATED_FAMILY_IDS:
        (family,) = [f for f in families if f.family_id == family_id]
        assert family.active is True
        mass = family_draw(family, np.random.default_rng(PROBE_SEED))
        assert family.m_min <= mass <= SUPPORT_M_MAX


# ----------------------------------------------------------------------
# Config integrity
# ----------------------------------------------------------------------
def test_config_integrity_seven_families_pinned_parameters(
    config_raw, families
):
    """The committed config carries exactly seven family records in
    stable order, every record in the full 13-key form stamped
    ``continuum_resonance``; the three activated records and the four
    declared records pin the pre-registered parameter values (including
    the exact ``n_norm``), the ``active`` flag as a strict boolean, and
    the loaded records cross-checked field by field against the raw
    JSON."""
    records = config_raw["families"]
    assert isinstance(records, list)
    assert [r["family_id"] for r in records] == list(EXPECTED_FAMILY_IDS), (
        "the committed config must load with exactly the seven "
        "families in stable file order"
    )
    assert len({tuple(r["qn"]) for r in records}) == 7, (
        "the seven families must carry distinct qn values"
    )
    for record in records:
        assert set(record.keys()) == REQUIRED_RECORD_KEYS, (
            f"family {record['family_id']}: the record form must be "
            f"exactly the 13-key set, got {sorted(record.keys())}"
        )
        assert record["state_type"] == CONTINUUM_STATE_TYPE
        for field, value in PRE_REGISTERED_PARAMETERS.items():
            assert record[field] == value, (
                f"family {record['family_id']}: pre-registered "
                f"{field} = {value!r}, recorded {record[field]!r}"
            )
        assert record["qn"] == list(EXPECTED_QNS[record["family_id"]])
    # The activated families: strict boolean active = True.
    for record in records:
        expected_active = record["family_id"] in ACTIVATED_FAMILY_IDS
        assert record["active"] is expected_active, (
            f"family {record['family_id']}: the active flag must be the "
            f"strict boolean {expected_active!r}, got "
            f"{record['active']!r}"
        )
    # The declared families carry the same parameter form as the
    # activated ones (identical density parameters, so the identical
    # recorded normalization).
    for record in records:
        if record["family_id"] in ACTIVATED_FAMILY_IDS:
            continue
        for field, value in PRE_REGISTERED_PARAMETERS.items():
            assert record[field] == value
    # Field-by-field cross-check of the loaded records against the raw
    # JSON (the loader must not rewrite any record field).
    for record, family in zip(records, families):
        assert family.family_id == record["family_id"]
        assert family.qn == tuple(record["qn"])
        assert family.m_min == record["m_min"]
        assert family.t_h == record["t_h"]
        assert family.b_power == record["b_power"]
        assert family.n_norm == record["n_norm"]
        assert family.ramp_width == record["ramp_width"]
        assert family.nominal_width == record["nominal_width"]
        assert family.active is record["active"]
        assert family.allowed_decay_classes == tuple(
            record["allowed_decay_classes"]
        )


# ----------------------------------------------------------------------
# Raw-JSON catalog audit
# ----------------------------------------------------------------------
def _class_daughters(class_id: str) -> tuple[int, int]:
    """Parse one multiplet-class identifier into its daughter pair."""
    match = _CLASS_ID_PATTERN.fullmatch(class_id)
    assert match is not None, (
        f"class identifier {class_id!r} is not a catalog multiplet-class "
        "key '(daughter0,daughter1)'"
    )
    return int(match.group(1)), int(match.group(2))


def test_mesonic_class_pairs_conserve_family_qn_raw_catalog_audit(
    config_raw, catalog_raw
):
    """All 48 class pairs of the five mesonic families conserve the
    family ``(B, Q, S)`` at the vertex, audited against the committed
    catalog read as raw JSON (independent of the ``HadronCatalog``
    layer the solver's test uses), with the per-family class counts
    pinned and the charge-conjugate symmetry verified: the anticharged
    and antistrange class sets are the charge-conjugate images of the
    charged and strange sets, and the neutral set is closed under the
    same map (up to daughter order, as the catalog records the
    self-conjugate pairs once)."""
    records_by_pdg = {r["pdg_id"]: r for r in catalog_raw["records"]}
    mesonic_records = [
        r for r in config_raw["families"]
        if r["family_id"].startswith("mesonic_")
    ]
    assert [r["family_id"] for r in mesonic_records] == [
        "mesonic_neutral",
        "mesonic_charged",
        "mesonic_anticharged",
        "mesonic_strange",
        "mesonic_antistrange",
    ]
    assert tuple(len(r["allowed_decay_classes"])
                 for r in mesonic_records) == MESONIC_CLASS_COUNTS
    total_pairs = sum(len(r["allowed_decay_classes"])
                      for r in mesonic_records)
    assert total_pairs == 71  # 48 + 23 Block B stage 1 classes (2026-09-29)

    # The catalog's charge-conjugation map, read from the raw records:
    # the conjugate of a recorded daughter is its antiparticle record
    # where the catalog records one, the daughter itself where it does
    # not (the self-conjugate states pi0, omega, eta, K0S, K0L).
    def conjugate(pdgid: int) -> int:
        return -pdgid if -pdgid in records_by_pdg else pdgid

    classes_by_family: dict[str, list[tuple[int, int]]] = {}
    for record in mesonic_records:
        family_qn = tuple(record["qn"])
        pairs = []
        for class_id in record["allowed_decay_classes"]:
            daughter0, daughter1 = _class_daughters(class_id)
            assert daughter0 in records_by_pdg, (
                f"family {record['family_id']}: daughter {daughter0} of "
                f"class {class_id} is not a catalog record"
            )
            assert daughter1 in records_by_pdg, (
                f"family {record['family_id']}: daughter {daughter1} of "
                f"class {class_id} is not a catalog record"
            )
            rec0 = records_by_pdg[daughter0]
            rec1 = records_by_pdg[daughter1]
            vertex = (
                int(rec0["B"]) + int(rec1["B"]),
                int(rec0["Q"]) + int(rec1["Q"]),
                int(rec0["S"]) + int(rec1["S"]),
            )
            assert vertex == family_qn, (
                f"family {record['family_id']}: class {class_id} "
                f"({rec0['name']} + {rec1['name']}) carries vertex "
                f"(B, Q, S) = {vertex}, family requires {family_qn}"
            )
            pairs.append((daughter0, daughter1))
        classes_by_family[record["family_id"]] = pairs

    # Charge-conjugate symmetry: the anticharged set is the conjugate
    # image of the charged set (and vice versa), the antistrange set
    # the conjugate image of the strange set, the conjugation
    # applied per daughter.
    for charged, anticharged in (
        ("mesonic_charged", "mesonic_anticharged"),
        ("mesonic_strange", "mesonic_antistrange"),
    ):
        image = {
            (conjugate(d0), conjugate(d1))
            for d0, d1 in classes_by_family[charged]
        }
        assert image == set(classes_by_family[anticharged]), (
            f"{anticharged}: the class set is not the charge-conjugate "
            f"image of {charged}: conjugate image {sorted(image)!r} vs "
            f"recorded {sorted(classes_by_family[anticharged])!r}"
        )
    # The neutral set is closed under the conjugation map up to
    # daughter order (self-conjugate pairs recorded once): the
    # unordered conjugate image equals the unordered neutral set.
    neutral = classes_by_family["mesonic_neutral"]

    def unordered(pair: tuple[int, int]) -> tuple[int, int]:
        a, b = pair
        return (a, b) if a <= b else (b, a)

    neutral_unordered = {unordered(p) for p in neutral}
    conjugate_unordered = {
        unordered((conjugate(d0), conjugate(d1))) for d0, d1 in neutral
    }
    assert conjugate_unordered == neutral_unordered, (
        "mesonic_neutral: the class set is not closed under the "
        f"catalog's charge-conjugation map: {sorted(conjugate_unordered)!r} "
        f"vs {sorted(neutral_unordered)!r}"
    )


# ----------------------------------------------------------------------
# Baryonic byte stability (independent of card 4's committed bytes)
# ----------------------------------------------------------------------
def _family_record_spans(path: Path) -> tuple[dict[str, bytes], list[str]]:
    """The raw byte span of each family record object in the config,
    and the record order.

    The records are located with ``json.JSONDecoder.raw_decode`` over
    the ``families`` array, so the returned bytes are the exact file
    bytes of each record object (the file is ASCII, so character and
    byte offsets coincide).
    """
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    assert len(raw) == len(text), "the config must be ASCII"
    decoder = json.JSONDecoder()
    key_index = text.index('"families"')
    array_index = text.index("[", key_index)
    pos = array_index + 1
    spans: dict[str, bytes] = {}
    order: list[str] = []
    while True:
        while text[pos] in " \t\r\n,":
            pos += 1
        if text[pos] == "]":
            break
        start = pos
        obj, end = decoder.raw_decode(text, pos)
        spans[obj["family_id"]] = raw[start:end]
        order.append(obj["family_id"])
        pos = end
    return spans, order


def test_baryonic_section_byte_stable_against_card4_commit():
    """The raw file-byte span of each baryonic family record in the
    committed config hashes to the digest recomputed from card 4's
    committed bytes (nested commit e08ed99, via ``git show e08ed99:
    config/v2_singlet/continuum_resonances.json`` in this work), and
    the baryonic records remain the first two entries of the families
    array in card 4's order."""
    spans, order = _family_record_spans(CONFIG_PATH)
    assert order[:2] == ["baryonic_np", "baryonic_strange"], (
        "the baryonic section must open the families array in card 4's "
        f"order, got first two records {order[:2]!r}"
    )
    for family_id, digest in BARYONIC_SPAN_SHA256.items():
        assert hashlib.sha256(spans[family_id]).hexdigest() == digest, (
            f"the baryonic record {family_id} is not byte-stable "
            f"against card 4's committed config (span digest "
            f"{hashlib.sha256(spans[family_id]).hexdigest()!r} vs the "
            f"card-4 recomputed digest {digest!r})"
        )
    # The mesonic section that follows is exactly the five card-5
    # records (no baryonic record content was reflowed into it).
    assert order[2:] == [
        "mesonic_neutral",
        "mesonic_charged",
        "mesonic_anticharged",
        "mesonic_strange",
        "mesonic_antistrange",
    ]


# ----------------------------------------------------------------------
# Density / draw edge behavior (neutral mesonic family)
# ----------------------------------------------------------------------
def test_neutral_mesonic_edge_values_and_draw_budget(config_raw, families):
    """The neutral mesonic density is exactly 0 at and below ``m_min``
    (the probe set extends to zero and negative mass), equals the
    independently recomputed ramp-top value at ``m_min + ramp_width``
    and the constant-branch value at ``m_max``, and 10,000 draws at
    the fixed probe seed all lie in ``[m_min, m_max]`` with the
    two-uniform budget verified at cumulative checkpoints and bit-
    identical reproducibility."""
    record = next(
        r for r in config_raw["families"] if r["family_id"] == "mesonic_neutral"
    )
    family = _neutral_family(families)
    m_min = float(record["m_min"])
    m_max = float(record["m_max"])
    t_h = float(record["t_h"])
    b_power = float(record["b_power"])
    n_norm = float(record["n_norm"])
    ramp_width = float(record["ramp_width"])

    # Exactly 0 at and below m_min (no continuum strength below the
    # catalog-completeness boundary, plan section 9.3).
    for probe in (m_min, m_min - 1e-12, m_min - 1e-3, m_min - 0.5, 0.0,
                  -5.0):
        assert family_density(family, probe) == 0.0, (
            f"the density must be exactly 0.0 at {probe!r} (at or below "
            f"m_min {m_min!r}), got {family_density(family, probe)!r}"
        )
    # Positive above the threshold (sanity of the zero probe set).
    assert family_density(family, m_min + 1e-9) > 0.0

    # Ramp-top value at m_min + ramp_width, recomputed from the raw
    # config values (the ramp is exactly 1 from there on).
    junction = m_min + ramp_width
    ramp_top = n_norm * (junction ** (-b_power) * math.exp(junction / t_h))
    assert family_density(family, junction) == ramp_top, (
        f"the density at the ramp-top mass {junction!r} is "
        f"{family_density(family, junction)!r}, expected {ramp_top!r}"
    )
    # Constant-branch value at m_max, recomputed from the raw config
    # values.
    constant_branch = n_norm * (m_max ** (-b_power) * math.exp(m_max / t_h))
    assert family_density(family, m_max) == constant_branch, (
        f"the density at m_max {m_max!r} is "
        f"{family_density(family, m_max)!r}, expected {constant_branch!r}"
    )

    # 10,000 draws at the fixed probe seed: support and the two-uniform
    # budget at cumulative checkpoints (1, 10, 100, 1,000, 10,000), so
    # no hidden rejection loop can hide in the running total.
    n_draws = 10_000
    proxy = _CountingGenerator(PROBE_SEED)
    draws = []
    for i in range(n_draws):
        draws.append(family_draw(family, proxy))  # type: ignore[arg-type]
        if (i + 1) in DRAW_CHECKPOINTS:
            assert proxy.calls == 2 * (i + 1), (
                f"after {i + 1} draws the proxy consumed {proxy.calls} "
                f"uniforms, expected {2 * (i + 1)} (exactly two per "
                "draw, the card-3 draw_mass convention)"
            )
    assert proxy.calls == 2 * n_draws
    assert all(m_min <= m <= m_max for m in draws), (
        f"a draw left [{m_min}, {m_max}]"
    )
    # Bit-identical reproducibility at the fixed probe seed: two
    # independent runs on the same bit stream (a fresh generator per
    # run, created before the draw loop).
    stream = np.random.default_rng(PROBE_SEED)
    rerun = [family_draw(family, stream) for _ in range(n_draws)]
    assert rerun == draws, (
        f"draws at seed {PROBE_SEED} are not reproducible"
    )


# ----------------------------------------------------------------------
# Record contract
# ----------------------------------------------------------------------
def test_loaded_family_records_are_frozen(families):
    """Every loaded family record is a frozen value object: attribute
    mutation raises ``dataclasses.FrozenInstanceError`` (the record
    contract of the module's docstring; a draw, a load, and a config
    edit must never silently rewrite a family)."""
    assert len(families) == len(EXPECTED_FAMILY_IDS)
    for family in families:
        assert type(family).__dataclass_params__.frozen
        with pytest.raises(dataclasses.FrozenInstanceError):
            family.family_id = family.family_id  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            family.active = not family.active  # type: ignore[misc]
        # The record still carries its loaded identity afterwards.
        assert family.family_id in EXPECTED_FAMILY_IDS
