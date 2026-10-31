"""Provenance-tagged initial catalog import tests (Round 2, card 2).

Task ``TASK-20260913-r2-provenance-catalog-import`` (plan section 22,
task 10; plan section 8.1 step 1).  Five deterministic tests pin the
``helpers/benchmarking/import_resonance_catalog.py`` driver:

- ``test_record_set_matches_vendored_table``: the driver's required
  record set equals the hand-constructed set
  (vendored unstable established hadrons - placeholder towers - muons -
  cluster - stable states - cascade-terminal ground states) plus
  eta-prime; the stable record set is the pre-registered 11-state
  termination set plus the seven cascade-terminal ground states.
- ``test_write_after_validate``: a transcription fixture with one
  channel violating (B, Q, S) conservation is refused (exit code 2, the
  error names the channel); the catalog file is not written.
- ``test_provenance_completeness``: a record with one missing
  ``ProvenanceTag`` field is refused.
- ``test_stable_termination_set``: the catalog ``stable_states`` list
  equals the pre-registered set (round 2 index, item 11).
- ``test_parse_roundtrip``: the written catalog parses into
  ``HadronCatalog`` with zero validation errors.

Hermetic: the fixtures are hand-constructed (a four-record mini catalog
whose channels only terminate on the pre-registered stable states); the
full import with the real transcription is the run itself.  No event
generation, no external tables.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "helpers", "benchmarking"))

import import_resonance_catalog as import_driver  # noqa: E402
from na49_mixing.hadron_catalog import (  # noqa: E402
    STABLE_STATES,
    HadronCatalog,
)
from na49_mixing.pdg_database import _PDG_TABLE  # noqa: E402

ACCESSED_ON = "2026-09-13"
TRANSCRIBER = "na49-res r2 card 2"

# ----------------------------------------------------------------------
# Hand-constructed expected record sets (no external tables)
# ----------------------------------------------------------------------
EXPECTED_UNSTABLE_IDS = frozenset({
    # light mesons (incl. eta-prime, PDG 331)
    113, 213, -213,               # rho(770)
    100113, 100213, -100213,      # rho(1450)
    30113, 30213, -30213,         # rho(1700)
    223,                          # omega(782)
    100223,                       # omega(1420)
    30223,                        # omega(1650)
    221,                          # eta
    377,                          # eta-prime (card-pinned key; PDG MC id 331)
    10113, 10213, -10213,         # b1(1235)
    20113, 20213, -20213,         # a1(1260)
    115, 215, -215,               # a2(1320)
    225,                          # f2(1270)
    117, 217, -217,               # rho3(1690)
    229,                          # f4(2050)
    # strange mesons
    323, -323, 313, -313,         # K*(892)
    10323, -10323, 10313, -10313, # K1(1270)
    325, -325, 315, -315,         # K2*(1430)
    333,                          # phi(1020)
    # non-strange baryons
    2224, 2214, 2114, 1114,       # Delta(1232)
    -1114, -2114, -2214, -2224,   # anti-Delta(1232)
    202224, 202214, 202114, 201114,   # Delta(1600)
    122224, 122214, 122114, 121114,   # Delta(1700)
    12112, 12212, -12112,         # N*(1440)
    2116, 2216, -2116,            # N*(1520)
    22112, 22212, -22112,         # N*(1680)
    102112, 102212, -102112,      # N(1535)
    122112, 122212, -122112,      # N(1650)
    212114, 212214, -212114, -212214,  # N(1720)
    # strange baryon resonances
    102134,                       # Lambda(1520)
    3224, 3214, 3114,             # Sigma(1385)
})

#: The pre-registered termination set (index item 11) ...
EXPECTED_PRE_REGISTERED_STABLE = frozenset({
    2212, 2112, -2212, -2112,     # p, n, pbar, nbar
    211, -211, 111,               # pi+, pi-, pi0
    321, -321,                    # K+, K-
    310, 130,                     # K0S, K0L
})
#: ... plus the cascade-terminal ground states (weak/EM decays only).
EXPECTED_GROUND_TERMINAL = frozenset({
    311, -311,                    # K0, aK0 flavor eigenstates
    3122, -3122,                  # Lambda, aLambda
    3222, 3212, 3112,             # Sigma+, Sigma0, Sigma-
})
EXPECTED_STABLE_IDS = EXPECTED_PRE_REGISTERED_STABLE | EXPECTED_GROUND_TERMINAL

#: The excluded placeholder towers and non-established objects.
EXPECTED_TOWER_IDS = frozenset({
    9911201, 9912201, 9921201, 9922201, 9931201, 9932201,
    -9911201, -9912201, -9921201, -9922201, -9931201, -9932201,
    9900113, 9900213, 9900313,
})
EXPECTED_NON_HADRON_IDS = frozenset({13, -13, 2219})

ETAPRIME_PDG_ID = 377  # card-pinned key; the PDG MC id of eta'(958) is 331


# ----------------------------------------------------------------------
# Hand-constructed four-record mini fixture (catalog_values structure)
# ----------------------------------------------------------------------
def _tag(field: str, value) -> dict:
    return {
        "document": "test fixture transcription (r2 card 2 deterministic test)",
        "section_or_table": "mini fixture",
        "field": field,
        "accessed_on": ACCESSED_ON,
        "transcribed_value": value,
        "transcriber": TRANSCRIBER,
    }


def _record_entry(pdg_id: int, B: int, Q: int, S: int,
                  isospin: float, isospin3: int,
                  spin: float, parity: str, m_up: float,
                  channels: list[dict]) -> dict:
    vendored = next(p for p in _PDG_TABLE if p.pdg_id == pdg_id)
    return {
        "pdg_id": pdg_id,
        "name": vendored.name,
        "B": B, "Q": Q, "S": S,
        "stable_for_decay": False,
        "values": {
            "pole_mass": _tag("pole_mass", vendored.mass),
            "nominal_width": _tag("nominal_width", vendored.width),
            "mass_range": _tag("mass_range", m_up),
            "spin": _tag("spin", spin),
            "parity": _tag("parity", parity),
            "isospin": _tag("isospin", isospin),
            "isospin3": _tag("isospin3", isospin3),
        },
        "record_provenance": _tag("record", None),
        "channels": channels,
    }


def _channel(daughters, branching: float, L: int,
             three_body: bool = False) -> dict:
    return {
        "daughters": list(daughters),
        "branching_fraction": branching,
        "orbital_angular_momentum": L,
        "is_three_body": three_body,
        "provenance": _tag("branching_fraction", branching),
    }


def _mini_values() -> dict:
    """Four records whose channels terminate on the stable states:
    rho0(770), omega(782), Delta++(1232), phi(1020)."""
    return {
        "card": "test-fixture",
        "recorded_on": ACCESSED_ON,
        "transcriber": TRANSCRIBER,
        "records": [
            _record_entry(113, 0, 0, 0, 1.0, 0, 1.0, "-", 0.77526 + 5 * 0.1491,
                          [_channel((211, -211), 1.0, 1)]),
            _record_entry(223, 0, 0, 0, 0.0, 0, 1.0, "-", 0.78265 + 5 * 0.00849,
                          [_channel((211, -211, 111), 0.983122441, 1, True),
                           _channel((211, -211), 0.016877559, 1)]),
            _record_entry(2224, 1, 2, 0, 1.5, 3, 1.5, "+", 1.232 + 5 * 0.117,
                          [_channel((2212, 211), 1.0, 1)]),
            _record_entry(333, 0, 0, 0, 0.0, 0, 1.0, "-", 1.019461 + 5 * 0.004249,
                          [_channel((321, -321), 0.492, 1),
                           _channel((310, 130), 0.508, 1)]),
        ],
    }


# ----------------------------------------------------------------------
# 1. test_record_set_matches_vendored_table
# ----------------------------------------------------------------------
def test_record_set_matches_vendored_table():
    """The driver's required record set equals the hand-constructed set:
    (vendored unstable ids - placeholder towers - muons - cluster - stable
    states - cascade-terminal ground states) | {eta-prime}, plus the stable
    record set (pre-registered termination set + ground-state endpoints)."""
    assert import_driver.required_unstable_ids() == EXPECTED_UNSTABLE_IDS
    assert import_driver.required_stable_ids() == EXPECTED_STABLE_IDS

    # The vendored-derived formula, cross-checked by hand.
    vendored_unstable = {p.pdg_id for p in _PDG_TABLE if not p.stable}
    derived = (
        vendored_unstable
        - EXPECTED_TOWER_IDS
        - EXPECTED_NON_HADRON_IDS
        - EXPECTED_PRE_REGISTERED_STABLE
        - EXPECTED_GROUND_TERMINAL
    ) | {ETAPRIME_PDG_ID}
    assert derived == EXPECTED_UNSTABLE_IDS

    # The placeholder towers keep their continuum role and never appear.
    assert not (EXPECTED_TOWER_IDS & EXPECTED_UNSTABLE_IDS)
    assert not (EXPECTED_TOWER_IDS & EXPECTED_STABLE_IDS)
    # The pre-registered termination set matches the card-1 STABLE_STATES.
    assert STABLE_STATES == EXPECTED_PRE_REGISTERED_STABLE


# ----------------------------------------------------------------------
# 2. test_write_after_validate
# ----------------------------------------------------------------------
def test_write_after_validate(tmp_path):
    """A catalog_values fixture with one channel violating (B, Q, S)
    conservation is refused (exit code 2, error names the channel); the
    catalog file is not written."""
    values = _mini_values()
    rho_entry = values["records"][0]
    rho_entry["channels"][0]["daughters"] = [211, 211]  # rho0 -> pi+ pi+ : Q violated
    values_path = tmp_path / "catalog_values_bad.json"
    values_path.write_text(json.dumps(values), encoding="utf-8")

    out_path = tmp_path / "catalog.json"
    report_dir = tmp_path / "report"
    code = import_driver.run_import(values, out_path, report_dir,
                                    full_mode=False, test_pass=False)
    assert code == 2
    assert not out_path.exists()
    assert not (report_dir / "catalog_import_report.json").exists()


# ----------------------------------------------------------------------
# 3. test_provenance_completeness
# ----------------------------------------------------------------------
def test_provenance_completeness(tmp_path):
    """A record with one missing ProvenanceTag field is refused."""
    values = _mini_values()
    channel = values["records"][0]["channels"][0]
    del channel["provenance"]["accessed_on"]
    values_path = tmp_path / "catalog_values_prov.json"
    values_path.write_text(json.dumps(values), encoding="utf-8")

    out_path = tmp_path / "catalog.json"
    report_dir = tmp_path / "report_prov"
    code = import_driver.run_import(values, out_path, report_dir,
                                    full_mode=False, test_pass=False)
    assert code == 2
    assert not out_path.exists()


# ----------------------------------------------------------------------
# 4. test_stable_termination_set
# ----------------------------------------------------------------------
def test_stable_termination_set():
    """The catalog ``stable_states`` list equals the pre-registered set
    (round 2 index, item 11); the record-level termination flags cover the
    pre-registered set plus the seven cascade-terminal ground states."""
    assert import_driver.STABLE_HEADER_STATES == (
        2212, 2112, -2212, -2112, 211, -211, 111, 321, -321, 310, 130)
    assert frozenset(import_driver.STABLE_HEADER_STATES) == (
        EXPECTED_PRE_REGISTERED_STABLE)
    assert import_driver.required_stable_ids() == EXPECTED_STABLE_IDS
    assert EXPECTED_GROUND_TERMINAL == {
        311, -311, 3122, -3122, 3222, 3212, 3112}


# ----------------------------------------------------------------------
# 5. test_parse_roundtrip
# ----------------------------------------------------------------------
def test_parse_roundtrip(tmp_path):
    """The written catalog parses into ``HadronCatalog`` with zero
    validation errors."""
    values = _mini_values()
    out_path = tmp_path / "catalog.json"
    report_dir = tmp_path / "report_roundtrip"
    code = import_driver.run_import(values, out_path, report_dir,
                                    full_mode=False, test_pass=False)
    assert code == 0
    assert out_path.is_file()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["catalog_version"] == 1
    assert payload["label"] == import_driver.CATALOG_LABEL
    assert payload["schema_version"] == 1
    assert payload["stable_states"] == list(import_driver.STABLE_HEADER_STATES)
    assert len(payload["records"]) == 4

    catalog = HadronCatalog.from_json(out_path)
    report = catalog.validate()
    assert report.errors == (), report.errors
    # Every record and every channel of the fixture carries a complete
    # ProvenanceTag (the fixture was transcribed with full tags).
    for record in catalog.records():
        assert record.provenance is not None
        assert record.provenance.document and record.provenance.accessed_on
        for channel in record.decay_channels:
            assert channel.provenance is not None
            assert channel.provenance.transcribed_value == (
                channel.branching_fraction)

    # The run artifacts: the staged transcription and the acceptance report.
    staged = json.loads((report_dir / "catalog_values.json").read_text(
        encoding="utf-8"))
    assert len(staged["records"]) == 4
    acceptance = json.loads(
        (report_dir / "catalog_import_report.json").read_text(encoding="utf-8"))
    assert acceptance["record_count"] == 4
    assert acceptance["unstable_record_count"] == 4
    assert acceptance["validation_errors"] == 0
    assert acceptance["eta_prime_added"] is False  # mini fixture carries no eta-prime