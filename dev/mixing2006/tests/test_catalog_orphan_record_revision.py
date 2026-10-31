"""The orphan-record revision of the committed catalog (2026-10-29;
TASK-20261029-catalog-orphan-etaprime): the Round 2 eta'(958) under the
card-pinned key 377, which no class and no channel reaches, is removed; the
revision is idempotent on the committed catalog, refuses a payload in which a
channel names the record, and the kernel with the eta' option holds one eta',
the record 331."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from na49_mixing.eta_prime import eta_prime_catalog, load_eta_prime
from na49_mixing.hadron_catalog import HadronCatalog

NESTED_ROOT = Path(__file__).resolve().parents[1]
CATALOG = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
SCRIPT = NESTED_ROOT / "helpers" / "benchmarking" / "catalog_orphan_record_revision.py"
MIRROR = NESTED_ROOT.parents[1] / "helpers" / "benchmarking" / SCRIPT.name


@pytest.fixture(scope="module")
def revision():
    spec = importlib.util.spec_from_file_location("catalog_orphan_record_revision", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _orphan_record() -> dict:
    """The Round 2 record as the import wrote it, reduced to its first channel."""
    return {"pdg_id": 377, "name": "eta-prime(958)", "B": 0, "Q": 0, "S": 0, "isospin": 0.0, "isospin3": 0,
            "spin": 0.0, "parity": "-", "pole_mass": 0.95778, "nominal_width": 0.000188,
            "mass_range": [0.5502, 0.95872], "stable_for_decay": False, "provenance": None,
            "decay_channels": [{"parent_pdg": 377, "daughters": [211, -211, 221], "branching_fraction": 0.425,
                                "orbital_angular_momentum": 0, "is_three_body": True, "provenance": None}]}


def test_the_committed_catalog_holds_no_orphan(revision):
    catalog = HadronCatalog.from_json(CATALOG)
    ids = {int(r.pdg_id) for r in catalog.records()}
    assert not ids & set(revision.ORPHAN_RECORDS)
    assert not any(set(int(d) for d in c.daughters) & set(revision.ORPHAN_RECORDS)
                   for r in catalog.records() for c in r.decay_channels)
    assert not catalog.validate().errors


def test_the_revision_is_idempotent(revision):
    payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    assert revision.apply_revision(payload) == payload


def test_the_revision_removes_the_record_and_nothing_else(revision):
    payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    before = copy.deepcopy(payload)
    before["records"].insert(40, _orphan_record())
    assert revision.apply_revision(before) == payload


def test_a_channel_into_the_orphan_is_refused(revision):
    payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    payload["records"].append(_orphan_record())
    parent = next(r for r in payload["records"] if r["decay_channels"])
    parent["decay_channels"][0]["daughters"] = [377, 111]
    with pytest.raises(ValueError, match="decays into"):
        revision.apply_revision(payload)


def test_a_renamed_key_is_refused(revision):
    payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    payload["records"].append({**_orphan_record(), "name": "another state"})
    with pytest.raises(ValueError, match="not 'eta-prime"):
        revision.apply_revision(payload)


def test_the_kernel_eta_prime_is_one_record():
    """With the option the catalog holds one 0- isoscalar at the eta' mass, the record 331."""
    catalog = eta_prime_catalog(HadronCatalog.from_json(CATALOG), load_eta_prime(NESTED_ROOT))
    found = [int(r.pdg_id) for r in catalog.records()
             if (r.B, r.Q, r.S) == (0, 0, 0) and r.spin == 0.0 and r.parity == "-" and abs(r.pole_mass - 0.958) < 0.01]
    assert found == [331]


def test_the_monorepo_mirror_is_byte_identical():
    if not MIRROR.exists():
        pytest.skip("standalone tree: the monorepo mirror is absent")
    assert MIRROR.read_bytes() == SCRIPT.read_bytes()
