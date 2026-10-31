"""The isospin-share revision of the committed catalog (2026-10-05;
TASK-20261005-catalog-delta1700-charge-shares): the Delta(1700)0 N pi split
and the N*+ Delta pi shares carry their Clebsch-Gordan values, the revision
is idempotent on the committed catalog, and the catalog's own closure report
names no Clebsch-Gordan or mirror departure except the physical phi(1020)
K+K- / K0S K0L split."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.resonance_decay import isospin_closure_report

NESTED_ROOT = Path(__file__).resolve().parents[1]
CATALOG = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
SCRIPT = NESTED_ROOT / "helpers" / "benchmarking" / "catalog_isospin_share_revision.py"


@pytest.fixture(scope="module")
def revision():
    spec = importlib.util.spec_from_file_location("catalog_isospin_share_revision", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shares(catalog, pdg, keys):
    chans = {tuple(ch.daughters): float(ch.branching_fraction) for ch in catalog.get(pdg).decay_channels}
    values = {k: chans.get(k, chans.get(k[::-1])) for k in keys}
    total = sum(values.values())
    return {k: v / total for k, v in values.items()}


def test_the_delta1700_0_split():
    catalog = HadronCatalog.from_json(CATALOG)
    shares = _shares(catalog, 122114, [(2212, -211), (2112, 111)])
    assert shares[(2212, -211)] == pytest.approx(1 / 3, abs=1e-8)
    assert shares[(2112, 111)] == pytest.approx(2 / 3, abs=1e-8)
    plus = _shares(catalog, 122214, [(2212, 111), (2112, 211)])
    assert plus[(2212, 111)] == pytest.approx(2 / 3, abs=1e-8)


@pytest.mark.parametrize("pdg", [12212, 2216, 22212, 122212, 212214, -212214])
def test_the_nstar_plus_delta_pi_shares(pdg):
    catalog = HadronCatalog.from_json(CATALOG)
    keys = [(2224, -211), (2214, 111), (2114, 211)]
    if pdg < 0:
        keys = [tuple(d if d == 111 else -d for d in k) for k in keys]
    shares = _shares(catalog, pdg, keys)
    assert [shares[k] for k in keys] == pytest.approx([1 / 2, 1 / 3, 1 / 6], abs=1e-8)


def test_the_revision_is_idempotent(revision):
    payload = json.loads(CATALOG.read_text())
    assert revision.apply_revision(payload) == payload


def test_the_closure_report_is_clean_but_for_phi():
    catalog = HadronCatalog.from_json(CATALOG)
    report = isospin_closure_report(catalog)
    assert report.errors == ()
    departures = [w for w in report.warnings if "Clebsch-Gordan" in w or "mirror mismatch" in w]
    assert departures and all("phi(1020)" in w or "pdg 333" in w for w in departures)
