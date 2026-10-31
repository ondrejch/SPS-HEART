"""The dipion isospin revision of the committed catalog (2026-10-21;
TASK-20261021-catalog-dipion-isospin, the Gemini review of R104, gap B4 item
3): the Lambda(1520) Lambda pi pi carries pi+ pi- : pi0 pi0 = 2 : 1 at the
unchanged mode total, the revision is idempotent on the committed catalog,
and every three-body mode of the committed catalog carries its three-body
isospin shares except the documented departures."""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import pytest

from na49_mixing.cascade_decay import _flavor_i3x2, _isospin_of, _multiplet_key
from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.resonance_decay import multiplet_charge_states
from na49_mixing.three_body_endings import three_body_isospin_share

NESTED_ROOT = Path(__file__).resolve().parents[1]
CATALOG = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"
SCRIPT = NESTED_ROOT / "helpers" / "benchmarking" / "catalog_dipion_isospin_revision.py"

#: The three-body modes whose charge pattern departs from the generic
#: sequential isospin share by construction: the eta's G-parity-violating
#: 3 pi (measured fractions), the f2(1270) 4 pi in the rho rho charge
#: pattern, the K1(1270) K pi pi through the K0*(1430) pi chain, the
#: K2*(1430) K*(892) pi pi with the I = 0 dipion (the PDG 2024 meson
#: tables' rule), and the rho3(1690) K Kbar pi, a record no class or decay
#: reaches (the orphan eta' duplicate 377 left the catalog in the
#: orphan-record revision of 2026-10-29).
DOCUMENTED = {221, 225, 10323, 10313, 325, 315, 117, 217}


@pytest.fixture(scope="module")
def revision():
    spec = importlib.util.spec_from_file_location("catalog_dipion_isospin_revision", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_lambda1520_dipion():
    catalog = HadronCatalog.from_json(CATALOG)
    modes = {tuple(sorted(c.daughters)): c.branching_fraction for c in catalog.get(102134).decay_channels if c.is_three_body}
    charged, neutral = modes[(-211, 211, 3122)], modes[(111, 111, 3122)]
    assert math.isclose(charged / neutral, 2.0, rel_tol=1e-12)
    assert math.isclose(charged + neutral, 0.10309278350515465, rel_tol=1e-12)
    assert math.isclose(math.fsum(c.branching_fraction for c in catalog.get(102134).decay_channels), 1.0, rel_tol=1e-12)
    assert not catalog.validate().errors


def test_the_revision_is_idempotent(revision):
    payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    assert revision.apply_revision(payload) == payload


def test_every_three_body_mode_carries_its_isospin_shares():
    catalog = HadronCatalog.from_json(CATALOG)
    bad = []
    for rec in catalog.records():
        pdg = int(rec.pdg_id)
        modes = [c for c in rec.decay_channels if c.is_three_body]
        if not modes or abs(pdg) in DOCUMENTED:
            continue
        parent_i, parent_i3x2 = float(_isospin_of(pdg, catalog)), int(_flavor_i3x2(pdg, catalog))
        groups = defaultdict(dict)
        for c in modes:
            groups[tuple(sorted(str(_multiplet_key(d, catalog)) for d in c.daughters))][tuple(sorted(c.daughters))] = c.branching_fraction
        for have in groups.values():
            expected = {}
            for combo in itertools.product(*(multiplet_charge_states(int(d), catalog) for d in next(iter(have)))):
                key = tuple(sorted(int(d) for d in combo))
                if key in expected or sum(int(_flavor_i3x2(int(d), catalog)) for d in combo) != parent_i3x2:
                    continue
                w = three_body_isospin_share(parent_i, parent_i3x2, tuple(int(d) for d in combo), catalog)
                if w > 1e-12:
                    expected[key] = w
            total, norm = math.fsum(have.values()), math.fsum(expected.values())
            if set(expected) != set(have) or any(abs(have[k] / total - expected[k] / norm) > 1e-9 for k in expected):
                bad.append((rec.name, sorted(have), sorted(expected)))
    assert not bad, bad
