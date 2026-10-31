"""The verification of the external review of 2026-10-16: the (B, Q, S) of a
catalog daughter comes from its record.  The vendored PDG table lists
neither the f0(500) nor the Lambda(1405) nor the Xi; read through the table
alone, the classes with those daughters found no allowed charge combination
(50 isospin-audit errors on the R98 catalog), and the decayer kept their
tabulated channels without the isospin filter.  The other findings carry
their tests beside the code they fix (test_intact_vertex_factor.py,
test_pdg_hyperons.py, test_completion_kernel.py)."""

from __future__ import annotations

import collections
import json
import math
from pathlib import Path

import pytest

from na49_mixing.cascade_decay import _kernel_shim
from na49_mixing.kernel_config import KernelConfig, load_kernel
from na49_mixing.resonance_decay import derive_charge_weights, isospin_closure_report

NESTED_ROOT = Path(__file__).resolve().parents[1]
R98 = json.loads((NESTED_ROOT / "config" / "runs" / "r98.json").read_text())


@pytest.fixture(scope="module")
def r98():
    return load_kernel(KernelConfig(**R98["kernel"]), NESTED_ROOT)


@pytest.mark.parametrize("xi", [False, True])
def test_the_isospin_audit_is_clean(xi):
    catalog, _fams = load_kernel(KernelConfig(**{**R98["kernel"], "xi_baryons": xi}), NESTED_ROOT)
    assert isospin_closure_report(catalog).errors == ()


def test_the_sigma_class_has_its_charge_weights(r98):
    """N(1440)+ -> p f0(500): one allowed combination, the tabulated one."""
    catalog, _fams = r98
    record = catalog.get(12212)
    weights = derive_charge_weights(record, (2212, 9000221), catalog)
    tabulated = math.fsum(c.branching_fraction for c in record.decay_channels if 9000221 in c.daughters)
    assert list(weights) == [(2212, 9000221)] and math.isclose(weights[(2212, 9000221)], tabulated, rel_tol=1e-12)


def test_the_decay_shares_are_unchanged(r98):
    """The open objects of the decayer keep the shares they had through the
    unfiltered fallback (N(1440)+ -> p f0(500) at 0.1726 of the pole width,
    Lambda(1670) -> Lambda f0(500) at 0.1762)."""
    catalog, _fams = r98
    decayer = _kernel_shim(catalog)
    for pid, share in ((12212, 0.1726), (33122, 0.1762)):
        record = catalog.get(pid)
        objects, total = decayer._open_objects(record, float(record.pole_mass))
        by = collections.defaultdict(float)
        for _channel, pair, weight in objects:
            by[tuple(sorted(pair))] += weight / total
        assert math.isclose(sum(v for k, v in by.items() if 9000221 in k), share, abs_tol=5e-5)
