"""The eta'(958) (2026-10-15): ``KernelConfig.eta_prime`` adds the record at
its PDG 2025 values and the eta' analog of every continuum class with an
eta; off, the committed kernel."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from na49_mixing.eta_prime import eta_prime_classes
from na49_mixing.kernel_config import KernelConfig, load_kernel

NESTED_ROOT = Path(__file__).resolve().parents[1]
R97 = json.loads((NESTED_ROOT / "config" / "runs" / "r97.json").read_text())


@pytest.fixture(scope="module")
def kernels():
    return (load_kernel(KernelConfig(**R97["kernel"]), NESTED_ROOT),
            load_kernel(KernelConfig(**{**R97["kernel"], "eta_prime": True}), NESTED_ROOT))


def test_the_record(kernels):
    (cat_off, _f0), (cat, _f1) = kernels
    assert 331 not in {int(r.pdg_id) for r in cat_off.records()}
    rec = cat.get(331)
    assert rec.pole_mass == 0.95778 and rec.nominal_width == 0.000188 and rec.isospin == 0.0 and rec.parity == "-"
    bf = {tuple(c.daughters): c.branching_fraction for c in rec.decay_channels}
    total = 0.425 + 0.295 + 0.224 + 0.0252 + 0.0231
    assert math.isclose(bf[(221, 211, -211)], 0.425 / total) and math.isclose(bf[(113, 22)], 0.295 / total)
    assert math.isclose(math.fsum(bf.values()), 1.0, rel_tol=1e-12)


def test_the_classes(kernels):
    """Every eta class has its eta' analog, and a class is an unordered pair:
    the (eta, eta) class gives (eta', eta) once and (eta', eta')."""
    (_c0, fams_off), (_c1, fams) = kernels
    off = {str(f.family_id): list(f.allowed_decay_classes) for f in fams_off}
    on = {str(f.family_id): list(f.allowed_decay_classes) for f in fams}
    added = {k: [c for c in on[k] if c not in off[k]] for k in on}
    assert added["mesonic_neutral"] == ["(331,111)", "(331,221)", "(331,331)", "(333,331)", "(331,223)"]
    assert added["baryonic_np"] == ["(2212,331)"] and added["anti_baryonic_np"] == ["(-2212,331)"]
    assert added["mesonic_charged"] == ["(331,211)"]
    assert all(on[k][: len(off[k])] == off[k] for k in on)
    assert eta_prime_classes(["(221,221)"]) == ["(331,221)", "(331,331)"]
