"""The isospin-allowed partner channels of the baryonic lines (2026-10-15):
``KernelConfig.hyperon_meson_partners`` and ``delta_isoscalar_pairs``;
off, the committed classes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from na49_mixing.kernel_config import KernelConfig, load_kernel

NESTED_ROOT = Path(__file__).resolve().parents[1]
R97 = json.loads((NESTED_ROOT / "config" / "runs" / "r97.json").read_text())


def _classes(**opts):
    _cat, fams = load_kernel(KernelConfig(**{**R97["kernel"], **opts}), NESTED_ROOT)
    return {str(f.family_id): list(f.allowed_decay_classes) for f in fams}


@pytest.fixture(scope="module")
def kernels():
    return _classes(), _classes(hyperon_meson_partners=True, delta_isoscalar_pairs=True)


def test_the_hyperon_partners(kernels):
    """Beside every (Y, pi0) class (Y, rho0), (Y, omega), (Y, eta), beside
    (Y, pi+-) (Y, rho+-), at both towers; the non-strange families and the
    mesonic ones are untouched."""
    off, on = kernels
    added = {k: [c for c in on[k] if c not in off[k]] for k in on}
    assert added["baryonic_strange"] == [
        "(3122,113)", "(3122,223)", "(3122,221)", "(3212,113)", "(3212,223)", "(3212,221)", "(3112,213)", "(3222,-213)",
        "(3114,213)", "(3214,113)", "(3214,223)", "(3214,221)", "(3224,-213)", "(102134,113)", "(102134,223)", "(102134,221)"]
    assert added["anti_baryonic_strange"][:3] == ["(-3122,113)", "(-3122,223)", "(-3122,221)"]
    assert added["baryonic_np"] == []
    assert not any(added[k] for k in added if k.startswith("mesonic"))


def test_the_delta_pairs(kernels):
    off, on = kernels
    added = {k: [c for c in on[k] if c not in off[k]] for k in on}
    assert added["baryonic_np_q2"] == ["(2224,223)", "(2224,221)", "(2224,9000221)", "(2224,9010221)"]
    assert added["anti_baryonic_np_qm1"] == ["(-1114,223)", "(-1114,221)", "(-1114,9000221)", "(-1114,9010221)"]
    assert added["baryonic_np"] == [] and added["mesonic_neutral"] == []


def test_the_refusal():
    bare = {k: v for k, v in R97["kernel"].items() if k != "light_scalar_mesons"}
    with pytest.raises(ValueError, match="delta_isoscalar_pairs"):
        load_kernel(KernelConfig(**{**bare, "delta_isoscalar_pairs": True}), NESTED_ROOT)


def test_the_charged_scalar_pairs():
    off, on = _classes(), _classes(charged_scalar_pairs=True)
    assert [c for c in on["mesonic_charged"] if c not in off["mesonic_charged"]] == ["(9010221,211)", "(9000211,111)", "(9000111,211)"]
    assert [c for c in on["mesonic_anticharged"] if c not in off["mesonic_anticharged"]] == [
        "(9010221,-211)", "(-9000211,111)", "(9000111,-211)"]
    assert on["mesonic_neutral"] == off["mesonic_neutral"]
