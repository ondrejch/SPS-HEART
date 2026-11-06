"""The S = -3 baryon (``KernelConfig.omega_baryon``, 2026-11-04): the Omega- record, its conjugate, the (Omega-, K)
classes of the S = -2 continuum, and the Omega pairs of the baryon-pair completion."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from na49_mixing import cascade_decay as C
from na49_mixing.kernel_config import KernelConfig, load_kernel
from na49_mixing.omega_baryon import OMEGA

NESTED_ROOT = Path(__file__).resolve().parents[1]
R137 = KernelConfig(**json.loads((NESTED_ROOT / "config" / "runs" / "r137.json").read_text())["kernel"])


@pytest.fixture(scope="module")
def kernel():
    return load_kernel(dataclasses.replace(R137, omega_baryon=True, baryon_pair_completion="strange"), NESTED_ROOT)


def test_off_is_the_committed_kernel():
    assert R137.as_dict()["omega_baryon"] is False
    cat, _ = load_kernel(R137, NESTED_ROOT)
    with pytest.raises(KeyError):
        cat.get(OMEGA)


def test_the_option_is_checked():
    with pytest.raises(ValueError, match="omega_baryon"):
        dataclasses.replace(R137, omega_baryon=1)
    with pytest.raises(ValueError, match="omega_baryon"):
        dataclasses.replace(R137, omega_baryon=True, xi_tower=None)


def test_the_records(kernel):
    cat, _ = kernel
    om, anti = cat.get(OMEGA), cat.get(-OMEGA)
    assert (om.B, om.Q, om.S, om.isospin, om.spin) == (1, -1, -3, 0.0, 1.5)
    assert om.pole_mass == pytest.approx(1.67245)
    assert om.stable_for_decay and not om.decay_channels
    assert (anti.B, anti.Q, anti.S) == (-1, 1, 3) and anti.stable_for_decay


def test_the_classes(kernel):
    cat, fams = kernel
    fam = {str(f.family_id): f for f in fams}
    assert "(3334,321)" in fam["baryonic_xi_q0"].allowed_decay_classes
    assert "(3334,311)" in fam["baryonic_xi_qm1"].allowed_decay_classes
    assert "(-3334,-321)" in fam["anti_baryonic_xi_q0"].allowed_decay_classes
    assert "(-3334,-311)" in fam["anti_baryonic_xi_qm1"].allowed_decay_classes
    assert "(3334,-3334)" in fam["mesonic_neutral"].completion_resonance_classes
    assert "(3334,-3312)" in fam["mesonic_strange"].completion_resonance_classes
    # an S = -2 line of 3 GeV opens (Omega-, K) channels
    state = C.CascadeState(momentum=(3.0, 0.0, 0.0, 0.0), mass=3.0, qn=(1, 0, -2), quantum={},
                           family_id="baryonic_xi_q0", axis=(0.0, 0.0, 1.0), alignment_label="isotropic", parent=None,
                           exchange_ancestry=("omega_test",), depth=0)
    channels = C.enumerate_channels(state, cat, fams)
    assert any("catalog:3334" in ch.daughters for ch in channels)
