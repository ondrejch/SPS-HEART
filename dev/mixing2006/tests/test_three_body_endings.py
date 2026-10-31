"""The three-body endings of the continuum cascade
(``KernelConfig.three_body_endings``, 2026-10-20; owner: "implement 3body
vtx"): three-body direct channels of stable hadrons in the floor's pool,
weighted kappa_3 prod(2J+1) M R_3 / pi C_iso beside the two-body pairs and
drawn from flat three-body phase space."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing.cascade_decay import (KIND_CONTINUUM, KIND_THREE_BODY, CascadeState, draw_decay_with_pool,
                                       enumerate_channels)
from na49_mixing.kernel_config import KernelConfig, load_kernel
from na49_mixing.three_body_endings import (phase_space_factor, three_body_classes_for, three_body_isospin_share,
                                            three_body_weight)
from na49_mixing.topology_runner import TopologyNominalParams, run_topology

NESTED_ROOT = Path(__file__).resolve().parents[1]
RUNS = NESTED_ROOT / "config" / "runs"
LABELS = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")


@pytest.fixture(scope="module")
def kernels():
    r102 = json.loads((RUNS / "r102.json").read_text())
    off = load_kernel(KernelConfig(**r102["kernel"]), NESTED_ROOT)
    on = load_kernel(KernelConfig(**{**r102["kernel"], "three_body_endings": 1.0}), NESTED_ROOT)
    return off, on


def _state(fid, qn, mass, depth=3):
    return CascadeState(momentum=(mass, 0.0, 0.0, 0.0), mass=mass, qn=qn, quantum={}, family_id=fid, axis=(0.0, 0.0, 1.0),
                        alignment_label="none", parent=None, exchange_ancestry=(), depth=depth)


def test_off_by_default(kernels):
    (cat, fams), _on = kernels
    assert KernelConfig().three_body_endings is None and KernelConfig().as_dict()["three_body_endings"] is None
    assert all(not f.three_body_classes and f.three_body_strength is None for f in fams)
    with pytest.raises(ValueError, match="three_body_endings"):
        KernelConfig(three_body_endings=0.0)
    with pytest.raises(ValueError, match="three_body_endings"):
        KernelConfig(three_body_endings=True)


def test_the_sets_follow_isospin(kernels):
    _off, (cat, fams) = kernels
    by_id = {f.family_id: f for f in fams}
    assert by_id["mesonic_neutral"].three_body_classes == ("(211,111,-211)", "(111,111,111)")
    assert set(by_id["baryonic_np"].three_body_classes) == {"(2212,211,-211)", "(2212,111,111)", "(2112,211,111)",
                                                            "(3322,321,311)", "(3312,321,321)"}
    assert all(f.three_body_strength == 1.0 for f in fams if f.three_body_classes)
    # the I = 0 three-pion state has no pi0 pi0 pi0 component (omega -> 3 pi0 is forbidden) and carries one multiplet
    assert three_body_isospin_share(0.0, 0, (111, 111, 111), cat) == pytest.approx(0.0, abs=1e-12)
    assert three_body_isospin_share(0.0, 0, (211, 111, -211), cat) == pytest.approx(1.0, rel=1e-12)
    # 1 x 1 x 1 holds three I = 1 multiplets; 1/2 x 1 x 1 two I = 1/2 multiplets; 1/2 x 1/2 x 1/2 two I = 1/2
    assert (three_body_isospin_share(1.0, 2, (211, 211, -211), cat)
            + three_body_isospin_share(1.0, 2, (211, 111, 111), cat)) == pytest.approx(3.0, rel=1e-12)
    assert sum(three_body_isospin_share(0.5, 1, t, cat) for t in ((2212, 211, -211), (2212, 111, 111), (2112, 211, 111))
               ) == pytest.approx(2.0, rel=1e-12)
    assert sum(three_body_isospin_share(0.5, 1, t, cat) for t in ((3322, 321, 311), (3312, 321, 321))
               ) == pytest.approx(2.0, rel=1e-12)
    assert three_body_classes_for((0, 0, -2), cat, ("pipipi",)) == ()


def test_the_phase_space_factor():
    """M R_3 / pi = pi M^3 / 8 for massless daughters; zero at threshold;
    the weight is linear in kappa_3."""
    for M in (0.8, 1.5, 3.0):
        assert phase_space_factor(M, 0.0, 0.0, 0.0) == pytest.approx(math.pi * M ** 3 / 8.0, rel=2e-3)
    assert phase_space_factor(0.4, 0.1396, 0.1396, 0.135) == 0.0
    w1 = three_body_weight(1.0, 1.5, (211, 211, -211), (0.1396, 0.1396, 0.1396), (1, 1, 1), 1.0)
    assert three_body_weight(2.5, 1.5, (211, 211, -211), (0.1396, 0.1396, 0.1396), (1, 1, 1), 1.0) == pytest.approx(2.5 * w1)


def test_the_endings_sit_at_the_floor(kernels):
    """Three-body channels open only where no continuation channel is; the
    committed kernel offers none."""
    (cat0, fams0), (cat, fams) = kernels
    floor = _state("mesonic_charged", (0, 1, 0), 1.6)
    on = enumerate_channels(floor, cat, fams)
    assert not any(c.kind == KIND_CONTINUUM for c in on) and any(c.kind == KIND_THREE_BODY for c in on)
    assert not any(c.kind == KIND_THREE_BODY for c in enumerate_channels(floor, cat0, fams0))
    high = _state("mesonic_charged", (0, 1, 0), 3.5)
    chans = enumerate_channels(high, cat, fams)
    assert any(c.kind == KIND_CONTINUUM for c in chans) and not any(c.kind == KIND_THREE_BODY for c in chans)
    assert math.isclose(math.fsum(c.weight for c in on), 1.0, rel_tol=1e-12)


def test_the_kinematics(kernels):
    """A drawn three-body ending conserves (B, Q, S) and four-momentum (the
    draw checks both), is isotropic, and fills its Dalitz plot flat."""
    _off, (cat, fams) = kernels
    st = _state("mesonic_charged", (0, 1, 0), 1.6)
    chans = [c for c in enumerate_channels(st, cat, fams) if c.kind == KIND_THREE_BODY]
    pick = next(c for c in chans if tuple(c.daughters) == ("catalog:211", "catalog:211", "catalog:-211"))
    rng = np.random.default_rng(5)
    cz, m12 = [], []
    for _ in range(4000):
        _sel, daughters, _pool = draw_decay_with_pool(st, [pick], cat, fams, np.random.default_rng(rng.integers(1 << 30)), rng)
        assert len(daughters) == 3
        p = [np.asarray(d.momentum) for d in daughters]
        assert np.allclose(sum(p), (1.6, 0.0, 0.0, 0.0), atol=1e-9)
        for v in p:
            cz.append(v[3] / np.linalg.norm(v[1:]))
        s01 = p[0] + p[1]
        m12.append(s01[0] ** 2 - np.dot(s01[1:], s01[1:]))
    assert abs(np.mean(cz)) < 0.03 and abs(np.mean(np.square(cz)) - 1.0 / 3.0) < 0.02
    # flat Dalitz: the m12^2 density follows the strip length q* q / m (phase space)
    from na49_mixing.resonance_widths import decay_momentum
    mpi = 0.13957
    lo, hi = (2 * mpi) ** 2, (1.6 - mpi) ** 2
    edges = np.linspace(lo, hi, 9)
    hist, _ = np.histogram(m12, bins=edges)
    mids = 0.5 * (edges[1:] + edges[:-1])
    dens = np.array([decay_momentum(math.sqrt(x), mpi, mpi) * decay_momentum(1.6, math.sqrt(x), mpi) / math.sqrt(x)
                     for x in mids])
    expect = dens / dens.sum() * hist.sum()
    assert np.all(np.abs(hist - expect) < 5.0 * np.sqrt(expect) + 10.0)


def test_events_conserve(kernels):
    _off, (cat, fams) = kernels
    r102 = json.loads((RUNS / "r102.json").read_text())
    params = TopologyNominalParams(epsilon=0.53, b_sd=6.44, epsilon_nd=0.247, nd_central=False,
                                   remnant_weights={"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.0,
                                                    "high_mass_baryonic_continuum": 1.0}, **r102["topology"])
    seen_three = 0
    for seed in range(30):
        rngs = {f"rng_{lab}": np.random.default_rng([seed, 8192, k]) for k, lab in enumerate(LABELS)}
        ev = run_topology("nd", params, cat, fams, **rngs)
        for i in range(3):
            assert sum(int(h.qn[i]) for h in ev.final_state) == (2, 2, 0)[i]
        seen_three += len(ev.final_state)
    assert seen_three > 0
