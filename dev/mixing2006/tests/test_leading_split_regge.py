"""The Regge-flux remainder law of the leading-baryon split (2026-10-28, ``KernelConfig.leading_split_regge``): the
remainder's epsilon per split channel from the exchanged trajectory (2 alpha(0) - 2), the committed law bit for bit
when the option is absent, and the draw's inverse CDF at the per-channel epsilon."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing import cascade_decay as C
from na49_mixing.kernel_config import KernelConfig, load_kernel

NESTED_ROOT = Path(__file__).resolve().parents[1]
R127 = KernelConfig(**json.loads((NESTED_ROOT / "config" / "runs" / "r127.json").read_text())["kernel"])


@pytest.fixture(scope="module")
def kernels():
    return load_kernel(R127, NESTED_ROOT), load_kernel(dataclasses.replace(R127, leading_split_regge=True), NESTED_ROOT)


def test_the_option_validates_and_reaches_every_family(kernels):
    assert R127.as_dict()["leading_split_regge"] is None
    assert dataclasses.replace(R127, leading_split_regge=True).leading_split_regge is True
    for bad in ({"leading_split_regge": 1}, {"leading_split_regge": "true"}):
        with pytest.raises(ValueError, match="leading_split_regge"):
            dataclasses.replace(R127, **bad)
    with pytest.raises(ValueError, match="drop leading_split_epsilon"):
        dataclasses.replace(R127, leading_split_regge=True, leading_split_epsilon=-0.3)
    with pytest.raises(ValueError, match="needs leading_split_fraction"):
        dataclasses.replace(R127, leading_split_fraction=None, leading_split_kappa=None, meson_split_fraction=None,
                            meson_split_slope=None, leading_split_regge=True)
    (_c0, f0), (_c1, f1) = kernels
    assert all(getattr(f, "leading_split_regge", None) is None for f in f0)
    assert all(getattr(f, "leading_split_regge", None) is True for f in f1)


def test_the_intercepts_and_the_epsilons(kernels):
    cat, _fams = kernels[0]
    assert C.REGGE_INTERCEPTS["reggeon"] == pytest.approx(0.5475)
    assert C.REGGE_INTERCEPTS["pion"] == pytest.approx(-0.93 * 0.13957 ** 2)
    assert C.REGGE_INTERCEPTS["kaon"] == pytest.approx(-0.93 * 0.493677 ** 2)
    root = (1, 1, 0)  # a proton-like root
    eps = {name: C.regge_split_epsilon(root, cat.get(pdg)) for name, pdg in
           (("proton", 2212), ("Delta+", 2214), ("neutron", 2112), ("Delta++", 2224), ("Lambda", 3122), ("Sigma+", 3222))}
    assert eps["proton"] == eps["Delta+"] == pytest.approx(2 * 0.5475 - 2)
    assert eps["neutron"] == eps["Delta++"] == pytest.approx(2 * C.REGGE_INTERCEPTS["pion"] - 2)
    assert eps["Lambda"] == eps["Sigma+"] == pytest.approx(2 * C.REGGE_INTERCEPTS["kaon"] - 2)
    assert eps["Lambda"] < eps["neutron"] < eps["proton"] < 0.0
    # the conjugate tower: an antiproton-like root and its antibaryons see the same exchanges
    assert C.regge_split_epsilon((-1, -1, 0), cat.get(-2212)) == eps["proton"]
    assert C.regge_split_epsilon((-1, -1, 0), cat.get(-3122)) == eps["Lambda"]


def _state(M, family_id="baryonic_np"):
    return C.CascadeState(momentum=(M, 0.0, 0.0, 0.0), mass=M, qn=(1, 1, 0), quantum={}, family_id=family_id,
                          axis=(0.0, 0.0, 1.0), alignment_label="isotropic", parent=None, exchange_ancestry=("t",), depth=0)


def test_the_remainder_draw_follows_the_per_channel_law(kernels):
    (cat0, f0), (cat1, f1) = kernels
    fam0 = next(f for f in f0 if f.family_id == "mesonic_charged")
    fam1 = next(f for f in f1 if f.family_id == "mesonic_charged")
    split0 = next(f for f in f0 if f.family_id == "baryonic_np")
    split1 = next(f for f in f1 if f.family_id == "baryonic_np")
    M, m_p = 6.0, float(cat0.get(2212).pole_mass)
    us = np.linspace(0.05, 0.95, 7)
    committed = [C._split_remainder_mass(fam0, M, m_p, cat0, u, split0) for u in us]
    # without the established record the Regge kernel falls back to the committed law
    assert [C._split_remainder_mass(fam1, M, m_p, cat1, u, split1) for u in us] == committed
    # with it, the proton channel (the Reggeon) draws heavier remainders than the committed law, the Lambda heavier still
    regge_p = [C._split_remainder_mass(fam1, M, m_p, cat1, u, split1, established=cat1.get(2212), root_qn=(1, 1, 0)) for u in us]
    m_l = float(cat1.get(3122).pole_mass)
    regge_l = [C._split_remainder_mass(fam1, M, m_l, cat1, u, split1, established=cat1.get(3122), root_qn=(1, 1, 0)) for u in us]
    assert all(r > c for r, c in zip(regge_p, committed))
    assert np.mean(regge_l) > np.mean(regge_p)
    # the inverse CDF of (mu^2)^(-1-eps) at eps = 2 alpha - 2 reproduces the closed form
    floor = C._continuation_floor(fam1, cat1); edge = M - m_p; a, b = floor ** 2, edge ** 2
    eps = 2 * 0.5475 - 2
    for u, r in zip(us, regge_p):
        mu2 = (a ** (-eps) + u * (b ** (-eps) - a ** (-eps))) ** (-1.0 / eps)
        assert r == pytest.approx(math.sqrt(mu2))


def test_events_conserve_under_the_regge_law(kernels):
    from na49_mixing.topology_runner import TopologyNominalParams, run_topology
    _c0, (cat, fams) = kernels
    params = TopologyNominalParams(epsilon=0.1, b_sd=5.65, epsilon_nd=0.248, nd_central=False,
                                   remnant_weights={"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.0,
                                                    "high_mass_baryonic_continuum": 1.0},
                                   exact_transfer_range=True, line_thresholds=True)
    labels = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")
    for seed in range(25):
        rngs = {f"rng_{lab}": np.random.default_rng([seed, 28102028, k]) for k, lab in enumerate(labels)}
        ev = run_topology("nd", params, cat, fams, **rngs)
        for i in range(3):
            assert sum(int(h.qn[i]) for h in ev.final_state) == (2, 2, 0)[i]
