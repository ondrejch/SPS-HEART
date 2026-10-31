"""The leading-baryon split of the baryonic continuum root (2026-10-08):
``KernelConfig.leading_split_fraction`` sends that share of the root decays
(depth 0, B = 1) through B* -> B + M*(mu), the established baryon and a
mesonic continuum daughter, mu^2 from (mu^2)^(-1-epsilon) on
[floor^2, edge^2], the baryon directed along the incoming baryon with
exp(kappa cos theta); the draw budget of the vertex is unchanged and the
absent option is the committed kernel bit for bit."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing import cascade_decay as C
from na49_mixing.kernel_config import R74, KernelConfig, load_kernel

NESTED_ROOT = Path(__file__).resolve().parents[1]
BASE = dataclasses.replace(R74, isospin_completion=("baryonic", "strange_baryonic", "strange_mesonic"), delta_lines=True,
                           three_body_phase_space=True)


class _Counting:
    """A numpy generator that counts its uniforms."""

    def __init__(self, seed):
        self.rng, self.n = np.random.default_rng(seed), 0

    def random(self):
        self.n += 1
        return self.rng.random()


def _root(M, pz=0.0, depth=0):
    e = math.sqrt(M * M + pz * pz)
    return C.CascadeState(momentum=(e, 0.0, 0.0, pz), mass=M, qn=(1, 1, 0), quantum={}, family_id="baryonic_np",
                          axis=(0.0, 0.0, 1.0), alignment_label="isotropic", parent=None, exchange_ancestry=("test",),
                          depth=depth)


def _is_split(channel, cat, fams):
    ids = [C._daughter_identity(cat, fams, d) for d in channel.daughters]
    kinds = [i[0] for i in ids]
    if sorted(kinds) != ["catalog", "family"]:
        return False
    return (C._baryon_number_of(ids[kinds.index("catalog")]) == 1
            and C._baryon_number_of(ids[kinds.index("family")]) == 0)


@pytest.fixture(scope="module")
def kernels():
    return {"plain": load_kernel(BASE, NESTED_ROOT),
            "all": load_kernel(dataclasses.replace(BASE, leading_split_fraction=1.0, leading_split_kappa=8.0), NESTED_ROOT),
            "third": load_kernel(dataclasses.replace(BASE, leading_split_fraction=0.3, leading_split_slope=2.0), NESTED_ROOT)}


def test_the_options_validate():
    split = dataclasses.replace(BASE, leading_split_fraction=0.2, leading_split_kappa=4)
    assert split.leading_split_kappa == 4.0 and isinstance(split.leading_split_kappa, float)
    assert split.as_dict()["leading_split_fraction"] == 0.2 and split.as_dict()["leading_split_slope"] is None
    for bad in ({"leading_split_fraction": 0.0}, {"leading_split_fraction": 1.2}, {"leading_split_fraction": True},
                {"leading_split_kappa": 2.0},  # needs the fraction
                {"leading_split_fraction": 0.2, "leading_split_kappa": 2.0, "leading_split_slope": 1.0},
                {"leading_split_fraction": 0.2, "leading_split_kappa": -1.0},
                {"leading_split_fraction": 0.2, "leading_split_epsilon": math.inf}):
        with pytest.raises(ValueError):
            dataclasses.replace(BASE, **bad)
    with pytest.raises(ValueError, match="density"):
        KernelConfig(leading_split_fraction=0.2)


def test_the_later_steps_are_unchanged(kernels):
    """With the option a vertex below the root (depth 1) draws bit for bit
    as without it; the root itself draws differently."""
    draws = {}
    for depth in (0, 1):
        for key in ("plain", "all"):
            cat, fams = kernels[key]
            state = _root(5.0, pz=3.0, depth=depth)
            channels = C.enumerate_channels(state, cat, fams)
            out = []
            for seed in range(20):
                ch, (d0, d1) = C.draw_decay(state, channels, cat, fams, np.random.default_rng(seed), np.random.default_rng(seed + 100))
                out.append((ch.daughters, d0.momentum, d1.momentum))
            draws[key, depth] = out
    assert draws["plain", 1] == draws["all", 1]
    assert draws["plain", 0] != draws["all", 0]


def test_the_split_root(kernels):
    """F = 1: every root decay is a split; the budget is 1 + 3 uniforms;
    the remainder is flat in ln mu^2 on its window; the baryon follows
    the incoming baryon (kappa = 8), on either side."""
    cat, fams = kernels["all"]
    floor = 1.7
    for pz, side in ((0.0, 1.0), (-0.4, -1.0)):
        state = _root(5.0, pz=pz)
        channels = C.enumerate_channels(state, cat, fams)
        logs, cosines = [], []
        for seed in range(300):
            rc, rk = _Counting(seed), _Counting(seed + 1000)
            ch, (d0, d1) = C.draw_decay(state, channels, cat, fams, rc, rk)
            assert _is_split(ch, cat, fams) and rc.n == 1 and rk.n == 3
            baryon, meson = (d0, d1) if d0.qn[0] == 1 else (d1, d0)
            edge = 5.0 - baryon.mass
            assert floor - 1e-9 <= meson.mass <= edge + 1e-9
            logs.append((math.log(meson.mass ** 2) - math.log(floor ** 2)) / (math.log(edge ** 2) - math.log(floor ** 2)))
            cosines.append(math.copysign(1.0, baryon.momentum[3] - pz * baryon.momentum[0] / state.momentum[0]))
        assert abs(np.mean(logs) - 0.5) < 0.06
        assert side * np.mean(cosines) > 0.8
    # the closure holds through the checked kinematics, and the later steps cascade on
    assert all(abs(a + b - c) < 1e-9 for a, b, c in zip(d0.momentum, d1.momentum, state.momentum))


def test_the_split_share_and_the_slope_law(kernels):
    """F = 0.3: the split-type channels take F + (1 - F) W_split / W of
    the root decays (the one channel uniform picks branch and channel)."""
    cat, fams = kernels["third"]
    state = _root(5.0, pz=2.0)
    channels = C.enumerate_channels(state, cat, fams)
    w_split = math.fsum(c.weight for c in channels if _is_split(c, cat, fams))
    expected = 0.3 + 0.7 * w_split / math.fsum(c.weight for c in channels)
    n, hits = 3000, 0
    rng_c, rng_k = np.random.default_rng(7), np.random.default_rng(8)
    for _ in range(n):
        ch, _d = C.draw_decay(state, channels, cat, fams, rng_c, rng_k)
        hits += _is_split(ch, cat, fams)
    assert abs(hits / n - expected) < 4.0 * math.sqrt(expected * (1.0 - expected) / n)
    # the slope law: kappa = 2 b q*^2 at the split's decay momentum
    split = next(f for f in fams if f.family_id == "baryonic_np")
    identities = [C._daughter_identity(cat, fams, "catalog:2212"), ("family", None, next(f for f in fams if f.family_id == "mesonic_neutral"))]
    q = C.decay_momentum(5.0, 0.93827, 3.0)
    kappa = 2.0 * 2.0 * q * q
    u = 0.37
    want = 1.0 + math.log(u + (1.0 - u) * math.exp(-2.0 * kappa)) / kappa
    assert C._leading_split_cos(u, _root(5.0, pz=1.0), identities, (0.93827, 3.0), split) == pytest.approx(want)
    assert C._leading_split_cos(u, _root(5.0, pz=-1.0), identities[::-1], (3.0, 0.93827), split) == pytest.approx(want)
