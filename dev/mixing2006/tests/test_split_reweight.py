"""The leading splits as fitted parameters (2026-11-03, the fourteen-parameter fit): the cascade's split recorder and
the importance weights of :mod:`na49_mixing.split_reweight`.

The recorder draws no random number, so the decays are those of the recorder off. The weights are one at the
generation point, and a sample drawn at one split point and reweighted to another reproduces a sample drawn there:
the branch shares, the channel content, and the direction of the leading daughter."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing import cascade_decay as C
from na49_mixing import split_reweight as SR
from na49_mixing.kernel_config import KernelConfig, load_kernel

NESTED_ROOT = Path(__file__).resolve().parents[1]
R136 = KernelConfig(**json.loads((NESTED_ROOT / "config" / "runs" / "r136.json").read_text())["kernel"])
GEN = (0.2, 0.045, 4.0, 4.0)
TRIAL = (0.3, 0.12, 6.0, 6.0)


def _kernel(point):
    fb, fm, kb, bm = point
    return load_kernel(dataclasses.replace(R136, leading_split_fraction=fb, meson_split_fraction=fm,
                                           leading_split_kappa=kb, meson_split_slope=bm), NESTED_ROOT)


@pytest.fixture(scope="module")
def kernels():
    return {"gen": _kernel(GEN), "trial": _kernel(TRIAL)}


def _root(M=5.0, pz=2.0):
    e = math.sqrt(M * M + pz * pz)
    return C.CascadeState(momentum=(e, 0.0, 0.0, pz), mass=M, qn=(1, 1, 0), quantum={}, family_id="baryonic_np",
                          axis=(0.0, 0.0, 1.0), alignment_label="isotropic", parent=None,
                          exchange_ancestry=("split_test",), depth=0)


def _draw(kernel, n, seed, record=True):
    """n root decays: per decay the record, the drawn channel's daughter ids and daughter 0's c.m. cos(theta)."""
    cat, fams = kernel
    state = _root()
    channels = C.enumerate_channels(state, cat, fams)
    rc, rk = np.random.default_rng(seed), np.random.default_rng(seed + 1)
    rows = []
    for _ in range(n):
        C.SPLIT_RECORDER = [] if record else None
        try:
            ch, (d0, _d1), _pool = C.draw_decay_with_pool(state, channels, cat, fams, rc, rk)
            recs = C.SPLIT_RECORDER
        finally:
            C.SPLIT_RECORDER = None
        p = d0.momentum
        rows.append((recs, tuple(ch.daughters), p[3] / math.sqrt(p[1] ** 2 + p[2] ** 2 + p[3] ** 2)))
    return rows


def _arrays(rows):
    cols = {}
    for recs, _ch, _c in rows:
        for key, value in SR.records_to_row(recs).items():
            cols.setdefault(key, []).append(value)
    return {k: np.asarray(v) for k, v in cols.items()}


def test_the_recorder_leaves_the_decays_unchanged(kernels):
    on = _draw(kernels["gen"], 300, 11, record=True)
    off = _draw(kernels["gen"], 300, 11, record=False)
    assert [(ch, c) for _r, ch, c in on] == [(ch, c) for _r, ch, c in off]
    assert all(len(r) == 1 for r, _ch, _c in on)            # one eligible root per decay
    assert {r[0]["branch"] for r, _ch, _c in on} == {0, 1, 2}


def test_the_weights_are_one_at_the_generation_point(kernels):
    a = _arrays(_draw(kernels["gen"], 400, 21))
    assert np.max(np.abs(SR.split_log_ratio(a, GEN, GEN))) < 1e-12


def test_a_reweighted_sample_reproduces_the_trial_sample(kernels):
    n = 30_000
    gen = _draw(kernels["gen"], n, 31)
    trial = _draw(kernels["trial"], n, 41)
    w = np.exp(SR.split_log_ratio(_arrays(gen), TRIAL, GEN))
    assert w.mean() == pytest.approx(1.0, abs=0.02)
    # the branch shares
    br_g = np.array([r[0]["branch"] for r, _ch, _c in gen])
    br_t = np.array([r[0]["branch"] for r, _ch, _c in trial])
    for b in (0, 1, 2):
        assert np.sum(w * (br_g == b)) / np.sum(w) == pytest.approx(np.mean(br_t == b), abs=0.012), b
    # the direction of daughter 0, in five bins
    edges = np.linspace(-1.0, 1.0, 6)
    h_w, _ = np.histogram([c for _r, _ch, c in gen], edges, weights=w)
    h_t, _ = np.histogram([c for _r, _ch, c in trial], edges)
    assert np.all(np.abs(h_w / np.sum(h_w) - h_t / np.sum(h_t)) < 0.012), (h_w / np.sum(h_w), h_t / np.sum(h_t))
    # the ten most common channels
    common = {}
    for _r, ch, _c in trial:
        common[ch] = common.get(ch, 0) + 1
    for ch in sorted(common, key=common.get, reverse=True)[:10]:
        f_w = np.sum(w * np.array([c == ch for _r, c, _x in gen])) / np.sum(w)
        assert f_w == pytest.approx(common[ch] / n, abs=0.01), ch


def test_the_direction_density_at_the_uniform_limit():
    """kappa = 0 is the uniform density 1/2 (the code review of 2026-11-04, finding S1), and the small-kappa series
    joins the closed form continuously."""
    c = np.array([-1.0, -0.3, 0.4, 1.0])
    assert np.allclose(SR._log_direction(np.zeros(4), c), -math.log(2.0))
    lo = SR._log_direction(np.full(4, 0.999e-6), c)
    hi = SR._log_direction(np.full(4, 1.001e-6), c)
    assert np.allclose(lo, hi, atol=1e-9)
    # the density integrates to one over cos(theta) in [-1, 1] at several kappa
    grid = np.linspace(-1.0, 1.0, 20001)
    for kappa in (0.0, 1e-8, 0.5, 4.0):
        dens = np.exp(SR._log_direction(np.full(grid.size, kappa), grid))
        assert float(np.sum(0.5 * (dens[1:] + dens[:-1]) * np.diff(grid))) == pytest.approx(1.0, abs=1e-6)

