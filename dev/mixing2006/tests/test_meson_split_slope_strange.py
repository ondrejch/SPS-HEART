"""KernelConfig.meson_split_slope_strange (2026-11-06): the kaon's own direction law on the leading-meson branch.

On the branch a root emits an established meson along the incoming baryon beside a baryonic continuum remainder,
with kappa = 2 b q*^2 at b = meson_split_slope.  The option gives the strange mesons of the branch (the kaon of
p* -> K Y*) their own slope b_K on the same uniform: the channel draw and the non-strange mesons stay bit for bit,
and a kaon moves closer to the incoming baryon as b_K grows.  The fit's split reweighting carries one slope, so the
engine refuses a spec that frees the split parameters together with the option.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing import cascade_decay as C
from na49_mixing.kernel_config import KernelConfig, load_kernel

NESTED_ROOT = Path(__file__).resolve().parents[1]
SPEC = NESTED_ROOT / "config" / "runs" / "r141.json"
R141 = KernelConfig(**json.loads(SPEC.read_text())["kernel"])


class _Counting:
    """A numpy generator whose first uniform is *first* (0 puts the root on the meson branch)."""

    def __init__(self, seed, first=None):
        self.rng, self.n, self.first = np.random.default_rng(seed), 0, first

    def random(self):
        self.n += 1
        if self.n == 1 and self.first is not None:
            return self.first
        return self.rng.random()


def _root(M):
    return C.CascadeState(momentum=(M, 0.0, 0.0, 0.0), mass=M, qn=(1, 1, 0), quantum={}, family_id="baryonic_np",
                          axis=(0.0, 0.0, 1.0), alignment_label="isotropic", parent=None,
                          exchange_ancestry=("meson_split",), depth=0)


def test_the_option_validates_and_defaults_to_the_branch_law():
    assert R141.meson_split_slope_strange is None and R141.as_dict()["meson_split_slope_strange"] is None
    on = dataclasses.replace(R141, meson_split_slope_strange=16)
    assert on.meson_split_slope_strange == 16.0 and isinstance(on.meson_split_slope_strange, float)
    for bad in (-1.0, 200.0, "4", True, math.nan):
        with pytest.raises(ValueError, match="meson_split_slope_strange"):
            dataclasses.replace(R141, meson_split_slope_strange=bad)
    with pytest.raises(ValueError, match="needs meson_split_fraction"):
        dataclasses.replace(R141, meson_split_fraction=None, meson_split_slope=None, meson_split_slope_strange=8.0)


@pytest.fixture(scope="module")
def kernels():
    return {"base": load_kernel(R141, NESTED_ROOT),
            "kaon": load_kernel(dataclasses.replace(R141, meson_split_slope_strange=16.0), NESTED_ROOT)}


def test_only_the_kaons_of_the_meson_branch_change_direction(kernels):
    state = _root(5.0)
    draws = {}
    for name, (cat, fams) in kernels.items():
        channels = C.enumerate_channels(state, cat, fams)
        out = []
        for seed in range(200):
            ch, (d0, d1), _pool = C.draw_decay_with_pool(state, channels, cat, fams, _Counting(seed, first=0.0),
                                                         _Counting(seed + 1000))
            meson = d0 if d0.qn[0] == 0 else d1
            out.append((ch.daughters, int(meson.qn[2]), meson.momentum))
        draws[name] = out
    strange = 0
    for (ch0, s0, p0), (ch1, s1, p1) in zip(draws["base"], draws["kaon"]):
        assert ch0 == ch1 and s0 == s1  # the channel draw is unchanged
        if s0 == 0:
            assert p0 == p1  # a non-strange meson keeps its draw bit for bit
        else:
            strange += 1
            c0 = p0[3] / math.sqrt(p0[1] ** 2 + p0[2] ** 2 + p0[3] ** 2)
            c1 = p1[3] / math.sqrt(p1[1] ** 2 + p1[2] ** 2 + p1[3] ** 2)
            assert c1 >= c0 - 1e-12  # a steeper law at the same uniform moves the kaon toward the incoming baryon
    assert strange > 0


def test_the_engine_refuses_the_option_with_free_split_parameters(tmp_path):
    spec = json.loads(SPEC.read_text())
    spec["kernel"]["meson_split_slope_strange"] = 8.0
    path = tmp_path / "r141_kaon_slope.json"
    path.write_text(json.dumps(spec))
    loader = importlib.util.spec_from_file_location("run_engine_kaon_slope",
                                                    NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py")
    engine = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(engine)
    with pytest.raises(ValueError, match="meson_split_slope_strange is not reweighted"):
        engine.configure(path)
