"""The leading-meson split of a root of the leading-baryon split
(2026-10-11): ``KernelConfig.meson_split_fraction`` sends that share of the
split roots' decays (depth 0, B = 1) through B* -> M + B*(mu), an
established meson and a baryonic continuum daughter.  One uniform on the
decay_channel substream, before the channel uniform, picks the branch; the
branch draws over those channels with the split's mass law and the meson
along the incoming baryon (``meson_split_slope``: kappa = 2 b q*^2).  The
absent option is the committed kernel bit for bit."""

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
R90 = KernelConfig(**json.loads((NESTED_ROOT / "config" / "runs" / "r90.json").read_text())["kernel"])


class _Counting:
    """A numpy generator that counts its uniforms (the first one *first*
    when given)."""

    def __init__(self, seed, first=None):
        self.rng, self.n, self.first = np.random.default_rng(seed), 0, first

    def random(self):
        self.n += 1
        if self.n == 1 and self.first is not None:
            return self.first
        return self.rng.random()


def _root(M, pz=0.0, depth=0, qn=(1, 1, 0), family_id="baryonic_np"):
    e = math.sqrt(M * M + pz * pz)
    return C.CascadeState(momentum=(e, 0.0, 0.0, pz), mass=M, qn=qn, quantum={}, family_id=family_id,
                          axis=(0.0, 0.0, 1.0), alignment_label="isotropic", parent=None,
                          exchange_ancestry=("meson_split",), depth=depth)


def _is_meson_split(channel, cat, fams):
    ids = [C._daughter_identity(cat, fams, d) for d in channel.daughters]
    kinds = [i[0] for i in ids]
    if sorted(kinds) != ["catalog", "family"]:
        return False
    return (C._baryon_number_of(ids[kinds.index("catalog")]) == 0
            and C._baryon_number_of(ids[kinds.index("family")]) == 1)


@pytest.fixture(scope="module")
def kernels():
    return {"off": load_kernel(R90, NESTED_ROOT),
            "third": load_kernel(dataclasses.replace(R90, meson_split_fraction=0.3, meson_split_slope=4.0), NESTED_ROOT),
            "steep": load_kernel(dataclasses.replace(R90, meson_split_fraction=0.5, meson_split_slope=8.0), NESTED_ROOT),
            "kappa": load_kernel(dataclasses.replace(R90, meson_split_fraction=0.3), NESTED_ROOT)}


def test_the_options_validate_and_reach_every_family(kernels):
    assert R90.as_dict()["meson_split_fraction"] is None and R90.as_dict()["meson_split_slope"] is None
    on = dataclasses.replace(R90, meson_split_fraction=0.03, meson_split_slope=4)
    assert on.meson_split_slope == 4.0 and isinstance(on.meson_split_slope, float)
    assert on.as_dict()["meson_split_fraction"] == 0.03
    for bad in ({"meson_split_fraction": 0.0}, {"meson_split_fraction": 1.0}, {"meson_split_fraction": True},
                {"meson_split_fraction": math.nan}, {"meson_split_slope": 4.0},  # needs the fraction
                {"meson_split_fraction": 0.03, "meson_split_slope": -1.0},
                {"meson_split_fraction": 0.03, "meson_split_slope": "4"}):
        with pytest.raises(ValueError, match="meson_split"):
            dataclasses.replace(R90, **bad)
    with pytest.raises(ValueError, match="needs the leading-baryon split"):
        dataclasses.replace(R90, leading_split_fraction=None, leading_split_kappa=None, meson_split_fraction=0.03)
    _cat, fams_off = kernels["off"]
    _cat, fams_on = kernels["third"]
    assert all(f.meson_split_fraction == 0.3 and f.meson_split_slope == 4.0 for f in fams_on)
    assert [dataclasses.replace(f, meson_split_fraction=None, meson_split_slope=None) for f in fams_on] == fams_off


def _draws(kernel, state, n=20):
    cat, fams = kernel
    channels = C.enumerate_channels(state, cat, fams)
    out = []
    for seed in range(n):
        ch, (d0, d1) = C.draw_decay(state, channels, cat, fams, np.random.default_rng(seed),
                                    np.random.default_rng(seed + 100))
        out.append((ch.daughters, d0.momentum, d1.momentum))
    return out


def test_the_other_vertices_keep_the_committed_draws(kernels):
    """A vertex below the root, a mesonic root and an antibaryonic root
    draw bit for bit as without the option; the split roots (B = 1, the
    strange ones included) draw differently."""
    for state in (_root(5.0, pz=3.0, depth=1), _root(4.0, pz=2.0, qn=(0, 0, 0), family_id="mesonic_neutral"),
                  _root(4.0, pz=2.0, qn=(-1, -1, 0), family_id="anti_baryonic_np")):
        assert _draws(kernels["off"], state) == _draws(kernels["third"], state)
    for state in (_root(5.0, pz=3.0), _root(4.0, pz=2.0, qn=(1, 0, -1), family_id="baryonic_strange")):
        assert _draws(kernels["off"], state) != _draws(kernels["third"], state)


def test_the_meson_branch(kernels):
    """On the branch (its uniform at 0) the root decays draw over the
    meson's split channels with two channel uniforms and three kinematic
    ones; the baryonic remainder is flat in ln mu^2 on its window; the meson
    follows the incoming baryon on either side; the draw returns the split
    channels as its pool."""
    cat, fams = kernels["steep"]
    floor = C._continuation_floor(C._family_by_id(fams, "baryonic_np"), cat)
    for pz, side in ((0.0, 1.0), (-0.4, -1.0)):
        state = _root(5.0, pz=pz)
        channels = C.enumerate_channels(state, cat, fams)
        subset = [c for c in channels if _is_meson_split(c, cat, fams)]
        logs, cosines = [], []
        for seed in range(300):
            rc, rk = _Counting(seed, first=0.0), _Counting(seed + 1000)
            ch, (d0, d1), pool = C.draw_decay_with_pool(state, channels, cat, fams, rc, rk)
            assert rc.n == 2 and rk.n == 3
            assert _is_meson_split(ch, cat, fams) and pool == subset
            meson, remainder = (d0, d1) if d0.qn[0] == 0 else (d1, d0)
            edge = 5.0 - meson.mass
            assert floor - 1e-9 <= remainder.mass <= edge + 1e-9
            logs.append((math.log(remainder.mass ** 2) - math.log(floor ** 2))
                        / (math.log(edge ** 2) - math.log(floor ** 2)))
            cosines.append(math.copysign(1.0, meson.momentum[3] - pz * meson.momentum[0] / state.momentum[0]))
            assert all(abs(a + b - c) < 1e-9 for a, b, c in zip(d0.momentum, d1.momentum, state.momentum))
        assert abs(np.mean(logs) - 0.5) < 0.06
        assert side * np.mean(cosines) > 0.9


def test_the_branch_share_and_the_direction_law(kernels):
    """F = 0.3: the meson's split channels take F + (1 - F)(1 - F_B) W_M / W
    of the root decays, with F_B the leading-baryon split's share; the
    branch's direction law is kappa = 2 b q*^2 at b = meson_split_slope, or
    the leading-baryon split's law without it."""
    cat, fams = kernels["third"]
    state = _root(5.0, pz=2.0)
    channels = C.enumerate_channels(state, cat, fams)
    family = C._family_by_id(fams, "baryonic_np")
    w_meson = math.fsum(c.weight for c in channels if _is_meson_split(c, cat, fams))
    expected = 0.3 + 0.7 * (1.0 - family.leading_split_fraction) * w_meson / math.fsum(c.weight for c in channels)
    n, hits = 4000, 0
    rng_c, rng_k = np.random.default_rng(7), np.random.default_rng(8)
    for _ in range(n):
        ch, _d = C.draw_decay(state, channels, cat, fams, rng_c, rng_k)
        hits += _is_meson_split(ch, cat, fams)
    assert abs(hits / n - expected) < 4.0 * math.sqrt(expected * (1.0 - expected) / n)
    split, _subset = C._meson_split(state, channels, cat, fams)
    assert split.leading_split_fraction == 1.0 and split.leading_split_slope == 4.0 and split.leading_split_kappa is None
    cat_k, fams_k = kernels["kappa"]
    split_k, _subset = C._meson_split(state, C.enumerate_channels(state, cat_k, fams_k), cat_k, fams_k)
    assert split_k.leading_split_kappa == R90.leading_split_kappa and split_k.leading_split_slope is None
    assert C._meson_split(state, channels, *kernels["off"]) is None


def test_events_conserve_baryon_number_and_charge(kernels):
    from na49_mixing.rng_substreams import derive_substreams
    from na49_mixing.topology_runner import NOMINAL_PARAMS, run_topology
    cat, fams = kernels["third"]
    labels = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")
    params = dataclasses.replace(NOMINAL_PARAMS, exact_transfer_range=True, line_thresholds=True, nd_central=False,
                                 remnant_weights={"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.05,
                                                  "high_mass_baryonic_continuum": 0.95})
    known = {int(r.pdg_id): r for r in cat.records()}
    for i in range(60):
        subs = derive_substreams(20261011, 95, 9_950_000_000 + i)
        rngs = {f"rng_{lab}": np.random.default_rng(subs.stream_for(lab)) for lab in labels}
        finals = run_topology("nd", params, cat, fams, **rngs).final_state
        assert sum(int(known[int(h.pdg_id)].B) for h in finals if int(h.pdg_id) in known) == 2
        assert sum(int(known[int(h.pdg_id)].Q) for h in finals if int(h.pdg_id) in known) == 2


def test_the_r91_spec_is_r90_with_the_meson_split_at_the_r90_optimum():
    runs = NESTED_ROOT / "config" / "runs"
    r90, r91 = (json.loads((runs / f"{r}.json").read_text()) for r in ("r90", "r91"))
    split = {key: r91["kernel"][key] for key in ("meson_split_fraction", "meson_split_slope")}
    assert r91["kernel"] == {**r90["kernel"], **split}
    assert 0.0 < split["meson_split_fraction"] < 0.1 and split["meson_split_slope"] > 0.0
    for block in ("topology", "fit", "diffractive_split", "isovector"):
        assert r91[block] == r90[block]
    assert r91["generation_0"]["run"] == "r90_nd_isospin_mixture_2026-10-10" and r91["generation_0"]["fit"] == "free"
    assert [p["name"] for p in r91["parameters"]] == [p["name"] for p in r90["parameters"]]
    assert r91["seeds"]["master"] == r90["seeds"]["master"]
    assert r91["seeds"]["raw_first"] > r90["seeds"]["raw_first"] + 511
    assert r91["seeds"]["partition_base"] != r90["seeds"]["partition_base"]
    assert KernelConfig(**r91["kernel"]).meson_split_fraction == split["meson_split_fraction"]
