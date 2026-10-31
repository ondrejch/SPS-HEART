"""The string-like continuation step of the heavy vertices (2026-10-27):
``KernelConfig.string_step_mass`` makes a vertex above that mass emit one
established hadron at a time (the draw over its continuation channels), draw
the daughter's mass flat in ln M'^2 on its window and direct the line's
carrier along the line's hemisphere and the sibling toward the center with
kappa = 2 b q*^2 (``string_step_slope``), on the committed uniforms.  Below the mass, and with the option absent, the cascade is the
committed one bit for bit."""

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
R124 = KernelConfig(**json.loads((NESTED_ROOT / "config" / "runs" / "r124.json").read_text())["kernel"])


class _Counting:
    def __init__(self, seed):
        self.rng, self.n = np.random.default_rng(seed), 0

    def random(self):
        self.n += 1
        return self.rng.random()


def _state(M, pz=0.0, depth=1, qn=(1, 1, 0), family_id="baryonic_np"):
    e = math.sqrt(M * M + pz * pz)
    return C.CascadeState(momentum=(e, 0.0, 0.0, pz), mass=M, qn=qn, quantum={}, family_id=family_id,
                          axis=(0.0, 0.0, 1.0), alignment_label="isotropic", parent=None,
                          exchange_ancestry=("string_step",), depth=depth)


@pytest.fixture(scope="module")
def kernels():
    return {"off": load_kernel(R124, NESTED_ROOT),
            "on": load_kernel(dataclasses.replace(R124, string_step_mass=3.0), NESTED_ROOT),
            "steep": load_kernel(dataclasses.replace(R124, string_step_mass=3.0, string_step_slope=16.0), NESTED_ROOT)}


def test_the_options_validate_and_reach_every_family(kernels):
    assert R124.as_dict()["string_step_mass"] is None
    on = dataclasses.replace(R124, string_step_mass=3, string_step_slope=4, string_step_epsilon=0)
    assert (on.string_step_mass, on.string_step_slope, on.string_step_epsilon) == (3.0, 4.0, 0.0)
    assert all(isinstance(v, float) for v in (on.string_step_mass, on.string_step_slope, on.string_step_epsilon))
    for bad in ({"string_step_mass": 0.0}, {"string_step_mass": -1.0}, {"string_step_mass": True},
                {"string_step_mass": math.nan}, {"string_step_slope": 4.0}, {"string_step_epsilon": 0.5},
                {"string_step_mass": 3.0, "string_step_slope": -1.0}, {"string_step_mass": "3"}):
        with pytest.raises(ValueError, match="string_step"):
            dataclasses.replace(R124, **bad)
    with pytest.raises(ValueError, match="density continuation law"):
        dataclasses.replace(R124, continuation_mass_law="edge", leading_split_fraction=None, leading_split_kappa=None,
                            meson_split_fraction=None, meson_split_slope=None, string_step_mass=3.0)
    _cat, fams_off = kernels["off"]
    _cat, fams_on = kernels["on"]
    assert all(f.string_step_mass == 3.0 and f.string_step_slope is None for f in fams_on)
    assert [dataclasses.replace(f, string_step_mass=None) for f in fams_on] == fams_off


def _draws(kernel, state, n=40):
    cat, fams = kernel
    channels = C.enumerate_channels(state, cat, fams)
    out = []
    for seed in range(n):
        ch, daughters = C.draw_decay(state, channels, cat, fams, np.random.default_rng(seed),
                                     np.random.default_rng(seed + 100))
        out.append((ch.daughters, tuple(d.momentum for d in daughters), tuple(d.mass for d in daughters)))
    return out


def test_the_vertices_below_the_mass_keep_the_committed_draws(kernels):
    for state in (_state(2.8, pz=1.5), _state(2.9, pz=-2.0, qn=(0, 0, 0), family_id="mesonic_neutral"),
                  _state(2.5, pz=0.3, qn=(-1, -1, 0), family_id="anti_baryonic_np")):
        assert _draws(kernels["off"], state) == _draws(kernels["on"], state)
    for state in (_state(6.0, pz=3.0), _state(5.0, pz=-2.0, qn=(0, 0, 0), family_id="mesonic_neutral")):
        assert _draws(kernels["off"], state) != _draws(kernels["on"], state)


def _two_body(kernel, state, n=400, strung_only=True):
    """The two-body continuation draws of *state*: (sibling, carrier, continuum daughter, floor, edge, uniforms
    used); under the string-like step every draw is one, without it most draws are direct pairs."""
    cat, fams = kernel
    family = C._family_by_id(fams, state.family_id)
    floor = C._continuation_floor(family, cat)
    channels = C.enumerate_channels(state, cat, fams)
    rows, kinds_seen = [], set()
    for seed in range(n):
        rc, rk = _Counting(seed), _Counting(seed + 1000)
        ch, daughters = C.draw_decay(state, channels, cat, fams, rc, rk)
        if len(daughters) != 2:
            continue
        kinds_seen.add(ch.kind)
        ids = [C._daughter_identity(cat, fams, d) for d in ch.daughters]
        if sorted(i[0] for i in ids) != ["catalog", "family"]:
            continue
        cont = [i[0] for i in ids].index("family")
        d_cont, d_est = daughters[cont], daughters[1 - cont]
        if int(state.qn[0]) != 0:
            carrier, sibling = (d_cont, d_est) if int(d_cont.qn[0]) != 0 else (d_est, d_cont)
        else:
            carrier, sibling = d_cont, d_est
        assert all(abs(a + b - c) < 1e-9 for a, b, c in zip(daughters[0].momentum, daughters[1].momentum, state.momentum))
        rows.append((sibling, carrier, d_cont, floor, state.mass - d_est.mass, rk.n))
    if strung_only:
        assert kinds_seen == {C.KIND_CONTINUUM} and len(rows) == n
    else:
        assert C.KIND_ESTABLISHED_RESONANCE in kinds_seen and len(rows) < n // 2
    return rows


def test_the_heavy_step_is_flat_in_the_log_mass_and_limited_in_p_t(kernels):
    """Above the mass the continuum daughter's mass is flat in ln M'^2 on its window, the sibling's p_T^2 averages
    near 1/b (4 GeV^-2: 0.25 GeV^2; 16: 0.0625) against the isotropic step's, the carrier keeps the hemisphere, and
    the draw budget (three kinematics uniforms) is unchanged."""
    state = _state(8.0, pz=4.0)
    off = _two_body(kernels["off"], state, strung_only=False)
    on, steep = (_two_body(kernels[k], state) for k in ("on", "steep"))
    # three kinematics uniforms, four when the committed line-shape draw of a broad sibling adds its own; the step adds
    # none (the same count per seed at the two slopes, which share the pool and the mass law)
    assert {r[5] for r in off} <= {3, 4} and {r[5] for r in on} <= {3, 4}
    assert [r[5] for r in on] == [r[5] for r in steep]
    frac = [(math.log(r[2].mass ** 2) - math.log(r[3] ** 2)) / (math.log(r[4] ** 2) - math.log(r[3] ** 2)) for r in on]
    assert all(-1e-9 <= f <= 1.0 + 1e-9 for f in frac)
    assert abs(float(np.mean(frac)) - 0.5) < 0.06 and 0.2 < float(np.mean(np.asarray(frac) < 0.5)) < 0.8
    pt2 = lambda rows: float(np.mean([r[0].momentum[1] ** 2 + r[0].momentum[2] ** 2 for r in rows]))  # noqa: E731
    assert pt2(on) < 0.6 * pt2(off)
    assert pt2(steep) < pt2(on)
    assert abs(pt2(on) - 0.25) < 0.12 and abs(pt2(steep) - 0.0625) < 0.04
    assert float(np.mean([r[1].momentum[3] - r[0].momentum[3] for r in on])) > 0.0
    assert float(np.mean([1.0 if r[1].momentum[3] > r[0].momentum[3] else 0.0 for r in on])) > 0.8


def test_the_mesonic_line_and_the_other_hemisphere(kernels):
    """A mesonic line on the negative side sends its continuum daughter on along -z and the sibling toward the
    center; an antibaryonic line keeps its antibaryon carrier forward."""
    for state in (_state(6.0, pz=-3.0, qn=(0, 0, 0), family_id="mesonic_neutral"),
                  _state(6.0, pz=-3.0, qn=(-1, -1, 0), family_id="anti_baryonic_np")):
        rows = _two_body(kernels["on"], state)
        assert float(np.mean([1.0 if r[1].momentum[3] < r[0].momentum[3] else 0.0 for r in rows])) > 0.8
    rows = _two_body(kernels["on"], _state(6.0, pz=3.0, qn=(0, 0, 0), family_id="mesonic_neutral"))
    assert float(np.mean([1.0 if r[1].momentum[3] > r[0].momentum[3] else 0.0 for r in rows])) > 0.8
