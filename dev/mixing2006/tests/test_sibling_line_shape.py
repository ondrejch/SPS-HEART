"""The line-shape masses of the established daughters of a continuum vertex
(2026-10-10, review grok-4.7 M2): ``KernelConfig.sibling_line_width`` lets
an established daughter at least that wide draw its mass from its line
shape on the on-shell window, so the f0(500) sibling of a continuation step
decays at its spectral masses instead of its pole; off, and on with no
daughter that wide, is the committed rule bit for bit."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing import cascade_decay as C
from na49_mixing.kernel_config import KernelConfig, load_kernel
from na49_mixing.resonance_decay import ResonanceDecayer
from na49_mixing.resonance_widths import BLATT_WEISSKOPF_BARRIER, spectral_function

NESTED_ROOT = Path(__file__).resolve().parents[1]
R89 = KernelConfig(**json.loads((NESTED_ROOT / "config" / "runs" / "r89.json").read_text())["kernel"])
F0 = 9000221


@pytest.fixture(scope="module")
def kernels():
    return {name: load_kernel(dataclasses.replace(R89, sibling_line_width=width), NESTED_ROOT)
            for name, width in (("off", None), ("on", 0.1), ("none_wide", 10.0))}


class _Counting:
    def __init__(self, seed):
        self.rng, self.n = np.random.default_rng(seed), 0

    def random(self):
        self.n += 1
        return self.rng.random()


def _state(fams, family_id, mass):
    family = C._family_by_id(fams, family_id)
    return C.CascadeState(momentum=(mass, 0.0, 0.0, 0.0), mass=mass, qn=tuple(family.qn), quantum={},
                          family_id=family_id, axis=(0.0, 0.0, 1.0), alignment_label=C.ISOTROPIC_ALIGNMENT_LABEL,
                          parent=None, exchange_ancestry=("sibling_line_shape",), depth=1)


def _f0_continuation(cat, fams, mass):
    state = _state(fams, "baryonic_np", mass)
    channels = [c for c in C.enumerate_channels(state, cat, fams)
                if c.kind == C.KIND_CONTINUUM and f"catalog:{F0}" in c.daughters]
    return state, channels


def test_the_option_validates_records_and_reaches_every_family(kernels):
    assert R89.as_dict()["sibling_line_width"] is None
    assert dataclasses.replace(R89, sibling_line_width=1).sibling_line_width == 1.0
    for bad in (0.0, -0.1, math.inf, True, "0.1"):
        with pytest.raises(ValueError, match="sibling_line_width must be a positive finite width"):
            dataclasses.replace(R89, sibling_line_width=bad)
    _cat, fams_on = kernels["on"]
    _cat, fams_off = kernels["off"]
    assert all(f.sibling_line_width == 0.1 for f in fams_on)
    assert [dataclasses.replace(f, sibling_line_width=None) for f in fams_on] == fams_off


def test_no_daughter_that_wide_is_the_committed_rule(kernels):
    """At a width no record reaches, every draw (and its uniform count) is
    the committed draw."""
    for mass in (2.6, 4.0):
        out = {}
        for name in ("off", "none_wide"):
            cat, fams = kernels[name]
            state = _state(fams, "baryonic_np", mass)
            channels = [c for c in C.enumerate_channels(state, cat, fams) if c.kind == C.KIND_CONTINUUM]
            draws = []
            for seed in range(30):
                rk = _Counting(seed + 50)
                ch, daughters = C.draw_decay(state, channels, cat, fams, np.random.default_rng(seed), rk)
                draws.append((ch.daughters, [d.momentum for d in daughters], rk.n))
            out[name] = draws
        assert out["off"] == out["none_wide"]


@pytest.mark.parametrize("mass", [2.6, 3.0, 5.0])
def test_the_f0_sibling_follows_its_line_shape_on_the_window(kernels, mass):
    cat, fams = kernels["on"]
    state, channels = _f0_continuation(cat, fams, mass)
    family = C._family_by_id(fams, "baryonic_np")
    record = cat.get(F0)
    floor = C._continuation_floor(family, cat)
    lo, hi = float(record.mass_range[0]), min(float(record.mass_range[1]), mass - floor)
    masses, remainders = [], []
    for seed in range(1500):
        rk = _Counting(seed + 7000)
        _ch, (d0, d1) = C.draw_decay(state, channels[:1], cat, fams, np.random.default_rng(seed), rk)
        sibling, remainder = (d0, d1) if d0.family_id == f"catalog:{F0}" else (d1, d0)
        # one uniform for the sibling's mass, one for the density step, two angles
        assert rk.n == 4
        masses.append(sibling.mass)
        remainders.append(remainder.mass)
        assert lo - 1e-12 <= sibling.mass <= hi + 1e-12
        assert remainder.mass >= floor - 1e-9 and remainder.mass <= mass - sibling.mass + 1e-9
    # the mean of the line shape truncated to the window
    decayer = ResonanceDecayer(cat, 0, 0, 0)
    xs = np.linspace(lo, hi, 4001)
    ys = np.array([spectral_function(float(x), record, decayer._daughter_mass_map(record), BLATT_WEISSKOPF_BARRIER, cat)
                   for x in xs])
    want = float(np.sum(xs * ys) / np.sum(ys))
    spread = float(math.sqrt(np.sum((xs - want) ** 2 * ys) / np.sum(ys)))
    assert abs(np.mean(masses) - want) < 4.0 * spread / math.sqrt(len(masses))
    assert np.std(masses) > 0.05


def test_a_pair_of_broad_daughters_stays_on_shell(kernels):
    cat, fams = kernels["on"]
    for family_id, mass in (("mesonic_neutral", 1.9), ("mesonic_charged", 1.9), ("baryonic_np", 1.95)):
        state = _state(fams, family_id, mass)
        direct = [c for c in C.enumerate_channels(state, cat, fams) if c.kind != C.KIND_CONTINUUM]
        broad_pairs = [c for c in direct
                       if all(C._is_line_shape_daughter(C._daughter_identity(cat, fams, d), 0.1) for d in c.daughters)]
        if not broad_pairs:
            continue
        for seed in range(300):
            _ch, (d0, d1) = C.draw_decay(state, broad_pairs, cat, fams, np.random.default_rng(seed),
                                         np.random.default_rng(seed + 1))
            assert d0.mass + d1.mass <= mass + 1e-12
            for d in (d0, d1):
                record = cat.get(int(d.family_id[len(C.CATALOG_FAMILY_PREFIX):]))
                assert float(record.mass_range[0]) - 1e-12 <= d.mass <= float(record.mass_range[1]) + 1e-12
        return
    pytest.fail("no direct pair of two broad daughters found")


def test_the_truncated_inverse_cdf():
    cat, _fams = load_kernel(R89, NESTED_ROOT)
    record = cat.get(F0)
    lo = float(record.mass_range[0])
    for hi in (0.45, 0.8, 2.0):
        top = min(hi, float(record.mass_range[1]))
        draws = [C._sibling_line_mass(record, hi, cat, u) for u in np.linspace(0.0, 1.0, 101)]
        assert draws[0] == pytest.approx(lo, abs=1e-12) and draws[-1] == pytest.approx(top, abs=1e-12)
        assert all(b >= a for a, b in zip(draws, draws[1:]))
    with pytest.raises(ValueError, match="window .* is empty"):
        C._sibling_line_mass(record, lo - 0.01, cat, 0.5)


def test_events_conserve_baryon_number_and_charge(kernels):
    from na49_mixing.rng_substreams import derive_substreams
    from na49_mixing.topology_runner import NOMINAL_PARAMS, run_topology
    cat, fams = kernels["on"]
    labels = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")
    params = dataclasses.replace(NOMINAL_PARAMS, exact_transfer_range=True, line_thresholds=True, nd_central=False)
    known = {int(r.pdg_id): r for r in cat.records()}
    for i in range(60):
        subs = derive_substreams(20261010, 91, 9_910_000_000 + i)
        rngs = {f"rng_{lab}": np.random.default_rng(subs.stream_for(lab)) for lab in labels}
        finals = run_topology("nd", params, cat, fams, **rngs).final_state
        assert sum(int(known[int(h.pdg_id)].B) for h in finals if int(h.pdg_id) in known) == 2
        assert sum(int(known[int(h.pdg_id)].Q) for h in finals if int(h.pdg_id) in known) == 2
